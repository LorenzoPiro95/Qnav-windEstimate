import numpy as np

from typing import Tuple, Dict

"""
    Environment provides the state dynamics and observations for the agent.
    It holds the spatiotemporal data grid and defines the source (goal) region.
"""

class Environment:
    
    def __init__(self, 
                 data : np.ndarray, 
                 data_vel : np.ndarray,
                 source_position : np.ndarray,
                 gamma : float,
                 batch_size : int,
                 mem_length : int,
                 source_radius : float = 2,
                 shaping_factor : float = 0,
                 seed : int = 1212121314
                 ):
        # DNS conc data
        self.data = data
        # DNS velocity data
        self.data_vel = data_vel
        # Discount factor to shape reward magnitude for non-terminal steps
        self.gamma = gamma
        # Location of the goal in the grid
        self.source_position = source_position
        # Radius around the goal that counts as reaching the source
        self.source_radius = source_radius
        # RNG for any sampling (not used directly here but available)
        self.rnd_state = np.random.RandomState(seed)
        # Maximum number of time steps corresponds to duration of DNS data
        self.MAX_T = self.data.shape[0] 
        # Domain size
        self.NY = self.data.shape[1] 
        self.NX = self.data.shape[2]
        # Memory length
        self.mem_length = mem_length
        # Batch size
        self.batch_size = batch_size
        # Shaping factor for the reward
        self.shaping_factor = shaping_factor
        #Exponential moving average quantities
        self.mean_flow = np.zeros((batch_size, 2), dtype=float)
        self.alpha = 1.0 - np.exp(-1.0 / float(mem_length))
        
        #############################################
        #Fixed memory stuff
        # Buffer to hold the last M measurements of shape (N, M, 2)
        #self.flow_memory = np.zeros((batch_size, mem_length, 2), dtype=float)
        # Next write index
        #self.ptrs = np.zeros(batch_size, dtype=int)
        #############################################
          
    # Determine if a given position is inside the goal region
    def source_reached(self, pos):
        return np.abs(pos - self.source_position).sum(-1) <= self.source_radius

    # Compute the Manhattan distance from pos to the goal, minus the source radius
    def get_distance_to_source(self, pos):
        manhattan_distance = np.abs(pos - self.source_position).sum(-1)
        distance_to_border = manhattan_distance - self.source_radius
        return np.ceil(distance_to_border)

    # Return the data observation at position pos and time t.
    # If out-of-bounds, returns a zero observation.
    def get_observation(self, pos, t, terminated): 
        mask = ( (pos[:,0] >=0) & (pos[:,0] < self.data.shape[1]) & (pos[:,1] >=0) & (pos[:,1] < self.data.shape[2]) )
        obs = np.zeros(pos.shape[0])
        obs[mask] = self.data[t, pos[mask, 0], pos[mask, 1]]
        obs[terminated] = -1.0
        return obs

    # Apply an action to move the agent and compute reward and termination                
    def make_move(self, pos, s, action : np.ndarray, T_min):        
        pos_old = pos.copy()
        active = np.argwhere(s>=0)[:,0]
        pos[active] = pos[active] + action[active]
        terminated = self.source_reached(pos)
        
        dist_old = self.get_distance_to_source(pos_old)
        dist_new = self.get_distance_to_source(pos)
        
        # Reward scheme: positive for success, small negative for step cost
        reward = np.ones(pos.shape[0])*(-1 + self.gamma + self.shaping_factor*(dist_old - self.gamma*dist_new)/T_min)
        reward[terminated] = 1.0
        return pos, reward, terminated
    
    # Initialize memory of the agent
    def init_flow_memory(self, pos, t):
        
        ####################################
        #Fixed memory implementation
        #for tp in range(self.mem_length):
        #    flow_vec = self.get_flow_vec(pos,t-tp-1)
        #    self.flow_memory[np.arange(self.batch_size),self.mem_length-tp-1,:] = flow_vec
        #####################################
        
        #Exponential moving average implementation
        # Reconstruct EMA from past mem_length samples (oldest -> newest)
        start = t - int(self.mem_length)
        mf = self.get_flow_vec(pos, start).copy()

        # iterate over the remaining samples in chronological order
        for tt in range(start + 1, t+1):  # up to t
            fv = self.get_flow_vec(pos, tt)
            mf = (1.0 - self.alpha) * mf + self.alpha * fv
        
        self.mean_flow = mf
        
        
    # Return instantaneous flow vector
    def get_flow_vec(self, pos, t):
        '''flow_x = np.zeros(self.batch_size,dtype=np.float32)
        flow_y = np.zeros(self.batch_size,dtype=np.float32)
        mask = ( (pos[:,0] >=0) & (pos[:,0] < self.data.shape[1]) & (pos[:,1] >=0) & (pos[:,1] < self.data.shape[2]) )
        
        flow_x[mask] = self.data_vel[t,0,pos[mask, 0],pos[mask, 1]]
        flow_y[mask] = self.data_vel[t,1,pos[mask, 0],pos[mask, 1]]'''
        
        
        tt = t%self.MAX_T
        
        flow_x = self.data_vel[tt,0,(pos[:, 0]+10*self.NY)%self.NY,(pos[:, 1]+10*self.NX)%self.NX]
        flow_y = self.data_vel[tt,1,(pos[:, 0]+10*self.NY)%self.NY,(pos[:, 1]+10*self.NX)%self.NX]
        
        flow_vec = np.stack([flow_x, flow_y], axis=1)  # shape (batch_size,2)
        return flow_vec
    
    def store_flow(self, flow_vec):
        """
        Store the new wind measurements in the circular buffer.
        flow_vec : array of shape (batch_size, 2)
        """
        ####################################################
        # Write into buffer
        #i = self.ptrs
        #self.flow_memory[np.arange(self.batch_size), i, :] = flow_vec
        # Advance pointers
        #self.ptrs = (i + 1) % self.mem_length
        ####################################################
        """
        Update exponential memory estimate in O(1).
        flow_vec : array shape (batch_size, 2)
        """
        # EMA update: new_mean = (1-alpha) * old_mean + alpha * new_measurement
        # Works elementwise and is vectorized for the batch.
        self.mean_flow = (1.0 - self.alpha) * self.mean_flow + self.alpha * flow_vec
    
    # Compute an estimate of the mean wind direction based on the current memory   
    def estimate_mean_flow(self):
        """
        Returns: numpy array of shape (batch_size,) with values in [-pi, pi]
        """
        ##########################################
        #Finite memory implementation
        #mean_flow = self.flow_memory.mean(axis=1)
        #mean_angle = np.arctan2(mean_flow[:, 1], mean_flow[:, 0])
        ##########################################
        
        #Exponential moving average implementation
        mean_angle = np.arctan2(self.mean_flow[:, 1], self.mean_flow[:, 0])
        
        return mean_angle
