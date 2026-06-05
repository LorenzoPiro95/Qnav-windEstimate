import numpy as np
from tqdm import tqdm

from qnav.agents import QAgent
from qnav.environment import Environment

# Simulator ties together the QAgent and Environment to run training and testing.

class Simulator:
    
    """
        Initialize with:
          env: an instance of Environment
          actions: movement vectors
    """
    def __init__(self, env : Environment, actions, max_lag: int = 500) -> None:
        self.env = env
        self.actions = actions
        self.max_lag = max_lag
        self.batch_size = env.batch_size
        
        # MSD accumulators: ⟨(x(t)-x(0))²⟩ as a function of t since last encounter
        self._msd = np.zeros(self.max_lag, dtype=float)
        self._msd_x = np.zeros(self.max_lag, dtype=float)
        self._msd_y = np.zeros(self.max_lag, dtype=float)
        self._msd_counts = np.zeros(self.max_lag, dtype=int)

        # Per-agent state for MSD
        # Will be initialised properly at the start of each episode
        self._last_hit_pos = None        # shape (batch_size, dim)
        self._time_since_last_hit = None # shape (batch_size,)
        self._occ0_counts = 0
        
    ################################################################################   
    #                         MSD calculation routines                             #
    ################################################################################   
    def _reset_msd_state(self, init_pos):
        """
        init_pos: array (batch_size, dim) positions at t=0 of the episode.
        """
        self._last_hit_pos = init_pos.copy()
        self._time_since_last_hit = np.zeros(self.batch_size, dtype=int)
        
    def _update_msd(self, pos: np.ndarray, encounter: np.ndarray, active_idx: np.ndarray):
        """
        pos:       (batch_size, dim) current positions x(t)
        encounter: (batch_size,) bool array, True if an odor particle is found at this step
        """
        
        idx = active_idx  # just a shorter alias

        if idx.size == 0:
            return
        
        # Reset origin x(0) where a new encounter happens
        hit_mask = encounter[idx]
        if np.any(hit_mask):
            hit_idx = idx[hit_mask]       # absolute indices in [0, batch_size)
            self._last_hit_pos[hit_idx] = pos[hit_idx]
            self._time_since_last_hit[hit_idx] = 0
            self._occ0_counts += hit_idx.size

        # For agents with no hit, advance the clock and accumulate (x(t)-x(0))^2
        move_idx = idx[~hit_mask]
        if move_idx.size == 0:
            return

        self._time_since_last_hit[move_idx] += 1
        lags = self._time_since_last_hit[move_idx]

        # Only use lags in [1, max_lag-1]
        valid = (lags > 0) & (lags < self.max_lag)

        contributing_idx = move_idx[valid]             # absolute indices of contributing agents
        contributing_lags = self._time_since_last_hit[contributing_idx]

        # Squared displacement from last-hit position x(0)
        disp2 = np.sum((pos[contributing_idx] - self._last_hit_pos[contributing_idx]) ** 2, axis=1)
        dx = pos[contributing_idx, 0] - self._last_hit_pos[contributing_idx, 0]
        dy = pos[contributing_idx, 1] - self._last_hit_pos[contributing_idx, 1]

        # Accumulate into global MSD bins
        np.add.at(self._msd, contributing_lags, disp2)
        np.add.at(self._msd_x, contributing_lags, dx * dx)
        np.add.at(self._msd_y, contributing_lags, dy * dy)
        np.add.at(self._msd_counts, contributing_lags, 1)

    def get_msd_since_last_encounter(self):
        """
        Returns:
            tau  : array of lag times (dt, 2*dt, ..., (max_lag-1)*dt)
            msd  : array of ⟨(x(t)-x(0))²⟩ for each lag
            counts: number of samples contributing to each lag
        """
        tau = np.arange(self.max_lag) * 1.0
        msd = np.zeros_like(self._msd, dtype=float)
        msd_x = np.zeros_like(self._msd_x, dtype=float)
        msd_y = np.zeros_like(self._msd_y, dtype=float)
        valid = self._msd_counts > 0
        msd[valid] = self._msd[valid] / self._msd_counts[valid]
        msd_x[valid] = self._msd_x[valid] / self._msd_counts[valid]
        msd_y[valid] = self._msd_y[valid] / self._msd_counts[valid]
        counts = self._msd_counts.astype(float)
        counts[0] = self._occ0_counts
        return tau, msd, msd_x, msd_y, counts
    
    ################################################################################    
    
    #Rotated such that left=upwind, right=downwind
    def rotate_action_upwind(self, action : np.ndarray, angle : np.ndarray):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        x_rot = cos_theta * action[:, 1] - sin_theta * action[:, 0]
        y_rot = sin_theta * action[:, 1] + cos_theta * action[:, 0]

        return np.stack((y_rot,x_rot), axis=1)
    
    def _perform_single_step(self, agent : QAgent, pos : np.ndarray, T_min : np.ndarray, time_idx : int, deterministic : bool = False):
        """
            Execute one time-step:
              1. Agent selects an action (optionally deterministic)
              2. Agent moves and makes an observation.
              3. Updates state

            Returns:
              new pos: np.array, agent's new position
              action a: int, index of chosen action
              reward r: float, reward received
              terminated: bool, whether goal reached or terminal state
              s_prime: int, agent's internal state after observing
              new time_idx: int, updated time index
        """   
        #a = agent.select_action_new(deterministic=deterministic)
        a = agent.select_action(deterministic=deterministic)
        
        pos_old_int = np.rint(pos).astype(int)
        
        flow_vec = self.env.get_flow_vec(pos_old_int, time_idx)
        self.env.store_flow(flow_vec)
        angle = self.env.estimate_mean_flow()
        
        act_rot = self.rotate_action_upwind(self.actions[a],angle)
        
        pos, r, terminated = self.env.make_move(pos, agent.current_state, act_rot, T_min)
        
        # Advance time
        time_idx += 1 #(time_idx + 1) % self.env.MAX_T
        
        # Get observation for the new state; terminal obs is -1
        pos_int = np.rint(pos).astype(int)
        obs = self.env.get_observation(pos_int, time_idx, terminated)
        s_prime, detected = agent.update_state(obs)
        
        return pos, a, r, terminated, s_prime, time_idx, detected, angle
        
    def train_agent(self, 
                    agent : QAgent, # agent to train
                    point_sampler, # probability of observation at every location 
                    horizon, # number of steps per episode (horizon)
                    num_episodes, # number of episodes
                    retrain,
                    init_file,
                    delta : int = 500, # size of the window used to compute the average cumulative reward per episode
                    seed_sim : int = 12131415, # random seed for the simulation
                    checkpoint_iters : int = 1000 # save Q-function every this many episodes
                    ):
        
        # Prepare storage
        cumulative_rewards, agent_speed, convergence =  [], [], []

        # RNG for sampling time initial indices
        rnd_state = np.random.RandomState(seed=seed_sim)

        #Criterion to pick best Q-policy
        maxConv = 0.
        
        iterator = tqdm(range(num_episodes))
        for k in iterator:
            
            # Initialization: Reset agent internal Q, state, etc.
            agent.reset()
            
            # Sample a starting time and position that is valid
            pos_int, time_idx = point_sampler()

            init_pos = pos_int.copy()
            pos = pos_int.copy().astype(float)
            batch_size = self.batch_size
            T_min = self.env.get_distance_to_source(init_pos)
            #print(f"Time={time_idx}, y0={init_pos[0]}, x0={init_pos[1]}")
            
            terminated = np.zeros(batch_size, dtype=bool)
            arr_times = np.zeros(batch_size, dtype=np.int32)
            
            # Initial observation, state and memory
            obs = self.env.get_observation(pos_int, time_idx, terminated)
            s, detected = agent.update_state(obs) 
            self.env.init_flow_memory(pos_int,time_idx)
            
            c_reward = np.zeros(batch_size)
            
            # Run episode up to horizon steps
            for t in range(horizon):
                
                arr_times[~terminated] += 1
                
                #if((t+1)%100 == 0):
                 #   print(terminated)
                
                
                # Single Q-learning step
                pos, a, r, terminated, s_prime, time_idx, obs, flow_angle = self._perform_single_step(agent,
                                                                                  pos,
                                                                                  T_min,
                                                                                  time_idx, 
                                                                                  deterministic = False)
                
                upToJustTerminated = np.argwhere(s>=0)[:,0]
                
                # Accumulate discounted reward
                c_reward[upToJustTerminated] += r[upToJustTerminated] * (agent.gamma**t) 
                    
                #if((t+1)%100 == 0):
                #    print(angle.mean())
                    
                # Q-value update: state, action -> next_state, reward
                #agent.update_q_new(s, a, s_prime, r)
                agent.update_q(s, a, s_prime, r)
                
                # Episode ends if reached goal
                if np.all(terminated):
                    break
                
                #s = s_prime
                s = s_prime.copy()
                
            # Record training metrics    
            convergence.append(np.sum(terminated.astype(int))/batch_size)

            # Compute speed if terminated: initial distance over steps (T_min/T)
            if np.sum(terminated.astype(int)) > 0:
                episode_speed = ((T_min/arr_times)[terminated]).mean()*np.sum(terminated.astype(int))/batch_size
            else:
                episode_speed = 0.
            agent_speed.append(episode_speed)
            
            ###################################################################
            # Periodically save and test intermediate Q-functions
            if ((k + 1) % checkpoint_iters == 0 and (num_episodes-k)<1000):
                
                if(retrain):
                    agent.store_q(f"Q_stored/Qfun_{k + 1 + 3000}")
                else:
                    agent.store_q(f"Q_stored/Qfun_{k + 1}")
                    
                results = []
                results, useless = self.test_agent(agent, 
                                         point_sampler=point_sampler, 
                                         num_episodes=10,
                                         init_file=init_file,
                                         training=1,
                                         horizon=horizon
                                         )
                
                results_array = np.vstack(results)
                creward_test, convergence_test = [],[]
                        
                for result in results_array:
                    creward_test.append(result[2])
                    convergence_test.append(result[5])
                    
                print("[--] G = {} +/- {}".format(np.mean(creward_test), np.std(creward_test)))
                print("[--] convergence = {} +/- {}".format(np.mean(convergence_test), np.std(convergence_test)))
                if(np.mean(convergence_test)>maxConv):
                    bestQ = k+1
                    maxConv=np.mean(convergence_test)
                    agent.store_q("Q_stored/Qfun_best") 
            ###################################################################
            
            cumulative_rewards.append(c_reward.mean())
            
            iterator.set_postfix({'avg G_t' : np.mean(cumulative_rewards[-delta:]), 
                                  'avg T_min/T' : np.mean(agent_speed[-delta:]),
                                  'avg C_t' : np.mean(convergence[-delta:]),
                                  'eps' : agent.eps_greedy(agent.num_episodes),
                                  'alpha' : agent.learning_rate(agent.num_episodes)})
            
            agent.num_episodes += 1
            
        return cumulative_rewards, agent_speed, convergence, bestQ, maxConv

    def test_agent(self, 
                   agent : QAgent, 
                   point_sampler,
                   num_episodes,
                   init_file,
                   training : bool,
                   horizon : int
                   ):
        
        results = []
        traj = []
        iep=0
        
        batch_size = self.batch_size
        
        #Using pre-defined initial conditions
        if not training:
            init_data = np.fromfile(init_file, dtype=np.int32).reshape(-1, 3)
            num_episodes=int(len(init_data.flatten())/3/batch_size)
            
        iterator = tqdm(range(num_episodes))
        
        for k in iterator:
            
            if(training):
                # Sample a starting time and position that is valid
                pos_int, time_idx = point_sampler()
            else:
                #Using pre-defined initial conditions
                #####################################
                time_idx = init_data[iep*batch_size, 0].astype(int)
                pos_int = init_data[iep*batch_size:(iep+1)*batch_size, 1:3]  # np.array([y, x], dtype=int)
                #####################################
            
            init_pos = pos_int.copy()
            pos = pos_int.copy().astype(float)
            
            self._reset_msd_state(pos)
            agent.reset()
            
            terminated = np.zeros(batch_size, dtype=bool)
            timeToReach = np.zeros(batch_size, dtype=np.int32)
            T_min = self.env.get_distance_to_source(init_pos) 
            
            obs = self.env.get_observation(pos_int, time_idx, terminated)
            s, detected = agent.update_state(obs)
            self.env.init_flow_memory(pos_int,time_idx)
            flow_angle = self.env.estimate_mean_flow()

            c_reward = np.zeros(batch_size)
            
            # Track maximum state per agent during the episode
            s_max = s.copy()
            
            # Perform up to 'horizon' steps or until goal reached
            for t in range(horizon):
                
                timeToReach[~terminated] += 1
                
                traj.append([iep, t, pos[0,1], pos[0,0], obs[0], flow_angle[0], time_idx])
                
                # Execute a deterministic (greedy) action
                pos, a, r, terminated, s_prime, time_idx, obs, flow_angle = self._perform_single_step(agent, 
                                                                              pos, 
                                                                              T_min,
                                                                              time_idx, 
                                                                              deterministic = True)
                upToJustTerminated = np.argwhere(s>=0)[:,0]
                
                self._update_msd(pos, obs, upToJustTerminated)
                
                # Accumulate discounted reward
                c_reward[upToJustTerminated] += r[upToJustTerminated] * (agent.gamma**t) 
                
                # Update maximum visited state
                s_max = np.maximum(s_max, s_prime)
                
                # Compute speed if goal reached
                if np.all(terminated):
                    break
                
                s = s_prime.copy()
    
            timeToReach[~terminated] = 0
            
            episode_results = np.column_stack([
                 init_pos[:, 1],           # x position
                 init_pos[:, 0],           # y position
                 c_reward,                 # cumulative reward
                 timeToReach,             # time to reach goal (or 0 if not reached)
                 T_min,                    # optimal time
                 terminated.astype(int),   # convert boolean to int (1 if done, else 0)
                 s_max                     # maximum state visited in episode
            ])

            # Append the entire batch to your list
            results.append(episode_results)
            iep+=1;
            
        return results, np.array(traj)
