import numpy as np

# QAgent implements a tabular Q-learning agent with a simple clock based on the time elapsed from the last observation

class QAgent:
    
    def __init__(self, 
                 batch_size : int = 1, 
                 horizon : int = 500,
                 eps_greedy = lambda t : 0.3,
                 gamma : float = 0.998,
                 learning_rate = lambda t : 0.25,
                 s_thr : float = 3e-6,
                 seed : int = 12131415,
                 out_dir : str = "./"
                ):
        self.horizon = horizon
        self.batch_size = batch_size
        # Threshold for classifying observations
        self.s_thr = s_thr
        # Discount factor
        self.gamma = gamma
        # Initialize Q-table with a single state (row) and 4 actions (columns), values = -1
        self.Q = np.full((self.horizon+1, 4), -1.0, dtype=np.float32)
        # Output directory for saving Q-values
        self.out_dir = out_dir
        self.k = 0
        # Episode counter to drive schedules
        self.num_episodes = 0
        # RNG for ε-greedy and random actions
        self.rnd_state = np.random.RandomState(seed)
        # Learning rate and epsilon for greedy policy
        self.learning_rate = learning_rate
        self.eps_greedy = eps_greedy
        # Current discrete state index
        self.current_state = np.zeros(self.batch_size, dtype=np.int32)
        # Initialize to zero the population table for each state-action pair
        self.n = np.zeros((self.horizon+1, 4), dtype=np.float32)
               
    # Save the current Q-table to disk under out_dir/fname.npy   
    def store_q(self, fname):
        np.save("{}/{}".format(self.out_dir, fname), self.Q, allow_pickle=True)

    def store_n(self, fname):
        np.save("{}/{}".format(self.out_dir, fname), self.n, allow_pickle=True)
        
    # Load a previously saved Q-table from disk
    def load_q(self, qpath):
        self.Q = np.load(qpath, allow_pickle=True)
        
    def current_state(self):
        return self.current_state

    # Map a raw observation into a discrete state index
    def update_state(self, raw_obs):

        maskTerm = raw_obs < 0
        maskDet = raw_obs >= self.s_thr
        maskVoid = np.argwhere((raw_obs < self.s_thr) & (raw_obs >= 0))[:,0]
        
        self.current_state[maskTerm] = -1
        self.current_state[maskDet] = 0
        self.current_state[maskVoid] += 1
                
        return self.current_state, maskDet
    
    #Choose an action index based on ε-greedy policy:
    #      - With probability ε (if not deterministic), pick a random action
    #      - Otherwise, pick the action with highest Q-value at current_state
    def select_action(self, deterministic = False):
        rans = self.rnd_state.rand(self.batch_size)
        num_actions = self.Q.shape[1]
        state = self.current_state
        action = np.zeros(self.batch_size, dtype=np.int32)
        
        if not deterministic:
            below_eps = np.argwhere(rans < self.eps_greedy(self.num_episodes))[:,0]
            if len(below_eps) != 0:
                num_below = below_eps.shape
                action[below_eps] = self.rnd_state.choice(num_actions, num_below)

            above_eps = np.argwhere(rans >= self.eps_greedy(self.num_episodes))[:,0]
            if len(above_eps) != 0:
                action[above_eps] = np.argmax(self.Q[state[above_eps]],axis=1)          
        else:
            action = np.argmax(self.Q[state],axis=1)   
          
        return action
    
    def select_action_new(self, deterministic = False):
        rans = self.rnd_state.rand(self.batch_size)
        num_actions = self.Q.shape[1]
        state = self.current_state
        action = np.zeros(self.batch_size, dtype=np.int32)
        
        #Epsilon scheduling depending on number of times a state is visited 
        epsilon = np.zeros(self.batch_size)
        states_count = self.n.sum(axis=1)/num_actions
        p1=100
        p2=100
        p3=4/5
        epsilon = p1/(p2**(1/p3)+states_count[state])**p3
        
        if not deterministic:
            below_eps = np.argwhere(rans < epsilon)[:,0]
            if len(below_eps) != 0:
                num_below = below_eps.shape
                action[below_eps] = self.rnd_state.choice(num_actions, num_below)

            above_eps = np.argwhere(rans >= epsilon)[:,0]
            if len(above_eps) != 0:
                action[above_eps] = np.argmax(self.Q[state[above_eps]],axis=1)          
        else:
            action = np.argmax(self.Q[state],axis=1)   
          
        return action

    # Perform the Q-learning update for state s, action a:
    #      Q[s,a] ← (1 - α) Q[s,a] + α [ r + γ max_a' Q[s',a'] ]
    #    If s_prime == -1 (terminal), we use 0 for future value
    def update_q(self, s, a, s_prime, r):
        alpha = self.learning_rate(self.num_episodes)
        maxQ = np.zeros(s.shape[0])
        active1 = np.argwhere(s_prime>=0)[:,0]
        maxQ[active1] = np.max(self.Q[s_prime[active1]],axis=1)
        active2 = np.argwhere(s>=0)[:,0]

        #Count how many times a given (s,a)-pair is visited
        self.n[s[active2],a[active2]] += 1
        
        self.Q[s[active2], a[active2]] = (1 - alpha) * self.Q[s[active2], a[active2]] + alpha * (r[active2] + self.gamma * maxQ[active2])
        
    def update_q_new(self, s, a, s_prime, r):
        maxQ = np.zeros(s.shape[0])
        active1 = np.argwhere(s_prime>=0)[:,0]
        maxQ[active1] = np.max(self.Q[s_prime[active1]],axis=1)
        active2 = np.argwhere(s>=0)[:,0]

        #Count how many times a given (s,a)-pair is visited
        self.n[s[active2],a[active2]] += 1
        
        #Learning rate scheduling depending on number of times a state-action pair is visited 
        alpha = np.zeros(self.batch_size)
        #p1=10
        #p2=100
        #p3=4/5
        alpha[active2] = 1./(1.+self.n[s[active2],a[active2]])
        #alpha[active2] = p1/(p2**(1/p3)+self.n[s[active2],a[active2]])**p3
        
        self.Q[s[active2], a[active2]] = (1 - alpha[active2]) * self.Q[s[active2], a[active2]] + alpha[active2] * (r[active2] + self.gamma * maxQ[active2])
        
    # Reset agent's internal state for a new episode
    def reset(self):
        self.current_state = np.zeros(self.batch_size, dtype=np.int32)
        
        
