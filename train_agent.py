import os
import argparse as ap
import numpy as np 

# Import core Q-learning agent and environment
from qnav import QAgent, Environment
from qnav.simulators import Simulator

# Define available movement actions as unit vectors
actions = np.array([
    [1., 0.],   # action up
    [-1., 0.],  # action down
    [0., 1.],   # action right
    [0., -1.]   # action left
], dtype=np.float32)

# Seeds for reproducibility of agent, environment, sampling, and simulation --- DO NOT CHANGE THIS
seeds = {
    'agent_seed' : 314,
    'env_seed' : 154, # NOT USEFUL RIGHT NOW
    'samp_seed' : 980,
    'sim_seed' : 723 # NOT USEFUL RIGHT NOW
}

# Epsilon-greedy schedule parameters for exploration
eps_init, eps_end, eps_decay = 1., 0.0001, 10001 
# Learning rate schedule parameters for Q-value updates
alpha_init, alpha_end, alpha_decay = 0.1, 0.0001, 10001

# Functions defining time-dependent epsilon and learning rate
#Piecewise-linear decay
#eps_greedy = lambda t : eps_end + (eps_init - eps_end) * max((eps_decay - t - 1)/eps_decay, 0)
#learning_rate = lambda t : alpha_end + (alpha_init - alpha_end) * max((alpha_decay - t - 1)/alpha_decay, 0)
#Exponential decay
eps_decay2 = -np.log(eps_end/eps_init)/eps_decay
alpha_decay2 = -np.log(alpha_end/alpha_init)/alpha_decay
eps_greedy = lambda t : eps_init*np.exp(-t*eps_decay2)
learning_rate = lambda t : alpha_init*np.exp(-t*alpha_decay2)

# How often to checkpoint (save) intermediate Q-functions
checkpoint_iters = 50

def train_agent(args):
    """
    Main training loop for the QAgent:
     1. Load data and initialize random seeds
     2. Create output directory structure
     3. Initialize environment, agent, and simulator
     4. Run training episodes, periodically saving Q-values
     5. Log cumulative rewards, speeds, and convergence flags
    """
    # Unpack noise threshold and load precomputed data
    n_thr = args.n_thr
    T_DNS,NX,NY = args.T_DNS,args.NX,args.NY   
    rescaleTime = args.rescaleTime
    
    #Marco's data
    if(args.Marco):
        data_conc_path = "../../data_Marco/nose_data_27_123.npy"
        data = np.load(data_conc_path) 
    #Maurizio's data
    else:
        data_conc_path = f"../../data_Maurizio/wind{args.wind}/smoothed_conc_{NX}x{NY}_tMax{T_DNS}_training.bin"
        data_vel_path = f"../../data_Maurizio/wind{args.wind}/smoothed_vel_{NX}x{NY}_tMax{T_DNS}_training.bin"
        data = np.fromfile(data_conc_path,dtype=np.float32).reshape(T_DNS,NY,NX)
        data = data[::rescaleTime] #frequency of time snapshots used from the smoothed DNS to be changed depending on system size
        data_vel = np.fromfile(data_vel_path,dtype=np.float32).reshape(T_DNS,2,NY,NX)
        data_vel = data_vel[::rescaleTime] 
    
    #Size of the domain
    NX = data.shape[2]
    NY = data.shape[1]
    T_dns = data.shape[0]
    
    #Simulation parameters
    batch_size = args.batch_size
    num_episodes = args.num_episodes
    
    # Source (goal) position and radius in the environment
    source_position = np.tile(np.array(args.source_pos), (batch_size, 1))
    source_radius = args.source_radius
    
    # Update seed values based on base_seed argument for reproducibility
    for key in seeds.keys():
        seeds[key] =(args.base_seed + 1) * seeds[key] + args.base_seed    
    
    print(f"Agent seed: {seeds['agent_seed']}, Env seed: {seeds['env_seed']}, Sim seed: {seeds['sim_seed']}, Sample seed: {seeds['samp_seed']}")

    # Prepare output directory for this training run
    out_dir = f"{args.out_path}/wind{args.wind}/noise_{n_thr}/size_{NX}x{NY}/horizon{args.horizon}/shaping_{args.shaping_factor}/memory_{args.mem_length}/agent_{seeds['agent_seed']}"
    os.makedirs(f"{out_dir}/Q_stored", exist_ok=True)
    os.makedirs(f"{out_dir}/N_stored", exist_ok=True)
    
    # Find time slices where any value exceeds threshold
    samp_state = np.random.RandomState(seeds['samp_seed'])
    
    # Horizon length for each episode (number of steps before termination)
    horizon = args.horizon
    
    #Memory length of the agent
    mem_length = args.mem_length

    '''def sample_init():
        # Pick a random valid time slice
        t_idx = samp_state.randint(mem_length,T_dns-horizon-1)
        # Get all coords in that slice above threshold
        coords = np.argwhere(data_start[t_idx] > n_thr)
        scra = samp_state.randint(0,len(coords),size=batch_size)
        # Select one at random
        pos = coords[scra]
        return pos, t_idx'''
    
    def sample_init():
        l=0
        while(l<=64):
            # Pick a random valid time slice
            t_idx = samp_state.randint(mem_length,T_dns-horizon-1)
            # Get all coords in that slice above threshold
            coords = np.argwhere(data_start[t_idx] > n_thr) 
            
            source = np.asarray(args.source_pos)         # e.g. [r, c]
            min_dist = 0              # distance threshold
            min_dist_sq = min_dist * min_dist
            d2 = np.sum((coords - source) ** 2, axis=1)  # shape (N,)
            # keep only coords that are further than min_dist
            valid_mask = d2 > min_dist_sq
            coords = coords[valid_mask]
            l=len(coords)
        
        picks = samp_state.randint(0,len(coords),size=batch_size)
        # Select one at random
        pos = coords[picks]
        return pos, t_idx
    
    data[data < n_thr] = 0.0
    data_start = data.copy()
    
    # Exclude source region
    below_source = args.source_pos[0] - source_radius - 2
    above_source = args.source_pos[0] + source_radius + 2
    upwind_source = args.source_pos[1] - source_radius - 2
    downwind_source = args.source_pos[1] + source_radius + 2
    data_start[:,below_source:above_source,upwind_source:downwind_source] = 0.0

    # Discount factor for future rewards
    gamma = 1.-1./horizon
    
    # Initialize environment with given parameters and noise threshold
    env = Environment(data, 
                      data_vel,
                      source_position=source_position, 
                      source_radius=source_radius,
                      gamma = gamma,
                      batch_size = batch_size,
                      mem_length = mem_length,
                      shaping_factor = args.shaping_factor,
                      seed = seeds['env_seed'])
    
    # Initialize Q-learning agent with schedules and output settings
    agent = QAgent(
        batch_size=batch_size,
        horizon=horizon,
        gamma=gamma,
        eps_greedy= eps_greedy,
        learning_rate= learning_rate,
        seed=seeds['agent_seed'],
        s_thr=n_thr,
        out_dir=f"{out_dir}"
    )
      
    retrain = args.retrain
    if(retrain):
        agent.load_q(f"{out_dir}/Q_stored/Qfun.npy")
    
    # Create simulator that ties the agent and environment together
    simulator = Simulator(env=env, actions=actions)
    
    #File for the initial conditions for testing
    init_file = f"../../ic/wind{args.wind}/noise_{n_thr}/size_{NX}x{NY}/init_cond_nTimes80_nAgentsperTime128.bin"
    
    # Run the training process: returns reward, speed, convergence arrays
    cumulative_rewards, agent_speed, convergence, bestQ, maxConv = simulator.train_agent(agent, 
                                                                        point_sampler=sample_init, 
                                                                        horizon=horizon, 
                                                                        num_episodes=num_episodes, 
                                                                        retrain=retrain,
                                                                        init_file=init_file,
                                                                        delta=10, 
                                                                        checkpoint_iters=checkpoint_iters, 
                                                                        seed_sim=seeds['sim_seed']
                                                                        )

    # After training, save the final Q-function and how much each (s,a)-pair has been visited
    agent.store_q("Q_stored/Qfun")
    agent.store_n("N_stored/Nfun")
    
    print(f"Best Q is the one at episode {bestQ} with convergence of {maxConv}")

    # Write training logs for analysis: reward, speed, convergence per episode
    with open(f"{out_dir}/training.log", 'w') as f:
        for i in range(len(cumulative_rewards)):
            f.write("{},{},{}\n".format(cumulative_rewards[i], agent_speed[i], convergence[i]))



if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = ap.ArgumentParser(description="Training QAgent.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # Required positional arguments
    parser.add_argument('--n_thr', type=float, default=0.2, help='noise threshold')
    parser.add_argument('--source_pos', type=int, nargs=2, default=[63,19])
    parser.add_argument('--source_radius', type=int, default=2)
    parser.add_argument('--NX', type=int, default=128)
    parser.add_argument('--NY', type=int, default=128)
    parser.add_argument('--rescaleTime', type=int, default=4)
    parser.add_argument('--T_DNS', type=int, default=10000)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--wind', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of agents in each update batch')
    parser.add_argument('--num_episodes', type=int, default=3001)
    parser.add_argument('--Marco', action='store_true', help='Enable Marco (default: False)')
    parser.add_argument('--mem_length', type=int, default=100)
    parser.add_argument('--retrain', action='store_true', help='Enable retrain (default: False)')
    parser.add_argument('--shaping_factor', type=float, default=0., help='shaping factor for the reward')

    # Optional parameters for reproducibility and output location
    parser.add_argument('--base_seed', type=int, default=143125, help='Seed used to initialize all RNGs')
    parser.add_argument('--out_path', type=str, default='./results', help='output directory')
    
    # Parse and launch training
    args = parser.parse_args()
    train_agent(args)
