import os
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
from qnav import QAgent, Environment
from qnav.simulators import Simulator

actions = np.array([
    [1., 0.],   # action 0
    [-1., 0.],  # action 1
    [0., 1.],   # action 2
    [0., -1.]   # action 3
], dtype=np.float32)

# Seeds for reproducibility of agent, environment, sampling, and simulation --- DO NOT CHANGE THIS
seeds = {
    'agent_seed': 314,
    'env_seed': 154,  # NOT USEFUL RIGHT NOW
    'samp_seed': 980,
    'sim_seed': 723  # NOT USEFUL RIGHT NOW
}

##### Useless stuff for testing
eps_init, eps_end, eps_decay = 0.99, 0.0001, 520000
alpha_init, alpha_end, alpha_decay = 1.0, 0.001, 10000
eps_greedy = lambda t : eps_end + ( eps_init - eps_end) * max((eps_decay - t - 1)/eps_decay, 0)
learning_rate = lambda t : alpha_end + ( alpha_init - alpha_end) * max((alpha_decay - t - 1)/alpha_decay, 0)
########################


def test_agent(args):
    n_thr = args.n_thr
    T_DNS, NX, NY = args.T_DNS, args.NX, args.NY
    rescaleTime = args.rescaleTime

    # Marco's data
    if(args.Marco):
        data_conc_path = "../../data_Marco/nose_data_27_123.npy"
        data = np.load(data_conc_path)
    else:
        # Maurizio's data
        data_conc_path = f"../../data_Maurizio/wind{args.wind}/smoothed_conc_{NX}x{NY}_tMax{T_DNS}_testing.bin"
        data_vel_path = f"../../data_Maurizio/wind{args.wind}/smoothed_vel_{NX}x{NY}_tMax{T_DNS}_testing.bin"
        data = np.fromfile(data_conc_path, dtype=np.float32).reshape(T_DNS, NY, NX)
        data = data[::rescaleTime] # frequency of time snapshots used from the smoothed DNS to be changed depending on system size
        data_vel = np.fromfile(data_vel_path, dtype=np.float32).reshape(T_DNS, 2, NY, NX)
        data_vel = data_vel[::rescaleTime]


    # Size of the domain
    NY = data.shape[1]
    NX = data.shape[2]

    batch_size = args.batch_size
    num_episodes = args.num_episodes

    # Source (goal) position and radius in the environment
    source_position = np.tile(np.array(args.source_pos), (batch_size, 1))
    source_radius = args.source_radius

    for key in seeds.keys():
        seeds[key] = (args.base_seed + 1) * seeds[key] + args.base_seed

    # TO TEST POLICIES AMONG MEMORIES
    if(args.cross_test):
        agent_dir = f"{args.out_path}/wind{args.wind}/noise_{n_thr}/size_{NX}x{NY}/horizon{args.horizon}/shaping_{args.shaping_factor}/memory_{args.mem_length}/cross-testing/wind{args.wind_crosstest}/memory_{args.mem_length_crosstest}/agent_{seeds['agent_seed']}"
        agent_dir2 = f"{args.out_path}/wind{args.wind_crosstest}/noise_{args.n_thr_crosstest}/size_{NX}x{NY}/horizon{args.horizon}/shaping_{args.shaping_factor}/memory_{args.mem_length_crosstest}/agent_{seeds['agent_seed']}"
    else:
        agent_dir = f"{args.out_path}/wind{args.wind}/noise_{n_thr}/size_{NX}x{NY}/horizon{args.horizon}/shaping_{args.shaping_factor}/memory_{args.mem_length}/agent_{seeds['agent_seed']}"
        agent_dir2 = agent_dir

    out_dir = f"{agent_dir}/test/"
    os.makedirs(f"{out_dir}", exist_ok=True)

    # Find time slices where any value exceeds threshold
    T_dns = data.shape[0]
    samp_state = np.random.RandomState(seeds['samp_seed'])

    # Horizon length for each episode (number of steps before termination)
    horizon = args.horizon

    # Memory length of the agent
    mem_length = args.mem_length

    def sample_init():
        # Pick a random valid time slice
        n_coords = 0
        while(n_coords < batch_size):
            t_idx = samp_state.randint(mem_length, T_dns-horizon-1)
            # Get all coords in that slice above threshold
            coords = np.argwhere(data_start[t_idx] > n_thr)
            ##################################
            # Conditioning on minimum distance
            source = np.asarray(args.source_pos)
            min_dist = 0  # distance threshold
            min_dist_sq = min_dist * min_dist
            d2 = np.sum((coords - source) ** 2, axis=1)
            valid_mask = d2 > min_dist_sq
            coords = coords[valid_mask]
            ###########################
            n_coords = len(coords)

        # Select initial positions at random without duplicates
        scra = samp_state.permutation(n_coords)[:batch_size]
        pos = coords[scra]
        return pos, t_idx

    data[data < n_thr] = 0.0
    data_start = data.copy()

    # Exclude source region
    below_source = args.source_pos[0] - source_radius - 2
    above_source = args.source_pos[0] + source_radius + 2
    upwind_source = args.source_pos[1] - source_radius - 2
    downwind_source = args.source_pos[1] + source_radius + 2
    data_start[:, below_source:above_source, upwind_source:downwind_source] = 0.0

    # Discount factor for future rewards
    gamma = 1.-1./horizon

    env = Environment(data,
                      data_vel,
                      source_position=source_position,
                      source_radius=source_radius,
                      gamma=gamma,
                      batch_size=batch_size,
                      mem_length=mem_length,
                      seed=seeds['env_seed'])

    agent = QAgent(
        batch_size=batch_size,
        gamma=gamma,
        eps_greedy=eps_greedy,
        learning_rate=learning_rate,
        seed=seeds['agent_seed'],
        s_thr=n_thr,
        out_dir=f"{agent_dir2}"
    )

    if(args.bestQ):
        agent.load_q(f"{agent_dir2}/Q_stored/Qfun_best.npy")
    else:
        agent.load_q(f"{agent_dir2}/Q_stored/Qfun.npy")

    simulator = Simulator(env=env, actions=actions)

    all_trajs = []

    # File for the initial conditions
    init_file = f"../../ic/wind{args.wind}/noise_{n_thr}/size_{NX}x{NY}/init_cond_nTimes{num_episodes}_nAgentsperTime{batch_size}.bin"

    results, all_trajs = simulator.test_agent(agent,
                                              point_sampler=sample_init,
                                              num_episodes=num_episodes,
                                              init_file=init_file,
                                              training=0,
                                              horizon=horizon
                                              )

    # Combine results across episodes: list of (batch_size, 6) → (n_episodes * batch_size, 6)
    results_array = np.vstack(results)

    creward, ttr, tmin, convergence = [], [], [], []

    with open(f"{out_dir}/test.log", 'w') as f:
        #(init_pos[0], init_pos[1], c_reward, ttr, tmin, int(terminated)), s_max
        for result in results_array:
            creward.append(result[2])
            if result[5]:
                ttr.append(result[3])
                tmin.append(result[4])
            convergence.append(result[5])
            f.write(",".join([str(elem) for elem in result]) + "\n")

    # Save MSD data
    tau, msd, msd_x, msd_y, counts = simulator.get_msd_since_last_encounter()
    msd_data = np.column_stack((tau, msd, msd_x, msd_y, counts))
    np.savetxt(f"{out_dir}/msd_xy.txt", msd_data, fmt="%.6e")

    # Save trajectories
    if(args.save_trajs):
        np.array(all_trajs, dtype=np.float32).tofile(f"{out_dir}/trajectories.bin")

    # Print final statement and save summary
    print("[--] G = {} +/- {}".format(np.mean(creward), np.std(creward)))
    print("[--] convergence = {} +/- {}".format(np.mean(convergence), np.std(convergence)))
    with open(f"{out_dir}/test_summary.log", 'w') as f:
        mu_crew, std_crew = np.mean(creward) if len(creward) > 0 else 0.0, np.std(creward) if len(creward) > 0 else 0.0
        mu_conv, std_conv = np.mean(convergence) if len(convergence) > 0 else 0.0, np.std(convergence) if len(convergence) > 0 else 0.0
        f.write(f"{mu_crew},{std_crew},{mu_conv},{std_conv}\n")


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Test QAgent.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # Data parameters
    parser.add_argument('--n_thr', type=float, default=0.2, help='noise threshold')
    parser.add_argument('--source_pos', type=int, nargs=2, default=[63, 19])
    parser.add_argument('--source_radius', type=int, default=2)
    parser.add_argument('--NX', type=int, default=128)
    parser.add_argument('--NY', type=int, default=128)
    parser.add_argument('--rescaleTime', type=int, default=4)
    parser.add_argument('--T_DNS', type=int, default=10000)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--wind', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128, help='Number of agents in each update batch')
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--Marco', action='store_true', help='Enable Marco (default: False)')
    parser.add_argument('--mem_length', type=int, default=1)
    parser.add_argument('--cross_test', action='store_true', help='Enable cross test with different memories (default: False)')
    parser.add_argument('--mem_length_crosstest', type=int, default=100)
    parser.add_argument('--wind_crosstest', type=int, default=2)
    parser.add_argument('--n_thr_crosstest', type=float, default=0.2)
    parser.add_argument('--bestQ', action='store_true', help='Enable using the best Q for testing (default: False)')
    parser.add_argument('--shaping_factor', type=float,default=0., help='shaping factor for the reward')
    parser.add_argument('--save_trajs', action='store_true', help='Enable save_trajs (default: False)')


    parser.add_argument("--base_seed", type=int, default=143125)
    parser.add_argument("--out_path", type=str, default='./results', help='output directory')

    args = parser.parse_args()
    test_agent(args)
