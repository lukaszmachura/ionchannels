import pickle
import time
import os
import ionchannels as ic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import entropy as scipy_shannon
from scipy.stats import mode
import EntropyHub as EH
# from scipy import signal
# from PyEMD import EMD

def my_shannon(x, base=None, bins=33): 
    """ Computes Shanon entropy """
    x = np.asarray(x)
    hx, bx = np.histogram(x, bins=bins, density=False)
    hx = np.asarray(hx, dtype=np.float64)
    hx /= x.size
    hx = hx[hx>0]
    
    if base == None:
        base = np.e
        
    SE = -np.sum(hx * np.log(hx)) / np.log(base)
    return SE


if __name__ == '__main__':
    # check if traj, sampen, plots directories exist, if not create them
    if not os.path.exists('traj'):
        os.makedirs('traj')
    if not os.path.exists('sampen'):
        os.makedirs('sampen')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # flags
    calc_trajectory = True
    calc_sampen = not True
    plot_sampen = not True

    # model
    icm = ic.IonChannelModel(states={'open': [1], 'open value': -1, 'close': [5], 'close value': 1})
    icm.full_stats = True
    print(icm)

    # simulation parameters
    h = 0.002
    tend = 50
    D = 0.01
    # blist = [(0.1, 0.5), (0.1, 0.9), (0.1, 1), (0.1, 1.1), (0.1, 1.5), (0.1, 2), (0.1, 5), (0.1, 10)]
    # blist = [(0.5, 0.1), (0.5, 0.9), (0.5, 1), (0.5, 1.1), (0.5, 1.5), (0.5, 2), (0.5, 5), (0.5, 10)]
    # blist = [(0.9, 0.1), (0.9, 0.5), (0.9, 1), (0.9, 1.1), (0.9, 1.5), (0.9, 2), (0.9, 5), (0.9, 10)]
    # blist = [(1, 0.1), (1, 0.5), (1, 0.9), (1, 1.1), (1, 1.5), (1, 2), (1, 5), (1, 10)]
    # blist = [(1.1, 0.1), (1.1, 0.5), (1.1, 0.9), (1.1, 1), (1.1, 1.5), (1.1, 2), (1.1, 5), (1.1, 10)]
    # blist = [(1.5, 0.1), (1.5, 0.5), (1.5, 0.9), (1.5, 1), (1.5, 1.1), (1.5, 2), (1.5, 5), (1.5, 10)]
    # blist = [(2.0, 0.1), (2.0, 0.5), (2.0, 0.9), (2.0, 1), (2.0, 1.1), (2.0, 1.5), (2.0, 5), (2.0, 10)]
    # blist = [(5.0, 0.1), (5.0, 0.5), (5.0, 0.9), (5.0, 1), (5.0, 1.1), (5.0, 1.5), (5.0, 2), (5.0, 10)]
    blist = [(10.0, 0.1), (10.0, 0.5), (10.0, 0.9), (10.0, 1), (10.0, 1.1), (10.0, 1.5), (10.0, 2), (10.0, 5)]
    # blist = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (1, 1), (1.1, 1.1), (1.5, 1.5), (2, 2), (5, 5), (10, 10)] 

    bfreq = str(mode(np.array(blist).flatten()).mode).replace('.','')

    if not calc_trajectory and not calc_sampen and not plot_sampen: 
        print("Nothing to do. Exiting.")
        exit(0)

    if calc_trajectory:
        print("Starting simulations...")
        for b in blist:
            bm = b[0]
            bp = b[1]
            print(f"Simulating for bplus = {bp}, bminus = {bm}")
            x = icm.simulate(tend=tend, h=h, D=D, levy_stat=False, 
                            force={
                                'potential': 'asym_quadratic', 
                                'params': {'bminus': bm, 'bplus': bp}
                                }
                            )

            with open(f'traj/dynamics_simulation_bplus_{bp}_bminus_{bm}.pkl', 'wb') as f:
                pickle.dump(x, f)
                print(f"Simulation data saved to traj/dynamics_simulation_bplus_{bp}_bminus_{bm}.pkl")

    if calc_sampen:
        print("Calculating SampEn...", blist)
        m = 10
        for b in blist:
            print(f"Processing bplus = {b[1]}, bminus = {b[0]}")
            bm = b[0]
            bp = b[1]
            with open(f'traj/dynamics_simulation_bplus_{bp}_bminus_{bm}.pkl', 'rb') as f:
                x = pickle.load(f)
                print(f"Loaded simulation data from traj/dynamics_simulation_bplus_{bp}_bminus_{bm}.pkl")

            print(f"Computing SampEn for bplus = {bp}, bminus = {bm}")
            t1 = time.time()
            sampen = EH.SampEn(x, m=m) #, r=0.2*np.std(x))
            t2 = time.time()
            print(f"SampEn computed in {t2 - t1:.4f} seconds")

            with open(f'sampen/sampen_bplus_{bp}_bminus_{bm}.pkl', 'wb') as f:
                pickle.dump(sampen, f)
                print(f"SampEn data saved to sampen/sampen_bplus_{bp}_bminus_{bm}.pkl")

    if plot_sampen:
        print("Plotting SampEn results...")
        m = 10
        plt.figure(figsize=(10, 6))
        for b in blist:
            bm = b[0]
            bp = b[1]
            with open(f'sampen/sampen_bplus_{bp}_bminus_{bm}.pkl', 'rb') as f:
                sampen = pickle.load(f)
                print(f"Loaded SampEn data from sampen/sampen_bplus_{bp}_bminus_{bm}.pkl")
                plt.semilogy(range(m+1), sampen[0], '-o', label=f'b+={bp}, b-={bm}', alpha=0.7)
        plt.xlabel('Embedding Dimension m')
        plt.ylabel('SampEn')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/sampen_plot_bm{bfreq}.png', dpi=300)
        

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(0, len(x) * h, h), x, label='Trajectory', alpha=0.7)
    # plt.title('Ion Channel Dynamics Simulation')
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('dynamics_simulation.png', dpi=300)
    # plt.show()