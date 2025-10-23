import pickle
import time
import os
import ionchannels as ic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mode


def cleanval(val):
    return str(val).replace('.', '').replace('-', 'm')


def autocorr1(x,lags):
    '''np.corrcoef, partial'''
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)


def autocorr2(x,lags):
    '''manualy compute, non partial'''
    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    return np.array(corr)


def autocorr3(x,lags):
    '''fft, pad 0s, non partial'''
    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp=x-np.mean(x)
    var=np.var(x)

    # do fft and ifft
    cf=np.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:len(lags)]


def autocorr4(x,lags):
    '''fft, don't pad 0s, non partial - fastest, use this'''
    x = np.asarray(x)
    mean=x.mean()
    var=np.var(x)
    xp=x-mean

    cf=np.fft.fft(xp)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real/var/len(x)

    return corr[:len(lags)]


def autocorr5(x,lags):
    '''np.correlate, non partial'''
    x = np.asarray(x)
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]




if __name__ == '__main__':
    VERBOSE = not True

    # check if traj, sampen, plots directories exist, if not create them
    if not os.path.exists('traj'):
        os.makedirs('traj')
    if not os.path.exists('plots_traj'):
        os.makedirs('plots_traj')
    if not os.path.exists('correlation'):
        os.makedirs('correlation')
    if not os.path.exists('plots_correlation'):
        os.makedirs('plots_correlation')

    # flags
    calc_trajectory = True
    plot_trajectory = True
    calc_correlation = True
    plot_correlation = True

    # model
    open_prob = 5
    open_val = -1
    close_prob = 10
    close_val = 1
    icm = ic.IonChannelModel(states={'open': [open_prob], 'open value': open_val, 
                                     'close': [close_prob], 'close value': close_val})
    icm.full_stats = True
    if VERBOSE:
        print(icm)

    # simulation parameters
    h = 0.001
    tend = 1000
    D = 0.01

    # symmetric cases
    blist = [(b, ) * 4 for b in np.logspace(-3, 2, 50)]
    # print(blist)
    # exit()  

    # asymmetric cases
    # bext = 10
    for bext in [0
        # 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        # 1, 2, 3, 4, 5, 6, 7, 8, 9,
        # 10, 20, 30, 40, 50, 60, 70, 80, 90,
        # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
        ]:
        # if not os.path.exists(f'correlation/bext{bext}ap{apd:08.2f}'):
        #     os.makedirs(f'correlation/bext{bext}ap{apd:08.2f}')
        if not os.path.exists(f'correlation/bext{bext}_inter_sym'):
            os.makedirs(f'correlation/bext{bext}_inter_sym')

        blist = [(b, b, b, b) for b in np.logspace(-3, 3, 50)]
        # blist = [(bext, b, b, bext) for b in np.logspace(-3, 3, 50)]
        # blist = [(bext, b, apd, bext) for b in np.logspace(-3, 3, 50)]

        if not calc_trajectory and not calc_correlation and not plot_correlation: 
            print("Nothing to do. Exiting.")
            exit(0)

        if calc_trajectory:
            print("Starting simulations...")
            for b in blist:
                am, bm, ap, bp = b 
                if file_exists := os.path.exists(f'traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl'):
                    if VERBOSE:
                        print(f"Simulation data already exists for a- = {am} b- = {bm} a+ = {ap}, b+ = {bp}. Skipping...")
                    continue
                print(f"Simulating for a- = {am} b- = {bm} a+ = {ap}, b+ = {bp}")
                x = icm.simulate(tend=tend, h=h, D=D, levy_stat=False, 
                                force={
                                    'potential': 'asym_quadratic', 
                                    'params': {'aminus': am, 'bminus': bm, 'aplus': ap, 'bplus': bp}
                                    }
                                )

                with open(f'traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'wb') as f:
                    pickle.dump(x, f)
                    if VERBOSE:
                        print(f"Simulation data saved to traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl")

        if plot_trajectory:
            print("Plotting Trajectories...")
            for b in blist:
                am, bm, ap, bp = b 
                with open(f'traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'rb') as f:
                    x = pickle.load(f)
                    if VERBOSE:
                        print(f"Loaded trajectory data from traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl")
                    plt.clf()
                    fig = plt.figure(figsize=[d / 2 / 2.54 for d in [29.7, 21]])  # half A4 landscape in cm 21 x 29.7
                    plt.plot(np.arange(len(x)) * h, x, lw=0.5)
                    plt.title(f'Trajectory (a-={am}, b-={bm}, a+={ap}, b+={bp})')
                    plt.xlabel('Time')
                    plt.ylabel('x')
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(f'plots_traj/traj_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.png')
                    plt.close()
                    if VERBOSE:
                        print(f"Trajectory plot saved to plots_traj/traj_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.png")

        if calc_correlation:
            print("Calculating Correlation...")
            lags = range(40)
            for b in blist:
                am, bm, ap, bp = b 
                if file_exists := os.path.exists(f'correlation/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl'):
                    if VERBOSE:
                        print(f"Correlation data already exists for a- = {am} b- = {bm} a+ = {ap}, b+ = {bp}. Skipping...")
                    continue

                with open(f'traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'rb') as f:
                    x = pickle.load(f)
                    if VERBOSE:
                        print(f"Loaded trajectory data from traj/sim_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl")
                    start_time = time.time()
                    # y = x[::10]  # downsample by factor 10
                    y = x  # no downsample
                    # corr = autocorr1(y, lags)
                    # corr = autocorr2(y, lags)
                    # corr = autocorr3(y, lags)
                    # corr = autocorr5(y, lags)

                    corr = autocorr4(y, lags) # fastest
                    noise_corr = autocorr4(np.diff(y), lags) # fastest

                    end_time = time.time()
                    if VERBOSE:
                        print(f"Autocorrelation computed in {end_time - start_time:.2f} seconds.")
                    with open(f'correlation/bext{bext}_inter_sym/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'wb') as f_corr:
                        pickle.dump({'lags': lags, 'corr': corr, 'noise_corr': noise_corr}, f_corr)
                        if VERBOSE:
                            print(f"Correlation data saved to correlation/bext{bext}_inter_sym/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl")

        if plot_correlation:
            print("Plotting Correlation...")
            for b in blist:
                am, bm, ap, bp = b
                with open(f'correlation/bext{bext}_inter_sym/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'rb') as f_corr:
                    data = pickle.load(f_corr)
                    lags = data['lags']
                    corr = data['corr']
                    noise_corr = data['noise_corr']
                    if VERBOSE:
                        print(f"Loaded correlation data from correlation/bext{bext}_inter_sym/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl")

                    plt.clf()
                    fig = plt.figure(figsize=[d / 2 / 2.54 for d in [29.7, 21]])  # half A4 landscape in cm 21 x 29.7
                    # plt.plot(lags, corr, label=f'L a{am} b{bm}, R a{ap}, b{bp}', marker='o', markersize=4, alpha=0.7)
                    # plt.plot(lags, noise_corr, label=f'L a{am} b{bm}, R a{ap}, b{bp} (noise)', marker='o', markersize=4, alpha=0.7)
                    plt.plot(lags, corr, label=f'signal', alpha=1, lw=2)
                    plt.bar(lags, noise_corr, width=1, color='tab:orange',  
                            label=f'noise', alpha=0.71) #, lw=2)
                    plt.title(f'a-={am}, b-={bm}, a+={ap}, b+={bp}')
                    plt.xlabel(r'$\tau$')
                    plt.ylabel(r'$R_{XX}(\tau)$')
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'plots_correlation/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.png')
                    plt.close()
                    if VERBOSE:
                        print(f"Correlation plot saved to plots_correlation/corr_Op{open_prob}_v{cleanval(open_val)}_Cp{close_prob}_v{cleanval(close_val)}_D{D}_T{tend}_h{h}_am{am}_bm{bm}_ap{ap}_bp{bp}.png")
