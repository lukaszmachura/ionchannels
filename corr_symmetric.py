import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 3d plot of correlation functions for different parameters
# correlations are precomputed and pickled in correlation/symmetric folder

DIR = 'symmetric'
bext = 1000
apd_values = [1
    # 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    # 1, 2, 3, 4, 5, 6, 7, 8, 9,
    # 10, 20, 30, 40, 50, 60, 70, 80, 90,
    # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
]
apd_values = sorted(set(apd_values))

# for apd in apd_values:
for bext in apd_values:
    # DIR = f'bext{bext}ap{apd}'
    # ODIR = f'bext{bext}ap{apd:08.2f}'
    # DIR = f'bext{bext}_inter_sym'
    # ODIR = f'bext{bext:08.3f}_inter_sym'

    blist = [(b, ) * 4 for b in np.logspace(-3, 3, 50)]

    stop = -1
    nc = []
    for b in blist:
        am, bm, ap, bp = b
        # with open(f'correlation/{DIR}/corr_Op1_vm1_Cp2_v1_D0.01_T500_h0.001_am{bext}_bm{bm}_ap{ap}_bp{bext}.pkl', 'rb') as f:
        with open(f'correlation/{DIR}/corr_Op5_vm1_Cp10_v1_D0.01_T1000_h0.001_am{am}_bm{bm}_ap{ap}_bp{bp}.pkl', 'rb') as f:    
            d = pickle.load(f)
            nc.append(d['noise_corr'][:stop])
        

    if not os.path.exists('corr3D'):
        os.makedirs('corr3D')
        
    fig = plt.figure(figsize=[d / 2 / 2.54 for d in [29.7, 21]])  # half A4 landscape in cm 21 x 29.7
    plt.imshow(np.array(nc), aspect='auto', origin='lower', 
            extent=[0, 40, np.log10(blist[0][0]), np.log10(blist[-1][0])], # type: ignore
            interpolation='lanczos',
            cmap='jet_r'
            )
    plt.colorbar(label='Autocorrelation')

    # plt.title(f'$a_-=b_+={bext}, a_+={apd}$')
    # plt.title(fr'$a_-=b_+={bext}, \quad b_{{-}}=a_{{+}}$')
    plt.title(fr'$a_{{-}}=b_{{-}}=a_{{+}}=b_{{+}}$')
    plt.xlabel(r'$\tau$')
    # plt.ylabel(r'$\log b_-$')
    plt.ylabel(r'$\log b_{{\pm}}, \log a_{{\pm}}$')

    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'corr3D/corr_{DIR}.png', dpi=300)
    plt.close(fig)
