import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Autocorrelation function
def autocorr(x, max_lag):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    result /= result[0]
    return result[:max_lag]

# Fast autocorrelation using FFT
def autocorr_fft(x):
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.fft(x, n*2)
    ps = np.abs(f)**2
    ac = np.fft.ifft(ps).real[:n]
    ac /= ac[0]
    return ac


# Potential and force (bistable: V(x, xs) = a(x-xs)^2/2 + b(x-xs) + c)
def V(x, xs, a1=1, a2=1, b=0, c=0):
    return np.where(x < xs, 
                    a1 * (x - xs)**2 / 2 + b * x + c, 
                    a2 * (x - xs)**2 / 2 + b * x + c
                    )

def F(x, xs, a1=1, a2=1, b=0):
    return np.where(x < xs, 
                    -a1 * (x - xs) - b, 
                    -a2 * (x - xs) - b
                    )

simtime_start = time.time()

# Parameters
dt = 0.001
T = 1000
N = int(T / dt)
D = 0.001
tau_averageOpen = 1  # average value
tau_averageClose = 10  # average value

xO = -1.0
xC = 1.0
a1open, a2open = 10, 5 # stiffness for open and close wells
a1close, a2close = 8, 10 # stiffness for open and close wells
b, c = 0, 0  # single well at xs=0

residence_timesOpen = []
residence_timesClose = []

# Initial state
tau = np.random.exponential(scale=tau_averageOpen)
residence_timesOpen.append(tau)
xs = xO  # open state
a1, a2 = a1open, a2open

print(f"State: xs={xs}, tau={tau}, a1={a1}, a2={a2}")

# Brownian dynamics simulation
x = np.zeros(N)
x[0] = xs
tautotal = tau
for i in range(1, N):
    x[i] = x[i-1] + F(x[i-1], xs, a1, a2, b) * dt + np.sqrt(2 * D * dt) * np.random.randn()
    tau -= dt
    if tau <= 0:
        # Switch state
        if xs == xO:
            xs = xC
            a1, a2 = a1close, a2close
            tau = np.random.exponential(scale=tau_averageClose)
            residence_timesClose.append(tau)
        else:
            xs = xO
            a1, a2 = a1open, a2open
            tau = np.random.exponential(scale=tau_averageOpen)
            residence_timesOpen.append(tau)
        tautotal += tau
        print(f"State switch: xs={xs}, tau={tau}, a1={a1}, a2={a2}")



max_lag = 20
lags = np.arange(max_lag) * dt
# ac = autocorr(x, max_lag)
ac_fft = autocorr_fft(x)[:max_lag]
ac_fft_diff = autocorr_fft(np.diff(x))[:max_lag]

# Power spectral density using Welch's method
freqs, psd = signal.welch(x, fs=1/dt, nperseg=2048)

# check if directory double_monostable_asymmetric exists
if not os.path.exists("double_monostable_asymmetric"):
    os.makedirs("double_monostable_asymmetric")

# Plot results
x_grid = np.linspace(-4, 4, 400)

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
N_target = 10000
indices = np.linspace(0, len(x)-1, N_target, dtype=int)
t_full = np.linspace(0, T, len(x))  # oryginalne czasy
x_downsampled = x[indices]
t_downsampled = t_full[indices]

plt.plot(t_downsampled, x_downsampled, label=rf'$T={T}, h={dt}$')
plt.yticks([xO, xC], ['Open', 'Close'])
plt.legend()
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Brownian Trajectory")

plt.subplot(2, 2, 2)
# Compute bin centers for plotting
plt.plot(x_grid, V(x_grid, xO, a1open, a2open, b, c), label=rf'$a1={a1open}, a2={a2open}, b={b}, c={c}$')
plt.plot(x_grid, V(x_grid, xC, a1close, a2close, b, c), label=rf'$a1={a1close}, a2={a2close}$')
plt.legend()
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Monostable Potential")

plt.subplot(2, 2, 3)
# plt.plot(lags, ac)
plt.plot(lags, ac_fft, linestyle='dashed', label=rf'$D={D}$')
plt.plot(lags, ac_fft_diff, '-o', label=rf'$\tau_O={tau_averageOpen}, \tau_C={tau_averageClose}$')
plt.legend()
plt.grid()
plt.xlabel("Lag time")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function")

plt.subplot(2, 2, 4)
plt.loglog(freqs[1:], psd[1:], label=rf'signal')
plt.loglog(freqs[1:], 1/freqs[1:]/100, linestyle='dashed', label='1/f')  # reference slope
plt.loglog(freqs[1:], 1/freqs[1:]**2/100, linestyle='dotted', label='1/f^2')  # reference slope
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Power Spectrum (Welch)")

plt.tight_layout()
plt.savefig(
    (
        f"double_monostable_asymmetric/"
        f"dm_D{D}_tauO{tau_averageOpen}_tauC{tau_averageClose}_"
        f"a1o{a1open}_a2o{a2open}_a1c{a1close}_a2c{a2close}_b{b}_c{c}_"
        f"T{T}_h{dt}.png"
    )
    , dpi=300
    )

simtime_end = time.time()
print(f"Total residence time recorded: {tautotal}, expected: {T} = {N*dt}")
print(f"Simulation and plotting time: {simtime_end - simtime_start} seconds")