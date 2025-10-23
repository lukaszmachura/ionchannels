import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
def V(x, xs, a, b=0, c=0):
    return a * (x - xs)**2 / 2 + b * x + c

def F(x, xs, a, b=0):
    return -a * (x - xs) - b


# Parameters
dt = 0.001
T = 500
N = int(T / dt)
D = 0.01
tau_averageOpen = 20  # average value
tau_averageClose = 10  # average value

xO = -1.0
xC = 1.0
a, b, c = 1, 0, 0  # single well at xs=0

residence_timesOpen = []
residence_timesClose = []
# Initial state
tau = np.random.exponential(scale=tau_averageOpen)
residence_timesOpen.append(tau)
xs = xO  # open state

print(f"State: xs={xs}, tau={tau}")

# Brownian dynamics simulation
x = np.zeros(N)
x[0] = xs
for i in range(1, N):
    x[i] = x[i-1] + F(x[i-1], xs, a, b) * dt + np.sqrt(2 * D * dt) * np.random.randn()
    tau -= dt
    if tau <= 0:
        # Switch state
        if xs == xO:
            xs = xC
            tau = np.random.exponential(scale=tau_averageClose)
            residence_timesClose.append(tau)
        else:
            xs = xO
            tau = np.random.exponential(scale=tau_averageOpen)
            residence_timesOpen.append(tau)
        print(f"State switch: xs={xs}, tau={tau}")



max_lag = 20
lags = np.arange(max_lag) * dt
ac = autocorr(x, max_lag)
ac_fft = autocorr_fft(x)[:max_lag]
ac_fft_diff = autocorr_fft(np.diff(x))[:max_lag]

# Power spectral density using Welch's method
freqs, psd = signal.welch(x, fs=1/dt, nperseg=2048)

# Plot results
x_grid = np.linspace(-4, 4, 400)

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(np.arange(N) * dt, x)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Brownian Trajectory")

plt.subplot(2, 2, 2)
# Compute bin centers for plotting
plt.plot(x_grid, V(x_grid, xO, a, b, c))
plt.plot(x_grid, V(x_grid, xC, a, b, c))
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Monostable Potential")

plt.subplot(2, 2, 3)
# plt.plot(lags, ac)
plt.plot(lags, ac_fft, linestyle='dashed')
plt.plot(lags, ac_fft_diff, '-o')
plt.grid()
plt.xlabel("Lag time")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function")

plt.subplot(2, 2, 4)
plt.loglog(freqs[1:], psd[1:])
plt.loglog(freqs[1:], 1/freqs[1:]/100, linestyle='dashed')  # reference slope
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Power Spectrum (Welch)")

plt.tight_layout()
plt.show()