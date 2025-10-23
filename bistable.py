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


# Potential and force (bistable: V(x) = x^4/4 - x^2/2)
def V(x, a, b, c=0):
    return a * x**4 / 4 - b * x**2 / 2 + c*x

def F(x, a, b, c=0):
    return -(a * x**3 - b * x + c)


# Parameters
dt = 0.001
T = 200
N = int(T / dt)
D = 0.01
x0 = -1.0
a, b, c = 1, 1., -0.3

# Brownian dynamics simulation
x = np.zeros(N)
x[0] = x0
for i in range(1, N):
    x[i] = x[i-1] + F(x[i-1], a, b, c) * dt + np.sqrt(2 * D * dt) * np.random.randn()


# Identify residence times
res_times_left = []
res_times_right = []

current_well = np.sign(x[0]) if x[0] != 0 else 1
res_time = 0.0

for i in range(N):
    res_time += dt
    if current_well == 0:  # initial phase, waiting to hit ±1
        if x[i] >= 1:
            current_well = 1
            res_time = 0.0
        elif x[i] <= -1:
            current_well = -1
            res_time = 0.0
    else:
        if current_well == 1 and x[i] <= -1:  # transition right → left
            res_times_right.append(res_time)
            res_time = 0.0
            current_well = -1
        elif current_well == -1 and x[i] >= 1:  # transition left → right
            res_times_left.append(res_time)
            res_time = 0.0
            current_well = 1

# Convert to numpy arrays
res_times_left.append(np.mean(res_times_left))  # avoid empty array
res_times_right.append(np.mean(res_times_right))  # avoid empty array

res_times_left = np.array(res_times_left)
res_times_right = np.array(res_times_right)
avg_left = np.mean(res_times_left)
avg_right = np.mean(res_times_right)
# Print average residence times
print('left well residence time:', len(res_times_left), avg_left)
print('right well residence time:', len(res_times_right), avg_right)

# Calculate histograms using numpy
bins = np.linspace(0, max(np.max(res_times_left), np.max(res_times_right)), 50)
if res_times_left.size:
    hist_left, bin_edges = np.histogram(res_times_left, bins=bins, density=True)
if res_times_right.size:
    hist_right, _ = np.histogram(res_times_right, bins=bins, density=True)

# exit()

max_lag = 20
lags = np.arange(max_lag) * dt
ac = autocorr(x, max_lag)
ac_fft = autocorr_fft(x)[:max_lag]
ac_fft_diff = autocorr_fft(np.diff(x))[:max_lag]

# Power spectral density using Welch's method
freqs, psd = signal.welch(x, fs=1/dt, nperseg=2048)

# Plot results
x_grid = np.linspace(-2, 2, 400)

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(np.arange(N) * dt, x)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Brownian Trajectory")

plt.subplot(2, 2, 2)
# Compute bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# Plot as lines
plt.hist(res_times_left, bins=bins, density=True, alpha=0.5, color='blue', label='Left well', histtype='stepfilled')
plt.hist(res_times_right, bins=bins, density=True, alpha=0.5, color='red', label='Right well', histtype='stepfilled')
plt.plot(bin_centers, (1/avg_left) * np.exp(-bin_centers / avg_left), linestyle='solid', color='blue', label='Left exp. fit')
plt.plot(bin_centers, (1/avg_right) * np.exp(-bin_centers / avg_right), linestyle='solid', color='red', label='Right exp. fit')
plt.xlabel('Residence time')
plt.ylabel('Probability density')
plt.title('Residence Time Distributions (Line Plot)')
plt.legend()
plt.grid(True)
# plt.plot(x_grid, V(x_grid, a, b, c))
# plt.xlabel("x")
# plt.ylabel("V(x)")
# plt.title("Bistable Potential")

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