import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters
a, b, c = 1, 1., -0.3

# Define derivative of potential
def dUdx(x):
    return 4*a*x**3 - 2*b*x - c

# Solve for stationary points
x_left = fsolve(dUdx, -1.0)[0]
x_max = fsolve(dUdx, 0.0)[0]
x_right = fsolve(dUdx, 1.0)[0]

# Potential function
def U(x):
    return a*x**4 - b*x**2 - c*x

U_left = U(x_left)
U_max = U(x_max)
U_right = U(x_right)

# Barrier heights
deltaU_left = U_max - U_left
deltaU_right = U_max - U_right

# Plot potential
x_vals = np.linspace(x_left-0.5, x_right+0.5, 400)
U_vals = U(x_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, U_vals, label='Tilted bistable potential', lw=2)
plt.scatter([x_left, x_max, x_right], [U_left, U_max, U_right],
            color=['blue','black','red'], zorder=5)
plt.text(x_left, U_left, f'Left min\nΔU={deltaU_left:.2f}', color='blue', ha='right', va='bottom')
plt.text(x_max, U_max, 'Saddle', color='black', ha='center', va='bottom')
plt.text(x_right, U_right, f'Right min\nΔU={deltaU_right:.2f}', color='red', ha='left', va='bottom')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Tilted Bistable Potential with Barrier Heights')
plt.grid(True)
plt.legend()
plt.show()