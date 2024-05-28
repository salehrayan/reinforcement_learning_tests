import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Objective function (Example: Rastrigin function)
def objective_function(x, y):
    return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

# Initialize fireflies
def initialize_fireflies(num_fireflies, lower_bound, upper_bound):
    fireflies = np.random.uniform(low=lower_bound, high=upper_bound, size=(num_fireflies, 2))
    return fireflies

# Calculate brightness
def calculate_brightness(fireflies):
    return np.apply_along_axis(lambda pos: -objective_function(pos[0], pos[1]), 1, fireflies)

# Update firefly positions
def update_positions(fireflies, brightness, alpha=0.2, beta=1, gamma=1):
    num_fireflies = len(fireflies)
    new_fireflies = np.copy(fireflies)
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if brightness[j] > brightness[i]:
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta_effect = beta * np.exp(-gamma * r**2)
                new_fireflies[i] += beta_effect * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(2) - 0.5)
    return new_fireflies

# Create a grid for the background
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
scat = ax.scatter([], [], c='red')

# Draw the initial background
background = ax.imshow(Z, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis', alpha=0.5)

# Update function for animation
def update(frame, fireflies, brightness):
    global num_iterations
    num_iterations += 1
    fireflies[:] = update_positions(fireflies, brightness)
    brightness[:] = calculate_brightness(fireflies)
    scat.set_offsets(fireflies)
    background.set_data(Z)
    return scat, background

# Parameters
num_fireflies = 20
lower_bound = -5
upper_bound = 5
num_iterations = 0

# Initialize fireflies and brightness
fireflies = initialize_fireflies(num_fireflies, lower_bound, upper_bound)
brightness = calculate_brightness(fireflies)

# Run animation
ani = animation.FuncAnimation(fig, update, fargs=(fireflies, brightness), interval=200, blit=True)
plt.show()
