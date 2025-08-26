import numpy as np
import matplotlib.pyplot as plt

def plot_box(center, orientation, width, length, color='b'):
    """
    Plot a box in 2D given its center, orientation, width and length.
    
    Parameters:
        center      -- (x, y) tuple for the center of the box
        orientation -- angle in radians (0 = along x-axis)
        width       -- width of the box (perpendicular to orientation)
        length      -- length of the box (along orientation)
        color       -- color for the box edges
    """
    x, y = center
    theta = orientation

    # Half dimensions
    dx = length / 2
    dy = width / 2

    # Rectangle corners relative to center (local coordinates)
    corners = np.array([
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy]  # close the box
    ])

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate and translate corners
    world_corners = (R @ corners.T).T + np.array([x, y])

    # Plot
    plt.plot(world_corners[:, 0], world_corners[:, 1], color)
    plt.axis('equal')

# Example usage
center = (5, 5)
orientation = np.pi / 3  # 30 degrees
width = 2
length = 4

plot_box(center, orientation, width, length)
plt.title("Oriented Box")
plt.grid(True)
plt.show()

print("donr :)")

plot_box([2,2],0.707,2,5)