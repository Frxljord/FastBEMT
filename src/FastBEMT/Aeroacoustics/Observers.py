import numpy as np


def uniform_observer_grid(size: float, nx: int, ny: int) -> np.ndarray:
    '''Generate a grid of observer points for acoustic analysis.

    Args:
        size: The extent of the grid in the x and y directions (m).
        nx: Number of observer points along the x-axis.
        ny: Number of observer points along the y-axis.

    Returns:
        np.ndarray: A 3D array of observer points.
    '''

    x_range = np.linspace(-size, size, nx)
    y_range = np.linspace(0, size, ny)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X) 

    r_observers = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return r_observers

def circular_observer_array(radius: float, n_points: int) -> np.ndarray:
    '''Generate a circular array of observer points for acoustic analysis.

    Args:
        radius: The radius of the circular array (m).
        n_points: The number of observer points to generate.

    Returns:
        np.ndarray: A 3D array of observer points.
    '''
    angles = np.linspace(-np.pi / 2 , np.pi / 2, n_points, endpoint=True)
    X = radius * np.sin(angles)
    Y = radius * np.cos(angles)
    Z = np.zeros_like(X)

    r_observers = np.stack([X, Y, Z], axis=1)
    return r_observers