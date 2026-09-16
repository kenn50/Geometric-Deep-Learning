import torch
import numpy as np
import plotly.graph_objects as go

# Made by ChatGpt 5.6 sol
def farthest_point_sample(points, k):
    """
    points: (N, D)
    k: number of points to select

    Returns
    -------
    sampled_points : (k, D)
    indices : (k,)
    """
    points = np.asarray(points)
    N = points.shape[0]

    if not 1 <= k <= N:
        raise ValueError(f"k must be between 1 and {N}, got {k}")

    indices = np.empty(k, dtype=np.int64)
    min_dist = np.full(N, np.inf)

    # Start from point 0
    current = 0

    for i in range(k):
        indices[i] = current

        # Squared Euclidean distance to newly selected point
        diff = points - points[current]
        dist = np.sum(diff * diff, axis=1)

        # Distance to closest selected point
        min_dist = np.minimum(min_dist, dist)

        # Farthest point from the selected set
        current = np.argmax(min_dist)

    return points[indices], indices



def generate_torus(num_points, R=1.0, r=0.3):
    uv = np.random.rand(num_points, 2) * 2*np.pi
    def f(row):
        u,v = row
        return np.array([np.cos(u) * (R + r*np.cos(v)), np.sin(u) * (R + r*np.cos(v)), r*np.sin(v)])

    data = np.apply_along_axis(f, 1, uv)
    return data



# Taken from Kevins Bachelor project
def plot3d(d, title="3D plot", color=None):
    """
    d: list of (dataset, label)
       Each dataset should be array-like with shape (n, 3) or reshapeable to (-1, 3).
    """

    fig = go.Figure()

    for dataset, label in d:
        points = np.asarray(dataset).reshape(-1, 3)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=label,
                marker=dict(size=2, color=color),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            
            font=dict(
                family="Arial",
                size=24,
                color="navy",
            ),
        ),
        margin=dict(t=60),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            
        ),
    )


    fig.show()
    
    


