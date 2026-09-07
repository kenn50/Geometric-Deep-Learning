import torch

#Made by ChatGpt 5.6 sol
def farthest_point_sample(points, k):
    """
    points: [N, D]
    k: number of points to select

    returns:
        sampled_points: [k, D]
        indices: [k]
    """
    N = points.shape[0]

    indices = torch.empty(k, dtype=torch.long, device=points.device)
    min_dist = torch.full((N,), float("inf"), device=points.device)

    # Start from point 0
    current = 0

    for i in range(k):
        indices[i] = current

        # Distance from every point to the newly selected point
        dist = ((points - points[current]) ** 2).sum(dim=1)

        # Distance to closest selected point so far
        min_dist = torch.minimum(min_dist, dist)

        # Choose the point farthest from the selected set
        current = torch.argmax(min_dist).item()

    return points[indices], indices