import numpy as np
from parla import Parla
from parla.tasks import spawn, AtomicTaskSpace as TaskSpace
from parla.devices import cpu, gpu
from typing import Callable, Optional


def run(function: Callable[[], Optional[TaskSpace]]):
    """
    Parla applications are typically run in a top-level task.
    This function encapsulates the creation of the top-level task and the Parla runtime.

    Args:
        function (Callable[[], Optional[TaskSpace]]): A parla app that (optionally) returns a task space.
    """
    # Start the Parla runtime
    with Parla():
        # Create an encapsulating top-level task to kick off the computation and wait for it to complete.
        @spawn(placement=cpu, vcus=0)
        async def top_level_task():
            # Run the Parla application
            await function()


def initialize_centroids(points, k):
    """Randomly initialize centroids"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    """Find the closest centroid for all points"""
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """Recompute centroids"""
    return np.array(
        [points[closest == k].mean(axis=0) for k in range(centroids.shape[0])]
    )


def kmeans(points, k, max_iters=100, tolerance=1e-5):
    """K-Means algorithm"""
    centroids = initialize_centroids(points, k)
    for _ in range(max_iters):
        closest = closest_centroid(points, centroids)
        new_centroids = move_centroids(points, closest, centroids)
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids

    return centroids, closest


def plot_kmeans(points, centroids, labels, k):
    from matplotlib import pyplot as plt

    """Plot KMeans result with different colors for each cluster"""
    colors = ["r", "g", "b", "y", "c", "m"]
    for i in range(k):
        # plot all points assigned to this cluster
        plt.scatter(
            points[labels == i, 0],
            points[labels == i, 1],
            c=colors[i],
            label=f"Cluster {i}",
        )
        # plot the centroid of this cluster
        plt.scatter(
            centroids[i, 0],
            centroids[i, 1],
            c=colors[i],
            marker="x",
            s=100,
            linewidths=3,
        )

    plt.title("KMeans Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("kmeans.png")


async def use_kmeans():
    import time

    n = 100000
    workers = 4
    points = np.random.rand(n, 2)  # 100 points in 2D
    k = 3  # Number of clusters
    T = TaskSpace("kmeans")
    start_t = time.perf_counter()

    for i in range(workers):

        @spawn(T[i], placement=cpu, vcus=0)
        def task():
            start_i = i * n // workers
            end_i = (i + 1) * n // workers
            centroids, labels = kmeans(points[start_i:end_i], k)

    T.wait()
    end_t = time.perf_counter()
    print(f"Time: {end_t - start_t:.2f}s", flush=True)
    # plot_kmeans(points, centroids, labels, k)


if __name__ == "__main__":
    run(use_kmeans)
