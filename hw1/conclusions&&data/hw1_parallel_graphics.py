import gzip
import pickle
import numpy as np
from collections import Counter
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing


def load_data():
    """Uploading training and test data"""
    with open("X_train (1)", "rb") as file:
        X_train = pickle.load(file)
    with open("Y_train (1)", "rb") as file:
        Y_train = pickle.load(file)
    with open("X_test (1)", "rb") as file:
        X_test = pickle.load(file)
    with open("Y_test (1)", "rb") as file:
        Y_test = pickle.load(file)
    return X_train, Y_train, X_test, Y_test


def compute_compressed_length(x):
    """Calculating the length of a gzip-compressed numpy array"""
    return len(gzip.compress(np.array(x).tobytes()))


def compute_ncd(x1, Cx1, x2, Cx2):
    """Calculating the Normalized Compression Distance (NCD) between two sequences"""
    x1x2 = x1 + x2
    Cx1x2 = compute_compressed_length(x1x2)
    return (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)


def original_knn(X_train, Y_train, X_test, k=101):
    """The original kNN implementation with nested loops"""
    predicts = []
    for x1 in X_test:
        Cx1 = compute_compressed_length(x1)
        distance_from_x1 = []
        for x2 in X_train:
            Cx2 = compute_compressed_length(x2)
            ncd = compute_ncd(x1, Cx1, x2, Cx2)
            distance_from_x1.append(ncd)
        sorted_idx = np.argsort(np.array(distance_from_x1))
        top_k_class = np.array(Y_train)[sorted_idx[:k]]
        predict_class = Counter(top_k_class).most_common(1)[0][0]
        predicts.append(predict_class)
    return predicts


def optimized_serial_knn(X_train, Y_train, X_test, C_train, k=101):
    """Optimized serial implementation of kNN using numpy"""
    predicts = []

    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    for x1 in X_test:
        Cx1 = compute_compressed_length(x1)
        # Calculating C(x1 + x2) for all x2 in X_train
        Cx1x2 = np.array([compute_compressed_length(x1 + x2)
                         for x2 in X_train_np])
        # Вычисление NCD
        ncd = (Cx1x2 - np.minimum(Cx1, C_train)) / np.maximum(Cx1, C_train)
        # Getting the top k indexes
        sorted_idx = np.argsort(ncd)[:k]
        top_k_class = Y_train_np[sorted_idx]
        predict_class = Counter(top_k_class).most_common(1)[0][0]
        predicts.append(predict_class)
    return predicts


def optimized_parallel_knn(X_train, Y_train, X_test, C_train, k=101, n_jobs=-1, backend='loky'):
    """Optimized parallel implementation of kNN using Joblab"""
    Y_train_np = np.array(Y_train)
    X_train_np = np.array(X_train)

    def predict(x1):
        Cx1 = compute_compressed_length(x1)
        Cx1x2 = np.array([compute_compressed_length(x1 + x2)
                         for x2 in X_train_np])
        ncd = (Cx1x2 - np.minimum(Cx1, C_train)) / np.maximum(Cx1, C_train)
        sorted_idx = np.argsort(ncd)[:k]
        top_k_class = Y_train_np[sorted_idx]
        return Counter(top_k_class).most_common(1)[0][0]

    predicts = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(predict)(x1) for x1 in X_test)
    return predicts


def optimized_parallel_knn_variants(X_train, Y_train, X_test, C_train, k=101, backends=['loky', 'threading'], max_workers=None):
    """
    Optimized parallel implementation of kNN using different backends and the number of cores.
    Returns a dictionary with the execution time for each combination of the backend and the number of cores.
    """
    Y_train_np = np.array(Y_train)
    X_train_np = np.array(X_train)
    num_cores = multiprocessing.cpu_count()

    results = {backend: [] for backend in backends}
    cores_range = range(1, num_cores + 1)

    for backend in backends:
        print(f"\nUsing the backend: {backend}")
        for n_jobs in cores_range:
            print(f"  Number of cores: {n_jobs}", end='', flush=True)
            start_time = time.time()

            def predict(x1):
                Cx1 = compute_compressed_length(x1)
                Cx1x2 = np.array([compute_compressed_length(x1 + x2)
                                 for x2 in X_train_np])
                ncd = (Cx1x2 - np.minimum(Cx1, C_train)) / \
                    np.maximum(Cx1, C_train)
                sorted_idx = np.argsort(ncd)[:k]
                top_k_class = Y_train_np[sorted_idx]
                return Counter(top_k_class).most_common(1)[0][0]
            _ = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(predict)(x1) for x1 in X_test)
            elapsed_time = time.time() - start_time
            print(f" -> The time is: {elapsed_time:.2f} seconds")
            results[backend].append(elapsed_time)
    return results, cores_range


def main():
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Training samples: {len(Y_train)}, Test samples: {len(Y_test)}")

    k = 101

    # Precomputation of C(x_train) to avoid repeated calculations in each iteration
    print("Precomputation of compressed lengths for training data...")
    start_time = time.time()
    C_train = np.array([compute_compressed_length(x2) for x2 in X_train])
    print(
        f"The pre-calculation is completed in {time.time() - start_time:.2f} seconds.\n")

    # Original implementation
    print("Execution of the original kNN implementation...")
    start_time = time.time()
    original_preds = original_knn(X_train, Y_train, X_test, k)
    original_time = time.time() - start_time
    print(f"The original implementation took {original_time:.2f} seconds.\n")

    # Optimized serial implementation
    print("Running optimized serial k-NN implementation...")
    start_time = time.time()
    optimized_serial_preds = optimized_serial_knn(
        X_train, Y_train, X_test, C_train, k)
    optimized_serial_time = time.time() - start_time
    print(f"The optimized serial implementation took {
          optimized_serial_time:.2f} seconds.")

    speedup_serial = original_time / optimized_serial_time
    print(f"Acceleration (serial): {speedup_serial:.2f}x\n")

    # Simultaneously with different number of cores and backends
    print("Performing an optimized parallel implementation of kNN with different backends and core numbers...")
    backends = ['loky', 'threading']
    results, cores_range = optimized_parallel_knn_variants(
        X_train, Y_train, X_test, C_train, k, backends=backends)

    # Comparison with the original implementation
    print(f"\nThe original implementation took {original_time:.2f} seconds.")
    print(f"The optimized serial implementation took {
          optimized_serial_time:.2f} seconds.")

    # Visualization of the results
    plt.figure(figsize=(12, 6))

    for backend in backends:
        plt.plot(cores_range, results[backend], label=f'Backend: {backend}')

    plt.axhline(y=original_time, color='r', linestyle='--',
                label='The original implementation')
    plt.xlabel('Number of cores')
    plt.ylabel('Execution time (seconds)')
    plt.title(
        'Comparing the performance of different backends and the number of cores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Acceleration compared to the original implementation
    plt.figure(figsize=(12, 6))

    for backend in backends:
        speedup = original_time / np.array(results[backend])
        plt.plot(cores_range, speedup, label=f'Backend: {backend}')

    speedup_serial = original_time / optimized_serial_time
    plt.axhline(y=speedup_serial, color='g', linestyle='--',
                label='Accelerating consistent implementation')
    plt.xlabel('Number of cores')
    plt.ylabel('x-factor')
    plt.title('Acceleration of parallel implementations compared to the original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Output of acceleration results
    print("\nAcceleration results:")
    for backend in backends:
        for n, t in zip(cores_range, results[backend]):
            print(f"Backend: {backend}, Cores: {n}, Time: {
                  t:.2f} seconds, Acceleration: {original_time / t:.2f}x")


if __name__ == "__main__":
    main()
