import gzip
import pickle
import numpy as np
from collections import Counter
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing


def load_data():
    """Load training and testing data from pickle files"""
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
    """Compute the length of gzip compressed numpy array"""
    return len(gzip.compress(np.array(x).tobytes()))


def compute_ncd(x1, Cx1, x2, Cx2):
    """Compute the Normalized Compression Distance (NCD) between two sequences"""
    x1x2 = x1 + x2
    Cx1x2 = compute_compressed_length(x1x2)
    return (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)


def original_knn(X_train, Y_train, X_test, k=101):
    """Original k-NN implementation with nested loops"""
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
    """Optimized serial kNN implementation using numpy"""
    predicts = []
    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    for x1 in X_test:
        Cx1 = compute_compressed_length(x1)
        # Compute C(x1 + x2) for all x2 in X_train
        # This can be parallelized or optimized further if possible
        Cx1x2 = np.array([compute_compressed_length(x1 + x2) for x2 in X_train_np])
        # Compute NCD
        ncd = (Cx1x2 - np.minimum(Cx1, C_train)) / np.maximum(Cx1, C_train)
        # Get top k indices
        sorted_idx = np.argsort(ncd)[:k]
        top_k_class = Y_train_np[sorted_idx]
        # Majority vote
        predict_class = Counter(top_k_class).most_common(1)[0][0]
        predicts.append(predict_class)
    return predicts


def optimized_parallel_knn(
    X_train, Y_train, X_test, C_train, k=101, n_jobs=-1, backend="loky"
):
    """Optimized parallel kNN implementation using Joblib"""
    Y_train_np = np.array(Y_train)

    def predict(x1):
        Cx1 = compute_compressed_length(x1)
        Cx1x2 = np.array([compute_compressed_length(x1 + x2) for x2 in X_train])
        ncd = (Cx1x2 - np.minimum(Cx1, C_train)) / np.maximum(Cx1, C_train)
        sorted_idx = np.argsort(ncd)[:k]
        top_k_class = Y_train_np[sorted_idx]
        return Counter(top_k_class).most_common(1)[0][0]

    predicts = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(predict)(x1) for x1 in X_test
    )
    return predicts


def main():
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Training samples: {len(Y_train)}, Testing samples: {len(Y_test)}")

    k = 101

    # Precompute C(x_train)
    print("Precomputing compressed lengths for training data...")
    start_time = time.time()
    C_train = np.array([compute_compressed_length(x2) for x2 in X_train])
    print(f"Precomputation done in {time.time() - start_time:.2f} seconds.\n")

    # Original implementation
    print("Running original k-NN implementation...")
    start_time = time.time()
    original_preds = original_knn(X_train, Y_train, X_test, k)
    original_time = time.time() - start_time
    print(f"Original implementation took {original_time:.2f} seconds.\n")

    # Optimized serial implementation
    print("Running optimized serial k-NN implementation...")
    start_time = time.time()

    optimized_serial_preds = optimized_serial_knn(X_train, Y_train, X_test, C_train, k)
    optimized_serial_time = time.time() - start_time
    print(f"Optimized serial implementation took {optimized_serial_time:.2f} seconds.")
    speedup_serial = original_time / optimized_serial_time
    print(f"Speedup (Serial): {speedup_serial:.2f}x\n")

    # Optimized parallel implementation
    num_cores = multiprocessing.cpu_count()
    print(f"Running optimized parallel k-NN implementation with {num_cores} cores...")
    start_time = time.time()
    optimized_parallel_preds = optimized_parallel_knn(
        X_train, Y_train, X_test, C_train, k, n_jobs=-1
    )
    optimized_parallel_time = time.time() - start_time
    print(
        f"Optimized parallel implementation took {optimized_parallel_time:.2f} seconds."
    )
    speedup_parallel = original_time / optimized_parallel_time
    print(f"Speedup (Parallel): {speedup_parallel:.2f}x\n")

    # Validate predictions (optional)
    from sklearn.metrics import accuracy_score
    print("Original Accuracy:", accuracy_score(Y_test, original_preds))
    print("Optimized Serial Accuracy:", accuracy_score(Y_test, optimized_serial_preds))
    print("Optimized Parallel Accuracy:", accuracy_score(Y_test, optimized_parallel_preds))

    # Plotting the performance
    methods = ["Original", "Optimized Serial", "Optimized Parallel"]
    times = [original_time, optimized_serial_time, optimized_parallel_time]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=["blue", "green", "red"])
    plt.ylabel("Time (seconds)")
    plt.title("k-NN Implementation Performance Comparison")
    for bar, time_taken in zip(bars, times):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.05,
            f"{time_taken:.2f}s",
            ha="center",
            va="bottom",
        )
    plt.show()

    # Speedup Bar Chart
    speedup_methods = ["Serial vs Original", "Parallel vs Original"]
    speedups = [speedup_serial, speedup_parallel]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(speedup_methods, speedups, color=["purple", "orange"])
    plt.ylabel("Speedup Factor")
    plt.title("Speedup Comparison")
    for bar, speedup in zip(bars, speedups):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.05,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
        )
    plt.show()


if __name__ == "__main__":
    main()
