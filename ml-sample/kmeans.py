import os
import time
import argparse
import numpy as np
import faiss
from threadpoolctl import threadpool_info


def configure_thread_env(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    faiss.omp_set_num_threads(threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads",  type=int, default=4)
    parser.add_argument("--samples",  type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    configure_thread_env(args.threads)

    print("Generating dataset...")
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(
        rng.normal(size=(args.samples, args.features)).astype(np.float32)
    )

    print("Threadpool info:")
    for item in threadpool_info():
        print(" ", item)

    kmeans = faiss.Kmeans(
        args.features,      # d
        args.clusters,      # k
        niter=args.max_iter,
        nredo=1,
        verbose=False,
        gpu=False,
        seed=42,
    )

    t0 = time.perf_counter()
    kmeans.train(X)
    train_time = time.perf_counter() - t0

    print()
    print("========== RESULTS ==========")
    print(f"Train time  : {train_time:.2f}s")
    print(f"Samples     : {args.samples:,}")
    print(f"Features    : {args.features}")
    print(f"Clusters    : {args.clusters}")
    print(f"Iterations  : {len(kmeans.obj)}")
    print(f"Inertia     : {kmeans.obj[-1]:.4f}")


if __name__ == "__main__":
    main()
