import argparse
import time
from pathlib import Path
import numpy as np
import polars as pl


def generate(output_dir: str, rows: int, chunk_size: int, seed: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    categories    = ["electronics", "clothing", "home", "sports",
                     "books", "toys", "beauty", "automotive"]
    regions       = ["north", "south", "east", "west", "central"]
    payment_types = ["credit_card", "debit_card", "upi", "netbanking", "wallet"]
    tax_rates     = np.array([0.05, 0.12, 0.18, 0.28], dtype=np.float32)

    rng = np.random.default_rng(seed)
    writer = None
    total_written = 0
    t0 = time.perf_counter()

    orders_path = out / "orders.parquet"

    while total_written < rows:
        n = min(chunk_size, rows - total_written)

        chunk = pl.DataFrame({
            "order_id":     np.arange(total_written, total_written + n, dtype=np.int32),
            "customer_id":  rng.integers(0, 500_000, size=n, dtype=np.int32),
            "product_id":   rng.integers(0, 10_000,  size=n, dtype=np.int32),
            "seller_id":    rng.integers(0, 2_000,   size=n, dtype=np.int32),
            "category":     [categories[i] for i in rng.integers(0, len(categories), n)],
            "region":       [regions[i]    for i in rng.integers(0, len(regions),    n)],
            "payment_type": [payment_types[i] for i in rng.integers(0, len(payment_types), n)],
            "quantity":     rng.integers(1, 20,   size=n, dtype=np.int32),
            "unit_price":   rng.uniform(50.0, 15_000.0, size=n).astype(np.float32),
            "discount_pct": rng.uniform(0.0,  0.45,     size=n).astype(np.float32),
            "tax_rate":     rng.choice(tax_rates, size=n).astype(np.float32),
            "shipping_cost":rng.uniform(0.0, 500.0,     size=n).astype(np.float32),
            "is_returned":  rng.random(size=n).astype(np.float32).__lt__(0.08).astype(np.int8),
            "day_of_week":  rng.integers(0, 7,  size=n, dtype=np.int8),
            "hour_of_day":  rng.integers(0, 24, size=n, dtype=np.int8),
            "month":        rng.integers(1, 13, size=n, dtype=np.int8),
        })

        if writer is None:
            writer = chunk.write_parquet(
                orders_path,
                use_pyarrow=True,
                row_group_size=500_000,
            )
            # re-open as append after first write
            import pyarrow.parquet as pq
            import pyarrow as pa
            schema = chunk.to_arrow().schema
            writer = pq.ParquetWriter(str(orders_path), schema, compression="snappy")

        writer.write_table(chunk.to_arrow())
        total_written += n

        elapsed = time.perf_counter() - t0
        print(f"  {total_written:>12,} / {rows:,}  "
              f"({total_written/rows:.0%})  {elapsed:.1f}s")

    if writer:
        writer.close()

    # products and customers are small — generate in one shot
    print("Generating products...")
    rng2 = np.random.default_rng(seed + 1)
    n_prod = 10_000
    pl.DataFrame({
        "product_id":   np.arange(n_prod, dtype=np.int32),
        "base_cost":    rng2.uniform(20.0, 10_000.0, n_prod).astype(np.float32),
        "rating":       rng2.uniform(1.0, 5.0,       n_prod).astype(np.float32),
        "review_count": rng2.integers(0, 50_000,     n_prod, dtype=np.int32),
        "weight_kg":    rng2.uniform(0.1, 30.0,      n_prod).astype(np.float32),
        "is_perishable":rng2.random(n_prod).__lt__(0.1).astype(np.int8),
    }).write_parquet(out / "products.parquet")

    print("Generating customers...")
    rng3 = np.random.default_rng(seed + 2)
    n_cust = 500_000
    pl.DataFrame({
        "customer_id":      np.arange(n_cust, dtype=np.int32),
        "age":              rng3.integers(18, 75,   n_cust, dtype=np.int8),
        "account_age_days": rng3.integers(1, 3650,  n_cust, dtype=np.int32).astype(np.float32),
        "credit_score":     rng3.integers(300, 900, n_cust, dtype=np.int32).astype(np.float32),
        "is_prime":         rng3.random(n_cust).__lt__(0.35).astype(np.float32),
        "lifetime_orders":  rng3.integers(1, 500,   n_cust, dtype=np.int32),
    }).write_parquet(out / "customers.parquet")

    size_gb = orders_path.stat().st_size / 1e9
    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.1f}s")
    print(f"orders.parquet: {size_gb:.2f} GB  |  {rows:,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",  default="ecommerce_data")
    parser.add_argument("--rows",        type=int, default=50_000_000)
    parser.add_argument("--chunk-size",  type=int, default=2_000_000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.rows, args.chunk_size, args.seed)


if __name__ == "__main__":
    main()