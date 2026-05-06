import argparse
import time
import polars as pl


def base_pipeline(data_dir: str) -> pl.LazyFrame:
    orders    = pl.scan_parquet(f"{data_dir}/orders.parquet")
    products  = pl.scan_parquet(f"{data_dir}/products.parquet")
    customers = pl.scan_parquet(f"{data_dir}/customers.parquet")

    return (
        orders
        .with_columns([
            (pl.col("unit_price") * pl.col("quantity")
             * (1.0 - pl.col("discount_pct"))).alias("net_revenue"),
            (pl.col("unit_price") * pl.col("quantity")
             * (1.0 - pl.col("discount_pct"))
             * pl.col("tax_rate")).alias("tax_amount"),
        ])
        .with_columns([
            (pl.col("net_revenue") + pl.col("tax_amount")
             + pl.col("shipping_cost")).alias("total_order_value"),
            pl.when((pl.col("month") >= 10) | (pl.col("month") == 1))
              .then(pl.lit(1)).otherwise(pl.lit(0))
              .cast(pl.Int8).alias("is_festive_season"),
            pl.when(pl.col("day_of_week") >= 5)
              .then(pl.lit(1)).otherwise(pl.lit(0))
              .cast(pl.Int8).alias("is_weekend"),
        ])
        .join(
            products.select([
                "product_id",
                pl.col("base_cost").cast(pl.Float32),
                pl.col("rating").cast(pl.Float32),
                pl.col("review_count").cast(pl.Float32),
            ]),
            on="product_id", how="left",
        )
        .join(
            customers.select([
                "customer_id",
                pl.col("account_age_days").cast(pl.Float32),
                pl.col("credit_score").cast(pl.Float32),
                pl.col("is_prime").cast(pl.Float32),
            ]),
            on="customer_id", how="left",
        )
        .with_columns([
            (pl.col("net_revenue")
             - pl.col("base_cost") * pl.col("quantity")).alias("gross_margin"),
            (pl.col("discount_pct") * 0.6
             + pl.col("is_returned").cast(pl.Float32) * 0.4).alias("seller_risk"),
            (pl.col("credit_score") / 900.0
             * pl.col("account_age_days").log1p()
             * (pl.col("is_prime") + 1.0)).alias("customer_value"),
            (pl.col("rating")
             * (pl.col("review_count") + 1.0).log()).alias("product_appeal"),
        ])

        # ── log base ─────────────────────────────────────────────────
        .with_columns([
            (pl.col("total_order_value").abs() + 1.0).log().alias("log_rev"),
            (pl.col("gross_margin").abs()       + 1.0).log().alias("log_margin"),
            (pl.col("customer_value").abs()     + 1.0).log().alias("log_cust"),
            (pl.col("product_appeal").abs()     + 1.0).log().alias("log_appeal"),
            (pl.col("seller_risk").abs()        + 1.0).log().alias("log_risk"),
            (pl.col("shipping_cost").abs()      + 1.0).log().alias("log_ship"),
        ])

        # ── polynomial cross terms ────────────────────────────────────
        .with_columns([
            (pl.col("log_rev")  * pl.col("log_margin")).alias("c_rev_margin"),
            (pl.col("log_cust") * pl.col("log_appeal")).alias("c_cust_appeal"),
            (pl.col("log_risk") * pl.col("log_ship")).alias("c_risk_ship"),
            (pl.col("log_rev").pow(2) - pl.col("log_margin").pow(2)).alias("diff2"),
            (pl.col("log_cust").pow(2) + pl.col("log_appeal").pow(2)).alias("sum2"),
            (pl.col("log_rev") * pl.col("log_cust") * pl.col("log_risk")).alias("triple"),
        ])

        # ── layer 1 ───────────────────────────────────────────────────
        .with_columns([
            (pl.col("c_rev_margin")  * 1.3).sin().alias("s1"),
            (pl.col("c_cust_appeal") * 0.7).cos().alias("s2"),
            (pl.col("c_risk_ship")   / 5.0).tanh().alias("s3"),
            (pl.col("diff2").abs()   + 1.0).sqrt().alias("s4"),
            (pl.col("sum2").abs()    + 1.0).log().alias("s5"),
            (pl.col("triple")        * 2.1).sin().alias("s6"),
        ])
        .with_columns([
            (pl.col("s1") + pl.col("s2") + pl.col("s3")
           + pl.col("s4") + pl.col("s5") + pl.col("s6")).alias("L1"),
        ])

        # ── layer 2 ───────────────────────────────────────────────────
        .with_columns([
            (pl.col("L1").abs() + 1.0).log().alias("t2_a"),
            (pl.col("L1") * 2.3).sin().alias("t2_b"),
            (pl.col("L1") * 1.7).cos().alias("t2_c"),
            (pl.col("L1") / 4.1).tanh().alias("t2_d"),
            (pl.col("L1").pow(2) * pl.col("log_risk")).alias("t2_e"),
            (pl.col("L1") * pl.col("log_rev") - pl.col("log_margin")).alias("t2_f"),
        ])
        .with_columns([
            (pl.col("t2_a") + pl.col("t2_b") + pl.col("t2_c")
           + pl.col("t2_d") + pl.col("t2_e").abs().log1p()
           + pl.col("t2_f").abs().log1p()).alias("L2"),
        ])

        # ── layer 3 ───────────────────────────────────────────────────
        .with_columns([
            (pl.col("L2") * 3.1).sin().alias("t3_a"),
            (pl.col("L2") / 2.7).cos().alias("t3_b"),
            (pl.col("L2").abs() + 1.0).sqrt().alias("t3_c"),
            (pl.col("L2") * pl.col("L1")).tanh().alias("t3_d"),
            ((pl.col("L2") - pl.col("L1")).abs() + 1.0).log().alias("t3_e"),
        ])
        .with_columns([
            (pl.col("t3_a") + pl.col("t3_b") + pl.col("t3_c")
           + pl.col("t3_d") + pl.col("t3_e")).alias("L3"),
        ])

        # ── composite ─────────────────────────────────────────────────
        .with_columns([
            (
                pl.col("L3")   * 0.35
                + pl.col("L2") * 0.25
                + pl.col("L1") * 0.15
                + pl.col("customer_value").abs().log1p() * 0.15
                + (pl.col("gross_margin")
                   / (pl.col("total_order_value").abs() + 1.0)) * 0.10
            ).alias("composite_score"),
        ])
        .with_columns([
            (pl.col("composite_score")
             * pl.col("customer_value").abs().log1p()).alias("score_x_cust"),
            (pl.col("composite_score")
             * pl.col("product_appeal").abs().log1p()).alias("score_x_appeal"),
            (pl.col("composite_score")
             / (pl.col("seller_risk").abs() + 0.01)).alias("score_per_risk"),
        ])
        .with_columns([
            (pl.col("score_x_cust").abs()   + 1.0).log().alias("lsc"),
            (pl.col("score_x_appeal") * 1.5).sin().alias("ssa"),
            (pl.col("score_per_risk") / 3.0).tanh().alias("spr"),
        ])
        .with_columns([
            (pl.col("lsc") + pl.col("ssa") + pl.col("spr")
             + pl.col("composite_score")).alias("final_score"),
        ])
    )


def run_benchmark(data_dir: str) -> None:
    phases = {}

    # ── agg 1: category × region ──────────────────────────────────────
    # n_unique removed — not supported in streaming
    t = time.perf_counter()
    print("Agg 1 — category × region...")
    cat_region = (
        base_pipeline(data_dir)
        .group_by(["category", "region"])
        .agg([
            pl.col("total_order_value").sum().alias("total_revenue"),
            pl.col("gross_margin").sum().alias("total_margin"),
            pl.col("final_score").mean().alias("avg_score"),
            pl.col("final_score").std().alias("std_score"),
            pl.col("is_returned").cast(pl.Float32).mean().alias("return_rate"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("seller_risk").mean().alias("avg_seller_risk"),
            pl.col("discount_pct").mean().alias("avg_discount"),
            pl.col("total_order_value").max().alias("max_order_value"),
        ])
        .sort(["category", "region"])
        .collect(engine="streaming")
    )
    phases["agg1_cat_region"] = time.perf_counter() - t
    print(f"  {phases['agg1_cat_region']:.2f}s")

    # ── agg 2: seller performance ─────────────────────────────────────
    t = time.perf_counter()
    print("Agg 2 — seller performance...")
    seller_perf = (
        base_pipeline(data_dir)
        .group_by("seller_id")
        .agg([
            pl.col("gross_margin").sum().alias("total_margin"),
            pl.col("total_order_value").sum().alias("total_gmv"),
            pl.col("final_score").mean().alias("avg_score"),
            pl.col("is_returned").cast(pl.Float32).mean().alias("return_rate"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("seller_risk").mean().alias("avg_risk"),
            pl.col("gross_margin").std().alias("margin_volatility"),
        ])
        .with_columns([
            (
                pl.col("avg_score")     * 0.5
                - pl.col("return_rate") * 0.3
                - pl.col("avg_risk")    * 0.2
            ).alias("seller_rank_score"),
        ])
        .sort("seller_rank_score", descending=True)
        .collect(engine="streaming")
    )
    phases["agg2_seller"] = time.perf_counter() - t
    print(f"  {phases['agg2_seller']:.2f}s")

    # ── agg 3: customer cohort ────────────────────────────────────────
    t = time.perf_counter()
    print("Agg 3 — customer cohort...")
    customer_cohort = (
        base_pipeline(data_dir)
        .group_by("customer_id")
        .agg([
            pl.col("total_order_value").sum().alias("lifetime_value"),
            pl.col("final_score").mean().alias("avg_score"),
            pl.col("gross_margin").sum().alias("lifetime_margin"),
            pl.col("is_returned").cast(pl.Float32).mean().alias("return_rate"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("total_order_value").std().alias("order_value_std"),
        ])
        .with_columns([
            (
                pl.col("lifetime_value").log1p() * 0.5
                + pl.col("avg_score")            * 0.3
                - pl.col("return_rate")          * 0.2
            ).alias("ltv_score"),
        ])
        .sort("ltv_score", descending=True)
        .collect(engine="streaming")
    )
    phases["agg3_cohort"] = time.perf_counter() - t
    print(f"  {phases['agg3_cohort']:.2f}s")

    total = sum(phases.values())

    print()
    print("========== RESULTS ==========")
    for name, t in phases.items():
        pct = t / total * 100
        print(f"  {name:<22} {t:>6.2f}s  ({pct:.0f}%)")
    print(f"  {'TOTAL':<22} {total:>6.2f}s")
    print()
    print("--- category × region ---")
    print(cat_region.head(10))
    print()
    print("--- seller performance (top 5) ---")
    print(seller_perf.head(5))
    print()
    print("--- customer cohort (top 5 LTV) ---")
    print(customer_cohort.head(5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ecommerce_data")
    args = parser.parse_args()
    run_benchmark(args.data_dir)


if __name__ == "__main__":
    main()
