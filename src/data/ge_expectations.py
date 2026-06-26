import great_expectations as ge
import pandas as pd


def validate_with_ge(df: pd.DataFrame) -> dict:
    ge_df = ge.from_pandas(df)
    results = ge_df.expect_table_columns_to_match_ordered_list(
        column_list=[
            "timestamp",
            "cpu_pct",
            "mem_pct",
            "disk_io",
            "latency_ms",
            "error_rate",
            "is_anomaly",
        ]
    )
    results &= ge_df.expect_column_values_to_not_be_null("timestamp")
    results &= ge_df.expect_column_values_to_be_between("cpu_pct", 0, 100)
    results &= ge_df.expect_column_values_to_be_between("mem_pct", 0, 100)
    results &= ge_df.expect_column_values_to_be_of_type("timestamp", "datetime64[ns]")
    results &= ge_df.expect_column_values_to_be_between("disk_io", 0, None)
    results &= ge_df.expect_column_values_to_be_between("latency_ms", 0, None)
    results &= ge_df.expect_column_values_to_be_between("error_rate", 0, None)
    results &= ge_df.expect_column_unique_values_to_be_within_set("is_anomaly", [0, 1])

    return {"success": results.success, "results": results.to_json_dict()}
