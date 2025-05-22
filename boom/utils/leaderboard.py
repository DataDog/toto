import os
import json
import pandas as pd
import numpy as np

NON_ZERO_METRICS = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]
ZERO_METRICS = [
    "eval_metrics/MAE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]
LOW_VARIANCE_DATASETS = [
    "dd-1040-5T",
    "dd-462-5T",
    "dd-2801-D",
    "dd-947-T",
    "dd-2332-H",
    "dd-491-T",
    "dd-1901-D",
    "dd-1721-30T",
    "dd-2806-D",
    "dd-438-5T",
    "dd-372-10S",
    "dd-1719-30T",
    "dd-1902-D",
    "dd-111-T",
    "dd-2026-D",
    "dd-2089-H",
    "dd-299-5T",
    "dd-1596-T",
    "dd-4-5T",
    "dd-1723-H",
    "dd-953-10S",
    "dd-2394-D",
    "dd-1838-D",
    "dd-489-T",
    "dd-2802-D",
    "dd-1925-D",
    "dd-441-10S",
    "dd-1520-10S",
    "dd-2782-H",
    "dd-607-10S",
    "dd-177-5T",
    "dd-1643-30T",
    "dd-442-T",
    "dd-1909-D",
    "dd-2762-D",
    "dd-1135-5T",
    "dd-139-5T",
    "dd-805-10S",
    "dd-784-T",
    "dd-949-T",
    "dd-977-5T",
    "dd-1641-30T",
    "dd-303-5T",
    "dd-162-5T",
    "dd-608-T",
    "dd-551-T",
    "dd-2567-30T",
    "dd-1731-D",
    "dd-2206-H",
    "dd-206-5T",
    "dd-1718-H",
    "dd-1722-D",
    "dd-1195-10S",
    "dd-2514-H",
    "dd-1264-5T",
    "dd-1720-H",
    "dd-1564-10S",
    "dd-532-10S",
    "dd-300-5T",
    "dd-207-5T",
    "dd-1039-T",
    "dd-1733-D",
    "dd-2805-D",
    "dd-979-T",
    "dd-1619-H",
    "dd-1818-H",
    "dd-492-T",
    "dd-1894-H",
    "dd-2804-D",
    "dd-2012-30T",
    "dd-181-T",
    "dd-1642-30T",
    "dd-2027-H",
    "dd-890-10S",
    "dd-1768-30T",
    "dd-458-10S",
    "dd-1767-H",
    "dd-137-10S",
    "dd-2197-D",
    "dd-493-T",
]



def shifted_gmean(x, epsilon=1e-5, dim=-1):
    logsum = np.sum(np.log(x + epsilon))
    n = x.shape[dim]
    return np.exp(logsum / n) - epsilon

def load_and_process_csv(path, boomlet_benchmark):
    df = pd.read_csv(path)
    df["full_dataset_name"] = df["dataset"]
    df["dataset"] = df["dataset"].str.split("/").str[0]
    if boomlet_benchmark:
        light_benchmark_datasets = json.load(open("boomlet_properties.json")).keys()
        df = df[df["dataset"].isin(light_benchmark_datasets)]
    return df

def load_model_results(models_dir, boomlet_benchmark):
    model_names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    dfs = [load_and_process_csv(os.path.join(models_dir, m, "all_results.csv"), boomlet_benchmark) for m in model_names]

    assert "seasonalnaive" in model_names, "seasonalnaive model must be present in models directory"
    
    i = model_names.index("seasonalnaive")
    model_names.append(model_names.pop(i))
    dfs.append(dfs.pop(i))
    
    print(f'Number of models in leaderboard: {len(dfs)} \n')
    return dfs, model_names



def separate_zero_inflated_data(dfs):
    zero_datasets = dfs[-1][(dfs[-1]["eval_metrics/MASE[0.5]"] == 0)]["dataset"].unique()
    datasets_to_exclude = set(LOW_VARIANCE_DATASETS) | set(zero_datasets)
    print(f"Number of datasets to exclude: {len(datasets_to_exclude)}\n")
    
    non_zero_dfs = [df[~df["dataset"].isin(datasets_to_exclude)] for df in dfs]
    zero_dfs = [df[df["dataset"].isin(datasets_to_exclude)] for df in dfs]

    return non_zero_dfs, zero_dfs

def scale_by_naive(df, naive_df, metrics):
    assert set(df["full_dataset_name"]) == set(naive_df["full_dataset_name"]), "All datasets must be the same"

    merged = df.merge(naive_df, on="full_dataset_name", suffixes=("", "_naive"))
    for col in metrics:
        merged[col] = merged[col] / merged[f"{col}_naive"]
    return merged[df.columns]


def replace_invalid_values(dfs, metrics):
    cleaned_dfs = []

    for df in dfs:
        df = df.copy()
        df[metrics] = df[metrics].replace({np.inf: np.nan, -np.inf: np.nan})
        column_means = df[metrics].mean()
        df[metrics] = df[metrics].fillna(column_means)

        cleaned_dfs.append(df[metrics + ["full_dataset_name", "dataset"]])

    return cleaned_dfs

def process_dd_benchmark_model_results(is_scale_by_naive, dfs, metrics): 
    dfs = replace_invalid_values(dfs, metrics)
    if is_scale_by_naive:
        dfs = [scale_by_naive(df, dfs[-1], metrics) for df in dfs]
    return dfs


def format_number(num):
    # Check if the value is numeric
    if isinstance(num, (int, float)):
        if abs(num) >= 10**2:
            return f"{num:.1e}"
        else:
            return f"{num:.3f}"
    # Return non-numeric values as-is
    return num

def rename_metrics(df):
    df = df.rename(
        columns={
            "eval_metrics/MASE[0.5]": "MASE",
            "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
            "rank": "Rank",
        }
    )
    return df

def get_leaderboard(dfs, names, agg_func, metrics, ranking_metric="eval_metrics/mean_weighted_sum_quantile_loss"):

    for df, name in zip(dfs, names):
        df["model"] = name

    combined_df = pd.concat(dfs)
    combined_df["rank"] = combined_df.groupby("full_dataset_name")[ranking_metric].rank(method="first", ascending=True)
    aggregation_functions = {metric: agg_func for metric in metrics}
    aggregation_functions["rank"] = "mean"
    agg = combined_df[["model"] + metrics + ["rank"]].groupby("model").agg(aggregation_functions).reset_index()

    # Create and format the leaderboard
    leaderboard = agg.set_index("model").sort_values(by="rank", ascending=True).map(format_number)

    return rename_metrics(leaderboard)


def get_separate_zero_inflated_leaderboard(non_zero_dfs, zero_dfs, dfs_names, agg_func, non_zero_metrics, zero_metrics):

    non_zero_leaderboard = get_leaderboard(non_zero_dfs, dfs_names, agg_func, metrics=non_zero_metrics)
    zero_leaderboard = get_leaderboard(zero_dfs, dfs_names, agg_func, metrics=zero_metrics)

    non_zero_count = len(non_zero_dfs[0])
    zero_count = len(zero_dfs[0])

    non_zero_leaderboard.columns = [f"{col}-{non_zero_count}-scaled" for col in non_zero_leaderboard.columns]
    zero_leaderboard.columns = [f"{col}-{zero_count}-unscaled" for col in zero_leaderboard.columns]

    combined_leaderboard = pd.merge(non_zero_leaderboard, zero_leaderboard, on="model", suffixes=("_non_zero", "_zero"))
    return combined_leaderboard


