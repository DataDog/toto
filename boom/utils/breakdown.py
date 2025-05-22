from utils.leaderboard import shifted_gmean
import pandas as pd
import numpy as np

METRIC_NAMES = {
    "eval_metrics/MASE[0.5]": "MASE",
    "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
}

def add_agg_columns(dfs, agg_columns, prop_map):
    for df in dfs:
        for col in agg_columns:
            df[col] = df['dataset'].map(lambda x: prop_map.get(x, {}).get(col, np.nan))
        df['full_benchmark'] = 'full_benchmark'
        df['real_term'] = df['full_dataset_name'].str.split('/').str[2]
    return dfs


def aggregate_df(dfs, signatures, agg_column, eval_columns, agg_method='mean', weight_col='num_variates'):
    """Aggregate each DataFrame using the specified method and return (agg_df, signature) pairs."""
    results = []
    for df, sig in zip(dfs, signatures):
        if agg_method == 'shifted_gmean':
            agg = df.groupby(agg_column)[eval_columns].apply(
                lambda group: pd.Series({
                    col: shifted_gmean(group[col].values)
                    for col in eval_columns
                })
            )
        elif agg_method == 'median':
            agg = df.groupby(agg_column)[eval_columns].median()
        elif agg_method == 'weighted_mean':
            agg = df.groupby(agg_column).apply(
                lambda group: (group[eval_columns].T * group[weight_col]).sum(axis=1) / group[weight_col].sum()
            )
        else:  # default to mean
            agg = df.groupby(agg_column)[eval_columns].mean()
        results.append((agg, sig))
    return results


def get_breakdown_table(dfs, signatures, agg_column, eval_columns, agg_method='mean', weight_col='num_variates'):
    """Generate pivot tables for evaluation metrics by model and aggregation column."""
    aggregated = aggregate_df(dfs, signatures, agg_column, eval_columns, agg_method, weight_col)

    tables = {}
    for eval_col in eval_columns:
        records = [
            (idx, model, df.loc[idx, eval_col])
            for df, model in aggregated
            if eval_col in df.columns
            for idx in df.index
        ]
        if not records:
            continue

        df_table = pd.DataFrame(records, columns=[agg_column, "model", eval_col])
        tables[eval_col] = df_table.pivot(index="model", columns=agg_column, values=eval_col)

    return tables


    
def expand_complex_column(dfs, agg_column=''):
    """
    Expands nested list columns (like 'variates_types' or 'domain') into multiple rows,
    assigning counts where applicable.
    """
    expanded_dfs = []

    for df in dfs:
        new_rows = []
        for _, row in df.iterrows():
            values = row[agg_column]

            if agg_column == 'variates_types':
                # Flatten nested lists
                flattened = [item for sublist in values for item in sublist]
                unique_elements, counts = np.unique(flattened, return_counts=True)

                for val, count in zip(unique_elements, counts):
                    new_row = row.copy()
                    new_row[agg_column] = val
                    new_row['num_variates'] = count
                    new_rows.append(new_row)

            else:
                # Assume simple list column (e.g., 'domain')
                for val in values:
                    new_row = row.copy()
                    new_row[agg_column] = val
                    new_rows.append(new_row)

        expanded_dfs.append(pd.DataFrame(new_rows).reset_index(drop=True))

    return expanded_dfs
