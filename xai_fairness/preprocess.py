import numpy as np
import pandas as pd


def get_corrs_python(predictions, df, features):
    """Compute correlation between prediction and features for each class (if any)."""
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)

    all_corrs = [
        [
            pd.DataFrame({"x": df[c].values, "y": predictions[:, i]})
            .corr(method="pearson")
            .values[0, 1]
            for c in features
        ] for i in range(predictions.shape[1])
    ]
    return all_corrs


def get_mashap_python(all_shap_values, df, features):
    """Compute mean(|SHAP|) for each feature and for each class (if any)."""
    all_mashap = [
        np.abs(shap_values).mean(axis=0)
        for shap_values in all_shap_values
    ]

    shap_df = [df[features]] + all_shap_values
    return all_mashap, shap_df


def get_shap_summary_dfs(all_corrs, all_mashap, features):
    shap_summary_dfs = [
        pd.DataFrame({
            "feature": features,
            "mas_value": mashap,
            "corrcoef": corrs,
        })
        for mashap, corrs in zip(all_mashap, all_corrs)
    ]
    return shap_summary_dfs


def sample_shap_df_python(shap_df, num_rows=3000, seed=42):
    """Sample shap_df."""
    if num_rows >= shap_df[0].shape[0]:
        return shap_df

    idx = np.random.choice(np.arange(shap_df[0].shape[0]), num_rows, replace=False)

    sample_dfs = [shap_df[0].iloc[idx].copy().reset_index(drop=True)]  # dataframe of features

    for shap_values in shap_df[1:]:
        sample_dfs.append(shap_values[idx])  # arrays of shap
    return sample_dfs
