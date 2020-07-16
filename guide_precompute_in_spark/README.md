# Guide to precompute `xai_data` in Spark
The guide `precompute_in_spark.ipynb` contains codes to precompute `xai_data` for the report generator. As this is compute-intensive, it is done in Spark outside of the report.

The codes do the following:
- compute correlation between prediction and all features
- compute and aggregate Shapley values for all features (and for all classes for multiclass classification)
- sample Shapley values across the data points
- save the output: `shap_summary_dfs` and `shap_sample_dfs`

The output are inputs to the report generator.


## Prerequisite
The model needs to be serializable in order to use Spark.


## Getting started
To get started, refer to `precompute_in_spark.ipynb`. It contains codes to precompute `shap_summary_dfs` and `shap_sample_dfs` for the cases:
- Regression
- Binary classification
- Multiclass classification

The user will need to amend accordingly the functions
- `udf_predict_score`: Spark UDF for predicting
  - scores for regression
  - probabilities for `target class = 1` for binary classification
  - probabilities for all classes for multiclass classification
- `get_shap_partition`: Function used in `mapPartitions` to compute Shapley values


## Precompute in Python
The user can also precompute `xai_data` in **non-Spark environment**. Please refer to `guides/precompute_in_python.ipynb`.

