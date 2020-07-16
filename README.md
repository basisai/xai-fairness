# xai-fairness
Toolkit for model explainability and fairness

## Getting started
There are 4 files that require user inputs.
- `cover_page.png`
- `xai_fairness.yaml`
- `constants.py`
- `report_utils.py`

To get started, refer to `sample_template`. Refer to `eg_*` for examples on
- binary classification
- regression
- multilayer model
- multiclass classification
- pyspark.ml.RandomForestClassification


### `xai_fairness.yaml`
- This file will provide the user write-ups to the report as well as the parameters.
- For instance, the report requires the user to give a write-up of the model description.
- All the fields are required in the report.

### `constants.py`
- This file will provide the **exact** names of 
  - feature columns: `FEATURES`
  - target column: `TARGET`
  - fairness configuration: `CONFIG_FAI`

### `report_utils.py`
- This file is the entry point for the report.
- The report uses the class `ReportPreparer`, which has 5 main methods:
  - `model_artefact`: This is used to set the model/predict function.
  - `xai_data`: This is to add the data required for model explainability.
  - `indiv_xai_data`: This is add the data required for sample individual explainability.
  - `fai_data`: This is used to compute fairness metrics.
  - `model_perf_to_print`: This is used to print the model performance in the format that the user wishes to show.

- Data should be in `pandas.DataFrame` for features, and array-like for labels.
  1. The data for `xai_data` must be in the same format as the training data. It will be used to compute Shapley values. Thus, this data should be a good representation of the data in order not to bias the interpretation of the model. It is also used for plotting purposes, and so it suffices to be a sample of 1000-10000s. **The user can also use `guide_precompute_in_spark` to precompute `xai_data` in Spark. In this case, the user only needs to input the data generated from the codes in the guide.**
  2. The data for `indiv_xai_data` should be a dict of (label, individual data points in `pandas.DataFrame` in the same format as the training data).
  3. The data required by `fai_data` consists of three parts:
    - a `pandas.DataFrame` of the sensitive attribute columns, which need not be part of the training data. They are only used to split the populations into privileged and unprivileged groups.
    - array-like of true classes
    - array-like of predicted classes

- For multilayer models, the user will have to set `predict_func` instead of `model`. In this case, background data will also be required.
  - This is the input to the KernelExplainer.
  - This is used for integrating out features.
  - To determine the impact of a feature, that feature is set to “missing” and the change in the model output is observed.
  - To handle arbitrary missing data at test time, it is simulated by replacing the feature with the values it takes in the background dataset.
  - For larger problems, kmeans is used to summarize the dataset.
  - Its runtime scales linearly with the size of the background dataset you use.
  - 100 to 1000 random background samples are good sizes to use.


### Generate the report
To generate the report, run the following in the command line:

`streamlit run report_utils.py`

You can view the report in your browser.

To save the report as a PDF, just print the report as PDF.


## Non-exhaustive list of prohibited features
- religion
- nationality
- birth_place
- gender
- race
- education
- neighbourhood
- country/region


## Guides
Include miscellaneous guides
- To compute Shapley values at scale in PySpark: `scaling_shap.ipynb`
- To convert trained `pyspark.ml.classification.LinearRegression` model to `sklearn.linear_model.LinearRegression` model: `convert_spark_model_python.ipynb`

