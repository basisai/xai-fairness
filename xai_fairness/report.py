"""
Report template
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st

from .static_xai import (
    model_xai_summary,
    model_xai_appendix,
    indiv_xai_appendix,
)
from .static_fai import (
    alg_fai_summary,
    alg_fai_appendix,
    color_red,
)
from .toolkit import compute_shap_values, compute_corrcoef
from util import page_break, add_header


class ReportPreparer:
    def __init__(self, config, config_fai, feature_names):
        """
        Configurations for the report.

        Args:
            config: Report configuration
            config_fai: Fairness configuration
            feature_names list[str]: list of feature names
        """
        self.config = config
        self.config_fai = config_fai
        self.feature_names = feature_names
        self.model_type = None
        self.model = None
        self.predict_func = None
        self.bkgrd_data = None
        self.x_valid = None
        self.shap_summary_dfs = None
        self.shap_sample_dfs = None
        self.indiv_samples = None
        self.text_model_perf = None
        self.valid_fai = None
        self.true_class = None
        self.pred_class = None
        self.max_rows = 3000

    def model_artefact(self, model_type=None, model=None, predict_func=None, bkgrd_data=None):
        """
        Model artefact required for XAI.
        Either (model_type, model) or predict_func must be provided.
        bkgrd_data is required, especially with predict_func.

        Args:
            model_type Optional[str]: 'tree' or 'linear'
            model Optional
            predict_func Optional[Callable]: predict function
            bkgrd_data Optional[pd.DataFrame]: background data
        """
        self.model_type = model_type
        self.model = model
        self.predict_func = predict_func
        self.bkgrd_data = bkgrd_data
        return self

    def xai_data(self, valid_features=None, shap_summary_dfs=None, shap_sample_dfs=None):
        """
        Sample validation data for generating XAI plots only.
        Data will be limited by sampling 3000 rows randomly.

        Args:
            valid_features pd.dataFrame: sample validation data
            shap_summary_dfs list(pd.dataFrame): list of dataframes containing
                average absolute SHAP values and correlations
        """
        self.x_valid = sampling(valid_features, max_rows=self.max_rows)
        self.shap_summary_dfs = shap_summary_dfs
        self.shap_sample_dfs = shap_sample_dfs
        return self

    def indiv_xai_data(self, indiv_samples):
        """
        Dict of individual data points for generating individual XAI plots.

        Args:
            indiv_samples dict[str: pd.DataFrame]
        """
        self.indiv_samples = indiv_samples
        return self

    def fai_data(self, valid_fai, true_class, pred_class):
        """
        To be used for fairness
        The classes need not correspond to the original target variable.
        Eg, for regression, one can bin the scores to classes for computing fairness metrics.

        Args:
            valid_fai pd.DataFrame: validation data containing sensitive attribute columns
            indicated in CONFIG_FAI
            true_class numpy.array: actual labels
            pred_class numpy.array: labels from predicted scores
        """
        self.valid_fai = valid_fai
        self.true_class = true_class
        self.pred_class = pred_class
        return self

    def model_perf_to_print(self, text_model_perf):
        """
        Used for printing model performance in report.

        Args:
            text_model_perf str
        """
        self.text_model_perf = text_model_perf
        return self

    def check_data(self):
        if self.x_valid is None and (self.shap_summary_dfs is None or self.shap_sample_dfs is None):
            raise AttributeError("'xai_data' is missing either shap_summary_dfs or shap_sample_dfs.")

        if self.indiv_samples is None:
            raise AttributeError("Dict of representative samples for individual XAI is missing")

        if (self.model is None and self.predict_func is None) or \
                (self.model is not None and self.predict_func is not None):
            raise AttributeError("Use either 'model' or 'predict_func'. If model_type "
                                 "is 'tree' or 'linear',set 'model'. Else set 'predict_func'.")

        if self.model is not None and self.model_type is None:
            raise AttributeError("'model_type' is missing. Only 'tree' and 'linear' are supported. "
                                 "Otherwise, use 'predict_func'")

        if self.predict_func is not None and self.bkgrd_data is None:
            raise AttributeError("'bkgrd_features' is missing. Required for use with 'predict_func'")

        if self.valid_fai is None:
            raise AttributeError("Data for fairness is missing")

        if self.text_model_perf is None:
            raise AttributeError("Model performance to print is missing")

    def generate_report(self):
        generate_report(self)

    def is_spark_model(self):
        """Check whether model is pyspark.ml"""
        if self.model is not None:
            return "pyspark.ml" in str(type(self.model))
        return False


@st.cache(allow_output_mutation=True)
def get_shap_values(
    x,
    model=None,
    model_type=None,
    predict_func=None,
    bkgrd_data=None,
    kmeans_size=10,
):
    return compute_shap_values(
        x,
        model=model,
        model_type=model_type,
        predict_func=predict_func,
        bkgrd_data=bkgrd_data,
        kmeans_size=kmeans_size,
    )


def get_shap_values_spark(
    x,
    model=None,
    model_type=None,
    predict_func=None,
    bkgrd_data=None,
    kmeans_size=10,
):
    return compute_shap_values(
        x,
        model=model,
        model_type=model_type,
        predict_func=predict_func,
        bkgrd_data=bkgrd_data,
        kmeans_size=kmeans_size,
    )


def sampling(df, max_rows=3000, random_state=0):
    """Select first 'max_rows' rows for plotting purposes."""
    if df is not None and df.shape[0] > max_rows:
        return df.sample(max_rows, random_state=random_state)
    return df


def get_repr_samples(x, y_class):
    """Select a representative sample from each class for individual XAI."""
    indiv_samples = {}
    for c in set(y_class):
        row = next(i for i in range(len(y_class)) if y_class[i] == c)
        indiv_samples[str(c)] = x.iloc[row: row + 1]
    return indiv_samples


def print_user_input(user_input):
    # st.markdown(
    #     "<span style='background-color: yellow'>*[For user to complete. Below is a sample.]*</span>",
    #     unsafe_allow_html=True)
    st.markdown(
        "<span style='color: blue'>{}</span>".format(user_input),
        unsafe_allow_html=True)


def generate_report(report_inputs):
    """Generate report."""
    config = report_inputs.config
    config_fai = report_inputs.config_fai

    report_inputs.check_data()

    feature_names = report_inputs.feature_names

    indiv_samples = OrderedDict(sorted(report_inputs.indiv_samples.items(), key=lambda t: t[0]))
    x_indiv = pd.concat(list(indiv_samples.values()), ignore_index=True)
    if not report_inputs.is_spark_model():
        indiv_shap_values, indiv_base_values = get_shap_values(
            x_indiv,
            model=report_inputs.model,
            model_type=report_inputs.model_type,
            predict_func=report_inputs.predict_func,
            bkgrd_data=report_inputs.bkgrd_data,
        )
    else:
        indiv_shap_values, indiv_base_values = get_shap_values_spark(
            x_indiv,
            model=report_inputs.model,
            model_type=report_inputs.model_type,
            predict_func=report_inputs.predict_func,
            bkgrd_data=report_inputs.bkgrd_data,
        )

    # Flag if model is multiclass
    num_classes = len(indiv_base_values)
    is_multiclass = (num_classes > 2)

    x_valid = report_inputs.x_valid
    if x_valid is not None:
        if not report_inputs.is_spark_model():
            all_shap_values, _ = get_shap_values(
                x_valid,
                model=report_inputs.model,
                model_type=report_inputs.model_type,
                predict_func=report_inputs.predict_func,
                bkgrd_data=report_inputs.bkgrd_data,
            )
        else:
            all_shap_values, _ = get_shap_values_spark(
                x_valid,
                model=report_inputs.model,
                model_type=report_inputs.model_type,
                predict_func=report_inputs.predict_func,
                bkgrd_data=report_inputs.bkgrd_data,
            )
        all_corrs = compute_corrcoef(x_valid, all_shap_values)

        shap_summary_dfs = [
            pd.DataFrame({
                "feature": feature_names,
                "mas_value": np.abs(shap_values).mean(axis=0),
                "corrcoef": corrs,
            })
            for shap_values, corrs in zip(all_shap_values, all_corrs)
        ]

    else:
        shap_summary_dfs = report_inputs.shap_summary_dfs
        x_valid = report_inputs.shap_sample_dfs[0]
        all_shap_values = report_inputs.shap_sample_dfs[1:]

    valid_fai = report_inputs.valid_fai
    true_class = report_inputs.true_class
    pred_class = report_inputs.pred_class

    # Get unique fairness classes
    unq_fai_classes = np.unique(true_class)
    # If there are 2 classes, select the latter
    if len(unq_fai_classes) == 2:
        unq_fai_classes = unq_fai_classes[1:]

    cover_page_path = config["cover_page_path"] or "../report_style/dbs/assets/cover_full.png"
    add_header(cover_page_path)

    page_break()
    add_header("../report_style/dbs/assets/header.png")

    st.header("I. Model Description")
    print_user_input(config["model_description"])

    st.header("II. List of Prohibited Features")
    st.write("religion, nationality, birth place, gender, race")

    st.header("III. Algorithmic Fairness")
    final_fairness = alg_fai_summary(valid_fai, unq_fai_classes, true_class, pred_class,
                                     config_fai, config)

    page_break()
    add_header("../report_style/dbs/assets/header.png")

    st.header("IV. Model Explainability")
    top_features, dict_feats = model_xai_summary(
        shap_summary_dfs, all_shap_values, x_valid, feature_names, config, is_multiclass)
    print_user_input(config["explainability"])

    page_break()
    add_header("../report_style/dbs/assets/header.png")

    st.header("V. Model Performance")
    st.text(report_inputs.text_model_perf)

    st.header("VI. Conclusion")
    print_user_input("**Model performance**: {}".format(config["conclusion"]["model_performance"]))
    print_user_input("**Explainability**: {}".format(config["conclusion"]["explainability"]))

    if not is_multiclass and dict_feats is not None:
        st.write("The top features that have positive correlation with their model output are `"
                 + "`, `".join(dict_feats["pos"]) + "`.")
        st.write("The top features that have negative correlation with their model output are `"
                 + "`, `".join(dict_feats["neg"]) + "`.")

    fair = "fair" if np.mean(final_fairness["Fair?"] == "Yes") == 1 else "not fair"
    st.write("**Fairness**: We consider the model to be fair if it is deemed to be fair for "
             f"all metrics. From the table below, overall the model is considered **{fair}**.")
    st.dataframe(final_fairness.style.applymap(color_red, subset=["Fair?"]))

    page_break()
    add_header("../report_style/dbs/assets/header.png")

    st.title("Appendix")

    st.header("Dependence Plots of Top Features")
    model_xai_appendix(all_shap_values, x_valid, feature_names, top_features, is_multiclass)

    st.header("Sample Individual Explainability")
    indiv_xai_appendix(indiv_samples, indiv_shap_values, indiv_base_values,
                       config, is_multiclass)

    st.header("Algorithmic Fairness")
    alg_fai_appendix(valid_fai, unq_fai_classes, true_class, pred_class, config_fai, config)

    st.header("Notes")
    st.write("**Equal opportunity**:")
    st.latex(r"\frac{\text{FNR}(D=\text{unprivileged})}{\text{FNR}(D=\text{privileged})}")
    st.write("**Predictive parity**:")
    st.latex(r"\frac{\text{PPV}(D=\text{unprivileged})}{\text{PPV}(D=\text{privileged})}")
