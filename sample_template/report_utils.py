"""
Entry point for the report.
"""
import sys
sys.path.append("..")

import yaml

import streamlit as st

from constants import FEATURES, TARGET, CONFIG_FAI
from xai_fairness.report import ReportPreparer

CONFIG = yaml.load(open("xai_fairness.yaml", "r"), Loader=yaml.SafeLoader)


@st.cache(allow_output_mutation=True)
def load_model():
    """Write model loading function."""
    pass


@st.cache
def load_valid_data():
    """
    Write validation data loading function.
    1000-10000s are good sizes to use.
    """
    pass


@st.cache
def load_background_data():
    """
    Background data loading function. Used for shap.KernelExplainer
    """
    pass

 
def predict_scores(model, x):
    """
    Predict function.
    """
    pass


def assign_pred_class(y_score):
    """
    Assign the predicted scores into classes.
    These classes will be used for computing fairness metrics.
    The approach will one vs one for two classes, or one vs all for more than two classes.
    """
    pass


def assign_true_class(y_true):
    """
    Assign the true target values into classes.
    These classes will be used for computing fairness metrics.
    The approach will one vs one for two classes, or one vs all for more than two classes.
    """
    pass


def print_model_perf(y_true, y_pred=None, y_score=None):
    """
    Write how you want to print model performance in text.
    Using validation data for computing model performance

    Args:
        y_true (1d array-like): Ground truth (correct) target values.
        y_pred (1d array-like): Estimated targets
        y-score (1d array-like): Target scores

    Returns:
        text (str)
    """
    pass


def get_repr_samples(x, y_class):
    """
    Select a representative sample from each class for individual XAI.

    Returns:
        dict(label: pd.DataFrame)
    """
    pass


@st.cache
def prepare_report():
    """
    Precompute the necessary ingredients for the report.
    It can be amended but the output dictionary must contain the necessary keys and values
    as listed below.
    """
    # Load model
    model = load_model()

    # Load valid data
    valid = load_valid_data()
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Background data
    x_bkgrd = load_background_data()

    # Predict on valid data
    y_score = predict_scores(model, x_valid)

    # Assign classes
    pred_class = assign_pred_class(y_score)
    true_class = assign_true_class(y_valid)

    # dict of individual instances of data for individual XAI
    indiv_samples = get_repr_samples(x_valid, pred_class)

    # Compute model performance
    text_model_perf = print_model_perf(y_valid, pred_class, y_score)

    # Valid data with sensitive attributes only
    valid_fai = valid[list(CONFIG_FAI.keys())]

    # Prepare the report to be generated
    report_inputs = ReportPreparer(
        CONFIG,
        CONFIG_FAI,
        FEATURES,
    )
    (report_inputs
     .model_artefact(...)
     .background_features(x_bkgrd)
     .model_perf_to_print(text_model_perf)
     .valid_features(x_valid)
     .representative_samples(indiv_samples)
     .fai_data(valid_fai, true_class, pred_class))

    return report_inputs


def main():
    report_inputs = prepare_report()
    report_inputs.generate_report()


if __name__ == "__main__":
    main()
