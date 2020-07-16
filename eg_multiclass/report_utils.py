"""
Entry point for the report.
"""
import sys
sys.path.append("..")

import pickle
import yaml

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

from constants import FEATURES, TARGET, CONFIG_FAI
from xai_fairness.report import ReportPreparer, get_repr_samples

CONFIG = yaml.load(open("xai_fairness.yaml", "r"), Loader=yaml.SafeLoader)


@st.cache(allow_output_mutation=True)
def load_model():
    """Write model loading function."""
    # Amend the code here between ##
    model = pickle.load(open("inputs/model.pkl", "rb"))
    ################################
    return model


@st.cache
def load_valid_data():
    """
    Write validation data loading function.
    1000-10000s are good sizes to use.
    """
    # Amend the code here between ##
    df = pd.read_csv("inputs/valid.csv")
    ################################
    return df

 
def predict_scores(model, x):
    """
    Write your predict function.
    For classification, predict probabilities.
    For regression, predict scores.

    Args:
        model
        x (pandas.DataFrame)

    Returns:
        scores (numpy.array or 1d array-like)
    """
    # Amend the code here between ##
    scores = model.predict_proba(x)
    ################################
    return scores


def assign_pred_class(y_score):
    """
    Assign the predicted scores into classes.
    These classes will be used for computing fairness metrics.
    The approach will one vs one for two classes, or one vs all for more than two classes.

    Args:
        y_score (1d array-like)

    Returns:
        pred_class (1d array-like)
    """
    # Amend the code here between ##
    pred_class = np.argmax(y_score, axis=1)
    ################################
    return pred_class


def assign_true_class(y_true):
    """
    Assign the true target values into classes.
    These classes will be used for computing fairness metrics.
    The approach will one vs one for two classes, or one vs all for more than two classes.

    Args:
        y_true (1d array-like)

    Returns:
        true_class (1d array-like)
    """
    # Amend the code here between ##
    true_class = y_true
    ################################
    return true_class


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
    text = ""
    # Amend the code here between ##
    text += "Model accuracy = {:.4f}\n".format(metrics.mean_squared_error(y_true, y_pred))
    text += metrics.classification_report(y_true, y_pred, digits=4)
    ################################
    return text


@st.cache
def prepare_report():
    """
    Prepare the data inputs for the report.
    Fill the code here to generate the data necessary for the report.
    """
    # Load model
    model = load_model()

    # Load valid data
    valid = load_valid_data()
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    # Predict on valid data
    y_score = predict_scores(model, x_valid)

    # Assign classes
    pred_class = assign_pred_class(y_score)
    true_class = assign_true_class(y_valid)

    # dict of individual instances of data for individual XAI
    indiv_samples = get_repr_samples(x_valid, pred_class)

    # Compute model performance
    text_model_perf = print_model_perf(y_valid, y_pred=pred_class)

    # Valid data with sensitive attributes only
    valid_fai = valid[list(CONFIG_FAI.keys())]

    # Prepare the report to be generated
    report_inputs = ReportPreparer(
        CONFIG,
        CONFIG_FAI,
        FEATURES,
    )
    (report_inputs
     .model_artefact(model_type="tree", model=model)
     .xai_data(x_valid)
     .indiv_xai_data(indiv_samples)
     .fai_data(valid_fai, true_class, pred_class)
     .model_perf_to_print(text_model_perf))

    return report_inputs


def main():
    report_inputs = prepare_report()
    report_inputs.generate_report()


if __name__ == "__main__":
    main()
