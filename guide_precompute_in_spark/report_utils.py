"""
Entry point for the report.
"""
import sys
sys.path.append("..")

import pickle
import yaml

import pandas as pd
import streamlit as st
from sklearn import metrics

from constants import FEATURES, TARGET, CONFIG_FAI
from xai_fairness.report import ReportPreparer

CONFIG = yaml.load(open("xai_fairness.yaml", "r"), Loader=yaml.SafeLoader)


@st.cache(allow_output_mutation=True)
def load_model():
    """Write model loading function."""
    # Amend the code here between ##
    model = pickle.load(open("inputs/lgb_model.pkl", "rb"))
    ################################
    return model


@st.cache(allow_output_mutation=True)
def load_all_data():
    # Amend the code here between ##
    indiv_samples = pickle.load(open("inputs/indiv_samples.pkl", "rb"))

    shap_summary_dfs = pickle.load(open("inputs/shap_summary_dfs.pkl", "rb"))
    shap_sample_dfs = pickle.load(open("inputs/shap_sample_dfs.pkl", "rb"))

    data = pd.read_csv("inputs/fai_data_df.csv")
    valid_fai = data[list(CONFIG_FAI.keys())]
    true_class = data["true_class"].values
    pred_class = data["pred_class"].values
    ################################
    return indiv_samples, shap_summary_dfs, shap_sample_dfs, valid_fai, true_class, pred_class


def print_model_perf(y_true, y_pred=None):
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
    text += "Model accuracy = {:.4f}\n\n".format(metrics.accuracy_score(y_true, y_pred))
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

    # Load data
    indiv_samples, shap_summary_dfs, shap_sample_dfs, valid_fai, true_class, pred_class = load_all_data()

    # Compute model performance
    text_model_perf = print_model_perf(true_class, y_pred=pred_class)

    # Prepare the report to be generated
    report_inputs = ReportPreparer(
        CONFIG,
        CONFIG_FAI,
        FEATURES,
    )
    (report_inputs
     .model_artefact(model_type="tree", model=model)
     .xai_data(shap_summary_dfs=shap_summary_dfs, shap_sample_dfs=shap_sample_dfs)
     .indiv_xai_data(indiv_samples)
     .fai_data(valid_fai, true_class, pred_class)
     .model_perf_to_print(text_model_perf))

    return report_inputs


def main():
    report_inputs = prepare_report()
    report_inputs.generate_report()


if __name__ == "__main__":
    main()
