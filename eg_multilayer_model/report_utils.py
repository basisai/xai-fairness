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


class Model:
    """Multilayer model.

    l1_model1----\
                  l2_model-----l3_model
    l1_model2----/
    """

    def __init__(self, l1_model1, l1_model2, l2_model, l3_model):
        self.l1_model1 = l1_model1
        self.l1_model2 = l1_model2
        self.l2_model = l2_model
        self.l3_model = l3_model

    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        
        x2 = pd.DataFrame()
        x2["model1"] = self.l1_model1.predict_proba(x[:, :20])[:, 1]
        x2["model2"] = self.l1_model2.predict_proba(x[:, 20:40])[:, 1]
        
        x3 = pd.DataFrame(x[:, 40:])
        x3["l2_model"] = self.l2_model.predict_proba(x2)[:, 1]

        output = self.l3_model.predict_proba(x3)
        return output
    
    def predict(self, x):
        result = self.predict_proba(x)
        return np.argmax(result, axis=1)


@st.cache(allow_output_mutation=True)
def load_model():
    """Write model loading function."""
    # Amend the code here between ##
    def _load_pkl(filename):
        return pickle.load(open(filename, "rb"))
    
    l1_model1 = _load_pkl("inputs/l1_model1.pkl")
    l1_model2 = _load_pkl("inputs/l1_model2.pkl")
    l2_model = _load_pkl("inputs/l2_model.pkl")
    l3_model = _load_pkl("inputs/l3_model.pkl")

    model = Model(l1_model1, l1_model2, l2_model, l3_model)
    ################################
    return model


@st.cache
def load_valid_data():
    """
    Write validation data loading function.
    1000-10000s are good sizes to use.
    """
    # Amend the code here between ##
    df = (
        pd.read_csv("inputs/valid.csv")
        .sample(100, random_state=0)
    )
    ################################
    return df


@st.cache
def load_background_data():
    """
    Background data loading function. Used for shap.KernelExplainer
    """
    # Amend the code here between ##
    df = pd.read_csv("inputs/train.csv", usecols=FEATURES)
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
    scores = model.predict_proba(x)[:, 1]
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
    pred_class = (y_score > 0.5).astype(int)
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


def print_model_perf(y_val, y_pred=None, y_score=None):
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
    text += "Model accuracy = {:.4f}\n".format(metrics.accuracy_score(y_val, y_pred))
    text += "Weighted Average Precision = {:.4f}\n".format(metrics.precision_score(y_val, y_pred, average="weighted"))
    text += "Weighted Average Recall = {:.4f}\n\n".format(metrics.recall_score(y_val, y_pred, average="weighted"))
    text += metrics.classification_report(y_val, y_pred, digits=4)
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
     .model_artefact(predict_func=model.predict_proba, bkgrd_data=x_bkgrd)
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
