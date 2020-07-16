"""
Entry point for the report.
"""
import sys
sys.path.append("..")

import yaml

import pandas as pd
import streamlit as st
from sklearn import metrics
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel

from constants import FEATURES, TARGET, CONFIG_FAI
from xai_fairness.report import ReportPreparer, get_repr_samples

CONFIG = yaml.load(open("xai_fairness.yaml", "r"), Loader=yaml.SafeLoader)


def load_model():
    """Write model loading function."""
    # Amend the code here between ##
    model = RandomForestClassificationModel.load("inputs/rf_model")
    ################################
    return model


@st.cache(allow_output_mutation=True)
def load_valid_data():
    """
    Write validation data loading function.
    1000-10000s are good sizes to use.
    """
    # Amend the code here between ##
    df = pd.read_parquet("inputs/valid.gz.parquet")
    ################################
    return df


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
    text += "Model accuracy = {:.4f}\n\n".format(metrics.accuracy_score(y_true, y_pred))
    text += metrics.classification_report(y_true, y_pred, digits=4)
    ################################
    return text


def prepare_report():
    """
    Prepare the data inputs for the report.
    Fill the code here to generate the data necessary for the report.
    """
    with (
            SparkSession.builder
            .appName("PySpark_Testing")
            .config('spark.driver.memory', '8g')
            .config('spark.driver.cores', '2')
            .config('spark.executor.instances', '1')
            .config('spark.executor.memory', '4g')
            .config('spark.executor.cores', '1')
            .getOrCreate()
    ) as spark:

        # Load model
        model = load_model()

        # Load valid data
        valid = load_valid_data()
        x_valid = valid[FEATURES]
        y_valid = valid[TARGET].values

        # Predict on valid data
        y_score = valid["probability"].values

        # Assign classes
        pred_class = (y_score > 0.5).astype(int)
        true_class = y_valid

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
