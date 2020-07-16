"""
Script containing commonly used functions.
"""
import numpy as np
import pandas as pd
import shap
from aif360.datasets import BinaryLabelDataset
from aif360.metrics.classification_metric import ClassificationMetric


def compute_shap_values(
    x,
    model=None,
    model_type=None,
    predict_func=None,
    bkgrd_data=None,
    kmeans_size=10,
):
    """Function to compute SHAP values, which are used for XAI.
    Use the relevant explainer for each type of model.
    :param pandas.DataFrame x: Validation/test data to use for explanation
    :param Optional model: Model to compute shap values for. In case model is of unsupported
        type, use predict_func to pass in a generic function instead
    :param Optional[str] model_type: Type of the model
    :param Optional[Callable] predict_func: Generic function to compute shap values for.
        It should take a matrix of samples (# samples x # features) and compute the
        output of the model for those samples.
        The output can be a vector (# samples) or a matrix (# samples x # model outputs).
    :param: Optional[pandas.DataFrame] bkgrd_data: background data for explainability analysis
    :param: Optional[int] kmeans_size: Number of k-means clusters. Only required for explaining generic
        predict_func
    :return Tuple[list(numpy.array), numpy_array]: shap_values, base_value.
    len(base_value) == len(shap_values)
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    else:
        if bkgrd_data is None:
            raise ValueError("Non tree model requires background data")
        if model_type == "linear":
            explainer = shap.LinearExplainer(model, bkgrd_data)
        else:
            explainer = _get_kernel_explainer(predict_func, bkgrd_data, kmeans_size)

    return _compute_shap(explainer, x)


def _get_kernel_explainer(predict_func, bkgrd_data, kmeans_size=10):
    if predict_func is None:
        raise ValueError("No target to compute shap values. Expected either model or predict_func")
    # rather than use the whole training set to estimate expected values,
    # summarize with a set of weighted kmeans, each weighted by
    # the number of points they represent.
    if kmeans_size is None:
        x_bkgrd_summary = bkgrd_data
    else:
        x_bkgrd_summary = shap.kmeans(bkgrd_data, kmeans_size)
    return shap.KernelExplainer(predict_func, x_bkgrd_summary)


def _compute_shap(explainer, x):
    """Get shap_values and base_value."""
    all_shap = explainer.shap_values(x)
    all_base = np.array(explainer.expected_value).reshape(-1)

    if len(all_base) == 1:
        # regressor or binary XGBClassifier
        return [all_shap], all_base

    elif len(all_base) == 2:
        # binary classifier, only take the values for class=1
        return all_shap[1:], all_base[1:]

    # multiclass classifier
    return all_shap, all_base


def compute_corrcoef(features, shap_values):
    """
    Compute correlation between each feature and its SHAP values.
    :param pandas.DataFrame features:
    :param numpy.array shap_values:
    :return numpy.array: (shape = (dim of predict output, number of features))
    """
    all_corrs = list()
    for cls_shap_val in shap_values:
        corrs= list()
        for i in range(features.shape[1]):
            df_ = pd.DataFrame({"x": features.iloc[:, i].values, "y": cls_shap_val[:, i]})
            corrs.append(df_.corr(method="pearson").values[0, 1])
        all_corrs.append(np.array(corrs))
    return all_corrs


def prepare_dataset(features,
                    labels,
                    protected_attribute,
                    privileged_attribute_values,
                    unprivileged_attribute_values,
                    favorable_label=1.,
                    unfavorable_label=0.):
    """Prepare dataset for computing fairness metrics."""
    df = features.copy()
    df['outcome'] = labels

    return BinaryLabelDataset(
        df=df,
        label_names=['outcome'],
        scores_names=list(),
        protected_attribute_names=[protected_attribute],
        privileged_protected_attributes=[np.array(privileged_attribute_values)],
        unprivileged_protected_attributes=[np.array(unprivileged_attribute_values)],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )


def get_aif_metric(valid,
                   true_class,
                   pred_class,
                   protected_attribute,
                   privileged_attribute_values,
                   unprivileged_attribute_values,
                   favorable_label=1.,
                   unfavorable_label=0.):
    """Get aif metric wrapper."""
    grdtruth = prepare_dataset(
        valid,
        true_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    predicted = prepare_dataset(
        valid,
        pred_class,
        protected_attribute,
        privileged_attribute_values,
        unprivileged_attribute_values,
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )

    aif_metric = ClassificationMetric(
        grdtruth,
        predicted,
        unprivileged_groups=[{protected_attribute: v} for v in unprivileged_attribute_values],
        privileged_groups=[{protected_attribute: v} for v in privileged_attribute_values],
    )
    return aif_metric


def compute_fairness_measures(aif_metric):
    """Compute fairness measures."""
    fmeasures = list()

    # Equal opportunity: equal FNR
    fnr_ratio = aif_metric.false_negative_rate_ratio()
    fmeasures.append([
        "Equal opportunity",
        "Separation",
        aif_metric.false_negative_rate(),
        aif_metric.false_negative_rate(False),
        aif_metric.false_negative_rate(True),
        fnr_ratio,
    ])

    # Predictive parity: equal PPV
    ppv_all = aif_metric.positive_predictive_value()
    ppv_up = aif_metric.positive_predictive_value(False)
    ppv_p = aif_metric.positive_predictive_value(True)
    ppv_ratio = ppv_up / ppv_p
    fmeasures.append([
        "Predictive parity",
        "Sufficiency",
        ppv_all,
        ppv_up,
        ppv_p,
        ppv_ratio,
    ])

    # Statistical parity
    disparate_impact = aif_metric.disparate_impact()
    fmeasures.append([
        "Statistical parity",
        "Independence",
        aif_metric.selection_rate(),
        aif_metric.selection_rate(False),
        aif_metric.selection_rate(True),
        disparate_impact,
    ])

    # Predictive equality: equal FPR
    fpr_ratio = aif_metric.false_positive_rate_ratio()
    fmeasures.append([
        "Predictive equality",
        "Separation",
        aif_metric.false_positive_rate(),
        aif_metric.false_positive_rate(False),
        aif_metric.false_positive_rate(True),
        fpr_ratio,
    ])

    # Equalized odds: equal TPR and equal FPR
    eqodds_all = (aif_metric.true_positive_rate() +
                  aif_metric.false_positive_rate()) / 2
    eqodds_up = (aif_metric.true_positive_rate(False) +
                 aif_metric.false_positive_rate(False)) / 2
    eqodds_p = (aif_metric.true_positive_rate(True) +
                aif_metric.false_positive_rate(True)) / 2
    eqodds_ratio = eqodds_up / eqodds_p
    fmeasures.append([
        "Equalized odds",
        "Separation",
        eqodds_all,
        eqodds_up,
        eqodds_p,
        eqodds_ratio,
    ])

    # Conditional use accuracy equality: equal PPV and equal NPV
    acceq_all = (aif_metric.positive_predictive_value(False) +
                 aif_metric.negative_predictive_value(False)) / 2
    acceq_up = (aif_metric.positive_predictive_value(False) +
                aif_metric.negative_predictive_value(False)) / 2
    acceq_p = (aif_metric.positive_predictive_value(True) +
               aif_metric.negative_predictive_value(True)) / 2
    acceq_ratio = acceq_up / acceq_p
    fmeasures.append([
        "Conditional use accuracy equality",
        "Sufficiency",
        acceq_all,
        acceq_up,
        acceq_p,
        acceq_ratio,
    ])

    return pd.DataFrame(fmeasures, columns=[
        "Metric", "Criterion", "All", "Unprivileged", "Privileged", "Ratio"])


def get_perf_measure_by_group(aif_metric, metric_name):
    """Get performance measures by group."""
    perf_measures = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']

    func_dict = {
        'selection_rate': lambda x: aif_metric.selection_rate(privileged=x),
        'precision': lambda x: aif_metric.precision(privileged=x),
        'recall': lambda x: aif_metric.recall(privileged=x),
        'sensitivity': lambda x: aif_metric.sensitivity(privileged=x),
        'specificity': lambda x: aif_metric.specificity(privileged=x),
        'power': lambda x: aif_metric.power(privileged=x),
        'error_rate': lambda x: aif_metric.error_rate(privileged=x),
    }

    if metric_name in perf_measures:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame({
        'Group': ['all', 'privileged', 'unprivileged'],
        metric_name: [metric_func(group) for group in [None, True, False]],
    })
    return df


def color_red(x):
    """Styling: color red."""
    return "color: red" if x == "No" else "color: black"
