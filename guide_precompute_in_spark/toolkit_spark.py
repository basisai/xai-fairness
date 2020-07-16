import numpy as np
import shap


def get_explainer(model=None, model_type=None, predict_func=None, bkgrd_data=None, kmeans_size=10):
    """Function to select the SHAP explainer.
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
    :return explainer
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    else:
        if bkgrd_data is None:
            raise ValueError("Non tree model requires background data")
        if model_type == "linear":
            explainer = shap.LinearExplainer(model, bkgrd_data)
        else:
            explainer = _get_kernel_explainer(predict_func, bkgrd_data, kmeans_size)
    return explainer


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


def compute_shap(explainer, x):
    """Compute shap_values and base_value."""
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
