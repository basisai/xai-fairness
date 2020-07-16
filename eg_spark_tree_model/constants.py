"""
This file provides the exact names of the feature columns, target column and fairness configuration.

FEATURES, TARGET and CONFIG_FAI are required in the report.
"""
FEATURES = [f"feat_{i}" for i in range(692)]
TARGET = "label"

CONFIG_FAI = {
    'feat_126': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [28, 42, 58, 73, 122, 155, 157, 253, 254, 255],
    },
    'feat_180': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [6, 86, 118, 119, 177, 221, 228, 251, 252, 253, 254],
    },
}
