"""
This file provides the exact names of the feature columns, target column and fairness configuration.

FEATURES, TARGET and CONFIG_FAI are required in the report.
"""
FEATURES =

TARGET =

CONFIG_FAI = {
    '<protected_attribute_1>': {
        'unprivileged_attribute_values': <unprivileged_attribute_values>,
        'privileged_attribute_values': <unprivileged_attribute_values>,
    },
    '<protected_attribute_2>': {
        'unprivileged_attribute_values': <unprivileged_attribute_values>,
        'privileged_attribute_values': <privileged_attribute_values>,
    },
}
