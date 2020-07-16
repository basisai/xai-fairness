"""
This file provides the exact names of the feature columns, target column and fairness configuration.

FEATURES, TARGET and CONFIG_FAI are required in the report.
"""
import json

# List numeric and categorical features
NUMERIC_FEATS = json.load(open("inputs/numeric_feats.txt"))

# For each categorical feature, get the one-hot encoded feature names
CATEGORY_MAP = json.load(open("inputs/category_map.txt"))

CATEGORICAL_FEATS = list(CATEGORY_MAP.keys())

OHE_CAT_FEATS = []
for f in CATEGORICAL_FEATS:
    OHE_CAT_FEATS.extend(CATEGORY_MAP[f])

# Train & validation features and target
FEATURES = OHE_CAT_FEATS + NUMERIC_FEATS

# [LightGBM] [Fatal] Do not support special JSON characters in feature name.
FEATURES = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in FEATURES]

TARGET = 'TARGET'

BINARY_MAP = {
    'CODE_GENDER': ['M', 'F'],
    'FLAG_OWN_CAR': ['N', 'Y'],
    'FLAG_OWN_REALTY': ['Y', 'N'],
}

CONFIG_FAI = {
    'CODE_GENDER': {
        'unprivileged_attribute_values': [1],
        'privileged_attribute_values': [0],
    },
    'NAME_EDUCATION_TYPE_Higher_education': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
}
