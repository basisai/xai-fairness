"""
This file provides the exact names of the feature columns, target column and fairness configuration.

FEATURES, TARGET and CONFIG_FAI are required in the report.
"""
# List numeric and categorical features
NUMERIC_FEATS = [
    'DayOfWeek', 'SchoolHoliday', 'month', 'weekofyear', 'StateHoliday_a',
    'StateHoliday_b', 'StateHoliday_c',
]

# For each categorical feature, get the one-hot encoded feature names
CATEGORY_MAP = {
    "StoreType": ['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d'],
    "Assortment": ["Assortment_a", "Assortment_b", "Assortment_c"],
}

CATEGORICAL_FEATS = list(CATEGORY_MAP.keys())

OHE_CAT_FEATS = []
for f in CATEGORICAL_FEATS:
    OHE_CAT_FEATS.extend(CATEGORY_MAP[f])

# Train & validation features and target
FEATURES = OHE_CAT_FEATS + NUMERIC_FEATS

TARGET = 'Sales'

# List privileged info
CONFIG_FAI = {
    'SchoolHoliday': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
    'DayOfWeek': {
        'unprivileged_attribute_values': [1, 2, 3, 4, 5],
        'privileged_attribute_values': [6, 7],
    },
}
