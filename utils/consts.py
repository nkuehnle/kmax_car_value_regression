from typing import List, Dict
import numpy as np
import pandas as pd

APPRAISAL_COLS: List[str] = [
    "value_appraisal",
    "online_appraisal_flag",
    "model_year_appraisal",
    "mileage_appraisal",
    "make_appraisal",
    "model_appraisal",
    "trim_descrip_appraisal",
    "body_appraisal",
    "color_appraisal",
    "engine_appraisal",
    "cylinders_appraisal",
    "mpg_city_appraisal",
    "mpg_highway_appraisal",
    "horsepower_appraisal",
    "fuel_capacity_appraisal",
    "market",
]

PURCHASE_COLS: List[str] = [
    "value_purchase",
    "model_year_purchase",
    "mileage_purchase",
    "make_purchase",
    "model_purchase",
    "trim_descrip_purchase",
    "body_purchase",
    "color_purchase",
    "engine_purchase",
    "cylinders_purchase",
    "mpg_city_purchase",
    "mpg_highway_purchase",
    "horsepower_purchase",
    "fuel_capacity_purchase",
    "market",
]


# Mapping of appraisal cols to purchase cols
LONG_MAPPER: Dict[str, str] = {
    "market": "market",
    "online_appraisal_flag": "online_appraisal_flag",
}
# Update LONG_MAPPER with other columns
for c in PURCHASE_COLS:
    if c not in LONG_MAPPER.keys():
        appraisal_c = c.replace("_purchase", "")
        appraisal_c = f"{appraisal_c}_appraisal"
        LONG_MAPPER.update({appraisal_c: c})


SHORT_MAPPER: Dict[str, str] = {
    v.replace("_purchase", ""): k for k, v in LONG_MAPPER.items()
}

SHARED_COLS: List[str] = ["market,", "itransation", "online_appraisal_flag"]