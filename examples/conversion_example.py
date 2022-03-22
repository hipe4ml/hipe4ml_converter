"""
Minimal example to run the package methods
"""
import os
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

from hipe4ml.model_handler import ModelHandler
from hipe4ml_converter.h4ml_converter import H4MLConverter

# DATA PREPARATION (load data from sklearn digits dataset)
# --------------------------------------------
SKLEARN_DATA = datasets.load_digits(n_class=2)
DIGITS_DATASET = pd.DataFrame(SKLEARN_DATA.data)  # pylint: disable=E1101
Y_DIGITS = SKLEARN_DATA.target  # pylint: disable=E1101
SIG_DF = DIGITS_DATASET[Y_DIGITS == 1]
BKG_DF = DIGITS_DATASET[Y_DIGITS == 0]
TRAIN_SET, TEST_SET, Y_TRAIN, Y_TEST = train_test_split(DIGITS_DATASET,
                                                        Y_DIGITS,
                                                        test_size=0.5,
                                                        random_state=42)
DATA = [TRAIN_SET, Y_TRAIN, TEST_SET, Y_TEST]
# --------------------------------------------

# TRAINING AND TESTING
# --------------------------------------------
INPUT_MODEL = xgb.XGBClassifier()
MODEL = ModelHandler(INPUT_MODEL)
MODEL.train_test_model(DATA)
Y_PRED = MODEL.predict(DATA[2])

# CONVERSION
MODEL_CONVERTER = H4MLConverter(MODEL)
# --------------------------------------------
# creates a ONNX model that can process a 1-dimension dataset at a time and saves it to file
MODEL_CONVERTER.convert_model_onnx(1)
MODEL_CONVERTER.dump_model_onnx("model_onnx.onnx")
# creates a tensorial ONNX model via hummingbird that can process a 1-dimension dataset at a time and saves it to file
MODEL_CONVERTER.convert_model_hummingbird("onnx", 1)
MODEL_CONVERTER.dump_model_hummingbird("model_hummingbird_onnx")
# creates a tensorial pytorch model via hummingbird
MODEL_CONVERTER.convert_model_hummingbird("torch")
MODEL_CONVERTER.dump_model_hummingbird("model_hummingbird_torch")
os._exit(0)
# ---------------------------------------------
