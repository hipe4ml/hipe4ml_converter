"""
Minimal example to run the package methods
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from onnxruntime import InferenceSession

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
TEST_SET_FLOAT = DATA[2].to_numpy().astype(np.float32)
Y_PRED = MODEL.predict(DATA[2], output_margin=False)

# CONVERSION
MODEL_CONVERTER = H4MLConverter(MODEL)
# --------------------------------------------
# creates a ONNX model that can process a N-dimension dataset at a time and saves it to file
MODEL_ONNX = MODEL_CONVERTER.convert_model_onnx(len(TEST_SET_FLOAT))
ONNX_SESSION = InferenceSession(MODEL_ONNX.SerializeToString())
OUTPUT_ONNX = ONNX_SESSION.run(None, {"input": TEST_SET_FLOAT})
# check that the two outputs are equal up to the 7th digit
np.testing.assert_almost_equal(np.transpose(OUTPUT_ONNX[1])[1], Y_PRED, decimal=7)
MODEL_CONVERTER.dump_model_onnx("model_onnx.onnx")

# creates a tensorial ONNX model via hummingbird that can process a N-dimension dataset at a time and saves it to file
MODEL_ONNX_HUMMINGBIRD = MODEL_CONVERTER.convert_model_hummingbird("onnx", len(TEST_SET_FLOAT))
OUTPUT_ONNX_HUMMINGBIRD = MODEL_ONNX_HUMMINGBIRD.predict_proba(DATA[2].to_numpy())
# check that the two outputs are equal up to the 7th digit
np.testing.assert_almost_equal(np.transpose(OUTPUT_ONNX_HUMMINGBIRD)[1], Y_PRED, decimal=7)
MODEL_CONVERTER.dump_model_hummingbird("model_hummingbird_onnx")

# creates a tensorial pytorch model via hummingbird
MODEL_TORCH_HUMMINGBIRD = MODEL_CONVERTER.convert_model_hummingbird("torch")
OUTPUT_TORCH_HUMMINGBIRD = MODEL_TORCH_HUMMINGBIRD.predict_proba(DATA[2].to_numpy())
# check that the two outputs are equal up to the 7th digit
np.testing.assert_almost_equal(np.transpose(OUTPUT_TORCH_HUMMINGBIRD)[1], Y_PRED, decimal=7)
MODEL_CONVERTER.dump_model_hummingbird("model_hummingbird_torch")

# force exit
os._exit(0) #pylint: disable=protected-access

# ---------------------------------------------
