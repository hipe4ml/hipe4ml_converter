"""
Example for conversion of a hipe4ml ModelHandler to ONNX format
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler

sys.path.append("../hipe4ml_converter") # needed because hipe4ml-converter is not installed
from h4ml_converter import H4MLConverter #pylint: disable=wrong-import-position,import-error

df_sgn = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
df_bkg = pd.DataFrame(3 * np.random.randn(100, 4) + 2, columns=list("ABCD"))
df_tot = pd.concat([df_bkg, df_sgn], sort=True)
labels = np.array([0]*100 + [1]*100)
train_set, test_set, y_train, y_test = train_test_split(df_tot, labels, test_size=0.5, random_state=42)
train_test_data = [train_set, y_train, test_set, y_test]

model_clf = xgb.XGBClassifier(use_label_encoder=False)
model_handler = ModelHandler(model_clf, df_sgn.columns)
model_handler.train_test_model(train_test_data, True, output_margin="raw")

model_converter = H4MLConverter(model_handler)
model_converter.convert_model_hummingbird("onnx", 1)
model_converter.dump_model_hummingbird("model_onnx")
model_converter.convert_model_hummingbird("pytorch", 1)
model_converter.dump_model_hummingbird("model_pytorch")
model_converter.convert_model_hummingbird("torch", 1)
model_converter.dump_model_hummingbird("model_torch")
