"""
Example for conversion of a hipe4ml ModelHandler to ONNX format
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from onnx import onnx_ml_pb2
from hummingbird.ml.containers.sklearn.onnx_containers import ONNXSklearnContainerClassification
from hummingbird.ml.containers.sklearn.pytorch_containers import PyTorchSklearnContainerClassification
from hipe4ml.model_handler import ModelHandler
from hipe4ml_converter.h4ml_converter import H4MLConverter

df_sgn = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
df_bkg = pd.DataFrame(3 * np.random.randn(100, 4) + 2, columns=list("ABCD"))
df_tot = pd.concat([df_bkg, df_sgn], sort=True)
labels = np.array([0] * 100 + [1] * 100)
train_set, test_set, y_train, y_test = train_test_split(df_tot,
                                                        labels,
                                                        test_size=0.5,
                                                        random_state=42)
train_test_data = [train_set, y_train, test_set, y_test]

model_clf = xgb.XGBClassifier(use_label_encoder=False)
model_handler = ModelHandler(model_clf, df_sgn.columns)
model_handler.train_test_model(train_test_data, True, output_margin="raw")

model_converter = H4MLConverter(model_handler)


def test_convert_model_onnx():
    """
    Test the conversion to onnx
    """
    assert isinstance(model_converter.convert_model_onnx(1),
                      onnx_ml_pb2.ModelProto)


def test_dump_model_onnx():
    """
    Test the dump of the onnx file
    """
    model_converter.dump_model_onnx("model.onnx")
    assert os.path.isfile("model.onnx")
    os.remove("model.onnx")


def test_convert_model_hummingbird_onnx():
    """
    Test the hummingbird conversion to onnx
    """
    assert isinstance(model_converter.convert_model_hummingbird("onnx", 1),
                      ONNXSklearnContainerClassification)


def test_dump_model_hummingbird_onnx():
    """
    Test the dump of the onnx file
    """
    model_converter.dump_model_hummingbird("model_onnx")
    assert os.path.isfile("model_onnx.zip")
    os.remove("model_onnx.zip")


def test_convert_model_hummingbird_pytorch():
    """
    Test the hummingbird conversion to pytorch
    """
    assert isinstance(model_converter.convert_model_hummingbird("pytorch", 1),
                      PyTorchSklearnContainerClassification)


def test_dump_model_hummingbird_pytorch():
    """
    Test the dump of the pytorch file
    """
    model_converter.dump_model_hummingbird("model_pytorch")
    assert os.path.isfile("model_pytorch.zip")
    os.remove("model_pytorch.zip")


def test_convert_model_hummingbird_torch():
    """
    Test the hummingbird conversion to torch
    """
    assert isinstance(model_converter.convert_model_hummingbird("torch", 1),
                      PyTorchSklearnContainerClassification)


def test_dump_model_hummingbird_torch():
    """
    Test the dump of the torch file
    """
    model_converter.dump_model_hummingbird("model_torch")
    assert os.path.isfile("model_torch.zip")
    os.remove("model_torch.zip")
