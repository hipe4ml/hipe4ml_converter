"""
Module containing the class used for wrapping the methods used to convert a model
stored into a hipe4ml ModelHandler to different formats
"""

import numpy as np
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType


class H4MLConverter:
    """
    class used for wrapping the methods used to convert a model
    stored into a hipe4ml.ModelHandler to ONNX format or to tensor formats
    (PyTorch, TorchScript, ONNX)

    Parameters
    -------------------------------------------------
    input_model: hipe4ml ModelHandler
    """

    def __init__(self, input_model=None):
        self.model_handler = input_model
        self.model_onnx = None

    def convert_model_onnx(self, input_shape, target_opset=13):
        """
        Convert the trained model to onnx format and save it

        Parameters
        -----------------------------------------------------
        input_shape: int
            The dimension of the sample for the application.
            For more info see https://github.com/onnx/onnxmltools
        target_opset: int
            ONNX opset version. The default is 13 supported by ONNX>=1.8 and ONNX Runtime>=1.6.
            For more info see https://onnxruntime.ai/docs/reference/compatibility#onnx-opset-support
        Returns
        -----------------------------------------------------
        model_onnx: onnxtools ModelProto
            The model converted to onnx format.
            For more info see https://github.com/onnx/onnxmltools
        """

        training_columns = self.model_handler.get_training_columns()
        n_features = len(training_columns)
        model = self.model_handler.get_original_model()
        feature_names = [f"f{i_feat}" for i_feat in range(n_features)]
        model.get_booster().feature_names = feature_names

        self.model_onnx = onnxmltools.convert.convert_xgboost(
            model, target_opset=target_opset,
            initial_types=[("input", FloatTensorType(shape=[input_shape, n_features]))]
        )

        # restore original names
        model.get_booster().feature_names = list(training_columns)

        return self.model_onnx

    def dump_model_onnx(self, filename):
        """
        Save the trained model into a .onnx file

        Parameters
        -----------------------------------------------------
        filename: str
            Name of the file in which the model is saved
        """

        if self.model_onnx is not None:
            onnxmltools.utils.save_model(self.model_onnx, filename)
            print(f"File {filename} saved")
        else:
            print("File not saved: the model should be first converted with convert_model_onnx")
