"""
Module containing the class used for wrapping the methods used to convert a model
stored into a hipe4ml ModelHandler to different formats
"""

import numpy as np
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from hummingbird import ml

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
        self.model_hummingbird = None

    def convert_model_onnx(self, input_shape):
        """
        Convert the trained model to onnx format and save it

        Parameters
        -----------------------------------------------------
        input_shape: int
            The dimension of the sample for the application.
            For more info see https://github.com/onnx/onnxmltools

        Returns
        -----------------------------------------------------
        model_onnx: onnxtools ModelProto
            The model converted to onnx format.
            For more info see https://github.com/onnx/onnxmltools
        """

        training_columns = self.model_handler.get_training_columns()
        n_features = len(training_columns)
        feature_names = [f"f{i_feat}" for i_feat in range(n_features)]
        model = self.model_handler.get_original_model()
        model.get_booster().feature_names = feature_names

        self.model_onnx = onnxmltools.convert.convert_xgboost(
            model, initial_types=[("input", FloatTensorType(shape=[input_shape, n_features]))]
        )

        return self.model_onnx

    def convert_model_hummingbird(self, backend, input_shape=None):
        """
        Convert the trained model to a tensor format
        and save it with hummingbird

        Parameters
        -----------------------------------------------------
        backend: str
            output backend: PyTorch, TorchScript, ONNX are supported
            For more information see https://github.com/microsoft/hummingbird

        input_shape: int
            The dimension of the sample for the application. Needed in case of onnx backend
            For more info see https://github.com/onnx/onnxmltools

        Returns
        -----------------------------------------------------
        model_hummingbird:
            The model converted to hummingbird format.
            For more info see https://github.com/microsoft/hummingbird
        """

        if backend not in ["pytorch", "torch", "onnx"]:
            print(f"backend {backend} not supported by hummingbird."
                  " Options: [pytorch, torch, onnx]"
                  " See documentation https://github.com/microsoft/hummingbird")
            return None

        training_columns = self.model_handler.get_training_columns()
        n_features = len(training_columns)
        feature_names = [f"f{i_feat}" for i_feat in range(n_features)]
        model = self.model_handler.get_original_model()
        model.get_booster().feature_names = feature_names

        if backend == "onnx":
            x_test = np.random.rand(input_shape, n_features)
            self.model_hummingbird = ml.convert(model, backend, x_test)
        else:
            self.model_hummingbird = ml.convert(model, backend, extra_config={"n_features":n_features})

        return self.model_hummingbird

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

    def dump_model_hummingbird(self, filename):
        """
        Save the trained model into a file.
        The format depends on the backend used in the hummingbird conversion

        Parameters
        -----------------------------------------------------
        filename: str
            Name of the file in which the model is saved
        """

        if self.model_hummingbird is not None:
            self.model_hummingbird.save(filename)
            print(f"File {filename} saved")
        else:
            print("File not saved: the model should be first converted with convert_model_onnx")
