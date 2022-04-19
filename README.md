[![](https://img.shields.io/github/license/hipe4ml/hipe4ml-converter)](https://github.com/hipe4ml/hipe4ml-converter/blob/main/LICENSE)
[![](https://img.shields.io/pypi/pyversions/hipe4ml_converter.svg?longCache=True)](https://pypi.org/project/hipe4ml_converter/)
[![](https://img.shields.io/pypi/v/hipe4ml_converter.svg?maxAge=3600)](https://pypi.org/project/hipe4ml_converter/)
![](https://github.com/hipe4ml/hipe4ml_converter/actions/workflows/pythonpackage.yml/badge.svg?branch=main)
![](https://github.com/hipe4ml/hipe4ml_converter/actions/workflows/pythonpublish.yml/badge.svg)

# hipe4ml_converter
Package for conversion of machine learning models trained with hipe4ml into ONNX and tensor formats

# Introduction

hipe4ml-converter enables you to convert models trained with [hipe4ml](https://github.com/hipe4ml/hipe4ml) into [ONNX](https://onnx.ai/) format and tensor formats ([PyTorch](https://pytorch.org/), [TorchScript](https://pytorch.org/docs/stable/jit.html), [TVM](https://tvm.apache.org/) or [ONNX](https://onnx.ai/)) via the [hummingbird](https://github.com/microsoft/hummingbird) library.

# Requirements

- [hipe4ml](https://github.com/hipe4ml/hipe4ml)
- [onnx](https://github.com/onnx/onnx)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
- [onnxmltools](https://github.com/onnx/onnxmltools)
- [hummingbird](https://github.com/microsoft/hummingbird)
