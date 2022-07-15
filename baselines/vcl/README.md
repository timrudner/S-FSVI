# Variational Continual Learning

**Acknowledgement**

All files in this subdirectory were copied from [nvcuong/variational-continual-learning](https://github.com/nvcuong/variational-continual-learning/tree/master/ddm) without changes, unless stated otherwise below.

```
.
+-- alg/
|   +-- cla_models_multihead.py
|   +-- coreset.py
|   +-- data_generator.py (We added a `CustomGenerator` as an adapter to our data loader.)
|   +-- utils.py
|   +-- vcl.py
+-- cl_vcl.py (We wrote this script to test VCL on our data loader. Tt is called by `run_vcl.py`.)
+-- run_vcl.py (We wrote this script as the entry point to run experiments on VCL.)
```
