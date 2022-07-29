import os

import numpy as np
import tensorflow as tf

from baselines.vcl.alg.data_generator import SplitMnistGenerator

tf.compat.v1.disable_eager_execution()
import pickle
import sys

sys.path.extend(["alg/"])
from baselines.vcl.alg import vcl, coreset, utils

hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False

# Run vanilla VCL
tf.random.set_seed(12)
np.random.seed(1)

coreset_size = 0
data_gen = SplitMnistGenerator()
vcl_result = vcl.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    coreset.rand_from_batch,
    coreset_size,
    batch_size,
    single_head,
)
print(vcl_result)

# Run random coreset VCL
tf.compat.v1.reset_default_graph()
tf.random.set_seed(12)
np.random.seed(1)

coreset_size = 40
data_gen = SplitMnistGenerator()
rand_vcl_result = vcl.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    coreset.rand_from_batch,
    coreset_size,
    batch_size,
    single_head,
)
print(rand_vcl_result)

# Run k-center coreset VCL
tf.compat.v1.reset_default_graph()
tf.random.set_seed(12)
np.random.seed(1)

data_gen = SplitMnistGenerator()
kcen_vcl_result = vcl.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    coreset.k_center,
    coreset_size,
    batch_size,
    single_head,
)
print(kcen_vcl_result)

to_save = {
    "vcl_result": vcl_result,
    "rand_vcl_result": rand_vcl_result,
    "kcen_vcl_result": kcen_vcl_result,
}
RESULT_FOLDER = utils.VCL_ROOT / "results" / "original_data"
os.makedirs(RESULT_FOLDER, exist_ok=True)
with open(RESULT_FOLDER / "smnist", "wb") as p:
    pickle.dump(to_save, p)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot("results/split.jpg", vcl_avg, rand_vcl_avg, kcen_vcl_avg)
