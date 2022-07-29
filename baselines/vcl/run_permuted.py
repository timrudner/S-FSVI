import os
import sys

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_folder)
sys.path.insert(0, os.path.join(root_folder, "function_space_vi"))
import numpy as np
import tensorflow as tf

from baselines.vcl.alg.data_generator import PermutedMnistGenerator, CustomGenerator

tf.compat.v1.disable_eager_execution()
import pickle
import sys

sys.path.extend(["alg/"])
from baselines.vcl.alg import vcl, coreset, utils

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

# Run vanilla VCL
tf.random.set_seed(12)
np.random.seed(1)

coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
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
"""
array([[0.9551,    nan,    nan,    nan,    nan],
       [0.8959, 0.931 ,    nan,    nan,    nan],
       [0.8645, 0.8857, 0.9293,    nan,    nan],
       [0.7912, 0.8758, 0.8994, 0.9284,    nan],
       [0.7566, 0.7895, 0.8942, 0.9117, 0.9322]])
"""

# Run random coreset VCL
tf.compat.v1.reset_default_graph()
tf.random.set_seed(12)
np.random.seed(1)

coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
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
tf.compat.v1.set_random_seed(12)
np.random.seed(1)

data_gen = PermutedMnistGenerator(num_tasks)
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
with open(RESULT_FOLDER / "10pmnist_fixed", "wb") as p:
    pickle.dump(to_save, p)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot("results/permuted.jpg", vcl_avg, rand_vcl_avg, kcen_vcl_avg)
