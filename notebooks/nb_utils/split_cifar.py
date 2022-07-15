import sfsvi.exps.utils.load_utils as lutils
from notebooks.nb_utils.common import read_config_and_run, show_final_average_accuracy

task_sequence = "cifar"

# `S-FSVI (ours)`
logdir = read_config_and_run("fsvi_cifar.pkl", task_sequence)
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp)
