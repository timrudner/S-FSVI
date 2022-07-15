import sfsvi.exps.utils.load_utils as lutils
from notebooks.nb_utils.common import read_config_and_run
from notebooks.nb_utils.common import show_final_average_accuracy

task_sequence = "smnist_mh"


# `S-FSVI (ours)`
logdir = read_config_and_run("fsvi_match.pkl", task_sequence)
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp)


# `S-FSVI (larger networks)`
logdir = read_config_and_run("fsvi_optimized.pkl", task_sequence)
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp)


# `S-FSVI (no coreset)`
logdir = read_config_and_run("fsvi_no_coreset.pkl", task_sequence)
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp)


# `VCL (random_choice coreset)`
logdir = read_config_and_run("vcl_random_coreset.pkl", task_sequence, "vcl")
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp, runner="vcl")