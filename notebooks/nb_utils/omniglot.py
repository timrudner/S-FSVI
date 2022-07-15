from notebooks.nb_utils.common import read_config_and_run, show_final_average_accuracy
import sfsvi.exps.utils.load_utils as lutils

task_sequence = "omniglot"


# `S-FSVI (ours)`
logdir = read_config_and_run("fsvi_omniglot.pkl", task_sequence)
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp)