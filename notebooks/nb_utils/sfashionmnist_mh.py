from notebooks.nb_utils.common import read_config_and_run, show_final_average_accuracy
import sfsvi.exps.utils.load_utils as lutils

task_sequence = "sfashionmnist_mh"


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


# `FRCL (with random_choice coreset)`
logdir = read_config_and_run("frcl_with_coreset.pkl", task_sequence, "frcl")
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp, runner="frcl")


# `FROMP (with lambda_descend coreset)`
logdir = read_config_and_run("fromp_with_coreset.pkl", task_sequence, runner="fromp")
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp, runner="fromp")


# `VCL (random_choice coreset)`
logdir = read_config_and_run("vcl_random_coreset.pkl", task_sequence, runner="vcl")
exp = lutils.read_exp(logdir)
show_final_average_accuracy(exp, runner="vcl")
