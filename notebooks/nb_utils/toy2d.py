import os
import pickle

import sfsvi.exps.utils.load_utils as lutils
from notebooks.nb_utils.toy_2d_plot_utils import plot_toy_2d
from notebooks.nb_utils.common import root
from cli import run_config


path = os.path.join(root, "notebooks/configs/toy2d.pkl")
with open(path, "rb") as p:
	config = pickle.load(p)

logdir = run_config(config)
plot_toy_2d(lutils.read_exp(logdir))
