from sfsvi.exps.utils.generate_cmds import (
    generate_configs,
)
from sfsvi.exps.utils.config_template import ConfigTemplate
from sfsvi.fsvi_utils.sfsvi_args import add_sfsvi_args


template = ConfigTemplate(add_args_fn=add_sfsvi_args)
print(template.default_config())
base_config = {
    "data_training": "not_specified",
    "data_ood": "not_specified",
    "model": "not_specified",
    "optimizer": "adam",
    "architecture": "not_specified",
    "activation": "not_specified",
    "prior_mean": "0",
    "prior_cov": "0",
    "prior_type": "not_specified",
    "epochs": 100,
    "batch_size": 100,
    "learning_rate": 0.001,
    "dropout_rate": 0.0,
    "regularization": 0,
    "n_inducing_inputs": 0,
    "inducing_input_type": "not_specified",
    "kl_scale": "1",
    "full_cov": False,
    "n_samples": 1,
    "tau": 1,
    "noise_std": 1,
    "inducing_inputs_bound": [0.0, 0.0],
    "logging_frequency": 10,
    "figsize": [10, 4],
    "seed": 0,
    "save_path": "debug",
    "save": False,
    "resume_training": False,
    "debug": False,
    "logroot": None,
    "subdir": None,
    "inducing_inputs_add_mode": 0,
    "n_samples_eval": 5,
    "plotting": False,
    "logging": 1,
    "use_val_split": True,
    "coreset": "random",
    "coreset_entropy_mode": None,
    "coreset_entropy_offset": 0.0,
    "coreset_entropy_n_mixed": 1,
}
string = template.config_to_str(base_config)
print(template.match_template(string))
print(string)

configs = [base_config]
configs.extend(
    generate_configs(base_configs=base_config, key="learning_rate", values=[0.1, 0.01])
)

template.configs_to_file(configs=configs, file_path="./jobs/test")
