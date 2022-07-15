"""Helper class for using arguments registered with `argparse` for
    generating experiment configs."""
import argparse
from typing import Dict
from typing import List

from sfsvi.exps.utils.generate_cmds import ADD_ARGS_FN_TYPE
from sfsvi.exps.utils.generate_cmds import ENTRY_POINT
from sfsvi.exps.utils.generate_cmds import generate_bash_file


class ConfigTemplate:
    """Helper class for using arguments registered with `argparse` for
    generating experiment configs."""

    KEYS_TO_IGNORE = {"template"}

    def __init__(self, add_args_fn: ADD_ARGS_FN_TYPE):
        self.add_args_fn_name = add_args_fn.__name__
        self._parser = self._extract_actions(add_args_fn=add_args_fn)
        self.bool_fields_store_val = self._extract_boolean_info_from_parser(
            parser=self._parser
        )

    @staticmethod
    def _extract_actions(add_args_fn: ADD_ARGS_FN_TYPE) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_args_fn(parser)
        return parser

    def default_config(self) -> Dict:
        config = {}
        for a in self._parser._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            config[a.dest] = "<required_field>" if a.required else a.default
        return config

    def match_template(self, cmd_str: str) -> bool:
        try:
            self._parser.parse_args(cmd_str.split(" "))
        except SystemExit as e:
            print(f"Error happened while matching:\n {e}")
            return False
        else:
            return True

    def parse_args(self, cmd_str: str, to_dict: bool = True, template_info: bool = False,
                   ignore_unknown: bool = False) -> Dict:
        # split into a list of strings, remove "\n" and empty strings
        cmd = list(filter(lambda x: x != "", cmd_str.replace("\n", "").split(" ")))
        if ignore_unknown:
            args = self._parser.parse_known_args(cmd)[0]
        else:
            args = self._parser.parse_args(cmd)
        if to_dict:
            args = vars(args)
        if template_info:
            args["template"] = self.add_args_fn_name
        return args

    def _format_bool_field(self, k, v: bool) -> str:
        assert isinstance(v, bool), f"v is not of type boolean: {v}"
        if self.bool_fields_store_val[k] and v:
            return f"--{k}"
        else:
            return ""

    def config_to_str(self, config: Dict) -> str:
        cmd = []
        for k, v in config.items():
            if k in self.KEYS_TO_IGNORE or v is None:
                continue
            if isinstance(v, bool):
                cmd.append(self._format_bool_field(k, v))
            elif isinstance(v, List):
                cmd.append(f"--{k} " + " ".join(map(str, v)))
            else:
                cmd.append(f"--{k} {v}")
        cmd = list(filter(lambda x: x != "", cmd))
        cmd_str = " ".join(cmd)
        if not self.match_template(cmd_str=cmd_str):
            print("WARNING! The formulated command string can't be parsed!")
        return cmd_str

    def configs_to_file(
        self, configs: List[Dict], file_path: str, prefix=ENTRY_POINT, mode="w"
    ):
        cmd_strs = self.configs_to_strings(configs=configs, prefix=prefix)
        generate_bash_file(file_path=file_path, cmd_strs=cmd_strs, mode=mode)

    def configs_to_strings(self, configs: List[Dict], prefix: str = "") -> List[str]:
        return [" ".join([prefix, self.config_to_str(c)]) for c in configs]

    @staticmethod
    def _extract_boolean_info_from_parser(
        parser: argparse.ArgumentParser,
    ) -> Dict[str, bool]:
        bool_fields_store_val = {}
        for a in parser._actions:
            if isinstance(a, argparse._StoreTrueAction):
                bool_fields_store_val[a.dest] = True
            elif isinstance(a, argparse._StoreFalseAction):
                bool_fields_store_val[a.dest] = False
        return bool_fields_store_val
