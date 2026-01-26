import copy
import json


class CfgNode:
    def __init__(self, init_dict=None):
        self.__dict__["_data"] = {}
        if init_dict:
            for k, v in init_dict.items():
                self[k] = v

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No attribute '{name}' in config.")

    def __setattr__(self, name, value):
        self._data[name] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = CfgNode(value)
        self._data[key] = value

    def merge_from_dict(self, cfg_dict):
        """Recursively merge keys from a dict into this config."""
        for k, v in cfg_dict.items():
            if (
                k in self._data
                and isinstance(self._data[k], CfgNode)
                and isinstance(v, dict)
            ):
                self._data[k].merge_from_dict(v)
            else:
                self[k] = v

    def to_dict(self):
        """Recursively convert to a regular dict."""
        result = {}
        for k, v in self._data.items():
            if isinstance(v, CfgNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def clone(self):
        new_cfg = CfgNode()
        new_cfg.merge_from_dict(self.to_dict())
        return new_cfg

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class SmartPrintConfig:
    """static class for storing the configuration of the smart print function"""

    log_file = None
    print_to_console = True


class CommonCallbackConfig:
    """static class for storing the configuration of the common callback functions"""

    save_model_every_n_epochs = 1
    save_model_dir = ""  # by default, save the model to the current working directory
