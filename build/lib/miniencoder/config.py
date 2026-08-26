import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(config):
    dropout = config.get("model", {}).get("dropout", 0.0)
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    return config
