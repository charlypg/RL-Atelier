import datetime
from omegaconf import OmegaConf, DictConfig
import os
import json
import hashlib

def load_config_from_path(hydra_config: DictConfig, field: str) -> DictConfig:
    config_path = hydra_config.get(field)
    if config_path is not None:
        return OmegaConf.load(os.path.abspath(config_path))
    else:
        return None

def merge_configs(acu_cfg: DictConfig, overriding_cfg: DictConfig) -> DictConfig:
    assert acu_cfg is not None
    if overriding_cfg is not None:
        # No custom replacements for the moment

        # Merge
        acu_cfg = OmegaConf.merge(acu_cfg, overriding_cfg)
    return acu_cfg

def merge_base_variant_cli(hydra_config: DictConfig) -> DictConfig:
    # Load base config
    cfg = load_config_from_path(hydra_config, "base")

    # Load variant and merge
    variant_cfg = load_config_from_path(hydra_config, "variant")
    cfg = merge_configs(cfg, variant_cfg)

    # get cli overrides and merge
    cli_overrides = hydra_config.get("overrides")
    cfg = merge_configs(cfg, cli_overrides)

    return cfg

def get_str_date() -> str:
    date = datetime.datetime.now()
    return "{0}_{1}_{2}_{3}_{4}".format(
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
    )

def resolve_tuple(*args):
    """Converts yaml string to tuple"""
    return tuple(args)

def dict_hash(dict_config: dict) -> str:
    # Convert dictionary to a JSON string with sorted keys
    dict_str = json.dumps(dict_config, sort_keys=True)
    # Encode the string to bytes
    dict_bytes = dict_str.encode('utf-8')
    # Create a hash (you can use sha256, md5, etc.)
    return hashlib.sha256(dict_bytes).hexdigest()

def get_no_seed_dict_config(dict_config: dict) -> dict:
    dict_config_no_seed = dict_config.copy()
    if "seed" in dict_config_no_seed:
        dict_config_no_seed.pop("seed")
    else: 
        raise KeyError("No 'seed' in config.")
    return dict_config_no_seed

def two_hashes_from_dict(dict_config: dict) -> dict:
    dict_config_no_seed = get_no_seed_dict_config(dict_config)
    return {
        "hash_no_seed": dict_hash(dict_config_no_seed),
        "hash": dict_hash(dict_config),
        "config": dict_config,
        "config_no_seed": dict_config_no_seed
    }
