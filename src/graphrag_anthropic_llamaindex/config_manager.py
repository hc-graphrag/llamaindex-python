import yaml
import os
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Loads the configuration from a YAML file and resolves paths relative to config file location."""
    try:
        config_path = Path(config_path).resolve()
        config_dir = config_path.parent

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Resolve relative paths in config relative to config file location
        if config:
            config = _resolve_relative_paths(config, config_dir)

        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please copy 'config/config.example.yaml' to 'config/config.yaml' and set your API key.")
        return None

def _resolve_relative_paths(config, config_dir):
    """Resolve relative paths in config relative to config file directory."""
    if not isinstance(config, dict):
        return config

    resolved_config = {}

    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = _resolve_relative_paths(value, config_dir)
        elif isinstance(value, str) and _is_relative_path_key(key):
            # URL（AlopexDB サーバ等）はパスとして解決せずそのまま使う
            if value.startswith(('http://', 'https://')):
                resolved_config[key] = value
            # Resolve relative paths
            elif value.startswith('./') or not os.path.isabs(value):
                resolved_path = config_dir / value
                resolved_config[key] = str(resolved_path.resolve())
            else:
                resolved_config[key] = value
        else:
            resolved_config[key] = value

    return resolved_config

def _is_relative_path_key(key):
    """Check if a config key represents a path that should be resolved."""
    path_keys = {
        'input_dir', 'output_dir', 'uri', 'data_dir', 'storage_dir',
        'database_path', 'index_path', 'cache_dir', 'log_dir'
    }
    return key in path_keys
