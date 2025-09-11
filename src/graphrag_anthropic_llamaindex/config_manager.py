import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Loads the configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please copy 'config/config.example.yaml' to 'config/config.yaml' and set your API key.")
        return None
