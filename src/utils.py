import logging
import yaml
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
