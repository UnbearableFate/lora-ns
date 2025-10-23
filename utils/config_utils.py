"""
Configuration utilities for loading and managing YAML configs.
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {save_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has required fields."""
    required_fields = ["task_name", "task_type", "model", "peft", "dataset", "training"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate model config
    if "name_or_path" not in config["model"]:
        raise ValueError("Missing 'name_or_path' in model config")
    
    # Validate PEFT config
    if "method" not in config["peft"]:
        raise ValueError("Missing 'method' in peft config")
    
    # Validate dataset config
    if "name" not in config["dataset"]:
        raise ValueError("Missing 'name' in dataset config")
    
    # Validate training config
    if "output_dir" not in config["training"]:
        raise ValueError("Missing 'output_dir' in training config")
    
    logger.info("Config validation passed")
    return True


def get_output_dir(config: Dict[str, Any]) -> str:
    """Get output directory from config."""
    return config["training"]["output_dir"]


def get_model_name(config: Dict[str, Any]) -> str:
    """Get model name from config."""
    return config["model"]["name_or_path"]


def get_task_type(config: Dict[str, Any]) -> str:
    """Get task type from config."""
    return config["task_type"]


def print_config(config: Dict[str, Any]):
    """Pretty print configuration."""
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info("=" * 50)
    logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))
    logger.info("=" * 50)
