"""
Validate configuration files to ensure they are correctly formatted.
"""

import os
import sys
import argparse
from pathlib import Path

from utils import load_config, validate_config

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YAML configuration files")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to specific config file to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all config files in configs/ directory",
    )
    return parser.parse_args()

def validate_single_config(config_path: str) -> bool:
    """Validate a single configuration file."""
    print(f"\nValidating: {config_path}")
    print("-" * 50)
    
    try:
        config = load_config(config_path)
        validate_config(config)
        print("✅ Configuration is valid!")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed!")
        print(f"Error: {str(e)}")
        return False

def validate_all_configs():
    """Validate all configuration files in configs/ directory."""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("❌ configs/ directory not found!")
        return False
    
    # Find all YAML files except accelerate and deepspeed configs
    yaml_files = [
        f for f in configs_dir.glob("*.yaml")
        if f.name not in ["accelerate_config.yaml", "template.yaml"]
    ]
    
    if not yaml_files:
        print("❌ No YAML configuration files found in configs/")
        return False
    
    print(f"Found {len(yaml_files)} configuration files to validate")
    
    results = {}
    for config_file in yaml_files:
        results[config_file.name] = validate_single_config(str(config_file))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    for config_name, is_valid in results.items():
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        print(f"{config_name:30s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print("-" * 50)
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 50)
    
    return failed == 0

def main():
    args = parse_args()
    
    if args.all:
        success = validate_all_configs()
    elif args.config:
        success = validate_single_config(args.config)
    else:
        print("Please specify --config <path> or --all")
        print("\nExamples:")
        print("  python validate_config.py --config configs/glue_mrpc.yaml")
        print("  python validate_config.py --all")
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
