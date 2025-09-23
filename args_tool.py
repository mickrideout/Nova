import logging
import yaml
from pathlib import Path
import argparse

# Initialize logger
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        raise


def dict_to_namespace(config: dict) -> argparse.Namespace:
    """
    Convert a dictionary to an argparse.Namespace object.

    Args:
        config (dict): Configuration parameters as a dictionary.

    Returns:
        argparse.Namespace: Configuration parameters as a Namespace object.
    """
    return argparse.Namespace(**config)


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments and load the appropriate configuration file.

    Returns:
        argparse.Namespace: Configuration parameters.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load configuration based on environment.")
    parser.add_argument("env", choices=["test", "dev", "prd"], help="Environment to load configuration for.")
    args = parser.parse_args()

    # Determine the config file based on the environment
    config_file = f"config.{args.env}.yaml"

    # Load configuration from the YAML file
    config = load_config(config_file)

    # Log the loaded configuration
    logger.info(f"Loaded configuration: {config}")

    # Convert the dictionary to an argparse.Namespace object
    args = dict_to_namespace(config)

    return args


if __name__ == "__main__":
    # Execute the get_args function when the script is run
    args = get_args()
    print(args)  # For debugging purposes