"""This module loads the config."""

import toml


# Load the config:
def load_config(config_path):
    """Loads configuration settings from a .toml file.

    Args:
        config_path (str): The path to the .toml configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Example:
        config = load_config("config.toml")
        print(config["settings"]["api_gateway"])
    """
    with open(config_path, encoding="utf-8") as file:
        configuration = toml.load(file)
    return configuration
