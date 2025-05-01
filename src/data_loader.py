"""Module for evaluating Survey Assist SIC/SOC code assignments.

This script loads configuration from 'config.toml', reads a gold standard dataset,
and provides classes to manage the evaluation process.

Usage:
------
1. Ensure 'config.toml' is present in the project root directory.
2. Set the following environment variables before running:
   - API_GATEWAY: URL for the API gateway (used for token generation).
   - SA_EMAIL: Service account email address for authentication.
   - JWT_SECRET: Path to the JWT secret file or the secret key itself.
3. Run the script:  python /home/user/survey-assist-utils/src/data_loader.py

Classes:
--------
    AppConfig: Loads and holds configuration settings from TOML file.
    GoldStandardLoader: Loads and validates the gold standard dataset.
    # Placeholder for future evaluation classes...

"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- FIX: Adjusted typing imports ---
# Removed List, Dict. Kept Optional, Any. Added Union for modern hints.
from typing import (  # Union is needed for Optional[list[...]] -> list[...] | None
    Any,
    Optional,
)

import pandas as pd
import tomllib  # Requires Python 3.11+; use 'toml' package for older versions

# --- Global Variables / Setup ---
try:
    _script_path = Path(__file__).resolve()
    PROJECT_ROOT = (
        _script_path.parent
        if _script_path.parent.name != "src"
        else _script_path.parent.parent
    )
except NameError:
    PROJECT_ROOT = Path.cwd()

CONFIG_FILE_PATH = PROJECT_ROOT / "config.toml"
logger = logging.getLogger(__name__)

# --- Configuration Class ---


@dataclass(frozen=True)
class AppConfig:
    """Loads and holds configuration settings."""

    # --- FIX: Use modern dict hint ---
    config_data: dict[str, Any]
    project_root: Path = PROJECT_ROOT

    @classmethod
    def load_from_toml(cls, config_path: Path = CONFIG_FILE_PATH) -> "AppConfig":
        """Loads configuration from the specified TOML file."""
        logger.info(f"Loading configuration from: {config_path}")
        if not config_path.is_file():
            logger.critical(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(
                f"Configuration file '{config_path.name}' not found at {config_path}"
            )

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            logger.info("Configuration loaded successfully.")
            return cls(config_data=data)
        except tomllib.TOMLDecodeError as e:
            logger.exception(f"Error decoding TOML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read or load config from {config_path}: {e}")
            raise

    @property
    # --- FIX: Use modern dict hint ---
    def paths(self) -> dict[str, str]:
        """Returns the [paths] section from the config."""
        return self.config_data.get("paths", {})

    @property
    # --- FIX: Use modern dict hint ---
    def column_names(self) -> dict[str, str]:
        """Returns the [column_names] section from the config."""
        return self.config_data.get("column_names", {})

    @property
    # --- FIX: Use modern dict hint ---
    def logging_config(self) -> dict[str, str]:
        """Returns the [logging] section from the config."""
        return self.config_data.get("logging", {})

    # Optional type hint allows Path or None
    def get_path(self, key: str) -> Optional[Path]:
        """Gets a path from [paths] config and resolves it relative to project root."""
        relative_path_str = self.paths.get(key)
        if relative_path_str:
            # Ensure path is treated as string before joining
            return (self.project_root / str(relative_path_str)).resolve()
        logger.warning(f"Path key '{key}' not found in [paths] section of config.")
        return None

    def get_column_name(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Gets a column name from [column_names] config."""
        col_name = self.column_names.get(key, default)
        if col_name is None and default is None:
            logger.warning(
                f"Column name key '{key}' not found in [column_names] section of config."
            )
        return col_name


# --- Logging Setup Function ---


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """Configures logging based on AppConfig or defaults."""
    log_level_str = "INFO"
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    log_file: Optional[Path] = None  # Explicitly type hint log_file

    if config:
        log_conf = config.logging_config
        log_level_str = log_conf.get("level", "INFO").upper()
        log_format = log_conf.get("format", log_format)
        log_file_rel = log_conf.get("file")
        if log_file_rel:
            log_file = (config.project_root / str(log_file_rel)).resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, log_level_str, logging.INFO)

    # --- FIX: Explicitly type hint log_handlers (addresses potential line 171 issue) ---
    # This list will contain logging Handler objects
    log_handlers: list[logging.Handler] = []
    if log_file:
        # Appending FileHandler is compatible with list[logging.Handler]
        log_handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # Appending StreamHandler is compatible with list[logging.Handler]
    log_handlers.append(logging.StreamHandler(sys.stdout))

    # logging.basicConfig expects handlers: list[logging.Handler] | None
    logging.basicConfig(
        level=log_level, format=log_format, handlers=log_handlers, force=True
    )
    logger.info(
        f"Logging configured. Level: {log_level_str}. File: {log_file or 'Console'}"
    )


# --- Data Loading Class ---


@dataclass
class GoldStandardLoader:
    """Loads and validates the gold standard dataset based on configuration."""

    config: AppConfig
    # Use Optional with modern Union syntax ( | ) if needed, or keep as is if always DataFrame
    _data: Optional[pd.DataFrame] = field(init=False, repr=False, default=None)

    # --- FIX: Use modern list hint ---
    def _get_required_cols(self) -> list[str]:
        """Gets the list of required columns from config."""
        cols = [
            self.config.get_column_name("unique_id"),
            self.config.get_column_name("gold_sic"),
            self.config.get_column_name("gold_flag"),
        ]
        # Filter out None values in case config keys were missing
        return [col for col in cols if col is not None]

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """Loads the gold standard CSV file specified in the config."""
        if self._data is not None and not force_reload:
            logger.info("Returning cached gold standard data.")
            return self._data.copy()

        file_path = self.config.get_path("gold_standard_csv")
        if not file_path:
            # Log critical error if path is essential
            logger.critical(
                "Path 'gold_standard_csv' not defined in config [paths] or is invalid."
            )
            raise ValueError("Path 'gold_standard_csv' not defined in config [paths].")

        logger.info(f"Loading gold standard data from: {file_path}")
        required_cols = self._get_required_cols()
        if not required_cols:  # Check if list is empty
            raise ValueError(
                "No required gold standard column names found in config [column_names]."
            )

        try:
            df = pd.read_csv(
                file_path,
                delimiter=",",
                dtype=str,
                na_filter=False,
                usecols=required_cols,
            )
            logger.info(f"Loaded {len(df)} gold standard rows.")

            # Validation already implicitly done by usecols raising ValueError if cols missing
            # Add explicit check if needed for clarity or other reasons:
            # if not all(col in df.columns for col in required_cols):
            #     missing = set(required_cols) - set(df.columns)
            #     raise ValueError(f"Gold standard file loaded but missing required columns: {missing}")

            self._data = df
            return df.copy()

        except FileNotFoundError:
            logger.exception(f"Gold standard file not found: {file_path}")
            raise
        except ValueError as ve:
            # Catches errors from usecols if columns don't exist in CSV
            logger.exception(
                f"Error loading gold standard columns (check config/CSV header): {ve}"
            )
            raise
        except Exception as e:
            logger.exception(f"Error loading gold standard file {file_path}: {e}")
            raise

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Returns the loaded data (loads if not already loaded). Returns None if loading fails."""
        if self._data is None:
            try:
                self.load()
            except (FileNotFoundError, ValueError, Exception):
                logger.error("Failed to load gold standard data on first access.")
                return None
        # Ensure _data is DataFrame before copy, otherwise return None
        return self._data.copy() if isinstance(self._data, pd.DataFrame) else None


# --- FIX for Line 150 (Append Error - General Advice) ---
# The error "argument 1 to 'append' of 'list' has incompatible type" usually means:
# 1. You have a list hinted like: `my_list: list[str] = []`
# 2. You try to append something that ISN'T a string: `my_list.append(None)` or `my_list.append(123)`
# TO FIX:
# a) Ensure you only append the correct type (e.g., convert to string `my_list.append(str(123))`).
# b) OR, adjust the type hint if `None` or other types are allowed:
#    `my_list: list[Optional[str]] = []` (using Python 3.9+ Union syntax: `list[str | None] = []`)
#    Then `my_list.append(None)` becomes valid.
# Check the code around line 150 (or wherever the error actually occurs) for list appends.

# --- Main Execution Block ---


def main():
    """Main script execution."""
    # 1. Load Configuration
    try:
        app_config = AppConfig.load_from_toml()
    except (FileNotFoundError, tomllib.TOMLDecodeError, Exception):
        print(
            f"\nFATAL: Could not load configuration from {CONFIG_FILE_PATH}. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Setup Logging
    setup_logging(app_config)
    logger.info("--- Starting SIC/SOC Evaluator Script ---")
    logger.info(f"Project Root: {PROJECT_ROOT}")

    # 3. Check for Required Environment Variables
    api_gateway = os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    jwt_secret_path = os.getenv("JWT_SECRET")
    if not all([api_gateway, sa_email, jwt_secret_path]):
        logger.critical(
            "Missing required environment variables: API_GATEWAY, SA_EMAIL, JWT_SECRET. Exiting."
        )
        sys.exit(1)
    logger.info("Required environment variables (secrets) found.")

    # 4. Load Gold Standard Data
    try:
        loader = GoldStandardLoader(config=app_config)
        gold_df = loader.load()
        if gold_df is not None and not gold_df.empty:
            logger.info(
                f"Successfully loaded gold standard data. Shape: {gold_df.shape}"
            )
            # logger.debug(f"Gold standard head:\n{gold_df.head()}") # Keep debug if needed
        else:
            logger.error(
                "Gold standard data is empty or failed to load. Cannot proceed."
            )
            sys.exit(1)  # Exit if gold standard is essential

    except (FileNotFoundError, ValueError, Exception) as e:
        logger.critical(
            f"Failed to load gold standard data required for evaluation. Exiting. Error: {e}"
        )
        sys.exit(1)

    # --- Placeholder for next steps ---
    logger.info("--- Script Execution Finished (Placeholder) ---")


if __name__ == "__main__":
    main()
