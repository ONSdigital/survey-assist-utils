import logging
import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer plots

# --- Configuration & Constants ---

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    filename='sic_evaluation.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# --- Default Column Names (Adjust if needed) ---
# Consider moving these to a config file or defining more robustly
COL_UNIQUE_ID = 'unique_id'
COL_GOLD_SIC = 'sic_ind_occ1' # Gold standard SIC code column
COL_GOLD_FLAG = 'sic_ind_code_flag' # Gold standard KB/CC/MC flag
COL_RESPONSE_CANDIDATES = 'sic_candidates' # Column in response JSON with candidates list
COL_RESPONSE_TOP_SIC = 'sic_code' # Top level sic_code in response JSON (if present)

COL_LLM_TOP_SIC = 'llm_sic_code_1' # Renamed for clarity after extraction
COL_LLM_TOP_LIKELIHOOD = 'llm_likelihood_1'

OUTPUT_DIR = 'data/analysis_outputs' # Centralize output directory

# --- Data Loading ---

def load_gold_standard(file_path: str, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Loads the gold standard CSV file."""
    logger.info(f"Loading gold standard data from: {file_path}")
    if required_cols is None:
        required_cols = [COL_UNIQUE_ID, COL_GOLD_SIC, COL_GOLD_FLAG]

    try:
        df = pd.read_csv(
            file_path,
            delimiter=',', # Assuming comma based on previous examples
            dtype=str,
            na_filter=False, # Treat empty fields as ""
            usecols=lambda col_name: col_name in required_cols # Load only needed columns
        )
        # Validate required columns are present after loading
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Gold standard file missing required columns: {missing}")
        logger.info(f"Loaded {len(df)} gold standard rows.")
        return df
    except FileNotFoundError:
        logger.exception(f"Gold standard file not found: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Error loading gold standard file {file_path}: {e}")
        raise

def load_llm_responses(file_path: str) -> pd.DataFrame:
    """Loads LLM responses from a JSON Lines file."""
    logger.info(f"Loading LLM responses from: {file_path}")
    data = []
    line_num = 0
    errors = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_num += 1
            try:
                # Add unique_id extraction directly if it's top-level JSON key
                loaded_json = json.loads(line.strip())
                data.append(loaded_json)
            except json.JSONDecodeError:
                errors += 1
                logger.warning(f"Skipping line {line_num} due to JSON decode error.")
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping line {line_num} due to unexpected error: {e}")

    if errors > 0:
        logger.warning(f"Encountered {errors} errors while loading responses.")

    if not data:
        logger.warning("No valid response data loaded.")
        # Return empty DataFrame with expected columns if needed downstream
        return pd.DataFrame(columns=[COL_UNIQUE_ID, COL_RESPONSE_CANDIDATES, COL_RESPONSE_TOP_SIC]) # Add other expected cols

    # Initial normalization - might need adjustment based on full structure
    # If unique_id isn't nested, json_normalize might not be best initially
    # Let's create DataFrame directly first
    df = pd.DataFrame(data)
    # Ensure unique_id exists
    if COL_UNIQUE_ID not in df.columns:
         raise ValueError(f"Unique ID column '{COL_UNIQUE_ID}' not found in response data.")

    logger.info(f"Loaded {len(df)} LLM response rows.")
    return df


# --- Data Processing & Merging ---

def merge_data(gold_df: pd.DataFrame, response_df: pd.DataFrame) -> pd.DataFrame:
    """Merges gold standard and response dataframes on unique_id."""
    logger.info("Merging gold standard and response data...")
    # Use inner merge to only keep IDs present in both datasets
    merged_df = pd.merge(
        gold_df,
        response_df,
        on=COL_UNIQUE_ID,
        how='inner', # Ensures we only evaluate where we have both gold and response
        suffixes=('_gold', '_resp') # Add suffixes if any overlapping columns exist besides ID
    )
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    if len(merged_df) == 0:
         logger.warning("Merge resulted in an empty DataFrame. Check unique IDs.")
    elif len(merged_df) < len(gold_df) or len(merged_df) < len(response_df):
         logger.warning(f"Merge resulted in fewer rows ({len(merged_df)}) than input gold ({len(gold_df)}) or responses ({len(response_df)}). Some unique IDs may not have matched.")
    return merged_df


def extract_llm_predictions(df: pd.DataFrame, max_candidates: int = 5) -> pd.DataFrame:
    """
    Extracts top N LLM predictions and likelihoods from the 'sic_candidates' column.

    Args:
        df (pd.DataFrame): DataFrame containing the response data (expects
                           COL_RESPONSE_CANDIDATES column with list of dicts).
        max_candidates (int): Maximum number of candidates to extract.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for top N predictions
                      (e.g., llm_sic_code_1, llm_likelihood_1, ... llm_sic_code_N, llm_likelihood_N).
    """
    logger.info(f"Extracting top {max_candidates} LLM predictions...")

    def safe_extract(candidates: Any, index: int, key: str) -> Optional[str]:
        """Helper to safely extract data from the candidates list."""
        # Check if candidates is a list and index is valid
        if isinstance(candidates, list) and index < len(candidates):
            # Check if the item at index is a dict and has the key
            if isinstance(candidates[index], dict) and key in candidates[index]:
                return str(candidates[index][key]) # Ensure string output
        return None # Return None if any check fails

    if COL_RESPONSE_CANDIDATES not in df.columns:
        logger.warning(f"Column '{COL_RESPONSE_CANDIDATES}' not found. Skipping prediction extraction.")
        # Add empty columns if needed downstream
        for i in range(1, max_candidates + 1):
            df[f'llm_sic_code_{i}'] = None
            df[f'llm_likelihood_{i}'] = None
        return df

    for i in range(1, max_candidates + 1):
        col_idx = i - 1 # 0-based index for list access
        df[f'llm_sic_code_{i}'] = df[COL_RESPONSE_CANDIDATES].apply(lambda c: safe_extract(c, col_idx, 'sic_code'))
        df[f'llm_likelihood_{i}'] = df[COL_RESPONSE_CANDIDATES].apply(lambda c: safe_extract(c, col_idx, 'likelihood'))
        # Convert likelihood to numeric, coercing errors to NaN
        df[f'llm_likelihood_{i}'] = pd.to_numeric(df[f'llm_likelihood_{i}'], errors='coerce')

    # Ensure the primary column exists even if candidates list was short/missing
    if COL_LLM_TOP_SIC not in df.columns:
         df[COL_LLM_TOP_SIC] = None
    if COL_LLM_TOP_LIKELIHOOD not in df.columns:
         df[COL_LLM_TOP_LIKELIHOOD] = np.nan

    # Optional: Verify if top-level 'sic_code' matches 'llm_sic_code_1' if both exist
    # ... add verification logic if desired ...

    logger.info("Finished extracting LLM predictions.")
    return df


def calculate_digit_matches(df: pd.DataFrame,
                            gold_col: str = COL_GOLD_SIC,
                            predicted_col: str = COL_LLM_TOP_SIC,
                            match_level_col: str = 'match_level'
                            ) -> pd.DataFrame:
    """
    Calculates the number of matching leading digits between two columns.

    Args:
        df (pd.DataFrame): DataFrame containing the gold and predicted codes.
        gold_col (str): Column name for the gold standard SIC code.
        predicted_col (str): Column name for the LLM's predicted SIC code.
        match_level_col (str): Name for the new column storing the match level (0-5).

    Returns:
        pd.DataFrame: DataFrame with the added match_level column.
    """
    logger.info(f"Calculating leading digit matches between '{gold_col}' and '{predicted_col}'...")

    # Ensure columns exist
    if not all(col in df.columns for col in [gold_col, predicted_col]):
        logger.error(f"Required columns '{gold_col}' or '{predicted_col}' not found. Skipping digit match calculation.")
        df[match_level_col] = -1 # Indicate error or skip
        return df

    # Fill potential None/NaN with empty strings for safe string operations
    s_gold = df[gold_col].fillna('').astype(str)
    s_pred = df[predicted_col].fillna('').astype(str)

    # Calculate matches efficiently using vectorized operations
    match_level = np.zeros(len(df), dtype=int) # Start with 0 matches

    for i in range(1, 6): # Check matches for 1 to 5 digits
        # Only update match_level where strings are long enough AND prefixes match
        mask = (s_gold.str.len() >= i) & \
               (s_pred.str.len() >= i) & \
               (s_gold.str[:i] == s_pred.str[:i])
        match_level[mask] = i # Update level for rows that match at this length

    df[match_level_col] = match_level
    logger.info(f"Added '{match_level_col}' column with digit match levels (0-5).")
    return df

# --- Analysis & Plotting ---

def plot_digit_match_histogram(df: pd.DataFrame,
                               match_level_col: str = 'match_level',
                               stratify_by: Optional[str] = COL_GOLD_FLAG,
                               output_dir: str = OUTPUT_DIR,
                               filename_prefix: str = 'digit_match_histogram'):
    """
    Generates and saves a histogram of digit match levels, optionally stratified.

    Args:
        df (pd.DataFrame): DataFrame containing the 'match_level' column and
                           optionally a stratification column.
        match_level_col (str): Name of the column with match levels (0-5).
        stratify_by (Optional[str]): Column name to group/stratify the histogram by
                                     (e.g., 'sic_ind_code_flag'). If None, plots overall distribution.
        output_dir (str): Directory to save the plot image.
        filename_prefix (str): Prefix for the output plot filename.
    """
    if match_level_col not in df.columns:
        logger.error(f"Match level column '{match_level_col}' not found. Cannot generate histogram.")
        return
    if stratify_by and stratify_by not in df.columns:
        logger.warning(f"Stratification column '{stratify_by}' not found. Plotting overall distribution.")
        stratify_by = None # Reset stratification if column missing

    logger.info(f"Generating digit match histogram. Stratify by: {stratify_by}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plot_title = f'Distribution of Matching Leading Digits (N={len(df)})'
    order = list(range(6)) # Order bars from 0 to 5

    if stratify_by:
        # Plot counts stratified by the flag column
        sns.countplot(data=df, x=match_level_col, hue=stratify_by, order=order, palette='viridis')
        plot_title += f' by {stratify_by}'
    else:
        # Plot overall counts
        sns.countplot(data=df, x=match_level_col, order=order, color='skyblue')

    plt.title(plot_title)
    plt.xlabel("Number of Matching Leading Digits (0 = No Match on First Digit)")
    plt.ylabel("Count of Responses")
    plt.xticks(ticks=order, labels=[str(i) for i in order]) # Ensure labels are 0-5
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strat_suffix = f"_by_{stratify_by}" if stratify_by else ""
    output_filename = f"{filename_prefix}{strat_suffix}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    try:
        plt.savefig(output_path)
        logger.info(f"Histogram saved to {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save histogram plot: {e}")
    plt.close() # Close the plot to free memory


# --- Main Orchestration ---

def run_evaluation(gold_standard_path: str,
                   response_path: str,
                   output_prefix: str = 'evaluation',
                   max_candidates_extract: int = 5):
    """
    Runs the full evaluation pipeline: load, merge, process, analyze.

    Args:
        gold_standard_path (str): Path to the gold standard CSV file.
        response_path (str): Path to the LLM responses JSON Lines file.
        output_prefix (str): Prefix for output analysis file and plots.
        max_candidates_extract (int): How many LLM candidates to extract.
    """
    logger.info("--- Starting Evaluation Pipeline ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 1. Load Data
        gold_df = load_gold_standard(gold_standard_path)
        response_df = load_llm_responses(response_path)

        # Exit if loading failed or produced no data
        if gold_df.empty or response_df.empty:
             logger.error("Cannot proceed with empty gold standard or response data.")
             return

        # 2. Merge Data
        merged_df = merge_data(gold_df, response_df)
        if merged_df.empty:
            logger.error("Cannot proceed with empty merged data.")
            return

        # 3. Extract LLM Predictions
        analysis_df = extract_llm_predictions(merged_df, max_candidates=max_candidates_extract)

        # 4. Calculate Digit Matches (between gold and top LLM prediction)
        analysis_df = calculate_digit_matches(analysis_df) # Uses default column names

        # 5. Save Detailed Analysis Results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_csv_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_results_{timestamp}.csv")
        analysis_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"Detailed analysis results saved to: {output_csv_path}")

        # 6. Generate Histogram Plot
        plot_digit_match_histogram(
            df=analysis_df,
            output_dir=OUTPUT_DIR,
            filename_prefix=f"{output_prefix}_digit_match_hist"
            # Stratify by default using COL_GOLD_FLAG ('sic_ind_code_flag')
        )

        # (Optional) Generate overall histogram as well
        plot_digit_match_histogram(
            df=analysis_df,
            stratify_by=None, # Pass None for overall plot
            output_dir=OUTPUT_DIR,
            filename_prefix=f"{output_prefix}_digit_match_hist_overall"
        )

        # --- Placeholder for Future Analysis ---
        # TODO: Implement Rank Analysis (Check if gold SIC is in alternatives)
        # TODO: Implement Confidence Analysis (Likelihood vs Accuracy)
        # ---------------------------------------

        logger.info("--- Evaluation Pipeline Finished ---")

    except Exception as e:
        logger.exception(f"An error occurred during the evaluation pipeline: {e}")


# --- Command Line Execution ---
if __name__ == "__main__":
    # For simplicity, using hardcoded paths.
    # Consider using argparse for command-line arguments:
    # import argparse
    # parser = argparse.ArgumentParser(description="Evaluate LLM SIC Code predictions.")
    # parser.add_argument("gold_standard", help="Path to gold standard CSV file.")
    # parser.add_argument("llm_responses", help="Path to LLM responses JSON Lines file.")
    # parser.add_argument("-o", "--output_prefix", default="evaluation", help="Prefix for output files.")
    # args = parser.parse_args()
    # run_evaluation(args.gold_standard, args.llm_responses, args.output_prefix)

    # --- Example Usage ---
    # Define file paths (MAKE SURE THESE ARE CORRECT)
    # Use absolute paths or ensure relative paths are correct from where you run the script
    gold_file = 'data/all_examples_comma.csv' # Adjust path as needed
    response_file = 'data/output_responses.jsonl' # Assumes .jsonl extension now
    output_file_prefix = 'sic_evaluation_run'

    # Ensure output directory exists before running
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_evaluation(gold_file, response_file, output_file_prefix)