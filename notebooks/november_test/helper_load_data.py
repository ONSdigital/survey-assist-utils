"""Load and prepare data for clerical and SurveyAssist coded data comparison."""

import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import (
    get_clean_n_digit_codes,
    parse_numerical_code,
)


def load_data(work_dir: str) -> pd.DataFrame:
    """Load and merge clerical and SurveyAssist coded data."""
    sa_coded_df = pd.read_parquet(
        work_dir + "/evaluation_df_with_sa_clean_codes_and_sic_section.parquet"
    )
    sa_closed_q = pd.read_parquet(
        work_dir + "/closed_questions/closed_questions_codes.parquet"
    )
    cc_coded_df = pd.read_parquet(
        work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
    )

    combined_df = sa_coded_df.merge(
        sa_closed_q.drop(
            columns=sa_closed_q.columns.intersection(sa_coded_df.columns).difference(
                ["unique_id", "user"]
            )
        ),
        on=["unique_id", "user"],
        how="outer",
    ).merge(
        cc_coded_df.drop(
            columns=cc_coded_df.columns.intersection(sa_coded_df.columns).difference(
                ["unique_id", "user"]
            )
        ),
        on=["unique_id", "user"],
        how="outer",
    )

    print(
        f"Loaded data with {combined_df.shape[0]} records. "
        f"Merging clerical ({cc_coded_df.shape[0]}) with model data ({sa_coded_df.shape[0]}) "
        f"and closed q data ({sa_closed_q.shape[0]})."
    )

    # parquet doesn't like sets it saves it as arrays, convert back
    set_cols = [
        "sa_initial_codes",
        "sa_final_codes_open_q",
        "cc_initial_codes",
        "cc_final_codes_open_q",
    ]

    for col in set_cols:
        msk = combined_df[col].notna()
        combined_df.loc[msk, col] = combined_df.loc[msk, col].apply(set)
        combined_df.loc[~msk, col] = [
            set() for _ in range(msk.sum(), combined_df.shape[0])
        ]

    # and convert closed q codes to set for consistency
    combined_df["sa_final_codes_closed_q"] = combined_df[
        "survey_assist_closed_question_response_code"
    ].apply(lambda x: get_clean_n_digit_codes(parse_numerical_code(x), n=5)[0])
    # use initial codes where no closed q asked
    msk = combined_df["survey_assist_open_question"].isna()
    combined_df.loc[msk, "sa_final_codes_closed_q"] = combined_df.loc[
        msk, "sa_initial_codes"
    ]

    return combined_df


def combine_small_groups(
    input_df: pd.DataFrame,
    group_col: str = "SIC Section",
    group_size_threshold: int = 30,
    add_total: bool = True,
    manual_groups: tuple | None = (("B", "D", "E"), ("R", "S", "T")),
) -> pd.DataFrame:
    """Combine small groups in the specified column into an 'Other' category.

    Args:
        input_df: DataFrame containing the groups.
        group_col: Name of the column representing groups.
        group_size_threshold: Minimum size of group to be shown separately.
            Groups smaller than this will be combined.
        add_total: Whether to add a 'Total' group aggregating all data.
        manual_groups: List of lists, where each sublist contains group names
            to be combined together manually before applying the size threshold.

    Returns:
        DataFrame with small groups combined into 'Other'.
    """
    temp_df = input_df.copy()

    if manual_groups:
        for group in manual_groups:
            msk = temp_df[group_col].isin(group)
            temp_df.loc[msk, group_col] = ",".join(group)

    section_sizes = temp_df[group_col].value_counts(dropna=False)
    temp_df1 = temp_df.copy()
    too_small = sorted(
        section_sizes[section_sizes < group_size_threshold].index.tolist()
    )

    if too_small:
        print(
            f"Combining {len(too_small)} groups smaller than {group_size_threshold} together:"
            f"{section_sizes[section_sizes < group_size_threshold].to_dict()}"
        )
        msk_small = temp_df1[group_col].isin(too_small)
        temp_df1.loc[msk_small, group_col] = "Suppressed:<br>" + ",".join(too_small)

    if not add_total:
        return temp_df1

    temp_df2 = temp_df.copy()
    temp_df2[group_col] = "Total"
    temp_df = pd.concat([temp_df1, temp_df2], axis=0, ignore_index=True)

    return temp_df
