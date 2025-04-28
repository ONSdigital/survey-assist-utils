# Read the response JSON
file_path = '../data/output_responses.txt'

# store the JSON objects
data = []

# Read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON object and append it to the list
        data.append(json.loads(line.strip()))

# Convert the list of JSON objects to a DataFrame
response_data = pd.json_normalize(data)

# Display the DataFrame
# print(response_data)

def check_digit_match(data, col1, col2):
    """
    Check the match of the first n digits between two columns of 5-digit numbers and add Boolean columns indicating the matches.
    Parameters:
        data (dict): A dictionary containing the data with columns to be compared.
        col1 (str): The name of the first column to compare.
        col2 (str): The name of the second column to compare.

    Returns:
    pd.DataFrame: A DataFrame with the original data and additional Boolean columns indicating the matches.
    """

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add boolean columns for matches with col2 as prefix
    for i in range(5, 0, -1):
        df[f'{col2}_match_{i}'] = df.apply(lambda row: row[col1][:i] == row[col2][:i] if pd.notna(row[col1]) and pd.notna(row[col2]) else pd.NA, axis=1)

    # Add a column for zero match (True if the first digit is different)
    df[f'{col2}_zero_match'] = df.apply(lambda row: row[col1][0] != row[col2][0] if pd.notna(row[col1]) and pd.notna(row[col2]) else pd.NA, axis=1)

    return df

# Function to find the number of columns needed to store all data
def find_columns_needed(df, json_column):
    max_columns = 0
    
    for index, row in df.iterrows():
        json_data = row[json_column]
        num_columns = len(json_data[0].keys()) * len(json_data)
        if num_columns > max_columns:
            max_columns = num_columns
    
    return max_columns


def store_json_as_dataframe(df, json_column, max_columns):
    """
    Store JSON data from a specified column in a DataFrame, ensuring the resulting DataFrame does not exceed a maximum column size.

    Parameters:
        df (pd.DataFrame): The original DataFrame containing the JSON data.
        json_column (str): The name of the column containing the JSON data.
        max_columns (int): The maximum number of columns allowed in the resulting DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the expanded JSON data.

    Raises:
    ValueError: If the resulting DataFrame exceeds the maximum column size.
    """
   
    # Initialize an empty list to store the expanded data
    expanded_data = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        json_data = row[json_column]
        expanded_row = {}
        
        # Iterate through each item in the JSON data
        for i, item in enumerate(json_data):
            for key, value in item.items():
                column_name = f"{key}_{i+1}"
                expanded_row[column_name] = value
        
        expanded_data.append(expanded_row)
    
    # Create a new DataFrame from the expanded data
    expanded_df = pd.DataFrame(expanded_data)
    
    # Ensure the DataFrame does not exceed the maximum column size
    if len(expanded_df.columns) > max_columns:
        raise ValueError(f"The DataFrame exceeds the maximum column size of {max_columns}.")
    
    return expanded_df


### We want to flatten the JSON to a dataframe, using the max number of columns corresponding to the variable responses from the LLM
# Find the number of columns needed
columns_needed = find_columns_needed(response_data, 'sic_candidates')
print(f"Number of columns needed to store all of the data: {columns_needed}")

# Store the JSON data as a DataFrame with a maximum column size of the max from the responses.
expanded_result_df = store_json_as_dataframe(response_data, 'sic_candidates', columns_needed)

# Add in the UID:
expanded_result_df['unique_id'] = response_data['unique_id']

print(expanded_result_df)
expanded_result_df.to_csv('../data/expanded_result_df.csv')

### Make a table of second choice hits

print(csv_filepath)
gold_df = pd.read_csv(csv_filepath, delimiter=',', dtype=str)

# Create a new dataframe containing both unique_id, and sic codes: 
comparison_df_all = pd.DataFrame({
    'unique_id_original': gold_df['unique_id'],
    'sic_ind1': gold_df['sic_ind1'],
    'unique_id_test': expanded_result_df['unique_id'],
    'sic_code_1': expanded_result_df['sic_code_1'],
    'sic_code_2': expanded_result_df['sic_code_2'],
    'sic_code_3': expanded_result_df['sic_code_3'],
    'sic_code_4': expanded_result_df['sic_code_4'],
    'sic_code_5': expanded_result_df['sic_code_5']
})


comparison_df_all = check_digit_match(comparison_df_all, 'sic_ind1', 'sic_code_1')
comparison_df_all = check_digit_match(comparison_df_all, 'sic_ind1', 'sic_code_2')
comparison_df_all = check_digit_match(comparison_df_all, 'sic_ind1', 'sic_code_3')
comparison_df_all = check_digit_match(comparison_df_all, 'sic_ind1', 'sic_code_4')
comparison_df_all = check_digit_match(comparison_df_all, 'sic_ind1', 'sic_code_5')
comparison_df_all.to_csv('../data/comparison_df_all.csv')
comparison_df_all

### Now we should merge different runs based on UID

def merge_and_deduplicate(df1, df2, unique_id_column):
    """
    Merge two dataframes on a specified column and drop duplicates based on this column, keeping only one instance of each unique ID.

    Parameters:
    df1 (pd.DataFrame): The first dataframe to merge.
    df2 (pd.DataFrame): The second dataframe to merge.
    unique_id_column (str): The column name on which to merge the dataframes and drop duplicates.

    Returns:
    pd.DataFrame: A new dataframe with merged data and deduplicated rows based on the unique_id_column.
    """

    # Concatenate the dataframes
    concatenated_df = pd.concat([df1, df2])
    
    # Drop duplicates based on the unique_id_column, keeping only the first occurrence
    deduplicated_df = concatenated_df.drop_duplicates(subset=[unique_id_column], keep='first')
    
    return deduplicated_df

### Handle lost dtype from bool to string. Read in and convert it using this converter

# Function to identify boolean-like columns
def identify_bool_columns(df):
    """
    Identify columns in a DataFrame that should be of boolean type but are currently stored as strings.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to be analyzed.

    Returns:
    tuple: A tuple containing two lists:
    - bool_columns (list): List of column names that should be boolean.
    - other_columns (list): List of column names that are not boolean-like.

    Examples:
    >>> data = {
    ...'col1': ['True', 'False', 'True'],
    ...'col2': ['yes', 'no', 'yes'],
    ...'col3': ['1', '0', '1'],
    ...'col4': ['apple', 'banana', 'cherry']
    ... }
    >>> df = pd.DataFrame(data)
    >>> identify_bool_columns(df)
    (['col1', 'col2', 'col3'], ['col4'])
    """

    bool_columns = []
    other_columns = []

    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if set(unique_values).issubset({'True', 'False', 'true', 'false', '1', '0', 'yes', 'no', 'Yes', 'No'}):
            bool_columns.append(col)
        else:
            other_columns.append(col)
    return bool_columns, other_columns

### Load in the csv and convert strings to bool, even though there are NAs amongst them!

def str_to_bool(val):
    """
    Convert a string representation of a boolean value to an actual boolean value.

    Parameters:
    val (str): The input value to be converted. This can be a string representing a boolean value ('True', 'False', 'yes', 'no', '1', '0') or NA.

    Returns:
    bool or None: Returns True if the input value is a string representation of a boolean True ('True', 'yes', '1').
      Returns False if the input value is a string representation of a boolean False ('False', 'no', '0').
      Returns None if the input value is NA or any other value that does not match the boolean representations.

    Examples:
    >>> str_to_bool('True')
    True
    >>> str_to_bool('False')
    False
    >>> str_to_bool('yes')
    True
    >>> str_to_bool('no')
    False
    >>> str_to_bool('1')
    True
    >>> str_to_bool('0')
    False
    >>> str_to_bool(None)
    None
    >>> str_to_bool('apple')
    None
    """
    if pd.isna(val):
        return None
    if val.lower() in ['true', 'yes', '1']:
        return True
    elif val.lower() in ['false', 'no', '0']:
        return False
    else:
        return None


bool_columns_1 = ['sic_code_1_match_5',
 'sic_code_1_match_4',
 'sic_code_1_match_3',
 'sic_code_1_match_2',
 'sic_code_1_match_1',
 'sic_code_1_zero_match',
 'sic_code_2_match_5',
 'sic_code_2_match_4',
 'sic_code_2_match_3',
 'sic_code_2_match_2',
 'sic_code_2_match_1',
 'sic_code_2_zero_match',
 'sic_code_3_match_5',
 'sic_code_3_match_4',
 'sic_code_3_match_3',
 'sic_code_3_match_2',
 'sic_code_3_match_1',
 'sic_code_3_zero_match',
 'sic_code_4_match_5',
 'sic_code_4_match_4',
 'sic_code_4_match_3',
 'sic_code_4_match_2',
 'sic_code_4_match_1',
 'sic_code_4_zero_match']

 # List columns that should be boolean
bool_columns_1, str_cols_1 = identify_bool_columns(first_five_results)
print("Columns that should be boolean:", bool_columns_1)
print("Other columns:", str_cols_1)

file_path = '../data/comparison_df_all_first_five.csv'
converters = {col: str_to_bool for col in bool_columns_1}
first_five_results = pd.read_csv(file_path, delimiter=',', converters=converters)

# Count TRUE values in each column
true_counts = all_tests[plot_list_columns].sum()

# Mapping dictionary for shorter names
short_names = {
    'sic_code_1_match_5': 'Match 5',
    'sic_code_1_match_4': 'Match 4',
    'sic_code_1_match_3': 'Match 3',
    'sic_code_1_match_2': 'Match 2',
    'sic_code_1_match_1': 'Match 1',
    'sic_code_1_zero_match': 'Zero Match'
}

# Rename the index of the true_counts Series using the mapping dictionary
true_counts.index = true_counts.index.map(short_names)

# Create a bar plot for the count of TRUE values
plt.figure(figsize=(10, 6))
true_counts.plot(kind='bar')
sample_size = all_tests[plot_list_columns].shape[0]
plt.title(f'Count of Digits matched (sample size {sample_size})')
plt.xlabel('Columns')
plt.ylabel('Count of Matches')
plt.xticks(rotation=0)


# Save the plot to a file
plt.savefig('true_values_histogram.png')

plt.show()

     