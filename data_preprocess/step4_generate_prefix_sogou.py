import pandas as pd

def generate_prefixes(query_pinyin):
    # Split the query_pinyin by space and generate the possible prefixes
    words = str(query_pinyin).split()
    return [' '.join(words[:i + 1]) for i in range(len(words))]

def generate_and_save_prefix_data(filename, data_name, n):
    # 1. Load the file
    df = pd.read_csv(filename)

    # 2. Generate all possible prefixes
    all_prefixes = []
    for _, row in df.iterrows():
        query_id = row["QueryID"]
        query_pinyin = row["Query_pinyin"]
        prefixes = generate_prefixes(query_pinyin)
        for prefix in prefixes:
            all_prefixes.append({"QueryID": query_id, "Query_pinyin": query_pinyin, "Query_prefix": prefix})

    # Convert to DataFrame for easier counting
    prefix_df = pd.DataFrame(all_prefixes)

    # 3. Count the occurrence of each prefix
    prefix_counts = prefix_df["Query_prefix"].value_counts()

    # 4. Filter prefixes with cumulative count greater than n
    valid_prefixes = prefix_counts[prefix_counts > n].index.tolist()
    prefix_df = prefix_df[prefix_df["Query_prefix"].isin(valid_prefixes)]

    # 5. Create and save the sorted prefix mapping
    prefix_counts = prefix_df["Query_prefix"].value_counts()
    sorted_prefix_df = prefix_counts.reset_index()
    sorted_prefix_df.columns = ["Query_prefix", "Count"]
    # Assign new IDs based on the sorted order
    sorted_prefix_df["Query_prefix_id"] = sorted_prefix_df.index
    # Re-arrange the columns for better readability
    sorted_prefix_df = sorted_prefix_df[["Query_prefix_id", "Query_prefix", "Count"]]
    # Save the sorted dataframe to a file
    sorted_prefix_df.to_csv(f'../data_preprocessed/{data_name}/i_mapping.csv', index=False)

    # 6. Add the Query_prefix_id to prefix_df
    prefix_id_map = sorted_prefix_df.set_index('Query_prefix')['Query_prefix_id'].to_dict()
    prefix_df['Query_prefix_id'] = prefix_df['Query_prefix'].map(prefix_id_map)

    # 7. Save the resulting data with desired column order
    prefix_df = prefix_df[["Query_pinyin", "Query_prefix", "QueryID", "Query_prefix_id"]]
    prefix_df.to_csv(f'../data_preprocessed/{data_name}/i_q_data.csv', index=False)

if __name__ == "__main__":
    data_name = 'sogou_small'
    # Usage
    generate_and_save_prefix_data(f'../data_preprocessed/{data_name}/q_mapping.csv', data_name, n=20)
