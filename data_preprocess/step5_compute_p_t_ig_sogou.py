import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 10))  # Adjust the figure size for better visualization
    sns.heatmap(matrix, cmap="YlGnBu", cbar_kws={'label': 'Probability'})
    plt.title(title)
    plt.xlabel('Prefix ID')
    plt.ylabel('Query ID')
    plt.show()


def compute_p_q_ig(data_file, mapping_file, iq_data_file, data_name):
    # Load the datasets
    df = pd.read_csv(data_file)
    prefix_mapping = pd.read_csv(mapping_file)
    iq_data = pd.read_csv(iq_data_file)

    # Filter queries in the df that have valid prefixes in iq_data
    valid_queries = iq_data['QueryID'].unique()
    df = df[df['QueryID'].isin(valid_queries)]

    # Create mappings for array-friendly indices
    query_id_to_index = {id: idx for idx, id in enumerate(valid_queries)}
    prefix_id_to_index = {id: idx for idx, id in enumerate(prefix_mapping['Query_prefix_id'].values)}

    # Determine unique groups
    unique_groups = df['UserGroup'].unique()

    # Initialize matrices for each group
    matrices = {}
    for group in unique_groups:
        matrices[group] = np.zeros((len(valid_queries), len(prefix_mapping)))

    # Process each row in the df
    for _, row in df.iterrows():
        query_id = row['QueryID']
        user_group = row['UserGroup']

        # Get the possible prefixes for this query from iq_data
        possible_prefixes = iq_data[iq_data['QueryID'] == query_id]['Query_prefix_id'].values

        # Increment the count in the matrix for each prefix using mapped indices
        for prefix_id in possible_prefixes:
            matrices[user_group][query_id_to_index[query_id], prefix_id_to_index[prefix_id]] += 1

    # Normalize each matrix
    for group, matrix in matrices.items():
        column_sums = matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # To avoid division by zero
        matrices[group] = matrix / column_sums[np.newaxis, :]

        # Save the matrix
        save_path = f"../data_preprocessed/{data_name}/prob_q_ig_{group}.npy"
        np.save(save_path, matrices[group])

        if os.path.exists(save_path):
            loaded_matrix = np.load(save_path)
            print("loaded_matrix:", loaded_matrix.shape)
            print("loaded_matrix:", loaded_matrix.sum(0))


if __name__ == "__main__":
    data_name = 'sogou_small'
    compute_p_q_ig(f'../data_preprocessed/{data_name}/updated_{data_name}_processed.csv',
                   f'../data_preprocessed/{data_name}/i_mapping.csv',
                   f'../data_preprocessed/{data_name}/i_q_data.csv',
                   data_name)

    # Load matrices
    matrix_active = np.load(f'../data_preprocessed/{data_name}/prob_q_ig_Active.npy')
    matrix_inactive = np.load(f'../data_preprocessed/{data_name}/prob_q_ig_Less Active.npy')

    # Plot heatmaps
    plot_heatmap(matrix_active, "Probability Distribution for Active User Group")
    plot_heatmap(matrix_inactive, "Probability Distribution for Inactive User Group")
