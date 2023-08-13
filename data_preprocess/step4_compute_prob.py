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


def compute_p_q_ig(data_name):
    data_file = f'../data_preprocessed/{data_name}/new_{data_name}_processed.csv'
    mapping_file = f'../data_preprocessed/{data_name}/i_mapping.csv'
    iq_data_file = f'../data_preprocessed/{data_name}/i_q_data.csv'

    # Load the datasets
    df = pd.read_csv(data_file)
    prefix_mapping = pd.read_csv(mapping_file)
    iq_data = pd.read_csv(iq_data_file)

    # Determine unique groups
    unique_groups = df['UserGroup'].unique()

    # Create mappings for array-friendly indices
    query_ids_sorted = sorted(df['QueryID'].unique())
    query_id_to_index = {id: idx for idx, id in enumerate(query_ids_sorted)}

    prefix_ids_sorted = sorted(prefix_mapping['Query_prefix_id'].values)
    prefix_id_to_index = {id: idx for idx, id in enumerate(prefix_ids_sorted)}

    # Determine unique groups and initialize matrices
    matrices = {}
    for group in unique_groups:
        matrices[group] = np.zeros((len(query_ids_sorted), len(prefix_ids_sorted)))

    # Process each row in the df
    for _, row in df.iterrows():
        query_id = row['QueryID']
        user_group = row['UserGroup']

        # Get the possible prefixes for this query from iq_data
        possible_prefixes = iq_data[iq_data['QueryID'] == query_id]['Query_prefix_id'].values

        # Increment the count in the matrix for each prefix using mapped indices
        for prefix_id in possible_prefixes:
            # Using get method to provide a default value of -1 for non-existing keys
            q_idx = query_id_to_index.get(query_id, -1)
            p_idx = prefix_id_to_index.get(prefix_id, -1)

            if q_idx != -1 and p_idx != -1:  # Ensure valid indices
                matrices[user_group][q_idx, p_idx] += 1

    # Normalize each matrix
    for group, matrix in matrices.items():
        column_sums = matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # To avoid division by zero
        matrices[group] = matrix / column_sums[np.newaxis, :]

        # Save the matrix using numerical label
        save_path = f"../data_preprocessed/{data_name}/prob_q_ig_{group}.npy"
        np.save(save_path, matrices[group])

        if os.path.exists(save_path):
            loaded_matrix = np.load(save_path)
            print("prob_q_ig:", loaded_matrix.shape)
            # print("prob_q_ig:", loaded_matrix.sum(0))

    return unique_groups


def compute_p_t_ig(data_name, unique_groups):
    # Load p(t|q)
    prob_t_q = np.load(f'../data_preprocessed/{data_name}/prob_t_q.npy')

    for group in unique_groups:
        # Load p(q|i,g) for the specific group using the numeric label
        prob_q_ig = np.load(f'../data_preprocessed/{data_name}/prob_q_ig_{group}.npy')

        # Compute p(t|i,g)
        prob_t_ig = np.matmul(prob_t_q, prob_q_ig)

        # Save the matrix using the numeric label
        save_path = f"../data_preprocessed/{data_name}/prob_t_ig_{group}.npy"
        np.save(save_path, prob_t_ig)

        # Optional: print to verify
        if os.path.exists(save_path):
            loaded_matrix = np.load(save_path)
            print("prob_t_ig:", loaded_matrix.shape)


def run_step4_compute_prob(data_name):
    unique_groups = compute_p_q_ig(data_name)
    compute_p_t_ig(data_name, unique_groups)


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step4_compute_prob(data_name)

    # # Load matrices
    # matrix_active = np.load(f'../data_preprocessed/{data_name}/prob_q_ig_Active.npy')
    # matrix_inactive = np.load(f'../data_preprocessed/{data_name}/prob_q_ig_Less Active.npy')
    #
    # # Plot heatmaps
    # plot_heatmap(matrix_active, "Probability Distribution for Active User Group")
    # plot_heatmap(matrix_inactive, "Probability Distribution for Inactive User Group")
