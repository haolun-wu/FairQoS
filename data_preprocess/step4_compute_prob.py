import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(matrix, title, xlabel_name, ylabel_name):
    plt.figure(figsize=(10, 10))  # Adjust the figure size for better visualization
    sns.heatmap(matrix, cmap="YlGnBu", cbar_kws={'label': 'Probability'})
    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.show()


def compute_p_o_ig(data_name):
    data_file = f'../data_preprocessed/{data_name}/new_{data_name}_processed.csv'
    mapping_file = f'../data_preprocessed/{data_name}/i_mapping.csv'
    io_data_file = f'../data_preprocessed/{data_name}/i_o_data.csv'

    # Load the datasets
    df = pd.read_csv(data_file)
    prefix_mapping = pd.read_csv(mapping_file)
    io_data = pd.read_csv(io_data_file)

    # Determine unique groups
    unique_groups = df['UserGroup'].unique()

    # Create mappings for array-friendly indices
    query_ids_sorted = sorted(df['QueryID'].unique())
    query_id_to_index = {id: idx for idx, id in enumerate(query_ids_sorted)}

    prefix_ids_sorted = sorted(prefix_mapping['Query_prefix_id'].values)
    prefix_id_to_index = {id: idx for idx, id in enumerate(prefix_ids_sorted)}

    # Initialize matrices for p(o|i,g)
    matrices = {}
    for group in unique_groups:
        matrices[group] = np.zeros((len(query_ids_sorted), len(prefix_ids_sorted)))

    # Process each row in the df
    for _, row in df.iterrows():
        query_id = row['QueryID']
        user_group = row['UserGroup']

        # Get the possible prefixes for this query from io_data
        possible_prefixes = io_data[io_data['QueryID'] == query_id]['Query_prefix_id'].values

        # Increment the count in the matrix for each prefix using mapped indices
        for prefix_id in possible_prefixes:
            q_idx = query_id_to_index.get(query_id, -1)
            p_idx = prefix_id_to_index.get(prefix_id, -1)

            if q_idx != -1 and p_idx != -1:  # Ensure valid indices
                matrices[user_group][q_idx, p_idx] += 1

    # Normalize each matrix and store in the dictionary
    prob_o_ig_dict = {}
    for group, matrix in matrices.items():
        column_sums = matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # To avoid division by zero
        prob_o_ig_dict[group] = matrix / column_sums[np.newaxis, :]

    print("--------")
    print("prob_o_ig_dict:")
    print("keys:", prob_o_ig_dict.keys())
    print("Shape and sum:", prob_o_ig_dict[0].shape, prob_o_ig_dict[0].sum())
    print("--------")

    # Save the entire dictionary
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/prob_o_ig_dict.npy", prob_o_ig_dict)

    return unique_groups


def compute_p_t_ig(data_name, unique_groups):
    # Load p(t|o)
    prob_t_o = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_o.npy')

    # Initialize prob_t_ig dictionary
    prob_t_ig_dict = {}
    for group in unique_groups:
        # Load p(o|i,g) for the specific group from the dictionary
        prob_o_ig = \
            np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_o_ig_dict.npy', allow_pickle=True).item()[group]

        # Compute p(t|i,g)
        prob_t_ig = np.matmul(prob_t_o, prob_o_ig)
        prob_t_ig_dict[group] = prob_t_ig

    print("--------")
    print("prob_t_ig_dict:")
    print("keys:", prob_t_ig_dict.keys())
    print("Shape and sum:", prob_t_ig_dict[0].shape, prob_t_ig_dict[0].sum())
    print("--------")

    # Save the entire dictionary
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/prob_t_ig_dict.npy", prob_t_ig_dict)


def compute_p_i_and_p_i_g(data_name, unique_groups):
    # Load the sorted prefix dataframe
    sorted_prefix_df = pd.read_csv(f'../data_preprocessed/{data_name}/i_mapping.csv')

    # Compute p(i)
    total_count = sorted_prefix_df["Count"].sum()
    prob_i = sorted_prefix_df["Count"] / total_count
    np.save(f'../data_preprocessed/{data_name}/prob_matrix/prob_i.npy', prob_i.values)

    # Load the prefix_df and data_df
    prefix_df = pd.read_csv(f'../data_preprocessed/{data_name}/i_o_data.csv')
    data_df = pd.read_csv(f'../data_preprocessed/{data_name}/new_{data_name}_processed.csv')

    # Merge them based on 'QueryID' to get 'UserGroup' in prefix_df
    merged_df = prefix_df.merge(data_df[['QueryID', 'UserGroup']], on='QueryID', how='left')

    # Compute p(i|g)
    prob_i_g_dict = {}
    for group in unique_groups:
        filtered_df = merged_df[merged_df['UserGroup'] == group]
        group_counts = filtered_df['Query_prefix'].value_counts()
        group_total = group_counts.sum()
        group_probs = group_counts / group_total

        # Ensure the length and order of probabilities
        probs = sorted_prefix_df['Query_prefix'].map(group_probs).fillna(0).values
        prob_i_g_dict[group] = probs

    print("prob_i:")
    print("shape and sum:", prob_i.shape, prob_i.sum())
    print("--------")
    print("p_i_g_dict:")
    print("keys:", prob_i_g_dict.keys())
    print("shape and sum:", prob_i_g_dict[0].shape, prob_i_g_dict[0].sum())
    print("--------")

    np.save(f'../data_preprocessed/{data_name}/prob_matrix/prob_i_g_dict.npy', prob_i_g_dict)

    return prob_i, prob_i_g_dict


def run_step4_compute_prob(data_name):
    unique_groups = compute_p_o_ig(data_name)
    compute_p_t_ig(data_name, unique_groups)
    compute_p_i_and_p_i_g(data_name, unique_groups)


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step4_compute_prob(data_name)

    # Load matrices
    prob_o_ig = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_o_ig_dict.npy', allow_pickle=True).item()
    prob_t_ig = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_ig_dict.npy', allow_pickle=True).item()

    # Plot heatmaps
    plot_heatmap(prob_o_ig[0], "Probability Distribution for User Group 0", xlabel_name='prefix (i)',
                 ylabel_name='query (o)')
    plot_heatmap(prob_o_ig[1], "Probability Distribution for User Group 1", xlabel_name='prefix (i)',
                 ylabel_name='query (o)')

    plot_heatmap(prob_t_ig[0], "Probability Distribution for User Group 0", xlabel_name='prefix (i)',
                 ylabel_name='intent (t)')
    plot_heatmap(prob_t_ig[1], "Probability Distribution for User Group 1", xlabel_name='prefix (i)',
                 ylabel_name='intent (t)')
