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


def compute_p_d_qg_and_p_d_q(data_name):
    data_file = f'../data_preprocessed/{data_name}/new_{data_name}_processed.csv'
    mapping_file = f'../data_preprocessed/{data_name}/q_mapping.csv'
    dq_data_file = f'../data_preprocessed/{data_name}/d_q_mapping.csv'

    # Load the datasets
    df = pd.read_csv(data_file)
    prefix_mapping = pd.read_csv(mapping_file)
    dq_data = pd.read_csv(dq_data_file)

    # Determine unique groups
    unique_groups = df['UserGroup'].unique()

    # Create mappings for array-friendly indices
    query_ids_sorted = sorted(df['QueryID'].unique())
    query_id_to_index = {id: idx for idx, id in enumerate(query_ids_sorted)}

    prefix_ids_sorted = sorted(prefix_mapping['Query_prefix_id'].values)
    prefix_id_to_index = {id: idx for idx, id in enumerate(prefix_ids_sorted)}

    # Initialize matrices for p(d|q,g)
    matrices = {}
    for group in unique_groups:
        matrices[group] = np.zeros((len(query_ids_sorted), len(prefix_ids_sorted)))

    # Initialize matrices for p(d|q)
    pdq_matrix = np.zeros((len(query_ids_sorted), len(prefix_ids_sorted)))

    # Process each row in the df
    for _, row in df.iterrows():
        query_id = row['QueryID']
        user_group = row['UserGroup']

        # Get the possible prefixes for this query from dq_data
        possible_prefixes = dq_data[dq_data['QueryID'] == query_id]['Query_prefix_id'].values

        # Increment the count in the matrix for each prefix using mapped indices
        for prefix_id in possible_prefixes:
            q_idx = query_id_to_index.get(query_id, -1)
            p_idx = prefix_id_to_index.get(prefix_id, -1)

            if q_idx != -1 and p_idx != -1:  # Ensure valid indices
                matrices[user_group][q_idx, p_idx] += 1
                pdq_matrix[q_idx, p_idx] += 1  # Increment count in the aggregate matrix

    # Normalize each matrix and store in the dictionary
    prob_d_qg_dict = {}
    for group, matrix in matrices.items():
        column_sums = matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # To avoid division by zero
        prob_d_qg_dict[group] = matrix / column_sums[np.newaxis, :]

    # Normalize the pdq_matrix
    pdq_column_sums = pdq_matrix.sum(axis=0)
    pdq_column_sums[pdq_column_sums == 0] = 1  # To avoid division by zero
    pdq_matrix = pdq_matrix / pdq_column_sums[np.newaxis, :]

    print("--------")
    print("prob_d_qg_dict:")
    print("keys:", prob_d_qg_dict.keys())
    print("Shape and sum:", prob_d_qg_dict[0].shape, prob_d_qg_dict[0].sum())
    print("--------")

    # Save the entire dictionary
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/p(d|qg).npy", prob_d_qg_dict)

    print("--------")
    print("p(d|q) matrix details:")
    print("Shape and sum:", pdq_matrix.shape, pdq_matrix.sum())
    print("--------")

    # Optionally, save the p(d|q) matrix
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/p(d|q).npy", pdq_matrix)

    return unique_groups


def compute_p_t_qg(data_name, unique_groups):
    # Load p(t|d)
    prob_t_d = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|d).npy')

    # Initialize prob_t_ig dictionary
    prob_t_qg_dict = {}
    for group in unique_groups:
        # Load p(d|q,g) for the specific group from the dictionary
        prob_d_ig = \
            np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(d|qg).npy', allow_pickle=True).item()[group]

        # Compute p(t|i,g)
        prob_t_qg = np.matmul(prob_t_d, prob_d_ig)
        prob_t_qg_dict[group] = prob_t_qg

    print("--------")
    print("prob_t_qg_dict:")
    print("keys:", prob_t_qg_dict.keys())
    print("Shape and sum:", prob_t_qg_dict[0].shape, prob_t_qg_dict[0].sum())
    print("--------")

    # Save the entire dictionary
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/p(t|qg).npy", prob_t_qg_dict)


def compute_p_t_q(data_name):
    prob_t_d = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|d).npy') # Load p(t|d)
    prob_d_q = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(d|q).npy') # Load p(d|q) matrix

    # Compute p(t|q) = sum_d p(t|d) * p(d|q)
    prob_t_q = np.matmul(prob_t_d, prob_d_q)

    print("--------")
    print("p(t|q) matrix details:")
    print("Shape:", prob_t_q.shape)
    print("Sum of elements:", prob_t_q.sum())
    print("--------")

    # Save the p(t|q) matrix
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/p(t|q).npy", prob_t_q)


# Example usage:
# compute_p_t_q('your_data_name')


def compute_p_q(data_name):
    # Load the sorted prefix dataframe
    sorted_prefix_df = pd.read_csv(f'../data_preprocessed/{data_name}/q_mapping.csv')

    # Compute p(q)
    total_count = sorted_prefix_df["Count"].sum()
    prob_q = sorted_prefix_df["Count"] / total_count
    np.save(f'../data_preprocessed/{data_name}/prob_matrix/p(q).npy', prob_q.values)

    print("prob_q:")
    print("shape and sum:", prob_q.shape, prob_q.sum())
    print("--------")


def compute_p_q_g(data_name, unique_groups):
    # Load the sorted prefix dataframe
    sorted_prefix_df = pd.read_csv(f'../data_preprocessed/{data_name}/q_mapping.csv')

    # Load the prefix_df and data_df
    prefix_df = pd.read_csv(f'../data_preprocessed/{data_name}/d_q_mapping.csv')
    data_df = pd.read_csv(f'../data_preprocessed/{data_name}/new_{data_name}_processed.csv')

    # Merge them based on 'QueryID' to get 'UserGroup' in prefix_df
    merged_df = prefix_df.merge(data_df[['QueryID', 'UserGroup']], on='QueryID', how='left')

    # Compute p(i|g)
    prob_q_g_dict = {}
    for group in unique_groups:
        filtered_df = merged_df[merged_df['UserGroup'] == group]
        group_counts = filtered_df['Query_prefix'].value_counts()
        group_total = group_counts.sum()
        group_probs = group_counts / group_total

        # Ensure the length and order of probabilities
        probs = sorted_prefix_df['Query_prefix'].map(group_probs).fillna(0).values
        prob_q_g_dict[group] = probs

    print("p_q_g_dict:")
    print("keys:", prob_q_g_dict.keys())
    print("shape and sum:", prob_q_g_dict[0].shape, prob_q_g_dict[0].sum())
    print("--------")

    np.save(f'../data_preprocessed/{data_name}/prob_matrix/p(q|g).npy', prob_q_g_dict)


def run_step4_compute_prob(data_name):
    unique_groups = compute_p_d_qg_and_p_d_q(data_name)
    # p(t|qg), p(q|g)
    compute_p_t_qg(data_name, unique_groups)
    compute_p_q_g(data_name, unique_groups)

    # p(t|q), p(q)
    compute_p_t_q(data_name)
    compute_p_q(data_name)



if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step4_compute_prob(data_name)

    # Load matrices
    prob_d_qg = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(d|qg).npy', allow_pickle=True).item()
    prob_t_qg = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|qg).npy', allow_pickle=True).item()

    # Plot heatmaps
    plot_heatmap(prob_d_qg[0], "Probability Distribution for User Group 0", xlabel_name='prefix (q)',
                 ylabel_name='query (d)')
    plot_heatmap(prob_d_qg[1], "Probability Distribution for User Group 1", xlabel_name='prefix (q)',
                 ylabel_name='query (d)')

    plot_heatmap(prob_t_qg[0], "Probability Distribution for User Group 0", xlabel_name='prefix (q)',
                 ylabel_name='intent (t)')
    plot_heatmap(prob_t_qg[1], "Probability Distribution for User Group 1", xlabel_name='prefix (q)',
                 ylabel_name='intent (t)')
