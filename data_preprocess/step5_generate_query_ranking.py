import pandas as pd
import numpy as np
from scipy.special import softmax


def add_gumbel_noise_and_rank(scores):
    # Adding Gumbel noise to each score
    # Gumbel(0,1) can be sampled using -log(-log(U)) where U is uniform(0,1)
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, size=scores.shape)))
    noisy_scores = scores + gumbel_noise

    # Getting indices of scores sorted from highest to lowest
    sorted_indices = np.argsort(-noisy_scores)

    return sorted_indices


def run_step5_compute_prob_exposure(data_name, ranking_method="MPC", patience=0.8, rand_tau=0):
    # Load data
    file_path = f'../data_preprocessed/{data_name}'
    q_mapping = pd.read_csv(f"{file_path}/q_mapping.csv")
    d_q_mapping = pd.read_csv(f"{file_path}/d_q_mapping.csv")
    d_t_mapping = pd.read_csv(f"{file_path}/d_t_mapping.csv")

    # Extract unique query_prefix ids and initialize ranking matrix
    query_prefix_ids = q_mapping["Query_prefix_id"].unique()
    num_query_prefix = len(query_prefix_ids)
    num_queries = len(d_t_mapping["QueryID"].unique())

    # Create a popularity mapping from d_t_mapping
    popularity_map = dict(zip(d_t_mapping["QueryID"], d_t_mapping["Popularity"]))

    # Initialize a new ranking matrix where each column represents the same query ID
    ranking_matrix = np.zeros((num_query_prefix, num_queries), dtype=int)
    exposure_matrix = np.zeros((num_query_prefix, num_queries))

    for index, i in enumerate(query_prefix_ids):
        # Filter out relevant queries (where the pair is present in d_q_mapping)
        relevant_queries = d_q_mapping[d_q_mapping["Query_prefix_id"] == i].drop_duplicates().copy()
        relevant_queries["Popularity"] = relevant_queries["QueryID"].map(popularity_map)

        # Filter and sort non-relevant queries (this includes those not in d_q_mapping)
        non_relevant_query_ids = set(d_t_mapping["QueryID"].values) - set(relevant_queries["QueryID"].values)
        non_relevant_queries = d_t_mapping[d_t_mapping["QueryID"].isin(non_relevant_query_ids)]
        non_relevant_queries = non_relevant_queries.copy()
        non_relevant_queries.loc[:, "Popularity"] -= 1000

        # Combine the relevant and non-relevant ranked lists
        combined_queries = pd.concat([relevant_queries, non_relevant_queries])

        # Use which model to offer score
        if ranking_method == "MPC":
            combined_queries["scores"] = combined_queries["Popularity"]
            scores_df = combined_queries.sort_values(by="scores", ascending=False)
            scores_array = scores_df["scores"].values
            id_array = scores_df["QueryID"].values

        if rand_tau == 0:
            # Create the ranking list for the current query_prefix and restructure it
            for rank, query_id in enumerate(id_array):
                ranking_matrix[index, query_id] = rank
        else:
            # print("scores_array:", scores_array.shape)
            scores_array[scores_array < 0] = 0
            scores_array = scores_array / np.sum(scores_array)
            weight = softmax(scores_array / rand_tau)

            sample_times = 50

            for sample_epoch in range(sample_times):
                exp_vector = np.power(patience, np.arange(num_queries)).astype("float")  # pre-compute the exposure
                selected_id = np.random.choice(num_queries, num_queries, replace=False, p=weight)
                exposure_matrix[index][selected_id] += exp_vector

    if rand_tau == 0:
        # Check each row
        for row in ranking_matrix:
            if len(set(row)) != len(row):
                print("Warning: Duplicate IDs found in a row.")
        # print("Check no-duplication for each row successfully.")

        exposure_matrix = np.power(patience, ranking_matrix)
    else:
        exposure_matrix /= sample_times

    # Save the exposure matrix to a numpy array file
    file_path = f"../data_preprocessed/{data_name}/d_q_exposure_matrix.npy"
    np.save(file_path, exposure_matrix)

    file_path = f"../data_preprocessed/{data_name}/prob_matrix/p(d_exposure|q).npy"
    np.save(file_path, exposure_matrix.T)

    # print("q_d_exposure_matrix shape:", exposure_matrix.shape)
    # print("p(d_exposure|q) shape and sum:", exposure_matrix.T.shape, exposure_matrix.T.sum())


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step5_compute_prob_exposure(data_name, ranking_method="MPC", patience=0.8, rand_tau=1)
