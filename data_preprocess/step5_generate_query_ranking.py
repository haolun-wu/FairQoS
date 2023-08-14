import pandas as pd
import numpy as np


def generate_ranking_matrix(data_name):
    # Load data
    file_path = f'../data_preprocessed/{data_name}'
    i_mapping = pd.read_csv(f"{file_path}/i_mapping.csv")
    i_o_data = pd.read_csv(f"{file_path}/i_o_data.csv")
    o_mapping = pd.read_csv(f"{file_path}/o_mapping.csv")

    # Extract unique query_prefix ids and initialize ranking matrix
    query_prefix_ids = i_mapping["Query_prefix_id"].unique()
    num_query_prefix = len(query_prefix_ids)
    num_queries = len(o_mapping["QueryID"].unique())

    # Create a popularity mapping from o_mapping
    popularity_map = dict(zip(o_mapping["QueryID"], o_mapping["Popularity"]))

    # Initialize a new ranking matrix where each column represents the same query ID
    restructured_ranking_matrix = np.zeros((num_query_prefix, num_queries), dtype=int)

    for index, i in enumerate(query_prefix_ids):
        # Filter out relevant queries (where the pair is present in i_o_data)
        relevant_queries = i_o_data[i_o_data["Query_prefix_id"] == i].copy()
        relevant_queries["Popularity"] = relevant_queries["QueryID"].map(popularity_map)

        # Sort relevant queries by popularity
        relevant_queries_sorted = relevant_queries.sort_values(by="Popularity", ascending=False)

        # Filter and sort non-relevant queries (this includes those not in i_o_data)
        non_relevant_query_ids = set(o_mapping["QueryID"].values) - set(relevant_queries["QueryID"].values)
        non_relevant_queries = o_mapping[o_mapping["QueryID"].isin(non_relevant_query_ids)]
        non_relevant_queries_sorted = non_relevant_queries.sort_values(by="Popularity", ascending=False)

        # Combine the relevant and non-relevant ranked lists
        combined_sorted = pd.concat([relevant_queries_sorted, non_relevant_queries_sorted])

        # Create the ranking list for the current query_prefix and restructure it
        current_ranking = combined_sorted["QueryID"].values
        for rank, query_id in enumerate(current_ranking):
            restructured_ranking_matrix[index, query_id] = rank + 1

    # Save restructured ranking matrix
    np.savetxt(f"{file_path}/i_o_ranking_position.csv", restructured_ranking_matrix, delimiter=",", fmt="%d")
    print("i_o_ranking_position:", restructured_ranking_matrix.shape)

    return restructured_ranking_matrix


def calculate_exposure(rank_matrix, data_name, patience=0.9):
    # User browsing model as RBP
    exposure_matrix = np.power(patience, rank_matrix - 1)

    # Save the exposure matrix to a numpy array file
    file_path = f"../data_preprocessed/{data_name}/i_o_exposure_matrix.npy"
    np.save(file_path, exposure_matrix)

    file_path = f"../data_preprocessed/{data_name}/prob_matrix/prob_o_i_exposure.npy"
    np.save(file_path, exposure_matrix.T)

    print("i_o_exposure_matrix shape:", exposure_matrix.shape)
    print("prob_o_i_exposure shape and sum:", exposure_matrix.T.shape, exposure_matrix.T.sum())


def run_step5_compute_prob_exposure(data_name):
    ranking_matrix = generate_ranking_matrix(data_name)

    # Check each row
    for row in ranking_matrix:
        if len(set(row)) != len(row):
            print("Warning: Duplicate IDs found in a row.")
    print("Check no-duplication for each row successfully.")

    calculate_exposure(ranking_matrix, data_name, patience=0.9)


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step5_compute_prob_exposure(data_name)
