import numpy as np


def run_step6_compute_p_t_i_success(data_name):
    # Load the probabilities from the files
    prob_o_t = np.load(f"../data_preprocessed/{data_name}/prob_matrix/prob_o_t.npy")
    prob_o_i_exposure = np.load(f"../data_preprocessed/{data_name}/prob_matrix/prob_o_i_exposure.npy")

    # Check the shapes for compatibility
    assert prob_o_t.shape[0] == prob_o_i_exposure.shape[
        0], "Mismatch in number of queries (o) between the two matrices."

    # Initialize the result matrix
    num_t = prob_o_t.shape[1]
    num_i = prob_o_i_exposure.shape[1]
    p_t_i_success = np.zeros((num_t, num_i))

    # Compute the probability for every combination of t and i
    for t in range(num_t):
        for i in range(num_i):
            product_term = (1 - prob_o_t[:, t] * prob_o_i_exposure[:, i]).prod()
            p_t_i_success[t, i] = 1 - product_term

    file_path = f"../data_preprocessed/{data_name}/prob_matrix/prob_t_i_success.npy"
    np.save(file_path, p_t_i_success)
    print("p_t_i_success:", p_t_i_success.shape)
    print("p_t_i_success:", p_t_i_success.sum())

    return p_t_i_success


if __name__ == "__main__":
    data_name = 'sogou_small'
    p_t_i_success = run_step6_compute_p_t_i_success(data_name)
