import numpy as np


def run_step6_compute_success_tq(data_name):
    # Load the probabilities from the files
    prob_d_t = np.load(f"./data_preprocessed/{data_name}/prob_matrix/p(d|t).npy")
    prob_d_q_exposure = np.load(f"./data_preprocessed/{data_name}/prob_matrix/p(d_exposure|q).npy")

    # print("prob_d_t:", prob_d_t.shape)
    # print("prob_d_q_exposure:", prob_d_q_exposure.shape)

    # Check the shapes for compatibility
    assert prob_d_t.shape[0] == prob_d_q_exposure.shape[
        0], "Mismatch in number of queries (q) between the two matrices."

    # Initialize the result matrix
    num_t = prob_d_t.shape[1]
    num_q = prob_d_q_exposure.shape[1]
    # success_d_tq = np.zeros((num_t, num_q))
    #
    # # Compute the probability for every combination of t and q
    # for t in range(num_t):
    #     for q in range(num_q):
    #         product_term = (1 - prob_d_t[:, t] * prob_d_q_exposure[:, q]).prod()
    #         success_d_tq[t, q] = 1 - product_term
    prob_d_t = prob_d_t[:, :, np.newaxis]
    # Then broadcast multiply it with p_e_d_given_q which gets automatically broadcasted from (#d, #q) to (#d, 1, #q)
    success_d_tq = prob_d_t * prob_d_q_exposure[:, np.newaxis, :]

    file_path = f"./data_preprocessed/{data_name}/prob_matrix/p(s_d|tq).npy"
    np.save(file_path, success_d_tq)
    # print("success_d_tq:", success_d_tq.shape)

    # Compute p(s|t,q)
    complement_ps_d_tq = 1 - success_d_tq

    # Product of the complement probabilities along the document dimension
    product_complement = complement_ps_d_tq.prod(axis=0)

    # Subtract the product from 1 to get p(s|t,q)
    success_tq = 1 - product_complement
    file_path = f"./data_preprocessed/{data_name}/prob_matrix/p(s|tq).npy"
    np.save(file_path, success_tq)
    # print("success_tq:", success_tq.shape)

    return success_tq

if __name__ == "__main__":
    data_name = 'sogou_small'
    success_tq = run_step6_compute_success_tq(data_name)
