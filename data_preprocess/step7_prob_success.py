import numpy as np


def compute_p_S_sum_of_product(data_name):
    # Load necessary matrices
    p_t_qg_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|qg).npy', allow_pickle=True).item()
    p_s_tq = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(s|tq).npy')
    p_q = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(q).npy')

    # Initialize p(S)
    p_S = 0

    # Iterate over all queries (assumes queries are the columns in p(s|t,q))
    for q_index in range(p_s_tq.shape[1]):
        # Initialize the product term for each group as 1 (since we will be multiplying terms)
        group_product_terms = np.ones(p_s_tq.shape[0])

        # Iterate over each user group 'g'
        for g, p_t_qg in p_t_qg_dict.items():
            # Multiply p(t|q,g) for this query by p(s|t,q) for this query
            # p_t_qg[:, q_index] is a 1D array for all intents given q
            # p_s_tq[:, q_index] is also a 1D array for all intents given q
            ptq_ps_tq = p_t_qg[:, q_index] * p_s_tq[:, q_index]

            # Multiply the resulting 1D array with the existing group product terms
            group_product_terms *= (1 - ptq_ps_tq)

        # The product term for all groups has been computed; now calculate its sum
        sum_over_groups = 1 - group_product_terms.prod()

        # Multiply by p(q) and add to p(S)
        p_S += sum_over_groups * p_q[q_index]

    return p_S


def compute_p_S_product_of_sum(data_name):
    # Load necessary matrices
    p_t_qg_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|qg).npy', allow_pickle=True).item()
    p_s_tq = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(s|tq).npy')
    p_q_g_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(q|g).npy', allow_pickle=True).item()

    # Initialize p(S) as 1 because we will be multiplying terms
    p_S = 1

    # Iterate over each user group 'g'
    for g in p_t_qg_dict.keys():
        # Retrieve p(t|q,g) for this user group
        p_t_qg = p_t_qg_dict[g]

        # Retrieve p(q|g) for this user group
        p_q_g = p_q_g_dict[g]

        # Compute the inner sum over queries and intents
        inner_sum = sum((p_t_qg * p_s_tq).sum(axis=0) * p_q_g)

        # Multiply across the groups
        p_S *= inner_sum

    return p_S


def compute_p_S_given_q(data_name):
    # Load the probabilities
    p_t_qg_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|qg).npy', allow_pickle=True).item()
    p_s_tq = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(s|tq).npy')

    # Number of queries is the second dimension in p(s|t,q)
    num_queries = p_s_tq.shape[1]

    # Initialize the array for p(S|q) with the size of the number of queries
    p_S_given_q = np.ones(num_queries)

    # Iterate over each query
    for q_index in range(num_queries):
        # Initialize the product term for each user group as 1
        # because we will be multiplying terms for each group
        group_product_terms = np.ones(len(p_t_qg_dict.keys()))

        # Iterate over each user group 'g'
        for g_index, (g, p_t_qg) in enumerate(p_t_qg_dict.items()):
            # Sum over intents t for a specific query q
            sum_over_intents = np.sum(p_t_qg[:, q_index] * p_s_tq[:, q_index])

            # Update the group product term for this group
            group_product_terms[g_index] = sum_over_intents

        # Compute the product of the group terms for this query
        p_S_given_q[q_index] = np.prod(group_product_terms)

    return p_S_given_q


def compute_p_S_given_q_no_group(data_name):
    # Load the probabilities
    p_t_q = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(t|q).npy')
    p_s_tq = np.load(f'../data_preprocessed/{data_name}/prob_matrix/p(s|tq).npy')

    # Compute p(S|q) by summing over intents (assumes first dimension of p(s|t,q) is intents)
    p_S_given_q_no_group = np.sum(p_t_q * p_s_tq, axis=0)

    return p_S_given_q_no_group


def run_step7_prob_success(data_name):
    p_S_sum_of_product = compute_p_S_sum_of_product(data_name)
    p_S_product_of_sum = compute_p_S_product_of_sum(data_name)
    p_S_given_q = compute_p_S_given_q(data_name)
    p_S_given_q_no_group = compute_p_S_given_q_no_group(data_name)
    print("Sum-of-Product p(S):", p_S_sum_of_product)
    print("Product-of-Sum p(S):", p_S_product_of_sum)
    print("single query p(S|q):", len(p_S_given_q), p_S_given_q.mean(), p_S_given_q.std())
    print("single query p(S|q) no group:", len(p_S_given_q_no_group), p_S_given_q_no_group.mean(), p_S_given_q_no_group.std())
    return


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step7_prob_success(data_name)
