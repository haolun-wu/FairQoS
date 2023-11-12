import numpy as np


def compute_sum_of_product(data_name):
    # Load necessary matrices
    p_i = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_i.npy')
    p_t_i_success = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_i_success.npy')
    p_t_ig_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_ig_dict.npy', allow_pickle=True).item()

    sum_i = 0.0
    for i, prob_i in enumerate(p_i):
        prod_g = 1.0
        for _, p_t_ig in p_t_ig_dict.items():
            sum_t = np.dot(p_t_ig[:, i], p_t_i_success[:, i])
            prod_g *= sum_t
        sum_i += prod_g * prob_i

    return sum_i


def compute_product_of_sum(data_name):
    # Load necessary matrices
    p_i_g_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_i_g_dict.npy', allow_pickle=True).item()
    p_t_i_success = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_i_success.npy')
    p_t_ig_dict = np.load(f'../data_preprocessed/{data_name}/prob_matrix/prob_t_ig_dict.npy', allow_pickle=True).item()

    prod_g = 1.0
    for g in p_i_g_dict.keys():
        sum_i = 0.0
        for i, _ in enumerate(p_i_g_dict[g]):
            sum_t = np.dot(p_t_ig_dict[g][:, i], p_t_i_success[:, i])
            sum_i += sum_t * p_i_g_dict[g][i]
        prod_g *= sum_i

    return prod_g


def run_step7_prob_success(data_name):
    p_s_sum_of_product = compute_sum_of_product(data_name)
    p_s_product_of_sum = compute_product_of_sum(data_name)
    print(f"Sum-of-Product p(S): {p_s_sum_of_product}")
    print(f"Product-of-Sum p(S): {p_s_product_of_sum}")
    return


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step7_prob_success(data_name)
