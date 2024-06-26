from data_preprocess.step1_load_data import run_step1_load_data
from data_preprocess.step2_embed_data import run_step2_embed_data
from data_preprocess.step3_generate_prefix import run_step3_generate_prefix
from data_preprocess.step4_compute_prob import run_step4_compute_prob
from step5_generate_query_ranking import run_step5_compute_prob_exposure
from step6_success_tq import run_step6_compute_success_tq
from step7_prob_success import run_step7_prob_success
from data_preprocess.get_statistics import run_statistics

if __name__ == '__main__':
    data_name = 'sogou'

    run_step1_load_data(data_name)
    run_step2_embed_data(data_name, ncluster=20)
    run_step3_generate_prefix(data_name, ncount=20)
    run_step4_compute_prob(data_name)
    print("Statistics:")
    run_statistics(data_name)
    for rand_tau in [8, 4, 2, 1, 0.5, 0.25, 0.125, 0]:
        print("Tau:", rand_tau)
        run_step5_compute_prob_exposure(data_name, ranking_method="MPC", patience=0.8, rand_tau=rand_tau)
        run_step6_compute_success_tq(data_name)
        run_step7_prob_success(data_name)
