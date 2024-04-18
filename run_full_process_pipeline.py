import argparse
from data_preprocess.step1_load_data import run_step1_load_data
from data_preprocess.step2_embed_data import run_step2_embed_data
from data_preprocess.step3_generate_prefix import run_step3_generate_prefix
from data_preprocess.step4_compute_prob import run_step4_compute_prob
from data_preprocess.step5_generate_query_ranking import run_step5_compute_prob_exposure
from data_preprocess.step6_success_tq import run_step6_compute_success_tq
from data_preprocess.step7_prob_success import run_step7_prob_success
from data_preprocess.get_statistics import run_statistics

def main(data_name, ncluster, ncount, ranking_method, patience, rand_tau_list):
    run_step1_load_data(data_name)
    run_step2_embed_data(data_name, ncluster=ncluster)
    run_step3_generate_prefix(data_name, ncount=ncount)
    run_step4_compute_prob(data_name)
    print("Statistics:")
    run_statistics(data_name)
    for rand_tau in rand_tau_list:
        print("***********")
        print("Tau:", rand_tau)
        run_step5_compute_prob_exposure(data_name, ranking_method=ranking_method, patience=patience, rand_tau=rand_tau)
        run_step6_compute_success_tq(data_name)
        run_step7_prob_success(data_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data preprocessing and analysis pipeline.')
    parser.add_argument('--data_name', type=str, default='sogou_small', help='The name of the dataset to process.')
    parser.add_argument('--ncluster', type=int, default=10, help='Number of clusters for step 2.')
    parser.add_argument('--ncount', type=int, default=20, help='Count parameter for step 3.')
    parser.add_argument('--ranking_method', type=str, default='MPC', help='Ranking method for step 5.')
    parser.add_argument('--patience', type=float, default=0.8, help='Patience level for step 5.')
    parser.add_argument('--rand_tau', nargs='+', type=float, default=[8, 4, 2, 1, 0.5, 0.25, 0.125, 0], help='Random tau values for step 5.')

    args = parser.parse_args()

    main(args.data_name, args.ncluster, args.ncount, args.ranking_method, args.patience, args.rand_tau)

