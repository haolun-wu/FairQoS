from data_preprocess.step1_load_data import run_step1_load_data
from data_preprocess.step2_embed_data import run_step2_embed_data
from data_preprocess.step3_generate_prefix import run_step3_generate_prefix
from data_preprocess.step4_compute_prob import run_step4_compute_prob
from data_preprocess.get_statistics import run_statistics

if __name__ == '__main__':
    data_name = 'sogou_small'

    run_step1_load_data(data_name)
    run_step2_embed_data(data_name)
    run_step3_generate_prefix(data_name, n=20)
    run_step4_compute_prob(data_name)
    print("Statistics:")
    run_statistics(data_name)
