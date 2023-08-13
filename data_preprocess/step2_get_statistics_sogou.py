import pandas as pd
import os
import numpy as np
from data_statistics import basic_statistics, prob_query_group, prob_topic_query_group, prob_query

if __name__ == '__main__':
    data_name = 'sogou_small'
    topic_name = 'URLID'  # assuming this is what you want to use as topic
    query_name = 'Query'
    group_name = 'UserGroup'  # assuming this is what you want to use as group

    """data loading"""
    cur_df = pd.read_csv(f'data_preprocessed/{data_name}/updated_{data_name}_processed.csv')  # replace with your sogou data file

    """statistics"""
    print("basic statistics:")
    basic_statistics(cur_df, data_name)

    # Save
    directory = f"file_saved/{data_name}/{topic_name}_{query_name}_{group_name}"

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        p_q, q_order = prob_query(cur_df.copy(deep=True), query_name)
        print(q_order[:5])
        print(q_order[-5:])
        p_qg = prob_query_group(cur_df.copy(deep=True), query_name, group_name, q_order, plot=True)
        p_tqg = prob_topic_query_group(cur_df.copy(deep=True), topic_name, query_name, group_name, q_order, plot=True)

        # Assuming p_q, p_qg, and p_tqg are your numpy arrays
        np.save(f'{directory}/p_q.npy', p_q)

        # For the dictionaries p_qg and p_tqg, we save each entry as a separate .npy file
        for key in p_qg.keys():
            np.save(f'{directory}/p_qg_{key}.npy', p_qg[key])

        for key in p_tqg.keys():
            np.save(f'{directory}/p_tqg_{key}.npy', p_tqg[key])
    else:
        # Load p_q
        p_q = np.load(f'{directory}/p_q.npy')

        # For the dictionaries p_qg and p_tqg, we load each entry as a separate .npy file
        p_qg = {}
        for file_name in os.listdir(directory):
            if 'p_qg' in file_name:
                key = file_name.split('_')[-1].split('.')[0]  # Extract the key from the file name
                p_qg[key] = np.load(f'{directory}/{file_name}')

        p_tqg = {}
        for file_name in os.listdir(directory):
            if 'p_tqg' in file_name:
                key = file_name.split('_')[-1].split('.')[0]  # Extract the key from the file name
                p_tqg[key] = np.load(f'{directory}/{file_name}')

    print("**************")
    print("p_q:", p_q.shape, p_q.sum())

    for key in p_qg.keys():
        print(f"p_qg ({key}):", p_qg[key].shape)
        print(f"p_qg ({key}) sum:", p_qg[key].sum())

    for key in p_tqg.keys():
        print(f"p_tqg ({key}) shape:", p_tqg[key].shape)
        print(f"p_tqg ({key}) sum:", p_tqg[key].sum(0))
