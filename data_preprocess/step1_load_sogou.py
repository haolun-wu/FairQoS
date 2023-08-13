import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Set the column names
col_names = ['AccessTime', 'UserID', 'Query', 'URLRank', 'ClickSequence', 'URLID']


# Define a function to parse each line
def parse_line(line):
    # Split the line into fields
    fields = line.strip().split('\t')

    # The third field contains both the URL rank and the ClickSequence, so split it further
    if ' ' in fields[3]:
        url_rank, click_sequence = fields[3].split(' ')
    else:
        url_rank = fields[3]
        click_sequence = None  # Or any default value

    # The fifth field is the URLID
    url_id = fields[4]

    # Create a new line with the correct fields
    new_line = fields[:3] + [url_rank, click_sequence, url_id] + fields[5:]

    return new_line


# Load the data
data_name = 'sogou'

if data_name == 'sogou_small':
    with open('../data/sogou/SogouQ.sample', 'r', encoding='gb18030') as f:
        data = [parse_line(line) for line in f]
elif data_name == 'sogou':
    with open('../data/sogou/SogouQ.reduced', 'r', encoding='gb18030') as f:
        data = [parse_line(line) for line in f]

# Convert data list to pandas DataFrame
data_df = pd.DataFrame(data, columns=col_names)

# Counting the number of queries for each user
user_query_count = data_df['UserID'].value_counts()

# Categorizing users into 'active' and 'less active' based on the median of their query counts
data_df['UserGroup'] = np.where(data_df['UserID'].map(user_query_count) > user_query_count.median(), 'Active',
                                    'Less Active')

# Saving processed data
data_df.to_csv(f'../data_preprocessed/{data_name}/{data_name}_processed.csv', index=False)

print("sogou_data:", data_df)
