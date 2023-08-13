import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy
import math
import pandas as pd


def basic_statistics(cur_df, data_name):
    if data_name in ['ml1m']:
        # @title basic statistics
        unique_genres = cur_df['user_id'].unique()
        print("# of unique users:", len(unique_genres))
        unique_genres = cur_df['item_id'].unique()
        print("# of unique items:", len(unique_genres))
        unique_genres = cur_df['gender'].unique()
        print("# of unique gender:", len(unique_genres))
        unique_genres = cur_df['age'].unique()
        print("# of unique age:", len(unique_genres))

        unique_genres = cur_df['genre'].str.split('|').explode().unique()
        print("# of unique genres:", len(unique_genres))
        unique_genres = cur_df['director'].str.split('|').explode().unique()
        print("# of unique directors:", len(unique_genres))
        unique_genres = cur_df['actor'].str.split('|').explode().unique()
        print("# of unique actors:", len(unique_genres))
    elif data_name in ['sogou', 'sogou_small']:
        # @title basic statistics
        unique_users = cur_df['UserID'].unique()
        print("# Users:", len(unique_users))
        unique_queries = cur_df['Query'].unique()
        print("# Queries:", len(unique_queries))
        unique_clusters = cur_df['ClusterID'].unique()
        print("# Clusters:", len(unique_clusters))
        unique_urls = cur_df['URLID'].unique()
        print("# Urls:", len(unique_urls))
        unique_groups = cur_df['UserGroup'].unique()
        print("# UserGroup:", len(unique_groups))

def plot_heatmaps(pivot_tables, k):
    # Determine number of rows and columns for the subplot grid
    num_groups = len(pivot_tables)
    num_cols = min(2, num_groups)
    num_rows = math.ceil(num_groups / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10 * num_cols, 6 * num_rows))

    # Flatten axs or make it a list if it's a single AxesSubplot object
    if num_groups > 1:
        if num_rows > 1:
            axs = [ax for sublist in axs for ax in sublist]
    else:
        axs = [axs]

    # Create colormap
    cmap = sns.cubehelix_palette(5, gamma=0.6, reverse=True, as_cmap=True)

    for i, group in enumerate(pivot_tables.keys()):
        pivot_table = pivot_tables[group]
        top_k_queries = pivot_table.columns[:k]

        sns.heatmap(pivot_table[top_k_queries], cmap=cmap, cbar=True, ax=axs[i])
        axs[i].set_title(f'Top {k} Queries for Group {group}')
        axs[i].set_xlabel('Queries')
        axs[i].set_ylabel('Topics')

    # Delete any unused subplots
    for i in range(num_groups, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def prob_query(cur_df, query_name):
    # p(q)
    cur_df[query_name] = cur_df[query_name].str.split('|')
    cur_df = cur_df.explode(query_name)

    # Compute the count for each query
    query_counts = cur_df[query_name].value_counts()

    # Compute the probability of each query
    query_prob = query_counts / query_counts.sum()

    # Convert to numpy array
    query_prob_np = query_prob.values

    return query_prob_np, query_prob.index.tolist()  # return both the probabilities and the corresponding queries


def prob_query_group(cur_df, query_name, group_name, query_order, plot=True, num_cols=2):
    # p(q|g)
    # Split the query_name column into separate queries and explode the values
    cur_df[query_name] = cur_df[query_name].str.split('|')
    cur_df = cur_df.explode(query_name)

    # Group by group_name and query_name, and calculate the count
    query_group_counts = cur_df.groupby([group_name, query_name]).size().reset_index(name='count')

    # Compute the total count for each group_name
    group_totals = cur_df.groupby(group_name).size().reset_index(name='group_total')

    # Compute the conditional probability of query_name given group_name
    query_group_prob = query_group_counts.merge(group_totals, on=group_name)
    query_group_prob['p(query|group)'] = query_group_prob['count'] / query_group_prob['group_total']

    # Convert query_group_prob to a new dataframe with group_name columns and query_name rows
    query_group_df = query_group_prob.pivot(index=query_name, columns=group_name, values='p(query|group)').fillna(0)

    # Now, normalize each column in the dataframe to ensure the sum of probabilities is 1
    query_group_df = query_group_df.divide(query_group_df.sum(axis=0), axis=1)

    # Use query_order to sort the dataframe
    query_group_df = query_group_df.reindex(query_order)

    # Fill missing values with 0
    query_group_df = query_group_df.fillna(0)
    print("query_group_df:", query_group_df)

    if plot:
        # Set the number of top queries to consider
        k = min(500, len(query_order))
        top_k_query = query_group_df.head(k)

        # Retrieve the unique groups
        groups = cur_df[group_name].unique()

        # Determine the number of rows and columns for the subplot grid
        num_groups = len(groups)
        num_cols = min(num_cols, num_groups)
        num_rows = math.ceil(num_groups / num_cols)

        # Create a figure with n subplots, where n is the number of unique groups
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))

        # Flatten axes in case it's a multi-dimension array
        axes = axes.flatten() if num_groups > 1 else [axes]

        # Iterate over each group
        for i, group in enumerate(groups):
            # Sort the dataframe by the current group column in descending order
            sorted_query_group_df = top_k_query.sort_values(by=group, ascending=False)

            # Iterate over each group and plot a line for each
            for grp in groups:
                axes[i].plot(sorted_query_group_df.index, sorted_query_group_df[grp], label=grp)

            axes[i].set_xlabel(query_name)
            axes[i].set_ylabel('Probability')
            axes[i].set_title(f"p({query_name}|{group_name}) - Top {k} {query_name}s sorted by {group}")
            axes[i].legend()
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=90)

        # Display the plots
        plt.tight_layout()
        plt.show()

    query_group_np = {}
    for unique_group in cur_df[group_name].unique():
        query_group_np[unique_group] = query_group_df[unique_group].values
    return query_group_np



def prob_topic_query_group(cur_df, topic_name, query_name, group_name, query_order, plot=True):
    # p(t|q, g) <=> p(genre|actor, gender)

    # explode the dataframe on 'genre' and 'director' columns
    cur_df[topic_name] = cur_df[topic_name].str.split('|')
    cur_df = cur_df.explode(topic_name)

    cur_df[query_name] = cur_df[query_name].str.split('|')
    cur_df = cur_df.explode(query_name)

    # compute total count for each query item
    query_counts = cur_df[query_name].value_counts().reset_index()
    query_counts.columns = [query_name, 'count']

    # convert the query order numpy array to a pandas series
    query_order = pd.Series(query_order)

    # compute counts for each combination of target, query and group
    counts = cur_df.groupby([topic_name, query_name, group_name]).size().reset_index(name='counts')

    # Convert counts to probabilities by dividing by the sum for each combination of query and group
    total_counts = counts.groupby([query_name, group_name])['counts'].transform('sum')
    counts['prob'] = counts['counts'] / total_counts

    # create a pivot table for each group
    pivot_tables = {}
    for unique_group in cur_df[group_name].unique():
        pivot_table = counts[counts[group_name] == unique_group].pivot_table(index=topic_name, columns=query_name, values='prob').fillna(0)

        # reorder the pivot table according to query_order and fill missing entries with 0
        pivot_table = pivot_table.reindex(query_order, axis=1).fillna(0)

        pivot_tables[unique_group] = pivot_table
        print("pivot_tables:", pivot_tables)

    if plot:
        # Call the function with your desired number of queries
        plot_heatmaps(pivot_tables, k=min(500, len(query_order)))

    topic_query_group_np = {}
    for unique_group in cur_df[group_name].unique():
        topic_query_group_np[unique_group] = pivot_tables[unique_group].values.reshape((pivot_tables[unique_group].shape[0], pivot_tables[unique_group].shape[1]))
    return topic_query_group_np

# def prob_topic_group(cur_df, topic_name, query_name, group_name, plot=True):
#     # @title p(t|g) <=> p(genre|gender)
#
#     # explode the dataframe on 'genre' and 'director' columns
#     cur_df[topic_name] = cur_df[topic_name].str.split('|')
#     cur_df = cur_df.explode(topic_name)
#
#     cur_df[query_name] = cur_df[query_name].str.split('|')
#     cur_df = cur_df.explode(query_name)
#
#     # compute total count for each query item
#     query_counts = cur_df[query_name].value_counts().reset_index()
#     query_counts.columns = [query_name, 'count']
#
#     # sort queries based on total count
#     sorted_queries = query_counts[query_name]
#
#     # compute counts for each combination of target, query and group
#     counts = cur_df.groupby([topic_name, query_name, group_name]).size().reset_index(name='counts')
#
#     # convert counts to probabilities by dividing by the sum for each group
#     total_counts = counts.groupby([group_name])['counts'].transform('sum')
#     counts['prob'] = counts['counts'] / total_counts
#
#     # create a pivot table for each group
#     pivot_tables = {}
#     for unique_group in cur_df[group_name].unique():
#         pivot_tables[unique_group] = counts[counts[group_name] == unique_group].pivot_table(index=topic_name,
#                                                                                             columns=query_name,
#                                                                                             values='prob').fillna(0)
#         pivot_tables[unique_group] = pivot_tables[unique_group].reindex(sorted_queries, axis=1).fillna(
#             0)  # Reorder and fill missing entries with 0
#
#     if plot:
#         # Call the function with your desired number of queries
#         plot_heatmaps(pivot_tables, k=1000)
#
#     topic_group_np = {}
#     for unique_group in cur_df[group_name].unique():
#         topic_group_np[unique_group] = pivot_tables[unique_group].values.reshape(
#             (pivot_tables[unique_group].shape[0], pivot_tables[unique_group].shape[1]))
#     return topic_group_np
