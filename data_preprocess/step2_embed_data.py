import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel, BertTokenizerFast, AutoModel
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
from pypinyin import lazy_pinyin
from scipy.special import softmax


# Load the saved CSV
def load_data(file_path):
    return pd.read_csv(file_path)


# Obtain BERT embeddings for queries
def get_bert_embeddings(queries):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # model = BertModel.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-tiny-chinese')

    embeddings = []

    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs[0].mean(1).detach().numpy().flatten())

    return np.array(embeddings)


# Cluster the embeddings using Ward's method
def cluster_embeddings(embeddings, n_clusters=None):
    if not n_clusters:
        linkage_array = ward(embeddings)
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_array)
        plt.show()
        n_clusters = int(input("Enter the number of clusters based on the dendrogram: "))

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(embeddings)

    cluster_centers = []
    for label in set(labels):
        members = embeddings[labels == label]
        cluster_center = members.mean(axis=0)
        cluster_centers.append(cluster_center)

    return labels, np.array(cluster_centers)


def similarity_and_probabilities(query_embeddings, cluster_centers):
    # 1. Compute similarity matrix using inner products
    similarity_matrix = np.dot(query_embeddings, cluster_centers.T)

    # 2. Compute p(query|center) - softmax over columns
    p_query_given_center = softmax(similarity_matrix, axis=0)

    # 3. Compute p(center|query) - softmax over rows
    p_center_given_query = softmax(similarity_matrix, axis=1).T

    return similarity_matrix, p_query_given_center, p_center_given_query


def run_step2_embed_data(data_name, ncluster=20):
    data_name = 'sogou_small'
    file_path = f"../data_preprocessed/{data_name}/raw_{data_name}_processed.csv"
    df = load_data(file_path)

    # Clean up 'Query' column
    df['Query'] = df['Query'].str.replace("[", "").str.replace("]", "").str.replace(" ", "")

    # Compute the popularity of each unique query as its frequency in the dataset
    query_popularity = df['Query'].value_counts().to_dict()

    # Assign each unique query an ID
    unique_queries = df["Query"].unique()
    query_id_map = {query: i for i, query in enumerate(unique_queries)}
    query_pinyin_map = {query: ' '.join(lazy_pinyin(query)) for query in unique_queries}

    # Save the ID, pinyin mapping, and popularity to a dataframe
    query_df = pd.DataFrame(list(query_id_map.items()), columns=["Query", "QueryID"])
    query_df['Query_pinyin'] = query_df['Query'].map(query_pinyin_map)
    query_df['Popularity'] = query_df['Query'].map(query_popularity)  # Adding the popularity column

    # Obtain embeddings
    query_embeddings = get_bert_embeddings(query_df["Query"].values)

    # Cluster embeddings and obtain cluster labels and centers
    cluster_labels, cluster_centers = cluster_embeddings(query_embeddings, n_clusters=ncluster)

    # Map queries to their cluster IDs
    query_cluster_map = dict(zip(unique_queries, cluster_labels))

    # Save query ID, cluster labels, embeddings, and cluster centers
    df["QueryID"] = df["Query"].map(query_id_map)
    df["ClusterID"] = df["Query"].map(query_cluster_map)
    df['Query_pinyin'] = df['Query'].map(query_pinyin_map)
    query_df["ClusterID"] = query_df["Query"].map(query_cluster_map)

    # Save
    query_df.to_csv(f"../data_preprocessed/{data_name}/o_mapping.csv", index=False)
    np.save(f"../data_preprocessed/{data_name}/embedding_matrix/o_emb_matrix.npy", query_embeddings)
    np.save(f"../data_preprocessed/{data_name}/embedding_matrix/t_emb_matrix.npy", cluster_centers)
    df.to_csv(f"../data_preprocessed/{data_name}/new_{data_name}_processed.csv", index=False)

    print(f"Obtained {len(cluster_centers)} cluster centers.")

    # Combute p(query|center) and p(center|query)
    similarity_matrix, p_ot, p_to = similarity_and_probabilities(query_embeddings, cluster_centers)
    print("p_ot:", p_ot.shape)
    print("p_ot:", p_ot.sum(0))
    print("p_to:", p_to.shape)
    print("p_to:", p_to.sum(0))
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/prob_o_t.npy", p_ot)
    np.save(f"../data_preprocessed/{data_name}/prob_matrix/prob_t_o.npy", p_to)


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step2_embed_data(data_name, ncluster=20)
