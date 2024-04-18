import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel, BertTokenizerFast, AutoModel
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
from pypinyin import lazy_pinyin
from scipy.special import softmax
import torch

# from cuml.decomposition import PCA as cuPCA
# from cuml.cluster import KMeans as cuKMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the saved CSV
def load_data(file_path):
    return pd.read_csv(file_path)


# Obtain BERT embeddings for queries
def get_bert_embeddings(queries):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-tiny-chinese')

    # Move model to GPU
    model.to(device)

    embeddings = []

    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

        with torch.no_grad():  # Disable gradient calculation to save memory and speed up
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(1))  # Keep as tensor

        # Concatenate all embeddings along the batch dimension
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


# Cluster the embeddings using Ward's method
# def cluster_embeddings(embeddings, n_clusters=None):
#     if not n_clusters:
#         linkage_array = ward(embeddings)
#         plt.figure(figsize=(10, 5))
#         dendrogram(linkage_array)
#         plt.show()
#         n_clusters = int(input("Enter the number of clusters based on the dendrogram: "))
#
#     cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
#     labels = cluster.fit_predict(embeddings)
#
#     cluster_centers = []
#     for label in set(labels):
#         members = embeddings[labels == label]
#         cluster_center = members.mean(axis=0)
#         cluster_centers.append(cluster_center)
#
#     return labels, np.array(cluster_centers)
# Kmeans
def cluster_embeddings(embeddings, n_clusters=None):
    if not n_clusters:
        n_clusters = int(input("Enter the number of clusters: "))

    # Initialize and fit the K-Means model on the reduced embeddings
    cluster = KMeans(n_clusters=n_clusters, random_state=42)
    labels = cluster.fit_predict(embeddings)
    cluster_centers = cluster.cluster_centers_

    return labels, cluster_centers


def similarity_and_probabilities(query_embeddings, cluster_centers):
    query_embeddings = torch.tensor(query_embeddings, device=device)
    cluster_centers = torch.tensor(cluster_centers, device=device)
    # 1. Compute similarity matrix using inner products
    similarity_matrix = torch.matmul(query_embeddings, cluster_centers.T)

    # 2. Compute p(query|center) - softmax over columns
    p_query_given_center = torch.softmax(similarity_matrix, dim=0).cpu().numpy()

    # 3. Compute p(center|query) - softmax over rows
    p_center_given_query = torch.softmax(similarity_matrix, dim=1).T.cpu().numpy()

    return similarity_matrix, p_query_given_center, p_center_given_query


def run_step2_embed_data(data_name, ncluster=20):
    file_path = f"./data_preprocessed/{data_name}/raw_{data_name}_processed.csv"
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
    query_embeddings = get_bert_embeddings(query_df["Query"].values).cpu().numpy()
    print("query_embeddings:", query_embeddings.shape)

    pca = PCA(n_components=8)
    query_embeddings = pca.fit_transform(query_embeddings)
    print("query_embeddings:", query_embeddings.shape)

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
    query_df.to_csv(f"./data_preprocessed/{data_name}/d_t_mapping.csv", index=False)
    np.save(f"./data_preprocessed/{data_name}/embedding_matrix/d_emb_matrix.npy", query_embeddings)
    np.save(f"./data_preprocessed/{data_name}/embedding_matrix/t_emb_matrix.npy", cluster_centers)
    df.to_csv(f"./data_preprocessed/{data_name}/new_{data_name}_processed.csv", index=False)

    print(f"Obtained {len(cluster_centers)} cluster centers.")

    # Combute p(query|center) and p(center|query)
    similarity_matrix, p_dt, p_td = similarity_and_probabilities(query_embeddings, cluster_centers)
    print("p_d|t:", p_dt.shape)
    print("p_d|t:", p_dt.sum(0))
    print("p_t|d:", p_td.shape)
    print("p_t|d:", p_td.sum(0))
    np.save(f"./data_preprocessed/{data_name}/prob_matrix/p(d|t).npy", p_dt)
    np.save(f"./data_preprocessed/{data_name}/prob_matrix/p(t|d).npy", p_td)


if __name__ == "__main__":
    data_name = 'sogou_small'
    run_step2_embed_data(data_name, ncluster=20)
