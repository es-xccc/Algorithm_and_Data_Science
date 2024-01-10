import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_kmeans(data, features, n_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(scaled_features)
    data['Cluster'] = kmeans.labels_

    cluster_sizes = data['Cluster'].value_counts()
    cluster_deaths = data.groupby('Cluster')['Label'].sum()

    print(f"Test k-means (k = {n_clusters})")

    for cluster in cluster_sizes.index:
        size = cluster_sizes[cluster]
        deaths = cluster_deaths[cluster]
        fraction = deaths / size
        print(f"Cluster of size {size} with fraction of death positives = {fraction:.4f} and {deaths} death.")

data = pd.read_csv('cardiacPatientData.txt', comment='#', header=None)
data.columns = ['Heart rate', 'Heart attacks', 'Age', 'ST elevation', 'Label']

features = ['Heart rate', 'Heart attacks', 'Age', 'ST elevation']

perform_kmeans(data, features, n_clusters=2)
print('-' * 50)
perform_kmeans(data, features, n_clusters=3)
print('-' * 50)
perform_kmeans(data, features, n_clusters=4)