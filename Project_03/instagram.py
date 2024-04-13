# Roll Number   : 21CS10042
# Project Code  : IUHC-AS
# Project Title : Instagram User Dynamics using Single Linkage Agglomerative (Bottom-Up) Clustering Technique

import csv
import numpy as np
import os

import time

# Set random seed for reproducibility
np.random.seed(42)

log_file = open("report_logs.txt", "w")

# Helper function to convert string numbers to integers
def farm_number(data):
    data = data.strip('"')
    data = data.replace(",", "")
    
    return int(data)

# Step 1: Read the dataset
def read_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            # Convert string numbers to integers
            posts = farm_number(row[1])
            followers = farm_number(row[2])
            followings = farm_number(row[3])
            likes_10 = farm_number(row[4])
            likes_11 = farm_number(row[5])
            likes_12 = farm_number(row[6])
            self_presenting_posts = farm_number(row[7])
            gender = (row[8]=='m')
            data.append([posts, followers, followings, likes_10, likes_11, likes_12, self_presenting_posts, gender])
    return np.array(data)

# Step 2: K-means Clustering
def k_means_clustering(data, k, tmp=0, max_iterations=20):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    # Assign each data point to the nearest centroid
    for _ in range(max_iterations):
        distances = np.sqrt(np.sum((data - centroids[:, np.newaxis])**2, axis=2))
        labels = np.argmin(distances, axis=0)
        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    # Save clustering information into a file
    
    filename = "kmeanstmp.txt" if tmp else "kmeans.txt"
    
    clusters = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_indices = cluster_indices.tolist()
        
        cluster_indices.sort()
        clusters.append(cluster_indices)
        
    clusters.sort(key=lambda x: x[0])
        
    with open(filename, "w") as file:
        for cluster in clusters:
            file.write(",".join(map(str, cluster)))
            file.write("\n")
            
    file.close()

# Step 3: Silhouette Coefficient Calculation
def silhouette_coefficient(data, labels):
    n = len(data)
    
    clusters = np.unique(list(labels.values()))
    
    if len(clusters) == 1:
        return np.nan
    
    silhouette_values = []
    for i in range(n):
        cluster_i = labels[i]
        if np.sum(np.fromiter((labels[i] == cluster_i for i in range(n)), dtype=bool)) == 1:  # Skip calculation for data points in empty clusters
            continue
            
        a_i = np.mean([np.linalg.norm(data[i] - data[j]) for j in range(n) if labels[j] == cluster_i and i != j])
        
        b_i_values = [np.mean([np.linalg.norm(data[i] - data[j]) for j in range(n) if labels[j] == c and c != cluster_i]) for c in clusters if c != cluster_i]
        b_i = min(b_i_values) if b_i_values else np.nan  # Handle case of no neighboring clusters
        
        silhouette_values.append((b_i - a_i) / max(a_i, b_i))
    
    return np.nanmean(silhouette_values)  # Use np.nanmean to handle NaN values gracefully

# Step 4: Find Optimal K
def find_optimal_k(data, silhouette_score_k3):
    max_k = 6
    best_k = 3
    best_silhouette_score = silhouette_score_k3
    
    labels = {}  # Initialize an empty dictionary to store labels
    for k in range(4, max_k + 1):
        k_means_clustering(data, k, 1)
        
        labels.clear()
        with open("kmeanstmp.txt", "r") as file:
            for count, line in enumerate(file):
                for ent in map(int, line.strip().split(',')):
                    labels[ent] = count
        
        file.close()

        silhouette_score = silhouette_coefficient(data, labels)
        
        print("\t", k, "\t", silhouette_score)
        log_file.write("\t" + str(k) + "\t" + str(silhouette_score) + "\n")
        
        if silhouette_score > best_silhouette_score:
            best_silhouette_score = silhouette_score
            best_k = k
            
    return best_k

# Step 5: Agglomerative Hierarchical Clustering (Bottom-Up) using Complete Linkage
def hierarchical_clustering(data, k):
    n = len(data)
    clusters = [[i] for i in range(n)]  # Initialize each data point as its own cluster
    distances = np.zeros((n, n))

    # Calculate pairwise distances between data points
    for i in range(n):
        for j in range(i+1, n):
            distances[i][j] = distances[j][i] = np.linalg.norm(data[i] - data[j])

    # Merge clusters until the desired number of clusters is reached
    while len(clusters) > k:
        min_distance = np.inf
        merge_indices = None

        # Find the closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_dist = max(distances[a][b] for a in clusters[i] for b in clusters[j])
                if max_dist < min_distance:
                    min_distance = max_dist
                    merge_indices = (i, j)

        # Merge the closest pair of clusters
        clusters[merge_indices[0]] += clusters[merge_indices[1]]
        del clusters[merge_indices[1]]

    return clusters

# Step 6: Jaccard Similarity
def jaccard_similarity(k_opt_means_clusters, hierarchical_clusters):
    # Compute Jaccard similarity between two clusters
    """
    Calculate the Jaccard similarity between two sets of clusters.

    Parameters:
    k_opt_means_clusters (list of lists): Set of clusters from clustering method 1.
    hierarchical_clusters (list of lists): Set of clusters from clustering method 2.

    Returns:
    list: Jaccard similarity scores for each mapping of clusters.
    """
    
    jaccard_scores = []
    for clusters1 in k_opt_means_clusters:
        max_similarity = 0
        for clusters2 in hierarchical_clusters:
            intersection = len(set(clusters1).intersection(clusters2))
            union = len(set(clusters1).union(clusters2))
            similarity = intersection / union if union > 0 else 0
            if similarity > max_similarity:
                max_similarity = similarity
        jaccard_scores.append(max_similarity)
        
    return jaccard_scores

# Main function
def main():
    # Step 1: Read the dataset
    start_time = time.time()
    
    data = read_dataset("instagram.csv")
    # data = read_dataset("sample.csv")
    
    end_time = time.time()
    
    print("\nRead the dataset in (", end_time - start_time, ") seconds")
    log_file.write("Read the dataset in (" + str(end_time - start_time) + ") seconds\n")
    
    # Step 2: K-means Clustering
    k = 3 # Initial value of K
    
    start_time = time.time()
    k_means_clustering(data, k)
    end_time = time.time()
    
    print("\nK-means Clustering in (", end_time - start_time, ") seconds")
    log_file.write("\nK-means Clustering in (" + str(end_time - start_time) + ") seconds\n")
    
    # Step 3: Silhouette Coefficient Calculation
    labels = {}  # Initialize an empty dictionary to store labels

    with open("kmeans.txt", "r") as file:
        for count, line in enumerate(file):
            for ent in map(int, line.strip().split(',')):
                labels[ent] = count
    
    silhouette_score = silhouette_coefficient(data, labels)

    print("\nSilhouette Coefficient Values :\n")
    print("\t k\t silhouette_score")
    log_file.write("\nSilhouette Coefficient Values :\n")
    log_file.write("\t k\t silhouette_score\n")
    
    print("\t", k, "\t", silhouette_score)
    log_file.write("\t" + str(k) + "\t" + str(silhouette_score) + "\n")
    
    # Step 4: Find Optimal K
    start_time = time.time()
    optimal_k = find_optimal_k(data, silhouette_score)
    end_time = time.time()
    
    print("\nFind Optimal K in (", end_time - start_time, ") seconds")
    print("\nOptimal K:", optimal_k)
    log_file.write("\nFind Optimal K in (" + str(end_time - start_time) + ") seconds\n")
    log_file.write("\nOptimal K: " + str(optimal_k) + "\n")
    
    # Step 5: K-means Clustering with optimal K
    start_time = time.time()
    k_means_clustering(data, optimal_k)
    end_time = time.time()
    
    print("\nK-means Clustering with optimal K in (", end_time - start_time, ") seconds")
    log_file.write("\nK-means Clustering with optimal K in (" + str(end_time - start_time) + ") seconds\n")
    
    k_opt_means_clusters = []
    with open("kmeans.txt", "r") as file:
        for _, line in enumerate(file):
            k_opt_means_clusters.append(list(map(int, line.strip().split(','))))
    
    # Step 6: Hierarchical Clustering
    start_time = time.time()
    hierarchical_clusters = hierarchical_clustering(data, optimal_k)
    end_time = time.time()
    
    print("\nHierarchical Clustering in (", end_time - start_time, ") seconds")
    log_file.write("\nHierarchical Clustering in (" + str(end_time - start_time) + ") seconds\n")
    
    for cluster in hierarchical_clusters:
        cluster.sort()
        
    hierarchical_clusters.sort(key=lambda x: x[0])
        
    with open("agglomerative.txt", "w") as file:
        for cluster in hierarchical_clusters:
            file.write(",".join(map(str, cluster)))
            file.write("\n")
    
    file.close()
    
    # print("\nk_opt_means_clusters :", k_opt_means_clusters)
    # print("\nhierarchical_clusters :", hierarchical_clusters)
    
    # Step 7: Jaccard Similarity
    start_time = time.time()
    jaccard_jaccard_scores = jaccard_similarity(k_opt_means_clusters, hierarchical_clusters)
    end_time = time.time()
    
    print("\nJaccard Similarity in (", end_time - start_time, ") seconds")
    log_file.write("\nJaccard Similarity in (" + str(end_time - start_time) + ") seconds\n")
    
    print("\nJaccard similarity values :\n")
    print("\t mapping\t jaccard_score")
    log_file.write("\nJaccard similarity values :\n")
    log_file.write("\t mapping\t jaccard_score\n")
    
    # Print Jaccard similarity scores for each mapping
    for i, score in enumerate(jaccard_jaccard_scores):
        print("\t", i, "\t\t", score)
        log_file.write("\t" + str(i) + "\t\t" + str(score) + "\n")
        
    # Print the best Jaccard similarity score
    best_jaccard_score = max(jaccard_jaccard_scores)
    best_jaccard_score_index = jaccard_jaccard_scores.index(best_jaccard_score)
    
    print("\nBest Jaccard similarity score:", best_jaccard_score, "for mapping", best_jaccard_score_index)
    log_file.write("\nBest Jaccard similarity score: " + str(best_jaccard_score) + " for mapping " + str(best_jaccard_score_index) + "\n")
    
    # Delete the file named kmeanstmp.txt
    file_path = "kmeanstmp.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        
    log_file.close()

# Entry point of the program
if __name__ == "__main__":
    main()
