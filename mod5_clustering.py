
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from gensim.matutils import corpus2dense, corpus2csc
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import gensim

from sklearn.cluster import OPTICS

import os

from mod4_clustering import getTfIdfMatrix, getClusters


filename_content = {}

def prefixInputVectorModification(path_to_dataset_output):

    ps = PorterStemmer()
    folder_path = path_to_dataset_output
    text_list = []
    file_names = []
    
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the file is a text file
            if filename.endswith(".txt"):
                # Read the content of the file and append it to the list
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_lines = file.readlines()
                    complete_content = []
                    for line in all_lines:
                        content = gensim.utils.simple_preprocess(line)
                        label = content[0]
                        
                        if label == 'weak_entity':
                            label = 'entity'
                        
                        elif label == 'rel' or label == 'ident_rel':
                            label = ''

                        content = content[1:]
                        content = [ps.stem(w) for w in content]
                        content = [label+"_"+w for w in content]    
                        complete_content += content
                    filename_content[filename[:3]] = complete_content
                    text_list.append(complete_content)
                    file_names.append(filename)

    text_dictionary = corpora.Dictionary(text_list)
    num_docs = text_dictionary.num_docs
    num_terms = len(text_dictionary.keys())
    text_bow = [text_dictionary.doc2bow(doc, allow_update=True) for doc in text_list]

    tfidf = gensim.models.TfidfModel(text_bow, smartirs='ltc')
    tfidf_corpus = tfidf[text_bow]

    corpus_tfidf_dense = corpus2dense(tfidf_corpus, num_terms, num_docs)
    tfidf_df = pd.DataFrame(corpus_tfidf_dense, columns=file_names)
    print(file_names)
    # corpus_tfidf_sparse = corpus2csc(tfidf_corpus, num_terms, num_docs)

    # tfidf_df = pd.DataFrame(corpus_tfidf_dense, columns=file_names)

    return tfidf_df, text_list

def addObjectNumberstoVector(corpus_tfidf_dense, path_to_dataset_output):
    
    ps = PorterStemmer()
    folder_path = path_to_dataset_output
    text_list = []
    file_names = []

    ERD_Object_Counts_pd = pd.DataFrame(columns=corpus_tfidf_dense.columns)
    
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the file is a text file
            if filename.endswith(".txt"):
                # Read the content of the file and append it to the list
                erd_object_count = {'entity':0, 'rel_attr':0, 'rel':0, 'weak_entity':0, 'ident_rel':0}

                with open(file_path, 'r', encoding='utf-8') as file:
                    all_lines = file.readlines()
                    complete_content = []
                    for line in all_lines:
                        content = gensim.utils.simple_preprocess(line)
                        label = content[0]

                        # Increment objects in erd_object dict
                        # if label != 'weak_entity' and label != 'ident_rel':
                        erd_object_count[label] += 1

                        content = content[1:]
                        content = [ps.stem(w) for w in content]
                        content = [label+"_"+w for w in content]    
                        complete_content += content

                    filename_content[filename[:3]] = complete_content
                    text_list.append(complete_content)
                    file_names.append(filename)

                    # Add column to pandas dataframe
                    ERD_Object_Counts_pd[filename] = erd_object_count.values()
    
    # Add the erd_object dataframe to the original dataframe
    return pd.concat([corpus_tfidf_dense, ERD_Object_Counts_pd], ignore_index=True, sort=False)

def getSpectralClusters(corpus_tfidf_dense, text_list, output_path, dataset_number, number_of_clusters):


    # Transpose the dataframe, so we can do pre-processing 
    X = corpus_tfidf_dense.transpose()
    
    '''# Scaling the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalizing the Data
    X_normalized = normalize(X_scaled)
    
    # Reducing the dimensions of the data
    pca = PCA(n_components = min(5,X_normalized.shape[0])
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)'''

    # Building the clustering model
    spectral_model_rbf= SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=number_of_clusters, n_components=None,
                   n_init=10, n_jobs=None, n_neighbors=3, random_state=None)
 
    # Training the model and Storing the predicted cluster labels
    labels_rbf = spectral_model_rbf.fit_predict(X)

    with open(f"{output_path}/advanced_clusters_{dataset_number}.txt", 'w') as file:

        # Print the clusters using the DataFrame
        for cluster_num in range(number_of_clusters):
            
            docs_belonging_cluster = []

            for i in range(len(labels_rbf)):
                if labels_rbf[i] == cluster_num:
                    docs_belonging_cluster.append(X.index.to_list()[i])
            
            #print(f"\nCluster {cluster_num + 1}:\n")
            docs_in_cluster_str = ""

            for document in docs_belonging_cluster:
                docs_in_cluster_str += document.split(".")[0]
                docs_in_cluster_str += ","

            docs_in_cluster_str = docs_in_cluster_str[:-1]
            
            file.write(f"{docs_in_cluster_str}\n")

def getGMMClusters(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters):
    
    gmm = GaussianMixture(n_components=number_of_clusters, covariance_type='full', random_state=42, max_iter=150).fit(corpus_tfidf_dense.transpose())
    labels = gmm.predict(corpus_tfidf_dense.transpose())

    with open(f"{output_path}/{output_filename}.txt", 'w') as file:

        # Print the clusters using the DataFrame
        for cluster_num in range(number_of_clusters):
            
            docs_belonging_cluster = []

            for i in range(len(labels)):
                if labels[i] == cluster_num:
                    docs_belonging_cluster.append(corpus_tfidf_dense.columns.to_list()[i])
            
            #print(f"\nCluster {cluster_num + 1}:\n")
            docs_in_cluster_str = ""

            for document in docs_belonging_cluster:
                docs_in_cluster_str += document.split(".")[0]
                docs_in_cluster_str += ","

            docs_in_cluster_str = docs_in_cluster_str[:-1]
            
            file.write(f"{docs_in_cluster_str}\n")

def getKMeansClusters(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters):

    model = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=10, random_state=1234, max_iter=50, algorithm='lloyd')
    
    clusters = model.fit_predict(corpus_tfidf_dense.transpose())
    # Create a new DataFrame with the file names and corresponding cluster labels
    cluster_results = pd.DataFrame({'Document': corpus_tfidf_dense.columns, 'Cluster': clusters})

    with open(f"{output_path}/{output_filename}.txt", 'w') as file:

        print(f"{output_path}/{output_filename}.txt")

        # Print the clusters using the DataFrame
        for cluster_num in range(number_of_clusters):
            cluster_documents = cluster_results[cluster_results['Cluster'] == cluster_num]['Document'].tolist()
            
            #print(f"\nCluster {cluster_num + 1}:\n")
            docs_in_cluster_str = ""

            for document in cluster_documents:
                docs_in_cluster_str += document.split(".")[0]
                docs_in_cluster_str += ","

            docs_in_cluster_str = docs_in_cluster_str[:-1]
            
            file.write(f"{docs_in_cluster_str}\n")
    
    # Calculate intercluster variation
    distances = pairwise_distances_argmin_min(corpus_tfidf_dense.transpose(), model.cluster_centers_)[1]

    #print(corpus_tfidf_dense.columns, clusters, distances)

    doc_distances_df = cluster_results
    doc_distances_df['distance'] = distances
    #print(doc_distances_df)

    var_df = pd.DataFrame()
    var_df['cluster_variation'] = doc_distances_df.groupby('Cluster')['distance'].var(ddof=1)
    
    #Debug Code 
    '''
    grouped = doc_distances_df.groupby('Cluster')
    
    for name, group in grouped:
        print(f"Group: {name}")
        print(name, group)
        print()
    '''

    return var_df['cluster_variation'].mean()

def getOpticsClusters(corpus_tfidf_dense, text_list, output_path, dataset_number, number_of_clusters):
   
    clustering = OPTICS(min_samples=2)
    labels = clustering.fit(corpus_tfidf_dense.transpose())

    print(labels.labels_)

    cluster_dict = {}

    for i in range(len(labels.labels_)):

        if labels.labels_[i] not in cluster_dict:
            cluster_dict[labels.labels_[i]] = []
        # print(f'cluster_dict[labels.labels_[i]]: {cluster_dict[labels.labels_[i]]}')
        cluster_dict[labels.labels_[i]].append(corpus_tfidf_dense.columns[i])
    
    clusters = cluster_dict.values()

    with open(f"{output_path}/advanced_clusters_{dataset_number}.txt", 'w') as file:

        # Print the clusters using the DataFrame
        for cluster in clusters:
            
            #print(f"\nCluster {cluster_num + 1}:\n")
            docs_in_cluster_str = ""

            for document in cluster:
                docs_in_cluster_str += document.split(".")[0]
                docs_in_cluster_str += ","

            docs_in_cluster_str = docs_in_cluster_str[:-1]
            
            file.write(f"{docs_in_cluster_str}\n")


def mixedClustering(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters):

    var = getKMeansClusters(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters)

    print(output_path, var)

    if var > 0.2:

        getGMMClusters(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters)


if __name__ == "__main__":

    datasets_for_eval = ['./ds1_N_8_K3']

    for index, dataset in enumerate(datasets_for_eval):
    
        #corpus_tfidf_dense, text_list = getTfIdfMatrix(dataset + '/OD_OCR_Output')

        corpus_tfidf_dense, text_list = prefixInputVectorModification(dataset + '/OD_OCR_Output')

        tfidf_matrix_w_erd_counts = addObjectNumberstoVector(corpus_tfidf_dense, dataset + '/OD_OCR_Output')

        var = getKMeansClusters(tfidf_matrix_w_erd_counts, text_list, './advanced_clusters', index + 1, int(dataset[-1]))

        print("Intercluster Variation Average of Dataset")
        print(var)
