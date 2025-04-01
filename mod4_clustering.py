import csv
import os
import re
import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from gensim.matutils import corpus2dense, corpus2csc
from ultralytics import YOLO

import pandas as pd
import shutil
import os

import main


#nltk.download('punkt')
#ps = PorterStemmer()

def getTfIdfMatrix(path_to_dataset_output):

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
                    file_names.append(filename)
                    content = re.findall(r'\b\w+\b', file.read())
                    content = [word for word in content if word not in ['entity', 'weak_entity','rel', 'ident_rel', 'rel_attr']]
                    text_list.append(content)

    ''' Print the list of strings
    for d in range(len(text_list)):
        text_list[d] = [ps.stem(w) for w in text_list[d]]
    print(text_list[3]) '''

    text_dictionary = corpora.Dictionary(text_list)
    num_docs = text_dictionary.num_docs
    num_terms = len(text_dictionary.keys())
    text_bow = [text_dictionary.doc2bow(doc, allow_update=True) for doc in text_list]

    tfidf = gensim.models.TfidfModel(text_bow, smartirs='ltc')
    tfidf_corpus = tfidf[text_bow]

    corpus_tfidf_dense = corpus2dense(tfidf_corpus, num_terms, num_docs)
    # corpus_tfidf_sparse = corpus2csc(tfidf_corpus, num_terms, num_docs)

    tfidf_df = pd.DataFrame(corpus_tfidf_dense, columns=file_names)

    return tfidf_df, text_list


def getClusters(corpus_tfidf_dense, text_list, output_path, output_filename, number_of_clusters):

    model = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=10, random_state=1234, max_iter=50, algorithm='lloyd')
    
    clusters = model.fit_predict(corpus_tfidf_dense.transpose())
    # Create a new DataFrame with the file names and corresponding cluster labels
    cluster_results = pd.DataFrame({'Document': corpus_tfidf_dense.columns, 'Cluster': clusters})

    with open(f"{output_path}/{output_filename}.txt", 'w') as file:

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
            
            #print(cluster_num)'''


if __name__ == "__main__":

    datasets_for_eval = ['./ds1_N_8_K3', './ds2_N_6_K4', './ds3_N_5_K3', './ds4_N_6_K4', './ds5_N_5_K3', './ds6_N_7_K3']

    model = YOLO('./kyobest.pt')

    # Use for OCR and then comment out when not in use 

    '''for index, dataset in enumerate(datasets_for_eval):

        OD_OCR_Output_save_path = dataset + '/OD_OCR_Output'
        
        if not os.path.exists(OD_OCR_Output_save_path):
            os.makedirs(OD_OCR_Output_save_path)
        
        main.getOutput(dataset, OD_OCR_Output_save_path, model)'''

    
    for index, dataset in enumerate(datasets_for_eval):
    
        corpus_tfidf_dense, text_list = getTfIdfMatrix(dataset + '/OD_OCR_Output')

        getClusters(corpus_tfidf_dense, text_list, './base_line_cluster', index + 1, int(dataset[-1]))