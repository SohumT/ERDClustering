import pandas as pd
import numpy as np
import shutil
import os
from ultralytics import YOLO

import mod4_clustering
import mod5_clustering
import main
import sys

import sklearn.metrics.pairwise as metrics

model = YOLO('./kyobest.pt')

def method4(dataset_path, output_path, output_filename,  cluster_number):

    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
        main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod4_clustering.getTfIdfMatrix(dataset_path + '/OD_OCR_Output')

    mod4_clustering.getClusters(corpus_tfidf_dense, text_list, output_path, output_filename, cluster_number)
    
def method5_1(dataset_path, output_path, output_filename,  cluster_number):


    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
        main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod5_clustering.getTfIdfMatrix(dataset_path + '/OD_OCR_Output')

    tfidf_matrix_w_erd_counts = mod5_clustering.addObjectNumberstoVector(corpus_tfidf_dense, dataset_path + '/OD_OCR_Output')

    mod5_clustering.getKMeansClusters(tfidf_matrix_w_erd_counts, text_list, output_path, output_filename, cluster_number)

def method5_2(dataset_path, output_path, output_filename,  cluster_number):

    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
        main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod5_clustering.getTfIdfMatrix(dataset_path + '/OD_OCR_Output')

    tfidf_matrix_w_erd_counts = mod5_clustering.addObjectNumberstoVector(corpus_tfidf_dense, dataset_path + '/OD_OCR_Output')

    mod5_clustering.mixedClustering(tfidf_matrix_w_erd_counts, text_list, output_path, output_filename, cluster_number)


def method5_3(dataset_path, output_path, output_filename,  cluster_number):

    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
        main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod5_clustering.prefixInputVectorModification(dataset_path + '/OD_OCR_Output')

    mod5_clustering.getKMeansClusters(corpus_tfidf_dense, text_list, output_path, output_filename, cluster_number)

def method5_4(dataset_path, output_path, output_filename,  cluster_number):

    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    print(OD_OCR_Output_save_path)

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
        main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod5_clustering.prefixInputVectorModification(dataset_path + '/OD_OCR_Output')
    print(corpus_tfidf_dense)

    tfidf_matrix_w_erd_counts = mod5_clustering.addObjectNumberstoVector(corpus_tfidf_dense, dataset_path + '/OD_OCR_Output')

    mod5_clustering.getKMeansClusters(tfidf_matrix_w_erd_counts, text_list, output_path, output_filename, cluster_number)


def find_closest_pairs(dataset_path, output_path):

    OD_OCR_Output_save_path = dataset_path + '/OD_OCR_Output'

    if not os.path.exists(OD_OCR_Output_save_path):
        os.makedirs(OD_OCR_Output_save_path)
    
    main.getOutput2(dataset_path, OD_OCR_Output_save_path, model)

    corpus_tfidf_dense, text_list = mod5_clustering.getTfIdfMatrix(dataset_path + '/OD_OCR_Output')

    tfidf_matrix_w_erd_counts = mod5_clustering.addObjectNumberstoVector(corpus_tfidf_dense, dataset_path + '/OD_OCR_Output')

    distances_matrix = metrics.pairwise_distances(tfidf_matrix_w_erd_counts.transpose(), tfidf_matrix_w_erd_counts.transpose(), metric='cosine')

    # Exclude distances to self by setting diagonal elements to a large value
    np.fill_diagonal(distances_matrix, np.inf)

    # Display results
    distance_pairs_df = pd.DataFrame(distances_matrix, columns=tfidf_matrix_w_erd_counts.columns, index=tfidf_matrix_w_erd_counts.columns)
    distance_pairs_df['closest document'] = distance_pairs_df.idxmin(axis=1)

    print(distance_pairs_df['closest document'])

    

if __name__ == "__main__":

    #print("Receive input here and apply respective method")

    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    output_filename = sys.argv[3]
    cluster_number = int(sys.argv[4])
    method = sys.argv[5]

    if method == 'mod_4':
        method4(dataset_path, output_path, output_filename,  cluster_number)
    elif method == 'mod5_1':
        method5_1(dataset_path, output_path, output_filename, cluster_number)
    elif method == 'mod5_2':
        method5_2(dataset_path, output_path, output_filename, cluster_number)
    elif method == 'closest_pairs':
        find_closest_pairs(dataset_path, output_path)
