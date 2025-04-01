# Final Model Parameters for OCR

from ultralytics import YOLO
import ocr
import os
import glob
import csv
import numpy as np
from PIL import Image
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score
from statistics import mean

from control import method5_1, method5_2, method5_3, method5_4, method4

def readOutputFile(filepath):
    file = open(filepath, 'r')
    content = file.read()
    file.close()

    lines = content.split('\n')
    clusters = []

    for line in lines:
        clusters.append(line.split(','))
    
    return clusters


def getRandIndex(dataset_path, output_filepath):

    gt = readOutputFile(dataset_path + '/gt.txt')

    pred = readOutputFile(output_filepath)

    if pred[len(pred)-1][0] == '':
        pred.pop(len(pred)-1)

    # Calculate Rand index

    all_diagram_names = list(itertools.chain.from_iterable(gt))

    gt_dict = {x: -1 for x in all_diagram_names}
    pred_dict = {x: -1 for x in all_diagram_names}

    # gt processing
    for i in range(len(gt)):
        for diagram in gt[i]:
            gt_dict[diagram] = i

    # pred processing
    for i in range(len(pred)):
        for diagram in pred[i]:
            pred_dict[diagram] = i
    
    return rand_score(list(gt_dict.values()), list(pred_dict.values()))

def create_plot(data, name):

        # Create a bar plot for
        plt.bar(range(len(datasets)), data)

        # Adding title and labels
        plt.title(f"{name} Rand Index Results")
        plt.xlabel('Datasets')
        plt.ylabel('Rand Index')

        custom_ticks = np.arange(1, len(datasets) + 1)
        custom_labels = list(range(1,len(datasets) + 1))

        # Save the Plot
        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        file_path = os.path.join('./plots', f'{name}.png')
        plt.savefig(file_path)


if __name__ == "__main__":

    datasets_parent_path = './datasets_mod/'

    datasets = ['airlines_dataset01_N_11_K_4', 'cars_Dataset08_N_10_K_5', 'cars_Dataset09_N_10_K_4', 'dataset1_K_4', 'dataset2_K_3',
                'dataset3_K_3','TVSeries_dataset03_N_10_K_3','videoGames_dataset_N_10_K_4', 'videogames_dataset05_N_5_K_3']

    dataset_path = datasets_parent_path + 'handdrawn'
    print(dataset_path)
    method5_4(dataset_path, dataset_path, '5_4', 2)

    '''
    # mod5_1_results
    mod_4_results = []
    mod5_1_results = []
    mod5_2_results = []
    mod5_3_results = []
    mod5_4_results = []

    for dataset in datasets:

        # Get Rand index for method 5_1
        dataset_path = datasets_parent_path + dataset
        method4(dataset_path, dataset_path, '4', int(dataset[-1]))
        method5_1(dataset_path, dataset_path, '5_1',  int(dataset[-1]))
        method5_2(dataset_path, dataset_path, '5_2', int(dataset[-1]))
        method5_3(dataset_path, dataset_path, '5_3', int(dataset[-1]))
        method5_4(dataset_path, dataset_path, '5_4', int(dataset[-1]))

        rand_val_5_1 = getRandIndex(dataset_path, f"{dataset_path}/5_1.txt")
        rand_val_5_2 = getRandIndex(dataset_path, f"{dataset_path}/5_2.txt")
        rand_val_5_3 = getRandIndex(dataset_path, f"{dataset_path}/5_3.txt")
        rand_val_5_4 = getRandIndex(dataset_path, f"{dataset_path}/5_4.txt")
        rand_val_4 = getRandIndex(dataset_path, f"{dataset_path}/4.txt")

        mod5_1_results.append(rand_val_5_1)
        mod5_2_results.append(rand_val_5_2)
        mod5_3_results.append(rand_val_5_3)
        mod5_4_results.append(rand_val_5_4)
        mod_4_results.append(rand_val_4)
    
    create_plot(mod_4_results, 'mod_4')
    create_plot(mod5_1_results, 'mod_5_1')
    create_plot(mod5_2_results, 'mod_5_2')
    create_plot(mod5_3_results, 'mod_5_3')
    create_plot(mod5_4_results, 'mod_5_4')

    print("Avg Results")
    print(mean(mod_4_results))
    print(mean(mod5_1_results))
    print(mean(mod5_2_results))
    print(mean(mod5_3_results))
    print(mean(mod5_4_results))
    '''