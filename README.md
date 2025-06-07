# ERD Clustering

A system for clustering similar Entity-Relationship Diagrams (ERDs) using OCR and advanced clustering techniques. This system was created as part of research study exploring utilization of clustering as a means to simplify grading processes for ERDs in database cousrs. The IEEE paper can be found [here](https://ieeexplore.ieee.org/document/10893428/authors#authors).

## Overview

This repository contains a comprehensive solution for processing, analyzing, and clustering Entity-Relationship Diagrams. The system uses a combination of object detection, optical character recognition (OCR), and various clustering algorithms to group similar ERD diagrams based on their content and structure.

The workflow consists of:
1. Object detection using YOLO to identify ERD elements (entities, relationships, attributes)
2. OCR processing to extract text from the detected elements
3. Text processing and normalization
4. Feature extraction using TF-IDF and other techniques
5. Clustering using various algorithms
6. Evaluation of clustering results

## Key Components

### Object Detection and OCR
- `control.py`: This is the main entry point for the application and is used in CLI commands to perform evaluation on input datasets and provide outputs at a specified path
- `main.py`: Utilized to train the Yolov8 model and export object detection / OCR output to a specified path
- `ocr.py`: Implements OCR functionality using EasyOCR to extract text from detected elements
- `edit_distance.py`: Post-processes OCR results, correcting misclassified words using edit distance calculations

### Clustering Methods

The repository implements several clustering approaches:

- **Method 4** (`mod4_clustering.py`): Basic TF-IDF with K-means clustering
- **Method 5.1**: TF-IDF with ERD object counts and K-means clustering
- **Method 5.2**: TF-IDF with ERD object counts and mixed clustering (K-means or GMM based on intercluster variation)
- **Method 5.3**: TF-IDF with prefix input vector modification and K-means clustering
- **Method 5.4**: TF-IDF with prefix input vector modification, ERD object counts, and K-means clustering

All advanced methods are implemented in `mod5_clustering.py`.

### Control and Evaluation

- `control.py`: Allows users to call specific clustering method on a specified dataset path using CLI commands
- `eval.py`: Evaluates clustering results using the Rand Index metric

## Models

- The repository uses a pre-trained YOLO model (`kyobest.pt`) for object detection.
- The model can be downloaded from hugging face [here](https://huggingface.co/ssthadan/ERDClassificationDetection/tree/main).

## Requirements

The project requires several Python packages, including:
- ultralytics (YOLO)
- easyocr
- numpy
- opencv-python
- editdistance
- nltk
- gensim
- scikit-learn
- pandas
- matplotlib

See `requirements.txt` for the complete list of dependencies.

You can install requirements using `pip install -r /path/to/requirements.txt`


## Usage

### Basic Usage

```bash
# Run method 4 clustering
python control.py <dataset_path> <output_path> <output_filename> <cluster_number> mod_4

# Run method 5.1 clustering
python control.py <dataset_path> <output_path> <output_filename> <cluster_number> mod5_1

# Run method 5.2 clustering
python control.py <dataset_path> <output_path> <output_filename> <cluster_number> mod5_2

# Find closest pairs
python control.py <dataset_path> <output_path> <output_filename> <cluster_number> closest_pairs
```

### Evaluation

```bash
python eval.py
```

## Docker Support

The repository includes a Dockerfile for containerization:

```bash
# Build the Docker image
docker build -t erd-clustering .

# Run the container
docker run -v <local_path>:<container_path> erd-clustering python control.py <args>
```

Performance-wise, the clustering tool can process a dataset of 150 diagrams in about 15 to 20 minutes when using a shared university cluster powered by an Nvidia Tesla V100 GPU. The same task takes about 90 minutes on a laptop with an M1 chip.

## Methodology

### OCR Processing

1. Object detection identifies ERD elements in the diagram
2. Each detected element is processed with OCR to extract text
3. Post-processing corrects misclassified words using edit distance and context from the question text

### Clustering Approaches

1. **Basic Clustering (Method 4)**:
   - Extract text from OCR results
   - Create TF-IDF vectors
   - Apply K-means clustering

2. **Advanced Clustering (Method 5)**:
   - Enhance feature vectors with ERD object counts
   - Apply prefix modifications to differentiate between entity attributes and relationship attributes
   - Use different clustering algorithms (K-means, GMM, Spectral Clustering)
   - Implement mixed clustering based on intercluster variation

### Evaluation

Clustering results are evaluated using the Rand Index, which measures the similarity between the predicted clusters and ground truth clusters.

## Project Structure for Set-up

- Please note that using  `eval.py` will require you to make path modifications to the code 
based on where your datasets are located. Some examples have been left in the code comments.

```
.
├── control.py              # Control methods for different clustering approaches
├── DockerFile              # Docker configuration
├── edit_distance.py        # Post-processing OCR results
├── eval.py                 # Evaluation of clustering results
├── kyobest.pt              # Pre-trained YOLO model
├── main.py                 # Main entry point
├── mod4_clustering.py      # Basic clustering implementation
├── mod5_clustering.py      # Advanced clustering implementations
├── ocr.py                  # OCR functionality
├── README.md               # This file
└── requirements.txt        # Project dependencies
