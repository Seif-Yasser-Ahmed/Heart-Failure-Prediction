# Heart-Failure-Prediction
This project explores clinical data, employs PCA for visualization, and develops classifiers like Naïve Bayes, SVM, KNN, and Decision Trees to predict heart failure risks. Includes dendrogram analysis for clustering insights and evaluates models using precision, recall, and F1-score.
# Heart Failure Prediction

This project aims to predict heart failure risk using machine learning models and the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). The analysis includes data preprocessing, visualization, classifier training, testing, and dendrogram analysis for clustering insights.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Project Workflow](#project-workflow)  
3. [Dataset](#dataset)  
4. [Files and Directory Structure](#files-and-directory-structure)  
5. [Setup and Installation](#setup-and-installation)  
6. [Usage](#usage)  
7. [Results](#results)  
8. [Acknowledgments](#acknowledgments)

---

## Introduction

Heart failure is one of the leading causes of death globally. Early detection and accurate risk assessment can significantly improve patient outcomes. This project explores the dataset, performs preprocessing, and evaluates machine learning classifiers, including:

- *Naïve Bayes*  
- *Support Vector Machines (SVM)*  
- *K-Nearest Neighbors (KNN)*  
- *Decision Trees*

Additionally, the project generates a *dendrogram* to analyze hierarchical clustering.

---

## Project Workflow

The project is divided into the following stages:

1. *Data Exploration and Visualization*  
   - Dimensionality reduction to 2D using PCA.
   - Visualizing and analyzing patterns.  

2. *Data Cleaning and Processing*  
   - Handling missing values, duplicates, and outliers.  
   - Splitting the data into training, validation, and testing sets.  

3. *Model Training*  
   - Training classifiers (Naïve Bayes, SVM, KNN, Decision Trees).  
   - Hyperparameter tuning via grid search and manual methods.  

4. *Model Evaluation*  
   - Calculating precision, recall, F1-score, and confusion matrix.  

5. *Dendrogram Analysis*  
   - Hierarchical clustering to identify natural groupings.  

---

## Dataset

The dataset used is from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).  
It includes features such as age, cholesterol levels, resting blood pressure, and maximum heart rate, with a target variable indicating heart disease presence.

---

## Files and Directory Structure

```plaintext
Heart-Failure-Prediction/
│
├── Dataset/                               # Dataset files
│   └── heart.csv
│
├── Logs/                          
│   ├───Losses
│   ├───Plots
│   └───Reports
│       ├───DT
│       ├───GaussBayes
│       ├───KNN
│       └───SVM
├── Models/                          
│   ├───DT
│   ├───GaussBayes
│   ├───KNN
│   └───SVM
│
├── Src/                            
│   ├── dendrogram.png
│   ├── mlartifacts/
│   ├── mlruns/
│   ├── Notebooks/
│   │   ├── naive_bayes_report.txt
│   │   ├── svm_report.txt
│   │   ├── knn_report.txt
│   │   └── decision_tree_report.txt
│   │
│   ├── Utils/
│   │   ├── mlflow.py
│   │   ├── Preprocessor.py
│   │   └── Visualizer.py
│   │
│   ├── config.py
│   └── Main.ipynb
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── SECURITY.md                         # Project Security
└── LICENSE                             # License file
```

## Setup and Installation
Clone the repository
```bash
git clone https://github.com/Seif-Yasser-Ahmed/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
```
Install dependencies
```python
pip install -r requirements.txt
```
Ensure the dataset is placed in the `Dataset/` directory.

## Usage
1. Open the Jupyter notebooks in the `Src/` directory to explore, process, and analyze the dataset
2. Run the cells in the notebooks sequentially to reproduce results.

## Results

## Acknowledgments
