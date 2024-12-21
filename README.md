# Heart-Failure-Prediction
This project explores clinical data, employs PCA for visualization, and develops classifiers like Logistic Regression, Naïve Bayes, SVM, KNN, and Decision Trees to predict heart failure risks. It includes dendrogram analysis for clustering insights and evaluates models using precision, recall, and F1-score, aiming to predict heart failure risk using machine learning models and the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). Additionally, MLflow was used to track the workflow and monitor model performance.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Project Workflow](#project-workflow)  
3. [Dataset](#dataset)  
4. [Pipeline](#pipeline)  
5. [Files and Directory Structure](#files-and-directory-structure)  
6. [Setup and Installation](#setup-and-installation)  
7. [Usage](#usage)  
8. [Results](#results)  

---

## Introduction

Heart failure is one of the leading causes of death globally. Early detection and accurate risk assessment can significantly improve patient outcomes. This project explores the dataset, performs preprocessing, and evaluates machine learning classifiers, including:

- *Logistic Regression*  
- *Naïve Bayes*  
- *Support Vector Machines (SVM)*  
- *K-Nearest Neighbors (KNN)*  
- *Decision Trees*

Additionally, the project generates a *dendrogram* to analyze hierarchical clustering and uses MLflow to streamline and monitor the model development workflow.

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
   - Training classifiers (Logistic Regression, Naïve Bayes, SVM, KNN, Decision Trees).  
   - Hyperparameter tuning via grid search and manual methods.  

4. *Model Evaluation*  
   - Calculating precision, recall, F1-score, and confusion matrix.  

5. *Dendrogram Analysis*  
   - Hierarchical clustering to identify natural groupings.  

6. *Workflow Tracking*  
   - Using MLflow to track experiments and compare model metrics.

---

## Dataset

The dataset used is from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).  
It includes features such as age, cholesterol levels, resting blood pressure, and maximum heart rate, with a target variable indicating heart disease presence.

---

## Pipeline
![image](https://github.com/user-attachments/assets/99b7fc05-6315-4663-ab03-da0e7372bab0)

The following pipeline was employed to achieve the objectives of the project:

1. **Data Loading**: Load the dataset from the `Dataset/` directory.
2. **Preprocessing**: Handle missing data, encode categorical variables, and scale numerical features.
3. **Dimensionality Reduction**: Perform PCA for visualization and feature reduction.
4. **Model Training**: Train classifiers (Logistic Regression, Naïve Bayes, SVM, KNN, Decision Trees) using grid search for hyperparameter optimization.
5. **Evaluation**: Evaluate the models using metrics like precision, recall, F1-score, and confusion matrix.
6. **Clustering Analysis**: Generate a dendrogram to explore clustering insights.
7. **Tracking Workflow**: Use MLflow to log model parameters, metrics, and artifacts.

---

## Files and Directory Structure

```plaintext
Heart-Failure-Prediction/
│
├── Dataset/                               
│   └── heart.csv
│
├── Logs/  
│   ├── Losses
│   ├── Plots
│   └── Reports
│       ├── DT
│       ├── GaussBayes
│       ├── KNN
│       └── SVM
│       └── EDA
│       └── Logistic
│
├── Models/  
│   ├── DT
│   ├── GaussBayes
│   ├── KNN
│   └── SVM
│   └── Logistic
│
├── Src/  
│   ├── figures/
│   ├── mlartifacts/
│   ├── mlruns/
│   ├── Notebooks/
│   │   ├── DT.ipynb
│   │   └── Naive Bayes.ipynb
│   └── Utils/
│       ├── mlflow.py
│       ├── Preprocessor.py
│       └── Visualizer.py
│
│   ├── config.py
│   ├── Logger.py
│   └── Main.ipynb
│
├── requirements.txt                    
├── README.md                           
├── SECURITY.md                         
└── LICENSE                                             
```

---

## Setup and Installation

Clone the repository
```bash
git clone https://github.com/Seif-Yasser-Ahmed/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
```
Install dependencies
```bash
pip install -r requirements.txt
```
Ensure the dataset is placed in the `Dataset/` directory.
   
---

## Usage

1. Open the Jupyter notebooks in the `Src/` directory to explore, process, and analyze the dataset.
2. Run the cells in the notebooks sequentially to reproduce results.

---

## Results
The results of the classifiers are summarized as follows:

| Model (Best Model)  | Accuracy (%) |Precision (C0-C1) | Recall(C0-C1) | F1-Score(C0-C1) |
|---------------------|--------------|------------------|---------------|-----------------|
| Logistic Regression |    85        |0.8  - 0.9         | 0.84 - 0.88   | 0.83 - 0.87     |
| Naïve Bayes         |    86        |0.81 -  0.9       | 0.87 - 0.85   | 0.84 -  0.88    |
| SVM                 |    85        |0.82 - 0.93       | 0.92 - 0.85   | 0.87 - 0.89     |
| KNN                 |    89          |0.86 -   0.92  | 0.89 -   0.89 | 0.87 -   0.9       |
| Decision Trees      |    86          |0.83 -   0.88  | 0.84 -  0.87 | 0.83 -  0.88        |
### Viewing Results
You can view the results using one of the following options:
1. **MLflow Tracking UI:** <br>
   Navigate to the `Src/` directory and start the MLflow UI:
   ```bash
   cd Src
   mlflow ui
   ```
   Open the provided local URL (e.g., [http://127.0.0.1:5000](http://127.0.0.1:5000)) in your browser to explore the experiment results.
2. **Streamlit Dashboard:** <br>
   Navigate to the `Src/` directory and run the Streamlit app:
   ```bash
   cd Src
   streamlit run Logger.py

   ```




