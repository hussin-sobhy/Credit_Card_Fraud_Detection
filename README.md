# Credit Card Fraud Detection

<!-- Stack Badges -->
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/) 
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/) 
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/) 
[![scikit‑learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) 
[![imbalanced‑learn](https://img.shields.io/badge/imbalanced--learn-094D41?logo=none&logoColor=white)](https://imbalanced-learn.org/) 
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)


This repository presents a comprehensive machine learning project aimed at detecting fraudulent credit card transactions. Leveraging the Kaggle Credit Card Fraud Detection dataset, the project explores various strategies to handle class imbalance and improve model performance through detailed experiments.

## Project Overview

Credit card fraud detection is crucial for minimizing financial losses in the banking sector. This project undertakes the following steps:

- **Exploratory Data Analysis (EDA):**  
  A thorough exploration of the dataset is conducted to understand its characteristics, distribution, and the extent of class imbalance.

- **Data Preprocessing:**  
  Involves data cleaning, normalization, and feature engineering, setting the stage for robust model training.

- **Modeling Approaches:**  
  Three different approaches are implemented to address class imbalance:
  
  1. **Under Sampling:**  
     - Reduces the majority class (legitimate transactions) to balance the dataset.
     - Models Evaluated: Logistic Regression and Random Forest Classifier.
  
  2. **SMOTE (Synthetic Minority Over-sampling Technique):**  
     - Generates synthetic data points for the minority class (fraudulent transactions) to balance the dataset.
     - Models Evaluated: Logistic Regression and Random Forest Classifier.
  
  3. **Weighted Model:**  
     - Adjusts the model’s training process by assigning higher weights to the minority class.
     - Models Evaluated: Logistic Regression and Random Forest Classifier.

- **Model Evaluation:**  
  Each approach is evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations and performance comparisons help in selecting the most effective strategy.

## Data

Due to its size, the dataset is not included in this repository. Please download the Kaggle Credit Card Fraud Detection dataset here:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Notebook Overview

The main notebook, `Fruad_Detection.ipynb`, is organized into the following sections:

1. **Exploratory Data Analysis (EDA):**  
   - Initial insights into the data distribution and detection of class imbalance.
   - Visualization of transaction patterns and outliers.

2. **Data Preprocessing:**  
   - Data cleaning and normalization.
   - Feature engineering to prepare the dataset for modeling.

3. **Experimental Approaches:**  
   - **Under Sampling:**  
     - Implementation details and evaluation using Logistic Regression and Random Forest Classifier.
   - **SMOTE:**  
     - Application of SMOTE to create synthetic minority samples, with subsequent model training.
   - **Weighted Model:**  
     - Integration of class weights into the modeling process and performance comparison.

4. **Evaluation and Results:**  
   - Detailed performance metrics for each experiment.
   - Comparative analysis and visualizations to highlight the strengths and weaknesses of each approach.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/hussin-sobhy/credit-card-fraud-detection.git

2. **Navigate to the Project Directory:**
    ```bash
    cd credit-card-fraud-detection

3. **Install the Required Packages:**
    ```bash
    pip install -r requirements.txt

5. **Launch the Jupyter Notebook**:

Open Fruad_Detection.ipynb in Jupyter Notebook to follow along with the experiments.

##Usage

The notebook is designed to guide you through the entire fraud detection process. You can experiment with different parameters, models, or approaches to further enhance the detection capabilities. Each section is well-documented to facilitate easy understanding and modification.

##Contributing

Contributions to this project are highly welcome. If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.
