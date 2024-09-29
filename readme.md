# DSTS Assignment 1 - U3268781

This repository contains the solution for the DSTS Assignment 1 (U3268781). The main objective of this assignment is to complete tasks related to Data Science and Technology Systems, including working with datasets, creating models, and analyzing results to develop Predictive Modelling of Eating-Out Problem.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Predictive Model Used](#predictive-model-used)
- [Expected Output](#expected-output)
- [License](#license)

## Introduction
The repository contains code written in Python, which is provided in the form of a Jupyter Notebook (`Part_A_and_B.ipynb`). The code covers various tasks such as data preprocessing, analysis, and model implementation.

## Installation
To run the code, you need to set up a Python environment with the necessary dependencies. Follow the instructions below to set it up.

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook
- Tableau Desktop to run Zomato_Sydney_Restaurant_VIz.twb file(Optional)
- Docker..
- Required Python libraries: pandas, numpy,seaborn,geopandas, matplotlib, scikit-learn, etc.

### Installing libraries
You can install the required libraries by running:

pip install libraries 
or 
conda install libraries

eg 
pip install pandas numpy seaborn geopandas matplotlib scikit-learn ast


## How to Run

You can run the code in your local machine or in Docker
### Run in Local Machine

Clone the repository as follows 
- git clone https://github.com/gtshen173/DSTS_Assignment_1_U3268781.git
- cd DSTS_Assignment_1_U3268781

Open the file Part_A_and_B.ipynb file in Jupyter Notebook or Visual Studio Code and execute the cells in sequence to run the code.

It has two Parts 
- Part A: Importing and Understanding Data with visulizations 
- Part B: Predictive Modelling with both Regression and Classification.

## Predictive Model Used

- For Regression
    Linear Regression 
    Liner Regression using Batch Gradient Boost and
    SGDRegressor

- For Classification
    Logistic Regression 
    Random Forest Classifier
    Support Vector Classifier
    KNN Neighbors


## Expected Output 

Running the notebook will generate the following:

- Data preprocessing results (summary statistics, visualizations) for first part of code.
- Model training and evaluation results such as MSE, Accurarcy and Confusion Matrix for each models
- Relevant visualizations and analysis based on the dataset used