# DSTS Assignment 1 - U3268781





## Overview

This repository contains the code and resources for **Predictive Modelling of Eating-Out Problem**. The project is focused on implementing concepts from the **Data Science Technology and Systems** and demonstrates the use of various data processing and machine learning techniques. It also contains the use of git and docker repository.  


## Table of Contents
- [Introduction](#introduction)
- [Folder Structure](#folder_structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Predictive Model Used](#predictive-model-used)
- [Expected Output](#expected-output)
- [License](#license)



## Introduction
The repository contains code written in Python, which is provided in the form of a Jupyter Notebook (`Part_A_and_B.ipynb`). The code covers various tasks such as data preprocessing, analysis, and model(Regression and Classification) implementation. 
It also have tableau file (`Zomato_Sydney_Restaurant_Viz.twb`) where drashboard was created. It also contains folder named (`docker`) which has codes to do preprocessing and run regession and classification model. These codes were pushed to docker repository

## Folder Structure

- `data/` : This folder contains the dataset(s) used for the project. (Make sure to add the data manually if necessary.)
- `Docker/` : Contains the core Python scripts used for data processing, model building, and evaluation which were uploaded to Docker Container/repository.
- `results/` : Contains any output results, including trained models, plots, and evaluation metrics.
- `notebooks(ipynb)/` : Jupyter notebooks detailing exploratory data analysis, model development, and testing.
- `requirements.txt` : A list of dependencies required to run the code in this project. It is in the 'Docker/'

## Installation
To run the code, you need to set up a Python environment with the necessary dependencies. Follow the instructions below to set it up.

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook
- Tableau Desktop to run Zomato_Sydney_Restaurant_VIz.twb file(Optional)
- Git (optional, for cloning the repository)
- Docker to run the code directly for docker repository
- Required Python libraries: pandas, numpy,seaborn,geopandas, matplotlib, scikit-learn, etc.

### Installing libraries
You can install the required libraries by running:

```bash
pip install -r requirements.txt
```
    or 
```bash
pip install libraries 
```
    or 
```bash
conda install libraries
```
or 

```bash
pip install pandas numpy seaborn geopandas matplotlib scikit-learn ast
```


## How to Run

You can run the code in your local machine or in Docker

### Run in Local Machine

Clone the repository as follows 
```bash
git clone https://github.com/gtshen173/DSTS_Assignment_1_U3268781.git
```

```bash
- cd DSTS_Assignment_1_U3268781
```

Open the file Part_A_and_B.ipynb file in Jupyter Notebook or Visual Studio Code and execute the cells in sequence to run the code.

It has two Parts 
- Part A: Importing and Understanding Data with visulizations 
- Part B: Predictive Modelling with both Regression and Classification.

**To run only models in Local Machine**
- go to open folder name *docker*
- Run py files(preprocessing.py, classification.py, regression.py)
    Open and run the file in code editor like Visual Studio Code. 

- Excute the main.py 
    It will run whole code both regression and classification model together. 

### Run Docker Image

- Pull the Docker Image on CMD or bash 

```bash
docker pull gtshen173/dsts_assignment1_u3268781***
```

- Run the Docker Images

```bash
docker run -it gtshen173/dsts_assignment1_u3268781***
```
        (OR)

```bash
docker run -it -p 8080:80 gtshen173/dsts_assignment1_u3268781
```

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

**Data Analysis:** Insightful visualizations and statistics will be generated to help understand the dataset.

**Classification Models:** The output for classification includes model accuracy, confusion matrix for above four models. 

**Regression Models:** The output for this will MSE(Mean Square Error) for Linear Regression, Batch Gradient Boosting and SGDRegressor. 

**Tableau Visualizations:** If using Sydney Restaurants.twb, you'll get interactive dashboard visualizations related to Sydney's restaurants' data.

**Docker Images:** You will see the tabular comparasion of Regression Model based on MSE and Classification model based on Accurarcy and Confusion Matrix. 

**Author:** <br>
**Tshering Gyeltshen | U3268781** <br> 
**Master of Data Science in AI and Computational Modelling Student**<br>
**Faculty of Science and Technology**<br>
**University of Canberra**<br>
**Email: U3268781@uni.canberra.edu.au**<br>

####                Thank You