# WhereTo: A Travel Recommendation System
Course project for CSE 6240: Web Search and Text Mining, Spring 2023

Team 10 - Divya Umapathy, Harshvardhan Baldwa, Mansi Bhandari, Pankhuri Singh

## Description

Tourism is heavily characterized by a tourist’s preferences and better recommendations are made with more knowledge about the tourist’s personality. Inspired by this, we aim to build a personalized recommendation system for the users and understand the impact of different features on the recommendations. In order to achieve this, we will be using two different publicly available datasets, Gowalla and Foursquare, and comparing our findings across the proposed collaborative filtering-based and spatio-temporal based recommendation systems. We have used Blurring-Sharpening Process Model (**BSPM**) for collaborative filtering and Spatio-Temporal Transformer Recommender (**STTR**) for sequential recommendations. We processed the datasets to get the information and model it accordingly as an input for both the methods. Results from both the methods are nearly similar to what was presented in the original papers.

## Data Gathering
We utilize two different datasets Gowalla and Foursquare both of which are publically available and have been crawled from the respective websites. The datasets are available at [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html) and [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_7rmPjnwFGIx9). However, we also provide a script here to fetch the data and place it in required folders for our codes.

Run following command in your terminal or command prompt to get the data
```
bash fetch_data.sh
```

## Installing Python Packages
This project uses Python 3.9 or higher. To install all needed packages, run `pip install -r requirements.txt` within your terminal or command prompt.

That's it! Now you have installed the required libraries and good to go ahead with the code execution.

## Method Execution:

### Method 1 - BSPM:


### Method 2 - STTR: 

For method 2 execution, you can directly open the `main.ipynb` in any python notebook supporting IDE like VSCode or Google Colab to run the code and get the results. Kindly change the `dname` according to which dataset you plan to choose amongst Gowalla, Foursquare and NYC.

If using Google Colab, add the below code to the first cell of `main.ipynb`:
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Project10_S23/method_2
```

The file follows the below sequence:

1. Executes `preprocess.py` to preprocess the raw data and generate numpy files of cleaned and sorted data. This step is not required for all the datasets as we have already added the generated numpy files for Foursquare and Gowalla datasets in the data folder.

2. Executes `load.py` to generate the user embeddings and store them in a pickle file. 

3. Executes `train.py` file to train the model and save the results. Using main.ipynb file here provides the advantage of tuning the hyperparameters easily without having to make changes within different code sections of `train.py` file

The results are saved in `<dname>_results.txt` file
