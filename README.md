# WhereTo: A Travel Recommendation System
Course project for CSE 6240: Web Search and Text Mining, Spring 2023

Team 10 - Divya Umapathy, Harshvardhan Baldwa, Mansi Bhandari, Pankhuri Singh

## Description

<Paste the Abstract/Introduction here>

## Data Gathering
We utilize two different datasets Gowalla and Foursquare both of which are publically available and have been crawled from the respective websites. The datasets are available at [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html) and [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_7rmPjnwFGIx9). However, we also provide a script here to fetch the data and place it in required folders for our codes.

Run following command in your terminal or command prompt to get the data
```
bash fetch_data.sh
```

<!---
The raw data for the datasets can be downloaded from links below:
- [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html): Download the two datasets present under the 'Files' section, unzip them and place the result files under './data/Gowalla/' folder
- [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_7rmPjnwFGIx9): Download the files for 'Global-scale Check-in Dataset with User Social Networks', unzip them and place the result folder under './data/Foursquare/' folder
-->

## Installing Python Packages
This project uses Python 3.9 or higher. To install all needed packages, run `pip install -r requirements.txt` within your terminal or command prompt.

That's it! Now you have installed the required libraries and good to go ahead with the code execution.

## Method Execution:

### Method 1:


### Method 2:

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
