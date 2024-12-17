import torch
import numpy as np
import pandas as pd
import os
        

def load_articles_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the articles.tsv file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    return pd.read_csv(path + "articles.tsv", names = ["articles"] ,skiprows=12, sep="\t")


def load_categories_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the categories.tsv file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    return pd.read_csv(path + "categories.tsv", names = ["articles", "category"], skiprows=13, sep="\t")


def load_links_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the links.tsv file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    return pd.read_csv(path + "links.tsv", names = ["linkSource", "linkTarget"],  skiprows=12, sep="\t")

def load_shortest_path_distance_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the shortest-path-distance.txt file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    shortest_path_distance = pd.read_csv(path + "shortest-path-distance-matrix.txt", names=["distances"], skiprows=17)
    split_shortest_path_distance = pd.DataFrame(shortest_path_distance.apply(lambda row: list((map(str, row['distances']))), axis=1), columns=['distances'])
    split_shortest_path_distance = pd.DataFrame(shortest_path_distance.apply(lambda row: ",".join(row.distances), axis=1), columns=['distances'])
    return split_shortest_path_distance["distances"].str.split(',', expand=True).replace('_', None).astype('Int64')


def load_path_finished_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the paths_finished.tsv file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    return pd.read_csv(path + "paths_finished.tsv", names = ["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"] ,skiprows=16, sep="\t")


def load_path_unfinished_distance_dataframe(path = "./data/wikispeedia_paths-and-graph/"):
    """Load the paths_unfinished.tsv file into a pandas dataframe

    Args:
        path (str, optional): Path to the file. Defaults to "./data/wikispeedia_paths-and-graph/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    return pd.read_csv(path + "paths_unfinished.tsv", names=["hashedIpAddress",   "timestamp ",  "durationInSec",  "path",   "target",   "type"], skiprows=17, sep="\t")

def load_publications_dataframe(path = "./data/"):
    """Load the publications.csv file into a pandas dataframe and keep only the relevant columns

    Args:
        path (str, optional): Path to the file. Defaults to "./data/".

    Returns:
        pandas.dataframe: The loaded dataframe
    """
    pub_data = pd.read_csv(path + "publications_2007.csv")
    cols_to_keep = ["Rank", "Country", "Citable documents"]
    return pub_data[cols_to_keep].rename(columns={'Country': 'country', 'Citable documents': 'Publications'})


def plaintext_files_iterator(path = "./data/plaintext_articles/"):
    """Iterate over the plaintext files in the directory

    Args:
        path (str, optional): Path to the directory. Defaults to "./data/plaintext_articles/".

    Yields:
        str: The content of the file
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            yield file_path
            

def file_finder(path = "./data/plaintext_articles/", article_name = None):
    """Find a file in the directory

    Args:
        path (str, optional): File folder to search for. Defaults to "./data/plaintext_articles/".
        article_name (str, optional): File name to search for. Defaults to None.

    Returns:
        file: file reader
    """
    
    if article_name is None:
        return None
    
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            if article_name in filename:
                with open(file_path, 'r',encoding="utf-8") as f:
                    content = f.read()
                    return content
    return None