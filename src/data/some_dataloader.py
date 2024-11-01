import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os



class SomeDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, dataset_parameters, **kwargs):
        super().__init__()
        ...


class SomeDatamodule(DataLoader):
    """
    Allows you to sample train/val/test data, to later do training with models.
        
    """
    def __init__(self):
        super().__init__()
        ...
        

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