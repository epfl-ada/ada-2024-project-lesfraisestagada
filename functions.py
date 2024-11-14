import pandas as pd
import networkx as nx

from itertools import chain
from collections import Counter

from src.data.some_dataloader import *


# Some functions 
def click_count_in_paths(articles, paths):
    """Count the number of clicks per article in the Wikispeedia game. 
    Consider both fiished and unfinished paths. 

    Args:
        articles (pd.DataFrame): aticles in the Wikipedia data 
        paths (pandas.core.series.Series): all elements (articles) contained in the Wikispeedia paths
    
    Returns:
        pd.DataFrame: A DataFrame with the counts of the clicks per article 
        The format is as follows: 
        index: article name,
        columns: click_count,
        values: number of clicks per article
    """
    clicks_list = paths.values.flatten().tolist()
    clicks_list = list(chain.from_iterable(clicks_list))

    print(f'there are {len(clicks_list)} clicks in the whole whikispeedia dataset (both finished and unfinished paths)')

    article_list = list(articles['articles'])

    # count number of occurences of each article
    counts = Counter(clicks_list)
    occurences = {item: counts[item] for item in article_list}

    df_articles_count = pd.DataFrame(list(occurences.items()), columns = ['articles', 'click_count']).set_index('articles') # df containing the click counts for each article
    df_articles_count.index.name = None

    return df_articles_count


def links_out_of_articles(links):
    """Count the number of links going out of each article, i.e. the number of links contained in each article.
    Also find the names of the articles that are referenced in each article. 

    Args:
        links (pd.DataFrame): a DataFrame with 2 columns (linkSource and linkTarget), summarizing all links contained in each article 

    Returns:
        pd.DataFrame: A DataFrame with the counts of the links per article 
        The format is as follows: 
        index: article name,
        columns: num_links_out, name_links_out 
        values: number of links per article, name of the linked articles 
    """
    num_links_out = links.groupby('linkSource').agg('count').reset_index()
    num_links_out = num_links_out.set_index('linkSource')
    num_links_out.index.name = None
    num_links_out = num_links_out.rename(columns={"linkTarget": "num_links_out"})

    num_links_out["name_links_out"] = None

    index_links_out = links.groupby('linkSource').groups # a dictionary whose keys are the computed unique groups and corresponding values are the axis labels belonging to each group.
    list_links_out = list(index_links_out.items())

    for article in range(len(list_links_out)):
        num_links_out["name_links_out"].iloc[article] = links.iloc[list_links_out[article][1]]['linkTarget'].tolist()

    num_links_out = num_links_out[:-1]

    return num_links_out


def links_into_articles(links):
    """Count the number of links leading to each article, i.e. the number of articles that reference article of interest.
    Also find the names of the articles that reference article of interest. 

    Args:
        links (pd.DataFrame): a DataFrame with 2 columns (linkSource and linkTarget), summarizing all links contained in each article 

    Returns:
        pd.DataFrame: A DataFrame with the counts of the links per article 
        The format is as follows: 
        index: article name,
        columns: num_links_in, name_links_in 
        values: number of links leading to each article, name of the articles that reference article of interest 
    """
    num_links_in = links.groupby('linkTarget').agg('count').reset_index()
    num_links_in = num_links_in.set_index('linkTarget')
    num_links_in.index.name = None
    num_links_in = num_links_in.rename(columns={"linkSource": "num_links_in"})

    num_links_in["name_links_in"] = None

    index_links_in = links.groupby('linkTarget').groups # a dictionary whose keys are the computed unique groups and corresponding values are the axis labels belonging to each group.
    list_links_in = list(index_links_in.items())

    for article in range(len(list_links_in)):
        num_links_in["name_links_in"].iloc[article] = links.iloc[list_links_in[article][1]]['linkSource'].tolist()

    return num_links_in


def pagerank(links):
    """Run the pagerank algorithm on the Wikipedia graph.

    Args:
        links (pd.DataFrame): a DataFrame with 2 columns (linkSource and linkTarget), summarizing all links contained in each article

    Returns:
        pd.DataFrame: A DataFrame with the rank of each article in the graph
        The format is as follows:
        columns: article_name, rank
        values: name of the article, rank computed by pagerank
    """
    edges = [(row['linkSource'], row['linkTarget']) for index, row in links.iterrows()]
    G = nx.DiGraph(edges)
    pagerank_result = nx.pagerank(G)
    df_pagerank = pd.DataFrame({
        'article_name': pagerank_result.keys(),
        'rank': pagerank_result.values()  # Order will correspond to keys
    })
    df_pagerank.sort_values(by='rank', ascending=False, inplace=True, ignore_index=True)
    return df_pagerank


def find_pairs(paths):
    """ Find all pairs of articles within game paths. If a path is [a, b, c, <, d], this function finds every path between articles so 
    [(a,b), (b,c), (b,d)].

    Args: game paths with every article split

    Returns: 
        List: a list containing every pair of articles found in the game paths
    """

    all_pairs = []

    # Iterate over all rows of paths
    for i in range(len(paths)): 
        # preprocess to get rid of "<" and not loose path information > [a, b, <, c] becomes [a, c]
        new_row = []

        for j in range(len(paths.iloc[i])):
            if paths.iloc[i][j] != '<': 
                new_row.append(paths.iloc[i][j])

            else :
                new_row.pop()
        new_row = pd.Series(new_row)

        # For each row with a path [a,b,c,d], we create a list of [(a,b), (b,c), (c,d)]
        pairs_row = [(new_row.iloc[j], new_row.iloc[j+1]) for j in range(len(new_row) - 1)]

        # Pairs found for each rows are combined in a unique list
        all_pairs = all_pairs + pairs_row

    return all_pairs