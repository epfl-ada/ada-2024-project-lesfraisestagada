import pandas as pd

from itertools import chain
from collections import Counter

from src.data.some_dataloader import *


"""
Some functions to be used in notebooks

Authors : 
    - Claire Friedrich
    - Theo Schifferli
    - Jeremy Barghorn
    - Bryan Gotti 
    - Oriane Petitphar
"""


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



def process_article_paths(finished_paths, unfinished_paths, articles):
    """
    Process the article paths to extract unique failure and success counts for each article 
    and calculate success and failure ratios.
    
    Args:   
        finished_paths (pd.DataFrame): DataFrame containing finished paths
        unfinished_paths (pd.DataFrame): DataFrame containing unfinished paths
    Returns: 
        pd.DataFrame: DataFrame containing the article counts and success/failure ratios
    """
    all_paths = pd.concat([finished_paths["path"], unfinished_paths["path"]])
    all_paths_merged = all_paths.apply(lambda row: row.split(';'))

    article_list = list(articles['articles'])

    # count total clicks per article
    clicks_list = list(chain.from_iterable(all_paths_merged.values.tolist()))
    total_click_counts = Counter(clicks_list)

    # helper function to count unique clicks
    def unique_click_counter(paths):
        unique_counts = Counter()
        for path in paths:
            unique_articles = set(path.split(';')) 
            unique_counts.update(unique_articles)
        return unique_counts

    successful_unique_counts = unique_click_counter(finished_paths["path"])
    unsuccessful_unique_counts = unique_click_counter(unfinished_paths["path"])

    data = []

    for article in article_list:
        total_click_count = total_click_counts[article]
        unique_success_count = successful_unique_counts[article]
        unique_failure_count = unsuccessful_unique_counts[article]
        
        # calc unique total click count (unique success + unique failure)
        unique_click_count = unique_success_count + unique_failure_count
        
        # calc ratios based on both total and unique counts
        success_ratio_total = successful_unique_counts[article] / total_click_count if total_click_count > 0 else 0
        failure_ratio_total = unsuccessful_unique_counts[article] / total_click_count if total_click_count > 0 else 0
        
        success_ratio_unique = unique_success_count / unique_click_count if unique_click_count > 0 else 0
        failure_ratio_unique = unique_failure_count / unique_click_count if unique_click_count > 0 else 0

        data.append({
            'article': article,
            'total_click_count': total_click_count,
            'unique_click_count': unique_click_count,
            'unique_success_count': unique_success_count,
            'unique_failure_count': unique_failure_count,
            'success_ratio_total': success_ratio_total,
            'failure_ratio_total': failure_ratio_total,
            'success_ratio_unique': success_ratio_unique,
            'failure_ratio_unique': failure_ratio_unique
        })

    # add "<" manually to take into account the go back action
    if "<" not in [entry['article'] for entry in data]:
        data.append({
            'article': "<",
            'total_click_count': total_click_counts.get("<", 0),
            'unique_click_count': successful_unique_counts.get("<", 0) + unsuccessful_unique_counts.get("<", 0),
            'unique_success_count': successful_unique_counts.get("<", 0),
            'unique_failure_count': unsuccessful_unique_counts.get("<", 0),
            'success_ratio_total': successful_unique_counts.get("<", 0) / total_click_counts.get("<", 1) if total_click_counts.get("<", 1) > 0 else 0,
            'failure_ratio_total': unsuccessful_unique_counts.get("<", 0) / total_click_counts.get("<", 1) if total_click_counts.get("<", 1) > 0 else 0,
            'success_ratio_unique': successful_unique_counts.get("<", 0) / (successful_unique_counts.get("<", 0) + unsuccessful_unique_counts.get("<", 0)) if (successful_unique_counts.get("<", 0) + unsuccessful_unique_counts.get("<", 0)) > 0 else 0,
            'failure_ratio_unique': unsuccessful_unique_counts.get("<", 0) / (successful_unique_counts.get("<", 0) + unsuccessful_unique_counts.get("<", 0)) if (successful_unique_counts.get("<", 0) + unsuccessful_unique_counts.get("<", 0)) > 0 else 0
        })

    df_articles_count = pd.DataFrame(data).sort_values(by='total_click_count', ascending=False).reset_index(drop=True)

    return df_articles_count


def get_articles_before_go_back(unfinished_paths, unique_dead_end_countries):
    """
    Analyzes the articles that appear immediately before the "<" in unfinished paths,
    relating them to country data from unique dead-end countries.

    Args:
        unfinished_paths (pd.DataFrame): DataFrame of unfinished paths, with a column 'articles' containing the path.
        unique_dead_end_countries (pd.DataFrame): DataFrame of unique dead-end countries with country data.

    Returns:
        pd.DataFrame: Sorted DataFrame of articles appearing before "<" with associated country data.
    """
    # get the article immediately before "<" for each path that contains "<"
    unfinished_paths['before_back_article'] = unfinished_paths['articles'].apply(
        lambda x: x[x.index('<') - 1] if '<' in x and x.index('<') > 0 else None
    )

    # drop rows where before_back_article is None (paths that do not contain "<")
    unfinished_paths_with_back = unfinished_paths.dropna(subset=['before_back_article'])

    # count occurrences of each article before "<" in unfinished paths
    before_last_article_counts = unfinished_paths_with_back['before_back_article'].value_counts().reset_index()
    before_last_article_counts.columns = ['article_before_back', 'count']

    # standardize format by removing underscores and converting to lowercase for matching
    before_last_article_counts['cleaned_article_before_back'] = before_last_article_counts['article_before_back'].str.replace('_', ' ').str.lower()
    unique_dead_end_countries['cleaned_country'] = unique_dead_end_countries['Top_1_name'].str.replace('_', ' ').str.lower()

    # merge to relate articles before "<" to country data from unique_dead_end_countries
    before_last_article_analysis = before_last_article_counts.merge(
        unique_dead_end_countries[['cleaned_country', 'Top_1_name', 'click_count', 'sum_num_links_out', 'mean_failure_ratio_unique']], 
        left_on='cleaned_article_before_back', 
        right_on='cleaned_country', 
        how='inner'
    )

    # sort to show articles with the highest click counts and lowest mean outgoing links
    sorted_before_last_article = before_last_article_analysis.sort_values(
        by=['click_count', 'sum_num_links_out'], 
        ascending=[False, True]
    )    

    return sorted_before_last_article

