import pandas as pd
import networkx as nx

from itertools import chain
from collections import Counter
from country_list import countries_for_language

from src.data.dataloader import *

from src.scripts.articles_clicks_links import bad_articles


"""
Some functions to be used in notebooks

Authors : 
    - Claire Friedrich
    - Theo Schifferli
    - Jeremy Barghorn
    - Bryan Gotti 
    - Oriane Petit-Phar
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
    # Remove bad articles
    edges = [(row['linkSource'], row['linkTarget']) for index, row in links.iterrows()
             if (row['linkSource'] not in bad_articles) and (row['linkTarget'] not in bad_articles)
            ]

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


def analyze_last_articles_in_unfinished_paths(unfinished_paths, unique_dead_end_countries):
    """
    Analyzes the last articles in unfinished paths and identifies which of them are known dead-end countries.

    Args:
        unfinished_paths (pd.DataFrame): DataFrame containing paths in unfinished articles, with 'path' column.
        unique_dead_end_countries (pd.DataFrame): DataFrame of dead-end countries with 'Top_1_name' column.

    Returns:
        pd.DataFrame: DataFrame with last articles in unfinished paths that are also dead-end countries.
    """
    # Split the paths into lists of articles
    unfinished_paths['articles'] = unfinished_paths['path'].astype(str).str.split(';')

    # Extract the last article (country) from each unfinished path
    unfinished_paths['last_article'] = unfinished_paths['articles'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

    # Count occurrences of each last article
    last_article_counts = unfinished_paths['last_article'].value_counts().reset_index()
    last_article_counts.columns = ['last_article', 'count']

    # Standardize format for merging by removing underscores and converting to lowercase
    last_article_counts['cleaned_last_article'] = last_article_counts['last_article'].str.replace('_', ' ').str.lower()
    unique_dead_end_countries['cleaned_country'] = unique_dead_end_countries['Top_1_name'].str.replace('_', ' ').str.lower()

    # Merge to identify last articles in unfinished paths that are also known dead-end countries
    last_dead_end_countries = last_article_counts.merge(
        unique_dead_end_countries, 
        left_on='cleaned_last_article', 
        right_on='cleaned_country', 
        how='inner'
    )

    return last_dead_end_countries


def generate_annotations_and_show_agreement(data, write = False, write_agreement = False):
    articles = pd.DataFrame(data["Text search"].index)

    subset_1 = articles.sample(10, random_state=0).values.flatten().tolist()
    subset_2 = articles.sample(10, random_state=1).values.flatten().tolist()
    subset_3 = articles.sample(10, random_state=2).values.flatten().tolist()
    subset_4 = articles.sample(10, random_state=3).values.flatten().tolist()
    subset_5 = articles.sample(10, random_state=4).values.flatten().tolist()
    subset_6 = articles.sample(10, random_state=5).values.flatten().tolist()

    claire = subset_1 + subset_3
    theo = subset_2 + subset_4
    oriane = subset_1 + subset_4
    bryan = subset_2 + subset_5
    jeremy = subset_3 + subset_5

    claire = pd.DataFrame(index=claire, columns=["country"])
    theo = pd.DataFrame(index=theo, columns=["country"])
    oriane = pd.DataFrame(index=oriane, columns=["country"])
    bryan= pd.DataFrame(index=bryan, columns=["country"])
    jeremy = pd.DataFrame(index=jeremy, columns=["country"])

    if write :
        pd.DataFrame(claire).to_csv("claire.csv")
        pd.DataFrame(theo).to_csv("theo.csv")
        pd.DataFrame(oriane).to_csv("oriane.csv")
        pd.DataFrame(bryan).to_csv("bryan.csv")
        pd.DataFrame(jeremy).to_csv("jeremy.csv")

        countries = list(dict(countries_for_language('en')).values())
        pd.DataFrame(countries).to_csv("countries.csv")

    annotation_path = "./data/annotated/"

    claire = pd.read_csv(annotation_path + "subset_claire.csv", index_col=0, na_values="None")
    theo = pd.read_csv(annotation_path + "subset_theo.csv", index_col=0, na_values="None")
    oriane = pd.read_csv(annotation_path + "subset_oriane.csv", index_col=0, na_values="None")
    bryan = pd.read_csv(annotation_path + "subset_bryan.csv", index_col=0, na_values="None")
    jeremy = pd.read_csv(annotation_path + "subset_jeremy.csv", index_col=0, na_values="None")

    subset_1_c = claire[:10]
    subset_3_c = claire[10:]

    subset_2_t = theo[:10]
    subset_4_t = theo[10:]

    subset_1_o = oriane[:10]
    subset_4_o = oriane[10:]

    subset_2_b = bryan[:10]
    subset_5_b = bryan[10:]

    subset_3_j = jeremy[:10]
    subset_5_j = jeremy[10:]

    comparison1 = subset_1_c["country"].str.lower().fillna("nan") == subset_1_o["country"].str.lower().fillna("nan")
    print(f"Agreement between Claire and Oriane: {comparison1.sum() * 10}%")
    comparison2 = subset_2_t["country"].str.lower().fillna("nan") == subset_2_b["country"].str.lower().fillna("nan")
    print(f"Agreement between Theo and Bryan: {comparison2.sum() * 10}%")
    comparison3 = subset_3_c["country"].str.lower().fillna("nan") == subset_3_j["country"].str.lower().fillna("nan")
    print(f"Agreement between Claire and Jeremy: {comparison3.sum() * 10}%")
    comparison4 = subset_4_t["country"].str.lower().fillna("nan") == subset_4_o["country"].str.lower().fillna("nan")
    print(f"Agreement between Theo and Oriane: {comparison4.sum() * 10}%")
    comparison5 = subset_5_b["country"].str.lower().fillna("nan") == subset_5_j["country"].str.lower().fillna("nan")
    print(f"Agreement between Bryan and Jeremy: {comparison5.sum() * 10}%")

    agreement_df = pd.concat([
        subset_1_c.loc[comparison1],
        subset_2_b.loc[comparison2],
        subset_3_c.loc[comparison3],
        subset_4_t.loc[comparison4],
        subset_5_b.loc[comparison5]
    ], ignore_index=False)

    if write:
        agreement_df.to_csv("data/annotated/consensus.csv")

    if write_agreement:
        print(data["Improved classification with LLaMa"].loc[agreement_df.index]["Top_1_name"].str.lower().fillna("nan"))
        
    return agreement_df


def normalize_clicks():
    """Computes the normalization of click counts of each country as described in part 5 of the notebook"""
    articles = pd.read_csv("data/country_clicks_links.csv", index_col=0).reset_index().rename(columns={'index': 'article'})
    countries_links_in = articles[['Top_1_name', 'num_links_in']].groupby('Top_1_name').agg('sum')
    countries_clicks = articles[['Top_1_name', 'click_count']].groupby('Top_1_name').agg('sum')
    countries = pd.concat([countries_links_in, countries_clicks], axis=1)
    countries['click_count_normalized'] = countries['click_count'] / countries['num_links_in']
    countries = countries.reset_index()

    return countries

def compute_player_frequencies():
    """Compute the players rank as described in PageRank analysis (part 5 of notebook)"""
    df_player_frequencies = pd.read_csv("data/country_clicks_links.csv", index_col=0)
    df_player_frequencies['rank'] = df_player_frequencies.click_count / df_player_frequencies.click_count.sum()
    df_player_frequencies.index.name = 'article_name'
    df_player_frequencies.reset_index(inplace=True)
    df_player_frequencies.sort_values(by='rank', ascending=False, inplace=True, ignore_index=True)
    df_player_frequencies = df_player_frequencies[['article_name', 'rank']]

    return df_player_frequencies

def aggregate_ranks_by_country(
    df_ranks
):
    """Aggregate player ranks and PageRanks by country as described PageRank analysis (part 5 of notebook)"""

    df_ranks = df_ranks[['country_name', 'rank']].groupby(['country_name'], as_index=False).sum()

    return df_ranks

def rank_diff(
    df_pagerank,
    df_player_frequencies
):
    """Computing rank difference between players rank and PageRank (part 5 of notebook)"""

    rank_v_freq_2_columns = pd.merge(df_pagerank[['article_name', 'rank']], df_player_frequencies[['country_name', 'article_name', 'rank']], on='article_name', suffixes=('_pagerank', '_players'), how='right')
    rank_v_freq_2_columns = rank_v_freq_2_columns.fillna({'rank_pagerank': 0}) # fill missing pagerank values (those where isolated articles that were not added to the graph, they all have click_count 0 anyway and the pagerank should be 0 too)
    rank_v_freq_2_columns['rank_diff'] = rank_v_freq_2_columns['rank_players'] - rank_v_freq_2_columns['rank_pagerank']
    rank_v_freq_2_columns.sort_values(by='rank_pagerank', inplace=True, ignore_index=True, ascending=False)
    rank_v_freq_countries = rank_v_freq_2_columns.drop(columns=['article_name']).groupby('country_name', as_index=False, dropna=False).sum()
    rank_v_freq_countries.sort_values(by='rank_pagerank', inplace=True, ignore_index=True, ascending=False)
    
    return rank_v_freq_countries


def generate_annotations_and_show_agreement(data, write = False, write_agreement = False):
    articles = pd.DataFrame(data["Text search"].index)

    subset_1 = articles.sample(10, random_state=0).values.flatten().tolist()
    subset_2 = articles.sample(10, random_state=1).values.flatten().tolist()
    subset_3 = articles.sample(10, random_state=2).values.flatten().tolist()
    subset_4 = articles.sample(10, random_state=3).values.flatten().tolist()
    subset_5 = articles.sample(10, random_state=4).values.flatten().tolist()
    subset_6 = articles.sample(10, random_state=5).values.flatten().tolist()

    claire = subset_1 + subset_3
    theo = subset_2 + subset_4
    oriane = subset_1 + subset_4
    bryan = subset_2 + subset_5
    jeremy = subset_3 + subset_5

    claire = pd.DataFrame(index=claire, columns=["country"])
    theo = pd.DataFrame(index=theo, columns=["country"])
    oriane = pd.DataFrame(index=oriane, columns=["country"])
    bryan= pd.DataFrame(index=bryan, columns=["country"])
    jeremy = pd.DataFrame(index=jeremy, columns=["country"])

    if write :
        pd.DataFrame(claire).to_csv("claire.csv")
        pd.DataFrame(theo).to_csv("theo.csv")
        pd.DataFrame(oriane).to_csv("oriane.csv")
        pd.DataFrame(bryan).to_csv("bryan.csv")
        pd.DataFrame(jeremy).to_csv("jeremy.csv")

        countries = list(dict(countries_for_language('en')).values())
        pd.DataFrame(countries).to_csv("countries.csv")

    annotation_path = "./data/annotated/"

    claire = pd.read_csv(annotation_path + "subset_claire.csv", index_col=0, na_values="None")
    theo = pd.read_csv(annotation_path + "subset_theo.csv", index_col=0, na_values="None")
    oriane = pd.read_csv(annotation_path + "subset_oriane.csv", index_col=0, na_values="None")
    bryan = pd.read_csv(annotation_path + "subset_bryan.csv", index_col=0, na_values="None")
    jeremy = pd.read_csv(annotation_path + "subset_jeremy.csv", index_col=0, na_values="None")

    subset_1_c = claire[:10]
    subset_3_c = claire[10:]

    subset_2_t = theo[:10]
    subset_4_t = theo[10:]

    subset_1_o = oriane[:10]
    subset_4_o = oriane[10:]

    subset_2_b = bryan[:10]
    subset_5_b = bryan[10:]

    subset_3_j = jeremy[:10]
    subset_5_j = jeremy[10:]

    comparison1 = subset_1_c["country"].str.lower().fillna("nan") == subset_1_o["country"].str.lower().fillna("nan")
    print(f"Agreement between Claire and Oriane: {comparison1.sum() * 10}%")
    comparison2 = subset_2_t["country"].str.lower().fillna("nan") == subset_2_b["country"].str.lower().fillna("nan")
    print(f"Agreement between Theo and Bryan: {comparison2.sum() * 10}%")
    comparison3 = subset_3_c["country"].str.lower().fillna("nan") == subset_3_j["country"].str.lower().fillna("nan")
    print(f"Agreement between Claire and Jeremy: {comparison3.sum() * 10}%")
    comparison4 = subset_4_t["country"].str.lower().fillna("nan") == subset_4_o["country"].str.lower().fillna("nan")
    print(f"Agreement between Theo and Oriane: {comparison4.sum() * 10}%")
    comparison5 = subset_5_b["country"].str.lower().fillna("nan") == subset_5_j["country"].str.lower().fillna("nan")
    print(f"Agreement between Bryan and Jeremy: {comparison5.sum() * 10}%")

    agreement_df = pd.concat([
        subset_1_c.loc[comparison1],
        subset_2_b.loc[comparison2],
        subset_3_c.loc[comparison3],
        subset_4_t.loc[comparison4],
        subset_5_b.loc[comparison5]
    ], ignore_index=False)

    if write:
        agreement_df.to_csv("data/annotated/consensus.csv")

    if write_agreement:
        print(data["Improved classification with LLaMa"].loc[agreement_df.index]["Top_1_name"].str.lower().fillna("nan"))
        
    return agreement_df
def normalize_clicks():
    """Computes the normalization of click counts of each country as described in part 5 of the notebook"""
    articles = pd.read_csv("data/country_clicks_links.csv", index_col=0).reset_index().rename(columns={'index': 'article'})
    countries_links_in = articles[['Top_1_name', 'num_links_in']].groupby('Top_1_name').agg('sum')
    countries_clicks = articles[['Top_1_name', 'click_count']].groupby('Top_1_name').agg('sum')
    countries = pd.concat([countries_links_in, countries_clicks], axis=1)
    countries['click_count_normalized'] = countries['click_count'] / countries['num_links_in']
    countries = countries.reset_index()

    return countries

def compute_player_frequencies():
    """Compute the players rank as described in PageRank analysis (part 5 of notebook)"""
    df_player_frequencies = pd.read_csv("data/country_clicks_links.csv", index_col=0)
    df_player_frequencies['rank'] = df_player_frequencies.click_count / df_player_frequencies.click_count.sum()
    df_player_frequencies.index.name = 'article_name'
    df_player_frequencies.reset_index(inplace=True)
    df_player_frequencies.sort_values(by='rank', ascending=False, inplace=True, ignore_index=True)
    df_player_frequencies = df_player_frequencies[['article_name', 'rank']]

    return df_player_frequencies

def aggregate_ranks_by_country(
    df_ranks
):
    """Aggregate player ranks and PageRanks by country as described PageRank analysis (part 5 of notebook)"""

    df_ranks = df_ranks[['country_name', 'rank']].groupby(['country_name'], as_index=False).sum()

    return df_ranks

def rank_diff(
    df_pagerank,
    df_player_frequencies
):
    """Computing rank difference between players rank and PageRank (part 5 of notebook)"""

    rank_v_freq_2_columns = pd.merge(df_pagerank[['article_name', 'rank']], df_player_frequencies[['country_name', 'article_name', 'rank']], on='article_name', suffixes=('_pagerank', '_players'), how='right')
    rank_v_freq_2_columns = rank_v_freq_2_columns.fillna({'rank_pagerank': 0}) # fill missing pagerank values (those where isolated articles that were not added to the graph, they all have click_count 0 anyway and the pagerank should be 0 too)
    rank_v_freq_2_columns['rank_diff'] = rank_v_freq_2_columns['rank_players'] - rank_v_freq_2_columns['rank_pagerank']
    rank_v_freq_2_columns.sort_values(by='rank_pagerank', inplace=True, ignore_index=True, ascending=False)
    rank_v_freq_countries = rank_v_freq_2_columns.drop(columns=['article_name']).groupby('country_name', as_index=False, dropna=False).sum()
    rank_v_freq_countries.sort_values(by='rank_pagerank', inplace=True, ignore_index=True, ascending=False)
    
    return rank_v_freq_countries