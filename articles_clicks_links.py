# This script creates a DataFrame containing the following 9 columns: 
# - Top_1_name = name of the country that occurs the most in the article
# - Top_2_name = name of the country that occurs the second most in the article
# - Top_1_count = number of times that the Top1 country occurs
# - Top_2_count = number of times that the Top2 country occurs
# - click_count = number of times the article occurs in the clicking paths of the Wikispeedia game
# - num_links_in = number of articles that lead to article of interest 
# - name_links_in = name of the artciles that lead to article of interest
# - num_links_out = number of links in the article of interest, leading out of the article
# - name_links_out = name of the articles that are references by the article of interest 

# Each row is an article. 
# This DataFrame is used for the rest of our analysis. 

# Imports
import pandas as pd
import seaborn as sns
import scipy

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


def plot_regression_clicks_links(data, x, y, ax, position, x_label, y_label, title):
    """ Plot scatter plot of the data and display regression line with equation and R-squared.

    Args:
        data (pd.DataFrame): a DataFrame with x and y as columns
        x (str): name of the column that we want to plot on the x-axis 
        y (str): name of the column that we want to plot on the y-axis
        ax: axis on which we plot 
        position (list): x-y position of the regression equation
        x_label (str): name of x-axis
        y_label (str): name of y-axis
        title (str): the title of the plot
   
    """
    plot = sns.regplot(data=data, x=x, y=y, ax=ax, order=1, line_kws={'color':'red'}, robust=True) # downweight the influence of outliers(1 outlier on the plot!)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Compute the regression
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                        y=plot.get_lines()[0].get_ydata())
    # Add regression equation to the plot
    ax.text(position[0], position[1], f"y = {round(intercept, 2)} + {round(slope, 2)} * x \n R-squared = {round(r, 2)}")



# main
if __name__ == "__main__":
    # load DataFrames containing the data
    articles = load_articles_dataframe()
    categories = load_categories_dataframe()
    finished_paths = load_path_finished_dataframe()
    unfinished_paths = load_path_unfinished_distance_dataframe()
    links = load_links_dataframe()

    paths = pd.concat([finished_paths["path"], unfinished_paths["path"]])
    paths_merged = paths.apply(lambda row: row.split(';'))

    # count number of clicks per article in the Wikispeedia game 
    df_clicks = click_count_in_paths(articles, paths_merged)

    # count number of links going out of each article, and get names 
    df_links_out = links_out_of_articles(links)

    # count number of links going into each article, and get names 
    df_links_in = links_into_articles(links)

    # load DatFrame contaning the ocuntry info for each article
    df_country_occurences = pd.read_csv("data/country_occurences.csv", names = ["Top_1_name", "Top_2_name", "Top_1_count", "Top_2_count"], skiprows=1)

    # merge the 4 DataFrames to get final DataFrame containing click count, countries and links
    df_tot = pd.concat([df_country_occurences, df_clicks, df_links_in, df_links_out], axis=1)
    df_tot.to_csv('data/country_clicks_links.csv', index=True)

