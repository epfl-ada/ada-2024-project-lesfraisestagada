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
from data.dataloader import *
from src.utils.functions import *

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

    # load publications data 
    df_publications = load_publications_dataframe()
    df_publications["Country"] = df_publications["Country"].str.lower()

    # merge the publications data with the rest of the data
    df_tot = df_tot.merge(df_publications, left_index=True, right_index=True)
    
    df_tot.to_csv('data/country_clicks_links.csv', index=True)

