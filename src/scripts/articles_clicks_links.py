# This script creates a DataFrame containing the following 9 columns: 
# - Top_1_name = name of the country the article was classified in
# - click_count = number of times the article occurs in the clicking paths of the Wikispeedia game
# - num_links_in = number of articles that lead to article of interest 
# - name_links_in = name of the artciles that lead to article of interest
# - num_links_out = number of links in the article of interest, leading out of the article
# - name_links_out = name of the articles that are references by the article of interest 

# Each row is an article. 
# This DataFrame is used for the rest of our analysis. 

# Imports
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.data.dataloader import *
from src.utils.functions import *

bad_articles = ['%C3%85land', '%C3%89douard_Manet', '%C3%89ire', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License']

def main():
    # load DataFrames containing the data
    articles = load_articles_dataframe()
    categories = load_categories_dataframe()
    finished_paths = load_path_finished_dataframe()
    unfinished_paths = load_path_unfinished_distance_dataframe()
    links = load_links_dataframe()

    # Remove start / target articles (they do not truly represent player behavior)
    paths_merged = pd.concat([finished_paths["path"].apply(lambda row: row.split(';')[1:-1]), unfinished_paths["path"].apply(lambda row: row.split(';')[1:])])

    # count number of clicks per article in the Wikispeedia game 
    df_clicks = click_count_in_paths(articles, paths_merged)

    # count number of links going out of each article, and get names 
    df_links_out = links_out_of_articles(links)

    # count number of links going into each article, and get names 
    df_links_in = links_into_articles(links)

    # load DatFrame contaning the ocuntry info for each article
    df_country_occurences = pd.read_csv("data/country_data_full_llama_improved.csv", index_col=0)

    # merge the 4 DataFrames to get final DataFrame containing click count, countries and links
    df_tot = pd.concat([df_country_occurences, df_clicks, df_links_in, df_links_out], axis=1)

    df_filtered = df_tot[~df_tot.index.isin(bad_articles)]
    
    df_filtered.to_csv('data/country_clicks_links.csv', index=True)

# main
if __name__ == "__main__":
    main()
