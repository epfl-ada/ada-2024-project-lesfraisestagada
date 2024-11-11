import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

def plot_res_stats_model(res):
    variables = res.params.index
    coefficients = res.params.values
    p_values = res.pvalues
    standard_errors = res.bse.values
    res.conf_int()

    l1, l2, l3, l4 = zip(*sorted(zip(coefficients[1:], variables[1:], standard_errors[1:], p_values[1:])))

    plt.errorbar(l1, np.array(range(len(l1))), xerr= 2*np.array(l3), linewidth = 1,
                linestyle = 'none',marker = 'o',markersize= 3,
                markerfacecolor = 'black',markeredgecolor = 'black', capsize= 5)

    plt.vlines(0,0, len(l1), linestyle = '--')

    plt.yticks(range(len(l2)),l2);


    plt.xlabel('Coefficients')
    plt.ylabel('Variables')
    plt.title('Coefficients of the linear regression model')
    
    
def plot_country_to_country(df):
    df.dropna(subset=["Top_1_name"], inplace=True)
    df["name_links_out"] = df["name_links_out"].fillna("[]")
    df["num_links_in"] = df["num_links_in"].fillna(0)
    df["num_links_out"] = df["num_links_out"].fillna(0)
    df["name_links_out"] = df["name_links_out"].apply(literal_eval)
    
    l = []

    for idx, row in df.iterrows():
        links_out_list = list(row["name_links_out"])
        for out_link in links_out_list:
            try:
                l.append(f"{row['Top_1_name']} -> {df.loc[out_link]['Top_1_name']}")
            except:
                pass
            
    df = pd.DataFrame(l, columns=["links"])
    df.groupby("links").size().sort_values(ascending=False).head(100).plot(kind="bar", figsize=(20, 5))
    plt.ylabel("Number of links between countries")
    plt.xlabel("Country links")
    plt.title("Top 100 country links by occurences")
    plt.show()
    

def paths_to_country(df, country_clicks, finished = True):
    country_clicks.dropna(subset=["Top_1_name"], inplace=True)

    l = []
    for idx, row in df.iterrows():
        start_article = row["path"][0]
        end_article = row["path"][-1]
        
        if start_article != '<' and end_article != '<':
        
            start_country = country_clicks.loc[start_article]["Top_1_name"]
            end_country = country_clicks.loc[end_article]["Top_1_name"]
            
            l.append(f"{start_country} -> {end_country}")
            
    
    res = pd.DataFrame(l, columns=["links"])
    res.groupby("links").size().sort_values(ascending=False).head(100).plot(kind="bar", figsize=(20, 5))
    plt.ylabel("Number of links between countries")
    plt.xlabel("Country links")
    if finished:
        plt.title("Top 100 country connections between the start and end of a finished path")
    else:
        plt.title("Top 100 country connections between the start and end of an unfinished path")
    plt.show()