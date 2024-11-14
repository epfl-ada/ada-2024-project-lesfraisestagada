import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from ast import literal_eval
from collections import Counter


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
    plt.title("Figure 4: Top 100 country links by occurences")
    plt.show()
    

def paths_to_country(df, country_clicks, fig_num= "", finished = True):
    country_clicks.dropna(subset=["Top_1_name"], inplace=True)

    start = []
    end = []
    for idx, row in df.iterrows():
        start_article = row["path"][0]
        end_article = row["path"][-1]
        
        if start_article != '<' and end_article != '<':
            try:
                start_country = country_clicks.loc[start_article]["Top_1_name"]
                end_country = country_clicks.loc[end_article]["Top_1_name"]
                
                start.append(start_country)
                end.append(end_country)
            except:
                pass
            
            
    start_counts = Counter(start)
    end_counts = Counter(end)
    
    start_sorted = dict(sorted(start_counts.items(), key=lambda item: item[1], reverse=True)[:40])
    end_sorted = dict(sorted(end_counts.items(), key=lambda item: item[1], reverse=True)[:40])

    fig, axs = plt.subplots(1, 2 if finished else 1, figsize=(14, 6))

    if finished:
        # Plot the start list occurrences
        axs[0].bar(start_sorted.keys(), start_sorted.values(), color="skyblue")
        axs[0].set_title("Start Country Occurrences" + " (Finished Paths)" if finished else "Start Country Occurrences" + " (Unfinished Paths)")
        axs[0].tick_params(axis='x', rotation=90)
        axs[0].set_ylabel("Occurrences")

    else :
        axs.bar(start_sorted.keys(), start_sorted.values(), color="skyblue")
        axs.set_title("Start Country Occurrences" + " (Finished Paths)" if finished else "Start Country Occurrences" + " (Unfinished Paths)")
        axs.tick_params(axis='x', rotation=90)
        axs.set_ylabel("Occurrences")
        
    if finished:
        # Plot the end list occurrences
        axs[1].bar(end_sorted.keys(), end_sorted.values(), color="salmon")
        axs[1].set_title("End Country Occurrences" + " (Finished Paths)" if finished else "Start Country Occurrences" + " (Unfinished Paths)")
        axs[1].tick_params(axis='x', rotation=90)
        axs[1].set_ylabel("Occurrences")

    plt.suptitle(f"Figure {fig_num}")
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    


def plot_regression_clicks_links(x, y, ax, position, x_label, y_label, title, intercept, slope, r):
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
    #sns.regplot(data=data, x=x, y=y, ax=ax, order=1, line_kws={'color':'red'}, robust=True) # downweight the influence of outliers(1 outlier on the plot!)
    ax.scatter(x=x, y=y)
    ax.plot(x, intercept + slope * x, color="r")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Compute the regression
    #slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                        #y=plot.get_lines()[0].get_ydata())
    # Add regression equation to the plot
    ax.text(position[0], position[1], f"y = {round(intercept, 2)} + {round(slope, 2)} * x \n R-squared = {round(r, 2)}")


def top_player_vs_pagerank_article_frequencies(df_player_frequencies, df_pagerank, k=40):
    df_player_frequencies['type'] = 'Player'
    df_pagerank['type'] = 'Pagerank'
    rank_v_freq = pd.concat([df_player_frequencies, df_pagerank], ignore_index=True)
    rank_v_freq.sort_values(by=['type', 'rank'], ignore_index=True, inplace=True, ascending=[True, False])
    # Get k article names with highest pagerank, we will plot that
    first_k_article_names = df_pagerank.sort_values(by='rank', ascending=False).head(k).article_name
    truncated_r_v_f = rank_v_freq[rank_v_freq.article_name.isin(first_k_article_names.values)].reset_index(drop=True)
    plt.figure(figsize=(8, 14), dpi=80)
    plt.title(f'Player vs PageRank for top {k} PageRank articles')
    sns.barplot(truncated_r_v_f, y='article_name', x='rank', hue='type', orient='y')


def top_player_vs_pagerank_country_frequencies(df_player_frequencies, df_pagerank, k=40):
    df_pagerank = df_pagerank[['country_name', 'rank']].groupby(['country_name'], as_index=False).sum()
    df_player_frequencies = df_player_frequencies[['country_name', 'rank']].groupby(['country_name'], as_index=False).sum()

    df_player_frequencies['type'] = 'Player'
    df_pagerank['type'] = 'Pagerank'
    rank_v_freq = pd.concat([df_player_frequencies, df_pagerank], ignore_index=True)
    rank_v_freq.sort_values(by=['type', 'rank'], ignore_index=True, inplace=True, ascending=[True, False])
    # Get k article names with highest pagerank, we will plot that
    first_k_country_names = df_pagerank.sort_values(by='rank', ascending=False).head(k).country_name
    truncated_r_v_f = rank_v_freq[rank_v_freq.country_name.isin(first_k_country_names.values)].reset_index(drop=True)
    plt.figure(figsize=(8, 14), dpi=80)
    plt.title(f'Player vs PageRank for top {k} PageRank countries')
    sns.barplot(truncated_r_v_f, y='country_name', x='rank', hue='type', orient='y')
