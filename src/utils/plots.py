import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
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
    plt.title("Figure 1: Top 100 country links by occurences")
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
    plt.title(f'Figure 1: Player vs PageRank for top {k} PageRank articles')
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
    plt.title(f'Figure 2: Player vs PageRank for top {k} PageRank countries')
    sns.barplot(truncated_r_v_f, y='country_name', x='rank', hue='type', orient='y')



def plot_top_k_unique_failure_success_counts(df_articles_count, k=10):
    """
    Display two plots with the top k articles by unique success and failure counts.
    Args:
        df_articles_count (pd.DataFrame): DataFrame containing the article counts and success/failure ratios
        k (int): number of top articles to display. Default is 10.
    """
    
    # set the style theme for better aesthetics :)
    sns.set_theme(style="whitegrid")

    # color palette
    success_palette = sns.light_palette("green", as_cmap=False, n_colors=k)
    failure_palette = sns.light_palette("red", as_cmap=False, n_colors=k)

    # plt 1: top 10 articles by unique success counts
    top_success_counts = df_articles_count.nlargest(k, 'unique_success_count')[::-1]
    plt.figure(figsize=(k, int(4/5 * k)))
    sns.barplot(
        data=top_success_counts,
        x='unique_success_count',
        y='article',
        palette=success_palette
    )
    plt.title(f"Figure 1: Top {k} Articles by Unique Success Counts", fontsize=16, weight='bold')
    plt.xlabel("Unique Success Count", fontsize=14)
    plt.ylabel("Article", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # plt 2: top 10 articles by unique failure counts
    top_failure_counts = df_articles_count.nlargest(k, 'unique_failure_count')[::-1]
    plt.figure(figsize=(k, int(4/5 * k)))
    sns.barplot(
        data=top_failure_counts,
        x='unique_failure_count',
        y='article',
        palette=failure_palette
    )
    plt.title(f"Figure 2: Top {k} Articles by Unique Failure Counts", fontsize=16, weight='bold')
    plt.xlabel("Unique Failure Count", fontsize=14)
    plt.ylabel("Article", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def analyze_top_articles_by_category(df_articles_count, categories, top_n=15):
    """
    Analyzes the top N most clicked articles by merging with categories and plotting success/failure ratios.

    Args:
        df_articles_count (pd.DataFrame): DataFrame containing article click counts and success/failure ratios.
        categories (pd.DataFrame): DataFrame containing article-category mapping with 'articles' and 'category' columns.
        top_n (int): Number of top articles to analyze based on total click count (default is 15).
    """
    # keep only the top N most clicked articles
    top_articles = df_articles_count.nlargest(top_n, 'total_click_count')
    
    # merge with categories df to add category information
    df_art_and_cat = top_articles.merge(categories, left_on='article', right_on='articles', how='left')
    
    # drop the extra 'articles' column resulting from the merge
    df_art_and_cat = df_art_and_cat.drop(columns=['articles'])
    
    # group by category and aggregate to calculate average success and failure ratios
    category_grouped = df_art_and_cat.groupby('category').agg({
        'success_ratio_unique': 'mean',
        'failure_ratio_unique': 'mean'
    }).reset_index()
    
    # scatter plot of average success vs. failure ratios by category
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=category_grouped, x='success_ratio_unique', y='failure_ratio_unique', 
                    hue='category', 
                    legend=True, 
                    s=200,
                    alpha=0.7,         
                    linewidth=0.5)
    plt.xlabel("Average Unique Success Ratio", fontsize=10)
    plt.xticks(fontsize=8)
    plt.ylabel("Average Unique Failure Ratio", fontsize=10)
    plt.yticks(fontsize=8)
    plt.title("Figure 3: Correlation between Article Categories and Success/Failure Ratios")
    plt.show()
    
    # heatmap of success ratio for top articles by category
    plt.figure(figsize=(10, 8))
    heatmap_df = df_art_and_cat.pivot(index='article', columns='category', values='success_ratio_unique')
    sns.heatmap(heatmap_df, cmap="RdYlGn", center=0.5, linewidths=0.1)
    plt.title(f"Figure 4: Top {top_n} Articles Heatmap by Categories and Success Ratio")
    plt.xlabel("Category", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel("Article", fontsize=10)
    plt.show()


def plot_top_dead_end_countries(unique_dead_end_countries, top_n=10, use_click_count=False, use_scaled=False, i=""):
    """
    Plots the top country-related dead-end articles by click count or scaled click count, with a gradient for mean links out.
    
    Args:
        unique_dead_end_countries (pd.DataFrame): DataFrame containing dead-end country-related articles with click counts and link information.
        top_n (int): Number of top articles to display (default is 10).
        use_click_count (bool): Whether to plot based on click count or mean failure ratio (default is False).
        use_scaled (bool): Whether to use the scaled click count or the original click count if use_click_count is True.
        i (int): number of the figure for the title
    """
    # choose the column to sort by based on the parameters
    if use_click_count:
        x_col = "scaled_click_count" if use_scaled else "click_count"
        title_click_type = "Click Count (Scaled by Outgoing Links)" if use_scaled else "Click Count"
    else:
        x_col = "mean_failure_ratio_unique"
        title_click_type = "Mean Failure Ratio"
    
    # select the top N unique dead-end countries based on the chosen metric
    if use_click_count:
        top_unique_dead_end_countries = unique_dead_end_countries.sort_values(by=[x_col, 'mean_failure_ratio_unique'], ascending=False).head(top_n)
    else:
        top_unique_dead_end_countries = unique_dead_end_countries.sort_values(by=['mean_failure_ratio_unique'], ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(
        data=top_unique_dead_end_countries,
        x=x_col,
        y='Top_1_name',
        hue='sum_num_links_out',  
        dodge=False, 
        palette="YlGnBu"  # Gradient palette
    )

    # adjust the color bar for the gradient
    norm = plt.Normalize(top_unique_dead_end_countries['sum_num_links_out'].min(), 
                         top_unique_dead_end_countries['sum_num_links_out'].max())
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])

    # add the color bar on the side
    cbar = barplot.figure.colorbar(sm)
    cbar.set_label("Sum Links Out")

    # title and labels
    plt.title(f"Figure {i}: Top {top_n} Country-Related Dead-End Articles by {title_click_type}")
    plt.xlabel(title_click_type)
    plt.ylabel("Country")
    plt.tight_layout()

    plt.show()


def plot_top_last_dead_end_countries(last_dead_end_countries, top_n=10, use_scaled=False, i=""):
    """
    Plots the top N last countries in unfinished paths that are also dead-end countries,
    using either the raw count or a scaled count based on sum of outgoing links.

    Args:
        last_dead_end_countries (pd.DataFrame): DataFrame containing last articles in unfinished paths
                                                that match dead-end countries, with count data.
        top_n (int): Number of top last countries to display (default is 10).
        use_scaled (bool): Whether to use the scaled count or the raw count (default is False).
    """
    # select the column to plot based on the use_scaled flag
    count_column = 'scaled_count' if use_scaled else 'count'
    title_suffix = " (Scaled by Outgoing Links)" if use_scaled else ""
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=last_dead_end_countries.sort_values(by=count_column, ascending=False).head(top_n), 
        x=count_column, 
        y='last_article', 
        palette='viridis'
    )
    plt.xlabel("Occurrences in Unfinished Paths" + title_suffix)
    plt.ylabel("Last Country (Dead End)")
    plt.title(f"Figure {i}: Top {top_n} Last Countries in Unfinished Paths Likely to Be Dead Ends" + title_suffix)
    plt.tight_layout()
    plt.show()


def plot_articles_before_go_back(before_last_article_analysis, use_scaled=False, top_n=10, i=""):
    """
    Plots the articles that appear immediately before players backtrack ("<") in unfinished paths.
    Displays a bar chart of either the raw counts or the scaled counts based on outgoing links.

    Args:
        before_last_article_analysis (pd.DataFrame): DataFrame containing articles before backtracking with related country data.
        use_scaled (bool): Whether to plot the scaled count by outgoing links (default is False).
        top_n (int): Number of top articles to display (default is 10).
    """
    # calculate scaled counts based on outgoing links if not already present
    before_last_article_analysis['scaled_count'] = (
        before_last_article_analysis['count'] / before_last_article_analysis['sum_num_links_out']
    )

    # choose the appropriate column for x-axis and title based on use_scaled parameter
    x_col = 'scaled_count' if use_scaled else 'count'
    title_suffix = " (Scaled by Outgoing Links)" if use_scaled else ""
    title = f"Figure {i}: Top {top_n} Articles Before Backtracking in Unfinished Paths{title_suffix}"
    xlabel = "Scaled Backtrack Occurrences (by Outgoing Links)" if use_scaled else "Backtrack Occurrences Count"


    # sort by the chosen column and select top N articles
    top_before_back = before_last_article_analysis.sort_values(by=x_col, ascending=False).head(top_n)

  
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_before_back,
        x=x_col,
        y='article_before_back',
        palette='viridis' if not use_scaled else 'mako'
    )
    plt.xlabel(xlabel)
    plt.ylabel("Article Before '<'")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_color_map(
        df: pd.DataFrame,
        value_column: str,
        title: str
    ):
    """
    Plot a choropleth map of the number of publications per country.
    Args:
        publications_per_country_df (pd.DataFrame): DataFrame containing at least two columns: {value_column} and country
    """
    fig = px.choropleth(
    df,
    locations="country",  # match on country names
    locationmode="country names",  
    color=value_column,  
    color_continuous_scale="Plasma",
    title=title,
    hover_name="country", 
    hover_data={
        value_column: ":,.0f", 
        "country": False
    }
)

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        title=dict(
            text=title,
            font=dict(
                family="Helvetica Neue, sans-serif", 
                size=18 
            ),
            x=0.5,  # center title
            y=0.95  # lower title on canvas
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.update_traces(
        hoverlabel=dict(
            bgcolor="white", 
            font_size=12, 
            font_family="Helvetica Neue"
        )
    )

    return fig