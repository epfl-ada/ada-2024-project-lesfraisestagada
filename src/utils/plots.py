import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from ast import literal_eval
from collections import Counter
from src.scripts.article_to_country import *
from itertools import combinations

import plotly.io as pio
import plotly.graph_objects as go


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
    df_player_frequencies['type'] = 'Player'
    df_pagerank['type'] = 'Pagerank'
    rank_v_freq = pd.concat([df_player_frequencies, df_pagerank], ignore_index=True)

    # Get k article names with highest pagerank, we will plot that
    first_k_country_names = df_pagerank.sort_values(by='rank', ascending=False).head(k).country_name
    
    # Sort top make top pagerank appear first
    def sorter(column):
        """Sort function"""
        correspondence = {country: order for order, country in enumerate(first_k_country_names)}
        return column.map(correspondence)
    
    rank_v_freq.sort_values(by="country_name", key=sorter, ignore_index=True, inplace=True)
    truncated_r_v_f = rank_v_freq[rank_v_freq.country_name.isin(first_k_country_names.values)].reset_index(drop=True)
    pagerank_truncated = truncated_r_v_f[truncated_r_v_f['type'] == 'Pagerank']
    player_truncated = truncated_r_v_f[truncated_r_v_f['type'] == 'Player']

    frames = [
        go.Frame(
            data = go.Bar(
                x=pagerank_truncated["rank"],
                y=pagerank_truncated["country_name"],
                orientation="h",
                name="PageRank",
                marker=dict(
                    color="steelblue",
                )
            ),
            name = 'pagerank'
        ),
        go.Frame(
            data = go.Bar(
                x=player_truncated["rank"],
                y=player_truncated["country_name"],
                orientation="h",
                name="Player",
                marker=dict(
                    color="goldenrod",
                )
            ),
            name='player'
        ),
        go.Frame(
            data = go.Bar(
                x=player_truncated["rank"].reset_index(drop=True) - pagerank_truncated["rank"].reset_index(drop=True),
                y=player_truncated["country_name"],
                orientation="h",
                name="Difference",
                marker=dict(
                    color="olivedrab",
                )
            ),
            name='diff'
        )
    ]

    layout = go.Layout(
        xaxis=dict(title=f"Player vs PageRank for top {k} PageRank countries", range=[-0.02, 0.15], autorange=False),
        yaxis=dict(title="Country"),
        updatemenus=[
            dict(
                type="buttons",
                direction = "left",
                x=0.5,
                xanchor="left",
                y=1.2,
                yanchor="top",
                pad={"l": -120},
                showactive=True,
                buttons=[
                    dict(
                        label="PageRank",
                        method="animate",
                        args=[['pagerank']]
                    ),
                    dict(
                        label="Player",
                        method="animate",
                        args=[['player']]
                    ),
                    dict(
                        label="Difference",
                        method="animate",
                        args=[['diff']]
                    )
                ]
            )
        ],
    )

    fig = go.Figure(
        data=frames[0].data,
        layout=layout,
        frames=frames
    )

    return fig

def top_bottom_rank_diff(rank_v_freq_countries, k=40):

    rank_v_freq_countries = rank_v_freq_countries.dropna().sort_values(by='rank_diff', ascending=False)

    top = rank_v_freq_countries.head(k)
    bottom = rank_v_freq_countries.tail(k).sort_values(by='rank_diff', ascending=True)

    frames = [
        go.Frame(
            data = go.Bar(
                x=top["rank_diff"],
                y=top["country_name"],
                orientation="h",
                name=f"Top {k}",
                marker=dict(
                    color="olivedrab",
                )
            ),
            name = 'top'
        ),
        go.Frame(
            data = go.Bar(
                x=bottom["rank_diff"],
                y=bottom["country_name"],
                orientation="h",
                name=f"Bottom {k}",
                marker=dict(
                    color="olivedrab",
                )
            ),
            name='bottom'
        )
    ]

    layout = go.Layout(
        xaxis=dict(title=f"Top and bottom {k} countries with respect to rank difference", range=[-0.02, 0.04], autorange=False),
        yaxis=dict(title="Country"),
        updatemenus=[
            dict(
                type="buttons",
                direction = "left",
                x=0.5,
                xanchor="left",
                y=1.2,
                yanchor="top",
                pad={"l": -80},
                showactive=True,
                buttons=[
                    dict(
                        label=f"Top {k}",
                        method="animate",
                        args=[['top']]
                    ),
                    dict(
                        label=f"Bottom {k}",
                        method="animate",
                        args=[['bottom']]
                    ),
                ]
            )
        ],
    )

    fig = go.Figure(
        data=frames[0].data,
        layout=layout,
        frames=frames
    )

    return fig



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


def start_end_countries(finished_paths, country_clicks):
    """Plot the distribution of source and target articles

    Args:
        finished_paths (dataframe)
        country_clicks (dataframe)
    """

    country_clicks.dropna(subset=["Top_1_name"], inplace=True)

    start = []
    end = []
    for idx, row in finished_paths.iterrows():
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
    
    start_sorted = sorted(start_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    start_sorted = pd.DataFrame(start_sorted, 
                                columns=["country", "count"])
    
    end_sorted = sorted(end_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    end_sorted = pd.DataFrame(end_sorted, 
                              columns=["country", "count"])

    # Generate a custom color map for each country
    all_countries = list(set(start_sorted["country"]).union(set(end_sorted["country"])))

    # Use a colormap (e.g., from Matplotlib) to assign unique colors
    cmap = plt.cm.get_cmap("tab20", len(all_countries)) 
    country_colors = {country: f"rgba({cmap(i)[0]*255:.0f}, {cmap(i)[1]*255:.0f}, {cmap(i)[2]*255:.0f}, {cmap(i)[3]:.2f})" 
                    for i, country in enumerate(all_countries)}

    # create bar plot for source countries
    trace_start = go.Bar(x=start_sorted["country"], 
                        y=start_sorted["count"],
                            name="Occurrence in source",
                            marker_color=[country_colors[country] for country in start_sorted["country"]],
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>Count:</b> %{y}<br>" 
                                            "<extra></extra>")    
                            )

    trace_end = go.Bar(x=end_sorted["country"], 
                        y=end_sorted["count"],
                            name="Occurrence in target",
                            marker_color=[country_colors[country] for country in end_sorted["country"]], 
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>Count:</b> %{y}<br>" 
                                            "<extra></extra>")    
                            )

    # create figures
    fig = go.Figure()

    # add traces
    fig.add_trace(trace_start)
    fig.add_trace(trace_end.update(visible=False))

    y_max = max(start_sorted["count"].max(), end_sorted["count"].max())
    # Create buttons to toggle between the two bar charts
    fig.update_layout(
        yaxis=dict(range=[0, y_max+200]),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Source countries",
                        method="update",
                        args=[{"visible": [True, False]}],  # Show first chart, hide second
                    ),
                    dict(
                        label="Target countries",
                        method="update",
                        args=[{"visible": [False, True]}],  # Hide first chart, show second
                    ),
                ],
                pad={"r": 0, "l":0, "t": 20, "b":0},
                showactive=True,
                x=0.,
                xanchor="left",
                y=1.1,
                yanchor="middle"
            ),
        ]
    )

    # Show the figure
    fig.show()

    # Export the figure to an HTML file
    pio.write_html(fig, file='graphs/topic_1/start_end_countries.html', auto_open=False)



def show_country_assignments(write = False):
    """Show the proportion of articles assigned to a country by different classification methods.

    Args:
        write (bool, optional): If the new plot should be written to disk. Defaults to False.

    Returns:
        results, data: dictionnary containing the results of the classification methods and the dataframes of the different classification methods
    """
    results = {}
    data = {}
    country_data = pd.read_csv('data/country_data.csv', index_col=0)
    counts = filter_top_k(country_data, k=2, N=1)
    total_number_of_articles = len(counts)

    nan_df = counts[counts.isna().all(axis=1)]
    print(f"Number of articles with no countries before completion with llama: {len(nan_df)}")
    results["Text search"] = (total_number_of_articles - len(nan_df)) / total_number_of_articles * 100

    refined_data = pd.read_csv("data/country_occurences.csv", index_col=0)
    nan_df = refined_data[refined_data.isna().all(axis=1)]
    print(f"Number of articles with no countries after completion with naive + llama: {len(nan_df)}")
    results["Text search + missing articles classified with LlaMa"] = (total_number_of_articles - len(nan_df)) / total_number_of_articles * 100

    qwen_country_data = pd.read_csv('data/country_data_full_qwen.csv', index_col=0)
    qwen_missing = len(qwen_country_data[qwen_country_data["Top_1_name"].isna()])
    print(f"Number of articles with no countries after completion with Qwen: {qwen_missing}")
    results["Full classification with Qwen"] = (total_number_of_articles - qwen_missing) / total_number_of_articles * 100


    llama_country_data = pd.read_csv('data/country_data_full_llama.csv', index_col=0)
    llama_missing = len(llama_country_data[llama_country_data["Top_1_name"].isna()])
    print(f"Number of articles with no countries after completion with LLAMA: {llama_missing}")
    results["Full classification with LLaMa"] = (total_number_of_articles - llama_missing) / total_number_of_articles * 100

    llama_country_data_imporved_normal = pd.read_csv('data/country_data_full_llama_improved_normal.csv', index_col=0)
    llama_country_data_imporved_reversed = pd.read_csv('data/country_data_full_llama_improved_reversed.csv', index_col=0)

    llama_country_data_imporved = pd.read_csv('data/country_data_full_llama_improved_reversed.csv', index_col=0)

    llama_country_data_imporved_normal["Top_1_name"] = llama_country_data_imporved_normal["Top_1_name"].where(llama_country_data_imporved_normal['Top_1_name'] == llama_country_data_imporved_reversed['Top_1_name'], np.nan)
    llama_country_data_imporved = llama_country_data_imporved_normal
    llama_missing_improved = len(llama_country_data_imporved[llama_country_data_imporved["Top_1_name"].isna()])
    print(f"Number of articles with no countries after completion with LLAMA: {llama_missing_improved}")
    results["Improved classification with LLaMa"] = (total_number_of_articles - llama_missing_improved) / total_number_of_articles * 100
    llama_country_data_imporved.drop(columns=["Predictions"], inplace=True)
    llama_country_data_imporved.to_csv('data/country_data_full_llama_improved.csv')

    fig = px.bar(x=results.keys(), y=results.values(), color=results.keys(), labels={"x": "Classification Method", "y": "% of articles assigned to a country"})
    fig.update_layout(title="Proportion of articles assigned to a country") #legend is not shown since we want to write it directly on the website in HTML for better readability
    fig.update_xaxes(showticklabels=False)  # Remove x-axis tick labels
    fig.update_yaxes(range=[0, 100])
    fig.show()
    
    if write:
        fig.write_html("./graphs/preprocessing/proportion_country_assignment.html")
    
    data["Text search"] = counts
    data["Text search + missing articles classified with LlaMa"] = refined_data
    data["Full classification with Qwen"] = qwen_country_data
    data["Full classification with LLaMa"] = llama_country_data
    data["Improved classification with LLaMa"] = llama_country_data_imporved
    
    return results, data


def show_overlap_heatmap(data, write=False):
    """Show the overlap between classification methods as a heatmap.

    Args:
        data (dictionnary): dictionnary containing the dataframes of the different classification methods
        write (bool, optional): If the new plot should be written to disk. Defaults to False.
    """
    
    models = {
        "Text search": data["Text search"],
        "Text search + missing articles classified with LlaMa": data["Text search + missing articles classified with LlaMa"],
        "Full classification with Qwen": data["Full classification with Qwen"],
        "Full classification with LLaMa": data["Full classification with LLaMa"],
        "Improved classification with LLaMa": data["Improved classification with LLaMa"],
    }

    heatmap_data = pd.DataFrame(index=models.keys(), columns=models.keys())

    for model in models.keys():
        heatmap_data.loc[model, model] = 1.0

    for model1, model2 in combinations(models.keys(), 2):
        df_1 = models[model1]
        df_2 = models[model2]
        overlap = (df_1["Top_1_name"] == df_2["Top_1_name"]).sum() / len(data["Text search"])
        heatmap_data.loc[model2, model1] = overlap.round(2)

    heatmap_data = heatmap_data.astype(float)

    # Replace row and column names with indices
    model_names = list(models.keys())
    index_to_name = {i: name for i, name in enumerate(model_names)}  # Create legend mapping

    # Create heatmap with indices
    heatmap_data.index = range(len(model_names))
    heatmap_data.columns = range(len(model_names))

    # Generate the heatmap
    fig = px.imshow(
        heatmap_data.values,
        labels={"x": "Index", "y": "Index", "color": "Overlap"},
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Blues",
        text_auto=True,
    )

    fig.update_layout(
        title="Overlap percentage between classification methods",
        xaxis_title="Classification Method Index",
        yaxis_title="Classification Method Index",
        coloraxis_colorbar=dict(title="Overlap"),
    )

    fig.update_xaxes(dtick=1)  # Force x-axis to show every tick
    fig.update_yaxes(dtick=1)  # Force y-axis to show every tick
    fig.update_layout(coloraxis_showscale=False)  # Hide the color scale

    # Display legend associating indices to model names
    legend_text = "<br>".join([f"{i}: {name}" for i, name in index_to_name.items()])
    fig.add_annotation(
        text=f"<b>Legend:</b><br>{legend_text}",
        xref="paper", yref="paper",
        x=0.0, y=0.5,
        showarrow=False,
        align="left",
        font=dict(size=12)
    )

    fig.show()
    if write:
        fig.write_html("./graphs/preprocessing/overlap_heatmap.html")



def show_agreement_plot(data, agreement_df, write = False):
    """Show the agreement between annotators and classification methods as a bar plot.

    Args:
        data (dictionnary): dictionnary containing the dataframes of the different classification methods
        agreement_df (dataframe): dataframe containing the agreement between annotators so that it can be compared to the classification methods
        write (bool, optional): If the new plot should be written to disk. Defaults to False.
    """
    models = {
        "Text search": data["Text search"],
        "Text search + missing articles classified with LlaMa": data["Text search + missing articles classified with LlaMa"],
        "Full classification with Qwen": data["Full classification with Qwen"],
        "Full classification with LLaMa": data["Full classification with LLaMa"],
        "Improved classification with LLaMa": data["Improved classification with LLaMa"],
    }

    values = []
    for model in models.keys():
        agreement_value = (models[model].loc[agreement_df.index]["Top_1_name"].str.lower().fillna("nan") == agreement_df["country"].str.lower().fillna("nan")).sum() / len(agreement_df) * 100
        values.append(agreement_value)
        
    fig = px.bar(
        x=models.keys(), 
        y=values, 
        color=models.keys(),
        title="Agreement between annotators and classification method in %", 
        labels={"x": "Classification Method", "y": "Agreement value in %"}
        )
    fig.update_xaxes(showticklabels=False)  # Remove x-axis tick labels

    fig.update_yaxes(range=[0, 100])
    fig.show()

    if write:
        fig.write_html("./graphs/preprocessing/agreement_bar_plot.html")