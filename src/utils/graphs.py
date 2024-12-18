# here we find all functions that are needed in order to reproduce the graphs of our project. 

# Imports 
import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pyvis.network import Network
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

import folium

import plotly.io as pio

from tqdm import tqdm
import time

import webbrowser


# Functions
def color_gradient(variable, color):
    """Define a gradient color palette for values of a variable

    Args:
        variable (list): variable for which the color gradient is defined
        color (_type_): base color of the color palette
        modif (int): parameter to lower contrast between highest and lowest value

    Return: 
        List of colors, one element per entry of variable
    """
    # define a global color map
    color_map = plt.get_cmap(color)

    # Define a color gradient to represent the number of clicks for each article
    norm = matplotlib.colors.Normalize(vmin=min(variable), vmax=max(variable))
    colors_hex = [matplotlib.colors.to_hex(color_map(norm(element))) for element in variable]

    return colors_hex



def draw_circle_graph(df, out_path, project_path):
    """Draw a circular directed graph where nodes are articles
    Edges represent connections between articles. 
    Article A is said to be connected to article B if article A contains a link that is pointing to article B. 

    Args:
        df (dataframe): containing the following variables = click_count, num_links_in, num_links_out and having as an index the name of articles
        out_path (str): path where the graph is saved
        project_path (str): path of the current project
    """
    # define the color palette
    colors_hex = color_gradient(df.click_count, 'Reds')

    net = Network(directed=True, 
                notebook=True, 
                font_color='#10000000', 
                cdn_resources='in_line')

    # Turn off physics so nodes stay fixed
    net.barnes_hut(gravity=-10000,  # Controls the strength of repulsion between nodes
                central_gravity=0.01,  # Weak central gravity so nodes spread out
                spring_length=300000,  # Increase distance between connected nodes
                spring_strength=0.05)  # Adjust spring tightness
    net.toggle_physics(False)

    # write the title of each node
    titles = [f"article: {df.index[i]} \n click count: {df.click_count.iloc[i]} \n in degree: {df.num_links_in.iloc[i]} \n out degree: {df.num_links_out.iloc[i]}" for i in range(len(df))]

    # define position of nodes 
    num_nodes = len(df)

    # Define a circular layout for the nodes
    radius = 500  # Adjust the radius of the circle
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)  # Evenly spaced angles
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)

    net.add_nodes(df.index.tolist(), 
                title=titles, 
                color=colors_hex, 
                x=x_positions, 
                y=y_positions)

        
    # Let's add edges between articles that are connected in Wikipedia
    for i, article1 in enumerate(df.index.tolist()):
        name_links_out = df.name_links_out.iloc[i]

        if pd.notna(name_links_out):
            for article2 in df.index.tolist(): 
                if article2 in name_links_out:
                    net.add_edge(article1, article2)

    # save to html for visualization on our website
    net.show(out_path)

    print(f"Map is saved in {out_path}!")

    # Open the graph in browser
    webbrowser.open(os.path.join(project_path, out_path))



def get_country_coordinates(country_name):
    """Get geographical coordinated of a country

    Args:
        country_name (str): name of the country
    
    Return: 
        (latitude, longitude)
    """
     # Get a set of coordinates (latitude, longitude) for each country for visualisation purposes
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(country_name, timeout=None)
    return (location.latitude, location.longitude)



def geolocalization(df):
    """Geolocalizes all countries of df

    Args:
        df (dataframe): contains a variable Top_1_name containing country names

    Returns:
        lists: 2 lists, first latitudes and then longitudes corresponding to the countries of df
    """

    countries = df.Top_1_name.tolist()
    coords = []
    cache = {}  # Dictionary to cache country coordinates

    for country in tqdm(countries, desc="Processing countries", unit="country"): 
        time.sleep(1)  # Add a delay to respect API rate limits
        if country in cache:  # Use cached coordinates if available
            coords.append(cache[country])
            continue

        # handle ambiguous cases (the geolocator function tends to place ambiguous countries in the US!)
        if country == "sudan": 
            result = get_country_coordinates("Sudan, Africa")
        elif country == "georgia": 
            result = get_country_coordinates("Georgia, Caucasus")
        elif country == "lebanon": 
            result = get_country_coordinates("Lebanon, Asia")
        elif country == "greenland": 
            result = get_country_coordinates("Greenland, Kingdom of Denmark")
        elif country == "jordan": 
            result = get_country_coordinates("Jordan, Middle East")
        elif country == "armenia": 
            result = get_country_coordinates("Armenia, Caucasus")
        elif country == "montenegro": 
            result = get_country_coordinates("Montenegro, Europe")
        elif country == "albania": 
            result = get_country_coordinates("Albania, Europe")

        else:
            result = get_country_coordinates(country)

        coords.append(result)
        cache[country] = result  # Cache the result for reuse

    latitudes = [coord[0] for coord in coords]
    longitudes = [coord[1] for coord in coords]

    return latitudes, longitudes



def overlap_world_map(df, country_connections, latitudes, longitudes, out_path, edge=True):
    """Constructs a graph with a world map as background 

    Args:
        df (dataframe): data containing an occurrence variable that represents the number of articles that are associated with a country
        country_connections (dataframe): data with 3 columns, start_country, end_country and count
        latitudes (list): latitudes of all countries
        longitudes (list): longitudes of all countries
        out_path (str)
        edge (bool, optional): whether to display edges on the graph. Defaults to True.
    """
    # Create a base map centered on an average coordinate (e.g., latitude 0, longitude 0)
    map_center = [0, 0]
    world_map = folium.Map(location=map_center, zoom_start=0.5, tiles='cartodbpositron')

     # define a global color map
    color_map = plt.get_cmap('Reds')
    colors_hex = [matplotlib.colors.to_hex(color_map(element + 100)) for element in df.occurrence.tolist()]

    countries = df.Top_1_name.tolist()

    # Add a slider for threshold
    slider_html = '''
        <div style="position: absolute; top: 10px; left: 100px; z-index: 9999;">
            <label for="threshold_slider" style="font-size: 14px; color: black;">Edge occurrence:</label>
            <input type="range" id="threshold_slider" min="1" max="611" value="1" step="10" style="width: 200px;">
            <span id="threshold_value" style="font-size: 14px; color: black;">1</span>
        </div>
    '''
    
    # Add the slider to the map as an HTML element
    world_map.get_root().html.add_child(folium.Element(slider_html))

    # JavaScript to update edges based on the slider value
    slider_js = '''
        <script>
            document.getElementById("threshold_slider").oninput = function() {
                var threshold = parseInt(this.value);
                document.getElementById("threshold_value").innerText = threshold;
                // Filter edges based on the threshold value
                var lines = document.querySelectorAll('path[stroke="blue"]');
                lines.forEach(function(line) {
                    var opacity = parseFloat(line.getAttribute("stroke-opacity"));
                    if (opacity >= (threshold / 1006)) {
                        line.style.display = "block";  // Show the edge
                    } else {
                        line.style.display = "none";   // Hide the edge
                    }
                });
            };
        </script>
    '''
    
    # Add the JavaScript to the map
    world_map.get_root().html.add_child(folium.Element(slider_js))

    # add edges
    if edge:
        for i in range(len(country_connections)):
            start_country = country_connections["start_country"].iloc[i]
            end_country = country_connections["end_country"].iloc[i]
            count = country_connections["count"].iloc[i]
            
            # Get coordinates for both countries
            try:
                start_idx = countries.index(start_country)
                end_idx = countries.index(end_country)
                start_coords = [latitudes[start_idx], longitudes[start_idx]]
                end_coords = [latitudes[end_idx], longitudes[end_idx]]
                
                # Add a line (edge) between the two countries
                folium.PolyLine(
                    locations=[start_coords, end_coords], 
                    color='blue',
                    weight=2, 
                    opacity=count/1006,   # Line transparency
                ).add_to(world_map)
            except ValueError:
                print(f"Connection skipped: {start_country} -> {end_country} (one of them not found in countries list)")


    # Add each country node as a CircleMarker with scaled sizes
    for country, lat, lon, color, size in zip(countries, latitudes, longitudes, colors_hex, df.occurrence.tolist()):
        folium.CircleMarker(
            location=[lat, lon],  # Use latitude and longitude
            radius=size/10,         # Scaled size based on occurrence
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{country}</b><br>{size} articles",
                max_width=100, 
                min_width=50
            )
        ).add_to(world_map)

    # Save the combined map to an HTML file
    world_map.save(out_path)

    print(f"Map is saved in {out_path}!")



def plot_node_degrees(df):
    """Plot Figure 2 of the website, 2 stakced bar plots representing the in and out degrees of countries

    Args:
        df (dataframe): data containing a occurrence column 
    """

    sorted_data_countries = df.sort_values(by='occurrence', ascending=False).iloc[:8:,]
    # create bar plot for most occuring countries
    trace_most_in = go.Bar(x=sorted_data_countries["Top_1_name"], 
                        y=sorted_data_countries["num_links_in"],
                            name="in degree",
                            marker_color="blue", 
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>In degree:</b> %{y}<br>" 
                                            "<extra></extra>")    
                            )

    trace_most_out = go.Bar(x=sorted_data_countries["Top_1_name"], 
                        y=sorted_data_countries["num_links_out"],
                            name="out degree",
                            marker_color="red", 
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>Out degree:</b> %{y}<br>" 
                                            "<extra></extra>")    
                            )


    # create bar plot for least occuring countries
    sorted_data_countries = df.sort_values(by='occurrence', ascending=True).iloc[:8:,]

    trace_least_in = go.Bar(x=sorted_data_countries["Top_1_name"], 
                        y=sorted_data_countries["num_links_in"],
                            name="in degree",
                            marker_color="blue", 
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>In degree:</b> %{y}<br>" 
                                            "<extra></extra>") )

    trace_least_out = go.Bar(x=sorted_data_countries["Top_1_name"], 
                        y=sorted_data_countries["num_links_out"],
                            name="out degree",
                            marker_color="red", 
                            hovertemplate=( "<b>Country:</b> %{x}<br>" 
                                            "<b>Out degree:</b> %{y}<br>" 
                                            "<extra></extra>") )

    # create figures
    fig = go.Figure()

    # add traces
    fig.add_trace(trace_most_in)
    fig.add_trace(trace_most_out)
    fig.add_trace(trace_least_in.update(visible=False))
    fig.add_trace(trace_least_out.update(visible=False))


    # Create buttons to toggle between the two bar charts
    fig.update_layout(
        barmode='stack',
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Most occurring countries",
                        method="update",
                        args=[{"visible": [True, True, False, False]}],  # Show first chart, hide second
                    ),
                    dict(
                        label="Least occurring countries",
                        method="update",
                        args=[{"visible": [False, False, True, True]}],  # Hide first chart, show second
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
    pio.write_html(fig, file='graphs/topic_1/bar_plot_distribution_of_degrees.html', auto_open=False)



def overlap_world_map_clicks_before(out_path, clicks, all_pairs_countries_normalized, countries, latitudes,longitudes):
    """
    World map of the click count per country and game path between countries before scaling
    Args:
        out_path (path): define the name and path of the output map 
        clicks (list): list of the click count per country
        all_pairs_countries_normalized (df): Pandas DataFrame of the number of time that a path between two articles is taken by players
        countries (list): list of the name of all the countries
        latitudes (list): first geographical coordinate of each country
        longitudes (list): second geographical coordinate of each country
    """
    # Create a base map centered on (0,0)
    map_center = [0, 0]
    world_map = folium.Map(location=map_center, zoom_start=0.5, tiles='cartodbpositron')

    # define size gradient proportional to the click count 
    size_scaler = MinMaxScaler(feature_range=(0.0001, max(clicks)))
    node_sizes = size_scaler.fit_transform([[count] for count in clicks]).flatten()

    # define color gradient for node color
    color_scaler = MinMaxScaler(feature_range=(0.25, 1.5))
    normalized_counts = color_scaler.fit_transform([[count] for count in clicks]).flatten()
    color_map = plt.cm.get_cmap('Purples')
    colors_hex_before = [matplotlib.colors.to_hex(color_map(norm)) for norm in normalized_counts]

    for one_edge, weight in all_pairs_countries_normalized.items():
        country_from, country_to = one_edge.split('-> ')
        
        # Get coordinates for both countries
        lon_from = longitudes[countries.index(country_from)]
        lat_from = latitudes[countries.index(country_from)]
        lon_to = longitudes[countries.index(country_to)]
        lat_to = latitudes[countries.index(country_to)]
        
        # Create a curved edge trace between the two different countries 
        if country_from != country_to: 
            folium.PolyLine(
                locations=[[lat_from, lon_from], [lat_to, lon_to]], 
                color='green',
                weight=2, 
                opacity=weight*100,
                interactive = True
            ).add_to(world_map)
    

    # Add each country node as a CircleMarker with scaled sizes
    for country, lat, lon, color, size in zip(countries, latitudes, longitudes, colors_hex_before, node_sizes):
        folium.CircleMarker(
            location=[lat, lon],  
            radius=size/500,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{country}</b><br>click count: {round(size,2)}",
                max_width=100, 
                min_width=50
            )
        ).add_to(world_map)
    
    # Save the combined map to an HTML file
    world_map.save(out_path)

    print(f"Map is saved in {out_path}!")


def overlap_world_map_clicks_after(out_path, clicks_scaled, all_pairs_countries_normalized, countries, latitudes,longitudes):
    """
    World map of the click count per country and game path between countries after scaling by the number of articles
    Args:
        out_path (path): define the name and path of the output map 
        clicks_scaled (list): list of the click count per country divided by the number of article associated with this country
        all_pairs_countries_normalized (df): Pandas DataFrame of the number of time that a path between two articles is taken by players
        countries (list): list of the name of all the countries
        latitudes (list): first geographical coordinate of each country
        longitudes (list): second geographical coordinate of each country
    """
    # Create a base map centered on (0,0))
    map_center = [0, 0]
    world_map = folium.Map(location=map_center, zoom_start=0.5, tiles='cartodbpositron')

    # define size gradient proportional to the scaled click count 
    size_scaler = MinMaxScaler(feature_range=(0.0001, max(clicks_scaled)))
    node_sizes_scaled = size_scaler.fit_transform([[count] for count in clicks_scaled]).flatten()

    # define color gradient for node color
    color_scaler = MinMaxScaler(feature_range=(0.25, 1.5))
    normalized_counts = color_scaler.fit_transform([[count] for count in clicks_scaled]).flatten()
    color_map = plt.cm.get_cmap('Purples')
    colors_hex_after = [matplotlib.colors.to_hex(color_map(norm)) for norm in normalized_counts]

    for one_edge, weight in all_pairs_countries_normalized.items():
        country_from, country_to = one_edge.split('-> ')
        
        # Get coordinates for both countries
        lon_from = longitudes[countries.index(country_from)]
        lat_from = latitudes[countries.index(country_from)]
        lon_to = longitudes[countries.index(country_to)]
        lat_to = latitudes[countries.index(country_to)]
        
        # Create edge trace between the two countries
        folium.PolyLine(
            locations=[[lat_from, lon_from], [lat_to, lon_to]], 
            color='green',
            weight=2, 
            opacity=weight*100,
            interactive = True
        ).add_to(world_map)
    

    # Add each country node as a CircleMarker with scaled sizes
    for country, lat, lon, color, size in zip(countries, latitudes, longitudes, colors_hex_after, node_sizes_scaled):
        folium.CircleMarker(
            location=[lat, lon],  # Use latitude and longitude
            radius=size/50,         # Scaled size based on occurrence
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{country}</b><br>scaled click count: {round(size,2)}",
                max_width=100, 
                min_width=50
            )
        ).add_to(world_map)

    
    # Save the combined map to an HTML file
    world_map.save(out_path)

    print(f"Map is saved in {out_path}!")


def overlap_world_map_normalized_clicks(out_path, clicks_normalized, countries, latitudes,longitudes):
    """
    World map of the click count per country and game path between countries after scaling by the number of articles
    Args:
        out_path (path): define the name and path of the output map 
        clicks_normalized (list): list of the click count per country divided by the number of links in for every article associated with this country
        countries (list): list of the name of all the countries
        latitudes (list): first geographical coordinate of each country
        longitudes (list): second geographical coordinate of each country
    """
    # Create a base map centered on (0,0))
    map_center = [0, 0]
    world_map = folium.Map(location=map_center, zoom_start=0.5, tiles='cartodbpositron')

    # define size gradient proportional to the scaled click count 
    size_scaler = MinMaxScaler(feature_range=(0.0001, max(clicks_normalized)))
    node_sizes_norm = size_scaler.fit_transform([[count] for count in clicks_normalized]).flatten()

    # define color gradient for node color
    color_scaler = MinMaxScaler(feature_range=(0,1))
    normalized_counts = color_scaler.fit_transform([[count] for count in clicks_normalized]).flatten()
    color_map = plt.cm.get_cmap('Oranges')
    colors_hex_norm = [matplotlib.colors.to_hex(color_map(norm)) for norm in normalized_counts]

    # Add each country node as a CircleMarker with scaled sizes
    for country, lat, lon, color, size in zip(countries, latitudes, longitudes, colors_hex_norm, node_sizes_norm):
        folium.CircleMarker(
            location=[lat, lon],  
            radius=size*3,      
            color=color,
            fill=True,
            fill_opacity=0.7,
            opacity=0.8,
            weight=1,
            popup=folium.Popup(
                f"<b>{country}</b><br>normalized click count: {round(size,2)}",
                max_width=100, 
                min_width=50
            )
        ).add_to(world_map)

    
    # Save the combined map to an HTML file
    world_map.save(out_path)

    print(f"Map is saved in {out_path}!")



def plot_start_stop_count(df_top_start, df_top_stop):
    """
    Plot the start and stop articles counts for the top 10 countries

    Args:
        df_top_start (dataframe): data containing the start and stop articles count (sorted by start articles count) for the top 10 countries
        df_top_stop (dataframe): data containing the start and stop articles count (sorted by stop articles count) for the top 10 countries
    """

    fig = go.Figure()

    # Add traces for start and stop articles
    fig.add_trace(go.Bar(
        x=df_top_start['country'],
        y=df_top_start['start'],
        name='Start Articles',
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        x=df_top_start['country'],
        y=df_top_start['stop'],
        name='Stop Articles',
        marker_color='#ff7f0e'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Count",
        barmode='group',
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,  # Center the buttons horizontally
                y=1.1,  # Position the buttons above the graph
                xanchor="center",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Sort by Start",
                        method="update",
                        args=[
                            {"x": [df_top_start['country'], df_top_start['country']],
                             "y": [df_top_start['start'], df_top_start['stop']]},
                        ]
                    ),
                    dict(
                        label="Sort by Stop",
                        method="update",
                        args=[
                            {"x": [df_top_stop['country'], df_top_stop['country']],
                             "y": [df_top_stop['start'], df_top_stop['stop']]},
                        ]
                    )
                ],
                showactive=True,
            )
        ],
        height=450,
        width=600,
        margin=dict(l=120, r=120, t=50, b=50), # Add margins to the plot
    )

    # Save to HTML and display
    fig.write_html("graphs/topic_1/start_stop_count.html")
    fig.show()


    # don't add title to the plot, will be added as html
# TITLE = "Top Country-Related Dead-End Articles (Before/After Scaling)"
def plot_top_dead_end_countries_plotly(unique_dead_end_countries, top_n=10):
    """
    Plots the top country-related dead-end articles with an interactive button
    to switch between before scaling and after scaling (scaled click counts).
    
    Args:
        unique_dead_end_countries (pd.DataFrame): DataFrame containing dead-end 
                                                  country-related articles with click counts and link information.
        top_n (int): Number of top articles to display (default is 10).
    """
    # get the top N data for both "before scaling" and "after scaling"
    top_before_scaling = unique_dead_end_countries.sort_values(
        by="click_count", ascending=False
    ).head(top_n)
    top_after_scaling = unique_dead_end_countries.sort_values(
        by="scaled_click_count", ascending=False
    ).head(top_n)
    
    # get global min and max for "Sum Links Out" across both datasets
    global_min = unique_dead_end_countries["sum_num_links_out"].min()
    global_max = unique_dead_end_countries["sum_num_links_out"].max()
    
    # create traces for before scaling
    trace_before = go.Bar(
        x=top_before_scaling["click_count"],
        y=top_before_scaling["Top_1_name"],
        orientation="h",
        marker=dict(
            color=top_before_scaling["sum_num_links_out"], 
            colorscale="Viridis", 
            cmin=global_min,
            cmax=global_max,
            colorbar=dict(title="Sum Links Out", x=1.02),  
        ),
        name="Before Scaling",
    )

    trace_after = go.Bar(
        x=top_after_scaling["scaled_click_count"],
        y=top_after_scaling["Top_1_name"],
        orientation="h",
        marker=dict(
            color=top_after_scaling["sum_num_links_out"], 
            colorscale="Viridis",
            cmin=global_min,
            cmax=global_max,
            colorbar=dict(title="Sum Links Out", x=1.02),
        ),
        name="After Scaling",
    )

    layout = go.Layout(
        xaxis=dict(title="Click Count"),
        yaxis=dict(title="Country"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Before Scaling",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {"xaxis.title.text": "Click Count"}  # Show first trace
                        ],
                    ),
                    dict(
                        label="After Scaling",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {"xaxis.title.text": "Scaled Click Count by Sum Links Out"}  # Show second trace
                        ],
                    ),
                ],
            )
        ],
    )

    # combine traces
    fig = go.Figure(data=[trace_before, trace_after], layout=layout)

    # initially set visibility
    fig.data[0].visible = True  # Before scaling
    fig.data[1].visible = False  # After scaling

    # fig.write_html('graphs/top_country_dead_end_articles.html')
   
    fig.show()