# here we find all functions that are needed in order to reproduce the graphs of our project. 

# Imports 
import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


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



def draw_circle_graph(df, title, out_path, project_path):
    """Draw a circular directed graph where nodes are articles
    Edges represent connections between articles. 
    Article A is said to be connected to article B if article A contains a link that is pointing to article B. 

    Args:
        df (dataframe): containing the following variables = click_count, num_links_in, num_links_out and having as an index the name of articles
        title(str): the title of the graph
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

    # Add a title into the saved HTML
    with open(out_path, 'r+', encoding="utf-8") as f:
        html = f.read()
        title_html = f"<h1 style='text-align:center; color:#333;font-size:18px;'>{title}</h1>"
        # Insert the title just after the <body> tag
        html = html.replace("<body>", f"<body>{title_html}")
        f.seek(0)
        f.write(html)
        f.truncate()

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