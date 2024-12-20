
# Analyzing cultural biases in Wikispeedia

## Abstract
Today, content on the internet is still mostly skewed towards Western societies [[1]](https://upload.wikimedia.org/wikipedia/commons/4/4a/Decolonizing_Wikipedia.pdf) [[2]](https://www.theguardian.com/commentisfree/2017/oct/05/internet-white-western-google-wikipedia-skewed#:~:text=of%20the%20world.-,For%20the%20first%20time%20in%20history%2C%20we%20are%20creating%20a,skewed%20towards%20rich%2C%20western%20countries.). Interestingly, those same societies also produce most of the human knowledge, which can be proxied by the number of citable publications [[3]](https://www.scimagojr.com/countryrank.php?year=2007&order=it&ord=desc#google_vignette). [Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/) is an online game built on 4604 Wikipedia articles from 2007, during which players are navigating from a given start to a target end article through the links contained in the articles. In this project we intend to investigate how players navigate through the game and how this navigation is influenced by the production of scientific knowledge in the world. More precisely, we are interested in understanding whether players are attracted towards articles linked to countries producing a lot of scientific knowledge.

## Data
The Wikispeedia dataset is from [here](https://snap.stanford.edu/data/wikispeedia.html). It contains around 4600 articles from the [Wikipedia for Schools selection of 2007](https://en.wikipedia.org/wiki/Wikipedia:Wikipedia_for_Schools). For each article, we have access to their text content, as well as a list of links between articles. The dataset also contains paths recordings from players of the [Wikispeedia game](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game).

Additionnaly to this, in order to quantify the knowledge produced by each country, we will use the number of citable publications per country in 2007. This data is available on the [Science and Engineering indicators](https://ncses.nsf.gov/pubs/nsb202333/publication-output-by-region-country-or-economy-and-by-scientific-field#utm_source=chatgpt.com) website.


## Research questions
Is the way players play Wikispeedia dependent on how knowledge is produced in the world? Or are they only influenced by the Wikipedia graph, which is itself biased towards countries that produce the most knowledge?
This main question can be subdivided into different sub-questions: 

1. Which countries are most represented in Wikispeedia? Are there cultural biases intrinsic to the Wikipedia graph?
2. Are there countries that are clicked more often by players in Wikispeedia? Are there "highway paths" that are very often used by players? 
3. What countries are most likely to cause a player to stop the game? What makes those countries "dead ends"? 
4. Can controlling for different cofactors be enough to remove the observed bias? Can we explain the variance in player's click counts with the number of links leading in and out of articles? Are players significantly more biased than a random walk on the graph?
5. Is the Wikipedia graph accurately representing the knowledge produced by each country? Is this also observed after scaling and controlling for any confounding factors?


## Data story
Our data story is available on the following [website](https://brygotti.github.io/lesfraisestagada/). Happy reading!

In case you want to look at the code of the website it is available [here](https://github.com/brygotti/lesfraisestagada).

## Methods

In this project, we will do analyses using graphical representations showing world knowledge production, the number of articles in the Wikipedia graph, the number of clicks on articles by players along with the main paths used by players in the game.

### 0. Data preprocessing
#### Assign countries to articles
First, we assigned one country to each articles of the Wikispeedia dataset. This will later be useful to
determine to which region of the world (and thus to which culture) a given article belongs to. To assign each article to a country we did two things: 

- Two approaches were used in order to classify the countries. A naïve text search approach and a more context aware aproach using LLMs. 
- In order to assess the quality of the classification, we used a consensus approach. We annotated 50 articles (where each article was annotated by two different team memebers) and we only kept the articles that were classified as the same by both annotators. This allowed to create an agreement metric used to compare the automated country classification methods.
- We then improved the models classifications in order in order to have a higher agreement with the human annotators while avoiding to classify too many articles.
- Some articles remained unclassified, as they represent entities that cannot be easily associated with a country (for example objects, celestial bodies, raw materials).

### 1. Which countries are most represented in Wikispeedia? Are there cultural biases intrinsic to the Wikipedia graph?
The first step of our analysis was to explore the features of the Wikipedia graph in terms of article usage, country occurrences, and node degrees. 

#### Links between articles
To find the number of links in and out of each article we simply counted the number of links in each article (this is the number of links out) and then counted the number of times an article occurs in other articles (this is the number of links in). To get the node degrees of countries, we summed the degrees of their associated countries.

### 2. Are there countries that are clicked more often by players in Wikispeedia? Are there "highway paths" that are very often used by players? 
#### Click counts
To understand whether some articles are clicked more often and whether some countries are associated to articles that are more often clicked we define a new variable: the click count. 
The click count of an article is defined as the total number of times it appears across all Wikispeedia paths (i.e. ```data/paths_finished.csv``` and ```data/paths_unfinished.csv```)
Mathematically, it can be expressed as:

$$Click Count_{article} = \sum_{p=1}^{P} Occurrences_{article, p}$$

Where:
* P represents the total number of paths in the datasets.
* $Occurrences_{article, p}$ indicates the number of times the article appears in the p-th path.

> To best proxy the clicking behavior of players, source/start and target/end articles were not counted as clicks. 

#### Highway paths
In this part, we investigate whether some paths are more often used than others. Those frequently used paths are called "highway paths". To identify them, we followed those steps:
- Extract each game path from paths_finished.csv and paths_unfinished.csv as a list
- Create a new list with every 1-unit long path between articles. Returns to previous articles (symbolized as '<') are carefully taken care of by removing the last article accessed from the list.
- Normalize the number of occurences of each pair by the total number of pairs found
- Map each pair of article to the top 1 country of both articles

### 3. What articles are most likely to cause a player to stop the game? What makes those articles "dead ends"? 
#### Dead ends analysis
In this section, we analyze "dead ends" in the Wikispeedia game by examining points where players frequently abandoned their paths before reaching the target. This includes tracking the success and failure rates of articles, associating dead-end articles with specific countries, and scaling click counts by the number of outgoing links.

The success and failure ratios are defined as follows:

$$Success Ratio = \frac{Successful Clicks}{Total Clicks}$$

$$Failure Ratio = \frac{Failure Clicks}{Total Clicks}$$


To incorporate scaling, the total clicks are adjusted by the number of outgoing links for each country, as expressed in the formula:

$$Scaled Total Clicks = \frac{Total Clicks}{Outgoing Links}$$

This scaling highlights the influence of article connectivity on player behavior, providing deeper insights into cultural patterns and player strategies across different regions.

### 4. Can controlling for different cofactors be enough to remove the observed bias? Can we explain the variance in player's click counts with the number of links leading in and out of articles? Are players significantly more biased than a random walk on the graph?
#### Regression analysis of the click counts
To investigate whether some inherent features of articles could influence le number of clicks that an article will receive in the Wikispeedia game, we performed a ordinary least squares regression using the number of clicks are the predicted/dependent variable and combinations of the number of links in and out of articles as predictors/independent variables. We also performed Pearson's correlations to find relationships between these three variables. 

#### Comparing with random walk

The basic PageRank algorithm of the `networkx` package was run on the full graph of the Wikispeedia game. The ranks were then compared with the players rank, defined as:<br/>

$$r_a = \frac{c_a}{\sum_{a' \in A}{c_{a'}}}$$

where
* $c_a$ is the click count of the article $a$ (the number of times it appears in the recorded player paths)
* $A$ is the set of all articles in the Wikipedia graph

The players and PageRank ranks were then aggregated by country for further analysis. The rank of a country was defined as: <br/>

$$r_c = \sum_{a \in A_c} r_a$$

where
* $A_c$ is the set of all articles linked to the country $c$

An analysis of the difference between the players and PageRank ranks was done to determine if players are significantly more biased than a random walk on the graph. Statistical significance was tested using a chi-square test with a number of trial $n$ set to the total number of clicks in all the recorded games.

## Team organization

### Weeks 29.11 - 20.12 (P3)

Jeremy
- [x] Refined the country classification for all the methods and improve the LLM generations. 
- [x] Organise human annotation and preprocess the data for the analysis part of the project.
- [x] Improved robustness of the country classification.

Claire
- [x] Analysis of the connectivity of the articles
- [x] Analysis of the countries represented by the articles and connections between countries
- [x] Analysis of the in degrees and out degrees of the articles
- [x] Introduction of the website

Oriane
- [x] Analysis of the click counts per country
- [x] Analysis of the highway/game paths between countries before and after scaling
- [x] Introduction of the website

Bryan
- [x] Rebalancing of the graph and further improvements on confounders.
- [x] Page rank implmentation and analysis.
- [x] Website creation and UI design.

Theo
- [x] Dead end analysis and scaling per country.
- [x] Analysis of the success and failure ratios of the articles.
- [x] Knowledge distribution per country with the number of citable publications.
  
Everyone
- [x] Merge own notebook into `results.py`
- [x] Write methods in `README.md`
- [x] Write conclusion in 3.4 of `results.py`
- [x] Contribute to the website and convert the graphs to plotty so that they can be shown on the website

## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-lesfraisestagada.git
cd ada-2024-project-lesfraisestagada

# [OPTIONAL] create conda environment
conda create -n ada-2024-project-lesfraisestagada python=3.11
conda activate ada-2024-project-lesfraisestagada

# install requirements
pip install -r requirements.txt
```

If you want to run the LLM part locally on cuda GPUs, you will need to use a different conda environment, named `ada` and defined in the `environment.yml` file. You can create and activate this environment by running the following command:
```bash
conda env create -f environment.yml
conda activate ada
```

To run the scripts in the `src/scripts` folder, you can use the following commands:

This will launch the country classification for the articles with LLM and Qwen. This will take nearly 40min per model and it assumes you have a GPU with at least 30GB of VRAM. The EPFL clusters were used to run this script.
```bash
python src/scripts/article_to_country.py
```

This will generate the country_clicks_links.csv file that is used in the analysis part of the project.
```bash
python src/scripts/article_clicks_links.py
```

## Project structure
The directory structure of the project looks like this:

```
├── data                                                    <- Project data files
|   ├── annotated                                               <- Directory containing the annotated articles
│   │   ├── consensus.csv                                           <- Consensus file for the country classification (only the articles that were classified as the same from both annotators)
│   │   ├── subset_bryan.csv                                        <- Annotations of the articles by Bryan
│   │   ├── subset_claire.csv                                       <- Annotations of the articles by Claire
│   │   ├── subset_jeremy.csv                                       <- Annotations of the articles by Jeremy
│   │   ├── subset_oriane.csv                                       <- Annotations of the articles by Oriane
│   │   └── subset_theo.csv                                         <- Annotations of the articles by Theo
│   │
│   ├── country_clicks_links.csv                                <- File containing main features of the articles used in our analysis
│   ├── country_data_full_llama_improved_normal.csv             <- Country classification file using the LLaMa model in nomal mode
│   ├── country_data_full_llama_improved_reversed.csv           <- Country classification file using the LLaMa model with a reversed list of countries (for robustness)
│   ├── country_data_full_llama_improved.csv                    <- Country classification file using the LLaMa model where a country is kept only if normal and reversed classifications are the same
│   ├── country_data_full_llama.csv                             <- Country classification file using the LLaMa model (first version)
│   ├── country_data_full_qwen.csv                              <- Country classification file using the Qwen model
│   ├── country_data.csv                                        <- Country classification temporary file used for naive text search approach
│   ├── country_occurences.csv                                  <- Country classification file using the naive text search approach with top 2 countries
│   └── publications_2007.csv                                   <- File containing the number of citable publications per country in 2007
│
├── src                         <- Source code
│   └── data                            <- Data directory
│       └── dataloader.py                           <- Functions used to load the data
│   └── models                          <- Model directory
│       └── llm_classifier.py                       <- Generator class abstracting the LLM model
│   └── utils                           <- Utility directory
│       ├── functions.py                           <- Utility functions used to process the data and render the results in the notebooks
│       ├── graphs.py                              <- Utility functions used to create the graphs
│       └── plots.py                               <- Utility functions used to render the plots in the notebooks
│   └── scripts                         <- External python scripts that are used to generate the data
│       ├── article_to_country.py                  <- Script allowing to launch the LLM in different configurations.
│       └── article_clicks_links.py                <- Scipt to generate the country_clicks_links.csv file
|
├── data                        <- Directory containing all the graphs exported for the data story
│
├── results.ipynb               <- Main notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── requirements.txt            <- File for installing python dependencies
├── environment.yml             <- File for installing python dependencies
└── README.md
```
