
# Analyzing cultural biases in Wikispeedia

## Abstract
Today, content on the internet is still still mostly produced by Western societies [[1]](https://upload.wikimedia.org/wikipedia/commons/4/4a/Decolonizing_Wikipedia.pdf) [[2]](https://www.theguardian.com/commentisfree/2017/oct/05/internet-white-western-google-wikipedia-skewed#:~:text=of%20the%20world.-,For%20the%20first%20time%20in%20history%2C%20we%20are%20creating%20a,skewed%20towards%20rich%2C%20western%20countries.). Interestingly, those same societies also produce most of the human knowledge, which we proxy as the number of citable publications [3]. [Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/) is an online game built on Wikipedia articles from 2007 during which players are navigating from a given start to a target end article through the links contained in the articles. In this project we intend to investigate players’ behaviors and their biases. More precisely, we ask if the way players play Wikispeedia is dependent on how knowledge is produced in the world? Or are they influenced by the Wikipedia graph, which is itself biased towards countries that produce the most knowledge?
 
The first step will be to understand the relationship between the way players play Wikispeedia and the production of knowledge in the world.
For this, we will compare two hypotheses, namely the “passive” and the “active” hypothesis. In the “passive” hypothesis, we assume that players “passively” play on a graph that is biased towards countries producing a lot of scientific knowledge. Thus, by removing the graph bias (i.e. by controlling for the number of articles per country, the number of links in and out of articles, balancing, propensity score matching, PageRank), there should be no bias anymore in the way players play. We consider that the Wikipedia graph is biased if it over or under represents some countries. On the other hand, the “active” hypothesis states that players “actively” add their own intrinsic biases when playing on that graph. A player bias can be defined as the inclination for or against one or several countries. By removing the graph bias, the player’s preference for some countries should still be visible.


## Data
The articles that we will use have been borrowed from a 4,600-article version of Wikipedia that was once available at the 2007 Wikipedia Selection for schools. We have the content of the articles (*data/plaintext_articles*) as well as data of the Wikispeedia games (*data/wikispeedia-paths-and-graph*). 


## Research questions
Is the way players play Wikispeedia dependent on how knowledge is produced in the world? Or are they influenced by the Wikipedia graph, which is itself biased towards countries that produce the most knowledge?
This main question can be subdivided into different smaller questions: 

1. Which countries are most represented in the Wikispeedia? Are there cultural biases intrinsic to the Wikipedia graph?
2. Are there countries that are clicked more often by players in Wikispeedia? Are there "highway paths" that are very often used by players? 
3. What articles are most likely to cause a player to stop the game? What makes those articles "dead ends"? 
4. Can controlling for different cofactors be enough to remove the observed bias? Can we explain the variance in player's click counts with the number of links leading in and out of articles? Are players significantly more biased than a random walk on the graph?
5. Do players click more often on articles that are associated with countries generating a high number of publications? Is this phenomenon also observed after balancing and controlling for any confounding factors?

## Data story
Our data story is available on the following [website](https://brygotti.github.io/lesfraisestagada/analysis.html). Happy reading!

## Methods

### 1. Are there cultural biases intrinsic to the Wikipedia graph?

#### Assign countries to articles
First, we assign one or multiple countries to each articles of the Wikispeedia dataset. This will then be useful to
determine to which region of the world (and thus to which culture) a given article belongs to. To assign each article to a country we did two things: 

- First a naive approach is used by doing a text search and finding all the country string names inside the plaintext. This is done with help of a regex that matches all the countries in all the files. The results are then parsed in a table containing as index the article name and as columns all the possible countries on earth along with the number of occurences in the specific article. After this the top 2 countries per articles are kept. This approach resulted in 1412 articles having no country assigned to them. By going manually through them it can be seen that as a human some articles can be further classified to countries even if the country name is not explicetly mentionned in the text.
- Two LLM's were tested (Qwen and Llama) and only Llama (a Meta LLM) was retained in order to assign the missing articles to countries. For this to be done the LLM was downloaded locally and used for inference on the plaintext articles. This approach allowed to classify 525 new articles having no country assigned to them. Articles classified with help of the LLM have a "Top_1_count" of 0 meaning 0 occurences of the exact country name in the article but stil able to be inferred by an llm.
- Some articles remained unclassified, as they represent entities that cannot be easily associated with a country (for example objects, celestial bodies, raw materials).

#### Links between articles
To find the number of links in and out of each article we simply counted the number of links in each article (this is the number of links out) and then counted the number of times an article occurs in other articles (this is the number of links in). 

### 2. Are there cultural biases in the way players play Wikispeedia?
#### Click counts
Here we were interested in understanding whether there is a cultural bias in the way players of the Wikispeedia game click on articles, meaning whether some articles are clicked more often and whether some countries are associated to articles that are more often clicked. 

The click count of an article is defined as the total number of times it appears across all Wikispeedia paths (i.e. ```data/paths_finished.csv``` and ```data/paths_unfinished.csv```)
Mathematically, it can be expressed as:

$$Click Count_{article} = \sum_{p=1}^{P} Occurrences_{article, p}$$

Where:
* P represents the total number of paths in the datasets.
* $Occurrences_{article, p}$ indicates the number of times the article appears in the p-th path.


#### Highway paths
In this part, the focus is made on player's behaviour to investigate whether they use some paths more often than others. Those frequently used paths are called "highway paths".
- Extract each game path from paths_finished.csv and paths_unfinished.csv as a list
- Create a new list with every 1-unit long path between articles (returns to previous articles '<' are carefully taken care of by removing the last article accessed from the list)
- Normalize the number of occurences of each pair by the total number of pairs found
- Associate each pair of article to the top 1 country of both articles
- Repeat a similar procedure for 2-unit long path between articles by extracting trios of articles from game paths

#### Dead ends analysis
In this section, we analyze "dead ends" in the Wikispeedia game by examining points where players frequently abandoned their paths before reaching the target. This includes tracking the success and failure rates of articles, associating dead-end articles with specific countries, and scaling click counts by the number of outgoing links.

The success and failure ratios are defined as follows:

$$Success Ratio = \frac{Successful Clicks}{Total Clicks}$$

$$Failure Ratio = \frac{Failure Clicks}{Total Clicks}$$

--- 

To incorporate scaling, the total clicks are adjusted by the number of outgoing links for each country, as expressed in the formula:

$$Scaled Total Clicks = \frac{Total Clicks}{Outgoing Links}$$

This scaling highlights the influence of article connectivity on player behavior, providing deeper insights into cultural patterns and player strategies across different regions.

### 3. How can we explain the players' biases?

#### Analyzing country distribution in start / target articles of proposed paths
We isolate start and end articles from paths and analyze the distribution of countries among them.

#### Regression analysis of the click counts
To investigate whether some inherent features of articles could influence le number of clicks that an article will receive in the Wikispeedia game, we performed a ordinary least squares regression using the number of clicks are the predicted/dependent variable and combinations of the number of links in and out of articles as predictors/independent variables. We also performed correlations to find relationships between these three variables. 

#### Comparing with random walk

The basic PageRank algorithm of the `networkx` package was run on the full graph of the Wikispeedia game. The ranks were then compared with the players rank, defined as:<br/>

$$r_a = \frac{c_a}{\sum_{a' \in A}{c_{a'}}}$$

where
* $c_a$ is the click count of the article $a$ (the number of times it appears in the recorded player paths)
* $A$ is the set of all articles in the Wikipedia graph

To remove as much bias from the graph as possible in the players rank (given that we want a fair comparison between the PageRanks which represents the graph's inherent bias and the players rank which represents the players' bias), we removed the start and target articles from the players paths before computing click counts. Those articles are not actually chosen by the players but are imposed by the game, so they do not represent the players' bias. Initial comparison of players vs PageRank was done.

The players and PageRank ranks were then aggregated by country for further analysis. The rank of a country was defined as: <br/>

$$r_c = \sum_{a \in A_c} r_a$$

where
* $A_c$ is the set of all articles linked to the country $c$

An analysis of the difference between the players and PageRank ranks was done to determine if players are significantly more biased than a random walk on the graph. Statistical significance was tested using a chi-square test with a number of trial $n$ set to the total number of clicks in all the recorded games.

## Team organization

### Week 28.10 - 15.11 (P2)
Jeremy
- [x]  Group articles by country first naïve approach, refine country calssification with help of an llm in order to have more articles (1.1)
- [x]  Explain the variance in player's click counts with the number of links leading in and out of articles (3.2)
- [x]  Connection between countries (1.2)
- [x] Analyze country distribution in start / target articles of proposed paths (3.1)
Bryan
- [x]  Initial skeleton of `README.md`
- [x]  Random walk analysis (3.3)
Oriane
- [x]  First test of visualisation on a map (2)
- [x]  Highway paths analysis, normalization and map to countries (2.2)
Theo
- [x]  Compute success/fail ratio → which articles have high success, high failure. Correlation with countries (2.3)
- [x]  Dead ends analysis, normalization by outgoing links and further analysis (2.3)
Claire
- [x]  Click counts analysis, compute usage of articles, draw distribution, extract most used, correlation with countries (2.1)
- [x]  Explain the variance in player's click counts with the number of links leading in and out of articles (3.2)
- [x]  Connection between countries (1.2)

### Week 29.11 - 20.12 (P3)

Jeremy
- [x] 
Claire
- [x] 
Oriane
- [x] 
Bryan
- [x]
Theo
- [x]
  
Everyone
- [x] Merge own notebook into `results.py`
- [x] Write methods in `README.md`
- [x] Write conclusion in 3.4 of `results.py`


## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-lesfraisestagada.git
cd ada-2024-project-lesfraisestagada

# [OPTIONAL] create conda environment
conda create -n ada-2024-project-lesfraisestagada python=3.11
conda activate ada-2024-project-lesfraisestagada

# install requirements
pip install -r pip_requirements.txt

# or create conda environment in case you want to run the llm locally on cuda GPUs
conda env create -f environment.yml
conda activate ada
```

## Project Structure
The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```
