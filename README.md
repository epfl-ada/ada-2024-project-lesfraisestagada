
# Analyzing cultural biases in Wikispeedia

## Abstract
In the last few centuries, Western societies like Europe and the USA have become wealthier and wealthier compared to
African or Asian countries. This has also come with a faster technological development, meaning that the early internet
was mostly used by people from Western culture. [references?] In this project, we will analyze the impact this had on
[Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/), an online game built on 4604 Wikipedia articles from 2007 during which players are navigating from a given start and end article only through the links contained in the articles. We intend to answer several questions: Are people players more likely to click on articles linked to Western countries? What are
the most used articles in Wikispeedia paths? Are there unconscious biases in the way players choose their paths? Do they tend to follow paths that connect articles from Western countries? Is that because
Wikipedia contains a larger proportion of articles about Western culture, or are the players themselves biased? 


## Data
The articles that we will use have been borrowed from a 4,600-article version of Wikipedia that was once available at the 2007 Wikipedia Selection for schools. We have the content of the articles as well as data of the Wikispeedia games. 

- `plaintext_articles`: **Plain text content of the 4604 Wikipedia articles in .txt format**, ordered by alphabetical order (A to Z). Those are the articles through which the players could navigate.
- `wikispeedia-paths-and-graph`: Navigation paths and Wikipedia hyperlink graph (without article content)
    - `articles.tsv`: **The list of all articles**
    - `categories.tsv`:  **Hierarchical categories of all articles**, categories like *subject.Countries* or *subject.Geography.Natural_Disaster*, the main category of the articles. 
    - `links.tsv`: **The list of all links between articles**, in one column the source page and in the second column all pages that are link in the source page 
    - `paths_finished.tsv`: **Successful (i.e., finished) Wikispeedia paths** from a source to a target (reached!!) page
    - `paths_unfinished`: **Unsuccessful (i.e., unfinished) Wikispeedia paths** from a source to a target (not reached) page
    - `shortest-path-distance-matrix.txt`: **The shortest-path distances between all pairs of articles**, computed using the Floyd-Warshall algorithm. This is what the computer would use to solve the Wikispeedia game!

## Research questions
1. Are there cultural biases intrinsic to the Wikipedia graph?
   * What countries are the most present in the Wikipedia graph?
   * Are articles about some countries on average more connected than others?
2. Are there cultural biases in the way players play Wikispeedia?
   * What articles are most often clicked on? How does this relate to the country of the article?
   * What paths do Wikispeedia players most follow? Are there "highway paths" that are very often used? Can this be
linked to a cultural bias?
   * What articles are most likely to cause a player to stop the game? What makes those articles "dead ends"? What does
   this have to do with the country of these articles?
3. How can we explain the players' biases?
   * What is the distribution of countries among start / target articles for the Wikispeedia games? If it is not balanced, could this be an explanation?
   * Does it simply derive from the fact that the Wikipedia graph is already biased in itself? (change question --> page rank le B)
   * Can we explain the variance in player's click counts with the number of links leading in and out of article?
   * Is there something inherent to the way players play the game?

## Methods 

### Assign countries to articles
First, we assign one or multiple countries to each articles of the Wikispeedia dataset. This will then be useful to
determine to which region of the world (and thus to which culture) a given article belongs to. To assign each article to a country we did two things: 

- First a naive approach is used by doing a text search and finding all the country string names inside the plaintext. This is done with help of a regex that matches all the countries in all the files. The results are then parsed in a table containing as index the article name and as columns all the possible countries on earth along with the number of occurences in the specific article. After this the top 2 countries per articles are kept. This approach resulted in 1412 articles having no country assigned to them. By going manually through them it can be seen that as a human some articles can be further classified to countries even if the country name is not explicetly mentionned in the text.
- Two LLM's were tested (Qwen and Llama) and only Llama (a Meta LLM) was retained in order to assign the missing articles to countries. For this to be done the LLM was downloaded locally and used for inference on the plaintext articles. This approach allowed to classify 525 new articles having no country assigned to them. Articles classified with help of the LLM have a "Top_1_count" of 0 meaning 0 occurences of the exact country name in the article but stil able to be inferred by an llm.
- Some articles remained without countries. This is due to the fact that they can't be classified (for example objects, stars, raw materials...)  

### 1. Are there cultural biases intrinsic to the Wikipedia graph?

- For each country we computed the number of articles that related to this country
- We then displayed the distribution of country occurrences (most and least occurring countries)

### 2. Are there cultural biases in the way players play Wikispeedia?
#### 2.1. Clicks
Here we were interested in understanding whether there is a cultural bias in the way players of the Wikispeedia game click on articles, meaning whether some articles are clicked more often and whether some countries are associated to articles that are more often clicked. 

- Merge all paths in the Wikispedia game (i.e. ```data/paths_finished.csv``` and ```data/paths_unfinished.csv```)
- Count the occurrences of each article in those paths 
- Plot the distribution of the article occurrences 
- Plot the distribution of the countries associated to those articles
- Normalize the click count of countries with respect the the number of occurrences of those countries 


#### Computing metrics

For each country, we compute the following metrics:
* Number of occurrences in Wikispeedia paths
* Number of times an article appeared at the end of an unfinished path (a "dead end count")

For each pair of countries (A,B), we compute the following metrics:
* Number of times a Wikispeedia player followed a link that made him go from an article about country A to an article
about country B

#### World map visualization

[TBD: Our current plan for this is the following]

For each country, create a point on the map.
* The size of the point depends on the number of times the article has been clicked on in Wikispeedia.
* The color of the point is a gradient between green and red, green meaning the article never appeared as a dead end,
and red meaning the article appeared very often as a dead end

Between points, draw edges that correspond to the paths of the players.
* The thickness of a given edge between countries A and B correspond to how often players went from an article about A
to an article about B by following a link

To make sure the visualization does not contain too many points / too many edges we might consider filtering the articles
to only keep the most used ones in the games. We could also do other visualizations where we only keep a subset of
categories of articles (for example, only the Historical Figures category)

### 3. How can we explain the players' biases?

#### Comparing with random walk

[TBD: The idea would be to run random walk and compare with the players' behavior. Do players spend a higher fraction
of time on certain articles than random walk?]

Random walk could be thought of as a baseline, where if a given article appear way more (or less) often on the Wikispeedia
games than on random walk, it means player do prefer this article, and it cannot be entirely explained by the number of
links to / from that article (that is, it cannot be explained by the structure of the Wikipedia graph)

#### Analyzing country distribution in start / target articles of proposed paths

[TBD: The idea is that if most paths have an article linked to Western culture as the target, players will tend to follow
paths that contain more Western culture articles.]


## Proposed timeline [current TODOs]

### Week 28.10 - 01.11
- [x]  Group articles by country first naïve approach (Jeremy)
- [ ]  Readme (Bryan jusqu’à jeudi, Oriane après)
- [ ]  Compute success/fail ratio → which articles have high success, high failure. Correlation with countries ? (Théo)
- [ ]  Compute usage of articles, draw distribution, extract most used, correlation with countries (Claire)

### Week 01.11 - 08.11
- [x] refine country calssification with help of an llm in order to have more articles (Jérémy)
- [ ]  Visualisation on a map
- [ ]  Links between countries (paths)
   - [ ]  transform article paths with country paths to find links between countries → edges on the map
- [ ]  Random walk thing ?
- [ ]  Question 1 ?

### Week 08.11 - 15.11
- [x] Statistical analysis on the number of clicks an article get in function of different features (Jérémy)
- [x] Can we explain the variance in player's click counts with the number of links leading in and out of articles (Jérémy)

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



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
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

