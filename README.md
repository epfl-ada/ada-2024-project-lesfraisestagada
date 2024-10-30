
# Analyzing cultural biases in Wikispeedia

## Abstract
In the last few centuries, Western societies like Europe and the USA have become wealthier and wealthier compared to
African or Asian countries. This has also come with a faster technological development, meaning that the early internet
was mostly used by people from Western culture. [references?] In this project, we will analyze the impact this had on
Wikispeedia. Are people playing Wikispeedia more likely to click on articles linked to Western countries? What are
the most used articles in Wikispeedia paths? Are there unconscious biases in the way players choose their paths in
Wikispeedia? Do they tend to follow paths that connect articles from Western countries? If yes, why? Is it because
Wikipedia contains a larger proportion of articles about Western culture, or are the players themselves biased?

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
   * Does it simply derive from the fact that the Wikipedia graph is already biased in itself?
   * What is the distribution of countries among start / target articles for the Wikispeedia games? If it is not balanced,
   could this be an explanation?
   * Is there something inherent to the way players play the game?

## Methods [draft]

### Assign countries to articles
First, we assign one or multiple countries to each articles of the Wikispeedia dataset. This will then be useful to
determine to which region of the world (and thus culture) a given article belongs to. 

[TBD: How do we do this?]

Possible ways include:
* Scan in right "infobox" table for reference to a country
* Scan in entire article for reference to a country (see how we can prevent unrelated countries from being assigned,
for example with a threshold on the number of occurrences of a country in the article text?)
* What to do if an article does not relate to a country? -> throw it away

[TBD: Should we be more fine-grained than countries?]

We can try to assign cities to articles instead of countries for a more precise analysis. But we might end up with fewer
articles to analyze since it is probably harder to relate an article to a city than to a country (more articles will
have references to countries than to cities)

### 1. Are there cultural biases intrinsic to the Wikipedia graph?

[TBD]

### 2. Are there cultural biases in the way players play Wikispeedia?

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
- [ ]  Group articles by country and city (Jeremy)
- [ ]  Readme (Bryan jusqu’à jeudi, Oriane après)
- [ ]  Compute success/fail ratio → which articles have high success, high failure. Correlation with countries ? (Théo)
- [ ]  Compute usage of articles, draw distribution, extract most used, correlation with countries (Claire)

### TODOs left
- [ ]  Visualisation on a map
- [ ]  Links between countries (paths)
    - [ ]  transform article paths with country paths to find links between countries → edges on the map
- [ ]  Random walk thing ?
- [ ]  Question 1 ?

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

