
# Analyzing cultural biases in Wikispeedia

## Abstract
In the last few centuries, Western societies like Europe and the USA have become wealthier and wealthier compared to
African or Asian countries. This has also come with a faster technological development, meaning that the early internet
was dominated by Western culture. Today, content on the internet is still still mostly produced by Western societies [[1]](https://upload.wikimedia.org/wikipedia/commons/4/4a/Decolonizing_Wikipedia.pdf) [[2]](https://www.theguardian.com/commentisfree/2017/oct/05/internet-white-western-google-wikipedia-skewed#:~:text=of%20the%20world.-,For%20the%20first%20time%20in%20history%2C%20we%20are%20creating%20a,skewed%20towards%20rich%2C%20western%20countries.). In this project, we will analyze the impact this had on
[Wikispeedia](https://dlab.epfl.ch/wikispeedia/play/), an online game built on 4604 Wikipedia articles from 2007 during which players are navigating from a given start and end article only through the links contained in the articles. We intend to answer several questions: Are people players more likely to click on articles linked to Western countries? What are
the most used articles in Wikispeedia paths? Are there unconscious biases in the way players choose their paths? Do they tend to follow paths that connect articles from Western countries? Is that because
Wikipedia contains a larger proportion of articles about Western culture, or are the players themselves biased? 


## Data
The articles that we will use have been borrowed from a 4,600-article version of Wikipedia that was once available at the 2007 Wikipedia Selection for schools. We have the content of the articles (*data/plaintext_articles*) as well as data of the Wikispeedia games (*data/wikispeedia-paths-and-graph*). 


## Research questions
1. Are there cultural biases intrinsic to the Wikipedia graph?
   * What countries are the most present in the Wikipedia graph?
   * Are articles about some countries on average more connected than others? [TODO: revoir le 1.2]
2. Are there cultural biases in the way players play Wikispeedia?
   * What articles are most often clicked on? How does this relate to the country of the article?
   * What paths do Wikispeedia players most follow? Are there "highway paths" that are very often used? Can this be
linked to a cultural bias?
   * What articles are most likely to cause a player to stop the game? What makes those articles "dead ends"? What does
   this have to do with the country of these articles?
3. How can we explain the players' biases?
   * What is the distribution of countries among start / target articles for the Wikispeedia games? If it is not balanced, could this be an explanation?
   * Can we explain the variance in player's click counts with the number of links leading in and out of article?
   * How does the players' clicking behavior compare itself to PageRank? Are players significantly more biased than a random walk on the graph?
   * Is there something inherent to the way players play the game?

## Methods

### 1. Are there cultural biases intrinsic to the Wikipedia graph?

### Assign countries to articles
First, we assign one or multiple countries to each articles of the Wikispeedia dataset. This will then be useful to
determine to which region of the world (and thus to which culture) a given article belongs to. To assign each article to a country we did two things: 

- First a naive approach is used by doing a text search and finding all the country string names inside the plaintext. This is done with help of a regex that matches all the countries in all the files. The results are then parsed in a table containing as index the article name and as columns all the possible countries on earth along with the number of occurences in the specific article. After this the top 2 countries per articles are kept. This approach resulted in 1412 articles having no country assigned to them. By going manually through them it can be seen that as a human some articles can be further classified to countries even if the country name is not explicetly mentionned in the text.
- Two LLM's were tested (Qwen and Llama) and only Llama (a Meta LLM) was retained in order to assign the missing articles to countries. For this to be done the LLM was downloaded locally and used for inference on the plaintext articles. This approach allowed to classify 525 new articles having no country assigned to them. Articles classified with help of the LLM have a "Top_1_count" of 0 meaning 0 occurences of the exact country name in the article but stil able to be inferred by an llm.
- Some articles remained without countries. This is due to the fact that they can't be classified (for example objects, stars, raw materials...) 

#### Average in/out degree of articles by country
[TODO: implement this?]

### 2. Are there cultural biases in the way players play Wikispeedia?
#### Click counts
Here we were interested in understanding whether there is a cultural bias in the way players of the Wikispeedia game click on articles, meaning whether some articles are clicked more often and whether some countries are associated to articles that are more often clicked. 

- Merge all paths in the Wikispedia game (i.e. ```data/paths_finished.csv``` and ```data/paths_unfinished.csv```)
- Count the occurrences of each article in those paths 
- Plot the distribution of the article occurrences 
- Plot the distribution of the countries associated to those articles
- Normalize the click count of countries with respect the the number of occurrences of those countries

#### Highway paths
In this part, the focus is made on player's behaviour to investigate whether they use some paths more often than others. Those frequently used paths are called "highway paths".
- Extract each game path from paths_finished.csv and paths_unfinished.csv as a list
- Create a new list with every 1-unit long path between articles (returns to previous articles '<' are carefully taken care of by removing the last article accessed from the list)
- Convert the list of pairs into a Pandas DataFrame and counts the number of time that each pair appear using .value_counts()
- Normalize the number of occurences of each pair by the total number of pairs found
- Associate each pair of article to the top 1 country of both articles
- Plot the distribution of the pair occurences within all game paths
- Repeat a similar procedure for 2-unit long path between articles by extracting trios of articles from game paths

#### Dead ends analysis
In this section, "dead ends" in the Wikispeedia game were analysed by examining points where players frequently abandoned paths before reaching their target. This analysis included tracking success and failure rates of articles, linking dead-end articles to specific countries, and scaling click counts by outgoing links to potentially highlight cultural patterns in player behavior.

- Process all game paths to count occurrences, success, and failure ratios for each article.
- Calculate and printed the success rate for the backtracking action ("<").
- Analyze top articles by unique failure and success counts, segmented by category.
- Identify articles with high failure ratios, indicating potential dead ends.
- Merge articles with country data to connect dead-end articles to specific countries.
- Aggregate data by country, calculating total clicks, outgoing/incoming links, and average failure ratios.
- Plot the most significant dead-end countries based on click count and outgoing links.
- Scale click counts by outgoing links and plotted scaled dead-end countries.
- Extract last articles from unfinished paths and analyzed their frequency as dead ends by country.
- Scale last article counts by outgoing links and visualized scaled dead-end countries.
- Identify articles appearing before backtracking and counted their occurrences.
- Plot unfinished paths' articles before backtracking, including a scaled version based on outgoing links.

### 3. How can we explain the players' biases?

#### Analyzing country distribution in start / target articles of proposed paths
[TBD: The idea is that if most paths have an article linked to Western culture as the target, players will tend to follow
paths that contain more Western culture articles.]

#### Regression analysis of the click counts
To investigate whether some inherent features of articles could influence le number of clicks that an article will receive in the Wikispeedia game, we performed a ordinary least squares regression using the number of clicks are the predicted/dependent variable and combinations of the number of links in and out of articles as predictors/independent variables. We also performed correlations to find relationships between these three variables. 

#### Comparing with random walk
[TBD: The idea would be to run random walk and compare with the players' behavior. Do players spend a higher fraction
of time on certain articles than random walk?]

Random walk could be thought of as a baseline, where if a given article appear way more (or less) often on the Wikispeedia
games than on random walk, it means player do prefer this article, and it cannot be entirely explained by the number of
links to / from that article (that is, it cannot be explained by the structure of the Wikipedia graph)


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

