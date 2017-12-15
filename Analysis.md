
# Robert Corbett

Final Project - Token Influence on Reddit Post Poularity

12/14/2017


```python
from datetime import datetime
import time
time1 = time.time()
```


```python
import numpy as np
import pandas as pd
import nltk
import json
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
punctset = set(['.',',','!','?','&','@','#','$','%','^','*',';',':','...','-'])
fullset=punctset.union(stopset)
```


```python
d = {'body':[],
     'score':[],
     'subreddit':[],
     'percentile':[]
    }
```


```python
e = {'Android':[],
     'boardgames':[],
     'Conservative':[],
     'hockey':[],
     'Libertarian':[],
     'neoliberal':[],
     'politics':[],
     'worldnews':[]
    }
```


```python
f = {'3':[],
     '2':[],
     '1':[]
    }
```

# create dataframe

The following function is used to take the relevant information from the json files and create a dataframe.  Because there were so many posts, I wrote the function so that I could build smalled dataframes to test with.  


```python
def reddit_df_create(sample_size, start_line, z):
    sample = sample_size
    start = start_line
    num_posts=0
    x=(1*z)*(sample*8)
    with open('../../json_files/Android.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<sample and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'Android'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/boardgames.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*2) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'boardgames'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/Conservative.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*3) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'Conservative'
                num_posts=num_posts+1  
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/hockey.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*4) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'hockey'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/Libertarian.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*5) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'Libertarian'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/neoliberal.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*6) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'neoliberal'
                num_posts=num_posts+1  
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/politics.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*7) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'politics'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break

    with open('../../json_files/worldnews.txt', 'r') as infile:
        y=0
        for line in infile:
            if(num_posts<(sample*8) and y>=start):
                line_in = json.loads(line)
                reddit_df.loc[x, 'body'] = line_in['body']
                reddit_df.loc[x, 'score'] = line_in['score']
                reddit_df.loc[x, 'subreddit'] = 'worldnews'
                num_posts=num_posts+1
                x=x+1
            y=y+1
            if(y>=sample+start):
                break
```

# return highest weighted tokens

The following function is used to return a dataframe with the highest weighted tokens in a classifier.  The function takes the classifier passed to it and sorts it.  It then takes the last 100 elements in the dataframe and matches those to the correct tokens in the vectorizer.  (the number of tokens returned can be easily changed)


```python
def return_top100(vectorizer, clf, class_labels):
    return_df = pd.DataFrame(data=e)
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top100 = np.argsort(clf.coef_[i])[-100:]
        if(class_label=='Android'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'Android'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='boardgames'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'boardgames'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='Conservative'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'Conservative'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='hockey'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'hockey'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='Libertarian'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'Libertarian'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='neoliberal'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'neoliberal'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='politics'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'politics'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label=='worldnews'):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, 'worldnews'] = (feature_names[j])
                iterator = iterator + 1
    return(return_df)
```

# return highest weighted tokens for score

The following function is the same as the above, but is used when the model is trying to predict scores.


```python
def return_top100_score(vectorizer, clf, class_labels):
    return_df = pd.DataFrame(data=f)
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top100 = np.argsort(clf.coef_[i])[-100:]
        if(class_label==3):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, '3'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label==2):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, '2'] = (feature_names[j])
                iterator = iterator + 1
        if(class_label==1):
            iterator = 0
            for j in top100:
                return_df.loc[iterator, '1'] = (feature_names[j])
                iterator = iterator + 1
    return(return_df)
```

# variables to build dataframe

I used these variables to change how many posts to retreive to test with and how many to retrieve at a time.  For the final test, both were set to 50,000 so that all 50,000 posts in each subreddit would be included in the dataframe. (400,000 total posts)


```python
num_retreive=50000
tot_num_posts=50000
```


```python
reddit_df = pd.DataFrame(data=d)
```

# first model

The model below calls reddit_df_create, it is the only time it is called in this notebook and that dataframe is used for the rest of the program.  This call takes the most time of any part of the program.  It than creates a multinomialNB model with the dataframe. I used the nltk word tokenizer to tokenize the post bodies. The min token frequency is 16.  I excluded my fullset, which is the union of the nltk english stopwords and a set of punctuation I defined in the first box with the import statements.  The data is split 80% train 20% test.  The model is trying to predict which subreddit the test post is from.  The accuracy when the model used 320,000 posts to test ended up being 69.5%.  this accuracy doesn't seem too bad when it is taken into consideration that the model had 8 targets to choose from and many of them were chosen because they have similar topics.


```python
reddit_vec = CountVectorizer(min_df=16, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(tot_num_posts/num_retreive)):

    iterator=iterator+1
    print(iterator)
    #start=start+num_retreive
    reddit_df_create(num_retreive, start, z)
    y = reddit_df['subreddit']

    reddit_counts = reddit_vec.fit_transform(reddit_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.6950625


# top 100 tokens for each subreddit

Below, I call return_top100 of the classifier and vectorizer and display the returned dataframe.  Each column has the 100 highest weighted tokens for each subreddit. Then I saved the dataframe as a csv file.


```python
reddit_top_unigrams_df = return_top100(reddit_vec, classifier, class_labels=np.unique(y_train))
reddit_top_unigrams_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Android</th>
      <th>Conservative</th>
      <th>Libertarian</th>
      <th>boardgames</th>
      <th>hockey</th>
      <th>neoliberal</th>
      <th>politics</th>
      <th>worldnews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>button</td>
      <td>everyone</td>
      <td>world</td>
      <td>3</td>
      <td>always</td>
      <td>since</td>
      <td>'ll</td>
      <td>probably</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stock</td>
      <td>first</td>
      <td>yes</td>
      <td>look</td>
      <td>take</td>
      <td>part</td>
      <td>government</td>
      <td>'ll</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
      <td>problem</td>
      <td>force</td>
      <td>definitely</td>
      <td>thing</td>
      <td>without</td>
      <td>*i</td>
      <td>let</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lot</td>
      <td>american</td>
      <td>give</td>
      <td>enough</td>
      <td>goal</td>
      <td>everyone</td>
      <td>every</td>
      <td>use</td>
    </tr>
    <tr>
      <th>4</th>
      <td>htc</td>
      <td>’</td>
      <td>socialist</td>
      <td>getting</td>
      <td>2</td>
      <td>idea</td>
      <td>research</td>
      <td>work</td>
    </tr>
    <tr>
      <th>5</th>
      <td>go</td>
      <td>money</td>
      <td>'d</td>
      <td>since</td>
      <td>trade</td>
      <td>years</td>
      <td>republicans</td>
      <td>threat</td>
    </tr>
    <tr>
      <th>6</th>
      <td>every</td>
      <td>let</td>
      <td>life</td>
      <td>everyone</td>
      <td>shit</td>
      <td>market</td>
      <td>contact</td>
      <td>everyone</td>
    </tr>
    <tr>
      <th>7</th>
      <td>update</td>
      <td>]</td>
      <td>public</td>
      <td>take</td>
      <td>something</td>
      <td>economic</td>
      <td>performed</td>
      <td>better</td>
    </tr>
    <tr>
      <th>8</th>
      <td>support</td>
      <td>[</td>
      <td>wrong</td>
      <td>someone</td>
      <td>look</td>
      <td>'</td>
      <td>ban</td>
      <td>far</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sure</td>
      <td>care</td>
      <td>agree</td>
      <td>say</td>
      <td>saying</td>
      <td>american</td>
      <td>work</td>
      <td>stop</td>
    </tr>
    <tr>
      <th>10</th>
      <td>actually</td>
      <td>back</td>
      <td>sure</td>
      <td>theme</td>
      <td>actually</td>
      <td>anything</td>
      <td>/message/compose/</td>
      <td>america</td>
    </tr>
    <tr>
      <th>11</th>
      <td>moto</td>
      <td>nazi</td>
      <td>law</td>
      <td>maybe</td>
      <td>maybe</td>
      <td>problem</td>
      <td>ideas</td>
      <td>mean</td>
    </tr>
    <tr>
      <th>12</th>
      <td>great</td>
      <td>http</td>
      <td>anything</td>
      <td>plays</td>
      <td>getting</td>
      <td>white</td>
      <td>take</td>
      <td>enough</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hardware</td>
      <td>bad</td>
      <td>care</td>
      <td>little</td>
      <td>yeah</td>
      <td>immigration</td>
      <td>many</td>
      <td>korean</td>
    </tr>
    <tr>
      <th>14</th>
      <td>going</td>
      <td>liberals</td>
      <td>business</td>
      <td>actually</td>
      <td>never</td>
      <td>first</td>
      <td>years</td>
      <td>nuke</td>
    </tr>
    <tr>
      <th>15</th>
      <td>never</td>
      <td>anyone</td>
      <td>means</td>
      <td>looking</td>
      <td>two</td>
      <td>’</td>
      <td>concerns</td>
      <td>lot</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8</td>
      <td>nothing</td>
      <td>country</td>
      <td>things</td>
      <td>cup</td>
      <td>money</td>
      <td>attack</td>
      <td>every</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ca</td>
      <td>speech</td>
      <td>every</td>
      <td>expansions</td>
      <td>ca</td>
      <td>'ll</td>
      <td>point</td>
      <td>back</td>
    </tr>
    <tr>
      <th>18</th>
      <td>charging</td>
      <td>states</td>
      <td>libertarianism</td>
      <td>always</td>
      <td>points</td>
      <td>saying</td>
      <td>law</td>
      <td>saying</td>
    </tr>
    <tr>
      <th>19</th>
      <td>thing</td>
      <td>support</td>
      <td>power</td>
      <td>kickstarter</td>
      <td>gt</td>
      <td>sure</td>
      <td>general</td>
      <td>attack</td>
    </tr>
    <tr>
      <th>20</th>
      <td>software</td>
      <td>democrats</td>
      <td>never</td>
      <td>enjoy</td>
      <td>puck</td>
      <td>got</td>
      <td>report</td>
      <td>government</td>
    </tr>
    <tr>
      <th>21</th>
      <td>work</td>
      <td>years</td>
      <td>'ve</td>
      <td>deck</td>
      <td>every</td>
      <td>sub</td>
      <td>back</td>
      <td>bomb</td>
    </tr>
    <tr>
      <th>22</th>
      <td>got</td>
      <td>wrong</td>
      <td>everyone</td>
      <td>probably</td>
      <td>bad</td>
      <td>literally</td>
      <td>someone</td>
      <td>'d</td>
    </tr>
    <tr>
      <th>23</th>
      <td>'d</td>
      <td>law</td>
      <td>problem</td>
      <td>sure</td>
      <td>’</td>
      <td>system</td>
      <td>anything</td>
      <td>things</td>
    </tr>
    <tr>
      <th>24</th>
      <td>something</td>
      <td>take</td>
      <td>believe</td>
      <td>better</td>
      <td>said</td>
      <td>every</td>
      <td>current</td>
      <td>nothing</td>
    </tr>
    <tr>
      <th>25</th>
      <td>pretty</td>
      <td>person</td>
      <td>healthcare</td>
      <td>never</td>
      <td>played</td>
      <td>shit</td>
      <td>others</td>
      <td>missile</td>
    </tr>
    <tr>
      <th>26</th>
      <td>since</td>
      <td>free</td>
      <td>saying</td>
      <td>need</td>
      <td>ice</td>
      <td>never</td>
      <td>'ve</td>
      <td>sure</td>
    </tr>
    <tr>
      <th>27</th>
      <td>way</td>
      <td>every</td>
      <td>better</td>
      <td>many</td>
      <td>point</td>
      <td>someone</td>
      <td>never</td>
      <td>anything</td>
    </tr>
    <tr>
      <th>28</th>
      <td>need</td>
      <td>sure</td>
      <td>said</td>
      <td>try</td>
      <td>playing</td>
      <td>less</td>
      <td>also</td>
      <td>first</td>
    </tr>
    <tr>
      <th>29</th>
      <td>’</td>
      <td>racist</td>
      <td>thing</td>
      <td>table</td>
      <td>fan</td>
      <td>political</td>
      <td>bot</td>
      <td>eu</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>gt</td>
      <td>way</td>
      <td>[</td>
      <td>great</td>
      <td>time</td>
      <td>trump</td>
      <td>us</td>
      <td>russia</td>
    </tr>
    <tr>
      <th>71</th>
      <td>''</td>
      <td>see</td>
      <td>libertarians</td>
      <td>much</td>
      <td>[</td>
      <td>right</td>
      <td>subreddit</td>
      <td>know</td>
    </tr>
    <tr>
      <th>72</th>
      <td>apps</td>
      <td>going</td>
      <td>market</td>
      <td>player</td>
      <td>]</td>
      <td>us</td>
      <td>right</td>
      <td>get</td>
    </tr>
    <tr>
      <th>73</th>
      <td>really</td>
      <td>said</td>
      <td>property</td>
      <td>players</td>
      <td>league</td>
      <td>make</td>
      <td>comments</td>
      <td>http</td>
    </tr>
    <tr>
      <th>74</th>
      <td>still</td>
      <td>make</td>
      <td>make</td>
      <td>playing</td>
      <td>see</td>
      <td>want</td>
      <td>downvotes</td>
      <td>could</td>
    </tr>
    <tr>
      <th>75</th>
      <td>camera</td>
      <td>say</td>
      <td>system</td>
      <td>fun</td>
      <td>''</td>
      <td>much</td>
      <td>even</td>
      <td>even</td>
    </tr>
    <tr>
      <th>76</th>
      <td>get</td>
      <td>know</td>
      <td>pay</td>
      <td>also</td>
      <td>teams</td>
      <td>take</td>
      <td>president</td>
      <td>country</td>
    </tr>
    <tr>
      <th>77</th>
      <td>one</td>
      <td>government</td>
      <td>free</td>
      <td>``</td>
      <td>play</td>
      <td>know</td>
      <td>know</td>
      <td>think</td>
    </tr>
    <tr>
      <th>78</th>
      <td>device</td>
      <td>want</td>
      <td>even</td>
      <td>good</td>
      <td>'re</td>
      <td>really</td>
      <td>'m</td>
      <td>'re</td>
    </tr>
    <tr>
      <th>79</th>
      <td>samsung</td>
      <td>conservative</td>
      <td>state</td>
      <td>''</td>
      <td>really</td>
      <td>even</td>
      <td>questions</td>
      <td>[</td>
    </tr>
    <tr>
      <th>80</th>
      <td>]</td>
      <td>even</td>
      <td>money</td>
      <td>people</td>
      <td>last</td>
      <td>get</td>
      <td>one</td>
      <td>]</td>
    </tr>
    <tr>
      <th>81</th>
      <td>[</td>
      <td>white</td>
      <td>want</td>
      <td>think</td>
      <td>'m</td>
      <td>also</td>
      <td>get</td>
      <td>one</td>
    </tr>
    <tr>
      <th>82</th>
      <td>app</td>
      <td>get</td>
      <td>get</td>
      <td>time</td>
      <td>player</td>
      <td>good</td>
      <td>think</td>
      <td>world</td>
    </tr>
    <tr>
      <th>83</th>
      <td>screen</td>
      <td>left</td>
      <td>'m</td>
      <td>would</td>
      <td>game</td>
      <td>one</td>
      <td>'re</td>
      <td>gt</td>
    </tr>
    <tr>
      <th>84</th>
      <td>would</td>
      <td>'m</td>
      <td>one</td>
      <td>'m</td>
      <td>get</td>
      <td>'re</td>
      <td>see</td>
      <td>like</td>
    </tr>
    <tr>
      <th>85</th>
      <td>'m</td>
      <td>one</td>
      <td>right</td>
      <td>'ve</td>
      <td>good</td>
      <td>'m</td>
      <td>like</td>
      <td>``</td>
    </tr>
    <tr>
      <th>86</th>
      <td>use</td>
      <td>(</td>
      <td>libertarian</td>
      <td>board</td>
      <td>one</td>
      <td>think</td>
      <td>https</td>
      <td>''</td>
    </tr>
    <tr>
      <th>87</th>
      <td>battery</td>
      <td>)</td>
      <td>think</td>
      <td>cards</td>
      <td>players</td>
      <td>https</td>
      <td>gt</td>
      <td>nuclear</td>
    </tr>
    <tr>
      <th>88</th>
      <td>pixel</td>
      <td>'re</td>
      <td>like</td>
      <td>really</td>
      <td>think</td>
      <td>]</td>
      <td>please</td>
      <td>war</td>
    </tr>
    <tr>
      <th>89</th>
      <td>iphone</td>
      <td>right</td>
      <td>'re</td>
      <td>get</td>
      <td>would</td>
      <td>[</td>
      <td>would</td>
      <td>north</td>
    </tr>
    <tr>
      <th>90</th>
      <td>apple</td>
      <td>think</td>
      <td>(</td>
      <td>one</td>
      <td>hockey</td>
      <td>like</td>
      <td>``</td>
      <td>(</td>
    </tr>
    <tr>
      <th>91</th>
      <td>like</td>
      <td>gt</td>
      <td>)</td>
      <td>played</td>
      <td>nhl</td>
      <td>would</td>
      <td>''</td>
      <td>people</td>
    </tr>
    <tr>
      <th>92</th>
      <td>phones</td>
      <td>like</td>
      <td>would</td>
      <td>like</td>
      <td>(</td>
      <td>``</td>
      <td>[</td>
      <td>)</td>
    </tr>
    <tr>
      <th>93</th>
      <td>google</td>
      <td>would</td>
      <td>``</td>
      <td>(</td>
      <td>season</td>
      <td>''</td>
      <td>]</td>
      <td>korea</td>
    </tr>
    <tr>
      <th>94</th>
      <td>android</td>
      <td>trump</td>
      <td>''</td>
      <td>)</td>
      <td>like</td>
      <td>gt</td>
      <td>people</td>
      <td>china</td>
    </tr>
    <tr>
      <th>95</th>
      <td>(</td>
      <td>``</td>
      <td>government</td>
      <td>n't</td>
      <td>)</td>
      <td>people</td>
      <td>(</td>
      <td>would</td>
    </tr>
    <tr>
      <th>96</th>
      <td>)</td>
      <td>''</td>
      <td>gt</td>
      <td>play</td>
      <td>year</td>
      <td>(</td>
      <td>)</td>
      <td>nk</td>
    </tr>
    <tr>
      <th>97</th>
      <td>n't</td>
      <td>people</td>
      <td>people</td>
      <td>'s</td>
      <td>team</td>
      <td>)</td>
      <td>n't</td>
      <td>us</td>
    </tr>
    <tr>
      <th>98</th>
      <td>'s</td>
      <td>'s</td>
      <td>'s</td>
      <td>games</td>
      <td>n't</td>
      <td>n't</td>
      <td>trump</td>
      <td>n't</td>
    </tr>
    <tr>
      <th>99</th>
      <td>phone</td>
      <td>n't</td>
      <td>n't</td>
      <td>game</td>
      <td>'s</td>
      <td>'s</td>
      <td>'s</td>
      <td>'s</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>




```python
reddit_top_unigrams_df.to_csv("csv_files\\reddit_top_unigrams.csv")
```

# heatmap for previous model

The heatmap for the previous model looks pretty good.  The control subreddits (hockye, boardgames and Android) are very well defined.  The confusion occurs around the political posts, which would be expected since they would be talking about very similar topics.  


```python
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Android','Conservative','Libertarian','boardgames','hockey','neoliberal','politics','worldnews'],
            yticklabels=['Android','Conservative','Libertarian','boardgames','hockey','neoliberal','politics','worldnews'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_21_0.png)


# repeat above model with bigrams

The next couple blocks, I did the same as above but with bigrams.  I did not recreate the large subreddit_df though, I was able to reuse that dataframe through the rest of the program.


```python
reddit_vec = CountVectorizer(ngram_range=(2,2),min_df=16, lowercase=True, tokenizer=nltk.word_tokenize,stop_words=fullset)
z=0
start=0
iterator=0
while(iterator<1):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = reddit_df['subreddit']

    reddit_counts = reddit_vec.fit_transform(reddit_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.6095125


# accuracy

As you can see above, the accuracy for this model is not as good as the unigram model.  the unigram had an accuracy of 69.5% while this model only has 60.9%.


```python
reddit_top_bigrams_df = return_top100(reddit_vec, classifier, class_labels=np.unique(y_train))
reddit_top_bigrams_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Android</th>
      <th>Conservative</th>
      <th>Libertarian</th>
      <th>boardgames</th>
      <th>hockey</th>
      <th>neoliberal</th>
      <th>politics</th>
      <th>worldnews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wiki_2._we_welcome_discussion-promoting_posts_...</td>
      <td>'re saying</td>
      <td>`` n't</td>
      <td>'s still</td>
      <td>'s great</td>
      <td>'re talking</td>
      <td>'s going</td>
      <td>/r/worldnews ]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>//www.reddit.com/r/android/wiki/rulesandregs w...</td>
      <td>president trump</td>
      <td>saying ``</td>
      <td>blood rage</td>
      <td>st. louis</td>
      <td>n't need</td>
      <td>fox news</td>
      <td>whatconstitutesspam )</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ebay ]</td>
      <td>illegal immigrants</td>
      <td>would like</td>
      <td>people n't</td>
      <td>round pick</td>
      <td>sounds like</td>
      <td>could n't</td>
      <td>http //www.reddit.com/r/help/comments/2bx3cj/r...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tracker ]</td>
      <td>like ``</td>
      <td>gon na</td>
      <td>'m looking</td>
      <td>player 's</td>
      <td>look like</td>
      <td>fake news</td>
      <td>//www.redditblog.com/2014/07/how-reddit-works....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>client [</td>
      <td>lot people</td>
      <td>) n't</td>
      <td>mage knight</td>
      <td>'s way</td>
      <td>n't believe</td>
      <td>'' ``</td>
      <td>pending manual</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[ aosp</td>
      <td>'s good</td>
      <td>black people</td>
      <td>deck building</td>
      <td>nhl team</td>
      <td>know 's</td>
      <td>united states</td>
      <td>self-promotion reddit</td>
    </tr>
    <tr>
      <th>6</th>
      <td>app recommendations</td>
      <td>'s ``</td>
      <td>gt 're</td>
      <td>even though</td>
      <td>top 4</td>
      <td>lot people</td>
      <td>donald trump</td>
      <td>domain new</td>
    </tr>
    <tr>
      <th>7</th>
      <td>selling buying</td>
      <td>first place</td>
      <td>taxation theft</td>
      <td>7 wonders</td>
      <td>'' ``</td>
      <td>n't say</td>
      <td>'s like</td>
      <td>minimum karma</td>
    </tr>
    <tr>
      <th>8</th>
      <td>might interested</td>
      <td>'s right</td>
      <td>'s ``</td>
      <td>video games</td>
      <td>n't going</td>
      <td>n't matter</td>
      <td>people n't</td>
      <td>requirements /r/worldnews</td>
    </tr>
    <tr>
      <th>9</th>
      <td>google play</td>
      <td>white supremacy</td>
      <td>'re saying</td>
      <td>games 've</td>
      <td>pretty sure</td>
      <td>'s going</td>
      <td>let 's</td>
      <td>account meet</td>
    </tr>
    <tr>
      <th>10</th>
      <td>google assistant</td>
      <td>freedom speech</td>
      <td>makes sense</td>
      <td>arkham horror</td>
      <td>red wings</td>
      <td>'' 's</td>
      <td>think 's</td>
      <td>regarding spam</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fingerprint scanner</td>
      <td>something like</td>
      <td>) [</td>
      <td>played game</td>
      <td>every time</td>
      <td>income tax</td>
      <td>n't want</td>
      <td>reddit 101</td>
    </tr>
    <tr>
      <th>12</th>
      <td>even though</td>
      <td>https //www.youtube.com/watch</td>
      <td>gt 'm</td>
      <td>n't see</td>
      <td>'s also</td>
      <td>minimum wage</td>
      <td>n't get</td>
      <td>manual approval</td>
    </tr>
    <tr>
      <th>13</th>
      <td>use phone</td>
      <td>'re right</td>
      <td>n't give</td>
      <td>playing game</td>
      <td>new arena</td>
      <td>'re going</td>
      <td>regarding removal</td>
      <td>reddit guidelines</td>
    </tr>
    <tr>
      <th>14</th>
      <td>web browser</td>
      <td>health care</td>
      <td>'m going</td>
      <td>really enjoy</td>
      <td>team n't</td>
      <td>people think</td>
      <td>'m sure</td>
      <td>may also</td>
    </tr>
    <tr>
      <th>15</th>
      <td>subreddit ]</td>
      <td>'s really</td>
      <td>gt would</td>
      <td>people play</td>
      <td>n't play</td>
      <td>global poor</td>
      <td>white house</td>
      <td>reddit suggest</td>
    </tr>
    <tr>
      <th>16</th>
      <td>submission removed</td>
      <td>trump supporters</td>
      <td>single payer</td>
      <td>new games</td>
      <td>4th line</td>
      <td>first place</td>
      <td>n't even</td>
      <td>meet minimum</td>
    </tr>
    <tr>
      <th>17</th>
      <td>years ago</td>
      <td>'re talking</td>
      <td>`` ''</td>
      <td>new game</td>
      <td>really n't</td>
      <td>bernie 's</td>
      <td>n't think</td>
      <td>account age</td>
    </tr>
    <tr>
      <th>18</th>
      <td>seems like</td>
      <td>north korea</td>
      <td>'s really</td>
      <td>games 's</td>
      <td>win cup</td>
      <td>bernie sanders</td>
      <td>wo n't</td>
      <td>'' ``</td>
    </tr>
    <tr>
      <th>19</th>
      <td>iphone 8</td>
      <td>liberal media</td>
      <td>sounds like</td>
      <td>game really</td>
      <td>white house</td>
      <td>'ve seen</td>
      <td>n't know</td>
      <td>works ]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>gon na</td>
      <td>left 's</td>
      <td>[ ]</td>
      <td>'s like</td>
      <td>'s got</td>
      <td>'s really</td>
      <td>would n't</td>
      <td>reddit works</td>
    </tr>
    <tr>
      <th>21</th>
      <td>'s still</td>
      <td>'s pretty</td>
      <td>ron paul</td>
      <td>player count</td>
      <td>nhl players</td>
      <td>saying ``</td>
      <td>( http</td>
      <td>http //www.reddit.com/r/worldnews/wiki/rules</td>
    </tr>
    <tr>
      <th>22</th>
      <td>http //www.reddit.com/r/android/wiki/rulesandregs</td>
      <td>saying ``</td>
      <td>health insurance</td>
      <td>2 player</td>
      <td>make playoffs</td>
      <td>) n't</td>
      <td>see [</td>
      <td>reddit ]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>removed gt</td>
      <td>free market</td>
      <td>people like</td>
      <td>every game</td>
      <td>every year</td>
      <td>black people</td>
      <td>[ faq</td>
      <td>karma account</td>
    </tr>
    <tr>
      <th>24</th>
      <td>//www.reddit.com/message/compose to=</td>
      <td>people would</td>
      <td>economic system</td>
      <td>'m going</td>
      <td>-- --</td>
      <td>something like</td>
      <td>faq ]</td>
      <td>also want</td>
    </tr>
    <tr>
      <th>25</th>
      <td>to= 2fr</td>
      <td>obama 's</td>
      <td>years ago</td>
      <td>love game</td>
      <td>two years</td>
      <td>good thing</td>
      <td>[ post</td>
      <td>suggest read</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2fr 2fandroid</td>
      <td>things like</td>
      <td>n't exist</td>
      <td>make sure</td>
      <td>'s one</td>
      <td>'' n't</td>
      <td>discussion ]</td>
      <td>n't make</td>
    </tr>
    <tr>
      <th>27</th>
      <td>feel like</td>
      <td>'ve seen</td>
      <td>gt gt</td>
      <td>'s game</td>
      <td>n't good</td>
      <td>free market</td>
      <td>) general</td>
      <td>read [</td>
    </tr>
    <tr>
      <th>28</th>
      <td>think 's</td>
      <td>civil rights</td>
      <td>n't believe</td>
      <td>'s one</td>
      <td>top 10</td>
      <td>really n't</td>
      <td>) current</td>
      <td>n't need</td>
    </tr>
    <tr>
      <th>29</th>
      <td>last year</td>
      <td>gt n't</td>
      <td>people want</td>
      <td>feels like</td>
      <td>got ta</td>
      <td>democratic party</td>
      <td>current research</td>
      <td>kim jong</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>android phone</td>
      <td>identity politics</td>
      <td>think 's</td>
      <td>game (</td>
      <td>feel like</td>
      <td>united states</td>
      <td>downvotes comments</td>
      <td>let 's</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2 years</td>
      <td>civil war</td>
      <td>n't see</td>
      <td>great game</td>
      <td>https //www.youtube.com/watch</td>
      <td>pretty much</td>
      <td>hate speech</td>
      <td>nk 's</td>
    </tr>
    <tr>
      <th>72</th>
      <td>phone 's</td>
      <td>n't really</td>
      <td>free speech</td>
      <td>card game</td>
      <td>) [</td>
      <td>n't like</td>
      <td>personal insults</td>
      <td>think 's</td>
    </tr>
    <tr>
      <th>73</th>
      <td>would like</td>
      <td>n't make</td>
      <td>gt 's</td>
      <td>worker placement</td>
      <td>'s going</td>
      <td>n't see</td>
      <td>) [</td>
      <td>| [</td>
    </tr>
    <tr>
      <th>74</th>
      <td>n't use</td>
      <td>black people</td>
      <td>people 's</td>
      <td>want play</td>
      <td>[ [</td>
      <td>) [</td>
      <td>please see</td>
      <td>/message/compose/ to=/r/worldnews</td>
    </tr>
    <tr>
      <th>75</th>
      <td>'m sure</td>
      <td>white people</td>
      <td>n't make</td>
      <td>looks like</td>
      <td>seems like</td>
      <td>free trade</td>
      <td>comments section</td>
      <td>'s like</td>
    </tr>
    <tr>
      <th>76</th>
      <td>n't get</td>
      <td>n't care</td>
      <td>'' ``</td>
      <td>play game</td>
      <td>could n't</td>
      <td>https //www.youtube.com/watch</td>
      <td>see comments</td>
      <td>'' )</td>
    </tr>
    <tr>
      <th>77</th>
      <td>google 's</td>
      <td>n't see</td>
      <td>let 's</td>
      <td>games n't</td>
      <td>] ]</td>
      <td>gt gt</td>
      <td>please report</td>
      <td>years ago</td>
    </tr>
    <tr>
      <th>78</th>
      <td>pretty much</td>
      <td>n't mean</td>
      <td>gt n't</td>
      <td>first time</td>
      <td>team 's</td>
      <td>\ gt</td>
      <td>[ civil</td>
      <td>'m sure</td>
    </tr>
    <tr>
      <th>79</th>
      <td>n't want</td>
      <td>united states</td>
      <td>n't like</td>
      <td>'m sure</td>
      <td>regular season</td>
      <td>'m sure</td>
      <td>post ]</td>
      <td>contact moderators</td>
    </tr>
    <tr>
      <th>80</th>
      <td>n't really</td>
      <td>'' ``</td>
      <td>'m sure</td>
      <td>n't get</td>
      <td>'s like</td>
      <td>gon na</td>
      <td>civil discussion</td>
      <td>to=/r/worldnews )</td>
    </tr>
    <tr>
      <th>81</th>
      <td>n't think</td>
      <td>white supremacists</td>
      <td>means production</td>
      <td>n't like</td>
      <td>years ago</td>
      <td>n't get</td>
      <td>please [</td>
      <td>china 's</td>
    </tr>
    <tr>
      <th>82</th>
      <td>apple 's</td>
      <td>people n't</td>
      <td>property rights</td>
      <td>think 's</td>
      <td>n't really</td>
      <td>people n't</td>
      <td>subreddit ]</td>
      <td>n't even</td>
    </tr>
    <tr>
      <th>83</th>
      <td>n't even</td>
      <td>let 's</td>
      <td>n't get</td>
      <td>n't played</td>
      <td>n't want</td>
      <td>'' ``</td>
      <td>[ contact</td>
      <td>united states</td>
    </tr>
    <tr>
      <th>84</th>
      <td>stock android</td>
      <td>'m sure</td>
      <td>( http</td>
      <td>wo n't</td>
      <td>'m sure</td>
      <td>open borders</td>
      <td>*i bot</td>
      <td>north korean</td>
    </tr>
    <tr>
      <th>85</th>
      <td>iphone x</td>
      <td>n't get</td>
      <td>n't mean</td>
      <td>would n't</td>
      <td>n't even</td>
      <td>n't even</td>
      <td>action performed</td>
      <td>) |</td>
    </tr>
    <tr>
      <th>86</th>
      <td>n't know</td>
      <td>n't like</td>
      <td>private property</td>
      <td>n't really</td>
      <td>n't see</td>
      <td>n't want</td>
      <td>questions concerns</td>
      <td>n't think</td>
    </tr>
    <tr>
      <th>87</th>
      <td>would n't</td>
      <td>( http</td>
      <td>n't even</td>
      <td>games like</td>
      <td>n't get</td>
      <td>single payer</td>
      <td>moderators subreddit</td>
      <td>n't know</td>
    </tr>
    <tr>
      <th>88</th>
      <td>pixel xl</td>
      <td>think 's</td>
      <td>people n't</td>
      <td>n't know</td>
      <td>n't know</td>
      <td>n't really</td>
      <td>bot action</td>
      <td>n't want</td>
    </tr>
    <tr>
      <th>89</th>
      <td>pixel 2</td>
      <td>n't want</td>
      <td>n't know</td>
      <td>n't think</td>
      <td>think 's</td>
      <td>[ ]</td>
      <td>performed automatically</td>
      <td>) [</td>
    </tr>
    <tr>
      <th>90</th>
      <td>wireless charging</td>
      <td>n't even</td>
      <td>wo n't</td>
      <td>game n't</td>
      <td>gon na</td>
      <td>wo n't</td>
      <td>contact moderators</td>
      <td>[ reddit</td>
    </tr>
    <tr>
      <th>91</th>
      <td>) [</td>
      <td>trump 's</td>
      <td>n't want</td>
      <td>feel like</td>
      <td>wo n't</td>
      <td>think 's</td>
      <td>automatically please</td>
      <td>wo n't</td>
    </tr>
    <tr>
      <th>92</th>
      <td>wo n't</td>
      <td>free speech</td>
      <td>^| [</td>
      <td>base game</td>
      <td>( http</td>
      <td>n't know</td>
      <td>( /message/compose/</td>
      <td>( https</td>
    </tr>
    <tr>
      <th>93</th>
      <td>note 8</td>
      <td>n't know</td>
      <td>) ^|</td>
      <td>game 's</td>
      <td>would n't</td>
      <td>would n't</td>
      <td>) questions</td>
      <td>would n't</td>
    </tr>
    <tr>
      <th>94</th>
      <td>battery life</td>
      <td>wo n't</td>
      <td>n't think</td>
      <td>'ve played</td>
      <td>last season</td>
      <td>n't think</td>
      <td>trump 's</td>
      <td>nuclear weapons</td>
    </tr>
    <tr>
      <th>95</th>
      <td>ca n't</td>
      <td>( https</td>
      <td>would n't</td>
      <td>ca n't</td>
      <td>n't think</td>
      <td>hot take</td>
      <td>to=/r/politics )</td>
      <td>south korea</td>
    </tr>
    <tr>
      <th>96</th>
      <td>headphone jack</td>
      <td>n't think</td>
      <td>free market</td>
      <td>board games</td>
      <td>ca n't</td>
      <td>( http</td>
      <td>/message/compose/ to=/r/politics</td>
      <td>ca n't</td>
    </tr>
    <tr>
      <th>97</th>
      <td>( https</td>
      <td>would n't</td>
      <td>( https</td>
      <td>( https</td>
      <td>last year</td>
      <td>ca n't</td>
      <td>ca n't</td>
      <td>( http</td>
    </tr>
    <tr>
      <th>98</th>
      <td>( http</td>
      <td>] (</td>
      <td>ca n't</td>
      <td>board game</td>
      <td>( https</td>
      <td>( https</td>
      <td>( https</td>
      <td>] (</td>
    </tr>
    <tr>
      <th>99</th>
      <td>] (</td>
      <td>ca n't</td>
      <td>] (</td>
      <td>] (</td>
      <td>] (</td>
      <td>] (</td>
      <td>] (</td>
      <td>north korea</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>



# save csv file

I save the dataframe as a csv file.  While adding the markdown file, I accidently ran the above box again and had to quit it so is looks like it wasn't run.  


```python
reddit_top_bigrams_df.to_csv("csv_files\\reddit_top_bigrams.csv")
```

# heatmap

heatmap looks similar to the other model.  The confusion is still happening in the same areas.


```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Android','Conservative','Libertarian','boardgames','hockey','neoliberal','politics','worldnews'],
            yticklabels=['Android','Conservative','Libertarian','boardgames','hockey','neoliberal','politics','worldnews'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_29_0.png)


# explore the posts scores

I printed the mean, max and min for all of the scores in the reddit_df.  As you can see, the values are spread accross a very large margin (-274 to 17,193) but the mean is rather low (only about 9).  I wanted to find a way to split the posts by score so that I would have the same number of posts in each group.  I began by trying to split by percentile, but that did not work as well as I had hoped (the column is still named percentile).  The solution I came up with is much simpler.  I seperated the reddit_df into 8 smaller dataframes, 1 for each subreddit represented.  I then sorted them and split them into thirds.


```python
print(reddit_df['score'].mean())
print(reddit_df['score'].max())
print(reddit_df['score'].min())
```

    8.9692925
    17193.0
    -274.0



```python
Android_df = reddit_df[reddit_df.subreddit == 'Android']
```


```python
Android_df = Android_df.sort_values(['score'])
```


```python
Android_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>percentile</th>
      <th>score</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10040</th>
      <td>1. First world problems 2. If it is that incon...</td>
      <td>NaN</td>
      <td>-260.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49434</th>
      <td>&amp;gt;In India whenever there is any conflict, t...</td>
      <td>NaN</td>
      <td>-90.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>4665</th>
      <td>Oh, yeah? In what way this cheap peace of Chin...</td>
      <td>NaN</td>
      <td>-79.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>4155</th>
      <td>I can't stand spongebob memes. Maybe because I...</td>
      <td>NaN</td>
      <td>-78.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3588</th>
      <td>All OLEDs are trash. Can't believe people stil...</td>
      <td>NaN</td>
      <td>-71.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>12785</th>
      <td>Google as a whole is a garbage company. They'v...</td>
      <td>NaN</td>
      <td>-69.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>13705</th>
      <td>Or...I don't know....quit relying on cellular ...</td>
      <td>NaN</td>
      <td>-68.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>33143</th>
      <td>Having a headphone jack has nothing to do with...</td>
      <td>NaN</td>
      <td>-62.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>17144</th>
      <td>No root required for theming, but now you have...</td>
      <td>NaN</td>
      <td>-60.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>36153</th>
      <td>Being a techy is absolutely no reason to buy a...</td>
      <td>NaN</td>
      <td>-57.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>5437</th>
      <td>So if I search for "brain blocks", a game call...</td>
      <td>NaN</td>
      <td>-57.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>44594</th>
      <td>The difference is that Google makes Android an...</td>
      <td>NaN</td>
      <td>-56.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>14156</th>
      <td>Tell that to everyone who has been getting alo...</td>
      <td>NaN</td>
      <td>-54.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>11253</th>
      <td>I gotta say, if I saw this on a spec sheet wit...</td>
      <td>NaN</td>
      <td>-54.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>31217</th>
      <td>I wouldn’t worry too much about it.  Samsung l...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>33591</th>
      <td>Omg shut up. Many of us have been using since ...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49801</th>
      <td>&amp;gt; Then there is the question of size: only ...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>29820</th>
      <td>I wouldn’t worry too much about it.  Samsung l...</td>
      <td>NaN</td>
      <td>-47.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3615</th>
      <td>Burn in isn't even the biggest issue with OLED...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>5884</th>
      <td>No it doesn't. If you can't afford the phone u...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>32951</th>
      <td>The Essential Phone was announced on May 30. T...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>30906</th>
      <td>That’s because they are incorrectly counting t...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>20017</th>
      <td>Might as well compare it to the $40 Verizon mo...</td>
      <td>NaN</td>
      <td>-44.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>11075</th>
      <td>It's amazing I never felt like I needed a rest...</td>
      <td>NaN</td>
      <td>-43.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>29508</th>
      <td>That’s because they are incorrectly counting t...</td>
      <td>NaN</td>
      <td>-41.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>31429</th>
      <td>For a subreddit that seems to have a group men...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49265</th>
      <td>Apple A 11 Bionic is a **TRUE 6 CORES** cpu, i...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>4277</th>
      <td>You know what? Samsung IS the BIGGEST Android ...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19532</th>
      <td>You're comparing geekebench results from iOS t...</td>
      <td>NaN</td>
      <td>-39.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19964</th>
      <td>This comment makes no sense, unless you mean t...</td>
      <td>NaN</td>
      <td>-38.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19275</th>
      <td>A little disappointing. While I welcome its ex...</td>
      <td>NaN</td>
      <td>1062.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>24302</th>
      <td>You would think you'd get a notification/email...</td>
      <td>NaN</td>
      <td>1069.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3280</th>
      <td>Lineage OS:\n\nSettings - buttons - volume but...</td>
      <td>NaN</td>
      <td>1088.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49873</th>
      <td>Android 9.0 is supposed to come out next year,...</td>
      <td>NaN</td>
      <td>1147.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3430</th>
      <td>And the funniest thing is that thoughout your ...</td>
      <td>NaN</td>
      <td>1161.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>4746</th>
      <td>Yeah a lot of people I know like to compare DS...</td>
      <td>NaN</td>
      <td>1183.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>33491</th>
      <td>It's so sad that people actually fall for Clea...</td>
      <td>NaN</td>
      <td>1190.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>39378</th>
      <td>Wish Google would have kept the Nexus line up ...</td>
      <td>NaN</td>
      <td>1191.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49863</th>
      <td>Why is everyone afraid of 9\n\n\n\nEdit: I get...</td>
      <td>NaN</td>
      <td>1316.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>45395</th>
      <td>Only if it were news... They explicitly state ...</td>
      <td>NaN</td>
      <td>1326.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19605</th>
      <td>I agree. It's not like the essential phone whe...</td>
      <td>NaN</td>
      <td>1332.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3349</th>
      <td>You're going to sit there and I'm not going to...</td>
      <td>NaN</td>
      <td>1339.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19362</th>
      <td>They don't black out the sides of the camera c...</td>
      <td>NaN</td>
      <td>1370.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>20528</th>
      <td>Lots of old iOS apps would test the OS version...</td>
      <td>NaN</td>
      <td>1517.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>18459</th>
      <td>During i/o\n\n\n\nr/Android: OH FFS WE DIDN'T ...</td>
      <td>NaN</td>
      <td>1646.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>10790</th>
      <td>/r/Android:\n\n"wow that would be stupid, Goog...</td>
      <td>NaN</td>
      <td>1717.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19663</th>
      <td>I think this image says it all: https://pbs.tw...</td>
      <td>NaN</td>
      <td>1762.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19536</th>
      <td>You can see (or rather not see) it here: https...</td>
      <td>NaN</td>
      <td>1902.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>39208</th>
      <td>That price point is atrocious. After using my ...</td>
      <td>NaN</td>
      <td>1955.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>21432</th>
      <td>"I buy Apple products for the Apple 'Ecosystem...</td>
      <td>NaN</td>
      <td>2108.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19289</th>
      <td>If I was an iPhone fan, I think I'd be upset t...</td>
      <td>NaN</td>
      <td>2170.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19287</th>
      <td>Am I mistaken or does video playback actually ...</td>
      <td>NaN</td>
      <td>2961.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>39222</th>
      <td>Phone prices just keep on going up at the top ...</td>
      <td>NaN</td>
      <td>3154.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19298</th>
      <td>The FaceID tech looks way slower than TouchID....</td>
      <td>NaN</td>
      <td>3465.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>41344</th>
      <td>HTC: "Surprise! We sold the company to Lenovo ...</td>
      <td>NaN</td>
      <td>3502.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19458</th>
      <td>Not iPhone X, but I think it was good for Appl...</td>
      <td>NaN</td>
      <td>3693.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19834</th>
      <td>2016: We have the courage to remove the headph...</td>
      <td>NaN</td>
      <td>3889.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19314</th>
      <td>I think the fact that animojis is something Ap...</td>
      <td>NaN</td>
      <td>5445.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3293</th>
      <td>Except it's not, because what should happen at...</td>
      <td>NaN</td>
      <td>6781.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19290</th>
      <td>Pretty much what the leaks said it would be. T...</td>
      <td>NaN</td>
      <td>9142.0</td>
      <td>Android</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 4 columns</p>
</div>



# reindex

I had to reindex the dataframes.


```python
Android_df = Android_df.reset_index(drop=True)
Android_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>percentile</th>
      <th>score</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1. First world problems 2. If it is that incon...</td>
      <td>NaN</td>
      <td>-260.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&amp;gt;In India whenever there is any conflict, t...</td>
      <td>NaN</td>
      <td>-90.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Oh, yeah? In what way this cheap peace of Chin...</td>
      <td>NaN</td>
      <td>-79.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I can't stand spongebob memes. Maybe because I...</td>
      <td>NaN</td>
      <td>-78.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>4</th>
      <td>All OLEDs are trash. Can't believe people stil...</td>
      <td>NaN</td>
      <td>-71.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Google as a whole is a garbage company. They'v...</td>
      <td>NaN</td>
      <td>-69.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Or...I don't know....quit relying on cellular ...</td>
      <td>NaN</td>
      <td>-68.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Having a headphone jack has nothing to do with...</td>
      <td>NaN</td>
      <td>-62.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>8</th>
      <td>No root required for theming, but now you have...</td>
      <td>NaN</td>
      <td>-60.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Being a techy is absolutely no reason to buy a...</td>
      <td>NaN</td>
      <td>-57.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>10</th>
      <td>So if I search for "brain blocks", a game call...</td>
      <td>NaN</td>
      <td>-57.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The difference is that Google makes Android an...</td>
      <td>NaN</td>
      <td>-56.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tell that to everyone who has been getting alo...</td>
      <td>NaN</td>
      <td>-54.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>13</th>
      <td>I gotta say, if I saw this on a spec sheet wit...</td>
      <td>NaN</td>
      <td>-54.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>14</th>
      <td>I wouldn’t worry too much about it.  Samsung l...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Omg shut up. Many of us have been using since ...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>16</th>
      <td>&amp;gt; Then there is the question of size: only ...</td>
      <td>NaN</td>
      <td>-48.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>17</th>
      <td>I wouldn’t worry too much about it.  Samsung l...</td>
      <td>NaN</td>
      <td>-47.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Burn in isn't even the biggest issue with OLED...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>19</th>
      <td>No it doesn't. If you can't afford the phone u...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>20</th>
      <td>The Essential Phone was announced on May 30. T...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>21</th>
      <td>That’s because they are incorrectly counting t...</td>
      <td>NaN</td>
      <td>-45.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Might as well compare it to the $40 Verizon mo...</td>
      <td>NaN</td>
      <td>-44.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>23</th>
      <td>It's amazing I never felt like I needed a rest...</td>
      <td>NaN</td>
      <td>-43.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>24</th>
      <td>That’s because they are incorrectly counting t...</td>
      <td>NaN</td>
      <td>-41.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>25</th>
      <td>For a subreddit that seems to have a group men...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Apple A 11 Bionic is a **TRUE 6 CORES** cpu, i...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>27</th>
      <td>You know what? Samsung IS the BIGGEST Android ...</td>
      <td>NaN</td>
      <td>-40.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>28</th>
      <td>You're comparing geekebench results from iOS t...</td>
      <td>NaN</td>
      <td>-39.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>29</th>
      <td>This comment makes no sense, unless you mean t...</td>
      <td>NaN</td>
      <td>-38.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49970</th>
      <td>A little disappointing. While I welcome its ex...</td>
      <td>NaN</td>
      <td>1062.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49971</th>
      <td>You would think you'd get a notification/email...</td>
      <td>NaN</td>
      <td>1069.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49972</th>
      <td>Lineage OS:\n\nSettings - buttons - volume but...</td>
      <td>NaN</td>
      <td>1088.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49973</th>
      <td>Android 9.0 is supposed to come out next year,...</td>
      <td>NaN</td>
      <td>1147.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49974</th>
      <td>And the funniest thing is that thoughout your ...</td>
      <td>NaN</td>
      <td>1161.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49975</th>
      <td>Yeah a lot of people I know like to compare DS...</td>
      <td>NaN</td>
      <td>1183.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49976</th>
      <td>It's so sad that people actually fall for Clea...</td>
      <td>NaN</td>
      <td>1190.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49977</th>
      <td>Wish Google would have kept the Nexus line up ...</td>
      <td>NaN</td>
      <td>1191.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49978</th>
      <td>Why is everyone afraid of 9\n\n\n\nEdit: I get...</td>
      <td>NaN</td>
      <td>1316.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49979</th>
      <td>Only if it were news... They explicitly state ...</td>
      <td>NaN</td>
      <td>1326.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49980</th>
      <td>I agree. It's not like the essential phone whe...</td>
      <td>NaN</td>
      <td>1332.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49981</th>
      <td>You're going to sit there and I'm not going to...</td>
      <td>NaN</td>
      <td>1339.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49982</th>
      <td>They don't black out the sides of the camera c...</td>
      <td>NaN</td>
      <td>1370.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49983</th>
      <td>Lots of old iOS apps would test the OS version...</td>
      <td>NaN</td>
      <td>1517.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49984</th>
      <td>During i/o\n\n\n\nr/Android: OH FFS WE DIDN'T ...</td>
      <td>NaN</td>
      <td>1646.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49985</th>
      <td>/r/Android:\n\n"wow that would be stupid, Goog...</td>
      <td>NaN</td>
      <td>1717.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49986</th>
      <td>I think this image says it all: https://pbs.tw...</td>
      <td>NaN</td>
      <td>1762.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49987</th>
      <td>You can see (or rather not see) it here: https...</td>
      <td>NaN</td>
      <td>1902.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49988</th>
      <td>That price point is atrocious. After using my ...</td>
      <td>NaN</td>
      <td>1955.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49989</th>
      <td>"I buy Apple products for the Apple 'Ecosystem...</td>
      <td>NaN</td>
      <td>2108.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49990</th>
      <td>If I was an iPhone fan, I think I'd be upset t...</td>
      <td>NaN</td>
      <td>2170.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49991</th>
      <td>Am I mistaken or does video playback actually ...</td>
      <td>NaN</td>
      <td>2961.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49992</th>
      <td>Phone prices just keep on going up at the top ...</td>
      <td>NaN</td>
      <td>3154.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49993</th>
      <td>The FaceID tech looks way slower than TouchID....</td>
      <td>NaN</td>
      <td>3465.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49994</th>
      <td>HTC: "Surprise! We sold the company to Lenovo ...</td>
      <td>NaN</td>
      <td>3502.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>Not iPhone X, but I think it was good for Appl...</td>
      <td>NaN</td>
      <td>3693.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>2016: We have the courage to remove the headph...</td>
      <td>NaN</td>
      <td>3889.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>I think the fact that animojis is something Ap...</td>
      <td>NaN</td>
      <td>5445.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>Except it's not, because what should happen at...</td>
      <td>NaN</td>
      <td>6781.0</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>Pretty much what the leaks said it would be. T...</td>
      <td>NaN</td>
      <td>9142.0</td>
      <td>Android</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 4 columns</p>
</div>



# set percentile

I set an interger value in the percentile column.  The bottom third are 1, middle third 2 and top third 3.


```python
third = (len(Android_df)/3)
iterator = 0
for x in Android_df.percentile:
    if(iterator < third):
        Android_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        Android_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        Android_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
print(Android_df)
```

                                                        body  percentile   score  \
    0      1. First world problems 2. If it is that incon...         1.0  -260.0   
    1      &gt;In India whenever there is any conflict, t...         1.0   -90.0   
    2      Oh, yeah? In what way this cheap peace of Chin...         1.0   -79.0   
    3      I can't stand spongebob memes. Maybe because I...         1.0   -78.0   
    4      All OLEDs are trash. Can't believe people stil...         1.0   -71.0   
    5      Google as a whole is a garbage company. They'v...         1.0   -69.0   
    6      Or...I don't know....quit relying on cellular ...         1.0   -68.0   
    7      Having a headphone jack has nothing to do with...         1.0   -62.0   
    8      No root required for theming, but now you have...         1.0   -60.0   
    9      Being a techy is absolutely no reason to buy a...         1.0   -57.0   
    10     So if I search for "brain blocks", a game call...         1.0   -57.0   
    11     The difference is that Google makes Android an...         1.0   -56.0   
    12     Tell that to everyone who has been getting alo...         1.0   -54.0   
    13     I gotta say, if I saw this on a spec sheet wit...         1.0   -54.0   
    14     I wouldn’t worry too much about it.  Samsung l...         1.0   -48.0   
    15     Omg shut up. Many of us have been using since ...         1.0   -48.0   
    16     &gt; Then there is the question of size: only ...         1.0   -48.0   
    17     I wouldn’t worry too much about it.  Samsung l...         1.0   -47.0   
    18     Burn in isn't even the biggest issue with OLED...         1.0   -45.0   
    19     No it doesn't. If you can't afford the phone u...         1.0   -45.0   
    20     The Essential Phone was announced on May 30. T...         1.0   -45.0   
    21     That’s because they are incorrectly counting t...         1.0   -45.0   
    22     Might as well compare it to the $40 Verizon mo...         1.0   -44.0   
    23     It's amazing I never felt like I needed a rest...         1.0   -43.0   
    24     That’s because they are incorrectly counting t...         1.0   -41.0   
    25     For a subreddit that seems to have a group men...         1.0   -40.0   
    26     Apple A 11 Bionic is a **TRUE 6 CORES** cpu, i...         1.0   -40.0   
    27     You know what? Samsung IS the BIGGEST Android ...         1.0   -40.0   
    28     You're comparing geekebench results from iOS t...         1.0   -39.0   
    29     This comment makes no sense, unless you mean t...         1.0   -38.0   
    ...                                                  ...         ...     ...   
    49970  A little disappointing. While I welcome its ex...         3.0  1062.0   
    49971  You would think you'd get a notification/email...         3.0  1069.0   
    49972  Lineage OS:\n\nSettings - buttons - volume but...         3.0  1088.0   
    49973  Android 9.0 is supposed to come out next year,...         3.0  1147.0   
    49974  And the funniest thing is that thoughout your ...         3.0  1161.0   
    49975  Yeah a lot of people I know like to compare DS...         3.0  1183.0   
    49976  It's so sad that people actually fall for Clea...         3.0  1190.0   
    49977  Wish Google would have kept the Nexus line up ...         3.0  1191.0   
    49978  Why is everyone afraid of 9\n\n\n\nEdit: I get...         3.0  1316.0   
    49979  Only if it were news... They explicitly state ...         3.0  1326.0   
    49980  I agree. It's not like the essential phone whe...         3.0  1332.0   
    49981  You're going to sit there and I'm not going to...         3.0  1339.0   
    49982  They don't black out the sides of the camera c...         3.0  1370.0   
    49983  Lots of old iOS apps would test the OS version...         3.0  1517.0   
    49984  During i/o\n\n\n\nr/Android: OH FFS WE DIDN'T ...         3.0  1646.0   
    49985  /r/Android:\n\n"wow that would be stupid, Goog...         3.0  1717.0   
    49986  I think this image says it all: https://pbs.tw...         3.0  1762.0   
    49987  You can see (or rather not see) it here: https...         3.0  1902.0   
    49988  That price point is atrocious. After using my ...         3.0  1955.0   
    49989  "I buy Apple products for the Apple 'Ecosystem...         3.0  2108.0   
    49990  If I was an iPhone fan, I think I'd be upset t...         3.0  2170.0   
    49991  Am I mistaken or does video playback actually ...         3.0  2961.0   
    49992  Phone prices just keep on going up at the top ...         3.0  3154.0   
    49993  The FaceID tech looks way slower than TouchID....         3.0  3465.0   
    49994  HTC: "Surprise! We sold the company to Lenovo ...         3.0  3502.0   
    49995  Not iPhone X, but I think it was good for Appl...         3.0  3693.0   
    49996  2016: We have the courage to remove the headph...         3.0  3889.0   
    49997  I think the fact that animojis is something Ap...         3.0  5445.0   
    49998  Except it's not, because what should happen at...         3.0  6781.0   
    49999  Pretty much what the leaks said it would be. T...         3.0  9142.0   

          subreddit  
    0       Android  
    1       Android  
    2       Android  
    3       Android  
    4       Android  
    5       Android  
    6       Android  
    7       Android  
    8       Android  
    9       Android  
    10      Android  
    11      Android  
    12      Android  
    13      Android  
    14      Android  
    15      Android  
    16      Android  
    17      Android  
    18      Android  
    19      Android  
    20      Android  
    21      Android  
    22      Android  
    23      Android  
    24      Android  
    25      Android  
    26      Android  
    27      Android  
    28      Android  
    29      Android  
    ...         ...  
    49970   Android  
    49971   Android  
    49972   Android  
    49973   Android  
    49974   Android  
    49975   Android  
    49976   Android  
    49977   Android  
    49978   Android  
    49979   Android  
    49980   Android  
    49981   Android  
    49982   Android  
    49983   Android  
    49984   Android  
    49985   Android  
    49986   Android  
    49987   Android  
    49988   Android  
    49989   Android  
    49990   Android  
    49991   Android  
    49992   Android  
    49993   Android  
    49994   Android  
    49995   Android  
    49996   Android  
    49997   Android  
    49998   Android  
    49999   Android  

    [50000 rows x 4 columns]


# create model

I created models for the new dataframes, changing the targets to the percentile column.  I also changed the min token frequency to 2, since this data set will be 1/8 the size of the main data set.


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Android_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Android_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4131


# accuracy

The accuracy for these models are much worse then when we were predicting subreddit.  


```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_43_0.png)


# save csv

I saved all of the top_100 dataframes as csv files


```python
Android_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Android_top_unigrams_df.to_csv("csv_files\Android_top_unigrams.csv")
```

# bigram model

Repeated model but with bigrams


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Android_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Android_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4217



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_48_0.png)



```python
Android_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Android_top_bigrams_df.to_csv("csv_files\Android_top_bigrams.csv")
```

# repeat

I repeated the above steps for each of the subreddits, saving the returned top_100 dataframes as csv files.


```python
boardgames_df = reddit_df[reddit_df.subreddit == 'boardgames']
boardgames_df = boardgames_df.sort_values(['score'])
boardgames_df = boardgames_df.reset_index(drop=True)

third = (len(boardgames_df)/3)
iterator = 0
for x in boardgames_df.percentile:
    if(iterator < third):
        boardgames_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        boardgames_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        boardgames_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = boardgames_df['percentile']

    reddit_counts = reddit_vec.fit_transform(boardgames_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4073



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_53_0.png)



```python
boardgames_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
boardgames_top_unigrams_df.to_csv("csv_files\\boardgames_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = boardgames_df['percentile']

    reddit_counts = reddit_vec.fit_transform(boardgames_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4094



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_56_0.png)



```python
boardgames_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
boardgames_top_bigrams_df.to_csv("csv_files\\boardgames_top_bigrams.csv")
```


```python
Conservative_df = reddit_df[reddit_df.subreddit == 'Conservative']
Conservative_df = Conservative_df.sort_values(['score'])
Conservative_df = Conservative_df.reset_index(drop=True)

third = (len(Conservative_df)/3)
iterator = 0
for x in Conservative_df.percentile:
    if(iterator < third):
        Conservative_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        Conservative_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        Conservative_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Conservative_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Conservative_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4044



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_60_0.png)



```python
Conservative_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Conservative_top_unigrams_df.to_csv("csv_files\Conservative_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Conservative_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Conservative_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.3992



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_63_0.png)



```python
Conservative_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Conservative_top_bigrams_df.to_csv("csv_files\Conservative_top_bigrams.csv")
```


```python
hockey_df = reddit_df[reddit_df.subreddit == 'hockey']
hockey_df = hockey_df.sort_values(['score'])
hockey_df = hockey_df.reset_index(drop=True)

third = (len(hockey_df)/3)
iterator = 0
for x in hockey_df.percentile:
    if(iterator < third):
        hockey_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        hockey_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        hockey_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = hockey_df['percentile']

    reddit_counts = reddit_vec.fit_transform(hockey_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4095



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_67_0.png)



```python
hockey_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
hockey_top_unigrams_df.to_csv("csv_files\hockey_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = hockey_df['percentile']

    reddit_counts = reddit_vec.fit_transform(hockey_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4055



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_70_0.png)



```python
hockey_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
hockey_top_bigrams_df.to_csv("csv_files\hockey_top_bigrams.csv")
```


```python
Libertarian_df = reddit_df[reddit_df.subreddit == 'Libertarian']
Libertarian_df = Libertarian_df.sort_values(['score'])
Libertarian_df = Libertarian_df.reset_index(drop=True)

third = (len(Libertarian_df)/3)
iterator = 0
for x in Libertarian_df.percentile:
    if(iterator < third):
        Libertarian_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        Libertarian_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        Libertarian_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Libertarian_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Libertarian_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4399



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_74_0.png)



```python
Libertarian_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Libertarian_top_unigrams_df.to_csv("csv_files\Libertarian_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = Libertarian_df['percentile']

    reddit_counts = reddit_vec.fit_transform(Libertarian_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4566



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_77_0.png)



```python
Libertarian_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
Libertarian_top_bigrams_df.to_csv("csv_files\Libertarian_top_bigrams.csv")
```


```python
neoliberal_df = reddit_df[reddit_df.subreddit == 'neoliberal']
neoliberal_df = neoliberal_df.sort_values(['score'])
neoliberal_df = neoliberal_df.reset_index(drop=True)

third = (len(neoliberal_df)/3)
iterator = 0
for x in neoliberal_df.percentile:
    if(iterator < third):
        neoliberal_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        neoliberal_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        neoliberal_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = neoliberal_df['percentile']

    reddit_counts = reddit_vec.fit_transform(neoliberal_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4596



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_81_0.png)



```python
neoliberal_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
neoliberal_top_unigrams_df.to_csv("csv_files\\neoliberal_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = neoliberal_df['percentile']

    reddit_counts = reddit_vec.fit_transform(neoliberal_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4461



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_84_0.png)



```python
neoliberal_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
neoliberal_top_bigrams_df.to_csv("csv_files\\neoliberal_top_bigrams.csv")
```


```python
politics_df = reddit_df[reddit_df.subreddit == 'politics']
politics_df = politics_df.sort_values(['score'])
politics_df = politics_df.reset_index(drop=True)

third = (len(politics_df)/3)
iterator = 0
for x in politics_df.percentile:
    if(iterator < third):
        politics_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        politics_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        politics_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = politics_df['percentile']

    reddit_counts = reddit_vec.fit_transform(politics_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4153



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_88_0.png)



```python
politics_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
politics_top_unigrams_df.to_csv("csv_files\politics_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = politics_df['percentile']

    reddit_counts = reddit_vec.fit_transform(politics_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4294



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_91_0.png)



```python
politics_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
politics_top_bigrams_df.to_csv("csv_files\politics_top_bigrams.csv")
```


```python
worldnews_df = reddit_df[reddit_df.subreddit == 'worldnews']
worldnews_df = worldnews_df.sort_values(['score'])
worldnews_df = worldnews_df.reset_index(drop=True)

third = (len(worldnews_df)/3)
iterator = 0
for x in worldnews_df.percentile:
    if(iterator < third):
        worldnews_df.loc[iterator, 'percentile'] = 1
    if(iterator >= third and iterator <= (third*2)):
        worldnews_df.loc[iterator, 'percentile'] = 2
    if(iterator > (third*2)):
        worldnews_df.loc[iterator, 'percentile'] = 3
    iterator = iterator + 1
```


```python
reddit_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = worldnews_df['percentile']

    reddit_counts = reddit_vec.fit_transform(worldnews_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.4022



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_95_0.png)



```python
worldnews_top_unigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
worldnews_top_unigrams_df.to_csv("csv_files\worldnews_top_unigrams.csv")
```


```python
reddit_vec = CountVectorizer(ngram_range=(2,2), min_df=2, tokenizer=nltk.word_tokenize, lowercase=True, stop_words=fullset)

z=0
start=0
iterator=0
while(iterator<(1)):

    iterator=iterator+1
    print(iterator)
    start=start+num_retreive

    y = worldnews_df['percentile']

    reddit_counts = reddit_vec.fit_transform(worldnews_df['body'])
    tfidf_transformer = TfidfTransformer()
    reddit_tfidf = tfidf_transformer.fit_transform(reddit_counts)

    X = reddit_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    classifier = MultinomialNB().partial_fit(X_train, y_train,classes=np.unique(y_train))

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
```

    1
    0.3884



```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['3','2','1'],
            yticklabels=['3','2','1'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```


![png](images\output_98_0.png)



```python
worldnews_top_bigrams_df = return_top100_score(reddit_vec, classifier, class_labels=np.unique(y_train))
worldnews_top_bigrams_df.to_csv("csv_files\worldnews_top_bigrams.csv")
```

# comparing lists

I compared the lists of top words for each subreddit to find which words appear in each list.

# Android unigrams


```python
x_df = reddit_top_unigrams_df.Android.tolist()
y_df = Android_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 90
    ['support', '8', 's8', 'software', 'features', 'sure', 'nexus', 'headphone', 'great', 'lg', 'never', 'buy', 'price', 'got', 'jack', 'devices', 'since', 'go', '[', ']', 'lot', 'something', 'note', 'need', 'work', "'ll", "'d", 'though', 'device', 'every', 'using', 'going', 'ca', 'actually', '’', 'thing', 'see', 'used', 'pretty', 'back', 'updates', '2', 'way', 'know', 'well', 'make', 'https', 'years', 'year', 'apps', 'could', 'new', 'camera', 'want', 'app', 'time', "'ve", 'screen', "'re", 'good', 'gt', 'battery', 'also', 'samsung', 'much', 'better', 'think', 'pixel', 'still', 'iphone', 'really', '``', 'even', 'use', "'m", 'apple', "''", 'get', 'people', 'phones', 'one', 'would', 'android', 'google', 'like', '(', ')', 'phone', "n't", "'s"]

    middle words: 88
    ['stock', 'support', 'lg', 'updates', 'lot', 'every', 'headphone', 'go', 'price', 'actually', 'great', 'note', 'nexus', '’', 'ca', 'never', 'buy', 'used', 'jack', 'sure', 'since', 'pretty', 'going', 'thing', 'something', 'devices', 'years', 'need', 'got', 'work', "'d", 'back', 'using', 'way', 'year', 'make', "'ll", '2', 'well', 'though', 'samsung', 'apps', 'know', 'could', 'camera', 'see', 'better', 'apple', 'time', 'screen', 'new', 'want', 'much', 'iphone', 'think', 'device', "'re", 'also', 'good', 'app', 'https', "'ve", '``', 'even', 'people', 'pixel', 'really', "''", 'still', 'http', 'gt', 'phones', 'battery', 'use', 'get', 'google', 'one', 'android', 'would', '[', ']', "'m", 'like', 'phone', '(', ')', "'s", "n't"]

    bottom words: 88
    ['s8', 'support', 'jack', 'lg', 'ios', 'note', 'lot', 'great', 'nexus', 'charging', 'actually', 'sure', 'every', 'pretty', 'go', 'buy', 'price', 'since', 'work', 'thing', 'apps', "'d", 'never', 'going', 'used', 'using', 'though', '2', 'need', "'ll", 'ca', 'year', 'something', '’', 'devices', 'years', 'way', 'back', 'got', 'make', 'well', 'could', 'app', 'know', 'camera', 'samsung', 'want', 'see', 'time', 'also', 'good', 'new', 'screen', 'better', 'much', "'ve", 'think', 'even', 'device', 'really', 'still', 'battery', 'https', '``', "'re", 'people', 'pixel', 'phones', "''", 'iphone', 'use', 'get', 'apple', 'one', 'gt', 'google', 'http', "'m", 'would', 'android', 'like', ']', '[', 'phone', '(', ')', "'s", "n't"]


# Android bigrams


```python
x_df = reddit_top_bigrams_df.Android.tolist()
y_df = Android_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 57
    ['iphone 7', 'new phone', 'google assistant', 'home button', 'even though', 'fingerprint scanner', 'nexus 6p', 'google play', 'iphone 8', 'https //play.google.com/store/apps/details', 'years ago', '( http', "n't see", "'s still", "phone n't", 'seems like', "phone 's", 'feel like', 'gon na', "n't need", "samsung 's", "n't care", 'play store', 'android phone', 'android phones', 'last year', "n't use", '2 years', "'s like", 'looks like', 'pixel xl', "n't like", 'much better', 'iphone x', 'stock android', "think 's", "google 's", "apple 's", "'m sure", "could n't", 'pretty much', "n't really", "n't get", 'pixel 2', 'wireless charging', "n't even", "n't want", "n't think", "n't know", 'note 8', "would n't", 'battery life', "wo n't", 'headphone jack', '( https', '] (', "ca n't"]

    middle words: 57
    ['might interested', '[ aosp', 'client [', 'ebay ]', 'web browser', 'customer service', "could n't", "device 's", 'pixel xl', 'stock android', 'http //www.reddit.com/r/android/wiki/rulesandregs', "n't even", "n't get", "'m sure", 'etc )', 'submission removed', 'to= 2fr', '2fr 2fandroid', '//www.reddit.com/message/compose to=', "n't want", 'pixel 2', 'removed gt', "n't really", 'clicking link', '[ message', 'appeal please', 'like appeal', 'moderators clicking', 'http //www.reddit.com/message/compose', 'note 8', 'subreddit ]', "n't think", ') would', 'message moderators', 'wiki page', 'page gt', 'information. ]', 'gt information.', 'see wiki', 'gt rule', 'link ]', '[ see', "'' gt", 'gt [', '[ ]', 'would like', 'please [', "n't know", "wo n't", "would n't", 'battery life', 'headphone jack', ') [', "ca n't", '( https', '( http', '] (']

    bottom words: 46
    ['//www.reddit.com/r/android/wiki/rulesandregs wiki_2._we_welcome_discussion-promoting_posts_that_benefit_the_community.2c_and_not_the_individual', 'wiki_2._we_welcome_discussion-promoting_posts_that_benefit_the_community.2c_and_not_the_individual )', 'selling buying', 'tracker ]', 'app recommendations', 'web browser', "device 's", 'submission removed', 'removed gt', '2fr 2fandroid', '//www.reddit.com/message/compose to=', 'to= 2fr', "would n't", 'wireless charging', 'http //www.reddit.com/r/android/wiki/rulesandregs', "n't know", '[ ]', 'clicking link', 'appeal please', '[ message', 'like appeal', 'http //www.reddit.com/message/compose', 'moderators clicking', 'link ]', ') would', 'gt rule', 'page gt', 'gt information.', 'information. ]', 'see wiki', 'battery life', '[ see', 'wiki page', 'gt [', "'' gt", 'message moderators', "wo n't", 'subreddit ]', 'would like', 'headphone jack', 'please [', "ca n't", ') [', '( https', '( http', '] (']


# boardgames unigrams


```python
x_df = reddit_top_unigrams_df.boardgames.tolist()
y_df = boardgames_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 88
    ['maybe', 'getting', 'amp', 'look', 'table', 'since', 'try', 'bit', 'little', 'enough', 'find', 'everyone', ']', '[', 'take', 'sure', 'buy', 'someone', 'say', 'might', 'always', 'actually', '2', 'things', 'probably', 'need', 'feel', 'better', 'many', 'different', 'card', 'https', 'never', 'pretty', 'go', "'ll", 'two', 'going', 'group', "'d", 'though', 'still', 'got', 'love', 'something', 'rules', 'could', 'players', 'player', 'well', 'see', 'fun', 'new', 'way', 'know', 'make', 'great', 'want', 'lot', 'first', 'even', 'playing', 'board', 'also', 'cards', 'much', 'good', "'re", 'played', 'time', "'ve", "'m", 'think', 'really', 'would', 'get', '``', "''", 'people', 'one', 'play', 'like', '(', ')', 'games', "n't", "'s", 'game']

    middle words: 90
    ['deck', 'everyone', 'someone', 'buy', 'enough', 'maybe', 'box', 'look', 'looking', 'since', 'always', 'things', 'actually', 'table', 'take', 'probably', 'amp', 'say', 'expansion', 'feel', 'never', 'many', 'find', 'sure', 'bit', 'need', 'better', 'try', 'might', '2', 'rules', 'card', 'going', 'different', 'go', 'way', 'see', 'group', 'pretty', 'got', 'something', 'love', 'https', ']', '[', "'d", 'still', 'make', 'two', "'ll", 'though', 'even', 'first', 'lot', 'know', 'could', 'playing', 'want', 'new', 'well', 'fun', 'great', 'player', 'board', 'players', 'much', "'re", 'cards', 'also', 'good', '``', "''", 'time', 'think', 'played', 'people', "'ve", "'m", 'would', 'really', 'get', 'one', 'play', 'like', '(', ')', 'games', "n't", "'s", 'game']

    bottom words: 90
    ['always', 'little', 'enough', 'everyone', 'look', 'since', 'getting', 'box', 'someone', 'looking', 'take', 'expansion', 'maybe', 'find', 'things', 'probably', 'say', 'try', 'bit', 'actually', 'feel', '2', 'might', 'better', 'many', 'buy', 'never', 'need', 'card', 'go', 'two', 'sure', 'different', 'pretty', 'rules', 'going', 'love', 'got', 'group', 'amp', 'something', 'fun', 'way', "'ll", 'new', 'see', 'lot', 'though', 'make', 'still', 'could', 'playing', "'d", 'know', 'even', 'first', 'player', 'well', 'great', 'want', 'players', 'cards', "'re", 'board', 'https', 'much', 'good', 'also', 'time', '``', "''", '[', ']', "'ve", 'think', 'played', 'people', 'really', "'m", 'would', 'get', 'one', 'play', 'like', '(', 'games', ')', "n't", "'s", 'game']


# boardgames bigrams


```python
x_df = reddit_top_bigrams_df.boardgames.tolist()
y_df = boardgames_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 81
    ['played game', 'never played', 'player game', 'make sure', 'end game', 'new games', 'game really', "'s game", 'game would', 'love game', 'many games', 'area control', 'game night', 'first game', 'game like', "'ve got", 'feels like', 'really good', "'s pretty", 'even though', 'game play', "'m going", 'good game', 'playing game', "'s one", "'ve seen", 'every game', "'s also", 'sounds like', "n't play", 'really like', 'star wars', 'every time', 'worker placement', 'card game', "'s great", 'like game', 'play games', "could n't", "n't see", 'pretty much', "'s good", 'something like', 'seems like', "'s like", 'great game', "people n't", "'ve never", ") 's", 'game (', '( http', "'s really", "n't played", 'want play', 'looks like', "'' ``", "games n't", 'games like', 'play game', "n't even", 'base game', 'first time', "n't want", "n't like", "'m sure", "game n't", "n't really", "n't get", "think 's", "wo n't", "n't know", "would n't", 'feel like', "n't think", "game 's", "'ve played", 'board game', 'board games', '( https', "ca n't", '] (']

    middle words: 84
    ["'s like", 'feels like', '7 wonders', 'good game', "n't see", 'end game', '2 player', 'first game', 'love game', 'really enjoy', "game 've", 'ticket ride', "'s game", 'deck building', 'player count', 'even though', 'game )', 'games (', 'star wars', 'really good', 'player game', "'m looking", "'' ``", 'game would', "'m going", 'area control', 'play games', 'make sure', "'ve got", "'s one", 'worker placement', 'like game', "'ve seen", 'never played', 'game like', ") 's", 'every time', 'sounds like', "n't even", 'game night', "'s great", "n't play", "'s pretty", 'play game', "'ve never", "games n't", 'want play', 'game (', "'s also", 'game play', "'s good", 'great game', 'something like', "could n't", 'card game', "'s really", 'pretty much', 'seems like', "n't want", 'really like', '( http', "n't get", 'first time', 'games like', 'looks like', "n't played", "game n't", "game 's", "wo n't", "n't like", "'m sure", 'base game', 'feel like', "n't really", "think 's", "would n't", "n't think", "n't know", 'board games', "'ve played", 'board game', "ca n't", '( https', '] (']

    bottom words: 55
    ["n't see", 'sounds like', "'s one", "game 've", "'' ``", 'great game', "'m going", ") 's", 'play games', "'s still", "'s good", "'s pretty", 'card game', 'games (', "'s great", 'first time', 'game play', 'something like', 'game (', "'s really", 'really like', "'ve never", 'pretty much', "could n't", "games n't", 'looks like', "n't even", 'want play', 'worker placement', 'seems like', 'play game', "n't want", "n't played", "game n't", 'base game', "game 's", 'games like', "'s also", "n't get", "think 's", "'m sure", "wo n't", "n't really", "n't like", 'feel like', "n't know", "would n't", "n't think", "'ve played", '( http', 'board games', 'board game', "ca n't", '( https', '] (']


# Conservative unigrams


```python
x_df = (reddit_top_unigrams_df.Conservative).tolist()
y_df = Conservative_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
prcint("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
3 print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 94
    ['first', 'american', 'nazi', 'democrats', 'support', 'law', 'wrong', 'liberals', '’', 'everyone', 'free', 'nothing', 'anyone', 'take', 'sure', 'back', 'http', 'bad', 'mean', 'person', 'state', 'speech', 'racist', 'nazis', 'believe', '[', ']', 'party', 'years', 'anything', 'obama', 'violence', 'black', "'ve", 'need', 'still', 'liberal', 'https', 'well', 'go', 'saying', 'things', 'every', 'antifa', 'political', 'thing', 'conservatives', 'many', 'something', 'never', 'someone', 'president', 'point', 'country', 'actually', 'ca', 'much', 'media', 'could', 'government', 'good', 'way', 'make', 'conservative', 'said', 'also', 'see', 'us', 'going', 'time', 'say', 'really', 'know', 'want', 'white', 'even', "'m", 'left', 'get', 'one', "'re", 'think', 'right', 'gt', '(', ')', 'would', 'like', 'trump', '``', "''", 'people', "'s", "n't"]

    middle words: 91
    ['wrong', 'liberal', 'states', 'everyone', 'obama', 'violence', 'black', 'let', 'first', 'nothing', 'http', 'support', '’', 'problem', 'president', 'money', 'free', 'person', 'back', 'party', 'every', 'bad', 'nazis', 'mean', 'agree', 'believe', 'political', 'anything', ']', '[', 'years', 'antifa', 'conservatives', 'state', 'take', 'sure', 'someone', "'ve", 'media', 'never', 'saying', 'still', 'go', 'need', 'country', 'https', 'many', 'things', 'ca', 'thing', 'actually', 'something', 'well', 'point', 'conservative', 'much', 'said', 'also', 'good', 'time', 'going', 'say', 'way', 'could', 'us', 'see', 'white', 'really', 'make', 'government', 'know', 'left', 'want', 'even', 'get', 'right', "'m", 'one', "'re", 'think', '(', 'gt', ')', 'trump', 'like', 'would', '``', "''", 'people', "'s", "n't"]

    bottom words: 87
    ['states', 'party', 'everyone', 'years', 'let', 'back', 'problem', 'http', 'nothing', 'person', 'bad', 'antifa', 'law', 'black', 'support', 'money', 'free', 'first', 'every', 'president', 'care', 'wrong', ']', '[', 'conservatives', 'take', 'believe', 'sure', 'political', 'state', 'many', 'anything', 'mean', 'agree', 'never', 'thing', 'go', 'something', "'ve", 'someone', 'country', 'need', 'still', 'actually', 'https', 'ca', 'well', 'things', 'left', 'much', 'conservative', 'us', 'saying', 'good', 'could', 'time', 'said', 'see', 'going', 'way', 'really', 'point', 'also', 'white', 'make', 'say', 'government', 'know', 'want', 'even', 'get', 'right', 'one', "'m", '(', 'trump', "'re", ')', 'think', 'gt', 'like', 'would', '``', "''", 'people', "'s", "n't"]


# Conservative bigrams


```python
x_df = reddit_top_bigrams_df.Conservative.tolist()
y_df = Conservative_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 88
    ['white nationalists', 'people would', 'lot people', "left 's", 'civil rights', "'s right", 'something like', 'white supremacy', "'s ``", "n't agree", 'national anthem', 'first amendment', 'things like', "obama 's", 'illegal immigration', 'like ``', 'supreme court', "n't say", 'republican party', 'saying ``', 'sounds like', "'s pretty", "'ve seen", 'liberal media', 'far left', 'president trump', 'north korea', 'https //www.youtube.com/watch', 'feel like', "'' n't", 'federal government', "n't matter", 'said ``', "'m saying", "'s going", 'even though', 'climate change', "really n't", "n't need", "'s really", "n't believe", 'many people', 'people like', 'right wing', 'civil war', 'gon na', 'pretty much', "'re going", 'years ago', "n't understand", "people 's", 'say ``', 'white house', "n't make", 'identity politics', 'seems like', 'black people', "'' 's", 'white supremacist', "could n't", "trump n't", "n't really", "n't see", "'s like", "n't mean", "n't care", 'white people', "'' ``", "let 's", 'white supremacists', 'united states', "people n't", "'m sure", "think 's", "n't get", "n't want", 'free speech', "n't even", "n't like", '( http', "n't know", "n't think", "trump 's", "wo n't", "would n't", '( https', '] (', "ca n't"]

    middle words: 83
    ["'s right", 'health care', 'lot people', 'things like', 'like ``', 'far left', "obama 's", 'something like', 'https //www.youtube.com/watch', 'supreme court', "gt 's", "gt n't", "really n't", 'first amendment', 'said ``', 'people would', 'white house', "'s pretty", 'free market', 'republican party', "n't agree", "n't say", "n't need", 'even though', "'ve seen", "'re talking", "'re going", 'sounds like', 'people like', 'pretty much', "'s good", "'' 's", "trump n't", "n't matter", 'feel like', "'s really", "'' n't", 'gon na', 'say ``', 'north korea', 'white supremacist', "'s going", "n't believe", "'m saying", 'years ago', 'federal government', 'climate change', 'black people', 'identity politics', "n't understand", 'many people', "people 's", "'s like", 'white people', "n't care", 'civil war', 'white supremacists', "n't really", 'seems like', "could n't", "n't make", "n't mean", 'united states', "n't see", "n't like", 'free speech', "'' ``", "let 's", "people n't", "'m sure", "trump 's", "n't get", "think 's", '( http', "n't want", "n't know", "n't even", "wo n't", "n't think", "would n't", '( https', '] (', "ca n't"]

    bottom words: 85
    ["'s really", 'freedom speech', 'right wing', 'said ``', 'something like', 'supreme court', 'first place', 'first amendment', 'lot people', 'even though', 'white house', "'re right", "gt 's", 'years ago', 'white supremacist', "trump n't", "really n't", "'ve seen", "'re saying", 'people would', 'national anthem', "'re talking", 'illegal immigration', "'s pretty", 'gon na', 'health care', 'civil rights', "'' 's", 'identity politics', 'free market', 'federal government', 'saying ``', "n't need", 'climate change', "'' n't", 'pretty much', 'things like', 'feel like', "n't matter", 'people like', 'many people', "'s going", 'sounds like', "n't say", 'north korea', "n't believe", 'civil war', "n't agree", 'seems like', "gt n't", "'re going", 'say ``', "could n't", "people 's", "n't understand", "'s like", 'white people', "n't make", 'united states', 'white supremacists', 'black people', "n't care", "'' ``", "n't really", "n't like", "n't get", "'m saying", '( http', "n't mean", "n't see", "let 's", "people n't", "trump 's", "'m sure", 'free speech', "n't want", "think 's", "n't even", "n't know", "wo n't", '( https', "would n't", "n't think", '] (', "ca n't"]


# hockey unigrams


```python
x_df = reddit_top_unigrams_df.hockey.tolist()
y_df = hockey_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 94
    ['saying', '2', 'maybe', 'always', 'leafs', 'fan', 'take', 'points', 'shit', 'playoffs', 'look', 'two', 'something', 'getting', 'thing', 'line', 'sure', 'playing', 'ca', 'bad', 'said', 'actually', 'played', 'point', 'probably', 'though', 'lot', 'never', 'every', '’', 'fans', 'guys', 'guy', 'top', 'great', 'best', "'ll", 'right', 'pretty', 'better', "'ve", 'gt', 'first', "'d", 'back', 'say', 'teams', 'well', 'go', 'want', 'games', 'league', 'got', 'know', 'way', 'make', 'still', 'much', 'also', 'https', 'years', 'going', 'see', 'could', 'last', '[', ']', 'player', 'time', 'play', 'hockey', 'even', "'m", "'re", 'nhl', 'really', 'players', 'season', 'game', 'get', 'people', 'good', '``', 'year', "''", 'one', 'think', 'would', 'team', 'like', '(', ')', "n't", "'s"]

    middle words: 92
    ['ice', 'look', 'fan', 'maybe', '2', 'always', '’', 'getting', 'never', 'points', 'take', 'thing', 'something', 'actually', 'fans', 'playing', 'every', 'played', 'two', 'yeah', 'bad', 'said', 'point', 'line', 'ca', 'guy', 'gt', 'say', 'probably', 'want', 'great', 'league', 'best', 'pretty', 'sure', 'go', 'guys', 'back', 'right', 'lot', "'ve", 'teams', 'first', 'top', 'though', "'d", "'ll", 'make', 'got', 'way', 'know', 'well', 'games', 'https', 'better', 'player', 'years', 'play', 'also', 'much', 'still', 'going', 'even', 'nhl', '``', 'could', 'players', 'people', 'hockey', 'last', 'time', '[', ']', 'see', "''", "'re", 'really', 'season', "'m", 'good', 'game', 'one', 'get', 'year', 'think', 'team', 'would', 'like', '(', ')', "n't", "'s"]

    bottom words: 93
    ['2', 'points', 'playing', 'shit', 'thing', 'look', 'getting', 'yeah', 'trade', 'take', 'maybe', 'played', 'two', 'something', 'actually', 'leafs', 'never', 'every', 'great', 'line', 'saying', '’', 'bad', 'ca', 'sure', 'fans', 'got', 'guy', 'probably', 'lot', 'said', 'league', "'ll", 'pretty', 'guys', 'point', 'though', 'first', 'best', 'back', "'ve", 'want', 'top', 'go', 'say', 'well', 'way', "'d", 'gt', 'https', 'right', 'teams', 'games', 'also', 'make', 'better', 'know', 'years', 'much', 'player', 'still', 'play', 'last', 'time', 'going', 'could', 'see', 'hockey', 'really', 'season', '``', 'even', 'nhl', "''", ']', '[', 'players', 'game', 'good', 'people', "'re", 'get', 'one', "'m", 'year', 'think', 'team', '(', 'like', 'would', ')', "'s", "n't"]


# hockey bigrams


```python
x_df = reddit_top_bigrams_df.hockey.tolist()
y_df = hockey_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 82
    ["n't going", 'win cup', "think 'll", "'m saying", "team n't", 'cap space', 'couple years', "'s way", "'s got", 'new arena', "n't good", 'much better', "would 've", "n't play", 'every year', 'got ta', 'look like', 'year old', "'s great", "'ve seen", 'makes sense', 'pretty sure', 'top 6', 'guys like', "year 's", 'something like', "let 's", "really n't", "'s one", "know 's", 'first round', "'s still", 'best player', 'every time', 'stanley cup', "'' ``", "'s really", "'s also", 'next year', 'white house', 'even though', "'re going", 'really good', "n't mean", "'d say", 'pretty good', "like 's", "n't like", 'one best', "'s pretty", "team 's", 'pretty much', 'regular season', "n't make", 'feel like', "'s good", "'s going", 'years ago', 'looks like', 'seems like', "n't really", ') [', "could n't", 'https //www.youtube.com/watch', "'s like", "'m sure", "n't see", "n't get", "n't even", "n't want", 'gon na', 'last season', "n't know", "think 's", "would n't", "wo n't", '( http', "n't think", 'last year', "ca n't", '( https', '] (']

    middle words: 75
    ['top 6', "'s also", "would 've", 'year old', 'make playoffs', "'s one", 'got ta', '2 years', 'much better', "'ve seen", "really n't", "'d say", 'every time', "'s got", "'s way", "'' ``", 'pretty sure', "know 's", "year 's", "'m saying", 'pretty good', 'next year', "'re going", "let 's", 'couple years', "n't mean", 'could see', 'look like', "n't like", 'makes sense', "think 'll", 'something like', "'s pretty", 'really good', "'s really", 'https //www.youtube.com/watch', "team 's", '-- --', 'one best', "like 's", 'regular season', "'s still", "'s good", 'even though', 'looks like', "n't make", 'feel like', 'pretty much', 'years ago', 'seems like', "could n't", "'s going", "'s like", "n't want", ') gt', ') [', "n't see", "n't really", "'m sure", "n't even", '] ]', '[ [', "n't get", 'last season', 'gon na', "n't know", "think 's", "wo n't", '( http', "would n't", "n't think", 'last year', "ca n't", '( https', '] (']

    bottom words: 65
    ['really good', "'d say", 'every year', 'white house', "would 've", 'much better', 'look like', "'s one", 'regular season', "year 's", "like 's", 'one best', "'re going", 'stanley cup', 'year old', "n't going", "really n't", 'makes sense', 'even though', "know 's", 'next year', "'ve seen", "n't good", 'something like', "'s really", "'s good", "'s pretty", 'pretty much', "team 's", "'s still", "let 's", 'seems like', 'looks like', "n't like", "n't mean", "could n't", 'years ago', '-- --', "'s like", "n't make", 'feel like', "'s going", "'m saying", ') [', "n't really", "n't want", 'last season', ') gt', 'gon na', "n't get", "n't see", "'m sure", "n't even", '] ]', "think 's", '[ [', "n't know", "wo n't", '( http', "n't think", "would n't", 'last year', "ca n't", '( https', '] (']


# Libertarian unigrams


```python
x_df = reddit_top_unigrams_df.Libertarian.tolist()
y_df = Libertarian_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 92
    ['life', 'libertarianism', 'agree', 'world', 'public', 'argument', 'society', 'saying', 'anything', 'care', 'said', 'power', 'sure', "'d", 'problem', 'healthcare', 'communism', 'never', 'believe', 'country', 'person', 'mean', 'every', 'everyone', "'ve", 'thing', 'use', 'rights', 'tax', 'better', 'go', 'take', 'socialism', 'something', 'well', 'many', 'without', 'private', 'still', 'actually', 'property', 'going', 'things', 'taxes', 'point', 'good', 'need', 'https', 'ca', 'libertarians', 'capitalism', 'see', 'system', 'know', 'time', 'someone', 'market', 'pay', 'say', '[', ']', 'really', 'work', 'could', 'way', 'much', 'also', 'state', 'us', 'money', 'make', 'free', 'libertarian', 'even', 'want', "'m", 'right', 'get', 'one', "'re", 'think', 'like', 'gt', '(', ')', 'government', 'would', '``', "''", 'people', "'s", "n't"]

    middle words: 92
    ['public', 'care', 'wrong', 'life', 'world', 'yes', 'law', 'country', 'society', 'everyone', 'every', 'agree', 'anything', 'means', "'d", 'capitalism', 'problem', 'sure', 'power', 'person', 'argument', 'never', 'believe', 'better', 'mean', 'private', 'saying', 'without', 'thing', 'libertarians', 'take', 'well', 'use', 'tax', 'said', 'go', 'actually', 'going', "'ve", 'something', 'still', 'many', 'see', 'taxes', 'someone', 'rights', 'property', 'really', 'us', 'time', 'ca', 'say', 'work', 'need', 'libertarian', 'much', 'things', 'system', 'could', 'good', 'market', 'point', 'also', 'free', 'pay', 'way', 'state', 'know', 'make', 'https', 'money', 'want', '[', ']', 'even', 'right', 'get', "'m", 'one', 'think', "'re", 'like', 'government', '(', '``', ')', "''", 'would', 'gt', 'people', "'s", "n't"]

    bottom words: 97
    ['agree', 'give', 'law', 'problem', 'business', 'care', 'force', 'power', 'life', 'believe', 'yes', 'world', 'sure', 'means', 'argument', 'healthcare', "'d", 'every', 'wrong', 'country', 'anything', 'everyone', 'libertarians', 'many', 'tax', "'ve", 'person', 'take', 'never', 'thing', 'society', 'mean', 'saying', 'use', 'actually', 'better', 'socialism', 'said', 'without', 'communism', 'going', 'still', 'well', 'go', 'something', 'taxes', 'time', 'market', 'good', 'need', 'see', 'private', 'someone', 'really', 'things', 'way', 'much', 'also', 'work', 'could', 'ca', 'libertarian', 'point', 'system', 'us', 'free', 'capitalism', 'rights', 'pay', 'property', 'money', 'say', 'know', 'state', 'https', ']', '[', 'make', 'even', 'want', 'get', 'right', "'m", 'one', 'think', 'like', "'re", 'government', '(', ')', 'would', '``', "''", 'people', 'gt', "'s", "n't"]


# Libertarian bigrams


```python
x_df = reddit_top_bigrams_df.Libertarian.tolist()
y_df = Libertarian_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 86
    ['makes sense', 'people want', "'re talking", ") n't", 'single payer', "gt 's", 'saying ``', "n't say", "`` ''", "government n't", 'black people', "'m going", 'gon na', 'private sector', 'ron paul', "`` n't", 'people like', "government 's", 'income tax', "'s ``", "'s really", "n't believe", 'things like', "'re going", 'price gouging', "n't pay", 'sounds like', 'even though', "gt n't", ') ^|', '^| [', "'' n't", 'seems like', 'years ago', 'means production', 'first place', "could n't", 'pay taxes', "n't need", 'poor people', 'say ``', 'everyone else', 'health care', "'' 's", "n't care", ') [', 'someone else', 'pretty much', 'property rights', "n't work", 'people would', "n't understand", '[ ]', "n't make", 'federal government', 'many people', 'minimum wage', "'s like", "n't really", 'private property', 'united states', "let 's", '-- --', 'nbsp amp', "people 's", 'free speech', "n't see", 'amp nbsp', "n't like", "n't mean", "'' ``", "'m sure", "think 's", '( http', "n't even", "n't know", "n't get", "wo n't", "people n't", "n't want", "n't think", 'free market', "would n't", '( https', "ca n't", '] (']

    middle words: 80
    ['poor people', 'would like', ") n't", 'sounds like', "`` ''", "n't believe", "n't exist", "n't matter", 'income tax', "'s ``", 'people like', 'nbsp amp', 'seems like', "'re going", 'things like', 'gon na', 'health care', "n't say", "'' 's", 'amp nbsp', 'gt gt', "n't work", "n't pay", 'people want', '[ ]', "n't care", 'federal government', 'even though', "government n't", 'years ago', 'private sector', "'s like", 'someone else', "'re talking", 'minimum wage', 'means production', 'say ``', 'price gouging', 'first place', "n't need", "n't understand", 'many people', "'' n't", 'private property', "n't really", 'pay taxes', "could n't", "people 's", "'' ``", 'united states', 'property rights', "gt 's", 'people would', "'m saying", "gt n't", "n't make", "n't see", 'free speech', "let 's", "think 's", '[ ^exclude', "'m sure", "n't like", '-- --', "n't get", "people n't", "n't mean", '( http', "n't want", "n't even", "wo n't", "n't know", "n't think", 'free market', "would n't", '^| [', ') ^|', "ca n't", '( https', '] (']

    bottom words: 89
    ["government 's", 'taxation theft', "'re saying", 'sounds like', "`` ''", "n't give", 'makes sense', "`` n't", 'years ago', ") n't", 'even though', 'would like', 'saying ``', 'private sector', 'people want', 'things like', 'gon na', '[ ]', 'everyone else', 'economic system', 'people would', 'single payer', "n't believe", "'s really", 'first place', "n't pay", 'pretty much', 'people like', "could n't", "n't exist", "government n't", "'' 's", "n't work", "'re talking", ') [', 'minimum wage', 'health care', "n't matter", 'pay taxes', "n't care", 'many people', 'gt gt', "n't say", 'federal government', "'re going", "'' n't", "n't really", 'seems like', '[ ^exclude', 'united states', "n't need", "n't understand", 'someone else', 'say ``', "'s like", "people 's", "gt n't", "let 's", 'nbsp amp', 'free speech', "gt 's", "think 's", "'' ``", 'amp nbsp', "n't get", "'m saying", 'property rights', "n't see", "n't like", "'m sure", 'means production', "n't mean", "n't make", 'private property', "people n't", '( http', '-- --', "n't even", "wo n't", "n't want", '^| [', ') ^|', "n't know", 'free market', "n't think", "would n't", "ca n't", '( https', '] (']


# neoliberal unigrams


```python
x_df = reddit_top_unigrams_df.neoliberal.tolist()
y_df = neoliberal_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 90
    ['system', 'market', 'saying', 'american', 'money', 'first', 'everyone', 'though', 'mean', 'social', 'state', 'literally', 'probably', "'d", "'ll", 'policy', 'years', 'political', 'point', 'got', 'world', 'party', "'", 'said', 'every', 'tax', 'shit', 'country', 'white', 'someone', 'pretty', 'never', 'work', 'many', 'lot', 'sub', 'need', 'go', 'ca', 'well', 'something', 'better', 'free', '’', 'things', 'thing', 'bernie', 'bad', 'still', 'amp', 'see', "'ve", 'say', 'going', 'way', 'could', 'actually', 'government', 'time', 'make', 'http', 'much', 'know', 'us', 'right', 'want', 'really', 'good', 'even', 'take', 'also', 'get', "'m", 'one', 'trump', "'re", 'think', 'would', 'like', 'https', '[', ']', 'gt', '``', 'people', "''", '(', ')', "n't", "'s"]

    middle words: 88
    ['literally', 'system', 'everyone', 'idea', 'problem', 'shit', 'country', 'political', 'years', 'anything', 'money', 'since', 'party', 'every', 'social', 'free', 'world', 'never', 'policy', 'first', 'said', 'many', 'sure', 'state', 'ca', "'ll", 'less', 'got', 'mean', 'tax', 'work', 'someone', 'things', 'need', 'point', 'go', 'thing', "'d", 'http', 'government', 'pretty', 'well', 'lot', 'bad', 'probably', 'something', 'still', 'trump', 'better', 'amp', 'see', 'say', 'us', 'actually', 'going', 'though', 'take', 'right', "'ve", 'way', 'time', 'make', 'want', 'could', 'know', 'much', 'even', 'really', 'also', 'get', 'good', "'re", 'one', "'m", 'think', 'https', ']', '[', '``', 'gt', 'would', "''", 'like', 'people', '(', ')', "n't", "'s"]

    bottom words: 87
    ["'", 'years', 'immigration', 'part', 'everyone', '’', 'free', 'political', 'social', 'without', 'problem', "'ll", 'someone', 'first', 'policy', 'every', 'state', 'less', 'money', 'system', 'anything', 'world', 'http', 'never', 'many', 'saying', 'sure', 'probably', 'pretty', 'trump', 'tax', 'work', 'country', 'said', 'take', "'d", 'need', 'mean', 'go', 'bad', 'ca', 'lot', 'thing', 'better', 'well', 'government', 'going', 'things', 'though', "'ve", 'something', 'time', 'actually', 'say', 'way', 'still', 'could', 'see', 'point', 'right', 'want', 'know', 'make', 'us', 'much', 'really', 'https', ']', '[', 'also', 'good', 'even', 'get', 'one', "'re", "'m", 'think', '``', 'like', "''", 'gt', 'would', '(', ')', 'people', "'s", "n't"]


# neoliberal bigrams


```python
x_df = reddit_top_bigrams_df.neoliberal.tolist()
y_df = neoliberal_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 82
    ['look like', 'first place', 'people would', "'' n't", "'' 's", 'minimum wage', 'good thing', "'re going", "n't actually", 'black people', ") n't", 'high school', "'s going", 'foreign policy', 'people think', 'climate change', ") 's", "n't care", "'s really", "n't make", 'saying ``', "'ve seen", 'people like', 'middle class', "'s pretty", "'s good", "bernie 's", 'white people', "'s also", 'even though', "n't see", "n't mean", "'m going", 'pretty much', "n't understand", 'years ago', 'seems like', "people 's", 'say ``', 'many people', 'free market', 'health care', 'things like', "let 's", ') gt', 'bernie sanders', 'discussion thread', 'feel like', 'like ``', "'m sure", "could n't", 'open borders', "'s like", "people n't", 'gt gt', "trump 's", 'united states', "n't like", "n't really", 'https //www.youtube.com/watch', "n't get", '( (', 'free trade', 'single payer', "n't want", ') )', 'gon na', "think 's", ') [', "'' ``", "n't even", '[ ]', "wo n't", "n't think", '\\ gt', "n't know", "would n't", 'hot take', "ca n't", '( http', '( https', '] (']

    middle words: 83
    ["n't believe", 'climate change', "gt n't", "'' n't", 'good thing', 'income tax', "gt 's", "know 's", "'' 's", 'poor people', '( (', 'sounds like', "n't care", "'m saying", 'look like', 'first place', "'s really", 'foreign policy', "n't need", 'open borders', ") n't", "'s pretty", 'people would', "'s also", 'lot people', "n't actually", 'makes sense', "'ve seen", "trump 's", 'people like', 'things like', "'m going", "'re going", 'free trade', 'like ``', "'s going", "n't understand", 'years ago', 'say ``', "really n't", "n't make", "people 's", 'many people', '\\ gt', 'united states', 'even though', ') )', 'something like', "'s good", "n't mean", "let 's", ") 's", 'high school', "'s like", 'single payer', "could n't", 'seems like', "n't see", "n't like", 'feel like', 'gt gt', 'gon na', 'pretty much', "n't get", 'https //www.youtube.com/watch', ') [', "people n't", "'' ``", "'m sure", 'hot take', "wo n't", "n't even", "n't want", '[ ]', "n't really", "think 's", "n't know", "would n't", "n't think", '( http', "ca n't", '( https', '] (']

    bottom words: 85
    [') )', "'m going", 'global poor', "trump 's", 'high school', "n't matter", 'good thing', "'ve seen", 'lot people', 'health care', "really n't", 'black people', "'s going", "n't believe", 'first place', 'foreign policy', "gt n't", 'say ``', 'middle class', 'minimum wage', "n't actually", "'re talking", 'poor people', "'' n't", "'s pretty", 'free trade', 'sounds like', "'re going", '\\ gt', ") n't", 'makes sense', 'people like', "'s also", 'many people', "n't need", "'s good", "n't say", 'white people', 'feel like', 'something like', "'s really", 'things like', 'years ago', 'even though', "could n't", ") 's", "people 's", "n't make", 'hot take', 'united states', "gt 's", "n't understand", "n't care", 'gon na', "'s like", "n't like", "'' ``", "n't mean", ') [', "let 's", 'gt gt', 'people would', 'https //www.youtube.com/watch', 'pretty much', 'seems like', 'single payer', '[ ]', 'open borders', "'m saying", "n't get", "people n't", "'m sure", "n't want", "n't see", "n't even", "think 's", "n't really", "wo n't", "n't know", '( http', "would n't", "n't think", "ca n't", '( https', '] (']


# politics unigrams


```python
x_df = reddit_top_unigrams_df.politics.tolist()
y_df = politics_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 70
    ['law', 'http', 'republicans', 'daca', 'vote', 'many', 'anything', 'take', 'point', 'someone', 'government', 'see', 'work', 'need', 'every', 'years', "'ll", 'news', 'https', 'back', 'things', 'white', "'ve", 'obama', 'well', 'something', 'go', 'shit', 'also', ']', '[', 'never', 'ca', 'still', 'thing', 'said', 'say', 'actually', 'money', 'much', 'make', 'way', 'good', 'want', 'country', 'could', 'really', 'time', 'going', 'us', 'right', "'m", 'president', 'know', 'even', 'think', 'get', "'re", 'one', '(', ')', 'gt', 'would', 'like', '``', "''", 'people', 'trump', "n't", "'s"]

    middle words: 82
    ['current', 'daca', 'republicans', 'others', 'post', 'personal', 'regarding', 'news', 'action', 'hate', 'work', 'every', 'white', 'obama', 'years', 'vote', 'government', 'take', 'many', 'never', 'http', 'anything', 'things', 'someone', 'point', 'shit', "'ll", 'back', 'may', 'go', 'need', 'actually', 'well', 'something', 'said', 'ca', 'also', 'downvotes', 'thing', "'ve", 'money', 'still', 'subreddit', 'much', 'say', 'good', 'comments', 'want', 'president', 'country', 'questions', 'make', 'time', 'really', 'way', 'could', 'going', 'us', 'right', 'know', 'even', 'get', "'m", 'one', 'please', 'see', 'think', "'re", 'https', 'gt', 'like', 'would', '``', '[', ']', "''", 'people', 'trump', '(', ')', "n't", "'s"]

    bottom words: 72
    ['well', 'also', 'still', 'money', 'much', 'say', 'said', 'http', 'way', 'president', 'country', 'good', 'time', 'really', 'going', 'could', 'want', 'make', 'even', 'right', 'know', 'us', 'ban', 'report', 'civil', 'research', 'ideas', 'general', 'attack', 'others', 'personal', 'speech', 'one', '*i', '/message/compose/', 'performed', 'current', 'contact', 'concerns', 'hate', 'post', 'get', 'bot', "'m", 'action', 'to=/r/politics', 'moderators', 'automatically', 'think', "'re", 'may', 'regarding', 'gt', 'like', '``', "''", 'would', 'trump', 'downvotes', 'comments', 'people', 'subreddit', 'questions', 'see', 'https', "'s", 'please', "n't", '(', ']', '[', ')']


# politics bigrams


```python
x_df = reddit_top_bigrams_df.politics.tolist()
y_df = politics_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 25
    ["'s going", 'fake news', "could n't", 'donald trump', 'fox news', "people n't", "let 's", "'m sure", 'united states', "n't want", "'' ``", "'s like", "think 's", "n't get", 'white house', "n't think", "n't even", "wo n't", "n't know", "would n't", '( http', '( https', "trump 's", '] (', "ca n't"]

    middle words: 97
    ['united states', 'donald trump', "'s like", "'s going", "people n't", "let 's", "n't get", 'white house', "n't want", "'' ``", "think 's", "'m sure", 'regarding removal', "n't even", "wo n't", "n't think", "n't know", "would n't", '( http', "trump 's", 'civil discussion', 'personal insults', 'discussion ]', 'permanent ban', 'https //www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/', 'downvotes comments', 'result permanent', 'users personal', 'may disabled', 'civility questions', 'user civility', 'subreddit [', 'general courteous', 'section may', 'others attack', '//www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/ )', 'incivility violations', '//www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/dkybt0h/ )', 'rules please', '/r/politics/wiki/index wiki_be_civil', 'troll accusations', 'wiki_be_civil )', 'ideas users', 'current research', 'https //www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/dkybt0h/', 'accusations hate', 'courteous others', 'violations result', 'violation rules', '*** *i', 'insults shill', 'disabled please', 'regarding effect', 'comments violation', 'speech incivility', 'reminder subreddit', 'downvotes user', 'ban see', 'report downvotes', 'shill troll', 'questions ***', '( /r/politics/wiki/index', 'effect downvotes', 'research regarding', 'see [', 'attack ideas', ') current', '[ post', 'please see', 'please report', '[ civil', 'post ]', ') general', 'see comments', 'comments section', 'hate speech', 'faq ]', '[ faq', 'questions concerns', '[ contact', 'subreddit ]', '*i bot', '( /message/compose/', 'performed automatically', 'action performed', 'automatically please', '/message/compose/ to=/r/politics', 'contact moderators', 'bot action', 'please [', 'to=/r/politics )', 'moderators subreddit', ') questions', ') [', "ca n't", '( https', '] (']

    bottom words: 83
    ["wo n't", "trump 's", "n't know", "would n't", '( http', 'regarding removal', "ca n't", 'post ]', ') general', '[ civil', 'please see', '[ post', 'please report', ') current', 'discussion ]', 'attack ideas', 'https //www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/', 'permanent ban', 'troll accusations', 'violations result', '( /r/politics/wiki/index', 'shill troll', 'courteous others', 'questions ***', 'violation rules', 'regarding effect', 'speech incivility', 'user civility', 'effect downvotes', 'general courteous', '//www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/ )', '//www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/dkybt0h/ )', 'may disabled', '/r/politics/wiki/index wiki_be_civil', 'downvotes user', 'users personal', 'wiki_be_civil )', 'ban see', 'accusations hate', 'others attack', 'rules please', 'subreddit [', 'current research', 'insults shill', 'disabled please', 'research regarding', 'comments violation', 'reminder subreddit', '*** *i', 'report downvotes', 'ideas users', 'https //www.reddit.com/r/politics/comments/6o1ipb/research_on_the_effect_downvotes_have_on_user/dkybt0h/', 'section may', 'incivility violations', 'civility questions', 'see [', 'result permanent', 'comments section', 'downvotes comments', 'personal insults', 'hate speech', '[ faq', 'faq ]', 'see comments', 'civil discussion', ') [', '[ contact', 'subreddit ]', 'questions concerns', 'performed automatically', 'please [', 'action performed', '/message/compose/ to=/r/politics', 'contact moderators', '*i bot', 'automatically please', 'moderators subreddit', 'to=/r/politics )', '( /message/compose/', 'bot action', ') questions', '( https', '] (']


# worldnews unigrams


```python
x_df = reddit_top_unigrams_df.worldnews.tolist()
y_df = worldnews_top_unigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 94
    ['saying', 'america', 'far', 'bomb', 'korean', 'better', 'enough', 'mean', 'work', 'everyone', 'eu', "'ll", 'attack', 'use', 'probably', 'nothing', 'back', 'anything', 'every', 'shit', 'sure', 'japan', 'things', "'d", 'weapons', 'kim', 'http', 'lot', 'government', 'need', 'ca', 'power', 'never', 'take', 'first', 'thing', 'said', 'point', 'go', 'south', 'something', 'many', 'nukes', 'see', 'good', 'well', 'say', 'countries', "'ve", 'still', 'military', 'right', 'make', 'actually', 'years', 'russia', 'going', 'way', '[', ']', 'really', 'also', 'trump', 'https', 'want', 'much', 'time', 'know', "'m", 'country', 'could', 'world', 'get', 'even', 'think', "'re", 'war', 'nuclear', 'one', 'gt', 'north', 'korea', 'nk', 'china', 'like', '``', "''", '(', ')', 'people', 'us', 'would', "n't", "'s"]

    middle words: 90
    ['nothing', 'everyone', 'work', 'far', 'back', 'probably', "'ll", 'government', 'use', 'better', 'every', 'saying', 'mean', 'japan', 'lot', 'enough', 'kim', 'things', 'shit', 'never', 'weapons', "'d", 'anything', 'south', 'countries', 'take', 'sure', 'ca', 'military', 'first', 'power', 'need', 'thing', 'go', 'many', 'say', 'something', "'ve", 'nukes', 'good', 'https', 'years', 'actually', 'well', 'said', 'russia', 'point', 'still', 'see', 'trump', 'right', 'way', 'make', 'much', 'going', 'time', 'really', 'country', 'also', 'reddit', 'want', 'world', 'know', 'get', "'m", 'could', 'even', 'nuclear', 'war', "'re", 'north', 'one', 'think', 'http', 'gt', 'korea', '``', 'china', 'nk', 'like', "''", ']', '[', 'us', 'people', 'would', '(', ')', "'s", "n't"]

    bottom words: 91
    ["'ll", 'use', 'work', 'lot', 'everyone', 'stop', 'every', "'d", 'mean', 'things', 'america', 'back', 'let', 'https', 'weapons', 'japan', 'kim', 'government', 'first', 'power', 'eu', 'many', 'saying', 'nothing', 'thing', 'better', 'sure', 'anything', 'shit', 'south', 'ca', 'take', 'something', 'need', 'good', 'never', "'ve", 'still', 'point', 'said', 'go', 'actually', 'countries', 'say', 'military', 'see', 'years', 'well', 'nukes', 'make', 'way', 'russia', 'really', 'right', 'much', 'time', 'going', 'also', 'reddit', "'m", 'could', 'trump', 'know', 'want', 'country', 'even', 'nuclear', 'http', 'get', 'world', 'one', 'think', "'re", 'north', '[', ']', 'gt', 'war', 'korea', '``', "''", 'like', 'nk', 'china', 'people', '(', ')', 'us', 'would', "'s", "n't"]


# worldnews bigrams


```python
x_df = reddit_top_bigrams_df.worldnews.tolist()
y_df = worldnews_top_bigrams_df
top_list = []
mid_list = []
bottom_list = []
for y in y_df.loc[:, '3']:
    if y in x_df:
        top_list.append(y)
for y in y_df.loc[:, '2']:
    if y in x_df:
        mid_list.append(y)
for y in y_df.loc[:, '1']:
    if y in x_df:
        bottom_list.append(y)
print("top words: " + str(len(top_list)) + "\n" + str(top_list))
print("\nmiddle words: " + str(len(mid_list)) + "\n" + str(mid_list))
print("\nbottom words: " + str(len(bottom_list)) + "\n" + str(bottom_list))
```

    top words: 56
    ["'m saying", "china n't", 'us would', ') [', 'amp 039', 'korea would', 'rest world', 'hydrogen bomb', 'kim jong', "n't need", 'north koreans', 'china russia', "n't make", 'middle east', 'cold war', 'amp quot', 'china would', "korea 's", "nk 's", '| [', 'gon na', "'s going", "could n't", 'many people', "n't see", "'' ``", "n't like", "n't mean", "'' )", 'nuclear war', "n't really", 'pretty much', "china 's", "people n't", "n't get", "let 's", "think 's", "'s like", ') |', "'m sure", 'years ago', 'united states', "n't even", 'north korean', "n't think", "n't know", "n't want", 'nuclear weapons', "wo n't", 'south korea', "would n't", '( http', "ca n't", '( https', '] (', 'north korea']

    middle words: 75
    ['pretty much', "could n't", "n't see", "china 's", "n't get", "'s like", 'north korean', 'years ago', "n't really", 'united states', "think 's", 'amp 039', "'m sure", 'meet minimum', 'reddit ]', 'karma account', 'reddit suggest', 'self-promotion reddit', 'requirements /r/worldnews', 'manual approval', 'domain new', '//www.redditblog.com/2014/07/how-reddit-works.html )', 'regarding spam', 'reddit 101', 'account meet', 'pending manual', 'reddit guidelines', 'whatconstitutesspam )', '/r/worldnews ]', 'minimum karma', 'http //www.reddit.com/r/help/comments/2bx3cj/reddit_101/', 'reddit works', 'http //www.reddit.com/r/worldnews/wiki/rules', 'may also', 'account age', 'works ]', 'suggest read', "n't even", 'read [', 'also want', 'amp quot', ') *i', '[ wiki', '| [', "'' )", "n't want", 'nuclear weapons', '/message/compose/ to=/r/worldnews', '( /message/compose/', "n't think", '*i bot', 'action performed', '[ contact', 'automatically please', 'questions concerns', 'moderators subreddit', 'to=/r/worldnews )', 'subreddit ]', 'bot action', 'performed automatically', 'please [', ') questions', "n't know", "wo n't", 'south korea', ') |', 'contact moderators', "would n't", '( https', "ca n't", '[ reddit', ') [', 'north korea', '( http', '] (']

    bottom words: 75
    ["n't like", "n't mean", "'m saying", 'china russia', "people n't", "'s like", "think 's", 'gon na', "china 's", 'north korean', "n't really", "n't get", 'years ago', "'m sure", "let 's", 'united states', ') |', 'reddit ]', 'works ]', 'account age', 'reddit suggest', 'self-promotion reddit', 'minimum karma', 'manual approval', 'http //www.reddit.com/r/help/comments/2bx3cj/reddit_101/', 'reddit 101', 'domain new', 'pending manual', 'whatconstitutesspam )', 'reddit guidelines', '/r/worldnews ]', 'requirements /r/worldnews', '//www.redditblog.com/2014/07/how-reddit-works.html )', 'account meet', 'regarding spam', 'suggest read', 'meet minimum', 'reddit works', 'http //www.reddit.com/r/worldnews/wiki/rules', 'karma account', 'read [', 'may also', 'also want', '[ wiki', ') *i', "n't even", "n't know", "n't want", "n't think", 'nuclear weapons', '/message/compose/ to=/r/worldnews', '( /message/compose/', '( https', 'automatically please', '[ contact', 'subreddit ]', 'moderators subreddit', 'to=/r/worldnews )', 'bot action', 'action performed', 'performed automatically', 'please [', 'questions concerns', '*i bot', ') questions', 'south korea', "wo n't", 'contact moderators', "would n't", "ca n't", '[ reddit', ') [', 'north korea', '( http', '] (']


# analysis

Above shows the tokens I was looking for.  Each token is

# Runtime

It took almost 6 hours for the code to run.


```python
time2 = time.time()
time_dif = time2 - time1
hours = (time_dif // 3600)
minutes = (time_dif // 60) - (hours * 60)
seconds = (time_dif // 1) - ((hours * 3600) + (minutes * 60))
print("hours: " + str(hours))
print("minutes: " + str(minutes))
print("seconds: " + str(seconds))
```

    hours: 5.0
    minutes: 47.0
    seconds: 26.0
