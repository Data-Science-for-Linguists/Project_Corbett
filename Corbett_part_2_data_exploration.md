
# Robert Corbett

Reddit post exploration part 2

This code is used to just find the number of posts I extracted from each subreddit. I have a large amount of data to choose from so I am only going to use a couple months of posts. The number of posts each subreddit has varies drastically. Some subreddits have hundreds of thousands of posts while some of a couple hundred. For the project, I want to use subreddits that have atleast 50,000 posts to work with.

I also want to make sure the body of the posts are of a useful length.  Looking at the average lengths of posts, I decided to make sure that each post is atleast 150 characters.  

In the end, I managed to extract 50,000 posts of atleast 150 characters long from 8 subreddits: Conservative, Libertarian, neoliberal, politics, worldnews, Android, boardgames and hockey.  They are saved as csv files on my local machine.  

This code took about 5 hours to run.


```python
import json
import numpy as np
from matplotlib import pyplot as plt
import nltk
import pandas as pd
```

I started by counting the number of lines in the json file for September, 2017.  I saved the total in a variable with the same name and printed the results.


```python
path = "../../Reddit_Data_Trimmed/September_17/"

print("The number of posts in each subreddit for the month of September, 2017")

x = 0
with open(path + "Android.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Android: " + str(x))
Android = x

x = 0
with open(path + "Archaeology.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Archaeology: " + str(x))
Archaeology = x

x = 0
with open(path + "AskTrumpSupporters.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of AskTrumpSupporters: " + str(x))
AskTrumpSupporters = x

x = 0
with open(path + "boardgames.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of boardgames: " + str(x))
boardgames = x

x = 0
with open(path + "Conservative.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Conservative: " + str(x))
Conservative = x

x = 0
with open(path + "DCcomics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of DCcomics: " + str(x))
DCcomics = x

x = 0
with open(path + "food.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of food: " + str(x))
food = x

x = 0
with open(path + "Futurism.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Futurism: " + str(x))
Futurism = x

x = 0
with open(path + "geopolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of geopolitics: " + str(x))
geopolitics = x

x = 0
with open(path + "gunpolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of gunpolitics: " + str(x))
gunpolitics = x

x = 0
with open(path + "headphones.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of headphones: " + str(x))
headphones = x

x = 0
with open(path + "hockey.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of hockey: " + str(x))
hockey = x

x = 0
with open(path + "indie_rock.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indie_rock: " + str(x))
indie_rock = x

x = 0
with open(path + "indieheads.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indieheads: " + str(x))
indieheads = x

x = 0
with open(path + "Libertarian.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Libertarian: " + str(x))
Libertarian = x

x = 0
with open(path + "MarchAgainstTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of MarchAgainstTrump: " + str(x))
MarchAgainstTrump = x

x = 0
with open(path + "neoliberal.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of neoliberal: " + str(x))
neoliberal = x

x = 0
with open(path + "NeverTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of NeverTrump: " + str(x))
NeverTrump = x

x = 0
with open(path + "photography.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of photography: " + str(x))
photography = x

x = 0
with open(path + "politics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of politics: " + str(x))
politics = x

x = 0
with open(path + "PoliticsWithoutTheBan.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of PoliticsWithoutTheBan: " + str(x))
PoliticsWithoutTheBan = x

x = 0
with open(path + "seinfeld.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of seinfeld: " + str(x))
seinfeld = x

x = 0
with open(path + "The_Donald.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of The_Donald: " + str(x))
The_Donald = x

x = 0
with open(path + "worldnews.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of worldnews: " + str(x))
worldnews = x
```

    The number of posts in each subreddit for the month of September, 2017
    Len of Android: 165509
    Len of Archaeology: 574
    Len of AskTrumpSupporters: 41290
    Len of boardgames: 61240
    Len of Conservative: 55766
    Len of DCcomics: 24647
    Len of food: 69709
    Len of Futurism: 197
    Len of geopolitics: 7313
    Len of gunpolitics: 4390
    Len of headphones: 29723
    Len of hockey: 199269
    Len of indie_rock: 491
    Len of indieheads: 35572
    Len of Libertarian: 103633
    Len of MarchAgainstTrump: 11761
    Len of neoliberal: 212190
    Len of NeverTrump: 180
    Len of photography: 36431
    Len of politics: 1390463
    Len of PoliticsWithoutTheBan: 0
    Len of seinfeld: 5277
    Len of The_Donald: 904159
    Len of worldnews: 777276
    

I did the same for August, 2017 but added the totals to the totals for September and printed the totals for the month below.


```python
path = "../../Reddit_Data_Trimmed/August_17/"

print("The number of posts in each subreddit for the month of August, 2017")

x = 0
with open(path + "Android.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Android: " + str(x))
Android = Android + x

x = 0
with open(path + "Archaeology.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Archaeology: " + str(x))
Archaeology = Archaeology + x

x = 0
with open(path + "AskTrumpSupporters.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of AskTrumpSupporters: " + str(x))
AskTrumpSupporters = AskTrumpSupporters + x

x = 0
with open(path + "boardgames.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of boardgames: " + str(x))
boardgames = boardgames + x

x = 0
with open(path + "Conservative.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Conservative: " + str(x))
Conservative = Conservative + x

x = 0
with open(path + "DCcomics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of DCcomics: " + str(x))
DCcomics = DCcomics + x

x = 0
with open(path + "food.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of food: " + str(x))
food = food + x

x = 0
with open(path + "Futurism.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Futurism: " + str(x))
Futurism = Futurism + x

x = 0
with open(path + "geopolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of geopolitics: " + str(x))
geopolitics = geopolitics + x

x = 0
with open(path + "gunpolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of gunpolitics: " + str(x))
gunpolitics = gunpolitics + x

x = 0
with open(path + "headphones.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of headphones: " + str(x))
headphones = headphones + x

x = 0
with open(path + "hockey.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of hockey: " + str(x))
hockey = hockey + x

x = 0
with open(path + "indie_rock.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indie_rock: " + str(x))
indie_rock = indie_rock + x

x = 0
with open(path + "indieheads.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indieheads: " + str(x))
indieheads = indieheads + x

x = 0
with open(path + "Libertarian.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Libertarian: " + str(x))
Libertarian = Libertarian + x

x = 0
with open(path + "MarchAgainstTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of MarchAgainstTrump: " + str(x))
MarchAgainstTrump = MarchAgainstTrump + x

x = 0
with open(path + "neoliberal.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of neoliberal: " + str(x))
neoliberal = neoliberal + x

x = 0
with open(path + "NeverTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of NeverTrump: " + str(x))
NeverTrump = NeverTrump + x

x = 0
with open(path + "photography.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of photography: " + str(x))
photography = photography + x

x = 0
with open(path + "politics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of politics: " + str(x))
politics = politics + x

x = 0
with open(path + "PoliticsWithoutTheBan.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of PoliticsWithoutTheBan: " + str(x))
PoliticsWithoutTheBan = PoliticsWithoutTheBan + x

x = 0
with open(path + "seinfeld.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of seinfeld: " + str(x))
seinfeld = seinfeld + x

x = 0
with open(path + "The_Donald.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of The_Donald: " + str(x))
The_Donald = The_Donald + x

x = 0
with open(path + "worldnews.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of worldnews: " + str(x))
worldnews = worldnews + x
```

    The number of posts in each subreddit for the month of August, 2017
    Len of Android: 143884
    Len of Archaeology: 805
    Len of AskTrumpSupporters: 45575
    Len of boardgames: 60254
    Len of Conservative: 66364
    Len of DCcomics: 27988
    Len of food: 68758
    Len of Futurism: 187
    Len of geopolitics: 6269
    Len of gunpolitics: 2836
    Len of headphones: 30372
    Len of hockey: 157833
    Len of indie_rock: 320
    Len of indieheads: 39867
    Len of Libertarian: 94092
    Len of MarchAgainstTrump: 19208
    Len of neoliberal: 184248
    Len of NeverTrump: 322
    Len of photography: 41420
    Len of politics: 1824215
    Len of PoliticsWithoutTheBan: 10
    Len of seinfeld: 4045
    Len of The_Donald: 1057483
    Len of worldnews: 741492
    

The data for July was corrupted and I could not get the json entries to open.  So I skipped July and am going to use June instead.  I do not see skipping July causing any problems with the project.


```python
path = "../../Reddit_Data_Trimmed/June_17/"

print("The number of posts in each subreddit for the month of June, 2017")
x = 0
with open(path + "Android.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Android: " + str(x))
Android = Android + x

x = 0
with open(path + "Archaeology.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Archaeology: " + str(x))
Archaeology = Archaeology + x

x = 0
with open(path + "AskTrumpSupporters.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of AskTrumpSupporters: " + str(x))
AskTrumpSupporters = AskTrumpSupporters + x

x = 0
with open(path + "boardgames.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of boardgames: " + str(x))
boardgames = boardgames + x

x = 0
with open(path + "Conservative.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Conservative: " + str(x))
Conservative = Conservative + x

x = 0
with open(path + "DCcomics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of DCcomics: " + str(x))
DCcomics = DCcomics + x

x = 0
with open(path + "food.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of food: " + str(x))
food = food + x

x = 0
with open(path + "Futurism.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Futurism: " + str(x))
Futurism = Futurism + x

x = 0
with open(path + "geopolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of geopolitics: " + str(x))
geopolitics = geopolitics + x

x = 0
with open(path + "gunpolitics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of gunpolitics: " + str(x))
gunpolitics = gunpolitics + x

x = 0
with open(path + "headphones.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of headphones: " + str(x))
headphones = headphones + x

x = 0
with open(path + "hockey.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of hockey: " + str(x))
hockey = hockey + x

x = 0
with open(path + "indie_rock.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indie_rock: " + str(x))
indie_rock = indie_rock + x

x = 0
with open(path + "indieheads.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of indieheads: " + str(x))
indieheads = indieheads + x

x = 0
with open(path + "Libertarian.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of Libertarian: " + str(x))
Libertarian = Libertarian + x

x = 0
with open(path + "MarchAgainstTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of MarchAgainstTrump: " + str(x))
MarchAgainstTrump = MarchAgainstTrump + x

x = 0
with open(path + "neoliberal.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of neoliberal: " + str(x))
neoliberal = neoliberal + x

x = 0
with open(path + "NeverTrump.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of NeverTrump: " + str(x))
NeverTrump = NeverTrump + x

x = 0
with open(path + "photography.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of photography: " + str(x))
photography = photography + x

x = 0
with open(path + "politics.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of politics: " + str(x))
politics = politics + x

x = 0
with open(path + "PoliticsWithoutTheBan.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of PoliticsWithoutTheBan: " + str(x))
PoliticsWithoutTheBan = PoliticsWithoutTheBan + x

x = 0
with open(path + "seinfeld.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of seinfeld: " + str(x))
seinfeld = seinfeld + x

x = 0
with open(path + "The_Donald.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of The_Donald: " + str(x))
The_Donald = The_Donald + x

x = 0
with open(path + "worldnews.txt", 'r') as file_in:
    for line in file_in:
        x = x + 1
print("Len of worldnews: " + str(x))
worldnews = worldnews + x
```

    The number of posts in each subreddit for the month of June, 2017
    Len of Android: 124963
    Len of Archaeology: 524
    Len of AskTrumpSupporters: 44644
    Len of boardgames: 63268
    Len of Conservative: 47492
    Len of DCcomics: 33775
    Len of food: 67223
    Len of Futurism: 161
    Len of geopolitics: 6086
    Len of gunpolitics: 3777
    Len of headphones: 26346
    Len of hockey: 486157
    Len of indie_rock: 250
    Len of indieheads: 39393
    Len of Libertarian: 83183
    Len of MarchAgainstTrump: 67415
    Len of neoliberal: 160842
    Len of NeverTrump: 327
    Len of photography: 40423
    Len of politics: 1818730
    Len of PoliticsWithoutTheBan: 0
    Len of seinfeld: 3882
    Len of The_Donald: 1158567
    Len of worldnews: 909968
    

Finally, below I printed the totals for each subreddit for the three months.


```python
print("The number of posts in each subreddit for both months")
print("Android: " + str(Android))
print("Archaeology: " + str(Archaeology))
print("AskTrumpSupporters: " + str(AskTrumpSupporters))
print("boardgames: " + str(boardgames))
print("Conservative: " + str(Conservative))
print("DCcomics: " + str(DCcomics))
print("food: " + str(food))
print("Futurism: " + str(Futurism))
print("geopolitics: " + str(geopolitics))
print("gunpolitics: " + str(gunpolitics))
print("headphones: " + str(headphones))
print("hockey: " + str(hockey))
print("indie_rock: " + str(indie_rock))
print("indieheads: " + str(indieheads))
print("Libertarian: " + str(Libertarian))
print("MarchAgainstTrump: " + str(MarchAgainstTrump))
print("neoliberal: " + str(neoliberal))
print("NeverTrump: " + str(NeverTrump))
print("photography: " + str(photography))
print("politics: " + str(politics))
print("PoliticsWithoutTheBan: " + str(PoliticsWithoutTheBan))
print("seinfeld: " + str(seinfeld))
print("The_Donald: " + str(The_Donald))
print("worldnews: " + str(worldnews))
```

    The number of posts in each subreddit for both months
    Android: 434356
    Archaeology: 1903
    AskTrumpSupporters: 131509
    boardgames: 184762
    Conservative: 169622
    DCcomics: 86410
    food: 205690
    Futurism: 545
    geopolitics: 19668
    gunpolitics: 11003
    headphones: 86441
    hockey: 843259
    indie_rock: 1061
    indieheads: 114832
    Libertarian: 280908
    MarchAgainstTrump: 98384
    neoliberal: 557280
    NeverTrump: 829
    photography: 118274
    politics: 5033408
    PoliticsWithoutTheBan: 10
    seinfeld: 13204
    The_Donald: 3120209
    worldnews: 2428736
    

For the project, I want political subreddits, both with and without political biases.  So, for biased political subreddits, I am going to use "Conservative", "Libertarian" and "neoliberal".  For nonbiased political subreddits, I am going to use "politics" and "worldnews".  As a control, I will use non political subreddits.  I will use "Android", "boardgames" and "hockey".

Next, I want to find out the average length of posts.  Many of the posts are very short.  I want to use the longest posts I can while mantaining the 50,000 post goal.  Below, I calculated the average lengths and how many posts are greater than 50, 100, 150, 200, 250, and 300 characters


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print("Number of posts in Conservative     : " + str(num_posts))
print("Average length of body              : " + str(num_chars/num_posts))
print("number of posts over 50   characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150  characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300  characters: " + str(over_300))
```

    Number of posts in Conservative     : 169622
    Average length of body              : 218.35662826755964
    number of posts over 50   characters: 122319
    number of posts over 100  characters: 91978
    number of posts over 150  characters: 69784
    number of posts over 200  characters: 54493
    number of posts over 250  characters: 43749
    number of posts over 300  characters: 35778
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print("Number of posts in Libertarian      : " + str(num_posts))
print("Average length of body              : " + str(num_chars/num_posts))
print("number of posts over 50   characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150  characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300  characters: " + str(over_300))
```

    Number of posts in Libertarian      : 280908
    Average length of body              : 287.30290344169623
    number of posts over 50   characters: 227488
    number of posts over 100  characters: 177428
    number of posts over 150  characters: 139855
    number of posts over 200  characters: 112474
    number of posts over 250  characters: 92363
    number of posts over 300  characters: 77299
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    557280
    Average length of body: 153.20315281366638
    number of posts over 50  characters: 350206
    number of posts over 100  characters: 221754
    number of posts over 150 characters: 150061
    number of posts over 200  characters: 108318
    number of posts over 250  characters: 82134
    number of posts over 300 characters: 64611
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    5033408
    Average length of body: 202.61892399741885
    number of posts over 50  characters: 3629278
    number of posts over 100  characters: 2579653
    number of posts over 150 characters: 1897668
    number of posts over 200  characters: 1456237
    number of posts over 250  characters: 1155323
    number of posts over 300 characters: 941578
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    2428736
    Average length of body: 200.97222794078894
    number of posts over 50  characters: 1691007
    number of posts over 100  characters: 1215445
    number of posts over 150 characters: 901245
    number of posts over 200  characters: 693553
    number of posts over 250  characters: 548246
    number of posts over 300 characters: 444789
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    434356
    Average length of body: 182.29371529344593
    number of posts over 50  characters: 319113
    number of posts over 100  characters: 222110
    number of posts over 150 characters: 156447
    number of posts over 200  characters: 113629
    number of posts over 250  characters: 85317
    number of posts over 300 characters: 65906
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    184762
    Average length of body: 254.62208138037042
    number of posts over 50  characters: 148076
    number of posts over 100  characters: 116951
    number of posts over 150 characters: 91284
    number of posts over 200  characters: 72144
    number of posts over 250  characters: 57927
    number of posts over 300 characters: 47144
    


```python
num_posts = 0
num_chars = 0
over_50 = 0
over_100 = 0
over_150 = 0
over_200 = 0
over_250 = 0
over_300 = 0

with open("../../Reddit_Data_trimmed/September_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            
        
        num_posts = num_posts + 1

with open("../../Reddit_Data_trimmed/August_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1   
                           
        num_posts = num_posts + 1
            
with open("../../Reddit_Data_trimmed/June_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        num_chars = num_chars + len(line_in['body'])
        if len(line_in['body'])>=50:
            over_50 = over_50 + 1
        if len(line_in['body'])>=100:
            over_100 = over_100 + 1
        if len(line_in['body'])>=150:
            over_150 = over_150 + 1
        if len(line_in['body'])>=200:
            over_200 = over_200 + 1
        if len(line_in['body'])>=250:
            over_250 = over_250 + 1
        if len(line_in['body'])>=300:
            over_300 = over_300 + 1            

        num_posts = num_posts + 1

print(num_posts)
print("Average length of body: " + str(num_chars/num_posts))
print("number of posts over 50  characters: " + str(over_50))
print("number of posts over 100  characters: " + str(over_100))
print("number of posts over 150 characters: " + str(over_150))
print("number of posts over 200  characters: " + str(over_200))
print("number of posts over 250  characters: " + str(over_250))
print("number of posts over 300 characters: " + str(over_300))
```

    843259
    Average length of body: 114.62820082560637
    number of posts over 50  characters: 501472
    number of posts over 100  characters: 285673
    number of posts over 150 characters: 179058
    number of posts over 200  characters: 120722
    number of posts over 250  characters: 86133
    number of posts over 300 characters: 63933
    

If I want to have atleast 50,000 posts from each subreddit, I will have to work with posts with lengths greater than 150 characters.  The subreddit with the least number of posts greater than 150 characters is "Conservative" with 69,784 posts.

I created an empty dataframe named temp_df and I populated it with the 11 elements of each post: parent_id, author, distinguished, body, gilded, score, author_flair_css_class, stickied, retreived_on, author_flair_text and id.  I probably won't use most of this meta data in my project, but I would rather preserve it incase I decide its useful later.  I then saved the dataframes as csv files to my local machine.  I will try to put the files on github.

The dataframes hold the data from the first 50,000 posts over length 150 characters in each subreddit.


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Android.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Android.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Android.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/boardgames.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/boardgames.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/boardgames.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Conservative.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Conservative.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Conservative.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/hockey.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/hockey.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/hockey.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Libertarian.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Libertarian.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/Libertarian.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/neoliberal.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/neoliberal.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/neoliberal.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/politics.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/politics.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/politics.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
x=0
with open("../../Reddit_Data_trimmed/September_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/worldnews.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/August_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/worldnews.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
with open("../../Reddit_Data_trimmed/June_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x <= 50000:
            if len(line_in['body']) >=150:
                with open('../../json_files/worldnews.txt', 'a') as outfile:
                    outfile.write(line)
                    if(x%1000==0):
                         print(x)
                    x=x+1
```

    0
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
    21000
    22000
    23000
    24000
    25000
    26000
    27000
    28000
    29000
    30000
    31000
    32000
    33000
    34000
    35000
    36000
    37000
    38000
    39000
    40000
    41000
    42000
    43000
    44000
    45000
    46000
    47000
    48000
    49000
    50000
    


```python
d = {'parent_id':[], 
     'author':[], 
     'distinguished':[], 
     'body':[], 
     'gilded':[], 
     'score':[], 
     'author_flair_css_class':[],
     'stickied':[],
     'retrieved_on':[],
     'author_flair_text':[],
     'id':[]
    }

Android_df = pd.DataFrame(data=d)

#Android_df = pd.read_json('../../json_files/Android.txt', orient='index')
x=0
with open('../../json_files/Android.txt', 'r') as infile:
    for line in infile:
        if(x<=49999):
            line_in = json.loads(line)
            Android_df.loc[x, 'parent_id'] = line_in['parent_id']
            Android_df.loc[x, 'author'] = line_in['author']
            Android_df.loc[x, 'distinguished'] = line_in['distinguished']
            Android_df.loc[x, 'body'] = line_in['body']
            Android_df.loc[x, 'gilded'] = line_in['gilded']
            Android_df.loc[x, 'score'] = line_in['score']
            Android_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
            Android_df.loc[x, 'stickied'] = line_in['stickied']
            Android_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
            Android_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
            Android_df.loc[x, 'id'] = line_in['id']
        x=x+1
```


```python
Android_df.head
```




    <bound method NDFrame.head of                      author author_flair_css_class  \
    0         Worth_The_Squeeze                    NaN   
    1             startupreddit              userBlack   
    2                   Deksloc                    AMA   
    3         _ALLLLRIGHTY_THEN                   None   
    4                  the_scam                   None   
    5                   ragnore                   None   
    6            ProfessorWeeto                   None   
    7                  Satukoro              userBlack   
    8                 thechaosz                   None   
    9                   Imp0924               userGray   
    10           notveryenglish               userGray   
    11       van_goghs_pet_bear                   None   
    12                 the_scam                   None   
    13          innervisions710                   None   
    14           blue-sunrising                   None   
    15                  bliblio                   None   
    16                  dzrtguy                   None   
    17               rushingkar              userBlack   
    18               asdfman123                   None   
    19       van_goghs_pet_bear                   None   
    20            startupreddit              userBlack   
    21            formerfatboys                   None   
    22                  tb03102                   None   
    23               xSagittiSx                   None   
    24                    oZiix              userBlack   
    25                  Wesside                   None   
    26       imUGLYandimPROOUUD                   None   
    27         WesternAddiction                   None   
    28             Bobby-Botato                   None   
    29                    nlofe               userTeal   
    ...                     ...                    ...   
    49970               V4ynard                   None   
    49971           regis_regis                   None   
    49972          agracadabara                   None   
    49973                 iskin                   None   
    49974                   jt4                userRed   
    49975                  BFCE               userGray   
    49976               Skripka               userBlue   
    49977   LifeLikeAndPoseable                   None   
    49978          benjimaestro                userRed   
    49979          agracadabara                   None   
    49980               Eridrus                   None   
    49981                bupku5                   None   
    49982  moto-throwaway-98252                   None   
    49983                  Rjwu                   None   
    49984              aecarol1                   None   
    49985           hellswaters              userBlack   
    49986       Didactic_Tomato              userBlack   
    49987               zeldar_                   None   
    49988               p4block                   None   
    49989              wilso850                   None   
    49990        anal_astronaut               userLime   
    49991   Crocoduck_The_Great                   None   
    49992               crescal              userBlack   
    49993                bupku5                   None   
    49994                mmapza               userBlue   
    49995              logoster               userGray   
    49996         avataraccount                   None   
    49997   LifeLikeAndPoseable                   None   
    49998       adithyathefreak                   None   
    49999                  jcpb               userTeal   
    
                                           author_flair_text  \
    0                                                    NaN   
    1                                                  Black   
    2                             /r/Android AMA Coordinator   
    3                                                   None   
    4                                                   None   
    5                                                   None   
    6                                                   None   
    7                                      Optimus Elite, GB   
    8                                                   None   
    9                                      Nexus 6P | 7.1.1    
    10                                           S4, S7 Edge   
    11                                                  None   
    12                                                  None   
    13                                                  None   
    14                                                  None   
    15                                                  None   
    16                                                  None   
    17                VZW Moto X (2014), 6.x CM | LG G Watch   
    18                                                  None   
    19                                                  None   
    20                                                 Black   
    21                                                  None   
    22                                                  None   
    23                                                  None   
    24                           Galaxy S7e/Nvidia Shield TV   
    25                                                  None   
    26                                                  None   
    27                                                  None   
    28                                                  None   
    29                                        T-Mobile LG G6   
    ...                                                  ...   
    49970                                               None   
    49971                                               None   
    49972                                               None   
    49973                                               None   
    49974                           Pixel 2 is disappointing   
    49975  HTC M8 -&gt; LG V10 that bootloops (fuck lg) -...   
    49976                                      LG v20 Sprint   
    49977                                               None   
    49978              Redmi Note 3 Pro | Ressurection Remix   
    49979                                               None   
    49980                                               None   
    49981                                               None   
    49982                                               None   
    49983                                               None   
    49984                                               None   
    49985                                            Nexus 5   
    49986                                        Quite Black   
    49987                                               None   
    49988                                               None   
    49989                                               None   
    49990                  Nexus4-16; Nexus 5 Black/White 32   
    49991                                               None   
    49992                                              Black   
    49993                                               None   
    49994                                               Blue   
    49995                                    ATT Samsung GS7   
    49996                                               None   
    49997                                               None   
    49998                                               None   
    49999             iPhone 6S+ | iPhone X after 2017.10.27   
    
                                                        body distinguished  \
    0      I know. It's the thing I call the "Apple spher...           NaN   
    1      I just really hope they keep the quality as it...           NaN   
    2      Sorry HollowmanNapkin, your submission has bee...     moderator   
    3      As someone who hasnt used a headphone jack in ...          None   
    4      I feel that they tend to submit patents for de...          None   
    5      1. For tablets, sure. For regular phones? Very...          None   
    6      I've never once ran out of juice using wireles...          None   
    7      The benchmark should be quality. LG has shown ...          None   
    8      Question. What is to gain from removing it? Ho...          None   
    9      Yeah but when there are only a few companies c...          None   
    10     I always carry a pen and tiny booklet in my po...          None   
    11     maybe for you, not for most people. wireless h...          None   
    12     This was November and then again in June. Same...          None   
    13     But what about Bluetooth headphone users? Many...          None   
    14     I really don't understand why people hate blue...          None   
    15     &gt;i disagree with that. same audio quality, ...          None   
    16     89% upvoted? It's insanity anyone would want t...          None   
    17     The main problem I have with the dongle is tha...          None   
    18     Maybe my thinking isn't laggardly, I just don'...          None   
    19     apple copies people all the time, obviously. b...          None   
    20     Oh boy, idk about anyone who would want to buy...          None   
    21     The biggest thing is that there are no headpho...          None   
    22     Man I love my LG tones. I'm on my 3rd pair. Us...          None   
    23     The 90% referred to having black bars while ga...          None   
    24     Agreed!  The same puff pieces pop on Apples re...          None   
    25     Except for the fact that it's still a screen t...          None   
    26     Anyone know anything about the "better sound s...          None   
    27     Bluetooth headphones are great until they die....          None   
    28     HTC makes some of the best designed phones: Th...          None   
    29     I'll never understand why people would use FLA...          None   
    ...                                                  ...           ...   
    49970  Looking for phones at the same price/performan...          None   
    49971  You can enable radio. The signal will be weake...          None   
    49972  &gt;Just search up "iPhone + battery + worse" ...          None   
    49973  I would need to see good proof of this claim. ...          None   
    49974  I've mentioned this before, but I have yet to ...          None   
    49975  * Speed\n\n* Charging\n\n* Battery life\n\n* S...          None   
    49976  Interestingly, in May 1911 the Harland and Wol...          None   
    49977  That's the mistake most consumers make: purcha...          None   
    49978  Wtf? Samsung are still copying apple after all...          None   
    49979  All that became irrelevant with Geekbench 4. L...          None   
    49980  Hmmm, can you link me to some instructions on ...          None   
    49981  and the Pixel C was another classic case of Go...          None   
    49982  Thanks for the input. I'm 99% sure I am going ...          None   
    49983  I have no idea what you're smoking, but iPhone...          None   
    49984  The 3DMark test is for the GPU, it’s not testi...          None   
    49985  As a Canadian in a province with the best cell...          None   
    49986  You should get your phone checked out. I just ...          None   
    49987  Cell service and data speeds can be very spott...          None   
    49988  Settings -&gt; Display -&gt; LiveDisplay \n\nI...          None   
    49989  I agree. As well as a clear all button for the...          None   
    49990  I pay 100 bucks for 2 lines and 50gb of data p...          None   
    49991  I know that list is complete as I sit here and...          None   
    49992  Samsung is probably doing better in US because...          None   
    49993  the Pixel2 will bellyflop hard but then Sundar...          None   
    49994  The best feature I miss from an old phone is F...          None   
    49995  Not me, I'll use fm over any streaming service...          None   
    49996  This article has design patent and renders. \n...          None   
    49997  Because I don't want Google and third party de...          None   
    49998  I agree.\n\nWhat makes Android different from ...          None   
    49999  Skip the link, it's a fucking generic Wordpres...          None   
    
           gilded       id   parent_id  retrieved_on  score stickied  
    0         0.0  dmehp79  t1_dme9izp  1.504557e+09    6.0    False  
    1         0.0  dmehp7m  t1_dmehj24  1.504557e+09    1.0    False  
    2         0.0  dmehpbd   t3_6xatxb  1.504557e+09    1.0    False  
    3         0.0  dmehpd3  t1_dme0181  1.504557e+09    2.0    False  
    4         0.0  dmehphy  t1_dmefgm7  1.504557e+09    1.0    False  
    5         0.0  dmehpye  t1_dmefsk0  1.504557e+09    3.0    False  
    6         0.0  dmehqjq  t1_dmehndm  1.504557e+09    0.0    False  
    7         0.0  dmehqvv  t1_dmehkqe  1.504557e+09    3.0    False  
    8         0.0  dmehrtr   t3_6x6o68  1.504557e+09    1.0    False  
    9         0.0  dmehtn0  t1_dmdj7va  1.504557e+09    1.0    False  
    10        0.0  dmehu2d   t3_6x676i  1.504557e+09    1.0    False  
    11        0.0  dmehvty  t1_dmeay68  1.504557e+09   -2.0    False  
    12        0.0  dmehwha  t1_dmegfzn  1.504557e+09    1.0    False  
    13        0.0  dmehwqv  t1_dmdozbz  1.504557e+09    1.0    False  
    14        0.0  dmehx3b  t1_dmeh0i3  1.504557e+09    4.0    False  
    15        0.0  dmehx7n  t1_dme8zyh  1.504557e+09    1.0    False  
    16        0.0  dmehx90   t3_6x6o68  1.504557e+09    2.0    False  
    17        0.0  dmehxyj  t1_dme8l01  1.504557e+09    1.0    False  
    18        0.0  dmehy1x  t1_dmegoa1  1.504557e+09    3.0    False  
    19        0.0  dmehy2p  t1_dme9vbg  1.504557e+09    0.0    False  
    20        0.0  dmehzka  t1_dmehtld  1.504557e+09   26.0    False  
    21        0.0  dmei18o   t3_6x6o68  1.504557e+09    2.0    False  
    22        0.0  dmei1ir  t1_dmdq3wa  1.504557e+09    1.0    False  
    23        0.0  dmei21e  t1_dmeg229  1.504557e+09    1.0    False  
    24        0.0  dmei2od  t1_dme5yxk  1.504557e+09    3.0    False  
    25        0.0  dmei3ln  t1_dmehsja  1.504557e+09   32.0    False  
    26        0.0  dmei5fz   t3_6x6o68  1.504557e+09    1.0    False  
    27        0.0  dmei5is  t1_dme8ykt  1.504557e+09    2.0    False  
    28        0.0  dmei5ok   t3_6x8phs  1.504557e+09    4.0    False  
    29        0.0  dmei5oz  t1_dme1i92  1.504557e+09   39.0    False  
    ...       ...      ...         ...           ...    ...      ...  
    49970     0.0  dnfykmb   t3_71iegx  1.507057e+09    1.0    False  
    49971     0.0  dnfykym  t1_dnfh703  1.507057e+09    3.0    False  
    49972     0.0  dnfymon  t1_dnfwlob  1.507057e+09   11.0    False  
    49973     0.0  dnfynfa  t1_dnfs4vf  1.507057e+09    4.0    False  
    49974     0.0  dnfyqa4   t3_724h9a  1.507057e+09   10.0    False  
    49975     0.0  dnfyqi6   t3_723wow  1.507057e+09    3.0    False  
    49976     0.0  dnfyqlz   t3_725wdx  1.507057e+09    5.0    False  
    49977     0.0  dnfys3d  t1_dnfn2zn  1.507057e+09    0.0    False  
    49978     0.0  dnfytra   t3_725809  1.507057e+09   95.0    False  
    49979     0.0  dnfytxu  t1_dnfv064  1.507057e+09   19.0    False  
    49980     0.0  dnfyvrm  t1_dnfrx0k  1.507057e+09    1.0    False  
    49981     0.0  dnfyw62  t1_dnfnz9t  1.507057e+09    3.0    False  
    49982     0.0  dnfywwj  t1_dnftanz  1.507057e+09    2.0    False  
    49983     0.0  dnfyxkm  t1_dnfxysv  1.507057e+09   12.0    False  
    49984     0.0  dnfyyk4  t1_dnfy1ja  1.507057e+09   12.0    False  
    49985     0.0  dnfyzf0  t1_dnftt8l  1.507058e+09    1.0    False  
    49986     0.0  dnfyzk9  t1_dnek8v8  1.507058e+09    1.0    False  
    49987     0.0  dnfyzoq  t1_dnfpibg  1.507058e+09    1.0    False  
    49988     0.0  dnfz007  t1_dnfyvrm  1.507058e+09    1.0    False  
    49989     0.0  dnfz19r  t1_dnfyfqd  1.507058e+09    1.0    False  
    49990     0.0  dnfz1lb  t1_dnfeib5  1.507058e+09    1.0    False  
    49991     0.0  dnfz1w0   t3_71zd54  1.507058e+09    1.0    False  
    49992     0.0  dnfz4u8  t1_dnfq7hr  1.507058e+09    8.0    False  
    49993     0.0  dnfz520  t1_dnfsgo3  1.507058e+09    9.0    False  
    49994     0.0  dnfz59f   t3_722jiv  1.507058e+09   13.0    False  
    49995     0.0  dnfz5ip  t1_dnfrssm  1.507058e+09    1.0    False  
    49996     0.0  dnfz77v  t1_dnfxbp3  1.507058e+09   88.0    False  
    49997     0.0  dnfz8bn  t1_dnfmnr0  1.507058e+09    1.0    False  
    49998     0.0  dnfz8h6  t1_dnfyed5  1.507058e+09   31.0    False  
    49999     0.0  dnfz95b   t3_725wdx  1.507058e+09    2.0    False  
    
    [50000 rows x 11 columns]>




```python
d = {'parent_id':[], 
     'author':[], 
     'distinguished':[], 
     'body':[], 
     'gilded':[], 
     'score':[], 
     'author_flair_css_class':[],
     'stickied':[],
     'retrieved_on':[],
     'author_flair_text':[],
     'id':[]
    }

x = 0

temp_df = pd.DataFrame(data=d)

t = '../../csv_files'

with open("../../Reddit_Data_trimmed/September_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/Conservative.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
    
temp_df.to_csv(t + '/Conservative.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/Libertarian.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/Libertarian.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/neoliberal.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break
    
temp_df.to_csv(t + '/neoliberal.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/politics.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/politics.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/worldnews.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/worldnews.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/Android.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/Android.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/boardgames.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/boardgames.csv', sep='\t')
```


```python
x = 0

with open("../../Reddit_Data_trimmed/September_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
            
with open("../../Reddit_Data_trimmed/August_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
                x = x +1
        else:
            break
        
with open("../../Reddit_Data_trimmed/June_17/hockey.txt", 'r') as file_in:
    for line in file_in:
        line_in = json.loads(line)
        if x < 50000:
            if len(line_in['body']) >= 150:
                temp_df.loc[x, 'parent_id'] = line_in['parent_id']
                temp_df.loc[x, 'author'] = line_in['author']
                temp_df.loc[x, 'distinguished'] = line_in['distinguished']
                temp_df.loc[x, 'body'] = line_in['body']
                temp_df.loc[x, 'gilded'] = line_in['gilded']
                temp_df.loc[x, 'score'] = line_in['score']
                temp_df.loc[x, 'author_flair_css_class'] = line_in['author_flair_css_class']
                temp_df.loc[x, 'stickied'] = line_in['stickied']
                temp_df.loc[x, 'retrieved_on'] = line_in['retrieved_on']
                temp_df.loc[x, 'author_flair_text'] = line_in['author_flair_text']
                temp_df.loc[x, 'id'] = line_in['id']
        else:
            break

temp_df.to_csv(t + '/hockey.csv', sep='\t')
```
