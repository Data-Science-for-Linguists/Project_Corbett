# Reddit Comment Classification
### Robert Corbett
### rwc27@pitt.edu
### last updated 12/15/2017

## Overview
The attempt of this project is to analyze comments gathered from reddit to find tokens that can be used to classify posts into their correct subreddit
and are also likely to be found in higher scoring posts.  The comments are seperated by subreddit categories.  
I choose subreddits that have different political biases (Conservative, Libertarian and neoliberal) and will see if any difference in comments can be used to create 
an algorithm capable of classfying future comments into the correct subreddit category. I also choose two non-biased subreddits that are likely to 
contain a lot of posts of a political nature (worldnews, politics).  Finally, I selected three control subreddits that should not contain any political 
discussion (boardgames, Android and hockey).

The data was too large to upload to github so a link to it is provided below.

I downloaded all posts from the months of June, August and September of 2017.  I skipped July because I had some difficulty with it and I did not think that 
it would effect the project if I skipped it.  The data came as JSON files, each about 45GB large.  Each post entry had information for parent_id, author, 
distinguished, body, gilded score, author_flair_css_class, stickied, retrieved_on, author_author flair_text, id, and subreddit.  I extracted all posts from 
the eight subreddits of interest while preserving all information as JSON format.  I then timmed that data by taking the first 50,000 posts in each subreddit post
that had a body of atleast 150 characters.

In the analysis, I loaded the posts into a dataframe, creating columns only for the body, subreddit and the score.  I ran the data through MultinomialNB model 
targeting the subreddits and then retreived the top 100 tokens by classifier coefficient so they are the highest weighted tokens.  I did the same targeting the 
scores.  I saved all of the dataframes as csv files so I can look at them later.  

Finally, I compared the lists to see which tokens are in each list.

Directory:
Corbett_part_1_data_extraction.ipynb - used to extract posts from main data files
Corbett_part_2_data_exploration.ipynb - used to extract 50,000 posts from each subreddit
Analysis.ipynb - Main Analysis

csv_files - csv files from Analysis

images - images for markdown files


Link to Visitors Log: https://github.com/Data-Science-for-Linguists/Shared-Repo/blob/master/todo10_visitors_log/visitors_log_rob.md
Link to Reddit Data:  http://files.pushshift.io/reddit/comments/