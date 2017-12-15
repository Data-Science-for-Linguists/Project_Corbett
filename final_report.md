# Final Report Reddit Post Classifier
Robert Corbett
rwc27@pitt.edu
12/15/2017

## Overview:
The attempt of this project is to analyze comments gathered from reddit to find tokens that can be used to classify posts into 
their correct subreddit and are also likely to be found in higher scoring posts.  The comments are seperated by subreddit 
categories.  I choose subreddits that have different political biases (Conservative, Libertarian and neoliberal) and will see 
if any difference in comments can be used to create an algorithm capable of classifying future comments into the correct 
subreddit category. I also choose two non-biased subreddits that are likely to contain a lot of posts of a political nature 
(worldnews, politics).  Finally, I selected three control subreddits that should not contain any political discussion (
boardgames, Android and hockey).

## Data:
The data was too large to upload to github so a link to it is provided below.
http://files.pushshift.io/reddit/comments/

I downloaded all posts from the months of June, August and September of 2017.  I skipped July because I had some difficulty 
with it and I did not think that it would effect the project if I skipped it. The data came in JSON format, June was 41.0GB, 
August was 48.4GB and September 47.2GB large.  Each post entry had information for parent_id, author, distinguished, body, 
gilded score, author_flair_css_class, stickied, retrieved_on, author_author flair_text, id, and subreddit.  I extracted all 
posts from the eight subreddits of interest while preserving all information as JSON format.  I then timmed that data by taking 
the first 50,000 posts in each subreddit post that had a body of atleast 150 characters.

I used Corbett_part_1_data_extraction.ipynb to extract the all posts from the main files.  I then used 
Corbett_part_2_data_exploration,ipynb to trim the data down to the first 50,000 post of at least length 150 characters.  

## Analysis:
In the analysis, I loaded the posts into a dataframe, creating columns only for the body, subreddit and the score.  This 
process took the longest of any part of the code but I only needed to do it once and then I was able to used the same dataframe 
to create all of the models.  I ran the data through MultinomialNB model, using the bodies of the posts  target the subreddits.  
I used the nltk word tokenizer to tokenize the bodies.  The data was split 80% training and 20% testing. 

Then I retreived the top 100 tokens by classifier coefficient so they are the highest weighted tokens.  The script I used sorted 
the classifier by coefficient and then matched the classifier tokens with the vector so that the correct tokens were returned in 
the form of a dataframe. I ran the models on both unigrams and bigrams and saved the dataframs as csv files. 

The accuracy of these models were not awful by not great.  The accuracy scores were about where I expected them to be.  For the 
unigram model, the accuracy was about 69.5%.  I don't think this accuracy is too bad because the data I chose was intentionally 
chosen to have similar content.  When you look at the heatmap of the models confusion matrix, the control subreddits are easily 
classified by the model.  It is around the political subreddits that the model has difficulty, especially around Conservative and 
Libertarian.  This makes sense when you consider that these subreddits would contain a lot of the same subjects and tokens.  Also,
because the model has eight target to choose from, you would expect the accuracy to be lower.  The accuracy score for the bigram 
model was worse then the unigram model.  The bigram model ended up with a score of about 61%.  This might be because the model was 
given too much data.  When exploring the data early in the project, I found that the accuracy score for the unigram model peaked 
around 74% and as I added more data, the accuracy score would drop.  Since splitting the posts into bigrams creates more tokens than 
would splitting them into unigrams, the models had more data to build the classifier with.  In the future, I would like to explore 
what causes this more.  

Before I could analyze the scores, I had to find a way to seperate them.  I broke the main dataframe into 8 smaller dataframes, one 
for each subreddit.  The scores are spread over a large area.  The lowest score was -274 and the highest score was 17,193.  But the 
mean of all the scores was 8.97.  I wanted to split the dataframes into thirds.  First, I tried to use the percentiles to split them, 
but that caused problems so I ended up using a much simpler solution.  I sorted those dataframes by scores,  they were sorted from l
owest to highest.  I was then able to take the top third, middle third and bottom third.  I gave the bottom third a value of 1, middle 
scores a value of 2 and a high score the value of 3.  

I was then able to run the data through a multinomialNB model again using the bodies to taget the scores.  The accuracy scores for 
these models were very bad.  The accuracy scores for the unigrams were 41.3%, 40.7%, 40.4%, 40.9%, 44.0%, 46.0%, 41.5% and 40.2%.  
the accuracy scores for the bigrams were 42.2%, 40.9%, 39.9%, 40.6%, 45.7%, 44.6%, 42.9%, and 38.8%.  I don't know why the accuracy 
scores were so low.  The model only had three targets. 

After creating all of the models, I used functions to return the 100 highest weighted tokens for each model.  For the models, I used the function return_top100 and for the scores, I used return_top100_score.  The both take a vectorizer, a classifier and class labels.  The function sorts the classifier by coefficient value, then takes the last 100 elements.  These correlate to the highest weighted tokens.  I could easily change the number of elements to return.  It then matches the classifier to the vectorizer to match the tokens and return them.  The takens are returned in dataframes.

At the end of the analysis, I compared the lists to see which tokens were in the list of highly weighted tokens for both classifying the posts to subreddit and classifying the posts to the scores.  The results were not what I expected.  Many of the lists contained the same tokens.  For Example, comparing the Android unigram lists, the top words lists contains 90 words.  I  would like to look into why this happens in the future.  

The data I did gather was interesting.  A lot of the tokens are words and phrases you would expect for the subreddits.  The problem comes with the reuse of tokens in the score classifiers.  The same tokens are returned for high and low scoring posts.

## Future Goals:

Obviously, I want to find out why the models for the scored had terrible accuracy scores.  Also, I want to find out why the same tokens are being returned as being highly weighted for both high and low scoring posts

I had to write a lot of the code to be specific to the data I was using.  In the future, I would like to rewrite the functions to be used on more general data.  