### Project Plan Reddit Comment Classification

## Overview
The attempt of this project is to analyze comments gathered from reddit.  The comments are seperated by subreddit categories.  
I will choose subreddits that have different political biases and will see if any difference in comments can be used to create 
an algorithm capable of classfying future comments into the correct subreddit category.

## Data
The data is collected in a corpus, seperated by month.  The corpus begins on December 2005 and currently ends August 2017.
The last month is the largest file, containing 84,658,503 comments.  The comments are formatted in JSON (JavaScript Object Notation).
Each comment comes with metadata about the subreddit it was posted in and date posted (as well as other data).  

## Project Plan
I will analyze the comments by different metrics such as average comment length, average sentence length, average word length.  
Using this data, I will attempt to build an algorithm that is able to classify future comments.  Later, I would like to use machine 
learning algorithms to do the same.