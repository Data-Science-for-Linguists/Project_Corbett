### Progress Report Reddit Comment Classification

10/31/2017
	- modified the extraction code to transform the json entries into a dataframe and save them as a csv file
	- examined the length of each file to see which subreddits have enough posts for me to work with
	- chose 9 subreddits to work with, number of total posts in June, August and September of this year
		- "Conservative" 	169,622 posts
		- "Libertarian" 	280,908 posts
		- "neoliberal"		557,280 posts
		- "politics"		5,033,408 posts
		- "worldnews"		2,428,736 posts
		- "Android"			434,356 posts
		- "boardgames"		184,762 posts
		- "food"			205,690 posts
	- Next, I will extract 100,000 random posts from each subreddit of atleast a certain length.

10/12/2017
	- downloaded one year worth of reddit data to begin with
	- created a list of subreddit threads (almost 10,000 threads)
	- wrote a program to iterate through all posts and extract and save to file the posts from designted subreddits
	- created a small sample data (2000 entries) set to demonstrate extraction
	- since the program takes several hours to run, I will run program over night for the next week or so
	- files are saved preserving the json format of the data

10/02/2017
	- began downloading JSON files (files are large so it may take time to download and extract)
	- began loading files into python and studying format
	- wrote a script that can look for instances of strings in the JSON objects
	

