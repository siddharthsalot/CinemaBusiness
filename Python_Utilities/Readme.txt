1) Install libraries BeautifulSoup, configparser and nltk
2) Download twitter sentiment data. Merge all csvs using Merge-CSV.com.
3) Open the merged csv and delete the first three lines. It is the header added by the website Merge-CSV.com. RENAME the file and set it to movie name.
4) Dump the file into a folder. Note the location of this folder. It will be used in steps below.
5) Copy and Paste script and config file to the same directory where you have your Input csv files.
6) Add miscellaneous meta data as follows:
[DarkKnightRises] 	<= Must match with name of Input csv file
Star_Score = 79		<= Ask Sid for star score
Theatres = 4404		<= Get it online
Opening_Collection = 160.89		<= Get it online
EXTENDED_LIST = dark			<= Comma separated list of words that you don't want to include while calculating sentiment
Rank = 1						<= Rank of the movie when it was released
Rating = 8.4					<= IMDB rating of movie
7) Execute script using command: python Sentiment_Analyzer.py

####################################################################################################################################################
In order to view the output, navigate to the path configured in OUTPUT_PATH configuration in config.properties file.