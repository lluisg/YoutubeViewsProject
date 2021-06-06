Data channels names from
https://www.kaggle.com/datasnaek/youtube-new?select=GBvideos.csv
https://data.world/sevenup13/youtube-video-and-channel-metadata
https://research.google.com/youtube8m/index.html

YOTUBE DATA_API info
https://developers.google.com/youtube/v3/docs/videos/list?apix_params=%7B%22part%22%3A%5B%22snippet%22%2C%22contentDetails%22%2C%22statistics%22%5D%2C%22id%22%3A%5B%22j2McbpSJDew%22%5D%7D#try-it

Data API quickstart
https://developers.google.com/youtube/v3/quickstart/python

CONSOLE CLOUD GOOGLE -> keys
https://console.cloud.google.com/apis/credentials?project=youtubetitlevalorator

When preparing the ids, run 3.4 with different keys (normally 4) until when you want to stop
then 3.41 to merge them into the _final files, and if everything is correct
copy this _final.txt file into the .txt file, which is the one really used on
the scripts to check already found ids (the .txt works as a savestate)
Finally run 3.42 and you can run again 3.4
