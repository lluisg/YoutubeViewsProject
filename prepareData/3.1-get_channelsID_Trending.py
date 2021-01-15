#importing libraries
import csv
import os
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import json
# API Key generated from the Youtube API console
from dotenv import load_dotenv
load_dotenv()
import os
KEY = 1
print('Using key', KEY)
if KEY == 1:
    api_key = os.getenv("key1")
elif KEY == 2:
    api_key = os.getenv("key2")
elif KEY == 3:
    api_key = os.getenv("key3")
elif KEY == 4:
    api_key = os.getenv("key4")

# Establishing connection with the YouTube API key
youtube  = build('youtube','v3',developerKey=api_key)

def get_youtube_channelId(name):
    print('Searching channel: ', name)
    # Using the API's list function to retrive the channel data
    channel_search_result = youtube.search().list(part="id", maxResults=1, q=name).execute()
    # print(channel_search_result)
    if channel_search_result['pageInfo']['totalResults'] == 0:
        with open('DATA/list_trending_channelsId_notF.txt', 'a') as f:
            f.write(name+'\n')
        print('--Channel search without result')
        return None
    elif not channel_search_result['items']:
        with open('DATA/list_trending_channelsId_notF.txt', 'a') as f:
            f.write(name+'\n')
        print('--Not results in list')
        return None
    elif channel_search_result['items'][0]['id']['kind'] != "youtube#channel":
        with open('DATA/list_trending_channelsId_notF.txt', 'a') as f:
            f.write(name+'\n')
        print('--Result not channel type')
        return None

    else:
        channel_id = channel_search_result['items'][0]['id']['channelId']

    with open('DATA/list_trending_channelsId.txt', 'a') as f:
        f.write(name+'\n')

    return channel_id

def check_channel_found(name):
    with open('DATA/list_trending_channelsId.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]

    if name in list_channels:
        return True
    else:
        return False

def get_next_channel_index():
    with open('DATA/list_trending_channelsId.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]
    with open('DATA/list_trending_channelsId_notF.txt', 'r') as f:
        content_not_f = f.readlines()
    list_notf_channels = [line.rstrip('\n') for line in content_not_f]

    if len(list_channels) > 0 or len(list_notf_channels) > 0:
        df_movies = pd.read_csv('DATA/AllTrendingChannels.csv', encoding = "ISO-8859-1")
        found = False

        for index, row in df_movies.iterrows():
            chan = row['channel_title']
            if chan not in list_channels and chan not in list_notf_channels:
                df_read = df_movies.loc[df_movies['channel_title'] == chan]
                index = df_read.index[0]
                found = True
                break
        if not found:
            return -1

    else:
        index = 0
    return index

if __name__ == "__main__":
    # Get the index of the last element we obtained
    index = get_next_channel_index()
    print('Starting from: ', index)

    # Retrieve the channels data from the DDBB
    df_movies = pd.read_csv('DATA/AllTrendingChannels.csv', encoding = "utf-8")

    # Creating pandas data frame appending to the previously obtained data
    if os.path.isfile('DATA/ChannelIDTrending.csv'):
        df_previous = pd.read_csv('DATA/ChannelIDTrending.csv', encoding = "utf-8")
        list_previous = df_previous['channelId'].tolist()
    else:
        print('Creating ChannelIDTrending file from 0')
        list_previous = None

    # Ask for the videos info for each channel
    while index != -1:
        name = df_movies.iloc[index].values[1]
        if not check_channel_found(name):
            channelId = get_youtube_channelId(name)
            # channelId_dict = {'index':index, 'channelId':channelId}
            if channelId is not None:
                if list_previous is None:
                    list_previous = channelId
                else:
                    list_previous.append(channelId)

                df_previous = pd.DataFrame(list_previous,columns=['channelId'])

                # Save into csv format in the desired location
                df_previous.to_csv('DATA/ChannelIDTrending.csv', encoding='utf-8', index=True)
            index = get_next_channel_index()

    print("you reached the last element from the AllTrendingChannels file")
