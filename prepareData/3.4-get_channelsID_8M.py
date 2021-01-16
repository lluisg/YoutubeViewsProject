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
KEY = 4
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

def get_youtube_channelId(video_id):
    print('Searching video: ', video_id)

    video_data = youtube.videos().list(part="snippet", id=video_id).execute()

    if video_data['pageInfo']['totalResults'] == 0:
        with open('../DATA/list_8M_channelsId_notF.txt', 'a') as f:
            f.write(video_id+'\n')
            print('-- video not found')
            return None

    with open('../DATA/list_8M_channelsId.txt', 'a') as f:
        f.write(video_id+'\n')

    channel_id = video_data['items'][0]['snippet']['channelId']
    return channel_id

def check_channel_found(name):
    with open('../DATA/list_8M_channelsId.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]

    if name in list_channels:
        return True
    else:
        return False


def get_next_channel_index():
    with open('../DATA/list_8M_channelsId.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]
    with open('../DATA/list_8M_channelsId_notF.txt', 'r') as f:
        content_not_f = f.readlines()
    list_notfound_channels = [line.rstrip('\n') for line in content_not_f]

    if len(list_channels) > 0 or len(list_notfound_channels) > 0:
        df_movies = pd.read_csv('../DATA/videosID8M_final.csv', encoding = "ISO-8859-1")
        found = False

        for index, row in df_movies.iterrows():
            chan = row['videosId']
            if chan not in list_channels and chan not in list_notfound_channels:
                df_read = df_movies.loc[df_movies['videosId'] == chan]
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
    df_movies = pd.read_csv('../DATA/videosID8M_final.csv', encoding = "utf-8")

    # Creating pandas data frame appending to the previously obtained data
    if os.path.isfile('../DATA/ChannelID8M.csv'):
        df_previous = pd.read_csv('DATA/ChannelID8M.csv', encoding = "utf-8")
        list_channelID = df_previous['channelId'].tolist()
    else:
        print('Creating ChannelsID8M file from 0')
        list_channelID = []

    # Ask for the videos info for each channel
    while index != -1:
        name = df_movies.iloc[index].values[0]
        if not check_channel_found(name):
            channelId = get_youtube_channelId(name)

            if channelId is not None:
                list_channelID.append(channelId)

                # Save into csv format in the desired location
                df_previous = pd.DataFrame(list_channelID, columns=['channelId'])
                df_previous.to_csv('../DATA/ChannelID8M.csv', encoding='utf-8', index=True)
            index = get_next_channel_index()

    print("you reached the last element from the videosID8M_final file")
