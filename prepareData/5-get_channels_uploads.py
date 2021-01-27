#importing libraries
import csv
import os
import googleapiclient
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import numpy as np
import json
# API Key generated from the Youtube API console
from dotenv import load_dotenv
load_dotenv()
import os
KEY = 1

def get_key(KEY):
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
    global youtube
    youtube = build('youtube','v3',developerKey=api_key)
    return api_key
# api_key = get_key(KEY)

import pause
from datetime import datetime
from datetime import timedelta
def wait_next_day():
    today = datetime.now()
    tomorrow = today.replace(day=today.day, hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1)
    # date + datetime.timedelta(days=1)
    pause.until(tomorrow)


def youtube_playlist_data(channel_id, NUM_VIDEOS=20):
    # print('Searching channel: ', channel_id)

    channel_uploadlist = youtube.channels().list(id=channel_id, part='contentDetails, statistics').execute()
    # print(channel_uploadlist)
    if channel_uploadlist['pageInfo']['totalResults'] == 0:
        with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
            f.write(channel_id+'\n')
        # print('-- 0 videos in this channel')
        return None
    elif channel_uploadlist['items'][0]['statistics']['hiddenSubscriberCount'] == True:
        with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
            f.write(channel_id+'\n')
        # print('-- channel subscribers hidden')
        return None
    elif int(channel_uploadlist['items'][0]['statistics']['subscriberCount']) < 50000 :
        with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
            f.write(channel_id+'\n')
        # print('-- channel with less than 50.000 subs')
        return None

    # Retrieving the "uploads" playlist Id from the channel
    upload_playlist_id = channel_uploadlist['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    video_data = []
    # print('playlist', upload_playlist_id)

    try:
        playlist_videos = youtube.playlistItems().list(playlistId = upload_playlist_id, maxResults = NUM_VIDEOS, part = "id,snippet,contentDetails").execute()

        if len(playlist_videos['items']) < 10 :
            with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
                f.write(channel_id+'\n')
            # print('-- channel with less than 10 uploads')
            return None
        # print(len(playlist_videos['items']), 'videos in this channel')

        for element in playlist_videos['items']:
            videoId = element['snippet']['resourceId']['videoId']
            y_video_data = youtube.videos().list(id=videoId,part="id,snippet,contentDetails,statistics").execute()
            video_data = video_data + y_video_data['items']

        with open(path+'/list_channels'+str(KEY)+'.txt', 'a') as f:
            f.write(channel_id+'\n')

        return video_data

    except googleapiclient.errors.HttpError:
        with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
            f.write(channel_id+'\n')
        print('-- unidentified playlist error ocurred')
        return None

    # except Exception as e:
    #     print('Uknown Error ocurred:', e)
    #
    #     with open(path+'/list_channels_not'+str(KEY)+'.txt', 'a') as f:
    #         f.write(channel_id+'\n')
    #     print('-- an error ocurred')
    #     return None


def channel_videos_data(id_channel):
    # This function asks for the videos of the channel and returns only the variables we are interested
    videos=[]
    user_videos={}
    video_info={}

    y_video_data = youtube_playlist_data(id_channel)
    if y_video_data is None:
        return None
    else:
        # iterating through videos data one by one
        for data in y_video_data:
            video_info['channelId'] = data['snippet']['channelId']
            video_info['id'] = data['id']
            video_info['title'] = data['snippet']['title']
            video_info['publishedAt'] = data['snippet']['publishedAt']
            video_info['categoryId'] = data['snippet']['categoryId']
            video_info['channelTitle'] = data['snippet']['channelTitle']

            video_info['duration'] = data['contentDetails']['duration']

            if "viewCount" in data['statistics']:
                video_info['viewCount'] = data['statistics']['viewCount']
            else:
                video_info['viewCount'] = None
            if "likeCount" in data['statistics']:
                video_info['likeCount'] = data['statistics']['likeCount']
            else:
                video_info['likeCount'] = None
            if "dislikeCount" in data['statistics']:
                video_info['dislikeCount'] = data['statistics']['dislikeCount']
            else:
                video_info['dislikeCount'] = None
            if "commentCount" in data['statistics']:
                video_info['commentCount'] = data['statistics']['commentCount']
            else:
                video_info['commentCount'] = None
            videos.append(video_info.copy())

    return videos

def get_next_channel_index(file_df):
    with open(path+'/list_channels'+str(KEY)+'.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]
    with open(path+'/list_channels_not'+str(KEY)+'.txt', 'r') as f:
        content_not_i = f.readlines()
    list_notread_channels = [line.rstrip('\n') for line in content_not_i]

    if len(list_channels) > 0 or len(list_notread_channels) > 0:
        found = False
        for index, row in file_df.iterrows():
            chan = row['channelId']
            if chan not in list_channels and chan not in list_notread_channels:
                df_read = file_df.loc[file_df['channelId'] == chan]
                index = df_read.index[0]
                found = True
                break
        if not found:
            return -1

    else:
        index = 0
    return index

if __name__ == "__main__":
    global path
    path = '../DATA/videosinfo'
    file_name = path+'/Final_ChannelID'+str(KEY)+'.csv'
    # file_name = 'DATA/ChannelIDTrending.csv'
    output_name = path+'/Final_VideosData'+str(KEY)+'.csv'


    # Retrieve the channels data from the DDBB
    df = pd.read_csv(file_name, encoding = "utf-8")
    # Get the index of the last element we obtained
    index = get_next_channel_index(df)
    print(index)


    # Creating pandas data frame appending to the previously obtained data
    if os.path.isfile(output_name):
        # df_previous = pd.read_csv('DATA/Final_VideosData.csv', encoding = "utf-8")
        df_previous = pd.read_csv(output_name, encoding = "utf-8", lineterminator='\n')
    else:
        print('Creating Final_VideosData file from 0')
        df_previous = None

    # Ask for the videos info for each channel
    number_days = 0
    while number_days < 365:
        print('DAY '+str(number_days)+':', datetime.now())

        api_key = get_key(KEY)
        got_correct = 0
        got_wrong = 0

        try:
            while index != -1:
                video_id = df.iloc[index].values[1]
                videos_list = channel_videos_data(video_id)
                if videos_list is not None:
                    got_correct += 1
                    if df_previous is None:
                        df_previous = pd.DataFrame(videos_list)
                    else:
                        df_previous = pd.concat([df_previous, pd.DataFrame(videos_list)], ignore_index=True)
                    # Save into csv format in the desired location
                    df_previous.to_csv(output_name, encoding='utf-8', index=False)
                else:
                    got_wrong += 1

                index = get_next_channel_index(df)
        except HttpError:
            print('------  end of the usages for the key')
        except Exception as e:
            print('-- another error ends usages for today')
            # print(e)


        print('Done for today: '+str(got_correct)+' videos correctly, '+str(got_wrong)+' wrongly')
        print('ended at :', datetime.now())
        print('------------------------------------------------')
        number_days += 1
        if index == -1:
            print("you reached the last element from the "+file_name+" file")
            break

        wait_next_day()

    if number_days >= 365:
        print('A YEAR HAS PASSED')
    print('DONE!')
