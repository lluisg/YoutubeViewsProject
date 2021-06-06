#importing libraries
import csv
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import numpy as np
import json
# API Key generated from the Youtube API console
from dotenv import load_dotenv
load_dotenv()
import os
KEY = 4

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

import pause
from datetime import datetime
from datetime import timedelta
def wait_next_day():
    today = datetime.now()
    tomorrow = today.replace(day=today.day, hour=11, minute=0, second=0, microsecond=0) + timedelta(days=1)
    # date + datetime.timedelta(days=1)
    pause.until(tomorrow)


def get_youtube_channelId(video_id):
    # print('Searching video: ', video_id)

    video_data = youtube.videos().list(part="snippet", id=video_id).execute()

    if video_data['pageInfo']['totalResults'] == 0:
        with open('../DATA/channelID8M/list_8M_channelsId_notF'+str(KEY)+'.txt', 'a') as f:
            # print(video_id, type(video_id))
            f.write(str(video_id)+'\n')
            # print('-- video not found')
            return None

    with open('../DATA/channelID8M/list_8M_channelsId'+str(KEY)+'.txt', 'a') as f:
        f.write(video_id+'\n')
        # print('--video found')

    channel_id = video_data['items'][0]['snippet']['channelId']
    return channel_id

def check_channel_found(name):
    with open('../DATA/channelID8M/list_8M_channelsId'+str(KEY)+'.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]

    if name in list_channels:
        return True
    else:
        return False

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i if line]
    return list_channels


def get_next_channel_index(file, lists_channels):
    # df_movies = pd.read_csv('../DATA/channelID8M/videosID8M_final'+str(KEY)+'.csv', encoding = "ISO-8859-1")
    found = False

    for index, row in file.iterrows():
        chan = row['videosId']
        chann_already = [chan in x for x in lists_channels]
        if True not in chann_already:
            df_read = file.loc[file['videosId'] == chan]
            index = df_read.index[0]
            found = True
            break
    if not found:
        return -1

    return index


if __name__ == "__main__":

    # Retrieve the channels data from the DDBB
    df_movies = pd.read_csv('../DATA/channelID8M/videosID8M_final'+str(KEY)+'.csv', encoding = "utf-8")

    # Creating pandas data frame appending to the previously obtained data
    if os.path.isfile('../DATA/channelID8M/ChannelID8M'+str(KEY)+'.csv'):
        df_previous = pd.read_csv('../DATA/channelID8M/ChannelID8M'+str(KEY)+'.csv', encoding = "utf-8")
        list_channelID = df_previous['channelId'].tolist()
    else:
        print('Creating ChannelsID8M file from 0')
        list_channelID = []

    # Get the index of the last element we obtained
    ids_alreadyfound = read_txt_file('../DATA/channelID8M/list_8M_channelsId.txt')
    ids_alreadyincorrect = read_txt_file('../DATA/channelID8M/list_8M_channelsId_notF.txt')
    print('Previous ids obtained')
    index = get_next_channel_index(df_movies, [ids_alreadyfound, ids_alreadyincorrect])
    print('Starting from: ', index)


    number_days = 0
    initial_index = 0
    while number_days < 365:
        print('DAY '+str(number_days)+':', datetime.now())

        api_key = get_key(KEY)
        got_correct = 0
        got_wrong = 0

        try:
            # Ask for the videos info for each channel
            while index != -1:
                name = df_movies.iloc[index].values[0]
                # if not check_channel_found(name):
                channelId = get_youtube_channelId(name)

                if channelId is not None:
                    list_channelID.append(channelId)

                    # Save into csv format in the desired location
                    df_previous = pd.DataFrame(list_channelID, columns=['channelId'])
                    df_previous.to_csv('../DATA/channelID8M/ChannelID8M'+str(KEY)+'.csv', encoding='utf-8', index=True)
                    got_correct += 1
                else:
                    got_wrong += 1

                if (got_correct+got_wrong)%1000 == 0:
                    print(got_correct+got_wrong, 'channels got, ', got_correct, 'got correct')
                if initial_index+1000 < index:
                    initial_index = index
                    print('increased index by 1.000, now at:', initial_index)
                ids_found = read_txt_file('../DATA/channelID8M/list_8M_channelsId'+str(KEY)+'.txt')
                ids_incorrect = read_txt_file('../DATA/channelID8M/list_8M_channelsId_notF'+str(KEY)+'.txt')

                index = get_next_channel_index(df_movies, \
                    [ids_alreadyfound, ids_alreadyincorrect, ids_found, ids_incorrect])

        except HttpError:
            print('------  end of the usages for the key')
        except Exception as e:
            print('-- theres an error on code, exception given with text:')
            print(e)
            print('end usages for today')


        print('Done for today: '+str(got_correct)+' videos correctly, '+str(got_wrong)+' wrongly')
        print('ended at :', datetime.now())
        print('------------------------------------------------')
        number_days += 1
        if index == -1:
            print("you reached the last element from the file with key "+str(KEY))
            break

        wait_next_day()

    if number_days >= 365:
        print('A YEAR HAS PASSED')
    print('DONE!')
