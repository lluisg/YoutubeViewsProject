import os
import glob
import pandas as pd
import numpy as np

def read_file(filename, check_r):
    print('Reading', filename)
    df = pd.read_csv(filename, encoding = "utf-8", lineterminator='\n')
    if 'commentCount\r' in df.columns and check_r:
        print('removed commentCount with r')
        df['commentCount\r'] = df['commentCount\r'].str[:-1]
        df['commentCount\r'] = pd.to_numeric(df['commentCount\r'])

        df['commentCount'].update(df.pop('commentCount\r'))
    # print(df.head())
    return df

if __name__ == "__main__":
    lite = False
    file = '../DATA/videosinfo/Final_VideosData_merged.csv'
    # file = 'DATA/Final_VideosData.csv'

    if not lite:
        df = read_file(file, False)

        #clean repeated channels
        list_channels = []
        rows_repeated = []
        rows_unique = []
        previous = None
        for index, row in df.iterrows():
            if row['channelId'] != previous:
                if row['channelId'] not in list_channels:
                    rows_unique.append(index)
                    previous = row['channelId']
                list_channels.append(row['channelId'])
            else:
                rows_unique.append(index)

        set_channel = set(list_channels)
        list_channels_unique = list(set_channel)
        print('We have {} videos from {} different channels, from which {} are unique'.format(df.shape[0], len(list_channels), len(list_channels_unique)))

    df_copy = read_file(file, True)
    print('copied')
    if not lite:
        df_copy = df_copy.iloc[rows_unique]

        list_channels_after = []
        previous = None
        for index, row in df_copy.iterrows():
            if row['channelId'] != previous:
                list_channels_after.append(row['channelId'])
                previous = row['channelId']
        set_channel_after = set(list_channels_after)
        list_channels_after_unique = list(set_channel_after)
        print('After the cleaning we have {} videos from {} different channels, from which {} are unique'.format(df_copy.shape[0], len(list_channels_after), len(list_channels_after_unique)))


        df_copy.to_csv('../DATA/Final_videosDataClean.csv', encoding='utf-8', index=False)
