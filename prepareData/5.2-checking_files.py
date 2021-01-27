import csv
import os
import pandas as pd

"""
comprovar que els fitxers descarregats hi siguin tots
mirar que el list_channels 3 + list channels not 3 sumi el mateix que els totals en final ID3
mirar que tots els list channels3 estiguin dins de final videos data3
mirar que no hi hagi cap list chanel not3 dins de final video data 3

"""

# with open('DATA/list_channels.txt', 'r') as f:
#     content_i = f.readlines()
# list_read_channels = [line.rstrip('\n') for line in content_i]
# with open('DATA/list_channels_not.txt', 'r') as f:
#     content_not_i = f.readlines()
# list_notread_channels = [line.rstrip('\n') for line in content_not_i]
# list_read_channels.extend(list_notread_channels)
# print(len(list_read_channels))
#
# df = pd.read_csv('DATA/Final_ChannelID.csv', encoding = "ISO-8859-1")
# list_df = df['channelId'].tolist()
# set_list = set(list_df)
# list_df = list(set_list)
# print(len(list_df))


base_file = 'Final_VideosData'
path = '../DATA/videosinfo/'

def read_final_file(key):
    print('Reading ', base_file, key)
    df = pd.read_csv(path+base_file+str(key)+'.csv')

    leng = len(df.index)
    print('length'+str(key)+':', leng)
    return df, leng

def read_extra_files(key):
    print('Reading ', base_file, key)
    with open(path+'/list_channels'+str(key)+'.txt', 'r') as f:
        content_i = f.readlines()
    list_channels = [line.rstrip('\n') for line in content_i]
    with open(path+'/list_channels_not'+str(key)+'.txt', 'r') as f:
        content_not = f.readlines()
    list_channels_not = [line.rstrip('\n') for line in content_not]

    print(len(list_channels), len(list_channels_not))


df1, length1 = read_final_file(1)
print(length1)
listc, listnot = read_extra_files(1)
