import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


files = ['ChannelID8M', 'channelID8M/ChannelID8M1', 'channelID8M/ChannelID8M2',
            'channelID8M/ChannelID8M3', 'channelID8M/ChannelID8M4']

list_channels = []
for f in files:
    print('Reading ', f)
    initial_length = len(list_channels)
    df_videos = pd.read_csv('../DATA/'+f+'.csv')
    df_out = df_videos[['channelId']]

    for index, row in df_out.iterrows():
        list_channels.append(row['channelId'])

    print('length: ', len(list_channels)-initial_length)


list_channels = list(filter(None, list_channels))
print('\nFinal length: ', len(list_channels))
set_channels = set(list_channels)
list_channels_nodupli = list(set_channels)
print('without duplicates: ', len(list_channels_nodupli))

with open('../DATA/channelID8M/list_8M_channelsId_final.txt', 'w') as l:
    for el in list_channels_nodupli:
        l.write(str(el))

df_channels = pd.DataFrame(list_channels_nodupli,columns=['channelId'])
df_channels.to_csv('../DATA/ChannelID8M_final.csv', encoding='utf-8', index=True)
print('last: 69117-1949-11325-14687-14574\n 11652-101412\n\n')

# merge incorrect channels too
full_list = []
print('Reading original')
with open('../DATA/channelID8M/list_8M_channelsId_notF.txt') as l:
    lines = l.readlines()
len_o = len(lines)
print('length_original: ', len(lines))

for i in range(1,5):
    print('Reading ', list_8M_channelsId_notF'+str(i)+'.txt')
    with open('../DATA/channelID8M/list_8M_channelsId_notF'+str(i)+'.txt') as l:
        lines = l.readlines()
    print('length: ', len(lines)-len_o)
    full_list += lines

print('\nFinal length: ', len(full_list))
full_list_set = set(full_list)
full_list_unique = list(full_list_set)
print('without duplicates: ', len(full_list_unique))

with open('../DATA/channelID8M/list_8M_channelsId_notF_final.txt', 'w') as l:
    for el in full_list_unique:
        l.write(str(el))
print('last: 30-155-234-216\n 4707-1653')
