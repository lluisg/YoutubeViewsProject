import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

files = ['ChannelID8M', 'channelID8M/ChannelID8M1', 'channelID8M/ChannelID8M2',
            'channelID8M/ChannelID8M3', 'channelID8M/ChannelID8M4']

list_channels = []
for f in files:
    print('Reading ', f)
    df_videos = pd.read_csv('../DATA/'+f+'.csv')

    for index, row in df_videos.iterrows():
        list_channels.append(row['channelId'])

    print('Incremental length: ', len(list_channels))

print('Reading previous version')
df_videos = pd.read_csv('../DATA/'+f+'.csv')
with open('../DATA/channelID8M/list_8M_channelsId.txt') as f:
    data = f.readlines()
data = [x.rstrip() for x in data]
data = [line for line in data if line] #ignore blank ones
list_channels.extend(data)
print('Length: ', len(data))

print('Final Complete length:', len(list_channels))
list_channels = list(filter(None, list_channels))
print('\nFinal length: ', len(list_channels))
set_channels = set(list_channels)
list_channels_nodupli = list(set_channels)
print('without duplicates: ', len(list_channels_nodupli))
print('last: 323982 - 155128\n\n')

with open('../DATA/channelID8M/list_8M_channelsId_final.txt', 'w') as l: #list of channels found into txt files
    for el in list_channels_nodupli:
        l.write(str(el)+'\n')

df_channels = pd.DataFrame(list_channels_nodupli,columns=['channelId'])
df_channels.to_csv('../DATA/ChannelID8M_final.csv', encoding='utf-8', index=True) #list of channels found into csv files0


# merge incorrect channels too -------------------------------------------------

full_list = []
for i in range(1,5):
    print('Reading ', 'list_8M_channelsId_notF'+str(i)+'.txt')
    with open('../DATA/channelID8M/list_8M_channelsId_notF'+str(i)+'.txt') as l:
        lines = l.readlines()
    lines = [x.rstrip() for x in lines]
    lines = [line for line in lines if line] #ignore blank ones
    print('Incremental length: ', len(lines))
    full_list.extend(lines)

print('Reading previous not F')
with open('../DATA/channelID8M/list_8M_channelsId_notF.txt') as l:
    lines = l.readlines()
lines = [x.rstrip() for x in lines]
lines = [line for line in lines if line] #ignore blank ones
print('Length: ', len(lines))
full_list.extend(lines)


print('\nFinal length: ', len(full_list))
full_list_set = set(full_list)
full_list_unique = list(full_list_set)
print('without duplicates: ', len(full_list_unique))
print('last: 5822 - 5819')

with open('../DATA/channelID8M/list_8M_channelsId_notF_final.txt', 'w') as l: #list of channels NOT found into txt files
    for el in full_list_unique:
        l.write(str(el)+'\n')
