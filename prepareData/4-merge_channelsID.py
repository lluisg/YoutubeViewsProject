import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

files = ['ChannelIDTrending', 'ChannelIDDataworld', 'ChannelID8M_final'']

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

df_channels = pd.DataFrame(list_channels_nodupli,columns=['channelId'])
df_channels.to_csv('../DATA/Final_ChannelID.csv', encoding='utf-8', index=True)
print('last: 4020-19906-69117\n13-755-719-713\n 93043-83833')


df_splitA, df_splitB = train_test_split(df_channels, test_size=0.5, random_state=42, shuffle=False)
df_split1, df_split2 = train_test_split(df_splitA, test_size=0.5, random_state=42, shuffle=False)
df_split3, df_split4 = train_test_split(df_splitB, test_size=0.5, random_state=42, shuffle=False)

df_split1.to_csv('../DATA/videosinfo/Final_ChannelID1.csv', encoding='utf-8', index=True)
df_split2.to_csv('../DATA/videosinfo/Final_ChannelID2.csv', encoding='utf-8', index=True)
df_split3.to_csv('../DATA/videosinfo/Final_ChannelID3.csv', encoding='utf-8', index=True)
df_split4.to_csv('../DATA/videosinfo/Final_ChannelID4.csv', encoding='utf-8', index=True)
print('partition done')
