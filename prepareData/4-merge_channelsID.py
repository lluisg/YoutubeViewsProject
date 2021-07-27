import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

files = ['ChannelIDTrending', 'ChannelIDDataworld', 'ChannelID8M_final']

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
print('\nfinal length: ', len(list_channels))
set_channels = set(list_channels)
list_channels_nodupli = list(set_channels)
print('without duplicates: ', len(list_channels_nodupli))

# df_channels = pd.DataFrame(list_channels_nodupli,columns=['channelId'])
# df_channels.to_csv('../DATA/all_ChannelID.csv', encoding='utf-8', index=True)
with open('../DATA/all_ChannelID.txt', 'w') as f:
    for l in list_channels_nodupli:
        f.write(l+'\n')
print('\nlast: 4020-19906-155128  \ntotal: 179054-171295')

df_splitA, df_splitB = train_test_split(list_channels_nodupli, test_size=0.5, random_state=42, shuffle=False)
df_split1, df_split2 = train_test_split(df_splitA, test_size=0.5, random_state=42, shuffle=False)
df_split3, df_split4 = train_test_split(df_splitB, test_size=0.5, random_state=42, shuffle=False)

splits = [df_split1, df_split2, df_split3, df_split4]
for file_number in range(1,5):
    print('../DATA/videosinfo/all_ChannelID'+str(file_number)+'.txt')
    with open('../DATA/videosinfo/all_ChannelID'+str(file_number)+'.txt', 'w') as f:
        for l in splits[file_number-1]:
            f.write(l+'\n')

print('partition done')
