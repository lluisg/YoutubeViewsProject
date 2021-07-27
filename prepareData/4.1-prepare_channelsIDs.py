import os
import glob
import pandas as pd

filen = "../DATA/videosdata.csv"

def read_file_r(filename, check_r):
    print('Reading', filename)
    df = pd.read_csv(filename, encoding = "utf-8", lineterminator='\n')

    if 'commentCount\r' in df.columns and check_r:
        print('removed commentCount with r')
        df['commentCount\r'] = df['commentCount\r'].str[:-1]
        df['commentCount\r'] = pd.to_numeric(df['commentCount\r'])

        df['commentCount'].update(df.pop('commentCount\r'))
        # print(df.head())
    return df

df_base = read_file_r(filen , True)
len_base = len(df_base.index)
print('length base:', len_base)

print('writing csv cleaned')
df_base.to_csv(filen , encoding='utf-8', index=False)
df_base.to_csv('../DATA/videosinfo/videosdata_final.csv' , encoding='utf-8', index=False)


print('writing txt chanels list')
list_channelIds = df_base['channelId'].tolist()
list_unique = list(set(list_channelIds))

with open('../DATA/videosdata_already.txt', 'w') as f:
    for l in list_unique:
        f.write(l+'\n')

with open('../DATA/videosinfo/list_channels.txt', 'w') as f:
    for l in list_unique:
        f.write(l+'\n')

#to create the empty file
open('../DATA/videosinfo/list_channels_not.txt', 'w')

print('Done')
