import os
import glob
import pandas as pd

langs = ['CA', 'DE', 'FR', 'GB', 'IN', 'JP', 'KR', 'MX', 'RU', 'US']
list_channels = []
list_errors = []
for lang in langs:
    print('Reading ', lang)
    df_videos = pd.read_csv('DDBB_Trending/'+lang+'videos.csv', encoding = "ISO-8859-1")
    df_out = df_videos[['channel_title']]
    error_encoding = 0

    for index, row in df_out.iterrows():
        try:
            row['channel_title'].encode('ascii')
        except UnicodeEncodeError:
            error_encoding += 1
            list_errors.append(row['channel_title'])
        else:
            list_channels.append(row['channel_title'])

    print('length: ', df_out.shape[0])
    print(error_encoding, 'ignored words (no valid encoding)')

list_channels = list(filter(None, list_channels))
print('\nFinal length: ', len(list_channels))
set_channels = set(list_channels)
list_channels_nodupli = list(set_channels)
print('without duplicates: ', len(list_channels_nodupli))

df_channels = pd.DataFrame(list_channels_nodupli,columns=['channel_title'])
df_channels.to_csv('DATA/AllTrendingChannels.csv', encoding='utf-8', index=True)

#errors version
list_errors = list(filter(None, list_errors))
print('\nFinal errors length: ', len(list_errors))
set_errors = set(list_errors)
list_errors_nodupli = list(set_errors)
print('errors without duplicates: ', len(list_errors_nodupli))

df_errors = pd.DataFrame(list_errors,columns=['channel_title'])
df_errors.to_csv('DATA/AllTrendingErrors.csv', encoding='utf-8', index=True)
