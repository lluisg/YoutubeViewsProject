import os
import glob
import pandas as pd

base_file = 'Final_VideosData'
path = 'DATA/videosinfo/'

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

print('Reading ', base_file, 'base file')
df_base = read_file_r('DATA/'+base_file+'.csv', True)
len_base = len(df_base.index)
print('length base:', len_base)


def read_file(key):
    print('Reading ', base_file, key)
    df = pd.read_csv(path+base_file+str(key)+'.csv')

    leng = len(df.index)
    print('length'+str(key)+':', leng)
    return df, leng

df1, length1 = read_file(1)
df2, length2 = read_file(2)
df3, length3 = read_file(3)
df4, length4 = read_file(4)

df_total = pd.concat([df_base, df1, df2, df3, df4])

print('MERGED SHAPE:', df_total.shape)
print('MERGED COLUMNS:', df_total.columns)

df_total.to_csv(path+base_file+'_merged.csv', encoding='utf-8', index=False)
print('last: 4020-19906-69117, 93043-83833')
