import os
import glob
import pandas as pd

base_file = 'videosdata'
path = '../DATA/videosinfo/'

def read_file(filename):
    print('Reading ', filename)
    df = pd.read_csv(path+filename+'.csv')

    leng = len(df.index)
    print('length '+filename+':', leng)
    return df, leng

df_previous, lengthprevious = read_file(base_file+'_final')

df1, length1 = read_file(base_file+'1')
df2, length2 = read_file(base_file+'2')
df3, length3 = read_file(base_file+'3')
df4, length4 = read_file(base_file+'4')

df_total = pd.concat([df_previous, df1, df2, df3, df4])

print('MERGED SHAPE:', df_total.shape)
print('MERGED COLUMNS:', df_total.columns)

df_total.to_csv('../DATA/videosinfo/'+base_file+'_aux.csv', encoding='utf-8', index=False)
print('last: 20, 0, 0, 2614,   total: 499130')

# -------------------------------------------------- ALREADY READ
readed = []
print('\nlen readed:')
for f in ['1', '2', '3', '4']:
    with open(path+'list_channels'+f+'.txt', 'r') as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    readed.extend(lines)
    print(len(lines))

read_unique = list(set(readed))
print('total', len(readed), ', unique', len(read_unique))
print('previous: 7-0-0-133, 140-140')

with open(path+'list_channels.txt', 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]
readed.extend(lines)

read_unique = list(set(readed))
print('list_final', len(read_unique))
print('previous: 26506')

with open(path+'list_channels_aux.txt', 'w') as f:
    for l in read_unique:
        f.write(l+'\n')


# ---------------------------------------NOT READ
not_readed = []
print('\nlen not readed:')
for f in ['1', '2', '3', '4']:
    with open(path+'list_channels_not'+f+'.txt', 'r') as f:
        lines_n = f.readlines()
    lines_n = [x.rstrip() for x in lines_n]
    not_readed.extend(lines_n)
    print(len(lines_n))

not_read_unique = list(set(not_readed))
print('total', len(not_readed), ', unique', len(not_read_unique))
print('previous: 36-0-0-737, 773-773')

with open(path+'list_channels_not.txt', 'r') as f:
    lines_n = f.readlines()
lines_n = [x.rstrip() for x in lines_n]
not_readed.extend(lines_n)

not_read_unique = list(set(not_readed))
print('list_final', len(not_read_unique))
print('previous: 144791')
with open(path+'list_channels_not_aux.txt', 'w') as f:
    for l in not_read_unique:
        f.write(l+'\n')
