#importing libraries
import csv
import os
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen

def get_real_id(random_id: str) -> str:
    url = 'http://data.yt8m.org/2/j/i/{}/{}.js'.format(random_id[0:2], random_id)
    try:
        request = urlopen(url).read()
    except:
        return None
    real_id = request.decode()
    return real_id[real_id.find(',') + 2:real_id.find(')') - 1]

if __name__ == "__main__":
    df_pseudo = pd.read_csv('DATA/videospseudoID8M.csv', encoding = "utf-8")
    list_pseudo = df_pseudo['pseudoId'].values.tolist()

    if os.path.isfile('DATA/videosID8M.csv'):
        df_ids = pd.read_csv('DATA/videosID8M.csv', encoding = "utf-8")
        list_ids = df_ids['pseudoId'].values.tolist()
    else:
        df_ids = None
        print('Creating videosID8M file from 0')
    if os.path.isfile('DATA/videosID8M_notfound.csv'):
        df_notf = pd.read_csv('DATA/videosID8M_notfound.csv', encoding = "utf-8")
        notfound_list = df_notf['pseudoId'].values.tolist()
        print('Started with {} not found'.format(len(notfound_list)))
    else:
        notfound_list = []
        print('notfound from 0')

    list_real_id = []
    none = 0
    repeated = 0
    saved = False
    for ind, pid in enumerate(list_pseudo):
        if pid not in list_ids and pid not in notfound_list:
            real_id = get_real_id(pid)
            if real_id != None:
                # print(ind, real_id)
                list_real_id.append([real_id, pid])
            else:
                notfound_list.append(pid)
        else:
            repeated += 1

        if len(list_real_id) != 0 and len(list_real_id) % 15000 == 0: #save every N elements done
            print('Saved at {} with {} new IDs, {} repeated and {} not found'.format(ind, len(list_real_id), repeated, len(notfound_list)))
            df_out = pd.concat([df_ids, pd.DataFrame(list_real_id, columns = ['videosId', 'pseudoId'])], ignore_index=True)
            df_out.to_csv('DATA/videosID8M_aux.csv', encoding='utf-8', index=False)
            #save not found to later check
            df_notfound = pd.DataFrame(notfound_list, columns = ['pseudoId'])
            df_notfound.to_csv('DATA/videosID8M_notfound.csv', encoding='utf-8', index=False)
            saved = True
            # break

    if saved == False:
        print('Saved only 1 time with {} new IDs, {} repeated and {} not found'.format(len(list_real_id), repeated, len(notfound_list)))
        df_out = pd.concat([df_ids, pd.DataFrame(list_real_id, columns = ['videosId', 'pseudoId'])], ignore_index=True)
        df_out.to_csv('DATA/videosID8M_aux.csv', encoding='utf-8', index=False)
        #save not found to later check
        df_notfound = pd.DataFrame(notfound_list, columns = ['pseudoId'])
        df_notfound.to_csv('DATA/videosID8M_notfound.csv', encoding='utf-8', index=False)
    else:
        print('Saved already')
