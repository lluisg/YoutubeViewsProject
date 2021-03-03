#importing libraries
import csv
import os
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('../DATA/videosID8M_aux.csv', encoding = "utf-8")
    df.drop('pseudoId', inplace=True, axis=1)
    df.to_csv('../DATA/videosID8M_final.csv', encoding='utf-8', index=False)

    df_splitA, df_splitB = train_test_split(df, test_size=0.5, random_state=42, shuffle=False)
    df_split1, df_split2 = train_test_split(df_splitA, test_size=0.5, random_state=42, shuffle=False)
    df_split3, df_split4 = train_test_split(df_splitB, test_size=0.5, random_state=42, shuffle=False)

    df_split1.to_csv('../DATA/channelID8M/videosID8M_final1.csv', encoding='utf-8', index=False)
    df_split2.to_csv('../DATA/channelID8M/videosID8M_final2.csv', encoding='utf-8', index=False)
    df_split3.to_csv('../DATA/channelID8M/videosID8M_final3.csv', encoding='utf-8', index=False)
    df_split4.to_csv('../DATA/channelID8M/videosID8M_final4.csv', encoding='utf-8', index=False)

    # cp all list files

    print('partition done')
