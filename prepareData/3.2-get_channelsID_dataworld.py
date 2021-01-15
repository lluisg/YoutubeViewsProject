#importing libraries
import csv
import os
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
    with open('DATA/YouTubeDataset_withChannelElapsed.json') as f:
      data = json.load(f)
    # print(data[0])

    channel_ids = []
    min_subs = 50000
    for element in data:
        if int(element['subscriberCount']) > min_subs:
            channel_ids.append(element['channelId'])

    print(len(data), 'elements')
    print(len(channel_ids), 'elements + '+str(min_subs)+' subscribers')
    set_channel_ids = set(channel_ids)
    list_channel_ids = list(set_channel_ids)
    print(len(list_channel_ids), 'elements no duplicates')

    df = pd.DataFrame(list_channel_ids, columns =['channelId'])
    df.to_csv('DATA/ChannelIDDataworld.csv', encoding='utf-8', index=True)
