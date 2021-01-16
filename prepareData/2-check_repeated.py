import pandas as pd
# channelId,id,title,publishedAt,categoryId,channelTitle,duration,viewCount,likeCount,dislikeCount,commentCount
# UCZyQcFvEE8VWNheDYyAHUzg,hxntADfGtqk,ObÃ© Fitness CEOs Talk 'Entertrainment' and Adapting to Change | WWD,2020-07-22T00:00:12Z,26,WWD,PT11M44S,350.0,3.0,0.0,0.0
# UC5nc_ZtjKW1htCVZVRxlQAQ,ftxx-l_0Lww,Midsplit - Out Of Here (feat. Tilsen),2020-10-24T17:56:08Z,24,MrSuicideSheep,PT2M45S,171873.0,9943.0,115.0,251.0

df_movies = pd.read_csv('../DATA/AllTrendingChannels.csv', encoding = "utf-8")
list_videosId = df_movies['channel_title'].tolist()
print(df_movies)

already_seen = {}
repeated = {}
for ind, id in enumerate(list_videosId):
    if id in already_seen:
        repeated[id] = [ind, already_seen[id]]
    else:
        already_seen[id] = ind

print('{} videos repeated'.format(len(repeated)))

for id in repeated:
    print(id, repeated[id])
    print(df_movies.iloc[repeated[id][0]].values)
    print(df_movies.iloc[repeated[id][1]].values)
