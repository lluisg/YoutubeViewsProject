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
