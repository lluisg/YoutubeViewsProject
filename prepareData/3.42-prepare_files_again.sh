#!/bin/bash

# SAVESTATE
cp ../DATA/channelID8M/list_8M_channelsId_final.txt ../DATA/channelID8M/list_8M_channelsId.txt

# RESTART TO 0
rm ../DATA/channelID8M/list_8M_channelsId1.txt
touch ../DATA/channelID8M/list_8M_channelsId1.txt
rm ../DATA/channelID8M/list_8M_channelsId2.txt
touch ../DATA/channelID8M/list_8M_channelsId2.txt
rm ../DATA/channelID8M/list_8M_channelsId3.txt
touch ../DATA/channelID8M/list_8M_channelsId3.txt
rm ../DATA/channelID8M/list_8M_channelsId4.txt
touch ../DATA/channelID8M/list_8M_channelsId4.txt

# SAVESTATE
cp ../DATA/channelID8M/list_8M_channelsId_notF_final.txt ../DATA/channelID8M/list_8M_channelsId_notF.txt

# RESTART TO 0
rm ../DATA/channelID8M/list_8M_channelsId_notF1.txt
touch ../DATA/channelID8M/list_8M_channelsId_notF1.txt
rm ../DATA/channelID8M/list_8M_channelsId_notF2.txt
touch ../DATA/channelID8M/list_8M_channelsId_notF2.txt
rm ../DATA/channelID8M/list_8M_channelsId_notF3.txt
touch ../DATA/channelID8M/list_8M_channelsId_notF3.txt
rm ../DATA/channelID8M/list_8M_channelsId_notF4.txt
touch ../DATA/channelID8M/list_8M_channelsId_notF4.txt

rm ../DATA/channelID8M/ChannelID8M1.csv
rm ../DATA/channelID8M/ChannelID8M2.csv
rm ../DATA/channelID8M/ChannelID8M3.csv
rm ../DATA/channelID8M/ChannelID8M4.csv
