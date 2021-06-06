#!/bin/bash

# SAVESTATE
cp ../DATA/videosinfo/list_channels.txt ../DATA/videosinfo/list_channels_savestate.txt

# RESTART TO 0
rm ../DATA/videosinfo/list_channels1.txt
touch ../DATA/videosinfo/list_channels1.txt
rm ../DATA/videosinfo/list_channels2.txt
touch ../DATA/videosinfo/list_channels2.txt
rm ../DATA/videosinfo/list_channels3.txt
touch ../DATA/videosinfo/list_channels3.txt
rm ../DATA/videosinfo/list_channels4.txt
touch ../DATA/videosinfo/list_channels4.txt

# SAVESTATE
cp ../DATA/videosinfo/list_channels_not.txt ../DATA/videosinfo/list_channels_not_savestate.txt

# RESTART TO 0
rm ../DATA/videosinfo/list_channels_not1.txt
touch ../DATA/videosinfo/list_channels_not1.txt
rm ../DATA/videosinfo/list_channels_not2.txt
touch ../DATA/videosinfo/list_channels_not2.txt
rm ../DATA/videosinfo/list_channels_not3.txt
touch ../DATA/videosinfo/list_channels_not3.txt
rm ../DATA/videosinfo/list_channels_not4.txt
touch ../DATA/videosinfo/list_channels_not4.txt

rm ../DATA/videosinfo/Final_VideosData1.csv
rm ../DATA/videosinfo/Final_VideosData2.csv
rm ../DATA/videosinfo/Final_VideosData3.csv
rm ../DATA/videosinfo/Final_VideosData4.csv
