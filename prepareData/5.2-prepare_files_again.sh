#!/bin/bash

# UPDATING
# you have to update manually the aux -> final
cp ../DATA/videosinfo/videosdata_final.csv ../DATA/videosdata_final.csv

# SAVESTATE
cp ../DATA/videosinfo/list_channels_aux.txt ../DATA/videosinfo/list_channels.txt

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
cp ../DATA/videosinfo/list_channels_not_aux.txt  ../DATA/videosinfo/list_channels_not.txt

# RESTART TO 0
rm ../DATA/videosinfo/list_channels_not1.txt
touch ../DATA/videosinfo/list_channels_not1.txt
rm ../DATA/videosinfo/list_channels_not2.txt
touch ../DATA/videosinfo/list_channels_not2.txt
rm ../DATA/videosinfo/list_channels_not3.txt
touch ../DATA/videosinfo/list_channels_not3.txt
rm ../DATA/videosinfo/list_channels_not4.txt
touch ../DATA/videosinfo/list_channels_not4.txt

rm ../DATA/videosinfo/videosdata1.csv
rm ../DATA/videosinfo/videosdata2.csv
rm ../DATA/videosinfo/videosdata3.csv
rm ../DATA/videosinfo/videosdata4.csv

echo 'you have to update manually the videosdata aux -> final'
