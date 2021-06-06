# YoutubeViewsProject (Not Finished)

A more detailed description and discussion of the project can be found in the research paper. [**LINK**](https://drive.google.com/file/d/1sjb48ug70FkwioMY3brQL2vwMok3z7si/view?usp=sharing).

## Objective
Youtube is one of the most important platforms in the world, and there exists a lot of interest in the content creators side to predict the number of views their future video will get.
This project consists of 2 main objectives:
- Recompile sequential information of the videos uploaded by different channels on Youtube.
- Create a model able to predict the future views obtained by a video taking into account information available in their previous videos

## Data
To obtain the data I want from Youtube, I used the data from 3 different sources:

- [**Trending YouTube Video Statistics by Mitchell J on Kaggle**](https://www.kaggle.com/datasnaek/youtube-new?select=GBvideos.csv).
- [**Youtube Video and Channel Metadata by Vishwanath Seshagiri on dataworld**](https://data.world/sevenup13/youtube-video-and-channel-metadata).
- [**YouTube-8M Dataset by Youtube**](https://research.google.com/youtube8m/index.html).

The final data used to train and evaluate the project is on the Final_videosDataClean.csv file inside the DATA directory. Each row is a unique video that contains 11 different features:
[ categoryId, channelId, channelTitle, commentCount, dislikeCount, duration, id, likeCount, publishedAt, title, viewCount]. Despite this, a preprocessing is applied before using the data to train or evaluate.

The other files inside the same directory are the different steps from the original datasets to the final, as implemented in the files on the prepareData directory.

## Results
I started collecting data around November 2020, although there have been some hiatus.

One thing to have in mind is that with the limitation of calls on the Youtube API, it is pretty slow to collect the channel uploaded videos information.

**As of 21/01/2021:**
I have compiled more than 80.000 unique channels from the original 3 sources.

From those channels, I collected information of more than 250.000 unique videos from more than 13.000 channels.

And with those videos information I built a model that obtained predictions with 17.53% accuracy.

This is not considered solving the problem, but it can indicate how complex is the problem.

## Usage:
To implement this project you require the Languages/Libraries explained the next section, be sure to have them installed.

If you want to modify the model, the model itself can be found on the model.py file, while the processes to train, evaluate, etc can be found on the tuningModel.py file.

If you don't want to change the implementation, only use different parameter values, call the model with the -h parameter to know which parameters are modifiable and their types and descriptions. Any parameter has to be called as in the following
```
$ python tuningModel.py --parameter_name=parameter_value
```
If done correctly the system will yield a folder with the name indicated (such as the model_extra and model_final examples), this contains the model, the dataloaders, the outputs of the test and the progression of the model during the training and validation. Additionally, a log file will appear with the name indicated plus '_output'.

In case you wanted to modify and/or get information with the use of the files inside prepareData, you should create a .env file with your google API key.


### Dependencies:
| Dependencies |         |          |         |        |                 |
|--------------|---------|----------|---------|--------|-----------------|
| argparse     | csv     | datetime | dotenv  | glob   | googleapiclient |
| json         | math    | numpy    | os      | pandas | pathlib         |
| pytorch      | raytune | shutil   | sklearn | time   |                 |

## Future Plans:
As the project is not finished I'm still working on it collecting data, and once I have a considerable amount of it (right now I don't have that much proportionally) I will try to train the model again. In the research paper, I mentioned improvements to be implemented, despite this once I finish the main parts I will decide if trying to implement any of them or not.

One thing I plan to implement in the future is to host the model in a website in a user-friendly manner, so anyone can ask for a prediction of a Youtube channel video.


## Rights
Feel free to use or fork the code and modify as you prefer. Although a mention is always appreciated.

Good Luck!
