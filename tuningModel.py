import argparse
import os
import pandas as pd
import numpy as np
import pathlib
import datetime
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import math
from sklearn.model_selection import train_test_split

from model import LSTMModel

def roundVisit(visit):
    # We decided that if we are rounding the vists on 1k, it would not help the
    # bigger visit videos, so a custom round function will be used.
    # It consists in rounding to tens of the videos, so it would be rounded to
    # the 1k's below 10k, to 10k between 10k and 100k, etc.
    # So a 2.123.432 visits video will be rounded to 2M, but a 530.234 visits
    # video will be rounded to 500k
    if visit < 1000:
        rounding_number = 1000
    else:
        power = math.floor(math.log10(visit))
        rounding_number = math.pow(10, power)

    rounded_result = round(visit/rounding_number)*rounding_number

    return rounded_result


def convertDuration(time):
    s = 0
    m = 0
    h = 0
    d = 0
    if 'S' in time:
        if 'M' in time:
            s = int(time.split('S')[0].split('M')[1])
        elif 'H' in time:
            s = int(time.split('S')[0].split('H')[1])
        elif 'D' in time:
            s = int(time.split('S')[0].split('DT')[1])
        else:
            s = int(time.split('S')[0].split('PT')[1])
    if 'M' in time:
        if 'H' in time:
            m = int(time.split('M')[0].split('H')[1])
        elif 'D' in time:
            m = int(time.split('M')[0].split('DT')[1])
        else:
            m = int(time.split('M')[0].split('PT')[1])
    if 'H' in time:
        if 'D' in time:
            h = int(time.split('H')[0].split('DT')[1])
        else:
            h = int(time.split('H')[0].split('PT')[1])
    if 'D' in time:
        d = int(time.split('D')[0].split('P')[1])

    return s + m*60 + h*60*60 + d*24*60*60


def publi2value(publication):
    date = publication.split('T')[0]
    month, day, year = date.split('-')

    value = int(year)*10000 + int(month)*100 + int(day)
    return value

def value2tens(value):
    # In order to reduce the number of outputs that will be needed on the model,
    # a custom outputs will be implemented. In it the tens will indicate
    # the power of 10 used.
    if value == 0:
        result = 0
    else:
        power = math.floor(math.log10(value))
        num = int(value / math.pow(10, power))
        result = power*10 -30 + num #rounded to 1k, that why -30

    return result


def prepareData(data_path):
    df = pd.read_csv('DATA/Final_videosDataClean.csv')

    #-------------------Mini version--------------------------
    if args.mini == 1:
        print('USING MINI VERSION')
        df = df.head(500)
        print('Mini data len', df.shape[0])

    #CLEANING ------------------------------------------------------------------
    # Removing elements from table which have NaN's
    df.dropna(how='any', inplace=True)

    # UNIQUE ------------------------------
    list_videosId = df['id'].tolist()
    print('We have {} different videos'.format(len(list_videosId)))
    set_videosId = set(list_videosId)
    list_videosId = list(set_videosId)
    print('from which {} are unique'.format(len(list_videosId)))

    list_channelId = df['channelId'].tolist()
    set_channelId = set(list_channelId)
    list_channelId = list(set_channelId)
    print('We have {} unique channels'.format(len(list_channelId)))

    # dropping duplicated videos
    df = df.drop_duplicates(subset='id', keep='first')

    #Removing channels if videos after deleting NaN and duplicates are below 10
    list_channelids = df['channelId'].value_counts()
    number_less10 = len([c for c in list_channelids if c < 10])

    for ch in [i for i in list_channelids.index if list_channelids[i] < 10]:
      df = df[df.channelId != ch]

    del list_channelids
    del number_less10

    list_videosId = df['id'].tolist()
    print('After dropping NaNs and duplicates we have {} different videos'.format(len(list_videosId)))
    set_videosId = set(list_videosId)
    list_videosId = list(set_videosId)

    list_channelId = df['channelId'].tolist()
    set_channelId = set(list_channelId)
    list_channelId = list(set_channelId)
    print('from which {} are unique. And {} diferent channels'.format( len(list_videosId), len(list_channelId)))

    views_channel = []
    publication_channel = []
    ratio_channel = []
    duration_channel = []
    comments_channel = []

    for channel in list_channelId:
        ratio_likes = []
        viewsvideo = [int(value2tens(roundVisit(visit))) for visit in df.loc[df['channelId'] == channel]['viewCount'].tolist()]
        publicationvideo = [publi2value(publi) for publi in df.loc[df['channelId'] == channel]['publishedAt'].tolist()]
        likesvideo = df.loc[df['channelId'] == channel]['likeCount'].tolist()
        dislikesvideo = df.loc[df['channelId'] == channel]['dislikeCount'].tolist()
        durationvideo = [convertDuration(time) for time in df.loc[df['channelId'] == channel]['duration'].tolist()]
        commentsvideo = df.loc[df['channelId'] == channel]['commentCount'].tolist()


        for likes, dislikes in zip(likesvideo, dislikesvideo):
            if likes == 0 and dislikes == 0:
                ratio_likes.append(0)
            else:
                ratio_likes.append(likes/(likes+dislikes))

        views_channel.append(viewsvideo)
        publication_channel.append(publicationvideo)
        ratio_channel.append(likesvideo)
        duration_channel.append(durationvideo)
        comments_channel.append(commentsvideo)

    print('views: ', np.shape(views_channel),
            'publi: ', np.shape(publication_channel),
            'ratio likes: ', np.shape(ratio_channel),
            'duration: ', np.shape(duration_channel),
            'comments: ', np.shape(comments_channel))

    del list_videosId
    del set_videosId
    del list_channelId
    del set_channelId
    del df

    print('Max value of views:', max([max(i) for i in views_channel]))

    group_input = []
    used = 'The model will use the features:'
    if '1' in args.input_elements:
        group_input.append(views_channel)
        used += ' views'
    if '2' in args.input_elements:
        group_input.append(duration_channel)
        used += ' duration'
    if '3' in args.input_elements:
        group_input.append(publication_channel)
        used += ' publication'
    if '4' in args.input_elements:
        group_input.append(ratio_channel)
        used += ' ratio'
    if '5' in args.input_elements:
        group_input.append(comments_channel)
        used += ' comments'
    print(used)

    X = []
    y = []
    if len(args.input_elements) == 5:
        for views, duration, publi, ratio, comments in zip(*group_input):
            elementX = []
            for v, d, p, r, c in zip(views, duration, publi, ratio, comments):
                elementX.append([v, d, p, r, c])

            X.append(elementX[-10:-1])
            y.append(views[-1])

    elif len(args.input_elements) == 4:
        for views, el2, el3, el4 in zip(*group_input):
            elementX = []
            for v, e2, e3, e4 in zip(views, el2, el3, el4):
                elementX.append([v, e2, e3, e4])

            X.append(elementX[-10:-1])
            y.append(views[-1])

    elif len(args.input_elements) == 3:
        for views, el2, el3 in zip(*group_input):
            elementX = []
            for v, e2, e3 in zip(views, el2, el3):
                elementX.append([v, e2, e3])

            X.append(elementX[-10:-1])
            y.append(int(value2tens(views[-1])))

    elif len(args.input_elements) == 2:
        for views, el2 in zip(*group_input):
            elementX = []
            for v, e2 in zip(views, el2):
                elementX.append([v, e2])

            X.append(elementX[-10:-1])
            y.append(int(value2tens(views[-1])))

    elif len(args.input_elements) == 1:
        for views in zip(*group_input):
            elementX = []
            for v in zip(views):
                elementX.append(v)

            X.append(elementX[-10:-1])
            y.append(int(value2tens(views[-1])))

    # to clean some space in variables
    del views_channel
    del publication_channel
    del ratio_channel
    del duration_channel

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=args.seed)
    Xvalid, Xtest, yvalid, ytest = train_test_split(Xtest, ytest, test_size=0.5, random_state=args.seed)
    print('X: {} -> Xtrain: {}, Xvalid: {}, Xtest: {}'.format(np.shape(X), np.shape(Xtrain), np.shape(Xvalid), np.shape(Xtest)))
    print('y: {} -> ytrain: {}, yvalid: {}, ytest: {}'.format(np.shape(y), np.shape(ytrain), np.shape(yvalid), np.shape(ytest)))

    Xtrain_batch = np.reshape(np.array(Xtrain),(-1,9, len(args.input_elements)))
    ytrain_batch = np.reshape(np.array(ytrain),(-1,1, args.output_elements))
    # print("Xtrain shape: {} --> (Batch Size, Sequence Length, Element Size)".format(Xtrain_batch.shape))
    # print("ytrain shape: {} --> (Batch Size, Sequence Length, Element Size)".format(ytrain_batch.shape))

    Xvalid_batch = np.reshape(np.array(Xvalid),(-1,9, len(args.input_elements)))
    yvalid_batch = np.reshape(np.array(yvalid),(-1,1, args.output_elements))
    # print("Xvalid shape: {} --> (Batch Size, Sequence Length, Element Size)".format(Xvalid_batch.shape))
    # print("yvalid shape: {} \t--> (Batch Size, Sequence Length, Element Size)".format(yvalid_batch.shape))

    Xtest_batch = np.reshape(np.array(Xtest),(-1,9, len(args.input_elements)))
    ytest_batch = np.reshape(np.array(ytest),(-1,1, args.output_elements))
    # print("Xtest shape: {} --> (Batch Size, Sequence Length, Element Size)".format(Xtest_batch.shape))
    # print("ytest shape: {} \t--> (Batch Size, Sequence Length, Element Size)".format(ytest_batch.shape))

    train_sample_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain_batch).float(), torch.from_numpy(ytrain_batch).float().type(torch.long))
    valid_sample_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xvalid_batch).float(), torch.from_numpy(yvalid_batch).float().type(torch.long))
    test_sample_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtest_batch).float(), torch.from_numpy(ytest_batch).float().type(torch.long))

    train_loader = DataLoader(train_sample_ds, shuffle=False, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_sample_ds, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_sample_ds, shuffle=False, batch_size=args.batch_size)

    torch.save(train_loader, data_path+'/train_dataloader.pth')
    torch.save(valid_loader, data_path+'/valid_dataloader.pth')
    torch.save(test_loader, data_path+'/test_dataloader.pth')
    # the other variables will clean themselves after getting out of scope
    return train_loader, valid_loader, test_loader

def loadData(data_path):
    # print('load data:', data_path)
    train_loader = torch.load(data_path+'/train_dataloader.pth')
    valid_loader = torch.load(data_path+'/valid_dataloader.pth')
    test_loader = torch.load(data_path+'/test_dataloader.pth')
    return train_loader, valid_loader, test_loader

def save_values(values, name, path):
    with open(os.path.join(path, name+'.txt'), 'w+') as f:
        f.write(str(values))

def train(config, model, train_loader, valid_loader, epochs, optimizer, loss_fn, device, path):
    global losses
    loss_return = []
    loss_return_valid = []
    best_loss = 1e30
    best_epoch = 0
    accuracies = []
    valid_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batchs_done = 0
        total_loss = 0
        outputs_epochs = []

        for batch in train_loader:
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # print('batchh:', np.shape(batch_X), np.shape(batch_y))

            out = model(batch_X)

            batch_loss = 0
            for result, target in zip(out, batch_y):
                result = result.reshape((1,-1))
                target = target.reshape((1))
                # print('heeey', np.shape(target), target, np.shape(result), result)
                loss = loss_fn(result, target) #index of the k
                batch_loss += loss

            optimizer.zero_grad() #no tinc molt clar
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
        loss_return.append(total_loss / len(train_loader))

        if epoch % 5 == 0:
            valid_epoch += 1
            model.eval()
            total_valid_loss = 0

            for batch in valid_loader:
                batch_X, batch_y = batch
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                out = model(batch_X)

                valid_loss = 0
                outputs_batch = []
                for result, target in zip(out, batch_y):
                    result = result.reshape((1,-1))
                    target = target.reshape((1))

                    loss = loss_fn(result, target) #index of the k
                    valid_loss += loss
                    result_value = np.argmax(result.detach().cpu().numpy())
                    outputs_batch.append(result_value)

                outputs_epochs.append(outputs_batch)
                total_valid_loss += valid_loss.data.item()

            valid_accuracy = computePerformanceTest(outputs_epochs, valid_loader, 0)
            # print('valid acc', valid_accuracy)
            accuracies.append(valid_accuracy)

            print("Validation Loss: {}, Accuracy: {}".format(total_valid_loss / len(valid_loader), valid_accuracy))
            loss_return_valid.append(total_valid_loss / len(valid_loader))

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            if args.save_training:
                folder_path = os.path.join(config['output_path'], str(config['dropout'])+'-'+str(config['epochs'])+ \
                    '-'+str(config['hidden_fc'])+'-'+str(config['hidden_lstm'])+'-'+str(config['lr']))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                save_values(loss_return, 'losses', folder_path)
                save_values(loss_return_valid, 'valid_losses', folder_path)
                save_values(accuracies, 'accuracies', folder_path)
                print('Outputs training/eval saved on epoch', epoch)

            tune.report(valid_loss = (total_valid_loss/len(valid_loader)), valid_acc = valid_accuracy)


def train_tuning(config, checkpoint_dir=None):
    model = LSTMModel(config["input"], config['hidden_lstm'], config['hidden_fc'], config['output'], config['dropout'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader, valid_loader, _ = loadData(config['data_path'])

    train(config, model, train_loader, valid_loader, config['epochs'], optimizer, loss_fn, device, config["path"])

def save_testresults(values, name, load_path, output_path):
    test_loader = torch.load(load_path+'/test_dataloader.pth')

    f = open(os.path.join(output_path, name+'.txt'), 'w')
    f.write('truth--output\n')
    f.close()
    with open(os.path.join(output_path, name+'.txt'), 'a') as f:
        for b, v in zip(test_loader, values):
            X, y = b
            for t, r in zip(y, v):
                r.reshape((1))
                # print('for truth: {}, result: {}'.format(t.numpy()[0], r)
                f.write(str(t.numpy()[0][0])+' -- '+str(r)+'\n')

def test(model, test_loader, diff, device, path):
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():

        for batch in test_loader:
            outputs_batch = []
            batch_X, batch_y = batch
            len_batch = len(batch_X)

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_X)

            for result in out:
                result = result.reshape((1,-1))
                result_value = np.argmax(result.detach().cpu().numpy())
                outputs_batch.append(result_value)

            outputs.append(outputs_batch)

    return outputs

def test_tuning(model, path, data_path, output_path, device="cpu"):
    _, _, test_loader = loadData(data_path)
    diff = 0
    output = test(model, test_loader, diff, device, path)

    if args.save_outputs == 1:
        save_testresults(output, 'outputs_test', data_path, path)

    for diff in [0, 1, 2, 10]: #extra distance values for test evaluation
        correct = computePerformanceTest(output, test_loader, diff, True)

    return correct

def computePerformanceTest(results, test_loader, difference=0, printing=False):
    correct_dict = {}
    total = 0
    correct = 0
    for b, o in zip(test_loader, results):
        X, y = b
        for t, r in zip(y, o):
            r.reshape((1))
            # print('for truth: {}, result: {}'.format(t.numpy()[0], r))
            if t-difference <= r <= t+difference:
                correct += 1
            total += 1

    if printing:
        print('The model outputs {:.2f}% ({}/{}) of correct values with a margin of {}'.format(correct/total*100, correct, total, difference))

    return correct/total


def main(path, num_samples=30, max_num_epochs=10000, gpus_per_trial=2):
    model_path = path+'/best_model.pt'

    config = {
        'input': len(args.input_elements),
        'hidden_lstm': tune.sample_from(lambda _: 2**np.random.randint(7, 12)), #hidden layers between 128 and 2048
        'hidden_fc': tune.sample_from(lambda _: 2**np.random.randint(7, 12)),
        'output': 70,
        'dropout': tune.sample_from(lambda _: 0.05*np.random.randint(0, 7)), #dropout between 0 and 0.3
        'lr': tune.loguniform(1e-6, 1e-3),
        'epochs': tune.choice([100, 500, 1000, 5000, 10000]),
        'path': path,
        'data_path': data_path,
        'output_path': output_path
      }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=max_num_epochs/5,
        grace_period=1,
        reduction_factor=2,
        brackets=2)

    result = tune.run(
        tune.with_parameters(train_tuning),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        metric='valid_loss',
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler)

    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["valid_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["valid_acc"]))

    best_trained_model = LSTMModel(best_trial.config["input"], best_trial.config["hidden_lstm"], best_trial.config["hidden_fc"], best_trial.config["output"], best_trial.config["dropout"])
    #SAVE BEST MODEL
    torch.save(best_trained_model.state_dict(), model_path)
    print('Model saved\n')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    print("Best trial test set accuracy:")
    test_acc = test_tuning(best_trained_model, path, data_path, output_path, device)

def checkVariables():
    # check variable input elements
    for i in str(args.input_elements):
        if not i in '12345': #number not 1,2,3,4 or 5
            return False, 'input elements'
    if str(args.input_elements).count('1') == 0: #obligatory use feature 1
        return False, 'input elements'
    for i in '12345': #not repeated numbers
        if str(args.input_elements).count(i) > 1:
            return False, 'input elements'

    return True, None


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--mini', type=int, default=0, metavar='MI', help='1 if you want to use a minified version of the data')
    parser.add_argument('--save_training', type=int, default=1, metavar='ST', help='Save the losses obtained in the training and validation')
    parser.add_argument('--save_outputs', type=int, default=1, metavar='SO', help='Save the values resultant on the test file to analyze')
    parser.add_argument('--new_model', type=int, default=1, metavar='NM', help='1 if you want to create the model from 0 again. THIS WILL REMOVE THE INFO FROM THE PREVIOUS MODEL!')
    parser.add_argument('--new_data', type=int, default=1, metavar='ND', help='1 if you want to repreprocess the data.')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='Number to random seed')
    parser.add_argument('--batch_size', type=int, default=256, metavar='BS', help='Batch size for the DataLoaders')
    parser.add_argument('--input_elements', type=str, default='12345', metavar='IE', help='Elements as input: 1-views(must), 2-duration, 3-publication, 4-ratio likes/dislikes')
    parser.add_argument('--output_elements', type=int, default=1, metavar='OE', help='Elements as output, for now only 1 is valid (the views)')

    parser.add_argument('--max_epochs', type=int, default=None, metavar='EME', help='Number of max epochs the model will train (it will be rounded to multiple of 5).')
    parser.add_argument('--iterations', type=int, default=15, metavar='NR', help='Number of iterations with different parameter values the model will run.')

    parser.add_argument('--name', type=str, default='', metavar='NA', help='Name of the model (folder too), if not it will have the date as the name')

    args = parser.parse_args()

    if not args.name:
        date = datetime.datetime.now().strftime("%m-%d-%Y")
        args.name = str(publi2value(date))

    path = os.path.join(str(pathlib.Path().absolute()), args.name)
    if args.new_model == 1:
        if os.path.exists(path):
            for fi in os.listdir(path):
                if fi != 'data':
                    try:
                        os.remove(os.path.join(path, fi))
                    except Exception as e:
                        shutil.rmtree(os.path.join(path, fi), ignore_errors=True)
    if not os.path.exists(path):
        os.mkdir(path)

    data_path = os.path.join(path, 'data')
    if args.new_data == 1:
        shutil.rmtree(data_path, ignore_errors=True)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    output_path = os.path.join(path, 'training_output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('Working on path:', path)

    checked, problem = checkVariables()
    if checked:
        if os.path.isfile(os.path.join(data_path,'train_dataloader.pth')) and \
            os.path.isfile(os.path.join(data_path,'valid_dataloader.pth')) and \
            os.path.isfile(os.path.join(data_path,'test_dataloader.pth')):
            print('--- Data already existing')
        else:
            print('--- Data from 0')
            prepareData(data_path)
            print('Data prepared\n')

        if os.path.isfile(os.path.join(path, 'best_model.pt')) and args.new_model == 0:
            print('--- The model already exists.')
        else:
            print('--- Model from 0')
            if args.max_epochs == None:
                main(path=path, num_samples=args.iterations, max_num_epochs=10000, gpus_per_trial=2)
                pass
            else:
                if args.max_epochs >= 20:
                    max_epochs = int(np.round(args.max_epochs/5)*5)
                    main(path=path, num_samples=args.iterations, max_num_epochs=max_epochs, gpus_per_trial=2)
                else:
                    print('Minimum training of 20 epochs')

        print('FINISHED TRAINING AND TESTING!')
    else:
        print('Check the variable {}.'.format(problem))

    end_time = time.time()
    print("Took --- {}m ---".format((end_time - start_time)/60))
