import argparse
import json
import os

import numpy as np
import scipy.io
from scipy.signal import butter

from utils import get_nframe_video, read_from_txt
from data_generator import rPPG_Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from higher_model import TSCAN
from post_process import calculate_metric
torch.manual_seed(100)
np.random.seed(100)
# %%
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str,
                    default='/gscratch/xxx/xxx/data3/mnt/', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='./checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-tr_data', '--tr_dataset', type=str, default='AFRL', help='training dataset name')
parser.add_argument('-ts_data', '--ts_dataset', type=str, default='AFRL', help='test dataset name')
parser.add_argument('-tr_txt', '--train_txt', type=str, default='./filelists/AFRL/36/meta/train.txt', help='train file')
parser.add_argument('-ts_txt', '--test_txt', type=str, default='./filelists/AFRL/36/meta/test.txt', help='test file')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=48,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-x', '--batch_size', type=int, default=24,
                    help='batch')
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for 3DCNN')
parser.add_argument('-save', '--save_all', type=int, default=1,
                    help='save all or not')
parser.add_argument('-shuf', '--shuffle', type=str, default=True,
                    help='shuffle samples')
parser.add_argument('-freq', '--fs', type=int, default=25,
                    help='shuffle samples')
parser.add_argument('--window_size', type=int, default=360,
                                    help='window size for filtering and FFT')
parser.add_argument('--eval_only', type=int, default=0,
                                    help='if eval only')
parser.add_argument('--signal', type=str, default='pulse')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Spliting Data

print('Spliting Data...')
subNum = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27])
taskList = list(range(1, args.nb_task+1))
[b, a] = butter(1, [0.75 / args.fs * 2, 2.5 / args.fs * 2], btype='bandpass')


def train(args):

    checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    print('================================')
    print('Train...')

    # Reading Data
    path_of_video_tr, path_of_video_test = read_from_txt(args.train_txt, args.test_txt, args.data_dir)
    nframe_per_video_tr = get_nframe_video(path_of_video_tr[0], dataset=args.tr_dataset)
    nframe_per_video_ts = get_nframe_video(path_of_video_test[0], dataset=args.ts_dataset)

    path_of_video_tr = path_of_video_tr[:-1]
    path_of_video_test = path_of_video_test[:-1]
    print('sample path: ', path_of_video_tr[0])
    print('Trian Length: ', len(path_of_video_tr))
    print('Test Length: ', len(path_of_video_test))
    print('nframe_per_video_tr', nframe_per_video_tr)
    print('nframe_per_video_ts', nframe_per_video_ts)

    # %% Create data genener
    training_dataset = rPPG_Dataset(path_of_video_tr, args.tr_dataset, frame_depth=20, signal=args.signal)
    testing_dataset = rPPG_Dataset(path_of_video_test, args.ts_dataset, frame_depth=20, signal=args.signal)

    batch_size = args.batch_size
    tr_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    ts_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    model = TSCAN()
    # model.load_state_dict(torch.load('./checkpoints/train_AFRL_UBFC_test_MMSE/train_AFRL_UBFC_test_MMSE_23.pth'))
    #model.load_state_dict(torch.load('./checkpoints/train_AFRL_MMSE_test_UBFC/train_AFRL_MMSE_test_UBFC_23.pth'))
    #model.load_state_dict(torch.load('./checkpoints/Pretrained_AFRL_20_Test_AFRL_5/Pretrained_AFRL_20_Test_AFRL_5_23.pth'))
    model.load_state_dict(torch.load('./checkpoints/Pretrained_AFRL_20_Test_AFRL_5_resp/Pretrained_AFRL_20_Test_AFRL_5_resp_23.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise ValueError('Your training is not using GPU!')

    model = model.to(device)
    train_loss_freq = 100
    ##############################################################################################################
    for epoch in range(24):
        if args.eval_only == 0:
            print('Epoch: ', epoch)
            running_loss = 0.0
            tr_loss = 0.0
            for i, data in enumerate(tr_dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.view(-1, 6, 36, 36)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tr_loss += loss
                if i % train_loss_freq == (train_loss_freq-1):
                    print('[%d, %5d] tr loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / train_loss_freq))
                    running_loss = 0.0
            print('Average Training loss: ', tr_loss / len(tr_dataloader))
        ##############################################################################################################
        if epoch % 24 == 23:
            # Evaluation
            with torch.no_grad():
                print('Evaluate...')
                ts_loss = 0.0
                mae_total = 0.0
                rmse_total = 0.0
                snr_total = 0.0
                final_preds = np.array([])
                final_labels = np.array([])
                final_HR = np.array([])
                final_HR0 = np.array([])

                for i, ts_data in enumerate(ts_dataloader):
                    ts_inputs, ts_labels = ts_data[0].to(device), ts_data[1].to(device)
                    ts_inputs = ts_inputs.view(-1, 6, 36, 36)
                    ts_labels = ts_labels.view(-1, 1)
                    ts_outputs = model(ts_inputs)
                    loss = criterion(ts_outputs, ts_labels)
                    ts_loss += loss
                    ts_outputs_numpy = ts_outputs.cpu().numpy()
                    ts_labels_numpy = ts_labels.cpu().numpy()
                    mae_temp, rmse_temp, snr_temp, HR0, HR = \
                        calculate_metric(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                         window_size=args.window_size, fs=args.fs, bpFlag=True)
                    mae_total += mae_temp
                    rmse_total += rmse_temp
                    snr_total += snr_temp
                    if i == 0:
                        final_preds = ts_outputs_numpy
                        final_labels = ts_labels_numpy
                        final_HR = HR
                        final_HR0 = HR0
                    else:
                        final_preds = np.concatenate([final_preds, ts_outputs_numpy], axis=0)
                        final_labels = np.concatenate([final_labels, ts_labels_numpy], axis=0)
                        final_HR = np.concatenate([final_HR, HR], axis=0)
                        final_HR0 = np.concatenate([final_HR0, HR0], axis=0)
                print('Avg Validation Loss: ', ts_loss / len(ts_dataloader))
                print('Avg MAE across subjects: ', mae_total / len(ts_dataloader))
                print('Avg RMSE across subjects: ', rmse_total / len(ts_dataloader))
                print('Avg SNR across subjects: ', snr_total / len(ts_dataloader))
                model_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '.pth'))
                pred_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_pred'))
                label_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_label'))
                final_HR_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR_all'))
                final_HR0_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR0_all'))
                torch.save(model.state_dict(), model_path)
                np.save(pred_path, final_preds)
                np.save(label_path, final_labels)
                np.save(final_HR_path, final_HR)
                np.save(final_HR0_path, final_HR0)
                print('Pearson Results')
                print('Pearson: ', abs(np.corrcoef(final_HR, final_HR0)[1, 0]))
    print('Finished Training')


if __name__ == '__main__':
    train(args)
