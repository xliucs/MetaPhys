import torch
import h5py
import scipy.io
import numpy as np
from scipy.signal import butter

class rPPG_Dataset(torch.utils.data.Dataset):
    def __init__(self, task_data_path, dataset, frame_depth=10, fs=30, signal='pulse'):
        self.task_data_path = task_data_path
        self.dataset = dataset
        self.frame_depth = frame_depth
        self.fs = fs
        self.signal = signal

    def __len__(self):
        return len(self.task_data_path)

    def __getitem__(self, index):
        temp_path = self.task_data_path[index]
        if self.dataset == 'MAHNOB-HCI':
            f1 = scipy.io.loadmat(temp_path)
        else:
            f1 = h5py.File(temp_path, 'r')
        output = np.transpose(np.array(f1["dXsub"]), [3, 0, 2, 1])
        label = np.array(f1["dysub"])
        if 'AFRL' in temp_path:
            self.fs == 30
        elif 'MMSE' in temp_path:
            self.fs = 25
        else:
            self.fs = 30

        if self.signal == 'pulse':
            [b, a] = butter(1, [0.75 / self.fs * 2, 2.5 / self.fs * 2], btype='bandpass')
        else:
            label = np.array(f1["drsub"])
            [b, a] = butter(1, [0.08 / self.fs * 2, 0.5 / self.fs * 2], btype='bandpass')

        label = scipy.signal.filtfilt(b, a, np.squeeze(label))
        label = np.float32(np.expand_dims(label, axis=1))
        # Average the frame
        motion_data = output[:, :3, :, :]
        apperance_data = output[:, 3:, :, :]
        apperance_data = np.reshape(apperance_data, (int(180/self.frame_depth), self.frame_depth, 3, 36, 36))
        apperance_data = np.average(apperance_data, axis=1)
        apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
        apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                     apperance_data.shape[2], apperance_data.shape[3],
                                                     apperance_data.shape[4]))
        output = np.concatenate((motion_data, apperance_data), axis=1)

        return (output, label)

