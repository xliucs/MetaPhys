import glob
import logging
import os

import higher  # tested with higher v0.2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmeta.utils.data import BatchMetaDataLoader

from higher_model import TSCAN
from post_process import calculate_metric
from rppg_dataset import RPPG_DATASET
from utils import ToTensor1D, read_txt

torch.manual_seed(100)
np.random.seed(100)


logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('--folder', type=str, default='/gscratch/xxx/xxx/data3/mnt/',
                    help='Path to the folder the data is downloaded to.')
parser.add_argument('--dataset', type=str, default='AFRL',
                    help='Name of the dataset (default: omniglot).')
parser.add_argument('-o', '--save_dir', type=str, default='./checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('--task_type', type=str,
                    choices=['person', 'task'], default='person',
                    help='task independent training or subject independent training')
parser.add_argument('--num-shots', type=int, default=6,
                    help='Number of examples per class (k in "k-shot", default: 5).')
parser.add_argument('--num-test-shots', type=int, default=6,
                    help='Number of test examples per class (k in "k-shot", default: 5).')
parser.add_argument('--inner-step-size', type=float, default=0.1,
                    help='Step-size for the gradient step for adaptation (default: 0.4).')
parser.add_argument('--output-folder', type=str, default=None,
                    help='Path to the output folder for saving the model (optional).')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of tasks in a mini-batch of tasks (default: 16).')
parser.add_argument('--eval_batch_size', type=int, default=20,
                    help='Number of files of 180 frames loaded during evaluation test (default: 25).')
parser.add_argument('--num-adapt-steps', type=int, default=1,
                    help='Number of fast adaptation steps, ie. gradient descent '
                         'updates (default: 1).')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of epochs of meta-training (default: 50).')
parser.add_argument('--num-batches', type=int, default=100,
                    help='Number of batches the model is trained over (default: 100).')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of workers for data loading (default: 1).')
parser.add_argument('--pre-trained', type=int, default=1,
                            help='pretrained or not')
parser.add_argument('--freeze', type=int, default=0,
                            help='freeze or not')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA if available.')
parser.add_argument('--tr-fs', type=int, default=30,
                    help='sampling rate of your dataset')
parser.add_argument('--ts-fs', type=int, default=30,
                            help='sampling rate of your dataset')
parser.add_argument('--window-size', type=int, default=360,
                                    help='window size for filtering and FFT')
parser.add_argument('--signal', type=str, default='pulse',
                                    help='window size for filtering and FFT')
parser.add_argument('--unsupervised', type=int, default=0,
                            help='unsupervised')

args = parser.parse_args()
args.device = torch.device('cuda' if args.use_cuda
                                     and torch.cuda.is_available() else 'cpu')


def main():

    checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    model = TSCAN()
    if args.pre_trained == 1:
        # Loading pre-trained model and freezing motion convs
        print('Using pre-trained on all ALL AFRL!')
        model.load_state_dict(torch.load('./checkpoints/AFRL_pretrained/meta_pretrained_all_AFRL.pth'))
    elif args.pre_trained == 2:
        print('Using pre-trained on all 15 subjs in AFRL!')
        model.load_state_dict(torch.load('./checkpoints/AFRL_pretrained/pre_trained_AFRL_15subj_23.pth'))
    elif args.pre_trained == 3:
        print('Using filtered pre-trained on all ALL AFRL!')
        model.load_state_dict(torch.load('./checkpoints/AFRL_pretrained/AFRL_full_filtered_23.pth'))
    elif args.pre_trained == 4:
        print('Using pre-trained on first 10 subjs in AFRL')
        model.load_state_dict(torch.load('./checkpoints/Pretrained_AFRL_10/Pretrained_AFRL_10_23.pth'))
    elif args.pre_trained == 5:
        print('Using pre-trained on first 10 subjs in AFRL')
        model.load_state_dict(torch.load('./checkpoints/Pretrained_AFRL_10_resp/Pretrained_AFRL_10_resp_23.pth'))
    else:
        print('Not using any pretrained models!')

    if args.freeze == 1:
        print('Freezing the motion branch!')
        model.motion_conv1.weight.requires_grad = False
        model.motion_conv1.bias.requires_grad = False
        model.motion_conv2.weight.requires_grad = False
        model.motion_conv2.bias.requires_grad = False
        model.motion_conv3.weight.requires_grad = False
        model.motion_conv3.bias.requires_grad = False
        model.motion_conv4.weight.requires_grad = False
        model.motion_conv4.bias.requires_grad = False

    model.to(device=args.device)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    transform = ToTensor1D()
    AFRL_filelists_path = sorted(glob.glob('./meta_filelists/' + 'AFRL' + '/person/*.txt'))
    UBFC_filelists_path = sorted(glob.glob('./meta_filelists/' + 'UBFC' + '/person/*.txt'))
    UBFC_filelists_path_unsupervised = sorted(glob.glob('./meta_filelists/' + 'UBFC' + '/person_unsupervised/*.txt'))
    MMSE_filelists_path_M = sorted(glob.glob('./meta_filelists/' + 'MMSE' + '/person/male/*.txt'))
    MMSE_filelists_path_F = sorted(glob.glob('./meta_filelists/' + 'MMSE' + '/person/female/*.txt'))
    MMSE_filelists_path_M_unsupervised = sorted(glob.glob('./meta_filelists/' + 'MMSE' + '/person_unsupervised/male/*.txt'))
    MMSE_filelists_path_F_unsupervised = sorted(glob.glob('./meta_filelists/' + 'MMSE' + '/person_unsupervised/female/*.txt'))

    if args.dataset == 'AFRL':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path[10:20]  # number of subject (task)
            test_list = AFRL_filelists_path[20:]  # number of subject (task)
        else:
            raise ValueError('AFRL Task traing not ready.')
    elif args.dataset == 'UBFC':
        if args.task_type == 'person':
            train_list = UBFC_filelists_path[:20]  # number of subject (task)
            test_list = UBFC_filelists_path[20:]  # number of subject (task)
        else:
            raise ValueError('UBFC Task traing not ready.')
    elif args.dataset == 'MMSE':
        if args.task_type == 'person':
            train_list = MMSE_filelists_path_M[:10] + MMSE_filelists_path_F[:10]  # number of subject (task)
            test_list = MMSE_filelists_path_M[10:] + MMSE_filelists_path_F[10:]  # number of subject (task)
        else:
            raise ValueError('UBFC Task traing not ready.')
    elif args.dataset == 'Meta_AFRL_UBFC_All':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path
            test_list = UBFC_filelists_path
    elif args.dataset == 'Meta_AFRL_MMSE_All':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path
            test_list = MMSE_filelists_path_M + MMSE_filelists_path_F
    elif args.dataset == 'Meta_AFRL_UBFC':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path + UBFC_filelists_path[:20]
            test_list = UBFC_filelists_path[20:]
    elif args.dataset == 'Meta_AFRL_UBFC_cv':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path
            test_list = UBFC_filelists_path[21:]
    elif args.dataset == 'Meta_AFRL_MMSE':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path + MMSE_filelists_path_M[:10] + MMSE_filelists_path_F[:10]
            test_list = MMSE_filelists_path_M[10:] + MMSE_filelists_path_F[10:]
    elif args.dataset == 'Meta_10_AFRL_UBFC':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path[15:]
            test_list = UBFC_filelists_path
    elif args.dataset == 'Meta_10_AFRL_MMSE':
        if args.task_type == 'person':
            train_list = AFRL_filelists_path[15:]
            test_list = MMSE_filelists_path_M + MMSE_filelists_path_F
    elif args.dataset == 'Pre_AFRL_Meta_MMSE_MTest_UBFC':
        train_list = MMSE_filelists_path_M + MMSE_filelists_path_F
        test_list = UBFC_filelists_path
    elif args.dataset == 'Pre_AFRL_Meta_MMSE_MTest_UBFC_unsupervised':
        train_list = MMSE_filelists_path_M_unsupervised + MMSE_filelists_path_F_unsupervised
        test_list = UBFC_filelists_path_unsupervised
    elif args.dataset == 'Pre_AFRL_Meta_UBFC_MTest_MMSE':
        train_list = UBFC_filelists_path
        test_list = MMSE_filelists_path_M + MMSE_filelists_path_F
    elif args.dataset == 'Pre_AFRL_Meta_UBFC_MTest_MMSE_unsupervised':
        train_list = UBFC_filelists_path_unsupervised
        test_list = MMSE_filelists_path_M_unsupervised + MMSE_filelists_path_F_unsupervised
    elif args.dataset == 'MTrain_AFRL_UBFC_MTest_MMSE':
        train_list = AFRL_filelists_path + UBFC_filelists_path
        test_list = MMSE_filelists_path_M + MMSE_filelists_path_F
    elif args.dataset == 'MTrain_AFRL_MMSE_MTest_UBFC':
        train_list = AFRL_filelists_path + MMSE_filelists_path_M + MMSE_filelists_path_F
        test_list = UBFC_filelists_path
    elif args.dataset == 'Pre_AFRL_10_MTrain_AFRL_10_MTest_AFRL_5':
        train_list = AFRL_filelists_path[10:20]
        test_list = AFRL_filelists_path[20:]
    elif args.dataset == 'MTrain_AFRL_20_MTest_AFRL_5':
        train_list = AFRL_filelists_path[:20]
        test_list = AFRL_filelists_path[20:]
    else:
        raise ValueError('Dataset is not supported!')

    train_path = read_txt(train_list, args.folder)
    test_path = read_txt(test_list, args.folder)

    meta_train_dataset = RPPG_DATASET(args.dataset,
                                      args.num_shots,
                                      args.num_test_shots,
                                      train_path,
                                      num_tasks=len(train_list),
                                      state='train',
                                      transform=transform,
                                      target_transform=transform,
                                      sample_type='task',
                                      frame_depth=20,
                                      fs=args.tr_fs,
                                      signal=args.signal,
                                      unsupervised=args.unsupervised)

    train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers)

    meta_test_dataset = RPPG_DATASET(args.dataset,
                                     args.num_shots,
                                     args.num_test_shots,  # This will be updated to the remaining samples during eval
                                     test_path,
                                     num_tasks=len(test_list),
                                     state='test',
                                     transform=transform,
                                     target_transform=transform,
                                     sample_type='task',
                                     frame_depth=20,
                                     fs=args.ts_fs,
                                     signal=args.signal,
                                     unsupervised=args.unsupervised)

    test_dataloader = BatchMetaDataLoader(meta_test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers)

    ##############################################################################################################
    # Training Loop

    for i in range(args.num_epochs):
        print('Epoch: ', i)
        train(args, model, meta_optimizer, train_dataloader)
        print('Finish training!')
        if i % 1 == 0:
            model, preds, labels, final_HR0, final_HR = test(args, model, test_dataloader, checkpoint_folder, epoch=i)
            print('Finish Eval')
            if (i+1) == args.num_epochs:
                print('Saving the final outputs from the last epoch!')
                pred_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(i) + '_pred_all'))
                label_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(i) + '_label_all'))
                final_HR_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(i) + '_HR_all'))
                final_HR0_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(i) + '_HR0_all'))
                np.save(pred_path, preds)
                np.save(label_path, labels)
                np.save(final_HR_path, final_HR)
                np.save(final_HR0_path, final_HR0)
                print('Pearson Results')
                print('Pearson: ', abs(np.corrcoef(final_HR, final_HR0)[1, 0]))
        print('====================================')

##############################################################################################################

def train(args, model, meta_optimizer, dataloader):
    # Training loop
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        model.zero_grad()

        batch['train'][0] = batch['train'][0].view(args.batch_size, -1, 6, 36, 36)
        batch['test'][0] = batch['test'][0].view(args.batch_size, -1, 6, 36, 36)
        batch['train'][1] = batch['train'][1].view(args.batch_size, -1, 1)
        batch['test'][1] = batch['test'][1].view(args.batch_size,-1, 1)

        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device=args.device)
        test_targets = test_targets.to(device=args.device)

        inner_optimiser = torch.optim.SGD(model.parameters(), lr=args.inner_step_size)
        for task_idx, (train_input, train_target, test_input, test_target) \
                in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(model, inner_optimiser, copy_initial_weights=False) as (fmodel, diffopt):
                for step in range(args.num_adapt_steps):
                    train_logit = fmodel(train_input)
                    inner_loss = F.mse_loss(train_logit, train_target)
                    diffopt.step(inner_loss)
                test_logit = fmodel(test_input)
                outer_loss = F.mse_loss(test_logit, test_target)
                outer_loss.backward()
        meta_optimizer.step()

##############################################################################################################

def test(args, model, dataloader, checkpoint_folder, epoch):

    # Training loop
    model.train()
    mean_test_loss = torch.tensor(0., device=args.device)
    mae_total = 0.0
    rmse_total = 0.0
    snr_total = 0.0
    final_preds = np.array([])
    final_labels = np.array([])
    final_HR = np.array([])
    final_HR0 = np.array([])
    for batch_idx, batch in enumerate(dataloader):
        batch['train'][0] = batch['train'][0].view(args.batch_size, -1, 6, 36, 36)
        batch['test'][0] = batch['test'][0].view(args.batch_size, -1, 6, 36, 36)
        batch['train'][1] = batch['train'][1].view(args.batch_size, -1, 1)
        batch['test'][1] = batch['test'][1].view(args.batch_size,-1, 1)

        train_inputs, train_targets = batch['train']

        train_inputs = train_inputs.to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        test_inputs, test_targets = batch['test']

        inner_optimiser = torch.optim.SGD(model.parameters(), lr=args.inner_step_size)

        for task_idx, (train_input, train_target, test_input, test_target) \
                in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(model, inner_optimiser, copy_initial_weights=False) as (fmodel, diffopt):
                for step in range(args.num_adapt_steps):
                    train_logit = fmodel(train_input)
                    inner_loss = F.mse_loss(train_logit, train_target)
                    diffopt.step(inner_loss)
                test_data_loader = DataLoader(test_input, batch_size=args.eval_batch_size, shuffle=False)
                test_logits = torch.tensor([], device=args.device)
                for i, test_batch in enumerate(test_data_loader):
                    pred = fmodel(test_batch.to(device=args.device)).detach()
                    test_logits = torch.cat((test_logits, pred), 0)
                temp_test_loss = F.mse_loss(test_logits, test_target.to(device=args.device))
                ts_outputs_numpy = test_logits.cpu().numpy()
                ts_labels_numpy = test_target.cpu().numpy()
                # calculate final metric
                mae_temp, rmse_temp, snr_temp, HR0, HR = \
                    calculate_metric(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                     window_size=args.window_size, fs=args.ts_fs, bpFlag=True)
                mean_test_loss += temp_test_loss
                mae_total += mae_temp
                rmse_total += rmse_temp
                snr_total += snr_temp
                # Saving the final label
                if batch_idx == 0:
                    final_preds = ts_outputs_numpy
                    final_labels = ts_labels_numpy
                    final_HR = HR
                    final_HR0 = HR0
                else:
                    final_preds = np.concatenate([final_preds, ts_outputs_numpy], axis=0)
                    final_labels = np.concatenate([final_labels, ts_labels_numpy], axis=0)
                    final_HR = np.concatenate([final_HR, HR], axis=0)
                    final_HR0 = np.concatenate([final_HR0, HR0], axis=0)

                model_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) +
                                              '_person' + str(batch_idx) + '.pth'))
                torch.save(model.state_dict(), model_path)
                pred_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + 'person_' + str(batch_idx) +
                                             '_pred'))
                label_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + 'person_' + str(batch_idx) +
                                              '_label'))
                np.save(pred_path, ts_outputs_numpy)
                np.save(label_path, ts_labels_numpy)

    print('length of dataloaader: ', len(dataloader))
    print('mean test_loss: ',  mean_test_loss.div_(len(dataloader)))
    print('Avg MAE across subjects: ', mae_total / len(dataloader))
    print('Avg RMSE across subjects: ', rmse_total / len(dataloader))
    print('Avg SNR across subjects: ', snr_total / len(dataloader))
    del mean_test_loss
    return model, final_preds, final_labels, final_HR0, final_HR


if __name__ == '__main__':
    main()



