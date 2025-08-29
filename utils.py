import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix

from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    # transforms.Compose([...]) creates a pipeline of transformations to apply to images. In this case, there is only one transformation transforms.ToTensor()
    # transforms.ToTensor() converts a PIL image or a NumPy array into a PyTorch tensor

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform) 

    # xray_train_ds.samples -> list of (image_path, class_index) tuples
    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples]) 

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    # A dictionary mapping each network/client (net_i) to another dictionary that maps class labels to their counts in that client's data
    # net_cls_counts = {
    #     0: {0: 10, 1: 15, 2: 5},
    #     1: {0: 8, 1: 12, 2: 10},
    #     ...
    # }
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        # y_train[dataidx] = [0, 1, 1, 2] -> unq = [0, 1, 2]
        # For the above, unq_cnt = [1, 2, 1]
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True) # get unique class labels and their counts
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))} # create a dictionary mapping class labels to counts
        net_cls_counts[net_i] = tmp # store the class counts for the current network

    data_list=[] # list to hold total counts of data for each network
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train) # Randomly shuffle the training data indices
        # idxs = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # n_parties = 3

        # batch_idxs = np.array_split(idxs, n_parties)
        # print(batch_idxs) 
        # [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
        batch_idxs = np.array_split(idxs, n_parties) # Split the shuffled indices into n_parties parts
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)} # Create a mapping of each party to its data indices


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0] # Total number of training samples
        net_dataidx_map = {} # Dictionary to hold the data indices for each party

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)] # Initialize a list of empty lists for each party
            for k in range(K):
                idx_k = np.where(y_train == k)[0] # Get the indices of samples belonging to class k
                np.random.shuffle(idx_k) # Randomly shuffle the indices of class k samples

                proportions = np.random.dirichlet(np.repeat(beta, n_parties)) # Generate random proportions for each party
                # Example of proportions: [0.2, 0.3, 0.5] for 3 parties

                # proportions: a list of values (one per client/party) sampled from a Dirichlet distribution, representing how much of the current class should go to each client
                # idx_batch: a list of lists, where each sublist contains the indices already assigned to each client
                # This condition checks if the current client has fewer samples than the average (N / n_parties). If true, it returns p (so the proportion is kept); if false, it returns 0 (so the proportion is set to zero)
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)]) # Clients that already have at least their "fair share" of data will not receive more samples for this class

                proportions = proportions / proportions.sum() # Normalize the proportions to sum to 1
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1] # Compute cumulative proportions to determine how many samples each party should get
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] # Split the indices of class k samples among the parties based on the computed proportions
                min_size = min([len(idx_j) for idx_j in idx_batch]) # Check the minimum size of data across all parties
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j]) # 
            net_dataidx_map[j] = idx_batch[j]

    n_test = y_test.shape[0] # Number of test samples
    test_idxs = np.random.permutation(n_test) # Randomly shuffle the test data indices
    test_batch_idxs = np.array_split(test_idxs, n_parties) # Split the test data indices among the parties
    net_test_dataidx_map = {i: test_batch_idxs[i] for i in range(n_parties)} # Map each party to its test data indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, net_test_dataidx_map)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    """
    This function extracts all trainable parameters (weights and biases that require gradients) from a PyTorch model (net) and flattens them into a single 1D tensor (vector)
    """
    trainable = filter(lambda p: p.requires_grad, net.parameters()) # Get only the parameters that require gradients (i.e., trainable parameters)
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel() # Calculate the total number of elements in all trainable parameters
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device) # Create an empty tensor to hold the trainable parameters
    X.fill_(0.0) # Initialize the tensor with zeros
    offset = 0 # Offset to keep track of where to place the next parameter values in the tensor
    for params in paramlist: # Iterate through each trainable parameter
        numel = params.numel() # Get the number of elements in the current parameter
        with torch.no_grad(): # Disable gradient tracking for this operation
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data)) # Copy the parameter values into the tensor
        offset += numel # Update the offset to point to the next position in the tensor
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    """
    This function takes a flat 1D tensor X (containing all trainable parameters) and copies its values back into the corresponding parameters of the PyTorch model net
    """
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    # was_training = False # If the model is in training mode, switch it to evaluation mode
    # if model.training: # Check if the model is currently in training mode
    #     model.eval() # Switch the model to evaluation mode
    #     was_training = True # Set a flag to remember that we switched the model to evaluation mode
    training_mode = model.training
    model.eval()  # Always switch to eval mode for inference

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0 # Initialize counters for correct predictions and total samples
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss() # Define the loss function for CPU
    elif "cuda" in device.type: 
        criterion = nn.CrossEntropyLoss().cuda() # Define the loss function for GPU
    loss_collector = [] # Initialize a list to collect loss values
    if multiloader: 
        for loader in dataloader: # Iterate through each data loader
            with torch.no_grad(): # Disable gradient tracking for the evaluation phase
                for batch_idx, (x, target) in enumerate(loader): # Iterate through each batch in the data loader
                    # x: input batch
                    # target: label batch
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda() # Move the input data and target labels to the appropriate device (GPU or CPU)
                    _, _, out = model(x) # Forward pass through the model to get the output predictions
                    if len(target)==1: # If the target has only one element, it is treated as a scalar
                        loss = criterion(out, target)  # Compute the loss using the output predictions and target labels
                    else:
                        loss = criterion(out, target) # Compute the loss using the output predictions and target labels
                    _, pred_label = torch.max(out.data, 1) # Get the predicted labels by taking the index of the maximum value in the output predictions
                    loss_collector.append(loss.item()) # Append the loss value to the loss collector
                    total += x.data.size()[0] # Update the total number of samples processed
                    correct += (pred_label == target.data).sum().item() # Count the number of correct predictions by comparing predicted labels with target labels

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy()) # Append predicted labels to the list
                        true_labels_list = np.append(true_labels_list, target.data.numpy()) # Append true labels to the list
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy()) # Append predicted labels to the list (for GPU)
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy()) # Append true labels to the list (for GPU)
        avg_loss = sum(loss_collector) / len(loss_collector) 
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda() # Move the input data and target labels to the appropriate device (GPU or CPU)
                _,_,out = model(x) # Forward pass through the model to get the output predictions
                loss = criterion(out, target) # Compute the loss using the output predictions and target labels
                _, pred_label = torch.max(out.data, 1) # Get the predicted labels by taking the index of the maximum value in the output predictions
                loss_collector.append(loss.item()) # Append the loss value to the loss collector
                total += x.data.size()[0] # Update the total number of samples processed
                correct += (pred_label == target.data).sum().item() # Count the number of correct predictions by comparing predicted labels with target labels

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list) # Compute the confusion matrix using true and predicted labels

    # if was_training:
    #     model.train() # Switch the model back to training mode if it was originally in training mode

    # Restore the original mode
    if training_mode:
        model.train()
    else:
        model.eval()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss 

    return correct / float(total), avg_loss 

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_: # Save the model state dictionary to a file
        torch.save(model.state_dict(), f_) # Save the model's state dictionary (weights and biases) to the specified file
    return


def load_model(model, model_index, device="cpu"):

    with open("trained_local_model" + str(model_index), "rb") as f_: # Load the model state dictionary from a file
        model.load_state_dict(torch.load(f_)) # Load the model's state dictionary (weights and biases) from the specified file
    if device == "cpu":
        model.to(device) # Move the model to the CPU
    else:
        model.cuda() # Move the model to the specified device (GPU)
    return model

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, test_dataidxs=None,noise_level=0):
    if dataset in ('cifar10', 'cifar100'): # sets up normalization and data augmentation (random crop, flip, rotation, etc.) for training, and just normalization for testing
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]) # Normalize the CIFAR-10 dataset images
            transform_train = transforms.Compose([ # Define the transformations for the training set
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([ # Define the transformations for the test set
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100': # uses a different normalization
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]) # Normalize the CIFAR-100 dataset images
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([ # Define the transformations for the training set
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([ # Define the transformations for the test set
                transforms.ToTensor(),
                normalize])



        # Initialize datasets
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, dataidxs=test_dataidxs, train=False, transform=transform_test, download=True)
        
        # Not using dataidx for the test set -> every client's test set is the full global test set, not a client-specific partition
        # test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        # Create iterable objects that efficiently load batches of data from the training and test datasets, respectively
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        # data prep for train set
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'./train/', dataidxs=dataidxs, transform=transform_train) 
        test_ds = dl_obj(datadir+'./val/', transform=transform_test) 

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True) # DataLoader for training set
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False) # DataLoader for test set


    return train_dl, test_dl, train_ds, test_ds
