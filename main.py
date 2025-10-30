import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import math


from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)} # initialize nets dict
    # nets is a dictionary where keys are party indices and values are the corresponding neural networks
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model: # use normal model
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else: # use aggregate model
        for net_i in range(n_parties):
            if args.use_project_head: # use projection head
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs) # ModelFedCon with projection head
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs) # ModelFedCon without projection head
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = [] # to store the shape (dimensions) of each layer in the model
    layer_type = [] # to store the names (types) of each layer in the model
    for (k, v) in nets[0].state_dict().items(): # iterate through the first net's state_dict
        # k: layer name, v: layer parameters
        model_meta_data.append(v.shape) # append the shape of each layer to model_meta_data
        layer_type.append(k) # append the layer name to layer_type

    return nets, model_meta_data, layer_type

# Train function for FedAvg algorithm
def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = nn.DataParallel(net) # wrap the model in DataParallel for multi-GPU training
    net.cuda() 
    logger.info('Training network %s' % str(net_id)) 
    logger.info('n_training: %d' % len(train_dataloader)) # log the number of training samples
    logger.info('n_test: %d' % len(test_dataloader)) # log the number of training and test samples

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda() 

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = [] # to collect loss values for each epoch
        for batch_idx, (x, target) in enumerate(train_dataloader): # iterate through the training dataloader
            x, target = x.cuda(), target.cuda() 
            # x: input data, target: ground truth labels

            optimizer.zero_grad() # zero the gradients
            x.requires_grad = False # disable gradient computation for input
            target.requires_grad = False # disable gradient computation for target
            target = target.long() # convert target to long type

            _,_,out = net(x) # forward pass through the model
            loss = criterion(out, target) # compute the cross-entropy loss
            # loss is something look like tensor(2.3026, device='cuda:0')

            loss.backward() # backpropagate the loss
            optimizer.step() # update the model parameters

            cnt += 1 # increment the counter
            epoch_loss_collector.append(loss.item()) # append the loss value to epoch_loss_collector
            # loss.item() is a Python float representing the loss value

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector) # compute the average loss for the epoch
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss)) # log the epoch number and loss value

        if epoch % 10 == 0: # log training and test accuracy every 10 epochs
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device) # _ avg loss
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc

# Train function for FedProx algorithm
def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters()) # collect global model parameters

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0 # initialize the fedprox regularization term
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()): # iterate through the model parameters
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2) # compute the fedprox regularization term
            loss += fed_prox_reg # add the fedprox regularization term to the loss

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg) # default

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets: 
        previous_net.cuda() # move previous nets to GPU
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1) # cosine similarity function
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = [] 
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad() # zero the gradients
            x.requires_grad = False 
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x) # forward pass through the local model
            # pro1: projected features of input from local model (anchor)
            # out: logits for classification

            _, pro2, _ = global_net(x)
            # pro2: projected features of input from global model

            posi = cos(pro1, pro2) # positive pair
            logits = posi.reshape(-1,1) # reshape the positive cosine similarity to a column vector

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x) 
                # pro3: projected features of input from previous local model
                nega = cos(pro1, pro3) # negative pair
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1) # concatenate the logits of positive and negative pairs horizontally (add more columns)
                # First column: positive similarity (with global model)
                # Remaining columns: negative similarities (with previous models)
                # You want each row to be one sample
                # And each column to be one comparison (positive and negatives)
                # So dim=1 makes each row contain all logits relevant to that sample — perfect for CrossEntropyLoss, which expects that format

                previous_net.to('cpu')

            logits /= temperature # scale the logits by the temperature parameter
            labels = torch.zeros(x.size(0)).cuda().long() # create labels for the logits

            # This performs: -log( exp(sim(z, z_glob)/τ) / (exp(sim(z, z_glob)/τ) + ∑ exp(sim(z, z_prev)/τ)) )
            loss2 = mu * criterion(logits, labels) # compute the contrastive loss

            loss1 = criterion(out, target) # compute the cross-entropy loss
            loss = loss1 + loss2
 
            loss.backward() # backpropagate the loss
            optimizer.step() # update the model parameters

            cnt += 1
            epoch_loss_collector.append(loss.item()) # total loss
            epoch_loss1_collector.append(loss1.item()) # cross-entropy loss
            epoch_loss2_collector.append(loss2.item()) # contrastive loss

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector) 
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector) 
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector) 
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_my(net_id, net, global_net, train_dataloader, test_dataloader_local, test_dataloader_global, epochs, lr, args_optimizer, mu, temperature, args, round, device="cpu"):
    # test_dataloader_local -> local client test set (32 each client)
    # test_dataloader_global -> global test set (313)
    
    net = nn.DataParallel(net) 
    net.cuda() 
    
    logger.info('Training network %s' % str(net_id)) 
    logger.info('n_training: %d' % len(train_dataloader)) 
    logger.info('n_test: %d' % len(test_dataloader_local))

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader_global, get_confusion_matrix=True, device=device)

    # Compute the test accuracy of the global model on local client test set
    if global_net is not None:
        with torch.no_grad():
            global_test_acc, _, _ = compute_accuracy(
                global_net, test_dataloader_local,
                get_confusion_matrix=True, device=device
            )
    else:
        global_test_acc = None

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc)) # local model on global test set
    logger.info('>> Pre-Training Global Test accuracy: {}'.format(global_test_acc)) # global model on local clients test set

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda() 

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = [] # to collect loss values for each epoch
        for batch_idx, (x, target) in enumerate(train_dataloader): # iterate through the training dataloader
            x, target = x.cuda(), target.cuda() 
            # x: input data, target: ground truth labels

            optimizer.zero_grad() # zero the gradients
            x.requires_grad = False # disable gradient computation for input
            target.requires_grad = False # disable gradient computation for target
            target = target.long() # convert target to long type

            _,_,out = net(x) # forward pass through the model
            loss = criterion(out, target) # compute the cross-entropy loss
            # loss is something look like tensor(2.3026, device='cuda:0')

            loss.backward() # backpropagate the loss
            optimizer.step() # update the model parameters

            cnt += 1 # increment the counter
            epoch_loss_collector.append(loss.item()) # append the loss value to epoch_loss_collector
            # loss.item() is a Python float representing the loss value

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector) # compute the average loss for the epoch
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss)) # log the epoch number and loss value

        if epoch % 10 == 0: # log training and test accuracy every 10 epochs
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader_global, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device) # _ avg loss
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader_global, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    
    return train_acc, test_acc, global_test_acc
    

# Client-side
def local_train_net(nets, args, net_dataidx_map, net_test_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu"):
    # test_dl -> global
    avg_acc = 0.0
    acc_list = []
    global_testacc_list = {}
    if global_model:
        global_model.cuda() 
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters()) # collect server_c parameters
        new_server_c_collector = copy.deepcopy(server_c_collector) # create a copy of server_c parameters
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id] # get the data indices for the current network

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        # train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # Get local dataloaders
        train_dl_local, test_dl_local, _, _ = get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32,
            dataidxs=net_dataidx_map[net_id],
            test_dataidxs=net_test_dataidx_map[net_id]
        )

        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id]) # collect previous models
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)

        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        elif args.alg == 'my_fedcon':
            trainacc, testacc, global_testacc = train_net_my(net_id, net, global_model, train_dl_local, test_dl_local, test_dl, n_epoch, args.lr,
                                                args.optimizer, args.mu, args.temperature, args, round, device=device)
            global_testacc_list[net_id] = global_testacc # store the global test accuracy for each net_id
            # trainacc, testacc = train_net_my(net_id, net, global_model, train_dl_local, test_dl_local, test_dl, n_epoch, args.lr,
            #                                     args.optimizer, args.mu, args.temperature, args, round, device=device)
            
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    if args.alg == 'my_fedcon':
        return nets, global_testacc_list
    else:
        return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, net_test_dataidx_map = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta) # partition the data
    # net_dataidx_map is a dictionary mapping each party to its data indices

    n_party_per_round = int(args.n_parties * args.sample_fraction) # number of parties sampled in each round
    party_list = [i for i in range(args.n_parties)] # list of parties
    party_list_rounds = [] # to store the parties sampled in each round
    # party_list_rounds = [
    #     [0, 2, 3],   # round 1: sampled clients
    #     [1, 2, 4],   # round 2: sampled clients
    #     [0, 3, 4]    # round 3: sampled clients
    # ]
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round)) # sample n_party_per_round parties for each round
    else: 
        for i in range(args.comm_round): 
            party_list_rounds.append(party_list) # if n_party_per_round equals total number of parties, sample all parties

    n_classes = len(np.unique(y_train)) # number of classes in the training data

    # Get the dataloaders for global training and testing datasets
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global)) # length of the global training dataset
    train_dl=None # train_dataloader for local training
    data_size = len(test_ds_global) # size of the test dataset

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual': # if a model file is provided, load the model
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum: # initialize server momentum
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    
    # Server-side
    if args.alg == 'moon':
        old_nets_pool = [] # buffer for previous local models (used as negatives)
        if args.load_pool_file: # Option A: Load model pool from file (optional)
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net: # Option B: Use current nets as the first "old" pool (if not loading from file)
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round] # get the parties sampled in this round (party_list_this_round: an array of participating parties indices)

            global_model.eval()
            for param in global_model.parameters(): # set requires_grad to False for global model
                param.requires_grad = False 
            global_w = global_model.state_dict() # get the state_dict of the global model

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict()) # create a copy of the global model state_dict for momentum update

            nets_this_round = {k: nets[k] for k in party_list_this_round} # get the nets for the sampled parties in this round
            
            # Step 1: Send current global model to clients
            for net in nets_this_round.values():
                net.load_state_dict(global_w) # load the global model state_dict into the nets for this round

            # Step 2: Local training with MOON contrastive loss
            # train_dl not used here
            # test_dl is the global test set -> 313
            local_train_net(nets_this_round, args, net_dataidx_map, net_test_dataidx_map, train_dl=train_dl, test_dl=test_dl_global, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round]) # total number of data points across the sampled parties
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round] # compute the frequency of each party's data points

            # Step 3: Aggregate local models
            for net_id, net in enumerate(nets_this_round.values()): 
                net_para = net.state_dict() 
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id] # initialize the global model with the first net's parameters weighted by its frequency
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id] # aggregate the parameters of the nets weighted by their frequencies

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w) # create a copy of the global model state_dict for momentum update
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key] # compute the change in parameters
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key] # update the momentum
                    global_w[key] = old_w[key] - moment_v[key] # update the global model parameters with momentum

            global_model.load_state_dict(global_w) # load the updated global model state_dict
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            # Step 4: Update model pool (used in next round's contrastive loss)
            # Each round, the oldest model gets removed, and the newest one is added to the end
            # model_buffer_size = 3
            # Round 7 ➜ Pool: [model4, model5, model6]
            # Round 8 ➜ Pool: [model5, model6, model7]
            # Round 9 ➜ Pool: [model6, model7, model8]
            
            if len(old_nets_pool) < args.model_buffer_size: # Case 1: If the pool is not full yet
                old_nets = copy.deepcopy(nets) 
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False # freeze the parameters so they won’t be accidentally updated
                old_nets_pool.append(old_nets) # append the current nets to the old_nets_pool
            elif args.pool_option == 'FIFO': # Case 2: If the pool is already full
                old_nets = copy.deepcopy(nets) 
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1] # shift the old nets in the pool
                old_nets_pool[args.model_buffer_size - 1] = old_nets # replace the last old net with the current nets


            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth') # save the global model state_dict
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth') # save the first net's state_dict
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth') # save the old nets' state_dicts in the pool


    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, net_test_dataidx_map, train_dl=train_dl, test_dl=test_dl_global, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, net_test_dataidx_map, train_dl=train_dl, test_dl=test_dl_global, global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'my_fedcon':
        accuracy_above_average_model_pool = [] # buffer for accuracy above average models
        accuracy_below_average_model_pool = [] # buffer for accuracy below average models
        global_testacc_list = {} # buffer for global model accuracy

        for round in range(n_comm_rounds):

            accuracy_above_average_model_pool.clear()
            accuracy_below_average_model_pool.clear()
            
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}

            # Step 1: Send current global model to clients
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # Step 2: Local training
            _, global_testacc_list = local_train_net(nets_this_round, args, net_dataidx_map, net_test_dataidx_map, train_dl=train_dl, test_dl=test_dl_global, global_model=global_model, device=device)
            global_average_testacc = sum(global_testacc_list.values()) / len(global_testacc_list)

            for net_id, test_acc in global_testacc_list.items():
                logger.info('>> Global Model Test accuracy for net %d: %f' % (net_id, test_acc))

            logger.info('>> Global Model Average Test accuracy: %f' % global_average_testacc)


            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

             # Step 3: Aggregate local models
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> After local training (before contrastive learning):')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            # Step 4: Partition the models into above and below average accuracy pools
            for net_id, net in nets_this_round.items():
                if global_testacc_list[net_id] >= global_average_testacc:
                    accuracy_above_average_model_pool.append(net) # positive samples
                else:
                    accuracy_below_average_model_pool.append(net) # negative samples

            # Check for empty pools and handle accordingly
            if len(accuracy_above_average_model_pool) == 0:
                logger.warning("No above-average models this round, using best model as positive sample.")
                best_net_id = max(global_testacc_list, key=global_testacc_list.get)
                accuracy_above_average_model_pool.append(nets_this_round[best_net_id])

            logger.info("Above-average model IDs: %s" % [id for id, net in nets_this_round.items() if net in accuracy_above_average_model_pool])

            if len(accuracy_below_average_model_pool) == 0:
                logger.warning("No below-average models this round, using worst model as negative sample.")
                worst_net_id = min(global_testacc_list, key=global_testacc_list.get)
                accuracy_below_average_model_pool.append(nets_this_round[worst_net_id])

            logger.info("Below-average model IDs: %s" % [id for id, net in nets_this_round.items() if net in accuracy_below_average_model_pool])

            # Step 5: Leverage accuracy_above_average_model_pool as positive samples and accuracy_below_average_model_pool as negative samples for contrastive loss

            import math

            base_lr = 0.000125

            # Cosine decay: starts at base_lr, goes smoothly to 0
            t = round / max(1, n_comm_rounds - 1)
            scaled_lr = 0.5 * base_lr * (1 - math.cos(2 * math.pi * t))

            logger.info(f"Contrastive learning rate (round {round}): {scaled_lr:.6f}")

            # Use scaled_lr for optimizer
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=scaled_lr, momentum=0.9, weight_decay=args.reg)

            criterion = nn.CrossEntropyLoss().cuda()
            cos = torch.nn.CosineSimilarity(dim=-1) # cosine similarity function
            global_w = global_model.state_dict()

            # Step 6: Train the global model with contrastive loss
            
            # Ensure all models and tensors are on the same device
            # global_model = global_model.cuda()
            for i in range(len(accuracy_above_average_model_pool)):
                accuracy_above_average_model_pool[i] = accuracy_above_average_model_pool[i].cuda()
            for i in range(len(accuracy_below_average_model_pool)):
                accuracy_below_average_model_pool[i] = accuracy_below_average_model_pool[i].cuda()

            for batch_idx, (x, target) in enumerate(train_dl_global):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, pro1, out = global_model(x) # anchor

                # Use the first above-average model as positive (or average their projections)
                pos_logits = []
                for accuracy_above_average_model in accuracy_above_average_model_pool:
                    _, pro_pos, _ = accuracy_above_average_model(x)
                    posi = cos(pro1, pro_pos)
                    pos_logits.append(posi.reshape(-1, 1))
                # Average positive logits if multiple models, or just use the first one
                if len(pos_logits) > 0:
                    if len(pos_logits) > 1:
                        posi = torch.mean(torch.cat(pos_logits, dim=1), dim=1, keepdim=True)
                    else:
                        posi = pos_logits[0]
                else:
                    raise ValueError("No positive samples available for contrastive loss.")
                logits = posi # Start logits with positive sample(s)

                # Now add negative samples
                for accuracy_below_average_model in accuracy_below_average_model_pool:
                    _, pro_neg, _ = accuracy_below_average_model(x)
                    nega = cos(pro1, pro_neg)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= args.temperature
                labels = torch.zeros(x.size(0)).cuda().long() # positive is always the first column
                loss = criterion(logits, labels) # compute the contrastive loss

                # Backprop & update
                loss.backward()
                optimizer.step()
                

            logger.info('global n_test: %d' % len(test_dl_global))
            
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> After contrastive learning:')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'my_fedcon/')
            
            # Ensure all models and tensors are on the same device
            global_model.to('cpu')
            for i in range(len(accuracy_above_average_model_pool)):
                accuracy_above_average_model_pool[i] = accuracy_above_average_model_pool[i].to('cpu')
            for i in range(len(accuracy_below_average_model_pool)):
                accuracy_below_average_model_pool[i] = accuracy_below_average_model_pool[i].to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'my_fedcon/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'my_fedcon/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, net_test_dataidx_map, train_dl=train_dl,test_dl=test_dl_global, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')