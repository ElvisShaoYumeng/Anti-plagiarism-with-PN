#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import matplotlib
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import random
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from scipy import optimize
import random
import cmath
import pdb

from Calculate import get_2_norm, get_2_diff, get_w_sum, calculate_grads, avg_grads
from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test
from averaging import average_weights, average_experiments
from Noise_add import noise_add,noise_add1
from PN_sequence import *
from averaging_H import *
from Nets import cifar_cnn, mnist_cnn, cifar_cnn1,ResNet,ResidualBlock,VGG, SVHN_nn, SVHN_nn1

if __name__ == '__main__':
    # return the available GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')

    args.gpu = 0  # -1 (CPU only) or GPU = 0
    args.lr = 0.01  # 0.001 for cifar dataset
    args.model = 'Resnet'  # 'mlp' or 'cnn'
    args.dataset = 'SVHN'  # 'mnist'
    args.num_users = 20  ### numb of users ###
    # args.num_Chosenusers = 30

    args.num_items_train = 768   # numb of local data size #
    args.num_items_test = 256
    args.local_bs = 64  ### Local Batch size (1200 = full dataset ###
    args.set_epoch = [20]
    args.set_local_ep = 10
    args.rho = []
    for idx in range(0, args.num_users):
        args.rho.append( 1 / args.num_users)
    print(args.rho)

    args.set_num_Chosenusers = [args.num_users]
    args.set_lazy = int(args.num_users * 0.4)  ### no lazy
    args.num_experiments = 8
    args.clipthr = 10

    w_mean_list =[]
    w_sigma_list =[]
    noise_scale = 0.01
    PN_scale = 0.03
    K = 9 # (0,N)
    drop_pr = 1 # (0,1)
    num_long =22 # ResNet 19, cnn1 21,cnn2 24, VGG 24
    #args.r_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7] # M=1
    #args.r_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.6, 0.7, 0.8, 0.8, 0.9]  # M=4
    args.r_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0] # M=8

    args.iid = True

    # load dataset and split users
    dict_users = {}
    dict_users_test = {}
    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dataset_train = datasets.SVHN('D:/pycharm_project/data/SVHN/', split='train', download=False,
               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    dataset_test = datasets.SVHN('D:/pycharm_project/data/SVHN/', split='test', download=False,
               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    # sample users
    if args.iid:
        dict_users = cifar_iid(args, dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test)
        dict_sever = cifar_iid(args, dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = cifar_noniid(args, dataset_train, args.num_users, args.num_items_train)
        dict_sever = cifar_noniid(args, dataset_test, args.num_users, args.num_items_test)

    img_size = dataset_train[0][0].shape

    for s in range(len(args.set_num_Chosenusers)):
        for j in range(len(args.set_epoch)):
            args.num_Chosenusers = copy.deepcopy(args.set_num_Chosenusers[s])
            args.epochs = copy.deepcopy(args.set_epoch[j])  # numb of global iters
            #args.tau = args.local_frequence * (args.total_time - args.bl_antifrequence * args.epochs)
            #args.tau_avg = args.tau // args.epochs
            args.local_ep = copy.deepcopy(args.set_local_ep)  # numb of local iters
            print("dataset:", args.dataset, " num_users:", args.num_users, " num_chosen_users:", args.num_Chosenusers,
                  " epochs:", args.epochs, \
                  "local_ep:", args.local_ep, "local train size", args.num_items_train, "batch size:", args.local_bs)
            loss_test = [0 for i in range(args.num_experiments)]
            loss_train = [0 for i in range(args.num_experiments)]
            acc_test = [0 for i in range(args.num_experiments)]
            acc_train = [0 for i in range(args.num_experiments)]
            smooth_L = [0 for i in range(args.num_experiments)]
            Lipschitz_chixi = [0 for i in range(args.num_experiments)]
            gap_delta = [0 for i in range(args.num_experiments)]
            lazy_theta = [0 for i in range(args.num_experiments)]
            PN_possibility_p = [0 for i in range(args.num_experiments)]
            PN_possibility_q = [0 for i in range(args.num_experiments)]

            for m in range(args.num_experiments):
                print('experiment numbers =', m)
                # build model
                net_glob = None
                # net_local = None
                if args.model == 'mnist_cnn':
                    net_glob = mnist_cnn().to(DEVICE)
                    # net_local = CNNMnist(args=args)
                elif args.model == 'cifar_cnn':
                    net_glob = VGG('VGG11').to(DEVICE)
                elif args.model == 'cifar_cnn1':
                    net_glob = cifar_cnn1().to(DEVICE)
                elif args.model == 'SVHN_net':
                    net_glob = SVHN_nn1().to(DEVICE)
                elif args.model == 'Resnet':
                    net_glob = ResNet(ResidualBlock, [2,2,2]).to(DEVICE)
                elif args.model == 'mlp':
                    len_in = 1
                    for x in img_size:
                        len_in *= x
                    #print(x)
                    if args.gpu != -1:
                        torch.cuda.set_device(args.gpu)
                        net_glob = MLP1(dim_in=len_in, dim_hidden=32, dim_out=args.num_classes).cuda()
                    #  net_local = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).cuda()
                    else:
                        net_glob = MLP1(dim_in=len_in, dim_hidden=32, dim_out=args.num_classes)
                        # net_local = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes)
                else:
                    exit('Error: unrecognized model')
                print("Nerual Net:", net_glob)

                net_glob.train()  # Train() does not change the weight values
                # copy weights
                w_glob = net_glob.state_dict()

                # net_local.train()  #Train() does not change the weight values
                # copy weights
                # w_local = net_local.state_dict()

                w_size = 0
                w_size_all = 0
                # training
                threshold_epochs = copy.deepcopy(args.epochs)
                threshold_epochs_list, noise_list = [], []
                loss_avg_list, acc_avg_list, list_loss, loss_avg = [], [], [], []
                eps_tot_list, eps_tot = [], 0
                ###  FedAvg Aglorithm  ###
                ### Compute noise scale ###
                loss_test_exp1, loss_train_exp1 = [], []
                acc_test_exp1, acc_train_exp1 = [], []
                smooth_L_exp1, Lipschitz_chixi_exp1, gap_delta_exp1, lazy_theta_exp1 = [], [], [], []
                PN_possibility_exp1p, PN_possibility_exp1q = [], []
                for iter in range(args.epochs):
                    if iter == 1:
                        K = 10
                        PN_scale = 0.5
                    else:  ## FROM MATLAB Calculation
                        K = 9
                        PN_scale = 0.03
                    print('\n', '*' * 20, f'Epoch: {iter}', '*' * 20)
                    if args.num_Chosenusers < args.num_users:
                        chosenUsers = random.sample(range(1, args.num_users), args.num_Chosenusers)
                        chosenUsers.sort()
                    else:
                        chosenUsers = range(args.num_users)
                    print("\nChosen users:", chosenUsers)
                    w_locals, w_locals_1ep, loss_locals, acc_locals, w_locals_2ep = [], [], [], [], []
                    w_difference, difference_loss = [], []
                    w_lazy_diff_list = []
                    PN_list = []
                    w_glob_pre = w_glob
                    # local_chixi, local_L, local_delta = [], [], []
                    ### local train  ###
                    for idx in range(len(chosenUsers)):
                        lazy_nounce = np.random.rand()
                        if lazy_nounce > args.r_index[idx]:
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                                tb=summary)
                            w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            ### weight distribution ###
                            w_length = weight_length(w)
                            sequence_peak = K * w_length * PN_scale / args.num_users # K L \alpha / N
                            mean, sigma = weight_distribution(w)
                            w_mean_list.append(mean)
                            w_sigma_list.append(sigma)

                            ### adding PN sequence ###
                            array = np.random.randint(0, 2, num_long)
                            random_num = np.random.randint(2, (len(array) + 1))
                            PN = generate_sequence(array, random_num, w_length)
                            PN_list.append(PN)
                            w = sequence_add(w, PN, PN_scale)

                            ### get updated local weights ###
                            w_locals.append(copy.deepcopy(w))
                            ### record 1st-ep and 2nd-ep local weights ###
                            w_locals_1ep.append(copy.deepcopy(w_1st_ep))
                            w_locals_2ep.append(copy.deepcopy(w_2st_ep))
                            ### get local loss ###
                            loss_locals.append(copy.deepcopy(loss))
                            # print("User:", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            acc_locals.append(copy.deepcopy(acc))

                            ### for lazy user ###
                        else:
                            ###  copy  ###
                            k = random.randint(0, (idx - 1))
                            #print(k)
                            lazy_locals = copy.deepcopy(w_locals[k])
                            lazy_locals_1ep = copy.deepcopy(w_locals_1ep[k])
                            lazy_locals_2ep = copy.deepcopy(w_locals_2ep[k])

                            ### generate PN sequence ###
                            w_length = weight_length(lazy_locals)
                            array = np.random.randint(0, 2, num_long)
                            random_num = np.random.randint(2, (len(array) + 1))
                            PN = generate_sequence(array, random_num, w_length)
                            #PN = [0 for number in range(w_length)]
                            PN_list.append(PN)
                            #lazy_locals = sequence_add(w, PN, PN_scale)

                            w_locals.append(copy.deepcopy(lazy_locals))
                            w_locals_1ep.append(copy.deepcopy(lazy_locals_1ep))
                            w_locals_2ep.append(copy.deepcopy(lazy_locals_2ep))
                            lazy_loss = copy.deepcopy(loss_locals[k])
                            lazy_acc = copy.deepcopy(acc_locals[k])
                            loss_locals.append(copy.deepcopy(lazy_loss))
                            acc_locals.append(copy.deepcopy(lazy_acc))

                            ### perturb 'w_local' ###
                            w_locals[idx] = noise_add1(args, noise_scale, w_locals[idx])  # noise variance is 0.01#
                            w_locals_1ep[idx] = noise_add1(args,noise_scale, w_locals_1ep[idx])
                            w_locals_2ep[idx] = noise_add1(args,noise_scale, w_locals_2ep[idx])

                            ### theta para estimate ###
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                            tb=summary)
                            w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            w_lazy_diff_list.append(get_2_norm(w, w_locals[k]))


                    ### PN detect
                    PN_correct = [1 for i in range(args.num_users)]
                    PN_p = 0
                    PN_q = 0
                    PN_index = [[0 for i in range(len(w_locals))] for j in range(len(PN_list))]
                    for j in range(len(PN_list)):
                        for i in range(len(w_locals)):
                            PN_detect = sequence_detect(w_locals[i], PN_list[j])
                            PN_index[j][i] = copy.deepcopy(PN_detect)
                    for j in range(len(PN_list)):
                        if j < (len(chosenUsers) - args.set_lazy):
                            for idx in range(len(PN_index[j])):
                                if (PN_index[j][idx] > sequence_peak) and (PN_index[idx][j] < sequence_peak):
                                    PN_correct[idx] = 0
                    for idx in range(len(PN_correct)):
                        if idx < (len(chosenUsers) - args.set_lazy):
                            if PN_correct[idx] == 1:
                                PN_q += 1
                        else:
                            if PN_correct[idx] == 0:
                                PN_p += 1
                    PN_correct_Q = PN_q / (len(chosenUsers) - args.set_lazy)
                    #PN_correct_P = PN_p / args.set_lazy


                    ### drop algoritm ###
                    for idx in range(args.num_users):
                        args.rho[idx] = (1 / args.num_users)
                    PN_zero = 0
                    for idx in range(len(PN_correct)):
                        if PN_correct[idx] == 0:
                            nounce = np.random.rand()
                            if nounce < drop_pr:
                                args.rho[idx] = 0
                                PN_zero += 1
                    for idx in range(len(args.rho)):
                        if args.rho[idx] != 0:
                            args.rho[idx] = 1 / (args.num_users - PN_zero)

                    ### update global weights ###
                    w_glob = average_compt_weights(w_locals, args.rho)

                    ###  update 1ep_weights  ###
                    w_1ep = average_compt_weights(w_locals_1ep, args.rho)

                    # copy weight to net_glob
                    net_glob.load_state_dict(w_glob)
                    # global test
                    list_acc, list_loss = [], []
                    grad_list, grad_local_list = [], []

                    chixi_list, delta_list, = [], []

                    w_avg, w_last_avg = [], [],
                    grad_local = []
                    grad_glob = []
                    para_loss = []
                    #print('PN')

                    net_glob.eval()
                    for c in range(args.num_users):
                        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[c], tb=summary)
                        acc, loss = net_local.test(net=net_glob)
                        # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                        list_acc.append(copy.deepcopy(acc))
                        list_loss.append(copy.deepcopy(loss))
                    for c in range(args.num_users):
                        net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_sever[c], tb=summary)
                        acc, loss = net_local.test(net=net_glob)
                        # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                        para_loss.append(copy.deepcopy(loss))

                    #print('test')
                    grad_locals_1ep, grad_locals_glob, grad_list, delta_list = [], [], [], []
                    for idx in range(len(chosenUsers)):
                        ###-calculate gradients-###
                        grad_locals_glob.append(calculate_grads(args, w_glob_pre, w_locals_1ep[idx]))
                        grad_locals_1ep.append(calculate_grads(args, w_locals_1ep[idx], w_locals_2ep[idx]))

                        grad_list.append(get_2_norm(grad_locals_glob[idx], grad_locals_1ep[idx]) / \
                                         get_2_norm(w_glob_pre, w_locals_1ep[idx]))

                    grad_glob = avg_grads(grad_locals_glob)
                    for idx in range(len(chosenUsers)):
                        delta_list.append(get_2_norm(grad_locals_glob[idx], grad_glob))

                    ###  different_w  ###

                    for idx in range(len(chosenUsers)):
                        w_difference.append(get_2_norm(w_locals[chosenUsers[idx]], w_glob))

                    ###  loss_difference  ###
                    for idx in range(len(chosenUsers)):
                        diff_loss = loss_locals[idx] - para_loss[idx]
                        difference_loss.append(np.linalg.norm(diff_loss))

                    ###  update lazy diff weights  ###
                    #w_lazy_diff = sum(w_lazy_diff_list) / len(w_lazy_diff_list)


                    ###  chixi_list  ###
                    for idx in range(len(chosenUsers)):
                        chixi_list.append(difference_loss[idx] / w_difference[idx])

                    chixi_avg = sum(chixi_list) / len(chixi_list)
                    L_avg = sum(grad_list) / len(grad_list)
                    delta_avg = sum(delta_list) / len(delta_list)

                    loss_avg = sum(loss_locals) / len(loss_locals)
                    acc_avg = sum(acc_locals) / len(acc_locals)
                    loss_avg_list.append(loss_avg)
                    acc_avg_list.append(acc_avg)


                    print("\nTrain loss: {}, Train acc: {}". \
                          format(loss_avg_list[-1], acc_avg_list[-1]))
                    print("\nTest loss: {}, Test acc: {}". \
                          format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)))
                    '''if iter >= 1:
                        print("\nTrain chixi: {}". \
                            format(chixi_avg_list[-1]))'''

                    Lipschitz_chixi_exp1.append(chixi_avg)
                    smooth_L_exp1.append(L_avg)
                    gap_delta_exp1.append(delta_avg)
                    #lazy_theta_exp1.append(w_lazy_diff)

                    loss_train_exp1.append(loss_avg)
                    acc_train_exp1.append(acc_avg)
                    loss_test_exp1.append(sum(list_loss) / len(list_loss))
                    acc_test_exp1.append(sum(list_acc) / len(list_acc))

                    #PN_possibility_exp1p.append(PN_correct_P)
                    #PN_possibility_exp1q.append(PN_correct_Q)

                Lipschitz_chixi[m] = copy.deepcopy(Lipschitz_chixi_exp1)
                smooth_L[m] = copy.deepcopy(smooth_L_exp1)
                gap_delta[m] = copy.deepcopy(gap_delta_exp1)
                lazy_theta[m] = copy.deepcopy(lazy_theta_exp1)

                loss_train[m] = copy.deepcopy(loss_train_exp1)
                acc_train[m] = copy.deepcopy(acc_train_exp1)
                loss_test[m] = copy.deepcopy(loss_test_exp1)
                acc_test[m] = copy.deepcopy(acc_test_exp1)

                PN_possibility_p[m] = copy.deepcopy(PN_possibility_exp1p)
                PN_possibility_q[m] = copy.deepcopy(PN_possibility_exp1q)

            if args.num_experiments > 1:
                final_train_loss = copy.deepcopy(average_experiments(loss_train))
                final_train_accuracy = copy.deepcopy(average_experiments(acc_train))
                final_test_loss = copy.deepcopy(average_experiments(loss_test))
                final_test_accuracy = copy.deepcopy(average_experiments(acc_test))

                final_Lipschitz_chixi = copy.deepcopy(average_experiments(Lipschitz_chixi))
                final_smooth_L = copy.deepcopy(average_experiments(smooth_L))
                final_gap_delta = copy.deepcopy(average_experiments(gap_delta))
                final_lazy_theta = copy.deepcopy(average_experiments(lazy_theta))

                final_PN_possibility_p = copy.deepcopy(average_experiments(PN_possibility_p))
                final_PN_possibility_q = copy.deepcopy(average_experiments(PN_possibility_q))
            else:
                final_train_loss = copy.deepcopy(loss_train)
                final_train_accuracy = copy.deepcopy(acc_train)
                final_test_loss = copy.deepcopy(loss_test)
                final_test_accuracy = copy.deepcopy(acc_test)

                final_Lipschitz_chixi = copy.deepcopy(Lipschitz_chixi)
                final_smooth_L = copy.deepcopy(smooth_L)
                final_gap_delta = copy.deepcopy(gap_delta)
                final_lazy_theta = copy.deepcopy(lazy_theta)

                final_PN_possibility_p  = copy.deepcopy(PN_possibility_p)
                final_PN_possibility_q = copy.deepcopy(PN_possibility_q)

        theo_mu = sum(w_mean_list) / len(w_mean_list)
        theo_sigma = sum(w_sigma_list) / len(w_sigma_list)

        print('\nFinal train loss:', final_train_loss)
        print('\nFinal train accuracy:', final_train_accuracy)
        print('\nFinal test loss:', final_test_loss)
        print('\nFinal test accuracy:', final_test_accuracy)

        print('\nFinal Lipschitz chixi:', final_Lipschitz_chixi)
        print('\nFinal smooth L:', final_smooth_L)
        print('\nFinal delta:', final_gap_delta)
        print('\nFinal theta:', final_lazy_theta)

        print('\nFinal p', final_PN_possibility_p)
        print('\nFinal q', final_PN_possibility_q)

    timeslot = int(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
    with open('./SimulationData/SVHN/new_fed_{}UEs_{}_{}_epoch{}_iid{}_PN{}_lazy{}_{}.csv'. \
                      format(args.num_users, args.dataset, \
                             args.model,  args.epochs,
                             args.iid, PN_scale, args.set_lazy, timeslot), 'w', encoding='utf-8') as f:
        f.write('Train_loss:')
        f.write(str(final_train_loss))
        f.write('\nTrain_accuracy:')
        f.write(str(final_train_accuracy))
        f.write('\nTest_loss:')
        f.write(str(final_test_loss))
        f.write('\nTest_accuracy:')
        f.write(str(final_test_accuracy))
        f.write('\nLipschitz chixi:')
        f.write(str(final_Lipschitz_chixi))
        f.write('\nsmooth L:')
        f.write(str(final_smooth_L))
        f.write('\ndelta:')
        f.write(str(final_gap_delta))
        f.write('\ntheta:')
        f.write(str(final_lazy_theta))
        f.write('\nLazy correct p')
        f.write(str(final_PN_possibility_p))
        f.write('\nNormal correct q')
        f.write(str(final_PN_possibility_q))
        f.write('\nmean:')
        f.write(str(theo_mu))
        f.write('\nsigma:')
        f.write(str(theo_sigma))