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
from Noise_add import noise_add
from PN_sequence import weight_length, sequence_detect, generate_sequence,  weight_distribution, normal_distribution
from  Nets import mnist_cnn

if __name__ == '__main__':
    # return the available GPU
    '''av_GPU = torch.cuda.is_available()
    if av_GPU == False:
        exit('No available GPU')'''
    DEVICE = 'cpu'
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
    ### computation allocation ###
    args.dp_mechanism = 'CRD'  ### CRD or non-CRD###
    args.dec_cons = 0.8  ## discount constant
    # args.privacy_budget = 100
    args.delta = 0.1


    # args.cauchy=0.1
    # args.T=3

    args.gpu = -1  # -1 (CPU only) or GPU = 0
    args.lr = 0.01  # 0.001 for cifar dataset
    args.model = 'mnist_cnn'  # 'mlp' or 'cnn'
    args.dataset = 'mnist'  # 'mnist'
    args.num_users = 20  ### numb of users ###
    # args.num_Chosenusers = 30

    args.num_items_train = 512   # numb of local data size #
    args.num_items_test = 256
    args.local_bs = 32  ### Local Batch size (1200 = full dataset ###
    ### size of a user for mnist, 2000 for cifar) ###
    # T_max = B *delta_t / (B * C + 1)
    #args.set_epoch = range(1, args.T_max + 1)
    args.set_epoch = [100]
    args.set_local_ep = 1
    PN_scale = 0

    # args.set_epochs = range(5,105,5)
    args.set_num_Chosenusers = [args.num_users]
    args.set_lazy = int(args.num_users * 0)  ### 20% lazy
    args.num_experiments = 50
    args.clipthr = 10
    noise_scale = 0.01
    num_long = 20

    interval = 30
    num_w = args.num_experiments * args.set_epoch[0] * args.num_users
    correlation_list = []
    w_mean_list =[]
    w_sigma_list =[]

    args.iid = True

    # load dataset and split users
    dict_users = {}
    dict_users_test = {}
    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    # sample users
    if args.iid:
        dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test)
        dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
        dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)

    img_size = dataset_train[0][0].shape


    for s in range(len(args.set_num_Chosenusers)):
        for j in range(len(args.set_epoch)):
            args.num_Chosenusers = copy.deepcopy(args.set_num_Chosenusers[s])
            args.epochs = copy.deepcopy(args.set_epoch[j])  # numb of global iters
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
            #lazy_theta = [0 for i in range(args.num_experiments)]
            #PN_possibility = [0 for i in range(args.num_experiments)]

            for m in range(args.num_experiments):
                print('experiment numbers =',m)
                # build model
                net_glob = None
                # net_local = None
                if args.model == 'mnist_cnn':
                    net_glob = mnist_cnn().to(DEVICE)
                elif args.model == 'mlp':
                    len_in = 1
                    for x in img_size:
                        len_in *= x
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
                for k in w_glob.keys():
                    size = w_glob[k].size()
                    if (len(size) == 1):
                        nelements = size[0]
                    else:
                        nelements = size[0] * size[1]
                    w_size += nelements * 4
                    w_size_all += nelements
                    # print("Size ", k, ": ",nelements*4)
                print("Weight Size:", w_size, " bytes")
                print("Weight & Grad Size:", w_size * 2, " bytes")
                print("Each user Training size:", 784 * 8 / 8 * args.local_bs, " bytes")
                print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
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
                PN_possibility_exp1 = []
                for iter in range(args.epochs):
                    print('\n', '*' * 20, f'Epoch: {iter}', '*' * 20)
                    if args.num_Chosenusers < args.num_users:
                        chosenUsers = random.sample(range(1, args.num_users), args.num_Chosenusers)
                        chosenUsers.sort()
                    else:
                        chosenUsers = range(args.num_users)
                    print("\nChosen users:", chosenUsers)
                    w_locals, w_locals_1ep, loss_locals, acc_locals, w_locals_2ep = [], [], [], [], []
                    w_difference, difference_loss = [], []
                    #w_lazy_diff_list = []
                    w_glob_pre = w_glob
                    # local_chixi, local_L, local_delta = [], [], []
                    ### local train  ###
                    for idx in range(len(chosenUsers)):
                        # m = max(int(args.frac * args.num_users), 1)
                        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                        if idx < (len(chosenUsers) - args.set_lazy):
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                                tb=summary)
                            w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            ### adding PN sequence ###
                            w_length = weight_length(w)
                            array = np.random.randint(0, 2, num_long)
                            random_num = np.random.randint(2, (len(array) + 1))
                            PN = generate_sequence(array, random_num, w_length)
                            correlation_list.append(float(sequence_detect(w, PN)))
                            mean, sigma = weight_distribution(w)
                            w_mean_list.append(mean)
                            w_sigma_list.append(sigma)

                            #w = sequence_add(w, PN, PN_scale)
                            #pdb.set_trace()
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
                            lazy_locals = copy.deepcopy(w_locals[k])
                            lazy_locals_1ep = copy.deepcopy(w_locals_1ep[k])
                            lazy_locals_2ep = copy.deepcopy(w_locals_2ep[k])
                            w_locals.append(copy.deepcopy(lazy_locals))
                            w_locals_1ep.append(copy.deepcopy(lazy_locals_1ep))
                            w_locals_2ep.append(copy.deepcopy(lazy_locals_2ep))
                            lazy_loss = copy.deepcopy(loss_locals[k])
                            lazy_acc = copy.deepcopy(acc_locals[k])
                            loss_locals.append(copy.deepcopy(lazy_loss))
                            acc_locals.append(copy.deepcopy(lazy_acc))

                            ### perturb 'w_local' ###
                            w_locals[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args, noise_scale, \
                                 w_locals[len(chosenUsers) - args.set_lazy:len(chosenUsers)])  # noise variance is 0.01#
                            w_locals_1ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args,noise_scale, \
                                 w_locals_1ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)])
                            w_locals_2ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args,noise_scale, \
                                 w_locals_2ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)])

                            ### theta para estimate ###
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                            tb=summary)
                            w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            #w_lazy_diff_list.append(get_2_norm(w, w_locals[k]))

                    #print(correlation_list)
                    #print(w_mean_list)
                    #print(w_sigma_list)
                    ### PN detect ###  1表示识别正确， 0表示识别错误
                    '''PN_correct = []
                    PN_p = 0
                    for i in range(len(w_locals)):
                        PN_index = []
                        for j in range(len(PN_list)):
                            PN_detect = sequence_detect(w_locals[i], PN_list[j])
                            PN_index.append(PN_detect)
                        for k in range(len(PN_index)): ## 提取最大值的位置
                            if PN_index[k] == max(PN_index):
                                break
                        if i == k:
                            flag =1
                        else:
                            flag =0
                        PN_correct.append(flag)
                    print(PN_correct)
                    for idx in range(len(PN_correct)):
                        if idx < (len(chosenUsers) - args.set_lazy):
                            if PN_correct[idx] == 0:
                                PN_p += 1
                        else:
                            if PN_correct[idx] == 1:
                                PN_p += 1
                    PN_correct_P = PN_p / len(PN_correct)
                    PN_correct_P = 1 - PN_correct_P'''
                    #pdb.set_trace()

                    ### perturb 'w_local' ###
                    # w_locals = noise_add(args, noise_scale, w_locals) #noise is none#


                    ### update global weights ###
                    # w_locals = users_sampling(args, w_locals, chosenUsers)
                    w_glob = average_weights(w_locals)

                    ###  update 1ep_weights  ###
                    w_1ep = average_weights(w_locals_1ep)

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
                        ###  for lazy user  ###
                        # else:
                    # print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".\
                    #      format(iter, sum(list_loss) / len(list_loss),100. * sum(list_acc) / len(list_acc)))
                    # print loss

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
                        # diff_w = w_locals_1ep[idx] - w_glob
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

                    # noise_list.append(noise_scale)
                    # print('\nNoise Scale:', noise_list)

                    '''if args.dp_mechanism == 'CRD' and iter >= 1:
                        threshold_epochs_list.append(threshold_epochs)
                        threshold_epochs = Adjust_T(args, loss_avg_list, threshold_epochs_list, iter)
                        # noise_scale = copy.deepcopy(Privacy_account(args, threshold_epochs, noise_list, iter))
                        print('\nThreshold epochs:', threshold_epochs_list)'''

                    Lipschitz_chixi_exp1.append(chixi_avg)
                    smooth_L_exp1.append(L_avg)
                    gap_delta_exp1.append(delta_avg)
                    #lazy_theta_exp1.append(w_lazy_diff)

                    loss_train_exp1.append(loss_avg)
                    acc_train_exp1.append(acc_avg)
                    loss_test_exp1.append(sum(list_loss) / len(list_loss))
                    acc_test_exp1.append(sum(list_acc) / len(list_acc))

                    #PN_possibility_exp1.append(PN_correct_P)
                #print(correlation_list)
                #print(w_mean_list)
                #print(w_sigma_list)

                Lipschitz_chixi[m] = copy.deepcopy(Lipschitz_chixi_exp1)
                smooth_L[m] = copy.deepcopy(smooth_L_exp1)
                gap_delta[m] = copy.deepcopy(gap_delta_exp1)
                #lazy_theta[m] = copy.deepcopy(lazy_theta_exp1)

                loss_train[m] = copy.deepcopy(loss_train_exp1)
                acc_train[m] = copy.deepcopy(acc_train_exp1)
                loss_test[m] = copy.deepcopy(loss_test_exp1)
                acc_test[m] = copy.deepcopy(acc_test_exp1)

                #PN_possibility[m] = copy.deepcopy(PN_possibility_exp1)

            if args.num_experiments > 1:
                final_train_loss = copy.deepcopy(average_experiments(loss_train))
                final_train_accuracy = copy.deepcopy(average_experiments(acc_train))
                final_test_loss = copy.deepcopy(average_experiments(loss_test))
                final_test_accuracy = copy.deepcopy(average_experiments(acc_test))

                final_Lipschitz_chixi = copy.deepcopy(average_experiments(Lipschitz_chixi))
                final_smooth_L = copy.deepcopy(average_experiments(smooth_L))
                final_gap_delta = copy.deepcopy(average_experiments(gap_delta))
                #final_lazy_theta = copy.deepcopy(average_experiments(lazy_theta))

                #final_PN_possibility = copy.deepcopy(average_experiments(PN_possibility))
            else:
                final_train_loss = copy.deepcopy(loss_train)
                final_train_accuracy = copy.deepcopy(acc_train)
                final_test_loss = copy.deepcopy(loss_test)
                final_test_accuracy = copy.deepcopy(acc_test)

                final_Lipschitz_chixi = copy.deepcopy(Lipschitz_chixi)
                final_smooth_L = copy.deepcopy(smooth_L)
                final_gap_delta = copy.deepcopy(gap_delta)
                #final_lazy_theta = copy.deepcopy(lazy_theta)

                #final_PN_possibility  = copy.deepcopy(PN_possibility)

            #print(correlation_list)
            #print(w_mean_list)
            #print(w_sigma_list)

        theo_mu = sum(w_mean_list) / len(w_mean_list)
        theo_sigma = sum(w_sigma_list) / len(w_sigma_list)
        print(theo_sigma)
        print(num_w)

        x_value = []
        count = [0 for i in range(interval)]
        cross_max = max(correlation_list)
        cross_min = min(correlation_list)
        gap = (cross_max - cross_min) / interval
        for i in range(interval):
            for j in range(len(correlation_list)):
                if correlation_list[j] >= (cross_min + i * gap) and correlation_list[j] <= (cross_min + (i + 1) * gap):
                    count[i] += 1
        '''for i in range(len(count)):
            count[i] = count[i] / num_w * 5.25
        print(count)'''
        for i in range(interval):
            x_value.append(cross_min + i * gap + gap / 2)
        #print(x_value)
        #plt.plot(x_value, count, 'b', label='Cross-correlation')

        ### plot theoretical gaussian distribution
        x1 = np.linspace(theo_mu - 6 * theo_sigma, theo_mu + 6 * theo_sigma, 100)
        y1 = normal_distribution(x1, theo_mu, theo_sigma)
        plt.plot(x1, y1, 'r', label='Gaussian approximation')
        plt.title(' length = {}, num_exp = {}'.format(w_length, num_w))

        plt.legend()
        plt.show()

        '''print('\nFinal train loss:', final_train_loss)
        print('\nFinal train accuracy:', final_train_accuracy)
        print('\nFinal test loss:', final_test_loss)
        print('\nFinal test accuracy:', final_test_accuracy)

        print('\nFinal Lipschitz chixi:', final_Lipschitz_chixi)
        print('\nFinal smooth L:', final_smooth_L)
        print('\nFinal delta:', final_gap_delta)
        #print('\nFinal theta:', final_lazy_theta)

        #print('\nFinal PN possibility', final_PN_possibility)'''

    timeslot = int(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
    with open('./SimulationData/Gaussian_approximation/new_fed_{}UEs_model{}_exp{}_num{}_{}.csv'. \
                      format(args.num_users,  args.model, args.num_experiments, \
                             num_w, timeslot), 'w', encoding='utf-8') as f:
        f.write('x:')
        f.write(str(x_value))
        f.write('\nFrequence:')
        f.write(str(count))
        f.write('\nmean:')
        f.write(str(theo_mu))
        f.write('\nsigma:')
        f.write(str(theo_sigma))
        f.write('\nx_value_min:')
        f.write(str(cross_min))
        f.write('\ngap:')
        f.write(str(gap))
        f.write('\ninterval:')
        f.write(str(interval))
        f.write('\nnum of experiments:')
        f.write(str(num_w))
        f.write('\ncrosscorrelation:')
        f.write(str(correlation_list))
