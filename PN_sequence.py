import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow as tf
import math

# 伪随机序列
# pseudo_random_state表示从界面上获取的状态值，分别对应着7,9,15,16,20,21,23和用户自定义
# init_value表示初始值，用字符串保存；expression为本原表达式的幂，用列表保存
# 字典用伪随机状态作为键，包含初始值和本原表达式的幂的列表作为值，进行一一对应
def generate_prbs(pseudo_random_state, init_value=None, expression=None):

    if pseudo_random_state == 'user_define':
        pseudo_random_sequence = real_calculate_prbs(init_value, expression)
    else:
        pseudo_random_dict = {'prbs_7': ['1111101', [7, 3]],
            'prbs_9': ['111110101', [9, 4]],
            'prbs_15': ['111110101101110', [15, 1]],
            'prbs_16': ['1111101011011100', [16, 12, 3, 1]],
            'prbs_20': ['11111010110111001011', [20, 3]],
            'prbs_21': ['111110101101110010111', [21, 2]],
            'prbs_23': ['11111010110111001011101', [23, 5]]}
        pseudo_random_sequence = real_calculate_prbs(pseudo_random_dict[pseudo_random_state][0],
                pseudo_random_dict[pseudo_random_state][1])
    return pseudo_random_sequence


# 真正计算伪随机序列

# 用xrange省空间同时提高效率
def real_calculate_prbs(value, expression):

#将字符串转化为列表
    value_list = [int(i) for i in list(value)]

#计算伪随机序列周期长度
    pseudo_random_length = (2 << (len(value) - 1))-1

    sequence = []

#产生规定长度的伪随机序列
    for i in range(pseudo_random_length):

        mod_two_add = sum([value_list[t-1] for t in expression])

        xor = mod_two_add % 2

#计算并得到伪随机序列的状态值
        value_list.insert(0, xor)


        sequence.append(value_list[-1])
        del value_list[-1]

    return sequence


def generate_sequence(array, random_num, length):
    result_data = generate_prbs('user_define', array, [random_num, 1])
    for i in range(len(result_data)):
        if result_data[i] == 1:
            result_data[i] = 1
        else:
            result_data[i] = -1
    result_data = np.array(result_data)
    result = np.zeros(length)
    result += result_data[0:length]
    return result

def sequence_scale(c, scale):
    a = copy.deepcopy(c)
    for i in range(len(a)):
        a[i] *= scale
    return a

#循环右移一次
def movenum(list):    #list表示列表
    n = len(list)
    end = list[n-1]
    for i in range(n-1,-1,-1):
        list[i] = list[i-1]
        list[0] = end
        list = np.array(list)
    return list

def xor_sequence(a,b): #a,b列表的长度相等，获取a，b异或后的列表
    c=[]
    for i in range(len(a)):
        if int(a[i]) * int(b[i]) == 1:
            c.append(-1)
        else:
            c.append(1)
    return c

def generate_gold_sequence(m1, m2, n): #m1,m2是输入m序列，n是序列阶数
    length = len(m1)
    final_result = 0
    if n % 2 == 1:
        t = 2 ** ((n + 1) / 2) + 1
    else:
        t = 2 ** ((n + 2) / 2) + 1
    for i in range(length):
        m2 = movenum(m2)
        co_result = np.correlate(m1, m2, mode='valid')
        if co_result == -1 or co_result == -t or co_result == t-2:
            final_result = xor_sequence(m1, m2)
            #print('移位次数：',i+1)
            return final_result
    return final_result

#右移
def right_move(list):
    n = len(list)
    for i in range(n-1,-1,-1):
        list[i] = list[i-1]
    list[0] = 0
    list = np.array(list)
    return list

#左移
def left_move(list):
    n = len(list)

    for i in range(0,n-1,1):
        list[i] = list[i+1]
    list[n - 1] = 0
    list = np.array(list)
    return list

def right_cross_correlate(a,b):  #a,b 为序列，m为左移右移绝对值
    m = len(b)
    c = [0 for i in range(m)]
    #result = np.correlate(a, b, mode='valid')
    for i in range (m):
        b = right_move(b)
        #print(b)
        c[i] = int(np.correlate(a, b, mode='valid'))
    return c

def left_cross_correlate(a,b):
    #m = len(b)
    result = int(np.correlate(a, b, mode='valid'))
    m = len(b)
    #print(result)
    d = [0 for j in range(m)]
    #result = int(np.correlate(a, b, mode='valid'))
    for i in range (m):
        b = left_move(b)
        #print(b)
        d[m-i-1] = int(np.correlate(a, b, mode='valid'))
    d.append(result)
    return d

def cross_correlate(a,b):
    c = copy.deepcopy(a)
    d = copy.deepcopy(b)
    e = left_cross_correlate(a, b)
    f = right_cross_correlate(c, d)
    #print(f)
    e.extend(f)
    return e

def range_Gaussian(lower, upper,mu,sigma,num):  #lower 代表了范围下限，upper代表了范围上限，mu 代表均值， sigma代表方差，num代表生成的随机数个数
    X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)#有区间限制的随机数
    #N = stats.norm(loc=mu, scale=sigma)#无区间限制的随机数
    a = X.rvs(num)                 #取其中的num个数，赋值给a；a为array类型
    return  a

def sequence_detect(w, b):
    a = []
    a_0 = copy.deepcopy(w)
    for k in a_0.keys():
        if k.find('tracked') == -1:
            a_0[k] = a_0[k].cpu().numpy()
            if a_0[k].ndim == 1:
                for i in range(len(a_0[k])):
                    a.append(a_0[k][i])
            if a_0[k].ndim == 2:
                for i in range(len(a_0[k])):
                    for j in range(len(a_0[k][i])):
                        a.append(a_0[k][i][j])
            if a_0[k].ndim == 3:
                for i in range(len(a_0[k])):
                    for j in range(len(a_0[k][i])):
                        for l in range(len(a_0[k][i][j])):
                            a.append(a_0[k][i][j][l])
            if a_0[k].ndim == 4:
                for i in range(len(a_0[k])):
                    for j in range(len(a_0[k][i])):
                        for l in range(len(a_0[k][i][j])):
                            for m in range(len(a_0[k][i][j][l])):
                                a.append(a_0[k][i][j][l][m])
    a = np.array(a)
    corr_a = np.correlate(a, b, 'valid')
    return corr_a

def weight_distribution(w):
    a = []
    a_0 = copy.deepcopy(w)
    for k in a_0.keys():
        a_0[k] = a_0[k].cpu().numpy()
        if a_0[k].ndim == 1:
            for i in range(len(a_0[k])):
                a.append(a_0[k][i])
        if a_0[k].ndim == 2:
            for i in range(len(a_0[k])):
                for j in range(len(a_0[k][i])):
                    a.append(a_0[k][i][j])
    a = np.array(a)
    mean = sum(a) / len(a)
    sigma = 0
    for i in range(len(a)):
        sigma += (a[i] - mean) ** 2
    sigma = sigma ** 0.5
    return mean, sigma

def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

def weight_length(w_raw):
    w = copy.deepcopy(w_raw)
    length = 0
    for k in w.keys():
        w[k] = w[k].cpu().numpy()
        if w[k].ndim == 1:
            length += len(w[k])
        if w[k].ndim == 2:
            for j in range(len(w[k])):
                length += len(w[k][j])
        if w[k].ndim == 3:
            for j in range(len(w[k])):
                for m in range(len(w[k][j])):
                    length += len(w[k][j][m])
        if w[k].ndim == 4:
            for j in range(len(w[k])):
                for m in range(len(w[k][j])):
                    for n in range(len(w[k][j][m])):
                        length += len(w[k][j][m][n])
    return length

def sequence_add(w, PN, scale):
    PN_0 = sequence_scale(PN, scale)
    idex = 0
    for k in w.keys():
        if k.find('tracked') == -1:
            w[k] = w[k].cpu().numpy()
            if w[k].ndim == 1:
                w[k] += PN_0[idex:(idex + len(w[k]))]
                idex += len(w[k])
            if w[k].ndim == 2:
                for j in range(len(w[k])):
                    w[k][j] += PN_0[idex:(idex + len(w[k][j]))]
                    idex += len(w[k][j])
            if w[k].ndim == 3:
                for j in range(len(w[k])):
                    for l in range(len(w[k][j])):
                        w[k][j][l] += PN_0[idex:(idex + len(w[k][j][l]))]
                        idex += len(w[k][j][l])
            if w[k].ndim == 4:
                for j in range(len(w[k])):
                    for l in range(len(w[k][j])):
                        for m in range(len(w[k][j][l])):
                            w[k][j][l][m] += PN_0[idex:(idex + len(w[k][j][l][m]))]
                            idex += len(w[k][j][l][m])
            w[k] = torch.FloatTensor(w[k])
    return w

def test_dim(testlist, dim=0):
   """tests if testlist is a list and how many dimensions it has
   returns -1 if it is no list at all, 0 if list is empty
   and otherwise the dimensions of it"""
   if isinstance(testlist, list):
      if testlist == []:
          return dim
      dim = dim + 1
      dim = test_dim(testlist[0], dim)
      return dim
   else:
      if dim == 0:
          return -1
      else:
          return dim

def power_weights(w_input):  ### 单个信号能量 ###
    w = copy.deepcopy(w_input)
    w_sum = 0
    if isinstance(w,np.ndarray) == True:
        for i in range(len(w)):
            w_sum += w[i] * w[i]
    else:
        for k in w.keys():
            w[k] = w[k].numpy()
            if w[k].ndim  == 1:
                for i in range(len(w[k])):
                    w_sum += w[k][i] * w[k][i]
            if w[k].ndim  == 2:
                for i in range(len(w[k])):
                    for j in range(len(w[k][i])):
                        w_sum += w[k][i][j] * w[k][i][j]
    return w_sum

def move_left1(a):
    x = copy.deepcopy(a)
    begin = x[0]
    for i in range(len(x)-1):
        x[i]=x[i+1]
    x[-1] = begin
    return x




# 不同的PN序列分别与用户权重进行互相关
if __name__ == '__main__':
    alpha = 0.68
    #print(math.erf(alpha))
    alpha_list = [0.02, 0.0035, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    num = 10
    m = 11
    length = 2 ** m - 1
    lower = -10
    upper = 10
    mu = 0
    sigma = 1
    num_exp = 20
    x_list = []
    correct_list = []
    for l in range(len(alpha_list)):
        x = ((length ** 0.5 * alpha_list[l]) / (num * ((alpha_list[l] ** 2 + 1) ** 0.5))) * (2 ** 0.5)
        #print(x)
        x_list.append(x)
        total_correct_pr = []
        for h in range(num_exp):
            pn_value, w_value, co_value, total_co_value, mean_value, total_mean_value = [], [], [], [], [], []
            for i in range(num):
                for j in range(100): # generate gold sequence
                    array = np.random.randint(0, 2, m)
                    random_num = np.random.randint(2, (len(array) + 1))
                    m1 = generate_sequence(array, random_num, length)
                    array = np.random.randint(0, 2, m)
                    random_num = np.random.randint(2, (len(array) + 1))
                    m2 = generate_sequence(array, random_num, length)
                    g1 = np.array(generate_gold_sequence(m1, m2, m))
                    #print(np.any(g1 != 0))
                    if np.any(g1 != 0):
                        #print(j)
                        break
                '''array = np.random.randint(0, 2, m) # generate m sequence
                random_num = np.random.randint(2, (len(array) + 1))
                m1 = generate_sequence(array, random_num, length)'''
                #print('m',m1)
                #w1 = np.random.random_sample((length)) - 0.5
                #w1 = w1 * 3.46     #范围[-1，1）
                w1 = range_Gaussian(lower, upper, mu, sigma, length)
                #w1 = np.random.triangular(-2.45, 0, 2.45,length)
                c = w1 + alpha_list[l] * g1
                pn_value.append(g1)
                w_value.append(c)
            #print(pn_value[0])
            #print(w_value[0])
            for i in range(len(pn_value)):
                for j in range(len(w_value)):
                    result = np.correlate( pn_value[i],w_value[j], mode='valid')
                    #print(result)
                    co_value.append(float(result))
                #print('co_value',co_value)
                total_co_value.append(co_value)
                co_value = []
            #print('total_co_value',total_co_value)
            correct_pr_each = []
            for i in range(len (total_co_value)):
                mean_value = sum(total_co_value[i]) / num
                total_mean_value.append(float(mean_value))
                correct_num = 0
                for j in range(len(total_co_value[i])):
                    if total_co_value[i][j] < mean_value:
                        correct_num +=1
                correct_pr_each.append(correct_num / (num-1))
            correct_pr = sum(correct_pr_each) / len(correct_pr_each)
            #print(correct_pr)
            total_correct_pr.append(correct_pr)
        print(total_correct_pr)
        correct_list.append( sum(total_correct_pr) / len(total_correct_pr))
    print(correct_list)
    print(x_list)
    #print(total_mean_value)

    '''# 画第1个图
    plt.figure(figsize=(12, 16))
    plt.subplot(331)

    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks, total_co_value[0])
    plt.axhline(y=total_mean_value[0], color="red")

    #画第二个图
    plt.subplot(332)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks, total_co_value[1])
    plt.axhline(y=total_mean_value[1], color="red")

    #画第三个图
    plt.subplot(333)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[2])
    plt.axhline(y=total_mean_value[2], color="red")

    # 画第四个图
    plt.subplot(334)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[3])
    plt.axhline(y=total_mean_value[3], color="red")
    plt.ylabel('Correlation coefficient',fontsize='18')

    # 画第五个图
    plt.subplot(335)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[4])
    plt.axhline(y=total_mean_value[4], color="red")

    # 画第六个图
    plt.subplot(336)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[5])
    plt.axhline(y=total_mean_value[5], color="red")

    # 画第七个图
    plt.subplot(337)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[6])
    plt.axhline(y=total_mean_value[6], color="red")

    # 画第八个图
    plt.subplot(338)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[7])
    plt.axhline(y=total_mean_value[7], color="red")

    plt.xlabel('Index of clients ',fontsize='18')
    # 画第九个图
    plt.subplot(339)
    my_x_ticks = np.arange(1, 10, 1)
    plt.xticks(my_x_ticks)
    plt.bar(my_x_ticks , total_co_value[8] )
    plt.axhline(y=total_mean_value[8], color="red")

    plt.suptitle('Gaussian  Distribution,m_sequence,m = 12, k = 0.39\n Theoretical correct probability 99%',fontsize='18')
    plt.show()'''


#同一个m序列和不同的权重加m序列  进行互相关性计算
'''if __name__ == '__main__':

    lower = -1
    upper = 1
    #mu = 0
    #sigma = 1
    #num = 1023
    k = 0.2
    m = 15
    length = 2 ** m - 1
    #gaussian_reusult = range_Gaussian(lower, upper, mu, sigma, num)
    #d = np.random.random_sample( (32767)) - 0.5
    d = np.random.triangular(-1, 0, 1, length)
    #d = d * 2
    print(d)
    array = np.random.randint(0, 2, m)
    random_num = np.random.randint(2, (len(array) + 1))
    m_1 = generate_sequence(array, random_num, length)
    #print('m_1',m_1)
    a = np.array(d)
    b = np.array(m_1)
    c = a + k*b
    c = list(c)
    print(c)
    value = []
    value.append(float(np.correlate(b,c, mode='valid')))
    print(value)
    for i in range(20):
        array = np.random.randint(0, 2, m)
        random_num = np.random.randint(2, (len(array) + 1))
        m2 = generate_sequence(array, random_num, length)
        #print(m2)
        result1 = np.correlate( c, m2, mode='valid')
        value.append(float(result1))
    print(value)
    mean_value = sum(value)/21
    print(mean_value)
    #value.append(mean_value)
    #plt.plot(np.arange(11),value)
    plt.bar( np.arange(21),value)
    plt.axhline(y=mean_value, color="red")
    #plt.hist(value, bins=11)
    plt.title('Triangular Distribution,m = 15, k = 0.2')
    plt.xlabel('Index of PN sequence ')
    plt.ylabel('Correlation coefficient')
    plt.grid()
    plt.show()'''






    # 随机产生序列和反馈线路
'''m = 8
    scale = 1
    length = 2 ** m - 1

    exp = 500
    c = []
    for i in range(exp):
        array = np.random.randint(0, 2, m)
        # print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        # print('random number =', random_num)

        result_data_0 = generate_sequence(array, random_num, length)
        #result_data_0 = sequence_scale(result_data_0, scale)
        # print(result_data_0)

        array = np.random.randint(0, 2, m)
        # print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        # print('random number =', random_num)

        result_data_1 = generate_sequence(array, random_num, length)
        #result_data_1 = sequence_scale(result_data_1, scale)
        # print(result_data_1)
        gold_sequence1 = generate_gold_sequence(result_data_0, result_data_1, m)


        array = np.random.randint(0, 2, m)
        # print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        # print('random number =', random_num)

        result_data_2 = generate_sequence(array, random_num, length)
        #result_data_2 = sequence_scale(result_data_0, scale)
        # print(result_data_0)

        array = np.random.randint(0, 2, m)
        # print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        # print('random number =', random_num)

        result_data_3 = generate_sequence(array, random_num, length)
        #result_data_3 = sequence_scale(result_data_1, scale)
        gold_sequence2 = generate_gold_sequence(result_data_2, result_data_3, m)

        c.append(int(np.correlate(gold_sequence1, gold_sequence2)))
    print(c)
    d = set(c)
    value, count_num = [], []
    for i in d:
        count = c.count(i)

        value.append(i)
        count_num.append(count)

    print(value)
    print(count_num)
    plt.bar(value, count_num)
    plt.xlim([-150, 150])
    plt.show()'''

'''m = 8
    scale = 1
    length = 2**m - 1

    exp=500
    c = []
    for i in range(exp):
        array = np.random.randint(0, 2, m)
        #print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        #print('random number =', random_num)

        result_data_0 = generate_sequence(array, random_num, length)
        #result_data_0 = sequence_scale(result_data_0, scale)
        #print(result_data_0)

        array = np.random.randint(0, 2, m)
        #print('random array =', array)
        random_num = np.random.randint(2, (len(array) + 1))
        #print('random number =', random_num)

        result_data_1 = generate_sequence(array, random_num, length)
        #result_data_1 = sequence_scale(result_data_1, scale)
        #print(result_data_1)
        c.append(int(np.correlate(result_data_0,result_data_1)))
    print(c)
    d = set(c)
    value, count_num = [], []
    for i in d:

        count =c.count(i)

        value.append(i)
        count_num.append(count)

    print(value)
    print(count_num)
    plt.bar(value, count_num)
    #plt.xlim([-150, 150])
    plt.show()'''


