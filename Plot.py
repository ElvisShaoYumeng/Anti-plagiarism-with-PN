import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from matplotlib.pyplot import MultipleLocator
from PN_sequence import normal_distribution

f = csv.reader(open('./SimulationData/Gaussian_approximation/new_fed_20UEs_epoch100_exp50_num100000_1625956024.csv', 'r'))
for i, rows in enumerate(f):  # 0 is x_value, 1 is count, 2 is mean, 3 is sigma, 7 is number of experiments, 8 is correlation list
    if i == 2:
        mean = rows
    if i == 3:
        sigma = rows
    if i == 7:
        num_exp = rows
    if i == 8:
        correlation = rows

s = re.findall(r'\-?\d+\.?\d*e?\-?\d*', mean[0])
theo_mu = float(s[0])
s = re.findall(r'\-?\d+\.?\d*e?\-?\d*', sigma[0])
theo_sigma = float(s[0])
s = re.findall(r'\-?\d+\.?\d*e?\-?\d*', num_exp[0])
num_w = float(s[0])
s = re.findall(r'\-?\d+\.?\d*e?\-?\d*', correlation[0])
correlation[0] = float(s[0])
s = re.findall(r'\-?\d+\.?\d*e?\-?\d*', correlation[-1])
correlation[-1] = float(s[0])
for i in range(len(correlation)):
    correlation[i] = float(correlation[i])

correlation_list = correlation

#interval = 60
#scale = 1.28
interval = 50
scale = 1.07
#theo_mu = -0.0001244068525504889
#theo_sigma = 4.759870584728691
w_length = 25450
#num_w = 10000

x_value = []
count = [0 for i in range(interval)]
cross_max = max(correlation_list)
cross_min = min(correlation_list)
gap = (cross_max - cross_min) / interval
for i in range(interval):
    for j in range(len(correlation_list)):
        if correlation_list[j] >= (cross_min + i * gap) and correlation_list[j] <= (cross_min + (i + 1) * gap):
            count[i] += 1
for i in range(len(count)):
    count[i] = count[i] / num_w * scale
print(count)
for i in range(interval):
    x_value.append(cross_min + i * gap + gap / 2)
#print(x_value)
plt.plot(x_value, count, 'b', label='Cross-correlation')

### plot theoretical gaussian distribution
x1 = np.linspace(theo_mu - 6 * theo_sigma, theo_mu + 6 * theo_sigma, 100)
y1 = normal_distribution(x1, theo_mu, theo_sigma)
plt.plot(x1, y1, 'r', label='Gaussian approximation')
plt.title(' length = {}, num_exp = {}'.format(w_length, num_w))

plt.legend()
plt.show()


# 自定义interval
'''interval = 100
scale = 0.74
num_w = 10000
x_value =[]
theo_mu = -5.4910506368886554e-05
theo_sigma = 4.1769660553840895
w_length = 25450

count = [0 for i in range(interval)]
cross_max = max(correlation_list)
cross_min = min(correlation_list)
gap = (cross_max - cross_min) / interval
for i in range(interval):
    for j in range(len(correlation_list)):
        if correlation_list[j] >= (cross_min + i * gap) and correlation_list[j] <= (cross_min + (i + 1) * gap):
            count[i] += 1
for i in range(len(count)):
    count[i] = count[i] * scale / num_w
for i in range(interval):
    x_value.append(cross_min + i * gap + gap / 2)

plt.plot(x_value, count, 'b', label='Cross-correlation')
plt.show()'''

# 默认interval
'''x_value = []
cross_min = -16.47362884067479
gap = 1.0103342707783667
interval = 40
for i in range(interval):
    x_value.append(cross_min + i * gap + gap / 2)
print(x_value)

scale = 0.95
theo_mu = -0.0001244068525504889
theo_sigma = 4.759870584728691
w_length = 25450
num_w = 10000

for i in range(len(count_list)):
    count_list[i] = count_list[i] * scale / num_w

plt.plot(x_value, count_list, 'b', label='Cross-correlation')

### plot theoretical gaussian distribution
x1 = np.linspace(theo_mu - 6 * theo_sigma, theo_mu + 6 * theo_sigma, 100)
y1 = normal_distribution(x1, theo_mu, theo_sigma)
plt.plot(x1, y1, 'r', label='Gaussian approximation')
plt.title(' length = {}, num_exp = {}'.format(w_length, num_w))

plt.legend()
plt.show()'''