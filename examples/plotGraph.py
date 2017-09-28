#!/usr/bin/env

import offline_experiment as offline
import matplotlib.pyplot as plt
import numpy as np


def display(acc_list, list_name):
    for x in xrange(0, acc_list.__len__()):
        print '\t{0} = {1}% to interval [{2}Hz~{3}Hz]'.format(list_name, acc_list[x][0], acc_list[x][2], acc_list[x][1])


x = [20, 30, 40, 60, 80]


def displayGraph(acc_, graph_name_):
    ACC = 0
    HIGH_CUT = 1
    LOWER_CUT = 2

    #x = np.arange(1, acc_.__len__() + 1)
    y = []
    for i in xrange(0, acc_.__len__()):
        y.append(acc_[i][ACC])

    plt.plot(x, y)
    plt.box('off')
    plt.show()


if __name__ == '__main__':
    SUBJECT_A = 0
    SUBJECT_B = 1

    acc = offline.TestWith(high_cut_=38, lower_cut_=0.1, sub_sampling_=x, resolution_=20)

# acc = []
# acc.append(((100, 50, 0), (10, 50, 0)))
# acc.append(((90, 40, 10), (20, 40, 10)))
# acc.append(((80, 30, 20), (30, 30, 20)))

    acc_A = []
    acc_B = []

    for i in xrange(0, acc.__len__()):
        acc_A.append(acc[i][SUBJECT_A])
        acc_B.append(acc[i][SUBJECT_B])

#
#    thefile = open('list.txt', 'w')
#    thefile.write("%s\t Subject A")
#    for item in acc_A:
#        thefile.write("%s\n" % item)

#    thefile.write("%s\t Subject B")
#    for item in acc_B:
#        thefile.write("%s\n" % item)
#
#    thefile.close()

    print
    displayGraph(acc_=acc_A, graph_name_="(Acc X Passa-Faixa) - Sujeito A")
    displayGraph(acc_=acc_B, graph_name_="(Acc X Passa-Faixa) - Sujeito B")
