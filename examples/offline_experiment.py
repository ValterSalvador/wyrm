#!/usr/bin/env python

from __future__ import division
from wyrm import io

import logging
import wyrm.processing as proc
import numpy as np

if __name__ == '__main__':
    print("Iniciado Experimento Offline...")

TRAIN_DATA_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Train.mat'
TEST_DATA_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Test.mat'
TRAIN_DATA_B = 'data/BCI_Comp_III_Wads_2004/Subject_B_Train.mat'
TEST_DATA_B = 'data/BCI_Comp_III_Wads_2004/Subject_B_Test.mat'

TRUE_LABELS_A = "WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU"

TRUE_LABELS_B = "MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR"

STIMULUS_CODE = {
    # cols from left to right
    1: "agmsy5",
    2: "bhntz6",
    3: "ciou17",
    4: "djpv28",
    5: "ekqw39",
    6: "flrx4_",
    # rows from top to bottom
    7: "abcdef",
    8: "ghijkl",
    9: "mnopqr",
    10: "stuvwx",
    11: "yz1234",
    12: "56789_"
}

MARKER_DEF_TRAIN = {'target': ['target'], 'nontarget': ['nontarget']}
MARKER_DEF_TEST = {i: [i] for i in STIMULUS_CODE.values()}

JUMPING_MEANS_INTERVALS = [150, 220], [200, 260], [310, 360], [550, 660]  # 91%

SEG_IVAL = [0, 700]

# Variaveis principais
HIGH_CUT = 38
LOWER_CUT = 0.1
SUBSAMPLING = 20


logging.basicConfig(format='%(relativeCreated)10.0f %(threadName)-10s %(name)-10s %(levelname)8s %(message)s', level=logging.NOTSET)
logger = logging.getLogger(__name__)


def train(filename_, high_cut_=HIGH_CUT, lower_cut_=LOWER_CUT, sub_sampling_=SUBSAMPLING):
    cnt = io.load_bcicomp3_ds2(filename_)

    fs_n = cnt.fs / 2

    b, a = proc.signal.butter(5, [high_cut_ / fs_n], btype='low')
    cnt = proc.lfilter(cnt, b, a)

    b, a = proc.signal.butter(5, [lower_cut_ / fs_n], btype='high')
    cnt = proc.lfilter(cnt, b, a)
    if __name__ == '__main__':
        print ("Filtragem aplicada em [{} Hz ~ {} Hz]".format(lower_cut_, high_cut_))

    cnt = proc.subsample(cnt, sub_sampling_)
    if __name__ == '__main__':
        print ("Sub-amostragem em {} Hz".format(sub_sampling_))

    epo = proc.segment_dat(cnt, MARKER_DEF_TRAIN, SEG_IVAL)
    if __name__ == '__main__':
        print ("Dados segmentados em intervalos de [{} ~ {}]".format(SEG_IVAL[0], SEG_IVAL[1]))

    # fv = proc.jumping_means(epo, JUMPING_MEANS_INTERVALS)
    fv = proc.create_feature_vectors(epo)

    if __name__ == '__main__':
        print("Iniciando treinamento da LDA...")
    cfy = proc.lda_train(fv)
    if __name__ == '__main__':
        print("Treinamento concluido!")
    return cfy


def offline_experiment(filename_, cfy_, true_labels_, high_cut_=HIGH_CUT, lower_cut_=LOWER_CUT, sub_sampling_=SUBSAMPLING):
    if __name__ == '__main__':
        print("\n")
    cnt = io.load_bcicomp3_ds2(filename_)

    fs_n = cnt.fs / 2

    b, a = proc.signal.butter(5, [high_cut_ / fs_n], btype='low')
    cnt = proc.filtfilt(cnt, b, a)

    b, a = proc.signal.butter(5, [lower_cut_ / fs_n], btype='high')
    cnt = proc.filtfilt(cnt, b, a)

    cnt = proc.subsample(cnt, sub_sampling_)

    epo = proc.segment_dat(cnt, MARKER_DEF_TEST, SEG_IVAL)

    # fv = proc.jumping_means(epo, JUMPING_MEANS_INTERVALS)
    fv = proc.create_feature_vectors(epo)

    lda_out = proc.lda_apply(fv, cfy_)
    markers = [fv.class_names[cls_idx] for cls_idx in fv.axes[0]]
    result = zip(markers, lda_out)
    endresult = []
    markers_processed = 0
    letter_prob = {i: 0 for i in 'abcdefghijklmnopqrstuvwxyz123456789_'}
    for s, score in result:
        if markers_processed == 180:
            endresult.append(sorted(letter_prob.items(), key=lambda x: x[1])[-1][0])
            letter_prob = {i: 0 for i in 'abcdefghijklmnopqrstuvwxyz123456789_'}
            markers_processed = 0
        for letter in s:
            letter_prob[letter] += score
        markers_processed += 1
    if __name__ == '__main__':
        print('Letras Encontradas-: %s' % "".join(endresult))
        print('Letras Corretas----: %s' % true_labels_)
    acc = np.count_nonzero(np.array(endresult) == np.array(list(true_labels_.lower()[:len(endresult)]))) / len(endresult)
    if __name__ == '__main__':
        print("Acertividade Final : %d" % (acc * 100))
    return (acc, high_cut_, lower_cut_)


def discover_maximun(high_cut_, lower_cut_, resolution_):
    _i = 1
    _h = high_cut_
    _l = lower_cut_
    while(_h > _l and _h != _l):
        _h = _h - resolution_
        _l = _l + resolution_
        _i += 1
    return _i


def TestWith(high_cut_, lower_cut_, sub_sampling_, resolution_=0.2):
    acc_list_A_B = []
    if(resolution_ <= 0):
        resolution_ = 0.2

    maximun_iterations = discover_maximun(high_cut_, lower_cut_, resolution_)
    print 'Max Iteration = {}'.format(maximun_iterations)
    i = 0
    # while (higmaskh_cut_ > lower_cut_ and high_cut_ != lower_cut_):
    while (i < sub_sampling_.__len__()):
        print 'Actual Iteration = {} with sample = {}'.format(i, sub_sampling_[i])

        matriz_de_pesos_A = train(TRAIN_DATA_A, high_cut_, lower_cut_, sub_sampling_[i])
        acc_A = offline_experiment(TEST_DATA_A, matriz_de_pesos_A, TRUE_LABELS_A, high_cut_, lower_cut_, sub_sampling_[i])

        matriz_de_pesos_B = train(TRAIN_DATA_B, high_cut_, lower_cut_, sub_sampling_[i])
        acc_B = offline_experiment(TEST_DATA_B, matriz_de_pesos_B, TRUE_LABELS_B, high_cut_, lower_cut_, sub_sampling_[i])

        acc_list_A_B.append((acc_A, acc_B))
        # high_cut_ = high_cut_ - resolution_
        # lower_cut_ = lower_cut_ + resolution_
        i += 1
    print 'Max Iteration = {}'.format(maximun_iterations)
    print 'Last Iteration = {}'.format(i)
    print 'Acc List Length = {}'.format(acc_list_A_B.__len__())
    return acc_list_A_B


if __name__ == '__main__':
    print("\n\t")
    logger.info('Treinando com dados do sujeito A...')
    matriz_de_pesos_A = train(TRAIN_DATA_A)

    print("\n\t")
    logger.info('Treinando com dados do sujeito B...')
    matriz_de_pesos_B = train(TRAIN_DATA_B)

    print("\n\n\n\n\n")

    print('Realizando analize offline da amostra do sujeito A...')
    offline_experiment(TEST_DATA_A, matriz_de_pesos_A, TRUE_LABELS_A)

    print('Realizando analize offline da amostra do sujeito B...')
    offline_experiment(TEST_DATA_B, matriz_de_pesos_B, TRUE_LABELS_B)

