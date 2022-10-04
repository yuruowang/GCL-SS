#!/usr/bin/python
import math
import sys
import numpy as np
import random
import torch as th
from sklearn.metrics.pairwise import cosine_similarity

class get_metapath:
    def __init__(self, unum, bnum, uage_num, uoc_num, mge_num):
        self.unum = unum + 1
        self.bnum = bnum + 1
        self.uage_num = uage_num + 1
        self.uoc_num = uoc_num + 1
        self.mge_num = mge_num + 1

        ub = self.load_ub('user_movie.txt')   #得到 944 * 1683的邻接矩阵
        # self.get_UMU(ub, 'metapaths/umu_w.txt')
        # self.get_UaU('user_age.txt', 'metapaths/uau_w.txt')
        # self.get_U_oc_U('user_occupation.txt', 'metapaths/uocu_cos_w.txt')
        # self.get_M_ge_M('movie_genre.txt', 'metapaths/mgem1_w.txt')
        self.get_UMgeMU(ub, 'movie_genre.txt','metapaths/umgemu_cos_1.txt')
        self.get_MUocUM(ub, 'user_occupation.txt', 'metapaths/muocum_cos_1.txt')
        #self.get_MUaUM(ub, 'data/user_age.txt', 'metapaths/muamu_.txt')

    def load_ub(self, ubfile):
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating, _ = line.strip().split('\t')
                ub[int(user)][int(item)] = 1
        return ub

    def get_UMU(self, ub, targetfile):
        print('UMU...')

        uu = ub.dot(ub.T) # user-user社交网络
        print(uu.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:    #user的节点i和节点j之间通过item相连，UMU返回自身的不算
                        w = uu[i][j] / math.sqrt((th.where(th.tensor(uu[i]) != 0)[0].shape[0]) * (th.where(th.tensor(uu[:, j]) != 0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) +'\t'+ str(round(w, 5))+ '\n')  # 写入ui  uj  评论过相同店铺的个数
                        total += 1
        print('total = ', total)

    def get_UaU(self, ucofile, targetfile):
        print('UaU...')
        uage = np.zeros((self.unum, self.uage_num))   #age分段，分成8段
        with open(ucofile, 'r') as infile:
            for line in infile.readlines():
                u, age = line.strip().split('\t')
                uage[int(u)][int(age)] = 1

        uu = uage.dot(uage.T)    #有相同年龄段的user相连
        uu_sim = cosine_similarity(uu)
        print(uu.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0])[1:]:
                for j in range(uu.shape[1])[1:]:
                    if uu[i][j] != 0 and i != j:
                        # w = uu[i][j]/math.sqrt((th.where(th.tensor(uu[i])!=0)[0].shape[0])*(th.where(th.tensor(uu[:,j])!=0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) +'\t'+ str(round(uu_sim[i][j], 5))+ '\n')
                        total += 1
        print('total = ', total)

    def get_U_oc_U(self, uocfile, targetfile):
        print('U_oc_U...')
        uoc = np.zeros((self.unum, self.uoc_num))
        with open(uocfile) as infile:
            for line in infile.readlines():
                u, ocu = line.strip().split('\t')
                uoc[int(u)][int(ocu)] = 1

        uu = uoc.dot(uoc.T)
        uu_sim = cosine_similarity(uu)
        print(uu.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0])[1:]:
                for j in range(uu.shape[1])[1:]:
                    if uu[i][j] != 0 and i != j:
                        # w = uu[i][j] / math.sqrt((th.where(th.tensor(uu[i,:]) != 0)[0].shape[0]) * (th.where(th.tensor(uu[:,j]) != 0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) +'\t'+ str(round(uu_sim[i][j], 5))+ '\n')
                        total += 1
        print('total = ', total)


    def get_M_ge_M(self, mgefile, targetfile):
        print('M_ge_M..')
        mge = np.zeros((self.bnum, self.mge_num))
        with open(mgefile) as infile:
            for line in infile.readlines():
                m, a = line.strip().split('\t')
                mge[int(m)][int(a)] = 1

        mm = mge.dot(mge.T)    #得到M-M
        mm_sim = cosine_similarity(mm)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        # w = mm[i][j] / math.sqrt((th.where(th.tensor(mm[i]) != 0)[0].shape[0]) * (th.where(th.tensor(mm[:,j]) != 0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) + '\t'+ str(round(mm_sim[i][j], 5)) + '\n')
                        total += 1
        print('total = ', total)

    def get_UMgeMU(self, ub, bcafile, targetfile):
        print('UBCaBU...')

        mge = np.zeros((self.bnum, self.mge_num))
        with open(bcafile, 'r') as infile:
            for line in infile.readlines():
                m, ge = line.strip().split('\t')
                mge[int(m)][int(ge)] = 1

        uu = ub.dot(mge).dot(mge.T).dot(ub.T)
        uu_sim = cosine_similarity(uu)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        w = uu[i][j] / math.sqrt((th.where(th.tensor(uu[i,:]) != 0)[0].shape[0]) * (th.where(th.tensor(uu[:,j]) != 0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(round(w, 5)) + '\n')
                        # outfile.write(str(i) + '\t' + str(j) + '\t' + str(round(uu_sim[i][j], 5)) + '\n')
                        total += 1
        print('total = ', total)

    def get_MUocUM(self, ub, ucofile, targetfile):
        print('MUocUM...')
        uoc = np.zeros((self.unum, self.uoc_num))
        with open(ucofile, 'r') as infile:
            for line in infile.readlines():
                u, oc = line.strip().split('\t')
                uoc[int(u)][int(oc)] = 1

        mm = ub.T.dot(uoc).dot(uoc.T).dot(ub)
        mm_sim = cosine_similarity(mm)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        w = mm[i][j] / math.sqrt((th.where(th.tensor(mm[i]) != 0)[0].shape[0]) * (th.where(th.tensor(mm[:, j]) != 0)[0].shape[0]))
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(round(w, 5)) +'\n')
                        total += 1
        print('total = ', total)

    def get_MUaUM(self, ub, uafile, targetfile):
        print('MUaUM...')

        ua = np.zeros((self.unum, self.uage_num))
        with open(uafile, 'r') as infile:
            for line in infile.readlines():
                u, a, _ = line.strip().split('\t')
                ua[int(u)][int(a)] = 1

        mm = ub.T.dot(ua).dot(ua.T).dot(ub)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\n')
                        total += 1
        print('total = ', total)



if __name__ == '__main__':
    # see __init__()
    get_metapath(unum=943, bnum=1682, uage_num=8, uoc_num=21, mge_num=18)
