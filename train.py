import os
import random
import time

import numpy as np
import torch.utils.data as data
from torch import optim
from tqdm import tqdm

from .load import *
from .models import *


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)

class Trainer:
    def __init__(self, model, record, load, trajs, mat1, mat2t, labels, lens, mat2s, num_neg, lr, epochs):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = 1
        self.num_neg = num_neg #10
        self.interval = 1000
        self.batch_size = 1 # N = 1
        self.learning_rate = lr  #3e-3
        self.num_epoch = epochs #100
        self.threshold = 0  # 0 if not update

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = trajs, mat1, mat2s, mat2t, labels, lens
        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, part, start, dname, emb_dim):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        if not os.path.exists("results/"):
            os.makedirs("results/")

        f = open(("results/" +dname + "_sttr.txt"),"w",encoding='utf-8')
        f2 = open((dname +"_rec_point.txt"), "w", encoding='utf-8')

        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            recall_valid, recall_test = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
            ndcg_valid, ndcg_test = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item
                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)!!!

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        recall_valid += evaluate_mets(prob, train_label, f2)[0]
                        ndcg_valid += evaluate_mets(prob, train_label, f2)[1]

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size += person_input.shape[0]
                        recall_test += evaluate_mets(prob, train_label, f2)[0]
                        ndcg_test += evaluate_mets(prob, train_label, f2)[1]

                bar.update(self.batch_size)
            bar.close()

            np.set_printoptions(precision=6)
            recall_valid = np.array(recall_valid) / valid_size
            ndcg_valid = np.array(ndcg_valid) / valid_size
            f.write('epoch:{}, time:{}, valid_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_valid))
            f.write('\n')
            f.write('epoch:{}, time:{}, valid_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_valid))
            f.write('\n')

            recall_test = np.array(recall_test) / test_size
            ndcg_test = np.array(ndcg_test) / test_size
            f.write('epoch:{}, time:{}, test_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_test))
            f.write('\n')
            f.write('epoch:{}, time:{}, test_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_test))
            f.write('\n')

            self.records['recall_valid'].append(recall_valid)
            self.records['recall_test'].append(recall_test)
            self.records['ndcg_valid'].append(ndcg_valid)
            self.records['ndcg_test'].append(ndcg_test)
            self.records['epoch'].append(self.start_epoch + t)

        f.close()

def evaluate_mets(prob, label, f2): 
    recall = [0.0, 0.0, 0.0, 0.0]
    ndcg = [0.0, 0.0, 0.0, 0.0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(convert_npy(topk_predict_batch)):
          target = convert_npy(label)[j]
          if target in topk_predict:
            recall[i] = 1.0
            rank_list = list(topk_predict)
            rank_index = rank_list.index(target)
            dcg = 5.0 / np.log2(rank_index + 2)
            idcg = 5.0 / np.log2(2)
            ndcg[i] = dcg / idcg
    return np.array(recall), np.array(ndcg) 

def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)