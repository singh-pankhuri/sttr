from load import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *

def calculate_acc(prob, label, f2):
  acc_train = [0, 0, 0, 0]
  recall_train = [0, 0, 0, 0]
  ndcg_train = [0, 0, 0, 0]

  for i, k in enumerate([1, 5, 10, 20]):
      # topk_batch (N, k)
      _, topk_predict_batch = torch.topk(prob, k=k)
      for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
          # topk_predict (k)
          if to_npy(label)[j] in topk_predict:
              acc_train[i] += 1
              recall_train[i] += 1

          # ndcg
          idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
          dcg = np.sum([1 / np.log2(rank + 2) if loc in topk_predict else 0 for rank, loc in enumerate(to_npy(topk_predict_batch)[j])])
          ndcg_train[i] += dcg / idcg

          if k == 20:
              f2.write('next loc:{}'.format(label[j]))
              f2.write('rec loc:{}'.format(topk_predict_batch))
              f2.write('\n')

  return np.array(acc_train), np.array(recall_train), np.array(ndcg_train)


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

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class DataSet(data.Dataset):#数据集
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)#拷贝到GPU
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


class Trainer:
    def __init__(self, model, record, load, trajs, mat1, mat2t, labels, lens, mat2s):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 1 # N = 1
        self.learning_rate = 3e-3
        self.num_epoch = 100
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens
        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, part, start, dname):
        # set optimizer 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
        f=open((dname + "_result.txt"),"w",encoding='utf-8')
        f2 = open("rec_point.txt", "w", encoding='utf-8')
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            recall_valid, recall_test = [0, 0, 0, 0], [0, 0, 0, 0]
            ndcg_valid, ndcg_test = [0, 0, 0, 0], [0, 0, 0, 0]

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)!!!

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        # acc_valid += calculate_acc(prob, train_label, f2)
                        recall_valid += calculate_acc(prob, train_label, f2)[1]
                        ndcg_valid = calculate_acc(prob, train_label, f2)[2]

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        # acc_test += calculate_acc(prob, train_label, f2)
                        recall_test += calculate_acc(prob, train_label, f2)[1]
                        ndcg_test = calculate_acc(prob, train_label, f2)[2]

                bar.update(self.batch_size)
            bar.close()

            recall_valid = np.array(recall_valid) / valid_size
            print('epoch:{}, time:{}, valid_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_valid))
            # with open("result.txt","w") as f:
            f.write('epoch:{}, time:{}, valid_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_valid))
            f.write('\n')
            print('epoch:{}, time:{}, valid_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_valid))
            f.write('epoch:{}, time:{}, valid_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_valid))
            f.write('\n')

            recall_test = np.array(recall_test) / test_size
            print('epoch:{}, time:{}, test_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_test))
            # with open("result.txt", "w") as f:
            f.write('epoch:{}, time:{}, test_recall:{}'.format(self.start_epoch + t, time.time() - start, recall_test))
            f.write('\n')
            print('epoch:{}, time:{}, valid_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_test))
            f.write('epoch:{}, time:{}, valid_ndcg:{}'.format(self.start_epoch + t, time.time() - start, ndcg_test))
            f.write('\n')

            self.records['recall_valid'].append(recall_valid)
            self.records['recall_test'].append(recall_test)
            self.records['ndcg_valid'].append(ndcg_valid)
            self.records['ndcg_test'].append(ndcg_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'best_stan_win_1000_' + dname + '.pth')
        f.close()