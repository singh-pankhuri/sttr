from sttr.preprocess import *
from sttr.train import *
from sttr.load import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from sttr.models import *

# Update to change the dataset
dname = 'Foursquare'

# # Generate the cleaned npy files from the raw data
# preprocess(dname)

# Generate the initial embeddings as pickle file from above npy files
process_traj(dname)

# Train the model with the hyper-parameters
part = 100
emb_dim = 512
dropout = 0
num_neg = 10
lr = 3e-3
epochs = 100

file = open('./data/sttr_files/' + dname + '_data.pkl', 'rb') 
file_data = joblib.load(file)
[trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), torch.FloatTensor(mat2t), torch.LongTensor(lens)

trajs, mat1, mat2t, labels, lens = trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]
ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
stan = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=emb_dim, ex=ex, dropout=dropout)

num_params = 0
for param in stan.parameters():
    num_params += param.numel()

records = {'epoch': [], 'recall_valid': [], 'recall_test': [], 'ndcg_valid': [], 'ndcg_test': []}
start = time.time()

trainer = Trainer(stan, records, load, trajs, mat1, mat2t, labels, lens, mat2s, num_neg, lr, epochs)
trainer.train(part, start, dname, emb_dim)