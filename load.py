import numpy as np
import torch
import haversine as hs 
import joblib
from torch.nn.utils.rnn import pad_sequence

max_len = 100 

def create_pickle(dataname):
    data = np.load('./data/sttr_files/' + dataname + '.npy')
    poi = np.load('./data/sttr_files/' + dataname + '_POI.npy')
    num_user = len(set(data[:, 0])) 
    data_user = data[:, 0] 
    trajs, labels, mat1, mat2t, lens = [], [], [], [], []
    u_max, l_max = np.max(data[:, 0]), np.max(data[:, 1])

    for u_id in range(1,num_user+1):
        init_mat1 = np.zeros((max_len, max_len, 2))
        init_mat2t = np.zeros((max_len, max_len)) 
        user_traj = data[np.where(data_user == u_id)].copy()

        if len(user_traj) > max_len + 1:
            user_traj = user_traj[-max_len-1:] 

        # spatial and temporal intervals
        user_len = len(user_traj[:-1]) 
        user_mat1 = rst_mat1(user_traj[:-1], poi)
        user_mat2t = rt_mat2t(user_traj[:, 2]) 
        init_mat1[0:user_len, 0:user_len] = user_mat1
        init_mat2t[0:user_len, 0:user_len] = user_mat2t

        trajs.append(torch.LongTensor(user_traj)[:-1])
        mat1.append(init_mat1) 
        mat2t.append(init_mat2t)
        labels.append(torch.LongTensor(user_traj[1:, 1]))
        lens.append(user_len-2)

    # padding zero to the vacancies in the right
    mat2s = rs_mat2s(poi, l_max)
    zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
    trajs, mat1, mat2t, labels, lens = zipped
    trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
    trajs = pad_sequence(trajs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    data = [trajs, np.array(mat1), mat2s, np.array(mat2t), labels, np.array(lens), u_max, l_max]
    del zipped, trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max

    data_pkl = './data/sttr_files/' + dataname + '_data.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump(data, pkl)

def rst_mat1(traj, poi):
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]
            mat[i, j, 0] = hs.haversine((poi_item[1],poi_item[2]),(poi_term[1],poi_term[2]))
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat 

def rs_mat2s(poi, l_max):
    candidate_loc = np.linspace(1, l_max, l_max)
    mat = np.zeros((l_max, l_max))
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1] 
            mat[i, j] = hs.haversine((poi1[1],poi1[2]), (poi2[1],poi2[2]))
    return mat

def rt_mat2t(traj_time): 
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time): 
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]): 
            mat[i-1, j] = np.abs(item - term)
    return mat 