# %%
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import random
import torch
import numpy as np

class Metabook(Dataset):
    def __init__(self, partition='train', test_way=None, path=None):
        super(Metabook, self).__init__()
        #self.dataset_path = args.data_root
        self.partition = partition
        # key (num_id): 1  x num_user_feature_dim
        dataset_path = "data"
        if partition == 'train':
            self.state = 'user_warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'user_warm_state'
                elif test_way == 'new_user_valid':
                    self.state = 'user_cold_state_valid'
                elif test_way == 'new_user_test':
                    self.state = 'user_cold_state_test'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            # str inside
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        with open("{}/{}.json".format(dataset_path, "book_profile"), encoding="utf-8") as f:
            # str inside
            self.book_dict = json.loads(f.read())
        with open("{}/{}.json".format(dataset_path, "user_profile"), encoding="utf-8") as f:
            self.user_dict = json.loads(f.read())

    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_book_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_book_len))
        # random.seed(53)
        if self.state=="warm_state":
            random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
        total_xs = None
        for m_id in tmp_x:
            m_id = int(m_id)
            tmp_x_converted = np.expand_dims(np.concatenate((self.book_dict[str(m_id)], self.user_dict[str(u_id)]), 0),0)
            try:
                total_xs = np.concatenate((total_xs, tmp_x_converted), 0)
            except:
                total_xs = tmp_x_converted

        # query_x_app = None
        # support_items = np.array([int(m_id) for m_id in tmp_x[indices[:-10]]])
        # test_items = np.array([int(m_id) for m_id in tmp_x[indices[-10:]]])
        # for m_id in tmp_x[indices[-10:]]:
        #     m_id = int(m_id)
        #     u_id = int(user_id)
        #     tmp_x_converted = np.expand_dims(np.concatenate((self.book_dict[str(m_id)], self.user_dict[str(u_id)]), 0),0)
        #     try:
        #         query_x_app = np.concatenate((query_x_app, tmp_x_converted), 0)
        #     except:
        #         query_x_app = tmp_x_converted
        total_xs = torch.tensor(total_xs).long()
        # query_x_app = torch.tensor(query_x_app).long()
        total_ys = torch.FloatTensor(tmp_y)
        # query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        # print(support_x_app)
        # print(support_y_app.view(-1,1))
        # print(query_x_app)
        # print(query_y_app.view(-1,1))
        # print(user_id)
        # print(test_items);exit()
        # user_id and tmp_x is str
        return total_xs, total_ys.view(-1,1)
        
    def __len__(self):
        return len(self.final_index)


import pickle
import os

data_dir = 'data'

for mode in ['train', 'valid', 'test']:

    total_xs = []
    total_ys = []
    ages = []

    if mode == 'train':
        dataset = Metabook()
    elif mode == 'valid':
        dataset = Metabook(partition='test', test_way="new_user_valid")
    else:
        dataset = Metabook(partition='test', test_way="new_user_test")
    
    dataset_sz = len(dataset)
    for i in range(dataset_sz):
        total_x, total_y  = dataset[i]
        total_xs.append(total_x)
        total_ys.append(total_y)

        ages.append(total_x[0][4])


    if not os.path.exists(f"./{data_dir}_processed"):
        os.mkdir(f"./{data_dir}_processed") 
    
    print(len(total_xs), len(total_ys))
    
    pickle.dump(total_xs, open(f"./{data_dir}_processed/{mode}_total_xs", 'wb'))
    pickle.dump(total_ys, open(f"./{data_dir}_processed/{mode}_total_ys", 'wb'))
    pickle.dump(ages, open(f"./{data_dir}_processed/{mode}_gender", 'wb'))