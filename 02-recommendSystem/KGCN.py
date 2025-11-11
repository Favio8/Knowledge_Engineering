import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import csv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=500, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--every_epoch_test', type=bool, default=True)
parser.add_argument('--recall_topk', type=bool, default=True)
parser.add_argument('--top_k', type=list, default=[1, 2, 5, 10, 20, 50, 100])

args = parser.parse_args(['--l2_weight', '1e-4'])
save_path = 'KGCN_1hot.pth'
data_loader = DataLoader(args.dataset)
df_dataset = data_loader.load_dataset()
kg = data_loader.load_kg()


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label


# train test split
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio,
                                                    shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

# prepare network, loss function, optimizer
num_user, num_entity, num_relation = data_loader.get_num()
user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)


# 冻结参数
# unfreeze_layers = ['aggregator.']
# for name, param in net.named_parameters():
#     print(name, param.size())
#
# print("*" * 30)
# print('\n')
#
# for name, param in net.named_parameters():
#     param.requires_grad = False
#     for ele in unfreeze_layers:
#         if ele in name:
#             param.requires_grad = True
#             break
#
# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())

# 定义Loss和优化器
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.l2_weight)
print('device: ', device)
CUDA_LAUNCH_BLOCKING = "1"

# train
loss_list = []
test_loss_list = []
auc_score_list = []
f1_score_list = []

best_acc = 0.0
for epoch in range(args.n_epochs):
    running_loss = 0.0
    if args.train:
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # print train loss per every 10 epochs
        loss_list.append(running_loss / len(train_loader))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        # evaluate per every epoch
        if args.every_epoch_test:
            with torch.no_grad():
                test_loss = 0
                total_roc = 0
                total_f1 = 0.0
                for user_ids, item_ids, labels in test_loader:
                    user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                    outputs = net(user_ids, item_ids)
                    test_loss += criterion(outputs, labels).item()
                    try:
                        total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                        outputs[outputs >= 0.5] = 1
                        outputs[outputs < 0.5] = 0
                        total_f1 += f1_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    except ValueError:
                        pass
                test_loss_list.append(test_loss / len(test_loader))
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
                auc_score_list.append(total_roc / len(test_loader))
                f1_score_list.append(total_f1 / len(test_loader))
            if auc_score_list[epoch] > best_acc:
                best_acc = auc_score_list[epoch]
                torch.save(net.state_dict(), save_path)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f'  ✓ 保存最佳模型 (AUC: {best_acc:.4f})')
# 绘图
if args.every_epoch_test:
    print('best_acc: ', best_acc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    ax1.plot(test_loss_list, label='test_loss')
    ax1.plot(loss_list, label='train_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(auc_score_list, label='auc_score')
    ax2.plot(f1_score_list, label='f1_score')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax2.legend()
    plt.tight_layout()
    # plt.savefig('train_result_2hot')
    plt.show()

# test
if args.test:
    with torch.no_grad():
        test_loss = 0
        total_roc = 0
        total_f1 = 0.0
        for user_ids, item_ids, labels in test_loader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            outputs = net(user_ids, item_ids)
            test_loss += criterion(outputs, labels).item()
            try:
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                total_f1 += f1_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            except ValueError:
                pass
        print('test_loss: ', test_loss / len(test_loader), 'RUC: ', total_roc / len(test_loader), 'F1: ',
              total_f1 / len(test_loader))

# 计算racall@K
if args.recall_topk:
    # 获取每个用户groundTruth列表(评分大于4的电影)
    def get_groundTruth():
        ratings_file = "./data/movie/ratings.csv"
        items_list = []
        OneHot_movie_ids = []
        user_list = []
        with open(ratings_file, "r", newline="") as file1:
            csv_reader = csv.reader(file1)
            i = 0
            for row in csv_reader:
                if float(row[2]) >= 4:
                    if int(row[0]) > len(items_list):
                        items_list.append([])
                    items_list[int(row[0]) - 1].append(row[1])

                if row[1] not in OneHot_movie_ids and i > 0:
                    OneHot_movie_ids.append(row[1])
                    i = 0
                i = i + 1

            for i in range(len(items_list)):
                user_list.append(i)

        item_ids = data_loader.oldindex_newindex_entity(OneHot_movie_ids)
        return items_list, item_ids, user_list

    def recall_topk(data_loader, net, k):
        items_list, item_ids, user_list = get_groundTruth()
        recall = 0
        item_ids = torch.LongTensor(item_ids)
        for i in user_list:
            user_ids = random.choices([i], k=len(item_ids))
            movie_list = []
            user_ids = torch.LongTensor(user_ids)
            item_ids = item_ids.to(device)
            user_ids = user_ids.to(device)


            outputs = net(user_ids, item_ids)
            out = F.softmax(outputs, dim=0)
            a, b = torch.topk(out, k, dim=0, largest=True, sorted=True)

            for _ in b:
                c = item_ids[int(_)]
                movie_list.append(data_loader.newindex_oldindex_entity(int(c)))

            correct_count = 0
            for j in movie_list:
                if str(j) in items_list[i]:
                    correct_count += 1
            recall += correct_count/len(items_list[i])


        recall_topk = recall/len(items_list)
        return recall_topk

    # 绘图
    recall_list = []
    for i in args.top_k:
        recall = recall_topk(data_loader, net, i)
        recall_list.append(recall)
        print(f'recall@{i}:', recall)
    plt.plot(args.top_k,recall_list)
    plt.xlabel('K')
    plt.ylabel('RECALL@K')
    plt.tight_layout()
    plt.savefig('RECALL@K')
    plt.show()

