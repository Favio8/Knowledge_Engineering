import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.nn.functional as F
import re
import csv


def movie_dict():
    csv_file = "data/movie/movies.csv"
    i = 0
    ids_name_movie_dict = {}
    name_ids_movie_dict = {}
    with open(csv_file, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            ids_name_movie_dict[int(row[0])] = row[1]
            name_ids_movie_dict[row[1]] = int(row[0])
        return ids_name_movie_dict, name_ids_movie_dict


def user_dict():
    csv_file = "data/movie/name_ids.csv"
    i = 0
    ids_name_user_dict = {}
    name_ids_user_dict = {}
    with open(csv_file, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            ids_name_user_dict[int(row[0])] = row[1]
            name_ids_user_dict[row[1]] = int(row[0])

        return ids_name_user_dict, name_ids_user_dict


def get_ids(name_ids_user_dict, name_ids_movie_dict, data_loader):
    d = {
        'users': {
        },
        'items': {
        }
    }
    info = {}
    for k, v in d.items():
        if k == 'users':
            info[k] = input(f"{k}（请输入用户名，如：王宇）:")
        else:
            info[k] = input(f"{k}（请输入电影名称，用逗号分隔，如：Toy Story (1995), Jumanji (1995）:")

    user = info['users'].split(',')
    item = info['items'].split(',')

    user_ids = []
    item_ids = []
    for i in user:
        user_ids.append(name_ids_user_dict[i])

    for i in item:
        item_ids.append(name_ids_movie_dict[i])

    if len(user_ids) != len(item_ids):
        user_ids = random.choices(user_ids, k=len(item))

    user_ids = data_loader.oldindex_newindex_user(user_ids)
    item_ids = data_loader.oldindex_newindex_entity(item_ids)

    return user_ids, item_ids


def get_user(name_ids_user_dict, data_loader):
    d = {
        'users': ''
    }
    info = {}
    for k, v in d.items():
        info[k] = input(f"{k}（请输入用户名，如：张伟）:")

    user = info['users'].split(',')
    user_ids = name_ids_user_dict[user[0]]
    user_ids = data_loader.oldindex_newindex_user([user_ids])

    return user, user_ids


def prediction(item_ids, user_ids, net, device, ids_name_user_dict, ids_name_movie_dict, data_loader):
    item_ids = torch.LongTensor(item_ids)
    user_ids = torch.LongTensor(user_ids)
    item_ids = item_ids.to(device)
    user_ids = user_ids.to(device)
    outputs = net(user_ids, item_ids)
    name = []
    user = []
    for i in item_ids:
        i = data_loader.newindex_oldindex_entity(int(i))
        name.append(ids_name_movie_dict[i])

    for i in user_ids:
        i = data_loader.newindex_oldindex_user([int(i)])
        user.append(ids_name_user_dict[int(i)])
    interested = []
    uninterested = []
    score = []
    for j in outputs.tolist():
        a = round(j*100, 2)
        score.append(a)

    for i in range(len(score)):
        names = name[i] + '(' + str(score[i]) + '%' + ')'
        # 设置感兴趣阈值
        if outputs[i] > 0.5:
            interested.append(names)
        else:
            uninterested.append(names)

    interested_movie = ''
    uninterested_movie = ''
    for i in interested:
        interested_movie = interested_movie + str(i) + ' | '
    interested_movie = interested_movie[:-3]

    for i in uninterested:
        uninterested_movie = uninterested_movie + str(i) + ' | '
    uninterested_movie = uninterested_movie[:-3]

    # 补全输出部分代码，效果：用户xxx对电影1|电影2感兴趣，对电影3不感兴趣
    if len(interested) > 0 and len(uninterested) > 0:
        print(f"用户{user[0]}对{interested_movie}感兴趣，对{uninterested_movie}不感兴趣")

    elif len(interested) > 0:
        print(f"用户{user[0]}对{interested_movie}感兴趣")

    else:
        print(f"用户{user[0]}对所有电影都不感兴趣")


def get_OneHot_movie_ids(data_loader):
    csv_file = "data/movie/ratings.csv"
    OneHot_movie_ids = []
    with open(csv_file, "r", newline="", encoding='utf-8') as file1:
        csv_reader = csv.reader(file1)
        i = 0
        for row in csv_reader:
            if row[1] not in OneHot_movie_ids and i > 0:
                OneHot_movie_ids.append(row[1])
                i = 0
            i = i + 1

    item_ids = data_loader.oldindex_newindex_entity(OneHot_movie_ids)
    return item_ids

def prediction_user_items(name_ids_user_dict, name_ids_movie_dict, ids_name_user_dict, ids_name_movie_dict, data_loader,
                          net, device):
    user_ids, item_ids = get_ids(name_ids_user_dict, name_ids_movie_dict, data_loader)
    prediction(item_ids, user_ids, net, device, ids_name_user_dict, ids_name_movie_dict, data_loader)


def RS_movie_list(data_loader, name_ids_user_dict, ids_name_movie_dict, device, net):
    n = int(input("请输入拟推荐电影个数："))
    OneHot_movie_ids = get_OneHot_movie_ids(data_loader)
    user, user_ids = get_user(name_ids_user_dict, data_loader)

    user_ids = random.choices(user_ids, k=len(OneHot_movie_ids))
    item_ids = torch.LongTensor(OneHot_movie_ids)
    user_ids = torch.LongTensor(user_ids)
    item_ids = item_ids.to(device)
    user_ids = user_ids.to(device)

    outputs = net(user_ids, item_ids)
    # out = F.softmax(outputs, dim=0)
    # rank
    a, b = torch.topk(outputs, n, dim=0, largest=True, sorted=True)
    items = []
    for i in b:
        items.append(item_ids[int(i)])
    movie_list = []
    for i in items:
        movie_list.append(ids_name_movie_dict[data_loader.newindex_oldindex_entity(int(i))])

    scores = []
    for j in a.tolist():
        score = round(j, 4)
        scores.append(score)

    outputs = ''
    score_ids = 0
    for i in movie_list:
        outputs = (outputs + str(i) + '(' + str(scores[score_ids] * 100) + '%' + ')' + ' | ')
        score_ids += 1
    outputs = outputs[:-3]
    # 补全输出部分代码，实现效果：向用户xxx推荐xx个电影：电影1(概率)，电影2(概率)，电影3(概率)
    print(f"向用户{user[0]}推荐{n}个电影：{outputs}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    # parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    #parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    #parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    #parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    # args = parser.parse_args(['--l2_weight', '1e-4'])
    args = parser.parse_args()
    data_loader = DataLoader(args.dataset)
    data_loader.get_encoders()
    ids_name_user_dict, name_ids_user_dict = user_dict()
    ids_name_movie_dict, name_ids_movie_dict = movie_dict()
    num_user, num_entity, num_relation = data_loader.get_num()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kg = data_loader.load_kg()
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)

    net_weight_path = "./KGCN_1hot.pth"
    net.load_state_dict(torch.load(net_weight_path, map_location=device))
    net.eval()
    while (1):
        print("推荐模式分为两种:"
              "1.用户对指定电影的感兴趣程度;"
              "2.为指定用户推荐电影."
              "您可通过输入1或2进行选择")
        pattern = input("请输入推荐模式:")
        if pattern == '1':
            prediction_user_items(name_ids_user_dict, name_ids_movie_dict, ids_name_user_dict, ids_name_movie_dict,
                                  data_loader, net, device)
        if pattern == '2':
            RS_movie_list(data_loader, name_ids_user_dict, ids_name_movie_dict, device, net)

        print("是否继续推荐："
              "请输入Y或者N")
        go_on = input("是否继续：")

        if go_on == "N":
            break


if __name__ == '__main__':
    main()
