"""
实验脚本：测试不同聚合器类型(aggregator)对AUC的影响
"""
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from model import KGCN
from data_loader import DataLoader
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 指定字体文件路径（确保路径正确）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
my_font = FontProperties(fname=font_path)

# 超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')

args = parser.parse_args(['--l2_weight', '1e-4'])

# 数据集准备
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

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], 
                                                    test_size=1 - args.ratio,
                                                    shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

num_user, num_entity, num_relation = data_loader.get_num()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCELoss()

# 测试不同的聚合器类型
aggregators = ['sum', 'concat', 'neighbor']
results = {'aggregator': [], 'auc': [], 'f1': []}

print("=" * 50)
print("实验：测试不同聚合器类型对模型性能的影响")
print("=" * 50)

for aggregator in aggregators:
    print(f"\n正在测试聚合器类型: {aggregator}...")
    args.aggregator = aggregator
    
    # 创建模型
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    
    # 训练模型
    best_auc = 0.0
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        net.train()
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证
        net.eval()
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
                    outputs_binary = (outputs >= 0.5).float()
                    total_f1 += f1_score(labels.cpu().detach().numpy(), outputs_binary.cpu().detach().numpy())
                except ValueError:
                    pass
            
            auc = total_roc / len(test_loader)
            f1 = total_f1 / len(test_loader)
            if auc > best_auc:
                best_auc = auc
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch [{epoch+1}/{args.n_epochs}], Loss: {running_loss/len(train_loader):.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')
    
    results['aggregator'].append(aggregator)
    results['auc'].append(best_auc)
    results['f1'].append(f1)
    print(f"聚合器 {aggregator} 的最佳 AUC: {best_auc:.4f}")

# 绘制结果
x_pos = np.arange(len(aggregators))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, results['auc'], width, label='AUC', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, results['f1'], width, label='F1 Score', alpha=0.8)

ax.set_xlabel('聚合器类型 (Aggregator Type)', fontsize=12, fontproperties=my_font)
ax.set_ylabel('评价指标', fontsize=12, fontproperties=my_font)
ax.set_title('不同聚合器类型对模型性能的影响', fontsize=14, fontproperties=my_font)
ax.set_xticks(x_pos)
ax.set_xticklabels(aggregators)
ax.legend(fontsize=10, prop=my_font)
ax.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('experiment_aggregator_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("实验结果总结：")
print("=" * 50)
for i in range(len(results['aggregator'])):
    print(f"聚合器={results['aggregator'][i]:8s}, AUC={results['auc'][i]:.4f}, F1={results['f1'][i]:.4f}")
print("\n实验结果已保存至: experiment_aggregator_results.png")

