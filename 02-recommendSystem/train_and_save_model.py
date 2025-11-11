"""
训练并保存KGCN模型
使用最佳参数配置训练模型，并保存以供API使用
"""
import pandas as pd
import numpy as np
import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from model import KGCN
from data_loader import DataLoader
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 指定字体文件路径（确保路径正确）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
my_font = FontProperties(fname=font_path)

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

args = parser.parse_args(['--l2_weight', '1e-4'])
save_path = 'KGCN_1hot.pth'

print("=" * 60)
print(" " * 15 + "KGCN模型训练程序")
print("=" * 60)
print("\n模型配置:")
print(f"  - 数据集: {args.dataset}")
print(f"  - 聚合器类型: {args.aggregator}")
print(f"  - 训练轮数: {args.n_epochs}")
print(f"  - 嵌入维度: {args.dim}")
print(f"  - 邻域采样大小: {args.neighbor_sample_size}")
print(f"  - 批次大小: {args.batch_size}")
print(f"  - 学习率: {args.lr}")
print(f"  - L2正则化: {args.l2_weight}")
print("=" * 60)

# 加载数据
print("\n正在加载数据...")
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

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], 
                                                    test_size=1 - args.ratio,
                                                    shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 准备模型
num_user, num_entity, num_relation = data_loader.get_num()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)

# 训练模型
loss_list = []
test_loss_list = []
auc_score_list = []
f1_score_list = []
best_auc = 0.0

print("\n开始训练...\n")

for epoch in range(args.n_epochs):
    # 训练阶段
    net.train()
    running_loss = 0.0
    for i, (user_ids, item_ids, labels) in enumerate(train_loader):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(user_ids, item_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    loss_list.append(avg_train_loss)
    
    # 验证阶段
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
        
        avg_test_loss = test_loss / len(test_loader)
        auc = total_roc / len(test_loader)
        f1 = total_f1 / len(test_loader)
        
        test_loss_list.append(avg_test_loss)
        auc_score_list.append(auc)
        f1_score_list.append(f1)
        
        # 每10轮或最后一轮打印
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.n_epochs:
            print(f'Epoch [{epoch+1:3d}/{args.n_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Test Loss: {avg_test_loss:.4f} | '
                  f'AUC: {auc:.4f} | '
                  f'F1: {f1:.4f}')
        
        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(net.state_dict(), save_path)
            print(f'  ✓ 保存最佳模型 (AUC: {best_auc:.4f})')

print("\n" + "=" * 60)
print(f"训练完成！最佳AUC: {best_auc:.4f}")
print(f"模型已保存至: {save_path}")
print("=" * 60)

# 绘制训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 损失曲线
ax1.plot(range(1, args.n_epochs+1), loss_list, label='训练损失', marker='o', markersize=4)
ax1.plot(range(1, args.n_epochs+1), test_loss_list, label='测试损失', marker='s', markersize=4)
ax1.set_xlabel('训练轮数 (Epoch)', fontproperties=my_font)
ax1.set_ylabel('损失 (Loss)', fontproperties=my_font)
ax1.set_title('训练和测试损失曲线', fontproperties=my_font)
ax1.legend(prop=my_font)
ax1.grid(True, alpha=0.3)

# 评价指标曲线
ax2.plot(range(1, args.n_epochs+1), auc_score_list, label='AUC', marker='o', markersize=4)
ax2.plot(range(1, args.n_epochs+1), f1_score_list, label='F1 Score', marker='s', markersize=4)
ax2.set_xlabel('训练轮数 (Epoch)', fontproperties=my_font)
ax2.set_ylabel('评价指标', fontproperties=my_font)
ax2.set_title('AUC和F1分数曲线', fontproperties=my_font)
ax2.legend(prop=my_font)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print(f"\n训练曲线已保存至: training_curves.png")
plt.show()

