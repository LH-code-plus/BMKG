import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dgl.nn import SAGEConv
import itertools
##  读取图数据
(g,), _ = dgl.load_graphs('../data/graph/ckg_graph.dgl')

u, v = g.edges()


##  划分训练集和测试集
eids = np.arange(g.num_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.15)
train_size = g.num_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# 重新构建邻接矩阵
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.num_nodes(), g.num_nodes()))

# 检查 adj 矩阵的形状
print("adj matrix shape:", adj.shape)
dense_adj_matrix = g.adjacency_matrix().to_dense()

# 创建一个DataFrame
adj_df = pd.DataFrame(dense_adj_matrix.numpy(), columns=range(g.num_nodes()), index=range(g.num_nodes()))

# 计算负样本矩阵
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())

# Find all negative edges and split them for training and testing
#adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

#adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.num_edges())
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]],
)

train_g = dgl.remove_edges(g, eids[:test_size])


##  定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")
        self.conv3 = SAGEConv(h_feats, h_feats, "mean")
        self.conv4 = SAGEConv(h_feats, h_feats, "mean")
        self.conv5 = SAGEConv(h_feats, h_feats, "mean")
        self.conv6 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        h = F.relu(h)
        h = self.conv6(g, h)
        return h

##  定义MLP预测器
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, h_feats)
        self.W3 = nn.Linear(h_feats, h_feats)
        self.W4 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        h = F.relu(self.W1(h))
        h = F.relu(self.W2(h))
        h = F.relu(self.W3(h))  # 添加新层
        return {"score": self.W4(h).squeeze(1)}
        #return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

##  实例化嵌入模型和预测器
model = GraphSAGE(train_g.ndata["h"].shape[1], 16)
pred = MLPPredictor(16)

optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.001
)
all_logits = []

import matplotlib.pyplot as plt
epoch_losses = []
### 训练模型
for e in range(200):
    # forward
    h = model(train_g, train_g.ndata["h"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_losses.append(loss.item())
     
    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

model.eval()
pred.eval()
torch.save(model, "data/ckg_model.pth")
torch.save(pred, "data/ckg_pred.pth")
##  计算测试集的AUC
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))
    
    # 将正负样本得分合并
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))])
    
    # 设置阈值，将得分转换为预测标签
    threshold = 0.0  # 可以调整阈值
    predictions = (scores > threshold).float()
    
    # 计算准确率、召回率和F1值
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1-score:", f1)

plt.figure(figsize=(10, 5))  # 设置图表大小
plt.plot(epoch_losses, label='Training Loss')  # 绘制损失曲线
plt.title('Training Loss over Epochs')  # 设置图表标题
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Loss')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.savefig('../data/photo/ckg_training_loss.png', format='png')  # 保存为PNG格式的文件

