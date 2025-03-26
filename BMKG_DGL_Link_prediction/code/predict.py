##  导入相关库
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

##  读取图数据
(g,), _ = dgl.load_graphs('../data/graph/ckg_graph.dgl')

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
        
        
##  定义链接预测方法
def predict_link(model, predictor, graph, node_a, node_b, threshold=1.75):
    """
    利用训练好的模型预测两个节点之间是否存在关系。

    Parameters
    ----------
    model : nn.Module
        训练好的图神经网络模型。
    predictor : nn.Module
        训练好的链接预测器。
    graph : dgl.DGLGraph
        包含节点特征的图。
    node_a : int
        第一个节点的索引。
    node_b : int
        第二个节点的索引。
    threshold : float, optional
        判断是否存在关系的阈值,默认值为0.5。

    Returns
    -------
    bool
        如果预测的分数大于阈值,则返回True,表示存在关系;否则返回False。
    # """
    with torch.no_grad():
        # 获取图的节点特征
        features = graph.ndata["h"]
        
        # 计算节点的嵌入表示
        h = model(graph, features)
        
        # 创建包含预测节点对的子图
        pos_g = dgl.graph(([node_a], [node_b]), num_nodes=graph.num_nodes())
        
        # 计算边的分数
        pos_score = predictor(pos_g, h)
        print(pos_score)
        print(pos_score.shape[0])
        # 根据阈值判断是否存在关系
        return pos_score.item() > threshold


model = GraphSAGE(g.ndata["h"].shape[1], 16)
pred = MLPPredictor(16)
model = torch.load('data/ckg_model.pth')
pred = torch.load('data/ckg_pred.pth')
# 示例用法
import pandas as pd
# 读取CSV文件

Meta = 'EC_number'
React = 'rid'
Metapath = "../data/csv/CKG_data/final_protein_gid.csv"
Reactpath = "../data/csv/final_reactions_gid.csv"
# 定义一个函数，输入名字，返回对应的id
def get_id_by_name(name, path):
    df = pd.read_csv(path)
    # 根据名字列名和id列名查找对应的id
    # 假设名字列的列名是'Name'，id列的列名是'ID'
    try:
        # 使用loc查找名字对应的id
        if path == Metapath:
            id_value = df.loc[df['EC_number'] == name, 'g_id'].values[0]
            return id_value
        elif path == Reactpath:
            id_value = df.loc[df['rid'] == name, 'g_id'].values[0]
            return id_value
    except IndexError:
        # 如果名字不存在，返回None或适当的错误信息
        return None

# 假设我们要查找的名字是'John Doe'
Metaname = '3.6.1.1'
##input("请输入代谢物名称：")
##Reactname = input("请输入反应名称：")
Metaid = get_id_by_name(Metaname, Metapath)
# ##Reactid = get_id_by_name(Reactname, Reactpath)
# if Metaid is not None:
#     print(f"名字 '{Metaid}' 对应的id是：{id}")
# else:
#     print(f"名字 '{Metaid}' 在文件中不存在。")
    
# if Reactid is not None:
#     print(f"名字 '{Reactid}' 对应的id是：{id}")
# else:
#     print(f"名字 '{Reactid}' 在文件中不存在。")
cout = 0
for Reactid in range(714, 2122):
    exists = predict_link(model, pred, g, Metaid, Reactid, 1)
    print(f"预测节点 {Metaid} 和 {Reactid} 之间 {'存在' if exists else '不存在'} 关系")
    if exists:
        cout += 1
print("cout:", cout)

# if id is not None:
#     print(f"名字 '{name_to_find}' 对应的id是：{id}")
# else:
#     print(f"名字 '{name_to_find}' 在文件中不存在。")
# node_a = int(input("请输入代谢物节点索引："))
# node_b = int(input("请输入反应节点索引："))
# exists = predict_link(model, pred, g, node_a, node_b, 1.75)
# print(f"预测节点 {node_a} 和 {node_b} 之间 {'存在' if exists else '不存在'} 关系")

##copper (Cu+2) transport via diffusion (extracellular to periplasm)