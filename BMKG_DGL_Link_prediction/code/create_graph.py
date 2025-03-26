import os
import dgl
import pandas as pd
import torch
os.environ["DGLBACKEND"] = "pytorch"

# 读取数据
metabolite_df = pd.read_csv("../data/csv/CKG_data/final_protein.csv")
reaction_df = pd.read_csv("../data/csv/CKG_data/final_reactions.csv")
participates_in_df = pd.read_csv("../data/csv/CKG_data/catalyzes_data.csv")

df = pd.concat([metabolite_df, reaction_df], ignore_index=True)

# 创建节点字典
node_id_map = {}
current_id = 0

for _, row in df.iterrows():
    node_id_map[row['id']] = current_id
    current_id += 1

# 构建边
src = [node_id_map[sourceId] for sourceId in participates_in_df['sourceId']]
dst = [node_id_map[targetId] for targetId in participates_in_df['targetId']]

etype = ['CATALYZES'] * len(src)

# 创建 DGL 图
g = dgl.heterograph({('Metabolite', 'CATALYZES', 'Metabolite'): (src, dst)})

# 添加节点特征
#metabolite_features = torch.tensor(df[['node2Vec1', 'node2Vec2', 'node2Vec3', 'node2Vec4', 'node2Vec5', 'node2Vec6', 'node2Vec7', 'node2Vec8', 'node2Vec9', 'node2Vec10', 'node2Vec11', 'node2Vec12', 'node2Vec13', 'node2Vec14', 'node2Vec15', 'node2Vec16', 'node2Vec17', 'node2Vec18', 'node2Vec19', 'node2Vec20', 'node2Vec21', 'node2Vec22', 'node2Vec23', 'node2Vec24', 'node2Vec25', 'node2Vec26', 'node2Vec27', 'node2Vec28', 'node2Vec29', 'node2Vec30', 'node2Vec31', 'node2Vec32', 'node2Vec33', 'node2Vec34', 'node2Vec35', 'node2Vec36', 'node2Vec37', 'node2Vec38', 'node2Vec39', 'node2Vec40', 'node2Vec41', 'node2Vec42', 'node2Vec43', 'node2Vec44', 'node2Vec45', 'node2Vec46', 'node2Vec47', 'node2Vec48', 'node2Vec49', 'node2Vec50', 'node2Vec51', 'node2Vec52', 'node2Vec53', 'node2Vec54', 'node2Vec55', 'node2Vec56', 'node2Vec57', 'node2Vec58', 'node2Vec59', 'node2Vec60', 'node2Vec61', 'node2Vec62', 'node2Vec63', 'node2Vec64', 'node2Vec65', 'node2Vec66', 'node2Vec67', 'node2Vec68', 'node2Vec69', 'node2Vec70', 'node2Vec71', 'node2Vec72', 'node2Vec73', 'node2Vec74', 'node2Vec75', 'node2Vec76', 'node2Vec77', 'node2Vec78', 'node2Vec79', 'node2Vec80', 'node2Vec81', 'node2Vec82', 'node2Vec83', 'node2Vec84', 'node2Vec85', 'node2Vec86', 'node2Vec87', 'node2Vec88', 'node2Vec89', 'node2Vec90', 'node2Vec91', 'node2Vec92', 'node2Vec93', 'node2Vec94', 'node2Vec95', 'node2Vec96', 'node2Vec97', 'node2Vec98', 'node2Vec99', 'node2Vec100', 'node2Vec101', 'node2Vec102', 'node2Vec103', 'node2Vec104', 'node2Vec105', 'node2Vec106', 'node2Vec107', 'node2Vec108', 'node2Vec109', 'node2Vec110', 'node2Vec111', 'node2Vec112', 'node2Vec113', 'node2Vec114', 'node2Vec115', 'node2Vec116', 'node2Vec117', 'node2Vec118', 'node2Vec119', 'node2Vec120', 'node2Vec121', 'node2Vec122', 'node2Vec123', 'node2Vec124', 'node2Vec125', 'node2Vec126', 'node2Vec127', 'node2Vec128', 'cnnVec1', 'cnnVec2', 'cnnVec3', 'cnnVec4', 'cnnVec5', 'cnnVec6', 'cnnVec7', 'cnnVec8', 'cnnVec9', 'cnnVec10', 'cnnVec11', 'cnnVec12', 'cnnVec13', 'cnnVec14', 'cnnVec15', 'cnnVec16', 'cnnVec17', 'cnnVec18', 'cnnVec19', 'cnnVec20', 'cnnVec21', 'cnnVec22', 'cnnVec23', 'cnnVec24', 'cnnVec25', 'cnnVec26', 'cnnVec27', 'cnnVec28', 'cnnVec29', 'cnnVec30', 'cnnVec31', 'cnnVec32', 'cnnVec33', 'cnnVec34', 'cnnVec35', 'cnnVec36', 'cnnVec37', 'cnnVec38', 'cnnVec39', 'cnnVec40', 'cnnVec41', 'cnnVec42', 'cnnVec43', 'cnnVec44', 'cnnVec45', 'cnnVec46', 'cnnVec47', 'cnnVec48', 'cnnVec49', 'cnnVec50', 'cnnVec51', 'cnnVec52', 'cnnVec53', 'cnnVec54', 'cnnVec55', 'cnnVec56', 'cnnVec57', 'cnnVec58', 'cnnVec59', 'cnnVec60', 'cnnVec61', 'cnnVec62', 'cnnVec63', 'cnnVec64', 'cnnVec65', 'cnnVec66', 'cnnVec67', 'cnnVec68', 'cnnVec69', 'cnnVec70', 'cnnVec71', 'cnnVec72', 'cnnVec73', 'cnnVec74', 'cnnVec75', 'cnnVec76', 'cnnVec77', 'cnnVec78', 'cnnVec79', 'cnnVec80', 'cnnVec81', 'cnnVec82', 'cnnVec83', 'cnnVec84', 'cnnVec85', 'cnnVec86', 'cnnVec87', 'cnnVec88', 'cnnVec89', 'cnnVec90', 'cnnVec91', 'cnnVec92', 'cnnVec93', 'cnnVec94', 'cnnVec95', 'cnnVec96', 'cnnVec97', 'cnnVec98', 'cnnVec99', 'cnnVec100', 'cnnVec101', 'cnnVec102', 'cnnVec103', 'cnnVec104', 'cnnVec105', 'cnnVec106', 'cnnVec107', 'cnnVec108', 'cnnVec109', 'cnnVec110', 'cnnVec111', 'cnnVec112', 'cnnVec113', 'cnnVec114', 'cnnVec115', 'cnnVec116', 'cnnVec117', 'cnnVec118', 'cnnVec119', 'cnnVec120', 'cnnVec121', 'cnnVec122', 'cnnVec123', 'cnnVec124', 'cnnVec125', 'cnnVec126', 'cnnVec127', 'cnnVec128']].values)

metabolite_features = torch.tensor(df[['cnnVec1', 'cnnVec2', 'cnnVec3', 'cnnVec4', 'cnnVec5', 'cnnVec6', 'cnnVec7', 'cnnVec8', 'cnnVec9', 'cnnVec10', 'cnnVec11', 'cnnVec12', 'cnnVec13', 'cnnVec14', 'cnnVec15', 'cnnVec16', 'cnnVec17', 'cnnVec18', 'cnnVec19', 'cnnVec20', 'cnnVec21', 'cnnVec22', 'cnnVec23', 'cnnVec24', 'cnnVec25', 'cnnVec26', 'cnnVec27', 'cnnVec28', 'cnnVec29', 'cnnVec30', 'cnnVec31', 'cnnVec32', 'cnnVec33', 'cnnVec34', 'cnnVec35', 'cnnVec36', 'cnnVec37', 'cnnVec38', 'cnnVec39', 'cnnVec40', 'cnnVec41', 'cnnVec42', 'cnnVec43', 'cnnVec44', 'cnnVec45', 'cnnVec46', 'cnnVec47', 'cnnVec48', 'cnnVec49', 'cnnVec50', 'cnnVec51', 'cnnVec52', 'cnnVec53', 'cnnVec54', 'cnnVec55', 'cnnVec56', 'cnnVec57', 'cnnVec58', 'cnnVec59', 'cnnVec60', 'cnnVec61', 'cnnVec62', 'cnnVec63', 'cnnVec64', 'cnnVec65', 'cnnVec66', 'cnnVec67', 'cnnVec68', 'cnnVec69', 'cnnVec70', 'cnnVec71', 'cnnVec72', 'cnnVec73', 'cnnVec74', 'cnnVec75', 'cnnVec76', 'cnnVec77', 'cnnVec78', 'cnnVec79', 'cnnVec80', 'cnnVec81', 'cnnVec82', 'cnnVec83', 'cnnVec84', 'cnnVec85', 'cnnVec86', 'cnnVec87', 'cnnVec88', 'cnnVec89', 'cnnVec90', 'cnnVec91', 'cnnVec92', 'cnnVec93', 'cnnVec94', 'cnnVec95', 'cnnVec96', 'cnnVec97', 'cnnVec98', 'cnnVec99', 'cnnVec100', 'cnnVec101', 'cnnVec102', 'cnnVec103', 'cnnVec104', 'cnnVec105', 'cnnVec106', 'cnnVec107', 'cnnVec108', 'cnnVec109', 'cnnVec110', 'cnnVec111', 'cnnVec112', 'cnnVec113', 'cnnVec114', 'cnnVec115', 'cnnVec116', 'cnnVec117', 'cnnVec118', 'cnnVec119', 'cnnVec120', 'cnnVec121', 'cnnVec122', 'cnnVec123', 'cnnVec124', 'cnnVec125', 'cnnVec126', 'cnnVec127', 'cnnVec128']].values)

print(metabolite_features.dtype)

# 将特征分配给对应的节点类型
g.nodes['Metabolite'].data['h'] = metabolite_features.float()
print(g)

dgl.save_graphs("../data/graph/ckg_graph.dgl", g)
print("Graph saved successfully.")
