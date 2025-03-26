from rxnfp.models import load_pretrained_rxn_fingerprinter
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载预训练模型
fingerprinter = load_pretrained_rxn_fingerprinter()

# 示例反应式
reaction_smarts = '[C:1][H:2]>>[C:1][OH:2]'

# 生成指纹
fingerprint_vector = fingerprinter.embed([reaction_smarts])[0]

print(f"生成的指纹向量: {fingerprint_vector}")