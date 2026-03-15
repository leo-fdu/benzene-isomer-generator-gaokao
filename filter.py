from rdkit import Chem
from rdkit.Chem import Descriptors
from math import gcd
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
CountSpec = Union[None, int, str]


def filter_stable_molecules(smiles_list):
    """
    输入 SMILES 字符串列表，返回剔除不稳定结构和非法 SMILES 后的新列表。
    过滤规则：
    1. 偕二醇
    2. 烯醇
    3. 炔醇
    4. 累积二烯烃
    5. 杂原子-杂原子相连（但硝基中的 N-O 允许）
    """
    unstable_patterns = {
        # 1. 偕二醇：同一个饱和碳上连两个羟基
        "偕二醇": "[CX4]([OX2H])([OX2H])",

        # 2. 烯醇：C=C-OH，要求 OH 直接连在烯键碳上
        "烯醇": "[CX3]=[CX3][OX2H]",

        # 3. 炔醇：C#C-OH，要求 OH 直接连在炔键碳上
        "炔醇": "[CX2]#[CX2][OX2H]",

        # 4. 累积二烯烃：C=C=C
        "累积二烯烃": "[CX3]=[CX3]=[CX3]",
    }

    patterns = {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in unstable_patterns.items()
    }

    stable_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        # 非法 SMILES 直接剔除
        if mol is None:
            continue

        is_stable = True

        # 先检查常规不稳定片段
        for pattern in patterns.values():
            if mol.HasSubstructMatch(pattern):
                is_stable = False
                break

        if not is_stable:
            continue

        # 再检查杂原子-杂原子相连，但硝基中的 N-O 键放行
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            z1 = a1.GetAtomicNum()
            z2 = a2.GetAtomicNum()

            # 杂原子定义：不是 C 也不是 H
            hetero1 = z1 not in (1, 6)
            hetero2 = z2 not in (1, 6)

            if not (hetero1 and hetero2):
                continue

            # 特判：如果这是硝基中的 N-O 键，则允许
            atoms = (a1, a2)
            n_atom = next((atom for atom in atoms if atom.GetAtomicNum() == 7), None)
            o_atom = next((atom for atom in atoms if atom.GetAtomicNum() == 8), None)

            is_nitro_bond = False
            if n_atom is not None and o_atom is not None:
                oxygen_neighbors = [nbr for nbr in n_atom.GetNeighbors() if nbr.GetAtomicNum() == 8]

                # 硝基的粗判定：N 连两个 O
                if len(oxygen_neighbors) == 2:
                    is_nitro_bond = True

            if is_nitro_bond:
                continue

            # 其余杂原子-杂原子相连一律排除
            is_stable = False
            break

        if is_stable:
            stable_smiles.append(smi)

    return stable_smiles
def count_chiral_centers(smi):
    """
    输入 SMILES，返回该分子中手性中心的个数。
    """
    mol = Chem.MolFromSmiles(smi) 
    # 关键步骤：必须感知立体化学
    # includeUnassigned=True 意味着即使没有标记 R/S，只要是潜在的手性点也会计入
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, force=True)
    return len(centers)

from rdkit import Chem
from collections import Counter

def get_total_unique_h_count(smi):
    """1. 返回分子中不等同氢的总个数"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    mol = Chem.AddHs(mol) # 必须补氢，否则无法计算氢的对称性
    
    # 获取所有原子的对称等级
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    
    # 提取所有氢原子的对称等级
    h_ranks = [ranks[i] for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == 'H']
    
    # 对称等级的种类数即为不等同氢的个数
    return len(set(h_ranks))

def get_h_ratio(smi):
    """2. 返回不等同氢的比例（由大到小排列）"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return []
    mol = Chem.AddHs(mol)
    
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    h_ranks = [ranks[i] for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == 'H']
    
    # 统计每种等价氢的数量
    counts = Counter(h_ranks)
    # 获取数量列表并降序排列
    ratio = sorted(counts.values(), reverse=True)
    return ratio


def normalize_ratio(values: List[int]) -> List[int]:
    """
    将正整数列表化为最简整数比，并按从大到小排序后返回。

    非法输入（空列表除外）返回空列表，以避免上层逻辑崩溃。
    """
    if not values:
        return []
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        return []

    divisor = reduce(gcd, values)
    if divisor <= 0:
        return []

    normalized = [value // divisor for value in values]
    return sorted(normalized, reverse=True)


def get_normalized_h_ratio(smi: str) -> List[int]:
    """
    返回分子的氢谱峰面积最简比，并按从大到小排序。
    """
    return normalize_ratio(get_h_ratio(smi))

def get_benzene_unique_h_count(smi):
    """3. 返回苯环上不等同氢的个数"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    
    # 匹配苯环模式
    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    matches = mol.GetSubstructMatches(benzene_pattern)
    if not matches: return 0
    
    mol = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    
    # 假设只处理第一个匹配到的苯环
    benzene_indices = set(matches[0])
    benzene_h_ranks = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == 'H':
            # 检查这个氢原子是否连在苯环的碳上
            neighbor = atom.GetNeighbors()[0]
            if neighbor.GetIdx() in benzene_indices:
                benzene_h_ranks.append(ranks[i])
                
    return len(set(benzene_h_ranks))

def get_benzene_substituent_count(smi):
    """4. 返回苯环的取代数（有多少个环上位置被非氢原子占据）"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    
    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    matches = mol.GetSubstructMatches(benzene_pattern)
    if not matches: return 0
    
    ring_indices = set(matches[0])
    sub_count = 0
    
    # 遍历苯环上的每一个碳原子
    for idx in ring_indices:
        atom = mol.GetAtomWithIdx(idx)
        # 统计其邻居中不在环上的非氢原子个数
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_indices and neighbor.GetSymbol() != 'H':
                sub_count += 1
                break # 一个碳原子即便连多个侧链原子（如叔丁基），通常也只计为1个取代位
                
    return sub_count

from rdkit import Chem

# =============================================================================
# 核心计数逻辑：使用高度精确的 SMARTS 表达式
# =============================================================================

def count_ester_groups(smi):
    """1. 酯基计数: [C;X3](=O)-O-[C;X4,a] 排除羧基"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    # 匹配：碳连双键氧，且连一个氧，该氧再连碳
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
    return len(mol.GetSubstructMatches(pattern))

def count_carboxylic_acids(smi):
    """2. 羧基计数: [C;X3](=O)-[O;H1]"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    return len(mol.GetSubstructMatches(pattern))

def count_carbonyl_groups(smi):
    """3. 羰基计数 (仅限酮): 排除醛、羧酸、酯、酰胺"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    # 匹配：C=O 且碳原子两边都连碳
    pattern = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")
    return len(mol.GetSubstructMatches(pattern))

def count_aldehydes(smi):
    """4. 醛基计数: [CX3H1](=O)"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    pattern = Chem.MolFromSmarts("[CX3H1](=O)")
    return len(mol.GetSubstructMatches(pattern))

def count_non_phenolic_hydroxyls(smi):
    """5. 非酚羟基 (醇羟基) 计数: 排除酚和羧酸"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    # 匹配：连在脂肪族碳上的 OH
    pattern = Chem.MolFromSmarts("[#6;!$(C=O);!$(c)]-[OX2H1]")
    return len(mol.GetSubstructMatches(pattern))

def count_phenolic_hydroxyls(smi):
    """6. 酚羟基计数: 连在芳香环上的 OH"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    pattern = Chem.MolFromSmarts("[OX2H1]-c")
    return len(mol.GetSubstructMatches(pattern))

def count_ether_bonds(smi):
    """7. 醚键计数: R-O-R 排除酯、酸"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    # 匹配：氧连接两个碳，且这两个碳都不是羰基碳
    pattern = Chem.MolFromSmarts("[#6;!$(C=O)]-[OD2]-[#6;!$(C=O)]")
    return len(mol.GetSubstructMatches(pattern))


# 新增函数：酰胺基计数
def count_amide_groups(smi: str) -> int:
    """酰胺基计数：C(=O)-N，包含一级/二级/三级酰胺。"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    return len(mol.GetSubstructMatches(patt))


# 新增函数：碳碳双键计数
def count_cc_double_bonds(smi: str) -> int:
    """碳碳双键计数：只统计 C=C，不把 C=O 等算进去。"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt = Chem.MolFromSmarts("[C]=[C]")
    return len(mol.GetSubstructMatches(patt))


# 新增函数：碳碳三键计数
def count_cc_triple_bonds(smi: str) -> int:
    """碳碳三键计数：只统计 C#C，不把腈基 C#N 算进去。"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt = Chem.MolFromSmarts("[C]#[C]")
    return len(mol.GetSubstructMatches(patt))


def count_nitro_groups(smi: str) -> int:
    """硝基计数：兼容 RDKit 常见的带电写法 [N+](=O)[O-]。"""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt = Chem.MolFromSmarts("[$([NX3](=O)=O),$([N+](=O)[O-])]")
    return len(mol.GetSubstructMatches(patt))


def count_nitriles(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt = Chem.MolFromSmarts("[C]#[N]")
    return len(mol.GetSubstructMatches(patt))


def count_amines(smi: str) -> int:
    """
    氨基计数：统计胺类氮中心（伯/仲/叔胺，包含芳胺），
    排除酰胺、硝基、腈、亚胺等相似结构。
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return 0
    patt_amine = Chem.MolFromSmarts(
        "[NX3;!$([N+](=O)[O-]);!$([NX3]-[CX3](=O));!$(N#[C,N]);!$(N=C)]"
    )
    return len(mol.GetSubstructMatches(patt_amine))

def compute_features(smi: str) -> Dict[str, Any]:
    """
    汇总本项目需要的“可筛选特征”。
    你后续可以把其中一部分迁移到 filter.py，保持 UI 只做展示。
    """
    # 官能团/键
    ester = count_ester_groups(smi)
    acid = count_carboxylic_acids(smi)
    ketone = count_carbonyl_groups(smi)
    aldehyde = count_aldehydes(smi)
    oh_non_ph = count_non_phenolic_hydroxyls(smi)
    oh_ph = count_phenolic_hydroxyls(smi)
    oh_total = oh_non_ph + oh_ph
    ether = count_ether_bonds(smi)

    nitro = count_nitro_groups(smi)
    nitrile = count_nitriles(smi)
    amine = count_amines(smi)
    amide = count_amide_groups(smi)
    cc_double = count_cc_double_bonds(smi)
    cc_triple = count_cc_triple_bonds(smi)

    # 空间/对称性/NMR
    chiral = count_chiral_centers(smi)
    uniq_h_total = get_total_unique_h_count(smi)
    uniq_h_benz = get_benzene_unique_h_count(smi)
    benz_subs = get_benzene_substituent_count(smi)
    h_ratio = get_normalized_h_ratio(smi)

    return {
        "smiles": smi,
        # 官能团
        "酯基": ester,
        "羧基": acid,
        "酮羰基": ketone,
        "醛基": aldehyde,
        "非酚羟基": oh_non_ph,
        "酚羟基": oh_ph,
        "羟基": oh_total,
        "醚键": ether,
        "硝基": nitro,
        "氰基": nitrile,
        "氨基": amine,
        "酰胺基": amide,
        "碳碳双键": cc_double,
        "碳碳三键": cc_triple,
        # 特征
        "手性中心": chiral,
        "不等同氢总数": uniq_h_total,
        "苯环不等同氢": uniq_h_benz,
        "苯环取代数": benz_subs,
        "氢谱峰面积比": h_ratio,
    }
