from rdkit import Chem

# =============================================================================
# 1. 扩展库定义 (支持并环)
# =============================================================================

# 并环骨架库
FUSED_CORES = {
    "苯环": "c1ccccc1",
}

S, D, T = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE

BLOCK_LIBRARY = {
    # --- 基础单原子 ---
    "氢原子": {"atoms": ["H"], "bonds": [], "new_slots": []},
    "氯原子": {"atoms": ["Cl"], "bonds": [], "new_slots": []},
    "溴原子": {"atoms": ["Br"], "bonds": [], "new_slots": []},
    "碳原子": {"atoms": ["C"], "bonds": [], "new_slots": [0, 0, 0]},
    "氮原子": {"atoms": ["N"], "bonds": [], "new_slots": [0, 0]},
    "氧原子": {"atoms": ["O"], "bonds": [], "new_slots": [0]},
    
    # --- 碳碳键延伸 ---
    "碳碳双键": {"atoms": ["C", "C"], "bonds": [(0, 1, D)], "new_slots": [0, 1, 1]},
    "碳碳三键": {"atoms": ["C", "C"], "bonds": [(0, 1, T)], "new_slots": [1]},
    
    # --- 官能团 ---
    "酮羰基": {"atoms": ["C", "O"], "bonds": [(0, 1, D)], "new_slots": [0]},
    "醛基":   {"atoms": ["C", "O"], "bonds": [(0, 1, D)], "new_slots": []},
    "羧基":   {"atoms": ["C", "O", "O"], "bonds": [(0, 1, D), (0, 2, S)], "new_slots": []},
    "酯基":   {"atoms": ["C", "O", "O"], "bonds": [(0, 1, D), (0, 2, S)], "new_slots": [2]},
    "酰胺基": {"atoms": ["C", "O", "N"], "bonds": [(0, 1, D), (0, 2, S)], "new_slots": [2, 2]},
    "硝基":   {"atoms": ["N", "O", "O"], "bonds": [(0, 1, D), (0, 2, D)], "new_slots": []},
    "氰基":   {"atoms": ["C", "N"], "bonds": [(0, 1, T)], "new_slots": []}
}

# 2. 积木挂载逻辑


def attach_block(rw_mol, anchor_idx, block_name):
    """
    将积木挂载到 anchor_idx。
    当前程序以“无氢苯环骨架 + 显式补氢”的方式工作：
    - 初始骨架上的可延伸位点默认都没有氢；
    - 跳过某个位点等价于在该位点显式补一个氢；
    - 氢原子不计入 inventory，可视为无限供应。
    """
    block = BLOCK_LIBRARY[block_name]

    local_to_global = []
    for atom_type in block["atoms"]:
        new_idx = rw_mol.AddAtom(Chem.Atom(atom_type))
        local_to_global.append(new_idx)

    for start, end, bond_type in block["bonds"]:
        rw_mol.AddBond(local_to_global[start], local_to_global[end], bond_type)

    entry_global = local_to_global[0]
    rw_mol.AddBond(anchor_idx, entry_global, Chem.BondType.SINGLE)

    new_active_sites = []
    for local_node_idx in block["new_slots"]:
        new_active_sites.append(local_to_global[local_node_idx])

    return new_active_sites

def get_inventory_list(strategy):
    inventory = []
    inventory.extend(strategy.get('functional_groups', []))
    name_map = {
        'c': '碳原子', 'n': '氮原子', 'o': '氧原子',
        'cl': '氯原子', 'br': '溴原子'
    }
    rem = strategy.get('remaining_atoms', {})
    for k, v in rem.items():
        if k in name_map:
            inventory.extend([name_map[k]] * v)
    return sorted(inventory)

# =============================================================================
# 3. 智能生成器
# =============================================================================

def get_initial_sites(core_mol):
    """
    识别初始苯环骨架上可延伸的位点。
    当前只支持苯环，因此直接返回六个芳香碳原子的索引。
    """
    return [atom.GetIdx() for atom in core_mol.GetAtoms()]

def molecule_generator(strategy, core_name="苯环"):
    inventory = get_inventory_list(strategy)
    unique_smiles = set()
    display_smiles_map = {}
    memo = set()
    
    # 1. 初始化骨架
    core_smiles = FUSED_CORES.get(core_name, "c1ccccc1")
    base_mol = Chem.MolFromSmiles(core_smiles)
    if base_mol is None:
        return []

    base_rw = Chem.RWMol(base_mol)
    for atom in base_rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    base_mol = base_rw.GetMol()
    Chem.SanitizeMol(base_mol)
    Chem.Kekulize(base_mol, clearAromaticFlags=True)
    
    # 2. 识别初始可取代位点
    initial_sites = get_initial_sites(base_mol)

    def backtrack(current_mol, sites_queue, remaining_inventory):
        # --- 拓扑去重 ---
        try:
            curr_smi = Chem.MolToSmiles(current_mol, canonical=True)
        except:
            return # 防止极少数中间态非法
        
        state = (curr_smi, tuple(sorted(remaining_inventory)))
        if state in memo:
            return
        memo.add(state)

        if not remaining_inventory:
            if not sites_queue:
                try:
                    final_mol = current_mol.GetMol()
                    Chem.SanitizeMol(final_mol)
                    final_mol = Chem.RemoveHs(final_mol)
                    Chem.SanitizeMol(final_mol)

                    aromatic_smi = Chem.MolToSmiles(final_mol, canonical=True)
                    kekule_mol = Chem.Mol(final_mol)
                    Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
                    display_smi = Chem.MolToSmiles(kekule_mol, canonical=True, kekuleSmiles=True)

                    unique_smiles.add(aromatic_smi)
                    display_smiles_map[aromatic_smi] = display_smi
                except Exception:
                    pass
                return

            target_site = sites_queue[0]
            other_sites = sites_queue[1:]

            h_mol = Chem.RWMol(current_mol)
            try:
                attach_block(h_mol, target_site, "氢原子")
                backtrack(h_mol, other_sites, remaining_inventory)
            except Exception:
                pass
            return

        if not sites_queue:
            return

        target_site = sites_queue[0]
        other_sites = sites_queue[1:]

        # 分支 1：挂载积木
        for comp_name in sorted(list(set(remaining_inventory))):
            new_mol = Chem.RWMol(current_mol)
            try:
                # 尝试挂载
                new_exts = attach_block(new_mol, target_site, comp_name)
                
                new_inv = remaining_inventory[:]
                new_inv.remove(comp_name)
                
                backtrack(new_mol, other_sites + new_exts, new_inv)
            except Exception:
                continue

        # 分支 2：跳过该位点（显式补一个氢）
        h_mol = Chem.RWMol(current_mol)
        try:
            attach_block(h_mol, target_site, "氢原子")
            backtrack(h_mol, other_sites, remaining_inventory)
        except Exception:
            pass

    # 启动
    backtrack(Chem.RWMol(base_mol), initial_sites, inventory)
    return [display_smiles_map[smi] for smi in sorted(unique_smiles)]
