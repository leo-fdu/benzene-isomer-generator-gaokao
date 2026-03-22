from collections import Counter

def get_omega_status(c, h, n, o, cl, br):
    if (h + cl + br - n) % 2 != 0:
        return None
    else:
        omega = c + 1 - (h + cl + br - n) / 2
        return int(omega)

#omega>0模块文库
BLOCK_LIBRARY = {
    "碳碳双键": {"atoms": {"c": 2}, "omega": 1},
    "碳碳三键": {"atoms": {"c": 2}, "omega": 2},
    "羰基": {"atoms": {"c": 1, "o": 1}, "omega": 1},
    "硝基": {"atoms": {"n": 1, "o": 2}, "omega": 1},
    "氰基": {"atoms": {"c": 1, "n": 1}, "omega": 2}
}

def get_chain_strategies(rem_c, rem_o, rem_n, rem_omega, constraints):
    results = []
    # 1. 预处理约束：先扣除至少需要出现的结构块
    minimum_constraints = {k: v for k, v in constraints.items() if v is not None and v > 0}
    forbidden = [k for k, v in constraints.items() if v == 0]
    base_combination = []
    c_left, o_left, n_left, w_left = rem_c, rem_o, rem_n, rem_omega
    
    for name, count in minimum_constraints.items():
        if name in BLOCK_LIBRARY:
            b = BLOCK_LIBRARY[name]
            for _ in range(count):
                c_left -= b['atoms'].get('c', 0)
                o_left -= b['atoms'].get('o', 0)
                n_left -= b['atoms'].get('n', 0)
                w_left -= b['omega']
                base_combination.append(name)
    # 资源合法性初步检查
    if c_left < 0 or o_left < 0 or n_left < 0 or w_left < 0:
        return []
    # 2. 构造候选池：只排除被禁止的块；正数约束只表示下界，仍允许额外补充
    available_names = [
        k for k, v in BLOCK_LIBRARY.items() 
        if k not in forbidden
    ]
    # 3. 递归搜索：遍历所有能耗尽 Omega 的组合
    def search(c, o, n, w, start_idx, path):
        # 只要 Omega 耗尽，就是一个合法的官能团分配方案
        if w == 0:
            group_counts = Counter(path)
            # 记录方案，并保留剩下的原子，它们将作为延伸单元（单原子节点）
            results.append({
                "functional_groups": dict(group_counts), 
                "remaining_atoms": {"c": c, "o": o, "n": n}
            })
            return
        for i in range(start_idx, len(available_names)):
            name = available_names[i]
            b = BLOCK_LIBRARY[name]
            # 判断资源是否足够支付该官能团
            if (b['omega'] <= w and 
                b['atoms'].get('c', 0) <= c and 
                b['atoms'].get('o', 0) <= o and 
                b['atoms'].get('n', 0) <= n):
                
                search(c - b['atoms'].get('c', 0),
                       o - b['atoms'].get('o', 0),
                       n - b['atoms'].get('n', 0),
                       w - b['omega'],
                       i,  # 允许重复
                       path + [name])
    search(c_left, o_left, n_left, w_left, 0, base_combination)
    if not results:
        return []
    return results
#输出示例：[ {  "functional_groups": {"羰基":1},  "remaining_atoms": {"c":4,"o":1,"n":0} }, {  "functional_groups": {"硝基":1},  "remaining_atoms": {"c":5,"o":0,"n":0} }]
