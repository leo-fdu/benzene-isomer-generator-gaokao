
# -*- coding: utf-8 -*-
"""
ui.py (Streamlit 初稿)
=================================================
目标：
- 提供一个接近截图风格的网页 UI（左侧：原子组成/骨架选择；右侧：筛选条件 tabs）
- 明确“输入数据类型 + 变量命名 + 约束表达方式”，方便你后续统一 preparation / generator / filter 的接口

运行方式：
    streamlit run ui.py

注意：
- 当前版本只支持以苯环为唯一核心骨架。
- UI 支持三种约束输入：留空(不限制) / “无”(等价于0) / “有”(>=1) / 数字(精确等于该数)
- 对“影响不饱和度的官能团”，preparation 当前只支持“精确/禁止/不限制”，
  因此“有(>=1)”会被延后到 filter 阶段再判定（不会被误当作“=1”）。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from rdkit import Chem

from preparation import get_chain_strategies, get_omega_status
from molecule_generator import FUSED_CORES, molecule_generator
from renderer import render as render_to_file
from filter import compute_features, filter_stable_molecules, normalize_ratio


st.set_page_config(
    page_title="上海高考同分异构体生成器",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
/* 主标题更大一些 */
h1, h2, h3 { font-weight: 800; }

/* 收紧输入框间距，让布局更像“表单网格” */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {
    gap: 0.5rem;
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


CountSpec = Union[None, int, str]  # None | int | ">=1"
BENZENE_ONLY_CORE_NAME = "苯环"
BENZENE_CORE_SMILES = FUSED_CORES[BENZENE_ONLY_CORE_NAME]

FEATURE_ORDER = [
    "smiles",
    "酯基",
    "羧基",
    "酮羰基",
    "醛基",
    "非酚羟基",
    "酚羟基",
    "羟基",
    "醚键",
    "硝基",
    "氰基",
    "氨基",
    "酰胺基",
    "碳碳双键",
    "碳碳三键",
    "手性中心",
    "不等同氢总数",
    "苯环不等同氢",
    "苯环取代数",
    "氢谱峰面积比",
]

COUNT_FEATURE_KEYS = [
    "酯基",
    "羧基",
    "酮羰基",
    "醛基",
    "非酚羟基",
    "酚羟基",
    "羟基",
    "醚键",
    "硝基",
    "氰基",
    "氨基",
    "酰胺基",
    "碳碳双键",
    "碳碳三键",
    "手性中心",
    "不等同氢总数",
    "苯环不等同氢",
    "苯环取代数",
]

PREPARATION_KEYS = [
    "碳碳双键",
    "碳碳三键",
    "酮羰基",
    "醛基",
    "羧基",
    "酯基",
    "酰胺基",
    "硝基",
    "氰基",
]

MIN_ATOMS_PER_FEATURE: Dict[str, Dict[str, int]] = {
    "酯基": {"c": 1, "o": 2},
    "羧基": {"c": 1, "o": 2},
    "酮羰基": {"c": 1, "o": 1},
    "醛基": {"c": 1, "o": 1},
    "非酚羟基": {"o": 1},
    "酚羟基": {"o": 1},
    "醚键": {"o": 1},
    "硝基": {"n": 1, "o": 2},
    "氰基": {"c": 1, "n": 1},
    "氨基": {"n": 1},
    "酰胺基": {"c": 1, "n": 1, "o": 1},
    "碳碳双键": {"c": 2},
    "碳碳三键": {"c": 2},
}


def parse_count_spec(raw: str) -> CountSpec:
    """
    将 UI 文本输入解析为约束：
    - "" / 空白 -> None (不限制)
    - "无" / "0" -> 0
    - "有" -> ">=1"
    - 整数 -> int
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    if s in {"不限制", "不限", "none", "None", "NONE"}:
        return None
    if s in {"无", "0"}:
        return 0
    if s in {"有", ">=1", ">0"}:
        return ">=1"
    if re.fullmatch(r"\d+", s):
        return int(s)
    raise ValueError(f"无法解析输入：{raw}（支持：留空 / 无 / 有 / 数字）")


def parse_ratio(raw: str) -> Optional[List[int]]:
    """
    解析氢谱峰面积比输入：
    - 留空 -> None (不限制)
    - 支持格式：'3:2:1' / '3,2,1' / '3 2 1'
    - 返回最简整数比，并按降序排列，例如 [3, 2, 1]
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None

    s = s.replace("，", ",").replace("：", ":")
    if not re.fullmatch(r"\d+(?:(?:\s*[:,]\s*|\s+)\d+)*", s):
        raise ValueError("面积比请输入正整数序列，例如 3:2:1 或 3,2,1")

    ratio = [int(part) for part in re.findall(r"\d+", s)]
    normalized_ratio = normalize_ratio(ratio)
    if not normalized_ratio:
        raise ValueError("面积比中的每一项都必须是正整数")
    return normalized_ratio


def count_atoms_in_smiles(smiles: str) -> Dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"c": 0, "h": 0, "n": 0, "o": 0, "cl": 0, "br": 0}
    mol = Chem.AddHs(mol)
    counts = {"c": 0, "h": 0, "n": 0, "o": 0, "cl": 0, "br": 0}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C":
            counts["c"] += 1
        elif sym == "H":
            counts["h"] += 1
        elif sym == "N":
            counts["n"] += 1
        elif sym == "O":
            counts["o"] += 1
        elif sym == "Cl":
            counts["cl"] += 1
        elif sym == "Br":
            counts["br"] += 1
    return counts


def calc_omega(c: int, h: int, n: int, cl: int, br: int) -> int:
    omega = c + 1 - (h + cl + br - n) / 2
    return int(round(omega))


def core_resource() -> Tuple[Dict[str, int], int]:
    counts = count_atoms_in_smiles(BENZENE_CORE_SMILES)
    omega_core = calc_omega(counts["c"], counts["h"], counts["n"], counts["cl"], counts["br"])
    return counts, omega_core


def passes_count_spec(value: int, spec: CountSpec) -> bool:
    if spec is None:
        return True
    if isinstance(spec, int):
        return value == spec
    if isinstance(spec, str) and spec == ">=1":
        return value >= 1
    return True


def passes_ratio_spec(value_ratio: List[int], spec_ratio: Optional[List[int]]) -> bool:
    if spec_ratio is None:
        return True
    normalized_value = normalize_ratio(value_ratio)
    normalized_spec = normalize_ratio(spec_ratio)
    if not normalized_value or not normalized_spec:
        return False
    return normalized_value == normalized_spec


def build_ui_constraints(raw_dict: Dict[str, str]) -> Dict[str, CountSpec]:
    parsed: Dict[str, CountSpec] = {}
    for k, v in raw_dict.items():
        try:
            parsed[k] = parse_count_spec(v)
        except Exception as e:
            raise ValueError(f"【{k}】输入有误：{e}")
    return parsed


def build_filter_constraint_specs(
    ui_constraints: Dict[str, CountSpec],
    ratio_spec: Optional[List[int]],
) -> Dict[str, Any]:
    constraint_specs = {key: None for key in FEATURE_ORDER}
    constraint_specs["氢谱峰面积比"] = ratio_spec
    for key in COUNT_FEATURE_KEYS:
        constraint_specs[key] = ui_constraints.get(key, None)
    return constraint_specs


def to_preparation_constraints(filter_constraint_specs: Dict[str, Any]) -> Dict[str, Optional[int]]:
    prep_constraints: Dict[str, Optional[int]] = {}
    for key in PREPARATION_KEYS:
        spec = filter_constraint_specs.get(key, None)
        if isinstance(spec, int):
            prep_constraints[key] = spec
        else:
            prep_constraints[key] = None
    return prep_constraints


def add_halogen_inventory(strategy: Dict[str, Any], rem_cl: int, rem_br: int) -> Dict[str, Any]:
    stg = dict(strategy)
    functional_groups = stg.get("functional_groups", {})
    if isinstance(functional_groups, dict):
        stg["functional_groups"] = [
            name
            for name, count in functional_groups.items()
            for _ in range(int(count))
        ]
    else:
        stg["functional_groups"] = list(functional_groups)

    remaining_atoms = dict(stg.get("remaining_atoms", {}))
    remaining_atoms["cl"] = int(rem_cl)
    remaining_atoms["br"] = int(rem_br)
    stg["remaining_atoms"] = remaining_atoms
    return stg


def estimate_min_required_atoms(filter_constraint_specs: Dict[str, Any]) -> Dict[str, int]:
    required = {"c": 6, "h": 0, "o": 0, "n": 0, "cl": 0, "br": 0}

    total_oh_spec = filter_constraint_specs.get("羟基", None)
    has_total_oh_constraint = total_oh_spec not in (None, 0)

    for key, atoms in MIN_ATOMS_PER_FEATURE.items():
        if has_total_oh_constraint and key in {"酚羟基", "非酚羟基"}:
            continue

        spec = filter_constraint_specs.get(key, None)
        if spec is None or spec == 0:
            continue
        lower_bound = 1 if spec == ">=1" else int(spec)
        for atom, count in atoms.items():
            if atom == "h":
                continue
            required[atom] += count * lower_bound

    if has_total_oh_constraint:
        total_oh_lower = 1 if total_oh_spec == ">=1" else int(total_oh_spec)
        required["o"] += total_oh_lower

    return required


def validate_inputs(
    c: int,
    h: int,
    o: int,
    n: int,
    cl: int,
    br: int,
    filter_constraint_specs: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    omega_status = get_omega_status(c, h, n, o, cl, br)
    if omega_status == None:    
        errors.append("由化学式计算得到的不饱和度必须是整数")

    if c < 6 :
        errors.append("当前程序只支持苯环骨架，因此化学式至少应当覆盖苯环骨架")

    min_required_atoms = estimate_min_required_atoms(filter_constraint_specs)
    actual_atoms = {"c": c, "h": h, "o": o, "n": n, "cl": cl, "br": br}
    atom_labels = {"c": "C", "h": "H", "o": "O", "n": "N", "cl": "Cl", "br": "Br"}
    for atom_key, need in min_required_atoms.items():
        if actual_atoms[atom_key] < need:
            errors.append(
                f"化学式中的 {atom_labels[atom_key]} 原子数不足：至少需要 {need} 个，但当前只有 {actual_atoms[atom_key]} 个"
            )

    unique_h_total_spec = filter_constraint_specs.get("不等同氢总数", None)
    if isinstance(unique_h_total_spec, int) and unique_h_total_spec > h:
        errors.append("不等同氢总数不能超过分子中的氢原子总数")
    if unique_h_total_spec == ">=1" and h == 0:
        errors.append("当前化学式没有氢原子，不能要求存在不等同氢")

    benzene_unique_h_spec = filter_constraint_specs.get("苯环不等同氢", None)
    if isinstance(benzene_unique_h_spec, int):
        if benzene_unique_h_spec > 4:
            errors.append("苯环上不等同氢数目不能超过 4")
        if benzene_unique_h_spec > h:
            errors.append("苯环上不等同氢数目不能超过分子中的氢原子总数")

    benzene_sub_spec = filter_constraint_specs.get("苯环取代数", None)
    if isinstance(benzene_sub_spec, int) and benzene_sub_spec > 6:
        errors.append("苯环取代数不能超过 6")

    ratio_spec = filter_constraint_specs.get("氢谱峰面积比", None)
    if ratio_spec is not None:
        ratio_sum = sum(ratio_spec)
        if h % ratio_sum != 0:
            errors.append("氢谱峰面积比输入的若干数之和必须能够整除化学式中的氢原子总数")

    chiral_spec = filter_constraint_specs.get("手性中心", None)
    if isinstance(chiral_spec, int) and chiral_spec > c - 6:
        errors.append("手性中心数目不能明显超过分子中可能作为手性中心的重原子数")

    return errors


@dataclass
class RunResult:
    candidates: List[str]
    stable: List[str]
    passed: List[str]
    features: List[Dict[str, Any]]
    strategies_count: int


def run_pipeline(
    c: int,
    h: int,
    o: int,
    n: int,
    cl: int,
    br: int,
    core_name: str,
    filter_constraint_specs: Dict[str, Any],
    max_candidates: Optional[int] = None,
) -> RunResult:
    omega_status = get_omega_status(c, h, n, o, cl, br)
    if omega_status == None:
        raise ValueError("由化学式计算得到的不饱和度必须是整数")

    core_counts, omega_core = core_resource()
    rem_c = c - core_counts["c"]
    rem_o = o - core_counts["o"]
    rem_n = n - core_counts["n"]
    rem_cl = cl - core_counts["cl"]
    rem_br = br - core_counts["br"]
    rem_omega = omega_status - omega_core

    if any(x < 0 for x in [rem_c, rem_o, rem_n, rem_cl, rem_br, rem_omega]):
        raise ValueError("化学式与苯环骨架不兼容：扣除骨架后出现负库存或负不饱和度。")

    prep_constraints = to_preparation_constraints(filter_constraint_specs)
    strategies = get_chain_strategies(rem_c, rem_o, rem_n, rem_omega, prep_constraints)

    if not strategies:
        return RunResult(candidates=[], stable=[], passed=[], features=[], strategies_count=0)

    candidates_set = set()
    for strategy in strategies:
        normalized_strategy = add_halogen_inventory(strategy, rem_cl=rem_cl, rem_br=rem_br)
        smiles_list = molecule_generator(normalized_strategy, core_name=core_name)
        for smi in smiles_list:
            candidates_set.add(smi)
        if max_candidates is not None and len(candidates_set) >= max_candidates:
            break

    candidates = sorted(candidates_set)
    stable = filter_stable_molecules(candidates)
    if stable is None:
        stable = candidates

    passed: List[str] = []
    features: List[Dict[str, Any]] = []

    for smi in stable:
        feat = compute_features(smi)

        ok = True
        for key in COUNT_FEATURE_KEYS:
            if not passes_count_spec(int(feat.get(key, 0)), filter_constraint_specs.get(key, None)):
                ok = False
                break
        if not ok:
            continue

        if not passes_ratio_spec(feat.get("氢谱峰面积比", []), filter_constraint_specs.get("氢谱峰面积比", None)):
            continue

        passed.append(smi)
        features.append(feat)

    return RunResult(
        candidates=candidates,
        stable=stable,
        passed=passed,
        features=features,
        strategies_count=len(strategies),
    )


st.title("🧪上海高考同分异构体生成器🧪")
st.divider()

with st.sidebar:
    st.header("1. 原子组成（必填）")

    c = st.number_input("碳 (C)", min_value=0, max_value=50, value=0, step=1)
    h = st.number_input("氢 (H)", min_value=0, max_value=200, value=0, step=1)
    o = st.number_input("氧 (O)", min_value=0, max_value=30, value=0, step=1)
    n = st.number_input("氮 (N)", min_value=0, max_value=30, value=0, step=1)
    cl = st.number_input("氯 (Cl)", min_value=0, max_value=30, value=0, step=1)
    br = st.number_input("溴 (Br)", min_value=0, max_value=30, value=0, step=1)

    st.markdown("---")
    st.header("2. 核心骨架选择")

    core_name = st.selectbox(
        "骨架类型",
        options=[BENZENE_ONLY_CORE_NAME],
        index=0,
        help="当前版本仅支持苯环作为基础骨架。",
    )

    formula_str = (
        f"C{c}H{h}"
        + (f"O{o}" if o else "")
        + (f"N{n}" if n else "")
        + (f"Cl{cl}" if cl else "")
        + (f"Br{br}" if br else "")
    )
    st.caption(f"当前化学式：**{formula_str}**")


st.subheader("3. 筛选限定条件")
st.caption("输入规则：留空=不限制；输入“无”=0；输入“有”=至少1；输入数字=精确数量。")

tab_fg, tab_geom, tab_nmr = st.tabs(["官能团数目", "空间/取代特征", "氢谱 (NMR) 特征"])

ui_constraints_raw: Dict[str, str] = {}

with tab_fg:
    col1, col2, col3, col4= st.columns(4)

    with col1:
        ui_constraints_raw["酯基"] = st.text_input("酯基数目", placeholder="数字 / 有 / 无", key="fg_ester")
        ui_constraints_raw["醛基"] = st.text_input("醛基数目", placeholder="数字 / 有 / 无", key="fg_aldehyde")
        ui_constraints_raw["氰基"] = st.text_input("氰基数目", placeholder="数字 / 有 / 无", key="fg_nitrile")
        ui_constraints_raw["羟基"] = st.text_input("羟基数目（总）", placeholder="数字 / 有 / 无", key="fg_oh_total")

    with col2:
        ui_constraints_raw["羧基"] = st.text_input("羧基数目", placeholder="数字 / 有 / 无", key="fg_acid")
        ui_constraints_raw["酮羰基"] = st.text_input("酮羰基数目", placeholder="数字 / 有 / 无", key="fg_ketone")
        ui_constraints_raw["酚羟基"] = st.text_input("酚羟基数目", placeholder="数字 / 有 / 无", key="fg_oh_ph")
        ui_constraints_raw["非酚羟基"] = st.text_input("非酚羟基数目", placeholder="数字 / 有 / 无", key="fg_oh_nonph")

    with col3:
        ui_constraints_raw["硝基"] = st.text_input("硝基数目", placeholder="数字 / 有 / 无", key="fg_nitro")
        ui_constraints_raw["氨基"] = st.text_input("氨基数目", placeholder="数字 / 有 / 无", key="fg_amine")
        ui_constraints_raw["醚键"] = st.text_input("醚键数目", placeholder="数字 / 有 / 无", key="fg_ether")
        ui_constraints_raw["酰胺基"] = st.text_input("酰胺基数目", placeholder="数字 / 有 / 无", key="fg_amide")

    with col4:
        ui_constraints_raw["碳碳双键"] = st.text_input("碳碳双键数目", placeholder="数字 / 有 / 无", key="fg_cc_double")
        ui_constraints_raw["碳碳三键"] = st.text_input("碳碳三键数目", placeholder="数字 / 有 / 无", key="fg_cc_triple")

with tab_geom:
    col1, col2, col3 = st.columns(3)
    with col1:
        ui_constraints_raw["手性中心"] = st.text_input("手性中心数目", placeholder="数字 / 有 / 无", key="geom_chiral")
    with col2:
        ui_constraints_raw["不等同氢总数"] = st.text_input("不等同氢总数", placeholder="数字 / 有 / 无", key="geom_uniq_h_total")
    with col3:
        ui_constraints_raw["苯环取代数"] = st.text_input("苯环取代数", placeholder="数字 / 有 / 无", key="geom_benz_sub")

    col4, col5 = st.columns(2)
    with col4:
        ui_constraints_raw["苯环不等同氢"] = st.text_input("苯环上不等同氢数目", placeholder="数字 / 有 / 无", key="geom_uniq_h_benz")
    with col5:
        st.caption("提示：当前版本只支持且默认只含一个苯环。")

with tab_nmr:
    ratio_raw = st.text_input("氢谱峰面积比", placeholder="例如：3:2:1、1:1:2 或 6,4", key="nmr_ratio")
    st.caption("输入与分子结果都会自动化为最简整数比，并按降序比较。")


st.markdown("")
run_btn = st.button("🚀 开始计算并筛选", use_container_width=False)

if run_btn:
    try:
        ui_constraints = build_ui_constraints(ui_constraints_raw)
        ratio_spec = parse_ratio(ratio_raw)
        filter_constraint_specs = build_filter_constraint_specs(ui_constraints, ratio_spec)

        validation_errors = validate_inputs(
            c=int(c),
            h=int(h),
            o=int(o),
            n=int(n),
            cl=int(cl),
            br=int(br),
            filter_constraint_specs=filter_constraint_specs,
        )
        if validation_errors:
            raise ValueError("\n".join(f"- {msg}" for msg in validation_errors))

        with st.status("正在计算…", expanded=True) as status:
            t0 = time.time()
            status.update(label="1/4 计算策略（preparation）…")
            res = run_pipeline(
                c=int(c),
                h=int(h),
                o=int(o),
                n=int(n),
                cl=int(cl),
                br=int(br),
                core_name=str(core_name),
                filter_constraint_specs=filter_constraint_specs,
            )
            status.update(label="2/4 候选分子生成完成（molecule_generator）…")
            status.update(label="3/4 筛选完成（filter + 特征判定）…")
            status.update(label="4/4 完成", state="complete")

        st.success(f"生成候选：{len(res.candidates)} 个；初筛稳定：{len(res.stable)} 个；最终通过：{len(res.passed)} 个")
        st.caption(f"官能团分配方案数：{res.strategies_count}；用时：{time.time()-t0:.2f} s")

        if res.features:
            st.subheader("通过筛选的分子（特征表）")
            show_cols = [
                "smiles",
                "酯基", "羧基", "酮羰基", "醛基", "羟基", "非酚羟基", "酚羟基", "醚键",
                "硝基", "氰基", "氨基", "酰胺基", "碳碳双键", "碳碳三键",
                "手性中心", "不等同氢总数", "苯环不等同氢", "苯环取代数",
                "氢谱峰面积比",
            ]
            table = [{k: f.get(k) for k in show_cols} for f in res.features]
            st.dataframe(table, use_container_width=True, hide_index=True)

        st.subheader("结构图预览")
        max_show = st.slider(
            "最多展示多少个结构图",
            min_value=1,
            max_value=80,
            value=min(24, max(1, len(res.passed))),
            step=1,
        )

        cols = st.columns(4)
        for i, smi in enumerate(res.passed[:max_show]):
            col = cols[i % 4]
            image = render_to_file(smi, width=400, height=400)
            with col:
                if image is not None:
                    col.image(image, caption=f"#{i+1}")
                else:
                    col.warning("渲染失败")
                col.code(smi)

        with st.expander("调试信息（可选）"):
            st.write(
                {
                    "formula": formula_str,
                    "core_name": core_name,
                    "ui_constraints": ui_constraints,
                    "filter_constraint_specs": filter_constraint_specs,
                    "preparation_constraints": to_preparation_constraints(filter_constraint_specs),
                }
            )

    except Exception as e:
        st.error(str(e))
