# -*- coding: utf-8 -*-
"""libdock.af3data —— AlphaFold3 原始输出读取(通用模块, 2026-10-04)
================================================================================
来源动机: 突变体分析中临时写的 PAE 查询/模板提取被证明是"模型质量论证"的
通用件(论文每次都要 pLDDT+PAE 双口径)。从 tools/oneoff/ 的
一次性实现收编为单一事实源。

数据源: AlphaFold Server 作业产物
  <name>_full_data_<i>.json  —— atom_plddts / pae(252x252) /
      token_chain_ids / token_res_ids / contact_probs
  <name>_summary_confidences_<i>.json —— ptm / iptm / ranking_score 等
  templates/<name>_template_hit_<k>_chains_a.cif —— 模板晶体(精简版 CIF,
      只有 _entry.id 可稳定提取, 无 _struct.title)

典型用法(局部置信度论证):
    from dockbuddy_dock.af3data import af3_pae_pairs
    pae = af3_pae_pairs("fold_xxx_full_data_4.json", [(205, 204), (205, 56)])
    # 与三联体内部 PAE 同级(≈0.8) = 该相对几何是全蛋白最确定的关系之一;
    # 远端对照(如 (205, 100))给出该蛋白的 PAE 基线。

pLDDT 逐残基提取请用 ensemble.extract_plddt(CIF 的 B_iso 列), 本模块不重复。
"""
import json
import os
import re
from typing import Dict, List, Sequence, Tuple, Union

from ._core import LOG

_AF3_REQUIRED_KEYS = ("pae", "token_res_ids")


def load_af3_full_data(path_or_dict: Union[str, dict]) -> dict:
    """读 full_data JSON(路径或已加载的 dict); 缺关键键时 raise KeyError。"""
    d = path_or_dict if isinstance(path_or_dict, dict) else \
        json.load(open(path_or_dict, encoding="utf-8"))
    missing = [k for k in _AF3_REQUIRED_KEYS if k not in d]
    if missing:
        raise KeyError(f"非 AF3 full_data 结构(缺 {missing}): {path_or_dict}")
    return d


def _residue_index(d: dict, chain: str = None) -> Dict[int, int]:
    """token_res_ids -> {残基号: token 索引}。

    多链/含配体作业中不同链残基号重复是常态: 指定 chain 时只索引该链
    (token_chain_ids 匹配); 未指定时取首现(单链作业等价行为)。
    token_res_ids 为空时 raise ValueError(带上下文, 不裸栈)。
    """
    res_ids = d["token_res_ids"]
    if not res_ids:
        raise ValueError("token_res_ids 为空(非有效 AF3 full_data)")
    chain_ids = d.get("token_chain_ids")
    idx: Dict[int, int] = {}
    for i, r in enumerate(res_ids):
        if chain is not None and chain_ids is not None:
            if str(chain_ids[i]) != str(chain):
                continue
        idx.setdefault(int(r), i)
    if chain is not None and not idx:
        raise KeyError(f"链 {chain} 在 token_chain_ids 中不存在")
    return idx


def _pae_at(d: dict, i: int, j: int) -> float:
    """PAE 矩阵取值 + 形状守卫(截断 JSON 报带上下文的 ValueError, 不裸 IndexError)。"""
    pae = d["pae"]
    if i >= len(pae) or j >= len(pae) or j >= len(pae[i]):
        raise ValueError(
            f"PAE 矩阵形状不足(索引 [{i}][{j}], 实际 {len(pae)} 行"
            f"{', 行 %d 长 %d' % (i, len(pae[i])) if i < len(pae) else ''})"
            f" —— full_data JSON 可能被截断")
    return float(pae[i][j])


def af3_pae(full_data: Union[str, dict], res_a: int, res_b: int,
            chain: str = None) -> float:
    """查询两残基间的预测对齐误差 PAE(A)。

    残基号不在 token_res_ids 时 raise KeyError(带残基号提示, 不静默猜)。
    多链作业传 chain 指定链(默认取首现, 兼容单链)。
    """
    d = load_af3_full_data(full_data)
    idx = _residue_index(d, chain)
    for r in (res_a, res_b):
        if r not in idx:
            raise KeyError(f"残基 {r} 不在 token_res_ids"
                           f"{'' if chain is None else f'(链 {chain})'} "
                           f"(范围 {min(idx)}-{max(idx)})")
    return _pae_at(d, idx[res_a], idx[res_b])


def af3_pae_pairs(full_data: Union[str, dict],
                  pairs: Sequence[Tuple[int, int]],
                  chain: str = None) -> Dict[Tuple[int, int], float]:
    """批量查询 PAE: [(a, b), ...] -> {(a, b): pae_A}。chain 语义同 af3_pae。"""
    d = load_af3_full_data(full_data)
    idx = _residue_index(d, chain)
    out: Dict[Tuple[int, int], float] = {}
    for a, b in pairs:
        for r in (a, b):
            if r not in idx:
                raise KeyError(f"残基 {r} 不在 token_res_ids"
                               f"{'' if chain is None else f'(链 {chain})'} "
                               f"(范围 {min(idx)}-{max(idx)})")
        out[(a, b)] = _pae_at(d, idx[a], idx[b])
    return out


def cif_entry_id(cif_path: str) -> str:
    """提取 CIF 的 _entry.id(如 '1jfr' -> '1JFR')。

    AF Server 模板是精简版 CIF: 无 _struct.title, 只有 _entry.id 稳定存在;
    读不到时 raise ValueError(调用方不应静默吞掉模板来源不明的情形)。
    """
    txt = open(cif_path, encoding="utf-8", errors="ignore").read(30000)
    m = re.search(r"_entry\.id\s+(\S+)", txt)
    if not m:
        raise ValueError(f"CIF 中未找到 _entry.id: {cif_path}")
    return m.group(1).upper()


def af3_confidence_summary(confidences_json: Union[str, dict]) -> Dict[str, float]:
    """读 summary_confidences JSON 的关键指标(ptm/ranking_score/fraction_disordered)。"""
    d = confidences_json if isinstance(confidences_json, dict) else \
        json.load(open(confidences_json, encoding="utf-8"))
    keys = ("ptm", "ranking_score", "fraction_disordered", "has_clash")
    out = {k: d[k] for k in keys if k in d}
    LOG.info("AF3 模型置信度: %s", {k: v for k, v in out.items()})
    return out


def list_af3_models(af_dir: str) -> List[str]:
    """列出 AF3 目录下全部模型 full_data JSON 路径(按编号排序)。"""
    paths = [os.path.join(af_dir, f)
             for f in os.listdir(af_dir)
             if re.match(r".*_full_data_\d+\.json$", f)]
    paths.sort(key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1)))
    return paths


# ---------------------------------------------------------------------------
# 复合物质量指标: MAS(ligand 改型 LDDT) / ipTM / chain_pair_pae_min(2026-10-08)
# 动机: 用户已有 Google AF3 预测结果数据, 后处理要按 MAS/ipTM/PAE 过滤酶-底物复合物。
# ---------------------------------------------------------------------------
def af3_summary_full(confidences_json: Union[str, dict]) -> Dict[str, float]:
    """读 summary_confidences 的全部复合物级指标。

    返回(存在则取): ptm / iptm / ranking_score / fraction_disordered / has_clash /
    num_recycles, 以及 chain_pair_pae_min(蛋白-配体最小 PAE 列表的**最小值**)。
    """
    d = confidences_json if isinstance(confidences_json, dict) else \
        json.load(open(confidences_json, encoding="utf-8"))
    out: Dict[str, float] = {}
    for k in ("ptm", "iptm", "ranking_score", "fraction_disordered", "num_recycles"):
        if k in d:
            out[k] = float(d[k])
    if "has_clash" in d:
        out["has_clash"] = 1.0 if d["has_clash"] else 0.0
    # chain_pair_pae_min: [[i, j, min_pae], ...] → 取跨链最小 PAE 的最小值
    cppm = d.get("chain_pair_pae_min") or []
    paes = [float(t[2]) for t in cppm if isinstance(t, (list, tuple)) and len(t) >= 3]
    if paes:
        out["chain_pair_pae_min"] = min(paes)
    LOG.info("AF3 复合物指标: %s", {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in out.items()})
    return out


def _ligand_chain_ids(full_data: dict, ligand_chain: str = None) -> List[str]:
    """推断配体(小分子)的 chain id 集合。

    AF3 full_data 里 token_chain_ids 只覆盖聚合物(蛋白/核酸) token; 配体原子的
    atom_chain_ids 不在 token_chain_ids 里。故: 显式给 ligand_chain 用给定值; 否则
    取 atom_chain_ids 中未出现在 token_chain_ids 集合里的 id。
    """
    token_chains = set(str(c) for c in (full_data.get("token_chain_ids") or []))
    if ligand_chain is not None:
        return [str(ligand_chain)]
    atom_chains = full_data.get("atom_chain_ids") or []
    return sorted({str(c) for c in atom_chains if str(c) not in token_chains})


def af3_ligand_mas(full_data: Union[str, dict], ligand_chain: str = None) -> float:
    """MAS = 配体原子的平均 pLDDT(ligand modified LDDT, 底物摆放置信度)。

    full_data 里 atom_plddts 覆盖全部原子; 配体原子按 atom_chain_ids 识别
    (显式 ligand_chain 或自动取非聚合物链)。无配体原子时 raise ValueError。
    仅依赖 atom_plddts/atom_chain_ids, 不要求 pae 等字段。
    """
    d = full_data if isinstance(full_data, dict) else \
        json.load(open(full_data, encoding="utf-8"))
    plddts = d.get("atom_plddts") or []
    atom_chains = d.get("atom_chain_ids") or []
    if len(plddts) != len(atom_chains):
        raise ValueError("atom_plddts 与 atom_chain_ids 长度不一致(full_data 可能被截断)")
    ligand_ids = set(_ligand_chain_ids(d, ligand_chain))
    vals = [float(p) for p, c in zip(plddts, atom_chains) if str(c) in ligand_ids]
    if not vals:
        raise ValueError(f"未找到配体原子(ligand_chain={ligand_chain!r}, "
                         f"候选链 {ligand_ids})")
    return sum(vals) / len(vals)


def af3_complex_metrics(full_data: Union[str, dict],
                        confidences: Union[str, dict],
                        ligand_chain: str = None) -> Dict[str, float]:
    """一站式: 复合物质量指标 = summary(ptm/iptm/ranking_score/chain_pair_pae_min)
    + full_data 派生的 MAS。供第二层 libgate 软过滤(MAS<50 或 ipTM<0.7 或 PAE>10 丢弃)。"""
    out = af3_summary_full(confidences)
    try:
        out["mas"] = af3_ligand_mas(full_data, ligand_chain)
    except ValueError as e:
        LOG.warning("MAS 提取失败(该作业可能无配体): %s", e)
        out["mas"] = float("nan")
    return out


__all__ = ["load_af3_full_data", "af3_pae", "af3_pae_pairs", "cif_entry_id",
           "af3_confidence_summary", "list_af3_models", "af3_summary_full",
           "af3_ligand_mas", "af3_complex_metrics"]
