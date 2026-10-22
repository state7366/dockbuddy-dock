# -*- coding: utf-8 -*-
"""evidence.py —— 证据门槛注册表(单一事实源) + 通用 verdict(自否证判词)辅助。

为什么需要它
------------
判据阈值(噪声地板 / 接触阈值 / 口径半径)与"是否越过地板才能写方向性结论"的逻辑,
此前分散在 common.py / orientation.py / retention.py / deacylation.py /
conformation_entropy.py / pocket_tunnel.py 等多个模块, 各自持有同名常量、各自实现
"|Δ| < 地板 → 禁写方向性结论"的判定。这带来两个隐患:
  (1) 同一类量(比例量 / 距离量 / 体积量)的噪声地板数值在不同模块可能漂移;
  (2) 新增证据线(MM/GBSA / ensemble 聚合 / 组合突变对比)时没有统一原语可复用,
      容易写出与既有判词口径不一致的禁写清单。

本模块做两件事(均刻意只依赖 common 的 stdlib 部分, 不引 openbabel, 以便 report /
解析层在不带第三方依赖的 base python 下也能安全 import):
  1. THRESHOLD_SPECS / thresholds(): 全部命名门槛的**唯一登记处**, 数值与释义集中
     维护; 环境量覆盖约定为 DOCK_EVID_<NAME>(大写), 并为既有模块的历史 env 名保留
     别名(见 _THRESHOLD_ALIASES), 改名不影响旧脚本。
  2. 通用辅助: classify_delta / frac_delta_verdict / dist_delta_verdict /
     vol_delta_verdict / replicate_sign_consistent / conformation_sensitive /
     VerdictBuilder —— 把"比例/距离/体积量的噪声地板判定 + 禁写/可写清单 + 叙事句"
     收敛成可调用的纯函数, 既有模块可渐进迁移, 新模块直接复用。

注意: 本模块**不持有任何几何/对接逻辑**, 只管"数值门槛 + 判定"。具体的 verdict
叙事句(禁写哪些话)由各业务模块在其 verdict 函数中组合, 本模块提供 VerdictBuilder
与判据原子, 但不替业务模块造句(避免把领域语义塞进通用层)。
"""
from __future__ import annotations

import logging
import os
import statistics as _st
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dockbuddy_common import env_float

LOG = logging.getLogger("dockbuddy_dock")


# ----------------------------------------------------------------------------
# 1. 阈值单一事实源
# ----------------------------------------------------------------------------
# 每个阈值: (默认值, [历史 env 名别名])。新增阈值只需在此登记一处。
# 类别(便于检索与报告引用): reliability / orientation / retention / deacylation /
# conformation_entropy / pocket_tunnel / multiconf。
THRESHOLD_SPECS: Dict[str, Tuple[float, List[str]]] = {
    # ---- reliability (common._DEFAULT_THRESHOLDS 迁入) ----
    "contact_unreliable": (0.3, ["DOCK_REL_CONTACT_UNRELIABLE"]),
    "contact_warn": (0.5, ["DOCK_REL_CONTACT_WARN"]),
    "buried_clash": (3.0, ["DOCK_REL_BURIED_CLASH"]),
    "buried_frac": (0.15, ["DOCK_REL_BURIED_FRAC"]),
    "energy_abs_max": (50.0, ["DOCK_REL_ENERGY_ABS_MAX"]),
    "heavy_max": (40.0, ["DOCK_REL_HEAVY_MAX"]),
    "top_rmsd_warn": (2.0, ["DOCK_REL_TOP_RMSD_WARN"]),
    "pose_scatter_warn": (3.0, ["DOCK_REL_POSE_SCATTER_WARN"]),
    "nuc_offsite": (5.0, ["DOCK_REL_NUC_OFFSITE"]),
    # ---- orientation (orientation.py) ----
    "orient_nuc_max": (3.5, ["DOCK_ORIENT_NUC_MAX"]),
    "orient_his_o_max": (3.5, ["DOCK_ORIENT_HIS_O_MAX"]),
    "orient_offsite_A": (10.0, ["DOCK_ORIENT_OFFSITE_A"]),
    "orient_noise_frac": (0.15, ["DOCK_ORIENT_NOISE_FRAC"]),
    "orient_dist_noise_A": (1.0, ["DOCK_ORIENT_DIST_NOISE_A"]),
    # ---- retention (retention.py) ----
    "retention_stack_A": (5.0, ["DOCK_RETENTION_STACK_A"]),
    "retention_noise_frac": (0.15, ["DOCK_RETENTION_NOISE_FRAC"]),
    "cross_frame_warn_A": (5.0, ["DOCK_CROSS_FRAME_WARN"]),
    # ---- deacylation (deacylation.py) ----
    "water_clearance_A": (1.4, ["DOCK_DEACYL_CLEARANCE"]),
    "water_nuc_min_A": (2.6, ["DOCK_DEACYL_NUC_MIN"]),
    "water_nuc_max_A": (3.4, ["DOCK_DEACYL_NUC_MAX"]),
    "bd_angle_target": (107.0, ["DOCK_DEACYL_BD_TARGET"]),
    "bd_angle_tol": (20.0, ["DOCK_DEACYL_BD_TOL"]),
    "his_water_max_A": (3.5, ["DOCK_DEACYL_HIS_WATER_MAX"]),
    "search_radius_A": (4.5, ["DOCK_DEACYL_SEARCH_RADIUS"]),
    "search_step_A": (0.4, ["DOCK_DEACYL_SEARCH_STEP"]),
    "deacyl_noise_vol_A3": (12.0, ["DOCK_DEACYL_NOISE_VOL"]),
    # ---- conformation_entropy (conformation_entropy.py) ----
    "conf_entropy_noise_J_per_mol_K": (8.0, ["DOCK_CONF_ENTROPY_NOISE_FLOOR_J"]),
    # ---- pocket_tunnel (pocket_tunnel.py) ----
    "bottleneck_replicate_noise_A": (8.0, ["DOCK_BOTTLENECK_REPLICATE_NOISE_A"]),
    "bottleneck_delta_floor_A": (0.5, ["DOCK_BOTTLENECK_DELTA_FLOOR"]),
    # ---- multiconf (multiconf.py: energy_std 解读地板) ----
    "energy_std_floor": (0.5, ["DOCK_MULTICONF_STD_FLOOR"]),
}

# 阈值人类可读释义(供 field_docs 的 evidence_registry 文档与报告引用, 单一事实来源)
THRESHOLD_DOCS: Dict[str, Dict[str, str]] = {
    "contact_unreliable": {"cat": "reliability",
        "meaning": "口袋接触分数低于此值=配体漂浮/偏靶(unreliable)"},
    "contact_warn": {"cat": "reliability",
        "meaning": "接触分数低于此值=弱接触(warn)"},
    "buried_clash": {"cat": "reliability",
        "meaning": "对撞数≥此且埋藏占比≥buried_frac=配体埋入蛋白内部"},
    "buried_frac": {"cat": "reliability",
        "meaning": "配体表面被完全包埋占比阈值(与 buried_clash 联用判埋藏)"},
    "energy_abs_max": {"cat": "reliability",
        "meaning": "|对接能量| > 此值=能量失真/不可信"},
    "heavy_max": {"cat": "reliability",
        "meaning": "配体重原子数 > 此值=超出 Vina 稳健尺度, 评分易失真"},
    "top_rmsd_warn": {"cat": "reliability",
        "meaning": "top1 与 top2 pose RMSD > 此值=对接未收敛(warn)"},
    "pose_scatter_warn": {"cat": "reliability",
        "meaning": "全 pose 两两自散 median RMSD > 此值=结合模式未收敛(soft)"},
    "nuc_offsite": {"cat": "reliability",
        "meaning": "亲核原子→配体氧距离 > 此值=配体不在催化位点(至少 5 Å 红线)"},
    "orient_nuc_max": {"cat": "orientation",
        "meaning": "亲核原子→scissile 羰基碳 接触阈值(Å), 成酰基酶几何前置"},
    "orient_his_o_max": {"cat": "orientation",
        "meaning": "His NE2→离去氧 接触阈值(Å), 质子 relay 闭合判据"},
    "orient_offsite_A": {"cat": "orientation",
        "meaning": "oriented=0 且最近 scissile C→亲核均值超此值=对接压根没摆进催化中心"},
    "orient_noise_frac": {"cat": "orientation",
        "meaning": "取向占比差 |Δfrac| 噪声地板(比例量, n=9 二项标准误~0.16)"},
    "orient_dist_noise_A": {"cat": "orientation",
        "meaning": "摆位距离差 |Δ距离| 噪声地板(Å), 与占比差各自独立判定"},
    "retention_stack_A": {"cat": "retention",
        "meaning": "芳香产物碳→受体芳香碳≤此值=仍卡在芳香锚定接触壳层内(滞留)"},
    "retention_noise_frac": {"cat": "retention",
        "meaning": "芳香滞留占比差 |Δfrac| 噪声地板(比例量)"},
    "cross_frame_warn_A": {"cat": "retention",
        "meaning": "ref 盒心用于 alt 时, alt 自身三联体距 ref 盒心超此值→能量不可相减"},
    "water_clearance_A": {"cat": "deacylation",
        "meaning": "水分子氧中心到蛋白重原子 vdW 表面余隙下限(Å, 水探针半径)"},
    "water_nuc_min_A": {"cat": "deacylation",
        "meaning": "亲核距离窗下限(Å); 短于=已成键/半键"},
    "water_nuc_max_A": {"cat": "deacylation",
        "meaning": "亲核距离窗上限(Å); 长于=不构成攻击构型"},
    "bd_angle_target": {"cat": "deacylation",
        "meaning": "Bürgi–Dunitz 攻击角目标值(度)"},
    "bd_angle_tol": {"cat": "deacylation",
        "meaning": "Bürgi–Dunitz 角容忍窗(度)"},
    "his_water_max_A": {"cat": "deacylation",
        "meaning": "水分子氧→His NE2 上限(Å, 通用氢键上限)"},
    "search_radius_A": {"cat": "deacylation",
        "meaning": "水位点搜索半径(Å, 绕 scissile 羰基碳)"},
    "search_step_A": {"cat": "deacylation",
        "meaning": "水位点搜索网格步长(Å)"},
    "deacyl_noise_vol_A3": {"cat": "deacylation",
        "meaning": "去酰化水空间差噪声地板(Å³, ≈一个水分子体积)"},
    "conf_entropy_noise_J_per_mol_K": {"cat": "conformation_entropy",
        "meaning": "局部构象熵差 |ΔS| 噪声地板(J/mol·K)"},
    "bottleneck_replicate_noise_A": {"cat": "pocket_tunnel",
        "meaning": "喉部坐标距离复现噪声地板(Å, 由序列不变对照实测)"},
    "bottleneck_delta_floor_A": {"cat": "pocket_tunnel",
        "meaning": "瓶颈半径差 |Δ| ≥ 此值才算真换档(Å)"},
    "energy_std_floor": {"cat": "multiconf",
        "meaning": "多构象能量 std 解读地板(kcal/mol); ≥此值=结论对起始构象敏感"},
}


def thresholds() -> Dict[str, float]:
    """返回全部阈值(dict, float)。默认值见 THRESHOLD_SPECS。

    覆盖优先级: 新统一名 DOCK_EVID_<NAME>(大写) > 历史别名 env > 默认值。
    非数字 env 回退默认并打 WARNING(不裸栈)。
    """
    out: Dict[str, float] = {}
    for name, (default, aliases) in THRESHOLD_SPECS.items():
        val = default
        new_env = os.environ.get("DOCK_EVID_" + name.upper())
        if new_env is not None:
            try:
                val = float(new_env)
            except ValueError:
                LOG.warning("环境变量 DOCK_EVID_%s=%r 非数字, 回退缺省 %g",
                            name.upper(), new_env, default)
        else:
            for a in aliases:
                av = os.environ.get(a)
                if av is not None:
                    try:
                        val = float(av)
                        break
                    except ValueError:
                        LOG.warning("环境变量 %s=%r 非数字, 忽略", a, av)
        out[name] = val
    return out


# 模块加载时定稿一份(与 env_float 同语义: import 期读一次)。
THRESHOLDS: Dict[str, float] = thresholds()


# ----------------------------------------------------------------------------
# 2. 通用 verdict 辅助(纯函数, 可单测)
# ----------------------------------------------------------------------------
def _as_num(v: Any) -> Optional[float]:
    """兼容 float 与 {"mean": ...} 两种形态(orientation/retention 跨种子聚合口径)。"""
    if isinstance(v, dict):
        v = v.get("mean")
    return v if isinstance(v, (int, float)) else None


def classify_delta(delta: Optional[float], floor: float,
                  direction: Optional[str] = None) -> Dict[str, Any]:
    """对单一 signed delta 做"是否越过噪声地板"判定。

    direction: None=任意方向 / 'pos'=只关心正(如能量更负=负, 用 'neg') /
               'neg'=只关心负。返回 {significant, sign, abs, passes,
               direction_ok, message}。
    """
    if delta is None:
        return {"significant": False, "sign": 0, "abs": None,
                "passes": False, "direction_ok": False, "message": "delta 缺失"}
    a = abs(delta)
    passes = a >= floor
    s = 1 if delta > 0 else (-1 if delta < 0 else 0)
    dir_ok = (direction is None
              or (direction == "pos" and s > 0)
              or (direction == "neg" and s < 0))
    sig = passes and dir_ok
    if sig:
        msg = "显著(越过地板且方向符合)"
    elif not passes:
        msg = "低于噪声地板, 与抽样抖动不可分辨"
    else:
        msg = "越过地板但方向不符限定"
    return {"significant": sig, "sign": s, "abs": a, "passes": passes,
            "direction_ok": dir_ok, "message": msg}


def frac_delta_verdict(wt, mut, *, floor: float, label: str = "frac",
                       n_wt=None, n_mut=None) -> Dict[str, Any]:
    """比例量(占比/分数)的噪声地板判定。wt/mut 各为 float 或 {"mean":...}。

    返回 {available, delta, passes, floor, fw, fm, label, reason}。
    passes = |mut − wt| ≥ floor —— 业务模块据此决定"能不能写方向性结论"。
    """
    fw, fm = _as_num(wt), _as_num(mut)
    if fw is None or fm is None:
        return {"available": False, "delta": None, "passes": False,
                "floor": floor, "fw": fw, "fm": fm, "label": label,
                "reason": "任一侧占比为 None, 不得写方向性结论"}
    delta = round(float(fm) - float(fw), 3)
    passes = abs(delta) >= floor
    return {"available": True, "delta": delta, "passes": passes,
            "floor": floor, "fw": fw, "fm": fm, "label": label,
            "reason": ("超噪声地板" if passes
                       else "低于噪声地板, 与抽样抖动不可分辨")}


def dist_delta_verdict(wt, mut, *, floor: float, label: str = "dist",
                       unit: str = "Å") -> Dict[str, Any]:
    """距离量的噪声地板判定(与占比量**独立**判定; 见 orientation_sides 注释)。"""
    fw, fm = _as_num(wt), _as_num(mut)
    if fw is None or fm is None:
        return {"available": False, "delta": None, "passes": False,
                "floor": floor, "fw": fw, "fm": fm, "label": label, "unit": unit,
                "reason": "任一侧距离缺失"}
    delta = round(float(fm) - float(fw), 3)
    passes = abs(delta) >= floor
    return {"available": True, "delta": delta, "passes": passes,
            "floor": floor, "fw": fw, "fm": fm, "label": label, "unit": unit,
            "reason": ("超距离噪声地板" if passes
                       else f"|Δ距离|={abs(delta):g}{unit} < 地板 {floor}{unit}")}


def vol_delta_verdict(wt, mut, *, floor: float, label: str = "vol",
                      unit: str = "Å³") -> Dict[str, Any]:
    """体积量的噪声地板判定(去酰化水空间差: 一个水分子体积 ≈ 12 Å³)。"""
    fw, fm = _as_num(wt), _as_num(mut)
    if fw is None or fm is None:
        return {"available": False, "delta": None, "passes": False,
                "floor": floor, "fw": fw, "fm": fm, "label": label, "unit": unit,
                "reason": "任一侧体积缺失"}
    delta = round(float(fm) - float(fw), 2)
    passes = abs(delta) >= floor
    return {"available": True, "delta": delta, "passes": passes,
            "floor": floor, "fw": fw, "fm": fm, "label": label, "unit": unit,
            "reason": ("超体积噪声地板" if passes
                       else f"Δ体积 {delta:+.1f}{unit} 小于一个水分子的体积 "
                            f"{floor}{unit} —— 没有'多/少一个水位点'的物理意义")}


def replicate_sign_consistent(deltas: Sequence[float], *, floor: float,
                              require_majority: bool = True) -> Dict[str, Any]:
    """多轮独立重复(多种子/多松弛)的"方向一致性"判定(对应既有判词里的
    replicate_required): 所有 delta 同号 且 多数(≥ceil(n/2))越过地板, 才认为
    '单轮观测收敛为可写的方向性结论'。

    返回 {consistent, n_total, n_pass, all_same_sign, direction, message}。
    """
    vals = [float(d) for d in deltas if isinstance(d, (int, float))]
    if not vals:
        return {"consistent": False, "n_total": 0, "n_pass": 0,
                "all_same_sign": False, "direction": 0,
                "message": "无有效 delta"}
    n = len(vals)
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
    all_same = (len(set(signs)) == 1) and (0 not in signs)
    n_pass = sum(1 for v in vals if abs(v) >= floor)
    majority = n_pass >= (n + 1) // 2
    consistent = all_same and (not require_majority or majority)
    direction = signs[0] if all_same else 0
    msg = (f"{n_pass}/{n} 轮越过地板且方向一致" if consistent
           else f"方向不一致或仅 {n_pass}/{n} 轮越过地板, 单轮观测不足以成结论")
    return {"consistent": bool(consistent), "n_total": n, "n_pass": int(n_pass),
            "all_same_sign": bool(all_same), "direction": int(direction),
            "message": msg}


def conformation_sensitive(energy_std: Optional[float]) -> Dict[str, Any]:
    """multiconf 的 energy_std 解读: 大=结论对起始构象敏感。

    地板取 energy_std_floor(默认 0.5 kcal/mol, 同 Vina 打分噪声量级)。
    返回 {sensitive(bool|None), level, note}。
    """
    floor = THRESHOLDS["energy_std_floor"]
    if energy_std is None:
        return {"sensitive": None, "level": "n/a",
                "note": "构象数<2, 无法估计散布(std 为 None)"}
    if energy_std < floor:
        return {"sensitive": False, "level": "robust",
                "note": f"energy_std={energy_std:.3f} < {floor}, "
                        f"结论对起始构象稳健"}
    return {"sensitive": True, "level": "sensitive",
            "note": f"energy_std={energy_std:.3f} ≥ {floor}, 结论对起始构象敏感, "
                    f"需多起点/多构象复核"}


class VerdictBuilder:
    """自否证判词装配器(减少各 verdict 函数的样板)。

    用法:
        v = VerdictBuilder()
        v.set(ok=True, noise_floor=0.15)
        v.ban("禁写 X")
        v.allow("可写 Y")
        v.narrative("...")
        return v.build()
    build() 输出与既有 verdict 同构: {ok, applicable, banned_phrases,
    allowed_phrases, ..., narrative}。
    """

    def __init__(self, ok: bool = True, applicable: bool = True):
        self.ok = ok
        self.applicable = applicable
        self.banned: List[str] = []
        self.allowed: List[str] = []
        self.data: Dict[str, Any] = {}
        self._narrative = ""

    def set(self, **kw) -> "VerdictBuilder":
        self.data.update(kw)
        return self

    def ban(self, phrase: str) -> "VerdictBuilder":
        self.banned.append(phrase)
        return self

    def allow(self, phrase: str) -> "VerdictBuilder":
        self.allowed.append(phrase)
        return self

    def narrative(self, text: str) -> "VerdictBuilder":
        self._narrative = text
        return self

    def build(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": self.ok,
            "applicable": self.applicable,
            "banned_phrases": list(self.banned),
            "allowed_phrases": list(self.allowed),
            "narrative": self._narrative,
        }
        out.update(self.data)
        return out


__all__ = [
    "THRESHOLD_SPECS", "THRESHOLD_DOCS", "thresholds", "THRESHOLDS",
    "classify_delta", "frac_delta_verdict", "dist_delta_verdict",
    "vol_delta_verdict", "replicate_sign_consistent",
    "conformation_sensitive", "VerdictBuilder",
]
