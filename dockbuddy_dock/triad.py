# -*- coding: utf-8 -*-
"""triad.py —— 催化三联体几何检出与受体链选择(纯结构判据, 不依赖对接)。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import math
from typing import Any, Dict, Optional
from ._core import LOG, ob, env_float
from .pdbio import _pdb_protein_chains


# ============================================================================
# 5. 催化三联体自动检测 (通用: Ser/Cys-His-Asp/Glu; 兼容突变 Ala)
# ============================================================================

# His_NE2 → 亲核原子(SerOG/CysSG/AlaCB)的最大可接受距离(Å)。
# 真实三联体 ≈2.8~3.4 Å; 物理上超过 5 Å 就不可能构成催化三联体。
NUC_MAX_DIST_A: float = env_float("DOCK_TRIAD_NUC_MAX", 5.0)  # 非数字 env 不再 import 期崩溃(第六轮 C7)


def _atom_meta(a):
    res = a.GetResidue()
    if res is None:
        return (None, None, None, None)
    # 原子名从 OBResidue 取: 本机 OpenBabel 3.2.1 的 OBAtom 无 GetAtomID() 方法,
    #   正确签名是 res.GetAtomID(a)(在残基上、传入原子)。若个别原子残基数据异常
    #   (GetResidue 返回非预期对象), 捕获后 aname=None, 由调用方跳过该原子。
    try:
        aname = res.GetAtomID(a).strip()
    except Exception:
        aname = None
    return (aname, res.GetName().strip(), res.GetNum(), res.GetChain())

def detect_catalytic_triad(pdb_path: str, chain_filter: str = "") -> Optional[Dict[str, Any]]:
    """检测经典水解酶催化三联体。返回最佳三联体字典或 None。
    判定: His-ND1 ~ Asp/Glu-OD 离子对(<3.6A), 且 His-NE2 附近有亲核残基
          (Ser-OG / Cys-SG / 突变 Ala-CB / 或一般 Ser/Cys 主链)。
    兼容两类: Ser-His-Asp(丝氨酸水解酶) 与 Cys-His-Asn(半胱氨酸蛋白酶)。
    chain_filter: 若给定(如 'A'), 仅在指定链内检测, 保证与受体预处理链一致。"""
    LOG.info("自动检测催化三联体..." + (f" (限链 {chain_filter})" if chain_filter else ""))
    conv = ob.OBConversion(); conv.SetInFormat("pdb")
    mol = ob.OBMol(); conv.ReadFile(mol, pdb_path)

    def collect(names):
        d = {}
        for a in ob.OBMolAtomIter(mol):
            n, rn, ri, rc = _atom_meta(a)
            if rn in names and n and (not chain_filter or rc == chain_filter):
                d.setdefault((ri, rc), {})[n] = (a.GetX(), a.GetY(), a.GetZ())
        return d

    his = collect(("HIS", "HID", "HIE", "HIP"))
    asp = collect(("ASP", "GLU"))          # Asp 或 Glu(部分酶用 Glu)
    asn = collect(("ASN", "GLN"))          # Cys-His-Asn 类的酸/碱对中 Asn
    ser = collect(("SER",))
    cys = collect(("CYS",))
    ala = collect(("ALA",))                # 突变体(如 S170A)
    # (ri,rc) -> 真实残基名(2026-10-03 修复: collect 合并收集后丢失原名, 旧实现把
    # GLU 一律记成 "ASP"、GLN 记成 "ASN" —— 催化键写成 ASP<n> 匹配不上受体 pdbqt
    # 的 GLU<n>A, 酸成员在 pose 分析的 cat_norm 匹配中被静默丢弃)
    resname_map: Dict = {}
    for a in ob.OBMolAtomIter(mol):
        n, rn, ri, rc = _atom_meta(a)
        if n and (not chain_filter or rc == chain_filter):
            resname_map.setdefault((ri, rc), rn)

    best = None
    for hk, h in his.items():
        if "ND1" not in h or "NE2" not in h:
            continue
        for ak, a in list(asp.items()) + list(asn.items()):
            # 离子对距离取该残基全部羧基氧(OD1+OD2 / OE1+OE2)的最近者(2026-10-03
            # 修复: 旧实现只查第一个存在的氧名——ASP 永远 OD1、GLU 永远 OE1,
            # His-ND1 实际与 OD2/OE2 成对的常见构象被漏检 -> 盒子回退 protein_centroid)
            od = None; od_d = None
            for odname in ("OD1", "OD2", "OE1", "OE2"):
                if odname in a:
                    d_ = math.dist(h["ND1"], a[odname])
                    if od_d is None or d_ < od_d:
                        od, od_d = odname, d_
            if od is None:
                continue
            if od_d > 3.6:
                continue
            # 找 His NE2 最近的亲核残基 (Ser/Cys/Ala)
            # 阈值 5.0 A(修复 2026-10-02, 旧为 9.0): 真实催化三联体的 HisNE2–亲核
            # (SerOG/CysSG/AlaCB) 距离约 2.8~3.4 A; 9.0 A 的松阈值会把远处同类型残基
            # 当成亲核(盒子/判定被带偏), 且"找最近"算法下阈值只起上界作用。
            # Ser/Cys 优先于 Ala(2026-10-03 修复): Ala 占残基 ~8%, 蛋白核心内 CB 距
            # His 常见 3-4A——真催化 Ser 存在时被更近的无关 Ala CB 顶替会让盒心与
            # 催化键同时错。Ala 仅作 S->A 突变体兜底(无 Ser/Cys 候选, 或候选本就是 Ala)。
            cand = None; cd = NUC_MAX_DIST_A; cand_atom = None
            for sk, s in ser.items():
                if "OG" in s and math.dist(h["NE2"], s["OG"]) < cd:
                    cand, cd, cand_atom = sk, math.dist(h["NE2"], s["OG"]), "OG"
            for ck, c in cys.items():
                if "SG" in c and math.dist(h["NE2"], c["SG"]) < cd:
                    cand, cd, cand_atom = ck, math.dist(h["NE2"], c["SG"]), "SG"
            # 2026-10-04 全盘审计: 原条件 `cand is None or cand in ala` —— cand
            # 只可能来自 ser/cys 的键, 与 ala 字典键不交集, 后半恒 False(死代码)。
            if cand is None:
                for ak2, al in ala.items():
                    if "CB" in al and math.dist(h["NE2"], al["CB"]) < cd:
                        cand, cd, cand_atom = ak2, math.dist(h["NE2"], al["CB"]), "CB"
            if cand is None:
                continue
            # 取三残基 Cα 质心
            ca_pts = []
            for key in (cand, hk, ak):
                for atom in ob.OBMolAtomIter(mol):
                    n, rn, ri, rc = _atom_meta(atom)
                    if ri == key[0] and rc == key[1] and n == "CA":
                        ca_pts.append((atom.GetX(), atom.GetY(), atom.GetZ()))
                        break
            if len(ca_pts) != 3:
                continue
            cx = sum(p[0] for p in ca_pts) / 3
            cy = sum(p[1] for p in ca_pts) / 3
            cz = sum(p[2] for p in ca_pts) / 3
            score = od_d + cd   # 越小越典型
            triad = {
                "nuc": {"resnum": cand[0], "chain": cand[1],
                        "resname": "SER" if cand in ser else ("CYS" if cand in cys else "ALA"),
                        "atom": cand_atom},
                "his": {"resnum": hk[0], "chain": hk[1], "resname": "HIS"},
                "asp": {"resnum": ak[0], "chain": ak[1],
                        "resname": resname_map.get(ak, "ASP" if ak in asp else "ASN")},
                "ca_centroid": (cx, cy, cz),
                "his_asp_d": od_d,
                "his_ne2_nuc_d": cd,
                "score": score,
            }
            if best is None or score < best["score"]:
                best = triad
            LOG.info(f"  候选三联体: {triad['nuc']['resname']}{cand[0]}{cand[1]} "
                     f"- His{hk[0]}{hk[1]} - {triad['asp']['resname']}{ak[0]}{ak[1]} "
                     f"(HisND1-AspOD={triad['his_asp_d']:.2f}A, HisNE2-核{cd:.2f}A)")
    if best:
        LOG.info(f"✅ 选定催化三联体: {best['nuc']['resname']}{best['nuc']['resnum']}{best['nuc']['chain']} "
                 f"- His{best['his']['resnum']}{best['his']['chain']} "
                 f"- {best['asp']['resname']}{best['asp']['resnum']}{best['asp']['chain']}")
        return best
    LOG.error("未检测到催化三联体; 请手动指定催化残基或盒子中心")
    return None

def pick_receptor_chain(pdb_path: str) -> str:
    """自动选链: 含催化三联体的链优先; 否则最长蛋白链。"""
    triad = detect_catalytic_triad(pdb_path)
    if triad:
        return triad["nuc"]["chain"] or ""
    counts = _pdb_protein_chains(pdb_path)
    if not counts:
        return ""
    return max(counts, key=counts.get)

__all__ = [
    "_atom_meta",
    "detect_catalytic_triad",
    "pick_receptor_chain",
]
