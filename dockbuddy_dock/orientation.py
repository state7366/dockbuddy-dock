# -*- coding: utf-8 -*-
"""orientation.py —— 「催化取向(catalytic orientation)」判据(libdock core 层)。

为什么需要它(能量给不了的东西)
--------------------------------
某丝氨酸水解酶对接聚酯寡聚物的实测结论(2026-10-04, A5): 能量极稳
**−4.813 ± 0.163 kcal/mol**(跨 5 个 AF 模型, 极差 0.438), 但 top1 pose 两两差
5.11 Å(最大 7.34 Å)、单次对接 9 个 pose 自己就散开 6.92 Å。也就是说:
**Borel 打分面能有效排序"结合", 却根本没把"结合模式"定下来**。

于是出现一个必须堵的叙事漏洞: 看到突变体能量更负, 就写"底物结合更紧 → 活性更高”。
但能量同/异都不能区分"底物被摆成可水解姿态"和"底物贴在别处" —— 后者对 kcat
毫无贡献, 却会产生一个同样漂亮的负能量。真正与 kcat 挂钩的量是:
**对接出的 pose 里, 有多少个把可水解的酯羰基碳摆到催化 Ser OG 旁边(成酰基酶的
几何前置条件)**。这就是本模块的 `oriented`。

判据(纯几何, 不依赖任何额外力场/协议)
--------------------------------------
- **scissile 位点** = 配体上的可水解酯羰基 C:
  ① C 与 O 的共价距离落在 [1.05, 1.35] Å(C=O 实测 1.21 Å; 上限 1.35 是为覆盖
     质子化/畸变构象, 下限排除 spuriously 接近的 O);
  ② 该 C 的**另一个** O 是酯氧(还连着第二个 C) → 酯 → 可形成酰基酶 = scissile;
     若该 O 只连 H(即 −COOH) → 羧酸 = **酸型位点**, 是水解**产物**端,
     **不形成酰基酶, 不计入 oriented**(羧酸产物端必须排除, 否则整个
     产物谱都会被误判成"摆好了")。
- **oriented(可攻击姿态)** = 存在至少一个 scissile 位点的**羰基碳**进入亲核
  原子的接触球: dist(scissile C, Ser OG) ≤ NUC_TO_SCISSILE_C_MAX(默认 3.5 Å)。
  用碳而不是氧: 成酰基酶时 Ser OG 攻击的就是羰基碳, C···OG 才是真正的成键距离
  (OG···O=C 的氧距离会系统性偏长 ~1.4 Å)。
- **relay_ok(质子通路闭合)** = oriented 且该位点的**离去氧**(酯氧)进入 His
  NE2 的接触球 ≤ HIS_TO_LEAVING_O_MAX(默认 3.5 Å) —— Ser-His-Asp 完整 relay
  要求 His 既能接 Ser 的质子又能喂给离去氧。
- **both_ended** = 双酯/双端底物(两侧各一个酯) 有两个及以上 scissile 位点
  同时进入接触球 → "双手柄都摆好", 是比单端更有说服力的催化前姿态。

统计口径(必须多种子 × 全 pose, 不能只看 top1)
----------------------------------------------
单次对接 num_modes=9 时, 9 个 pose 自己就散 6.92 Å → 用 top1 的"是不是取向 pose"
下结论等于抛硬币。故 `summarize_side()` 统计的是**该侧全部 pose 的 oriented
占比**, 并随 `--eval_smiles` 的种子维度逐种子汇总成 mean±std。

自否证(`orientation_sides`)
----------------------------
占比是**比例量**, 小样本(n=9/侧)天然噪声大, 所以判词与 pocket_tunnel 同款:
`orientation_sides(wt, mut)` 是**唯一**产出 banned/allowed/narrative 的地方,
report 照抄不得改写。规则(见函数 docstring):
两侧都没检出取向 → 全禁且升级为「本轮对接未复现催化复合物, 能量不得用于机理
归因」; 两侧占比差 < ORIENT_NOISE_FRAC → 禁写"取向变好/变差"; 只有一侧检出时
必须带"单侧检出"限定。

摆位**距离**量(`nearest_mean_A`)另设 ORIENT_DIST_NOISE_A 地板, 与占比差**各自独立**
判定: 占比差超地板不等于距离差可读, 占比差在地板内也不因距离差略大而自动放行。
理由见该常量注释(2026-10-05 实测 0.99 Å 差会被误读成"摆得更近")。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 阈值单一事实源(2026-10-07 证据门槛注册表): 取值统一从 evidence.THRESHOLDS 读,
# 历史 env 名(DOCK_ORIENT_*)仍经 evidence 别名机制生效。
from .evidence import THRESHOLDS as _EV, frac_delta_verdict, dist_delta_verdict
from .geom import ref_key_match
from .poses import _NUC_ATOM_BY_RESNAME, load_receptor_coords


# ----------------------------------------------------------------------------
# 阈值(单一事实源; 均可由 env 覆盖, 改了就要换口径标记)
# ----------------------------------------------------------------------------
# 配体内按共价距离构键: pdbqt 不自带键表(只有 ROOT/TREE/BRANCH 柔性树), 而
# scissile 识别完全依赖成键拓扑, 所以这里必须先"看距离重建"共价图。
# 小分子内最长常规共价键 ≈1.75 Å(C–Cl 1.77 是例外, 塑料降解底物不含卤素)。
COVALENT_BOND_MAX = 1.75
# 羰基 C=O 判据窗(实测酮/酯羰基 1.20~1.23 Å)
CARBONYL_O_MIN = 1.05
CARBONYL_O_MAX = 1.35
# 酯氧 C–O(单键) 判据上限(实测 1.34~1.43 Å)
ESTER_O_MAX = 1.55

# 亲核原子 → scissile 羰基碳 的接触阈值(Å)。成酰基酶时 Ser OG 攻击羰基碳,
# 过渡态 C···O 距离 ≈1.3~1.4 Å, 而可反应的前复合物上端~3.2 Å; 3.5 是留出
# 抽样/构象抖动的安全上界, 与 triad.NUC_MAX_DIST_A(5.0, 纯"是不是三联体")不冲突:
# 那里判的是 His NE2↔Ser OG 的离子对, 这里是底物→亲核, 量不同、不可互换。
# 取值见 libdock.evidence.THRESHOLDS["orient_nuc_max"]。
NUC_TO_SCISSILE_C_MAX = _EV["orient_nuc_max"]
# His NE2 → 离去氧(酯氧) 的接触阈值(Å), 质子 relay 闭合判据
HIS_TO_LEAVING_O_MAX = _EV["orient_his_o_max"]
# "配体压根没进活性位"的判定线(Å): 一侧 oriented=0 且最近 scissile C→亲核的
# **均值**超过它, 就不是"取向不好", 而是**对接压根没有把配体摆进催化中心** ——
# 典型症状是盒心错位/跨坐标系(同一批 AF 模型原始坐标相差 9~25 Å, 拿一个模型的
# 盒心去对接另一个模型, Vina 会在 20 Å 外照样返回一个"最优解")。
# 实测(2026-10-05, m4 vs m0 双边对照): 该值 27.96 Å, 而正常侧 6.83 Å。
ORIENT_OFFSITE_A = _EV["orient_offsite_A"]
# 取向占比差值的"噪声地板"(比例量)。n=9 pose 时二项标准误
# sqrt(p(1-p)/9) 在 p=0.5 处已达 0.167 —— 地板取 0.15 即约等于 1σ, 低于它
# 的差值与"抽样抖动"不可分辨, 一律禁写方向性结论。
ORIENT_NOISE_FRAC = _EV["orient_noise_frac"]
# 摆位**距离**量的"噪声地板"(Å)。占比量有 ORIENT_NOISE_FRAC 兜底, 距离量起初
# 没有 —— 2026-10-05 某丝氨酸水解酶真实对照(WT vs 突变体, 同源坐标系)暴露: 两侧
# oriented 占比 0.222 vs 0.333(Δ=0.111 < 地板 0.15, 已正确判"不可分辨"), 但
# 最近 scissile C→亲核均值 7.07 vs 6.08 Å(差 0.99 Å)同样在摆位抖动内。若不给
# 距离地板, 这一行数字会被顺手读成"突变让底物更靠近催化中心" —— 正是本模块
# 存在的目的所要堵的那类误读。1.0 Å 相对 3.5 Å 接触球阈值(约 29%)是保守取值。
ORIENT_DIST_NOISE_A = _EV["orient_dist_noise_A"]


# ----------------------------------------------------------------------------
# 1. 配体内部拓扑(纯几何重建共价键)
# ----------------------------------------------------------------------------
def ligand_bond_graph(atoms: Sequence[Sequence[Any]]) -> List[List[int]]:
    """按共价距离阈值重建配体内共价键, 返回邻接表(索引 = atoms 下标)。

    atoms = parse_pdbqt_poses 的 pose 元组 (x, y, z, elem, name, resn, resi)。
    只比较**重原子**(elem != 'H'): 与氢成键只会让图更胖, 而 scissile 识别
    只关心 C/O, 氢的加入不影响结论, 反而会稀释成键判据。
    """
    n = len(atoms)
    heavy = [i for i in range(n) if str(atoms[i][3]).upper() != "H"]
    graph: List[List[int]] = [[] for _ in range(n)]
    for ii in range(len(heavy)):
        for jj in range(ii + 1, len(heavy)):
            a, b = heavy[ii], heavy[jj]
            if math.dist(atoms[a][:3], atoms[b][:3]) <= COVALENT_BOND_MAX:
                graph[a].append(b)
                graph[b].append(a)
    return graph


def find_scissile_sites(atoms: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    """识别配体上的 scissile(可水解酯羰基)位点与酸型(产物端)位点。

    返回 [{"index": 羰基C下标, "carbonyl_O": 羰基O下标, "leaving_O": 单键O下标,
           "acid": 是否羧酸(True=产物端不计入 oriented), "n_carbonyl": 双键O数}]。

    设计要点(2026-10-05): **必须把羧酸端排除出 oriented**。水解后的单/二羧酸
    产物(如对苯二甲酸及其单酯)已不 forms acyl-enzyme, 若把
    `O=C(O)c1ccc(C(=O)O)cc1` 算成"摆好了两个可攻击位点", 产物对接会凭空产出
    100% oriented 的假阳性, 直接把「产物释放限速」这类结论推反。故 acid=True
    的位点只做记录、不参与 oriented。
    """
    graph = ligand_bond_graph(atoms)
    elem = [str(a[3]).upper() for a in atoms]
    sites: List[Dict[str, Any]] = []
    for i, e in enumerate(elem):
        if e != "C":
            continue
        nb = graph[i]
        oxy = [j for j in nb if elem[j] == "O"]
        if not oxy:
            continue
        carbonyl = [j for j in oxy
                    if CARBONYL_O_MIN <= math.dist(atoms[i][:3], atoms[j][:3]) <= CARBONYL_O_MAX]
        if not carbonyl:
            continue
        for j in oxy:
            if j in carbonyl:
                continue
            # 单键氧(酯氧/羟基氧): 除羰基碳外还连 C → 酯氧; 否则是羟基 → 酸型
            leaves_carbon = any(k != i and elem[k] == "C" for k in graph[j])
            sites.append({
                "index": i,
                "carbonyl_O": carbonyl[0],
                "leaving_O": j,
                "acid": not leaves_carbon,
                "n_carbonyl": len(carbonyl),
            })
    return sites


# ----------------------------------------------------------------------------
# 2. 受体锚点(亲核原子 / His NE2)
# ----------------------------------------------------------------------------
def receptor_anchors(rec_pdbqt: str, triad: Dict[str, Any]) -> Dict[str, Any]:
    """从受体 pdbqt 取亲核原子坐标与 His NE2 坐标。

    triad = detect_catalytic_triad 的输出(含 nuc/his/asp 三个
    {resname, resnum, chain})。亲核原子**必须按残基名取 OG/SG 而非整残基最近
    原子**(同 poses.py `_NUC_ATOM_BY_RESNAME` 的口径): 取整残基会把 CB/主链
    的最近距离当成"亲核距离", 系统性偏短, 3.5 Å 阈值会被刷成满分。
    """
    rec = load_receptor_coords(rec_pdbqt)
    nuc_res = (triad.get("nuc") or {})
    his_res = (triad.get("his") or {})
    nuc_atom_name = _NUC_ATOM_BY_RESNAME.get(str(nuc_res.get("resname", "")).upper())
    nuc_pts, his_pts = [], []
    nuc_key = f"{nuc_res.get('resname','')}{nuc_res.get('resnum','')}{nuc_res.get('chain','')}"
    hkey = f"{his_res.get('resname','')}{his_res.get('resnum','')}{his_res.get('chain','')}"
    for k, pts in rec.items():
        if nuc_atom_name and ref_key_match(k, nuc_key):
            nuc_pts.extend([p for p in pts if len(p) > 3 and p[3] == nuc_atom_name])
        if his_res and ref_key_match(k, hkey):
            his_pts.extend([p for p in pts if len(p) > 3 and p[3] in ("NE2", "ND1")])
    return {
        "rec_coords": rec,
        "nuc_pts": nuc_pts,
        "his_pts": his_pts,
        "nuc_key": f"{nuc_res.get('resname','')}{nuc_res.get('resnum','')}{nuc_res.get('chain','')}",
        "his_key": f"{his_res.get('resname','')}{his_res.get('resnum','')}{his_res.get('chain','')}",
        "nuc_atom": nuc_atom_name or "",
    }


# ----------------------------------------------------------------------------
# 3. 单 pose 取向
# ----------------------------------------------------------------------------
def pose_orientation(pose_atoms: Sequence[Sequence[Any]],
                     sites: Sequence[Dict[str, Any]],
                     nuc_pts: Sequence[Any],
                     his_pts: Sequence[Any],
                     nuc_max: float = NUC_TO_SCISSILE_C_MAX,
                     his_max: float = HIS_TO_LEAVING_O_MAX) -> Dict[str, Any]:
    """单 pose 的催化取向判定。

    返回 {"n_sites"(含酸型总数), "n_ester"(酯型 scissile 数),
          "nearest_A"(最近酯型羰基C→亲核, None=无 scissile/无亲核点),
          "oriented", "n_close", "close_indices", "relay_ok", "relay_A",
          "per_site":[...]}

    oriented 用**酯型**位点(acid=False)判定; 酸型位点的距离仍记进 per_site 供
    人工判读(例如"产物端摆得很好但它是羧酸, 不算数")。
    """
    ester_sites = [s for s in sites if not s["acid"]]
    if not ester_sites or not nuc_pts:
        return {"n_sites": len(sites), "n_ester": len(ester_sites),
                "nearest_A": None, "oriented": False, "n_close": 0,
                "close_indices": [], "relay_ok": False, "relay_A": None,
                "per_site": [], "note": "无 scissile 位点或无亲核点"}
    per_site: List[Dict[str, Any]] = []
    for s in sites:
        c = pose_atoms[s["index"]][:3]
        d_nuc = min(math.dist(c, p[:3]) for p in nuc_pts)
        d_his = None
        if his_pts:
            o = pose_atoms[s["leaving_O"]][:3]
            d_his = min(math.dist(o, p[:3]) for p in his_pts)
        per_site.append({
            "index": s["index"],
            "acid": s["acid"],
            "nuc_C_A": round(d_nuc, 2),
            "relay_A": round(d_his, 2) if d_his is not None else None,
            "oriented": (not s["acid"]) and d_nuc <= nuc_max,
        })
    ester_rows = [r for r in per_site if not r["acid"]]
    nearest = min(r["nuc_C_A"] for r in ester_rows)
    close = [r["index"] for r in ester_rows if r["nuc_C_A"] <= nuc_max]
    relay_ok = any(r["oriented"] and r["relay_A"] is not None and r["relay_A"] <= his_max
                   for r in ester_rows)
    return {
        "n_sites": len(sites), "n_ester": len(ester_sites),
        "nearest_A": round(nearest, 2),
        "oriented": bool(close),
        "n_close": len(close),
        "close_indices": close,
        "both_ended": len(close) >= 2,
        "relay_ok": bool(relay_ok),
        "relay_A": min((r["relay_A"] for r in ester_rows if r["oriented"]
                        and r["relay_A"] is not None), default=None),
        "per_site": per_site,
    }


def pose_orientation_full(pose_atoms, rec_pdbqt, triad, nuc_max=NUC_TO_SCISSILE_C_MAX,
                          his_max=HIS_TO_LEAVING_O_MAX) -> Dict[str, Any]:
    """一步到位: 受体锚点 + scissile 识别 + 取向判定(供测试与独立调用)。"""
    anch = receptor_anchors(rec_pdbqt, triad)
    sites = find_scissile_sites(pose_atoms)
    return pose_orientation(pose_atoms, sites, anch["nuc_pts"], anch["his_pts"],
                            nuc_max=nuc_max, his_max=his_max)


# ----------------------------------------------------------------------------
# 4. 单侧汇总(全 pose + 跨种子)
# ----------------------------------------------------------------------------
def summarize_side(per_pose: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把一侧(WT 或突变体)跨种子、跨 pose 的取向明细汇总成可比对的表。

    per_pose = 逐 pose 的 pose_orientation 结果 + 可选 'energy' 键。
    oriented 是**比例量**, 必须报 n/total 与 frac, 不能只给一句"有没有"。
    """
    tot = len(per_pose)
    if not tot:
        return {"n_pose": 0, "n_oriented": 0, "oriented_frac": None,
                "n_relay": 0, "relay_frac": None, "n_scissile_sites": None,
                "n_both_ended": 0, "both_ended_frac": None,
                "nearest_mean_A": None, "nearest_min_A": None,
                "oriented_energy_mean": None, "oriented_energy_n": 0}
    n_or = sum(1 for p in per_pose if p.get("oriented"))
    n_relay = sum(1 for p in per_pose if p.get("relay_ok"))
    n_both = sum(1 for p in per_pose if p.get("both_ended"))
    near = [p["nearest_A"] for p in per_pose
            if p.get("nearest_A") is not None]
    en = [p["energy"] for p in per_pose if p.get("oriented") and p.get("energy") is not None]
    sites = [p.get("n_ester") for p in per_pose if p.get("n_ester") is not None]
    return {
        "n_pose": tot,
        "n_oriented": n_or,
        "oriented_frac": round(n_or / tot, 3),
        "n_relay": n_relay,
        "relay_frac": round(n_relay / tot, 3),
        "n_scissile_sites": max(sites) if sites else None,
        "n_both_ended": n_both,
        "both_ended_frac": round(n_both / tot, 3),
        "nearest_mean_A": round(sum(near) / len(near), 2) if near else None,
        "nearest_min_A": round(min(near), 2) if near else None,
        "oriented_energy_mean": round(sum(en) / len(en), 2) if en else None,
        "oriented_energy_n": len(en),
        "nuc_max_A": NUC_TO_SCISSILE_C_MAX,
        "his_max_A": HIS_TO_LEAVING_O_MAX,
    }


# ----------------------------------------------------------------------------
# 5. 判词单点(唯一允许产出"取向结论"的地方)
# ----------------------------------------------------------------------------
def orientation_sides(wt: Optional[Dict[str, Any]],
                      mut: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """WT / 突变体两侧 summarize_side 结果 → 可引用对比 + **自否证判词**。

    返回齐 {"ok", "orientation_applicable", "wt"/"mut" 简报,
            "oriented_frac_wt", "oriented_frac_mut", "oriented_frac_delta",
            "noise_floor", "banned_phrases", "allowed_phrases", "narrative"}。

    禁写规则(顺序即优先级, 命中即止):
      1. 任一侧缺数据/无 pose → 禁一切取向结论(含"两侧无差异")。
      2. 两侧 scissile 位点数 = 0 → 指标不适用(典型: 二羧酸产物)。
         可写"该底物无酯羰基, 取向判据不适用"。
      3. 两侧 oriented 均为 0 → **最硬的红线**: 本轮对接根本没复现催化复合物,
         此时任何能量差(含 ΔE)都不得用于机理归因; 禁写"活性提升/下降"。
      4. |Δfrac| < 噪声地板 → 禁写"取向改善/恶化"; 可写"占比无实质差异"。
      5. 单侧检出(一侧 0 一侧 >0) → 允许, 但必须带"单侧检出"限定, 并提示
         大概率是 pose 采样而非构象效应(n 小, 一个 pose 就能把 frac 推 0.11)。
      6. 两侧都检出且 Δ 超地板 → 允许写方向性结论, 同时强制带上占比数字与
         噪声地板, 不得只写形容词。

    入参形态宽容: oriented_frac 既可给 summarize_side 的 float, 也可给跨种子
    聚合的 {"mean":...} —— 两者都按数值读, 缺失(None/非数值)视为缺数据。

    注意: 本函数**不判断"能量变好"** —— 取向与能量是两条独立证据线, 合并
    叙事(例如"能量更负 + 取向更高 → kcat 提升")必须由调用方在 narrative 里
    显式组合, 且两条线各自都要先过自己的自否证。
    """
    def _brief(tag: str, s: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not s:
            return {"tag": tag, "ok": False, "error": "该侧无取向数据"}
        return {"tag": tag, "ok": True,
                "oriented_frac": s.get("oriented_frac"),
                "n_pose": s.get("n_pose"),
                "n_scissile_sites": s.get("n_scissile_sites")}

    out: Dict[str, Any] = {
        "ok": True, "orientation_applicable": True,
        "noise_floor": ORIENT_NOISE_FRAC,
        "dist_noise_floor_A": ORIENT_DIST_NOISE_A,
        "banned_phrases": [], "allowed_phrases": [],
        "wt": _brief("WT", wt), "mut": _brief("MUT", mut),
        "oriented_frac_wt": None, "oriented_frac_mut": None,
        "oriented_frac_delta": None,
        # 距离三件套同样在初始化就占位: 失败分支(规则 1~4)也必须带齐, 下游 report
        # 按这些键取名, 缺键会 KeyError —— 与 compare_sides 同款契约。
        "nearest_C_to_nuc_mean_wt_A": None, "nearest_C_to_nuc_mean_mut_A": None,
        "nearest_delta_A": None,
        "narrative": "",
    }
    if not wt or not mut:
        out["ok"] = False
        out["banned_phrases"].append(
            "任一侧缺取向数据 —— 不得写『突变改变/未改变底物催化取向』, 也不得"
            "用取向支撑或推翻任何活性结论; 需重跑双侧 --eval_smiles 补齐")
        out["narrative"] = "取向对比不可用: 单侧或双侧缺失数据"
        return out

    # oriented_frac 有两种形态, 调用方都要吃得下:
    #  (a) 单侧/单种子 = 汇总形态 summarize_side 的 float;
    #  (b) 跨种子聚合 = _summarize_orientation_runs 的 {"mean", "std", "n"} dict。
    # 判词只按数值说事, 这里统一归一成 float(None=缺)。
    def _num(v):
        if isinstance(v, dict):
            v = v.get("mean")
        return v if isinstance(v, (int, float)) else None

    fw = _num(wt.get("oriented_frac"))
    fm = _num(mut.get("oriented_frac"))
    out["oriented_frac_wt"], out["oriented_frac_mut"] = fw, fm
    sw = wt.get("n_scissile_sites") or 0
    sm = mut.get("n_scissile_sites") or 0
    n_pose_w, n_pose_m = wt.get("n_pose") or 0, mut.get("n_pose") or 0
    _fmt_w = f"{fw:.2f}" if fw is not None else "n/a"
    _fmt_m = f"{fm:.2f}" if fm is not None else "n/a"

    # 2. 指标不适用(底物无酯羰基)
    if sw == 0 and sm == 0:
        out["orientation_applicable"] = False
        out["allowed_phrases"].append(
            "两侧配体均无可水解酯键(scissile 位点 0), 催化取向判据不适用 —— "
            "典型情形是**产物端羧酸盐**: pH 7.4 下 −COOH 去质子化成 −COO⁻, 两个 O "
            "到 C 距离同在羰基窗内, 没有可判别的离去氧, 于是正确地不给位点")
        out["allowed_phrases"].append(
            "可写: 该底物不形成酰基酶, oriented 占比本就应为 0, 不得读作"
            "『突变让取向变差』")
        out["narrative"] = ("本底物不提供 scissile 位点(无可水解酯键 / 羧酸盐在 "
                            "pH 7.4 下两氧等价), 取向判据不适用; 产物释放是否限速"
                            "应换产物对照对接来问, 不是本判据能回答的")
        for b in ("不得写『突变改变了底物催化取向』(本底物无酯羰基, 判据不成立)",
                  "不得以取向占比为由解释 kcat 变化"):
            out["banned_phrases"].append(b)
        return out

    # 3. 单侧"配体根本没进活性位" → 比"取向不好"更硬的红线
    # 先于规则 4/5 判定: 0/9 且最近距离 >> 接触球, 说明该侧对接找的是别处的
    # 最优点(盒心错位/跨坐标系), 此时任何比值都没有意义。
    _offside: List[str] = []
    for _tag, _s in (("WT", wt), ("MUT", mut)):
        _near = _s.get("nearest_mean_A")
        if isinstance(_near, dict):
            _near = _near.get("mean")
        if not isinstance(_near, (int, float)):
            continue
        _frac_here = _num((_s.get("oriented_frac")))
        if _frac_here == 0 and _near > ORIENT_OFFSITE_A:
            _offside.append(f"{_tag}(最近 scissile C→亲核 均值 {_near:.1f} Å)")
    if _offside:
        out["offsite_detected"] = True
        out["banned_phrases"].append(
            f"禁写一切取向/活性结论 —— {', '.join(_offside)} 远超 "
            f"{ORIENT_OFFSITE_A} Å: 该侧对接**没有把配体摆进催化中心**, 不是"
            "『取向变差』, 而是盒心错位/跨坐标系(拿一个结构的坐标去用另一个结构)")
        out["banned_phrases"].append(
            "禁写『突变改变取向』甚至『突变损害活性』—— 差值来自受体坐标系/盒子, "
            "与突变位点无关")
        out["allowed_phrases"].append(
            f"可写事实: 该侧 {ORIENT_OFFSITE_A} Å 外接球内无任何 scissile 羰基碳, "
            "配体落在别处")
        out["allowed_phrases"].append(
            "修法(必须先做): 对 alt 受体做 **Kabsch 变换** `geom.kabsch_transform` + "
            "`apply_rigid` 对齐到 ref 坐标系, 或两侧各自算自己的三联体盒心并分别"
            "独立报告, 不要把两侧的对 energetics 直接相减")
        out["narrative"] = (f"对接未复现催化复合物(配体落在催化中心外 "
                            f"{', '.join(_offside)}), 本轮能量/取向均不可用于该突变"
                            "的归因 —— 需先解决受体坐标系与盒子对齐")
        return out

    # 4. 双侧都没检出取向 → 最硬红线
    if fw == 0 and fm == 0:
        out["banned_phrases"].append(
            "两侧 oriented 均为 0/None —— 本轮对接**没有复现催化复合物**, "
            "任何『活性提升/下降』『结合更紧/更松』『突变改变催化取向』结论一律禁止")
        out["banned_phrases"].append(
            "禁止把 ΔE 能量差(无论正负)归因于 kcat; 该能量来自未摆到催化位的姿态")
        out["allowed_phrases"].append(
            f"可写事实: 双侧 {n_pose_w}/{n_pose_m} 个 pose 均未把酯羰基碳摆进"
            f"催化亲核原子(Ser OG)的 {NUC_TO_SCISSILE_C_MAX} Å 接触球 —— "
            "说明本次盒子/采样未找到催化姿态, 需放大盒子或提高 exhaustiveness 重跑")
        out["narrative"] = ("双侧均未检出可攻击姿态, 本轮能量数据不可用于机理归因"
                            "(对接未复现催化复合物)")
        return out

    # 4.5 摆位距离量也要有地板 —— 与占比差是两个独立量, 必须各自判
    #    占比差超地板(规则 5/6/7 放行)并不代表距离差也可读; 反过来占比差在地板
    #    内时, 距离差若恰好超地板也不构成额外证据。这里只做"禁写距离方向"的红线。
    _nw = _num(wt.get("nearest_mean_A"))
    _nm = _num(mut.get("nearest_mean_A"))
    out["nearest_C_to_nuc_mean_wt_A"] = _nw
    out["nearest_C_to_nuc_mean_mut_A"] = _nm
    if _nw is not None and _nm is not None:
        _dnear = round(_nm - _nw, 3)
        out["nearest_delta_A"] = _dnear
        if not dist_delta_verdict(_nw, _nm, floor=ORIENT_DIST_NOISE_A)["passes"]:
            out["banned_phrases"].append(
                f"禁写『突变让底物更靠近/远离催化亲核』: |Δ距离|={abs(_dnear):.2f} Å "
                f"< 距离地板 {ORIENT_DIST_NOISE_A} Å(两侧均值 {_nw:.2f} → {_nm:.2f} Å)"
                " —— 摆位距离差落在对接抖动内不可分辨; 本节唯一可做方向判断的量是 "
                "oriented 占比(可/不可攻击几何), 距离只作辅助描述")
    else:
        out["nearest_delta_A"] = None

    # 5. 差值 vs 噪声地板
    if fw is None or fm is None:
        out["ok"] = False
        out["banned_phrases"].append("存在一侧 oriented 占比为 None, 不得写方向性结论")
        out["narrative"] = "取向对比不可用: 存在侧无有效占比"
        return out
    delta = round(fm - fw, 3)
    out["oriented_frac_delta"] = delta
    _msg = (f"oriented 占比 WT {_fmt_w} → 突变体 {_fmt_m}"
            f"(Δ={delta:+.3f}, 地板 {ORIENT_NOISE_FRAC})")
    if not frac_delta_verdict(fw, fm, floor=ORIENT_NOISE_FRAC)["passes"]:
        out["banned_phrases"].append(
            f"禁写『突变改善/恶化了催化取向』: |Δfrac|={abs(delta):.3f} < "
            f"噪声地板 {ORIENT_NOISE_FRAC}(n={n_pose_w}/{n_pose_m} pose, "
            f"二项标准误 ~0.16) —— 两侧占比差与抽样抖动不可分辨")
        out["banned_phrases"].append(
            "禁写『底物摆位更准/催化准备度更高』类表述, 也不得用取向支撑能量差")
        out["allowed_phrases"].append(
            f"可写: 两侧可攻击姿态占比无实质差异({_msg})")
        out["allowed_phrases"].append(
            "可写: 若能量差本身在噪声内(±0.15 kcal/mol), 则本轮对接对该突变无"
            "取向层面的区分力")
        out["narrative"] = (f"催化取向占比无实质变化({_msg}); 突变体的差异"
                            "不能归因于『底物摆得更好』")
        return out

    # 6. 单侧检出
    if fw == 0 or fm == 0:
        out["banned_phrases"].append(
            "命题『单侧检出』本身可写, 但**必须同时说明是单侧**: 只占 1/9 pose "
            "就能把占比从 0 推到 0.11, 单侧检出高度提示 pose 采样不足而非构象效应")
        if fw == 0 and fm > 0:
            out["allowed_phrases"].append(
                f"可写(带限定): 突变体侧检出可攻击姿态 {int(round(fm * (mut.get('n_pose') or 1)))}"
                f"/{mut.get('n_pose')} pose, 而 WT 侧 0 —— 但应视为采样问题优先复核, "
                f"除非多种子重复稳定复现")
            out["allowed_phrases"].append(
                "可写: 该方向**不构成**『突变提升活性』的成立证据(单侧 + 小样本)")
        else:
            out["allowed_phrases"].append(
                f"可写(带限定): WT 侧检出可攻击姿态, 突变体侧 0 — 同样不构成"
                f"『突变损害活性』的成立证据, 需多种子复核")
        out["narrative"] = f"催化取向呈单侧检出({_msg}) —— 采样不足, 不构成方向性证据"
        return out

    # 7. 双侧都检出且超地板
    better = "突变体" if delta > 0 else "WT"
    up = delta > 0
    out["allowed_phrases"].append(
        f"可写: 可攻击姿态占比 WT {_fmt_w} → 突变体 {_fmt_m}"
        f"(Δ={delta:+.3f} > 地板 {ORIENT_NOISE_FRAC}), 即 {better}"
        f"{'更多' if up else '更少'}的对接姿态把酯羰基碳摆进亲核接触球")
    out["allowed_phrases"].append(
        "可写时必须同句带上占比分母与地板数字, 不得只写形容词"
        "『摆位更准/催化准备度更高』")
    out["banned_phrases"].append(
        "禁止把占比差直接读成 kcat 倍数或活性比例 —— 取向占比是几何可用性,"
        "不含速率常数, 也不构成 KM 解释")
    out["banned_phrases"].append(
        "禁止在占比差之外, 用本节的取向结论去解释已经写进报告的 ΔE")
    # 单轮边界声明 —— 规则 5/6/7 判的都是"这一轮采样", 不是"这个突变体的性质"。
    # 实证(2026-10-05, 某丝氨酸水解酶 WT vs 突变体, 同源坐标系, seeds 固定 42/43,
    # exh=8):
    #   run#1 oriented 占比 WT 0.222 → mut 0.333, Δ=+0.111(未过地板)
    #   run#2 oriented 占比 WT 0.111 → mut 0.333, Δ=+0.222(过地板)
    #   seed 43 的 WT 侧单独看 3/9 → 1/9, 同一个 seed 两次跑对不上。
    # 即"固定 seed ≠ 逐位可复现"(配体 3D 构象/受体 pdbqt 每次重新生成, OB 侧有非确定
    # 性)。因此任何超地板的占比差都只是**单次采样的观测**, 不构成可写进论文的
    # 方向性结论, 除非按下面的协议独立重复过。
    out["banned_phrases"].append(
        "本条只覆盖**单轮采样**(含本轮多个种子): 单轮放行的占比差不等于该突变体的"
        "性质 —— 必须以同一协议独立重复 ≥2 次、且每次 Δfrac 符号与量级方向一致"
        "(Δ 需同号且多数次同过/同不过地板)后才能进叙事; 方向翻转或幅度跨地板即判"
        "『本轮对接无区分力』")
    out["narrative"] = (f"催化取向占比 突变体 vs WT: {_fmt_m} vs {_fmt_w}"
                        f"(Δ={delta:+.3f}, 超地板 {ORIENT_NOISE_FRAC}) —— "
                        f"可表述为『{better}侧的对接姿态更常处于可攻击几何』, "
                        f"但不等价于 kcat 变化")
    return out


__all__ = [
    "COVALENT_BOND_MAX", "CARBONYL_O_MIN", "CARBONYL_O_MAX", "ESTER_O_MAX",
    "NUC_TO_SCISSILE_C_MAX", "HIS_TO_LEAVING_O_MAX",
    "ORIENT_NOISE_FRAC", "ORIENT_OFFSITE_A", "ORIENT_DIST_NOISE_A",
    "ligand_bond_graph", "find_scissile_sites", "receptor_anchors",
    "pose_orientation", "pose_orientation_full",
    "summarize_side", "orientation_sides",
]
