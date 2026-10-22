# -*- coding: utf-8 -*-
"""pocketgeom.py —— 口袋几何度量: 网格自由体积 / 残基对通道宽 / 残基邻域图。

起源(2026-10-03): 突变体机理分析中的一次性脚本转正为 libdock 模块 —— 突变体筛选
需要"这个突变把口袋改成了什么"的**阳性指标**(Δpocket 体积/Δ通道宽), 与既有 QC
的阴性指标(Cα RMSD / clash / 三联体几何, 即"结构没坏")互补。

口径固定(跨版本 QC 可比; 任何缺省改动的历史 Δpocket 即失效, 必须换口径版本号):
  - 格点 step=0.8 A, 探针 probe=1.4 A(水分子半径)
  - 口袋核心球: **WT 侧**三联体 Cα 质心(detect_catalytic_triad 的 ca_centroid),
    半径 radius=8.0 A; WT 与突变体用**同一参照坐标**(可比性关键)
  - 自由体积 = 球内到最近重原子表面 > probe 的格点体积和 —— 近似"分子表面内
    空腔", 不做闭合性判定(口袋体积绝对值依赖闭合算法; 对比用途要的是同一把尺子)
  - 溶剂/离子残基(_SOLVENT_RESNAMES)不计入(晶体水的有无不反映口袋几何)
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._core import LOG
from .ligand import _SOLVENT_RESNAMES
from .triad import detect_catalytic_triad

_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

# 口径常量(见模块 docstring; 消费方应原样转写进产物元数据)
POCKET_STEP_A = 0.8
POCKET_PROBE_A = 1.4
POCKET_RADIUS_A = 8.0


def _parse_pdb_heavy(path: str) -> List[Tuple]:
    """重原子列表 [(chain, resnum, atom, resname, x, y, z, vdw)]。

    - 跳过 H/D; 溶剂/离子残基(_SOLVENT_RESNAMES)仅限 HETATM 记录
      (蛋白 ATOM GLY 不是溶剂, 见 _parse_pdb_heavy 内注释)
    - altLoc 取首构象(' ' / 'A'); PDBFixer 突变产物为 ATOM-only 纯蛋白
    """
    out: List[Tuple] = []
    try:
        f = open(path, encoding="utf-8", errors="ignore")
    except OSError as e:
        LOG.warning("pocketgeom: 读取 %s 失败(%s)", path, e)
        return out
    with f:
        for ln in f:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            if ln[16] not in (" ", "A"):
                continue
            name = ln[12:16].strip()
            if not name:
                continue
            resn = ln[17:20].strip()
            # 溶剂/离子只滤 HETATM 记录: 结晶添加剂(EDO/GLY/GOL 等)是 HETATM,
            # 而蛋白自身的甘氨酸是 ATOM GLY(氧阴离子穴 G55/G128 就常在口袋里!)
            # —— 按残基名一刀切会把蛋白 GLY 也滤掉, 口袋体积虚高。
            if ln.startswith("HETATM") and resn in _SOLVENT_RESNAMES:
                continue
            try:
                resnum = int(ln[22:26])
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            el = (ln[76:78].strip() or name[0]).upper()
            if el in ("H", "D"):
                continue
            out.append((ln[21], resnum, name, resn, x, y, z, _VDW.get(el, 1.7)))
    return out


def free_volume_sphere(atoms: List[Tuple], center, radius: float = POCKET_RADIUS_A,
                       probe: float = POCKET_PROBE_A, step: float = POCKET_STEP_A) -> float:
    """球内自由体积(Å³): 到最近重原子 vdW 表面 > probe 的格点体积和。

    atoms 为 _parse_pdb_heavy 输出; center=(x,y,z)。numpy 分块广播, ~1s 内。"""
    if not atoms:
        return 0.0
    import numpy as np
    xyz = np.array([[a[4], a[5], a[6]] for a in atoms], float)
    rad = np.array([a[7] for a in atoms], float)
    n = int(radius / step)
    axes = np.arange(-n, n + 1, dtype=float) * step
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    mask = (gx ** 2 + gy ** 2 + gz ** 2) <= radius * radius
    pts = np.stack([gx[mask], gy[mask], gz[mask]], axis=1) + np.asarray(center, float)
    occ = np.zeros(len(pts), dtype=bool)
    for i in range(0, len(pts), 512):
        blk = pts[i:i + 512]
        d = np.linalg.norm(blk[:, None, :] - xyz[None, :, :], axis=2)
        occ[i:i + 512] = (d <= (rad[None, :] + probe)).any(axis=1)
    return float(int((~occ).sum()) * step ** 3)


def pocket_free_volume(pdb_path: str, center, radius: float = POCKET_RADIUS_A,
                       probe: float = POCKET_PROBE_A, step: float = POCKET_STEP_A) -> float:
    """路径版 free_volume_sphere(单结构查询用)。"""
    return free_volume_sphere(_parse_pdb_heavy(pdb_path), center, radius, probe, step)


POCKET_MIN_BURIED_A = 3.0   # 埋深下限: 游离点至少离原子表面这么远才算"埋在内部"


def detect_pockets(pdb_path: str, chain: str = "", step: float = 1.0,
                   probe: float = POCKET_PROBE_A, padding: float = 3.0,
                   min_volume_A3: float = 100.0,
                   min_buried_A: float = POCKET_MIN_BURIED_A,
                   contact_cutoff_A: float = 5.0) -> List[Dict[str, Any]]:
    """埋深口径的口袋检测: 找蛋白**埋在内部**的空腔, 按体积降序返回。

    用途(2026-10-04 优化项 A7): dock 的 auto 盒回退链此前是
    三联体 → 手动催化 → triad_transfer → protein_centroid —— 前三级都失败时盒心
    直接落在蛋白质几何中心(对结合位点不在几何中心的蛋白基本不可靠)。
    本函数补上"口袋检测层": 用网格找蛋白内部最深的空腔, 以其质心为盒心。
    这比蛋白质心有物理依据(酶的结合口袋就是这种深埋空腔), 但仍需人工复核。

    算法(不依赖 scipy, 纯 numpy):
      1. 重原子 vdW 膨胀 probe(水半径)得到 occupied(不可达)布尔网格
      2. **埋深(depth)口径**: 从 occupied 出发对 free 区域做迭代膨胀, 第 k 轮
         才被吃掉的游离点, 其埋深 ≈ k*step。depth >= min_buried_A 的点 = 埋在
         内部的空腔。
         为什么不用"封闭空腔"口径: 实测这类丝氨酸水解酶在 probe=1.4 下,
         催化空腔通过入口通道与体相连通(水探针进得去), "封闭腔"判据会返回 0 个,
         检测层形同虚设(2026-10-04 实测)。埋深口径不要求完全封闭, 更贴合酶口。
      3. 埋深达标点做连通标记(同样是膨胀法), 逐块算体积与质心
      4. 体积 >= min_volume_A3 的按体积降序返回, 并附接触残基(<= contact_cutoff)

    口径固定(与 free_volume_sphere 同源同尺度参数, 便于交叉核对):
      step / probe 与模块常量一致; vdW 表同 _VDW。

    注意: 空腔体积的**绝对值**依赖网格与埋深阈值, 只做相对比较用(哪个空腔最大/最
    深), 不要当实验测得的口袋体积引用。

    🔴 已知口径缺陷(2026-10-05 实测): 埋深是"离**最近原子**表面多远", 于是蛋白
    外表面上的溶剂也满足 depth >= min_buried, 会被算进"内部空腔" —— 实测
    给出 22828 Å³ 的单一"空腔"(而蛋白本体体积才 ~2×10⁴ Å³), 这个数字实际装的是
    "溶剂+内腔+通道"的全体。对 dock auto 盒心回退足够(只比大小/深浅), 但**绝对
    值不可引用**。通道/连通性/瓶颈半径问答**不在本模块** —— 见 `pocket_tunnel`
    (connected_to_bulk / bottleneck_radius_A / channel_profile), CLI 入口为
    `dockbuddy pocket-tunnel`(那一处从"体相"起算, 口径正确)。
    """
    atoms = _parse_pdb_heavy(pdb_path)
    if not atoms:
        return []
    if chain:
        atoms = [a for a in atoms if (a[0] == chain)]
    if not atoms:
        return []
    xyz = np.array([[a[4], a[5], a[6]] for a in atoms], dtype=float)
    vdw = np.array([a[7] for a in atoms], dtype=float)

    mn = xyz.min(axis=0) - padding
    mx = xyz.max(axis=0) + padding
    n = [max(2, int(math.ceil((mx[i] - mn[i]) / step)) + 1) for i in range(3)]
    # padded: 四周多一圈, 边框恒为 occupied(体相), 保证 BFS 从"最外层游离点"起跑
    P = (n[0] + 2, n[1] + 2, n[2] + 2)
    a, b, c = P
    npts = a * b * c

    # 全部网格运算在**一维扁平索引**上做: flat = (i*b + j)*c + k
    # occ=True = **被原子占据**(到原子中心 <= vdW+probe, 水分子也进不来)。
    # 判据必须是 d2 <= rr*rr; 写成 d2 > rr*rr 会把"远离原子"当成占据, 于是
    # 游离区反而是蛋白最密实的内核, 检测出的"空腔"是实心块 —— 2026-10-04 实测踩坑。
    occ = np.zeros(npts, dtype=bool)
    # 逐原子标记: 只扫该原子周边的局部网格, 避免全网格 × 全原子的 O(n²) 距离计算
    reach = vdw + probe
    for x, y, z, rr in zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], reach):
        # 2026-10-04 全盘审计(P1-L2): 世界→网格换算 i=(x-mn)/step+1(见 169 行),
        # 上界 int(...) 缺 +1 → 每个原子占据盒在 +x/+y/+z 侧各漏标最外一格
        # (边界上真实在 rr 内的格保持 free → 口袋体积/边界 ≤1 step 系统不对称)。
        # 下界多扫一格无害(d2<=rr² 精确复核兜底), 只补上界。
        # ⚠ 口径变更: 本修复后 occupied 略增, 口袋体积数字与此前产物不可直接比
        # (全系列需重算, 见 data/reports/审计报告_2026-10-04_全盘审计.md §六)。
        i0 = max(1, int((x - rr - mn[0]) / step))
        i1 = min(a - 2, int((x + rr - mn[0]) / step) + 1)
        j0 = max(1, int((y - rr - mn[1]) / step))
        j1 = min(b - 2, int((y + rr - mn[1]) / step) + 1)
        k0 = max(1, int((z - rr - mn[2]) / step))
        k1 = min(c - 2, int((z + rr - mn[2]) / step) + 1)
        if i1 < i0 or j1 < j0 or k1 < k0:
            continue
        #  padded 网格 index i ↔ 世界坐标 mn[0] + (i-1)*step
        ii, jj, kk = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1),
                                 np.arange(k0, k1 + 1), indexing="ij")
        d2 = ((mn[0] + (ii - 1) * step - x) ** 2
              + (mn[1] + (jj - 1) * step - y) ** 2
              + (mn[2] + (kk - 1) * step - z) ** 2)
        occ[(ii * b + jj) * c + kk] |= (d2 <= rr * rr)

    free = ~occ
    if not free.any():
        return []

    # --- 埋深: 从网格四壁的游离点(体相)起逐层膨胀, 埋深=到体相的图距(网格步) ---
    depth = _buried_depth(free, P, max_steps=max(1, int(min_buried_A / step) + 1))
    buried = free & (depth >= max(1, int(round(min_buried_A / step))))
    if not buried.any():
        return []

    comp = _label_components(buried, P)
    out: List[Dict[str, Any]] = []
    for cid in sorted({int(v) for v in np.unique(comp[comp > 0])}):
        idx = np.flatnonzero(comp == cid)
        vol = float(len(idx) * step ** 3)
        if vol < min_volume_A3:
            continue
        pts = np.stack([mn[0] + ((idx // (b * c)) - 1) * step,
                        mn[1] + (((idx // c) % b) - 1) * step,
                        mn[2] + ((idx % c) - 1) * step], axis=1)
        center = pts.mean(axis=0)
        residues = _residues_near(pts, atoms, contact_cutoff_A)
        # 该连通块内最深点的埋深(A): 区分"深隧道末端的口袋"与"大片浅网络"
        deep = depth[idx]
        out.append({"center": [round(float(v), 2) for v in center],
                    "volume_A3": round(vol, 1),
                    "max_depth_A": round(float(deep.max()) * step, 1),
                    "n_grid": int(len(idx)),
                    "n_residues_within": len(residues),
                    "residues": residues[:40]})
    out.sort(key=lambda d: -d["volume_A3"])
    return out[:5]


def _buried_depth(free_flat: np.ndarray, P, max_steps: int = 64) -> np.ndarray:
    """游离点**埋深**(单位: 网格步) = 到最近"体相"游离点的 6 邻域图距。

    起点 = 网格四壁(i==0 / i==a-1 / ...)上的游离点 —— 那里在包围盒+padding 之
    外, 恒为体相。从这些点起逐层 6 邻域膨胀, 第 l 轮才吃到的游离点埋深记为 l。
    埋深小 = 贴着蛋白外表面/通道口; 大 = 深埋在蛋白内部。故"埋深达标"≈这条
    通路绕开了体相这么远, 而不是"整条通路被固体封死"(见 docstring 末尾)。

    不采用"封闭空腔 / 从体相洪水填充到不了"口径的原因(2026-10-04 实测):
    实测真实酶在 probe=1.4 下, 催化空腔经入口通道与体相连通,
    水探针进得去 —— 闭合判据返回 0 个空腔, 检测层形同虚设。埋深口径不要求
    完全封闭(入口再窄也是通的), 更贴合真实酶口袋。

    实现要点: 整体膨胀迭代而非逐点 BFS —— 10 万级网格的 Python 逐点 BFS 要跑
    上万轮 np.unique, 实测 >2min 挂死; 膨胀法每轮 6 次切片 OR, ~40 轮, 毫秒级。

    返回与 free_flat 等长的 int32 数组(未到达处 -1; 游离区恒全连通, 实际不出现)。
    """
    a, b, c = P
    npts = a * b * c
    depth = np.full(npts, -1, dtype=np.int32)
    free3 = free_flat.reshape(P)
    idx = np.arange(npts, dtype=np.int64)
    i, j, k = idx // (b * c), (idx // c) % b, idx % c
    border = ((i == 0) | (i == a - 1) | (j == 0) | (j == b - 1)
              | (k == 0) | (k == c - 1)).reshape(P)
    frontier = free3 & border
    if not frontier.any():
        return depth
    depth[frontier.reshape(-1)] = 0
    ts = [np.zeros(P, bool) for _ in range(6)]
    lvl = 0
    while frontier.any() and lvl < max(1, max_steps):
        fi = frontier
        for t in ts:
            t[...] = False
        ts[0][1:, :, :] = fi[:-1, :, :]
        ts[1][:-1, :, :] = fi[1:, :, :]
        ts[2][:, 1:, :] = fi[:, :-1, :]
        ts[3][:, :-1, :] = fi[:, 1:, :]
        ts[4][:, :, 1:] = fi[:, :, :-1]
        ts[5][:, :, :-1] = fi[:, :, 1:]
        nxt = ts[0].copy()
        for t in ts[1:]:
            nxt |= t
        # 只在 free 内蔓延, 且尚未赋值: occ 永远不进 frontier, 隧道不能穿固体
        nxt &= free3 & (depth.reshape(P) < 0)
        if not nxt.any():
            break
        lvl += 1
        nxt_flat = nxt.reshape(-1)
        depth[nxt_flat] = lvl
        frontier = nxt
    return depth


def _flood_border(seed_flat: np.ndarray, P, free_flat: np.ndarray) -> np.ndarray:
    """保留兼容: 从 seed_flat 起做 6 邻域膨胀标出体相区域(现由 _buried_depth 取代)。

    free_flat: 全网格游离点掩码 —— 膨胀必须**与它求交**, 否则一步就跨进原子
    内部, 体相会吞掉整个网格(2026-10-04 实测 outside == npts)。
    """
    vol = np.zeros(P, dtype=bool)
    if len(seed_flat):
        vol.reshape(-1)[np.unique(seed_flat)] = True
    ts = [np.zeros(P, bool) for _ in range(6)]
    while True:
        fi = vol
        for t in ts:
            t[...] = False
        ts[0][1:, :, :] = fi[:-1, :, :]
        ts[1][:-1, :, :] = fi[1:, :, :]
        ts[2][:, 1:, :] = fi[:, :-1, :]
        ts[3][:, :-1, :] = fi[:, 1:, :]
        ts[4][:, :, 1:] = fi[:, :, :-1]
        ts[5][:, :, :-1] = fi[:, :, 1:]
        n = vol.copy()
        fmask = free_flat.reshape(P)
        for t in ts:
            n |= t & fmask
        if n.sum() == vol.sum():
            break
        vol = n
    return vol.reshape(-1)


def _grow(seed_flat: np.ndarray, P, mask_flat: np.ndarray) -> np.ndarray:
    """从 seed 出发在 mask_flat 内做 6 邻域连通生长(同一膨胀法, 只在 mask 内蔓延)。

    返回 bool 掩码; 逐块生长由 _label_components 调用。
    """
    vol = np.zeros(P, dtype=bool)
    if len(seed_flat):
        vol.reshape(-1)[np.unique(seed_flat)] = True
    ts = [np.zeros(P, bool) for _ in range(6)]
    while True:
        fi = vol.reshape(P)
        for t in ts:
            t[...] = False
        ts[0][1:, :, :] = fi[:-1, :, :]
        ts[1][:-1, :, :] = fi[1:, :, :]
        ts[2][:, 1:, :] = fi[:, :-1, :]
        ts[3][:, :-1, :] = fi[:, 1:, :]
        ts[4][:, :, 1:] = fi[:, :, :-1]
        ts[5][:, :, :-1] = fi[:, :, 1:]
        n = vol.copy()
        m = mask_flat.reshape(P)
        for t in ts:
            n |= t & m
        if n.sum() == vol.sum():
            break
        vol = n
    return vol.reshape(-1)


def _label_components(mask_flat: np.ndarray, P) -> np.ndarray:
    """扁平 bool 掩码的 6 邻域连通标记(纯 numpy 膨胀法, 不依赖 scipy)。

    返回与 mask_flat 等长的 int 标签(0=未命中掩码)。
    """
    lab = np.zeros(len(mask_flat), dtype=int)
    cur = 0
    remaining = mask_flat.copy()
    for _ in range(64):            # 空腔块数上限, 兜底防失控
        start = np.flatnonzero(remaining)
        if len(start) == 0:
            break
        # 取剩余未标记里最"外围"的一点当种子: 用最左端索引即可, 膨胀会蔓延全块
        seed = np.array([start[0]], dtype=int)
        grown = _grow(seed, P, remaining & (lab == 0))
        if not grown.any():
            break
        cur += 1
        lab[grown] = cur
        remaining &= ~grown
    return lab


def _residues_near(cavity_pts: np.ndarray, atoms, cutoff: float) -> List[str]:
    """空腔点上 <= cutoff 的残基标签(去重, 按残基号排序)。"""
    xyz = np.array([[a[4], a[5], a[6]] for a in atoms], dtype=float)
    hit: Dict[Tuple[str, int], str] = {}
    step = max(1, int(len(cavity_pts) / 200) + 1)
    for s in range(0, len(cavity_pts), step):
        blk = cavity_pts[s:s + step]
        d = np.linalg.norm(blk[:, None, :] - xyz[None, :, :], axis=2)
        near = d <= cutoff
        rows, cols = np.nonzero(near)
        for r, cc in zip(rows, cols):
            a = atoms[cc]
            hit.setdefault((a[0], a[1]), f"{a[3]}{a[1]}({a[0]})")
    # 2026-10-04 全盘审计: label 本身已含残基号(f"{resname}{resnum}({chain})"),
    # 原返回再拼 rn → "ASP204(A)204" 双重残基号。
    return [n for _, n in sorted(hit.items())]


def residue_min_distance(atoms: List[Tuple], key_a: Tuple[str, int],
                         key_b: Tuple[str, int]) -> Optional[float]:
    """两残基(各取全部重原子)间最小距离 —— 通道宽口径(如两个芳香环之间)。

    key=(chain, resnum); 任一侧无原子返回 None。"""
    A = [a for a in atoms if (a[0], a[1]) == key_a]
    B = [a for a in atoms if (a[0], a[1]) == key_b]
    if not A or not B:
        return None
    return min(math.dist(a[4:7], b[4:7]) for a in A for b in B)


def residue_environment(atoms: List[Tuple], key: Tuple[str, int],
                        cutoff: float = 5.5) -> Dict[str, int]:
    """指定残基的邻域接触图 {resname resnum(chain): 接触原子数}(排除自身)。

    任一原子对距离 <= cutoff 即计 1 个接触原子(同一残基只计一次)。"""
    env: Dict[str, int] = {}
    tgt = [a for a in atoms if (a[0], a[1]) == key]
    if not tgt:
        return env
    for a in atoms:
        if (a[0], a[1]) == key:
            continue
        hit = False
        for t in tgt:
            if math.dist(a[4:7], t[4:7]) <= cutoff:
                hit = True
                break
        if hit:
            label = f"{a[3]}{a[1]}({a[0]})"
            env[label] = env.get(label, 0) + 1
    return env


def pocket_volume_qc(wt_pdb: str, mut_pdb: str, chain: str = "",
                     radius: float = POCKET_RADIUS_A) -> Dict[str, Any]:
    """mutate QC 入口: WT vs 突变体口袋核心自由体积对比。

    - 参照坐标 = WT 三联体 Cα 质心(两侧共用; 突变只做局部松弛, 质心漂移 ~0.1 Å 级,
      且两侧同坐标才能把"Δ=口袋变化"与"坐标漂移"解耦)
    - WT 检不出三联体 → skipped(不猜测口袋心, 宁缺勿错)
    - 注意: Δ>0 通常=口袋打开(如侧链缩小), Δ<0=收紧; 阈值无普适标准, 供人工判读

    🔴 输入必须是已收敛的 PDB(2026-10-05): 本函数自己**不做格式转换**, 而
    detect_catalytic_triad 底层是 OpenBabel 的 PDB 固定列解析器 —— 喂 mmCIF 时它
    先在 stderr 抛 "not a valid PDB file"、返回 0 残基, 于是这里输出"WT 未检出
    催化三联体", 把**格式问题伪装成化学判断**。实测后果: 同一条 pocket-tunnel
    命令里主路径(走过 resolve_receptor 收敛)识别出 SER101A:OG 并算出 142.8 Å³,
    而 QC 交叉核对却同时报"WT 未检出催化三联体, 口袋心不可定位", 一份 JSON 自相
    矛盾(还让人误以为通道数字也不可信)。调用方务必先走
    fetch.ensure_pdb_input() / libdock.pocket_tunnel.resolve_receptor() 收敛路径;
    这里只负责把格式问题说成人话并给出可直接执行的修法。
    """
    # 先自己按固定列解析: 这一步与 detect_catalytic_triad 的成败无关, 用来把
    # "文件/格式问题"与"真的没有催化三联体"分开 —— 两类原因不能共用一句判词。
    wa = _parse_pdb_heavy(wt_pdb)
    ma = _parse_pdb_heavy(mut_pdb)
    for _tag, _path, _ats in (("WT", wt_pdb, wa), ("突变体", mut_pdb, ma)):
        if _ats:
            continue
        _cif = str(_path).lower().endswith(".cif")
        return {
            "skipped": f"{_tag}重原子解析为空(不是有效 PDB 固定列)"
                       + ("; 路径以 .cif 结尾 —— 极可能就是该原因" if _cif else ""),
            "hint": ("请先把受体路径收敛成 PDB 再调用: "
                     "fetch.ensure_pdb_input(path) / "
                     "libdock.pocket_tunnel.resolve_receptor(path)"
                     "(pocket-tunnel 命令的主路径已内置收敛; 本函数的调用方必须自己收敛)"),
        }
    triad = detect_catalytic_triad(wt_pdb, chain_filter=chain)
    if not triad:
        return {"skipped": "WT 未检出催化三联体, 口袋心不可定位",
                "hint": (f"WT 已按有效 PDB 解析出 {len(wa)} 个重原子 —— 排除格式问题, "
                         "即该结构确实不含经典 Ser/Cys-His-Asp(Glu) 三联体几何"
                         "(合成体或 AF 模型缺催化残基时如此), 此时不得猜测口袋心")}
    center = triad["ca_centroid"]
    v_wt = free_volume_sphere(wa, center, radius)
    v_mut = free_volume_sphere(ma, center, radius)
    delta = v_mut - v_wt
    return {
        "center_A": [round(float(c), 2) for c in center],
        "radius_A": radius, "step_A": POCKET_STEP_A, "probe_A": POCKET_PROBE_A,
        "free_volume_wt_A3": round(v_wt),
        "free_volume_mut_A3": round(v_mut),
        "delta_A3": round(delta),
        "delta_pct": round(delta / v_wt * 100, 1) if v_wt > 0 else None,
        "note": ("同一参照坐标(WT 三联体质心)下的分子表面内空腔近似; "
                 "Δ>0=口袋打开, Δ<0=收紧"),
    }


__all__ = [
    "POCKET_PROBE_A",
    "POCKET_RADIUS_A",
    "POCKET_STEP_A",
    "_parse_pdb_heavy",     # 2026-10-04: 补导出(tools/family_stats --radius 依赖;
                            # 与"含下划线内部符号保持可用"契约一致)
    "detect_pockets",
    "free_volume_sphere",
    "pocket_free_volume",
    "pocket_volume_qc",
    "residue_environment",
    "residue_min_distance",
]
