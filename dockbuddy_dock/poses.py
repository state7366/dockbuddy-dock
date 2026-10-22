# -*- coding: utf-8 -*-
"""poses.py —— 对接结果解析与 pose 分析: 接触度、埋藏度、催化距离、芳香作用、自对接 RMSD。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import json, math, os, re
from typing import Any, Dict, List, Optional, Tuple
from ._core import LOG, _TYPE2ELEM, ad_type_of, ob
from .config import DockResult
from .geom import _pose_heavy_rmsd, ref_key_match
from .pdbio import _norm_resname
from dockbuddy_dock.reliability import reliability_thresholds

# 阈值单一事实源(2026-10-03 修复): 旧实现把 energy_abs_max=50 / heavy_max=40 硬编码,
# 绕过 DOCK_REL_* 环境覆盖 —— 用户调阈值后引擎侧判定与 classify_reliability_detail
# 口径分裂, 同一 summary 内自相矛盾。模块级取一次(进程内环境不变, 与 ensemble 同风格)。
_TH = reliability_thresholds()

# 催化亲核原子名(2026-10-03 修复): 亲核→酯氧距离必须用**亲核原子**本身 ——
# 旧实现取残基全部原子(含极化氢/CB/主链)的最近者, 系统性偏短(H···O 可低至
# ~1.8A, 文献口径 OG···O 典型 2.7~3.5A), nuc_offsite(>5A 判偏靶)存在漏报
# (OG 距氧 6A 但 CB 距氧 4.8A 时误判在位)。THR 的 OG 与 SER 同为羟基亲核氧。
_NUC_ATOM_BY_RESNAME = {"SER": "OG", "THR": "OG", "CYS": "SG"}


# ----------------------------------------------------------------------------
# 共晶配体双边对接比较 (mutate 有效性评估 / 对接级 GT 验证共用, 2026-09-30 新增)
# ----------------------------------------------------------------------------
def _pdbqt_ligand_coords(pdbqt_path: str) -> Dict[str, Any]:
    """读已制备配体 pdbqt(单构象), 返回 {原子名: np.array(x,y,z)}(仅重原子)。"""
    import numpy as np
    coords: Dict[str, Any] = {}
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            # 判氢取类型码首字符(0-based 76:77); 用第 2 个字符会把 "HD" 读成 "D" 放氢过关
            if ln.startswith(("ATOM", "HETATM")) and (lambda t: _TYPE2ELEM.get(t, t[:1]).upper())(ad_type_of(ln)) != "H":
                an = ln[12:16].strip()
                # C5(2026-10-02): 重复原子名会静默覆盖 dict —— 显式告警(长链配体
                # 可能出现; 覆盖会让下游"按名匹配 RMSD"对错原子)。
                if an in coords:
                    LOG.warning(f"_pdbqt_ligand_coords: 重复原子名 {an!r}({pdbqt_path}), "
                                f"后者覆盖前者 —— 按名匹配的 RMSD 可能失真")
                coords[an] = np.array(
                    [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], float)
    return coords

def _kabsch_rmsd_by_name(ref: Dict[str, Any], pose_atoms) -> float:
    """ref={原子名: xyz}; pose_atoms=parse_pdbqt_poses 的 atoms[(x,y,z,elem,name,...)]。
    按原子名交集 Kabsch 最优叠合算重原子 RMSD(Å); 交集不完整返回 inf。
    (原子顺序陷阱: 两集合必须按名对齐, 见 self_dock_check。)

    C4 口径: **同源配体**(pose 与晶体出自同一制备 pdbqt, 原子名必然一致)要求
    100% 名字匹配 —— 缺名说明配错对, 宁可 inf 也不算部分 RMSD。
    C8: Kabsch 收敛到 geom.kabsch_rmsd 单一实现。"""
    import numpy as np
    pose = {a[4]: np.array(a[:3], float) for a in pose_atoms}
    names = [n for n in ref if n in pose]
    if not names or len(names) != len(ref):
        return float("inf")
    P = np.array([ref[n] for n in names], float)
    Q = np.array([pose[n] for n in names], float)
    from .geom import kabsch_rmsd
    return kabsch_rmsd(P, Q)

# ============================================================================
# 8. 解析与验证分析 (通用)
# ============================================================================

def parse_pdbqt_poses(pdbqt_path: str):
    """解析 Vina 输出: [(energy, [(x,y,z,element,name,resName,resNum), ...]), ...] 按能量升序。"""
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    poses = []
    cur_energy = None
    cur_atoms = []
    in_model = False
    for ln in lines:
        if ln.startswith("REMARK VINA RESULT"):
            m = re.search(r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
            if m:
                cur_energy = float(m.group(1))
        elif ln.startswith("MODEL"):
            in_model = True; cur_atoms = []
            # 跨 MODEL 重置能量(2026-10-03 修复): 缺 REMARK VINA RESULT 的 model
            # 静默继承上一 model 的能量(错误归因), 应回到 None 哨兵
            cur_energy = None
        elif ln.startswith("ENDMDL"):
            if cur_atoms:
                poses.append((cur_energy, cur_atoms))
            in_model = False
        elif (ln.startswith("ATOM") or ln.startswith("HETATM")) and in_model:
            name = ln[12:16].strip()
            resn = ln[17:20].strip()
            resi = ln[22:26].strip()
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            # autodock 类型列(0-based 76:79): 芳香碳为 'A', 脂肪碳为 'C', 氧为
            #   'OA'/'O'... 必须映射到真实元素, 否则芳香碳 elem='A' 会被后续
            #   'elem=="C"' 过滤排除(此前 docked_arom 恒为空、arom 距离恒 null 的 bug)。
            typ = ln[76:79].strip().upper()
            elem = _TYPE2ELEM.get(typ, ln[76:78].strip() or name[0])
            cur_atoms.append((x, y, z, elem, name, resn, resi))
    poses.sort(key=lambda p: (p[0] if p[0] is not None else 1e9))
    return poses

def load_receptor_coords(pdbqt_path: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """读受体 pdbqt, 返回 {resKey: [(x,y,z,atomName), ...]}。resKey 形如 'SER170A'。
    残基名经 _norm_resname 归一化(HID/HIE/HIP->HIS, CYH/CYM->CYS), 以兼容其它
    OpenBabel 构建/prepare_receptor4 的质子化命名, 保证催化键匹配稳健。
    (2026-10-03: 值追加原子名第 4 元, 供亲核距离按亲核原子 OG/SG 过滤;
    全部既有消费点均以 [:3] 切片取坐标, 新增字段向后兼容。"""
    conv = ob.OBConversion(); conv.SetInFormat("pdbqt")
    mol = ob.OBMol(); conv.ReadFile(mol, pdbqt_path)
    data: Dict[str, List[Tuple[float, float, float]]] = {}
    for a in ob.OBMolAtomIter(mol):
        res = a.GetResidue()
        if not res:
            continue
        key = f"{_norm_resname(res.GetName())}{res.GetNum()}{res.GetChain()}"
        try:
            aname = res.GetAtomID(a).strip()
        except Exception:
            aname = ""
        data.setdefault(key, []).append((a.GetX(), a.GetY(), a.GetZ(), aname))
    return data

def load_receptor_aromatic_carbons(pdbqt_path: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """读受体 pdbqt, 返回芳香碳坐标按残基分组 {resKey: [pts]}。
    用于通用'配体芳香环 -> 受体芳香残基(Trp/Phe/Tyr)'距离, 免硬编码参考残基。"""
    conv = ob.OBConversion(); conv.SetInFormat("pdbqt")
    mol = ob.OBMol(); conv.ReadFile(mol, pdbqt_path)
    data: Dict[str, List[Tuple[float, float, float]]] = {}
    for a in ob.OBMolAtomIter(mol):
        if not (a.IsAromatic() and a.GetAtomicNum() == 6):
            continue
        res = a.GetResidue()
        if not res:
            continue
        key = f"{_norm_resname(res.GetName())}{res.GetNum()}{res.GetChain()}"
        data.setdefault(key, []).append((a.GetX(), a.GetY(), a.GetZ()))
    return data

def _receptor_heavy(receptor_pdbqt) -> List[Tuple[float, float, float]]:
    """读受体 PDBQT 重原子坐标, 供 burial / 接触度复用, 避免重复读文件。

    M6(2026-10-02): 滤氢按 **AutoDock 类型码**(H/HD/HS) 而非原子名首字符 ——
    按名会把 HG(汞)误杀(探针实证: 3 原子输入返回 1), 且漏掉前导数字氢名
    ("1HB"式, 旧 PDB 命名)。与 poses/selfdock 其它判氢处同口径。"""
    rec_heavy = []
    with open(receptor_pdbqt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            t = ad_type_of(line)
            if _TYPE2ELEM.get(t, t[:1] if t else "X").upper() == "H":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            rec_heavy.append((x, y, z))
    return rec_heavy

def _burial_metrics(rec_heavy, lig_pts):
    """量化最优 pose 配体是否被埋入蛋白内部(健壮性指标)。
    仅对受体**重原子**计算(排除氢), 否则配体-受体氢的 vdW 接触(~2.3A)会被误判为 clash。
    推理: 正确口袋姿势的配体重原子与蛋白重原子主要是 vdW 接触(~3.2-3.8A), 极少 <2.8A;
          埋入核心的配体原子会大量与蛋白重原子重叠(<2.8A, 甚至<2.5A 被包夹)。
    rec_heavy: 受体重原子坐标列表(由 _receptor_heavy 提供, 已排除氢)。
    返回 (n_clashes, buried_frac, med_near, buried)。"""
    if not lig_pts or not rec_heavy:
        return 0, 0.0, None, False
    nears = []
    for lp in lig_pts:
        best = 1e9
        for rp in rec_heavy:
            d = math.dist(lp, rp)
            if d < best:
                best = d
        nears.append(best)
    clashes = sum(1 for d in nears if d < 2.8)
    buried_frac = sum(1 for d in nears if d < 2.5) / len(nears)
    med = sorted(nears)[len(nears) // 2]
    buried = clashes >= 3 and buried_frac >= 0.15
    return clashes, round(buried_frac, 3), round(med, 2), buried

# ============================================================================
# 8.5 多构象分析与排序(2026-08-21 新增)
# ============================================================================

def detect_aromatic_carbons(pdbqt_path: str) -> List[str]:
    """从已有配体 pdbqt 重新检测芳香碳原子名列表。
    供后处理复用(旧 summary 无 ligand_arom_names 字段时据此恢复)。"""
    if not (pdbqt_path and os.path.exists(pdbqt_path)):
        return []
    names = []
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                names.append(ln[12:16].strip())
    arom: List[str] = []
    try:
        c2 = ob.OBConversion(); c2.SetInFormat("pdbqt")
        m2 = ob.OBMol()
        if c2.ReadFile(m2, pdbqt_path) and m2.NumAtoms() > 0:
            m2.PerceiveBondOrders()
            for i, a in enumerate(ob.OBMolAtomIter(m2)):
                if i < len(names) and a.IsAromatic() and a.GetAtomicNum() == 6:
                    arom.append(names[i])
    except Exception:
        pass
    return arom

def _load_crystal_coords(pdbqt_path: str) -> Optional[Dict[str, Any]]:
    """从晶体配体 pdbqt 加载重原子坐标 {atom_name: (x,y,z)}。
    用于 pose-vs-crystal RMSD 计算(原子名匹配, 避免顺序陷阱)。"""
    import numpy as np
    crystal = {}
    if not (pdbqt_path and os.path.exists(pdbqt_path)):
        return None
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            # 2026-10-02 修复: 判氢要取类型码的**首字符**(列 0-based 76:77)。原用
            # ln[77:78](第 2 个字符)会把 OB 写的极化氢 "HD" 读成 "D" → 氢被当成重原子
            # 混进重原子集合(去质子氧 "OA" 读成 "A" 则恰好没错, 属侥幸)。
            if ln.startswith(("ATOM", "HETATM")) and (lambda t: _TYPE2ELEM.get(t, t[:1]).upper())(ad_type_of(ln)) != "H":
                an = ln[12:16].strip()
                if an in crystal:  # C5: 重复原子名显式告警, 不静默覆盖
                    LOG.warning(f"_load_crystal_coords: 重复原子名 {an!r}({pdbqt_path}), "
                                f"后者覆盖前者 —— 按名匹配的 RMSD 可能失真")
                crystal[an] = np.array([float(ln[30:38]), float(ln[38:46]),
                                        float(ln[46:54])], float)
    return crystal if crystal else None

def _pdbqt_heavy_composition(pdbqt_path: str) -> Dict[str, int]:
    """统计 pdbqt 文件重原子元素组成 {elem: count}(排除 H)。
    用于判断对接配体与共晶配体是否同一分子(self-docking vs cross-docking)。"""
    comp: Dict[str, int] = {}
    if not (pdbqt_path and os.path.exists(pdbqt_path)):
        return comp
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                typ = ln[76:79].strip().upper()
                name = ln[12:16].strip()
                elem = _TYPE2ELEM.get(typ, ln[76:78].strip() or (name[:1] if name else ""))
                if elem and elem != "H":
                    comp[elem] = comp.get(elem, 0) + 1
    return comp

def _pose_heavy_composition(atoms) -> Dict[str, int]:
    """从 pose atoms 统计重原子元素组成 {elem: count}(排除 H)。
    atoms: [(x,y,z,elem,name,resName,resNum), ...]。"""
    comp: Dict[str, int] = {}
    for a in atoms:
        elem = a[3] if len(a) > 3 else ""
        if elem and elem != "H":
            comp[elem] = comp.get(elem, 0) + 1
    return comp

# ----------------------------------------------------------------------------
# C4(2026-10-02): 晶体 RMSD 原子名匹配阈值 —— 两套口径, 各有理由, 单一事实源在此。
#   ① 同源配体(pose 与晶体都出自同一制备 pdbqt, 原子名必然一致): 要求 100% 匹配,
#     缺名即配错对, 宁可 inf —— `_kabsch_rmsd_by_name` / selfdock 用此口径。
#   ② 异源配体(晶体侧直接从原始 PDB 提取, deposited 命名与对接侧 OB 命名不保证
#     一致): 允许部分匹配, 但必须 ≥ MIN_ATOMS 且 ≥ MIN_FRAC, 否则判 None ——
#     宽松侧防"两套命名系统不同导致全灭", 下限防 2~3 个原子的假 RMSD。
_CRYSTAL_MATCH_MIN_ATOMS = 3
_CRYSTAL_MATCH_MIN_FRAC = 0.5

def _calc_pose_rmsd_to_crystal(pose_atoms, crystal: Dict[str, Any]) -> Optional[float]:
    """计算单个 pose 与晶体配体的 Kabsch 重原子 RMSD(Å)。
    pose_atoms: [(x,y,z,elem,name,...), ...] —— 使用 name(列12-16) 匹配。
    crystal: {atom_name: np.array([x,y,z]), ...}。
    返回 None 表示无可匹配原子(匹配阈值见 _CRYSTAL_MATCH_*, C4 口径)。"""
    import numpy as np
    P_coords = []
    Q_coords = []
    for a in pose_atoms:
        # a = (x, y, z, elem, name, resName, resNum)
        elem = a[3]
        if elem == "H":
            continue
        an = a[4]  # atom name
        if an in crystal:
            P_coords.append(crystal[an])
            Q_coords.append(np.array([a[0], a[1], a[2]], float))
    if len(P_coords) < _CRYSTAL_MATCH_MIN_ATOMS or len(P_coords) < len(crystal) * _CRYSTAL_MATCH_MIN_FRAC:
        return None
    from .geom import kabsch_rmsd
    return kabsch_rmsd(np.array(P_coords, float), np.array(Q_coords, float))

def _analyze_pose_light(receptor_pdbqt: str, atoms, energy: float,
                        catalytic: List[str], ref_residue: str,
                        ligand_arom_names: List[str],
                        crystal: Optional[Dict[str, Any]] = None,
                        box_center=None) -> Dict[str, Any]:
    """对单个 pose 做轻量分析，返回排序所需的关键指标字典。
    只计算：催化距离、参考距离、接触度、埋藏、与晶体 RMSD。"""
    rec = load_receptor_coords(receptor_pdbqt)
    res_by_key = {k: v for k, v in rec.items()}
    cat_norm = [_norm_resname(c[:3]) + c[3:] for c in catalytic]

    ligand_oxy = [a for a in atoms if a[3] == "O"]
    arom_name_set = set(ligand_arom_names or [])
    docked_arom = [(a[0], a[1], a[2]) for a in atoms if a[4] in arom_name_set]
    ligand_carb = [a for a in atoms if a[3] == "C" or a[4] in arom_name_set]
    ref_targets = docked_arom if docked_arom else [(a[0], a[1], a[2]) for a in ligand_carb]

    # 催化亲核
    nuc_res_key0 = (catalytic[0] if catalytic else "")
    nuc_resname = nuc_res_key0[:3]
    has_nuc = nuc_resname in ("SER", "CYS", "THR")
    nuc_keys = [k for k in rec if any(ref_key_match(k, c) for c in cat_norm)]  # 2026-10-04: 统一 ref_key_match(防 'SER17' 前缀撞号 SER170A~179A)
    # 亲核原子口径(2026-10-03): 只取亲核原子(SER/THR=OG, CYS=SG)到最近酯氧 ——
    # 旧实现取残基全部原子(含氢), 见模块头 _NUC_ATOM_BY_RESNAME 注释
    _na = _NUC_ATOM_BY_RESNAME.get(nuc_resname)
    nuc_pts = [p for k in nuc_keys for p in res_by_key.get(k, [])
               if _na is None or (len(p) > 3 and p[3] == _na)]
    nuc_dist = None
    if has_nuc and nuc_pts and ligand_oxy:
        best = 1e9
        for q in nuc_pts:
            for t in ligand_oxy:
                d = math.dist(q[:3], t[:3])
                if d < best:
                    best = d
        nuc_dist = best if best < 1e8 else None

    # 参考残基
    ref_dist = None
    if ref_residue and ref_targets:
        rk = [k for k in rec if ref_key_match(k, ref_residue)]
        ref_pts = [p for k in rk for p in res_by_key.get(k, [])]
        if ref_pts:
            best = 1e9
            for q in ref_pts:
                for t in ref_targets:
                    d = math.dist(q[:3], t[:3])
                    if d < best:
                        best = d
            ref_dist = best if best < 1e8 else None

    # 口袋/接触度/埋藏
    # 埋藏/clash 只算配体**重原子**(修复 2026-10-02): 配体极性氢与受体重原子最近距
    # 本就在 2.2~2.6 A(氢键), 混进 lig_pts 会虚增 clash 数并误判"埋藏"。
    lig_heavy = [(a[0], a[1], a[2]) for a in atoms if a[3] != "H"]
    rec_heavy = _receptor_heavy(receptor_pdbqt)
    clashes, buried_frac, med_near, buried = _burial_metrics(rec_heavy, lig_heavy)
    contact = 0.0
    if lig_heavy and rec_heavy:
        contact = sum(1 for lp in lig_heavy if any(math.dist(lp, rp) < 4.5 for rp in rec_heavy)) / len(lig_heavy)
    pocket_contact_frac = round(contact, 3)
    energy_reliable = (energy is not None and abs(energy) <= _TH["energy_abs_max"])
    ligand_n_heavy = len(lig_heavy)

    # 晶体 RMSD
    rmsd_crystal = None
    if crystal:
        rmsd_crystal = _calc_pose_rmsd_to_crystal(atoms, crystal)

    return {
        "pose_index": 0,  # 由调用方填充
        "energy_kcal_mol": energy,
        "nuc_to_ligO_A": round(nuc_dist, 2) if nuc_dist is not None else None,
        "ref_to_ligAromC_A": round(ref_dist, 2) if ref_dist is not None else None,
        "pocket_contact_frac": pocket_contact_frac,
        "ligand_buried": buried,
        "ligand_clashes": clashes,
        "ligand_buried_frac": round(buried_frac, 3),
        "ligand_n_heavy": ligand_n_heavy,
        "energy_reliable": energy_reliable,
        "rmsd_to_crystal_A": round(rmsd_crystal, 2) if rmsd_crystal is not None else None,
    }

def _sort_poses(light_results: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    """按策略排序 pose 轻量结果，返回排序后的列表。
    strategy:
      - "rmsd_to_crystal": 优先按 rmsd_to_crystal_A 升序(越小越接近晶体)，无 RMSD 的排最后
      - "energy": 按 energy 升序(Vina 更负=更好)
    """
    if strategy == "rmsd_to_crystal":
        def key_fn(r):
            rmsd = r.get("rmsd_to_crystal_A")
            if rmsd is not None:
                return (0, rmsd)
            e = r.get("energy_kcal_mol")
            return (1, e if e is not None else 1e9)
        return sorted(light_results, key=key_fn)
    else:  # energy (default)
        # 哨兵必须显式判 None: .get(key, 1e9) 在"键存在且值为 None"时返回 None
        # (缺 REMARK VINA RESULT 的截断/外部文件), None 参与比较直接 TypeError
        # (2026-10-03 审查修复, 与 parse_pdbqt_poses:87 的 1e9 哨兵意图对齐)
        return sorted(light_results,
                      key=lambda r: (r.get("energy_kcal_mol")
                                     if r.get("energy_kcal_mol") is not None else 1e9))

# ============================================================================
# 8.6 单 pose 完整分析(从原 analyze_top_pose 抽出，供 analyze_and_select 复用)
# ============================================================================

def _pose_set_self_scatter(poses) -> Optional[Dict[str, Any]]:
    """全 pose 两两重原子 RMSD 的自散分布(结合模式收敛性的量化指标)。

    仅凭 top1~top2 的 RMSD(top_pose_rmsd_A)只能回答"前两名是否一致"; 柔性/
    多模态对接下打分面可能退化, 整个 pose 集合是否收敛到**唯一结合模式**要看
    全部两两对的距离分布 —— 本函数把这一手算量自动化(旧口径靠人工在笔记里
    记"9 pose 自散 6.92 Å")。

    返回 {"n_poses", "n_pairs", "mean_A", "median_A", "max_A", "min_A"};
    <2 个 pose 或无任何可比对返回 None(不静默填 0 —— 缺失=未测量, 不是"收敛到 0")。
    """
    if not poses or len(poses) < 2:
        return None
    rmsds: List[float] = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            d = _pose_heavy_rmsd(poses[i][1], poses[j][1])
            if d is not None:
                rmsds.append(d)
    if not rmsds:
        return None
    s = sorted(rmsds)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        "n_poses": len(poses),
        "n_pairs": n,
        "mean_A": round(mean, 2),
        "median_A": round(median, 2),
        "max_A": round(s[-1], 2),
        "min_A": round(s[0], 2),
    }


def _pose_divergence_advice(engine: str) -> str:
    """top1~top2 pose 发散 warn 的建议文案, 按引擎口径分叉(2026-10-04 GPU E2E)。

    vina_gpu 的 exhaustiveness 不映射(仅随元数据记录, lane 覆盖度由 thread 保证),
    "建议提高 exhaustiveness" 在 GPU 口径下是无效建议; 论文确认层回 Vina CPU。
    """
    if engine == "vina_gpu":
        return ("建议提高 DOCK_VINA_GPU_THREAD(lane 覆盖度)或检查盒子; "
                "论文确认层请回 Vina CPU exh=32")
    return "建议提高 exhaustiveness 或检查盒子"


def _analyze_single_pose_full(receptor_pdbqt: str, energy: float, atoms,
                              catalytic: List[str], ref_residue: str,
                              ligand_arom_names: List[str], out_json: str,
                              box_center=None, receptor_chain: str = "",
                              box_assumption: str = "",
                              top_pose_rmsd_A: Optional[float] = None,
                              sort_strategy: str = "",
                              selected_pose_index: int = 1,
                              crystal_rmsd_available: bool = False,
                              is_self_dock: bool = False,
                              total_poses: int = 0,
                              poses_light: Optional[List[Dict[str, Any]]] = None,
                              pose_scatter: Optional[Dict[str, Any]] = None,
                              engine: str = "vina") -> Optional[DockResult]:
    """对单个 pose 做完整分析(同原 analyze_top_pose 的核心逻辑)。
    额外接收 top_pose_rmsd_A / sort_strategy / selected_pose_index / poses_light 以写入 DockResult。"""
    notes: List[str] = []
    TH = reliability_thresholds()
    LOG.info("\n=== 最优 pose 验证分析 (能量=%s kcal/mol, pose#%s, 策略=%s) ===" %
             (energy, selected_pose_index, sort_strategy))

    ligand_oxy = [a for a in atoms if a[3] == "O"]
    arom_name_set = set(ligand_arom_names or [])
    ligand_carb = [a for a in atoms if a[3] == "C" or a[4] in arom_name_set]
    docked_arom = [(a[0], a[1], a[2]) for a in atoms if a[4] in arom_name_set]
    ref_targets = docked_arom if docked_arom else [(a[0], a[1], a[2]) for a in ligand_carb]

    rec = load_receptor_coords(receptor_pdbqt)
    res_by_key = {k: v for k, v in rec.items()}
    cat_norm = [_norm_resname(c[:3]) + c[3:] for c in catalytic]

    # 催化亲核
    nuc_res_key0 = (catalytic[0] if catalytic else "")
    nuc_resname = nuc_res_key0[:3]
    has_nuc = nuc_resname in ("SER", "CYS", "THR")
    nuc_keys = [k for k in rec if any(ref_key_match(k, c) for c in cat_norm)]  # 2026-10-04: 统一 ref_key_match(防 'SER17' 前缀撞号 SER170A~179A)
    # 亲核原子口径(2026-10-03): 只取亲核原子(SER/THR=OG, CYS=SG), 见模块头注释
    _na = _NUC_ATOM_BY_RESNAME.get(nuc_resname)
    nuc_pts = [p for k in nuc_keys for p in res_by_key.get(k, [])
               if _na is None or (len(p) > 3 and p[3] == _na)]
    nuc_dist = None
    if has_nuc and nuc_pts and ligand_oxy:
        best = 1e9
        for q in nuc_pts:
            for t in ligand_oxy:
                d = math.dist(q[:3], t[:3])
                if d < best:
                    best = d
        nuc_dist = best if best < 1e8 else None
        if nuc_dist is not None:
            LOG.info(f"  催化亲核残基 -> 配体最近氧: {nuc_dist:.2f} A")
    else:
        if nuc_res_key0:
            msg = (f"催化核亲核体为 {nuc_res_key0} ({nuc_resname}), 非 Ser/Cys/Thr"
                   f"(如 S→A 突变体) — 无亲核氧, nuc→O 不适用, 置为 null")
        else:
            msg = ("未检出催化残基(非水解酶体系或链/三联体未指定) — nuc→O 距离不适用, 置为 null")
        notes.append(msg)
        LOG.info(f"  ⚠ {msg}")

    # 参考残基
    ref_dist = None
    if ref_residue and ref_targets:
        rk = [k for k in rec if ref_key_match(k, ref_residue)]
        ref_pts = [p for k in rk for p in res_by_key.get(k, [])]
        if ref_pts:
            best = 1e9
            for q in ref_pts:
                for t in ref_targets:
                    d = math.dist(q[:3], t[:3])
                    if d < best:
                        best = d
            ref_dist = best if best < 1e8 else None
        if ref_dist is not None:
            LOG.info(f"  参考残基 {ref_residue} -> 配体{'芳香碳' if docked_arom else '碳'}: {ref_dist:.2f} A")
        else:
            LOG.info(f"  参考残基 {ref_residue} 未找到")

    # 芳香环 -> 受体芳香残基
    arom_pocket_dist = None
    arom_pocket_res = ""
    if docked_arom:
        rec_arom = load_receptor_aromatic_carbons(receptor_pdbqt)
        best = 1e9; best_res = ""
        for rk, rpts in rec_arom.items():
            if not (rk.startswith("TRP") or rk.startswith("PHE") or rk.startswith("TYR")):
                continue
            if not rpts or not docked_arom:
                continue
            b = 1e9
            for q in rpts:
                for t in docked_arom:
                    d = math.dist(q[:3], t[:3])
                    if d < b:
                        b = d
            if b < best:
                best, best_res = b, rk
        if best < 1e8:
            arom_pocket_dist, arom_pocket_res = best, best_res
            LOG.info(f"  配体芳香环 -> 受体最近芳香残基 {best_res}: {best:.2f} A")

    # 口袋/埋藏/接触度
    env = set()
    # C9(2026-10-02): 口袋残基枚举与 contact_frac 统一用**重原子**口径 ——
    # 原实现 env 用全原子(含氢), 氢会让距离偏短导致残基虚增入 env。
    lig_heavy = [(a[0], a[1], a[2]) for a in atoms if a[3] != "H"]
    rec_heavy = _receptor_heavy(receptor_pdbqt)
    clashes, buried_frac, med_near, buried = _burial_metrics(
        rec_heavy, lig_heavy if lig_heavy else [])
    if buried:
        notes.append(f"配体被埋入蛋白内部(clash={clashes}, buried_frac={buried_frac}) — pose 不可信")
        LOG.info(f"  ⚠ 配体埋藏: clash={clashes}, buried_frac={buried_frac}, med_near={med_near}A")
    else:
        LOG.info(f"  配体-蛋白 clash={clashes}, med_near={med_near}A (未埋藏)")

    contact = 0.0
    if lig_heavy and rec_heavy:
        contact = sum(1 for lp in lig_heavy if any(math.dist(lp, rp) < 4.5 for rp in rec_heavy)) / len(lig_heavy)
    pocket_contact_frac = round(contact, 3)
    centroid_offset = None
    if box_center and lig_heavy:
        cx = sum(p[0] for p in lig_heavy) / len(lig_heavy)
        cy = sum(p[1] for p in lig_heavy) / len(lig_heavy)
        cz = sum(p[2] for p in lig_heavy) / len(lig_heavy)
        centroid_offset = round(math.dist((cx, cy, cz), tuple(box_center)), 2)
    energy_reliable = (energy is not None and abs(energy) <= _TH["energy_abs_max"])
    ligand_n_heavy = len(lig_heavy)
    if ligand_n_heavy > _TH["heavy_max"]:
        notes.append(f"配体过重({ligand_n_heavy} 重原子), 超出 Vina 稳健尺度, 评分不可靠")
        LOG.info(f"  ⚠ 配体过重: {ligand_n_heavy} 重原子")
    if not energy_reliable:
        notes.append(f"能量 {energy} 失真(疑似超大/异常配体或盲对接失败), 不计入能量排序")
        LOG.info(f"  ⚠ 能量失真: {energy}")
    LOG.info(f"  配体接触度={pocket_contact_frac} (1=全包埋, 0=漂浮), "
             f"质心偏移={centroid_offset}A, 重原子={ligand_n_heavy}, 能量可靠={energy_reliable}")
    for k, v in res_by_key.items():
        for rp in v:
            for lp in lig_heavy:
                if math.dist(rp[:3], lp) < 4.5:
                    env.add(k); break
    env = sorted(env)
    LOG.info(f"  口袋内(<4.5A)受体残基 ({len(env)}): {', '.join(env[:30])}{' ...' if len(env) > 30 else ''}")

    # 可靠性研判
    rel_warnings = list(notes)
    reliability = "ok"
    if buried or (not energy_reliable) or (pocket_contact_frac is not None and pocket_contact_frac < TH["contact_unreliable"]):
        reliability = "unreliable"
    elif box_assumption == "protein_centroid":
        reliability = "warn"
        rel_warnings.append("盒子落在蛋白质几何中心(未指定口袋/无催化三联体/无共晶配体), 结果可靠性低 — 建议用 --box_center 指定已知口袋或 --box_mode ligand")
    elif ligand_n_heavy > TH["heavy_max"] or (pocket_contact_frac is not None and pocket_contact_frac < TH["contact_warn"]):
        reliability = "warn"
    if top_pose_rmsd_A is not None and top_pose_rmsd_A > TH["top_rmsd_warn"]:
        if reliability == "ok":
            reliability = "warn"
        rel_warnings.append(f"top1~top2 pose RMSD={top_pose_rmsd_A:.2f}A "
                           f"发散(对接未收敛), top 打分 pose 不可独信 — "
                           f"{_pose_divergence_advice(engine)}")
    # 全 pose 自散(2026-10-07): 结合模式收敛性 —— 与 top1~top2 发散互补,
    # 打分面退化时 top1~top2 可能恰好相近而整个集合是散开的(多模态)。
    if pose_scatter is not None and pose_scatter["median_A"] > TH["pose_scatter_warn"]:
        if reliability == "ok":
            reliability = "warn"
        rel_warnings.append(
            f"全 pose 两两自散 median RMSD={pose_scatter['median_A']:.2f}A "
            f"(共 {pose_scatter['n_pairs']} 对) —— 结合模式未收敛, "
            f"只能报能量统计, 不能从 top1 断言单一结合模式")
    if reliability == "ok" and has_nuc and nuc_dist is not None and nuc_dist > TH["nuc_offsite"]:
        reliability = "warn"
        rel_warnings.append(f"催化几何偏离: 亲核→酯氧 {nuc_dist:.2f}A > {TH['nuc_offsite']}A, 配体未就位催化位点(三连体附近无接触)")
    if reliability == "unreliable":
        rel_warnings.append("结果不可信: 存在埋藏/能量失真/接触不足, 勿直接采信")
    if reliability != "ok":
        LOG.warning(f"⚠ 结果可靠性 = {reliability}; 关注: {'; '.join(rel_warnings[-3:])}")

    det = classify_reliability_detail({
        "ligand_buried": buried, "energy_reliable": energy_reliable,
        "energy_kcal_mol": energy, "pocket_contact_frac": pocket_contact_frac,
        "box_assumption": box_assumption, "ligand_n_heavy": ligand_n_heavy,
        "has_nucleophile": has_nuc, "nuc_to_ligO_A": nuc_dist,
        "top_pose_rmsd_A": top_pose_rmsd_A, "reliability": reliability,
        "pose_set_rmsd_median_A": (pose_scatter["median_A"] if pose_scatter else None),
    }, TH)
    warn_class, warn_reasons = det["warn_class"], det["warn_reasons"]
    if warn_class == "real":
        LOG.warning(f"  ⮕ warn 二分 = 真问题(real); 原因: {', '.join(warn_reasons)}")
    elif warn_class == "soft":
        LOG.info(f"  ⮕ warn 二分 = 保守提醒(soft, 仅 pose 收敛性); top1~top2 RMSD={top_pose_rmsd_A:.2f}A")

    result = DockResult(ok=True, energy_kcal_mol=energy,
                        catalytic=list(catalytic),
                        nuc_to_ligO_A=nuc_dist,
                        ref_to_ligAromC_A=ref_dist,
                        arom_pocket_dist_A=arom_pocket_dist,
                        arom_pocket_residue=arom_pocket_res,
                        pocket_residues=env, n_poses=(total_poses if total_poses else (len(poses_light) if poses_light else 1)),
                        box_center=tuple(box_center) if box_center else None,
                        receptor_chain=receptor_chain,
                        has_nucleophile=has_nuc, nuc_resname=nuc_resname,
                        ligand_clashes=clashes, ligand_buried_frac=buried_frac,
                        ligand_med_near=med_near, ligand_buried=buried,
                        pocket_contact_frac=pocket_contact_frac,
                        ligand_centroid_offset_A=centroid_offset,
                        ligand_n_heavy=ligand_n_heavy, energy_reliable=energy_reliable,
                        box_assumption=box_assumption, reliability=reliability,
                        warnings=rel_warnings, notes=rel_warnings,
                        top_pose_rmsd_A=top_pose_rmsd_A,
                        pose_set_rmsd_mean_A=(pose_scatter["mean_A"] if pose_scatter else None),
                        pose_set_rmsd_median_A=(pose_scatter["median_A"] if pose_scatter else None),
                        pose_set_rmsd_max_A=(pose_scatter["max_A"] if pose_scatter else None),
                        pose_set_rmsd_n_pairs=(pose_scatter["n_pairs"] if pose_scatter else 0),
                        warn_class=warn_class, warn_reasons=warn_reasons,
                        sort_strategy=sort_strategy,
                        selected_pose_index=selected_pose_index,
                        crystal_rmsd_available=crystal_rmsd_available,
                        is_self_dock=is_self_dock,
                        poses=(poses_light or []))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2, ensure_ascii=False)
    LOG.info(f"✅ 分析已写入: {out_json}")
    return result

def analyze_top_pose(receptor_pdbqt: str, poses, catalytic: List[str],
                     ref_residue: str, ligand_arom_names: List[str],
                     out_json: str, box_center=None,
                     receptor_chain: str = "", box_assumption: str = "",
                     crystal_ligand_pdbqt: Optional[str] = None,
                     engine: str = "vina") -> Optional[DockResult]:
    """对所有 pose 做轻量分析，按策略排序，再对最优 pose 做完整验证分析。
    若提供了 crystal_ligand_pdbqt，则按 pose-vs-crystal RMSD 排序；否则按 Vina energy 排序。"""
    if not poses:
        LOG.error("无对接 pose"); return None

    # 1. 加载晶体配体坐标(若提供)
    crystal = _load_crystal_coords(crystal_ligand_pdbqt)

    # 1.5) 判断是否 self-docking(对接配体 == 共晶配体):
    #   只有配体与共晶配体是同一分子时, pose-vs-crystal RMSD 才有物理意义;
    #   cross-docking(配体不同)时 RMSD 大是正常的, 应回退 energy 排序。
    #   M3(2026-10-02 披露): 同一性判据是**重原子元素组成相等** —— 是必要非充分
    #   条件, 同分异构体(如葡萄糖 vs 果糖)不可区分; 误判后果是排序策略选错,
    #   有 SMILES 时应在调用方先行校验。
    is_self_dock = False
    if crystal and poses:
        lig_comp = _pose_heavy_composition(poses[0][1])
        crystal_comp = _pdbqt_heavy_composition(crystal_ligand_pdbqt)
        is_self_dock = (lig_comp and crystal_comp and lig_comp == crystal_comp)

    # 2. 轻量分析所有 pose
    light_results = []
    for i, (energy, atoms) in enumerate(poses, 1):
        m = _analyze_pose_light(receptor_pdbqt, atoms, energy, catalytic, ref_residue,
                                ligand_arom_names, crystal=crystal, box_center=box_center)
        m["pose_index"] = i
        light_results.append(m)

    # 3. 排序策略: self-docking 按晶体 RMSD; 无共晶配体或 cross-docking 按 energy
    strategy = "rmsd_to_crystal" if is_self_dock else "energy"
    sorted_light = _sort_poses(light_results, strategy)
    strat_desc = ("按晶体配体 RMSD" if is_self_dock
                  else ("按 Vina 能量(cross-docking, 配体≠共晶配体)" if crystal else "按 Vina 能量"))
    LOG.info(f"\n=== 多 pose 排序策略: {strategy} ({strat_desc}) ===")
    for i, r in enumerate(sorted_light[:5], 1):
        extra = f"  RMSD_vs_crystal={r.get('rmsd_to_crystal_A')}A" if r.get('rmsd_to_crystal_A') is not None else ""
        # None 能量防护(2026-10-03): 缺 REMARK VINA RESULT 的 pose 能量为 None,
        # `:.2f` 直接 TypeError —— None 哨兵修复(排序侧)的展示侧漏网点。
        _e = r.get('energy_kcal_mol')
        _es = f"{_e:.2f}" if _e is not None else "N/A"
        LOG.info(f"  pose#{r['pose_index']} energy={_es} contact={r['pocket_contact_frac']}{extra}")

    # 4. 取最优 pose 做完整分析
    best_idx = sorted_light[0]["pose_index"] - 1
    best_energy, best_atoms = poses[best_idx]

    # 5. 计算 Vina top1~top2 RMSD(收敛性指标，不变)
    top_rmsd = None
    if len(poses) >= 2:
        top_rmsd = _pose_heavy_rmsd(poses[0][1], poses[1][1])
        if top_rmsd is not None:
            LOG.info(f"  top1~top2 pose 重原子 RMSD={top_rmsd:.2f}A "
                     f"({'发散, 未收敛' if top_rmsd > 2.0 else '一致'})")

    # 6. 全 pose 自散(2026-10-07): 结合模式收敛性量化 —— 与 top1~top2 发散互补
    pose_scatter = _pose_set_self_scatter(poses)
    if pose_scatter is not None:
        LOG.info(f"  全 pose 两两自散: median={pose_scatter['median_A']:.2f}A, "
                 f"max={pose_scatter['max_A']:.2f}A (共 {pose_scatter['n_pairs']} 对)")

    return _analyze_single_pose_full(
        receptor_pdbqt, best_energy, best_atoms, catalytic, ref_residue,
        ligand_arom_names, out_json, box_center=box_center,
        receptor_chain=receptor_chain, box_assumption=box_assumption,
        top_pose_rmsd_A=top_rmsd, sort_strategy=strategy,
        selected_pose_index=sorted_light[0]["pose_index"],
        crystal_rmsd_available=(crystal is not None),
        is_self_dock=is_self_dock,
        total_poses=len(poses),
        poses_light=sorted_light[:5],
        pose_scatter=pose_scatter,
        engine=engine)

from dockbuddy_dock.reliability import reliability_thresholds, classify_reliability_detail

__all__ = [
    "_analyze_pose_light",
    "_analyze_single_pose_full",
    "_burial_metrics",
    "_calc_pose_rmsd_to_crystal",
    "_kabsch_rmsd_by_name",
    "_load_crystal_coords",
    "_pdbqt_heavy_composition",
    "_pdbqt_ligand_coords",
    "_pose_heavy_composition",
    "_receptor_heavy",
    "_sort_poses",
    "_pose_set_self_scatter",
    "analyze_top_pose",
    "classify_reliability_detail",
    "detect_aromatic_carbons",
    "load_receptor_aromatic_carbons",
    "load_receptor_coords",
    "parse_pdbqt_poses",
    "reliability_thresholds",
]
