# -*- coding: utf-8 -*-
"""对接后重评分: OpenMM OBC(GB) 单点的 MM-GBSA 近似 (优化项 **A2**, 2026-10-03)
==============================================================================
动机
----
Vina 的打分函数是**经验项拟合**出来的(项权重套在 AutoDock4 项上), 绝对值没有
物理意义, 而且同一批 pose 里它倾向于"小而紧"的构象。一个 run 的 ``results.pdbqt``
里通常有 9 个 pose, 排序由 Vina 给出。若换一个**物理项**(溶剂化自由能)再排一次,
排序是否稳定, 就是"这个 run 的 top1 到底靠不靠得住"的直接证据 —— 审稿人常问的
ranking robustness。

做法(MM-GBSA 的**单点**版本)
------------------------------
对每个 pose 建三个体系, 各做一次 OBC(GBneck2) 单点:

    E_complex  = E(受体 + 配体)
    E_receptor = E(受体)
    E_ligand   = E(配体)
    ΔG_bind ≈ E_complex − E_receptor − E_ligand

再与 Vina 打分并列, 输出**两套排序 + 两者一致性**。

配体为什么不能走 ``ForceField.createSystem``(关键工程约束)
----------------------------------------------------------
本机只有 openmm 8.6.1 / openbabel 3.2.1 / pdbfixer 1.12.0 —— 没有任何离线可用的
小分子力场模板(openmmforcefield / openff / rdkit / AMBER antechamber 全都不在)。
``app.ForceField("amber14-all.xml").createSystem()`` 只要拓扑里出现 ``UNL`` 就必然
抛 ``No template found for residue``(实测: 连"和 DLPA 最像但缺 50 个 H"这种误判
模板都被翻出来了), 因为 amber14 只带标准氨基酸/核酸模板。

所以这里换个做法: **受体照旧走 amber14**(``createSystem`` 取出它的
``(q, sigma, epsilon)``、质量、键表), 配体则按 :mod:`libdock.ligparam` 的口径
自己补 ``(q, sigma, epsilon, mass)``, 然后由 ``build_scoring_system`` 把两类粒子
塞进**同一个 System**。三个体系(复合物/受体/配体)走完全相同的构造路径, 保证口径
一致 —— 否则 ``E_cx − E_rec − E_lg`` 里会混进构造差异, 那就是自欺欺人。

必须讲清楚的口径(写报告时别越界)
--------------------------------
1. ΔG_bind 是**单点**值: 不松弛、无构象平均、无熵项(``-TΔS``), 而且配体 LJ 用的是
   UFF **元素型**近似、氢电荷取 0。它是**排序量(re-ranking)不是自由能预测值**,
   别写成 "ΔG = -8.3 kcal/mol" 这种当实验值的表述。
2. 单位: OpenMM 内部 kJ/mol, 对外按 ``KJ_PER_KCAL`` 换 kcal 并注明。
3. 氢: 受体由 ``Modeller.addHydrogens(amber14, pH 7.0)`` 补(与 ``--relax md`` 同
   一 pH 口径); 配体由 obabel ``--addhydrogens`` 补(必须给对 BABEL_DATADIR, 否则
   插件静默失效、一个 H 都不加)。
4. OBC 装配(半径换算 0.5612…、screen 0.5)统一走 ``libdock.solvents``。

成本
----
单点, **不做最小化**(配体无键合项, 一最小化就会散架, 能量失真)。单 pose ≈ 1 次
obabel 补氢 + 2~3 个体系装配, 本机秒级~十几秒; 不需要时别开。

用法(库)
---------
::

    from dockbuddy_dock.rescore import rescore_poses
    recs, meta = rescore_poses("receptor.pdbqt",
                               results_pdbqt="run/results.pdbqt", top_n=9)
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ligparam import (ANGSTROM_TO_NM as ANGstrom_TO_NM,
                       add_ligand_hydrogens, charges_from_pdbqt, ligand_bonds,
                       ligand_particles)
from .poses import parse_pdbqt_poses
from .solvents import OBC_ID, attach_obc_force, obc_params
from dockbuddy_common import openmm_platform  # OpenMM 平台(默认 CPU, DOCK_OPENMM_PLATFORM 可切 CUDA)

LOG = logging.getLogger("libdock.rescore")

KJ_PER_KCAL = 4.184
# amber14 的 1-4 缩放(ForceField.createExceptionsFromBonds 的默认值, 必须用同一套
# 否则三个体系的例外项口径不一致)
COULOMB14_SCALE = 0.8333333333333334
LJ14_SCALE = 0.5

# pose PDB 里配体的落位: 残基名 UNL / 链 X / 残基号 1
_LIG_RESNAME, _LIG_CHAIN, _LIG_RESSEQ = "UNL", "X", 1


def resolve_receptor_pdb(receptor_pdbqt_or_pdb: str) -> str:
    """把受体 PDBQT 解析成 OpenMM 能读的 PDB。

    优先 ``<file>.protein_only.pdb``(``libdock.receptor.prepare_receptor`` 的副产物,
    已剥水/杂原子); 没有就用 obabel 转(``-ipdbqt -opdb``), 便于直接吃 run 目录里的
    ``receptor.pdbqt``。
    """
    p = receptor_pdbqt_or_pdb
    if not p.endswith(".pdbqt"):
        return p
    cand = p + ".protein_only.pdb"
    if os.path.exists(cand):
        return cand
    out = p[:-len(".pdbqt")] + "_protein.pdb"
    try:
        from .obabel import _find_obabel
        import subprocess
        subprocess.run([_find_obabel(), "-ipdbqt", p, "-opdb", "-O", out],
                       capture_output=True, text=True, timeout=600)
    except Exception as e:                                   # pragma: no cover
        LOG.warning("受体 PDBQT→PDB 转换失败(%s), 将直接尝试读原文件", e)
    if not os.path.exists(out):
        LOG.warning("未得到受体 PDB(%s), 仍尝试直接读", out)
    return out if os.path.exists(out) else p


def _read_pdb_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def write_pose_pdb(atoms: Sequence[Sequence[Any]], out_path: str,
                   resname: str = _LIG_RESNAME, chain: str = _LIG_CHAIN,
                   resseq: int = _LIG_RESSEQ) -> str:
    """把 atoms 序列(每项 ``(x, y, z, element, name, ...)``)写成单构象 PDB。

    兼容 ``parse_pdbqt_poses`` 的 7 元组和 ``read_single_pose_pdb`` 的 5 元组;
    元素取 ``atoms[i][3]``, 名字取 ``atoms[i][4]``。

    列宽按 PDB 标准写死(元素在 77–78 列), 不能省格子 —— 后续 obabel / OpenMM 都要
    按这个位置读元素, 少一格就会被"按原子名查表"猜元素。
    """
    with open(out_path, "w", encoding="utf-8") as f:
        _write_pose_lines(f, atoms, resname=resname, chain=chain, resseq=resseq)
    return out_path


def _write_pose_lines(f, atoms: Sequence[Sequence[Any]],
                     resname: str = _LIG_RESNAME, chain: str = _LIG_CHAIN,
                     resseq: int = _LIG_RESSEQ) -> None:
    """落盘原子行(见 :func:`write_pose_pdb` 的列宽约定)。

    注意 resName 在**第 0 基 17–19 列**(PDB 规范的 1 基 18–20 列), 与
    ``poses.parse_pdbqt_poses`` 的 ``ln[17:20]`` 对齐 —— 差一格就会把 "UNL" 读成
    "NL", 这类错最容易在"自己写PDB自己读"的小工具里悄悄发生。
    """
    for i, a in enumerate(atoms, 1):
        x, y, z = float(a[0]), float(a[1]), float(a[2])
        # 2026-10-04 全盘审计: 元素不再截 1 字符 —— 原 [:1] 把 CL→C/FE→F,
        # 写回 PDB 后下游 ligand_particles 给碳/氟的 LJ 与质量(卤代/金属配体全错)。
        elem = (str(a[3]) or "C").strip().upper()[:2]
        name = (str(a[4]) or "C")[:4]
        f.write("HETATM"                        # 1-6
                + f"{i:>5d}"                    # 7-11 serial
                + " "                           # 12
                + f"{name:<4s}"                 # 13-16 atom name
                + " "                           # 17 altLoc
                + f"{resname:>3s}"              # 18-20 resName
                + " "                           # 21
                + f"{chain}"                    # 22 chainID
                + f"{int(resseq):>4d}"          # 23-26 resSeq
                + " "                           # 27 iCode
                + "   "                         # 28-30
                + f"{x:8.3f}{y:8.3f}{z:8.3f}"  # 31-54
                + f"{1.00:6.2f}"               # 55-60 occupancy
                + f"{0.00:6.2f}"               # 61-66 tempFactor
                + " "*10                        # 67-76
                + f"{elem:>2s}"                 # 77-78 element
                + "\n")


def read_single_pose_pdb(path: str) -> List[Tuple[float, float, float, str, str]]:
    """读"单构象 PDB pose"文件, 返回与 write_pose_pdb 同构的列表。"""
    out = []
    for ln in _read_pdb_lines(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip()
        # 2026-10-04 全盘审计: 不截 1 字符(原 [:1] 使 CL→C/FE→F, 查
        # _COV_RADII/UFF 表全错); 元素列缺失时回退原子名首字母。
        elem = ln[76:78].strip().upper() or name[:1].upper()
        out.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54]), elem, name))
    return out


def concat_complex_pdb(receptor_pdb: str, pose_pdb: str, out_path: str) -> str:
    """受体 + 配体拼成一个 PDB —— 仅为落盘/排查用; 真算能量不读它。

    配体统一走 ``write_pose_pdb`` 重排: 残基 UNL / 链 X / 残基号 1(与受体不同链,
    免得配体氨基酸和受体残基被并进同一个 residue)。
    """
    rec = [ln for ln in _read_pdb_lines(receptor_pdb)
           if ln.startswith(("ATOM", "HETATM")) and ln[17:20].strip() != "HOH"]
    lig = read_single_pose_pdb(pose_pdb)
    if not lig:
        raise ValueError(f"pose 文件 {pose_pdb} 无 ATOM/HETATM 记录")
    # 必须先落盘配体、再写受体行: ``write_pose_pdb`` 是 open(path,"w") 语义,
    # 否则后写受体会把已写的配体(以及自身)截断掉 —— 同一个文件句柄被两处打开。
    lig_path = out_path + ".lig"
    write_pose_pdb(lig, lig_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in rec:
            f.write(ln.rstrip("\n") + "\n")
        _write_pose_lines(f, lig)
    if os.path.exists(lig_path):
        os.remove(lig_path)
    return out_path


def prepare_receptor_reference(rec_pdb: str) -> Optional[Dict[str, Any]]:
    """受体 PDB → amber14 补氢后的粒子口径(质量/(q,σ,ε)/键表/坐标)。

    **命名不能叫 ``prepare_receptor``**: ``libdock/__init__.py`` 是 ``import *``
    汇总的, 与 ``libdock.receptor.prepare_receptor`` 同名会把它**静默覆盖**掉,
    于是所有 ``libdock.prepare_receptor(...)`` 调用方(含 tools/oneoff/bench_protocol.py)
    拿到的是本函数、报莫名其妙的 unexpected keyword argument。

    返回 ``None`` 表示失败(调用方应降级而不是崩整条链路)。
    """
    import openmm as mm
    import openmm.app as app
    try:
        pdb = app.PDBFile(rec_pdb)
        ff = app.ForceField("amber14-all.xml")
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff, pH=7.0)
        base = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff,
                               constraints=app.HBonds)
        nb = None
        for i in range(base.getNumForces()):
            f = base.getForce(i)
            if isinstance(f, mm.NonbondedForce):
                nb = f
                break
        if nb is None:
            LOG.warning("rescore: 受体体系里没有 NonbondedForce")
            return None
        n = base.getNumParticles()
        params, masses = [], []
        for i in range(n):
            q, sig, eps = nb.getParticleParameters(i)
            params.append((q.value_in_unit(mm.unit.elementary_charge),
                           sig.value_in_unit(mm.unit.nanometer),
                           eps.value_in_unit(mm.unit.kilojoule_per_mole)))
            masses.append(base.getParticleMass(i).value_in_unit(mm.unit.dalton))
        bonds = [(b[0].index, b[1].index) for b in mod.topology.bonds()]
        return {"n": n, "params": params, "masses": masses, "bonds": bonds,
                "positions": mod.positions,
                "n_res": sum(1 for _ in mod.topology.residues()),
                "receptor_pdb": rec_pdb}
    except Exception as e:                                   # pragma: no cover
        LOG.warning("rescore: 受体体系准备失败(%s)", e)
        return None


def build_scoring_system(rec: Dict[str, Any],
                         lig: Optional[Dict[str, Any]] = None,
                         solvent: str = "obc"):
    """由受体口径 + 可选配体装配一个可算能量的 OpenMM Context。

    ``lig``: ``{"particles": [(q, sigma, eps, mass)...], "bonds": [(i,j)...],
    "positions": 坐标序列}`` —— 由 :func:`libdock.ligparam` 的
    ``ligand_particles`` / ``ligand_bonds`` 产出。

    配体粒子追加在受体粒子之后; 1-4 例外项用**受体的键 + (偏移后的)配体键**一次性
    生成, 保证复合物/受体/配体三个体系的例外项口径一致。
    """
    import openmm as mm
    sys_ = mm.System()
    for m in rec["masses"]:
        sys_.addParticle(m)
    off = rec["n"]
    lig_bonds: List[Tuple[int, int]] = []
    if lig:
        for _p in lig["particles"]:
            sys_.addParticle(_p[3])
        lig_bonds = [(i + off, j + off) for i, j in lig.get("bonds", ())]
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for (q, sig, eps) in rec["params"]:
        nb.addParticle(q, sig, eps)
    if lig:
        for (q, sig, eps, _m) in lig["particles"]:
            nb.addParticle(q, sig, eps)
    # 例外项必须在所有粒子加完之后生成
    bonds = list(rec["bonds"]) + lig_bonds
    nb.createExceptionsFromBonds(bonds, COULOMB14_SCALE, LJ14_SCALE)
    sys_.addForce(nb)
    if solvent == "obc" and attach_obc_force(sys_, LOG) is None:
        return None
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), openmm_platform())
    pos = rec.get("positions")
    if lig:
        import numpy as np
        arr = [list(p) for p in lig["positions"]]
        if rec.get("n") and rec.get("positions") is not None:
            arr = (np.asarray(rec["positions"].value_in_unit(mm.unit.nanometer),
                              dtype=float).tolist() + arr)
        pos = mm.unit.Quantity(np.asarray(arr, dtype=float), mm.unit.nanometer)
    ctx.setPositions(pos)
    return ctx


def potential_energy(ctx) -> Optional[float]:
    """体系单点势能, kJ/mol; 失败返回 None。"""
    if ctx is None:
        return None
    try:
        import openmm as mm
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            mm.unit.kilojoule_per_mole)
    except Exception:                                        # pragma: no cover
        return None


def rescore_poses(receptor_pdb: str, results_pdbqt: str = "",
                  pose_pdbs: Sequence[str] = (), *, top_n: int = 0,
                  minimize_steps: int = 0, seed: int = 20260401,
                  solvent: str = "obc", workdir: str = "",
                  charge_pdbqt: str = "") -> Tuple[List[Dict], Dict]:
    """对一批 pose 做 OBC 单点重评分。

    ``results_pdbqt``: Vina 多构象输出(``REMARK VINA RESULT``), 从中逐 pose 切出;
    ``pose_pdbs``: 已逐个 pose 落盘为单构象 PDB 的文件(与 results_pdbqt 二选一,
    都给则 results_pdbqt 优先)。

    ``charge_pdbqt``: 供电荷的 PDBQT(通常是 run 目录里的 ``ligand.pdbqt`` / 第一个
    ``results.pdbqt``), 按原子名取最后一列电荷。空则配体电荷全记 0。

    返回 ``(records, meta)``; 每个 record 含
    ``{pose, index, n_atoms, vina_energy, E_complex, E_receptor, E_ligand,
    ddG_bind(kJ), ddG_bind_kcal, rank_gb, rank_vina}``。
    """
    wd = workdir or tempfile.mkdtemp(prefix="rescope_")
    os.makedirs(wd, exist_ok=True)
    rec_pdb = resolve_receptor_pdb(receptor_pdb)

    # --- 1) 受体口径(三体系共用, 只准备一次) ---
    rec = prepare_receptor_reference(rec_pdb)
    if rec is None:
        return [], {"receptor_pdb": rec_pdb, "n_poses": 0,
                    "error": "受体体系准备失败"}

    # --- 2) 切 pose ---
    poses: List[Dict[str, Any]] = []
    if results_pdbqt and os.path.exists(results_pdbqt):
        for i, (energy, atoms) in enumerate(parse_pdbqt_poses(results_pdbqt), 1):
            out = os.path.join(wd, f"pose{i:02d}.pdb")
            write_pose_pdb(atoms, out)
            poses.append({"path": out, "vina_energy": energy,
                          "index": i, "n_atoms": len(atoms)})
            if top_n and len(poses) >= top_n:
                break
    for p in pose_pdbs or ():
        if not os.path.exists(p):
            LOG.warning("rescore: pose 文件不存在 %s, 跳过", p)
            continue
        atoms = read_single_pose_pdb(p)
        if not atoms:
            continue
        poses.append({"path": p, "vina_energy": None, "index": len(poses) + 1,
                      "n_atoms": len(atoms)})
    if not poses:
        LOG.warning("rescore: 没有可评分的 pose")
        return [], {"receptor_pdb": rec_pdb, "n_poses": 0, "error": "no pose"}

    cmap = charges_from_pdbqt(charge_pdbqt or results_pdbqt) if (charge_pdbqt or results_pdbqt) else {}

    # --- 3) 逐 pose: 补氢 → 配体粒子 → 三个体系 ---
    e_rec = potential_energy(build_scoring_system(rec, None, solvent=solvent))
    records: List[Dict[str, Any]] = []
    for p in poses:
        pose_path = p["path"]
        rec_r: Dict[str, Any] = {"pose": os.path.basename(pose_path),
                                 "index": p["index"], "n_atoms": p["n_atoms"],
                                 "vina_energy": p.get("vina_energy")}
        lig_h = add_ligand_hydrogens(pose_path,
                                     os.path.join(wd, f"pose_h{os.path.basename(pose_path)}"))
        if lig_h is None:
            rec_r["error"] = "配体补氢失败, 该 pose 跳过"
            records.append(rec_r)
            continue
        atoms = read_single_pose_pdb(lig_h)
        if not atoms:
            rec_r["error"] = "补氢后 pose 无原子"
            records.append(rec_r)
            continue
        # 位置必须是纳米: PDB 存埃、OpenMM 存纳米, 直接拼会让配体跑到几十 nm 外,
        # 于是"复合物 − 受体 − 配体"恰好抵消成 0(实测 ddG ~ 1e-3 kcal/mol)。
        lig = {"particles": ligand_particles(atoms, cmap),
               "bonds": ligand_bonds(atoms),
               "positions": [(a[0] * ANGstrom_TO_NM, a[1] * ANGstrom_TO_NM,
                              a[2] * ANGstrom_TO_NM) for a in atoms]}
        e_cx = potential_energy(build_scoring_system(rec, lig, solvent=solvent))
        # 2026-10-04 全盘审计: 配体单点**逐 pose 各算一次**, 废止"刚性对接各 pose
        # 内构象相同 → 复用"的旧前提 —— DockingConfig.ligand_flexible 默认 True,
        # 柔性树采样使各 pose 配体内构象不同, 而配体体系只装 Nonbonded+OBC(内能
        # 完全由构象决定), 复用会把 pose1 的内能摊到所有 pose → rank_gb 失真。
        # 配体体系粒子数十个, 逐 pose 成本秒级以下。
        e_lg = potential_energy(build_scoring_system(
            {"n": 0, "params": [], "masses": [], "bonds": []}, lig, solvent=solvent))
        ddG = None if (None in (e_cx, e_rec, e_lg)) else e_cx - e_rec - e_lg
        rec_r.update({"E_complex": e_cx, "E_receptor": e_rec, "E_ligand": e_lg,
                      "n_lig_atoms": len(atoms),
                      "ddG_bind": ddG,
                      "ddG_bind_kcal": None if ddG is None else ddG / KJ_PER_KCAL})
        if e_cx is not None:
            rec_r["E_complex_kcal"] = e_cx / KJ_PER_KCAL
        records.append(rec_r)

    # --- 4) 两套排序(空值一律沉底) ---
    def _nullsink(r, key, asc=True):
        return (r.get(key) is None, r.get(key) if asc else -(r.get(key) or 0))

    if any(r.get("ddG_bind") is not None for r in records):
        for i, r in enumerate(sorted(records, key=lambda r: _nullsink(r, "ddG_bind")), 1):
            r["rank_gb"] = i
    if any(r.get("vina_energy") is not None for r in records):
        for i, r in enumerate(sorted(records, key=lambda r: _nullsink(r, "vina_energy")), 1):
            r["rank_vina"] = i

    pairs = [r for r in records if r.get("rank_gb") and r.get("rank_vina")]
    meta = {
        "receptor_pdb": rec_pdb, "workdir": wd, "n_poses": len(records),
        "seed": seed, "top_n": top_n,
        "solvent": (OBC_ID if solvent == "obc"
                    else "vacuum(无隐式溶剂, 仅对照)"),
        "obc_params": (obc_params() if solvent == "obc" else {}),
        "model": ("MM-GBSA single-point(OBC/GBneck2 + LJ, 不松弛/无熵项, 仅供 re-ranking)"
                  if solvent == "obc" else "vacuum single-point(仅对照)"),
        "ligand_model": ("UFF 元素型 LJ + PDBQT 电荷(配体), amber14 ff14SB(受体), "
                         "氢电荷取 0"),
        "ligand_charge_source": (charge_pdbqt or (results_pdbqt if results_pdbqt else "")),
        # 2026-10-04 起: 配体单点逐 pose 独立计算(柔性配体各 pose 内构象不同);
        # 旧产物该字段为 True 的 rank_gb 口径与本轮起不可直接比。
        "e_ligand_reused_across_poses": False,
        "n_ranked_both": len(pairs),
        "rank_agreement": (None if len(pairs) < 2 else
                           round(sum(1 for r in pairs
                                     if r["rank_gb"] == r["rank_vina"]) / len(pairs), 3)),
    }
    return records, meta


def mmgbsa_single_structure(receptor_pdb: str, ligand_pose_pdb: str, *,
                            solvent: str = "obc", charge_pdbqt: str = "",
                            ligand_smiles: str = "", seed: int = 20260401,
                            workdir: str = "") -> Dict[str, Any]:
    """单结构 MM-GBSA 结合能估算(单点 OBC/GBneck2 + UFF 元素型 LJ)。

    给定**受体** + **单个已对接 pose**, 估算

        ΔG_bind ≈ E_complex − E_receptor − E_ligand

    三体系走与 :func:`rescore_poses` 完全相同的构造路径(受体 amber14 取参,
    配体自补 UFF 元素型 LJ + PDBQT 电荷), 保证口径一致。区别在于输入是**单个
    已确定的 pose**(一个配体位姿 PDB), 产出单个 ΔG_bind 估计 + 自否证判词,
    不做多 pose 排序。

    **必读口径(写报告时别越界, 与 rescore_poses 同款诚实纪律)**
    -----------------------------------------------------------
    1. ΔG_bind 是**单点值**: 不松弛、无构象平均、无熵项(−TΔS)。它是**排序量
       (re-ranking)** 不是自由能预测值, 别写成 "ΔG = −8.3 kcal/mol" 这种当
       实验值的表述。
    2. 配体 LJ 用 UFF **元素型**近似、氢电荷取 0; 受体 amber14 ff14SB 补氢(pH 7.0)。
       单位: OpenMM 内部 kJ/mol, 对外按 ``KJ_PER_KCAL`` 换 kcal 并注明。
    3. 单点能量对 pose 内构象极敏感(未最小化 → 坐标冲突/氢取向会显著搬动能量),
       只能用于**同力场/同溶剂/同电荷口径**下的相对比较。
    4. 电荷来源: 优先 ``charge_pdbqt``(配体 PDBQT 末列电荷, 按原子名对齐);
       不给则配体电荷全 0(已知近似, 已在 verdict 标注)。

    **输入约定**
    ------------
    ``receptor_pdb``: 受体 .pdb(或 .pdbqt, 自动转 PDB 并缓存); 期望是**已准备好的
    蛋白**(含氢与否都行, 这里统一 amber14 补氢)。
    ``ligand_pose_pdb``: **只含配体**的单构象 PDB(HETATM, resName 任意; 推荐 UNL),
    坐标单位 Å。若传进含受体的合并 PDB, read_single_pose_pdb 会把所有 ATOM/HETATM
    当配体读 → 结果无意义, 调用方须保证只含配体。

    返回 ``{ok, available, E_complex_kJ, E_receptor_kJ, E_ligand_kJ,
    deltaG_bind_kJ, deltaG_bind_kcal, verdict, ...}``; 任何前置失败 ``ok=False``
    并带 ``error``, 不抛异常(调用方降级, 别崩整条链路)。
    """
    out: Dict[str, Any] = {
        "ok": False, "available": False,
        "receptor_pdb": receptor_pdb, "ligand_pose_pdb": ligand_pose_pdb,
        "ligand_smiles": ligand_smiles or "",
        "seed": seed, "solvent": (OBC_ID if solvent == "obc" else "vacuum"),
    }
    if not os.path.exists(receptor_pdb):
        out["error"] = f"受体文件不存在: {receptor_pdb}"
        return out
    if not os.path.exists(ligand_pose_pdb):
        out["error"] = f"配体位姿文件不存在: {ligand_pose_pdb}"
        return out

    wd = workdir or tempfile.mkdtemp(prefix="mmgbsa_")
    os.makedirs(wd, exist_ok=True)

    # --- 受体口径(与 rescore_poses 同一 prepare_receptor_reference) ---
    rec_pdb = resolve_receptor_pdb(receptor_pdb)
    rec = prepare_receptor_reference(rec_pdb)
    if rec is None:
        out["error"] = "受体体系准备失败(amber14 补氢/建系失败; 详见日志)"
        return out

    # --- 配体: 补氢 → 粒子(电荷按原子名对齐) ---
    lig_h = add_ligand_hydrogens(
        ligand_pose_pdb, os.path.join(wd, "lig_h.pdb"))
    if lig_h is None:
        out["error"] = ("配体补氢失败(obabel 不可用或失败); 单结构 MM-GBSA 无法继续")
        return out
    atoms = read_single_pose_pdb(lig_h)
    if not atoms:
        out["error"] = "补氢后配体位姿无 ATOM/HETATM 记录"
        return out
    cmap = charges_from_pdbqt(charge_pdbqt) if charge_pdbqt else {}
    lig = {
        "particles": ligand_particles(atoms, cmap),
        "bonds": ligand_bonds(atoms),
        "positions": [(a[0] * ANGstrom_TO_NM, a[1] * ANGstrom_TO_NM,
                       a[2] * ANGstrom_TO_NM) for a in atoms],
    }

    # --- 三体系单点(同一构造路径, 口径一致) ---
    e_rec = potential_energy(build_scoring_system(rec, None, solvent=solvent))
    e_cx = potential_energy(build_scoring_system(rec, lig, solvent=solvent))
    # 配体体系: 空受体(0 粒子) + 配体, 仅装 Nonbonded+OBC
    e_lg = potential_energy(build_scoring_system(
        {"n": 0, "params": [], "masses": [], "bonds": []}, lig, solvent=solvent))
    if None in (e_cx, e_rec, e_lg):
        out["error"] = ("单点能量计算失败(体系装配或 Context 构建失败); "
                        "详见日志(openmm 异常)")
        return out

    ddG = e_cx - e_rec - e_lg
    ddG_kcal = ddG / KJ_PER_KCAL

    # --- 自否证 verdict(与 evidence.VerdictBuilder 同构) ---
    verdict = {
        "ok": True,
        "applicable": True,
        "model": ("MM-GBSA single-point(OBC/GBneck2 + LJ, 不松弛/无熵项, 仅供 re-ranking)"
                  if solvent == "obc" else "vacuum single-point(仅对照)"),
        "banned_phrases": [
            "结合自由能为 -X kcal/mol（把单点重评分当实验 ΔG 自由能写）",
            "MM-GBSA 预测结合自由能 = X kcal/mol",
            "ΔG_bind 越负 → 活性越高（单点值无 -TΔS 熵项, 不可推断 kcat/活性）",
            "可与实验测得的 ΔG 直接比较",
        ],
        "allowed_phrases": [
            "该值为 MM-GBSA 单点重评分(不松弛/无构象平均/无 -TΔS 熵项), "
            "仅供同体系内 pose 间重排序与受体间相对比较",
            "ΔG_bind 是单点 OBC/GBneck2 + 配体 UFF 元素型 LJ 近似, 配体氢电荷取 0",
            "仅当比较对象使用完全相同的力场/溶剂/截断/电荷口径时, 差值才可解释",
        ],
        "single_point_warning": (
            "单点值对 pose 内构象极敏感(未最小化, 坐标冲突/氢取向会搬动能量), "
            "只能用于同口径相对比较, 不能当作绝对结合自由能"),
        "narrative": (
            f"单结构 MM-GBSA 单点估算 ΔG_bind ≈ {ddG_kcal:.2f} kcal/mol "
            f"(E_complex={e_cx / KJ_PER_KCAL:.1f} − E_receptor="
            f"{e_rec / KJ_PER_KCAL:.1f} − E_ligand={e_lg / KJ_PER_KCAL:.1f} kJ/mol)。"
            f"这是重排序量而非自由能预测值, 仅适用于同力场/同溶剂口径下的相对比较。"),
    }

    out.update({
        "ok": True, "available": True,
        "receptor_pdb": rec_pdb, "ligand_pose_pdb": ligand_pose_pdb,
        "ligand_n_atoms": len(atoms),
        "ligand_charge_source": (charge_pdbqt if charge_pdbqt
                                 else "(none; 配体电荷全 0)"),
        "model": ("MM-GBSA single-point(OBC/GBneck2 + LJ, 不松弛/无熵项, 仅供 re-ranking)"
                  if solvent == "obc" else "vacuum single-point(仅对照)"),
        "ligand_model": ("UFF 元素型 LJ + PDBQT 电荷(配体), amber14 ff14SB(受体), "
                         "氢电荷取 0"),
        "obc_params": (obc_params() if solvent == "obc" else {}),
        "E_complex_kJ": e_cx, "E_receptor_kJ": e_rec, "E_ligand_kJ": e_lg,
        "E_complex_kcal": e_cx / KJ_PER_KCAL,
        "E_receptor_kcal": e_rec / KJ_PER_KCAL,
        "E_ligand_kcal": e_lg / KJ_PER_KCAL,
        "deltaG_bind_kJ": ddG, "deltaG_bind_kcal": ddG_kcal,
        "note": ("ΔG_bind 为单点 OBC/GBneck2 + UFF 元素型 LJ 近似: 不松弛、无构象平均、"
                 "无 -TΔS 熵项; 配体氢电荷取 0。仅供同口径相对比较(re-ranking), "
                 "不当绝对结合自由能。"),
        "verdict": verdict, "workdir": wd,
    })
    return out


__all__ = [
    "KJ_PER_KCAL", "rescore_poses", "mmgbsa_single_structure", "mmgbsa_md",
    "prepare_receptor_reference", "build_scoring_system",
    "resolve_receptor_pdb", "write_pose_pdb", "read_single_pose_pdb",
    "concat_complex_pdb", "potential_energy",
    # OBC_ID / obc_params 不在此再导出: 它们属于 libdock.solvents, 两边都导出会让
    # "libdock 顶层同名符号"出现两份来源(虽是同一对象, 但排查时会误导)。
]


def _mmgbsa_md_subprocess(receptor_pdb, ligand_pose_pdb, ligand_smiles,
                          md_steps, md_step_ps, temperature_k, snapshots, seed,
                          workdir):
    """在专用 conda 环境里用子进程跑 MD 轨迹版 MM-GBSA(见 mmgbsa_md_worker.py)。

    返回结果 dict; 若专用环境/worker 不可用, 返回 None(由调用方走诚实报错)。
    """
    import json
    import subprocess
    py = os.environ.get("DOCK_MMGBSA_PYTHON", "/opt/miniconda3/envs/mmgbsa/bin/python")
    if not os.path.exists(py):
        return None
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mmgbsa_md_worker.py")
    if not os.path.exists(worker):
        return None
    wd = workdir or tempfile.mkdtemp(prefix="mmgbsa_md_")
    os.makedirs(wd, exist_ok=True)
    out_json = os.path.join(wd, "mmgbsa_md_result.json")
    device = os.environ.get("DOCK_OPENMM_PLATFORM") or "CPU"
    cmd = [py, worker, "--receptor", receptor_pdb, "--smiles", ligand_smiles,
           "--ligand", ligand_pose_pdb, "--md_steps", str(md_steps),
           "--md_step_ps", str(md_step_ps), "--temperature_k", str(temperature_k),
           "--snapshots", str(snapshots), "--seed", str(seed),
           "--workdir", wd, "--out", out_json, "--device", device]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"MM-GBSA 子进程启动失败: {e}"}
    if os.path.exists(out_json):
        try:
            return json.load(open(out_json, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"MM-GBSA 结果解析失败: {e}"}
    tail = (r.stderr or r.stdout or "")[-400:]
    return {"ok": False, "error": f"MM-GBSA 子进程 rc={r.returncode}: {tail}"}


def mmgbsa_md(receptor_pdb: str, ligand_pose_pdb: str, *, ligand_smiles: str = "",
              md_steps: int = 10000, md_step_ps: float = 0.002,
              temperature_k: float = 300.0, snapshots: int = 20,
              seed: int = 20260401, workdir: str = "") -> Dict[str, Any]:
    """MD 弛豫 + 轨迹 MM-GBSA(结合自由能, 动态口径, 2026-10-08)。

    对酶-底物复合物做能量最小化 + 短 NVT MD(释放 AF3 静态结构的空间应力), 在轨迹
    快照上逐帧算 ΔG_bind = E_complex − E_receptor − E_ligand, 返回 mean±std。

    **关键口径(与 mmgbsa_single_structure 同款诚实纪律, 且更严格)**:
    1. 需要**完整的小分子力场**才能做有物理意义的 MD —— 即 openff/openmmforcefield
       + rdkit(配体 SMILES 参数化)。本机若缺这两者, 直接返回 ok=False(不假装能跑)。
       **版本约束(2026-10-10 核实)**: openff-toolkit 0.17.0+ / openmmforcefields
       0.15.0+ 的 requires_python 已升到 >=3.11; 且两者的 PyPI 渠道已被 Open Force
       Field 官方**整体撤回**(yanked), 唯一可靠渠道是 conda-forge。py3.10 镜像须在
       独立 conda 环境 pin ``openff-toolkit=0.16`` + ``openmmforcefields=0.14``
       (见 deploy/setup_v6.sh 的 'mmgbsa' 环境); 本函数在自身环境缺 openff 时, 用
       ``DOCK_MMGBSA_PYTHON`` 指向该环境, 经 :mod:`libdock.mmgbsa_md_worker` 子进程跑。
    2. 受体 amber14(ff14SB) 补氢 pH 7.0; 配体 openff 参数化; 溶剂 GBneck2 隐式。
    3. 即便跑通, ΔG_bind 仍是隐式溶剂 + 短轨迹 + 无显式熵项的**排序量**, 不是
       绝对自由能; 快照平均能压掉单点对坐标的敏感性, 但仍不可与实验 ΔG 直接对标。

    返回 ``{ok, available, deltaG_bind_kcal, deltaG_bind_std_kcal, md_steps,
    snapshots, verdict, ...}``; 前置失败 ok=False 带 error, 不抛异常。
    """
    out: Dict[str, Any] = {"ok": False, "available": False, "model": "MM-GBSA MD",
                           "md_steps": md_steps, "snapshots": snapshots,
                           "receptor_pdb": receptor_pdb,
                           "ligand_pose_pdb": ligand_pose_pdb}
    if not ligand_smiles:
        out["error"] = "MD 轨迹 MM-GBSA 需要 ligand_smiles(openff 参数化配体)"
        return out
    # 完整 FF 能力探测(诚实: 缺 openff/rdkit/openmmforcefields 不做伪 MD)
    try:
        import openff.toolkit  # noqa: F401
        import rdkit  # noqa: F401
        import openmm  # noqa: F401
        import openmmforcefields  # noqa: F401
    except ImportError as e:
        # 主环境缺 openff(openff-toolkit/openmmforcefields 的 PyPI 渠道已被官方撤回,
        # 只能走 conda-forge 独立环境)。尝试用 DOCK_MMGBSA_PYTHON 指向的环境子进程重跑。
        sub = _mmgbsa_md_subprocess(receptor_pdb, ligand_pose_pdb, ligand_smiles,
                                    md_steps, md_step_ps, temperature_k,
                                    snapshots, seed, workdir)
        if sub is not None:
            out.update(sub)
            return out
        out["error"] = ("MD 轨迹 MM-GBSA 需要 openff-toolkit + rdkit + openmmforcefields "
                        f"(缺 {getattr(e, 'name', e)})。请安装; 或退回 "
                        "mmgbsa_single_structure(单点静态口径)。"
                        # py3.10 版本约束: openff-toolkit 0.17+ / openmmforcefields
                        # 0.15+ 起 requires_python>=3.11; 且 PyPI 渠道已被官方撤回,
                        # 镜像内走 conda-forge 独立环境(见 deploy/setup_v6.sh)。
                        "注: 镜像已提供 conda-forge 'mmgbsa' 环境"
                        "(DOCK_MMGBSA_PYTHON=/opt/miniconda3/envs/mmgbsa/bin/python)。")
        return out

    wd = workdir or tempfile.mkdtemp(prefix="mmgbsa_md_")
    os.makedirs(wd, exist_ok=True)

    try:
        result = _run_md_mmgbsa(receptor_pdb, ligand_pose_pdb, ligand_smiles,
                                md_steps, md_step_ps, temperature_k, snapshots, seed, wd)
    except Exception as e:  # noqa: BLE001 —— 前置失败降级, 不崩链路
        out["error"] = f"MD 轨迹 MM-GBSA 失败: {e}"
        return out
    out.update(result)
    return out


def _run_md_mmgbsa(receptor_pdb, ligand_pose_pdb, ligand_smiles, md_steps,
                   md_step_ps, temperature_k, snapshots, seed, wd) -> Dict[str, Any]:
    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmforcefields.generators import SystemGenerator

    rec_pdb = resolve_receptor_pdb(receptor_pdb)
    complex_pdb = concat_complex_pdb(rec_pdb, ligand_pose_pdb,
                                     os.path.join(wd, "complex.pdb"))
    # 2026-10-10 修正: "openff-2.0.0.offxml" 不能放进 forcefields 列表(OpenMM 按
    # 文件路径加载, 而 offxml 是 openff-toolkit 内部打包的, 会报 "Could not locate
    # file"); 小分子力场必须走 small_molecule_forcefield="openff-2.0.0"。
    # 修正2: molecules 参数必须传 openff Molecule 对象(不是 SMILES 字符串), 否则
    # 报 "AttributeError: 'str' object has no attribute 'to_smiles'"。
    from openff.toolkit.topology import Molecule
    mol = Molecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
    gen = SystemGenerator(forcefields=["amber/ff14SB.xml"],
                          small_molecule_forcefield="openff-2.0.0",
                          molecules=[mol],
                          cache=os.path.join(wd, "ffcache"))

    # 复合物拓扑 + 补氢 + 建系
    cx_pdb = app.PDBFile(complex_pdb)
    modeller = app.Modeller(cx_pdb.topology, cx_pdb.positions)
    modeller.addHydrogens(forcefield=app.ForceField("amber14-all.xml"), pH=7.0)
    cx_top = modeller.topology

    # 受体-only 拓扑(蛋白链) 与 配体-only 拓扑(小分子链): 按 chain 类型区分
    rec_top, lig_top = _split_complex_topology(cx_top, ligand_smiles)

    # molecules 参数是「小分子列表」(openff Molecule 对象), 不是 gen.forcefield,
    # 也不是 SMILES 字符串(见修正2)。
    cx_sys = gen.create_system(cx_top, molecules=[mol])
    rec_sys = gen.create_system(rec_top, molecules=[])
    lig_sys = gen.create_system(lig_top, molecules=[mol])

    integrator = mm.LangevinIntegrator(temperature_k * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       md_step_ps * unit.picoseconds)
    sim = app.Simulation(cx_top, cx_sys, integrator, openmm_platform())
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)

    # 三个 Context(受体/配体): 建好一次, 每帧只改 positions
    rec_ctx = mm.Context(rec_sys, mm.VerletIntegrator(0.001), openmm_platform())
    lig_ctx = mm.Context(lig_sys, mm.VerletIntegrator(0.001), openmm_platform())

    # 粒子 → 拓扑原子 映射(用于每帧拆分快照坐标到受体/配体)
    cx_n = cx_top.getNumAtoms()
    rec_n = rec_top.getNumAtoms()
    lig_n = lig_top.getNumAtoms()
    if rec_n + lig_n != cx_n:
        return {"ok": False, "error": f"拓扑拆分不一致({rec_n}+{lig_n}!={cx_n})"}

    ddgs = []
    step_per_snap = max(1, md_steps // snapshots)
    for _ in range(snapshots):
        sim.step(step_per_snap)
        state = sim.context.getState(getPositions=True)
        xyz = state.getPositions(asNumpy=True)  # (cx_n, 3) nm
        rec_ctx.setPositions(xyz[:rec_n])
        lig_ctx.setPositions(xyz[rec_n:])
        ecx = sim.context.getState(getEnergy=True).getPotentialEnergy() \
            .value_in_unit(unit.kilojoule_per_mole)
        erec = rec_ctx.getState(getEnergy=True).getPotentialEnergy() \
            .value_in_unit(unit.kilojoule_per_mole)
        elig = lig_ctx.getState(getEnergy=True).getPotentialEnergy() \
            .value_in_unit(unit.kilojoule_per_mole)
        ddgs.append(ecx - erec - elig)

    mean = float(np.mean(ddgs)); std = float(np.std(ddgs))
    return {"ok": True, "available": True,
            "deltaG_bind_kJ": mean, "deltaG_bind_std_kJ": std,
            "deltaG_bind_kcal": mean / KJ_PER_KCAL,
            "deltaG_bind_std_kcal": std / KJ_PER_KCAL,
            "verdict": {
                "ok": True, "applicable": True,
                "model": "MM-GBSA MD(openff 配体 + amber14 受体 + GBneck2, "
                         "短 NVT 轨迹平均; 排序量, 非绝对自由能)",
                "banned_phrases": ["绝对结合自由能", "ΔG 预测", "与实验 ΔG 直接对标"],
            }}


def _split_complex_topology(cx_top, ligand_smiles):
    """把复合物 topology 拆成受体(蛋白/核酸链)与配体(小分子链)两个 topology。

    按 chain 数拆分: 假定最后一(或若干)条链是配体小分子(openff 参数化的链)。
    这里用「链 id 是否为配体」的最简启发: 配体链的 residue 不含蛋白质/核酸标准名。
    """
    import openmm.app as app
    import openmm.app.element as elem
    import openmm.app.topology as top

    rec_chains = []
    lig_chains = []
    PROTEIN_RES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
                   "TYR", "VAL", "A", "C", "G", "U", "HOH", "WAT"}
    for chain in cx_top.chains():
        names = {r.name for r in chain.residues()}
        if names and names.issubset(PROTEIN_RES):
            rec_chains.append(chain)
        else:
            lig_chains.append(chain)

    def _topology_from_chains(chains):
        t = top.Topology()
        new_chain = None
        for chain in chains:
            new_chain = t.addChain(chain.id)
            for res in chain.residues():
                new_res = t.addResidue(res.name, new_chain)
                for a in res.atoms():
                    t.addAtom(a.name, a.element, new_res)
        return t

    rec_top = _topology_from_chains(rec_chains)
    lig_top = _topology_from_chains(lig_chains)
    return rec_top, lig_top
