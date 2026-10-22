# -*- coding: utf-8 -*-
"""MM-GBSA 轨迹版 **自包含** worker(在专用 conda 环境里真跑, 不依赖 pipeline 包)
===================================================================================
为什么单独一个 worker: MD 轨迹版 MM-GBSA 需要 ``openff-toolkit`` + ``openmmforcefields``
+ ``rdkit`` + ``openmm`` 同处一个 python 环境; 而这两者(openff-toolkit / openmmforcefields)
的 PyPI 渠道已被 Open Force Field 官方**整体撤回**(yanked), 唯一可靠渠道是 conda-forge。
故在镜像里把它们装进独立的 conda 环境 ``mmgbsa``(python 3.10), 本 worker 由该环境的
python 执行; 主流程(rescore.mmgbsa_md)在自身环境缺 openff 时, 用子进程回调本 worker。

**自包含硬约束**: 本文件只能 import 标准库 + openmm/openmmforcefields/openff/rdkit/numpy,
不得 import pipeline 内部模块(common/libdock/...), 否则 conda 环境找不到。

口径(与 rescore.mmgbsa_md 一致, 更严格):
  - 受体 amber14(ff14SB) 补氢 pH 7.0; 配体 openff-2.0.0 参数化; 溶剂 GBneck2 隐式。
  - ΔG_bind = E_complex − E_receptor − E_ligand, 在短 NVT 轨迹快照上逐帧算, 返回 mean±std。
  - 结果只是**排序量**, 不是绝对结合自由能。

关键工程决策(2026-10-10 定稿, 踩坑记录):
  1. **配体拓扑用 ``OFFTopology.from_molecules([mol]).to_openmm()`` 建**, 不做 PDB 手工
     重命名残基。原因: openmmforcefields 的小分子模板按**图同构**匹配(元素 + 键), 与残基
     名无关; 而 ``app.PDBFile`` 读 HETATM 时**不建键**, 配体残基变成 0 边图, 图同构必失败
     (报 "No template found for residue ... similar to DT5")。from_molecules 自带正确键序。
  2. **配体电荷用 Gasteiger**(``assign_partial_charges("gasteiger")``, 走 RDKit, 无需
     ambertools/OpenEye)。openff-2.0.0 默认 AM1-BCC 需要 antechamber/OpenEye(镜像没装),
     会报 "No registered toolkits can provide assign_partial_charges"。Gasteiger 是标准
     快速电荷近似, 与"排序量"定位匹配, 已在结果 note 注明。
  3. **openff 0.16 的 ``Molecule.to_smiles()`` 默认返回显式氢 SMILES**, 但配体命名不再依赖
     to_smiles(), 所以此坑已无关紧要。
  4. **镜像内 mmgbsa 环境的 openmm 须用系统 CUDA 12 版**(symlink), conda-forge openmm 8.6.0
     在 A10 驱动 580.x(CUDA 13) 上抛 ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``。见
     setup_v6.sh §6c2(把 /usr/local 的 openmm + OpenMM.libs symlink 进 mmgbsa env)。

用法:
  python mmgbsa_md_worker.py --receptor rec.pdb --smiles '...' \
      [--ligand lig.pdb] [--md_steps 10000] [--snapshots 20] \
      [--workdir /tmp/x] [--out result.json] [--device CUDA]
  --ligand 缺省时用 openff(RDKit ETKDG) 生成 3D 构象并平移到 +80 Å(远离蛋白),
  供冒烟/自检; 真实使用时 --ligand 传对接 pose, 坐标经 RDKit 子结构匹配映射回 openff 分子。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback

KJ_PER_KCAL = 4.184


def _platform(name: str):
    import openmm as mm
    try:
        return mm.Platform.getPlatformByName(name)
    except Exception:  # noqa: BLE001
        return mm.Platform.getPlatformByName("CPU")


def _molecule_from_smiles(smiles: str):
    from openff.toolkit.topology import Molecule
    return Molecule.from_smiles(smiles, allow_undefined_stereo=True)


def _assign_pose_conformer(smiles: str, ligand_pdb: str):
    """把对接 pose 的坐标映射回 openff 分子(返回**带 pose 坐标**的新 Molecule)。

    对接 pose PDB 通常无氢、键序缺失(PDBQT→PDB 只有单键)。用 SMILES 模板先赋键序、
    再 3D 补氢、再转 openff, 保证参数化的图与 SMILES 一致。
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit.topology import Molecule
    rd = Chem.MolFromPDBFile(ligand_pdb, removeHs=False, sanitize=True)
    if rd is None:
        rd = Chem.MolFromPDBFile(ligand_pdb, removeHs=False, sanitize=False)
    if rd is None:
        raise ValueError(f"rdkit 无法解析配体 pose: {ligand_pdb}")
    template = Chem.MolFromSmiles(smiles)
    if template is not None:
        try:
            rd = AllChem.AssignBondOrdersFromTemplate(template, rd)
        except Exception:  # noqa: BLE001
            pass
    rd = Chem.AddHs(rd, addCoords=True)
    return Molecule.from_rdkit(rd, allow_undefined_stereo=True)


def _mol_positions_nm(mol, offset_nm: float = 0.0):
    """openff 分子构象 → openmm 坐标 Quantity(nm)。offset_nm 沿 +x 平移(自检远离蛋白)。"""
    import numpy as np
    import openmm.unit as unit
    xyz_ang = np.asarray(mol.conformers[0].m, dtype=float)  # (n,3) 埃
    xyz_nm = xyz_ang * 0.1
    if offset_nm:
        xyz_nm = xyz_nm + np.array([offset_nm, 0.0, 0.0])
    return unit.Quantity(xyz_nm, unit.nanometer)


def run(receptor_pdb, ligand_pdb, smiles, md_steps, md_step_ps, temperature_k,
        snapshots, seed, wd, device) -> dict:
    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmforcefields.generators import SystemGenerator
    from openff.toolkit.topology import Topology as OFFTopology

    os.makedirs(wd, exist_ok=True)
    is_self_test = not (ligand_pdb and os.path.exists(ligand_pdb))

    # --- 1) 配体 openff 分子: 显式氢 + 构象(pose 或生成) + Gasteiger 电荷 ---
    mol = _molecule_from_smiles(smiles)
    if is_self_test:
        mol.generate_conformers(n_conformers=1)   # openff 用 RDKit ETKDG
    else:
        mol = _assign_pose_conformer(smiles, ligand_pdb)
    mol.assign_partial_charges(partial_charge_method="gasteiger")

    # --- 2) SystemGenerator(小分子按图同构匹配, 与残基名无关) ---
    gen = SystemGenerator(forcefields=["amber/ff14SB.xml"],
                          small_molecule_forcefield="openff-2.0.0",
                          molecules=[mol],
                          cache=os.path.join(wd, "ffcache"))

    # --- 3) 受体(蛋白)补氢 ---
    rec_pdb = app.PDBFile(receptor_pdb)
    rec_mod = app.Modeller(rec_pdb.topology, rec_pdb.positions)
    rec_mod.addHydrogens(gen.forcefield, pH=7.0)
    rec_top = rec_mod.topology

    # --- 4) 配体拓扑(from_molecules: 正确键序 + 原子序) + 坐标 ---
    lig_top = OFFTopology.from_molecules([mol]).to_openmm()
    lig_pos = _mol_positions_nm(mol, offset_nm=(8.0 if is_self_test else 0.0))

    # --- 5) 复合物 = 受体 + 配体 ---
    cx_mod = app.Modeller(rec_top, rec_mod.positions)
    cx_mod.add(lig_top, lig_pos)
    cx_top = cx_mod.topology

    rec_n = rec_top.getNumAtoms()
    lig_n = lig_top.getNumAtoms()
    cx_n = cx_top.getNumAtoms()
    if rec_n + lig_n != cx_n:
        return {"ok": False, "error": f"拓扑拆分不一致({rec_n}+{lig_n}!={cx_n})"}

    # --- 6) 三个体系(复合物/受体/配体, 口径一致) ---
    cx_sys = gen.create_system(cx_top, molecules=[mol])
    rec_sys = gen.create_system(rec_top, molecules=[])
    lig_sys = gen.create_system(lig_top, molecules=[mol])

    # --- 7) 短 NVT MD + 逐帧能量分解 ---
    integrator = mm.LangevinIntegrator(temperature_k * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       md_step_ps * unit.picoseconds)
    plat = _platform(device)
    plat_name = plat.getName()   # 缺 CUDA 时 _platform 已回退 CPU
    sim = app.Simulation(cx_top, cx_sys, integrator, plat)
    sim.context.setPositions(cx_mod.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)

    rec_ctx = mm.Context(rec_sys, mm.VerletIntegrator(0.001), plat)
    lig_ctx = mm.Context(lig_sys, mm.VerletIntegrator(0.001), plat)

    ddgs = []
    step_per_snap = max(1, md_steps // snapshots)
    for _ in range(snapshots):
        sim.step(step_per_snap)
        xyz = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
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
    return {"ok": True,
            "platform": plat_name,
            "deltaG_bind_kJ": mean, "deltaG_bind_std_kJ": std,
            "deltaG_bind_kcal": mean / KJ_PER_KCAL,
            "deltaG_bind_std_kcal": std / KJ_PER_KCAL,
            "md_steps": md_steps, "snapshots": snapshots,
            "n_atoms_complex": cx_n, "n_atoms_receptor": rec_n,
            "n_atoms_ligand": lig_n,
            "charge_method": "gasteiger(RDKit, 无 ambertools)",
            "self_test": is_self_test,
            "workdir": wd,
            "note": "MD 轨迹 MM-GBSA(openff-2.0.0 配体 + amber14 受体 + GBneck2, "
                    "短 NVT 轨迹平均; Gasteiger 电荷; 排序量, 非绝对自由能)"}


def worker_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--ligand", default="")
    ap.add_argument("--md_steps", type=int, default=10000)
    ap.add_argument("--md_step_ps", type=float, default=0.002)
    ap.add_argument("--temperature_k", type=float, default=300.0)
    ap.add_argument("--snapshots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260401)
    ap.add_argument("--workdir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    wd = args.workdir or tempfile.mkdtemp(prefix="mmgbsa_md_")
    os.makedirs(wd, exist_ok=True)
    try:
        result = run(args.receptor, args.ligand, args.smiles, args.md_steps,
                     args.md_step_ps, args.temperature_k, args.snapshots,
                     args.seed, wd, args.device)
    except Exception as e:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    # stdout 用 ensure_ascii=True 保证 ASCII 安全(避免 ASCII locale 下 UnicodeEncodeError);
    # 权威结果以 --out 的 result.json(UTF-8) 为准, 此行仅作人类可读/调试。
    print("MMGBSA_MD_RESULT " + json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(worker_main())
