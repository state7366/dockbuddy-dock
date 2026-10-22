# -*- coding: utf-8 -*-
"""multiconf.py —— 多构象起始对接(A3, 2026-10-07): 拆分多构象配体, 逐构象复用完整 dock 流水线。

设计约束(与 retention/conformation-entropy/protonation 一致的工程纪律):
  - 不动既有 dock 主链路: CLI 子命令 multiconf 调本模块 split_multiconformer_ligand
    拆出逐构象文件, 再对每个构象构造独立 DockingConfig 喂 cli.dock.run_pipeline
    (完整流水线: 受体预处理→配体→盒子→对接→分析→selfdock→报告), 结果挂
    conformer_index 汇总到 multiconf_summary.json。
  - 诚实纪律: 拆分失败/零构象 → 返回 ok=False + error, 不伪造; 各构象独立跑完整
    pipeline, 不声称"构象采样已穷举"(仅多起点, 非构象搜索穷举)。

输入要求: 多记录 SDF/MOL2/PDB(每记录一个构象); 单记录多构象 SDF 也按内部构象展开。
"""
import os
import statistics

from ._core import LOG, ob


def split_multiconformer_ligand(path: str, out_dir: str, ext: str = "mol2"):
    """拆多构象/多记录配体 → [(index, filepath), ...](0-based), 每构象一个坐标文件。

    支持:
      - 多记录 SDF/MOL2/PDB(每记录一个构象, 逐分子读取为独立 OBMol);
      - 单记录多构象 SDF(同一分子多 CONF, 按内部构象展开为独立 OBMol)。
    拆不出任何构象(空/解析失败)返回 []。

    几何保真: 每个拆出的分子带原始坐标, 后续 prepare_ligand_from_file 因 has_coords
    命中 asis 分支 —— 构象几何原样交接, 不被 gen3d 重建(见 ligand._build_3d_sandbox)。
    """
    if not path or not os.path.exists(path):
        LOG.error(f"多构象配体不存在: {path}")
        return []
    os.makedirs(out_dir, exist_ok=True)
    conv = ob.OBConversion()
    ext_in = os.path.splitext(path)[1].lstrip(".").lower() or "mol2"
    if not conv.SetInFormat(ext_in):
        ok = False
        for f in ("mol2", "sdf", "mol", "pdb"):
            if conv.SetInFormat(f):
                ext_in, ok = f, True
                break
        if not ok:
            LOG.error(f"无法识别配体格式: {path}")
            return []
    mol = ob.OBMol()
    # 多记录读取必须用 OpenInAndOutFiles 持续开流: ReadFile 会读一条后**关闭**流,
    # 其后 while Read 永远读不到第二条(实测多记录 SDF 只拆出 1 个构象的 bug)。
    # 输出侧喂 os.devnull(只借输入流), 循环逐分子拷贝为独立 OBMol。
    mols = []
    if not conv.OpenInAndOutFiles(path, os.devnull):
        LOG.error(f"OpenBabel 无法打开配体: {path}")
        return []
    try:
        while conv.Read(mol):
            mols.append(ob.OBMol(mol))
    finally:
        for cm in ("CloseInFile", "CloseOutFile"):
            if hasattr(conv, cm):
                getattr(conv, cm)()
    if not mols:
        LOG.error(f"OpenBabel 未从配体读出任何分子: {path}")
        return []
    # 若仅 1 个分子但含内部多构象, 展开各构象为独立 OBMol
    if len(mols) == 1 and mols[0].NumConformers() > 1:
        try:
            base = mols[0]
            n = base.NumConformers()
            expanded = []
            for ci in range(n):
                nm = ob.OBMol(base)
                if n > 1:
                    nm.DeleteConformers([j for j in range(n) if j != ci])
                expanded.append(nm)
            if expanded:
                mols = expanded
        except Exception as e:  # noqa: BLE001 - 内部构象展开失败不致命, 降级到首构象
            LOG.warning(f"内部多构象展开失败, 仅用首构象: {e}")
            mols = [mols[0]]
    w = ob.OBConversion()
    if not w.SetOutFormat(ext):
        ext = "mol2"
        w.SetOutFormat("mol2")
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for i, m in enumerate(mols):
        op = os.path.join(out_dir, f"{stem}_conf{i + 1}.{ext}")
        if w.WriteFile(m, op) and os.path.exists(op):
            out.append((i, op))
        else:
            LOG.warning(f"构象 {i + 1} 写出失败, 跳过")
    if out:
        LOG.info(f"多构象配体 {os.path.basename(path)} 拆出 {len(out)} 个构象")
    else:
        LOG.error(f"多构象配体 {os.path.basename(path)} 拆出 0 个构象(文件可能为空/无坐标)")
    return out


# 每个构象汇总进 multiconf_summary 时挑选的 DockResult 字段(避免 ~40 字段全量搬运,
# 与 docking_summary 的字段口径对齐, 便于下游复用判读)。
_CONF_FIELDS = (
    "ok", "energy_kcal_mol", "reliability", "n_poses", "nuc_to_ligO_A",
    "ref_to_ligAromC_A", "pocket_contact_frac", "ligand_buried",
    "energy_reliable", "selfdock_available", "selfdock_ok",
    "box_assumption", "top_pose_rmsd_A", "pose_set_rmsd_mean_A", "notes",
)


def assemble_multiconf_summary(run_id, receptor, ligand_file, params, per_conf):
    """汇总逐构象结果 → multiconf_summary 字典(结构见 field_docs 'multiconf_summary')。

    per_conf: list of (index, dock_result_dict) —— dock_result_dict 来自
      DockResult.__dict__ 复制(已含 conformer_index)。
    """
    confs = []
    for idx, d in per_conf:
        rec = {"conformer_index": idx}
        for k in _CONF_FIELDS:
            rec[k] = d.get(k)
        confs.append(rec)
    # 候选 best: 先取 ok + energy_reliable 且能量有值者; 否则放宽到 ok 且有能量
    cand = [c for c in confs if c.get("ok") and c.get("energy_reliable")
            and c.get("energy_kcal_mol") is not None]
    if not cand:
        cand = [c for c in confs if c.get("ok")
                and c.get("energy_kcal_mol") is not None]
    best = None
    if cand:
        b = min(cand, key=lambda c: c["energy_kcal_mol"])
        best = {"conformer_index": b["conformer_index"],
                "energy_kcal_mol": b["energy_kcal_mol"],
                "nuc_to_ligO_A": b.get("nuc_to_ligO_A")}
    energies = [c["energy_kcal_mol"] for c in confs
                if c.get("energy_kcal_mol") is not None]
    summary = {
        "ok": any(c.get("ok") for c in confs),
        "multiconf_id": run_id,
        "receptor": os.path.basename(receptor) if receptor else "",
        "ligand_file": os.path.basename(ligand_file) if ligand_file else "",
        "n_conformers": len(confs),
        "n_ok": sum(1 for c in confs if c.get("ok")),
        "n_failed": sum(1 for c in confs if not c.get("ok")),
        "params": params or {},
        "conformers": confs,
        "best": best,
        "energy_min": (min(energies) if energies else None),
        "energy_max": (max(energies) if energies else None),
        "energy_mean": (round(statistics.fmean(energies), 3) if energies else None),
        "energy_std": (round(statistics.pstdev(energies), 3)
                       if len(energies) > 1 else None),
        "note": ("多构象起始对接: 每个构象独立跑完整 dock 流水线, 取最低能量构象为 best。"
                 "仅多起点采样, 非构象穷举; 结论是否依赖起始构象由 conformers[].energy_kcal_mol "
                 "的散布反映(energy_std 大=起始构象敏感)。"),
        "error": None,
    }
    return summary


__all__ = ["split_multiconformer_ligand", "assemble_multiconf_summary",
           "_CONF_FIELDS"]
