# -*- coding: utf-8 -*-
"""selfdock.py —— 自对接校验: 把共晶配体重新对接回受体, 与晶体姿态比对 RMSD。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。

M1(2026-10-02): 判据从"最佳 pose RMSD < 2 Å"扩展为双字段 ——
  - rmsd_min_A / best_pose_index: 9 个 mode 里**最接近晶体**的 pose(采样覆盖能力);
  - rmsd_top1_A / reproducible_top1: Vina 按 energy 排名的 **top1** pose 的 RMSD
    (打分排序能力)。只看最佳 pose 会系统性高估协议可靠性 —— "能找到"与
    "排在第一"是两个命题, 报告两者必须分开披露(与 tools/oneoff/verify_fixed.py 的
    top-pose=PASS / best-pose=SKIP 三态判据对齐)。
"""
import os
from ._core import LOG, _TYPE2ELEM, ad_type_of
from .geom import rotate_pdbqt_file
from .ligand import prepare_ligand_from_file, resolve_crystal_ligand
from .backends import VinaBackend
from .poses import _kabsch_rmsd_by_name


def self_dock_check(raw_pdb: str, receptor_pdbqt: str, center, size, R,
                    ligand_name: str = "", flexible: bool = True, ph: float = 7.4,
                    seed: int = 42, exhaustiveness: int = 8, num_modes: int = 9):
    """晶体配体重对接自校验 (self-docking)。
    取受体 PDB 中的晶体配体(按 ligand_name 或自动探测), 用与主对接相同的盒子/框心重新对接,
    再把各 pose 与晶体坐标对齐算 Kabsch 重原子 RMSD, 评估"协议能否复现晶体姿态"。

    约定: receptor_pdbqt 已处于旋转后的坐标系(若 R 被应用); 故晶体配体坐标也须绕 center 施加 R
          变换到同一坐标系后再比较(由调用方传 R; 无定向时 R=None 则原样)。

    返回字段(2026-10-02 起):
      rmsd_min_A / best_pose_index —— 全部 mode 中最接近晶体的(采样覆盖);
      rmsd_top1_A / reproducible_top1 —— energy 排名 top1 的(打分排序);
      reproducible —— min < 2.0(兼容旧字段, 语义=采样可覆盖)。
    """
    import numpy as np
    import tempfile

    # 晶体配体统一解析(方案 B, 2026-10-02): 按名精确提取(ligand_name, 用户意图
    # 优先)→ 失败回退自动挑最大非溶剂 HETATM 组。同一分子的 CCD 残基名常与
    # manifest 俗称不同(同一分子的 CCD 名常不同, 如 MHET 实为 C9C), 原先精确匹配失败即整体跳过。
    # selfdock 对晶体分子本身重对接, 天然自洽, 无跨分子守卫需求(那是 pose 排序
    # 路径 analyze.is_self_dock 的职责)。
    tmp_pdb, resname = resolve_crystal_ligand(raw_pdb, ligand_name)
    if not tmp_pdb:
        if not resname:
            LOG.info("[self-dock] 未检测到晶体配体, 跳过自对接校验")
        else:
            # 2026-10-03 修复: 指名/回退都提取失败却配体名已知 —— 完全静默会让下游
            # 无法区分"无共晶配体"与"提取环节出错"
            LOG.warning(f"[self-dock] 晶体配体 {resname} 提取失败(按名与回退均未成功), "
                        f"跳过自对接校验 —— 与'无共晶配体'不同, 属提取环节问题")
        return None
    fd, lig_pdbqt = tempfile.mkstemp(prefix="selfdock_lig_", suffix=".pdbqt")
    os.close(fd)
    try:
        ok, _ = prepare_ligand_from_file(tmp_pdb, lig_pdbqt, flexible=flexible, ph=ph)
        if not ok:
            LOG.warning(f"[self-dock] 晶体配体 {resname} 3D 制备失败, 跳过自对接校验")
            return None
        if R:
            rotate_pdbqt_file(lig_pdbqt, R, center)

        # 基准: 已制备晶体配体(单构象), 原子名与对接输出一致, 可直接按名匹配
        crystal = {}
        with open(lig_pdbqt, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith(("ATOM", "HETATM")) and (lambda t: _TYPE2ELEM.get(t, t[:1]).upper())(ad_type_of(ln)) != "H":
                    an = ln[12:16].strip()
                    if an in crystal:  # C5: 重复原子名显式告警, 不静默覆盖
                        LOG.warning(f"[self-dock] 重复原子名 {an!r}, 后者覆盖前者 —— RMSD 可能失真")
                    crystal[an] = np.array([float(ln[30:38]), float(ln[38:46]),
                                            float(ln[46:54])], float)

        fd2, out_pdbqt = tempfile.mkstemp(prefix="selfdock_out_", suffix=".pdbqt")
        os.close(fd2)
        try:
            from dockbuddy_dock.tools import find_vina
            backend = VinaBackend(find_vina())
            ok_dock = backend.dock(receptor_pdbqt, lig_pdbqt, center, size, out_pdbqt,
                                  num_modes=num_modes, exhaustiveness=exhaustiveness, seed=seed)
            if not ok_dock:
                # 柔性制备偶发 ROOT/解析失败(芳香键 kekulize 畸形柔性树); 退回刚性重跑一次
                LOG.warning("[self-dock] 柔性自对接失败, 退回刚性配体重试")
                prepare_ligand_from_file(tmp_pdb, lig_pdbqt, flexible=False, ph=ph)
                if R:
                    rotate_pdbqt_file(lig_pdbqt, R, center)
                ok_dock = backend.dock(receptor_pdbqt, lig_pdbqt, center, size, out_pdbqt,
                                       num_modes=num_modes, exhaustiveness=exhaustiveness, seed=seed)
            if not ok_dock:
                LOG.warning("[self-dock] 自对接 Vina 运行失败")
                return None

            # 解析所有 pose(保持文件顺序 = Vina energy 升序), 对每个 pose 按原子名
            # 匹配算 Kabsch RMSD(C8: 经 poses._kabsch_rmsd_by_name → geom.kabsch_rmsd
            # 单一实现; 同源配体 100% 名字匹配, 缺名即 inf —— C4 口径①)。
            models = []
            cur = {}
            with open(out_pdbqt, encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.startswith("MODEL"):
                        if cur:
                            models.append(cur); cur = {}
                        continue
                    if ln.startswith("ENDMDL"):
                        if cur:
                            models.append(cur); cur = {}
                        continue
                    # 2026-10-02: 判氢取类型码首字符(0-based 76:77)。原 ln[77:78] 取的是
                    # 第 2 个字符, 极化氢 "HD" 会被判成"非氢"而混进重原子集合。
                    if ln.startswith(("ATOM", "HETATM")) and (lambda t: _TYPE2ELEM.get(t, t[:1]).upper())(ad_type_of(ln)) != "H":
                        an = ln[12:16].strip()
                        if an in cur:
                            LOG.warning(f"[self-dock] pose 内重复原子名 {an!r}, 后者覆盖前者")
                        cur[an] = np.array([float(ln[30:38]), float(ln[38:46]),
                                            float(ln[46:54])], float)

            if not models:
                return None
            rmsds = [_kabsch_rmsd_by_name(crystal,
                                          [(v[0], v[1], v[2], "", k, "", "") for k, v in m.items()])
                     for m in models]
            # inf 净化(2026-10-03 修复): 名字不匹配的 pose 返回 inf, round(inf,3)=inf
            # 会被 json.dump 写成非标准 Infinity(严格解析器/JS 看板出错)。
            # inf 置 None + 显式告警, 下游(report 的 selfdock_rmsd_A)已有 None 口径。
            import math as _math
            n_unmatched = sum(1 for r in rmsds if not _math.isfinite(r))
            if n_unmatched:
                LOG.warning(f"[self-dock] {n_unmatched}/{len(rmsds)} 个 pose 与晶体基准"
                            f"原子名无法匹配(RMSD=inf), 对应值置 None")
            finite = [r for r in rmsds if _math.isfinite(r)]
            if not finite:
                LOG.warning(f"[self-dock] 晶体配体 {resname}: 全部 pose 与基准原子名无法"
                            f"匹配, RMSD 不可用(available=True, 数值 None)")
                return {
                    "available": True,
                    "resname": resname,
                    "rmsd_min_A": None,
                    "best_pose_index": None,
                    "rmsd_top1_A": None,
                    "reproducible": False,
                    "reproducible_top1": False,
                    "n_poses": len(models),
                    "n_unmatched": n_unmatched,
                    "note": "全部 pose 与晶体基准原子名无法匹配, RMSD 不可用",
                }
            # 2026-10-03 回归修复: best_pose_index 必须是**原始 rmsds 列表**的下标。
            # 此前 argmin(finite) 用的是 inf 过滤后列表的下标, inf 夹在中间时
            # 序号系统性指错 pose(如 [3.2, inf, 1.5, inf] 报 2, 实为第 3 个)。
            finite_idx = [i for i, r in enumerate(rmsds) if _math.isfinite(r)]
            best_idx = min(finite_idx, key=lambda i: rmsds[i])
            min_rmsd = float(rmsds[best_idx])
            top_rmsd = (float(rmsds[0]) if _math.isfinite(rmsds[0]) else None)
            ok_repro = min_rmsd < 2.0
            ok_top = top_rmsd is not None and top_rmsd < 2.0
            LOG.info(f"[self-dock] 晶体配体 {resname}: "
                     f"top1 RMSD={('%.2f' % top_rmsd) if top_rmsd is not None else 'N/A'}A"
                     f"({'✅可复现' if ok_top else '⚠️排序未达'}) | "
                     f"最佳 pose#{best_idx + 1} RMSD={min_rmsd:.2f}A"
                     f"({'✅' if ok_repro else '⚠️'}) (阈值 2.0A)")
            return {
                "available": True,
                "resname": resname,
                "rmsd_min_A": round(min_rmsd, 3),
                "best_pose_index": best_idx + 1,
                "rmsd_top1_A": round(top_rmsd, 3) if top_rmsd is not None else None,
                "reproducible": ok_repro,
                "reproducible_top1": ok_top,
                "n_poses": len(models),
                "n_unmatched": n_unmatched,
            }
        finally:
            if os.path.exists(out_pdbqt):
                os.remove(out_pdbqt)
    finally:
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)
        if os.path.exists(lig_pdbqt):
            os.remove(lig_pdbqt)

__all__ = [
    "self_dock_check",
]
