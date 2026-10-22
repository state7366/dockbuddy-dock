# -*- coding: utf-8 -*-
"""report.py —— 产物落盘: 失败摘要、单篇报告与 PyMOL 可视化脚本生成。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import json, os, re
from typing import List, Optional
from ._core import LOG
from .config import DockResult, DockingConfig


def write_fail_summary(path: str, notes: List[str], **extra):
    """在早退(对接引擎失败/分析失败)前写入一份 fresh 失败 summary, 覆盖任何
    历史成功产物, 避免旧 docking_summary.json 残留误导下游汇总。"""
    try:
        data = {"ok": False, "notes": list(notes)}
        for k, v in extra.items():
            data.setdefault(k, v)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        LOG.warning(f"写入失败 summary 异常: {e}")

# ============================================================================
# 8.6 PyMOL 可视化脚本生成 (2026-09-29)
#   每次对接分析完成后在产物目录写出 view.pml: 受体卡通灰、催化三联体红棍、
#   口袋残基橙、选中 pose 绿棍、其余 pose 淡蓝线; 日志同步打印速查命令。
#   设计为零依赖(纯文本生成), 失败仅 WARNING, 不影响主流程。
# ============================================================================

def _resi_list(keys) -> str:
    """['SER101A','HIS203A'] -> '101+203' (PyMOL resi 选择表达式)。"""
    out = []
    for k in keys or []:
        m = re.search(r"(\d+)", str(k))
        if m:
            out.append(m.group(1))
    return "+".join(out)

def write_pymol_script(outdir: str, res, catalytic_keys=None,
                       run_id: str = "") -> Optional[str]:
    """生成 view.pml 并打印 PyMOL 速查命令。返回脚本路径(失败 None)。"""
    try:
        rec = os.path.join(outdir, "receptor.pdbqt")
        poses = os.path.join(outdir, "results.pdbqt")
        if not (os.path.exists(rec) and os.path.exists(poses)):
            return None
        triad_resi = _resi_list(catalytic_keys or getattr(res, "catalytic_keys", None)
                                or getattr(res, "catalytic", None))
        pocket_resi = _resi_list(getattr(res, "pocket_residues", None))
        sel_pose = int(getattr(res, "selected_pose_index", 1) or 1)
        crystal = os.path.join(outdir, "crystal_ligand.pdbqt")
        has_crystal = os.path.exists(crystal)
        # 绝对路径(正斜杠): PyMOL 的 load 相对其自身 cwd 解析, 不随脚本位置
        rec_u = rec.replace(os.sep, "/")
        poses_u = poses.replace(os.sep, "/")
        crystal_u = crystal.replace(os.sep, "/")

        lines = [
            "# view.pml —— DockBuddy 自动生成, PyMOL 中 File > Run Script 执行",
            "reinitialize",
            "bg_color white",
            f"load {rec_u}, rec",
            f"load {poses_u}, poses",
        ]
        if has_crystal:
            lines.append(f"load {crystal_u}, crystal")
        lines += [
            "hide everything",
            "show cartoon, rec",
            "color gray80, rec",
            "show sticks, poses",
            "set stick_radius, 0.15",
            "# ---- 全部 pose 淡蓝 ----",
            "color lightblue, poses",
            "",
        ]
        if triad_resi:
            lines += [
                "# ---- 催化三联体: 红棍 + 标签 ----",
                f"select triad, rec and resi {triad_resi}",
                "show sticks, triad",
                "color tv_red, triad",
                "label triad and name CA, '%s%s' % (resn, resi)",
                "",
            ]
        if pocket_resi:
            lines += [
                "# ---- 口袋残基(<4.5A): 橙色 ----",
                f"select pocket, rec and resi {pocket_resi}" + (f" and not resi {triad_resi}" if triad_resi else ""),
                "show sticks, pocket",
                "color tv_orange, pocket",
                "",
            ]
        lines += [
            f"# ---- 选中最优 pose#{sel_pose}: 绿棍 ----",
            f"select best_pose, poses and state {sel_pose}",
            "color tv_green, best_pose",
            "set stick_radius, 0.22, best_pose",
        ]
        if has_crystal:
            lines += [
                "# ---- 共晶配体(地面真值): 紫 ----",
                "show sticks, crystal",
                "color violet, crystal",
            ]
        lines += [
            "",
            "zoom poses",
            "set ray_shadows, 0",
            "# 出图: png fig.png, dpi=300, ray=1",
        ]
        pml = os.path.join(outdir, "view.pml")
        with open(pml, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        LOG.info(f"🎨 PyMOL 脚本: {pml}  (PyMOL> run {os.path.basename(pml)})")
        LOG.info(f"   速查: select triad, resi {triad_resi or '?'} | "
                 f"select pocket, resi {pocket_resi or '?'} | "
                 f"select best_pose, poses and state {sel_pose}")
        return pml
    except Exception as e:
        LOG.warning(f"生成 PyMOL 脚本失败(不影响主流程): {e}")
        return None

def _resname_at(pdb_path: str, resnum: Optional[int], chain: Optional[str]) -> str:
    """在 PDB 中查某残基号的残基名(用于手动催化残基推断 resname)。"""
    if resnum is None:
        return "XXX"
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            ri = ln[22:26].strip()
            ch = ln[21].strip()
            if ri == str(resnum) and (chain is None or ch == chain):
                return ln[17:20].strip()
    return "XXX"

# ============================================================================
# 10. 报告生成
# ============================================================================

def write_report(cfg: DockingConfig, res: DockResult, path: str) -> None:
    """写出单篇论文的 Markdown 验证报告。"""
    lines = []
    lines.append(f"# 对接验证: {cfg.id}\n")
    lines.append(f"- **论文**: {cfg.title}")
    lines.append(f"- **PDB**: {cfg.pdb_id or cfg.receptor}")
    lines.append(f"- **配体**: {cfg.ligand_name} (`{cfg.ligand_smiles or cfg.ligand_file}`)")
    lines.append(f"- **引擎**: {cfg.engine}  穷举度={cfg.exhaustiveness}  模式数={cfg.num_modes}")
    lines.append(f"- **受体链**: {res.receptor_chain or '(auto)'}")
    lines.append(f"- **催化三联体(自动/手动)**: {', '.join(res.catalytic) or '(未检出)'}")
    lines.append(f"- **盒子中心**: {tuple(round(c, 3) for c in res.box_center) if res.box_center else '-'}  尺寸={tuple(cfg.box_size)}")
    lines.append("")
    lines.append("## 结果\n")
    if res.ok:
        lines.append(f"- **最优 pose 能量**: {res.energy_kcal_mol} kcal/mol")
        lines.append(f"- **催化亲核原子 → 配体最近氧**: {res.nuc_to_ligO_A:.2f} Å" if res.nuc_to_ligO_A is not None else "- 催化亲核距离: 未计算")
        if res.ref_to_ligAromC_A is not None:
            lines.append(f"- **参考残基 {cfg.ref_residue} → 配体芳香碳**: {res.ref_to_ligAromC_A:.2f} Å")
        if res.arom_pocket_dist_A is not None:
            lines.append(f"- **配体芳香环 → 受体最近芳香残基 {res.arom_pocket_residue}**: {res.arom_pocket_dist_A:.2f} Å")
        lines.append(f"- **口袋残基(<4.5Å, {len(res.pocket_residues)} 个)**: {', '.join(res.pocket_residues)}")
    else:
        lines.append("- ❌ 对接未成功")
    if res.notes:
        lines.append("\n## 备注\n")
        for n in res.notes:
            lines.append(f"- {n}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    LOG.info(f"✅ 报告已写出: {path}")

__all__ = [
    "_resi_list",
    "_resname_at",
    "write_fail_summary",
    "write_pymol_script",
    "write_report",
]
