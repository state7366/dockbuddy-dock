# -*- coding: utf-8 -*-
"""legacy_cli.py —— 拆分前的单文件命令行入口(保留以维持对外契约); 正式 CLI 请使用 dockbuddy.py。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import argparse, json, os, sys
from typing import Tuple
from .config import DockingConfig
from .report import write_report
from .run import run_docking


# ============================================================================
# 11. 命令行入口 (单篇)
# ============================================================================

def _cfg_from_args(argv) -> Tuple[DockingConfig, str]:
    # 返回 (DockingConfig, proxy) 元组 —— 标注修正(2026-10-03 第六轮审查):
    # 原标注 `-> DockingConfig` 与实际返回值不符, 误导静态检查与维护。
    ap = argparse.ArgumentParser(description="通用蛋白质-配体对接 (libdock)")
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--receptor", default="")
    ap.add_argument("--pdb_id", default="")
    ap.add_argument("--smiles", default="")
    ap.add_argument("--ligand_file", default="")
    ap.add_argument("--ligand_name", default="")
    ap.add_argument("--chain", default="")
    ap.add_argument("--box_mode", default="auto", choices=["auto", "ligand", "manual"])
    ap.add_argument("--center", nargs=3, type=float, default=None)
    ap.add_argument("--size", nargs=3, type=float, default=[12, 12, 12])
    ap.add_argument("--cat", nargs=3, default=["", "", ""],
                    help="催化三联体 亲核/His/Asp 残基, 如 170.A 248.A 216.A")
    ap.add_argument("--ref", default="", help="参考残基 如 TRP190.A")
    ap.add_argument("--engine", default="vina", choices=["vina", "vina_gpu", "ledock"])
    ap.add_argument("--vina", default="")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--num_modes", type=int, default=9)
    ap.add_argument("--outdir", default="dock_out")
    ap.add_argument("--proxy", default="")
    a = ap.parse_args(argv)
    return DockingConfig(
        id=a.id, title=a.title, receptor=a.receptor, pdb_id=a.pdb_id,
        ligand_smiles=a.smiles, ligand_file=a.ligand_file, ligand_name=a.ligand_name,
        receptor_chain=a.chain, box_mode=a.box_mode, box_center=a.center,
        box_size=tuple(a.size), catalytic=tuple(a.cat), ref_residue=a.ref,
        engine=a.engine, vina_exe=a.vina, exhaustiveness=a.exhaustiveness,
        num_modes=a.num_modes, outdir=a.outdir), a.proxy

def main(argv=None):
    cfg, proxy = _cfg_from_args(argv if argv is not None else sys.argv[1:])
    res = run_docking(cfg, proxy)
    report = os.path.join(cfg.outdir, "report.md")
    write_report(cfg, res, report)
    # 打印 JSON 摘要
    print(json.dumps(res.__dict__, ensure_ascii=False))

__all__ = [
    "_cfg_from_args",
    "main",
]
