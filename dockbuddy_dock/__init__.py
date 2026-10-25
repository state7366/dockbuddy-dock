# -*- coding: utf-8 -*-
"""dockbuddy_dock —— 分子对接引擎(从 DockBuddy libdock 抽出的解耦版本)。

含通用对接(引擎/盒/配体/受体/pose/几何/质子化/多构象/重打分) + 对接盒决策所需的
三联体/口袋几何/催化取向 + 隐式溶剂 + 报告。

向后兼容(契约等价): 拆分前 libdock.* 命名空间可见的顶层符号在此 re-export,
使 `import dockbuddy_dock` 后仍能以 `dockbuddy_dock.<符号>` / `.<子模块>.<符号>`
访问(主仓库 tests 沿用 libdock fixture = dockbuddy_dock, 零改动)。
"""
from .config import DockResult, DockingConfig
from .backends import DockingBackend, VinaBackend, VinaGPUBackend, LeDockBackend
from .tools import find_obabel, find_vina, find_vina_gpu, fix_babel_datadir
from .reliability import reliability_thresholds, classify_reliability_detail

# ---- 子模块 re-export(契约等价: libdock 拆分前的顶层可见名) ----
from . import (poses, receptor, ligparam, gnina, caver, config, figures,
               ligand, box, pdbio, pocketgeom, comparative, af3data, triad,
               geom, backends, reliability, tools, report)

from .ligand import (_is_solvent_resname, _native_ligand_resname,
                     _native_ligand_info, _ligand_geom_fragmented,
                     extract_named_ligand, resolve_crystal_ligand,
                     prepare_ligand, prepare_ligand_from_file)
from .pdbio import _ligand_unsupported_elements
from .triad import detect_catalytic_triad
from .poses import analyze_top_pose, _kabsch_rmsd_by_name
from .pocketgeom import (free_volume_sphere, pocket_volume_qc,
                         residue_environment, residue_min_distance,
                         _parse_pdb_heavy)
from .comparative import _summarize_smiles_runs, smiles_comparative_dock
from .af3data import (cif_entry_id, af3_confidence_summary, af3_pae,
                      af3_pae_pairs, list_af3_models, load_af3_full_data)
from .receptor import prepare_receptor
from .gnina import _find_gnina
from .caver import _find_java

__version__ = "0.1.0"

__all__ = ["DockingConfig", "DockResult", "DockingBackend", "VinaBackend",
           "VinaGPUBackend", "LeDockBackend", "find_obabel", "find_vina",
           "find_vina_gpu", "fix_babel_datadir", "reliability_thresholds",
           "classify_reliability_detail", "__version__"]
