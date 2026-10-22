# -*- coding: utf-8 -*-
"""config.py —— 配置与结果数据结构(DockingConfig / DockResult)——流水线唯一的契约载体。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# 1. 配置数据结构 (全参数化, 无硬编码酶)
# ============================================================================

@dataclass
class DockingConfig:
    """一篇论文/一个体系的对接配置。所有字段均可选, 缺省走自动推断。"""
    id: str                                   # 唯一标识(用于文件夹/产物名)
    title: str = ""                           # 论文标题(仅记录)
    receptor: str = ""                        # 受体 PDB 本地路径
    pdb_id: str = ""                          # 若 receptor 为空, 则据此从 RCSB 下载
    ligand_smiles: str = ""                   # 配体 SMILES
    ligand_file: str = ""                     # 或配体坐标文件(.mol/.mol2/.pdb)
    ligand_name: str = ""                     # 共晶配体残基名(按名从受体提取配体/报告展示);
                                              #   空 = 未指定。历史默认 "ligand" 会被当作真实残基名
                                              #   去受体里找("未找到名为 LIGAND" 警告)并污染 ligand: 自动 tag
    receptor_chain: str = ""                  # "" => 自动选择
    box_mode: str = "auto"                    # auto | ligand | manual
    box_center: Optional[Tuple[float, float, float]] = None
    box_size: Tuple[float, float, float] = (12.0, 12.0, 12.0)
    box_orient: bool = True                     # 沿隧道轴旋转拉伸盒子(核心泛化优化)
    tube_cross: float = 12.0                    # 定向后横截面尺寸
    tube_len: float = 0.0                       # 定向后隧道长度(0=按轴长自动+10)
    # 催化三联体: ("170.A","248.A","216.A") = (亲核, His, Asp/Glu); 空串 => 自动检测
    catalytic: Tuple[str, str, str] = ("", "", "")
    ref_residue: str = ""                     # 参考残基(如 "TRP190.A"), 用于配体定位验证
    engine: str = "vina"                      # vina | vina_gpu | ledock
    vina_exe: str = ""                        # vina 可执行路径(默认 tools/vina.exe)
    vina_gpu_exe: str = ""                    # Vina-GPU+ 2.0 exe(与 Kernel*.bin 同目录;
                                              # 默认 tools/vina-gpu-2.0/ 下探测, DOCK_VINA_GPU_EXE 覆盖)
    exhaustiveness: int = 8
    num_modes: int = 9
    outdir: str = "dock_out"
    detect_aromatic: bool = True              # 配体芳香碳检测(用于 ref 残基距离)
    seed: Optional[int] = None                # Vina 随机种子; None=用默认(run_pipeline 解析为固定值,
                                              # 保证可复现/可审计), 可用 --seed/DOCK_SEED/api seed 覆盖
    ligand_flexible: bool = True              # 配体柔性: True=保留 BRANCH/TORSDOF 采样内部构象(默认);
                                              #   False=旧的刚性 TORSDOF 0 兜底(仅作回归对照/逃生开关)
    ligand_ph: float = 7.4                    # 配体质子化 pH; <=0 时不做 pH 校正(保持原始质子化态)
    keep_metals: bool = False                 # 受体预处理保留金属辅因子/催化离子(优化项G, 金属酶用)
    keep_resnames: Tuple[str, ...] = ()       # 额外保留的 HETATM 残基名(如 ("FMN","FAD","HEM"))
    tags: Tuple[str, ...] = ()                # 实验分组标签(如 ("wt","baseline")); 写入 summary 供 batch --tag 过滤与报告
    per_chain: bool = True                    # 多链受体是否逐链展开为独立对接(manifest 可写 "per_chain": false 关闭;
                                              # CLI --chain 当场指定链时也自动关闭)

    def resolve_vina(self) -> str:
        if self.vina_exe:
            return self.vina_exe
        # 2026-10-04 全盘审计: 原缺省指向 pipeline/libdock/tools/vina.exe(不存在);
        # 捆绑 exe 在仓库根 tools/。
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        return os.path.join(repo_root, "tools", "vina.exe")

    def resolve_ledock(self) -> str:
        """LeDock exe 解析: DOCK_LEDOCK_EXE 环境变量(无捆绑, 未安装即不可用)。"""
        return os.environ.get("DOCK_LEDOCK_EXE", "")

    def resolve_vina_gpu(self) -> str:
        """Vina-GPU exe 解析: 显式字段 > DOCK_VINA_GPU_EXE > ""(由调用方 find_vina_gpu 兜底)。"""
        if self.vina_gpu_exe:
            return self.vina_gpu_exe
        return os.environ.get("DOCK_VINA_GPU_EXE", "")

@dataclass
class DockResult:
    ok: bool = False
    energy_kcal_mol: Optional[float] = None
    box_center: Optional[Tuple[float, float, float]] = None
    receptor_chain: str = ""
    catalytic: List[str] = field(default_factory=list)   # ["SER170A","HIS248A","ASP216A"]
    nuc_to_ligO_A: Optional[float] = None        # 亲核原子 -> 配体最近酯氧
    ref_to_ligAromC_A: Optional[float] = None    # 参考残基 -> 配体最近芳香碳
    arom_pocket_dist_A: Optional[float] = None   # 配体芳香环 -> 受体最近芳香残基(通用隧道指标)
    arom_pocket_residue: str = ""                # 最近受体芳香残基
    pocket_residues: List[str] = field(default_factory=list)
    n_poses: int = 0
    notes: List[str] = field(default_factory=list)
    # --- 健壮性增强字段(2026-08-16 加固) ---
    has_nucleophile: bool = False    # 催化核亲核体是否为真实亲核残基(Ser/Cys/Thr)
    nuc_resname: str = ""            # 核亲核体残基名(如 SER/ALA), 用于判定
    ligand_clashes: int = 0          # 配体重原子与蛋白重原子最近距 <2.8A 的原子数
    ligand_buried_frac: float = 0.0  # 配体中"被蛋白包夹"(最近距<2.5A)的原子比例
    ligand_med_near: Optional[float] = None  # 配体各原子最近蛋白距离的中位数
    ligand_buried: bool = False      # 配体是否埋入蛋白内部( clash>=3 且 buried_frac>=0.15 )
    # --- 新一轮健壮性增强(2026-08-16 续): pose 可靠性量化 ---
    pocket_contact_frac: float = 0.0      # 配体重原子中处于蛋白(<4.5A)接触的比例(0=漂浮, 1=全包埋)
    ligand_centroid_offset_A: Optional[float] = None  # 配体质心到盒子中心距离(诊断用, 不作分类阈值)
    ligand_n_heavy: int = 0               # 配体重原子数(>40 视为超出 Vina 稳健尺度)
    energy_reliable: bool = True          # |能量|<=50 且非超大/异常配体, 才计入能量排序
    # --- 通用泛化加固(2026-08-16): 结果可靠性与盒子假设 ---
    box_assumption: str = ""              # 盒子中心来源假设: manual|native_ligand|triad|protein_centroid|none
    reliability: str = "ok"               # 结果可靠性: ok|warn|unreliable (泛化手动调用时直接看这个)
    warnings: List[str] = field(default_factory=list)  # 可靠性预警汇总(供报告/日志直接展示)
    top_pose_rmsd_A: Optional[float] = None  # top1 与 top2 pose 重原子 RMSD(Å); 大=对接未收敛
    # --- 全 pose 自散(2026-10-07): 结合模式收敛性量化(与 top1~top2 发散互补) ---
    pose_set_rmsd_mean_A: Optional[float] = None    # 全 pose 两两 RMSD 的均值(Å)
    pose_set_rmsd_median_A: Optional[float] = None  # 全 pose 两两 RMSD 的中位数(Å); >pose_scatter_warn -> 结合模式未收敛
    pose_set_rmsd_max_A: Optional[float] = None     # 全 pose 两两 RMSD 的最大值(Å)
    pose_set_rmsd_n_pairs: int = 0                  # 实际参与计算的两两对数(= n_poses*(n_poses-1)/2, 排除不可比)
    # --- warn 二分(2026-08-19): 真问题(real) vs 保守提醒(soft) ---
    warn_class: str = ""                # "" / "real"(真问题) / "soft"(保守提醒, 仅收敛性)
    warn_reasons: List[str] = field(default_factory=list)  # 触发分类的具体原因键(real_*/pose_div)
    # --- 自对接校验(2026-08-20): 晶体配体重对接复现晶体姿态 ---
    selfdock_rmsd_A: Optional[float] = None     # 最佳 pose 与晶体配体重原子 Kabsch RMSD
    selfdock_pose_idx: Optional[int] = None     # 最接近晶体的 pose 序号(1-based)
    selfdock_ok: bool = False                   # RMSD<2.0A 视为协议可复现
    selfdock_available: bool = False            # 是否存在晶体配体可校验(否则未运行)
    # --- 自对接双判据(2026-10-02, M1): 采样覆盖 vs 打分排序分开披露 ---
    selfdock_top_rmsd_A: Optional[float] = None  # energy 排名 top1 pose 的晶体 RMSD
    selfdock_repro_top1: Optional[bool] = None   # top1 pose 即达 2.0A(协议排序正确)
    schema_version: int = 1               # docking_summary.json 结构版本(优化项H: 供下游兼容判断)
    # --- 多构象后处理(2026-08-21): 保留并排序所有 poses ---
    poses: List[Dict[str, Any]] = field(default_factory=list)  # 前N个 pose 的轻量指标(排序后)
    sort_strategy: str = ""            # "energy" | "rmsd_to_crystal" | "hybrid"
    selected_pose_index: int = 1       # 被选为最优 pose 的原始序号(1-based)
    crystal_rmsd_available: bool = False  # 是否有晶体配体用于 pose-vs-crystal RMSD
    is_self_dock: bool = False         # 是否 self-docking(对接配体==共晶配体, RMSD 才有物理意义)
    # --- 多构象对接(2026-10-07, A3): 逐构象跑完整 dock, 标记来源构象索引 ---
    conformer_index: Optional[int] = None  # 该结果来自多构象配体的第几个构象(0-based); 单构象对接为 None

__all__ = [
    "DockResult",
    "DockingConfig",
]
