# -*- coding: utf-8 -*-
"""run.py —— 编排层: 把受体/配体/盒子/对接/分析串成一次完整对接(run_docking)。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import math, os
from ._core import LOG
from .config import DockResult, DockingConfig
from .geom import parse_residue_token, rotate_pdbqt_file, rotation_align_to_z
from .pdbio import _count_pdb_atoms, _ensure_single_model, download_pdb
from .triad import detect_catalytic_triad, pick_receptor_chain
from .ligand import _native_ligand_info, prepare_ligand, prepare_ligand_from_file
from .receptor import prepare_receptor
from .backends import LeDockBackend, VinaBackend, VinaGPUBackend
from .poses import analyze_top_pose, parse_pdbqt_poses
from .box import auto_box_center
from .report import _resname_at, write_fail_summary


# ============================================================================
# 9. 主流程编排
# ============================================================================

def run_docking(cfg: DockingConfig, proxy: str = "") -> DockResult:
    """[DEPRECATED] 旧的单体编排入口, 仅被本模块 __main__ 遗留使用。

    优化项H: 请改用解耦编排 `cli.dock.run_pipeline`(或 `api.run_docking`),
    二者走 02~06 解耦阶段、带前置检查/审计日志/可靠性研判/柔性配体等全部加固。
    本函数保留仅为向后兼容, 不再维护, 未来版本将移除。"""
    import warnings
    warnings.warn("libdock.run_docking 已废弃, 请改用 cli.dock.run_pipeline / api.run_docking",
                  DeprecationWarning, stacklevel=2)
    LOG.info(f"===== 开始对接: {cfg.id} | {cfg.title} =====")
    os.makedirs(cfg.outdir, exist_ok=True)
    receptor_pdbqt = os.path.join(cfg.outdir, "receptor.pdbqt")
    ligand_pdbqt = os.path.join(cfg.outdir, "ligand.pdbqt")
    results_pdbqt = os.path.join(cfg.outdir, "results.pdbqt")
    summary_json = os.path.join(cfg.outdir, "docking_summary.json")

    # --- 受体 ---
    rp = cfg.receptor
    if not rp and cfg.pdb_id:
        rp = os.path.join(cfg.outdir, f"{cfg.pdb_id.upper()}.pdb")
    # 健壮性: 若已有受体文件无效(如早前下载到 404 HTML 页面), 且可重新下载,
    #   先删除再走下载流程, 避免无效文件被当成合法受体。
    # 2026-10-03: 只允许清理"自家下载产物"(outdir / raw_pdb 缓存内);
    #   用户显式提供的本地源文件绝不删除(破坏性操作不得作用于用户输入)。
    # 来源判定(2026-10-03 第六轮审查 D7): 原判定只看**位置**不看**来源** ——
    #   用户显式 --receptor 指向 outdir 内的文件(如复用上次输出的受体)且无效
    #   且给了 pdb_id 时会被误删。cfg.receptor 非空 = 用户显式提供, 无效也不删;
    #   为空 = rp 由 pdb_id 派生, 属自家产物可清理(缓存同理, 可重下载自愈)。
    if rp and os.path.exists(rp) and cfg.pdb_id and _count_pdb_atoms(rp) < 10:
        _user_provided = bool((cfg.receptor or "").strip())
        try:
            from dockbuddy_dock.paths import RAW_PDB_CACHE as _CACHE
            _cache_dir = os.path.abspath(_CACHE)
        except Exception:
            _cache_dir = None
        _rp_abs = os.path.abspath(rp)
        _ours = ((not _user_provided)
                 or (_cache_dir is not None and _rp_abs.startswith(_cache_dir + os.sep)))
        if _ours:
            LOG.warning(f"已有受体文件无效(疑似坏下载产物), 删除后重新下载: {rp}")
            try:
                os.remove(rp)
            except OSError:
                pass
        else:
            LOG.error(f"受体文件无有效 ATOM 坐标, 且为用户显式指定路径(不自动删除): {rp}")
    if rp and cfg.pdb_id and not os.path.exists(rp):
        if not download_pdb(cfg.pdb_id, rp, proxy):
            return DockResult(notes=[f"无法下载 PDB {cfg.pdb_id}"])
    if not rp or not os.path.exists(rp):
        return DockResult(notes=["未提供受体 PDB 且无 pdb_id"])

    # --- 受体健壮性预处理(批量暴露的共性问题) ---
    # (a) 无效 PDB 检测: 下载到错误页(如 404 HTML)会导致下游静默失败; 此处显式拦截。
    if _count_pdb_atoms(rp) < 10:
        LOG.error(f"受体 PDB 无有效 ATOM 坐标(可能下载失败/ID 无效): {rp}")
        return DockResult(notes=[f"受体 PDB 无有效坐标({cfg.pdb_id or rp})"])
    # (b) 多模型(NMR)缩减: 只保留首个模型, 防止 obabel 对海量原子卡死。
    if _ensure_single_model(rp):
        LOG.info(f"受体 PDB 已缩减为首模型(NMR/多模型): {rp}")

    # --- 先确定链与催化三联体(必须在受体预处理之前, 保证二者链一致) ---
    triad = None
    catalytic_keys = []
    chain = cfg.receptor_chain
    # token 可解析性守卫(2026-10-03 第六轮审查 S6): parse_residue_token 失败返回
    # (None, None), 第二遍 f"{nm}{rn}{rc or chain}" 会拼出 "XXXNoneA" 垃圾 key
    # —— 催化距离分析静默失效(poses 的 nuc_keys 永不命中, summary 无任何痕迹)。
    # box.py 盒心侧有同款 rn 守卫, 催化侧此前没有 —— 现对齐: 坏 token 即警告
    # 并回退自动检测。
    _cat_toks = ([parse_residue_token(t) for t in cfg.catalytic]
                 if all(cfg.catalytic) else [])
    if _cat_toks and all(rn is not None for rn, _rc in _cat_toks):
        # 手动指定: 由 token 推断链(若无则回退 receptor_chain/最长链)。
        # 2026-10-03: 补链必须在查名**之前**完成(两遍式) —— 原实现单遍循环内
        # `rc or chain` 在"token 无链号 + receptor_chain 为空"时以空串查名,
        # _resname_at 只匹配无链号的行, 蛋白实际在链 A → 恒返回 "XXX",
        # catalytic_keys 全成 ["XXX170", ...], 催化距离分析静默失效。
        for rn, rc in _cat_toks:
            if rc:
                chain = rc
        if not chain:
            chain = pick_receptor_chain(rp)
        for (rn, rc) in _cat_toks:
            nm = _resname_at(rp, rn, rc or chain)
            catalytic_keys.append(f"{nm}{rn}{rc or chain}")
        LOG.info(f"使用手动催化残基: {catalytic_keys} (链 {chain})")
    else:
        if _cat_toks:
            LOG.warning(f"催化残基 token 无法解析(需 '残基名+编号', 如 SER101): "
                        f"{list(cfg.catalytic)} —— 回退自动检测")
        triad = detect_catalytic_triad(rp, chain_filter=chain)
        if triad:
            chain = triad["nuc"]["chain"] or chain
            catalytic_keys = [
                f"{triad['nuc']['resname']}{triad['nuc']['resnum']}{triad['nuc']['chain']}",
                f"{triad['his']['resname']}{triad['his']['resnum']}{triad['his']['chain']}",
                f"{triad['asp']['resname']}{triad['asp']['resnum']}{triad['asp']['chain']}",
            ]
        else:
            # 无催化三联体: 选最长链
            if not chain:
                chain = pick_receptor_chain(rp)
            LOG.warning(f"未检测到催化三联体, 受体链={chain}, 催化距离分析将跳过")

    ok, chain = prepare_receptor(rp, receptor_pdbqt, chain)
    if not ok:
        return DockResult(notes=["受体预处理失败"])

    # --- 配体 ---
    arom = []
    if cfg.ligand_smiles:
        ok, arom = prepare_ligand(cfg.ligand_smiles, ligand_pdbqt)
    elif cfg.ligand_file:
        ok, arom = prepare_ligand_from_file(cfg.ligand_file, ligand_pdbqt)
    else:
        return DockResult(notes=["未提供配体(SMILES 或文件)"])
    if not ok:
        return DockResult(notes=["配体生成失败"])

    # --- 盒子中心 + 定向(沿隧道轴旋转拉伸) ---
    center, box_assumption, _box_note = auto_box_center(rp, cfg, triad)
    if center is None:
        return DockResult(notes=["无法确定盒子中心"])

    R = None
    size = tuple(cfg.box_size)
    # ligand 模式: 用共晶配体的空间范围自适应盒尺寸(比固定 box_size 更贴合口袋)
    if cfg.box_mode == "ligand":
        _, lig_size = _native_ligand_info(rp)
        if lig_size:
            size = lig_size
            LOG.info(f"ligand 模式盒子尺寸(按共晶配体范围): {tuple(round(s,1) for s in size)}")
    # 定向仅在 ligand_mode 且结构内确有有机结合配体时启用: 此时结合配体质心给出
    # "真实隧道轴" = (结合配体质心 -> 催化三联体 Cα 质心), 沿该轴拉伸盒子才能可靠
    # 复现 TPA-嵌入-芳香隧道 姿势。auto 模式无真实轴, 用对称立方体(让配体环落入
    # 芳香口袋更自然), 避免过度假设隧道方向。
    if cfg.box_orient and cfg.box_mode == "ligand":
        lc, _ = _native_ligand_info(rp)
        if lc and triad:
            # 隧道轴方向: 结合配体质心 - 催化三联体 Cα 质心(亲核残基一侧)
            anchor = triad.get("ca_centroid")
            if anchor:
                axis = (lc[0] - anchor[0], lc[1] - anchor[1], lc[2] - anchor[2])
                if math.sqrt(sum(x * x for x in axis)) > 1.0:
                    R = rotation_align_to_z(axis)
                    alen = math.sqrt(sum(x * x for x in axis))
                    # 防御: 隧道轴过长时(如 51A)若不做上限, 盒子会被拉成 60A+ 滑道,
                    #   配体滑到远端远离催化残基。盒长上限 30A, 保证配体始终贴近催化中心。
                    tl = cfg.tube_len if cfg.tube_len > 0 else (alen + 10.0)
                    tl = min(tl, 30.0)
                    size = (cfg.tube_cross, cfg.tube_cross, tl)
                    LOG.info(f"盒子定向: 沿隧道轴(长 {alen:.1f}A, 盒长上限30A)对齐 Z, 尺寸 {size}")

    # 受体/配体坐标旋转到定向坐标系(Vina 在旋转系内运行; 中心在旋转轴上故不变)
    if R:
        rotate_pdbqt_file(receptor_pdbqt, R, center)
        rotate_pdbqt_file(ligand_pdbqt, R, center)

    # --- 对接 ---
    # 2026-10-04: engine 三分发(vina_gpu=Vina-GPU 2.1 OpenCL, 筛选层用;
    # 参数模型不同: thread/search_depth, 无 seed —— 见 backends.VinaGPUBackend)
    if cfg.engine == "vina":
        backend = VinaBackend(cfg.resolve_vina())
    elif cfg.engine == "vina_gpu":
        backend = VinaGPUBackend(cfg.resolve_vina_gpu())
    else:
        # 2026-10-04 全盘审计: 原误把 vina 的 exe 当 ledock_exe 传入(错误接线)。
        backend = LeDockBackend(cfg.resolve_ledock())
    try:
        # seed 透传(2026-10-03 漏网点): 本地 VinaBackend 调用此前丢 seed
        # → libdock 路径不可复现。
        if not backend.dock(receptor_pdbqt, ligand_pdbqt, center, size,
                           results_pdbqt, cfg.num_modes, cfg.exhaustiveness,
                           seed=cfg.seed):
            write_fail_summary(summary_json, ["对接引擎运行失败"])
            return DockResult(notes=["对接引擎运行失败"])

        # --- 分析 (旋转系内, 距离不变) ---
        poses = parse_pdbqt_poses(results_pdbqt)
        LOG.info(f"共解析 {len(poses)} 个 pose")
        res = analyze_top_pose(receptor_pdbqt, poses, catalytic_keys,
                               cfg.ref_residue, arom, summary_json,
                               box_center=center, receptor_chain=chain,
                               box_assumption=box_assumption,
                               engine=getattr(cfg, "engine", "vina"))
        if res is None:
            write_fail_summary(summary_json, ["最优 pose 分析失败"])
            return DockResult(notes=["最优 pose 分析失败"])
    finally:
        # 旋回原坐标系交付(成功/失败一律; 2026-10-03): 原实现失败早退不逆旋,
        # 产物留在旋转系, 重跑/下游叠合坐标系整体错位。
        if R:
            rotate_pdbqt_file(receptor_pdbqt, R, center, inverse=True)
            rotate_pdbqt_file(ligand_pdbqt, R, center, inverse=True)
            if os.path.exists(results_pdbqt):
                rotate_pdbqt_file(results_pdbqt, R, center, inverse=True)
    res.box_center = tuple(center)
    res.receptor_chain = chain
    return res

__all__ = [
    "run_docking",
]
