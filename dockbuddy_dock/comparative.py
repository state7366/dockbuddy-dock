# -*- coding: utf-8 -*-
"""comparative.py —— 跨结构对照对接(GT 晶体 vs 本地模型), 用于模型可信度评估。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import math
import os, shutil, tempfile
from typing import Any, Dict, List, Optional
from ._core import LOG, env_float


def _fin3(x):
    """inf → None(2026-10-03): kabsch 原子名不匹配返回 inf,
    round(inf,3) 会把非标准 JSON Infinity 写进产物(严格解析器/JS 报错),
    且 inf 参与 `<= 2.0` 判定属侥幸。与 selfdock 已修口径对齐。"""
    import numpy as np
    x = float(x)
    return round(x, 3) if np.isfinite(x) else None


from .pdbio import _ligand_unsupported_elements
from .ligand import (_native_ligand_resname, extract_named_ligand, prepare_ligand,
                     prepare_ligand_from_file)
from .receptor import prepare_receptor
from .backends import VinaBackend
from .poses import (_analyze_pose_light, _kabsch_rmsd_by_name,
                    _pdbqt_ligand_coords, load_receptor_aromatic_carbons,
                    parse_pdbqt_poses)
from .triad import detect_catalytic_triad
# 催化取向(2026-10-05): 「底物有没有被摆成可攻击姿态」——能量给不了的判据。
# 口径实现全在 orientation.py(单一事实源), 本文件只做编排与落盘。
from .orientation import (find_scissile_sites, pose_orientation,
                          receptor_anchors, summarize_side,
                          orientation_sides, NUC_TO_SCISSILE_C_MAX,
                          HIS_TO_LEAVING_O_MAX)

# 盒心与该受体自身三联体 Cα 质心的容忍距离(Å); 超过即判"跨坐标系/盒子错位"。
# 正常同源对照 <1 Å; 实测跨 AF 模型可直接飙到 25~28 Å。
CROSS_FRAME_WARN_A: float = env_float("DOCK_CROSS_FRAME_WARN", 5.0)


def _make_docker(vina_exe: str):
    """标准对接执行器工厂(2026-10-17): 移除 RemoteRunner 远程派发, 恒本地 VinaBackend。

    返回 dock_fn, 签名兼容 VinaBackend.dock(receptor, ligand, center, size, out,
    num_modes, exhaustiveness, seed, cpu), 另收 run_id(本地忽略, 兼容旧调用)。
    """
    backend = VinaBackend(vina_exe)

    def _local_dock(rec, lig, center, size, out, num_modes=9,
                    exhaustiveness=8, seed=None, cpu=None, run_id="eval"):
        return backend.dock(rec, lig, center, size, out,
                            num_modes, exhaustiveness, seed=seed, cpu=cpu)
    return _local_dock


def cocrystal_comparative_dock(ref_pdb: str, alt_pdb: str, chain: str = "",
                               ligand_resname: str = "", seed: int = 42,
                               exhaustiveness: int = 8, num_modes: int = 9,
                               ph: float = 7.4) -> Optional[Dict[str, Any]]:
    """共晶配体双边对接比较: 同一配体/盒子/种子分别对接进 ref 与 alt 两个受体。

    两种用途(语义仅差在 ref/alt 指代):
      - mutate 有效性评估: ref=WT 受体(含共晶配体), alt=突变体模型
        → ΔE(mutant-WT) 即突变对结合能的影响; ref 侧同时是自对接 sanity。
      - 对接级 GT 验证: ref=数据库突变体晶体(含共晶配体), alt=本地建模突变体
        → 检验建模误差是否传导到对接结论。

    配体从 ref_pdb 提取(显式 ligand_resname 或自动探测最大有机 HETATM);
    盒子中心=晶体配体 centroid, 尺寸=配体跨度+6 限幅 [22,34]Å。
    判定: |ΔE|<=0.5 kcal/mol 且 top-pose RMSD(alt vs ref)<=2.0Å → equivalent。
    返回 None 表示无法评估(无共晶配体/vina 缺失/对接失败), 否则 dict。"""
    import numpy as np
    import shutil
    import tempfile

    resname = (ligand_resname or "").strip().upper() or _native_ligand_resname(ref_pdb)
    if not resname:
        LOG.info("[eval-dock] %s 无共晶配体, 跳过对接评估", os.path.basename(ref_pdb))
        return None
    # 催化取向需要亲核残基位置; 共晶通道的盒心来自晶体配体, 与这里解耦,
    # 所以检不出三联体时**只丢取向、不丢对接**(不能因为取向失效就整条 GT 评估没了)。
    triad = detect_catalytic_triad(ref_pdb, chain_filter=chain) or \
        detect_catalytic_triad(alt_pdb, chain_filter=chain)
    try:
        from dockbuddy_dock.tools import find_vina
        vina = find_vina("")
    except Exception as e:
        LOG.warning("[eval-dock] 定位 vina 失败, 跳过对接评估: %s", e)
        return None

    _dock_fn = _make_docker(vina)

    workdir = tempfile.mkdtemp(prefix="evaldock_")
    lig_tmps: List[str] = []

    class _SkipEval(Exception):
        """不应重试的确定性跳过(如配体过小)。"""

    try:
        def _attempt(lig_tmp: str):
            """配体制备 + 双边对接一轮; 失败返回 None(可重试)。"""
            lig_pdbqt = os.path.join(workdir, "lig.pdbqt")
            ok, _ = prepare_ligand_from_file(lig_tmp, lig_pdbqt, flexible=True, ph=ph)
            if not ok:
                # 柔性制备偶发失败(芳香键 kekulize 畸形柔性树) → 退回刚性
                LOG.warning("[eval-dock] 柔性配体制备失败, 退回刚性重试")
                ok, _ = prepare_ligand_from_file(lig_tmp, lig_pdbqt, flexible=False, ph=ph)
                if not ok:
                    return None
            crystal = _pdbqt_ligand_coords(lig_pdbqt)
            if not crystal:
                return None
            if len(crystal) < 5:
                # 羟胺/水合分子级"配体"(如 AHOH 2 原子): 对接比较无统计意义
                raise _SkipEval(
                    f"配体 {resname} 仅 {len(crystal)} 重原子(<5), 对接评估无意义")
            pts = np.array(list(crystal.values()), float)
            center = pts.mean(0)
            span = pts.max(0) - pts.min(0)
            size = tuple(float(min(34.0, max(22.0, float(s) + 6.0))) for s in span)

            def _dock_one(tag: str, pdb: str) -> Optional[Dict[str, Any]]:
                rec_pdbqt = os.path.join(workdir, f"{tag}_rec.pdbqt")
                ok_r, used_chain = prepare_receptor(pdb, rec_pdbqt, receptor_chain=chain)
                if not ok_r:
                    return None
                out_pdbqt = os.path.join(workdir, f"{tag}_out.pdbqt")
                # 对照评估链路恒用 Vina CPU + cpu=1(2026-10-04 GPU 接入后维持不变):
                # 论文口径的可复现配对(ΔE/锚定)必须单一引擎单一环境; Vina-GPU 仅作
                # 主 dock 流程(--engine vina_gpu)的筛选层, 不进本链路。
                # 2026-10-10: 经 _make_docker 派发 —— 远程/本地同一代码路径,
                # run_id 用 tag(ref/alt)保证远端工作目录唯一。
                if not _dock_fn(rec_pdbqt, lig_pdbqt, center, size, out_pdbqt,
                                num_modes=num_modes, exhaustiveness=exhaustiveness,
                                seed=seed, cpu=1, run_id=tag):
                    return None
                poses = parse_pdbqt_poses(out_pdbqt)
                if not poses:
                    return None
                # None 能量防护(2026-10-03 第六轮审查 C3): 缺 REMARK VINA RESULT 的
                # MODEL 能量为 None 哨兵(同 commit 的 poses.py 修复) —— 全部 pose 缺
                # REMARK(截断/外部重组文件)时 poses[0][0] 为 None, 下游 float(None)
                # 与 d_e 算术都会 TypeError。此路径不可信, 视同该边对接失败跳过。
                if poses[0][0] is None:
                    LOG.warning("[eval-dock] %s 侧 results 无能量标注"
                                "(缺 REMARK VINA RESULT, 疑截断/外部文件), 跳过该边", tag)
                    return None
                rmsds = [_kabsch_rmsd_by_name(crystal, atoms) for _, atoms in poses]
                if not any(np.isfinite(r) for r in rmsds):
                    LOG.warning("[eval-dock] pose 与晶体基准原子名全部无法匹配(RMSD=inf)")
                    return None
                best = int(np.argmin(rmsds))
                # ---- 催化取向: 全 pose 统计(2026-10-05) ----
                # 与 --eval_smiles 通道同口径(见 orientation.py), 使共晶 GT 通道与
                # 外源对照通道可互相印证 —— 共晶通道自带晶体基准, 取向占比在这里是
                # 「对接找到催化姿态的能力」, 而非「突变改变取向」。
                try:
                    _sites = find_scissile_sites(poses[0][1]) if poses[0][1] else []
                    _anch = receptor_anchors(rec_pdbqt, triad)
                    _rows = []
                    for _e, _at in poses:
                        if _e is None:
                            continue
                        _o = pose_orientation(_at, _sites,
                                              _anch["nuc_pts"], _anch["his_pts"])
                        _o["energy"] = _e
                        _rows.append(_o)
                    _or = summarize_side(_rows)
                    _or["sites_total"] = len(_sites)
                    _or["sites_acid"] = sum(1 for s in _sites if s["acid"])
                except Exception as e:  # 附加指标, 失败不得拖垮 GT 对照评估
                    LOG.warning("[eval-dock] %s 侧催化取向统计失败(%s: %s)",
                                tag, type(e).__name__, e)
                    _or = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                return {"chain_used": used_chain,
                        "orientation": _or,
                        "top_energy": round(float(poses[0][0]), 2),
                        "top_rmsd_vs_crystal_A": _fin3(rmsds[0]),
                        "best_rmsd_vs_crystal_A": _fin3(rmsds[best]),
                        "best_pose_index": best + 1,
                        "n_poses": len(poses),
                        "_top_atoms": poses[0][1]}

            LOG.info("[eval-dock] 共晶配体 %s 双边对接 (盒心=%s, 尺寸=%s, seed=%d)",
                     resname, np.round(center, 2).tolist(),
                     [round(s, 1) for s in size], seed)
            ref = _dock_one("ref", ref_pdb)
            alt = _dock_one("alt", alt_pdb)
            if not ref or not alt:
                LOG.warning("[eval-dock] 双边对接未完成(ref=%s, alt=%s)",
                            "ok" if ref else "fail", "ok" if alt else "fail")
                return None
            return center, size, ref, alt

        # 2026-10-04: 提取必须带 info —— 连通碎片过滤可能把完整配体切掉大半
        # (实测 1MOO 4MZ: altLoc 并集 12 原子被判成 [6,6] 两个碎片, 只留 6)。
        # 拿截断配体做双边对接, ref/alt 两侧用的是同一个碎片, RMSD 与 ΔE 都会
        # "完美一致" —— 一个 0.00 Å 的假等效结论, 比不做还危险。
        _ex = extract_named_ligand(ref_pdb, resname, return_info=True)
        lig_tmp, lig_info = (_ex if isinstance(_ex, tuple) else (_ex, {}))
        if not lig_tmp:
            return None
        lig_tmps.append(lig_tmp)
        bad_el = _ligand_unsupported_elements(lig_tmp)
        if bad_el:
            LOG.info("[eval-dock] 配体 %s 含 Vina 不支持的元素 %s"
                     "(如硼酸/共价抑制剂), 跳过对接评估", resname, "/".join(bad_el))
            return None
        try:
            got = _attempt(lig_tmp)
        except _SkipEval as e:
            LOG.info("[eval-dock] %s, 跳过", e)
            return None
        if got is None:
            # 同名残基多个分离副本 → OB 写多 ROOT pdbqt → Vina 拒收;
            # 退化为最大单实例配体重试(成键多残基配体如 MAN 链首轮即可成功, 不走这里)
            LOG.warning("[eval-dock] 首轮失败, 退化: 仅取最大单实例配体重试")
            _ex = extract_named_ligand(ref_pdb, resname, single_instance=True,
                                       return_info=True)
            lig_tmp, lig_info = (_ex if isinstance(_ex, tuple) else (_ex, dict(lig_info)))
            if not lig_tmp:
                return None
            lig_tmps.append(lig_tmp)
            try:
                got = _attempt(lig_tmp)
            except _SkipEval as e:
                LOG.info("[eval-dock] %s, 跳过", e)
                return None
            if got is None:
                return None
        center, size, ref, alt = got

        # ---- 催化取向: 共晶通道的单种子判词(2026-10-05) ----
        # 与 --eval_smiles 通道同口径; 这里的 oriented_frac 读作「对接复现催化
        # 复合物的能力」, 不是突变效应 —— 共晶通道 ref/alt 是同一套配体制备,
        # 两侧取向占比差才有突变含义。
        if triad:
            _or_wt = ref.get("orientation") or {}
            _or_mut = alt.get("orientation") or {}
            out_orientation = {
                "per_side": {"wt": _or_wt, "mut": _or_mut},
                "verdict": orientation_sides(_or_wt or None, _or_mut or None),
                "thresholds": {
                    "nuc_to_scissile_C_max_A": NUC_TO_SCISSILE_C_MAX,
                    "his_to_leaving_O_max_A": HIS_TO_LEAVING_O_MAX},
            }
        else:
            out_orientation = {"per_side": {}, "verdict": None,
                               "error": "未检出催化三联体, 取向判据跳过"}

        d_e = round(alt["top_energy"] - ref["top_energy"], 2)
        ref_top = {a[4]: np.array(a[:3], float) for a in ref.pop("_top_atoms")}
        pose_rmsd = _fin3(_kabsch_rmsd_by_name(ref_top, alt.pop("_top_atoms")))
        _p_ok = pose_rmsd is not None and pose_rmsd <= 2.0
        equiv = (abs(d_e) <= 0.5) and _p_ok
        e_part = ("能量无显著差异" if abs(d_e) <= 0.5
                  else ("结合增强" if d_e < 0 else "结合减弱") + f"({d_e:+.2f})")
        p_part = ("pose 等效" if _p_ok
                  else (f"pose 偏移({pose_rmsd:.2f}A)" if pose_rmsd is not None
                        else "pose RMSD 不可判(原子名不匹配)"))
        verdict = f"{e_part}, {p_part}" + (" → 对接层面等效" if equiv else "")
        out = {"available": True, "ligand_resname": resname,
               "box_center": [round(float(c), 3) for c in center],
               "box_size": [round(float(s), 2) for s in size],
               "seed": seed, "exhaustiveness": exhaustiveness,
               "ref": ref, "alt": alt, "orientation": out_orientation,
               "delta_E_alt_minus_ref": d_e,
               "top_pose_rmsd_alt_vs_ref_A": pose_rmsd,
               "equivalent": equiv, "verdict": verdict}
        # 配体被连通碎片真正截断(残基组被劈开) → 结论不可信, 但**不静默改判
        # equivalent**(改判会把"不可信"伪装成"不等效", 同样是错误的断言)。用独立
        # 字段表达, 由下游(validate_dock_sensitivity / mutate --eval_dock)决定处置。
        # 注意(2026-10-04 v2): 多副本配体取一个完整副本是**合法**的单实例选择,
        # 不在此列 —— 实测 FTA(1PZP, 两份完整副本)与 4MZ(1MOO, FORMUL 2(C4H6N2))
        # 都是合法案例, 初版守卫曾把它们误判成截断。
        if lig_info.get("fragment_truncated"):
            out["ligand_truncated"] = True
            out["trustworthy"] = False
            out["unreliable_reason"] = (
                f"配体 {resname} 提取时有残基组被连通碎片劈开: "
                f"{lig_info.get('n_atoms')}/{lig_info.get('n_before_fragment_filter')} 原子"
                f"(真截断), 对接比较的 ref/alt 两侧用的是同一个碎片, "
                f"ΔE 与 pose RMSD 都会虚假地'一致'")
            out["verdict"] += " ⚠ 配体被截断, 结论不可信"
            LOG.warning("[eval-dock] %s", out["unreliable_reason"])
        else:
            out["ligand_truncated"] = False
            out["trustworthy"] = True
        return out
    finally:
        for _t in lig_tmps:
            if os.path.exists(_t):
                os.remove(_t)
        shutil.rmtree(workdir, ignore_errors=True)

def _summarize_smiles_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多种子对照汇总(2026-10-03): ΔE mean±std / 锚定迁移发生率与分布。

    输入 = smiles_comparative_dock 的 runs(逐种子 {seed, ref, alt,
    delta_E_alt_minus_ref, arom_anchor_changed}); 独立成函数供单测直接调用。"""
    import statistics as _st
    d_es = [r["delta_E_alt_minus_ref"] for r in runs]
    n_ch = sum(1 for r in runs if r["arom_anchor_changed"])
    anchor_pairs: Dict[str, int] = {}
    for r in runs:
        k = f"{r['ref'].get('arom_pocket_residue') or '(无)'}→" \
           f"{r['alt'].get('arom_pocket_residue') or '(无)'}"
        anchor_pairs[k] = anchor_pairs.get(k, 0) + 1
    d_mean = round(_st.mean(d_es), 3)
    d_std = round(_st.pstdev(d_es), 3) if len(d_es) > 1 else 0.0
    e_tag = ("噪声内" if abs(d_mean) <= 0.5 else
             ("突变体更优" if d_mean < 0 else "突变体更弱"))
    verdict = (f"ΔE {d_mean:+.2f}±{d_std:.2f} kcal/mol({e_tag}); "
               f"芳香锚定迁移 {n_ch}/{len(runs)}"
               + (f"({','.join(k for k, v in anchor_pairs.items() if '→' in k and k.split('→')[0] != k.split('→')[1])})"
                  if n_ch else ""))
    def _mean_or_none(vals):
        vals = [v for v in vals if v is not None]
        return round(_st.mean(vals), 3) if vals else None

    return {
        "n_seeds": len(runs),
        "delta_E_mean": d_mean, "delta_E_std": d_std,
        "delta_E_min": min(d_es), "delta_E_max": max(d_es),
        "ref_energy_mean": _mean_or_none(
            [r["ref"]["energy_kcal_mol"] for r in runs]),
        "alt_energy_mean": _mean_or_none(
            [r["alt"]["energy_kcal_mol"] for r in runs]),
        "nuc_to_ligO_ref_mean": _mean_or_none(
            [r["ref"].get("nuc_to_ligO_A") for r in runs]),
        "nuc_to_ligO_alt_mean": _mean_or_none(
            [r["alt"].get("nuc_to_ligO_A") for r in runs]),
        "anchor_changed_frac": round(n_ch / len(runs), 3),
        "anchor_pairs": anchor_pairs,
        "verdict": verdict,
    }


def _summarize_orientation_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """催化取向跨种子汇总(2026-10-05)。

    输入 = smiles_comparative_dock 的 runs(逐种子 ref/alt.orientation)。
    输出只**追加**字段, 不改动既有 summary 的任何键 —— 既有下游
    (report / tests)按名字取值, 加键不得引起漂移。
    """
    import statistics as _st

    def _mm(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"mean": None, "std": 0.0, "n": 0}
        return {"mean": round(_st.mean(vals), 3),
                "std": round(_st.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                "n": len(vals)}

    f_ref = _mm([r["ref"].get("orientation", {}).get("oriented_frac") for r in runs])
    f_alt = _mm([r["alt"].get("orientation", {}).get("oriented_frac") for r in runs])
    r_ref = _mm([r["ref"].get("orientation", {}).get("relay_frac") for r in runs])
    r_alt = _mm([r["alt"].get("orientation", {}).get("relay_frac") for r in runs])
    d_fr = [round((r["alt"].get("orientation", {}).get("oriented_frac") or 0)
                  - (r["ref"].get("orientation", {}).get("oriented_frac") or 0), 3)
            for r in runs]
    near_ref = _mm([r["ref"].get("orientation", {}).get("nearest_mean_A")
                    for r in runs])
    near_alt = _mm([r["alt"].get("orientation", {}).get("nearest_mean_A")
                    for r in runs])
    n_site = None
    for r in runs:
        o = r["ref"].get("orientation") or {}
        if o.get("sites_total") is not None:
            n_site = o["sites_total"]
            break
    d_mm = _mm(d_fr)
    return {
        "oriented_frac_ref": f_ref, "oriented_frac_alt": f_alt,
        "oriented_frac_delta": d_mm,
        "n_pose_ref": sum(r["ref"].get("orientation", {}).get("n_pose") or 0 for r in runs),
        "n_pose_alt": sum(r["alt"].get("orientation", {}).get("n_pose") or 0 for r in runs),
        "n_oriented_ref": sum(r["ref"].get("orientation", {}).get("n_oriented") or 0 for r in runs),
        "n_oriented_alt": sum(r["alt"].get("orientation", {}).get("n_oriented") or 0 for r in runs),
        "relay_frac_ref": r_ref, "relay_frac_alt": r_alt,
        "nearest_C_to_nuc_ref_A": near_ref, "nearest_C_to_nuc_alt_A": near_alt,
        "n_scissile_sites": n_site,
        "nuc_max_A": NUC_TO_SCISSILE_C_MAX,
        "his_max_A": HIS_TO_LEAVING_O_MAX,
        "per_seed_oriented_frac": {
            str(r["seed"]): {"wt": r["ref"].get("orientation", {}).get("oriented_frac"),
                             "mut": r["alt"].get("orientation", {}).get("oriented_frac")}
            for r in runs},
    }


def smiles_comparative_dock(ref_pdb: str, alt_pdb: str, smiles: str, chain: str = "",
                            seeds=(42,), exhaustiveness: int = 8,
                            num_modes: int = 9, ph: float = 7.4,
                            box_size: float = 24.0,
                            ligand_file: str = "", rigid: bool = False) -> Optional[Dict[str, Any]]:
    """SMILES 多种子双边对照对接: 同一配体/盒子, 逐种子分别对接 ref(WT) 与 alt(突变体)。

    适用: 受体**无共晶配体**(如 AlphaFold 模型)时 mutate --eval_smiles 的对照评估;
    有共晶配体应优先 cocrystal_comparative_dock(ref 侧兼自对接 sanity, 属 GT 语义)。

    与 cocrystal 版的三点差异:
      - 配体来自 SMILES(prepare_ligand, gen3d 沙箱 + 内容寻址缓存), 非晶体提取;
        或由 ligand_file 提供**固定 3D 构象**(prepare_ligand_from_file, 跳过 SMILES
        gen3d —— 跨环境大底物交叉对比时消除 obabel gen3d 版本差异, 见 2026-10-10 修正方案);
      - 盒子 = ref 侧三联体 Cα 质心, 尺寸缺省 24³(两侧共用同一坐标 —— ΔE 的可比性
        关键; 检不出三联体则返回 None, 不猜测盒心);
      - 无晶体基准 → 不做等效判定, 每侧输出 top energy / 亲核-酯氧距离 /
        芳香锚定残基(+距离) / 接触率, 供人工判读。

    多种子(2026-10-03): seeds 为 int 列表(缺省 [42]); 配体制备一次共享, 逐种子
    双边对接 → runs(逐种子明细) + summary(ΔE mean±std / 锚定迁移发生率/分布)。
    可复现性: Vina 钉 cpu=1(多线程下种群交换时序受调度影响, 固定 seed 也不逐位复现)。
    rigid(2026-10-10): True 时配体按刚性对接(TORSDOF 0, 减自由度, 大底物欠收敛时用)。
    返回 None 表示无法评估(三联体缺失/配体失败/全部种子对接失败)。"""
    import numpy as np

    smiles = (smiles or "").strip()
    if not smiles and not ligand_file:
        return None
    # 种子去重保序(缺省/空 → [42])
    _seeds = [int(s) for s in (seeds or [42])]
    _seen = set()
    seeds = [s for s in _seeds if not (s in _seen or _seen.add(s))]
    triad = detect_catalytic_triad(ref_pdb, chain_filter=chain)
    if not triad:
        LOG.info("[eval-smiles] ref 侧未检出催化三联体, 盒心不可定位, 跳过")
        return None
    center = np.asarray(triad["ca_centroid"], float)
    # box_size 兼容单值(立方)或三元组(2026-10-10: 大底物可传非立方盒)
    if isinstance(box_size, (tuple, list)):
        size = tuple(float(s) for s in box_size)
    else:
        size = (float(box_size), float(box_size), float(box_size))
    try:
        from dockbuddy_dock.tools import find_vina
        vina = find_vina("")
    except Exception as e:
        LOG.warning("[eval-smiles] 定位 vina 失败, 跳过对照: %s", e)
        return None

    _dock_fn = _make_docker(vina)

    workdir = tempfile.mkdtemp(prefix="evalsmi_")
    try:
        lig_pdbqt = os.path.join(workdir, "lig.pdbqt")
        flexible = not rigid
        if ligand_file:
            # 固定构象输入(2026-10-10 修正方案): 跨环境复现大底物时跳过 SMILES
            # gen3d, 用同一份坐标文件消除 obabel 版本差异; 该文件不重新 3D 构建。
            ok, arom_names = prepare_ligand_from_file(
                ligand_file, lig_pdbqt, flexible=flexible, ph=ph)
            if not ok:
                LOG.error("[eval-smiles] 固定构象配体文件制备失败: %s", ligand_file)
                return None
        else:
            ok, arom_names = prepare_ligand(smiles, lig_pdbqt, flexible=flexible, ph=ph)
            if not ok and flexible:
                LOG.warning("[eval-smiles] 柔性配体制备失败, 退回刚性重试")
                ok, arom_names = prepare_ligand(smiles, lig_pdbqt, flexible=False, ph=ph)
            if not ok:
                return None

        def _dock_one(tag: str, pdb: str, seed: int) -> Optional[Dict[str, Any]]:
            rec_pdbqt = os.path.join(workdir, f"{tag}_rec.pdbqt")
            ok_r, used_chain = prepare_receptor(pdb, rec_pdbqt, receptor_chain=chain)
            if not ok_r:
                return None
            # 跨坐标系守卫(2026-10-05 实测): 本通道两侧**共用 ref 的盒心**以保证
            # ΔE 可比, 但同一批 AF 模型并不在同一坐标系(Kabsch 叠合后 Cα RMSD
            # 仅 0.07~0.11 Å, 原始坐标却差 9.4~25.0 Å)。拿 ref 的盒心去对接 alt,
            # Vina 会在 20 Å 外照常返回一个"最优解" —— 静默产出一个假对照。
            # 这里把"盒心与该受体自身三联体质心对不上"显式叫出来。
            try:
                _tri_here = detect_catalytic_triad(pdb, chain_filter=chain)
                if _tri_here:
                    _gap = float(math.dist(_tri_here["ca_centroid"], center))
                    if _gap > CROSS_FRAME_WARN_A:
                        LOG.warning(
                            "[eval-smiles] %s 侧盒心(%s)距该受体自身三联体 Cα 质心 %s Å"
                            " > %s Å —— 疑似受体不在同一坐标系(拿 ref 的盒心去对接 alt)。"
                            "该侧对接结果与另一侧**不能直接相减**: 先做 Kabsch 变换"
                            "(geom.kabsch_transform + apply_rigid)把 alt 对齐到 ref, "
                            "或两侧各自算自己的三联体盒心并独立报告",
                            tag, [round(float(c), 2) for c in center],
                            round(_gap, 2), CROSS_FRAME_WARN_A)
            except Exception as e:
                LOG.debug("[eval-smiles] 跨坐标系检测跳过(%s: %s)", type(e).__name__, e)
            out_pdbqt = os.path.join(workdir, f"{tag}_out.pdbqt")
            # 2026-10-10: 经 _make_docker 派发 —— 远程/本地同一代码路径,
            # run_id 用 tag(ref/alt)保证远端工作目录唯一。
            if not _dock_fn(rec_pdbqt, lig_pdbqt, center, size, out_pdbqt,
                            num_modes=num_modes, exhaustiveness=exhaustiveness,
                            seed=seed, cpu=1, run_id=tag):
                return None
            poses = parse_pdbqt_poses(out_pdbqt)
            if not poses or poses[0][0] is None:
                LOG.warning("[eval-smiles] %s 侧无 pose 或无能量标注, 跳过该边", tag)
                return None
            catalytic = [f"{triad[k]['resname']}{triad[k]['resnum']}{triad[k]['chain']}"
                         for k in ("nuc", "his", "asp")]
            m = _analyze_pose_light(rec_pdbqt, poses[0][1], float(poses[0][0]),
                                    catalytic, "", arom_names or [])
            # 芳香锚定残基(与 poses.py _analyze_single_pose_full 同口径):
            # 配体芳香碳 → 最近受体芳香碳 → 所属残基
            arom_name_set = set(arom_names or [])
            docked_arom = [(a[0], a[1], a[2]) for a in poses[0][1]
                           if a[4] in arom_name_set]
            best, best_res = 1e9, ""
            if docked_arom:
                for rk, pts in load_receptor_aromatic_carbons(rec_pdbqt).items():
                    for p in pts:
                        for q in docked_arom:
                            d = math.dist(p, q)
                            if d < best:
                                best, best_res = d, rk
            m["arom_pocket_dist_A"] = round(best, 2) if best < 1e8 else None
            m["arom_pocket_residue"] = best_res if best < 1e8 else ""
            # ---- 催化取向: 全 pose 统计(2026-10-05) ----
            # 不看 top1: 单次对接 9 个 pose 自身就散开 6.92 Å, 拿 top1 判"有没有
            # 摆到催化位"等于抛硬币。oriented = 至少一个可水解(scissile)酯羰基
            # 碳进入亲核接触球 —— 成酰基酶的几何前置条件, 与 kcat 直接相关。
            # 羧酸端(酸型位点)不计入 oriented: 那是水解产物, 本就不成酰基酶。
            try:
                _sites = find_scissile_sites(poses[0][1]) if poses[0][1] else []
                _anch = receptor_anchors(rec_pdbqt, triad)
                _rows = []
                for _e, _at in poses:
                    if _e is None:
                        continue
                    _o = pose_orientation(_at, _sites,
                                          _anch["nuc_pts"], _anch["his_pts"])
                    _o["energy"] = _e
                    _rows.append(_o)
                _sm = summarize_side(_rows)
                _sm["sites_total"] = len(_sites)
                _sm["sites_acid"] = sum(1 for s in _sites if s["acid"])
                m["orientation"] = _sm
            except Exception as e:  # 取向是附加指标, 失败不得拖垮对照评估本身
                LOG.warning("[eval-smiles] %s 侧催化取向统计失败(%s: %s)",
                            tag, type(e).__name__, e)
                m["orientation"] = {"ok": False,
                                    "error": f"{type(e).__name__}: {e}"}
            m["chain_used"] = used_chain
            return m

        LOG.info("[eval-smiles] 多种子双边对照 (盒心=三联体质心 %s, 尺寸 %s, seeds=%s, "
                 "exh=%d, cpu=1)", np.round(center, 2).tolist(),
                 [round(s, 1) for s in size], seeds, exhaustiveness)
        runs: List[Dict[str, Any]] = []
        for s in seeds:
            ref = _dock_one("ref", ref_pdb, s)
            alt = _dock_one("alt", alt_pdb, s)
            if not ref or not alt:
                LOG.warning("[eval-smiles] seed=%d 双边未完成(ref=%s, alt=%s), 跳过该种子",
                            s, "ok" if ref else "fail", "ok" if alt else "fail")
                continue
            d_e = round(alt["energy_kcal_mol"] - ref["energy_kcal_mol"], 2)
            _anchor_changed = bool(ref["arom_pocket_residue"] and
                                   alt["arom_pocket_residue"] and
                                   ref["arom_pocket_residue"] != alt["arom_pocket_residue"])
            runs.append({
                "seed": s,
                "ref": {k: v for k, v in ref.items() if k != "chain_used"},
                "alt": {k: v for k, v in alt.items() if k != "chain_used"},
                "delta_E_alt_minus_ref": d_e,
                "arom_anchor_changed": _anchor_changed,
            })
        if not runs:
            LOG.warning("[eval-smiles] 全部种子双边对接失败")
            return None
        summary = _summarize_smiles_runs(runs)
        # ---- 催化取向: 跨种子汇总 + 自否证判词(2026-10-05) ----
        # 判词由 orientation_sides 单点产出, report 照抄。两侧都没检出取向时
        # verdict 会自带「能量不得用于机理归因」的红线。
        _o_runs = _summarize_orientation_runs(runs)
        _per_side = {
            "wt": {"oriented_frac": _o_runs["oriented_frac_ref"],
                   "oriented_frac_std": _o_runs["oriented_frac_ref"]["std"],
                   "relay_frac": _o_runs["relay_frac_ref"],
                   "nearest_mean_A": _o_runs["nearest_C_to_nuc_ref_A"],
                   "n_scissile_sites": _o_runs["n_scissile_sites"],
                   "n_pose": _o_runs["n_pose_ref"],
                   "n_oriented": _o_runs["n_oriented_ref"]},
            "mut": {"oriented_frac": _o_runs["oriented_frac_alt"],
                    "oriented_frac_std": _o_runs["oriented_frac_alt"]["std"],
                    "relay_frac": _o_runs["relay_frac_alt"],
                    "nearest_mean_A": _o_runs["nearest_C_to_nuc_alt_A"],
                    "n_scissile_sites": _o_runs["n_scissile_sites"],
                    "n_pose": _o_runs["n_pose_alt"],
                    "n_oriented": _o_runs["n_oriented_alt"]},
        }
        _o_verdict = orientation_sides(_per_side["wt"], _per_side["mut"])
        return {
            "orientation": {
                "per_side": _per_side,
                "verdict": _o_verdict,
                "per_seed": _o_runs.get("per_seed_oriented_frac", {}),
                "thresholds": {"nuc_to_scissile_C_max_A": NUC_TO_SCISSILE_C_MAX,
                               "his_to_leaving_O_max_A": HIS_TO_LEAVING_O_MAX},
            },
            "available": True, "mode": "smiles",
            "smiles": smiles,
            "box_center": [round(float(c), 3) for c in center],
            "box_size": [round(float(s), 2) for s in size],
            "box_source": "triad_ca_centroid(ref)",
            "seeds": [r["seed"] for r in runs],
            "exhaustiveness": exhaustiveness, "num_modes": num_modes,
            "vina_cpu": 1,
            "runs": runs,
            "summary": summary,
            "note": ("外源配体无晶体基准, 不做等效判定; 对照必须同机同环境产出"
                     "(跨 OpenBabel 版本存在 0.2~0.5 kcal/mol 系统差)"),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


__all__ = [
    "cocrystal_comparative_dock",
    "smiles_comparative_dock",
    "_summarize_smiles_runs",
    "_summarize_orientation_runs",
]
