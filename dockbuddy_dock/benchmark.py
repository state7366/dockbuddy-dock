# -*- coding: utf-8 -*-
"""
benchmark.py —— 对接协议科学回归测试(三锚定标) · 单一事实源
================================================================================
把「协议能把已知晶体姿态复现到多准」这个前置问题, 变成**可重复运行、可回归比对**
的正式能力(不再是 tools/oneoff 的一次性脚本)。

三个锚(Anchor) —— 同一配体/框心/穷举, 只换受体, 因而可做差:
  mut_crystal     突变体晶体(剥配体)          → 协议自身的 pose 复现地板(纯对接误差)
  wt_crystal      WT 晶体                     → 受体蛋白自身结构差异带来的偏移
  modeled_mutant  由 WT 经 mutate 建模的突变体 → 建模误差 + 对接误差

核心产出:
  protocol_error_bar_A = mut_crystal 锚 top1 RMSD 中位数 → 协议地板, 结论里一切
  小于它的姿态差都标记为"不可分辨"。

科学回归(本模块相较一次性脚本新增的能力):
  跑一次 benchmark 落盘结果 → `save_baseline` 固化基线 → 之后每次改代码重跑
  benchmark, `check_baseline` 断言 protocol_error_bar_A 中位数没有退化(变差超过
  容忍阈值)。这是「改代码不破坏对接精度」的科学安全网, 与单元测试(只验逻辑)互补。
"""
import os
import json
import statistics

from ._core import LOG

# 默认盒子(Å); 框心取共晶配体质心
DEFAULT_BOX_SIZE = (22.0, 22.0, 22.0)
DEFAULT_SEED = 42
DEFAULT_EXH = 8
DEFAULT_NUM_MODES = 9

# 配体门槛: 少于 8 个重原子 / 少于 4 个碳的基本不是"底物级配体"
MIN_HEAVY_ATOMS = 8
MIN_CARBONS = 4

# Vina 1.2.x 的 PDBQT parser 不认的元素(实测: 硼 CB4 → "Atom type B is not a valid
# AutoDock type")。这些配体**不是**定标素材, 提前筛掉。
VINA_UNSUPPORTED_ELEMENTS = {"B"}

# 1) 非配体名单(结晶添加剂/缓冲液/离子/辅基/糖)
#    与 libdock._SOLVENT_RESNAMES 的关系: 后者是**对接受体制备**口径(去水/去盐),
#    本名单是**基准配体挑选**口径 —— 两者用途不同, 不能互相顶替。
#    新增条目请同步写注释, 便于审稿人复核"为什么这不是配体"。
# ---------------------------------------------------------------------------
NON_LIGAND_RESNAMES = {
    # --- 离子 / 金属(母液离子或重金属化衍生物, 单原子) ---
    "RB": "铷离子", "CS": "铯离子", "HG": "汞离子(重金属衍生物)",
    "CD": "镉离子", "BA": "钡离子", "SR": "锶离子", "LI": "锂离子",
    "PT": "铂", "AU": "金", "AG": "银", "NH4": "铵",
    # --- 缓冲液 / pH 调节剂 ---
    "BCT": "Bis-Tris 缓冲液", "TRS": "Tris", "MES": "MES 缓冲",
    "EPE": "HEPES", "ADA": "ADA 缓冲", "TAU": "牛磺酸(结晶添加剂)",
    # --- 短羧酸 / 有机小酸(多为结晶液成分) ---
    "ACT": "乙酸", "ACY": "乙酸根", "FMT": "甲酸", "MLA": "苹果酸",
    "CIT": "柠檬酸", "TLA": "酒石酸", "BTB": "BTB 缓冲",
    # --- 低温保护剂 / 共溶剂 ---
    "GOL": "甘油", "EDO": "1,2-乙二醇", "DIO": "1,4-二氧六环", "MPD": "MPD",
    "PEG": "PEG", "PGE": "PEG 衍生物", "PGO": "丙二醇",
    "BME": "β-巯基乙醇", "DMS": "DMSO(助溶剂)", "MSA": "甲磺酸",
    "SCN": "硫氰酸根", "AZI": "叠氮根",
    # --- 辅基 / 辅酶 / 铁硫簇(大分子 prosthetic, 非可对接小分子) ---
    "HEM": "血红素(辅基)", "HEME": "血红素", "NAD": "NAD", "NADP": "NADP",
    "FAD": "FAD", "FMN": "FMN", "ATP": "ATP", "ADP": "ADP", "COA": "辅酶A",
    "SAH": "SAH", "SAM": "SAM", "FES": "铁硫簇", "SF4": "铁硫簇",
    "PLC": "磷脂",
    # --- 糖 / 糖醇(结晶或结合位点填充, 非本管线塑料降解底物) ---
    "MAN": "甘露糖", "MMA": "甲基甘露糖苷", "GLC": "葡萄糖", "GAL": "半乳糖",
    "BGC": "β-D-葡萄糖", "MAL": "麦芽糖", "LAC": "乳糖", "SUC": "蔗糖",
    "TRE": "海藻糖", "SOR": "山梨醇", "XYL": "木糖", "FRU": "果糖",
}


# ---------------------------------------------------------------------------
# 2) 共晶配体 curated 挑选
# ---------------------------------------------------------------------------
def _het_groups(pdb_path):
    """按 (残基名, 链, 残基号) 聚合 HETATM 行 —— **不能只按残基名**。

    理由: 同一个 resname 在同一个晶体里可能出现**多份拷贝**(不同 resSeq, 如
    1PZP 的 FTA 在 300/301 两个残基号各一份)。只按名聚合会把两份加成一个"更大
    的分子", 误判成配体; 而且 ``extract_named_ligand(..., single_instance=True)``
    之后只保留最大连通碎片, 与这里的计数口径就不一致了。

    返回 {resname: [(key, [(元素, 行), ...])]}, key = "链+残基号"。
    """
    from .ligand import _SOLVENT_RESNAMES

    groups = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("HETATM"):
                continue
            rn = ln[17:20].strip()
            if not rn or rn in _SOLVENT_RESNAMES or rn in NON_LIGAND_RESNAMES:
                continue
            el = (ln[76:78].strip() or "C")[:1].upper()
            key = f"{ln[21]!r}{ln[22:26].strip()}"
            groups.setdefault(rn, []).append((key, (el, ln)))
    return groups


def curate_reference_ligand(pdb_path, override_name=""):
    """挑出可作为定标基准的共晶配体。

    返回 dict:
      ``ok``        是否可用; ``resname`` 为 None 时不可用
      ``resname`` / ``n_heavy`` / ``n_carbon``
      ``candidates`` 全部候选(含原子数)
      ``excluded``   被排除的候选及理由
      ``reason``     不可用的原因(成功时为 "")
    门槛: 重原子 ≥ MIN_HEAVY_ATOMS 且 碳数 ≥ MIN_CARBONS。
    """
    try:
        groups = _het_groups(pdb_path)
    except OSError as e:
        return {"ok": False, "resname": None, "reason": f"读文件失败: {e}",
                "candidates": {}, "excluded": {}}

    def _heavy(gs):
        return sum(1 for _, (el, _) in gs if el != "H")

    override = (override_name or "").strip().upper()
    cands = {rn: gs for rn, gs in groups.items()}
    excluded = {}

    if override:
        picked = cands.get(override)
        if not picked:
            return {"ok": False, "resname": None,
                    "reason": f"指定的配体 {override} 不在此结构(或已被判为非配体)",
                    "candidates": {k: len(v) for k, v in cands.items()},
                    "excluded": excluded}

        n_heavy = _heavy(picked)
        n_c = sum(1 for _, (el, _) in picked if el == "C")
        if n_heavy < MIN_HEAVY_ATOMS or n_c < MIN_CARBONS:
            return {"ok": False, "resname": None,
                    "reason": f"指定配体 {override} 太小(重原子 {n_heavy} / 碳 {n_c})",
                    "candidates": {k: len(v) for k, v in cands.items()},
                    "excluded": excluded}
        return {"ok": True, "resname": override, "n_heavy": n_heavy,
                "n_carbon": n_c, "candidates": {k: len(v) for k, v in cands.items()},
                "excluded": excluded, "reason": "", "source": "override"}

    if not cands:
        return {"ok": False, "resname": None,
                "reason": "无非溶剂/无非配体 HETATM(纯 apo 或只有缓冲液)",
                "candidates": {}, "excluded": {}}

    # 先看门槛以下的最大候选, 记入 excluded 便于审计(不是"忘了"而是"太小")
    below = {rn: gs for rn, gs in cands.items() if _heavy(gs) < MIN_HEAVY_ATOMS}
    excluded = {rn: f"重原子 {_heavy(gs)} < {MIN_HEAVY_ATOMS}" for rn, gs in below.items()}
    above = {rn: gs for rn, gs in cands.items() if _heavy(gs) >= MIN_HEAVY_ATOMS}
    if not above:
        best = max(cands.items(), key=lambda kv: _heavy(kv[1]))
        return {"ok": False, "resname": None,
                "reason": (f"最大候选 {best[0]} 仅 {_heavy(best[1])} 个重原子"
                           f"(< {MIN_HEAVY_ATOMS});且无候选含 ≥{MIN_CARBONS} 碳"),
                "candidates": {k: len(v) for k, v in cands.items()}, "excluded": excluded}

    rn, gs = max(above.items(), key=lambda kv: (_heavy(kv[1]),
                                                sum(1 for _, (el, _) in kv[1] if el == "C")))
    n_heavy = _heavy(gs)
    n_c = sum(1 for _, (el, _) in gs if el == "C")
    if n_c < MIN_CARBONS:
        return {"ok": False, "resname": None,
                "reason": f"候选 {rn} 含碳 {n_c} < {MIN_CARBONS}(不像有机配体)",
                "candidates": {k: len(v) for k, v in cands.items()}, "excluded": excluded}
    comp = {}
    for _, (el, _) in gs:
        comp[el] = comp.get(el, 0) + 1
    out = {"ok": True, "resname": rn, "n_heavy": n_heavy, "n_carbon": n_c,
           "composition": dict(sorted(comp.items())),
           "n_instances": len({k for k, _ in gs}),
           "candidates": {k: len(v) for k, v in cands.items()},
           "excluded": excluded, "reason": "", "source": "auto"}
    # 提取一致性守卫(2026-10-04, v2 修误伤): 初版拿"挑选总数/实例数"估算单实例
    # 应有原子数, 对 altLoc 配体会双计数(1MOO 4MZ: 挑 18 含 A/B 双构象, per=9,
    # 实际单副本只有 6) → 把合法的多副本配体错杀。FORMUL 明确写着 2(C4H6N2):
    # 这是两份完整的 4-甲基咪唑, single_instance 取一个完整副本完全正确。
    # v2 直接复用 ligand.extract_named_ligand 的 fragment_truncated(v2 判据:
    # **残基组被连通碎片劈开**才是截断; 多副本整进/出 = 合法单实例选择)。
    try:
        from .ligand import extract_named_ligand
        res = extract_named_ligand(pdb_path, rn, single_instance=True,
                                    return_info=True)
        if isinstance(res, tuple):
            tmp, info = res
        else:
            tmp, info = res, {}
        if tmp and os.path.exists(tmp):
            n_ext = sum(1 for ln in open(tmp, encoding="utf-8", errors="ignore")
                        if ln.startswith(("ATOM", "HETATM"))
                        and ln[76:78].strip().upper() not in ("H", "D"))
            out["n_extracted_heavy"] = n_ext
            out["n_heavy_per_instance"] = round(
                n_heavy / max(len({k for k, _ in gs}), 1), 1)  # 诊断: 含 altLoc 双计数, 偏大
            out["fragment_truncated"] = bool(info.get("fragment_truncated"))
            if out["fragment_truncated"]:
                out["ok"] = False
                out["reason"] = ("提取时有残基组被连通碎片劈开(真截断), "
                                 f"提取到 {n_ext} 重原子, 该配体不可用作基准")
            try:
                os.remove(tmp)
            except OSError:
                pass
    except Exception as e:                                    # pragma: no cover
        out["extract_check_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# 3) 聚合与误差地板
# ---------------------------------------------------------------------------
def _finite(vals):
    return [v for v in vals if v is not None and v == v and abs(v) != float("inf")]


def percentiles(vals, qs=(50, 90)):
    """简单分位(线性插值), 空输入返回 {q: None}。"""
    xs = sorted(_finite(vals))
    out = {}
    for q in qs:
        if not xs:
            out[q] = None
            continue
        if len(xs) == 1:
            out[q] = xs[0]
            continue
        k = (len(xs) - 1) * (q / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        out[q] = xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return out


def aggregate_anchors(pairs_out):
    """把逐对结果聚合成误差地板口径。"""
    def col(anchor, key="rmsd_top1_A"):
        return [p["anchors"][anchor][key] for p in pairs_out
                if p.get("anchors", {}).get(anchor, {}).get(key) is not None]

    res = {"n_pairs_used": len(pairs_out)}
    for anchor, label in (("mut_crystal", "protocol"),
                          ("wt_crystal", "receptor_diff"),
                          ("modeled_mutant", "modeling_plus_protocol")):
        top1 = col(anchor)
        rmin = col(anchor, "rmsd_min_A")
        res[anchor] = {
            "label": label, "n": len(top1),
            "rmsd_top1_median": (statistics.median(top1) if top1 else None),
            "rmsd_top1_mean": (statistics.fmean(top1) if top1 else None),
            "rmsd_top1_p90": percentiles(top1, (90,))[90],
            "rmsd_top1_max": (max(top1) if top1 else None),
            "rmsd_min_median": (statistics.median(rmin) if rmin else None),
        }
    # 配对差: 建模锚 − 晶体锚(同一对子内作差, 消掉体系间的系统偏差)
    deltas = []
    for p in pairs_out:
        a = p.get("anchors", {}).get("mut_crystal", {}).get("rmsd_top1_A")
        b = p.get("anchors", {}).get("modeled_mutant", {}).get("rmsd_top1_A")
        if a is not None and b is not None:
            deltas.append(b - a)
    res["modeling_delta"] = {
        "n": len(deltas),
        "delta_median": (statistics.median(deltas) if deltas else None),
        "delta_mean": (statistics.fmean(deltas) if deltas else None),
        "delta_p90": percentiles(deltas, (90,))[90],
        "delta_max": (max(deltas) if deltas else None),
    }
    floor = res["mut_crystal"]["rmsd_top1_median"]
    res["protocol_error_bar_A"] = floor
    res["pose_change_resolvable_A"] = res["mut_crystal"]["rmsd_top1_p90"]
    res["note"] = (
        "protocol_error_bar_A = 协议把共晶配体重新对接回自身晶体时的 pose 复现误差"
        "中位数; 任何 |Δpose| 小于它的突变-induced 位移都不可分辨。"
        "modeling_delta = 由 WT 建模突变体再对接, 相对'突变体晶体'锚的 RMSD 增量中位数,"
        "即 in-silico 建模(侧链替换+局部松弛)本身引入的额外姿态误差。")
    return res


# ---------------------------------------------------------------------------
# 4) 主流程
# ---------------------------------------------------------------------------
def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pair(entry, raw_dir, workdir, *, seed=DEFAULT_SEED, exh=DEFAULT_EXH,
             num_modes=DEFAULT_NUM_MODES):
    """跑一个对子的三锚。返回逐对结果 dict。"""
    from .ligand import resolve_crystal_ligand
    from .receptor import prepare_receptor
    from .selfdock import self_dock_check
    import dockbuddy_mutate

    wt_id, mut_id = entry["wt_pdb"], entry["mut_pdb"]
    mut_pdb = os.path.join(raw_dir, mut_id + ".pdb")
    wt_pdb = os.path.join(raw_dir, wt_id + ".pdb")
    if not (os.path.exists(mut_pdb) and os.path.exists(wt_pdb)):
        return {"pair": f"{wt_id}->{mut_id}", "family": entry.get("family", ""),
                "ok": False, "reason": "结构未缓存(缺 raw_pdb)"}

    cur = curate_reference_ligand(mut_pdb, entry.get("ligand_name", ""))
    base = {"family": entry.get("family", ""), "pair": f"{wt_id}->{mut_id}",
            "mutation": entry.get("mutation", ""), "chain": entry.get("chain", ""),
            "wt_res": entry.get("wt_res"), "mut_res": entry.get("mut_res"),
            "ligand": {k: v for k, v in cur.items() if k != "candidates"},
            "ligand_all_candidates": cur.get("candidates", {}),
            "ok": bool(cur.get("ok")), "reason": cur.get("reason", "")}
    if not cur.get("ok"):
        return base

    lig_name = cur["resname"]
    os.makedirs(workdir, exist_ok=True)
    center = None
    try:
        # --- 框心: 共晶配体(来自突变体晶体)的质心 ---
        lig_pdb, _ = resolve_crystal_ligand(mut_pdb, lig_name)
        import numpy as np
        xyz = []
        for ln in open(lig_pdb, encoding="utf-8", errors="ignore"):
            if ln.startswith(("ATOM", "HETATM")):
                xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if not xyz:
            base.update(ok=False, reason=f"配体 {lig_name} 无坐标")
            return base
        center = list(np.asarray(xyz).mean(axis=0).tolist())

        # --- 受体制备(三锚都要, 统一 keep_metals=True 以保住 ZN 等催化金属) ---
        def _prep(src, tag):
            out_pdbqt = os.path.join(workdir, f"{tag}_{mut_id}.pdbqt")
            ok, _ = prepare_receptor(src, out_pdbqt,
                                     receptor_chain=entry.get("chain", "") or "",
                                     keep_metals=True)
            return out_pdbqt if ok else None

        # Vina 不认的元素(如硼)会直接把对接 kills 掉, 提前判定该配体不适合作基准
        bad_el = sorted({e for e in (cur.get("composition") or {})
                         if e in VINA_UNSUPPORTED_ELEMENTS})
        if bad_el:
            base.update(ok=False,
                        reason=f"配体 {lig_name} 含 Vina 不支持的元素 {bad_el}"
                               f"(实测 PDBQT 'Atom type {bad_el[0]} is not a valid "
                               f"AutoDock type'), 不进入定标")
            return base

        rec_mut = _prep(mut_pdb, "mut")
        rec_wt = _prep(wt_pdb, "wt")

        # --- 锚 3: WT → 突变体建模(本流水线 mutate 管线, 与生产同口径) ---
        mut_modeled = os.path.join(workdir, f"{wt_id}_{mut_id}_modeled.pdb")
        try:
            eng = dockbuddy_mutate.PdbfixerOpenMMEngine()
            rep = eng.mutate(wt_pdb, list(dockbuddy_mutate.parse_mutations([entry["mutation"]],
                                                                    entry.get("chain", "") or "")),
                             mut_modeled)
            modeled_ok = bool(rep.get("minimized")) and os.path.exists(mut_modeled)
            model_note = f"relax={rep.get('relax_mode','')}"
        except Exception as e:  # 建模失败不拖垮整个定标
            LOG.warning("建模失败 %s->%s: %s", wt_id, mut_id, e)
            modeled_ok, mut_modeled, model_note = False, None, f"建模异常 {type(e).__name__}"

        rec_modeled = _prep(mut_modeled, "modeled") if modeled_ok else None

        # --- self_dock_check: 同一配体/同一框心/同一穷举, 只换受体 ---
        common = dict(center=center, size=DEFAULT_BOX_SIZE, R=None,
                      ligand_name=lig_name, seed=seed,
                      exhaustiveness=exh, num_modes=num_modes)
        anchors = {}
        for tag, rec in (("mut_crystal", rec_mut), ("wt_crystal", rec_wt),
                         ("modeled_mutant", rec_modeled)):
            if not rec:
                anchors[tag] = {"ok": False, "reason": "受体制备失败"}
                continue
            # self_dock_check 在对接全盘失败时返回 None(不是 dict), 必须防
            r = self_dock_check(mut_pdb, rec, **common) or {}
            anchors[tag] = {"ok": bool(r), "rmsd_top1_A": r.get("rmsd_top1_A"),
                            "rmsd_min_A": r.get("rmsd_min_A"),
                            "best_pose_index": r.get("best_pose_index"),
                            "reproducible": r.get("reproducible"),
                            "receptor_pdbqt": os.path.basename(rec)}
        base.update(ok=True, center=[round(c, 3) for c in center],
                    ligand_name=lig_name, anchors=anchors,
                    modeled_ok=modeled_ok, model_note=model_note)
    except Exception as e:  # 单个对子崩了不拖垮全量
        import traceback
        LOG.warning("对子 %s 异常: %s\n%s", base["pair"], e, traceback.format_exc()[-800:])
        base.update(ok=False, reason=f"异常 {type(e).__name__}: {e}")
    finally:
        for p in (base.get("_tmp") or ()):
            pass
    return base




# ---------------------------------------------------------------------------
# 基线机制(科学回归)
# ---------------------------------------------------------------------------
DEFAULT_BASELINE_REGRESSION_TOL_A = 0.5   # 协议地板中位数允许回退的上限(Å)


def save_baseline(aggregate, path):
    """把一次 benchmark 的聚合结果固化为基线(供后续回归比对)。"""
    payload = {
        "protocol_error_bar_A": aggregate.get("protocol_error_bar_A"),
        "pose_change_resolvable_A": aggregate.get("pose_change_resolvable_A"),
        "n_pairs_used": aggregate.get("n_pairs_used"),
        "mut_crystal_top1_median": (aggregate.get("mut_crystal") or {}).get("rmsd_top1_median"),
        "modeling_delta_median": (aggregate.get("modeling_delta") or {}).get("delta_median"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_baseline(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_baseline(baseline, aggregate, tol_A=DEFAULT_BASELINE_REGRESSION_TOL_A):
    """与基线比对, 断言协议地板未退化。返回 {ok, verdict, details}。

    退化判据: 本次 protocol_error_bar_A 中位数 > 基线 + tol_A(误差变大=协议变差)。
    基线缺失或本次无有效对子时返回 ok=False 但不抛(由调用方决定是 skip 还是 fail)。
    """
    base_floor = baseline.get("protocol_error_bar_A")
    cur_floor = aggregate.get("protocol_error_bar_A")
    if base_floor is None or cur_floor is None:
        return {"ok": False, "verdict": "unavailable",
                "details": f"基线或本次误差地板缺失(base={base_floor}, cur={cur_floor})"}
    drift = cur_floor - base_floor
    ok = drift <= tol_A
    return {
        "ok": ok,
        "verdict": "pass" if ok else "regression",
        "baseline_floor_A": base_floor,
        "current_floor_A": cur_floor,
        "drift_A": round(drift, 3),
        "tol_A": tol_A,
        "details": (f"协议地板 {base_floor:.2f} → {cur_floor:.2f} Å "
                    f"(漂移 {drift:+.2f} Å, 容忍 {tol_A} Å)"),
    }


def run_benchmark(pairs, raw_dir, workdir, out_path, *, seed=DEFAULT_SEED,
                  exh=DEFAULT_EXH, num_modes=DEFAULT_NUM_MODES, limit=0,
                  family="", skip_modeling=False):
    """跑完整 benchmark: 逐对三锚 → 聚合 → 落盘。返回 payload dict。

    参数:
      pairs      GT 对子清单(每项含 wt_pdb/mut_pdb/ligand_name/mutation/chain/family)
      raw_dir    原始 PDB 缓存目录(存 <id>.pdb)
      workdir    临时受体/配体/建模产物落盘
      out_path   结果 JSON 路径
      limit      最多跑几个对子(0=全部); family 只跑指定族; skip_modeling 跳过建模锚
    """
    os.makedirs(workdir, exist_ok=True)
    if family:
        pairs = [p for p in pairs if p.get("family") == family]
    if limit:
        pairs = pairs[:limit]

    results, used = [], 0
    for i, e in enumerate(pairs, 1):
        LOG.info("[%d/%d] %s -> %s", i, len(pairs), e["wt_pdb"], e["mut_pdb"])
        r = run_pair(e, raw_dir, workdir, seed=seed, exh=exh, num_modes=num_modes)
        if r.get("ok") and r.get("anchors", {}).get("mut_crystal", {}).get("ok"):
            used += 1
        results.append(r)
        # 增量写, 中断也留部分结果
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"n_total": len(pairs), "n_used": used, "results": results},
                      f, ensure_ascii=False, indent=1)

    valid = [r for r in results if r.get("ok")]
    agg = aggregate_anchors(valid)
    payload = {
        "config": {"exh": exh, "num_modes": num_modes, "seed": seed,
                   "box_size": list(DEFAULT_BOX_SIZE),
                   "min_heavy_atoms": MIN_HEAVY_ATOMS, "min_carbons": MIN_CARBONS,
                   "skip_modeling": bool(skip_modeling)},
        "n_total": len(pairs), "n_used": len(valid),
        "results": results, "aggregate": agg,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


__all__ = [
    "curate_reference_ligand", "aggregate_anchors", "run_pair", "run_benchmark",
    "save_baseline", "load_baseline", "check_baseline",
    "DEFAULT_BASELINE_REGRESSION_TOL_A", "NON_LIGAND_RESNAMES",
]
