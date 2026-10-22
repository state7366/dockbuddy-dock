# -*- coding: utf-8 -*-
"""box.py —— 对接盒子中心与尺寸的自动决策(三联体 / 共晶配体 / 手动三模式)。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import math, os
from typing import Dict, List, Optional, Tuple
from ._core import LOG, _TYPE2ELEM, ad_type_of, env_float
from .config import DockingConfig
from .geom import parse_residue_token
from .triad import detect_catalytic_triad
from .ligand import _native_ligand_info


def _ligand_span(pdbqt_path: str) -> Optional[Tuple[float, float, float]]:
    """配体(刚性或柔性 pdbqt)**重原子**在 x/y/z 上的跨度(Å)。用于优化项D:
    auto/triad 模式按配体跨度自适应放大盒子, 避免长配体被固定小盒裁剪。
    读取失败/无原子返回 None。

    M7(2026-10-02): 滤氢按 AutoDock 类型码(H/HD/HS) —— 原实现不滤氢,
    文档却称"重原子跨度"(探针实证: x=20 的氢 → span=20.0)。氢在最外沿时
    盒子被多放大 1~2 Å, 属保守偏差, 但口径必须与文档一致。"""
    xs, ys, zs = [], [], []
    try:
        with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.startswith(("ATOM", "HETATM")):
                    continue
                t = ad_type_of(ln)
                if _TYPE2ELEM.get(t, t[:1] if t else "X").upper() == "H":
                    continue
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                except ValueError:
                    continue
                xs.append(x); ys.append(y); zs.append(z)
    except OSError:
        return None
    if not xs:
        return None
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

BOX_LIG_TRIAD_MAX = env_float("DOCK_BOX_LIG_TRIAD_MAX", 15.0)  # 非数字 env 不再 import 期崩溃(第六轮 C7)


def _ca_centroid_on_residues(pdb_path: str, residues: List[Tuple[Optional[int], Optional[str]]],
                             default_chain: str = "") -> Optional[Tuple[float, float, float]]:
    """按 [(残基序号, 链或None), ...] 收集 Cα 原子并求质心; 任一残基缺 CA 返回 None。

    - token 链号优先; 无链号用 default_chain(逐链任务 = 当前任务链, 由调用方传入);
    - 所有 token 均未落到具体链时(纯序号 token + 空 default_chain), 取文件中
      **首条齐全链**(同号残基都在) —— 兼容多链组装体的链无关写法;
    - altLoc 成对原子只取首构象(M9 口径); 同残基多模型/重复行取首个。

    用途(2026-10-03):
      - 手动催化残基(cfg.catalytic)驱动盒心 —— 催化 Ser→Ala 灭活突变体
        三联体几何判据必然放宽, 自动检测失败时这是唯一锚定手段;
      - triad_transfer: 姊妹链检出的三联体残基号移植到当前链同号残基。
    """
    eff: List[Tuple[int, str]] = []
    for rn, rc in residues:
        if rn is None:
            return None
        eff.append((int(rn), (rc or default_chain) or ""))
    want_nums = {rn for rn, _ in eff}
    want_any = not any(ch for _, ch in eff)
    per_chain: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
    try:
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.startswith("ATOM") or ln[12:16].strip() != "CA":
                    continue
                alt = ln[16] if len(ln) > 16 else " "
                if alt not in (" ", "A"):
                    continue
                try:
                    ri = int(ln[22:26].strip())
                    xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                except ValueError:
                    continue
                if ri not in want_nums:
                    continue
                ch = ln[21].strip()
                d = per_chain.setdefault(ch, {})
                if ri not in d:
                    d[ri] = xyz
    except OSError:
        return None
    pts: List[Tuple[float, float, float]] = []
    if want_any:
        # 链无关: 取首条齐全链(文件顺序)
        for ch in per_chain:
            if len(per_chain[ch]) == len(want_nums):
                pts = list(per_chain[ch].values())
                break
        else:
            return None
    else:
        for rn, ch in eff:
            xyz = per_chain.get(ch, {}).get(rn)
            if xyz is None:
                return None
            pts.append(xyz)
    n = len(pts)
    return (sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)


def _resnames_for_resnums(pdb_path: str, resnums, chain: str = "") -> Dict[int, str]:
    """按残基号取残基名(每残基取首个 ATOM 行; altLoc 只认首构象)。

    chain 非空 = 链限定; chain 空 = 任意链首个命中(链无关)。找不到的号不在返回值。
    用途: triad_transfer 的残基名守卫 —— hetero-寡聚体的"姊妹链"可能是不同蛋白,
    同号残基语义错位时必须拒绝移植而不是静默给出错误盒心。
    """
    want = {int(r) for r in resnums}
    out: Dict[int, str] = {}
    try:
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.startswith("ATOM"):
                    continue
                alt = ln[16] if len(ln) > 16 else " "
                if alt not in (" ", "A"):
                    continue
                ri = ln[22:26].strip()
                if not ri.isdigit():
                    continue
                rnum = int(ri)
                if rnum not in want or rnum in out:
                    continue
                if chain and ln[21].strip() != chain:
                    continue
                out[rnum] = ln[17:20].strip()
                if len(out) == len(want):
                    break
    except OSError:
        pass
    return out


def _log_pocket_candidates(pdb_path: str, n: int = 3) -> List[Dict]:
    """兜底盒心之前打印口袋检测候选 —— 优化项 A7 的**诊断**用法, **不改盒心**。

    为什么只诊断不自动改盒心(2026-10-04 实测):
      - detect_pockets 的最大空腔质心距蛋白质几何中心仅 1.7 Å(它找到的是"蛋白内核",
        不是催化口袋); 而真催化位点距蛋白质心 15 Å, 只出现在"最深自由点"列表边缘。
      - 原因: α/β 水解酶(含本品类)的催化 cleft 是**通向表面的开放隧道**, 纯几何
        "最大/最深空腔"天然收敛到内核而非隧道末端。自动改盒心会把盒子推离真实
        位点(实测比蛋白质心还差 ~10 Å), 因此保留 protein_centroid, 只把候选列给
        人工复核(疑似口袋时用显式 --box_center 覆盖)。

    口袋检测是纯诊断, 任何异常都不该阻断对接 —— 一律吞掉并降级为 debug 日志。
    """
    try:
        from .pocketgeom import detect_pockets
    except Exception as e:                                   # pragma: no cover
        LOG.debug("pocketgeom 不可用, 口袋候选诊断跳过: %s", e)
        return []
    try:
        cands = detect_pockets(pdb_path, step=1.0)
    except Exception as e:                                   # pragma: no cover
        LOG.debug("detect_pockets 异常, 口袋候选诊断跳过: %s", e)
        return []
    for i, c in enumerate(cands[:n], 1):
        LOG.warning("   口袋候选%d: 中心=(%.2f, %.2f, %.2f) 体积=%.0f Å³ "
                    "最深埋深=%.1f Å 接触残基=%d —— 疑似结合位点时请显式给 --box_center",
                    i, c["center"][0], c["center"][1], c["center"][2],
                    c["volume_A3"], c.get("max_depth_A", 0.0),
                    c["n_residues_within"])
    return cands


def auto_box_center(pdb_path: str, cfg: DockingConfig,
                    triad: Optional[Dict] = None,
                    full_pdb: str = ""
                    ) -> Tuple[Optional[Tuple[float, float, float]], str, Optional[Dict]]:
    """根据 box_mode 计算盒子中心, 返回 (center, assumption, note)。

    assumption ∈ {manual, native_ligand, triad, triad_ligand_far, catalytic,
                  triad_transfer, protein_centroid, none}。
    assumption 用于结果可靠性研判: manual/native_ligand/triad/triad_ligand_far 都是把
    盒子锚定在已知位点; catalytic/triad_transfer 也是位点锚定(2026-10-03 新增,
    见下), 可靠性介于检测三联体与 protein_centroid 之间;
    protein_centroid 是把盒子放在蛋白质几何中心(大概率不可靠, 必须预警)。

    auto 路径回退链(2026-10-03 补全, 修复逐链拆分任务"未检出三联体即盲盒"):
      检测三联体 → 手动催化残基(cfg.catalytic 三 token 的 Cα 质心, assumption=catalytic;
      催化 Ser→Ala 灭活突变体构象超自动判据时的唯一锚定)
      → triad_transfer(full_pdb 完整组装体姊妹链检出的三联体残基号, 移植到当前链
      同号残基 Cα 质心, assumption=triad_transfer; 逐链任务的 pdb_path 是拆分文件
      无姊妹链, 拷贝间构象差异致本链漏检时用) → protein_centroid 兜底。
      手动 catalytic 优先于检测三联体(用户显式意图)。
      兜底 protein_centroid 时会**额外打印**口袋检测候选(见 _log_pocket_candidates)
      供人工复核, 但盒心仍用蛋白质心 —— 实测本类酶的"最大空腔"是蛋白内核, 换成
      它反而更远离催化位点, 详见该函数 docstring。

    note: None 或 dict, 记录"共晶配体 vs 三联体"的距离判读(2026-09-30 新增):
      - ligand 模式找到共晶配体时, 量其质心到三联体 Cα 质心的距离:
        - 距离 ≤ BOX_LIG_TRIAD_MAX(默认 15 Å): 配体在催化位点, 用配体盒子(note 记距离);
        - 距离 > 上限且**非自对接**: 配体不在催化位点(次级口袋/变构位/添加剂),
          盒子改用三联体中心(assumption="triad_ligand_far"), note 记录距离与原因;
        - 距离 > 上限但**是自对接**(cfg.ligand_name 非空, 对接的就是共晶配体本身):
          仍用配体盒子(移动盒子会让 selfdock RMSD 失去协议验证意义), note 记录距离与
          "配体结合位点远离催化三联体"的提示。
      - triad_transfer 成功时记 {transferred_from, resnums}(姊妹链链名 + 三残基号)。
    """
    note: Optional[Dict] = None
    if cfg.box_mode == "manual" and cfg.box_center:
        LOG.info(f"盒子中心来源=manual(用户指定): {cfg.box_center}")
        return (tuple(cfg.box_center), "manual", None)
    if cfg.box_mode == "manual" and not cfg.box_center:
        # 配置错误显式提示(2026-10-03): manual 但没给中心, 此前静默落入
        # catalytic/triad/protein_centroid 自动回退链, 配置错误被当结构问题处理。
        LOG.warning("box_mode=manual 但未提供 box_center —— 将回退自动链"
                    "(catalytic→triad→ligand→protein_centroid)确定盒心;"
                    " 如需手动请补 --box_center x,y,z")
    if triad is None and cfg.box_mode in ("ligand", "auto", "triad", ""):
        triad = detect_catalytic_triad(pdb_path)
    if cfg.box_mode == "ligand":
        c, _ = _native_ligand_info(pdb_path)
        if c:
            if triad:
                d = math.dist(c, tuple(triad["ca_centroid"]))
                if d > BOX_LIG_TRIAD_MAX:
                    if cfg.ligand_name:
                        note = {"ligand_triad_dist_A": round(d, 2), "switched": False,
                                "reason": f"共晶配体质心距三联体 {d:.1f}A > {BOX_LIG_TRIAD_MAX}A: "
                                          f"配体结合位点远离催化三联体(次级口袋/变构位); 自对接仍用配体盒子"}
                        LOG.warning(f"⚠ {note['reason']}")
                    else:
                        note = {"ligand_triad_dist_A": round(d, 2), "switched": True,
                                "reason": f"共晶配体质心距三联体 {d:.1f}A > {BOX_LIG_TRIAD_MAX}A: "
                                          f"配体不在催化位点, 盒子改用三联体中心(更适合催化对接)"}
                        LOG.warning(f"⚠ {note['reason']}")
                        return (tuple(triad["ca_centroid"]), "triad_ligand_far", note)
                else:
                    note = {"ligand_triad_dist_A": round(d, 2), "switched": False,
                            "reason": f"共晶配体质心距三联体 {d:.1f}A (≤{BOX_LIG_TRIAD_MAX}A, 在位)"}
            LOG.info("盒子中心来源=native_ligand(共晶配体质心)")
            return (c, "native_ligand", note)
        LOG.warning("ligand 模式未找到共晶配体, 回退 auto(三联体/蛋白质心)")
    # === auto 路径新回退链(2026-10-03) ===
    # (a) 手动催化残基: cfg.catalytic 三 token 的 Cα 质心。
    #     之前 receptor 阶段手动残基只产出 catalytic_keys 而 triad=None,
    #     盒心盲走 protein_centroid(催化残基灭活后自动判据失效案例的根因)。
    #     优先于自动检测(用户显式意图)与 triad_transfer。
    if cfg.catalytic and all(cfg.catalytic):
        toks = [parse_residue_token(t) for t in cfg.catalytic]
        if all(rn is not None for rn, _ in toks):
            cat_c = _ca_centroid_on_residues(pdb_path, toks, default_chain=cfg.receptor_chain)
            if cat_c is not None:
                LOG.info(f"盒子中心来源=catalytic(手动催化残基 Cα 质心): {list(cfg.catalytic)}")
                return (cat_c, "catalytic", None)
            LOG.warning(f"手动催化残基 {list(cfg.catalytic)} 在受体中未齐全"
                        f"(缺 CA 或链不符), 回退自动检测/移植链")
        else:
            LOG.warning(f"催化残基 token 无法解析: {list(cfg.catalytic)} "
                        f"(应如 '170' / '170.A' / 'SER170'), 回退自动检测/移植链")
    # (b) triad_transfer: 逐链拆分文件未检出三联体时, 借完整组装体(full_pdb)
    #     姊妹链检出的三联体残基号, 移植到当前链同号残基(拷贝间构象差异致本链漏检)。
    if triad is None and full_pdb and os.path.exists(full_pdb) \
            and os.path.abspath(full_pdb) != os.path.abspath(pdb_path):
        t2 = detect_catalytic_triad(full_pdb)
        if t2:
            resnums = [t2["nuc"]["resnum"], t2["his"]["resnum"], t2["asp"]["resnum"]]
            tch = cfg.receptor_chain or (t2["nuc"]["chain"] or "")
            # 残基名守卫: 目标链同号残基须与三联体身份一致(亲核体允许 S/T/C/A 家族,
            # 兼容 S→A 突变体); 不一致说明姊妹链不是同源拷贝, 拒绝移植。
            names = _resnames_for_resnums(pdb_path, resnums, chain=tch)
            if (len(names) == 3
                    and names.get(resnums[0]) in ("SER", "THR", "CYS", "ALA")
                    and names.get(resnums[1]) == "HIS"
                    and names.get(resnums[2]) in ("ASP", "GLU")):
                c2 = _ca_centroid_on_residues(
                    pdb_path, [(r, None) for r in resnums], default_chain=tch)
                if c2 is not None:
                    LOG.info(f"盒子中心来源=triad_transfer(完整组装体链 {t2['nuc']['chain'] or '-'} 检出三联体 "
                             f"{t2['nuc']['resname']}{t2['nuc']['resnum']}/{t2['his']['resname']}{t2['his']['resnum']}/"
                             f"{t2['asp']['resname']}{t2['asp']['resnum']}, 移植到当前链({tch or '首条齐全链'})同号残基)")
                    return (c2, "triad_transfer",
                            {"transferred_from": t2["nuc"]["chain"] or "",
                             "resnums": resnums})
            else:
                LOG.warning("triad_transfer 残基名守卫未通过(目标链同号残基与三联体身份不符, "
                            "可能姊妹链非同源拷贝), 放弃移植回退检测/蛋白质心")
    # auto: 催化三联体 Cα 质心
    if triad is None:
        triad = detect_catalytic_triad(pdb_path)
    if triad:
        LOG.info("盒子中心来源=triad(催化三联体 Cα 质心)")
        return (tuple(triad["ca_centroid"]), "triad", None)
    # 最后兜底: 蛋白质心(泛化时可靠性低, 必须预警)
    pts = []
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("ATOM"):
                # M9(2026-10-02): altLoc 成对原子(A/B 双构象)只取首个构象,
                # 否则同一原子被双计, 质心向高分辨率侧偏移。(短行缺列视为无 altLoc)
                alt = ln[16] if len(ln) > 16 else " "
                if alt not in (" ", "A"):
                    continue
                try:
                    pts.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError:
                    pass
    if pts:
        LOG.warning("⚠ 盒子中心来源=protein_centroid(未检测到三联体/共晶配体, "
                    "盒子落在蛋白质几何中心——对结合位点不在中心的蛋白, 结果大概率不可靠!)")
        _log_pocket_candidates(pdb_path)
        return ((sum(p[0] for p in pts)/len(pts),
                 sum(p[1] for p in pts)/len(pts),
                 sum(p[2] for p in pts)/len(pts)), "protein_centroid", None)
    return (None, "none", None)

__all__ = [
    "BOX_LIG_TRIAD_MAX",
    "_ca_centroid_on_residues",
    "_ligand_span",
    "_resnames_for_resnums",
    "auto_box_center",
]
