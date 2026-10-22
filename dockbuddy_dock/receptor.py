# -*- coding: utf-8 -*-
"""receptor.py —— 受体侧: 链/残基过滤 + 加氢 -> pdbqt(纯文本路线, 避免 OB 重排残基)。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import os, subprocess
from typing import List, Tuple
from ._core import LOG
from .obabel import _find_obabel
from .geom import ref_key_match
from .pdbio import _count_atom_lines, _restamp_chain, _strip_non_atom_records
from .triad import pick_receptor_chain


def _ref_residue_ca(pdb_path: str, ref_residue: str):
    """在 PDB 中查找参考残基的 Cα 坐标(用于隧道轴终点)。"""
    best = None
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            key = f"{ln[17:20].strip()}{ln[22:26].strip()}{ln[21].strip()}"
            # 2026-10-03: startswith → 精确匹配(geom.ref_key_match), 见其 docstring
            if ref_key_match(key, ref_residue) and ln[12:16].strip() == "CA":
                try:
                    return (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                except ValueError:
                    continue
    return None

_METALS = {"ZN", "MG", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD",
           "MO", "W", "V", "CA", "NA", "K", "CL"}

def prepare_receptor(pdb_path: str, out_pdbqt: str, receptor_chain: str = "",
                     keep_metals: bool = False,
                     keep_resnames: Tuple[str, ...] = (),
                     altloc_dedup: bool = True) -> Tuple[bool, str]:
    """受体预处理(两阶段, 关键在保住链/残基号):
      (1) 文本过滤: 仅保留目标链的蛋白 ATOM 记录(原样保留链字母与残基号),
          剔除 HETATM(水/离子/配体, 如 HOH/EDO/SO4/TPO 等);
      (2) OpenBabel: 仅做"加氢(pH7.4) + 写刚性 pdbqt", 全程不调用 DeleteAtom。
    推理: OpenBabel mol.DeleteAtom() 会在重写 pdbqt 时触发链/残基重排(实测单一
      蛋白链被拆成 A/D/F/H... 并丢失残基号), 导致 Vina 解析与催化残基定位失败。
      -> 改为"先文本过滤、再纯加氢写 pdbqt"完整保住链与残基号。

    优化项G(健壮性):
      - keep_metals / keep_resnames: 金属酶/含辅因子体系可保留催化金属(如 ZN/MG)
        或指定 HETATM(如 FMN/FAD/HEM), 避免结合位点因剔除辅因子而失真。
      - altloc_dedup(默认开): 同一(链,残基,原子名)的多套 altLoc 构象只保留首套
        (通常 occupancy 最高), 去除重复原子导致的虚假 clash。
      - 序列缺口检测: 同链 resSeq 跳变>1 视为可能缺失密度, 打日志预警(便于审计)。
    返回 (ok, 实际使用的链)。"""
    if not receptor_chain:
        receptor_chain = pick_receptor_chain(pdb_path)
    LOG.info(f"受体预处理: 读取 {pdb_path} (保留链 {receptor_chain or '全部'}"
             f"{', 保留金属' if keep_metals else ''}"
             f"{', 保留HETATM:' + ','.join(keep_resnames) if keep_resnames else ''})")
    keep_res = {r.strip().upper() for r in (keep_resnames or ())}
    prot_lines: List[str] = []
    kept_het: List[str] = []
    seen_atom = set()
    altloc_dropped = 0
    prev_resseq = None
    seq_gaps: List[Tuple[int, int]] = []
    # 多链标记(2026-10-03 第六轮审查 D8): 指定链时若结构存在其它链的 ATOM,
    # 无链标识(ch 空串)的 HETATM 更可能是外源污染(跨链辅因子/金属), 不应
    # 经 _restamp_chain 归入目标链 —— 多链时丢弃, 单链结构仍保留(属该链)。
    _other_chain_atom = False
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            rec = ln[:6]
            if rec == "ATOM  ":
                ch = ln[21].strip()
                if receptor_chain and ch and ch != receptor_chain:
                    _other_chain_atom = True
                    continue
                resname = ln[17:20].strip()
                resseq_s = ln[22:26].strip()
                atomname = ln[12:16].strip()
                icode = ln[26:27].strip()
                if altloc_dedup:
                    key = (ch, resseq_s, icode, atomname, resname)
                    if key in seen_atom:
                        altloc_dropped += 1
                        continue
                    seen_atom.add(key)
                if resseq_s.isdigit():
                    rs = int(resseq_s)
                    if prev_resseq is not None and rs - prev_resseq > 1:
                        seq_gaps.append((prev_resseq, rs))
                    prev_resseq = rs
                prot_lines.append(ln.rstrip("\n"))
            elif rec == "HETATM":
                resname = ln[17:20].strip().upper()
                # 链过滤(2026-10-03): 与 ATOM 分支同口径 —— 指定链时跨链辅因子/金属
                # 不得混入(否则经 _restamp_chain 统一盖 A 后出现同号 HEM 重叠坐标,
                # 口袋被双重占据 → 假 clash/配体被挤出真实口袋)。
                ch = ln[21].strip()
                if receptor_chain and ch and ch != receptor_chain:
                    continue
                # 无链 HETATM(2026-10-03 第六轮审查 D8): 多链结构中无链标识的
                # 辅因子归属不明 —— 指定链且存在其它链的 ATOM 时按外源污染丢弃。
                if receptor_chain and _other_chain_atom and not ch:
                    continue
                if (keep_metals and resname in _METALS) or (resname in keep_res):
                    kept_het.append(ln.rstrip("\n"))
    if altloc_dropped:
        LOG.info(f"  altLoc 去重: 丢弃重复构象原子 {altloc_dropped} 个(保留首套)")
    if kept_het:
        LOG.info(f"【AUDIT·真实数据】受体保留 HETATM(金属/辅因子) {len(kept_het)} 个原子")
    if seq_gaps:
        LOG.warning(f"  ⚠ 序列缺口(可能缺失密度) {len(seq_gaps)} 处, 如前5: "
                    f"{['%d-%d' % g for g in seq_gaps[:5]]} — 若缺口邻近活性位点, 结果需谨慎")
    if not prot_lines:
        LOG.error("过滤后无蛋白 ATOM 记录"); return (False, receptor_chain)
    tmp_pdb = out_pdbqt + ".protein_only.pdb"
    with open(tmp_pdb, "w", encoding="utf-8") as f:
        f.write("\n".join(prot_lines + kept_het) + "\n")
    LOG.info(f"  蛋白 ATOM 数(链{receptor_chain or '全部'}): {len(prot_lines)}"
             + (f" + 保留HETATM {len(kept_het)}" if kept_het else ""))

    # 受体 pdbqt 由 obabel CLI 写出(避免 Python 绑定的大链截断 bug)。
    #   -h 加(极性)氢; -p 7.4 指定质子化 pH; pdbqt 输出默认刚性受体。
    # 2026-10-04 全盘审计: 删除与此矛盾的旧注释块(曾声称"必须 shell=True 经
    # cmd.exe 派发") —— 实测对照(helper vs 直接、清/不清 BABEL_*)均产出完整
    # 5008 原子; 所谓"受体截断"实为下游 _strip_non_atom_records 记录名误判,
    # 已修复。直接派发即可(更简洁、无额外进程开销)。env 留空以继承运行时已
    # 正确的 BABEL_DATADIR, 切忌覆盖为 bin/data。
    obabel = _find_obabel()
    LOG.info(f"obabel 解析路径: {obabel}")
    cmd = [obabel, tmp_pdb, "-O", out_pdbqt, "-h", "-p", "7.4"]
    LOG.info("obabel 受体转换: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        LOG.error("obabel 受体转换超时"); return (False, receptor_chain)
    # 始终记录 obabel stderr 尾部用于诊断(截断常伴随 kekulize/perceive 警告)
    _ob_err = (proc.stderr or proc.stdout or "")[-1500:]
    if proc.returncode != 0 or not os.path.exists(out_pdbqt):
        LOG.error("obabel 受体转换失败:\n" + _ob_err)
        return (False, receptor_chain)
    n_raw = _count_atom_lines(out_pdbqt)
    LOG.info(f"obabel 原始输出原子数(未 strip/restamp)= {n_raw}")
    # 受体: 刚性(仅 ATOM/HETATM), keep_flex=False
    _strip_non_atom_records(out_pdbqt, keep_flex=False)
    _restamp_chain(out_pdbqt, receptor_chain or "A")
    # 健全性校验: 转换后原子数应接近过滤后蛋白原子数(允许加氢后增多, 但绝不可大幅减少)
    n_out = _count_atom_lines(out_pdbqt)
    if n_out < len(prot_lines) * 0.5:
        LOG.error(f"obabel 输出异常偏少({n_out} vs 过滤 {len(prot_lines)}); 受体可能被截断\nobabel stderr 尾部:\n{_ob_err}")
        return (False, receptor_chain)
    LOG.info(f"✅ 受体已写出: {out_pdbqt} ({n_out} 原子, 链 {receptor_chain or 'A'})")
    # 临时蛋白 PDB 用完即删(2026-10-03: 此前每次 prepare_receptor 残留一个完整
    # 蛋白 PDB, 批量×逐链场景磁盘静默累积; 失败路径保留供诊断)。
    try:
        os.remove(tmp_pdb)
    except OSError:
        pass
    return (True, receptor_chain)

__all__ = [
    "_METALS",
    "_ref_residue_ca",
    "prepare_receptor",
]
