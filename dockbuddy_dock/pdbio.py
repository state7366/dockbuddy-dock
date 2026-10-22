# -*- coding: utf-8 -*-
"""pdbio.py —— PDB / PDBQT 文本的读写与规整: 原子计数、单模型化、柔性树、链拆分与展开。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import os, subprocess, hashlib
from typing import Dict, List, Tuple
from ._core import LOG, _TYPE2ELEM, ob
from .obabel import _find_obabel


def download_pdb(pdb_id: str, out_path: str, proxy: str = "") -> bool:
    """从 RCSB 下载结构到 out_path(统一输出为 .pdb)。返回是否成功。
    健壮性(批量 32 篇暴露的共性问题):
      - 优先 .pdb; 若 404/过小(部分新发布条目如 9M8I 仅提供 mmCIF), 回退 .cif
        并用 obabel 转 PDB, 保证下游 PDB 流程不变。
      - 设 --max-time 60 超时, 避免代理/网络卡死导致整批挂起。"""
    base = f"https://files.rcsb.org/download/{pdb_id.upper()}"
    urls = [f"{base}.pdb", f"{base}.cif"]
    env = dict(os.environ)
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
    tmp = out_path + ".dl"
    ok = False
    for url in urls:
        try:
            proc = subprocess.run(
                ["curl", "-sSL", "--compressed", "-o", tmp, "--max-time", "60", url],
                env=env, capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 100:
                continue
            head = open(tmp, encoding="utf-8", errors="ignore").read(200)
            # RCSB 对不存在的条目返回 HTML 错误页(以 '<' 开头), 即使 curl 退出码 0、
            #   体积>100 也会被误判为有效 PDB -> 必须显式识别并跳过, 触发 cif 回退。
            if head.lstrip().startswith("<"):
                LOG.warning(f"  {url} 返回 HTML 错误页(非结构数据), 尝试下一回退")
                continue
            is_cif = head.lstrip().lower().startswith("data_") or "_atom_site" in head
            if is_cif:
                # mmCIF -> PDB (obabel 直转, 不加氢; 加氢留待 prepare_receptor)
                # 关键: 临时文件扩展名为 .dl, obabel 无法据此判输入格式, 必须显式
                #   -icif 强制; 否则报 "Cannot read input format .dl" 静默写空。
                obabel = _find_obabel()
                cproc = subprocess.run([obabel, "-icif", tmp, "-opdb", "-O", out_path],
                                       env=env, capture_output=True, text=True, timeout=300)
                if cproc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                    ok = True
                    LOG.info(f"  已下载(mmCIF->{os.path.basename(out_path)}) {pdb_id.upper()} ({os.path.getsize(out_path)} bytes)")
                    break
                else:
                    LOG.warning(f"  mmCIF 转换失败: rc={cproc.returncode} "
                                f"stderr={cproc.stderr[-200:].replace(chr(10), ' ')}")
            else:
                os.replace(tmp, out_path)
                ok = True
                LOG.info(f"  已下载 PDB {pdb_id.upper()} -> {out_path} ({os.path.getsize(out_path)} bytes)")
                break
        except Exception as e:
            LOG.error(f"  下载异常 {url}: {e}")
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass
    if not ok:
        LOG.error(f"  下载 PDB {pdb_id.upper()} 失败(尝试 pdb/cif 均失败)")
    return ok

def _file_fingerprint(path: str, digest_len: int = 16) -> str:
    """源数据内容指纹(sha256 前 N 位十六进制), 用作**缓存键**的组成部分。

    为什么需要它(2026-10-01 缓存审计实证):
      早期实现把"cif→pdb 转换缓存"和"链拆分"的键取为**文件名 stem**
      (`<stem>.pdb` / `<stem>_chA.pdb`)，于是"换结构但保持文件名不变"会
      静默命中旧产物——实测把同名的 model_A.cif 从 1UBQ(602 atoms) 整体换成
      真值 1864 atoms 的大蛋白后, ensure_pdb_input 仍返回 602 atoms 的
      旧 pdb，且不报错、日志还写着"命中缓存"。
    改成内容指纹后，源数据一变键就变，陈旧缓存自然失效；换目录/改名也不再撞车。
    读取失败返回 "unk"(调用方按需处理)。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()[:digest_len]
    except OSError:
        return "unk"


def _count_pdb_atoms(pdb_path: str) -> int:
    """统计 PDB 中 ATOM 记录数(用于无效 PDB 检测)。"""
    try:
        n = 0
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("ATOM"):
                    n += 1
        return n
    except Exception:
        return 0

def _ensure_single_model(pdb_path: str) -> bool:
    """把多模型(NMR 集成/多构象) PDB 缩减为首个模型, 避免 obabel 对几十万原子
    卡死(实测 4CG3 为 77 模型/30 万 ATOM, obabel 吃满 2.7GB 内存挂起)。
    原地重写文件, 返回是否做了缩减。"""
    try:
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return False
    model_start = None
    end_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("MODEL") and model_start is None:
            model_start = i
        elif model_start is not None and ln.startswith("ENDMDL"):
            end_idx = i
            break
    if model_start is None:
        return False  # 单模型, 不动
    header = lines[:model_start]
    body = lines[model_start + 1:end_idx]  # 首个模型内容(不含 MODEL/ENDMDL 标记)
    out = header + body + ["END\n"]
    try:
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.writelines(out)
        return True
    except Exception:
        return False

def _strip_non_atom_records(pdbqt_path: str, keep_flex: bool = False) -> None:
    """只保留 pdbqt 合法记录, 删除 REMARK/HEADER/COMPND 等头注(避免 Vina 报
    'Unknown or inappropriate tag')。
      keep_flex=False(默认, 受体): 仅保留 ATOM/HETATM -> 刚性受体(Vina 要求)。
      keep_flex=True(配体): 额外保留 ROOT/ENDROOT/BRANCH/ENDBRANCH/TORSDOF/END
        -> 柔性配体, Vina 据此识别可旋转键。
    关键 bug 修复(6QGA 实测 5008->999): 记录名位于列 1-6(0-based 0:6)。旧实现误用
    ln[:8].strip(), 而 ATOM 序列号占列 7-11, 使 ln[:8] 含序列号(如 'ATOM   1'),
    .strip() 后非 'ATOM' -> 绝大多数 ATOM 行被误删 -> 受体/配体被截断。已改为
    ln[0:6].strip()。"""
    keep = {"ATOM", "HETATM"}
    if keep_flex:
        keep.update({"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "END"})
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if ln[0:6].strip() in keep]
    with open(pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def _normalize_ligand_pdbqt(pdbqt_path: str) -> bool:
    """把 obabel 写出的柔性配体 pdbqt 重写为严格固定列的"平铺" ATOM 列表(不包裹):
      - 固定列切片提取字段(坐标列严格定宽, 比正则稳健; 单体原子名 C/O/H 不会
        因正则空白分隔误判而被丢弃 -> 修复此前配体只剩 5 个氧原子的严重 bug);
      - 原子名改为唯一 '元素+序号'(C1,O1,H1...), 使芳香碳按名匹配可行。
    包裹为 ROOT/ENDROOT(供 Vina)由调用方在芳香性感知之后完成, 以免 ROOT/ENDROOT
    干扰 OpenBabel 对平铺 pdbqt 的芳香性再感知。返回是否写入了任何原子行。"""
    out_lines = []
    elem_count: dict = {}
    try:
        with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                rec = ln[0:6]
                if rec not in ("ATOM  ", "HETATM"):
                    continue  # 跳过 REMARK/ROOT/BRANCH/TORSDOF 等非原子记录
                # 固定列切片(0-based): 6-11 serial, 12-16 name, 17 altLoc,
                #   17-20 resName, 21 chain, 22-26 resSeq, 30-38/38-46/46-54 xyz,
                #   54-60 occ, 60-66 tf, 70-76 charge, 76-79 type。
                try:
                    serial = int(ln[6:11])
                    name = ln[12:16].strip()
                    resName = ln[17:20].strip()
                    chain = ln[21:22].strip()
                    resSeq = int(ln[22:26])
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                    # 空白列防护(2026-10-03): AutoDockTools 风格 pdbqt 的 occ/bfactor/
                    # charge 列常为纯空白, `float("   ")` 抛 ValueError → 整行原子被
                    # 静默丢弃且无日志(配体残缺进入对接, 失败根因不可见)。
                    occ = float(ln[54:60].strip() or "0")
                    tf = float(ln[60:66].strip() or "0")
                    _chg = ln[70:76].strip()
                    charge = float(_chg) if _chg else 0.0
                    typ = ln[76:79].strip()
                except (ValueError, IndexError):
                    LOG.warning(f"pdbqt 原子行解析失败(跳过该行): {ln.rstrip()[:80]!r}")
                    continue
                elem = _TYPE2ELEM.get(typ.upper(), typ[:1] or "X")
                elem_count[elem] = elem_count.get(elem, 0) + 1
                newname = f"{elem}{elem_count[elem]}"
                if len(newname) > 4:
                    newname = newname[:4]
                out_lines.append(
                    f"ATOM  {serial:>5} {newname:<4} {resName:>3} "
                    f" {resSeq:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{tf:6.2f}    "
                    f"{charge:+6.3f} {typ:<2}\n"
                )
    except Exception as e:
        LOG.error(f"配体 pdbqt 规范化异常: {e}")
        return False
    if not out_lines:
        LOG.error("配体 pdbqt 规范化后无原子行")
        return False
    with open(pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return True

def _flexibilize_ligand_pdbqt(pdbqt_path: str) -> bool:
    """保留 obabel 原生 BRANCH/TORSDOF **柔性树**(供 Vina 采样内部构象), 仅把每个
    原子的"原子名"改为唯一'元素+序号'(C1,O1,H1...), 使芳香碳可按名匹配。

    优化项A 背景: 旧实现把配体压平为 TORSDOF 0(刚性)——实测发现那是对"多分子/多
      TORSDOF 块输入"的误修; 单分子 BRANCH 配体本就被捆绑 vina 1.2.7 正常接受。
      故这里不再压平, 而是保留 obabel 原生树, 只在原子名列(0-based 12:16)原位替换
      唯一名(不动序号/坐标/类型等任何其它列), 因此对 vina 完全等价、对分析可命名。
      BRANCH/ENDBRANCH 引用的是**原子序号**(不随改名改变), 树结构保持有效。
    返回是否写入了任何原子行。"""
    out_lines = []
    elem_count: dict = {}
    natom = 0
    try:
        with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith(("ATOM", "HETATM")):
                    # 仅原位替换原子名列(12:16), 保留 obabel 其余列(序号/坐标/电荷/类型)原样
                    typ = ln[76:79].strip() if len(ln.rstrip("\n")) >= 78 else ln[-3:].strip()
                    elem = _TYPE2ELEM.get(typ.upper(), (typ[:1] or "X"))
                    elem_count[elem] = elem_count.get(elem, 0) + 1
                    newname = (f"{elem}{elem_count[elem]}")[:4]
                    ln = ln[:12] + f"{newname:<4}" + ln[16:]
                    natom += 1
                    out_lines.append(ln if ln.endswith("\n") else ln + "\n")
                elif ln.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
                    out_lines.append(ln if ln.endswith("\n") else ln + "\n")
                # 跳过 REMARK 等非结构记录
    except Exception as e:
        LOG.error(f"配体 pdbqt 柔性化异常: {e}")
        return False
    if natom < 1:
        LOG.error("配体 pdbqt 柔性化后无原子行")
        return False
    with open(pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return True

def _restamp_chain(pdbqt_path: str, chain: str) -> None:
    """将 pdbqt 中所有 ATOM/HETATM 的链字母列(第 22 列)强制改为 chain。
    OpenBabel 在感知到残基不连续时可能重分配链字母(A/B/C...), 而我们的文本过滤
    已确保只保留单链; 此处重盖章可彻底消除链字母歧义, 保证催化残基键匹配一致。"""
    if not chain:
        return
    ch = chain[0]
    out = []
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("ATOM") or ln.startswith("HETATM"):
                if len(ln) >= 22:
                    ln = ln[:21] + ch + ln[22:]
            out.append(ln)
    with open(pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(out)

def _count_atom_lines(pdbqt_path: str) -> int:
    n = 0
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("ATOM") or ln.startswith("HETATM"):
                n += 1
    return n

_HIS_VARIANTS = {"HID", "HIE", "HIP"}

_CYS_VARIANTS = {"CYH", "CYM", "CYX"}

def _norm_resname(name: str) -> str:
    name = (name or "").strip().upper()
    if name in _HIS_VARIANTS:
        return "HIS"
    if name in _CYS_VARIANTS:
        return "CYS"
    return name

def _has_coords(mol) -> bool:
    """判断分子是否已具备非零 3D 坐标(检测 gen3d 假成功)。"""
    for a in ob.OBMolAtomIter(mol):
        if a.GetX() != 0.0 or a.GetY() != 0.0 or a.GetZ() != 0.0:
            return True
    return False

# ============================================================================
# 3. 受体预处理 (保链与残基号; 文本过滤 + 纯加氢)
# ============================================================================

def _pdb_protein_chains(pdb_path: str) -> Dict[str, int]:
    """统计 PDB 各链蛋白 ATOM 数(用于自动选链)。"""
    counts = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            ch = ln[21].strip()
            counts[ch] = counts.get(ch, 0) + 1
    return counts

_AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "MSE", "SEC", "PYL"}

MIN_CHAIN_AA_ATOMS = 300

def dockable_chains(pdb_path: str, min_aa_atoms: int = MIN_CHAIN_AA_ATOMS) -> Dict[str, int]:
    """统计各链**氨基酸** ATOM 数并过滤短链, 返回可独立对接的链 {chain: n_atoms}。
    与 _pdb_protein_chains 的区别: 只数标准/修饰氨基酸残基(排除核酸/糖链),
    且过滤 <min_aa_atoms 的短肽/标签链。"""
    counts: Dict[str, int] = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            if ln[17:20].strip() not in _AA3:
                continue
            ch = ln[21].strip() or "_"
            counts[ch] = counts.get(ch, 0) + 1
    return {c: n for c, n in counts.items() if n >= min_aa_atoms}

def split_pdb_chain(pdb_path: str, chain: str, out_path: str) -> int:
    """把 PDB 中指定链拆成独立 PDB: 该链 ATOM + 该链 HETATM(配体/水/辅因子随链走)。
    关键场景: 两条链但仅一条链带共晶配体 —— 拆出的无配体链天然没有配体
    HETATM, 后续盒子/自对接逻辑自动正确回退(triad 盒子, selfdock N/A)。
    返回写出的坐标记录数(0 = 该链不存在)。"""
    n = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(pdb_path, encoding="utf-8", errors="ignore") as f, \
            open(out_path, "w", encoding="utf-8") as w:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")) and (ln[21].strip() or "_") == chain:
                w.write(ln)
                n += 1
        w.write("END\n")
    return n

def expand_per_chain(pdb_path: str, out_dir: str,
                     min_aa_atoms: int = MIN_CHAIN_AA_ATOMS) -> List[Tuple[str, str]]:
    """多链受体 -> 逐链独立 PDB 文件, 返回 [(chain, path)]; 不足 2 条可对接链时返回 []。"""
    chains = dockable_chains(pdb_path, min_aa_atoms)
    if len(chains) < 2:
        return []
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    # 缓存键带源受体内容指纹(旧实现只用 stem, 源换结构仍会命中旧链产物 —— 见 _file_fingerprint 注释)
    src_fp = _file_fingerprint(pdb_path)
    out = []
    for ch in sorted(chains):
        p = os.path.join(out_dir, f"{stem}_{src_fp}_ch{ch}.pdb")
        if not os.path.exists(p):
            # legacy 回退: 升级前生成的旧命名产物存在则复用, 老产物目录不被打断
            legacy = os.path.join(out_dir, f"{stem}_ch{ch}.pdb")
            if os.path.exists(legacy):
                LOG.info(f"链拆分命中 legacy 产物(旧命名缓存): {legacy}, "
                         f"若源受体 {pdb_path} 已更新请删除该文件重建")
                p = legacy
            else:
                split_pdb_chain(pdb_path, ch, p)
        out.append((ch, p))
    return out

def _ligand_unsupported_elements(pdb_path: str) -> List[str]:
    """返回配体 PDB 中 Vina(AutoDock4 原子类型)不支持的元素列表(如 B/Si)。
    共价抑制剂(硼酸/硼酸酯, 如 1M40 的 CB4)进 Vina 必失败, 前置检出避免无效对接。"""
    bad = set()
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            el = ln[76:78].strip().upper()
            if not el:
                # 元素列缺失: 从原子名推导(剥前导数字, 两字母类型优先)
                nm = ln[12:16].strip().lstrip("0123456789")
                el = (nm[:2] if len(nm) >= 2 and nm[:2].upper() in _VINA_SUPPORTED_ELEMENTS
                      else nm[:1]).upper()
            if el and el not in _VINA_SUPPORTED_ELEMENTS:
                bad.add(el)
    return sorted(bad)

_VINA_SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S",
                            "CL", "BR", "I", "MG", "ZN", "CA", "FE", "MN"}

__all__ = [
    "MIN_CHAIN_AA_ATOMS",
    "_AA3",
    "_CYS_VARIANTS",
    "_HIS_VARIANTS",
    "_VINA_SUPPORTED_ELEMENTS",
    "_count_atom_lines",
    "_count_pdb_atoms",
    "_ensure_single_model",
    "_file_fingerprint",
    "_flexibilize_ligand_pdbqt",
    "_has_coords",
    "_ligand_unsupported_elements",
    "_norm_resname",
    "_normalize_ligand_pdbqt",
    "_pdb_protein_chains",
    "_restamp_chain",
    "_strip_non_atom_records",
    "dockable_chains",
    "download_pdb",
    "expand_per_chain",
    "split_pdb_chain",
]
