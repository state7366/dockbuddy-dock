# -*- coding: utf-8 -*-
"""ligand.py —— 配体侧: SMILES/坐标文件 -> 3D -> pdbqt; 共晶配体识别与提取。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import os, subprocess, sys, tempfile
from typing import Dict, List, Optional, Tuple
from ._core import LOG, ad_type_of, ob, env_float
from .obabel import _find_obabel
from .pdbio import _count_atom_lines, _flexibilize_ligand_pdbqt, _normalize_ligand_pdbqt


# ============================================================================
# 4. 配体生成 (SMILES / 文件 -> 柔性 pdbqt)
# ============================================================================

# gen3d 沙箱超时(秒): 正常 15~35 重原子配体 gen3d < 10s; 超时判失败
GEN3D_SANDBOX_TIMEOUT = env_float("DOCK_GEN3D_TIMEOUT", 120)  # 非数字 env 不再 import 期崩溃(第六轮 C7)

# 沙箱子进程脚本: 输入(SMILES 或坐标文件) -> 3D(gen3d 优先, OBBuilder 回退) -> SDF 落盘。
# 打印实际成功的路径(gen3d/obbuilder)供父进程记日志。
_GEN3D_SANDBOX_CODE = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
import openbabel                  # wheel __init__ 会把 BABEL_DATADIR 覆写为无数据目录
from dockbuddy_dock.tools import fix_babel_datadir
fix_babel_datadir()
from openbabel import openbabel as ob

mode, src, out_sdf = sys.argv[2], sys.argv[3], sys.argv[4]

def has_coords(m):
    for a in ob.OBMolAtomIter(m):
        if a.GetX() or a.GetY() or a.GetZ():
            return True
    return False

def read_input():
    mol = ob.OBMol()
    conv = ob.OBConversion()
    if mode == "smi":
        conv.SetInFormat("smi")
        return mol if conv.ReadString(mol, src) else None
    ext = os.path.splitext(src)[1].lstrip(".").lower() or "mol2"
    if not conv.SetInFormat(ext):
        for fmt in ("mol2", "mol", "pdb"):
            if conv.SetInFormat(fmt):
                break
    return mol if conv.ReadFile(mol, src) else None

mol = read_input()
if mol is None:
    sys.exit(3)
if has_coords(mol):               # 文件输入已带坐标: 无需 3D 构建, 直接交接
    path = "asis"
else:
    path = "gen3d"
    op = ob.OBOp.FindType("gen3d")
    if not (op and op.Do(mol) and has_coords(mol)):
        path = "obbuilder"
        rebuilt = read_input()    # gen3d 失败可能已部分改写 mol, 重建干净的再做几何构建
        if rebuilt is None:
            sys.exit(3)
        mol = rebuilt
        ob.OBBuilder().Build(mol)
        if not has_coords(mol):
            sys.exit(4)
w = ob.OBConversion(); w.SetOutFormat("sdf")
if not w.WriteFile(mol, out_sdf):
    sys.exit(5)
print(path)
sys.exit(0)
'''


def _build_3d_cli(smiles: str, timeout_s: float = GEN3D_SANDBOX_TIMEOUT):
    """obabel CLI gen3d 回退(SMILES -> 含 3D 坐标的 OBMol)。

    根因(2026-10-10 云实例实测): pip 的 openbabel wheel 在 Linux 上打包的
    _openbabel.so(3.1.0) 对 OBOp gen3d 存在**确定性原生段错误**(即使 BABEL_DATADIR
    正确、力场数据齐全, BHET 这类简单酯也必崩), 而同一 wheel 的 obabel CLI 走
    executable.py 子进程执行 --gen3d 却正常。故在 Python 模块 gen3d 失败时回退到
    CLI(跨平台: 本机 Windows wheel 走 Python 模块正常, 回退只在此触发)。
    返回 (mol, "cli_gen3d") 或 (None, "")。"""
    obabel = _find_obabel()
    if not obabel:
        return (None, "")
    fd, out_sdf = tempfile.mkstemp(suffix=".sdf")
    os.close(fd)
    try:
        p = subprocess.run([obabel, f"-:{smiles}", "-osdf", "--gen3d", "-O", out_sdf],
                           capture_output=True, text=True, timeout=timeout_s)
        if p.returncode != 0:
            LOG.error("obabel CLI gen3d 回退失败(rc=%s):\n%s",
                      p.returncode, (p.stderr or "")[-800:])
            return (None, "")
        conv = ob.OBConversion()
        conv.SetInFormat("sdf")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, out_sdf) or mol.NumAtoms() == 0:
            LOG.error("obabel CLI gen3d 回退产物 SDF 无法读回")
            return (None, "")
        return (mol, "cli_gen3d")
    except subprocess.TimeoutExpired:
        LOG.error("obabel CLI gen3d 回退超时(>%ss)", timeout_s)
        return (None, "")
    finally:
        try:
            os.remove(out_sdf)
        except OSError:
            pass


def _build_3d_sandbox(src: str, mode: str = "smi",
                      timeout_s: float = GEN3D_SANDBOX_TIMEOUT):
    """输入(SMILES 或坐标文件) -> 含 3D 坐标的 OBMol, **全程在独立子进程沙箱中执行**。

    根因(2026-10-03 实锤, 本机+快机双复现): OpenBabel 3.2.1 的
    OBBuilder::GetFragmentCoord 对 rigid-fragments.txt 片段匹配存在确定性原生崩溃
    (Windows 段错误 0xC0000005) —— 触发条件含呋喃环(c1ccco1 家族, 如
    BHEF/FDCA 二酯 "OCCOC(=O)c1oc(cc1)C(=O)OCCO"), gen3d 与 OBBuilder 两条路径
    **均必崩**, Kekule 写法不能绕过, rigid-fragments.txt 数据本身完好(呋喃条目有坐标)。
    进程内调用一旦崩溃, multiprocessing.Pool 的 worker 无声消失(无 traceback),
    Pool 永等其结果 -> 整批挂起不退出(快机 9M8I 条目两轮同点卡死的真相)。
    沙箱化后: 子进程崩溃/超时只损失该配体, worker 存活, 条目记失败。

    mode: "smi"(src=SMILES 字符串) 或 "file"(src=坐标文件路径, 已带坐标则原样交接)。
    返回: (mol, 路径标签) 或 (None, "") —— 失败时调用方判定配体生成失败。
    """
    pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fd, out_sdf = tempfile.mkstemp(suffix=".sdf")
    os.close(fd)
    try:
        try:
            p = subprocess.run(
                [sys.executable, "-c", _GEN3D_SANDBOX_CODE,
                 pipeline_dir, mode, src, out_sdf],
                capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            LOG.error(f"⚠ 3D 构建沙箱超时(>{timeout_s:.0f}s), 判定配体 3D 构建失败: {src}")
            return (None, "")
        if p.returncode != 0:
            why = {3: "输入无法解析", 4: "gen3d/OBBuilder 均未产出坐标",
                   5: "SDF 写出失败"}.get(p.returncode, "原生崩溃(段错误/访问违例)")
            LOG.error(f"⚠ 3D 构建沙箱失败(returncode={p.returncode}, {why}), "
                      f"判定配体 3D 构建失败: {src}")
            # 回退 obabel CLI gen3d(Linux pip wheel _openbabel.so 对 gen3d 段错误,
            # 2026-10-10 云实例实测; 本机 Windows wheel 不触发此分支)
            if mode == "smi":
                return _build_3d_cli(src, timeout_s)
            return (None, "")
        conv = ob.OBConversion()
        conv.SetInFormat("sdf")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, out_sdf) or mol.NumAtoms() == 0:
            LOG.error(f"⚠ 3D 构建沙箱产物 SDF 无法读回: {src}")
            return (None, "")
        path = (p.stdout or b"").decode("utf-8", "ignore").strip()[:40] or "gen3d"
        return (mol, path)
    finally:
        try:
            os.remove(out_sdf)
        except OSError:
            pass

def _ligand_geom_fragmented(pdbqt_path: str, max_iso_A: float = 2.5) -> bool:
    """检测 pdbqt 配体坐标是否"碎片化"(3D 生成不完整: gen3d 假成功后被部分构建,
    或环模板/力场数据缺失导致分子片段被散布到远处)。

    判据: 任一原子到其最近邻的距离 > max_iso_A 即视为存在孤立碎片。
    united-atom pdbqt(非极性氢合并)中成键重原子距离不超过 ~2.1A(C-S 1.8/S-S 2.05),
    2.5A 留足余量, 不会误杀合法配体。单原子配体(如金属离子)直接放行。
    纯文本解析不依赖 openbabel, 缓存命中路径(ligand 阶段)也用它做坏缓存自愈。"""
    import math
    pts = []
    with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                try:
                    pts.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError:
                    pass
    if len(pts) < 2:
        return False
    worst = 0.0
    for i, p in enumerate(pts):
        dmin = min(math.dist(p, q) for j, q in enumerate(pts) if j != i)
        if dmin > worst:
            worst = dmin
    if worst > max_iso_A:
        LOG.warning(f"配体坐标碎片化: 存在孤立原子(最近邻 {worst:.2f}A > {max_iso_A}A), "
                    f"疑似 3D 生成不完整(gen3d 假成功/力场或环模板数据缺失)")
        return True
    return False

def _ligand_mol_to_pdbqt(mol, out_pdbqt, flexible: bool = True,
                         ph: float = 7.4) -> Tuple[bool, List[str]]:
    """把内存配体 OBMol 经 obabel CLI 写出为 Vina 可解析的 pdbqt, 并返回
    配体芳香碳原子名列表(用于对接后按名匹配, 使'芳香环距离'建立在 DOCKED 坐标上)。

    流程: (可选 pH 校正) -> mol 写临时 .mol2 -> obabel 转 pdbqt(-h 加氢)
      -> flexible=True: _flexibilize_ligand_pdbqt(保留 BRANCH/TORSDOF 柔性树, 仅改唯一原子名)
         flexible=False: _normalize_ligand_pdbqt(平铺) + 包裹 TORSDOF 0(旧刚性兜底)
      -> 用 OpenBabel 重读 pdbqt 感知芳香碳(按 ATOM 顺序索引对齐原子名)。

    优化项A(柔性): 实测捆绑 vina 1.2.7 正常接受单分子 BRANCH 配体, 旧的"压平为
      TORSDOF 0"是对多分子/多 TORSDOF 输入的误修; 故默认 flexible=True 恢复构象采样。
    优化项C(pH): 用 OBPhModel.CorrectForPH 按生理 pH 质子化(羧基->羧酸根), 确定性
      实现(不依赖 CLI -p 的数据文件); ph<=0 时跳过。之所以不在 mol 上直接判
      IsAromatic: gen3d/OBBuilder 会改写 mol 且破坏 aromaticity 标记。"""
    import tempfile
    fd, tmpmol2 = tempfile.mkstemp(suffix=".mol2")
    os.close(fd)
    arom: List[str] = []
    try:
        # 优化项C: 按生理 pH 质子化/去质子化(在 DeleteHydrogens 之前设定形式电荷,
        #   后续 obabel -h 只加满足价态的 H, 不会去质子化已带 [O-] 的羧酸根)。
        if ph and ph > 0:
            try:
                ob.OBPhModel().CorrectForPH(mol, float(ph))
            except Exception as e:
                LOG.warning("配体 pH 校正失败(忽略, 用原始质子化态): %s", e)
        mol.DeleteHydrogens()   # 由 obabel -h 统一加氢, 避免重复氢
        tconv = ob.OBConversion(); tconv.SetOutFormat("mol2")
        if not tconv.WriteFile(mol, tmpmol2):
            LOG.error("配体临时 mol2 写出失败"); return (False, [])
        obabel = _find_obabel()
        # 直接派发 obabel CLI 写出 pdbqt(配体小, 无截断风险; 与受体统一走 CLI)。
        cmd = [obabel, tmpmol2, "-O", out_pdbqt, "-h"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not os.path.exists(out_pdbqt):
            LOG.error("obabel 配体转换失败:\n" + (proc.stderr or "")[-1200:])
            return (False, [])
        # 柔性(保留树)或刚性(平铺)处理。
        if flexible:
            if not _flexibilize_ligand_pdbqt(out_pdbqt):
                return (False, [])
        else:
            if not _normalize_ligand_pdbqt(out_pdbqt):
                return (False, [])
        if _count_atom_lines(out_pdbqt) < 1:
            LOG.error("配体 pdbqt 为空"); return (False, [])
        # 碎片化守卫: gen3d 假成功/部分构建会产出散布的分子碎片(视觉上"两条链"),
        # 绝不能让其进入对接/缓存(实测: BABEL_DATADIR 未修复环境生成的坏缓存被沿用)
        if _ligand_geom_fragmented(out_pdbqt):
            LOG.error("配体 3D 坐标碎片化, 判定生成失败"
                      "(请检查 OpenBabel 力场数据/BABEL_DATADIR 配置)")
            return (False, [])
        # 芳香碳名: 重新读 pdbqt, 感知芳香性后按 ATOM 出现顺序(索引)取芳香碳名。
        names = []
        with open(out_pdbqt, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith(("ATOM", "HETATM")):
                    names.append(ln[12:16].strip())
        c2 = ob.OBConversion(); c2.SetInFormat("pdbqt")
        m2 = ob.OBMol()
        if c2.ReadFile(m2, out_pdbqt) and m2.NumAtoms() > 0:
            m2.PerceiveBondOrders()
            for i, a in enumerate(ob.OBMolAtomIter(m2)):
                if i < len(names) and a.IsAromatic() and a.GetAtomicNum() == 6:
                    arom.append(names[i])
        if not flexible:
            # 旧刚性兜底: 包裹 ROOT/ENDROOT + TORSDOF 0(仅作回归对照/逃生开关)。
            with open(out_pdbqt, encoding="utf-8") as f:
                body = f.read()
            with open(out_pdbqt, "w", encoding="utf-8") as f:
                f.write("ROOT\n" + body + "ENDROOT\nTORSDOF 0\n")
        return (True, arom)
    finally:
        try:
            os.remove(tmpmol2)
        except OSError:
            pass

def prepare_ligand(smiles: str, out_pdbqt: str, flexible: bool = True,
                   ph: float = 7.4):
    """由 SMILES 生成 3D 构象并输出 pdbqt。返回 (ok, 芳香碳原子名列表)。
    flexible=True 保留 BRANCH/TORSDOF 柔性树(默认); ph 为质子化 pH(<=0 跳过)。"""
    LOG.info(f"配体生成(SMILES): {smiles} (flexible={flexible}, pH={ph})")
    # 说明: 芳香碳索引不再在此处计算。gen3d/OBBuilder 会改写键级并破坏
    #   aromaticity 标记(对构建后几何调用 IsAromatic 会误判到氧原子), 故芳香性改在
    #   _ligand_mol_to_pdbqt 内部: 先写出"平铺"pdbqt, 再用 OpenBabel 重新读回并
    #   PerceiveBondOrders() 感知芳香碳名(与 pdbqt 原子顺序严格对齐, 稳健可靠)。
    # 2026-10-03 沙箱化(根因见 _build_3d_sandbox docstring): OB 3.2.1 对呋喃环等
    #   片段的 3D 构建(gen3d + OBBuilder)存在确定性原生崩溃(0xC0000005), 进程内
    #   调用会无声杀死 worker(multiprocessing.Pool 永等其结果 -> 整批挂起)。
    #   gen3d 优先(OBBuilder 回退)的语义不变, 只是整体搬进子进程沙箱执行。
    mol, path = _build_3d_sandbox(smiles)
    if mol is None:
        LOG.error("配体 3D 构建失败(沙箱判定), 此配体无法进入对接")
        return (False, [])
    LOG.info(f"  3D 构象: 沙箱 {path}(gen3d MMFF94/UFF → OBBuilder 回退)")
    LOG.info(f"  配体重原子数: {mol.NumHvyAtoms()}  总原子数: {mol.NumAtoms()}")
    ok, arom = _ligand_mol_to_pdbqt(mol, out_pdbqt, flexible=flexible, ph=ph)
    if not ok:
        return (False, [])
    # 与文件路径对称(2026-10-03): 含硼/硅 SMILES 经 OB 同样写出 Vina 不认的类型码,
    # 缺前置拦截时失败只剩 Vina 启动后一句无上下文报错。
    _warn_unsupported_ad_types(out_pdbqt)
    LOG.info(f"✅ 配体已写出: {out_pdbqt}")
    return (True, arom)

def prepare_ligand_from_file(path: str, out_pdbqt: str, flexible: bool = True,
                             ph: float = 7.4):
    """由坐标文件(.mol/.mol2/.pdb)生成 pdbqt。返回 (ok, 芳香碳原子名列表)。
    flexible=True 保留 BRANCH/TORSDOF 柔性树(默认); ph 为质子化 pH(<=0 跳过)。"""
    LOG.info(f"配体生成(文件): {path} (flexible={flexible}, pH={ph})")
    # 2026-10-03: 文件读取+缺坐标时的 3D 构建整体走沙箱 —— 进程内 OBBuilder 对
    # 呋喃环等片段存在确定性原生崩溃(见 _build_3d_sandbox), 文件输入同样可能触发。
    mol, _path = _build_3d_sandbox(path, mode="file")
    if mol is None:
        LOG.error("配体文件读取/3D 构建失败(沙箱判定)"); return (False, [])
    # 说明: 芳香碳索引不在读取时计算(gen3d/OBBuilder 会破坏 aromaticity 标记),
    #   改由 _ligand_mol_to_pdbqt 内部平铺重读并 PerceiveBondOrders() 稳健获取。
    if mol.NumHvyAtoms() == 0:
        LOG.error("配体无重原子"); return (False, [])
    LOG.info(f"  配体重原子数: {mol.NumHvyAtoms()}")
    ok, arom = _ligand_mol_to_pdbqt(mol, out_pdbqt, flexible=flexible, ph=ph)
    if not ok:
        return (False, [])
    _warn_unsupported_ad_types(out_pdbqt)
    LOG.info(f"✅ 配体已写出: {out_pdbqt}")
    return (True, arom)


def _warn_unsupported_ad_types(pdbqt_path: str) -> List[str]:
    """检查制好的 pdbqt 是否含 Vina 解析不了的 AutoDock 原子类型, 给出可操作提示。

    2026-10-02 实测定型: OpenBabel 把**硼**写成 PDBQT 类型 `B`(如 1ERM 的硼酸抑制剂
    BJI、6YZN 的 Q3Q), 而 Vina 的 PDBQT parser 只认 AutoDock 类型表里的名字 —— 跑到一半
    直接 `PDBQT parsing error: Atom type B is not a valid AutoDock type` 失败。**这种失败
    发生在 Vina 启动后、产物写出前**, 日志里只剩一句没上下文的 Vina 报错, 排查成本极高。
    这里在写出后立刻前置检查: 元素取自 _TYPE2ELEM, B 不在其中即命中。

    返回 [未支持的原子类型, ...] 供上层决定是否降级。
    """
    from ._core import _TYPE2ELEM
    bad, seen = [], set()
    try:
        with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.startswith(("ATOM", "HETATM")):
                    continue
                typ = ad_type_of(ln)  # 行尾字段, 不硬编码列号(Vina 输出右移 1 列)
                if typ and typ not in _TYPE2ELEM and typ not in seen:
                    seen.add(typ)
                    bad.append(typ)
    except OSError:
        return []
    if bad:
        LOG.warning(
            "配体含 Vina 不支持的 AutoDock 原子类型 %s(元素: %s)。OpenBabel 用元素原始符号"
            "写类型, Vina 的 PDBQT parser 不接受 → 对接必然失败。规避: (1) 用 --smiles 给该"
            "配体一个可用 SMILES 重制配体; (2) 把该类型临时改成 C/N/O 等通用类型(会改变"
            "原子参数, 仅作探索)。", bad, ", ".join(bad))
    return bad


_SOLVENT_RESNAMES = {"HOH", "WAT", "NA", "CL", "SO4", "PO4", "EDO", "GLY", "GOL",
                     "PEG", "ACT", "CIT", "MG", "CA", "ZN", "K", "TPO", "GDP", "GTP",
                     "MPD", "FMT", "DMS", "BME", "IOD", "BR",
                     "DIO",  # 1,2-乙二醇(EDO 的另一种常见残基名, 如 3F8V 系)
                     "SCN",  # 硫氰酸根(结晶添加物, 不是配体)
                     "MLA", "TLA", "PGE", "NO3", "CO3",
                     "ACY",  # 冰醋酸/乙酸添加剂(与 ACT 同类, 如 7CY0)
                     # 常见结晶缓冲剂(不是配体, 对接比较无意义):
                     "BTB",  # bis-tris
                     "BTP",  # bis-tris propane
                     "MES", "EPE",  # MES / HEPES
                     "TRS",  # Tris
                     "MPS",  # MOPS
                     "ADA",  # ADA buffer
                     }

def _is_solvent_resname(rn: str) -> bool:
    """HETATM 残基名(列 17-20 strip 后)是否应排除的溶剂/缓冲剂。

    兼容两种 PDB 排布: 3 字母名(" HOH"→"HOH")直接查表;
    4 字母名若首字符是 altLoc(占列 17)会形成 "AHOH"/"BHOH"(A/B 构象的水)等形式,
    剥去首字符再查一次。空名/超长名(>4)一律视为排除。"""
    if not rn or len(rn) > 4:
        return True
    if rn in _SOLVENT_RESNAMES:
        return True
    return len(rn) == 4 and rn[1:] in _SOLVENT_RESNAMES

def _native_ligand_info(pdb_path: str) -> Tuple[Optional[Tuple[float, float, float]],
                                                 Optional[Tuple[float, float, float]]]:
    """在 PDB 中找已结合配体(HETATM), 用于 ligand 模式盒子中心/尺寸。

    策略(批量弱生物意义案例暴露的共性修复):
      - 按 (残基名, 序号, 链) 分组, 排除水/离子/常见缓冲剂/冷冻保护剂。
      - 取**原子数最多**的那组作为"真实底物"(避免与 MPD/GOL 等共晶添加物混淆)。
      - 质心 = 该组坐标均值; 盒子尺寸 = 该组各轴跨度 + 余量(夹在 [12, 34] Å)。

    返回 (centroid, box_size); 无有机结合配体时 (None, None)。"""
    groups: Dict[tuple, list] = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("HETATM"):
                continue
            # 残基名解析: 兼顾 3 字母(列 18-20)与 4 字母(列 17-20, 如 MHET)两种排布。
            rn = ln[16:20].strip()
            if _is_solvent_resname(rn):
                continue
            # 定盒只需质心/跨度, 4 字符名剥首字符归并(altLoc 构象/4 字母名都聚合到同组)
            base = rn[1:] if len(rn) == 4 else rn
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            key = (base, ln[22:26].strip(), ln[21:22].strip())
            groups.setdefault(key, []).append((x, y, z))
    if not groups:
        return (None, None)
    # 选原子数最多的组 = 真实底物
    best_key = max(groups, key=lambda k: len(groups[k]))
    pts = groups[best_key]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    MARGIN = 5.0
    def _sz(a):
        return min(34.0, max(12.0, (max(a) - min(a)) + 2 * MARGIN))
    size = (_sz(xs), _sz(ys), _sz(zs))
    LOG.info(f"  共晶配体 {best_key[0]} ({len(pts)} 原子) 质心 ({cx:.3f},{cy:.3f},{cz:.3f}) "
             f"建议盒尺寸 {tuple(round(s,1) for s in size)}")
    return ((cx, cy, cz), size)

# 共价半径(A)近似表: 判连通碎片用。缺项用 0.77(C 的默认值)兜底。
# 2026-10-02 修复: 缺 FE/B/MN/CU/CO/NI 等金属 → 判键阈值走默认 0.77, Fe–N(配位键实测
# 2.0~2.1 Å)算不出成键, Fe 成孤立碎片, 被 _largest_fragment 当成"最大碎片之外"整条丢掉。
# 实测 9T6Y 的 HEM 就因此丢了中心铁(72/73 原子) —— 血红素/金属辅因子配体的金属中心
# 与配位键一并消失。金属配位键按共价半径上限放宽到 2.48 Å 即可正确归入主碎片。
_COV_RADII = {"H": 0.31, "D": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
              "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39,
              "SE": 1.20, "ZN": 1.25, "MG": 1.41, "CA": 1.74, "NA": 1.66, "K": 2.03,
              "FE": 1.32, "B": 0.84, "MN": 1.39, "CU": 1.32, "CO": 1.26,
              "NI": 1.24, "LI": 1.28, "CD": 1.44}


def _atom_element(line: str) -> str:
    """PDB 行的元素符号: 优先元素列(77-78), 退化用原子名首字符。"""
    if len(line) >= 78:
        el = line[76:78].strip().upper()
        if el:
            return el
    return line[12:16].strip()[:1].upper()


def _largest_fragment(lines: List[str]) -> List[str]:
    """按共价键连通性把原子行切分成碎片, 返回原子数最多的那一片的所有行。

    并查集 + 共价半径判键。用于 single_instance 的"只留一份分子"需求:
      - 分离副本(同一残基号出现两次、或 5M17 的 AHOH/BHOH)→ 多碎片 → 留最大者;
      - 成键多残基配体(MAN×3 糖链、肽底物)→ 单碎片 → 原样保留(旧实现会误删)。"""
    import math
    atoms: List[tuple] = []
    for ln in lines:
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except (ValueError, IndexError):
            continue
        atoms.append((ln.rstrip("\n"), xyz, _atom_element(ln)))
    if len(atoms) < 2:
        return [a[0] for a in atoms]
    parent = list(range(len(atoms)))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = math.dist(atoms[i][1], atoms[j][1])
            r = _COV_RADII.get(atoms[i][2], 0.77) + _COV_RADII.get(atoms[j][2], 0.77)
            if 0.4 < d < r + 0.45:
                ra, rb = _find(i), _find(j)
                if ra != rb:
                    parent[ra] = rb
    groups: Dict[int, List[str]] = {}
    for i, a in enumerate(atoms):
        groups.setdefault(_find(i), []).append(a[0])
    biggest = max(groups.values(), key=len)
    LOG.info("配体连通碎片 %s 个, 取最大碎片 %d/%d 原子",
             sorted(len(v) for v in groups.values()), len(biggest), len(atoms))
    return biggest


def extract_named_ligand(pdb_path: str, resname: str,
                         single_instance: bool = False,
                         return_info: bool = False):
    """从受体 PDB 中**按残基名**提取共晶配体(如 mhet / TPA / MHET), 写出独立 .pdb 并返回路径。

    用途: 当用户"按配体名指定配体"(未给 SMILES / 坐标文件)时, 直接从结构里取真实配体做对接
          (等价于对该共晶配体重对接, 用于验证/交叉检验)。大小写不敏感: mhet == MHET。

    参数 single_instance: 同名残基有多个分离副本(如 5M17 的两个 AHOH)时,
          只保留原子数最多的那一个(链,残基号)实例。多副本全抓会让 OpenBabel
          写出多 ROOT 块的 pdbqt, Vina 直接拒收(Unknown or inappropriate tag > ROOT)。
          注意: 成键连接的多残基配体(如甘露糖链 MAN×3)不受此影响——OB 键感知
          会把它识别为单一分子, 不需要(也不应该)开此开关。

    altLoc 感知: 超高分辨结构会把同一配体按交替构象拆开——带 altLoc 的原子
          其列 17-20 呈现为 "A"+残基名(如 1M40 的硼酸抑制剂 CB4 → ACB4/BCB4 +
          无 altLoc 的共享原子 CB4)。此时按 (共享 + 原子数最多的单一构象) 提取,
          保证配体完整且不含重复构象原子。显式传 4 字符全名(如 "ACB4")则只精确匹配该名。

    返回: 临时 .pdb 路径(调用方用完应删除); 结构里没有该 HETATM 残基时返回 None。

    注意: 只按名匹配, 不过滤溶剂/缓冲剂(与 _native_ligand_info 的自动探测不同),
          因为配体名是用户显式意图。"""
    name = (resname or "").strip().upper()
    if not name:
        return None
    shared: List[str] = []                 # rn == name 的原子(无 altLoc 的共享部分)
    alt: Dict[str, List[str]] = {}         # altLoc 前缀 -> 该构象专有原子(rn = 前缀+name)
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("HETATM"):
                continue
            # 残基名解析: 兼顾 3 字母(列 18-20)与 4 字母(列 17-20, 如 MHET)两种 PDB 排布。
            # 取列 17-20 并 strip, 3 字母名会带前导空格被剥离, 4 字母名直接命中。
            rn = ln[16:20].strip().upper()
            if rn == name:
                shared.append(ln.rstrip("\n"))
            elif len(rn) == len(name) + 1 and rn[1:] == name:
                alt.setdefault(rn[0], []).append(ln.rstrip("\n"))
    if not shared and not alt:
        LOG.warning("受体 %s 中未找到名为 %s 的 HETATM 共晶配体",
                    os.path.basename(pdb_path), name)
        return None
    if alt:
        # 构象合并口径: **所有构象按原子名取并集去重**, 而不是"只留原子数最多的那个构象"。
        # 2026-10-02 实测缺陷: 9T6Y 的 HEM 有 altLoc A/B, Fe 只写在构象 B 里; 原实现
        # 选了原子数最多的构象 A → Fe 整条被丢, 单实例配体只剩 72/73 个原子(缺金属中心,
        # 配位键也一并消失)。PDB altLoc 语义下, 只存在于某构象的原子依然是分子的组成。
        # 按原子名并集去重: 同名(即同一原子)只留首次出现的那一行, 跨构象独有原子自然补上。
        conf = min(sorted(alt), key=lambda c: -len(alt[c]))
        seen, kept = set(), []
        for ln in shared + [x for c in sorted(alt) for x in alt[c]]:
            key = (ln[12:16].strip(), ln[21].strip(), ln[22:27].strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(ln.rstrip("\n"))
        if len(alt) > 1:
            LOG.info("配体 %s 存在 altLoc 交替构象(%s), 共享 %d + 各构象按原子名并集去重后"
                     "保留 %d 原子(原子数最多的构象为 %s)",
                     name, "/".join(sorted(alt)), len(shared), len(kept), conf)
    else:
        kept = shared
    before = len(kept)
    truncated = False
    if single_instance:
        # 2026-10-02 修复: 原实现按 (链, 残基号) 分组去重, 对"成键连接的多残基配体"
        # (如 MAN×3 糖链、连续残基构成的共价底物)会误删残基、把完整配体截成碎片。
        # 改成按**共价连通性**找最大碎片: 分离副本(19 号 10 ROOT/170 原子那类)归为
        # 多个碎片→只留最大者; 成键多残基则是单一碎片→原样保留。两种情形都对。
        #
        # 2026-10-04 截断判定(v2, 修初版误伤): 初版拿 len(kept)/before(总量) 比,
        # 会把**多副本配体的合法单实例选择**误判成截断 —— 实测 1PZP 的 FTA 是两个
        # 完整独立副本(A/300、A/301 各 23 原子), 取一个完整副本完全正确, 初版却报
        # "23/46 (50%) 截断", 把有效对子错误剔除了。
        # 正确判据: **某个 (链,残基号) 组被连通碎片劈开(部分保留)**才是截断;
        # 每组整进/出 = 多副本里选一个完整副本(合法)。实测对照(2026-10-04 P1
        # 勘误: 此处曾把 4MZ 误记为"单组(A,300) 被劈开"——实测 1MOO 原始 PDB 是
        # A/264、A/265 两个残基号槽, 各 6 个唯一重原子, FORMUL 2(C4 H6 N2), 即
        # 两份完整 4-甲基咪唑; altLoc 仅是副本内部构象交替):
        #   FTA: 组(A,300)整进、(A,301)整出 → 合法单实例;
        #   4MZ: 组(A,264)/组(A,265) 整进/出 → 合法单实例(v2 重跑 ligand_truncated
        #        =False, 见 output/bench/dock_sens_4MZ_recheck.json 为证)。
        # 真截断(单组被劈开)的实测正面案例暂未出现 —— 判据正例由
        # tests/test_unit_ligand.py 的合成数据测试固化(含多副本+碎化副本两分支)。
        pre = list(kept)
        kept = _largest_fragment(kept)
        LOG.info("残基 %s: single_instance 按连通碎片去重后保留 %d 原子", name, len(kept))
        kept_set = set(kept)
        groups: Dict[Tuple[str, str], List[str]] = {}
        for ln in pre:
            groups.setdefault((ln[21], ln[22:27]), []).append(ln)
        truncated = any(
            any(x in kept_set for x in g) and any(x not in kept_set for x in g)
            for g in groups.values())
        if truncated:
            LOG.warning(f"⚠ 残基 {name}: single_instance 保留 {len(kept)}/{before} 原子, "
                        f"且有残基组被连通碎片**劈开** —— 配体被截断, 该配体的"
                        f"重对接/对接结果不可信(altLoc 并集被切碎的情形)")
    fd, tmp = tempfile.mkstemp(prefix=f"lig_{name}_", suffix=".pdb")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    LOG.info("【AUDIT·真实数据】从受体 %s 提取共晶配体 %s (%d 原子) -> %s",
             os.path.basename(pdb_path), name, len(kept), tmp)
    if return_info:
        # 2026-10-04: 调用方(comparative/bench_protocol)必须能判定"这个配体是不是被
        # 连通碎片切过" —— 拿截断配体做 self-dock 会得到 0.00 Å 的完美复现假象。
        # v2 判据(见上): 残基组被劈开 = 截断; 多副本整进/出 = 合法单实例, 不是截断。
        return tmp, {"n_atoms": len(kept), "n_before_fragment_filter": before,
                     "fragment_truncated": truncated,
                     "single_instance": bool(single_instance)}
    return tmp

def _native_ligand_resname(pdb_path: str) -> Optional[str]:
    """返回 PDB 中有机结合配体的残基名(取原子数最多的 HETATM 组, 排除溶剂/缓冲剂)。
    供 self_dock_check / cocrystal_comparative_dock 在"未显式指定配体名"时
    也能找到晶体配体。无则返回 None。"""
    groups: Dict[str, int] = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("HETATM"):
                continue
            rn = ln[16:20].strip()
            if _is_solvent_resname(rn):
                continue
            groups[rn] = groups.get(rn, 0) + 1
    if not groups:
        return None
    # 合并 altLoc 前缀家族: 4 字符 rn 且"剥首字符"的形式独立存在(共享原子)
    # 或另有同尾 4 字符 rn(另一构象)时, 视为同一配体的 altLoc 拆分
    # (如 1M40: ACB4/BCB4/CB4 → CB4), 返回基名供 extract_named_ligand 构象感知提取。
    merged: Dict[str, int] = {}
    for rn, n in groups.items():
        if len(rn) == 4 and (rn[1:] in groups
                             or any(len(o) == 4 and o != rn and o[1:] == rn[1:]
                                    for o in groups)):
            merged[rn[1:]] = merged.get(rn[1:], 0) + n
        else:
            merged[rn] = merged.get(rn, 0) + n
    return max(merged, key=merged.get)


def resolve_crystal_ligand(pdb_path: str, ligand_name: str = ""
                           ) -> Tuple[Optional[str], str]:
    """晶体配体统一解析: manifest 俗称 → CCD 残基名回退(方案 B, 2026-10-02)。

    背景(批量重跑实测 0/60 提取成功的根因): manifest 的 ligand_name 常用
    俗称/配体库键, 而晶体里**同一分子**的 CCD 残基名不同 —— 7XTW/7XTV 的 MHET
    实为 C9C、8BRA 的 TPA 实为 UB7、6JTT 的 BHET 实为 C8X; 自造名(PBATdimer/
    BABTAB)更不可能命中。精确提取全灭 → selfdock 与 pose-vs-crystal 排序全灭。

    策略:
      1. ligand_name 非空 → 先按名精确提取(用户显式意图优先);
      2. 提取失败 → 回退自动挑(原子数最多的非溶剂 HETATM 组,
         _native_ligand_resname), 打 WARNING 说明回退与理由;
      3. 未指定名字 → 直接自动挑(原行为不变)。
    跨分子误用由上层既有守卫兜住: analyze 的 is_self_dock 走**重原子元素组成
    比对**(poses.py, M3)决定 RMSD 排序是否启用; self_dock_check 对晶体分子
    本身重对接, 天然自洽。即: 回退挑到"长得不一样的分子"(如 6QGA 的 J1K 酰胺
    vs 对接 MHET 酯, 组成 C10NO4 vs C10O5)不会产生假 RMSD 排序, 只会退回
    energy 排序 —— 与"配体不同的 cross-docking"同一处置路径。

    返回 (tmp_pdb 或 None, 实际使用的 resname)。tmp_pdb 由调用方用后删除;
    全空时 resname 为 ""(调用方据此打"未检测到晶体配体"日志)。
    """
    requested = (ligand_name or "").strip().upper()
    resname = requested or (_native_ligand_resname(pdb_path) or "")
    if not resname:
        return None, ""
    tmp = extract_named_ligand(pdb_path, resname, single_instance=True)
    if tmp is None and requested:
        auto = _native_ligand_resname(pdb_path)
        if auto and auto != requested:
            LOG.warning("⚠ 指定共晶配体 %s 不在受体 %s 中; 回退自动挑最大非溶剂 "
                        "HETATM 组 %s(同一分子的 CCD 名常不同, 如 MHET→C9C/TPA→UB7/"
                        "BHET→C8X; 跨分子误用由 is_self_dock 元素组成比对守卫)",
                        requested, os.path.basename(pdb_path), auto)
            resname = auto
            tmp = extract_named_ligand(pdb_path, resname, single_instance=True)
    return tmp, resname

__all__ = [
    "GEN3D_SANDBOX_TIMEOUT",
    "_SOLVENT_RESNAMES",
    "_build_3d_sandbox",
    "_is_solvent_resname",
    "_ligand_geom_fragmented",
    "_ligand_mol_to_pdbqt",
    "_native_ligand_info",
    "_native_ligand_resname",
    "extract_named_ligand",
    "prepare_ligand",
    "prepare_ligand_from_file",
    "resolve_crystal_ligand",
]
