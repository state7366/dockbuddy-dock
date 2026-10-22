# -*- coding: utf-8 -*-
"""配体侧粒子参数(优化项 **A2** 的配体部分, 2026-10-03)
==============================================================
为什么需要这个文件
------------------
``libmutate --relax md`` 的核心是 OpenMM + amber14, 但那条路径只处理**标准氨基酸**,
突变残基换掉后依然落在 amber14 的模板里。A2(对接后重评分)要处理的配体是**任意有机
小分子**(聚酯寡聚体、GT 基准里的 CB4/FTA/4MZ…), 而本机环境::

    openmm 8.6.1 / openbabel 3.2.1 / pdbfixer 1.12.0   ← 仅此三件套

没有任何离线可用的小分子力场(openmmforcefield / openff / rdkit / AMBER
antechamber 全都不在), 而 ``app.ForceField("amber14-all.xml").createSystem()``
一旦拓扑里出现 ``UNL`` 这类非模板残基就必然抛 ``No template found for residue``
(实测消息里连"和 DLPA 最像但缺 50 个 H"这种误判模板都找到了)。

所以配体**不能走 createSystem**, 只能自己给每个原子一个 ``(q, sigma, epsilon)``。
这里的口径(在三处被反复问到, 统一写死):
1. **电荷**: 直接取对接本身用的那份 —— 从 run 目录的 ``*.pdbqt`` 按**原子名**读
   最后一列。这样"打分用的电荷"和"对接用的电荷"是同一份, 而不是另起一套。
2. **LJ**: UFF 的**元素型**参数(R/eps, Rappé 1992), 单位 Å / kcal·mol⁻¹,
   交给 OpenMM 时再换成 nm / kJ·mol⁻¹。元素型而非杂化型, 因为 PDBQT 里的
   AutoDock 类型(C/A/N/OA/…)已经丢了杂化细节, 反而元素型更稳。
3. **氢**: 由 obabel ``--addhydrogens`` 补(必须显式给对 BABEL_DATADIR, 否则 op
   插件静默失效、一个 H 都不加 —— 见 :func:`add_ligand_hydrogens`), 氢电荷取 0
   (极性氢的静电贡献被低估, 已知且会在 meta 里标注)。
4. OBC 半径/屏蔽 :func:`libdock.solvents` 的同一个换算系数, 不许第二处漂移。

这套东西是**排序量(re-ranking)不是自由能预测值**: 没有熵项、没有构象平均、没有
受体/配体松弛(只做单点), 而且配体 LJ 是元素型近似。别把它写成 "ΔG = -x kcal/mol"。
"""
import logging
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from .obabel import _find_obabel, _obabel_env

LOG = logging.getLogger("libdock.ligparam")

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1

# 元素质量(g/mol)
ELEMENT_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.90,
    "SE": 78.96, "SI": 28.0855,
}

# UFF 元素型 LJ: (R / Å, epsilon / kcal·mol⁻¹)
UFF_ELEMENT_LJ = {
    "H":  (2.426, 0.0208),
    "C":  (3.400, 0.0860),
    "N":  (3.200, 0.0680),
    "O":  (2.960, 0.0600),
    "F":  (2.820, 0.0500),
    "P":  (3.740, 0.2000),
    "S":  (3.550, 0.2500),
    "CL": (3.430, 0.2650),
    "BR": (3.620, 0.4200),
    "I":  (3.980, 0.5500),
    "SE": (3.950, 0.5500),
    "SI": (3.660, 0.1400),
}
_DEFAULT_LJ = (3.400, 0.1000)
_DEFAULT_MASS = 12.011


def _norm_element(elem: str) -> str:
    # 2026-10-04 全盘审计: 原只豁免 CL/BR —— SE→S/SI→S(表内条目成死键),
    # 更糟 FE→F(血红素铁拿氟的 LJ)/CA→C/NA→N(钙钠被当碳氮)。两字母已知
    # 元素一律不截断: 查表 miss 走 _DEFAULT 也比拿错元素的参数合理。
    e = (str(elem) or "").strip().upper()
    if e in _TWO_LETTER_ELEMENTS:
        return e
    return e[:1]


# 已知两字母元素(含表外金属: 截首字符会静默变成另一个元素, 见上)
_TWO_LETTER_ELEMENTS = {
    "CL", "BR", "SE", "SI", "FE", "ZN", "MG", "MN", "CA", "NA", "CU",
    "NI", "CO", "CD", "HG", "LI", "AL", "AG", "AU", "PT", "PD", "CS",
    "BA", "SR", "RB",
}


def element_lj(element: str) -> Tuple[float, float]:
    """元素 → ``(sigma_nm, epsilon_kJ_per_mol)``(OpenMM NonbondedForce 单位)。"""
    e = _norm_element(element)
    r_ang, eps_kcal = UFF_ELEMENT_LJ.get(e, _DEFAULT_LJ)
    return r_ang * ANGSTROM_TO_NM, eps_kcal * KCAL_TO_KJ


def element_mass(element: str) -> float:
    return ELEMENT_MASSES.get(_norm_element(element), _DEFAULT_MASS)


def add_ligand_hydrogens(pose_pdb: str, out_pdb: str,
                         timeout: int = 180) -> Optional[str]:
    """用 obabel 给 pose 补氢, 返回补氢后的 PDB 路径; 失败返回 None(不抛)。

    **必须走 ``_obabel_env()``**: openbabel wheel 会把 BABEL_DATADIR 指向一个
    只有 splash.png 的 ``share/openbabel/<ver>``, op 插件(加氢/电荷)因此加载
    不到并**静默跳过** —— 表现是输出文件和输入一样、一个 H 都没加, 很容易被误判成
    "配体不需要氢"。指对到 ``bin/data`` 后立刻生效(实测 O6→H2 正常出现)。

    注意 obabel 的 ``-O`` 必须显式给, 只写 ``-opdb`` 输出后缀会报
    ``Cannot open <file>``。
    """
    if not os.path.exists(pose_pdb):
        LOG.warning("ligparam: pose 文件不存在 %s", pose_pdb)
        return None
    exe = _find_obabel()
    if not exe or not os.path.exists(exe):
        LOG.warning("ligparam: 找不到 obabel, 无法给配体补氢")
        return None
    try:
        subprocess.run([exe, "-ipdb", pose_pdb, "-O", out_pdb, "--addhydrogens"],
                       env=_obabel_env(), capture_output=True, text=True,
                       timeout=timeout, check=True)
    except Exception as e:                                    # pragma: no cover
        LOG.warning("ligparam: 配体补氢失败(%s), 将按无氢处理", e)
        return None
    return out_pdb if os.path.exists(out_pdb) else None


def charges_from_pdbqt(pdbqt_path: str) -> Dict[str, float]:
    """从 PDBQT 读 ``{原子名: 电荷}``。

    PDBQT 行末两列是"电荷 + AutoDock 类型", 例如
    ``... 1.00  0.00    +0.076 A`` → 电荷取倒数第二列。按**原子名**存, 供补氢后
    的 pose(重原子名不变、H 在末尾)按下标对齐时用。
    """
    out: Dict[str, float] = {}
    if not pdbqt_path or not os.path.exists(pdbqt_path):
        return out
    try:
        with open(pdbqt_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.startswith(("ATOM", "HETATM")):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                try:
                    q = float(parts[-2])
                except ValueError:
                    continue
                out[ln[12:16].strip()] = q
    except Exception as e:                                    # pragma: no cover
        LOG.warning("ligparam: 读电荷失败(%s)", e)
    return out


def ligand_particles(atoms: Sequence[Sequence[float]],
                     charge_map: Optional[Dict[str, float]] = None) -> List[Tuple[float, float, float, float]]:
    """pose 原子 → 粒子参数 ``[(q, sigma_nm, eps_kJ, mass), ...]``。

    ``charge_map`` 按**原子名**给重原子配电荷, 没给到的(含所有氢)记 0.0。
    """
    out = []
    for a in atoms:
        elem = _norm_element(a[3])
        q = 0.0
        if charge_map and a[4] in charge_map:
            q = float(charge_map[a[4]])
        sigma, eps = element_lj(elem)
        out.append((q, sigma, eps, element_mass(elem)))
    return out


def ligand_bonds(atoms: Sequence[Sequence[float]],
                 cov_radii: Optional[Dict[str, float]] = None) -> List[Tuple[int, int]]:
    """按共价半径判键(纯几何, 不需要化学感知), 返回 ``[(i, j), ...]``。

    为什么要自己判: 配体走的是我们自己拼的 System, 不走 ``createSystem``, 拿不到
    amber 的键表; 而 ``createExceptionsFromBonds`` 必须拿到键才能生成 1-4 例外项,
    否则配体内部的 1-4 库仑会被当成全强度作用。
    """
    try:
        from .ligand import _COV_RADII
    except Exception:                                         # pragma: no cover
        _COV_RADII = {}
    radii = cov_radii or _COV_RADII or {}
    n = len(atoms)
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = ((atoms[i][0] - atoms[j][0]) ** 2 + (atoms[i][1] - atoms[j][1]) ** 2
                 + (atoms[i][2] - atoms[j][2]) ** 2) ** 0.5
            if d < 1e-6:
                continue
            ri = radii.get(str(atoms[i][3]), 0.77)
            rj = radii.get(str(atoms[j][3]), 0.77)
            if d <= 1.3 * (ri + rj):
                bonds.append((i, j))
    return bonds


__all__ = [
    "KCAL_TO_KJ", "ANGSTROM_TO_NM", "ELEMENT_MASSES", "UFF_ELEMENT_LJ",
    "element_lj", "element_mass", "add_ligand_hydrogens",
    "charges_from_pdbqt", "ligand_particles", "ligand_bonds",
]
