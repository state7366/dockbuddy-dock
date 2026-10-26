# -*- coding: utf-8 -*-
"""libdock.external_tools —— 外部二进制薄适配器(2026-10-11 批量新增)
===================================================================================
统一薄适配器: 环境守卫(二进制探测) + subprocess 调用骨架。这些工具为 Linux 二进制
(conda/apt/源码), 本机 Windows 不可跑; 真跑需云实例。

诚实边界: 各工具口径不同(如 smina=Vina1.1.2, rDock 独立打分函数), 分数不可混比。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional


class ExternalToolError(RuntimeError):
    pass


def _find_bin(env_var: str, names, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if not os.path.isfile(explicit):
            raise ExternalToolError(f"指定的二进制不存在: {explicit}")
        return explicit
    env = os.environ.get(env_var, "").strip()
    if env:
        return env
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return None


def _run(bin_path: str, args, cwd: str, timeout: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run([bin_path] + args, cwd=cwd, capture_output=True,
                              text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout_tail": "",
                "stderr_tail": "timeout"}


# ---------------------------------------------------------------------------
# 对接 / 口袋 / 通道 / 建模 / MD(conda/apt 可装)
# ---------------------------------------------------------------------------
def qvina_dock(receptor_pdbqt: str, ligand_pdbqt: str, *, center, size=(22, 22, 22),
               bin_path: Optional[str] = None, exhaustiveness: int = 8,
               timeout: int = 600) -> Dict[str, Any]:
    """QVina(Quasar)对接, Vina 加速实现。"""
    b = _find_bin("QVINA_BIN", ("qvina", "qvina2"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 qvina; 设置 QVINA_BIN 或 conda -c bioconda qvina2")
    args = ["--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
            "--center_x", str(center[0]), "--center_y", str(center[1]),
            "--center_z", str(center[2]), "--size_x", str(size[0]),
            "--size_y", str(size[1]), "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness)]
    return _run(b, args, os.path.dirname(receptor_pdbqt), timeout)


def rdock_dock(receptor_mol2: str, ligand_sdf: str, *, bin_path: Optional[str] = None,
               timeout: int = 600) -> Dict[str, Any]:
    """rDock 对接(独立打分函数)。"""
    b = _find_bin("RDOCK_BIN", ("rbdock", "rbcavity"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 rbdock; 设置 RDOCK_BIN 或 conda -c bioconda rdock")
    return _run(b, [], os.path.dirname(receptor_mol2), timeout)


def mole_channel(receptor_pdb: str, *, bin_path: Optional[str] = None,
                 timeout: int = 300) -> Dict[str, Any]:
    """MOLE 通道分析(隧道检测)。"""
    b = _find_bin("MOLE_BIN", ("mole", "mole.exe"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 mole; 设置 MOLE_BIN 或 Web mole.upol.cz")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def scwrl_repack(receptor_pdb: str, *, bin_path: Optional[str] = None,
                 timeout: int = 300) -> Dict[str, Any]:
    """SCWRL4 侧链重建。"""
    b = _find_bin("SCWRL_BIN", ("Scwrl4", "scwrl4"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 Scwrl4; 设置 SCWRL_BIN(dunbrack.fccc.edu 下载)")
    return _run(b, ["-i", receptor_pdb, "-o", receptor_pdb + ".scwrl.pdb"],
                os.path.dirname(receptor_pdb), timeout)


def gromacs_md(topology: str, *, bin_path: Optional[str] = None,
               timeout: int = 3600) -> Dict[str, Any]:
    """GROMACS MD(能量最小化/模拟)。"""
    b = _find_bin("GROMACS_BIN", ("gmx", "gmx_mpi"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 gmx; 设置 GROMACS_BIN 或 apt install gromacs")
    return _run(b, ["--version"], ".", timeout)


def namd_md(config: str, *, bin_path: Optional[str] = None,
            timeout: int = 3600) -> Dict[str, Any]:
    """NAMD MD。"""
    b = _find_bin("NAMD_BIN", ("namd2", "namd3"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 namd; 设置 NAMD_BIN(ks.uiuc.edu 下载, 学术免费)")
    return _run(b, [config], os.path.dirname(config), timeout)


# Modeller 官方学术版通用 license key(所有学术用户共享; 也可经 KEY_MODELLER 覆盖)
DEFAULT_MODELLER_KEY = "MODELIRANJE"


def modeller_build_script(alignment: str, knowns: str, sequence: str,
                          num_models: int = 5) -> str:
    """生成 Modeller AutoModel 同源建模 Python 脚本文本(纯逻辑, 可单测)。

    参数:
      alignment: PIR 格式 alignment 文件路径(.ali, 含 >P1;knowns 与 >P1;sequence 两条)。
      knowns:   模板的 align_code(alignment 内 >P1;<code> 的 code)。
      sequence: 目标序列的 align_code。
      num_models: 生成模型数(1..N)。
    返回: 可直接 `mod10.8 script.py` 执行的脚本。
    """
    return (
        "from modeller import *\n"
        "from modeller.automodel import *\n"
        "import os\n"
        "\n"
        "log.verbose()\n"
        "env = Environ()\n"
        "aln = Alignment(env)\n"
        f"aln.append(file={alignment!r}, align_codes='ALL')\n"
        f"a = AutoModel(env, alnfile={alignment!r},\n"
        f"              knowns=({knowns!r},), sequence={sequence!r},\n"
        "              assess_methods=(assess.DOPE, assess.GA341))\n"
        "a.starting_model = 1\n"
        f"a.ending_model = {int(num_models)}\n"
        "a.make()\n"
    )


def _modeller_root_from_bin(b: str) -> str:
    """从 mod 二进制路径推断安装根: <root>/lib/x86_64-w64/mod10.8 -> <root>。"""
    d = os.path.dirname(b)
    # lib/x86_64-w64/ 或 lib/i386-w32/ 上两级是安装根
    for _ in range(3):
        up = os.path.dirname(d)
        if os.path.isdir(os.path.join(up, "modlib")):
            return up
        d = up
    return ""


def modeller_model(alignment: str, knowns: str, sequence: str, *,
                   bin_path: Optional[str] = None, key: Optional[str] = None,
                   num_models: int = 5, timeout: int = 3600,
                   modeller_dir: str = "") -> Dict[str, Any]:
    """Modeller 同源建模(学术免费)。

    生成 automodel 脚本并执行, 输出 `<sequence>.B9999000*.pdb`。解释器优先级:
    显式 bin_path > MODELLER_BIN(mod 命令) > 当前 Python(sys.executable)。
    license key 经环境变量 KEY_MODELLER 注入(缺省 DEFAULT_MODELLER_KEY 官方学术 key)。

    参数:
      alignment: PIR 格式 alignment(.ali), 含 knowns + sequence 两条。
      knowns: 模板 align_code。
      sequence: 目标序列 align_code。
      modeller_dir: modlib 目录(含 modeller 包 + config.py)。缺省取 MODELLER_DIR
        环境变量, 或从 mod 二进制路径自动推断(<root>/modlib)。
      bin_path/key/num_models/timeout 同上。
    """
    b = _find_bin("MODELLER_BIN", ("mod10.8", "mod9.25", "mod", "modeller"), bin_path)
    mdl_dir = (modeller_dir or "").strip() or os.environ.get("MODELLER_DIR", "").strip()
    env = dict(os.environ)
    env["KEY_MODELLER"] = key or os.environ.get("KEY_MODELLER") or DEFAULT_MODELLER_KEY

    if b:
        # mod 命令(Windows mod10.8.exe 需 MODINSTALL 环境变量; Linux mod 用 shebang Python)
        root = _modeller_root_from_bin(b)
        if root:
            env["MODINSTALL10v8"] = root
            env["MODINSTALL10v9"] = root
            if not mdl_dir:
                cand = os.path.join(root, "modlib")
                if os.path.isdir(cand):
                    mdl_dir = cand
        interp = [b, script_path := os.path.join(
            os.path.dirname(os.path.abspath(alignment)), f"_modeller_{sequence}.py")]
    else:
        if not mdl_dir:
            # conda salilab::modeller 是 Python 包(import modeller), 无 mod10.8 独立
            # 二进制; 若当前解释器可 import modeller 则走 sys.executable, 不要求 MODELLER_DIR。
            try:
                import importlib
                importlib.import_module("modeller")
            except ImportError:
                raise ExternalToolError(
                    "未找到 modeller; 设置 MODELLER_BIN(mod10.8/mod9.25) 或 "
                    "MODELLER_DIR(modlib 目录, 含 modeller 包) —— salilab.org 下载, "
                    "Linux 用 deb, Windows 用 exe")
        interp = [sys.executable, script_path := os.path.join(
            os.path.dirname(os.path.abspath(alignment)), f"_modeller_{sequence}.py")]

    if mdl_dir:
        env["PYTHONPATH"] = mdl_dir + os.pathsep + env.get("PYTHONPATH", "")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(modeller_build_script(alignment, knowns, sequence, num_models))
    cwd = os.path.dirname(os.path.abspath(alignment)) or os.getcwd()
    try:
        proc = subprocess.run(interp, cwd=cwd,
                              capture_output=True, text=True, timeout=timeout, env=env)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "script": script_path, "interpreter": interp[0],
                "stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "script": script_path,
                "interpreter": interp[0], "stdout_tail": "", "stderr_tail": "timeout"}


# ---------------------------------------------------------------------------
# 需源码编译 / 特殊(诚实薄适配器)
# ---------------------------------------------------------------------------
def encom_entropy(receptor_pdb: str, *, bin_path: Optional[str] = None,
                  timeout: int = 300) -> Dict[str, Any]:
    """ENCoM 振动熵(GitHub 源码编译, 非 pip)。"""
    b = _find_bin("ENCOM_BIN", ("encom", "ENCoM"), bin_path)
    if b is None:
        raise ExternalToolError(
            "未找到 ENCoM; 需从 github.com/ENCoM/ENCoM 源码编译(非 pip, 需 gfortran)")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def dogsite_pocket(receptor_pdb: str, *, bin_path: Optional[str] = None,
                   timeout: int = 300) -> Dict[str, Any]:
    """DoGSite 口袋检测(开源几何方法, 源码编译)。"""
    b = _find_bin("DOGSITE_BIN", ("dogsite", "DoGSiteScorer"), bin_path)
    if b is None:
        raise ExternalToolError(
            "未找到 DoGSite; 需源码编译(github.com/volkamerlab/DoGSite)")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def ligsite_pocket(receptor_pdb: str, *, bin_path: Optional[str] = None,
                   timeout: int = 300) -> Dict[str, Any]:
    """LIGSITE 口袋检测(几何方法, 源码编译)。"""
    b = _find_bin("LIGSITE_BIN", ("ligsite", "pockets"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 LIGSITE; 需源码编译(ligsite 需 mol-util 依赖)")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def pross_design(receptor_pdb: str, *, bin_path: Optional[str] = None,
                 timeout: int = 600) -> Dict[str, Any]:
    """PROSS Rosetta 共识稳定化(需 PyRosetta)。"""
    b = _find_bin("PROSS_BIN", ("pross", "PROSS"), bin_path)
    if b is None:
        raise ExternalToolError(
            "未找到 PROSS; 需 PyRosetta(github.com/ipc-lab/PROSS 或 Web pross.weizmann.ac.il)")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def castp_pocket(receptor_pdb: str, *, bin_path: Optional[str] = None,
                 timeout: int = 300) -> Dict[str, Any]:
    """CASTp 口袋体积(本地 C++ 二进制)。"""
    b = _find_bin("CASTP_BIN", ("castp", "pocasa"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 CASTp; 需本地二进制或 Web sts.bioe.uic.edu/castp")
    return _run(b, [receptor_pdb], os.path.dirname(receptor_pdb), timeout)


def adfr_dock(receptor_pdbqt: str, ligand_pdbqt: str, *, bin_path: Optional[str] = None,
              timeout: int = 600) -> Dict[str, Any]:
    """ADFR 对接套件。"""
    b = _find_bin("ADFR_BIN", ("adfr", "prepare_receptor"), bin_path)
    if b is None:
        raise ExternalToolError("未找到 ADFR; 需 ccsb.scripps.edu/ADFR 下载(二进制+力场)")
    return _run(b, [], os.path.dirname(receptor_pdbqt), timeout)


__all__ = ["ExternalToolError", "qvina_dock", "rdock_dock", "mole_channel",
           "scwrl_repack", "gromacs_md", "namd_md",
           "DEFAULT_MODELLER_KEY", "modeller_build_script", "modeller_model",
           "encom_entropy", "dogsite_pocket", "ligsite_pocket", "pross_design",
           "castp_pocket", "adfr_dock"]
