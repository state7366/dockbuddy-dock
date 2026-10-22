# -*- coding: utf-8 -*-
"""obabel.py —— OpenBabel 可执行/数据目录探测与子进程环境准备(其它模块通过它调用 obabel)。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import glob
import os, shutil, subprocess, sys
from typing import Optional
from ._core import LOG


_OBABEL_CACHE: Optional[str] = None

def _obabel_supports_pdbqt(exe: str) -> bool:
    """探测 obabel 是否真的支持 pdbqt 格式。

    背景(2026-09-28 实机踩坑): 本机 C:\\Program Files\\OpenBabel-3.1.1\\obabel.EXE
    在 PATH 中且能执行, 但格式插件缺失/未注册 pdbqt —— 对任何 -O xxx.pdbqt 都报
    "Missing or unknown output file or format spec" 并退出码 1。仅探测可执行
    存在性会把整个受体预处理引到这个坏 exe 上, 因此必须做能力探测。
    用 `-:C`(单碳甲烷)转 pdbqt 做最小探测, 成功且无错误输出即认为可用。
    """
    import tempfile
    out = ""
    try:
        # 关键(2026-09-28/2026-10-10 实机踩坑): 不带 -O 输出文件时 obabel 只打印
        # Usage(rc=0), 会被误判为「支持 pdbqt」。必须 -O 到临时文件, 且校验产物
        # 真生成了 ROOT/ENDROOT/TORSDOF 特征骨架, 才算真正支持。
        fd, out = tempfile.mkstemp(suffix=".pdbqt")
        os.close(fd)
        r = subprocess.run([exe, "-:C", "-opdbqt", "-O", out],
                           capture_output=True, text=True, timeout=30)
        blob = (r.stdout or "") + (r.stderr or "")
        low = blob.lower()
        if r.returncode != 0:
            return False
        if "cannot write output format" in low or ("unknown" in low and "format" in low):
            return False
        if "ROOT" in blob and "TORSDOF" in blob:
            return True
        # 有些构建把结果只写文件、stdout 无骨架 → 重读产物
        if os.path.exists(out):
            with open(out, encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
            return "ROOT" in body and "TORSDOF" in body
        return False
    except Exception:
        return False
    finally:
        if out and os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass

def _find_obabel() -> str:
    """定位可用的 obabel 可执行文件(必须真正支持 pdbqt 格式)。

    优先级: 环境量 OBABEL_EXE > 当前 python venv > workbuddy venv > PATH。
    每个候选都做 pdbqt 能力探测(见 _obabel_supports_pdbqt), 首个通过者胜出;
    全部探测失败时回退到第一个存在的候选并打 WARNING(保持旧行为可用性)。
    结果进程级缓存, 避免每次转换重复探测。
    """
    global _OBABEL_CACHE
    if _OBABEL_CACHE:
        return _OBABEL_CACHE
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    env = os.environ.get("OBABEL_EXE")
    if env:
        cands.append(env)
    # 当前 python 解释器所在 venv(与调用方同环境, 最可能装好 openbabel)
    cands.append(os.path.join(sys.prefix, "Scripts", "obabel.EXE"))
    cands.append(os.path.join(sys.prefix, "bin", "obabel"))
    # workbuddy 托管 venv
    cands.append(os.path.expanduser("~/.workbuddy/binaries/python/envs/default/Scripts/obabel.exe"))
    cands.append(os.path.expanduser("~/.workbuddy/binaries/python/envs/default/bin/obabel"))
    cands.append(os.path.join(here, "..", "..", ".workbuddy", "binaries", "python",
                              "envs", "default", "Scripts", "obabel.exe"))
    cands.append(os.path.join(here, "..", "..", ".workbuddy", "binaries", "python",
                              "envs", "default", "bin", "obabel"))
    # PATH 最后(可能有坏安装, 如缺 pdbqt 插件的 OpenBabel 3.1.1)
    p = shutil.which("obabel")
    if p:
        cands.append(p)

    seen, existing = set(), []
    for c in cands:
        if not c:
            continue
        key = os.path.normcase(os.path.abspath(c)) if os.path.exists(c) else c
        if key in seen or not os.path.exists(c):
            continue
        seen.add(key)
        existing.append(os.path.abspath(c))
    for c in existing:
        if _obabel_supports_pdbqt(c):
            _OBABEL_CACHE = c
            return c
    if existing:
        LOG.warning(f"所有候选 obabel 均未通过 pdbqt 能力探测, 回退使用 {existing[0]} "
                    f"(可能不支持 pdbqt, 受体/配体转换会失败)")
        _OBABEL_CACHE = existing[0]
        return _OBABEL_CACHE
    raise RuntimeError("找不到 obabel 可执行文件; 请确认 OpenBabel 已安装并在 PATH 中")

def _obabel_data_dir() -> str:
    """返回 OpenBabel 数据目录(含 UFF.prm/phmodel.txt 等)。obabel/Python 绑定在
    部分环境(尤其经 subprocess 调用时)无法自动定位数据目录, 导致键感知/3D 生成
    失败甚至受体被截断写出 -> 必须显式设置 BABEL_DATADIR。

    布局(2026-10-10 云实例实测): openbabel-wheel 的数据在
    ``openbabel/share/openbabel/3.1.0``(非 ``bin/data``), 旧代码只查 bin/data
    导致返回空串 -> --addhydrogens 失败。此处按「含 UFF.prm/phmodel.txt」判定。
    """
    env = os.environ.get("BABEL_DATADIR", "")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    cands: list = []
    # 1) openbabel python 绑定目录: bin/data + share/openbabel/<ver>
    try:
        from openbabel import openbabel as _ob
        mod_dir = os.path.dirname(os.path.abspath(_ob.__file__))
        cands.append(os.path.join(mod_dir, "bin", "data"))
        cands += sorted(glob.glob(os.path.join(mod_dir, "share", "openbabel", "*")))
    except Exception:
        pass
    # 2) 常见 venv 布局(workbuddy 托管 venv)
    here = os.path.dirname(os.path.abspath(__file__))
    venv_roots = [
        os.path.join(here, "..", "..", ".workbuddy", "binaries", "python",
                     "envs", "default", "Lib", "site-packages", "openbabel"),
        os.path.expanduser("~/.workbuddy/binaries/python/envs/default/Lib/site-packages/openbabel"),
        os.path.join(sys.prefix, "Lib", "site-packages", "openbabel"),
        os.path.join(sys.prefix, "lib", "python" + ".".join(map(str, sys.version_info[:2])),
                     "site-packages", "openbabel"),
    ]
    for vr in venv_roots:
        cands.append(os.path.join(vr, "bin", "data"))
        cands += sorted(glob.glob(os.path.join(vr, "share", "openbabel", "*")))
    # 3) 系统 share/openbabel 布局(apt 装的 openbabel)
    cands += sorted(glob.glob("/usr/share/openbabel/*"))
    cands += sorted(glob.glob("/usr/local/share/openbabel/*"))

    # 判定: 目录须含 UFF.prm / phmodel.txt 之类的力场数据, 才算真数据目录
    for c in cands:
        if os.path.isdir(c) and (os.path.exists(os.path.join(c, "UFF.prm"))
                                 or os.path.exists(os.path.join(c, "phmodel.txt"))):
            return os.path.abspath(c)
    # 兜底: 任一存在的目录(可能数据不全, 但比空串好)
    for c in cands:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return ""

def _obabel_env() -> dict:
    """返回带 BABEL_DATADIR 的 subprocess 环境, 确保 obabel 能定位力场/键数据。"""
    env = dict(os.environ)
    d = _obabel_data_dir()
    if d:
        env["BABEL_DATADIR"] = d
    return env

__all__ = [
    "_OBABEL_CACHE",
    "_find_obabel",
    "_obabel_data_dir",
    "_obabel_env",
    "_obabel_supports_pdbqt",
]
