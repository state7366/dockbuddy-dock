# -*- coding: utf-8 -*-
"""_core.py —— libdock 包的公共底座

承载: OpenBabel 导入与数据目录修正(BABEL_DATADIR)、OpenBabel 日志级别、
模块级日志器 LOG, 以及被多个子模块共用的常量。

拆分原则: 底座不依赖包内任何子模块(依赖图的第 0 层), 避免循环导入。
"""
import os
import logging

from openbabel import openbabel as ob

# openbabel-wheel(Windows)把 BABEL_DATADIR 指向无数据的 share/ 目录,
# 导致 UFF/MMFF94 力场参数找不到(gen3d 不可用); 此处修正到包内 bin/data/。
# 必须在 import openbabel 之后(wheel __init__ 覆写该变量)、首次使用 OB 数据前执行。
try:
    from dockbuddy_dock.tools import fix_babel_datadir  # noqa: E402
    fix_babel_datadir()
except Exception:
    pass

# OpenBabel 警告降噪: 受体 pdbqt 每次读入都会刷 "Failed to kekulize aromatic bonds"
# 警告——蛋白 pdbqt 无键级信息, 属良性噪声。默认只放行 error 级;
# 调试时设 DOCK_OB_VERBOSE=1 恢复警告。
if os.environ.get("DOCK_OB_VERBOSE", "0") != "1":
    try:
        ob.obErrorLog.SetOutputLevel(ob.obError)
    except Exception:
        pass

LOG = logging.getLogger('libdock')
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [libdock] %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

LOG = logging.getLogger("libdock")

# PDBQT 原子类型 -> 元素符号。
# 修复(2026-10-02): 旧表把 "CA" 映射成 "C", 让**钙离子被当成碳**计入元素组成
# 比对(self-dock vs cross-docking 的身份判定会被带金属晶体误导);
# 且 "Cl"/"BR" 是大小写错配(调用侧用 typ.upper() 查表 → "CL"/"BR" 永远查不到,
# 是两处死键)。现按 PDBQT 官方类型表修正, 大小写用大写键。
_TYPE2ELEM = {
    "A": "C", "C": "C", "CA": "Ca", "N": "N", "NA": "N", "NS": "N", "OA": "O",
    "O": "O", "OS": "O", "H": "H", "HD": "H", "HS": "H", "S": "S", "SA": "S",
    "SG": "S", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "MG": "Mg",
    "ZN": "Zn", "FE": "Fe", "MN": "Mn", "CA2": "Ca", "CU": "Cu",
    # 2026-10-02(M6): 补全常见金属/类金属 —— 原表缺 HG 时判氢 fallback 取首字符,
    # "HG"[:1]=="H" 把汞当氢排除(与原子名陷阱同源的类型码陷阱)。核验侧
    # tools/oneoff/verify_fixed.py 的 _AD_TYPE2ELEM 同步补充。
    "HG": "Hg", "CD": "Cd", "NI": "Ni", "CO": "Co", "SE": "Se", "SI": "Si",
    "K": "K", "LI": "Li",
}
# 旧键名的向后兼容别名(混合大小写旧键, 保留避免外部脚本/旧产物代码 KeyError)。
# 2026-10-03 修复: 原 update({"Cl": "Cl", "BR": "BR"}) 的 "BR" 会把上方大写键
# "BR": "Br" 覆盖成非规范写法 "BR" —— 元素符号大小写敏感的比对(与晶体侧
# "Br" 比较)将永不相等, 含溴配体的身份判定/connectivity 整类假 FAIL。值必须
# 仍是规范元素符号。
_TYPE2ELEM.update({"Cl": "Cl", "Br": "Br"})

# 向后兼容: 拆分前 libdock.* 命名空间可见的导入名(契约等价用)
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
# 与上方 :17-21 的 try/except 同口径(2026-10-03): 该裸 import 在未经 bootstrap_path
# (`sys.path` 无 pipeline/)直接 `import libdock` 时抛 ModuleNotFoundError 且无人捕获,
# 吞错防护形同虚设。符号可见性在可导入时保持(契约面不变)。
# 2026-10-03 第六轮审查 S1: 原修复无效 —— except 吞错后 `__all__` 仍声明该名,
# `from ._core import *` 按 __all__ 逐名 getattr 会照样 ImportError(只是换了报错
# 形式)。失败分支显式绑 None: 符号存在且可 import *,调用方在无 common 场景下
# 得到明确的 AttributeError 而非 import 期崩溃。
try:
    from dockbuddy_dock.tools import fix_babel_datadir  # noqa: E402,F401
except Exception:
    fix_babel_datadir = None  # type: ignore[assignment]


def env_float(name: str, default: float) -> float:
    """模块级常量的 env 数值读取(2026-10-03 第六轮审查 C7)。

    此前 `float(os.environ.get(X) or d)` 对非数字 env(如 "abc"/纯空白)在
    **import 期** ValueError 裸栈且无变量名上下文 —— 统一为: 空串回退缺省、
    非数字回退缺省并打 WARNING(带变量名, 可定位)。
    """
    raw = os.environ.get(name) or ""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        LOG.warning("环境变量 %s=%r 非数字, 回退缺省 %g", name, raw, default)
        return default



def ad_type_of(line: str) -> str:
    """返回 PDBQT 行的 AutoDock 类型码, 已 upper。

    2026-10-02 修复: 不能硬编码列号。标准 PDBQT 把类型放在 1-based 77-78(0-based
    ``ln[76:78]``), 但 **Vina 输出的结果文件整行只有 79 列、类型落在 0-based
    ``[77:79]``**(实测 9T6Y 的 `ATOM     29 Fe1  HEM A 201 ... +0.000 Fe`, 长度 79)。
    用固定列号读会把 **Fe 读成 F**、**OA 读成 O**, 于是元素组成对不上、键图同构失败、
    金属辅因子的自对接 RMSD 整条判据假 FAIL。

    取"行尾最后一个字段"是各 writer(OB / Vina / AutoDockTools)的共同约定 ——
    ATOM/HETATM 行的类型永远是最后一个 token。
    """
    toks = line.rstrip().split()
    return toks[-1].upper() if toks else ""


__all__ = ['ob', 'LOG', 'LOG', '_TYPE2ELEM', 'ad_type_of', 'Any', 'Dict', 'List',
           'Optional', 'Tuple', 'dataclass', 'field', 'fix_babel_datadir', 'env_float']
