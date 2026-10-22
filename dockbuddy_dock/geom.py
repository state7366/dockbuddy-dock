# -*- coding: utf-8 -*-
"""geom.py —— 纯几何/向量运算: 点积叉积、旋转对齐、坐标变换与 PDBQT 刚体旋转。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import math, re
from typing import Optional, Tuple
from ._core import LOG

def _pose_heavy_rmsd(atomsA, atomsB) -> Optional[float]:
    """两 pose 的重原子 RMSD(Å), 按原子序对齐(同一配体, 顺序一致)。
    用于优化项E: top1 与 top2 pose 分歧大 -> 对接未收敛。无法比较返回 None。

    C3(2026-10-02): 原子数不等(同一配体的缓存保证下不应发生)时打 WARN 再按
    min 截断 —— 静默截断会掩盖"比的根本不是同一分子"的事故。"""
    pa = [a[:3] for a in atomsA if a[3] != "H"]
    pb = [a[:3] for a in atomsB if a[3] != "H"]
    n = min(len(pa), len(pb))
    if n == 0:
        return None
    if len(pa) != len(pb):
        LOG.warning(f"_pose_heavy_rmsd: 两 pose 重原子数不等({len(pa)} vs {len(pb)}), "
                    f"按较短者截断 —— 请确认比较的是同一配体")
    s = 0.0
    for i in range(n):
        dx = pa[i][0] - pb[i][0]; dy = pa[i][1] - pb[i][1]; dz = pa[i][2] - pb[i][2]
        s += dx * dx + dy * dy + dz * dz
    return math.sqrt(s / n)


def kabsch_rmsd(P, Q) -> float:
    """Kabsch 最优刚体叠合后 P/Q 的 RMSD(Å)。P/Q: 等长 (N,3) 数组/列表。

    C8(2026-10-02): libdock 侧 Kabsch 的**单一实现**(poses.py 的两处与
    selfdock.py 的内联拷贝收敛到这里; rmsd.py 与 tools/oneoff/verify_fixed.py 保持
    独立拷贝是**有意设计** —— 前者承诺不 import openbabel, 后者是核验器的
    独立性原则(核验工具不得依赖被核验代码), 均有跨实现一致性测试锁定
    (tests/test_unit_rmsd.py::test_kabsch_cross_implementation_agreement)。

    det 修正项拒绝镜像: 反射变换(非旋转)不可叠合, RMSD 保持非零。
    """
    import numpy as np
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"原子数不一致: {P.shape} vs {Q.shape}")
    R, t = kabsch_transform(P, Q)
    Qr = (R @ Q.T).T + t
    return float(np.sqrt(((P - Qr) ** 2).sum(axis=1).mean()))


def kabsch_transform(P, Q):
    """Kabsch 最优刚体变换: 求 ``R``/``t`` 使 ``R @ q + t ≈ p``(把 Q 叠到 P 上)。

    返回 ``(R, t)`` —— ``R`` 是 (3,3) 旋转矩阵, ``t`` 是 (3,) 平移; 失败抛
    ``ValueError``(与 :func:`kabsch_rmsd` 一致, 不静默返回 None)。

    **为什么要单独暴露变换本身而不只是 RMSD**(2026-10-04, A5 修缺陷时发现):
    同一批 AlphaFold 模型(m0..m4)常常**不在同一坐标系** —— 实测某体系 5 个
    模型未叠合 Cα RMSD 高达 9.4~25.0 Å, 而叠合后只有 0.07~0.11 Å(构象几乎
    相同, 只是刚体位姿不同)。这类场景下:
      * 只拿 RMSD 会得出"模型差异巨大"的错误结论;
      * 更糟的是**跨模型传坐标** —— 把参照模型的盒心直接用到另一个模型上,
        会在离口袋 20 Å 的地方开搜(静默产出垃圾对接结果)。
    正确做法是拿到 ``(R, t)`` 之后, 把盒心/pose 坐标用
    :func:`apply_rigid` 显式变换到目标坐标系。

    det 修正项拒绝镜像(与 ``kabsch_rmsd`` 同)。
    """
    import numpy as np
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    # N>=3 时旋转才是良定的; N<3(两点/一点)退化为欠定, 但不报错 ——
    # kabsch_rmsd 的历史行为就是允许任意 N, 这里不能收紧(会破坏既有调用方)。
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"需要等长的 (N,3) 坐标, 实得 {P.shape} vs {Q.shape}")
    if P.shape[0] < 3:
        LOG.debug("kabsch_transform: 只有 %d 个点, 旋转欠定(结果不唯一)", P.shape[0])
    Pm, Qm = P.mean(axis=0), Q.mean(axis=0)
    H = (P - Pm).T @ (Q - Qm)
    U, _S, Vt = np.linalg.svd(H)
    det = float(np.linalg.det(Vt.T @ U.T))
    # 2026-10-04 全盘审计: det=0(共线/重复点的退化输入)时 sign(0)=0 →
    # R=U@diag([1,1,0])@Vt 是奇异矩阵, 静默产出垃圾变换。本函数是跨坐标系
    # 传坐标的唯一正确通道, 退化输入必须显式告警(不 raise: 历史行为容忍任意 N)。
    if det == 0.0:
        LOG.warning("kabsch_transform: 退化输入(det=0, 共线/重复点?), "
                    "变换不唯一, 结果不可信")
    d = float(np.sign(det)) or 1.0
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    t = Pm - R @ Qm
    return R, t


def apply_rigid(points, R, t):
    """把 :func:`kabsch_transform` 得到的刚体变换应用到任意坐标上。

    ``points`` 可以是单个 (3,) 点或 (N,3) 数组 —— 盒心是前者, pose 重原子是后者,
    两者必须用**同一个**变换, 否则"盒心在 A 系、pose 在 B 系"又是静默错位。
    返回与输入同形状的 ``np.ndarray``。
    """
    import numpy as np
    arr = np.asarray(points, dtype=float)
    return (R @ arr.T).T + t

# ============================================================================
# 2. 工具
# ============================================================================

# 残基 token: [可选 1-3 字母残基名前缀] + 序号 + [可选链字母], 如
# '170' / '170.A' / 'SER170' / 'SER170.A' / '170A'。前缀仅用于校验提示,
# 定位(残基序号+链)不受前缀影响 —— 修复 CLI help 'SER170,HIS248,ASP216'
# 风格与旧数字开头正则不兼容的潜伏 bug(旧正则对 'SER170' 返回 (None,None),
# _resname_at 随之产出垃圾 key 'XXXNone<链>')。
_RE_RES_TOKEN = re.compile(r"^([A-Za-z]{0,3})\s*(\d+)\s*\.?\s*([A-Za-z]?)$")


def parse_residue_token_full(tok: str) -> Tuple[str, Optional[int], Optional[str]]:
    """残基 token 全解析, 返回 (残基名前缀, 序号, 链)。

    'SER170' -> ('SER', 170, None); 'SER170.A' -> ('SER', 170, 'A');
    '170' -> ('', 170, None); '170.A' -> ('', 170, 'A'); '' -> ('', None, None)。
    前缀统一大写(与 PDB 残基名比较用)。无法解析返回 ('', None, None)。
    """
    m = _RE_RES_TOKEN.match((tok or "").strip())
    if not m:
        return ("", None, None)
    return (m.group(1).upper(), int(m.group(2)), (m.group(3) or None))


def parse_residue_token(tok: str) -> Tuple[Optional[int], Optional[str]]:
    """'SER170'/'170' -> (170, None); 'SER170.A'/'170.A' -> (170, 'A'); '' -> (None, None)。"""
    _, rn, rc = parse_residue_token_full(tok)
    return (rn, rc)


def ref_key_match(key: str, tok: str) -> bool:
    """残基键(如 'TRP190A') 与参考残基 token('TRP190.A'/'190A'/'TRP190') 的精确匹配。

    2026-10-03 取代 startswith 前缀匹配(原缺陷双向): ① token 无残基名前缀
    ('190A'/'190')时 startswith 对键 'TRP190A' 恒不命中 → 参考残基静默失配;
    ② 'TRP190' 会同时命中 'TRP1900A'(前缀撞号) → 取到错误残基。
    规则: 残基号必须相等; token 给出的残基名/链(若有)也必须相等, 未给则不限。
    无法解析的 token 退化为原字符串精确比较(保持旧行为可追溯)。"""
    t_res, t_num, t_ch = parse_residue_token_full(tok)
    if t_num is None:
        return key == (tok or "").replace(".", "")
    k_res, k_num, k_ch = parse_residue_token_full(key)
    if k_num is None or k_num != t_num:
        return False
    if t_res and k_res != t_res:
        return False
    if t_ch and k_ch != t_ch:
        return False
    return True

# ============================================================================
# 3.5 盒子定向 (沿隧道轴旋转拉伸) —— 核心泛化优化
# ----------------------------------------------------------------------------
# 洞见: 轴对齐立方体盒子会让柔性配体在活性位点内"翻转", 失去生物学正确的
#   TPA-嵌入-芳香隧道 姿势(如 PETase 中 TPA 应深插 W159/W185 隧道)。
# 解法: 将隧道轴(亲核残基 -> 参考芳香残基 / 已结合配体质心)对齐到 Z 轴,
#   受体与配体坐标同步旋转, 盒子沿 Z 拉长(cross-section 小, tunnel 长),
#   Vina 在旋转坐标系内运行; 完成后把产物旋回原坐标系交付。旋转不改变距离。
# ============================================================================

def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-9 else [0.0, 0.0, 1.0]

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def rotation_align_to_z(axis_vec):
    """返回 3x3 旋转矩阵(list), 将 axis_vec 对齐到 +Z=(0,0,1)。"""
    v = _norm(axis_vec)
    z = [0.0, 0.0, 1.0]
    c = _dot(v, z)
    if c > 0.9999:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if c < -0.9999:
        return [[1, 0, 0], [0, -1, 0], [0, 0, -1]]   # 180° 绕 X
    axis = _norm(_cross(v, z))
    s = math.sqrt(1 - c * c)
    # Rodrigues 旋转矩阵 (轴分量 x,y,zz)
    x, y, zz = axis
    C = c; S = s
    R = [
        [C + x * x * (1 - C),     x * y * (1 - C) - zz * S,   x * zz * (1 - C) + y * S],
        [y * x * (1 - C) + zz * S, C + y * y * (1 - C),      y * zz * (1 - C) - x * S],
        [zz * x * (1 - C) - y * S, zz * y * (1 - C) + x * S, C + zz * zz * (1 - C)],
    ]
    return R

def _apply_rot(p, R, pivot):
    d = [p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]]
    r = [sum(R[i][j] * d[j] for j in range(3)) for i in range(3)]
    return (r[0] + pivot[0], r[1] + pivot[1], r[2] + pivot[2])

def rotate_pdbqt_file(path: str, R, pivot, inverse=False) -> None:
    """将 pdbqt 文件内 ATOM/HETATM 坐标绕 pivot 旋转(R 或 R 的逆)。"""
    Ruse = R
    if inverse:
        # 3x3 逆 = 转置
        Ruse = [[R[j][i] for j in range(3)] for i in range(3)]
    out = []
    overflow = False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("ATOM") or ln.startswith("HETATM"):
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                except ValueError:
                    out.append(ln); continue
                nx, ny, nz = _apply_rot((x, y, z), Ruse, pivot)
                # C2(2026-10-02): %8.3f 超宽(|v|>=1000)会右挤破坏 PDB 列布局,
                # 后续所有固定列号解析(坐标/类型码)全部失效 —— 显式拦截。
                fx, fy, fz = f"{nx:8.3f}", f"{ny:8.3f}", f"{nz:8.3f}"
                if any(len(s) > 8 for s in (fx, fy, fz)):
                    overflow = True
                    fx, fy, fz = fx[-8:], fy[-8:], fz[-8:]
                ln = ln[:30] + fx + fy + fz + ln[54:]
            out.append(ln)
    if overflow:
        LOG.warning(f"rotate_pdbqt_file: {path} 存在 |坐标|>=1000 的原子, 已截断为 8 列 —— "
                    f"旋转后列布局可能损坏, 请检查坐标系(正常蛋白坐标应在 ±1000 内)")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)

__all__ = [
    "_apply_rot",
    "_cross",
    "_dot",
    "_norm",
    "_pose_heavy_rmsd",
    "apply_rigid",
    "kabsch_rmsd",
    "kabsch_transform",
    "parse_residue_token",
    "parse_residue_token_full",
    "rotate_pdbqt_file",
    "rotation_align_to_z",
]
