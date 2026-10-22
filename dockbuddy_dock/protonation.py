# -*- coding: utf-8 -*-
"""protonation.py —— 受体精确质子化(PROPKA/H++, 缺失优雅降级 SKIP)。

为什么需要它
-------------
当前受体加氢走 OpenBabel `-h -p 7.4`(libdock.receptor.prepare_receptor)——这是
**固定 pH 的启发式**, 不分逐残基 pKa。对可质子化残基(Asp/Glu/Lys/His/Tyr/Cys 末端)
在 pH 7.4 下的默认态常与真实 pKa 偏移, 其中 **His 的单/双质子化态对催化三联体
(如 His204)的质子中转功能极关键**。PROPKA(pdb2pqr+propka / propka 包)可算逐残基
pKa 并据此指派质子化态 = "precise" 路径。

诚实优先(本模块核心约束)
------------------------
若 PROPKA 不可用, **绝不伪造 "precise" 结论** —— 退回固定 pH 的启发式加氢, 状态
标 `skipped`, 并在 JSON 顶层写明 `precise=False` / `provenance`, 下游报告不得据此
声称"精确质子化"。精确路径仅当 propka/pdb2pqr 真实可用时才运行, 且任何异常都降级到
启发式 SKIPPED, 绝不静默产出一张"声称精确"却实为启发式的结构。
"""
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from ._core import LOG, env_float

# env 覆盖: 强制后端 / 默认 pH
PROTONATION_BACKEND_ENV = "DOCK_PROTONATION_BACKEND"   # auto|propka|heuristic
PROTONATION_PH_DEFAULT = env_float("DOCK_PROTONATION_PH", 7.0)


# ============================================================================
# 后端探测(与实现同源, 不重复维护清单)
# ============================================================================
def _propka_available() -> bool:
    """PROPKA 精确路径是否真实可用(propka 包 或 pdb2pqr/propka3 可执行文件)。

    仅当此函数返回 True 时才允许进入 precise 分支, 否则一律走启发式。
    """
    try:
        import propka  # noqa: F401
        return True
    except Exception:
        pass
    if shutil.which("pdb2pqr") or shutil.which("propka3") or shutil.which("propka"):
        return True
    return False


def detect_protonation_backend() -> str:
    """返回 'propka'(精确路径可用) 或 'heuristic'(退回固定 pH 启发式)。

    可被 DOCK_PROTONATION_BACKEND 强制覆盖(auto/propka/heuristic); 未设时按
    _propka_available() 实测决定。
    """
    env = os.environ.get(PROTONATION_BACKEND_ENV, "").strip().lower()
    if env == "propka":
        return "propka"
    if env == "heuristic":
        return "heuristic"
    # env == "auto" 或未设
    return "propka" if _propka_available() else "heuristic"


# ============================================================================
# 两条路径
# ============================================================================
def _protonate_heuristic(pdb_path: str, out_pdb: str, ph: float,
                         chain: str = "") -> Dict[str, Any]:
    """启发式加氢(amber14 addHydrogens @pH), 状态恒 'skipped'(非精确)。"""
    import openmm.app as app
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff, pH=ph)
    app.PDBFile.writeFile(mod.topology, mod.positions, open(out_pdb, "w"))
    n_h = sum(1 for a in mod.topology.atoms() if a.element.symbol == "H")
    n_heavy = sum(1 for a in mod.topology.atoms() if a.element.symbol != "H")
    return {
        "ok": True, "status": "skipped", "backend": "heuristic_addH",
        "precise": False,
        "method": f"amber14-all addHydrogens @pH={ph} (固定 pH 启发式, 非 pKa 感知)",
        "out_pdb": out_pdb, "ph": ph, "chain": chain,
        "n_heavy": n_heavy, "n_hydrogen": n_h,
        "provenance": "heuristic amber14 addHydrogens (NOT pKa-aware)",
    }


def _protonate_propka(pdb_path: str, out_pdb: str, ph: float,
                      chain: str = "") -> Dict[str, Any]:
    """精确质子化(pdb2pqr+propka 逐残基 pKa 指派), 状态 'precise'。

    仅当 _propka_available() 为 True 时调用; 任何失败由调用方降级到启发式。
    """
    binpath = (shutil.which("pdb2pqr")
               or shutil.which("propka3")
               or shutil.which("propka"))
    if not binpath:
        raise RuntimeError("未找到 pdb2pqr/propka 可执行文件(propka 精确路径不可用)")
    cmd = [binpath, "--ff=AMBER", f"--with-ph={ph}",
           "--titration-state-method=propka", pdb_path, out_pdb]
    LOG.info("[protonation] 运行 %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not os.path.exists(out_pdb):
        raise RuntimeError(
            f"{os.path.basename(binpath)} 失败(rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-800:]}")
    # 原子计数(避免引入 openbabel 依赖, 直接读 PDB 行)
    n_h = n_heavy = 0
    with open(out_pdb, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            elem = ln[76:78].strip()
            if elem == "H":
                n_h += 1
            else:
                n_heavy += 1
    return {
        "ok": True, "status": "precise", "backend": "propka",
        "precise": True,
        "method": f"pdb2pqr+propka pKa-aware @pH={ph}",
        "out_pdb": out_pdb, "ph": ph, "chain": chain,
        "n_heavy": n_heavy, "n_hydrogen": n_h,
        "provenance": "propka pKa-aware titration states",
    }


# ============================================================================
# 主入口
# ============================================================================
def protonate_receptor(pdb_path: str, out_pdb: str, ph: float = PROTONATION_PH_DEFAULT,
                       backend: str = "auto", chain: str = "") -> Optional[Dict[str, Any]]:
    """受体质子化(精确优先, 缺失优雅降级 SKIP)。

    返回 dict:
      - ok=True:  {status: 'precise'|'skipped', precise, backend, method, out_pdb,
                   ph, n_heavy, n_hydrogen, provenance}
      - ok=False: {status: 'error'|'unavailable', error, precise=False}
    backend='auto' → 探测后端; propka 缺失时退回启发式并标 skipped(诚实, 不伪造)。
    backend='propka'(显式)但不可用 → 返回 unavailable(不静默假装精确)。
    backend='heuristic' → 强制启发式(状态 skipped)。
    """
    if not (pdb_path and os.path.exists(pdb_path)):
        return {"ok": False, "status": "error",
                "error": f"受体不存在: {pdb_path}", "precise": False}
    if backend == "auto":
        bk = detect_protonation_backend()
    else:
        bk = backend
    if bk == "propka":
        if not _propka_available():
            if backend == "auto":
                # 诚实降级: 自动模式下 propka 缺失 → 启发式 SKIPPED(不伪造精确)
                LOG.warning("[protonation] propka 不可用, 自动退回启发式 SKIPPED")
                res = _protonate_heuristic(pdb_path, out_pdb, ph, chain)
                res["note"] = "propka 不可用, 自动退回固定 pH 启发式(非精确)"
                return res
            # 显式要求 propka 但不可用 → 明确报错, 不静默降级
            return {"ok": False, "status": "unavailable",
                    "error": "显式请求 propka 精确质子化, 但 propka/pdb2pqr 不可用",
                    "precise": False}
        try:
            return _protonate_propka(pdb_path, out_pdb, ph, chain)
        except Exception as e:
            LOG.warning("[protonation] propka 精确路径异常, 退回启发式: %s: %s",
                        type(e).__name__, e)
            res = _protonate_heuristic(pdb_path, out_pdb, ph, chain)
            res["status"] = "skipped"
            res["backend_fallback"] = "heuristic_addH"
            res["note"] = f"propka 精确路径异常, 退回启发式: {e}"
            return res
    # bk == "heuristic"
    return _protonate_heuristic(pdb_path, out_pdb, ph, chain)


__all__ = ["PROTONATION_PH_DEFAULT", "detect_protonation_backend",
           "protonate_receptor"]
