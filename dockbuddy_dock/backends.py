# -*- coding: utf-8 -*-
"""backends.py —— 对接引擎适配器抽象(DockingBackend)与 Vina / LeDock 实现。

本模块由 libdock.py 按职责拆分而来(仅搬移, 逻辑未变; 路径 bootstrap 统一走 common.bootstrap_path)。
"""
import os, subprocess
from ._core import LOG


# ============================================================================
# 7. 对接引擎适配器
# ============================================================================

class DockingBackend:
    name = "abstract"

    def dock(self, receptor_pdbqt, ligand_pdbqt, center, size,
             out_pdbqt, num_modes=9, exhaustiveness=8, seed=None,
             cpu=None) -> bool:
        raise NotImplementedError

class VinaBackend(DockingBackend):
    name = "vina"

    def __init__(self, vina_exe: str):
        self.vina_exe = vina_exe

    def dock(self, receptor_pdbqt, ligand_pdbqt, center, size,
             out_pdbqt, num_modes=9, exhaustiveness=8, seed=None,
             cpu=None) -> bool:
        cmd = [
            self.vina_exe,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", f"{center[0]:.4f}",
            "--center_y", f"{center[1]:.4f}",
            "--center_z", f"{center[2]:.4f}",
            "--size_x", f"{size[0]:.2f}",
            "--size_y", f"{size[1]:.2f}",
            "--size_z", f"{size[2]:.2f}",
            "--num_modes", str(num_modes),
            "--exhaustiveness", str(exhaustiveness),
            "--out", out_pdbqt,
        ]
        # 随机种子: 用于重复对接稳定性检验(同一协议不同种子 -> pose RMSD 散布)
        if seed is not None:
            cmd += ["--seed", str(seed)]
        # CPU 钉死(2026-10-03): Vina 多线程下种群交换时序受调度影响, 固定 seed
        # 也不逐位复现(同机两次 run 的 ref 侧实测漂 0.755 kcal/mol)。对照评估链路
        # (comparative.py)钉 cpu=1 换严格可复现; 缺省不传 → Vina 引擎默认(全部核心)。
        if cpu is not None:
            cmd += ["--cpu", str(cpu)]
        LOG.info("运行 Vina: " + " ".join(cmd[:8]) + " ...")
        # 陈旧产物防护(2026-10-03 第六轮审查 D6 修订): 原修复为"跑前删旧 out_pdbqt"
        # —— 但 Vina 超时/崩溃/returncode!=0 的**常见失败路径**下, 上一次成功的对接
        # 产物被无条件删除(断点续跑场景直接损失数据, 损失面大于被修的
        # "returncode==0 未写文件"罕见误判面)。改为 ".prev 改名 + 失败恢复":
        # 跑期间旧文件不占 out 路径(防误判), 失败时旧结果原位恢复。
        _prev = out_pdbqt + ".prev"
        _had_prev = False
        if os.path.exists(out_pdbqt):
            try:
                os.replace(out_pdbqt, _prev)
                _had_prev = True
            except OSError:
                pass

        def _restore_prev():
            if _had_prev and os.path.exists(_prev) and not os.path.exists(out_pdbqt):
                try:
                    os.replace(_prev, out_pdbqt)
                except OSError:
                    pass

        try:
            # 高柔性长链配体(36+ 可旋转键)搜索空间大, 单核耗时可达数十分钟;
            # 超时放宽至 3600s 以免被截断。
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            LOG.error("Vina 超时"); _restore_prev(); return False
        except FileNotFoundError:
            # exe 路径无效(如 resolve_vina 默认路径指向不存在的目录): 给可读报错而非裸栈
            LOG.error(f"找不到对接引擎可执行文件: {self.vina_exe!r} —— 请确认安装位置"
                      f"或经 --vina / vina_exe 显式指定")
            _restore_prev(); return False
        if proc.returncode != 0:
            LOG.error("Vina 失败:\n" + (proc.stderr or proc.stdout)[-2000:])
            _restore_prev(); return False
        for line in (proc.stdout or "").splitlines():
            if "Affinity" in line or "RMSD" in line:
                LOG.info("  " + line.strip())
        if _had_prev and os.path.exists(_prev):
            # 本次已成功写出新结果(或 returncode==0 未写的罕见路径), 旧档清理
            try:
                os.remove(_prev)
            except OSError:
                pass
        return os.path.exists(out_pdbqt)

class LeDockBackend(DockingBackend):
    name = "ledock"

    def __init__(self, ledock_exe: str):
        self.ledock_exe = ledock_exe

    def dock(self, receptor_pdbqt, ligand_pdbqt, center, size,
             out_pdbqt, num_modes=9, exhaustiveness=8, seed=None,
             cpu=None) -> bool:
        # seed/cpu 形参仅为满足基类多态契约(LeDock 引擎自身无种子/CPU 概念, 忽略);
        # 缺失会在 backend.dock(..., seed=42, cpu=1) 多态调用时抛 TypeError。
        # 2026-10-03 第六轮审查: 显式给 seed 时打 WARNING —— 上层日志会打出
        # seed=42 暗示可复现, 实际 LeDock 忽略之, 运行时无提示会误导审计。
        if seed is not None:
            LOG.warning("LeDock 引擎无随机种子概念, seed=%s 被忽略 —— "
                        "结果不可按种子复现, 与 Vina 的可复现性口径不同", seed)
        cfg = os.path.splitext(out_pdbqt)[0] + ".config"
        with open(cfg, "w") as f:
            f.write(f"Receptor\n{receptor_pdbqt}\n\n")
            f.write(f"Rmsd\n2.0\n\n")
            f.write(f"Binding pocket\ncenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
            f.write(f"size {size[0]:.1f} {size[1]:.1f} {size[2]:.1f}\n\n")
            f.write(f"Number of binding poses\n{num_modes}\n\n")
            f.write(f"Ligand\n{ligand_pdbqt}\n")
        LOG.warning("LeDock 需要 .mol2 配体; 请将 ligand_pdbqt 转 .mol2 后运行 ledock %s" % cfg)
        return False


class VinaGPUBackend(DockingBackend):
    """Vina-GPU 2.1 (AutoDock-Vina-GPU-2.1 变体, OpenCL; 2026-10-04 调研接入)。

    实测 CLI 面(2026-10-04 快机 Vina-GPU+.exe --help, 与上游 README 有出入,
    以实测为准): Vina 1.2 风格 —— 单配体 ``--ligand``/``--out``/``--log``,
    **有** ``--seed``/``--num_modes``/``--energy_range``; ``--config`` 文件的
    解析格式与 README 描述不符(逐行 ``--arg value`` 报 invalid line), 故全部
    走命令行直传。参数模型与 Vina CPU 的差异:
      - ``--thread``: 并行 lane 数(默认 8000)。GPU 搜索覆盖度由 lane 数保证,
        远超 CPU exhaustiveness 语义(32 MC run vs 8000 lane), 故 exhaustiveness
        **不映射**仅随产物元数据记录; search_depth 缺省引擎启发式
        (DOCK_VINA_GPU_SEARCH_DEPTH 覆盖)。
      - ``--seed`` 虽有旗标但 GPU 并行归约顺序不定, **逐位复现无保证**(传入打
        WARNING; 筛选层可接受, 论文确认层回 Vina CPU)。
      - 盒子限制 ≤30 Å(2.0 实测上限, 见 BOX_LIMIT 注释; 超限拒绝)。
      - 无 ``--cpu``。

    科学口径: 与 Vina CPU 同族打分但非逐位一致 → 筛选层用 GPU, 论文确认层
    回 Vina CPU exh=32; 产物 engine 字段区分两种口径, 报告禁止混比。

    exe 与 Kernel1/2_Opt.bin 需同目录(上游仓库自带编译好的 exe); 首次运行若
    报 kernel 缺失, 先跑一次带 _K 后缀的版本生成 Kernel bin。运行日志落
    <out_pdbqt 同 stem>.log 供审计。
    """

    name = "vina_gpu"
    # Vina-GPU-2.0 kernel2.h MAX_NUM_OF_GRID_MI=128 网格点, 盒子单边按引擎
    # 自适应格距折算 ~25-30 Å(快机实测 24³ 稳定); 2.1 的 100 Å 上限不适用
    # (其 exe/kernel 与 2.0 源码不配套, 6GB 卡 grids buffer 亦物理放不下)。
    # 保守取 30, 超限回 Vina CPU。
    BOX_LIMIT = 30.0
    DEFAULT_THREAD = 8000    # 官方默认; 6GB 级卡建议 DOCK_VINA_GPU_THREAD=2000

    def __init__(self, vina_gpu_exe: str):
        self.exe = vina_gpu_exe

    # -- 陈旧产物防护(与 VinaBackend 同款): .prev 改名 + 失败恢复 ------------
    @staticmethod
    def _stash_prev(out_pdbqt):
        _prev = out_pdbqt + ".prev"
        had = False
        if os.path.exists(out_pdbqt):
            try:
                os.replace(out_pdbqt, _prev)
                had = True
            except OSError:
                pass
        return _prev, had

    @staticmethod
    def _restore_prev(out_pdbqt, _prev, had):
        if had and os.path.exists(_prev) and not os.path.exists(out_pdbqt):
            try:
                os.replace(_prev, out_pdbqt)
            except OSError:
                pass

    def dock(self, receptor_pdbqt, ligand_pdbqt, center, size,
             out_pdbqt, num_modes=9, exhaustiveness=8, seed=None,
             cpu=None) -> bool:
        if not self.exe or not os.path.exists(self.exe):
            LOG.error(f"Vina-GPU 可执行文件不存在: {self.exe!r} —— "
                      f"放置 tools/vina-gpu-2.0/(Vina-GPU+/ 目录, exe 与 "
                      f"Kernel*.bin 同目录)或设 DOCK_VINA_GPU_EXE")
            return False
        if max(float(s) for s in size) > self.BOX_LIMIT:
            LOG.error(f"Vina-GPU 盒子单边 {max(float(s) for s in size):.0f} Å "
                      f"超上限 {self.BOX_LIMIT:.0f} Å —— 请改用 Vina CPU 引擎")
            return False
        if not os.path.exists(receptor_pdbqt):
            LOG.error(f"Vina-GPU 受体文件不存在: {receptor_pdbqt!r}")
            return False
        if not os.path.exists(ligand_pdbqt):
            LOG.error(f"Vina-GPU 配体文件不存在: {ligand_pdbqt!r}")
            return False
        # seed 旗标存在且透传(审计价值), 但 GPU 并行归约顺序不定 → 复现性降级提示
        if seed is not None:
            LOG.warning("Vina-GPU 已传 --seed=%s, 但 GPU 并行归约顺序不定, 逐位复现"
                        "无保证; 筛选层可接受, 论文确认层请回 Vina CPU", seed)
        if cpu is not None:
            LOG.warning("Vina-GPU 引擎无 CPU 亲和参数, cpu=%s 被忽略", cpu)

        _prev, had = self._stash_prev(out_pdbqt)
        out_abs = os.path.abspath(out_pdbqt)
        log_path = os.path.splitext(out_abs)[0] + ".log"
        # exe 一律走绝对路径(2026-10-04 审计): subprocess 换 cwd=kernel_dir 后,
        # 相对路径的 self.exe 会解析到错误位置 → 误导性 "找不到 exe" 报错。
        exe_abs = os.path.abspath(self.exe)
        kernel_dir = os.path.dirname(exe_abs)
        # 2026-10-04 全盘审计: env 非数字时 int() 裸栈(同 C7 env_float 口径), 改防御。
        try:
            thread = int(os.environ.get("DOCK_VINA_GPU_THREAD", self.DEFAULT_THREAD))
        except ValueError:
            thread = self.DEFAULT_THREAD
        cmd = [
            exe_abs,
            "--receptor", os.path.abspath(receptor_pdbqt),
            "--ligand", os.path.abspath(ligand_pdbqt),
            "--out", out_abs,
            "--log", log_path,
            "--center_x", f"{center[0]:.4f}",
            "--center_y", f"{center[1]:.4f}",
            "--center_z", f"{center[2]:.4f}",
            "--size_x", f"{size[0]:.2f}",
            "--size_y", f"{size[1]:.2f}",
            "--size_z", f"{size[2]:.2f}",
            "--thread", str(thread),
            "--num_modes", str(num_modes),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        _sd = os.environ.get("DOCK_VINA_GPU_SEARCH_DEPTH", "")
        if _sd:
            try:
                cmd += ["--search_depth", str(int(_sd))]
            except ValueError:
                LOG.warning("DOCK_VINA_GPU_SEARCH_DEPTH=%r 非数字, 忽略", _sd)
        try:
            LOG.info(f"运行 Vina-GPU: {os.path.basename(self.exe)} "
                     f"(thread={thread}, num_modes={num_modes}"
                     + (f", seed={seed}" if seed is not None else "")
                     + f", box={size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f}Å)")
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=3600, cwd=kernel_dir)
            if proc.returncode != 0:
                LOG.error("Vina-GPU 失败:\n" + (proc.stderr or proc.stdout)[-2000:])
                self._restore_prev(out_pdbqt, _prev, had)
                return False
            for line in (proc.stdout or "").splitlines():
                if "Affinity" in line or "score" in line.lower():
                    LOG.info("  " + line.strip())
            if not os.path.exists(out_pdbqt):
                LOG.error(f"Vina-GPU returncode=0 但未写出 {out_pdbqt!r} —— "
                          f"请贴回 stdout 与 {log_path} 核对:\n"
                          + (proc.stdout or "")[-1000:])
                self._restore_prev(out_pdbqt, _prev, had)
                return False
            # 成功清理 .prev(对齐 VinaBackend: 成功后旧档删除, 防产物目录
            # 残留幽灵文件; 失败路径才走 _restore_prev 原位恢复)
            if had and os.path.exists(_prev):
                try:
                    os.remove(_prev)
                except OSError:
                    pass
            return True
        except subprocess.TimeoutExpired:
            LOG.error("Vina-GPU 超时(3600s)")
            self._restore_prev(out_pdbqt, _prev, had)
            return False
        except FileNotFoundError:
            LOG.error(f"找不到 Vina-GPU 可执行文件: {self.exe!r}")
            self._restore_prev(out_pdbqt, _prev, had)
            return False

__all__ = [
    "DockingBackend",
    "LeDockBackend",
    "VinaBackend",
    "VinaGPUBackend",
]
