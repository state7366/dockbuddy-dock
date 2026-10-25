# -*- coding: utf-8 -*-
"""field_docs.py —— 各命令 JSON 产物的字段字典(单一事实源)。

===================================================================================
定位
===================================================================================
2026-10-05/06 审计结论: 各命令的 JSON 产物(docking_summary / run_meta / .mutate.json /
pocket-tunnel / pocket-decomp / deacylation / msa-report / product-egress / compare)
此前只有代码注释里的零散说明, 没有一处能回答"这个字段是什么意思、单位是什么、
怎么判读"。本模块把**全部字段的详细解释**收进一个字典, 供两处消费:

  1. `libdock.run_report.write_run_report()` —— 每条命令落盘 JSON 后自动生成
     `<stem>.report.md`(全量字段表 + 解释 + 关键指标);
  2. 人工查阅: `python -c "from libdock.field_docs import FIELD_DOCS; ..."` 或
     生成的 report.md 直接看。

字段覆盖审计: `coverage(doc_type, payload)` 返回"已登记/未登记"计数与清单 ——
新代码往 JSON 里加了字段而忘了在这里登记, report.md 的脚注会把它顶出来
(与契约基线锁 --help 同思路: 锁不住的至少要被看见)。

路径记法(flatten 约定):
  - 嵌套 dict  : 点连接, 如 `qc.triad.his_asp_d_A` / `tunnel.wt.cavity_volume_A3`;
  - dict 数组  : 元素字段记 `poses[i].energy_kcal_mol`(i=任意下标的模板);
  - 动态键     : 键名本身是数据(如 egress verdicts 的受体路径、msa sites 的列号),
                 记 `sites.*.n_total`(* = 通配);
  - 标量数组   : 整体记一个键(如 `tags`), 不逐项展开。

本模块**刻意不进** `libdock/__init__.py` 的全量再导出(与 pocket_tunnel 同款约定):
纯 stdlib、按点号路径显式导入, 避免把不必要的符号塞进 `libdock.<TAB>` 面。

数据来源: 字段清单逐一核对自 2026-10-05 快机 v2 全集真实产物
(output_full_v2 的 10 篇 docking + 10 个 mutate + tunnel/decomp/egress/deacyl/msa),
并与写盘代码(cli/dock.py 6.95/6.97 段、pipeline/mutate.py _mutate_one、
cli/pocket_tunnel_cmd.py、cli/analysis_cmd.py)逐字段对照。
===================================================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# 字段字典本体
# ----------------------------------------------------------------------------
# 每个条目: {"title": 报告标题, "key_metrics": 关键指标路径列表,
#            "fields": {路径: 解释}}
FIELD_DOCS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # dock: docking_summary.json
    # ------------------------------------------------------------------
    "docking_summary": {
        "title": "对接结果 (docking_summary.json)",
        "key_metrics": ["energy_kcal_mol", "reliability", "nuc_to_ligO_A",
                        "box_assumption", "n_poses"],
        "fields": {
            "ok": "本篇对接是否成功走完全流程(受体/配体/盒子/对接/分析)。False 时其余字段可能缺失, 看notes。",
            "energy_kcal_mol": "Vina 预测结合自由能(kcal/mol), 越负越有利。只适合同引擎同体系内相对比较(vina_gpu 为 1.1.2 移植打分, 与 CPU 1.2.7 禁止混比); 能量差 |ΔE|<0.5 kcal/mol 视为打分噪声量级。",
            "box_center": "实际使用的盒子中心坐标 [x,y,z](Å, 定向旋转系内)。",
            "receptor_chain": "本 run 使用的受体链(自动选最长链或显式 --chain)。",
            "catalytic": "检出/指定的催化三联体残基(含链号, 如 SER101A)。空=未检出(盒子回退蛋白心)。",
            "nuc_to_ligO_A": "亲核原子(Ser OG / Cys SG)到配体最近氧的距离(Å)。酯水解几何前提, ≤3.5 Å 视为亲核进攻位; offsite 红线 10 Å。",
            "ref_to_ligAromC_A": "参考残基(--ref)到配体最近芳香碳的距离(Å)。null=未设参考残基或配体无芳香碳。",
            "arom_pocket_dist_A": "配体芳香碳到口袋芳香残基的最近距离(Å; 芳香锚定判读)。null=无芳香接触。",
            "arom_pocket_residue": "配体芳香碳锚定到的口袋芳香残基名(如 PHE56A)。空=无。",
            "pocket_residues": "口袋残基列表(距配体 ≤4.5 Å, 含链号); 供 PyMOL select 与报告引用。",
            "n_poses": "Vina 输出并被分析的 pose 总数(= --num_modes, 缺省 9)。",
            "notes": "人类可读附注(pose 发散警告/盒子标注等); warnings 是结构化的判读告警, 两者互补。",
            "has_nucleophile": "是否检出亲核残基(SER/CYS/THR/LYS…)。False 时 nuc_to_ligO_A 不可用。",
            "nuc_resname": "亲核残基名(如 SER)。",
            "ligand_clashes": "配体与受体重原子的硬球对撞数。理想 0; >0 看该 pose 是否可用。",
            "ligand_buried_frac": "配体被完全包埋的表面占比(0~1); 高值+小 ligand_med_near=埋藏信号。",
            "ligand_med_near": "配体重原子到受体最近距离的中位数(Å)。",
            "ligand_buried": "配体是否整体埋入蛋白内部(reliability=warn 的信号之一; 无出口埋藏对接不可信)。",
            "pocket_contact_frac": "与口袋残基有接触的重原子占比(0~1); 过低=偏靶(没进袋)。",
            "ligand_centroid_offset_A": "配体质心相对盒子中心的偏移(Å); 大偏移=pose 贴盒边, 采样受限。",
            "ligand_n_heavy": "配体重原子数(与元素组成守卫配合, 识别晶体配体误提取)。",
            "energy_reliable": "能量数值是否可信(False 时流水线已自动用更高 exhaustiveness 重试, 仍 False 则不可引用)。",
            "box_assumption": "盒子中心的假设来源: triad=三联体质心 / ligand=共晶配体质心 / protein_centroid=蛋白心(盲盒回退, 结果需谨慎) / manual。",
            "reliability": "三态可靠性结论: ok / warn(结果可用但需看 warnings) / unreliable(不可引用)。",
            "warnings": "可靠性告警明细(埋藏/能量失真/接触不足/pose 发散等)。",
            "top_pose_rmsd_A": "top1 与 top2 pose 的 RMSD(Å); 大值(>2)=pose 发散, 单一 top1 不可独信(只能报能量统计)。",
            "pose_set_rmsd_mean_A": "全 pose 两两重原子 RMSD 的均值(Å); 结合模式收敛性量化(与 top1~top2 发散互补)。",
            "pose_set_rmsd_median_A": "全 pose 两两 RMSD 的中位数(Å); >3(pose_scatter_warn)=整个 pose 集合散开(多模态), 结合模式未收敛, 只能报能量统计不能断言单一结合模式。",
            "pose_set_rmsd_max_A": "全 pose 两两 RMSD 的最大值(Å; 最远的两个 pose 差多远)。",
            "pose_set_rmsd_n_pairs": "实际参与自散计算的两两对数(= n_poses×(n_poses−1)/2, 排除不可比对)。",
            "warn_class": "告警大类(soft 等)。",
            "warn_reasons": "告警原因码列表(如 pose_div=pose 发散)。",
            "selfdock_available": "是否执行了自对接校验(需要共晶配体; AF 模型等无共晶时为 False)。",
            "selfdock_rmsd_A": "晶体配体重对接与晶体姿态的最小 RMSD(Å), 逐全 pose 取最优; ≤2 Å 视为协议可复现。null=原子名不匹配, 测量不可用。",
            "selfdock_pose_idx": "达到最小 RMSD 的 pose 序号(1-based)。",
            "selfdock_ok": "采样是否覆盖晶体姿态(最小 RMSD ≤2 Å)。",
            "selfdock_top_rmsd_A": "按打分排序的 top1 pose 与晶体的 RMSD(Å)。与 selfdock_rmsd_A 分开记录: 排序可复现 ≠ 采样可覆盖(M1 口径)。",
            "selfdock_repro_top1": "top1 排序是否可复现(top1 恰为最小 RMSD pose)。",
            "selfdock_resname": "参与自对接的共晶配体残基名(如 C9C)。",
            "selfdock_note": "自对接附注(如全 pose 原子名不匹配 → RMSD 不可用的原因)。",
            "selfdock_n_unmatched": "原子名不匹配导致 RMSD 不可用的 pose 数(>0 时 selfdock_rmsd_A=null)。",
            "schema_version": "本 JSON 结构版本号(当前 1; 结构性变更时递增)。",
            "poses": "逐 pose 轻量指标数组(默认前 5 个 pose), 字段见 poses[i].*。",
            "poses[i].pose_index": "pose 序号(1-based, 与 results.pdbqt 的 MODEL 顺序一致)。",
            "poses[i].energy_kcal_mol": "该 pose 的 Vina 能量(kcal/mol)。",
            "poses[i].nuc_to_ligO_A": "该 pose 的亲核-配体氧距离(Å)。",
            "poses[i].ref_to_ligAromC_A": "该 pose 的参考残基-芳香碳距离(Å)。",
            "poses[i].pocket_contact_frac": "该 pose 的口袋接触占比。",
            "poses[i].ligand_buried": "该 pose 配体是否埋藏。",
            "poses[i].ligand_clashes": "该 pose 的对撞数。",
            "poses[i].ligand_buried_frac": "该 pose 的包埋占比。",
            "poses[i].ligand_n_heavy": "该 pose 的配体重原子数。",
            "poses[i].energy_reliable": "该 pose 的能量可信性。",
            "poses[i].rmsd_to_crystal_A": "该 pose 与晶体配体的 RMSD(Å); null=无晶体基准或不可比。",
            "sort_strategy": "多构象排序策略: energy=Vina 能量升序; crystal_rmsd=self-dock 场景按 pose-vs-晶体 RMSD 升序。",
            "selected_pose_index": "被选为『最优』参与主指标(top-level 字段)计算的 pose 序号。",
            "crystal_rmsd_available": "poses 数组里是否带晶体 RMSD(共晶配体提取且比对成功)。",
            "is_self_dock": "本 run 是否为自对接(对接配体==该受体共晶配体, 按元素组成自动判定)。",
            "tags": "实验 tag 并集(用户 --tag + 自动 tag id:/chain:/pdb:/ligand:); 供 data/reports/batch --filter 过滤。",
            "orient_fallback": "隧道定向(box_orient)困住配体时的自动回退留痕: triggered(是否触发)/adopted(是否采纳)/orig_nuc_to_ligO_A/retry_nuc_to_ligO_A(两侧亲核距离)/retry_box_size。仅触发时存在。",
            "box_ligand_triad_dist_A": "共晶配体质心与三联体质心的距离(Å); 过远时盒子改用三联体中心(box_note 记原因)。仅判读发生时存在。",
            "box_note": "盒子决策附注文本(与 box_ligand_triad_dist_A 同时存在)。",
        },
    },

    # ------------------------------------------------------------------
    # dock: run_meta.json
    # ------------------------------------------------------------------
    "run_meta": {
        "title": "运行元数据 (run_meta.json)",
        "key_metrics": ["run_id", "engine", "seed", "exhaustiveness",
                        "box_center", "box_source_model"],
        "fields": {
            "run_id": "本 run 标识(= 产物目录 dock/single/<id>/ 的名字)。",
            "seed": "Vina 随机种子(缺省固定 42; 显式 --seed 或 DOCK_SEED 覆盖)。复现协议四件套之一。",
            "exhaustiveness": "Vina 穷举度(缺省 8; 论文确认层口径 exh=32)。复现协议四件套之一。",
            "num_modes": "输出 pose 数(缺省 9)。复现协议四件套之一。",
            "engine": "对接引擎(vina / vina_gpu / ledock)。复现协议四件套之一; 跨引擎能量禁止混比。",
            "box_center": "实际盒子中心 [x,y,z](Å) —— 定向生效时为隧道盒的旋转系中心。",
            "box_size": "实际盒子尺寸 [x,y,z](Å)。定向生效时是隧道盒(如 12×12×轴长+10)而非配置值 —— 与 box_orient_applied 配合判读。",
            "box_mode": "盒心决策模式(auto / ligand / manual / triad 回退)。",
            "box_assumption": "实际落地的盒心假设(见 docking_summary.box_assumption)。",
            "box_orient": "配置是否开启隧道定向(--no_box_orient 可关)。",
            "box_orient_applied": "本次定向是否实际生效(需要三联体+锚点; False=配置开但没触发)。",
            "orient_fallback": "定向困住配体时的自动回退留痕(同 docking_summary.orient_fallback; None=未触发)。",
            "box_source_model": "盒心取自哪个 AF 模型(m0..m4, 从受体文件名解析; 解析不出留空)。审计缺陷#1 的溯源字段: 报告盒心与代表模型不一致时用它事后判定来源。",
            "receptor_chain": "受体链。",
            "receptor": "受体源文件名(不含路径)。",
            "pdb_id": "RCSB PDB ID(本地受体为空)。",
            "ligand_smiles": "配体 SMILES(文件/共晶提取时为空)。",
            "ligand_name": "共晶配体名(--ligand_query 或 manifest ligand_name)。",
            "ligand_flexible": "配体是否柔性对接(TORSDOF>0)。",
            "ligand_ph": "配体制备质子化 pH(≤0 表示不做 pH 校正)。",
            "timestamp": "本 run 完成时间(本地时间, YYYY-MM-DD HH:MM:SS)。",
            "pipeline": "来源标识(DockBuddy)。",
            "triad": "检出的催化三联体文本形式(如 SER101-HIS203-ASP171)。",
            "triad_ca_centroid": "三联体 Cα 质心坐标 [x,y,z](Å; mutate --inherit_box 的盒心候选之一)。",
        },
    },

    # ------------------------------------------------------------------
    # mutate: <id>.mutate.json
    # ------------------------------------------------------------------
    "mutate": {
        "title": "定点突变建模结果 (<id>.mutate.json)",
        "key_metrics": ["mutations", "relax_mode", "qc.ca_rmsd_vs_wt_A",
                        "qc.clashes_at_sites"],
        "fields": {
            "engine": "突变引擎(pdbfixer = PDBFixer 侧链替换 + OpenMM 局部松弛)。",
            "receptor_in": "输入(WT)受体路径(相对 OUTPUT_ROOT, 跨机器可迁移)。",
            "out_pdb": "输出突变体 PDB 路径(纯蛋白, HETATM 已丢弃)。",
            "mutations": "突变明细数组, 字段见 mutations[i].*。",
            "mutations[i].label": "突变标签(如 A123G)。",
            "mutations[i].chain": "突变所在链。",
            "mutations[i].resnum": "残基号。",
            "mutations[i].wt": "该位点**要求**的 WT 残基三字码(残基身份硬校验: 喂 WT 不符的结构会直接报错)。",
            "mutations[i].mut": "目标残基三字码。",
            "mutations[i].applied": "该突变是否实际应用(多突变组合中逐位点核验)。",
            "n_mutations": "本次建模应用的突变总数(组合突变时 >1)。",
            "is_combo": "是否多突变组合(>1 个位点同时建模, 同一/跨链均可)。",
            "combo_chains": "组合突变涉及的链集合(仅 is_combo=True 时出现)。",
            "minimized": "是否做了局部受限最小化(--no_minimize 关)。",
            "restraint_radius_A": "局部松弛放行半径(Å) —— 只动突变位点邻域, 骨架远处受约束。",
            "ligands_dropped": "建模时丢弃的 HETATM 残基名列表(共晶配体/水/离子等)。",
            "bfact_restored": "是否回填 B-factor(松弛后重原子)。",
            "notes": "引擎附注(如回写重原子数)。",
            "relax_mode": "松弛方式: minimize=真空力场最小化(默认) / md=隐式溶剂 OBC(GB) 短程 MD(结果刚体回正, MD 固定种子可复现) / none=只换侧链。",
            "n_free_atoms": "松弛时自由原子数(突变位点邻域)。",
            "n_restrained_atoms": "松弛时受约束原子数(其余全蛋白)。",
            "positions_unit": "输出坐标单位(angstrom; 出现 nm 即坏产物信号)。",
            "qc_failed": "QC 拒绝标记(True 时产物已删除或落 .pdb.qc_failed, 增量不会复用)。",
            "qc_fail_reason": "QC 拒绝原因(如 Cα 匹配失败 / RMSD>2 Å / 坐标单位异常)。",
            "qc.ca_rmsd_vs_wt_A": "突变体与 WT 的 Cα RMSD(Å, 不叠合直接差)。局部松弛预期 <0.5; >2.0 拒绝产出(骨架被搬动或单位错配)。null=Cα 匹配失败。",
            "qc.ca_matched": "匹配上的 Cα 个数(0 = 链 ID/编号体系不一致, QC 不可执行)。",
            "qc.coord_span_A": "三方向坐标跨度(Å): wt 与 mutant 各一组; 同量级=单位一致。",
            "qc.coord_span_A.wt": "WT 的 [x,y,z] 跨度(Å)。",
            "qc.coord_span_A.mutant": "突变体的 [x,y,z] 跨度(Å)。",
            "qc.coord_span_ratio": "突变体/Wt 最大跨度之比; 0.5~2.0 之外判坐标单位错配(nm/Å)。",
            "qc.focus_residues": "焦点残基标签(链:号+突变标签), clash 检查的作用域。",
            "qc.clashes_at_sites": "突变位点邻域的原子对撞列表(理想为 0)。",
            "qc.focus_atoms_matched": "焦点残基在突变体中命中的原子数; 0 = clash 检查空跑(见 focus_check_vacuous)。",
            "qc.focus_check_vacuous": "True = 焦点一个都没命中(链推断失败), 需 --chain 显式指定; 这时 clashes_at_sites 无效。",
            "qc.clashes_total_mutant": "突变体全蛋白对撞数(与 WT 同跑对照, 差量才有意义)。",
            "qc.clashes_total_wt": "WT 全蛋白对撞数。",
            "qc.triad": "催化三联体 QC(突变体与 WT 同跑取差量), 字段见 qc.triad.*。",
            "qc.triad.detected": "突变体是否检出三联体。",
            "qc.triad.label": "三联体文本形式(如 SER101-HIS203-ASP171)。",
            "qc.triad.his_asp_d_A": "突变体 His→Asp 距离(Å)。",
            "qc.triad.his_asp_d_wt_A": "WT His→Asp 距离(Å)。",
            "qc.triad.his_asp_d_delta": "差量(Å); 大差值=突变扰动传播到三联体。",
            "qc.triad.his_ne2_nuc_d_A": "突变体 His NE2→亲核原子距离(Å)。",
            "qc.triad.his_ne2_nuc_d_wt_A": "WT 同键距离(Å)。",
            "qc.triad.his_ne2_nuc_d_delta": "差量(Å)。",
            "qc.triad.label_changed": "三联体身份是否改变(True=突变把催化残基本身换掉了, warnings 会预警)。",
            "qc.triad.verdict": "三联体 QC 判定: ok / warn(有 warnings 时)。绝对区间: his_asp_d 2.5~3.6 Å, his_ne2_nuc_d 2.7~4.5 Å。",
            "qc.triad.warnings": "三联体 QC 告警(如催化残基被突变掉/键距超区间)。",
            "qc.triad.raw_mutant": "突变体侧原始检出(detected/label/his_asp_d_A/his_ne2_nuc_d_A/warnings)。",
            "qc.triad.raw_mutant.detected": "突变体是否检出三联体。",
            "qc.triad.raw_mutant.label": "突变体检出的三联体文本形式。",
            "qc.triad.raw_mutant.his_asp_d_A": "突变体 His→Asp 距离(Å)。",
            "qc.triad.raw_mutant.his_ne2_nuc_d_A": "突变体 His NE2→亲核原子距离(Å)。",
            "qc.triad.raw_mutant.warnings": "突变体侧检出告警。",
            "qc.triad.raw_wt": "WT 侧原始检出(字段同 raw_mutant)。",
            "qc.triad.raw_wt.detected": "WT 是否检出三联体。",
            "qc.triad.raw_wt.label": "WT 检出的三联体文本形式。",
            "qc.triad.raw_wt.his_asp_d_A": "WT His→Asp 距离(Å)。",
            "qc.triad.raw_wt.his_ne2_nuc_d_A": "WT His NE2→亲核原子距离(Å)。",
            "qc.triad.raw_wt.warnings": "WT 侧检出告警。",
            "qc.pocket_geom": "口袋几何 QC(历史口径, 球心=催化三联体 Cα 质心), 字段见 qc.pocket_geom.*。⚠ 只可和历史 mutate QC 对照, 与 pocket_tunnel 的 Ser OG 球心体积禁止相减/混用。",
            "qc.pocket_geom.center_A": "QC 球心坐标(Cα 质心)。",
            "qc.pocket_geom.radius_A": "QC 球半径(Å)。",
            "qc.pocket_geom.step_A": "QC 网格步长(Å; 该口径 0.8, 与 pocket_tunnel 的 0.6 不同)。",
            "qc.pocket_geom.probe_A": "探针半径(Å)。",
            "qc.pocket_geom.free_volume_wt_A3": "WT 侧自由体积(Å³)。",
            "qc.pocket_geom.free_volume_mut_A3": "突变体侧自由体积(Å³)。",
            "qc.pocket_geom.delta_A3": "体积差量(Å³; >0=口袋打开)。",
            "qc.pocket_geom.delta_pct": "差量百分比(%)。",
            "qc.pocket_geom.note": "口径说明(球心与单位约定)。",
            "qc.pocket_tunnel": "opt-in(--pocket_tunnel)的口袋+通道 QC 摘要(正式几何在 pocket-tunnel 命令; 判词同源 compare_sides)。字段同 tunnel.json 的 verdict.* —— 见 pocket_tunnel 文档。",
            "qc.disulfide": "二硫键风险 QC(仅当突变引入 Cys 时挂载): pairs=全部 SG-SG<2.5 Å 对(dist_A/preexisting), new_pairs=突变引入的**新**二硫键(预警: 锁死构象/干扰折叠/埋藏催化位点), preexisting 对为 WT 既存结构特征非风险。",
            "qc.disulfide.pairs": "突变体全部二硫键对列表(a/b=链:号, dist_A=键长, preexisting=WT 中已存在)。",
            "qc.disulfide.pairs[i].a": "半胱氨酸 1(链:残基号)。",
            "qc.disulfide.pairs[i].b": "半胱氨酸 2。",
            "qc.disulfide.pairs[i].dist_A": "SG-SG 距离(Å; <2.5 记为二硫键)。",
            "qc.disulfide.pairs[i].preexisting": "WT 中同残基对是否已成键(既存特征, 非风险)。",
            "qc.disulfide.new_pairs": "突变引入的新二硫键(pairs 的子集); 非空时日志会 ⚠ 预警。",
            "manifest": "--inherit_box 生成的可批跑 manifest 路径(直接 dock --manifest 即跑)。",
            "manifest_entry": "生成的 manifest 条目(id/receptor/盒子/seed 等全套继承字段)。",
            "eval_dock": "共晶配体双边对接有效性评估(WT vs 突变体, 需受体带共晶配体; AF 模型自动跳过)。",
            "eval_smiles_dock": "外源配体对照评估(--eval_smiles; 同配体/盒子双边对接, 多种子 ΔE mean±std + 锚定迁移发生率 + 催化取向判词)。不做等效判定(无晶体基准)。",
        },
    },

    # ------------------------------------------------------------------
    # pocket-tunnel: tunnel.json
    # ------------------------------------------------------------------
    "pocket_tunnel": {
        "title": "口袋 + 入口通道几何对比 (tunnel.json)",
        "key_metrics": ["verdict.cavity_delta_A3", "verdict.cavity_delta_pct",
                        "verdict.bottleneck_moved", "verdict.narrative"],
        "dynamic": ["tunnel.*.channel_profile.buckets_A"],
        "fields": {
            "pocket_center_A": "口袋球心坐标 [x,y,z](Å; = WT 侧亲核原子位置)。",
            "pocket_center_source": "球心来源标注(如 SER101A:OG —— 催化亲核原子, 正式口径; 与 mutate QC 的 Cα 质心口径不同)。",
            "mutated_residues": "由两份 PDB 序列差异反推的突变位点标签; 喉部残基差集若全部落在这里 →『喉部搬家』判为位点替换的 artifact。",
            "verdict": "compare_sides 判词(唯一叙事出口, mutate QC 与 report 共用), 字段见 verdict.*。",
            "verdict.ok": "判词是否可用(False 时几何数字仍有效, 但不能写结论)。",
            "verdict.radius_A": "口袋球半径(Å; 两侧同半径)。",
            "verdict.center_source": "球心来源(同 pocket_center_source)。",
            "verdict.same_cavity": "两侧是不是同一条空腔(球心 gap 判定; False 时 Δ 体积不可比)。",
            "verdict.seed_gap_A": "两侧口袋球心间距(Å); 小=同腔对照成立。",
            "verdict.both_connected_to_bulk": "两侧是否都经入口通道连到体相(死端腔对降解零贡献)。",
            "verdict.cavity_wt_A3": "WT 侧空腔体积(Å³; 正式口径 cavity_volume_A3)。",
            "verdict.cavity_mut_A3": "突变体侧空腔体积(Å³)。",
            "verdict.cavity_delta_A3": "空腔差量(Å³; mut − wt, 正=扩大)。",
            "verdict.cavity_delta_pct": "空腔差量百分比(%)。",
            "verdict.bottleneck_wt_A": "WT 瓶颈半径(Å)。",
            "verdict.bottleneck_mut_A": "突变体瓶颈半径(Å)。",
            "verdict.bottleneck_delta_A": "瓶颈半径差量(Å; |Δ|≥0.5 才算真换档)。",
            "verdict.bottleneck_interval_wt": "WT 瓶颈真值区间(如 [1.4, 1.6); capped 口径=中断前一档)。",
            "verdict.bottleneck_interval_mut": "突变体瓶颈真值区间。",
            "verdict.bottleneck_moved": "喉部位置是否**判定**为搬家(过了 replicate 噪声地板 8.0 Å 且非 artifact 才 True)。",
            "verdict.bottleneck_moved_raw": "未经地板守卫的原始判定(参考量, 禁止单独引用)。",
            "verdict.bottleneck_move_artifact": "True=喉部残基差集全落在突变位点自身(位点被替换当然从列表消失), 判为 artifact 不算搬家。",
            "verdict.bottleneck_diff_residues": "两侧喉部残基差集(标签列表)。",
            "verdict.bottleneck_move_A": "两侧喉部坐标距离(Å; 对坐标级扰动敏感, 须过 8.0 Å 地板才可引用)。",
            "verdict.bottleneck_replicate_noise_A": "喉部复现噪声地板(8.0 Å; 由序列不变对照的多次松弛实测)。",
            "verdict.replicate_required": "True=喉部类结论需 ≥2 个独立松弛模型重复且终点一致才可写(单样本喉部量是噪声)。",
            "verdict.bottleneck_residues_wt": "WT 喉部残基列表。",
            "verdict.bottleneck_residues_mut": "突变体喉部残基列表。",
            "verdict.entry_wider": "入口是否**判定**变宽(两侧瓶颈同档 capped 时恒 False —— 此前『入口变宽』类表述是误读)。",
            "verdict.banned_phrases": "⛔ 禁写清单(基于本份数据不能写的话, 如『入口变宽』『喉部搬家』)。",
            "verdict.allowed_phrases": "✅ 可写清单(本份数据支持的表述)。",
            "verdict.narrative": "判词唯一叙事句(报告直接引用, 不许各写各的)。",
            "qc_crosscheck": "历史 mutate QC 口径(Cα 质心球心)交叉核对, 字段见 qc_crosscheck.*。⚠ 其体积与 tunnel.* 的 Ser OG 球心体积禁止相减/混用。",
            "qc_crosscheck.center_A": "QC 球心(三联体 Cα 质心)坐标。",
            "qc_crosscheck.radius_A": "QC 球半径(Å)。",
            "qc_crosscheck.step_A": "QC 网格步长(Å)。",
            "qc_crosscheck.probe_A": "QC 探针半径(Å)。",
            "qc_crosscheck.free_volume_wt_A3": "QC 口径 WT 自由体积(Å³)。",
            "qc_crosscheck.free_volume_mut_A3": "QC 口径突变体自由体积(Å³)。",
            "qc_crosscheck.delta_A3": "QC 口径差量(Å³; 只可和历史 mutate QC 对照)。",
            "qc_crosscheck.delta_pct": "QC 口径差量百分比(%)。",
            "qc_crosscheck.note": "口径与限制说明文本。",
            "tunnel": "两侧几何明细, 字段见 tunnel.wt.* / tunnel.mut.*(结构对称)。",
            "tunnel.pocket_center_A": "口袋球心坐标副本(与顶层 pocket_center_A 同值, _print_report 的入参约定)。",
            "tunnel.pocket_center_source": "球心来源副本(同上)。",
            "tunnel.wt": "WT 侧几何明细(结构与 tunnel.mut 完全对称)。",
            "tunnel.mut": "突变体侧几何明细(结构与 tunnel.wt 完全对称)。",
            "tunnel.wt.connected_to_bulk": "口袋是否连通体相(开放)。",
            "tunnel.wt.domain_reaches_shell": "口袋域是否触到外壳层。",
            "tunnel.wt.n_entrance_grid": "腔内触壳网格点数。",
            "tunnel.wt.n_shell_grid": "外壳层网格点数。",
            "tunnel.wt.exit_criterion": "出口判据文本(口袋球内 6-连通至外壳层)。",
            "tunnel.wt.connected_to_bulk_scope": "连体相判定的网格域文本(全蛋白两档 flood 口径)。",
            "tunnel.wt.n_bulk_boundary_free_grid": "盒边界体相自由网格数。",
            "tunnel.wt.bulk_domain_scope": "体相域文本(全蛋白包围盒 ±10 Å, 与 --radius 解耦 —— 换 radius 不影响)。",
            "tunnel.wt.dead_end_frac": "死端占比(0=全部体积都通体相)。",
            "tunnel.wt.pdb": "分析的受体文件名。",
            "tunnel.wt.chain": "分析的受体链(缺省自动选最长链)。",
            "tunnel.wt.seed_A": "口袋球心坐标(= 亲核原子)。",
            "tunnel.wt.seed_source": "球心来源(SER101A:OG 等)。",
            "tunnel.wt.step_A": "网格步长(0.6 Å, 正式口径)。",
            "tunnel.wt.probe_A": "探针半径(1.4 Å)。",
            "tunnel.wt.sphere_radius_A": "口袋球半径(Å)。",
            "tunnel.wt.n_cavity_grid": "腔内网格点数。",
            "tunnel.wt.cavity_volume_A3": "空腔体积(Å³) —— **报告唯一可引用的体积口径**(与 QC 的 free_volume 不同球心)。",
            "tunnel.wt.free_volume_in_sphere_A3": "球内自由体积(Å³; 与腔体积的网格口径微差)。",
            "tunnel.wt.max_radius_in_cavity_A": "腔内最大可容探针半径(Å)。",
            "tunnel.wt.bottleneck_radius_A": "瓶颈半径(Å; 膨胀扫描中断前一档的 capped 值)。",
            "tunnel.wt.bottleneck_capped": "瓶颈是否为 capped 口径(True 时真值落在 bottleneck_interval_A 内)。",
            "tunnel.wt.bottleneck_interval_A": "瓶颈真值区间(如 [1.4, 1.6))。",
            "tunnel.wt.bottleneck_xyz_A": "喉部坐标 [x,y,z](Å) —— 对扰动敏感, 迁移距离类结论须过噪声地板。",
            "tunnel.wt.bottleneck_residues": "喉部残基列表(构成最窄处的残基)。",
            "tunnel.wt.channel_profile": "通道剖面(沿最短出口路径逐档余量), 字段见 tunnel.wt.channel_profile.*。",
            "tunnel.wt.channel_profile.n_step": "剖面采样步数。",
            "tunnel.wt.channel_profile.path_len_A": "通道长度(Å)。",
            "tunnel.wt.channel_profile.r_min_A": "沿途最小余量半径(Å; 与瓶颈同源)。",
            "tunnel.wt.channel_profile.r_min_at_A": "最小余量出现位置坐标(Å)。",
            "tunnel.wt.channel_profile.r_mean_A": "沿途平均余量半径(Å)。",
            "tunnel.wt.channel_profile.buckets_A": "余量分桶统计(键 = 桶标签)。",
            "tunnel.wt.channel_profile.buckets_A.*": "该桶标签覆盖的路径步数(桶: <1.4 / >=probe,<2.0 / >=2.0,<3.0 / >=3.0, 单位 Å)。",
            "tunnel.wt.channel_profile.r_at_cavity_end_A": "腔端余量(Å)。",
            "tunnel.wt.channel_profile.r_at_exit_end_A": "出口端余量(Å)。",
            "tunnel.wt.n_gridpoints": "总网格点数。",
            "tunnel_wider": "自动放大域(默认域两侧不通外壳时逐级放大 +4/+8/+12 Å 取『两侧都通』的最小域; extended_radius_A=采用的半径, tunnel=该域两侧几何; note=各档都不通时的说明)。",
            "ligand": "底物横截面分析(--ligand 给底物时), 字段见 ligand.*。",
            "ligand.ligand": "底物文件名。",
            "ligand.max_pair_dist_A": "最大原子对距离(Å)。",
            "ligand.extent_long_axis_A": "主轴尺度(Å)。",
            "ligand.extent_mid_axis_A": "中轴尺度(Å)。",
            "ligand.extent_short_axis_A": "短轴尺度(Å)。",
            "ligand.effective_radius_A": "有效半径(Å; governing_diameter 的一半口径, 与瓶颈/底物半径比用)。",
            "radius_used_A": "实际采用的口袋球半径(Å; 自动放大生效时=放大后的半径)。",
        },
    },

    # ------------------------------------------------------------------
    # pocket-decomp: decomp.json
    # ------------------------------------------------------------------
    "pocket_decomp": {
        "title": "空腔增益逐残基分解 (pocket_decomp)",
        "key_metrics": ["cavity_delta_A3", "released_total_A3",
                        "backfill_total_A3", "self_consistent"],
        "fields": {
            "ok": "分解是否成功。",
            "radius_A": "口袋球半径(Å; 与 pocket-tunnel 同口径, 两侧同半径)。",
            "step_A": "体素步长(0.6 Å)。",
            "probe_A": "探针半径(1.4 Å)。",
            "wt": "WT 受体路径(绝对路径)。",
            "mut": "突变体受体路径。",
            "seed_wt": "WT 口袋球心坐标(= 亲核原子; 两侧同腔对照的锚)。",
            "seed_mut": "突变体球心坐标。",
            "seed_source_wt": "WT 球心来源(如 SER101A:OG)。",
            "seed_source_mut": "突变体球心来源。",
            "seed_gap_A": "两侧球心间距(Å; 小=同一空腔, Δ 才有意义)。",
            "cavity_wt_A3": "WT 空腔体积(Å³)。",
            "cavity_mut_A3": "突变体空腔体积(Å³)。",
            "cavity_delta_A3": "空腔净差量(Å³; = mut − wt)。",
            "sphere_grid_volume_wt_A3": "WT 侧球内网格总体积(Å³; 自洽校验用)。",
            "sphere_grid_volume_mut_A3": "突变体侧球内网格总体积(Å³)。",
            "grid_count_delta_A3": "网格离散项差量(Å³; 步长一致时应为 0)。",
            "sum_contributions_A3": "全部残基贡献之和(Å³)。",
            "residual_A3": "自洽残差(Å³; Σ贡献 + 网格项 − 实测差量, 应≈0)。",
            "self_consistent": "自洽恒等式是否成立(False 时本份分解不可引用)。",
            "mutated_residues": "突变位点标签列表。",
            "mutated_rows": "突变位点自身的贡献行(字段同 per_residue[i].*)。",
            "released_total_A3": "腾出合计(Å³; 被替换残基及邻域让出的体积)。",
            "backfill_total_A3": "回填合计(Å³; 松弛后其它残基侧链移进来占掉的体积, 负值)。",
            "released_top": "腾出 Top N 残基行(按贡献绝对值排序)。",
            "backfill_top": "回填 Top N 残基行。",
            "per_residue": "全部参与残基的逐残基贡献行, 字段见 per_residue[i].*。",
            "per_residue[i].label": "残基标签(如 PHE101A)。",
            "per_residue[i].resname_wt": "WT 侧残基名。",
            "per_residue[i].resname_mut": "突变体侧残基名。",
            "per_residue[i].occupancy_wt_A3": "WT 侧该残基占据的球内体积(Å³)。",
            "per_residue[i].occupancy_mut_A3": "突变体侧占据体积(Å³)。",
            "per_residue[i].delta_occupancy_A3": "占据差量(Å³; 负=腾出)。",
            "per_residue[i].contribution_A3": "该残基对空腔净增的贡献(Å³; 正=腾出/负=回填)。",
            "n_residue_reported": "报告的残基数(过了 min_report 阈值的)。",
            "n_residue_total": "球内残基总数。",
            "min_report_A3": "报告阈值(Å³; 贡献绝对值小于它的残基折叠不列)。",
            "banned_phrases": "⛔ 禁写清单。",
            "allowed_phrases": "✅ 可写清单。",
            "replicate_required": "True=单次松弛的分解只作 QC, 正式引用需独立重复。",
            "narrative": "分解叙事句(腾出+回填+网格项的恒等式)。",
        },
    },

    # ------------------------------------------------------------------
    # deacylation: deacylation.json
    # ------------------------------------------------------------------
    "deacylation": {
        "title": "去酰化水可及性 (deacylation)",
        "key_metrics": ["verdict.frac_delta", "verdict.vol_delta_A3",
                        "verdict.narrative"],
        "fields": {
            "wt": "WT 侧分析(结构与 mut 完全对称), 字段见 wt.*。",
            "mut": "突变体侧分析(结构与 wt 完全对称)。",
            "wt.ok": "该侧分析是否成功。",
            "wt.rec": "受体路径。",
            "wt.poses": "poses 文件路径(results.pdbqt)。",
            "wt.n_poses": "分析的 pose 总数(≤ --max_poses)。",
            "wt.n_oriented": "oriented pose 数(酯羰基碳进亲核 ≤3.5 Å 的 pose)。",
            "wt.n_poses_with_water": "存在合格去酰化水位点的 pose 数。",
            "wt.frac_poses_with_water": "带水位点的 pose 占比(0~1)。",
            "wt.total_clusters": "合格水位置的水簇总数(跨全部 pose)。",
            "wt.qualified_volume_A3": "合格水空间总体积(Å³)。",
            "wt.qualified_volume_oriented_A3": "oriented pose 内的合格水体积(Å³; 主分母口径)。",
            "wt.volume_per_oriented_pose_A3": "每 oriented pose 的平均合格水体积(Å³; **主指标** —— 去酰化水可及性)。",
            "wt.clusters_per_oriented_pose": "每 oriented pose 的平均水簇数。",
            "wt.per_pose": "逐 pose 明细, 字段见 wt.per_pose[i].*。",
            "wt.per_pose[i].pose_index": "pose 序号。",
            "wt.per_pose[i].energy": "该 pose 能量(kcal/mol)。",
            "wt.per_pose[i].oriented": "是否 oriented(亲核进攻位)。",
            "wt.per_pose[i].nearest_A": "水到亲核/His 的最近距离(Å)。",
            "wt.per_pose[i].n_ester_sites": "配体酯位点数。",
            "wt.per_pose[i].n_clusters": "该 pose 的合格水簇数。",
            "wt.per_pose[i].qualified_volume_A3": "该 pose 的合格水体积(Å³)。",
            "wt.per_pose[i].best": "是否该侧最优 pose。",
            "wt.params": "判据参数, 字段见 wt.params.*。",
            "wt.params.water_clearance_A": "水分子 clearance 半径(Å)。",
            "wt.params.nuc_range_A": "亲核进攻距离窗(Å)。",
            "wt.params.bd_angle_deg": "Bürgi-Dunitz 角窗(度)。",
            "wt.params.his_water_max_A": "His NE2 到水距离上限(Å)。",
            "wt.params.search_radius_A": "水位点搜索半径(Å)。",
            "wt.params.step_A": "网格步长(Å)。",
            "verdict": "compare_sides_deacylation 判词(唯一出口), 字段见 verdict.*。",
            "verdict.ok": "判词是否可用。",
            "verdict.frac_wt": "WT 带水位点 pose 占比。",
            "verdict.frac_mut": "突变体带水位点 pose 占比。",
            "verdict.frac_delta": "占比差量(mut − wt); 低于 0.15 噪声地板不可写。",
            "verdict.vol_per_oriented_wt_A3": "WT 每 oriented pose 水体积(Å³)。",
            "verdict.vol_per_oriented_mut_A3": "突变体每 oriented pose 水体积(Å³)。",
            "verdict.vol_delta_A3": "水体积差量(Å³; 低于 12 Å³ 噪声地板不可写)。",
            "verdict.clusters_wt": "WT 水簇总数。",
            "verdict.clusters_mut": "突变体水簇总数。",
            "verdict.clusters_delta": "水簇差量。",
            "verdict.n_poses_wt": "WT pose 数。",
            "verdict.n_poses_mut": "突变体 pose 数。",
            "verdict.n_oriented_wt": "WT oriented pose 数。",
            "verdict.n_oriented_mut": "突变体 oriented pose 数。",
            "verdict.replicate_required": "True=需多种子/多轮重复才可进叙事(单轮 9 pose 的稀有事件统计不稳)。",
            "verdict.banned_phrases": "⛔ 禁写清单(如未过地板时禁写『去酰化水空间变化』)。",
            "verdict.allowed_phrases": "✅ 可写清单。",
            "verdict.narrative": "判词叙事句。",
        },
    },

    # ------------------------------------------------------------------
    # msa-report: site_report.json
    # ------------------------------------------------------------------
    "msa_site_report": {
        "title": "突变位点家族保守性 (msa-report)",
        "key_metrics": [],
        "dynamic": ["sites.*.aa_freqs", "sites.*.top3_variants"],
        "fields": {
            "a3m": "输入 A3M 路径(首条 = query, 列号即残基号)。",
            "sites": "逐位点统计(键 = 残基号), 字段见 sites.*.*。",
            "sites.*.n_total": "该列非 gap 序列总数(含 query)。",
            "sites.*.n_effective": "有效序列数(去冗余权重; 不足 min_effective 判读无效)。",
            "sites.*.gap_frac": "gap 占比(高 = 该位点覆盖不足, 判读降权)。",
            "sites.*.top_aa": "最常见氨基酸(单字码)。",
            "sites.*.top_freq": "其频率(0~1; ≥conserved_thr 判保守)。",
            "sites.*.is_conserved": "是否保守(判定阈值 --conserved_thr, 默认 0.9)。",
            "sites.*.aa_freqs": "全部氨基酸频率表(键 = 单字码)。折叠为 sites.*.aa_freqs.*。",
            "sites.*.aa_freqs.*": "某氨基酸的频率(键 = 单字码 G/P/S/A/…, 值 = 0~1)。",
            "sites.*.verdict": "该位点判读句(如『高度保守(G 98.7%)』)。",
            "sites.*.small_residue_frac": "天然小残基(G/A/S/C/T/P)占比(0~1; **理性设计依据**: 家族里小侧链可行=突变到 Gly 不破坏折叠的直接证据)。null=覆盖不足。",
            "sites.*.top3_variants": "频率前 3 的变体(键 = 单字码)。折叠为 sites.*.top3_variants.*。",
            "sites.*.top3_variants.*": "某变体的频率(键 = 单字码, 值 = 0~1)。",
            "small_residues": "小残基集合(G/A/S/C/T/P)。",
        },
    },

    # ------------------------------------------------------------------
    # product-egress: egress.json
    # ------------------------------------------------------------------
    "product_egress": {
        "title": "产物释放可及性 (product-egress)",
        "key_metrics": [],
        "dynamic": ["products[i].verdicts"],
        "fields": {
            "params": "参数(键 = ph: 产物质子化 pH, 默认 7.4 羧酸根口径)。",
            "params.ph": "产物质子化 pH。",
            "tunnel_sides": "通道侧数据(逐受体), 字段见 tunnel_sides[i].*。",
            "tunnel_sides[i].rec": "受体路径(瓶颈数据来源)。",
            "tunnel_sides[i].bottleneck_radius_A": "瓶颈半径(Å, capped 口径)。",
            "tunnel_sides[i].bottleneck_capped": "瓶颈是否 capped(True 时真值在 bottleneck_interval_A 内)。",
            "tunnel_sides[i].bottleneck_interval_A": "瓶颈真值区间。",
            "tunnel_sides[i].connected_to_bulk": "通道是否连通体相。",
            "tunnel_sides[i].cavity_volume_A3": "空腔体积(Å³)。",
            "products": "逐产物分析, 字段见 products[i].*。",
            "products[i].product": "产物名(如水解产物单体/寡聚片段)。",
            "products[i].smiles": "产物 SMILES。",
            "products[i].product_class": "产物类型(monocarboxylic_acid=单羧酸 / dicarboxylic_acid=二酸 / aliphatic_diacid=脂肪族二酸 / diol=二醇; 自定义配体为 null)。F205G 机理 3 真正关心的是单羧酸中间体 MHET 与二酸终产物 TPA 这两类芳香产物。",
            "products[i].aromatic": "是否含芳香环(true 关联机理 3 芳香滞留; 自定义配体为 null)。",
            "products[i].role": "产物角色(monocarboxylate_intermediate=单羧酸中间体(如 MHET) / terminal_diacid=终产物二酸(如 TPA) / aliphatic_diacid=脂肪族二酸 / diol=二醇)。",
            "products[i].informative": "是否对叙事有信息量(EG 等极小产物截面恒 < 瓶颈上界、永远『可能通过』, 无区分力, 设为 false, 报告不应据此支撑结论)。",
            "products[i].governing_diameter_A": "产物最小截面的控制直径(Å; 单构象 gen3d, 不是系综)。",
            "products[i].effective_radius_A": "有效半径(Å; = 控制直径/2, 与瓶颈半径直接比)。",
            "products[i].n_heavy": "重原子数。",
            "products[i].verdicts": "逐受体三态判词(键 = 受体路径)。折叠为 products[i].verdicts.*。",
            "products[i].verdicts.*": "某受体的判词对象: passable 三态(True=可能通过 / False=过不去 / null=无法排除, capped 口径限制) + why 判据明细句(截面 vs 瓶颈上界)。",
            "note": "口径说明(capped 上界语义: 『产物≤上界』不等于『能通过』)。",
        },
    },

    # ------------------------------------------------------------------
    # compare: compare_summary.json
    # ------------------------------------------------------------------
    "compare_summary": {
        "title": "四件套聚合对比汇总 (compare_summary.json)",
        "key_metrics": ["ok", "pocket_tunnel.cavity_delta_A3",
                        "pocket_decomp.cavity_delta_A3",
                        "pocket_tunnel.narrative"],
        "fields": {
            "wt": "WT/对照受体路径。",
            "mut": "突变体受体路径。",
            "chain": "链(auto=自动)。",
            "radius_A": "口袋球半径(Å; tunnel/decomp 共用, 两侧同半径)。",
            "items": "分项执行状态(键 = pocket_tunnel/pocket_decomp/product_egress/deacylation; 值: ok=是否成功, json=分项 JSON 路径, error=失败原因)。",
            "ok": "全部分项是否成功(有分项失败为 False)。",
            "out_dir": "产物目录(compare/<WT名>_vs_<突变体名>/)。",
            "pocket_tunnel": "从落盘 tunnel.json 读回的通道摘要(字段同 pocket-tunnel 的 verdict.*: cavity_wt_A3/cavity_mut_A3/cavity_delta_A3/bottleneck_wt_A/bottleneck_mut_A/narrative/banned_phrases/allowed_phrases)。",
            "pocket_decomp": "从落盘 decomp.json 读回的分解摘要(cavity_delta_A3/released_total_A3/backfill_total_A3/mutated_rows/self_consistent)。",
            "product_egress": "从落盘 egress.json 读回的产物摘要(products[].product/governing_diameter_A/verdicts)。",
            "deacylation": "从落盘 deacylation.json 读回的去酰化摘要(vol_per_oriented_wt_A3/vol_per_oriented_mut_A3/narrative; 仅两侧 poses 都给时存在)。",
            "summary_json": "本汇总 JSON 自身路径。",
        },
    },

    # ------------------------------------------------------------------
    # retention: retention_summary.json (§2.3 芳香产物滞留解除定量)
    # ------------------------------------------------------------------
    "retention_summary": {
        "title": "芳香产物滞留解除定量 (retention_summary.json)",
        "key_metrics": ["summary.delta_E_mean", "summary.aromatic_retention.wt.mean",
                        "summary.aromatic_retention.mut.mean",
                        "verdict.retention_frac_delta"],
        "fields": {
            "available": "本份分析是否成功走完全流程(配体/受体/盒子/双边对照对接/汇总)。False 时其余字段可能缺失。",
            "mode": "对照模式('smiles': 单产物 SMILES 双边对照; 'product_set': 多产物批量对照, 产物流逐产物结果)。",
            "smiles": "单产物模式下的产物 SMILES(如 TPA 'O=C(O)c1ccc(C(=O)O)cc1' 或 MHET); product_set 模式下为 null。",
            "product_set": "product_set 模式下的产物名列表(顺序与 products 键一致; 不填 --smiles 时为内置芳香产物集 TPA+MHET)。",
            "products": "product_set 模式逐产物结果(键=产物名, 值=单产物 retention_summary 结构: available/smiles/box_center/summary/verdict)。",
            "box_center": "实际使用盒心坐标 [x,y,z](Å; = ref 侧三联体 Cα 质心)。",
            "box_size": "盒尺寸 [x,y,z](Å; 双侧共用 ref 三联体盒心)。",
            "box_source": "盒心来源标注('triad_ca_centroid(ref)')。",
            "seeds": "对接种子列表(逐种子明细在 runs)。",
            "exhaustiveness": "Vina 穷举度(缺省 8; 论文确认层 exh=32)。",
            "num_modes": "输出 pose 数(缺省 9; 滞留占比需全 pose 统计)。",
            "vina_cpu": "Vina CPU 钉死数(对照链路恒 1, 保证同种子可复现)。",
            "retention_stack_A": "芳香滞留接触阈值(Å; 配体芳香碳→受体芳香碳 ≤ 此值判滞留; 默认 5.0, 可由 DOCK_RETENTION_STACK_A 覆盖)。",
            "retention_noise_floor": "滞留占比差的噪声地板(比例量; 默认 0.15, |Δ|<地板禁写方向性结论)。",
            "runs": "逐种子明细, 字段见 runs[i].*。",
            "runs[i].seed": "该轮种子。",
            "runs[i].delta_E_alt_minus_ref": "该轮 ΔE(突变体−WT, kcal/mol; 同盒心可比)。",
            "runs[i].ref": "WT 侧该轮对接结果(结构与 runs[i].alt 对称), 字段见 runs[i].ref.*。",
            "runs[i].alt": "突变体侧该轮对接结果。",
            "runs[i].ref.energy_kcal_mol": "该侧 top1 能量(kcal/mol)。",
            "runs[i].ref.orientation": "该侧催化取向汇总(summarize_side 结果; 含 oriented_frac 等), 字段同 orientation 模块口径。",
            "runs[i].ref.aromatic_retention_frac": "该侧芳香滞留占比(全 pose 中滞留 pose 占比 0~1); 产物芳香环仍卡在芳香锚定接触内的比例。",
            "runs[i].ref.aromatic_retention_nearest_A": "该侧配体芳香碳→受体芳香碳 最小距离(Å; 诊断用)。",
            "runs[i].ref.n_pose": "该侧分析 pose 数。",
            "runs[i].ref.n_retained": "该侧滞留 pose 数。",
            "runs[i].ref.arom_name_count": "该产物芳香碳数(0=无芳香环, 滞留指标不适用)。",
            "runs[i].ref.chain_used": "实际使用的受体链。",
            "summary": "跨种子聚合, 字段见 summary.*。",
            "summary.n_seeds": "汇总用的种子数。",
            "summary.delta_E_mean": "ΔE(突变体−WT)均值(kcal/mol); 与能量同口径比较。",
            "summary.delta_E_std": "ΔE 标准差(多种子复现性)。",
            "summary.delta_E_min": "ΔE 最小值。",
            "summary.delta_E_max": "ΔE 最大值。",
            "summary.ref_energy_mean": "WT 侧 top1 能量均值(kcal/mol)。",
            "summary.alt_energy_mean": "突变体侧 top1 能量均值(kcal/mol)。",
            "summary.oriented": "催化取向占比跨种子聚合, 字段见 summary.oriented.wt/mut.*(均含 mean/std/n)。",
            "summary.oriented.wt": "WT 侧 oriented_frac 聚合({mean,std,n})。",
            "summary.oriented.mut": "突变体侧 oriented_frac 聚合。",
            "summary.aromatic_retention": "芳香滞留占比跨种子聚合, 字段见 summary.aromatic_retention.wt/mut.*。",
            "summary.aromatic_retention.wt": "WT 侧 aromatic_retention_frac 聚合({mean,std,n})。",
            "summary.aromatic_retention.mut": "突变体侧 aromatic_retention_frac 聚合。",
            "verdict": "retention_sides 自否证判词(唯一叙事出口), 字段见 verdict.*。",
            "verdict.ok": "判词是否可用(False 时数字仍有效, 但不能写结论)。",
            "verdict.retention_applicable": "滞留指标是否适用(两侧产物均无芳香碳时为 False)。",
            "verdict.noise_floor": "滞留占比差噪声地板(0.15)。",
            "verdict.oriented_frac_wt": "WT 侧 oriented_frac(数值)。",
            "verdict.oriented_frac_mut": "突变体侧 oriented_frac(数值)。",
            "verdict.oriented_frac_delta": "oriented_frac 差值(突变体−WT)。",
            "verdict.retention_frac_wt": "WT 侧芳香滞留占比(数值)。",
            "verdict.retention_frac_mut": "突变体侧芳香滞留占比(数值)。",
            "verdict.retention_frac_delta": "芳香滞留占比差值(突变体−WT); 负值=突变体滞留更少=滞留解除。",
            "verdict.banned_phrases": "⛔ 禁写清单(基于本份数据不能写的话, 如『滞留解除』『kcat 提升』)。",
            "verdict.allowed_phrases": "✅ 可写清单(本份数据支持的表述, 须带占比数字与地板)。",
            "verdict.narrative": "判词唯一叙事句(报告直接引用, 不许各写各的)。",
            "note": "口径说明: 外源产物无晶体基准不判定等效; 对照须同机同环境; 滞留占比低于噪声地板不得写方向性结论。",
        },
    },

    # ------------------------------------------------------------------
    # conformation-entropy: conformation_entropy.json (§2.3 局部构象熵差, NMA/准谐振)
    # ------------------------------------------------------------------
    "conformation_entropy": {
        "title": "局部构象熵差 WT vs 突变体 (conformation_entropy.json)",
        "key_metrics": ["available", "delta_S_J_per_mol_K",
                        "verdict.ok", "verdict.replicate_required"],
        "fields": {
            "available": "本份分析是否成功走完全流程(WT/突变体选定原子集合的局部受限极小化 + 有限差分质量加权 Hessian 对角化 → 谐振子振动熵差)。False 时其余字段可能缺失, 看 error。",
            "mode": "对照模式(固定 'local_nma': 局部受限极小化 + 有限差分质量加权 Hessian → 谐振子振动熵 S = R·Σ[x/tanh x − ln(2·sinh x)])。",
            "selection": "选定原子集合描述(默认 'triad_sphere_r<半径>' = 以催化三联体 Cα 质心为球心、select_radius_A 为半径的球形邻域; 或 --select 指定的该 PDB 实际残基编号串)。",
            "n_selected": "选定原子数(局部 Hessian 3n×3n 自由度来源; 冻结环境近似下量化的是该集合的局部柔性差)。",
            "select_radius_A": "默认球形邻域半径(Å; 围绕三联体 Cα 质心; 免疫 AF 模型『规范编号 vs 文件编号』不一致)。",
            "solvent": "隐式溶剂(obc=GBneck2 / vacuum); 不同溶剂下的绝对熵不可比, 只比同协议 ΔS。",
            "ff": "力场(固定 'amber14-all.xml')。",
            "T_K": "振动熵计算温度(K; 振动熵对 T 敏感)。",
            "fd_step_nm": "有限差分步长(nm; Hessian 数值误差量级, 与极小化极小值附近的曲率估计直接相关)。",
            "restraint_k": "非选定原子的固定谐波约束(kJ/mol/nm²; 冻结环境近似的来源 —— 给局部集体低频模引入固有偏差)。",
            "noise_floor_J_per_mol_K": "ΔS 噪声地板(J/mol·K; 默认 8.0, 可由 DOCK_CONF_ENTROPY_NOISE_FLOOR_J 覆盖)。|ΔS| 低于它禁写任何方向性结论。",
            "wt": "WT 侧局部振动熵明细(结构与 mut 完全对称), 字段见 wt.*。",
            "wt.ok": "WT 侧局部熵计算是否成功(力场/加氢/极小化/力查询任一失败则 False)。",
            "wt.S_J_per_mol_K": "WT 侧局部振动熵(J/(mol·K))。",
            "wt.n_modes": "参与求和的正定振动模数(剔除零/负本征值后)。",
            "wt.n_zero_or_neg_modes": "被剔除的零/负本征值模数(数值零模 + 真实平移/旋转约束残差)。",
            "wt.n_selected": "WT 侧选定原子数(= 顶层 n_selected)。",
            "wt.n_total": "该侧体系总原子数。",
            "wt.solvent": "WT 侧溶剂(逐侧回显, 复现用)。",
            "wt.ff": "WT 侧力场。",
            "wt.T_K": "WT 侧温度(K)。",
            "wt.fd_step_nm": "WT 侧有限差分步长(nm)。",
            "wt.hessian_ok": "Hessian 构造与对角化是否成功(恒 True; 失败侧走 ok=False, 不出现此键)。",
            "mut": "突变体侧局部振动熵明细(结构与 wt 完全对称), 字段见 mut.*。",
            "mut.ok": "突变体侧局部熵计算是否成功。",
            "mut.S_J_per_mol_K": "突变体侧局部振动熵(J/(mol·K))。",
            "mut.n_modes": "突变体侧正定振动模数。",
            "mut.n_zero_or_neg_modes": "突变体侧被剔除模数。",
            "mut.n_selected": "突变体侧选定原子数。",
            "mut.n_total": "突变体侧总原子数。",
            "mut.solvent": "突变体侧溶剂。",
            "mut.ff": "突变体侧力场。",
            "mut.T_K": "突变体侧温度(K)。",
            "mut.fd_step_nm": "突变体侧有限差分步长(nm)。",
            "mut.hessian_ok": "突变体侧 Hessian 构造与对角化是否成功。",
            "delta_S_J_per_mol_K": "构象熵差 ΔS = S_mut − S_wt(J/(mol·K)); null = 任一侧失败或判定不适用。负值=突变体更僵, 正值=更松。",
            "verdict": "entropy_sides 自否证判词(唯一叙事出口), 字段见 verdict.*。",
            "verdict.ok": "判词是否可用(False 时数字仍有效, 但不能写结论)。",
            "verdict.applicable": "指标是否适用(两侧 S 都算成功即 True; 任一失败则 False 且 ok=False)。",
            "verdict.noise_floor": "ΔS 噪声地板(默认 8.0 J/mol·K)。",
            "verdict.S_wt": "WT 侧振动熵数值回显(与顶层同)。",
            "verdict.S_mut": "突变体侧振动熵数值回显。",
            "verdict.delta_S": "ΔS 数值回显(= 顶层 delta_S_J_per_mol_K)。",
            "verdict.banned_phrases": "⛔ 禁写清单(基于本份数据不能写的话, 如『地板下禁写构象熵差结论』『把 ΔS 读成 kcat/活性』『仅凭单次极小化下结论』『把绝对熵当物理量』)。",
            "verdict.allowed_phrases": "✅ 可写清单(本份数据支持的表述, 须带 ΔS 数字与地板、注明『冻结环境局部 NMA』口径)。",
            "verdict.replicate_required": "True=有限差分 Hessian 数值误差大、极小化对初值/约束敏感, 必须 ≥2 次独立极小化重复且 ΔS 符号一致才可进叙事(单样本 ΔS 是噪声)。",
            "verdict.narrative": "判词唯一叙事句(报告直接引用, 不许各写各的)。",
            "note": "口径说明: 局部 NMA(冻结环境近似)仅量化选定残基集合的局部柔性差, 非全蛋白; 绝对熵不可比, 仅同协议 ΔS 可解释; ΔS 超地板仍需独立重复确认符号。",
            "error": "不可用时的原因串(如 'WT: <err> | MUT: <err>' 或选择失败); 仅 available=False 时存在。",
        },
    },

    # ------------------------------------------------------------------
    # protonation: protonation.json (受体精确质子化, PROPKA/缺失优雅降级)
    # ------------------------------------------------------------------
    "protonation": {
        "title": "受体精确质子化 (protonation.json)",
        "key_metrics": ["ok", "status", "precise", "backend"],
        "fields": {
            "ok": "本份质子化是否成功(精确或启发式产物已写出)。False 时看 status/error。",
            "status": "质子化状态: 'precise'(PROPKA pKa 感知精确路径) / 'skipped'(退回固定 pH 启发式加氢, 非精确) / 'unavailable'(显式要求 propka 但不可用) / 'error'(受体缺失等)。",
            "precise": "是否 pKa 感知的精确质子化(True 仅当 status='precise')。False 时下游报告**不得**声称『精确质子化』。",
            "backend": "实际使用的后端('propka' / 'heuristic_addH')。",
            "backend_fallback": "若精确路径异常后退回启发式, 此键记 'heuristic_addH'(说明曾尝试精确)。",
            "method": "方法描述串(精确: 'pdb2pqr+propka pKa-aware @pH=…'; 启发式: 'amber14-all addHydrogens @pH=… (…非 pKa 感知)')。",
            "out_pdb": "质子化后 PDB 输出路径。",
            "ph": "质子化 pH。",
            "chain": "限定链标注(仅记录, 不影响加氢)。",
            "n_heavy": "产物重原子数。",
            "n_hydrogen": "产物氢原子数(启发式由 amber14 模板加氢; 精确由 propka 指派可质子化态)。",
            "provenance": "口径溯源串: 精确='propka pKa-aware titration states'; 启发式='heuristic amber14 addHydrogens (NOT pKa-aware)'。",
            "note": "补充说明(如 'propka 不可用, 自动退回固定 pH 启发式(非精确)' 或 精确路径异常退回的异常文本)。",
            "error": "失败时原因(status='unavailable'/'error' 时存在)。",
        },
    },

    # ------------------------------------------------------------------
    # multiconf: multiconf_summary.json (多构象起始对接 A3, 2026-10-07)
    # ------------------------------------------------------------------
    "multiconf_summary": {
        "title": "多构象起始对接汇总 (multiconf_summary.json)",
        "key_metrics": ["ok", "n_conformers", "n_ok", "n_failed",
                        "energy_min", "energy_mean", "energy_std", "best"],
        "fields": {
            "ok": "整体是否成功(任一构象对接成功即 True; 全失败为 False)。",
            "multiconf_id": "运行 id(来自 --id; 各构象产物目录 conf<N>/)。",
            "receptor": "受体文件名(仅文件名, 不泄露路径)。",
            "ligand_file": "多构象配体文件名(仅文件名)。",
            "n_conformers": "拆出的构象总数(= conformers[] 长度)。",
            "n_ok": "对接成功的构象数(ok=True 且有能量)。",
            "n_failed": "失败的构象数(ok=False 或无能量)。",
            "params": "本次对接参数回显(便于复现): exhaustiveness / num_modes / seed / ligand_ph / engine / chain / box_size / n_conformers_split。",
            "conformers": "逐构象结果数组(conformer_index 升序), 字段见 conformers[].*。",
            "conformers[].conformer_index": "该构象的 0-based 索引(与 conf<N+1> 对应)。",
            "conformers[].ok": "该构象完整 dock 流水线是否成功(ok=True 且有能量)。",
            "conformers[].energy_kcal_mol": "该构象最低结合能(kcal/mol; Vina 打分)。null=无能量。",
            "conformers[].reliability": "可靠性标记(ok/warn/unreliable; 来自 run_pipeline)。",
            "conformers[].n_poses": "输出 pose 数。",
            "conformers[].nuc_to_ligO_A": "亲核残基原子 → 配体 O 距离(Å; 催化机制关键几何, Ser OG→酯羰基 O)。null=无参考残基/未命中。",
            "conformers[].ref_to_ligAromC_A": "参考残基芳香 C → 配体最近芳香 C 距离(Å; 芳香锚定几何)。null=无参考残基/无芳香。",
            "conformers[].pocket_contact_frac": "口袋接触分数(0~1)。",
            "conformers[].ligand_buried": "配体被埋分数(0~1)。",
            "conformers[].energy_reliable": "该构象能量是否可信(run_pipeline 判定的可靠性, 非 warn/unreliable)。",
            "conformers[].selfdock_available": "是否 self-docking 模式(共晶配体重对接, 才有 RMSD 物理意义)。",
            "conformers[].selfdock_ok": "self-docking 姿态是否达标(RMSD 在阈值内)。",
            "conformers[].box_assumption": "盒子假设描述串(如 'auto→配体盒' / 'manual')。",
            "conformers[].top_pose_rmsd_A": "top1 pose 相对参考的 RMSD(Å; 仅 selfdock 有意义)。",
            "conformers[].pose_set_rmsd_mean_A": "pose 集合平均 RMSD(Å; 反映 pose 簇紧致度)。",
            "conformers[].notes": "该构象的附注列表(run_pipeline 落盘的判语/警告)。",
            "best": "best 构象摘要(取所有 ok+energy_reliable 中能量最低者; 若全无则放宽到 ok 有能量最低者)。",
            "best.conformer_index": "best 构象的 0-based 索引。",
            "best.energy_kcal_mol": "best 构象能量(kcal/mol)。",
            "best.nuc_to_ligO_A": "best 构象亲核距离(Å; 催化几何判读优先用 best)。",
            "energy_min": "所有构象能量最小值(kcal/mol; 与 best.energy_kcal_mol 一致, 当 best 来自能量可靠集)。",
            "energy_max": "所有构象能量最大值(kcal/mol; 散布上限)。",
            "energy_mean": "所有构象能量均值(kcal/mol)。",
            "energy_std": "所有构象能量总体标准差(kcal/mol; **核心判据** —— 大=结论对起始构象敏感, 小=收敛稳健)。null=构象数<2。",
            "note": "口径说明: 每个构象独立跑完整 dock 流水线, 取最低能量构象为 best; 仅多起点采样, 非构象穷举。结论是否依赖起始构象由 energy_std(及 conformers[].energy_kcal_mol 散布)反映。",
            "error": "失败时原因串(如拆构象失败 / 全部构象对接失败); 仅 ok=False 时存在。",
        },
    },

    # ------------------------------------------------------------------
    # mmgbsa: mmgbsa_summary.json (§2.3 非 MD 路线升级补全, 2026-10-07)
    # 单结构 MM-GBSA 结合能估算(单点 OBC/GBneck2 + UFF 元素型 LJ)
    # ------------------------------------------------------------------
    "mmgbsa_summary": {
        "title": "单结构 MM-GBSA 结合能估算 (mmgbsa_summary.json)",
        "key_metrics": ["available", "deltaG_bind_kcal", "deltaG_bind_kJ",
                        "ligand_n_atoms"],
        "fields": {
            "ok": "本份分析是否成功走完全流程(受体/配体/三体系单点能量均算出)。False 时看 error。",
            "available": "MM-GBSA 估算是否可用(同 ok; 缺省口径下恒与 ok 同值, 保留以兼容通用报告渲染)。",
            "receptor_pdb": "实际使用的受体 PDB 路径(.pdbqt 输入时自动转 PDB 后路径)。",
            "ligand_pose_pdb": "配体位姿 PDB 路径(只含配体原子的单构象 PDB)。",
            "ligand_smiles": "配体 SMILES(可选溯源记录, 不影响计算)。",
            "seed": "随机种子(该估算为确定性单点, 仅作复现标注)。",
            "solvent": "隐式溶剂标识(OBC/GBneck2 或 vacuum 仅对照)。",
            "ligand_n_atoms": "补氢后配体重原子+氢原子数(能量口径基数)。",
            "ligand_charge_source": "配体电荷来源(供电荷 PDBQT 路径, 或 '(none; 配体电荷全 0)' 表示未给 → 配体电荷全记 0 的已知近似)。",
            "model": "能量模型描述(单点 OBC/GBneck2 + LJ, 不松弛/无熵项, 仅供 re-ranking)。",
            "ligand_model": "配体力场口径(UFF 元素型 LJ + PDBQT 电荷, 受体 amber14 ff14SB, 氢电荷取 0)。",
            "obc_params": "OBC/GBneck2 装配参数(半径换算/屏幕因子), 仅 obc 溶剂非空。",
            "E_complex_kJ": "复合物体系单点势能(kJ/mol; 受体+配体共系)。",
            "E_receptor_kJ": "受体体系单点势能(kJ/mol)。",
            "E_ligand_kJ": "配体体系单点势能(kJ/mol; 空受体+配体, 仅 Nonbonded+OBC)。",
            "E_complex_kcal": "E_complex_kJ 折算 kcal/mol(÷4.184)。",
            "E_receptor_kcal": "E_receptor_kJ 折算 kcal/mol。",
            "E_ligand_kcal": "E_ligand_kJ 折算 kcal/mol。",
            "deltaG_bind_kJ": "单点结合能估算 ΔG_bind = E_complex − E_receptor − E_ligand(kJ/mol)。",
            "deltaG_bind_kcal": "ΔG_bind 折算 kcal/mol(÷4.184)。**单点重评分, 非自由能预测值**, 仅供同口径相对比较。",
            "note": "口径说明: ΔG_bind 为单点 OBC/GBneck2 + UFF 元素型 LJ 近似(不松弛、无构象平均、无 -TΔS 熵项, 配体氢电荷取 0); 仅供同力场/同溶剂口径下的相对比较(re-ranking), 不当绝对结合自由能。",
            "verdict": "自否证判词(与 evidence.VerdictBuilder 同构), 字段见 verdict.*。",
            "verdict.ok": "判词是否可用(恒 True)。",
            "verdict.applicable": "方法是否适用(恒 True: 作为单点重评分工具适用)。",
            "verdict.model": "能量模型标识(同 model)。",
            "verdict.banned_phrases": "禁写表述(如把单点重评分当实验 ΔG 自由能写、据 ΔG_bind 推断活性)。",
            "verdict.allowed_phrases": "可写表述(单点重评分仅供同体系内 pose 间重排序与受体间相对比较等)。",
            "verdict.single_point_warning": "单点值敏感性警告(未最小化 → 坐标冲突/氢取向会搬动能量, 只能同口径相对比较)。",
            "verdict.narrative": "一句话口径叙事(含 ΔG_bind 数值与『重排序量非自由能预测值』定性)。",
            "workdir": "计算临时目录(含补氢后配体 PDB 等中间产物, 便于审计)。",
            "error": "失败时原因串(受体/配体缺失、补氢失败、体系装配失败等); 仅 ok=False 时存在。",
        },
    },

    # ------------------------------------------------------------------
    # mutation_comprehensive_report: mutation-report 聚合产物(#282, 2026-10-07)
    # 把 compare 几何四件套之外的证据线(构象熵差 / 芳香滞留 / 进化保守性 / 取向 /
    # 结构有效性)聚合成一份综合报告, 每条证据原样引用其来源子报告的 verdict。
    # ------------------------------------------------------------------
    "mutation_comprehensive_report": {
        "title": "突变综合报告聚合 (mutation_comprehensive_report.json)",
        "key_metrics": ["ok", "evidence_categories", "verdict.narrative"],
        "fields": {
            "ok": "整份报告是否可用(几何四件套成功 + 各可用证据线 verdict.ok 均 True)。",
            "available": "同 ok(保留兼容渲染)。",
            "wt": "WT/对照受体路径(应喂两侧均松弛的对称对照组合)。",
            "mut": "突变体受体路径。",
            "symmetry": "对称对照标注(如 F205F 序列不变回突), 仅溯源用。",
            "generated_at": "生成时间戳(ISO)。",
            "evidence_categories": "实际参与聚合的证据线 key 列表(子集 of structure_validity/pocket_tunnel/pocket_decomp/product_egress/deacylation/conformation_entropy/retention/orientation/msa)。",
            "evidence": "各证据线条目集合(键=证据线名); 每条含 available/source_doc/verdict, 部分含 detail(来源子报告的关键字段)。",
            "evidence.<line>.available": "该证据线是否成功产出(几何四件套从 compare_summary 读回; 其余按需计算/读取)。",
            "evidence.<line>.source_doc": "溯源 doc_type(如 pocket_tunnel / retention_summary / msa_site_report / mutate / orientation / foldx_ddg_summary)。",
            "evidence.<line>.detail": "来源子报告关键字段透传(几何四件套为 compare_summary 子 dict; conformation_entropy 含 delta_S_J_per_mol_K; retention 含 summary; msa 含 sites; orientation 为 verdict dict; structure_validity 含 qc 摘要; foldx_ddg 含 ddG_kcal 与能量组分)。",
            "evidence.<line>.verdict": "该证据线的自否证判词(原样引用来源子报告, 综合报告不另造措辞); 含 ok/narrative/banned_phrases/allowed_phrases。",
            "evidence.foldx_ddg": "FoldX 5.1 ΔΔG 稳定性证据线(A3 接入, 来自 foldx_ddg_summary; 经 --foldx_json / --foldx_mut 接入或现算)。仅稳定性代理, 非催化活性。",
            "evidence.foldx_ddg.detail.mutation": "本次检索用的突变标签(来自 --foldx_mut 或 mut 路径推导)。",
            "evidence.foldx_ddg.detail.matched": "实际在 foldx 结果中匹配到的 mutation 条目标签。",
            "evidence.foldx_ddg.detail.ddG_kcal": "该突变 ΔΔG(kcal/mol); 取自 foldx 结果中 mutation 匹配条目。ΔΔG>0=不稳定, <0=稳定; 与活性无直接因果。",
            "evidence.foldx_ddg.detail.wt_energy": "FoldX 内部为突变构建的 WT 参照总能量(溯源)。",
            "evidence.foldx_ddg.detail.mut_energy": "突变体总能量(溯源)。",
            "evidence.foldx_ddg.detail.repair_energy": "RepairPDB 总能量(诊断)。",
            "evidence.foldx_ddg.detail.n_mutations_in_summary": "所提供 foldx 结果包含的突变条目总数(审计用)。",
            "evidence.foldx_ddg.verdict": "自否证判词: ΔΔG 仅稳定性代理, 不得据其直接推断活性升降; 禁写『ΔΔG<0=活性提升』。",
            "verdict": "顶层聚合判词(证据线 verdict 的 banned/allowed 并集 + 保守元叙事)。",
            "verdict.ok": "聚合是否可用(同 ok)。",
            "verdict.applicable": "恒定 True(作为证据聚合工具适用)。",
            "verdict.banned_phrases": "所有证据线禁写表述的并集(如把单点 Δ 当自由能、结构 QC 通过=活性提升、ΔG 越负活性越高)。",
            "verdict.allowed_phrases": "所有证据线可写表述的并集(各线均为相对/重排序量, 仅同口径可比)。",
            "verdict.narrative": "保守元叙事: 罗列参与证据线 + 统一免责(各线均为相对量, 非绝对自由能; 机制结论须湿实验 kcat/KM 与酰基酶捕获质谱收口, 不得由单点 Δ 推活性升降)。",
            "note": "报告是证据聚合层, 不重算任何量; 数值/判词均引自来源子报告。几何证据须两侧均松弛对称组合, 不得把未松弛 WT 参照当正式结论。",
            "meta": "可选附加元信息(由编排层写入)。",
        },
    },

    # ------------------------------------------------------------------
    # foldx_ddg_summary: foldx-ddg 子命令产物(#287, 2026-10-08)
    # 调用本地 FoldX 5.1(RepairPDB+BuildModel)算单点突变 ΔΔG, 替代 ddg.py 的
    # 禁用 proxy。二进制由环境变量 FOLDX_BIN 或 --bin 指定。
    # ------------------------------------------------------------------
    "foldx_ddg_summary": {
        "title": "FoldX 5.1 真 ΔΔG 稳定性预测 (foldx_ddg_summary.json)",
        "key_metrics": ["ok", "mutations[].ddG_kcal", "mutations[].verdict.narrative"],
        "fields": {
            "command": "固定为 'foldx-ddg'。",
            "pdb": "WT 结构 PDB 路径(计算入口)。",
            "chain": "突变所在链(默认 A)。",
            "ph": "FoldX 计算 pH。",
            "temperature": "FoldX 计算温度 K。",
            "ok": "全部突变是否均成功算出 ΔΔG(任一失败则为 False)。",
            "mutations": "逐突变结果列表(与 CLI --mut 顺序一致)。",
            "mutations[].mutation": "本次突变字符串(归一化后, 如 FA205G)。",
            "mutations[].ok": "该突变是否成功算出 ΔΔG。",
            "mutations[].available": "能力是否可用(二进制存在且运行过; 即使建模失败也为 True)。",
            "mutations[].ddG_kcal": "ΔΔG = E_mut(BuildModel) − E_wt(BuildModel), kcal/mol。repair 失败的 AF 模型下为 None。",
            "mutations[].ddG_std": "多 run 标准差(当前单 run 为 0.0)。",
            "mutations[].method": "固定 'FoldX5.1 BuildModel'。",
            "mutations[].wt_energy": "FoldX 内部为突变构建的 WT 参照总能量(来自 Raw fxout)。",
            "mutations[].mut_energy": "突变体总能量(来自 Raw fxout)。",
            "mutations[].repair_energy": "RepairPDB 总能量(诊断用)。",
            "mutations[].runs": "运行次数(当前 1)。",
            "mutations[].note": "诚实口径: 单 run ΔΔG 须先标定再写入论文; 单点 ΔΔG 不得直接推活性升降。",
            "mutations[].verdict": "该突变自否证判词(ok/narrative/banned_phrases/allowed_phrases)。",
            "mutations[].verdict.ok": "同 mutations[].ok。",
            "mutations[].verdict.narrative": "保守叙事: ΔΔG 数值 + '稳定性代理, 须实验标定, 不得直接推活性'。",
            "mutations[].verdict.banned_phrases": "禁写: 活性提升/增强/增高/上升、'ΔΔG = 精确'、'自由能严格'。",
            "mutations[].verdict.allowed_phrases": "可写: 稳定性代理、ΔΔG 近似、需实验标定。",
            "note": "FoldX ΔΔG 是真实热力学代理(含 unfolded 态参考 + 系综思想), 远优于 ddg.py 噪声地板 proxy; 但须先做 1~2 个有实验 ΔΔG 的突变体标定再写入论文。AF 预测模型现已跑通(BuildModel 突变文件须命名 individual_list.txt 且每行以 ';' 结尾); ΔΔG 取自 Dif fxout, WT/MUT 绝对能量取自 Raw fxout(FoldX 内部构建 WT 参照)。",
        },
    },

    # ------------------------------------------------------------------
    # dca_pairs: dca 子命令产物(2026-10-08, B类/C类纯计算)
    # 互信息 MI + 平均乘积校正 APC 的残基对耦合排序(弱/间接共进化信号, 非 plmDCA)。
    # ------------------------------------------------------------------
    "dca_pairs": {
        "title": "DCA 共进化耦合排序 (dca_pairs.json)",
        "key_metrics": ["ok", "n_sequences", "n_positions", "pairs[].apc_mi"],
        "fields": {
            "ok": "是否成功(序列≥2 条)。",
            "n_sequences": "比对序列数(含 query)。",
            "n_positions": "比对 match 列数(=1-based 残基号上限)。",
            "pseudocount": "Laplace 平滑强度。",
            "min_separation": "近邻过滤阈值(|i-j|<此值不参与排序)。",
            "apc": "是否使用 APC 校正(默认 True)。",
            "method": "方法描述(含『非 plmDCA, 仅弱/间接信号』免责)。",
            "pairs": "按耦合强度降序的残基对列表(取 top_n)。",
            "pairs[].i": "残基 i(1-based)。",
            "pairs[].j": "残基 j(1-based)。",
            "pairs[].mi": "原始互信息(nats)。",
            "pairs[].apc_mi": "APC 校正后的互信息(排序依据)。",
            "pairs[].gap_i": "残基 i 的 gap 频率。",
            "pairs[].gap_j": "残基 j 的 gap 频率。",
            "verdict": "自否证判词: 强调非 plmDCA、仅理性设计线索、不得写进论文结论。",
        },
    },

    # ------------------------------------------------------------------
    # consensus_constraints: consensus 子命令产物(2026-10-08, B类/C类纯计算)
    # 把 msa.site_report 的进化保守性转成位点允许的/禁忌的残基约束。
    # ------------------------------------------------------------------
    "consensus_constraints": {
        "title": "序列共识约束 (consensus_constraints.json)",
        "key_metrics": ["ok", "n_sites", "verdict.narrative"],
        "fields": {
            "ok": "是否成功(位点非空)。",
            "n_sites": "参与约束的位点数。",
            "conserved_thr": "保守性判定阈值(交叉校验用)。",
            "allowed_freq": "天然频率≥此值视为 allowed 的阈值。",
            "sites": "逐位点约束(键=残基号)。",
            "sites.<site>.is_conserved": "是否高度保守。",
            "sites.<site>.consensus": "共识残基(保守位点时存在)。",
            "sites.<site>.allowed": "设计可行空间(家族天然频率≥阈值的残基)。",
            "sites.<site>.disfavored": "规避空间(家族从未出现过的残基)。",
            "sites.<site>.small_residue_frac": "天然小残基占比。",
            "sites.<site>.n_effective": "有效序列数。",
            "sites.<site>.hard": "是否硬约束(有效序列充足才 True)。",
            "sites.<site>.rationale": "该位点约束生成理由。",
            "verdict": "自否证判词: 共识约束是进化保守性线索, 非功能可行性充分判据, 不得写『只能突成 X』。",
        },
    },

    # ------------------------------------------------------------------
    # epistasis_summary: epistasis 子命令产物(2026-10-08, B类/C类纯计算)
    # 双突变上位性统计(消费单点/双点 ΔΔG)。
    # ------------------------------------------------------------------
    "epistasis_summary": {
        "title": "双突变上位性统计 (epistasis_summary.json)",
        "key_metrics": ["ok", "mode", "threshold_kcal", "summary.n_synergistic"],
        "fields": {
            "ok": "是否成功(至少可分类一个双突变)。",
            "mode": "加性/乘性模型。",
            "threshold_kcal": "近加性判定阈值(kcal/mol)。",
            "n_singles": "单点突变数。",
            "items": "逐双突变异构条目。",
            "items[].mutation": "双突变字符串(如 F205G/Y209A)。",
            "items[].components": "拆分出的单点组件标签。",
            "items[].observed": "观测 ΔΔG(双突变)。",
            "items[].expected": "加性(或乘性)预期 ΔΔG。",
            "items[].epistasis": "ε = 观测 − 预期。",
            "items[].classification": "synergistic/antagonistic/neutral/indeterminate。",
            "items[].z": "显著性 z 值(有误差时; 否则 None)。",
            "items[].significant": "是否通过误差显著性检验(None=未知)。",
            "summary": "汇总计数(可分类/协同/拮抗/近加性/显著)。",
            "verdict": "自否证判词: 上位性需 ΔΔG 误差远小于 |ε| 才成立, 不得由 FoldX 单 run 直接宣称。",
        },
    },

    # ------------------------------------------------------------------
    # scan_ranking: mutate-scan/rank 子命令产物(2026-10-08, B类)
    # 多目标突变候选排序结果(优先级启发式, 非活性预测)。
    # ------------------------------------------------------------------
    "scan_ranking": {
        "title": "突变扫描多目标排序 (scan_ranking.json)",
        "key_metrics": ["ok", "method", "n_active", "n_excluded", "top_id"],
        "fields": {
            "ok": "是否成功(至少 1 个通过硬门的候选)。",
            "method": "排序方法(weighted/pareto/both)。",
            "n_active": "通过硬门参与排序的候选数。",
            "n_excluded": "被硬门剔除的候选数。",
            "n_objectives": "目标维度数。",
            "objectives": "目标定义(name/key/direction/weight)。",
            "objectives_stats": "逐目标 min/max/present/direction(透明度)。",
            "ranked": "排序后候选列表(rank 升序)。",
            "ranked[].rank": "综合排名(1=首位)。",
            "ranked[].id": "候选突变标签。",
            "ranked[].score": "加权综合分(0~1, 越大越优)。",
            "ranked[].pareto_rank": "帕累托前沿层级(0=前沿)。",
            "ranked[].components": "各目标归一化分值。",
            "ranked[].metrics": "各目标原始数值。",
            "ranked[].incomplete": "是否含缺失目标(中性折算)。",
            "excluded": "被硬门剔除的候选(id/reason)。",
            "top_id": "排首位候选标签(仅优先级建议, 非活性最高)。",
            "verdict": "自否证判词: 排序仅为实验优先级建议, 结论须湿实验 kcat/KM 收口。",
        },
    },

    # ------------------------------------------------------------------
    # evidence: evidence_registry (证据门槛注册表, 纯文档型, 无独立 JSON 产物)
    # 2026-10-07 起阈值单一事实源迁至 libdock.evidence.THRESHOLD_SPECS /
    # THRESHOLDS, 本 doc_type 仅作人工查阅与审计发现锚点。
    # ------------------------------------------------------------------
    "evidence_registry": {
        "title": "证据门槛注册表 (evidence.THRESHOLDS)",
        "key_metrics": ["n_thresholds", "categories"],
        "fields": {
            "n_thresholds": "注册表收录的命名门槛总数(2026-10-07 起集中维护于 libdock.evidence.THRESHOLD_SPECS, 单一事实源; 各业务模块不再各自持有同名常量, 而是从 THRESHOLDS 读取)。",
            "categories": "门槛分 7 类: reliability(对接可靠性) / orientation(催化取向) / retention(芳香产物滞留) / deacylation(去酰化水) / conformation_entropy(局部构象熵) / pocket_tunnel(口袋通道) / multiconf(多构象散布)。",
            "env_override": "覆盖优先级: DOCK_EVID_<NAME大写>(统一新名) > 各模块历史 env 别名(见 THRESHOLD_SPECS 的 aliases, 如 DOCK_ORIENT_NOISE_FRAC) > 默认值。非数字 env 回退默认并打 WARNING。",
            "thresholds.contact_unreliable": "reliability: 口袋接触分数<此值=漂浮/偏靶(unreliable)。",
            "thresholds.contact_warn": "reliability: 接触分数<此值=弱接触(warn)。",
            "thresholds.buried_clash": "reliability: 对撞数≥此且埋藏占比≥buried_frac=配体埋入蛋白内部。",
            "thresholds.buried_frac": "reliability: 配体表面被完全包埋占比阈值(与 buried_clash 联用)。",
            "thresholds.energy_abs_max": "reliability: |对接能量|>此值=能量失真/不可信。",
            "thresholds.heavy_max": "reliability: 配体重原子数>此值=超出 Vina 稳健尺度。",
            "thresholds.top_rmsd_warn": "reliability: top1~top2 pose RMSD>此值=对接未收敛(warn)。",
            "thresholds.pose_scatter_warn": "reliability: 全 pose 两两自散 median RMSD>此值=结合模式未收敛(soft)。",
            "thresholds.nuc_offsite": "reliability: 亲核原子→配体氧距离>此值=配体不在催化位点。",
            "thresholds.orient_nuc_max": "orientation: 亲核原子→scissile 羰基碳接触阈值(Å), 成酰基酶几何前置。",
            "thresholds.orient_his_o_max": "orientation: His NE2→离去氧接触阈值(Å), 质子 relay 闭合。",
            "thresholds.orient_offsite_A": "orientation: oriented=0 且最近 scissile C→亲核均值超此值=对接没摆进催化中心。",
            "thresholds.orient_noise_frac": "orientation: 取向占比差 |Δfrac| 噪声地板(比例量)。",
            "thresholds.orient_dist_noise_A": "orientation: 摆位距离差 |Δ距离| 噪声地板(Å, 与占比差各自独立判定)。",
            "thresholds.retention_stack_A": "retention: 芳香产物碳→受体芳香碳≤此值=仍卡在芳香锚定接触壳层内(滞留)。",
            "thresholds.retention_noise_frac": "retention: 芳香滞留占比差 |Δfrac| 噪声地板(比例量)。",
            "thresholds.cross_frame_warn_A": "retention: ref 盒心用于 alt 时, alt 自身三联体距 ref 盒心超此值→能量不可相减。",
            "thresholds.water_clearance_A": "deacylation: 水分子氧到蛋白重原子 vdW 表面余隙下限(Å, 水探针半径)。",
            "thresholds.water_nuc_min_A": "deacylation: 亲核距离窗下限(Å)。",
            "thresholds.water_nuc_max_A": "deacylation: 亲核距离窗上限(Å)。",
            "thresholds.bd_angle_target": "deacylation: Bürgi–Dunitz 攻击角目标(度)。",
            "thresholds.bd_angle_tol": "deacylation: Bürgi–Dunitz 角容忍窗(度)。",
            "thresholds.his_water_max_A": "deacylation: 水分子氧→His NE2 上限(Å)。",
            "thresholds.search_radius_A": "deacylation: 水位点搜索半径(Å)。",
            "thresholds.search_step_A": "deacylation: 水位点搜索网格步长(Å)。",
            "thresholds.deacyl_noise_vol_A3": "deacylation: 去酰化水空间差噪声地板(Å³, ≈一个水分子体积)。",
            "thresholds.conf_entropy_noise_J_per_mol_K": "conformation_entropy: 局部构象熵差 |ΔS| 噪声地板(J/mol·K)。",
            "thresholds.bottleneck_replicate_noise_A": "pocket_tunnel: 喉部坐标距离复现噪声地板(Å)。",
            "thresholds.bottleneck_delta_floor_A": "pocket_tunnel: 瓶颈半径差 |Δ|≥此值才算真换档(Å)。",
            "thresholds.energy_std_floor": "multiconf: 多构象能量 std 解读地板(kcal/mol); ≥此值=结论对起始构象敏感。",
            "helper.classify_delta": "通用: 单一 signed delta 过噪地板判定(支持 direction 限定)。",
            "helper.frac_delta_verdict": "通用: 比例量 |Δfrac| 过噪地板判定(兼容 float 与 {'mean':...})。",
            "helper.dist_delta_verdict": "通用: 距离量 |Δ距离| 过噪地板判定(与占比量独立)。",
            "helper.vol_delta_verdict": "通用: 体积量 |Δ体积| 过噪地板判定。",
            "helper.replicate_sign_consistent": "通用: 多轮独立重复方向一致性(同号+多数过地板)判定(replicate_required 逻辑)。",
            "helper.conformation_sensitive": "通用: multiconf energy_std 解读(对起始构象是否敏感)。",
            "helper.VerdictBuilder": "通用: 自否证判词装配器(ban/allow/narrative/build)。",
        },
    },
}


# ----------------------------------------------------------------------------
# 对称结构别名(程序化拷贝, 避免同一解释手抄两份)
# ----------------------------------------------------------------------------
# 1) mutate 的 qc.pocket_tunnel.* = pocket-tunnel 的 verdict.*(同一 compare_sides
#    判词, mutate QC 只引用不另编); 2) deacylation 的 mut.* = wt.*(两侧结构完全
#    对称); 3) pocket_tunnel 的 tunnel.mut.* = tunnel.wt.*(同上)。
def _alias(src_type: str, src_prefix: str, dst_type: str, dst_prefix: str) -> None:
    src = FIELD_DOCS[src_type]["fields"]
    dst = FIELD_DOCS[dst_type]["fields"]
    for k, v in list(src.items()):
        if k.startswith(src_prefix):
            sub = k[len(src_prefix):]
            dst.setdefault(dst_prefix + sub, v)


_alias("pocket_tunnel", "verdict.", "mutate", "qc.pocket_tunnel.")
_alias("deacylation", "wt.", "deacylation", "mut.")
_alias("pocket_tunnel", "tunnel.wt.", "pocket_tunnel", "tunnel.mut.")
# 4) pocket_decomp 的 mutated_rows / released_top / backfill_top 与 per_residue
#    同构(逐残基贡献行), 同一份解释别名到四组键。
for _arr in ("mutated_rows", "released_top", "backfill_top"):
    _alias("pocket_decomp", "per_residue[i].", "pocket_decomp", f"{_arr}[i].")


# ----------------------------------------------------------------------------
# 扁平化与覆盖统计
# ----------------------------------------------------------------------------
def _norm_index(seg: str) -> str:
    """数组下标段归一: `poses[0]` / `poses[12]` → `poses[i]`(模板)。"""
    if seg.endswith("]") and "[" in seg:
        return seg[:seg.index("[")] + "[i]"
    return seg


def iter_flat_fields(payload: Any, prefix: str = "",
                     dynamic: Any = ()) -> List[Tuple[str, Any]]:
    """把 JSON payload 扁平化为 (路径, 展示值) 列表。

    约定: dict 数组只展开首元素的字段(路径记 `key[i].sub`), 同时给出
    `(key[i], '共 N 项')` 的概览行; 标量数组整体作为叶子; dict 键原样
    点连接。dynamic = 动态键容器路径集(通配记法): 递归到该容器时不展开
    键, 整体记 `path.*` 一个叶子 —— 键名本身是数据(受体路径/列号/桶标签/
    单字码)的容器必须声明, 否则键里的点号/盘符会破坏路径匹配。
    """
    out: List[Tuple[str, Any]] = []
    if isinstance(payload, dict):
        # 动态容器: 通配匹配(与 _coverage_candidates 同一套候选)
        if prefix and any(c in set(dynamic) for c in _coverage_candidates(prefix)):
            out.append((prefix + ".*", payload))
            return out
        for k, v in payload.items():
            out.extend(iter_flat_fields(
                v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}", dynamic))
    elif isinstance(payload, list):
        # 概览行
        if payload and isinstance(payload[0], dict):
            out.append((f"{prefix}[i]", f"共 {len(payload)} 项"))
            out.extend(iter_flat_fields(payload[0], f"{prefix}[i]", dynamic))
        else:
            out.append((prefix, payload))
    else:
        out.append((prefix, payload))
    return out


def _coverage_candidates(path: str) -> List[str]:
    """path 的文档匹配候选: 原样 → 单段通配 → 双段通配。

    `tunnel.wt.seed_A` → ["tunnel.wt.seed_A", "*.wt.seed_A", "tunnel.*.seed_A", ...]
    `sites.55.aa_freqs.G`(双层动态键) → 含 "sites.*.aa_freqs.*" 等双通配候选。
    数组段(`poses[0]`)先归一为 `poses[i]`。
    """
    from itertools import combinations
    segs = [_norm_index(s) for s in path.split(".")]
    cands = [".".join(segs)]
    # 单段通配
    for i in range(len(segs)):
        mod = list(segs)
        mod[i] = "*"
        cands.append(".".join(mod))
    # 双段通配(msa 的 sites.<列>.aa_freqs.<单字码> 这类两层动态键)
    for i, j in combinations(range(len(segs)), 2):
        mod = list(segs)
        mod[i] = "*"
        mod[j] = "*"
        cands.append(".".join(mod))
    return cands


def coverage(doc_type: str, payload: dict) -> Dict[str, Any]:
    """字段覆盖率审计: 本 payload 的字段有多少已在 FIELD_DOCS 登记。

    返回 {"n_total", "n_documented", "undocumented": [路径...]}。
    未登记字段会在 report.md 脚注顶出 —— 新代码加字段忘登记时可见。
    """
    spec = FIELD_DOCS.get(doc_type) or {}
    known = spec.get("fields", {})
    pairs = iter_flat_fields(payload, dynamic=tuple(spec.get("dynamic", ())))
    # 概览行(`key[i]` = "共 N 项")不是叶子字段, 不计入
    paths = [p for p, v in pairs
             if not (p.endswith("[i]") and isinstance(v, str) and v.startswith("共 "))]
    undocumented: List[str] = []
    n_doc = 0
    for p in paths:
        hit = any(c in known for c in _coverage_candidates(p))
        if hit:
            n_doc += 1
        else:
            # 纯容器键(如 "qc")不算未登记(其子字段已逐一登记)
            is_container = any(q.startswith(p + ".") or q.startswith(p + "[i]")
                                for q in paths)
            if not is_container:
                undocumented.append(p)
    return {"n_total": len(paths), "n_documented": n_doc,
            "undocumented": undocumented}


def get_spec(doc_type: str) -> Optional[Dict[str, Any]]:
    """取某文档类型的完整 spec(title/fields/key_metrics); 未知类型返回 None。"""
    return FIELD_DOCS.get(doc_type)


__all__ = ["FIELD_DOCS", "coverage", "get_spec", "iter_flat_fields"]
