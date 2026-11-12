from pathlib import Path
import json, sys, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from models import CASES,EQUATIONS,MENUS,INITIAL,RANGES,T,TIMES

CN={'reaction':'一阶/二阶反应','chemostat':'共反应物控制','reversible':'可逆反应','enzyme':'线性/饱和动力学','oscillator':'非线性振幅动力学','duffing':'Duffing 恢复力','forced_damping':'受迫非线性阻尼','predator':'双组分捕食耦合','diffusion_pde':'二阶/四阶扩散 PDE','advection_diffusion':'平流–扩散 PDE','allen_cahn':'非线性反应–扩散 PDE'}
CONTROL={'reaction':'初始浓度','chemostat':'共反应物浓度，单次内固定','reversible':'初始组分比例，总量为1','enzyme':'初始浓度；已知饱和常数K=.4','oscillator':'初始半径','duffing':'初始振幅','forced_damping':'正弦驱动频率，驱动幅度/时长固定','predator':'两个初始浓度/群体量','diffusion_pde':'初始空间波数，能量固定','advection_diffusion':'初始空间波数，能量固定','allen_cahn':'初始场幅值与波数'}

def table(header,rows):
    return '| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)

r=pd.read_csv(ROOT/'case_results.csv');p=r.pivot(index='case',columns='policy',values='accuracy')
over=pd.read_csv(ROOT/'overall.csv');stop=pd.read_csv(ROOT/'overall_stopping.csv');ci=pd.read_csv(ROOT/'paired_comparisons.csv')
raw=pd.read_csv(ROOT/'replication'/'trials.csv');last=raw[raw['round']==3]
three=pd.read_csv(ROOT/'stress'/'reaction3'/'trials.csv');tf=three[three['round']==3]
three_summary=tf.groupby(['sigma','policy']).agg(n=('correct','size'),accuracy=('correct','mean'),risk=('risk','mean'),forecast_rmse=('forecast_rmse','median')).reset_index();three_summary.to_csv(ROOT/'three_term_results.csv',index=False)
stress=pd.read_csv(ROOT/'stress'/'robustness'/'trials.csv');sr=[]
for (scenario,policy),d in stress.groupby(['scenario','policy']):
    f=d[d['round']==3];s=d[d.stopped_here==1]
    sr.append(dict(scenario=scenario,policy=policy,n=len(f),full_accuracy=f.correct.mean(),mean_runs=s['round'].mean(),confident_fraction=s.confident.mean(),accuracy_if_confident=s.loc[s.confident==1,'correct'].mean(),unresolved=1-s.confident.mean()))
sr=pd.DataFrame(sr);sr.to_csv(ROOT/'robustness_results.csv',index=False)
quad=pd.read_csv(ROOT/'audits'/'quadrature.csv');q3=pd.read_csv(ROOT/'audits'/'reaction3_quadrature.csv');mc=pd.read_csv(ROOT/'audits'/'predictive_mc.csv');sol=pd.read_csv(ROOT/'audits'/'solver.csv')
comp=ci[(ci.case=='ALL')&(ci.baseline=='D_OPT')].iloc[0]
noise=pd.read_csv(ROOT/'noise_model_comparison.csv')
protocol=json.loads((ROOT/'replication'/'protocol.json').read_text())
headline=table(['case','改变什么','初始准确率','A D-optimal','B 结构风险','先验优化固定'],[
    [CN[c],CONTROL[c],f"{100*raw[(raw.case==c)&(raw.policy=='D_OPT')&(raw['round']==0)].correct.mean():.2f}%",f'{100*p.loc[c,"D_OPT"]:.2f}%',f'{100*p.loc[c,"STRUCT_RISK"]:.2f}%',f'{100*p.loc[c,"FIXED"]:.2f}%'] for c in CASES])
stop_table=table(['方法','平均追加次数','提前停止强制MAP正确率','作出单一结构声明比例','这些声明中的正确率'],[
    [x.policy,f'{x.mean_runs:.3f}',f'{100*x.forced_map_accuracy:.2f}%',f'{100*x.confident_fraction:.2f}%',f'{100*x.confident_accuracy:.2f}%'] for _,x in stop.iterrows()])
three_p=three_summary.pivot(index='sigma',columns='policy',values='accuracy')
three_table=table(['噪声SD','随机','固定多样化','A','B'],[[s]+[f'{100*three_p.loc[s,k]:.2f}%' for k in ['RANDOM','FIXED','D_OPT','STRUCT_RISK']] for s in three_p.index])
noise_table=table(['误差模型','方法','完整预算结构正确率','单一声明比例','声明中的正确率'],[[x.inference,x.policy,f'{100*x.full_accuracy:.2f}%',f'{100*x.confident_fraction:.2f}%',f'{100*x.accuracy_if_confident:.2f}%'] for _,x in noise.iterrows()])
text=f'''# v21：11 类动力学与三机制扩展，两个选择器公平对照

## 核心结论

两种方案都已实跑。A 是完整候选项观测灵敏度上的局部 D-optimal；B 是贝叶斯结构 MAP 错误风险的预期下降。两者共用一个 Bayesian sparse-library / SINDy-style 推断器、同一 spike-and-slab 先验、同一原始观测似然与停止规则。不是两个互斥的“统计学派”。

主实验 11 类 × 2 个噪声水平 × 每格 192 个独立任务 = **4,224**。另有 **192** 个三机制反应任务和 **672** 个单列压力任务，合计 **5,088** 个评估任务。192个相关噪声任务又用正确协方差做配对复验，**不重复计数**。88个实现检查任务、先验设计模拟和数值审计不计入这个评估总数。

主实验 A = 97.75%，B = 97.92%。B−A={comp.diff_pp:.3f}个百分点，按任务配对 bootstrap 95%区间 [{comp.low_pp:.3f}, {comp.high_pp:.3f}]。**没有确立 B 普遍优于 A。** 11个分case比较均未通过 Holm 校正显著性门槛。

## 1. 不延长时间，不用未来数据选点

所有条件固定 **T=1.5**，(0,T]内每通道16个采样时刻。一次共同初始实验，最多三次新增实验。初态已知，不把t=0当免费带噪观测。每次只能改变初态、配比、驱动频率或空间激励。所有方法在同一case具有同一动作菜单和费用。

主表指标是**所有候选项完整支持集是否正确**。每次新增后共同更新所有历史实验的同一组系数。没有给选择器真实支持、真实参数或未执行实验数据。连续真值不在 Gauss–Legendre 积分节点上抽样。

采集次数、单次时间和采样量匹配；不同初态的制备、材料消耗、能量成本未实测，不能声称实验室美元成本已经匹配或减少。

## 2. 主结果

每行384个独立任务，均匀汇总噪声SD=.03与.10。不同case的状态/噪声是声明的无量纲 toy，不能将百分比当作同等真实应用难度。

{headline}

固定对照不是故意挑差的点：在独立先验模拟上比较35个三实验组合，再选择前缀较好的顺序。使用40阶积分与96个先验世界；不读取复验数据，属于近似优化而非解析全局最优。计划在主实验前固定并保存散列。

主测试每项以概率.4存在，存在时在声明的正系数区间连续抽样；约35.9%任务没有这两个可选项。已知基础机制仍可存在。为避免零支持集抬高成绩，`overall.csv`也给出非空支持集与初始不确定子集：非空2,709个任务中 A=96.53%，B=96.75%；初始不确定3,267任务中 A=97.15%，B=97.37%。仍无普遍优势。

较容易的平流–扩散在初始实验已达95.57%，不能把最后100%解读为巨大主动收益。受迫非线性阻尼最难，最终只有约87%；频率菜单下线性阻尼和二次阻尼仍容易混淆。未为了制造优势移除固定方案或改变候选库。

## 3. 停止：足够确定就结束，未解决不能算成功

共同规则：max_S p(S|D) ≥ .95则停止；否则继续到三次预算上限，仍未达到则标记未解决。**本轮不使用“单步价值低就停止”**，因为上一轮已有互补实验反例。没有实现完整最优停止。

为公平比较同时保存完整三次路径，停止指标只使用第一触发点及之前的数据，不利用未来正确率选择停止点。

{stop_table}

这是给定候选库、先验与噪声模型下的条件后验，不是任意真值下的95%频率学保证。B声明更多任务，A在所声明任务中的正确率稍高；不能只报声明子集准确率而隐去未解决比例。

## 4. 三机制反应库：不再只有两个待选项

`x'=-a*x-b*x^2-c*x^3`，三个系数可分别严格为零，共8种支持集。仍然只改变初始浓度，T=1.5、每轮16个时刻、最多三次。每个噪声96个独立任务。先验非零范围[.05,.7]，主积分48阶/维；固定对照在这一额外实验里是预设低/高/中浓度，不冒称为先验优化固定方案。

{three_table}

总体 A=81.77%，B=82.29%，初始只有41.67%。共同停止下，A平均2.276次、B平均2.255次；只有45.31%/50.52%达到95%结构后验。**复杂度上来后，两种方法都没有“几乎解决”。** 初始/采样预算相同并不保证足以区分相邻的多项式机制。

## 5. 六组压力测试，不与主结果混报

- 限制可干预条件（128任务）：共反应物保持原浓度，PDE只能原空间模态。A/B相同，强制MAP只有75%；73.44%不达到95%后验要求。重复轨迹没有自动制造辨识信息。
- 时间相关噪声（192任务）：实际AR(1)相关ρ=.7，却按独立噪声推断。A/B完整预算准确率79.17%/83.33%，高置信声明中的正确率仅85.82%/87.16%。
- 真实噪声是声明值2倍（96任务）：完整预算A/B82.29%/81.25%。
- 非零机制小于slab下限（128任务）：精确结构恢复严重失败；这是明确的先验错设，不是任意微弱机制可被有限数据辨识的证明。
- 非线性PDE用64格真值、32格推断（64任务）：本组A/B均100%；只是特定步长/幅值设置的空间失配检查。
- 反应方程含库外三次项（64任务）：库内完整结构正确率必为0，仍有87.5%作出高置信库内声明。**贝叶斯概率不能证明候选库充分。**

完整逐项数据见 `robustness_results.csv`，不要把有意构造的库外失败计入正常case平均准确率。

## 6. 配对噪声模型检查：修正似然，而不是再换选择器

复用上述192个相关噪声任务，将真实ρ=.7提供给两个算法，进行相同的AR(1)白化：第一个读数不变，随后以 `(y_t-rho*y_(t-1))/sqrt(1-rho^2)` 计算似然与敏感度。没有增加真实观测。

{noise_table}

已知正确相关性后，两个方案都更谨慎：可作单一结论的比例下降，但声明可靠性明显改善。**这是oracle协方差诊断，不是已实现未知噪声参数学习。** 改进不能归功于B选点，也不能只报条件正确率而不报覆盖比例。

## 7. SINDy到底是什么角色？

从候选物理项组成稀疏字典，用“精确零 + 连续非零”的先验同时辨识支持与参数，前向方程把数据与模型连接。它是 Bayesian SINDy-style sparse-library discovery，**不是原版 PySINDy/STLSQ 或 UQ-SINDy 完整复现**。主实验只两项可选机制，额外实验三项，不能宣传为大型开放符号搜索。

A利用当前MAP结构内的条件均值做局部线性化，但敏感度包含全部候选列，包括当前零系数列；其正则精度取声明先验的二阶矩近似。B在完整结构与连续参数后验中积分新观测，最大化 `R(D)-E R(D+Y)`，其中`R=1-max p(S|D)`。都不是新发明的评分。

主测试使用4种策略：RANDOM、FIXED、D_OPT、STRUCT_RISK。换评分没有同时换推断器、先验或最终判别规则。

## 8. 数值核验

- {len(sol)}组独立DOP853/步长/敏感度核验；最大前向差异 {sol.dop853_max.max():.3g}，最大敏感度相对差异 {sol.jac_rel.max():.3g}。
- Allen–Cahn采用周期32点中心差分MOL、8个观测点；32→64格的最大观测差异 {sol.spatial32_64.max():.3g}，64→128为 {sol.spatial64_128.max():.3g}。这不是任意非线性PDE的网格收敛证明。
- {len(quad)}个固定数据后验120→240阶：最大结构概率变化 {quad.max_p_change.max():.4g}，MAP变更 {int(quad.map_changed.sum())}，.95阈值变更 {int(quad.confidence_changed.sum())}。
- 三机制 {len(q3)}个固定后验48→64阶：最大变化 {q3.max_p_change.max():.4g}，MAP变更 {int(q3.map_changed.sum())}，.95阈值变更 {int(q3.confidence_changed.sum())}。
- {len(mc)}个设计状态，每支持集预测积分32→128→256样本：32→256有 {int(mc.changed32_256.sum())}次动作变化，128→256有 {int(mc.changed128_256.sum())}次。按256样本估计的最大选点遗憾 {mc.regret32.max():.4g}。因此B评分仍是有噪声近似；原始越界值与按理论界限截断均记录，不能声称精确最优。
- `REPLICATION_LOCK.json`、`lock_audit.json`确认主脚本与固定计划在复验期间未改变。

## 9. 示例与可执行产物

`examples/`选例规则预先写入脚本：各指定case按seed排序，取最初错误且不确定、最终正确的第一项，展示首次支持集正确的更新。它们只是说明性案例，不是独立性能统计。旧系数和旧曲线在新条件下保持冻结，横轴全部[0,1.5]。

饱和反应示例：初始 x0=.25 时误拟合 `x'=-.5042*x`；算法选择 x0=2，时长不变，得到 `x'=-.3153*x/(.4+x)`，对应结构后验.999325；模拟真系数.32084只用于评分。没有借助延长来修正。

## 10. 对研究与Lila的意义

这套可复现实验对应“实验数据→显式方程→新实验→恢复/泛化/噪声/物理一致性评估”。**没有Lila内部数据，没有新增真实电化学或材料实验。** 结论是两个选择器都值得保留，而不是B显著优于A；更迫切的问题是三项以上的结构混淆与错误噪声/候选库导致的伪置信。

## 参考

- Hirsh, Barajas-Solano, Kutz (2022), Sparsifying priors for Bayesian uncertainty quantification in model discovery. https://pmc.ncbi.nlm.nih.gov/articles/PMC8864363/
- Hainy et al. (2022), Optimal Bayesian design for model discrimination via classification. https://pmc.ncbi.nlm.nih.gov/articles/PMC8924111/
- Huan & Marzouk (2013), Simulation-based optimal Bayesian experimental design for nonlinear systems. https://arxiv.org/abs/1108.4146
- Seeger (2008), Bayesian Inference and Optimal Design for the Sparse Linear Model. https://www.jmlr.org/papers/v9/seeger08a.html
- Lila Co-Op, AI-Driven Equation Discovery: https://job-boards.greenhouse.io/lilasciences/jobs/4469152009

主机制、贝叶斯结构先验与D-optimal都是已有原理。不能用case数量或版本号替代新颖性和强证据。
'''
(ROOT/'RESULTS_ZH.md').write_text(text,encoding='utf-8')
readme='''# Fixed-duration active sparse equation discovery: two selectors, many systems

This is a synthetic numerical benchmark, not new SOTA, hardware validation, or an official UQ-SINDy reproduction. Read RESULTS_ZH.md for all outcomes and limits.

## Reproduce

Python 3.12/3.13, numpy, scipy, pandas>=2.2, numba, matplotlib. Use one BLAS thread per process. The tensor-quadrature caches are intentionally excluded from the ZIP and regenerate on the first run. Preserve fixed_plans.json: it was trained using independent prior draws, before validation.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python experiment.py --n 192 --order 120 --power 5 --seed 61000000 --workers 4 --out replication
OPENBLAS_NUM_THREADS=1 python experiment.py --cases reaction3 --n 96 --order 48 --power 5 --sigmas .03,.10 --seed 82000000 --workers 1 --out stress/reaction3
OPENBLAS_NUM_THREADS=1 python run_stress.py
OPENBLAS_NUM_THREADS=1 python noise_aware.py
python analyze.py
python analyze_extra.py
python audit_solvers.py
python audit_inference.py
python audit_reaction3.py
python export_examples.py
python make_figures.py
python make_report.py
```

`fixed_plan.py` documents the prior-only fixed-design search; do not rerun it before replaying the stored validation. To rebuild it intentionally, save the original JSON first. For a smoke test use `python experiment.py --cases reaction,enzyme --n 2 --order 80 --power 4 --out smoke`.

## Files and scope

- models.py: 11 two-unknown-mechanism dynamics, plus 3-mechanism reaction.
- experiment.py: one shared spike-and-slab posterior, independent Gaussian observation likelihood, prospective Bayesian structural-risk integration, full-library D-optimality, and identical stopping.
- replication/trials.csv: 4,224 independent tasks; four policies and four rounds are paired rows, not additional independent systems.
- stress/reaction3: 192 additional tasks with eight supports.
- stress/robustness: 672 deliberate stress tasks.
- stress/noise_aware: reuses the same 192 correlated-noise worlds with the true covariance supplied; not another 192 independent tasks.
- audits: timestep, independent solver, spatial, quadrature and predictive-sampling checks.
- examples: archived illustrative observations, all old model coefficients and retained curves.

Raw observations are deterministically reconstructed by `obs` using the recorded case, truth, seed, action, stage, noise and scenario. Examples additionally store raw readings in NPZ. Future observations do not enter the acquisition objective. The final trajectories were simulated to full budget for fair accuracy comparisons; reported stopping uses the first crossing, without future data.

The ordinary library has only two optional terms, known signs, fully observed state or known sufficient Fourier/radial projections, and known reset conditions. The additional library has three optional terms. This is not an arbitrary PDE library or a noise-robust universal discovery algorithm.

The default stopping rule is model-based 95% posterior confidence or budget exhaustion. One-step low-value stopping was intentionally removed because it can overlook complementary experiments. The nonlinear PDE is a 32-site periodic method-of-lines system, with separate grid checks and 64-site truth stress tests.

The fixed-duration design matches acquisition count, duration and sample count, not reagent, reset energy, preparation time or money. Do not describe it as measured lab cost reduction.
'''
(ROOT/'README.md').write_text(readme)
(ROOT/'requirements.txt').write_text('numpy\nscipy\npandas>=2.2\nnumba\nmatplotlib\n')
import scipy,numba,matplotlib
(ROOT/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,numba=numba.__version__,matplotlib=matplotlib.__version__),indent=2))
(ROOT/'case_manifest.json').write_text(json.dumps({c:dict(equation=EQUATIONS[c],ranges=RANGES[c].tolist(),initial=INITIAL[c],actions=MENUS[c]) for c in CASES+['reaction3']},indent=2))
print('Report written',len(text),'characters')
