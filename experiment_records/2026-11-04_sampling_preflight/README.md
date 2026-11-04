# v0 高层选路示范软采样：首轮协议

研究问题：保持 v0 模型、value/low loss、推理方式不变，仅改善 high actor 的首次选路示范，能否改变实际首次出口并减少后续返回岔口的行为？所有模型从头训练，value继续按v0目标训练，不冻结已有checkpoint参数。

## 实验矩阵

16 个正式运行：H2/L1/M1、H2/L1/M3、H4/L1/M3、H4/L2/M3 × 原采样/软采样 × seed0/1。同结构同seed成对排队，物理GPU1上两个训练进程并行。原采样重新运行，已有v0结果仅作外部核对。固定beta3、50步间隔、width32、final-only监督、完整BPTT、1M更新；executor仍采用v0数据command训练及v0原生推理。没有改critic公式、encoder梯度、计划保持时间或引入beta schedule。

## 干预

先完整调用原MultiHGCDataset；原batch及其RNG供low/value原样使用。仅为high损失另建完整真实查询，原forward仍计算executor损失，high-only forward使用相同JAX key且不消耗原sampler RNG。

当起点在岔口核心、50步实际标签已出岔口且尚未返回、goal取自首次返回时或之后，且此前出岔口后未经过goal格时，检查替代示范。必须与原查询同岔口、同来向、同goal格、同时间跨度bin；不同首出口且来自其它episode，真实goal位于donor首次返回之前。排除源episode和同出口后，替代池至少5次不同物理访问、3个原始episode，才以概率0.5替换。没有支持时原样保留，不删原始轨迹。支持门槛用于替代证据这一侧；不是要求原查询所在侧也凑足5/3的事后效应审计。

替换的是完整s0、完整goal及同一donor轨迹实际50/100/150标签，禁止跨轨迹拼接。按独立访问均匀选donor，再均匀选可行真实起点和goal。bin为[0,50)、[50,100)、[100,150)、[150,300)、[300,600)、[600,1200)、[1200,∞)，因此每个batch的条件组、goal格、时间bin及三个跨度的截短模式精确保留。具体连续位置、姿态、速度与bin内时间距离仍可能改变，保存漂移统计，不声称完全相同初始条件。

拓扑仅由训练集XY占据格推断（格宽4、原点(-4,-4)、核心半宽1.6）。不用验证集建索引、不按BFS最短路筛选、不把地图输入网络。“未返回”只是软采样证据，不是路径正确性的标签。所有模块继续共享原v0 encoder，因此high变化仍可间接改变value/low的学习参数；不把单步数据流隔离说成整个训练参数互不影响。

## 启动前证据

训练集500个episode、4661次岔口访问。原采样1,048,576查询中触发19419；完整匹配与支持门槛后8156有替代证据，来自1145次访问、368个episode、全部10个岔口；实际替换4056（全部query的0.3868%）。这是一轮保守、局部干预，不保证全局成功率立刻显著变化。

替换子集中起始XY漂移均值1.48（格宽4），XY速度漂移均值1.92，goal时间距离均值736→701；bin/截短模式保持。原low/value所有采样payload及后续RNG序列与未干预控制一致。完整数据和索引在impls/exp/junction_sampling_preparation_20261104，报告及hash随本目录保存。

冻结v0对照验证：独立进程加载真实H4L2M3模型并进行一次诊断更新（结果丢弃）；527个参数/loss/Adam/RNG/指标数组完全一致。新数据流4测试、sampler11测试、behavior7测试和runner/deferred41测试通过。另独立复核全部71602 donor segments和100个触发查询的实际支持配对。

## 评估与产物

沿用v0每100k评估、每task50个episode、评估温度0；800k/900k/1M保存checkpoint和已有评估的完整轨迹，不添加新policy调用。保存full goal、reset/policy seed、实际首次出口、first-visit与纠错revisit、首次返回、出岔口耗时、返回游走时长、成功前未返回与超时右截断。跨岔口游走时间另算区间并集，避免重复计时；游走时长不是因果意义的可避免成本。

正式目录：/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v0_junction_sampling_hlm_seed01_20261104
每run的artifacts/train.csv记录loss与sampling统计；eval.csv记录成功率；behavior.csv及behavior/step_*/task_*.json,npz记录行为和原始轨迹。原生rollout的访问上下文可能改变，须按共同岔口/来向/goal格报告覆盖，再比较首次出口和返回；不能只比较总返回次数并忽略更远的访问范围。

checkpoint保存独立donor RNG及索引签名，启用干预时缺失或不匹配会拒绝恢复。旧无干预checkpoint签名保持兼容。GPU通路/被动记录的短检查独立于16个正式运行，不用于科研效果结论。
