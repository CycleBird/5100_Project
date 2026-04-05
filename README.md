# AlphaZero Gomoku 使用方法

## 1. 创建并进入 Conda 环境

```bash
conda create -n gomoku python=3.11 -y
conda activate gomoku
```

## 2. 安装依赖

```bash
pip install numpy torch matplotlib pandas
```

## 3. 进入项目目录

```bash
cd AlphaZero_Gomoku
```

## 4. 运行训练

最简单的训练方式：

```bash
python train.py
```

如果想生成适合 progress report 的训练数据和图，建议使用带参数的方式，例如：

```bash
python train.py \
  --game-batch-num 20 \
  --check-freq 5 \
  --eval-games 4 \
  --batch-size 32 \
  --buffer-size 1000 \
  --epochs 3 \
  --n-playout 50 \
  --pure-mcts-playout-num 100 \
  --output-dir report_run_small
```

训练过程中会额外保存：

- `metrics.csv`：每个 batch 的训练指标
- `run_config.json`：本次训练配置

## 5. 生成训练过程图表

训练完成后运行：

```bash
python plot_training_metrics.py \
  --metrics-file report_run_small/metrics.csv \
  --output-dir report_run_small/plots
```

会生成这些图：

- `episode_len.svg`
- `buffer_size.svg`
- `loss.svg`
- `entropy.svg`
- `win_ratio.svg`
- `effective_lr.svg`

## 6. 输出内容说明

训练输出目录示例：`AlphaZero_Gomoku/report_run_small`

- `metrics.csv`：原始训练数据，可用于报告表格或进一步分析
- `plots/*.svg`：可直接插入 progress report 的图
- `plots/summary.json`：训练结果摘要

## 7. 常用说明

- `train.py` 是训练入口
- `plot_training_metrics.py` 用于把训练日志画成图
- 如果只想快速验证流程，使用较小的 `--game-batch-num` 和 `--n-playout`
- 如果想得到更稳定的结果，可以增大 `--game-batch-num`、`--eval-games` 和 `--n-playout`