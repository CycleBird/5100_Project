# AlphaZero Gomoku Usage Guide

## 1. Create and Activate the Conda Environment

```bash
conda create -n gomoku python=3.11 -y
conda activate gomoku
```

## 2. Install Dependencies

```bash
pip install numpy torch matplotlib pandas
```

## 3. Enter the Project Directory

```bash
cd AlphaZero_Gomoku
```

## 4. Run Training

Basic training:

```bash
python train.py
```

Pruning-based training:

885 rule:
1. Compare with the best 885 model:
```bash
python prun_best.py
```

2. Play against basic MCTS:
```bash
python prun_mcts.py
```
3. 884 rule:
```bash
python prun_884.py
```


To generate training data and figures suitable for a progress report, it is recommended to run training with arguments, for example:

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

During training, the following files will also be saved:

- `metrics.csv`: training metrics for each batch
- `run_config.json`: configuration for this training run
- `.model`: the model trained in this run

## 5. Generate Training Process Charts

After training is complete, run:

```bash
python plot_training_metrics.py \
  --metrics-file report_run_small/metrics.csv \
  --output-dir report_run_small/plots
```

This will generate the following figures:

- `episode_len.svg`
- `buffer_size.svg`
- `loss.svg`
- `entropy.svg`
- `win_ratio.svg`
- `effective_lr.svg`

## 6. Output Description

Example training output directory: `AlphaZero_Gomoku/report_run_small`

- `metrics.csv`：raw training data, which can be used for report tables or further analysis
- `plots/*.svg`：figures that can be directly inserted into the progress report
- `plots/summary.json`：summary of the training results

## 7. Common Notes

- `train.py` is the main training entry point
- `plot_training_metrics.py` is used to turn training logs into figures
- To quickly test the workflow, use smaller values for `--game-batch-num` and `--n-playout`
- To obtain more stable results, increase `--game-batch-num`、`--eval-games` and `--n-playout`