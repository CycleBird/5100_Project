"""
This script compares the training results from multiple experiments. 
It reads metrics.csv files from different training folders, 
extracts key metrics such as episode length, replay buffer size, loss, entropy, win ratio, and learning rate, 
and draws them together in the same SVG line charts.

@author: Zengzheng Jiang
"""

import argparse
import csv
import os


def load_rows(metrics_file):
    with open(metrics_file, 'r', newline='', encoding='utf-8') as csvfile:
        return list(csv.DictReader(csvfile))


def to_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def series_from_rows(rows, y_key):
    points = []
    for row in rows:
        x = to_float(row.get('batch'))
        y = to_float(row.get(y_key))
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def compute_global_bounds(all_series):
    xs = []
    ys = []
    for points in all_series:
        for x, y in points:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    return min_x, max_x, min_y, max_y


def scale_points(points, width, height, padding, bounds):
    min_x, max_x, min_y, max_y = bounds
    scaled = []

    for x, y in points:
        px = padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)
        py = height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)
        scaled.append((px, py))

    return scaled


def make_polyline(points):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)


def write_multi_svg_chart(output_file, title, y_label, named_series, colors):
    width = 1000
    height = 520
    padding = 70

    valid_series = [(name, pts) for name, pts in named_series if pts]
    if not valid_series:
        return

    bounds = compute_global_bounds([pts for _, pts in valid_series])
    if bounds is None:
        return

    min_x, max_x, min_y, max_y = bounds

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg_lines.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    bottom = height - padding
    right = width - padding
    svg_lines.append(f'  <line x1="{padding}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333333" stroke-width="2" />')
    svg_lines.append(f'  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{bottom}" stroke="#333333" stroke-width="2" />')

    svg_lines.append(f'  <text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>')
    svg_lines.append(f'  <text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="16" font-family="Arial">Batch</text>')
    svg_lines.append(
        f'  <text x="28" y="{height/2}" text-anchor="middle" font-size="16" font-family="Arial" '
        f'transform="rotate(-90 28 {height/2})">{y_label}</text>'
    )

    svg_lines.append(f'  <text x="{padding}" y="{padding-12}" font-size="13" font-family="Arial">{max_y:.3f}</text>')
    svg_lines.append(f'  <text x="{padding}" y="{bottom+20}" font-size="13" font-family="Arial">{min_y:.3f}</text>')
    svg_lines.append(f'  <text x="{padding}" y="{bottom+38}" font-size="13" font-family="Arial">{min_x:.0f}</text>')
    svg_lines.append(f'  <text x="{right}" y="{bottom+38}" text-anchor="end" font-size="13" font-family="Arial">{max_x:.0f}</text>')

    for i in range(1, 5):
        y = padding + i * (height - 2 * padding) / 5
        svg_lines.append(
            f'  <line x1="{padding}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            f'stroke="#dddddd" stroke-width="1" stroke-dasharray="4,4" />'
        )

    for idx, (name, points) in enumerate(valid_series):
        color = colors[idx % len(colors)]
        scaled = scale_points(points, width, height, padding, bounds)
        polyline = make_polyline(scaled)
        svg_lines.append(
            f'  <polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}" />'
        )

    legend_x = right - 210
    legend_y = padding + 10
    legend_w = 190
    legend_h = 28 * len(valid_series) + 15
    svg_lines.append(
        f'  <rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" '
        f'fill="#ffffff" stroke="#cccccc" stroke-width="1" rx="8" ry="8" />'
    )

    for idx, (name, _) in enumerate(valid_series):
        color = colors[idx % len(colors)]
        y = legend_y + 25 + idx * 28
        svg_lines.append(
            f'  <line x1="{legend_x+12}" y1="{y}" x2="{legend_x+42}" y2="{y}" '
            f'stroke="{color}" stroke-width="4" />'
        )
        svg_lines.append(
            f'  <text x="{legend_x+50}" y="{y+5}" font-size="14" font-family="Arial">{name}</text>'
        )

    svg_lines.append('</svg>')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))


def main():
    parser = argparse.ArgumentParser(description='Create comparison SVG charts from multiple metrics.csv files')
    parser.add_argument('--output-dir', default='comparison_plots')
    args = parser.parse_args()

    experiments = [
        ('base_data', 'base_data/metrics.csv'),
        ('pruning_884', 'pruning_training_884/metrics.csv'),
        ('pruning_mcts_885', 'pruning_training_mcts/metrics.csv'),
        ('pruning_best885', 'pruning_training_best885/metrics.csv'),
    ]

    charts = [
        ('episode_len', 'Self-Play Episode Length', 'Episode Length'),
        ('buffer_size', 'Replay Buffer Size Comparison', 'Buffer Size'),
        ('loss', 'Training Loss Comparison', 'Loss'),
        ('entropy', 'Policy Entropy Comparison', 'Entropy'),
        ('win_ratio', 'Evaluation Win Ratio Comparison', 'Win Ratio'),
        ('effective_lr', 'Effective Learning Rate Comparison', 'Learning Rate'),
    ]

    colors = ['#2563eb', '#dc2626', '#16a34a', '#ea580c']

    os.makedirs(args.output_dir, exist_ok=True)

    loaded_data = {}
    for exp_name, metrics_path in experiments:
        if not os.path.exists(metrics_path):
            print(f'[Warning] Missing file: {metrics_path}')
            loaded_data[exp_name] = []
            continue
        loaded_data[exp_name] = load_rows(metrics_path)

    for key, title, y_label in charts:
        named_series = []
        for exp_name, _ in experiments:
            rows = loaded_data.get(exp_name, [])
            points = series_from_rows(rows, key)
            named_series.append((exp_name, points))

        output_file = os.path.join(args.output_dir, f'{key}.svg')
        write_multi_svg_chart(output_file, title, y_label, named_series, colors)
        print(f'Saved: {output_file}')


if __name__ == '__main__':
    main()