import argparse
import csv
import json
import os


def load_rows(metrics_file):
    with open(metrics_file, 'r', newline='') as csvfile:
        return list(csv.DictReader(csvfile))


def to_float(value):
    if value in (None, ''):
        return None
    return float(value)


def series_from_rows(rows, y_key):
    points = []
    for row in rows:
        x = to_float(row.get('batch'))
        y = to_float(row.get(y_key))
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def scale_points(points, width, height, padding):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0
    scaled = []
    for x, y in points:
        px = padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)
        py = height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)
        scaled.append((px, py))
    return scaled, (min_x, max_x, min_y, max_y)


def write_svg_chart(output_file, title, y_label, points, color):
    width = 800
    height = 420
    padding = 60
    scaled_points, bounds = scale_points(points, width, height, padding)
    polyline = ' '.join('{:.2f},{:.2f}'.format(x, y) for x, y in scaled_points)
    min_x, max_x, min_y, max_y = bounds
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
  <line x1="{padding}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333333" stroke-width="2" />
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{bottom}" stroke="#333333" stroke-width="2" />
  <text x="{center}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>
  <text x="{center}" y="{xlabel_y}" text-anchor="middle" font-size="16" font-family="Arial">Batch</text>
  <text x="24" y="{center_y}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {center_y})">{y_label}</text>
  <text x="{padding}" y="{label_top}" font-size="13" font-family="Arial">{max_y:.3f}</text>
  <text x="{padding}" y="{label_bottom}" font-size="13" font-family="Arial">{min_y:.3f}</text>
  <text x="{padding}" y="{xlabel_value_y}" font-size="13" font-family="Arial">{min_x:.0f}</text>
  <text x="{right}" y="{xlabel_value_y}" text-anchor="end" font-size="13" font-family="Arial">{max_x:.0f}</text>
  <polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}" />
</svg>
""".format(
        width=width,
        height=height,
        padding=padding,
        bottom=height - padding,
        right=width - padding,
        center=width / 2,
        center_y=height / 2,
        xlabel_y=height - 15,
        label_top=padding - 10,
        label_bottom=height - padding + 20,
        xlabel_value_y=height - padding + 35,
        title=title,
        y_label=y_label,
        max_y=max_y,
        min_y=min_y,
        min_x=min_x,
        max_x=max_x,
        color=color,
        polyline=polyline,
    )
    with open(output_file, 'w') as svgfile:
        svgfile.write(svg)


def write_summary(rows, output_dir):
    summary = {
        'total_batches': len(rows),
        'total_updates': sum(1 for row in rows if row.get('loss')),
        'total_evaluations': sum(1 for row in rows if row.get('win_ratio')),
    }
    if rows:
        last_row = rows[-1]
        for key in ['episode_len', 'buffer_size', 'loss', 'entropy', 'win_ratio', 'best_win_ratio']:
            summary['final_' + key] = to_float(last_row.get(key))
    with open(os.path.join(output_dir, 'summary.json'), 'w') as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description='Create SVG charts from training metrics.csv')
    parser.add_argument('--metrics-file', default='training_artifacts/metrics.csv')
    parser.add_argument('--output-dir', default='training_artifacts/plots')
    args = parser.parse_args()

    rows = load_rows(args.metrics_file)
    os.makedirs(args.output_dir, exist_ok=True)

    charts = [
        ('episode_len', 'Self-Play Episode Length', 'Episode Length', '#2563eb'),
        ('buffer_size', 'Replay Buffer Size', 'Buffer Size', '#7c3aed'),
        ('loss', 'Training Loss', 'Loss', '#dc2626'),
        ('entropy', 'Policy Entropy', 'Entropy', '#ea580c'),
        ('win_ratio', 'Evaluation Win Ratio', 'Win Ratio', '#16a34a'),
        ('effective_lr', 'Effective Learning Rate', 'Learning Rate', '#0891b2'),
    ]

    for key, title, y_label, color in charts:
        points = series_from_rows(rows, key)
        if not points:
            continue
        write_svg_chart(
            os.path.join(args.output_dir, key + '.svg'),
            title,
            y_label,
            points,
            color,
        )

    write_summary(rows, args.output_dir)


if __name__ == '__main__':
    main()
