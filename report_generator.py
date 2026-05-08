"""
PDF Report Generator for Bayes Tree
Generates professional PDF reports from simulation results.

Copyright (c) 2026 Ari-Pekka Sihvonen
MIT License — see LICENSE file
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from bayes_engine import NodeResult


def _make_histogram_image(posteriors, width_cm=16, height_cm=8):
    """Create histogram as PNG bytes."""
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.hist(posteriors, bins=30, color='#3b82f6', edgecolor='#1e40af',
            alpha=0.85)
    ax.set_xlabel('Posterior probability', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title('Posterior Distribution', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_importance_image(importance, baseline, width_cm=16, height_cm=None):
    """Create importance ranking bar chart as PNG bytes."""
    if not importance:
        return None

    n = len(importance)
    if height_cm is None:
        height_cm = max(4, n * 1.2 + 2)

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))

    names = []
    deltas = []
    bar_colors = []
    for item in reversed(importance):
        name = item['name']
        if len(name) > 40:
            name = name[:37] + '...'
        names.append(name)
        deltas.append(item['delta'])
        bar_colors.append('#22c55e' if item['delta'] > 0 else '#ef4444')

    y_pos = range(len(names))
    ax.barh(y_pos, [abs(d) for d in deltas], color=bar_colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('|Δ posterior|', fontsize=9)
    ax.set_title('Importance Ranking (leave-one-out)', fontsize=11,
                 fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _tree_rows(node, depth=0):
    """Flatten tree into table rows."""
    rows = []
    indent = '    ' * depth
    etype_map = {'for': '(+)', 'against': '(-)', 'neutral': '( )'}
    etype_str = etype_map.get(node.etype, '( )')

    if depth == 0:
        rows.append([
            node.name,
            '',
            f'{node.prior:.1%}',
            f'{node.med:.3%}',
            f'[{node.p5:.3%}–{node.p95:.3%}]',
        ])
    else:
        if node.lr_pt is not None:
            lr_str = f'{node.lr_pt:.2f}'
        else:
            lr_str = f'[{node.lr_min:.2f}–{node.lr_max:.2f}]'
        rows.append([
            f'{indent}{etype_str} {node.name}',
            lr_str,
            f'{node.prior:.2%}',
            f'{node.med:.2%}',
            f'[{node.p5:.2%}–{node.p95:.2%}]',
        ])

    for child in node.children:
        rows.extend(_tree_rows(child, depth + 1))
    return rows


def generate_report(results, yaml_filename, output_path):
    """
    Generate a PDF report from simulation results.

    Args:
        results: dict from bayes_engine.run_simulation()
        yaml_filename: source YAML filename (for title)
        output_path: path to write the PDF
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        'Title2', parent=styles['Title'],
        fontSize=20, spaceAfter=6 * mm,
    ))
    styles.add(ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=12, textColor=colors.grey, alignment=TA_CENTER,
        spaceAfter=12 * mm,
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading2'],
        fontSize=14, spaceBefore=8 * mm, spaceAfter=4 * mm,
        textColor=colors.HexColor('#1e3a5f'),
    ))

    story = []
    tree = results['tree']

    # ── Title page ────────────────────────────────────────────────────────
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph('Bayesian Evidence Tree Report', styles['Title2']))
    story.append(Paragraph(tree.name, styles['Heading2']))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f'Source: {os.path.basename(yaml_filename)}<br/>'
        f'Simulations: {results["n_sim"]:,}<br/>'
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        styles['Subtitle'],
    ))
    story.append(PageBreak())

    # ── Summary statistics ────────────────────────────────────────────────
    story.append(Paragraph('Summary Statistics', styles['SectionHead']))

    s = results['stats']
    s_lr = results['lr_stats']
    stat_data = [
        ['Metric', 'Value'],
        ['Prior', f'{results["prior"]:.1%}'],
        ['Posterior median', f'{s["median"]:.3%}'],
        ['Posterior mean', f'{s["mean"]:.3%}'],
        ['Standard deviation', f'{s["std"]:.3%}'],
        ['90% CI', f'[{s["p5"]:.3%} – {s["p95"]:.3%}]'],
        ['Range', f'[{s["min"]:.3%} – {s["max"]:.3%}]'],
        ['Effective LR (median)', f'{s_lr["median"]:.4f}'],
        ['Effective LR (90% CI)', f'[{s_lr["p5"]:.4f} – {s_lr["p95"]:.4f}]'],
    ]

    t = Table(stat_data, colWidths=[45 * mm, 70 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8 * mm))

    # ── Histogram ─────────────────────────────────────────────────────────
    story.append(Paragraph('Posterior Distribution', styles['SectionHead']))
    hist_buf = _make_histogram_image(results['posteriors'])
    story.append(Image(hist_buf, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 6 * mm))

    # ── Sensitivity ───────────────────────────────────────────────────────
    story.append(Paragraph('Sensitivity Analysis', styles['SectionHead']))
    sens = results['sensitivity']
    sens_data = [
        ['Threshold', 'Probability'],
        ['P(posterior < 5%)', f'{sens["p_lt_5"]:.1%}'],
        ['P(posterior < 10%)', f'{sens["p_lt_10"]:.1%}'],
        ['P(posterior > 50%)', f'{sens["p_gt_50"]:.1%}'],
    ]
    t2 = Table(sens_data, colWidths=[50 * mm, 40 * mm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)

    story.append(PageBreak())

    # ── Evidence tree ─────────────────────────────────────────────────────
    story.append(Paragraph('Evidence Tree', styles['SectionHead']))
    tree_data = [['Evidence Node', 'LR', 'Prior', 'Posterior', '90% CI']]
    tree_data.extend(_tree_rows(tree))

    col_widths = [75 * mm, 25 * mm, 18 * mm, 20 * mm, 30 * mm]
    t3 = Table(tree_data, colWidths=col_widths, repeatRows=1)
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t3)

    # ── Importance ranking ────────────────────────────────────────────────
    if results['importance']:
        story.append(PageBreak())
        story.append(Paragraph('Importance Ranking (Leave-One-Out)',
                               styles['SectionHead']))

        imp_buf = _make_importance_image(results['importance'],
                                         results['baseline'])
        if imp_buf:
            n_items = len(results['importance'])
            chart_h = max(4, n_items * 1.2 + 2)
            story.append(Image(imp_buf, width=16 * cm,
                               height=chart_h * cm))
            story.append(Spacer(1, 6 * mm))

        imp_data = [['Rank', 'Evidence', 'Type', 'Δ Posterior', 'Without']]
        etype_map = {'for': '(+)', 'against': '(-)', 'neutral': '( )'}
        for rank, item in enumerate(results['importance'], 1):
            name = item['name']
            if len(name) > 45:
                name = name[:42] + '...'
            arrow = '↑' if item['delta'] > 0 else '↓'
            imp_data.append([
                str(rank),
                name,
                etype_map.get(item['evidence_type'], '( )'),
                f'{arrow} {abs(item["delta"]):.4%}',
                f'{item["median_without"]:.4%}',
            ])

        t4 = Table(imp_data, colWidths=[12 * mm, 75 * mm, 12 * mm, 28 * mm, 25 * mm],
                   repeatRows=1)
        t4.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t4)

    # ── Warnings ──────────────────────────────────────────────────────────
    if results['warnings']:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph('Warnings', styles['SectionHead']))
        for w in results['warnings']:
            cleaned = w.replace('\n', '<br/>')
            story.append(Paragraph(cleaned, styles['Normal']))
            story.append(Spacer(1, 2 * mm))

    doc.build(story)
    return output_path
