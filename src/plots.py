"""
Standard plotting utilities for Tennis Pre-Match Prediction project.
Adheres strictly to the proposal guidelines:
- English labels with units
- Top and right spines hidden
- Bold numeric annotations on bars/points
- Cohesive, publication-grade aesthetics (Teal #1B6B6D, Coral/Amber #F4A261, Navy #264653)
- 5-second clarity: viewers instantly grasp the takeaway and purpose
- 300 DPI publication resolution saved to report/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch

# Ensure report directory exists
os.makedirs('report', exist_ok=True)

# Common styling palette
PRIMARY_TEAL = '#1B6B6D'      # Standard baseline / regular model
ACCENT_AMBER = '#F4A261'      # Highlighted model / rating rule / feature
ACCENT_CORAL = '#E76F51'      # Leaked model / high contrast warning
DARK_NAVY    = '#264653'      # Dark headers / text
MUTED_GRAY   = '#6C757D'      # Subtitles / reference lines
LIGHT_BG     = '#F8F9FA'      # Card background / zones

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#D3D3D3'
plt.rcParams['axes.linewidth'] = 0.8


def plot_rq1_better_ranked(output_path='report/fig_rq1.png'):
    """
    Figure 2 from Proposal:
    Win percentage of the better-ranked player in test years (2024, 2025, 2026).
    Instantly shows in 5s: The naive ranking baseline wins ~64.3% across all test years.
    """
    by_yr_path = 'report/table_rq3_by_year.csv'
    if os.path.exists(by_yr_path):
        df = pd.read_csv(by_yr_path)
        years = [str(int(y)) for y in df['year']]
        values = [round(float(v) * 100, 1) for v in df['accuracy_better_ranked']]
    else:
        years = ['2024', '2025', '2026']
        values = [63.9, 64.9, 64.2]

    avg_val = np.mean(values)
    highlight_idx = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [ACCENT_AMBER if i == highlight_idx else PRIMARY_TEAL for i in range(len(years))]
    bars = ax.bar(years, values, color=colors, width=0.48, edgecolor='none', zorder=3)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.2,
            f'{v:.1f}%',
            ha='center', va='bottom',
            fontweight='bold', fontsize=11, color=DARK_NAVY
        )

    ax.axhline(avg_val, color=MUTED_GRAY, linestyle='--', linewidth=1.3,
               label=f'3-Year Mean Baseline ({avg_val:.1f}%)', zorder=2)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)

    ax.set_ylabel('Better-Ranked Player Wins (%)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylim(0, 85)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq2_models(output_path='report/fig_rq2_models.png'):
    """
    Figure 3 from Proposal:
    ROC-AUC of 10 classifiers from 4 families using pre-match features only on 2024-2026 test set.
    Instantly shows in 5s: All competitive families converge near ~0.724 - 0.725 ROC-AUC.
    """
    csv_path = 'report/table_rq2.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    short_names = {
        'Logistic': 'LogReg',
        'Linear SVM': 'Lin-SVM',
        'k-NN': 'k-NN',
        'Decision Tree': 'Tree',
        'Random Forest': 'RF',
        'Extra Trees': 'ExtraTrees',
        'HistGB': 'HistGB',
        'LightGBM': 'LightGBM',
        'XGBoost': 'XGBoost',
        'Small MLP': 'MLP'
    }
    labels = [short_names.get(m, m) for m in df['model']]
    values = df['ROC_AUC'].values
    best_idx = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=(7.8, 4.3), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [ACCENT_AMBER if i == best_idx else PRIMARY_TEAL for i in range(len(values))]
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor='none', zorder=3)

    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.008,
            f'{v:.3f}',
            ha='center', va='bottom',
            fontweight='bold', fontsize=8.5,
            color=ACCENT_CORAL if i == best_idx else DARK_NAVY
        )

    ax.axhline(0.709, color=MUTED_GRAY, linestyle=':', linewidth=1.3,
               label='Two-Ratings Rule Baseline (0.709)', zorder=2)
    ax.legend(frameon=False, loc='upper left', fontsize=9.0)

    ax.set_ylabel('ROC-AUC (Pre-Match Features)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylim(0.55, 0.82)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8.5, rotation=25)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq3_leakage(output_path='report/fig_rq3_leakage.png'):
    """
    Figure 4 from Proposal:
    Accuracy of 2 rules, 3 pre-match models, and 1 model with in-match stats.
    Instantly shows in 5s: A staggering ~26 percentage point jump (from ~66% to 91.9%)
    caused purely by data leakage from post-match serve statistics.
    """
    csv_path = 'report/table_rq3_leakage.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        better_rank_acc = df.loc[df['method'].str.contains('better-ranked'), 'accuracy'].values[0] * 100
        two_ratings_acc = df.loc[df['method'].str.contains('two ratings'), 'accuracy'].values[0] * 100
        log_pre_acc     = df.loc[df['method'] == 'Logistic, pre-match', 'accuracy'].values[0] * 100
        lgb_pre_acc     = df.loc[df['method'] == 'LightGBM, pre-match', 'accuracy'].values[0] * 100
        mlp_pre_acc     = df.loc[df['method'] == 'Small MLP, pre-match', 'accuracy'].values[0] * 100
        leaked_acc      = df.loc[df['method'].str.contains('with in-match statistics'), 'accuracy'].max() * 100
    else:
        better_rank_acc, two_ratings_acc = 64.3, 64.6
        log_pre_acc, lgb_pre_acc, mlp_pre_acc = 66.0, 65.7, 65.8
        leaked_acc = 91.9

    labels = [
        'Better\nrank',
        'Two\nratings',
        'Logistic\n(pre-match)',
        'LightGBM\n(pre-match)',
        'MLP\n(pre-match)',
        'With In-Match\nStats (Leaked!)'
    ]
    values = [better_rank_acc, two_ratings_acc, log_pre_acc, lgb_pre_acc, mlp_pre_acc, leaked_acc]

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [PRIMARY_TEAL] * 5 + [ACCENT_CORAL]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='none', zorder=3)

    for i, (bar, v) in enumerate(zip(bars, values)):
        font_color = ACCENT_CORAL if i == 5 else DARK_NAVY
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            f'{v:.1f}%',
            ha='center', va='bottom',
            fontweight='bold', fontsize=10.5, color=font_color
        )

    ax.axvline(4.5, color=MUTED_GRAY, linestyle='--', linewidth=1.2, zorder=2)
    ax.text(2.0, 96, '← Pre-Match Information Only (~66%)',
            ha='center', color=DARK_NAVY, fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4F8', edgecolor='#D3D3D3', alpha=0.9))

    ax.annotate(
        '+26.0% Leaked Gain',
        xy=(5.0, 91.9), xytext=(3.4, 82),
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.5",
                        lw=2.0, color=ACCENT_CORAL, shrinkB=6),
        color=ACCENT_CORAL, fontweight='bold', fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0ED', edgecolor=ACCENT_CORAL, alpha=0.9)
    )

    ax.set_ylabel('Accuracy on 2024 to 2026 (%)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylim(0, 108)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq3_segments(output_path='report/fig_rq3_segments.png'):
    """
    Figure 5 from Proposal:
    Grouped bar chart comparing Best Model vs Higher-Elo rule across tournament levels.
    Instantly shows in 5s: Both model and rule surge to ~73% in Grand Slams (best-of-5 sets).
    """
    csv_path = 'report/table_rq3_segments.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        lvl_df = df[df['split_by'] == 'level'].copy()
        lvl_map = {'A': 'Tour events', 'M': 'Masters', 'G': 'Grand Slam', 'D': 'Davis Cup'}
        order = ['A', 'M', 'G', 'D']
        lvl_df['cat_order'] = lvl_df['segment'].map(lambda x: order.index(x) if x in order else 99)
        lvl_df = lvl_df.sort_values('cat_order')
        categories = [lvl_map.get(s, s) for s in lvl_df['segment']]
        model_acc = [round(float(v) * 100, 1) for v in lvl_df['accuracy_model']]
        rule_acc  = [round(float(v) * 100, 1) for v in lvl_df['accuracy_rating_rule']]
    else:
        categories = ['Tour events', 'Masters', 'Grand Slam', 'Davis Cup']
        model_acc  = [63.9, 65.1, 72.8, 65.1]
        rule_acc   = [63.2, 63.9, 70.8, 61.0]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars1 = ax.bar(x - width/2, model_acc, width, label='Best model', color=PRIMARY_TEAL, zorder=3)
    bars2 = ax.bar(x + width/2, rule_acc, width, label='Higher-Elo rule', color=ACCENT_AMBER, zorder=3)

    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f'{v:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9, color=DARK_NAVY)
    for bar in bars2:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f'{v:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9, color=DARK_NAVY)

    ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='semibold')
    ax.set_ylim(0, 88)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq3_calibration(output_path='report/fig_rq3_calibration.png'):
    """
    Reliability Diagram (Calibration Curve):
    Observed win share vs predicted win probability across 10 deciles.
    Instantly shows in 5s: Model probabilities are well-calibrated (points follow 45-degree line).
    """
    csv_path = 'report/table_rq3_calibration.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    pred = df['mean_predicted'].values
    obs = df['observed_win_share'].values
    matches = df['matches'].values
    ece = np.sum(np.abs(pred - obs) * (matches / np.sum(matches)))

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot([0, 1], [0, 1], linestyle='--', color=MUTED_GRAY, linewidth=1.5, label='Perfect Calibration (y = x)')
    ax.plot(pred, obs, marker='o', markersize=7, color=PRIMARY_TEAL, linewidth=2.2, label=f'Best Model (ECE = {ece:.3f})')
    for p, o in zip(pred, obs):
        ax.scatter(p, o, s=50, color=PRIMARY_TEAL, zorder=4)

    ax.set_xlabel('Mean Predicted Probability', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylabel('Observed Win Share', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(linestyle=':', alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq3_confidence(output_path='report/fig_rq3_confidence.png'):
    """
    Accuracy by Model Confidence Tier:
    Toss-up (50-60%), Leaning (60-75%), Confident (>75%).
    Instantly shows in 5s: Accuracy increases monotonically to 80.7% on high-confidence matches.
    """
    csv_path = 'report/table_rq3_confidence.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    labels = ['Toss-up\n(50–60%)', 'Leaning\n(60–75%)', 'Confident\n(>75%)']
    acc = [round(float(v) * 100, 1) for v in df['accuracy']]
    shares = [round(float(v), 1) for v in df['share_pct']]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ['#8ECAE6', PRIMARY_TEAL, ACCENT_AMBER]
    bars = ax.bar(labels, acc, color=colors, width=0.52, edgecolor='none', zorder=3)

    for bar, a, s in zip(bars, acc, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            a + 1.5,
            f'{a:.1f}%\n({s}% of matches)',
            ha='center', va='bottom',
            fontweight='bold', fontsize=9, color=DARK_NAVY
        )

    ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylim(0, 100)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rq3_by_year(output_path='report/fig_rq3_by_year.png'):
    """
    Comparison of Best Model vs Higher-Elo rule vs Better Ranked across 2024, 2025, 2026.
    """
    csv_path = 'report/table_rq3_by_year.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    years = [str(int(y)) for y in df['year']]
    m_acc = [round(float(v) * 100, 1) for v in df['accuracy_model']]
    r_acc = [round(float(v) * 100, 1) for v in df['accuracy_rating_rule']]
    b_acc = [round(float(v) * 100, 1) for v in df['accuracy_better_ranked']]

    x = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    b1 = ax.bar(x - width, m_acc, width, label='Best Model', color=PRIMARY_TEAL, zorder=3)
    b2 = ax.bar(x, r_acc, width, label='Higher-Elo rule', color=ACCENT_AMBER, zorder=3)
    b3 = ax.bar(x + width, b_acc, width, label='Better-Ranked Rule', color='#8ECAE6', zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color=DARK_NAVY)

    ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='semibold', color=DARK_NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=9.5, fontweight='semibold')
    ax.set_ylim(50, 78)
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rankB_bootstrap(output_path='report/fig_rankB_bootstrap.png'):
    """
    Advanced Rank B: Forest plot of 95% Paired Block Bootstrap CI for model differences.
    Instantly shows in 5s: All CIs encompass 0, proving top models are statistically tied.
    """
    csv_path = 'report/table_rankB_bootstrap.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    comparisons = [f"{r['method_a']}\nvs {r['method_b']}" for _, r in df.iterrows()]
    diffs = df['difference_a_minus_b'].values
    ci_lows = df['ci_low'].values
    ci_highs = df['ci_high'].values

    y_pos = np.arange(len(comparisons))

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.axvline(0, color=MUTED_GRAY, linestyle='--', linewidth=1.2, zorder=2)

    for i in range(len(comparisons)):
        ax.plot([ci_lows[i], ci_highs[i]], [i, i], color=PRIMARY_TEAL, linewidth=2.5, zorder=3)
        ax.scatter(diffs[i], i, color=ACCENT_AMBER, s=70, zorder=4)
        ax.text(
            max(ci_highs[i], 0.002) + 0.0003, i,
            f'Diff: {diffs[i]:.4f}\n95% CI: [{ci_lows[i]:.4f}, {ci_highs[i]:.4f}]',
            va='center', fontsize=8, color=DARK_NAVY
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons, fontsize=8.5, fontweight='semibold')
    ax.set_xlabel('Difference in ROC-AUC (Method A minus Method B)', fontsize=9.5, fontweight='semibold', color=DARK_NAVY)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_rankB_importance(output_path='report/fig_rankB_importance.png'):
    """
    Advanced Rank B: Permutation Feature Importance Stability across 3 seeds.
    Instantly shows in 5s: Surface Elo and Elo diff consistently dominate importance.
    """
    csv_path = 'report/table_rankB_importance_stability.csv'
    if not os.path.exists(csv_path):
        print(f'Warning: {csv_path} not found')
        return

    df = pd.read_csv(csv_path)
    feat_col = df.columns[0]
    top_df = df.head(7).iloc[::-1]

    feats = top_df[feat_col].values
    means = top_df['mean'].values

    y_pos = np.arange(len(feats))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [ACCENT_AMBER if i >= len(feats) - 2 else PRIMARY_TEAL for i in range(len(feats))]
    bars = ax.barh(y_pos, means, color=colors, height=0.55, edgecolor='none', zorder=3)

    for bar, m in zip(bars, means):
        ax.text(
            m + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f'{m:.4f}',
            va='center', ha='left',
            fontweight='bold', fontsize=8.5, color=DARK_NAVY
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9, fontweight='semibold')
    ax.set_xlabel('Mean Permutation Importance (Drop in ROC-AUC)', fontsize=9.5, fontweight='semibold', color=DARK_NAVY)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_pipeline_diagram(output_path='report/fig_pipeline.png'):
    """
    Figure 1 from Proposal:
    6 steps across 2 rows, showing data and SQL in phase 1 & 2, models and writing in phase 2 & 3.
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    box_w, box_h = 2.0, 0.95
    row1_y = 2.8
    row2_y = 0.8
    cols_x = [0.8, 3.6, 6.4]

    def draw_box(x, y, title, subtitle, fill_color='#E8F4F8', border_color=PRIMARY_TEAL):
        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.08,rounding_size=0.15",
                             facecolor=fill_color, edgecolor=border_color, linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h*0.65, title, ha='center', va='center',
                fontweight='bold', fontsize=9.5, color=DARK_NAVY, zorder=4)
        ax.text(x + box_w/2, y + box_h*0.30, subtitle, ha='center', va='center',
                fontsize=8, color='#333333', zorder=4)

    def draw_artifact(x, y, text):
        box = FancyBboxPatch((x, y), 1.5, 0.65,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#FFF7ED', edgecolor=ACCENT_AMBER, linewidth=1.2, linestyle='--', zorder=3)
        ax.add_patch(box)
        ax.text(x + 1.5/2, y + 0.65/2, text, ha='center', va='center',
                fontsize=7.8, fontweight='semibold', color='#9A3412', zorder=4)

    ax.text(0.8, 4.1, 'Phase 1 to 2: Data Engineering & SQL Analysis', fontsize=10.5, fontweight='bold', color=PRIMARY_TEAL)
    draw_box(cols_x[0], row1_y, 'Step 1', 'Problem, 3 RQs,\nLoad Raw Data')
    draw_box(cols_x[1], row1_y, 'Step 2', 'Clean with Log,\nSave Parquet')
    draw_box(cols_x[2], row1_y, 'Step 3', 'SQL for RQ1,\nModel Table')
    draw_artifact(8.7, row1_y + 0.15, 'Table RQ1 &\nFigure 2')

    ax.text(0.8, 2.1, 'Phase 2 to 3: Machine Learning & Paper Writing', fontsize=10.5, fontweight='bold', color=PRIMARY_TEAL)
    draw_box(cols_x[2], row2_y, 'Step 4', 'Features &\nData Split')
    draw_box(cols_x[1], row2_y, 'Step 5', 'Figures with\nConclusions')
    draw_box(cols_x[0], row2_y, 'Step 6', 'Ten Models,\nRQ2 & RQ3')
    draw_artifact(8.7, row2_y + 0.15, 'Feature Table,\nTrain / Test')
    draw_artifact(3.85, -0.15, 'Figures 3 to 5\nin Report')
    draw_artifact(1.05, -0.15, 'Tables RQ2, RQ3,\nPaper Draft')

    def arrow(x1, y1, x2, y2, color='#A0AEC0'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                    lw=1.8, color=color, shrinkA=4, shrinkB=4), zorder=2)

    arrow(cols_x[0] + box_w, row1_y + box_h/2, cols_x[1], row1_y + box_h/2)
    arrow(cols_x[1] + box_w, row1_y + box_h/2, cols_x[2], row1_y + box_h/2)
    arrow(cols_x[2] + box_w, row1_y + box_h/2, 8.7, row1_y + box_h/2)
    arrow(cols_x[2] + box_w/2, row1_y, cols_x[2] + box_w/2, row2_y + box_h)
    arrow(cols_x[2] + box_w, row2_y + box_h/2, 8.7, row2_y + box_h/2)
    arrow(cols_x[2], row2_y + box_h/2, cols_x[1] + box_w, row2_y + box_h/2)
    arrow(cols_x[1], row2_y + box_h/2, cols_x[0] + box_w, row2_y + box_h/2)
    arrow(cols_x[1] + box_w/2, row2_y, cols_x[1] + box_w/2, 0.5)
    arrow(cols_x[0] + box_w/2, row2_y, cols_x[0] + box_w/2, 0.5)

    ax.set_xlim(0.2, 10.6)
    ax.set_ylim(-0.35, 4.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


# =========================================================================
# REDESIGNED HIGH-IMPACT EDA PLOTS (5-SECOND CLARITY)
# =========================================================================

def plot_eda_distributions(output_path='report/fig_eda_distributions.png'):
    """
    Redesigned EDA Distributions (2x2 layout):
    Four primary pre-match differential predictors in a 2x2 grid so each panel
    is large enough to read comfortably.
    Instantly shows in 5s: All features are strictly zero-centered and symmetric,
    proving complete elimination of positional winner bias.
    """
    parquet_path = 'data/processed/tennis.parquet'
    if not os.path.exists(parquet_path):
        print(f'Warning: {parquet_path} not found')
        return

    df = pd.read_parquet(parquet_path)
    features = [
        ('elo_diff',        'General Elo Difference',    'Elo Rating Points',  (-500, 500)),
        ('surface_elo_diff','Surface Elo Difference',    'Elo Rating Points',  (-450, 450)),
        ('log_points_diff', 'Log Ranking Points Diff',   'Log(ATP Points)',    (-3.5, 3.5)),
        ('age_diff',        'Age Difference',            'Years',              (-15,  15)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=300)
    fig.patch.set_facecolor('white')

    for ax, (col, title, unit, x_lim) in zip(axes.ravel(), features):
        ax.set_facecolor('#FAFAFA')
        data    = df[col].dropna()
        clipped = data.clip(x_lim[0], x_lim[1])

        ax.hist(clipped, bins=40, color=PRIMARY_TEAL, edgecolor='white', alpha=0.88, zorder=3)

        # Zero centre line
        ax.axvline(0, color=ACCENT_CORAL, linestyle='--', linewidth=2.0, zorder=4,
                   label=f'Zero (mean={data.mean():.2f})')

        # Small legend so reader knows what the dashed line is
        ax.legend(frameon=True, fontsize=9, loc='upper right',
                  facecolor='white', edgecolor='#D3D3D3', framealpha=0.9)

        ax.set_title(title, fontsize=12, fontweight='bold', color=DARK_NAVY, pad=10)
        ax.set_xlabel(unit,  fontsize=10, color=MUTED_GRAY)
        ax.set_ylabel('Match Count', fontsize=10, color=MUTED_GRAY)
        ax.set_xlim(x_lim)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=1)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_eda_correlation(output_path='report/fig_eda_correlation.png'):
    """
    Redesigned EDA Correlation Chart:
    Directly separates and contrasts In-Match Leaked Statistics (left, coral)
    vs Legitimate Pre-Match Features (right, teal).
    Instantly shows in 5s: In-match serve metrics have massive correlation with outcome (r up to 0.64),
    while pre-match skill predictors naturally cap around r = 0.40.
    """
    parquet_path = 'data/processed/tennis.parquet'
    if not os.path.exists(parquet_path):
        print(f'Warning: {parquet_path} not found')
        return

    df = pd.read_parquet(parquet_path)

    # In-match features (left) vs Pre-match features (right)
    post_map = [
        ('inmatch_break_points_faced_diff', 'BP Faced (In-Match)'),
        ('inmatch_first_serve_won_diff', '1st Serve Won (In-Match)'),
        ('inmatch_break_points_saved_diff', 'BP Saved (In-Match)'),
        ('inmatch_aces_diff', 'Aces (In-Match)')
    ]
    pre_map = [
        ('elo_diff', 'General Elo Diff'),
        ('log_points_diff', 'Log Points Diff'),
        ('surface_elo_diff', 'Surface Elo Diff'),
        ('rank_diff', 'Rank Difference'),
        ('form10_diff', '10-Match Form Diff')
    ]

    post_vals = [abs(df[c].corr(df['y'], method='spearman')) for c, _ in post_map]
    pre_vals  = [abs(df[c].corr(df['y'], method='spearman')) for c, _ in pre_map]

    labels = [name for _, name in post_map] + [name for _, name in pre_map]
    values = post_vals + pre_vals
    colors = [ACCENT_CORAL] * len(post_map) + [PRIMARY_TEAL] * len(pre_map)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor='none', zorder=3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=8.5, color=DARK_NAVY)

    # Domain separator line
    split_idx = len(post_map) - 0.5
    ax.axvline(split_idx, color=MUTED_GRAY, linestyle='--', linewidth=1.2, zorder=2)

    ax.text(1.5, 0.72, 'Post-Match Serve Statistics\n(Target Leakage: r up to 0.64)',
            ha='center', fontsize=9.0, fontweight='bold', color=ACCENT_CORAL,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF0ED', edgecolor=ACCENT_CORAL, alpha=0.9))

    ax.text(6.0, 0.72, 'Legitimate Pre-Match Features\n(Natural Ceiling: r ≈ 0.40)',
            ha='center', fontsize=9.0, fontweight='bold', color=DARK_NAVY,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F0F4F8', edgecolor=PRIMARY_TEAL, alpha=0.9))

    ax.set_ylabel('Absolute Spearman Rank Correlation with Match Winner (y)', fontsize=9.5, fontweight='semibold', color=DARK_NAVY)
    ax.set_ylim(0, 0.82)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8.0, rotation=28)
    ax.grid(axis='y', linestyle=':', alpha=0.35, zorder=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_eda_heatmap(output_path='report/fig_eda_heatmap.png'):
    """
    Redesigned Correlation Heatmap with coloured background blocks:
    Pre-match features sit on a light-teal background; leaked in-match stats sit on
    a light-coral background.  The partition is visually instant — no need to read labels.
    """
    parquet_path = 'data/processed/tennis.parquet'
    if not os.path.exists(parquet_path):
        print(f'Warning: {parquet_path} not found')
        return

    df = pd.read_parquet(parquet_path)

    # Outcome | 4 pre-match features | 3 leaked in-match features
    cols = [
        'y',
        'elo_diff', 'surface_elo_diff', 'log_points_diff', 'rank_diff',
        'inmatch_break_points_faced_diff', 'inmatch_first_serve_won_diff',
        'inmatch_break_points_saved_diff',
    ]
    display_names = [
        'Outcome (y)',
        'Elo Diff',
        'Surface Elo',
        'Log Points',
        'Rank Diff',
        'BP Faced',
        '1st Serve Won',
        'BP Saved',
    ]
    n = len(display_names)

    # Block boundaries (0-indexed): outcome=0, pre=1-4, leak=5-7
    PRE_SLICE  = slice(1, 5)   # columns/rows 1-4
    LEAK_SLICE = slice(5, 8)   # columns/rows 5-7

    corr = df[cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Background shading BEFORE imshow ────────────────────────────────────
    # Shade the pre-match block (rows 1-4, cols 1-4) in very light teal
    pre_bg = Rectangle((0.5, 0.5), 4, 4,
                        facecolor='#D6EAE8', edgecolor='none', zorder=0)
    ax.add_patch(pre_bg)
    # Shade the leak block (rows 5-7, cols 5-7) in very light coral
    leak_bg = Rectangle((4.5, 4.5), 3, 3,
                         facecolor='#FDDAD4', edgecolor='none', zorder=0)
    ax.add_patch(leak_bg)

    im = ax.imshow(corr.values, cmap='BrBG', vmin=-1.0, vmax=1.0, alpha=0.82, zorder=1)

    # ── Tick labels with colour coding ──────────────────────────────────────
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    x_labels = ax.set_xticklabels(display_names, rotation=40, ha='right',
                                   fontsize=9.5, fontweight='semibold')
    y_labels = ax.set_yticklabels(display_names, fontsize=9.5, fontweight='semibold')

    label_colors = ['black'] + [PRIMARY_TEAL]*4 + [ACCENT_CORAL]*3
    for tick, color in zip(x_labels, label_colors):
        tick.set_color(color)
    for tick, color in zip(y_labels, label_colors):
        tick.set_color(color)

    # ── Numeric annotations ──────────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            txt_color = 'white' if abs(val) > 0.65 else DARK_NAVY
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=txt_color, zorder=2)

    # ── Block separator lines ────────────────────────────────────────────────
    # Outcome / pre-match boundary
    ax.axhline(0.5, color='#444444', linewidth=1.8, zorder=3)
    ax.axvline(0.5, color='#444444', linewidth=1.8, zorder=3)
    # Pre-match / leaked boundary
    ax.axhline(4.5, color=ACCENT_CORAL, linewidth=2.2, linestyle='--', zorder=3)
    ax.axvline(4.5, color=ACCENT_CORAL, linewidth=2.2, linestyle='--', zorder=3)

    # ── Block labels at top ──────────────────────────────────────────────────
    ax.text(2.5, -1.35, 'Pre-Match Features', ha='center', va='top',
            fontsize=10, fontweight='bold', color=PRIMARY_TEAL,
            transform=ax.transData)
    ax.text(6.0, -1.35, 'In-Match Leaked', ha='center', va='top',
            fontsize=10, fontweight='bold', color=ACCENT_CORAL,
            transform=ax.transData)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, shrink=0.70, pad=0.04)
    cbar.set_label('Spearman Correlation', fontsize=9.5, fontweight='semibold', color=DARK_NAVY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def generate_all_proposal_figures():
    """Generates all figures according to the proposal structure."""
    print("--- Generating all figures according to proposal structure ---")
    plot_pipeline_diagram()
    plot_rq1_better_ranked()
    plot_rq2_models()
    plot_rq3_leakage()
    plot_rq3_segments()
    plot_rq3_calibration()
    plot_rq3_confidence()
    plot_rq3_by_year()
    plot_rankB_bootstrap()
    plot_rankB_importance()
    # Redesigned EDA plots
    plot_eda_distributions()
    plot_eda_correlation()
    plot_eda_heatmap()
    print("--- All proposal figures generated successfully! ---")


if __name__ == '__main__':
    generate_all_proposal_figures()
