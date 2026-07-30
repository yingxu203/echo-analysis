# -*- coding: utf-8 -*-
"""
analyze_study.py

Post-processes the raw echoanalysis_main.py / echoanalysis_mmode_main.py
output.csv files into the study's two comparison requirements:

  Requirement 1 (cross-sectional): within each family (WT: WT/WF/WM,
  OX: OX/OF/OM) and each day (0/7/14), compare treatments
  (VEH/PF/ISO/PF_ISO) against each other.

  Requirement 2 (longitudinal): within each family x treatment combo,
  track the same animal's metrics from day 0 -> 7 -> 14.

Every figure and summary-stats table is generated TWICE: once with all
points included, once with per-metric outliers excluded (1.5*IQR rule,
computed within each family/treatment/day cell - the same cell each
comparison is actually drawn from). Outputs go to
figures/ALL_POINTS_INCLUDING_OUTLIERS/ and figures/OUTLIERS_REMOVED/
respectively - if you're checking whether a specific bad point is gone,
look in OUTLIERS_REMOVED, not the other one.

USAGE:
    python analyze_study.py \
        --bmode_csv "<B-Mode>/output/output.csv" \
        --mmode_csv "<M-Mode>/output/output.csv" \
        --out_dir ./study_results
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from parse_metadata import get_metadata

TREATMENTS = ['VEH', 'PF', 'ISO', 'PF_ISO']
DAYS = [0, 7, 14]
FAMILIES = ['WT', 'OX']

# OX-family (OX/OF/OM) = red, WT-family (WT/WF/WM) = blue - matches the line
# color convention already used in the main tools' --verbose QC overlays.
FAMILY_COLORS = {'OX': '#FF2C2C', 'WT': '#0000FF'}

# Treatment is encoded as opacity within the family color: VEH is 75%
# transparent (faintest), PF 50%, ISO 25%, PF_ISO fully opaque (solid).
TREATMENT_ALPHA = {'VEH': 0.25, 'PF': 0.50, 'ISO': 0.75, 'PF_ISO': 1.00}

BMODE_METRICS = ['ejection_fraction', 'dia_volume (mm3)', 'sys_volume (mm3)']
MMODE_METRICS = ['FS (%)', 'LVID_dia (mm)', 'LV_Mass (mg)', 'Heart_Rate']


def opacity_caption():
    """Builds the figure-caption opacity legend from the CURRENT module-level
    TREATMENTS (not a fixed string), so it never mentions a treatment that
    --exclude_treatments has dropped from this run."""
    return ' → '.join(f"{t} {int(TREATMENT_ALPHA[t] * 100)}%" for t in TREATMENTS)


def sig_stars(p):
    """Standard significance convention: * p<0.05, ** p<0.01, *** p<0.001."""
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def req1_pvalue(day_df, metric, tx):
    """Welch's t-test: this treatment vs VEH, within the same family/day."""
    veh = day_df.loc[day_df['treatment'] == 'VEH', metric].dropna().values
    vals = day_df.loc[day_df['treatment'] == tx, metric].dropna().values
    if len(veh) < 2 or len(vals) < 2:
        return np.nan
    _, p = stats.ttest_ind(veh, vals, equal_var=False)
    return p


def req2_pvalue(tx_df, metric, day):
    """Paired t-test: this day vs Day 0, within the same family/treatment,
    restricted to animals with data at both timepoints."""
    piv = tx_df.pivot_table(index='subject_id', columns='day', values=metric)
    if 0 not in piv.columns or day not in piv.columns:
        return np.nan
    paired = piv[[0, day]].dropna()
    if len(paired) < 2:
        return np.nan
    _, p = stats.ttest_rel(paired[0], paired[day])
    return p


def ox_vs_wt_pvalue(df_slice, metric):
    """Welch's t-test: OX vs WT within a df already scoped to one
    treatment/day (or treatment, for the trend figures)."""
    wt = df_slice.loc[df_slice['family'] == 'WT', metric].dropna().values
    ox = df_slice.loc[df_slice['family'] == 'OX', metric].dropna().values
    if len(wt) < 2 or len(ox) < 2:
        return np.nan
    _, p = stats.ttest_ind(wt, ox, equal_var=False)
    return p


def get_combined_order_v1():
    """Version 1: grouped by family first, then treatment - WT block, then
    OX block. Computed from the CURRENT module-level TREATMENTS (not a fixed
    constant), so it reflects any --exclude_treatments filtering applied in
    main() before this is called."""
    return [(fam, tx) for fam in ['WT', 'OX'] for tx in TREATMENTS]


def get_combined_order_v2():
    """Version 2: VEH/PF/ISO grouped by family (WT block, OX block), but the
    PF_ISO combo treatment pulled out to the end so WT-PF_ISO and OX-PF_ISO
    sit directly adjacent for easy side-by-side comparison. Falls back to
    the same order as v1 if PF_ISO isn't in play (nothing to pull out)."""
    if 'PF_ISO' not in TREATMENTS:
        return get_combined_order_v1()
    others = [t for t in TREATMENTS if t != 'PF_ISO']
    return ([(fam, tx) for fam in ['WT', 'OX'] for tx in others] +
            [('WT', 'PF_ISO'), ('OX', 'PF_ISO')])


def load_and_annotate(csv_path, mode):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'animal_id': 'file_id'})

    metas = df['file_id'].apply(get_metadata)
    ok = metas.notna()
    if (~ok).any():
        print(f"  WARNING [{mode}]: {(~ok).sum()} file(s) failed metadata parsing, dropping:")
        print('   ', df.loc[~ok, 'file_id'].tolist())
    df = df.loc[ok].copy()
    meta_df = pd.DataFrame(list(metas[ok])).rename(columns={'animal_id': 'subject_id'})
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    df['mode'] = mode
    return df


def aggregate_per_animal_day(df, metric_cols):
    group_cols = ['subject_id', 'family', 'treatment', 'day']
    agg = df.groupby(group_cols, as_index=False)[metric_cols].mean()
    n_captures = df.groupby(group_cols, as_index=False).size().rename(columns={'size': 'n_captures'})
    agg = agg.merge(n_captures, on=group_cols)
    return agg


def compute_outlier_flags(agg, metric_cols):
    """Adds boolean 'outlier_<metric>' columns using the 1.5*IQR rule,
    computed within each (family, treatment, day) cell - the same cell
    each Req1 box and each Req2 point-at-a-day is drawn from, so a flagged
    point is consistent across every figure and stats table."""
    agg = agg.copy()
    for m in metric_cols:
        flag_col = f'outlier_{m}'
        agg[flag_col] = False
        for _, idx in agg.groupby(['family', 'treatment', 'day']).groups.items():
            vals = agg.loc[idx, m]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            agg.loc[idx, flag_col] = (vals < lo) | (vals > hi)
    return agg


def apply_outlier_mask(df, metric, exclude_outliers):
    """Returns df with rows flagged as outliers for this metric dropped,
    if exclude_outliers is set. A row outlier-flagged for one metric may
    still be kept for another metric's own filtering."""
    if not exclude_outliers:
        return df
    col = f'outlier_{metric}'
    if col not in df.columns:
        return df
    return df[~df[col].fillna(False)]


def outlier_summary(agg, metric_cols):
    """Flat list of every (animal, day, metric) point flagged as an
    outlier, for transparency/audit."""
    rows = []
    for m in metric_cols:
        flagged = agg[agg[f'outlier_{m}'].fillna(False)]
        for _, row in flagged.iterrows():
            rows.append({
                'subject_id': row['subject_id'], 'family': row['family'],
                'treatment': row['treatment'], 'day': row['day'],
                'metric': m, 'value': row[m],
            })
    return pd.DataFrame(rows)


def filter_min_heart_rate(agg, min_hr=450):
    """Drop animal-day rows with Heart_Rate below the threshold. A very low
    reading usually signals a poor-quality capture (anesthesia too deep, or
    bad cardiac-cycle detection) that would also compromise the other
    metrics computed from that same frame, so the whole row is dropped
    rather than just blanking Heart_Rate. No-op if there's no Heart_Rate
    column (i.e. B-mode)."""
    if 'Heart_Rate' not in agg.columns:
        return agg.copy()
    return agg[agg['Heart_Rate'] >= min_hr].copy()


def filter_complete_animals(agg):
    """Keep only animals (subject_id) with a row at all of Day 0, 7, and 14.
    Run this AFTER any row-dropping filter (like the heart-rate one) so an
    animal that lost a day to QC is correctly treated as incomplete too."""
    has_all_days = agg.groupby('subject_id')['day'].apply(lambda d: set(d) >= {0, 7, 14})
    keep_ids = has_all_days[has_all_days].index
    return agg[agg['subject_id'].isin(keep_ids)].copy()


def excluded_incomplete_report(agg):
    """Which animals got dropped by filter_complete_animals, and what days
    they actually had - for transparency."""
    has_all_days = agg.groupby('subject_id')['day'].apply(lambda d: set(d) >= {0, 7, 14})
    incomplete_ids = has_all_days[~has_all_days].index
    rows = []
    for subj in incomplete_ids:
        sdf = agg[agg['subject_id'] == subj]
        rows.append({
            'subject_id': subj, 'family': sdf['family'].iloc[0], 'treatment': sdf['treatment'].iloc[0],
            'days_present': ','.join(str(d) for d in sorted(sdf['day'].unique())),
        })
    return pd.DataFrame(rows)


def excluded_low_hr_report(df, min_hr=450):
    """Which rows got dropped by filter_min_heart_rate. Works on either the
    raw per-file df (includes file_id) or the aggregated per-animal-day df."""
    if 'Heart_Rate' not in df.columns:
        return pd.DataFrame()
    low = df[df['Heart_Rate'] < min_hr]
    cols = [c for c in ['file_id', 'subject_id', 'family', 'treatment', 'day', 'Heart_Rate'] if c in low.columns]
    return low[cols].copy()


def jitter(n, width=0.12):
    return np.random.default_rng(0).uniform(-width, width, n)


def plot_req1(agg, family, metrics, mode_label, out_path, exclude_outliers=False):
    fam_df = agg[agg['family'] == family]
    color = FAMILY_COLORS[family]
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.2 * n_rows), squeeze=False)

    for r, metric in enumerate(metrics):
        for c, day in enumerate(DAYS):
            ax = axes[r][c]
            day_df = apply_outlier_mask(fam_df[fam_df['day'] == day], metric, exclude_outliers)
            box_data = []
            positions = []
            for i, tx in enumerate(TREATMENTS):
                vals = day_df.loc[day_df['treatment'] == tx, metric].dropna().values
                box_data.append(vals)
                positions.append(i)
                if len(vals):
                    x = i + jitter(len(vals))
                    ax.scatter(x, vals, color=color, alpha=TREATMENT_ALPHA[tx], s=18,
                               zorder=3, edgecolors='none')

            bp = ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=False,
                             patch_artist=True, zorder=2)
            for patch, tx in zip(bp['boxes'], TREATMENTS):
                patch.set_facecolor(color)
                patch.set_alpha(TREATMENT_ALPHA[tx])

            # significance vs VEH, annotated above each non-VEH box
            non_empty = [v for v in box_data if len(v)]
            if non_empty:
                y_all_max = max(v.max() for v in non_empty)
                y_all_min = min(v.min() for v in non_empty)
                span = (y_all_max - y_all_min) or (abs(y_all_max) or 1)
                any_star = False
                for i, tx in enumerate(TREATMENTS):
                    if tx == 'VEH' or not len(box_data[i]):
                        continue
                    stars = sig_stars(req1_pvalue(day_df, metric, tx))
                    if stars:
                        any_star = True
                        ax.text(i, box_data[i].max() + 0.04 * span, stars, ha='center', va='bottom',
                                fontsize=13, fontweight='bold', color='black', zorder=4)
                if any_star:
                    ax.set_ylim(top=ax.get_ylim()[1] + 0.10 * span)

            ax.set_xticks(positions)
            ax.set_xticklabels(TREATMENTS, rotation=20, fontsize=8)
            if r == 0:
                ax.set_title(f"Day {day}", fontsize=11)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)

    outlier_note = 'outliers excluded (1.5xIQR per cell)' if exclude_outliers else 'all points included'
    fig.suptitle(f"{mode_label} - {family} family - treatment comparison by day ({outlier_note})\n"
                 f"opacity: {opacity_caption()}   "
                 f"|   * p<0.05  ** p<0.01  *** p<0.001 vs VEH", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_req2(agg, family, metrics, mode_label, out_path, exclude_outliers=False):
    fam_df = agg[agg['family'] == family]
    color = FAMILY_COLORS[family]
    n_rows = len(metrics)
    n_cols = len(TREATMENTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.75 * n_cols, 3.2 * n_rows), squeeze=False)

    for r, metric in enumerate(metrics):
        for c, tx in enumerate(TREATMENTS):
            ax = axes[r][c]
            tx_df = apply_outlier_mask(fam_df[fam_df['treatment'] == tx], metric, exclude_outliers)
            alpha = TREATMENT_ALPHA[tx]

            for subj, sdf in tx_df.groupby('subject_id'):
                sdf = sdf.sort_values('day')
                ax.plot(sdf['day'], sdf[metric], '-o', color=color,
                        alpha=alpha, linewidth=1, markersize=3, zorder=2)

            means = tx_df.groupby('day')[metric].agg(['mean', 'sem']).reindex(DAYS)
            # mean ± SEM overlay always drawn fully opaque so it stays legible
            # even for the faint (VEH) treatment opacity
            ax.errorbar(means.index, means['mean'], yerr=means['sem'], color=color,
                        linewidth=2.4, marker='o', markersize=7, capsize=4, zorder=3,
                        alpha=1.0, markeredgecolor='black', markeredgewidth=0.6, label='mean ± SEM')

            # significance vs Day 0 (paired, same animals)
            valid_means = means['mean'].dropna()
            if len(valid_means):
                span = (valid_means.max() - valid_means.min()) or (abs(valid_means.max()) or 1)
                for day in [7, 14]:
                    if day not in means.index or np.isnan(means.loc[day, 'mean']):
                        continue
                    stars = sig_stars(req2_pvalue(tx_df, metric, day))
                    if stars:
                        sem_val = means.loc[day, 'sem'] if not np.isnan(means.loc[day, 'sem']) else 0
                        y = means.loc[day, 'mean'] + sem_val + 0.08 * span
                        ax.text(day, y, stars, ha='center', va='bottom', fontsize=13,
                                fontweight='bold', color='black', zorder=4)

            ax.set_xticks(DAYS)
            if r == 0:
                ax.set_title(f"{family} {tx}", fontsize=11)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('Day', fontsize=9)

    outlier_note = 'outliers excluded (1.5xIQR per cell)' if exclude_outliers else 'all points included'
    fig.suptitle(f"{mode_label} - {family} family - individual animal trends, day0→7→14 ({outlier_note})\n"
                 f"opacity: {opacity_caption()}   "
                 f"|   * p<0.05  ** p<0.01  *** p<0.001 vs Day 0", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_req1_combined(agg, metrics, mode_label, order, order_label, out_path, exclude_outliers=False):
    """Req1 with WT and OX on the same axes: 8 boxes per subplot (family x
    treatment), colored by family, shaded by treatment opacity. Significance
    is OX_tx vs WT_tx (the matching treatment in the other family), starred
    above the OX box."""
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.4 * n_rows), squeeze=False)

    for r, metric in enumerate(metrics):
        for c, day in enumerate(DAYS):
            ax = axes[r][c]
            day_df = apply_outlier_mask(agg[agg['day'] == day], metric, exclude_outliers)
            box_data = []
            for fam, tx in order:
                vals = day_df.loc[(day_df['family'] == fam) & (day_df['treatment'] == tx), metric].dropna().values
                box_data.append(vals)

            positions = list(range(len(order)))
            for i, (fam, tx) in enumerate(order):
                vals = box_data[i]
                if len(vals):
                    x = i + jitter(len(vals))
                    ax.scatter(x, vals, color=FAMILY_COLORS[fam], alpha=TREATMENT_ALPHA[tx],
                               s=16, zorder=3, edgecolors='none')

            bp = ax.boxplot(box_data, positions=positions, widths=0.6, showfliers=False,
                             patch_artist=True, zorder=2)
            for patch, (fam, tx) in zip(bp['boxes'], order):
                patch.set_facecolor(FAMILY_COLORS[fam])
                patch.set_alpha(TREATMENT_ALPHA[tx])

            # significance: OX_tx vs WT_tx, starred above the OX box
            non_empty = [v for v in box_data if len(v)]
            if non_empty:
                y_max = max(v.max() for v in non_empty)
                y_min = min(v.min() for v in non_empty)
                span = (y_max - y_min) or (abs(y_max) or 1)
                any_star = False
                for i, (fam, tx) in enumerate(order):
                    if fam != 'OX' or not len(box_data[i]):
                        continue
                    sub = day_df[day_df['treatment'] == tx]
                    stars = sig_stars(ox_vs_wt_pvalue(sub, metric))
                    if stars:
                        any_star = True
                        ax.text(i, box_data[i].max() + 0.04 * span, stars, ha='center', va='bottom',
                                fontsize=13, fontweight='bold', color='black', zorder=4)
                if any_star:
                    ax.set_ylim(top=ax.get_ylim()[1] + 0.10 * span)

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{fam}\n{tx}" for fam, tx in order], fontsize=7.5)
            if r == 0:
                ax.set_title(f"Day {day}", fontsize=11)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)

    outlier_note = 'outliers excluded' if exclude_outliers else 'all points included'
    fig.suptitle(f"{mode_label} - WT vs OX combined ({order_label}, {outlier_note})\n"
                 f"blue=WT, red=OX   |   opacity: {opacity_caption()}   "
                 f"|   * p<0.05  ** p<0.01  *** p<0.001, OX vs matching WT treatment", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_req2_combined(agg, metrics, mode_label, out_path, exclude_outliers=False):
    """Req2 with WT and OX overlaid in the same subplot per treatment.
    Significance is OX vs WT at each day, starred above the higher mean."""
    n_rows = len(metrics)
    n_cols = len(TREATMENTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.75 * n_cols, 3.2 * n_rows), squeeze=False)

    for r, metric in enumerate(metrics):
        for c, tx in enumerate(TREATMENTS):
            ax = axes[r][c]
            tx_df = apply_outlier_mask(agg[agg['treatment'] == tx], metric, exclude_outliers)
            alpha = TREATMENT_ALPHA[tx]

            for fam in ['WT', 'OX']:
                fam_df = tx_df[tx_df['family'] == fam]
                color = FAMILY_COLORS[fam]
                for subj, sdf in fam_df.groupby('subject_id'):
                    sdf = sdf.sort_values('day')
                    ax.plot(sdf['day'], sdf[metric], '-o', color=color,
                            alpha=alpha, linewidth=1, markersize=3, zorder=2)

                means = fam_df.groupby('day')[metric].agg(['mean', 'sem']).reindex(DAYS)
                ax.errorbar(means.index, means['mean'], yerr=means['sem'], color=color,
                            linewidth=2.4, marker='o', markersize=7, capsize=4, zorder=3,
                            alpha=1.0, markeredgecolor='black', markeredgewidth=0.6,
                            label=f'{fam} mean ± SEM')

            # significance: OX vs WT at each day
            all_vals = tx_df[metric].dropna()
            if len(all_vals):
                span = (all_vals.max() - all_vals.min()) or (abs(all_vals.max()) or 1)
                for day in DAYS:
                    sub = tx_df[tx_df['day'] == day]
                    stars = sig_stars(ox_vs_wt_pvalue(sub, metric))
                    if stars:
                        wt_mean = sub.loc[sub['family'] == 'WT', metric].mean()
                        ox_mean = sub.loc[sub['family'] == 'OX', metric].mean()
                        y_ref = np.nanmax([wt_mean, ox_mean])
                        ax.text(day, y_ref + 0.08 * span, stars, ha='center', va='bottom',
                                fontsize=13, fontweight='bold', color='black', zorder=4)

            ax.set_xticks(DAYS)
            if r == 0:
                ax.set_title(f"{tx}", fontsize=11)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('Day', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='best')

    outlier_note = 'outliers excluded' if exclude_outliers else 'all points included'
    fig.suptitle(f"{mode_label} - WT vs OX combined, individual animal trends, day0→7→14 ({outlier_note})\n"
                 f"blue=WT, red=OX   |   opacity: {opacity_caption()}   "
                 f"|   * p<0.05  ** p<0.01  *** p<0.001, OX vs WT at that day", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def oxvswt_stats(agg, metric_cols, exclude_outliers=False):
    """OX vs WT comparison at each treatment x day (the same test used to
    star the combined req1/req2 figures), as a standalone numeric table."""
    rows = []
    for tx in TREATMENTS:
        for day in DAYS:
            base = agg[(agg['treatment'] == tx) & (agg['day'] == day)]
            row = {'treatment': tx, 'day': day}
            for m in metric_cols:
                sub = apply_outlier_mask(base, m, exclude_outliers)
                row[f'n_WT_{m}'] = len(sub[sub['family'] == 'WT'][m].dropna())
                row[f'n_OX_{m}'] = len(sub[sub['family'] == 'OX'][m].dropna())
                row[f'{m}_mean_WT'] = sub.loc[sub['family'] == 'WT', m].dropna().mean()
                row[f'{m}_mean_OX'] = sub.loc[sub['family'] == 'OX', m].dropna().mean()
                row[f'{m}_p_OXvsWT'] = ox_vs_wt_pvalue(sub, m)
            rows.append(row)
    return pd.DataFrame(rows)


def summary_stats(agg, metric_cols, exclude_outliers=False):
    """Req1 cross-sectional stats: mean/sd/sem per family/day/treatment,
    plus a Welch's t-test p-value vs VEH for each metric (NaN for VEH itself)."""
    rows = []
    for family in FAMILIES:
        for day in DAYS:
            base = agg[(agg['family'] == family) & (agg['day'] == day)]
            for tx in TREATMENTS:
                row = {'family': family, 'day': day, 'treatment': tx}
                for m in metric_cols:
                    day_df = apply_outlier_mask(base, m, exclude_outliers)
                    vals = day_df.loc[day_df['treatment'] == tx, m].dropna()
                    row['n'] = len(vals)
                    row[f'{m}_mean'] = vals.mean()
                    row[f'{m}_sd'] = vals.std()
                    row[f'{m}_sem'] = vals.sem()
                    row[f'{m}_p_vs_VEH'] = np.nan if tx == 'VEH' else req1_pvalue(day_df, m, tx)
                rows.append(row)
    return pd.DataFrame(rows)


def req2_summary_stats(agg, metric_cols, exclude_outliers=False):
    """Req2 longitudinal stats: mean/sem per family/treatment/day, plus a
    paired t-test p-value vs Day 0 for each metric (NaN for Day 0 itself)."""
    rows = []
    for family in FAMILIES:
        for tx in TREATMENTS:
            base = agg[(agg['family'] == family) & (agg['treatment'] == tx)]
            for day in DAYS:
                row = {'family': family, 'treatment': tx, 'day': day}
                for m in metric_cols:
                    tx_df = apply_outlier_mask(base, m, exclude_outliers)
                    vals = tx_df.loc[tx_df['day'] == day, m].dropna()
                    row['n'] = len(vals)
                    row[f'{m}_mean'] = vals.mean()
                    row[f'{m}_sem'] = vals.sem()
                    row[f'{m}_p_vs_Day0'] = np.nan if day == 0 else req2_pvalue(tx_df, m, day)
                rows.append(row)
    return pd.DataFrame(rows)


def generate_all_figures(bmode_agg, mmode_agg, fig_dir, exclude_outliers):
    sub_dir = os.path.join(fig_dir, 'OUTLIERS_REMOVED' if exclude_outliers else 'ALL_POINTS_INCLUDING_OUTLIERS')
    os.makedirs(sub_dir, exist_ok=True)

    combined_v1 = get_combined_order_v1()
    combined_v2 = get_combined_order_v2()

    for family in FAMILIES:
        plot_req1(bmode_agg, family, BMODE_METRICS, 'B-mode',
                  os.path.join(sub_dir, f'req1_bmode_{family}.png'), exclude_outliers)
        plot_req1(mmode_agg, family, MMODE_METRICS, 'M-mode',
                  os.path.join(sub_dir, f'req1_mmode_{family}.png'), exclude_outliers)
        plot_req2(bmode_agg, family, BMODE_METRICS, 'B-mode',
                  os.path.join(sub_dir, f'req2_bmode_{family}.png'), exclude_outliers)
        plot_req2(mmode_agg, family, MMODE_METRICS, 'M-mode',
                  os.path.join(sub_dir, f'req2_mmode_{family}.png'), exclude_outliers)

    plot_req1_combined(bmode_agg, BMODE_METRICS, 'B-mode', combined_v1, 'version 1: family-grouped',
                        os.path.join(sub_dir, 'req1_combined_v1_bmode.png'), exclude_outliers)
    plot_req1_combined(bmode_agg, BMODE_METRICS, 'B-mode', combined_v2, 'version 2: PF_ISO paired',
                        os.path.join(sub_dir, 'req1_combined_v2_bmode.png'), exclude_outliers)
    plot_req1_combined(mmode_agg, MMODE_METRICS, 'M-mode', combined_v1, 'version 1: family-grouped',
                        os.path.join(sub_dir, 'req1_combined_v1_mmode.png'), exclude_outliers)
    plot_req1_combined(mmode_agg, MMODE_METRICS, 'M-mode', combined_v2, 'version 2: PF_ISO paired',
                        os.path.join(sub_dir, 'req1_combined_v2_mmode.png'), exclude_outliers)

    plot_req2_combined(bmode_agg, BMODE_METRICS, 'B-mode',
                        os.path.join(sub_dir, 'req2_combined_bmode.png'), exclude_outliers)
    plot_req2_combined(mmode_agg, MMODE_METRICS, 'M-mode',
                        os.path.join(sub_dir, 'req2_combined_mmode.png'), exclude_outliers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bmode_csv', required=True)
    parser.add_argument('--mmode_csv', required=True)
    parser.add_argument('--out_dir', default='./study_results',
                         help="Base directory - each run creates its own timestamped, "
                              "self-labeled subfolder here rather than overwriting the last run")
    parser.add_argument('--min_heart_rate', type=float, default=400,
                         help="Drop M-mode animal-day captures with Heart_Rate below this")
    parser.add_argument('--bmode_qc_pdf', nargs='+', default=[],
                         help="One or more B-mode QC overlay PDF(s) (output_images.pdf) to copy "
                              "into this run's folder for reference, e.g. the full-dataset PDF "
                              "plus any subset-correction PDF")
    parser.add_argument('--mmode_qc_pdf', nargs='+', default=[],
                         help="Same as --bmode_qc_pdf, for M-mode")
    parser.add_argument('--exclude_treatments', default=None,
                         help="Comma-separated treatment names to drop entirely from this run "
                              "(e.g. 'PF_ISO'). Affects every table and figure - the excluded "
                              "treatment's data never enters the clean base at all.")
    args = parser.parse_args()

    excluded_treatments = set()
    if args.exclude_treatments:
        excluded_treatments = set(t.strip() for t in args.exclude_treatments.split(','))
        # mutate in place (not rebind) so every function referencing the
        # module-level TREATMENTS global sees the same filtered list
        TREATMENTS[:] = [t for t in TREATMENTS if t not in excluded_treatments]
        print(f"Excluding treatments: {sorted(excluded_treatments)}. Remaining: {TREATMENTS}\n")

    import datetime
    import shutil
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    treat_suffix = ('_excl' + '-'.join(sorted(excluded_treatments))) if excluded_treatments else ''
    run_name = f"run_{timestamp}_HRge{args.min_heart_rate:g}_completeDaysOnly{treat_suffix}"
    run_dir = os.path.join(args.out_dir, run_name)
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f"This run's output folder: {run_dir}\n")

    if args.bmode_qc_pdf or args.mmode_qc_pdf:
        qc_dir = os.path.join(run_dir, 'qc_pdfs')
        os.makedirs(qc_dir, exist_ok=True)
        for mode_label, pdf_list in [('Bmode', args.bmode_qc_pdf), ('Mmode', args.mmode_qc_pdf)]:
            for i, src in enumerate(pdf_list):
                # label each copy with its parent run folder so multiple PDFs
                # (e.g. full-dataset + a subset correction) don't collide/overwrite
                parent_label = os.path.basename(os.path.dirname(src)) or f'{i}'
                dst = os.path.join(qc_dir, f'{mode_label}_{parent_label}_output_images.pdf')
                shutil.copy2(src, dst)
                print(f"Copied QC PDF: {src} -> {dst}")

    print("Loading + annotating B-mode results (full, unanalyzed raw data)...")
    bmode_df = load_and_annotate(args.bmode_csv, 'bmode')
    print("Loading + annotating M-mode results (full, unanalyzed raw data)...")
    mmode_df = load_and_annotate(args.mmode_csv, 'mmode')

    if excluded_treatments:
        b_before, m_before = len(bmode_df), len(mmode_df)
        bmode_df = bmode_df[~bmode_df['treatment'].isin(excluded_treatments)].copy()
        mmode_df = mmode_df[~mmode_df['treatment'].isin(excluded_treatments)].copy()
        print(f"Dropped excluded-treatment rows: B-mode {b_before}->{len(bmode_df)}, "
              f"M-mode {m_before}->{len(mmode_df)}\n")

    bmode_agg_full = aggregate_per_animal_day(bmode_df, BMODE_METRICS)
    mmode_agg_full = aggregate_per_animal_day(mmode_df, MMODE_METRICS)

    print(f"B-mode raw: {len(bmode_df)} files -> {len(bmode_agg_full)} animal-day points "
          f"({bmode_agg_full['subject_id'].nunique()} unique animals)")
    print(f"M-mode raw: {len(mmode_df)} files -> {len(mmode_agg_full)} animal-day points "
          f"({mmode_agg_full['subject_id'].nunique()} unique animals)")

    # Build the clean BASE population that every downstream result draws
    # from: starting from the full raw per-file data, (1) drop any M-mode
    # capture with Heart_Rate below threshold - done on RAW per-file rows,
    # BEFORE replicate captures get averaged into an animal-day value, so a
    # bad capture (e.g. HR=380) can't get diluted by a good one in the same
    # animal-day (HR=550 -> avg 465) and silently pass; (2) keep only
    # animals with a row at all of Day 0, 7, and 14 - checked AFTER the HR
    # drop, since losing a day to the HR filter can itself make an animal
    # incomplete. The with/without-outliers split happens only after this,
    # as the last step, purely via the 1.5xIQR rule on top of this same
    # clean base - it is no longer a separate raw-vs-filtered distinction.
    print(f"\nBuilding clean base: M-mode Heart_Rate >= {args.min_heart_rate} "
          f"(pre-aggregation), then complete Day0/7/14 animals only...")
    mmode_df_hr_ok = filter_min_heart_rate(mmode_df, args.min_heart_rate)
    mmode_agg_hr_ok = aggregate_per_animal_day(mmode_df_hr_ok, MMODE_METRICS)

    bmode_agg = filter_complete_animals(bmode_agg_full)
    mmode_agg = filter_complete_animals(mmode_agg_hr_ok)

    bmode_excluded_incomplete = excluded_incomplete_report(bmode_agg_full)
    mmode_excluded_incomplete = excluded_incomplete_report(mmode_agg_hr_ok)
    mmode_excluded_lowhr = excluded_low_hr_report(mmode_df, args.min_heart_rate)

    print(f"  B-mode clean base: {bmode_agg['subject_id'].nunique()} animals "
          f"(dropped {bmode_agg_full['subject_id'].nunique() - bmode_agg['subject_id'].nunique()} incomplete)")
    print(f"  M-mode clean base: {mmode_agg['subject_id'].nunique()} animals, {len(mmode_agg)} rows "
          f"(dropped {len(mmode_excluded_lowhr)} rows for low HR, "
          f"{mmode_agg_full['subject_id'].nunique() - mmode_agg['subject_id'].nunique()} animals now incomplete)")

    print("\nGenerating figures WITH outliers included (clean base, no IQR exclusion)...")
    generate_all_figures(bmode_agg, mmode_agg, fig_dir, exclude_outliers=False)

    print("Flagging outliers within the clean base (1.5xIQR per family/treatment/day cell, per metric)...")
    bmode_clean = compute_outlier_flags(bmode_agg, BMODE_METRICS)
    mmode_clean = compute_outlier_flags(mmode_agg, MMODE_METRICS)
    bmode_outliers = outlier_summary(bmode_clean, BMODE_METRICS)
    mmode_outliers = outlier_summary(mmode_clean, MMODE_METRICS)
    print(f"  B-mode: {len(bmode_outliers)} (animal, day, metric) points flagged")
    print(f"  M-mode: {len(mmode_outliers)} (animal, day, metric) points flagged")

    print("Generating figures WITH outliers excluded (clean base, IQR outliers removed)...")
    generate_all_figures(bmode_clean, mmode_clean, fig_dir, exclude_outliers=True)

    print("Computing summary statistics (both versions)...")
    bmode_summary = summary_stats(bmode_agg, BMODE_METRICS, False)
    mmode_summary = summary_stats(mmode_agg, MMODE_METRICS, False)
    bmode_summary_no = summary_stats(bmode_clean, BMODE_METRICS, True)
    mmode_summary_no = summary_stats(mmode_clean, MMODE_METRICS, True)

    bmode_req2_summary = req2_summary_stats(bmode_agg, BMODE_METRICS, False)
    mmode_req2_summary = req2_summary_stats(mmode_agg, MMODE_METRICS, False)
    bmode_req2_summary_no = req2_summary_stats(bmode_clean, BMODE_METRICS, True)
    mmode_req2_summary_no = req2_summary_stats(mmode_clean, MMODE_METRICS, True)

    bmode_oxvswt = oxvswt_stats(bmode_agg, BMODE_METRICS, False)
    mmode_oxvswt = oxvswt_stats(mmode_agg, MMODE_METRICS, False)
    bmode_oxvswt_no = oxvswt_stats(bmode_clean, BMODE_METRICS, True)
    mmode_oxvswt_no = oxvswt_stats(mmode_clean, MMODE_METRICS, True)

    # Per-animal trajectory pivot tables (req2 raw numbers, clean base), one per primary metric
    bmode_traj = bmode_agg.pivot_table(index=['family', 'treatment', 'subject_id'],
                                        columns='day', values='ejection_fraction').reset_index()
    mmode_traj = mmode_agg.pivot_table(index=['family', 'treatment', 'subject_id'],
                                        columns='day', values='FS (%)').reset_index()

    xlsx_path = os.path.join(run_dir, 'study_results.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        bmode_df.to_excel(writer, sheet_name='Bmode_per_file', index=False)
        bmode_agg.to_excel(writer, sheet_name='Bmode_per_animal_day', index=False)
        bmode_summary.to_excel(writer, sheet_name='Bmode_summary_by_day', index=False)
        bmode_summary_no.to_excel(writer, sheet_name='Bmode_summary_by_day_noOut', index=False)
        bmode_req2_summary.to_excel(writer, sheet_name='Bmode_summary_by_treat', index=False)
        bmode_req2_summary_no.to_excel(writer, sheet_name='Bmode_summary_by_treat_noOut', index=False)
        bmode_oxvswt.to_excel(writer, sheet_name='Bmode_OXvsWT_stats', index=False)
        bmode_oxvswt_no.to_excel(writer, sheet_name='Bmode_OXvsWT_stats_noOut', index=False)
        bmode_outliers.to_excel(writer, sheet_name='Bmode_outliers_flagged', index=False)
        bmode_excluded_incomplete.to_excel(writer, sheet_name='Bmode_excluded_incomplete', index=False)
        bmode_traj.to_excel(writer, sheet_name='Bmode_EF_trajectories', index=False)

        mmode_df.to_excel(writer, sheet_name='Mmode_per_file', index=False)
        mmode_agg.to_excel(writer, sheet_name='Mmode_per_animal_day', index=False)
        mmode_summary.to_excel(writer, sheet_name='Mmode_summary_by_day', index=False)
        mmode_summary_no.to_excel(writer, sheet_name='Mmode_summary_by_day_noOut', index=False)
        mmode_req2_summary.to_excel(writer, sheet_name='Mmode_summary_by_treat', index=False)
        mmode_req2_summary_no.to_excel(writer, sheet_name='Mmode_summary_by_treat_noOut', index=False)
        mmode_oxvswt.to_excel(writer, sheet_name='Mmode_OXvsWT_stats', index=False)
        mmode_oxvswt_no.to_excel(writer, sheet_name='Mmode_OXvsWT_stats_noOut', index=False)
        mmode_outliers.to_excel(writer, sheet_name='Mmode_outliers_flagged', index=False)
        mmode_excluded_incomplete.to_excel(writer, sheet_name='Mmode_excluded_incomplete', index=False)
        mmode_excluded_lowhr.to_excel(writer, sheet_name='Mmode_excluded_lowHR', index=False)
        mmode_traj.to_excel(writer, sheet_name='Mmode_FS_trajectories', index=False)

    print(f"\nSaved: {xlsx_path}")
    print(f"Saved figures to: {fig_dir}/ALL_POINTS_INCLUDING_OUTLIERS/ (clean base: complete Day0/7/14, "
          f"HR>={args.min_heart_rate:g}, but statistical outliers still shown) "
          f"and {fig_dir}/OUTLIERS_REMOVED/ (same clean base, plus 1.5xIQR outliers dropped - use THIS one to check a bad point is gone)")
    print(f"\nFull run output: {run_dir}")


if __name__ == '__main__':
    main()
