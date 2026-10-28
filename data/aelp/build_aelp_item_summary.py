"""
Item-level summary of the AELP auditory lexical decision data (Goh et al., 2020)
for the US-accented talkers, following the trial filtering of the lexsemcat
project (lexsemcat24-wrk/R_scripts/stage1_7_prep_aelp_data.R).

Accuracy (word trials): after removing exact duplicate rows and the first two
trials of each 113-trial block (post-break RT spikes). Timeouts count as errors.

RT (correct word trials only), in addition:
    - offset RT (RT - duration) <= 10000 ms
    - drop trials with |z| > 3 on log(offset RT + pad), z computed over the pooled
      talkers, pad = 200 + |min offset RT|   (as in lexsemcat)

Output: one row per item, pooled over talkers and per talker (suffix _M2, _F2):
    n_trials, accuracy, n_rt, rt_onset (mean ms from word onset), rt_offset (mean
    ms from word offset), dur (ms). RTs are in ms; the log scale is used only for
    the outlier filter.

Usage, from the swr-bayes directory:
    python3 data/aelp/build_aelp_item_summary.py
    python3 data/aelp/build_aelp_item_summary.py --talkers M2 F2 M1 F1 --out other.csv
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd


def read_talker(path):
    with gzip.open(path, 'rt') as f:
        sep = '\t' if '\t' in f.readline() else ','   # F2 is comma-delimited, the rest tabs
    t = pd.read_csv(path, sep=sep, keep_default_na=False, na_values=['', ' '], low_memory=False)
    t['Item'] = t['Item'].astype(str).str.strip()
    return t


def summarize(df, rt, suffix=''):
    acc = df.groupby('Item').agg(n_trials=('RespACC', 'size'), accuracy=('RespACC', 'mean'),
                                 dur=('Dur', 'mean'))
    r = rt.groupby('Item').agg(n_rt=('RT', 'size'), rt_onset=('RT', 'mean'),
                               rt_offset=('RTmDur', 'mean'))
    out = acc.join(r, how='left')
    return out.add_suffix(suffix)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='data/aelp')
    ap.add_argument('--talkers', nargs='+', default=['M2', 'F2'])
    ap.add_argument('--out', default='data/aelp/aelp_item_summary_us.csv')
    args = ap.parse_args()

    trials = []
    for tk in args.talkers:
        t = read_talker(os.path.join(args.dir, f'{tk} ALDT Trial-level Data.csv.gz'))
        trials.append(t)
        print(f'{tk}: {len(t):,} trials')
    a = pd.concat(trials, ignore_index=True)

    a = a.drop_duplicates()
    a['trialrev'] = a['TrialNo'] % 113 + 1
    a = a[(a['ItemType'] == 1) & (a['trialrev'] > 2)].copy()
    a['RespACC'] = a['RespACC'].fillna(0)
    print(f'word trials after dedup and block-start removal: {len(a):,}')

    rt = a[(a['RespACC'] == 1) & (a['RTmDur'] <= 10000)].copy()
    pad = 200 + abs(rt['RTmDur'].min())
    lz = np.log(rt['RTmDur'] + pad)
    rt = rt[((lz - lz.mean()) / lz.std()).abs() <= 3].copy()
    print(f'correct word trials kept for RT: {len(rt):,} '
          f'({len(rt) / (a["RespACC"] == 1).sum():.1%} of correct trials)')

    parts = [summarize(a, rt)]
    for tk in args.talkers:
        parts.append(summarize(a[a['Talker'] == tk], rt[rt['Talker'] == tk], f'_{tk}'))
    out = pd.concat(parts, axis=1)
    out.index.name = 'item'
    out = out.reset_index().sort_values('item')
    out.round(4).to_csv(args.out, index=False)
    print(f'{len(out):,} items; mean accuracy {out.accuracy.mean():.3f}; '
          f'mean RT from onset {out.rt_onset.mean():.0f} ms, from offset {out.rt_offset.mean():.0f} ms')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
