"""
Phoneme distances from Cutler, Weber, Smits & Cooper (2004), American English
listeners, and a comparison with the Luce (1986) distances the notebook uses.
Not used by the notebook; kept as an independent check (see README.md).

Same method as ../build_confusion_distances.py: Shepard (1957) distances,
d_ij = -ln sqrt((n_ij / n_ii) (n_ji / n_jj)), with 0.5 added to every count,
computed separately for CV (initial) and VC (final) syllables at one SNR.
'No response' trials (23miss) are dropped.

Usage, from the swr-bayes directory:
    python3 data/confusion-tables/cutler2004/cutler_distances.py
    python3 data/confusion-tables/cutler2004/cutler_distances.py --snr 8 --out cutler_8dB.csv

--out writes a combined 38-phoneme matrix in the format of
data/phoneme_confusion_distances.csv (CV and VC averaged; untested pairs,
including all consonant-vowel pairs, set to the largest distance measured).
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONSONANTS = {'01p': 'P', '02b': 'B', '03t': 'T', '04d': 'D', '05k': 'K', '06g': 'G', '07f': 'F',
              '08v': 'V', '09th': 'TH', '10dth': 'DH', '11s': 'S', '12z': 'Z', '13sh': 'SH',
              '14h': 'HH', '14j': 'ZH', '15ch': 'CH', '16dj': 'JH', '17m': 'M', '18n': 'N',
              '19ng': 'NG', '19y': 'Y', '20l': 'L', '21r': 'R', '22w': 'W'}
# The codes follow a long/short pattern (ii = beat, i = bit; uu = boot, u = book), so
# oo = AO (caught) and o = AA (cot). The data alone cannot decide this one: both
# mappings agree equally well with Luce's vowels.
VOWELS = {'01i': 'IH', '02ii': 'IY', '03ei': 'EY', '04ai': 'AY', '05oi': 'OY', '06e': 'EH',
          '07a': 'AE', '08sr': 'ER', '09u': 'UH', '10uu': 'UW', '11au': 'AW', '12ou': 'OW',
          '13oo': 'AO', '14uh': 'AH', '15o': 'AA'}
SMOOTH = 0.5


def shepard_distance(counts):
    n = counts.values + SMOOTH
    ratio = n / np.diag(n)[:, None]
    d = -0.5 * np.log(ratio * ratio.T)
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=counts.index, columns=counts.index)


def distances(path, labels, snr):
    """Shepard distances for CV and VC syllables at one SNR; also returns P(correct) by stimulus."""
    df = pd.read_csv(path)
    df = df[(df['signalnoise'] == snr) & (df['answer'] != '23miss')]
    df = df.assign(stim=df['stimulus'].map(labels), resp=df['answer'].map(labels))
    out = {}
    for pos in ['cv', 'vc']:
        sub = df[df['init/final'] == pos]
        stim = sorted(sub['stim'].unique())
        counts = pd.crosstab(sub['stim'], sub['resp']).reindex(index=stim, columns=stim, fill_value=0)
        acc = pd.Series(np.diag(counts) / counts.sum(axis=1).values, index=stim)
        out[pos] = (shepard_distance(counts), acc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snr', type=int, default=16, choices=[0, 8, 16], help='signal-to-noise ratio (dB)')
    ap.add_argument('--luce', default='data/phoneme_confusion_distances.csv')
    ap.add_argument('--lexicon', default='data/lemmalex.csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tables = {}
    for kind, labels in [('cons', CONSONANTS), ('vowel', VOWELS)]:
        for pos, (d, acc) in distances(os.path.join(HERE, f'cutler_AE_{kind}_trials.csv'), labels, args.snr).items():
            tables[f'{kind}-{pos}'] = d
            print(f"{kind} {pos.upper()} at {args.snr} dB: {len(d)} phonemes, mean P(correct) {acc.mean():.2f}"
                  + (f", /h/ {acc['HH']:.2f}" if 'HH' in acc else ''))

    phonemes = sorted(set(' '.join(pd.read_csv(args.lexicon, keep_default_na=False,
                                                na_values=[''])['Pronunciation']).split()))
    stacked = np.stack([d.reindex(index=phonemes, columns=phonemes).values for d in tables.values()])
    n_estimates = np.sum(~np.isnan(stacked), axis=0)
    fill = max(d.values.max() for d in tables.values())
    values = np.where(n_estimates > 0, np.nansum(stacked, axis=0) / np.maximum(n_estimates, 1), fill)
    np.fill_diagonal(values, 0.0)
    combined = pd.DataFrame(values, index=phonemes, columns=phonemes)
    combined.index.name = 'phoneme'

    # Where does P-HH rank among the consonant pairs?
    cons = [p for p in phonemes if p in CONSONANTS.values()]
    upper = np.triu_indices(len(cons), k=1)
    pairs = combined.loc[cons, cons].values[upper]
    tested = n_estimates[np.ix_([phonemes.index(p) for p in cons], [phonemes.index(p) for p in cons])][upper] > 0
    rank = int((pairs[tested] < combined.loc['P', 'HH']).sum()) + 1
    print(f"\nP-HH distance {combined.loc['P', 'HH']:.2f}: rank {rank} of {tested.sum()} tested consonant pairs"
          f" (closest to HH: {combined.loc['HH'].drop('HH').sort_values().head(3).round(2).to_dict()})")

    # Agreement with the Luce distances, over pairs tested in both
    luce = pd.read_csv(args.luce, index_col='phoneme').loc[phonemes, phonemes]
    luce_fill = luce.values.max()
    for name, group in [('consonants', cons), ('vowels', [p for p in phonemes if p not in cons])]:
        up = np.triu_indices(len(group), k=1)
        x = pd.DataFrame({'cutler': combined.loc[group, group].values[up],
                          'luce': luce.loc[group, group].values[up]})
        x = x[(x['cutler'] < fill) & (x['luce'] < luce_fill)]
        print(f"Agreement with {args.luce}, {name}: {len(x)} pairs, "
              f"Spearman rho = {x.corr('spearman').iloc[0, 1]:.2f}")

    if args.out:
        combined.round(4).to_csv(args.out)
        print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
