"""
Build a phoneme-to-phoneme distance matrix from human confusion data,
for the Bayesian word recognition notebook (07-swr-bayes).

Source: Luce (1986, dissertation), Tables 3.3-3.11 (TablesForJim_LucePhD.xlsx).
Listeners identified phonemes in noise at three signal-to-noise ratios. We use
only the CLEAREST condition of each set, as the best available guide to clear
speech:
    Table 3.3  initial consonants (22, incl. HH W Y)
    Table 3.6  vowels (15, incl. ER, which the lexicon does not use)
    Table 3.9  final consonants (21, incl. NG ZH)

Method (Shepard, 1957; Luce's biased-choice model):
    s_ij = sqrt( (n_ij / n_ii) * (n_ji / n_jj) ),   d_ij = -ln s_ij
where n_ij = number of times stimulus i was reported as j (+ 0.5 smoothing).
Each ratio uses counts from a single stimulus row, so the row totals cancel
(several rows in the source tables do not sum to the stated number of trials)
and so does any overall response bias toward particular phonemes. Under the
biased-choice model, P(j | i) is proportional to bias_j * exp(-d_ij), which is
the exp(-c * d) form the notebook uses (with c = 1 at Luce's noise level).

Combining:
    - consonant pairs measured in both initial and final position: mean of the
      two distances; pairs measured in only one position: that distance
    - pairs never tested together (consonant-vowel; HH/W/Y vs NG/ZH): set to
      the largest distance measured in any table, i.e., at least as distant as
      the least confusable pair that was actually tested

One adjustment (skip it with --no-adjust):
    - /p/-/h/: in Table 3.3, /h/ was reported as /p/ more often (49 of 113 trials)
      than as /h/ (30), which made P-HH by far the closest pair (1.05). In word
      recognition, this one pair caused 28 of the 177 errors at the notebook's
      clarity target (.95), e.g., pit -> hit, hay -> pay. The confusion is real
      in noise (Cutler et al., 2004, also put P-HH among the closest consonant
      pairs; see cutler2004/), but /h/ is far more identifiable in clearer speech
      (84% correct at Cutler's 16 dB SNR vs. 27% here). Since we use these data to
      stand in for clear speech, P-HH is set to the median distance among the
      consonant pairs that were tested, i.e., treated as an ordinary consonant pair.

Peterson & Barney (1952) vowels in quiet are used only as a check (printed).

Usage, from the swr-bayes directory:
    python3 data/confusion-tables/build_confusion_distances.py
    python3 data/confusion-tables/build_confusion_distances.py --out some/other.csv
    python3 data/confusion-tables/build_confusion_distances.py --no-adjust --out unadjusted.csv
"""
import argparse
import warnings
from itertools import combinations

import numpy as np
import openpyxl
import pandas as pd

warnings.filterwarnings('ignore', message='Unknown extension')

CONSONANTS = dict(p='P', t='T', k='K', b='B', d='D', g='G', C='CH', J='JH', s='S', S='SH',
                  z='Z', f='F', T='TH', v='V', D='DH', h='HH', n='N', m='M', l='L', r='R',
                  w='W', y='Y', G='NG', Z='ZH')
VOWELS = {'ii': 'IY', 'I': 'IH', 'E': 'EH', 'e': 'EY', '@': 'AE', 'a': 'AA', 'W': 'AW',
          'Y': 'AY', '^': 'AH', 'c': 'AO', 'O': 'OY', 'o': 'OW', 'U': 'UH', 'u': 'UW',
          'R': 'ER'}
TABLES = {'initial': 'Table 3.3', 'vowel': 'Table 3.6', 'final': 'Table 3.9'}
SMOOTH = 0.5
ADJUSTED_PAIRS = [('P', 'HH')]          # set to the median tested consonant distance


def read_table(wb, sheet):
    """Return the stimulus-by-response count matrix (Arpabet labels, no 'XX' column)."""
    rows = list(wb[sheet].iter_rows(values_only=True))
    head = [h for h in rows[0][1:] if h is not None]
    stim, data = [], []
    for r in rows[1:]:
        if r[0] is None or r[0] in ('Sum', 'Diff', 'XX') or str(r[0]).startswith('Table'):
            continue
        stim.append(r[0])
        data.append([0 if v is None else v for v in r[1:1 + len(head)]])
    m = pd.DataFrame(data, index=stim, columns=head, dtype=float)
    labels = VOWELS if 'ii' in head else CONSONANTS
    m = m.rename(index=labels, columns=lambda c: labels.get(c, c))
    return m.loc[m.index, m.index]          # square: drop the 'no response' (XX) column


def shepard_distance(counts):
    n = counts.values + SMOOTH
    ratio = n / np.diag(n)[:, None]         # n_ij / n_ii
    d = -0.5 * np.log(ratio * ratio.T)
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=counts.index, columns=counts.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xlsx', default='data/confusion-tables/TablesForJim_LucePhD.xlsx')
    ap.add_argument('--pb', default='data/confusion-tables/peterson_barney_1952_vowel_confusion_matrix.csv')
    ap.add_argument('--lexicon', default='data/lemmalex.csv')
    ap.add_argument('--out', default='data/phoneme_confusion_distances.csv')
    ap.add_argument('--no-adjust', action='store_true', help='keep the measured P-HH distance')
    args = ap.parse_args()

    phonemes = sorted(set(' '.join(pd.read_csv(args.lexicon)['Pronunciation']).split()))
    wb = openpyxl.load_workbook(args.xlsx, data_only=True)
    dist = {k: shepard_distance(read_table(wb, t)) for k, t in TABLES.items()}
    for k, d in dist.items():
        print(f"{TABLES[k]} ({k}): {len(d)} phonemes, distances {d.values[d.values > 0].min():.2f}"
              f" to {d.values.max():.2f}")

    missing = set(phonemes) - set().union(*[set(d.index) for d in dist.values()])
    if missing:
        raise ValueError(f"No confusion data for lexicon phonemes: {sorted(missing)}")

    # Average the available estimates for each pair; NaN where the pair was never tested
    stacked = np.stack([d.reindex(index=phonemes, columns=phonemes).values for d in dist.values()])
    n_estimates = np.sum(~np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    fill = max(d.values.max() for d in dist.values())
    untested = int((n_estimates == 0).sum() // 2)
    values = np.where(n_estimates > 0, total / np.maximum(n_estimates, 1), fill)
    np.fill_diagonal(values, 0.0)
    combined = pd.DataFrame(values, index=phonemes, columns=phonemes)
    print(f"\n{len(phonemes)} lexicon phonemes; {untested} untested pairs set to {fill:.2f}")

    if not args.no_adjust:
        consonants = [p for p in phonemes if p in CONSONANTS.values()]
        idx = [phonemes.index(p) for p in consonants]
        upper = np.triu_indices(len(idx), k=1)
        tested = n_estimates[np.ix_(idx, idx)][upper] > 0
        median = np.median(values[np.ix_(idx, idx)][upper][tested])
        for a, b in ADJUSTED_PAIRS:
            print(f"Adjusted {a}-{b}: {combined.loc[a, b]:.2f} -> {median:.2f} "
                  f"(median of {tested.sum()} tested consonant pairs)")
            combined.loc[a, b] = combined.loc[b, a] = median

    # Check: Luce vowels (noise) vs Peterson & Barney vowels (quiet)
    pb = pd.read_csv(args.pb, index_col=0)
    pb.index = pb.index.str.upper()
    pb.columns = pb.columns.str.upper()
    shared = [v for v in pb.index if v in dist['vowel'].index]
    pb_d = shepard_distance(pb.loc[shared, shared])
    pairs = list(combinations(shared, 2))
    x = pd.DataFrame({'luce': [dist['vowel'].loc[a, b] for a, b in pairs],
                      'pb': [pb_d.loc[a, b] for a, b in pairs]})
    print(f"Check: Luce vowels vs Peterson & Barney (quiet), {len(pairs)} pairs, "
          f"Spearman rho = {x.corr('spearman').iloc[0, 1]:.2f}")

    assert np.allclose(combined.values, combined.values.T)
    combined.index.name = 'phoneme'
    combined.round(4).to_csv(args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
