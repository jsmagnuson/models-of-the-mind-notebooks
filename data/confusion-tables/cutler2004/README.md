# Cutler, Weber, Smits & Cooper (2004): American English listeners

Cutler, A., Weber, A., Smits, R., & Cooper, N. (2004). Patterns of English phoneme
confusions by native and non-native listeners. *Journal of the Acoustical Society of
America, 116*(6), 3668-3678.

Not used by the notebook. Kept as an independent check on the Luce (1986) distances in
`../../phoneme_confusion_distances.csv`, and in particular on the /p/-/h/ adjustment made in
`../build_confusion_distances.py`.

## Files

- `AEcons.zip`, `AEvowel.zip`: the original trial-level data (one Excel 97 file each),
  downloaded 2026-10-19 from the paper's supplementary data page,
  https://www.mpi.nl/world/persons/private/anne/excel.html
- `cutler_AE_cons_trials.csv`, `cutler_AE_vowel_trials.csv`: the same trials as CSV, one row per
  trial (the first 8 columns of the `consonants` / `vowels` sheet, unchanged; the .xls files
  were converted to .xlsx with LibreOffice and read with openpyxl).
  Columns: `subject, answer, syllable, stimulus, vowel/cons, init/final, signalnoise, language`.
  `answer == 23miss` is a missed trial.
- `cutler_distances.py`: Shepard distances from these data, by the same method as the Luce
  builder, plus a comparison with the Luce distances.

## Design

16 American English listeners; 30,960 consonant trials and 30,960 vowel trials. Syllables are CV
and VC, in multispeaker babble at 0, 8, and 16 dB SNR. There are 24 consonants: /h/, /w/, and /j/
occur only in CV syllables, and /ŋ/ and /ʒ/ only in VC syllables. There are 15 vowels, including
ER, which our lexicon does not use.

Codes: consonants `14h` = HH, `14j` = ZH, `16dj` = JH, `19y` = Y, `10dth` = DH. The vowel codes
follow a long/short pattern (`ii` = IY, `i` = IH; `uu` = UW, `u` = UH), so `13oo` = AO and
`15o` = AA. The data alone cannot settle that last pair: both mappings agree equally well with
Luce's vowels.

## What it showed (2026-10-19)

- At 16 dB, /h/ is 84% correct in CV syllables. In Luce's clearest initial-consonant condition
  it is 27% correct, and was reported as /p/ more often than as /h/.
- Even so, P-HH is the 4th-closest of the 270 consonant pairs tested at 16 dB (distance 1.88), so
  the /p/-/h/ confusion is real in noise, not an artifact of Luce's data.
- Agreement with the Luce distances (Spearman, over pairs tested in both): consonants .67,
  vowels .73.
- Swapping Cutler distances in for Luce's (consonants only, all, or averaged) gave more word
  recognition errors at the notebook's clarity target (.95), 240-274 vs. 177. The cleaner
  consonant data lower the calibrated clarity, which makes the vowels noisier.
- Decision: keep Luce, and set P-HH to the median consonant distance (see
  `../build_confusion_distances.py`).
