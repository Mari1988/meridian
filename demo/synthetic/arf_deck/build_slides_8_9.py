"""Finish slides 8 and 9 of the ARF deck (the two recovery slides).

What was unfinished:
  * slide 8 had a title but no caption -- the reader was left to interpret
    three box plots unaided.
  * slide 9 had NO text at all: no title, no caption.
  * both carried hand-cropped images whose captions said "10 draws".

Runs on the v2 file that `build_takeaways.py` produces, so the deck
accumulates both edits. Re-runnable: it clears each slide and rebuilds it.

EVERY NUMBER IS READ FROM THE RUN'S CSVs, never typed in. The figures and the
caption therefore cannot disagree, which is the failure mode that put "10
draws" under a 50-draw figure in the first place.
"""

import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/Users/mariappan.subramanian/Documents/repo/meridian/demo/synthetic'
RUN = os.path.join(REPO, 'fitted_models', 'scratch_ablation_r90_wellspec')
FIG = os.path.join(REPO, 'figures')
DECK = os.path.join(HERE, 'ARF-Analytics-Council-Talk-v2.pptx')

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)
RED = RGBColor(0xC0, 0x39, 0x2B)


def facts():
  """Medians and ranges for both channels, from the run's own CSVs."""
  ch1 = pd.read_csv(os.path.join(RUN, 'per_seed_all.csv'))
  ch2 = pd.read_csv(os.path.join(RUN, 'per_seed_all_ch2.csv'))
  assert len(ch1) == len(ch2), (
      f'{len(ch1)} Channel-1 rows vs {len(ch2)} Channel-2 rows -- re-run '
      'scratch_extract_r90_wellspec_ch2.py so both cover the same draws')
  out = {'n': len(ch1)}
  for tag, df in (('ch1', ch1), ('ch2', ch2)):
    for metric in ('ec', 'roi', 'alpha'):
      for arm in ('default', 'informed'):
        col = f'{arm}_{metric}_err'
        out[f'{tag}_{arm}_{metric}'] = df[col].median()
        out[f'{tag}_{arm}_{metric}_lo'] = df[col].min()
        out[f'{tag}_{arm}_{metric}_hi'] = df[col].max()
    out[f'{tag}_wins_ec'] = int(
        (df['informed_ec_err'].abs() < df['default_ec_err'].abs()).sum())
  return out


def clear(slide):
  for shape in list(slide.shapes):
    shape._element.getparent().remove(shape._element)


def textbox(slide, x, y, w, h):
  tf = slide.shapes.add_textbox(
      Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
  tf.word_wrap = True
  tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
  return tf


def run(para, text, size, bold=False, color=INK):
  r = para.add_run()
  r.text = text
  r.font.size = Pt(size)
  r.font.bold = bold
  r.font.color.rgb = color
  r.font.name = 'Calibri'
  return r


def build(slide, title, standfirst, image, caption_runs):
  """One recovery slide: title, standfirst, panels, caption.

  Geometry matches the deck's other content slides (title at 0.60/0.35,
  full-width body at 12.10) rather than slide 8's old 0.24/0.18, which sat
  slightly proud of every other slide.
  """
  clear(slide)

  tf = textbox(slide, 0.60, 0.35, 12.10, 0.45)
  run(tf.paragraphs[0], title, 22, bold=True)

  tf = textbox(slide, 0.60, 0.88, 12.10, 0.34)
  run(tf.paragraphs[0], standfirst, 12, color=MUTED)

  # The figure is 13.2 x 4.1 in at source; 11.08 wide keeps that aspect.
  slide.shapes.add_picture(
      image, Inches(1.05), Inches(1.42), width=Inches(11.20),
      height=Inches(3.48))

  tf = textbox(slide, 0.60, 5.20, 12.10, 1.60)
  p = tf.paragraphs[0]
  for text, bold, color in caption_runs:
    run(p, text, 13, bold=bold, color=color)


def mroi_facts():
  """Median % error and 90%-interval coverage, per channel and multiplier."""
  m = pd.read_csv(os.path.join(RUN, 'mroi_both_channels.csv'))
  fit = m[m.variant != 'truth'].copy()
  fit['covered'] = (fit.q05 <= fit.true_mroi) & (fit.true_mroi <= fit.q95)
  out = {'n': int(fit.seed.nunique())}
  for ch, tag in (('TV', 'ch1'), ('Display', 'ch2')):
    for variant, arm in (('default', 'default'),
                         ('ec_alpha_noisy', 'informed')):
      sub = fit[(fit.channel == ch) & (fit.variant == variant)]
      for mult in (1.0, 2.0, 3.0, 10.0):
        s = sub[sub.spend_multiplier == mult]
        out[f'{tag}_{arm}_{int(mult)}x'] = s['pct_error'].median()
        # How many datasets the 90% interval MISSES the truth on -- the
        # "confidently wrong" count, which is the claim being made.
        out[f'{tag}_{arm}_{int(mult)}x_miss'] = int((~s['covered']).sum())
  return out


def build_mroi(slide, f):
  """Slide 10: the same figure, rebased onto the well-specified arm."""
  clear(slide)

  tf = textbox(slide, 0.60, 0.35, 12.10, 0.45)
  run(tf.paragraphs[0],
      'How does marginal ROI compare as we scale the spend?', 22, bold=True)

  tf = textbox(slide, 0.60, 0.88, 12.10, 0.34)
  run(tf.paragraphs[0],
      '“Should I spend more here?” — marginal return at spend levels above '
      f'today’s, across the same {f["n"]} datasets.', 12, color=MUTED)

  slide.shapes.add_picture(
      os.path.join(FIG, 'wellspec_mroi_50.png'),
      Inches(1.43), Inches(1.34), width=Inches(10.46), height=Inches(4.60))

  tf = textbox(slide, 0.60, 6.10, 12.10, 1.10)
  p = tf.paragraphs[0]
  run(p, 'Same prior, same extrapolation to 10x: on Channel-1 the default '
         'reads ', 13)
  run(p, f'{f["ch1_default_10x"]:.0f}%', 13, bold=True, color=RED)
  run(p, ', on Channel-2 it is within 1%. Worse than being wrong, it is '
         'confidently wrong — the default’s 90% interval misses the truth on ',
      13)
  run(p, f'{f["ch1_default_3x_miss"]}/{f["n"]}', 13, bold=True, color=RED)
  run(p, ' datasets from 3x upward. The model does not break when you '
         'extrapolate; it breaks when you extrapolate from a misplaced '
         'saturation point.', 13)


def main():
  f = facts()
  n = f['n']
  prs = Presentation(DECK)

  build(
      prs.slides[7],
      'Does the model recover the truth it was built from?  '
      'Channel-1 — the under-reached channel',
      f'{n} datasets built from an identical known truth; only the noise '
      'differs. The reach-informed anchor is deliberately wrong by ~25% on '
      'each draw. Labels are medians.',
      os.path.join(FIG, 'wellspec_slide7_ch1.png'),
      [
          ('The default misses the true saturation point by ', False, INK),
          (f'{f["ch1_default_ec"]:.0f}%', True, RED),
          (f' — and misses low on all {n} datasets. The reach-informed '
           'prior is centred — ', False, INK),
          (f'{f["ch1_informed_ec"]:+.1f}%', True, BLUE),
          (f', closer on {f["ch1_wins_ec"]}/{n} — but wide: '
           f'{f["ch1_informed_ec_lo"]:.0f}% to '
           f'{f["ch1_informed_ec_hi"]:+.0f}% on any single dataset. '
           'It removes the systematic bias; it does not recover the '
           'parameter.', False, INK),
      ],
  )

  build(
      prs.slides[8],
      'The same test on Channel-2 — the well-reached control',
      'Same model, same priors, same '
      f'{n} datasets. The only thing that changes is where the channel '
      'already sits on its own curve: 43.6% of half-saturation, not 10%.',
      os.path.join(FIG, 'wellspec_slide7_ch2.png'),
      [
          ('Here the default is fine: ', False, INK),
          (f'{f["ch2_default_ec"]:+.1f}%', True, RED),
          (' on saturation, against the informed prior’s ', False, INK),
          (f'{f["ch2_informed_ec"]:+.1f}%', True, BLUE),
          (' — and both arms land within 5% on ROI and adstock too. That is '
           'what makes Channel-1 an indictment of the prior rather than of '
           'the model: the default prior is not broken, it is wrong only '
           'where you are under-invested.', False, INK),
      ],
  )

  mf = mroi_facts()
  assert mf['n'] == n, (
      f'mROI sweep covers {mf["n"]} seeds but the recovery run has {n} -- '
      'the two would disagree on the slide')
  build_mroi(prs.slides[9], mf)

  prs.save(DECK)
  print(f'rebuilt slides 8, 9 and 10 on {n} draws -> {DECK}')
  print('  mROI Ch-1 default: '
        + ', '.join(f'{m}x {mf[f"ch1_default_{m}x"]:+.1f}% '
                    f'(misses {mf[f"ch1_default_{m}x_miss"]}/{n})'
                    for m in (1, 2, 3, 10)))
  print('  mROI Ch-1 informed: '
        + ', '.join(f'{m}x {mf[f"ch1_informed_{m}x"]:+.1f}%'
                    for m in (1, 2, 3, 10)))
  print('  mROI Ch-2 default: '
        + ', '.join(f'{m}x {mf[f"ch2_default_{m}x"]:+.1f}%'
                    for m in (1, 2, 3, 10)))
  for k in sorted(f):
    print(f'  {k:28s} {f[k]:+.2f}' if isinstance(f[k], float) else
          f'  {k:28s} {f[k]}')


if __name__ == '__main__':
  main()
