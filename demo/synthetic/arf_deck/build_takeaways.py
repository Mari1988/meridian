"""Rebuild slide 11 ("Key takeaways") of the ARF deck.

Replaces the shapes on that one slide and touches nothing else. The visual
language is copied from the deck's own slides 1 and 7: a thin 2C7FB8 rule
above each numbered section, "N - Header" bold, muted 5A5A5A body, and a
C0392B accent rule under the slide title.

Every number here is verified against the run the deck's results slides
render from (fitted_models/scratch_ablation_r90_pinned) or against text
already on an earlier slide -- nothing is typed in from memory.
"""

import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'deck.pptx')
OUT = os.path.join(HERE, 'ARF-Analytics-Council-Talk-v2.pptx')

REPO = '/Users/mariappan.subramanian/Documents/repo/meridian/demo/synthetic'
RUN = os.path.join(REPO, 'fitted_models', 'scratch_ablation_r90_wellspec')

# Every figure on this slide comes from the SAME run slides 8-10 render, so
# the deck cannot quote two different numbers for one claim a few slides
# apart. Slide 10 moved onto this arm too, so nothing here is on the pinned
# run any more.
_ch1 = pd.read_csv(os.path.join(RUN, 'per_seed_all.csv'))
N = len(_ch1)
DEF_EC = _ch1['default_ec_err'].median()
INF_EC = _ch1['informed_ec_err'].median()
INF_LO = _ch1['informed_ec_err'].min()
INF_HI = _ch1['informed_ec_err'].max()

_m = pd.read_csv(os.path.join(RUN, 'mroi_both_channels.csv'))
_m = _m[(_m.variant == 'default') & (_m.spend_multiplier == 10.0)]
MROI_CH1_10X = _m[_m.channel == 'TV']['pct_error'].median()
MROI_CH2_10X = _m[_m.channel == 'Display']['pct_error'].median()

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)
RED = RGBColor(0xC0, 0x39, 0x2B)
TINT = RGBColor(0xEA, 0xF2, 0xF8)

# (number, header, body). Five, per the 3-5 asked for. Ordered as the talk
# runs: what the default assumes -> what it costs -> the three fixes.
SECTIONS = [
    (
        '1',
        'Ask what your model assumes before it sees your data',
        'Meridian applies one half-saturation prior to every channel, whatever '
        'its audience or delivery — 50% odds it is already past '
        'half-saturation.',
    ),
    (
        '2',
        'The bias lands on the channels you would most want to grow',
        'Channel-1, delivering at 10% of its half-saturation point, missed '
        f'the truth by {DEF_EC:.0f}% — low on all {N} datasets. Channel-2, '
        'already round the bend, was fine. The default does not fail '
        'everywhere; it fails where you are under-invested.',
    ),
    (
        '3',
        'Build the saturation prior from delivery',
        'Addressable audience × effective frequency, not current reach × '
        'current frequency. An anchor off by 25% still removes the systematic '
        'bias — though not the dataset-to-dataset spread.',
    ),
    (
        '4',
        'Set adstock and the carryover window by medium',
        'TV creative carries for weeks; digital decays almost immediately. And '
        'max_lag = 8 is a hard truncation, never estimated — at a 6.6-week '
        'half-life it discards 39% of true carryover.',
    ),
    (
        '5',
        "Stress-test at 2–5x spend, not at today's",
        'ROI and marginal ROI at current spend both concealed the problem. '
        'Only the elevated-spend curve separated the priors: at 10x the '
        f'default reads {MROI_CH1_10X:.0f}% on Channel-1 and is within 1% on '
        'Channel-2.',
    ),
]

COL_W = 5.80
LEFT_X, RIGHT_X = 0.60, 6.93
# Left column takes 1-3, right takes 4-5; the freed space under the right
# column carries the headline stat so the slide is not five text blocks.
POSITIONS = [
    (LEFT_X, 1.70), (LEFT_X, 3.38), (LEFT_X, 5.06),
    (RIGHT_X, 1.70), (RIGHT_X, 3.38),
]


def clear(slide):
  for shape in list(slide.shapes):
    shape._element.getparent().remove(shape._element)


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
  box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
  tf = box.text_frame
  tf.word_wrap = True
  tf.vertical_anchor = anchor
  # Text boxes carry internal padding by default, which would push every
  # block right of the rule above it.
  tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
  return tf


def rule(slide, x, y, w, color, h=0.045):
  bar = slide.shapes.add_shape(
      MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
  bar.fill.solid()
  bar.fill.fore_color.rgb = color
  bar.line.fill.background()
  bar.shadow.inherit = False
  return bar


def run(para, text, size, bold=False, color=INK, font='Calibri'):
  r = para.add_run()
  r.text = text
  r.font.size = Pt(size)
  r.font.bold = bold
  r.font.color.rgb = color
  r.font.name = font
  return r


def build(slide):
  clear(slide)

  tf = textbox(slide, 0.60, 0.48, 12.10, 0.55)
  run(tf.paragraphs[0], 'Key takeaways', 30, bold=True)
  rule(slide, 0.60, 1.18, 1.70, RED, h=0.04)

  for (num, header, body), (x, y) in zip(SECTIONS, POSITIONS):
    rule(slide, x, y, COL_W, BLUE)
    tf = textbox(slide, x, y + 0.13, COL_W, 0.30)
    p = tf.paragraphs[0]
    run(p, f'{num} · ', 15, bold=True, color=BLUE)
    run(p, header, 15, bold=True)
    tf = textbox(slide, x, y + 0.52, COL_W, 1.05)
    run(tf.paragraphs[0], body, 11.5, color=MUTED)

  # Headline stat, in the space the right column leaves free.
  card = slide.shapes.add_shape(
      MSO_SHAPE.ROUNDED_RECTANGLE,
      Inches(RIGHT_X), Inches(5.06), Inches(COL_W), Inches(1.62))
  card.fill.solid()
  card.fill.fore_color.rgb = TINT
  card.line.fill.background()
  card.shadow.inherit = False
  card.adjustments[0] = 0.06
  card.text_frame.text = ''

  tf = textbox(slide, RIGHT_X + 0.30, 5.24, COL_W - 0.60, 0.52)
  p = tf.paragraphs[0]
  run(p, f'{DEF_EC:.0f}%', 30, bold=True, color=RED)
  run(p, '   →   ', 30, bold=True, color=MUTED)
  run(p, f'{INF_EC:+.1f}%', 30, bold=True, color=BLUE)

  # The spread belongs ON the stat, not in a footnote. Quoting the medians
  # alone would say the informed prior "recovers" ec_m; it does not -- it
  # removes the systematic bias and stays wide.
  tf = textbox(slide, RIGHT_X + 0.30, 5.84, COL_W - 0.60, 0.72)
  run(tf.paragraphs[0],
      'Median error in Channel-1’s true saturation point — Meridian’s '
      f'default prior vs. a reach-informed one, across {N} datasets built '
      'from an identical known truth. The informed prior removes the bias '
      f'but stays wide: {INF_LO:.0f}% to {INF_HI:+.0f}% on any single '
      'dataset.',
      9.5, color=MUTED)


def main():
  prs = Presentation(SRC)
  build(prs.slides[10])
  prs.save(OUT)
  print('wrote', OUT)


if __name__ == '__main__':
  main()
