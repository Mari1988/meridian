# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds the ARF Analytics Council deck as a .pptx.

Slides 1-4 -- "What does an MMM assume before it sees your data?" -- cover three
Meridian defaults (saturation prior, fixed Hill slope, adstock decay), plus a
framing slide. The adstock slide (4) is deliberately gentler than the other
two: `alpha_m ~ Uniform(0, 1)` genuinely is flat -- no skew story like `ec_m`'s
-- so it makes the half-life implications of that flat prior legible instead,
then shows the real binding assumption, `max_lag`, which is not a prior at all.

Slides 5-8 carry the results: the simulated DGP and why it is a fair test,
where the simulated execution actually sits on each channel's true response
curve, parameter recovery, and marginal ROI as spend scales. The
response-curve slide was cut -- the mROI slide makes the same point in the
planner's own units, and makes it on two channels.

Slide text and figure annotations read their numbers from
`prior_plots.default_prior_facts()` (slides 1-4), `results_facts` (slide 6),
and the `_r90_*_facts()` functions below (slides 5, 7, 8), so they cannot
drift from the runs that produced them. The deck ships with empty speaker-notes
fields -- see `EMIT_NOTES` and `_notes()`.

Channels are presented as Channel-1 / Channel-2 via `channel_labels`; TV /
Display remain the keys on the data side.

Usage:
  .venv/bin/python demo/synthetic/export_curve_data.py       # ~5 min, fits
  .venv/bin/python demo/synthetic/prior_plots_check.py       # verify numbers
  .venv/bin/python demo/synthetic/build_section1_figures.py  # slides 1-4
  .venv/bin/python demo/synthetic/build_results_figures.py   # slide 6
  .venv/bin/python demo/synthetic/scratch_build_slide5_r90.py       # slide 5
  .venv/bin/python demo/synthetic/scratch_plot_recovery_boxplot_r90_fulldraws.py
  .venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8_pinned.py
  .venv/bin/python demo/synthetic/build_section1_deck.py     # build the deck
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu
from pptx.util import Inches
from pptx.util import Pt

import channel_labels
import prior_plots
import results_facts


HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(HERE, 'figures')
OUTPUT_PATH = os.path.join(HERE, 'arf_section1_deck.pptx')
TAKEAWAYS_ONLY_PATH = os.path.join(HERE, 'arf_takeaways_slide.pptx')

SLIDE_W = Inches(13.333)  # 16:9.
SLIDE_H = Inches(7.5)

# Speaker notes are written by hand outside this script; the deck ships with
# empty notes fields. See `_notes()` for why the text stays in the source.
EMIT_NOTES = False

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)

# Every results slide reads from this one run: 10 seeds whose true parameters
# are identical, fitted with the default prior and with an achievable
# (~25%-perturbed) reach-informed prior. See `r90_basis`.
PINNED_RUN_DIR = os.path.join(
    HERE, 'fitted_models', 'scratch_ablation_r90_pinned'
)

MERIDIAN_PRIOR_DOC = (
    'https://developers.google.com/meridian/docs/advanced-modeling/'
    'default-prior-distributions#ec_m_and_ec_om'
)


def _blank_slide(prs: Presentation):
  """Adds a slide using the blank layout (index 6 in the default template)."""
  return prs.slides.add_slide(prs.slide_layouts[6])


def _textbox(slide, left, top, width, height, *, align_top=True):
  box = slide.shapes.add_textbox(left, top, width, height)
  frame = box.text_frame
  frame.word_wrap = True
  if align_top:
    frame.margin_top = 0
  return frame


def _para(
    frame, text, *, size, bold=False, color=INK, space_after=6, first=False
):
  """Appends a paragraph (or fills the first, which always exists)."""
  para = frame.paragraphs[0] if first else frame.add_paragraph()
  para.text = text
  para.space_after = Pt(space_after)
  font = para.font
  font.size = Pt(size)
  font.bold = bold
  font.color.rgb = color
  return para


def _title(slide, text, subtitle=None):
  frame = _textbox(slide, Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0))
  _para(frame, text, size=30, bold=True, first=True, space_after=2)
  if subtitle:
    _para(frame, subtitle, size=15, color=MUTED)


def _footnote(slide, text):
  frame = _textbox(slide, Inches(0.6), Inches(7.05), Inches(12.1), Inches(0.35))
  _para(frame, text, size=9, color=MUTED, first=True)


def _picture_fitted(slide, image_path, top, max_h_in=4.0):
  """Places an image centered horizontally, scaled to fit without distortion."""
  with Image.open(image_path) as img:
    px_w, px_h = img.size
  aspect = px_h / px_w
  width_in = 12.1
  height_in = width_in * aspect
  if height_in > max_h_in:
    height_in = max_h_in
    width_in = height_in / aspect
  left = Emu(int((SLIDE_W - Inches(width_in)) / 2))
  slide.shapes.add_picture(
      image_path, left, top, width=Inches(width_in), height=Inches(height_in)
  )


def _picture_at(slide, image_path, left_in, top_in, width_in):
  """Places an image at an explicit position, scaled to a fixed width."""
  slide.shapes.add_picture(
      image_path, Inches(left_in), Inches(top_in), width=Inches(width_in)
  )


def _notes(slide, text: str) -> None:
  """Attaches speaker notes, unless `EMIT_NOTES` is off.

  The notes text is kept in this source file even when it is not emitted: it
  is the study's do-not-overclaim record (CLAUDE.md points at it before any
  results slide is changed), so deleting it would lose the guidance, not just
  the notes. Flip `EMIT_NOTES` to put it back in the .pptx.
  """
  if not EMIT_NOTES:
    return
  slide.notes_slide.notes_text_frame.text = text


def build_deck() -> str:
  """Builds the nine-slide deck and returns the output path."""
  facts = prior_plots.default_prior_facts()
  results = results_facts.all_facts()
  prs = Presentation()
  prs.slide_width = SLIDE_W
  prs.slide_height = SLIDE_H

  _build_framing_slide(prs)
  _build_saturation_slide(prs, facts)
  _build_slope_slide(prs)
  _build_adstock_slide(prs, facts)

  _build_dgp_slide(prs, _r90_dgp_facts())
  _build_execution_slide(prs, results['dgp'])
  _build_recovery_table_slide(prs, _r90_recovery_facts())
  _build_mroi_slide(prs, _r90_mroi_facts())
  _build_takeaways_slide(prs, _r90_recovery_facts(), _r90_mroi_facts(), facts)

  prs.save(OUTPUT_PATH)
  return OUTPUT_PATH


def build_takeaways_only() -> str:
  """Builds a one-slide .pptx containing just the closing takeaways slide.

  For iterating on that slide's wording without rebuilding the whole deck (and
  without the figure dependencies the results slides carry). Writes to a
  separate path so it can never be mistaken for, or overwrite, the real deck.
  """
  prs = Presentation()
  prs.slide_width = SLIDE_W
  prs.slide_height = SLIDE_H
  _build_takeaways_slide(
      prs,
      _r90_recovery_facts(),
      _r90_mroi_facts(),
      prior_plots.default_prior_facts(),
  )
  prs.save(TAKEAWAYS_ONLY_PATH)
  return TAKEAWAYS_ONLY_PATH


def _build_framing_slide(prs) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'What does an MMM assume before it sees your data?',
      'Every marketing mix model commits to three answers before reading a '
      'single row.',
  )

  frame = _textbox(slide, Inches(0.9), Inches(1.9), Inches(11.5), Inches(4.2))
  for i, (question, gloss) in enumerate(
      [
          (
              '1.  How fast does response flatten?',
              'Where is the point of diminishing returns for this channel?',
          ),
          (
              '2.  What shape is the curve?',
              'Does response build before it bends, or diminish from '
              'impression one?',
          ),
          (
              '3.  How long does an impression keep working?',
              'How much of the effect lands this week, and how far does it '
              'carry?',
          ),
      ]
  ):
    _para(
        frame,
        question,
        size=22,
        bold=True,
        color=BLUE,
        first=(i == 0),
        space_after=2,
    )
    _para(frame, f'      {gloss}', size=15, color=MUTED, space_after=20)

  _para(
      frame,
      'You are never asked these questions. They ship as defaults.',
      size=20,
      bold=True,
      color=ACCENT,
      space_after=10,
  )
  _para(
      frame,
      'We can inspect these choices only because Meridian is open source — '
      'every closed MMM makes the same three choices, unauditably.',
      size=14,
      color=MUTED,
  )

  _notes(
      slide,
      'Section 1 opener, ~45 seconds.\n\n'
      'The point of the slide is that the questions exist at all. Most of the '
      'room has never been asked any of the three.\n\n'
      'Say the open-source line out loud. This section is not a criticism of '
      'the tool; it is an argument for the kind of scrutiny only an open tool '
      'permits. That framing carries the whole talk and heads off the reading '
      'that this is a hit piece on one vendor.',
  )


def _build_saturation_slide(prs, facts) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Assumption 1 — Saturation: the default says you are already there',
      'Meridian scales each channel by its own median execution first, so '
      '"1.0" on the model’s x-axis means "what you already run".',
  )
  _picture_fitted(
      slide, os.path.join(FIGURE_DIR, 'section1_saturation.png'), Inches(1.55)
  )

  frame = _textbox(slide, Inches(0.6), Inches(5.7), Inches(12.1), Inches(1.3))
  _para(
      frame,
      f'Out of the box: {facts["p_past_half_saturation_today"]:.0%} odds you '
      f'are already past half-saturation, and '
      f'{facts["p_over_third_of_ceiling_today"]:.0%} odds you have captured '
      'more than a third of the channel’s maximum effect — applied '
      'identically to every channel, whatever its audience or reach.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
  )
  _footnote(
      slide,
      f'ec_m ~ TruncatedNormal(0.8, 0.8, [0.1, 10]).  {MERIDIAN_PRIOR_DOC}',
  )

  _notes(
      slide,
      'THE key mechanic: Meridian scales media by that channel’s own median '
      'non-zero per-capita execution before the saturation curve. So x = 1.0 '
      'is "what you already run", and ec_m is denominated in multiples of your '
      'own status quo.\n\n'
      f'Prior median ec_m = {facts["ec_m_median"]:.2f}x current execution.\n'
      'P(past half-saturation today) = '
      f'{facts["p_past_half_saturation_today"]:.0%}.\n'
      'P(>1/3 of ceiling captured today) = '
      f'{facts["p_over_third_of_ceiling_today"]:.0%}.\n'
      f'90% prior interval = [{facts["ec_m_q05"]:.2f}, '
      f'{facts["ec_m_q95"]:.2f}]x.\n\n'
      'The line: the model gives you coin-flip odds you are already past the '
      'point of diminishing returns, for every channel, regardless of audience '
      'size or how much of it you actually reach.\n\n'
      'IF ASKED "is this hidden?" — No, and do not claim it is. Google '
      'documents it: half-saturation occurs "at the median of the non-zero '
      'media units per capita across geos and time", and the truncation is '
      'justified as preserving identifiability. The point is that it is easy '
      'to miss and rarely interrogated, not that it is concealed.',
  )


def _build_slope_slide(prs) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Assumption 2 — Slope: the parameter that is never estimated',
      'slope_m = Deterministic(1.0). It has no prior, because it is not a '
      'random variable.',
  )
  _picture_fitted(
      slide, os.path.join(FIGURE_DIR, 'section1_slope.png'), Inches(1.55)
  )

  frame = _textbox(slide, Inches(0.6), Inches(5.7), Inches(12.1), Inches(1.3))
  _para(
      frame,
      'Fixing the slope at 1 means diminishing returns from the very first '
      'impression — no build-up, no threshold effect. The stated reason is '
      'tractability, not advertising: concave curves guarantee the budget '
      'optimizer a global optimum.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
  )
  _footnote(
      slide,
      'Meridian: hill(x) = x^slope / (x^slope + ec^slope).',
  )

  _notes(
      slide,
      'Slope and half-saturation are the two parameters of the SAME curve, so '
      'this follows directly from the previous slide.\n\n'
      'slope_m is fixed, not estimated — it has no prior because it is not '
      'a random variable. Google’s stated rationale is that concave Hill '
      'curves are required so budget optimization "produces a global optimum". '
      'A legitimate engineering trade-off, and one the practitioner inherited '
      'without being asked. It forecloses exactly the effective-frequency '
      'threshold behavior most media planners believe in.\n\n'
      'READ THE RIGHT PANEL ALOUD: dots mark peak marginal return. Under the '
      'default it sits at ZERO — the model believes your first impression '
      'was your most productive one, always. At slope 2 the peak is at 0.58x '
      'today’s spend; at slope 3, 0.79x. Those build-up phases are '
      'unrepresentable at any parameter value of the default.\n\n'
      'FAIRNESS CAVEAT — say it: Meridian does let you override slope_m; it '
      'warns that doing so may break the optimizer’s global optimum. The '
      'claim is about the default, not that the door is locked.',
  )


def _build_adstock_slide(prs, facts) -> None:
  slide = _blank_slide(prs)
  hl_03 = float(prior_plots.half_life(0.3))
  hl_09 = float(prior_plots.half_life(0.9))
  _title(
      slide,
      'Assumption 3 — Adstock: a fair prior, and a hard cutoff behind it',
      'alpha_m ~ Uniform(0, 1) is honestly flat — no skew to complain about '
      'here. The binding assumption sits elsewhere.',
  )
  _picture_fitted(
      slide, os.path.join(FIGURE_DIR, 'section1_adstock.png'), Inches(1.55)
  )

  frame = _textbox(slide, Inches(0.6), Inches(5.7), Inches(12.1), Inches(1.3))
  _para(
      frame,
      f'Decay rate alone is not interpretable — half-life is. The same flat '
      f'prior spans routines from a {hl_03:.1f}-week half-life at alpha=0.3 '
      f'to a {hl_09:.1f}-week half-life at alpha=0.9.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
      space_after=4,
  )
  _para(
      frame,
      'And the binding assumption is not a prior at all: max_lag = 8 is a '
      'hard truncation, never estimated. At a decay rate of 0.9 '
      f'({hl_09:.1f}-week half-life), '
      f'{facts["carryover_lost_beyond_max_lag_at_alpha_090"]:.0%} of true '
      'carryover falls beyond week 8 and is silently discarded.',
      size=14,
      bold=True,
      color=ACCENT,
      space_after=4,
  )

  _notes(
      slide,
      'This slide is deliberately gentler than the previous two — alpha_m '
      '~ Uniform(0,1) genuinely is uninformative, and Google describes it '
      'that way. There is no skew story here, and we should say so plainly.'
      '\n\n'
      '(a) The parameter itself is not what a planner reasons about — '
      'half-life is. Translate it: alpha=0.3 decays to half its effect in '
      f'{hl_03:.1f} weeks; alpha=0.9 takes {hl_09:.1f} weeks. Read straight '
      'off the legend.\n\n'
      '(b) The binding assumption is not a prior at all. max_lag = 8 is a '
      'hard truncation in the model spec — never estimated, no prior, no '
      'posterior uncertainty. At decay 0.9, '
      f'{facts["carryover_lost_beyond_max_lag_at_alpha_090"]:.0%} of true '
      'carryover falls beyond week 8 and is discarded and renormalized '
      'away. If your TV genuinely works over a quarter, the default cannot '
      'see it, and nothing in the output tells you so.\n\n'
      'THE TAXONOMY IS THE THESIS: some defaults are skewed toward a '
      'conclusion; some are genuinely flat and fair on the parameter, but '
      'the real structure hides outside the prior altogether. Land that, '
      'then move to the TV case study.',
  )


def _model_form_equation(slide, left_in, top_in, width_in):
  """The DGP's model form as a single named equation, for slide 5.

  Two text frames: the heading + `Sales_g,t = Intercept_g` sit flush left;
  the remaining terms sit in a second, indented frame below, so the '+' list
  reads as a continuation of the first line rather than a new block.
  """
  head = _textbox(
      slide, Inches(left_in), Inches(top_in), Inches(width_in), Inches(0.9)
  )
  _para(
      head,
      'Realistic Data Generation Process',
      size=15,
      bold=True,
      color=BLUE,
      first=True,
      space_after=8,
  )
  _para(head, 'Sales_g,t  =  Intercept_g', size=14, bold=True, space_after=10)

  terms = _textbox(
      slide,
      Inches(left_in + 0.35),
      Inches(top_in + 1.0),
      Inches(width_in - 0.35),
      Inches(2.4),
  )
  term_lines = [
      'Trend_t',
      'Seasonality_t',
      'SharedShock_t',
      'Σ_c ControlWeight_g,c · Control_g,t,c',
      'Σ_m Coefficient_g,m · Saturation( Carryover(Impressions_g,t,m; '
      'Decay_m) ; HalfSaturation_m, Slope_m )',
      'Noise_g,t',
  ]
  for i, line in enumerate(term_lines):
    _para(
        terms, f'+   {line}', size=13, bold=True, first=(i == 0), space_after=6
    )
  return head, terms


def _plain_table(slide, left_in, top_in, width_in, header, rows, col_widths_in):
  """A plain pptx table with uniform (non-comparison) text colour."""
  shape = slide.shapes.add_table(
      len(rows) + 1,
      len(header),
      Inches(left_in),
      Inches(top_in),
      Inches(width_in),
      Inches(0.42 * (len(rows) + 1)),
  )
  table = shape.table
  for i, w in enumerate(col_widths_in):
    table.columns[i].width = Inches(w)
  for j, text in enumerate(header):
    cell = table.cell(0, j)
    cell.text = text
    para = cell.text_frame.paragraphs[0]
    para.font.size = Pt(12)
    para.font.bold = True
    para.font.color.rgb = INK
  for i, row in enumerate(rows, start=1):
    for j, text in enumerate(row):
      cell = table.cell(i, j)
      cell.text = str(text)
      para = cell.text_frame.paragraphs[0]
      para.font.size = Pt(12)
      para.font.color.rgb = INK
  return table


def _table(slide, left_in, top_in, width_in, header, rows, col_widths_in):
  """A plain pptx table -- selectable text, not a screenshot of one."""
  shape = slide.shapes.add_table(
      len(rows) + 1,
      len(header),
      Inches(left_in),
      Inches(top_in),
      Inches(width_in),
      Inches(0.4 * (len(rows) + 1)),
  )
  table = shape.table
  for i, w in enumerate(col_widths_in):
    table.columns[i].width = Inches(w)
  for j, text in enumerate(header):
    cell = table.cell(0, j)
    cell.text = text
    para = cell.text_frame.paragraphs[0]
    para.font.size = Pt(12)
    para.font.bold = True
    para.font.color.rgb = INK
  for i, row in enumerate(rows, start=1):
    for j, text in enumerate(row):
      cell = table.cell(i, j)
      cell.text = str(text)
      para = cell.text_frame.paragraphs[0]
      para.font.size = Pt(12)
      # Colour the two fitted columns by which prior they belong to.
      if j == 3:
        para.font.color.rgb = ACCENT
      elif j == 4:
        para.font.color.rgb = BLUE
      else:
        para.font.color.rgb = INK
  return table


def _fmt_true(values, fmt: str = '{:.2f}', pinned_rtol: float = 1e-5) -> str:
  """Formats a truth as a single value if pinned across draws, else a range.

  Slide 5 sits in front of slides 7-8, which pool ten simulator draws. On the
  pinned basis (`r90_basis`) every truth is constant across them, so every
  cell should print one value; a range appearing here means the DGP has
  started varying the estimand again, which would make the seed sweep measure
  two things at once.

  The tolerance is RELATIVE, not absolute. These are float32 quantities
  recomputed per draw, so a pinned value still differs in the last bit or two
  (~2e-7 relative on a true ROI of 6.64). An absolute tolerance renders that
  as the nonsense range "6.6 - 6.6"; 1e-5 relative absorbs the float noise
  while still catching real drift by orders of magnitude.
  """
  lo, hi = float(min(values)), float(max(values))
  if hi - lo <= pinned_rtol * max(abs(lo), abs(hi), 1e-12):
    return fmt.format(lo)
  return f'{fmt.format(lo)} – {fmt.format(hi)}'


def _r90_dgp_facts() -> dict[str, object]:
  """Slide 5's finalized basis: alpha_m=0.3, oracle_r2=0.9, seeds 1320/7/42.

  Sourced from `scratch_build_slide5_r90.py` (the seed-1320 dataset-level
  diagnostics and the plotted draw) and `scratch_export_r90_truths.py` (every
  channel's truth in each of the three seeds slides 7-9 pool), not
  `results_facts` -- that module is scoped to the older canonical
  alpha_m=0.8/oracle_r2=0.80 run, which this slide no longer describes.

  The per-channel truths come back as **formatted strings**, single-valued
  where the quantity is pinned across draws and a range where it is not, so
  the table cannot present one draw as the ground truth. See `_fmt_true`.
  """
  run_dir = os.path.join(HERE, 'fitted_models', 'scratch_slide5_r90')
  dataset = pd.read_csv(os.path.join(run_dir, 'dataset.csv')).iloc[0]
  truths = pd.read_csv(os.path.join(PINNED_RUN_DIR, 'true_params_by_seed.csv'))
  out = {
      'n_geos': int(dataset['n_geos']),
      'n_times': int(dataset['n_times']),
      'oracle_r2': float(dataset['oracle_r2']),
      'media_share_pct': float(dataset['media_share_pct']),
      'n_seeds': int(truths.seed.nunique()),
      'channels': list(pd.unique(truths.channel)),
  }
  for channel in out['channels']:
    key = channel.lower()
    rows = truths[truths.channel == channel]
    out[f'true_ec_{key}'] = _fmt_true(rows['true_ec_m'])
    out[f'true_roi_{key}'] = _fmt_true(rows['true_roi_m'], '{:.1f}')
    out[f'true_alpha_{key}'] = _fmt_true(rows['true_alpha_m'])
    # Kept numeric for the assertions in `prior_plots_check`.
    out[f'true_ec_{key}_min'] = float(rows['true_ec_m'].min())
    out[f'true_alpha_{key}_min'] = float(rows['true_alpha_m'].min())
  return out


def _build_dgp_slide(prs, dgp) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Simulated data with known ground truth',
      'In the real world the true saturation point is never observed, only the '
      'fitted one. So: simulate geo × time data from a known generative model, '
      'fit an MMM to it, and check whether it recovers the truth it was built '
      'from.',
  )

  _model_form_equation(slide, 0.6, 1.7, 7.0)
  _picture_at(
      slide,
      os.path.join(FIGURE_DIR, 'slide5_series_r90.png'),
      8.0,
      1.45,
      4.75,
  )

  _para_frame = _textbox(
      slide, Inches(0.6), Inches(4.75), Inches(7.0), Inches(0.4)
  )
  _para(
      _para_frame,
      f'What we know that no real dataset tells you — across all '
      f'{dgp["n_seeds"]} simulated datasets',
      size=15,
      bold=True,
      first=True,
  )

  _plain_table(
      slide,
      0.6,
      5.15,
      7.0,
      [
          'Channel',
          'Half-saturation (× current delivery)',
          'True ROI',
          'True carryover (α)',
      ],
      [
          [
              channel_labels.label(channel),
              f'{dgp[f"true_ec_{channel.lower()}"]}×',
              dgp[f'true_roi_{channel.lower()}'],
              dgp[f'true_alpha_{channel.lower()}'],
          ]
          for channel in dgp['channels']
      ],
      [1.5, 2.6, 1.4, 1.5],
  )
  dataset_frame = _textbox(
      slide, Inches(0.6), Inches(6.55), Inches(7.0), Inches(0.5)
  )
  _para(
      dataset_frame,
      f'{dgp["n_geos"]} regions × {dgp["n_times"]} weeks, media '
      f'{dgp["media_share_pct"]:.0f}% of the outcome — a well-powered dataset, '
      'not a starved one.',
      size=12,
      color=MUTED,
      first=True,
  )
  _footnote(
      slide,
      f'Ranges span the {dgp["n_seeds"]} simulated datasets the results '
      'slides pool. A single value means the quantity is pinned across all of '
      'them by construction, so the comparison isolates it; ROI is '
      're-derived from whatever data each draw produces, so it moves. The '
      'specific saturation threshold is a stated modelling assumption, not a '
      'claim about real-world saturation; what is being tested is the '
      'mechanism, not that number.',
  )
  _notes(
      slide,
      'The point of this slide is credibility, not detail. Do not read the '
      'numbers out — land two ideas.\n\n'
      'ONE: you cannot answer "is my MMM right about saturation" with real '
      'data, because you never observe the true saturation point. Only '
      'simulation gives you a known answer to check against.\n\n'
      'TWO: walk through the equation to show what "known answer" actually '
      'means here — a geo intercept, trend, seasonality, a shared shock '
      'standing in for competitor activity/pricing/national promotions, and '
      'observed controls, all plus geo-level noise; media runs through the '
      'same adstock/Hill maths from slides 2-4 (alpha for carryover, '
      'ec/slope for the saturation curve), scaled by a per-channel '
      'coefficient tuned to hit its true ROI. The plot on the right is that '
      "equation's actual output — the same seed-1320 draw the rest of the "
      'deck fits to, at the same basis. The panels below it are the two '
      "channels' actual weekly delivery, so the room can see the inputs and "
      'the outcome side by side. It is not a toy: it is deliberately a '
      'baseline the fitted model structurally cannot represent, with noise '
      'that persists week to week and moves regions together — which is what '
      'competitor activity, pricing and promotions actually do — achieving a '
      'fit ceiling of 0.90.\n\n'
      'IF ASKED why 0.90: we swept it — 0.80, 0.90 and 0.99 side by side. The '
      'saturation result holds across all three and gets worse for the default '
      'prior as the data gets noisier, not better; 0.90 is the deliberate '
      'middle ground, not the near-noiseless extreme. (An earlier, more '
      "convenient version of this DGP drew its baseline from the model's own "
      'spline basis and used independent noise, giving an unrealistic R² of '
      '0.996 — that version was discarded.)\n\n'
      'IF ASKED about the saturation assumption: we are not claiming Channel-1 '
      'half-saturates at 9x its current delivery. That is the stated premise '
      'of the simulation. The question is whether the model can recover a '
      'premise it was given — and it cannot.',
  )


def _build_execution_slide(prs, dgp) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Where the spend actually sits on each channel’s own curve',
      'Same x-axis as slide 2: media measured in multiples of that channel’s '
      'own median week, so 1.0 is "what you already run".',
  )
  _picture_fitted(
      slide,
      os.path.join(FIGURE_DIR, 'results_execution_vs_curve.png'),
      Inches(1.75),
      max_h_in=4.1,
  )
  frame = _textbox(slide, Inches(0.6), Inches(6.05), Inches(12.1), Inches(1.0))
  _para(
      frame,
      f'{channel_labels.label("TV")} is running at {dgp["ceiling_frac_tv"]} of '
      'its ceiling — every week of it sits on the straight part of its own '
      f'curve. {channel_labels.label("Display")}, at '
      f'{dgp["ceiling_frac_display"]}, is up around the bend.',
      size=15,
      bold=True,
      color=ACCENT,
      first=True,
  )
  _para(
      frame,
      'This is the whole setup: one genuinely under-invested channel, one '
      'that is not. No statistics yet — just where the money currently lands.',
      size=12,
      color=MUTED,
  )
  _footnote(
      slide,
      'Each dot is one region-week of simulated delivery, placed on the true '
      'response curve it was generated from.',
  )
  _notes(
      slide,
      'This is the slide that makes the setup physical, before any modelling. '
      "Walk the left panel: the black line is Channel-1's true response "
      'curve. Every dot is a real week of delivery. They all sit down in the '
      'straight part — Channel-1 is nowhere near its ceiling, it has captured '
      'about a tenth of what it could.\n\n'
      'Then the right panel: Channel-2 is a normal, reasonably-invested '
      'channel, sitting up around the knee at roughly 44%.\n\n'
      'That contrast is deliberate. If the model gets both wrong, it is a '
      'broken model. If it gets Channel-2 right and Channel-1 wrong, the '
      'problem is specific to under-invested channels — which is exactly the '
      'case where a planner would be asking "should I spend more here?"\n\n'
      "Callback to slide 2: the x-axis is in multiples of the channel's own "
      'median week. That is the same denomination the default prior is written '
      'in. So when the prior says "you are probably near half-saturation", it '
      'is asserting these dots should be sitting around 1.0 on the curve. For '
      'Channel-1, the truth is 9.',
  )


def _r90_recovery_facts() -> dict[str, object]:
  """Slide 7: the pinned basis -- 10 seeds, both channels, achievable prior.

  Reads `pooled_box_stats.csv`, which
  `scratch_plot_recovery_boxplot_r90_fulldraws.py` writes as it renders the
  figure, so every number the slide quotes describes a box the audience is
  looking at. Not `results_facts`: that module is scoped to the older
  canonical alpha_m=0.8/oracle_r2=0.80 run.

  Two things about this run are load-bearing and easy to lose (see
  `r90_basis`): every true parameter is IDENTICAL across the ten seeds, so
  seed-to-seed spread is noise and nothing else; and the informed arm is
  `ec_alpha_noisy`, whose ec_m and alpha_m anchors are perturbed ~25% per
  seed rather than centred on the truth. An oracle prior would be unbiased by
  construction and beating the default with it would prove little.

  Keys are `{param}_{channel-key}_{variant}_{stat}`, e.g. `ec_ch1_default_med`.
  All values are % error.
  """
  df = pd.read_csv(os.path.join(PINNED_RUN_DIR, 'pooled_box_stats.csv'))
  per_seed = pd.read_csv(os.path.join(PINNED_RUN_DIR, 'per_seed_all.csv'))
  out = {
      'n_seeds': int(len(per_seed)),
      'n_draws': int(df['n_draws'].iloc[0]),
  }
  channel_key = {'TV': 'ch1', 'Display': 'ch2'}
  variant_key = {'default': 'default', 'ec_alpha_noisy': 'informed'}
  for _, row in df.iterrows():
    key = (
        f'{row["param"].replace("_m", "")}_{channel_key[row["channel"]]}'
        f'_{variant_key[row["variant"]]}'
    )
    out[f'{key}_med'] = float(row['median'])
    out[f'{key}_q25'] = float(row['q25'])
    out[f'{key}_q75'] = float(row['q75'])
  # Per-seed win counts exist only on the Channel-1 summary.
  for param in ('ec', 'roi', 'alpha'):
    out[f'{param}_informed_wins'] = int(
        (
            per_seed[f'informed_{param}_err'].abs()
            < per_seed[f'default_{param}_err'].abs()
        ).sum()
    )
    out[f'{param}_default_worst'] = float(
        per_seed[f'default_{param}_err'].abs().max()
    )
  return out


def _build_recovery_table_slide(prs, r90) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Does the model recover the truth it was built from?',
      'Same data, same model, same settings. The only difference between the '
      'two boxes in each panel is the saturation prior.',
  )
  _picture_fitted(
      slide,
      os.path.join(FIGURE_DIR, 'slide7_recovery_boxplot_r90_fulldraws.png'),
      Inches(1.55),
      max_h_in=4.3,
  )

  ch1, ch2 = channel_labels.label('TV'), channel_labels.label('Display')
  frame = _textbox(slide, Inches(0.6), Inches(5.85), Inches(12.1), Inches(1.2))
  _para(
      frame,
      f'The same default prior misses {ch1}’s saturation point by '
      f'{r90["ec_ch1_default_med"]:+.0f}% and {ch2}’s by '
      f'{r90["ec_ch2_default_med"]:+.0f}%. The error tracks how far the truth '
      'sits from the prior — not which channel it is.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
      space_after=6,
  )
  _para(
      frame,
      f'A reach-informed prior — built from ordinary planning inputs and '
      f'~25% wrong — removes that bias ({r90["ec_ch1_informed_med"]:+.0f}% '
      f'median) and wins on {r90["ec_informed_wins"]}/{r90["n_seeds"]} '
      f'datasets, though it is far noisier on any single one. ROI improves '
      f'({r90["roi_ch1_default_med"]:+.0f}% → '
      f'{r90["roi_ch1_informed_med"]:+.0f}%) without being fixed; carryover '
      'is unchanged either way.',
      size=13,
      color=BLUE,
      space_after=6,
  )
  _footnote(
      slide,
      f'{r90["n_seeds"]} simulated datasets with IDENTICAL true parameters — '
      'only the noise differs, so seed-to-seed spread is noise and nothing '
      f'else. Posterior draws pooled across seeds and chains '
      f'({r90["n_draws"]} per box); box widths show combined spread, not one '
      'model’s calibrated credible interval. The informed prior’s anchors are '
      'perturbed ~25% per dataset rather than set to the truth — an oracle '
      'prior would be unbiased by construction. oracle_r2 = 0.9.',
  )
  _notes(
      slide,
      'This is the core evidence slide. A box plot rather than a table so it '
      'reads as a distribution, not a single lucky (or unlucky) draw, and two '
      'rows so the room can see the mechanism rather than take it on '
      'trust.\n\n'
      'THE READ IS DOWN THE FIRST COLUMN, not across. Default misses '
      f'{ch1}\'s saturation point by a median {r90["ec_ch1_default_med"]:+.0f}%'
      f' and {ch2}\'s by {r90["ec_ch2_default_med"]:+.0f}%. Same prior, same '
      'model, same settings, same ten datasets. What differs is where the '
      f"truth sits relative to the prior: {ch2}'s (~1.3x a median week) is "
      f"inside its high-density region, {ch1}'s (9x) is far outside it. That "
      'is why this is a prior problem and not a broken-model problem, and it '
      'is the strongest form of the argument — a model that got everything '
      'wrong would just be dismissed as a bad model.\n\n'
      'WHAT MAKES THIS A CLEAN TEST: every true parameter is identical in all '
      f'{r90["n_seeds"]} datasets. Only the noise differs. An earlier version '
      'of this sweep let the truth drift between draws (true ROI ran 6.6 to '
      '10.9), which mixed "the estimator is noisy" with "the estimator '
      'behaves differently at a different truth" — two questions, one number. '
      'That is fixed; seed-to-seed spread here is noise and nothing else.\n\n'
      'THE INFORMED PRIOR IS NOT AN ORACLE. Its audience and adstock anchors '
      'are perturbed ~25% per dataset, so it is the quality of input a media '
      'team could actually supply. Say this out loud — a prior centred on the '
      'true value is unbiased by construction, and beating the default with '
      'one would prove almost nothing. The cost of honesty is visible: it is '
      f'centred ({r90["ec_ch1_informed_med"]:+.0f}% median, winning '
      f'{r90["ec_informed_wins"]}/{r90["n_seeds"]}) but WIDE. Do not say it '
      '"recovers" the parameter; say it removes the systematic bias.\n\n'
      f'ROI: informed wins {r90["roi_informed_wins"]}/{r90["n_seeds"]} on '
      f'{ch1} and roughly halves the median error '
      f'({r90["roi_ch1_default_med"]:+.0f}% -> '
      f'{r90["roi_ch1_informed_med"]:+.0f}%), but neither box crosses zero. '
      f'On {ch2} the two are indistinguishable '
      f'({r90["roi_ch2_default_med"]:+.0f}% vs '
      f'{r90["roi_ch2_informed_med"]:+.0f}%). DO NOT claim informed "fixes" '
      'ROI, and DO NOT present any single ROI number as characteristic — ours '
      'included. This is the caveat the cut seed-stability slide used to '
      'carry.\n\n'
      'CARRYOVER shows nothing: both priors land around '
      f'{r90["alpha_ch1_default_med"]:+.0f}% on {ch1} and the boxes overlap. '
      f'On {ch2} informed reads worse ({r90["alpha_ch2_informed_med"]:+.0f}% '
      f'vs {r90["alpha_ch2_default_med"]:+.0f}%) but the underlying gap is '
      'about 0.01 of retention — noise, not a finding. Carryover recovery is '
      'governed by max_lag, not by which prior is used, and the posterior '
      'does not follow the adstock anchor at all (r = 0.05) the way it '
      'follows the saturation anchor (r = 0.80).\n\n'
      f'CAVEAT: {r90["n_seeds"]} seeds. Treat the box widths as illustrative '
      'of spread, not a formally calibrated interval.',
  )


def _r90_mroi_facts() -> dict[str, object]:
  """Slide 8: mROI as spend scales, BOTH channels, on the pinned basis.

  Sourced from `scratch_mroi_both_channels_r90.py`'s `mroi_both_channels.csv`,
  which reattaches each saved posterior to a rebuilt scenario rather than
  refitting. Channel-2 is the control: it is what distinguishes "the default
  breaks when you extrapolate" from "the default breaks when you extrapolate
  from a misplaced saturation point".

  Keys: `m{mult}x_{ch1|ch2}_{default|informed}_median` (% error), plus
  `covered_{mult}x_{ch1|ch2}_default` counting seeds whose 90% interval
  contains the truth. Coverage is deliberately computed at 90%, not from the
  50% band the figure draws -- a narrower band excludes the truth more
  readily, which would flatter the claim.
  """
  df = pd.read_csv(os.path.join(PINNED_RUN_DIR, 'mroi_both_channels.csv'))
  fitted = df[df.variant != 'truth']
  seeds = sorted(df.seed.unique().tolist())
  out = {'n_seeds': len(seeds), 'seeds': seeds}
  channel_key = {'TV': 'ch1', 'Display': 'ch2'}
  variant_key = {'default': 'default', 'ec_alpha_noisy': 'informed'}
  for channel, ckey in channel_key.items():
    for variant, vkey in variant_key.items():
      for mult in (1, 2, 3, 5, 10):
        sub = fitted[
            (fitted.channel == channel)
            & (fitted.variant == variant)
            & (fitted.spend_multiplier == mult)
        ]
        if sub.empty:
          continue
        out[f'm{mult}x_{ckey}_{vkey}_median'] = float(sub.pct_error.median())
        # 90% interval, from the stored per-seed quantiles.
        out[f'covered_{mult}x_{ckey}_{vkey}'] = int(
            ((sub.q05 <= sub.true_mroi) & (sub.true_mroi <= sub.q95)).sum()
        )
  # Largest absolute Channel-2 error at any multiplier, either prior -- the
  # single number that says "the control channel never breaks".
  ch2 = fitted[fitted.channel == 'Display']
  out['ch2_worst_abs_median'] = float(
      ch2.groupby(['variant', 'spend_multiplier'])['pct_error']
      .median()
      .abs()
      .max()
  )
  return out


def _build_mroi_slide(prs, mroi) -> None:
  slide = _blank_slide(prs)
  ch1, ch2 = channel_labels.label('TV'), channel_labels.label('Display')
  _title(
      slide,
      'So ask the model the question you actually bought it for',
      '"Should I spend more here?" — marginal return at spend levels above '
      f'today’s, across {mroi["n_seeds"]} datasets. {ch2} is the control.',
  )
  _picture_fitted(
      slide,
      os.path.join(FIGURE_DIR, 'slide8_mroi_r90_pinned.png'),
      Inches(1.5),
      max_h_in=4.6,
  )

  frame = _textbox(slide, Inches(0.6), Inches(6.2), Inches(12.1), Inches(1.0))
  _para(
      frame,
      f'Same prior, same extrapolation to 10x: on {ch1} the default is off '
      f'{mroi["m10x_ch1_default_median"]:+.0f}%, on {ch2} just '
      f'{mroi["m10x_ch2_default_median"]:+.0f}%. The model does not break '
      'when you extrapolate — it breaks when you extrapolate from a '
      'misplaced saturation point.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
      space_after=6,
  )
  _para(
      frame,
      f'On {ch1} the default’s error also changes sign — '
      f'{mroi["m1x_ch1_default_median"]:+.0f}% at today’s spend, '
      f'{mroi["m10x_ch1_default_median"]:+.0f}% at 10x — flattering the '
      'channel today and starving it tomorrow. The reach-informed prior is '
      'biased but stably so, which is the property a reallocation needs.',
      size=13,
      color=BLUE,
  )
  _footnote(
      slide,
      f'{mroi["n_seeds"]} datasets with identical true parameters; shaded '
      'band is the middle 50% of pooled draws, so it mixes posterior and '
      'between-seed spread rather than being one model’s calibrated '
      'interval. At a 90% interval the default contains the truth on '
      f'{mroi["covered_1x_ch1_default"]}/{mroi["n_seeds"]} datasets at 1x and '
      f'{mroi["covered_10x_ch1_default"]}/{mroi["n_seeds"]} at 10x for {ch1}, '
      f'versus {mroi["covered_10x_ch2_default"]}/{mroi["n_seeds"]} for {ch2}. '
      'Y-axes differ by channel.',
  )
  _notes(
      slide,
      'The bridge from the previous slide: if you only ever look at ROI at '
      "today's spend, you will not see this. Budgeting is by definition a "
      'question about changing spend, so it evaluates channels at levels you '
      'have never observed.\n\n'
      'THE TWO PANELS ARE THE ARGUMENT. Do not walk them separately. The '
      'obvious objection to the left panel alone is "maybe your model just '
      'degrades whenever it extrapolates". The right panel kills that: same '
      'prior, same model, same 10x, and the three lines lie on top of each '
      f'other. {ch2} never exceeds about '
      f'{mroi["ch2_worst_abs_median"]:.0f}% median error at ANY multiplier, '
      f'for either prior, while {ch1} reaches '
      f'{mroi["m10x_ch1_default_median"]:+.0f}%.\n\n'
      'IF THE ROOM SKIMS: the right panel can read as "nothing here". That IS '
      'the point — say so explicitly rather than letting them move on. The '
      'control passing is what makes the left panel mean something.\n\n'
      'THE SIGN FLIP is the practitioner-facing detail. Default reads '
      f'{mroi["m1x_ch1_default_median"]:+.0f}% at 1x and '
      f'{mroi["m10x_ch1_default_median"]:+.0f}% at 10x, crossing zero between '
      '1x and 2x. So the same error over-credits the channel historically and '
      'under-funds it going forward. Those are not two mistakes; they are one '
      'mistake seen from both ends.\n\n'
      'THE INFORMED PRIOR IS BIASED HIGH, visibly, at every multiplier '
      f'({mroi["m1x_ch1_informed_median"]:+.0f}% to '
      f'{mroi["m10x_ch1_informed_median"]:+.0f}%). Do not claim it tracks the '
      'truth. The claim is narrower and still decision-relevant: its error '
      'does not grow with the size of the decision. Volunteering this is what '
      'buys credibility for the saturation result.\n\n'
      'ON THE BAND: it is a 50% interval pooled across datasets, chosen for '
      'legibility. Quote coverage at 90% (in the footnote) rather than '
      'reading it off the band — a narrower band excludes the truth more '
      'readily and would flatter the argument for the wrong reason.\n\n'
      'CAUTION: ROI-family measures are the least stable family in this '
      'study. The direction — error grows with extrapolation on the '
      'under-invested channel and not on the other — is the robust part. Do '
      'not present any single ROI magnitude as characteristic, ours included.',
  )


def _build_takeaways_slide(prs, r90, mroi, facts) -> None:
  """Closing slide: what the study established, and what to do about it.

  Two columns on purpose. The left is what the ten datasets showed; the right
  is what a practitioner changes on Monday. Keeping them visually separate is
  what stops an action being read as a finding -- the actions are prescriptions
  we believe follow from the evidence, not things this study measured.

  Every number interpolates from the facts functions the results slides render
  from, so this slide cannot drift away from slides 7 and 8.

  Three claims are DELIBERATELY ABSENT, and each was considered:

    * "the default is also wrong for over-saturated channels". Never tested.
      Channel-2 is well-reached (truth near the prior's mass), not
      over-saturated (truth below it). Nothing here supports the symmetric
      claim, and an ARF room would ask.
    * "inform your adstock prior". The alpha_m priors were indistinguishable
      (see the numbers on the right column). What moved alpha_m was `max_lag`,
      which is a truncation rather than a prior -- so the action is about the
      window, and saying otherwise would contradict our own table.
    * "know your audience size". Audience size cancels out of the derivation:
      what survives is the share of it you reach and how often. Stating it as
      audience size invites a practitioner to gather the one input that does
      not actually enter.
  """
  slide = _blank_slide(prs)
  _title(
      slide,
      'Key takeaways',
      'The defaults are a starting point, not a measurement — and you are '
      'never asked to confirm them.',
  )

  left = _textbox(slide, Inches(0.7), Inches(1.75), Inches(5.7), Inches(4.9))
  _para(
      left, 'WHAT THE TEN DATASETS SHOWED', size=13, bold=True, color=ACCENT,
      first=True, space_after=12,
  )
  for i, (head, body) in enumerate([
      (
          'The saturation prior is the same for every channel.',
          f'It assumes you are already near the knee. On the under-reached '
          f'channel it missed the truth by '
          f'{r90["ec_ch1_default_med"]:+.0f}%.',
      ),
      (
          'It only misleads where your channel sits far from it.',
          f'Same prior, same fit, same {r90["n_seeds"]} datasets: the '
          f'well-reached channel came back {r90["ec_ch2_default_med"]:+.0f}%. '
          'That is a prior problem, not a broken model.',
      ),
      (
          'The error compounds as you extrapolate.',
          f'Marginal ROI ran {mroi["m1x_ch1_default_median"]:+.0f}% at '
          f'today\'s spend to {mroi["m10x_ch1_default_median"]:+.0f}% at 10x, '
          f'with the 90% interval missing the truth on '
          f'{mroi["n_seeds"] - mroi["covered_10x_ch1_default"]}/'
          f'{mroi["n_seeds"]} datasets.',
      ),
  ]):
    _para(left, f'{i + 1}.  {head}', size=14, bold=True, color=BLUE,
          space_after=2)
    _para(left, f'     {body}', size=12, color=MUTED, space_after=14)

  right = _textbox(slide, Inches(6.9), Inches(1.75), Inches(5.8), Inches(4.9))
  _para(
      right, 'WHAT TO CHANGE IN YOUR OWN MODEL', size=13, bold=True,
      color=ACCENT, first=True, space_after=12,
  )
  for i, (head, body) in enumerate([
      (
          'Ask what your model assumes before it sees your data.',
          f'We did: the default gives '
          f'{facts["p_past_half_saturation_today"]:.0%} odds that every '
          'channel — whatever its audience or delivery — is already past '
          'half-saturation.',
      ),
      (
          'Build the saturation prior from delivery.',
          'What share of your addressable audience you reach, and how often. '
          'Audience size alone is not the input — it cancels; the shortfall '
          'against it is what moves the answer.',
      ),
      (
          'Set the carryover window deliberately.',
          f'max_lag is a hard truncation, not a prior, and it moved our '
          f'results. The decay prior alone did not: '
          f'{r90["alpha_ch1_default_med"]:+.0f}% vs '
          f'{r90["alpha_ch1_informed_med"]:+.0f}%, indistinguishable.',
      ),
      (
          'Stress-test at 2–5x spend, not at today\'s.',
          'ROI and marginal ROI at current spend both concealed this. Only '
          'the elevated-spend curve separated the two priors.',
      ),
  ]):
    # Tighter than the left column: four items in the same vertical space.
    _para(right, f'{i + 1}.  {head}', size=14, bold=True, color=BLUE,
          space_after=2)
    _para(right, f'     {body}', size=12, color=MUTED, space_after=10)

  _footnote(
      slide,
      f'An informed prior removes the systematic bias; it does not recover '
      f'the parameter. Its anchors were perturbed ~25% per dataset and it won '
      f'{r90["ec_informed_wins"]}/{r90["n_seeds"]}, but it is wide on any '
      f'single one ({r90["ec_ch1_informed_q25"]:+.0f}% to '
      f'{r90["ec_ch1_informed_q75"]:+.0f}% interquartile) — its accuracy is '
      'inherited from your planning inputs, not estimated from the data.',
  )

  _notes(
      slide,
      'Closing slide, ~90 seconds. Read the left column, then the right.\n\n'
      'MAKE THE CALLBACK EXPLICIT. Right-hand action 1 is the same question '
      'slide 1 opened on. Say so — "we started by asking what your model '
      'assumes before it sees your data; that is the one thing to take home". '
      'The 50% figure beside it is what stops the point being a platitude: it '
      'is what you find the moment you actually look.\n\n'
      'DO NOT let the right column be heard as findings. The left is what we '
      'measured; the right is what we think follows. The carryover bullet in '
      'particular is a prescription about the WINDOW, and the number beside '
      'it is there to stop anyone hearing "inform your adstock prior" — our '
      'own alpha_m priors were indistinguishable, and saying otherwise '
      'contradicts slide 7.\n\n'
      'IF ASKED ABOUT OVER-SATURATED CHANNELS: we did not test that. '
      'Channel-2 is well-reached, meaning its truth sits inside the prior\'s '
      'high-density region — not below it. The symmetric claim is plausible '
      'and unsupported; say so rather than reaching for it. A deeply '
      'saturated channel did become confounded with the baseline in earlier '
      'work, which is why it was dropped, but that is a different finding.\n\n'
      'IF ASKED WHY ROI IS NOT A TAKEAWAY: because roughly two-thirds of the '
      'default\'s ROI overstatement turned out to be baseline '
      'misspecification rather than the prior (the well-specified ablation '
      'takes it from +40% to +15%). Fixing your priors will not fix your ROI '
      'if your baseline is wrong. That is a real limitation and it belongs in '
      'the answer, not in the bullets.\n\n'
      'THE FOOTNOTE IS THE CREDIBILITY OF THE WHOLE TALK. Say it out loud. A '
      'room that hears "and here is what our fix does not do" believes the '
      'rest.',
  )


def main() -> None:
  path = build_deck()
  print(f'Wrote {path}')
  print('\nSlides:')
  print('  1. What does an MMM assume before it sees your data?')
  print('  2. Saturation — the default says you are already there')
  print('  3. Slope — the parameter that is never estimated')
  print('  4. Adstock — a fair prior, and a hard cutoff behind it')
  print('  5. Simulated data with known ground truth')
  print('  6. Where the spend sits on each channel’s own curve')
  print('  7. Does the model recover the truth?')
  print('  8. Marginal ROI as spend scales')
  print('  9. Key takeaways')


if __name__ == '__main__':
  main()
