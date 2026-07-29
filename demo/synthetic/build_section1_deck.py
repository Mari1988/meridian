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

"""Builds section 1 of the ARF Analytics Council deck as a .pptx.

Section 1 -- "What does an MMM assume before it sees your data?" -- covers three
Meridian defaults (saturation prior, fixed Hill slope, adstock), each contrasted
against Robyn, plus a framing slide.

Slide text and figure annotations both read their numbers from
`prior_plots.default_prior_facts()`, so they cannot drift apart. Speaker notes
are attached to each slide's notes field.

Usage:
  .venv/bin/python demo/synthetic/prior_plots_check.py       # verify numbers
  .venv/bin/python demo/synthetic/build_section1_figures.py  # render figures
  .venv/bin/python demo/synthetic/build_section1_deck.py     # build the deck
"""

from __future__ import annotations

import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu
from pptx.util import Inches
from pptx.util import Pt

import prior_plots


HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(HERE, 'figures')
OUTPUT_PATH = os.path.join(HERE, 'arf_section1_deck.pptx')

SLIDE_W = Inches(13.333)  # 16:9.
SLIDE_H = Inches(7.5)

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)

MERIDIAN_PRIOR_DOC = (
    'https://developers.google.com/meridian/docs/advanced-modeling/'
    'default-prior-distributions#ec_m_and_ec_om'
)
ROBYN_DOC = (
    'https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM'
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


def _notes(slide, text: str) -> None:
  slide.notes_slide.notes_text_frame.text = text


def build_deck() -> str:
  """Builds the four-slide section and returns the output path."""
  facts = prior_plots.default_prior_facts()
  prs = Presentation()
  prs.slide_width = SLIDE_W
  prs.slide_height = SLIDE_H

  _build_framing_slide(prs)
  _build_saturation_slide(prs, facts)
  _build_slope_slide(prs)
  _build_adstock_slide(prs, facts)

  prs.save(OUTPUT_PATH)
  return OUTPUT_PATH


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
      'We can inspect these choices only because Meridian and Robyn are open '
      'source — that is a credit to both. Every closed MMM makes the same '
      'three choices, unauditably.',
      size=14,
      color=MUTED,
  )

  _notes(
      slide,
      'Section 1 opener, ~45 seconds.\n\n'
      'The point of the slide is that the questions exist at all. Most of the '
      'room has never been asked any of the three.\n\n'
      'Say the open-source line out loud. This section is not a criticism of '
      'open tools; it is an argument for the kind of scrutiny only open tools '
      'permit. That framing carries the whole talk and heads off the reading '
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
  _para(
      frame,
      'Robyn makes the same category of choice: its gamma plays this role and '
      'is searched over [0.3, 1], also identically across channels.',
      size=12,
      color=MUTED,
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
  _para(
      frame,
      'Identical Hill equation in both tools (gamma ≡ ec_m, alpha ≡ '
      'slope_m). Robyn searches alpha over [0.5, 3.0], spanning C- through '
      'S-shape; Meridian fixes it at 1.',
      size=12,
      color=MUTED,
  )
  _footnote(
      slide,
      'Meridian: hill(x) = x^slope / (x^slope + ec^slope).   Robyn: '
      f'saturated = 1 / (1 + (gamma/adstocked)^alpha).   {ROBYN_DOC}',
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
      'The Robyn contrast is exact, not analogy — the two saturation '
      'functions are algebraically identical (verified numerically to ~1e-7 in '
      'prior_plots_check.py). Robyn searches alpha over [0.5, 3.0]. Meridian '
      'fixes it at 1. Same equation, opposite decision about whether the '
      'curve’s shape is knowable from data.\n\n'
      'READ THE RIGHT PANEL ALOUD: dots mark peak marginal return. Under the '
      'default it sits at ZERO — the model believes your first impression '
      'was your most productive one, always. At slope 2 the peak is at 0.58x '
      'today’s spend; at slope 3, 0.79x. Those build-up phases are '
      'unrepresentable at any parameter value of the default.\n\n'
      'FAIRNESS CAVEAT — say it: Meridian does let you override slope_m; it '
      'warns that doing so may break the optimizer’s global optimum. The '
      'claim is about the default and about which dimensions each tool '
      'searches by design, not that the door is locked.',
  )


def _build_adstock_slide(prs, facts) -> None:
  slide = _blank_slide(prs)
  _title(
      slide,
      'Assumption 3 — Adstock: flat is not neutral, and the real '
      'assumption is not a prior',
      'alpha_m ~ Uniform(0, 1) genuinely is flat. The problem is elsewhere.',
  )
  _picture_fitted(
      slide, os.path.join(FIGURE_DIR, 'section1_adstock.png'), Inches(1.55)
  )

  frame = _textbox(slide, Inches(0.6), Inches(5.7), Inches(12.1), Inches(1.6))
  _para(
      frame,
      'Flat on the parameter is not flat on the decision: the implied share of '
      f'effect landing in week 0 has median '
      f'{facts["week0_share_median"]:.0%}, with a '
      f'{facts["p_week0_share_ge_70"]:.0%} chance it is mostly immediate and a '
      f'{facts["p_week0_share_le_30"]:.0%} chance it mostly lingers.',
      size=14,
      bold=True,
      color=ACCENT,
      first=True,
      space_after=4,
  )
  _para(
      frame,
      'And the binding assumption is not a prior at all: max_lag = 8 is a hard '
      'truncation, never estimated. At a decay rate of 0.9, '
      f'{facts["carryover_lost_beyond_max_lag_at_alpha_090"]:.0%} of true '
      'carryover falls beyond week 8 and is discarded.',
      size=14,
      bold=True,
      color=ACCENT,
      space_after=4,
  )
  _para(
      frame,
      'Robyn bounds decay per channel type — TV [0.3, 0.8], OOH/print/radio '
      '[0.1, 0.4], digital [0, 0.3]. Meridian applies one Uniform(0, 1) to '
      'every channel.',
      size=12,
      color=MUTED,
  )

  _notes(
      slide,
      'SET THIS UP AS A DELIBERATE CONTRAST. alpha_m ~ Uniform(0,1) genuinely '
      'is flat, and Google describes it as uninformative so the data can '
      'inform decay. There is no skew story here. If the section stopped at '
      'the saturation slide it would read as tool-bashing.\n\n'
      '(a) Flat on the parameter is not flat on what you care about. Nobody '
      'decides about "alpha"; they decide about WHEN advertising works. '
      f'Translated: median {facts["week0_share_median"]:.0%} of effect in week '
      f'0, {facts["p_week0_share_ge_70"]:.0%} chance mostly immediate (>=70% '
      f'week 0), {facts["p_week0_share_le_30"]:.0%} chance mostly lingering '
      '(<=30%). Nearly bimodal on the only question a planner would ask.\n\n'
      '(b) The binding assumption is not a prior at all. max_lag = 8 is a hard '
      'truncation in the model spec — never estimated, no prior, no '
      'posterior uncertainty. At decay 0.9, '
      f'{facts["carryover_lost_beyond_max_lag_at_alpha_090"]:.0%} of true '
      'carryover falls beyond week 8 and is discarded and renormalized away. '
      'If your TV genuinely works over a quarter, the default cannot see it, '
      'and nothing in the output tells you so.\n\n'
      '(c) Robyn bounds decay per channel type, encoding the ordinary planning '
      'knowledge that TV lingers and digital does not.\n\n'
      'THE TAXONOMY IS THE THESIS: some defaults are skewed toward a '
      'conclusion; some are flat on the parameter but misleading on the '
      'decision; some are not priors at all, but hard-coded structure wearing '
      'the costume of neutrality. Land that, then move to the TV case '
      'study.\n\n'
      'CAVEAT for any Robyn claim: Robyn is ridge regression with Nevergrad '
      'search, not Bayesian — those are search bounds, not priors. Say it '
      'before a Robyn user in the room says it for you.',
  )


def main() -> None:
  path = build_deck()
  print(f'Wrote {path}')
  print('\nSlides:')
  print('  1. What does an MMM assume before it sees your data?')
  print('  2. Saturation — the default says you are already there')
  print('  3. Slope — the parameter that is never estimated')
  print('  4. Adstock — flat is not neutral')


if __name__ == '__main__':
  main()
