"""Title slide, prepended as slide 1 of the ARF deck.

Replaces a centred, unstyled slide that did not match the rest of the deck and
misspelled the presenter's name. Built in the deck's own language: left-aligned
at 0.60", Calibri, the C0392B rule under the title, muted 5A5A5A supporting
text.

The hierarchy is deliberately inverted relative to the old slide, which set the
VENUE largest and the subject smaller. The talk's own argument leads; the
council and session name become context above it.

MUST RUN LAST in the deck chain. It prepends a slide, so every later slide
index shifts by one -- `build_slides_8_9.py` and `build_appendix.py` address
slides by position and would write to the wrong ones if this ran first.
"""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt
from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, 'ARF-Analytics-Council-Talk-v2.pptx')
FONTS = '/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/'

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
RED = RGBColor(0xC0, 0x39, 0x2B)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)

EYEBROW = 'ARF ANALYTICS COUNCIL   ·   MMM BEST PRACTICES'
TITLE = 'The default assumptions your Bayesian MMM is making for you'
SUBTITLE = 'What Meridian decides before it sees your data — and what to do about it'
PRESENTER = 'Mariappan Subramanian'          # was misspelled "Mariappn"
ORG = 'The Trade Desk'
DATE = '4 August 2026'

BODY_W = 12.10


def text_width(text, pt, bold):
  f = ImageFont.truetype(FONTS + ('Calibrib.ttf' if bold else 'Calibri.ttf'),
                         int(pt * 4))
  return f.getlength(text) / 4 / 72.0


def textbox(slide, x, y, w, h):
  tf = slide.shapes.add_textbox(
      Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
  tf.word_wrap = True
  tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
  return tf


def run(para, text, size, bold=False, color=INK, spacing=None):
  r = para.add_run()
  r.text = text
  r.font.size = Pt(size)
  r.font.bold = bold
  r.font.color.rgb = color
  r.font.name = 'Calibri'
  return r


def main():
  prs = Presentation(DECK)
  slide = prs.slides.add_slide(prs.slides[0].slide_layout)
  for sh in list(slide.shapes):
    sh._element.getparent().remove(sh._element)

  tf = textbox(slide, 0.60, 2.02, BODY_W, 0.26)
  run(tf.paragraphs[0], EYEBROW, 11.5, bold=True, color=BLUE)

  # 40pt wraps this title to two lines at 12.10" -- intended, and the box is
  # sized for it. Asserted below so a future edit cannot silently overflow.
  tf = textbox(slide, 0.60, 2.42, BODY_W, 1.50)
  run(tf.paragraphs[0], TITLE, 40, bold=True)

  tf = textbox(slide, 0.60, 3.98, BODY_W, 0.34)
  run(tf.paragraphs[0], SUBTITLE, 15, color=MUTED)

  bar = slide.shapes.add_shape(
      MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(4.62), Inches(1.70),
      Inches(0.04))
  bar.fill.solid()
  bar.fill.fore_color.rgb = RED
  bar.line.fill.background()
  bar.shadow.inherit = False

  tf = textbox(slide, 0.60, 4.92, BODY_W, 0.30)
  run(tf.paragraphs[0], PRESENTER, 16, bold=True)

  tf = textbox(slide, 0.60, 5.28, BODY_W, 0.26)
  p = tf.paragraphs[0]
  run(p, ORG, 12.5, color=MUTED)
  run(p, '   ·   ', 12.5, color=MUTED)
  run(p, DATE, 12.5, color=MUTED)

  # Move the new slide (appended at the end) to position 0.
  sld_lst = prs.slides._sldIdLst
  entries = list(sld_lst)
  sld_lst.remove(entries[-1])
  sld_lst.insert(0, entries[-1])

  prs.save(DECK)

  lines = -(-text_width(TITLE, 40, True) // BODY_W)  # ceil
  assert lines <= 2, f'title wraps to {lines:.0f} lines, box holds 2'
  print(f'title slide prepended; deck is now {len(prs.slides.__iter__.__self__._sldIdLst)} slides')
  print(f'  title width {text_width(TITLE, 40, True):.2f}" over {BODY_W}" '
        f'-> {lines:.0f} lines')
  print(f'  eyebrow  {text_width(EYEBROW, 11.5, True):.2f}"')
  print(f'  subtitle {text_width(SUBTITLE, 15, False):.2f}"')


if __name__ == '__main__':
  main()
