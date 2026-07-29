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

"""Renders the section 1 slide figures to `demo/synthetic/figures/`.

Usage: `.venv/bin/python demo/synthetic/build_section1_figures.py`

Run `prior_plots_check.py` first -- it verifies every number these charts
annotate themselves with.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import prior_plots


FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
DPI = 200
# 16:9, sized so 11pt text stays legible when projected.
FIGSIZE = (13.33, 5.6)


def _save(fig, name: str) -> str:
  path = os.path.join(FIGURE_DIR, name)
  fig.tight_layout()
  fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  return path


def build_all() -> list[str]:
  """Renders every section 1 figure. Returns the paths written."""
  os.makedirs(FIGURE_DIR, exist_ok=True)
  prior_plots.apply_slide_style()
  paths = []

  # Slide 2 -- saturation.
  fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
  prior_plots.plot_ec_prior_and_curves(ax_left, ax_right)
  paths.append(_save(fig, 'section1_saturation.png'))

  # Slide 3 -- the fixed Hill slope.
  fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
  prior_plots.plot_slope_shapes(ax_left, ax_right)
  paths.append(_save(fig, 'section1_slope.png'))

  # Slide 4 -- adstock, plus the per-channel-bounds contrast.
  fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
  prior_plots.plot_adstock_decay_and_immediacy(ax_left, ax_right)
  paths.append(_save(fig, 'section1_adstock.png'))

  fig, ax = plt.subplots(1, 1, figsize=(13.33, 3.4))
  prior_plots.plot_robyn_theta_comparison(ax)
  paths.append(_save(fig, 'section1_adstock_robyn_bounds.png'))

  return paths


def main() -> None:
  paths = build_all()
  print('Wrote:')
  for path in paths:
    print(f'  {path}')
  print('\nNumbers on these slides:')
  for key, value in prior_plots.default_prior_facts().items():
    print(f'  {key:<44} {value:.4f}')


if __name__ == '__main__':
  main()
