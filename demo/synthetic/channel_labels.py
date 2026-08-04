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

"""Presentation-only channel names for the ARF deck.

The deck talks about "Channel-1" and "Channel-2" rather than "TV" and
"Display": the finding is about under-invested channels in general, and naming
a medium invites the audience to argue about that medium instead of the
mechanism.

This is a **rendering** concern only. `TV` / `Display` remain the real keys
everywhere they carry meaning -- `SimulationConfig.channel_names`, the
`rf_source_map` / `plain_source_map` wiring onto the real demo data, the
`media_channel` coordinate inside every saved `.nc` InferenceData, and every
CSV column written by the run scripts. Renaming those would invalidate the
fitted models on disk; renaming at the point of display costs nothing and
needs no refit.

Usage:
  import channel_labels
  channel_labels.label('TV')            # -> 'Channel-1'
  channel_labels.label('TV ec_m')       # -> 'Channel-1 ec_m'  (substring form)
"""

from __future__ import annotations

# Data key -> the name that appears on a slide.
LABELS = {
    'TV': 'Channel-1',
    'Display': 'Channel-2',
}


def label(name: str) -> str:
  """Maps a data-side channel key to its presentation name.

  Falls back to substring replacement so composed strings ('TV marginal ROI',
  'TV ec_m') come out right too. Unknown names pass through unchanged.
  """
  if name in LABELS:
    return LABELS[name]
  out = name
  for key, display in LABELS.items():
    out = out.replace(key, display)
  return out
