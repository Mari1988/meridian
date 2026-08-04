# ARF Analytics Council deck — results-slide builders

The scripts that build slides 8–12 of `ARF-Analytics-Council-Talk-v2.pptx`
(presented 2026-08-04). The `.pptx` itself is **not in this repo** — it lives in
the author's OneDrive. These scripts operate on a working copy, so set `DECK` at
the top of each to wherever yours is before running.

They are here because they are the only record of how those slides were made,
and because **every number on them is read from a CSV at build time** rather
than typed in. Changing a number means re-running the producer, not editing
text. That property is worth preserving.

| script | what it builds |
|---|---|
| `build_slides_8_9.py` | slides 8, 9 (recovery, both channels) and 10 (mROI) |
| `build_takeaways.py` | slide 11, the five key takeaways + headline stat |
| `build_appendix.py` | slide 12, how to override Meridian's shape priors |
| `build_mroi_table.py` | standalone one-slide `.pptx` with the mROI table, to copy in |
| `check_fit.py` | text-fit QA: wraps every box at its real width using the actual Calibri files and flags overflow or collision |

Run order matters: `build_takeaways.py` rebuilds the deck from the untouched
original, so it must run **first**; the others edit that output in place.

```sh
python build_takeaways.py && python build_slides_8_9.py && python build_appendix.py
python check_fit.py <deck>.pptx
```

## Where the numbers come from

All of it is the **well-specified arm at 50 draws** —
`../fitted_models/scratch_ablation_r90_wellspec/`, whose summary CSVs are
tracked in this repo (the `.nc` posteriors are not; regenerate with
`../run_wellspec_50.sh`, ~90 min, then `../run_wellspec_mroi_50.sh`).

`check_fit.py` exists because LibreOffice is not available on the author's
machine and PowerPoint's AppleScript export started failing with a sandbox
error mid-project, so the usual render-and-look QA pass was not possible for the
final revisions. It catches text overflow, which is the defect that matters
most here — it does **not** catch alignment, colour or overlap. Look at the
slides before presenting them.
