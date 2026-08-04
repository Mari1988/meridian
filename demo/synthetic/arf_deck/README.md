# ARF Analytics Council deck — results-slide builders

The scripts that build the title slide and slides 9–13 of
`ARF-Analytics-Council-Talk-v2.pptx` (presented 2026-08-04; a title slide was
added afterwards, which is why the deck is 13 slides and the results start at
9 rather than 8). The `.pptx` itself is **not in this repo** — it lives in
the author's OneDrive. These scripts operate on a working copy, so set `DECK` at
the top of each to wherever yours is before running.

They are here because they are the only record of how those slides were made,
and because **every number on them is read from a CSV at build time** rather
than typed in. Changing a number means re-running the producer, not editing
text. That property is worth preserving.

| script | what it builds |
|---|---|
| `build_takeaways.py` | the takeaways slide + headline stat. **Runs first** — it rebuilds the deck from the untouched original |
| `build_slides_8_9.py` | the two recovery slides (both channels) and the mROI slide |
| `build_appendix.py` | the appendix: how to override Meridian's shape priors |
| `build_title.py` | the title slide. **Runs LAST** — it prepends, shifting every later index by one |
| `build_mroi_table.py` | standalone one-slide `.pptx` with the mROI table, to copy in |
| `check_fit.py` | text-fit QA: wraps every box at its real width using the actual Calibri files and flags overflow or collision |

**Run order is load-bearing.** The middle scripts address slides by POSITION,
so `build_title.py` must come last -- prepending a slide first would send every
later write to the wrong slide.

```sh
python build_takeaways.py && python build_slides_8_9.py \
  && python build_appendix.py && python build_title.py
python check_fit.py <deck>.pptx          # all slides; add numbers to check only some
```

## Where the numbers come from

All of it is the **well-specified arm at 50 draws** —
`../fitted_models/scratch_ablation_r90_wellspec/`, whose summary CSVs are
tracked in this repo (the `.nc` posteriors are not; regenerate with
`../run_wellspec_50.sh`, ~90 min, then `../run_wellspec_mroi_50.sh`).

`check_fit.py` walks every slide rather than a fixed list, for the same reason:
a hardcoded index list silently checks the wrong slides the moment anything is
inserted. It assumes 18pt (the theme default) for runs carrying no explicit
size, so it over-flags footnotes -- treat its warnings on slides it did not
build as "look at this", not "this is broken".

It exists because LibreOffice is not available on the author's
machine and PowerPoint's AppleScript export started failing with a sandbox
error mid-project, so the usual render-and-look QA pass was not possible for the
final revisions. It catches text overflow, which is the defect that matters
most here — it does **not** catch alignment, colour or overlap. Look at the
slides before presenting them.
