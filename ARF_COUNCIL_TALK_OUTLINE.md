# ARF Analytics Council talk — outline

**Status: DELIVERED 2026-08-04.** The talk was given from
`ARF-Analytics-Council-Talk-v2.pptx` — twelve slides, in the author's OneDrive, **not in
this repo**. It is built by `demo/synthetic/arf_deck/` (slides 8-12) off the
**well-specified arm at 50 draws**, `fitted_models/scratch_ablation_r90_wellspec/`.
CLAUDE.md's "THE LANDED STATE" block is the authority on settings, scripts and objects;
read it before anything below.

Slides 1-6 (defaults, DGP, execution-vs-curve) come from the earlier eight-slide
`demo/synthetic/arf_section1_deck.pptx` pipeline. Slides 7-12 are the results and the
appendix. Sections 1, 3, 4, 6, 7, 8 of the outline below were never built as slides.

**Two things changed between the last outline update and delivery, and both matter:**
(1) the results section moved from the realistic baseline to the **well-specified** one,
so every ROI number below that is labelled "realistic" is no longer what was shown;
(2) the sweep went from 10 draws to 50, which **retracted** the Channel-2 `alpha_m`/ROI
miss recorded below as a small-sample artifact. Everything below this block predates
delivery and is kept for provenance.

**Changes on 2026-08-02 — a SECOND arm, not a replacement. The deck still renders from
the realistic-baseline run described below; nothing here supersedes it.**

- **`scratch_ablation_r90_wellspec.py`** re-runs the same 10 seeds, same truths, same
  seeded anchor perturbation, with a baseline the fitted spline represents exactly
  (`mu_ar1_sd=0`, `seasonal_amplitude=0`; oracle R² held at 0.9 by rescaling the
  geo-level residual). See CLAUDE.md's deck section for the full record.
- **The ROI numbers in §5 and on slide 8 are baseline-conditional and must be said as
  such.** Channel-1's default ROI overstatement is **+40.3% on the realistic baseline and
  +15.4% on a well-specified one**; informed goes **+20.6% → +1.9%**. Roughly two-thirds
  of the default's ROI error is baseline misspecification, not the prior. `ec_m` barely
  moves (−67.7% → −64.2%), which is the argument for leading on it.
- **Adstock on Channel-1 was largely a baseline artifact** (+15.2% → +1.2% default): a
  persistent national AR(1) shock reads as carryover. The `max_lag` discussion below is
  still true but is not the whole story.
- **Two things that cut against us, say them:** with the confound removed the informed
  arm's `ec_m` tracks its anchor at r = 0.96, slope 1.19 — its accuracy is inherited, not
  estimated. And `alpha_m` misses in *both* arms on Channel-2 (−3.3 / −2.1 percentage
  points against a true 0.15), unexplained.
- **Use the pair, not the replacement.** "The saturation failure survives even when the
  model is handed a baseline it can fit perfectly" removes the "your simulation was
  adversarial" objection, which is the strongest attack currently available on §5.

**Changes on 2026-08-01, read before quoting the deck. The results were RE-RUN;
every figure below this line supersedes anything above it.**

- **The seed sweep was rebuilt because it varied two things at once.** True ROI was
  drifting between seeds (Channel-1: 6.60 / 10.88 / 8.35) because `target_roi_m`
  normalises by a *cross-channel* geometric mean, so Channel-2's per-seed `alpha_m`
  draw moved Channel-1's true ROI. A sweep whose estimand moves cannot separate "the
  estimator is noisy" from "the estimator behaves differently at a different truth".
  `demo/synthetic/r90_basis.py` now pins every truth for both channels and asserts it
  on every build. **10 seeds**, identical truths, only noise varies.
- **The informed prior is no longer an oracle.** It was centred on the exact true
  `ec_m`/`alpha_m`, which no media team could supply and which is unbiased by
  construction. Both anchors are now perturbed ~25% per seed. **This is the single
  most important change for credibility, and it costs accuracy — say so.**
- **Headline results at the pinned basis (10 seeds, achievable prior):**
  Channel-1 `ec_m` default **−69%** median (wrong on 10/10, range −56 to −76) vs
  informed **−4%** median (10/10 wins, but range −53 to +43 — centred, not precise).
  Channel-2 `ec_m` default **+5%**. ROI default +44% → informed +21%. Adstock
  unchanged either way (~+17% both).
  mROI: Channel-1 default **+13% at 1x → −58% at 10x** with the 90% interval
  containing the truth on **0/10** datasets at 5x and 10x; Channel-2 never worse than
  **6%** at any multiplier for either prior.
- **The mechanism, measured:** the informed posterior's `ec_m` error tracks its anchor
  error with **slope 0.98, r = 0.80** — one-for-one, no meaningful shrinkage from the
  data. Adstock does *not* follow its anchor (r = 0.05), so the likelihood does pin
  that down. This is the thesis on both arms: the model relays the saturation
  assumption rather than estimating it. Caveat: R² = 0.63, so a good anchor is no
  guarantee on any single dataset (seeds 42 and 99 had near-correct anchors and still
  came out +43% / +31%).
- **Slide 9 (response curves) was cut** — slide 8's two mROI panels make the same
  point in the planner's own units, and make it on two channels.

**Superseded by the above:**
- **Slides 5, 7, 8, 9 are on the r90 basis** (`alpha_m=0.3`, `oracle_r2=0.9`, 3 seeds:
  1320, 7, 42), not the `alpha_m=0.8`/`oracle_r2=0.80` ten-draw run this outline's §5
  discusses. Slide 6 is the only results slide still on the canonical run, and only
  because its ceiling fractions depend on `ec_m` alone, which is identical across the two.
- **The seed-stability slide was cut** (was slide 10). Slide 7's box plot carries the
  replication story. **Its unique job — "do not quote any single ROI figure as
  characteristic" — now has no slide of its own**; say it aloud on slide 8, or the
  saturation claim loses the honesty that made it credible.
- **The deck ships with no speaker notes.** They are written by hand. The guidance text
  survives in `build_section1_deck.py` behind `EMIT_NOTES = False`.
- **Channels are named Channel-1 (was TV) and Channel-2 (was Display)** on every slide, so
  the argument reads as being about under-invested channels rather than about television.
  The data side still uses `TV` / `Display` — mapped by `demo/synthetic/channel_labels.py`.
- ~~Headline numbers at the 3-seed r90 basis: Channel-1 `ec_m` `-71%` to `-52%`
  (default) vs `-4%` to `+8%` (informed); `roi_m` median `+32%` → `+14%`; default mROI
  `+8%` at 1x → `-47%` at 10x; response curves crossing at `4.1x`.~~ **All from the
  drifting-truth sweep with an oracle prior. Do not quote.**

> **⚠ The results slides are built from a different, harder DGP than §5 below describes.**
> §5's numbers come from `final/roi-vs-mroi-metric-selection.ipynb`, whose simulation
> flatters the model in two ways since corrected: the baseline was drawn from the model's
> own spline basis, and the noise was independent week to week. The deck's slides 5–10 use
> `realistic-baseline-noise-2ch.ipynb` (+ `-seeds`) instead: a baseline the fitted spline
> structurally cannot represent, persistent cross-region noise, and an achievable R² of
> 0.80 rather than 0.996. **Where the two disagree, the deck is current and §5 is
> superseded** — see "What the realistic DGP changed" below.

**Context:** invited by Sable (ARF) to present to the ARF Analytics Council, currently focused on "unpacking key challenges and emerging solutions in MMM." Session scheduled for Tuesday, August 4 at 11:00am PT / 1:00pm CT / 2:00pm ET. Council mission: identify common data uses in the current martech environment, explore evolving methodologies, and develop guidelines/best practices across audience development, channel optimization, and cross-media measurement — with a core goal of demystifying/democratizing the "black box" of analytics.

**Source material:** `demo/synthetic/data_simulator.py` (the synthetic ground-truth data generator) and `demo/synthetic/meridian_tv_underreach_case_study.ipynb` (the sharper case study), plus the companion `demo/synthetic/meridian_ec_prior_case_study.ipynb`. See the "Synthetic ground-truth prior-recovery study" section of `CLAUDE.md` for the full technical map.

For §5–6 specifically, the metric-selection evidence lives in
`demo/synthetic/final/roi-vs-mroi-metric-selection.ipynb` (which metric exposes the failure,
and which conceals it) — **this is the landed final notebook; every §5 number below is read
from its executed outputs**. It is *itself* the real-data-scaled scenario — it builds on
Meridian's own `geo_media_rf.csv` demo data via `build_real_augmented_input`, and it carries
the ROI/curve-shape coupling (`roi_ec_elasticity=-0.3`) that used to require a separate
notebook. **There is no longer a second notebook to cite for §5–6**; the earlier
`final-tv-test-under-real-demo-data-roi-shape-coupling.ipynb` is superseded and now sits in
`demo/synthetic/archive/` under its corrected name. The low-ROI
robustness run is in `demo/synthetic/fitted_models/lowroi_metric_selection/` (CSVs +
`run.log`; gitignored, regenerate rather than cite from memory).

**Two changes in the final notebook that reset earlier numbers — read before quoting anything:**
- All fits now use `media_prior_type='roi'`, **Meridian's own default**. Earlier runs used
  `'coefficient'`, which Meridian warns against on every fit; critiquing defaults while using
  a non-default parameterization was a fair procedural objection. Switching *sharpens* nearly
  every result. Any figure in this outline not marked as re-read from the final notebook is
  from the superseded `'coefficient'` run.
- TV's true `ec_m` is now pinned at **9.0**, inside the default prior's `[0.1, 10]` support,
  rather than 11.069 which sat outside it. This removes the objection *"you picked a truth
  your opponent's prior forbids"* at a cost of a few points of headline error. The mechanism
  to lead with is the prior's **concentration, not its truncation**: 99% of its mass sits
  below `ec_m` 2.724, and only 0.0039% above 4.0, so the density at either 9.0 or 11.069 is
  negligible either way.

**Finding, one sentence:** Meridian's (and MMM tooling generally) default `ec_m` (half-saturation) prior implicitly assumes a channel's current spend is already near its saturation point; for a channel that's genuinely under-reached *and* under-frequency'd (e.g. a large-audience TV buy running well below its addressable audience, at low frequency), that assumption is systematically wrong — and a reach/frequency-informed prior, built from ordinary media-planning inputs, fixes it.

---

## Title options
- *"Is Your MMM Telling You You're Saturated When You're Not?"*
- *"The Hidden Assumption in Your MMM's Default Settings"*
- *"When 'Optimized' Isn't Optimal: A Prior Problem in Marketing Mix Modeling"*

## Outline (~20 min talk + discussion)

### 1. Hook — the business problem (2 min)
Open with the scenario, not the statistics: a channel with a large addressable audience, but currently reached at low reach *and* low frequency — genuinely under-invested by any media-planning definition. Ask the room: *if your MMM concluded this channel was already near its ceiling, would you know to question that?* Most wouldn't — the model looks confident either way.

### 2. What's actually inside the black box (4 min) — **BUILT**

Four slides, `demo/synthetic/arf_section1_deck.pptx`. Speaker notes:
`demo/synthetic/arf_section1_notes.md`. Figures regenerate from
`demo/synthetic/build_section1_figures.py`; every quoted number is asserted by
`demo/synthetic/prior_plots_check.py`. Each Meridian default is contrasted against
Robyn so the section reads as cross-tool methodology, not a critique of one tool.

**Slide 1 — every MMM must assume something.** Three questions a model answers before
reading a row: how fast does response flatten, what shape is the curve, how long does an
impression keep working. You are never asked; they ship as defaults. Say aloud that we
can audit this *only because* Meridian and Robyn are open source.

**Slide 2 — saturation.** `ec_m ~ TruncatedNormal(0.8, 0.8, [0.1, 10])`. The unlock is
the units: Meridian scales media by that channel's own median non-zero per-capita
execution, so x = 1.0 means "what you already run." Therefore the default gives **50%**
odds you are already past half-saturation and **92%** odds you have captured more than a
third of the channel's maximum effect — identically for every channel. Robyn's `gamma`
does the same job over `[0.3, 1]`, also channel-agnostic.

**Slide 3 — the Hill slope is never estimated.** `slope_m = Deterministic(1.0)`: no
prior, because it is not a random variable. Fixing it at 1 means diminishing returns from
the first impression — no build-up, no threshold. The documented rationale is
*tractability* (concave curves guarantee the budget optimizer a global optimum), not a
claim about advertising. Robyn's Hill is algebraically identical (`gamma` ≡ `ec_m`,
`alpha` ≡ `slope_m`, verified to ~1e-7) but Robyn *searches* alpha over `[0.5, 3.0]`.
Peak marginal return: x = 0 at slope 1, x = 0.58 at slope 2, x = 0.79 at slope 3.

**Slide 4 — adstock, deliberately the contrasting case.** `alpha_m ~ Uniform(0,1)` really
is flat, so there is no skew story. Instead: (a) flat on the parameter is not flat on the
decision — implied share of effect in week 0 has median **50%**, with **30%** odds of
mostly-immediate and **28%** odds of mostly-lingering; (b) the binding assumption is not a
prior at all — `max_lag = 8` is a hard truncation, and at decay 0.9 **39%** of true
carryover is discarded; (c) Robyn bounds decay *per channel type* (TV `[0.3, 0.8]`,
OOH/print/radio `[0.1, 0.4]`, digital `[0, 0.3]`).

**The taxonomy is the thesis:** some defaults are skewed toward a conclusion, some are
flat on the parameter but misleading on the decision, and some are hard-coded structure
wearing the costume of neutrality. That sets up §§3–7.

*Guardrails (in the notes file):* Robyn is ridge + Nevergrad, not Bayesian — its ranges
are search bounds, not priors. Don't claim these defaults are undocumented; Google
documents both the scaling and the rationales. Meridian permits overriding `slope_m`; the
claim is about the default. No reach/frequency in this section.

### 3. Why under-reach + under-frequency is the sharpest version of this problem (3 min)
- Two independent, compounding forms of under-delivery: not enough people reached, *and* not enough frequency per person reached.
- Walk through the scenario numbers (large audience, ~10% reach, ~1.5x/week frequency) and show the gap between "where the model's default assumes saturation" and "where the data suggests it actually is."

### 4. Why we built a simulation instead of pointing at a real client (2 min)
- You can't validate "is my MMM right about saturation" against real data — you never observe the *true* saturation point in the real world, only the fitted one.
- So: build synthetic geo x time data from a fully known generative model (known audience sizes, reach, frequency, true half-saturation, true ROI), fit MMM to it, and check whether it recovers the truth it was built from.
- **Caveat to state explicitly, up front:** the *specific* saturation threshold used to build the simulation is a stylized assumption, not a claim about real-world saturation levels. The point being tested is the *mechanism* (does the prior distort recovery when truth is far from the default's assumption), not that specific number.

### 5. The finding (5 min — the core section)

> **SUPERSEDED — read this first.** Every number in the six beats below comes from
> `final/roi-vs-mroi-metric-selection.ipynb`, whose DGP drew the time-varying baseline
> from the *same 8-knot spline basis the model fits* (`n_knots_mu_t=8`) and used iid
> residuals, giving an unrealistic R² of 0.996. The rebuilt DGP
> (`realistic-baseline-noise-2ch.ipynb`, `realistic_baseline.py`) removes both
> advantages and lands at an achievable R² of 0.80. **The findings survive and sharpen**,
> but the figures move, and two claims below are now retired outright:
>
> - **Beat 2's "ROI looks fine" hook does not hold on the realistic DGP.** The default's
>   TV ROI error is +79.0% there, not +13.7%, and its interval excludes the truth — a
>   practitioner would catch it. Worse, across ten simulator draws the ROI error ranges
>   −12.7% to +181.7% for the default and −27.7% to +78.6% for the informed prior. **Do
>   not quote any single ROI figure as characteristic**, ours included.
> - **The `alpha_m` prior earns nothing.** Across ten draws the informed prior is better
>   on alpha in only 3 of 10 — chance. The entire benefit comes from the `ec_m` prior, so
>   §7's ask drops to *one* elicited quantity, not two.
>
> Current headline figures, all from the two-channel realistic DGP at oracle R² = 0.80:
> TV `ec_m` −89.4% (default) vs −0.8% (informed); mROI at 3× −65.5% with the default's
> 90% band **excluding** the truth at 2×, 3×, 5× and 10×; across ten draws default `ec_m`
> −66.5% to −90.0% against informed −4.7% to +3.1%, non-overlapping.
>
> **Slides 5–10 of `demo/synthetic/arf_section1_deck.pptx` are built from these numbers.**

**Six beats, in this order.** The sequence matters more than any single number: the
"ROI looks fine" beat is not a concession, it is the hook. Do not cut it (see the
"why not just show mROI" note below).

**Beat 1 — the mechanism is broken.** TV has captured **10.0%** of its ceiling effect at
current delivery. The default-prior model believes **26.0%** — it thinks TV has banked
roughly 2.6x as much of its available effect as it really has. (True `ec_m` 9.0 vs.
fitted 2.84; ceiling fraction is `1/(1+ec_m)` since media is scaled by its own median. The
informed prior lands at 9.14, i.e. 9.9%.) Pull the response-curve overlay here — the
default curve visibly flattens far earlier.

**Beat 2 — and yet every standard diagnostic is clean.** ROI comes back at **+13.7%**
against truth. R-hat is fine. The model summary is fine. *This is the hook.* Say plainly:
if you had this model in production you would have no reason to suspect anything.

**Beat 3 — one line on why.** Not a stats lesson. One sentence and one picture:

> The likelihood only sees what media actually did at the spend you actually ran. It can
> fit that with a small saturation point and a small coefficient, or a large one of each —
> those are indistinguishable in your data. Both reproduce today's ROI. They disagree
> completely about tomorrow's.

The picture is two response curves through the same observed point, diverging past it.
Evidence in reserve if pushed (from the ROI 3/2/1 signal-matched run): `ec_m` off −65%,
`beta_gm` off −53%, ROI off only +15.4% — the coefficient absorbs almost the whole
saturation error. Across posterior draws `corr(beta, ec)` is +0.64, i.e. the sampler is
walking a ridge, not exploring a well. **Keep this in backup, not on the slide.**

**Beat 4 — so ask the question you built the model for.** Should I spend more here?
Evaluate marginal ROI at elevated spend, where the concealed error surfaces:

| spend | true mROI | default | error | default 90% interval | covers truth? |
|---|---|---|---|---|---|
| 1x | 5.977 | 5.550 | **−7.1%** — small enough to shrug off | (4.97, 6.14) | **yes** |
| 2x | 4.933 | 3.514 | **−28.8%** | (3.03, 4.05) | **no** |
| 3x | 4.150 | 2.442 | **−41.2%** | (2.00, 2.91) | **no** |
| 5x | 3.066 | 1.388 | **−54.7%** | (1.04, 1.71) | **no** |
| 10x | 1.698 | 0.540 | **−68.2%** | (0.38, 0.72) | **no** |

The error is small where the data is and compounds monotonically with extrapolation. Note
it is *not* zero at 1x under the `'roi'` default — say "small," not "invisible."

**The interval columns are the beat, not the error columns.** Computed in §5 of
`demo/synthetic/final/roi-vs-mroi-metric-selection.ipynb` through Meridian's own
`Analyzer.marginal_roi(new_data=...)`, so the whole calculation is the library's, not ours.
The single sentence to say out loud:

> The default model's 90% interval covers the truth at exactly one spend level — the one
> you already have data for. At every level where you'd actually make a decision, it
> excludes the truth entirely.

`ec_alpha_only` covers the truth at **5 of 5** levels. That contrast — 1-of-5 vs. 5-of-5 —
is what makes this "confidently wrong" rather than "uncertain," and it is the claim the
section's framing sentence has been resting on. The slide is the banded chart at
`fitted_models/mroi_metric_selection/ec9_mroi_intervals.png`: the default's band separates
from the truth after 1x and never rejoins.

Point estimates shifted by ≲0.4pp versus the earlier posterior-mean construction (3x was
−40.9%, now −41.2%) because these come from Meridian's analyzer rather than the hand-rolled
curve. **Quote the interval version.** The agreement is itself worth one line in backup: it
means the original §5 math was right and only lacked uncertainty.

**Beat 5 — the informed prior fixes it.** `ec_alpha_only` tracks truth to within ~2%
across the entire sweep (−2.0% at 1x, −1.6% at 3x, −0.8% at 10x) — and it gets *more*
accurate as you extrapolate, while the default gets worse. Leads into §7.

**Beat 6 — and it isn't a rigged choice of truth (the invariance result).** The strongest
single slide in the study, and the answer to *"you engineered a truth your prior couldn't
reach."* Two scenarios differing in exactly one thing — TV's true `ec_m`, 11.069 vs. 9.0 —
with **bit-identical media execution** (asserted in the notebook, not assumed; the knob is
`saturation_frequency`, which moves `ec_m` linearly while leaving impressions untouched).
Ask: when the truth moves, does the posterior follow?

| | true `ec_m` moves | fitted moves | fraction of the move absorbed |
|---|---|---|---|
| truth | −18.7% | — | 1.00 |
| `default` | −18.7% | −1.3% | **0.07** |
| `ec_alpha_only` | −18.7% | −18.5% | **0.99** |

Say it plainly: **the default posterior lands in nearly the same place regardless of what
generated the data. It is reporting its prior, not estimating the parameter.** This claim
does not depend on which truth was picked, which is why it survives the objection that any
single error percentage does not.

**Framing sentence for the section:** *the model isn't just uncertain — it's confidently
wrong, it's wrong in the direction of telling you not to spend more, and it's wrong
invisibly.*

#### Why not just show the mROI story and skip ROI?
Considered and rejected. Opening on "mROI at 3x is off by 28%" invites the obvious
question — *does your model get ROI wrong?* — and if the answer is no and you didn't
volunteer it, it reads as having gone hunting for the metric that failed. Said up front,
the same fact is the strongest asset in the talk: it converts "Meridian is wrong" (invites
methodological defense) into "your MMM is silently wrong exactly where you use it"
(invites action).

#### Do not overclaim
Say *ROI **can** look fine while the saturation curve is badly wrong*, never *ROI is
reliably recovered*. Counter-example from our own runs: in the low-signal variant
(media 13.7% of revenue) the default's TV ROI was off only +2.0% but **Social's was off
+43.9%** — and it still passed a 90% HDI coverage check, because the interval was wide
enough to swallow the truth. The weaker claim is the one the evidence supports and it is
sufficient for the argument.

#### Metric ranking (backup slide, or hand-out)
| rank | metric | default error | verdict |
|---|---|---|---|
| 1 | mROI at 3x spend | −41.2%, 90% interval (2.00, 2.91) excludes truth 4.150 | Sharpest, and it *is* the planner's question. Quote the interval, not the point error. |
| 2 | mROI/ROI at current spend | 0.733 vs. true 0.897 | No extrapolation needed; `ec_m` in decision language |
| 3 | `roi_m` | +13.7% | Weak — interval nearly touches truth |
| 4 | mROI at current spend | −7.1% | **Least discriminating of the four. Do not lead with it.** |

(The earlier "incremental-outcome headroom at 3x, −16.1%" row is dropped: it came from the
superseded `'coefficient'` run and is not recomputed in the final notebook. Re-derive it
before using it.)

Rank 4 is worth a sentence out loud, but **state it carefully** — the earlier "it cancels
to ~1.0" line overclaims and is no longer supported. What the final run shows is *partial*
cancellation: the default overstates ROI (+13.7%) and understates the local elasticity
(0.733 vs. 0.897), and because `mROI = ROI x elasticity` the two errors point opposite ways
and partly offset, leaving −7.1% instead of the +13.7% you would otherwise see. Under the
old `'coefficient'` parameterization the cancellation was near-exact; under the `'roi'`
default it is partial. **The defensible claim is that this metric is consistently the least
discriminating — never that it always reads as near-zero error.**

#### What the realistic DGP changed — read before quoting anything above

The deck's slides 5–10 are built on `realistic-baseline-noise-2ch.ipynb` (+ `-seeds`),
which removes the two ways the DGP above flattered the model: the baseline is no longer
drawn from the model's own spline basis, and the noise is persistent and correlated across
regions rather than independent. Achievable R² falls from 0.996 to 0.80. Everything in this
section was recomputed there. Four things changed.

**1. The saturation failure got worse, and it is seed-independent.** Fitted TV `ec_m` under
the default is 0.958 against a truth of 9.0 (−89.4%), versus 2.84 (−68%) on the old DGP.
Across ten draws the default lands between **−66.5% and −90.0%** and the reach-informed
prior between **−4.7% and +3.1%** — the distributions never overlap. Quote the range, not
seed 1320's −89.4%, which sits at the pessimistic edge.

**2. The mROI sweep is sharper and now has credible intervals.** Superseding beat 4's
table: default error is −7.0% at 1x, −49.4% at 2x, **−65.5% at 3x**, −78.5% at 5x, −87.9%
at 10x, and the default's 90% band **excludes the truth at 2x and beyond** while containing
it at 1x. That substantiates "confidently wrong, not merely uncertain" at elevated spend
for the first time. The reach-informed prior tracks the truth to within 1% at every
multiplier.

**3. Beat 2's "every standard diagnostic is clean" is no longer available.** On this DGP
the default's TV ROI is **+79%** and its interval excludes the truth — a practitioner would
catch it. The invisible-failure framing has to be rewritten. The replacement is easier to
land anyway: *the default prior overstates TV's ROI and understates its saturation point;
one informed prior fixes the saturation point.*

**4. Do not quote any ROI figure as characteristic.** Across DGP variants the default's TV
ROI error runs +13.7% (old DGP), −22.5% (3-channel realistic), +79.0% (2-channel
realistic); across ten draws of the last of these the *informed* prior's own error spans
−27.7% to +78.6%, and on one draw it is worse than the default. The informed prior wins on
9 of 10 draws and halves the median error, so the direction is real — the magnitude is not
quotable. Slide 10 says this out loud, which is what makes the saturation claim credible.

**Also retired:** the reach-informed prior's `alpha_m` component earns nothing. Across ten
draws it beats the default on carryover 3 times out of 10 — chance. The entire benefit
comes from the `ec_m` prior, which simplifies §7's ask to a single elicited quantity.

**Not yet recomputed on the realistic DGP:** the budget-reallocation table in §6, the
`ec_noisy` robustness run in §7, and the metric ranking above. Treat those as provisional.

### 6. Does the mistake actually change a decision? (3 min)
- Yes — pull the budget-reallocation table: the default-prior fit recommends materially less spend on the under-reached channel than the true optimum; an audience/reach-informed prior tracks the true optimum instead.
- This is the "so what": it flows straight into a lower budget recommendation for exactly the channel that should be getting more.
- **Have the pushback answer ready:** *"if current-spend ROI is well-identified and I budget on ROI, why do I care?"* Because budgeting is by definition a question about *changing* spend — a reallocation evaluates channels at spend levels you have not observed, so current-spend ROI is never sufficient for the decision it is being used to make. This is the cleanest bridge from §5 into §6.

### 7. The fix, and why it's practical (3 min)
- You don't need to know the true saturation point exactly — an *approximately right*, audience/reach-informed prior (built from third-party audience data, planned/delivered frequency, category effective-frequency norms, or prior incrementality tests) still dramatically outperforms defaulting to "assume you're already near saturation."
- Show the robustness result: even a noisy version of the informed prior (~25% off) still beats default.
- Tie to a concrete, elicitable ask: *"what's your rough audience size and typical frequency for this channel"* is a question most media teams can already answer.

### 8. Discussion / ask of the council (2–3 min)
- Frame as an invitation, matching their "guidelines and best practices" mission: should MMM vendors/practitioners adopt a norm of *requiring* an audience/reach-informed saturation prior per channel, rather than defaulting silently?
- Possible council output: a checklist item for MMM vendor evaluation / a best-practice note on prior elicitation.

### Backup slides (for Q&A, not the main flow)
- How the simulation is built (adstock/Hill mechanics, why reach × frequency was used, per-channel parameters).
- The 90%-credible-interval coverage study across many simulated datasets (the calibration evidence).
- Explicit statement of assumptions and their limits (the "50%" caveat, spelled out fully) — have this ready in case someone pushes on "how do you know 50% is right": we don't claim it is — that's exactly the point.
- **The identification argument in full** (§5 beat 3): `ec_m` −65%, `beta_gm` −53%, ROI +15.4%; `corr(beta, ec)` +0.64 across draws; `beta/ec` better determined than `ec` alone (CV 0.074 vs. 0.094). The conserved quantity is `beta * hill(x)` at *observed* delivery, not `beta/ec` — the ratio is only its leading-order approximation and breaks once the fitted `ec` is small enough to bend the curve inside the observed range. That residual curvature mismatch is exactly what explodes under extrapolation.
- **Robustness: does this depend on the ROI levels assumed?** No. Re-run at true ROI 3/2/1 instead of 8/5/3, holding media's share of revenue fixed at ~31%: `ec_m` −65% (vs. −63%), ROI +15.4% (vs. +16.9%), mROI at 3x −31.0% (vs. −27.9%), at 10x −60.7% (vs. −57.9%). Every conclusion holds at a third of the ROI. **Both sides of these parentheticals are from the superseded `'coefficient'` run** — the comparison is still internally valid (like vs. like), but the baselines no longer match §5, whose current figures are −40.9% at 3x and −68.2% at 10x. Re-run the low-ROI variant under `media_prior_type='roi'` before quoting this alongside §5. What *does* matter is signal strength — leaving `baseline_scale` untouched drops media from 31% to 13.7% of revenue and degrades both variants (the informed prior included, +19% to +29% mROI error). Frame any weak-signal caveat as a signal-to-noise limitation, not an ROI-level one.
- **Why coverage checkmarks can mislead.** Wider posteriors earn checkmarks. Two live examples: default's Social ROI at +43.9% error passing on a 1.47–3.27 interval; default's Display ROI at 23.978 vs. a true 3.875 passing on a 3.84–37.36 interval. If any table with ✓/✗ marks goes on a slide, pair it with point errors.

---

## Open items / TODO
- [ ] Re-run `meridian_tv_underreach_case_study.ipynb` at full MCMC settings (`n_adapt=2000`, `N_SEEDS=20`, etc.) before pulling numbers/charts into slides — figures currently referenced in this outline are from reduced-precision smoke-test runs.
- [x] ~~Turn this outline into an actual slide deck.~~ **9 slides built**
      (`demo/synthetic/arf_section1_deck.pptx`): §2's four assumption slides, plus five
      results slides — the DGP and why it is a fair test, where delivery sits on each
      channel's true curve, parameter recovery, marginal ROI as spend scales, and the
      response curves. (The tenth, stability across draws, was cut on 2026-08-01.)
      Sections 1, 3, 4, 6, 7, 8 still to build. Every quoted number is asserted by
      `prior_plots_check.py`; the full rebuild sequence is in `CLAUDE.md` under "The ARF
      deck" — it is no longer a single figures script, since slides 5 and 7–9 render from
      the r90 scratch scripts.
- [ ] **Write the speaker notes.** The deck now ships with empty notes fields by choice.
      The old generated notes remain in `build_section1_deck.py` (suppressed by
      `EMIT_NOTES = False`) and are worth reading first — they hold the do-not-overclaim
      guidance, not just narration.
- [ ] Decide on final framing: general MMM-methodology point vs. anything Meridian-specific (lean general — avoid reading as criticism of a specific open-source tool).
- [x] ~~**§5 beat 4 needs credible intervals before it goes in the deck.**~~ **Done** —
      §5 of `demo/synthetic/final/roi-vs-mroi-metric-selection.ipynb` computes the sweep through
      Meridian's own `Analyzer.marginal_roi(new_data=...)`, reattaching the saved `.nc`
      posteriors so it runs in seconds without refitting. Result: `default` covers the truth
      at 1 of 5 spend levels, `ec_alpha_only` at 5 of 5. The framing sentence is now
      substantiated rather than asserted. Point estimates agree with the old posterior-mean
      construction to ≲0.4pp, so no conclusion changed.
- [x] ~~Merge the intervals notebook back into `final/roi-vs-mroi-metric-selection.ipynb`.~~
      **Done** — §5 now computes the sweep through `Analyzer.marginal_roi(new_data=...)` and
      reports 90% posterior intervals; the stale §5 caveat and §7 caveat 2 are gone. The
      notebook was re-executed end-to-end and the refit is bit-for-bit reproducible: every
      figure quoted in this outline is unchanged. `final/` now holds exactly one notebook,
      and the standalone intervals notebook is in `archive/`.
- [ ] **Re-check beats 1–5 against beat 6's scenario labelling when building slides.** The
      final notebook's primary scenario is `ec9` (true `ec_m` 9.0); `ec11` (11.069) now
      exists only as the invariance comparison. Do not mix figures across the two.
- [x] ~~**§5 numbers come from a single simulator draw (`SIM_SEED = 0`).** Either run a
      handful of seeds and quote a range, or state "single scenario" on the slide.~~
      **Done, and the mechanism was broken.** `SIM_SEED` never did anything:
      `GeoMediaDataSimulator.__init__` called `tf.random.set_seed(config.seed_num)`
      immediately after, overriding any caller-set seed, and `data_simulator.py` makes no
      `np.random` calls — so **every figure in this study came from `seed_num` 1320**.
      `seed_num` is now `int | None` and only sets the seed when not `None`; default
      behaviour is unchanged and the landed scenario still reproduces bit-identically
      (KPI checksum 5.841398e+09). Ten draws are swept in
      `realistic-baseline-noise-2ch-seeds.ipynb` via `seed_sweep_worker.py` (one process
      per draw — forty fits in one kernel exhausts memory and kills it). Result: `ec_m` and
      the invariance separation are structural; **ROI is not** — see below.
- [x] ~~Build the §5 two-curves-through-one-point figure (beat 3).~~ **Built as slide 9,
      with the premise corrected.** On the realistic DGP the curves do *not* pass through
      the same point: the default's curve sits above the truth at today's spend and below
      it further out, crossing at ~3.4x. The honest framing is stronger — a curve forced to
      flatten too early can only match the observed outcome by being too steep at the
      start, so the same error both over-credits the channel today and under-funds it
      tomorrow. Do not say "both curves fit today's data equally well"; the chart
      contradicts it.
- [ ] Still `SIM_SEED`-based no-ops in `final/roi-vs-mroi-metric-selection.ipynb` and the
      other study notebooks — harmless but misleading. Remove them, or pass `seed_num`
      explicitly, when those notebooks are next touched.
