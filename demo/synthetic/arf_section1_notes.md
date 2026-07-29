# ARF Council talk — Section 1 speaker notes

**Section title:** *What does an MMM assume before it sees your data?*

Four slides, ~4 minutes. This expands §2 of
[ARF_COUNCIL_TALK_OUTLINE.md](../../ARF_COUNCIL_TALK_OUTLINE.md). Its job is to make
"MMMs have hidden default assumptions" concrete *before* the talk narrows to the
under-reached-TV finding in §§3–7 — so that result lands as a consequence, not a
curiosity.

Every number below is regenerated and verified by
`prior_plots_check.py`; the figures come from `build_section1_figures.py`.

---

## Slide 1 — Every MMM must assume something

*No chart.*

Before an MMM sees a single row of your data, it has already committed to three
answers:

1. **How fast does response flatten?** (the half-saturation point)
2. **What shape is the curve?** (does response build before it bends?)
3. **How long does an impression keep working?** (carryover)

You never get asked these questions. They ship as defaults.

**Say this out loud:** we can inspect these choices *only because* Meridian and Robyn
are open source. That's a credit to both. Every closed MMM makes the same three
choices — you just can't audit them. This section is not a criticism of open tools;
it's an argument for the kind of scrutiny only open tools permit.

---

## Slide 2 — Saturation: the default assumes you're already there

*Figure: `figures/section1_saturation.png`*

Meridian's default half-saturation prior is `ec_m ~ TruncatedNormal(0.8, 0.8, [0.1, 10])`.
On its own that's an unreadable line of code. It becomes readable once you know the
units.

**The key mechanic:** Meridian scales each channel's media by *that channel's own
median non-zero per-capita execution* before applying the saturation curve. So on the
model's x-axis, **1.0 means "what you already run."** `ec_m` is therefore denominated in
multiples of your own status quo.

| Quantity | Value |
|---|---|
| Prior median `ec_m` | **0.99× current execution** |
| P(already past half-saturation today) | **50%** |
| P(>1/3 of max effect already captured today) | **92%** |
| 90% prior interval for `ec_m` | [0.21, 2.20] × current execution |

**The line:** out of the box, the model gives you coin-flip odds that you are already
past the point of diminishing returns — for every channel, identically, regardless of
how big its audience is or how much of that audience you're actually reaching.

**Robyn contrast (one line):** Robyn's `gamma` plays the same role and is searched over
`[0.3, 1]` — also identically across channels. Neither tool asks about the channel.
This is not a Meridian quirk; it's how the category works.

**Footnote the slide with:**
`https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions#ec_m_and_ec_om`

**If asked "is this hidden?"** — No, and don't claim it is. Google documents it: the
docs state half-saturation occurs "at the median of the non-zero media units per capita
across geos and time," and they give a rationale (the truncation preserves
identifiability). The point is that it's *easy to miss and rarely interrogated*, not
that it's concealed.

---

## Slide 3 — Slope: the parameter that is never estimated

*Figure: `figures/section1_slope.png`*

Slope and half-saturation are the two parameters of the *same* curve, so this follows
directly from slide 2.

Meridian's default is `slope_m = Deterministic(1.0)` — **fixed, not estimated.** It has
no prior because it isn't a random variable. Fixing it at 1 means the curve is concave
everywhere: diminishing returns from the very first impression, no build-up phase, no
threshold effect.

**The official rationale is about tractability, not about advertising.** Google's docs
say concave Hill curves are required so budget optimization "produces a global
optimum." That is a legitimate engineering trade-off. It is also a modeling choice a
practitioner inherited without being asked — and it forecloses exactly the
effective-frequency threshold behavior most media planners believe in.

**The Robyn contrast is the anchor here — and it's exact, not analogy.** Both tools use
algebraically identical saturation functions:

- Robyn: `saturated = 1 / (1 + (gamma / adstocked) ^ alpha)`
- Meridian: `hill(x) = x^slope / (x^slope + ec^slope)` = `1 / (1 + (ec/x) ^ slope)`

So Robyn's `alpha` ≡ Meridian's `slope_m`, and Robyn's `gamma` ≡ Meridian's `ec_m`.
(`prior_plots_check.py` verifies these agree numerically to ~1e-7.) Robyn **searches**
alpha over `[0.5, 3.0]`, explicitly spanning C-shape through S-shape. Meridian fixes it
at exactly 1. Same equation, opposite decision about whether the curve's shape is
knowable from data.

**Read the right-hand panel aloud:** the dots mark peak marginal return. Under the
default (slope 1), peak marginal return is at **zero** — the model believes your very
first impression was your most productive one, always. At slope 2 the peak is at
**0.58×** today's spend; at slope 3, **0.79×**. Those are build-up phases the default
cannot represent at any parameter value.

**Fairness caveat — say it:** Meridian *lets* you override `slope_m`; it just warns
that doing so may break the optimizer's global optimum. The claim is about the default
and about which dimensions each tool searches by design, not that the door is locked.

---

## Slide 4 — Adstock: flat isn't neutral, and the real assumption isn't a prior

*Figures: `figures/section1_adstock.png`, optionally
`figures/section1_adstock_robyn_bounds.png`*

**Set this slide up as a deliberate contrast.** `alpha_m ~ Uniform(0, 1)` genuinely is
flat, and Google describes it as uninformative so the data can inform the decay rate.
There is no skew story here. If the section only had slide 2, it would read as
tool-bashing. Three different points instead:

**(a) Flat on the parameter is not flat on the quantity you care about.** Nobody makes
decisions about "alpha." They make decisions about *when* advertising works. Translate
the flat prior into the share of effect landing in week 0 and it is not flat at all:
median **50%**, with a **30%** chance the effect is mostly immediate (≥70% in week 0)
and a **28%** chance it mostly lingers (≤30% in week 0). The prior is nearly bimodal
on the only question a planner would ask.

**(b) The binding assumption isn't a prior at all.** `max_lag = 8` is a hard truncation
in the model spec — never estimated, no prior attached, no posterior uncertainty. At a
decay rate of 0.9, **39%** of the true geometric carryover falls beyond week 8, and is
discarded and renormalized away. If your TV genuinely works over a quarter, the default
cannot see it, and nothing in the output will tell you so.

**(c) Robyn contrast:** Robyn recommends decay bounds **per channel type** — TV
`[0.3, 0.8]`, OOH/print/radio `[0.1, 0.4]`, digital `[0, 0.3]` — encoding the ordinary
media-planning knowledge that TV lingers and digital doesn't. Meridian's default applies
one identical `Uniform(0, 1)` to every channel.

**The taxonomy is the thesis of the whole section.** Some defaults are skewed toward a
conclusion (slide 2). Some are flat on the parameter but misleading on the decision
(4a). And some aren't priors at all — they're hard-coded structure wearing the costume
of neutrality (4b). Land that, then move to the TV case study.

---

## Accuracy guardrails — read before presenting

- **Robyn is not Bayesian.** It's ridge regression with Nevergrad evolutionary
  hyperparameter search. Its `c(...)` values are *search bounds*, not priors. The
  comparison is about which quantities each tool treats as knowable from data — not a
  prior-vs-prior comparison. Say this if you make any Robyn claim; a Robyn user in the
  room will otherwise say it for you.
- **Don't overclaim Robyn's gamma scaling.** Robyn's docs say only that gamma is
  "scaled to the inflexion point of the variable," without detailing the mechanism.
  Meridian's median-based scaling is precisely documented; Robyn's is not. Compare the
  two at the level of *role*, not units, unless someone verifies Robyn's source first.
- **These are defaults, and both tools let you override them.** The argument is about
  what happens when nobody does — which is the common case.
- **Nothing in this section is Meridian-specific as a critique.** The framing is: this
  is what the category assumes, and we can see it here because it's open.
- **No reach/frequency in this section.** Out of scope for this talk.

## Sources

- `meridian/model/prior_distribution.py` — `ec_m` (383–391), `alpha_m` (355–362),
  `slope_m` (419–423), slope warning (844–856)
- `meridian/model/adstock_hill.py` — Hill (304–328), adstock with hardcoded
  `normalize=True` (250–294)
- `meridian/model/spec.py` — `max_lag = 8` (237)
- `meridian/model/transformers.py` — median-based media scaling (46–107)
- https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions
- https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM
