# ARF Analytics Council talk — outline

**Status:** draft outline, not yet turned into slides. Update this file as the talk gets prepared/delivered.

**Context:** invited by Sable (ARF) to present to the ARF Analytics Council, currently focused on "unpacking key challenges and emerging solutions in MMM." Session scheduled for Tuesday, August 4 at 11:00am PT / 1:00pm CT / 2:00pm ET. Council mission: identify common data uses in the current martech environment, explore evolving methodologies, and develop guidelines/best practices across audience development, channel optimization, and cross-media measurement — with a core goal of demystifying/democratizing the "black box" of analytics.

**Source material:** `demo/synthetic/data_simulator.py` (the synthetic ground-truth data generator) and `demo/synthetic/meridian_tv_underreach_case_study.ipynb` (the sharper case study), plus the companion `demo/synthetic/meridian_ec_prior_case_study.ipynb`. See the "Synthetic ground-truth prior-recovery study" section of `CLAUDE.md` for the full technical map.

**Finding, one sentence:** Meridian's (and MMM tooling generally) default `ec_m` (half-saturation) prior implicitly assumes a channel's current spend is already near its saturation point; for a channel that's genuinely under-reached *and* under-frequency'd (e.g. a large-audience TV buy running well below its addressable audience, at low frequency), that assumption is systematically wrong — and a reach/frequency-informed prior, built from ordinary media-planning inputs, fixes it.

---

## Title options
- *"Is Your MMM Telling You You're Saturated When You're Not?"*
- *"The Hidden Assumption in Your MMM's Default Settings"*
- *"When 'Optimized' Isn't Optimal: A Prior Problem in Marketing Mix Modeling"*

## Outline (~20 min talk + discussion)

### 1. Hook — the business problem (2 min)
Open with the scenario, not the statistics: a channel with a large addressable audience, but currently reached at low reach *and* low frequency — genuinely under-invested by any media-planning definition. Ask the room: *if your MMM concluded this channel was already near its ceiling, would you know to question that?* Most wouldn't — the model looks confident either way.

### 2. What's actually inside the black box (3 min)
- MMMs use a "saturation curve" (Hill function) per channel — every channel has a *half-saturation point*: the delivery level at which incremental spend starts paying off less.
- The tools' out-of-the-box defaults have to assume *something* about where that point is — and the common default assumption is, roughly, "wherever you're spending today is already close to it."
- That's a reasonable prior for a channel running near its historical norms. It's a bad prior for a channel that's structurally under-delivered.

### 3. Why under-reach + under-frequency is the sharpest version of this problem (3 min)
- Two independent, compounding forms of under-delivery: not enough people reached, *and* not enough frequency per person reached.
- Walk through the scenario numbers (large audience, ~10% reach, ~1.5x/week frequency) and show the gap between "where the model's default assumes saturation" and "where the data suggests it actually is."

### 4. Why we built a simulation instead of pointing at a real client (2 min)
- You can't validate "is my MMM right about saturation" against real data — you never observe the *true* saturation point in the real world, only the fitted one.
- So: build synthetic geo x time data from a fully known generative model (known audience sizes, reach, frequency, true half-saturation, true ROI), fit MMM to it, and check whether it recovers the truth it was built from.
- **Caveat to state explicitly, up front:** the *specific* saturation threshold used to build the simulation is a stylized assumption, not a claim about real-world saturation levels. The point being tested is the *mechanism* (does the prior distort recovery when truth is far from the default's assumption), not that specific number.

### 5. The finding, in one number (4 min — the core slide)
- Headline diagnostic: "fraction of ceiling effect captured at current spend" — truth vs. what the default-prior model believes.
- Pull the response-curve overlay chart (true vs. default-posterior vs. informed-posterior) — visually, the default curve flattens out far earlier than the truth.
- One sentence framing the takeaway: *the model isn't just uncertain — it's confidently wrong, and it's wrong in the direction of telling you not to spend more.*

### 6. Does the mistake actually change a decision? (3 min)
- Yes — pull the budget-reallocation table: the default-prior fit recommends materially less spend on the under-reached channel than the true optimum; an audience/reach-informed prior tracks the true optimum instead.
- This is the "so what": it flows straight into a lower budget recommendation for exactly the channel that should be getting more.

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

---

## Open items / TODO
- [ ] Re-run `meridian_tv_underreach_case_study.ipynb` at full MCMC settings (`n_adapt=2000`, `N_SEEDS=20`, etc.) before pulling numbers/charts into slides — figures currently referenced in this outline are from reduced-precision smoke-test runs.
- [ ] Turn this outline into an actual slide deck.
- [ ] Decide on final framing: general MMM-methodology point vs. anything Meridian-specific (lean general — avoid reading as criticism of a specific open-source tool).
