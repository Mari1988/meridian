from __future__ import annotations

import sys
home_dir = "/Users/mariappan.subramanian/Documents/"
sys.path.append(f'{home_dir}/repo/forked/meridian/meridian/')

from meridian.mpa.mini_meridian_utils import create_xarray_and_tf_tensor

from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd, tfb, tfmcmc = tfp.distributions, tfp.bijectors, tfp.mcmc

# -----------------------------------------------------------------------------
# 0.  PROCESS RAW DATA
# -----------------------------------------------------------------------------

# read raw model ready data
input_file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/InteractionEffects/data/testdata_Mazda_national_False_mdf.csv"
sample_data = pd.read_csv(input_file_path)

# config
imp_variables = ["TV_I", "Display_I", "Video_I"]
spend_variables = ["TV_AC", "Display_AC", "Video_AC"]
kpi = "conversions"
date_field = "WES"
geo_field = "Region"

# create raw media tensors
media_tensors = create_xarray_and_tf_tensor(sample_data, date_field, geo_field, imp_variables)
media_spend_tensors = create_xarray_and_tf_tensor(sample_data, date_field, geo_field, spend_variables)

# create raw outcome tensor
kpi_tensor = create_xarray_and_tf_tensor(sample_data, date_field, geo_field, [kpi])

# create raw population tensor
population_tensor = create_xarray_and_tf_tensor(sample_data, date_field, geo_field, ["Population"])


# -----------------------------------------------------------------------------
# 0.  SMALL HELPERS
# -----------------------------------------------------------------------------

def _non_zero_median(arr: np.ndarray) -> float:
    """Return median of non‑zero elements (or 1.0 if array is all zeros)."""
    nz = arr[arr != 0]
    return float(np.median(nz)) if nz.size else 1.0


# -----------------------------------------------------------------------------
# 1.  SCALING PIPELINE  (per‑capita  ➜  y‑zscore  ➜  X / median)
# -----------------------------------------------------------------------------

def scale_mmm_inputs(
    impressions: np.ndarray,  # shape (G, T, M)
    outcome: np.ndarray,      # shape (G, T)
    population: np.ndarray    # shape (G, T)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Apply the three‑step scaling and return X_scaled, y_scaled, factors."""
    if impressions.shape[:2] != outcome.shape or population.shape != outcome.shape:
        raise ValueError("Shapes of impressions, outcome, and population must align in first two dims.")

    # ---- Step 1: create per‑capita values ----
    impressions_pc = impressions / population[..., None]
    outcome_pc = outcome / population

    # ---- Step 2: z‑score outcome ----
    y_mean = outcome_pc.mean()
    y_std = outcome_pc.std(ddof=0)
    y_scaled = (outcome_pc - y_mean) / y_std

    # ---- Step 3: divide each channel by its non‑zero median ----
    G, T, M = impressions_pc.shape
    X_scaled = np.empty_like(impressions_pc, dtype=np.float32)
    medians = np.zeros(M, dtype=np.float32)
    for m in range(M):
        med = _non_zero_median(impressions_pc[..., m])
        medians[m] = med
        X_scaled[..., m] = impressions_pc[..., m] / med

    factors = {
        "y_mean": y_mean,
        "y_std": y_std,
        "X_median": medians,
        "population": population,
    }
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32), factors


# -----------------------------------------------------------------------------
# 2.  MODEL –  HIERARCHICAL GEO MMM WITH INTERACTIONS
# -----------------------------------------------------------------------------

def build_geo_hier_mmm(
    X_scaled: tf.Tensor,         # (G, T, M)  – per‑cap, median‑scaled impressions
    y_scaled: tf.Tensor,         # (G, T)
    interaction_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[tfp.distributions.JointDistributionCoroutine, int, List[Tuple[int, int]]]:
    """Create the TFP JointDistribution and return it plus helper sizes.

    Parameters
    ----------
    X_scaled : tf.Tensor
        Scaled impressions (float32) of shape (G, T, M).
    y_scaled : tf.Tensor
        Scaled outcome of shape (G, T).
    interaction_pairs : list[(m,k)] or None
        If None, use all m<k combinations.
    """
    G, T, M = X_scaled.shape
    if interaction_pairs is None:
        interaction_pairs = [(m, k) for m in range(M) for k in range(m + 1, M)]
    P = len(interaction_pairs)  # number of interaction terms

    # --- adstock + hill util inside model ---
    def adstock_1d(x, d):
        ta = tf.TensorArray(dtype=x.dtype, size=T)
        carry = tf.constant(0., dtype=x.dtype)
        for t in tf.range(T):
            carry = x[t] + d * carry
            ta = ta.write(t, carry)
        return ta.stack()

    def saturate(x, a, g):
        return tf.pow(x, a) / (tf.pow(x, a) + tf.pow(g, a))

    # --- model coroutine ---
    def model():
        # Hyper‑priors for intercepts and main‑effect β
        mu_intercept = yield tfd.Normal(0., 10., name="mu_intercept")
        tau_intercept = yield tfd.HalfNormal(1., name="tau_intercept")

        # Hyper‑priors for main‑effect β
        mu_beta = yield tfd.Sample(tfd.HalfNormal(0.01), sample_shape=M, name="mu_beta")
        tau_beta = yield tfd.Sample(tfd.HalfNormal(0.005), sample_shape=M, name="tau_beta")

        # Likelihood noise
        sigma = yield tfd.HalfNormal(1., name="sigma")

        # Shared transform params across geos for each channel
        decay = yield tfd.Sample(tfd.Beta(2., 2.), sample_shape=M, name="decay")
        alpha = yield tfd.Sample(tfd.Normal(1., 0.1), sample_shape=M, name="alpha")
        gamma = yield tfd.Sample(tfd.HalfNormal(1.), sample_shape=M, name="gamma")

        # --- INTERACTION hyper‑priors ---
        mu_beta_int = yield tfd.Sample(tfd.Normal(0., 0.002), sample_shape=P, name="mu_beta_int")
        tau_beta_int = yield tfd.Sample(tfd.HalfNormal(0.001), sample_shape=P, name="tau_beta_int")

        # --- geo‑specific parameters ---
        intercept_geo = yield tfd.Sample(tfd.Normal(mu_intercept, tau_intercept), sample_shape=G, name="intercept_geo")  # (G,)

        beta_geo = yield tfd.Sample(tfd.Normal(mu_beta, tau_beta), sample_shape=G, name="beta_geo")  # (G, M)

        beta_int_geo = yield tfd.Sample(tfd.Normal(mu_beta_int, tau_beta_int), sample_shape=G, name="beta_int_geo")  # (G, P)

        # --- compute mean sales (vectorised across geos) ---
        X = tf.convert_to_tensor(X_scaled, dtype=tf.float32)  # ensure tf.Tensor inside coroutine
        sat_cache = []  # list length M, each (G, T) tensor
        for m in range(M):
            ad_g = tf.vectorized_map(lambda g: adstock_1d(X[g, :, m], decay[m]), tf.range(G))  # (G, T)
            sat_g = saturate(ad_g, alpha[m], gamma[m])
            sat_cache.append(sat_g)

        # main effects
        mu_main = tf.add_n([
            beta_geo[:, m][:, None] * sat_cache[m] for m in range(M)
        ])  # (G, T)

        # interaction effects
        if P > 0:
            mu_int_parts = []
            for p, (m, k) in enumerate(interaction_pairs):
                mu_int_parts.append(beta_int_geo[:, p][:, None] * sat_cache[m] * sat_cache[k])
            mu_int = tf.add_n(mu_int_parts)
        else:
            mu_int = 0.

        mu = intercept_geo[:, None] + mu_main + mu_int  # (G, T)

        yield tfd.Independent(
            tfd.Normal(loc=mu, scale=sigma), reinterpreted_batch_ndims=2, name="likelihood"
        )

    jd = tfd.JointDistributionCoroutine(model)
    return jd, M, interaction_pairs


# -----------------------------------------------------------------------------
# 3.  SAMPLER WRAPPER
# -----------------------------------------------------------------------------

def run_hierarchical_mmm(
    impressions: np.ndarray,
    outcome: np.ndarray,
    population: np.ndarray,
    interaction_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    num_results: int = 1000,
    num_burnin: int = 500,
    step_size: float = 0.03,
    seed: int = 0,
):
    """Scale data, build model, and run NUTS.  Returns samples & scaling factors."""
    # ---------- scaling ----------
    X_scaled, y_scaled, scalers = scale_mmm_inputs(impressions, outcome, population)
    X_tf = tf.convert_to_tensor(X_scaled)
    y_tf = tf.convert_to_tensor(y_scaled)

    # ---------- model ----------
    jd, M, pairs = build_geo_hier_mmm(X_tf, y_tf, interaction_pairs)
    P = len(pairs)

    # target log‑prob (all params except likelihood) -------------------
    def _tlp(*params):
        return jd.log_prob((*params, y_tf))

    # ---------- bijectors ----------
    bijectors: List[tfb.Bijector] = [
        tfb.Identity(),                # mu_intercept
        tfb.Softplus(),                # tau_intercept
        tfb.Softplus(),                # mu_beta
        tfb.Softplus(),                # tau_beta
        tfb.Softplus(),                # sigma
        tfb.Sigmoid(),                 # decay
        tfb.Identity(),                # alpha
        tfb.Softplus(),                # gamma
        tfb.Identity(),                # mu_beta_int
        tfb.Softplus(),                # tau_beta_int
        tfb.Identity(),                # intercept_geo
        tfb.Identity(),                # beta_geo
        tfb.Identity(),                # beta_int_geo
    ]

    # ---------- initial state ----------
    G = impressions.shape[0]
    init_state = [
        tf.zeros([]),                              # mu_intercept
        tf.ones([]),                               # tau_intercept
        tf.ones(M) * 0.005,                        # mu_beta (small)
        tf.ones(M) * 0.0025,                       # tau_beta
        tf.ones([]),                               # sigma
        tf.ones(M) * 0.5,                          # decay
        tf.ones(M),                                # alpha
        tf.ones(M),                                # gamma
        tf.zeros(P),                               # mu_beta_int
        tf.ones(P) * 0.0005,                       # tau_beta_int
        tf.zeros(G),                               # intercept_geo
        tf.ones((G, M)) * 0.005,                   # beta_geo
        tf.zeros((G, P)),                          # beta_int_geo
    ]

    # ---------- NUTS ----------
    kernel = tfmcmc.TransformedTransitionKernel(
        inner_kernel=tfmcmc.NoUTurnSampler(
            target_log_prob_fn=_tlp, step_size=step_size, seed=seed
        ),
        bijector=bijectors,
    )

    adaptive_kernel = tfmcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=0.8,
    )

    @tf.function(autograph=False, jit_compile=True)
    def _run_chain():
        return tfmcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin,
            current_state=init_state,
            kernel=adaptive_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=seed,
        )

    samples, is_accepted = _run_chain()
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()
    print(f"NUTS accept rate: {accept_rate:.2%}")

    return samples, scalers, pairs


# -----------------------------------------------------------------------------
# 4.  INCREMENTAL OUTCOME HELPER (main & interaction)
# -----------------------------------------------------------------------------
def incremental_effect(
    impressions: np.ndarray,
    population: np.ndarray,
    samples,
    scalers: Dict[str, np.ndarray],
    interaction_pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return incremental outcome in raw units.

    Returns
    -------
    inc_main_raw : ndarray (G, T, M)
        Incremental attributable to each **main** channel.
    inc_int_raw : ndarray (G, T, P)
        Incremental attributable to each interaction term (same order as pairs).
    """
    # -------- retrieve posterior means --------
    # samples layout matches model yields + likelihood; indexes:
    # 11 = beta_geo (G,M,S), 12 = beta_int_geo (G,P,S)
    beta_geo_draws = samples[11].numpy()  # (G,M,S)
    beta_geo_mean = beta_geo_draws.mean(axis=2)  # (G,M)

    beta_int_draws = samples[12].numpy()  # (G,P,S)
    beta_int_mean = beta_int_draws.mean(axis=2)  # (G,P)

    decay_mean = samples[5].numpy().mean(axis=0)
    alpha_mean = samples[6].numpy().mean(axis=0)
    gamma_mean = samples[7].numpy().mean(axis=0)

    # -------- rebuild scaled predictors --------
    X_pc = impressions / population[..., None]
    X_scaled = X_pc / scalers["X_median"][None, None, :]

    G, T, M = X_scaled.shape
    P = len(interaction_pairs)

    # -------- adstock + saturate --------
    sat = np.empty((G, T, M), dtype=np.float32)

    def adstock_np(series, d):
        out = np.empty_like(series)
        carry = 0.0
        for t in range(series.shape[0]):
            carry = series[t] + d * carry
            out[t] = carry
        return out

    for m in range(M):
        for g in range(G):
            ad = adstock_np(X_scaled[g, :, m], decay_mean[m])
            sat[g, :, m] = np.power(ad, alpha_mean[m]) / (
                np.power(ad, alpha_mean[m]) + np.power(gamma_mean[m], alpha_mean[m])
            )

    # -------- main effects --------
    inc_main_scaled = beta_geo_mean[:, None, :] * sat  # broadcast (G,1,M)*(G,T,M)

    # -------- interaction effects --------
    inc_int_scaled = np.zeros((G, T, P), dtype=np.float32)
    for p, (m, k) in enumerate(interaction_pairs):
        inc_int_scaled[..., p] = beta_int_mean[:, p][:, None] * sat[..., m] * sat[..., k]

    # -------- back‑transform to raw units --------
    y_std = scalers["y_std"]
    inc_main_pc = inc_main_scaled * y_std
    inc_int_pc = inc_int_scaled * y_std

    inc_main_raw = inc_main_pc * population[..., None]
    inc_int_raw = inc_int_pc * population[..., None]
    return inc_main_raw, inc_int_raw


# -----------------------------------------------------------------------------
# 5.  DEMO / SMOKE‑TEST  (synthetic data)
# -----------------------------------------------------------------------------






rng = np.random.default_rng(42)
G, T, M = 5, 104, 3

population = rng.integers(800_000, 1_200_000, size=(G, T))
# True betas for impressions‑driven outcome
true_beta = np.array([[2.1e-5, 1.4e-5, 0.9e-5],
                      [2.4e-5, 1.6e-5, 1.1e-5],
                      [1.9e-5, 1.8e-5, 0.7e-5],
                      [2.3e-5, 1.3e-5, 1.0e-5],
                      [2.0e-5, 1.7e-5, 0.8e-5]])

impressions = rng.random((G, T, M)) * 5e6  # up to ~5 million imps / week
noise = rng.normal(0, 0.5, size=(G, T))
outcome = 5 + (impressions * true_beta[:, None, :]).sum(-1) + noise

# We model only the interaction between channel 0 and 1
interaction_pairs = [(0, 1)]

samples, scalers, pairs = run_hierarchical_mmm(
    impressions, outcome, population, interaction_pairs, num_results=600, num_burnin=300
)

inc_main_raw, inc_int_raw = incremental_effect(
    impressions, population, samples, scalers, pairs
)

print("\nTotal incremental outcome per channel (main effects):")
for m in range(M):
    print(f"  Channel {m}: {inc_main_raw[..., m].sum():,.1f}")
print("\nIncremental from interaction term(s):")
for p, (m, k) in enumerate(pairs):
    print(f"  {m}×{k}: {inc_int_raw[..., p].sum():,.1f}")
