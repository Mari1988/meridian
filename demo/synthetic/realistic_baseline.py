"""A realistic-baseline / realistic-noise variant of the synthetic DGP.

Motivation
----------
The landed study (`final/roi-vs-mroi-metric-selection.ipynb`) runs against a
DGP that is a *best case* for the model in two ways that flatter it:

1.  **The baseline is drawn from the model's own basis.** `SimulationConfig.
    n_knots_mu_t=8` generates `mu_t` from an 8-knot spline and
    `model_utils.build_model_spec` fits an 8-knot spline, so the true
    time-varying baseline is exactly representable by the fitted one.
2.  **The residual is iid.** `eps_gt ~ Normal(0, 0.5)` independently across
    geo and week, giving a measured DGP R^2 of 0.9958 with literally zero
    autocorrelation (-0.008) and zero cross-geo correlation (-0.010).

Real MMM residuals are neither. The unmodeled drivers -- competitor activity,
pricing, promotions, distribution, stockouts -- are *persistent* in time, and
mostly *national or regional* rather than geo-idiosyncratic. That distinction
matters more than the noise magnitude: iid geo noise averages down at
`1/sqrt(n_geos)` (a 4.5x reduction at 20 geos), so a geo hierarchy disposes of
it almost for free, whereas a common shock gets no such reduction and competes
with media for explanatory power exactly the way a national flight does.

What this module changes
------------------------
Two overrides on `GeoMediaDataSimulator`, and nothing else -- the media,
cost, adstock/Hill, ROI-calibration and ground-truth steps are inherited
unchanged, so the same known-ground-truth recovery check still applies.

`simulate_intercepts()` -- `mu_t` becomes `trend + seasonality + AR(1) shock`
    instead of a draw from the fitted spline's basis. The seasonal term is
    deliberately *phase-shifted* relative to the observed seasonality control
    (`sin` at the annual frequency, plus a second harmonic), so it is not
    absorbable by that control -- but, being orthogonal to it over a whole
    number of years, it also does not induce omitted-variable bias. The AR(1)
    shock is the part that matters: at `mu_ar1_phi=0.85` its correlation
    length is ~7 weeks, while 8 knots over 156 weeks spaces knots ~20 weeks
    apart, so the fitted spline structurally cannot represent it.

`simulate_coefficients()` -- `eps_gt` becomes a per-geo AR(1) in time with
    Student-t innovations, replacing the iid draw. This is geo-*idiosyncratic*
    by design: all cross-geo-correlated structure lives in `mu_t`, so the two
    dials stay non-overlapping and separately interpretable.
        - `mu_ar1_sd`   -> common shocks; defeats the spline *and* the
                           cross-geo averaging the hierarchy relies on.
        - `resid_sd`    -> geo-idiosyncratic shocks; defeats the likelihood's
                           iid assumption but still averages across geos.

Calibrating the noise level
---------------------------
`oracle_r2()` reports the R^2 ceiling achievable by Meridian's *mean
structure* given the true media contribution: geo intercepts, geo-varying
control coefficients, an 8-knot common time spline, and true incremental
media. It is an upper bound no fitted model can beat, and it is the number to
target -- unlike `1 - var(eps)/var(kpi)`, it charges the DGP for baseline
misspecification rather than only for residual variance.

`calibrate_to_oracle_r2()` bisects a single multiplier applied jointly to the
AR(1) shock and the residual (leaving trend/seasonality fixed, since those are
structure rather than noise) until the oracle R^2 hits a target.

Isolation
---------
Nothing here mutates `data_simulator.py` or `model_utils.py`. The landed
notebook imports neither this module nor anything it changes;
`assert_landed_path_unchanged()` verifies that explicitly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from meridian.model import knots as knots_lib

from data_simulator import GeoMediaDataSimulator
from data_simulator import SimulationConfig
from model_utils import build_input_data


@dataclasses.dataclass
class RealisticBaselineConfig:
  """Structure of the rewritten baseline and residual.

  All scales are in *pre-`baseline_scale`* per-capita KPI units, i.e. the same
  units as `tau_g` (whose mean is 15.0). `generate_kpi_and_revenue()` then
  multiplies the whole baseline term by `SimulationConfig.baseline_scale`.
  """

  # --- mu_t: trend -------------------------------------------------------
  # Peak-to-trough drift across the full window, as a fraction of the mean
  # `tau_g` level. Real category baselines drift; a spline fits this easily,
  # which is fine -- it is here for realism, not to break anything.
  trend_frac_of_level: float = 0.10

  # --- mu_t: seasonality -------------------------------------------------
  # Amplitude of the annual seasonal term. Phase-shifted 90 degrees from the
  # observed seasonality control (a `cos` peaking at
  # `SimulationConfig.demand_seasonal_peak_week`), so the control cannot
  # absorb it, while remaining orthogonal to it over whole years -- realism
  # without omitted-variable bias.
  seasonal_amplitude: float = 2.0
  # Second harmonic (26-week period) as a fraction of the annual amplitude.
  # Not spanned by the single-sinusoid control at any phase.
  seasonal_harmonic2_frac: float = 0.35

  # --- mu_t: common AR(1) shock -----------------------------------------
  # The load-bearing term: common across all geos, so neither the spline nor
  # the geo hierarchy's cross-geo averaging can dispose of it.
  #
  # `phi` is tuned, not guessed. An AR(1)'s power is concentrated at low
  # frequencies (the spectral density at f=0 over f=Nyquist is ~150x at
  # phi=0.85, ~16x at phi=0.6), and low-frequency power is exactly what a
  # spline absorbs. Measured share of the shock an 8-knot spline can
  # represent: 70% at phi=0.85, 44% at 0.7, 34% at 0.6, 26% at 0.5. phi=0.6
  # keeps two thirds of the shock genuinely unrepresentable while staying a
  # plausible ~2.5-week correlation length for promo/competitor activity.
  mu_ar1_phi: float = 0.6
  mu_ar1_sd: float = 1.0
  mu_ar1_df: float = 5.0

  # --- eps_gt: geo-idiosyncratic AR(1) residual -------------------------
  # Persistent within geo, independent across geos. Replaces the iid draw.
  # Only the *ratio* `resid_sd / mu_ar1_sd` is a free choice -- the absolute
  # level is set by `calibrate_to_oracle_r2`. Equal scales put ~38% of the
  # unexplainable variance in the common component, a defensible middle
  # ground: high enough that cross-geo averaging cannot rescue the fit, low
  # enough that the DGP is not simply one national shock in a trench coat.
  resid_phi: float = 0.4
  resid_sd: float = 1.0
  resid_df: float = 5.0


def _ar1(
    phi: float,
    stationary_sd: float,
    df: float,
    shape: list[int],
    n_times: int,
    dtype,
) -> tf.Tensor:
  """AR(1) over the leading axis with a given stationary sd.

  Student-t innovations (`Var = scale^2 * df/(df-2)`) are rescaled so the
  stationary sd is `stationary_sd` regardless of `df`, matching the
  convention already used by `GeoMediaDataSimulator._simulate_ar1_reach_noise`.
  """
  variance_inflation = df / (df - 2.0)
  innovation_scale = stationary_sd * np.sqrt(
      max(1.0 - phi**2, 1e-6) / variance_inflation
  )
  innovations = tfp.distributions.StudentT(
      df=tf.constant(df, dtype=dtype),
      loc=tf.constant(0.0, dtype=dtype),
      scale=tf.constant(innovation_scale, dtype=dtype),
  ).sample([n_times] + shape)
  phi_t = tf.constant(phi, dtype=dtype)
  return tf.scan(
      lambda prev, innovation: phi_t * prev + innovation,
      innovations,
      initializer=tf.zeros(shape, dtype=dtype),
  )


class RealisticBaselineSimulator(GeoMediaDataSimulator):
  """`GeoMediaDataSimulator` with a non-spline baseline and AR(1) residuals."""

  def __init__(
      self,
      config: SimulationConfig,
      realism: RealisticBaselineConfig | None = None,
  ):
    super().__init__(config)
    self.realism = realism or RealisticBaselineConfig()
    # Populated by the overrides below; `_apply_noise_scale` recombines them.
    self.mu_trend_t = None
    self.mu_seasonal_t = None
    self.mu_shock_t = None
    self.eps_base_gt = None
    self.noise_scale = 1.0

  # 4. Time-varying intercepts -- overridden.
  def simulate_intercepts(self) -> tf.Tensor:
    """`mu_t = trend + phase-shifted seasonality + common AR(1) shock`.

    Deliberately *not* drawn from `knots.get_knot_info`, so the true baseline
    is not a member of the fitted spline's span. `tau_g` is inherited
    unchanged.
    """
    config = self.config
    realism = self.realism
    dtype = self.p_g.dtype
    n_times = config.n_times

    self.tau_g = tfp.distributions.Normal(15.0, 1.2).sample(self.n_geos)

    t_frac = np.linspace(-0.5, 0.5, n_times)
    level = 15.0
    self.mu_trend_t = tf.constant(
        realism.trend_frac_of_level * level * t_frac, dtype=dtype
    )

    week_of_year = self.time_index.isocalendar().week.to_numpy(dtype=np.float64)
    annual_phase = (
        2 * np.pi * (week_of_year - config.demand_seasonal_peak_week) / 52.0
    )
    # `sin` where the observed control is `cos`: unspanned by the control,
    # yet orthogonal to it over whole years (no OVB). Plus a second harmonic,
    # which no single-frequency control spans at any phase.
    seasonal = realism.seasonal_amplitude * (
        np.sin(annual_phase)
        + realism.seasonal_harmonic2_frac * np.cos(2 * annual_phase)
    )
    self.mu_seasonal_t = tf.constant(seasonal, dtype=dtype)

    self.mu_shock_t = _ar1(
        phi=realism.mu_ar1_phi,
        stationary_sd=realism.mu_ar1_sd,
        df=realism.mu_ar1_df,
        shape=[],
        n_times=n_times,
        dtype=dtype,
    )

    self.mu_t = (
        self.mu_trend_t
        + self.mu_seasonal_t
        + self.noise_scale * self.mu_shock_t
    )
    return self.mu_t

  # 6. Coefficients and residual -- overridden for `eps_gt` only.
  def simulate_coefficients(self) -> tf.Tensor:
    """Inherited coefficients; `eps_gt` becomes a per-geo AR(1) in time."""
    beta_gm = super().simulate_coefficients()

    realism = self.realism
    # `_ar1` scans over the leading axis, so build [time, geo] then transpose.
    self.eps_base_gt = tf.transpose(
        _ar1(
            phi=realism.resid_phi,
            stationary_sd=realism.resid_sd,
            df=realism.resid_df,
            shape=[self.n_geos],
            n_times=self.config.n_times,
            dtype=self.p_g.dtype,
        )
    )
    self.sigma = tf.fill([1], realism.resid_sd)
    self.eps_gt = self.noise_scale * self.eps_base_gt
    return beta_gm

  # --- noise-level control ------------------------------------------------
  def apply_noise_scale(self, scale: float) -> None:
    """Rescales the *unexplainable* components and regenerates the KPI.

    Trend and seasonality are left fixed: they are baseline structure a
    modeller would legitimately try to fit, not noise. Only the common AR(1)
    shock and the geo-idiosyncratic residual are scaled.
    """
    self.noise_scale = float(scale)
    self.mu_t = (
        self.mu_trend_t + self.mu_seasonal_t + self.noise_scale * self.mu_shock_t
    )
    self.eps_gt = self.noise_scale * self.eps_base_gt
    self.sigma = tf.fill([1], self.realism.resid_sd * self.noise_scale)
    self.generate_kpi_and_revenue()


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def _oracle_design(sim: RealisticBaselineSimulator, n_knots: int) -> np.ndarray:
  """Design matrix spanning Meridian's mean structure, given true media.

  Columns: geo intercepts, geo-varying control coefficients (geo x control
  interactions, matching Meridian's `gamma_gc`), a common `n_knots` time
  spline (matching `ModelSpec.knots`), and the true incremental media
  contribution. Regressing the KPI on this is an *upper bound* on what any
  fitted Meridian model could explain -- it is handed the true media effect
  and only has to find linear coefficients.
  """
  n_geos, n_times = sim.n_geos, sim.config.n_times
  geo_eye = np.eye(n_geos)

  blocks = [np.repeat(geo_eye, n_times, axis=0)]  # geo intercepts

  controls = sim.transformed_control_gtc.numpy()  # (g, t, c)
  for c in range(controls.shape[-1]):  # geo x control interactions
    col = controls[:, :, c]
    blocks.append(
        np.einsum('gt,gj->gtj', col, geo_eye).reshape(n_geos * n_times, n_geos)
    )

  spline = np.asarray(  # (n_knots, t) -> common across geos
      knots_lib.get_knot_info(n_times, n_knots, False).weights
  )
  blocks.append(np.tile(spline.T, (n_geos, 1)))

  media = np.einsum(
      'gtm,gm->gt', sim.media_transformed.numpy(), sim.beta_gm.numpy()
  )
  blocks.append(media.reshape(-1, 1))

  return np.concatenate(blocks, axis=1)


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Rank-deficiency-safe least squares.

  The design matrix is deliberately collinear -- the spline basis rows sum to
  a constant, which the geo-intercept block already spans -- so an unguarded
  `lstsq` can return inf/nan coefficients that still reduce the residual.
  An explicit singular-value cutoff keeps the fitted values finite.
  """
  return np.linalg.pinv(x, rcond=1e-10) @ y


def _spline_basis(n_times: int, n_knots: int) -> np.ndarray:
  """`(n_times, n_knots)` basis matching `ModelSpec.knots`."""
  return np.asarray(knots_lib.get_knot_info(n_times, n_knots, False).weights).T


def _spline_residual(series: np.ndarray, n_knots: int) -> np.ndarray:
  """The part of a common time series an `n_knots` spline cannot represent."""
  basis = _spline_basis(len(series), n_knots)
  return series - basis @ _lstsq(basis, series)


def oracle_r2(sim: RealisticBaselineSimulator, n_knots: int = 8) -> float:
  """R^2 ceiling for Meridian's mean structure with true media handed to it."""
  x = _oracle_design(sim, n_knots)
  y = (sim.kpi_gt.numpy() / sim.p_g.numpy()[:, None]).reshape(-1)
  resid = y - x @ _lstsq(x, y)
  return float(1.0 - resid.var() / y.var())


def dgp_r2(sim: RealisticBaselineSimulator) -> float:
  """`1 - var(eps)/var(kpi)`, the metric reported for the landed DGP (0.9958).

  Kept for comparability only. It ignores baseline misspecification, so it
  always reads higher than `oracle_r2` -- quote the latter.
  """
  y = sim.kpi_gt.numpy() / sim.p_g.numpy()[:, None]
  eps = sim.config.baseline_scale * sim.eps_gt.numpy()
  return float(1.0 - eps.var() / y.var())


def calibrate_to_oracle_r2(
    sim: RealisticBaselineSimulator,
    target_r2: float,
    n_knots: int = 8,
    tol: float = 2e-4,
    max_iter: int = 40,
) -> float:
  """Bisects the noise multiplier until `oracle_r2(sim) == target_r2`.

  Mutates `sim` in place, leaving it at the calibrated noise level. Returns
  the multiplier found. Oracle R^2 is monotonically decreasing in the
  multiplier, so plain bisection is safe.
  """
  lo, hi = 1e-3, 1.0
  sim.apply_noise_scale(hi)
  while oracle_r2(sim, n_knots) > target_r2 and hi < 1e4:
    lo, hi = hi, hi * 2.0
    sim.apply_noise_scale(hi)

  for _ in range(max_iter):
    mid = np.sqrt(lo * hi)  # geometric: the scale spans orders of magnitude
    sim.apply_noise_scale(mid)
    achieved = oracle_r2(sim, n_knots)
    if abs(achieved - target_r2) < tol:
      return mid
    if achieved > target_r2:
      lo = mid
    else:
      hi = mid
  return mid


def baseline_diagnostics(
    sim: RealisticBaselineSimulator, n_knots: int = 8
) -> dict[str, float]:
  """Numbers that characterise how unfriendly the DGP now is to the model."""
  bs = sim.config.baseline_scale
  p_g = sim.p_g.numpy()[:, None]
  kpi_pc = sim.kpi_gt.numpy() / p_g
  eps = bs * sim.eps_gt.numpy()
  mu_shock = bs * sim.noise_scale * sim.mu_shock_t.numpy()
  media = np.einsum(
      'gtm,gm->gt', sim.media_transformed.numpy(), sim.beta_gm.numpy()
  )

  # How much of the true mu_t an `n_knots` spline can actually represent.
  mu = sim.mu_t.numpy()
  shock = sim.mu_shock_t.numpy()
  mu_unfit = _spline_residual(mu, n_knots)
  shock_unfit = _spline_residual(shock, n_knots)
  # The genuinely unexplainable common component, in KPI units: what is left
  # of the common shock after the spline has taken everything it can.
  common_unexplained_t = bs * sim.noise_scale * shock_unfit

  baseline_term = (
      bs
      * (
          sim.tau_g.numpy()[:, None]
          + np.einsum(
              'gtc,gc->gt',
              sim.transformed_control_gtc.numpy(),
              sim.gamma_gc.numpy(),
          )
          + sim.eps_gt.numpy()
          + sim.mu_t.numpy()[None, :]
      )
  )

  return {
      'oracle_r2': oracle_r2(sim, n_knots),
      'dgp_r2_eps_only': float(1.0 - eps.var() / kpi_pc.var()),
      'noise_scale': sim.noise_scale,
      'media_share_pct': float(media.sum() / kpi_pc.sum() * 100),
      'kpi_pc_mean': float(kpi_pc.mean()),
      'kpi_pc_sd': float(kpi_pc.std()),
      'eps_sd': float(eps.std()),
      'mu_shock_sd': float(mu_shock.std()),
      'spline_explains_mu_pct': float(
          (1 - mu_unfit.var() / mu.var()) * 100
      ),
      'spline_explains_mu_shock_pct': float(
          (1 - shock_unfit.var() / shock.var()) * 100
      ),
      'common_unexplained_sd': float(common_unexplained_t.std()),
      'resid_lag1_autocorr': float(
          np.mean([
              np.corrcoef(eps[g, :-1], eps[g, 1:])[0, 1]
              for g in range(eps.shape[0])
          ])
      ),
      # Cross-geo correlation of what the model genuinely cannot explain:
      # the spline-residual common shock plus the idiosyncratic residual.
      # Uses the spline residual, not the raw shock, so the metric is not
      # inflated by common variation the spline would have absorbed anyway.
      'unexplained_cross_geo_corr': _mean_cross_geo_corr(
          eps + common_unexplained_t[None, :]
      ),
      'clipped_frac_pct': float((baseline_term <= 0).mean() * 100),
  }


def _mean_cross_geo_corr(x: np.ndarray) -> float:
  cc = np.corrcoef(x)
  off_diagonal = cc.sum() - np.trace(cc)
  return float(off_diagonal / (cc.size - cc.shape[0]))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_real_augmented_realistic(
    config: SimulationConfig,
    real_df: pd.DataFrame,
    rf_source_map: dict[str, str],
    plain_source_map: dict[str, str],
    realism: RealisticBaselineConfig | None = None,
    target_oracle_r2: float | None = None,
    n_knots: int = 8,
):
  """`model_utils.build_real_augmented_input` on the realistic-baseline DGP.

  Same call sequence, with `RealisticBaselineSimulator` substituted. When
  `target_oracle_r2` is set, the noise multiplier is calibrated *after* ROI
  calibration (which depends only on media and cost, so it is unaffected) and
  *before* the dataframe / ground-truth tail -- necessary because
  `compute_ground_truth` rescales `beta_gm` by the KPI's standard deviation,
  which the noise level moves.

  Returns `(sim, data, ground_truth)`, matching the landed driver.
  """
  sim = RealisticBaselineSimulator(config, realism)
  sim.simulate_population_from_real(real_df)
  sim.align_time_index_to_real(real_df)
  sim.simulate_controls()
  sim.simulate_media_from_real(
      real_df, rf_source_map, plain_source_map, verbose=False
  )
  sim.simulate_cost_and_unit_value()
  sim.simulate_intercepts()
  sim.simulate_coefficients()
  sim.simulate_adstock_hill_params()
  sim.transform_media()
  sim.calibrate_channel_effects()
  sim.generate_kpi_and_revenue()

  if target_oracle_r2 is not None:
    calibrate_to_oracle_r2(sim, target_oracle_r2, n_knots=n_knots)

  df = sim.to_dataframe()
  ground_truth = sim.compute_ground_truth(verbose=False)
  data = build_input_data(df, config.channel_names, sim.control_col_names)
  return sim, data, ground_truth


def assert_landed_path_unchanged(real_df: pd.DataFrame, base_overrides: dict):
  """Verifies the landed notebook's scenario still reproduces bit-identically.

  Guards the isolation claim: this module subclasses rather than edits, so
  `model_utils.build_real_augmented_input` must be byte-for-byte unaffected.
  """
  from model_utils import build_real_augmented_input

  def run():
    np.random.seed(0)
    tf.random.set_seed(0)
    cfg = SimulationConfig.from_dict(dict(base_overrides))
    return build_real_augmented_input(
        cfg,
        real_df,
        rf_source_map={'TV': 'Channel3'},
        plain_source_map={'Display': 'Channel2', 'Social': 'Channel1'},
    )

  sim_a, _, gt_a = run()
  sim_b, _, gt_b = run()
  assert np.array_equal(sim_a.kpi_gt.numpy(), sim_b.kpi_gt.numpy()), (
      'landed path is not reproducible'
  )
  assert np.allclose(gt_a['roi_m'], gt_b['roi_m'])
  return {
      'true_ec_m': sim_a.ec_m.numpy(),
      'true_roi_m': np.asarray(gt_a['roi_m']),
      'kpi_checksum': float(sim_a.kpi_gt.numpy().sum()),
  }
