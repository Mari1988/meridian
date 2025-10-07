"""Helper function to create contribution priors from mean expectations."""
import numpy as np
import tensorflow_probability as tfp
from meridian import constants


def beta_params_from_mean_std(mean, std):
    """
    Calculate Beta distribution parameters (alpha, beta) from desired mean and std.

    Args:
        mean: Desired mean (e.g., 0.115 for 11.5%)
        std: Desired standard deviation (e.g., 0.03 for 3%)

    Returns:
        (alpha, beta): Parameters for Beta distribution

    Example:
        >>> alpha, beta = beta_params_from_mean_std(0.115, 0.03)
        >>> # Creates Beta distribution with mean=11.5%, std=3%
    """
    # Beta distribution formulas:
    # mean = alpha / (alpha + beta)
    # var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

    variance = std ** 2

    # Solve for alpha and beta
    # From mean: beta = alpha * (1 - mean) / mean
    # Substitute into variance formula and solve
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta = alpha * (1 - mean) / mean

    return alpha, beta


def create_contribution_prior(
    channel_names,
    mean_contributions,
    std_contributions=None,
    confidence='moderate'
):
    """
    Create contribution_m prior from expected mean contributions.

    Args:
        channel_names: List of channel names (e.g., ['Display', 'TV', 'Video'])
        mean_contributions: Dict or list of expected contributions as percentages
                           e.g., {'Display': 11.5, 'TV': 30, 'Video': 15}
                           or [11.5, 30, 15]
        std_contributions: Optional dict/list of std deviations as percentages
                          If None, uses confidence level
        confidence: 'low', 'moderate', or 'high' - controls uncertainty
                   low: std ≈ 0.15 * mean (flexible)
                   moderate: std ≈ 0.10 * mean (balanced)
                   high: std ≈ 0.05 * mean (tight)

    Returns:
        tfp.distributions.Beta distribution ready for PriorDistribution

    Example:
        >>> contribution_m = create_contribution_prior(
        ...     channel_names=['Display', 'TV', 'Video'],
        ...     mean_contributions={'Display': 11.5, 'TV': 30, 'Video': 15},
        ...     confidence='moderate'
        ... )
    """
    # Convert to list if dict
    if isinstance(mean_contributions, dict):
        means = [mean_contributions[ch] for ch in channel_names]
    else:
        means = list(mean_contributions)

    # Convert percentages to proportions
    means = np.array(means) / 100.0

    # Calculate standard deviations
    if std_contributions is not None:
        if isinstance(std_contributions, dict):
            stds = np.array([std_contributions[ch] for ch in channel_names]) / 100.0
        else:
            stds = np.array(std_contributions) / 100.0
    else:
        # Use confidence level
        confidence_map = {
            'low': 0.15,      # std = 15% of mean
            'moderate': 0.10,  # std = 10% of mean
            'high': 0.05,      # std = 5% of mean
        }
        factor = confidence_map.get(confidence, 0.10)
        stds = means * factor

    # Calculate Beta parameters for each channel
    alphas = []
    betas = []

    print(f"means: {means}")
    print(f"stds: {stds}")
    for i, (mean, std) in enumerate(zip(means, stds)):
        # Ensure valid parameters
        if mean <= 0 or mean >= 1:
            raise ValueError(f"Channel {i}: mean must be between 0 and 1, got {mean}")
        if std <= 0:
            raise ValueError(f"Channel {i}: std must be positive, got {std}")

        # Check if variance is achievable
        max_var = mean * (1 - mean)
        if std**2 >= max_var:
            # Reduce std to maximum possible
            std = np.sqrt(max_var * 0.95)
            print(f"Warning: Channel {i} std reduced to {std*100:.2f}% (max possible)")

        print(f"Channel {channel_names[i]}: mean={mean*100:.2f}%, std={std*100:.2f}%")
        alpha, beta = beta_params_from_mean_std(mean, std)
        alphas.append(alpha.round(2).item())
        betas.append(beta.round(2).item())

    print("=" * 80)
    print(f"Contribution Prior Configuration: {confidence}")
    print("=" * 80)
    for i, ch in enumerate(channel_names):
        print(f"{ch:10s}: mean={means[i]*100:5.1f}%, std={stds[i]*100:4.1f}%, "
              f"Beta(α={alphas[i]:.2f}, β={betas[i]:.2f})")
    print(f"\nTotal expected contribution: {np.sum(means)*100:.1f}%")
    print("=" * 80)

    return_dict = {}
    return_dict['alphas'] = dict(zip(channel_names, alphas))
    return_dict['betas'] = dict(zip(channel_names, betas))

    return return_dict


# Example usage
if __name__ == "__main__":
    # Your specific case
    print("\n" + "=" * 80)
    print("YOUR CASE: Display 11.5%, TV 30%, Video 15%")
    print("=" * 80)

    channels = ['Display', 'TV', 'Video']
    expected_contributions = {'Display': 11.5, 'TV': 30, 'Video': 15}

    print("\n--- Moderate Confidence (recommended) ---")
    prior_moderate = create_contribution_prior(
        channel_names=channels,
        mean_contributions=expected_contributions,
        confidence='moderate'
    )

    print("\n--- Low Confidence (more flexible) ---")
    prior_low = create_contribution_prior(
        channel_names=channels,
        mean_contributions=expected_contributions,
        confidence='low'
    )

    print("\n--- High Confidence (tight priors) ---")
    prior_high = create_contribution_prior(
        channel_names=channels,
        mean_contributions=expected_contributions,
        confidence='high'
    )
