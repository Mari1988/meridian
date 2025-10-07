"""Visualize Beta distribution priors for contribution."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Display channel: 11.5% with different confidence levels
mean = 0.115

confidence_configs = {
    'Low': (39.22, 301.81, 'blue'),
    'Moderate': (88.38, 680.18, 'green'),
    'High': (353.88, 2723.38, 'red'),
}

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Probability density functions
ax1 = axes[0]
x = np.linspace(0.05, 0.20, 1000)

for name, (alpha, beta, color) in confidence_configs.items():
    dist = stats.beta(alpha, beta)
    pdf = dist.pdf(x)
    ax1.plot(x * 100, pdf / 100, label=f'{name}: Beta({alpha:.1f}, {beta:.1f})',
             color=color, linewidth=2)

    # Mark mean
    ax1.axvline(mean * 100, color='black', linestyle='--', alpha=0.3, linewidth=1)

ax1.set_xlabel('Contribution (%)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Display Channel: Beta Prior with Different Confidence Levels', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axvline(11.5, color='black', linestyle='--', label='Mean = 11.5%', linewidth=2, alpha=0.5)

# Plot 2: Cumulative distribution functions (easier to read percentiles)
ax2 = axes[1]

for name, (alpha, beta, color) in confidence_configs.items():
    dist = stats.beta(alpha, beta)
    cdf = dist.cdf(x)
    ax2.plot(x * 100, cdf, label=f'{name}', color=color, linewidth=2)

# Add reference lines for key percentiles
percentile_lines = [0.05, 0.25, 0.5, 0.75, 0.95]
percentile_labels = ['5%', '25%', '50%', '75%', '95%']
for p, label in zip(percentile_lines, percentile_labels):
    ax2.axhline(p, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.text(19.5, p, label, fontsize=9, va='center')

ax2.axvline(11.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Contribution (%)', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Distribution: How likely is contribution ≤ x%?', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5, 20)

plt.tight_layout()
plt.savefig('/Users/mariappan.subramanian/Documents/repo/forked/meridian/meridian/model/beta_prior_visualization.png', dpi=150)
print("✓ Visualization saved to: meridian/model/beta_prior_visualization.png")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY: Display Channel (Mean = 11.5%)")
print("=" * 80)
print(f"\n{'Confidence':<12} {'α':<10} {'β':<10} {'Std':<8} {'90% CI':<20} {'Range'}")
print("-" * 80)

for name, (alpha, beta, color) in confidence_configs.items():
    dist = stats.beta(alpha, beta)
    std = dist.std() * 100
    p5 = dist.ppf(0.05) * 100
    p95 = dist.ppf(0.95) * 100
    range_val = p95 - p5

    print(f"{name:<12} {alpha:<10.2f} {beta:<10.2f} {std:<8.2f} "
          f"[{p5:5.2f}%, {p95:5.2f}%]   {range_val:5.2f}%")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
• LOW confidence: Wide range [8.80%, 14.46%] - lets data override prior
  → Use when you're uncertain about the 11.5% estimate

• MODERATE confidence: Balanced [9.67%, 13.45%] - reasonable flexibility
  → Use when 11.5% is your best estimate but data can adjust it

• HIGH confidence: Tight range [10.57%, 12.46%] - strongly anchors near 11.5%
  → Use when you're very confident in the 11.5% estimate

The model will update these priors based on observed data during MCMC sampling.
""")
