"""Generate figures for the MY ice resistance report."""
import sys
sys.path.insert(0, 'src')

import matplotlib.pyplot as plt
import numpy as np
from tensordem.mixed_ice.field import MixedFieldSpec
from tensordem.mixed_ice.resistance import mixed_resistance
from tensordem.mixed_ice.literature_validation import validate_my_resistance, compute_validation_metrics
from tensordem.mixed_ice.ridge_resistance import RidgeFieldSpec, mixed_field_with_ridges
from tensordem.level_ice.formula import ICEBREAKER

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Figure 1: Resistance vs MY fraction
print("Generating Figure 1: Resistance vs MY fraction...")
my_fractions = np.linspace(0, 1, 11)
resistances = []
for p in my_fractions:
    spec = MixedFieldSpec(my_fraction=p)
    result = mixed_resistance(spec, speed_m_s=3.0)
    resistances.append(result.resistance_mean_n / 1e3)  # Convert to kN

fig, ax = plt.subplots()
ax.plot(my_fractions * 100, resistances, 'b-o', linewidth=2, markersize=8)
ax.set_xlabel('MY Ice Fraction [%]', fontsize=14)
ax.set_ylabel('Resistance [kN]', fontsize=14)
ax.set_title('Ice Resistance vs Multi-Year Ice Fraction', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
plt.tight_layout()
plt.savefig('artifacts/fig1_resistance_vs_my_fraction.png', dpi=150)
plt.close()

# Figure 2: Validation against literature data
print("Generating Figure 2: Validation against literature data...")
validation = validate_my_resistance()
metrics = compute_validation_metrics(validation)

fig, ax = plt.subplots()
measured = [r['measured_n'] / 1e3 for r in validation['results']]
predicted = [r['predicted_n'] / 1e3 for r in validation['results']]
errors = [r['relative_error'] * 100 for r in validation['results']]
case_ids = [r['case_id'] for r in validation['results']]

x = np.arange(len(case_ids))
width = 0.35
bars1 = ax.bar(x - width/2, measured, width, label='Measured (Literature)', color='steelblue')
bars2 = ax.bar(x + width/2, predicted, width, label='Predicted (Model)', color='coral')

ax.set_xlabel('Validation Case', fontsize=14)
ax.set_ylabel('Resistance [kN]', fontsize=14)
ax.set_title('Model Validation Against Literature Data', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(case_ids, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('artifacts/fig2_validation.png', dpi=150)
plt.close()

# Figure 3: Error analysis
print("Generating Figure 3: Error analysis...")
fig, ax = plt.subplots()
colors = ['green' if abs(e) <= 15 else 'red' for e in errors]
ax.bar(x, errors, color=colors, alpha=0.7)
ax.axhline(y=15, color='r', linestyle='--', linewidth=2, label='15% threshold')
ax.axhline(y=-15, color='r', linestyle='--', linewidth=2)
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.set_xlabel('Validation Case', fontsize=14)
ax.set_ylabel('Relative Error [%]', fontsize=14)
ax.set_title('Model Error vs Literature Data', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(case_ids, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('artifacts/fig3_error_analysis.png', dpi=150)
plt.close()

# Figure 4: Ice ridge contribution
print("Generating Figure 4: Ice ridge contribution...")
ridge_densities = [0, 1, 3, 5, 10]
level_resistances = []
ridge_resistances = []
total_resistances = []

for density in ridge_densities:
    spec = RidgeFieldSpec(my_fraction=0.4, ridge_density_per_km=density)
    result = mixed_field_with_ridges(spec, speed_m_s=3.0)
    level_resistances.append(result['level_resistance_N'] / 1e3)
    ridge_resistances.append(result['ridge_resistance_avg_N'] / 1e3)
    total_resistances.append(result['total_resistance_N'] / 1e3)

fig, ax = plt.subplots()
x = np.arange(len(ridge_densities))
width = 0.35
bars1 = ax.bar(x - width/2, level_resistances, width, label='Level Ice', color='steelblue')
bars2 = ax.bar(x + width/2, ridge_resistances, width, label='Ridge Contribution', color='coral')
ax.plot(x, total_resistances, 'k-o', linewidth=2, markersize=8, label='Total')

ax.set_xlabel('Ridge Density [ridges/km]', fontsize=14)
ax.set_ylabel('Resistance [kN]', fontsize=14)
ax.set_title('Ice Resistance with Ridges (40% MY)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(ridge_densities)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('artifacts/fig4_ridge_contribution.png', dpi=150)
plt.close()

# Figure 5: Component breakdown
print("Generating Figure 5: Component breakdown...")
my_fractions_plot = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
crushing = []
bending = []
submersion = []
drag = []

for p in my_fractions_plot:
    spec = MixedFieldSpec(my_fraction=p)
    result = mixed_resistance(spec, speed_m_s=3.0)
    shares = result.component_shares
    total = result.resistance_mean_n
    crushing.append(shares['crushing'] * total / 1e3)
    bending.append(shares['bending'] * total / 1e3)
    submersion.append(shares['submersion'] * total / 1e3)
    drag.append(shares['drag'] * total / 1e3)

fig, ax = plt.subplots()
x = np.arange(len(my_fractions_plot))
width = 0.2
ax.bar(x - 1.5*width, crushing, width, label='Crushing', color='darkred')
ax.bar(x - 0.5*width, bending, width, label='Bending', color='orange')
ax.bar(x + 0.5*width, submersion, width, label='Submersion', color='steelblue')
ax.bar(x + 1.5*width, drag, width, label='Drag', color='lightblue')

ax.set_xlabel('MY Ice Fraction', fontsize=14)
ax.set_ylabel('Resistance Component [kN]', fontsize=14)
ax.set_title('Resistance Component Breakdown', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([f'{p*100:.0f}%' for p in my_fractions_plot])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('artifacts/fig5_component_breakdown.png', dpi=150)
plt.close()

print("All figures generated successfully!")
print("Figures saved to artifacts/")
