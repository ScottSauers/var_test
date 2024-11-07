import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

n_individuals = 1000
n_genes = 1000
n_phenotypes = 1000
sparsity = 0.05

np.random.seed(42)

genomes = np.random.normal(0, 1, size=(n_individuals, n_genes))

effects = np.random.normal(0, np.sqrt(1/n_genes), size=(n_genes, n_phenotypes))

phenotypes_genetic = genomes @ effects

environmental_noise = np.random.normal(0, 1, size=(n_individuals, n_phenotypes))
phenotypes_total = phenotypes_genetic + environmental_noise

true_genetic_correlations = np.corrcoef(phenotypes_genetic.T)
observed_genetic_correlations = np.corrcoef(phenotypes_genetic.T)
observed_phenotypic_correlations = np.corrcoef(phenotypes_total.T)

observed_genetic_correlations = (observed_genetic_correlations + observed_genetic_correlations.T) / 2
observed_phenotypic_correlations = (observed_phenotypic_correlations + observed_phenotypic_correlations.T) / 2

def calculate_summary_statistics(true, observed, label):
    mse = np.mean((observed.flatten() - true.flatten()) ** 2)
    mae = np.mean(np.abs(observed.flatten() - true.flatten()))
    max_error = np.max(np.abs(observed.flatten() - true.flatten()))
    mean_true_mag = np.mean(np.abs(true.flatten()))
    mean_observed_mag = np.mean(np.abs(observed.flatten()))
    median_abs_diff = np.median(np.abs(observed.flatten() - true.flatten()))
    var_error = np.var(observed.flatten() - true.flatten())
    corr_error_sd = np.std(observed.flatten() - true.flatten())
    frobenius_norm = np.linalg.norm(observed - true, 'fro')
    
    print(f"\n--- Summary Statistics for {label} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Max Error: {max_error:.4f}")
    print(f"Mean True Magnitude: {mean_true_mag:.4f}")
    print(f"Mean Observed Magnitude: {mean_observed_mag:.4f}")
    print(f"Median Absolute Difference: {median_abs_diff:.4f}")
    print(f"Variance of Errors: {var_error:.4f}")
    print(f"Standard Deviation of Correlation Errors: {corr_error_sd:.4f}")
    print(f"Frobenius Norm of Error: {frobenius_norm:.4f}")
    print(f"Proportion of Zero Observed Correlations: {(observed == 0).mean():.4f}")
    print(f"Proportion of Zero True Correlations: {(true == 0).mean():.4f}")

calculate_summary_statistics(true_genetic_correlations, observed_genetic_correlations, "Genetic")
calculate_summary_statistics(true_genetic_correlations, observed_phenotypic_correlations, "Phenotypic")

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

cax1 = axs[0, 0].matshow(observed_phenotypic_correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax1, ax=axs[0, 0], fraction=0.046, pad=0.04)
axs[0, 0].set_title('Observed Phenotypic Correlation Matrix')
axs[0, 0].set_xlabel('Phenotype Index')
axs[0, 0].set_ylabel('Phenotype Index')

for i in range(n_phenotypes):
    for j in range(n_phenotypes):
        if true_genetic_correlations[i % n_genes, j % n_genes] != 0:
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            axs[0, 0].add_patch(rect)

mean_true_mag_phen = np.mean(np.abs(true_genetic_correlations))
mean_observed_phenotypic_mag = np.mean(np.abs(observed_phenotypic_correlations))

axs[0, 1].bar(['Observed', 'True'], [mean_observed_phenotypic_mag, mean_true_mag_phen], color=['blue', 'orange'])
axs[0, 1].set_title('Phenotypic Correlation Magnitude Comparison')
axs[0, 1].set_ylabel('Correlation Magnitude')

cax2 = axs[1, 0].matshow(observed_genetic_correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax2, ax=axs[1, 0], fraction=0.046, pad=0.04)
axs[1, 0].set_title('Observed Genetic Correlation Matrix')
axs[1, 0].set_xlabel('Phenotype Index')
axs[1, 0].set_ylabel('Phenotype Index')

for i in range(n_genes):
    for j in range(n_genes):
        if true_genetic_correlations[i, j] != 0:
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            axs[1, 0].add_patch(rect)

mean_true_mag_gen = np.mean(np.abs(true_genetic_correlations))
mean_observed_genetic_mag = np.mean(np.abs(observed_genetic_correlations))

axs[1, 1].bar(['Observed', 'True'], [mean_observed_genetic_mag, mean_true_mag_gen], color=['blue', 'orange'])
axs[1, 1].set_title('Genetic Correlation Magnitude Comparison')
axs[1, 1].set_ylabel('Correlation Magnitude')

plt.tight_layout()
plt.savefig('correlation_analysis.png')
plt.show()
