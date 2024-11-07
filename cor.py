import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
n_individuals = 1000  # Number of individuals (sample size)
n_genes = 1000  # Number of genes
n_phenotypes = 1000  # Number of phenotypes
sparsity = 0.05  # Proportion of true correlations (sparsity level)

# Set a random seed for reproducibility
np.random.seed(42)

# Step 1: Generate Random Genotypes
genomes = np.random.normal(0, 1, size=(n_individuals, n_genes))

# Step 2: Create Polygenic Effects for Each Phenotype
effects = np.random.normal(0, 0.1, size=(n_genes, n_phenotypes))

# Step 3: Define Ground Truth Correlation Structures
# 3a: True Genetic Correlation Structure (Sparse)
true_genetic_correlations = np.zeros((n_genes, n_genes))
upper_triangle_indices = np.triu_indices(n_genes, k=1)

# Apply sparsity to the upper triangle for genetic correlations
genetic_random_values = np.random.normal(0.5, 0.2, size=upper_triangle_indices[0].shape)
genetic_sparse_values = genetic_random_values * (np.random.rand(len(genetic_random_values)) < sparsity)
true_genetic_correlations[upper_triangle_indices] = genetic_sparse_values

# Make genetic correlation matrix symmetric and set diagonal elements to 1
true_genetic_correlations = true_genetic_correlations + true_genetic_correlations.T
np.fill_diagonal(true_genetic_correlations, 1)

# 3b: True Phenotypic Correlation Structure (Sparse)
true_phenotypic_correlations = np.zeros((n_phenotypes, n_phenotypes))
upper_triangle_indices = np.triu_indices(n_phenotypes, k=1)

# Apply sparsity to the upper triangle for phenotypic correlations
phenotypic_random_values = np.random.normal(0.5, 0.2, size=upper_triangle_indices[0].shape)
phenotypic_sparse_values = phenotypic_random_values * (np.random.rand(len(phenotypic_random_values)) < sparsity)
true_phenotypic_correlations[upper_triangle_indices] = phenotypic_sparse_values

# Make phenotypic correlation matrix symmetric and set diagonal elements to 1
true_phenotypic_correlations = true_phenotypic_correlations + true_phenotypic_correlations.T
np.fill_diagonal(true_phenotypic_correlations, 1)

# Step 4: Generate Phenotypes Based on Genomic Effects
phenotypes = genomes @ effects

# Step 5: Compute Observed Correlation Matrices
# Observed Phenotypic Correlation Matrix
observed_phenotypic_correlations = np.corrcoef(phenotypes, rowvar=False)
observed_phenotypic_correlations = (observed_phenotypic_correlations + observed_phenotypic_correlations.T) / 2

# Observed Genetic Correlation Matrix (correlations between effects)
observed_genetic_correlations = np.corrcoef(effects, rowvar=False)
observed_genetic_correlations = (observed_genetic_correlations + observed_genetic_correlations.T) / 2

# Step 6: Summary Statistics Calculation Function
def calculate_summary_statistics(true, observed, label):
    true_vals = true.flatten()
    observed_vals = observed.flatten()

    mse = np.mean((observed_vals - true_vals) ** 2)
    mae = np.mean(np.abs(observed_vals - true_vals))
    max_error = np.max(np.abs(observed_vals - true_vals))

    mean_true_mag = np.mean(np.abs(true_vals))
    mean_observed_mag = np.mean(np.abs(observed_vals))

    median_abs_diff = np.median(np.abs(observed_vals - true_vals))
    var_error = np.var(observed_vals - true_vals)
    corr_error_sd = np.std(observed_vals - true_vals)

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

# Step 7: Calculate and Print Summary Statistics
print("Phenotypic Correlation Matrix Summary:")
calculate_summary_statistics(true_phenotypic_correlations, observed_phenotypic_correlations, "Phenotypic")

print("\nGenetic Correlation Matrix Summary:")
calculate_summary_statistics(true_genetic_correlations, observed_genetic_correlations, "Genetic")

# Step 8: Plotting the Phenotypic and Genetic Correlation Matrices
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Phenotypic Correlation Heatmap
cax1 = axs[0, 0].matshow(observed_phenotypic_correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax1, ax=axs[0, 0], fraction=0.046, pad=0.04)
axs[0, 0].set_title('Observed Phenotypic Correlation Matrix')
axs[0, 0].set_xlabel('Phenotype Index')
axs[0, 0].set_ylabel('Phenotype Index')

# Outline squares for True Correlations (Phenotypic)
for i in range(n_phenotypes):
    for j in range(n_phenotypes):
        if true_phenotypic_correlations[i, j] != 0:
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            axs[0, 0].add_patch(rect)

# Phenotypic Correlation Magnitude Comparison
mean_true_mag_phen = np.mean(np.abs(true_phenotypic_correlations))
mean_observed_phenotypic_mag = np.mean(np.abs(observed_phenotypic_correlations))

axs[0, 1].bar(['Observed', 'True'], [mean_observed_phenotypic_mag, mean_true_mag_phen], color=['blue', 'orange'])
axs[0, 1].set_title('Phenotypic Correlation Magnitude Comparison')
axs[0, 1].set_ylabel('Correlation Magnitude')

# Genetic Correlation Heatmap
cax2 = axs[1, 0].matshow(observed_genetic_correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax2, ax=axs[1, 0], fraction=0.046, pad=0.04)
axs[1, 0].set_title('Observed Genetic Correlation Matrix')
axs[1, 0].set_xlabel('Gene Index')
axs[1, 0].set_ylabel('Gene Index')

# Outline squares for True Correlations (Genetic)
for i in range(n_genes):
    for j in range(n_genes):
        if true_genetic_correlations[i, j] != 0:
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            axs[1, 0].add_patch(rect)

# Genetic Correlation Magnitude Comparison
mean_true_mag_gen = np.mean(np.abs(true_genetic_correlations))
mean_observed_genetic_mag = np.mean(np.abs(observed_genetic_correlations))

axs[1, 1].bar(['Observed', 'True'], [mean_observed_genetic_mag, mean_true_mag_gen], color=['blue', 'orange'])
axs[1, 1].set_title('Genetic Correlation Magnitude Comparison')
axs[1, 1].set_ylabel('Correlation Magnitude')

plt.tight_layout()
plt.savefig('correlation_analysis_full.png')
plt.show()
