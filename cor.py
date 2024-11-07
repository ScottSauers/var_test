import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

# Parameters
n_individuals = 1000  # Sample size
n_genes = 1000  # Number of genes
n_phenotypes = 1000  # Number of phenotypes
sparsity = 0.05  # Proportion of true correlations (sparsity level)

# Set random seed
np.random.seed(42)

# Step 1: Generate Random Genotypes
genomes = np.random.normal(0, 1, size=(n_individuals, n_genes))

# Step 2: Generate Polygenic Effects
effects = np.random.normal(0, 0.1, size=(n_genes, n_phenotypes))

# Step 3: Define Sparse True Genetic Correlation Structure
true_genetic_correlations = np.zeros((n_genes, n_genes))
upper_triangle_indices = np.triu_indices(n_genes, k=1)

# Sparse structure for genetic correlations
genetic_random_values = np.random.normal(0.5, 0.2, size=upper_triangle_indices[0].shape)
genetic_sparse_values = genetic_random_values * (np.random.rand(len(genetic_random_values)) < sparsity)
true_genetic_correlations[upper_triangle_indices] = genetic_sparse_values

# Make symmetric and set diagonal elements to 1
true_genetic_correlations = true_genetic_correlations + true_genetic_correlations.T
np.fill_diagonal(true_genetic_correlations, 1)

# Step 4: Generate Phenotypes Using Genetic and Environmental Effects
phenotypes_genetic = genomes @ effects  # Genetic contribution
environmental_noise = np.random.normal(0, 0.1, size=(n_individuals, n_phenotypes))
phenotypes_total = phenotypes_genetic + environmental_noise  # Phenotypes with both effects

# Step 5: Observed Genetic and Phenotypic Correlation Matrices
# Calculate genetic correlations by regressing out environmental noise
# Here we apply PCA to approximate the genetic contribution alone
pca = PCA(n_components=n_phenotypes)
genetic_effects_isolated = pca.fit_transform(phenotypes_total)

observed_phenotypic_correlations = np.corrcoef(phenotypes_total, rowvar=False)
observed_genetic_correlations = np.corrcoef(genetic_effects_isolated, rowvar=False)

# Symmetrize correlation matrices
observed_phenotypic_correlations = (observed_phenotypic_correlations + observed_phenotypic_correlations.T) / 2
observed_genetic_correlations = (observed_genetic_correlations + observed_genetic_correlations.T) / 2

# Step 6: Summary Statistics Calculation
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

# Step 7: Summary Statistics Output
calculate_summary_statistics(true_genetic_correlations, observed_genetic_correlations, "Genetic")
calculate_summary_statistics(true_genetic_correlations, observed_phenotypic_correlations, "Phenotypic")

# Step 8: Visualization
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
        if true_genetic_correlations[i % n_genes, j % n_genes] != 0:
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            axs[0, 0].add_patch(rect)

# Phenotypic Correlation Magnitude Comparison
mean_true_mag_phen = np.mean(np.abs(true_genetic_correlations))
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
plt.savefig('correlation_analysis_corrected.png')
plt.show()
