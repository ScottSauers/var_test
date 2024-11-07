import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
n_individuals = 500  # Number of individuals (sample size)
n_genes = 1000  # Number of genes
n_phenotypes = 50  # Number of phenotypes
sparsity = 0.1  # Proportion of true correlations (sparseness level)

# Set a random seed for reproducibility
np.random.seed(42)

# Step 1: Generate Random Genotypes
genomes = np.random.normal(0, 1, size=(n_individuals, n_genes))

# Step 2: Create Polygenic Effects for Each Phenotype
effects = np.random.normal(0, 0.1, size=(n_genes, n_phenotypes))

# Step 3: Define Symmetric True Correlation Structure (Sparse)
# Generate an upper triangular matrix with random values where mask applies
true_correlations = np.zeros((n_phenotypes, n_phenotypes))
upper_triangle_indices = np.triu_indices(n_phenotypes, k=1)  # Indices for the upper triangle

# Apply sparsity to the upper triangle
random_values = np.random.normal(0.5, 0.2, size=upper_triangle_indices[0].shape)
sparse_values = random_values * (np.random.rand(len(random_values)) < sparsity)
true_correlations[upper_triangle_indices] = sparse_values

# Make the matrix symmetric by reflecting the upper triangle to the lower triangle
true_correlations = true_correlations + true_correlations.T
np.fill_diagonal(true_correlations, 1)  # Set diagonal to 1

# Step 4: Generate Phenotypes Based on Genomic Effects
phenotypes = genomes @ effects  # Phenotypes influenced by genomic data (no explicit noise)

# Step 5: Compute Observed Correlations Based on Sample (Sampling Error Included)
observed_correlations = np.corrcoef(phenotypes, rowvar=False)

# Enforce symmetry on the observed correlation matrix to ensure consistency with true matrix
observed_correlations = (observed_correlations + observed_correlations.T) / 2

# Step 6: Calculate and Print Summary Statistics, Including All Pairs
def calculate_summary_statistics(true, observed):
    # Flatten matrices to calculate metrics on all pairs (including zeros)
    true_vals = true.flatten()
    observed_vals = observed.flatten()

    # Error metrics
    mse = np.mean((observed_vals - true_vals) ** 2)
    mae = np.mean(np.abs(observed_vals - true_vals))
    msae = np.mean((np.abs(observed_vals) - np.abs(true_vals)) ** 2)
    max_error = np.max(np.abs(observed_vals - true_vals))

    # Correlation magnitudes (all pairs included)
    mean_true_mag = np.mean(np.abs(true_vals))
    mean_observed_mag = np.mean(np.abs(observed_vals))

    # Pairwise differences
    median_abs_diff = np.median(np.abs(observed_vals - true_vals))
    var_error = np.var(observed_vals - true_vals)
    corr_error_sd = np.std(observed_vals - true_vals)

    # Frobenius norms
    frobenius_norm = np.linalg.norm(observed - true, 'fro')
    frobenius_norm_magnitude = np.linalg.norm(np.abs(observed) - np.abs(true), 'fro')

    print("\n--- Summary Statistics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Absolute Error (MSAE): {msae:.4f}")
    print(f"Max Error: {max_error:.4f}")
    print(f"Mean True Magnitude: {mean_true_mag:.4f}")
    print(f"Mean Observed Magnitude: {mean_observed_mag:.4f}")
    print(f"Median Absolute Difference: {median_abs_diff:.4f}")
    print(f"Variance of Errors: {var_error:.4f}")
    print(f"Standard Deviation of Correlation Errors: {corr_error_sd:.4f}")
    print(f"Frobenius Norm of Error: {frobenius_norm:.4f}")
    print(f"Frobenius Norm of Magnitude Error: {frobenius_norm_magnitude:.4f}")
    print(f"Proportion of Zero Observed Correlations: {(observed == 0).mean():.4f}")
    print(f"Proportion of Zero True Correlations: {(true == 0).mean():.4f}")

    return mse, mae

# Run summary statistics calculation
mse, mae = calculate_summary_statistics(true_correlations, observed_correlations)

# Step 7: Plot the Observed Correlation Matrix and Comparison Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Observed Correlation Heatmap
cax1 = ax1.matshow(observed_correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('Observed Correlation Matrix')
ax1.set_xlabel('Phenotype Index')
ax1.set_ylabel('Phenotype Index')

# Outline squares for True Correlations
for i in range(n_phenotypes):
    for j in range(n_phenotypes):
        if true_correlations[i, j] != 0:  # Only outline true correlations
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
            ax1.add_patch(rect)

# Bar Chart of Mean Observed vs True Correlation Magnitudes
mean_true_mag = np.mean(np.abs(true_correlations))
mean_observed_mag = np.mean(np.abs(observed_correlations))

ax2.bar(['Observed', 'True'], [mean_observed_mag, mean_true_mag], color=['blue', 'orange'])
ax2.set_title(f'Correlation Magnitude Comparison\nMSE: {mse:.4f}, MAE: {mae:.4f}')
ax2.set_ylabel('Correlation Magnitude')

plt.tight_layout()
plt.savefig('correlation_analysis.png')
plt.show()
