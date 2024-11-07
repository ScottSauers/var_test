import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Simulation parameters
n_individuals = 1000
n_genes = 1000 
n_traits = 1000
sparsity = 0.05  # 5% of genes affect each trait
h2 = 0.5  # heritability

# Set random seed
np.random.seed(42)

# Generate true genetic architecture
# First create effect presence/absence matrix (n_genes x n_traits)
effect_exists = np.random.binomial(1, sparsity, size=(n_genes, n_traits))

# Generate actual effect sizes where effects exist
raw_effects = np.random.normal(0, 1, size=(n_genes, n_traits))
genetic_effects = raw_effects * effect_exists

# Scale effects to achieve desired variance
scale_factor = np.sqrt(h2 / (sparsity * n_genes))  # Each trait affected by sparsity*n_genes genes on average
genetic_effects *= scale_factor

# Generate individual genotypes
genotypes = np.random.normal(0, 1, size=(n_individuals, n_genes))

# Calculate genetic values
genetic_values = genotypes @ genetic_effects

# Generate environmental effects
env_effects = np.random.normal(0, np.sqrt(1-h2), size=(n_individuals, n_traits))

# Calculate total phenotypes
phenotypes = genetic_values + env_effects

# Calculate true genetic correlations - theoretical from effect matrix
true_gen_var = genetic_effects.T @ genetic_effects  # Genetic covariance matrix
true_gen_sd = np.sqrt(np.diag(true_gen_var))
true_genetic_corr = true_gen_var / np.outer(true_gen_sd, true_gen_sd)

# Calculate observed correlations from simulated data
obs_genetic_corr = np.corrcoef(genetic_values.T)
obs_pheno_corr = np.corrcoef(phenotypes.T)

def calculate_summary_statistics(true, observed, label):
    """Calculate and print summary statistics comparing matrices"""
    mse = np.mean((observed - true) ** 2)
    mae = np.mean(np.abs(observed - true))
    max_error = np.max(np.abs(observed - true))
    mean_true_mag = np.mean(np.abs(true))
    mean_obs_mag = np.mean(np.abs(observed))
    median_abs_diff = np.median(np.abs(observed - true))
    var_error = np.var(observed - true)
    std_error = np.std(observed - true)
    frob_norm = np.linalg.norm(observed - true, 'fro')
    
    print(f"\n--- Summary Statistics for {label} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Max Error: {max_error:.4f}")
    print(f"Mean True Magnitude: {mean_true_mag:.4f}")
    print(f"Mean Observed Magnitude: {mean_obs_mag:.4f}")
    print(f"Median Absolute Difference: {median_abs_diff:.4f}")
    print(f"Variance of Errors: {var_error:.4f}")
    print(f"Standard Deviation of Errors: {std_error:.4f}")
    print(f"Frobenius Norm of Error: {frob_norm:.4f}")
    print(f"Proportion of Zero Observed Correlations: {np.mean(np.abs(observed) < 1e-10):.4f}")
    print(f"Proportion of Zero True Correlations: {np.mean(np.abs(true) < 1e-10):.4f}")

# Calculate summary statistics
calculate_summary_statistics(true_genetic_corr, obs_genetic_corr, "Genetic Correlations")
calculate_summary_statistics(true_genetic_corr, obs_pheno_corr, "Phenotypic vs Genetic")

# Visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# True Genetic Correlations
cax1 = axs[0, 0].matshow(true_genetic_corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax1, ax=axs[0, 0], fraction=0.046, pad=0.04)
axs[0, 0].set_title('True Genetic Correlation Matrix')
axs[0, 0].set_xlabel('Trait Index')
axs[0, 0].set_ylabel('Trait Index')

# Observed Genetic Correlations
cax2 = axs[0, 1].matshow(obs_genetic_corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax2, ax=axs[0, 1], fraction=0.046, pad=0.04)
axs[0, 1].set_title('Observed Genetic Correlation Matrix')
axs[0, 1].set_xlabel('Trait Index')
axs[0, 1].set_ylabel('Trait Index')

# Observed Phenotypic Correlations
cax3 = axs[1, 0].matshow(obs_pheno_corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax3, ax=axs[1, 0], fraction=0.046, pad=0.04)
axs[1, 0].set_title('Observed Phenotypic Correlation Matrix')
axs[1, 0].set_xlabel('Trait Index')
axs[1, 0].set_ylabel('Trait Index')

# Error in Genetic Correlations
error_mat = obs_genetic_corr - true_genetic_corr
cax4 = axs[1, 1].matshow(error_mat, cmap='coolwarm', vmin=-0.2, vmax=0.2)
fig.colorbar(cax4, ax=axs[1, 1], fraction=0.046, pad=0.04)
axs[1, 1].set_title('Error in Genetic Correlations')
axs[1, 1].set_xlabel('Trait Index')
axs[1, 1].set_ylabel('Trait Index')

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Verify heritability
realized_h2 = np.var(genetic_values, axis=0) / np.var(phenotypes, axis=0)
print(f"\nRealized mean heritability: {np.mean(realized_h2):.3f}")
print(f"Realized heritability std: {np.std(realized_h2):.3f}")

# Verify sparsity
actual_sparsity = np.mean(effect_exists)
print(f"\nRealized sparsity: {actual_sparsity:.3f}")
