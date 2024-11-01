import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import t
import multiprocessing as mp
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

print("Starting simulation pipeline...")

# Set random seed for reproducibility
np.random.seed(2024)

# Parameters
print("\nInitializing simulation parameters...")
n_snps = 1000
n_individuals_A = 5000
n_individuals_B = 1000
n_individuals = n_individuals_A + n_individuals_B
n_causal_snps = int(0.2 * n_snps)
snp_indices = np.arange(n_snps)
causal_snps = np.random.choice(snp_indices, n_causal_snps, replace=False)
print(f"Total SNPs: {n_snps}")
print(f"Population A size: {n_individuals_A}")
print(f"Population B size: {n_individuals_B}")
print(f"Number of causal SNPs: {n_causal_snps}")

# Simulate allele frequencies
print("\nSimulating allele frequencies...")
p_A = np.random.uniform(0.1, 0.9, n_snps)
print("Population A allele frequencies generated")

# Adjust frequencies for population B
print("Adjusting allele frequencies for Population B...")
diff_snps = np.random.choice(snp_indices, int(0.5 * n_snps), replace=False)
delta_p = np.random.uniform(-0.2, 0.2, int(0.5 * n_snps))
p_B = p_A.copy()
p_B[diff_snps] += delta_p
p_B = np.clip(p_B, 0.01, 0.99)
print(f"Adjusted {len(diff_snps)} SNPs for Population B")

# Simulate genotypes
print("\nSimulating genotypes...")
print("Generating Population A genotypes...")
genotypes_A = np.array([np.random.binomial(2, p_A) for _ in range(n_individuals_A)])
print("Generating Population B genotypes...")
genotypes_B = np.array([np.random.binomial(2, p_B) for _ in range(n_individuals_B)])
genotypes = np.vstack([genotypes_A, genotypes_B])
pop_labels = np.array(['A'] * n_individuals_A + ['B'] * n_individuals_B)
print("Genotype simulation complete")

# Genetic effects
print("\nAssigning genetic effects...")
beta = np.zeros(n_snps)
beta[causal_snps] = np.random.normal(0, 1, n_causal_snps)
print(f"Assigned effect sizes to {n_causal_snps} causal SNPs")

print("Validating effect sizes...")
print(f"Mean effect size: {np.mean(beta[causal_snps]):.4f}")
print(f"SD effect size: {np.std(beta[causal_snps]):.4f}")

# Compute genetic values
print("Computing genetic values...")
G = genotypes.dot(beta)
var_G = np.var(G)
var_E = var_G  # Set environmental variance for h^2 = 0.5
print(f"Genetic variance: {var_G:.4f}")

# Add this after computing genetic values
print(f"Checking heritability...")
actual_h2 = var_G / (var_G + var_E)
print(f"Target h2: 0.5, Actual h2: {actual_h2:.4f}")

# Environmental effects
print("\nSimulating environmental effects...")
E_A = np.random.normal(0, np.sqrt(var_E), n_individuals_A)
E_B = np.random.normal(0, np.sqrt(var_E), n_individuals_B)

# Population B specific environmental effect
print("Adding population-specific environmental effect to Population B...")
mean_env_effect = -1.0
sd_env_effect = 0.5
E_env = np.random.normal(mean_env_effect, sd_env_effect, n_individuals_B)
E_B += E_env
E = np.concatenate([E_A, E_B])
print(f"Environmental effect mean: {mean_env_effect}, SD: {sd_env_effect}")

# Total phenotype
print("\nCalculating total phenotypes...")
Y = G + E
print("Phenotype calculation complete")

print("\nComputing population structure PCs...")
# Center the genotypes, but don't scale yet
genotypes_centered = genotypes - genotypes.mean(axis=0)

# Use TruncatedSVD which is much faster for just a few components
svd = TruncatedSVD(n_components=5, random_state=42)
PCs = svd.fit_transform(genotypes_centered)
print(f"Variance explained by PCs: {svd.explained_variance_ratio_}")
print("PC computation complete")

# GWAS function with covariates
def perform_gwas_covariates(genotypes, phenotype, covariates=None):
    print("Starting GWAS...")
    n_snps = genotypes.shape[1]
    betas = np.zeros(n_snps)
    
    # Standardize phenotype
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()
    
    if covariates is None:
        print("Running GWAS without covariates")
        # Standardize genotypes
        genotypes_std = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
        # Simple regression for each SNP
        betas = np.array([np.dot(genotypes_std[:, i], phenotype) / len(phenotype) 
                         for i in tqdm(range(n_snps), desc="Processing SNPs")])
    else:
        print("Running GWAS with covariates")
        for i in tqdm(range(n_snps), desc="Processing SNPs"):
            # Standardize genotype
            X = genotypes[:, i]
            X = (X - X.mean()) / X.std()
            # Add covariates
            X = np.column_stack((X, covariates))
            reg = LinearRegression().fit(X, phenotype)
            betas[i] = reg.coef_[0]
    
    print("GWAS complete")
    return betas

# Task 2: GWAS on population A alone
print("\nTask 2: Performing GWAS on Population A alone...")
genotypes_A = genotypes[:n_individuals_A]
Y_A = Y[:n_individuals_A]
betas_A = perform_gwas_covariates(genotypes_A, Y_A)
PRS_A = genotypes_A.dot(betas_A)
r2_A = r2_score(Y_A, PRS_A)
print(f'R^2 in Population A (GWAS on A alone): {r2_A:.4f}')

# Task 3: GWAS on all data
print("\nTask 3: Performing GWAS on all data...")
betas_all = perform_gwas_covariates(genotypes, Y)
PRS_all = genotypes.dot(betas_all)
PRS_A_all = PRS_all[:n_individuals_A]
r2_A_all = r2_score(Y_A, PRS_A_all)
PRS_B_all = PRS_all[n_individuals_A:]
Y_B = Y[n_individuals_A:]
r2_B_all = r2_score(Y_B, PRS_B_all)
print(f'R^2 in Population A (GWAS on all data): {r2_A_all:.4f}')
print(f'R^2 in Population B (GWAS on all data): {r2_B_all:.4f}')

# Task 4: Remove environmental effect
print("\nTask 4: Redoing GWAS without environmental effect...")
E_B_noenv = np.random.normal(0, np.sqrt(var_E), n_individuals_B)
E_noenv = np.concatenate([E_A, E_B_noenv])
Y_noenv = G + E_noenv
betas_noenv = perform_gwas_covariates(genotypes, Y_noenv)
PRS_all_noenv = genotypes.dot(betas_noenv)
PRS_A_noenv = PRS_all_noenv[:n_individuals_A]
Y_A_noenv = Y_noenv[:n_individuals_A]
r2_A_noenv = r2_score(Y_A_noenv, PRS_A_noenv)
PRS_B_noenv = PRS_all_noenv[n_individuals_A:]
Y_B_noenv = Y_noenv[n_individuals_A:]
r2_B_noenv = r2_score(Y_B_noenv, PRS_B_noenv)
print(f'R^2 in Population A (no env effect): {r2_A_noenv:.4f}')
print(f'R^2 in Population B (no env effect): {r2_B_noenv:.4f}')

# Task 4.5: Compare results
print("\nTask 4.5: Comparing environmental effect impact...")
env_effect_diff = r2_B_all - r2_B_noenv
print(f'Increase in R^2 for Population B due to environmental effect: {env_effect_diff:.4f}')

# Task 5: GWAS with PC control
print("\nTask 5: Performing GWAS with PC control...")
betas_pc = perform_gwas_covariates(genotypes, Y, PCs)
PRS_pc = genotypes.dot(betas_pc)
PRS_A_pc = PRS_pc[:n_individuals_A]
r2_A_pc = r2_score(Y_A, PRS_A_pc)
PRS_B_pc = PRS_pc[n_individuals_A:]
r2_B_pc = r2_score(Y_B, PRS_B_pc)
print(f'R^2 in Population A (GWAS controlling for PCs): {r2_A_pc:.4f}')
print(f'R^2 in Population B (GWAS controlling for PCs): {r2_B_pc:.4f}')

# Task 6.5: Compare differences
print("\nTask 6.5: Comparing PC control effects...")
diff_r2_A = r2_A_all - r2_A_pc
diff_r2_B = r2_B_all - r2_B_pc
print(f'Difference in R^2 for Population A: {diff_r2_A:.4f}')
print(f'Difference in R^2 for Population B: {diff_r2_B:.4f}')

# Parallel bootstrap function
def bootstrap_iteration(args):
    idx, seed = args
    print(f"\nBootstrap iteration {idx+1}/1000")
    
    np.random.seed(seed)

    idx = np.random.choice(n_individuals, n_individuals, replace=True)
    genotypes_boot = genotypes[idx]
    Y_boot = Y[idx]
    PCs_boot = PCs[idx]
    
    # GWAS without PCs
    betas_all_boot = perform_gwas_covariates(genotypes_boot, Y_boot)
    PRS_all_boot = genotypes_boot.dot(betas_all_boot)
    
    # GWAS with PCs
    betas_pc_boot = perform_gwas_covariates(genotypes_boot, Y_boot, PCs_boot)
    PRS_pc_boot = genotypes_boot.dot(betas_pc_boot)
    
    # Compute r^2 in A and B
    idx_A = np.where(pop_labels[idx] == 'A')[0]
    idx_B = np.where(pop_labels[idx] == 'B')[0]
    
    Y_A_boot = Y_boot[idx_A]
    Y_B_boot = Y_boot[idx_B]
    PRS_A_all_boot = PRS_all_boot[idx_A]
    PRS_B_all_boot = PRS_all_boot[idx_B]
    PRS_A_pc_boot = PRS_pc_boot[idx_A]
    PRS_B_pc_boot = PRS_pc_boot[idx_B]
    
    r2_A_all_boot = r2_score(Y_A_boot, PRS_A_all_boot)
    r2_B_all_boot = r2_score(Y_B_boot, PRS_B_all_boot)
    r2_A_pc_boot = r2_score(Y_A_boot, PRS_A_pc_boot)
    r2_B_pc_boot = r2_score(Y_B_boot, PRS_B_pc_boot)
    
    return (r2_A_all_boot - r2_A_pc_boot, r2_B_all_boot - r2_B_pc_boot)

if __name__ == '__main__':
    # Bootstrap analysis
    print("\nPerforming bootstrap analysis...")
    n_bootstraps = 5
    n_cores = mp.cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(bootstrap_iteration, range(n_bootstraps)),
                           total=n_bootstraps,
                           desc="Bootstrap progress"))
    
    diff_r2_A_boot, diff_r2_B_boot = zip(*results)
    diff_r2_A_boot = np.array(diff_r2_A_boot)
    diff_r2_B_boot = np.array(diff_r2_B_boot)
    
    # Calculate confidence intervals
    ci_diff_r2_A = np.percentile(diff_r2_A_boot, [2.5, 97.5])
    ci_diff_r2_B = np.percentile(diff_r2_B_boot, [2.5, 97.5])
    print("\nBootstrap results:")
    print(f'Difference in R^2 for Population A: {diff_r2_A:.4f} with 95% CI [{ci_diff_r2_A[0]:.4f}, {ci_diff_r2_A[1]:.4f}]')
    print(f'Difference in R^2 for Population B: {diff_r2_B:.4f} with 95% CI [{ci_diff_r2_B[0]:.4f}, {ci_diff_r2_B[1]:.4f}]')
    
    # Plotting results
    print("\nGenerating plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: PCA plot
    print("Creating PCA plot...")
    axs[0, 0].set_title('PCA of Genotype Data', fontsize=12)
    for pop in ['A', 'B']:
        idx = np.where(pop_labels == pop)
        axs[0, 0].scatter(genotypes_pca[idx, 0], genotypes_pca[idx, 1], 
                         label=f'Population {pop}', alpha=0.5)
    axs[0, 0].set_xlabel('PC1')
    axs[0, 0].set_ylabel('PC2')
    axs[0, 0].legend()
    
    # Subplot 2: R^2 in Population A
    print("Creating Population A R^2 plot...")
    methods = ['GWAS on A alone', 'GWAS on all data', 'GWAS no env effect', 'GWAS controlling PCs']
    r2_values_A = [r2_A, r2_A_all, r2_A_noenv, r2_A_pc]
    axs[0, 1].bar(range(len(methods)), r2_values_A, color='blue')
    axs[0, 1].set_title('R^2 in Population A', fontsize=12)
    axs[0, 1].set_ylabel('R^2')
    axs[0, 1].set_xticks(range(len(methods)))
    axs[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    
    # Subplot 3: R^2 in Population B
    print("Creating Population B R^2 plot...")
    r2_values_B = [np.nan, r2_B_all, r2_B_noenv, r2_B_pc]
    axs[1, 0].bar(range(len(methods)), r2_values_B, color='green')
    axs[1, 0].set_title('R^2 in Population B', fontsize=12)
    axs[1, 0].set_ylabel('R^2')
    axs[1, 0].set_xticks(range(len(methods)))
    axs[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    
    # Subplot 4: Difference in R^2 due to PC adjustment
    print("Creating PC adjustment effect plot...")
    diff_r2_values = [diff_r2_A, diff_r2_B]
    populations = ['Population A', 'Population B']
    bars = axs[1, 1].bar(populations, diff_r2_values, color=['blue', 'green'])
    axs[1, 1].set_title('Difference in R^2 due to PC Adjustment', fontsize=12)
    axs[1, 1].set_ylabel('Difference in R^2')
    
    # Add confidence interval error bars
    axs[1, 1].vlines(x=0, ymin=ci_diff_r2_A[0], ymax=ci_diff_r2_A[1], color='black')
    axs[1, 1].vlines(x=1, ymin=ci_diff_r2_B[0], ymax=ci_diff_r2_B[1], color='black')
    
    plt.tight_layout()
    print("\nSaving plot...")
    plt.savefig('gwas_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis pipeline complete!")
