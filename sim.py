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
import seaborn as sns
from collections import defaultdict

print("Starting simulation pipeline...")

# Set random seed for reproducibility
np.random.seed(2024)

def calculate_standardized_prs(y_true, y_pred):
   """Calculate R² between standardized true and predicted values"""
   # Standardize inputs
   y_true = (y_true - y_true.mean()) / y_true.std()
   y_pred = (y_pred - y_pred.mean()) / y_pred.std()
   return r2_score(y_true, y_pred)

def generate_populations(n_snps=1000, n_pop_a=5000, n_pop_b=1000, diff_freq_prop=0.5, causal_prop=0.2):
    """Generate genotype matrices for two populations, ensuring frequency differences 
    never occur in causal SNPs"""
    # First determine which SNPs will be causal (20%)
    n_causal = int(n_snps * causal_prop)
    causal_snps = np.random.choice(np.arange(n_snps), n_causal, replace=False)
    
    # Get the non-causal SNPs
    non_causal_snps = np.setdiff1d(np.arange(n_snps), causal_snps)
    
    # Base frequencies for all SNPs
    p_A = np.random.uniform(0.1, 0.9, n_snps)
    p_B = p_A.copy()
    
    # Adjust frequencies ONLY for non-causal SNPs in population B
    n_diff = int(n_snps * diff_freq_prop)
    diff_snps = np.random.choice(non_causal_snps, n_diff, replace=False)  # Only choose from non-causal SNPs
    delta_p = np.random.uniform(-0.2, 0.2, len(diff_snps))
    p_B[diff_snps] += delta_p
    p_B = np.clip(p_B, 0.01, 0.99)
    
    # Generate genotypes
    genotypes_A = np.array([np.random.binomial(2, p_A) for _ in range(n_pop_a)])
    genotypes_B = np.array([np.random.binomial(2, p_B) for _ in range(n_pop_b)])
    
    return genotypes_A, genotypes_B, causal_snps  # Now returns causal_snps

def generate_phenotypes(genotypes_A, genotypes_B, causal_snps, heritability=0.5):
    """Generate phenotypes with genetic and environmental components using pre-defined causal SNPs"""
    n_snps = genotypes_A.shape[1]
    
    # Use the pre-determined causal SNPs
    beta = np.zeros(n_snps)
    beta[causal_snps] = np.random.normal(0, 1, len(causal_snps))
    
    # Rest remains the same
    G_A = genotypes_A.dot(beta)
    G_B = genotypes_B.dot(beta)
    G = np.concatenate([G_A, G_B])
    var_G = np.var(G)
    
    # Generate environmental effects
    n_A = len(genotypes_A)
    n_B = len(genotypes_B)
    E_A = np.random.normal(0, np.sqrt(var_G), n_A)
    E_B = np.random.normal(0, np.sqrt(var_G), n_B)
    
    # Add population B specific effect
    mean_env_effect = -1.0 * np.sqrt(var_G)
    sd_env_effect = 0.5 * np.sqrt(var_G)
    E_env = np.random.normal(mean_env_effect, sd_env_effect, n_B)
    E_B += E_env
    
    # Combine and rescale environmental effects
    E = np.concatenate([E_A, E_B])
    E = E * np.sqrt(var_G/np.var(E))
    
    # Total phenotype
    Y = G + E
    
    return Y[:n_A], Y[n_A:], beta

def perform_gwas_covariates(genotypes, phenotype, covariates=None):
   """Perform GWAS with optional covariate control"""
   print("Starting GWAS...")
   n_snps = genotypes.shape[1]
   
   # Standardize inputs
   phenotype_std = (phenotype - phenotype.mean()) / phenotype.std()
   genotypes_std = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
   
   if covariates is None:
       print("Running GWAS without covariates")
       betas = np.sum(genotypes_std * phenotype_std[:, np.newaxis], axis=0)
   else:
       print("Running GWAS with covariates")
       reg_cov = LinearRegression().fit(covariates, phenotype_std)
       phenotype_resid = phenotype_std - reg_cov.predict(covariates)
       
       betas = np.zeros(n_snps)
       for i in tqdm(range(n_snps), desc="Processing SNPs"):
           reg_snp = LinearRegression().fit(covariates, genotypes_std[:, i])
           snp_resid = genotypes_std[:, i] - reg_snp.predict(covariates)
           betas[i] = np.dot(snp_resid, phenotype_resid)
   
   print("GWAS complete")
   return betas

def run_all_analyses(genotypes_A, genotypes_B, Y_A, Y_B, PCs):
   """Run all GWAS analyses and return metrics"""
   genotypes = np.vstack([genotypes_A, genotypes_B])
   Y = np.concatenate([Y_A, Y_B])
   n_A = len(genotypes_A)
   
   # Task 2: GWAS on population A alone
   print("\nTask 2: GWAS on Population A alone...")
   betas_A = perform_gwas_covariates(genotypes_A, Y_A)
   PRS_A = genotypes_A.dot(betas_A)
   r2_A = calculate_standardized_prs(Y_A, PRS_A)
   print(f'R^2 in Population A (GWAS on A alone): {r2_A:.4f}')
   
   # Task 3: GWAS on all data
   print("\nTask 3: GWAS on all data...")
   betas_all = perform_gwas_covariates(genotypes, Y)
   PRS_all = genotypes.dot(betas_all)
   r2_A_all = calculate_standardized_prs(Y_A, PRS_all[:n_A])
   r2_B_all = calculate_standardized_prs(Y_B, PRS_all[n_A:])
   print(f'R^2 in Population A (GWAS on all data): {r2_A_all:.4f}')
   print(f'R^2 in Population B (GWAS on all data): {r2_B_all:.4f}')
   
   # Task 5: GWAS with PC control
   print("\nTask 5: GWAS with PC control...")
   betas_pc = perform_gwas_covariates(genotypes, Y, PCs)
   PRS_pc = genotypes.dot(betas_pc)
   r2_A_pc = calculate_standardized_prs(Y_A, PRS_pc[:n_A])
   r2_B_pc = calculate_standardized_prs(Y_B, PRS_pc[n_A:])
   print(f'R^2 in Population A (PC controlled): {r2_A_pc:.4f}')
   print(f'R^2 in Population B (PC controlled): {r2_B_pc:.4f}')
   
   return {
       'r2_A': r2_A,
       'r2_A_all': r2_A_all,
       'r2_B_all': r2_B_all,
       'r2_A_pc': r2_A_pc,
       'r2_B_pc': r2_B_pc
   }

def bootstrap_iteration(args):
   """Perform one bootstrap iteration"""
   idx, genotypes_A, genotypes_B, Y_A, Y_B = args
   np.random.seed(idx)
   
   # Sample with replacement
   idx_a = np.random.choice(len(genotypes_A), len(genotypes_A), replace=True)
   idx_b = np.random.choice(len(genotypes_B), len(genotypes_B), replace=True)
   
   # Create bootstrap samples
   boot_geno_A = genotypes_A[idx_a]
   boot_geno_B = genotypes_B[idx_b]
   boot_Y_A = Y_A[idx_a]
   boot_Y_B = Y_B[idx_b]
   
   # Compute PCs
   boot_geno = np.vstack([boot_geno_A, boot_geno_B])
   boot_geno_centered = boot_geno - boot_geno.mean(axis=0)
   svd = TruncatedSVD(n_components=5, random_state=idx)
   boot_PCs = svd.fit_transform(boot_geno_centered)
   
   # Run analyses
   metrics = run_all_analyses(boot_geno_A, boot_geno_B, boot_Y_A, boot_Y_B, boot_PCs)
   return metrics

def calculate_confidence_intervals(results):
   """Calculate confidence intervals from bootstrap results"""
   ci_results = {}
   metrics = defaultdict(list)
   
   for r in results:
       for k, v in r.items():
           metrics[k].append(v)
           
   for metric, values in metrics.items():
       ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
       mean_val = np.mean(values)
       ci_results[metric] = {
           'mean': mean_val,
           'ci_lower': ci_lower,
           'ci_upper': ci_upper
       }
   return ci_results

def plot_results(metrics, ci_results, PCs, n_individuals_A):
   """Plot results including PCA and confidence intervals"""
   plt.figure(figsize=(15, 10))
   
   # PCA plot
   plt.subplot(231)
   plt.scatter(PCs[:n_individuals_A, 0], PCs[:n_individuals_A, 1],
              alpha=0.5, label='Population A')
   plt.scatter(PCs[n_individuals_A:, 0], PCs[n_individuals_A:, 1],
              alpha=0.5, label='Population B')
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   plt.title('Population Structure')
   plt.legend()
   
   # R² plot
   plt.subplot(232)
   x = np.arange(len(metrics))
   means = [metrics[m] for m in metrics.keys()]
   errors = [(metrics[m] - ci_results[m]['ci_lower'],
             ci_results[m]['ci_upper'] - metrics[m])
            for m in metrics.keys()]
   
   plt.errorbar(x, means, yerr=np.array(errors).T, fmt='o', capsize=5)
   plt.xticks(x, list(metrics.keys()), rotation=45)
   plt.title('R² Values with 95% CIs')
   plt.ylabel('R²')
   
   plt.tight_layout()
   plt.savefig('gwas_results.png', dpi=300, bbox_inches='tight')
   plt.show()

def main():
   """Main function to run simulation and analysis"""
   # Parameters
   n_snps = 1000
   n_individuals_A = 5000
   n_individuals_B = 1000
   
   print("\nGenerating populations...")
   genotypes_A, genotypes_B = generate_populations(n_snps, n_individuals_A, n_individuals_B)
   print("Generating phenotypes...")
   Y_A, Y_B, true_effects = generate_phenotypes(genotypes_A, genotypes_B)
   
   # Compute PCs
   print("\nComputing population structure...")
   genotypes = np.vstack([genotypes_A, genotypes_B])
   genotypes_centered = genotypes - genotypes.mean(axis=0)
   svd = TruncatedSVD(n_components=5, random_state=42)
   PCs = svd.fit_transform(genotypes_centered)
   print(f"Variance explained by PCs: {svd.explained_variance_ratio_}")
   
   # Run initial analyses
   print("\nRunning main analyses...")
   metrics = run_all_analyses(genotypes_A, genotypes_B, Y_A, Y_B, PCs)
   
   # Bootstrap
   print("\nPerforming bootstrap analysis...")
   n_bootstraps = 100
   pool = mp.Pool(processes=mp.cpu_count())
   bootstrap_args = [(i, genotypes_A, genotypes_B, Y_A, Y_B) 
                    for i in range(n_bootstraps)]
   
   try:
       bootstrap_results = list(tqdm(pool.imap(bootstrap_iteration, bootstrap_args),
                                   total=n_bootstraps,
                                   desc="Bootstrap progress"))
   finally:
       pool.close()
       pool.join()
   
   # Calculate confidence intervals
   print("\nCalculating confidence intervals...")
   ci_results = calculate_confidence_intervals(bootstrap_results)
   
   # Plot results
   print("\nPlotting results...")
   plot_results(metrics, ci_results, PCs, n_individuals_A)

if __name__ == "__main__":
   main()
