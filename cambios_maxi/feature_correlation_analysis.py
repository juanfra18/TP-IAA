#!/usr/bin/env python3
"""
Feature Correlation Analysis Script

This script analyzes feature correlations in the preprocessed dataset and identifies
highly correlated feature pairs that may cause multicollinearity issues.
It also proposes feature subsets to reduce redundancy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_correlations(df: pd.DataFrame, threshold: float = 0.8) -> tuple:
    """
    Analyze feature correlations and identify highly correlated pairs.
    
    Args:
        df: DataFrame with features (no target)
        threshold: Correlation threshold for flagging pairs
        
    Returns:
        Tuple of (correlation_matrix, high_corr_pairs, moderate_corr_pairs)
    """
    corr_matrix = df.corr()
    
    high_corr_pairs = []
    moderate_corr_pairs = []
    features = corr_matrix.columns
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append((features[i], features[j], corr_val))
            elif 0.7 <= abs(corr_val) < threshold:
                moderate_corr_pairs.append((features[i], features[j], corr_val))
    
    # Sort by absolute correlation value (descending)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    moderate_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return corr_matrix, high_corr_pairs, moderate_corr_pairs


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """Create and save a correlation heatmap."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved correlation heatmap to {output_path}")


def plot_high_correlations(high_corr_pairs: list, output_path: str):
    """Create a bar plot of highly correlated feature pairs."""
    if not high_corr_pairs:
        print("No highly correlated pairs found.")
        return
    
    labels = [f"{p[0][:15]}\n↔\n{p[1][:15]}" for p in high_corr_pairs]
    values = [p[2] for p in high_corr_pairs]
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]
    plt.barh(range(len(values)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.title('Highly Correlated Feature Pairs (|r| ≥ 0.8)', fontsize=14, pad=15)
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Threshold (0.8)')
    plt.axvline(x=-0.8, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved high correlation plot to {output_path}")


def propose_feature_subsets(high_corr_pairs: list, all_features: list) -> dict:
    """
    Propose different feature subsets to handle multicollinearity.
    
    Strategy:
    1. Minimal: Keep only one representative from each correlated group
    2. Moderate: Keep 2 features from each correlated group
    3. VIF-based: Use variance inflation factor approach
    """
    # Identify employment-related features (the main correlated group)
    employment_features = {
        'employment_pre_lockdown', 'work_type', 'less_work', 'lost_work'
    }
    
    # Find which features are actually in correlation pairs
    correlated_features = set()
    for feat1, feat2, _ in high_corr_pairs:
        correlated_features.add(feat1)
        correlated_features.add(feat2)
    
    # Subset 1: Minimal - Keep only primary feature
    subset1_drop = ['work_type', 'less_work', 'lost_work']
    subset1 = [f for f in all_features if f not in subset1_drop]
    
    # Subset 2: Moderate - Keep 2 employment features
    subset2_drop = ['less_work', 'lost_work']
    subset2 = [f for f in all_features if f not in subset2_drop]
    
    # Subset 3: Drop only the most correlated (lost_work)
    subset3_drop = ['lost_work']
    subset3 = [f for f in all_features if f not in subset3_drop]
    
    return {
        'subset1_minimal': {
            'features': subset1,
            'dropped': subset1_drop,
            'description': 'Minimal: Keep only employment_pre_lockdown (drops 3 features)'
        },
        'subset2_moderate': {
            'features': subset2,
            'dropped': subset2_drop,
            'description': 'Moderate: Keep employment_pre_lockdown + work_type (drops 2 features)'
        },
        'subset3_conservative': {
            'features': subset3,
            'dropped': subset3_drop,
            'description': 'Conservative: Drop only lost_work (drops 1 feature)'
        }
    }


def main():
    # Load data
    data_path = "datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='latin1')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1} (excluding target)")
    
    # Separate features from target
    X = df.drop(columns=['target'])
    feature_names = X.columns.tolist()
    
    # Analyze correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    corr_matrix, high_corr_pairs, moderate_corr_pairs = analyze_correlations(X, threshold=0.8)
    
    # Display results
    print(f"\n📊 Found {len(high_corr_pairs)} highly correlated pairs (|r| ≥ 0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  • {feat1:30s} ↔ {feat2:30s} : {corr:6.3f}")
    
    print(f"\n📊 Found {len(moderate_corr_pairs)} moderately correlated pairs (0.7 ≤ |r| < 0.8):")
    for feat1, feat2, corr in moderate_corr_pairs[:10]:  # Show first 10
        print(f"  • {feat1:30s} ↔ {feat2:30s} : {corr:6.3f}")
    if len(moderate_corr_pairs) > 10:
        print(f"  ... and {len(moderate_corr_pairs) - 10} more")
    
    # Create output directory
    os.makedirs("results/feature_analysis", exist_ok=True)
    
    # Save correlation matrix
    corr_matrix.to_csv("results/feature_analysis/correlation_matrix.csv")
    print(f"\n✓ Saved correlation matrix to results/feature_analysis/correlation_matrix.csv")
    
    # Save correlation pairs
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
    high_corr_df.to_csv("results/feature_analysis/high_correlations.csv", index=False)
    print(f"✓ Saved high correlation pairs to results/feature_analysis/high_correlations.csv")
    
    # Create visualizations
    plot_correlation_heatmap(corr_matrix, "results/feature_analysis/correlation_heatmap.png")
    plot_high_correlations(high_corr_pairs, "results/feature_analysis/high_correlation_pairs.png")
    
    # Propose feature subsets
    print("\n" + "="*60)
    print("PROPOSED FEATURE SUBSETS")
    print("="*60)
    
    subsets = propose_feature_subsets(high_corr_pairs, feature_names)
    
    subset_info = []
    for subset_name, subset_data in subsets.items():
        print(f"\n{subset_name.upper()}:")
        print(f"  Description: {subset_data['description']}")
        print(f"  Dropped features: {', '.join(subset_data['dropped'])}")
        print(f"  Remaining features: {len(subset_data['features'])}")
        
        subset_info.append({
            'Subset': subset_name,
            'Description': subset_data['description'],
            'Dropped_Features': ', '.join(subset_data['dropped']),
            'Num_Features': len(subset_data['features']),
            'Features': ', '.join(subset_data['features'])
        })
    
    # Save subset recommendations
    subset_df = pd.DataFrame(subset_info)
    subset_df.to_csv("results/feature_analysis/feature_subsets.csv", index=False)
    print(f"\n✓ Saved feature subset recommendations to results/feature_analysis/feature_subsets.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All results saved to results/feature_analysis/")


if __name__ == "__main__":
    main()
