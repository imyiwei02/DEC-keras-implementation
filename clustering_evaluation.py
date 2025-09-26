#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering result evaluation tool

Provide multiple unsupervised clustering evaluation metrics, used to evaluate clustering quality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def calculate_clustering_metrics(X, labels, save_dir=None):
    """
    Calculate multiple clustering evaluation metrics
    
    Args:
        X (ndarray): original data or feature data
        labels (ndarray): clustering labels
        save_dir (str): results save directory
    
    Returns:
        dict: dictionary containing various evaluation metrics
    """
    print("Calculating clustering evaluation metrics...")
    
    metrics = {}
    
    # 1. silhouette score (Silhouette Score)
    # range: [-1, 1], the closer to 1 the better
    silhouette_avg = silhouette_score(X, labels)
    metrics['silhouette_score'] = silhouette_avg
    print(f"silhouette score (Silhouette Score): {silhouette_avg:.4f}")
    
    # 2. Calinski-Harabasz index (variance ratio criterion)
    # the larger the better, indicating the ratio of cluster separation and cluster compactness
    ch_score = calinski_harabasz_score(X, labels)
    metrics['calinski_harabasz_score'] = ch_score
    print(f"Calinski-Harabasz index: {ch_score:.2f}")
    
    # 3. Davies-Bouldin index
    # range: [0, +∞), the smaller the better
    db_score = davies_bouldin_score(X, labels)
    metrics['davies_bouldin_score'] = db_score
    print(f"Davies-Bouldin index: {db_score:.4f}")
    
    # 4. Within-cluster Sum of Squares (WCSS)
    wcss = calculate_wcss(X, labels)
    metrics['wcss'] = wcss
    print(f"Within-cluster Sum of Squares (WCSS): {wcss:.2f}")
    
    # 5. cluster separation
    separation = calculate_cluster_separation(X, labels)
    metrics['cluster_separation'] = separation
    print(f"cluster separation: {separation:.4f}")
    
    # 6. cluster balance
    balance = calculate_cluster_balance(labels)
    metrics['cluster_balance'] = balance
    print(f"cluster balance: {balance:.4f}")
    
    if save_dir:
        # save evaluation report
        save_evaluation_report(metrics, labels, save_dir)
    
    return metrics


def calculate_wcss(X, labels):
    """Calculate Within-cluster Sum of Squares (WCSS)"""
    wcss = 0
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            cluster_center = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - cluster_center) ** 2)
    
    return wcss


def calculate_cluster_separation(X, labels):
    """Calculate cluster separation (average distance between cluster centers)"""
    unique_labels = np.unique(labels)
    centers = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            centers.append(center)
    
    centers = np.array(centers)
    if len(centers) < 2:
        return 0.0
    
    # calculate all cluster center distances
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    
    return np.mean(distances)


def calculate_cluster_balance(labels):
    """
    Calculate cluster balance
    Return value close to 1 indicates cluster size balance, close to 0 indicates imbalance
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if len(unique_labels) <= 1:
        return 1.0
    
    # calculate balance: use the reciprocal of the coefficient of variation
    mean_size = np.mean(counts)
    std_size = np.std(counts)
    
    if mean_size == 0:
        return 0.0
    
    cv = std_size / mean_size  # coefficient of variation
    balance = 1 / (1 + cv)  # convert to value between 0 and 1
    
    return balance


def plot_silhouette_analysis(X, labels, save_dir):
    """Plot silhouette analysis plot"""
    print("Generating silhouette analysis plot...")
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_lower = 10
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        cluster_silhouette_values = sample_silhouette_values[labels == label]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label + 1))
        y_lower = y_upper + 10
    
    ax.set_xlabel('silhouette score value')
    ax.set_ylabel('cluster labels')
    ax.set_title(f'silhouette analysis for each cluster\nmean silhouette score: {silhouette_avg:.4f}')
    
    # add average line
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'mean: {silhouette_avg:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'silhouette_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"silhouette analysis plot saved to: {os.path.join(save_dir, 'silhouette_analysis.png')}")


def compare_clustering_results(results_dirs, labels_list=None):
    """
    Compare multiple clustering results
    
    Args:
        results_dirs (list): results directory list
        labels_list (list): cluster number list, used to identify
    """
    print("Comparing multiple clustering results...")
    
    if labels_list is None:
        labels_list = [f"K={i+2}" for i in range(len(results_dirs))]
    
    comparison_data = []
    
    for i, results_dir in enumerate(results_dirs):
        # read clustering results
        results_file = os.path.join(results_dir, 'clustering_results.csv')
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            
            # read statistical information (if any)
            stats_file = os.path.join(results_dir, 'clustering_statistics.txt')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # extract cluster number
                    import re
                    match = re.search(r'cluster number: (\d+)', content)
                    n_clusters = int(match.group(1)) if match else len(df['cluster'].unique())
            else:
                n_clusters = len(df['cluster'].unique())
            
            comparison_data.append({
                'method': labels_list[i],
                'n_clusters': n_clusters,
                'n_samples': len(df),
                'results_dir': results_dir
            })
    
    # create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print("\nclustering results comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def save_evaluation_report(metrics, labels, save_dir):
    """save evaluation report"""
    report_path = os.path.join(save_dir, 'clustering_evaluation_report.txt')
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("clustering quality evaluation report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("clustering basic information:\n")
        f.write(f"  total sample number: {len(labels)}\n")
        f.write(f"  cluster number: {len(unique_labels)}\n\n")
        
        f.write("clustering quality metrics:\n")
        f.write(f"  silhouette score: {metrics['silhouette_score']:.4f}\n")
        f.write(f"    explanation: range [-1,1], the closer to 1 the better\n")
        f.write(f"    {get_silhouette_interpretation(metrics['silhouette_score'])}\n\n")
        
        f.write(f"  Calinski-Harabasz index: {metrics['calinski_harabasz_score']:.2f}\n")
        f.write(f"    explanation: the larger the better, indicating the ratio of cluster separation and cluster compactness\n\n")
        
        f.write(f"  Davies-Bouldin index: {metrics['davies_bouldin_score']:.4f}\n")
        f.write(f"    explanation: the smaller the better, indicating the ratio of cluster separation and cluster compactness\n\n")
        
        f.write(f"  Within-cluster Sum of Squares (WCSS): {metrics['wcss']:.2f}\n")
        f.write(f"    explanation: the smaller the better, indicating the ratio of cluster separation and cluster compactness\n\n")
        
        f.write(f"  cluster separation: {metrics['cluster_separation']:.4f}\n")
        f.write(f"    explanation: the larger the better, indicating the ratio of cluster separation and cluster compactness\n\n")
        
        f.write(f"  cluster balance: {metrics['cluster_balance']:.4f}\n")
        f.write(f"    explanation: range [0,1], the closer to 1 the better\n\n")
        
        f.write("cluster size distribution:\n")
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            percentage = count / len(labels) * 100
            f.write(f"  cluster {label + 1}: {count} samples ({percentage:.1f}%)\n")
        
        f.write(f"\noverall evaluation:\n")
        f.write(f"  {get_overall_assessment(metrics)}\n")
    
    print(f"evaluation report saved to: {report_path}")


def get_silhouette_interpretation(score):
    """explain silhouette score"""
    if score >= 0.7:
        return "cluster structure is strong"
    elif score >= 0.5:
        return "cluster structure is reasonable"
    elif score >= 0.25:
        return "cluster structure is weak, possibly overlapping"
    else:
        return "cluster structure is poor or artificial structure"


def get_overall_assessment(metrics):
    """overall evaluation of clustering quality"""
    silhouette = metrics['silhouette_score']
    db_score = metrics['davies_bouldin_score']
    balance = metrics['cluster_balance']
    
    score = 0
    if silhouette >= 0.5:
        score += 1
    if db_score <= 1.0:
        score += 1
    if balance >= 0.8:
        score += 1
    
    if score >= 3:
        return "cluster quality is excellent"
    elif score >= 2:
        return "cluster quality is good"
    elif score >= 1:
        return "cluster quality is average"
    else:
        return "suggest adjusting clustering parameters or selecting different cluster numbers"


def main():
    """main function - example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='clustering result evaluation')
    parser.add_argument('--results_dir', default='results/protein_5clusters',
                       help='clustering result directory')
    parser.add_argument('--data_path', default='protein.csv',
                       help='original data path (for calculating metrics)')
    
    args = parser.parse_args()
    
    # here we need to load the original data and clustering results
    print("clustering evaluation tool")
    print("please call the related functions in other scripts for evaluation")


if __name__ == "__main__":
    main()
