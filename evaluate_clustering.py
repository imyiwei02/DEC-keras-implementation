#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate existing clustering results script
"""

import numpy as np
import pandas as pd
import os
from clustering_evaluation import calculate_clustering_metrics, plot_silhouette_analysis, compare_clustering_results


def load_clustering_results(results_dir, data_path):
    """
    Load clustering results and original data
    """
    # load clustering results
    results_file = os.path.join(results_dir, 'clustering_results.csv')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"failed to find clustering results file: {results_file}")
    
    results_df = pd.read_csv(results_file)
    labels = results_df['cluster'].values - 1  # convert to 0-based labels
    
    # load original data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"failed to find data file: {data_path}")
    
    # read original data
    df = pd.read_csv(data_path, index_col=0)
    X = df.T.values.astype(np.float32)  # transpose, so that the rows are samples
    
    # data standardization (keep consistent with training)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"loaded successfully:")
    print(f"  data shape: {X_scaled.shape}")
    print(f"  cluster number: {len(np.unique(labels))}")
    print(f"  sample number: {len(labels)}")
    
    return X_scaled, labels, results_df


def evaluate_single_result(results_dir, data_path):
    """Evaluate single clustering result"""
    print(f"\nEvaluating: {results_dir}")
    print("=" * 60)
    
    # load data
    X, labels, results_df = load_clustering_results(results_dir, data_path)
    
    # calculate evaluation metrics
    metrics = calculate_clustering_metrics(X, labels, results_dir)
    
    # generate silhouette analysis plot
    plot_silhouette_analysis(X, labels, results_dir)
    
    return metrics


def compare_3_vs_5_clusters(data_path):
    """Compare 3 clusters and 5 clusters results"""
    print("\n" + "=" * 60)
    print("Compare 3 clusters vs 5 clusters results")
    print("=" * 60)
    
    results_dirs = [
        'results/protein_3clusters',
        'results/protein_5clusters'
    ]
    
    all_metrics = {}
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            try:
                X, labels, _ = load_clustering_results(results_dir, data_path)
                metrics = calculate_clustering_metrics(X, labels)
                all_metrics[results_dir] = metrics
            except Exception as e:
                print(f"Error evaluating {results_dir}: {e}")
        else:
            print(f"Directory does not exist: {results_dir}")
    
    # create comparison table
    if len(all_metrics) >= 2:
        comparison_data = []
        for results_dir, metrics in all_metrics.items():
            n_clusters = 3 if '3clusters' in results_dir else 5
            comparison_data.append({
                'cluster number': n_clusters,
                'silhouette score': f"{metrics['silhouette_score']:.4f}",
                'calinski_harabasz_score': f"{metrics['calinski_harabasz_score']:.1f}",
                'davies_bouldin_score': f"{metrics['davies_bouldin_score']:.4f}",
                'cluster_balance': f"{metrics['cluster_balance']:.4f}",
                'wcss': f"{metrics['wcss']:.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nCluster results comparison:")
        print(comparison_df.to_string(index=False))
        
        # recommend best cluster number
        metrics_3 = all_metrics['results/protein_3clusters']
        metrics_5 = all_metrics['results/protein_5clusters']
        
        print("\nRecommend analysis:")
        
        # compare silhouette score
        if metrics_5['silhouette_score'] > metrics_3['silhouette_score']:
            print(f"  silhouette score: 5 clusters better ({metrics_5['silhouette_score']:.4f} vs {metrics_3['silhouette_score']:.4f})")
        else:
            print(f"  silhouette score: 3 clusters better ({metrics_3['silhouette_score']:.4f} vs {metrics_5['silhouette_score']:.4f})")
        
        # compare davies_bouldin_score (越小越好)
        if metrics_5['davies_bouldin_score'] < metrics_3['davies_bouldin_score']:
            print(f"  davies_bouldin_score: 5 clusters better ({metrics_5['davies_bouldin_score']:.4f} vs {metrics_3['davies_bouldin_score']:.4f})")
        else:
            print(f"  davies_bouldin_score: 3 clusters better ({metrics_3['davies_bouldin_score']:.4f} vs {metrics_5['davies_bouldin_score']:.4f})")
        
        # 比较平衡度
        if metrics_5['cluster_balance'] > metrics_3['cluster_balance']:
            print(f"  cluster_balance: 5 clusters better ({metrics_5['cluster_balance']:.4f} vs {metrics_3['cluster_balance']:.4f})")
        else:
            print(f"  cluster_balance: 3 clusters better ({metrics_3['cluster_balance']:.4f} vs {metrics_5['cluster_balance']:.4f})")


def main():
    """Main function"""
    data_path = r"C:\Users\yiwei\Desktop\DEC\DEC-keras\protein.csv"
    
    print("🔬 Protein data clustering results evaluation")
    print("=" * 60)
    
    # evaluate 5 clusters results
    if os.path.exists('results/protein_5clusters'):
        evaluate_single_result('results/protein_5clusters', data_path)
    
    # evaluate 3 clusters results
    if os.path.exists('results/protein_3clusters'):
        evaluate_single_result('results/protein_3clusters', data_path)
    
    # compare two clustering results
    compare_3_vs_5_clusters(data_path)
    
    print("\nEvaluation completed!")
    print("\nGenerated files:")
    for results_dir in ['results/protein_3clusters', 'results/protein_5clusters']:
        if os.path.exists(results_dir):
            eval_file = os.path.join(results_dir, 'clustering_evaluation_report.txt')
            silhouette_file = os.path.join(results_dir, 'silhouette_analysis.png')
            if os.path.exists(eval_file):
                print(f"  {eval_file}")
            if os.path.exists(silhouette_file):
                print(f"  {silhouette_file}")


if __name__ == "__main__":
    main()
