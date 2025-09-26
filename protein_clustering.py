#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Protein Clustering Analysis (DEC-keras project adaptation)

Based on the DEC algorithm for clustering analysis of protein expression data.
Data format: CSV file, the first row is the person ID, the first column is the protein ID, the numerical matrix is the protein expression量。

Usage:
    python protein_clustering.py --data_path protein.csv --n_clusters 5 --save_dir results/protein

Author: Yiwei Li
Date: 2025
"""

import os
import argparse
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

# Import modules from the project
from DEC import DEC, autoencoder
import metrics


def load_protein_data(data_path='protein.csv', scaling_method='standard'):
    """
    Load protein data and preprocess it
    
    Args:
        data_path (str): Protein data file path
        scaling_method (str): Data scaling method ('standard', 'minmax', 'none')
    
    Returns:
        tuple: (x, person_ids, protein_ids)
            - x: Preprocessed feature matrix, shape (n_persons, n_proteins)
            - person_ids: Person ID list
            - protein_ids: Protein ID list
    """
    print(f"Loading protein data: {data_path}")
    
    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(data_path, index_col=0)
        print(f"Data file read successfully, original shape: {df.shape}")
        
        # Get person ID and protein ID
        person_ids = list(df.columns)
        protein_ids = list(df.index)
        
        # Transpose matrix, so that the rows are persons, and the columns are proteins
        data_matrix = df.T.values.astype(np.float32)
        
        print(f"Transposed data shape: {data_matrix.shape}")
        print(f"Person number: {len(person_ids)}")
        print(f"Protein number: {len(protein_ids)}")
        
        # Check for missing values in the data
        nan_count = np.isnan(data_matrix).sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} missing values, will be filled with mean")
            # Fill missing values with the mean of each protein
            col_means = np.nanmean(data_matrix, axis=0)
            for i in range(data_matrix.shape[1]):
                data_matrix[np.isnan(data_matrix[:, i]), i] = col_means[i]
        
        # Check for infinite values in the data
        inf_count = np.isinf(data_matrix).sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values, will be processed")
            data_matrix = np.where(np.isinf(data_matrix), np.nan, data_matrix)
            # Fill again
            col_means = np.nanmean(data_matrix, axis=0)
            for i in range(data_matrix.shape[1]):
                data_matrix[np.isnan(data_matrix[:, i]), i] = col_means[i]
        
        # Data scaling
        if scaling_method == 'standard':
            print("Using StandardScaler for data scaling")
            scaler = StandardScaler()
            x = scaler.fit_transform(data_matrix)
        elif scaling_method == 'minmax':
            print("Using MinMaxScaler for data scaling")
            scaler = MinMaxScaler()
            x = scaler.fit_transform(data_matrix)
        elif scaling_method == 'none':
            print("No data scaling")
            x = data_matrix
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
        # Ensure data type is float32 to save memory
        x = x.astype(np.float32)
        
        print(f"Final data shape: {x.shape}")
        print(f"Data range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Data mean: {x.mean():.4f}, standard deviation: {x.std():.4f}")
        
        return x, person_ids, protein_ids
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def estimate_optimal_clusters(x, max_clusters=10, random_state=42):
    """
    Estimate the optimal number of clusters using the elbow method and silhouette score
    
    Args:
        x (ndarray): Input data
        max_clusters (int): Maximum number of clusters
        random_state (int): Random seed
    
    Returns:
        dict: Contains evaluation metrics for different number of clusters
    """
    print("Estimating the optimal number of clusters...")
    
    from sklearn.metrics import silhouette_score
    
    cluster_range = range(2, min(max_clusters + 1, x.shape[0] // 2))
    inertias = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        print(f"  Testing {n_clusters} clusters...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(x)
        
        # Calculate inertia
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(x, labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"    Inertia: {kmeans.inertia_:.2f}, Silhouette score: {silhouette_avg:.4f}")
    
    # Find the number of clusters with the highest silhouette score
    best_idx = np.argmax(silhouette_scores)
    recommended_clusters = list(cluster_range)[best_idx]
    
    print(f"\nRecommended number of clusters: {recommended_clusters} (Silhouette score: {silhouette_scores[best_idx]:.4f})")
    
    return {
        'cluster_range': list(cluster_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'recommended_clusters': recommended_clusters
    }


def plot_cluster_analysis(cluster_analysis, save_dir):
    """
    Plot cluster analysis chart
    
    Args:
        cluster_analysis (dict): Cluster analysis results
        save_dir (str): Save directory
    """
    print("Generating cluster analysis chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow method chart
    ax1.plot(cluster_analysis['cluster_range'], cluster_analysis['inertias'], 'bo-')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow method - Optimal number of clusters')
    ax1.grid(True)
    
    # Silhouette score chart
    ax2.plot(cluster_analysis['cluster_range'], cluster_analysis['silhouette_scores'], 'ro-')
    ax2.axvline(x=cluster_analysis['recommended_clusters'], color='g', linestyle='--', 
                label=f'Recommended: {cluster_analysis["recommended_clusters"]}')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette score')
    ax2.set_title('Silhouette score - Cluster quality evaluation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster analysis chart saved to: {os.path.join(save_dir, 'cluster_analysis.png')}")


def visualize_results(x, y_pred, person_ids, save_dir, method='tsne'):
    """
    Visualize cluster results
    
    Args:
        x (ndarray): Original data
        y_pred (ndarray): Cluster prediction results
        person_ids (list): Person ID list
        save_dir (str): Save directory
        method (str): Dimensionality reduction method ('tsne', 'pca')
    """
    print(f"Using {method.upper()} for dimensionality reduction visualization...")
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, x.shape[0]-1))
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    # Dimensionality reduction
    x_2d = reducer.fit_transform(x)
    
    # Plot cluster results
    plt.figure(figsize=(12, 8))
    
    # Get cluster number and color
    n_clusters = len(np.unique(y_pred))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_mask = y_pred == i
        plt.scatter(x_2d[cluster_mask, 0], x_2d[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {i+1} (n={cluster_mask.sum()})', 
                   alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.title(f'Protein data clustering results - {method.upper()} visualization\nTotal samples: {len(person_ids)}, Clusters: {n_clusters}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'clustering_visualization_{method}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster visualization chart saved to: {os.path.join(save_dir, f'clustering_visualization_{method}.png')}")


def save_clustering_results(y_pred, person_ids, save_dir):
    """
    Save cluster results to file
    
    Args:
        y_pred (ndarray): Cluster prediction results
        person_ids (list): Person ID list
        save_dir (str): Save directory
    """
    print("Saving cluster results...")
    
    # Create result DataFrame
    results_df = pd.DataFrame({
        'person_id': person_ids,
        'cluster': y_pred + 1  # Cluster labels start from 1
    })
    
    # Save to CSV file
    results_path = os.path.join(save_dir, 'clustering_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Cluster results saved to: {results_path}")
    
    # Count the number of people in each cluster
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    
    print("\nCluster statistics:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} people")
    
    # save statistics information
    stats_path = os.path.join(save_dir, 'clustering_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Protein data clustering statistics report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total number of people: {len(person_ids)}\n")
        f.write(f"Number of clusters: {len(cluster_counts)}\n\n")
        f.write("Number of people in each cluster:\n")
        for cluster_id, count in cluster_counts.items():
            percentage = count / len(person_ids) * 100
            f.write(f"  Cluster {cluster_id}: {count} people ({percentage:.1f}%)\n")
    
    print(f"Statistics information saved to: {stats_path}")
    
    return results_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Protein data DEC clustering analysis',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Data related parameters
    parser.add_argument('--data_path', default='protein.csv', 
                       help='Protein data CSV file path')
    parser.add_argument('--scaling_method', default='standard', 
                       choices=['standard', 'minmax', 'none'],
                       help='Data scaling method')
    
    # Cluster related parameters
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters, if not specified will be automatically estimated')
    parser.add_argument('--estimate_clusters', action='store_true',
                       help='Whether to estimate the number of clusters')
    parser.add_argument('--max_clusters', type=int, default=10,
                       help='Maximum number of clusters to estimate')
    
    # DEC model parameters
    parser.add_argument('--dims', nargs='+', type=int, default=None,
                       help='Autoencoder dimension configuration, if not specified will be automatically set')
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                       help='Pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--maxiter', type=int, default=20000,
                       help='Maximum number of iterations for DEC training')
    parser.add_argument('--update_interval', type=int, default=140,
                       help='Target distribution update interval')
    parser.add_argument('--tol', type=float, default=0.001,
                       help='Convergence tolerance')
    
    # Output related parameters
    parser.add_argument('--save_dir', default='results/protein',
                       help='Result save directory')
    parser.add_argument('--visualization', action='store_true',
                       help='Whether to generate visualization charts')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.random_state)
    
    # create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f"Results will be saved to: {args.save_dir}")
    
    try:
        # 1. Load and preprocess data
        print("\n" + "="*60)
        print("Step 1: Load and preprocess data")
        print("="*60)
        
        x, person_ids, protein_ids = load_protein_data(args.data_path, args.scaling_method)
        
        # 2. Cluster number estimation (if needed)
        if args.estimate_clusters or args.n_clusters is None:
            print("\n" + "="*60)
            print("Step 2: Cluster number estimation")
            print("="*60)
            
            cluster_analysis = estimate_optimal_clusters(x, args.max_clusters, args.random_state)
            
            if args.visualization:
                plot_cluster_analysis(cluster_analysis, args.save_dir)
            
            if args.n_clusters is None:
                args.n_clusters = cluster_analysis['recommended_clusters']
                print(f"Automatically set cluster number to: {args.n_clusters}")
        
        # 3. Set network structure
        if args.dims is None:
            n_features = x.shape[1]
            # Automatically set network structure: gradually reduce dimensions
            if n_features > 2000:
                args.dims = [n_features, 1000, 500, 200, 10]
            elif n_features > 1000:
                args.dims = [n_features, 500, 200, 50, 10]
            else:
                args.dims = [n_features, 500, 200, 10]
        
        print(f"\nNetwork structure: {args.dims}")
        print(f"Cluster number: {args.n_clusters}")
        
        # 4. DEC model training
        print("\n" + "="*60)
        print("Step 3: DEC model training")
        print("="*60)
        
        # create DEC model
        dec = DEC(dims=args.dims, n_clusters=args.n_clusters)
        
        # pretrain autoencoder
        print("\nStarting pretraining autoencoder...")
        start_time = time()
        dec.pretrain(x=x, epochs=args.pretrain_epochs, batch_size=args.batch_size, 
                    save_dir=args.save_dir)
        pretrain_time = time() - start_time
        print(f"Pretraining completed, time: {pretrain_time:.2f} seconds")
        
        # DEC clustering training
        print("\nStarting DEC clustering training...")
        dec.compile(optimizer='sgd', loss='kld')
        start_time = time()
        y_pred = dec.fit(x, maxiter=args.maxiter, batch_size=args.batch_size,
                        update_interval=args.update_interval, tol=args.tol,
                        save_dir=args.save_dir)
        clustering_time = time() - start_time
        print(f"Clustering training completed, time: {clustering_time:.2f} seconds")
        
        # 5. Save and visualize results
        print("\n" + "="*60)
        print("Step 4: Save and visualize results")
        print("="*60)
        
        # save clustering results
        results_df = save_clustering_results(y_pred, person_ids, args.save_dir)
        
        # generate visualization charts
        if args.visualization:
            print("\nGenerating visualization charts...")
            try:
                visualize_results(x, y_pred, person_ids, args.save_dir, 'tsne')
                visualize_results(x, y_pred, person_ids, args.save_dir, 'pca')
            except Exception as e:
                print(f"Visualization generation failed: {e}")
        
        # 6. Final report
        print("\n" + "="*60)
        print("Clustering analysis completed!")
        print("="*60)
        
        print(f"Dataset information:")
        print(f"  - Number of people: {len(person_ids)}")
        print(f"  - Number of proteins: {len(protein_ids)}")
        print(f"  - Cluster number: {args.n_clusters}")
        
        print(f"\nTraining time:")
        print(f"  - Pretraining: {pretrain_time:.2f} seconds")
        print(f"  - Clustering training: {clustering_time:.2f} seconds")
        print(f"  - Total time: {pretrain_time + clustering_time:.2f} seconds")
        
        print(f"\nResult files:")
        print(f"  - Clustering results: {os.path.join(args.save_dir, 'clustering_results.csv')}")
        print(f"  - Statistics information: {os.path.join(args.save_dir, 'clustering_statistics.txt')}")
        print(f"  - Model weights: {os.path.join(args.save_dir, 'DEC_model_final.weights.h5')}")
        
        if args.visualization:
            print(f"  - Visualization charts: {args.save_dir}/clustering_visualization_*.png")
        
        return results_df
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
