#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Protein Clustering Analysis Script

Run Configuration:
- Number of clusters: 3
- Data scaling: Standard scaling
- Enable visualization

"""

import os
import sys
import subprocess

def main():
    """Run protein clustering analysis"""
    
    print("="*60)
    print("Protein Clustering Analysis")
    print("="*60)
    print("Configuration Parameters:")
    print("  - Data file: protein.csv")
    print("  - Number of clusters: 5")
    print("  - Data scaling: Standard (standard)")
    print("  - Visualization: Enabled")
    print("  - Results directory: results/protein_5clusters")
    print("="*60)
    
    # Check if data file exists
    data_file = r"C:\Users\yiwei\Desktop\DEC\DEC-keras\protein.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} file not found")
        print("Please ensure protein.csv file exists in the specified path")
        return False
    
    # Create results directory
    results_dir = "results/protein_5clusters"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Create results directory: {results_dir}")
    
    # Build run command
    cmd = [
        "python", "protein_clustering.py",
        "--data_path", r"C:\Users\yiwei\Desktop\DEC\DEC-keras\protein.csv",
        "--n_clusters", "5",
        "--scaling_method", "standard",
        "--save_dir", results_dir,
        "--visualization",
        "--pretrain_epochs", "200",
        "--maxiter", "20000",
        "--batch_size", "256",
        "--update_interval", "140",
        "--random_state", "42"
    ]
    
    print("\nStart running clustering analysis...")
    print("Command:", " ".join(cmd))
    print("\n" + "-"*60)
    
    try:
        # Execute command
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n" + "-"*60)
        print("Clustering analysis completed!")
        print(f"Results saved in: {results_dir}")
        print("\nGenerated files:")
        
        # List generated files
        expected_files = [
            "clustering_results.csv",
            "clustering_statistics.txt", 
            "clustering_visualization_tsne.png",
            "clustering_visualization_pca.png",
            "DEC_model_final.weights.h5",
            "ae_weights.weights.h5",
            "pretrain_log.csv",
            "dec_log.csv"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                print(f"  {filename} exists")
            else:
                print(f"  {filename} not generated")
        
        print(f"\nView clustering results:")
        print(f"  - Clustering Results: {os.path.join(results_dir, 'clustering_results.csv')}")
        print(f"  - Statistics Report: {os.path.join(results_dir, 'clustering_statistics.txt')}")
        print(f"  - Visualization Table: {results_dir}/clustering_visualization_*.png")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nRun failed, error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("\nError: protein_clustering.py file not found")
        print("Please ensure protein_clustering.py is in the current directory")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nClustering analysis completed successfully!")
    else:
        print("\nClustering analysis failed")
        sys.exit(1)
