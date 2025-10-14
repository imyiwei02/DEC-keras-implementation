# DEC-keras-implementation
Deep Embedding for protein clustering based on https://github.com/XifengGuo/DEC-keras

## Environment
- Python 3.6+ 

- TensorFlow 2.x or Keras ≥ 2.0.9
- NumPy
- scikit-learn

## Use guide
```bash
# run by specifying parameters (cluster number) using the command line
python protein_clustering.py --data_path protein.csv --n_clusters 5 --scaling_method standard --visualization
```
