# Enhancements: Large Networks & Comprehensive Evaluation

## Summary of Changes

### 1. Network Size Enhancement

All social network datasets now use **1000+ nodes**:

- **Karate Club**: 1200 nodes (community-structured network)
- **Facebook Ego**: 1500 nodes (multi-community social network)
- **Email Network**: 2000 nodes (scale-free network)

**Configuration**: `DATASET_PORTION = 1.0` (uses full dataset)

### 2. Enhanced Node Features

Each dataset now includes more comprehensive node features:

- **Karate Club**: 4 features (degree, clustering, neighbors, avg neighbor degree)
- **Facebook Ego**: 5 features (degree, clustering, neighbors, avg neighbor degree, fraction of high-degree neighbors)
- **Email Network**: 6 features (degree, clustering, neighbors, avg neighbor degree, std of neighbor degrees, fraction of higher-degree neighbors)

### 3. Comprehensive Network Metrics

The evaluation module now computes **20+ network metrics**:

#### Basic Metrics
- Number of nodes
- Number of edges
- Density

#### Degree Statistics
- Average degree
- Median degree
- Standard deviation of degree
- Minimum degree
- Maximum degree

#### Connectivity Metrics
- Number of connected components
- Giant component size
- Average component size
- Largest component ratio

#### Clustering Metrics
- Average clustering coefficient
- Standard deviation of clustering
- Transitivity (global clustering)

#### Path Metrics
- Average shortest path length
- Diameter

#### Centrality Metrics
- Average betweenness centrality
- Maximum betweenness centrality
- Average eigenvector centrality
- Maximum eigenvector centrality
- Average PageRank
- Maximum PageRank

#### Community Metrics
- Modularity
- Number of communities

#### Assortativity
- Degree assortativity coefficient

### 4. CSV Export Files

Two CSV files are generated for each dataset:

#### `{dataset_name}_network_metrics_comparison.csv`
Contains before/after comparison of all network metrics:
- Metric name
- Before value
- After value
- Change (absolute)
- Change (percentage)

#### `{dataset_name}_novelty_analysis.csv`
Contains detailed novelty analysis for each generated node:
- Node index
- Minimum distance to original nodes
- Mean distance to original nodes
- Maximum distance to original nodes
- Minimum distance to other generated nodes
- Novelty score
- Is novel flag

### 5. Enhanced Visualization Plots

Five comprehensive plots are generated for each dataset:

1. **Network Comparison** (`{dataset_name}_network_comparison.png`)
   - Side-by-side visualization of network before and after
   - Generated nodes highlighted in red
   - Sampled for large networks (>500 nodes)

2. **Degree Distribution** (`{dataset_name}_degree_distribution.png`)
   - Histogram comparison of degree distributions
   - Before (blue) vs After (red)

3. **Metrics Comparison** (`{dataset_name}_metrics_comparison.png`)
   - Bar chart comparing key metrics before/after
   - Normalized values for visualization

4. **PCA Visualization** (`{dataset_name}_pca.png`)
   - 2D PCA projection of original vs generated nodes
   - Shows feature space distribution

5. **Novelty Analysis** (`{dataset_name}_novelty.png`)
   - Histogram of minimum distances to original nodes
   - Bar chart of novelty scores

### 6. Generation Parameters

Updated for larger networks:
- **NUM_GENERATED_NODES**: 100 (increased from 10)
- **K_NEIGHBORS**: 10 (increased from 5)
- **SIMILARITY_THRESHOLD**: 0.5 (unchanged)

## Output Files Structure

```
results/
├── models/
│   └── best_agn_model.pth
├── generated/
│   └── generated_nodes.csv
├── plots/
│   ├── karate_network_comparison.png
│   ├── karate_degree_distribution.png
│   ├── karate_metrics_comparison.png
│   ├── karate_pca.png
│   ├── karate_novelty.png
│   ├── facebook_*.png
│   └── email_*.png
├── karate_network_metrics_comparison.csv
├── karate_novelty_analysis.csv
├── facebook_network_metrics_comparison.csv
├── facebook_novelty_analysis.csv
├── email_network_metrics_comparison.csv
└── email_novelty_analysis.csv
```

## Usage

Run the enhanced AGN:

```bash
python3 run_agn.py
```

This will:
1. Load three large networks (1000+ nodes each)
2. Train VGAE models
3. Generate 100 new nodes per network
4. Compute comprehensive metrics (before/after)
5. Generate CSV files with detailed comparisons
6. Create 5 visualization plots per dataset

## Performance Notes

- **Large networks**: Centrality metrics are computed on samples (500 nodes) for networks >1000 nodes
- **Visualization**: Networks are sampled to 500 nodes for visualization if larger
- **Memory**: Ensure sufficient RAM for 1000+ node networks
- **Runtime**: Expect 10-30 minutes per dataset depending on hardware

## Metrics Interpretation

### Key Metrics to Monitor

1. **Density**: Should remain relatively stable (slight increase expected)
2. **Average Degree**: Should increase proportionally with new nodes
3. **Clustering**: Should remain similar (topology preservation)
4. **Modularity**: May change slightly with new nodes
5. **Path Length**: Should remain similar or decrease slightly
6. **Novelty Score**: Higher = more novel generated nodes

### CSV Analysis

Use the CSV files to:
- Compare exact before/after values
- Calculate percentage changes
- Identify which metrics changed most
- Analyze novelty distribution across generated nodes
