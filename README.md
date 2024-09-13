# TriMoGCL: A Comprehensive Graph Neural Network Method for Predicting Triplet Motifs in Disease-Drug-Gene Interactions.
## Overview
This repository contains codes necessary to run the TriMoGCL algorithm.

## Running Environment
- Python == 3.8.0
- PyTorch == 1.13.1
- scikit-learn == 1.3.0


## Dataset
**Drug Repurposing Knowledge Graph (DRKG)** is a large comprehensive biological knowledge graph that involves genes, drugs, diseases, biological processes, side effects, and symptoms. It includes 97,238 entities belonging to 13 entity types; and 5,874,261 triplets belonging to 107 edge types. https://github.com/gnn4dr/DRKG

**Multi-scale interactome (MS) network** is a small heterogeneous knowledge graph containing drug, disease, protein, and gene ontology biological function nodes. 
The MS consists of 29,959 nodes belonging to 4 node types and 478,728 edges belonging to 4 edge types.
https://github.com/snap-stanford/multiscale-interactome

## Usage
- Run code for multi-classification for motifs of triplets.
```
bash mc.sh
```

- Run code for binary-classification for motifs of triplets.
```
bash binary.sh
```



