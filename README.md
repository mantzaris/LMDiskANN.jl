[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Docs - stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mantzaris.github.io/LMDiskANN.jl/stable)<!-- [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mantzaris.github.io/LMDiskANN.jl/)  -->
[![Docs - dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/LMDiskANN.jl/dev)
[![Build Status](https://github.com/mantzaris/LMDiskANN.jl/actions/workflows/ci.yml/badge.svg?branch=main&refresh=1)](https://github.com/mantzaris/LMDiskANN.jl/actions)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08199/status.svg)](https://doi.org/10.21105/joss.08199)

# LMDiskANN.jl
Julia Implementation of Low Memory Disk ANN (LM-DiskANN)

**LM-DiskANN** is a lightweight library for approximate nearest‐neighbor (ANN) indexing on disk. It creates a graph structure over vector embeddings, storing them in memory‐mapped files to keep the in‐memory footprint low. It also supports optional LevelDB databases for user‐key ↔ numeric ID lookups, making it easy to associate each embedding with a custom string ID.

## Key Features
- **Disk‐Resident**: Vectors and adjacency lists are stored in memory‐mapped files, reducing RAM usage.
- **Graph‐Based Search**: Leverages a BFS expansion (`EF_SEARCH`) for approximate neighbor lookups.
- **Insert & Delete**: Dynamically insert or remove embeddings without needing to rebuild the index.
- **User Keys**: Link a string key (e.g. `"image123"`) to your internal node ID; retrieve or delete by either integer ID or key.

## Quick Start Example


Install via: `(@v1.6) pkg> add https://github.com/mantzaris/LMDiskANN.jl`

```julia
using LMDiskANN
using Random

# create an index with dimension = 5
index = create_index("my_index", 5)

# insert a random vector
v1 = rand(Float32, 5)
(key1, id1) = ann_insert!(index, v1)  # returns (autoKey, 1)

# insert another vector with a custom string key
v2 = rand(Float32, 5)
(mykey, id2) = ann_insert!(index, v2; key="myvec")

# search for v1
results = search(index, v1, topk=3)
println("Results for v1 => ", results)

# retrieve embeddings
retrieved_v1 = get_embedding_from_id(index, id1)
retrieved_v2 = get_embedding_from_key(index, "myvec")

# delete by ID or key
ann_delete!(index, id1)
ann_delete!(index, "myvec")
```

## citing this work

- Mantzaris, A. V., (2025). LMDiskANN.jl: An Implementation of the Low Memory Disk Approximate Nearest Neighbors Search Algorithm. Journal of Open Source Software, 10(110), 8199, https://doi.org/10.21105/joss.08199

- @article{Mantzaris2025, doi = {10.21105/joss.08199}, url = {https://doi.org/10.21105/joss.08199}, year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {110}, pages = {8199}, author = {Alexander V. Mantzaris}, title = {LMDiskANN.jl: An Implementation of the Low Memory Disk Approximate Nearest Neighbors Search Algorithm}, journal = {Journal of Open Source Software} } 

## original paper introducing the LM-DiskANN algorithm
"LM-diskann: Low memory footprint in disk-native dynamic graph-based ann indexing." Pan, Yu, Jianxin Sun, and Hongfeng Yu, 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023.
