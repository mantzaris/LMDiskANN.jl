```@meta
CurrentModule = LMDiskANN
```

# LMDiskANN

**LM-DiskANN** is an approximate nearest‐neighbor indexing library that builds and maintains a disk‐resident graph structure for high‐dimensional data. It aims to provide a balance between fast search times, memory efficiency, and support for incremental updates (insertion and deletion). Under the hood, LM-DiskANN stores vector embeddings in memory‐mapped files, limiting in‐memory usage while preserving quick random access. It also leverages a graph-based BFS expansion to find candidate neighbors, with user‐configurable parameters like search expansion factor (`EF_SEARCH`) and maximum neighborhood size (`maxdegree`).

"LM-diskann: Low memory footprint in disk-native dynamic graph-based ann indexing." Pan, Yu, Jianxin Sun, and Hongfeng Yu, 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023.

It offers the *optional user‐key mapping*, allowing users to associate string keys with each embedding. This means you can insert vectors with a custom ID (e.g., "image001"), search for nearest neighbors, then retrieve or delete by either the numeric internal ID or the user key. The library includes utility functions to open, close, and clear these user‐key databases, enabling a more flexible integration with real-world applications where data might be identified by strings or external IDs.


### Example Usage with Synthetic Data

Below is a short script you can run to see LM-DiskANN in action. We’ll:
1. Create an index for `dim=5` vectors.
2. Insert a few vectors—some with user keys, some without.
3. Search for a known vector.
4. Delete by either numeric ID or user key.

```julia
using LMDiskANN  
using Random

# 1 make an index with dimension 5
dim = 5
index = create_index("example_index", dim)

# 2 insert a couple of random vectors
vec1 = rand(Float32, 5)
(key1, id1) = ann_insert!(index, vec1)
@show key1, id1   # e.g. might be ("1", 1) if no user key was specified

# insert a vector with a user key
vec2 = rand(Float32, 5)
(user_key, id2) = ann_insert!(index, vec2; key="my_special_vector")
@show user_key, id2  # e.g. ("my_special_vector", 2)

# 3 search for vec1
results = search(index, vec1, topk=3)  # returns an array of (maybeKey, ID)
println("Search results for vec1 = ", results)

# 4 retrieve the actual embedding from the index by numeric ID
retrieved_vec1 = get_embedding_from_id(index, id1)
println("retrieved_vec1 == vec1 ? ", all(retrieved_vec1 .== vec1))

# retrieve by user key
retrieved_vec2 = get_embedding_from_key(index, "my_special_vector")
println("retrieved_vec2 == vec2 ? ", all(retrieved_vec2 .== vec2))

# 5 delete an entry
ann_delete!(index, id1) # by numeric ID
ann_delete!(index, "my_special_vector") # by user key

# 6 confirm they're gone (search or retrieval should fail)
println("After deletions, search(vec1) => ", search(index, vec1, topk=3))
try
    get_embedding_from_id(index, id1)
catch e
    println("Caught an expected error when retrieving a deleted vector: ", e)
end
```


### Example real data embeddings from GLoVe

Build a 10 000-word cosine LM-DiskANN index and explore semantic neighbours using embedding vectors from GLoVe.

```julia
using Embeddings #downloads pre-trained vectors of word embeddings
using LMDiskANN #this package added via, add https://github.com/mantzaris/LMDiskANN.jl
using Distances, LinearAlgebra, Random, Printf

tab = load_embeddings(GloVe{:en}, 2; max_vocab_size = 10_000)
words = tab.vocab
vectors = tab.embeddings #100 × 10 000 Float32
(dim, N) = size(vectors)
@info "Loaded $N by $dim GloVe vectors"

word2id = Dict(w => i for (i, w) in enumerate(words)) #helper function
vectors ./= sqrt.(sum(abs2, vectors; dims = 1)) #normalisation

idx = create_index("glove_demo", dim; metric = CosineDist()) #<- create the store

for (vec, w) in zip(eachcol(vectors), words)  #insert into the lmdiskann the (vector, word)
    ann_insert!(idx, vec; key = w) #<- insert into the store
end
@info "Index built with $(idx.num_points) points"


nearest(w; k = 5) = let id = word2id[w]
    hits = search(idx, vectors[:, id], topk = k + 1) #<- search the store, first high result would be the word itself
    [words[h[2]] for h in hits[2:end]]
end

for w in ("king", "queen", "paris", "cat", "coffee")
           @printf("  %-6s -> %s\n", w, join(nearest(w), ", "))
       end
# prints eg:
#  king   -> prince, queen, son, brother, monarch
#  queen  -> princess, king, elizabeth, royal, lady
#  paris  -> france, london, brussels, french, rome
#  cat    -> dog, cats, pet, dogs, mouse
#  coffee -> tea, drinks, beer, wine, drink

#classic analogy for vectors

v = vectors[:, word2id["king"]] .- vectors[:, word2id["man"]] .+ vectors[:, word2id["woman"]]
v ./= norm(v) 


ignore = Set(["king", "man", "woman"]) #skip seed words
for (_key, id) in search(idx, v, topk = 10)
    w = words[id]
    w ∉ ignore && (println("king - man + woman -> ", w); break)
end

#prints:
# king - man + woman -> queen
```



Documentation for [LMDiskANN](https://github.com/mantzaris/LMDiskANN.jl).

```@index
```

```@autodocs
Modules = [LMDiskANN]
Private = false
```
