module LMDiskANN

using Mmap
using Random
using LinearAlgebra
using Serialization
using LevelDB2
using Distances

include("UserIdMapping.jl")

export open_databases, close_databases, insert_key!, get_id_from_key, get_key_from_id
export delete_by_key!, delete_by_id!, count_entries, clear_database!, clear_all_databases!, list_all_keys

export create_index, load_index, close_id_mapping, ann_insert!, ann_delete!, search
export get_embedding_from_id, get_embedding_from_key

const DEFAULT_MAX_DEGREE = 64  # max number of neighbors
const SEARCH_LIST_SIZE   = 64  # search BFS/greedy queue size
const EF_SEARCH          = 300 # search expansion factor
const EF_CONSTRUCTION    = 400 # construction expansion factor

"""
    LMDiskANNIndex{T<:AbstractFloat}

Main data structure for LM-DiskANN, parametrized on `T` (the floating-point type
for the vectors). By default, `T=Float32`. The adjacency remains `Int32`.

Fields:
- `vecfile`, `adjfile`, `metafile`: Disk file paths
- `dim`: Dimensionality of vectors
- `maxdegree`: Max number of neighbors
- `vecs::Array{T}`: Memory-mapped array for vectors
- `adjs::Array{Int32}`: Memory-mapped adjacency structure
- `num_points`: Current number of points
- `freelist`: List of deleted IDs
- `entrypoint`: Graph entry point
- `id_mapping_forward`, `id_mapping_reverse`: Optional LevelDB for user key ↔ ID
"""
mutable struct LMDiskANNIndex{T<:AbstractFloat}
    vecfile::String
    adjfile::String
    metafile::String
    
    dim::Int
    maxdegree::Int
    
    vecs::Array{T}
    adjs::Array{Int32}
    
    num_points::Int
    freelist::Vector{Int}
    entrypoint::Int

    id_mapping_forward::Union{LevelDB2.DB{String,String},Nothing}
    id_mapping_reverse::Union{LevelDB2.DB{String,String},Nothing}
end

"""
    _read_metadata(metafile::String)
Reads metadata (num_points, dim, maxdegree, freelist, entrypoint) via Serialization.
"""
function _read_metadata(metafile::String)
    open(metafile, "r") do io
        md = deserialize(io)
        return md["num_points"], md["dim"], md["maxdegree"], md["freelist"], md["entrypoint"]
    end
end


"""
    _write_metadata(metafile::String, num_points, dim, maxdegree, freelist, entrypoint)

Writes metadata to a `.meta` file using Julia's Serialization.
"""
function _write_metadata(metafile::String,
                         num_points::Int,
                         dim::Int,
                         maxdegree::Int,
                         freelist::Vector{Int},
                         entrypoint::Int)
    md = Dict(
        "num_points" => num_points,
        "dim"        => dim,
        "maxdegree"  => maxdegree,
        "freelist"   => freelist,
        "entrypoint" => entrypoint
    )
    open(metafile, "w") do io
        serialize(io, md)
    end
end


"""
    _mmap_arrays(vecfile::String, adjfile::String, dim::Int, maxdegree::Int, num_points::Int;T::Type=Float32)

Creates or opens memory maps for the vectors and adjacency arrays. Optionally set the type
Returns a tuple (vec_mmap, adj_mmap).
 - `vec_mmap` will be of shape (dim, num_points)
 - `adj_mmap` is a 2D array: (maxdegree, num_points)
"""
function _mmap_arrays(vecfile::String,
                      adjfile::String,
                      dim::Int,
                      maxdegree::Int,
                      num_points::Int;
                      T::Type=Float32)
    # total size for vectors
    vsize = dim * num_points
    # total size for adjacency
    asize = maxdegree * num_points
    
    # ensure vector file is large enough
    open(vecfile, "a+") do f
        seekend(f)
        cursz = position(f)
        needed = vsize * sizeof(T)   # T instead of Float32
        if cursz < needed
            write(f, zeros(UInt8, needed - cursz))
        end
    end
    
    # ensure adjacency file is large enough
    open(adjfile, "a+") do f
        seekend(f)
        cursz = position(f)
        needed = asize * sizeof(Int32)
        if cursz < needed
            write(f, zeros(UInt8, needed - cursz))
        end
    end
    
    # memory-map the vector file as T
    vec_mmap = open(vecfile, "r+") do f
        Mmap.mmap(f, Array{T,2}, (dim, num_points))
    end
    
    # memory-map the adjacency file as Int32
    adj_mmap = open(adjfile, "r+") do f
        Mmap.mmap(f, Array{Int32,2}, (maxdegree, num_points))
    end
    
    return vec_mmap, adj_mmap
end


"""
    _init_files(vecfile, adjfile, metafile, dim)

Initializes (or overwrites) new files for a fresh index with no points.
"""
function _init_files(vecfile::String,
                     adjfile::String,
                     metafile::String;
                     dim::Int,
                     maxdegree::Int=DEFAULT_MAX_DEGREE)
    num_points = 0
    freelist   = Int[]
    entrypoint = -1

    _write_metadata(metafile, num_points, dim, maxdegree, freelist, entrypoint)
    
    open(vecfile, "w") do f end
    open(adjfile, "w") do f end
end

# """
#     _compute_distance(x, y)

# Compute Euclidean distance for T=Float32 (the default param).
# """
# @inline function _compute_distance(x::AbstractVector{Float32}, y::AbstractVector{Float32})
#     return evaluate(Euclidean(), x, y)
# end


"""
    _get_neighbors(index, node_id) -> Vector{Int}

Gets the neighbor list of node_id from the adjacency matrix, ignoring -1 values
which indicate empty entries. Returns an Int vector of neighbor IDs.
"""
function _get_neighbors(index::LMDiskANNIndex{T}, node_id::Int) where {T<:AbstractFloat}
    neighbors = Int[]
    for i in 1:index.maxdegree
        nbr_id = index.adjs[i, node_id+1]
        if nbr_id >= 0
            push!(neighbors, nbr_id)
        end
    end
    return neighbors
end


"""
    _set_neighbors(index, node_id, neighbor_ids::Vector{Int})

Overwrite adjacency entries for node_id with neighbor_ids (up to index.maxdegree).
If neighbor_ids is shorter than maxdegree, fill the remainder with -1 to indicate empty.
The type parameterization for the index is also there.
"""
function _set_neighbors(index::LMDiskANNIndex{T}, node_id::Int, neighbor_ids::Vector{Int}) where {T<:AbstractFloat}
    maxd = index.maxdegree
    if length(neighbor_ids) > maxd
        neighbor_ids = neighbor_ids[1:maxd]
    end
    for i in 1:maxd
        index.adjs[i, node_id+1] = i <= length(neighbor_ids) ? Int32(neighbor_ids[i]) : Int32(-1)
    end
end


"""
    create_index(path_prefix::String, dim::Int; maxdegree::Int=DEFAULT_MAX_DEGREE)

Creates a brand new LM-DiskANN index on disk with the given dimension, storing
to files: `path_prefix.vec`, `path_prefix.adj`, `path_prefix.meta`.

# Arguments
- `path_prefix::String`: Prefix for the index files (without extension)
- `dim::Int`: Dimensionality of the vectors to be indexed

# Optional arguments
= `T::Type=Float32`: the typer for the embedding vectors
- `maxdegree::Int=DEFAULT_MAX_DEGREE`: Maximum number of neighbors per node

# Returns
- `LMDiskANNIndex`: A new index instance

# Example
```julia
index = LMDiskANN.create_index("my_index", 128)
```
"""
function create_index(path_prefix::String, dim::Int;
                     T::Type=Float32,
                     maxdegree::Int=DEFAULT_MAX_DEGREE)
    vecfile  = path_prefix * ".vec"
    adjfile  = path_prefix * ".adj"
    metafile = path_prefix * ".meta"

    _init_files(vecfile, adjfile, metafile; dim=dim, maxdegree=maxdegree)
    num_points, dim_, maxdegree_, freelist, entrypoint = _read_metadata(metafile)
    
    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim_, maxdegree_,
                                      max(1, num_points); T=T)

    forward_path = path_prefix * "forward_db.leveldb"
    reverse_path = path_prefix * "reverse_db.leveldb"
    db_forward, db_reverse = open_databases(forward_path, reverse_path; create_if_missing=true)
    
    return LMDiskANNIndex{T}(vecfile, adjfile, metafile,
                             dim_, maxdegree_,
                             vec_mmap, adj_mmap,
                             num_points, freelist,
                             entrypoint,
                             db_forward, db_reverse)
end

"""
    load_index(path_prefix::String; T=Float32)
Loads an existing index, specifying T if it wasn't Float32 originally.
"""
function load_index(path_prefix::String; T::Type=Float32)
    vecfile  = path_prefix * ".vec"
    adjfile  = path_prefix * ".adj"
    metafile = path_prefix * ".meta"

    forward_file = path_prefix * "forward_db.leveldb"
    reverse_file = path_prefix * "reverse_db.leveldb"
    
    if !(isfile(vecfile) && isfile(adjfile) && isfile(metafile) &&
         isfile(forward_file) && isfile(reverse_file))
        error("Index files not found at prefix: $path_prefix")
    end
    
    num_points, dim_, maxdeg_, freelist, entrypoint = _read_metadata(metafile)
    
    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim_, maxdeg_,
                                      max(1, num_points); T=T)

    db_forward, db_reverse = open_databases(forward_file, reverse_file; create_if_missing=false)
    
    return LMDiskANNIndex{T}(vecfile, adjfile, metafile,
                             dim_, maxdeg_,
                             vec_mmap, adj_mmap,
                             num_points, freelist,
                             entrypoint,
                             db_forward, db_reverse)
end


"""
    resize_index!(index::LMDiskANNIndex{T}, new_size::Int) where {T<:AbstractFloat}

Resizes the memory-mapped files to accommodate at least new_size points.
This is used internally when the index needs to grow.
"""
function resize_index!(index::LMDiskANNIndex{T}, new_size::Int) where {T<:AbstractFloat}
    vecfile = index.vecfile
    adjfile = index.adjfile
    dim     = index.dim
    maxd    = index.maxdegree

    old_vecs = index.vecs
    old_adjs = index.adjs
    finalize(old_vecs)
    finalize(old_adjs)
    GC.gc()

    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim, maxd, new_size; T=T)
    index.vecs = vec_mmap
    index.adjs = adj_mmap
    return index
end


"""
    save_index(index::LMDiskANNIndex{T}) where {T<:AbstractFloat}

Saves the current state of the index metadata to disk.
The vector and adjacency data are already on disk via memory mapping.

# Arguments
- `index::LMDiskANNIndex`: The index to save

# Returns
- `LMDiskANNIndex`: The same index instance

# Example
```julia
LMDiskANN.save_index(index)
```
"""
function save_index(index::LMDiskANNIndex{T}) where {T<:AbstractFloat}
    _write_metadata(index.metafile,
                    index.num_points,
                    index.dim,
                    index.maxdegree,
                    index.freelist,
                    index.entrypoint)
    return index
end

"""
    close_id_mapping(index::LMDiskANNIndex{T})
Closes the user-key DB if open.
"""
function close_id_mapping(index::LMDiskANNIndex{T}) where {T<:AbstractFloat}
    if index.id_mapping_forward !== nothing && index.id_mapping_reverse !== nothing
        close_databases(index.id_mapping_forward, index.id_mapping_reverse)
        index.id_mapping_forward = nothing
        index.id_mapping_reverse = nothing
    end
    return index
end


"""
    _search_graph(index::LMDiskANNIndex{T}, query_vec::Vector{T}, ef::Int) where {T<:AbstractFloat}

Greedy BFS-like search using the adjacency graph.
This returns up to `ef` candidate neighbor IDs (approx nearest).
Implements the core of Algorithm 1 from the LM-DiskANN paper.
"""
function _search_graph(index::LMDiskANNIndex{T}, query_vec::Vector{T}, ef::Int) where {T<:AbstractFloat}
    if index.entrypoint < 0 || index.num_points == 0
        return Int[]
    end
    
    visited    = Set{Int}()
    candidates = Vector{Tuple{T, Int}}()
    results    = Vector{Tuple{T, Int}}()

    entry_id   = index.entrypoint
    entry_vec  = index.vecs[:, entry_id+1]
    entry_dist = evaluate(Euclidean(), entry_vec, query_vec)  

    push!(visited, entry_id)
    push!(candidates, (entry_dist, entry_id))
    push!(results, (entry_dist, entry_id))

    while !isempty(candidates)
        sort!(candidates, by=x->x[1])
        current_dist, current_id = popfirst!(candidates)
        
        if !isempty(results) && last(results)[1] < current_dist
            break
        end
        
        neighbors = _get_neighbors(index, current_id)
        for nbr_id in neighbors
            if nbr_id < 0 || (nbr_id in visited)
                continue
            end
            push!(visited, nbr_id)

            nbr_vec = index.vecs[:, nbr_id+1]
            nbr_dist= evaluate(Euclidean(), nbr_vec, query_vec)

            sort!(results, by=x->x[1])
            if length(results) < ef || nbr_dist < last(results)[1]
                push!(candidates, (nbr_dist, nbr_id))
                push!(results, (nbr_dist, nbr_id))
                if length(results) > ef
                    sort!(results, by=x->x[1])
                    pop!(results)
                end
            end
        end
    end
    sort!(results, by=x->x[1])
    return [pair[2] for pair in results]
end


"""
    search(index::LMDiskANNIndex{T},
                query_vec::AbstractVector{<:AbstractFloat};
                topk::Int=10) where {T<:AbstractFloat}

Returns top-k approximate nearest neighbors for query_vec.

# Arguments
- `index::LMDiskANNIndex`: The index to search
- `query_vec::AbstractVector{Float32}`: The query vector
- `topk::Int=10`: Number of nearest neighbors to return

# Returns
- `Tuple (key,id)`: Keys and IDs of the top-k nearest neighbors (string, int)

# Example
```julia
results = LMDiskANN.search(index, query_vec, topk=5)
```
"""
function search(index::LMDiskANNIndex{T},
                query_vec::AbstractVector{<:AbstractFloat};
                topk::Int=10) where {T<:AbstractFloat}
    if index.num_points == 0
        return Vector{Tuple{Union{String,Nothing}, Int}}()
    end
    local_q = convert(Vector{T}, query_vec)
    ef_candidates = _search_graph(index, local_q, max(topk, EF_SEARCH))

    dist_id_pairs = Vector{Tuple{T,Int}}()
    for cid in ef_candidates
        v = index.vecs[:, cid+1]
        d = evaluate(Euclidean(), v, local_q)
        push!(dist_id_pairs, (d, cid))
    end
    sort!(dist_id_pairs, by=x->x[1])

    k = min(topk, length(dist_id_pairs))
    results = Vector{Tuple{Union{String,Nothing}, Int}}()
    for i in 1:k
        cid = dist_id_pairs[i][2]
        user_key = get_key_from_id(index.id_mapping_reverse, cid+1)
        push!(results, (user_key, cid+1))
    end
    return results
end



"""
    _prune_neighbors(index::LMDiskANNIndex, node_id::Int, candidates::Vector{Int})

Prune the candidate neighbors for a node to maintain the GSNG property.
Returns a pruned list of neighbors.
"""
function _prune_neighbors(index::LMDiskANNIndex{T}, node_id::Int, candidates::Vector{Int}) where {T<:AbstractFloat}
    if length(candidates) <= index.maxdegree
        return candidates
    end
    
    node_vec = index.vecs[:, node_id+1]
    dist_id_pairs = Vector{Tuple{T,Int}}()
    for cand_id in candidates
        cand_vec = index.vecs[:, cand_id+1]
        d = evaluate(Euclidean(), node_vec, cand_vec)
        push!(dist_id_pairs, (d, cand_id))
    end
    sort!(dist_id_pairs, by=x->x[1])
    return [p[2] for p in dist_id_pairs[1:index.maxdegree]]
end


"""
    ann_insert!(index::LMDiskANNIndex{T},
                     new_vec::AbstractVector{<:AbstractFloat};
                     key::Union{Nothing,String}=nothing) where {T<:AbstractFloat}

Insert a new vector into the index. Updates the adjacency structure.
Returns the assigned ID of the newly inserted vector.
Implements Algorithm 2 from the LM-DiskANN paper.

# Arguments
- `index::LMDiskANNIndex{T}`: The index to insert into
- `new_vec::Vector{<:AbstractFloat}`: The vector to insert

# Optional Arguments
- `key`:: String: the key the user wants to associate with the new vector
# Returns
- `(key,Int)`: The tuple of the key (string) and ID (int) assigned to the inserted vector

# Example
```julia
(key,id) = LMDiskANN.ann_insert!(index, vector)
```
"""
function ann_insert!(index::LMDiskANNIndex{T},
                     new_vec::AbstractVector{<:AbstractFloat};
                     key::Union{Nothing,String}=nothing) where {T<:AbstractFloat}
    new_id = !isempty(index.freelist) ? pop!(index.freelist) : index.num_points
    if new_id == index.num_points
        index.num_points += 1
    end
    needed_size = new_id + 1
    curr_cap    = size(index.vecs, 2)
    if needed_size > curr_cap
        growby = max(1024, curr_cap)
        new_cap= max(needed_size, curr_cap + growby)
        resize_index!(index, new_cap)
    end

    local_vec = convert(Vector{T}, new_vec)
    index.vecs[:, new_id+1] = local_vec

    if index.entrypoint < 0
        index.entrypoint = new_id
        _set_neighbors(index, new_id, Int[])
        save_index(index)
        if isnothing(key)
            insert_key!(index.id_mapping_forward, index.id_mapping_reverse,
                        string(new_id+1), new_id+1)
            return (string(new_id+1), new_id+1)
        else
            insert_key!(index.id_mapping_forward, index.id_mapping_reverse, key, new_id+1)
            return (key, new_id+1)
        end
    end

    # BFS
    nearest_1based = [r[2] for r in search(index, local_vec, topk=index.maxdegree)]
    nearest_0based = [x-1 for x in nearest_1based]

    _set_neighbors(index, new_id, nearest_0based)
    for nbr_id in nearest_0based
        nbrs = _get_neighbors(index, nbr_id)
        push!(nbrs, new_id)
        pruned = _prune_neighbors(index, nbr_id, nbrs)
        _set_neighbors(index, nbr_id, pruned)
    end
    save_index(index)

    if isnothing(key)
        insert_key!(index.id_mapping_forward, index.id_mapping_reverse,
                    string(new_id+1), new_id+1)
        return (string(new_id+1), new_id+1)
    else
        insert_key!(index.id_mapping_forward, index.id_mapping_reverse, key, new_id+1)
        return (key, new_id+1)
    end
end


"""
    ann_delete!(index::LMDiskANNIndex{T}, node_id::Union{Int,String}) where {T<:AbstractFloat}

Delete a vector (and adjacency) from the index.
Implements Algorithm 3 from the LM-DiskANN paper.

# Arguments
- `index::LMDiskANNIndex`: The index to delete from
- `node_id::Int`: The ID of the vector to delete

# Returns
- `LMDiskANNIndex`: The updated index instance

# Example
```julia
LMDiskANN.ann_delete!(index, id)
```
"""
function ann_delete!(index::LMDiskANNIndex{T}, node_id::Union{Int,String}) where {T<:AbstractFloat}
    if node_id isa String
        local_id = get_id_from_key(index.id_mapping_forward, node_id)
        if local_id === nothing
            return nothing
        end
        node_id = local_id
    end

    local_id = node_id - 1
    if local_id < 0 || local_id >= index.num_points
        error("Invalid local_id: $node_id")
    end
    if local_id in index.freelist
        error("Node $local_id is already deleted")
    end

    neighbors = _get_neighbors(index, local_id)
    for nbr_id in neighbors
        nbr_neighbors = _get_neighbors(index, nbr_id)
        filter!(x-> x!=local_id, nbr_neighbors)
        _set_neighbors(index, nbr_id, nbr_neighbors)
    end

    _set_neighbors(index, local_id, Int[])
    if index.entrypoint == local_id
        index.entrypoint = -1
        for cand in 0:(index.num_points-1)
            if cand != local_id && !(cand in index.freelist)
                index.entrypoint = cand
                break
            end
        end
    end

    push!(index.freelist, local_id)
    index.vecs[:, local_id+1] .= zero(T)
    save_index(index)

    delete_by_id!(index.id_mapping_forward, index.id_mapping_reverse, node_id)
    return nothing
end


"""
    get_embedding_from_id(index::LMDiskANNIndex{T}, id::Int)::Vector{T} where {T<:AbstractFloat}

Given a **1-based** ID (the same ID you'd see returned by `ann_insert!` or in `search`),
retrieve the stored embedding vector from `index.vecs`.

Throws an error if `id` is out of range or if it's already deleted (i.e., in the
freelist).
"""
function get_embedding_from_id(index::LMDiskANNIndex{T}, id::Int)::Vector{T} where {T<:AbstractFloat}
    local_id = id - 1
    if local_id < 0 || local_id >= index.num_points
        error("Invalid ID: $id (local_id=$local_id).")
    end
    if local_id in index.freelist
        error("ID $id refers to a deleted node.")
    end
    return copy(index.vecs[:, local_id+1])
end


"""
    get_embedding_from_key(index::LMDiskANNIndex{T}, key::String)::Vector{T} where {T<:AbstractFloat}

ARgument is a **string key** (which was stored during `ann_insert!(..., key=...)`),
look up the 1-based ID from the forward DB, then retrieve the embedding.

Throws an error if the key doesn't exist
"""
function get_embedding_from_key(index::LMDiskANNIndex{T}, key::String)::Vector{T} where {T<:AbstractFloat}
    if index.id_mapping_forward === nothing
        error("No forward LevelDB is open.")
    end
    the_id = get_id_from_key(index.id_mapping_forward, key)
    if the_id === nothing
        error("Key '$key' not found.")
    end
    return get_embedding_from_id(index, the_id)
end

end # module
