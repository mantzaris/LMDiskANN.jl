module LMDiskANN

using Mmap
using Random
using LinearAlgebra
using Serialization


const DEFAULT_MAX_DEGREE = 32 #max number of neighbors
const SEARCH_LIST_SIZE   = 64 #search BFS/greedy queue size
const EF_SEARCH          = 100 #search expansion factor
const EF_CONSTRUCTION    = 200 #construction expansion factor


"""
    LMDiskANNIndex

Main data structure for the LM-DiskANN index.
Stores vectors and adjacency information on disk using memory-mapped files.

Fields:
- `vecfile`: Path to vectors file
- `adjfile`: Path to adjacency file
- `metafile`: Path to metadata file
- `dim`: Dimensionality of vectors
- `maxdegree`: Maximum number of neighbors per node
- `vecs`: Memory-mapped array for vectors
- `adjs`: Memory-mapped adjacency structure
- `num_points`: Current number of points in the index
- `freelist`: List of deleted IDs that can be reused
- `entrypoint`: Entry point for graph search
"""
mutable struct LMDiskANNIndex
    vecfile::String #path to vectors file
    adjfile::String #path to adjacency file
    metafile::String # path to metadata file
    
    dim::Int #dimensionality of vectors
    maxdegree::Int #max number of neighbors
    
    vecs::Array{Float32} #memory-mapped array for vectors
    adjs::Array{Int32} #memory-mapped adjacency structure
    
    #cache for metadata (in RAM/core)
    num_points::Int #current number of points
    freelist::Vector{Int} #list of "deleted IDs" that can be reused
    
    #entry point (for graph search)
    entrypoint::Int
end



"""
    _read_metadata(metafile::String)

Reads metadata from a `.meta` file using Julia's Serialization.
Returns a tuple (num_points, dim, maxdegree, freelist, entrypoint).
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
function _write_metadata(metafile::String, num_points::Int, dim::Int, maxdegree::Int,
                         freelist::Vector{Int}, entrypoint::Int)
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
    _mmap_arrays(vecfile::String, adjfile::String, dim::Int, maxdegree::Int, num_points::Int)

Creates or opens memory maps for the vectors and adjacency arrays.
Returns a tuple (vec_mmap, adj_mmap).
 - `vec_mmap` will be of shape (dim, num_points)
 - `adj_mmap` is a 2D array: (maxdegree, num_points)
"""
function _mmap_arrays(vecfile::String, adjfile::String, dim::Int, maxdegree::Int, num_points::Int)
    #find total size
    vsize = dim * num_points
    asize = maxdegree * num_points
    
    #TODO: check!!!
    #create or resize vector file
    open(vecfile, "a+") do f
        #ensure file size is enough for vsize * 4 bytes (Float32)
        seekend(f)
        cursz = position(f)
        needed = vsize * sizeof(Float32)
        if cursz < needed
            write(f, zeros(UInt8, needed - cursz))
        end
    end
    
    #TODO: check!!!
    #create or resize adjacency file
    open(adjfile, "a+") do f
        # each adjacency entry is an Int32
        esz = asize * sizeof(Int32)
        seekend(f)
        cursz = position(f)
        if cursz < esz
            write(f, zeros(UInt8, esz - cursz))
        end
    end
    
    #TODO: check!!!
    #memory-map the vector file
    vec_mmap = open(vecfile, "r+") do f
        Mmap.mmap(f, Array{Float32,2}, (dim, num_points))
    end
    
    #memory-map the adjacency file
    adj_mmap = open(adjfile, "r+") do f
        Mmap.mmap(f, Array{Int32,2}, (maxdegree, num_points))
    end
    
    return vec_mmap, adj_mmap
end

"""
    _init_files(vecfile, adjfile, metafile, dim)

Initializes (or overwrites) new files for a fresh index with no points.
"""
function _init_files(vecfile::String, adjfile::String, metafile::String;
                     dim::Int, maxdegree::Int=DEFAULT_MAX_DEGREE)
    #no points initially
    num_points = 0
    freelist = Int[]
    entrypoint = -1
    
    #write empty metadata
    _write_metadata(metafile, num_points, dim, maxdegree, freelist, entrypoint)
    
    #create empty files for vectors and adjacency
    open(vecfile, "w") do f
        # Create an EMPTY file
    end
    
    open(adjfile, "w") do f
        # Create an EMPTY file
    end
    
    return
end

"""
    _compute_distance(x, y)

Compute Euclidean distance between two vectors x and y.
"""
@inline function _compute_distance(x::AbstractVector{Float32}, y::AbstractVector{Float32})
    return norm(x .- y)
end

"""
    _get_neighbors(index, node_id) -> Vector{Int}

Gets the neighbor list of node_id from the adjacency matrix, ignoring -1 values
which indicate empty entries. Returns an Int vector of neighbor IDs.
"""
function _get_neighbors(index::LMDiskANNIndex, node_id::Int)
    neighbors = Int[]
    for i in 1:index.maxdegree
        nbr_id = index.adjs[i, node_id+1]  #node_id+1 because of 1-based indexing in Julia
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
"""
function _set_neighbors(index::LMDiskANNIndex, node_id::Int, neighbor_ids::Vector{Int})
    maxd = index.maxdegree
    #truncate if needed
    if length(neighbor_ids) > maxd
        neighbor_ids = neighbor_ids[1:maxd]
    end
    #fill the adjacency column
    for i in 1:maxd
        if i <= length(neighbor_ids)
            index.adjs[i, node_id+1] = Int32(neighbor_ids[i])
        else
            index.adjs[i, node_id+1] = Int32(-1)
        end
    end
end



"""
    createIndex(path_prefix::String, dim::Int; maxdegree::Int=DEFAULT_MAX_DEGREE)

Creates a brand new LM-DiskANN index on disk with the given dimension, storing
to files: `path_prefix.vec`, `path_prefix.adj`, `path_prefix.meta`.

# Arguments
- `path_prefix::String`: Prefix for the index files (without extension)
- `dim::Int`: Dimensionality of the vectors to be indexed
- `maxdegree::Int=DEFAULT_MAX_DEGREE`: Maximum number of neighbors per node

# Returns
- `LMDiskANNIndex`: A new index instance

# Example
```julia
index = LMDiskANN.createIndex("my_index", 128)
```
"""
function createIndex(path_prefix::String, dim::Int; maxdegree::Int=DEFAULT_MAX_DEGREE)
    vecfile  = path_prefix * ".vec"
    adjfile  = path_prefix * ".adj"
    metafile = path_prefix * ".meta"
    
    #initialize new files
    _init_files(vecfile, adjfile, metafile; dim=dim, maxdegree=maxdegree)
    
    #now load them in a struct
    num_points, dim_, maxdegree_, freelist, entrypoint = _read_metadata(metafile)
    
    #create initial memory maps (empty at this point)
    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim, maxdegree, max(1, num_points))
    
    return LMDiskANNIndex(vecfile, adjfile, metafile,
                          dim_, maxdegree_,
                          vec_mmap, adj_mmap,
                          num_points, freelist,
                          entrypoint)
end

"""
    loadIndex(path_prefix::String)

Loads an existing LM-DiskANN index from disk.

# Arguments
- `path_prefix::String`: Prefix for the index files (without extension)

# Returns
- `LMDiskANNIndex`: The loaded index instance

# Example
```julia
index = LMDiskANN.loadIndex("my_index")
```
"""
function loadIndex(path_prefix::String)
    vecfile  = path_prefix * ".vec"
    adjfile  = path_prefix * ".adj"
    metafile = path_prefix * ".meta"
    
    #check if files exist
    if !isfile(vecfile) || !isfile(adjfile) || !isfile(metafile)
        error("Index files not found at prefix: $path_prefix")
    end
    
    #read metadata
    num_points, dim, maxdegree, freelist, entrypoint = _read_metadata(metafile)
    
    #memory-map the files
    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim, maxdegree, max(1, num_points))
    
    return LMDiskANNIndex(vecfile, adjfile, metafile,
                          dim, maxdegree,
                          vec_mmap, adj_mmap,
                          num_points, freelist,
                          entrypoint)
end

"""
    resizeIndex!(index::LMDiskANNIndex, new_size::Int)

Resizes the memory-mapped files to accommodate at least new_size points.
This is used internally when the index needs to grow.
"""
function resizeIndex!(index::LMDiskANNIndex, new_size::Int)
    #store the current values
    vecfile = index.vecfile
    adjfile = index.adjfile
    dim = index.dim
    maxdegree = index.maxdegree
    
    #close existing memory maps by removing references
    old_vecs = index.vecs
    old_adjs = index.adjs
    
    #explicitly close the memory maps
    finalize(old_vecs)
    finalize(old_adjs)
    GC.gc() #force garbage collection to ensure memory maps are closed
    
    #resize the files
    vec_mmap, adj_mmap = _mmap_arrays(vecfile, adjfile, dim, maxdegree, new_size)
    
    #update the index with new memory maps
    index.vecs = vec_mmap
    index.adjs = adj_mmap
    
    return index
end

"""
    saveIndex(index::LMDiskANNIndex)

Saves the current state of the index metadata to disk.
The vector and adjacency data are already on disk via memory mapping.

# Arguments
- `index::LMDiskANNIndex`: The index to save

# Returns
- `LMDiskANNIndex`: The same index instance

# Example
```julia
LMDiskANN.saveIndex(index)
```
"""
function saveIndex(index::LMDiskANNIndex)
    _write_metadata(index.metafile, index.num_points, index.dim, index.maxdegree,
                   index.freelist, index.entrypoint)
    return index
end


"""
    _search_graph(index, query_vec, ef)

Greedy BFS-like search using the adjacency graph.
This returns up to `ef` candidate neighbor IDs (approx nearest).
Implements the core of Algorithm 1 from the LM-DiskANN paper.
"""
function _search_graph(index::LMDiskANNIndex, query_vec::Vector{Float32}, ef::Int)
    if index.entrypoint < 0 || index.num_points == 0
        return Int[] #no points in the index!
    end
    
    #initialize visited set, candidate set, and result set
    visited = Set{Int}()
    candidates = Vector{Tuple{Float32, Int}}() # (distance, node_id)
    results = Vector{Tuple{Float32, Int}}() # (distance, node_id)
    
    #start with the entry point
    entry_id = index.entrypoint
    entry_vec = index.vecs[:, entry_id+1]
    entry_dist = _compute_distance(entry_vec, query_vec)
    
    push!(visited, entry_id)
    push!(candidates, (entry_dist, entry_id))
    push!(results, (entry_dist, entry_id))
    
    #process candidates until none are left
    while !isempty(candidates)
        #get closest candidate to query
        sort!(candidates)
        current_dist, current_id = popfirst!(candidates)
        
        #if the farthest result is closer than the closest candidate, done
        if !isempty(results) && last(results)[1] < current_dist
            break
        end
        
        #get neighbors of current node
        neighbors = _get_neighbors(index, current_id)
        
        #process each neighbor
        for nbr_id in neighbors
            if nbr_id < 0 || nbr_id in visited
                continue
            end
            
            push!(visited, nbr_id)
            
            #calculate distance from neighbor to query
            nbr_vec = index.vecs[:, nbr_id+1]
            nbr_dist = _compute_distance(nbr_vec, query_vec)
            
            #if results not full or neighbor closer than furthest result
            if length(results) < ef || nbr_dist < last(sort!(results))[1]
                push!(candidates, (nbr_dist, nbr_id))
                push!(results, (nbr_dist, nbr_id))
                
                #hold only ef closest results
                if length(results) > ef
                    sort!(results)
                    pop!(results)
                end
            end
        end
    end
    
    #return node IDs of the closest neighbors
    sort!(results)
    return [id for (_, id) in results]
end

"""
    search(index::LMDiskANNIndex, query_vec::AbstractVector{Float32}; topk::Int=10)

Returns top-k approximate nearest neighbors for query_vec.

# Arguments
- `index::LMDiskANNIndex`: The index to search
- `query_vec::AbstractVector{Float32}`: The query vector
- `topk::Int=10`: Number of nearest neighbors to return

# Returns
- `Vector{Int}`: IDs of the top-k nearest neighbors

# Example
```julia
results = LMDiskANN.search(index, query_vec, topk=5)
```
"""
function search(index::LMDiskANNIndex, query_vec::AbstractVector{Float32}; topk::Int=10)
    if index.num_points == 0
        return Int[]
    end
    
    #convert query to Float32 vector if needed
    query = convert(Vector{Float32}, query_vec)
    
    # 1 gather candidates using graph search
    ef_candidates = _search_graph(index, query, max(topk, EF_SEARCH))
    
    # 2 re-rank candidates by actual distance
    dist_id_pairs = Vector{Tuple{Float32, Int}}()
    for cid in ef_candidates
        v = index.vecs[:, cid+1]
        dist = _compute_distance(v, query)
        push!(dist_id_pairs, (dist, cid))
    end
    sort!(dist_id_pairs)
    
    # 3 return top-k
    k = min(topk, length(dist_id_pairs))
    return [id+1 for (_, id) in dist_id_pairs[1:k]] #return +1 for julia 1 based index
end



"""
    _prune_neighbors(index::LMDiskANNIndex, node_id::Int, candidates::Vector{Int})

Prune the candidate neighbors for a node to maintain the GSNG property.
Returns a pruned list of neighbors.
"""
function _prune_neighbors(index::LMDiskANNIndex, node_id::Int, candidates::Vector{Int})
    if length(candidates) <= index.maxdegree
        return candidates
    end
    
    #get the vector for the node
    node_vec = index.vecs[:, node_id+1]
    
    #calculate distances to all candidates
    dist_id_pairs = Vector{Tuple{Float32, Int}}()
    for cand_id in candidates
        cand_vec = index.vecs[:, cand_id+1]
        dist = _compute_distance(node_vec, cand_vec)
        push!(dist_id_pairs, (dist, cand_id))
    end
    
    #sort by distance (closest first)
    sort!(dist_id_pairs)
    
    #take only the closest maxdegree neighbors
    return [id for (_, id) in dist_id_pairs[1:min(index.maxdegree, length(dist_id_pairs))]]
end

"""
    insert!(index::LMDiskANNIndex, new_vec::Vector{Float32})

Insert a new vector into the index. Updates the adjacency structure.
Returns the assigned ID of the newly inserted vector.
Implements Algorithm 2 from the LM-DiskANN paper.

# Arguments
- `index::LMDiskANNIndex`: The index to insert into
- `new_vec::Vector{Float32}`: The vector to insert

# Returns
- `Int`: The ID assigned to the inserted vector

# Example
```julia
id = LMDiskANN.insert!(index, vector)
```
"""
function insert!(index::LMDiskANNIndex, new_vec::Vector{Float32})
    
    # 1 determine new ID
    new_id = 0
    if !isempty(index.freelist)
        #reuse a deleted ID
        new_id = pop!(index.freelist)
    else
        #will append a new slot
        new_id = index.num_points
        index.num_points += 1
    end
    
    # 2 possibly resize the memory-mapped files if new_id >= the current allocated size
    needed_size = new_id + 1  #because 0-based ID means we need new_id+1 capacity
    current_capacity = size(index.vecs, 2)  #second dimension
    if needed_size > current_capacity
        #we must resize the .vec and .adj files
        growby = max(1024, current_capacity)  # double the size or add 1024, whichever is larger
        new_capacity = max(needed_size, current_capacity + growby)
        
        #resize the index
        resizeIndex!(index, new_capacity)
    end
    
    # 3 store new_vec in the index.vecs at column (new_id+1)
    index.vecs[:, new_id+1] = new_vec
    
    # 4 Find neighbors for new vector using search
    # if this is the first insertion, set as entry point and return
    if index.entrypoint < 0
        index.entrypoint = new_id
        _set_neighbors(index, new_id, Int[])
        saveIndex(index)
        return new_id + 1 #return 1 based for julia
    end
    
    #search for nearest neighbors to connect to
    query = convert(Vector{Float32}, new_vec)
    # nearest_neighbors = search(index, query, topk=index.maxdegree)
    nearest_neighbors_1based = search(index, query, topk=index.maxdegree)
    nearest_neighbors_0based = [nbr_id - 1 for nbr_id in nearest_neighbors_1based]
    
    # 5 set neighbors of new node
    _set_neighbors(index, new_id, nearest_neighbors_0based)
    
    # 6 for each nearest neighbor, we might add new_id to its adjacency list
    for nbr_id in nearest_neighbors_0based
        #get current neighbors of nbr_id
        nbr_neighbors = _get_neighbors(index, nbr_id)
        
        #add new_id to its neighbors
        push!(nbr_neighbors, new_id)
        
        #prune if necessary to maintain GSNG property
        pruned_neighbors = _prune_neighbors(index, nbr_id, nbr_neighbors)
        
        #update the neighbor list
        _set_neighbors(index, nbr_id, pruned_neighbors)
    end
    
    # 7 Save updated metadata
    saveIndex(index)
    
    return new_id+1 # return 1 based for julia
end

"""
    delete!(index::LMDiskANNIndex, node_id::Int)

Delete a vector (and adjacency) from the index.
Implements Algorithm 3 from the LM-DiskANN paper.

# Arguments
- `index::LMDiskANNIndex`: The index to delete from
- `node_id::Int`: The ID of the vector to delete

# Returns
- `LMDiskANNIndex`: The updated index instance

# Example
```julia
LMDiskANN.delete!(index, id)
```
"""
function delete!(index::LMDiskANNIndex, node_id::Int)
    # 1 safety checks
    node_id += -1 #make zero based for here
    if node_id < 0 || node_id >= index.num_points
        error("Invalid node_id: $node_id")
    end
    
    if node_id in index.freelist
        error("Node $node_id is already deleted")
    end
    
    # 2 get neighbors of the node to be deleted
    neighbors = _get_neighbors(index, node_id)
    
    # 3 for each neighbor, remove node_id from its adjacency list
    for nbr_id in neighbors
        nbr_neighbors = _get_neighbors(index, nbr_id)
        
        #remove node_id from neighbor's list
        filter!(x -> x != node_id, nbr_neighbors)
        
        #update neighbor's adjacency list
        _set_neighbors(index, nbr_id, nbr_neighbors)
    end
    
    # 4 clear adjacency of node_id
    _set_neighbors(index, node_id, Int[])
    
    # 5 if node_id was the entrypoint, we need a new entrypoint
    if index.entrypoint == node_id
        #find a new entry point (any non-deleted node)
        index.entrypoint = -1
        for cand in 0:(index.num_points-1)
            if cand != node_id && !(cand in index.freelist)
                index.entrypoint = cand
                break
            end
        end
    end
    
    # 6 mark node_id in freelist
    push!(index.freelist, node_id)
    
    # 7 zero out the vector data on disk
    index.vecs[:, node_id+1] .= Float32(0.0)
    
    # 8 update metadata
    saveIndex(index)
    
    return nothing
end

end # module
