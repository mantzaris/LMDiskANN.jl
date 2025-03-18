using LMDiskANN
using Test

using Random
using LinearAlgebra

Random.seed!(1)

function clean_up()
    current_dir = pwd()
    for fname in readdir(current_dir)
        if startswith(fname, "temp")
            path = joinpath(current_dir, fname)
            if isdir(path)
                rm(path; recursive=true, force=true)
            else
                rm(path; force=true)
            end
        end
    end
end

@testset "LMDiskANN Tests Basic" begin
    clean_up()

    test_dir = joinpath(dirname(@__FILE__), "temp_test_data")
    mkpath(test_dir)
    
    dim = 10
    num_vectors = 100
    
    # Generate random test vectors
    test_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    
    # Normalize vectors
    for i in 1:length(test_vectors)
        test_vectors[i] ./= norm(test_vectors[i]) + 1e-9
    end
    
    @testset "Index Creation and Loading" begin
        index_path = joinpath(test_dir, "test_index")
        index = LMDiskANN.createIndex(index_path, dim)
        
        @test index.dim == dim
        @test index.maxdegree == LMDiskANN.DEFAULT_MAX_DEGREE
        @test index.num_points == 0
        @test isempty(index.freelist)
        @test index.entrypoint == -1  # -1 internally (no points)
        
        LMDiskANN.saveIndex(index)
        loaded_index = LMDiskANN.loadIndex(index_path)
        
        @test loaded_index.dim == index.dim
        @test loaded_index.maxdegree == index.maxdegree
        @test loaded_index.num_points == index.num_points
        @test loaded_index.entrypoint == index.entrypoint
    end
    
    @testset "Insertion and Search" begin
        # Create a new index
        index_path = joinpath(test_dir, "test_index_insert")
        index = LMDiskANN.createIndex(index_path, dim)
        
        # Insert first vector
        id1 = LMDiskANN.insert!(index, test_vectors[1])
        @test id1 == 1  # 1-based ID for the first vector
        @test index.num_points == 1
        # The internal entrypoint is 0
        @test index.entrypoint == 0  # 0-based inside the struct
        # If you like, you can also check:
        #   @test (index.entrypoint + 1) == id1
        
        # Insert second vector
        id2 = LMDiskANN.insert!(index, test_vectors[2])
        @test id2 == 2  # second ID should be 2
        @test index.num_points == 2
        
        # Insert a few more vectors
        for i in 3:10
            LMDiskANN.insert!(index, test_vectors[i])
        end
        @test index.num_points == 10
        
        # Test search with a vector already in the index
        # The 5th vector => user-facing ID is 5
        results = LMDiskANN.search(index, test_vectors[5], topk=1)
        @test length(results) == 1
        @test results[1] == 5  # 1-based ID should match the original insertion order
        
        # Test search with a new random vector
        query = rand(Float32, dim)
        query ./= norm(query) + 1e-9
        
        results = LMDiskANN.search(index, query, topk=3)
        @test length(results) == 3
        
        # Verify results are sorted by distance
        distances = [norm(query - test_vectors[id]) for id in results]
        @test issorted(distances)
    end
    
    @testset "Deletion" begin
        index_path = joinpath(test_dir, "test_index_delete")
        index = LMDiskANN.createIndex(index_path, dim)
        
        # Insert vectors
        ids = [LMDiskANN.insert!(index, vec) for vec in test_vectors[1:20]]
        @test index.num_points == 20
        @test isempty(index.freelist)
        
        # Delete a vector => user ID = 5
        delete_id = 5
        LMDiskANN.delete!(index, delete_id)
        # internally the library stored (delete_id - 1) = 4 in freelist
        @test (delete_id - 1) in index.freelist
        @test index.num_points == 20  # num_points doesn't decrement
        
        # Search should not return that deleted ID
        query = test_vectors[delete_id]  # The exact vector
        results = LMDiskANN.search(index, query, topk=20)
        @test !(delete_id in results)  # 1-based ID won't appear
        
        # Insert a new vector, should reuse the deleted ID => i.e. 1-based = 5
        new_vec = rand(Float32, dim)
        new_vec ./= norm(new_vec) + 1e-9
        new_id = LMDiskANN.insert!(index, new_vec)
        @test new_id == delete_id  # i.e. 5
        @test isempty(index.freelist)
        
        # Delete the entry point => internally entrypoint is 0-based
        old_entry_1based = index.entrypoint + 1  # user-facing
        LMDiskANN.delete!(index, old_entry_1based)
        
        # The library might pick a new entrypoint
        @test index.entrypoint != -1  # some valid node or stays -1 if everything is deleted
        @test index.entrypoint != (old_entry_1based - 1)  # definitely not the same internal ID
    end
    
    @testset "Larger Scale Test" begin
        index_path = joinpath(test_dir, "test_index_large")
        index = LMDiskANN.createIndex(index_path, dim)
        
        for vec in test_vectors
            LMDiskANN.insert!(index, vec)
        end
        @test index.num_points == num_vectors
        
        # Basic multi-query check
        num_queries = 10
        k = 5
        for _ in 1:num_queries
            query = rand(Float32, dim)
            query ./= norm(query) + 1e-9
            results = LMDiskANN.search(index, query, topk=k)
            @test length(results) == k
            
            distances = [norm(query - test_vectors[id]) for id in results]
            @test issorted(distances)
        end
    end
    
    clean_up()
end


@testset "medium-Scale Basic Test (dim=100, num_vectors=10_000)" begin
    clean_up()
    
    base_path = mktempdir(prefix="temp_lm_diskann_medium_test_")
    index_path = joinpath(base_path, "test_index_medium")
    
    dim = 100
    num_vectors = 10_000
    
    test_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    for i in 1:num_vectors
        test_vectors[i] ./= norm(test_vectors[i]) + 1e-9
    end
    
    @info "making index with dim=$dim ..."
    index = LMDiskANN.createIndex(index_path, dim)
    
    @info "Inserting $num_vectors vectors ..."
    for v in test_vectors
        LMDiskANN.insert!(index, v)
    end
    @test index.num_points == num_vectors
    
    @testset "Search Checks" begin
        num_queries = 10
        top_k = 10
        for q in 1:num_queries
            query = rand(Float32, dim)
            query ./= norm(query) + 1e-9
            
            results = LMDiskANN.search(index, query, topk=top_k)
            @test length(results) == top_k
            
            # check distance ordering
            dists = [norm(query - test_vectors[id]) for id in results]
            @test issorted(dists)
        end
    end
    
    @info "Cleaning up..."
    clean_up()
end


@testset "Large-Scale Recall Test (dim=100, num_vectors=10_000)" begin
    clean_up()
    
    base_path = mktempdir(prefix="temp_lm_diskann_recall_test_")
    index_path = joinpath(base_path, "test_index_recall")
    
    dim = 100
    num_vectors = 10_000
    
    test_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    for i in 1:num_vectors
        test_vectors[i] ./= norm(test_vectors[i]) + 1e-9
    end
    
    index = LMDiskANN.createIndex(index_path, dim)
    for v in test_vectors
        LMDiskANN.insert!(index, v)
    end
    
    num_queries = 50
    top_k = 10
    
    query_ids = rand(1:num_vectors, num_queries)  # random 1-based indices
    queries = [test_vectors[qid] for qid in query_ids]
    
    function brute_force_search(vec::Vector{Float32}, all_vecs::Vector{Vector{Float32}}, k::Int)
        dist_id_pairs = [(norm(vec - all_vecs[i]), i) for i in 1:length(all_vecs)]
        sort!(dist_id_pairs, by = x->x[1])
        return [p[2] for p in dist_id_pairs[1:k]]
    end
    
    total_recall = 0.0
    
    for i in 1:num_queries
        qvec = queries[i]
        true_topk = brute_force_search(qvec, test_vectors, top_k)  # 1-based vector indices
        approx_topk_1based = LMDiskANN.search(index, qvec, topk=top_k) # also 1-based
        # measure overlap in terms of these 1-based indices
        overlap = intersect(Set(true_topk), Set(approx_topk_1based))
        recall = length(overlap) / top_k
        total_recall += recall
        
        @info "Query $i: recall = $(round(recall, digits=3))"
    end
    
    avg_recall = total_recall / num_queries
    @info "Average recall over $num_queries queries = $(round(avg_recall, digits=3))"
    
    @test round(avg_recall, digits=3) > 0.5
    clean_up()
end

@testset "Exact Match in Top-K" begin
    clean_up()
    
    dim = 100
    num_vectors = 10_000
    top_k = 10
    num_repeats = 3
    
    all_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    for i in 1:num_vectors
        all_vectors[i] ./= (norm(all_vectors[i]) + 1e-9)
    end
    
    index_path = mktempdir() * "/temp_test_index_exact_match"
    index = LMDiskANN.createIndex(index_path, dim)
    
    @info "Inserting $num_vectors vectors..."
    for v in all_vectors
        LMDiskANN.insert!(index, v)
    end
    @test index.num_points == num_vectors
    
    # pick one vector's ID (1-based)
    target_id = rand(1:num_vectors)
    target_vec = all_vectors[target_id]
    
    @testset "Check that querying the exact same vector returns its ID" begin
        for i in 1:num_repeats
            query_vec = Vector{Float32}(target_vec)
            results = LMDiskANN.search(index, query_vec, topk=top_k)
            @test target_id in results
        end
    end
    
    clean_up()
end
