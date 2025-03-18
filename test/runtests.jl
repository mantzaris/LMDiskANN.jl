using LMDiskANN
using Test

using Random
using LinearAlgebra

Random.seed!(1)



function clean_up()
    #clean up
    current_dir = pwd()

    for fname in readdir(current_dir)
        if startswith(fname, "temp")
            path = joinpath(current_dir, fname)
            #if a directory remove recursively or its a file remove it
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
    
    #generate random test vectors
    test_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    
    #normalize vectors
    for i in 1:length(test_vectors)
        test_vectors[i] = test_vectors[i] ./ norm(test_vectors[i])
    end
    
    @testset "Index Creation and Loading" begin
        #create a new index
        index_path = joinpath(test_dir, "test_index")
        index = LMDiskANN.createIndex(index_path, dim)
        
        #test basic properties
        @test index.dim == dim
        @test index.maxdegree == LMDiskANN.DEFAULT_MAX_DEGREE
        @test index.num_points == 0
        @test isempty(index.freelist)
        @test index.entrypoint == -1
        
        #save and reload the index
        LMDiskANN.saveIndex(index)
        loaded_index = LMDiskANN.loadIndex(index_path)
        
        #test that loaded index matches original
        @test loaded_index.dim == index.dim
        @test loaded_index.maxdegree == index.maxdegree
        @test loaded_index.num_points == index.num_points
        @test loaded_index.entrypoint == index.entrypoint
    end
    
    @testset "Insertion and Search" begin
        #create a new index
        index_path = joinpath(test_dir, "test_index_insert")
        index = LMDiskANN.createIndex(index_path, dim)
        
        #insert first vector
        id1 = LMDiskANN.insert!(index, test_vectors[1])
        @test id1 == 0  #first ID should be 0
        @test index.num_points == 1
        @test index.entrypoint == 0
        
        #insert second vector
        id2 = LMDiskANN.insert!(index, test_vectors[2])
        @test id2 == 1  #second ID should be 1
        @test index.num_points == 2
        
        #insert a few more vectors
        for i in 3:10
            LMDiskANN.insert!(index, test_vectors[i])
        end
        @test index.num_points == 10
        
        #test search with a vector already in the index
        results = LMDiskANN.search(index, test_vectors[5], topk=1)
        @test length(results) == 1
        @test results[1] == 4  # ID is 0-based, so vector 5 has ID 4
        
        #test search with a new vector (should find nearest)
        query = rand(Float32, dim)
        query = query ./ norm(query)
        
        results = LMDiskANN.search(index, query, topk=3)
        @test length(results) == 3
        
        #verify results are reasonable by checking distances
        distances = [norm(query - test_vectors[id+1]) for id in results]
        @test issorted(distances)  #results should be sorted by distance
    end
    
    @testset "Deletion" begin
        #create a new index
        index_path = joinpath(test_dir, "test_index_delete")
        index = LMDiskANN.createIndex(index_path, dim)
        
        #insert vectors
        ids = [LMDiskANN.insert!(index, vec) for vec in test_vectors[1:20]]
        @test index.num_points == 20
        @test isempty(index.freelist)
        
        #delete a vector
        delete_id = 5
        LMDiskANN.delete!(index, delete_id)
        @test delete_id in index.freelist
        @test index.num_points == 20  # num_points doesn't change
        
        #search should not return the deleted vector
        query = test_vectors[delete_id+1]  # The exact vector we deleted
        results = LMDiskANN.search(index, query, topk=20)
        @test !(delete_id in results)
        
        #insert a new vector, should reuse the deleted ID
        new_vec = rand(Float32, dim)
        new_vec = new_vec ./ norm(new_vec)
        new_id = LMDiskANN.insert!(index, new_vec)
        @test new_id == delete_id
        @test isempty(index.freelist)
        
        #delete the entry point and verify a new one is selected
        old_entry = index.entrypoint
        LMDiskANN.delete!(index, old_entry)
        @test index.entrypoint != old_entry
        @test index.entrypoint >= 0  # a valid entry point is selected
    end
    
    @testset "Larger Scale Test" begin
        # make a new index
        index_path = joinpath(test_dir, "test_index_large")
        index = LMDiskANN.createIndex(index_path, dim)
        
        #insert all test vectors
        for vec in test_vectors
            LMDiskANN.insert!(index, vec)
        end
        @test index.num_points == num_vectors
        
        #  search with multiple queries
        num_queries = 10
        k = 5
        for _ in 1:num_queries
            query = rand(Float32, dim)
            query = query ./ norm(query)
            
            results = LMDiskANN.search(index, query, topk=k)
            @test length(results) == k
            
            #verify results are ok
            distances = [norm(query - test_vectors[id+1]) for id in results]
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
    
    #make new index
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
            
            #check distance ordering
            dists = [norm(query .- test_vectors[id+1]) for id in results]
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
    
    query_ids = rand(1:num_vectors, num_queries)  #random indices
    queries = [test_vectors[qid] for qid in query_ids]

    function brute_force_search(vec::Vector{Float32}, all_vecs::Vector{Vector{Float32}}, k::Int)
        #return the indices of the top-k nearest neighbors in all_vecs
        dist_id_pairs = [(norm(vec .- all_vecs[i]), i) for i in 1:length(all_vecs)]
        sort!(dist_id_pairs, by = x->x[1])  #sort by distance ascending
        return [p[2] for p in dist_id_pairs[1:k]]  #top-k indices
    end
    
    total_recall = 0.0
    
    for i in 1:num_queries
        qvec = queries[i]
        true_topk = brute_force_search(qvec, test_vectors, top_k)
        approx_topk = LMDiskANN.search(index, qvec, topk=top_k)
        
        #convert approx_topk from 0-based IDs to 1-based for matching with `true_topk`
        approx_topk_1based = [id+1 for id in approx_topk]
        
        #measure overlap
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
    num_repeats = 3  #query the same vector multiple times
    
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
    
    #pick one vector's ID at random to test
    target_id = rand(0:(num_vectors-1))  # 0-based ID
    target_vec = all_vectors[target_id+1]
    

    @testset "Check that querying the exact same vector returns its ID" begin
        for i in 1:num_repeats
            query_vec = Vector{Float32}(target_vec)
            results = LMDiskANN.search(index, query_vec, topk=top_k)
            
            @test target_id in results
        end
    end
    
    clean_up()
end
