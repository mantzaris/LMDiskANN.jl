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




@testset "Simple LevelDB2 Tests" begin
    clean_up()
    
    temp_dir = mktempdir(prefix="temp_leveldb_")
    forward_path = joinpath(temp_dir, "forward_db.leveldb")
    reverse_path = joinpath(temp_dir, "reverse_db.leveldb")

    dbf, dbr = open_databases(forward_path, reverse_path; create_if_missing=true)

    @testset "Insertion and Lookup" begin
        #insert a few mappings
        insert_key!(dbf, dbr, "alex", 100)
        insert_key!(dbf, dbr, "bob",   101)
        insert_key!(dbf, dbr, "cat", 300)

        # test forward lookups
        @test get_id_from_key(dbf, "alex") == 100
        @test get_id_from_key(dbf, "bob") == 101
        @test get_id_from_key(dbf, "cat") == 300

        # test reverse lookups
        @test get_key_from_id(dbr, 100) == "alex"
        @test get_key_from_id(dbr, 101) == "bob"
        @test get_key_from_id(dbr, 300) == "cat"

        # test a missing key
        @test get_id_from_key(dbf, "dan") === nothing
        @test get_key_from_id(dbr, 999) === nothing
    end

    @testset "Deletion Checks" begin
        #delete by key
        @test delete_by_key!(dbf, dbr, "bob") == true
        @test get_id_from_key(dbf, "bob") === nothing
        @test get_key_from_id(dbr, 101) === nothing

        #delete by id
        @test delete_by_id!(dbf, dbr, 300) == true
        @test get_id_from_key(dbf, "cat") === nothing
        @test get_key_from_id(dbr, 300) === nothing

        #"alex" should still exist
        @test get_id_from_key(dbf, "alex") == 100
        @test get_key_from_id(dbr, 100) == "alex"
        
        # Test deleting non-existent entries
        @test delete_by_key!(dbf, dbr, "nonexistent") == false
        @test delete_by_id!(dbf, dbr, 999) == false
    end
    
    @testset "Utility Functions" begin
        # Test count_entries
        @test count_entries(dbf) == 1
        @test count_entries(dbr) == 1  # Only "alex" should remain
        
        # Test list_all_keys
        @test "alex" in list_all_keys(dbf)
        @test length(list_all_keys(dbf)) == 1
        
        # Test clear_database!
        clear_database!(dbf)
        @test count_entries(dbf) == 0
        @test count_entries(dbr) == 1  # Only forward db was cleared
        
        # Test clear_all_databases!
        insert_key!(dbf, dbr, "test", 500)
        @test count_entries(dbf) == 1
        @test count_entries(dbr) == 2  # "100" -> "alex" still exists
        
        clear_all_databases!(dbf, dbr)
        @test count_entries(dbf) == 0
        @test count_entries(dbr) == 0
    end

    # No need to explicitly close in LevelDB2
    # Force garbage collection to release locks
    dbf = nothing
    dbr = nothing
    GC.gc()
    sleep(1) # Add a small delay to ensure locks are released

    # Create new database handles to avoid lock issues
    forward_path2 = joinpath(temp_dir, "forward_db2.leveldb")
    reverse_path2 = joinpath(temp_dir, "reverse_db2.leveldb")
    
    dbf2, dbr2 = open_databases(forward_path2, reverse_path2; create_if_missing=true)

    @testset "Persistence Checks" begin
        # Add new data
        insert_key!(dbf2, dbr2, "persistence", 600)
        @test get_id_from_key(dbf2, "persistence") == 600
        @test get_key_from_id(dbr2, 600) == "persistence"
    end

    # No need to explicitly close in LevelDB2
    # Force garbage collection to release locks
    dbf2 = nothing
    dbr2 = nothing
    GC.gc()

    clean_up()
end


# LM-DiskANN tests now



@testset "Minimal LMDiskANN Tests" begin
    clean_up()
    
    index_prefix = "temp_test_index"
    dim = 4
    index = create_index(index_prefix, dim)

    vec2 = rand(Float32, dim)
    emb1 = rand(Float32, dim)

    @testset "Index Creation" begin        
        @test index.dim == dim
        @test index.num_points == 0
        @test index.entrypoint == -1
        
        #see if the DBs are open
        @test index.id_mapping_forward !== nothing
        @test index.id_mapping_reverse !== nothing
    end
    
    @testset "Insertion and Search" begin
        vec1 = rand(Float32, 4)
        (genkey, id1) = ann_insert!(index, vec1)
        @test id1 == 1  #see first insertion => 1-based ID is 1
        @test index.num_points == 1
        
        #insert a second vector with a custom key
        vec2 = rand(Float32, 4)
        (key2, id2) = ann_insert!(index, vec2; key="my_key")
        @test id2 == 2
        @test key2 == "my_key"
        @test index.num_points == 2
        
        #basic adjacency or BFS check is trivial with only 2 vectors
        #do a quick search
        results = search(index, vec1, topk=2)
        #see something like [(maybeNothing, 1), (maybeNothingOrKey, 2)]
        @test length(results) <= 2
        
        #get/retrieve embedding
        emb1 = get_embedding_from_id(index, id1)
        @test emb1 == vec1
        
        emb2 = get_embedding_from_key(index, "my_key")
        @test emb2 == vec2
    end
    
    # deletion
    @testset "Deletion" begin
        #delete the second vector by key
        ann_delete!(index, "my_key")
        
        #now searching for it should not return it
        results = search(index, vec2, topk=2)
        #expect not to find "my_key"
        @test all(r[2] != 2 for r in results)
        
        # try to get/retrieve embedding by key => error or fail
        try
            get_embedding_from_key(index, "my_key")
            @test false
        catch e
            @test e isa ErrorException  # or KeyError
        end
        
        #first vector should still be present
        results = search(index, emb1, topk=1)
        @test length(results) == 1
        @test results[1][2] == 1
    end
    
    @info "All minimal tests passed."
    clean_up()
end




@testset "Integration Tests for LMDiskANN" begin
    clean_up()

    index_prefix = "temp_test_index_integration"
    dim = 5

    index = create_index(index_prefix, dim)
    @test index.dim == dim
    @test index.num_points == 0
    @test index.entrypoint == -1
    @test index.id_mapping_forward !== nothing
    @test index.id_mapping_reverse !== nothing
    
    num_vectors = 10

    vectors = Vector{Vector{Float32}}(undef, num_vectors)
    assigned_keys = Vector{Union{String,Nothing}}(undef, num_vectors)
    assigned_ids  = Vector{Int}(undef, num_vectors)

    for i in 1:num_vectors
        vec = rand(Float32, dim)
        vectors[i] = vec
        if isodd(i)
            # use no key, store the auto-generated
            assigned_keys[i], assigned_ids[i] = ann_insert!(index, vec)
        else
            #give it a user key
            user_key = "vec_$(i)"
            assigned_keys[i], assigned_ids[i] = ann_insert!(index, vec; key=user_key)
        end
    end
    @test index.num_points == num_vectors
    #"entrypoint" should be >= 0 now
    @test index.entrypoint >= 0
    
    #see that each inserted vector can be found via search
    @testset "Search Each Inserted Vector" begin
        for i in 1:num_vectors
            # search for the vector with topk=3
            results = search(index, vectors[i], topk=3)
            #results is a vector of (key, id) pairs, expect that the id we inserted is in those top results, since its the same vector
            #check the presence of assigned_ids[i]
            found_my_id = any(r[2] == assigned_ids[i] for r in results)
            @test found_my_id == true
        end
    end
    
    @testset "Embedding Retrieval" begin
        for i in 1:num_vectors
            #ID stored
            current_id = assigned_ids[i]
            #vector we originally inserted
            original_vec = vectors[i]
            
            retrieved_vec_id = get_embedding_from_id(index, current_id)
            @test retrieved_vec_id == original_vec
            
            #a key (i.e. even i's in this example)
            if !(assigned_keys[i] isa Nothing)
                #retrieve by key
                retrieved_vec_key = get_embedding_from_key(index, assigned_keys[i])
                @test retrieved_vec_key == original_vec
            end
        end
    end
    
    #deletion of some subset
    #delete half of them (the odd ones) by ID, and the even ones by key
    @testset "Deletion Checks" begin
        for i in 1:num_vectors
            if isodd(i)
                # delete by ID
                ann_delete!(index, assigned_ids[i])  # 1-based ID
            else
                # even => delete by key
                if !(assigned_keys[i] isa Nothing)
                    ann_delete!(index, assigned_keys[i])
                else
                    #Nothing, skip or do ID
                    ann_delete!(index, assigned_ids[i])
                end
            end
        end
        
        
        for i in 1:num_vectors
            #searching for the vector should not produce it
            results = search(index, vectors[i], topk=3)
            @test !any(r[2] == assigned_ids[i] for r in results)
            
            #try retrieving the embedding, it should error
            try
                get_embedding_from_id(index, assigned_ids[i])
                @test false  #not good here
            catch e
                #expect an error
                @test true
            end
        end
    end
    
    @info "Integration tests completed for LMDiskANN with $num_vectors vectors."
    clean_up()
end





function brute_force_topk(query_vec::Vector{Float32}, all_vecs::Vector{Vector{Float32}}, k::Int)
    #return the indices (1-based) of the top-k nearest neighbors to query_vec
    dist_id_pairs = [(norm(query_vec .- all_vecs[i]), i) for i in 1:length(all_vecs)]
    sort!(dist_id_pairs, by = x->x[1])  # ascending distance
    return [p[2] for p in dist_id_pairs[1:k]]
end

@testset "LMDiskANN Larger-Scale with Recall" begin
    clean_up()

    base_path = "temp_lmdiskann_2000vecs"
    dim = 100
    index = create_index(base_path, dim)
    
    num_vectors = 2000
    all_vectors = [rand(Float32, dim) for _ in 1:num_vectors]
    
    @info "Inserting $num_vectors vectors..."
    for i in 1:num_vectors
        ann_insert!(index, all_vectors[i])
    end
    @test index.num_points == num_vectors
    
    num_queries = 30
    query_ids = rand(1:num_vectors, num_queries)
    top_k = 10
    
    
    total_recall = 0.0
    @testset "Approximate Recall Checks" begin
        for (qi, qid) in enumerate(query_ids)
            query_vec = all_vectors[qid]
            #true top-k
            ground_truth = brute_force_topk(query_vec, all_vectors, top_k)
            
            #approximate top-k (ann search returns vector of (key, id))
            ann_results = search(index, query_vec, topk=top_k)
            
            ann_ids = [res[2] for res in ann_results]
            
            #measure recall -> fraction of overlap in the top-k sets
            overlap = intersect(Set(ground_truth), Set(ann_ids))
            recall = length(overlap) / top_k
            
            @info "Query $qi (vector $qid) => recall=$(round(recall, digits=3))"
            total_recall += recall
        end
    end
    
    avg_recall = total_recall / num_queries
    @info "Average recall over $num_queries queries = $(round(avg_recall, digits=3))"
    @test avg_recall > 0.7
    
    
    @info "Larger-scale test with $num_vectors vectors, dimension=$dim completed."
    clean_up()
end

#############################################


#do a partial/portion/full brute force on a subset of the data to compare to the ann results
function partial_brute_force_topk(
    query_vec::Vector{Float32},
    all_vecs::Vector{Vector{Float32}},
    top_k::Int;
    sample_size::Int=0
)

    #if sample_size==0 or >= length(all_vecs), we do the full set
    n = length(all_vecs)
    chosen_indices = sample_size==0 || sample_size >= n ? collect(1:n) :
                     rand(1:n, sample_size)

    dist_id_pairs = Vector{Tuple{Float32, Int}}()

    for i in chosen_indices
        d = norm(query_vec .- all_vecs[i]) #TODO: use Distances.jl
        push!(dist_id_pairs, (d, i))
    end

    sort!(dist_id_pairs, by=x->x[1])
    actual_topk = min(top_k, length(dist_id_pairs))
    return [p[2] for p in dist_id_pairs[1:actual_topk]]
end

#a test scenario function for larger sets of vectors
function run_lmdiskann_test_scenario(path_prefix::String, dim::Int, num_vectors::Int;
                                     top_k::Int=10, num_queries::Int=20, sample_size_for_brute::Int=500)

    @testset "LMDiskANN scenario: dim=$dim, n=$num_vectors" begin
        # 1 create index
        index = create_index(path_prefix, dim)
        @test index.dim == dim
        @test index.num_points == 0

        # 2 generate random vectors
        all_vectors = [rand(Float32, dim) for _ in 1:num_vectors]

        # 3 insert them with ann_insert!
        @info "Inserting $num_vectors vectors (dim=$dim)..."
        for v in all_vectors
            ann_insert!(index, v)
        end
        @test index.num_points == num_vectors

        # 4  perform a recall check
        #pick 'num_queries' random queries from the dataset and compare top-k results with a partial brute force
        query_indices = rand(1:num_vectors, num_queries)
        total_recall = 0.0

        for qidx in query_indices
            qvec = all_vectors[qidx]

            bf_topk = partial_brute_force_topk(qvec, all_vectors, top_k; sample_size=sample_size_for_brute)

            ann_results = search(index, qvec, topk=top_k)  #each is (maybeKey, ID)
            approx_ids = [r[2] for r in ann_results]  #1-based indices to all_vectors

            #find overlap with the brute force subset
            overlap = intersect(Set(bf_topk), Set(approx_ids))
            recall = length(overlap) / top_k
            total_recall += recall
        end

        avg_recall = total_recall / num_queries
        @info "Scenario dim=$dim, n=$num_vectors => average recall = $(round(avg_recall, digits=3))"

        @test avg_recall >= 0.70

    end
end

@testset "LMDiskANN Larger Tests" begin
    clean_up()
    
    #SCENARIO 1: 3,000 vectors of dimension=100
    run_lmdiskann_test_scenario("temp_scenario1", 100, 3000;
                                top_k=20, num_queries=10, sample_size_for_brute=3000)

    #SCENARIO 2: 100,000 vectors of dimension=10
    run_lmdiskann_test_scenario("temp_scenario2", 10, 10_000;
                                top_k=20, num_queries=10, sample_size_for_brute=10_000)

    clean_up()
end



@testset "LMDiskANN Parametric Type Tests" begin
    clean_up()

    @testset "Default Float32" begin
        index_prefix_32 = "temp_param_index32"
        dim = 4
        index32 = create_index(index_prefix_32, dim)

        vec1_32 = rand(Float32, dim)
        (key1, id1) = ann_insert!(index32, vec1_32)
        @test index32.num_points == 1
        @test typeof(index32.vecs) == Array{Float32,2}

        results32 = search(index32, vec1_32, topk=1)
        @test length(results32) == 1
        @test results32[1][2] == id1

        retrieved_32 = get_embedding_from_id(index32, id1)
        @test retrieved_32 == vec1_32
    end

    @testset "Float64 Param" begin
        clean_up()

        index_prefix_64 = "temp_param_index64"
        dim = 4
        index64 = create_index(index_prefix_64, dim; T=Float64)

        #check underlying array type
        @test typeof(index64.vecs) == Array{Float64,2}

        # insert a vector in Float64
        vec1_64 = rand(Float64, dim)
        (key64, id64) = ann_insert!(index64, vec1_64)
        @test index64.num_points == 1

        # insert a vector in Float16 (will be converted to Float64 internally)
        vec2_16 = rand(Float16, dim)
        (key16, id16) = ann_insert!(index64, vec2_16)
        @test index64.num_points == 2

        # search for vec1_64
        results64 = search(index64, vec1_64, topk=2)
        @test length(results64) <= 2
        found_id64 = any(r[2] == id64 for r in results64)
        @test found_id64 == true

        # get the second vector
        retrieved_2 = get_embedding_from_id(index64, id16)
        @test length(retrieved_2) == dim
        #check type
        @test typeof(retrieved_2) == Vector{Float64}

        # confirm approximate equality to original (converted from Float16 to Float64)
        for i in 1:dim
            @test isapprox(retrieved_2[i], Float64(vec2_16[i]), atol=1e-7)
        end
    end

    @testset "Float16 Param (Careful with precision)" begin
        # create an index with T=Float16
        clean_up()

        index_prefix_16 = "temp_param_index16"
        dim = 4
        index16 = create_index(index_prefix_16, dim; T=Float16)

        #check underlying array type
        @test typeof(index16.vecs) == Array{Float16,2}

        # insert a vector in Float64 => it will be converted to Float16
        vec1_64 = rand(Float64, dim)
        (k16, i16) = ann_insert!(index16, vec1_64)
        @test index16.num_points == 1

        # retrieve it
        retrieved_16 = get_embedding_from_id(index16, i16)
        @test typeof(retrieved_16) == Vector{Float16}
        @test length(retrieved_16) == dim
        #  check approximate equality within Float16 precision
        for i in 1:dim
            @test isapprox(Float64(retrieved_16[i]), vec1_64[i], atol=1e-2)
        end
    end

    clean_up()
end
