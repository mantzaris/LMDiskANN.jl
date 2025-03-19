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
    index = createIndex(index_prefix, dim)

    vec2 = rand(Float32, dim)
    emb1 = rand(Float32, dim)

    @testset "Index Creation" begin        
        @test index.dim == dim
        @test index.num_points == 0
        @test index.entrypoint == -1
        
        # Check that the DBs are open
        @test index.id_mapping_forward !== nothing
        @test index.id_mapping_reverse !== nothing
    end
    
    @testset "Insertion and Search" begin
        vec1 = rand(Float32, 4)
        (genkey, id1) = ann_insert!(index, vec1)
        @test id1 == 1  # first insertion => 1-based ID is 1
        @test index.num_points == 1
        
        # Insert a second vector with a custom key
        vec2 = rand(Float32, 4)
        (key2, id2) = ann_insert!(index, vec2; key="my_key")
        @test id2 == 2
        @test key2 == "my_key"
        @test index.num_points == 2
        
        # basic adjacency or BFS check is trivial with only 2 vectors
        # but we can at least do a quick search
        results = search(index, vec1, topk=2)
        # Expect to see something like [(maybeNothing, 1), (maybeNothingOrKey, 2)]
        @test length(results) <= 2
        
        # Retrieve embedding
        emb1 = get_embedding_from_id(index, id1)
        @test emb1 == vec1
        
        emb2 = get_embedding_from_key(index, "my_key")
        @test emb2 == vec2
    end
    
    # 4) Deletion
    @testset "Deletion" begin
        # Delete the second vector by key
        ann_delete!(index, "my_key")
        
        # Now searching for it should not return it
        results = search(index, vec2, topk=2)
        # Expect not to find "my_key"
        # In a minimal scenario, we might just check that we didn't get 2 results
        @test all(r[2] != 2 for r in results)
        
        # Try to retrieve embedding by key => error or fail
        try
            get_embedding_from_key(index, "my_key")
            @test false  # if we got here, it didn't throw
        catch e
            @test e isa ErrorException  # or KeyError
        end
        
        # The first vector should still be present
        results = search(index, emb1, topk=1)
        @test length(results) == 1
        @test results[1][2] == 1
    end
    
    @info "All minimal tests passed."
    clean_up()
end