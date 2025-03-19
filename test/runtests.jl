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


