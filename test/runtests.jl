using LMDiskANN
using Test

using Random
using LinearAlgebra

Random.seed!(1)

@testset "LMDiskANN.jl" begin
    @test 1 == 1
end


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
