
using LevelDB2

###############################################################################
# ppen and closing the level DBs
###############################################################################

"""
    open_databases(forward_path::String, reverse_path::String; create_if_missing=true)

Opens (or creates) two LevelDB databases:
 - db_forward: string_key -> string_of_internal_id
 - db_reverse: string_of_internal_id -> string_key

Returns a tuple (db_forward, db_reverse).
"""
function open_databases(forward_path::String, reverse_path::String; create_if_missing=true)
    db_forward = DB(forward_path; create_if_missing=create_if_missing)
    db_reverse = DB(reverse_path; create_if_missing=create_if_missing)
    return (db_forward, db_reverse)
end

"""
    close_databases(db_forward, db_reverse)

Closes both databases.
"""
function close_databases(db_forward, db_reverse)
    close(db_forward)
    close(db_reverse)
end

###############################################################################
#insert
###############################################################################

"""
    insert_key!(db_forward, db_reverse, user_key::String, internal_id::Int)

Inserts `user_key -> internal_id` into `db_forward`,
and `internal_id -> user_key` into `db_reverse`.

Both must be done to keep them in sync.
"""
function insert_key!(db_forward, db_reverse, user_key::String, internal_id::Int)
    #convert the internal_id to a string, so it can be stored as bytes
    id_str = string(internal_id)

    LevelDB2.put!(db_forward, id_str, user_key)
    LevelDB2.put!(db_reverse, user_key, id_str)
end

###############################################################################
# lookups
###############################################################################

"""
    get_id_from_key(db_forward, user_key::String) -> Union{Int, Nothing}

Fetches the internal_id associated with `user_key` from `db_forward`.
Returns `nothing` if not found.
"""
function get_id_from_key(db_forward, user_key::String)::Union{Int, Nothing}
    try
        # fetch(db, key) in LevelDB2
        val = LevelDB2.fetch(db_forward, user_key)
        return parse(Int, val)
    catch e
        if isa(e, KeyError)
            return nothing
        else
            rethrow(e)
        end
    end

    # val = get(db_forward, user_key)
    # return val === nothing ? nothing : parse(Int, String(val))
end

"""
    get_key_from_id(db_reverse, internal_id::Int) -> Union{String, Nothing}

Fetches the user_key from `db_reverse` given the internal_id.
Returns `nothing` if not found.
"""
function get_key_from_id(db_reverse, internal_id::Int)::Union{String, Nothing}
    try
        # fetch(db, key) in LevelDB2
        val = LevelDB2.fetch(db_reverse, string(internal_id))
        return val
    catch e
        if isa(e, KeyError)
            return nothing
        else
            rethrow(e)
        end
    end

    # val = get(db_reverse, string(internal_id))
    # return val === nothing ? nothing : String(val)
end

###############################################################################
# deletions
###############################################################################

"""
    delete_by_key!(db_forward, db_reverse, user_key::String)

Deletes the entry for `user_key` in `db_forward`,
and also deletes the corresponding entry in `db_reverse`.
"""
function delete_by_key!(db_forward, db_reverse, user_key::String)

    id = get_id_from_key(db_forward, user_key)
    if id === nothing
        return false
    end
    
    #delete from db_reverse (the reverse side)
    LevelDB2.del!(db_reverse, string(id))
    
    #delete from db_forward
    LevelDB2.del!(db_forward, user_key)
    
    return true
end

"""
    delete_by_id!(db_forward, db_reverse, internal_id::Int)

Deletes the entry for `internal_id` in `db_reverse`,
and also deletes the corresponding entry in `db_forward`.
"""
function delete_by_id!(db_forward, db_reverse, internal_id::Int)

    k = get_key_from_id(db_reverse, internal_id)
    if k === nothing
        return false
    end
    
    LevelDB2.del!(db_forward, k)
    
    LevelDB2.del!(db_reverse, string(internal_id))
    
    return true
end

function clear_database!(db)
    for (k, _) in db
        LevelDB2.del!(db, k)
    end
end

"""
    clear_all_databases!(db_forward, db_reverse)

Removes all entries from both forward and reverse databases.

# Arguments
- `db_forward`: Forward mapping database handle
- `db_reverse`: Reverse mapping database handle
"""
function clear_all_databases!(db_forward, db_reverse)
    clear_database!(db_forward)
    clear_database!(db_reverse)
end

function count_entries(db)::Int
    count = 0
    for _ in db
        count += 1
    end
    return count
end

"""
    list_all_keys(db) -> Vector{String}

Returns a list of all keys in the database as strings.

# Arguments
- `db`: LevelDB database handle

# Returns
- `Vector{String}`: List of all keys in the database
"""
function list_all_keys(db)::Vector{String}
    keys = String[]
    for (k, _) in db
        push!(keys, k)
    end
    return keys
end