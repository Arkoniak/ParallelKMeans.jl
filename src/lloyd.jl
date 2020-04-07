"""
    Lloyd <: AbstractKMeansAlg

Basic algorithm for k-means calculation.
"""
struct Lloyd <: AbstractKMeansAlg end

kmeans(design_matrix, k;
    n_threads = Threads.nthreads(),
    k_init = "k-means++", max_iters = 300, tol = 1e-6,
    verbose = false, init = nothing) =
        kmeans(Lloyd(), design_matrix, k; n_threads = n_threads, k_init = k_init, max_iters = max_iters, tol = tol,
            verbose = verbose, init = init)

"""
    create_containers(::Lloyd, k, nrow, ncol, n_threads)

Internal function for the creation of all necessary intermidiate structures.

- `new_centroids` - container which holds new positions of centroids
- `centroids_cnt` - container which holds number of points for each centroid
- `labels` - vector which holds labels of corresponding points
"""
function create_containers(::Lloyd, k, nrow, ncol, n_threads)
    new_centroids = Vector{Array{Float64, 2}}(undef, n_threads)
    centroids_cnt = Vector{Vector{Int}}(undef, n_threads)

    for i in 1:n_threads
        new_centroids[i] = Array{Float64, 2}(undef, nrow, k)
        centroids_cnt[i] = Vector{Int}(undef, k)
    end

    labels = Vector{Int}(undef, ncol)

    return (new_centroids = new_centroids, centroids_cnt = centroids_cnt,
            labels = labels)
end

update_containers!(containers, ::Lloyd, centroids, n_threads) = nothing

function chunk_update_centroids!(centroids, containers, ::Lloyd,
    design_matrix, r, idx)

    # unpack containers for easier manipulations
    new_centroids = containers.new_centroids[idx]
    centroids_cnt = containers.centroids_cnt[idx]
    labels = containers.labels

    new_centroids .= 0.0
    centroids_cnt .= 0
    J = 0.0
    @inbounds for i in r
        min_distance = Inf
        label = 1
        for k in axes(centroids, 2)
            dist = distance(design_matrix, centroids, i, k)
            label = min_distance > dist ? k : label
            min_distance = min_distance > dist ? dist : min_distance
        end
        labels[i] = label
        centroids_cnt[label] += 1
        for j in axes(design_matrix, 1)
            new_centroids[j, label] += design_matrix[j, i]
        end
        J += min_distance
    end

    return J
end
