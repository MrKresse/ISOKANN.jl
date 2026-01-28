using Plots
using Statistics
using StatsPlots


"""
 coordGradient(iso::Iso,xs::AbstractMatrix)

Compute ∇χ according to the coordinates.

# Arguments
- `iso::Iso`: The isomer for which ∇χ is to be computed.
-  `xs`: A trajectory of a reaction path. e.g. xs = ISOKANN.reactionpath_minimum(iso).
# Returns
- `gradients`: ∇χ according to coordinates.
"""
function coordGradient(iso::Iso,xs::AbstractMatrix)
    num_frames = size(xs, 2)
    num_coords = size(xs,1)
    gradients = zeros(num_coords,num_frames)
    # Loop through each frame
    gradients .= hcat(map(frame_coord -> ISOKANN.dchidx(iso, frame_coord), eachcol(xs))...)
    return gradients
end


function averageGradient(iso; xs=iso.data.coords[1], nbins=100)
    gradients = coordGradient(iso, xs)  # (3N, n_frames)
    chi_vals = [iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)]

    num_coords, num_frames = size(gradients)
    num_atoms = div(num_coords, 3)

    # Reshape to (3, num_atoms, num_frames), compute squared norm per atom
    reshaped = reshape(gradients, (3, num_atoms, num_frames))
    grad_squared = sum(reshaped .^ 2, dims=1)[1, :, :]  # (num_atoms, num_frames)

    # Bin by chi_vals
    bin_edges = range(minimum(chi_vals), stop=maximum(chi_vals), length=nbins+1)
    binned = fill(NaN, num_atoms, nbins)
    bin_counts = zeros(Int, nbins)

    for frame in 1:num_frames
        χ = chi_vals[frame]
        bin = findfirst(b -> χ ≤ bin_edges[b+1], 1:nbins)
        if isnothing(bin)
            continue
        end
        bin = bin::Int
        if bin_counts[bin] == 0
            binned[:, bin] = grad_squared[:, frame]
        else
            binned[:, bin] .+= grad_squared[:, frame]
        end
        bin_counts[bin] += 1
    end

    # Normalize by count
    for b in 1:nbins
        if bin_counts[b] > 0
            binned[:, b] ./= bin_counts[b]
        end
    end

    # Plot heatmap
   bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    # Define nicely rounded tick values over χ range
    xtick_vals = 0.0:0.1:1.0
    xtick_positions = round.(Int, xtick_vals .* nbins)  # χ=0.1 → bin 10, etc.

    # Avoid out-of-bounds (χ=1.0 → 100+1 if bin_edges includes upper edge)
    xtick_positions = clamp.(xtick_positions, 1, nbins)
    # Determine y-tick spacing automatically
    if num_atoms <= 50
        ytick_positions = 1:num_atoms
    else
        step = ceil(Int, num_atoms / 50)  # show ~50 labels max
        ytick_positions = 1:step:num_atoms
    end
    ytick_labels = string.(ytick_positions)
    gr()
   

   p = heatmap(
    binned;
    xlabel="Reaction coordinate χ",
    ylabel="Atom",
    size=(1000, 1200),
    color=:viridis,
    xlabelfontsize=26,
    ylabelfontsize=26,
    legendfontsize=26,
    tickfontsize=18,
    #colorbar_title="χ-sensitivity [1/nm²]",
    colorbar_titlefontsize=26,
    left_margin=6mm,
    right_margin=12mm,
    top_margin=2mm,
    bottom_margin=2mm,
    xticks=(xtick_positions, string.(xtick_vals)),
    yticks=(ytick_positions, ytick_labels),
)

    display(p)
    return binned
end

function rankAtomImportance(binned::AbstractMatrix{<:Real}; ylabel="Atom", xlabel="Total Importance")
    num_atoms, _ = size(binned)

    # Sum over χ-bins (no normalization)
    importance = sum(binned; dims=2)[:, 1]

    # Sort in descending order
    sorted_indices = sortperm(importance; rev=true)
    sorted_importance = importance[sorted_indices]

    # Labels (e.g. Atom 1, Atom 2, …)
    atom_labels = string.(sorted_indices)

    # Bar plot: clean horizontal layout
    p= bar(
        sorted_importance;
        orientation=:h,
        yticks=(1:num_atoms, atom_labels),
        xlabel=xlabel,
        ylabel=ylabel,
        legend=false,
        bar_width=0.8,
        size=(600, 800),
        yflip=true,  # largest at top
        tickfontsize=12,
        xlabelfontsize=14,
        ylabelfontsize=14,
        title=L"Atom Importance from $\sum_z \|\partial_i \chi\|^2$"
    )
    display(p)
    return sorted_indices, sorted_importance
end


"""
 featuregradientGradient(iso::Iso,xs::AbstractMatrix)

Compute ∇χ according to the features (pairwise distances).

# Arguments
- `iso::Iso`: The isomer for which ∇χ is to be computed.
-  `xs`: A trajectory of a reaction path. e.g. xs = ISOKANN.reactionpath_minimum(iso).
# Returns
- `gradients`: ∇χ according to features.
"""
function featureGradient(iso::Iso,xs::AbstractMatrix)
    f(x) = iso.model(x)[1] # Output is a 1-element vector
    df(x) = Flux.gradient(f, x)
    features = iso.data.featurizer(xs)
    num_frames = size(features, 2)
    num_features = size(features, 1)
    gradients = zeros(num_features, num_frames)
    gradients .= hcat(map(frame_feature -> df(frame_feature)[1], eachcol(features))...)
    return gradients
end
