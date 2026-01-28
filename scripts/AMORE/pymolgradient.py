import pymol
from pymol import cmd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import matplotlib.cm as cm

def showFramewiseGradient(gradient_path, path_pdb,cmap = 'viridis', style='sticks'):
    """
    Maps the colormap cmap of the framewise gradient matrix onto the molecule in path_pdb.
    """
    gradient_matrix=loadGradient(gradient_path)
    cmd.load(path_pdb)
    cmd.hide('everything')
    cmd.remove('solvent')
    cmd.remove('resn POP')
    cmd.show(style)
    cmd.bg_color('white')
    # Convert gradient matrix to numpy array
    gradient_matrix = np.array(gradient_matrix)
    num_atoms, num_frames = gradient_matrix.shape
    gradient_matrix = (gradient_matrix - gradient_matrix.min()) \
            / (gradient_matrix.max() - gradient_matrix.min())
    # Get the inferno colormap
    cmap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize between 0 and 1
    
    # Set up the movie timeline with 'mset'
    # This creates a timeline of frames from 1 to num_frames, each corresponding to a state
    cmd.mset(f"1 x{num_frames}")

    # Loop over each frame and define the color changes for the frame
    for frame in range(num_frames):
        # Prepare a list of commands for this frame
        frame_cmds = []
        
        gradient_frame = gradient_matrix[:, frame]
        #gradient_normalized = (gradient_frame - gradient_frame.min()) \
        #    / (gradient_frame.max() - gradient_frame.min())
        gradient_normalized = gradient_frame
        # Ensure PyMOL updates the molecular state (i.e., positions) for each frame
        frame_cmds.append(f"frame {frame + 1}; set state, {frame + 1}")
        # Loop over each atom and define color changes
        for atom_index in range(num_atoms):
            if gradient_frame[atom_index]==0:
                continue

            atom_id = atom_index + 1  #
            norm_value = gradient_normalized[atom_index]
            
            # Map the normalized value to a color using the inferno colormap
            color_rgba = cmap(norm(norm_value))
            color_hex = mcolors.rgb2hex(color_rgba[:3])  # Convert RGBA to HEX
            
            # Convert HEX to RGB for PyMOL
            color_rgb = tuple(int(x * 255) for x in mcolors.hex2color(color_hex))
            color_name = f"dynamic_color_{atom_id}_frame_{frame}"
            
            # Define the color for this atom and frame
            cmd.set_color(color_name, color_rgb)
            
            # Prepare the color command for this frame
            frame_cmds.append(f"color {color_name}, id {atom_id}")

        # Combine all commands for this frame into a single mdo command
        cmd.mdo(frame + 1, "; ".join(frame_cmds))

    # Start the movie playback
    #cmd.mplay()

def showAverageGradient(gradient_path, path_pdb, cmap='viridis', style='cartoon'):
    """
    Averaged version of showFramewiseGradient:
    - Normalize the full gradient matrix to [0,1] (global, like the framewise function)
    - Take the mean across frames
    - Color protein residues via their Cα value (byres)
    - Color ligand (resn LIG) atoms individually
    - Skip all other atoms for speed
    """
    # Clean up previous runs (optional but helps avoid clutter)
    try:
        cmd.delete("gradcolor_*")
        cmd.delete("ca_atom_*")
    except:
        pass

    obj = "mol"
    cmd.load(path_pdb, obj)
    cmd.hide('everything', obj)
    cmd.delete('solvent')  # harmless if none
    cmd.bg_color('white')

    # Load gradient and normalize like the framewise function
    G = np.array(loadGradient(gradient_path))   # shape: (n_atoms, n_frames)
    n_atoms, n_frames = G.shape
    G_norm = (G - G.min()) / (G.max() - G.min() + 1e-12)  # global normalization
    grad_avg = G_norm.mean(axis=1)                        # average across frames

    # Color map
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Iterate over atoms in PDB order; use id = index+1 (matches your framewise code)
    for atom_index, val in enumerate(grad_avg):
        atom_id = atom_index + 1

        # Only act on Cα (color whole residue) OR ligand atoms (color atom)
        is_ca = cmd.count_atoms(f"({obj}) and id {atom_id} and name CA") > 0
        is_lig = cmd.count_atoms(f"({obj}) and id {atom_id} and resn LIG") > 0
        if not (is_ca or is_lig):
            continue

        # Map value to RGB
        rgba = cmap_obj(norm(val))
        hexcol = mcolors.rgb2hex(rgba[:3])
        rgb255 = tuple(int(x * 255) for x in mcolors.hex2color(hexcol))
        cname = f"gradcolor_{atom_id}"
        cmd.set_color(cname, rgb255)

        if is_ca:
            # color the entire residue by Cα value
            sel = f"({obj}) and id {atom_id}"
            cmd.select(f"ca_atom_{atom_id}", sel)
            cmd.color(cname, f"byres {sel}")
        else:
            # ligand atom: color atom itself
            cmd.color(cname, f"({obj}) and id {atom_id}")

    cmd.show('cartoon', f"{obj}")
    cmd.show('sticks', f"{obj} and resn LIG")

def showGradient_atoms_nonzero_first(path_gradient, path_pdb, cmap='viridis',
                                     style='cartoon', obj='mol',
                                     base_color='grey70'):
    """
    Colors ALL residues/atoms to a base color, then colors only NONZERO-gradient atoms/residues
    using a matplotlib colormap normalized over NONZERO values.
    Uses PDB serial `id` mapping: gradient[i] -> id (i+1).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import cm

    cmd.delete(obj)
    cmd.load(path_pdb, obj)
    cmd.hide('everything', obj)
    cmd.show(style, obj)
    cmd.bg_color('white')

    # --- load gradient, force 1D ---
    G = np.array(loadGradient(path_gradient), dtype=float).squeeze()
    if G.ndim != 1:
        raise ValueError(f"Gradient must be 1D after squeeze; got shape {G.shape}")

    n_atoms = int(G.shape[0])

    # --- diagnostics: are you basically all zeros? ---
    nz_mask = (G != 0.0)
    nz = int(np.count_nonzero(nz_mask))
    print(f"Gradient: n={n_atoms}, nonzero={nz} ({100*nz/n_atoms:.3f}%), min={G.min():.3g}, max={G.max():.3g}")

    # --- paint everything a base color first so changes are visible ---
    cmd.color(base_color, obj)

    # nothing to highlight?
    if nz == 0:
        print("All gradients are zero; nothing to highlight.")
        return

    # normalize over NONZERO values only
    nz_vals = G[nz_mask]
    vmin, vmax = float(nz_vals.min()), float(nz_vals.max())
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    # color cache so we don't create 70k unique colors
    rgb_cache = {}

    cmd.set("suspend_updates", 1)
    try:
        # iterate only nonzeros
        for atom_index in np.flatnonzero(nz_mask):
            atom_id = int(atom_index) + 1  # id is 1-based

            val = float(G[atom_index])
            rgba = cmap_obj(norm(val))
            rgb255 = tuple(int(x * 255) for x in rgba[:3])

            cname = rgb_cache.get(rgb255)
            if cname is None:
                cname = f"grad_{rgb255[0]}_{rgb255[1]}_{rgb255[2]}"
                cmd.set_color(cname, rgb255)
                rgb_cache[rgb255] = cname

            # if you want per-atom coloring: use "{obj} and id {atom_id}"
            # if you want residue coloring: use byres(...)
            cmd.color(cname, f"byres ({obj} and id {atom_id})")

        cmd.rebuild()
    finally:
        cmd.set("suspend_updates", 0)
        cmd.refresh()




def _register_matplotlib_cmap(cmap_name, n=20, prefix="mpl"):
    """Sample a matplotlib colormap and register the colors in PyMOL."""
    cmap = cm.get_cmap(cmap_name, n)
    names = []
    for i in range(n):
        rgba = cmap(i)
        rgb255 = tuple(int(x * 255) for x in rgba[:3])
        cname = f"{prefix}_{cmap_name}_{i}"
        cmd.set_color(cname, rgb255)
        names.append(cname)
    return names

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def showAverageGradient_CA(gradient_path, path_pdb, cmap='viridis',
                                style='cartoon', mode='ca', ignore_zeros=True):
    """
    Average gradient across frames and color:
      - protein residues by their Cα (color whole residue with 'byres')
      - optionally ligand atoms individually (mode='all')
    Mapping is by PDB serial **id** (row i -> id i+1), same as your framewise function.

    Parameters
    ----------
    gradient_path : str         # gradient matrix (n_atoms x n_frames)
    path_pdb      : str         # PDB file
    cmap          : str         # matplotlib colormap name ('viridis','inferno',...)
    style         : str         # PyMOL drawing style for object (e.g. 'sticks')
    mode          : {'all','ca'}  # 'all' colors CA + LIG; 'ca' only CA
    ignore_zeros  : bool        # if True, zero rows are treated as missing when computing averages
    """

    obj = "mol"
    cmd.delete(obj)
    cmd.load(path_pdb, obj)
    cmd.hide('everything', obj)
    cmd.show(style, obj)
    cmd.show('cartoon', f"{obj} and polymer.protein")
    if mode == "all":
        cmd.show('sticks', f"{obj} and resn LIG")
    cmd.bg_color('white')

    # 1) Load gradients and average across frames
    G = np.array(loadGradient(gradient_path))  # shape: (n_atoms, n_frames)
    G = np.array(loadGradient(gradient_path))

    if G.ndim == 1:
        # Already an (n_atoms,) vector
        gavg = G.astype(float)
        if ignore_zeros:
            gavg = gavg.copy()
            gavg[gavg == 0.0] = np.nan
            gavg = np.nan_to_num(gavg, nan=0.0)
    else:
        # (n_atoms, n_frames), do the usual averaging
        if ignore_zeros:
            G = G.astype(float)
            G[G == 0.0] = np.nan
            with np.errstate(invalid='ignore', divide='ignore'):
                gavg = np.nanmean(G, axis=1)
            gavg = np.nan_to_num(gavg, nan=0.0)
        else:
            gavg = G.mean(axis=1)

    n_atoms = gavg.shape[0]
    cmap_obj = cm.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=None, vmax=None)  # we'll set limits from subset

    # 2) Build per-residue value from all atoms: "only nonzero value per residue"
    mdl = cmd.get_model(obj)

    # residue key = (chain, resi, resn, segi)
    atoms_by_res = {}
    for a in mdl.atom:
        if not (1 <= a.id <= n_atoms):
            continue
        key = (a.chain, a.resi, a.resn, a.segi)
        val = float(gavg[a.id - 1])
        atoms_by_res.setdefault(key, []).append(val)

    # For each residue: use the *unique* nonzero value.
    # If no nonzero values -> 0.0
    # If multiple nonzero values -> pick the one with largest absolute magnitude.
    res_grad = {}
    for key, vals in atoms_by_res.items():
        nz = [v for v in vals if v != 0.0]
        if not nz:
            res_val = 0.0
        elif len(nz) == 1:
            res_val = nz[0]
        else:
            # multiple nonzero values: choose the dominant one
            res_val = max(nz, key=abs)
        res_grad[key] = res_val

    # 3) Collect targets by **PDB serial id** but *use per-residue* value
    targets = []  # list of (id_serial, value)
    for a in mdl.atom:
        is_ca  = (a.name == "CA" and a.resn != "LIG")
        is_lig = (mode == "all" and a.resn == "LIG")
        if (is_ca or is_lig) and 1 <= a.id <= n_atoms:
            key = (a.chain, a.resi, a.resn, a.segi)
            val = res_grad.get(key, 0.0)
            targets.append((a.id, float(val)))

    if not targets:
        print("No target atoms (CA and/or LIG) found; nothing to color.")
        return

    # 4) Normalize ONLY over the values we will color (better dynamic range)
    vals = np.array([v for _, v in targets], dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1e-6  # avoid flat colormap
    norm.vmin, norm.vmax = vmin, vmax

    # 5) Color — one command per target; residues via 'byres (id ..)', ligand per atom
    cmd.set("suspend_updates", 1)
    try:
        for aid, val in targets:

            # --- NEW: exact zeros → grey ---
            if val == 0.0:
                grey = (180, 180, 180)       # choose your grey, e.g. (150,150,150)
                cname = f"grad_zero_{aid}"
                cmd.set_color(cname, grey)
                if cmd.count_atoms(f"{obj} and id {aid} and name CA"):
                    cmd.color(cname, f"byres ({obj} and id {aid})")
                else:
                    cmd.color(cname, f"{obj} and id {aid}")
                continue
            # --------------------------------

            # normal colored values
            rgba = cmap_obj(norm(val))
            rgb255 = tuple(int(x * 255) for x in rgba[:3])
            cname = f"grad_{aid}"
            cmd.set_color(cname, rgb255)
            if cmd.count_atoms(f"{obj} and id {aid} and name CA"):
                cmd.color(cname, f"byres ({obj} and id {aid})")
            else:
                cmd.color(cname, f"{obj} and id {aid}")
        cmd.rebuild()
    finally:
        cmd.set("suspend_updates", 0)
        cmd.refresh()

def labelAlphaCarbons(size, start_residue, end_residue):
    """
    Labels alpha carbons (Cα) for a specified range of amino acids.
    
    :param size: Size of the label font
    :param start_residue: The starting residue number
    :param end_residue: The ending residue number
    """
    cmd.set("label_size", size)
    
    # Iterate over the specified residue range
    for i in range(start_residue, end_residue + 1):
        # Select the Cα atom for the current amino acid
        cmd.select('temp_atom', f'resi {i} and name CA')
        
        # Label the selected atom with its residue number
        cmd.label('temp_atom', f'"{i}"')  

    # Clean up the temporary selection
    cmd.delete('temp_atom')

def labelMOR():
    labelAlphaCarbons(18,265,265)
    labelAlphaCarbons(18,328,329)
    labelAlphaCarbons(18,154,154)
    labelAlphaCarbons(18,150,150)
    labelAlphaCarbons(18,114,114)
    labelAlphaCarbons(18,326,326)
    labelAlphaCarbons(18,124,124)
    labelAlphaCarbons(18,147,151)
    labelAlphaCarbons(18,292,295)

def labelAtoms(size):
    """
    Labels all atoms by their ID.
    """
    cmd.set("label_size", size)
    # Get the number of atoms in the current selection
    num_atoms = cmd.count_atoms('all')
    
    # Iterate over all atom indices
    for i in range(1, num_atoms):
        # Select the atom by ID
        cmd.select('temp_atom', f'id {i}')
        
        # Label the selected atom with its ID using a proper string
        cmd.label('temp_atom', f'"{i}"')  

    # Clean up the temporary selection
    cmd.delete('temp_atom')

def showAtomFeature(path_coord,path_feature,path_pdb):
    gradient_matrix=loadGradient(path_feature)
    # Load PDB file and set initial view
    cmd.load(path_pdb)
    cmd.hide('everything')
    cmd.delete('solvent')
    cmd.show('sticks')

    # Convert the gradient matrix to a numpy array if it's not already
    gradient_matrix = np.array(gradient_matrix)
    num_features, num_frames = gradient_matrix.shape
    num_atoms = round(0.5 + np.sqrt(0.25+2*num_features)) 

    # Create the lookup table (lut) for feature indexing
    lut = []
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            lut.append((i + 1, j + 1))  # PyMOL uses 1-based indexing

    coord_matrix = loadGradient(path_coord)
    #extract maximum atoms for each frame, coordinate based
    max_indices = np.argmax(coord_matrix,axis=0)
    #extract features of that atom
    max_feature_indices = np.zeros(num_frames)
    for frame in range(num_frames):
        atom_index=max_indices[frame]+1
        matching_indices = [index for index, (first, second) in enumerate(lut) \
                            if first == atom_index or second == atom_index]
        max_feature_indices[frame] = np.argmax(gradient_matrix[matching_indices,frame])

    # Set up the movie timeline with 'mset'
    cmd.mset(f"1 x{num_frames}")
    # Loop over each frame and define the dashed lines for the frame
    for frame in range(num_frames):
        feature_idx = max_feature_indices[frame]
        frame_cmds = []

        # Set the frame and state for each frame
        frame_cmds.append(f"frame {frame + 1}; set state, {frame + 1}")
        
        # Remove any existing dashed lines from the previous frame
        frame_cmds.append("delete all_dashed_lines")

        # Generate dashed lines for each of the top features
        atom1, atom2 = lut[int(feature_idx)]
        # Create a unique name for each dashed line based on frame, atom1, and atom2
        dash_name = f"dashed_line_{frame + 1}_{atom1}_{atom2}"
        frame_cmds.append(f"distance {dash_name}, id {atom1}, id {atom2}")
        frame_cmds.append(f"set dash_gap, 0.5, {dash_name}")
        frame_cmds.append(f"set dash_length, 0.2, {dash_name}")
        frame_cmds.append(f"color blue, {dash_name}")

        # Store dashed lines in a group to facilitate deletion later
        frame_cmds.append(f"group all_dashed_lines, {dash_name}")

        # Combine all commands for this frame into a single mdo command
        cmd.mdo(frame + 1, "; ".join(frame_cmds))

def writeResidueGradients(gradient_path, path_pdb, output_path, obj="mol",
                          ignore_zeros=True, sort_by="abs", descending=True):
    """
    Build a per-residue gradient table and write it to a human-readable file.

    Parameters
    ----------
    gradient_path : str
        Path to gradient matrix (n_atoms x n_frames or n_atoms).
    path_pdb : str
        PDB file corresponding to `obj`.
    output_path : str
        File path to write the table to (e.g. 'gradients_table.txt').
    obj : str
        PyMOL object name.
    ignore_zeros : bool
        If True, residues with grad==0 are omitted.
    sort_by : {'abs','value','resi'}
        Sorting mode.
    descending : bool
        Sort order for abs/value.
    """

    # --- 1) Load and average gradients exactly like before ---
    G = np.array(loadGradient(gradient_path))

    if G.ndim == 1:
        gavg = G.astype(float)
        if ignore_zeros:
            gavg = gavg.copy()
            gavg[gavg == 0.0] = np.nan
            gavg = np.nan_to_num(gavg, nan=0.0)
    else:
        if ignore_zeros:
            G = G.astype(float)
            G[G == 0.0] = np.nan
            with np.errstate(invalid='ignore', divide='ignore'):
                gavg = np.nanmean(G, axis=1)
            gavg = np.nan_to_num(gavg, nan=0.0)
        else:
            gavg = G.mean(axis=1)

    n_atoms = gavg.shape[0]

    # --- 2) Per-residue aggregation (same rules as your coloring function) ---
    mdl = cmd.get_model(obj)
    atoms_by_res = {}  # key = (chain, resi, resn, segi)

    for a in mdl.atom:
        if not (1 <= a.id <= n_atoms):
            continue
        key = (a.chain, a.resi, a.resn, a.segi)
        atoms_by_res.setdefault(key, []).append(float(gavg[a.id - 1]))

    res_grad = {}
    for key, vals in atoms_by_res.items():
        nz = [v for v in vals if v != 0.0]
        if not nz:
            res_val = 0.0
        elif len(nz) == 1:
            res_val = nz[0]
        else:
            res_val = max(nz, key=abs)   # dominant value
        res_grad[key] = float(res_val)

    # --- 3) Convert to list of rows ---
    rows = []
    for (chain, resi, resn, segi), val in res_grad.items():
        if ignore_zeros and val == 0.0:
            continue
        try:
            resi_int = int(resi)
        except ValueError:
            resi_int = resi
        rows.append({
            "chain": chain or "-",
            "resi": resi_int,
            "resn": resn,
            "grad": float(val),
        })

    if not rows:
        print("No residues with nonzero gradient.")
        return []

    # --- 4) Sorting ---
    if sort_by == "abs":
        rows.sort(key=lambda r: abs(r["grad"]), reverse=descending)
    elif sort_by == "value":
        rows.sort(key=lambda r: r["grad"], reverse=descending)
    elif sort_by == "resi":
        rows.sort(key=lambda r: (r["chain"], r["resi"]))
    else:
        raise ValueError("sort_by must be one of: 'abs','value','resi'.")

    # --- 5) Write to disk (human-readable text) ---
    header = f"{'Chain':<6} {'Resi':>6} {'Resn':>6} {'Gradient':>14}\n"
    sep = "-" * len(header) + "\n"

    with open(output_path, "w") as f:
        f.write(header)
        f.write(sep)
        for r in rows:
            f.write(f"{r['chain']:<6} {str(r['resi']):>6} {r['resn']:>6} {r['grad']:>14.6g}\n")

    print(f"Wrote residue gradient table to: {output_path}")
    return rows

def loadGradient(path):
    with h5py.File(path, 'r') as f:
        gradients = np.array(f['mean_gradients']).T
    return gradients