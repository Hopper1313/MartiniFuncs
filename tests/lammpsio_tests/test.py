import MartiniFuncs.graph as mfg
import MartiniFuncs.lammpsio as mflio
import MartiniFuncs.geom_utils as mfgu

asedump = mflio.make_ase_obj_and_fix_atomtypes(
    dumpfile,
    datafile,
)

mydump = mflio.my_readlammps(
    dumpfile
)


# --------------------------------------------------
# Find pre-reactive molecular pairs
# --------------------------------------------------

pairs, shifts = mfgu.get_all_prereactive_pairs(
    asedump,
    scale=1.5,
    reactive_radius=3.25,
    printing=True,
)


# --------------------------------------------------
# Add 3 Å molecular environment
# --------------------------------------------------

clusters = mfgu.get_pair_shell_clusters(
    asedump,
    pairs,
    shifts,
    shell_radius=3.0,
    scale=1.5,
)


# --------------------------------------------------
# Attach velocities / forces
# --------------------------------------------------

cluster_data = mfgu.get_cluster_positions_and_forces(
    asedump,
    mydump,
    clusters,
)
