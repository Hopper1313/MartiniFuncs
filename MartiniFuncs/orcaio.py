from glob import glob

import numpy as np

from .coreio import initialize_constraints_boolean, ase_atoms_to_molecule, coords_str_to_ase
import McUtils.Coordinerds as coordops
from McUtils import Numputils as nput
from Psience.Molecools import Molecule

from ase import Atoms
from ase.io import read,write

from contextlib import nullcontext
from pathlib import Path
import sys



def positivify_angle(angle):
    """Return angle+360 if angle<0, otherwise angle (degrees).
    This is a single-turn adjustment for conventional signed angles, not modulo
    normalization of arbitrary multi-turn inputs.
    """
    angle=float(angle)
    if angle < 0:
        return angle + 360
    else:
        return angle

def get_dat_from_line(line):
    bondlength=0
    angle=0
    dihedral=0
    atom1=0
    atom2=0
    atom3=0
    atomsym=None
    if len(line.split()) == 1:
        atomsym = line.split()[0]
    if len(line.split()) == 3:
        atomsym= line.split()[0]
        atom1 = int(line.split()[1])
        bondlength = float(line.split()[2])
    if len(line.split()) == 5:
        atomsym= line.split()[0]
        atom1 = int(line.split()[1])
        bondlength = float(line.split()[2])
        atom2 = int(line.split()[3])
        angle = float(line.split()[4])
    if len(line.split()) == 7:
        atomsym= line.split()[0]
        atom1 = int(line.split()[1])
        atom2 = int(line.split()[3])
        atom3 = int(line.split()[5])
        bondlength = float(line.split()[2])
        angle = float(line.split()[4])
        dihedral = float(line.split()[6])
    return atomsym,atom1,atom2,atom3,bondlength,angle,dihedral

def rewrite_internals_for_orca(internals):
    new_lines = []
    def _newline_from_dat(atomsym,atom1,atom2,atom3,bondlength,angle,dihedral):
        line = f"{atomsym} {atom1:3.0f} {atom2:3.0f} {atom3:3.0f} {bondlength:11.8f} {angle:11.6f} {dihedral:11.5f}"
        return line

    for line in internals.splitlines():
        atomsym,atom1,atom2,atom3,bondlength,angle,dihedral = get_dat_from_line(line)
        new_line = _newline_from_dat(atomsym,atom1,atom2,atom3,bondlength,angle,dihedral)
        new_lines.append(new_line)
    # print(new_lines)
    return new_lines

def orca_internals_from_molecule(molecule):
    zmat = molecule.get_canonical_zmatrix()
    internals = molecule.modify(internals=zmat).to_string('zmat')
    return rewrite_internals_for_orca(internals)

def orca_internals_to_constraints(internals_lines,constraint_bools,values=False):
    """Convert 1-based Z-matrix reference indices to 0-based indices by i-1;
    zero placeholders become -1. Angles use positivify_angle (degrees).
    This legacy writer always constrains angles/dihedrals and writes bonds
    where constraint_bools[:,0] is False; it does not evaluate geometry.
    """
    lines = ['%geom Constraints\n']
    def _constraint(type,indices,value, values=values):
        if type == 'A' and len(indices) != 3:
            raise ValueError("Angle constraints must have 3 indices.")
        elif type == 'D' and len(indices) != 4:
            raise ValueError("Dihedral constraints must have 4 indices.")
        elif type == 'B' and len(indices) != 2:
            raise ValueError("Bond constraints must have 2 indices.")
        elif type != 'A' and type != 'D' and type != 'B':
            raise ValueError("Constraint type must be 'A' for angle, 'D' for dihedral, or 'B' for bond.")
        indices_str = ' '.join(str(idx) for idx in indices)
        if values:
            return '\t{ ' + f"{type} {indices_str} {float(value):.4f}" + ' C }\n'
        else:
            return '\t{ ' + f"{type} {indices_str}" + ' C }\n'
    for i,line in enumerate(internals_lines, start=1):
        atomsym,atom1,atom2,atom3,bondlength,angle,dihedral = line.split()
        atom1,atom2,atom3 = int(atom1), int(atom2), int(atom3)
        atom0 = i-1
        atom1 = atom1-1
        atom2 = atom2-1
        atom3 = atom3-1
        if atom3 == -1 and atom2 != -1:
            lines.append(_constraint('A',[atom0, atom1, atom2], positivify_angle(angle), values=values))
        if atom3 != -1:
            lines.append(_constraint('A',[atom0,atom1,atom2], positivify_angle(angle), values=values))
            lines.append(_constraint('D',[atom0,atom1,atom2,atom3], positivify_angle(dihedral), values=values))
        if not constraint_bools[i-1,0]:
            lines.append(_constraint('B',[atom0, atom1], bondlength, values=values))
        # if angle != 0 and dihedral == 0:
        #     lines.append(_constraint('A',[atom0, atom1, atom2], angle))
        # if angle != 0 and dihedral != 0:
        #     lines.append(_constraint('A',[atom0,atom1,atom2], angle))
        #     lines.append(_constraint('D',[atom0,atom1,atom2,atom3], dihedral))
    lines.append('\t  end\n')
    lines.append('\tend\n')
    return lines

def orca_constraints_from_molecule(molecule,values=False):
    internals_lines = orca_internals_from_molecule(molecule)
    constraint_lines = orca_internals_to_constraints(internals_lines,initialize_constraints_boolean(molecule),values=values)
    return constraint_lines



# orca_header = '''!r2scan-3c OPT TIGHTOPT TIGHTSCF SlowConv
# %pal
# 	nprocs 4
# end

# '''

def write_constrainedopt_orca_file(molecule: Molecule,header,filename=None,constraint_lines=None, zmat_lines=None,values=False):
    if isinstance(molecule, Molecule) == False:
        if isinstance(molecule, Atoms):
            try:
                molecule = ase_atoms_to_molecule(molecule)
            except Exception as e:
                raise ValueError("molecule object is not a Psience.Molecools.Molecule instance.  failed to convert ASE atoms to Molecule") from e
        else:
            raise ValueError("molecule object is not a Psience.Molecools.Molecule or ase.Atoms object")
    if constraint_lines is None:
        constraint_lines = orca_constraints_from_molecule(molecule,values=values)
    if zmat_lines is None:
        zmat_lines = orca_internals_from_molecule(molecule)

    if filename is None:
        lines = [header]
        # Print to stdout
        print(header)
        for line in constraint_lines:
            print(line, end='')
            lines.append(line)
        print('\n')
        lines.append('\n')
        print('* int 0 1')
        lines.append('* int 0 1\n')
        for line in zmat_lines:
            print('\t'+line)
            lines.append('\t'+line)
        return lines
    elif filename is not None:
        with open(filename, 'w') as f:
            f.write(header)
            for line in constraint_lines:
                f.write(line)
            f.write('\n')
            f.write('* int 0 1\n')
            for line in zmat_lines:
                f.write('\t'+line+'\n')
            f.write('*\n')

def get_header_from_inp(inp_filename):
    with open(inp_filename, 'r') as f:
        lines = f.readlines()
    header_lines = []
    for line in lines:
        if '%geom Constraints' in line or '* int' in line:
            break
        header_lines.append(line)
    return ''.join(header_lines)

def extract_constraints_from_inp(inp_filename):
    with open(inp_filename, 'r') as f:
        lines = f.readlines()
    constraint_lines = []
    in_constraints = False
    internal_geom_lines = []
    in_geom = False
    for line in lines:
        if '%geom Constraints' in line:
            in_constraints = True
            continue
        if in_constraints:
            if 'end' in line:
                break
            constraint_lines.append(line)
    for line in lines:
        if '* int' in line:
            in_geom = True
            continue
        if in_geom:
            if '*' in line:
                break
            internal_geom_lines.append(line)
    return constraint_lines, internal_geom_lines

def _check_if_constraint_lines_have_values(constraint_lines):
    # Check the first few lines of the constraint file to see if they include explicit values
    #    Angle example:  { A 0 1 2 109.5 C }    : if the line has 8 elements, it includes a value
    #                    { A 0 1 2 C }          : if the line has 7 elements, it does not include a value
    for line in constraint_lines[:5]:
        print(line)
        #Angle
        if 'A' in line:
            _line = line.split()
            if len(_line) == 8:
                return True
            elif len(_line) == 7:
                return False
        #Dihedral
        if 'D' in line:
            _line = line.split()
            if len(_line) == 9:
                return True
            elif len(_line) == 8:
                return False
        #Bond
        if 'B' in line:
            _line = line.split()
            if len(_line) == 7:
                return True
            elif len(_line) == 6:
                return False
    raise ValueError("Unable to determine if constraint lines have values")

# def _check_if_previous_output_internals_match_constraints(molecule, internals_lines, constraint_lines):
#     values = _check_if_constraint_lines_have_values(constraint_lines)
#     constraint_lines_molecule = orca_constraints_from_molecule(molecule,values=values)


def restart_unfinished_orca_constrained_opt_from_log(orcalog, header='Auto', old_inputfile = None, new_inputfile='restart.inp', validate_constraints=True, ignore_validation_error=False, **kwargs):
    if header == 'Auto':
        # Get cwd .inp with glob
        if old_inputfile is not None:
            header = get_header_from_inp(old_inputfile)
        else:
            inp_files = glob('*.inp')
            if not inp_files:
                raise FileNotFoundError("No .inp file found in the current directory.")
            header = get_header_from_inp(inp_files[0])


    natoms, trjcoords = get_opt_trj_from_log(orcalog)
    atoms_final = coords_str_to_ase(trjcoords[-1])
    restart_molecule = ase_atoms_to_molecule(atoms_final)

    if validate_constraints:
        atoms_initial = coords_str_to_ase(trjcoords[0])
        molecule_initial = ase_atoms_to_molecule(atoms_initial)
        validation = _validate_constraint_between_two_structures(molecule_initial, restart_molecule, old_inputfile)
        if not validation:
            if not ignore_validation_error:
                raise ValueError("Constraint validation failed between initial and restart structures.")
            else:
                print("Warning: Constraint validation failed between initial and restart structures, but ignoring as per user request.")


    return write_constrainedopt_orca_file(restart_molecule, header, filename=new_inputfile, values=False, **kwargs)

def get_opt_trj_from_log(logfile: str | Path = 'orcaJob.log',
                         outfile: str | Path | None = None):
    #
    # Extract the optimized trajectory from the ORCA log file in XYZ format
    #   Use this when calculation prematurely terminates (i.e. crashes and doesn't write basename_trj.xyz file)

    context = (
        nullcontext(sys.stdout)
        if outfile is None
        else open(outfile, "w")
    )

    with open(logfile, 'r') as f:
        trjcoords = []
        natoms = None
        while True:
            line = f.readline()
            if not line:
                break
            if 'Number of atoms ' in line:
                natoms = int(line.split()[-1])
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                f.readline()  # Skip the header line
                coords = []
                for _ in range(natoms):
                    line = f.readline()
                    coords.append(line)
                    # trjxyz.append(line)
                trjcoords.append(coords)

    frameno=0
    with context as output:
        for xyz in trjcoords:
            output.write(f'{natoms}\n')
            output.write(f'Frame {frameno}\n')
            frameno+=1
            for line in xyz:
                output.write(line)
    return natoms, trjcoords

def _get_specs_from_inp(file_path):
    constraints_lines,geom_lines = extract_constraints_from_inp(file_path)
    constraints_atomnos = []
    for constraint in constraints_lines:
        if 'end' in constraint or constraint == '':
            continue
        else:
            constraints_atomnos.append(tuple(map(int,constraint.split()[2:-2])))
    return constraints_atomnos

def _validate_constraint_in_optimization(trajxyz,inpfile,endpoints_only=True, atol=1e-8,rtol=1e-3):
    """For each internal q from McUtils internal_coordinate_tensors(order=0),
    compare against the first frame q0 with |q-q0|<=atol+rtol*|q0|.
    Psience coordinates are Bohr; distance internals are Bohr and angular
    internals radians. Compare raw values, not wrapped dihedral differences.
    Return False if any inspected frame violates the tolerance.
    """
    found_discrepancy = False
    specs = _get_specs_from_inp(inpfile)
    asemols = read(trajxyz, index=':')
    mols = [ase_atoms_to_molecule(ase_atoms) for ase_atoms in asemols]
    mol0 = mols[0]

    fwd0 = nput.internal_coordinate_tensors(
        mol0.coords,
        specs,
        order=0
    )

    if endpoints_only:
        fwd_trj = [fwd0[0], nput.internal_coordinate_tensors(
            mols[-1].coords,
            specs,
            order=0
        )[0]]
        for i, fwd in enumerate(fwd_trj):
            for j, (val0, val) in enumerate(zip(fwd0[0], fwd)):
                if not np.isclose(val, val0, rtol=rtol,atol=atol):
                    found_discrepancy=True
                    print(f"Discrepancy found, internal {j} [{specs[j]}]: {val0} vs {val}")

    else:
        fwd_trj = [nput.internal_coordinate_tensors(
            mol.coords,
            specs,
            order=0
        )[0] for mol in mols[1:]]

        for i, fwd in enumerate(fwd_trj):
                for j, (val0, val) in enumerate(zip(fwd0[0], fwd)):
                    if not np.isclose(val, val0, rtol=rtol,atol=atol):
                        found_discrepancy=True
                        print(f"Discrepancy found at frame {i+1}, internal {j}: {val0} vs {val}")


    # True if no discrepancies were found, False otherwise
    return not found_discrepancy

def _validate_constraint_between_two_structures(mol1,mol2,previous_input, atol=1e-8, rtol=1e-3):
    """For internals q1,q2 from McUtils internal_coordinate_tensors(order=0),
    accept iff |q1-q2|<=atol+rtol*|q2| for every internal. Coordinates must be
    Bohr (angles radians). Dihedral differences are not reduced modulo 2*pi.
    """
    specs = _get_specs_from_inp(previous_input)
    fwd1 = nput.internal_coordinate_tensors(
        mol1.coords,
        specs,
        order=0
    )[0]
    fwd2 = nput.internal_coordinate_tensors(
        mol2.coords,
        specs,
        order=0
    )[0]
    found_discrepancy = False
    for j, (val1, val2) in enumerate(zip(fwd1, fwd2)):
        if not np.isclose(val1, val2, rtol=rtol, atol=atol):
            found_discrepancy = True
            print(f"Discrepancy found for internal {j} [{specs[j]}]: {val1} vs {val2}")
    return not found_discrepancy

