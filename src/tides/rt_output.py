import numpy as np
import os
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

'''
Real-time Output Functions
'''

# ==== start of new additions
def _output_prefix_info(rt_scf):
    prefix = getattr(rt_scf, 'save_prefix', None)
    if prefix in (None, ''):
        return None, None

    prefix = os.fspath(prefix)
    output_dir = prefix
    file_stem = os.path.basename(prefix.rstrip(os.sep)) or prefix
    return output_dir, file_stem


def _ensure_output_dir(rt_scf):
    output_dir, _ = _output_prefix_info(rt_scf)
    if output_dir is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _step_index(rt_scf):
    return int(np.rint(rt_scf.current_time / rt_scf.timestep))


def _save_array(rt_scf, filename, data):
    output_dir = _ensure_output_dir(rt_scf)
    if output_dir is None:
        return

    try:
        array_data = np.asarray(data)
    except ValueError:
        if isinstance(data, (list, tuple)):
            array_data = np.empty(len(data), dtype=object)
            for index, value in enumerate(data):
                array_data[index] = value
        else:
            array_data = np.array([data], dtype=object)

    np.save(os.path.join(output_dir, filename), array_data)


def _save_npz(rt_scf, filename, **data):
    output_dir = _ensure_output_dir(rt_scf)
    if output_dir is None:
        return

    np.savez(os.path.join(output_dir, filename), **data)


def _save_grid_metadata(rt_scf):
    if getattr(rt_scf, '_cube_metadata_saved', False):
        return
    if not getattr(rt_scf, 'save_cube_density', False):
        return

    output_dir, file_stem = _output_prefix_info(rt_scf)
    if output_dir is None:
        return

    from pyscf.tools.cubegen import Cube

    cube = Cube(rt_scf._scf.mol, nx=rt_scf.cube_nx, ny=rt_scf.cube_ny, nz=rt_scf.cube_nz)
    grid_coords = cube.get_coords().reshape(cube.nx, cube.ny, cube.nz, 3)
    _save_npz(
        rt_scf,
        f'{file_stem}_grid_data.npz',
        grid_coords=grid_coords,
        grid_shape=np.asarray([cube.nx, cube.ny, cube.nz]),
        grid_origin=np.asarray(cube.boxorig),
        grid_box=np.asarray(cube.box),
    )
    rt_scf._cube_metadata_saved = True


def _save_matrix_order(rt_scf):
    if getattr(rt_scf, '_matrix_order_saved', False):
        return
    if not getattr(rt_scf, 'save_matrix_order', False):
        return

    output_dir, file_stem = _output_prefix_info(rt_scf)
    if output_dir is None:
        return

    ao_labels = np.asarray(rt_scf._scf.mol.ao_labels(), dtype=str)
    _save_npz(
        rt_scf,
        f'{file_stem}_ao_matrix_order.npz',
        ao_labels=ao_labels,
        nmat=np.asarray(rt_scf.nmat),
        step=np.asarray(_step_index(rt_scf)),
    )
    rt_scf._matrix_order_saved = True


def prepare_saved_outputs(rt_scf):
    if not getattr(rt_scf, 'save_prefix', None):
        return

    _ensure_output_dir(rt_scf)
    if getattr(rt_scf, 'save_cube_density', False):
        _save_grid_metadata(rt_scf)
    if getattr(rt_scf, 'save_matrix_order', False):
        _save_matrix_order(rt_scf)


def save_step_artifacts(rt_scf):
    if not getattr(rt_scf, 'save_prefix', None):
        return

    output_dir, file_stem = _output_prefix_info(rt_scf)
    if output_dir is None:
        return

    step = _step_index(rt_scf)

    if getattr(rt_scf, 'save_density_matrix', False):
        _save_array(rt_scf, f'{file_stem}_density_matrix_step_{step}.npy', rt_scf.den_ao)

    if getattr(rt_scf, 'save_fock_matrix', False) and hasattr(rt_scf, 'fock_ao'):
        _save_array(rt_scf, f'{file_stem}_fock_matrix_step_{step}.npy', rt_scf.fock_ao)

    if not getattr(rt_scf, 'save_observables_to_npy', False):
        return

    observable_values = {
        'energy': lambda: rt_scf._energy,
        'dipole': lambda: rt_scf._dipole,
        'quadrupole': lambda: rt_scf._quadrupole,
        'charge': lambda: rt_scf._charge,
        'mulliken_charge': lambda: rt_scf._atom_charges,
        'mulliken_atom_charge': lambda: rt_scf._atom_charges,
        'hirsh_charge': lambda: rt_scf._hirshfeld_charges,
        'hirsh_atom_charge': lambda: rt_scf._hirshfeld_charges,
        'mag': lambda: rt_scf._mag,
        'hirsh_mag': lambda: np.transpose([rt_scf._hirshfeld_mx_atoms, rt_scf._hirshfeld_my_atoms, rt_scf._hirshfeld_mz_atoms]),
        'hirsh_atom_mag': lambda: np.transpose([rt_scf._hirshfeld_mx_atoms, rt_scf._hirshfeld_my_atoms, rt_scf._hirshfeld_mz_atoms]),
        'mo_occ': lambda: rt_scf._mo_occ,
        'mo_occ_separate': lambda: rt_scf._mo_occ_separate,
        'nuclei': lambda: rt_scf._nuclei,
        'spin_square': lambda: np.asarray([rt_scf._s2, rt_scf._2s_p1]),
        'spin_observables': lambda: np.asarray([rt_scf._s2, rt_scf._2s_p1, rt_scf._sz, rt_scf._sz2, rt_scf._spin_polarization]),
        'mo_coeff': lambda: rt_scf._scf.mo_coeff,
        'den_ao': lambda: rt_scf.den_ao,
        'fock_ao': lambda: rt_scf.fock_ao,
    }

    for key, getter in observable_values.items():
        if not rt_scf.observables.get(key, False):
            continue
        try:
            _save_array(rt_scf, f'{file_stem}_{key}_step_{step}.npy', getter())
        except AttributeError:
            continue
# ==== end of new additions

def update_output(rt_scf):
    rt_scf._log.note(f'{"="*25} \n')
    rt_scf._log.note(f'Current Time (AU): {rt_scf.current_time:.8f} \n')
    for key, function in rt_scf._observables_functions.items():
        function[1](rt_scf)

    rt_scf._log.note(f'{"="*25} \n')
    save_step_artifacts(rt_scf)

def _print_energy(rt_scf):
    energy = rt_scf._energy
    rt_scf._log.note(f'Total Energy (AU): {energy[0]} \n')
    if len(energy) > 1:
        for index, fragment in enumerate(energy[1:]):
            rt_scf._log.note(f'Fragment {index + 1} Energy (AU): {fragment} \n')
    if rt_scf.istype('RT_Ehrenfest'):
        kinetic_energy = rt_scf._kinetic_energy
        rt_scf._log.note(f'Total Kinetic Energy (AU): {np.sum(kinetic_energy)} \n')
        rt_scf._log.info(f'Atom Kinetic Energies (AU):')
        for atom in zip(rt_scf.nuc.labels, kinetic_energy):
            rt_scf._log.info(f' {atom[0]} {atom[1]}')
        rt_scf._log.info(' ')
        for index, frag in enumerate(rt_scf.fragments):
            rt_scf._log.note(f'Fragment {index + 1} Kinetic Energy (AU): {np.sum(kinetic_energy[frag.match_indices])} \n')

def _print_mo_occ(rt_scf):
    mo_occ = rt_scf._mo_occ
    rt_scf._log.note(f'Molecular Orbital Occupations: {" ".join(map(str,mo_occ))} \n')

def _print_mo_occ_separate(rt_scf):
    mo_occ_separate = rt_scf._mo_occ_separate
    rt_scf._log.note(f'Molecular Orbital Alpha Occupations: {" ".join(map(str,mo_occ_separate[0]))} \n')
    rt_scf._log.note(f'Molecular Orbital Beta Occupations: {" ".join(map(str,mo_occ_separate[1]))} \n')

def _print_charge(rt_scf):
    charge = rt_scf._charge
    rt_scf._log.note(f'Total Electronic Charge: {np.real(charge[0])} \n')
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            rt_scf._log.note(f'Fragment {index + 1} Electronic Charge: {np.real(fragment)} \n')

def _print_hirshfeld_charge(rt_scf):
    labels = rt_scf.labels
    atom_charges = rt_scf._hirshfeld_charges
    rt_scf._log.note('Hirshfeld Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_scf._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_scf._log.note(' ')

def _print_dipole(rt_scf):
    dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')

def _print_quadrupole(rt_scf):
    quadrupole = rt_scf._quadrupole
    rt_scf._log.note(f'Total Quadrupole Moment [[XX,XY,XZ], [YX,YY,YZ], [ZX,ZY,ZZ]] (AU): {" ".join(map(str,quadrupole))} \n')

def _print_mag(rt_scf):
    mag = rt_scf._mag
    rt_scf._log.note(f'Total Magnetization [X, Y, Z]: {" ".join(map(str,np.real(mag)))} \n')

def _print_hirshfeld_mag(rt_scf):
    labels = rt_scf.labels
    mx = rt_scf._hirshfeld_mx_atoms
    my = rt_scf._hirshfeld_my_atoms
    mz = rt_scf._hirshfeld_mz_atoms
    m = np.transpose([mx, my, mz])
    rt_scf._log.note(f'Hirshfeld Magnetization [X, Y, Z]:')
    for atom in zip(labels, m):
        rt_scf._log.note(f' {atom[0]}: {np.real(atom[1][0])} {np.real(atom[1][1])} {np.real(atom[1][2])}')
    rt_scf._log.note(' ')

def _print_mulliken_charge(rt_scf):
    labels = rt_scf.labels
    atom_charges = rt_scf._atom_charges
    rt_scf._log.note('Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_scf._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_scf._log.note(' ')

def _print_nuclei(rt_scf):
    rt_scf._xyz_log.note(f'{rt_scf._scf.mol.natm}')
    rt_scf._xyz_log.note(f'Current Time (AU): {rt_scf.current_time:.8f}')
    rt_scf._update_xyz(rt_scf, rt_scf._nuclei)

def _nuclei_coords(rt_scf, nuclei):
    for atom in zip(nuclei[0], nuclei[1]):
        atom_coords = "	".join(map(lambda x: f"{x:.11f}", atom[1]))
        rt_scf._xyz_log.note(f'{atom[0]} 	 {atom_coords}')

def _nuclei_coords_vels(rt_scf, nuclei):
    for atom in zip(nuclei[0], nuclei[1], nuclei[2]):
        atom_coords = "	".join(map(lambda x: f"{x:.11f}", atom[1]))
        atom_vels = "	".join(map(lambda x: f"{x:.11f}", atom[2]))
        rt_scf._xyz_log.note(f'{atom[0]} 	 {atom_coords} 	 {atom_vels}')

def _nuclei_coords_vels_forces(rt_scf, nuclei):
    for atom in zip(nuclei[0], nuclei[1], nuclei[2], nuclei[3]):
        atom_coords = "	".join(map(lambda x: f"{x:.11f}", atom[1]))
        atom_vels = "	".join(map(lambda x: f"{x:.11f}", atom[2]))
        atom_forces = "	".join(map(lambda x: f"{x:.11f}", atom[3]))
        rt_scf._xyz_log.note(f'{atom[0]} 	 {atom_coords} 	 {atom_vels} 	 {atom_forces}')

def _print_spin_square(rt_scf):
    s2 = rt_scf._s2
    _2s_p1 = rt_scf._2s_p1
    rt_scf._log.note(f'S^2: {s2}')
    rt_scf._log.note(f'2S+1: {_2s_p1} \n')

def _print_spin_observables(rt_scf):
    s2 = rt_scf._s2
    _2s_p1 = rt_scf._2s_p1
    sz = rt_scf._sz
    sz2 = rt_scf._sz2
    spin_polarization = rt_scf._spin_polarization
    rt_scf._log.note(f'S^2: {s2}')
    rt_scf._log.note(f'2S+1: {_2s_p1}')
    rt_scf._log.note(f'Sz: {sz}')
    rt_scf._log.note(f'Sz^2: {sz2}')
    rt_scf._log.note(f'Spin Polarization: {spin_polarization} \n')

def _print_mo_coeff(rt_scf):
    #rt_scf._log.note(f'\n{"*"*25} Molecular Orbital Coefficients (AO Basis): {"*"*25}\n {rt_scf._scf.mo_coeff} \n{"*"*50}\n')
    rt_scf._log.note(f'\n{"*"*25} Molecular Orbital Coefficients (AO Basis): {"*"*25}\n \n{"*"*50}\n')

def _print_den_ao(rt_scf):
    # rt_scf._log.note(f'\n{"@"*25} Density Matrix (AO Basis): {"@"*25}\n {rt_scf.den_ao} \n{"@"*50}\n')
    rt_scf._log.note(f'\n{"@"*25} Density Matrix (AO Basis): {"@"*25}\n \n{"@"*50}\n')

def _print_fock_ao(rt_scf):
    # rt_scf._log.note(f'\n{"+"*25} Fock Matrix (AO Basis): {"+"*25}\n {rt_scf.fock_ao} \n{"+"*50}\n')
    rt_scf._log.note(f'\n{"+"*25} Fock Matrix (AO Basis): {"+"*25}\n \n{"+"*50}\n')
