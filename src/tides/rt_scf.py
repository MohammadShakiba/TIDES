import numpy as np
from scipy.linalg import inv
from pyscf import gto
from pyscf.scf import addons
from pyscf.lib import logger
from tides.rt_prop import propagate
from tides import rt_observables
from tides import rt_output
from tides.rt_utils import restart_from_chkfile
import os


'''
Real-time SCF Class
'''

class RT_SCF:
    def __init__(self, scf, timestep, max_time, filename=None, prop=None, frequency=1, orth=None, chkfile=None, verbose=3,
                 save_prefix=None, cube_nx=80, cube_ny=80, cube_nz=80, save_outputs=None):

        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time
        self._scf = scf
        self.ovlp = self._scf.get_ovlp()
        self.occ = self._scf.get_occ()
        
        self.verbose = verbose
        # ==== start of new additions
        self.save_prefix = save_prefix
        self.cube_nx = cube_nx
        self.cube_ny = cube_ny
        self.cube_nz = cube_nz
        self.save_outputs = {} if save_outputs is None else dict(save_outputs)
        self.save_cube_density = bool(self.save_outputs.get('cube_density', False))
        self.save_density_matrix = bool(self.save_outputs.get('density_matrix', False))
        self.save_fock_matrix = bool(self.save_outputs.get('fock_matrix', False))
        self.save_observables_to_npy = bool(self.save_outputs.get('observables', False))
        self.save_matrix_order = bool(self.save_outputs.get('matrix_order', False))
        self.cube_density_indices = None
        self._sz = 0.0
        self._sz2 = 0.0
        self._spin_polarization = 0.0
        self._save_metadata_written = False
        self._matrix_order_saved = False
        self._cube_metadata_saved = False
        # ==== end of new additions
        self._potential = []
        self.fragments = []

        self.labels = [self._scf.mol._atom[idx][0] for idx, _ in enumerate(self._scf.mol._atom)]
        if prop is None: prop = 'magnus_interpol'
        if orth is None: orth = addons.canonical_orth_(self.ovlp)
        self.prop = prop

        self.orth = orth

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            #self._fh = open(filename, 'w')
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        if len(np.shape(self.den_ao)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        # Restart from chkfile, or create a chkfile
        # If restarting from chkfile, self.den_ao will be rewritten
        if chkfile is not None:
            self.chkfile = chkfile
        else:
            print('Warning: chkfile not set, defaulting to tides.chk')
            #self._log.note('Warning: chkfile not set, defaulting to tides.chk')
            self.chkfile = 'tides.chk'
        if os.path.exists(self.chkfile):
            restart_from_chkfile(self)
            self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        else:
            self.current_time = 0
        self._t0 = self.current_time

        rt_observables._init_observables(self)

    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)

    def update_time(self):
        self.current_time += self.timestep

    def get_fock_orth(self, den_ao):
        self.fock_ao = self._scf.get_fock(dm=den_ao).astype(np.complex128)
        if self._potential: self.apply_potential()
        return np.matmul(self.orth.conj().T, np.matmul(self.fock_ao, self.orth))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(inv(self.orth), coeff_ao)

    def rotate_coeff_to_ao(self, coeff_orth):
        return np.matmul(self.orth, coeff_orth)

    def add_potential(self, *args):
        for v_ext in args:
            self._potential.append(v_ext)

    def apply_potential(self):
        for v_ext in self._potential:
            self.fock_ao += v_ext.calculate_potential(self)

    def kernel(self, mo_coeff_print=None):
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            #if np.isclose(self.current_time, self.max_time + self._t0):
            if np.isclose(self.current_time, self.max_time): # So calculation terminates once max_time is reached after restarts
                self._log.note('Done')
            else:
                self._log.note('Propagation Stopped Early')
            if hasattr(self, 'fh'):
                self.fh.close()
            if hasattr(self, '_xyz_fh'):
                # This is only important for unfrozen nuclei, printing .xyz files
                # Putting this here anyways for RT_Ehrenfest and other future derived classes
                self._xyz_fh.close()

        return self
