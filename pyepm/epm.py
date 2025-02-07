import os
import numba
import numpy as np
from tqdm import tqdm
from pyepm.VOL.dx import DX
from pyepm.QM.aux import AUX
from pyepm.MOL.PDB import PDB
from pyepm.VOL.cube import Cube
from pyepm.QM.orca_out import OrcaOut
from pyepm.FF.forceField import ForceField
from concurrent.futures import ThreadPoolExecutor

class EPM:
    def __init__(self) -> None:
        self.vol = None
        

    def convert_to_kcal_mol(self):
        self.convert_to_ev()
        self.vol.values *= 23.0605

    def convert_to_ev(self):
        self.vol.values *= 14.3996

    def not_convert(self):
        pass
    
    def calculate(self, pdb:PDB, aux:AUX=AUX(), orca_out:OrcaOut=OrcaOut(), res:float=0.5, gpu:bool=False, charges:np.array=np.array([]), FF:str="", form:str="cube", margim:float=0.3, cutoff: float = 15, gpus_id:list=[], unit:str=""):
        if gpu:
            import cupy as cp
            
        coords_atoms = pdb.coordinates

        x_min, y_min, z_min = coords_atoms.min(axis=0)
        x_max, y_max, z_max = coords_atoms.max(axis=0)

        dist_max = np.linalg.norm([x_max - x_min, y_max - y_min, z_max - z_min])

        xorg, yorg, zorg = coords_atoms[:, 0].min() - (dist_max * margim), coords_atoms[:, 1].min() - (dist_max * margim), coords_atoms[:, 2].min() - (dist_max * margim)
        xmax, ymax, zmax = coords_atoms[:, 0].max() + (dist_max * margim), coords_atoms[:, 1].max() + (dist_max * margim), coords_atoms[:, 2].max() + (dist_max * margim)

        xn = int((xmax - xorg) / res) + 1
        yn = int((ymax - yorg) / res) + 1
        zn = int((zmax - zorg) / res) + 1

        d = DX()
        d.make(xdel=res , ydel=res , zdel=res,
               xn=xn, yn=yn, zn=zn,
               xorg=xorg, yorg=yorg, zorg=zorg,
               )
        x = d.coordinates[:, 0]
        y = d.coordinates[:, 1]
        z = d.coordinates[:, 2]
            
        if len(aux.atom_charges) > 0:
            zatoms = np.array(aux.atom_charges)

        elif len(orca_out.mulliken_atomic_charges) > 0:
            zatoms = orca_out.mulliken_atomic_charges

        elif FF == "charmm":
            zatoms = np.zeros([len(pdb.atoms)],dtype=np.float32)
            dir_app = os.path.dirname(os.path.realpath(__file__))
            ff = ForceField(path_ff=f"{dir_app}/FF/top_all36_prot.rtf")
            
            for i, at in enumerate(pdb.atoms):
                zatoms[i] = ff.get_atom_charge(at.resname, at.name)

            zatoms = np.nan_to_num(zatoms, nan=.0)

        elif charges.size > 0:
            zatoms = charges

        if gpu:
            fun = self.comput_mep_multi_gpu
        else:
            fun = self.comput_mep
        
        mask = zatoms != 0  
        coords_atoms = coords_atoms[mask]
        zatoms = zatoms[mask]

        gmep = fun(catoms=coords_atoms, zatoms=zatoms, x=x, y=y, z=z, xn=d.xn, yn=d.yn, zn=d.zn, cutoff=cutoff, gpus_id=gpus_id)

        if gpu:
            gmep = cp.asnumpy(gmep)

        d.values = gmep.flatten()

        if form == "cube":
            a_to_bohr = 1.8897259886
            c = Cube()
            c.make(xdel=res*a_to_bohr , ydel=res*a_to_bohr , zdel=res*a_to_bohr,
                xn=xn, yn=yn, zn=zn,
                xorg=xorg*a_to_bohr, yorg=yorg*a_to_bohr, zorg=zorg*a_to_bohr,
                )
            c.natoms = len(pdb.atoms)
            c.header = "MEP\n"

            for at in pdb.atoms:
                at.atomic_number
                at.coordinates.convert_to(unit="bohr")
            c.atoms = pdb.atoms
            c.values = d.values
            d = c 
        
        self.vol = d

        fun_convert = {"": self.not_convert, "kcal/mol": self.convert_to_kcal_mol}
        fun_convert[unit]()

        return self.vol

    @staticmethod
    @numba.njit(parallel=True, cache=True, fastmath=True)
    def comput_mep(catoms: np.array, zatoms: np.array, x: np.array, y: np.array, z: np.array, xn: int, yn: int, zn: int,  cutoff=20, gpus_id:list=[]) -> np.array:
        grid = np.stack((x, y, z), axis=-1)
        gmep = np.zeros((xn, yn, zn),  dtype=np.float32)
        
        for i in numba.prange(catoms.shape[0]):
            r = np.sqrt(np.sum((grid - catoms[i]) ** 2, axis=-1))
            valid_mask = (r < cutoff) & (r > 0)
            contribution = np.zeros_like(r)
            contribution[valid_mask] = zatoms[i] / r[valid_mask]
            gmep += contribution.reshape((xn, yn, zn))
        return gmep

    @staticmethod    
    def comput_mep_gpu(catoms, zatoms, x, y, z, xn:int, yn:int, zn:int, cutoff=20, gpus_id:list=[]):
        import cupy as cp
        grid = cp.stack((x, y, z), axis=-1) 
        gmep = cp.zeros((xn, yn, zn), dtype=cp.float32)  

        for i in tqdm(range(catoms.shape[0]), desc="Processing MEP", unit=" atoms"):
            r = cp.sqrt(cp.sum((grid - catoms[i]) ** 2, axis=-1))
            valid_mask = (r < cutoff) & (r > 0)
            contribution = cp.zeros_like(r)
            contribution[valid_mask] = zatoms[i] / r[valid_mask]
            gmep += contribution.reshape((xn, yn, zn))
        return gmep
    
    @staticmethod
    def comput_mep_multi_gpu(catoms: np.array, zatoms: np.array, x: np.array, y: np.array, z: np.array, xn: int, yn: int, zn: int, cutoff: float = 20, gpus_id: list = []):
        import cupy as cp

        n_gpus = len(gpus_id)

        if n_gpus == 0:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            gpus_id = list(range(n_gpus))

        gmep = np.zeros((xn, yn, zn), dtype=cp.float32)

        batch_size = catoms.shape[0] // n_gpus

        def process_gpu(gpu_id, start_idx, end_idx, progress_bar, x, y, z, catoms, zatoms, cutoff):
            with cp.cuda.Device(gpu_id):
                local_gmep = cp.zeros((xn, yn, zn), dtype=cp.float32)
                x_i = cp.asarray(x, dtype=cp.float32)
                y_i = cp.asarray(y, dtype=cp.float32)
                z_i = cp.asarray(z, dtype=cp.float32)
                
                grid = cp.stack((x_i, y_i, z_i), axis=-1)

                catoms_i = cp.asarray(catoms, dtype=cp.float32)
                zatoms_i = cp.asarray(zatoms, dtype=cp.float32)

                for i in range(start_idx, end_idx):
                    r = cp.sqrt(cp.sum((grid - catoms_i[i]) ** 2, axis=-1))
                    valid_mask = (r < cutoff) & (r > 0)
                    contribution = cp.zeros_like(r)
                    contribution[valid_mask] = zatoms_i[i] / r[valid_mask]
                    local_gmep += contribution.reshape((xn, yn, zn))

                    progress_bar.update(1)

                return local_gmep
            
        
        futures = []
        with tqdm(total=catoms.shape[0], desc="Processing MEP", unit=" atoms") as pbar:
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                for gpu_id in gpus_id:
                    start_idx = gpu_id * batch_size
                    end_idx = (gpu_id + 1) * batch_size if gpu_id < n_gpus - 1 else catoms.shape[0]
                    futures.append(executor.submit(process_gpu, gpu_id, start_idx, end_idx, pbar, x, y, z, catoms, zatoms, cutoff))

                for future in futures:
                    gmep += future.result().get()

        return gmep
