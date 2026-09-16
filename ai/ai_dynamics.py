import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Compatibility shim: ensures IsAtomInRingOfAnySize exists on RingInfo across RDKit builds
if not hasattr(Chem.rdchem.RingInfo, "IsAtomInRingOfAnySize"):
    Chem.rdchem.RingInfo.IsAtomInRingOfAnySize = lambda self, idx: self.NumAtomRings(idx) > 0

class AIDynamicsEngine:
    def __init__(self, molecule: Chem.Mol):
        self.mol = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(self.mol)
        
        conf = self.mol.GetConformer()
        self.base_coords = conf.GetPositions()
        self.num_atoms = self.mol.GetNumAtoms()

    def compute_flexibility_priors(self):
        flexibility = np.zeros(self.num_atoms)
        ring_info = self.mol.GetRingInfo()

        for i, atom in enumerate(self.mol.GetAtoms()):
            if ring_info.IsAtomInRingOfAnySize(atom.GetIdx()):
                flexibility[i] = 0.15
            elif atom.GetDegree() == 1:
                flexibility[i] = 0.65
            else:
                flexibility[i] = 0.35
        return flexibility

    def generate_trajectory(self, total_time_ps=100.0, num_frames=50, temperature_k=300.0):
        time_points = np.linspace(0, total_time_ps, num_frames)
        flexibility = self.compute_flexibility_priors()
        thermal_factor = np.sqrt(temperature_k / 300.0) * 0.4
        
        trajectory = np.zeros((num_frames, self.num_atoms, 3))
        trajectory[0] = self.base_coords
        mode_frequencies = np.array([0.05, 0.12, 0.28])
        
        for t_idx, t in enumerate(time_points):
            if t_idx == 0:
                continue
            drift = np.zeros((self.num_atoms, 3))
            for f in mode_frequencies:
                drift += np.sin(2 * np.pi * f * t) * 0.1 * thermal_factor

            noise = np.random.normal(0, 0.05 * thermal_factor, size=(self.num_atoms, 3))
            scaled_displacement = (drift + noise) * flexibility[:, np.newaxis]
            trajectory[t_idx] = trajectory[t_idx - 1] * 0.85 + (self.base_coords + scaled_displacement) * 0.15

        return time_points, trajectory

    @staticmethod
    def calculate_metrics(trajectory, base_coords):
        num_frames = trajectory.shape[0]
        diff = trajectory - base_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))

        mean_positions = np.mean(trajectory, axis=0)
        rmsf = np.sqrt(np.mean(np.sum((trajectory - mean_positions)**2, axis=-1), axis=0))

        rg = np.zeros(num_frames)
        for f in range(num_frames):
            center_of_mass = np.mean(trajectory[f], axis=0)
            rg[f] = np.sqrt(np.mean(np.sum((trajectory[f] - center_of_mass)**2, axis=-1)))

        return rmsd, rmsf, rg