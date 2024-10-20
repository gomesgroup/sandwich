import pandas as pd
import numpy as np
import ase.io
import pandas as pd
from ase.units import Bohr
from ase import Atoms
import multiprocessing
import numpy.linalg as la

class Traj():
    '''
    trajectory class on top of reading xyz file
    '''
    
    def __init__(self, xyzFilename):
        '''
        The xyz file provided will be converted into a list of Atom class self.xyz
        '''
        self.xyzFile = xyzFilename
        self.xyz = ase.io.read(xyzFilename, index=':')
        self.atmNumber = len(self.xyz[0].get_chemical_symbols()) # look at the first frame of the traj and figure out the number of atoms
        
    def getCommentLine(self):
        '''
        There might me important info in the comment line, such energy or state info of NAMD, we can pull out such info and store commentline in a list

        Returns
        -------
        commentLineList : TYPE
            a list of all comment lines
        '''
        frameNum = len(self.xyz)
        commentLineIndex = [(self.atmNumber + 2) * x + 1 for x in range(frameNum)]
        commentLineList = []
        
        with open(self.xyzFile) as f:
            fileList = f.readlines()
            for i in commentLineIndex:
                commentLineList.append(fileList[i])
                
        return commentLineList
    
    def writexyzBohr(self):
        '''
        convert original xyz into a list of pandas dataframes, with units converted to Bohr for molden file
        '''
        DFlist = []
        
        for frame in self.xyz:
            molinBohr = Atoms(frame.get_chemical_symbols(), frame.get_positions() / Bohr)
            DFlist.append(molinBohr)
            self.xyzinDFList = DFlist
        
        return self.xyzinDFList
    
    def getDistanceArr(self, a0, a1):
        distArr = []
        
        for frame in self.xyz:
            mol = Atoms(frame.get_chemical_symbols(), frame.get_positions())
            distArr.append(mol.get_distance(a0, a1))
        
        return distArr
    
    def getAngleArr(self, a0, a1, a2):
        angleArr = []
        
        for frame in self.xyz:
            mol = Atoms(frame.get_chemical_symbols(), frame.get_positions())
            angleArr.append(mol.get_angle(a0, a1, a2))
        
        return angleArr
        
    def getDihedralArr(self, a0, a1, a2, a3):
        dihArr = []
        
        for frame in self.xyz:
            mol = Atoms(frame.get_chemical_symbols(), frame.get_positions())
            dihArr.append(mol.get_dihedral(a0, a1, a2, a3))
            
        return dihArr

def firstTimeStepExitRegion(a, b, c, d):
    time = len(a)
    BDarray = np.zeros((time, 4))
    
    for i in range(time):
        BDarray[i, :] = np.sort([a[i], b[i], c[i], d[i]])
    
    for i in range(time - 50):
        current_bonds = BDarray[i, :]
        
        if sum(current_bonds > 1.8) >= 3:
            next_10_steps = BDarray[i + 1:i + 11, :]
            if all([sum(step > 1.8) >= 3 for step in next_10_steps]):
                exittime = i
                return exittime
    
    return 1000

def classify_trajectory(traj_file):
    try:
        traj = Traj(traj_file)

        a = traj.getDistanceArr(12, 16)
        b = traj.getDistanceArr(13, 16)
        c = traj.getDistanceArr(4, 16)
        d = traj.getDistanceArr(5, 16)

        exitSandwich = firstTimeStepExitRegion(a, b, c, d)

        classification = None

        if exitSandwich == 1000:
            classification = "sandwich"
        elif all(val < 1.5 for val in a[-50:]) and \
             all(val > 1.5 for val in b[-50:]) and \
             all(val > 1.5 for val in c[-50:]) and \
             all(val > 1.5 for val in d[-50:]):
            classification = "pro1"
        elif all(val < 1.8 for val in b[-50:]) and \
             all(val > 1.5 for val in a[-50:]) and \
             all(val > 1.5 for val in c[-50:]) and \
             all(val > 1.5 for val in d[-50:]):
            classification = "pro2"
        elif all(val < 1.5 for val in c[-50:]) and \
             all(val > 1.5 for val in a[-50:]) and \
             all(val > 1.5 for val in b[-50:]) and \
             all(val > 1.5 for val in d[-50:]):
            classification = "pro3"
        elif all(val < 1.5 for val in d[-50:]) and \
             all(val > 1.5 for val in a[-50:]) and \
             all(val > 1.5 for val in b[-50:]) and \
             all(val > 1.5 for val in c[-50:]):
            classification = "pro4"
        else:
            classification = "unclassified"

        return classification, exitSandwich, a, b, c, d
    except FileNotFoundError:
        print(f"File not found: {traj_file}")
        return None, None, None, None, None, None, None
 
# Function to parse XYZ data
def parse_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        steps = []
        current_step = []
        for line in lines:
            if line.strip().isdigit():
                if current_step:
                    steps.append(pd.DataFrame(current_step, columns=['Element', 'X', 'Y', 'Z']).astype({'X': 'float64', 'Y': 'float64', 'Z': 'float64'}))
                    current_step = []
            elif not line.strip().startswith('Step'):  # Skip lines starting with '#'
                current_step.append(line.strip().split()[0:4])
        if current_step:  # Add the last step
            steps.append(pd.DataFrame(current_step, columns=['Element', 'X', 'Y', 'Z']).astype({'X': 'float64', 'Y': 'float64', 'Z': 'float64'}))
        return steps

def atom_momentum(momentum_df, atom_index):
    if atom_index < len(momentum_df):
        # Ensure the momentum values are treated as numeric (floats)
        mom_values = pd.to_numeric(momentum_df.iloc[atom_index][['X', 'Y', 'Z']], errors='coerce').fillna(0).values
        return mom_values
    else:
        print(f"Index {atom_index} is out of bounds for the momentum DataFrame.")
        return np.zeros(3)
    
   
def angle_between_vectors(vec1, vec2):
    """Calculate the angle in degrees between vectors 'vec1' and 'vec2', rounded to 1 decimal place."""
    dot_prod = np.dot(vec1, vec2)
    norms_prod = la.norm(vec1) * la.norm(vec2)
    
    if norms_prod == 0 or dot_prod / norms_prod > 1 or dot_prod / norms_prod < -1:
        return 0

    angle_rad = np.arccos(dot_prod / norms_prod)
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg, 1)

def combined_momentum(momentum_df, atom_indices):
    """Combine the momenta of atoms in the specified indices."""
    combined_mom = np.zeros(3)
    for index in atom_indices:
        combined_mom += atom_momentum(momentum_df, index)
    return combined_mom
 
def save_to_csv(momentum_steps, start_frame, end_frame, csv_file_name, classification, a, b, c, d, momentum_combine_vectors_groups, momentum_H_vectors, momentum_C_vectors, momentum_combine_all_H_vectors, momentum_combine_all_C_vectors, momentum_combine_all_C_and_H_vectors, combined_momentum_H14_H19, combined_momentum_H17_H20, combined_momentum_C5_C12, combined_momentum_C4_C13, combined_momentum_H14_H19_C5_C12, combined_momentum_H17_H20_C4_C13):
    data_list = []

    for step in range(start_frame, end_frame + 1):
    
        momentum_16 = atom_momentum(momentum_steps[step], 16)
    
        # Dot product for all H atoms combined
        dot_pro_all_H = np.dot(combined_momentum(momentum_steps[step], momentum_combine_all_H_vectors['all H']), momentum_16)

        # Dot product for all C atoms combined
        dot_pro_all_C = np.dot(combined_momentum(momentum_steps[step], momentum_combine_all_C_vectors['all C']), momentum_16)
        
        dot_pro_all_C_and_H = np.dot(combined_momentum(momentum_steps[step], momentum_combine_all_C_and_H_vectors['all C and H']), momentum_16)

        
        # H atoms
        momentum_H19 = atom_momentum(momentum_steps[step], 19)  # pro1
        momentum_H17 = atom_momentum(momentum_steps[step], 17)  # pro2
        momentum_H20 = atom_momentum(momentum_steps[step], 20)  # pro3
        momentum_H14 = atom_momentum(momentum_steps[step], 14)  # pro4
        # Corresponding C atoms
        momentum_C12 = atom_momentum(momentum_steps[step], 12)  # pro1
        momentum_C13 = atom_momentum(momentum_steps[step], 13)  # pro2
        momentum_C4 = atom_momentum(momentum_steps[step], 4)    # pro3
        momentum_C5 = atom_momentum(momentum_steps[step], 5)    # pro4

        # Calculate and round magnitudes of the momentum vectors
        magnitude_16 = round(la.norm(momentum_16), 1)
        magnitude_H19 = round(la.norm(momentum_H19), 1)
        magnitude_H17 = round(la.norm(momentum_H17), 1)
        magnitude_H20 = round(la.norm(momentum_H20), 1)
        magnitude_H14 = round(la.norm(momentum_H14), 1)
        magnitude_C12 = round(la.norm(momentum_C12), 1)
        magnitude_C13 = round(la.norm(momentum_C13), 1)
        magnitude_C4 = round(la.norm(momentum_C4), 1)
        magnitude_C5 = round(la.norm(momentum_C5), 1)

        # Calculate angles between Atom 16 and other atoms
        angle_16_H19 = angle_between_vectors(momentum_16, momentum_H19)
        angle_16_H17 = angle_between_vectors(momentum_16, momentum_H17)
        angle_16_H20 = angle_between_vectors(momentum_16, momentum_H20)
        angle_16_H14 = angle_between_vectors(momentum_16, momentum_H14)
        angle_16_C12 = angle_between_vectors(momentum_16, momentum_C12)
        angle_16_C13 = angle_between_vectors(momentum_16, momentum_C13)
        angle_16_C4 = angle_between_vectors(momentum_16, momentum_C4)
        angle_16_C5 = angle_between_vectors(momentum_16, momentum_C5)

        
        magnitudes_H = f"{magnitude_H19}, {magnitude_H17}, {magnitude_H20}, {magnitude_H14}"
        angles_H = f"{angle_16_H19}, {angle_16_H17}, {angle_16_H20}, {angle_16_H14}"
        magnitudes_C = f"{magnitude_C12}, {magnitude_C13}, {magnitude_C4}, {magnitude_C5}"
        angles_C = f"{angle_16_C12}, {angle_16_C13}, {angle_16_C4}, {angle_16_C5}"
        
        
        dot_C4_H = np.dot(momentum_16, momentum_C4)
        dot_C5_H = np.dot(momentum_16, momentum_C5)
        dot_C12_H = np.dot(momentum_16, momentum_C12)
        dot_C13_H = np.dot(momentum_16, momentum_C13)
        dot_H14_H = np.dot(momentum_16, momentum_H14)
        dot_H17_H = np.dot(momentum_16, momentum_H17)
        dot_H19_H = np.dot(momentum_16, momentum_H19)
        dot_H20_H = np.dot(momentum_16, momentum_H20)
        

        bonds_greater_than_1_8 = sum(np.array([a[step], b[step], c[step], d[step]]) > 1.8)
            
        average_magnitudes_H = np.mean([magnitude_H19, magnitude_H17, magnitude_H20, magnitude_H14])
        average_angles_H = np.mean([angle_16_H19, angle_16_H17, angle_16_H20, angle_16_H14])
        average_magnitudes_C = np.mean([magnitude_C12, magnitude_C13, magnitude_C4, magnitude_C5])
        average_angles_C = np.mean([angle_16_C12, angle_16_C13, angle_16_C4, angle_16_C5])
        

            
        data_list.append({
            'Step': step, 
            'Momentum_16': momentum_16.tolist(), 
            # H atoms
            'Momentum_H19 - pro1': momentum_H19.tolist(), 
            'Magnitude_H19 - pro1': magnitude_H19,
            'Angle_16_H19 - pro1': angle_16_H19,
            'Momentum_H17 - pro2': momentum_H17.tolist(), 
            'Magnitude_H17 - pro2': magnitude_H17,
            'Angle_16_H17 - pro2': angle_16_H17,
            'Momentum_H20 - pro3': momentum_H20.tolist(), 
            'Magnitude_H20 - pro3': magnitude_H20,
            'Angle_16_H20 - pro3': angle_16_H20,
            'Momentum_H14 - pro4': momentum_H14.tolist(), 
            'Magnitude_H14 - pro4': magnitude_H14,
            'Angle_16_H14 - pro4': angle_16_H14,
            # C atoms
            'Momentum_C12 - pro1': momentum_C12.tolist(),
            'Magnitude_C12 - pro1': magnitude_C12,
            'Angle_16_C12 - pro1': angle_16_C12,
            'Momentum_C13 - pro2': momentum_C13.tolist(),
            'Magnitude_C13 - pro2': magnitude_C13,
            'Angle_16_C13 - pro2': angle_16_C13,
            'Momentum_C4 - pro3': momentum_C4.tolist(),
            'Magnitude_C4 - pro3': magnitude_C4,
            'Angle_16_C4 - pro3': angle_16_C4,
            'Momentum_C5 - pro4': momentum_C5.tolist(),
            'Magnitude_C5 - pro4': magnitude_C5,
            'Angle_16_C5 - pro4': angle_16_C5,
            'Magnitudes_H_atoms': magnitudes_H,
            'Angles_H_atoms': angles_H,
            'Magnitudes_C_atoms': magnitudes_C,
            'Angles_C_atoms': angles_C,

            'H_Cpro1': a[step],
            'H_Cpro2': b[step],
            'H_Cpro3': c[step],
            'H_Cpro4': d[step],
            'dot_C4_H': dot_C4_H,
            'dot_C5_H': dot_C5_H,
            'dot_C12_H': dot_C12_H,
            'dot_C13_H': dot_C13_H,
            'dot_H14_H': dot_H14_H,
            'dot_H17_H': dot_H17_H,
            'dot_H19_H': dot_H19_H,
            'dot_H20_H': dot_H20_H,
            'dot_pro_all_H': dot_pro_all_H,
            'dot_pro_all_C': dot_pro_all_C,
            'dot_pro_all_C_and_H': dot_pro_all_C_and_H,
            'average Angles_H_atoms': average_angles_H,
            'average Angles_C_atoms': average_angles_C,
            'Magnitude_16': magnitude_16,
            'average Magnitudes_H_atoms': average_magnitudes_H,
            'average Magnitudes_C_atoms': average_magnitudes_C,
            'Bonds > 1.8': bonds_greater_than_1_8,  # Add the count here
        })

    data_df = pd.DataFrame(data_list)

    data_df.to_csv(csv_file_name, index=False)
 
base_path = 'path'
base_path2_template = '/run{}'

def process_run(i):
    base_path2 = base_path2_template.format(i)
    positions_file_path = f"{base_path}{base_path2}/sandwich.xyz"
    momentum_file_path = f"{base_path}{base_path2}/momenta_au.xyz"

    try:
        # Classify the trajectory and get relevant information
        classification, exitSandwich, a, b, c, d = classify_trajectory(positions_file_path)
        if classification is None or exitSandwich is None:
            print(f"Skipping run {i}: File not found or trajectory could not be classified.")
            return

        momentum_combine_vectors_groups = {
            'pro1': [12, 19],
            'pro2': [13, 17],
            'pro3': [4, 20],
            'pro4': [5, 14],
        }

        momentum_H_vectors = {
            'Hpro1': [19],
            'Hpro2': [17],
            'Hpro3': [20],
            'Hpro4': [14],
        }

        momentum_C_vectors = {
            'Cpro1': [12],
            'Cpro2': [13],
            'Cpro3': [4],
            'Cpro4': [5],
        }

        momentum_combine_all_H_vectors = {
            'all H': [19, 17, 20, 14]
        }

        momentum_combine_all_C_vectors = {
            'all C': [12, 13, 4, 5]
        }

        momentum_combine_all_C_and_H_vectors = {
            'all C and H': [12, 13, 4, 5, 19, 17, 20, 14]
        }

        start_frame = exitSandwich - 100
        end_frame = exitSandwich

        # Parse the XYZ data for momentum
        momentum_steps = parse_xyz(momentum_file_path)
        classification, exitSandwich, a, b, c, d = classify_trajectory(positions_file_path)

        combined_momentum_H14_H19 = {
            'combined_momentum_H14_H19': [14, 19]
        }

        combined_momentum_H17_H20 = {
            'combined_momentum_H17_H20': [17, 20]
        }

        combined_momentum_C5_C12 = {
            'combined_momentum_C5_C12': [5, 12]
        }

        combined_momentum_C4_C13 = {
            'combined_momentum_C4_C13': [4, 13]
        }

        combined_momentum_H14_H19_C5_C12 = {
            'combined_momentum_H14_H19_C5_C12': [14, 19, 5, 12]
        }

        combined_momentum_H17_H20_C4_C13 = {
            'combined_momentum_H17_H20_C4_C13': [17, 20, 4, 13]
        }

        csv_file_name = f"path/run{i}_{classification}_{exitSandwich}fs_{exitSandwich}steps.csv"

        save_to_csv(momentum_steps, start_frame, end_frame, csv_file_name, classification, a, b, c, d, momentum_combine_vectors_groups, momentum_H_vectors, momentum_C_vectors, momentum_combine_all_H_vectors, momentum_combine_all_C_vectors, momentum_combine_all_C_and_H_vectors, combined_momentum_H14_H19, combined_momentum_H17_H20, combined_momentum_C5_C12, combined_momentum_C4_C13, combined_momentum_H14_H19_C5_C12, combined_momentum_H17_H20_C4_C13)
    except Exception as e:
        print(f"Skipping run {i}: An error occurred - {str(e)}")

if __name__ == '__main__':
    for i in range(1, 501):
        process_run(i)
