import math
def parse_velocities(filename):
    velocities = []
    atoms = []
    current_velocities = []
    current_atoms = []
    reading_velocities = False
    with open(filename, 'r') as f:
        for line in f:
            if "Velocities (meter/second):" in line:
                if current_velocities:
                    velocities.append(current_velocities)
                    atoms.append(current_atoms)
                current_velocities = []
                current_atoms = []
                reading_velocities = True
            elif reading_velocities and line.strip():
                parts = line.split()
                if len(parts) == 4 and parts[0] in ['C', 'H']:
                    try:
                        current_atoms.append(parts[0])
                        current_velocities.append([float(x) for x in parts[1:4]])
                    except ValueError:
                        continue
                else:
                    reading_velocities = False
            elif reading_velocities and not line.strip():
                reading_velocities = False

    if current_velocities:
        velocities.append(current_velocities)
        atoms.append(current_atoms)

    return atoms, velocities

def write_xyz(filename, atoms, data, comment):
    with open(filename, 'w') as f:
        for step, (step_atoms, step_data) in enumerate(zip(atoms, data)):
            f.write(f"{len(step_atoms)}\n")
            f.write(f"{comment.format(step)}\n")
            for atom, values in zip(step_atoms, step_data):
                f.write(f"{atom} {values[0]:15.6e} {values[1]:15.6e} {values[2]:15.6e}\n")

# Atomic masses (in atomic units)
atomic_masses = {'H': 1836.15267389, 'C': 21874.66159}

# Physical constants
alpha = 7.2973525693e-3  # Fine structure constant
c = 299792458  # Speed of light in m/s

# Conversion factor
conversion_factor = 1 / (alpha * c)

# Parse the file
file_path = '/path/job.out'
atoms, velocities = parse_velocities(file_path)

# Convert velocities to atomic units
au_velocities = [[[v * conversion_factor for v in vel] for vel in step_vel] for step_vel in velocities]

# Convert velocities to momenta (in atomic units)
au_momenta = [[[v * atomic_masses[atom] for v in vel] for atom, vel in zip(step_atoms, step_vel)] 
              for step_atoms, step_vel in zip(atoms, au_velocities)]
# Write the XYZ files
write_xyz('/path/velocities_ms.xyz', atoms, velocities, 'Step {}: Velocities (m/s)')
write_xyz('/path/velocities_au.xyz', atoms, au_velocities, 'Step {}: Velocities (atomic units, a.u.). 1 a.u. of velocity = 2.18769126364e6 m/s. This means 1 a.u. = αc, where α is the fine structure constant and c is the speed of light. In the atomic units system: length unit is Bohr radius (a₀), time unit is ℏ/E_h (ℏ: reduced Planck constant, E_h: Hartree energy), mass unit is electron mass (m_e). Conversion equation: v(a.u.) = v(m/s) / (αc) = v(m/s) * 4.5710289e-7.')
write_xyz('/path/momenta_au.xyz', atoms, au_momenta, 'Step {}: Momenta (atomic units, where 1 a.u. of momentum = 1.99285191410e-24 kg⋅m/s). Conversion equation: p(a.u.) = m(a.u.) * v(a.u.), where m(H) = 1836.15267389 a.u., m(C) = 21874.66159 a.u.')

print("XYZ files have been generated.")
print(f"Number of steps processed: {len(velocities)}")
print(f"Number of atoms in each step: {len(atoms[0]) if atoms else 0}")