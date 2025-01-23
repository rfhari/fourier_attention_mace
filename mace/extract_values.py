def extract_dipole_moments(file_path):
    dipole_moments = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "dipole: tensor" in line and "device='cuda:" in line:
                    try:
                        start = line.index("[[") + 2
                        end = line.index("]]")
                        values = line[start:end]
                        dipole_vector = [float(x.strip()) for x in values.split(',')]
                        dipole_moments.append(dipole_vector)
                    except (ValueError, IndexError):
                        print(f"Error parsing line: {line.strip()}")
        
        return dipole_moments

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    file_path = "mace_lr_194028.out"
    moments = extract_dipole_moments(file_path)
    if moments:
        print("Extracted dipole moments:")
        for idx, moment in enumerate(moments, start=1):
            print(f"Dipole moment {idx}: {moment}")
    else:
        print("No dipole moments found or an error occurred.")
