
import re

class ForceField:
    def __init__(self, path_ff:str):
        self.path_ff = path_ff
        self.residue_info:dict = {}
        if not self.residue_info:
            self.residue_info = self.extract_residue_info()
        

    def __extract_atom_charges_from_line(self, line):
            match = re.search(r'ATOM\s+(\w+)\s+\w+\s+([-+]?\d*\.\d+|\d+)', line)
            if match:
                atom_name = match.group(1)
                charge = float(match.group(2))
                return atom_name, charge
            return None, None
    

    def read_file(self) -> str:
        try:
            with open(self.path_ff, "r") as file:
                return file.readlines()
        except Exception as e:
            print("Erro ao ler o arquivo:", e)
            return []
        

    def extract_residue_info(self) -> dict:
        residue_info = {}
        lines = self.read_file()
        current_residue = None

        for line in lines:
            if line.startswith("RESI "):
                current_residue = line.split()[1]
            else:
                atom_name, charge = self.__extract_atom_charges_from_line(line)
                if atom_name is not None and current_residue is not None:
                    if current_residue not in residue_info:
                        residue_info[current_residue] = {}
                    residue_info[current_residue][atom_name] = charge

        return residue_info
    
    
    def get_atom_charge(self, residue, atom_name) -> float:
        if residue in self.residue_info and atom_name in self.residue_info[residue]:
            return self.residue_info[residue][atom_name]
        print(f"Erro charge {residue} - {atom_name}", flush=True)
        return None        

    def display_residue_info(self) -> dict:
        print(self.residue_info)


