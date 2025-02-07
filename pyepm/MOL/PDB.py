#!/usr/bin/env python3

# Description
###############################################################################

""" 
Module to store and manipulate data from PDB files.

This module contains the PDB class, which is used to store and manipulate
data from PDB files. The class has methods to read, write, and show data 
from PDB files.

Example:
--------

    >>> pdb = PDB()
    >>> pdb.read('example.pdb')
    >>> pdb.show()

"""

'''
PDB format example (Protein Data Bank):
---------------------------------------

    0         1  2    3   4  5        6       7       8      9      10    11 
    ATOM      1  N   PHE X   8     -26.367  17.707  48.952  0.00  0.00            
    ATOM      2  CA  PHE X   8     -26.921  19.002  48.556  0.00  0.00            
    ATOM      3  C   PHE X   8     -25.789  19.915  48.121  0.00  0.00   

COLUMNS        DATA TYPE       FIELD         DEFINITION
-------------------------------------------------------

 1 -  6        Record name     "ATOM  "      Atom record.
 7 - 11        Integer         serial        Atom serial number.
13 - 16        Atom            name          Atom name.
17             Character       altLoc        Alternate location indicator.
18 - 20        Residue name    resName       Residue name.
22             Character       chainID       Chain identifier.
23 - 26        Integer         resSeq        Residue sequence number.
27             AChar           iCode         Code for insertion of residues.
31 - 38        Real(8.3)       x             X coordinate in Angstroms.
39 - 46        Real(8.3)       y             Y coordinate in Angstroms.
47 - 54        Real(8.3)       z             Z coordinate in Angstroms.
55 - 60        Real(6.2)       occupancy     Occupancy.
61 - 66        Real(6.2)       tempFactor    Temperature factor.
73 - 76        LString(4)      segID         Segment identifier (left-justified).
77 - 78        LString(2)      element       Element symbol (right-justified).
79 - 80        LString(2)      charge        Charge on the atom.
'''

# Imports
###############################################################################
import os
import copy
import logging
import tempfile
import subprocess
import numpy as np
from pyepm.MOL.atom import Atom
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R


# License
###############################################################################
'''
pyepm
Authors: -
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: -
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################
class PDB:
	""" Class to store and manipulate data from PDB files.

	Attributes
	----------
	text : str
		PDB file content.
	data : np.array
		Structured array to store the data.
	atoms : list
		List of atoms.
	resids : list
		List of residue IDs.
	name : str
		PDB name.
	inid : np.array
		Atom IDs.
	resinid : np.array
		Residue IDs.
	serials : np.array
		Atom serial numbers.
	names : np.array
		Atom names.
	altlocs : np.array
		Alternate location indicators.
	resnames : np.array
		Residue names.
	chainids : np.array
		Chain identifiers.
	resseq : np.array
		Residue sequence numbers.
	icodes : np.array
		Code for insertion of residues.
	xs : np.array
		Orthogonal coordinates for X in Angstroms.
	ys : np.array
		Orthogonal coordinates for Y in Angstroms.
	zs : np.array
		Orthogonal coordinates for Z in Angstroms.
	coordinates : np.array
		Atom coordinates.
	occupancys : np.array
		Occupancies.
	bfactors : np.array
		Temperature factors.
	segids : np.array
		Segment identifiers.
	elements : np.array
		Element symbols.
	charges : np.array
		Charges on the atoms.
	"""

	def __init__(self, path: str = "") -> None:
		''' Constructor for PDB class.
		
		Parameters
		----------
		path : str, optional
			Path to the PDB file. Default is an empty string. If path is not provided the PDB object is created empty.
		'''

		self.text:str = ""
		self.data:np.array = np.array([])
		self.atoms:list = []
		self.resids:list = []
		self.name = "None"
		self.inid = np.array([], dtype=np.int32)
		self.resinid = np.array([], dtype=np.int32)
		self.serials = np.array([], dtype=np.int32)
		self.names = np.array([], dtype=np.str_)
		self.altlocs = np.array([], dtype=np.str_)
		self.resnames = np.array([], dtype=np.str_)
		self.chainids = np.array([], dtype=np.str_)
		self.resseq = np.array([], dtype=np.int32)
		self.icodes = np.array([], dtype=np.str_)
		self.xs = np.array([], dtype=np.float32)
		self.ys = np.array([], dtype=np.float32)
		self.zs = np.array([], dtype=np.float32)
		self.coordinates = np.array([], dtype=np.float32)
		self.occupancys = np.array([], dtype=np.float32)
		self.bfactors = np.array([], dtype=np.float32)
		self.segids = np.array([], dtype=np.str_)
		self.elements = np.array([], dtype=np.str_)
		self.charges = np.array([], dtype=np.float32)
		if path != "":
			self.read(path)
			
		
	def set_name(self, name: str) -> None:
		''' Set the PDB name.

		Parameters
		----------
		name : str
			PDB name.
		'''

		self.name = name
	
	def get_name(self) -> str:
		''' Get the PDB name.

		Returns
		-------
		str
			PDB name.
		'''

		return self.name
	
	def set_bfactor(self, value: float, ids: list) -> None:
		''' Set the B-factor for a list of atoms.

		Parameters
		----------
		value : float
			B-factor value.
		ids : list
			List of atom IDs.
		'''

		self.bfactors[ids] = value
		self.data.bfactors[ids] = value
		for i in ids:
			self.atoms[i].bfactor = value

	def set_occupancy(self, value: float, ids: list) -> None:
		''' Set the occupancy for a list of atoms.

		Parameters
		----------
		value : float
			Occupancy value.
		ids : list
			List of atom IDs.
		'''

		self.occupancys[ids] = value
		self.data.occupancys[ids] = value
		for i in ids:
			self.atoms[i].occupancy = value

	def __to_float(self, s: str) -> float:
		''' Convert a string to a float.

		Parameters
		----------
		s : str
			String to be converted.

		Returns
		-------
		float
			Converted float.
		'''

		if any(not c.isdigit() for c in s) or '' == s:
			return .0
		else:
			return float(s)

	def remark_index(self) -> None:
		''' Remark the atom indexes. '''

		self.serials = np.array([i for i in range(self.serials.size)])
		for i, at in enumerate(self.atoms):
			at.index = i+1
		self.update_data()

	def remark_inid(self) -> None:
		''' Remark the atom IDs. '''

		self.inid = np.array([i for i in range(self.inid.size)])
		for i, at in enumerate(self.atoms):
			at.id = i+1
		self.update_data()

	def make(self, atoms: list=[], data: np.recarray=np.recarray) -> None:
		''' Create a PDB object.
		
		Parameters
		----------
		atoms : list, optional
			List of atoms. Default is an empty list.
		data : np.recarray, optional
			Structured array with the data. Default is an empty array.
		'''

		if len(atoms) > 0:
			inid = []
			resinid = []
			serials = []
			names = []
			altlocs = []
			resnames = []
			chainids = []
			resseq = []
			xs = []
			ys = []
			zs = []
			occupancys = []
			bfactors = []
			segids = []
			elements = []
			charges = []
			coordinates = []
			icodes = []
			self.atoms = copy.deepcopy(atoms)
			for at in atoms:
				inid.append(at.id)
				resinid.append(at.resinid)
				serials.append(at.index)
				names.append(at.name)
				altlocs.append(at.altloc)
				resnames.append(at.resname)
				chainids.append(at.chain)
				resseq.append(at.resid)
				xs.append(at.coordinates.x)
				ys.append(at.coordinates.y)
				zs.append(at.coordinates.z)
				coordinates.append(np.array([at.coordinates.x, at.coordinates.y, at.coordinates.z]))
				occupancys.append(at.occupancy)
				bfactors.append(at.bfactor)
				segids.append(at.segment)
				elements.append(at.element)
				charges.append(at.charge)
				icodes.append('')
			
			self.inid = np.array(inid)
			self.resinid = np.array(resinid)
			self.serials = np.array(serials)
			self.names = np.array(names)
			self.altlocs = np.array(altlocs)
			self.resnames = np.array(resnames)
			self.chainids = np.array(chainids)
			self.resseq = np.array(resseq)
			self.xs = np.array(xs)
			self.ys = np.array(ys)
			self.zs = np.array(zs)
			self.occupancys = np.array(occupancys)
			self.bfactors = np.array(bfactors)
			self.segids = np.array(segids)
			self.elements = np.array(elements)
			self.charges = np.array(charges,  dtype=np.float32)
			self.coordinates = np.array(coordinates,  dtype=np.float32)
			self.icodes = np.array(icodes)
			self.update_data()

		elif data.size > 0:
			self.inid = data.inid
			self.resinid = data.resinid
			self.serials = data.serials
			self.names = data.names 
			self.altlocs = data.altlocs
			self.resnames = data.resnames
			self.chainids = data.chainids
			self.resseq = data.resseq
			self.icodes = data.icodes
			self.xs = data.xs
			self.ys = data.ys
			self.zs = data.zs
			self.occupancys = data.occupancys
			self.bfactors = data.bfactors
			self.segids = data.segids
			self.elements = data.elements 
			self.charges = data.charges
			self.coordinates = np.vstack((data['xs'],data['ys'],data['zs'])).T
			self.data = data
			self.updata_atoms()			

	def read(self, path: str) -> np.recarray:
		''' Read a PDB file.

		Parameters
		----------
		path : str
			Path to the PDB file.

		Returns
		-------
		np.recarray
			Structured array with the data.
		'''

		if path:
			self.name = path.split("/")[-1].split(".")[-2]
		else:
			self.name = ""
		inid = []
		serials = []
		names = []
		altlocs = []
		resnames = []
		chainids = []
		resseq = []
		resinid = []
		xs = []
		ys = []
		zs = []
		occupancys = []
		bfactors = []
		segids = []
		elements = []
		charges = []
		coordinates = []
		icodes = []

		try:
			with open(path, "r") as file:
				pdb_lines = file.readlines()

			i = 0
			rinid = 0

			for line in pdb_lines:
				if line.startswith('ATOM') or line.startswith('HETATM'):
					record = line[0:6].strip()
					inid.append(i)
					index = line[6:11].strip()
					if any(not c.isdigit() for c in index) or "*****" in index:
						index = 00000
					index = int(index)
					serials.append(index)
			
					atom = line[12:16].strip()
					names.append(atom)
					
					resname = line[17:21].strip()
					resnames.append(resname)
					
					altloc = line[16].strip()
					altlocs.append(altloc)
					
					chain = line[21].strip()
					chainids.append(chain)
					
					residue_index = int(line[22:26].strip())
					resseq.append(residue_index)
					if residue_index != resseq[i-1]:  
						rinid += 1

					resinid.append(rinid)

					icodes.append(line[26].strip())

					x = float(line[30:38].strip())
					xs.append(x)
					
					y = float(line[38:46].strip())
					ys.append(y)
					
					z = float(line[46:54].strip())
					zs.append(z)
					
					coordinates.append(np.array([x, y, z], dtype=np.float32))

					occupancy = float(line[54:60].strip())
					occupancys.append(occupancy)
					
					beta = float(line[60:66].strip())
					bfactors.append(beta)
					
					segID = line[72:76].strip()
					segids.append(segID)
					
					element = line[76:78].strip()
					elements.append(element)
					
					charge = self.__to_float(line[78:80].strip())
					charges.append(charge)
					
					self.atoms.append(Atom(index=index, name=atom, altloc=altloc, resname=resname, 
											resid=residue_index, chain=chain, coordinates=(x, y, z), 
											occupancy=occupancy, bfactor=beta, segment=segID, charge=charge, 
											element=element, id=i, resinid=rinid
											))
					i += 1
			
			self.inid = np.array(inid)
			self.resinid = np.array(resinid)
			self.serials = np.array(serials)
			self.names = np.array(names)
			self.altlocs = np.array(altlocs)
			self.resnames = np.array(resnames)
			self.chainids = np.array(chainids)
			self.resseq = np.array(resseq)
			self.xs = np.array(xs)
			self.ys = np.array(ys)
			self.zs = np.array(zs)
			self.occupancys = np.array(occupancys)
			self.bfactors = np.array(bfactors)
			self.segids = np.array(segids)
			self.elements = np.array(elements)
			self.charges = np.array(charges)
			self.coordinates = np.array(coordinates,  dtype=np.float32)
			self.icodes = np.array(icodes)
			self.update_data()	
		except FileNotFoundError:
			print(f"File '{path}' not found.")
			return None

		return self.data

	def add_atoms(self, atoms: list) -> None:
		''' Add atoms to the PDB object.

		Parameters
		----------
		atoms : list
			List of atoms.
		'''

		self.atoms += copy.deepcopy(atoms)
		inid = []
		resinid = []
		serials = []
		names = []
		altlocs = []
		resnames = []
		chainids = []
		resseq = []
		xs = []
		ys = []
		zs = []
		occupancys = []
		bfactors = []
		segids = []
		elements = []
		charges = []
		coordinates = []
		icodes = []
		for at in atoms:
			inid.append(at.id)
			resinid.append(at.resinid)
			serials.append(at.index)
			names.append(at.name)
			altlocs.append(at.altloc)
			resnames.append(at.resname)
			chainids.append(at.chain)
			resseq.append(at.resid)
			xs.append(at.coordinates.x)
			ys.append(at.coordinates.y)
			zs.append(at.coordinates.z)
			coordinates.append(np.array([at.coordinates.x, at.coordinates.y, at.coordinates.z],  dtype=np.float32))
			occupancys.append(at.occupancy)
			bfactors.append(at.bfactor)
			segids.append(at.segment)
			elements.append(at.element)
			charges.append(at.charge)
			icodes.append('')

		self.inid = np.append(self.inid, np.array(inid))
		self.resinid = np.append(self.inid, np.array(resinid))
		self.serials = np.append(self.serials, np.array(serials))
		self.names = np.append(self.names, np.array(names))
		self.altlocs = np.append(self.altlocs, np.array(altlocs))
		self.resnames = np.append(self.resnames, np.array(resnames))
		self.chainids = np.append(self.chainids, np.array(chainids))
		self.resseq = np.append(self.resseq, np.array(resseq))
		self.xs = np.append(self.xs, np.array(xs))
		self.ys = np.append(self.ys, np.array(ys))
		self.zs = np.append(self.zs, np.array(zs))
		self.occupancys = np.append(self.occupancys, np.array(occupancys))
		self.bfactors = np.append(self.bfactors, np.array(bfactors))
		self.segids = np.append(self.segids, np.array(segids))
		self.elements = np.append(self.elements, np.array(elements))
		self.charges = np.append(self.charges, np.array(charges))
		self.coordinates = np.append(self.coordinates, np.array(coordinates))
		self.icodes = np.append(self.icodes, np.array(icodes))
		self.update_data()

	def get_text(self) -> str:
		''' Get the PDB file content.
		
		Returns
		-------
		str
			PDB file content.
		'''

		txt = ""
		for i in range(self.data.shape[0]):
			txt += f"ATOM  {self.serials[i]:>5} {self.names[i]:<4}{self.altlocs[i]:<1}{self.resnames[i]:>3} {self.chainids[i]:>1}{self.resseq[i]:>4}{self.icodes[i]:>1}   {self.xs[i]:8.3f}{self.ys[i]:8.3f}{self.zs[i]:8.3f}{self.occupancys[i]:6.2f}{self.bfactors[i]:6.2f}{self.segids[i]:<4}{self.elements[i]:>2}{self.charges[i]:2}\n"
		return txt
	
	def show(self, software: str = "vmd") -> None:
		''' Show the PDB file content.

		Parameters
		----------
		software : str, optional
			Software to show the PDB file. Default is "vmd". Options are "vmd" and "pymol".

		Raises
		------
		ValueError
			If the software is invalid.
		'''

		# Check if the software is valid
		if software not in ["vmd", "pymol"]:
			raise ValueError(f"Invalid software. Choose 'vmd' or 'pymol'. Got: '{software}'")

		text_pdb = self.get_text()

		with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".pdb") as temp_file:
			# Escrever conteúdo no arquivo temporário
			temp_file.write(text_pdb)
			temp_file.flush()

			# Obter o nome do arquivo temporário
			temp_file_name = temp_file.name 

			# Imprimir o nome do arquivo temporário (opcional, apenas para fins de visualização)
			if software == "vmd":
				subprocess.run(f"vmd -pdb {temp_file_name}", shell=True)
			elif software == "pymol":
				subprocess.run(f"pymol {temp_file_name}", shell=True)

	def write(self, path: str) -> None:
		''' Write the PDB file content to a file.

		Parameters
		----------
		path : str
			Path to save the PDB file.
		
		Raises
		------
		ValueError
			If the path is invalid or the file already exists.
		'''

		# Check if the path is valid and if the file already exists
		if path == "":
			raise ValueError("Invalid path. Provide a valid path to save the PDB file.")
		if os.path.exists(path):
			raise ValueError("File already exists. Provide a new path to save the PDB file.")

		with open(path, "w") as file_pdb:
			file_pdb.write(self.get_text())

	def update_data(self) -> None:
		''' Update the structured array with the data. '''

		dtype = [
			('inid', self.inid.dtype),
			('serials', self.serials.dtype),
			('names', self.names.dtype),
			('altlocs', self.altlocs.dtype),
			('resnames', self.resnames.dtype),
			('chainids', self.chainids.dtype),
			('resinid', self.resinid.dtype),
			('resseq', self.resseq.dtype),
			('icodes', self.icodes.dtype),
			('xs', self.xs.dtype),
			('ys', self.ys.dtype),
			('zs', self.zs.dtype),
			('occupancys', self.occupancys.dtype),
			('bfactors', self.bfactors.dtype),
			('segids', self.segids.dtype),
			('elements', self.elements.dtype),
			('charges', self.charges.dtype)
		]

		self.data = np.recarray(self.inid.size, dtype=dtype)

		self.data['inid'] = self.inid
		self.data['resinid'] = self.resinid
		self.data['serials'] = self.serials
		self.data['names'] = self.names
		self.data['altlocs'] = self.altlocs
		self.data['resnames'] = self.resnames
		self.data['chainids'] = self.chainids
		self.data['resseq'] = self.resseq
		self.data['icodes'] = self.icodes
		self.data['xs'] = self.xs
		self.data['ys'] = self.ys
		self.data['zs'] = self.zs
		self.data['occupancys'] = self.occupancys
		self.data['bfactors'] = self.bfactors
		self.data['segids'] = self.segids
		self.data['elements'] = self.elements
		self.data['charges'] = self.charges

	def updata_atoms(self) -> None:
		''' Update the atoms list. '''
		self.atoms = [Atom(
							id=e.inid, index=e.serials, name=e.names, altloc=e.altlocs, 
					  		resname=e.resnames, chain=e.chainids, resid=e.resseq, coordinates=(e.xs, e.ys, e.zs),
							occupancy=e.occupancys, bfactor=e.bfactors, segment=e.segids, element=e.elements, charge=e.charges
							) 
							for e in self.data]	
		
	def rotate(self, angle: float, axis: str) -> None:
		''' Rotate the PDB object.

		Parameters
		----------
		angle : float
			Angle to rotate the PDB object.
		axis : str
			Axis to rotate the PDB object. Options are 'x', 'y', and 'z'.
		'''

		# Criar a rotação usando scipy
		rotation = R.from_euler(axis, angle, degrees=True)  # 'degrees=True' se o ângulo estiver em graus

		# Aplicar a rotação a todas as coordenadas
		self.coordinates = rotation.apply(self.coordinates)
		self.xs, self.ys, self.zs = self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
		self.update_data()

		for i, at in enumerate(self.atoms):
			at.coordinates.x, at.coordinates.y, at.coordinates.z = self.coordinates[i] 

	def get_distance_matrix(self) -> np.array:
		''' Get the distance matrix.

		Returns
		-------
		np.array
			Distance matrix.
		'''

		return distance.cdist(self.coordinates, self.coordinates, 'euclidean')

	def get_center(self) -> np.array:
		''' Get the center of the PDB object.

		Returns
		-------
		np.array
			Center of the PDB object.
		'''

		center = np.mean(self.coordinates, axis = 0)

		return center

	def move_center_to(self, center: tuple = (0, 0, 0)) -> None:
		''' Move the center of the PDB object to a specific point.

		Parameters
		----------
		center : tuple, optional
			Center of the PDB object. Default is (0, 0, 0).
		'''

		current_center = self.get_center()
		translation_vector = np.array(center) - np.array(current_center)
		self.coordinates += translation_vector
		self.xs, self.ys, self.zs = self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
		self.update_data()

		for i, at in enumerate(self.atoms):
			at.coordinates.x, at.coordinates.y, at.coordinates.z = self.coordinates[i] 

	def __copy__(self):
		''' Copy the PDB object.

		Returns
		-------
		PDB
			Copied PDB object.
		'''

		cls = self.__class__
		result = cls.__new__(cls)
		result.__dict__.update(self.__dict__)

		return result
		
	def __deepcopy__(self, memo: dict):
		''' Deep copy the PDB object.

		Paremeters
		----------
		memo : dict
			Memory dictionary.

		Returns
		-------
		PDB
			Deep copied PDB object.
		'''

		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result

		for k, v in self.__dict__.items():
			setattr(result, k, copy.deepcopy(v, memo))

		return result

# Functions
###############################################################################
## Private ##

## Public ##

# Aliases
###############################################################################
