#!/usr/bin/env python3

# Description
###############################################################################

""" Module to store and manipulate data from XYZ files.

This module contains the XYZ class, which is used to store and manipulate
data from XYZ files. The class has methods to read, write, and show data from
XYZ files. The data is stored in a structured array, which can be accessed
using the get_text method.

Example
-------
	>>> xyz = XYZ()
	>>> xyz.read("file.xyz")
	>>> print(xyz.get_text())
	>>> xyz.show("vmd")
	>>> xyz.write("file.xyz")

"""

# Imports
###############################################################################
import subprocess
import tempfile
import logging

import numpy as np

from pyepm.MOL.atom import Atom
from scipy.spatial import distance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
class XYZ:
	""" Class to store and manipulate data from XYZ files.
	
	Attributes
	----------
	name : str
		XYZ file name.
	description : str
		XYZ file description.
	atoms : list
		List of Atom objects.
	natoms : int
		Number of atoms.
	inid : np.array
		Array of atom IDs.
	elements : np.array
		Array of atom elements.
	xs : np.array
		Array of x coordinates.
	ys : np.array
		Array of y coordinates.
	zs : np.array
		Array of z coordinates.
	coordinates : np.array
		Array of coordinates.
	data : np.array
		Structured array to store the data.
	"""

	def __init__(self) -> None:
		''' Constructor for XYZ class. '''

		self.description:str = ""
		self.atoms = []
		self.natoms:int = 0
		self.inid = np.array([], dtype=np.int64)
		self.elements = np.array([], dtype=np.str_)
		self.xs = np.array([], dtype=np.float64)
		self.ys = np.array([], dtype=np.float64)
		self.zs = np.array([], dtype=np.float64)
		self.coordinates = np.array([], dtype=np.float64)
		self.data:np.array = np.array([])

	def set_name(self, name: str) -> None:
		''' Set the XYZ file name.

		Parameters
		----------
		name : str
			XYZ file name.
		'''

		self.name = name

	def get_name(self) -> str:
		''' Return the XYZ file name.

		Returns
		-------
		str
			XYZ file name.
		'''

		return self.name

	def read(self, path: str) -> None:
		''' Read data from an XYZ file.

		Parameters
		----------
		path : str
			Path to the XYZ file.
		
		Raises
		------
		FileNotFoundError
			If the file is not found.
		'''

		self.name = path.split("/")[-1].split(".")[-2]
		inid = []
		xs = []
		ys = []
		zs = []
		coordinates = []
		elements = []
		
		try:
			with open(path, "r") as file:
				xyz_lines = file.readlines()
		except FileNotFoundError:
			logging.error(f"File '{path}' not found.")
			return
		
		self.natoms = int(xyz_lines[0])
		self.description = xyz_lines[1].strip()

		for i, line in enumerate(xyz_lines[2:]):
			inid.append(i)
			ls = line.split()
			elements.append(ls[0].strip())
			xs.append(float(ls[1]))
			ys.append(float(ls[2]))
			zs.append(float(ls[3]))
			coordinates.append(np.array([float(ls[1]), float(ls[2]), float(ls[3])]))

			self.atoms.append(Atom(index=i, name=elements[i], altloc="", resname="UNK", 
						   				resid=0, chain="Z", coordinates=(xs[i], ys[i], zs[i]), 
										occupancy=0.00, bfactor=0.00, segment="UNKK", charge=0.00, 
										element=elements[i], id=i, resinid=0
										))
		
		self.inid = np.array(inid)
		self.xs = np.array(xs)
		self.ys = np.array(ys)
		self.zs = np.array(zs)
		self.elements = np.array(elements)
		self.coordinates = np.array(coordinates)
		self.__update_data()	

	def __update_data(self) -> None:
		''' Update the structured array data. '''

		dtype = [
			('inid', self.inid.dtype),
			('xs', self.xs.dtype),
			('ys', self.ys.dtype),
			('zs', self.zs.dtype),
			('elements', self.elements.dtype),
		]
		
		self.data = np.recarray(self.inid.size, dtype=dtype)

		self.data['inid'] = self.inid
		self.data['xs'] = self.xs
		self.data['ys'] = self.ys
		self.data['zs'] = self.zs
		self.data['elements'] = self.elements   

	def __update_atoms(self) -> None:
		''' Update the atoms list. '''

		self.atoms = [Atom(
							id=e.inid, index=e.inid, name=e.elements, altloc="", 
					  		resname="UKN", chain="Z", resid=0, coordinates=(e.xs, e.ys, e.zs),
							occupancy=0.00, bfactor=0.00, segment="UKNN", element=e.elements, charge=0.00
							) 
							for e in self.data]	

	def get_text(self) -> str:
		''' Return the text representation of the data.

		Returns
		-------
		str
			Text representation of the data.
		'''

		txt = f"{self.natoms}\n{self.description}\n"

		for i in range(self.data.shape[0]):
			txt += f"{self.elements[i]:<2}\t{self.xs[i]:8.3f}{self.ys[i]:8.3f}{self.zs[i]:8.3f}\n"

		return txt
	
	def show(self, software: str = "vmd") -> None:
		''' Show the data using a visualization software.

		Parameters
		----------
		software : str, optional
			Visualization software. Default is "vmd".
		
		Raises
		------
		ValueError
			If the software is not available.
		'''

		# Check if the software is available
		if software not in ["vmd", "pymol"]:
			raise ValueError(f"Software '{software}' not available. Use 'vmd' or 'pymol'.")
		
		text_pdb = self.get_text()

		with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".pdb") as temp_file:
			# Write the text to a temporary file
			temp_file.write(text_pdb)
			temp_file.flush()

			# Get the name of the temporary file
			temp_file_name = temp_file.name 

			# Print the name of the temporary file (optional, only for visualization purposes)
			if software == "vmd":
				cmd = f"vmd -xyz {temp_file_name}"
			elif software == "pymol":
				cmd = f"pymol {temp_file_name}"
			else:
				cmd = ""
			
			if cmd:
				logging.debug(f"Running command: {cmd}")
				subprocess.run(cmd, shell = True)

	def write(self, path: str) -> bool:
		''' Write data to an XYZ file.

		Parameters
		----------
		path : str
			Path to the XYZ file.
		
		Returns
		-------
		bool
			True if the data was written successfully, False otherwise.
		'''

		try:
			with open(path, "w") as file_pdb:
				file_pdb.write(self.get_text())
			return True
		except Exception as e:
			logging.error(f"Error writing data to file: {e}")
			return False

	def get_distance_matrix(self) -> np.array:
		''' Return the distance matrix.

		Returns
		-------
		np.array
			Distance matrix.
		'''

		return distance.cdist(self.coordinates, self.coordinates, 'euclidean')

	def get_center(self) -> np.array:
		''' Return the center of the molecule.

		Returns
		-------
		np.array
			Center of the molecule.
		'''
		
		return np.mean(self.coordinates, axis=0)

	def move_center_to(self, center: tuple = (0, 0, 0)) -> None:
		''' Move the center of the molecule to a new position.

		Parameters
		----------
		center : tuple, optional
			New center position. Default is (0, 0, 0).
		'''

		current_center = self.get_center()
		translation_vector = np.array(center) - np.array(current_center)
		self.coordinates += translation_vector
		self.xs, self.ys, self.zs = self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
		self.__update_data()

		for i, at in enumerate(self.atoms):
			at.coordinates.x, at.coordinates.y, at.coordinates.z = self.coordinates[i] 

# Functions
###############################################################################
## Private ##

## Public ##

# Aliases
###############################################################################
