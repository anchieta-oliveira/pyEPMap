# -*- coding: utf-8 -*-
##############################
#Modulos

#############################

"""
 Sym= symmetry_label_1
 Ene= mo_energy_1
 Spin= (Alpha|Beta)
 Occup= mo_occupation_number_1
 ao_number_1 mo_coefficient_1
"""

class MO: 
	#A ideia é fazer classes para AO, MO, GTO, Atoms (já existe) e conectar todas pelo index do atomo
	def __init__(self, 
			  	title:str = "", 
				id:int = 0,
				energy:float = .0, 
				spin:str = "",
				occupation:int = 0 
			  ):
		self.title:str = title
		self.id:int = id
		self.energy:float = energy
		self.spin:str = spin
		self.occupation:int = occupation
		self.ao_number:list = []
		self.coefficients:list = []
		self.symtype:list = []
		self.ao_atomindex:list = []


		 