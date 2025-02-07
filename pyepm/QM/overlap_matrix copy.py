# -*- coding: utf-8 -*-
##############################
#Modulos
import numpy as np

#############################

class OverlapMatrix:
  def __init__(self, title="", lower_half_triangle:list = [], liner:list = [], matrix:list = []):
    self.title:str = ""
    self.lower_half_triangle = lower_half_triangle
    self.liner = liner
    self.matrix = matrix
  
  def overlap_ij(self, Ri, exp_i, Rj, exp_j):
    """
    Calcula a sobreposição entre duas funções gaussianas primitivas 
    Parâmetros:
    - Ri, Rj: Vetores de posição dos centros gaussianos (np.array([x, y, z])).
    - exp_i, exp_j: Exponenciais das funções gaussianas primitivas  
    Retorna:
    - Valor da sobreposição entre as duas funções gaussianas primitivas.
    """
    p = exp_i + exp_j
    alpha = exp_i * exp_j / p
    R = Ri - Rj  
    # Coeficientes de normalização
    ai = (2 * exp_i / np.pi)**0.75
    aj = (2 * exp_j / np.pi)**0.75
    #ai = m.GTOs[0].contraction_coefficient[0][0] 
    #aj = m.GTOs[0].contraction_coefficient[0][0]   
    result = ai * aj * (np.pi / p)**1.5 * np.exp(-alpha * np.dot(R, R))
    return result
  
  def make_matrix(self, GTOs:list, atoms:list):
    id_i = 1 
    id_j = 1
    """for gto_i in GTOs:
      for exp_i_shell in gto_i.exponent_primitive:
        for exp_i in exp_i_shell:
          for gto_j in GTOs:
            for exp_j_shell in gto_j.exponent_primitive:
              for exp_j in exp_j_shell:
                atom_i = np.array(atoms[gto_i.atom_number-1].coordinates.get_tuple())
                atom_j = np.array(atoms[gto_j.atom_number-1].coordinates.get_tuple())
                print(id_i, id_j, self.overlap_ij(Ri=atom_i, exp_i=exp_i, exp_j=exp_j, Rj=atom_j))
                id_j +=1
            id_j = 1
          id_i += 1"""
    

  def gaussian_overlap(self, alpha_i, R_i, l_i, alpha_j, R_j, l_j):
    """
    Calcula a sobreposição direta entre dois GTOs.
    Parâmetros:
    - alpha_i, alpha_j: Expoentes dos GTOs
    - R_i, R_j: Vetores de posição dos centros dos GTOs
    - l_i, l_j: Números quânticos azimutais (0 para s, 1 para p, 2 para d, etc.)
    Retorna:
    - S: Sobreposição direta
    """
    prefactor = (np.pi / (alpha_i + alpha_j))**(3/2) * np.exp(
        - (alpha_i * alpha_j) / (alpha_i + alpha_j) * np.dot(R_i - R_j, R_i - R_j)
    )
    if l_i == 0 and l_j == 0:  # S orbitais
        return prefactor
    elif l_i == 1 and l_j == 0:  # p_x orbital
        return R_i[0] * prefactor
    elif l_i == 0 and l_j == 1:  # p_x orbital
        return R_j[0] * prefactor
    elif l_i == 1 and l_j == 1:  # p_y orbital
        return R_i[1] * R_j[1] * prefactor
    # Adicione mais casos para outros tipos de orbitais conforme necessário



  def read_from_multiwfn(self, path:str) -> list:
    with open(path, "r") as file:
      lines = file.readlines()
    matrix = []
    matrix_lin = []
    num_col = 0
    for line in lines:
      line_split = line.split()
      if "*" in line:
         pass
      elif " \n" == line:
        break
      elif line[:6] == "      ":
        num_col = int(len(line_split))
        init_col = int(line_split[0])
        for n in range(num_col):
          matrix.append([])
        
      else:
        for id_col, v_col in enumerate(line_split[1:]):
          matrix[init_col+id_col-1].append(float(v_col))
          matrix_lin.append(float(v_col))
    self.liner = matrix_lin
    self.lower_half_triangle = matrix

    n = len(matrix)
    # Inicialize uma matriz completa com zeros
    self.matrix  = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n - i):
            self.matrix[i][i + j] = self.lower_half_triangle[i][j]
            self.matrix[i + j][i] = self.lower_half_triangle[i][j]

    return self.matrix 
           
           
      

      


