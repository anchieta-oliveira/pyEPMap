import sys
import time
import numpy as np
import seaborn as sns
sys.path.append('../')
from pyepm.epm import EPM
from pyepm.MOL.PSF import PSF
from pyepm.MOL.PDB import PDB
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def read_charges_file(charge_file) -> np.array:
    with open(charge_file, "r") as file:
        c = file.readlines()
        charges = np.array(list(map(lambda x: float(x),c)))
        return charges

def run_trp_cage():
    pdb = PDB()
    epm = EPM()

    print("Lendo PDB and PSF...")
    pdb.read("2luf_ph74_autopsf.pdb")
    psf = PSF("2luf_ph74_autopsf.psf")

    ts_cpu = []
    gps_cpu = []
    gs = 1

    while gs >= 0.4:
        print(f"EPMap res: {gs}")
        start_time = time.time() 
        dx = epm.calculate(pdb=pdb, charges=psf.charges, res=gs, gpu=False, margim=0.3, cutoff=30, gpus_id=[0], form="dx")
        end_time = time.time()
        t = end_time - start_time
        gp = dx.xn * dx.yn * dx.zn

        ts_cpu.append(t)
        gps_cpu.append(gp)
        print(f"Time {t} s - {gp} grid points")
        gs -= 0.1
    #dx.write("fim_cpu.dx")

    ts_gpu = []
    gps_gpu = []
    gs = 1
    while gs >= 0.4:
        print(f"EPMap res: {gs}")
        start_time = time.time() 
        dx = epm.calculate(pdb=pdb, charges=psf.charges, res=gs, gpu=True, margim=0.3, cutoff=30, gpus_id=[0], form="dx")
        end_time = time.time()
        t = end_time - start_time
        gp = dx.xn * dx.yn * dx.zn

        ts_gpu.append(t)
        gps_gpu.append(gp)
        print(f"Time {t} s - {gp} grid points")
        gs -= 0.1
    #dx.write("fim_gpu.dx")
    
    plt.figure(figsize=(10,6))
    sns.set(style="whitegrid")

    sns.lineplot(x=[gp / 1e6 for gp in gps_cpu], y=ts_cpu, marker='o', color='b', label='CPU')

    sns.lineplot(x=[gp / 1e6 for gp in gps_gpu], y=ts_gpu, marker='o', color='r', label='GPU')

    plt.title('Benchmark: Execution Time vs. Number of Grid Points (CPU vs GPU)')
    plt.xlabel('Number of Grid Points (millions)')
    plt.ylabel('Execution Time (seconds)')

    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}M'))

    plt.legend()
    plt.savefig("2luf_cpu_gpu.png", dpi=250)
    #plt.show()

    plt.close()

    # Calcula aceleração GPU/CPU
    speedup = np.array(ts_cpu) / np.array(ts_gpu)

    # Configurações do gráfico
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Criando deslocamento para separar as barras
    x = np.arange(len(gps_cpu))  # Índices das barras
    width = 0.4  # Largura das barras

    # Plotando as barras
    plt.bar(x - width/2, ts_cpu, width, label='CPU', color='b')
    plt.bar(x + width/2, ts_gpu, width, label='GPU', color='r')


    for i in range(len(speedup)):
        plt.text( (x[i] - width/2 + x[i] + width/2) / 2, ts_cpu[i] * 1.01,  
                f'×{speedup[i]:.1f}', 
                ha='center', va='bottom', fontsize=10,
                color='black', fontweight='bold')
        
   
    plt.xticks(x, [f"{gp/1e6:.1f}M" for gp in gps_cpu])  
    plt.xlabel('Number of Grid Points (millions)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Benchmark: Execution Time (CPU vs GPU)')

    plt.legend()
    plt.savefig("2luf_cpu_gpu_bar.png", dpi=250)
    #plt.show()



def run_8y3c():
    epm = EPM()
    print("Lendo PDB and Charges...")
    pdb = PDB("8Y3C_protein_autopsf.pdb")
    psf = PSF("8Y3C_protein_autopsf.psf")

    ts_cpu = []
    gps_cpu = []
    gs = 1

    while gs >= 0.4:
        print(f"EPMap gs: {gs}")
        start_time = time.time() 
        dx = epm.calculate(pdb=pdb, charges=psf.charges, res=gs, gpu=False, margim=0.3, cutoff=30, gpus_id=[0], form="dx")
        end_time = time.time()
        t = end_time - start_time
        gp = dx.xn * dx.yn * dx.zn

        ts_cpu.append(t)
        gps_cpu.append(gp)
        print(f"Grid spacing {t} s - {gp} grid points")
        gs -= 0.1
    #dx.write(f"1o0h_gs{gs}_cpu_mep.dx")

    ts_gpu = []
    gps_gpu = []
    gs = 1
    while gs >= 0.4:
        print(f"EPMap gs: {gs}")
        start_time = time.time() 
        dx = epm.calculate(pdb=pdb, charges=psf.charges, res=gs, gpu=True, margim=0.3, cutoff=30, gpus_id=[0], form="dx")
        end_time = time.time()
        t = end_time - start_time
        gp = dx.xn * dx.yn * dx.zn

        ts_gpu.append(t)
        gps_gpu.append(gp)
        print(f"Grid spacing {t} s - {gp} grid points")
        gs -= 0.1

    #dx.write(f"1o0h_gs{gs}_gpu_mep.dx")
    
    plt.figure(figsize=(10,6))
    sns.set(style="whitegrid")

    sns.lineplot(x=[gp / 1e6 for gp in gps_cpu], y=ts_cpu, marker='o', color='b', label='CPU')

    sns.lineplot(x=[gp / 1e6 for gp in gps_gpu], y=ts_gpu, marker='o', color='r', label='GPU')

    plt.title('Benchmark: Execution Time vs. Number of Grid Points (CPU vs GPU) ')
    plt.xlabel('Number of Grid Points (millions)')
    plt.ylabel('Execution Time (seconds)')

    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}M'))

    plt.legend()
    plt.savefig("1o0h_cpu_gpu.png", dpi=250)

    plt.close()

    # Calcula aceleração GPU/CPU
    speedup = np.array(ts_cpu) / np.array(ts_gpu)

    # Configurações do gráfico
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Criando deslocamento para separar as barras
    x = np.arange(len(gps_cpu))  # Índices das barras
    width = 0.4  # Largura das barras

    # Plotando as barras
    plt.bar(x - width/2, ts_cpu, width, label='CPU', color='b')
    plt.bar(x + width/2, ts_gpu, width, label='GPU', color='r')


    for i in range(len(speedup)):
        plt.text( (x[i] - width/2 + x[i] + width/2) / 2, ts_cpu[i] * 1.01,  
                f'×{speedup[i]:.1f}', 
                ha='center', va='bottom', fontsize=10,
                color='black', fontweight='bold')
        
   
    plt.xticks(x, [f"{gp/1e6:.1f}M" for gp in gps_cpu])  
    plt.xlabel('Number of Grid Points (millions)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Benchmark: Execution Time (CPU vs GPU)')

    plt.legend()
    plt.savefig("1o0h_cpu_gpu_bar.png", dpi=250)

# Run...
run_trp_cage()
#run_8y3c()