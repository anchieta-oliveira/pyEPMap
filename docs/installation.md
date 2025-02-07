    <div style="text-align: center;">
        <h1>Tutorial</h1>
    </div>

## Prerequisites
Ensure you have the following installed on your system:
- [Python 3.8+](https://www.python.org/downloads/)
- [Conda Package Manager](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- (Optional) Git - for cloning the repository

## Installation Guide 

### Step 1: Clone the Repository
Use Git to clone the repository to your local machine:
```bash
git clone https://github.com/anchieta-oliveira/pyepm
```
### Step 2: Create a virtual environment dedicated to LabMonitor
Navigate to the cloned Git folder:
```bash
cd pyepm
```
Create a new Conda environment using the provided requirements.yml file:
```bash
conda env create -f environment.yml
```
### Step 3: Then install the package
Optionally, install install the package
```bash
python -m pip install -e .
```