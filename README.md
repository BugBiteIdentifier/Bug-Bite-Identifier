# Bug-Bite-Identifier

## Setup

### Create a virtual environment: Anaconda method
Create a virtual environment to install all dependencies. May require Anaconda install. Run the following commands

conda env create -f env.yml
conda activate my_env

### Create a virtual environment: pip method
Create a virtual environment using pip. Chage env.yml to requirements.txt. I am not certain if this will work. Run the following command. Activation of virtual environment is dependent on OS.

python -m venv venv

pip install -r requirements.txt

### Run the Code
python BugBiteIdentifier.py