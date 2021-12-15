seedproject
=============================

Use this as a cookiecutter

.. code-block:: bash

   cookiecutter 


Layout
~~~~~~

.. code-block::

   <seedproject>/
   ├── .github                   # CI jobs to run on every push
   │   └── workflows
   │       └── test.yml
   ├── docs                      # Sphinx documentation of this package
   │   └── conf.py               
   ├── scripts                   # Helper script for launching
   │   ├── multi-gpu.sh          # tasks with slurms
   │   ├── multi-nodes.sh
   │   ├── single-gpu.sh
   │   └── hpo.sh
   ├── seedproject
   │   ├── conf                  # configurations
   |   |   ├── slurm.yml          
   │   │   └── hydra.yml           
   │   ├── models                # Models
   │   │   ├── mymodel.py        
   │   │   └── lenet.py          
   │   ├── tasks                 # Trainer 
   │   │   ├── classification.py 
   │   │   └── reinforcement.py  
   │   └── train.py              # main train script
   ├── .readthedocs.yml          # how to generate the docs in readthedocs
   ├── LICENSE                   # 
   ├── README.rst                # description of current project
   ├── requirements.txt          # requirements of this package
   ├── setup.py                  # installation configuration
   └── tox.ini                   # used to configure test/coverage


Slurm Cluster
~~~~~~~~~~~~~

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below will launch 100 jobs, each jobs will use 1 GPU with 4 CPU cores and 16Go of RAM.
Each jobs are independant and will work toward finding the best set of Hyperparameters.

.. code-block:: bash

   sbatch --array=0-100 --gres=gpu:1 --cpus-per-gpu=4 --mem=16Go scripts/hpo.sh seedproject/train.py


Multi GPU single node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below schedule a job to run on 3 nodes.
It will use a total of 16 CPUs, 16 Go of RAM and 4 GPUs.

.. code-block:: bash

   sbatch --nodes 1 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G scripts/multi-gpu.sh seedproject/train.py


Multi GPU multiple node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below schedule a job to run on 3 nodes.
It will use a total of 48 CPUs, 48 Go of RAM and 12 GPUs.

.. code-block:: bash

   sbatch --nodes 3 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G scripts/multi-gpu.sh seedproject/train.py


Contributing
~~~~~~~~~~~~

.. code-block:: bash

   pip install git+https://github.com/Delaunay/slurm-examples

