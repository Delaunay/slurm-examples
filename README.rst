seedproject
=============================

.. image:: https://readthedocs.org/projects/seedrepo/badge/?version=latest
   :target: https://seedrepo.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://github.com/seedgithub/seedrepo/actions/workflows/test.yml/badge.svg
   :target: https://github.com/seedgithub/seedrepo/actions/workflows/test.yml
   :alt: Tests
      

.. image:: https://codecov.io/gh/seedgithub/seedrepo/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/seedgithub/seedrepo
   :alt: Coverage


Features
~~~~~~~~

* Environment setup scripts

  * miniconda
  * virtualenv

* Generic  Slurm launch scripts

  * Single GPU
  * Multi GPU
  * Multi Node
  * Hyperparameter Optimization

* Checkpoints
* Dataset Local Copy
* Dataset splits (train, valid, test)
* Hyper parameter search with Orion


Getting Started
~~~~~~~~~~~~~~~

Use this as a cookiecutter

.. code-block:: bash

   cookiecutter https://github.com/Delaunay/ml-seed


Layout
~~~~~~

.. code-block:: bash

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
   ├── tests                     # testing
   │   ├── test_model.py 
   |   └── test_loader.py
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

   pip install git+https://github.com/seedgithub/seedrepo

