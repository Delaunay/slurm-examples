install:
	pip install -e .[all]
	pip install -r requirements.txt
	pip install -r docs/requirements.txt
	pip install -r tests/requirements.txt

doc: build-doc

build-doc:
	sphinx-build -W --color -c docs/ -b html docs/ _build/html

serve-doc:
	sphinx-serve

update-doc: build-doc serve-doc


#
#	Launch helpers
#

jobname = output-magic.txt
resouces = --cpus-per-gpu=2 --mem-per-gpu=16G
trainscript = seedproject/train_normal.py -vvv --cuda

conda-setup:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --gres=gpu:1 --cpus-per-gpu=1 --mem-per-gpu=1G scripts/setup_conda.sh
	tail -f $(jobname)

venv-setup:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --gres=gpu:1 --cpus-per-gpu=1 --mem-per-gpu=1G scripts/setup_virtualenv.sh
	tail -f $(jobname)

hpo:
	# 100 Jobs with 20 in parallel max
	# 1 GPU | 4 CPU | 16 Go RAM
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --array=1%20 --gres=gpu:1 $(resouces) scripts/hpo.sh $(trainscript)
	tail -f $(jobname)

single-gpu:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --time=10:00 --gres=gpu:1 $(resouces) scripts/single-gpu.sh $(trainscript)
	tail -f $(jobname)

multi-gpu:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --time=10:00 --gres=gpu:2 $(resouces) scripts/multi-gpu.sh $(trainscript)
	tail -f $(jobname)

multi-node:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --time=10:00 --nodes 3 --gres=gpu:4 $(resouces) scripts/multi-nodes.sh $(trainscript)
	tail -f $(jobname)
