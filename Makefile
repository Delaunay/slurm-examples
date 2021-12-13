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
#
#

jobname = output-magic.txt

single-gpu:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --time=10:00 --gres=gpu:1 --cpus-per-gpu=4 --mem=16G scripts/single-gpu.sh seedproject/train_normal.py -vvv --cuda
	tail -f $(jobname)

multi-gpu:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --time=10:00 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G scripts/multi-gpu.sh seedproject/train_normal.py -vvv --cuda
	tail -f $(jobname)

multi-node:
	rm -rf $(jobname)
	touch $(jobname)
	sbatch -o $(jobname) --nodes 3 --time=10:00 --gres=gpu:4 --cpus-per-gpu=4 --mem=16G scripts/multi-nodes.sh seedproject/train_normal.py -vvv --cuda
	tail -f $(jobname)
