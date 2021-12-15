#!/bin/bash

set -vx

#
# Generate a cookiecutter version of this repository
#

# remove the folders we do not want to copy
rm -rf .tox
rm -rf seedproject.egg-info
rm -rf cifar10.lock
rm -rf seedproject/__pycache__
rm -rf seedproject/models/__pycache__
rm -rf seedproject/tasks/__pycache__

dest=$(mktemp -d)

# Get the latest version of the cookiecutter
git clone git@github.com:Delaunay/ml-seed.git $dest

# Copy the current version of our code in the cookiecutter
rsync -av --progress . $dest/'{{cookiecutter.project_name}}'/   \
    --exclude .git                                              \
    --exclude __pycache__

# The basic configs
cat > $dest/cookiecutter.json <<- EOM
    {
        "project_name": "myproject",
        "author": "Anonymous",
        "email": "anony@mous.com",
        "description": "Python seed project for productivity",
        "copyright": "2021",
        "url": "http://github.com/test",
        "version": "version",
        "license": "BSD 3-Clause License",
        "_copy_without_render": [
            ".github"
        ]   
    }
EOM

cat > $dest/README.rst <<- EOM
Machine Learning Seed Repository
===============================

Features
~~~~~~~~

* Sphinx doc generation

  * Read the docs ready

* Github CI

  * Test Coverage + Doctest
  * black formating
  * pylint
  * isort
  * docs8

* slurm launch script

* pip installable

Get started
~~~~~~~~~~~

.. code-block:: bash

    pip install cookiecutter
    cookiecutter https://github.com/Delaunay/ml-seed
    

Automation
~~~~~~~~~~

Auto format your code before pushing

.. code-block:: bash

    tox -e run-block

    tox -e run-isort
EOM


COOKIED=$dest/'{{cookiecutter.project_name}}'/

# Copy the current version of our code in the cookiecutter
cp -a . $COOKIED

# Remove the things we do not need in the cookie
rm -rf $COOKIED/scripts/generate_cookie.sh
rm -rf $COOKIED/.git

# Find the instance of all the placeholder variables that
# needs to be replaced by their cookiecutter template

cat > mappings.json <<- EOM
    [
        ["seedproject", "project_name"],
        ["seeddescription", "author"],
        ["seed@email", "email"],
        ["seeddescription", "description"],
        ["seedcopyright", "copyright"],
        ["seedurl", "url"],
        ["seedversion", "version"],
        ["seedlicense", "license"]
    ]
EOM

jq -c '.[]' mappings.json | while read i; do
    oldname=$(echo "$i" | jq -r -c '.[0]')
    newname=$(echo "$i" | jq -r -c '.[1]')

    echo "Replacing $oldname by $newname"
    find $COOKIED -print0 | xargs -0 sed -e "s/$oldname/\{\{cookiecutter\.$newname\}\}/g"
done

rm -rf mappings.json

# Push the change
#   use the last commit message of this repository 
#   for the  cookiecutter
PREV=$(pwd)
MESSAGE=$(git show -s --format=%s)

cd $dest
git checkout -b auto
git add --all
git commit -m "$MESSAGE"
git push origin auto

# Remove the folder
cd $PREV
rm -rf $dest
