#!/bin/bash

set -evx

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

# dest=$(mktemp -d)
dest=../ml-seed

# Get the latest version of the cookiecutter
git clone git@github.com:Delaunay/ml-seed.git $dest

# Copy the current version of our code in the cookiecutter
COOKIED=$dest/'{{cookiecutter.project_name}}'

rsync -av --progress . $COOKIED/                                \
    --exclude .git                                              \
    --exclude __pycache__

# The basic configs
cat > $dest/cookiecutter.json <<- EOM
    {
        "project_name": "myproject",
        "PROJECT_NAME": "MY_PROJECT",
        "author": "Anonymous",
        "github_nickname": "githubacct",
        "github_repo": "reponame",
        "email": "anony@mous.com",
        "description": "Python seed project for productivity",
        "copyright": "2021",
        "url": "http://github.com/github/project",
        "version": "version",
        "license": "BSD 3-Clause License",
        "_copy_without_render": [
            ".github"
        ]   
    }
EOM

# Remove the things we do not need in the cookie
rm -rf $COOKIED/scripts/generate_cookie.sh
rm -rf $COOKIED/.git
 
# Find the instance of all the placeholder variables that
# needs to be replaced by their cookiecutter template

cat > mappings.json <<- EOM
    [
        ["seedproject", "project_name"],
        ["seedauthor", "author"],
        ["seedlicense", "license"],
        ["seed@email", "email"],
        ["seeddescription", "description"],
        ["seedcopyright", "copyright"],
        ["seedurl", "url"],
        ["seedversion", "version"],
        ["seedgithub", "github_nickname"],
        ["seedrepo", "github_repo"],
        ["SEEDPROJECT", "PROJECT_NAME"]
    ]
EOM

jq -c '.[]' mappings.json | while read i; do
    oldname=$(echo "$i" | jq -r -c '.[0]')
    newname=$(echo "$i" | jq -r -c '.[1]')

    echo "Replacing $oldname by $newname"
    find $COOKIED -type f -print0 | xargs -0 sed -i -e "s/$oldname/\{\{cookiecutter\.$newname\}\}/g"
done

# Move project folder with its new name
rsync -av --remove-source-files --progress $COOKIED/seedproject/ $COOKIED/'{{cookiecutter.project_name}}'/

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
# git checkout master
# git branch -D auto

# Remove the folder
cd $PREV
