#!/bin/bash

root="$(git root)"
dest_dir="$root/dist"

files=(masterthesis setup.py) # requirements.txt
find $root -type d -name '__pycache__' -exec rm -rf "{}" \; 2> /dev/null

if [[ -e $dest_dir ]]; then
    rm -r $dest_dir
fi

mkdir $dest_dir

echo ${files[@]}
zip -r $dest_dir/masterthesis.zip ${files[@]}
