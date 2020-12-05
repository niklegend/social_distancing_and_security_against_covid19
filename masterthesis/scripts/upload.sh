#!/bin/bash

if [ "$#" != "1" ]; then
    echo "usage: ${0##*/} <pathspec>" >&2
    exit 1
fi

dest_dir="$1"

if [[ ! -d $dest_dir ]]; then
    echo "$dest_dir either does not exist or is a file" >&2
    exit 1
fi

src_dir="$(git root)"

source "$src_dir/scripts/replace.sh"

files=(masterthesis setup.py requirements.txt)

# Remove python cache files and compiled python files
find "$src_dir" -type d -name '__pycache__' -exec rm -rf "{}" \; 2> /dev/null
#find "$src_dir" -type f -name '*.pyc' -exec rm -f "{}" \; 2> /dev/null

for file in "${files[@]}"; do
    echo "Uploading $file"
    replace "$src_dir/$file" "$dest_dir"
    echo "$file uploaded"

    unset file
done

unset files
unset src_dir
unset dest_dir
