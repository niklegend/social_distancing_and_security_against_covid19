#!/bin/sh

set -e

replace() {
    if [ "$#" != "2" ]; then
        echo "usage: $0 <src> <dest>" >&2
        exit 1
    fi

    src="$1"
    dest="$2"

    if [ ! -d "$dest" ]; then
        echo "destination is not a directory." >&2
        exit 1
    fi

    file=${src##*/}
    destPath=$dest/$file

    if [ -d "$destPath" ]; then
        echo "Removing directory $destPath"
        rm -rf "$destPath"
    elif [ -f "$destPath" ] || [ -L "$destPath" ]; then
        echo "Removing file $destPath"
        rm -f "$destPath"
    fi

    if [ -d "$src" ]; then
        echo "Copying directory $src to $dest"
        cp -R "$src" "$dest"
    else
        echo "Copying $src to $dest"
        cp "$src" "$dest"
    fi
    #ln -sfn "$src" "$dest"
}
