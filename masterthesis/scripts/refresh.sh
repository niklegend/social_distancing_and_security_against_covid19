#!/bin/bash

__conda_setup="$('/Users/mattiavandi/anaconda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/mattiavandi/anaconda/etc/profile.d/conda.sh" ]; then
        . "/Users/mattiavandi/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/Users/mattiavandi/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup

if [ "$#" -eq "0" ]; then
  envname=$CONDA_DEFAULT_ENV
else
  envname=$1
fi

echo "Activating $envname"
conda activate $envname

PACKAGE_NAME='masterthesis'

if [[ "$(pip freeze | grep '$PACKAGE_NAME' | wc -l)" > "0" ]]; then
    echo "Uninstalling $PACKAGE_NAME"
    pip3 uninstall $PACKAGE_NAME -y
fi

echo "Installing Master Thesis"
pip3 install .

conda deactivate
