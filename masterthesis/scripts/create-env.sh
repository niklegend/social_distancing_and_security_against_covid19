#!/bin/bash

set -e

if [[ $# -eq 1 ]] ; then
    envname=$1
else
    envname=vandi_mattia_tesi
fi

echo "Creating environment $envname..."

TEMPFILE="environment-$(date '+%Y%m%d_%s').yml"
sed "s/ENVNAME/$envname/" environment.yml > $TEMPFILE

conda env create -f $TEMPFILE

rm $TEMPFILE
