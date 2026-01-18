#!/usr/bin/env bash
this_dir=$(dirname "$0")

echo "********build fps************"
cd $this_dir/fps
rm -rf build
python setup.py

