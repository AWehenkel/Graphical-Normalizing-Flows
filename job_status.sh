#!/usr/bin/env bash

jobs="$(squeue -u awehenkel -h -o '%.18i' | sed -r 's/^[[:space:]]*//')"
echo "--------------------- Displaying jobs running's results -----------------------"
while IFS= read -r line; do
    echo "--------------------------------------------------------"
    echo "$(squeue -u awehenkel -h -j $line)"
    echo "$(grep 'Namespace' *$line*)"
    echo "$(grep 'epoch:' *$line* | tail -n $1)"
done <<< "$jobs"