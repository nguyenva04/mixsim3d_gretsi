#!/bin/bash

#topaze user to be adjusted
user=lecomtej
direct_copy=true
#dst_path="/ccc/scratch/cont013/ifpr11/lecomtej/dataset/"
dst_path="/R11-data/drp/dataset/2024-01-18/"

numbers=('4435' '4444' '4445' '4451' '4483' '4515')
for n in ${numbers[@]}; do
    cube_path="/mnt/r11pocdrp/data/heatmap/"$n
    csvfile="$(ls $cube_path/*uniform*)"
    datfile="$(ls $cube_path/*16bit*)"
    echo cp $datfile
    if [ "$direct_copy" = true ] ; then 
        dstfile="${dst_path}"$n
        mkdir $dstfile
        cp "$csvfile" "$datfile" "$dstfile"
    else
        dstfile="${user}@topaze.ccc.cea.fr:${dst_path}"$n"/"
        sshpass -f filepass scp "$csvfile" "$datfile" "$dstfile"
    fi
done