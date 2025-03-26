!/bin/bash
#MSUB -r ccc_train_drp
#MSUB -c 128
#MSUB -n 1 
#MSUB -T 86400 
#MSUB -A ifp00083@A100 
#MSUB -q a100
#MSUB -m work,scratch,store

ccc_mprun -n 1 -c 128 -C drp3d -E'--ctr-module nvidia' -- env PYTHONPATH=/ccc/scratch/cont013/ifpr11/nguyenvt/drp3d2 python3 $CCCSCRATCHDIR/drp3d2/scripts/train_simclr.py $CCCSCRATCHDIR/dataset/ -e 100 -b 128 -c "/ccc/scratch/cont013/ifpr11/nguyenvt/drp3d2/drp/utils/cf/config_simclr.json"

