

seeds=(0 1 2 3 4 5 6 7 8 9)
for seed in "${seeds[@]}";do
 python main_tri_mc.py --res_dir 'yy-mm-dd' --lr 0.0001 --seed "$seed"  --batch-size 1000  --hidden-channels 256\
   --epoch-num 200  --tau 1000 --lam1 0.1 --lam2 0.1 --data-name 'ms' >> ms_all_seed.txt
done
python get_results.py --res_dir "ms yy-mm-dd"

seeds=(0 1 2 3 4 5 6 7 8 9)
for seed in "${seeds[@]}";do
 python main_tri_mc.py --res_dir 'yy-mm-dd' --lr 0.0005 --seed "$seed"  --batch-size 5000  --hidden-channels 256\
   --epoch-num 150  --tau 1000  --lam1 0.1 --lam2 0.1 --data-name 'drkg' >> drkg_all_seed.txt
done
python get_results.py --res_dir "drkg yy-mm-dd"
