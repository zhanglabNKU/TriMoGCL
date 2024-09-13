seeds=(0 1 2 3 4 5 6 7 8 9)
for seed in "${seeds[@]}";do
 python main_tri_binary.py --res_dir 'yy-mm-dd' --lr 0.0001  --batch_num 10  --hidden-channels 256\
   --epoch-num 150 --tau 1000  --lam 0.1 --data-name ms --seed "$seed"
done

seeds=(0 1 2 3 4 5 6 7 8 9)
for seed in "${seeds[@]}";do
 python main_tri_binary.py --res_dir 'yy-mm-dd' --lr 0.0001  --batch_num 10  --hidden-channels 256\
   --epoch-num 150 --tau 1000  --lam 0.1 --data-name drkg --seed "$seed"
done