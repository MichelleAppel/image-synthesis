conda activate ML

$run_name = "fake2real_608x448"
$root = "..\simulation-synthesis\output\608x448\"

python .\cyclegan\train.py --dataroot $root --use_wandb --name $run_name --batch_size 1 --load_size 448 --crop_size 448 #--preprocess none  