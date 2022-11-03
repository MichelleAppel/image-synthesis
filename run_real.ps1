conda activate ML

$project_name = "pipeline_test3"
$run_name = "dataset_608x448_outlines_cycleGAN"
$data_root = "..\simulation-synthesis\output\608x448\labeled_real"

python .\image_synthesis\train.py --project_name $project_name --run_name $run_name --batch_size 4 --modality outlines --n_classes 2 --mode train --data_root $data_root
python .\image_synthesis\train.py --project_name $project_name --run_name $run_name --batch_size 4 --modality outlines --n_classes 2 --mode test --load ./checkpoints/dataset_128_outlines_r/checkpoint_epoch5.pth --save_path .\output\$project_name_$run_name\
python .\image_synthesis\test.py  --project_name $project_name --run_name $run_name --n_images 64 --root ./output/$project_name_$run_name