conda activate ML

$project_name = "pipeline_test3"
$run_name = "dataset_608x448_outlines_cycleGAN"
$data_root = "..\simulation-synthesis\output\608x448\labeled_fake"

python .\image_synthesis\train.py --project_name $project_name --run_name $run_name --batch_size 2 --modality outlines --n_classes 2 --mode train --domain_transfer True --data_root $data_root
python .\image_synthesis\train.py --project_name $project_name --run_name $run_name --batch_size 2 --modality outlines --n_classes 2 --mode test --load ./checkpoints/dataset_128_outlines/checkpoint_epoch5.pth --save_path .\output\$project_name_$run_name\ --domain_transfer True
python .\image_synthesis\test.py  --project_name $project_name --run_name $run_name --n_images 64 --root ./output/$project_name_$run_name