conda activate ML

$img_size = "448"

$cycleGANrun_name = "TEST4_"+$img_size
$net_G_path = "checkpoints\"+$cycleGANrun_name+"\latest_net_G_A.pth"

$project_name = "outlines_TEST"

$run_name_fake = "__dataset_"+$img_size+"_outlines_fake"
$run_name_real = "__dataset_"+$img_size+"_outlines_real"

$root = "C:\Users\appel\Documents\Project\simulation-synthesis\output\448"
$data_root_fake_train = $root + "\labeled_fake"
$data_root_real_train = $root + "\labeled_real"
$data_root_fake_test  = $root + "\labeled_fake_test"
$data_root_real_test  = $root + "\labeled_real_test"

$batch_size = 1

# python .\cyclegan\train.py --dataroot $root --use_wandb --name $cycleGANrun_name --batch_size 1 --load_size 448 --crop_size 448 --save_latest_freq 5

# python .\image_synthesis\train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode train --domain_transfer True --data_root $data_root_fake_train --net_G_path $net_G_path 
# python .\image_synthesis\train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_fake_test --load ./checkpoints/$run_name_fake/checkpoint_epoch5.pth --save_path .\output\$project_name_$run_name_fake\ # --domain_transfer True --net_G_path $net_G_path
# python .\image_synthesis\test.py  --project_name $project_name --run_name $run_name_fake --n_images 64 --root .\output\$project_name_$run_name_fake\

# python .\image_synthesis\train.py --project_name $project_name --run_name $run_name_real --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_real_train -s 0.05
# python .\image_synthesis\train.py --project_name $project_name --run_name $run_name_real --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_fake_test --load ./checkpoints/$run_name_real/checkpoint_epoch5.pth --save_path .\output\$project_name_$run_name_real\ # --domain_transfer True --net_G_path $net_G_path
# python .\image_synthesis\test.py  --project_name $project_name --run_name $run_name_real --n_images 64 --root .\output\$project_name_$run_name_real\