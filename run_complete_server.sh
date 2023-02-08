#!/bin/bash 

img_size='448'

cycleGANrun_name='dataset_'$img_size'_resnet_9blocks'
net_G_path='checkpoints/'$cycleGANrun_name'/20_net_G_A.pth'

project_name='image_synthesis'

run_name='dataset_'$img_size
run_name_fake_noDT='dataset_'$img_size'_outlines_fake_noDT'
run_name_fake='dataset_'$img_size'_outlines_fake'
run_name_real='dataset_'$img_size'_outlines_real'

root='data/'$img_size
# data_root_fake_train=$root'/labeled_fake_test' # small dataset to test architecture
data_root_fake_train=$root'/fake_labeled100k'
data_root_real_train=$root'/labeled_real'
data_root_fake_test=$root'/fake_labeled100k_test'
data_root_real_test=$root'/labeled_real_test'

batch_size=4

# Train cycleGAN
python ./cyclegan/train.py --dataroot $root --use_wandb --name $cycleGANrun_name --batch_size 1 --load_size 448 --crop_size 448 --gpu_ids $1 --netG resnet_9blocks

# Edge detection on real dataset
### Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_real --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_real_train --gpu $1 --epochs 25 --learning_rate 0.0001

# ### Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_real --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_fake_test --load ./checkpoints/$project_name/$run_name_real/checkpoint_epoch7.pth --save_path ./output/$project_name/$run_name_real/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_real --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_real_test --load ./checkpoints/$project_name/$run_name_real/checkpoint_epoch7.pth --save_path ./output/$project_name/$run_name_real/real/ --gpu $1

### Evaluation
# python ./image_synthesis/test.py  --project_name $project_name --run_name $run_name_real --n_images 100 --root $data_root_fake_test



# Edge detection on fake dataset w/o domain transfer
### Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_fake_train --gpu $1

### Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_fake_test --load ./checkpoints/$project_name/$run_name_fake_noDT/checkpoint_epoch5.pth --save_path ./output/$project_name/$run_name_fake_noDT/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_real_test --load ./checkpoints/$project_name/$run_name_fake_noDT/checkpoint_epoch5.pth --save_path ./output/$project_name/$run_name_fake_noDT/real/ --gpu $1

### Evaluation
# python ./image_synthesis/test.py  --project_name $project_name --run_name $run_name_fake_noDT --n_images 100 --root $data_root_fake_test




# Edge detection on fake dataset with domain transfer
### Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_fake_train --gpu $1 --domain_transfer False --net_G_path $net_G_path

### Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_fake_test --load ./checkpoints/$project_name/$run_name_fake/checkpoint_epoch5.pth --save_path ./output/$project_name/$run_name_fake/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_real_test --load ./checkpoints/$project_name/$run_name_fake/checkpoint_epoch5.pth --save_path ./output/$project_name/$run_name_fake/real/ --gpu $1 

### Evaluation
# python ./image_synthesis/test.py  --project_name $project_name --run_name $run_name_fake --n_images 100 --root $data_root_fake_test
