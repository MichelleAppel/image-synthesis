#!/bin/bash 

img_size='448'

cycleGANrun_name='dataset_'$img_size'_resnet_9blocks'
net_G_path='checkpoints/'$cycleGANrun_name'/40_net_G_A.pth'

project_name='image_synthesis'

run_name='norm'
run_name_fake_noDT=$run_name'_outlines_fake_noDT'
run_name_fake=$run_name'_outlines_ArchVizPro'
run_name_NYUD=$run_name'_outlines_NYUDv2'
run_name_BDSD=$run_name'_outlines_BDSD500'

root='data/'$img_size

data_root_ArchVizPro_train=$root'/fake_labeled100k'
data_root_ArchVizPro_test=$root'/fake_labeled100k_test'

data_root_NYUD_train=$root'/labeled_real'
data_root_NYUD_test=$root'/labeled_real_test'
data_root_NYUD_val=$root'/labeled_real_val'

data_root_BDSD_train=$root'/labeled_BDSD_train'
data_root_BDSD_test=$root'/labeled_BDSD_test'

batch_size=10
steps=40000
model_step=30000

## Train cycleGAN
# python ./cyclegan/train.py --dataroot $root --use_wandb --name $cycleGANrun_name --batch_size 1 --load_size 448 --crop_size 256 --gpu_ids $1 --netG unet_256


# # Edge detection on NYUD dataset
# ## Train
python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_NYUD --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_ArchVizPro_train --gpu $1 --steps $steps

# ### Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_NYUD --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_ArchVizPro_test --load ./checkpoints/$project_name/$run_name_real/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_real/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_NYUD --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_NYUD_test --load ./checkpoints/$project_name/$run_name_real/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_real/real/ --gpu $1


# Edge detection on BDSD dataset
## Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_BDSD --batch_size $batch_size --modality outlines --n_classes 2 --mode train --crop 320 --data_root $data_root_BDSD_train --gpu $1 --steps $steps

# # ### Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_BDSD --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_ArchVizPro_test --load ./checkpoints/$project_name/$run_name_BDSD/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_real/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_BDSD --batch_size $batch_size --modality outlines --n_classes 2 --mode test --crop 320 --data_root $data_root_NYUD_test --load ./checkpoints/$project_name/$run_name_BDSD/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_real/real/ --gpu $1


## Edge detection on fake dataset w/o domain transfer
# # Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_ArchVizPro_train --gpu $1 --steps $steps

# # Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_ArchVizPro_test --load ./checkpoints/$project_name/$run_name_fake_noDT/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_fake_noDT/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake_noDT --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_NYUD_test --load ./checkpoints/$project_name/$run_name_fake_noDT/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_fake_noDT/real/ --gpu $1


# # Edge detection on fake dataset with domain transfer
# Train
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode train --data_root $data_root_ArchVizPro_train --data_root_val $data_root_val --gpu $1 --domain_transfer True --net_G_path $net_G_path --steps $steps

# # Test
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_ArchVizPro_test --load ./checkpoints/$project_name/$run_name_fake/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_fake/fake/ --gpu $1 # --domain_transfer True --net_G_path $net_G_path
# python ./image_synthesis/train.py --project_name $project_name --run_name $run_name_fake --batch_size $batch_size --modality outlines --n_classes 2 --mode test --data_root $data_root_NYUD_test --load ./checkpoints/$project_name/$run_name_fake/checkpoint_step$model_step.pth --save_path ./output/$project_name/$run_name_fake/real/ --gpu $1 

