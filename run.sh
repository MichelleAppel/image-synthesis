#!/bin/bash

python .\python\train.py --project_name pipeline_test --run_name dataset_128_outlines --batch_size 4 --modality outlines --n_classes 2 --mode train
python .\python\train.py --batch_size 4 --modality outlines --n_classes 2 --mode test --load ./checkpoints/pipeline_test/checkpoint_epoch5.pth --save_path .\output\dataset_128_outlines\
python .\python\test.py --project_name pipeline_test --run_name dataset_128_outlines --n_images 64 --root ./output/dataset_128_outlines