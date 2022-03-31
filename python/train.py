import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchmetrics

from tqdm import tqdm

import wandb

from synthesis_dataset import SynthesisDataset
from dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

from utils import AP, IOU

torch.manual_seed(42)
torch.cuda.manual_seed(42)

dir_checkpoint = Path('./checkpoints/')

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    dataset = SynthesisDataset("..\simulation-synthesis\output\MLDataset_128rot", scale=args.scale, extension='.png', do_domain_transfer=args.domain_transfer)
    dataset.modalities = ['img', args.modality]

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project=args.project_name, name=args.run_name, entity="michelleappel")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Regression:      {args.regression}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    if args.regression:
        criterion = nn.MSELoss()
        type=torch.float32
    else:
        criterion = nn.CrossEntropyLoss()
        type = torch.long
        
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['img']
                true_masks = batch[args.modality]
                if args.modality in ['outlines', 'class']:
                    true_masks = true_masks[:, 0, :, :]
                if args.modality == 'depth':
                    true_masks = true_masks / 35000

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=type)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if args.modality in ['normals', 'depth']:
                        loss = criterion(masks_pred, true_masks)
                    else:
                        loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, args.modality)

                        if args.regression:
                            val_score = 1-val_score

                        scheduler.step(val_score)

                        if args.modality == 'class':
                            true = dataset.class_to_color(true_masks.unsqueeze(1)).float().cpu()[0]
                            pred = dataset.class_to_color(torch.softmax(masks_pred, dim=1).argmax(dim=1).unsqueeze(1))[0].float().cpu()
                            conf = torch.softmax(masks_pred, dim=1).max(dim=1)[0][0].float().cpu()
                        elif args.modality == 'normals' or args.modality == 'depth':
                            true = true_masks[0].float().cpu()
                            pred = masks_pred[0].float().cpu()
                            conf = masks_pred[0].float().cpu()
                        else:
                            true = true_masks[0].float().cpu()
                            pred = torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
                            conf = 1-torch.softmax(masks_pred, dim=1)[0].float().cpu()

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true),
                                'pred': wandb.Image(pred),
                                'conf': wandb.Image(conf)
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

def test_net(net,
              device,
              batch_size: int = 1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):


    print('testing...')
    if args.modality == 'outlines':
        pass

    if args.regression:
        criterion = nn.MSELoss()
        type=torch.float32
    else:
        criterion = nn.CrossEntropyLoss()
        type = torch.long

    dataset = SynthesisDataset("..\simulation-synthesis\output\MLDataset_128rot", scale=args.scale, extension='.png', do_domain_transfer=args.domain_transfer)
    dataset.modalities = ['img', args.modality]

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project=args.project_name, name=args.run_name+'_test', entity="michelleappel")
    experiment.config.update(dict(batch_size=batch_size, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    # store metrics
    net.eval()

    for step, batch in tqdm(enumerate(test_loader)):
        images = batch['img']
        true_masks = batch[args.modality]
        if args.modality in ['outlines', 'class']:
            true_masks = true_masks[:, 0, :, :]
        if args.modality == 'depth':
            true_masks = true_masks / 35000

        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=type)

        aps1 = []
        aps3 = []
        aps6 = []

        F1_1s = []
        F1_3s = []
        F1_6s = []

        with torch.cuda.amp.autocast(enabled=amp):
            masks_pred = net(images).detach()
                
            if args.modality == 'class':
                true = dataset.class_to_color(true_masks.unsqueeze(1)).float().cpu()[0]
                pred = dataset.class_to_color(torch.softmax(masks_pred, dim=1).argmax(dim=1).unsqueeze(1))[0].float().cpu()
                conf = torch.softmax(masks_pred, dim=1).max(dim=1)[0][0].float().cpu()
            elif args.modality == 'normals' or args.modality == 'depth':
                true = true_masks[0].float().cpu()
                pred = masks_pred[0].float().cpu()
                conf = masks_pred[0].float().cpu()
            else:
                true = true_masks[0].float().cpu()
                pred = torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
                conf = 1-torch.softmax(masks_pred, dim=1)[0].float().cpu()

            experiment.log({
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(true),
                    'pred': wandb.Image(pred),
                    'conf': wandb.Image(conf)
                }
            })

            if args.modality in ['normals', 'depth']:
                # appropriate metrics: MSE?
                pass
            elif args.modality == 'outlines':
                ap1, F1_1 = AP(pred, true_masks, pos_label=1, matching_distance=1)
                ap3, F1_3 = AP(pred, true_masks, pos_label=1, matching_distance=3)
                ap6, F1_6 = AP(pred, true_masks, pos_label=1, matching_distance=6)

                aps1 += [ap1]
                aps3 += [ap3]
                aps6 += [ap6]

                F1_1s += [F1_1]
                F1_3s += [F1_3]
                F1_6s += [F1_6]

                experiment.log({'AP 1': ap1})
                experiment.log({'AP 3': ap3})
                experiment.log({'AP 6': ap6})

                experiment.log({'F1 1': F1_1})
                experiment.log({'F1 3': F1_3})
                experiment.log({'F1 6': F1_6})
                
                # appropriate metrics: ODF, OIF, AP, IOU
            else:
                # IOU
                pass

        if step > 10:
            break

        if args.modality == 'outlines':
            wandb.run.summary["AP total 1"] = sum(aps1)/len(aps1)
            wandb.run.summary["AP total 3"] = sum(aps3)/len(aps3)
            wandb.run.summary["AP total 6"] = sum(aps6)/len(aps6)
            
            wandb.run.summary["F1 total 1"] = sum(F1_1s)/len(F1_1s)
            wandb.run.summary["F1 total 3"] = sum(F1_3s)/len(F1_3s)
            wandb.run.summary["F1 total 6"] = sum(F1_6s)/len(F1_6s)
        elif args.modality == ''


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--mode', '-m', type=str, default='train', help='test or train')
    parser.add_argument('--domain_transfer', '-d', type=bool, default=False, help='domain transfer from fake to real')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--run_name', default=None, help='Wandb run name')
    parser.add_argument('--project_name', default=None, help='Wandb project name')
    parser.add_argument('--modality', default='outlines', help='Modalities to predict, e.g. outlines, class, normals, depth')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--regression', type=bool, default=False, help='Regression or classification')

    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    # parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.n_classes, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    print('DT', args.domain_transfer)

    net.to(device=device)
    if args.mode == 'train':
        try:
            train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100,
                    amp=args.amp)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)
    else:
        try:
            test_net(net=net,
                    batch_size=args.batch_size,
                    device=device,
                    img_scale=args.scale,
                    amp=args.amp)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)

