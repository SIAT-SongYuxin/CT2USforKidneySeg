import argparse
import logging
import os
import sys
import torchvision
import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
from eval import eval_net
from unet import UNet

from visdom import Visdom
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
#rom Segnet import SegNet
import FCN
from Focal_Loss import FocalLoss
from dice_loss import DiceCoeff
import semseg.modelloader.deeplabv3 as deeplabv3


dir_img ='/home/syx/kits19-master/CT2US/CT2US_epoch3/slice'#'/home/syx/US_kidney/SZRH/train&val/slice'#'/home/syx/US_kidney/SUGH/png_mini_mini/slice_png'# '/home/syx/kits19-master/png_datasize/style_transfer/GAN_test15/slice/images' #
dir_mask ='/home/syx/kits19-master/CT2US/CT2US_epoch3/mask'#'/home/syx/US_kidney/SZRH/train&val/mask'#'/home/syx/US_kidney/SUGH/png_mini_mini/mask_png'#   '/home/syx/kits19-master/png_datasize/style_transfer/GAN_test15/mask'  #
dir_checkpoint = 'checkpoints/'


import torch.nn as nn
import torch.nn.functional as F


# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#     def forward(self, logits, targets):
#         num = targets.size(0)
#         #// 为了防止除0的发生
#         smooth = 1
#
#         probs = F.sigmoid(logits)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score



def train_net(net,
              device,
              epochs=20,
              batch_size=4,
              lr=0.0001,
              val_percent=0.2,
              save_cp=False,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    transform=A.Compose([A.HorizontalFlip(p=0.5),A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1,p=0.5)])
    train_loader = DataLoader(train,batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    #writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    viz=Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0.], [0.], win='learning_rate', opts=dict(title='learning_rate'))
    viz.line([0.], [0.], win='Dice/test', opts=dict(title='Dice/test'))
    viz.line([0.], [0.], win='IoU/test', opts=dict(title='IoU/test'))
    viz.line([0.], [0.], win='F1/test', opts=dict(title='F1/test'))

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)#optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)##RMSprop, momentum=0.9
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,threshold=0.05,eps=1e-6)
    #if net.n_classes > 1:
        #criterion = nn.CrossEntropyLoss()
    #else:
    criterion =nn.BCEWithLogitsLoss()#SoftDiceLoss()## ##

    best_dice=0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # assert imgs.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'
                transformed=transform(image=imgs.numpy(),mask=true_masks.numpy())
                imgs=transformed['image']
                true_masks=transformed['mask']
                imgs=torch.from_numpy(imgs.copy()).type(torch.FloatTensor)
                true_masks=torch.from_numpy(true_masks.copy()).type(torch.FloatTensor)
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 #if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                #print(imgs.shape)
                #print(imgs.type)

                masks_pred = net(imgs)#['out']
                #print(masks_pred.shape)

                #print('mask_pred',masks_pred.shape)
                # print('masks_pred',masks_pred.shape)
                # print('true_masks', true_masks.shape)
                # print(imgs.shape)
                # print(masks_pred.shape)
                # print(true_masks.shape)
                viz.images(imgs, win='imgs/train')
                viz.images(true_masks, win='masks/true/train')
                viz.images(masks_pred, win='masks/pred/train')
                loss = criterion(masks_pred, true_masks)
                #print(loss)
                epoch_loss += loss.item()
                #writer.add_scalar('Loss/train', loss.item(), global_step)
                viz.line([loss.item()],[global_step],win='train_loss',update='append')
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (4* batch_size)) == 0:
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score,iou_score,f1 = eval_net(net, val_loader, device)
                    #scheduler.step(val_score)
                    #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    viz.line([optimizer.param_groups[0]['lr']], [global_step], win='learning_rate', update='append')

                    # if net.n_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(val_score))
                    #     #writer.add_scalar('Loss/test', val_score, global_step)
                    #else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                        #writer.add_scalar('Dice/test', val_score, global_step)
                    viz.line([val_score], [global_step], win='Dice/test', update='append')
                    viz.line([iou_score], [global_step], win='IoU/test', update='append')
                    viz.line([f1], [global_step], win='F1/test', update='append')
                    if val_score>best_dice:
                        best_dice=val_score
                        try:
                            os.mkdir(dir_checkpoint)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save(net.state_dict(),
                                   dir_checkpoint + f'best_dice.pth')
                        logging.info(f'best_dice saved !')


                    viz.images(imgs, win='images')
                    #if net.n_classes == 1:
                    #print('true_mask',true_masks.shape,true_masks.type)
                    viz.images(true_masks, win='masks/true')
                    #print('pred',(torch.sigmoid(masks_pred) > 0.5).squeeze(0).shape)
                    viz.images((torch.sigmoid(masks_pred) > 0.5),win='masks/pred')

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    #writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,#12(FCN),
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    #net=FCN.Fcn_Vgg16(module_type='32s', n_classes=1, pretrained=True)#UNet(n_channels=3, n_classes=1, bilinear=True)
    #net=SegNet()#net = nn.DataParallel(net)
    net = nn.DataParallel(net)
    net.to(device=device)
    #vis=nn.Sequential(*list(net.children())[])
    #net.load_state_dict(torch.load('/home/syx/US_kidney/Pytorch-UNet-master/checkpoints/simCT3_U-Net_lr=1e-4_Adam_0.966_scratch_batch=24_scale=0.5-epoch=50.pth', map_location=device))
    # net=
    # #logging.info(f'Network:\n'
    #              #f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     net.load_state_dict(
    #         torch.load(args.load, map_location=device)
    #     )
    #     logging.info(f'Model loaded from {args.load}')


    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
