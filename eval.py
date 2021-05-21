import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from dice_loss import dice_coeff,IoULoss,F1
import FCN
from torch.utils.data import DataLoader, random_split
import numpy as np

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 #if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    iou=0
    f1=0


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        list_dice = []
        list_iou=[]
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)#['out']

            # if net.n_classes > 1:
            #     tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot = dice_coeff(pred, true_masks)[0]#.item()
            list_dice.append(tot)
            iou =IoULoss().forward(pred, true_masks).item()
            list_iou.append(iou)
            #f1 +=F1().forward(pred, true_masks).item()

            pbar.update()

    net.train()
    #print(len(list_dice))
    #print(list_dice)
    return np.mean(list_dice),np.mean(list_iou),np.std(list_dice,ddof=1),np.std(list_iou,ddof=1)#list_dice[0]#

# if __name__ == "__main__":
#     batch_size=8
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = FCN.Fcn_Vgg16(module_type='32s', n_classes=1, pretrained=False)  #
#     # net=SegNet()net = nn.DataParallel(net)
#     net = nn.DataParallel(net)
#     net.to(device=device)
#     # vis=nn.Sequential(*list(net.children())[])
#     # net.load_state_dict(torch.load('/home/syx/US_kidney/Pytorch-UNet-master/checkpoints/simCT6000_batch=8_fcn_scale=0.5_epoch=18_transnature.pth', map_location=device))
#     # net=
#     val_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
#
#     val_score, val_iou, val_f1 = eval_net(net, val_loader, device)
#     # test_score, test_iou, test_f1 = eval_net(net, test_loader, device)
#     sugh_score, sugh_iou, sugh_f1 = eval_net(net, sugh_loader, device)
#     dice.append([global_step, val_score, sugh_score])
#
#     # scheduler.step(val_score)
#     # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
#     viz.line([optimizer.param_groups[0]['lr']], [global_step], win='learning_rate', update='append')
#
#     # if net.n_classes > 1:
#     #     logging.info('Validation cross entropy: {}'.format(val_score))
#     #     #writer.add_scalar('Loss/test', val_score, global_step)
#     # else:
#     logging.info('Validation Dice Coeff: {}'.format(val_score))
#     # writer.add_scalar('Dice/test', val_score, global_step)
#     viz.line([[val_score,  # test_score,
#                sugh_score]], [global_step], win='Dice/test', update='append')
#     viz.line([[val_iou,  # test_iou,
#                sugh_iou]], [global_step], win='IoU/test', update='append')
#     viz.line([[val_f1,  # test_f1,
#                sugh_f1]], [global_step], win='F1/test', update='append')
#     if val_score > best_dice:
#         best_dice = val_score
#         try:
#             os.mkdir(dir_checkpoint)
#             logging.info('Created checkpoint directory')
#         except OSError:
#             pass
#         torch.save(net.state_dict(),
#                    dir_checkpoint + f'best_dice.pth')
#         logging.info(f'best_dice saved !')
#
#     viz.images(imgs, win='images')
#     # if net.n_classes == 1:
#     # print('true_mask',true_masks.shape,true_masks.type)
#     viz.images(true_masks, win='masks/true')
#     # print('pred',(torch.sigmoid(masks_pred) > 0.5).squeeze(0).shape)
#     viz.images((torch.sigmoid(masks_pred) > 0.5), win='masks/pred')