

import os
import time
import torch


import numpy as np
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd.variable import Variable


from data_sampler import ICAODataset
from models.architecture.DeepLabV3_ICAO_SE_lite import ICAO_DEEPLAB

import pandas as pd
from tabulate import tabulate
import argparse

from utils.misc import save_checkpoint, AverageMeter, adjust_learning_rate, get_eer



time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
sigmoid_fun = torch.nn.Sigmoid()
softmax_fun = torch.nn.Softmax(dim=1)

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    
    parser = argparse.ArgumentParser(description = 'Script to train the developed region model on multiple datasets') 
    
    parser.add_argument('--experience_path','-exp', default='runs/exp_RN18Enc_' + time_stp, type=str, 
                        help='directory to save experiment outputs. If directory already exists, it will resume the training procedure from the last stored checkpoint')
    parser.add_argument('--devices', type=str, default='0', help='device id')
    parser.add_argument('--model_name', default='MbNet', help='model name')
    parser.add_argument('--best_only', default=True, type=bool, help='Save only the best checkpoint (False implies save all)')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', '-nw', default=8, type=int, help='number of workers')
    parser.add_argument('--learning_rate','-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--every_decay','-dec', default=30, type=int, help='number of epochs to decay learning rate by half')
    parser.add_argument('--epochs','-ep', default=100, type=int, help='number of train epochs')
    parser.add_argument('--momentum','-m', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay','-wd', default=1e-4, type=float, help='SGD weight decay')

    args = parser.parse_args()
      
    return args


def main(args):
    
    global device
    gpu_enabled = torch.cuda.is_available()
    device = torch.device('cuda:' + args.devices[0] if gpu_enabled else "cpu")


    start_epoch = 0
    best_prec1 = 0
    best_epoch = 0
    last_best_epoch = 0
    
    
    lr=args.learning_rate
    bs=args.batch_size
    epochs=args.epochs
    every_decay=args.every_decay
    model_name = args.model_name
    
    
    save_path = args.experience_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]
    
    
    normalize = transforms.Normalize(mean=means, 
                                      std=stds)
    
    
    train_dataset = ICAODataset(
        transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            normalize
        ]),phase_train=True)


    
    val_dataset = ICAODataset( transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        normalize
    ]),phase_train=False,phase_test=False)


    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, num_workers=args.num_workers,pin_memory=False, shuffle =True, drop_last = True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs,  num_workers=args.num_workers,pin_memory=False, shuffle=False)
    
    

    model = ICAO_DEEPLAB(n_maps = 8, n_reqs = 26)

    
    if gpu_enabled:
        torch.backends.cudnn.benchmark = True
    model.to(device)

    
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total_params',pytorch_total_params)
    
    
    
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    

    df_best = pd.DataFrame()
    
    for epoch in range(start_epoch, epochs):
        print("epoch: ", epoch)


        optimizer = adjust_learning_rate(optimizer, epoch, every_decay)

        _ = train(train_loader, model, criterion_cls, optimizer, epoch, args)

        # evaluate on validation set
        prec1, df_res = validate(val_loader, model, criterion_cls, epoch, args)


        # remember best prec@1 and save checkpoint
        is_best = prec1 >= best_prec1
        if is_best:
            last_best_epoch = best_epoch
            best_epoch = epoch

            print('epoch: {} The best is {} last best is {} at epoch {}'.format(epoch,prec1,best_prec1, last_best_epoch))


            with open('{}/val_result_{}.txt'.format(args.experience_path, args.model_name),'a+') as f_result:
                f_result.write('{}\n'.format('!!!!!!!!!!!!!!  Best So Far !!!!!!!!!!!!!!!!! '))

            if args.best_only:
                save_name = '{}/{}_best.pth.tar'.format(save_path, model_name)
            else:
                save_name = '{}/{}_{epoch:03d}_best.pth.tar'.format(save_path, model_name, epoch=epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
                'optimizer': optimizer.state_dict(),
            }, filename=save_name)
        else:
            if not args.best_only:
                save_name = '{}/{}_{epoch:03d}.pth.tar'.format(save_path, model_name, epoch=epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': model_name,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, filename=save_name)
            print('epoch: {} The current is {} last best is {} at epoch {}'.format(epoch,prec1,best_prec1, last_best_epoch))

            with open('{}/val_result_{}.txt'.format(args.experience_path, args.model_name),'a+') as f_result:
                f_result.write('epoch: {} The current is {} last best is {} at epoch {}'.format(epoch,prec1,best_prec1, last_best_epoch))
        
        
        best_prec1 = max(prec1, best_prec1)
        last_best_epoch = best_epoch
        

    
    return df_best
            

def train(train_loader, model, criterion_cls, optimizer, epoch, args):
    
    batch_time = AverageMeter()
    
    losses_small = AverageMeter()
    losses_class_all_regs = AverageMeter()
    losses = AverageMeter()


    model.train()
    
    end = time.time()


    pos_weight_offline = torch.tensor([ 29.18604651,  29.67800729,   9.72540666,  27.1487524 ,
        14.75788348,  14.35070423,  62.63362069,   2.92142568,
        15.6169863 ,   9.92889138,  21.93419506,  16.40677966,
        22.08464329,  22.09822485,  22.93643032,  15.92925026,
       156.80645161,   8.64891462,   8.44806202,  14.27514079,
        16.02511848,   4.71405608,  41.23853211,  41.23853211,
        11.51924759,  41.23853211])

    req_weight = torch.tensor([0.94966316, 0.9271123 , 0.9017848 , 0.80047785, 0.82010423,
       2.22246717, 0.7791865 , 1.0464771 , 0.79452363, 1.27698178,
       1.21315322, 1.23122651, 1.23985534, 1.21269844, 1.20694585,
       1.5011414 , 1.16461212, 0.75780463, 1.03893916, 0.63809507,
       0.66970449, 0.85794946, 0.62971963, 0.62971963, 0.85993688,
       0.62971963])
    
    for i, (image, final_masks, region_requirement_labels, supervision_surpression) in enumerate(train_loader):


        input_var = Variable(image).float().to(device)
        target_masks = Variable(final_masks).float().to(device)
        target_labels =  Variable(region_requirement_labels).float().to(device)


        num_pos_per_mask = final_masks.sum(dim = 0).sum(dim=1).sum(dim=1) + 1
        pos_weights4loss = (num_pos_per_mask.max()/num_pos_per_mask).unsqueeze(1).unsqueeze(1).repeat(1,512, 512)


        inverted_lab = 1 - region_requirement_labels


        pos_num_per_class = torch.logical_and(inverted_lab.int() == 1, supervision_surpression.int() == 1).sum(dim = 0)
        neg_num_per_class = torch.logical_and(inverted_lab.int() == 0, supervision_surpression.int() == 1).sum(dim = 0)
        pos_weights4loss_class = neg_num_per_class / pos_num_per_class
        pos_weights4loss_class[pos_weights4loss_class == 0] = 1
        pos_weights4loss_class[pos_weights4loss_class == torch.inf] = 0
        pos_weights4loss_class[torch.isnan(pos_weights4loss_class)] = 0
        

        criterion4seg = nn.BCEWithLogitsLoss(pos_weight = pos_weights4loss.detach().to(device))


        final_req_weight = supervision_surpression*(req_weight.repeat(supervision_surpression.shape[0],1))
        criterion4class = nn.BCEWithLogitsLoss(weight = final_req_weight.to(device), pos_weight = pos_weight_offline.detach().to(device))

        infered_masks, region_requirement_preds = model(input_var)

        loss_seg = criterion4seg(infered_masks, target_masks)
        total_class_reg_loss = criterion4class(region_requirement_preds, 1 - target_labels)

        if epoch >= 0:
            total_loss = loss_seg + total_class_reg_loss
        else:
            total_loss = loss_seg
        


        # compute gradient and do SGD step
        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        
        losses.update(total_loss.item())
        losses_small.update(loss_seg.item())
        losses_class_all_regs.update(total_class_reg_loss.item())



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]['lr']


        if (i % 10 == 0) or (i==len(train_loader)-1):
            # print(target.to('cpu').detach().numpy().mean())

            with open('{}/{}.log'.format(args.experience_path, args.model_name), 'a+') as flog:
                

                line = 'Epoch: [{0}][{1}/{2}]\t lr:{3:.5f}  \t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  \t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})  \t' \
                        'Loss Seg {losses_small.val:.4f} ({losses_small.avg:.4f})  \t' \
                        'Loss Class Regs Total {losses_class_all_regs.val:.4f} ({losses_class_all_regs.avg:.4f})  \t' \
                        .format(epoch, i, len(train_loader),lr,
                        batch_time=batch_time,
                        losses=losses, losses_small=losses_small, losses_class_all_regs=losses_class_all_regs)
                print(line)
                flog.write('{}\n'.format(line))


        if i >=300:
            break

                
    
    return pd.DataFrame()
    

   

def validate(loader, model, criterion_cls, epoch, args):
    
    batch_time = AverageMeter()
    
    losses_small = AverageMeter()
    losses_class_all_regs = AverageMeter()
    losses = AverageMeter()

    model.eval()
    
    end = time.time()
    
    predictions = []
    labels = []
    supressions = []
    
    names = ['eyes_closed',
             'non_neutral_expression',
             'mouth_open',
             'rotated_shoulders',
             'roll_pitch_yaw',
             'looking_away',
             'hair_across_eyes',
             'head_coverings',
             'veil_over_face',
             'other_faces_objects',
             'dark_tinted_lenses',
             'frame_covering_eyes',
             'flash_reflection_lenses',
             'frame_too_heavy',
             'shadows_behind_head',
             'shadows_across_face',
             'flash_reflection_skin',
             'unnatural_skin_tone',
             'red_eyes',
             'too_dark_light',
             'blurred',
             'varied_background',
             'pixelation',
             'washed_out',
             'ink_marked_creased',
             'posterization']


    pos_weight_offline = torch.tensor([ 29.18604651,  29.67800729,   9.72540666,  27.1487524 ,
        14.75788348,  14.35070423,  62.63362069,   2.92142568,
        15.6169863 ,   9.92889138,  21.93419506,  16.40677966,
        22.08464329,  22.09822485,  22.93643032,  15.92925026,
       156.80645161,   8.64891462,   8.44806202,  14.27514079,
        16.02511848,   4.71405608,  41.23853211,  41.23853211,
        11.51924759,  41.23853211])

    req_weight = torch.tensor([0.94966316, 0.9271123 , 0.9017848 , 0.80047785, 0.82010423,
       2.22246717, 0.7791865 , 1.0464771 , 0.79452363, 1.27698178,
       1.21315322, 1.23122651, 1.23985534, 1.21269844, 1.20694585,
       1.5011414 , 1.16461212, 0.75780463, 1.03893916, 0.63809507,
       0.66970449, 0.85794946, 0.62971963, 0.62971963, 0.85993688,
       0.62971963])

    for i, (image, final_masks, region_requirement_labels, supervision_surpression) in enumerate(loader):


        with torch.no_grad():
            
            input_var = Variable(image).float().to(device)
            target_masks = Variable(final_masks).float().to(device)
            target_labels = Variable(region_requirement_labels).float().to(device)


            num_pos_per_mask = final_masks.sum(dim = 0).sum(dim=1).sum(dim=1) + 1
            pos_weights4loss = (num_pos_per_mask.max()/num_pos_per_mask).unsqueeze(1).unsqueeze(1).repeat(1,512, 512)

            num_pos_per_mask = final_masks.sum(dim = 0).sum(dim=1).sum(dim=1)
            num_neg_per_mask = (final_masks==0).sum(dim = 0).sum(dim=1).sum(dim=1)
            pos_weights4loss = num_neg_per_mask/num_pos_per_mask
            pos_weights4loss[pos_weights4loss == torch.inf] = 0
            pos_weights4loss = pos_weights4loss.unsqueeze(1).unsqueeze(1).repeat(1,512, 512)

            inverted_lab = 1 - region_requirement_labels

            pos_num_per_class = torch.logical_and(inverted_lab.int() == 1, supervision_surpression.int() == 1).sum(dim = 0)
            neg_num_per_class = torch.logical_and(inverted_lab.int() == 0, supervision_surpression.int() == 1).sum(dim = 0)
            pos_weights4loss_class = neg_num_per_class / pos_num_per_class
            pos_weights4loss_class[pos_weights4loss_class == 0] = 1
            pos_weights4loss_class[pos_weights4loss_class == torch.inf] = 0
            pos_weights4loss_class[torch.isnan(pos_weights4loss_class)] = 0

            criterion4seg = nn.BCEWithLogitsLoss(pos_weight = pos_weights4loss.detach().to(device))
            
            final_req_weight = supervision_surpression*(req_weight.repeat(supervision_surpression.shape[0],1))
            criterion4class = nn.BCEWithLogitsLoss(weight = final_req_weight.to(device), pos_weight = pos_weight_offline.detach().to(device))

            infered_masks, region_requirement_preds = model(input_var)

            loss_seg = criterion4seg(infered_masks, target_masks)
            total_class_reg_loss = criterion4class(region_requirement_preds, 1 - target_labels)



            total_loss = loss_seg + total_class_reg_loss

            
            losses.update(total_loss.item())
            losses_small.update(loss_seg.item())
            losses_class_all_regs.update(total_class_reg_loss.item())


        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        all_preds = 1 - sigmoid_fun(region_requirement_preds.to('cpu').detach()).numpy()

        all_labs = region_requirement_labels.to('cpu').detach().numpy()

        all_sups = supervision_surpression.to('cpu').detach().numpy().astype('int')
       

        
        predictions.append(all_preds)
        labels.append(all_labs)
        supressions.append(all_sups)

        


        if (i % 10 == 0) or (i==len(loader)-1):


            with open('{}/{}.log'.format(args.experience_path, args.model_name), 'a+') as flog:
                

                line = 'Test Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  \t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})  \t' \
                        'Loss Seg {losses_small.val:.4f} ({losses_small.avg:.4f})  \t' \
                        'Loss Class Regs Total {losses_class_all_regs.val:.4f} ({losses_class_all_regs.avg:.4f})  \t' \
                        .format(epoch, i, len(loader),
                        batch_time=batch_time,
                        losses=losses, losses_small=losses_small, losses_class_all_regs=losses_class_all_regs)
                print(line)
                flog.write('{}\n'.format(line))
                
                
        # if i >=30:
        #     break

                


    predictions_np = np.concatenate(predictions)
    labels_np = np.concatenate(labels)
    supressions_np = np.concatenate(supressions)


    
    AUCs, EERs, HTERs = get_eer(labels_np, predictions_np, supressions_np)
    

    df_res = pd.DataFrame(data = {'AUC': AUCs, 'EER': EERs, 'HTER': HTERs}, index = names)
    
    table_form = tabulate(df_res, headers = 'keys', tablefmt = 'fancy_grid')
    print(table_form)
    

    with open('{}/val_result_{}.txt'.format(args.experience_path, args.model_name),'a+') as f_result:
        f_result.write('{}\n'.format(table_form))
        
        
    final_metric = 1 - np.array(HTERs).mean()
        

    
    return final_metric, df_res

if __name__ == '__main__':

    
    args = parse_args()
    val_best = main(args)
    





    