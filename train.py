import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import argparse
from data_loader_2tp_correct import SliceData
# from dataloader_NEW import SliceData
# from dataloader_NEW import SliceData

# from models import ConvLSTM,Discriminator
# from models import UnetModel
# from model_uformer_tri_attn import Uformer,Discriminator
# from model_uformer_tri_attn_NEW import Uformer,Discriminator
# from model_uformer_tri_attn_alt_winsize import Uformer,Discriminator
from model_dcetriformer import Uformer,Discriminator

# from model_uformer import Uformer,Discriminator
from utils import gradient_penalty
from frequency_losses import get_gaussian_kernel, find_fake_freq, fft_L1_loss_color, decide_circle,fft_L1_loss_mask
from Mi_using_gauss_kernel import MutualInformation
from restormer_block import TransformerBlock 
#logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO)
#logger = logging.getLogger(__name__)
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
def create_datasets(args):

    train_data = SliceData(args.train_path)
    dev_data = SliceData(args.validation_path,mode='test')

    return dev_data, train_data


def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, dev_loader, display_loader

def train_epoch_wgangp(args, epoch, modelG, modelD, data_loader, optimizerG, optimizerD, writer):

    modelG.train()
    modelD.train()

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    #running_lossG = 0 Batch loss is used in graph and print statement
    #running_lossD = 0
    avg_G_loss = 0.
    avg_D_loss = 0.
    criterionG = nn.MSELoss()
    criterionD = nn.BCEWithLogitsLoss()

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader)

    for iter, data in enumerate(loop):

#         input,_,target = data
        t2,pd,adc, dce_1, dce_2, dce_3,s1,s = data
        
        # t2, pd, adc,b400,ktrans,org_mask,les_mask,  dce_1, dce_2, dce_3,dce10,dce12,dce15,pz,s1,s = data
        input = torch.cat((t2,pd, adc,dce_1), dim=1)
        # input = torch.cat((t2, pd,adc, dce_1), dim=1)
        # input = t2
#         input = torch.stack((input,input,input),dim=1) # (1,3,3,160,160), (1,3,2,160,160)
#         input = torch.stack((input,input),dim=1) # (1,3,2,160,160), (1,4,2,160,160) ---? (1,2,4,160,160)

        input = input.float().to(args.device) # (1,3,3,160,160), (1,3,2,160,160) ----> (1,4,160,160)
#         dce_1 = dce_1.float().to(args.device) # (1,1,160,160)
        dce_2 = dce_2.float().to(args.device) # (1,1,160,160)
        dce_3 = dce_3.float().to(args.device) # (1,1,160,160)
        # dce15 = dce15.float().to(args.device)
        # ktrans = ktrans.float().to(args.device) # (1,1,160,160)

        outG = modelG(input) # (1,2,160,160)
#         print("output shape", outG.shape)
#         outG = outG + input # learning residual and adding to input and use this as loss

#         dce_1_fake = modelD(outG[:,0,:,:,:])
#         dce_1_real = modelD(dce_1)
#         gp_1 = gradient_penalty(modelD, dce_1, outG[:,0,:,:,:], device=args.device)
        
#         dce_2_fake = modelD(outG[:,0,:,:].unsqueeze(1))
#         dce_2_real = modelD(dce_2)
#         gp_2 = gradient_penalty(modelD, dce_2, outG[:,0,:,:].unsqueeze(1),device=args.device)
        
#         dce_3_fake = modelD(outG[:,1,:,:].unsqueeze(1))
#         dce_3_real = modelD(dce_3)
#         gp_3 = gradient_penalty(modelD, dce_3, outG[:,1,:,:].unsqueeze(1), device=args.device)
        dce_2_fake = modelD(outG[:,0,:,:].unsqueeze(1))
        dce_2_real = modelD(dce_2)
        gp_2 = gradient_penalty(modelD, dce_2, outG[:,0,:,:].unsqueeze(1),device=args.device)
        
        dce_3_fake = modelD(outG[:,1,:,:].unsqueeze(1))
        dce_3_real = modelD(dce_3)
        gp_3 = gradient_penalty(modelD, dce_3, outG[:,1,:,:].unsqueeze(1), device=args.device)
        
        # ktrans_fake = modelD(outG[:,2,:,:].unsqueeze(1))
        # ktrans_real = modelD(ktrans)
        # gp = gradient_penalty(modelD, ktrans, outG[:,2,:,:].unsqueeze(1),device=args.device)
#         lossD_1=(-(torch.mean(dce_1_real) - torch.mean(dce_1_fake)) + args.LAMBDA_GP * gp_1)
#         lossD_2=(-(torch.mean(dce_2_real) - torch.mean(dce_2_fake)) + args.LAMBDA_GP * gp_2)
#         lossD_3=(-(torch.mean(dce_3_real) - torch.mean(dce_3_fake)) + args.LAMBDA_GP * gp_3)
        lossD_2=(-(torch.mean(dce_2_real) - torch.mean(dce_2_fake)) + args.LAMBDA_GP * gp_2)
        lossD_3=(-(torch.mean(dce_3_real) - torch.mean(dce_3_fake)) + args.LAMBDA_GP * gp_3)
        # lossD=(-(torch.mean(ktrans_real) - torch.mean(ktrans_fake)) + args.LAMBDA_GP * gp)
        lossD = lossD_2 + lossD_3 #+ lossD#lossD_1 + 
        optimizerD.zero_grad()

        lossD.backward(retain_graph=True)
        optimizerD.step()


#         dce_1_fake = modelD(outG[:,0,:,:,:])
#         dce_2_fake = modelD(outG[:,0,:,:].unsqueeze(1))
#         dce_3_fake = modelD(outG[:,1,:,:].unsqueeze(1))
#         lossG_1 = -torch.mean(dce_1_fake) + F.l1_loss(outG[:,0,:,:,:],dce_1)
#         lossG_2 = -torch.mean(dce_2_fake) + args.LAMBDA_L1_DCE_2 * F.l1_loss(outG[:,0,:,:].unsqueeze(1),dce_2)
#         lossG_3 = -torch.mean(dce_3_fake) + args.LAMBDA_L1_DCE_3 * F.l1_loss(outG[:,1,:,:].unsqueeze(1),dce_3)
        dce_2_fake = modelD(outG[:,0,:,:].unsqueeze(1))
        
        dce_3_fake = modelD(outG[:,1,:,:].unsqueeze(1))
        # ktrans_fake = modelD(outG[:,2,:,:].unsqueeze(1))
        
        gauss_kernel = get_gaussian_kernel(args.gauss_size).cuda()
        dce_2_real_freq = find_fake_freq(dce_2, gauss_kernel)  
        dce_2_fake_freq = find_fake_freq(outG[:,0,:,:].unsqueeze(1), gauss_kernel)
        loss_rec_blur_DCE_2 = F.l1_loss(dce_2_fake_freq, dce_2_real_freq)
        dce_3_real_freq = find_fake_freq(dce_3, gauss_kernel)  
        dce_3_fake_freq = find_fake_freq(outG[:,1,:,:].unsqueeze(1), gauss_kernel)
        loss_rec_blur_DCE_3 = F.l1_loss(dce_3_fake_freq, dce_3_real_freq)
        # ktrans_real_freq = find_fake_freq(ktrans,gauss_kernel)
        # ktrans_fake_freq = find_fake_freq(outG[:,2,:,:].unsqueeze(1),gauss_kernel)
        # loss_rec_blur_ktrans = F.l1_loss(ktrans_fake_freq, ktrans_real_freq)
        
        loss_recon_fft_DCE_2 = fft_L1_loss_color(outG[:,0,:,:].unsqueeze(1), dce_2)
        loss_recon_fft_DCE_3 = fft_L1_loss_color(outG[:,1,:,:].unsqueeze(1), dce_3)
        # loss_recon_fft_ktrans = fft_L1_loss_color(outG[:,2,:,:].unsqueeze(1), ktrans)
        
        MI = MutualInformation(num_bins=160, sigma=0.3, normalize=True).to(args.device)
        
#         print(dce_2.shape,outG[:,0,:,:].unsqueeze(1).shape)
        
    # mutual information loss - subtract mean of mi_dce_2 and mi_dce_3
        dce_2_fake_MI = outG[:,0,:,:].unsqueeze(1)
        dce_3_fake_MI = outG[:,1,:,:].unsqueeze(1)
        # ktrans_fake_MI = outG[:,2,:,:].unsqueeze(1)
        
        input1_dce2 = torch.cat([dce_2, dce_2_fake_MI]).to(args.device)
        input2_dce2 = torch.cat([dce_2_fake_MI, dce_2_fake_MI]).to(args.device)
        
        input1_dce3 = torch.cat([dce_3, dce_3_fake_MI]).to(args.device)
        input2_dce3 = torch.cat([dce_3_fake_MI, dce_3_fake_MI]).to(args.device)
        
        # input1_ktrans = torch.cat([ktrans,ktrans_fake_MI]).to(args.device)
        # input2_ktrans = torch.cat([ktrans_fake_MI,ktrans_fake_MI]).to(args.device)
        
#         input1_adc_dce2 = torch.cat([adc, dce_2_fake_MI]).to(args.device)
#         input1_adc_dce3 = torch.cat([adc, dce_3_fake_MI]).to(args.device)
    
        mi_dce2 = MI(input1_dce2, input2_dce2)
        mi_dce3 = MI(input1_dce3, input2_dce3)
        # mi_ktrans = MI(input1_ktrans, input2_ktrans)
        
        lossG_2 = -torch.mean(dce_2_fake) + args.LAMBDA_L1_DCE_2 * F.l1_loss(outG[:,0,:,:].unsqueeze(1),dce_2)  + (args.lambda_recon_blur * loss_rec_blur_DCE_2) + (args.lambda_recon_fft * loss_recon_fft_DCE_2) + (10* (1 -  mi_dce2[0]))
        
        lossG_3 = -torch.mean(dce_3_fake) + args.LAMBDA_L1_DCE_3 * F.l1_loss(outG[:,1,:,:].unsqueeze(1),dce_3) + (args.lambda_recon_blur * loss_rec_blur_DCE_3) + (args.lambda_recon_fft * loss_recon_fft_DCE_3) + (10*  (1 - mi_dce3[0])) 
        
        # lossG_ktrans = -torch.mean(ktrans_fake) + args.LAMBDA_L1 * F.l1_loss(outG[:,2,:,:].unsqueeze(1),ktrans) + (args.lambda_recon_blur * loss_rec_blur_ktrans) + (args.lambda_recon_fft * loss_recon_fft_ktrans) + (5*  (1 - mi_ktrans[0])) 
        lossG = lossG_2 + lossG_3 #+ lossG_ktrans#lossG_1 +
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

        avg_G_loss = 0.99 * avg_G_loss + 0.01 * lossG.item() if iter > 0 else lossG.item()
        avg_D_loss = 0.99 * avg_D_loss + 0.01 * lossD.item() if iter > 0 else lossD.item()
        writer.add_scalar('GenLoss', lossG.item(), global_step + iter)
        writer.add_scalar('DiscLoss', lossD.item(), global_step + iter)
        #writer.add_scalar('lossD_fake', lossD_fake.item(), global_step+iter)
        #writer.add_scalar('lossD_real', lossD_real.item(), global_step+iter)
        #break
        
        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'LossG = {lossG.item():.4g} Avg G Loss = {avg_G_loss:.4g} '
                f'LossD = {lossD.item():.4g} Avg D Loss = {avg_D_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'Epoch': epoch, 'LossG': avg_G_loss, 'lossD': avg_D_loss})

    return lossG.item(), lossD.item(), time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    # losses_ktrans = []
    losses_2 = []
    losses_3 = []
    losses = []
    start = time.perf_counter()
    avg_loss_ktrans = 0.
    avg_2_loss = 0.
    avg_3_loss = 0.
    loop = tqdm(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(loop):
            
#             t2, pd, adc,  dce_1, dce_2, dce_3,s1,s = data
            t2,pd,adc, dce_1, dce_2, dce_3,s1,s = data
            # t2, pd, adc,b400,ktrans,org_mask,les_mask,  dce_1, dce_2, dce_3,dce10,dce12,dce15,pz,s1,s = data
            input = torch.cat((t2,pd, adc,dce_1), dim=1)
           

            input = input.float().to(args.device) # (1,3,3,160,160), (1,3,2,160,160)
#             dce_1 = dce_1.float().to(args.device) # (1,1,160,160)
            dce_2 = dce_2.float().to(args.device) # (1,1,160,160)
            dce_3 = dce_3.float().to(args.device) # (1,1,160,160)
            # ktrans = ktrans.float().to(args.device) # (1,1,160,160)

            #print(input.shape)
            output = model(input) # (1,3,1,160,160)
#             output = output + input # learning residual and adding to input and use this as loss         

#             loss_1 = F.l1_loss(output[:,0,:,:,:],dce_1)
            loss_2 = F.l1_loss(output[:,0,:,:].unsqueeze(1),dce_2)
            loss_3 = F.l1_loss(output[:,1,:,:].unsqueeze(1),dce_3)
            # loss_ktrans = F.l1_loss(output[:,2,:,:].unsqueeze(1),ktrans)
            
            losses_2.append(loss_2.item())
            losses_3.append(loss_3.item())
            # losses_ktrans.append(loss_ktrans.item())
#             losses_3.append(loss_3.item())
            losses.append(np.mean([loss_2.item(),loss_3.item()])) #,loss_3.item()
            
            # avg_loss_ktrans = 0.99 * avg_loss_ktrans + 0.01 * loss_ktrans.item() if iter > 0 else loss_ktrans.item()
            avg_2_loss = 0.99 * avg_2_loss + 0.01 * loss_2.item() if iter > 0 else loss_2.item()
            avg_3_loss = 0.99 * avg_3_loss + 0.01 * loss_3.item() if iter > 0 else loss_3.item()
            
            loop.set_postfix({'Epoch': epoch, 'Loss2': avg_2_loss,'Loss3':avg_3_loss}) #, 'Loss_3': avg_3_loss

            #break
        

        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
       
    return np.mean(losses), time.perf_counter() - start

def visualize(args, epoch, model, data_loader, writer):

    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            
            # t2, pd, adc,b400,ktrans,org_mask,les_mask,  dce_1, dce_2, dce_3,dce10,dce12,dce15,pz,s1,s = data
            t2,pd,adc, dce_1, dce_2, dce_3,s1,s = data
            input = torch.cat((t2,pd, adc,dce_1), dim=1)
            

            input = input.float().to(args.device) # (1,3,3,160,160), (1,3,2,160,160)
#             dce_1 = dce_1.float().to(args.device) # (1,1,160,160)
            dce_2 = dce_2.float().to(args.device) # (1,1,160,160)
            dce_3 = dce_3.float().to(args.device) # (1,1,160,160)
            # ktrans = ktrans.float().to(args.device) # (1,1,160,160)
            
#             print("input: ", torch.min(input), torch.max(input))
            output = model(input)
#             output = output + input # learning residual and adding to input and use this as loss         
            
#             print("Predicted: ", torch.min(output), torch.max(output))
            save_image(input[:,0,:,:].unsqueeze(1), 'Input_t2')
            
            break

def save_model(args, exp_dir, epoch, modelG,optimizerG, modelD, optimizerD, best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'modelG': modelG.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'modelD':modelD.state_dict(),
            'optimizerD':optimizerD.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir+'/model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir+'/model.pt', exp_dir+'/best_model.pt')


def build_model(args):

    # model = UnetModel(
    #     in_chans=4,
    #     out_chans=2,
    #     chans=args.num_chans,
    #     num_pool_layers=args.num_pools,
    #     drop_prob=args.drop_prob
    # ).to(args.device)
    
#     model = ConvLSTM(input_dim=4,
#                      hidden_dim=32,
#                      output_dim=1,
#                      num_layers=3,
#                      timesteps=2).to(args.device)

    model = Uformer(in_chans=2, dd_in=4).to(args.device)

    return model

def build_discriminator(args):
    
    netD = Discriminator(input_nc=1).to(args.device)
    #optimizerD = optim.SGD(netD.parameters(),lr=1e-5)
    optimizerD = torch.optim.Adam(netD.parameters(), args.lr_D, weight_decay=args.weight_decay, betas=(0,0.9))
    return netD, optimizerD

def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    modelG = build_model(args)

    if args.data_parallel:
        modelG = torch.nn.DataParallel(modelG)

    modelG.load_state_dict(checkpoint['modelG'])

    optimizerG = build_optim(args, modelG.parameters())
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    modelD,optimizerD = build_discriminator(args)

    if args.data_parallel:
        modelD = torch.nn.DataParallel(modelD)

    modelD.load_state_dict(checkpoint['modelD'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])

    return checkpoint, modelG, optimizerG, modelD, optimizerD


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr_G, weight_decay=args.weight_decay, betas=(0,0.9))
    return optimizer

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
        
    return target


def main(args):

    try:
        os.mkdir(args.exp_dir)
    except:
        pass
    writer = SummaryWriter(log_dir=str(args.exp_dir+'/summary'))
    #vgg = Vgg16(requires_grad=False).to(args.device)
#     vgg = Vgg16().to(args.device)


    if args.resume: 
        print('resuming model, batch_size', args.batch_size)
        checkpoint, modelG, optimizerG, modelD, optimizerD = load_model(args.checkpoint)
        bs = args.batch_size
        num_epochs = args.num_epochs
        args = checkpoint['args']
        args.num_epochs = num_epochs
        args.batch_size = bs
        start_epoch = checkpoint['epoch'] + 1
        best_dev_loss = checkpoint['best_dev_loss']
        del checkpoint

    else:

        modelG = build_model(args)
        modelD, optimizerD = build_discriminator(args)
        print("model G and D created")

        if args.data_parallel:
            modelG = torch.nn.DataParallel(modelG)
            modelD = torch.nn.DataParallel(modelD)

        optimizerG = build_optim(args, modelG.parameters())
        start_epoch = 0
        best_dev_loss = 1e9

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)
#     schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs+1):
        schedulerG.step(epoch)
#         schedulerD.step(epoch)
        train_lossG, train_lossD, train_time = train_epoch_wgangp(args, epoch, modelG, modelD, train_loader, optimizerG , optimizerD, writer)
        dev_loss, dev_time = evaluate(args, epoch, modelG, dev_loader, writer)
        visualize(args, epoch, modelG, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)

        save_model(args, args.exp_dir, epoch, modelG, optimizerG, modelD, optimizerD, best_dev_loss,is_new_best)
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLossG = {train_lossG:.4g} TrainLossD = {train_lossD:.4g} 'f'DevNLL = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s')
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr_G', type=float, default=0.0001, help='Learning rate') #0.0001
    parser.add_argument('--lr_D', type=float, default=0.00001, help='Learning rate') #0.0001
    parser.add_argument('--lr-step-size', type=int, default=40, #40
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')
    parser.add_argument('--critic-iterations', type=int, default=5, help='Number of critic iterations')
    parser.add_argument('--LAMBDA_GP', type=int, default=10, help='Gradient Penalty lambda')
    parser.add_argument('--LAMBDA_L1', type=int, default=1, help='Number of critic iterations')
    parser.add_argument('--LAMBDA_L1_DCE_2', type=int, default=5, help='Number of critic iterations')
    parser.add_argument('--LAMBDA_L1_DCE_3', type=int, default=5, help='Number of critic iterations')
    parser.add_argument('--lambda_recon_blur', type = int, default = 10, help = 'Reconstruction loss using Gaussian kernel')
    parser.add_argument('--lambda_recon_fft', type = int, default = 10, help = 'Reconstruction loss using FFT')
    parser.add_argument('--gauss_size', type = int, default = 13, help = 'size of gaussian kernel')
    # parser.add_argument('--LAMBDA_RES_21', type = int, default =10 , help = 'hyperparameter for residual loss between DCE_1 and DCE_2')
    # parser.add_argument('--LAMBDA_RES_31', type = int, default = 10, help = 'hyperparameter for residual loss between DCE_1 and DCE_3')
#     parser.add_argument('--LAMBDA_L1_DCE_3', type=int, default=1, help='Number of critic iterations')

    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    args.exp_dir = str(args.exp_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
