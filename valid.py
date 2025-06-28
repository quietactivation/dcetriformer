import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader_2tp_correct import SliceData
# from dataloader_NEW import SliceData
# from data_loader_pmri import SliceDatapmri
# from models import UnetModel
# from model_uformer_tri_attn_NEW import Uformer,Discriminator
# from model_uformer_tri_attn_alt_winsize import Uformer,Discriminator
from model_dcetriformer import Uformer,Discriminator

# from model_uformer import Uformer, Discriminator
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):

    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceData(args.validation_path,mode='test')
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader

def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
#     model = UnetModel(4, 2, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
#     model = ConvLSTM(input_dim=3,
#                      hidden_dim=32,
#                      output_dim=1,
#                      num_layers=3,
#                      timesteps=2).to(args.device)
    model = Uformer(in_chans=2, dd_in=4).to(args.device)

    #print(model)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['modelG'])

    return model


def run_gan(args, model, data_loader):

    model.eval()
    recons_dce_2 = defaultdict(list)
    recons_dce_3 = defaultdict(list)
    # recons_ktrans = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):
            
            t2, pd, adc,  dce_1, dce_2, dce_3,fnames,slices = data
            # t2, pd, adc,b400,ktrans,org_mask,les_mask,  dce_1, dce_2, dce_3,dce10,dce12,dce15,pz,fnames,slices = data
            input = torch.cat((t2,pd,adc,dce_1), dim=1) #(1,4,160,160)
            
            input = input.float().to(args.device) # (1,3,3,160,160), (1,3,2,160,160)
#             dce_1 = dce_1.float().to(args.device) # (1,1,160,160)
            dce_2 = dce_2.float().to(args.device) # (1,1,160,160)
            dce_3 = dce_3.float().to(args.device) # (1,1,160,160)
            # ktrans = ktrans.float().to(args.device)
            
            #print(input.shape)
            output = model(input) # (1,3,1,160,160)
            
            pred_dce_2 = output[:,0,:,:].unsqueeze(1).to('cpu').squeeze(1)
            pred_dce_3 = output[:,1,:,:].unsqueeze(1).to('cpu').squeeze(1)
            # pred_ktrans = output[:,0,:,:].unsqueeze(1).to('cpu').squeeze(1)
            
            for i in range(pred_dce_2.shape[0]):
                pred_dce_2[i] = pred_dce_2[i] 
                recons_dce_2[fnames[i]].append((slices[i].numpy(), pred_dce_2[i].numpy()))
                
            for i in range(pred_dce_3.shape[0]):
                pred_dce_3[i] = pred_dce_3[i] 
                recons_dce_3[fnames[i]].append((slices[i].numpy(), pred_dce_3[i].numpy()))
            
            # for i in range(pred_ktrans.shape[0]):
            #     pred_ktrans[i] = pred_ktrans[i]
            #     recons_ktrans[fnames[i]].append((slices[i].numpy(), pred_ktrans[i].numpy()))
                
    recons_dce_2 = {
    fname: np.stack([pred for _, pred in sorted(slice_preds)])
    for fname, slice_preds in recons_dce_2.items()
    }
        
    recons_dce_3 = {
    fname: np.stack([pred for _, pred in sorted(slice_preds)])
    for fname, slice_preds in recons_dce_3.items()
    }
    
    # recons_ktrans = {
    # fname: np.stack([pred for _, pred in sorted(slice_preds)])
    # for fname, slice_preds in recons_ktrans.items()
    # }
    # print(recons_dce_2.keys())
    return  recons_dce_2, recons_dce_3


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    # recons_ktrans = run_gan(args, model, data_loader) #recons_dce_2, recons_dce_3,
    recons_dce_2, recons_dce_3 = run_gan(args, model, data_loader)
    save_reconstructions(recons_dce_2, args.out_dir_dce_2)
    save_reconstructions(recons_dce_3, args.out_dir_dce_3)
    # save_reconstructions(recons_ktrans, args.out_dir_ktrans)
    


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir-dce-2', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--out-dir-dce-3', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    # parser.add_argument('--out_dir-ktrans', type = pathlib.Path, required = True,
    #                     help = 'Path to save reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
