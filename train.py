import torch
import torch.optim as optim

import argparse
import os
import time
import datetime
import json
from pathlib import Path
import numpy as np

# from dataloader import parse_datasets
from models.conv_odegru import *
import utils
from dataset.echo_dynamic import Echo_dynamic
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default="video_dynamics", help='Specify experiment')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--phase', default="train", choices=["train", "test"])
    
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument('--window_size', type=int, default=20, help="Window size to sample")
    parser.add_argument('--sample_size', type=int, default=10, help="Number of time points to sub-sample")
    
    # Hyper-parameters
    parser.add_argument('--lamb_adv', type=float, default=0.003, help="Adversarial Loss lambda")
    
    # Network variants for experiment..
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    parser.add_argument('--n_layers', type=int, default=2, help='A number of layer of ODE func')
    parser.add_argument('--n_downs', type=int, default=2)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--input_norm', action='store_true', default=False)
    parser.add_argument('--run_backwards', action='store_true', default=True)
    
    # Log
    parser.add_argument("--ckpt_save_freq", type=int, default=5000)
    parser.add_argument("--log_print_freq", type=int, default=10)
    parser.add_argument("--image_print_freq", type=int, default=1000)
    
    # Path (Data & Checkpoint & Tensorboard)
    parser.add_argument('--dataset', type=str, default='echo', choices=["echo"])
    parser.add_argument('--log_dir', type=str, default='./logs', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='save checkpoint infos')
    parser.add_argument('--test_dir', type=str, help='load saved model')
    
    opt = parser.parse_args()

    opt.input_dim = 1
    opt.extrap = True
    
    if opt.phase == 'train':
        # Make Directory
        STORAGE_PATH = utils.create_folder_ifnotexist("./storage")
        STORAGE_PATH = STORAGE_PATH.resolve()
        LOG_PATH = utils.create_folder_ifnotexist(STORAGE_PATH / "logs")
        CKPT_PATH = utils.create_folder_ifnotexist(STORAGE_PATH / "checkpoints")

        # Modify Desc
        now = datetime.datetime.now()
        month_day = f"{now.month:02d}{now.day:02d}"
        opt.name = f"dataset{opt.dataset}_{opt.name}"
        opt.log_dir = utils.create_folder_ifnotexist(LOG_PATH / month_day / opt.name)
        opt.checkpoint_dir = utils.create_folder_ifnotexist(CKPT_PATH / month_day / opt.name)

        # Write opt information
        with open(str(opt.log_dir / 'options.json'), 'w') as fp:
            opt.log_dir = str(opt.log_dir)
            opt.checkpoint_dir = str(opt.checkpoint_dir)
            opt_dict = vars(opt)
            json.dump(opt_dict, fp=fp, indent=4)
            print("option.json dumped!")
            opt.log_dir = Path(opt.log_dir)
            opt.checkpoint_dir = Path(opt.checkpoint_dir)

        
        # read opt information
        # with open(str(opt.log_dir / 'options.json'), "r") as file:
        #     opt_dict = json.load(file)
        # opt = argparse.Namespace(**opt_dict)
        
        opt.train_image_path = utils.create_folder_ifnotexist(opt.log_dir / "train_images")
        opt.test_image_path = utils.create_folder_ifnotexist(opt.log_dir / "test_images")
    else:
        print("[Info] In test phase, skip dumping options.json..!")
    print(type(opt))
    return opt


def main():
    # Option
    opt = get_opt()
    print(opt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    
    # Dataloader
    # loader_objs = parse_datasets(opt, device)
    writer = SummaryWriter(log_dir=opt.log_dir)
    
    # Model
    model = VidODE(opt, device)

    data_loader = Echo_dynamic(root='/data/liujie/data/echocardiogram/EchoNet-Dynamic', split='ALL')
    train_loader = DataLoader(data_loader, batch_size=16, shuffle=True, num_workers=4)

    optimizer = optim.Adamax(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch)

    for epoch in range(opt.epoch):
        train_loss = train(opt, model, optimizer, train_loader, epoch, writer)
        scheduler.step()


        # Save model
        if (epoch) % 10 == 0:
            checkpoint_path = os.path.join(opt.checkpoint_dir, f'model_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

        # Save images
        if (epoch) % 10 == 0:
            save_images(opt, pred, epoch + 1)


def train(opt, model, optimizer, data_loader, epoch, writer):
    model.train()
    total_loss = 0.0
    pbar = tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{opt.epoch}')

    for i, (target, target_gt, fps, timestamps) in enumerate(data_loader):
        batch_dicts = {"observed_data": None,
                    "observed_tp": None,
                    "data_to_predict": None,
                    "tp_to_predict": None,
                    "observed_mask": None,
                    "mask_predicted_data": None
                    }
        b,c,t,w,h = target.shape
        batch_dicts['observed_data'] = target[:,:,:t//2,:,:].transpose(1, 2)
        batch_dicts['observed_tp'] = timestamps[0,:t//2]
        batch_dicts['observed_mask'] = torch.ones((b, t // 2, 1))
        batch_dicts['data_to_predict'] = target[:,:,t//2:,:,:].transpose(1, 2)
        batch_dicts['tp_to_predict'] = timestamps[0,t//2:]
        batch_dicts['mask_predicted_data'] = torch.ones((b, t // 2, 1))

        res = model.compute_all_losses(batch_dicts)
        loss = res["loss"]
        pred = res["pred_y"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)

        # Update progress bar
        pbar.set_postfix(loss=avg_loss)
        pbar.update(1)

        # Log learning rate to TensorBoard
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + i)
    
    pbar.close()
    return avg_loss

def save_images(opt, pred, epoch):
    if not os.path.exists(opt.train_image_path):
        os.makedirs(opt.train_image_path)

    for i in range(pred.size(0)):
        for j in range(pred.size(1)):
            image = pred[i, j, 0, :, :].cpu().numpy()  # Assuming single-channel grayscale images
            image_path = os.path.join(opt.train_image_path, f'{epoch}_{i}_{j}.png')
            utils.save_image(image, image_path)


def train_old(opt, netG, loader_objs, device):
    # Optimizer
    optimizer_netG = optim.Adamax(netG.parameters(), lr=opt.lr)
    
    train_dataloader = loader_objs['train_dataloader']
    test_dataloader = loader_objs['test_dataloader']
    n_train_batches = loader_objs['n_train_batches']
    n_test_batches = loader_objs['n_test_batches']
    total_step = 0
    start_time = time.time()
    
    for epoch in range(opt.epoch):
        
        utils.update_learning_rate(optimizer_netG, decay_rate=0.99, lowest=opt.lr / 10)
        
        for it in range(n_train_batches):
            
            data_dict = utils.get_data_dict(train_dataloader)
            batch_dict = utils.get_next_batch(data_dict)
            
            res = netG.compute_all_losses(batch_dict)
            loss_netG = res["loss"]
            
            # Compute Adversarial Loss
            real = batch_dict["data_to_predict"]
            fake = res["pred_y"]
            input_real = batch_dict["observed_data"]

            # Filter out mask
            if opt.irregular:
                b, _, c, h, w = real.size()
                observed_mask = batch_dict["observed_mask"]
                mask_predicted_data = batch_dict["mask_predicted_data"]

                selected_timesteps = int(observed_mask[0].sum())
                input_real = input_real[observed_mask.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
                real = real[mask_predicted_data.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)

            
            # Train G
            optimizer_netG.zero_grad()
            loss_netG.backward()
            optimizer_netG.step()
            
            if (total_step + 1) % opt.log_print_freq == 0 or total_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = f"Elapsed [{et}] Epoch [{epoch:03d}/{opt.epoch:03d}]\t"\
                        f"Iterations [{(total_step + 1):6d}] \t"\
                        f"Mse [{res['loss'].item():.4f}]\t"\
                        f"Adv_G [{loss_adv_netG.item():.4f}]\t"\
                        f"Adv_D [{loss_netD.item():.4f}]"
                
                print(log)

            if (total_step + 1) % opt.ckpt_save_freq == 0 or (epoch + 1 == opt.epoch and it + 1 == n_train_batches) or total_step == 0:
                utils.save_checkpoint(netG, os.path.join(opt.checkpoint_dir, f"ckpt_{(total_step + 1):08d}.pth"))
            
            total_step += 1



if __name__ == '__main__':
    main()