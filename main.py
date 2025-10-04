import argparse
import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import time
from tensorboardX import SummaryWriter

from core_scripts.startup_config import set_random_seed
from datautils import SUPPORTED_DATALOADERS
from model import SUPPORTED_MODELS
from datautils.data_utils import genSpoof_list, Dataset_eval, Dataset_train


############################################
# Evaluation Function
############################################
def eval_model(args, config, device):
    
    # Load protocol
    file_eval = genSpoof_list(os.path.join(args.database_path, "protocol.txt"), is_eval=True)

    # Dataset
    eval_set = Dataset_eval(list_IDs=file_eval, base_dir=args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load Model
    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(config["model"], device)

    if args.model_path:
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded model checkpoint from {args.model_path}")

    model = model.to(device)
    model.eval()

    # Run evaluation
    with torch.no_grad(), open(args.eval_output, "w") as f:
        for batch_x, utt_id in tqdm(eval_loader, ncols=90):
            batch_x = batch_x.to(device)
            out = model(batch_x)
            logits = out["logits"]
            scores = torch.softmax(logits, dim=1)  # [bonafide, spoof]
            for u, s in zip(utt_id, scores.cpu().numpy()):
                f.write(f"{u} {s[0]:.6f} {s[1]:.6f}\n")

    print(f"[INFO] Eval scores saved to {args.eval_output}")


############################################
# Training Functions
############################################
def train_epoch(train_loader, model, optimizer, device):
    import torch.nn.functional as F
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for info, batch_x, batch_y in tqdm(train_loader, ncols=90):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
        out = model(batch_x, labels=batch_y)
        loss = out["loss"]
        logits = out["logits"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    return running_loss, acc


def eval_epoch(dev_loader, model, device):
    import torch.nn.functional as F
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for info, batch_x, batch_y in tqdm(dev_loader, ncols=90):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
            out = model(batch_x, labels=batch_y)
            loss = out["loss"]
            logits = out["logits"]

            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total * 100
    return val_loss, acc


############################################
# Main
############################################
def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # ========== EVAL MODE ==========
    if args.eval:
        eval_model(args, config, device)
        sys.exit(0)

    # ========== TRAIN MODE ==========
    print("[INFO] Training mode activated")
    set_random_seed(args.seed)

    # Load dataloader
    if config["data"]["name"] not in SUPPORTED_DATALOADERS:
        raise ValueError(f"Dataloader {config['data']['name']} not supported")
    genList = importlib.import_module("datautils." + config["data"]["name"]).genSpoof_list
    Dataset_for = importlib.import_module("datautils." + config["data"]["name"]).Dataset_train

    d_label_trn, file_train = genList(os.path.join(args.database_path, "protocol.txt"), is_train=True)
    d_label_dev, file_dev = genList(os.path.join(args.database_path, "protocol.txt"), is_train=False)

    train_set = Dataset_for(args, list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path, algo=args.algo)
    dev_set = Dataset_for(args, list_IDs=file_dev, labels=d_label_dev, base_dir=args.database_path, algo=args.algo)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    if config["model"]["name"] not in SUPPORTED_MODELS:
        raise ValueError(f"Model {config['model']['name']} not supported")
    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(config["model"], device).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # Train loop
    for epoch in range(args.start_epoch, args.num_epochs):
        tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, device)
        val_loss, val_acc = eval_epoch(dev_loader, model, device)
        print(f"Epoch {epoch}: Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")


############################################
# Entry
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFM-ADD Training/Evaluation")

    # Dataset
    parser.add_argument("--database_path", type=str, default="/AISRC2/Dataset/")
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="CCE")

    # Misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)

    # Eval mode
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default="eval_scores.txt")


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    args = parser.parse_args()
    main(args)
