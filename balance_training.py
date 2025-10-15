import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import importlib
import time
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from datautils import SUPPORTED_DATALOADERS
from model import SUPPORTED_MODELS

import pandas as pd
from evaluate_metrics import compute_eer

############################################
# Evaluation Function
############################################
def eval_model(args, config, device):
    data_module = importlib.import_module("datautils." + config["data"]["name"])
    genSpoof_list = data_module.genSpoof_list
    Dataset_eval = data_module.Dataset_eval
    # Load protocol
    file_eval = genSpoof_list(args.protocol_path, is_eval=True)

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
    with torch.no_grad(), open(args.eval_output, 'w') as fh:
        for batch_x, utt_id in tqdm(eval_loader, desc="Evaluation", leave=False):
            batch_x = batch_x.to(device)

            # Model returns (logits, attn_score)
            logits, _ = model(batch_x)
            score_list = logits.cpu().numpy().tolist()

            for f, scores in zip(utt_id, score_list):
                # scores[0]: spoof, scores[1]: bonafide
                fh.write('{} {} {}\n'.format(f, scores[0], scores[1]))

    print(f"[INFO] Eval scores saved to {args.eval_output}")

    eer(args)
    print("[INFO] Evaluation complete.")

def eer(args):
    eval_df = pd.read_csv(args.protocol_path, sep=" ", header=None)
    eval_df.columns = ["utt","subset","label"]
    pred_df = pd.read_csv(args.eval_output, sep=" ", header=None)
    pred_df.columns = ["utt", "spoof", "bonafide"]

    res_df = pd.merge(eval_df, pred_df, on='utt')

    spoof_scores = res_df[res_df['label'] == 'spoof']['bonafide']
    bonafide_scores = res_df[res_df['label'] == 'bonafide']['bonafide']

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print("EER: {:.4f}%, threshold: {:.4f}".format(eer*100, threshold))

############################################
# Training Functions
############################################
def train_epoch(train_loader, model, optimizer, device, epoch):

    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Define weighted CrossEntropyLoss (bonafide=0.1, spoof=0.9)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Unpack train_loader (sampler, base_dir, args)
    train_sampler, base_dir, args = train_loader

    # Import collate function
    import importlib
    from collections import defaultdict
    data_module = importlib.import_module("datautils.data_utils_balance")
    online_augmentation_collate_fn = data_module.online_augmentation_collate_fn

    # Track augmentation statistics for the entire epoch
    epoch_aug_stats = defaultdict(int)

    pbar = tqdm(train_sampler, ncols=120, desc=f"Epoch {epoch} [Train]", total=len(train_sampler))
    for batch_idx, batch_info in enumerate(pbar):
        # Apply online augmentation and create batch
        # Log augmentation only for first batch to verify
        log_aug = (batch_idx == 0)
        if log_aug:
            batch_x, batch_y, _, batch_aug_stats = online_augmentation_collate_fn(batch_info, base_dir, args, log_augmentation=True)
            # Accumulate stats
            for aug_type, count in batch_aug_stats.items():
                epoch_aug_stats[aug_type] += count
            # Print immediately after first batch
            print(f"\n[Augmentation Summary - First Batch of Epoch {epoch}]")
            for aug_type in sorted(batch_aug_stats.keys()):
                print(f"  ✓ {aug_type}: {batch_aug_stats[aug_type]} samples")
            print()  # Empty line for readability
        else:
            batch_x, batch_y, _ = online_augmentation_collate_fn(batch_info, base_dir, args)

        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

        # Model returns (logits, attn_score)
        logits, _ = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
        num_batches += 1

        # Update progress bar with current loss and accuracy
        current_avg_loss = running_loss / num_batches
        current_acc = correct / total * 100
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = running_loss / num_batches
    acc = correct / total * 100
    return avg_loss, acc


def eval_epoch(dev_loader, model, device, epoch):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Define weighted CrossEntropyLoss (bonafide=0.1, spoof=0.9)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(dev_loader, ncols=120, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_x, batch_y, _ in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

            # Model returns (logits, attn_score)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)

            val_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            num_batches += 1

            # Update progress bar
            current_avg_loss = val_loss / num_batches
            current_acc = correct / total * 100
            pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = val_loss / num_batches
    acc = correct / total * 100
    return avg_loss, acc


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

    # ---------------------------------------
    # Load dataloader
    # ---------------------------------------
    if config["data"]["name"] not in SUPPORTED_DATALOADERS:
        raise ValueError(f"Dataloader {config['data']['name']} not supported")

    # 모듈 로드
    data_module = importlib.import_module("datautils." + config["data"]["name"])
    genList = data_module.genSpoof_list
    Dataset_for = data_module.Dataset_train
    Dataset_eval = data_module.Dataset_eval
    BalancedNoiseSampler = data_module.BalancedNoiseSampler
    online_augmentation_collate_fn = data_module.online_augmentation_collate_fn

    print(f"[INFO] Using protocol file: {args.protocol_path}")

    # ---------------------------------------
    # Train/Dev Split 불러오기
    # ---------------------------------------
    d_label_trn, file_train, noise_trn, clean_bonafide_trn, clean_spoof_trn = genList(args.protocol_path, is_train=True)
    d_label_dev, file_dev, noise_dev = genList(args.protocol_path, is_train=False)

    print(f"[INFO] Loaded {len(file_train)} training and {len(file_dev)} validation samples")
    print(f"[INFO] Clean bonafide: {len(clean_bonafide_trn)}, Clean spoof: {len(clean_spoof_trn)}")

    # ---------------------------------------
    # Dataset 생성 (dev set only, train will use collate_fn)
    # ---------------------------------------
    dev_set = Dataset_for(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        noise_labels=noise_dev,
        base_dir=args.database_path,
        algo=args.algo
    )

    # ---------------------------------------
    # DataLoader 설정 with Online Augmentation
    # ---------------------------------------
    # For train: use custom sampler that directly generates batches
    train_sampler = BalancedNoiseSampler(
        dataset=None,  # Not used, sampler generates (utt_id, noise_type, label) tuples
        noise_labels=noise_trn,
        label_dict=d_label_trn,
        clean_bonafide_list=clean_bonafide_trn,
        clean_spoof_list=clean_spoof_trn,
        base_dir=args.database_path,
        batch_size=24  # 11 aug bonafide + 11 aug spoof + 2 clean = 24
    )

    # Store sampler and args for use in train_epoch
    train_loader = (train_sampler, args.database_path, args)

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )


    # Load model
    if config["model"]["name"] not in SUPPORTED_MODELS:
        raise ValueError(f"Model {config['model']['name']} not supported")

    # Transfer model config to args for conformertcm compatibility
    if "emb_size" in config["model"]:
        args.emb_size = config["model"]["emb_size"]
    if "heads" in config["model"]:
        args.heads = config["model"]["heads"]
    if "kernel_size" in config["model"]:
        args.kernel_size = config["model"]["kernel_size"]
    if "num_encoders" in config["model"]:
        args.num_encoders = config["model"]["num_encoders"]

    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(args, device).to(device)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # Setup logging
    log_dir = f"logs/{args.comment}" if args.comment else "logs/default"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Setup text log file
    log_file = os.path.join(log_dir, "training.log")
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.patience

    # Create directory for model saving
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # Train loop
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")

    start_time_total = time.time()

    for epoch in range(args.start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*80}")

        epoch_start_time = time.time()

        # Train
        tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, device, epoch)

        # Validate
        val_loss, val_acc = eval_epoch(dev_loader, model, device, epoch)

        epoch_time = time.time() - epoch_start_time

        # Calculate loss delta
        if epoch > args.start_epoch:
            loss_delta = val_loss - prev_val_loss
            loss_delta_str = f"({loss_delta:+.4f})"
        else:
            loss_delta_str = ""

        prev_val_loss = val_loss

        # Print summary
        print(f"\n{'─'*80}")
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"{'─'*80}")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} {loss_delta_str} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"{'─'*80}")

        # Log to tensorboard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.2f},"
                    f"{val_loss:.4f},{val_acc:.2f}\n")

        # Early stopping check
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), args.model_save_path)
            print(f"✅ Validation loss improved by {improvement:.4f}! Model saved to {args.model_save_path}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                break

    total_time = time.time() - start_time_total

    writer.close()

    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Logs saved to: {log_dir}")
    print(f"  Model saved to: {args.model_save_path}")
    print("="*80 + "\n")


############################################
# Entry
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFM-ADD Training/Evaluation")

    # Dataset
    parser.add_argument("--database_path", type=str, default="/AISRC2/Dataset/")
    parser.add_argument("--protocol_path", type=str, default="protocol.txt")
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=26)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="CCE")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default="out/best_model.pth", help="Path to save the best model")

    # Eval mode
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default="eval_scores.txt")


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')
    parser.add_argument('--rb_prob', type=float, default=0.5)
    parser.add_argument('--rb_random', action='store_true',
                    help='Use random selection of RawBoost algorithms')
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
