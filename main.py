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

    # Load protocol - different return values depending on data module
    eval_result = genSpoof_list(args.protocol_path, is_eval=True)
    if isinstance(eval_result, tuple):
        # data_utils_balance returns (file_list, noise_dict)
        file_eval, _ = eval_result
    else:
        # data_utils_curriculum returns file_list only
        file_eval = eval_result

    # Dataset
    eval_set = Dataset_eval(list_IDs=file_eval, base_dir=args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Transfer model config to args for conformertcm compatibility
    if "emb_size" in config["model"]:
        args.emb_size = config["model"]["emb_size"]
    if "heads" in config["model"]:
        args.heads = config["model"]["heads"]
    if "kernel_size" in config["model"]:
        args.kernel_size = config["model"]["kernel_size"]
    if "num_encoders" in config["model"]:
        args.num_encoders = config["model"]["num_encoders"]

    # Load Model
    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(args, device)

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
def train_epoch(train_loader, model, optimizer, device, epoch, use_balance_training=True, use_curriculum=False, curriculum_stage=1):

    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Define weighted CrossEntropyLoss (bonafide=0.1, spoof=0.9)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    if use_balance_training or use_curriculum:
        # BALANCE TRAINING MODE OR CURRICULUM LEARNING MODE
        # Unpack train_loader (sampler, base_dir, args)
        train_sampler, base_dir, args = train_loader

        # Import collate function
        import importlib
        from collections import defaultdict
        if use_curriculum:
            data_module = importlib.import_module("datautils.data_utils_curriculum")
        else:
            data_module = importlib.import_module("datautils.data_utils_balance")
        online_augmentation_collate_fn = data_module.online_augmentation_collate_fn

        # Track augmentation statistics for the entire epoch
        epoch_aug_stats = defaultdict(int)

        # Set description based on mode
        if use_curriculum:
            desc = f"Epoch {epoch} [Train - Stage {curriculum_stage}]"
        else:
            desc = f"Epoch {epoch} [Train]"

        pbar = tqdm(train_sampler, ncols=120, desc=desc, total=len(train_sampler))
        for batch_idx, batch_info in enumerate(pbar):
            # Apply online augmentation and create batch
            # Log augmentation only for first batch to verify
            log_aug = (batch_idx == 0)
            if log_aug:
                if use_curriculum:
                    batch_x, batch_y, _, batch_aug_stats = online_augmentation_collate_fn(
                        batch_info, base_dir, args, curriculum_stage=curriculum_stage, log_augmentation=True)
                else:
                    batch_x, batch_y, _, batch_aug_stats = online_augmentation_collate_fn(
                        batch_info, base_dir, args, log_augmentation=True)
                # Accumulate stats
                for aug_type, count in batch_aug_stats.items():
                    epoch_aug_stats[aug_type] += count
                # Print immediately after first batch
                if use_curriculum:
                    print(f"\n[Augmentation Summary - First Batch of Epoch {epoch} - Curriculum Stage {curriculum_stage}]")
                else:
                    print(f"\n[Augmentation Summary - First Batch of Epoch {epoch}]")
                for aug_type in sorted(batch_aug_stats.keys()):
                    print(f"  ✓ {aug_type}: {batch_aug_stats[aug_type]} samples")
                print()  # Empty line for readability
            else:
                if use_curriculum:
                    batch_x, batch_y, _ = online_augmentation_collate_fn(
                        batch_info, base_dir, args, curriculum_stage=curriculum_stage)
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
    else:
        # BASELINE TRAINING MODE
        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch} [Train]")
        for batch_x, batch_y in pbar:
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


def eval_epoch(dev_loader, model, device, epoch, use_balance_training=True):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Define weighted CrossEntropyLoss (bonafide=0.1, spoof=0.9)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(dev_loader, ncols=120, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        if use_balance_training:
            # BALANCE TRAINING MODE - dev loader returns (batch_x, batch_y, noise_type)
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
        else:
            # BASELINE TRAINING MODE - dev loader returns (batch_x, batch_y)
            for batch_x, batch_y in pbar:
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

    # Check training mode: curriculum, balance, or baseline
    use_curriculum = config["data"]["name"] == "data_utils_curriculum"
    use_balance_training = hasattr(data_module, 'BalancedNoiseSampler') and not use_curriculum

    if use_curriculum:
        print("[INFO] Using CURRICULUM LEARNING mode with CurriculumNoiseSampler")
        # Load curriculum config
        curriculum_config = config.get("curriculum", {})
        if not curriculum_config.get("enabled", False):
            print("[WARNING] Curriculum config found but not enabled. Using as balance training.")
            use_curriculum = False
            use_balance_training = True
        else:
            Dataset_train = data_module.Dataset_train
            Dataset_dev = data_module.Dataset_dev
            Dataset_eval = data_module.Dataset_eval
            CurriculumNoiseSampler = data_module.CurriculumNoiseSampler
            online_augmentation_collate_fn = data_module.online_augmentation_collate_fn
            print(f"[INFO] Curriculum stages: {len([k for k in curriculum_config.keys() if k.startswith('stage')])}")
    elif use_balance_training:
        print("[INFO] Using BALANCE TRAINING mode with BalancedNoiseSampler")
        Dataset_for = data_module.Dataset_train
        Dataset_dev = data_module.Dataset_dev
        Dataset_eval = data_module.Dataset_eval
        BalancedNoiseSampler = data_module.BalancedNoiseSampler
        online_augmentation_collate_fn = data_module.online_augmentation_collate_fn
    else:
        print("[INFO] Using BASELINE TRAINING mode with standard DataLoader")
        Dataset_train = data_module.Dataset_train
        Dataset_eval = data_module.Dataset_eval

    print(f"[INFO] Using protocol file: {args.protocol_path}")

    # ---------------------------------------
    # Train/Dev Split 불러오기
    # ---------------------------------------
    if use_balance_training or use_curriculum:
        d_label_trn, file_train, noise_trn, clean_bonafide_trn, clean_spoof_trn = genList(args.protocol_path, is_train=True)
        d_label_dev, file_dev, noise_dev = genList(args.protocol_path, is_train=False)
    else:
        d_label_trn, file_train = genList(args.protocol_path, is_train=True)
        d_label_dev, file_dev = genList(args.protocol_path, is_train=False)

    print(f"[INFO] Loaded {len(file_train)} training and {len(file_dev)} validation samples")
    if use_balance_training or use_curriculum:
        print(f"[INFO] Clean bonafide: {len(clean_bonafide_trn)}, Clean spoof: {len(clean_spoof_trn)}")

    # ---------------------------------------
    # Dataset 생성 및 DataLoader 설정
    # ---------------------------------------
    if use_curriculum:
        # CURRICULUM LEARNING MODE
        # Note: DataLoader will be recreated for each curriculum stage in training loop
        dev_set = Dataset_dev(
            list_IDs=file_dev,
            labels=d_label_dev,
            noise_labels=noise_dev,
            base_dir=args.database_path
        )

        dev_loader = DataLoader(
            dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Prepare data for curriculum sampler creation (done in training loop per stage)
        curriculum_data = {
            'noise_labels': noise_trn,
            'label_dict': d_label_trn,
            'clean_bonafide_list': clean_bonafide_trn,
            'clean_spoof_list': clean_spoof_trn,
            'base_dir': args.database_path,
            'args': args
        }

    elif use_balance_training:
        # BALANCE TRAINING MODE
        # Dev set: Apply random augmentation to spoof samples only
        # - Bonafide: use as-is (already has augmented samples)
        # - Spoof: randomly augment from 12 noise types (clean samples only)
        dev_set = Dataset_dev(
            list_IDs=file_dev,
            labels=d_label_dev,
            noise_labels=noise_dev,
            base_dir=args.database_path
        )

        # For train: use custom sampler that directly generates batches
        train_sampler = BalancedNoiseSampler(
            dataset=None,  # Not used, sampler generates (utt_id, noise_type, label) tuples
            noise_labels=noise_trn,
            label_dict=d_label_trn,
            clean_bonafide_list=clean_bonafide_trn,
            clean_spoof_list=clean_spoof_trn,
            base_dir=args.database_path,
            batch_size=args.batch_size  # 11 aug bonafide + 11 aug spoof + 2 clean = 24
        )

        # Store sampler and args for use in train_epoch
        train_loader = (train_sampler, args.database_path, args)

        dev_loader = DataLoader(
            dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        # BASELINE TRAINING MODE
        # Use standard Dataset and DataLoader
        train_set = Dataset_train(
            args=args,
            list_IDs=file_train,
            labels=d_label_trn,
            base_dir=args.database_path,
            algo=args.algo,
            rb_prob=args.rb_prob,
            random_algo=args.rb_random
        )

        dev_set = Dataset_train(
            args=args,
            list_IDs=file_dev,
            labels=d_label_dev,
            base_dir=args.database_path,
            algo=0,  # No augmentation for dev set
            rb_prob=0.0,
            random_algo=False
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

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

    if use_curriculum:
        # CURRICULUM LEARNING MODE
        # Parse curriculum stages from config
        stages = []
        for key in sorted(curriculum_config.keys()):
            # Only match stage1, stage2, etc (not stage_patience)
            if key.startswith('stage') and key[5:].isdigit():
                stage_num = int(key.replace('stage', ''))
                stage_epochs = curriculum_config[key]['epochs']
                stages.append((stage_num, stage_epochs))

        # Check early stopping strategy
        early_stopping_per_stage = curriculum_config.get('early_stopping_per_stage', False)
        stage_patience_limit = curriculum_config.get('stage_patience', 5)

        print(f"[INFO] Curriculum Learning Schedule:")
        if early_stopping_per_stage:
            print(f"  Early Stopping: PER-STAGE (patience={stage_patience_limit})")
            print(f"  Stage epochs = MAX limit (will stop early if no improvement)")
        else:
            print(f"  Early Stopping: GLOBAL (patience={patience})")
            print(f"  Stage epochs = FIXED")
        print()
        for stage_num, stage_epochs in stages:
            print(f"  Stage {stage_num}: {stage_epochs} epochs (max)" if early_stopping_per_stage else f"  Stage {stage_num}: {stage_epochs} epochs")
        print()

        epoch = args.start_epoch
        for stage_num, stage_epochs in stages:
            print(f"\n{'#'*80}")
            print(f"# CURRICULUM STAGE {stage_num}: {curriculum_config[f'stage{stage_num}']['description']}")
            print(f"{'#'*80}\n")

            # Initialize stage-specific early stopping variables
            if early_stopping_per_stage:
                stage_best_val_loss = float('inf')
                stage_patience_counter = 0
                print(f"[Stage {stage_num}] Per-stage early stopping enabled (patience={stage_patience_limit})\n")

            # Create curriculum sampler for this stage
            train_sampler = CurriculumNoiseSampler(
                dataset=None,
                noise_labels=curriculum_data['noise_labels'],
                label_dict=curriculum_data['label_dict'],
                clean_bonafide_list=curriculum_data['clean_bonafide_list'],
                clean_spoof_list=curriculum_data['clean_spoof_list'],
                base_dir=curriculum_data['base_dir'],
                batch_size=args.batch_size,
                curriculum_stage=stage_num
            )

            train_loader = (train_sampler, curriculum_data['base_dir'], curriculum_data['args'])

            # Train for this stage's epochs
            for stage_epoch in range(stage_epochs):
                if epoch >= args.num_epochs:
                    break

                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{args.num_epochs} [Stage {stage_num}, Epoch {stage_epoch+1}/{stage_epochs}]")
                print(f"{'='*80}")

                epoch_start_time = time.time()

                # Train
                tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, device, epoch,
                                             use_balance_training=False, use_curriculum=True,
                                             curriculum_stage=stage_num)

                # Validate
                val_loss, val_acc = eval_epoch(dev_loader, model, device, epoch, use_balance_training=True)

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
                print(f"📊 Epoch {epoch+1} Summary [Stage {stage_num}]:")
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
                writer.add_scalar("Curriculum/stage", stage_num, epoch)

                # Log to file
                with open(log_file, "a") as f:
                    f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.2f},"
                            f"{val_loss:.4f},{val_acc:.2f},{stage_num}\n")

                # Early stopping check
                if early_stopping_per_stage:
                    # PER-STAGE early stopping
                    if val_loss < stage_best_val_loss:
                        improvement = stage_best_val_loss - val_loss
                        stage_best_val_loss = val_loss
                        stage_patience_counter = 0

                        # Also update global best and save model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), args.model_save_path)
                            print(f"✅ [Stage {stage_num}] Validation loss improved by {improvement:.4f}! Model saved to {args.model_save_path}")
                        else:
                            print(f"✅ [Stage {stage_num}] Stage best improved by {improvement:.4f} (global best: {best_val_loss:.4f})")
                    else:
                        stage_patience_counter += 1
                        print(f"⚠️  [Stage {stage_num}] No improvement. Stage patience: {stage_patience_counter}/{stage_patience_limit}")

                        if stage_patience_counter >= stage_patience_limit:
                            print(f"\n🛑 [Stage {stage_num}] Stage early stopping triggered after {stage_epoch + 1} epochs")
                            print(f"   Moving to next stage...")
                            break
                else:
                    # GLOBAL early stopping (original behavior)
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

                epoch += 1

            # Check if global early stopping was triggered (only for non-per-stage mode)
            if not early_stopping_per_stage and patience_counter >= patience:
                break

    else:
        # BASELINE OR BALANCE TRAINING MODE (original loop)
        for epoch in range(args.start_epoch, args.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"{'='*80}")

            epoch_start_time = time.time()

            # Train
            tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, device, epoch, use_balance_training)

            # Validate
            val_loss, val_acc = eval_epoch(dev_loader, model, device, epoch, use_balance_training)

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
