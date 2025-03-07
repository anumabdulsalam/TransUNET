import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_Queens import Queens_dataset, RandomGenerator
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume


def evaluate_model(args, model, best_performance):
    """Evaluates the model on the validation dataset."""
    dataset = Queens_dataset(base_dir=args.volume_path, split="val_vol", list_dir=args.list_dir,
                              nclass=args.num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(dataloader)} test iterations per epoch")

    model.eval()
    total_metrics = 0.0

    for _, batch in tqdm(enumerate(dataloader)):
        img, lbl, case_name = batch["image"], batch["label"], batch['case_name'][0]
        metrics = val_single_volume(img, lbl, model, classes=args.num_classes,
                                    patch_size=[args.img_size, args.img_size],
                                    case=case_name, z_spacing=args.z_spacing)
        total_metrics += np.array(metrics)

    avg_metrics = total_metrics / len(dataset)
    mean_performance = np.mean(avg_metrics, axis=0)
    logging.info(f'Testing performance: mean_dice = {mean_performance:.6f}, best_dice = {best_performance:.6f}')
    return mean_performance


def train_model(args, model, save_path):
    """Trains the model on the dataset."""
    logging.basicConfig(filename=os.path.join(save_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu

    dataset = Queens_dataset(base_dir="Root\\skin\\NIDI", list_dir=args.list_dir, split="train",
                              nclass=args.num_classes,
                              transform=transforms.Compose(
                                  [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    logging.info(f"Training dataset size: {len(dataset)}")

    def seed_worker(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            worker_init_fn=seed_worker)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    loss_ce = CrossEntropyLoss()
    loss_dice = DiceLoss(args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(save_path, 'log'))

    iter_count = 0
    max_iterations = args.max_epochs * len(dataloader)
    logging.info(f"{len(dataloader)} iterations per epoch, {max_iterations} max iterations")

    best_performance = 0.0
    label_combinations = [x for x in powerset([0, 1, 2, 3])]

    for epoch in range(args.max_epochs):
        for _, batch in enumerate(dataloader):
            img_batch, lbl_batch = batch['image'].cuda(), batch['label'].squeeze(1).cuda()

            predictions = model(img_batch)
            total_loss = 0.0
            weight_ce, weight_dice = 0.3, 0.7

            for subset in label_combinations:
                if not subset:
                    continue

                subset_output = sum(predictions[idx] for idx in subset)
                loss_ce_val = loss_ce(subset_output, lbl_batch.long())
                loss_dice_val = loss_dice(subset_output, lbl_batch, softmax=True)
                total_loss += (weight_ce * loss_ce_val + weight_dice * loss_dice_val)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_count += 1
            writer.add_scalar('info/lr', base_lr, iter_count)
            writer.add_scalar('info/total_loss', total_loss.item(), iter_count)

            if iter_count % 50 == 0:
                logging.info(
                    f'Iteration {iter_count}, Epoch {epoch}: Loss = {total_loss.item():.6f}, LR = {base_lr:.6f}')

        logging.info(f'End of Epoch {epoch}: Loss = {total_loss.item():.6f}, LR = {base_lr:.6f}')

        model_path = os.path.join(save_path, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Model saved to {model_path}')

        if epoch == args.max_epochs - 1:
            break

    writer.close()
    return "Training Complete!"
