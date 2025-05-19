# train.py
import datetime
import logging
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.model import EnhancedPointNet2, BridgeStructureLoss
from utils.BriPCDMulti_new import BriPCDMulti
from utils.logger_config import initialize_logger, get_logger

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Configuration parameters
config = {
    'num_points': 4096,
    'chunk_size': 4096,
    'overlap': 1024,
    'batch_size': 16,
    'num_workers': 6,
    'learning_rate': 0.001,
    'num_classes': 5,
    'num_epochs': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train():
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime('%m%d%H')
    case = 'CB_onepart_bridgeSeg_new_1'
    exp_dir = Path(f'experiments/exp_{timestamp}_{case}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Paths to training and validation data
    train_dir = 'data/all/train'
    val_dir = 'data/all/val'

    # Set up tensorboard and logging
    writer = SummaryWriter(exp_dir / 'tensorboard')
    logger = initialize_logger(exp_dir)

    # Log configuration
    logger.info("Training configuration:")
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    # Set device
    device = torch.device(config['device'])
    logger.info(f'Using device: {device}')

    # Create data loaders
    train_dataset = BriPCDMulti(
        data_dir=train_dir,
        num_points=4096,
        block_size=1.0,
        sample_rate=0.4,
        logger=get_logger(),
        transform=True
    )

    val_dataset = BriPCDMulti(
        data_dir=val_dir,
        num_points=4096,
        block_size=1.0,
        sample_rate=0.4,
        logger=get_logger()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Initialize model
    num_classes = config['num_classes']
    model = EnhancedPointNet2(num_classes).to(device)

    # Copy model and utility files to experiment directory
    source_path = Path('models')
    destination_path = exp_dir / 'models'
    utils_path = Path('utils')
    shutil.copytree(source_path, destination_path)
    shutil.copytree(utils_path, exp_dir / 'utils')
    shutil.copy2('train_MulSca_BriStruNet_CB.py', exp_dir / 'train_MulSca_BriStruNet_CB.py')

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    # Compute class weights
    class_weights = compute_class_weights(train_loader, num_classes)
    criterion = BridgeStructureLoss(
        num_classes=5,
        alpha=80,
        rel_margin=0.3,
        class_weights=class_weights
    )

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch}/{config["num_epochs"]}] Training',
                    total=len(train_loader), position=0, leave=True)

        for batch in pbar:
            points = batch['points'].to(device)
            colors = batch['colors'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(points, colors)
            loss = criterion(outputs, labels, points=points)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = outputs.max(1)[1]
            acc = pred.eq(labels).float().mean()

            # Update metrics
            train_loss.update(loss.item())
            train_acc.update(acc.item())

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{train_loss.avg:.4f}',
                'Acc': f'{train_acc.avg * 100:.2f}%'
            })

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Validation phase
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            all_preds = []
            all_labels = []
            pbar = tqdm(val_loader, desc=f'Epoch [{epoch}/{config["num_epochs"]}] Validating',
                        total=len(val_loader), position=0, leave=True)
            for batch in pbar:
                points = batch['points'].to(device)
                colors = batch['colors'].to(device)
                labels = batch['labels'].to(device, non_blocking=True)

                outputs = model(points, colors)
                loss = criterion(outputs, labels, points=points)

                # Calculate accuracy
                pred = outputs.max(1)[1]
                acc_1 = pred.eq(labels).float().mean()

                # Update metrics
                val_loss.update(loss.item())
                val_acc.update(acc_1.item())

                all_preds.append(pred)
                all_labels.append(labels)

                # Update progress bar
                pbar.set_postfix({
                    'Val_Loss': f'{val_loss.avg:.4f}',
                    'Val_Acc': f'{val_acc.avg * 100:.2f}%'
                })

        # Update learning rate
        scheduler.step(val_acc.avg)

        # Log metrics
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}:")
        logger.info(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg * 100:.2f}%")
        logger.info(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg * 100:.2f}%")

        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/val', val_acc.avg, epoch)

        # Save the best model
        if val_acc.avg > best_val_acc:
            best_val_acc = val_acc.avg
            best_model_path = exp_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss.avg,
            }, best_model_path)
            logger.info(f"Saved best model with accuracy: {best_val_acc * 100:.2f}%")

        # Save the latest checkpoint
        checkpoint_path = exp_dir / 'latest_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc.avg,
            'val_loss': val_loss.avg,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    logger.info("Training completed!")

def compute_class_weights(dataset, num_classes=None):
    """Compute class weights for imbalanced datasets."""
    class_counts = torch.zeros(num_classes)
    for batch in dataset:
        labels = batch['labels']
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum()

    class_counts = class_counts + 1e-6  # Avoid division by zero
    total = class_counts.sum()
    weights = total / (class_counts * num_classes)
    weights = torch.clamp(weights, min=0.5, max=3.0)  # Clamp weights to avoid extreme values
    logging.info(f"Class weights: {weights}")
    return weights

class AverageMeter(object):
    """Tracks and computes the average and current value of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.exception("Training failed with exception:")
        raise
