import logging
import os
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from utils.BriPCDMulti_new import BriPCDMulti
from models.model import EnhancedPointNet2, BridgeStructureLoss
from utils.BriPCDMulti_new import BriPCDMulti

# Configure matplotlib for better visualization
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Define scientific color palette
sci_colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', 
              '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']

def setup_logging(log_dir):
    """Set up logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir + '/testing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Configuration parameters
    config = {
        'num_points': 4096,
        'batch_size': 16,
        'num_workers': 6,
        'num_classes': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'exp_dir': 'experiments/YBC/exp_051419_PTV3_YBC_Bridge',
        'val_dir': 'data/YBC/val',
    }

    exp_dir = config['exp_dir']
    checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
    logger = setup_logging(exp_dir)
    device = torch.device(config['device'])
    logger.info(f'Using device: {device}')

    # Class names for the dataset
    class_names = ['Background', 'U_Flg', 'Web', 'B_Flg', 'Vert_Stiff', 
                   'Vert_Stiff_other', 'Horz_Stiff', 'Horz_Stiff_other']

    # Initialize dataset and dataloader
    test_dataset = BriPCDMulti(
        data_dir=config['val_dir'],
        num_points=config['num_points'],
        block_size=1.0,
        sample_rate=0.4,
        logger=logger
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Load model
    model = EnhancedPointNet2(config['num_classes']).to(device)
    logger.info("Initialized PointTransformerV3")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {checkpoint_path}')
    else:
        logger.error(f'No checkpoint found at {checkpoint_path}')
        return

    model.eval()

    # Create output directories
    output_dir = Path(exp_dir) / 'predicted_las'
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(exp_dir) / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)

    val_acc = AverageMeter()
    confusion_matrix = torch.zeros(config['num_classes'], config['num_classes']).to(device)

    with torch.no_grad():
        all_preds = []
        all_labels = []

        pbar = tqdm(test_loader, total=len(test_loader), desc='Processing files', position=0, leave=True)

        for batch in pbar:
            points = batch['points'].to(device, non_blocking=True)
            colors = batch['colors'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            outputs = model(points, colors)

            # Calculate accuracy
            pred = outputs.max(1)[1]
            acc_1 = pred.eq(labels).float().mean()
            val_acc.update(acc_1.item())

            all_preds.append(pred)
            all_labels.append(labels)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Update progress bar
            pbar.set_postfix({'Val_Acc': f'{val_acc.avg * 100:.2f}%'})

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        total_acc = all_preds.eq(all_labels).float().mean()

        # Calculate metrics
        metrics = calculate_metrics(confusion_matrix)

        # Log results
        logger.info("\n===== Evaluation Results =====")
        logger.info(f"Mean IoU: {metrics['mIoU'] * 100:.2f}%")
        logger.info(f"Overall Accuracy: {metrics['OA'] * 100:.2f}%")
        logger.info(f"Mean Accuracy: {metrics['mAcc'] * 100:.2f}%")
        logger.info(f"Precision: {metrics['Precision'] * 100:.2f}%")
        logger.info(f"Recall: {metrics['Recall'] * 100:.2f}%")
        logger.info(f"F1 Score: {metrics['F1_score'] * 100:.2f}%")

        for i, class_iou in enumerate(metrics['IoU_per_class']):
            logger.info(f"Class {i} ({class_names[i]}) IoU: {class_iou * 100:.2f}%")

def calculate_metrics(confusion_matrix):
    """Calculate evaluation metrics from the confusion matrix."""
    cm = confusion_matrix.cpu().numpy()

    # Calculate per-class IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou_per_class = intersection / (union + 1e-6)
    miou = np.nanmean(iou_per_class)

    # Calculate overall accuracy
    oa = np.sum(np.diag(cm)) / np.sum(cm)

    # Calculate per-class accuracy
    acc_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    macc = np.nanmean(acc_per_class)

    # Calculate precision and recall
    precision_per_class = np.diag(cm) / (np.sum(cm, axis=0) + 1e-6)
    recall_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)

    # Weighted precision and recall
    weights = np.sum(cm, axis=1) / np.sum(cm)
    precision = np.sum(precision_per_class * weights)
    recall = np.sum(recall_per_class * weights)

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'OA': oa,
        'mAcc': macc,
        'Acc_per_class': acc_per_class,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'Confusion_Matrix': cm
    }

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
    main()
