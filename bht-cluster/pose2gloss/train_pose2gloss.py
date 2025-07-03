import os
import json
import csv
import argparse
import math
import logging
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Logging Setup
# ------------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ------------------------------
# Dataset & Vocabulary
# ------------------------------
class PoseGlossDataset(Dataset):
    def __init__(self, json_dir: str, gloss2idx: dict):
        self.json_paths = []
        self.index_map = []
        self.gloss2idx = gloss2idx
        files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        for file_idx, fname in enumerate(files):
            path = os.path.join(json_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)['data']
            self.json_paths.append(path)
            for sample_idx in range(len(data)):
                self.index_map.append((file_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        path = self.json_paths[file_idx]
        with open(path, 'r', encoding='utf-8') as f:
            item = json.load(f)['data'][sample_idx]
        pose_seq = []
        for frame in item['pose_sequence']:
            vec = frame['pose_keypoints_2d'] + \
                  frame['face_keypoints_2d'] + \
                  frame['hand_left_keypoints_2d'] + \
                  frame['hand_right_keypoints_2d']
            pose_seq.append(vec)
        poses = torch.tensor(pose_seq, dtype=torch.float)
        glosses = item['gloss_sequence'].split(',')
        gloss_idx = [self.gloss2idx[g] for g in glosses]
        gloss = torch.tensor(gloss_idx, dtype=torch.long)
        return poses, gloss

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    poses, glosses = zip(*batch)
    lengths = [p.size(0) for p in poses]
    gloss_lens = [g.size(0) for g in glosses]
    max_len = max(lengths)
    feat_dim = poses[0].size(1)
    padded = torch.zeros(len(poses), max_len, feat_dim)
    mask = torch.zeros(len(poses), max_len, dtype=torch.bool)
    for i, p in enumerate(poses):
        padded[i, :p.size(0), :] = p
        mask[i, :lengths[i]] = False
    target = torch.cat(glosses)
    return padded, mask, target, torch.tensor(lengths), torch.tensor(gloss_lens)

# ------------------------------
# Model Components
# ------------------------------
class PoseEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.size()
        y = self.net(x.view(B*T, D))
        return y.view(B, T, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :]

class Pose2GlossSLRT(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.pose_embed = PoseEmbedding(input_dim, d_model, dropout)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, poses: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        x = self.pose_embed(poses)
        x = self.pos_enc(x)
        x = x.transpose(0,1)
        z = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = z.transpose(0,1)
        logits = self.fc_out(z)
        return self.log_softmax(logits)

# ------------------------------
# Training
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss = 0
    logger.info(f"Starting Epoch {epoch_num}...")

    for batch_idx, (poses, mask, targets, in_lens, tgt_lens) in enumerate(loader):
        poses, mask, targets = poses.to(device), mask.to(device), targets.to(device)
        optimizer.zero_grad()
        log_probs = model(poses, src_key_padding_mask=mask)
        logp = log_probs.transpose(0,1)
        loss = criterion(logp, targets, in_lens, tgt_lens)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    logger.info(f"Epoch {epoch_num} Complete. Average Loss: {avg_loss:.4f}")
    return avg_loss

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, required=True, help='Directory of JSON files')
    parser.add_argument('--gloss_csv', type=str, default='unique-glosses.csv', help='CSV of unique glosses')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    logger.info("Starting Training Script...")
    logger.info(f"JSON directory: {args.json_dir}")
    logger.info(f"Gloss CSV: {args.gloss_csv}")
    logger.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    logger.info(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")

    gloss2idx = {"<blank>": 0}
    with open(args.gloss_csv, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.reader(f)):
            gloss2idx[row[0]] = i + 1
    logger.info(f"Loaded {len(gloss2idx)} glosses.")

    dataset = PoseGlossDataset(args.json_dir, gloss2idx)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(dataset[0][0][0])
    logger.info(f"Using device: {device}")
    logger.info(f"Pose input dimension: {input_dim}")

    model = Pose2GlossSLRT(input_dim, args.d_model, args.n_heads, args.n_layers, len(gloss2idx), args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, loader, optimizer, criterion, device, epoch)
