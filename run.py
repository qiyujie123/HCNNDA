import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from model import HierarchicalConsensusPredictor
from datasets import load_lncRNA_drug_data


def configure_args():
    parser = argparse.ArgumentParser(description='Hierarchical Consensus Network Training')
    parser.add_argument('--gpu', type=str, default='', help='GPU device ID')
    parser.add_argument('--data_dir', default='data/', help='Data directory')
    parser.add_argument('--log_dir', default='./logs', help='Log directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=64, help='Encoder hidden dimension')
    parser.add_argument('--pred_dim', type=int, default=32, help='Predictor dimension')
    return parser.parse_args()


def prepare_data(args):
    cfg = {
        'data_dir': args.data_dir,
        'lncRNA_view_files': [
            'lncrna doca sim.csv',
            'lncrna exe sim.csv',
            'lncrna-seq-sim.csv',
        ],
        'drug_view_files': [
            'drug atc sim.csv',
            'drug smiles sim.csv',
            'drug tar sim.csv',
        ],
        'adj_file': 'association_matrix.csv'
    }
    lnc_views, drug_views, adj = load_lncRNA_drug_data(cfg)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    # 正样本
    pos = np.argwhere(adj == 1)

    # 负样本，随机采样为正样本数量的3倍（防止数量不足）
    neg = np.argwhere(adj == 0)
    rng = np.random.default_rng(args.seed)
    num_neg = min(len(neg), 1 * len(pos))
    neg_sampled = neg[rng.choice(len(neg), size=num_neg, replace=False)]

    # 组合正负样本
    pairs = np.vstack([pos, neg_sampled])
    labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg_sampled))])

    return lnc_views, drug_views, torch.LongTensor(pairs), torch.FloatTensor(labels)


def train_epoch(model, loader, optimizer, lnc_views, drug_views, device, args):
    model.train()
    total_loss = 0.0
    for pairs, labels in loader:
        pairs, labels = pairs.to(device), labels.to(device)
        lnc_feats = [v[pairs[:, 0]].to(device) for v in lnc_views]
        drug_feats = [v[pairs[:, 1]].to(device) for v in drug_views]
        optimizer.zero_grad()
        preds, losses = model(lnc_feats, drug_feats)
        cls_loss = F.binary_cross_entropy(preds.squeeze(), labels)
        loss = cls_loss + args.lambda1 * losses['coding'] + args.lambda2 * losses['global']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, lnc_views, drug_views, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for pairs, labels in loader:
            pairs = pairs.to(device)
            lnc_feats = [v[pairs[:, 0]].to(device) for v in lnc_views]
            drug_feats = [v[pairs[:, 1]].to(device) for v in drug_views]
            outputs, _ = model(lnc_feats, drug_feats)
            preds.extend(outputs.squeeze().cpu().numpy())
            truths.extend(labels.numpy())
    return roc_auc_score(truths, preds), average_precision_score(truths, preds)


def run_training(args, lnc_views, drug_views, pairs, labels, device):
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    aucs, auprs = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(pairs, labels)):
        print(f"\n=== Fold {fold + 1}/{args.folds} ===")
        train_pairs, train_labels = pairs[train_idx], labels[train_idx]
        test_pairs, test_labels = pairs[test_idx], labels[test_idx]
        train_loader = DataLoader(TensorDataset(train_pairs, train_labels), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_pairs, test_labels), batch_size=args.batch_size)

        model = HierarchicalConsensusPredictor(
            lncRNA_dims=[v.shape[1] for v in lnc_views],
            drug_dims=[v.shape[1] for v in drug_views],
            hidden_dim=args.hidden_dim,
            pred_dim=args.pred_dim
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_auc, best_aupr, early_stop_cnt = 0, 0, 0
        best_model_state = None

        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, lnc_views, drug_views, device, args)
            auc, aupr = evaluate(model, test_loader, lnc_views, drug_views, device)
            print(f"Epoch {epoch + 1}: AUC={auc:.4f}, AUPR={aupr:.4f}")

            if auc > best_auc:
                best_auc, best_aupr = auc, aupr
                best_model_state = model.state_dict()
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt >= args.early_stop:
                print("Early stopping.")
                break

        # 恢复最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)

        aucs.append(best_auc)
        auprs.append(best_aupr)

    print("\n=== Final Results ===")
    print(f"AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")


def main():
    args = configure_args()
    device = torch.device('cuda:0') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')

    # 固定随机种子，保证复现性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.log_dir, exist_ok=True)

    lnc_views, drug_views, pairs, labels = prepare_data(args)
    run_training(args, lnc_views, drug_views, pairs, labels, device)


if __name__ == '__main__':
    main()
