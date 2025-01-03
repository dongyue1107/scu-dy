import torch
from rdkit import RDLogger

from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
RDLogger.DisableLog('rdApp.*')
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch import optim
import os
import argparse
import numpy as np
import pandas as pd
import torch.jit

from torch.utils.tensorboard import SummaryWriter

from model import GINGraphPooling, MyEvaluator
from data import MyDataset
from valid_measure import train, eval, eval_accu, own_accu, eval_loss, eval_precision, eval_f1_score, eval_recall, test
from sklearn.metrics import r2_score
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_param_importances


def parse_args():

    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    # parser.add_argument('--num_classes', type=int, default=10,
    #                     help='number of dataset unique labels')
    parser.add_argument('--weight_decay', type=float, default=0.004,#0.0004
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,  # 4,
                        help='number of workers (default: 4)')
    parser.add_argument('--filepath', type=str, default='/home/dy/1/GIN/unique1-6.csv',help = 'path to the input file')

    args = parser.parse_args()

    return  args

import sys
sys.argv=['']
del sys



def preparation(args):

    save_dir = os.path.join('/home/dy/1/GIN/zengguang1', args.task_name)

    if os.path.exists(save_dir):
        for idx in range(1000):
            if not os.path.exists(save_dir + '=' + str(idx)):
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)


def objective(trial):
    args = parse_args()

    args.num_layers = trial.suggest_int('num_layers', 2, 8)
    args.emb_dim = trial.suggest_int('emb_dim', 64, 512)
    args.drop_ratio = trial.suggest_float('drop_ratio', 0.0, 0.5)
    args.batch_size = trial.suggest_int('batch_size', 16, 128)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)
    args.epochs = trial.suggest_int('epochs', 100, 300)
    args.early_stop = trial.suggest_int('early_stop', 10, 50)
    args.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    preparation(args)

    nn_params = {
        'num_layers':args.num_layers,
        'emb_dim' : args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
    }

    dataset = MyDataset(filepath=args.filepath)
    split_idx = dataset.get_idx_split()
    train_data = [dataset[idx] for idx in split_idx['train']]
    valid_data = [dataset[idx] for idx in split_idx['valid']]
    test_data = [dataset[idx] for idx in split_idx['test']]
    # valid_data += test_data
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    evaluator = MyEvaluator()
    criterion_fn = torch.nn.MSELoss()
    device = torch.device('cpu')
    model = GINGraphPooling(**nn_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
    writer = SummaryWriter(log_dir=args.save_dir)
    not_improved = 0
    best_accu = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion_fn)
        train_mae, train_y_true, train_y_pred = eval(model, device, train_loader, evaluator)
        train_accu = eval_accu(model, device, train_loader, evaluator)

        valid_mae, valid_y_true, valid_y_pred = eval(model, device, valid_loader, evaluator)
        valid_accu = eval_accu(model, device, valid_loader, evaluator)
        test_mae, test_y_true, test_y_pred = eval(model, device, test_loader, evaluator)
        test_accu = eval_accu(model, device, test_loader, evaluator)

        writer.add_scalar('valid/mae', valid_mae, epoch)
        writer.add_scalar('train/mae', train_mae, epoch)
        writer.add_scalar('valid/accu', valid_accu, epoch)
        writer.add_scalar('test/accu', test_accu, epoch)

        if valid_accu > best_accu:
            best_accu = valid_accu
            checkpoint = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_accu': best_accu
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))
            train_predictions = pd.DataFrame(
                {'y_true': np.array(train_y_true).flatten(), 'y_pred': np.array(train_y_pred).flatten()})
            valid_predictions = pd.DataFrame(
                {'y_true': np.array(valid_y_true).flatten(), 'y_pred': np.array(valid_y_pred).flatten()})
            test_predictions = pd.DataFrame(
                {'y_true': np.array(test_y_true).flatten(), 'y_pred': np.array(test_y_pred).flatten()})

            train_predictions.to_csv(os.path.join(args.save_dir, 'train_predictions.csv'), index=False)
            valid_predictions.to_csv(os.path.join(args.save_dir, 'valid_predictions.csv'), index=False)
            test_predictions.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index=False)

            evaluator.save_test_submission({'y_pred': test_y_pred}, args.save_dir)
            not_improved = 0

            with open(os.path.join(args.save_dir, f'{best_accu:.4f}'), 'w') as f:
                pass
        else:
            not_improved += 1
            if not_improved == args.early_stop:
                break

        scheduler.step()
    return best_accu

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')

    fig1 = plot_optimization_history(study)
    fig1.show()

    # 可视化参数重要性
    fig2 = plot_param_importances(study)
    fig2.show()

    # 可视化贝叶斯优化探索区域
    fig3 = plot_parallel_coordinate(study)
    fig3.show()

    fig4 = plot_slice(study)
    fig4.show()
