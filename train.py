import os
import pickle
import argparse
from sympy import Line2D
import torch
import random
import numpy as np
from tqdm import tqdm
from end_point import *
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pypots.optim.adam import Adam

# Reproducibility
torch.manual_seed(1729)
random.seed(1729)
np.random.seed(1729)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_epgcn', type=int, default=1,
                    help='Number of EPGCN layers for endpoint prediction')
parser.add_argument('--n_epcnn', type=int, default=6,
                    help='Number of EPCNN layers for endpoint prediction')
parser.add_argument('--n_trgcn', type=int, default=1,
                    help='Number of TRGCN layers for trajectory refinement')
parser.add_argument('--n_trcnn', type=int, default=3,
                    help='Number of TRCNN layers for trajectory refinement')
parser.add_argument('--n_ways', type=int, default=3,
                    help='Number of control points for endpoint prediction')
parser.add_argument('--n_smpl', type=int, default=20,
                    help='Number of samples for refine')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='zara1',
                    help='Dataset name(eth,hotel,univ,zara1,zara2)')

# Training specifc parameters
parser.add_argument('--batch_size', type=int,
                    default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int,
                    default=512, help='Number of epochs')
parser.add_argument('--clip_grad', type=float,
                    default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=128,
                    help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true",
                    default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--nans', default=0.1, type=float, help='Number of nans prior to training')
parser.add_argument('--saits_lr', default=1e-4, type=float, help='Learning Rate of pre-trained SAITS model')
args = parser.parse_args()

plt.figure(figsize=(20,20))
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layecliprs in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    # for n, p in named_parameters:
    #     print("parameters:", n,p)
    i = 0
    for n, p in named_parameters:
        i += 1
        if(p.requires_grad) and ("bias" not in n):
            # print(f'{type(p.grad)=}')
            # print(n)
            # if p.grad != None:
            if p.grad == None:
                #print(i, n, p)
                continue
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            # max_grads.append(p.grad.abs().max())
    plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.plot(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f"img/gradient-hotel")


# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoint/' + args.tag + '/'

train_dataset = TrajectoryDataset(
    dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TrajectoryDataset(
    dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = end_point(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.to(device)
saits = create_saits_model(epochs=128)
all_parameters = list(model.parameters()) + list(saits.model.parameters())

optimizer = torch.optim.SGD(all_parameters, lr=args.lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=(1e-5)/2)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + f'args-{args.nans}.pkl', 'wb') as f:
    pickle.dump(args, f)

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def transform_imputed(X):

    X_abs = X[:, :, :2]
    X_abs = X_abs.reshape(1, *X_abs.shape)

    X_rel = X[:, :, 2:]
    X_rel = X_rel.reshape(1, *X_rel.shape)

    X_abs = X_abs.permute(0, 1, 3, 2)
    X_rel = X_rel.permute(0, 1, 3, 2)

    S_obs = torch.stack((X_abs, X_rel), dim=1).permute(0, 1, 4, 2, 3)

    return S_obs


def saits_loader(original_tensor, nans = 0.1):
    nelems = original_tensor.numel()
    ne_nan = int(nans * nelems)
    nan_indices = random.sample(range(nelems), ne_nan)
    new_tensor = original_tensor.clone().reshape(-1)
    new_tensor[nan_indices] = float('nan')
    return new_tensor.reshape(*original_tensor.shape)


def train(epoch, nan=0):
    global metrics, model, saits
    model.train()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description(
        'Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]
        # print(f"{S_obs[:, 1]=}")
        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        S_actual = S_obs.clone()
        X_obs_saits = S_obs[:, 0].clone().to(device).permute(0, 2, 3, 1)
        X_obs_rel_saits = S_obs[:, 1].clone().to(device).permute(0, 2, 3, 1)


        _, npeds, _, step_size = X_obs_saits.shape
        X_obs_saits = X_obs_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        _, npeds, _, step_size = X_obs_rel_saits.shape
        X_obs_rel_saits = X_obs_rel_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        for i in range(npeds):
            X_i = X_obs_saits[i]
            X_obs_saits[i] = saits_loader(X_i, nan)

        for i in range(npeds):
            X_i = X_obs_rel_saits[i]
            X_obs_rel_saits[i] = saits_loader(X_i, nan)

        X_saits = torch.cat((X_obs_saits, X_obs_rel_saits), dim=2)
        X_obs_saits = saits_impute(X_saits)
        S_obs_imputed = transform_imputed(X_obs_saits)
        absolute_diff = torch.abs(S_obs_imputed - S_obs)
        mae_diff = torch.max(absolute_diff)
        S_obs = S_obs_imputed.clone()

        # Run Graph-TERN model
        # try:
        V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt, vgg_list = vgg_list)
        # except:
        #     print(f"{S_actual=}")
        #     print(f"{S_obs_imputed=}")
        #     exit(1)
        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask)
        loss = r_loss + m_loss

        if torch.isnan(loss):
            pass
        else:
            loss.backward()
            # plot_grad_flow(model.named_parameters())
            loss_batch += loss.item()

        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)
            optimizer.step()

            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f} Max_MAE: {2: .8f}'.format(
            epoch, loss.item() / args.batch_size, mae_diff))
        progressbar.update(1)

    progressbar.close()
    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics, model, saits
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description(
        'Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs, vgg_list = vgg_list)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(
            epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir +
                   args.dataset + str(args.nans) + f'_{epoch}_best.pth')
        torch.save(saits.model.state_dict(), checkpoint_dir+f'saits/{args.dataset}_{args.nans}_{epoch}_best.pth')

def get_dataset():
    tensors  = []
    # print(saits.optimizer)
    for batch_idx, batch in enumerate(train_loader):
        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]

        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        # print(S_obs[:,0].shape)

        X_obs_saits = S_obs[:, 0].clone().to(device).permute(0, 2, 3, 1)
        X_obs_rel_saits = S_obs[:, 1].clone().to(device).permute(0, 2, 3, 1)

        # print(f'{X_obs_saits.shape=}')

        _, npeds, _, step_size = X_obs_saits.shape
        X_obs_saits = X_obs_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        _, npeds, _, step_size = X_obs_rel_saits.shape
        X_obs_rel_saits = X_obs_rel_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        for i in range(npeds):
            X_i = X_obs_saits[i]
            X_obs_saits[i] = saits_loader(X_i)

        for i in range(npeds):
            X_i = X_obs_rel_saits[i]
            X_obs_rel_saits[i] = saits_loader(X_i)

        X_saits = torch.cat((X_obs_saits, X_obs_rel_saits), dim=2)
        tensors.append(X_saits)
    combined_dataset = torch.cat(tensors, dim=0)
    return combined_dataset

def pre_train_saits():
    saits_model(get_dataset())

def main():
    import os
    import pickle
    global saits
    saits_pkl = f'pre-train/saits-{args.dataset}-tune64-Adam-Scaled.pth'

    if os.path.exists(saits_pkl):
        saits.model.load_state_dict(torch.load(saits_pkl))
    else:
        pre_train_saits()
        torch.save(saits.model.state_dict(), saits_pkl)
    print("saits_lr",args.saits_lr)
    print("lr", args.lr)
    saits.optimizer = Adam(lr = args.saits_lr, weight_decay= args.saits_lr/20)
    saits.model.train()
    # init_params = {}
    
    # for n,p in saits.model.named_parameters():
    #     init_params[n] = p.clone()

    for epoch in range(args.num_epochs):
        train(epoch)
        train(epoch, args.nans)
        # curr_params={}
        # for n,p in saits.model.named_parameters():
        #     curr_params[n] = p

        # for key in init_params:
        #     print(init_params[key])
        #     print(curr_params[key])
        #     break
        # print(init_params)
        # print(curr_params)
        # are_equal = all(torch.equal(init_params[key], curr_params[key]) for key in init_params)
        # assert are_equal
        valid(epoch)
        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(
            metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(
            constant_metrics['min_val_epoch'], constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + f'metrics-{args.nans}.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + f'constant_metrics-{args.nans}.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()
