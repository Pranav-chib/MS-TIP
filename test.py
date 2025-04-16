import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from end_point.model import end_point
from utils.dataloader import TrajectoryDataset
from torch.utils.data import DataLoader
from utils.visualizer import trajectory_visualizer, controlpoint_visualizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
parser.add_argument('--nan', type=float, default=0.1, help='Percentage of nans')
parser.add_argument('--back_prop', type=str, default=False, help='Pre-Trained SAITS not back propagated')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoint/' + test_args.tag + '/'
args_suffix = 'args.pkl'
if test_args.back_prop == "true":
    args_suffix = f'/args-{test_args.nan}.pkl'

args_path = checkpoint_dir + args_suffix
with open(args_path, 'rb') as f:
    args = pickle.load(f)

dataset_path = './datasets/' + args.dataset + '/'
best_suffix = '_best.pth'
if test_args.back_prop == 'true':
    best_suffix = f'{test_args.nan}_best.pth'

model_paths = []


for filename in sorted(os.listdir(checkpoint_dir)):
    # Check if the file ends with ".pth"
    if filename.endswith(".pth") and filename.startswith(f"{args.dataset}{test_args.nan}_"):
        # Print or use the full path of the file
        file_path = os.path.join(checkpoint_dir, filename)
        model_paths.append(file_path)

# Data preparation
test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)



# Model preparation
def best(model_path):
    model = end_point(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                    seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))


    def test(KSTEPS=20):
        model.eval()
        model.n_smpl = KSTEPS
        ade_refi_all = []
        fde_refi_all = []

        progressbar = tqdm(range(len(test_loader)))
        progressbar.set_description('Testing {}'.format(test_args.tag))

        for batch_idx, batch in enumerate(test_loader):
            S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]

            # Run Graph-TERN model
            V_init, V_pred, V_refi, valid_mask = model(S_obs, pruning=4, clustering=True, vgg_list = vgg_list)

            # Calculate ADEs and FDEs for each refined trajectory
            V_trgt_abs = S_trgt[:, 0].squeeze(dim=0)
            temp = (V_refi - V_trgt_abs).norm(p=2, dim=-1)
            ADEs = temp.mean(dim=1).min(dim=0)[0]
            FDEs = temp[:, -1, :].min(dim=0)[0]
            ade_refi_all.extend(ADEs.tolist())
            fde_refi_all.extend(FDEs.tolist())
            #controlpoint_visualizer(V_init.detach(),batch_idx = batch_idx)
            #trajectory_visualizer(V_refi.detach(), S_obs, S_trgt, batch_idx)
            progressbar.update(1)

        progressbar.close()

        ade_refi = sum(ade_refi_all) / len(ade_refi_all)
        fde_refi = sum(fde_refi_all) / len(fde_refi_all)
        return ade_refi, fde_refi

    return test(KSTEPS=test_args.n_samples)


def main():
    ade_refi, fde_refi = [], []

    # Repeat the evaluation to reduce randomness
    for i, model_path in enumerate(model_paths):
        ade_refi, fde_refi = [], []
        # Repeat the evaluation to reduce randomness
        print(f"Epoch: {model_path}")
        #if not model_path.endswith('eth0.15_47_best.pth'):
        #   continue
        repeat = 10
        for i in range(repeat):
            temp = best(model_path)
            ade_refi.append(temp[0])
            fde_refi.append(temp[1])


        ade_refi = np.mean(ade_refi)
        fde_refi = np.mean(fde_refi)

        result_lines = ["Evaluating model: {}".format(test_args.tag),
                    "Refined_ADE: {0:.8f}, Refined_FDE: {1:.8f}".format(ade_refi, fde_refi)]

        for line in result_lines:
            print(line)


if __name__ == "__main__":
    main()