from .dataloader import TrajectoryDataset
from .augmentor import data_sampler
from .visualizer import trajectory_visualizer, controlpoint_visualizer
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")