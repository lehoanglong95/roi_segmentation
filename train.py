from training_wrapper.training_model_wrapper import TrainingModelWrapper
import os
from utils.utils import set_seed
import argparse
import torch
import utils.constant


def load_training_model_wrapper(seed, cuda_visible_devices, config_file, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    device = torch.device(f"cuda: {device}")
    training_model_wrapper = TrainingModelWrapper(config_file=config_file, seed=seed, device=device)
    return training_model_wrapper

# def train(seed, cuda_visible_devices, config_file, device):
#     if seed is not None:
#         set_seed(seed)
#     training_model_wrapper = load_training_model_wrapper(seed, cuda_visible_devices, config_file)
#     print(training_model_wrapper)
#     training_model_wrapper.train()

def train_and_evaluate_model(seed, cuda_visible_devices, config_file, device):
    if seed is not None:
        set_seed(seed)
    training_model_wrapper = load_training_model_wrapper(seed, cuda_visible_devices, config_file, device)
    print(training_model_wrapper)
    training_model_wrapper.train()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--seed", type=int, default=None)
    parser.add_argument('-c', "--cuda_visible_devices", type=str, default="1")
    parser.add_argument('-cf', "--config_file", type=str, required=True)
    parser.add_argument('-d', "--device", type=int, default=0)
    args = vars(parser.parse_args())
    train_and_evaluate_model(args['seed'], args['cuda_visible_devices'], args['config_file'], args['device'])

