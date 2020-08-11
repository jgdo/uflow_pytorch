import pickle
from torch.utils.data import DataLoader, TensorDataset
import uflow_utils
import torch

def create_minecraft_loader(batch_size=64, shuffle=True):
    p = pickle.load(open('dataset/UFlow_data/ep1_pickle_doc.pkl', 'rb'))

    img1 = []
    img2 = []

    for i in range(1, len(p)):
        img1.append(torch.from_numpy(p[i-1][0]).permute(2, 0, 1) / 255.0)
        img2.append(torch.from_numpy(p[i][0]).permute(2, 0, 1) / 255.0)

    img1 = uflow_utils.upsample(torch.stack(img1).cuda(), is_flow=False, scale_factor=1)
    img2 = uflow_utils.upsample(torch.stack(img2).cuda(), is_flow=False, scale_factor=1)

    dataset = TensorDataset(img1, img2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader
