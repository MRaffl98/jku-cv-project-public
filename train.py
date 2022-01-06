import torch
import torch.nn as nn

from config import *
from tqdm import tqdm

from dataloading.dataset import WisarDataset, identity_collator
from preprocessing.patch.patch_generator import fill_patch_dict
from preprocessing.patch.black_patch_removal import remove
from preprocessing.patch.patch_dataset import PatchDataset
from models.modules.pca_network import PCA
from models.trainer.flat_trainer import FlatTrainer


#### DATA LOADING
train_dataset = WisarDataset(data_path, subset='train', integrate=integrate, mask_timestemp=mask_timestemp, random_rotation=random_rotation)
train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=identity_collator)
print('data loading finished ...')

#### PATCH PREPARATION
patch_dict = {}
for (sample_path, data) in tqdm(zip(train_dataset.samples, train_loader)):
    sample_id = sample_path.split('/')[-1]
    for (camera, img) in data[0][0].items():
        patch_dict = fill_patch_dict(patch_dict, img, sample_id, camera, patch_size)
patch_dict = remove(patch_dict, percent=percent)
patch_dataset = PatchDataset(patch_dict)
print('patch preparation finished ...')

#### TRAINING PREPARATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PCA(n_components, input_shape=input_shape).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
print('training preparation finished ...')

#### TRAINING LOOP
trainer = FlatTrainer(model, input_shape, optimizer, criterion, device)
trainer.fit(patch_dataset, epochs, batch_size, shuffle=True)
print('training loop finished ...')

#### SAVE MODEL
torch.save(model.state_dict(), model_path)
print('save model finished ...')