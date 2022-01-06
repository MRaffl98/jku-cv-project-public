import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from config import *

from dataloading.dataset import WisarDataset
from preprocessing.patch.patch_generator import fill_patch_dict
from preprocessing.patch.black_patch_removal import remove
from preprocessing.patch.patch_dataset import PatchDataset
from models.modules.pca_network import PCA
from models.trainer.flat_trainer import FlatPredictor
from postprocessing.patch2image import patch2image
from postprocessing.thresholding import threshold
from postprocessing.dbscan import create_dbscan_dataset, cluster
from evaluation.utils import BoundingBox


#### DATA LOADING
valid_dataset = WisarDataset(data_path, subset='validation', integrate=integrate, mask_timestemp=mask_timestemp)

#### PATCH PREPARATION
sample_id = valid_dataset.samples[sample_idx].split('/')[-1]
data = valid_dataset[sample_idx]
patch_dict = {}
for (camera, img) in data[0].items():
    patch_dict = fill_patch_dict(patch_dict, img, sample_id, camera, patch_size)
patch_dict = remove(patch_dict, percent=percent)
patch_dataset = PatchDataset(patch_dict)

#### PREDICTION PREPARATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reconstruction_loss = nn.MSELoss(reduction='none')
model = PCA(n_components, input_shape=input_shape).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

#### PREDICTION
predictor = FlatPredictor(model, input_shape, reconstruction_loss, device)
losses = predictor.predict(patch_dataset, batch_size=batch_size, shuffle=False)

#### POSTPROCESSING
# handle patch losses
losses = torch.cat(losses, dim=0)
patch_losses = torch.sum(losses, dim=1).numpy()

# handle patch identifiers
patch_names = patch_dataset.patch_ids
patch_numbers = np.array([x.split('-') [-1] for x in patch_names]).astype('int')
timesteps = np.array([x.split('-')[-3] for x in patch_names]).astype('int')

# draw saliency map
saliency_map = patch2image(patch_numbers[timesteps==timestep], patch_losses[timesteps==timestep], patch_size)
#plt.imshow(saliency_map)
#plt.show()

# draw anomaly map
anomalies = threshold(patch_losses, mode=threshold_mode)
anomaly_map = patch2image(patch_numbers[timesteps==timestep], anomalies[timesteps==timestep], patch_size)
#plt.imshow(anomaly_map)
#plt.show()

# dbscan
X = create_dbscan_dataset(anomaly_map)
labels, clusters, stats = cluster(X)

#for (i, c) in enumerate(clusters):
#    plt.scatter(x=c[:, 0], y=c[:, 1], label='cluster' + str(i))
#plt.legend()
#plt.show()

# find bounding boxes
candidates = stats[stats['range_x'].between(min_range, max_range) & stats['range_y'].between(min_range, max_range)]
bbs = [BoundingBox((c['min_x'], c['min_y'], c['range_x'], c['range_y'])) for (i, c) in candidates.iterrows()]