import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data

from config import *
from dataloading.dataset import WisarDataset, identity_collator



"""
The idea here is to use temporal differences of warped timesteps, i.e. where the 10 cameras of each timestep
were averaged before. Ideally, we can find some configuration that highlights the moving humans and kind of removes
non-moving anomalies such as beacons.

To explore the position of the bounding boxes, have a look at evaluation/main.py!
"""


if __name__ == '__main__':

    #### 1. LOAD SOME VALIDATION SAMPLE FOR EXPLORATION
    # load just one (batch_size=1) validation set image for the experimentation
    validation_dataset = WisarDataset(data_path, 'validation', integrate=True, mask_timestemp=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, collate_fn=identity_collator)

    # get the sample (dict with integrated (=warped) image for each timestep
    # e.g.: key 'valid-1-4-0' is warped image of timestep 0 for valid-1-4 sample
    for data in validation_loader:
        sample, _ = data[0]
        break

    # e.g.: sample_id = 'valid-1-4-' if we loaded valid-1-4
    sample_id = validation_dataset.samples[0].split('/')[-1]
    # warped image at timestep 3
    central_img = sample[str(3) + '-integrated']


    #### 2. EXPLORE WARPED IMAGES OF TIMESTEPS
    plt.imshow(sample[str(3) + '-integrated'])
    plt.show()


    #### 3. TRY OUT DIFFERENCE OF LAST TIMESTEP (=6) AND FIRST TIMESTEP (=0)
    # plot central timestep + last timestep - first timestep
    plt.imshow(sample['3' + '-integrated'].astype(np.int16) + sample['6' + '-integrated'].astype(np.int16) - sample['0' + '-integrated'].astype(np.int16))
    plt.show()

    # plot last timestep - first timestep
    plt.imshow(sample[sample_id + '6'].astype(np.int16) - sample[sample_id + '0'].astype(np.int16) + 100)
    plt.show()


    #### 4. TRY OUT DIFFERENCES OF CENTRAL TIMESTEP WITH OTHER TIMESTEPS
    plt.imshow(np.abs(central_img.astype('float') - sample[sample_id + '0'].astype('float')).astype(np.int8))
    plt.show()
    plt.imshow(central_img - sample[sample_id + '1'])
    plt.show()
    plt.imshow(central_img - sample[sample_id + '2'])
    plt.show()
    plt.imshow(central_img - sample[sample_id + '4'])
    plt.show()
    plt.imshow(central_img - sample[sample_id + '5'])
    plt.show()
    plt.imshow(central_img - sample[sample_id + '6'])
    plt.show()


    #### 5. CHECK OUT HOW WELL IMAGES FROM THE SAME CAMERA BUT DIFFERENT TIMESTEPS OVERLAP
    # TODO