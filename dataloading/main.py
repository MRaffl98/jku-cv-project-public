import os
import matplotlib.pyplot as plt
import cv2 as cv

from dataloading.dataset import WisarDataset, get_loader
from dataloading.datahelper import get_images_at_timestep
from dataloading.datahelper import load_labels, get_sample_paths, get_center_image

# DATA_PATH: Absolute path to your "data_WiSAR/data" directory

if __name__ == "__main__":
    #### 1. Example how to create and use the dataset
    wisar_dataset = WisarDataset(os.getenv("DATA_PATH"), False, False, True, False)
    fig = plt.figure()

    for i in range(len(wisar_dataset)):
        # dicts with 70 elements each
        images, homographies = wisar_dataset[i]

        # plot image 0-B04 for multiple samples
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(cv.cvtColor(images['0-B04'], cv.COLOR_BGR2RGB))

        if i == 3:
            plt.show()
            break


    #### 2. Rotating samples randomly
    wisar_dataset = WisarDataset(os.getenv("DATA_PATH"), False, False, True, True)
    fig = plt.figure()

    for i in range(len(wisar_dataset)):
        # dicts with 70 elements each
        images, homographies = wisar_dataset[i]

        # plot image 0-B04 for multiple samples
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(cv.cvtColor(images['0-B04'], cv.COLOR_BGR2RGB))

        if i == 3:
            plt.show()
            break


    #### 3. A helper function if someone needs it
    # dicts with 10 elements each
    images_at_timestep, homographies_at_timestep = get_images_at_timestep(wisar_dataset[0][0], wisar_dataset[0][0], timestep=0)


    #### 4. A training loader. Might be mostly used for iterative training procedures like SGD.
    def identity_collator(batch):
        return batch

    for data in get_loader(os.getenv("DATA_PATH"), batch_size=4, collate_fn=identity_collator):
        print(data)
        break


    #### 5. Draw bounding boxes on evaluation images
    paths = get_sample_paths(os.getenv("DATA_PATH"), "validation")
    labels = load_labels(paths[4])
    center_image = get_center_image(paths[4])

    for bb in labels:
        x, y, w, h = bb
        cv.rectangle(center_image, (x, y), (x + w, y + h), (0, 0, 255), 5)

    plt.imshow(cv.cvtColor(center_image, cv.COLOR_BGR2RGB))
    plt.show()