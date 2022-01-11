import os

#### DATA PATH
os.environ['DATA_PATH'] = '/home/michael/Studium/2021W/JKU/cv/project-private/data'
data_path = os.getenv('DATA_PATH')
model_path = os.path.join(os.getenv('DATA_PATH'), '../output/models/pca.pth')
bbs_path = {'validation': os.path.join(os.getenv('DATA_PATH'), '../output/bbs/validation.json'),
            'test': os.path.join(os.getenv('DATA_PATH'), '../output/bbs/test.json')}
target_path = {'validation': os.path.join(os.getenv('DATA_PATH'), 'validation/labels.json')}

##### IMAGE CONFIG
integrate = True
mask_timestemp = True
random_rotation = False

#### PATCH CONFIG
patch_size = 8
percent = 1

#### MODEL CONFIG
n_components = 10
input_shape = 3 * patch_size**2

#### OPTIMIZATION CONFIG
epochs = 5
batch_size = 256
learning_rate = 1e-3

#### POSTPROCESSING CONFIG
threshold_mode = 'q90'
min_range = 25
max_range = 55

#### EXPLORATION CONFIG
sample_idx = 1
timestep = 3

#### PREDICTION CONFIG
subset = 'validation'