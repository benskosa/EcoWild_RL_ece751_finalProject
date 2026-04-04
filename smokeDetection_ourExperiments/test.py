from feature_extraction import make_lbp_motion_image, make_lbp_motion_image_nframes
from model import SmokeDataset, build_model, get_transforms
import torch

TRAIN = '../smokeDetection_baseline_ecoWild/Dataset/train'

# Test n_frames=2 (classic paper default)
ds2 = SmokeDataset(TRAIN, n_frames=2, transform=get_transforms(False))
print(f'n_frames=2  samples: {len(ds2)}')
img, label = ds2[0]
print(f'  sample shape={tuple(img.shape)}  label={label.item()}')

# Test n_frames=3
ds3 = SmokeDataset(TRAIN, n_frames=3, transform=get_transforms(False))
print(f'n_frames=3  samples: {len(ds3)}')
img3, label3 = ds3[0]
print(f'  sample shape={tuple(img3.shape)}  label={label3.item()}')

# Test n_frames=4
ds4 = SmokeDataset(TRAIN, n_frames=4, transform=get_transforms(False))
print(f'n_frames=4  samples: {len(ds4)}')

# Model forward pass
m = build_model('v3_small', pretrained=False)
out = m(img.unsqueeze(0))
print(f'Model output shape: {tuple(out.shape)}  (logit={out.item():.4f})')
print('All OK!')