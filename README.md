# Feature Upsampling for Frozen-ViT Object Counting (Pilot)

A small pilot comparing bilinear upsampling with DySample (a learned dynamic
upsampler) on top of frozen DINOv2 features for object counting on FSC-147.
This is preparation for a proposed MSc project on exemplar-guided upsampling
for class-agnostic counting.

## Setup
- Data: FSC-147 (6146 images; official train/val/test split: 3659/1286/1190),
  images at 384 px height and the official adaptive density maps.
- Backbone: DINOv2-small, frozen. Images are resized to 378x378 (aspect ratio
  not preserved), giving a 27x27 token grid with 384 channels. Features are
  extracted once and cached.
- Density targets: official maps resized to 108x108 with the sum (object count)
  preserved.
- Counting head: 1x1 conv (384->128) + BatchNorm + GELU, then the upsampler
  (x4), then 3x3 conv (128->64) + GELU + 1x1 conv + ReLU. The predicted count
  is the sum of the density map.
- Upsamplers compared: bilinear (no parameters) vs. DySample (style "lp",
  4 groups, 16.5K parameters); everything else is identical.
- Training: MSE on density (scaled by 100), AdamW (lr 1e-3, weight decay 1e-4),
  cosine schedule, 30 epochs, batch size 32, no augmentation, 3 seeds.
  Exemplars are NOT used (exemplar-free counting).
- Metrics: MAE and RMSE of the predicted counts.

## Results (test set, mean +- std over 3 seeds)
| Upsampler | MAE | RMSE | Upsampler params |
|---|---|---|---|
| Bilinear | 26.93 +- 0.47 | 127.06 +- 0.48 | 0 |
| DySample | 25.93 +- 0.33 | 127.34 +- 1.11 | 16,512 |

DySample had lower test MAE than bilinear in each of the 3 seeds; RMSE was
unchanged. Validation MAE: 32.21 +- 0.19 (bilinear) vs. 31.73 +- 0.82 (DySample),
a difference within seed noise.

### Test MAE by number of objects in the image (mean over 3 seeds)
| Objects | Images | Bilinear | DySample |
|---|---|---|---|
| <= 20 | 315 | 6.69 | 6.01 |
| 21-50 | 424 | 12.22 | 10.59 |
| 51-100 | 253 | 23.43 | 22.49 |
| > 100 | 198 | 95.12 | 94.73 |

(This analysis re-trained the models; small run-to-run differences from the main
table are expected.)

## Findings
- DySample gave a modest improvement in MAE (about 4%) over bilinear upsampling.
- The gain was concentrated in images with few to moderately many objects;
  it was negligible for very dense images (> 100 objects), which account for
  most of the total error. RMSE, which is dominated by those images, did not change.
- A possible explanation is that in very dense scenes objects are smaller than a
  ViT token, so an upsampler working only on the same features cannot recover
  the missing detail. This is a hypothesis, not tested here.

## Limitations
- Only 3 seeds (they vary initialization and data order only); no significance tests.
- Small frozen backbone, one fixed hyperparameter setting, no augmentation,
  no exemplars. Numbers are not comparable to published class-agnostic counting results.
- DySample has 16.5K more parameters than bilinear, so gains may partly come from
  added capacity rather than dynamic sampling; a learned non-dynamic upsampler
  with similar capacity would be a fairer control.
- The per-object-count analysis was post hoc, uses counts (not object size), and
  has no confidence intervals.

## Next steps
- Add a capacity-matched non-dynamic control.
- Use exemplars and design an exemplar-guided upsampler.
- Evaluate across object sizes and on plant-counting data.

## Run
Install requirements.txt, then run notebooks/pilot.ipynb on a GPU
(a Colab T4 is sufficient). The data are not included: download FSC-147 from the
official repository (https://github.com/cvlab-stonybrook/LearningToCountEverything).
DySample code: https://github.com/tiny-smart/dysample (MIT license).

## References
- Ranjan, Sharma, Nguyen, Hoai, Learning To Count Everything, CVPR 2021 (FSC-147).
- Liu, Lu, Fu, Cao, Learning to Upsample by Learning to Sample, ICCV 2023 (DySample).
- Oquab, M., Darcet, T., Moutakanni, T., et al. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research (TMLR), 2024 (arXiv:2304.07193).
