## Methodology

### Dataset

The experiments used the Labeled Faces in the Wild (LFW) dataset accessed through `scikit-learn`. The dataset was loaded with `min_faces_per_person=2`, `resize=0.5`, and `color=True`. The resulting dataset contained 9,164 face images belonging to 1,680 identities. The images were resized to half their original dimensions before being processed by the face embedding model.

### Face Representation

Face images were represented using a pretrained FaceNet-based model, `InceptionResnetV1`, with weights pretrained on the VGGFace2 dataset. Each image was converted into a 512-dimensional embedding. The embeddings were compared using cosine similarity, with higher similarity scores indicating a stronger match.

### Verification Protocol

The identities were divided into separate validation and test groups. The validation group was used to select a similarity threshold of 0.4085. The held-out test group contained 840 genuine pairs and 840 impostor pairs.

Genuine pairs consisted of images belonging to the same identity, while impostor pairs consisted of images belonging to different identities. The fixed validation-selected threshold was applied to the test set to calculate the false acceptance rate (FAR) and false rejection rate (FRR). AUC and EER were also calculated to describe verification performance across decision thresholds.

### Image Degradation Experiments

Two controlled transformations were evaluated:

1. **Gaussian blur:** Blur radii of 0, 1, 2, 3, 5, and 7 were tested.
2. **Brightness reduction:** Brightness factors of 1.0, 0.7, 0.5, 0.3, and 0.15 were tested. A factor of 1.0 represents the original brightness.

For each pair, the first image retained its original embedding, while the second image was transformed and re-embedded. This design evaluated verification when one image in a pair was degraded and the other remained unchanged. The same test pairs and fixed threshold were used across the conditions.

### Evaluation Metrics

- **AUC:** Measures how well genuine and impostor pairs are ranked across thresholds.
- **Equal Error Rate (EER):** Summarizes performance near the point where FAR and FRR are approximately equal.
- **False Acceptance Rate (FAR):** The proportion of impostor pairs incorrectly accepted as genuine.
- **False Rejection Rate (FRR):** The proportion of genuine pairs incorrectly rejected.

A preliminary pair-level bootstrap was used to estimate a 95% confidence interval for baseline AUC. Since pairs can share images, this interval may not fully account for dependence in the evaluation data.

All experiments used a fixed pretrained model and controlled image transformations. The study did not train or fine-tune the face recognition model.

## Results and Discussion

### Baseline Performance

The pretrained FaceNet-based face verification system was evaluated on an identity-disjoint held-out subset of the Labeled Faces in the Wild (LFW) dataset. The test set contained 840 genuine pairs and 840 impostor pairs. The similarity threshold was selected using the validation set and fixed at 0.4085 for the threshold-based evaluation on the held-out test set.

Under the original image condition, the model achieved an area under the receiver operating characteristic curve (AUC) of 0.9971 and an equal error rate (EER) of 0.0179. At the fixed validation-selected threshold, the false acceptance rate (FAR) was 1.43%, while the false rejection rate (FRR) was 1.67%.

A preliminary pair-level bootstrap produced a 95% confidence interval of [0.9942, 0.9991] for baseline AUC. This interval should be interpreted cautiously because image reuse across pairs can introduce dependence that pair-level resampling does not fully account for.

### Effect of Image Degradation

Two image transformations were investigated: Gaussian blur and brightness reduction. For each test pair, the first image retained its original embedding, while the second image was transformed and re-embedded. The same validation-selected threshold of 0.4085 was used to calculate FAR and FRR across conditions.

#### Gaussian Blur

The model was highly sensitive to increasing blur severity. AUC declined from 0.9971 under the original condition to 0.9432 at blur radius 2 and 0.7314 at radius 3. At radius 7, AUC reached 0.5254, close to chance-level discrimination.

The false rejection rate increased from 1.67% in the baseline condition to 42.38% at blur radius 2 and 92.38% at radius 3. At radius 7, FRR reached 99.64%. These results indicate that, under the tested setup, severe blur substantially reduced the model's ability to distinguish genuine pairs from impostor pairs at the chosen operating threshold.

#### Brightness Reduction

Moderate brightness reduction had a comparatively limited effect. AUC remained approximately 0.997 for brightness factors between 1.0 and 0.3. At the strongest tested reduction, factor 0.15, AUC decreased to 0.9509 and FRR increased to 22.38%.

This suggests that the model tolerated the tested moderate brightness changes better than the tested blur transformations. However, the strongest brightness reduction still impaired verification performance, particularly at the fixed threshold.

### Comparative Interpretation

Across the tested conditions, blur caused a substantially larger decline in AUC and a greater increase in false rejections than moderate brightness reduction. The contrast between the two transformations suggests that preserving facial detail was particularly important for this model and dataset.

These findings apply only to the selected pretrained model, LFW sample, image preprocessing, transformations, and verification protocol. They should not be interpreted as a universal ranking of all image-quality factors or face-recognition systems.


### Figures

**Figure 1.** AUC under image degradation.

![Held-out AUC under image degradation](results/figures/heldout_auc.png)

**Figure 2.** False rejection rate at the fixed validation threshold.

![False rejection rate under image degradation](results/figures/heldout_frr.png)


## Limitations and Future Work

This study has several limitations.

First, the experiments use a single pretrained face verification model and the Labeled Faces in the Wild dataset. The results may not generalize to other models, datasets, image-capture environments, or deployment settings.

Second, the evaluation uses controlled image transformations rather than naturally occurring changes in lighting, motion blur, camera quality, pose, or occlusion. Only the second image in each pair was altered, so the experiments do not cover every possible verification scenario.

Third, the genuine and impostor pairs were randomly sampled. Although validation and test identities were kept separate, images can be reused across pairs within a split. The resulting pair-level measurements are therefore not fully independent. The baseline confidence interval is preliminary and may underestimate uncertainty.

Fourth, the similarity threshold was selected on the validation set and held fixed for FAR and FRR calculations on the test set. EER was calculated by sweeping thresholds on the test scores and is therefore a descriptive metric rather than a threshold selected independently of the test data.

Finally, this version of the study does not include a demographic fairness evaluation. Differences in performance across demographic groups have not been measured, and no conclusions about demographic fairness can be drawn from these results.

Future work should include identity-aware uncertainty estimation, additional pretrained models, more realistic image degradations, and a carefully designed fairness analysis using an appropriately documented dataset and ethical evaluation protocol.
