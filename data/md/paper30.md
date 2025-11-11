**Noname manuscript No.**
(will be inserted by the editor)

## **Image Matching Across Wide Baselines: From Paper to Practice**


**Yuhe Jin** _**·**_ **Dmytro Mishkin** _**·**_ **Anastasiia Mishchuk** _**·**_ **Jiri Matas** _**·**_ **Pascal Fua** _**·**_
**Kwang Moo Yi** _**·**_ **Eduard Trulls**


Received: date / Accepted: date


**Abstract** We introduce a comprehensive benchmark for local features and robust estimation algorithms, focusing on
the downstream task – the accuracy of the reconstructed
camera pose – as our primary metric. Our pipeline’s modular
structure allows easy integration, configuration, and combi
nation of different methods and heuristics. This is demon
strated by embedding dozens of popular algorithms and
evaluating them, from seminal works to the cutting edge of
machine learning research. We show that with proper settings, classical solutions may still outperform the _perceived_
_state of the art_ .
Besides establishing the _actual state of the art_, the conducted experiments reveal unexpected properties of Structure from Motion (SfM) pipelines that can help improve



This work was partially supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant
“Deep Visual Geometry Machines” (RGPIN-2018-03788), by systems supplied by Compute Canada, and by Google’s Visual Positioning Service. DM and JM were supported by OP VVV
funded project CZ.02.1.01/0.0/0.0/16 019/0000765 “Research Center for Informatics”. DM was also supported by CTU student grant
SGS17/185/OHK3/3T/13 and by the Austrian Ministry for Transport,
Innovation and Technology, the Federal Ministry for Digital and Economic Affairs, and the Province of Upper Austria in the frame of the
COMET center SCCH. AM was supported by the Swiss National Science Foundation.


Y. Jin, K.M. Yi
Visual Computing Group, University of Victoria
E-mail: _{_ yuhejin, kyi _}_ @uvic.ca


A. Mishchuk, P. Fua
Computer Vision Lab, Ecole Polytechnique F´ed´erale de Lausanne [´]
E-mail: _{_ anastasiia.mishchuk, pascal.fua _}_ @epfl.ch


D. Mishkin, J. Matas
Center for Machine Perception, Czech Technical University in Prague
E-mail: _{_ mishkdmy, matas _}_ @cmp.felk.cvut.cz


E. Trulls

Google Research
E-mail: trulls@google.com



**Fig. 1** Every paper claims to outperform the state of the art. Is this
possible, or an artifact of insufficient validation? On the left, we show
stereo matches obtained with **D2-Net** (2019) [37], a state-of-the-art local feature, using OpenCV RANSAC with its default settings. We color
the inliers in green if they are correct and in red otherwise. On the right,
we show **SIFT** (1999) [54] with a carefully tuned MAGSAC [15] – notice how the latter performs much better. This illustrates our take-home
message: to correctly evaluate a method’s performance, it needs to be
embedded within the pipeline used to solve a given problem, and the
different components in said pipeline need to be tuned carefully and
jointly, which requires engineering and domain expertise. We fill this
need with a new, modular benchmark for sparse image matching, incorporating dozens of built-in methods.


their performance, for both algorithmic and learned methods. Data and code are online [1], providing an easy-to-use and
flexible framework for the benchmarking of local features
and robust estimation methods, both _alongside_ and _against_
top-performing methods. This work provides a basis for the
Image Matching Challenge [2] .


1 [https://github.com/vcg-uvic/image-matching-benchmark](https://github.com/vcg-uvic/image-matching-benchmark)
2 [https://vision.uvic.ca/image-matching-challenge](https://vision.uvic.ca/image-matching-challenge)


2 Yuhe Jin et al.



**1 Introduction**


Matching two or more views of a scene is at the core of
fundamental computer vision problems, including image retrieval [54, 7, 78, 104, 70], 3D reconstruction [3, 47, 90, 122],
re-localization [85, 86, 57], and SLAM [68, 33, 34]. Despite
decades of research, image matching remains unsolved in
the general, wide-baseline scenario. Image matching is a
challenging problem with many factors that need to be taken
into account, e.g., viewpoint, illumination, occlusions, and
camera properties. It has therefore been traditionally approached with sparse methods – that is, with local features.
Recent effort have moved towards _holistic_, end-to-end
solutions [50, 12, 26]. Despite their promise, they are yet to
outperform the _separatists_ [88, 121] that are based on the
classical paradigm of step-by-step solutions. For example,
in a classical wide baseline stereo pipeline [74], one (1) extracts local features, such as SIFT [54], (2) creates a list of
tentative matches by nearest-neighbor search in descriptor
space, and (3) retrieves the pose with a minimal solver inside a robust estimator, such as the 7-point algorithm [45]
in a RANSAC loop [40]. To build a 3D reconstruction out
of a set of images, same matches are fed to a bundle adjustment pipeline [44, 106] to jointly optimize the camera
intrinsics, extrinsics, and 3D point locations. This modular
structure simplifies engineering a solution to the problem
and allows for incremental improvements, of which there
have been hundreds, if not thousands.
New methods for each of the sub-problems, such as feature extraction and pose estimation, are typically studied in
isolation, using intermediate metrics, which simplifies their
evaluation. However, there is no guarantee that gains in one
part of the pipeline will translate to the final application, as
these components interact in complex ways. For example,
patch descriptors, including very recent work [46, 110,103,
67], are often evaluated on Brown’s seminal patch retrieval
database [24], introduced in 2007. They show dramatic improvements – up to 39x relative [110] – over handcrafted
methods such as SIFT, but it is unclear whether this remains
true on real-world applications. In fact, we later demonstrate
that the gap narrows dramatically when decades-old baselines are properly tuned.
We posit that it is critical to look beyond intermediate metrics and focus on downstream performance. This
is particularly important _now_ as, while deep networks are
reported to outperform algorithmic solutions on classical,
sparse problems such as outlier filtering [114, 79, 117,97,
22], bundle adjustment [99, 93], SfM [109, 111] and SLAM

[100, 51], our findings in this paper suggest that this may not
always be the case. To this end, we introduce a benchmark
for wide-baseline image matching, including:


(a) A dataset with thousands of phototourism images of 25
landmarks, taken from diverse viewpoints, with different



cameras, in varying illumination and weather conditions
– all of which are necessary for a comprehensive evaluation. We reconstruct the scenes with SfM, without the
need for human intervention, providing depth maps and
ground truth poses for 26k images, and reserve another
4k for a private test set.
(b) A modular pipeline incorporating dozens of methods for
feature extraction, matching, and pose estimation, both
classical and state-of-the-art, as well as multiple heuristics, all of which can be swapped out and tuned separately.
(c) Two downstream tasks – stereo and multi-view recon
struction – evaluated with both downstream and inter
mediate metrics, for comparison.
(d) A thorough study of dozens of methods and techniques, both hand-crafted and learned, and their combination, along with a recommended procedure for effective hyper-parameter selection.


The framework enables researchers to evaluate how a

new approach performs in a standardized pipeline, both
_against_ its competitors, and _alongside_ state-of-the-art solutions for other components, from which it simply cannot be
detached. This is crucial, as true performance can be easily
hidden by sub-optimal hyperparameters.


**2 Related Work**


The literature on image matching is too vast for a thorough

overview. We cover relevant methods for feature extraction

and matching, pose estimation, 3D reconstruction, applicable datasets, and evaluation frameworks.


2.1 Local features


Local features became a staple in computer vision with the
introduction of SIFT [54]. They typically involve three distinct steps: keypoint detection, orientation estimation, and
descriptor extraction. Other popular classical solutions are
SURF [18], ORB [83], and AKAZE [5].
Modern descriptors often train deep networks on
pre-cropped patches, typically from SIFT keypoints
( _i.e._ Difference of Gaussians or DoG). They include Deepdesc [94], TFeat [13], L2-Net [102], HardNet [62], SOSNet [103], and LogPolarDesc [39] – most of them are
trained on the same dataset [24]. Recent works leverage additional cues, such as geometry or global context, including
GeoDesc [56] and ContextDesc [55]. There have been multiple attempts to learn keypoint detectors separately from
the descriptor, including TILDE [108], TCDet [119], QuadNet [89], and Key.Net [16]. An alternative is to treat this as
an end-to-end learning problem, a trend that started with the


Image Matching Across Wide Baselines: From Paper to Practice 3


(a) (b) (c) (d) (e) (f) (g) (h) (i) (j)


**Fig. 2 On the limitations of previous datasets.** To highlight the need for a new benchmark, we show examples from datasets and benchmarks
featuring posed images that have been previously used to evaluate local features and robust matchers (some images are cropped for the sake of
presentation). From left to right: (a) VGG Oxford [60], (b) HPatches [11], (c) Edge Foci [123], (d) Webcam [108], (e) AMOS [48], (f) Kitti [42],
(g) Strecha [95], (h) SILDa [10], (i) Aachen [85], (j) Ours. Notice that many (a-e) contain only planar structures or illumination changes, which
makes it easy – or trivial – to obtain ground truth poses, encoded as homographies. Other datasets are small – (g) contains very accurate depth
maps, but only two scenes and 19 images total – or do not contain a wide enough range of photometric and viewpoint transformations. Aachen (i)
is closer to ours (j), but relatively small, limited to one scene, and focused on re-localization on the day vs. night use-case.



introduction of LIFT [113] and also includes DELF [70],
SuperPoint [34], LF-Net [71], D2-Net [37] and R2D2 [81].


2.2 Robust matching


Inlier ratios in wide-baseline stereo can be below 10% –

and sometimes even lower. This is typically approached
with iterative sampling schemes based on RANSAC [40],
relying on closed-form solutions for pose solving such as
the 5- [69], 7- [45] or 8-point algorithm [43]. Improvements to this classical framework include local optimization [28], using likelihood instead of reprojection (MLESAC) [105], speed-ups using probabilistic sampling of hypotheses (PROSAC) [27], degeneracy check using homographies (DEGENSAC) [30], Graph Cut as a local optimization
(GC-RANSAC) [14], and auto-tuning of thresholds using
confidence margins (MAGSAC) [15].
As an alternative direction, recent works, starting with
CNe (Context Networks) in [114], train deep networks for
outlier rejection taking correspondences as input, often followed by a RANSAC loop. Follow-ups include [79, 120,97,
117]. Differently from RANSAC solutions, they typically
process all correspondences in a single forward pass, without the need to iteratively sample hypotheses. Despite their
promise, it remains unclear how well they perform in realworld settings, compared to a well-tuned RANSAC.


2.3 Structure from Motion (SfM)


In SfM, one jointly optimizes the location of the 3D points
and the camera intrinsics and extrinsics. Many improvements have been proposed over the years [3, 47, 31, 41,122].
The most popular frameworks are VisualSFM [112] and
COLMAP [90] – we rely on the latter, to both generate the
ground truth and as the backbone of our multi-view task.



2.4 Datasets and benchmarks


Early works on local features and robust matchers typically
relied on the Oxford dataset [60], which contains 48 images and ground truth homographies. It helped establish two
common metrics for evaluating local feature performance:
repeatability and matching score. Repeatability evaluates the
keypoint detector: given two sets of keypoints over two images, projected into each other, it is defined as the ratio of
keypoints whose support regions’ overlap is above a threshold. The matching score (MS) is similarly defined, but also
requires their descriptors to be nearest neighbours. Both require pixel-to-pixel correspondences – _i.e._, features outside
valid areas are ignored.
A modern alternative to Oxford is HPatches [11], which
contains 696 images with differences in illumination _or_
viewpoint – but not both. However, the scenes are planar,
without occlusions, limiting its applicability.

Other datasets that have been used to evaluate local fea
tures include DTU [1], Edge Foci [123], Webcam [108],
AMOS [75], and Strecha’s [95]. They all have limitations
– be it narrow baselines, noisy ground truth, or a small number of images. In fact, most learned descriptors have been
trained and evaluated on [25], which provides a database of
pre-cropped patches with correspondence labels, and measures performance in terms of patch retrieval. While this
seminal dataset and evaluation methodology helped developed many new methods, it is not clear how results translate
to different scenarios – particularly since new methods outperform classical ones such as SIFT by orders of magnitude,
which suggests overfitting.
Datasets used for navigation, re-localization, or SLAM,
in outdoor environments are also relevant to our problem.
These include KITTI [42], Aachen [87], Robotcar [58],
and CMU seasons [86,9]. However, they do not feature the
wide range of transformations present in phototourism data.


4 Yuhe Jin et al.


**Fig. 3** **The Image Matching Challenge PhotoTourism (IMC-PT) dataset.** We show a few selected images from our dataset and their corresponding depth maps, with occluded pixels marked in red.



Megadepth [53] is a more representative, phototourismbased dataset, which using COLMAP, as in our case – it
could, in fact, easily be folded into ours.
Benchmarks, by contrast, are few and far between – they
include VLBenchmark [49], HPatches [11], and SILDa [10]
– all limited in scope. A large-scale benchmark for SfM
was proposed in [91], which built 3D reconstructions with
different local features. However, only a few scenes contain ground truth, so most of their metrics are qualitative
– _e.g._ number of observations or average reprojection error.
Yi _et al._ [114] and Bian _et al._ [21] evaluate different methods for pose estimation on several datasets – however, few
methods are considered and they are not carefully tuned.
We highlight some of these datasets/benchmarks, and
their limitations, in Fig. 2. We are, to the best of our knowledge, the first to introduce a public, modular benchmark for
3D reconstruction with sparse methods using downstream
metrics, and featuring a comprehensive dataset with a large
range of image transformations.


**3 The Image Matching Challenge PhotoTourism Dataset**


While it is possible to obtain very accurate poses and depth
maps under controlled scenarios with devices like LIDAR,
this is costly and requires a specific set-up that does not scale
well. For example, Strecha’s dataset [95] follows that approach but contains only 19 images. We argue that a truly
representative dataset must contain a wider range of transformations – including different imaging devices, time of
day, weather, partial occlusions, etc. Phototourism images
satisfy this condition and are readily available.
We thus build on 25 collections of popular landmarks
originally selected in [47, 101], each with hundreds to thousands of images. Images are downsampled with bilinear interpolation to a maximum size of 1024 pixels along the longside and their poses were obtained with COLMAP [90],
which provides the (pseudo) ground truth. We do exhaustive
image matching before Bundle Adjustment – unlike [92],
which uses only 100 pairs for each image – and thus provide enough matching images for any conventional SfM to
return near-perfect results in standard conditions.



Our approach is to obtain a ground truth signal using reliable, off-the-shelf technologies, while making the problem
as easy as possible – and then evaluate new technologies on
a much harder problem, using only a subset of that data. For
example, we reconstruct a scene with hundreds or thousands
of images with vanilla COLMAP and then evaluate “modern” features and matchers against its poses using only two
images (“stereo”) or up to 25 at a time (“multiview” with
SfM). For a discussion regarding the accuracy of our ground
truth data, please refer to Section 3.3.
In addition to point clouds, COLMAP provides dense
depth estimates. These are noisy, and have no notion of occlusions – a depth value is provided for every pixel. We
remove occluded pixels from depth maps using the reconstructed model from COLMAP; see Fig. 3 for examples. We
rely on these “cleaned” depth maps to compute classical,
pixel-wise metrics – repeatability and matching score. We
find that some images are flipped 90 _[◦]_, and use the reconstructed pose to rotate them – along with their poses – so
they are roughly ‘upright’, which is a reasonable assumption for this type of data.


3.1 Dataset details


Out of the 25 scenes, containing almost 30k registered images in total, we select 2 for validation and 9 for testing. The
remaining scenes can be used for training, if so desired, and
are not used in this paper. We provide images, 3D reconstructions, camera poses, and depth maps, for every training and validation scene. For the test scenes we release only
a subset of 100 images and keep the ground truth private,
which is an integral part of the Image Matching Challenge.
Results on the private test set can be obtained by sending
submissions to the organizers, who process them.
The scenes used for training, validation and test are
listed in Table 1, along with the acronyms used in several
figures. For the validation experiments – Sections 5 and 7 –
we choose two of the larger scenes, “Sacre Coeur” and “St.
Peter’s Square”, which in our experience provide results that
are quite representative of what should be expected on the
Image Matching Challenge Dataset. These two subsets have


Image Matching Across Wide Baselines: From Paper to Practice 5



Name Group Images 3D points


“Brandenburg Gate” T 1363 100040
“Buckingham Palace” T 1676 234052
“Colosseum Exterior” T 2063 259807

“Grand Place Brussels” T 1083 229788

“Hagia Sophia Interior” T 888 235541
“Notre Dame Front Facade” T 3765 488895

“Palace of Westminster” T 983 115868

“Pantheon Exterior” T 1401 166923

“Prague Old Town Square” T 2316 558600
“Reichstag” T 75 17823
“Taj Mahal” T 1312 94121
“Temple Nara Japan” T 904 92131
“Trevi Fountain” T 3191 580673

“Westminster Abbey” T 1061 198222
“Sacre Coeur” (SC) V 1179 140659
“St. Peter’s Square” (SPS) V 2504 232329


Total T + V 25.7k 3.6M


“British Museum” (BM) E 660 73569
“Florence Cathedral Side” (FCS) E 108 44143
“Lincoln Memorial Statue” (LMS) E 850 58661
“London (Tower) Bridge” (LB) E 629 72235
“Milan Cathedral” (MC) E 124 33905
“Mount Rushmore” (MR) E 138 45350
“Piazza San Marco” (PSM) E 249 95895
“Sagrada Familia” (SF) E 401 120723
“St. Paul’s Cathedral” (SPC) E 615 98872


Total E 3774 643k


**Table 1 Scenes in the IMC-PT dataset.** We provide some statistics
for the training (T), validation (V), and test (E) scenes.


been released, so that the validation results are reproducible
and comparable. They can all be downloaded from the challenge website [2] .


3.2 Estimating the co-visibility between two images


When evaluating image pairs, one has to be sure that two
given images share a common part of the scene – they may
be registered by the SfM reconstruction without having any
pixels in common as long as other images act as a ‘bridge’
between them. Co-visible pairs of images are determined
with a simple heuristic. 3D model points, detected in both
the considered images, are localised in 2D in each image.
The bounding box around the keypoints is estimated. The
ratio of the bounding box area and the whole image is the
“visibility” of image _i_ in _j_ and _j_ in _i_ respectively; see Fig. 5
for examples. The minimum of the two “visibilities” is the
“co-visibility” ratio _v_ _i,j_ _∈_ [0 _,_ 1], which characterizes how
challenging matching of the particular image pair is. The
co-visibility varies significantly from scene to scene. The
histogram of co-visibilities is shown in Fig. 4, providing insights into how “hard” each scene is – without accounting

for some occlusions.



Co-visibility histogram

|Col1|Col2|
|---|---|
|||


|Col1|Col2|
|---|---|
|||
|||
|||
|||
|||


|Col1|Col2|
|---|---|
|||
|||
|||
|||
|||
|||
|||
|||
|||



**Fig. 4 Co-visibility histogram.** We break down the co-visibility measure for each scene in the validation (green) and test (red) sets, as well
as the average (purple). Notice how the statistics may vary significantly
from scene to scene.


For stereo, the minimum co-visibility threshold is set to
0.1. For the multi-view task, subsets where at least 100 3D
points are visible in each image, are selected, as in [114,
117]. We find that both criteria work well in practice.


3.3 On the quality of the “ground-truth”


Our core assumption is that accurate poses can be obtained
from large sets of images without human intervention. Such
poses are used as the “ground truth” for evaluation of image matching performance on pairs or small subsets of images – a harder, proxy task. Should this assumption hold, the
(relative) poses retrieved with a large enough number of images would not change as more images are added, and these
poses would be the same regardless of which local feature is
used. To validate this, we pick the scene “Sacre Coeur” and
compute SfM reconstructions with a varying number of images: 100, 200, 400, 800, and 1179 images (the entire “Sacre
Coeur” dataset), where each set contains the previous one;
new images are being added and no images are removed.
We run each reconstruction three times, and report the aver

6 Yuhe Jin et al.


**Fig. 5 Co-visibility examples** – image pairs from validation scenes “Sacre Coeur” (top) and “St. Peter’s Square” (bottom). Image keypoints that
are part of the 3D reconstruction are blue if they are co-visible in both images, are red otherwise. The bounding box of the co-visible points, used
to compute a per-image co-visibility ratio, is shown in blue. The co-visibility value for the image pair is the lower of these two values. Examples
include different ‘difficulty’ levels. All of the pairs are used in the evaluation except the top-right one, as we set a cut-off at 0.1.



Number of Images
Local featured type


100 200 400 800 all


SIFT [54] 0.06 _[◦]_ 0.09 _[◦]_ 0.06 _[◦]_ 0.07 _[◦]_ 0.09 _[◦]_

SIFT (Upright) [54] 0.07 _[◦]_ 0.07 _[◦]_ 0.04 _[◦]_ 0.06 _[◦]_ 0.09 _[◦]_

HardNet (Upright) [62] 0.06 _[◦]_ 0.06 _[◦]_ 0.06 _[◦]_ 0.04 _[◦]_ 0.05 _[◦]_

SuperPoint [34] 0.31 _[◦]_ 0.25 _[◦]_ 0.33 _[◦]_ 0.19 _[◦]_ 0.32 _[◦]_

R2D2 [80] 0.12 _[◦]_ 0.08 _[◦]_ 0.07 _[◦]_ 0.08 _[◦]_ 0.05 _[◦]_


**Table 2** **Standard deviation of the pose difference of three**
**COLMAP runs with different number of images** . Most of them are
below 0.1 _[◦]_, except for SuperPoint.


age result of the three runs, to account for the variance inside COLMAP. The standard deviation among different runs
is reported in Table 2. Note that these reconstructions are
only defined up to a scale factor – we do not know the absolute scale that could be used to compare the reconstructions
against each other. That is why we use a simple, pairwise
metric instead. We pick all the pairs out of the 100 images



Number of images
Local feature type


100 vs. all 200 vs. all 400 vs. all 800 vs. all


SIFT [54] 0.58 _[◦]_ / 0.22 _[◦]_ 0.31 _[◦]_ / 0.08 _[◦]_ 0.23 _[◦]_ / 0.05 _[◦]_ 0.18 _[◦]_ / 0.04 _[◦]_

SIFT (Upright) [54] 0.52 _[◦]_ / 0.16 _[◦]_ 0.29 _[◦]_ / 0.08 _[◦]_ 0.22 _[◦]_ / 0.05 _[◦]_ 0.16 _[◦]_ / 0.03 _[◦]_

HardNet (Upright) [62] 0.35 _[◦]_ / 0.10 _[◦]_ 0.33 _[◦]_ / 0.08 _[◦]_ 0.23 _[◦]_ / 0.06 _[◦]_ 0.14 _[◦]_ / 0.04 _[◦]_

SuperPoint [34] 1.22 _[◦]_ / 0.71 _[◦]_ 1.11 _[◦]_ / 0.67 _[◦]_ 1.08 _[◦]_ / 0.48 _[◦]_ 0.74 _[◦]_ / 0.38 _[◦]_

R2D2 [80] 0.49 _[◦]_ / 0.14 _[◦]_ 0.32 _[◦]_ / 0.10 _[◦]_ 0.25 _[◦]_ / 0.08 _[◦]_ 0.18 _[◦]_ / 0.05 _[◦]_


**Table 3** **Pose convergence in SfM.** We report the mean/median of
the difference (in degrees) between the poses extracted with the full set
of 1179 images for “Sacre Coeur”, and different subsets of it, for four
local feature methods – to keep the results comparable we only look at
the 100 images in common across all subsets. We report the maximum
among the angular difference between rotation matrices and translation
vectors. The estimated poses are stable, with as little as 100 images.


present in the smallest subset, and compare how much their
relative pose change with respect to their counterparts reconstructed using the entire set – we do this for every subset,


Image Matching Across Wide Baselines: From Paper to Practice 7


Reference Compared to


SIFT (Upright) HardNet (Upright) SuperPoint R2D2


SIFT [54] 0.20 _[◦]_ / 0.05 _[◦]_ 0.26 _[◦]_ / 0.05 _[◦]_ 1.01 _[◦]_ / 0.62 _[◦]_ 0.26 _[◦]_ / 0.09 _[◦]_


**Table 4** **Difference between poses obtained with different local**
**features.** We report the mean/median of the difference (in degrees)
between the poses extracted with SIFT (Upright), HardNet (Upright),
SuperPoint, or R2D2, and those extracted with SIFT. We use the maximum of the angular difference between rotation matrices and translation vectors. SIFT (Upright), HardNet (Upright), and R2D2 give nearidentical results to SIFT.


(a) SIFT (b) SuperPoint (c) R2D2



_i.e._, 100, 200, etc. Ideally, we would like the differences between the relative poses to approach zero as more images are
added. We list the results in Table 3, for different local feature methods. Notice how the poses converge, especially in
terms of the median, as more images are used, for all methods – and that the reconstructions using only 100 images are
already very stable. For SuperPoint we use a smaller number
of features (2k per image), which is not enough to achieve
pose convergence, but the error is still reduced as more images are used.


We conduct a second experiment to verify that there is
no bias towards using SIFT features for obtaining the ground
truth. We compare the poses obtained with SIFT to those obtained with other local features – note that our primary metric uses nothing but the _estimated poses_ for evaluation. We
report results in Table 4. We also show the histogram of pose
differences in Fig. 7. The differences in pose due to the use
of different local features have a median value below 0.1 _[◦]_ .

In fact, the pose variation of individual reconstructions with
SuperPoint is of the same magnitude as the difference between reconstructions from SuperPoint and other local features: see Table 3. We conjecture that the reconstructions
with SuperPoint, which does not extract a large number of
keypoints, are less accurate and stable. This is further supported by the fact that the point cloud obtained with the entire scene generated with SuperPoint is less dense (125K 3D
points) than the ones generated with SIFT (438K) or R2D2
(317k); see Fig. 6. In addition, we note that SuperPoint keypoints have been shown to be less accurate when it comes
to precise alignment [34, Table 4, _ϵ_ = 1]. Note also that the
poses from R2D2 are nearly identical to those from SIFT.


These observations reinforce our trust in the accuracy
of our ground truth – given a sufficient number of images,
the choice of local feature is irrelevant, at least for the purpose of retrieving accurate poses. Our evaluation considers
pose errors of up to 10 _[◦]_, at a resolution of 1 _[◦]_ – significantly
smaller than the fluctuations observed here, which we consider negligible. Note that these conclusions may not hold
on large-scale SfM requiring loop closures, but our dataset
contains landmarks, which do not suffer from this problem.



**Fig. 6 COLMAP with different local features.** We show the reconstructed point cloud for the scene “Sacre Coeur” using three different
local features: SIFT, SuperPoint, and R2D2, using all images available
(1179). The reconstructions with SIFT and R2D2 are both dense, albeit
somewhat different. The reconstruction with SuperPoint is quite dense,
considering it can only extract a much smaller number of features effectively, but its poses appear less accurate.


In addition, we note that the use of dense, ground truth
depth from SfM, which is arguably less accurate that camera
poses, has been verified by multiple parties, for training and
evaluation, including: CNe [114], DFE [79], LF-Net [71],
D2-Net [37], LogPolarDesc [39], OANet [118], and SuperGlue [84] among others, suggesting it is sufficiently accurate
– several of these rely on the data used in our paper.
As a final observation, while the poses are _stable_, they
could still be _incorrect_ . This can happen on highly symmetric structures: for instance, a tower with a square or circular
cross section. In order to prevent such errors from creeping
into our evaluation, we visually inspected all the images in
our test set. Out of 900 of them, we found 4 misregistered
samples, all of them from the same scene, “London Bridge”,

which were removed from our data.


**4 Pipeline**


We outline our pipeline in Fig. 8. It takes as input _N_ =100
images per scene. The feature extraction module computes
up to _K_ features from each image. The feature matching
module generates a list of putative matches for each image
pair, _i.e._ [1] 2 _[N]_ [(] _[N][ −]_ [1) = 4950][ combinations. These matches]

can be optionally processed by an outlier pre-filtering module. They are then fed to two tasks: stereo, and multiview re
construction with SfM. We now describe each of these com
ponents in detail.


4.1 Feature extraction


We consider three broad families of local features. The first
includes full, “classical” pipelines, most of them handcrafted: SIFT [54] (and RootSIFT [8]), SURF [18], ORB [83],


8 Yuhe Jin et al.




























|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|~~ard~~|~~et (Uprig~~|~~t) vs. SIFT (~~|~~Uprig~~|
|||||
|||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|~~ardNet (Up~~|~~right) vs. S~~|~~per~~|~~Poin~~|
|||||
|||||


|Col1|Col2|
|---|---|
|||
|||
|~~IFT vs. SIFT (U~~|~~IFT vs. SIFT (U~~|
|||
|||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||~~IFT v~~|~~s. Super~~|~~s. Super~~|~~s. Super~~|
||||||
||||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|~~FT (U~~|~~pright) ~~|~~vs. Super~~|~~vs. Super~~|
|||||
|||||





**Fig. 7 Histograms of pose differences between reconstructions with different local feature methods.** We consider five different local features
– including rotation-sensitive and upright SIFT – resulting in 10 combinations. The plots show that about 80% percent of image pairs are within a
0.2 _[o]_ pose difference, with the exception of those involving SuperPoint.























**Fig. 8** **The benchmark pipeline** takes a subset of _N_ images of a scene as input, extracts features for each, and computes matches for all _M_
image pairs, _M_ = [1] 2 _[N]_ [(] _[N][ −]_ [1)][. After an optional filtering step, the matches are fed to two different tasks. Performance is measured] _[ downstream]_ [,]

by a pose-based metric, common across tasks. The ground truth is extracted once, on the full set of images.



and AKAZE [5]. We also consider FREAK [4] descriptors with BRISK [52] keypoints. We take these from
OpenCV. For all of them, except ORB, we lower the detection threshold to extract more features, which increases
performance when operating with a large feature budget.
We also consider DoG alternatives from VLFeat [107]:
(VL-)DoG, Hessian [19], Hessian-Laplace [61], HarrisLaplace [61], MSER [59]; and their affine-covariant versions: DoG-Affine, Hessian-Affine [61, 17], DoG-AffNet

[64], and Hessian-AffNet [64].


The second group includes descriptors learned on DoG
keypoints: L2-Net [102], Hardnet [62], Geodesc [56], SOSNet [103], ContextDesc [55], and LogPolarDesc [39].


The last group consists of pipelines learned end-to-end
(e2e): Superpoint [34], LF-Net [71], D2-Net [37] (with both
single- (SS) and multi-scale (MS) variants), and R2D2 [80].


Additionally, we consider Key.Net [16], a learned detector paired with HardNet and SOSNet descriptors – we pair
it with original implementation of HardNet instead than the
one provided by the authors, as it performs better [3] . **Post-**
**IJCV update.** We have added 3 more local features to the
benchmark after official publication of the paper – DoGAffNet-HardNet [62, **?** ], DoG-TFeat [13] and DoG-MKDConcat [66]. We take them from the kornia [38] library. Re

3 In [16] the models are converted to TensorFlow – we use the original PyTorch version.



sults of this features are included into the Tables and Figures, but not discussed in the text.


4.2 Feature matching


We break this step into four stages. Given images **I** _[i]_ and **I** _[j]_,
_i ̸_ = _j_, we create an initial set of matches by nearest neighbor (NN) matching from **I** _[i]_ to **I** _[j]_, obtaining a set of matches
**m** _i_ � _j_ . We optionally do the same in the opposite direction,
**m** _j_ � _i_ . Lowe’s ratio test [54] is applied to each list to filter
out non-discriminative matches, with a threshold _r ∈_ [0 _,_ 1],
creating “curated” lists ˜ **m** _i_ � _j_ and ˜ **m** _j_ � _i_ . The final set of putative matches is lists intersection, ˜ **m** _i_ � _j_ _∩_ **m** ˜ _j_ � _i_ = ˜ **m** _[∩]_ _i↔j_
(known in the literature as one-to-one, mutual NN, bipartite,
or cycle-consistent), or union ˜ **m** _i_ � _j_ _∪_ **m** ˜ _j→i_ = ˜ **m** _[∪]_ _i↔j_ [(sym-]
metric). We refer to them as “both” and “either”, respectively. We also implement a simple unidirectional matching,
_i.e._, ˜ **m** _i_ � _j_ . Finally, the distance filter is optionally applied,
removing matches whose distance is above a threshold.


The “both” strategy is similar to the “symmetrical nearest neighbor ratio” (sNNR) [20], proposed concurrently –
SNNR combines the nearest neighbor ratio in both directions into a single number by taking the harmonic mean,

while our test takes the maximum of the two values.


Image Matching Across Wide Baselines: From Paper to Practice 9



4.3 Outlier pre-filtering


Context Networks [114], or CNe for short, proposed a
method to find sparse correspondences with a permutationequivariant deep network based on PointNet [77], sparking a
number of follow-up works [79, 32, 120, 117, 97]. We embed
CNe into our framework. It often works best when paired
with RANSAC [114,97], so we consider it as an _optional_
pre-filtering step before RANSAC – and apply it to both
stereo and multiview. As the published model was trained
on one of our validation scenes, we re-train it on “Notre
Dame Front Facade” and “Buckingham Palace”, following
CNe training protocol, _i.e._, with 2000 SIFT features, unidirectional matching, and no ratio test. We evaluated the new
model on the test set and observed that its performance is
better than the one that was released by the authors. It could
be further improved by using different matching schemes,
such as bidirectional matching, but we have not explored
this in this paper and leave as future work.


We perform one additional, but necessary, change: CNe
(like most of its successors) was originally trained to esti
mate the Essential matrix instead of the Fundamental ma
trix [114], _i.e._, it assumes known intrinsics. In order to use
it within our setup, we normalize the coordinates by the size
of the image instead of using ground truth calibration matrices. This strategy has also been used in [97], and has been
shown to work well in practice.


4.4 Stereo task


The list of tentative matches is given to a robust estimator, which estimates **F** _i,j_, the Fundamental matrix between
**I** _i_ and **I** _j_ . In addition to (locally-optimized) RANSAC [40,
29], as implemented in OpenCV [23], and sklearn [72], we
consider more recent algorithms with publicly available implementations: DEGENSAC [30], GC-RANSAC [14] and
MAGSAC [15]. We use the original GC-RANSAC implementation [4], and not the most up-to-date version, which incorporates MAGSAC, DEGENSAC, and later changes.


For DEGENSAC we additionally consider disabling the
degeneracy check, which theoretically should be equivalent to the OpenCV and sklearn implementations – we call
this variant “PyRANSAC”. Given **F** _i,j_, the known intrinsics
**K** _{i,j}_ were used to compute the Essential matrix **E** _i,j_, as
**E** _i,j_ = **K** _[T]_ _j_ **[F]** _[i,j]_ **[K]** _[i]_ [. Finally, the relative rotation and trans-]
lation vectors were recovered with a cheirality check with
OpenCV’s recoverPose.


4 [https://github.com/danini/graph-cut-ransac/](https://github.com/danini/graph-cut-ransac/tree/benchmark-version)
[tree/benchmark-version](https://github.com/danini/graph-cut-ransac/tree/benchmark-version)



4.5 Multiview task


Large-scale SfM is notoriously hard to evaluate, as it requires accurate ground truth. Since our goal is to benchmark _local features_ and _matching methods_, and not SfM algorithms, we opt for a different strategy. We reconstruct a
scene from small image subsets, which we call “bags”. We
consider bags of 5, 10, and 25 images, which are randomly
sampled from the original set of 100 images per scene – with
a co-visibility check. We create 100 bags for bag sizes 5, 50
for bag size 10, and 25 for bag size 25 – _i.e._, 175 SfM runs

in total.

We use COLMAP [90], feeding it the matches computed
by the previous module – note that this comes before the
robust estimation step, as COLMAP implements its own
RANSAC. If multiple reconstructions are obtained, we consider the largest one. We also collect and report statistics
such as the number of landmarks or the average track length.
Both statistics and error metrics are averaged over the three
bag sizes, each of which is in turn averaged over its individual bags.


4.6 Error metrics


Since the stereo problem is defined up to a scale factor [44],
our main error metric is based on _angular errors_ . We compute the difference, in degrees, between the _estimated_ and
_ground-truth_ translation and rotation _vectors_ between two
cameras. We then threshold it over a given value for all possible – _i.e._, co-visible – pairs of images. Doing so over different angular thresholds renders a curve. We compute the
mean Average Accuracy (mAA) by integrating this curve
up to a maximum threshold, which we set to 10 _[◦]_ – this is
necessary because large errors always indicate a bad pose:
30 _[◦]_ is not necessarily better than 180 _[◦]_, both estimates are
wrong. Note that by computing the area under the curve we
are giving more weight to methods which are more accurate
at lower error thresholds, compared to using a single value
at a certain designated threshold.
This metric was originally introduced in [114], where it
was called mean Average Precision (mAP). We argue that
“accuracy” is the correct terminology, since we are simply
evaluating how many of the predicted poses are “correct”,
as determined by thresholding over a given value – _i.e._, our
problem does not have “false positives”.

The same metric is used for multiview. Because we do

not know the scale of the scene _a priori_, it is not possible to

measure translation error in metric terms. While we intend

to explore this in the future, such a metric, while more interpretable, is not without problems – for instance, the range
of the distance between the camera and the scene can vary
drastically from scene to scene and make it difficult to compare their results. To compute the mAA in pose estimation


10 Yuhe Jin et al.



for the multiview task, we take the mean of the average accuracy for every pair of cameras – setting the pose error to
_∞_ for pairs containing unregistered views. If COLMAP returns multiple models which cannot be co-registered (which
is rare) we consider only the largest of them for simplicity.

For the stereo task, we can report this value for different
co-visibility thresholds: we use _v_ = 0 _._ 1 by default, which
preserves most of the “hard” pairs. Note that this is not applicable to the multiview task, as all images are registered at
once via bundle adjustment in SfM.

Finally, we consider repeatability and matching score.
Since many end-to-end methods do not report and often do
not have a clear measure of scale – or support region – we
simply threshold by pixel distance, as in [82]. For the multiview task, we also compute the Absolute Trajectory Error
(ATE) [96], a metric widely used in SLAM. Since, once
again, the reconstructed model is scale-agnostic, we first
scale the reconstructed model to that of the ground truth and
then compute the ATE. Note that ATE needs a minimum of
three points to align the two models.


4.7 Implementation


The benchmark code has been open-sourced [1] along with
every method used in the paper [5] . The implementation relies on SLURM [115] for scalable job scheduling, which is
compatible with our supercomputer clusters – we also provide on-the-cloud, ready-to-go images [6] . The benchmark can
also run on a standard computer, sequentially. It is computationally expensive, as it requires matching about 45k
image pairs. The most costly step – leaving aside feature
extraction, which is very method-dependent – is typically
feature matching: 2–6 seconds per image pair [7], depending
on descriptor size. Outlier pre-filtering takes about 0.5–0.8
seconds per pair, excluding some overhead to reformat the
data into its expected format. RANSAC methods vary between 0.5–1 second – as explained in Section 5 we limit
their number of iterations based on a compute budget, but
the actual cost depends on the number of matches. Note that
these values are computed on the validation set – for the test
set experiments we increase the RANSAC budget, in order
to remain compatible with the rules of the Image Matching
Challenge [2] . We find COLMAP to vary drastically between
set-ups. New methods will be continuously added, and we

welcome contributions to the code base.


5 [https://github.com/vcg-uvic/](https://github.com/vcg-uvic/image-matching-benchmark-baselines)
[image-matching-benchmark-baselines](https://github.com/vcg-uvic/image-matching-benchmark-baselines)

6 [https://github.com/etrulls/slurm-gcp](https://github.com/etrulls/slurm-gcp)
7 Time measured on ‘n1-standard-2’ VMs on Google Cloud Compute: 2 vCPUs with 7.5 GB of RAM and no GPU.


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||



**Fig. 9 Validation – Performance vs. cost for RANSAC.** We evaluate
six RANSAC variants, using 8k SIFT features with “both” matching
and a ratio test threshold of _r_ =0 _._ 8. The inlier threshold _η_ and iterations limit _Γ_ are variables – we plot only the best _η_ for each method,
for clarity, and set a budget of 0.5 seconds per image pair (dotted red
line). For each RANSAC variant, we pick the largest _Γ_ under this time
“limit” and use it for all validation experiments. Computed on ‘n1standard-2’ VMs on Google Compute (2 vCPUs, 7.5 GB).


**5 Details are Important**


Our experiments indicate that each method needs to be carefully tuned. In this section we outline the methodology we
used to find the right hyperparameters on the validation set,
and demonstrate why it is crucial to do so.


5.1 RANSAC: Leveling the field


Robust estimators are, in our experience, the most sensitive
part of the stereo pipeline, and thus the one we first turn to.
All methods considered in this paper have three parameters
in common: the confidence level in their estimates, _τ_ ; the
outlier (epipolar) threshold, _η_ ; and the maximum number of
iterations, _Γ_ . We find the confidence value to be the least
sensitive, so we set it to _τ_ = 0 _._ 999999.

We evaluate each method with different values for _Γ_

and _η_, using reasonable defaults: 8k SIFT features with bidirectional matching with the “both” strategy and a ratio test
threshold of 0.8. We plot the results in Fig. 9, against their
computational cost – for the sake of clarity we only show
the curve corresponding to the best reprojection threshold _η_

for each method.

Our aim with this experiment is to place all methods on
an “even ground” by setting a common budget, as we need
to find a way to compare them. We pick 0.5 seconds, where
all methods have mostly converged. Note that these are different implementations and are obviously not directly comparable to each other, but this is a simple and reasonable
approach. We set this budget by choosing _Γ_ as per Fig. 9,
instead of actually _enforcing_ a time limit, which would not
be comparable across different set-ups. Optimal values for
_Γ_ can vary drastically, from 10k for MAGSAC to 250k for



0.52


0.50


0.48


0.46


0.44


0.42


0.40


0.38


0.36



Performance vs cost: mAA(5 _[o]_ )



0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
Time in seconds (per image pair)



0.60 Performance vs cost: mAA(10 _[o]_ )


0.58


0.56


0.54


0.52


0.50


0.48


0.46


0.44


0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
Time in seconds (per image pair)



CV-RANSAC, _η_ = 0.5 px
sklearn-RANSAC, _η_ = 0.75 px



PyRANSAC, _η_ = 0.25 px

DEGENSAC, _η_ = 0.5 px



MAGSAC, _η_ = 1.25 px

GC-RANSAC, _η_ = 0.5 px


Image Matching Across Wide Baselines: From Paper to Practice 11



MULTIVIEW: mAA(10 _[o]_ )


60 65 70 75 80 85 90 95 None

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||



Ratio test _r_



STEREO: mAA(10 _[o]_ )


60 65 70 75 80 85 90 95 None

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||



Ratio test _r_



0.7


0.6


0.5


0.4


0.3


0.2


0.1

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||



0.7


0.6


0.5


0.4


0.3


0.2


0.1



STEREO: mAA(10 _[o]_ ) vs Inlier Threshold


(a) PyRANSAC (b) DEGENSAC


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||



GeoDesc

Key.Net-HardNet

Key.Net-SOSNet

L2-Net

LF-Net (2k)



(d) MAGSAC


Inlier threshold _η_



0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0



CV-AKAZE

CV-FREAK

CV-ORB

CV-RootSIFT

CV-SIFT



CV-SURF

ContextDesc

D2-Net (MS)

D2-Net (SS)

DoG-HardNet



GeoDesc

Key.Net-HardNet

Key.Net-SOSNet

L2-Net

LF-Net (2k points)



LogPolarDesc
R2D2 (wasf-n8-big)

SOSNet

SuperPoint (2k points)



VL-DoG-SIFT

VL-DoGAff-SIFT

VL-Hess-SIFT

VL-HessAffNet-SIFT



SuperPoint (2k)

VL-DoG-SIFT

VL-DoGAff-SIFT

VL-Hess-SIFT

VL-HessAffNet-SIFT



CV-AKAZE

CV-FREAK

CV-ORB

CV-RootSIFT

CV-SIFT



(c) GC-RANSAC


Inlier threshold _η_


CV-SURF

ContextDesc

D2-Net (MS)

D2-Net (SS)

DoG-HardNet



LogPolarDesc

R2D2 (waf-n16)

R2D2 (wasf-n16)

R2D2 (wasf-n8-big)

SOSNet



**Fig. 10 Validation – Inlier threshold for RANSAC,** _η_ **.** We determine
_η_ for each combination, using 8k features (2k for LF-Net and SuperPoint) with the “both” matching strategy and a reasonable value for the
ratio test. Optimal parameters (diamonds) are listed in the Section 7.


PyRANSAC. MAGSAC gives the best results for this experiment, closely followed by DEGENSAC. We patch OpenCV
to increase the limit of iterations, which was hardcoded
to _Γ_ = 1000; this patch is now integrated into OpenCV.
This increases performance by 10-15% relative, within our
budget. However, PyRANSAC is significantly better than
OpenCV version even with this patch, so we use it as our
“vanilla” RANSAC instead. The sklearn implementation is
too slow for practical use.
We find that, in general, default settings can be woefully
inadequate. For example, OpenCV recommends _τ_ = 0 _._ 99
and _η_ = 3 pixels, which results in a mAA at 10 _[◦]_ of 0.3642
on the validation set – a performance drop of 29.3% relative.


5.2 RANSAC: One method at a time


The last free parameter is the inlier threshold _η_ . We expect
the optimal value for this parameter to be different for each
local feature, with looser thresholds required for methods
operating on higher recall/lower precision, and end-to-end

methods trained on lower resolutions.

We report a wide array of experiments in Fig. 10 that
confirm our intuition: descriptors learned on DoG keypoints
are clustered, while others vary significantly. Optimal values

are also different for each RANSAC variant. We use the ratio



**Fig. 11 Validation – Optimal ratio test** _r_ **for matching with “both”.**
We evaluate bidirectional matching with the “both” strategy (the best
one), and different ratio test thresholds _r_, for each feature type. We
use 8k features (2k for SuperPoint and LF-Net). For stereo, we use
PyRANSAC.


test with the threshold recommended by the authors of each
feature, or a reasonable value if no recommendation exists,
and the “both” matching strategy – this cuts down on the

number of outliers.


5.3 Ratio test: One feature at a time


Having “frozen” RANSAC, we turn to the feature matcher
– note that it comes _before_ RANSAC, but it cannot be evaluated in isolation. We select PyRANSAC as a “baseline”
RANSAC and evaluate different ratio test thresholds, separately for the stereo and multiview tasks. For this experiment, we use 8k features with all methods, except for those
which cannot work on this regime – SuperPoint and LF-Net.
This choice will be substantiated in Section 5.4. We report
the results for bidirectional matching with the “both” strategy in Fig. 11, and with the “either” strategy in Fig. 12. We
find that “both” – the method we have used so far – performs
best overall. Bidirectional matching with the “either” strategy produces many (false) matches, increasing the computational cost in the estimator, and requires very small ratio
test thresholds – as low as _r_ =0 _._ 65. Our experiments with
unidirectional matching indicate that is slightly worse, and
it depends on the order of the images, so we did not explore

it further.

As expected, each feature requires different settings, as
the distribution of their descriptors is different. We also observe that optimal values vary significantly between stereo
and multiview, even though one might expect that bundle
adjustment should be able to better deal with outliers. We
suspect that this indicates that there is room for improvement in COLMAP’s implementation of RANSAC.


12 Yuhe Jin et al.



MULTIVIEW: mAA(10 _[o]_ )

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||



60 65 70 75 80 85 90 95 None

Ratio test _r_



STEREO (mAA(10 _[o]_ )): Ratio Test VS Distance Threshold



0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0



STEREO: mAA(10 _[o]_ )

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||



60 65 70 75 80 85 90 95 None

Ratio test _r_



ORB

16 Hamming Distance Threshold 24 32 40 48 64



FREAK

16 Hamming Distance Threshold 32 64 96 128


0.4


0.3


0.2


0.1



AKAZE

16 Hamming Distance Threshold 32 64 96 128


0.4


0.3


0.2


0.1



0.4


0.3


0.2


0.1





0.0



0.0



0.0


|Col1|M|atc|h RT|: ”eit|her|”|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||M<br>M|~~at~~<br>atc<br>atc|~~h R~~<br>h DT<br>h DT|~~: ”bo~~<br>: ”eit<br>: ”bo|~~h”~~<br>he<br>th”|r”<br>|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||



0.60 0.70 0.80 0.90 None

Ratio test



0.60 0.70 0.80 0.90 None

Ratio test



0.60 0.70 0.80 0.90 None

Ratio test



CV-AKAZE

CV-DoG/HardNet

CV-FREAK



CV-ORB

CV-RootSIFT

CV-SIFT



CV-SURF

D2-Net (MS)

D2-Net (SS)



GeoDesc

LF-Net (2k points)
LogPolarDesc



SOSNet

SuperPoint (2k)



**Fig. 12** **Validation – Optimal ratio test** _r_ **for matching with “ei-**
**ther”.** Equivalent to Fig. 11 but with the “either” matching strategy.
This strategy requires aggressive filtering and does not reach the performance of “both”, we thus explore only a subset of the methods.



MULTIVIEW: mAA(10 _[o]_ )

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||U<br>Bi|nid<br>di|ire<br>ect|cti<br>o|onal<br>nal +|(_r_<br>’e|=0.80<br>ither’|)<br> (_r_=0.65)|||||||||||||||
|||||||||||||||||||||||||||
|||||Bi|dir|ect|io|nal +|’b|oth’ (|_r_=0.80)|||||||||||||||
|||||||||||||||||||||||||||



500 2k 4k 6k 8k 10k 12k 15k

Number of features



**Fig. 14** **Validation – Matching binary descriptors.** We filter out
non-discriminative matches with the ratio test or a distance threshold.

The latter (the standard) performs worse in our experiments.


for different values of _K_ in Fig. 13. We use PyRANSAC
with reasonable defaults for all three matching strategies,

with SIFT features.

As expected, performance is strongly correlated with the
number of features. We find 8k to be a good compromise between performance and cost, and also consider 2k (actually
2048) as a ‘cheaper’ alternative – this also provides a fair
comparison with some learned methods which only operate
on that regime. We choose these two values as valid categories for the open challenge [2] linked to the benchmark, and
do the same on this paper for consistency.


5.5 Additional experiments


Some methods require additional considerations before
evaluating them on the test set. We briefly discuss them in
this section. Further experiments are available in Section 7.


**Binary features (Fig. 14).** We consider three binary descriptors: ORB [83], AKAZE [5], and FREAK [4] Binary
descriptor papers historically favour a distance threshold in
place of the ratio test to reject non-discriminative matches

[83], although some papers have used the ratio test for ORB
descriptors [6]. We evaluate both in Fig. 14 – as before, we
use up to 8k features and matching with the “both” strategy.

The ratio test works better for all three methods – we use

it instead of a distance threshold for all experiments in the
paper, including those in the previous sections.


**On the influence of the detector (Fig. 15).** We embed several popular blob and corner detectors into our pipeline,
with OpenCV’s DoG [54] as a baseline. We combine multiple methods, taking advantage of the VLFeat library: Difference of Gaussians (DoG), Hessian [19], HessianLaplace

[61], HarrisLaplace [61], MSER [59], DoGAffine, HessianAffine [61,17], DoG-AffNet [64], and Hessian-AffNet [64].
We pair them with SIFT descriptors, also computed with
VLFeat, as OpenCV cannot process affine keypoints, and
report the results in Fig. 15. VLFeat’s DoG performs



0.6


0.5


0.4


0.3


0.2


0.1


0.0



STEREO: mAA(10 _[o]_ )


500 2k 4k 6k 8k 10k 12k 15k

Number of features



**Fig. 13 Validation – Number of features.** Performance on the stereo
and multi-view tasks while varying the number of SIFT features, with
three matching strategies, and reasonable defaults for the ratio test _r_ .


Note how the ratio test is critical for performance, and
one could arbitrarily select a threshold that favours one
method over another, which shows the importance of proper
benchmarking. Interestingly, D2-Net is the _only_ method that
clearly performs best without the ratio test. It also performs
poorly overall in our evaluation, despite reporting state-ofthe-art results in other benchmarks [60, 11, 85, 98] – without
the ratio test, the number of tentative matches might be too
high for RANSAC or COLMAP to perform well.
Additionally, we implement the first-geometricinconsistent ratio threshold, or FGINN [63]. We find that
although it improves over unidirectional matching, its gains
mostly disappear against matching with “both”. We report

these results in Section 7.2.


5.4 Choosing the number of features


The ablation tests in this section use (up to) _K_ =8000 feature
(2k for SuperPoint and LF-Net, as they are trained to extract
fewer keypoints). This number is commensurate with that
used by SfM frameworks [112, 90]. We report performance


Image Matching Across Wide Baselines: From Paper to Practice 13



0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0



STEREO w/ SIFT descriptors

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



Detector



0.60


0.58


0.56


0.54


0.52


0.50


0.48



STEREO w/ SIFT descriptors


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||



**Fig. 15 Validation – Benchmarking detectors.** We evaluate the performance on the stereo task while pairing different detectors with SIFT
descriptors. The dashed, black line indicates OpenCV SIFT – the baseline. **Left:** OpenCV DoG vs. VLFeat implementations of blob detectors (DoG, Hessian, HesLap) and corner detectors (Harris, HarLap),
and MSER. **Right:** Affine shape estimation for DoG and Hessian keypoints, against the plain version. We consider a classical approach,
Baumberg (Affine) [17], and the recent, learned AffNet [64] – they
provide a small but inconsistent boost.


STEREO: mAA(10 _[o]_ )


0.60


0.55


0.50


0.45


0.40



0.35




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||SIFT|SIFT|
||||||HardNe|t|



6 8 12 16 20 24
Patch scaling factor


**Fig. 16** **Validation – Scaling the descriptor support region.** Performance with SIFT and HardNet descriptors while applying a scaling
factor _λ_ to the keypoint scale (note that OpenCV’s default value is
_λ_ =12). We consider SIFT and HardNet. Default values are optimal or
near-optimal.


marginally better than OpenCV’s. Its affine version gives a
small boost. Given the small gain and the infrastructure burden of interacting with a Matlab/C library, we use OpenCV’s
DoG implementation for most of this paper.


**On increasing the support region (Fig. 16).** The size
(“scale”) of the support region used to compute a descriptor can significantly affect its performance [36, 116, 2]. We
experiment with different scaling factors, using DoG with
SIFT and HardNet [62], and find that 12 _×_ the OpenCV scale
(the default value) is already nearly optimal, confirming the
findings reported in [39]. We show these results in Fig. 16.
Interestingly, SIFT descriptors do benefit from increasing
the scaling factor from 12 to 16, but the difference is very

small – we thus use the recommended value of 12 for the

rest of the paper. This, however, suggests that deep descriptors such as HardNet might be able to increase performance
slightly by training on larger patches.



**6 Establishing the State of the Art**


With the findings and the optimal parameters found in Section 5, we move on to the test set, evaluating many methods with their optimal settings. All experiments in this
section use bidirectional matching with the ‘’both” strategy. We consider a large feature budget (up to 8k features)
and a smaller one (up to 2k), and evaluate many detector/descriptor combinations.
We make three changes with respect to the validation experiments of the previous section. (1) We double the RANSAC budget from 0.5 seconds (used for validation) to 1 second per image pair, and adjust the maximum number of iterations _Γ_ accordingly – we made this decision to encourage participants to the challenge based on this benchmark to

use built-in methods rather than run RANSAC themselves to

squeeze out a little extra performance, and use the same values in the paper for consistency. (2) We run each stereo and
multiview evaluation three times and average the results, in
order to decrease the potential randomness in the results –
in general, we found the variations within these three runs
to be negligible. (3) We use brute-force to match descriptors
instead of FLANN, as we observed a drop in performance.
For more details, see Section 6.6 and Table 12.

For stereo, we consider DEGENSAC and MAGSAC,
which perform the best in the validation set, and PyRANSAC as a ‘baseline’ RANSAC. We report the results with
both 8k features and 2k features in the following subsections. All observations are in terms of mAA, our primary
metric, unless stated otherwise.


6.1 Results with 8k features — Tables 5 and 6


On the stereo task, deep descriptors extracted on DoG keypoints are at the top in terms of mAA, with SOSNet being
#1, closely followed by HardNet. Interestingly, ‘HardNetAmos+’ [76], a version trained on more datasets – Brown [24],
HPatches [11], and AMOS-patches [76] – performs worse
than the original models, trained only on the “Liberty” scene
from Brown’s dataset. On the multiview task, HardNet edges
out ContextDesc, SOSNet and LogpolarDesc by a small
margin.
We also pair HardNet and SOSNet with Key.Net, a
learned detector, which performs worse than with DoG
when extracting a large number of features, with the exception of Key.Net + SOSNet on the multiview task.
R2D2, the best performing end-to-end method, does
well on multiview (#7), but performs worse than SIFT on
stereo – it produces a much larger number of “inliers”
(which may be correct or incorrect) than most other methods. This suggests that, like D2-Net, its lack of compatibility with the ratio test may be a problem when paired with
sample-based robust estimators, due to a lower inlier ratio.


14 Yuhe Jin et al.



PyRANSAC DEGENSAC MAGSAC
Method NF NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ Rank


CV-SIFT 7861.1 167.6 .3996 243.6 .4584 297.4 .4583 14

VL-SIFT 7880.6 179.7 .3999 261.6 .4655 326.2 .4633 13

VL-Hessian-SIFT 8000.0 204.4 .3695 290.2 .4450 348.9 .4335 15

VL-DoGAff-SIFT 7892.1 171.6 .3984 250.1 .4680 317.1 .4666 11

VL-HesAffNet-SIFT 8000.0 209.3 .3933 299.0 .4679 350.0 .4626 12


CV- ~~_√_~~ SIFT 7860.8 192.3 .4228 281.7 .4930 347.5 .4941 10

CV-SURF 7730.0 107.9 .2280 113.6 .2593 145.3 .2552 19

CV-AKAZE 7857.1 131.4 .2570 246.8 .3074 301.8 .3036 17

CV-ORB 7150.2 123.7 .1220 150.0 .1674 178.9 .1570 22

CV-FREAK 8000.0 123.3 .2273 131.0 .2711 196.7 .2656 18


L2-Net 7861.1 213.8 .4621 366.0 .5295 481.0 .5252 5

DoG-HardNet 7861.1 286.5 .4801 432.3 **.5543** 575.1 .5502 2

DoG-HardNetAmos+ 7861.0 265.7 .4607 398.6 .5385 528.7 .5329 3
Key.Net-HardNet 7997.6 448.1 .3997 598.3 .4986 815.4 .4739 9
Key.Net-SOSNet 7997.6 275.5 .4236 587.4 .5019 766.4 .4780 8
GeoDesc 7861.1 205.4 .4328 348.5 .5111 453.4 .5056 7

ContextDesc 7859.0 278.2 .4684 493.6 .5098 544.1 .5143 6

DoG-SOSNet 7861.1 281.6 .4784 424.6 **.5587** 563.3 **.5517** 1
LogPolarDesc 7861.1 254.4 .4574 441.8 .5340 591.2 .5238 4


D2-Net (SS) 5665.3 280.8 .1933 482.3 .2228 781.3 .2032 21
D2-Net (MS) 6924.1 278.2 .2160 470.6 .2506 741.2 .2321 20
R2D2 (wasf-n8-big) 7940.5 457.6 .3683 842.2 .4437 998.9 .4236 16


DoG-AffNet-HardNet 7834.0 267.9 .4505 403.4 .5447 516.8 .5412 3 _[∗]_

DoG-MKD-Concat 7860.8 208.0 .4061 305.8 .4846 381.4 .4810 11 _[∗]_

DoG-TFeat 7860.8 160.8 .4008 234.8 .4649 292.6 .4668 13 _[∗]_


**Table 5** **Test – Stereo results with 8k features.** We report: **(NF)**
Number of Features; **(NI)** Number of Inliers produced by RANSAC;
and **mAA(10** _[o]_ **)** . Top three methods by mAA marked in red, green and
blue. _[∗]_ **The last group of results is obtained after the paper publi-**
**cation and not described in the text of the paper. Their rank does**
**not influence other entries ranks.**


Method NL _[↑]_ SR _[↑]_ RC _[↑]_ TL _[↑]_ mAA(5 _[o]_ ) _[↑]_ mAA(10 _[o]_ ) _[↑]_ ATE _[↓]_ Rank

CV-SIFT 2577.6 96.7 94.1 3.95 .5309 .6261 .4721 14
VL-SIFT 3030.7 97.9 95.4 4.17 .5273 .6283 .4669 13
VL-Hessian-SIFT 3209.1 97.4 94.1 4.13 .4857 .5866 .5175 16
VL-DoGAff-SIFT 3061.5 98.0 96.2 4.11 .5263 .6296 .4751 12
VL-HesAffNet-SIFT 3327.7 97.7 95.2 4.08 .5049 .6069 .4897 15


CV- ~~_√_~~ SIFT 3312.1 98.5 96.6 4.13 .5778 .6765 .4485 9

CV-SURF 2766.2 94.8 92.6 3.47 .3897 .4846 .6251 18
CV-AKAZE 4475.9 99.0 95.4 3.88 .4516 .5553 .5715 17
CV-ORB 3260.3 97.2 91.1 3.45 .2697 .3509 .7377 22
CV-FREAK 2859.1 92.9 91.7 3.53 .3735 .4653 .6229 20

L2-Net 3424.9 98.6 96.2 4.21 .5661 .6644 .4482 10
DoG-HardNet 4001.4 99.5 **97.7** **4.34** **.6090** **.7096** **.4187** 1
DoG-HardNetAmos+ 3550.6 98.8 96.9 4.28 .5879 .6888 .4428 6
Key.Net-HardNet 3366.0 98.9 96.7 4.32 .5391 .6483 .4622 11
Key.Net-SOSNet **5505.5** 100.0 **98.7** **4.46** .5989 **.7038** .4286 2
GeoDesc 3839.0 99.1 97.2 4.26 .5782 .6803 .4445 8
ContextDesc 3732.5 99.3 97.6 4.22 **.6036** **.7035** **.4228** 3
DoG-SOSNet 3796.0 99.3 97.4 4.32 **.6032** .7021 **.4226** 4
LogPolarDesc 4054.6 99.0 96.4 4.32 .5928 .6928 .4340 5

D2-Net (SS) **5893.8** **99.8** 97.5 3.62 .3435 .4598 .6361 21
D2-Net (MS) **6759.3** **99.7** **98.2** 3.39 .3524 .4751 .6283 19
R2D2 (wasf-n8-big) 4432.9 **99.7** 97.2 **4.59** .5775 .6832 .4333 7

DoG-AffNet-HardNet 4671.3 **99.9** **98.1** **4.56** **.6296** **.7267** **.4021** 1 _[∗]_
DoG-MKD-Concat 3507.4 98.5 96.1 4.17 .5461 .6476 .4668 11 _[∗]_
DoG-TFeat 2905.3 97.1 94.8 4.04 .5270 .6261 .4873 14 _[∗]_


**Table 6 Test – Multiview results with 8k features.** We report: **(NL)**
Number of 3D Landmarks; **(SR)** Success Rate (%) in the 3D reconstruction across “bags”; **(RC)** Ratio of Cameras (%) registered in a
“bag”; **(TL)** Track Length or number of observations per landmark;
**mAA** at 5 and 10 _[o]_ ; and **(ATE)** Absolute Trajectory Error. All metrics
are averaged across different “bag” sizes, as explained in Section 4. We
rank them by mAA at 10 _[o]_ and color-code them as in Table 5. _[∗]_ **The**
**last group of results is obtained after the paper publication and**
**not described in the text of the paper. Their rank does not influ-**
**ence other entries ranks.**



Note that D2-net performs poorly on our benchmark, despite state-of-the-art results on others. On the multiview task
it creates many more 3D landmarks than any other method.
Both issues may be related to its poor localization (pixel)
accuracy, due to operating on downsampled feature maps.
Out of the handcrafted methods, SIFT – RootSIFT
specifically – remains competitive, being #10 on stereo and
#9 on multiview, within 13.1% and 4.9% relative of the
top performing method, respectively, while previous benchmarks report differences in performance of _orders of mag-_

_nitude._ Other “classical” features do not fare so well. One

interesting observation is that among these, their ranking on

validation and test set is not consistent – Hessian is better

on validation than DoG, but significantly worse on the test
set, especially in the multiview setup. While this is a special
case, this nonetheless demonstrates that a small-scale benchmark can be misleading, and a method needs to be tested on
a variety of scenes, which is what our test set aims to pro
vide.

Regarding the robust estimators, DEGENSAC and
MAGSAC both perform very well, with the former edging
out the latter for most local feature methods. This may be
due to the nature of the scenes, which often contain dominant planes.


6.2 Results with 2k features — Tables 7 and 8


Results change slightly on the low-budget regime, where
the top two spots on both tasks are occupied by
Key.Net+SOSNet and Key.Net+HardNet. It is closely followed by LogPolarDesc (#3 on stereo and #4 on multiview),
a method trained on DoG keypoints – but using a much
larger support region, resampled into log-polar patches.
R2D2 performs very well on the multiview task (#3), while
once again falling a bit short on the stereo task (#8, and
14.5% relative below the #1 method), for which it retrieves
a number of inliers significantly larger than its competitors.
The rest of the end-to-end methods do not perform so well,
other than SuperPoint, which obtains competitive results on

the multiview task.

The difference between classical and learned methods is

more pronounced than with 8k points, with RootSIFT once
again at the top, but now within 31.4% relative of the #1
method on stereo, and 26.9% on multiview. This is somewhat to be expected, given that with fewer keypoints, the
quality of each individual point matters more.


6.3 2k features vs 8k features — Figs. 17 and 18


We compare the results between the low- and high-budget
regimes in Fig. 17, for stereo (with DEGENSAC), and
Fig. 18, for multiview. Note how methods can behave


Image Matching Across Wide Baselines: From Paper to Practice 15



PyRANSAC DEGENSAC MAGSAC
Method NF NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ Rank


CV-SIFT 2048.0 84.9 .2489 79.0 .2875 99.2 .2805 12


CV- ~~_√_~~ SIFT 2048.0 84.2 .2724 88.3 .3149 106.8 .3125 10

CV-SURF 2048.0 37.9 .1725 72.7 .2086 87.0 .2081 15

CV-AKAZE 2048.0 96.1 .1780 91.0 .2144 115.5 .2127 14

CV-ORB 2031.8 56.3 .0610 63.5 .0819 71.5 .0765 19

CV-FREAK 2048.0 62.5 .1461 65.6 .1761 78.4 .1698 17


L2-Net 1936.3 66.1 .3131 92.4 .3752 114.7 .3691 6

DoG-HardNet 1936.3 111.9 .3508 117.7 .4029 150.5 .4033 5
Key.Net-HardNet 2048.0 134.4 .3272 174.8 **.4139** 228.4 .3897 1
Key.Net-SOSNet 2048.0 88.0 .3279 171.9 **.4132** 212.9 .3928 2
GeoDesc 1936.3 98.9 .3127 103.9 .3662 129.7 .3640 7

ContextDesc 2048.0 118.8 .2965 124.1 .3510 146.4 .3485 9

DoG-SOSNet 1936.3 111.1 .3536 132.1 .3976 149.6 .4092 4
LogPolarDesc 1936.3 118.8 .3569 124.9 **.4115** 161.0 .4064 3


D2-Net (SS) 2045.6 107.6 .1157 134.8 .1355 259.3 .1317 18
D2-Net (MS) 2038.2 149.3 .1524 188.4 .1813 302.9 .1703 16
LF-Net 2020.3 100.2 .1927 106.5 .2344 141.0 .2226 13
SuperPoint 2048.0 120.1 .2577 126.8 .2964 127.3 .2676 11
R2D2 (wasf-n16) 2048.0 191.0 .2829 215.6 .3614 215.6 .3614 8


DoG-AffNet-HardNet 2047.8 105.6 .3589 152.1 **.4197** 195.2 **.4175** 1 _[∗]_


**Table 7** **Test – Stereo results with 2k features.** Same as Table 5.

_∗_ **The last group of results is obtained after the paper publication**
**and not described in the text of the paper. Their rank does not**
**influence other entries ranks.**


Method NL _[↑]_ SR _[↑]_ RC _[↑]_ TL _[↑]_ mAA(5 _[o]_ ) _[↑]_ mAA(10 _[o]_ ) _[↑]_ ATE _[↓]_ Rank

CV-SIFT 1081.2 87.6 87.4 3.70 .3718 .4562 .6136 13


CV- ~~_√_~~ SIFT 1174.7 90.3 89.4 3.82 .4074 .4995 .5589 12

CV-SURF 1186.6 90.2 88.6 3.55 .3335 .4184 .6701 15
CV-AKAZE 1383.9 94.7 90.9 3.74 .3393 .4361 .6422 14
CV-ORB 683.3 74.9 73.0 3.21 .1422 .1914 .8153 19
CV-FREAK 1075.2 87.2 86.3 3.52 .2578 .3297 .7169 17

L2-Net 1253.3 94.7 92.6 3.96 .4369 .5392 .5419 9
DoG-HardNet 1338.2 96.3 93.7 4.03 .4624 .5661 .5093 6
Key.Net-HardNet 1276.3 97.8 **95.7** **4.49** **.5050** **.6161** **.4902** 2
Key.Net-SOSNet 1475.5 **99.3** **96.5** **4.42** **.5229** **.6340** **.4853** 1
GeoDesc 1133.6 93.6 91.3 4.02 .4246 .5244 .5455 10
ContextDesc 1504.9 95.6 93.3 3.92 .4529 .5568 .5327 7
DoG-SOSNet 1317.4 96.0 93.8 4.05 .4739 .5784 .5194 5
LogPolarDesc 1410.2 96.0 93.8 4.05 .4794 .5849 .5090 4

D2-Net (SS) **2357.9** **98.9** 94.7 3.39 .2875 .3943 .7010 16
D2-Net (MS) **2177.3** 98.2 93.4 3.01 .1921 .3007 .7861 20
LF-Net 1385.0 95.6 90.4 4.14 .4156 .5141 .5738 11
SuperPoint 1184.3 95.6 92.4 **4.34** .4423 .5464 .5457 8
R2D2 (wasf-n16) 1228.4 **99.4** **96.2** 4.29 **.5045** **.6149** **.4956** 3

DoG-AffNet-HardNet **1788.7** 98.7 **95.7** 4.19 .4771 .5854 .5114 4 _[∗]_


**Table 8 Test – Multiview results with 2k features.** Same as Table 6.

_∗_ **The last group of results is obtained after the paper publication**
**and not described in the text of the paper. Their rank does not**
**influence other entries ranks.**


quite differently. Those based on DoG significantly benefit from an increased feature budget, whereas those learned
end-to-end may require re-training – this is exemplified
by the difference in performance between 2k and 8k for
Key.Net+Hardnet, specially on multiview, which is very narrow despite quadrupling the budget. Overall, learned detectors – KeyNet, SuperPoint, R2D2, LF-Net – show relatively
better results on multiview setup than on stereo. Our hypothesis is that they have good robustness, but low localization
precision, which is later corrected during bundle adjustment.


6.4 Outlier pre-filtering with deep networks — Table 9


Next, we study the performance of CNe [114] for outlier rejection, paired with PyRANSAC, DEGENSAC, and


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
|||||||~~2k~~|~~features~~||
|||||||8k|eatures||
||||||||||
||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||~~2~~|~~k featu~~|~~res~~||
|||||||||8|k featu|res||
|||||||||||||
|||||||||||||



0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
Mean Average Accuracy (mAA)


**Fig. 18 Test – Multiview performance: 2k vs 8k features.** Same as
Fig. 17, for multiview.


MAGSAC. Its training data does not use the ratio test, so
we omit it here too – note that because of this, it expects
a relatively large number of input matches. We thus evaluate it only for the 8k feature setting, while using the “both”
matching strategy.


Our experiments with SIFT, the local feature used to
train CNe, are encouraging: CNe aggressively filters out
about 80% of the matches in a single forward pass, boosting mAA at 10 _[◦]_ by 2-4% relative for stereo task and 8% for
multiview task. In fact, it is surprising that nearly all classical methods benefit from it, with gains of up to 20% relative.
By contrast, it damages performance with most learned descriptors, even those operating on DoG keypoints, and particularly for methods learned end-to-end, such as D2-Net
and R2D2. We hypothesize this might be because the models
performed better on the “classical” keypoints it was trained



STEREO: mAA(10 _[o]_ )



CV-SIFT


CV-RootSIFT

CV-SURF


CV-AKAZE

CV-ORB


CV-FREAK

L2-Net


DoG-HardNet


DoG-AffNet-HardNet

DoG-SOSNet

Key.Net-HardNet
Key.Net-SOSNet
GeoDesc


ContextDesc

LogPolarDesc

R2D2 (best model)
SuperPoint
LF-Net

D2-Net (SS)

D2-Net (MS)





0.0 0.1 0.2 0.3 0.4 0.5
Mean Average Accuracy (mAA)


**Fig. 17 Test – Stereo performance: 2k vs 8k features.** We compare
the results obtained with different methods using either 2k or 8k features – we use DEGENSAC, which performs better than other RANSAC variants under most circumstances. Dashed lines indicate SIFT’s

performance. For LF-Net and SuperPoint we do not include results
with 8k features, as we failed to obtain meaningful results. For R2D2,
we use the best model for each setting.


MULTIVIEW: mAA(10 _[o]_ )


CV-SIFT


CV-RootSIFT

CV-SURF


CV-AKAZE

CV-ORB


CV-FREAK

L2-Net


DoG-HardNet


DoG-AffNet-HardNet

DoG-SOSNet

Key.Net-HardNet
Key.Net-SOSNet
GeoDesc


ContextDesc

LogPolarDesc



R2D2 (best model)
SuperPoint
LF-Net

D2-Net (SS)

D2-Net (MS)




16 Yuhe Jin et al.



Stereo Task
PyRANSAC DEGENSAC MAGSAC Multi-view Task
Method mAA(10 _[o]_ ) _[↑]_ _∆_ (%) _[↑]_ mAA(10 _[o]_ ) _[↑]_ _∆_ (%) _[↑]_ mAA(10 _[o]_ ) _[↑]_ _∆_ (%) _[↑]_ mAA(10 _[o]_ ) _[↑]_ _∆_ (%) _[↑]_


CV-SIFT .4086 +2.24 .4751 +3.65 .4694 +2.42 .6815 +8.85


CV- ~~_√_~~ SIFT .4205 -0.53 .4927 -0.06 .4848 -1.87 **.6978** +3.16

CV-SURF .2490 +9.18 .3071 +18.42 .2954 +15.75 .5750 +18.67
CV-AKAZE .2857 +11.18 .3417 +11.18 .3316 +9.23 .6026 +8.51
CV-ORB .1323 +8.49 .1856 +10.87 .1748 +11.34 .4171 +18.88
CV-FREAK .2532 +11.36 .3204 +18.18 .3053 +14.93 .5574 +19.79


L2-Net .4377 -5.27 .5012 -5.35 .4937 -5.99 .6951 +4.62
DoG-HardNet .4427 -7.80 **.5156** -6.98 .5056 -8.11 **.7061** -.50
Key.Net-HardNet .3081 -22.92 .4226 -15.23 .4012 -15.36 .6620 +2.11
GeoDesc .4239 -2.05 .4924 -3.67 .4807 -4.93 .6956 +2.25
ContextDesc .3976 -15.11 .4482 -12.09 .4535 -11.83 .6900 -1.91
DoG-SOSNet .4439 -7.21 **.5187** -7.15 **.5073** -8.04 **.7103** +1.18
LogPolarDesc .4259 -6.89 .4898 -8.27 .4808 -8.22 .6871 -.82


D2-Net (SS) .1231 -36.32 .1717 -22.95 .1608 -20.86 .4639 +0.89
D2-Net (MS) .0998 -53.78 .1370 -45.33 .1316 -43.29 .4132 -13.02
R2D2 (wasf-n8-big) .2218 -39.78 .3141 -29.21 .3032 -28.43 .6229 -8.83


**Table 9** **Test – Outlier pre-filtering with CNe (8k features).** We
report mAP at 10 _[o]_ with CNe, on stereo and multi-view, and its increase
in performance w.r.t. Table 6 – positive _∆_ meaning CNe helps. When
using CNe, we disable the ratio test.


CV- _√_ SIFT HardNet SOSNet LogPolarDesc

NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_


Standard 281.7 0.4930 432.3 0.5543 424.6 0.5587 441.8 0.5340


Upright 270.0 0.4878 449.2 0.5542 432.9 0.5554 461.8 0.5409
_∆_ (%) -4.15 -1.05 +3.91 -0.02 +1.95 -0.59 +4.53 +1.29


Upright++ 358.9 0.5075 527.6 0.5728 508.4 0.5738 543.2 0.5510
_∆_ (%) +27.41 +2.94 +22.04 +3.34 +19.74 +2.70 +22.95 +3.18


**Table 10 Test – Stereo performance with upright descriptors (8k**
**features).** We report **(NI)** the number of inliers and **mAA** at 10 _[o]_ for
the stereo task, using DEGENSAC. As DoG may return multiple orientations for the same point [54] (up to 30%), we report: **(top)** with
orientation estimation; **(middle)** setting the orientation to zero while
removing duplicates; and **(bottom)** adding new points until hitting the
8k-feature budget.


CV- _√_ SIFT HardNet SOSNet LogPolarDesc

NL _[↑]_ mAA(10 _[o]_ ) _[↑]_ NL _[↑]_ mAA(10 _[o]_ ) _[↑]_ NL _[↑]_ mAA(10 _[o]_ ) _[↑]_ NL _[↑]_ mAA(10 _[o]_ ) _[↑]_


Standard 3312.1 0.6765 4001.4 0.7096 3796.0 0.7021 4054.6 0.6928


Upright 3485.1 0.6572 3594.6 0.6962 4025.1 0.7054 3737.4 0.6934
_∆_ (%) +5.22 -2.85 -10.17 -1.89 +6.04 +0.47 -7.82 +0.09


Upright++ 4404.6 0.6792 4250.4 0.7231 3988.6 0.7129 4414.1 0.7109
_∆_ (%) +32.99 +0.40 +6.22 +1.90 +5.07 +1.54 +8.87 +2.61


**Table 11 Test – Multiview performance with upright descriptors**
**(8k features).** Analogous to Table 10, on the multiview task. We report
the number of landmarks (NL) instead of the number of inliers (NI).


with – [97] reports that re-training them for a specific feature
helps.


6.5 On the effect of local feature orientation estimation —

Tables 10 and 11


In contrast with classical methods, which estimate the orientation of each keypoint, modern, end-to-end pipelines [34,
37, 80] often skip this step, assuming that the images are
roughly aligned (upright), with the descriptor shouldering
the increased invariance requirements. As our images meet
this condition, we experiment with setting the orientation of
keypoints to a fixed value (zero). DoG often returns multi


Exact 281.7 0.4930 432.0 0.5532 424.3 0.5575 470.6 0.2506


FLANN 274.6 0.4879 363.3 0.5222 339.8 0.5179 338.9 0.2046

_∆_ (%) -2.52 -1.03 -15.90 -5.60 -19.92 -7.10 -27.99 -18.36


**Table 12 Test – Stereo performance with OpenCV FLANN, using**
**kd-tree approximate nearest neighbors (8k features).** kd-tree parameters are: 4 trees, 128 checks. We report **(NI)** the number of inliers
and **mAA** at 10 _[o]_ for the stereo task, using DEGENSAC.


ple orientations for the same keypoint, so we consider two
variants: one where we simply remove keypoints which become duplicates after setting the orientation to a constant
value (Upright), and a second one where we fill out the budget with new keypoints (Upright++). We list the results in
Table 10, for stereo, and Table 11, for multiview. Performance increases across the board with Upright++ – albeit
by a small margin.


6.6 On the effect of approximate nearest neighbor
matching — Table 12


While it is known that approximate nearest neighbor search
algorithms have non-perfect recall [65], it is not clear how
their usage influences downstream performance. We thus
compare exact (brute-force) nearest neighbor search with
a popular choice for approximate nearest neighbor search,
FLANN [65], as implemented in OpenCV. We experimented
with different parameters and found that 4 trees and 128
checks provide a reasonable trade-off between precision and
runtime. We report results with and without FLANN in Table 12. The performance drop varies for different methods:
from a moderate 1% for RootSIFT, to 5-7% HardNet and

SOSNet, and 18% for D2-Net.


6.7 Pose mAA vs. traditional metrics — Fig. 19


To examine the relationship between our pose metric and
traditional metrics, we compare mAA against repeatability
and matching score on the stereo task, with DEGENSAC.
While the matching score seems to correlate with mAA, repeatability is harder to interpret. However, note that even for
the matching score, which shows correlation, higher value
does not guarantee high mAA – see _e.g._ RootSIFT vs ContextDesc. We remind the reader that, as explained in Section 4, our implementation differs from the classical formulation, as many methods do not have a strict notion of a support region. We compute these metrics at a 3-pixel threshold,
and provide more granular results in Section 7.
As shown, all methods based on DoG are clustered, as
they operate on the same keypoints. Key.Net obtains the best
repeatability, but performs worse than DoG in terms of mAA



CV- _√_ SIFT HardNet SOSNet D2Net

NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_ NI _[↑]_ mAA(10 _[o]_ ) _[↑]_


Image Matching Across Wide Baselines: From Paper to Practice 17



mAA(10 _[o]_ ) vs Matching Score

0.7


0.6


0.5


0.4


0.3


0.2


0.4 0.6 0.8
Matching Score



mAA(10 _[o]_ ) vs Repeatability


0.3 0.4 0.5
Repeatability



0.6


0.5


0.4


0.3


0.2


CV-SIFT

CV-RootSIFT

CV-SURF

CV-AKAZE



STEREO: mAA(10 _[o]_ )


0.0+ 0.1+ 0.2+ 0.3+ 0.4+ 0.5+ 0.6+
Co-visibility threshold



CV-ORB

CV-FREAK

L2-Net

DoG-HardNet













**Fig. 19 Test – Downstream vs. traditional metrics (8k features).**
We cross-reference stereo mAA at 10 _[◦]_ with repeatability and matching
score, with a 3-pixel threshold.



DoG-SOSNet

Key.Net-HardNet

Key.Net-SOSNet

GeoDesc



ContextDesc

LogPolarDesc
R2D2 (wasf-n8-big)
SuperPoint (2k points)



LF-Net (2k points)

D2-Net (SS)

D2-Net (MS)



0.8


0.6


0.4


0.2



STEREO, per sequence: mAA(10)



0.0 BM FCS LMS LB MC MR PSM SF SPC Average

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


MULTIVIEW, per sequence: mAA(10)


0.8


0.6


0.4


0.2


0.0 BM FCS LMS LB MC MR PSM SF SPC Average


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||











**Fig. 20** **Test – Breakdown by scene (8k features).** For the stereo
task, we use DEGENSAC. Note how performance can vary drastically
between scenes, and the relative rank of a given local feature fluctuates
as well. Please refer to Table 1 for a legend.


with the same descriptors (HardNet). AKAZE and FREAK
perform surprisingly well in terms of repeatability – #2 and
#3, respectively – but obtain a low mAA, which may be
related to their descriptors, which are binary. R2D2 shows
good repeatability but a poor matching score and is outperformed by DoG-based features.


6.8 Breakdown by scene — Fig. 20


Results may vary drastically from scene to scene, as shown
in Fig. 20. A given method may also perform better on some
than others – for instance, D2-Net nears the state of the

art on “Lincoln Memorial Statue”, but is 5x times worse

on “British Museum”. AKAZE and ORB show similar be
haviour. This can provide valuable insights on limitations

and failure cases.



**Fig. 21 Test – Local features vs. co-visibility (8k features).** We plot
mAA at 10 _[◦]_ on the stereo task – using DEGENSAC – at different covisibility thresholds for different local feature types. Bin “0+” consists
of all possible image pairs, including potentially unmatchable ones.
Bin “0.1+” includes pairs with a minimum co-visibility value of 0.1 –
this is the default value we use for all other experiments in this paper
– and so forth. Results are mostly consistent, with end-to-end methods
performing better at higher than lower co-visibility.


STEREO: mAA(10 _[o]_ )


0.6


0.5


0.4


0.3

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||



0.0+ 0.1+ 0.2+ 0.3+ 0.4+ 0.5+ 0.6+
Co-visibility threshold



**Fig. 22 Test – RANSAC vs. co-visibility (8k features).** We plot mAA
at 10 _[◦]_ on the stereo task at different co-visibility thresholds for different RANSAC variants, binning the results as in Fig. 21. We pair them
with different local features methods. The difference in performance
between RANSAC variants seems consistent across pairs, regardless
of their difficulty.


6.9 Breakdown by co-visibility — Figs. 21 and 22


Next, in Fig. 21 we evaluate stereo performance at different
co-visibility thresholds, for several local feature methods,
using DEGENSAC. Bins are encoded as _v_ +, with _v_ the covisibility threshold, and include all image pairs with a covisibility value larger or equal than _v_ . This means that the
first bin may contain unmatchable images – we use 0.1+ for



CV-SIFT-PYRANSAC

CV-AKAZE-PYRANSAC

DoG-HardNet-PYRANSAC

R2D2 (wasf-n8-big)-PYRANSAC



CV-SIFT-MAGSAC

CV-AKAZE-MAGSAC

DoG-HardNet-MAGSAC

R2D2 (wasf-n8-big)-MAGSAC



CV-SIFT-DEGENSAC

CV-AKAZE-DEGENSAC

DoG-HardNet-DEGENSAC

R2D2 (wasf-n8-big)-DEGENSAC


18 Yuhe Jin et al.



MULTIVIEW: AA(10)


1 _[o]_ 2 _[o]_ 3 _[o]_ 4 _[o]_ 5 _[o]_ 6 _[o]_ 7 _[o]_ 8 _[o]_ 9 _[o]_ 10 _[o]_
Angular threshold



1.0 Matching Score


0.8


0.6


0.4


0.2



1.0 Repeatability


0.8


0.6


0.4


0.2



STEREO: AA(10)


1 _[o]_ 2 _[o]_ 3 _[o]_ 4 _[o]_ 5 _[o]_ 6 _[o]_ 7 _[o]_ 8 _[o]_ 9 _[o]_ 10 _[o]_
Angular threshold



0.0


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||



1 2 3 4 5 6 7 8 9 10

Pixel threshold



1 2 3 4 5 6 7 8 9 10

Pixel threshold



0.0



0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0



CV-SIFT

CV-RootSIFT

CV-SURF

CV-AKAZE



CV-SIFT

CV-RootSIFT

CV-SURF

CV-AKAZE



CV-ORB

CV-FREAK

L2-Net

DoG-HardNet



DoG-SOSNet

Key.Net-HardNet

Key.Net-SOSNet

GeoDesc



ContextDesc

LogPolarDesc
R2D2 (wasf-n8-big)
SuperPoint (2k points)



LF-Net (2k points)

D2-Net (SS)

D2-Net (MS)



CV-ORB

CV-FREAK

L2-Net

DoG-HardNet



DoG-SOSNet

Key.Net-HardNet

Key.Net-SOSNet

GeoDesc



ContextDesc

LogPolarDesc
R2D2 (wasf-n8-big)
SuperPoint (2k points)



LF-Net (2k points)

D2-Net (SS)

D2-Net (MS)



**Fig. 23 Test – Classical metrics, by pixel threshold (8k features).**
Repeatability and matching score computed at different pixel thresholds are shown.


all other experiments in the paper. We do not report values
above 0.6+ as there are only a handful of them and thus the
results are very noisy.

Performance for all local features and RANSAC vari
ants increases with the co-visibility threshold, as expected.
Results are consistent, with end-to-end methods performing
better at higher than co-visibility than lower, and singlescale D2-Net outperforming its multi-scale counterpart at
0.4+ and above, where the images are more likely aligned

in terms of scale.

We also break down different RANSAC methods in

Fig. 22, along with different local fetures, including AKAZE
(binary), SIFT (also handcrafted), HardNet (learned descriptor on DoG points) and R2D2 (learned end-to-end). We do
not observe significant variations in the trend of each curve
as we swap RANSAC and local feature methods. DEGENSAC and MAGSAC show very similar performance.


6.10 Classical metrics vs. pixel threshold — Fig. 23


In Fig. 19 we plot repeatability and matching score against
mAA at a fixed error threshold of 3 pixels. In Fig. 23 we
show them at different pixel thresholds. End-to-end methods
tend to perform better at higher pixel thresholds, which is expected – D2-Net in particular extracts keypoints from downsampled feature maps. These results are computed from the
depth maps estimated by COLMAP, which are not pixelperfect, so the results for very low thresholds are not completely trustworthy.
Note that repeatability is typically lower than matching
score, which might be counter-intuitive as the latter is more
strict – it requires two features to be nearest-neighbors in
descriptor space in addition to being physically close (after reprojection). We compute repeatability with the raw set
of keypoints, whereas the matching score is computed with
optimal matching settings – bidirectional matching with the
“both” strategy and the ratio test. This results in a much



**Fig. 24 Test – Breakdown by angular threshold, for local features**
**(8k features).** Accuracy in pose estimation is evaluated at every error
threshold. Mean Average Accuracy used in the rest of paper is the area
under this curve. Ranks are consistent across thresholds.


STEREO: AA w.r.t. angular threshold


0.7


0.6


0.5


0.4


0.3


0.2


0.1


1 _[o]_ 2 _[o]_ 3 _[o]_ 4 _[o]_ 5 _[o]_ 6 _[o]_ 7 _[o]_ 8 _[o]_ 9 _[o]_ 10 _[o]_
Angular threshold



CV-SIFT-DEGENSAC

CV-AKAZE-DEGENSAC

DoG-HardNet-DEGENSAC

R2D2 (wasf-n8-big)-DEGENSAC



CV-SIFT-MAGSAC

CV-AKAZE-MAGSAC

DoG-HardNet-MAGSAC

R2D2 (wasf-n8-big)-MAGSAC



CV-SIFT-PYRANSAC

CV-AKAZE-PYRANSAC

DoG-HardNet-PYRANSAC

R2D2 (wasf-n8-big)-PYRANSAC



**Fig. 25 Test – Breakdown by angular threshold, for RANSAC (8k**
**features).** We plot performance on the stereo task at different angular error thresholds, for four different local features and three RANSAC variants: MAGSAC (solid line), DEGENSAC (dashed line), and
PyRANSAC (dotted line). We observe a similar behaviour across all
thresholds.


smaller pool – 8k features are typically narrowed down to
200–400 matches (see Table 13). This better isolates the performance of the detector and the descriptor where it matters.


6.11 Breakdown by angular threshold — Figs. 24 and 25


We summarize pose accuracy by mAA at 10 _[◦]_ in order to
have a single, easy-to-interpret number. In this section we
show how performance varies across different error thresholds – we look at the _average accuracy_ at every angular error
threshold, rather than the mean Average Accuracy. Fig. 24
plots performance on stereo and multiview for different local feature types, showing that the ranks remain consistent
across thresholds. Fig. 25 shows how it affects different
RANSAC variants, with four different local features. Again,


Image Matching Across Wide Baselines: From Paper to Practice 19



ranks do not change. DEGENSAC and MAGSAC perfom
nearly identically for all features, except for R2D2. The consistency in the ranks demonstrate that summarizing the results with a single number, mAA at 10 _[◦]_, is reasonable.


6.12 Qualitative results — Figs. 28, 29, 30, and 31


Figs. 28 and 29 show qualitative results for the stereo task.
We draw the inliers produced by DEGENSAC and colorcode them using the ground truth depth maps, from green
(0) to yellow (5 pixels off) if they are correct, and in red
if they are incorrect (more than 5 pixels off). Matches including keypoints which fall on occluded pixels are drawn
in blue. Note that while the depth maps are somewhat noisy
and not pixel-accurate, they are sufficient for this purpose.
Notice how the best performing methods have more correct
matches that are well spread across the overlapping region.
Fig. 30 shows qualitative results for the multiview task,
for handcrafted detectors. We illustrate it by drawing keypoints used in the SfM reconstruction in blue and the rest in
red. It showcases the importance of the detector, specially
on unmatchable regions such as the sky. ORB and Hessian keypoints are too concentrated in the high-contrast regions, failing to provide evenly-distributed features. In contrast, SURF fails to filter-out background and sky points.
Fig. 31 shows results for learned methods: DoG+HardNet,
Key.Net+HardNet, SuperPoint, R2D2, and D2-Net (multiscale). They have different detection patterns: Key.Net resembles a “cleaned” version of DoG, while R2D2 seems to
be evenly distributed. D2Net looks rather noisy, while SuperPoint fires precisely on corner points on structured parts,
and sometimes form a regular grid on sky-like homogeneous
regions, which might be due to the method running out of locations to place points at – note however that the best results
were obtained with larger NMS (non-maxima suppression)

thresholds.


**7 Further results and considerations**


In this section we provide additional results on the validation
set. These include a study of the typical outlier ratios under optimal settings in Section 7.1, matching with FGINN
in Section 7.2, image-preprocessing techniques for feature
extraction in Section 7.3, and a breakdown of the optimal
settings in Section 7.4, provided to serve as a reference.


7.1 Number of inliers per step — Table 13


We list the number of input matches and their _resulting_ inliers for the stereo task, in Table 13. As before, we remind
the reader that these inliers are what each method reports,



Method # matches # inliers Ratio (%) mAA(10 _[◦]_ )


CV-SIFT 328.3 113.0 34.4 0.548


CV- ~~_√_~~ SIFT 331.6 131.4 39.6 0.584


CV-SURF 221.5 77.4 35.0 0.309

CV-AKAZE 369.1 143.5 38.9 0.360

CV-ORB 193.4 74.7 38.6 0.265

CV-FREAK 216.6 75.5 34.8 0.329


DoG-HardNet 433.1 173.1 40.0 0.627

Key.Net-HardNet 644.1 166.6 25.9 0.580
L2Net 368.0 144.0 39.1 0.601

GeoDesc 322.6 133.4 41.3 0.564

ContextDesc 560.8 165.9 29.6 0.587

SOSNet 391.7 161.8 41.3 0.621

LogPolarDesc 522.8 175.2 33.5 0.618


SuperPoint (2k points) 202.0 62.6 31.0 0.312
LF-Net (2k points) 165.9 73.2 44.1 0.293
D2-Net (SS) 657.3 148.9 22.7 0.263
D2-Net (MS) 601.7 161.7 26.9 0.343
R2D2 (wasf-n8-big) 901.5 477.3 52.9 0.473


**Table 13 Validation – Number of inliers with optimal settings.** We
use 8k features with optimal parameters, and PyRANSAC as a robust
estimator. The number of inliers varies significantly between methods,
despite tuning the matcher and the ratio test, and lower inlier ratios
tend to correlate with low performance.


_i.e._, the matches that are actually used to estimate the poses.
We list the number of input matches, the number of inliers
produced by each method (which may still contain outliers),
their ratio, and the mAA at 10 _[◦]_ . We use PyRANSAC with
optimal settings for each method, the ratio test, and bidirectional matching with the “both” strategy.

We see that inlier-to-outlier ratios hover around 35–40%

for all features relying on classical detectors. Key.Net with
HardNet descriptors sees a significant drop in inlier ratio and
mAA, when compared to its DoG counterpart. D2-Net similarly has inlier ratios around 25%. R2D2 has the largest inlier ratio by far at 53%, but is outperformed by many other
methods in terms of mAA, suggesting that many of these are
not actual inliers. In general, we observe that the methods
which produce a large number of matches, such as Key.Net
(600+), D2-Net (600+) or R2D2 (900+) are less accurate in
terms of pose estimation.


7.2 Feature matching with an advanced ratio test — Fig. 26


We also compare the benefits of applying firstgeometrically-inconsistent-neighbor-ratio (FGINN) [63] to
DoG/SIFT, DoG/HardNet and Key.Net/HardNet, against
Lowe’s standard ratio test [54]. FGINN performs the ratiotest with second-nearest neighbors that are “far enough”
from the tentative match (10 pixels in [63]). In other words,
it loosens the test to allow for nearby-thus-similar points.
We test it for 3 matching strategies: unidirectional (“uni”),


20 Yuhe Jin et al.


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||



0.60 0.70 0.80 0.90 None
Ratio test _r_ (”both”)


Key.Net-HardNet, RT
Key.Net-HardNet, FGINN-RT



0.6


0.5


0.4


0.3


0.2


0.1



_η_ PyR _η_ DEGEN _η_ GCR _η_ MAG _r_ stereo _r_ multiview


CV-SIFT 0.25 0.5 0.5 1.25 0.85 0.75



0.00.60 0.70 0.80 0.90 None

Ratio test _r_ (”uni”)


CV-SIFT, RT

CV-SIFT, FGINN-RT



STEREO: mAA(10 _[o]_ )


0.60 0.70 0.80 0.90 None
Ratio test _r_ (”either”)


DoG-HardNet, RT

DoG-HardNet, FGINN-RT



CV- ~~_√_~~ SIFT 0.25 0.5 0.5 1.25 0.85 0.85

CV-SURF 0.75 0.75 0.75 2 0.85 0.90

CV-AKAZE 0.25 0.75 0.75 1.5 0.85 0.90

CV-ORB 0.75 1 1.25 2 0.85 0.95

CV-FREAK 0.5 0.5 0.75 2 0.85 0.85


VL-DoG-SIFT 0.25 0.5 0.5 1.5 0.85 0.80

VL-DoGAff-SIFT 0.25 0.5 0.5 1.5 0.85 0.80

VL-Hess-SIFT 0.2 0.5 0.5 1.5 0.85 0.80

VL-HessAffNet-SIFT 0.25 0.5 0.5 1 0.85 0.80


CV-DoG/HardNet 0.25 0.5 0.5 1.5 0.90 0.80
KeyNet/Hardnet 0.5 0.75 0.75 2 0.95 0.85
KeyNet/SOSNet 0.25 0.75 0.75 1.5 0.95 0.95
CV-DoG/L2Net 0.2 0.5 0.5 1.5 0.90 0.80

CV-DoG/GeoDesc 0.2 0.5 0.75 1.5 0.90 0.85

ContextDesc 0.25 0.75 0.5 1 0.95 0.85

CV-DoG/SOSNet 0.25 0.5 0.75 1.5 0.90 0.80
CV-DoG/LogPolarDesc 0.2 0.5 0.5 1.5 0.90 0.80


D2-Net (SS) 1 2 2 7.5 — —
D2-Net (MS) 1 2 2 5 — —
R2D2 (wasf-n8-big) 0.75 1.25 1.25 2 — 0.95


CV-DoG/TFeat 0.25 0.5 – 1.25 0.85 0.80

CV-DoG/MKD-Concat 0.25 0.5 – 1.25 0.85 0.80

CV-DoGAffNet/HardNet 0.25 0.5 – 1.25 0.85 0.85



**Fig. 26 Validation – FGINN vs. ratio test (8k features).** We evaluate
the ratio test with FGINN [63] (dashed line), and the standard ratio
test (solid line). With FGINN, the valid range for _r_ – the ratio test
threshold – is significantly wider, but the best performance with the
“both” matching strategy is not significantly better than for the standard
ratio test.



STEREO: mAA [10] _[o]_



0.6


0.4


0.2


0.0



0.6


0.4


0.2


0.0



MULTIVIEW: mAA [10] _[o]_


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
|||||~~C~~<br>C<br>|~~LAHE:~~<br>LAHE: D<br>|~~one~~<br>et<br>|
|||||~~C~~<br>C|~~LAHE:~~<br>LAHE: B|~~esc~~<br>oth|



**Fig. 27** **Validation – Image pre-processing with CLAHE (8k fea-**
**tures).** We experiment with contrast normalization for keypoint and
descriptor extraction. Results are very similar with or without it.


“both” and “either”. We report the results in Fig. 26. As
shown, FGINN provides minor improvements over the
standard ratio test in case of unidirectional matching, and
not as much when “both” is used. It also behaves differently
compared to the standard strategy, in that performance at
stricter thresholds degrades less.


7.3 Image pre-processing — Fig. 27


Contrast normalization is key to invariance against illumination changes – local feature methods typically apply
some normalization strategy over small patches [54,62].
Therefore, we experiment with contrast-limited adaptive
histogram equalization (CLAHE) [73], as implemented in
OpenCV. We apply it prior to feature detection and/or description with SIFT and several learned descriptors, and display the results in Fig. 27. Performance decreases for all
learned methods, presumably because they are not trained
for it. Contrary to our initial expectations, SIFT does not
benefit much from it either: the only increase in performance
comes from applying it for descriptor extraction, at 2.5% relative for stereo task and 0.56% relative for multi-view. This

might be due to the small number of night-time images in
our data. It also falls in line with the observations in [35],



**Table 14 Optimal hyper-parameters with 8k features.** We summarize the optimal hyperparameters – the maximum number of RANSAC
iterations _η_ and the ratio test threshold _r_ – for each combination of
methods. The number of RANSAC iterations _Γ_ is set to 250k for PyRANSAC, 50k for DEGENSAC, and 10k for both GC-RANSAC and
MAGSAC (for the experiments on the validation set). We use bidirectional matching with the “both” strategy.


which show that SIFT descriptors are actually optimal under certain assumptions.


7.4 Optimal settings breakdown — Tables 14 and 15


For the sake of clarity, we summarize the optimal hyperparameter combinations from Figs. 9, 10 and 11 in Table 14
(for 8k features) and Table 15 (for 2k features). We set the
confidence value to _τ_ =0 _._ 999999 for all RANSAC variants.

Notice how it is better to have more features and a stricter

ratio test threshold to filter them out, than having fewer features from the beginning.


**8 Conclusions**


We introduce a comprehensive benchmark for local features
and robust estimation algorithms. The modular structure of
its pipeline allows to easily integrate, configure, and combine methods and heuristics. We demonstrate this by evaluating dozens of popular algorithms, from seminal works to
the cutting edge of machine learning research, and show that
classical solutions may still outperform the perceived state
of the art with proper settings.
The experiments carried out through the benchmark and
reported in this paper have already revealed unexpected,


Image Matching Across Wide Baselines: From Paper to Practice 21


(a) RootSIFT (b) Hessian-SIFT (c) SURF (d) AKAZE (e) ORB


**Fig. 28 Qualitative results for the stereo task – “Classical” features.** We plot the matches predicted by each local feature, with DEGENSAC.
Matches above a 5-pixel error threshold are drawn in **red**, and those below are color-coded by their error, from 0 ( **green** ) to 5 pixels ( **yellow** ).
Matches for which we do not have depth estimates are drawn in **blue** .


22 Yuhe Jin et al.


(a) DoG+HardNet (b) KeyNet+HardNet (c) R2D2 (d) D2-Net (MS) (e) SuperPoint (2k)


**Fig. 29 Qualitative results for the stereo task – Learned features.** Color-coded as in Fig. 28.


Image Matching Across Wide Baselines: From Paper to Practice 23


(a) RootSIFT (b) Hessian-SIFT (c) SURF (d) AKAZE (e) ORB


**Fig. 30 Qualitative results for the multi-view task – “Classical” features.** We show images and the keypoints detected on them for different
methods, with the points reconstructed by COLMAP in **blue**, and the ones that are not used in the 3D model in **red** – more blue points indicate
denser 3D models. These results correspond the multiview-task for a 25-image subset.



non-intuitive properties of various components of the SfM
pipeline, which will benefit SfM development, _e.g._, the need
to tune RANSAC to the particular feature detector _and_ descriptor and to select specific settings for a particular RANSAC variant. Other interesting facts have been uncovered
by our tests, such as that the optimal set-ups across different
tasks (stereo and multiview) may differ, or that methods that
perform better on proxy tasks, like patch retrieval or repeata


bility, may be inferior on the downstream task. Our work is
open-sourced and makes the basis of an open challenge for
image matching with sparse methods.


**References**


1. Aanaes, H., Dahl, A. L., & Steenstrup Pedersen, K. Interesting
Interest Points. _International Journal of Computer Vision_, 97:18–


24 Yuhe Jin et al.


(a) DoG+HardNet (b) KeyNet+HardNet (c) R2D2 (d) D2-Net (MS) (e) SuperPoint (2k)


**Fig. 31 Qualitative results for the multi-view task – Learned features.** Color-coded as in Fig. 30.



35, 2012. 3
2. Aanaes, H. & Kahl, F. Estimation of Deformable Structure and
Motion. In _Vision and Modelling of Dynamic Scenes Workshop_,
2002. 13

3. Agarwal, S., Snavely, N., Simon, I., Seitz, S., & Szeliski, R.
Building Rome in One Day. In _International Conference on_
_Computer Vision_, 2009. 2, 3
4. Alahi, A., Ortiz, R., & Vandergheynst, P. FREAK: Fast Retina
Keypoint. In _Conference on Computer Vision and Pattern Recog-_
_nition_, 2012. 8, 12
5. Alcantarilla, P. F., Nuevo, J., & Bartoli, A. Fast Explicit Diffusion
for Accelerated Features in Nonlinear Scale Spaces. In _British_



_Machine Vision Conference_, 2013. 2, 8, 12


6. Aldana-Iuit, J., Mishkin, D., Chum, O., & Matas, J. Saddle: Fast
and repeatable features with good coverage. _Image and Vision_
_Computing_, 2019. 12


7. Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., & Sivic,
J. NetVLAD: CNN Architecture for Weakly Supervised Place
Recognition. In _Conference on Computer Vision and Pattern_
_Recognition_, 2016. 2


8. Arandjelovic, R. & Zisserman, A. Three things everyone should
know to improve object retrieval. In _Conference on Computer_
_Vision and Pattern Recognition_, 2012. 7


Image Matching Across Wide Baselines: From Paper to Practice 25



_η_ PyR _η_ DEGEN _η_ MAG _r_ stereo _r_ multiview


CV-SIFT 0.75 0.5 2 0.90 0.90


CV- ~~_√_~~ SIFT 0.5 0.5 1.25 0.90 0.90

CV-SURF 0.25 1 3 0.90 0.95

CV-AKAZE 1 0.75 2 0.90 0.95

CV-ORB 1 1.25 3 0.90 0.90

CV-FREAK 0.75 0.75 2 0.9 0.95


CV-DoG/HardNet 0.5 0.5 1.5 0.95 0.90
KeyNet/HardNet 0.5 0.75 2.5 0.95 0.90
KeyNet/SOSNet 0.5 0.75 1.5 0.95 0.95
CV-DoG/L2Net 0.25 0.5 1.5 0.90 0.90

CV-DoG/GeoDesc 0.5 0.5 1.5 0.95 0.90

ContextDesc 0.75 0.75 2 0.95 0.95

CV-DoG/SOSNet 0.5 0.75 1.5 0.95 0.90
CV-DoG/LogPolarDesc 0.5 0.5 1.5 0.95 0.90


D2-Net (SS) 1.5 2 7.5 — —
D2-Net (MS) 1.5 2 10 — —
SuperPoint 0.75 1 3 0.95 0.90
LF-Net 1 1 4 0.95 0.95
R2D2 (wasf-n16) 1.5 1.25 2 — —


CV-DoGAffNet/HardNet 0.25 0.5 1.25 0.95 0.95


**Table 15** **Optimal hyper-parameters with 2k features.** Equivalent
to Table 14. We do not evaluate GC-RANSAC, as it is always outperformed by DEGENSAC and MAGSAC, but keep PyRANSAC as a
baseline RANSAC.


9. Badino, H., Huber, D., & Kanade, T. The CMU Visual
Localization Data Set. [http://3dvis.ri.cmu.edu/](http://3dvis.ri.cmu.edu/data-sets/localization)
[data-sets/localization, 2011. 3](http://3dvis.ri.cmu.edu/data-sets/localization)
10. Balntas, V. SILDa: A Multi-Task Dataset for Evaluating Visual
Localization. [https://research.scape.io/silda/,](https://research.scape.io/silda/)
2018. 3, 4
11. Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. HPatches:
A Benchmark and Evaluation of Handcrafted and Learned Lo
cal Descriptors. In _Conference on Computer Vision and Pattern_
_Recognition_, 2017. 3, 4, 12, 13
12. Balntas, V., Li, S., & Prisacariu, V. RelocNet: Continuous Metric
Learning Relocalisation using Neural Nets. In _European Confer-_
_ence on Computer Vision_, September 2018. 2
13. Balntas, V., Riba, E., Ponsa, D., & Mikolajczyk, K. Learning Local Feature Descriptors with Triplets and Shallow Convolutional
Neural Networks. In _British Machine Vision Conference_, 2016.
2, 8
14. Barath, D. & Matas, J. Graph-Cut RANSAC. In _Conference on_
_Computer Vision and Pattern Recognition_, June 2018. 3, 9
15. Barath, D., Matas, J., & Noskova, J. MAGSAC: marginalizing
sample consensus. In _Conference on Computer Vision and Pat-_
_tern Recognition_, 2019. 1, 3, 9
16. Barroso-Laguna, A., Riba, E., Ponsa, D., & Mikolajczyk, K.
Key.Net: Keypoint Detection by Handcrafted and Learned CNN
Filters. In _International Conference on Computer Vision_, 2019.
2, 8
17. Baumberg, A. Reliable Feature Matching Across Widely Separated Views. In _Conference on Computer Vision and Pattern_
_Recognition_, 2000. 8, 12, 13
18. Bay, H., Tuytelaars, T., & Van Gool, L. SURF: Speeded Up
Robust Features. In _European Conference on Computer Vision_,
2006. 2, 7
19. Beaudet, P. R. Rotationally invariant image operators. In _Pro-_
_ceedings of the 4th International Joint Conference on Pattern_
_Recognition_, pages 579–583, Kyoto, Japan, Nov. 1978. 8, 12
20. Bellavia, F. & Colombo, C. Is there anything new to say about
sift matching? _International Journal of Computer Vision_, pages
1–20, 2020. 8



21. Bian, J.-W., Wu, Y.-H., Zhao, J., Liu, Y., Zhang, L., Cheng, M.M., & Reid, I. An Evaluation of Feature Matchers for Fundamental Matrix Estimation. In _British Machine Vision Conference_,
2019. 4
22. Brachmann, E. & Rother, C. Neural- Guided RANSAC: Learning Where to Sample Model Hypotheses. In _International Con-_
_ference on Computer Vision_, 2019. 2
23. Bradski, G. The OpenCV Library. _Dr. Dobb’s Journal of Soft-_
_ware Tools_, 2000. 9
24. Brown, M., Hua, G., & Winder, S. Discriminative Learning of
Local Image Descriptors. _IEEE Transactions on Pattern Analysis_
_and Machine Intelligence_, 2011. 2, 13
25. Brown, M. & Lowe, D. Automatic Panoramic Image Stitching
Using Invariant Features. _International Journal of Computer Vi-_
_sion_, 74:59–73, 2007. 3
26. Bui, M., Baur, C., Navab, N., Ilic, S., & Albarqouni, S. Adversarial Networks for Camera Pose Regression and Refinement. In
_International Conference on Computer Vision_, Oct 2019. 2
27. Chum, O. & Matas, J. Matching with PROSAC - Progressive
Sample Consensus. In _Conference on Computer Vision and Pat-_
_tern Recognition_, June 2005. 3
28. Chum, O., Matas, J., & Kittler, J. Locally Optimized RANSAC.
In _Pattern Recognition_, 2003. 3
29. Chum, O., Matas, J., & Kittler, J. Locally optimized ransac. In
_Pattern Recognition_, 2003. 9
30. Chum, O., Werner, T., & Matas, J. Two-View Geometry Estimation Unaffected by a Dominant Plane. In _Conference on Com-_
_puter Vision and Pattern Recognition_, 2005. 3, 9
31. Cui, H., Gao, X., Shen, S., & Hu, Z. Hsfm: Hybrid structurefrom-motion. In _CVPR_, July 2017. 3
32. Dang, Z., Yi, K. M., Hu, Y., Wang, F., Fua, P., & Salzmann, M.
Eigendecomposition-Free Training of Deep Networks with Zero
Eigenvalue-Based Losses. In _European Conference on Computer_
_Vision_, 2018. 9
33. Detone, D., Malisiewicz, T., & Rabinovich, A. Toward Geometric Deep SLAM. _arXiv preprint arXiv:1707.07410_, 2017. 2
34. Detone, D., Malisiewicz, T., & Rabinovich, A. Superpoint: SelfSupervised Interest Point Detection and Description. _CVPR_
_Workshop on Deep Learning for Visual SLAM_, 2018. 2, 3, 6,
7, 8, 16
35. Dong, J., Karianakis, N., Davis, D., Hernandez, J., Balzer, J., &
Soatto, S. Multi-view feature engineering and learning. In _Con-_
_ference on Computer Vision and Pattern Recognition_, June 2015.
20
36. Dong, J. & Soatto, S. Domain-Size Pooling in Local Descriptors: DSP-SIFT. In _Conference on Computer Vision and Pattern_
_Recognition_, 2015. 13
37. Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii,
A., & Sattler, T. D2-Net: A Trainable CNN for Joint Detection
and Description of Local Features. In _Conference on Computer_
_Vision and Pattern Recognition_, 2019. 1, 3, 7, 8, 16
38. E. Riba, D. Mishkin, D. P. E. R. & Bradski, G. Kornia: an open
source differentiable computer vision library for pytorch. In _Win-_
_ter Conference on Applications of Computer Vision_, 2020. 8
39. Ebel, P., Mishchuk, A., Yi, K. M., Fua, P., & Trulls, E. Beyond
Cartesian Representations for Local Descriptors. In _International_
_Conference on Computer Vision_, 2019. 2, 7, 8, 13
40. Fischler, M. & Bolles, R. Random Sample Consensus: A
Paradigm for Model Fitting with Applications to Image Analysis
and Automated Cartography. _Communications ACM_, 24(6):381–
395, 1981. 2, 3, 9
41. Gay, P., Bansal, V., Rubino, C., & Bue, A. D. Probabilistic Structure from Motion with Objects (PSfMO). In _International Con-_
_ference on Computer Vision_, 2017. 3
42. Geiger, A., Lenz, P., & Urtasun, R. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In _Con-_
_ference on Computer Vision and Pattern Recognition_, 2012. 3
43. Hartley, R. In Defense of the Eight-Point Algorithm. _IEEE_
_Transactions on Pattern Analysis and Machine Intelligence_,
19(6):580–593, June 1997. 3


26 Yuhe Jin et al.



44. Hartley, R. & Zisserman, A. _Multiple View Geometry in Com-_
_puter Vision_ . Cambridge University Press, 2000. 2, 9
45. Hartley, R. I. Projective reconstruction and invariants from multiple images. _IEEE Transactions on Pattern Analysis and Machine_
_Intelligence_, 16(10):1036–1041, Oct 1994. 2, 3
46. He, K., Lu, Y., & Sclaroff, S. Local Descriptors Optimized for
Average Precision. In _Conference on Computer Vision and Pat-_
_tern Recognition_, 2018. 2
47. Heinly, J., Schoenberger, J., Dunn, E., & Frahm, J.-M. Reconstructing the World in Six Days. In _Conference on Computer_
_Vision and Pattern Recognition_, 2015. 2, 3, 4
48. Jacobs, N., Roman, N., & Pless, R. Consistent Temporal Variations in Many Outdoor Scenes. In _Conference on Computer_
_Vision and Pattern Recognition_, 2007. 3
49. Karel Lenc and Varun Gulshan and Andrea Vedaldi. VLBench
[marks. http://www.vlfeat.org/benchmarks/, 2011.](http://www.vlfeat.org/benchmarks/)
4
50. Kendall, A., Grimes, M., & Cipolla, R. Posenet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization. In

_International Conference on Computer Vision_, 2015. 2
51. Krishna Murthy, J., Iyer, G., & Paull, L. gradSLAM: Dense
SLAM meets Automatic Differentiation. _arXiv_, 2019. 2
52. Leutenegger, S., Chli, M., & Siegwart, R. Y. Brisk: Binary robust invariant scalable keypoints. In _International Conference on_
_Computer Vision_, 2011. 8
53. Li, Z. & Snavely, N. MegaDepth: Learning Single-View Depth
Prediction from Internet Photos. In _Conference on Computer_
_Vision and Pattern Recognition_, 2018. 4
54. Lowe, D. G. Distinctive Image Features from Scale-Invariant
Keypoints. _International Journal of Computer Vision_, 20(2):91–
110, November 2004. 1, 2, 6, 7, 8, 12, 16, 19, 20
55. Luo, Z., Shen, T., Zhou, L., Zhang, J., Yao, Y., Li, S., Fang, T.,
& Quan, L. ContextDesc: Local Descriptor Augmentation with
Cross-Modality Context. In _Conference on Computer Vision and_
_Pattern Recognition_, 2019. 2, 8
56. Luo, Z., Shen, T., Zhou, L., Zhu, S., Zhang, R., Yao, Y., Fang, T.,
& Quan, L. Geodesc: Learning Local Descriptors by Integrating
Geometry Constraints. In _European Conference on Computer_
_Vision_, 2018. 2, 8
57. Lynen, S., Zeisl, B., Aiger, D., Bosse, M., Hesch, J., Pollefeys,
M., Siegwart, R., & Sattler, T. Large-scale, real-time visualinertial localization revisited. _arXiv Preprint_, 2019. 2
58. Maddern, W., Pascoe, G., Linegar, C., & Newman, P. 1 year,
1000 km: The Oxford RobotCar dataset. _International Journal_

_of Robotics Research_, 36(1):3–15, 2017. 3
59. Matas, J., Chum, O., Urban, M., & Pajdla, T. Robust WideBaseline Stereo from Maximally Stable Extremal Regions. _Im-_
_age and Vision Computing_, 22(10):761–767, 2004. 8, 12
60. Mikolajczyk, K. & Schmid, C. A Performance Evaluation of
Local Descriptors. _IEEE Transactions on Pattern Analysis and_
_Machine Intelligence_, 27(10):1615–1630, 2004. 3, 12
61. Mikolajczyk, K., Schmid, C., & Zisserman, A. Human Detection
Based on a Probabilistic Assembly of Robust Part Detectors. In
_European Conference on Computer Vision_, 2004. 8, 12
62. Mishchuk, A., Mishkin, D., Radenovic, F., & Matas, J. Working Hard to Know Your Neighbor’s Margins: Local Descriptor
Learning Loss. In _Advances in Neural Information Processing_
_Systems_, 2017. 2, 6, 8, 13, 20
63. Mishkin, D., Matas, J., & Perdoch, M. MODS: Fast and robust
method for two-view matching. _Computer Vision and Image Un-_
_derstanding_, 2015. 12, 19, 20
64. Mishkin, D., Radenovic, F., & Matas, J. Repeatability is Not
Enough: Learning Affine Regions via Discriminability. In _Euro-_
_pean Conference on Computer Vision_, 2018. 8, 12, 13
65. Muja, M. & Lowe, D. G. Fast Approximate Nearest Neighbors
with Automatic Algorithm Configuration. In _International Con-_
_ference on Computer Vision_, 2009. 16
66. Mukundan, A., Tolias, G., Bursuc, A., J´egou, H., & Chum, O.
Understanding and improving kernel local descriptors. _IJCV_,
2018. 8



67. Mukundan, A., Tolias, G., & Chum, O. Explicit Spatial Encoding
for Deep Local Descriptors. In _Conference on Computer Vision_
_and Pattern Recognition_, 2019. 2
68. Mur-Artal, R., Montiel, J., & Tard´os, J. Orb-Slam: A Versatile
and Accurate Monocular Slam System. _IEEE Transactions on_
_Robotics_, 31(5):1147–1163, 2015. 2
69. Nister, D. An Efficient Solution to the Five-Point Relative Pose
Problem. In _Conference on Computer Vision and Pattern Recog-_
_nition_, June 2003. 3
70. Noh, H., Araujo, A., Sim, J., & nd Bohyung Han, T. W. LargeScale Image Retrieval with Attentive Deep Local Features. In
_International Conference on Computer Vision_, 2017. 2, 3
71. Ono, Y., Trulls, E., Fua, P., & Yi, K. M. LF-Net: Learning Local Features from Images. In _Advances in Neural Information_
_Processing Systems_, 2018. 3, 7, 8
72. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg,
V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. Scikit-learn: Machine learning in
Python. _Journal of Machine Learning Research_, 12:2825–2830,
2011. 9
73. Pizer, S. M., Amburn, E. P., Austin, J. D., Cromartie, R.,
Geselowitz, A., Greer, T., ter Haar Romeny, B., Zimmerman,
J. B., & Zuiderveld, K. Adaptive histogram equalization and
its variations. _Computer vision, graphics, and image processing_,
1987. 20
74. Pritchett, P. & Zisserman, A. Wide baseline stereo matching. In
_ICCV_, pages 754–760, 1998. 2
75. Pultar, M., Mishkin, D., & Matas, J. Leveraging Outdoor Webcams for Local Descriptor Learning. In _Computer Vision Winter_
_Workshop_, 2019. 3
76. Pultar, M., Mishkin, D., & Matas, J. Leveraging Outdoor Webcams for Local Descriptor Learning. In _Computer Vision Winter_
_Workshop_, 2019. 13
77. Qi, C., Su, H., Mo, K., & Guibas, L. Pointnet: Deep Learning on
Point Sets for 3D Classification and Segmentation. In _Conference_
_on Computer Vision and Pattern Recognition_, 2017. 9
78. Radenovic, F., Tolias, G., & Chum, O. CNN image retrieval
learns from BoW: Unsupervised fine-tuning with hard examples.
In _European Conference on Computer Vision_, 2016. 2
79. Ranftl, R. & Koltun, V. Deep Fundamental Matrix Estimation.
In _European Conference on Computer Vision_, 2018. 2, 3, 7, 9
80. Revaud, J., Weinzaepfel, P., De Souza, C., Pion, N., Csurka, G.,
Cabon, Y., & Humenberger, M. R2D2: Repeatable and Reliable
Detector and Descriptor. In _arXiv Preprint_, 2019. 6, 8, 16
81. Revaud, J., Weinzaepfel, P., de Souza, C. R., Pion, N., Csurka,
G., Cabon, Y., & Humenberger, M. R2D2: Repeatable and Reliable Detector and Descriptor. In _Advances in Neural Information_
_Processing Systems_, 2019. 3
82. Rosten, E., Porter, R., & Drummond, T. Faster and Better: A
Machine Learning Approach to Corner Detection. _IEEE Trans-_
_actions on Pattern Analysis and Machine Intelligence_, 32:105–
119, 2010. 10
83. Rublee, E., Rabaud, V., Konolidge, K., & Bradski, G. ORB: An
Efficient Alternative to SIFT or SURF. In _International Confer-_
_ence on Computer Vision_, 2011. 2, 7, 12
84. Sarlin, P., DeTone, D., Malisiewicz, T., & Rabinovich, A. Superglue: Learning feature matching with graph neural networks. In
_Conference on Computer Vision and Pattern Recognition_, 2020.
7
85. Sattler, T., Leibe, B., & Kobbelt, L. Improving Image-Based Localization by Active Correspondence Search. In _European Con-_
_ference on Computer Vision_, 2012. 2, 3, 12
86. Sattler, T., Maddern, W., Toft, C., Torii, A., Hammarstrand, L.,
Stenborg, E., Safari, D., Okutomi, M., Pollefeys, M., Sivic, J.,
Kahl, F., & Pajdla, T. Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. In _Conference on Computer_
_Vision and Pattern Recognition_, 2018. 2, 3


Image Matching Across Wide Baselines: From Paper to Practice 27



87. Sattler, T., Weyand, T., Leibe, B., & Kobbelt, L. Image Retrieval
for Image-Based Localization Revisited. In _British Machine Vi-_
_sion Conference_, 2012. 3
88. Sattler, T., Zhou, Q., Pollefeys, M., & Leal-Taixe, L. Understanding the Limitations of CNN-based Absolute Camera Pose Regression. In _Conference on Computer Vision and Pattern Recog-_
_nition_, 2019. 2
89. Savinov, N., Seki, A., Ladicky, L., Sattler, T., & Pollefeys, M.
Quad-Networks: Unsupervised Learning to Rank for Interest
Point Detection. _Conference on Computer Vision and Pattern_
_Recognition_, 2017. 2
90. Sch¨onberger, J. & Frahm, J. Structure-From-Motion Revisited.
In _Conference on Computer Vision and Pattern Recognition_,
2016. 2, 3, 4, 9, 12
91. Sch¨onberger, J., Hardmeier, H., Sattler, T., & Pollefeys, M. Comparative Evaluation of Hand-Crafted and Learned Local Features. In _Conference on Computer Vision and Pattern Recog-_
_nition_, 2017. 4
92. Sch¨onberger, J., Zheng, E., Pollefeys, M., & Frahm, J. Pixelwise
View Selection for Unstructured Multi-View Stereo. In _European_
_Conference on Computer Vision_, 2016. 4
93. Shi, Y., Zhu, J., Fang, Y., Lien, K., & Gu, J. Self-Supervised
Learning of Depth and Ego-motion with Differentiable Bundle
Adjustment. _arXiv Preprint_, 2019. 2
94. Simo-serra, E., Trulls, E., Ferraz, L., Kokkinos, I., Fua, P., &
Moreno-Noguer, F. Discriminative Learning of Deep Convolutional Feature Point Descriptors. In _International Conference on_
_Computer Vision_, 2015. 2
95. Strecha, C., Hansen, W., Van Gool, L., Fua, P., & Thoennessen,
U. On Benchmarking Camera Calibration and Multi-View Stereo
for High Resolution Imagery. In _Conference on Computer Vision_
_and Pattern Recognition_, 2008. 3, 4
96. Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers,
D. A Benchmark for the Evaluation of RGB-D SLAM Systems.
In _International Conference on Intelligent Robots and Systems_,
2012. 10
97. Sun, W., Jiang, W., Trulls, E., Tagliasacchi, A., & Yi, K. M.
ACNe: Attentive Context Normalization for Robust Permutation
Equivariant Learning. In _Conference on Computer Vision and_
_Pattern Recognition_, 2020. 2, 3, 9, 16
98. Taira, H., Okutomi, M., Sattler, T., Cimpoi, M., Pollefeys, M.,
Sivic, J., Pajdla, T., & Torii, A. InLoc: Indoor Visual Localization
with Dense Matching and View Synthesis. _IEEE Transactions on_
_Pattern Analysis and Machine Intelligence_, 2019. 12
99. Tang, C. & Tan, P. Ba-Net: Dense Bundle Adjustment Network.
In _International Conference on Learning Representations_, 2019.
2
100. Tateno, K., Tombari, F., Laina, I., & Navab, N. Cnn-slam: Realtime dense monocular slam with learned depth prediction. In
_CVPR_, July 2017. 2
101. Thomee, B., Shamma, D., Friedland, G., Elizalde, B., Ni, K.,
Poland, D., Borth, D., & Li, L. YFCC100M: the New Data in
Multimedia Research. In _Communications of the ACM_, 2016. 4
102. Tian, Y., Fan, B., & Wu, F. L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space. In _Conference on_
_Computer Vision and Pattern Recognition_, 2017. 2, 8
103. Tian, Y., Yu, X., Fan, B., Wu, F., Heijnen, H., & Balntas, V. SOSNet: Second Order Similarity Regularization for Local Descriptor Learning. In _Conference on Computer Vision and Pattern_
_Recognition_, 2019. 2, 8
104. Tolias, G., Avrithis, Y., & J´egou, H. Image Search with Selective
Match Kernels: Aggregation Across Single and Multiple Images.
_IJCV_, 116(3):247–261, Feb 2016. 2



105. Torr, P. & Zisserman, A. MLESAC: A New Robust Estimator
with Application to Estimating Image Geometry. _Computer Vi-_
_sion and Image Understanding_, 78:138–156, 2000. 3
106. Triggs, B., Mclauchlan, P., Hartley, R., & Fitzgibbon, A. Bundle
Adjustment – A Modern Synthesis. In _Vision Algorithms: Theory_
_and Practice_, pages 298–372, 2000. 2
107. Vedaldi, A. & Fulkerson, B. Vlfeat: An open and portable library
of computer vision algorithms. In _Proceedings of the 18th ACM_
_International Conference on Multimedia_, MM ’10, pages 1469–
1472, 2010. 8
108. Verdie, Y., Yi, K. M., Fua, P., & Lepetit, V. TILDE: A Temporally
Invariant Learned DEtector. In _Conference on Computer Vision_
_and Pattern Recognition_, 2015. 2, 3
109. Vijayanarasimhan, S., Ricco, S., Schmid, C., Sukthankar, R., &
Fragkiadaki, K. Sfm-Net: Learning of Structure and Motion from
Video. _arXiv Preprint_, 2017. 2
110. Wei, X., Zhang, Y., Gong, Y., & Zheng, N. Kernelized Subspace
Pooling for Deep Local Descriptors. In _Conference on Computer_
_Vision and Pattern Recognition_, 2018. 2
111. Wei, X., Zhang, Y., Li, Z., Fu, Y., & Xue, X. DeepSFM: Structure From Motion Via Deep Bundle Adjustment. In _European_
_Conference on Computer Vision_, 2020. 2
112. Wu, C. Towards Linear-Time Incremental Structure from Motion. In _3DV_, 2013. 3, 12
113. Yi, K. M., Trulls, E., Lepetit, V., & Fua, P. LIFT: Learned Invariant Feature Transform. In _European Conference on Computer_
_Vision_, 2016. 3
114. Yi, K. M., Trulls, E., Ono, Y., Lepetit, V., Salzmann, M., & Fua,
P. Learning to Find Good Correspondences. In _Conference on_
_Computer Vision and Pattern Recognition_, 2018. 2, 3, 4, 5, 7, 9,
15
115. Yoo, A. B., Jette, M. A., & Grondona, M. Slurm: Simple linux
utility for resource management. In _Workshop on Job Scheduling_
_Strategies for Parallel Processing_, pages 44–60. Springer, 2003.
10
116. Zagoruyko, S. & Komodakis, N. Learning to Compare Image
Patches via Convolutional Neural Networks. In _Conference on_
_Computer Vision and Pattern Recognition_, 2015. 13
117. Zhang, J., Sun, D., Luo, Z., Yao, A., Zhou, L., Shen, T., Chen,
Y., Quan, L., & Liao, H. Learning Two-View Correspondences
and Geometry Using Order-Aware Network. _International Con-_
_ference on Computer Vision_, 2019. 2, 3, 5, 9
118. Zhang, J., Sun, D., Luo, Z., Yao, A., Zhou, L., Shen, T., Chen,
Y., Quan, L., & Liao, H. Learning two-view correspondences
and geometry using order-aware network. In _ICCV_, 2019. 7
119. Zhang, X., Yu, F. X., Karaman, S., & Chang, S.-F. Learning Discriminative and Transformation Covariant Local Feature Detec
tors. In _Conference on Computer Vision and Pattern Recognition_,
July 2017. 2
120. Zhao, C., Cao, Z., Li, C., Li, X., & Yang, J. NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences. In _Con-_
_ference on Computer Vision and Pattern Recognition_, 2019. 3,
9
121. Zhou, Q., Sattler, T., Pollefeys, M., & Leal-Taixe, L. To learn or
not to learn: Visual localization from essential matrices. In _ICRA_,

2020. 2
122. Zhu, S., Zhang, R., Zhou, L., Shen, T., Fang, T., Tan, P., & Quan,
L. Very Large-Scale Global SfM by Distributed Motion Averaging. In _Conference on Computer Vision and Pattern Recognition_,
June 2018. 2, 3
123. Zitnick, C. & Ramnath, K. Edge Foci Interest Points. In _Inter-_
_national Conference on Computer Vision_, 2011. 3


