## **Argoverse 2: Next Generation Datasets for** **Self-Driving Perception and Forecasting**

**Benjamin Wilson** _[∗†][,]_ [1], **William Qi** _[∗†]_, **Tanmay Agarwal** _[∗†]_, **John Lambert** _[†]_, **Jagjeet Singh** _[†]_,


**Siddhesh Khandelwal** [2], **Bowen Pan** _[†][,]_ [3], **Ratnesh Kumar** _[†]_, **Andrew Hartnett** _[†]_,


**Jhony Kaesemodel Pontes** _[†]_, **Deva Ramanan** _[†][,]_ [4], **Peter Carr** _[†]_, **James Hays** _[†][,]_ [1]


1 Georgia Tech, 2 UBC, 3 MIT, 4 CMU


**Abstract**


We introduce Argoverse 2 (AV2) — a collection of three datasets for perception and
forecasting research in the self-driving domain. The annotated _Sensor Dataset_ contains 1,000 sequences of multimodal data, encompassing high-resolution imagery
from seven ring cameras, and two stereo cameras in addition to lidar point clouds,
and 6-DOF map-aligned pose. Sequences contain 3D cuboid annotations for 26
object categories, all of which are sufficiently-sampled to support training and
evaluation of 3D perception models. The _Lidar Dataset_ contains 20,000 sequences
of unlabeled lidar point clouds and map-aligned pose. This dataset is the largest
ever collection of lidar sensor data and supports self-supervised learning and the
emerging task of point cloud forecasting. Finally, the _Motion Forecasting Dataset_
contains 250,000 scenarios mined for interesting and challenging interactions between the autonomous vehicle and other actors in each local scene. Models are
tasked with the prediction of future motion for “scored actors" in each scenario
and are provided with track histories that capture object location, heading, velocity,
and category. In all three datasets, each scenario contains its own _HD Map_ with 3D
lane and crosswalk geometry — sourced from data captured in six distinct cities.
We believe these datasets will support new and existing machine learning research
problems in ways that existing datasets do not. All datasets are released under the
CC BY-NC-SA 4.0 license.


**1** **Introduction**


In order to achieve the goal of safe, reliable autonomous driving, a litany of machine learning tasks
must be addressed, from stereo depth estimation to motion forecasting to 3D object detection. In recent
years, numerous high quality self-driving datasets have been released to support research into these and
other important machine learning tasks. Many datasets are annotated “sensor” datasets [ 4, 45, 39, 40,
24, 33, 18, 14, 41, 36 ] in the spirit of the influential KITTI dataset [ 17 ]. The Argoverse 3D Tracking
dataset [ 6 ] was the first such dataset with “HD maps” — maps containing lane-level geometry. Also
influential are self-driving “motion prediction” datasets [ 12, 22, 34, 4, 52 ] — containing abstracted
object tracks instead of raw sensor data — of which the Argoverse Motion Forecasting dataset [ 6 ]
was the first.


In the last two years, the Argoverse team has hosted six competitions on 3D tracking, stereo depth
estimation, and motion forecasting. We maintain evaluation servers and leaderboards for these tasks,


*Equal contribution.

_†_ Work completed while at Argo AI.


35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks.


as well as 3D detection. The leaderboards collectively contain thousands of submissions from four
hundred teams [1] . We also maintain the Argoverse API and have addressed more than one hundred
issues [2] . From these experiences we have formed the following guiding principles to guide the creation
of the next iteration of Argoverse datasets.


1. **Bigger isn’t always better.** Self-driving vehicles capture a flood of sensor data which is logistically difficult to work with. Sensor datasets are several terabytes in size, even when compressed.
If standard benchmarks grow further, we risk alienating much of the academic community and
leaving progress to well-resourced industry groups. _For this reason, we match but do not exceed_
_the scale of sensor data in nuScenes [4] and Waymo Open [45]_ .


2. **Make every instance count.** Much of driving is boring. Datasets should focus on the difficult,
interesting scenarios where current forecasting and perception systems struggle. _Therefore we_
_mine for especially crowded, dynamic, and kinematically unusual scenarios._


3. **Diversity matters.** Training on data from wintertime Detroit is not sufficient for detecting objects
in Miami — Miami has 15 times the frequency of motorcycles and mopeds. Behaviors differ
as well, so learned pedestrian motion behavior might not generalize. _Accordingly, each of our_
_datasets are drawn from six diverse cities — Austin, Detroit, Miami, Palo Alto, Pittsburgh, and_
_Washington D.C. — and different seasons, as well, from snowy to sunny_ .


4. **Map the world.** HD maps are powerful priors for perception and forecasting. Learning-based
methods that found clever ways to encode map information [ 31 ] performed well in Argoverse
competitions. _For this reason, we augment our HD map representation with 3D lane geometry,_
_paint markings, crosswalks, higher resolution ground height, and more_ .


5. **Self-supervise.** Other machine learning domains have seen enormous success from self-supervised
learning in recent years. Large-scale lidar data from dynamic scenes, paired with HD maps, could
lead to better representations than current supervised approaches. _For this reason, we build the_
_largest dataset of lidar sensor data._


6. **Fight the heavy tail.** Passenger vehicles are common, and thus we can assess our forecasting
and detection accuracy for cars. However, with existing datasets, we cannot assess forecasting
accuracy for buses and motorcycles with their distinct behaviors, nor can we evaluate stroller and
wheel chair detection. _Thus we introduce the largest taxonomy to date for sensor and forecasting_
_datasets, and we ensure enough samples of rare objects to train and evaluate models._


With these guidelines in mind we built the three Argoverse 2 (AV2) datasets. Below, we highlight
some of their contributions.


1. The 1,000 scenario _Sensor dataset_ has the largest self-driving taxonomy to date – 30 categories.
26 categories contain at least 6,000 cuboids to enable diverse taxonomy training and testing. The
dataset also has stereo imagery, unlike recent self-driving datasets.


2. The 20,000 scenario _Lidar dataset_ is the largest dataset for self-supervised learning on lidar. The
only similar dataset, concurrently developed ONCE [36], does not have HD maps.


3. The 250,000 scenario _Motion Forecasting Dataset_ has the largest taxonomy – 5 types of dynamic
actors and 5 types of static actors – and covers the largest mapped area of any such dataset.


We believe these datasets will support research into problems such as 3D detection, 3D tracking,
monocular and stereo depth estimation, motion forecasting, visual odometry, pose estimation, lane
detection, map automation, self-supervised learning, structure from motion, scene flow, optical flow,
time to contact estimation, and point cloud forecasting.


**2** **Related Work**

The last few years have seen rapid progress in self-driving perception and forecasting research,
catalyzed by many high quality datasets.


**Sensor datasets and 3D Object Detection and Tracking.** New sensor datasets for 3D object
detection [ 4, 45, 39, 40, 24, 33, 18, 14, 41, 36 ] have led to influential detection methods such as


1 This count includes private submissions not posted to the public leaderboards.
2 `[https://github.com/argoverse/argoverse-api](https://github.com/argoverse/argoverse-api)`


2


anchor-based approaches like PointPillars [ 27 ], and more recent anchor-free approaches such as
AFDet [ 16 ] and CenterPoint [ 51 ]. These methods have led to dramatic accuracy improvements on all
datasets. In turn, these improvements have made isolation of object-specific point clouds possible,
which has proven invaluable for offboard detection and tracking [ 42 ], and for simulation [ 8 ], which
previously required human-annotated 3D bounding boxes [ 35 ]. New approaches explore alternate
point cloud representations, such as range images [ 5, 2, 46 ]. Streaming perception [ 29, 21 ] introduces
a paradigm to explore the tradeoff between accuracy and latency. A detailed comparison between the
AV2 _Sensor Dataset_ and recent 3D object detection datasets is provided in Table 1.


**Motion Forecasting.** For motion forecasting, the progress has been just as significant. A transition
to attention-based methods [ 28, 38, 37 ] has led to a variety of new vector-based representations for
map and trajectory data [ 15, 31 ]. New datasets have also paved the way for new algorithms, with
nuScenes [ 4 ], Lyft L5 [ 22 ], and the Waymo Open Motion Dataset [ 12 ] all releasing lane graphs
after they proved to be essential in Argoverse 1 [ 6 ]. Lyft also introduced traffic/speed control data,
while Waymo added crosswalk polygons, lane boundaries (with marking type), speed limits, and stop
signs to the map. More recently, Yandex has released the Shifts [ 34 ] dataset, which is the largest (by
scenario hours) collection of forecasting data available to date. Together, these datasets have enabled
exploration of multi-actor, long-range motion forecasting leveraging both static and dynamic maps.


Following upon the success of Argoverse 1.1, we position AV2 as a large-scale repository of highquality motion forecasting scenarios - with guarantees on data frequency (exactly 10 Hz) and diversity
(>2000 km of unique roadways covered across 6 cities). This is in contrast to nuScenes (reports data
at just 2 Hz) and Lyft (collected on a single 10 km segment of road), but is complementary to Waymo
Open Motion Dataset (employs a similar approach for scenario mining and data configuration).
Complementary datasets are essential for these safety critical problems as they provide opportunities
to evaluate generalization and explore transfer learning. To improve ease of use, we have also
designed AV2 to be widely accessible both in terms of data size and format — a detailed comparison
vs. other recent forecasting datasets is provided in Table 2.


**Broader Problems of Perception for Self-Driving.** Aside from the tasks of object detection and
motion forecasting, new, large-scale sensor datasets for self-driving present opportunities to explore
dozens of new problems for perception, especially those that can be potentially solved via selfsupervision. A number of new problems have been recently proposed; real-time 3D semantic
segmentation in video has received attention thanks to SemanticKITTI [ 1 ]. HD map automation

[ 54, 30 ] and HD map change detection [ 26 ] have received additional attention, along with 3D
scene flow and pixel-level scene simulation [ 50, 8 ]. Datasets exist with unique modalities such as
thermal imagery [ 10, 9 ]. Our new _Lidar Dataset_ enables large-scale self-supervised training of new
approaches for freespace forecasting [23] or point cloud forecasting [48, 49].


**3** **The Argoverse 2 Datasets**


**3.1** **Sensor Dataset**


The _Argoverse 2 Sensor Dataset_ is the successor to the _Argoverse 1 3D Tracking Dataset_ . AV2 is
larger, with 1,000 scenes, up from 113 in Argoverse 1, but each AV2 scene is also richer – there
are 23x as many non-vehicle, non-pedestrian cuboids in AV2. The constituent 30 s scenarios in
the Argoverse 2 Sensor Dataset were manually selected by the authors to contain crowded scenes
with under-represented objects, noteworthy weather, and interesting behaviors, e.g., cut ins and
jaywalking. Each scenario is fifteen seconds in duration. Table 1 compares the AV2 Sensor Dataset
with a selection of self-driving datasets. Figures 1, 2, and 3 plot how the scenarios of AV2 compare
favorably to other datasets in terms of annotation range, object diversity, object density, and scene
dynamism.


The most similar sensor dataset to ours is the highly influential nuScenes [ 4 ] – both datasets have
1,000 scenarios and HD maps, although Argoverse is unique in having ground height maps. nuScenes
contains radar data while AV2 contains stereo imagery. nuScenes has a large taxonomy – twenty-three
object categories of which ten have suitable data for training and evaluation. Our dataset contains
thirty object categories of which twenty-six are well sampled enough for training and evaluation.
nuScenes spans two cities, while our proposed dataset spans six.


3


Table 1: Comparison of the Argoverse 2 _Sensor_ and _Lidar_ datasets with other sensor datasets.


Name # Scenes Cities Lidar? # Cameras Stereo HD Maps? # Classes # Evaluated Classes


Argoverse 1 [6] 113 2 ✓ 7 ✓ ✓ 15 3

KITTI [17] 22 1 ✓ 2 ✓ 3 3

nuScenes [4] 1,000 2 ✓ 6 ✓ 23 10

ONCE [36] 581 – ✓ 7 5 3

Waymo Open [45] 1,150 3 ✓ 5 4 4


Argoverse 2 Sensor 1,000 6 ✓ 9 ✓ ✓ 30 26
Argoverse 2 Lidar 20,000 6 ✓ - ✓ - 


10M Argoverse 1
5

1M

2

2

5

2
1000
5

2
100











in Argoverse 2.


°



250k





200k


150k


100k


50k









0

|Argoverse 2|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
|||||||||||~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|~~nuScenes~~|
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|~~Waymo Open~~|
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
|ar vehicle<br>Pedestrian<br>Bollard<br>Co<br>Numbe<br>_Dataset_<br>o 10 Hz<br>se 2.<br>**ite.**<br>L<br> to prov<br>o-vehic<br>am lida<br>as trigg<br>e synch<br> of view<br>rdinate<br>**chroniz**<br>ly over<br>avorab<br>**ns.** The<br>xonom|nstruction<br>Const<br>r of<br>, nu<br> for<br>idar<br>ide<br>le po<br>rs, s<br>er in<br>roni<br>. In<br> fram<br>**atio**<br> Arg<br>ly to<br> AV<br>y (F|cone<br>ruction<br>Stop s<br>ann<br>Sce<br> Ar<br> sw<br>a f<br>se<br>pin<br>-sy<br>zed<br>the<br>es<br>**n a**<br>ov<br> th<br>2 S<br>igu|barrel<br>ign<br>Bicycl<br><br>ota<br>ne<br>gov<br>ee<br>ully<br> in<br>nin<br>nc<br> to<br> A<br>.<br>**cc**<br>ers<br>e W<br>ens<br>re|e<br>Large<br>te<br>s,<br>er<br>ps<br> p<br>a g<br>g<br>wi<br> th<br>ppe<br>**ur**<br>e 1<br>ay<br>or<br>1).|vehicle<br>Wheel<br><br>d 3<br>ON<br>se,<br>are<br>ano<br>lob<br>at 1<br>th b<br>e li<br>nd<br>**acy**<br>. O<br>mo<br> Da<br> Cu<br>|ed devi<br>Bus<br>D<br>C<br>but<br> c<br>ra<br>al<br>0<br>ot<br>da<br>ix,<br>**.** I<br>ur<br> O<br>tas<br>bo<br>|ed devi<br>Bus<br>D<br>C<br>but<br> c<br>ra<br>al<br>0<br>ot<br>da<br>ix,<br>**.** I<br>ur<br> O<br>tas<br>bo<br>|ce<br>Box tr<br>cub<br>E,<br> th<br>olle<br>mic<br>co<br>Hz <br>h li<br>r to<br> we<br>n A<br> sy<br>pe<br>et<br>id<br>|uck<br>Sig<br>o<br>an<br>at<br>c<br> <br>or<br>in<br>d<br> h<br> <br>V<br>n<br>n<br>co<br>s<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|n<br><br>id<br>d<br> d<br>te<br>ﬁe<br>di<br> t<br>ar<br>a<br>pr<br>2<br>c<br>D<br>n<br>ha<br>|School bus<br>Wheeled rider<br>Strolle<br><br>rgoverse<br>e nuSce<br>he relativ<br>ith 20 fp<br>on, came<br>vided. L<br>but separ<br> frame-ra<br>ntered on<br>ﬁgure ill<br>nchroniz<br>y is with<br>rted as [<br>id annot<br>that are|r<br>Articu<br> 1<br>nes<br>e i<br>s i<br>ra<br>ida<br>ate<br>te.<br> th<br>ust<br>ati<br>in <br>_−_6<br>ati<br>co|r<br>Articu<br> 1<br>nes<br>e i<br>s i<br>ra<br>ida<br>ate<br>te.<br> th<br>ust<br>ati<br>in <br>_−_6<br>ati<br>co|lated bu<br>Messa<br><br>_ 3D_<br> an<br>ncr<br>ma<br> int<br>r r<br>d i<br> Th<br>e li<br>rati<br>on<br>[_−_<br>_,_ 7]<br>ons<br>nsis|s<br>ge boar<br>Mobile<br><br>_ Tr_<br>no<br>eas<br>ge<br>rin<br>etu<br>n o<br>e s<br>da<br>ng<br>of<br>1_._3<br> m<br> fo<br>ten|d trail<br> pede<br>Whe<br>_a_<br>ta<br>e i<br>ry<br>si<br>rn<br>ri<br>ev<br>r s<br>th<br>ca<br>9_,_ <br>s [<br>r o<br>t<br>|er<br>strian si<br>elchair<br>Railed<br>_ckin_<br>tion<br>n o<br> fro<br>cs,<br>s ar<br>ent<br>en<br>we<br>e ca<br>me<br>1_._3<br>45]<br>bje<br>ove<br>|gn<br> vehicl<br>Offici<br>_g_,<br> ra<br>bje<br>m<br>ext<br>e c<br>atio<br>glo<br>epi<br>r s<br>ras<br>9]<br>.<br>cts<br>r ti<br>|gn<br> vehicl<br>Offici<br>_g_,<br> ra<br>bje<br>m<br>ext<br>e c<br>atio<br>glo<br>epi<br>r s<br>ras<br>9]<br>.<br>cts<br>r ti<br>|e<br>al signa<br>Traffi<br>Ar<br>te<br>ct d<br>7 c<br>rin<br>apt<br>n<br>ba<br>ng<br>ens<br> an<br>ms<br>wi<br>me<br>|e<br>al signa<br>Traffi<br>Ar<br>te<br>ct d<br>7 c<br>rin<br>apt<br>n<br>ba<br>ng<br>ens<br> an<br>ms<br>wi<br>me<br>|
||||||~~W~~<br>|~~ay~~<br>||~~o O~~<br>|~~p~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|~~n~~<br>|||||||~~ay~~<br>||<br>|~~pe~~<br>|~~pe~~<br>|
||||||~~**A**~~<br>|~~**g**~~<br>||~~**ers**~~<br>|~~**e**~~|||||||||||||||~~**rg**~~<br>|~~**v**~~<br>|~~**r**~~<br>|~~**e 2**~~<br>|~~**e 2**~~<br>|
||||||~~n~~<br>|~~S~~<br>||~~nes~~<br>||||||||||||||||~~uSc~~<br>~~rgo~~|~~n~~<br>~~e~~||~~e 1~~|~~e 1~~|
||||||~~A~~<br>|~~g~~<br>||~~ers~~<br>|~~ 1~~|||||||||||||||~~N~~|~~E~~||||
||||||~~O~~||||||||||||||||||||||||

0 50 100 150 200 250


Range (m)



0 50 100 150 200 250 300


Number of 3D cuboids



Figure 2: **Left:** Number of annotated 3D cuboids by range in the Argoverse 2 _Sensor Dataset_ . About
14% of the Argoverse 2 cuboids are beyond 75 m – Waymo Open, nuScenes, and ONCE have less
than 1%. **Right:** Number of 3D cuboids per lidar frame. Argoverse 2 has an average of 75 3D
cuboids per lidar frame – Waymo Open has an average of 61, nuScenes 33, and ONCE 30.


4


80k


60k


40k


20k





60k


40k


20k



0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|W|ay|m|o|Op|en|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||
||||||||||||||**A**|**rg**|**ov**|**er**|**se**|**2**|
||||||||||||||||||||
||||||||||||||A|rg|ov|ers|e 1||
||||||||||||||||||||
||||||||||||||n|uS|ce|nes|||
||||||||||||||||||||
||||||||||||||O|N|CE||||
||||||||||||||||||||
||||||||||||||||||||

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19


Number of different categories



0

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||~~**Argov**~~<br>Waym|~~**rse 2**~~<br> Open|
|||Argove<br>|rse 1<br>|
|||~~nuScen~~|~~es~~|
|||||

5 10 15 20 25


Speed (m/s)



Figure 3: **Left:** Number of annotated categories per lidar frame in the Argoverse 2 _Sensor Dataset_ .
Per scene, Argoverse 2 is about 2 _×_ more diverse than Argoverse 1 and 2.3 _×_ more diverse than
Waymo Open. **Right:** Speed distribution for the vehicle category. We consider only moving vehicles
with speeds greater than 0.5 m/s. Argoverse 2 has about 1.3 _×_ more moving vehicles than Waymo
Open. About 28 % of the vehicles in Argoverse 2 are moving with an average speed of 7.27 m/s. We
did not compare against the ONCE dataset because it does not provide tracking information for the
3D cuboids.


same object instance. Objects are annotated if they are within the “region of interest” (ROI) – within
five meters of the mapped “driveable” area.


**Privacy.** All faces and license plates, whether inside vehicles or outside of the driveable area, are
blurred extensively to preserve privacy.


**Sensor Dataset splits** . We randomly partition the dataset with train, validation, and test splits of 700,
150, and 150 scenarios, respectively.


**3.2** **Lidar Dataset**


The _Argoverse 2 Lidar Dataset_ is intended to support research into self-supervised learning in the
lidar domain as well as point cloud forecasting [ 48, 49 ]. Because lidar data is more compact than the
full sensor suite, we can include double-length scenarios ( 30 s instead of 15 s ), and far more – 20,000
instead of 1,000 – equating to roughly 40x as many driving hours, for 5x the space budget. The AV2
Lidar Dataset is mined with the same criteria as the Forecasting Dataset (Section 3.3.2) to ensure that
each scene is interesting. While the Lidar Dataset does not have 3D object annotations, each scenario
carries an HD map with rich, 3D information about the scene.


Our dataset is the largest such collection to date with 20,000 thirty second sequences. The only
similar dataset, concurrently released ONCE [ 36 ], contains 1 M lidar frames compared to 6 M lidar
frames in ours. Our dataset is sampled at 10 Hz instead of 2 Hz, as in ONCE, making our dataset
more suitable for point cloud forecasting or self-supervision tasks where point cloud evolution over
time is important.


**Lidar Dataset splits** . We randomly partition the dataset with train, validation, and test splits of
16,000, 2,000, and 2,000 scenarios, respectively.


**3.3** **Motion Forecasting Dataset**


Motion forecasting addresses the problem of predicting future states (or occupancy maps) for dynamic
actors within a local environment. Some examples of relevant actors for autonomous driving include:
vehicles (both parked and moving), pedestrians, cyclists, scooters, and pets. Predicted futures
generated by a forecasting system are consumed as the primary inputs in motion planning, which
conditions trajectory selection on such forecasts. Generating these forecasts presents a complex,
multi-modal problem involving many diverse, partially-observed, and socially interacting agents.
However, by taking advantage of the ability to “self-label” data using observed ground truth futures,
motion forecasting becomes an ideal domain for application of machine learning.


Building upon the success of Argoverse 1, the Argoverse 2 Motion Forecasting dataset provides
an updated set of prediction scenarios collected from a self-driving fleet. The design decisions
enumerated below capture the collective lessons learned from both our internal research/development,


5


Table 2: Comparison between the Argoverse 2 Motion Forecasting dataset and other recent motion
forecasting datasets. Hyphens "-" indicate that attributes are either not applicable, or not available.
We define “mined for interestingness” to be true if interesting scenarios/actors are mined _after data_
_collection_, instead of taking all/random samples. _†_ Public leaderboard counts as retrieved on Aug. 27,
2021.


A RGOVERSE [6] I NTER [52] L YFT [22] W AYMO [12] N U S CENES [4] Y ANDEX [34] O URS


# S CENARIOS 324k            - 170k 104k 41k 600k 250k

# U NIQUE T RACKS 11.7M 40k 53.4M 7.6M         - 17.4M 13.9M

A VERAGE T RACK L ENGTH 2.48 s 19.8 s 1.8 s 7.04 s    -    - 5.16 s

T OTAL T IME 320 h 16.5 h 1118 h 574 h 5.5 h 1667 h 763 h

S CENARIO D URATION 5 s       - 25 s 9.1 s 8 s 10 s 11 s

T EST F ORECAST H ORIZON 3 s 3 s 5 s 8 s 6 s 5 s 6 s

S AMPLING R ATE 10 Hz 10 Hz 10 Hz 10 Hz 2 Hz 5 Hz 10 Hz

# C ITIES 2 6 1 6 2 6 6

U NIQUE R OADWAYS 290 km 2 km 10 km 1750 km      -      - 2220 km

A VG . # TRACKS PER SCENARIO 50   - 79   - 75 29 73

# E VALUATED OBJECT CATEGORIES 1 1 3 3 1 2 5

M ULTI   - AGENT EVALUATION _×_ ✓ ✓ ✓ _×_ ✓ ✓

M INED FOR I NTERESTINGNESS ✓ _×_  - ✓ _×_ _×_ ✓

V ECTOR M AP ✓ _×_ _×_ ✓ ✓ _×_ ✓

D OWNLOAD S IZE 4.8 GB       - 22 GB 1.4 TB 48 GB 120 GB 58 GB

# P UBLIC L EADERBOARD E NTRIES _[†]_ 194  - 935 23 18 3  

as well as feedback from more than 2,700 submissions by nearly 260 unique teams [3] across 3
competitions [43]:


1. **Motion forecasting is a safety critical system in a long-tailed domain.** Consequently, our
dataset is biased towards diverse and interesting scenarios containing different types of focal
agents (see section 3.3.2). Our goal is to encourage the development of methods that ensure safety
during tail events, rather than to optimize the expected performance on “easy miles”.


2. **There is a “Goldilocks zone” of task difficulty.** Performance on the Argoverse 1 test set has
begun to plateau, as shown in Figure 10 of the appendix. Argoverse 2 is designed to increase
prediction difficulty incrementally, spurring productive focused research for the next few years.
These changes are intended to incentivize methods that perform well on extended forecast horizons
(3 s _→_ 6 s), handle multiple types of dynamic objects (1 _→_ 5), and ensure safety in scenarios
from the long tail. Future Argoverse releases could continue to increase the problem difficulty by
reducing observation windows and increasing forecasting horizons.


3. **Usability matters.** Argoverse 1 benefited from a large and active research community—in large
part due to the simplicity of setup and usage. Consequently, we took care to ensure that existing
Argoverse models can be easily ported to run on Argoverse 2. In particular, we have prioritized
intuitive access to map elements, encouraging methods which use the lane graph as a strong prior.
To improve training and generalization, all poses have also been interpolated and resampled at
exactly 10 Hz (Argoverse 1 was approximate). The new dataset includes fewer, but longer and
more complex scenarios; this ensures that total dataset size remains large enough to train complex
models but small enough to be readily accessible.


**3.3.1** **Data Representation**


The dataset consists of 250,000 non-overlapping scenarios (80/10/10 train/val/test random splits)
mined from six unique urban driving environments in the United States. It contains a total of 10
object types, with 5 from each of the dynamic and static categories (see Figure 4). Each scenario
includes a local vector map and 11 s ( 10 Hz ) of trajectory data (2D position, velocity, and orientation)
for all tracks observed by the ego-vehicle in the local environment. The first 5 s of each scenario is
denoted as the _observed_ window, while the subsequent 6 s is denoted as the _forecasted_ horizon.


Within each scenario, we mark a single track as the “focal agent”. Focal tracks are guaranteed to
be fully observed throughout the duration of the scenario and have been specifically selected to
maximize interesting interactions with map features and other nearby actors (see Section 3.3.2). To
evaluate multi-agent forecasting, we also mark a subset of tracks as “scored actors” (as shown in
Figure 5), with guarantees for scenario relevance and minimum data quality.


3 This count includes private submissions not posted to the public leaderboards.


6


Figure 4: Object type and geographic histograms for the Motion Forecasting Dataset. **Left** : Histogram
of object types over the “focal” and “scored” categories. **Center** : Histogram of object types over all
tracks present in the dataset. The fine grained distinctions between different static object types (e.g.
_Construction Cone_ vs _Riderless Bicycle_ ) are unique among forecasting datasets. **Right** : Histogram of
metropolitan areas included in the dataset.


Figure 5: Visualization of a few interesting scenarios from the Motion Forecasting Dataset. The
scenarios demonstrate a mix of the various object types ( _Vehicle_, _Pedestrian_, _Bus_, _Cyclist_, or _Motor-_
_cyclist_ ). The ego-vehicle is indicated in green, the focal agent is purple, and scored actors are orange.
Other un-scored tracks are shown in blue. Object positions are captured at the last timestep of the
_observed_ history. For visualization purposes the full 5 s history and 6 s future are rendered for the
focal agent, while only 1 _._ 5 s of future are shown for the other scored actors. **Left** shows a pedestrian
crossing in front of the ego-vehicle, while **center** and **right** depict a motorcyclist weaving through
traffic.


**3.3.2** **Mining Interesting Scenarios**


The source data for Argoverse 2 was drawn from fleet logs tagged with annotations consistent
with interesting or difficult-to-forecast events. Each log was trimmed to 30 s and run through an
_interestingness_ scoring module in order to bias data selection towards examples from the long-tail of
the natural distribution. We employ heuristics to score each track in the scene across five dimensions:
object category, kinematics, map complexity, social context, and relation to the ego-vehicle (details
in Appendix).


The final scenarios are generated by extracting non-overlapping 11 s windows where at least one
candidate track is fully observed for the entire duration. The highest scoring candidate track is
denoted as the “focal agent”; all other fully observed tracks within 30 m of the ego-vehicle are
denoted as “scored actors”. The resulting dataset is diverse, challenging, and still right-sized for
widespread use (see the download size in Table 2). In Figure 6, we show that the resulting dataset is
significantly more interesting than Argoverse 1.1 and validate our intuition that actors scoring highly
in our heuristic module are more challenging to accurately forecast.


**3.4** **HD Maps**


Each scenario in the three datasets described above shares the same HD map representation. Each
scenario carries its own local map region, similar to the Waymo Open Motion [ 12 ] dataset. This
is a departure from the original Argoverse datasets in which all scenarios were localized onto two
city-scale maps—one for Pittsburgh and one for Miami. In the Appendix, we provide examples.


7


|Col1|Col2|Col3|Fitted Regres<br>Bin Centers (|sion Model<br>1000 Scena|rios Each)|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
|0<br>.0|0<br>.0|0<br>.0|0<br>.0|0<br>.0|0<br>.0|0<br>.0|


Figure 6: **Left:** Histogram comparing the distribution of interestingness scores assigned to focal
agents in both Argoverse 1.1 and 2. **Right:** Plot showing the relationship between total _interestingness_
_score_ and prediction difficulty on the Argoverse 2 test split. We evaluate WIMP [ 25 ] over each
scenario and fit a regression model to the computed miss rate (K=6, 2m threshold).


Advantages of per-scenario maps include more efficient queries and their ability to handle _map_
_changes_ . A particular intersection might be observed multiple times in our datasets, and there could
be changes to the lanes, crosswalks, or even ground height in that time.


**Lane graph.** The core feature of the HD map is the lane graph, consisting of a graph _G_ = ( _V, E_ ),
where _V_ are individual lane segments. In the Appendix, we enumerate and define the attributes we
provide for each lane segment. Unlike Argoverse 1, we provide the actual 3D lane boundaries, instead
of only centerlines. However, our API provides code to quickly infer the centerlines at any desired
sampling resolution. Polylines are quantized to 1 cm resolution. Our representation is richer than
nuScenes, which provides lane geometry only in 2D, not 3D.


**Driveable area.** Instead of providing driveable area segmentation in a rasterized format, as we did in
Argoverse 1, we release it in a vector format, i.e. as 3D polygons. This offers multiple advantages,
chiefly in compression, allowing us to store separate maps for tens of thousands of scenarios, yet the
raster format is still easily derivable. The polygon vertices are quantized to 1 cm resolution.


**Ground surface height.** Only the sensor dataset includes a dense ground surface height map
(although other datasets still have sparse 3D height information on polylines). Ground surface height
is provided for areas within a 5 m isocontour of the driveable area boundary, which we define as
the _region of interest_ (ROI) [ 6 ]. We do so because the notion of ground surface height is ill-defined
for the interior of buildings and interior of densely constructed city blocks, areas where ground
vehicles cannot observe due to occlusion. The raster grid is quantized to a 30 cm resolution, a higher
resolution than the 1 m resolution in Argoverse 1.


**Area of Local Maps.** Each scenario’s local map includes all entities found within a 100 m dilation
in _l_ 2 -norm from the ego-vehicle trajectory.


**4** **Experiments**


Argoverse 2 supports a variety of downstream tasks. In this section we highlight three different
learning problems: 3D object detection, point cloud forecasting, and motion forecasting — each
supported by the sensor, lidar, and motion forecasting datasets, respectively. First, we illustrate the
_challenging_ and _diverse_ taxonomy within the Argoverse 2 sensor dataset by training a state-of-theart 3D detection model on our twenty-six evaluation classes including “long-tail” classes such as
stroller, wheel chairs, and dogs. Second, we showcase the utility of the Argoverse 2 lidar dataset
through _large-scale_, self-supervised learning through the point cloud forecasting task. Lastly, we
demonstrate motion forecasting experiments which provide the first baseline for broad taxonomy
motion prediction.


**4.1** **3D Object Detection**


8


[Table 3: 3d object detection results on the Argoverse 2 Sensor Dataset, taken from the leaderboard](https://eval.ai/web/challenges/challenge-page/1710)
on Dec 21, 2022. _Detectors_ is the winner of the CVPR 2022 _Workshop on Autonomous Driving_
Argoverse 2 3D Object Detection challenge.


|METHOD|MCDS (↑) MAP (↑) MATE (↓) MASE (↓)|
|---|---|
|CENTERPOINT (OURS)<br>DETECTORS [13]<br>BEVFUSION [32]|0.14<br>0.18<br>0.49<br>0.34<br>0.34<br>0.41<br>**0.40**<br>**0.30**<br>**0.37**<br>**0.46**<br>**0.40**<br>**0.30**|



We provide baseline 3D detection results using
a state-of-the-art, anchorless 3D object detection model – CenterPoint [ 51 ]. Our CenterPoint implementation takes a point cloud as
input and crops it to a 200 m _×_ 200 m grid
with a voxel resolution of [ 0 _._ 1 m, 0 _._ 1 m ] in the
_xy_ (bird’s-eye-view) plane and 0 _._ 2 m in the _z_ axis. To accommodate our larger taxonomy, we
include six detection heads to encourage feature specialization. Figure 7 characterizes the
performance of our 3D detection baseline using the nuScenes [ 4 ] average precision metric. Our large taxonomy allows us to evaluate
classes such as “Wheeled Device” (e-Scooter),
“Stroller”, “Dog”, and “Wheelchair” and we find
that performance on these categories with strong
baselines is poor despite significant amounts of
training data.



|0.7|Col2|
|---|---|
|egular Vehicle<br>Bus<br>Pedestrian<br>Stop Sign<br>Box Truck<br>Bollard<br>Construction Barrel<br>Motorcyclist<br>Truck<br>Bicyclist<br>Mobile Crossing Sign<br>Average Metric<br>Motor<br>A<br>0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>AP||
|egular Vehicle<br>Bus<br>Pedestrian<br>Stop Sign<br>Box Truck<br>Bollard<br>Construction Barrel<br>Motorcyclist<br>Truck<br>Bicyclist<br>Mobile Crossing Sign<br>Average Metric<br>Motor<br>A<br>0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>AP|s<br>cycle<br>Bicycle<br>rticulated Bus<br>School Bus<br>Truck Cab<br>Construction Cone<br>Vehicular Trailer<br>Sign<br>Wheeled Device<br>Large Vehicle<br>Stroller<br>essage Board Trailer<br>Dog<br>Wheeled Rider<br>Wheelchair|


Figure 7: Average precision of our 3D object detection baseline on the _validation_ split of the _Sen-_
_sor Dataset (Beta)_ . Our experiments showcase
both our _diverse_ taxonomy and _difficult_ “long-tail”
classes.



In Table 3, we provide a snapshot of submissions
to the Argoverse 2 3D Object Detection Leaderboard.


**4.2** **Point Cloud Forecasting**


We perform point cloud forecasting according to the experimental protocol of SPF2 [ 49 ] using the
Argoverse 2 Lidar Dataset. Given a sequence of past scene point clouds, a model is required to predict
a sequence of future scene point clouds. We take the scene point clouds in the past 1 s ( 10 Hz ) in
the range image format as input, and then predict the next 1 s of range images. SPFNet predicts two
output maps at each time step – the first output map is the predicted range values, while the second
output is a validity mask. Previous point cloud forecasting models were evaluated on smaller datasets
such as KITTI or nuScenes. To explore how the amount of training data affects the performance, we
use increasing amounts of data for training the same model architecture, up to the full training set of
16,000 sequences.


**Evaluation.** We use three metrics to evaluate the performance of our forecasting model: _mean IoU_,
_l_ 1 _-norm_, and _Chamfer distance_ . The _mean IoU_ evaluates the predicted range mask. The _l_ 1 _-norm_
measures the average _l_ 1 distance between the pixel sets of predicted range image and the groundtruth image, which are both masked out by the ground-truth range mask. The _Chamfer distance_ is
obtained by adding up the Chamfer distances in both directions (forward and backward) between the
ground-truth point cloud and the predicted scene point cloud which is obtained by back-projecting
the predicted range image.


Table 4: Results of point cloud forecasting on the _test_ split of the _Lidar Dataset_ .


# T RAIN L OGS

125 250 500 1k 2k 4k 16k


MEAN I O U (%) ( _↑_ ) 55.5 63.4 61.7 65.1 68.0 68.4 **70.9**

_l_ 1                       - NORM ( _↓_ ) 13.5 12.5 11.8 9.9 8.9 8.7 **7.4**

C HAMFER DIST . ( _↓_ ) 31.1 25.9 22.4 22.9 20.5 18.2 **14.0**


9


**Results of SPF2 and Discussion.** Table 4 contains the results of our point cloud forecasting
experiments. With increasing training data, the performance of the model grows steadily in all three
metrics. These results and the works from the self-supervised learning literature [ 3, 7 ] indicate
that a large amount of training data can make a substantial difference. Another observation is that
the Chamfer distances for predictions on our dataset are significantly higher than predictions on
KITTI [ 49 ]. We conjecture that this could be due to two reasons: (1) the Argoverse 2 Lidar Dataset
has a much larger sensing range (above 200 m versus 120 m of the KITTI lidar sensor), which tends
to significantly increase the value of Chamfer distance. (2) the Argoverse 2 Lidar Dataset has a higher
proportion of dynamic scenes compared with KITTI Dataset.


**4.3** **Motion Forecasting**


We present several forecasting baselines [ 6 ] which try to make use of different aspects of the data.
Those which are trained using the focal agent only and do not capture any social interaction include:
constant velocity, nearest neighbor, and LSTM encoder-decoder models (both with and without a
map-prior). We also evaluate WIMP [ 25 ] as an example of a graph-based attention method that
captures social interaction. All hyper-parameters are obtained from the reference implementations.


**Evaluation.** Baseline approaches are evaluated according to standard metrics. Following [ 6 ],
we use _minADE_ and _minFDE_ as the metrics; they evaluate the average and endpoint L2 distance
respectively, between the best forecasted trajectory and the ground truth. We also use _Miss Rate_
_(MR)_ which represents the proportion of test samples where none of the forecasted trajectories were
within 2.0 meters of ground truth according to endpoint error. The resulting performance illustrates
both the community’s progress on the problem and the significant increase in dataset difficulty when
compared with Argoverse 1.1.


Table 5: Performance of motion forecasting baseline methods on vehicle-like ( _vehicle_, _bus_, _mo-_
_torcyclist_ ) object types from the _Argoverse 2 Motion Forecasting (Beta)_ Dataset. Usage of map
prior indicates access to map information whereas usage of social context entails encoding other
actors’ states in the feature representation. Mining intersection (multimodal) scenarios leads to poor
performance at K=1 for all methods. Constant Velocity models have particularly poor performance
due to the dataset bias towards kinematically interesting trajectories. Note that modern deep methods
such as WIMP still have a miss rate of 0.42 at K=6, indicating the increased difficulty of the Argoverse
2 dataset. Numbers within 1% of the best are in bold.

|MODEL MAP PRIOR SOCIAL CONTEXT|K=1 K=6<br>MINADE ↓ MINFDE ↓ MR ↓ MINADE ↓ MINFDE ↓ MR ↓|Col3|
|---|---|---|
|CONST. VEL. [6]<br>NN [6]<br>NN [6]<br>✓<br>LSTM [6]<br>LSTM [6]<br>✓<br>WIMP [25]<br>✓<br>✓|7.75<br>17.44<br>0.89<br>4.46<br>11.71<br>**0.81**<br>6.45<br>15.51<br>0.84<br>**3.05**<br>8.28<br>0.85<br>5.07<br>12.71<br>0.90<br>**3.09**<br>**7.71**<br>0.84|-<br>-<br>-<br>2.18<br>4.94<br>0.60<br>4.30<br>10.08<br>0.78<br>-<br>-<br>-<br>3.73<br>9.09<br>0.85<br>**1.47**<br>**2.90**<br>**0.42**|



Table 6: Motion forecasting results on the Argoverse 2 Motion Forecasting Dataset, taken from
[the online leaderboard on Dec 21, 2022.](https://eval.ai/web/challenges/challenge-page/1719) _BANet_ is the winner of the CVPR 2022 _Workshop on_
_Autonomous Driving_ Argoverse 2 Motion Forecasting challenge (#1), and QML and GANet received
honorable mention (HM) prizes. Entries are sorted below according to _Brier-minFDE_ .

|METHOD|K=1 K=6<br>MINADE ↓ MINFDE ↓ MR ↓ MINADE ↓ MINFDE ↓ MR ↓ BRIER-MINFDE ↓|Col3|
|---|---|---|
|THOMAS (GOHOME SCALAR) [20]<br>GORELA (W/O ENSEMBLE) [11]<br>GANET (ENSEMBLE) (HM) [47]<br>GANET (W/O ENSEMBLE) [47]<br>QML (HM) [44]<br>BANET (OPPRED) (#1) [53]|1.95<br>4.71<br>0.64<br>1.82<br>4.62<br>0.61<br>1.81<br>4.57<br>0.61<br>**1.77**<br>**4.48**<br>**0.59**<br>1.84<br>4.98<br>0.62<br>1.79<br>4.61<br>0.60|0.88<br>1.51<br>0.20<br>2.16<br>0.76<br>1.48<br>0.22<br>2.01<br>0.73<br>1.36<br>**0.17**<br>1.98<br>0.72<br>**1.34**<br>**0.17**<br>1.96<br>**0.69**<br>1.39<br>0.19<br>1.95<br>0.71<br>1.36<br>0.19<br>**1.92**|



**Baseline Results.** Table 5 summarizes the results of baselines. For K=1, Argoverse 1 [ 6 ] showed that
a constant velocity model ( _minFDE_ =7.89) performed better than NN+map(prior) ( _minFDE_ =8.12),


10


which is not the case here. This further proves that Argoverse 2 is kinematically more diverse and
cannot be solved by making constant velocity assumptions. Surprisingly, NN and LSTM variants that
make use of a map prior perform worse than those which do not, illustrating the scope of improvement
in how these baselines leverage the map. For K=6, WIMP significantly outperforms every other
baseline. This emphasizes that it is imperative to train expressive models that can leverage map
prior and social context along with making diverse predictions. The trends are similar to our past 3
Argoverse Motion Forecasting competitions [ 43 ]: Graph-based attention methods (e.g. [ 25, 31, 37 ])
continued to dominate the competition, and were nearly twice as accurate as the next best baseline
(Nearest Neighbor) at K=6. That said, some of the rasterization-based (e.g. [ 19 ]) methods also
showed promising results. Finally, we also evaluated baseline methods in the context of transfer
learning and varied object types, the results of which are summarized in the Appendix.


In Table 6, we provide a snapshot of submissions to the Argoverse 2 Motion Forecasting Leaderboard.


**5** **Conclusion**


**Discussion.** In this work, we have introduced three new datasets that constitute Argoverse 2. We
provide baseline explorations for three tasks – 3d object detection, point cloud forecasting and motion
forecasting. Our datasets provide new opportunities for many other tasks. We believe our datasets
compare favorably to existing datasets, with HD maps, rich taxonomies, geographic diversity, and
interesting scenes.


**Limitations.** As in any human annotated dataset, there is label noise, although we seek to minimize
it before release. 3D bounding boxes of objects are not included in the motion forecasting dataset,
but one can make reasonable assumptions about the object extent given the object type. The motion
forecasting dataset also has imperfect tracking, consistent with state-of-the-art 3D trackers.


**References**


[1] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and
Jurgen Gall. SemanticKITTI: A dataset for semantic scene understanding of lidar sequences. In
_ICCV_, October 2019.


[2] Alex Bewley, Pei Sun, Thomas Mensink, Drago Anguelov, and Cristian Sminchisescu. Range
conditioned dilated convolutions for scale invariant 3d object detection. In _Conference on Robot_
_Learning_, 2020.


[3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _arXiv preprint arXiv:2005.14165_, 2020.


[4] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu,
Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A Multimodal
Dataset for Autonomous Driving. In _CVPR_, 2020.


[5] Yuning Chai, Pei Sun, Jiquan Ngiam, Weiyue Wang, Benjamin Caine, Vijay Vasudevan, Xiao
Zhang, and Dragomir Anguelov. To the point: Efficient 3d object detection in the range image
with graph convolution kernels. In _CVPR_, June 2021.


[6] Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew
Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. Argoverse: 3D
Tracking and Forecasting With Rich Maps. In _CVPR_, 2019.


[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework
for contrastive learning of visual representations. In _ICML_, 2020.


[8] Yun Chen, Frieda Rong, Shivam Duggal, Shenlong Wang, Xinchen Yan, Sivabalan Manivasagam, Shangjie Xue, Ersin Yumer, and Raquel Urtasun. GeoSim: Realistic video simulation
via geometry-aware composition for self-driving. In _CVPR_, June 2021.


[9] Yukyung Choi, Namil Kim, Soonmin Hwang, Kibaek Park, Jae Shin Yoon, Kyounghwan An,
and In So Kweon. Kaist multi-spectral day/night data set for autonomous and assisted driving.
_IEEE Transactions on Intelligent Transportation Systems_, 19(3):934–948, 2018.


11


[10] Yukyung Choi, Namil Kim, Kibaek Park, Soonmin Hwang, Jae Shin Yoon, Yoon In, and Inso
Kweon. All-day visual place recognition: Benchmark dataset and baseline. In _IEEE Conference_
_on Computer Vision and Pattern Recognition Workshops. Workshop on Visual Place Recognition_
_in Changing Environments_, 2015.

[11] Alexander Cui, Sergio Casas, Kelvin Wong, Simon Suo, and Raquel Urtasun. Gorela: Go
relative for viewpoint-invariant motion forecasting. _arXiv preprint arXiv:2211.02545_, 2022.

[12] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan,
Yuning Chai, Benjamin Sapp, Charles Qi, Yin Zhou, Zoey Yang, Aurelien Chouard, Pei
Sun, Jiquan Ngiam, Vijay Vasudevan, Alexander McCauley, Jonathon Shlens, and Dragomir
Anguelov. Large scale interactive motion forecasting for autonomous driving : The waymo
open motion dataset. _CoRR_, abs/2104.10133, 2021.

[13] Jin Fang, Qinghao Meng, Dingfu Zhou, Chulin Tang, Jianbing Shen, Cheng-Zhong Xu, and
Liangjun Zhang. Technical report for cvpr 2022 workshop on autonomous driving argoverse 3d
object detection competition, 2022.

[14] Nils Gählert, Nicolas Jourdan, Marius Cordts, Uwe Franke, and Joachim Denzler. Cityscapes
3d: Dataset and benchmark for 9 dof vehicle detection. _CoRR_, abs/2006.07864, 2020.

[15] Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia
Schmid. VectorNet: Encoding hd maps and agent dynamics from vectorized representation. In
_CVPR_, June 2020.

[16] Runzhou Ge, Zhuangzhuang Ding, Yihan Hu, Yu Wang, Sijia Chen, Li Huang, and Yuan Li.
Afdet: Anchor free one stage 3d object detection. In _CVPR Workshops_, 2020.

[17] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? The
KITTI vision benchmark suite. In _CVPR_, June 2012.

[18] Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi, Xavier Ricou, Rupesh Durgesh, Andrew S.
Chung, Lorenz Hauswald, Viet Hoang Pham, Maximilian Mühlegg, Sebastian Dorn, Tiffany
Fernandez, Martin Jänicke, Sudesh Mirashi, Chiragkumar Savani, Martin Sturm, Oleksandr
Vorobiov, Martin Oelker, Sebastian Garreis, and Peter Schuberth. A2D2: audi autonomous
driving dataset. _CoRR_, abs/2004.06320, 2020.

[19] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde.
Home: Heatmap output for future motion estimation. _arXiv preprint arXiv:2105.10968_, 2021.

[20] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde.
Thomas: Trajectory heatmap output with learned multi-agent sampling. In _ICLR_, 2022.

[21] Wei Han, Zhengdong Zhang, Benjamin Caine, Brandon Yang, Christoph Sprunk, Ouais Alsharif,
Jiquan Ngiam, Vijay Vasudevan, Jonathon Shlens, and Zhifeng Chen. Streaming object detection
for 3-d point clouds. In _ECCV_, 2020.

[22] John Houston, Guido Zuidhof, Luca Bergamini, Yawei Ye, Long Chen, Ashesh Jain, Sammy
Omari, Vladimir Iglovikov, and Peter Ondruska. One Thousand and One Hours: Self-driving
Motion Prediction Dataset. _arXiv:2006.14480 [cs]_, November 2020. Comment: Presented at
CoRL2020.

[23] Peiyun Hu, Aaron Huang, John Dolan, David Held, and Deva Ramanan. Safe local motion
planning with self-supervised freespace forecasting. In _CVPR_, June 2021.

[24] R. Kesten, M. Usman, J. Houston, T. Pandya, K. Nadhamuni, A. Ferreira, M. Yuan, B. Low,
A. Jain, P. Ondruska, S. Omari, S. Shah, A. Kulkarni, A. Kazakova, C. Tao, L. Platinsky,
W. Jiang, and V. Shet. Lyft level 5 av dataset. _arXiv_, 2019.

[25] Siddhesh Khandelwal, William Qi, Jagjeet Singh, Andrew Hartnett, and Deva Ramanan. What-if
motion prediction for autonomous driving. _arXiv preprint arXiv:2008.10587_, 2020.

[26] John W. Lambert and James Hays. Trust, but Verify: Cross-modality fusion for hd map change
detection. In _Advances in Neural Information Processing Systems Track on Datasets and_
_Benchmarks_, 2021.

[27] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom.
PointPillars: Fast encoders for object detection from point clouds. In _CVPR_, June 2019.

[28] Edouard Leurent and Jean Mercat. Social attention for autonomous decision-making in dense
traffic. _CoRR_, abs/1911.12250, 2019.


12


[29] Mengtian Li, Yu-Xiong Wang, and Deva Ramanan. Towards streaming perception. In _ECCV_,
2020.


[30] Qi Li, Yue Wang, Yilun Wang, and Hang Zhao. Hdmapnet: An online HD map construction
and evaluation framework. _CoRR_, abs/2107.06307, 2021.


[31] Ming Liang, Bin Yang, Rui Hu, Yun Chen, Renjie Liao, Song Feng, and Raquel Urtasun.
Learning lane graph representations for motion forecasting. In _ECCV_, 2020.


[32] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, and Song
Han. Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation.
_arXiv preprint arXiv:2205.13542_, 2022.


[33] Yuexin Ma, Xinge Zhu, Sibo Zhang, Ruigang Yang, Wenping Wang, and Dinesh Manocha.
Trafficpredict: Trajectory prediction for heterogeneous traffic-agents. _CoRR_, abs/1811.02146,
2018.


[34] Andrey Malinin, Neil Band, Alexander Ganshin, German Chesnokov, Yarin Gal, Mark J. F.
Gales, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, Vatsal
Raina, Vyas Raina, Mariya Shmatova, Panos Tigas, and Boris Yangel. Shifts: A dataset of real
distributional shift across multiple large-scale tasks. _CoRR_, abs/2107.07455, 2021.


[35] Sivabalan Manivasagam, Shenlong Wang, Kelvin Wong, Wenyuan Zeng, Mikita Sazanovich,
Shuhan Tan, Bin Yang, Wei-Chiu Ma, and Raquel Urtasun. LiDARsim: Realistic lidar simulation by leveraging the real world. In _CVPR_, June 2020.


[36] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang,
Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, Jie Yu, Hang Xu, and Chunjing Xu. One
Million Scenes for Autonomous Driving: ONCE Dataset. _arXiv:2106.11037 [cs]_, August 2021.
Comment: Accepted to NeurIPS 2021 Datasets and Benchmarks Track.


[37] Jean Mercat, Thomas Gilles, Nicole El Zoghby, Guillaume Sandou, Dominique Beauvois, and
Guillermo Pita Gil. Multi-head attention for multi-modal joint vehicle motion forecasting. In
_ICRA_ . IEEE, 2020.


[38] Jean Mercat, Thomas Gilles, Nicole El Zoghby, Guillaume Sandou, Dominique Beauvois, and
Guillermo Pita Gil. Multi-head attention for multi-modal joint vehicle motion forecasting, 2019.


[39] Abhishek Patil, Srikanth Malla, Haiming Gang, and Yi-Ting Chen. The H3D dataset
for full-surround 3d multi-object detection and tracking in crowded urban scenes. _CoRR_,
abs/1903.01568, 2019.


[40] Quang-Hieu Pham, Pierre Sevestre, Ramanpreet Singh Pahwa, Huijing Zhan, Chun Ho Pang,
Yuda Chen, Armin Mustafa, Vijay Chandrasekhar, and Jie Lin. A*3d dataset: Towards autonomous driving in challenging environments. _CoRR_, abs/1909.07541, 2019.


[41] Matthew Pitropov, Danson Evan Garcia, Jason Rebello, Michael Smart, Carlos Wang, Krzysztof
Czarnecki, and Steven Waslander. Canadian adverse driving conditions dataset. _The Interna-_
_tional Journal of Robotics Research_, 40(4-5):681–690, Dec 2020.


[42] Charles R. Qi, Yin Zhou, Mahyar Najibi, Pei Sun, Khoa Vo, Boyang Deng, and Dragomir
Anguelov. Offboard 3d object detection from point cloud sequences. In _CVPR_, June 2021.


[43] Jagjeet Singh, William Qi, Tanmay Agarwal, and Andrew Hartnett. Argoverse motion forecasting competition. `[https://eval.ai/web/challenges/challenge-page/454/overview](https://eval.ai/web/challenges/challenge-page/454/overview)` .
Accessed: 08-27-2021.


[44] Tong Su, Xishun Wang, and Xiaodong Yang. Qml for argoverse 2 motion forecasting challenge,
2022.


[45] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul
Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan
Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya
Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in
Perception for Autonomous Driving: Waymo Open Dataset. In _CVPR_, 2020.


[46] Pei Sun, Weiyue Wang, Yuning Chai, Gamaleldin Elsayed, Alex Bewley, Xiao Zhang, Cristian
Sminchisescu, and Dragomir Anguelov. Rsn: Range sparse net for efficient, accurate lidar 3d
object detection. In _CVPR_, June 2021.


13


[47] Mingkun Wang, Xinge Zhu, Changqian Yu, Wei Li, Yuexin Ma, Ruochun Jin, Xiaoguang Ren,
Dongchun Ren, Mingxu Wang, and Wenjing Yang. Ganet: Goal area network for motion
forecasting, 2022.


[48] Xinshuo Weng, Jianren Wang, Sergey Levine, Kris Kitani, and Nicholas Rhinehart. 4d forecasting: Sequential forecasting of 100,000 points. In _Proceedings of ECCV ’20 Workshops_, August
2020.


[49] Xinshuo Weng, Jianren Wang, Sergey Levine, Kris Kitani, and Nick Rhinehart. Inverting the
forecasting pipeline with spf2: Sequential pointcloud forecasting for sequential pose forecasting.
In _Proceedings of (CoRL) Conference on Robot Learning_, November 2020.


[50] Zhenpei Yang, Yuning Chai, Dragomir Anguelov, Yin Zhou, Pei Sun, Dumitru Erhan, Sean
Rafferty, and Henrik Kretzschmar. Surfelgan: Synthesizing realistic sensor data for autonomous
driving. In _CVPR_, June 2020.


[51] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Center-based 3d object detection and
tracking. In _CVPR_, June 2021.


[52] Wei Zhan, Liting Sun, Di Wang, Haojie Shi, Aubrey Clausse, Maximilian Naumann, Julius
Kummerle, Hendrik Konigshof, Christoph Stiller, Arnaud de La Fortelle, et al. Interaction
dataset: An international, adversarial and cooperative motion dataset in interactive driving
scenarios with semantic maps. _arXiv preprint arXiv:1910.03088_, 2019.


[53] Chen Zhang, Honglin Sun, Chen Chen, and Yandong Guo. Banet: Motion forecasting with
boundary aware network, 2022.


[54] Jannik Zürn, Johan Vertens, and Wolfram Burgard. Lane graph estimation for scene understanding in urban driving. _CoRR_, abs/2105.00195, 2021.


**6** **Appendix**


**6.1** **Additional Information About Sensor Suite**


In Figure 8, we provide a diagram of the sensor suite used to capture the Argoverse 2 datasets.
Figure 9 shows the speed distribution for annotated pedestrian 3D cuboids and the yaw distribution.


Figure 8: Car sensor schematic showing the three coordinate systems: (1) the vehicle frame in the
rear axle; (2) the camera frame; and the lidar frame.


**6.2** **Additional Information About Motion Forecasting Dataset**


**6.2.1** **Interestingness Scores**


Kinematic scoring selects for trajectories performing sharp turns or significant (de)accelerations. The
map complexity program biases the data set towards trajectories complex traversals of the underlying
lane graph. In particular, complex map regions, paths through intersections, and lane-changes score


14


100k

5


2


10k

5


2


1000

5


2


−180 −135 −90 −45 0 45 90 135


Yaw (degrees)





50k





0

|Col1|Col2|Col3|Ar|gove|
|---|---|---|---|---|
||||Wa<br>nuS<br>|ymo<br>cene<br>|
||||~~Ar~~|~~over~~|

0 0.5 1 1.5 2 2.5 3 3.5


Speed (m/s)



Figure 9: **Left** : Number of moving 3D cuboids for pedestrians by speed distribution. We define
moving objects when the speed is greater than 0.5 m/s. **Right** : Number of annotated 3D cuboids by
yaw distribution.


highly. Social scoring rewards tracks through dense regions of other actors. Social scoring also selects
for non-vehicle object classes to ensure adequate samples from rare classes, such as motorcycles, for
training and evaluation. Finally, the autonomous vehicle scoring program encourages the selection of
tracks that intersect the ego-vehicle’s desired route.







|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
|State~~-~~of~~-~~ar<br>Competitio<br>Neurips 20<br>|t minFDE (K=6)<br>n Phase<br>19<br>||||||||||||
|State~~-~~of~~-~~ar<br>Competitio<br>Neurips 20<br>|t minFDE (K=6)<br>n Phase<br>19<br>||||||||||||
|~~CVPR 2020~~<br>None<br>CVPR 2021|||||||||||||
|~~CVPR 2020~~<br>None<br>CVPR 2021||||020/0|020/0|020/0|020/0|20/09|20/|20/|020/|020/|


Figure 10: MinFDE metric values for submissions on Argoverse 1.1 over time. Individual points
indicate submissions to the public leader board. Colors indicate specific competition phases. The solid
black line indicates SOTA performance. The research community made massive gains which have
plateaued since early 2020. However, we note that the number and diversity of methods performing
at or near the SOTA continues to grow. Additionally, later competitions sorted the leaderboard by
“Miss Rate” and probability weighted FDE, and those metrics showed progress. Still, minFDE did
not improve significantly.


**6.3** **Additional Information About HD Maps**


**Examples of HD maps from the Sensor Dataset** In Figure 12, we display examples of local HD
maps associated with individual logs/scenarios.


**6.4** **Additional 3D Detection Results**


In Figure 13, we show additional evaluation metrics for our detection baseline.


15


Figure 11: Histogram of the number of actors (both scored and all types) present in the Motion
Forecasting Dataset scenarios. The Lidar Dataset is mined by the same criteria and thus follows the
same distribution.


**M** **AP** **E** **NTITY** **P** **ROVIDED** **A** **TTRIBUTES** **T** **YPE** **D** **ESCRIPTION**



L ANE

S EGMENTS



IS _ INTERSECTION BOOLEAN WHETHER OR NOT THIS LANE SEGMENT LIES WITHIN AN INTERSECTION .


LANE TYPE ENUMERATED TYPE DESIGNATION OF WHICH VEHICLE TYPES MAY LEGALLY UTILIZE THIS LANE FOR TRAVEL .


LEFT LANE BOUNDARY 3D POLYLINE THE POLYLINE OF THE LEFT BOUNDARY IN THE CITY MAP COORDINATE SYSTEM


RIGHT LANE BOUNDARY 3D POLYLINE THE POLYLINE OF THE RIGHT BOUNDARY IN THE CITY MAP COORDINATE SYSTEM .


LEFT LANE MARK TYPE ENUMERATED TYPE TYPE OF PAINTED LANE MARKING TO THE LEFT OF THE LANE SEGMENT ON THE ROAD .


RIGHT LANE MARK TYPE ENUMERATED TYPE TYPE OF PAINTED LANE MARKING TO THE RIGHT OF THE LANE SEGMENT ON THE ROAD .


LEFT NEIGHBOR INTEGER THE UNIQUE LANE SEGMENT IMMEDIATELY TO THE LEFT OF SEGMENT, OR NONE .


RIGHT NEIGHBOR INTEGER THE UNIQUE LANE SEGMENT IMMEDIATELY TO THE RIGHT OF SEGMENT, OR NONE .



SUCCESSOR ID S INTEGER LIST LANE SEGMENTS THAT MAY BE ENTERED BY FOLLOWING FORWARD .

ID INTEGER UNIQUE IDENTIFIER

D RIVABLE AREA BOUNDARY 3D POLYGONS AREA WHERE IT IS POSSIBLE FOR THE AV TO DRIVE WITHOUT DAMAGING ITSELF
A REA ID INTEGER UNIQUE IDENTIFIER

P EDESTRIAN E DGE 1, E DGE 2 3D POLYLINES ENDPOINTS OF BOTH EDGE ALONG THE PRINCIPAL AXIS, THUS DEFINING A POLYGON .
C ROSSINGS ID INTEGER UNIQUE IDENTIFIER

G ROUND SURFACE HEIGHT - 2 D RASTER ARRAY R ASTER GRID QUANTIZED TO A 30 cm RESOLUTION .


Table 7: HD map attributes for each Argoverse 2 scenario.


**Average Precision (AP)**



1
_AP_ =
101



�

_t∈T_



� _p_ _interp_ ( _r_ ) (1)


_r∈R_



**True Positive Metrics Average Translation Error (ATE)**


_ATE_ = _∥t_ _det_ _−_ _t_ _gt_ _∥_ 2 (2)


**Average Scaling Error (ASE)**



_ASE_ = 1 _−_ �

_d∈D_


**Average Orientation Error (AOE)**



min( _d_ _det_ _, d_ _gt_ ) (3)
_max_ ( _d_ _det_ _, d_ _gt_ )



_AOE_ = _|θ_ _det_ _−_ _θ_ _gt_ _|_ (4)


**Composite Detection Score (CDS)**


_CDS_ = _mAP ·_ � (1 _−_ _x_ ) (5)


_x∈X_


where _X_ = _{mATE_ _unit_ _, mASE_ _unit_ _, mAOE_ _unit_ _}_


16


17


Figure 13: 3D object detection performance on the _validation_ split of the _Sensor Dataset (Beta)_ .
**Top Row:** Composite detection score (left). Average translation error (right) **Bottom Row:** Average
scaling error (left), and average orientation error (right). Results are shown on the _validation_ set of
the _Sensor Dataset_ .


**6.5** **Training Details of SPF2 baseline**


We sample 2-second training snippets (representing 1 second of past and 1 second of future data)
every 0.5 seconds. Thus, for a training log with 30 second duration, 59 training snippets would be
sampled. We train the model for 16 epochs by using the Adam optimizer with the learning rate of
4 _e −_ 3, betas of 0.9 and 0.999, and batch size of 16 per GPU.


**6.6** **Additional Motion Forecasting Experiments**


**6.6.1** **Transfer Learning**


The results of transfer learning experiments are summarized in Table 8. WIMP was trained and tested
in different settings with Argoverse 1.1 and Argoverse 2. As expected, the model works best when
it is trained and tested on the same distribution (i.e. both train and test data come from Argoverse
1.1, or both from Argoverse 2). For example, when WIMP is tested on Argoverse 2 (6s), the model
trained on Argoverse 2 (6s) has a _minFDE_ of 2.91, whereas the one trained on Argoverse 1.1 (3s) has
a _minFDE_ of 6.82 (i.e. approximately 2.3x worse). Likewise, in the reverse setting, when WIMP is
tested on Argoverse 1.1 (3s), the model trained on Argoverse 1.1 (3s) has a _minFDE_ of 1.14 and the
one trained on Argoverse 2 (6s) has _minFDE_ of 2.05 (i.e. approximately 1.8x worse). This indicates
that transfer learning from _Argoverse 2 (Beta)_ to _Argoverse 1.1_ is more useful than the reverse setting,
despite being smaller in the number of scenarios. However, the publicly released version of _Argoverse_
_2 Motion Forecasting_ (the non-beta 2.0 version) has comparable size with Argoverse 1.1.


We note that it is a common practice to train and test sequential models on varied sequence length (e.g.
machine translation). As such, it is still reasonable to expect a model trained with 3s to do well on 6s
horizon. Several factors may contribute to distribution shift, including differing prediction horizon,
cities, mining protocols, object types. Notably, however, these results indicate that Argoverse 2 is
significantly more challenging and diverse than its predecessor.


18


**6.6.2** **Experiment with different object types**


Table 9 shows the results on Nearest Neighbor baseline (without map prior) on different object types.
As one would expect, the displacement errors in pedestrians are significantly lower than other object
types. This occurs because they move at significantly slower velocities. However, this does not imply
that pedestrian motion forecasting is a solved problem and one should rather focus on other object
types. This instead means that we need to come up with better metrics that can capture that fact lower
displacement errors in pedestrians can often be more critical than higher errors in vehicles. We leave
this line of work for future scope.


Table 8: Performance of WIMP when trained and tested on different versions of Argoverse motion
forecasting datasets. Training and evaluation is restricted to vehicle-like (vehicle, bus, motorcyclist)
object types as only vehicles were present in Argoverse 1.1. All the results are for K=6, and prediction
horizon is specified in parentheses. Notably, the model trained on a 3s horizon performs poorly on the
longer 6s horizon. ‘Argoverse 2’ below denotes the _Argoverse 2 (Beta) Motion Forecasting Dataset._


Train Split (pred. horizon) Test Split (pred. horizon) minADE _↓_ minFDE _↓_ MR _↓_


Argoverse 1.1 (3s) Argoverse 1.1 (3s) **0.75** **1.14** **0.12**
Argoverse 2 (6s) Argoverse 1.1 (3s) 1.68 2.05 0.27
Argoverse 1.1 (3s) Argoverse 2 (3s) 0.94 1.88 0.26
Argoverse 1.1 (3s) Argoverse 2 (6s) 4.93 6.82 0.77
Argoverse 2 (6s) Argoverse 2 (6s) **1.48** **2.91** **0.43**


Table 9: Performance of Nearest Neighbor baseline on different object types for K=6. The most
accurately predicted object type for each evaluation metric is highlighted in bold.


Object Type #Samples minADE _↓_ minFDE _↓_ MR _↓_


All 9955 2.48 5.52 0.66

Vehicle 8713 2.62 5.87 0.70

Bus 439 2.69 5.59 0.73

Pedestrians 677 **0.69** **1.31** **0.17**
Motorcyclist 39 2.33 5.07 0.61
Cyclist 87 1.48 2.80 0.42


**7** **Datasheet for Argoverse 2**


**For what purpose was the dataset created?** Was there a specific task in mind? Was there a
specific gap that needed to be filled? Please provide a description.
Argoverse was created to support the global research community in improving the state of the art in
machine learning tasks vital for self driving. The Argoverse 2 datasets described in this manuscript
improve upon the initial Argoverse datasets. These datasets support many tasks, from 3D perception
to motion forecasting to HD map automation.


The three datasets proposed in this manuscript address different gaps in this space. See the comparison
charts in the main manuscript for a more detailed breakdown.


The Argoverse 2 _Sensor Dataset_ has a richer taxonomy than similar datasets. It is the only dataset of
similar size to have stereo imagery. The 1,000 logs in the dataset were chosen to have a variety of
object types with diverse interactions.


The Argoverse 2 _Motion Forecasting Dataset_ also has a richer taxonomy than existing datasets. The
scenarios in the dataset were mined with an emphasis on unusual behaviors that are difficult to predict.


The Argoverse 2 _Lidar Dataset_ is the largest _Lidar Dataset_ . Only the concurrent ONCE dataset is
similarly sized to enable self-supervised learning in lidar space. Unlike ONCE, our dataset contains
HD maps and high frame rate lidar.


**Who created this dataset (e.g., which team, research group) and on behalf of which entity**
**(e.g., company, institution, organization)?**


19


The Argoverse 2 datasets were created by researchers at Argo AI.


**What support was needed to make this dataset?** (e.g.who funded the creation of the dataset? If
there is an associated grant, provide the name of the grantor and the grant name and number, or if it
was supported by a company or government agency, give those details.)
The creation of this dataset was funded by Argo AI.


**Any other comments?**
n/a


**COMPOSITION**


**What do the instances that comprise the dataset represent (e.g., documents, photos, people,**
**countries)?** Are there multiple types of instances (e.g., movies, users, and ratings; people and
interactions between them; nodes and edges)? Please provide a description.


The three constituent datasets of Argoverse 2 have different attributes, but the core instances for each
are brief “scenarios” or “logs” of 11, 15, or 30 seconds that represent a continuous observation of a
scene around a self-driving vehicle.


Each scenario in all three datasets has an HD map that includes lane boundaries, crosswalks, driveable
area, etc. Scenarios for the _Sensor Dataset_ additionally contain a raster map of ground height at .3
meter resolution.


**How many instances are there in total (of each type, if appropriate)?**
The _Sensor Dataset_ has 1,000 15 second scenarios.


The _Lidar Dataset_ has 20,000 30 second scenarios.


The _Motion Forecasting Dataset_ has 250,000 11 second scenarios.


**Does the dataset contain all possible instances or is it a sample (not necessarily random) of**
**instances from a larger set?** If the dataset is a sample, then what is the larger set? Is the
sample representative of the larger set (e.g., geographic coverage)? If so, please describe how
this representativeness was validated/verified. If it is not representative of the larger set, please
describe why not (e.g., to cover a more diverse range of instances, because instances were withheld
or unavailable).
The scenarios in the dataset are a sample of the set of observations made by a fleet of self-driving
vehicles. The data is not uniformly sampled. The particular samples were chosen to be geographically
diverse (spanning 6 cities - Pittsburgh, Detroit, Austin, Palo Alto, Miami, and Washington D.C.), to
include interesting behavior (e.g. cars making unexpected maneuvers), to contain interesting weather
(e.g. rain and snow), and to contain scenes with many objects of diverse types in motion (e.g. a
crowd walking, riders on e-scooters splitting lanes between many vehicles, an excavator operating at
a construction site, etc.).


**What data does each instance consist of?** “Raw” data (e.g., unprocessed text or images) or
features? In either case, please provide a description.
Each _Sensor Dataset_ scenario is 15 seconds in duration. Each scenario has 20 fps video from 7 ring
cameras, 20 fps video from two forward facing stereo cameras, and 10 hz lidar returns from two
out-of-phase 32 beam lidars. The ring cameras are synchronized to fire when either lidar sweeps
through their field of view. Each scenario contains vehicle pose over time and calibration data to
relate the various sensors.


Each _Lidar Dataset_ scenario is 30 seconds in duration. These scenarios are similar to those of the
_Sensor Dataset_, except that there is no imagery.


Each _Motion Forecasting_ scenario is 11 seconds in duration. These scenarios contain no sensor data,
but instead contain tracks of objects such as vehicles, pedestrians, and bicycles. The tracks specify
the category of each object (e.g. bus or bicycle) as well as their location and heading at a 10 hz
sampling interval.


20


The HD map associated with all three types of scenarios contains polylines describing lanes, crosswalks, and driveable area. Lanes form a graph with predecessors and successors, e.g. a lane that
splits can have two successors. Lanes have precisely localized lane boundaries that include paint
type (e.g. double solid yellow). Driveable area, also described by a polygon, is the area where it is
possible but not necessarily legal to drive. It includes areas such as road shoulders.


**Is there a label or target associated with each instance?** If so, please provide a description.
Each _Sensor Dataset_ scenario has 3D track annotations for dynamic objects such as vehicles, pedestrians, strollers, dogs, etc. The tracks are suitable as ground truth for tasks such as 3D object detection
and 3D tracking. The 3D track labels are intentionally held out from the test set. The HD map
could also be thought of as labels for each instance, and would be suitable as ground truth for lane
detection or map automation. The vehicle pose data could be considered ground truth labels for visual
odometry. The lidar depth estimates can act as ground truth for monocular or stereo depth estimation.


The _Lidar Dataset_ does not have human annotations beyond the HD map. Still, the evolving point
cloud itself can be considered ground truth for point cloud forecasting.


Each _Motion Forecasting Dataset_ scenario provides labels specifying which tracks are associated
with “scored actors”. These tracks exhibit interesting behavior and are guaranteed to be observed
over the entire duration of each scenario; algorithms will be asked to forecast the future motion for
these tracks. The future motion of actors in each scenario is intentionally held out in the test set.


**Is any information missing from individual instances?** If so, please provide a description,
explaining why this information is missing (e.g., because it was unavailable). This does not include
intentionally removed information, but might include, e.g., redacted text.
In the _Sensor Dataset_, objects are only labeled within 5 meters of the driveable area. For example, a
person sitting on their front porch will not be labeled.


In the _Sensor Dataset_ and _Motion Forecasting Dataset_, instances are not necessarily labeled for the
full duration of each scenario if the objects move out of observation range or become occluded.


**Z Are relationships between individual instances made explicit (e.g., users’ movie ratings,**
**social network links)?** If so, please describe how these relationships are made explicit.
The instances of the three datasets are disjoint. They each carry their own HD map for the region
around the scenario. These HD maps may overlap spatially, though. For example, many forecasting
scenarios may take place in the same intersection. If a user of the dataset wanted to recover the
spatial relationship between scenarios, they could do so through the Argoverse API.


**Are there recommended data splits (e.g., training, development/validation, testing)?** If so,
please provide a description of these splits, explaining the rationale behind them.
We define splits of each dataset. The _Sensor Dataset_ is split 700 / 150 / 150 between train, validation,
and test. The _Lidar Dataset_ is split 16,000 / 2,000 / 2,000 and the _Motion Forecasting Dataset_ is split
200,000 / 25,000 / 25,000. In all cases, the splits are designed to make the training dataset as large
as possible while keeping the validation and test datasets large and diverse enough to accurately
benchmark models learned on the training set.


**Are there any errors, sources of noise, or redundancies in the dataset?** If so, please provide a
description.
Every sensor used in the dataset – ring cameras, stereo cameras, and lidar – has noise associated with
it. Pixel intensities, lidar intensities, and lidar point 3D locations all have noise. Lidar points are
also quantized to float16 which leads to roughly a centimeter of quantization error. Six degree of
freedom vehicle pose also has noise. The calibration specifying the relationship between sensors can
be imperfect.


The HD map for each scenario can contain noise, both in terms of lane boundary locations and precise
ground height.


The 3D object annotations in the _Sensor Dataset_ do not always match the spatial extent and motion of
an object in the real world. For example, we assume that objects do not change size during a scenario,
but this could be violated by a car opening its door. 3D annotations for distant objects with relatively
few pixels and lidar returns are less accurate.


21


The object tracks in the _Motion Forecasting_ dataset are imperfect and contain errors typical of a
real-time 3D tracking method. Our expectation is that a motion forecasting algorithm should operate
well despite this noise.


**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g.,**
**websites, tweets, other datasets)?** If it links to or relies on external resources, a) are there
guarantees that they will exist, and remain constant, over time; b) are there official archival versions
of the complete dataset (i.e., including the external resources as they existed at the time the dataset
was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external
resources that might apply to a future user? Please provide descriptions of all external resources and
any restrictions associated with them, as well as links or other access points, as appropriate.
The data itself is self-hosted, like Argoverse 1 [see `[https://www.argoverse.org/](https://www.argoverse.org/)` ], and we
maintain public links to all previous versions of the dataset in case of updates. The data is independent
of any previous datasets, including Argoverse 1.


**Does the dataset contain data that might be considered confidential (e.g., data that is**
**protected by legal privilege or by doctor-patient confidentiality, data that includes the content**
**of individuals’ non-public communications)?** If so, please provide a description.
No.


**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threaten-**
**ing, or might otherwise cause anxiety?** If so, please describe why.
No.


**Does the dataset relate to people?** If not, you may skip the remaining questions in this section.
Yes, the dataset contains images and behaviors of thousands of people on public streets.


**Does the dataset identify any subpopulations (e.g., by age, gender)?** If so, please describe how
these subpopulations are identified and provide a description of their respective distributions within
the dataset.

No.


**Is it possible to identify individuals (i.e., one or more natural persons), either directly or**
**indirectly (i.e., in combination with other data) from the dataset?** If so, please describe how.
We do not believe so. All image data has been anonymized. Faces and license plates are obfuscated
by replacing them with a 5x5 grid, where each grid cell is the average color of the original pixels in
that grid cell. This anonymization is done manually and is not limited by our 3D annotation policy.
For example, a person sitting on their front porch 10 meters from the road would not be labeled with
a 3D cuboid, but their face would still be obscured.


**Does the dataset contain data that might be considered sensitive in any way (e.g., data that**
**reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or**
**union memberships, or locations; financial or health data; biometric or genetic data; forms of**
**government identification, such as social security numbers; criminal history)?** If so, please
provide a description.
No.


**Any other comments?**
n/a


**COLLECTION**


**How was the data associated with each instance acquired?** Was the data directly observable
(e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly
inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)?
If data was reported by subjects or indirectly inferred/derived from other data, was the data


22


validated/verified? If so, please describe how.
The sensor data was directly acquired by a fleet of autonomous vehicles.


**Over what timeframe was the data collected?** Does this timeframe match the creation timeframe
of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please
describe the timeframe in which the data associated with the instances was created. Finally, list when
the dataset was first published.
The data was collected in 2020 and 2021. The dataset was made public after NeurIPS 2021, in March
2022.


**What mechanisms or procedures were used to collect the data (e.g., hardware apparatus**
**or sensor, manual human curation, software program, software API)?** How were these
mechanisms or procedures validated?
The Argoverse 2 data comes from Argo ‘Z1’ fleet vehicles. These vehicles use Velodyne lidars and
traditional RGB cameras. All sensors are calibrated by Argo. HD maps and 3D object annotations
are created and validated through a combination of computational tools and human annotations.
Object tracks in the _Motion Forecasting Dataset_ are created by a 3D tracking algorithm.


**What was the resource cost of collecting the data?** (e.g. what were the required computational
resources, and the associated financial costs, and energy consumption - estimate the carbon footprint.
See Strubell _et al._ for approaches in this area.)
The data was captured during normal fleet operations, so there was minimal overhead for logging
particular events. The transformation and post-processing of several terabytes of data consumed an
estimated 1,000 machine hours. We estimate a Carbon footprint of roughly 1,000 lbs based on the
CPU-centric workload.


**If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic,**
**probabilistic with specific sampling probabilities)?**
The _Sensor Dataset_ scenarios were chosen from a larger set through manual review. The _Lidar_
_Dataset_ and _Motion Forecasting Dataset_ scenarios were chosen by heuristics which looked for
interesting object behaviors during fleet operations.


**Who was involved in the data collection process (e.g., students, crowdworkers, contractors)**
**and how were they compensated (e.g., how much were crowdworkers paid)?**
Argo employees and Argo interns curated the data. Data collection and data annotation was done by
Argo employees. Crowdworkers were not used.


**Were any ethical review processes conducted (e.g., by an institutional review board)?** If so,
please provide a description of these review processes, including the outcomes, as well as a link or
other access point to any supporting documentation.
No.


**Does the dataset relate to people?** If not, you may skip the remainder of the questions in this
section.

Yes.


**Did you collect the data from the individuals in question directly, or obtain it via third parties**
**or other sources (e.g., websites)?**
The data is collected from vehicles on public roads, not from a third party.


**Were the individuals in question notified about the data collection?** If so, please describe (or
show with screenshots or other information) how notice was provided, and provide a link or other
access point to, or otherwise reproduce, the exact language of the notification itself.
No, but the data collection was not hidden. The Argo fleet vehicles are well marked and have obvious


23


cameras and lidar sensors. The vehicles only capture data from public roads.


**Did the individuals in question consent to the collection and use of their data?** If so, please
describe (or show with screenshots or other information) how consent was requested and provided,
and provide a link or other access point to, or otherwise reproduce, the exact language to which the
individuals consented.
No. People in the dataset were in public settings and their appearance has been anonymized. Drivers,
pedestrians, and vulnerable road users are an intrinsic part of driving on public roads, so it is important that datasets contain people so that the community can develop more accurate perception systems.


**If consent was obtained, were the consenting individuals provided with a mechanism to**
**revoke their consent in the future or for certain uses?** If so, please provide a description, as well
as a link or other access point to the mechanism (if appropriate)
n/a


**Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data**
**protection impact analysis) been conducted?** If so, please provide a description of this analysis,
including the outcomes, as well as a link or other access point to any supporting documentation.
No.


**Any other comments?**
n/a


**PREPROCESSING / CLEANING / LABELING**


**Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing,**
**tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing**
**of missing values)?** If so, please provide a description. If not, you may skip the remainder of the
questions in this section.
Yes. Images are reduced from their full resolution. 3D point locations are quantized to float16.
Ground height maps are quantized to .3 meter resolution from their full resolution. HD map polygon
vertex locations are quantized to .01 meter resolution. 3D annotations are smoothed. For the _Motion_
_Forecasting Dataset_, transient 3D tracks are suppressed and object locations are smoothed over time.


**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to**
**support unanticipated future uses)?** If so, please provide a link or other access point to the “raw”
data.
Yes, but such data is not public.


**Is the software used to preprocess/clean/label the instances available?** If so, please provide a
link or other access point.
No.


**Any other comments?**
n/a


**USES**


**Has the dataset been used for any tasks already?** If so, please provide a description.
Yes, this manuscript benchmarks a contemporary 3D object detection method on the _Sensor Dataset_
and a contemporary motion forecasting method on the _Motion Forecasting Dataset_ .


24


**Is there a repository that links to any or all papers or systems that use the dataset?** If so,
please provide a link or other access point.
Yes, the Argoverse 2 API can be found at `[https://github.com/argoverse/av2-api](https://github.com/argoverse/av2-api)` .


For the Argoverse 2 datasets, we maintain two leaderboards for 3D Detection [ `[https://eval.ai/](https://eval.ai/web/challenges/challenge-page/1710)`
`[web/challenges/challenge-page/1710](https://eval.ai/web/challenges/challenge-page/1710)` ] and Motion Forecasting [ `[https://eval.ai/web/](https://eval.ai/web/challenges/challenge-page/1719)`
`[challenges/challenge-page/1719](https://eval.ai/web/challenges/challenge-page/1719)` ].


For the Argoverse 1 datasets, we maintain four leaderboards for 3D Tracking

[ `[https://eval.ai/web/challenges/challenge-page/453/overview](https://eval.ai/web/challenges/challenge-page/453/overview)` ], 3D Detection

[ `[https://eval.ai/web/challenges/challenge-page/725/overview](https://eval.ai/web/challenges/challenge-page/725/overview)` ], Motion Forecasting [ `[https://eval.ai/web/challenges/challenge-page/454/overview](https://eval.ai/web/challenges/challenge-page/454/overview)` ], and Stereo
Depth Estimation [ `[https://eval.ai/web/challenges/challenge-page/917/overview](https://eval.ai/web/challenges/challenge-page/917/overview)` ].
Argoverse 1 was also used as the basis for a Streaming Perception challenge [ `[https:](https://eval.ai/web/challenges/challenge-page/800/overview)`
`[//eval.ai/web/challenges/challenge-page/800/overview](https://eval.ai/web/challenges/challenge-page/800/overview)` ].


**What (other) tasks could the dataset be used for?**
The datasets could be used for research on visual odometry, pose estimation, lane detection, map
automation, self-supervised learning, structure-from-motion, scene flow, optical flow, time to contact
estimation, pseudo-lidar, and point cloud forecasting.


**Is there anything about the composition of the dataset or the way it was collected and**
**preprocessed/cleaned/labeled that might impact future uses?** For example, is there anything
that a future user might need to know to avoid uses that could result in unfair treatment of individuals
or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial
harms, legal risks) If so, please provide a description. Is there anything a future user could do to
mitigate these undesirable harms?
No.


**Are there tasks for which the dataset should not be used?** If so, please provide a description.
The dataset should not be used for tasks which depend on faithful appearance of faces or license
plates since that data has been obfuscated. For example, running a face detector to try and estimate
how often pedestrians use crosswalks will not result in meaningful data.


**Any other comments?**
n/a


**DISTRIBUTION**


**Will the dataset be distributed to third parties outside of the entity (e.g., company, institution,**
**organization) on behalf of which the dataset was created?** If so, please provide a description.
Yes, the dataset is hosted on `[https://www.argoverse.org/](https://www.argoverse.org/)` like Argoverse 1 and 1.1.


**How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?** Does the
dataset have a digital object identifier (DOI)?
We provide both tar.gz archives and raw files for two of the Argoverse 2 datasets ( _Motion Forecasting_,
_Sensor_ ), but provide only raw files for the _Lidar_ datasets), available via AWS transfer. See `[https:](https://www.argoverse.org/av2.html#download-link)`
`[//www.argoverse.org/av2.html#download-link](https://www.argoverse.org/av2.html#download-link)` .


The Argoverse 1 and Argoverse 1.1 were distributed as a series of tar.gz files (See
`[https://www.argoverse.org/av1.html#download-link](https://www.argoverse.org/av1.html#download-link)` . The files are broken up to
make the process more robust to interruption (e.g. a single 2 TB file failing after 3 days would be
frustrating) and to allow easier file manipulation (an end user might not have 2 TB free on a single
drive, and if they do they might not be able to decompress the entire file at once).


25


**When will the dataset be distributed?**
The data was made available for download after NeurIPS 2021, in March 2022.


**Will the dataset be distributed under a copyright or other intellectual property (IP) license,**
**and/or under applicable terms of use (ToU)?** If so, please describe this license and/or ToU, and
provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU,
as well as any fees associated with these restrictions.
Yes, the dataset was released under the same Creative Commons license as Argoverse 1 – CC BYNC-SA 4.0. Details can be seen at `[https://www.argoverse.org/about.html#terms-of-use](https://www.argoverse.org/about.html#terms-of-use)` .


**Have any third parties imposed IP-based or other restrictions on the data associated with**
**the instances?** If so, please describe these restrictions, and provide a link or other access point
to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these
restrictions.

No.


**Do any export controls or other regulatory restrictions apply to the dataset or to individual**
**instances?** If so, please describe these restrictions, and provide a link or other access point to, or
otherwise reproduce, any supporting documentation.
No.


**Any other comments?**
n/a


**MAINTENANCE**


**Who is supporting/hosting/maintaining the dataset?**
Argo AI


**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
The Argoverse team responds through the Github page for the Argoverse 2 API: `[https://github.](https://github.com/argoverse/av2-api/issues)`
`[com/argoverse/av2-api/issues](https://github.com/argoverse/av2-api/issues)` .


The Argoverse team responds through the Github page for the Argoverse 1 API: `[https://github.](https://github.com/argoverse/argoverse-api/issues)`
`[com/argoverse/argoverse-api/issues](https://github.com/argoverse/argoverse-api/issues)` . It currently contains 2 open issues and 126 closed
issues.


For privacy concerns, contact information can be found here: `[https://www.argoverse.org/](https://www.argoverse.org/about.html#privacy)`
```
about.html#privacy

```

**Is there an erratum?** If so, please provide a link or other access point.
No.


**Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete**
**instances)?** If so, please describe how often, by whom, and how updates will be communicated to
users (e.g., mailing list, GitHub)?
It is possible that the constituent Argoverse 2 datasets are updated to correct errors. This was the
case with Argoverse 1 which was incremented to Argoverse 1.1. Updates will be communicated on
Github and through our mailing list.


**If the dataset relates to people, are there applicable limits on the retention of the data**
**associated with the instances (e.g., were individuals in question told that their data would be**
**retained for a fixed period of time and then deleted)?** If so, please describe these limits and
explain how they will be enforced.


26


No.


**Will older versions of the dataset continue to be supported/hosted/maintained?** If so, please
describe how. If not, please describe how its obsolescence will be communicated to users.
Yes. We still host Argoverse 1 even though we have declared it “deprecated”. See
`[https://www.argoverse.org/av1.html#download-link](https://www.argoverse.org/av1.html#download-link)` . We will use the same warning if we ever deprecate Argoverse 2. Note: Argoverse 2 does not deprecate Argoverse 1. They are
independent collections of datasets.


**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for**
**them to do so?** If so, please provide a description. Will these contributions be validated/verified? If
so, please describe how. If not, why not? Is there a process for communicating/distributing these
contributions to other users? If so, please provide a description.
Yes. For example, the streaming perception challenge was built by CMU researchers who added new
2D object annotations to Argoverse 1.1 data. The Creative Commons license we use for Argoverse 2
ensures that the community can do the same thing without needing Argo’s permission.


We do not have a mechanism for these contributions/additions to be incorporated back into the ‘base’
Argoverse 2. Our preference would generally be to keep the ‘base’ dataset as is, and to give credit to
noteworthy additions by linking to them as we have done in the case of the Streaming Perception
Challenge (see link at the top of this Argoverse page `[https://www.argoverse.org/tasks.html](https://www.argoverse.org/tasks.html)` ).


**Any other comments?**
n/a


**Environmental Impact Statement.** Amount of Compute Used: We estimate 2,000 CPU and 500
GPU hours were used in the collection of the dataset and the performance of baseline experiments.


27


