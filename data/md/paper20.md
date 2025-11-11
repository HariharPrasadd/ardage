**Noname manuscript No.**
(will be inserted by the editor)

## **Human Action Recognition and Prediction: A Survey**


**Yu Kong** _**·**_ **Yun Fu**


Received: date / Accepted: date



**Abstract** Derived from rapid advances in computer
vision and machine learning, video analysis tasks have
been moving from inferring the present state to predicting the future state. Vision-based action recognition and prediction from videos are such tasks, where
action recognition is to infer human actions (present
state) based upon complete action executions, and action prediction to predict human actions (future state)
based upon incomplete action executions. These two
tasks have become particularly prevalent topics recently
because of their explosively emerging real-world applications, such as visual surveillance, autonomous driving
vehicle, entertainment, and video retrieval, etc. Many
attempts have been devoted in the last a few decades in
order to build a robust and effective framework for action recognition and prediction. In this paper, we survey
the complete state-of-the-art techniques in action recognition and prediction. Existing models, popular algorithms, technical difficulties, popular action databases,
evaluation protocols, and promising future directions
are also provided with systematic discussions.


**1 Introduction**


Every human action, no matter how trivial, is done
for some purpose. For example, in order to complete


Yu Kong
B. Thomas Golisano College of Computing and Information
Sciences, Rochester Institute of Technology, Rochester, NY,
USA
E-mail: yu.kong@rit.edu


Yun Fu

Department of ECE and College of CIS, Northeastern University, Boston, MA, USA
E-mail: yunfu@ece.neu.edu



a physical exercise, a patient is interacting with and responding to the environment using his/her hands, arms,
legs, torsos, bodies, etc. An action like this denotes everything that can be observed, either with bare eyes or
measured by visual sensors. Through the human vision
system, we can understand the action and the purpose
of the actor. We can easily know that a person is exercising, and we could guess with a certain confidence
that the person’s action complies with the instruction
or not. However, it is way too expensive to use human labors to monitor human actions in a variety of
real-world scenarios, such as smart rehabilitation and
visual surveillance. Can a machine perform the same as
a human?

One of the ultimate goals of artificial intelligence
research is to build a machine that can accurately understand humans’ actions and intentions, so that it can
better serve us. Imagine that a patient is undergoing a
rehabilitation exercise at home, and his/her robot assistant is capable of recognizing the patient’s actions,
analyzing the correctness of the exercise, and preventing the patient from further injuries. Such an intelligent
machine would be greatly beneficial as it saves the trips
to visit the therapist, reduces the medical cost, and
makes remote exercise into reality. Other important applications including visual surveillance, entertainment,
and video retrieval also need to analyze human actions
in videos. In the center of these applications is the computational algorithms that can understand human actions. Similar to the human vision system, the algorithms ought to produce a label after observing the entire or part of a human action execution [15,220]. Building such algorithms is typically addressed in computer
vision research, which studies how to make computers
gain high-level understanding from digital images and
videos.


2 Yu Kong, Yun Fu


**Fig. 1** Framework of the survey. The picture presents the topics discussed in the survey organized in a hierarchical tree, a list
of representative works are also included for each topic.









































The term _human action_ studied in computer vision research ranges from the simple limb movement to
joint complex movement of multiple limbs and the human body. This process is dynamic, and thus is usually
conveyed in a video lasting a few seconds. Though it
might be difficult to give a formal definition of human
action studied in the computer vision community, we
provide some examples used in the community. Typical
example actions are, 1) an individual action in KTH
dataset [227] (Fig. 2(a)), which contains simple daily
actions such as “clapping” and “running”; 2) a human
interaction in UT-Interaction dataset [217] (Fig. 2(b)),
which consists of human interactions including “handshake” and “push”; 3) a human-object interaction in
UCF Sports dataset [213] (Fig. 2(c)), which comprises
of sport actions and human-object interactions; 4) a
group action in Hollywood 2 dataset [170] (Fig. 2(d));
5) an action captured by a RGB-D sensor in UTKinect
dataset [295] (Fig. 2(e)); and 6) a multi-view action in
Multicamera dataset [238] (Fig. 2(f)) capturing human
actions from multiple camera views. In all these examples, a human action attempts to achieve a certain goal,
in which some of them can be achieved by simply moving arms, and the others need to be accomplished in
several steps.



Technology advances in computer science and engineering have been enabling machines to understand
human actions in videos. There are two basic topics
in the computer vision community, vision-based human
action recognition and prediction:


1. **Action recognition** : recognize a human action from
a video containing complete action execution.
2. **Action prediction** : reason a human action from
temporally incomplete video data.


_Action recognition_ is a fundamental task in the computer vision community that recognizes human actions
based on the complete action execution in a video (see
Figure 3(a)) [15,48,285,133,153,251,255]. It has been
studied for decades and is still a very popular topic
due to broad real-world applications including video retrieval [30], visual surveillance [83,238], etc. Researchers
have made great efforts to create an intelligent system
mimicking humans’ capability that can recognize complex human actions in cluttered environments. However, to a machine, an action in a video is just an array
of pixels. The machine has no idea about how to convert these pixels into an effective representation, and
how to infer human actions from the representation.
These two problems are considered as _action represen-_
_tation_ and _action classification_ in action recognition,


Human Action Recognition and Prediction: A Survey 3


(a) (b) (c) (d) (e) (f)


**Fig. 2** Example frames of action videos used in computer vision research. (a) single person’s action; (b) human interaction;









to address these two problems.
On the contrary, _action prediction_ is a before-thefact video understanding task and is focusing on the
future state. In some real-world scenarios ( _e.g._, vehicle
accidents and criminal activities), intelligent machines
do not have the luxury of waiting for the entire action
execution before having to react to the action contained
in it. For example, being able to predict a dangerous
driving situation before it occurs; opposed to recognizing it thereafter. This is referred to as the action prediction task where approaches that can recognize and
infer a label from a temporally incomplete video (see
Figure 3(b)) [220,117,118], different to action recognition approaches that expect to see the entire set of
action dynamics extracted from a full video.
The major difference between action recognition and
action prediction lies in _when to make a decision_ . Human action recognition is to infer the action label _af-_
_ter_ the entire action execution has been observed. This

task is generally useful in non-urgent scenarios, such as
video retrieval, entertainment, etc. Nevertheless, action
prediction is to infer _before_ fully observing the entire
execution, which is of particular important in certain
scenarios. For example, it would be very helpful if an
intelligent system on a vehicle can predict a traffic accident before it happens; opposed to recognizing the
dangerous accident event thereafter.
We will mainly discuss recent advance in action recognition and prediction in this survey. To ease the navigation of this paper, Fig. 1 illustrates the topics discussed in this paper and the representative works are



also included. Different from recent survey papers [79,
199], studies in action prediction are also described in
this paper. Human action recognition and prediction
are closely related to other computer vision tasks such
as human gesture analysis, gait recognition, and event
recognition. In this survey, we focus on the vision-based
recognition and prediction of actions from videos that
usually involve one or more people. The input is a series of video frames and the output is an action label.
We are also interested in learning human actions from
RGB-D videos. Some of existing studies [310,309] aim
at learning actions from static images, which is not the
focus of this paper. This paper will first give an overview
of recent studies in action recognition and prediction,
describe popular human actions datasets, and will then
discuss several interesting future directions in details.


1.1 Real-World Applications


Action recognition and prediction algorithms empower
many real-world applications (examples are shown in
Figure 4). State-of-the-art algorithms [278,53,114,165]
remarkably reduce the human labor in analyzing a largescale of video data and provide understanding on the
current state and future state of ongoing video data.


_1.1.1 Visual Surveillance_


Security issue is becoming more important in our daily
life, and it is one of the most frequently discussed topics nowadays. Places under surveillance typically allow
certain human actions, and other actions are not allowed [83]. With the input of a network of cameras [285,
238], a visual surveillance system powered by action
recognition [90,237,99] and prediction [220,117,118] algorithms may increase the chances of capturing a criminal on video, and reduce the risk caused by criminal
actions. For example, in Boston marathon bombing site,
if we had such an intelligent visual surveillance system
that can forewarn the public by looking at the criminal’s suspicious action, the victims’ lives could be saved.
The cameras also make some people feel more secure,
knowing the criminals are being watched.


4 Yu Kong, Yun Fu


_1.1.2 Video Retrieval_



Nowadays, due to the fast growth of technology, people can easily upload and share videos on the Internet.
However, managing and retrieving videos according to
video content is becoming a tremendous challenge as
most search engines use the associated text data to
manage video data [205]. The text data, such as tags,
titles, descriptions, and keywords, can be incorrect, obscure, and irrelevant, making video retrieval unsuccessful [325]. An alternative method is to analyze human
actions in videos, as the majority of these videos contain such a cue. For example, in [30], researchers created
a video retrieval framework by computing the similarity
between action representations, and used the proposed
framework to retrieve videos of children with autism in

a classroom setting. Compared to conventional human
action recognition task, the video retrieval task relies
on the retrieval ranking instead of classification [205].


_1.1.3 Entertainment_


The gaming industry in recent years has attracted an
increasingly large and diverse group of people. A new
generation of games based on full body play such as
dance and sports games have increased the appeal of
gaming to family members of all ages. To enable accurate perception of human actions, these games use
cost-effective RGB-D sensors ( _e.g._, Kinect [232]) which
provide an additional depth channel data [294,305,75].
This depth data encode rich structural information of
the entire scene, and facilitate action recognition task
as it simplifies intra-class motion variations and reduces
cluttered background noise [111,113,91,156].


_1.1.4 Human-Robot Interaction_


Human-robot interaction is popularly applied in home
and industry environment. Imagine that a person is interacting with a robot and asking it to perform certain
tasks, such as “passing a cup of water” or “performing an assembling task”. Such an interaction requires
communications between robots and humans, and visual communication is one of the most efficient ways

[219,124].


_1.1.5 Autonomous Driving Vehicle_


Action prediction algorithms [218,112] could be one of
the potentials and maybe most important building components in an autonomous driving vehicle. Action prediction algorithms can predict a person’s intention [194,



(a) Human-robot interaction (b) Entertainment (c) Autonomous driving car


**Fig. 4** Examples of real-world applications using action
recognition techniques.


143,124] in a short period of time. In an urgent situation, a vehicle equipped with an action prediction algorithm can predict a pedestrian’s future action or motion
trajectory in the next few seconds, and this could be
critical to avoid a collision. By analyzing human body
motion characteristics at an early stage of an action
using so-called interest points or convolutional neural
network [118], action prediction algorithms [118,112]
can understand the possible actions by analyzing the
action evolution without the need to observe the entire

action execution.


1.2 Research Challenges


Despite significant progress has been made in human
action recognition and prediction, state-of-the-art algorithms still misclassify actions due to several major
challenges in these tasks.


_1.2.1 Intra- and inter-class Variations_


As we all know, people behave differently for the same
actions. For a given semantic meaningful action, for example, “running”, a person can run fast, slow, or even
jump and run. That is to say, one action category may
contain multiple different styles of human movements.
In addition, videos in the same action can be captured
from various viewpoints. They can be taken in front
of the human subject, on the side of the subject, or
even on top of the subject, showing appearance variations in different views (see Figure 5). Furthermore,
different people may show different poses in executing the same action. All these factors will result in
large intra-class appearance and pose variations, which
confuse a lot of existing action recognition algorithms.
These variations will be even larger on real-world action datasets [99,50]. This triggers the investigation of
more advanced action recognition algorithms that can
be deployed in real-world scenarios. Furthermore, similarities exist in different action categories. For instance,
“running” and “walking” involve similar human motion
patterns. These similarities would also be challenging to


_1.2.2 Cluttered Background and Camera Motion_


It is interesting to see that a number of human action
recognition algorithms work very well in indoor controlled environments but not in outdoor uncontrolled

environments. This is mainly due to the background
noise. In fact, most of the existing activity features such
as histograms of oriented gradient [135] and interest
points [41] also encode background noise, and thus degrade the recognition performance. Camera motion is
another factor that should be considered in real-world
applications. Due to significant camera motion, action
features cannot be accurately extracted. In order to better extract action features, camera motion should be
modeled and compensated [268]. Other environmentrelated issues such as illumination conditions, viewpoint
changes, dynamic background will also be the challenges that prohibit action recognition algorithms from
being used in practical scenarios.


_1.2.3 Insufficient Annotated Data_


Even though existing action recognition approaches [108,
152,185] have shown impressive performance on smallscale datasets in laboratory settings, it is really challenging to generalize them to real-world applications
due to their inability of training on large-scale datasets.
Recent deep approaches [278,53] have shown promising results on datasets captured in uncontrolled settings, but they normally require a large amount of annotated training data. Action datasets such as HMDB51

[127] and UCF-101 [105] contain thousands of videos,
but still far from enough for training deep networks
with millions of parameters. Although Youtube-8M [4]
and Sposrts-1M datasets [99] provide millions of action
videos, their annotations are generated by a retrieval
method, and thus may not be accurate. Training on
such datasets would hurt the performance of action
recognition algorithms that do not have a tolerance to
inaccurate labels. However, it is possible that some of
the data annotations are available, which would result
in a training setting with a mixture of labeled data and



fine and analyze these different kinds of actions is very
important.


_1.2.5 Uneven Predictability_


Not all frames are equally discriminative. As shown
in [206,260], a video can be effectively represented by
a small set of key frames. This indicates that lots of
frames are redundant, and discriminative frames may
appear anywhere in the video. However, action prediction methods [220,117,165,130] require the beginning
portions of the video to be discriminative in order to
maximize predictability. To solve this problem, context
information is transferred to the beginning portions of
the videos [118], but the performance is still limited due
to the insufficient discriminative information.
In addition, actions differ in their predictabilities

[143,118]. As shown in [118], some actions are instantly
predictable while the other ones need more frames to be
observed. However, in practical scenarios, it is necessary
to predict any actions as early as possible. This requires
us to create general action prediction algorithms that
can make accurate and early predictions for most of or
all actions.


**2 Human Perception of Actions**


Human actions, particularly those involving whole-body
and limb ( _e.g._, arms and legs) movements, and interactions with their environment contain rich informa
tion about the performer’s intention, goal, mental status, etc. Understanding the actions and intentions of
other people is one of the most important social skills
we have, and the human vision system provides a particularly rich source of information in support of this
skill [13]. Compared to static images, human actions in
videos provide even more reliable and more expressive
information, and thus speak louder than images when
it comes to understanding what others are doing [36].
There are a number of information we can tell from human actions, including the action categories [171], emotional implication [31], identity [32,258], gender [244,


6 Yu Kong, Yun Fu



257], etc. The human visual system is finely optimized
for the perception of human movements [38].
Action understanding by humans is a complex cognitive capability performed by a complex cognitive mechanism. Such a mechanism can be decomposed into three
major components, including action recognition, intention understanding, and narrative understanding [104].
Ricoeur [210] suggested that actions can be approached
with a set of interrelated questions including, who, what,
why, how, where, and when. Three questions are prioritized, which offer different perspectives on the action:
what is the action, why is the action being done, and
who is the agent. Computational models for the first
two questions have been extensively investigated in action recognition [14,193,26,170,127,90,255] and prediction [220,117,165,20] research in the computer vision
community. The last question “who is the agent” refers
to the agent’s identity, or social role, which provides
a more thoroughgoing understanding of the “who” behind it, and thus is referred to as narrative understanding [210]. Few work in the computer vision community
studies this question [131,204].
Some of the human actions are goal-oriented, i.e.,
a goal is completed by performing one or a series of
actions. Understanding such actions is crucial for predicting the effects or outcomes of the actions. As humans, we make inferences about the action goals of
an individual by evaluating the end state that would
be caused by their actions, given particular situational
or environmental constraints. The inference is possibly
made by a direct matching process of a mirror neuron
system, which maps the observed action onto our own
motor representation of that action [211,212]. According to the direct matching hypothesis, the prediction of
one’s action goal is heavily relying on the observer’s action vocabulary or knowledge. Another cue for making
action prediction is from emotional or attentional information, such as the facial expression and gaze or the
other individuals. Such referential information makes
the observer pay attention to the specific objects because of the particular relations that link these cues to
their referents. These psychological and cognitive findings would be helpful for designing action prediction
approaches.


**3 Action Recognition**


A typical action recognition flowchart generally contains two major components [227,265,199], action representation and action classification. The action representation component basically converts an action video
into a feature vector [133,41,267,228] or a series of vectors [185,118,179], and the action classification compo


**Fig. 6** Examples of an input video frame, the corresponding
motion energy image and motion history image computed by

[15].


nent infers an action label from the vector [152,239,
231]. Recently, deep networks [90,255,53] merge these
two components into a unified end-to-end trainable framework, which further enhance the classification performance in general. In this section we will discuss recent
work in action representation, action classification, and
deep networks.


3.1 Shallow Approaches


_3.1.1 Action Representation_


The first and the foremost important problem in action
recognition is _how to represent an action in a video_ . Human actions appearing in videos differ in their motion
speed, camera view, appearance and pose variations,
etc, making action representation a really challenging
problem. A successful action representation method should
be efficient to compute, effective to characterize actions,
and can maximize the discrepancy between actions, in
order to minimize the classification error.
One of the major challenges in action recognition is
large appearance and pose variations in one action category, making the recognition task difficult. The goal
of action representation is to convert an action video
into a feature vector, extract representative and discriminative information of human actions, and minimize the variations, thereby improving the recognition
performance. Action representation approaches can be
roughly categorized into holistic features and local features, which will be discussed next.
Many attempts have been made in action recognition to convert action videos into discriminative and

representative features, in order to minimize with-in

class variations and maximize between class variations.

Here, we focus on _hand-crafted_ action representation
methods, which means the parameters in these methods
are pre-defined by experts. This differs from deep networks, which can automatically learn parameters from
data.


_Holistic Representations_ Human action in a video generates a space-time shape in the 3D volume. This spacetime shape encodes both spatial information of the human pose at various times, and dynamic information of


Human Action Recognition and Prediction: A Survey 7


**Fig. 7** Examples of the original frame, optical flow, and the
flow field in four channels computed by [48].



the human body. Holistic representation methods capture the motion information of the entire human sub
ject, providing rich and expressive motion information
for action recognition. However, holistic representations
tend to be sensitive to noise. It captures the information
in a certain rectangle region, and thus may introduce irrelevant information and noise from the human subject
and cluttered background.
One pioneering work in [15] presented Motion Energy Image (MEI) and Motion History Image (MHI)
to encode dynamic human motion into a single image.
As shown in Figure 6, the two methods work on the
silhouettes. The MEI method shows “where” the mo
tion is occurring: the spatial distribution of motion is
represented and the highlighted region suggests both
the action occurring and the viewing condition. In addition to MEI, the MHI method shows both “where”
and “how” the motion is occurring. Pixel intensity on
a MHI is a function of the motion history at that location, where brighter values correspond to more recent

motion.

Although MEI and MHI showed promising results
in action recognition, they are sensitive to viewpoint
changes. To address this problem, [285] generalized [15]
to 3D motion history volume (MHV) to remove the
viewpoint dependency in the final action representation. MHV relies on the 3D voxels obtained from mul
tiple camera views, and shows the 3D occupancy in the
resulting volume. Fourier transform is then used to create features invariant to locations and rotations.

To capture space-time information in human actions, [69,14] utilized the Poisson equation to extract
various shape properties for action representation and
classification. Their method takes a space-time volume
as input. Then the method discovers space-time saliency
of moving body parts, and locally computes the orientation using the Poisson equation. These local properties
are finally converted into a global feature by weighted
averaging each point inside the volume. Another method
to describe shape and motion was presented in [313]. In
this method, a spatio-temporal volume is first generated
by computing correspondences between frames. Then,



**Fig. 8** Illustration of interest points detected on human
body. Revised based on the original figure in [79].


spatio-temporal features by analyzing differential geometric surface properties from the volume.
Instead of computing silhouette or shape for action
representation, motion information can also be computed from videos. One typical motion information is
computed by the so-called optical flow algorithms [161,
81,245], which indicate the pattern of apparent motion
of objects on two consecutive frames. Under the assumption that illumination conditions do not change
on the frames, optical flow computes the motion in the
horizontal and vertical axis. An early work by Efros _et_
_al._ [48] split the flow field into four channels (see Figure 7) capturing the horizontal and vertical motion in
successive frames. This method was then used in [283]
to describe the features of both the human body and
the body parts.


_Local Representations_ Local representations only identify local regions having salient motion information,
and thus inherently overcome the problem in holistic
representations. Successful methods such as space-time
interest points [41,134,108,17] and motion trajectory

[266,265] have shown their robustness to translation,
appearance variation, etc. Different from holistic features, local features describe the local motion of a person in space-time regions. These regions are detected
since the motion information within the regions is more
informative and salient than the surrounding areas. After detection, the regions are described by extracting
features in the regions.
Space-time interest points (STIPs) [134,133]-based
approaches is one of the most important local representations. Laptev’s seminal work [134,133] extended
the Harris corner detector [76] to space-time domain.
A spatio-temporal separable Gaussian kernel is applied
on a video to obtain its response function for finding
large motion changes in both spatial and temporal dimensions (see Figure 8). An alternative method was


8 Yu Kong, Yun Fu



proposed in [41], which detects dense interest points.
2D Gaussian smoothing kernel is applied only along
the spatial dimension, and the 1D Gabor filter is applied to the temporal dimension. Around each interest
point, raw pixel values, gradient, and optical flow features are extracted and concatenated into a long vector.
The principal component analysis is applied on the vector to reduce the dimensionality, and a k-means clustering algorithm is then employed to create the codebook of these feature vectors and generate one vector
representation for a video [227]. Bregonzio _et al._ [17]
detected spatial-temporal interest points using Gabor
filters. Spatiotemporal interest points can also be detected by using the spatiotemporal Hessian matrix [286].
Other detection algorithms detect spatiotemporal interest points by extending their counterparts of 2D detectors to spatiotemporal domains, such as 3D SIFT

[228], HOG3D [108], local trinary patterns [311], etc.
Several descriptors have been proposed to describe the
motion and appearance information within the small
region of the detected interest points such as optical
flow and gradient. Optical flow feature computed in a
local neighborhood is further aggregated in histograms,
called histograms of optical flow (HOF) [135], and combined with HOG features [34,108] to represent complex
human activities [108,135,269]. Gradients over optical
flow fields are computed to build the so-called motion
boundary histograms (MBH) for describing trajectories

[269].


However, spatiotemporal interest points only capture information within a short temporal duration and
cannot capture long-term duration information. It would
be better to track these interest points and describe
their changes of motion properties. Feature trajectory is
a straightforward way of capturing such long-duration
information [269,266,207]. To obtain features for trajectories, in [173], interest points are first detected and
tracked using Harris3D interest points with a KLT tracker

[161]. The method in [246] finds trajectories by matching corresponding SIFT points over consecutive frames.
Hierarchical context information is captured in this method
to generate more accurate and robust trajectory representation. Trajectories are described by a concatenation
of HOG, HOF and MBH features [266,265,88] (see Figure 9), intra- and inter-trajectory descriptors [246], or
HOG/HOF and averaged descriptors [207]. In order to
reduce the side effect of camera motion, [268,267] find
correspondences between two frames first and then use
RANSAC to estimate the homography.



**Fig. 9** Tracked point trajectories over frames, and are described by HOG, HOF and MBH features. Revised based on
the original figure in [265].


_3.1.2 Action Classifiers_


After action representations have been computed, action classifiers should be learned from training samples
that determine the class boundaries for various action
classes. Action classifiers can be roughly divided into
the following categories:


_Direct Classification_ This type of approaches summarize an action video into a feature vector, and then directly classify the vector into action categories using
off-the-shelf classifiers such as support vector machine

[227,135,170], k-nearest neighbor (k-NN) [14,137,256],
etc. In these methods, action dynamics are characterized in a holistic way using action shape [69,14], or
using the so-called bag-of-words model, which captures
local motion patterns using a histogram of visual words

[14,137,227,135,170].
In fact, bag-of-words approaches received lots of attention in the last few years. As shown in Figure 10,
these approaches first detect local salient regions using the spatiotemporal interest point detectors [41,227,
133,108]. Features such as gradient and optical flow are
extracted around each 3D interest point. The principal
component analysis is adopted to reduce the dimensionality of the features. Then the so-called visual words
can be computed by k-means clustering [227], or Fisher
vector [197]. Finally, an action can be represented by
a histogram of visual words, and can be recognized by
a classifier such as the support vector machine. The
bag-of-words model has been shown to be insensitive to
appearance and pose variations [269]. However, it does
not consider the temporal characteristics of human actions, as well as their structural information, which can
be addressed by sequential approaches [231,206] and
space-time approaches [217], respectively.


_Sequential Approaches_ This line of work captures temporal evolution of appearance or pose using sequential
state models such as hidden Markov models (HMMs)



Trajectory Description


_n_ _τ_



HOF



_n_ _σ_



_N_



HOG









MBH


approaches mainly use holistic features from frames,
which are sensitive to background noise and generally
do not perform well on challenging datasets.


_Space-time Approaches_ Although direct approaches have
shown promising results on some action datasets [227,
135,170], they do not consider the spatiotemporal correlations between local features, and do not take the potentially valuable information about the global spatiotemporal distribution of interest points into account.
This problem was addressed in [291], which learns a
global Gaussian mixture model (GMM) using the relative coordinates features, and uses multiple GMMs to
describe the distribution of interest points over local
regions at multiple scales. A global feature on top of
interest points was proposed in [318] to capture the detailed geometrical distribution of interest points. The
feature is computed by extended 3D discrete Radon
transform. Such a feature captures the geometrical information of the interest points, and is robust to geometrical transformation and noise. The spatiotemporal
distribution of interest points is described by a Directional Pyramid Co-occurrence Matrix in (DPCM) [319].
DPCM characterizes the co-occurrence statistics of lo
cal features as well as the spatio-temporal positional
relationships among the concurrent features. Graph is
a powerful tool for modeling structured objects, and it
was used in [289] to capture the spatial and temporal
relationships among local features. Local features are
used as the vertices of the two-graph model and the
relationships among local features in the intra-frames
and inter-frames are characterized by the edges. A novel
family of context-dependent graph kernels (CGKs) was
proposed in [289] to measure the similarity between the
two-graph models. Although the above methods have
achieved promising results, they are limited to small
datasets as the correlations between interest points in
their methods which are explosive on large datasets.


_Part-based Approaches_ Human bodies are structured
objects, and thus it is straightforward to model human actions using motion information from body parts.







Part-based approaches consider motion information from
both the entire human body as well as body parts. The
benefit of this line of approaches is it inherently captures the geometric relationships between body parts,
which is an important cue for distinguishing human actions. A constellation model was proposed in [51], which
models the position, appearance and velocity of body
parts. Inspired by [51], a part-based hierarchical model
was presented in [186], in which a part is generated by
the model hypothesis and local visual words are generated from a body part (see Figure 11).
The method in [288] considers local visual words as
parts, and models the structure information between
parts. This work was further extended in [187], where
the authors assume an action is generated from a multinomial distribution, and then each visual word is generated from distribution conditioned on the action. These

part-based generated models were further improved by
discriminative models for better classification performance [282,283]. In [282,283], a part is considered as a
hidden variable in their models. It is corresponding to
a salient region with the most positive energy.


_Manifold Learning Approaches_ Human action videos
can be described by temporally variational human silhouettes. However, the representation of these silhouettes is usually high-dimensional and prevents us from
efficient action recognition. To solve this problem, manifold learning approaches were proposed in [275,92] to
reduce the dimensionality of silhouette representation
and embed them on nonlinear low-dimensional dynamic
shape manifolds. The method in [275] adopts kernel
PCA to perform dimensionality reduction, and discover
the nonlinear structure of actions in the manifold. Then,
a two-chain factorized CRF model is used to classify
silhouette features in the low-dimensional space into
human actions. A novel manifold embedding method
was presented in [92], which finds the optimal embedding that maximizes the principal angles between temporal subspaces associated with silhouettes of different
classes. Although these methods tend to achieve very
high performance in action recognition, they heavily


10 Yu Kong, Yun Fu


**Interactive Phrases**


**Arms:** A chest-level moving arm and a
free swinging arm
**Torsos:** A leaning forward torso and a
leaning backward torso
**Legs:** A stepping forward leg and a
stepping backward leg



**Fig. 11** Example of body parts detected by the constellation
model in [186]. Revised based on the original figure in [186].


rely on clean human silhouettes which could be difficult to obtain in real-world scenarios.


_Mid-Level Feature Approaches_ Bag-of-words models have
shown to be robust to background noise but may not be
expressive enough to describe actions in the presence
of large appearance and pose variations. In addition,
they may not well represent actions due to the large
semantic gap between low-level features and high-level
actions. To address these two problems, hierarchical approaches [283,27,152,116] are proposed to learn an additional layer of representations, and expect to better
abstract the low-level features for classification.
Hierarchical approaches learn mid-level features from
low-level features, which are then used in the recognition task. The learned mid-level features can be consid
ered as knowledge discovered from the same database
used for training or being specified by experts. Recently,
semantic descriptions or attributes (see Figure12) are
popularly investigated in action recognition. These semantics are defined and further introduced into the activity classifiers in order to characterize complex human
actions [115,116,152]. Other hierarchical approaches such
as [206,260] select key poses from observed frames, which
also learn better action representations during model
learning. These approaches have shown superior results
due to the use of human knowledge, but require extra
annotations which is labor-intensive.


_Feature Fusion Approaches_ Fusing multiple types of
features from videos is a popular and effective way for
action recognition. Since these features are generated
from the same visual inputs, they are inter-related. However, the inter-relationship is complicated and is usually
ignored in the existing fusion approaches. This problem
was addressed in [162], in which the maximum margin distance learning method is used to combine global
temporal dynamics and local visual spatio-temporal appearance features for human action recognition. A MultiTask Sparse Learning (MTSL) model was presented
in [317] to fuse multiple features for action recognition. They assume multiple learning tasks share priors,



**Fig. 12** Interaction recognition by learning semantic descriptions from videos. Revised based on the original figure in

[116].


one for each type of features, and exploit the correlations between tasks to better fuse multiple features. A
multi-feature max-margin hierarchical Bayesian model
(M3HBM) was proposed in [303] to learn a high-level
representation by combining a hierarchical generative
model (HGM) and discriminative max-margin classifiers in a unified Bayesian framework. HGM represents
actions by distributions over latent spatial temporal
patterns (STPs) learned from multiple feature modalities. This work was further extended in [320] to combine spatial interest points with context-aware kernels
for action recognition. Specifically, a video set is modeled as an optimized probabilistic hypergraph, and a
robust context-aware kernel is used to measure high order relationships among videos.


_3.1.3 Classifiers for Human Interactions_


Human interaction is typical in daily life. Recognizing
human interactions focuses on the actions performed by
multiple people, such as “handshake”, “talking”, etc.
Even though some of the early work such as [135,217,
316,170,153] used action videos containing human interactions, they recognize actions in the same way as
single-person action recognition. Specifically, interactions are treated as a whole and are represented as a
motion descriptor including all the people in a video.
Then an action classifier such as a linear support vector machine is adopted to classify interactions. Despite
reasonable performance has been achieved, these approaches do not explicitly consider the intrinsic methods of interactions, and fail to consider the co-occurrence
information between interacting people. Furthermore,
they do not extract the motion of each person from the
group, and thus their methods can not infer the action
label of each interacting person.
Action co-occurrence of individual person is a piece
of valuable information in human interaction recognition. In [189], action co-occurrence is captured by coupling motion state of one person with the other interaction person. Human interactions such as “hug”, “push”,


Human Action Recognition and Prediction: A Survey 11



and “hi-five” usually involve frequent close physical contact, and thus some body parts may be occluded. To
robustly find body parts, Ryoo and Aggarwal [216] utilized body part tracker to extract each individual in
videos and then applied context-free grammar to model
spatial and temporal relationships between people. A
human detector is adopted in [192] to localize each individual. Spatial relationships between individuals are
captured using the structured learning technique [55].
Spatiotemporal context of a group of people including
human pose, velocity and spatiotemporal distribution
of individuals is captured in [27] to recognize human
interactions. Their method shows promising results on
collective actions without close physical contact such
as “crossing the road”, “talking”, or “waiting”. They
further extended their work that can simultaneously
track and recognize human interactions [25]. A hierarchical representation of interactions is proposed in

[25] that models atomic action, interaction, and collective action. The method in [132] also utilizes the
idea of hierarchical representation, and studies the collective activity recognition problem using crowd context. Different from these methods, the work in [260]
represents individuals in interactions as a set of key
poses, and models spatial and temporal relationships of
the key poses for interaction recognition. In our earlier
work [116,115], a semantic description-based approach
is proposed to represent complex human interactions by
learned motion relationships (see Figure 12). Instead of
directly modeling action co-occurrence, we propose to
learn phrases that describe the motion relationships between body parts. This will describe complex interactions in more details, and introduce human knowledge
into the model. All these methods may not perform well
in interactions with close physical contact due to the
ambiguities in feature-to-person assignments. To address this problem, a patch-aware model was proposed
in [110] to learn discriminative patches for interaction
recognition, and determine the assignments at a patch
level.


_3.1.4 Classifiers for RGB-D Videos_


Action recognition from RGB-D videos has been receiving a lot of attentions [270,271,75,294,156,190] due
to the advent of the cost-effective Kinect sensor [232].
RGB-D videos provide an additional depth channel compared with conventional RGB videos, allowing us to
capture 3D structural information that is very useful in
reducing background noise and simplifying intra-class
motion variations [184,284,190,75,188].
Effective features have been proposed for the recognition task using depth data, such as histogram of ori


ented 4D normals [190,305] and depth spatiotemporal
interest points [294,75]. Features from depth sequences
can be encoded by [163], or be used to build actionlets

[284] for recognition. An efficient binary range-sample
feature for depth data was proposed in [160]. This binary depth feature is fast, and has shown to be invariant to changes in scale, viewpoint, and background. The
work in [249,123] built layered action graph structures
to model actions and subactions in a RGB-D video.
Recent work [156] also showed that features of RGB-D
data can be learned using deep learning techniques.
The methods in [145,190,305,75,270,163] only use
depth data, and thus would fail if depth data were missing. Joint use of both RGB and depth data for action
recognition is investigated in [82,91,151,156,284,111].
However, they only learn features shared between the
two modalities and do not learn modality-specific or
private features. To address this problem, shared features and privates features are jointly learned in [113],
which learns extra discriminative information for classification, and demonstrate superior performance than

[82,91,151,156,284,111]. The methods in [111,113] also
show that they can achieve high recognition performance even though one modality is missing in training
or testing.
Auxiliary information has also shown to be useful
in RGB-D action recognition. Skeleton data provided
by a Kinect sensor was used in [82,284,113], and has
shown to be very effective in action recognition. The
method in [82] learns a shared feature space for various types of features including skeleton features and local HOG features, and project these features onto the
shared space for action recognition. Different from this
work, the method in [113] jointly learns RGB-D and
skeleton features and action classifiers. The projection
matrices in [113] are learned by minimizing the noise
after projection and classification error using the projected features. Using auxiliary databases to improve
the recognition performance was studied in [91,151], in
which actions are assumed to be reconstructed by entries in the auxiliary databases.


3.2 Deep Architectures


Although great success has been made by global and local features, these hand-crafted features require heavy
human labor and domain expert knowledge to develop
effective feature extraction methods. In addition, they
normally do not generalize very well on large datasets.
In recent years, feature learning using deep learning
techniques has been receiving increasing attention due
to their capability of learning powerful features that


12 Yu Kong, Yun Fu




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||


|Wt<br>t|Col2|
|---|---|
|t<br>W<br>**t**||







volution.


can be generalized very well [90,255,42,237]. The success of deep networks in action recognition can also be
attributed to scaling up the networks to tens of millions of parameters and massive labeled datasets. Recent deep networks [261,255,53,97] have achieved surprisingly high recognition performance on a variety of
action datasets.

Action features learned by deep learning techniques
has been popularly investigated [308,272,254,247,198,
138,99,90,89,77,10,237] in recent years. The two major
variables in developing deep networks for action recognition are the convolution operation and temporal modeling, leading to a few lines of networks.

The convolution operation is one of the fundamental components in deep networks for action recognition, which aggregates pixel values in a small spatial
(or spatiotemporal) neighborhood using a kernel matrix. **2D vs 3D Convolution:** 2D convolution over
images (Figure 13(a)) is one of the basic operation in
deep networks, and thus it is straightforward to use 2D
convolution on video frames. The work in [99] presented
a single-frame architecture based on a 2D CNN model,
and extracted a feature vector for each frame. Such a
2D convolution network (2D ConvNet) also enjoys the
benefit of using the networks pre-trained on large-scale
image datasets such as ImageNet. However, 2D ConvNets do not inherently model temporal information,
and requires an additional aggregation or modeling of
such information.

As multiple frames are presenting in videos, 3D convolution (Figure 13(b)) is more intuitive to capture
temporal dynamics in a short period of time. Using
3D convolution, 3D convolutional networks (3D ConvNets) directly create hierarchical representations of
spatio-temporal data [89,90,254,255]. However, the issue is they have many more parameters than 2D ConvNets, making them hard to train. In addition, they
are prevented from enjoying the benefits of ImageNet
pre-training.

Another key variable in designing deep networks
is **Temporal Modeling** . Generally, there are roughly
three methods in temporal modeling. One straightfor


trains on optical flow frames, which essentially capture
motion information in the adjacent two frames. However, these approaches largely disregard the long-term
temporal structure of videos. 2D convolution is usually used in these approaches, and thus they can easily exploit the new ultra-deep architectures and models
pre-trained for still images. The third category of approaches uses temporal pooling [97,66] or aggregation
to capture temporal information in a video. The aggregation can be performed by using a LSTM model on
top of 2D ConvNets [42,183].


_3.2.1 Space-time Networks_


Space-time networks are straightforward extensions of
2D ConvNets as they capture temporal information using 3D convolutions.
The method in [89] was one of the pioneering works
in using convolution neural networks (CNN) for action
recognition. They perform 3D convolutions over adjacent frames, and thus extract features from both spatial and temporal dimensions. Their 3D CNN network
architecture starts with 5 hardwired kernels including
gray, gradient-x, gradient-y, optflow-x, and optflow-y,
resulting in 33 feature maps. Then the network repeats
3D convolution and subsampling, and uses a fully-connected
layer to generate a 128-dimensional feature vector for
action classification. In a later extension [90], the authors regularized the network to encode long-term action information by encouraging the network to learn
feature vector close to high-level motion features such
as the bag-of-words representation of SIFT features.
The 3D ConvNet [89,90] was later extended to a
modern deep architecture called C3D [255] that learns
on large-scale datasets. The C3D network contains 5
convolution layers, 5 max-pooling layers, 2 fully-connected
layers, and a softmax loss layer, subject to the machine
memory limit and computation affordability. Their work
demonstrated that C3D learns a better feature embedding for videos (see Figure 14). Results showed that the
C3D method with a linear classifier can outperform or
approach the state-of-the-art methods on a variety of
video analysis benchmarks including action recognition
and object recognition.


Human Action Recognition and Prediction: A Survey 13













































Imagenet C3D


**Fig. 14** Feature embedding by Imagenet and C3D. C3D features show better class separation than Imagenet, indicating
its capability in learning better features for videos. Originally
shown in [255].


Still, 3D ConvNets [89,90,255] for action recognition are relatively shallow with up to 8 layers. To further improve the generalization power of 3D ConvNets,

[21] inflated very deep networks for image classification into spatio-temporal feature extractors by repeating 2D filters along the time dimension, allowing the
network to reuse 2D filters pre-trained on ImageNet.
This work also shows that pre-training on the Kinetics
dataset achieves better recognition accuracy on UCF101 and HMDB51 datasets. Another solution to build
a deep 3D ConvNet was proposed in [201], which uses a
combination of one 1 _×_ 3 _×_ 3 convolutional layer and one
3 _×_ 1 _×_ 1 convolutions to take the place of a standard

3D convolution.

One limitation of 3D ConvNets is that they typically consider very short temporal intervals, such as
16 frames in [255], thereby failing to capture long-term
temporal information. To address this problem, [261]
increases the temporal extent in the 3D convolutions,
and empirically shows that they can significantly improve the recognition performance.


_3.2.2 Multi-Stream Networks_


Multi-stream networks utilize multiple convolutional networks to model both appearance and motion information in action videos. Even though the network in [99]
achieved great success, its results were significantly worse
than those of the best hand-crafted shallow representations [267,265]. To address this problem, a successful
work by [237] explored a new architecture related to the
two-stream hypothesis [68]. Their architecture contains
two separate streams, a spatial ConvNet and a temporal ConvNet (see Figure 15). The former one learns
actions from still images, and the latter one performs
recognition based on the optical flow field.
The two-stream network [237] directly fuses the outputs of the two streams generated by their respective
softmax function, which may not be appropriate for



**Fig. 15** Two-stream network proposed in [237] contains a
spatial network and a temporal network, which are used for
modeling static information in still frames and motion information in optical flow images, respectively. Revised based on
the original figure in [237].


gathering information over a long period of time. An
improvement was proposed in [274], which used the twostream network to obtain multi-scale convolutional fea
ture maps, and pooled the feature maps together with
the detected trajectories to compute ConvNet responses
centered at the trajectories. Such a scheme encodes
deep features into effective descriptors constrained by
sampled trajectories. Temporal feature pooling in the
two-stream network was investigated in [183], which is
capable of making video-level predictions after the pooling layer. The work in [66] also presented a novel pooling layer named ActionVLAD that aggregates convolutional feature descriptors in different image portions
and temporal spans. They also used ActionVLAD to
combine appearance and motion streams together. The
network named temporal linear encoding [40] aggregates temporal features sampled from a video, and then
projects onto a low-dimensional feature space. By doing so, long-range temporal structure in different frames
can be captured and be encoded into a compact representation. AdaScan proposed in [97] evaluated the importance of the next frame, so that only informative
frames will be pooled, and non-informative frames will
be disregarded in the video-level representation. Their
AdaScan method uses a multilayer perceptron to compute the importance for the next frame given temporally pooled features up to the current frame. The importance score will then be used as a weight for the feature pooling operation for aggregating the next frame.
Despite effective, most of the feature encoding methods lack of considering spatio-temporal information. To
address this problem, the work in [46] proposed a new
feature encoding method for deep features. More specifically, they proposed locally max-pooling that groups
features according to their similarity and then performs
max-pooling. In addition, they performed max-pooling
and sum-pooling over the positions of features to achieve
spatio-temporal encoding.
Temporal sampling in the two-stream network was
proposed in Temporal Segment Networks (TSN) [278].
In TSN long-range dynamics are gathered by analyz

14 Yu Kong, Yun Fu



ing short video snippets formed from randomly sampled
frames from segments of the full video. The idea here is
that directly analyzing densely sampled video sequence
makes no sense since the consecutive frames in the video

contain a lot of redundancy. Moreover, some actions
reveal them-self at different temporal scales, such as
sprinting, which requires multiple actions over a long
span of time, compared to just crouching. The original
TSN network [278], was based on two-stream architecture from [237]. The prediction from temporal segments
was summaries by applying consensus function to frame
features extracted with pre-trained Deep CNN classification network. As for consensus function was used a

simple pooling operation. The advantage of this network is that it can enjoy the benefits of using big pretrained classification networks for feature extraction. To
improve the performance of temporal sampling in [332]
was suggested to perform sampling at different temporal scales, and substitute pooling operation with a
fully connected network, which should encode the temporal ordering of frames. The TSN can be also incorporated into another action recognition frameworks as
illustrated in [202]. Recently, Liu _et al._ [157] attempted
to use all video frames for classification by clustering
the activations along the temporal dimension based on
the assumption that similar frames should have similar
activation values. However, this method is limited in its
ability of dynamically selecting the number of clusters.
Wang _et al._ [276] proposed Temporal Difference Network (TDN) which aims to recognize actions from the
entire video. TDN contains short-term temporal difference modules to encode local motion information and
long-term temporal difference modules to capture motion across segments.


One of the major problems in the two-stream networks [237,274,183] is that they do not allow interactions between the two streams. However, such an interaction is really important for learning spatiotemporal features. To address this problem, Feichtenhofer _et_
_al._ [54] proposed a series of spatial fusion functions that
make channel responses at the same pixel position be
in the same correspondence. These fusion layers are
placed in the middle of the two-streams allowing interactions between them. They further injected residual
connections between the two streams [52,53], and allow
a stream to be multiplicatively scaled by the opposing
stream’s input [53]. Such a strategy bridges the gap between the two streams, and allows information transfer
in learning spatiotemporal features.















**Fig. 16** Network architecture of LRCN [42] with a hybrid of
ConvNets and LSTMs. Revised based on the original figure
in [42].


_3.2.3 Hybrid Networks_


Another solution to aggregate temporal information is
to add a recurrent layer on top of the CNNs, such as
LSTMs, to build hybrid networks [42,183]. Such hybrid networks take the advantages of both CNNs and
LSTMs, and thus have shown promising results in capturing spatial motion patterns, temporal orderings and
long-range dependencies [274,40,97].
Donahue _et al._ [42] explored the use of LSTM in
modeling time series of frame features generated by 2D
ConvNets. As shown in Figure 16, the recurrence nature of LSTMs allows their network to generate textual
descriptions of variable lengths, and recognize human
actions in the videos. Ng _et al._ [183] compared temporal
pooling and using LSTM on top of CNNs. They discussed six types of temporal pooling methods including
slow pooling and Conv pooling, and empirically showed
that adding a LSTM layer generally outperforms temporal pooling by a small margin because it capture the
temporal orderings of the frames. A hybrid network using CNNs and LSTMs was proposed in [292]. They used
two-stream CNN [237] to extract motion features from
video frames, and then fed into a bi-directional LSTM
to model long-term temporal dependencies. A regularized fusion scheme was proposed in order to capture the
correlations between appearance and motion features.

Hybrid networks have also been applied to skeletonbased action recognition. Skeleton data can be easily
obtained by depth sensors such as Kinect or pose estimation algorithms. In these methods, hybrid deep neural networks [229,335,155,101,301] are developed to model
the structure information of various body joints as well
as temporal information of body movement. Recurrent


Human Action Recognition and Prediction: A Survey 15



neural networks are widely used to capture the features consisting of ordered joints [229,335,155]. Temporal CNN [101] is also applied to capture the features
of structured body joints. Recently, graph convolution
networks have shown superior performance over RNNs
and Temporal CNNs, and become the backbone for
capturing the structural information of joints. Yan _et_
_al._ [301] proposed a spatio-temporal graph convolution
to learn the structural and temporal information at the
same time. Si _et al._ [236] applied GCN-LSTM to model
the temporal dependencies of skeleton and proposed an
attention model to learn the importance of each joint.


3.3 Learning with Limited Data/Label


Due to the necessity of training deep neural networks,
recent video are becoming extremely large. For example, Youtube-8M dataset [4] consists of over 8 million
videos. For such large-scale datasets, it is expensive and
almost impossible to annotate all the video data. Even
though search engines were given action labels and were
used to retrieve videos, they also make mistakes and
thus the compiled video data could be noisy. One solution is to learn action models in a weakly-supervised
fashion or an unsupervised fashion. Therefore, the models do not necessarily require fully-annotated video data
and can learn under very limited or no supervisory signals. Few-shot learning was also recently introduced to
learn in the low-sample regime.


_3.3.1 Weakly-supervised Action Learning_


Weakly-supervised learning methods [136,16,64] are developed to deal with the scenarios where each of the
videos is not fully annotated. One promising application
scenario is to understand human actions in untrimmed

videos, in which the temporal boundaries of various actions in the videos are not annotated. Such a learning
capability enhances most of the existing action recognition methods [255,119,237,274,183], as require all the
action videos to be trimmed which is expensive and
time-consuming to achieve.
Movie with script data is a typical scenario to evaluate weakly-supervised action learning methods. Pioneering work made by Laptev _et al._ [136] presented a
novel realistic action dataset from movies. Annotations
were made using movie scripts. Duchenel _et al._ [44] followed this work and addressed the problem of weaklysupervised learning of action models and localizing action instances in videos given the corresponding movie
scripts.
Another type of work is weak-supervised action understanding given a temporally ordered list of action



classes that will appear in the video. For example, Bojanowski _et al._ [16] formulated the problem as a weakly
supervised temporal assignment and proposed a clustering method that assigns the action labels to the temporal segments in videos. Huang _et al._ [84] adapted
the Connectionist Temporal classification model from
speech recognition to perform weakly-supervised action
labeling.
Recent works have extended weakly-supervised action representation learning to untrimmed videos with
unordered action lists. Wang _et al._ [277] proposed the
UntrimmedNet for untrimmed video understanding by
learning action models and reasoning temporal duration of action instances in an end-to-end framework.
Ghadiyaram _et al._ [64] took advantage of large-scale
noisy labeled web videos to learn a pre-trained model
for video action recognition.


_3.3.2 Unsupervised and Self-supervised Action_
_Learning_


Unsupervised or self-supervised representation learning
is becoming popular in recent years as it allows deep
neural networks to be pre-trained utilizing the supervisory signals within the training data, rather than given
by humans. Such pre-trained models can be beneficial
for downstream tasks, such as action recognition and localization. Many attempts have leveraged the temporal
coherence, motion consistency and temporal continuity
as supervision, which will be discussed below.
The chronological order of frames is a typical free
supervision signal for videos. Action models learn to tell
whether the frame sequence is ordered or not, given either shuffled or unshuffled videos [176,56]. Another related task is training the model to tell the actual order
of the shuffled video frames [139]. Xu _et al._ [297] extended the order prediction tasks from frames to clips.
This helps to train a 3D CNN framework using chronological order supervision. Buchler _et al._ [18] applied
deep reinforcement learning to sample new permutations according to their expected utility to adapts to
the state of the network.

The motion of objects in videos can also be used as
supervision. Wang _et al._ [280] found the corresponding
pairs using visual tracking, based on Siamese-Triplet
network. Purushwalkam _et al._ [200] utilized pose as
free supervision since similar pose should have similar motion. Wang _et al._ [281] explored different selfsupervised methods to learn the representations invariant to the variations between the object patches, which
is extracted by motion cues. Gan _et al._ [61] used geometry cues flow field and disparity maps to learn the
video representations.


16 Yu Kong, Yun Fu


**Table 1** Pros and cons of action recognition approaches.


|Col1|Approaches|Pros|Cons|
|---|---|---|---|
|Shallow|Direct [227,268]<br>Sequential [179,231]<br>Space-time [289,291]<br>Part-based [283,185]<br>Manifold [275,92]<br>Mid-level feature [27,152]<br>Feature fusion [317,320]|Easy and quick to use.<br>Models temporal evolution.<br>Captures spatiotemporal structures.<br>Models body parts at a ﬁner level.<br>Tend to achieve high performance.<br>Introduce knowledge to models.<br>Tend to achieve high performance.|Performance is limited.<br>Sensitive to noise.<br>Limited to small datasets.<br>Limited to small datasets.<br>Rely on human silhouettes.<br>Require extra annotations.<br>Slow in feature extraction.|
|Deep|Space-time [90,255]<br>Multi-stream [237,53]<br>Hybrid [42,183]|Natural extension of 2D convolution.<br>Able to use pre-trained 2D ConvNets.<br>Easy to build using existing networks.|Short temporal interval.<br>Int. b/w networks is diﬃcult.<br>Diﬃcult to ﬁne-tune.|



_3.3.3 Few-shot Learning_


Few-shot learning aims at learning reliable models from
minimalist data sets. In extreme cases, there could be
no training sample for some categories which is called
the zero-shot learning. Majority of few-shot works target at recognising images, while only a few address
the video action recognition challenge. [334] proposed
a compound memory network (CMN) which predicts
the unseen video by retrieving a similar video stored in
the memory of the CMN architecture. ProtoGAN [47]
learns the class-prototype vectors through a feature aggregator network called _Class Prototype Transfer Net-_
_work_ (CPTN), then generates additional video features
for the recognition classifier. Neural Graph Matching
(NGM) network [73] is a graph-based approach that
generates graph representations for 3D action videos
and match unseen videos and seen videos by the similarity of their graph representations. [175] proposes a
framework for zero-shot action recognition which models each action class as a probability distribution and
the distribution parameters are a linear combination
of the attributes of the action class. The weights of
the attributes are learnt from the labeled samples. One
challenge in few-shot action recognition is the variation of temporal lengths. Temporal Attentive Relation
Network (TARN) [12] uses attention modules to align
video segments and learns a distance measure between
the aligned representations for few-shot and zero-shot
learning. Action Relation Network (ARN) [327] encodes
the video clips features of the query set and support
set into a Power Normalized Autocorrelation Matrix
(AM) from which a relation network learns to captures the relations. Similar to ARN, Ordered Temporal Alignment Module (OTAM) [19] extracts per-frame
feature through an embedding network, then computes
an alignment score of the distance matrix. TemporalRelational CrossTransformers (TRX) [196] classifies the
query video by matching each sub-sequence to all subsequences in the support set using CrossTransformer
attention modules.



3.4 Summary


Deep networks are dominant in action recognition research but shallow methods are still useful. Compared
with deep networks, shallow methods are easy to train,
and generally perform well on small datasets. Recent
shallow methods such as improved dense trajectory with
linear SVM [268] have also shown promising results on
large datasets, and thus they are still popularly used recently in the comparison with deep networks [255,261,
53]. It would be helpful to use shallow approaches first
if the datasets are small, or each video exhibits complex
structures that need to be modeled. However, there are
lots of pre-trained deep networks on the Internet such
as C3D [255] and TSN [278] that can be easily employed. It would be also helpful to try these methods
and fine-tune the models to particular datasets. Table 1
summarizes the pros and cons of action recognition approaches.


**4 Action Localization and Detection**


In order to recognize and predict an action, the machine needs to know where is the action in a video. This

is achieved by action localization and detection, which
find out the spatiotemporal regions containing certain
human actions in videos. Both of the two tasks have

attracted a large amount of research in recent years. As
an analogy to object localization and detection in the
image domain, action detection is additionally required
to identify the action type of each action that occurs
in the video compared to the action localization. Based
on the feature learning paradigms, related work can be
categorized into shallow and deep learning methods, for
which we will make a comprehensive literature review.

Table 2 summarizes some recent detection methods and

compares results on thresholds of 0.3, 0.4, and 0.5. The
mAP@ _α_ denotes the mean Average Precision at different IOU threshold which measures the average prevision on each action category.


Human Action Recognition and Prediction: A Survey 17



**Table 2** Results of action detection methods on THUMOS’14 [93]. The mAP@ _α_ denotes the mean Average Precision at different threshold, _α_ . “-” indicates the result is not

|reported.|Col2|Col3|Col4|
|---|---|---|---|
||mAP@0.5|mAP@0.4|mAP@0.3|
|End-to-End [312]|17.1|26.4|36.0|
|Multi-stage [234]|19.0|28.7|36.3|
|TURN [62]|24.5|35.3|46.3|
|Temporal Context<br>Network [33]|25.6|33.3|-|
|Single-stream<br>R-C3D [298]|28.9|35.6|44.8|
|SSN [331]|29.8|41.0|51.9|
|Single-stream<br>R-C3D+OHEM [299]|35.8|43.1|51.1|
|Two-stream<br>R-C3D [299]|36.1|43.0|51.2|
|BSN [150]|36.9|45.0|53.5|
|MGG UNet [158]|37.4|46.8|53.9|
|BMN [149]|38.8|47.4|56.0|



4.1 Shallow Approaches


Early work [98,273] formulated action detection as a
classification task by firstly using temporal segmentation or sliding window methods. In these work, the
untrimmed video is segmented into short video clips and
the multiple features are extracted for classifiers such as
support vector machine (SVM) to recognize the action
types. Eventually, the actions that appear in the video
as well as their temporal locations are determined. Jian
_et al._ [87] proposed to generate a set of bounding boxes
from the video which are called tubelets for action localization. However, these methods suffer from handcraft
feature engineering and multi-stage model tuning, leading to quite inaccurate detection results.


4.2 Deep Architectures


Recent approaches to action localization and detection
make full use of deep neural networks for learning better video feature representation. To this end, Shou _et_
_al._ [234] proposed to first generate action proposals
from the long videos. Then, a localization network is
introduced to fine-tune the trained action classification network to recognize the action labels. The idea of
their action proposals inspired many later research [49,
233,277,331,298,62,22]. For these methods, Escorcia _et_
_al._ [49] proposed a deep action proposals (DAP) method
which achieves high efficiency and demonstrates to have
good generalization capability. To detect human actions
in frame-level granularity, Shou _et al._ [233] proposed an
end-to-end learning framework in which a CDC convolutional filter is designed on top of 3D ConvNet. To



model the temporal structure of each action instance,
Zhao _et al._ [331] proposed a structured segment network (SSN) with a temporal pyramid and a dubbed
temporal actionness grouping (TAG) model for action
proposals generation. As the action detection is similar to the object detection, Chao _et al._ [22] revisited
the most widely-used object detection method Faster
R-CNN and propose a temporal action localization network (TAL-Net) to address the unsolved challenges, including the large variation of action durations, temporal context modeling, and multi-stream feature fusion. Song _et al._ [241] noted that the ambiguous transition states of an action and long-term temporal context
are critical for accurate action detection. Thus, they
propose a transition-aware context network and it is
demonstrated to be significantly effective for untrimmed
video dataset. To modeling the relations among action
proposals, Zeng _et al._ [324] recently proposed to introduce the graph convolutional neural networks (GCN)
for temporal action localization. Song _et al._ [240] introduced the action pattern tree (AP-Tree) in which
the temporal information can be utilized. Inspired by
the conventional idea of coarse-to-fine detection, Yang
_et al._ [306] proposed a spatio-temporal progressively
learning method for video action detection, achieving
remarkable performance on existing benchmarks. Recently, Xu _et al._ [300] raised the importance of online
action detection and propose a temporal recurrent network (TRN) by simultaneously performing online action detection and anticipation, significantly outperforming the state-of-the-art. Chen _et al._ [24] unified the
tasks of actor localization and action classification into
the same backbone, which reduces model complexity
and improves efficiency compared to SOTA methods.
Li _et al._ [147] designed two auxiliary pretext tasks to
recycle the limited labeled data and benefit both features extraction as well as prediction.


Different from previous full-supervised methods that
require large-scale frame-level annotations of action instances, weakly-supervised methods need only the videoor clip-level action annotations so that they are more
promising in practice. Wang _et al._ [277] proposed a
weakly-supervised action detection model that is directly learned on the untrimmed video data, achieving
performance on-par-with those of the full-supervised
action detection methods. Recently, Yu _et al._ [315] introduced the temporal structure mining (TSM) approach
to the weakly-supervised action detection problem. In
their method, an action instance is modeled as a multiphase process so that the phase filters can be utilized to
compute the confidence score, indicating the action occurrence probability. For weakly-supervised action localization problem, it also attracts much attention in


18 Yu Kong, Yun Fu



recent years. Gao _et al._ [174] proposed a weakly supervised framework that consists of two modules, one module generates the pseudo ground truth of action boundaries which are used to supervise the action recognition module. Yang _et al._ [304] proposed to incorporate
the uncertainty for reducing the noise in the generated
pseudo labels. To handle the challenge of limited temporal annotations, Yang _et al._ [302] used an one-shot
learning technique of matching network for temporal
action localization. Narayan _et al._ [181] introduced a
novel loss function comprising the action classification
loss, multi-label center loss, and the counting loss, setting the new state-of-the-art on weakly-supervised action localization.

In addition to using visual data, other data modalities such as skeleton and RGB-D data can also be

utilized for temporal action localization and detection.
To learn the features of discriminative skeleton joints,
Song _et al._ [242] introduced a spatio-temporal attention LSTM model for action recognition and detection. To handle the modality discrepancy in a multimodal setting, Luo _et al._ [164] proposed a graph distillation method that privileged information is learned
from a large-scale multi-modal dataset in the source domain and their model can be effectively deployed to the
modality-scarce target domain. For the continuous action stream scenario, Dawar _et al._ [37] designed a multimodal fusion system to incorporate depth camera data
and wearable inertial sensor signals for action detection.


**5 Action Prediction**


After-the-fact action recognition has been extensively
studied in the last few decades, and fruitful results have
been achieved. State-of-the-art methods [42,66,278] are
capable of accurately giving action labels after observing the entire action executions. However, in many realworld scenarios ( _e.g._, vehicle accident and criminal activity), intelligent systems do not have the luxury of
waiting for the entire video before having to react to the
action contained in it. For example, being able to predict a dangerous driving situation before it occurs; opposed to recognizing it thereafter. In addition, it would
be great if an autonomous driving vehicle could predict
the motion trajectory of a pedestrian on the street and
avoid the crash, rather than identify the trajectory after the crash into the pedestrian. Unfortunately, most
of the existing action recognition approaches are unsuitable for such early classification tasks as they expect to
see the entire set of action dynamics from a full video,
and then make decisions.



**Beginning** **Current**

Past Future


**?** **?**


Partially observed action video Unobserved action video


**Fig. 17** Early action classification methods predicts action
label given a partially observed video. Revised based on the
original figure in [117].


Different from action recognition approaches, action
or motion prediction [1] approaches reason about the future and infer labels before action executions end. These

labels could be the discrete action categories, or continuous positions on a motion trajectory. The capability of
making a prompt reaction makes action/motion prediction approaches more appealing in time-sensitive tasks.
However, action/motion prediction is really challenging
because accurate decisions have to be made on partial

action videos.


5.1 Action Prediction


Action prediction tasks can be roughly categorized into
two types, _short-term prediction_ and _long-term predic-_
_tion_ . The former one, short-term prediction focuses on
short-duration action videos, which generally last for
several seconds, such as action videos in UCF-101 and
Sports-1M datasets. The goal of this task is to infer
action labels based upon temporally incomplete action
videos. Formally, given an incomplete action video **x** 1: _t_
containing _t_ frames, i.e., **x** 1: _t_ = _{f_ 1 _, f_ 2 _, · · ·, f_ _t_ _}_, the goal
is to infer the action label _y_ : **x** 1: _t_ _→_ _y_ . Here, the incomplete action video **x** 1: _t_ contains the beginning portion of
a complete action execution **x** 1: _T_, which only contains
one single action. The latter one, long-term prediction
or intention prediction, infers the future actions based
on current observed human actions. It is intended for

modeling action transition, and thus focuses on longduration videos that last for several minutes. In other

words, this task predicts the action that is going to happen in the future. More formally, given an action video
**x** _a_, where **x** _a_ could be a complete or an incomplete
action execution, the goal is to infer the next action
**x** _b_ . Here, **x** _a_ and **x** _b_ are two independent, semantically
meaningful, and temporally correlated actions.


1 In this paper, action prediction refers to the task of predicting action category, and motion prediction refers to the
task of predicting motion trajectory. Video prediction is not
discussed in this paper as it focuses on motion in videos rather
than motion of human.


Human Action Recognition and Prediction: A Survey 19


**Table 3** Results of early action classification methods on various datasets. X@Y denotes the prediction results at Y dataset
when observation ratio is set to X. “-” indicates the result is not reported.


|Methods|Year|0.1@BIT|0.5@BIT|0.1@UTI-1|0.5@UTI-1|0.1@UCF-101|0.5@UCF-101|0.1@Sports-1M|0.5@Sports-1M|
|---|---|---|---|---|---|---|---|---|---|
|Integral BoW [220]<br>MSSC [20]<br>Poselet [206]<br>HM [130]<br>MTSSVM [117]<br>MMAPM [112]<br>DeepSCN [118]<br>GLTSD [129]<br>mem-LSTM [114]|2011<br>2013<br>2013<br>2014<br>2014<br>2016<br>2017<br>2018<br>2018|22_._66%<br> 21_._09%<br>-<br>-<br> 28_._12%<br> 32_._81%<br> 37_._50%<br> 26_._60%<br>-|48_._44%<br>48_._44%<br>-<br>-<br>60_._00%<br>67_._97%<br>78_._13%<br>79_._40%<br>-|18_._00%<br>28_._00%<br>-<br>38_._33%<br>36_._67%<br>46_._67%<br>-<br>-<br>-|48_._00%<br>70_._00%<br>73_._33%<br>83_._10%<br>78_._33%<br>78_._33%<br>-<br>-<br>-|36_._29%<br>34_._05%<br>-<br>-<br>40_._05%<br>-<br>45_._02%<br>-<br>51_._02%|74_._39%<br>61_._79%<br>82_._39%<br>-<br>85_._75%<br>88_._37%|43_._47%<br>46_._70%<br>-<br>49_._92%<br>-<br>55_._02%<br>-<br>57_._60%|55_._99%<br>57_._16%<br>66_._90%<br>70_._23%<br>-<br>71_._63%|



Video _x_


Segments ( _K_ = 10) **……**



Partial video _x_ [(] _[k]_ [)]



**Progress level** _g_ = _k_ = 3


**Observation ratio** _r_ = _k_ / _K_ = 0.3



**Fig. 18** Example of a temporally partial video, and graphical
illustration of progress level and observation ratio. Revised
based on the original figure in [118].


_5.1.1 Early Action Classification_


This task aims at recognizing a human action at an
early stage, i.e., based on a temporally incomplete video
(see Figure 17). The goal is to achieve high recognition
accuracy when only the beginning portion of a video
is observed. The observed video contains an unfinished
action, and thus making the prediction task challenging. Although this task may be solved by action recognition methods [206,260,310,309], they were developed
for recognizing complete action executions, and were
not optimized for partial action observations, making
action recognition approaches unsuitable for predicting
actions at an early stage. Table 3 provides some results
of early action classification on four datasets.

Most of the short-term action prediction approaches
follow the problem setup described in [117] shown in
Figure 18. To mimic sequential data arrival, a complete
video **x** with _T_ frames is segmented into _K_ = 10 segments. Consequently, each segment contains _K_ _[T]_ [frames.]

Video lengths _T_ may vary for different videos, thereby
causing different lengths in their segments. For a video
of length _T_, its _k_ -th segment ( _k ∈{_ 1 _, · · ·, K}_ ) contains
frames starting from the [( _k −_ 1) _·_ _K_ _[T]_ [+ 1]-th frame to]

the ( _[kT]_ _K_ [)-th frame. A temporally] _[ partial video]_ [ or] _[ partial]_

_observation_ **x** [(] _[k]_ [)] is defined as a temporal subsequence
that consists of the beginning _k_ segments of the video.
The _progress level g_ of the partial video **x** [(] _[k]_ [)] is defined
by the number of the segments contained in the partial



video **x** [(] _[k]_ [)] : _g_ = _k_ . The _observation ratio r_ of a partial
_k_
video **x** [(] _[k]_ [)] is _[k]_
_K_ [:] _[ r]_ [ =] _K_ [.]

Action prediction approaches aim at recognizing unfinished action videos. Ryoo [220] proposed the integral bag-of-words (IBoW) and dynamic bag-of-words
(DBoW) approaches for action prediction. The action
model of each progress level is computed by averaging
features of a particular progress level in the same category. However, the learned model may not be representative if the action videos of the same class have large
appearance variations, and it is sensitive to outliers. To
overcome these two problems, Cao _et al._ [20] built action
models by learning feature bases using sparse coding
and used the reconstruction error in the likelihood computation. Li _et al._ [144] explored long-duration action
prediction problem. However, their work detects segments by motion velocity peaks, which may not be applicable to complex outdoor datasets. Compared with

[20,144,220], [117] incorporates an important prior knowledge that informative action information is increasing
when new observations are available. In addition, the
method in [117] models label consistency of segments,
which is not presented in their methods. From a perspective of interfering social interaction, Lan _et al._ [130]
developed “hierarchical movements” for action prediction, which is able to capture the typical structure of human movements before an action is executed. An early
event detector [80] was proposed to localize the starting and ending frames of an incomplete event. Their
method first introduces a monotonically increasing scoring function in the model constraint, which has been
popularly used in a variety of action prediction methods [117,112,165]. Different from the aforementioned
methods, [219] studied the action prediction problem in
a first-person scenario, which allows a robot to predict
a person’s action during human-computer interactions.

Deep learning methods have also shown in action
prediction. The work in [165] proposed a new monotonically decreasing loss function in learning LSTMs for
action prediction. Inspired by that, the work in [118]
adopted an autoencoder to model sequential context


20 Yu Kong, Yun Fu



100


90

|Instantly Predictable|Early Predictable|Late Predictable|Col4|
|---|---|---|---|
|~~Billiards~~<br>IceDancing<br>RockClimbingIndoor<br>PlayingPiano<br>PommelHorse<br>Rowing<br>Skijet<br>JugglingBalls<br>SoccerJuggling<br>TaiChi|~~Fencing~~<br>FrisbeeCatch<br>SoccerPenalty<br>VolleyballSpiking<br>HulaHoop<br>FieldHockeyPenalty<br>BasketballDunk<br>CliffDiving<br>Bowling<br>TennisSwing|~~JavelinThrow~~<br>HighJump<br>FrontCrawl<br>HeadMassage<br>Haircut<br>PlayingViolin<br>HandstandWalking<br>PoleVault<br>CricketBowling<br>ThrowDiscus|~~JavelinThrow~~<br>HighJump<br>FrontCrawl<br>HeadMassage<br>Haircut<br>PlayingViolin<br>HandstandWalking<br>PoleVault<br>CricketBowling<br>ThrowDiscus|



**Fig. 19** Top 10 instantly, early, and late predictable actions
in UCF101 dataset. Action names are colored and sorted according to the percentage of their testing samples falling in
the category of instant predictable, early predictable, or late
predictable. Originally shown in [118].


information for action prediction. This method learns
such information from fully-observed videos, and transfer it to partially observed videos. We enforced that
the amount of the transferred information is temporally
ordered for the purpose of modeling the temporal orderings of inhomogeneous action segments. We demonstrated that actions differ in their predictability, and
show the top 10 instantly, early, and late predictable
actions in Figure 19. We also studied the action prediction problem following the popular two-stream framework [237]. In [114], we proposed to use memory to store
hard-to-predict training samples in order to improve the
prediction performance at the early stage. The memory
module used in [114] measures the predictability of each
training sample, and will store those challenging ones.
Such a memory retains a large pool of samples, and
allows us to create complex classification boundaries,
which are particularly useful for discriminating partial
videos at the beginning stage.


_5.1.2 Action Anticipation_


Action anticipation aims to anticipate future actions
from a history of actions [96]. This task is fundamental
to many real world applications. For example, surveillance cameras can raise an alarm before a road accident

happens, robots can make better plans and decisions by
anticipating human actions [122]. Action anticipation is
a challenging task because the models not only need to
detect the actions, but also infer future actions from
the seen actions. RED [96] uses an encoder-decoder
LSTM structure to predict the future video representations from the extracted representations of the historical video frames. Similarly, two LSTMs were used in [60]
to summarize the past and infer the future for egocentric videos. The work in [262] trains a CNN to regress
the future representations from the past ones in an un


supervised way. Three similarity metrics between the
past and future video representations were presented
in [57], namely Jaccard vector similarity, Jaccard crosscorrelation, and Jaccard Frobenius inner product over
covariances for early action anticipation. Future actions
are predicted in [172] by learning a distribution of future actions using Variational Auto-Encoder. The work
in [102] predicts actions at different future timestamps
in one-shot by incorporating a temporal parameter and
skip connections. Hyperbolic space is used in [250] to
predict future actions because it can represent actions
through a compact hierarchy. In [103], the authors proposed a model that consists of a conditional VAE for
modeling the uncertainty of the action starting time
and a MLP to predict whether the action will happen.
Recently, [214] presents a new model called Anticipative Video Transformer and a self-supervised future
prediction loss for action anticipation.


_5.1.3 Intention Prediction_


In practice, there are certain types of actions that contain several primitive action patterns and exhibit complex temporal arrangements, such as “make a dish”.
Typically, the length of these complex actions is longer
than that of short-term actions. Prediction of these

long-term actions is receiving a surge of interest as it
allows us to understand “what is going to happen”, including the final goal of complex human action and the
person’s plausible intended action in the near future.
However, long-term action prediction is extremely
challenging due to the large uncertainty in human future actions. Cognitive science shows that context information is critical to action understanding, as they
typically occur with certain object interactions under
particular scenes. Therefore, it would be helpful to consider the interacting objects together with the human
actions, in order to achieve accurate long-term action
prediction. Such knowledge can provide valuable clues
for two questions “what is happening now?” and “what
is going to happen next?”. It also limits the search
space for potential actions using the interacting object.
For example, if an action “a person grabbing a cup” is
observed, most likely the person is going to “drink a
beverage”, rather than going to “answering a phone”.
Therefore, a prediction method considering such context is expected to provide opportunities to benefit from
contextual constraints between actions and objects.
Pei _et al._ [194] addressed the problem of goal inference and intent prediction using an And-Or-Graph
method, in which the Stochastic Context Sensitive Grammar is embodied. They modeled agent-object interactions, and generated all possible parse graphs of a single


Human Action Recognition and Prediction: A Survey 21



event. Combining all the possibilities generates the interpretation of the input video and achieves the global
maximum posterior probability. They also show that
ambiguities in the recognition of atomic actions can
be reduced largely using hierarchical event contexts.
Li _et al._ [144] proposed a long-term action prediction
method using Probabilistic Suffix Tree (PST), which
captures variable Markov dependencies between action
primitives in complex action. For example, as shown
in Figure 20, a wedding ceremony can be decomposed
into primitives of “hold-hands”, “kneel”, “kiss”, and
“put-ring-on”. In their extension [143], object context
is added to the prediction model, which enables the
prediction of human-object interactions occurring in actions such as “making a dish”. Their work first introduced a concept “predictability”, and used the Predictive Accumulative Function (PAF) to show that some
actions can be early predictable while others cannot be
early predicted. Prediction of human action and object
affordance was investigated in [124]. They proposed an
anticipatory temporal conditional random field (ATCRF)
to model three types of context information, including the hierarchical structure of action primitives, the
rich spatial-temporal correlations between objects and
their affordances, and motion anticipation of objects
and humans. In order to find the most likely motion,
ATCRFs are considered as particles, which are propagated over time to represent the distribution of possible
actions in the future. The work in [65] introduces a new
dataset called LOKI (LOng term and Key Intentions)
for autonomous driving. The authors also proposed a
long-term goal proposal network and a scene graph refinement and trajectory decoder module for jointly predicting the future trajectory and intention of pedestrians. In [11], the authors provided a new dataset for
pedestrian trajectory prediction in dense urban scenarios. A Joint- _β_ -cVAE is further designed to effectively
model the interaction between pedestrians and vehicles,
the model is trained by optimizing the ELBO The authors in [208] proposed a multi-task learning framework
which predicts both trajectories and actions of pedestrians conditioned on multi-modal data. They proposed a
bi-fold feature fusion to effectively fuse multiple modalities, also a semantic map as an additional input to the
model for categorical interaction modeling during training.


5.2 Summary


The availability of big data and recent advance in computer vision and machine learning enable the reasoning
about the future. The key in this research is how to discover temporal correlations in large-scale data and how



**Weigh** Coffee Bean **Grind** Coffee Bean **Place** ground coffee **Heat** water **Pour** water


**Fig. 20** A complex action can be decomposed into a series
of action primitives. Revised based on the original figure in

[143].


**Table** **4** Results of motion trajectory prediction on
ETH/UCF datasets. ADE is the minimum average displacement error, and FDE denotes the final displacement error.
“-” indicates the result is not reported.

|Col1|ADE|FDE|
|---|---|---|
|Social GAN [74]|0.58|-|
|Sophie [223]|0.54|-|
|CGNS [142]|0.49|-|
|Social BiGAT [125]|0.48|-|
|Next [148]|0.46|-|
|Social-STCNN [177]|0.44|-|
|MANTRA [169]|0.32|0.65|
|Transformer TF [67]|0.31|-|
|PECNet [168]|0.29|0.48|
|Social-NCE [159]|0.19|0.40|
|SGNet [264]|0.18|0.35|
|AgentFormer [323]|0.18|0.29|
|Y-Net [167]|0.18|0.27|



to model such correlations. Results shown in Table 19

demonstrate the predictability of actions that can be
used as a prior and inspiring more powerful action prediction methods. There are still some unexplored opportunities in this research, such as interpretability of
temporal extent, how to model long-term temporal correlations, and how to utilize multi-modal data to enrich
the prediction model, which will be discussed in Sec
tion 9.


**6 Motion Trajectory Prediction**


Besides predicting human actions, the other key aspect
in human-centered prediction is motion trajectory prediction, which aims at predicting a pedestrian’s moving
path. Motion trajectory prediction, an inherent capability of us, reasons the possible destination and motion
trajectory of the target person. We can predict with
high confidence that a person is going to walk on sidewalks than streets, and will avoid any obstacles during walking. Therefore, it is interesting to study how to
make machines do the same job. Table 4 shows the ADE
and FDE results on ETH/UCF dataset. ADE and FDE
are standard metrics on motion trajectory prediction.
Some works do not report the FDE result.
Vision-based motion trajectory prediction is essential for practical applications such as visual surveil

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||


accommodate other people or the environment in the
vicinity. Jointly modeling such complex dependencies
is really difficult in dynamic environments. In addition,
the predicted trajectories should not only be _physically_
_acceptable_, but also _socially acceptable_ [74]. Pedestrians always respect personal space while walking, and
thus yield the right-of-way. Human-human and humanobject interactions are typically subtle and complex in
crowded environments, making the problem even more
challenging. Furthermore, there are multiple future predictions in a crowded environment, which are all socially acceptable. Thus uncertainty estimation for the
multimodal predictions is desired.

Forecasting trajectory and destination by understanding the physical scene was investigated in [107], which
was one of the pioneering work in trajectory prediction in the computer vision community. The proposed
method models the effect of the physical environment
on the choice of human actions. The authors integrate
state-of-the-art semantic scene understanding with the
ideas from inverse optimal control (IOC) or inverse reinforcement learning [3,336]. In this work, human motion
is modeled as a sequence of decision-making process,
and a prediction is made by maximizing the reward.
Lee and Kitani [141] extends [107] to a dynamic environment. The state reward function is extended to a lin
ear combination of static and dynamic state functions
to update the forecasting distribution in a dynamic environment. However, IOC is limited to controlled settings as the goal state of the pedestrian’s destination
requires a priori. To relax this assumption, the concept
of _goal set_ was introduced in [166,43], which defines a
target task space. The work in [5] introduced a largescale dataset of 42 million trajectories and studied the



is built upon the dynamic Bayesian network (DBN),
which considers the pedestrian’s decision to stop by
three cues, including the existence of an approaching
vehicle, the pedestrian’s awareness, and the spatial layout of the scene. Walker _et al._ in [263] predicted the behavior of agents ( _e.g._, a car) in a visual scene. Ziebart
_et al._ [337] presented a planning-based approach for trajectory prediction.


Thanks to the recent advance in deep networks, motion trajectory prediction problem can be solved using
RNN/LSTM networks [6,140,243,74], which have the
capability of generating long sequences. More specifically, a single LSTM model was used to account for one
single person’s trajectory, and a social pooling layer in
LSTMs was proposed to model dependencies between
LSTMs, and preserve the spatial information [6]. Compared to previous work [107,141,5,7,120], the method
in [6] is end-to-end trainable, and generalizes well in
complex scenes. An encoder-decoder framework was proposed in [140] for path prediction in more natural scenarios where agents interact with each other and dynamically adapt their future behaviors. Past trajectories are encoded in a RNN and then future trajectory
hypotheses are generated using another decoder implemented by a separate RNN. This method also extends inverse optimal control (IOC) [141,107] to a deep
model, which has shown promising results in robot control [58] and driving [293] tasks. The proposed Deep
IOC is used to rank all the possible hypotheses. The
scene context is captured using a CNN model, which is
part of the input to the RNN encoder. A Social-GAN
network in [74] was proposed to address the limitation
of L2 loss in [140]. Using an adversarial loss, [74] can
potentially learn the distribution of multiple socially


Human Action Recognition and Prediction: A Survey 23



acceptable trajectories, rather than learning the average trajectories in the training data. The work in [39]
proposes a Multi-Generator Model (MGM) to address
the problem of out-of-distribution samples generated
using a single generator. A categorical distribution over
different trajectory types is first predicted by a Path
Module Network, from which the generator is chosen
to sample the future trajectories. Thus, the model can
select scene-specific generators and deactivate unsuitable ones. A divide and conquer method was proposed
in [182] which prevents mode collapse problems in trajectory prediction under the winner-takes-all objective.
The work in [329] proposes a model for goal-conditioned
trajectory prediction which exploits nearest examples
for goal position query and considers multi-modality
and physical constraints.


**7 Datasets**


This section discusses some of the popular action video
datasets, including actions captured in a controlled and
uncontrolled environment. A detailed list is shown in
Table 5. These datasets differ in the number of human
subjects, background noise, appearance and pose variations, camera motion, etc., and have been widely used
for the comparison of various algorithms.


7.1 Controlled Action Video Datasets


We first describe individual action datasets captured in
controlled settings, and then list datasets with two or
more people involved in actions. We also discuss some
of the RGB-D action datasets captured using a costeffective Kinect sensor.


_7.1.1 Individual Action Datasets_


**Weizmann dataset** [14] is a popular video dataset
for human action recognition. The dataset contains 10
action classes such as “walking”, “jogging”, “waving”
performed by 9 different subjects, to provide a total of
90 video sequences. The videos are taken with a static
camera under a simple background.
**KTH dataset** [227] consists of 6 types of human
actions (boxing, hand clapping, hand waving, jogging,
running and walking) repeated several times by 25 different subjects in 4 scenarios (outdoors, outdoors with
scale variation, outdoors with different clothes and indoors). There are 600 action videos in the dataset.
**INRIA XMAS multiview dataset** [285] was complied for multi-view action recognition. It contains videos
captured from 5 views including a top-view camera.



This dataset consists of 13 actions, each of which is
repeated 3 times by 10 actors.


_7.1.2 Group Action Datasets_


**UT-Interaction dataset** [221] is comprised of 2 sets
of 10 videos with different background and camera settings. The videos contain 6 classes of human-human
interactions: handshake, hug, kick, point, punch, and
push.
**BIT-Interaction dataset** [115] consists of 8 classes
of human interactions (bow, boxing, handshake, highfive, hug, kick, pat, and push), with 50 videos per class.
Videos are captured in realistic scenes with cluttered
backgrounds, partially occluded body parts, moving objects, and variations in subject appearance, scale, illumination condition, and viewpoint.
**TV-Interaction dataset** [193] contains 300 videos
clips with human interactions. These videos are categorized into 4 interaction categories: handshake, high five,
hug, and kiss, and annotated with the upper body of
people, discrete head orientation and interaction.
**MultiSports dataset** [146] is a multi-person dataset
that contains 3200 video clips of 4 sport classes. The
dataset contains 37701 action instances with 902 _,_ 000
bounding boxes, which helps for more fine-grained spatiotemporal action detection and localization.


7.2 Unconstrained Datasets


Although the aforementioned datasets lay a solid foundation for action recognition research, they were captured in controlled settings, and may not be able to
train approaches that can be used in real-world scenarios. To address this problem, researchers collected action videos from the Internet, and compiled large-scale
action datasets, which will be discussed in the following.
**UCF101 dataset** [105] has been widely used in action recognition research. It comprises of realistic videos
collected from Youtube. It contains 101 action cate
gories, with 13320 videos in total. UCF101 gives the
largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered
background, illumination conditions, etc. The dataset
can be roughly divided into 5 categories: 1) HumanObject Interaction 2) Body-Motion Only 3) HumanHuman Interaction 4) Playing Musical Instruments 5)
Sports. It should be noted that many clips are collected
from the same video. Consequently, different clips may
have the same person or the same scenario, or the same
lighting, etc. This seems different from practical scenarios, and thus its difficulty is limited.


24 Yu Kong, Yun Fu


**Table 5** A list of popular action video datasets used in action recognition research.


|Datasets|Year|#Videos|#Views|#Actions|#Subjects|#Modality|Env.|
|---|---|---|---|---|---|---|---|
|KTH [227]<br>Weizmann [14]<br>INRIA XMAS [285]<br>IXMAS [321]<br>UCF Sports [213]<br>Hollywood [135]<br>Hollywood2 [170]<br>UCF 11 [95]<br>CA [26]<br>MSR-I [321]<br>MSR-II [322]<br>MHAV [222]<br>UT-I [221]<br>TV-I [193]<br>MSR-A [145]<br>Olympic [185]<br>HMDB51 [127]<br>CAD-60 [248]<br>BIT-I [115]<br>LIRIS [287]<br>MSRDA [271]<br>UCF50 [209]<br>UCF101 [105]<br>MSR-G [128]<br>UTKinect-A [295]<br>ASLAN [109]<br>MSRAP [190]<br>CAD-120 [121]<br>THUMOS’14 [93]<br>Sports-1M [99]<br>3D Online [314]<br>FCVID [94]<br>ActivityNet [50]<br>YouTube-8M [4]<br>Charades [230]<br>NTU-RGB+D [229]<br>PKU-MMD (Phase 1) [29]<br>PKU-MMD (Phase 2) [29]<br>NEU-UB<br>Kinetics [100]<br>AVA [72]<br>20BN-Something-Something [70]<br>SLAC [330]<br>Moments in Time [178]<br>EPIC-Kitchens [35]<br>COIN [253]<br>HACS Segments [328]<br>HAA00 [28]<br>MultiSports [146]|2004<br>2005<br>2006<br>2006<br>2008<br>2008<br>2009<br>2009<br>2009<br>2009<br>2010<br>2010<br>2010<br>2010<br>2010<br>2010<br>2011<br>2011<br>2012<br>2012<br>2012<br>2012<br>2012<br>2012<br>2012<br>2012<br>2013<br>2013<br>2014<br>2014<br>2014<br>2015<br>2015<br>2016<br>2016<br>2016<br>2017<br>2017<br>2017<br>2017<br>2017<br>2017<br>2017<br>2017<br>2018<br>2019<br>2019<br>2021<br>2021|600<br>90<br>390<br>1,148<br>150<br>-<br>3,669<br>1,100+<br>44<br>63<br>54<br>238<br>60<br>300<br>567<br>783<br>6849<br>60<br>400<br>828<br>320<br>50<br>13,320<br>336<br>200<br>3,698<br>360<br>120<br>413<br>1,133,158<br>567<br>91,233<br>28,000<br>8,000,000<br>9,848<br>56,680<br>1076<br>2000<br>600<br>500,000<br>57,600<br>108,499<br>520,000<br>1,000,000<br>90,000+<br>11,827<br>50,000+<br>10,000<br>3200|1<br>1<br>5<br>5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>8<br>2<br>-<br>-<br>-<br>-<br>-<br>-<br>1<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>1<br>-<br>-<br>-<br>-<br>-<br>2<br>-<br>3<br>3<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>1<br>1<br>-<br>-|6<br>10<br>13<br>11<br>10<br>8<br>12<br>11<br>5<br>3<br>3<br>17<br>6<br>4<br>20<br>16<br>51<br>12<br>8<br>10<br>16<br>50<br>101<br>12<br>10<br>432<br>6 pairs<br>12<br>20<br>487<br>20<br>239<br>203<br>4,716<br>157<br>120<br>51<br>49<br>6<br>600<br>80<br>174<br>200<br>339<br>397<br>180<br>200<br>500<br>4|25<br>9<br>10(3 times)<br>-<br>-<br>-<br>10<br>-<br>-<br>10<br>-<br>14<br>10<br>-<br>1<br>-<br>-<br>4<br>50<br>-<br>10<br>-<br>-<br>1<br>-<br>-<br>10<br>4<br>–<br>-<br>-<br>-<br>-<br>-<br>-<br>106<br>66<br>13<br>20<br>-<br>-<br>-<br>-<br>-<br>32<br>–<br>–<br>-<br>-|RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB-D<br>RGB<br>RGB<br>RGB-D<br>RGB<br>RGB<br>RGB-D<br>RGB<br>RGB<br>RGB-D<br>RGB-D<br>RGB<br>RGB-D<br>RGB-D<br>RGB<br>RGB<br>RGB-D<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB+D+IR+Skeleton<br>RGB+D+IR+Skeleton<br>RGB+D+IR+Skeleton<br>RGB-D<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB<br>RGB|Controlled<br>Controlled<br>Controlled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Crowded<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Controlled<br>Controlled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Controlled<br>Uncontrolled<br>Controlled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Controlled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled<br>Uncontrolled|



**HMDB51 dataset** [127] contains a total of about
6849 video clips distributed in a large set of 51 action categories. Each category contains a minimum of
101 video clips. In addition to the label of the action
category, each clip is annotated with an action label
as well as a meta-label describing the property of the
clip, such as visible body parts, camera motion, camera viewpoint, number of people involved in the action,
and video quality. The actions can be grouped into five



categories, including general facial actions ( _e.g._ smile,
chew, talk), Facial actions with object manipulation
( _e.g._ smoke, eat, drink), General body movements ( _e.g._ cartwheel,
clap hands, climb), Body movements with object interaction ( _e.g._ brush hair, catch, draw sword), Body movements for human interaction ( _e.g._ fencing, hug, kick someone). The dataset also has two distinct categories namely
“no motion” and “camera motion”. The dataset is ex
tremely challenging mainly due to the presence of a sig

Human Action Recognition and Prediction: A Survey 25



nificant camera/background motion. To remove camera
motion, standard image stitching techniques to can be
used to align frames of a clip.
**Kinetics** [21] dataset comprises of 700 human action classes and approximately 650 _,_ 000 video clips, including human-object interactions and human-human
interactions. The videos were compiled from YouTube
by matching its title and the prepared Kinetics actions
list. After that, the videos were segmented by tracking actions on Google Image Search, and then labeled
by Amazon’s Mechanical Turk(AMT). In the end, this
dataset is cleaned and de-noised using machine learning techniques. Different from previous datasets, one
clip in this dataset may contain several different actions in sequence, but it is only classified into one action category. This means these clips don’t have complete action labels. As described in their work, the top-5
measure is supposed to be used because of the incomplete labels. Within the same action category, clips are
captured from different videos, including TV and film
videos. Consequently, there is a large appearance variation, for example, people in clips may have different
age, height, clothes, _etc._, and there are various types
of camera motion/shake, background clutter. Besides,
each clip lasts around 10s and has a variable resolution.
**Sports-1M dataset** [99] contains 1 _,_ 133 _,_ 158 video
URLs, which have been annotated automatically with
487 labels. It is one of the largest video datasets. Very
diverse sports videos are included in this dataset, such
as Shaolin Kung Fu, Wing Chun, etc. The dataset is extremely challenging due to very large appearance and
pose variations, significant camera motion, noisy background motion, etc.
**THUMOS’14 dataset [93]** contains more than
20 hours of sport videos. Though the training sets are
trimmed videos labeled with 20 action classes, the validation and testing sets include 200 and 213 untrimmed
videos, respectively. This dataset has been the most
widely used dataset for action detection and localiza
tion.

**ActivityNet dataset [78]** has two versions for
action detection and localization. The first is Activity v1.2, which covers 100 activity classes and contains
4,819 training videos and 2,383 videos for validation.
The other version is Activity v1.3, which consists of
10,024 videos for training and 4,926 videos for validation with 200 activity classes.
**PKU-MMD dataset [29]** is a large-scale multimodal datasets focusing on long continuous sequences
action detection and multi-modality action analysis. The
first phase contains 51 action categories, performed by
66 distinct subjects in 3 camera views. Each video lasts
about 3 _∼_ 4 minutes and contains approximately 20 ac


tion instances. The second phase contains 2,000 short
video sequences in 49 action categories, performed by
13 subjects in 3 camera views. Each video lasts about
1 _∼_ 2 minutes and contains approximately 7 action in
stances.


**AVA dataset [71]** provides audio-visual annotations for about 15 minute long movie clips. For the AVA
Action subsets, it contains 430 videos split into 235 for
training, 64 for validation, and 131 for test. Each video
has 15 minutes annotated in 1-second intervals.


**COIN dataset [253]** is a recently released largescale dataset to address instruction video analysis problems. It contains 11,827 daily activity videos of 180 different classes. Different from other action datasets, human actions in COIN dataset are hierarchically structured with practical semantics.


**HACS dataset [328]** is also a recently released
large-scale dataset for action localization and recognition. For the HACS Segments subset, it contains 139K
action segments densely annotated in 50K untrimmed
videos spanning 200 action categories.


**20BN-SOMETHING-SOMETHING dataset**

[70] is a dataset shows human interaction with everyday
objects. In the dataset, human performs pre-defined action with a daily object. It contains 108 _,_ 499 video clips
across 174 classes. The dataset enables the learning of
visual representations for the physical properties of the
objects and the world.


**Moments-in-Time dataset** [178] is a large-scale
video dataset for action understanding. It contains over
1 _,_ 000 _,_ 000 3-second labeled video clips distributed in
339 categories. The visual elements of the videos include people, animals, objects or natural phenomena.
The dataset is dedicated to building models that are
capable of abstracting and reasoning complex human

actions.


**EPIC-Kitchens** dataset [35] is one of the largest
first-person vision dataset. It consists of 55 hours videos
and 125 verb classes and 300 noun classes recorded by
head-mounted camera. These videos are shot at different cities and different styles kitchens and divided to
39 _,_ 600 action segments with object bounding boxes.
Besides, these videos contain human doing different kitchen
tasks at the same time. To better annotate these ac
tions, voice notes for the actions are collected in the

dataset.


**HAA500** dataset [28] is a human-centric atomic action dataset. It consists of 500 atomic classes, where 212
are sport/athletics, 51 are playing musical instruments,
82 are games and hobbies, and 155 are daily actions.


26 Yu Kong, Yun Fu



7.3 RGB-D Action Video Datasets


All the datasets described above were captured by RGB
video cameras. Recently, there is an increasing interest
in using cost-effective Kinect sensors to capture human
actions due to the extra depth data channel. Compared
to RGB data channels, the extra depth data channel elegantly provides scene structure, which can be used to
simplify intra-class motion variations and reduce cluttered background noise [113]. Popular RGB-D action
datasets are listed in the following.


**MSR Daily Activity dataset** [271]: there are 16
categories of actions: drink, eat, read book, call cellphone, write on a paper, use laptop, use vacuum cleaner,
cheer up, sit still, toss paper, play game, lie down on
sofa, walk, play guitar, stand up, sit down. All these actions are performed by 10 subjects. There are 320 RGB
samples and 320 depth samples available.


**3D Online Action dataset** [314] was compiled for
three evaluation tasks: same-environment action recognition, cross-environment action recognition and continuous action recognition. The dataset contains human action or human-object interaction videos captured from RGB-D sensors. It contains 7 action cate
gories, such as drinking, eating, and reading cellphone.


**CAD-120 dataset** [121] comprises of 120 RGB-D
action videos of long daily activities. It is also captured
using the Kinect sensor. Action videos are performed
by 4 subjects. The dataset consists of 12 action types,
such as rinsing mouth, talking on the phone, cooking,
writing on whiteboard, etc. Tracked skeletons, RGB images, and depth images are provided in the dataset.


**UTKinect-Action dataset** [296] was captured by
a Kinect device. There are 10 high-level action categories contained in the dataset such as walk, sit down,
etc. The dataset comprises 200 action vidos and three
channels were recorded: RGB, depth and skeleton joint
locations.


**NTU-RGB+D** [230,154] dataset contains 60 action classes and 56 _,_ 880 video samples. Recently, it has
been extended to 120 action classes and another 114 _,_ 480
video samples in [154]. All the samples were collected
from 106 distinct subjects by Kinect sensors. RGB videos,
depth map sequences, 3D skeletal data, and infrared
(IR) videos are provided for each sample. There is higher
variation of environmental conditions compared with
previous datasets, including 96 different backgrounds
with illumination variations.



**8 Evaluation Protocols for Action Recognition**

**and Prediction**


Due to different application purposes, action recognition and prediction techniques are evaluated in different

ways.
Shallow action recognition methods such as [227,
186,291] were usually evaluated on small-scale datasets,
for example, Weizmann dataset [14], KTH dataset [227],
UCF Sports dataset [213]. The leave-one-out training
scheme is popularly used on these datasets, and the
confusion matrix is usually adopted to show the recognition accuracy of each action category. For sequential
approaches such as [282,283], per-frame recognition accuracy is often used. In [170,251], average precision that
approximates the area under the precision-recall curve
is also adopted for each individual action class. Deep
networks [21,255,261] are generally evaluated on largescale datasets such as UCF-101 [105] and HMDB51

[127] and thus can only report overall recognition performance on each dataset. Please refer to [79] for a list
of performance of recent action recognition methods on
various datasets.

Most of action prediction methods [220,20,117,118]
were evaluated on existing action datasets. Different
from the evaluation method used in action recognition,
recognition accuracy at each observation ratio (ranging from 10% to 100%) is reported for action prediction methods. As described in [118], the goal of these
methods is to achieve high recognition accuracy at the
beginning stage of action videos, in order to accurately
recognize actions as early as possible. Table 3 summarizes the performance of action prediction methods on

various datasets.

There are several popular metrics for evaluating motion trajectory prediction methods, including _Average_
_Displacement Error_ (ADE), _Final Displacement Error_
(FDE), and _Average Non-linear Displacement Error_ (ANDE).
ADE is the mean square error computed over all estimated points of a trajectory and the ground-truth
points. FDE is defined as the distance between the predicted final destination and the ground-true final destination. ANDE is the MSE at the non-linear turning
regions of a trajectory arising from human-human in
teractions.

Vairous metrics exist to evaluate action detection
and localization methods. Recall that Recall (R) measures the number of true positives over the total number of true positives and false negatives. Average Recall
(AR) is the average of recalls over multiple Intersection
over Union (IoU) values. Area under the AR vs. AN
curve (AUC) measures how well the detection method
is able to distinguish between positive and negative pro

Human Action Recognition and Prediction: A Survey 27



posals. Another metric called mean Average Precision
(mAP) @ _α_ where _α_ denotes different IoU threshold
which measures the Average Prevision (AP) on each
action category.


**9 Future Directions**


In this section, we discuss some future directions in action recognition and prediction research that might be
interesting to explore.
**Dataset.** Significant efforts have been made to collect different types of action video datasets in recent
years in order to advance the research of action recognition and prediction. Nevertheless, existing action recognition and prediction models trained on these datasets
are still difficult to be generalized to real-world scenarios, possibly because the incapability of these datasets
in covering all the aspects that may happen in practical scenarios. First of all, majority of the video datasets
were collected under good lighting and weather conditions. However, this assumption may not hold in practice. A visual surveillance system may need to run 24
hours a day whatever the weather is. Unfortunately,
existing methods are still difficult to be generalized to
poor lighting conditions or extreme weather. Second
of all, some datasets are restricted to certain scenarios, for example, UCF101 contains sports videos and
EPIC-Kitchens dataset captured in kitchens. Although
one well-trained model may perform well in one scenario, it may perform poorly in a new scenario. This
could be attributed to the new environment, camera
motion, appearance changes, _etc._ that have not been
seen in the previous scenario. Last but not least, existing deep neural networks based methods require a
significant amount of data for training. However, video
data could be limited in some research areas, such as

biomedical research or human rehabilitation research. Is

it possible to create and render virtual training video
data using game engines such as UnReal [1,2] based
on existing small-scale data? This could serve as an
alternative solution to directly generalizing deep neural networks to small-scale data. All of these challenges
bring new problems to action recognition research and
prompt us to collect new datasets to advance the re
search.
**Benefitting from image models.** Deep architectures are dominating the action recognition research
lately like the trend of other developments in the computer vision community. However, training deep networks on videos is difficult, and thus benefiting from
deep models pre-trained on images or other sources
would be a better solution to explore. In addition, image models have done a good job of capturing spatial



relationships of objects, which could also be exploited
in action understanding. It is interesting to explore how
to transfer knowledge from image models to video models using the idea of inflation [21] or domain adaptation

[252].

**Interpretability on temporal extent.** Interpretability of image models has been discussed but it has not
been extensively discussed in video models. As shown
in [224,206], not all frames are equally important for
action recognition; only few of them are critical. Therefore, there are a few things that require a deep understanding of the temporal interpretability of video
models. First of all, actions, especially long-duration
actions can be considered as a sequence of primitives.
It would be interesting to have interpretability of these
primitives, such as how are these primitives organized
in the temporal domain in actions, how do they contribute to the classification task, can we only use few
of them without sacrificing recognition performance in
order to achieve fast training? In addition, actions differ in their temporal characteristics. Some actions can
be understood at their early stage and some actions
require more frames to be observed. It would be interesting to ask why these actions can be early predicted,
and what are the salient signals that are captured by
the machine. Such an understanding would be useful in
developing more efficient action prediction models.

**Learning from multi-modal data.** Humans are
observing multi-modal data everyday, including visual,
audio, text, etc. These multi-modal data help the understanding of each type of data. For example, reading a book helps us to reconstruct the corresponding
part of the visual scene. However, little work is paying
attention to action recognition/prediction using multimodal data. It is beneficial to use multi-modal data to
help visual understanding of complex actions because
the multi-modal data such as text data contain rich

semantic knowledge given by humans. In addition to
action labels, which can be considered as verbs, textual data may include other entities such as nouns (objects), prepositions (spatial structure of the scene), adjectives and adverbs, etc. Although learning from nouns
and prepositions have been explored in action recognition and human-object interaction, few studies have
been devoted to learning from adjectives and adverbs.
Such learning tasks provide more descriptive information about human actions such as motion strength, thereby
making fine-grained action understanding into reality.

**Learning long-term temporal correlations.** Multimodal data also enable the learning of long-term temporal correlations between visual entities from the data,
which might be difficult to directly learn from visual
data. Long-term temporal correlations characterize the


28 Yu Kong, Yun Fu



sequential order of actions occurring in a long sequence,
which is similar to what our brain stores. When we want

to recall something, one pattern evokes the next pattern, suggesting the associations spanning in long-term
videos. Interactions between visual entities are also crit
ical to understanding long-term correlations. Typically,
certain actions occur with certain object interactions
under particular scene settings. Therefore, it needs to
involve not only actions, but also an interpretation of
objects, scenes and their temporal arrangements with
actions, since this knowledge can provide a valuable clue
for “what’s happening now” and “what’s going to happen next”. This learning task also allows us to predict
actions in a long-duration sequence.

**Physical aspect of actions.** Action recognition
and prediction are tasks fairly targeting at high-level
aspects of videos, and not focusing on finding action
primitives that encode basic physical properties. Recently, there has been an increasing interest in learning the physical aspects of the world, which studies
fine-grained actions. One example is the somethingsomething dataset introduced in [70] that studies humanobject interactions. Interestingly, this dataset provides
labels or textual description templates such as “Dropping [something] into [something]”, to describe the interaction between humans and objects, and an object
and an object. This allows us to learn models that
can understand physical aspects of the world including human actions, object-object interactions, spatial
relationships, etc.

Even though we can infer a large amount of information from action videos, there are still some physical aspects that are challenging to be inferred. We are
wondering that can we make a step further, saying understanding more physical aspects, such as the motion
style, force, acceleration, etc, from videos? Physics-101

[290] studied this problem in objects, but can we extend
it to actions? A new action dataset containing such finegrained information is needed. To achieve this goal, our
ongoing work is providing a dataset containing human
actions with EMG signals, which we hope to benefit
fine-grained action recognition.

**Learning actions without annotations.** For increasingly large action datasets such as Something-Something

[70] and Sports-1M [99], manual labeling becomes prohibitive. Automatic labeling using search engines [99,4],
video subtitles and movie scripts [170,135] is possible
in some domains, but still requires manual verification.
Crowdsourcing [70] would be a better option but still
suffers from labeling diversity problem, and may generate incorrect action labels. In addition, videos in almost
all the action datasets are temporally segmented, with
only one action in each of the videos. However, this



assumption does not hold as videos may be streaming
and it is difficult to know the exact starting and ending frames of an action execution in streaming videos.
This prompts us to develop more robust and efficient
action recognition/prediction approaches that can automatically learn from unlabeled videos or untrimmed
videos.

**Actions in open-world.** Human action recognition in real-world is essentially an open set problem,
which requires the model to simultaneously recognize
the known action classes and reject the unknown actions [63,8]. However, existing open set recognition (OSR)
research works mainly focus on image modality [226,
225,326,9,191,195,23], except for a few works on videos [235,
215] and other modalities [307]. These works typically
do not work well on video data due to the following
challenges. First, the temporal nature of videos leads
to high diversity of human actions, which is challenging for an OSR model to be aware of _what it does not_
_know_ when given human actions with unknown temporal dynamics. Besides, the static bias (i.e., appearance
of the video background and foreground actor) in video
data could be easily over-fitted by deep learning models. The model finally could hardly identify unknown
actions in an unbiased open vision world. These challenges motivate recent work [8] to build an uncertaintyaware and unbiased model for open set action recognition (OSAR). Since open-world actions can be regarded
as out-of-distribution (OOD) data, developing more advanced OOD detection methods to tackle the distribu
tional shift of human actions under OSAR setting is
promising in the future.



**10 Conclusion**



The availability of big data and powerful models diverts the research focus on human actions from un
derstanding the present to reasoning the future. We
have presented a complete survey of state-of-the-art
techniques for action recognition and prediction from
videos. These techniques became particularly interesting in recent decades due to their promising and practical applications in several emerging fields focusing on
human movements. We investigate several aspects of
the existing attempts including hand-crafted feature design, models and algorithms, deep architectures, datasets,
and system performance evaluation protocols. Future
research directions are also discussed in this survey.



**References**



1. Unreal engine. `[https://www.unrealengine.com/](https://www.unrealengine.com/)`


Human Action Recognition and Prediction: A Survey 29



2. UnrealCV. `[https://unrealcv.org](https://unrealcv.org)`
3. Abbeel, P., Ng, A.: Apprenticeship learning via inverse
reinforcement learning. In: ICML (2004)
4. Abu-El-Haija, S., Kothari, N., Lee, J., Natsev, P.,
Toderici, G., Varadarajan, B., Vijayanarasimhan, S.:
Youtube-8m: A large-scale video classification benchmark. arXiv preprint arXiv:1609.08675 (2016)
5. Alahi, A., Fei-Fei, V.R.L.: Socially-aware large-scale
crowd forecasting. In: CVPR (2014)
6. Alahi, A., Goel, K., Ramanathan, V., Robicquet, A.,
Fei-Fei, L., Savarese, S.: Social lstm: Human trajectory
prediction in crowded spaces. In: CVPR (2016)
7. Ballan, L., Castaldo, F., Alahi, A., Palmieri, F.,
Savarese, S.: Knowledge transfer for scene-specific motion prediction. In: ECCV (2016)
8. Bao, W., Yu, Q., Kong, Y.: Evidential deep learning for
open set action recognition. In: ICCV (2021)
9. Bendale, A., Boult, T.E.: Towards open set deep networks. In: CVPR (2016)
10. Bengio, Y., Courville, A., Vincent, P.: Representation
learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence
(2013)
11. Bhattacharyya, A., Reino, D.O., Fritz, M., Schiele, B.:
Euro-pvi: Pedestrian vehicle interactions in dense urban
centers. In: CVPR (2021)
12. Bishay, M., Zoumpourlis, G., Patras, I.: Tarn: Temporal attentive relation network for few-shot and zero-shot
action recognition. In: BMVC (2019)
13. Blake, R., Shiffrar, M.: Perception of human motion.
Annu. Rev. Psychol. **58**, 47–73 (2007)
14. Blank, M., Gorelick, L., Shechtman, E., Irani, M., Basri,
R.: Actions as space-time shapes. In: Proc. ICCV (2005)
15. Bobick, A., Davis, J.: The recognition of human movement using temporal templates. IEEE Trans Pattern Analysis and Machine Intelligence **23** (3), 257–267
(2001)
16. Bojanowski, P., Lajugie, R., Bach, F., Laptev, I., Ponce,
J., Schmid, C., Sivic, J.: Weakly supervised action labeling in videos under ordering constraints. In: European
Conference on Computer Vision, pp. 628–643. Springer
(2014)
17. Bregonzio, M., Gong, S., Xiang, T.: Recognizing action
as clouds of space-time interest points. In: CVPR (2009)
18. Buchler, U., Brattoli, B., Ommer, B.: Improving
spatiotemporal self-supervision by deep reinforcement
learning. In: Proceedings of the European Conference
on Computer Vision (ECCV), pp. 770–786 (2018)
19. Cao, K., Ji, J., Cao, Z., Chang, C.Y., Niebles, J.C.: Fewshot video classification via temporal alignment. In:
CVPR (2020)
20. Cao, Y., Barrett, D., Barbu, A., Narayanaswamy, S.,
Yu, H., Michaux, A., Lin, Y., Dickinson, S., Siskind, J.,
Wang, S.: Recognizing human activities from partially
observed videos. In: CVPR (2013)
21. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: CVPR
(2017)
22. Chao, Y.W., Vijayanarasimhan, S., Seybold, B., Ross,
D.A., Deng, J., Sukthankar, R.: Rethinking the Faster
R-CNN architecture for temporal action localization. In:
CVPR (2018)
23. Chen, G., Qiao, L., Shi, Y., Peng, P., Li, J., Huang,
T., Pu, S., Tian, Y.: Learning open set network with
discriminative reciprocal points. In: ECCV (2020)



24. Chen, S., Sun, P., Xie, E., Ge, C., Wu, J., Ma, L., Shen,
J., Luo, P.: Watch only once: An end-to-end video action
detection framework. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV),
pp. 8178–8187 (2021)
25. Choi, W., Savarese, S.: A unified framework for multitarget tracking and collective activity recognition. In:
ECCV, pp. 215–230. Springer (2012)
26. Choi, W., Shahid, K., Savarese, S.: What are they
doing? : Collective activity classification using spatiotemporal relationship among people. In: Computer Vision Workshops (ICCV Workshops), 2009 IEEE 12th
International Conference on, pp. 1282 –1289 (2009)
27. Choi, W., Shahid, K., Savarese, S.: Learning context for
collective activity recognition. In: CVPR (2011)
28. Chung, J., hsin Wuu, C., ru Yang, H., Tai, Y.W., Tang,
C.K.: Haa500: Human-centric atomic action dataset
with curated videos. In: ICCV (2021)
29. Chunhui, L., Yueyu, H., Yanghao, L., Sijie, S., Jiaying, L.: Pku-mmd: A large scale benchmark for continuous multi-modal human action understanding. arXiv
preprint arXiv:1703.07475 (2017)
30. Ciptadi, A., Goodwin, M.S., Rehg, J.M.: Movement pattern histogram for action recognition and retrieval. In:
D. Fleet, T. Pajdla, B. Schiele, T. Tuytelaars (eds.)
Computer Vision – ECCV 2014, pp. 695–710. Springer
International Publishing, Cham (2014)
31. Clarke, T., Bradshaw, M., Field, D., Hampson, S., Rose,
D.: The perception of emotion from body movement in
point-light displays of interpersonal dialogue. Perception **24**, 1171–80 (2005)
32. Cutting, J., Kozlowski, L.: Recognition of friends by
their work: gait perception without familarity cues.
Bull. Psychon. Soc. **9**, 353–56 (1977)
33. Dai, X., Singh, B., Zhang, G., Davis, L., Chen, Y.: Temporal context network for activity localization in videos.
2017 IEEE International Conference on Computer Vision (ICCV) pp. 5727–5736 (2017)
34. Dalal, N., Triggs, B.: Histograms of oriented gradients
for human detection. In: CVPR (2005)
35. Damen, D., Doughty, H., Farinella, G.M., Fidler, S.,
Furnari, A., Kazakos, E., Moltisanti, D., Munro, J., Perrett, T., Price, W., Wray, M.: Scaling egocentric vision:
The epic-kitchens dataset. In: European Conference on
Computer Vision (2018)
36. Darwin, C.: The Expression of the Emotions in Man and
Animals. London: John Murray (1872)
37. Dawar, N., Kehtarnavaz, N.: Action detection and
recognition in continuous action streams by deep
learning-based sensing fusion. IEEE Sensors Journal
**18** (23), 9660–9668 (2018)
38. Decety, J., Grezes, J.: Neural mechanisms subserving
the perception of human actions. Neural mechanisms of
perception and action **3** (5), 172–178 (1999)
39. Dendorfer, P., Elflein, S., Leal-Taix´e, L.: Mg-gan: A
multi-generator model preventing out-of-distribution
samples in pedestrian trajectory prediction. In: ICCV
(2021)
40. Diba, A., Sharma, V., Gool, L.V.: Deep temporal linear
encoding networks. In: CVPR (2017)
41. Dollar, P., Rabaud, V., Cottrell, G., Belongie, S.: Behavior recognition via sparse spatio-temporal features.
In: ICCV VS-PETS (2005)
42. Donahue, J., Hendricks, L., Guadarrama, S., Rohrbach,
M., Venugopalan, S., Saenko, K., Darrell, T.: Long-term
recurrent convolutional networks for visual recognition
and description. In: CVPR (2015)


30 Yu Kong, Yun Fu



43. Dragan, A., Ratliff, N., Srinivasa, S.: Manipulation planning with goal sets using constrained trajectory optimization. In: ICRA (2011)
44. Duchenne, O., Laptev, I., Sivic, J., Bach, F., Ponce, J.:
Automatic annotation of human actions in video. In:

2009 IEEE 12th International Conference on Computer
Vision, pp. 1491–1498. IEEE (2009)
45. Duong, T.V., Bui, H.H., Phung, D.Q., Venkatesh, S.:
Activity recognition and abnormality detection with the
switching hidden semi-markov model. In: CVPR (2005)
46. Duta, I.C., Ionescu, B., Aizawa, K., Sebe, N.: spatiotemporal vector of locally max pooled features for action
recognition in videos. In: CVPR (2017)
47. Dwivedi, S.K., Gupta, V., Mitra, R., Ahmed, S., Jain,
A.: Protogan: Towards few shot learning for action
recognition. In: ICCVW (2019)
48. Efros, A., Berg, A., Mori, G., Malik, J.: Recognizing
action at a distance. In: ICCV, vol. 2, pp. 726 –733
(2003)
49. Escorcia, V., Caba Heilbron, F., Niebles, J.C., Ghanem,
B.: DAPs: Deep action proposals for action understanding. In: ECCV (2016)
50. Fabian Caba Heilbron Victor Escorcia, B.G., Niebles,
J.C.: Activitynet: A large-scale video benchmark for
human activity understanding. In: Proceedings of the
IEEE Conference on Computer Vision and Pattern
Recognition, pp. 961–970 (2015)
51. Fanti, C., Zelnik-Manor, L., Perona, P.: Hybrid models
for human motion recognition. In: CVPR (2005)
52. Feichtenhofer, C., Pinz, A., Wildes, R.P.: Spatiotemporal residual networks for video action recognition. In:
NIPS (2016)
53. Feichtenhofer, C., Pinz, A., Wildes, R.P.: Spatiotemporal multiplier networks for video action recognition. In:
2017 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 7445–7454. IEEE (2017)
54. Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusion for video action recognition. In: CVPR (2016)
55. Felzenszwalb, P., McAllester, D., Ramanan, D.: A
discriminatively trained, multiscale, deformable part
model. In: CVPR (2008)
56. Fernando, B., Bilen, H., Gavves, E., Gould, S.: Selfsupervised video representation learning with odd-oneout networks. In: Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 3636–3645
(2017)
57. Fernando, B., Herath, S.: Anticipating human actions by
correlating past with the future with jaccard similarity
measures. In: CVPR (2021)
58. Finn, C., Levine, S., Abbeel, P.: Guided cost learning:
deep inverse optimal control via policy optimization. In:
arXiv preprint arXiv:1603.00448 (2016)
59. Fouhey, D.F., Zitnick, C.L.: Predicting object dynamics
in scenes. In: CVPR (2014)
60. Furnari, A., Farinella, G.M.: Rolling-unrolling lstms for
action anticipation from first-person video. IEEE Transactions on Pattern Analysis and Machine Intelligence
(PAMI) (2020)
61. Gan, C., Gong, B., Liu, K., Su, H., Guibas, L.J.: Geometry guided convolutional neural networks for selfsupervised video representation learning. In: Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 5589–5597 (2018)
62. Gao, J., Yang, Z., Chen, K., Sun, C., Nevatia, R.: TURN
TAP: Temporal unit regression network for temporal
action proposals. In: ICCV (2017)



63. Geng, C., Huang, S.j., Chen, S.: Recent advances in open
set recognition: A survey. IEEE transactions on pattern
analysis and machine intelligence (2020)
64. Ghadiyaram, D., Tran, D., Mahajan, D.: Large-scale
weakly-supervised pre-training for video action recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 12046–12055
(2019)
65. Girase, H., Gang, H., Malla, S., Li, J., Kanehara, A.,
Mangalam, K., Choi, C.: Loki: Long term and key intentions for trajectory prediction. In: ICCV (2021)
66. Girdhar, R., Ramanan, D., Gupta, A., Sivic, J., Russell, B.: Actionvlad: Learning spatio-temporal aggregation for action classification. In: CVPR (2017)
67. Giuliari, F., Hasan, I., Cristani, M., Galasso, F.: Transformer networks for trajectory forecasting. In: 2020
25th International Conference on Pattern Recognition
(ICPR), pp. 10335–10342. IEEE (2021)
68. Goodale, M.A., Milner, A.D.: Separate visual pathways
for perception and action. Trends in Neurosciences
**15** (1), 20–25 (1992)
69. Gorelick, L., Blank, M., Shechtman, E., Irani, M., Basri,
R.: Actions as space-time shapes. Transactions on Pattern Analysis and Machine Intelligence **29** (12), 2247–
2253 (2007)
70. Goyal, R., Kahou, S.E., Michalski, V., Materzynska, J.,
Westphal, S., Kim, H., Haenel, V., Fruend, I., Yianilos,
P., Mueller-Freitag, M., et al.: The” something something” video database for learning and evaluating visual
common sense. In: Proc. ICCV (2017)
71. Gu, C., Sun, C., Ross, D.A., Vondrick, C., Pantofaru,
C., Li, Y., Vijayanarasimhan, S., Toderici, G., Ricco, S.,
Sukthankar, R., et al.: AVA: A video dataset of spatiotemporally localized atomic visual actions. In: CVPR
(2018)
72. Gu, C., Sun, C., Vijayanarasimhan, S., Pantofaru,
C., Ross, D.A., Toderici, G., Li, Y., Ricco, S., Sukthankar, R., Schmid, C., et al.: Ava: A video dataset of
spatio-temporally localized atomic visual actions. arXiv
preprint arXiv:1705.08421 (2017)
73. Guo, M., Chou, E., Huang, D.A., Song, S., Yeung, S.,
Fei-Fei, L.: Neural graph matching networks for fewshot
3d action recognition. In: ECCV (2018)
74. Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi,
A.: Social gan: Socially acceptable trajectories with generative adversarial networks. In: CVPR (2018)
75. Hadfield, S., Bowden, R.: Hollywood 3d: Recognizing actions in 3d natural scenes. In: CVPR. Portland, Oregon
(2013)
76. Harris, C., Stephens., M.: A combined corner and edge
detector. In: Alvey Vision Conference (1988)
77. Hasan, M., Roy-Chowdhury, A.K.: Continuous learning
of human activity models using deep nets. In: ECCV
(2014)
78. Heilbron, F.C., Escorcia, V., Ghanem, B., Niebles, J.C.:
ActivityNet: A large-scale video benchmark for human
activity understanding. In: CVPR (2015)
79. Herath, S., Harandi, M., Porikli, F.: Going deeper into
action recognition: A survey. Image and Vision Computing (2017)
80. Hoai, M., la Torre, F.D.: Max-margin early event detectors. In: CVPR (2012)
81. Horn, B., Schunck, B.: Determining optical flow. Artificial Intelligence **17**, 185–203 (1981)
82. Hu, J.F., Zheng, W.S., Lai, J., Zhang, J.: Jointly learning heterogeneous features for rgb-d activity recognition. In: CVPR (2015)


Human Action Recognition and Prediction: A Survey 31



83. Hu, W., Xie, D., Fu, Z., Zeng, W., Maybank, S.:
Semantic-based surveillance video retrieval. Image Processing, IEEE Transactions on **16** (4), 1168–1181 (2007)
84. Huang, D.A., Fei-Fei, L., Niebles, J.C.: Connectionist
temporal modeling for weakly supervised action labeling. In: European Conference on Computer Vision, pp.
137–153. Springer (2016)
85. Huang, D.A., Kitani, K.M.: Action-reaction: Forecasting the dynamics of human interaction. In: ECCV
(2008)
86. Ikizler, N., Forsyth, D.: Searching video for complex activities with finite state models. In: CVPR (2007)
87. Jain, M., van Gemert, J., Jegou, H., Bouthemy, P.,
Snoek, C.G.: Action localization with tubelets from motion. In: CVPR (2014)
88. Jain, M., J´egou, H., Bouthemy, P.: Better exploiting motion for better action recognition. In: CVPR (2013)
89. Ji, S., Xu, W., Yang, M., Yu, K.: 3d convolutional neural
networks for human action recognition. In: ICML (2010)
90. Ji, S., Xu, W., Yang, M., Yu, K.: 3d convolutional neural
networks for human action recognition. IEEE Trans.
Pattern Analysis and Machine Intelligence (2013)
91. Jia, C., Kong, Y., Ding, Z., Fu, Y.: Latent tensor transfer learning for rgb-d action recognition. In: ACM Multimedia (2014)
92. Jia, K., Yeung, D.Y.: Human action recognition using local spatio-temporal discriminant embedding. In: CVPR
(2008)
93. Jiang, Y.G., Liu, J., Roshan Zamir, A., Toderici, G.,
Laptev, I., Shah, M., Sukthankar, R.: THUMOS challenge: Action recognition with a large number of classes.
`[http://crcv.ucf.edu/THUMOS14/](http://crcv.ucf.edu/THUMOS14/)` (2014)
94. Jiang, Y.G., Wu, Z., Wang, J., Xue, X., Chang, S.F.:
Exploiting feature and class relationships in video categorization with regularized deep neural networks. IEEE
Transactions on Pattern Analysis and Machine Intelligence **40** (2), 352–364 (2018). DOI 10.1109/TPAMI.
2017.2670560. URL `[https://doi.org/10.1109/TPAMI.](https://doi.org/10.1109/TPAMI.2017.2670560)`

```
  2017.2670560
```

95. Jingen Liu, J.L., Shah, M.: Recognizing realistic actions
from videos ”in the wild”. In: CVPR (2009)
96. Jiyang Gao Zhenheng Yang, R.N.: Red: Reinforced
encoder-decoder networks for action anticipation. In:
BMVC (2017)
97. Kar, A., Rai, N., Sikka, K., Sharma, G.: Adascan: Adaptive scan pooling in deep convolutional neural networks
for human action recognition in videos. In: CVPR
(2017)
98. Karaman, S., Seidenari, L., Bimbo, A.D.: Fast saliency
based pooling of fisher encoded dense trajectories. In:
ECCV THUMOS Workshop (2014)
99. Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., Fei-Fei, L.: Large-scale video classification
with convolutional neural networks. In: CVPR (2014)
100. Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier,
C., Vijayanarasimhan, S., Viola, F., Green, T., Back,
T., Natsev, P., et al.: The kinetics human action video
dataset. arXiv preprint arXiv:1705.06950 (2017)
101. Ke, Q., Bennamoun, M., An, S., Sohel, F., Boussaid, F.:
A new representation of skeleton sequences for 3d action
recognition. In: Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 3288–3297
(2017)
102. Ke, Q., Fritz, M., Schiele, B.: Time-conditioned action
anticipation in one shot. In: CVPR (2019)
103. Ke, Q., Fritz, M., Schiele, B.: Future moment assessment
for action query. In: Proceedings of the IEEE/CVF



Winter Conference on Applications of Computer Vision
(2021)
104. Keestra, M.: Understanding human action. integraiting
meanings, mechanisms, causes, and contexts. TRANSDISCIPLINARITY IN PHILOSOPHY AND SCIENCE: APPROACHES, PROBLEMS, PROSPECTS
pp. 201–235 (2015)
105. Khurram Soomro, A.R.Z., Shah, M.: Ucf101: A dataset
of 101 human action classes from videos in the wild
(2012). CRCV-TR-12-01
106. Kim, K., Lee, D., Essa, I.: Gaussian process regression
flow for analysis of motion trajectories. In: ICCV (2011)
107. Kitani, K.M., Ziebart, B.D., Bagnell, J.A., Hebert, M.:
Activity forecasting. In: ECCV (2012)
108. Klaser, A., Marszalek, M., Schmid, C.: A spatiotemporal descriptor based on 3d-gradients. In: BMVC
(2008)
109. Kliper-Gross, O., Hassner, T., Wolf, L.: The action similarity labeling challenge. IEEE Transactions on Pattern
Analysis and Machine Intelligence **34** (3) (2012)
110. Kong, Y., Fu, Y.: Modeling supporting regions for close
human interaction recognition. In: ECCV workshop
(2014)
111. Kong, Y., Fu, Y.: Bilinear heterogeneous information
machine for rgb-d action recognition. In: CVPR (2015)
112. Kong, Y., Fu, Y.: Max-margin action prediction machine. TPAMI **38** (9), 1844 – 1858 (2016)
113. Kong, Y., Fu, Y.: Max-margin heterogeneous information machine for rgb-d action recognition. International
Journal of Computer Vision (IJCV) **123** (3), 350–371
(2017)
114. Kong, Y., Gao, S., Sun, B., Fu, Y.: Action prediction
from videos via memorizing hard-to-predict samples. In:
AAAI (2018)
115. Kong, Y., Jia, Y., Fu, Y.: Learning human interaction
by interactive phrases. In: Proc. European Conf. on
Computer Vision (2012)
116. Kong, Y., Jia, Y., Fu, Y.: Interactive phrases: Semantic descriptions for human interaction recognition. In:
PAMI (2014)
117. Kong, Y., Kit, D., Fu, Y.: A discriminative model
with multiple temporal scales for action prediction. In:
ECCV (2014)
118. Kong, Y., Tao, Z., Fu, Y.: Deep sequential context networks for action prediction. In: CVPR (2017)
119. Kong, Y., Tao, Z., Fu, Y.: Adversarial action prediction
networks. IEEE TPAMI (2018)
120. Kooij, J.F.P., Schneider, N., Flohr, F., Gavrila, D.M.:
Context-based pedestrian path prediction. In: European
Conference on Computer Vision, pp. 618–633. Springer
(2014)
121. Koppula, H.S., Gupta, R., Saxena, A.: Learning human
activities and object affordances from rgb-d videos. International Journal of Robotics Research (2013)
122. Koppula, H.S., Saxena, A.: Anticipating human activities for reactive robotic response. In: IROS (2013)
123. Koppula, H.S., Saxena, A.: Learning spatio-temporal
structure from rgb-d videos for human activity detection and anticipation. In: ICML (2013)
124. Koppula, H.S., Saxena, A.: Anticipating human activities using object affordances for reactive robotic response. IEEE transactions on pattern analysis and machine intelligence **38** (1), 14–29 (2016)
125. Kosaraju, V., Sadeghian, A., Mart´ın-Mart´ın, R.,
Reid, I., Rezatofighi, S.H., Savarese, S.: Social-bigat:
Multimodal trajectory forecasting using bicycle-gan
and graph attention networks. arXiv preprint
arXiv:1907.03395 (2019)


32 Yu Kong, Yun Fu



126. Kretzschmar, H., Kuderer, M., Burgard, W.: Learning to predict trajecteories of cooperatively navigation
agents. In: International Conference on Robotics and
Automation (2014)
127. Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., Serre,
T.: Hmdb: A large video database for human motion
recognition. In: ICCV (2011)
128. Kurakin, A., Zhang, Z., Liu, Z.: A real-time system for
dynamic hand gesture recognition with a depth sensor.
In: EUSIPCO (2012)
129. Lai, S., Zhang, W.S., Hu, J.F., Zhang, J.: Global-local
temporal saliency action prediction. IEEE Transactions
on Image Processing **27** (5), 2272–2285 (2018)
130. Lan, T., Chen, T.C., Savarese, S.: A hierarchical representation for future action prediction. In: European
Conference on Computer Vision, pp. 689–704. Springer
(2014)
131. Lan, T., Sigal, L., Mori, G.: Social roles in hierarchical
models for human activity. In: CVPR (2012)
132. Lan, T., Wang, Y., Yang, W., Robinovitch, S.N., Mori,
G.: Discriminative latent models for recognizing contextual group activities. TPAMI **34** (8), 1549–1562 (2012)
133. Laptev, I.: On space-time interest points. IJCV **64** (2),
107–123 (2005)
134. Laptev, I., Lindeberg, T.: Space-time interest points. In:
ICCV, pp. 432–439 (2003)
135. Laptev, I., Marszalek, M., Schmid, C., Rozenfeld, B.:
Learning realistic human actions from movies. In:
CVPR (2008)
136. Laptev, I., Marsza�lek, M., Schmid, C., Rozenfeld, B.:
Learning realistic human actions from movies (2008)
137. Laptev, I., Perez, P.: Retrieving actions in movies. In:
ICCV (2007)
138. Le, Q.V., Zou, W.Y., Yeung, S.Y., Ng, A.Y.: Learning
hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis. In:
CVPR (2011)
139. Lee, H.Y., Huang, J.B., Singh, M., Yang, M.H.: Unsupervised representation learning by sorting sequences.
In: Proceedings of the IEEE International Conference
on Computer Vision, pp. 667–676 (2017)
140. Lee, N., Choi, W., Vernaza, P., Choy, C.B., Torr, P.H.,
Chandraker, M.: Desire: Distant future prediction in dynamic scenes with interacting agents. In: CVPR (2017)
141. Lee, N., Kitani, K.M.: Predicting wide receiver trajectories in american football. In: WACV2016
142. Li, J., Ma, H., Tomizuka, M.: Conditional generative
neural system for probabilistic trajectory prediction.
In: 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 6150–6156. IEEE
(2019)
143. Li, K., Fu, Y.: Prediction of human activity by discovering temporal sequence patterns. IEEE Transactions on
Pattern Analysis and Machine Intelligence **36** (8), 1644–
1657 (2014)
144. Li, K., Hu, J., Fu, Y.: Modeling complex temporal composition of actionlets for activity prediction. In: ECCV
(2012)
145. Li, W., Zhang, Z., Liu, Z.: Action recognition based on
a bag of 3d points. In: CVPR workshop (2010)
146. Li, Y., Chen, L., He, R., Wang, Z., Wu, G., Wang,
L.: Multisports: A multi-person video dataset of spatiotemporally localized sports actions. In: ICCV (2021)
147. Li, Z., Yao, L.: Three birds with one stone: Multi-task
temporal action detection via recycling temporal annotations. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR),
pp. 4751–4760 (2021)



148. Liang, J., Jiang, L., Niebles, J.C., Hauptmann, A.G.,
Fei-Fei, L.: Peeking into the future: Predicting future
person activities and locations in videos. In: Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 5725–5734 (2019)
149. Lin, T., Liu, X., Li, X., Ding, E., Wen, S.: Bmn:
Boundary-matching network for temporal action proposal generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3889–
3898 (2019)
150. Lin, T., Zhao, X., Su, H., Wang, C., Yang, M.: Bsn:
Boundary sensitive network for temporal action proposal generation. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 3–19 (2018)
151. Lin, Y.Y., Hua, J.H., Tang, N.C., Chen, M.H., Liao,
H.Y.M.: Depth and skeleton associated action recognition without online accessible rgb-d cameras. In: CVPR
(2014)
152. Liu, J., Kuipers, B., Savarese, S.: Recognizing human
actions by attributes. In: CVPR (2011)
153. Liu, J., Luo, J., Shah, M.: Recognizing realistic actions
from videos “in the wild”. In: Proc. IEEE Conf. on
Computer Vision and Pattern Recognition (2009)
154. Liu, J., Shahroudy, A., Perez, M., Wang, G., Duan, L.Y.,
Kot, A.C.: Ntu rgb+d 120: A large-scale benchmark for
3d human activity understanding. IEEE Transactions
on Pattern Analysis and Machine Intelligence **42** (10),
2684–2701 (2020)
155. Liu, J., Shahroudy, A., Xu, D., Wang, G.: Spatiotemporal lstm with trust gates for 3d human action
recognition. In: European Conference on Computer Vision, pp. 816–833. Springer (2016)
156. Liu, L., Shao, L.: Learning discriminative representations from rgb-d video data. In: IJCAI (2013)
157. Liu, X., Pintea, S.L., Nejadasl, F.K., Booij, O., van
Gemert, J.C.: No frame left behind: Full video action
recognition. In: CVPR (2021)
158. Liu, Y., Ma, L., Zhang, Y., Liu, W., Chang, S.F.: Multigranularity generator for temporal action proposal. In:
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 3604–3613 (2019)
159. Liu, Y., Yan, Q., Alahi, A.: Social nce: Contrastive
learning of socially-aware motion representations. arXiv
preprint arXiv:2012.11717 (2020)
160. Lu, C., Jia, J., Tang, C.K.: Range-sample depth feature
for action recognition. In: CVPR (2014)
161. Lucas, B.D., Kanade, T.: An iterative image registration
technique with an application to stereo vision. In: Proceedings of Imaging Understanding Workshop (1981)
162. Luo, G., Yang, S., Tian, G., Yuan, C., Hu, W., Maybank, S.J.: Learning human actions by combining global
dynamics and local appearance. IEEE Transactions
on Pattern Analysis and Machine Intelligence **36** (12),
2466–2482 (2014)
163. Luo, J., Wang, W., Qi, H.: Group sparsity and geometry constrained dictionary learning for action recognition from depth maps. In: ICCV (2013)
164. Luo, Z., Hsieh, J.T., Jiang, L., Carlos Niebles, J., FeiFei, L.: Graph distillation for action detection with privileged modalities. In: ECCV (2018)
165. Ma, S., Sigal, L., Sclaroff, S.: Learning activity progression in lstms for activity detection and early detection.
In: CVPR (2016)
166. Mainprice, J., Hayne, R., Berenson, D.: Goal set inverse
optimal control and iterative re-planning for predicting
human reaching motions in shared workspace. In: arXiv
preprint arXiv:1606.02111 (2016)


Human Action Recognition and Prediction: A Survey 33



167. Mangalam, K., An, Y., Girase, H., Malik, J.: From goals,
waypoints & paths to long term human trajectory forecasting. arXiv preprint arXiv:2012.01526 (2020)
168. Mangalam, K., Girase, H., Agarwal, S., Lee, K.H., Adeli,
E., Malik, J., Gaidon, A.: It is not the journey but
the destination: Endpoint conditioned trajectory prediction. In: European Conference on Computer Vision,
pp. 759–776. Springer (2020)
169. Marchetti, F., Becattini, F., Seidenari, L., Bimbo, A.D.:
Mantra: Memory augmented networks for multiple trajectory prediction. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 7143–7152 (2020)
170. Marsza�lek, M., Laptev, I., Schmid, C.: Actions in context. In: IEEE Conference on Computer Vision & Pattern Recognition (2009)
171. Mass, J., Johansson, G., Jason, G., Runeson, S.: Motion
perception I and II [film]. Boston: Houghton Mifflin
(1971)
172. Mehrasa, N., Jyothi, A.A., Durand, T., He, J., Sigal, L.,
Mori, G.: A variational auto-encoder model for stochastic point processes. In: CVPR (2019)
173. Messing, R., Pal, C., Kautz, H.: Activity recognition
using the velocity histories of tracked keypoints. In:
ICCV (2009)
174. Mingfei Gao Yingbo Zhou, R.X.R.S.C.X.: Woad:
Weakly supervised online action detection in untrimmed
videos. In: CVPR (2021)
175. Mishra, A., Verma, V., Reddy, M.K.K., Subramaniam,
A., Rai, P., Mittal, A.: A generative approach to zeroshot and few-shot action recognition (2018)
176. Misra, I., Zitnick, C.L., Hebert, M.: Shuffle and learn:
unsupervised learning using temporal order verification.
In: European Conference on Computer Vision, pp. 527–
544. Springer (2016)
177. Mohamed, A., Qian, K., Elhoseiny, M., Claudel, C.:
Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14424–14432
(2020)
178. Monfort, M., Zhou, B., Bargal, S.A., Yan, T., Andonian,
A., Ramakrishnan, K., Brown, L., Fan, Q., Gutfruend,
D., Vondrick, C., et al.: Moments in time dataset: one
million videos for event understanding
179. Morency, L.P., Quattoni, A., Darrell, T.: Latentdynamic discriminative models for continuous gesture
recognition. In: CVPR (2007)
180. Morrisand, B., Trivedi, M.: Trajectory learning for
activity understanding: Unsupervised, multilevel, and
long-term adaptive approach. Pattern Analysis and Machine Intelligence, IEEE Transactions on **33** (11), 2287–
2301 (2011)
181. Narayan, S., Cholakkal, H., Khan, F.S., Shao, L.:
3C-Net: Category count and center loss for weaklysupervised action localization. In: ICCV (2019)
182. Narayanan, S., Moslemi, R., Pittaluga, F., Liu, B.,
Chandraker, M.: Divide-and-conquer for lane-aware diverse trajectory prediction. In: CVPR (2021)
183. Ng, J.Y.H., Hausknecht, M., Vijayanarasimhan, S.,
Vinyals, O., Monga, R., Toderici, G.: Beyond short snippets: Deep networks for video classification. In: CVPR
(2015)
184. Ni, B., Wang, G., Moulin, P.: RGBD-HuDaAct: A colordepth video database for human daily activity recognition. In: ICCV Workshop on CDC3CV (2011)



185. Niebles, J.C., Chen, C.W., Fei-Fei, L.: Modeling temporal structure of decomposable motion segments for
activity classification. In: ECCV (2010)
186. Niebles, J.C., Fei-Fei, L.: A hierarchical model of shape
and appearance for human action classification. In:
CVPR (2007)
187. Niebles, J.C., Wang, H., Fei-Fei, L.: Unsupervised learning of human action categories using spatial-temporal
words. International Journal of Computer Vision **79** (3),
299–318 (2008)
188. Ofli, F., Chaudhry, R., Kurillo, G., Vidal, R., Bajcsy, R.:
Berkeley mhad: A comprehensive multimodal human
action database. In: Proceedings of the IEEE Workshop
on Applications on Computer Vision (2013)
189. Oliver, N.M., Rosario, B., Pentland, A.P.: A bayesian
computer vision system for modeling human interactions. PAMI **22** (8), 831–843 (2000)
190. Oreifej, O., Liu, Z.: Hon4d: Histogram of oriented 4d
normals for activity recognition from depth sequences.
In: CVPR (2013)
191. Oza, P., Patel, V.M.: C2AE: Class conditioned autoencoder for open-set recognition. In: CVPR (2019)
192. Patron-Perez, A., Marszalek, M., Reid, I., Zissermann,
A.: Structured learning of human interaction in tv
shows. PAMI **34** (12), 2441–2453 (2012)
193. Patron-Perez, A., Marszalek, M., Zisserman, A., Reid,
I.: High five: Recognising human interactions in tv
shows. In: Proc. British Conference on Machine Vision
(2010)
194. Pei, M., Jia, Y., Zhu, S.C.: Parsing video events with
goal inference and intent prediction. In: ICCV, pp. 487–
494. IEEE (2011)
195. Perera, P., Morariu, V.I., Jain, R., Manjunatha, V.,
Wigington, C., Ordonez, V., Patel, V.M.: Generativediscriminative feature representations for open-set
recognition. In: CVPR (2020)
196. Perrett, T., Masullo, A., Burghardt, T., Mirmehdi, M.,
Damen, D.: Temporal-relational crosstransformers for
few-shot action recognition. In: CVPR (2021)
197. Perronnin, F., Dance, C.: Fisher kernels on visual vocabularies for image categorization. In: CVPR (2006)
198. Plotz, T., Hammerla, N.Y., Olivier, P.: Feature learning
for activity recognition in ubiquitous computing. In:
IJCAI (2011)
199. Poppe, R.: A survey on vision-based human action
recognition. Image and Vision Computing **28**, 976–990
(2010)
200. Purushwalkam, S., Gupta, A.: Pose from action: Unsupervised learning of pose features based on motion.
arXiv preprint arXiv:1609.05420 (2016)
201. Qiu, Z., Yao, T., Mei, T.: Learning spatio-temporal representation with pseudo-3d residual network. In: ICCV
(2017)
202. Qiu, Z., Yao, T., Ngo, C.W., Tian, X., Mei, T.: Learning spatio-temporal representation with local and global
diffusion. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pp. 12056–
12065 (2019)
203. Rajko, S., Qian, G., Ingalls, T., James, J.: Real-time
gesture recognition with minimal training requirements
and on-line learning. In: CVPR (2007)
204. Ramanathan, V., Yao, B., Fei-Fei, L.: Social role discovery in human events. In: CVPR (2013)
205. Ramezani, M., Yaghmaee, F.: A review on human action
analysis in videos for retrieval applications. Artificial
Intelligence Review **46** (4), 485–514 (2016)


34 Yu Kong, Yun Fu


206. Raptis, M., Sigal, L.: Poselet key-framing: A model for 226. Scheirer, W.J., de Rezende Rocha, A., Sapkota, A.,
human activity recognition. In: CVPR (2013) Boult, T.E.: Toward open set recognition. IEEE trans207. Raptis, M., Soatto, S.: Tracklet descriptors for action actions on pattern analysis and machine intelligence
modeling and video analysis. In: ECCV (2010) **35** (7), 1757–1772 (2012)
208. Rasouli, A., Rohani, M., Luo, J.: Bifold and semantic 227. Sch¨uldt, C., Laptev, I., Caputo, B.: Recognizing human
reasoning for pedestrian behavior prediction. In: CVPR actions: A local svm approach. In: IEEE ICPR (2004)
(2021) 228. Scovanner, P., Ali, S., Shah, M.: A 3-dimensional sift
209. Reddy, K.K., Shah, M.: Recognizing 50 human action descriptor and its application to action recognition. In:
categories of web videos. Machine Vision and Applica- Proc. ACM Multimedia (2007)
tions Journal (2012) 229. Shahroudy, A., Liu, J., Ng, T.T., Wang, G.: Ntu rgb+d:
210. Ricoeur, P.: Oneself as another (K. Blamey, Trans.). A large scale dataset for 3d human activity analysis.
Chicago: University of Chicago Press (1992) In: IEEE Conference on Computer Vision and Pattern
211. Rizzolatti, G., Craighero, L.: The mirror-neuron system. Recognition (2016)
Annu. Rev. Neurosci. **27**, 169–192 (2004) 230. Shahroudy, A., Liu, J., Ng, T.T., Wang, G.: Ntu rgb+d:

A large scale dataset for 3d human activity analysis. In:

212. Rizzolatti, G., Sinigaglia, C.: The functional role of the

CVPR (2016)

parieto-frontal mirror circuit: interpretations and misinterpretations. Nat. Rev. Neurosci. **11**, 264–274 (2010) 231. Shi, Q., Cheng, L., Wang, L., Smola, A.: Human action

segmentation and recognition using discriminative semi
213. Rodriguez, M.D., Ahmed, J., Shah, M.: Action mach:

markov models. IJCV **93**, 22–32 (2011)

A spatio-temporal maximum average correlation height

232. Shotton, J., Girshick, R., Fitzgibbon, A., Sharp, T.,

filter for action recognition. In: CVPR (2008)

Cook, M., Finocchio, M., Moore, R., Kohli, P., Crim
214. Rohit, G., Kristen, G.: Anticipative video transformer.

inisi, A., Kipman, A., Blake, A.: Efficient human pose

In: ICCV (2021)

estimation from single depth images. PAMI (2013)

215. Roitberg, A., Ma, C., Haurilet, M., Stiefelhagen, R.:

233. Shou, Z., Chan, J., Zareian, A., Miyazawa, K., Chang,

Open set driver activity recognition. In: IVS (2020) S.F.: CDC: Convolutional-de-convolutional networks
216. Ryoo, M., Aggarwal, J.: Recognition of composite hu
for precise temporal action localization in untrimmed

man activities through context-free grammar based rep
videos. In: CVPR (2017)

resentation. In: CVPR, vol. 2, pp. 1709–1718 (2006)

234. Shou, Z., Wang, D., Chang, S.F.: Temporal action lo
217. Ryoo, M., Aggarwal, J.: Spatio-temporal relationship

calization in untrimmed videos via multi-stage CNNs.

match: Video structure comparison for recognition of

In: CVPR (2016)

complex human activities. In: ICCV, pp. 1593–1600

235. Shu, Y., Shi, Y., Wang, Y., Zou, Y., Yuan, Q., Tian,

(2009)

Y.: ODN: Opening the deep network for open-set action

218. Ryoo, M., Aggarwal, J.: Stochastic representation and recognition. In: ICME (2018)
recognition of high-level group activities. IJCV **93**, 183–

236. Si, C., Chen, W., Wang, W., Wang, L., Tan, T.: An

200 (2011) attention enhanced graph convolutional lstm network
219. Ryoo, M., Fuchs, T.J., Xia, L., Aggarwal, J.K., for skeleton-based action recognition. In: Proceedings of
Matthies, L.: Robot-centric activity prediction from the IEEE Conference on Computer Vision and Pattern
first-person videos: What will they do to me? In: Pro- Recognition, pp. 1227–1236 (2019)
ceedings of the Tenth Annual ACM/IEEE International 237. Simonyan, K., Zisserman, A.: Two-stream convolutional
Conference on Human-Robot Interaction, pp. 295–302. networks for action recognition in videos. In: NIPS
ACM (2015) (2014)
220. Ryoo, M.S.: Human activity prediction: Early recogni- 238. Singh, S., Velastin, S.A., Ragheb, H.: Muhavi: A multion of ongoing activities from streaming videos. In: ticamera human action video dataset for the evaluation
ICCV (2011) of action recognition methods. In: Advanced Video and
221. Ryoo, M.S., Aggarwal, J.K.: UT-Interaction Signal Based Surveillance (AVSS), 2010 Seventh IEEE
Dataset, ICPR contest on Semantic De- International Conference on, pp. 48–55. IEEE (2010)
scription of Human Activities (SDHA). 239. Sminchisescu, C., Kanaujia, A., Li, Z., Metaxas, D.:
http://cvrc.ece.utexas.edu/SDHA2010/Human ~~I~~ nteraction.html Conditional models for contextual human motion recog(2010) nition. In: International Conference on Computer Vision
222. S Singh, S.V., Ragheb, H.: Muhavi: A multicamera (2005)
human action video dataset for the evaluation of ac- 240. Song, H., Wu, X., Zhu, B., Wu, Y., Chen, M., Jia, Y.:
tion recognition methods. In: 2nd Workshop on Ac- Temporal action localization in untrimmed videos using
tivity monitoring by multi-camera surveillance systems action pattern trees. IEEE Transactions on Multimedia
(AMMCSS), pp. 48–55 (2010) (TMM) **21** (3), 717–730 (2019)
223. Sadeghian, A., Kosaraju, V., Sadeghian, A., Hirose, N., 241. Song, L., Zhang, S., Yu, G., Sun, H.: TACNet:
Rezatofighi, H., Savarese, S.: Sophie: An attentive gan Transition-aware context network for spatio-temporal
for predicting paths compliant to social and physical action detection. In: CVPR (2019)
constraints. In: Proceedings of the IEEE/CVF Confer- 242. Song, S., Lan, C., Xing, J., Zeng, W., Liu, J.: Spatioence on Computer Vision and Pattern Recognition, pp. temporal attention-based LSTM networks for 3d action
1349–1358 (2019) recognition and detection. IEEE Transactions on Image
224. Satkin, S., Hebert, M.: Modeling the temporal extent of Processing (TIP) **27** (7), 3459–3471 (2018)
actions. In: ECCV (2010) 243. Su, H., Zhu, J., Dong, Y., Zhang, B.: Forecast the plau225. Scheirer, W.J., Jain, L.P., Boult, T.E.: Probability mod- sible paths in crowd scenes. In: IJCAI (2017)
els for open set recognition. IEEE transactions on pat- 244. Sumi, S.: Perception of point-light walker produced by
tern analysis and machine intelligence **36** (11), 2317– eight lights attached to the back of the walker. Swiss J.
2324 (2014) Psychol. **59**, 126–32 (2000)


Human Action Recognition and Prediction: A Survey 35



245. Sun, D., Roth, S., Black, M.J.: Secrets of optical flow
estimation and their principles. In: CVPR (2010)
246. Sun, J., Wu, X., Yan, S., Cheong, L., Chua, T., Li, J.: Hierarchical spatio-temporal context modeling for action
recognition. In: CVPR (2009)
247. Sun, L., Jia, K., Chan, T.H., Fang, Y., Wang, G., Yan,
S.: Dl-sfa: Deeply-learned slow feature analysis for action recognition. In: CVPR (2014)
248. Sung, J., Ponce, C., Selman, B., Saxena, A.: Human
activity detection from rgbd images. In: AAAI workshop
on Pattern, Activity and Intent Recognition (2011)
249. Sung, J., Ponce, C., Selman, B., Saxena, A.: Unstructured human activity detection from rgbd images. In:
ICRA (2012)
250. Sur´ıs, D., Liu, R., Vondrick, C.: Learning the predictability of the future. In: CVPR (2021)
251. Tang, K., Fei-Fei, L., Koller, D.: Learning latent temporal structure for complex event detection. In: CVPR
(2012)
252. Tang, K., Ramanathan, V., Fei-Fei, L., Koller, D.: Shifting weights: Adapting object detectors from image to
video. In: Advances in Neural Information Processing
Systems (2012)
253. Tang, Y., Ding, D., Rao, Y., Zheng, Y., Zhang, D., Zhao,
L., Lu, J., Zhou, J.: COIN: A large-scale dataset for
comprehensive instructional video analysis. In: CVPR
(2019)
254. Taylor, G.W., Fergus, R., LeCun, Y., Bregler, C.: Convolutional learning of spatio-temporal features. In:
ECCV (2010)
255. Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri,
M.: Learning spatiotemporal features with 3d convolutional networks. In: ICCV (2015)
256. Tran, D., Sorokin, A.: Human activity recognition with
metric learning. In: ECCV (2008)
257. Troje, N.: Decomposing biological motion: a framework
for analysis and synthesis of human gait patterns. J.
Vis. **2**, 371–87 (2002)
258. Troje, N., Westhoff, C., Lavrov, M.: Person identification from biological motion: effects of structural and
kinematic cues. Percept. Psychophys **67**, 667–75 (2005)
259. Turek, M., Hoogs, A., Collins, R.: Unsupervised learning of functional categories in video scenes. In: ECCV
(2010)
260. Vahdat, A., Gao, B., Ranjbar, M., Mori, G.: A discriminative key pose sequence model for recognizing human interactions. In: ICCV Workshops, pp. 1729 –1736
(2011)
261. Varol, G., Laptev, I., Schmid, C.: Long-term temporal
convolutions for action recognition. IEEE Transactions
on Pattern Analysis and Machine Intelligence (2017)
262. Vondrick, C., Pirsiavash, H., Torralba, A.: Anticipating
visual representations from unlabeled video. In: CVPR
(2016)
263. Walker, J., Gupta, A., Hebert, M.: Patch to the future: Unsupervised visual prediction. In: Proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3302–3309 (2014)
264. Wang, C., Wang, Y., Xu, M., Crandall, D.J.: Stepwise
goal-driven networks for trajectory prediction. arXiv
preprint arXiv:2103.14107 (2021)
265. Wang, H., Kla _a_ ser, A., Schmid, C., Liu, C.L.: Dense
trajectories and motion boundary descriptors for action
recognition. IJCV **103** (60-79) (2013)
266. Wang, H., Kl¨aser, A., Schmid, C., Liu, C.L.: Action
Recognition by Dense Trajectories. In: IEEE Conference



on Computer Vision & Pattern Recognition, pp. 3169–
3176. Colorado Springs, United States (2011). URL
```
  http://hal.inria.fr/inria-00583818/en
```

267. Wang, H., Oneata, D., Verbeek, J., Schmid, C.: A robust
and efficient video representation for action recognition.
IJCV (2015)
268. Wang, H., Schmid, C.: Action recognition with improved trajectories. In: IEEE International Conference
on Computer Vision. Sydney, Australia (2013). URL
```
  http://hal.inria.fr/hal-00873267
```

269. Wang, H., Ullah, M.M., Kl _a_ ser, A., Laptev, I., Schmid,
C.: Evaluation of local spatio-temporal features for action recognition. In: BMVC (2009)
270. Wang, J., Liu, Z., Chorowski, J., Chen, Z., Wu, Y.: Robust 3d action recognition with random occupancy patterns. In: ECCV (2012)
271. Wang, J., Liu, Z., Wu, Y., Yuan, J.: Mining actionlet
ensemble for action recognition with depth cameras. In:
CVPR (2012)
272. Wang, K., Wang, X., Lin, L., Wang, M., Zuo, W.: 3d
human activity recognition with reconfigurable convolutional neural networks. In: ACM Multimedia (2014)
273. Wang, L., Qiao, Y., Tang, X.: Action recognition and detection by combining motion and appearance features.
In: ECCV THUMOS Workshop (2014)
274. Wang, L., Qiao, Y., Tang, X.: Action recognition with
trajectory-pooled deep-convolutional descriptors. In:
CVPR (2015)
275. Wang, L., Suter, D.: Recognizing human activities from
silhouettes: Motion subspace and factorial discriminative graphical model. In: CVPR (2007)
276. Wang, L., Tong, Z., Ji, B., Wu, G.: Tdn: Temporal
difference networks for efficient action recognition. In:
CVPR, pp. 1895–1904 (2021)
277. Wang, L., Xiong, Y., Lin, D., Van Gool, L.: UntrimmedNets for weakly supervised action recognition and detection. In: CVPR (2017)
278. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang,
X., Gool, L.V.: Temoral segment networks: Toward good
practices for deep action recognition. In: ECCV (2016)
279. Wang, S.B., Quattoni, A., Morency, L.P., Demirdjian,
D., Darrell, T.: Hidden conditional random fields for
gesture recognition. In: CVPR (2006)
280. Wang, X., Gupta, A.: Unsupervised learning of visual
representations using videos. In: Proceedings of the
IEEE International Conference on Computer Vision, pp.
2794–2802 (2015)
281. Wang, X., He, K., Gupta, A.: Transitive invariance for
self-supervised visual representation learning. In: Proceedings of the IEEE international conference on computer vision, pp. 1329–1338 (2017)
282. Wang, Y., Mori, G.: Learning a discriminative hidden
part model for human action recognition. In: NIPS
(2008)
283. Wang, Y., Mori, G.: Hidden part models for human action recognition: Probabilistic vs. max-margin. PAMI
(2010)
284. Wang, Z., Wang, J., Xiao, J., Lin, K.H., Huang, T.S.:
Substructural and boundary modeling for continuous
action recognition. In: CVPR (2012)
285. Weinland, D., Ronfard, R., Boyer, E.: Free viewpoint
action recognition using motion history volumes. Computer Vision and Image Understanding **104** (2-3), 249–
257 (2006)
286. Willems, G., Tuytelaars, T., Gool, L.: An efficient dense
and scale-invariant spatio-temporal interest poing detector. In: ECCV (2008)


36 Yu Kong, Yun Fu



287. Wolf, C., Lombardi, E., Mille, J., Celiktutan, O., Jiu,
M., Dogan, E., Eren, G., Baccouche, M., Dellandr´ea, E.,
Bichot, C.E., et al.: Evaluation of video activity localizations integrating quality and quantity measurements.
Computer Vision and Image Understanding **127**, 14–30
(2014)
288. Wong, S.F., Kim, T.K., Cipolla, R.: Learning motion
categories using both semantic and structural information. In: CVPR (2007)
289. Wu, B., Yuan, C., Hu, W.: Human action recognition
based on context-dependent graph kernels. In: Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 2609–2616 (2014)
290. Wu, J., Yildirim, I., Lim, J.J., Freeman, W.T., Tenenbaum, J.B.: Galileo: Perceiving physical object properties by integrating a physics engine with deep learning.
In: Advances in Neural Information Processing Systems,
pp. 127–135 (2015)
291. Wu, X., Xu, D., Duan, L., Luo, J.: Action recognition
using context and appearance distribution features. In:
CVPR (2011)
292. Wu, Z., Wang, X., Jiang, Y.G., Ye, H., Xue, X.: Modeling spatial-temporal clues in a hybrid deep learning
framework for video classification. In: ACM Multimedia (2015)
293. Wulfmeier, M., Wang, D., Posner, I.: Watch this: scalable cost function learning for path planning in urban environment. In: arXiv preprint arXiv:1607:02329
(2016)
294. Xia, L., Aggarwal, J.: Spatio-temporal depth cuboid
similarity feature for activity recognition using depth
camera. In: CVPR (2013)
295. Xia, L., Chen, C., Aggarwal, J.: View invariant human
action recognition using histograms of 3d joints. In:
Computer Vision and Pattern Recognition Workshops
(CVPRW), 2012 IEEE Computer Society Conference
on, pp. 20–27. IEEE (2012)
296. Xia, L., Chen, C.C., Aggarwal, J.K.: View invariant human action recognition using histograms of 3d joints.
In: CVPRW (2012)
297. Xu, D., Xiao, J., Zhao, Z., Shao, J., Xie, D., Zhuang, Y.:
Self-supervised spatiotemporal learning via video clip
order prediction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.
10334–10343 (2019)
298. Xu, H., Das, A., Saenko, K.: R-c3d: Region convolutional 3d network for temporal activity detection. In:
Proceedings of the IEEE international conference on
computer vision, pp. 5783–5792 (2017)
299. Xu, H., Das, A., Saenko, K.: Two-stream region convolutional 3d network for temporal activity detection.
IEEE transactions on pattern analysis and machine intelligence **41** (10), 2319–2332 (2019)
300. Xu, M., Gao, M., Chen, Y.T., Davis, L.S., Crandall,
D.J.: Temporal recurrent networks for online action detection. In: ICCV (2019)
301. Yan, S., Xiong, Y., Lin, D.: Spatial temporal graph convolutional networks for skeleton-based action recognition. In: Thirty-Second AAAI Conference on Artificial
Intelligence (2018)
302. Yang, H., He, X., Porikli, F.: One-shot action localization by learning sequence matching network. In: CVPR
(2018)
303. Yang, S., Yuan, C., Wu, B., Hu, W., Wang, F.: Multifeature max-margin hierarchical bayesian model for action recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.
1610–1618 (2015)



304. Yang, W., Zhang, T., Yu, X., Qi, T., Zhang, Y., Wu,
F.: Uncertainty guided collaborative training for weakly
supervised temporal action detection. In: CVPR (2021)
305. Yang, X., Tian, Y.: Super normal vector for activity
recognition using depth sequences. In: CVPR (2014)
306. Yang, X., Yang, X., Liu, M.Y., Xiao, F., Davis, L.S.,
Kautz, J.: STEP: Spatio-temporal progressive learning
for video action detection. In: CVPR (2019)
307. Yang, Y., Hou, C., Lang, Y., Guan, D., Huang, D.,
Xu, J.: Open-set human activity recognition based on
micro-doppler signatures. Pattern Recognition **85**, 60–
69 (2019)
308. Yang, Y., Shah, M.: Complex events detection using
data-driven concepts. In: ECCV (2012)
309. Yao, B., Fei-Fei, L.: Action recognition with exemplar
based 2.5d graph matching. In: ECCV (2012)
310. Yao, B., Fei-Fei, L.: Recognizing human-object interactions in still images by modeling the mutual context of
objects and human poses. TPAMI **34** (9), 1691–1703
(2012)
311. Yeffet, L., Wolf, L.: Local trinary patterns for human
action recognition. In: CVPR (2009)
312. Yeung, S., Russakovsky, O., Mori, G., Fei-Fei, L.: Endto-end learning of action detection from frame glimpses
in videos. In: Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 2678–2687
(2016)
313. Yilmaz, A., Shah, M.: Actions sketch: A novel action
representation. In: CVPR (2005)
314. Yu, G., Liu, Z., Yuan, J.: Discriminative orderlet mining
for real-time recognition of human-object interaction.
In: ACCV (2014)
315. Yu, T., Ren, Z., Li, Y., Yan, E., Xu, N., Yuan, J.: Temporal structure mining for weakly supervised action detection. In: ICCV (2019)
316. Yu, T.H., Kim, T.K., Cipolla, R.: Real-time action
recognition by spatiotemporal semantic and structural
forests. In: BMVC (2010)
317. Yuan, C., Hu, W., Tian, G., Yang, S., Wang, H.: Multitask sparse learning with beta process prior for action
recognition. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pp. 423–429
(2013)
318. Yuan, C., Li, X., Hu, W., Ling, H., Maybank, S.J.: 3d r
transform on spatio-temporal interest points for action
recognition. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pp. 724–730
(2013)
319. Yuan, C., Li, X., Hu, W., Ling, H., Maybank, S.J.: Modeling geometric-temporal context with directional pyramid co-occurrence for action recognition. IEEE Transactions on Image Processing **23** (2), 658–672 (2014)
320. Yuan, C., Wu, B., Li, X., Hu, W., Maybank, S.J., Wang,
F.: Fusing r features and local features with contextaware kernels for action recognition. International Journal of Computer Vision **118** (2), 151–171 (2016)
321. Yuan, J., Liu, Z., Wu, Y.: Discriminative subvolume
search for efficient action detection. In: IEEE Conference on Computer Vision and Pattern Recognition
(2009)
322. Yuan, J., Liu, Z., Wu, Y.: Discriminative video pattern
search for efficient action detection. IEEE Transactions
on Pattern Analysis and Machine Intelligence (2010)
323. Yuan, Y., Weng, X., Ou, Y., Kitani, K.: Agentformer:
Agent-aware transformers for socio-temporal multiagent forecasting. arXiv preprint arXiv:2103.14023
(2021)


Human Action Recognition and Prediction: A Survey 37



324. Zeng, R., Huang, W., Tan, M., Rong, Y., Zhao, P.,
Huang, J., Gan, C.: Graph convolutional networks for
temporal action localization. In: ICCV (2019)
325. Zhai, X., Peng, Y., Xiao, J.: Cross-media retrieval by
intra-media and inter-media correlation mining. Multimedia Systems **19** (5), 395–406 (2013)
326. Zhang, H., Patel, V.M.: Sparse representation-based
open set recognition. IEEE transactions on pattern
analysis and machine intelligence **39** (8), 1690–1696
(2016)
327. Zhang, H., Zhang, L., Qi, X., Li, H., Torr, P.H.S., Koniusz, P.: Few-shot action recognition with permutationinvariant attention. In: ECCV (2020)
328. Zhao, H., Torralba, A., Torresani, L., Yan, Z.: HACS:
Human action clips and segments dataset for recognition
and temporal localization. In: ICCV (2019)
329. Zhao, H., Wildes, R.P.: Where are you heading? dynamic trajectory prediction with expert goal examples.
In: ICCV (2021)
330. Zhao, H., Yan, Z., Wang, H., Torresani, L., Torralba,
A.: Slac: A sparsely labeled dataset for action classification and localization. arXiv preprint arXiv:1712.09374
(2017)



331. Zhao, Y., Xiong, Y., Wang, L., Wu, Z., Tang, X., Lin,
D.: Temporal action detection with structured segment
networks. In: ICCV (2017)
332. Zhou, B., Andonian, A., Oliva, A., Torralba, A.: Temporal relational reasoning in videos. In: Proceedings of
the European Conference on Computer Vision (ECCV),
pp. 803–818 (2018)
333. Zhou, B., Wang, X., Tang, X.: Random field topic model
for semantic region analysis in crowded scenes from
tracklets. In: CVPR (2011)
334. Zhu, L., Yang, Y.: Compound memory networks for fewshot video classification. In: ECCV (2018)
335. Zhu, W., Lan, C., Xing, J., Zeng, W., Li, Y., Shen,
L., Xie, X.: Co-occurrence feature learning for skeleton based action recognition using regularized deep lstm
networks. In: Thirtieth AAAI Conference on Artificial
Intelligence (2016)
336. Ziebart, B., Maas, A., Bagnell, J., Dey, A.: Maximum
entropy inverse reinforcement learning. In: AAAI (2008)
337. Ziebart, B., Ratliff, N., Gallagher, G., Mertz, C., Peterson, K., Bagnell, J., Hebert, M., Dey, A., Srinivasa,
S.: Planning-based prediction for pedestrians. In: IROS
(2009)


