## **DriveGAN: Towards a Controllable High-Quality Neural Simulation**

Seung Wook Kim [1] _[,]_ [2] _[,]_ [3] Jonah Philion [1] _[,]_ [2] _[,]_ [3] Antonio Torralba [4] Sanja Fidler [1] _[,]_ [2] _[,]_ [3]

1 NVIDIA 2 University of Toronto 3 Vector Institute 4 MIT


_{_ seungwookk,jphilion,sfidler _}_ @nvidia.com torralba@mit.edu



**Abstract**


_Realistic simulators are critical for training and verify-_
_ing robotics systems. While most of the contemporary simu-_
_lators are hand-crafted, a scaleable way to build simulators_
_is to use machine learning to learn how the environment be-_
_haves in response to an action, directly from data. In this_
_work, we aim to learn to simulate a dynamic environment_
_directly in pixel-space, by watching unannotated sequences_
_of frames and their associated actions. We introduce a novel_
_high-quality neural simulator referred to as DriveGAN that_
_achieves controllability by disentangling different compo-_
_nents without supervision. In addition to steering controls,_
_it also includes controls for sampling features of a scene,_
_such as the weather as well as the location of non-player_
_objects. Since DriveGAN is a fully differentiable simulator,_
_it further allows for re-simulation of a given video sequence,_
_offering an agent to drive through a recorded scene again,_
_possibly taking different actions. We train DriveGAN on_
_multiple datasets, including 160 hours of real-world driv-_
_ing data. We showcase that our approach greatly surpasses_
_the performance of previous data-driven simulators, and al-_
_lows for new key features not explored before._


**1. Introduction**


The ability to _simulate_ is a key component of intelligence. Consider how animals make thousands of decisions
each day. Some of the decisions are critical for survival,
such as deciding to step away from an approaching car.
Mentally simulating the future given the current situation
is key in planning successfully. In robotic applications such
as autonomous driving, simulation is also a scaleable, robust and safe way of testing self-driving vehicles in safetycritical scenarios before deploying them in the real world.
Simulation further allows for a fair comparison of different
autonomous driving systems since one has control over the
repeatability of the scenarios.
Desired properties of a good robotic simulator include
accepting an action from an agent and generating a plausible next world state, allowing for user control over the scene
elements, and the ability to re-simulate an observed scenario



Figure 1: We aim to learn a controllable neural simulator that
can generate high-fidelity real-world scenes. DriveGAN takes user
controls ( _e.g_ . steering weel, speed) as input and renders the next
screen. It allows users to control different aspects of the scene,
such as weather and objects.


with plausible variations. This is no easy feat as the world
is incredibly rich in situations one can encounter. Most of
the existing simulators [10, 49, 34, 54] are hand-designed in
a game engine, which involves significant effort in content
creation, and designing complex behavior models to control
non-player objects. Grand Theft Auto, one of the most realistic driving games to date, set in a virtual replica of Los
Angeles, took several years to create and involved hundreds
of artists and engineers. In this paper, we advocate for datadriven simulation as a way to achieve scaleability.

Data-driven simulation has recently gained attention. LidarSim [42] used a catalog of annotated 3D scenes to sample layouts into which reconstructed objects obtained from
a large number of recorded drives are placed, in the quest
to achieve diversity for training and testing a LIDAR-based
perception system. [27, 9, 51], on the other hand, learn
to synthesize road-scene 3D layouts directly from images
without supervision. These works do not model the dynamics of the environment and object behaviors.

As a more daring alternative, recent works attempted to
create neural simulators [30, 15] that learn to simulate the
environment in response to the agent’s actions directly in
pixel-space by digesting large amounts of video data along
with actions. This line of work provides a scaleable way to
simulation, as we do not rely on any human-provided annotations, except for the agent’s actions which are cheap
to obtain from odometry sensors. It is also a more chal












1


lenging way, since the complexity of the world and the dynamic agents acting inside it, needs to be learned in a highresolution camera view. In this paper, we follow this route.
We introduce DriveGAN, a neural simulator that learns
from sequences of video footage and associated actions
taken by an ego-agent in an environment. DriveGAN leverages Variational-Auto Encoder [33] and Generative Adversarial Networks [14] to learn a latent space for images on
which a dynamics engine learns the transitions within the
latent space. The key aspects of DriveGAN are its disentangled latent space and high-resolution and high-fidelity
frame synthesis conditioned on the agent’s actions. The
disentanglement property of DriveGAN gives users additional control over the environment, such as changing the
weather and locations of non-player objects. Furthermore,
since DriveGAN is an end-to-end differentiable simulator,

we are able to re-create the scenarios observed from real

video footage allowing the agent to drive again through the
recorded scene but taking different actions. This property
makes DriveGAN the first neural driving simulator of its
kind. By learning on 160 hours of real driving data, we
showcase DriveGAN to learn high-fidelity simulation, surpassing all existing neural simulators by a significant margin, and allowing for the control over the environment not
possible previously.


**2. Related Work**


**2.1. Video Generation and Prediction**


As in image generation, the standard architectures for
video generation are VAEs [7, 21], auto-regressive models [50, 56, 26, 64], flow-based models [35], and GANs

[43, 62, 52, 53, 5, 60]. For a generator to sample videos, it
must be able to generate realistic looking frames as well as
realistic transitions between frames. Video prediction models [47, 39, 12, 45, 1, 37, 66] learn to produce future frames
given a reference frame, and they share many similarities to
video generation models. Similar architectures can be applied to the task of conditional video generation in which
information such as semantic segmentation is given as input to the model [63, 41]. In this work, we use a VAE-GAN

[36] based on StyleGAN [28] to learn a latent space of natural images, then train a dynamics model within the space.


**2.2. Data-driven Simulation and Model-based RL**


The goal of data-driven simulation is to learn simulators given observations from the environment to be simulated. Meta-Sim [27, 9] learns to produce scene parameters
in a synthetic scene. LiDARSim [42] leverages deep learning and physics engine to produce LiDAR point clouds.
In this work, we focus on data-driven simulators that produce future frames given controls. World Model [15] use
a VAE [33] and LSTM [20] to model transition dynamics





_x_ _t_





















_x_ _t_ +1



_a_ _t_


Figure 2: DriveGAN takes an image _x_ _t_ and action _a_ _t_ as input
at time _t_ . With encoder _ξ_, _x_ _t_ is encoded into disentangled latent
codes _z_ _t_ [theme] and _z_ _t_ [content] . Dynamics Engine learns the transition
function for the latent codes given _a_ _t_ . Image Generator produces
_x_ _t_ +1, which is fed to the next time step, autoregressively.


and rendering functionality. In GameGAN [30], a GAN
and a memory module are used to mimic the engine behind games such as Pacman and VizDoom. Model-based
RL [57, 6, 16, 25, 15] also aims at learning a dynamics
model of some environment which agents can utilize to plan
their actions. While prior work has applied neural simulation to simple environments [2, 58] in which a ground-truth
simulator is already known, we also apply our model to realworld driving data and focus on improving the quality of
simulations. Furthermore, we show how users can interatively edit scenes to create diverse simulation environments.


**3. Methodology**


Our objective is to learn a high-quality controllable neural simulator by watching sequences of video frames and
their associated actions. We aim to achieve controllability
in two aspects: 1) We assume there is an egocentric agent
that can be controlled by a given action. 2) We want to control different aspects of the current scene, for example, by
modifying an object or changing the background color.
Let us denote the video frame at time _t_ as _x_ _t_ and the
continuous action as _a_ _t_ . We learn to produce the next frame
_x_ _t_ +1 given the previous frames _x_ 1: _t_ and actions _a_ 1: _t_ . Fig 2
provides an overview of our model. Image encoder _ξ_ produces the disentangled latent codes _z_ [theme] and _z_ [content] for _x_
in an unsupervised manner. We define _theme_ as information
that does not depend on pixel locations such as the background color or weather of the scene, and _content_ as spatial
content (Fig 4). Dynamics Engine, a recurrent neural network, learns to produce the next latent codes _z_ _t_ [theme] +1 [,] _[ z]_ _t_ [content] +1
given _z_ _t_ [theme], _z_ _t_ [content], and _a_ _t_ . _z_ _t_ [theme] +1 and _z_ _t_ [content] +1 go through
an image decoder that generates the output image.
Generating high-quality temporally-consistent image sequences is a challenging problem [35, 43, 5, 60, 63, 41].
Rather than generating a sequence of frames directly, we
split the learning process into two steps, motivated by World
Model [15]. Sec 3.1 introduces our encoder-decoder architecture that is pre-trained to produce the latent space for
images. We propose a novel architecture that disentangles
themes and content while achieving high-quality generation
by leveraging a Variational Auto-Encoder (VAE) and Gen


2


_z_ [theme]





Figure 3: Pretraining stage learns the encoder and decoder for
images. The encoder _ξ_ produces _z_ [content] and _z_ [theme] which comprise
the disentangled latent space that the dynamics engine trains on.
The gaussian blocks represent reparameterization steps [33].


erative Adversarial Networks (GAN). Sec A.2 describes the
Dynamics Engine that learns the latent space dynamics. We
also show how the Dynamics Engine further disentangles
action-dependent and action-independent content.


**3.1. Pre-trained Latent Space**


We build our image decoder on top of the popular StyleGAN [28, 29], but make several modifications that allow
for theme-content disentanglement. Since extracting the
GAN’s latent code that corresponds to an input image is
not trivial, we introduce an encoder _ξ_ that maps an image
_x_ into its latent code _z_ . We utilize the VAE formulation,
particularly the _β_ -VAE [19] to control the KL term better.
Therefore, on top of the adversarial losses from StyleGAN,
we add the following loss at each step of generator training:


_L_ _V AE_ = _E_ _z∼q_ ( _z|x_ ) [ _log_ ( _p_ ( _x|z_ ))] + _βKL_ ( _q_ ( _z|x_ ) _||p_ ( _z_ ))


where _p_ ( _z_ ) is the standard normal prior distribution, _q_ ( _z|x_ )
is the approximate posterior from the encoder _ξ_, and _KL_
is the Kullback-Leibler divergence. For the reconstruction
term, we reduce the perceptual distance [67] between the
input and output images rather than the pixel-wise distance.
This form of combining VAE and GAN has been explored before [36]. To achieve our goal of controllable simulation, we introduce several novel modifications to the encoder and decoder. Firstly, we disentangle the _theme_ and
_content_ of the input image. Our encoder _ξ_ is composed of
a feature extractor _ξ_ [feat] and two encoding heads _ξ_ [content] and
_ξ_ [theme] (Figure 17). _ξ_ [feat] takes an image _x_ as input and consists of several convolution layers whose output is passed
to the two heads. _ξ_ [content] produces _z_ [content] _∈_ R _[N]_ _[×][N]_ _[×][D]_ [1]
which has _N × N_ spatial dimension. On the other hand,
_ξ_ [theme] produces _z_ [theme] _∈_ R _[D]_ [2], a single vector, which controls the theme of the output image. Let us denote _z_ =
_{z_ [content] _, z_ [theme] _}_ . Note that _z_ [content] and _z_ [theme] are matched to
be from the standard normal prior by the reparametrization
and training of VAE. We feed _z_ into the StyleGAN decoder.
StyleGAN controls the appearance of generated images
with adaptive instance normalization ( _AdaIN_ ) [11, 22, 13]
layers after each convolution layer of its generator. _AdaIN_
applies the same scaling and bias to each spatial location of
a normalized feature map:

_AdaIN_ ( **m** _, α, γ_ ) = _A_ ( **m** _, α, γ_ ) = _α_ **[m]** _[ −]_ _[µ]_ [(] **[m]** [)] + _γ_ (1)

_σ_ ( **m** )



Randomly Generated + Random _z_ [theme] + Random _z_ [theme] + Random _z_ [theme]


Figure 4: Left column shows randomly generated images from
different environments. By sampling _z_ [theme], we can change theme
information such as weather while keeping the content consistent.


where **m** _∈_ R _[N]_ _[×][N]_ _[×]_ [1] is a feature map with _N × N_ spatial
dimension and _α, γ_ are scalars for scaling and bias. Thus,
_AdaIN_ layers are perfect candidates for inserting _theme_
information. We pass _z_ [theme] through an _MLP_ to get the
scaling and bias values for each _AdaIN_ layer. Now, because of the shape of _z_ [content], it naturally encodes the content information from the corresponding _N × N_ grid locations. Rather than having a constant block as the input to
the first layer as in StyleGAN, we pass _z_ [content] as the input.
Furthermore, we can sample a new vector _v ∈_ R [1] _[×]_ [1] _[×][D]_ [1]
from the normal prior distribution to swap out the content
of some grid location. Preliminary experiments showed that
encoding information only using the plain StyleGAN decoder is not adequate for capturing the details of scenes with
multiple objects because the generator must recover spatial information from the inputs to _AdaIN_ layers, which
apply the same scaling and bias to all spatial locations.
We use the multi-scale multi-patch discriminator architecture [63, 24, 55], which results in higher quality images for
complex scenes. We use the same adversarial losses _L_ _GAN_
from StyleGAN, and the final loss function is _L_ _pretrain_ =
_L_ _V AE_ + _L_ _GAN_ .
We observe that balancing the KL loss with suitable _β_
in _L_ _V AE_ is essential. Smaller _β_ gives better reconstruction
quality, but the learned latent space could be far away from
the prior, in which case the dynamics model (Sec.A.2) had
a harder time learning the dynamics. This causes _z_ to be
overfit to _x_, and it becomes more challenging to learn the
transitions between frames in the overfitted latent space.


**3.2. Dynamics Engine**


With the pre-trained encoder and decoder, the Dynamics
Engine learns the transition between latent codes from one
time step to the next given an action _a_ _t_ . We fix the parameters of the encoder and decoder, and only learn the parameters of the engine. This allows us to pre-extract latent codes
for a dataset before training. The training process becomes
faster and significantly easier than directly working with
images, as latent codes typically have dimensionality much
smaller than the input. In addition, we further disentangle



3


Figure 5: Dynamics Engine produces the next latent codes, given
an action and previous latent codes. It disentangles content information into action-dependent and action-independent features
with its two separate LSTMs. Dashed lines correspond to temporal
connections. Gaussian blocks indicate reparameterization steps.


_content_ information from _z_ [content] into _action-dependent_ and
_action-independent_ features without supervision.
In a 3D environment, the view-point shifts as the ego
agent moves. This shifting naturally happens spatially, so
we employ a convolutional LSTM module (Figure 18) to
learn the spatial transition between each time step:


_v_ _t_ = _F_ ( _H_ ( _h_ [conv] _t−_ 1 _[, a]_ _[t]_ _[, z]_ _t_ [content] _, z_ _t_ [theme] )) (2)


_i_ _t_ _, f_ _t_ _, o_ _t_ = _σ_ ( _v_ _t_ _[i]_ [)] _[, σ]_ [(] _[v]_ _t_ _[f]_ [)] _[, σ]_ [(] _[v]_ _t_ _[o]_ [)] (3)


_c_ [conv] _t_ = _f_ _t_ _⊙_ _c_ [conv] _t−_ 1 [+] _[ i]_ _[t]_ _[⊙]_ [tanh(] _[v]_ _t_ _[g]_ [)] (4)


_h_ [conv] _t_ = _o_ _t_ _⊙_ tanh( _c_ [conv] _t_ ) (5)


where _h_ [conv] _t_ _, c_ [conv] _t_ are the hidden and cell state of the convLSTM module, and _i_ _t_ _, f_ _t_ _, o_ _t_ are the input, forget, output
gates, respectively. _H_ replicates _a_ _t_ and _z_ _t_ [theme] spatially to
match the _N × N_ spatial dimension of _z_ _t_ [content] . It fuses all
inputs by concatenating and running through a 1 _×_ 1 convolution layer. _F_ is composed of two 3 _×_ 3 convolution layers.
_v_ _t_ is split into intermediate variables _v_ _t_ _[i]_ _[, v]_ _t_ _[f]_ _[, v]_ _t_ _[o]_ _[, v]_ _t_ _[g]_ [. All state]
and intermediate variables have the same size R _[N]_ _[×][N]_ _[×][D]_ [conv] .

The hidden state _h_ [conv] _t_ goes through two separate convolution layers to produce _z_ _t_ [theme] +1 [and] _[ z]_ _t_ _[a]_ +1 [dep] [. The action dependent]
feature _z_ _t_ _[a]_ +1 [dep] [is used to produce] _[ z]_ _t_ [content] +1, along with _z_ _t_ _[a]_ +1 [indep] [.]
We also add a plain LSTM [20] module that only takes
_z_ _t_ as input. Therefore, this module is responsible for information that does not depend on the action _a_ _t_ . The input _z_ _t_
is flattened into a vector, and all variables inside this module have size R _[D]_ [linear] . The hidden state goes through a linear
layer that outputs _z_ _t_ _[a]_ +1 [indep] [. Finally,] _[ z]_ _t_ _[a]_ +1 [dep] [and] _[ z]_ _t_ _[a]_ +1 [indep] [are used as]
inputs to two _AdaIN_ + Conv blocks.


_**α**_ _,_ _**β**_ = _MLP_ ( _z_ _t_ _[a]_ +1 [indep] [)] (6)


_z_ _t_ [content] +1 = _C_ ( _A_ ( _C_ ( _A_ ( _z_ _t_ _[a]_ +1 [dep] _[,]_ _**[ α]**_ _[,]_ _**[ β]**_ [))] _[,]_ _**[ α]**_ _[,]_ _**[ β]**_ [))] (7)


where we denote convolution and _AdaIN_ layers as _C_ and
_A_, respectively. An _MLP_ is used to produce _**α**_ and _**β**_ . We



reparameterize _z_ _[a]_ [dep], _z_ _[a]_ [indep], _z_ [theme] into the standard normal
distribution _N_ (0 _, I_ ) which allows sampling at test time:


_z_ = _µ_ + _ϵσ,_ _ϵ ∼_ _N_ (0 _, I_ ) (8)


where _µ_ and _σ_ are the intermediate variables for the mean
and standard deviation for each reparameterization step.
Intuitively, _z_ _[a]_ [indep] is used as _style_ for the spatial tensor
_z_ _[a]_ [dep] through _AdaIN_ layers. _z_ _[a]_ [indep] does not get action information, so it alone cannot learn to generate plausible
next frames. This architecture thus allows disentangling
action-dependent features such as the layout of a scene from
action-independent features such as object types. Note that
the engine could ignore _z_ _[a]_ [indep] and only use _z_ _[a]_ [dep] to learn dynamics. If we keep the model size small and use a high _KL_
penalty on the reparameterized variables, it will utilize full
model capacity and make use of _z_ _[a]_ [indep] . We can also enforce
disentanglement between _z_ _[a]_ [indep] and _z_ _[a]_ [dep] using an adversarial loss [8]. In practice, we found that our model was able
to disentangle information well without such a loss.
**Training:** We extend the training procedure of
GameGAN [30] in latent space to train our model with
adversarial and VAE losses. Our adversarial losses _L_ _adv_
come from two networks: 1) single latent discriminator,
and 2) temporal action-conditioned discriminator. We first
flatten _z_ _t_ into a vector with size R _[N]_ [2] _[D]_ [1] [+] _[D]_ [2] . The single latent discriminator is an _MLP_ that tries to discriminate produced _z_ _t_ from the real latent codes. The temporal
action-conditioned discriminator is implemented as a temporal convolution network such that we apply filters in the
temporal dimension [31] where the actions _a_ _t_ are fused to
the temporal dimension. We also sample negative actions
_a_ ¯ _t_, and the job of the discriminator is to figure out if the
given sequence of latent codes is realistic and faithful to the
given action sequences. We use the temporal discriminator
features to reconstruct the input action sequence and reduce
the action reconstruction loss _L_ _action_ to help the dynamics
engine to be faithful to the given actions. Finally, we add
latent code reconstruction loss _L_ _latent_ so that the generated
_z_ _t_ matches the ground truth latent codes, and reduce the _KL_
penalty _L_ _KL_ for _z_ _t_ _[a]_ [dep], _z_ _t_ _[a]_ [indep], _z_ _t_ [theme] . The final loss function
is _L_ _DE_ = _L_ _adv_ + _L_ _latent_ + _L_ _action_ + _L_ _KL_ . Our model is
trained with 32 time-steps with a warm-up phase similar to
GameGAN. Further details are provided in the Appendix.


**3.3. Differentiable Simulation**


One compelling aspect of DriveGAN is that it can create
an editable simulation environment from a real video. As

DriveGAN is fully differentiable, it allows for recovering
the scene and scenario by discovering the underlying factors of variations that comprise a video, while also recovering the actions that the agent took, if these are not provided.
We refer to this as _differentiable simulation_ . Once these
parameters are discovered, the agent can use DriveGAN to



4


Real Video


**Optimized**
**Sequence**


Optimized
Sequence
**+ tree**


Optimized
Sequence
**+ building**


Optimized
Sequence
**+ building**
**+ foggy**


**Optimized**
**Actions**


for the underlying sequence of inputs that can reproduce a real
video. With its controllability, we can replay the same scenario
with modified content or scene condition.


re-simulate the scene and take different actions. DriveGAN

further allows sampling and modification of various components of a scene, thus testing the agent in the same scenario
under different weather conditions or objects.
First, note that reparametrization steps (Eq. 10) involve
a stochastic variable _ϵ_ which gives stochasticity in a simulation to produce diverse future scenarios. Given a sequence
of frames from a real video _x_ 0 _, ..., x_ _T_, our model can be
used to find the underlying _a_ 0 _, ..., a_ _T −_ 1 _, ϵ_ 0 _, ...ϵ_ _T −_ 1 :



generate the dataset. The ego-agent and other vehicles are
randomly placed and use random policy to drive in the environment. Each sequence has a randomly sampled weather
condition and consists of 80 frames sampled at 4Hz. 48K
sequences are extracted, and 43K are used for training.
**Gibson** environment [65] virtualizes real-world indoor
buildings and has an integrated physics engine with which
virtual agents can be controlled. We first train a reinforcement learning agent that can navigate towards a given destination coordinate. In each sequence, we randomly place
the agent in a building and sample a destination. 85K sequences each with 30 frames are extracted from 100 indoor
environments, and 76K sequences are used for training.
**Real World Driving** (RWD) data consists of real-world
recordings of human driving on multiple different highways
and cities. It was collected in a variety of different weather
and times. RWD is composed of 128K sequences each with
36 frames extracted at 8Hz. It corresponds to _∼_ 160 hours
of driving, and we use 125K sequences for training.
Figure 7 illustrates scenes from the datasets. Each sequence consists of the extracted frames (256 _×_ 256) and the
actions the ego agent takes at each time step. The 2-dim
actions consist of the agent’s speed and angular velocity.


**4.1. Quantitative Results**


The quality of simulators needs to be evaluated in two
aspects. The generated videos from simulators have to look
realistic, and their distribution should match the distribution
of the real videos. They also need to be faithful to the action
sequences used to produce them. This is essential to be useful for downstream tasks, such as training a robot. Therefore, we use two automatic metrics to measure the performance of models. The experiments are carried out by using
the first frames and action sequences of the test set. The
remaining frames are generated autoregressively.
We compare with four baseline models: Action-RNN [3]
is a simple action-conditioned RNN model trained with reconstruction loss on the pixel space, Stochastic Adversarial Video Prediction (SAVP) [37] and GameGAN [30] are
trained with adversarial loss along with reconstruction loss
on the pixel space, World Model [15] trains a vision model
based on VAE and an RNN based on mixture density networks (MDN-RNN). World Model is similar to our model
as they first extract latent codes and learn MDN-RNN on
top of the learned latent space. However, their VAE is not
powerful enough to model the complexities of the datasets
studied in this work. Fig 9 shows how a simple VAE cannot
reconstruct the inputs; thus, the plain World Model cannot
produce realistic video sequences by default. Therefore, we
include a variant, denoted as World Model*, that uses our
proposed latent space to train the MDN-RNN component.

We also conduct human evaluations with Amazon Me
chanical Turk. For 300 generated sequences from each



minimize

_a_ 0 _..T −_ 1 _,ϵ_ 0 _..T −_ 1



_T_


ˆ

� _||z_ _t_ _−z_ _t_ _||_ + _λ_ 1 _||a_ _t_ _−a_ _t−_ 1 _||_ + _λ_ 2 _||ϵ_ _t_ _||_ (9)


_t_ =1



where _z_ _t_ is the output of our model, ˆ _z_ _t_ is the encoding of _x_ _t_
with the encoder, and _λ_ 1 _, λ_ 2 are hyperparameters for regularizers. We add action regularization assuming the action
space is continuous and _a_ _t_ does not differ significantly from
_a_ _t−_ 1 . To prevent the model from utilizing _ϵ_ _t_ to explain all
differences between frames, we also add the _ϵ_ regularizer.


**4. Experiments**


We perform thorough quantitative (Sec 4.1) and qualitative (Sec 4.2) experiments on the following datasets.
**Carla** [10] simulator is an open-source simulator for autonomous driving research. We use five towns in Carla to



samples from three
datasets studied in this

and real-world driving,



Gibson



RWD



5


_t_ = 1 _t_ = 3 _t_ = 7 _t_ = 13 _t_ = 21 _t_ = 53


World Model*


Stochastic

Adversarial


Video

Prediction


GameGAN


**DriveGAN**


Given Actions


a high-quality temporally consistent simulation that conforms to the action sequence.

|Model|Frechet Video Distance ↓<br>Carla Gibson RWD|
|---|---|
|Action-RNN<br>World Model<br>World Model*<br>SAVP<br>GameGAN<br>Ours|1523.3<br>1109.2<br>2560.7<br>1663.0<br>1212.0<br>2795.6<br>1138.6<br>561.1<br>591.7<br>1018.2<br>470.7<br>977.9<br>739.5<br>**311.4**<br>801.0<br>**281.9**<br>360.0<br>**518.0**|



Table 1: Results on FVD [61]. Lower is better.



Figure 9: **Left:** original images, **Middle:** reconstructed images from
VAE, **Right:** reconstructed images from our encoder-decoder model.


dataset, we show one video from our model and one video

from a baseline model for the same test data. The workers

are asked to mark their preferences on ours versus the baseline model on visual qulity and action consistency (Fig 10).
**Video Quality:** Tab 1 shows the result on Fr´echet Video
Distance (FVD) [61]. FVD measures the distance between
the distributions of the ground truth and generated video sequences. FVD is an extension of FID [18] for videos and is
suitable for measuring the quality of generated videos. Our
model achieves lower FVD than all baseline models except
for GameGAN on Gibson. The primary reason we suspect
is that our model on Gibson sometimes slightly changes the
brightness. In contrast, GameGAN, being a model directly
learned on pixel space, produced more consistent brightness. Human evaluation of visual quality (Fig 10) shows
that subjects strongly prefer our model, even for Gibson.
**Action Consistency:** We measure if generated sequences conform to the input action sequences. We train a
CNN model that takes two images from real videos as input
and predicts the action that caused the transition between
them. The model is trained by reducing the mean-squared
error loss between the predicted and input actions. The
trained model can be applied to the generated sequences
from simulator models to evaluate action consistency. Ta


Prefer Other Model Prefer Our Model


Figure 10: **Human evaluation:** Our model outperforms baseline


**4.2. Controllability and Differentiable Simulation**


DriveGAN learns to disentangle factors comprising a
scene without supervision, and it naturally allows controllability on all _z_ s as _z_ _[a]_ [dep], _z_ _[a]_ [indep], _z_ [content] and _z_ [theme] can be sampled from the prior distribution. Fig 4 demonstrates how
we can change the background color or weather condition
by sampling and swapping _z_ [theme] . Fig 12 shows how sampling different _z_ _[a]_ [indep] modifies the interior parts, such as ob


Human Evaluation – Visual Quality



100%
90%
80%
70%
60%
50%
40%
30%
20%
10%
0%


100%
90%
80%
70%
60%
50%
40%
30%
20%
10%
0%



SAVP 

Carla


SAVP 

Carla



SAVP 
Gibson


SAVP 
Gibson



SAVP 
Pilotnet



World Model*


  - Carla



World Model*


  - Gibson



World Model* GameGAN 

  - Pilotnet Carla



GameGAN 

Pilotnet


GameGAN 

Pilotnet



Prefer Other Model Prefer Our Model


Human Evaluation – Action Consistency



SAVP 
Pilotnet



World Model*


  - Carla



World Model*


  - Gibson



World Model* GameGAN 

  - Pilotnet Carla



GameGAN 

Gibson


GameGAN 

Gibson



6


|Model|Action Prediction Loss ↓<br>Carla Gibson RWD|
|---|---|
|Action-RNN<br>World Model<br>World Model*<br>SAVP<br>GameGAN<br>Ours|4.850<br>0.062<br>0.586<br>5.310<br>0.167<br>0.721<br>17.384<br>0.082<br>0.885<br>3.178<br>0.070<br>0.645<br>2.341<br>0.065<br>0.638<br>**1.686**<br>**0.045**<br>**0.412**|
|Real Data|0.370<br>0.005<br>0.159|


Table 2: Results on Action Prediction. Lower is better.


Randomly Generated + Random + Random + Random


Figure 11: Users can randomly sample a vector for a grid cell in
_z_ _[theme]_ to change the cell’s content. The white figner corresponds
to the locations a user clicked to modify.


Randomly Generated + Random + Random + Random


Figure 12: Swapping _z_ _[a]_ [indep] modifies objects in a scene while
keeping layout, such as the shape of the road, consistent. **Top:**
right turn, **Middle:** road for slight left, **Bottom:** straight road.


ject shapes, while keeping the layout and theme consistent.
This allows users to sample various scenarios for specific
layout shapes. As _z_ [content] is a spatial tensor, we can sample
each grid cell to change the content of the cell. In the bottom row of Fig 11, a user clicks specific locations to erase a
tree, add a tree, and add a building.
We also record the sampled _z_ s corresponding to specific
content and build an editable neural simulator, as in Fig 1.
This editing procedure lets users create unique simulation
scenarios and selectively focus on the ones they want. Note
that we can even sample the first screen, unlike some previous works such as GameGAN [30].

**Differentiable Simulation:** Sec 3.3 introduces how we

can create an editable simulation environment from a real

video by recovering the underlying actions _a_ and stochastic









Figure 13: We optimize action ( _a_ _[A]_ 0 _..T −_ 1 _[, a]_ _[B]_ 0 _..T −_ 1 [) and stochastic]
variable sequences ( _ϵ_ _[A]_ 0 _..T −_ 1 _[, ϵ]_ _[B]_ 0 _..T −_ 1 [) for real videos A and B. Let]
_z_ 0 _[A]_ [be the latent code of A’s initial frame. We show re-played]
sequences using ( _z_ 0 _[A]_ _[, a]_ _[A]_ _[, ϵ]_ _[A]_ [), (] _[z]_ 0 _[A]_ _[, a]_ _[B]_ _[, ϵ]_ _[A]_ [) and (] _[z]_ 0 _[A]_ _[, a]_ _[A]_ _[, ϵ]_ _[B]_ [).]


variables _ϵ_ with Eq.(9). Fig 13 illustrates the result of differentiable simulation. The third row exhibits how we can

recover the original video A by running DriveGAN with
optimized _a_ and _ϵ_ . To verify we have recovered _a_ successfully and not just overfitted using _ϵ_, we evaluate the quality
of optimized _a_ from test data using the Action Prediction
loss from Tab 2. Optimized _a_ results in a loss of 1.91 and
0.57 for Carla and RWD, respectively. These numbers are
comparable to Tab 2 and much lower than the baseline performances of 3.64 and 1.01, calculated with the mean of actions from the training data, demonstrating that DriveGAN
can recover unobserved actions successfully. We can even
recover _a_ and _ϵ_ for non-existing intermediate frames. That
is, we can do _frame interpolation_ to discover in-between
frames given a reference and a future frame. If the time between the two frames is small, even a naive linear interpolation could work. However, for a large gap ( _≥_ 1 second), it
is necessary to reason about the environment’s dynamics to
properly interpolate objects in a scene. We modify Eq.(9) to
minimize the reconstruction term for the last frame _z_ _T_ only,
and add a regularization _||z_ _t_ _−_ _z_ _t−_ 1 _||_ on the intermediate
_z_ s. Fig 14 shows the result. Top row, which shows interpolation in the latent space, produces reasonable in-between
frames, but if inspected closely, we can see the transition
is unnatural ( _e.g_ . a tree appears out of nowhere). On the
contrary, with differentiable simulation, we can see how
it learns to utilize the dynamics of DriveGAN to produce
plausible transitions between frames. In Fig 15, we calculate the action prediction loss with optimized actions from
frame interpolation. We discover optimized actions that follow the ground-truth actions closely when we interpolate
frames one second apart. As the interpolation interval becomes larger, the loss increases since many possible action



7


Top: linear interpolation with latent codes **Bottom: interpolation with differentiable simulation**


Figure 14: **Frame Interpolation** We run differentiable simulation to produce Frame 2 given Frame 1. **Top:** Linear interpolation in latent
space does not account for transition dynamics correctly. **Bottom:** DriveGAN keeps dynamics consistent with respect to the environment.



**Interpolation with Differentiable Simulation - Action Prediction Loss**









4.2


3.7


3.2


2.7


2.2


1.7


1.2







Figure 15: Optimized actions from frame interpolation discovers
in-between actions. Mean action measures action prediction loss
when the mean of actions from the training dataset is used as input.


sequences lead to the same resulting frame. This shows the
possibility of using differentiable simulation for video compression as it can decode missing intermediate frames.
Differentiable simulation also allows replaying the same
scenario with different inputs. In Fig 13, we get optimized
_a_ _[A]_ _, ϵ_ _[A]_ and _a_ _[B]_ _, ϵ_ _[B]_ for two driving videos, A and B. We replay starting with the encoded first frame _z_ 0 _[A]_ [of A. On the]
fourth row, ran with ( _a_ _[B]_, _ϵ_ _[A]_ ), we see that a vehicle is placed
at the same location as A, but since we use the slightlyleft action sequence _a_ _[B]_, the ego agent changes the lane
and slides toward the vehicle. The fifth row, replayed with
( _a_ _[A]_, _ϵ_ _[B]_ ), shows the same ego-agent’s trajectory as A, but it
puts a vehicle at the same location as B due to _ϵ_ _[B]_ . This effectively shows that we can _blend-in_ two different scenarios
together. Furthermore, we can modify the content and run
a simulation with the environment inferred from a video. In

Fig 6, we create a simulation environment from a RWD test
data, and replay with modified objects and weather.


**4.3. Additional Experiments**


**LiftSplat** [48] proposed a model for producing the
Bird’s-Eye-View (BEV) representation of a scene from
camera images. We use LiftSplat to get BEV lane predictions from a simulated sequence from DriveGAN (Fig 16).
Simulated scenes are realistic enough for LiftSplat to produce accurate predictions. This shows the potential of
DriveGAN being used with other perception models to be
useful for downstream tasks such as training an autonomous



Simulated 𝑡= 1 BEV 𝑡= 5 BEV


Figure 16: Bird’s-Eve-View (BEV) lane prediction with LiftSplat [48] model on generated scenes.


driving agent. Furthermore, in real-time driving, LiftSplat
can potentially employ DriveGAN’s simulated frames as a
safety measure to be robust to sudden camera drop-outs.
**Plain StyleGAN latent space:** StyleGAN [29] proposes
an optimization scheme to project images into their latent
codes without an encoder. The projection process optimizes each image and requires significant time ( _∼_ 19200
GPU hours for Gibson). Therefore, we use 25% of Gibson
data to compare with the projection approach. We train the
same dynamics model on top of the projected and proposed
latent spaces. The projection approach resulted in FVD of
**636.8** with the action prediction loss of **0.225**, whereas ours
achieved **411.9** (FVD) and **0.050** (action prediction loss).


**5. Conclusion**


We proposed DriveGAN for a controllable high-quality
simulation. DriveGAN leverages a novel encoder and
an image GAN to produce a latent space on which the
proposed dynamics engine learns the transitions between
frames. DriveGAN allows sampling and disentangling of
different components of a scene without supervision. This
lets users interactively edit scenes during a simulation and
produce unique scenarios. We showcased _differentiable_
_simulation_ which opens up promising ways for utilizing
real-world videos to discover the underlying factors of variations and train robots in the re-created environments.



8


**References**


[1] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan,
Roy H. Campbell, and Sergey Levine. Stochastic variational
video prediction. _CoRR_, abs/1710.11252, 2017. 2

[2] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael
Bowling. The arcade learning environment: An evaluation
platform for general agents. _Journal of Artificial Intelligence_
_Research_, 47:253–279, 2013. 2

[3] Silvia Chiappa, S´ebastien Racaniere, Daan Wierstra, and
Shakir Mohamed. Recurrent environment simulators. _arXiv_

_preprint arXiv:1704.02254_, 2017. 5

[4] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and
Yoshua Bengio. Empirical evaluation of gated recurrent
neural networks on sequence modeling. _arXiv preprint_
_arXiv:1412.3555_, 2014. 14

[5] Aidan Clark, Jeff Donahue, and Karen Simonyan. Efficient video generation on complex datasets. _CoRR_,
abs/1907.06571, 2019. 2

[6] Marc Deisenroth and Carl E Rasmussen. Pilco: A modelbased and data-efficient approach to policy search. In _Pro-_
_ceedings of the 28th International Conference on machine_
_learning (ICML-11)_, pages 465–472, 2011. 2

[7] Emily Denton and Rob Fergus. Stochastic video generation
with a learned prior. _CoRR_, abs/1802.07687, 2018. 2

[8] Emily L Denton et al. Unsupervised learning of disentangled
representations from video. In _Advances in neural informa-_
_tion processing systems_, pages 4414–4423, 2017. 4

[9] Jeevan Devaranjan, Amlan Kar, and Sanja Fidler. Metasim2: Unsupervised learning of scene structure for synthetic
data generation. _arXiv preprint arXiv:2008.09092_, 2020. 1,
2

[10] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio
Lopez, and Vladlen Koltun. CARLA: An open urban driving
simulator. In _Proceedings of the 1st Annual Conference on_
_Robot Learning_, pages 1–16, 2017. 1, 5

[11] Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. _arXiv preprint_
_arXiv:1610.07629_, 2016. 3

[12] Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In _Advances in neural information processing systems_,
pages 64–72, 2016. 2

[13] Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent
Dumoulin, and Jonathon Shlens. Exploring the structure of a
real-time, arbitrary neural artistic stylization network. _arXiv_
_preprint arXiv:1705.06830_, 2017. 3

[14] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial networks, 2014. 2,
13

[15] David Ha and J¨urgen Schmidhuber. Recurrent world models
facilitate policy evolution. In _Advances in Neural Informa-_
_tion Processing Systems_, pages 2450–2462, 2018. 1, 2, 5

[16] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning
latent dynamics for planning from pixels. In _International_



_Conference on Machine Learning_, pages 2555–2565. PMLR,
2019. 2

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In _Proceed-_
_ings of the IEEE conference on computer vision and pattern_
_recognition_, pages 770–778, 2016. 12

[18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a
two time-scale update rule converge to a local nash equilibrium. In _Advances in neural information processing systems_,
pages 6626–6637, 2017. 6

[19] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess,
Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and
Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016. 3

[20] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term
memory. _Neural computation_, 9(8):1735–1780, 1997. 2, 4,
14

[21] Jun-Ting Hsieh, Bingbin Liu, De-An Huang, Li Fei-Fei, and
Juan Carlos Niebles. Learning to decompose and disentangle
representations for video prediction. _CoRR_, abs/1806.04166,
2018. 2

[22] Xun Huang and Serge Belongie. Arbitrary style transfer in
real-time with adaptive instance normalization. In _Proceed-_
_ings of the IEEE International Conference on Computer Vi-_
_sion_, pages 1501–1510, 2017. 3

[23] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal covariate shift. _arXiv preprint arXiv:1502.03167_, 2015. 14

[24] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A
Efros. Image-to-image translation with conditional adversarial networks. In _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pages 1125–1134,
2017. 3, 13, 15

[25] Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos,
Blazej Osinski, Roy H Campbell, Konrad Czechowski,
Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey
Levine, et al. Model-based reinforcement learning for atari.
_arXiv preprint arXiv:1903.00374_, 2019. 2

[26] Nal Kalchbrenner, A¨aron van den Oord, Karen Simonyan,
Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks. _CoRR_,
abs/1610.00527, 2016. 2

[27] Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci,
Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba,
and Sanja Fidler. Meta-sim: Learning to generate synthetic
datasets. In _Proceedings of the IEEE International Confer-_
_ence on Computer Vision_, pages 4551–4560, 2019. 1, 2

[28] Tero Karras, Samuli Laine, and Timo Aila. A style-based
generator architecture for generative adversarial networks. In
_Proceedings of the IEEE conference on computer vision and_
_pattern recognition_, pages 4401–4410, 2019. 2, 3

[29] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten,
Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pages 8110–8119, 2020. 3, 8, 12, 13



9


[30] Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with gamegan. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_,
pages 1231–1240, 2020. 1, 2, 4, 5, 7

[31] Yoon Kim. Convolutional neural networks for sentence classification. _arXiv preprint arXiv:1408.5882_, 2014. 4

[32] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014. 13

[33] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2014. 2, 3, 12, 13

[34] Eric Kolve, Roozbeh Mottaghi, Daniel Gordon, Yuke Zhu,
Abhinav Gupta, and Ali Farhadi. Ai2-thor: An interactive
3d environment for visual ai. In _arXiv:1712.05474_, 2017. 1

[35] Manoj Kumar, Mohammad Babaeizadeh, Dumitru Erhan,
Chelsea Finn, Sergey Levine, Laurent Dinh, and Durk
Kingma. Videoflow: A flow-based generative model for
video. _CoRR_, abs/1903.01434, 2019. 2

[36] Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, and
Ole Winther. Autoencoding beyond pixels using a learned
similarity metric. _CoRR_, abs/1512.09300, 2015. 2, 3

[37] Alex X Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel,
Chelsea Finn, and Sergey Levine. Stochastic adversarial
video prediction. _arXiv preprint arXiv:1804.01523_, 2018.
2, 5

[38] Jae Hyun Lim and Jong Chul Ye. Geometric gan. _arXiv_
_preprint arXiv:1705.02894_, 2017. 15

[39] William Lotter, Gabriel Kreiman, and David Cox. Deep predictive coding networks for video prediction and unsupervised learning. _arXiv preprint arXiv:1605.08104_, 2016. 2

[40] Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models.
In _Proc. icml_, volume 30, page 3, 2013. 12

[41] Arun Mallya, Ting-Chun Wang, Karan Sapra, and Ming-Yu
Liu. World-consistent video-to-video synthesis, 2020. 2

[42] Sivabalan Manivasagam, Shenlong Wang, Kelvin Wong,
Wenyuan Zeng, Mikita Sazanovich, Shuhan Tan, Bin Yang,
Wei-Chiu Ma, and Raquel Urtasun. Lidarsim: Realistic lidar
simulation by leveraging the real world. In _Proceedings of_
_the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pages 11167–11176, 2020. 1, 2

[43] Michael Mathieu, Camille Couprie, and Yann LeCun. Deep
multi-scale video prediction beyond mean square error,
2016. 2

[44] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin.
Which training methods for gans do actually converge?
_arXiv preprint arXiv:1801.04406_, 2018. 15

[45] Matthias Minderer, Chen Sun, Ruben Villegas, Forrester
Cole, Kevin P Murphy, and Honglak Lee. Unsupervised
learning of object structure and dynamics from videos. In
_Advances in Neural Information Processing Systems_, pages
92–102, 2019. 2

[46] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and
Yuichi Yoshida. Spectral normalization for generative adversarial networks. _arXiv preprint arXiv:1802.05957_, 2018.

14




[47] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L Lewis,
and Satinder Singh. Action-conditional video prediction using deep networks in atari games. In _Advances in neural_
_information processing systems_, pages 2863–2871, 2015. 2

[48] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding
images from arbitrary camera rigs by implicitly unprojecting
to 3d. _arXiv preprint arXiv:2008.05711_, 2020. 8, 15

[49] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu
Wang, Sanja Fidler, and Antonio Torralba. Virtualhome:
Simulating household activities via programs. In _CVPR_,
2018. 1

[50] Marc’Aurelio Ranzato, Arthur Szlam, Joan Bruna, Micha¨el
Mathieu, Ronan Collobert, and Sumit Chopra. Video (language) modeling: a baseline for generative models of natural
videos. _CoRR_, abs/1412.6604, 2014. 2

[51] Nataniel Ruiz, Samuel Schulter, and Manmohan Chandraker.
Learning to simulate. _arXiv preprint arXiv:1810.02513_,
2018. 1

[52] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping,
2017. 2

[53] M. Saito and Shunta Saito. Tganv2: Efficient training of
large models for video generation with multiple subsampling
layers. _ArXiv_, abs/1811.09245, 2018. 2

[54] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish
Kapoor. Aerial Informatics and Robotics platform. Technical Report MSR-TR-2017-9, Microsoft Research, 2017. 1

[55] Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. Singan: Learning a generative model from a single natural image. In _Proceedings of the IEEE International Conference_
_on Computer Vision_, pages 4570–4580, 2019. 3, 13, 15

[56] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhutdinov. Unsupervised learning of video representations using
lstms. _CoRR_, abs/1502.04681, 2015. 2

[57] Richard S Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In _Machine learning proceedings 1990_, pages
216–224. Elsevier, 1990. 2

[58] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A
physics engine for model-based control. In _2012 IEEE/RSJ_
_International Conference on Intelligent Robots and Systems_,
pages 5026–5033. IEEE, 2012. 2

[59] Dustin Tran, Rajesh Ranganath, and David Blei. Hierarchical implicit models and likelihood-free variational inference. In _Advances in Neural Information Processing Sys-_
_tems_, pages 5523–5533, 2017. 15

[60] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan
Kautz. Mocogan: Decomposing motion and content for
video generation. _CoRR_, abs/1707.04993, 2017. 2

[61] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach,
Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric &
challenges. _arXiv preprint arXiv:1812.01717_, 2018. 6

[62] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba.
Generating videos with scene dynamics, 2016. 2

[63] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu,
Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-tovideo synthesis. _CoRR_, abs/1808.06601, 2018. 2, 3, 13, 15



10


[64] Dirk Weissenborn, Oscar T¨ackstr¨om, and Jakob Uszkoreit. Scaling autoregressive video models. _CoRR_,
abs/1906.02634, 2019. 2

[65] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson env: real-world perception for embodied agents. In _Computer Vision and Pat-_
_tern Recognition (CVPR), 2018 IEEE Conference on_ . IEEE,
2018. 5

[66] Wei Yu, Yichao Lu, Steve Easterbrook, and Sanja Fidler.
Efficient and information-preserving future frame prediction
and beyond. In _ICLR_, 2020. 2

[67] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In _Proceedings of the_
_IEEE conference on computer vision and pattern recogni-_
_tion_, pages 586–595, 2018. 3, 13



11


**Supplementary Materials for**
**DriveGAN: Towards a Controllable High-Quality Neural Simulation**


**A. Model Architecture and Training**


We provide detailed descriptions of the model architecture and training process for the pre-trained image encoder-decoder
(Sec. A.1) and dynamics engine (Sec. A.2). Unless noted otherwise, we denote tensor dimensions by _H × W × D_ where _H_
and _W_ are the spatial height and width of a feature map, and _D_ is the number of channels.


**A.1. Pre-trained Latent Space**


The latent space is pretrained with an encoder, generator and discrminator. Figure 17 shows the overview of the pretraining
model.



_z_ [theme]





Figure 17: The pretraining stage learns the encoder and decoder for images. The encoder _ξ_ produces _z_ [content] and _z_ [theme] which comprise
the disentangled latent space that the dynamics engine trains on. The gaussian blocks represent reparameterization steps [33].


**A.1.1** **Encoder**


Encoder _ξ_ takes an RGB image _x ∈_ R [256] _[×]_ [256] _[×]_ [3] as input and produces disentangled latent codes _z_ = _{z_ [theme] _, z_ [content] _}_ where
_z_ [theme] _∈_ R [128] and _z_ [content] _∈_ R [4] _[×]_ [4] _[×]_ [64] . _ξ_ is composed of a feature extractor _ξ_ [feat] and two encoding heads _ξ_ [content] and _ξ_ [theme] .



**Layer** **Output dimension**
Conv2d 3 _×_ 3 256 _×_ 256 _×_ 128

ResBlock 128 _×_ 128 _×_ 256

ResBlock 64 _×_ 64 _×_ 512

ResBlock 32 _×_ 32 _×_ 512


Table 3: _ξ_ [feat] architecture



**Layer** **Output dimension**
ResBlock 16 _×_ 16 _×_ 512

ResBlock 8 _×_ 8 _×_ 512

ResBlock 4 _×_ 4 _×_ 512

Conv2d 3 _×_ 3 4 _×_ 4 _×_ 512

Conv2d 3 _×_ 3 4 _×_ 4 _×_ 128


Table 4: _ξ_ [content] architecture



**Layer** **Output dimension**
Conv2d 3 _×_ 3 32 _×_ 32 _×_ 512

AvgPool2d 32 _×_ 32 512
Linear 256


Table 5: _ξ_ [theme] architecture



The above tables show the architecture for each component. _ξ_ [feat] takes _x_ as input and consists of several convolution
layers whose output is passed to the two heads. Conv2d 3 _×_ 3 denotes a 2D convolution layer with 3 _×_ 3 filters and padding
of 1 to produce the same spatial dimension as input. ResBlock denotes a residual block [17] with downsampling by 2 _×_
which is composed of two 3 _×_ 3 convolution layers and a skip connection layer. After each layer, we put the leaky ReLU [40]
activation function, except for the last layer of _ξ_ [content] and _ξ_ [theme] . The outputs of _ξ_ [content] and _ξ_ [theme] are equally split into two
chunks by the channel dimension, and used as _µ_ and _σ_ for the reparameterization steps:


_z_ = _µ_ + _ϵσ,_ _ϵ ∼_ _N_ (0 _, I_ ) (10)


producing _z_ [theme] _∈_ R [128] and _z_ [content] _∈_ R [4] _[×]_ [4] _[×]_ [64] .


**A.1.2** **Generator**


The generator architecture closely follows the generator of StyleGAN [29]. Here, we discuss a few differences. _z_ [content] goes
through a 3 _×_ 3 convolution layer to make it a 4 _×_ 4 _×_ 512 tensor. StyleGAN takes a constant tensor as an input to the first layer.
We concatenate the constant tensor with _z_ [content] channel-wise and pass it to the first layer. _z_ [theme] goes through 8 linear layers,
each outputting a 1024-dimensional vector, and the output is used for the adaptive instance normalization layers in the same
way _style_ vectors are used in StyleGAN. The generator outputs a 256 _×_ 256 _×_ 3 image.


12


**A.1.3** **Discriminator**


Dicriminator takes the real and generated images (256 _×_ 256 _×_ 3) as input. We use multi-scale multi-patch discriminators [63,
24, 55], which results in higher quality images for complex scenes.



**Layer** **Output dimension**
Conv2d 3 _×_ 3 256 _×_ 256 _×_ 128

ResBlock 128 _×_ 128 _×_ 256

ResBlock 64 _×_ 64 _×_ 512

ResBlock 32 _×_ 32 _×_ 512

ResBlock 16 _×_ 16 _×_ 156

ResBlock 8 _×_ 8 _×_ 512

ResBlock 4 _×_ 4 _×_ 512

Conv2d 3 _×_ 3 4 _×_ 4 _×_ 512

Linear 512

Linear 1


Table 6: _D_ 1 architecture



**Layer** **Output dimension**
Conv2d 3 _×_ 3 256 _×_ 256 _×_ 128

ResBlock 128 _×_ 128 _×_ 256

ResBlock 64 _×_ 64 _×_ 512

ResBlock 32 _×_ 32 _×_ 512

ResBlock 16 _×_ 16 _×_ 512

Conv2d 3 _×_ 3 16 _×_ 16 _×_ 1


Table 7: _D_ 2 architecture



**Layer** **Output dimension**

Conv2d 3 _×_ 3 128 _×_ 128 _×_ 128

ResBlock 64 _×_ 64 _×_ 256

ResBlock 32 _×_ 32 _×_ 512

ResBlock 16 _×_ 16 _×_ 512

ResBlock 8 _×_ 8 _×_ 512

Conv2d 3 _×_ 3 8 _×_ 8 _×_ 1


Table 8: _D_ 3 architecture



We use three discrminators _D_ 1 _, D_ 2 _,_ and _D_ 3 . _D_ 1 takes a 256 _×_ 256 _×_ 3 image as input and produces a single number. _D_ 2
takes a 256 _×_ 256 _×_ 3 image as input and produces 16 _×_ 16 patches each with a single number. _D_ 3 takes a 128 _×_ 128 _×_ 3 image
as input and produces 8 _×_ 8 patches each with a single number. The adversarial losses for _D_ 2 and _D_ 3 are averaged across the
patches. The inputs to _D_ 1 _, D_ 2 _, D_ 3 are the real and generated images, except that the input to _D_ 3 is downsampled by 2 _×_ .
The model architectures are described in the above tables, and we use the same convolution layer and residual blocks from
the previous sections. Each layer is followed by a leaky ReLU activation function except for the last layer.


**A.1.4** **Training**


We combine the loss functions of VAE [33] and GAN [14], and let _L_ _pretrain_ = _L_ _V AE_ + _L_ _GAN_ . We use the same loss
function for the adversarial loss _L_ _GAN_ from StyleGAN [29], except that we have three terms for each discriminator. _L_ _V AE_
is defined as:
_L_ _V AE_ = _E_ _z∼q_ ( _z|x_ ) [ _log_ ( _p_ ( _x|z_ ))] + _βKL_ ( _q_ ( _z|x_ ) _||p_ ( _z_ ))


where _p_ ( _z_ ) is the standard normal prior distribution, _q_ ( _z|x_ ) is the approximate posterior from the encoder _ξ_, and _KL_ is
the Kullback-Leibler divergence. For the reconstruction term, we reduce the perceptual distance [67] between the input and
output images rather than the pixel-wise distance, and this term is weighted by 25.0. We use separate _β_ values _β_ [theme] and
_β_ [content] for _z_ [content] and _z_ [theme] . We also found different _β_ values work better for different environments. We use _β_ [theme] =
1 _._ 0 _, β_ [content] = 2 _._ 0 for Carla, _β_ [theme] = 1 _._ 0 _, β_ [content] = 4 _._ 0 for Gibson, and _β_ [theme] = 1 _._ 0 _, β_ [content] = 1 _._ 0 for RWD. Adam [32]
optimizer is employed with learning rate of 0.002 for 310,000 optimization steps. We use a batch size of 16.


**A.2. Dynamics Engine**


With the pre-trained encoder and decoder, the Dynamics Engine learns the transition between latent codes from one time
step to the next given an action _a_ _t_ . We first pre-extract the latent codes for each image in the training data, and only learn
the transition between the latent codes. All neural network layers described below are followed by a leaky ReLU activation
function, except for the outputs of discriminators, the outputs for _µ, σ_ variables used for reparameterization steps, and the
outputs for the AdaIN parameters.
The major components of the Dynamics Engine are its two LSTM modules. The first one learns the spatial transition
between the latent codes and is implemented as a convolutional LSTM module (Figure 18).


_v_ _t_ = _F_ ( _H_ ( _h_ [conv] _t−_ 1 _[, a]_ _[t]_ _[, z]_ _t_ [content] _, z_ _t_ [theme] )) (11)


_i_ _t_ _, f_ _t_ _, o_ _t_ = _σ_ ( _v_ _t_ _[i]_ [)] _[, σ]_ [(] _[v]_ _t_ _[f]_ [)] _[, σ]_ [(] _[v]_ _t_ _[o]_ [)] (12)


_c_ [conv] _t_ = _f_ _t_ _⊙_ _c_ [conv] _t−_ 1 [+] _[ i]_ _[t]_ _[⊙]_ [tanh(] _[v]_ _t_ _[g]_ [)] (13)


13


Figure 18: Dynamics Engine produces the next latent codes, given an action and previous latent codes. It disentangles content information
into action-dependent and action-independent features with its two separate LSTMs. Dashed lines correspond to temporal connections.
Gaussian blocks indicate reparameterization steps.


_h_ [conv] _t_ = _o_ _t_ _⊙_ tanh( _c_ [conv] _t_ ) (14)


where _h_ [conv] _t_ _, c_ [conv] _t_ are the hidden and cell state of the convLSTM module, and _i_ _t_ _, f_ _t_ _, o_ _t_ are the input, forget, output gates,
respectively. _H_ replicates _a_ _t_ and _z_ _t_ [theme] spatially to match the 4 _×_ 4 spatial dimension of _z_ _t_ [content] . It fuses all inputs by
concatenating and running through a 1 _×_ 1 convolution layer, resulting in a 4 _×_ 4 _×_ 48 tensor. _F_ is composed of two 3 _×_ 3
convolution layers with a padding of 1, and produces _v_ _t_ _∈_ R [4] _[×]_ [4] _[×]_ [512] . _v_ _t_ is split channel-wise into intermediate variables
_v_ _t_ _[i]_ _[, v]_ _t_ _[f]_ _[, v]_ _t_ _[o]_ _[, v]_ _t_ _[g]_ [. All state and intermediate variables have the same size][ R] [4] _[×]_ [4] _[×]_ [128] [. The hidden state] _[ h]_ [conv] _t_ goes through two
separate convolution layers: 1) 1 _×_ 1 Conv2d layer that produces 4 _×_ 4 _×_ 128 tensor which is split into two chuncks with equal
size 4 _×_ 4 _×_ 64 and used for the reparameterization step (Eq. 10) to produce _z_ _t_ _[a]_ +1 [dep] _[∈]_ [R] [4] _[×]_ [4] _[×]_ [64] [, and 2) 4] _[×]_ [4 conv2d layer]
with no padding that produces a 256 dimensional vector; this is also split into two chunks and reparameterized to produce
_z_ _t_ [theme] +1 _[∈]_ [R] [128] [.]
The second one is a plain LSTM [20] module that only takes _z_ _t_ as input. Therefore, this module is responsible for
information that does not depend on the action _a_ _t_ . The input _z_ _t_ is flattened into a vector _∈_ R [1152] and goes through five
linear layers each outputting 1024-dimensional vectors. The encoded _z_ _t_ is fed to the LSTM module and all variables inside
this module have size R [1024] . We experimented with both LSTM and GRU [4] but did not observe much difference. The
hidden state goes through a linear layer that outputs a 2048-dimensional vector. This vector is split into two chunks for
reparmetrization and produces _z_ _t_ _[a]_ +1 [indep] _[∈]_ [R] [1024] [.]
Finally, _z_ _t_ _[a]_ +1 [dep] [and] _[ z]_ _t_ _[a]_ +1 [indep] [are used as inputs to two] _[ AdaIN]_ [ + Conv blocks.]


_**α**_ _,_ _**β**_ = _MLP_ ( _z_ _t_ _[a]_ +1 [indep] [)] (15)


_z_ _t_ [content] +1 = _C_ ( _A_ ( _C_ ( _A_ ( _z_ _t_ _[a]_ +1 [dep] _[,]_ _**[ α]**_ _[,]_ _**[ β]**_ [))] _[,]_ _**[ α]**_ _[,]_ _**[ β]**_ [))] (16)


where we denote convolution and _AdaIN_ layers as _C_ and _A_, respectively. The two _MLP_ s (for each block) consist of two
linear layers. They produce 64 and 256 dimensional _**α**_ _,_ _**β**_, respectively. The first 3 _×_ 3 conv2d layer _C_ produces 4 _×_ 4 _×_ 256
tensor, and the second 3 _×_ 3 conv2d layer produces _z_ _t_ [content] +1 _∈_ R [4] _[×]_ [4] _[×]_ [64] .


**A.2.1** **Discriminator**


We use disciminators on the flattened 1152 dimensional latent codes _z_ (concatenation of _z_ [theme] and flattened _z_ [content] ). There
are two discriminators 1) single latent discriminator _D_ _single_, and 2) temporal action-conditioned discriminator _D_ _temporal_ .
We denote SNLinear and SNConv as linear and convolution layers with Spectral Normalization [46] applied, and BN as 1D
Batch Normalization layers [23]. _D_ _single_ is a 6-layer _MLP_ that tries to discriminate generated _z_ from the real latent codes.
It takes a single _z_ as input and produces a single number. For the temporal action-conditioned discriminator _D_ _temporal_, we
first reuse the 1024-dimensional feature representation from the fourth layer of _D_ _single_ for each _z_ _t_ . The represenations for _z_ _t_
and _z_ _t−_ 1 are concatenated and go through a SNLinear layer to produce the 1024-dimensional temporal discriminator feature.
Let us denote the temporal discriminator feature as _z_ _t,t−_ 1 . The action _a_ _t_ also goes through a SNLinear layer to produce the


14


**Layer** **Output dimension**

SNLinear + BN 1024

SNLinear + BN 1024

SNLinear + BN 1024

SNLinear + BN 1024

SNLinear + BN 1024

SNLinear 1


Table 9: _D_ _single_ architecture



**Layer** **Input dimension** **Output dimension**
SNConv1d 2048 _×_ 31 128 _×_ 15

SNConv1d 128 _×_ 15 256 _×_ 13

SNConv1d 256 _×_ 13 512 _×_ 6


Table 10: _D_ _temporal_ architecture. Input and output dimensions contain two
numbers, the first one for the number of channels or vector dimension, and
the second one for the temporal dimension. Note that Conv1d is applied on
the temporal dimension.



1024-dimensional action embedding. _z_ _t,t−_ 1 and the action embedding are concatenated and used as the input to _D_ _temporal_ .
We use 32 time-steps to train DriveGAN, so the input to _D_ _temporal_ has size 2048 _×_ 31 where 31 is the temporal dimension.
Table 8 shows the architecture of _D_ _temporal_ . After each layer of _D_ _temporal_, we put a 3-timestep wide convolution layer that
produces a single number for each resulting time dimension. Therefore, there are three outputs of _D_ _temporal_ with sizes 14,
11, and 4 which can be thought of as _patches_ in the temporal dimension. We also sample negative actions ¯ _a_ _t_, and the job of
_D_ _temporal_ is to figure out if the given sequence of latent codes is realistic and faithful to the given action sequences. ¯ _a_ _t_ is
sampled randomly from the training dataset.


**A.2.2** **Training**


We use Adam optimizer with learning rate of 0.0001 for 400,000 optimization steps. We use batch size of 128 each with
32 time-steps and train with a warm-up phase. In the warm-up phase, we feed in the ground-truth latent codes as input for
the first 18 time-steps and linearly decay the number to 1 at 100-th epoch, which corresponds to completely autoregressive
training at that point. We use the loss _L_ _DE_ = _L_ _adv_ + _L_ _latent_ + _L_ _action_ + _L_ _KL_ . _L_ _adv_ is the adversarial losses, and we use the
hinge loss [38, 59]. We also add a _R_ 1 gradient regularizer [44] to _L_ _adv_ that penalizes the gradients of discriminators on true
data . _L_ _action_ is the action reconstruction loss (implemented as a mean squared error loss) which we obtain by running the
temporal discriminator features _z_ _t,t−_ 1 through a linear layer to reconstruct the input action _a_ _t−_ 1 . Finally, we add the latent
code reconstruction loss _L_ _latent_ (implemented as a mean squared error loss) so that the generated _z_ _t_ matches the input latent
codes, and reduce the _KL_ penalty _L_ _KL_ for _z_ _t_ _[a]_ [dep], _z_ _t_ _[a]_ [indep], _z_ _t_ [theme] . _L_ _latent_ is weighted by 10.0 and we use different _β_ for the
_KL_ penalty terms. We use _β_ _[a]_ [dep] = 0 _._ 1 _, β_ _[a]_ [indep] = 0 _._ 1 _, β_ [theme] = 1 _._ 0 for Carla, and _β_ _[a]_ [dep] = 0 _._ 5 _, β_ _[a]_ [indep] = 0 _._ 25 _, β_ [theme] = 1 _._ 0 for
Gibson and RWD.


**B. Additional Analysis on Experiments**


**Multi-patch Multi-scale discriminator** We experimented with Carla dataset to choose the image discriminator architecture. In contrast to the plain StyleGAN, the datasets studied in this work contain much more diverse objects in multiple
locations. Using a multi-patch multi-scale discriminator [63, 24, 55] improved our FID score on Carla images from **72.3** to
**67.1** over the StyleGAN discriminator.
**LiftSplat** [48] proposed a model for producing the Bird’s-Eye-View (BEV) representation of a scene from camera images.
Section 4.3 in the main text shows how we can leverage LiftSplat to get BEV lane predictions from a simulated sequence
from DriveGAN. We can further analyze the qualitative result by comparing how the perception model (LiftSplat) perceives
the ground truth and generated sequences differently. We fit a quadratic function to the LiftSplat BEV lane prediction for
each image in the ground-truth sequence, and compare the distance between the fitted quadratic and the predicted lanes.


**BEV Prediction Look-ahead Distance**

**Model** 25m 50m 75m 100m

Random 0.91m 1.78m 2.95m 4.74m

DriveGAN 0.58m 1.00m 1.70m 2.99m

Ground-Truth 0.31m 0.37m 0.88m 2.07m


Table 11: Mean distance from the BEV lane predictions and the fitted quadratic function
in meters.


15


We show results on different look-ahead distances, which denote how far from the ego-car we are making the BEV predictions for. The above table lists the mean distance from the BEV lane predictions and the fitted quadratic function. _Random_
compares the distance between the fitted quadratic and the BEV prediction for a randomly sampled RWD sequence. _Drive-_
_GAN_ compares the distance for the BEV prediction for the optimized sequence with _differentiable simulation_ of DriveGAN.
_Ground-Truth_ compares the distance for the BEV prediction for the ground-truth image. Note that _Ground-Truth_ is not 0
since the fitted quadratic does not necessarily follow the lane prediction from the ground-truth image exactly. We can see that
DriveGAN-optimized sequences produce lanes that follow the ground-truth lanes, which demonstrates how we could find the
underlying actions and stochastic variables from a real video through differentiable simulation.


**C. DriveGAN Simulator User Interface**


Figure 19: UI for DriveGAN simulator


We build an interactive user interface for users to play with DriveGAN. Figure 19 shows the application screen. It has
controls for the steering wheel and speed, which can be controlled by the keyboard. We can randomize different components
by sampling _z_ _[a]_ [indep] _, z_ [content] or _z_ [theme] . We also provide a pre-defined list of themes and objects that users can selectively use
for specific changes. The supplementary video demonstrates how this UI can enable interactive simulation.


16


