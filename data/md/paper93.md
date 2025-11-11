1


## Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features

Simone Palazzo, Concetto Spampinato, _Member, IEEE,_ Isaak Kavasidis,
Daniela Giordano, _Member, IEEE,_ Joseph Schmidt, and Mubarak Shah, _Fellow, IEEE,_


**Abstract** —This work presents a novel method of exploring human brain-visual representations, with a view towards replicating these
processes in machines. The core idea is to learn plausible computational and biological representations by correlating human neural
activity and natural images. Thus, we first propose a model, _EEG-ChannelNet_, to learn a brain manifold for EEG classification. After
verifying that visual information can be extracted from EEG data, we introduce a multimodal approach that uses deep image and EEG
encoders, trained in a siamese configuration, for learning a joint manifold that maximizes a compatibility measure between visual
features and brain representations.
We then carry out image classification and saliency detection on the learned manifold. Performance analyses show that our approach
satisfactorily decodes visual information from neural signals. This, in turn, can be used to effectively supervise the training of deep
learning models, as demonstrated by the high performance of image classification and saliency detection on out-of-training classes.
The obtained results show that the learned brain-visual features lead to improved performance and simultaneously bring deep models
more in line with cognitive neuroscience work related to visual perception and attention.


!



**1** **I** **NTRODUCTION**

# H ficial systems, mainly thanks to the recent advances UMAN visual capabilities are nearly rivaled by arti
in deep learning. Indeed, deep feedforward and recurrent
neural networks, loosely inspired by the primate visual
system, have led to a significant boost in performance
of computer vision, natural language processing, speech
recognition and game playing. In addition to the significant
performance gain in such tasks, the representations learned
by deep computational models appear to be highly correlated with brain representations. For example, correlations
can be found between brain representations in the visual
pathway and the hierarchical structures of layers in deep
neural networks (DNNs) [1], [2]. These findings have paved
the way for multidisciplinary efforts involving cognitive
neuroscientists and artificial intelligence researchers, with
the aim of reverse-engineering the human mind and its
adaptive capabilities [3], [4], [5], [6]. Nevertheless, this multidisciplinary field is still in its infancy. Indeed, the existing
computational neural models loosely emulate computations
and connections of biological neurons, but they often ignore
feedforward and feedback neural interactions. For example,
visual recognition in humans appears to be mitigated by
a multi-level aggregation of information being processed
forward and backward across cortical brain regions [7], [8],

[9], [10]. Recent approaches [11], inspired by the hierarchical
predictive coding in neuroscience [12], [13], have attempted
to encode such additional information into computational


_•_ _S. Palazzo, C. Spampinato, I. Kavasidis and D. Giordano are with_
_the Department of Electrical, Electronic and Computer Engineering,_
_University of Catania, Viale Andrea Doria, 6, Catania, 95125, Italy._


_E-mail: palazzosim, cspampin, kavasidis, dgiordan@dieei.unict.it_

_•_ _M. Shah and C. Spampinato are with the Center of Research in Computer_
_Vision, University of Central Florida. E-mail: shah@crcv.ucf.edu_

_•_ _J. Schmidt is with the Department of Psychology, University of Central_
_Florida, E-mail: Joseph.Schmidt@ucf.edu_



models by proposing recurrent neural networks with feedforward, feedback, and recurrent connections. These models
have shown promising performance in visual classification
tasks and demonstrate that understanding the human brain
in more detail may allow us to transfer that knowledge
to engineering models to create better machines. Clearly,
before human level classification performance can be transferred to computational models, it is first necessary to better
understand the human visual system. To accomplish this,
we intend to correlate neural activity data recorded from
human subjects while performing specific tasks with our
computational models developed to complete the same task.
By investigating the learned computational representations
and how they correlate with neural activity over time, it is
possible to infer, analyze and eventually replicate complex
brain processes in machines.
In this paper, we first propose a model to learn neural
representations by classifying brain responses to natural
images. Then, we introduce a multimodal approach based
on deep learning EEG and image encoders, trained in a
siamese configuration. Given EEG brain activity data from
several human subjects performing visual categorization
of still images, our multimodal approach learns a joint
brain-visual embedding and finds similarities between brain
representations and visual features. The embedding is then
used to perform image classification, saliency detection, and
to hypothesize possible representations generated in the
human brain for visual scene analysis.
In particular, this paper demonstrates that a) neural activity
data can be used to provide richer supervision to deep learning models, resulting in visual classification and saliency
detection methods aligned with human neural data; b) joint
artificial intelligence and cognitive neuroscience efforts may
lead to uncover neural processes involved in human visual
perception by maximizing the similarity of deep models
with human neural responses. Indeed, we propose a method


to extract visual saliency (and its evolution over time), as
well as to localize the cortical region producing such information; and c) there is potential similarity between computational representations and brain processes, providing
interesting insights about consistency between biological
and deep learning models.
Summarizing, the proposed approach to learn and correlate
brain processes to visual cues, both in time and space,
results in a twofold contribution:


_•_ **Artificial Intelligence.** We introduce new models for
decoding EEG signals related to visual tasks with
state-of-the-art performance and in a biologicallyplausible way. Moreover, our approach allows to
automatically identify computational features that
consistently match human neural activity, representing a new direction to help explain AI models.

_•_ **Cognitive Neuroscience.** Our approach is a step
forward towards providing cognitive neuroscientists with AI-based methodology for understanding
neural responses both in space and time, without
the need to design experiments with multiple subjects and trials. When highly accurate AI is designed, it will allow cognitive neuroscientists to simulate human responses rather than collect significant
amounts of costly data.


The paper is organized as follows. In the next section, we
review recent methods combining brain data and computational models, as well as related approaches to multimodal
learning. In Sect. 3 we describe the core of our approach and
review the specific framework used to learn a joint brainvisual embedding. Next we describe the methods to extract
the most relevant visual information from the image and
the brain activity patterns, as well as their interpretation
(Sect. 4–6). Sect. 7 reports the achieved experimental results
related to image classification and saliency detection, and
shows the learned representations that mostly affect visual
categorization. In the last section, the conclusion and future
directions are presented.


**2** **R** **ELATED WORK**


Our work most directly relates to the fields of EEG
data classification, computational neuroscience for brain
decoding, machine learning guided by brain activity and
multimodal learning. The recent state of the art in these
areas are briefly reviewed in this section.


**EEG data classification** . In recent years, the deep
learning–based approach to classification EEG data has
grown in popularity (a comprehensive review can be
found in [14]). Most of these methods propose a custom
AI solutions used to categorize data in a BCI (Brain
Computer Interface) application (e.g., motor imagery,
speech imagery, emotion recognition, etc.) [15], [16], in
clinical applications [17], [18] (e.g., epilepsy detection and
prediction) or for monitoring cognitive functions [19], [20]
(mental workload, engagement, fatigue, etc.). The above
work focuses on the classification of a few categories (from
binary classification to less than 10 classes), and none of



2


them have the primary objective of understanding human
visual processes (in space and time). Furthermore, most
of the proposed models have been applied to single BCI
paradigms, with very controlled stimuli. The obvious
concern is that these methods may fail to generalize
and their capabilities may collapse with small changes
to the stimuli and task. EEGNet attempts to address
this generalization problem [21]. To do so, the authors
implement a compact convolutional neural network
for EEG classification. The model employs depth-wise
separable convolutions and can be applied to different EEG
experiments, leading to good performance in several tasks:
P300 visual-evoked potentials, error-related negativity
responses (ERN), movement-related cortical potentials
(MRCP), and sensory motor rhythms (SMR). Analogously,

[22] proposes parameterized convolutional filters that learn
relevant information at different frequency bands while
targeting synchrony.
The EEG classification approach proposed in this paper
aims to improve the architectural design concepts of [21],

[22] by modelling more general spatio-temporal features
of neural responses with a goal of supporting cognitive
neuroscience studies to improve the interpretability of
human neural data in time and space.


**Computational** **neuroscience** **for** **decoding** **brain**
**representations** . Decoding brain representations has
been a long sought objective and it still is a great challenge
of our times. In particular, cognitive neuroscience works
have made great progress in understanding neural
representations originating in the primary visual cortex
(V1). Indeed, it is known that the primary visual cortex is a
retinotopically organized series of oriented edge and color
detectors [23] that feed-forward into neural regions focused
on more complex shapes and feature dimensions, which
operate over larger receptive fields in areas V4 [24], before
finally arriving at object and category representations
in the inferior temporal (IT) cortex [25]. Neuroimaging
methods, such as fMRI, MEG, and EEG, have been crucial
for these findings. However, to recreate human level
neural representations that fully represent our visual
processes would require precisely monitoring the activity
of every neuron in the brain simultaneously. Although
these methods are clearly incapable of accomplishing this
lofty goal, they contain enough information to accurately
reconstruct many visual experiences [26]. To that end,
brain representation decoding has recently examined the
correlation between neural activity data and computational
models [1], [2]. However, these approaches mainly perform
simple correlations between deep learned representations
and neuroimaging data and, according to the obtained
outcomes, draw conclusions about brain representations,
which is too simplistic from our point of view. Indeed, the
core point of our idea is that understanding the human
visual system will come as a result of training automated
models to maximize signal correlation between brain
activity and the evoking stimuli, not as a pure analysis
of brain activity data. In addition, while most of the
methods attempt to decode brain representations using
brain images from high spatial resolution fMRI, our work is
the first one to employ EEG data that, despite being lower


spatial resolution, has higher temporal resolution, which
makes it more suitable to decode fast brain processes like
those involved in the visual pathway. Additionally, unlike
fMRI, EEG is portable, ambulatory, and can even be used
wirelessly, traits that would improve any BCI.


**Machine learning guided by brain activity** . The intersection and overlap between machine learning and cognitive
neuroscience has increased significantly in recent years.
Deep learning methods are used, for instance, for
neural response prediction [27], [28], [29], and, in turn,
biologically-inspired mechanisms such as coding theory

[11], working memory [30] and attention [31], [32] are
increasingly being adopted. However, to date, human
cognitive abilities still seem too complex to be understood
computationally, and a data-driven approach for “reverseengineering” the human mind might be the best way to
inform and advance artificial intelligence [33]. Under this
scenario, recent studies have employed neural activity data
to constrain model training. For example, in our recent
work [3], we mapped visual features learned by a deep
feed-forward model to brain-features learned directly from
EEG data to perform automated visual classification. The
authors of [34] employed fMRI data to bias the output
of a machine learning algorithm and push it to exploit
representations found in visual cortex. This work resembles
one of the first methods relying on brain activity data to
perform visual categorization [35], with the distinction that
the former, i.e., [34], explicitly utilizes neural activity to
weigh the training process (similarly to [36]), while the
latter, i.e. [35], proposes a kernel alignment algorithm to
fuse the decision of a visual classifier with brain data.
In this paper, we propose a deeper interconnection between
the two fields: instead of using neural data as a signal
to weigh computationally-learned representations, we
learn a mapping between images and the corresponding
neural activity, so that visual patterns are related in a
one-to-one fashion to neural processes. This mapping, as
we demonstrate in the experimental results, may reveal
much more information about brain representations and be
able to guide the training process in a more intrinsic and
comprehensive way. Thus, our approach is not just a hybrid
machine learning method that is inspired or constrained by
neural data, but a method that implicitly finds similarities
between computational representations, visual patterns
and brain representations, and uses them to perform visual
tasks.


**Multimodal learning** . Our research utilizes multimodal
learning. We exploit the fact that real-world information
comes from multiple modalities, each carrying different
— yet equally useful — content for building intelligent
systems. Multimodal learning methods [37], [38], [39], in
particular, attempt to learn embeddings by finding a joint
representation that encodes the real-world features of the
stimulus across multiple modalities to create a common
concept of the input data.
An effective joint representation must preserve both
intra-modality similarity (e.g., two similar images should
have close vector representations in the joint space; like


3


wise, two equivalent text descriptions should have similar
representations as well) and inter-modality similarity (e.g.,
an image and a piece of text describing the content of that
image should be closer in the joint space than an image
and an unrelated piece of text). Following this property,
most methods find correspondences between visual data
and text [39], [40], [41], [42] or audio [43], [44], [45], [46]
to support either discriminative tasks (e.g., classification) or
prediction of one modality conditioned on another (e.g., image synthesis or retrieval). For the former type of methods,
captions and tags have been used to improve accuracy of
both shallow and deep classifiers [42], [47]. Analogously,

[44] used audio to supervise visual representations; [45], [46]
used vision to supervise audio representations; [48] used
sound and vision to jointly supervise each other; and [43]
investigated how to separate and localize multiple sounds
in videos by analyzing motion and semantic cues. Other
works, have instead focused on predicting missing data in
one modality from the other modality, for example, generate
text descriptions from images and vice versa [49], [50], [51],

[52], [53]. Reed et al. in [52] propose a joint representation
space to condition generative adversarial networks (GANs)
for synthesizing images from text descriptions. Similarly,
Mansimov et al. [53] synthesized images from text captions
using a variational autoencoder. In our recent paper [54], we
used an embedding learned from brain signals to synthesize
images both using GANs and variational autoencoders in a
brain-to-image effort.
In this paper, our approach is inspired by the methods
that learn a shared multimodal representation, with several
crucial differences. First, one of the modalities we utilize is
brain activity data (EEG), which is almost certainly noisier
than then text/audio. This makes it much harder to discover
relationships between the visual and brain modalities. In
this sense, our approach is intended to improve prediction
accuracy and to act as a knowledge discovery tool to uncover brain processes. Thus, our main objective is to learn
a reliable joint representation and explore the learned space
to find correspondences between visual and brain features
that can uncover brain representations; these, in turn, can be
employed to build better deep learning models.
In addition, the proposed deep multimodal network, consisting of two encoders (one per modality), is trained in a
siamese configuration and employs a loss function enforcing
the learned embedding to be representative of intra-class
differences between samples, and not just of the inter-class
discriminative features (as done, for instance, in [52]).


**3** **M** **ULTIMODAL LEARNING OF VISUAL** **-** **BRAIN FEA** **-**


**TURES**


Neural activity (recorded by EEG) and visual data have very
different structures, and finding a common representation
is not trivial. Previous approaches [3] have attempted to
find such representations by training individual models: for
example, by first learning brain representations by training
a recurrent classifier on EEG signals, and then training a
CNN to regress the visual features to brain features for
corresponding EEG/image pairs. While this provides useful
representations, the utility of the learned features is strongly
tied to the proxy task employed to compute the initial


EEG/image pair:



4


_F_ ( _e, v_ ) = _ϕ_ ( _e_ ) _[T]_ _θ_ ( _v_ ) _._ (1)



Fig. 1: **Siamese network for learning a joint brain-image**
**representation** . The idea is to learn a space by maximizing
a compatibility function between two embeddings of each
input representation. Given a positive match between an
image and the related EEG from one subject, and a negative
match between the same EEG and a different image, the
network is trained to ensure a closer similarity (higher compatibility) between related EEG/image pairs than unrelated

ones.


representation (e.g., image classification), and focuses more
on learning class-discriminative features than on finding
relations between EEG and visual patterns.
Hence, we argue that any transformations from human
neural signals and images to a common space should be
learned jointly by maximizing the similarity between the
embeddings of each input representation. To this aim, we
define a siamese network for learning a structured joint
embedding between EEG signals and images using deep
encoders, and maximize a measure of similarity between
the two modalities. The architecture of our model is shown

in Fig. 1.
More formally, let _D_ = _{e_ _i_ _, v_ _i_ _}_ _[N]_ _i_ =1 [be a dataset of neural]
signal samples and images, such that each neural (EEG)
sample _e_ _i_ is recorded on a human subject in response to
viewing image _v_ _i_ . Ideally, latent information content should
be shared by _e_ _i_ and _v_ _i_ . Also, let _E_ be the space of EEG
signal samples and _V_ the space of images. The objective of
our method is to train two encoders that respectively map
neural responses and images to a common space _J_, namely
_ϕ_ : _E →J_ and _θ_ : _V →J_ .

In other approaches for structured learning (e.g. [52]),
the training of the encoders is proxied as a classification
problem based on the definition of a _compatibility function_
_F_ : _E × V →_ R, that computes a similarity measure as
the dot product between the respective embeddings of an



While we employ the same modeling framework, we
formulate the problem as an embedding task whose only
objective is to maximize similarity between corresponding
pairs, without implicitly performing classification, as this
would take us back to the limitation of [3], i.e., learning
representations tied to the classification task.
In order to abstract the learning process from any specific
task, we train our siamese network with a triplet loss aimed
at mapping the representations of matching EEGs and images to nearby points in the joint space, while pushing
apart mismatched representations. We can then stick to
the structured formulation of the compatibility function in
Eq. 1 by employing _F_ directly for triplet loss computation.
Thus, given two pairs of EEG/image ( _e_ 1 _, v_ 1 ) and ( _e_ 2 _, v_ 2 ),
we consider _e_ 1 as the _anchor_ item, _v_ 1 as the _positive_ item
and _v_ 2 as the _negative_ item. Using compatibility _F_ (which
is a similarity measure rather than a distance metric, as is
more commonly used in triplet loss formulations), the loss
function employed to train the encoders becomes:


_L_ ( _e_ 1 _, v_ 1 _, v_ 2 ) = max _{_ 0 _, F_ ( _e_ 1 _, v_ 2 ) _−_ _F_ ( _e_ 1 _, v_ 1 ) _} ._ (2)


This equation assigns a zero loss only when compatibility
is larger for ( _e_ 1 _, v_ 1 ) than for ( _e_ 1 _, v_ 2 ) . Note that class labels
are not used anywhere in the equation. This makes sure
that the resulting embedding does not just associate classdiscriminative vectors to EEG and images, but tries to extract more comprehensive patterns that explain the relations
between the two data modalities. Also, there is no margin
term in Eq. 2, as would be typical in hinge loss formulations
of a triplet loss. This is due to ( _e_ 1 _, v_ 1 ) and ( _e_ 2 _, v_ 2 ) possibly being members of the same visual class, and forcing
a minimum distance between the same-class items is not
strictly needed: as long as the learned representation assigns
a larger compatibility to matching EEG/image pairs and
learns general and meaningful patterns, class separability
would still be implicitly achieved.
In the next subsection we present the architectures of the
EEG encoder _ϕ_ and the image encoder _θ_ .


**3.1** **Encoders’ architectures**


EEG encoder _ϕ_ ( _·_ ), which maps neural activity signals to
the joint space _J_, is a convolutional network, dubbed _EEG-_
_ChannelNet_ and shown in Fig. 2, with a _temporal block_, a
_spatial block_ and a _residual block_, which process different
dimensions of the input signal in different steps following a
hierarchical approach.

The _temporal block_ first processes the input signal along
the temporal dimension, by applying 1D convolutions to
each channel independently, with the twofold purpose of
extracting significant features and reducing the size of the
input signal. In order to be able to capture patterns at multiple temporal scales, the temporal block internally includes
a set of 1D convolutions with different kernel dilation values [55], whose output maps are then concatenated. The role
of the temporal block is to extract information representing
significant temporal patterns _within_ each channel.


The following _spatial block_ aims instead at finding correlations between different channels at corresponding time
intervals, by applying 1D convolutions over the channel
dimension. To clarify this aspect, note that an input EEG
signal of size _C × L_ (with _C_ being the number of channels,
and _L_ the temporal length) will be transformed by the
temporal block into a tensor of size _F ×C_ _×L_ _T_, with _F_ being
the number of concatenated feature maps and _L_ _T_ being the
“new” temporal dimension, after the application of the 1D
convolution. Each element of this tensor will not temporally
correspond to a single sample in the original signal, but it
will “cover” a specific temporal receptive field, depending
on the kernel size and dilation. The spatial block, then,
operates on the feature and channel dimensions across each
element in the _L_ _T_ dimension, with the objective of analyzing spatial correlations at corresponding times (on multiple
scales). Similar to the temporal block, the spatial block also
consists of multiple 1D convolutional layers whose outputs
are concatenated. In this case, the channel dimension is
sorted so that “rows” of channels (according to the 10-20
layout depicted in Fig. 5) are appended consecutively in the
signal matrix; then, each spatial 1D convolution operates
with different kernel sizes. All convolutional layers in the
temporal and spatial blocks are followed by batch normalization and ReLU activations. Once the model has worked
independently on the temporal and spatial dimensions, the
final _residual block_, consisting of a set of residual layers [56],
performs 2D convolution on the spatio-temporal representation to find more complex relations and representations
from the signal. Each residual layer performs two convolutions (with batch normalization and ReLU activation)
before summing the input to the residual. The output is
then provided to a final convolutional layer followed by
a fully-connected layer, having the same size as the joint
embedding dimensionality.

The proposed encoder is first tested for EEG classification, by suitably adding a softmax layer after the fully
connected one, in order to understand its capabilities to
decode visual information from neural data (see Sect. 7.3).
Afterwards, the encoder is trained using the siamese schema
presented earlier.

Visual encoder, _θ_ ( _·_ ) maps, instead, images to the joint
space _J_ through convolutional neural networks. We use a
pre-trained CNN to extract visual features and feed them
to a linear layer for mapping to the joint embedding space.
Differently from [52], we learn the compatibility function
in an end-to-end fashion, also by fine-tuning the image
encoder, in order to better identify low- and middle- level
visual-brain representations, which — suitably decoded —
may provide hints on what information is used by humans
when analyzing visual scenes.


**4** **I** **MAGE CLASSIFICATION AND SALIENCY DETEC** **-**


**TION**


Our siamese network learns visual and EEG embeddings in
order to maximize the similarities between images and related neural activities. We can leverage the learned manifold
for performing visual tasks. In cognitive neuroscience there
is converging evidence that: a) brain activity recordings



5


Fig. 2: **Detailed EEG-ChannelNet Architecture** . The EEG
signal is first processed by a bank of concatenated 1D convolutions over channels ( _temporal block_ ), followed by a bank
of concatenated 1D convolutions across channels ( _spatial_
_block_ ). The resulting features are then processed by a cascade
of residual layers, followed by a final convolution and
a fully-connected layer projecting to the joint embedding
dimensionality.


contain information about visual object categories (as also
demonstrated in [3]) and b) attention influences the processing of visual information even in the earliest areas of the
primate visual cortex [57]. In particular, bottom-up sensory
information and top-down attention mechanisms seem to
fuse in an integrated saliency map, which in turn, distributes
across the visual cortex. Thus, EEG recordings in response to
visual stimuli should encode both visual class and saliency
information. However, for image classification we can simply use the trained encoders as feature extractors for a


subsequent classification layer (see performance evaluation
in Sect. 7), whereas for saliency detection we designed a
_multiscale suppression-based_ approach, inspired by the methods identifying pixels relevant to CNN neuron activations
(e.g., [58]), that analyzes fluctuations in the compatibility
measure _F_ (1). The idea is based on measuring how brainvisual compatibility varies as image patches are suppressed
at multiple scales. Indeed, the most important features in an
image are those that, when inhibited in an image, lead to the
largest drop in compatibility score (computed by feeding an
EEG/image pair to the siamese network proposed in the
previous section) with respect to the corresponding neural
activity signal. Thus, we employ compatibility variations
at multiple scales for _saliency detection_ . Note that, for this
approach to work, the EEG encoder must have learned to
identify patterns related to specific visual features in the
observed image, so that the absence of those features reflects
on smaller similarity scores on the joint embedding space.
The saliency detection method is illustrated in Fig. 3
and can be formalized as follows. Let ( _e, v_ ) be an
EEG/image pair, with compatibility _F_ ( _e, v_ ) . The saliency
value _S_ ( _x, y, σ, e, v_ ) at pixel ( _x, y_ ) and scale _σ_ is obtained by
removing the _σ_ _×_ _σ_ image region around ( _x, y_ ) and computing the difference between the original compatibility score
and the one after suppressing that patch. More formally, if
_m_ _σ_ ( _x, y_ ) is a binary mask where all pixels within the _σ × σ_
window around ( _x, y_ ) are set to zero, we have:


_S_ ( _x, y, σ, e, v_ ) = _F_ ( _e, v_ ) _−_ _F_ ( _e, m_ _σ_ ( _x, y_ ) _⊙_ _v_ ) _,_ (3)


where _⊙_ denotes element-wise multiplication (Hadamard
product). For multiple scale values, we set the overall
saliency value for pixel ( _x, y_ ) to the normalized sum of (per
scale) saliency scores:


_S_ ( _x, y, e, v_ ) = � _S_ ( _x, y, σ, e, v_ ) _._ (4)


_σ_


Normalization is then performed on an image-by-image
basis for visualization.


**5** **V** **ISUAL** **-** **RELATED BRAIN PROCESSES**


While the saliency detection approach studies how alterations in images reflect on compatibility scores, it is even
more interesting to analyze how neural patterns act on
the learned representations. Indeed, following the principle
that large variations in compatibility can be found when
the most important visual features are masked, we may
similarly expect compatibility to drop when we remove
“important” (from a visual feature–matching point of view)
components from neural activity signals. Performing this
analysis traditionally requires a combination of _a priori_
knowledge on brain signal patterns and manual analysis: for
example, it is common to investigate the effect of provided
stimuli while monitoring the emergence of event-related
potentials (ERPs) known to be associated to specific brain
processes. Of course, posing the problem in this way still
requires that the processes under observation be at least partially known, which makes it complicated to automatically
detect previously-unknown signal patterns.
Instead, the joint representation makes it easy to correlate brain signals with visual stimuli by analyzing how



6


Fig. 3: **Our multiscale suppression-based saliency detec-**
**tion** . Given an EEG/image pair, we estimate the saliency
of an image patch by masking it and computing the corresponding variation in compatibility. Performing the analysis
at multiple scales and for all image pixels results in a
saliency map of the whole image. Note that, although the
example scale-specific saliency maps appear pixellated, that
is only a graphical artifact to give the effect of scale: in
practice, scale-specific maps are still computed pixel by
pixel.


compatibility varies in response to targeted modifications
of the inputs. Thus, similar to saliency detection, we can
identify the spatial components in brain activity that convey
visual information.
As mentioned in Sect. 2, object recognition in humans
is performed by a multi-level aggregation of shape and
feature information across cortical regions, resulting in a
distributed representation that can easily adapt to a wide
variety of tasks on the received stimuli. For these reasons,
understanding how this distributed representation is spatially localized over the brain cortex is a fundamental step
towards a successful emulation of the human visual system.
In order to evaluate the importance of each EEG channel
(and corresponding brain area), we employ the learned joint
embedding space to “filter” (the exact procedure is defined
below) that channel from the EEG signal and measure the
corresponding change in compatibility between images and
filtered signals.
The importance of each channel for a single EEG/image
pair can be measured by computing the difference between
the pair’s compatibility score and the compatibility obtained
when suppressing that channel from the EEG signal. Ideally,
given a generic EEG/image pair ( _e, v_ ), and indicating with
_e_ _−c_ a transformation of _e_ such that information on channel
_c_ is suppressed, we could define the _importance_ of channel _c_
for the ( _e, v_ ) pair as:


_I_ ( _e, v, c_ ) = _F_ ( _e, v_ ) _−_ _F_ ( _e_ _−c_ _, v_ ) _._ (5)


The intuition behind this formulation is that the suppression
of a channel that conveys unnecessary information (at least,


from the point of view of the representation learned by
the EEG encoder) should result in a small difference in
the compatibility score; analogously, if a channel contains
important information that match brain activity data to
visual data, compatibility should drop when that channel
is suppressed.

In practice, finding a single ideal replacement of channel _c_ to compute _e_ _−c_ is hard, since different substitutions
yield varying compatibility scores. However, we found that
averaging compatibility differences over a large number of
random replacements of the _c_ channel gives stable results:
hence, we modify Eq. 5 to compute the importance score as
the expected value of the difference in compatibility when
replacing channel _c_ with sequences of random Gaussian
samples, low-pass filtered at 100 Hz and distributed according to the original channel’s estimated statistics (mean and
variance).

More formally, if EEG signal _e_ is represented as a matrix
with one channel per row:



7


_generators_ . To fill this gap, we propose an additional modality for interpreting compatibility differences, by employing
the learned manifold to carry out an analysis of the EEG
channels — and, therefore, the corresponding brain regions
— that are most solicited in the detection of visual characteristics at different scales, from edges to textures to objects and
visual concepts. To carry out this analysis, we evaluate the
differences in compatibility scores computed when specific
feature maps in the image encoder are removed, and map
the corresponding features to the EEG channels that appear
to be least active (compatibility-wise) when those features
were removed. In practice, given EEG/image pair ( _e, v_ ),
let us define _F_ ( _e, v_ _−l,f_ ) as the value of the compatibility
function computed by suppressing the _f_ -th feature map
at the _l_ -th layer of the image encoder. According to Eq. 7,
given EEG/image pair ( _e, v_ ), the importance of channel _c_
computed when a certain layer’s feature is removed is:


_I_ ( _e, v_ _−l,f_ _, c_ ) =













































_F_ ( _e, v_ _−l,f_ ) _−_ E






_F_





_e_ 1

_e_ 2

_. . ._
_H_ � _N_ ( _µ_ _c_ _, σ_ _c_ 2 [)] _[L][×]_ [1] �


_. . ._

_e_ _n_



_, v_ _−l,f_



_e_ =


we compute _I_ ( _e, v, c_ ) as:















_e_ 1

_e_ 2

_. . ._

_e_ _c_

_. . ._

_e_ _n_



_,_ (6)















_._ (9)































_e_ 1

_e_ 2

_. . ._
_H_ � _N_ ( _µ_ _c_ _, σ_ _c_ 2 [)] _[L][×]_ [1] �


_. . ._

_e_ _n_





















_I_ ( _e, v, c_ ) = _F_ ( _e, v_ ) _−_ E






_F_





_, v_















_,_



We then define the _association_ between feature ( _l, f_ ) and
channel _c_ for a pair ( _e, v_ ) as follows:


_A_ ( _e, v, c, l, f_ ) = _I_ ( _e, v_ _−l,f_ _, c_ ) _−_ _I_ ( _e, v, c_ ) _._ (10)


We consider channel _c_ and feature ( _l, f_ ) “associated” if, after
removing the intrinsic importance score for that channel for
a given ( _e, v_ ) pair, the variation in compatibility for channel
_c_ does not vary when that feature is removed, which would
mean no visual component in the encoded representation is
left unmatched.

We can estimate the association between channel _c_ and
layer _l_ by averaging over all features in that layer:


_A_ ( _e, v, c, l_ ) = E _f_ [ _A_ ( _e, v, c, l, f_ )] _._ (11)


The resulting score provides an interesting indication of
how much the features computed at a certain layer in a
computational model resemble the features processed by the
brain in specific scalp locations.

Finally, as for the channel importance score, we can
compute general association scores by averaging over the
entire dataset:


_A_ ( _c, l_ ) = E ( _e,v_ ) [ _A_ ( _e, v, c, l_ )] _._ (12)


**7** **E** **XPERIMENTS AND APPLICATIONS**


Before we apply our joint learning strategy, we first tested
the EEG classification accuracy of the EEG-ChannelNet
model (described in Sect. 7.2) on the brain-visual dataset.
The objective of this stage is to investigate the extent to
which EEG data encodes visual information, along with
the nature and importance of EEG temporal and spectral



(7)


where _µ_ _c_ and _σ_ _c_ [2] [are the sample mean and variance for]
channel _c_, _L_ is the EEG temporal length, _N_ ( _µ, σ_ [2] ) _N_ _×M_ is
an _N × M_ matrix sampled from the specified distribution,
and _H_ is a low-pass filter at 100 Hz.

Finally, since channel importance scores computed over
single EEG/image pairs may not be significant by themselves to draw general conclusions, we extend the definition
of channel importance over multiple data samples:


_I_ ( _c_ ) = E ( _e,v_ ) [ _I_ ( _e, v, c_ )] _,_ (8)


where the expectation is computed over all dataset samples
(or a subset thereof, e.g. when grouping by class).


**6** **D** **ECODING BRAIN REPRESENTATIONS**


Each of the previous approaches investigated the effect of
altering either the brain activity signals or the image content,
but they are limited in that the differential analysis they
provide is carried out in only one modality: we can identify
the visual features that impact the similarity between two
corresponding encodings the most, or we can identify the
spatial patterns in brain activity that are more relevant to the
learned representation. However, we still do not know _which_
visual features give rise to _which_ brain responses, i.e. _neural_


content. Lastly, we hope to provide a baseline for the subsequent investigations. We then evaluate the quality and
meaningfulness of the joint encoding learned by our model.
The main objective is to assess the correspondence of visual
and neural content in the shared representation:


_•_ _Brain signal/image classification_ : we evaluate if and
how the learned neuro-visual manifold is representative of the two input modalities by assessing their
capabilities to support classification tasks (i.e., brain
signal classification, and image classification). We additionally compare the obtained results to the same
models trained in a traditional supervised classification tasks.

_•_ _Visual saliency detection from neural activity/image com-_
_patibility variations_ : the similarity of the mapped neural signals and images should be based on the most
salient features in each image, as described in Sect. 4.
This evaluation, therefore, assesses the performance
of our brain-based saliency detection and compares
it to the state of the art methods.

_•_ _Localizing neural processes related to visual content_ : this
experiment identifies neural locations (as indicated
by learned EEG scalp activity), related to specific
image patches by using the method described in
Sect. 5. By combining the results of this analysis
with the saliency detection results, we obtain the first
ever retinotopic saliency map created by training an
artificial model with salient visual features correlated
with neural activity.

_•_ _Correlating deep learned representations with brain activ-_
_ity_ : by analyzing the learned visual and neural patterns, we identify the most influential learned visual
features (kernels) and how they correlate with neural
activity. The outcome of this evaluation indicates
roughly what visual features are correlated with human visual representations. This is an important first
step towards developing a methodology to better
uncover and emulate human brain mechanisms.


**7.1** **Brain-Visual Dataset**


In order to test and validate our method we acquired EEG
data from 6 subjects.

The recording protocol included 40 object classes with 50
images each, taken from the ImageNet dataset [59], giving a
total of 2,000 images. Object classes were chosen according
to the following criteria:


_•_ The object classes should be known and recognizable
by all the subjects with a single glance;

_•_ The object classes should be conceptually distinct
and distant from each other (e.g., dog and cat categories are a good choice, whereas German shepherd
and Dalmatian are not);

_•_ The images corresponding to an object class should
occupy a large portion of the image, and background
of the image should be minimally complex (e.g., no
other salient or distracting objects in the image).


Visual stimuli were presented to the users in a blockbased setting, with images of each class shown consecutively in a single sequence. Each image was shown for 0.5



8


Fig. 4: Power spectral density of a subject’s EEG recording,
after 5-95 Hz band-pass filtering and notch filtering at 50
Hz.


seconds. A 10-second black screen (during which we kept
recording EEG data) was presented between class blocks.

The collected dataset contains in total 11,964 segments
(time intervals recording the response to each image); 36
have been excluded from the expected 6 _×_ 2,000 = 12,000
segments due to low recording quality or subjects not
looking at the screen, checked by using the eye movement
data described in Sect. 7.5. Each EEG segment contains
128 channels, recorded for 0.5 seconds at 1 kHz sampling
rate, represented as a 128 _×_ L matrix, with _L ≈_ 500 being
the number of samples contained in each segment on each
channel. The exact duration of each signal may vary, so we
discarded the first 20 samples (20 ms) to reduce interference
from the previous image and then cut the signal to a
common length of 440 samples (to account for signals with
_L <_ 500 ).
All signals were initially frequency-filtered. In detail, we
applied a second-order Butterworth bandpass filter between
5 Hz and 95 Hz and a notch filter at 50 Hz. The filtered
signal is then z-scored — per channel — to obtain zerocentered values with unitary standard deviation. We use 95
Hz as high-frequency cut-off since frequencies above _[∼]_ 100
Hz rarely have the power to penetrate through the skull. An
example of the power spectral density of a subject’s EEG
recording after filtering is shown in Fig. 4.
EEG placement and a mapping to the brain cortices
is shown in Fig. 5, where we also show the neural activity visualization scale employed in this paper. Activity
heatmaps for an image/EEG pair are generated by applying
Eq. 7 and 8 to estimate how much each channel affects
the pair’s compatibility, then plotting normalized channel
importance scores on a 2D map of the scalp (at the positions
corresponding to the electrodes of the employed EEG cap),
and applying a Gaussian filter for smoothing (using a kernel
with standard deviation of 13 pixels, for a 400 _×_ 400 map).
In order to replicate the training conditions employed
in [3] and to make a fair comparison for brain signal
classification, we use the same training, validation and test
splits of the EEG dataset for the block-based design signals,
consisting respectively of 1600 (80%), 200 (10%), 200 (10%)
images with associated EEG signals, ensuring that all signals
related to a given image belong to the same split.


Fig. 5: **Mapping between EEG channels and brain cor-**
**tices.** (Left) EEG channel placement and corresponding
brain cortices (background image source: Brain Products
GmbH, Gilching, Germany). We use a 128-channel EEG,
where each channel in the figure is identified by a prefix
letter referring to brain cortex (Fp: frontal, T: temporal, C:
central, P: parietal, O: occipital) and a number indicating
the electrode. (Right) Neural activation visualization — top
view of the scalp — employed in this paper. A detailed
mapping between EEG channels and brain cortices can be
found in [60].


**7.2** **Model implementation**


In our implementation of the EEG encoder described in
Sect. 3.1, we employ five convolutional layers in the temporal block, with increasing kernel dilation values (1, 2, 4,
8, 16), and four convolutional layers in the spatial block,
with kernel sizes ranging from 128 (i.e., including all EEG
channels) to 16, scaling down in powers of two. Note that in
both cases, the convolutional layers do not process the input
in cascade, but are applied in parallel and the corresponding
outputs are concatenated along the feature dimension. On
the contrary, the following residual block contains four
sequential residual layers, that finally lead to a non-residual
convolution and a fully-connected layer that projects to the
joint embedding space with a size of 1000.

Fig. 2 (bottom part) shows the details of the architecture
in terms of layer parameters and feature map sizes. For
an efficient implementation of the temporal and spatial
1D convolution, we treat the input EEG signal as a onechannel bidimensional “image” of size 1 (feature map) _×_
128 (channels) _×_ 440 (time), and apply 2D convolutions with
one dimension of the kernels equal to 1 (namely, 1 _× K_ for
temporal layers, and _K ×_ 1 for spatial layers). Finally, note
that padding values were used in the temporal and spatial
blocks so that the output of each layer has the same size
concatenation dimensions.


**7.3** **EEG classification**


Our first experiment is intended to assess whether the
architecture employed for the EEG encoder is able to extract visual-related information from the EEG signals, and
whether additional data post-processing (frequency filtering, temporal subsequencing) can affect performance. To



9


**Band** **Frequency - Hz** **Accuracy**


Theta, alpha and beta 5-32 19.7%
Low frequency gamma 32-45 26.1%
**High frequency gamma** **55-95** **48.1%**
All gamma 32-95 40.0%
All frequencies 5-95 31.3%


TABLE 1: EEG classification accuracy using different EEG
frequency bands. We discard the 45–55 frequencies because
of the power line frequency at 50Hz.


this end, we trained the EEG encoder by itself to carry out
visual classification from EEG signals, in the same fashion
as [3]. Note, this serves as a validation procedure of the
EEG encoder architecture and as a baseline comparison with
other state of the art models. This does not serve as any
sort of pre-training for the full joint-embedding of the EEG
and image encoders, rather to demonstrate the EEG carries
visual information.

To run this experiment, we append a softmax classification layer to the EEG-ChannelNet architecture (defined
in Sect. 3.1 and shown in Figure 2), and train the whole
model to estimate the visual class corresponding to each
EEG signal. The model is trained for 100 epochs, using
a mini-batch size of 16, employing the Adam optimizer
with suggested hyperparameters (learning rate: 0.001, _β_ 1 :
0.9, _β_ 2 : 0.999), and a mini-batch size of 16. As classification
accuracy we report the accuracy obtained on the test set
at the training epoch when the maximum accuracy on the
validation set was achieved, according to the splits defined
in Sect. 7.1.

Additionally, we also report the results obtained when
training on certain frequency sub-bands and when employing only portions of the original signals.

EEG signals contain several frequencies which are usually grouped into five bands: delta (0-4 Hz), theta (4-8
Hz), alpha (8-16 Hz), beta (16-32 Hz) and gamma (32 Hz
to 95 Hz; this band can be further split into low- and
high-frequency gamma with 55 Hz as cut-off). Gamma
is believed to be responsible for cognitive processes, and
high-frequency gamma (from 55 to 100 Hz) for short-term
memory matching of recognized objects (working memory)
and attention. Given that, our experimental design should
elicit high gamma in participants. Thus, we also compute
the performance when selecting only the above frequency
bands: the results, given in Table 1, show that higher performance is achieved on high gamma frequencies, which
is consistent with the cognitive neuroscience literature on
attention, working memory and perceptual processes involved in visual tasks [61], [62], [63]. Given these results,
all following evaluations are carried out by applying a 5595 Hz band-pass filter.
We then evaluate performance when using the entire
time course of the EEG signals and a set of temporal EEG
sub-sequences (of size 220 ms, 330 ms, or 440 ms) to understand when the neural response results in the strongest
classification performance. Note that the EEG encoder architecture described in Sect. 7.2 was not adapted for this
evaluation: the convolutional portion of the model is unaffected by the reduced temporal length, which reflects only
the input size to the fully-connected layer. Results are shown


**EEG Time interval [ms]** **EEG Classification Accuracy**


20-240 39.4%

20-350 43.8%

**20-460** **48.1%**

130-350 23.5%

130-460 26.3%

240-460 38.9%


**SOTA methods**


Method **EEG Classification Accuracy**


[3] 21.8%

[21] 31.9%

[22] 31.7%


TABLE 2: (Top) EEG classification accuracy using different
EEG time intervals with data filtered in the [55-95] Hz
band. (Bottom): Classification performance of state of the
art methods using the whole EEG time course (i.e., 20-460
ms).


in Tab. 2, and show that when using shorter EEG segments,
performance is lower than when we use the entire time
course (20-460 ms). Additionally, leaving out the first 110 ms
negatively affects the performance, suggesting that human
perceptual processes tied to stimulus onset are critical for
decoding image classes. Furthermore, adding the last 110 ms
increases the performance by 5 percentage points (20-350 ms
vs 20-460 ms), meaning that the later, more cognitive operations performs a further refinement of the learned features
which enhances classification; this is also in line with the
neurocognitive literature about experiments based on eventrelated potentials (ERP) [64]. The almost equal performance
between 20-240 ms (39.4%) and 240-460 ms (38.9%) may
suggest a balanced importance between an initial low-level
visual feature extraction and a more cognitive aggregation
into different abstraction levels (evidence of this processing
can be found in the saliency detection analysis in Sect. 7.5).
Finally, we compare classification performance achieved
by our EEG encoder and other state-of-the-art methods,
namely [3] [1] [21], [22], using high-frequency gamma band
data, i.e., 55-95 Hz. EEG classification accuracy on the test
split is given in Tab. 2 and shows that our approach reaches
an average classification accuracy of 48.1%, outperforming
previous methods, such as EEGNet, which, only achieves a
maximum accuracy of 31.9%.


**7.4** **Siamese network training for classification**


The previous section tested the EEG-ChannelNet architecture and compared it to the state of the art standard
supervised EEG classification task. This showed that our
method is capable of uncovering brain activity patterns, and
demonstrated the frequency bands and temporal ranges that
produce the highest classification accuracy. This provides a
clear baseline to understand the contribution of our methods. In this section, we describe the training procedure for
our siamese network and evaluate the quality of the learned


1. Note that its performance are lower than those reported in [3],
since in that work, frequency filtering was carried out incorrectly, i.e.,
DC component was not removed, leaving the EEG drift that induced a
bias in signals and, consequently, higher performance.



10


**Image encoder** **EEG** **Image** **Avg**


Inception-v3 60.4 % 94.4 % 77.4 %
ResNet-101 50.3 % 90.5 % 70.4 %

DenseNet-161 54.7 % 92.1 % 73.4 %

AlexNet 46.2 % 69.4 % 57.8 %


TABLE 3: EEG and image classification accuracy obtained
using the joint-learning approach, for different layouts of
the image encoders.


joint embedding. In particular, we investigate a) what configurations of the two EEG and image encoders (defined
in Sect. 3) provide the best trade-off between EEG and
image classification, b) how conditioning the classifier for
one modality over the other affects classification accuracy,
and c) if augmenting the visual representation space with
features derived from the brain leads to better performance
than state-of-the-art methods that only use visual features.
We train our siamese network (the EEG and image
encoders), by sampling a triplet ( _e_ _i_ _, v_ _i_ _, v_ _j_ ) of one EEG
( _e_ _i_ ) and two images ( _v_ _i_, _v_ _j_ ), representing, the positive
( _e_ _i_ _, v_ _i_ ) and negative ( _e_ _i_ _, v_ _j_ ) samples. Similar to the pure
classification experiment, we used an Adam optimization
algorithm with hyperparameters for our contrastive loss, a
mini-batch size of 16, and the number of training epochs
was set to 100. We also test different configurations of the
image encoders to investigate whether the achieved results
are independent of the underlying model. In particular, we
employ different image classification backbones as feature
extractors, namely, ResNet-101, DenseNet-161, Inception-v3,
and AlexNet. All of these models are first pre-trained on the
ImageNet dataset, and then fine-tuned during our siamese
network training. We perform data augmentation by generating multiple crops for an image associated to a given EEG
sample. In particular, we resize each image by a factor of
1.1 with respect to the image encoder’s expected input size
(299 _×_ 299 for Inception-v3, 224 _×_ 224 for the others). We then
extract ten crops from the four corners and the center of the
input image, with corresponding horizontal flips.
Once training is completed, we use the trained EEG and
image encoders as feature extractors in the joint embedding
space, followed by a softmax layer, for both image and
EEG classification. The classification tasks provide a way
to assess the quality of our multimodal learning approach
and allow us to identify the best encoders’ layouts, based
on the accuracy of the validation set. The specific values for
the number of convolutional layers, layer sizes, number of
filters, manifold size are empirically derived to produce the
best validation performance in our experiments.
Table 3 shows the obtained EEG and image classification
accuracy for all the tested models. Note that all configurations benefit from the joint embedding learning, and
achieve a classification accuracy on par or better than when
training the EEG encoder alone in the standard supervised
classification scenario.

Next, we test the impact of one modality on the other,
i.e., the effect of jointly learning brain activity–derived features and visual features with respect to training singlemodality models. We first compare the image classification
performance obtained by the pre-trained image encoders


**Image classification performance**


Model **Visual Learning** **Joint Learning**


Inception-v3 93.1 % 94.4 %
ResNet-101 90.3 % 90.5 %

DenseNet-161 91.4 % 92.1 %

AlexNet 65.5 % 69.4 %


**EEG classification performance**


**EEG encoder** **EEG Learning** **Joint Learning**


EEG-ChannelNet 48.1% 60.4%


TABLE 4: Comparison of image and EEG classification
performance when using only one modality (either image
or EEG) relative to when we use the joint neural-visual
features. For each model, we report the best performance
according to Tab. 3. The reported EEG classification performance for our approach are achieved when training the
image encoder using Inception-v3.


alone and by the image encoders obtained after fine-tuning
with our joint-embedding approach. Both our model and
pre-trained visual encoders are used as feature extractors,
followed by a softmax layer, and performance is computed
on the test split of the employed visual dataset. Note that
since the 40 target images classes are included in ImageNet,
the pre-trained visual encoders were previously trained
on them. Therefore, we simply perform fine-tuning with
the joint embedding learning, i.e., the pre-trained visual
encoders are trained to maximize the correlation between
the visual and EEG content, rather than on classification _per_
_se_ . The results in Tab. 4 indicate that learning features that
maximize EEG-visual correlation (as discussed in Sect. 3)
leads to enhanced performance in all models. The largest increase occurred when AlexNet was the image encoder. This
is likely due to the fact that the other models are complex
enough to “saturate” the classification capacity (i.e. there is
a ceiling effect), and suggests that the proposed approach
might be useful for domain-specific tasks (e.g., medical
imaging) that are particularly complex and/or where data
may be limited.
Analogously, we compare the performance of the EEG
signal classification accuracy to the EEG encoder described
in Sect. 3 and the one obtained by our joint neural-visual
learning. The results are given in Table 4 and show that
the addition of visual features to the EEG classification
improves performance by about 12 percentage points. Thus,
the proposed joint learning scheme allows us to bring EEG
classification from 48.1%, using state-of-the-art approaches
(see Tab. 2), to 60.4%.
By comparing performance of the EEG and image classification in Tab. 4, it is important to note that the EEG
classification benefits more from the use of both modalities
than the image classification does. This is not surprising,
given the high image only classification accuracy, and the
noisy and mostly-unexplored nature of neural activity data.
In this case, the integration of the more easily-classifiable
visual features helps to “guide” the learning from the neural
data to create more discriminative representations and to
produce a better-performing model. Note, however, EEG



11


classification relies on the features computed by the EEG
encoder guided by the joint visual representation, rather
than classifying the visual information itself (i.e. visual
features are not employed during the EEG classification).
Importantly, the addition of the EEG information improved
performance the most when image classification was lower
(i.e. classification accuracy was not yet at ceiling). This
possibly suggests that when human classification accuracy
is far higher than model classification accuracy, the neural
data helps more.


**7.5** **Saliency detection**


In the previous experiments, we demonstrated that the
learned EEG/image embedding is able to encode enough
visual information to perform both EEG and image classification. Now we investigate if and how the shared visualbrain space relates to visual saliency using the approach described in Sect. 4. We measured how compatibility between
the trained encoders and various image patches changes.
The values for the _σ_ parameter in Eq. 3 are set to 3, 5, 9, 17,
33, and 65 pixels. Note that this evaluation does not require
any additional training, and can be based on the same EEG
and image encoders as described in Sect. 7.4. However,
in order to avoid bias due to pre-trained encoders on the
same dataset, the saliency experiments are carried out on retrained versions of the models using a leave-one-out setup:
for each visual class in our dataset, new EEG and image
encoders are trained on the remaining 39 classes, using the
same joint-embedding configuration described in Sect. 7.4.
In this case, compatibility essentially measures how much a
given image patch accounts for the joint representation. This
serves as a measure of the importance of the given image
patch; patches associated to large drops in compatibility
must contribute more to the joint representation.

For this analysis, we used eye movement data recorded
— through a 60-Hz Tobii T60 eye-tracker — on the same six
subjects of above, at the same time of EEG data acquisition,
i.e., while they were looking the 2,000 displayed images.
We employed this data as saliency detection dataset and the
images were divided into the same training, validation and
test splits of the EEG classification experiment. As a baseline
comparison, we used the pre-trained SALICON [65] and
SalNet [66] models, fine-tuned on the dataset’s training data.
In addition, to demonstrate that EEG indeed encodes visual
saliency information and that the generated maps are not
simply driven by the image encoder, we include an additional baseline by implementing an approach similiar to the
one described in Sect. 4. We used the pre-trained Inceptionv3 visual classifier because it produces better classification
performance (see Tab. 3). We then apply the same multiscale patch-suppression method, however, in this case, the
saliency score is not based on compatibility, but rather it
is based on the log-likelihood variation for the image’s
correct class. More formally, given image _v_ and denoting
with _p_ ( _v_ ) the log-likelihood of _v_ ’s correct class as estimated
by a pre-trained Inception-v3 network, the saliency value
_S_ classifier ( _x, y, σ, v_ ) at pixel ( _x, y_ ) and scale _σ_ is computed as:


_S_ classifier ( _x, y, σ, v_ ) = _p_ ( _v_ ) _−_ _p_ ( _m_ _σ_ ( _x, y_ ) _⊙_ _v_ ) _,_ (13)


where _m_ _σ_ ( _x, y_ ) _⊙_ _v_, as previously, is the result of the
removal of the _σ × σ_ region around ( _x, y_ ) . Also in this


case, the computed saliency value at a certain location is
the normalized sum over multiple scales.
Fig. 6 shows qualitatively the saliency maps obtained
by our approach, relative to the state-of-the-art saliency
detectors [65], [66] and our baseline. We also quantitatively assess the accuracy of the maps generated by our
joint-embedding–driven saliency detector by computing the
metrics defined by [67] — shuffled area under curve (sAUC), normalized scanpath saliency (NSS) and correlation
coefficient (CC) scores. Tab. 5 reports the results achieved
by our saliency detection method, showing a) that our
method outperforms the baseline saliency detectors and b)
the contribution of the joint neural/visual features improving performance w.r.t. visual features alone. Importantly,
this suggests that our joint embedding method accounts
for more of the regions that human subjects fixate during
free viewing (i.e. empirically derived saliency) than any
other tested saliency methods or visual classification alone.
It is also interesting to note that the metric for which our
method yields the largest improvement is NSS, which is the
most relevant to the nature of EEG signals, being related to
the gaze fixation scan path and thus measuring a temporal
aspect of saliency.

To understand how saliency evolves over time, we additionally evaluate the importance of different temporal
subsamples of the EEG signal on the saliency maps. This
is similar to the procedure that was done in Sect. 7.3.
Fig. 7 shows that we tested saliency across a variety of time
ranges (20–240 ms, 130–350 ms and 240–460 ms). Over time,
subjects appear to focus on different parts of the image.
Interestingly, early on visual attention seems to be more
controlled by visual features such as color contrast and
edges, while later times show that attention tends to be
oriented moreso towards the context or object category (i.e.,
the object of most interest to the observer). This matches
theories of visual attention in humans, i.e. early on, attention
is dominated by an early bottom-up unconscious process
driven by basic visual features such as color, luminance,
orientation, and edge detection; whereas later on, attention
is driven by top-down process, which bias the observer
towards regions that demonstrate context and consciously
attract attention due to task demands [68]. Furthermore,
saliency changing over time shows that humans also pay
attention to basic visual features as well as context. This is
consistent with the idea that object categorization in humans
is based on a combination of object and context features [69].
Finally, Fig. 8 indicates that the the saliency derived using
both brain and visual features is not strictly connected to the
features necessary for visual recognition. For example, in the
first row of Fig. 8, the ImageNet class is “mobile phone” but
the derived saliency focuses more on the baby (and in the
employed image dataset there is no _face/person_ class). This
holds for all the reported examples.


**7.6** **Decoding Brain Representations**


The objective of this analysis is to approximate the spatial distribution of the cortex-level representations: indeed,
while the hierarchical multi-stage architecture of the human
visual pathway is known, the representations generated at
each stage are poorly understood. In these experiments,



12


**Method** **s-AUC** **NSS** **CC**


SalNet 0.637 0.618 0.271

SALICON 0.678 0.728 0.348
Visual classifier–driven detector 0.532 0.495 0.173
**Our neural-driven detector** **0.643** **0.942** **0.357**


Human Baseline 0.939 3.042 1


TABLE 5: Saliency performance comparison in terms of
shuffled area under curve (s-AUC), normalized scanpath
saliency (NSS) and correlation coefficient (CC) between
our compatibility-driven saliency detector and the baseline
models. We also report the human baseline, i.e., the scores
computed using the ground truth maps. Since we adopt a
leave-out-one setup the reported values for our approach
are averaged over all the 40 experiments.


Fig. 6: **Qualitative comparison of generated saliency maps.**
From left to right: input image, human gaze data (ground
truth), SALICON, SalNet, visual classifier–driven detector,
and our visual/EEG–driven detector. It can be noted a) that
the maps generated by our method resemble the ground
truth masks more than the state-of-the-art methods; b)
adding brain activity information to visual features results
in an improved reconstruction (more details and less noise)
in the saliency calcualtion (compare the 5 [th] and 6 [th] columns).


we performed a coarse analysis on the global interaction
between neural activity and images, and a fine analysis
on the interaction between neural activity and the deeplearned visual features. This procedure allows us to identify
which neural areas (scalp regions) are the most informative.
Whereas the underlying cortex cannot be precisely isolated
with EEG alone, this procedure points to the temporal
and spatial components of the joint representation that are
sensitive to relevant to visual cues. Of course, this analysis is
purely qualitative, since no “correct” or unequivocal answer
is available. Nevertheless, we believe that it is important
to verify that the generated representations are intuitively
meaningful and consistent with what can be expected from
the neurocognitive point of view.


Fig. 7: **Qualitative evaluation of saliency detection at**
**different times.** From left to right: input image, saliency
detection using EEG data in time range [20–240] ms, saliency
detection in time range [130–350] ms, saliency detection in
time range [240–460] ms and saliency detection using the
entire EEG time course, i.e, [20–460] ms. It can be noted
that, at the beginning, saliency is more focused on local
and global visual features, and later focused on context
and ultimately on objects of interest; with the last column
integrating all contributions in one saliency map.


_7.6.1_ _Global analysis of the cortical-visual representations_


In this experiment we aim to identify high-level correlations
between EEG channels and visual content, by applying
Eq. 5, which assesses how average compatibility changes
when each EEG channel is suppressed. Fig. 9 shows some
examples of the mean activation maps per object class. These
were obtained by averaging channel importance scores over
all images for each class. To show the relationship between
the temporal and spatial activation of EEG, Fig. 10 shows
the average activation map over all classes, by evaluating
channel importance when restricting the EEG signal to
specific time intervals.
From these results, some interesting conclusions can be
drawn: 1) All visual classes rely heavily on early visual areas
including V1 cortex — known to be responsible for early
visual processing [26] — and this region is important in all
tested time windows; 2) The average activation maps over
time clearly show that the process starts in early visual areas
and then flows to the frontal regions (responsible of higher
cognitive functions) and temporal regions (responsible for
visual categorization [10]); 3) The pattern of activation



13


Fig. 8: **Examples of our brain-derived saliency detection.**
In all cases, the ImageNet class (from top to bottom: “mobile
phone”, “mug”, “banana” and “pizza”) is different from
objects receiving more attention by the human observers.
We report the saliency in the same time ranges of Fig. 6.
Please note that all the four images were correctly classified
by the employed visual encoder, i.e, Inception-v3.


changes with the visual content; e.g., the “piano” or the
“electric guitar” visual class, activates scalp regions closer to
auditory cortex (left-most and right-most areas of the scalp),
and this is in line with evidence that the sensation of sounds
is often associated with sight [70].


_7.6.2_ _Extracting neural representations from the cortical-_
_visual data over time_


The goal of this final analysis is to probe the various
DCNN layers and relate them to the joint EEG/visual data,
over time and over scalp location, to examine the lowand middle-level visual representations responsible for the
given neural activation. To accomplish this task, we employ
the learned compatibility measure to find mutual correspondence between the deep features and the scalp region
generating the activity. Using the _association_ score defined
in Sect. 6 (Eq. 12), we investigate the neural encoding of
the visual information by deriving neural activation maps
that maximally respond to the deep-learned visual features.
Fig. 11 shows the activation maps of the association scores
related to specific layers of our best-performing image encoder as per Tab. 3. This analysis employs a pre-trained
Inception network fine-tuned on our brain/image dataset
during encoder training. To show the complexity of the
features learned at each level, we show a few examples
obtained by performing activation maximization [71] on a
subset of features for each layer. For each feature/neural
association, we also measure the relative contribution to
brain activity by different temporal portions of the EEG, by
feeding each interval to the EEG encoder (as described in the
previous section) when applying Eq. 12. In this case, unlike
the representations in Fig. 10, we are not interested in the
differences in activation between cortical regions. Therefore,


14



Anemone Panda Cellular phone Electric guitar Piano Airliner Locomotive


Fig. 9: **Activation maps per visual class** . Average activation maps for some of the 40 visual classes in the dataset.



Average activation
map



0-80 ms 80-160 ms 160-320 ms 320-440 ms



Fig. 10: **Average activation maps** . (Left image). Average activation map across all image classes. (Right images). Average
activation in different time ranges.



we compute the average unnormalized association scores
over all channels, and use that as the measure of how
associated each layer’s features are with each portion of the
EEG activation. By using all of this information, we are able
to probe the underlying neural representations, the spatial
location on the scalp that relate to the representation and
their timing. The results suggest that hierarchical representations in DCNNs tightly correlate with the hierarchical processing stages in the human visual pathway. In particular,
at the lowest layer, simple texture and color features are
generated and they correspond with early visual areas near
V1. Moving to deeper layers in the DCNN, we see that the
activation propagates from early visual areas to temporal
regions and then back to the early visual regions. Moreover,
more complex features (at higher layers) are influenced by
the activity occurring later in time. Whereas early visual
areas, known to encode basic visual features, correspond to
the early DCNN layers, which also encode simple visual
features, later layers, which produce more complex classlevel representations, seem to correspond to later EEG time
windows. The timing of the EEG activity and the associated
DCNN layers are in line with the known hierarchical object
processing stream in the cognitive neuroscience literature.
This consistency suggests that we have produced reliable
approximations of human brain representations. It is interesting to note that we observe a consistent drop in the relationship between the joint EEG activation and DCNN layers
in the 100-200 ms time window. Importantly, the end of this
time window corresponds to the well-established transition
from (primarily) perceptual processing to (primarily) higher
order, cognitive and recurrent processing [64]. This suggests
a logical relationship to known human neural processing.
Alternatively, this could originate from the relocation of
visual cognitive processes to deeper cortical areas that are
less detectable via EEG, followed by feedback activity to the
initial regions in the visual pathway. Clearly, future work



will need to explore these interesting possibilities further,
since a comprehensive neurological interpretation is outside
of the scope of this paper.


**8** **C** **ONCLUSION**


In this work, we present a multi-modal approach to learn
a joint feature space for images and EEG signals recorded
while users look at pictures on a screen. We trained two
encoders in a siamese configuration and maximize the _com-_
_patibility_ score between the corresponding images and EEGs.
The learned embeddings make the representation useful to
perform several computer vision tasks, supervised by brain
activity. Our experiments show the neural activity can be reliably used to drive the development of image classification
and saliency detection methods. In addition to advancing
the work related to brain-guided image classification [3], our
approach provides a way to extract neural representation
from EEG data and to map it to the most important/salient
visual features.
While drawing general cognitive neuroscience conclusions
from these findings is not the main goal of this work, given
also the small scale of the cognitive experiment, we propose
an AI-based strategy that seems to produce reliable approximations of brain representations and their corresponding
scalp activity, by jointly learning a model that maximizes
the correlation between neural activity and visual images.
The natural extension of this work in the future is to further
investigate these associations, with the objective of finding
a finer correspondence between EEG signals and visual
patterns — e.g., by identifying different responses in the
brain activity corresponding to specific objects, patterns,
or categories of varying specificity. We believe that a joint
research effort combining artificial intelligence (through the
development of more sophisticated methods) and neuroscience (through more tailored and large scale experiments)
is necessary to advance both fields, by studying how brain


15


Image encoder, layer 3/20


Image encoder, layer 12/20


Image encoder, layer 20/20


Fig. 11: **Brain activity associated with specific visual representations extracted from the DCNN layers** . Each row shows
a set of feature maps (manually picked for interpretability and visualized through activation maximization) from a specific
layer in the image encoder, the neural activity areas with the highest association to the layer’s features, and the contribution
that different time ranges in the EEG signal give rise to association scores. It can be noted that, as feature complexity
increases, the activated brain regions move from the V1 visual cortex (occipital region) to the IT cortex (temporal region);
moreover, the initial temporal portions of EEG signals seem to be more related to simpler features, while there is a stronger
association between more complex features and later temporal dynamics.



processes relate to artificial model structures and, in turn,
using the uncovered neural dynamics to propose novel
neural architectures to make computational models more
closely approximate human perceptual and cognitive performance.


**A** **CKNOWLEDGMENTS**


The authors would like to thank Dr. Martina Platania for
supporting the data acquisition phase, Dr. Demian Faraci
for the experimental results, and NVIDIA for the generous
donation of two Titan X GPUs.


**R** **EFERENCES**


[1] T. Horikawa and Y. Kamitani, “Generic decoding of seen and
imagined objects using hierarchical visual features,” _Nat Commun_,
vol. 8, p. 15037, May 2017.




[2] R. M. Cichy, A. Khosla, D. Pantazis, A. Torralba, and A. Oliva,
“Comparison of deep neural networks to spatio-temporal cortical
dynamics of human visual object recognition reveals hierarchical
correspondence,” _Sci Rep_, vol. 6, p. 27755, 06 2016.

[3] C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly,
and M. Shah, “Deep Learning Human Mind for Automated Visual
Classification,” in _CVPR_, jul 2017, pp. 4503–4511.

[4] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, and M. Shah,
“Generative adversarial networks conditioned by brain signals,”
in _ICCV_, Oct 2017, pp. 3430–3438.

[5] S. Nishimoto, A. T. Vu, T. Naselaris, Y. Benjamini, B. Yu, and J. L.
Gallant, “Reconstructing visual experiences from brain activity
evoked by natural movies,” _Curr. Biol._, vol. 21, no. 19, pp. 1641–
1646, Oct 2011.

[6] D. E. Stansbury, T. Naselaris, and J. L. Gallant, “Natural scene
statistics account for the representation of scene categories in
human visual cortex,” _Neuron_, vol. 79, no. 5, pp. 1025–1034, Sep
2013.

[7] J. Bullier, “Integrated model of visual processing,” _Brain Res. Brain_
_Res. Rev._, vol. 36, no. 2-3, pp. 96–107, Oct 2001.

[8] Z. Kourtzi and C. E. Connor, “Neural representations for object
perception: structure, category, and adaptive coding,” _Annu. Rev._
_Neurosci._, vol. 34, pp. 45–67, 2011.

[9] D. J. Kravitz, K. S. Saleem, C. I. Baker, and M. Mishkin, “A new


neural framework for visuospatial processing,” _Nat. Rev. Neurosci._,
vol. 12, no. 4, pp. 217–230, Apr 2011.

[10] J. J. DiCarlo, D. Zoccolan, and N. C. Rust, “How does the brain
solve visual object recognition?” _Neuron_, vol. 73, no. 3, pp. 415–
434, Feb 2012.

[11] H. Wen, K. Han, J. Shi, Y. Zhang, E. Culurciello, and Z. Liu,
“Deep predictive coding network for object recognition,” in _35th_
_International Conference on Machine Learning_, ser. Proceedings of
Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80.
Stockholmsmssan, Stockholm Sweden: PMLR, 10–15 Jul 2018, pp.
5266–5275.

[12] A. Clark, “Whatever next? Predictive brains, situated agents, and
the future of cognitive science,” _Behav Brain Sci_, vol. 36, no. 3, pp.
181–204, Jun 2013.

[13] A. M. Bastos, W. M. Usrey, R. A. Adams, G. R. Mangun, P. Fries,
and K. J. Friston, “Canonical microcircuits for predictive coding,”
_Neuron_, vol. 76, no. 4, pp. 695–711, Nov 2012.

[14] Y. Roy, H. J. Banville, I. Albuquerque, A. Gramfort, T. H. Falk, and
J. Faubert, “Deep learning-based electroencephalography analysis:
a systematic review,” _CoRR_, vol. abs/1901.05498, 2019.

[15] D. Zhang, L. Yao, X. Zhang, S. Wang, W. Chen, R. Boots, and B. Benatallah, “Cascade and parallel convolutional recurrent neural
networks on eeg-based intention recognition for brain computer
interface,” 2018.

[16] V. A. N. P. A. M. K.R.Rao, “Cognitive analysis of working memory
load from eeg, by a deep recurrent neural network,” 2018.

[17] P. Yan, F. Wang, and Z. Grinspan, “: Spectrographic seizure detection using deep learning with convolutional neural networks
(s19.004),” _Neurology_, vol. 90, no. 15 Supplement, 2018.

[18] L. Vidyaratne, A. Glandon, M. Alam, and K. M. Iftekharuddin,
“Deep recurrent neural network for seizure detection,” in _2016_
_International Joint Conference on Neural Networks (IJCNN)_, 2016, pp.
1202–1207.

[19] T. Zhang, W. Zheng, Z. Cui, Y. Zong, and Y. Li, “Spatialtemporal
recurrent neural network for emotion recognition,” _IEEE Transac-_
_tions on Cybernetics_, vol. 49, no. 3, pp. 839–847, 2019.

[20] Z. Tang, C. Li, and S. Sun, “Single-trial eeg classification of motor
imagery using deep convolutional neural networks,” _Optik_, vol.
130, pp. 11 – 18, 2017.

[21] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon,
C. P. Hung, and B. J. Lance, “EEGNet: a compact convolutional
neural network for EEG-based brain–computer interfaces,” _Journal_
_of Neural Engineering_, vol. 15, no. 5, p. 056013, jul 2018.

[22] Y. Li, m. Murias, s. Major, g. Dawson, K. Dzirasa, L. Carin,
and D. E. Carlson, “Targeting eeg/lfp synchrony with neural
nets,” in _Advances in Neural Information Processing Systems 30_,
I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, Eds. Curran Associates, Inc.,
[2017, pp. 4620–4630. [Online]. Available: http://papers.nips.cc/](http://papers.nips.cc/paper/7048-targeting-eeglfp-synchrony-with-neural-nets.pdf)
[paper/7048-targeting-eeglfp-synchrony-with-neural-nets.pdf](http://papers.nips.cc/paper/7048-targeting-eeglfp-synchrony-with-neural-nets.pdf)

[23] K. J. Seymour, M. A. Williams, and A. N. Rich, “The Representation of Color across the Human Visual Cortex: Distinguishing
Chromatic Signals Contributing to Object Form Versus Surface
Color,” _Cereb. Cortex_, vol. 26, no. 5, pp. 1997–2005, May 2016.

[24] J. W. Peirce, “Understanding mid-level representations in visual
processing,” _J Vis_, vol. 15, no. 7, p. 5, 2015.

[25] C. P. Hung, G. Kreiman, T. Poggio, and J. J. DiCarlo, “Fast readout
of object identity from macaque inferior temporal cortex,” _Science_,
vol. 310, no. 5749, pp. 863–866, Nov 2005.

[26] A. K. Robinson, P. Venkatesh, M. J. Boring, M. J. Tarr, P. Grover, and
M. Behrmann, “Very high density EEG elucidates spatiotemporal
aspects of early visual processing,” _Sci Rep_, vol. 7, no. 1, p. 16248,
Nov 2017.

[27] D. L. Yamins, H. Hong, C. Cadieu, and J. J. DiCarlo, “Hierarchical
modular optimization of convolutional networks achieves representations similar to macaque it and human ventral stream,” in
_Advances in Neural Information Processing Systems 26_, C. J. C. Burges,
L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, Eds.
Curran Associates, Inc., 2013, pp. 3093–3101.

[28] D. L. Yamins, H. Hong, C. F. Cadieu, E. A. Solomon, D. Seibert, and
J. J. DiCarlo, “Performance-optimized hierarchical models predict
neural responses in higher visual cortex,” _Proc. Natl. Acad. Sci._
_U.S.A._, vol. 111, no. 23, pp. 8619–8624, Jun 2014.

[29] N. Kriegeskorte, M. Mur, and P. Bandettini, “Representational
similarity analysis - connecting the branches of systems neuroscience,” _Front Syst Neurosci_, vol. 2, p. 4, 2008.



16


[30] A. Graves, G. Wayne, M. Reynolds, T. Harley, I. Danihelka,
A. Grabska-Barwi?ska, S. G. Colmenarejo, E. Grefenstette, T. Ramalho, J. Agapiou, A. P. Badia, K. M. Hermann, Y. Zwols,
G. Ostrovski, A. Cain, H. King, C. Summerfield, P. Blunsom,
K. Kavukcuoglu, and D. Hassabis, “Hybrid computing using a
neural network with dynamic external memory,” _Nature_, vol. 538,
no. 7626, pp. 471–476, 10 2016.

[31] K. Gregor, I. Danihelka, A. Graves, D. Rezende, and D. Wierstra,
“Draw: A recurrent neural network for image generation,” in _32nd_
_International Conference on Machine Learning_, ser. Proceedings of
Machine Learning Research, F. Bach and D. Blei, Eds., vol. 37.
Lille, France: PMLR, 07–09 Jul 2015, pp. 1462–1471.

[32] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov,
R. Zemel, and Y. Bengio, “Show, attend and tell: Neural image
caption generation with visual attention,” in _32nd International_
_Conference on Machine Learning_, ser. Proceedings of Machine Learning Research, F. Bach and D. Blei, Eds., vol. 37. Lille, France:
PMLR, 07–09 Jul 2015, pp. 2048–2057.

[33] B. M. Lake, T. D. Ullman, J. B. Tenenbaum, and S. J. Gershman,
“Building machines that learn and think like people,” _Behav Brain_
_Sci_, vol. 40, p. e253, Jan 2017.

[34] R. C. Fong, W. J. Scheirer, and D. D. Cox, “Using human brain
activity to guide machine learning,” _Sci Rep_, vol. 8, no. 1, p. 5397,
Mar 2018.

[35] A. Kapoor, P. Shenoy, and D. Tan, “Combining brain computer
interfaces with vision for object categorization,” in _2008 CVPR_,
June 2008, pp. 1–8.

[36] W. J. Scheirer, S. E. Anthony, K. Nakayama, and D. D. Cox,
“Perceptual Annotation: Measuring Human Vision to Improve
Computer Vision,” _IEEE Trans Pattern Anal Mach Intell_, vol. 36,
no. 8, pp. 1679–1686, Aug 2014.

[37] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng,
“Multimodal deep learning.” in _ICML_, L. Getoor and T. Scheffer,
Eds. Omnipress, 2011, pp. 689–696.

[38] K. Sohn, W. Shang, and H. Lee, “Improved multimodal deep
learning with variation of information,” in _Advances in Neural_
_Information Processing Systems 27_, Z. Ghahramani, M. Welling,
C. Cortes, N. D. Lawrence, and K. Q. Weinberger, Eds. Curran
Associates, Inc., 2014, pp. 2141–2149.

[39] N. Srivastava and R. Salakhutdinov, “Multimodal learning with
deep boltzmann machines,” _Journal of Machine Learning Research_,
vol. 15, pp. 2949–2980, 2014.

[40] S. Venugopalan, L. Anne Hendricks, M. Rohrbach, R. Mooney,
T. Darrell, and K. Saenko, “Captioning images with diverse objects,” in _The CVPR (CVPR)_, July 2017.

[41] I. Ilievski and J. Feng, “Multimodal learning and reasoning for
visual question answering,” in _Advances in Neural Information_
_Processing Systems 30_, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. Curran
Associates, Inc., 2017, pp. 551–562.

[42] M. Guillaumin, J. Verbeek, and C. Schmid, “Multimodal semisupervised learning for image classification,” in _2010 IEEE Com-_
_puter Society Conference on Computer Vision and Pattern Recognition_,
June 2010, pp. 902–909.

[43] H. Zhao, C. Gan, A. Rouditchenko, C. Vondrick, J. McDermott, and
A. Torralba, “The sound of pixels,” _arXiv preprint arXiv:1804.03160_,
2018.

[44] A. Owens, J. Wu, J. H. McDermott, W. T. Freeman, and A. Torralba,
“Ambient sound provides supervision for visual learning,” in
_Computer Vision – ECCV 2016_, B. Leibe, J. Matas, N. Sebe, and
M. Welling, Eds. Cham: Springer International Publishing, 2016,
pp. 801–816.

[45] Y. Aytar, C. Vondrick, and A. Torralba, “Soundnet: Learning sound
representations from unlabeled video,” in _30th International Confer-_
_ence on Neural Information Processing Systems_, ser. NIPS’16. USA:
Curran Associates Inc., 2016, pp. 892–900.

[46] S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen,
R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold,
M. Slaney, R. J. Weiss, and K. Wilson, “Cnn architectures for largescale audio classification,” in _2017 IEEE International Conference on_
_Acoustics, Speech and Signal Processing (ICASSP)_, March 2017, pp.
131–135.

[47] M. J. Huiskes, B. Thomee, and M. S. Lew, “New trends and ideas
in visual concept detection: The mir flickr retrieval evaluation
initiative,” in _International Conference on Multimedia Information_
_Retrieval_, ser. MIR ’10. New York, NY, USA: ACM, 2010, pp.
527–536.


[48] R. Arandjelovic and A. Zisserman, “Look, listen and learn,” in _The_
_IEEE International Conference on Computer Vision (ICCV)_, Oct 2017.

[49] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and tell:
A neural image caption generator,” in _Computer Vision and Pattern_
_Recognition_, 2015.

[50] A. Karpathy and L. Fei-Fei, “Deep visual-semantic alignments for
generating image descriptions,” _IEEE Trans. Pattern Anal. Mach._
_Intell._, vol. 39, no. 4, pp. 664–676, 2017.

[51] J. Donahue, L. A. Hendricks, S. Guadarrama, M. Rohrbach,
S. Venugopalan, T. Darrell, and K. Saenko, “Long-term recurrent
convolutional networks for visual recognition and description.” in
_CVPR_ . IEEE Computer Society, 2015, pp. 2625–2634.

[52] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee,
“Generative adversarial text to image synthesis,” in _33rd Interna-_
_tional Conference on Machine Learning_, ser. Proceedings of Machine
Learning Research, M. F. Balcan and K. Q. Weinberger, Eds.,
vol. 48. New York, New York, USA: PMLR, 20–22 Jun 2016, pp.
1060–1069.

[53] E. Mansimov, E. Parisotto, L. J. Ba, and R. Salakhutdinov, “Generating images from captions with attention,” _ICLR2016_, vol.
abs/1511.02793, 2016.

[54] I. Kavasidis, S. Palazzo, C. Spampinato, D. Giordano, and M. Shah,
“Brain2image: Converting brain signals into images,” in _ACM MM_
_’17_, 2017, pp. 1809–1817.

[55] F. Yu and V. Koltun, “Multi-scale context aggregation by dilated
convolutions,” _ICLR2016_, vol. abs/1511.07122, 2015.

[56] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for
Image Recognition,” in _CVPR (CVPR)_, Las Vegas, USA, 2016.

[57] S. Treue, “Visual attention: the where, what, how and why of
saliency,” _Current Opinion in Neurobiology_, vol. 13, no. 4, pp. 428
– 432, 2003.

[58] M. D. Zeiler and R. Fergus, “Visualizing and understanding
convolutional networks,” in _European conference on computer vision_ .
Springer, 2014, pp. 818–833.

[59] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” in _CVPR09_,
2009.

[60] R. Oostenveld and P. Praamstra, “The five percent electrode
system for high-resolution EEG and ERP measurements,” _Clin_
_Neurophysiol_, vol. 112, no. 4, pp. 713–719, Apr 2001.

[61] M. S. Clayton, N. Yeung, and R. C. Kadosh, “The roles of cortical
oscillations in sustained attention,” _Trends in Cognitive Sciences_,
vol. 19, no. 4, pp. 188 – 195, 2015.

[62] C. Tallon-Baudry and O. Bertrand, “Oscillatory gamma activity in
humans and its role in object representation,” _Trends in Cognitive_
_Sciences_, vol. 3, no. 4, pp. 151 – 162, 1999.

[63] O. Jensen, J. Kaiser, and J.-P. Lachaux, “Human gamma-frequency
oscillations associated with attention and memory,” _Trends in_
_Neurosciences_, vol. 30, no. 7, pp. 317 – 324, 2007.

[64] S. J. Luck, _An introduction to the event-related potential technique_ .
MIT press, 2014.

[65] X. Huang, C. Shen, X. Boix, and Q. Zhao, “Salicon: Reducing
the semantic gap in saliency prediction by adapting deep neural
networks,” in _ICCV 2015_, 2015, pp. 262–270.

[66] J. Pan, E. Sayrol, X. Giro-i Nieto, K. McGuinness, and N. E.
O’Connor, “Shallow and deep convolutional networks for saliency
prediction,” in _CVPR 2016_, 2016.

[67] A. Borji, D. N. Sihite, and L. Itti, “Quantitative analysis of humanmodel agreement in visual saliency modeling: A comparative
study,” _TIP 2013_, 2013.

[68] L. Itti and C. Koch, “A saliency-based search mechanism for overt
and covert shifts of visual attention,” _Vision Research_, vol. 40,
no. 10, pp. 1489 – 1506, 2000.

[69] A. Oliva and A. Torralba, “The role of context in object recognition,” _Trends in Cognitive Sciences_, vol. 11, no. 12, pp. 520 – 527,
2007.

[70] A. M. Proverbio, G. E. D’Aniello, R. Adorni, and A. Zani, “When
a photograph can be heard: vision activates the auditory cortex
within 110 ms,” _Sci Rep_, vol. 1, p. 54, 2011.

[71] C. Olah, A. Mordvintsev, and L. Schubert, “Feature visualization,”
_Distill_, vol. 2, no. 11, p. e7, 2017.



17


**Simone Palazzo** is an assistant professor at the
University of Catania, Italy. His research interests are in the areas of deep learning, computer
vision, integration of human feedback into AI
systems, medical image analysis. He has been
part of the program committees of several workshops and conferences on computer vision. He
has co-authored over 50 papers in international
refereed journals and conference proceedings.


**Concetto Spampinato** is an assistant professor
at the University of Catania, Italy. He is also
Courtesy Faculty member of the Center for Research in Computer Vision at the Unviersity of
Central Florida (USA). His research interests
lie mainly in the artificial intelligence, computer
vision and pattern recognition research fields
with a particular focus on human-based and
brain-driven computation systems. He has coauthored over 150 publications in international
refereed journals and conference proceedings.


**Isaak Kavasidis** is an assistant researcher at
the University of Catania in Italy. His research
interests include the areas of medical data processing and brain data processing using machine and deep learning methods and the decoding of human brain functions and transfer
to computerized methods. In 2014, he participated in the Marie Curie RELATE ITN project as
an experienced researcher. He has co-authored
more than 40 scientific papers in peer-reviewed
international conferences and journals


**Daniela Giordano** is an associate professor
at the University of Catania, Italy. She also
holds the Ph.D. degree in Educational Technology from Concordia University, Montreal (1998).
Her main research interests include advanced
learning technology, knowledge discovery, and
information technology in medicine. She has coauthored over 200 publications in international
refereed journals and conference proceedings.


**Joseph Schmidt** is an assistant professor at
the University of Central Florida, Department of
Psychology and he currently oversees the Attention and Memory Lab. Dr. Schmidts research
investigates how memory representations of target objects affect deployments of attention. He
has developed expertise in many psychophysiological techniques including eye tracking and
electroencephalogram (EEG)/event-related potentials (ERPs). He has co-authored about many
publications in international refereed journals.


**Mubarak Shah** is the trustee chair professor of
computer science and the founding director of
the Center for Research in Computer Vision at
University of Central Florida. His research interests include video surveillance, visual tracking, human activity recognition, visual analysis of
crowded scenes, video registration, UAV video
analysis, and so on. He is a fellow of the IEEE,
AAAS, IAPR, and SPIE.


