## **Domain Separation Networks**



**Konstantinos Bousmalis** _[∗]_
Google Brain
Mountain View, CA
```
konstantinos@google.com

```


**George Trigeorgis** _[∗†]_
Imperial College London
London, UK
```
g.trigeorgis@imperial.ac.uk

```


**Nathan Silberman**
Google Research
New York, NY
```
nsilberman@google.com

```


**Dilip Krishnan**
Google Research
Cambridge, MA
```
dilipkay@google.com

```


**Dumitru Erhan**
Google Brain
Mountain View, CA
```
dumitru@google.com

```


**Abstract**

The cost of large scale data collection and annotation often makes the application
of machine learning algorithms to new tasks or datasets prohibitively expensive.
One approach circumventing this cost is training models on synthetic data where
annotations are provided automatically. Despite their appeal, such models often
fail to generalize from synthetic to real images, necessitating domain adaptation
algorithms to manipulate these models before they can be successfully applied. Existing approaches focus either on mapping representations from one domain to the
other, or on learning to extract features that are invariant to the domain from which
they were extracted. However, by focusing only on creating a mapping or shared
representation between the two domains, they ignore the individual characteristics
of each domain. We suggest that explicitly modeling what is unique to each domain
can improve a model’s ability to extract domain–invariant features. Inspired by
work on private–shared component analysis, we explicitly learn to extract image
representations that are partitioned into two subspaces: one component which is
private to each domain and one which is shared across domains. Our model is
trained not only to perform the task we care about in the source domain, but also
to use the partitioned representation to reconstruct the images from both domains.
Our novel architecture results in a model that outperforms the state–of–the–art on
a range of unsupervised domain adaptation scenarios and additionally produces
visualizations of the private and shared representations enabling interpretation of
the domain adaptation process.


**1** **Introduction**

The recent success of supervised learning algorithms has been partially attributed to the large-scale
datasets [ 17, 23 ] on which they are trained. Unfortunately, collecting, annotating, and curating such
datasets is an extremely expensive and time-consuming process. An alternative would be creating
large-scale datasets in non–realistic but inexpensive settings, such as computer generated scenes.
While such approaches offer the promise of effectively unlimited amounts of labeled data, models
trained in such settings do not generalize well to realistic domains. Motivated by this, we examine the
problem of learning representations that are domain–invariant in scenarios where the data distributions
during training and testing are different. In this setting, the source data is labeled for a particular task
and we would like to transfer knowledge from the source to the target domain for which we have no
ground truth labels.


In this work, we focus on the tasks of object classification and pose estimation, where the object of
interest is in the foreground of a given image, for both source and target domains. The source and


_∗_ Authors contributed equally.

_†_ This work was completed while George Trigeorgis was at Google Brain in Mountain View, CA.


29th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain.


target pixel distributions can differ in a number of ways. We define “low-level” differences in the
distributions as those arising due to noise, resolution, illumination and color. “High-level” differences
relate to the number of classes, the types of objects, and geometric variations, such as 3D position
and pose. We assume that our source and target domains differ mainly in terms of the distribution of
low level image statistics and that they have high level parameters with similar distributions and the
same label space.


We propose a novel method, the Domain Separation Networks (DSN), for learning domain–invariant
representations. Previous work attempts to either find a mapping from representations of the source
domain to those of the target [ 27 ], or find representations that are shared between the two domains [ 8,
29, 18 ]. While this, in principle, is a good idea, it leaves the shared representations vulnerable to
contamination by noise that is correlated with the underlying shared distribution [ 25 ]. Our model, in
contrast, introduces the notion of a private subspace for each domain, which captures domain specific
properties, such as background and low level image statistics. A shared subspace, enforced through
the use of autoencoders and explicit loss functions, captures representations shared by the domains.
By finding a shared subspace that is orthogonal to the subspaces that are private, our model is able to
separate the information that is unique to each domain, and in the process produce representations
that are more more meaningful for the task at hand. Our method outperforms the state–of–the–art
domain adaptation techniques on a range of datasets for object classification and pose estimation,
while having an interpretability advantage by allowing the visualization of these private and shared
representations. In Section 2, we survey related work and introduce relevant terminology. Our
architecture, loss functions and learning regime are presented in Section 3. Experimental results
and discussion are given in Section 4. Finally, conclusions and directions for future work are in
Section 5.

**2** **Related Work**

Learning to perform unsupervised domain adaptation is an open theoretical and practical problem.
While much prior art exists, our literature review focuses primarily on Convolutional Neural Network
(CNN) based methods due to their empirical superiority on this problem [ 8, 18, 27, 30 ]. Ben-David
et al. [ 4 ] provide upper bounds on a domain-adapted classifier in the target domain. They introduce
the idea of training a binary classifier trained to distinguish source and target domains. The error
that this “domain incoherence” classifier provides (along with the error of a source domain specific
classifier) combine to give the overall bounds. Mansour et al. [ 19 ] extend the theory of [ 4 ] to handle
the case of multiple source domains.


Ganin et al. [ 7, 8 ] and Ajakan et al. [ 2 ] use adversarial training to find domain–invariant representations in-network. Their Domain–Adversarial Neural Networks (DANN) exhibit an architecture
whose first few feature extraction layers are shared by two classifiers trained simultaneously. The first
is trained to correctly predict task-specific class labels on the source data while the second is trained
to predict the domain of each input. DANN minimizes the domain classification loss with respect
to parameters specific to the domain classifier, while maximizing it with respect to the parameters
that are common to both classifiers. This minimax optimization becomes possible via the use of a
gradient reversal layer (GRL).


Tzeng et al. [ 30 ] and Long et al. [ 18 ] proposed versions of this model where the maximization of
the domain classification loss is replaced by the minimization of the Maximum Mean Discrepancy
(MMD) metric [ 11 ]. The MMD metric is computed between features extracted from sets of samples
from each domain. The Deep Domain Confusion Network by Tzeng et al. [ 30 ] has an MMD loss at
one layer in the CNN architecture while Long et al. [ 18 ] proposed the Deep Adaptation Network
that has MMD losses at multiple layers.


Other related techniques involve learning a transformation from one domain to the other. In this setup,
the feature extraction pipeline is fixed during the domain adaptation optimization. This has been
applied in various non-CNN based approaches [ 9, 5, 10 ] as well as the recent CNN-based Correlation
Alignment (CORAL) [ 27 ] algorithm which “recolors” whitened source features with the covariance
of features from the target domain.


**3** **Method**

While the Domain Separation Networks (DSNs) could in principle be applicable to other learning
tasks, without loss of generalization, we mainly use image classification as the cross-domain task.
Given a labeled dataset in a source domain and an unlabeled dataset in a target domain, our goal is to


2


Figure 1: Training of our Domain Separation Networks. A shared-weight encoder _E_ _c_ ( **x** ) learns
to capture representation components for a given input sample that are shared among domains. A
private encoder _E_ _p_ ( **x** ) (one for each domain) learns to capture domain–specific components of the
representation. A shared decoder learns to reconstruct the input sample by using both the private
and source representations. The private and shared representation components are pushed apart with
soft subspace orthogonality constraints _L_ difference, whereas the shared representation components are
kept similar with a similarity loss _L_ similarity . See text for more information.


train a classifier on data from the source domain that generalizes to the target domain. Like previous
efforts [ 7, 8 ], our model is trained such that the representations of images from the source domain are
similar to those from the target domain. This allows a classifier trained on images from the source
domain to generalize as the inputs to the classifier are in theory invariant to the domain of origin.
However, these representations might trivially include noise that is highly correlated with the shared
representation, as shown by Salzmann et al. [25].


Our main novelty is that, inspired by recent work [ 15, 25, 31 ] on shared–space component analysis,
DSNs explicitly and jointly model both private and shared components of the domain representations.
The private component of the representation is specific to a single domain and the shared component
of the representation is shared by both domains. To induce the model to produce such split representations, we add a loss function that encourages independence of these parts. Finally, to ensure
that the private representations are still useful (avoiding trivial solutions) and to add generalizability,
we also add a reconstruction loss. The combination of these objectives is a model that produces a
shared representation that is similar for both domains and a private representation that is different. By
partitioning the space in such a manner, the classifier trained on the shared representation is better
able to generalize across domains as its inputs are uncontaminated with aspects of the representation
that are unique to each domain.


More specifically, let **X** _S_ = _{_ ( **x** _[s]_ _i_ _[,]_ **[ y]** _[s]_ _i_ [)] _[}]_ _[N]_ _i_ =0 _[s]_ [represent a labeled dataset of] _[ N]_ _[s]_ [ samples from the source]
domain where **x** _[s]_ _i_ _[∼D]_ _[S]_ [ and let] **[ X]** _[t]_ [ =] _[ {]_ **[x]** _[t]_ _i_ _[}]_ _[N]_ _i_ =0 _[t]_ [represent an unlabeled dataset of] _[ N]_ _[t]_ [ samples from]
the target domain where **x** _[t]_ _i_ _[∼D]_ _[T]_ [ . Let] _[ E]_ _[c]_ [(] **[x]** [;] _**[ θ]**_ _[c]_ [)] [ be a function parameterized by] _**[ θ]**_ _[c]_ [ which maps]
an image **x** to a hidden representation **h** _c_ representing features that are common or _shared_ across
domains. Let _E_ _p_ ( **x** ; _**θ**_ _p_ ) be an analogous function which maps an image **x** to a hidden representation
**h** _p_ representing features that are _private_ to each domain. Let _D_ ( **h** ; _**θ**_ _d_ ) be a decoding function
mapping a hidden representation **h** to an image reconstruction ˆ **x** . Finally, _G_ ( **h** ; _**θ**_ _g_ ) represents a taskspecific function, parameterized by _**θ**_ _g_ that maps from hidden representations **h** to the task-specific
predictions ˆ **y** . The resulting Domain Separation Network (DSN) model is depicted in Figure 1.


**3.1** **Learning**


Inference in a DSN model is given by ˆ **x** = _D_ ( _E_ _c_ ( **x** ) + _E_ _p_ ( **x** )) and ˆ **y** = _G_ ( _E_ _c_ ( **x** )) where ˆ **x** is the
reconstruction of the input **x** and ˆ **y** is the task-specific prediction. The goal of training is to minimize


3


the following loss with respect to parameters **Θ** = _{_ _**θ**_ _c_ _,_ _**θ**_ _p_ _,_ _**θ**_ _d_ _,_ _**θ**_ _g_ _}_ :


_L_ = _L_ task + _α L_ recon + _β L_ difference + _γ L_ similarity (1)


where _α, β, γ_ are weights that control the interaction of the loss terms. The classification loss _L_ task
trains the model to predict the output labels we are ultimately interested in. Because we assume the
target domain is unlabeled, the loss is applied only to the source domain. We want to minimize the
negative log–likelihood of the ground truth class for each source domain sample:



_L_ task = _−_



_N_ _s_
� **y** _[s]_ _i_ _[·]_ [ log ˆ] **[y]** _[s]_ _i_ _[,]_ (2)


_i_ =0



where **y** _[s]_ _i_ [is the one–hot encoding of the class label for source input] _[ i]_ [ and] [ ˆ] **[y]** _i_ _[s]_ [are the softmax]
predictions of the model: ˆ **y** _[s]_ _i_ [=] _[ G]_ [(] _[E]_ _[c]_ [(] **[x]** _[s]_ _i_ [))] [. We use a scale–invariant mean squared error term [] [6] []]
for the reconstruction loss _L_ recon which is applied to both domains:



_N_ _s_

_L_ recon = � _L_ si_mse ( **x** _[s]_ _i_ _[,]_ [ ˆ] **[x]** _[s]_ _i_ [) +]


_i_ =1



_N_ _t_
� _L_ si_mse ( **x** _[t]_ _i_ _[,]_ [ ˆ] **[x]** _[t]_ _i_ [)] (3)


_i_ =1



_L_ si_mse ( **x** _,_ ˆ **x** ) = [1]




[1] 2 _[−]_ [1]

_k_ _[∥]_ **[x]** _[ −]_ **[x]** [ˆ] _[∥]_ [2] _k_



_k_ [2] [([] **[x]** _[ −]_ **[x]** [ˆ][]] _[ ·]_ **[ 1]** _[k]_ [)] [2] _[,]_ (4)



where _k_ is the number of pixels in input _x_, **1** _k_ is a vector of ones of length _k_ ; and _∥· ∥_ 2 [2] [is the squared]
_L_ 2 -norm. While a mean squared error loss is traditionally used for reconstruction tasks, it penalizes
predictions that are correct up to a scaling term. Conversely, the scale-invariant mean squared error
penalizes differences between _pairs_ of pixels. This allows the model to learn to reproduce the overall
shape of the objects being modeled without expending modeling power on the absolute color or
intensity of the inputs. We validated that this reconstruction loss was indeed the correct choice
experimentally in Section 4.3 by training a version of our best DSN model with the traditional mean
squared error loss instead of the one in Equation 3.


The difference loss is also applied to both domains and encourages the shared and private encoders to
encode different aspects of the inputs. We define the loss via a soft subspace orthogonality constraint
between the private and shared representation of each domain. Let **H** _[s]_ _c_ [and] **[ H]** _[t]_ _c_ [be matrices whose]
rows are the hidden _shared_ representations **h** _[s]_ _c_ [=] _[ E]_ _[c]_ [(] **[x]** _[s]_ [)] [ and] **[ h]** _[t]_ _c_ [=] _[ E]_ _[c]_ [(] **[x]** _[t]_ [)] [ from samples of source]
and target data respectively. Similarly, let **H** _[s]_ _p_ [and] **[ H]** _[t]_ _p_ [be matrices whose rows are the] _[ private]_
representation **h** _[s]_ _p_ [=] _[ E]_ _p_ _[s]_ [(] **[x]** _[s]_ [)] [ and] **[ h]** _[t]_ _p_ [=] _[ E]_ _p_ _[t]_ [(] **[x]** _[t]_ [)] [ from samples of source and target data respectively.]
The difference loss encourages orthogonality between the shared and the private representations of
each domain:



2 _F_ [+] ��� **H** _tc⊤_ **H** _tp_ ��� 2



2
_L_ difference = ��� **H** _sc⊤_ **H** _sp_ ��� _F_



(5)
_F_ _[,]_



where _∥· ∥_ [2] _F_ [is the squared Frobenius norm. Finally, the similarity loss encourages the hidden]
representations **h** _[s]_ _c_ [and] **[ h]** _[t]_ _c_ [from the shared encoder to be as similar as possible irrespective of the]
domain. We experimented with two similarity losses, which we discuss in detail.


**3.2** **Similarity Losses**


The domain adversarial similarity loss [ 7, 8 ] is used to train a model to produce representations
such that a classifier cannot reliably predict the domain of the encoded representation. Maximizing
such “confusion” is achieved via a Gradient Reversal Layer (GRL) and a _domain classifier_ trained
to predict the domain producing the hidden representation. The GRL has the same output as the
identity function, but reverses the gradient direction. Formally, for some function _f_ ( **u** ), the GRL
_d_
is defined as _Q_ ( _f_ ( **u** )) = _f_ ( **u** ) with a gradient _d_ **u** _[Q]_ [(] _[f]_ [(] **[u]** [)) =] _[ −]_ _d_ _[d]_ **u** _[f]_ [(] **[u]** [)] [. The domain classifier]

_Z_ ( _Q_ ( **h** _c_ ); _**θ**_ _z_ ) _→_ _d_ [ˆ] parameterized by _**θ**_ _z_ maps a shared representation vector **h** _c_ = _E_ _c_ ( **x** ; _**θ**_ _c_ ) to a
prediction of the label _d_ [ˆ] _∈{_ 0 _,_ 1 _}_ of the input sample **x** . Learning with a GRL is adversarial in that
_**θ**_ _z_ is optimized to increase _Z_ ’s ability to discriminate between encodings of images from the source
or target domains, while the reversal of the gradient results in the model parameters _**θ**_ _c_ learning
representations from which domain classification accuracy is reduced. Essentially, we _maximize_ the
binomial cross–entropy for the domain prediction task with respect to _**θ**_ _z_, while _minimizing_ it with


4


respect to _**θ**_ _c_ :



_L_ [DANN] similarity [=]



_N_ _s_ + _N_ _t_
�


_i_ =0



_d_ _i_ log _d_ [ˆ] _i_ + (1 _−_ _d_ _i_ ) log(1 _−_ _d_ [ˆ] _i_ ) _._ (6)
� �



where _d_ _i_ _∈{_ 0 _,_ 1 _}_ is the ground truth domain label for sample _i_ .


The Maximum Mean Discrepancy (MMD) loss [ 11 ] is a kernel-based distance function between pairs
of samples. We use a biased statistic for the squared population MMD between shared encodings of
the source samples **h** _[s]_ _c_ [and the shared encodings of the target samples] **[ h]** _[t]_ _c_ [:]



_N_ _[s]_ _,N_ _[t]_

1
_i,j_ � =0 _κ_ ( **h** _[s]_ _ci_ _[,]_ **[ h]** _[t]_ _cj_ [) +] ( _N_ _[t]_ ) [2]



_N_ _[t]_
� _κ_ ( **h** _[t]_ _ci_ _[,]_ **[ h]** _[t]_ _cj_ [)] _[,]_ [ (7)]

_i,j_ =0



1
_L_ [MMD] similarity [=] ( _N_ _[s]_ ) [2]



_N_ _[s]_

2

� _κ_ ( **h** _[s]_ _ci_ _[,]_ **[ h]** _[s]_ _cj_ [)] _[ −]_ _N_ _[s]_ _N_ _[t]_

_i,j_ =0



where _κ_ ( _·, ·_ ) is a PSD kernel function. In our experiments we used a linear combination of multiple
RBF kernels: _κ_ ( _x_ _i_ _, x_ _j_ ) = [�] _n_ _[η]_ _[n]_ [ exp] _[{−]_ 2 _σ_ 1 _n_ _[∥]_ **[x]** _[i]_ _[ −]_ **[x]** _[j]_ _[∥]_ [2] _[}]_ [, where] _[ σ]_ _[n]_ [ is the standard deviation and] _[ η]_ _[n]_

is the weight for our _n_ _[th]_ RBF kernel. Any additional kernels we include in the multi–RBF kernel are
additive and guarantee that their linear combination remains characteristic. Therefore, having a large
range of kernels is beneficial since the distributions of the shared features change during learning,
and different components of the multi–RBF kernel might be responsible at different times for making
sure we reject a false null hypothesis, i.e. that the loss is sufficiently high when the distributions are
not similar [ 18 ]. The advantage of using an RBF kernel with the MMD distance is that the Taylor
expansion of the Gaussian function allows us to match all the moments of the two populations. The
caveat is that it requires finding optimal kernel bandwidths _σ_ _n_ .


**4** **Evaluation**

We are motivated by the problem of learning models on a clean, synthetic dataset and testing on noisy,
real–world dataset. To this end, we evaluate on object classification datasets used in previous work [3]
including MNIST and MNIST-M [ 8 ], the German Traffic Signs Recognition Benchmark (GTSRB)

[ 26 ], and the Streetview House Numbers (SVHN) [ 21 ]. We also evaluate on the cropped LINEMOD
dataset, a standard for object instance recognition and 3D pose estimation [ 13, 32 ], for which we
have synthetic and real data [4] . We tested the following unsupervised domain adaptation scenarios: _(a)_
from MNIST to MNIST-M; _(b)_ from SVHN to MNIST; _(c)_ from synthetic traffic signs to real ones
with GTSRB; _(d)_ from synthetic LINEMOD object instances rendered on a black background to the
same object instances in the real world.


We evaluate the efficacy of our method with each of the two similarity losses outlined in Section 3.2
by comparing against the prevailing visual domain adaptation techniques for neural networks: Correlation Alignment (CORAL) [ 27 ], Domain–Adversarial Neural Networks (DANN) [ 7, 8 ], and MMD
regularization [ 30, 18 ]. For each scenario we provide two additional baselines: the performance on
the target domain of the respective model with no domain adaptation and trained _(a)_ on the source
domain (“Source–only” in Table 4) and _(b)_ on the target domain (“Target–only”), as an empirical
lower and upper bound respectively.


We have not found a universally applicable way to optimize hyperparameters for unsupervised domain
adaptation. Previous work [ 8 ] suggests the use of reverse validation. We implemented this (see
Supplementary Material for details) but found that that the reverse validation accuracy did not always
align well with test accuracy. Ideally we would like to avoid using labels from the target domain,
as it can be argued that if ones does have target domain labels, they should be used during training.
However, there are applications where a labeled target domain set cannot be used for training. An
example is the labeling of a dataset with the use of AprilTags [ 22 ], 2D barcodes that can be used to


3 The most commonly used dataset for visual domain adaptation in the context of object classification is
Office [ 24 ]. However, this dataset exhibits significant variations in both low-level and high-level parameter
distributions. Low-level variations are due to the different cameras and background textures in the images (e.g.
Amazon versus DSLR). However, there are significant high-level variations due to object identity: e.g. the
motorcycle class contains non-motorcycle objects; the backpack class contains a laptop; some domains contain
the object in only one pose. Other commonly used datasets such as Caltech-256 suffer from similar problems.
We therefore exclude these datasets from our evaluation. For more information, see our Supplementary Material.
4 `[https://cvarlab.icg.tugraz.at/projects/3d_object_detection/](https://cvarlab.icg.tugraz.at/projects/3d_object_detection/)`


5


Table 1: Mean classification accuracy (%) for the unsupervised domain adaptation scenarios we
evaluated all the methods on. We have replicated the experiments from Ganin et al. [ 8 ] and in
parentheses we show the results reported in their paper. The “Source–only” and “Target–only” rows
are the results on the target domain when using no domain adaptation and training only on the source
or the target domain respectively. The results that perform best in each domain adaptation task are in
bold font.

|Model|MNIST to<br>MNIST-M|Synth Digits to<br>SVHN|SVHN to<br>MNIST|Synth Signs to<br>GTSRB|
|---|---|---|---|---|
|Source-only|56.6 (52.2)|86.7 (86.7)|59.2 (54.9)|85.1 (79.0)|
|CORAL [27]|57.7|85.2|63.1|86.9|
|MMD [30, 18]|76.9|88.0|71.1|91.1|
|DANN [8]|77.4 (76.6)|90.3 (91.0)|70.7 (73.8)|92.9 (88.6)|
|DSN w/ MMD (ours)|80.5|88.5|72.2|92.6|
|DSN w/ DANN (ours)|**83.2**|**91.2**|**82.7**|**93.1**|
|Target-only|98.7|92.4|99.5|99.8|



label the pose of an object, provided that a camera is calibrated and the physical dimensions of the
barcode are known. These images should not be used when learning features from pixels, because the
model might be able to decipher the tags. However, they can be part of a test set that is not available
during training, and an equivalent dataset without the tags could be used for unsupervised domain
adaptation. We thus chose to use a small set of labeled target domain data as a validation set for
the hyperparameters of all the methods we compare. All methods were evaluated using the same
protocol, so comparison numbers are fair and meaningful. The performance on this validation set
can serve as an _upper bound_ of a satisfactory validation metric for unsupervised domain adaptation,
which to our knowledge is still an open research question, and out of the scope of this work.


**4.1** **Datasets and Adaptation Scenarios**


**MNIST to MNIST-M.** In this domain adaptation scenario we use the popular MNIST [ 16 ] dataset
of handwritten digits as the source domain, and MNIST-M, a variation of MNIST proposed for
unsupervised domain adaptation by [ 8 ]. MNIST-M was created by using each MNIST digit as a
binary mask and inverting with it the colors of a background image. The background images are
random crops uniformly sampled from the Berkeley Segmentation Data Set (BSDS500) [ 3 ]. In all
our experiments, following the experimental protocol by [ 8 ]. Out of the 59 _,_ 001 MNIST-M training
examples, we used the labels for 1 _,_ 000 of them to find optimal hyperparameters for our models. This
scenario, like all three digit adaptation scenarios, has 10 class labels.


**Synthetic Digits to SVHN.** In this scenario we aim to learn a classifier for the Street-View House
Number data set (SVHN) [ 21 ], our target domain, from a dataset of purely synthesized digits,
our source domain. The synthetic digits [ 8 ] dataset was created by rasterizing bitmap fonts in a
sequence (one, two, and three digits) with the ground truth label being the digit in the center of the
image, just like in SVHN. The source domain samples are further augmented by variations in scale,
translation, background colors, stroke colors, and Gaussian blurring. We use 479 _,_ 400 Synthetic
Digits for our source domain training set, 73 _,_ 257 unlabeled SVHN samples for domain adaptation,
and 26 _,_ 032 SVHN samples for testing. Similarly to above, we used the labels of 1 _,_ 000 SVHN
training examples to find optimal hyperparameters for our models.


**SVHN to MNIST.** Although the SVHN dataset contains significant variations (in scale, background
clutter, blurring, embossing, slanting, contrast, rotation, sequences to name a few) there is not a lot of
variation in the actual digits shapes. This makes it quite distinct from a dataset of handwritten digits,
like MNIST, where there are a lot of elastic distortions in the shapes, variations in thickness, and
noise on the digits themselves. Since the ground truth digits in both datasets are centered, this is a
well–posed and rather difficult domain adaptation scenario. As above, we used the labels of 1 _,_ 000
MNIST training examples for validation.


**Synthetic Signs to GTSRB.** We also perform an experiment using a dataset of synthetic traffic
signs from [ 20 ] to real world dataset of traffic signs (GTSRB) [ 26 ]. While the three digit adaptation
scenarios have 10 class labels, this scenario has 43 different traffic signs. The synthetic signs were


6


Table 2: Mean classification accuracy and pose error for the “Synth Objects to LINEMOD” scenario.

|Method|Classification Accuracy|Mean Angle Error|
|---|---|---|
|Source-only|47.33%|89_._2~~_◦_~~<br>|
|MMD|72.35%|70_._62~~_◦_~~<br>|
|DANN|99.90%|56_._58~~_◦_~~<br>|
|DSN w/ MMD (ours)|99.72%|66_._49~~_◦_~~<br>|
|DSN w/ DANN (ours)|**100.00**%|**53.27**~~_◦_~~<br>|
|Target-only|100.00%|6_._47~~_◦_~~|



(a) MNIST (source) (b) MNIST-M (target) (c) Synth Objects (source) (d) LINEMOD (target)


Figure 2: Reconstructions for the representations of the two domains for “MNIST to MNIST-M”
and for “Synth Objects to LINEMOD”. In each block from left to right: the original image **x** _t_ ;
reconstructed image _D_ ( _E_ _c_ ( **x** _[t]_ ) + _E_ _p_ ( **x** _[t]_ )) ; shared only reconstruction _D_ ( _E_ _c_ ( **x** _[t]_ )) ; private only
reconstruction _D_ ( _E_ _p_ ( **x** _[t]_ )).


obtained by taking relevant pictograms and adding various types of variations, including random
backgrounds, brightness, saturation, 3D rotations, Gaussian and motion blur. We use 90 _,_ 000 synthetic
signs for training, 1 _,_ 280 random GTSRB real–world signs for domain adaptation and validation, and
the remaining 37 _,_ 929 GTSRB real signs as the test set.


**Synthetic Objects to LineMod.** The LineMod dataset [ 32 ] consists of CAD models of objects in a
cluttered environment and a high variance of 3D poses for each object. We use the 11 non–symmetric
objects from the cropped version of the dataset, where the images are cropped with the object in the
center, for the task of object instance recognition and 3D pose estimation. We train our models on
16 _,_ 962 images for these objects rendered on a black background without additional noise. We use a
target domain training set of 10 _,_ 673 real–world images for domain adaptation and validation, and a
target domain test set of 2 _,_ 655 for testing. For this scenario our task is both classification and pose
estimation; our task loss is therefore _L_ task = [�] _[N]_ _i_ =0 _[s]_ _[{−]_ **[y]** _i_ _[s]_ _[·]_ [ log ˆ] **[y]** _i_ _[s]_ [+] _[ ξ]_ [ log(1] _[ −|]_ **[q]** _[s]_ _[ ·]_ [ ˆ] **[q]** _[s]_ _[|]_ [)] _[}]_ [, where] **[ q]** _[s]_
is the positive unit quaternion vector representing the ground truth 3D pose, and ˆ **q** _[s]_ is the equivalent
prediction. The first term is the classification loss, similar to the rest of the experiments, the second
term is the log of a 3D rotation metric for quaternions [ 14 ], and _ξ_ is the weight for the pose loss.
Quaternions are a convenient angle–axis representation for 3D rotations. In Table 2 we report the
mean angle the object would need to be rotated (on a fixed 3D axis) to move from the predicted pose
to the ground truth [13].


**4.2** **Implementation Details**


All the models were implemented using TensorFlow [5] [ 1 ] and were trained with Stochastic Gradient
Descent plus momentum [ 28 ]. Our initial learning rate was multiplied by 0 _._ 9 every 20 _,_ 000 steps
(mini-batches). We used batches of 32 samples from each domain for a total of 64 and the input
images were mean-centered and rescaled to [ _−_ 1 _,_ 1] . In order to avoid distractions for the main
classification task during the early stages of the training procedure, we activate any additional domain
adaptation loss after 10 _,_ 000 steps of training. For all our experiments our CNN topologies are based
on the ones used in [ 8 ], to be comparable to previous work in unsupervised domain adaptation. The
exact architectures for all models are shown in our Supplementary Material.


5 Our code will be open–sourced under `[https://github.com/tensorflow/models/](https://github.com/tensorflow/models/)` before the NIPS
2016 meeting.


7


Table 3: Effect of our difference and reconstruction losses on our best model. The first row is
replicated from Table 4. In the second row, we remove the soft orthogonality constraint. In the third
row, we replace the scale–invariant MSE with regular MSE.

|Model|MNIST to<br>MNIST-M|Synth. Digits to<br>SVHN|SVHN to<br>MNIST|Synth. Signs to<br>GTSRB|
|---|---|---|---|---|
|All terms|**83.23**|**91.22**|**82.78**|**93.01**|
|No_ L_diﬀerence|80.26|89.21|80.54|91.89|
|With_ L_~~_L_2~~<br>recon|80.42|88.98|79.45|92.11|



In our framework, CORAL [ 27 ] would be equivalent to fixing our shared representation matrices
**H** _[s]_ _c_ [and] **[ H]** _[t]_ _c_ [, normalizing them and then minimizing] _[ ∥]_ **[AH]** _[s]_ _c⊤_ **H** _sc_ **[A]** _[⊤]_ _[−]_ **[H]** _[t]_ _c⊤_ **H** _tc_ _[∥]_ [2] _F_ [with respect to a]
weight matrix **A** that aligns the two correlation matrices. For the CORAL experiments, we follow the
suggestions of [ 27 ], and extract features for both source and target domains from the penultimate layer
of each network. Once the correlation matrices for each domain are aligned, we evaluate on the target
test data the performance of a linear support vector machine (SVM) classifier trained on the source
training data. The SVM penalty parameter was optimized based on the target domain validation set
for each of our domain adaptation scenarios. For MMD regularization, we used a linear combination
of 19 RBF kernels [6] . We applied MMD on _fc3_ on all our model architectures and minimized
_L_ = _L_ class + _γ L_ [MMD] similarity [with respect to] _**[ θ]**_ _[c]_ _[,]_ _**[ θ]**_ _[g]_ [. Preliminary experiments with having MMD]
applied on more than one layers did not show any performance improvement for our experiments and
architectures. For DANN regularization, we applied the GRL and the domain classifier as prescribed
in [ 8 ] for each scenario. We optimized _L_ = _L_ class + _γ L_ [DANN] similarity [by minimizing it with respect to]
_**θ**_ _c_ _,_ _**θ**_ _g_ and maximizing it with respect to the domain classifier parameters _**θ**_ _z_ .


For our Domain Separation Network experiments, our similarity losses are always applied at the
first fully connected layer of each network after a number of convolutional and max pooling layers.
For each private space encoder network we use a simple convolutional and max pooling structure
followed by a fully-connected layer with a number of nodes equal to the number of nodes at the final
layer **h** _c_ of the equivalent shared encoder _E_ _c_ . The output of the shared and private encoders gets
added before being fed to the shared decoder _D_ . For the latter we use a deconvolutional architecture

[ 33 ] which consists of a fully connected layer with 300 nodes, a resizing layer to 10 _×_ 10 _×_ 3,
two 3 _×_ 3 _×_ 16 convolutional layers, one upsampling layer to 32 _×_ 32 _×_ 16, another 3 _×_ 3 _×_ 16
convolutional layer, followed by the reconstruction output.


**4.3** **Discussion**


The DSN with DANN model outperforms all the other methods we experimented with for all our
unsupervised domain adaptation scenarios (see Table 4 and 2). Our unsupervised domain separation
networks are able to improve both upon MMD regularization and DANN. Using DANN as a similarity
loss (Equation 6) worked better than using MMD (Equation 7) as a similarity loss, which is consistent
with results obtained for domain adaptation using MMD regularization and DANN alone.


In order to examine the effect of the soft orthogonality constraints ( _L_ difference ), we took our best
model, our DSN model with the DANN loss, and removed these constraints by setting the _β_ coefficient
to 0 . Without them, the model performed consistently worse in all scenarios. We also validated our
choice of our scale–invariant mean squared error reconstruction loss as opposed to the more popular
mean squared error loss by running our best model with _L_ _[L]_ recon [2] [=] _k_ [1] _[||]_ **[x]** _[ −]_ **[x]** [ˆ] _[||]_ 2 [2] [. With this variation]

we also get worse classification results consistently, as shown in experiments from Table 3.


The shared and private representations of each domain are combined for the reconstruction of samples.
Individually decoding the shared and private representations gives us reconstructions that serve as
useful depictions of our domain adaptation process. In Figure 2 we use the “MNIST to MNIST-M”
and the “Synth. Objects to LINEMOD” scenarios for such visualizations. In the former scenario,
the model clearly separates the foreground from the background and produces a shared space that is
very similar to the source domain. This is expected since the target is a transformation of the source.
In the latter scenario, the model is able to produce visualizations of the shared representation that


6 The Supplementary Material has details on all the parameters.


8


look very similar between source and target domains, which are useful for classification and pose
estimation, as shown in Table 2.


**5** **Conclusion**

We present in this work a deep learning model that improves upon existing unsupervised domain
adaptation techniques. The model does so by explicitly separating representations private to each
domain and shared between source and target domains. By using existing domain adaptation
techniques to make the shared representations similar, and soft subspace orthogonality constraints to
make private and shared representations dissimilar, our method outperforms all existing unsupervised
domain adaptation methods in a number of adaptation scenarios that focus on the synthetic–to–real
paradigm.


**Acknowledgments**


We would like to thank Samy Bengio, Kevin Murphy, and Vincent Vanhoucke for valuable comments
on this work. We would also like to thank Yaroslav Ganin and Paul Wohlhart for providing some of
the datasets we used.


9


**References**


[1] M. Abadi et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. _Preprint_
_arXiv:1603.04467_, 2016.

[2] H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, and M. Marchand. Domain-adversarial neural
networks. In _Preprint, http://arxiv.org/abs/1412.4446_, 2014.

[3] P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik. Contour detection and hierarchical image segmentation.
_TPAMI_, 33(5):898–916, 2011.

[4] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan. A theory of learning
from different domains. _Machine learning_, 79(1-2):151–175, 2010.

[5] R. Caseiro, J. F. Henriques, P. Martins, and J. Batist. Beyond the shortest path: Unsupervised Domain
Adaptation by Sampling Subspaces Along the Spline Flow. In _CVPR_, 2015.

[6] D. Eigen, C. Puhrsch, and R. Fergus. Depth map prediction from a single image using a multi-scale deep
network. In _NIPS_, pages 2366–2374, 2014.

[7] Y. Ganin and V. Lempitsky. Unsupervised domain adaptation by backpropagation. In _ICML_, pages
513–520, 2015.

[8] Y. Ganin et al. . Domain-Adversarial Training of Neural Networks. _JMLR_, 17(59):1–35, 2016.

[9] B. Gong, Y. Shi, F. Sha, and K. Grauman. Geodesic flow kernel for unsupervised domain adaptation. In
_CVPR_, pages 2066–2073. IEEE, 2012.

[10] R. Gopalan, R. Li, and R. Chellappa. Domain Adaptation for Object Recognition: An Unsupervised
Approach. In _ICCV_, 2011.

[11] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A Kernel Two-Sample Test.
_JMLR_, pages 723–773, 2012.

[12] G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. _CNS-TR-2007-001_, 2007.

[13] S. Hinterstoisser et al. . Model based training, detection and pose estimation of texture-less 3d objects in
heavily cluttered scenes. In _ACCV_, 2012.

[14] D. Q. Huynh. Metrics for 3d rotations: Comparison and analysis. _Journal of Mathematical Imaging and_
_Vision_, 35(2):155–164, 2009.

[15] Y. Jia, M. Salzmann, and T. Darrell. Factorized latent spaces with structured sparsity. In _NIPS_, pages
982–990, 2010.

[16] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.
_Proceedings of the IEEE_, 86(11):2278–2324, 1998.

[17] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft
coco: Common objects in context. In _ECCV 2014_, pages 740–755. Springer, 2014.

[18] M. Long and J. Wang. Learning transferable features with deep adaptation networks. _ICML_, 2015.

[19] Y. Mansour et al. . Domain adaptation with multiple sources. In _NIPS_, 2009.

[20] B. Moiseev, A. Konev, A. Chigorin, and A. Konushin. _Evaluation of Traffic Sign Recognition Meth-_
_ods Trained on Synthetically Generated Data_, chapter ACIVS, pages 576–583. Springer International
Publishing, 2013.

[21] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with
unsupervised feature learning. In _NIPS Workshops_, 2011.

[22] E. Olson. Apriltag: A robust and flexible visual fiducial system. In _Robotics and Automation (ICRA), 2011_
_IEEE International Conference on_, pages 3400–3407. IEEE, 2011.

[23] O. Russakovsky et al. ImageNet Large Scale Visual Recognition Challenge. _IJCV_, 115(3):211–252, 2015.

[24] K. Saenko et al. . Adapting visual category models to new domains. In _ECCV_ . Springer, 2010.

[25] M. Salzmann et. al. Factorized orthogonal latent spaces. In _AISTATS_, pages 701–708, 2010.

[26] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. Man vs. computer: Benchmarking machine learning
algorithms for traffic sign recognition. _Neural Networks_, 2012.

[27] B. Sun, J. Feng, and K. Saenko. Return of frustratingly easy domain adaptation. In _AAAI_ . 2016.

[28] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in
deep learning. In _ICML_, pages 1139–1147, 2013.

[29] E. Tzeng, J. Hoffman, T. Darrell, and K. Saenko. Simultaneous deep transfer across domains and tasks. In
_CVPR_, pages 4068–4076, 2015.

[30] E. Tzeng, J. Hoffman, N. Zhang, K. Saenko, and T. Darrell. Deep domain confusion: Maximizing for
domain invariance. _Preprint arXiv:1412.3474_, 2014.

[31] S. Virtanen, A. Klami, and S. Kaski. Bayesian CCA via group sparsity. In _ICML_, pages 457–464, 2011.

[32] P. Wohlhart and V. Lepetit. Learning descriptors for object recognition and 3d pose estimation. In _CVPR_,
pages 3109–3118, 2015.

[33] M. D. Zeiler, D. Krishnan, G. W. Taylor, and R. Fergus. Deconvolutional networks. In _CVPR_, pages
2528–2535. IEEE, 2010.


10


**Supplementary Material**


**A** **Correlation Regularization**


Correlation Alignment (CORAL) [ 27 ] aims to find a mapping from the representations of the source
domain to the representations of the target domain by matching only the second–order statistics. In our
framework, this would be equivalent to fixing our common representation matrices **H** _[s]_ _c_ [and] **[ H]** _[t]_ _c_ [after]



normalizing them and then finding a weight matrix **A** [ˆ] = argmin
**A**



**AH** _sc⊤_ **H** _sc_ **[A]** _[⊤]_ _[−]_ **[H]** _[t]_ _c⊤_ **H** _tc_ 2
��� ��� _F_ [that]



aligns the two correlation matrices. Although this has the advantage that the optimization is convex
and can be solved in closed form, all convolutional features remain fixed during the process, which
might not be optimal for the task at hand. Also, because of this we are not able to use it as a similarity
loss for our DSNs. Motivated by this shortcoming, we propose here a new domain adaptation
method, Correlation Regularization (CorReg). We show in Table 4 that our new domain adaptation
method, which is theoretically as powerful as an MMD loss with a second–order polynomial kernel,
outperforms CORAL in all our datasets. Adapting a feature hierarchy to be domain–invariant is
more powerful than learning a mapping from the representations of one domain to those of another.
Moreover, we use it as yet another similarity loss for our Domain Separation Networks:

_L_ [CorReg] similarity [=] ��� **H** _sc⊤_ **H** _sc_ _[−]_ **[H]** _[t]_ _c⊤_ **H** _tc_ ��� 2 _F_ (8)


Our DNS with CorReg performs better than both CORAL and CorReg, which is consistent with the
rest of our results.


Table 4: Our main results from the paper with two additional lines for CorReg and DSN with CorReg.

|Model|MNIST to<br>MNIST-M|Synth Digits to<br>SVHN|SVHN to<br>MNIST|Synth Signs to<br>GTSRB|
|---|---|---|---|---|
|Source-only|56.6 (52.2)|86.7 (86.7)|59.2 (54.9)|85.1 (79.0)|
|CORAL [27]|57.7|85.2|63.1|86.9|
|CorReg (Ours)|62.06|87.33|69.20|90.75|
|MMD [30, 18]|76.9|88.0|71.1|91.1|
|DANN [8]|77.4 (76.6)|90.3 (91.0)|70.7 (73.8)|92.9 (88.6)|
|DSN w/ MMD (ours)|80.5|88.5|72.2|92.6|
|DSN w/ DANN (ours)|**83.2**|**91.2**|**82.7**|**93.1**|
|Target-only|98.7|92.4|99.5|99.8|



**B** **Office Dataset Criticism**


The most commonly used dataset for visual domain adaptation in the context of object classification
is Office [ 24 ], sometimes combined with the Caltech–256 dataset [ 12 ] as an additional domain.
However, these datasets exhibit significant variations in both low-level and high-level parameter
distributions. Low-level variations are due to the different cameras and background textures in the
images (e.g. Amazon versus DSLR), which is welcome. However, there are significant high-level
variations due to elements like label pollution: e.g. the motorcycle class contains non-motorcycle
objects; the backpack class contains 2 laptops; some classes contain the object in only one pose.
Other commonly used datasets such as Caltech-256 suffer from similar problems. We illustrate some
of these issues for the ‘back_pack’ class for its 92 Amazon samples, its 12 DSLR samples, its 29
Webcam samples, and its 151 Caltech samples in Figure 3. Other classes exhibit similar problems.
For these reasons some works, eg [ 27 ], pretrain their models on Imagenet before performing the
domain adaptation in these scenarios. This essentially involves another source domain (Imagenet) in
the transfer.


**C** **Domain Separation**


We visualize in Figure 4 reconstructions for both source and target domains of each domain adaptation
scenario. Although the visualizations are not as clear as with the “MNIST to MNIST-M” scenario,


11


Figure 3: Examples of the ‘back_pack’ class in the different domains in Office and Caltech–256.
**First Row:** 5 of the 92 images in the Amazon domain. **Second Row:** The DSLR domain contains
4 images for the rightmost image from different frontal angles, 2 images for the other 4 backpacks
for a total of 12 images for this class. **Third Row:** The webcam domain contains the exact same
backpacks with DSLR with similar poses for a total of 29 images for this class. **Fourth Row:** Some
of the 151 backpack samples Caltech domain.


where the target domain was a direct transformation of the source domain, it is interesting to note
the similarities of the visualizations of the shared representations, and the exclusion of some shared
information in the private domains.


(a) (b) (c) (d)


Figure 4: Reconstructions for the representations of the two domains for _a) Synthetic Digits to SVHN,_
_b) SVHN to MNIST, c) Synthetic Signs to GTSRB, d) Synthetic Objects to LineMOD. In each block_
_from left to right: the original image_ **x** _t_ _; reconstructed image_ _D_ ( _E_ _c_ ( **x** _[t]_ ) + _E_ _p_ ( **x** _[t]_ )) _; shared only_
_reconstruction_ _D_ ( _E_ _c_ ( **x** _[t]_ )) _; private only reconstruction_ _D_ ( _E_ _p_ ( **x** _[t]_ )) _.Reconstructions of target (top_
_row) and source (bottom row) domains._ .


12


**shared** encoder _**Ec(·;**_ _**θc**_ _**)**_













_**E**_ **p** _**t**_ _**(·;**_ _**θp**_ _t_ _**)**_



_t_



**private target** encoder













_**E**_ **p** _**s**_ _**(·;**_ _**θp**_ _s_ _**)**_



**private source** encoder



_s_

















**shared decoder** _**D(·; θd)**_















**classifier** _**G(·; θc)**_







**domain adversarial network** _**Z(·; θz)**_









Figure 5: The network topology for “MNIST to MNIST-M”


**D** **Network Topologies and Optimal Parameters**


Since we used different network topologies for our domain adaptation scenarios, there was not enough
space to include these in the main paper. We present the exact topologies used in Figures 5–8.


Similarly, we list here all hyperparameters that are important for total reproducibility of all our results.
For CORAL, the SVM penalty parameter that was optimized based on the validation set for each of
our domain adaptation scenarios: 1 _e_ _[−]_ [4] for “MNIST to MNIST-M”, “Synth Digits to SVHN”, “Synth
Signs to GTSRB”, and 1 _e_ _[−]_ [3] for “SVHN to MNIST”. For MMD we use 19 RBF kernels with the
following standard deviation parameters:


_**σ**_ = [10 _[−]_ [6] _,_ 10 _[−]_ [5] _,_ 10 _[−]_ [4] _,_ 10 _[−]_ [3] _,_ 10 _[−]_ [2] _,_ 10 _[−]_ [1] _,_ 1 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 25 _,_ 30 _,_ 35 _,_ 100 _,_ 10 [3] _,_ 10 [4] _,_ 10 [5] _,_ 10 [6] ]


and equal _η_ weights. We use learning rate between [0 _._ 01 _,_ 0 _._ 015] and _γ ∈_ [0 _._ 1 _,_ 0 _._ 3] . For DANN
we use learning rate between [0 _._ 01 _,_ 0 _._ 015] and _γ ∈_ [0 _._ 15 _,_ 0 _._ 25] . For DSN w/ DANN and DSN w/
MMD we use a constant initial learning rate of 0 _._ 01 use the hyperparameters in the range of: _α ∈_

[0 _._ 01 _,_ 0 _._ 15] _, β ∈_ [0 _._ 05 _,_ 0 _._ 075] _, γ ∈_ [0 _._ 25 _,_ 0 _._ 3], whereas for DNS w/ CorReg we use _γ ∈_ [20 _,_ 100] .
For the GTSRB experiment we use _α ∈_ [0 _._ 01 _,_ 0 _._ 015] . In all cases we use an exponential decay of
0 _._ 95 on the learning rate every 20 _,_ 000 iterations. For the LINEMOD experiments we use _ξ_ = 0 _._ 125 .


13


**shared** encoder _**[E]**_ _**c**_ _**[(·; ]**_ _**[θ]**_ _**c**_ _**[)]**_













**private target** encoder _**E**_ **p** _**t**_ _**(·;**_ _**θp**_ _t_ _**)**_













**private source** encoder



_**E**_ **p** _**s**_ _**(·;**_ _**θp**_ _s_ _**)**_

















**shared decoder** _**D(·;**_ _**θd**_ _**)**_

















**classifier** _**G(·;**_ _**θc**_ _**)**_







**domain adversarial network** _**Z(·;**_ _**θz**_ _**)**_









Figure 6: The network topology for “Synth SVHN to SVHN” and “SVHN to MNIST” experiments.


**shared** encoder _**[E]**_ _**c**_ _**[(·; ]**_ _**[θ]**_ _**c**_ _**[)]**_

















**private target** encoder _**E**_ **p** _**t**_ _**(·;**_ _**θp**_ _t_ _**)**_

















**private source** encoder



_**E**_ **p** _**s**_ _**(·;**_ _**θp**_ _s_ _**)**_















**shared decoder** _**D(·;**_ _**θd**_ _**)**_















**classifier** _**[G(][·][; ]**_ _**[θ]**_ _**c**_ _**[)]**_







**domain adversarial network** _**Z(·;**_ _**θz**_ _**)**_









Figure 7: The network topology for “Synth Signs to GTSRB”


14


**shared** encoder _**[E]**_ _**c**_ _**[(·; ]**_ _**[θ]**_ _**c**_ _**[)]**_













_**E**_ **p** _**t**_ _**(·;**_ _**θp**_ _t_ _**)**_



**private target** encoder



_t_

















_**E**_ **p** _**s**_ _**(·;**_ _**θp**_ _s_ _**)**_



**private source** encoder



_s_

















**shared decoder** _**D(·;**_ _**θd**_ _**)**_



















**task specific network** _**G(·;**_ _**θc**_ _**)**_







**domain adversarial network** _**Z(·;**_ _**θz**_ _**)**_









Figure 8: The network topology for “Synthetic Objects to Linemod”


15


