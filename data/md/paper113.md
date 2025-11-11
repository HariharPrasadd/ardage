## **Learning Perceptual Inference by Contrasting**

**Chi Zhang** _[⋆,]_ [1] _[,]_ [4] **, Baoxiong Jia** _[⋆,]_ [1] **, Feng Gao** [3] _[,]_ [4] **, Yixin Zhu** [3] _[,]_ [4] **, Hongjing Lu** [2] **, Song-Chun Zhu** [1] _[,]_ [3] _[,]_ [4]

1 Department of Computer Science, University of California, Los Angeles
2 Department of Psychology, University of California, Los Angeles
3 Department of Statistics, University of California, Los Angeles
4 International Center for AI and Robot Autonomy (CARA)
```
    {chi.zhang,baoxiongjia,f.gao,yixin.zhu,hongjing,sczhu}@ucla.edu

```

**Abstract**


“Thinking in pictures,” [ 1 ] _i.e_ ., spatial-temporal reasoning, effortless and instantaneous for humans, is believed to be a significant ability to perform logical induction
and a crucial factor in the intellectual history of technology development. Modern
Artificial Intelligence ( AI ), fueled by massive datasets, deeper models, and mighty
computation, has come to a stage where (super-)human-level performances are
observed in certain specific tasks. However, current AI ’s ability in “thinking in
pictures” is still far lacking behind. In this work, we study how to improve machines’ reasoning ability on one challenging task of this kind: Raven’s Progressive
Matrices ( RPM ). Specifically, we borrow the very idea of “contrast effects” from
the field of psychology, cognition, and education to design and train a permutationinvariant model. Inspired by cognitive studies, we equip our model with a simple
inference module that is jointly trained with the perception backbone. Combining all the elements, we propose the _Contrastive Perceptual Inference_ network
(CoPINet) and empirically demonstrate that CoPINet sets the new state-of-the-art
for permutation-invariant models on two major datasets. We conclude that spatialtemporal reasoning depends on envisaging the possibilities consistent with the
relations between objects and can be solved from pixel-level inputs.


**1** **Introduction**


Among the broad spectrum of computer vision tasks are ones where dramatic progress has been
witnessed, especially those involving visual information retrieval [ 2 – 5 ]. Significant improvement
has also manifested itself in tasks associating visual and linguistic understanding [ 6 – 9 ]. However, it
was only until recently that the research community started to re-investigate tasks relying heavily
on the ability of “thinking in pictures” with modern AI approaches [ 1, 10, 11 ], particularly spatialtemporal inductive reasoning [ 12 – 14 ]; this line of work primarily focuses on Raven’s Progressive
Matrices ( RPM ) [ 15, 16 ]. It is believed that RPM is closely related to real intelligence [ 17 ], diagnostic
of abstract and structural reasoning ability [ 18 ], and characterizes _fluid intelligence_ [ 19 – 22 ]. In such
a test, subjects are provided with two rows of figures following certain _unknown_ rules and asked
to pick the correct answer from the choices that would best complete the third row with a missing
entry; see Figure 1(a) for an example. As shown in early works [ 12, 14 ], despite the fact that _visual_
_elements_ are relatively straightforward, there is still a notable performance gap between human and
machine _visual reasoning_ in this challenging task.


One missing ingredient that may result in this performance gap is a proper form of contrasting
mechanism. Originated from perceptual learning [ 23, 24 ], it is well established in the field of
psychology and education [ 25 – 29 ] that teaching new concepts by comparing with noisy examples is


_⋆_ indicates equal contribution.


33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.


quite effective. Smith and Gentner [30] summarize that comparing cases facilitates transfer learning
and problem-solving, as well as the ability to learn relational categories. Gentner [31] in his structuremapping theory points out that learners generate a structure alignment between two representation
when they compare two cases. A more recent study from Schwartz et al. [32] also shows that
contrasting cases help foster an appreciation of a deep understanding of concepts.


We argue that such a _contrast effect_ [ 33 ], found in both humans and animals [ 34 – 38 ], is essential to
machines’ reasoning ability as well. With access to how the data is generated, a recent attempt [ 13 ]
finds that models demonstrate better generalizability if the choice of data and the manner in which
it is presented to the model are made “contrastive.” In this paper, we try to address a more direct
and challenging question, _independent_ of how the data is generated: how to incorporate an explicit
contrasting mechanism during model _training_ in order to improve machines’ reasoning ability?
Specifically, we come up with two levels of contrast in our model: a novel contrast module and
a new contrast loss. At the model level, we design a permutation-invariant contrast module that
summarizes the common features and distinguishes each candidate by projecting it onto its residual on
the common feature space. At the objective level, we leverage ideas in contrastive estimation [ 39 – 41 ]
and propose a variant of Noise-Contrastive Estimation (NCE) loss.


Another reason why RPM is challenging for existing machine reasoning systems could be attributed
to the demanding nature of the _interplay_ between perception and inference. Carpenter et al. [17]
postulate that a proper understanding of one RPM instance requires not only an accurate encoding
of individual elements and their visual attributes but also the correct induction of the hidden rules.
In other words, to solve RPM, machine reasoning systems are expected to be equipped with _both_
perception and inference subsystems; lacking either component would only result in a sub-optimal
solution. While existing work primarily focuses on perception, we propose to bridge this gap with
a simple inference module _jointly_ trained with the perception backbone; specifically, the inference
module reasons about which category the current problem instance falls into. Instead of training the
inference module to predict the ground-truth category, we borrow the basis learning idea from [ 42 ]
and jointly learn the inference subsystem with perception. This basis formulation could also be
regarded as a hidden variable and trained using a log probability estimate.


Furthermore, we hope to make a critical improvement to the model design such that it is truly
_permutation-invariant_ . The invariance is mandatory, as an ideal RPM solver should not change the
representation simply because the rows or columns of answer candidates are swapped or the order
of the choices alters. This characteristic is an essential trait missed by all recent works [ 12, 14 ].
Specifically, Zhang et al. [12] stack all choices in the channel dimension and feed it into the network
in one pass. Barrett et al. [14] add additional positional tagging to their Wild Relational Network
( WReN ). Both of them _explicitly_ make models permutation-sensitive. We notice in our experiments
that removing the positional tagging in WReN decreases the performance by 28%, indicating that the
model bypasses the intrinsic complexity of RPM by remembering the positional association. Making
the model permutation-invariant also shifts the problem from classification to ranking.


Combining contrasting, perceptual inference, and permutation invariance, we propose the _Contrastive_
_Perceptual Inference_ network (CoPINet). To verify its effectiveness, we conduct comprehensive
experiments on two major datasets: the RAVEN dataset [ 12 ] and the PGM dataset [ 14 ]. Empirical
studies show that our model achieves human-level performance on RAVEN and a new record on
PGM, setting new state-of-the-art for permutation-invariant models on the two datasets. Further
ablation on RAVEN and PGM reveals how each component contributes to performance improvement.
We also investigate how the model performance varies under different sizes of datasets, as a step
towards an ideal machine reasoning system capable of low-shot learning.


This paper makes four major contributions:

_•_ We introduce two levels of contrast to improve machines’ reasoning ability in RPM . At the model
level, we design a contrast module that aggregates common features and projects each candidate
to its residual. At the objective level, we use an NCE loss variant instead of the cross-entropy to
encourage contrast effects.

_•_ Inspired by Carpenter et al. [17], we incorporate an inference module to learn with the perception
backbone jointly. Instead of using ground-truth, we regularize it with a fixed number of bases.

_•_ We make our model permutation-invariant in terms of swapped rows or columns and shuffled
answer candidates, shifting the previous view of RPM from classification to ranking.

_•_ Combining ideas above, we propose CoPINet that sets new state-of-the-art on two major datasets.


2


(a)



(b)


_O_


_O [ a_ 1


...


_O [ a_ 8



...


Perception Branch





Contrast Module



ResBlock





MLP (Gumbel-)SoftMax Contrast Loss

Inference Branch





(Gumbel-)SoftMax



(c)


_F_ _O[a_ 1


_F_ _O[a_ 2


...


_F_ _O[a_ 8







Contrast Module





Figure 1: (a) An example of RPM . The hidden rule(s) in this problem can be denoted as _{_ [OR _,_ line _,_ type] _}_,
where an OR operation is applied to the type attribute of all lines, following the notations in Barrett et al. [14] .
It is further noted that the OR operation is applied row-wise, and there is only one choice that satisfies the
row-wise OR constraint. Hence the correct answer should be 5 . (b) The proposed CoPINet architecture. Given a
RPM problem, the inference branch samples a most likely rule for each attribute based only on the context _O_ of
the problem. Sampled rules are transformed and fed into each contrast module in the perception branch. Note
that the combination of the contrast module and the residual block can be repeated. Dashed lines indicate that
parameters are shared among the modules. (c) A sketch of the contrast module.


**2** **Related Work**


**Contrastive Learning** Teaching concepts by comparing cases, or contrasting, has proven effective
in both human learning and machine learning. Gentner [31] postulates that human’s learning-bycomparison process is a structural mapping and alignment process. A later article [ 43 ] firmly supports
this conjecture and shows finding the individual difference is easier for humans when similar items
are compared. Recently, Smith and Gentner [30] conclude that learning by comparing two contrastive
cases facilitates the distinction between two complex interrelated relational concepts. Evidence
in educational research further strengthens the importance of contrasting—quantitative structure
of empirical phenomena is less demanding to learn when contrasting cases are used [ 32, 44, 45 ].
All the literature calls for a similar treatment of contrast in machine learning. While techniques
from [ 46 – 48 ] are based on triplet loss using max margin to separate positive and negative samples,
negative contrastive samples and negative sampling are proposed for language modeling [ 40 ] and
word embedding [ 49, 50 ], respectively. Gutmann and Hyvärinen [39] discuss a general learning
framework called Noise-Contrastive Estimation ( NCE ) for estimating parameters by taking noise
samples into consideration, which Dai and Lin [41] follow to learn an effective image captioning
model. A recent work [ 13 ] leverages contrastive learning in RPM ; however, it focuses on data
presentation while leaving the question of modeling and learning unanswered.


**Computational Models on RPM** The cognitive science community is the first to investigate RPM
with computational models. Assuming access to a perfect state representation, structure-mapping
theory [ 31 ] and the high-level perception theory of analogy [ 51, 52 ] are designed with heuristics to
solve the RPM problem at a symbolic level [ 17, 53 – 55 ]. Another stream of research approaches the
problem by measuring the image similarity with hand-crafted state representations [ 56 – 60 ]. More
recently, end-to-end data-driven methods with raw image input are proposed [ 12 – 14, 61 ]. Wang and
Su [61] introduce an automatic RPM generation method. Barrett et al. [14] release the first large-scale
RPM dataset and present a relational model [ 62 ] designed for it. Steenbrugge et al. [63] propose a
pretrained _β_ -VAE to improve the generalization performance of models on RPM . Zhang et al. [12]
provide another dataset with structural annotations using stochastic image grammar [ 64 – 66 ]. Hill
et al. [13] take a different approach and study how data presentation affects learning.


**3** **Learning Perceptual Inference by Contrasting**


The task of RPM can be formally defined as: given a list of observed images _O_ = _{o_ _i_ _}_ [8] _i_ =1 [, forming]
a 3 _×_ 3 matrix with a final missing element, a solver aims to find an answer _a_ _⋆_ from an _unordered_ set


3


of choices _A_ = _{a_ _i_ _}_ [8] _i_ =1 [to best complete the matrix. Permutation invariance is a unique property for]
RPM problems: (1) According to [ 17 ], the same set of rules is applied either row-wise or column-wise.
Therefore, swapping the first two rows or columns should not affect how one solves the problem.
(2) In any multi-choice task, changing the order of answer candidates should not affect how one
solves the problem either. These properties require us to use a permutation-invariant encoder and
reformulate the problem from a typical classification problem into a ranking problem. Formally, in a
probabilistic formulation, we seek to find a model such that


_p_ ( _a_ _⋆_ _|O_ ) _≥_ _p_ ( _a_ _[′]_ _|O_ ) _,_ _∀a_ _[′]_ _∈A, a_ _[′]_ = _̸_ _a_ _⋆_ _,_ (1)


where the probability is invariant when rows or columns in _O_ are swapped. This formulation also
calls for a model that produces a density estimation for each choice, regardless of its order in _A_ .
To that end, we model the probability with a neural network equipped with a permutation-invariant
encoder for each observation-candidate pair _f_ ( _O ∪_ _a_ ) . However, we argue such a purely perceptive
system is far from sufficient without contrasting and perceptual inference.


**3.1** **Contrasting**


To provide the reasoning system with a mechanism of contrasting, we propose to explicitly build two
levels of contrast: model-level contrast and objective-level contrast.


**3.1.1** **Model-level Contrast**


As the central notion of contrast is comparing cases [ 30, 32, 44, 45 ], we propose an explicit modellevel contrasting mechanism in the following form,


_̸_



_̸_


�


_̸_



_̸_


Contrast( _F_ _O∪a_ ) = _F_ _O∪a_ _−_ _h_


_̸_



_̸_


_F_ _O∪a_ _′_

�� _a_ _[′]_ _∈A_


_̸_



_̸_


_,_ (2)


_̸_



_̸_


where _F_ denotes features of a specific combination and _h_ ( _·_ ) summarizes the common features in all
candidate answers. In our experiments, _h_ ( _·_ ) is a composition of BatchNorm [67] and Conv.


Intuitively, this explicit contrasting computation enables a reasoning system to tell distinguishing
features for each candidate in terms of fitting and following the rules hidden among all panels in
the incomplete matrix. The philosophy behind this design is to constrain the functional form of the
model to capture both the commonality and the difference in each instance. It is expected that the
very inductive bias on comparing similarity and distinctness is baked into the entire reasoning system
such that learning in the challenging task becomes easier.


In a generalized setting, each _O ∪_ _a_ could be abstracted out as an object. Then the design becomes a
general contrast module, where each object is distinguished by comparing with the common features
extracted from an object set.


We further note that the contrasting computation can be encapsulated into a single neural module
and repeated: the addition and transformation are shared and the subtraction is performed on each
individual element. See Figure 1(c) for a sketch of the contrast module. After such operations,
permutation invariance of a model will not be broken.


**3.1.2** **Objective-level Contrast**


To further enforce the contrast effects, we propose to use an NCE variant rather than the cross-entropy
loss commonly used in previous works [ 12, 14 ]. While there are several ways to model the probability
in Equation 1, we use a Gibbs distribution in this work:


_p_ ( _a|O_ ) = [1] (3)

_Z_ [exp(] _[f]_ [(] _[O ∪]_ _[a]_ [))] _[,]_


where _Z_ is the partition function, and our model _f_ ( _·_ ) corresponds to the negative potential function.
Note that such a distribution has been widely adopted in image generation models [68–70].


In this case, we can take the log of both sides in Equation 1 and rearrange terms:


log _p_ ( _a_ _⋆_ _|O_ ) _−_ log _p_ ( _a_ _[′]_ _|O_ ) = _f_ ( _O ∪_ _a_ _⋆_ ) _−_ _f_ ( _O ∪_ _a_ _[′]_ ) _≥_ 0 _,_ _∀a_ _[′]_ _∈A, a_ _[′]_ = _̸_ _a_ _⋆_ _._ (4)


4


This formulation could potentially lead to a max margin loss. However, we notice in our preliminary
experiments that max margin is not sufficient; we realize it is inferior to make the negative potential
of the wrong choices only _slightly lower_ . Instead, we would like to further push the difference to
_infinity_ . To do that, we leverage the _sigmoid_ function _σ_ ( _·_ ) and train the model, such that:


_f_ ( _O ∪_ _a_ _⋆_ ) _−_ _f_ ( _O ∪_ _a_ _[′]_ ) _→∞⇐⇒_ _σ_ ( _f_ ( _O ∪_ _a_ _⋆_ ) _−_ _f_ ( _O ∪_ _a_ _[′]_ )) _→_ 1 _, ∀a_ _[′]_ _∈A, a_ _[′]_ = _̸_ _a_ _⋆_ _._ (5)


However, we notice that the relative difference of negative potential is still problematic. We hypothesize this deficiency is due to the lack of a baseline—without such a regularization, the negative
potential of wrong choices could still be very high, resulting in difficulties in learning the negative
potential of the correct answer. To this end, we modify Equation 5 into its sufficient conditions:


_f_ ( _O ∪_ _a_ _⋆_ ) _−_ _b_ ( _O ∪_ _a_ _⋆_ ) _→∞⇐⇒_ _σ_ ( _f_ ( _O ∪_ _a_ _⋆_ ) _−_ _b_ ( _O ∪_ _a_ _⋆_ )) _→_ 1 (6)

_f_ ( _O ∪_ _a_ _[′]_ ) _−_ _b_ ( _O ∪_ _a_ _[′]_ ) _→−∞⇐⇒_ _σ_ ( _f_ ( _O ∪_ _a_ _[′]_ ) _−_ _b_ ( _O ∪_ _a_ _[′]_ )) _→_ 0 _,_ (7)


where _b_ ( _·_ ) is a fixed baseline function and _a_ _[′]_ _∈A, a_ _[′]_ = _̸_ _a_ _⋆_ . For implementation, _b_ ( _·_ ) could be either
a randomly initialized network or a constant. Since the two settings do not produce significantly
different results in our preliminary experiments, we set _b_ ( _·_ ) to be a constant to reduce computation.


We then optimize the network to maximize the following objective as done in [39]:


_ℓ_ = log( _σ_ ( _f_ ( _O ∪_ _a_ _⋆_ ) _−_ _b_ ( _O ∪_ _a_ _⋆_ ))) + � log(1 _−_ _σ_ ( _f_ ( _O ∪_ _a_ _[′]_ ) _−_ _b_ ( _O ∪_ _a_ _[′]_ ))) _._ (8)

_a_ _[′]_ _∈A,a_ _[′]_ = _̸_ _a_ _⋆_


**Connection to NCE** If we treat the baseline as the negative potential of a fixed noise model of the
same Gibbs form and ignore the difference between the partition functions, Equation 6 and Equation 7
become the _G_ function used in NCE [ 39 ]. But unlike NCE, we do not need to multiply the size ratio
in the sigmoid function [41].


**3.2** **Perceptual Inference**


As indicated in Carpenter et al. [17], a mere perceptive model for RPM is arguably not enough.
Therefore, we propose to incorporate a simple inference subsystem into the model: the inference
branch should be responsible for inferring the hidden rules in the problem. Specifically, we assume
there are at most _N_ attributes in each problem, each of which is subject to the governance of one of
_M_ rules. Then hidden rules _T_ in one problem instance can be decomposed into


_̸_



_̸_


_̸_


_̸_


_p_ ( _T |O_ ) =


_̸_



_̸_


_̸_


_̸_


_N_
� _p_ ( _t_ _i_ _|O_ ) _,_ (9)


_i_ =1


_̸_



_̸_


_̸_


_̸_


where _t_ _i_ = 1 _. . . M_ denotes the rule type on attribute _n_ _i_ . For the actual form of the probability of
rules on each attribute, we propose to model it using a multinomial distribution. This assumption is
consistent with the way datasets are usually generated [ 12, 14, 61 ]: one rule is independently picked
from the rule set for each attribute. In this way, each rule could also be regarded as a basis in a rule
dictionary and jointly learned, as done in active basis [42] or word embedding [49, 71].


If we treat rules as hidden variables, the log probability in Equation 4 can be decomposed into


log _p_ ( _a|O_ ) = log � _p_ ( _a|T, O_ ) _p_ ( _T |O_ ) = log E _T ∼p_ ( _T |O_ ) [ _p_ ( _a|T, O_ )] _._ (10)

_T_


Note that writing the summation in the form of expectation affords sampling algorithms, which can
be done on each individual attribute due to the independence assumption.


In addition, if we model _p_ ( _T |O_ ) as an inference branch _g_ ( _·_ ) and sample only once from it, the model
can be modified into _f_ ( _O ∪_ _a,_ _T_ [ˆ] ) with _T_ [ˆ] sampled from _g_ ( _O_ ) . Following the same derivation above,
we now optimize the new objective:


_ℓ_ = log( _σ_ ( _f_ ( _O ∪_ _a_ _⋆_ _,_ _T_ [ˆ] ) _−_ _b_ ( _O ∪_ _a_ _⋆_ ))) + � log(1 _−_ _σ_ ( _f_ ( _O ∪_ _a_ _[′]_ _,_ _T_ [ˆ] ) _−_ _b_ ( _O ∪_ _a_ _[′]_ ))) _._ (11)

_a_ _[′]_ _∈A,a_ _[′]_ = _̸_ _a_ _⋆_


To sample from a multinomial, we could either use hard sampling like Gumbel-SoftMax [ 72, 73 ] or a
soft one by taking expectation. We do not observe significant difference between the two settings.


5


The expectation in Equation 10 is proposed primarily to make the computation of the exact log
probability controllable and tractable: while the full summation requires _O_ ( _M_ _[N]_ ) passes of the model,
a Monte Carlo approximation of it could be calculated in _O_ (1) time. We also note that if _p_ ( _T |O_ ) is
highly peaked ( _e.g_ ., ground truth), the Monte Carlo estimate could be accurate as well. Despite the
fact that we only sample once from an inference branch to reduce computation, we find in practice
the Monte Carlo estimate works quite well.


**3.3** **Architecture**


Combining contrasting, perceptual inference, and permutation invariance, we propose a new network
architecture to solve the challenging RPM problem, named _Contrastive Perceptual Inference_ network
(CoPINet). The perception branch is composed of a common feature encoder and shared interweaving
contrast modules and residual blocks [ 3 ]. The encoder first extracts image features independently for
each panel and sum ones in the corresponding rows and columns before the final transformation into
a latent space. The inference branch consists of the same encoder and a (Gumbel-)SoftMax output
layer. The sampled results will be transformed and concatenated channel-wise into the summation in
Equation 2. In our implementation, we prepend each residual block with a contrast module; such a
combination can be repeated while keeping the network permutation-invariant. The network finally
uses an MLP to produce a negative potential for each observation and candidate pair and is trained
using Equation 11; see Figure 1(b) for a graphical illustration of the entire CoPINet architecture.


**4** **Experiments**


**4.1** **Experimental Setup**


We verify the effectiveness of our models on two major RPM datasets: RAVEN [ 12 ] and PGM [ 14 ].
Across all experiments, we train models on the training set, tune hyper-parameters on the validation
set, and report the final results on the test set. All of the models are implemented in PyTorch [ 74 ]
and optimized using ADAM [ 75 ]. While a good performance of WReN [ 14 ] and ResNet+DRT [ 12 ]
relies on external supervision, such as rule specifications and structural annotations, the proposed
model achieves better performance with only _O_, _A_, and _a_ _⋆_ . Models are trained on servers with
four Nvidia RTX Titans. For the WReN model, we use a public implementation that reproduces
results in [ 14 ] [1] . We implement our models in PyTorch [ 74 ] and optimize using ADAM [ 75 ]. During
training, we perform early-stop based on validation loss. We use the same network architecture and
hyper-parameters in both RAVEN and PGM experiments.


**4.2** **Results on RAVEN**


There are 70 _,_ 000 problems in the RAVEN dataset [ 12 ], equally distributed in 7 figure configurations.
In each configuration, the dataset is randomly split into 6 folds for training, 2 folds for validation,
and 2 folds for testing. We compare our model with several simple baselines (LSTM [ 76 ], CNN [ 77 ],
and vanilla ResNet [ 3 ]) and two strong baselines ( WReN [ 14 ] and ResNet+DRT [ 12 ]). Model
performance is measured by accuracy.


**General Performance on RAVEN** In this experiment, we train the models on all 42 _,_ 000 training
samples and measure how they perform on the test set. The first part of Table 1 shows the testing
accuracy of all models. We also retrieve the performance of humans and a solver with perfect
information from [ 12 ] for comparison. As shown in the table, the proposed model CoPINet achieves
the best performance among all the models we test. For the relational model WReN proposed in [ 14 ],
we run the tests on a permutation-invariant version, _i.e_ ., one without positional tagging (NoTag),
and tune the model also to minimize an auxiliary loss (Aux) [ 14 ]. While the auxiliary loss could
boost the performance of WReN as we will show later in the ablation study, we do not observe
similar effects on CoPINet. As indicated in the detailed comparisons in Table 1, WReN is biased
towards images of grid configurations and does poorly on ones demanding compositional reasoning,
_i.e_ ., ones with independent components. We further note that compared to previously proposed
models ( WReN [ 14 ] and ResNet+DRT [ 12 ]), CoPINet does not require additional information such
as structural annotations and meta targets and still shows human-level performance in this task. When


1 `[https://github.com/Fen9/WReN](https://github.com/Fen9/WReN)`


6


comparing the performance of CoPINet and human on specific figure configurations, we notice that
CoPINet is inferior in learning samples of grid-like compositionality but efficient in distinguishing
images consisting of multiple components, implying the efficiency of the contrasting mechanism.


**Ablation Study** One problem of particular interest in building CoPINet is how each component
contributes to performance improvement. To answer this question, we measure model accuracy by
gradually removing each construct in CoPINet, _i.e_ ., the perceptual inference branch, the contrast
loss, and the contrast module. In the second part of Table 1, we show the results of ablation on
CoPINet. Both the full model (CoPINet) and the one without the perceptual inference branch
(CoPINet-Contrast-CL) could achieve human-level performance, with the latter slightly inferior to the
former. If we further replace the contrast loss with the cross-entropy loss (CoPINet-Contrast-XE), we
observe a noticeable performance decrease of around 4%, verifying the effectiveness of the contrast
loss. A catastrophic performance downgrade of 66% is observed if we remove the contrast module,
leaving only the network backbone (CoPINet-Backbone-XE). This drastic performance gap shows
that the functional constraint on modeling an explicit contrasting mechanism is arguably a crucial
factor in machines’ reasoning ability as well as in humans’. The ablation study shows that all the three
proposed constructs, especially the contrast module, are critical to the performance of CoPINet. We
also study how the requirement of permutation invariance and auxiliary training affect the previously
proposed WReN . As shown in Table 1, sacrificing the permutation invariance (Tag) provides the
model a huge upgrade during auxiliary training (Aux), compared to the one without tagging (NoTag)
and auxiliary loss (NoAux). This effect becomes even more significant on the PGM dataset, as we
will show in Section 4.3.


**Dataset Size and Performance** Even though CoPINet surpasses human performance on RAVEN,
this competition is inherently unfair, as the human subjects in this study never experience such an
intensive training session as our model does. To make the comparison fairer and also as a step towards
a model capable of human learning efficiency, we further measure how the model performance
changes as the training set size shrinks. To this end, we train our CoPINet on subsets of the full
RAVEN training set and test it on the full test set. As shown on Table 2 and Figure 2, the model
performance varies roughly log-linearly with the training set size. One surprising observation is:
with only half of the amount of the data, we could already achieve human-level performance. On a
training set 16 _×_ smaller, CoPINet outperforms all previous models. And on a subset 64 _×_ smaller,
CoPINet already outshines WReN.


**4.3** **Results on PGM**


We use the neutral regime of the PGM dataset for model evaluation due to its diversity and richness in
relationships, objects, and attributes. This split of the dataset has in total 1 _._ 42 million samples, with
1 _._ 2 million for training, 2 _,_ 000 for validation, and 200 _,_ 000 for testing. We train the models on the
training set, tune the hyperparameters on the validation set, and evaluate the performance on the test


Table 1: Testing accuracy of models on RAVEN. Acc denotes the mean accuracy of each model. Same as in [ 12 ],
L-R denotes the Left-Right configuration, U-D Up-Down, O-IC Out-InCenter, and O-IG Out-InGrid.


Method Acc Center 2x2Grid 3x3Grid L-R U-D O-IC O-IG


LSTM 13 _._ 07% 13 _._ 19% 14 _._ 13% 13 _._ 69% 12 _._ 84% 12 _._ 35% 12 _._ 15% 12 _._ 99%
WReN-NoTag-Aux 17 _._ 62% 17 _._ 66% 29 _._ 02% 34 _._ 67% 7 _._ 69% 7 _._ 89% 12 _._ 30% 13 _._ 94%
CNN 36 _._ 97% 33 _._ 58% 30 _._ 30% 33 _._ 53% 39 _._ 43% 41 _._ 26% 43 _._ 20% 37 _._ 54%
ResNet 53 _._ 43% 52 _._ 82% 41 _._ 86% 44 _._ 29% 58 _._ 77% 60 _._ 16% 63 _._ 19% 53 _._ 12%
ResNet+DRT 59 _._ 56% 58 _._ 08% 46 _._ 53% 50 _._ 40% 65 _._ 82% 67 _._ 11% 69 _._ 09% 60 _._ 11%
CoPINet **91** _._ **42** % **95** _._ **05** % **77** _._ **45** % **78** _._ **85** % **99** _._ **10** % **99** _._ **65** % **98** _._ **50** % **91** _._ **35** %


WReN-NoTag-NoAux 15 _._ 07% 12 _._ 30% 28 _._ 62% 29 _._ 22% 7 _._ 20% 6 _._ 55% 8 _._ 33% 13 _._ 10%
WReN-Tag-NoAux 17 _._ 94% 15 _._ 38% 29 _._ 81% 32 _._ 94% 11 _._ 06% 10 _._ 96% 11 _._ 06% 14 _._ 54%
WReN-Tag-Aux 33 _._ 97% 58 _._ 38% 38 _._ 89% 37 _._ 70% 21 _._ 58% 19 _._ 74% 38 _._ 84% 22 _._ 57%
CoPINet-Backbone-XE 20 _._ 75% 24 _._ 00% 23 _._ 25% 23 _._ 05% 15 _._ 00% 13 _._ 90% 21 _._ 25% 24 _._ 80%
CoPINet-Contrast-XE 86 _._ 16% 87 _._ 25% 71 _._ 05% 74 _._ 45% 97 _._ 25% 97 _._ 05% 93 _._ 20% 82 _._ 90%
CoPINet-Contrast-CL 90 _._ 04% 94 _._ 30% 74 _._ 00% 76 _._ 85% 99 _._ 05% 99 _._ 35% 98 _._ 00% 88 _._ 70%


Human 84 _._ 41% 95 _._ 45% 81 _._ 82% 79 _._ 55% 86 _._ 36% 81 _._ 81% 86 _._ 36% 81 _._ 81%
Solver 100% 100% 100% 100% 100% 100% 100% 100%


7


Figure 2: CoPINet on RAVEN
and PGM as the training set size
shrinks.



Table 2: Model performance under
different training set sizes on RAVEN
dataset. The full training set has
42 _,_ 000 samples.


Training set size Acc


658 44 _._ 48%
1 _,_ 316 57 _._ 69%
2 _,_ 625 65 _._ 55%
5 _,_ 250 74 _._ 53%
10 _,_ 500 80 _._ 92%
21 _,_ 000 86 _._ 43%



Table 3: Model performance under
different training set sizes on PGM
dataset. The full training set has 1 _._ 2
million samples.


Training set size Acc


293 14 _._ 73%
1 _,_ 172 15 _._ 48%
4 _,_ 688 18 _._ 39%
18 _,_ 750 22 _._ 07%
75 _,_ 000 32 _._ 39%
300 _,_ 000 43 _._ 89%



**80**


**60**


**40**


**20**




|RAVEN<br>PGM|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||



**10** **[2]** **10** **[4]** **10** **[6]**

**Training set size**



Table 4: Testing accuracy of models on PGM. Acc denotes the mean accuracy of each model.

Method CNN LSTM ResNet Wild-ResNet WReN-NoTag-Aux CoPINet


Acc 33 _._ 00% 35 _._ 80% 42 _._ 00% 48 _._ 00% 49 _._ 10% **56** _._ **37** %


set. We compare our models with baselines set up in [ 14 ], _i.e_ ., LSTM, CNN, ResNet, Wild-ResNet,
and WReN . As ResNet+DRT proposed in [ 12 ] requires structural annotations not available in PGM,
we are unable to measure its performance. Again, all performance is measured by accuracy. Due to
the lack of further stratification on this training regime, we only report the final mean accuracy.


**General Performance on PGM** In this experiment, we train the models on all 1 _._ 2 million training
samples and report performance on the entire test set. As shown in Table 4, CoPINet achieves the best
performance among all permutation-invariant models, setting a new state-of-the-art on this dataset.
Similar to the setting in RAVEN, we make the previously proposed WReN permutation-invariant by
removing the positional tagging (NoTag) and train it with both cross-entropy loss and auxiliary loss
(Aux) [ 14 ]. The auxiliary loss could boost the performance of WReN . However, in coherence with
the study on RAVEN and a previous work [ 12 ], we notice that the auxiliary loss does not help our
CoPINet. It is worth noting that while WReN demands additional training supervision from meta
targets to reach the performance, CoPINet only requires basic annotations of ground truth indices _a_ _⋆_
and achieves better results.


**Ablation Study** We perform ablation studies on both WReN and CoPINet to see how the requirement of permutation invariance affects WReN and how each module in CoPINet contributes to its
superior performance. The notations are the same as those used in the ablation study for RAVEN.
As shown in the first part of Table 5, adding a proper auxiliary loss does provide WReN a 10%
performance boost. However, additional supervision is required. Making the model permutationsensitive gives the model a significant benefit by up to a 28% accuracy increase; however, it also
indicates that WReN learns to shortcut the solutions by coding the positional association, instead
of truly understanding the differences among distinctive choices and their potential effects on the
compatibility of the entire matrix. The second part of Table 5 demonstrates how each construct
contributes to the performance improvement of CoPINet on PGM. Despite the smaller enhancement
of the contrast loss compared to that in RAVEN, the upgrade from the contrast module for PGM is still
significant, and the perceptual inference branch keeps raising the final performance. In accordance
with the ablation study on the RAVEN dataset, we show that all the proposed components contribute
to the final performance increase.


Table 5: Ablation study on PGM.

Method WReN-NoTag-NoAux WReN-NoTag-Aux WReN-Tag-NoAux WReN-Tag-Aux


Acc 39 _._ 25% 49 _._ 10% 62 _._ 45% 77 _._ 94%


Method CoPINet-Backbone-XE CoPINet-Contrast-XE CoPINet-Contrast-CL CoPINet


Acc 42 _._ 10% 51 _._ 04% 54 _._ 19% 56 _._ 37%


8


**Dataset Size and Performance** Motivated by the idea of fairer comparison and low-shot reasoning,
we also measure how the performance of the proposed CoPINet changes as the training set size of
PGM varies. Specifically, we train CoPINet on subsets of the PGM training set and test it on the
entire test set. As shown in Table 3 and Figure 2, CoPINet performance on PGM varies roughly
log-exponentially with respect to the training set size. We further note that when trained on a 16 _×_
smaller dataset, CoPINet already achieves results similar to CNN and LSTM.


**5** **Conclusion and Discussion**


In this work, we aim to improve machines’ reasoning ability in “thinking in pictures” by jointly
learning perception and inference via contrasting. Specifically, we introduce the contrast module,
the contrast loss, and the joint system of perceptual inference. We also require our model to be
permutation-invariant. In a typical and challenging task of this kind, Raven’s Progressive Matrices ( RPM ), we demonstrate that our proposed model— _Contrastive Perceptual Inference_ network
(CoPINet)—achieves the new state-of-the-art for permutation-invariant models on two major RPM
datasets. Further ablation studies show that all the three proposed components are effective towards
improving the final results, especially the contrast module. It also shows that the permutation invariance forces the model to understand the effects of different choices on the compatibility of an entire
RPM matrix, rather than remembering the positional association and shortcutting the solutions.


While it is encouraging to see the performance improvement of the proposed ideas on two big datasets,
it is the last part of the experiments, _i.e_ ., dataset size and performance, that really intrigues us. With
infinitely large datasets that cover the entirety of an arbitrarily complex problem domain, it is arguably
possible that a simple over-parameterized model could solve it. However, in reality, there is barely
any chance that one would observe all the domain, yet humans still learn quite efficiently how the
hidden rules work. We believe this is the core where the real intelligence lies: learning from only
a few samples and generalizing to the extreme. Even though CoPINet already demonstrates better
learning efficiency, it would be ideal to have models capable of few-shot learning in the task of RPM .
Without massive datasets, it would be a real challenge, and we hope the paper could call for future
research into it.


Performance, however, is definitely not the end goal in the line of research on relational and analogical
visual reasoning: other dimensions for measurements include generalization, generability, and
transferability. Is it possible for a model to be trained on a single configuration and generalize to
other settings? Can we generate the final answer based on the given context panels, in a similar way
to the top-down and bottom-up method jointly applied by humans for reasoning? Can we transfer
the relational and geometric knowledge required in the reasoning task from other tasks? Questions
like these are far from being answered. While Zhang et al. [12] show in the experiments that neural
models do possess a certain degree of generalizability, the testing accuracy is far from satisfactory. In
the meantime, there are a plethora of discriminative approaches towards solving reasoning problems
in question answering, but generative methods and combined methods are lacking. The relational and
analogical reasoning was initially introduced as a way to measure a human’s intelligence, without
training humans on the task. However, current settings uniformly reformulate it as a learning problem
rather than a transfer problem, contradictory to why the task was started. Up to now, there has been
barely any work that measures how knowledge on another task could be transferred to this one. We
believe that significant advances in these dimensions would possibly enable Artificial Intelligence ( AI )
models to go beyond data fitting and acquire symbolized knowledge.


While modern computer vision techniques to solve Raven’s Progressive Matrices ( RPM ) are based
on neural networks, a promising ingredient is nowhere to be found: Gestalt psychology. Traces of the
perceptual grouping and figure-ground organization are gradually faded out in the most recent wave
of deep learning. However, the principles of grouping, both classical ( _e.g_ ., proximity, closure, and
similarity) and new ( _e.g_ ., synchrony, element, and uniform connectedness) play an essential role in
RPM, as humans arguably solve these problems by first figuring out groups and then applying the
rules. We anticipate that modern deep learning methods integrated with the tradition of conceptual and
theoretical foundations of the Gestalt approach would further improve models on abstract reasoning
tasks like RPM.


**Acknowledgments:** This work reported herein is supported by MURI ONR N00014-16-1-2007,
DARPA XAI N66001-17-2-4029, ONR N00014-19-1-2153, NSF BSC-1827374, and an NVIDIA
GPU donation grant.


9


**References**


[1] Temple Grandin. _Thinking in pictures: And other reports from my life with autism_ . Vintage,
2006.


[2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep
convolutional neural networks. In _Proceedings of Advances in Neural Information Processing_
_Systems (NIPS)_, 2012.


[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, 2016.


[4] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified,
real-time object detection. In _Proceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition (CVPR)_, 2016.


[5] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object
detection with region proposal networks. In _Proceedings of Advances in Neural Information_
_Processing Systems (NIPS)_, 2015.


[6] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra,
C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings_
_of International Conference on Computer Vision (ICCV)_, 2015.


[7] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick,
and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary
visual reasoning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, 2017.


[8] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei,
C Lawrence Zitnick, and Ross B Girshick. Inferring and executing programs for visual reasoning.
In _Proceedings of International Conference on Computer Vision (ICCV)_, 2017.


[9] Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Kate Saenko. Learning
to reason: End-to-end module networks for visual question answering. In _Proceedings of_
_International Conference on Computer Vision (ICCV)_, 2017.


[10] Rudolf Arnheim. _Visual thinking_ . Univ of California Press, 1969.


[11] Francis Galton. _Inquiries into human faculty and its development_ . Macmillan, 1883.


[12] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for
relational and analogical visual reasoning. In _Proceedings of the IEEE Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, 2019.


[13] Felix Hill, Adam Santoro, David GT Barrett, Ari S Morcos, and Timothy Lillicrap. Learning to
make analogies by contrasting abstract relational structure. _arXiv:1902.00120_, 2019.


[14] David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measuring abstract
reasoning in neural networks. In _Proceedings of International Conference on Machine Learning_
_(ICML)_, 2018.


[15] James C Raven. Mental tests used in genetic studies: The performance of related individuals on
tests mainly educative and mainly reproductive. Master’s thesis, University of London, 1936.


[16] John C Raven and John Hugh Court. _Raven’s progressive matrices and vocabulary scales_ .
Oxford pyschologists Press, 1998.


[17] Patricia A Carpenter, Marcel A Just, and Peter Shell. What one intelligence test measures:
a theoretical account of the processing in the raven progressive matrices test. _Psychological_
_review_, 97(3):404, 1990.


[18] R E Snow, Patrick Kyllonen, and B Marshalek. The topography of ability and learning
correlations. _Advances in the psychology of human intelligence_, pages 47–103, 1984.


[19] Charles Spearman. _The abilities of man_ . Macmillan, 1927.


10


[20] Charles Spearman. _The nature of "intelligence" and the principles of cognition_ . Macmillan,
1923.


[21] Douglas R Hofstadter. _Fluid concepts and creative analogies: Computer models of the funda-_
_mental mechanisms of thought._ Basic books, 1995.


[22] Susanne M Jaeggi, Martin Buschkuehl, John Jonides, and Walter J Perrig. Improving fluid
intelligence with training on working memory. _Proceedings of the National Academy of Sciences_,
105(19):6829–6833, 2008.


[23] James J Gibson and Eleanor J Gibson. Perceptual learning: Differentiation or enrichment?
_Psychological review_, 62(1):32, 1955.


[24] James J Gibson. _The ecological approach to visual perception: classic edition_ . Psychology
Press, 2014.


[25] Richard Catrambone and Keith J Holyoak. Overcoming contextual limitations on problemsolving transfer. _Journal of Experimental Psychology: Learning, Memory, and Cognition_, 15
(6):1147, 1989.


[26] Dedre Gentner and Virginia Gunn. Structural alignment facilitates the noticing of differences.
_Memory & Cognition_, 29(4):565–577, 2001.


[27] Rubi Hammer, Gil Diesendruck, Daphna Weinshall, and Shaul Hochstein. The development of
category learning strategies: What makes the difference? _Cognition_, 112(1):105–119, 2009.


[28] Mary L Gick and Katherine Paterson. Do contrasting examples facilitate schema acquisition
and analogical transfer? _Canadian Journal of Psychology/Revue canadienne de psychologie_, 46
(4):539, 1992.


[29] Etsuko Haryu, Mutsumi Imai, and Hiroyuki Okada. Object similarity bootstraps young children
to action-based verb extension. _Child Development_, 82(2):674–686, 2011.


[30] Linsey Smith and Dedre Gentner. The role of difference-detection in learning contrastive
categories. In _Proceedings of the Annual Meeting of the Cognitive Science Society (CogSci)_,
2014.


[31] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. _Cognitive science_, 7
(2):155–170, 1983.


[32] Daniel L Schwartz, Catherine C Chase, Marily A Oppezzo, and Doris B Chin. Practicing versus
inventing with contrasting cases: The effects of telling first on learning and transfer. _Journal of_
_Educational Psychology_, 103(4):759, 2011.


[33] Gordon H Bower. A contrast effect in differential conditioning. _Journal of Experimental_
_Psychology_, 62(2):196, 1961.


[34] Donald R Meyer. The effects of differential rewards on discrimination reversal learning by
monkeys. _Journal of Experimental Psychology_, 41(4):268, 1951.


[35] Allan M Schrier and Harry F Harlow. Effect of amount of incentive on discrimination learning
by monkeys. _Journal of comparative and physiological psychology_, 49(2):117, 1956.


[36] Robert M Shapley and Jonathan D Victor. The effect of contrast on the transfer properties of cat
retinal ganglion cells. _The Journal of physiology_, 285(1):275–298, 1978.


[37] Reed Lawson. Brightness discrimination performance and secondary reward strength as a
function of primary reward amount. _Journal of Comparative and Physiological Psychology_, 50
(1):35, 1957.


[38] Abram Amsel. Frustrative nonreward in partial reinforcement and discrimination learning:
Some recent history and a theoretical extension. _Psychological review_, 69(4):306, 1962.


[39] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation
principle for unnormalized statistical models. In _Proceedings of the International Conference_
_on Artificial Intelligence and Statistics (AISTATS)_, 2010.


11


[40] Noah A Smith and Jason Eisner. Contrastive estimation: Training log-linear models on unlabeled
data. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics_
_(ACL)_, 2005.


[41] Bo Dai and Dahua Lin. Contrastive learning for image captioning. In _Proceedings of Advances_
_in Neural Information Processing Systems (NIPS)_, 2017.


[42] Ying Nian Wu, Zhangzhang Si, Haifeng Gong, and Song-Chun Zhu. Learning active basis
model for object detection and recognition. _International Journal of Computer Vision (IJCV)_,
90(2):198–235, 2010.


[43] Dedre Gentner and Arthur B Markman. Structural alignment in comparison: No difference
without similarity. _Psychological science_, 5(3):152–158, 1994.


[44] Catherine C Chase, Jonathan T Shemwell, and Daniel L Schwartz. Explaining across contrasting cases for deep understanding in science: An example using interactive simulations. In
_Proceedings of the 9th International Conference of the Learning Sciences_, 2010.


[45] Daniel L Schwartz and Taylor Martin. Inventing to prepare for future learning: The hidden
efficiency of encouraging original student production in statistics instruction. _Cognition and_
_Instruction_, 22(2):129–184, 2004.


[46] Sumit Chopra, Raia Hadsell, Yann LeCun, et al. Learning a similarity metric discriminatively,
with application to face verification. In _Proceedings of the IEEE Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, 2005.


[47] Kilian Q Weinberger and Lawrence K Saul. Distance metric learning for large margin nearest
neighbor classification. _Journal of Machine Learning Research_, 10(Feb):207–244, 2009.


[48] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using
videos. In _Proceedings of International Conference on Computer Vision (ICCV)_, 2015.


[49] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In _Proceedings of Advances in Neural_
_Information Processing Systems (NIPS)_, 2013.


[50] Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio
Torralba, and Sanja Fidler. Skip-thought vectors. In _Proceedings of Advances in Neural_
_Information Processing Systems (NIPS)_, 2015.


[51] David J Chalmers, Robert M French, and Douglas R Hofstadter. High-level perception, representation, and analogy: A critique of artificial intelligence methodology. _Journal of Experimental_
_& Theoretical Artificial Intelligence_, 4(3):185–211, 1992.


[52] Melanie Mitchell. _Analogy-making as perception: A computer model_ . MIT Press, 1993.


[53] Andrew Lovett and Kenneth Forbus. Modeling visual problem solving as analogical reasoning.
_Psychological Review_, 124(1):60, 2017.


[54] Andrew Lovett, Kenneth Forbus, and Jeffrey Usher. A structure-mapping model of raven’s
progressive matrices. In _Proceedings of the Annual Meeting of the Cognitive Science Society_
_(CogSci)_, 2010.


[55] Andrew Lovett, Emmett Tomai, Kenneth Forbus, and Jeffrey Usher. Solving geometric analogy
problems through two-stage analogical mapping. _Cognitive science_, 33(7):1192–1231, 2009.


[56] Daniel R Little, Stephan Lewandowsky, and Thomas L Griffiths. A bayesian model of rule
induction in raven’s progressive matrices. In _Proceedings of the Annual Meeting of the Cognitive_
_Science Society (CogSci)_, 2012.


[57] Keith McGreggor and Ashok K Goel. Confident reasoning on raven’s progressive matrices tests.
In _Proceedings of AAAI Conference on Artificial Intelligence (AAAI)_, 2014.


[58] Keith McGreggor, Maithilee Kunda, and Ashok Goel. Fractals and ravens. _Artificial Intelligence_,
215:1–23, 2014.


12


[59] Can Serif Mekik, Ron Sun, and David Yun Dai. Similarity-based reasoning, raven’s matrices, and general intelligence. In _Proceedings of International Joint Conference on Artificial_
_Intelligence (IJCAI)_, 2018.


[60] Snejana Shegheva and Ashok Goel. The structural affinity method for solving the raven’s
progressive matrices test for intelligence. In _Proceedings of AAAI Conference on Artificial_
_Intelligence (AAAI)_, 2018.


[61] Ke Wang and Zhendong Su. Automatic generation of raven’s progressive matrices. In _Proceed-_
_ings of International Joint Conference on Artificial Intelligence (IJCAI)_, 2015.


[62] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter
Battaglia, and Tim Lillicrap. A simple neural network module for relational reasoning. In
_Proceedings of Advances in Neural Information Processing Systems (NIPS)_, 2017.


[63] Xander Steenbrugge, Sam Leroux, Tim Verbelen, and Bart Dhoedt. Improving generalization for abstract reasoning tasks using disentangled feature representations. _arXiv preprint_
_arXiv:1811.04784_, 2018.


[64] Song-Chun Zhu, David Mumford, et al. A stochastic grammar of images. _Foundations and_
_Trends⃝_ R _in Computer Graphics and Vision_, 2(4):259–362, 2007.


[65] Seyoung Park and Song-Chun Zhu. Attributed grammars for joint estimation of human attributes,
part and pose. In _Proceedings of International Conference on Computer Vision (ICCV)_, 2015.


[66] Tian-Fu Wu, Gui-Song Xia, and Song-Chun Zhu. Compositional boosting for computing
hierarchical image structures. In _Proceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition (CVPR)_, 2007.


[67] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training
by reducing internal covariate shift. In _Proceedings of International Conference on Machine_
_Learning (ICML)_, 2015.


[68] Song Chun Zhu, Yingnian Wu, and David Mumford. Filters, random fields and maximum
entropy (frame): Towards a unified theory for texture modeling. _International Journal of_
_Computer Vision_, 27(2):107–126, 1998.


[69] Ying Nian Wu, Jianwen Xie, Yang Lu, and Song-Chun Zhu. Sparse and deep generalizations of
the frame model. _Annals of Mathematical Sciences and Applications_, 3(1):211–254, 2018.


[70] Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu. A theory of generative convnet. In
_Proceedings of International Conference on Machine Learning (ICML)_, 2016.


[71] Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word
representation. In _Proceedings of the conference on Empirical Methods in Natural Language_
_Processing (EMNLP)_, 2014.


[72] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax.
_arXiv:1611.01144_, 2016.


[73] Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous
relaxation of discrete random variables. _arXiv:1611.00712_, 2016.


[74] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito,
Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in
pytorch. In _NIPS-W_, 2017.


[75] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _International_
_Conference on Learning Representations (ICLR)_, 2014.


[76] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural computation_,
1997.


[77] Dokhyam Hoshen and Michael Werman. Iq of neural networks. _arXiv:1710.01692_, 2017.


13


