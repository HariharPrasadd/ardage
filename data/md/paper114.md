## **Latent Alignment and Variational Attention**

**Yuntian Deng** _[∗]_ **Yoon Kim** _[∗]_ **Justin Chiu** **Demi Guo** **Alexander M. Rush**

```
 {dengyuntian@seas,yoonkim@seas,justinchiu@g,dguo@college,srush@seas}.harvard.edu

```

School of Engineering and Applied Sciences
Harvard University
Cambridge, MA, USA


**Abstract**


Neural attention has become central to many state-of-the-art models in natural
language processing and related domains. Attention networks are an easy-to-train
and effective method for softly simulating alignment; however, the approach does
not marginalize over latent alignments in a probabilistic sense. This property makes
it difficult to compare attention to other alignment approaches, to compose it with
probabilistic models, and to perform posterior inference conditioned on observed
data. A related latent approach, hard attention, fixes these issues, but is generally
harder to train and less accurate. This work considers _variational attention_ networks, alternatives to soft and hard attention for learning latent variable alignment
models, with tighter approximation bounds based on amortized variational inference. We further propose methods for reducing the variance of gradients to make
these approaches computationally feasible. Experiments show that for machine
translation and visual question answering, inefficient exact latent variable models
outperform standard neural attention, but these gains go away when using hard
attention based training. On the other hand, variational attention retains most of
the performance gain but with training speed comparable to neural attention.


**1** **Introduction**


Attention networks [ 6 ] have quickly become the foundation for state-of-the-art models in natural
language understanding, question answering, speech recognition, image captioning, and more [ 15, 81,
16, 14, 63, 80, 71, 62 ]. Alongside components such as residual blocks and long-short term memory
networks, soft attention provides a rich neural network building block for controlling gradient flow
and encoding inductive biases. However, more so than these other components, which are often
treated as black-boxes, researchers use intermediate attention decisions directly as a tool for model
interpretability [ 43, 1 ] or as a factor in final predictions [ 25, 68 ]. From this perspective, attention
plays the role of a latent alignment variable [ 10, 37 ]. An alternative approach, hard attention [ 80 ],
makes this connection explicit by introducing a latent variable for alignment and then optimizing a
bound on the log marginal likelihood using policy gradients. This approach generally performs worse
(aside from a few exceptions such as [80]) and is used less frequently than its soft counterpart.


Still the latent alignment approach remains appealing for several reasons: (a) latent variables facilitate
reasoning about dependencies in a probabilistically principled way, e.g. allowing composition with
other models, (b) posterior inference provides a better basis for model analysis and partial predictions
than strictly feed-forward models, which have been shown to underperform on alignment in machine
translation [38], and finally (c) directly maximizing marginal likelihood may lead to better results.


_∗_ Equal contribution.


32nd Conference on Neural Information Processing Systems (NIPS 2018), Montréal, Canada.


The aim of this work is to quantify the issues with attention and propose alternatives based on recent
developments in variational inference. While the connection between variational inference and hard
attention has been noted in the literature [ 4, 41 ], the space of possible bounds and optimization
methods has not been fully explored and is growing quickly. These tools allow us to better quantify
whether the general underperformance of hard attention models is due to modeling issues (i.e. soft
attention imbues a better inductive bias) or optimization issues.



Our main contribution is a _variational attention_
approach that can effectively fit latent alignments while remaining tractable to train. We
consider two variants of variational attention:
_categorical_ and _relaxed_ . The categorical method
is fit with amortized variational inference using
a learned inference network and policy gradient
with a soft attention variance reduction baseline.
With an appropriate inference network (which
conditions on the entire source/target), it can be
used at training time as a drop-in replacement
for hard attention. The relaxed version assumes
that the alignment is sampled from a Dirichlet
distribution and hence allows attention over multiple source elements.



Figure 1: Sketch of variational attention applied to
machine translation. Two alignment distributions are
shown, the blue prior _p_, and the red variational posterior
_q_ taking into account future observations. Our aim is to
use _q_ to improve estimates of _p_ and to support improved
inference of _z_ .



_x_ 1: _T_









_x_ ˜ _y_ 3



Experiments describe how to implement this _p_

_q_ taking into account future observations. Our aim is to

approach for two major attention-based models:

use _q_ to improve estimates of _p_ and to support improved

neural machine translation and visual question
answering (Figure 1 gives an overview of our inference of _z_ .
approach for machine translation). We first show
that maximizing exact marginal likelihood can increase performance over soft attention. We further
show that with variational (categorical) attention, alignment variables significantly surpass both
soft and hard attention results without requiring much more difficult training. We further explore
the impact of posterior inference on alignment decisions, and how latent variable models might be
employed. Our code is available at `[https://github.com/harvardnlp/var-attn/](https://github.com/harvardnlp/var-attn/)` .


**Related Work** Latent alignment has long been a core problem in NLP, starting with the seminal IBM
models [ 11 ], HMM-based alignment models [ 75 ], and a fast log-linear reparameterization of the IBM
2 model [ 20 ]. Neural soft attention models were originally introduced as an alternative approach
for neural machine translation [ 6 ], and have subsequently been successful on a wide range of tasks
(see [ 15 ] for a review of applications). Recent work has combined neural attention with traditional
alignment [ 18, 72 ] and induced structure/sparsity [ 48, 33, 44, 85, 54, 55, 49 ], which can be combined
with the variational approaches outlined in this paper.


In contrast to soft attention models, hard attention [ 80, 3 ] approaches use a single sample at training
time instead of a distribution. These models have proven much more difficult to train, and existing
works typically treat hard attention as a black-box reinforcement learning problem with log-likelihood
as the reward [ 80, 3, 53, 26, 19 ]. Two notable exceptions are [ 4, 41 ]: both utilize amortized variational
inference to learn a sampling distribution which is used obtain importance-sampled estimates of the
log marginal likelihood [ 12 ]. Our method uses uses different estimators and targets the single sample
approach for efficiency, allowing the method to be employed for NMT and VQA applications.


There has also been significant work in using variational autoencoders for language and translation
application. Of particular interest are those that augment an RNN with latent variables (typically
Gaussian) at each time step [ 17, 22, 66, 23, 40 ] and those that incorporate latent variables into
sequence-to-sequence models [ 84, 7, 70, 64 ]. Our work differs by modeling an explicit model
component (alignment) as a latent variable instead of auxiliary latent variables (e.g. topics). The
term "variational attention" has been used to refer to a different component the output from attention
(commonly called the context vector) as a latent variable [ 7 ], or to model both the memory and the
alignment as a latent variable [ 9 ]. Finally, there is some parallel work [ 78, 67 ] which also performs
exact/approximate marginalization over latent alignments for sequence-to-sequence learning.


2


**2** **Background: Latent Alignment and Neural Attention**


We begin by introducing notation for latent alignment, and then show how it relates to neural attention.
For clarity, we are careful to use _alignment_ to refer to this probabilistic model (Section 2.1), and _soft_
and _hard_ attention to refer to two particular inference approaches used in the literature to estimate
alignment models (Section 2.2).


**2.1** **Latent Alignment**


Figure 2(a) shows a latent alignment model. Let _x_ be an observed set with associated members
_{x_ 1 _, . . ., x_ _i_ _, . . ., x_ _T_ _}_ . Assume these are vector-valued (i.e. _x_ _i_ _∈_ R _[d]_ ) and can be stacked to form a
matrix _X ∈_ R _[d][×][T]_ . Let the observed ˜ _x_ be an arbitrary “query”. These generate a discrete output
variable _y ∈Y_ . This process is mediated through a latent alignment variable _z_, which indicates
which member (or mixture of members) of _x_ generates _y_ . The generative process we consider is:


_z ∼D_ ( _a_ ( _x,_ ˜ _x_ ; _θ_ )) _y ∼_ _f_ ( _x, z_ ; _θ_ )


where _a_ produces the parameters for an alignment distribution _D_ . The function _f_ gives a distribution
over the output, e.g. an exponential family. To fit this model to data, we set the model parameters _θ_
by maximizing the log marginal likelihood of training examples ( _x,_ ˜ _x,_ ˆ _y_ ): [2]


max log _p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) = max log E _z_ [ _f_ ( _x, z_ ; _θ_ ) _y_ ˆ ]
_θ_ _θ_



Directly maximizing this log marginal likelihood in the presence of the latent variable _z_
is often difficult due to the expectation (though
tractable in certain cases).


For this to represent an alignment, we restrict
the variable _z_ to be in the simplex ∆ _[T][ −]_ [1] over
source indices _{_ 1 _, . . ., T_ _}_ . We consider two distributions for this variable: first, let _D_ be a _cat-_
_egorical_ where _z_ is a one-hot vector with _z_ _i_ = 1
if _x_ _i_ is selected. For example, _f_ ( _x, z_ ) could use
_z_ to pick from _x_ and apply a softmax layer to
predict _y_, i.e. _f_ ( _x, z_ ) = softmax( **W** _Xz_ ) and
**W** _∈_ R _[|Y|×][d]_,



(a)





(b)





Figure 2: Models over observed set _x_, query ˜ _x_, and
alignment _z_ . (a) Latent alignment model, (b) Soft attention with _z_ absorbed into prediction network.



log _p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) = log



_T_
� _p_ ( _z_ _i_ = 1 _| x,_ ˜ _x_ ) _p_ ( _y_ = ˆ _y | x, z_ _i_ = 1) = log E _z_ [softmax( **W** _Xz_ ) _y_ ˆ ]


_i_ =1



This computation requires a factor of _O_ ( _T_ ) additional runtime, and introduces a major computational
factor into already expensive deep learning models. [3]


Second we consider a _relaxed_ alignment where _z_ is a mixture taken from the interior of the simplex by
letting _D_ be a Dirichlet. This objective looks similar to the categorical case, i.e. log _p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) =
log E _z_ [softmax( **W** _Xz_ ) _y_ ˆ ], but the resulting expectation is intractable to compute exactly.


**2.2** **Attention Models: Soft and Hard**


When training deep learning models with gradient methods, it can be difficult to use latent alignment
directly. As such, two alignment-like approaches are popular: _soft attention_ replaces the probabilistic
model with a deterministic soft function and _hard attention_ trains a latent alignment model by
maximizing a lower bound on the log marginal likelihood (obtained from Jensen’s inequality) with
policy gradient-style training. We briefly describe how these methods fit into this notation.


2 When clear from context, the random variable is dropped from E[ _·_ ] . We also interchangeably use _p_ (ˆ _y | x,_ ˜ _x_ )
and _f_ ( _x, z_ ; _θ_ ) _y_ ˆ to denote _p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ).
3 Although not our main focus, explicit marginalization is sometimes tractable with efficient matrix operations
on modern hardware, and we compare the variational approach to explicit enumeration in the experiments. In
some cases it is also possible to efficiently perform exact marginalization with dynamic programming if one
imposes additional constraints (e.g. monotonicity) on the alignment distribution [83, 82, 58].


3


**Soft Attention** Soft attention networks use an altered model shown in Figure 2b. Instead of using a
latent variable, they employ a deterministic network to compute an expectation over the alignment
variable. We can write this model using the same functions _f_ and _a_ from above,


log _p_ soft ( _y | x,_ ˜ _x_ ) = log _f_ ( _x,_ E _z_ [ _z_ ]; _θ_ ) = log softmax( **W** _X_ E _z_ [ _z_ ])


A major benefit of soft attention is efficiency. Instead of paying a multiplicative penalty of _O_ ( _T_ )
or requiring integration, the soft attention model can compute the expectation before _f_ . While
formally a different model, soft attention has been described as an approximation of alignment [ 80 ].
Since E[ _z_ ] _∈_ ∆ _[T][ −]_ [1], soft attention uses a convex combination of the input representations _X_ E[ _z_ ]
(the _context vector_ ) to obtain a distribution over the output. While also a “relaxed” decision, this
expression differs from both the latent alignment models above. Depending on _f_, the gap between
E[ _f_ ( _x, z_ )] and _f_ ( _x,_ E[ _z_ ]) may be large.


However there are some important special cases. In the case where _p_ ( _z | x,_ ˜ _x_ ) is deterministic, we
have E[ _f_ ( _x, z_ )] = _f_ ( _x,_ E[ _z_ ]), and _p_ ( _y | x,_ ˜ _x_ ) = _p_ soft ( _y | x,_ ˜ _x_ ) . In general we can bound the absolute
difference based on the maximum curvature of _f_, as shown by the following proposition.
**Proposition 1.** _Define_ _g_ _x,y_ ˆ : ∆ _[T][ −]_ [1] _�→_ [0 _,_ 1] _to be the function given by_ _g_ _x,y_ ˆ ( _z_ ) = _f_ ( _x, z_ ) _y_ ˆ _(i.e._
_g_ _x,y_ ˆ ( _z_ ) = _p_ ( _y_ = ˆ _y | x,_ ˜ _x, z_ )) _for a twice differentiable function_ _f_ _. Let_ _H_ _g_ _x,y_ ˆ ( _z_ ) _be the Hessian of_
_g_ _x,y_ ˆ ( _z_ ) _evaluated at_ _z_ _, and further suppose_ _∥H_ _g_ _x,y_ ˆ ( _z_ ) _∥_ 2 _≤_ _c_ _for all_ _z ∈_ ∆ _[T][ −]_ [1] _,_ ˆ _y ∈Y_ _, and_ _x_ _, where_
_∥· ∥_ 2 _is the spectral norm. Then for all_ ˆ _y ∈Y,_


_| p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) _−_ _p_ soft ( _y_ = ˆ _y | x,_ ˜ _x_ ) _| ≤_ _c_


The proof is given in Appendix A. [4] Empirically the soft approximation works remarkably well, and
often moves towards a sharper distribution with training. Alignment distributions learned this way
often correlate with human intuition (e.g. word alignment in machine translation) [38]. [5]


**Hard Attention** Hard attention is an approximate inference approach for latent alignment (Figure 2a) [ 80, 4, 53, 26 ]. Hard attention takes a single hard sample of _z_ (as opposed to a soft mixture)
and then backpropagates through the model. The approach is derived by two choices: First apply Jensen’s inequality to get a lower bound on the log marginal likelihood, log E _z_ [ _p_ ( _y | x, z_ )] _≥_
E _z_ [log _p_ ( _y | x, z_ )], then maximize this lower-bound with policy gradients/REINFORCE [ 76 ] to obtain
unbiased gradient estimates,


_∇_ _θ_ E _z_ [log _f_ ( _x, z_ ))] = E _z_ [ _∇_ _θ_ log _f_ ( _x, z_ ) + (log _f_ ( _x, z_ ) _−_ _B_ ) _∇_ _θ_ log _p_ ( _z | x,_ ˜ _x_ )] _,_


where _B_ is a baseline that can be used to reduce the variance of this estimator. To implement this
approach efficiently, hard attention uses Monte Carlo sampling to estimate the expectation in the
gradient computation. For efficiency, a single sample from _p_ ( _z | x,_ ˜ _x_ ) is used, in conjunction with
other tricks to reduce the variance of the gradient estimator (discussed more below) [80, 50, 51].


**3** **Variational Attention for Latent Alignment Models**


Amortized variational inference (AVI, closely related to variational auto-encoders) [ 36, 61, 50 ] is a
class of methods to efficiently approximate latent variable inference, using learned inference networks.
In this section we explore this technique for deep latent alignment models, and propose methods for
_variational attention_ that combine the benefits of soft and hard attention.


First note that the key approximation step in hard attention is to optimize a lower bound derived from
Jensen’s inequality. This gap could be quite large, contributing to poor performance. [6] Variational


4 It is also possible to study the gap in finer detail by considering distributions over the inputs of _f_ that have
high probability under approximately linear regions of _f_, leading to the notion of _approximately expectation-_
_linear_ functions, which was originally proposed and studied in the context of dropout [46].
5 Another way of viewing soft attention is as simply a non-probabilistic learned function. While it is possible
that such models encode better inductive biases, our experiments show that when properly optimized, latent
alignment attention with explicit latent variables do outperform soft attention.
6 Prior works on hard attention have generally approached the problem as a black-box reinforcement learning
problem where the rewards are given by log _f_ ( _x, z_ ) . Ba et al. (2015) [ 4 ] and Lawson et al. (2017) [ 41 ] are
the notable exceptions, and both works utilize the framework from [ 51 ] which obtains multiple samples from a
learned sampling distribution to optimize the IWAE bound [12] or a reweighted wake-sleep objective.


4


**Algorithm 1** Variational Attention

_λ ←_ enc( _x,_ ˜ _x, y_ ; _φ_ ) _▷_ _Compute var. params_
_z ∼_ _q_ ( _z_ ; _λ_ ) _▷_ _Sample var. attention_
log _f_ ( _x, z_ ) _▷_ Compute output dist
_z_ _[′]_ _←_ E _p_ ( _z_ _′_ _| x,x_ ˜) [ _z_ _[′]_ ] _▷_ Compute soft atten.
_B_ = log _f_ ( _x, z_ _[′]_ ) _▷_ Compute baseline dist
Backprop _∇_ _θ_ and _∇_ _φ_ based on eq. 1 and KL



**Algorithm 2** Variational Relaxed Attention

max _θ_ E _z∼p_ [log _p_ ( _y | x, z_ )] _▷_ _Pretrain fixed θ_

_. . ._
_u ∼U_ _▷_ _Sample unparam._
_z ←_ _g_ _φ_ ( _u_ ) _▷_ _Reparam sample_
log _f_ ( _x, z_ ) _▷_ Compute output dist
Backprop _∇_ _θ_ and _∇_ _φ_, reparam and KL



inference methods directly aim to tighten this gap. In particular, the _evidence lower bound_ (ELBO)
is a parameterized bound over a family of distributions _q_ ( _z_ ) _∈Q_ (with the constraint that the
supp _q_ ( _z_ ) _⊆_ supp _p_ ( _z | x,_ ˜ _x, y_ )),


log E _z∼p_ ( _z | x,x_ ˜) [ _p_ ( _y | x, z_ )] _≥_ E _z∼q_ ( _z_ ) [log _p_ ( _y | x, z_ )] _−_ KL[ _q_ ( _z_ ) _∥_ _p_ ( _z | x,_ ˜ _x_ )]


This allows us to search over variational distributions _q_ to improve the bound. It is tight when the
variational distribution is equal to the posterior, i.e. _q_ ( _z_ ) = _p_ ( _z | x,_ ˜ _x, y_ ) . Hard attention is a special
case of the ELBO with _q_ ( _z_ ) = _p_ ( _z | x,_ ˜ _x_ ).


There are many ways to optimize the evidence lower bound; an effective choice for deep learning
applications is to use _amortized variational inference_ . AVI uses an _inference network_ to produce the
parameters of the variational distribution _q_ ( _z_ ; _λ_ ) . The inference network takes in the input, query,
and the output, i.e. _λ_ = _enc_ ( _x,_ ˜ _x, y_ ; _φ_ ) . The objective aims to reduce the gap with the inference
network _φ_ while also training the generative model _θ_,

max _φ,θ_ [E] _[z][∼][q]_ [(] _[z]_ [;] _[λ]_ [)] [[log] _[ p]_ [(] _[y][ |][ x, z]_ [)]] _[ −]_ [KL[] _[q]_ [(] _[z]_ [;] _[ λ]_ [)] _[ ∥]_ _[p]_ [(] _[z][ |][ x,]_ [ ˜] _[x]_ [)]]


With the right choice of optimization strategy and inference network this form of variational attention
can provide a general method for learning latent alignment models. In the rest of this section, we
consider strategies for accurately and efficiently computing this objective; in the next section, we
describe instantiations of _enc_ for specific domains.


**Algorithm 1: Categorical Alignments** First consider the case where _D_, the alignment distribution,
and _Q_, the variational family, are categorical distributions. Here the generative assumption is that
_y_ is generated from a single index of _x_ . Under this setup, a low-variance estimator of _∇_ _θ_ ELBO, is
easily obtained through a single sample from _q_ ( _z_ ) . For _∇_ _φ_ ELBO, the gradient with respect to the
KL portion is easily computable, but there is an optimization issue with the gradient with respect to
the first term E _z∼q_ ( _z_ ) [log _f_ ( _x, z_ ))].


Many recent methods target this issue, including neural estimates of baselines [ 50, 51 ], RaoBlackwellization [ 59 ], reparameterizable relaxations [ 31, 47 ], and a mix of various techniques

[ 73, 24 ]. We found that an approach using REINFORCE [ 76 ] along with a specialized baseline was
effective. However, note that REINFORCE is only one of the inference choices we can select, and
as we will show later, alternative approaches such as reparameterizable relaxations work as well.
Formally, we first apply the likelihood-ratio trick to obtain an expression for the gradient with respect
to the inference network parameters _φ_,


_∇_ _φ_ E _z∼q_ ( _z_ ) [log _p_ ( _y | x, z_ )] = E _z∼q_ ( _z_ ) [(log _f_ ( _x, z_ ) _−_ _B_ ) _∇_ _φ_ log _q_ ( _z_ )]


As with hard attention, we take a single Monte Carlo sample (now drawn from the variational
distribution). Variance reduction of this estimate falls to the baseline term _B_ . The ideal (and intuitive)
baseline would be E _z∼q_ ( _z_ ) [log _f_ ( _x, z_ )], analogous to the value function in reinforcement learning.
While this term cannot be easily computed, there is a natural, cheap approximation: soft attention (i.e.
log _f_ ( _x,_ E[ _z_ ])). Then the gradient is



_∇_ _φ_ log _q_ ( _z | x,_ ˜ _x_ ) (1)
� �



E _z∼q_ ( _z_ )



_f_ ( _x, z_ )
��log _f_ ( _x,_ E _z_ _′_ _∼p_ ( _z_ _′_ _| x,x_ ˜) [ _z_ _[′]_ ])



Effectively this weights gradients to _q_ based on the ratio of the inference network alignment approach
to a soft attention baseline. Notably the expectation in the soft attention is over _p_ (and not over _q_ ),
and therefore the baseline is constant with respect to _φ_ . Note that a similar baseline can also be used
for hard attention, and we apply it to both variational/hard attention models in our experiments.


5


**Algorithm 2: Relaxed Alignments** Next consider treating both _D_ and _Q_ as Dirichlets, where _z_
represents a mixture of indices. This model is in some sense closer to the soft attention formulation
which assigns mass to multiple indices, though fundamentally different in that we still formally treat
alignment as a latent variable. Again the aim is to find a low variance gradient estimator. Instead of
using REINFORCE, certain continuous distributions allow the use reparameterization [ 36 ], where
sampling _z ∼_ _q_ ( _z_ ) can be done by first sampling from a simple unparameterized distribution _U_, and
then applying a transformation _g_ _φ_ ( _·_ ), yielding an unbiased estimator,


E _u∼U_ [ _∇_ _φ_ log _p_ ( _y|x, g_ _φ_ ( _u_ ))] _−∇_ _φ_ KL [ _q_ ( _z_ ) _∥_ _p_ ( _z | x,_ ˜ _x_ )]


The Dirichlet distribution is not directly reparameterizable. While transforming the standard uniform
distribution with the inverse CDF of Dirichlet would result in a Dirichlet distribution, the inverse
CDF does not have an analytical solution. However, we can use rejection based sampling to get a
sample, and employ implicit differentiation to estimate the gradient of the CDF [32].


Empirically, we found the random initialization would result in convergence to uniform Dirichlet
parameters for _λ_ . (We suspect that it is easier to find low KL local optima towards the center of the
simplex). In experiments, we therefore initialize the latent alignment model by first minimizing the
Jensen bound, E _z∼p_ ( _z | x,x_ ˜) [log _p_ ( _y | x, z_ )], and then introducing the inference network.


**4** **Models and Methods**


We experiment with variational attention in two different domains where attention-based models are
essential and widely-used: neural machine translation and visual question answering.


**Neural Machine Translation** Neural machine translation (NMT) takes in a source sentence and
predicts each word of a target sentence _y_ _j_ in an auto-regressive manner. The model first contextually
embeds each source word using a bidirectional LSTM to produce the vectors _x_ 1 _. . . x_ _T_ . The query

˜
_x_ consists of an LSTM-based representation of the previous target words _y_ 1: _j−_ 1 . Attention is used
to identify which source positions should be used to predict the target. The parameters of _D_ are
generated from an MLP between the query and source [ 6 ], and _f_ concatenates the selected _x_ _i_ with
the query ˜ _x_ and passes it to an MLP to produce the distribution over the next target word _y_ _j_ .


For variational attention, the inference network applies a bidirectional LSTM over the source and
the target to obtain the hidden states _x_ 1 _, . . ., x_ _T_ and _h_ 1 _, . . ., h_ _S_, and produces the alignment scores
at the _j_ -th time step via a bilinear map, _s_ [(] _i_ _[j]_ [)] = exp( _h_ _[⊤]_ _j_ **[U]** _[x]_ _[i]_ [)] [. For the categorical case, the scores]

are normalized, _q_ ( _z_ _i_ [(] _[j]_ [)] = 1) _∝_ _s_ [(] _i_ _[j]_ [)] [; in the relaxed case the parameters of the Dirichlet are] _[ α]_ _i_ [(] _[j]_ [)] =
_s_ _i_ [(] _[j]_ [)] [. Note, the inference network sees the entire target (through bidirectional LSTMs). The word]
embeddings are shared between the generative/inference networks, but other parameters are separate.


**Visual Question Answering** Visual question answering (VQA) uses attention to locate the parts of
an image that are necessary to answer a textual question. We follow the recently-proposed “bottom-up
top-down” attention approach [ 2 ], which uses Faster R-CNN [ 60 ] to obtain object bounding boxes
and performs mean-pooling over the convolutional features (from a pretrained ResNet-101 [ 27 ]) in
each bounding box to obtain object representations _x_ 1 _, . . ., x_ _T_ . The query ˜ _x_ is obtained by running
an LSTM over the question, the attention function _a_ passes the query and the object representation
through an MLP. The prediction function _f_ is also similar to the NMT case: we concatenate the
chosen _x_ _i_ with the query ˜ _x_ to use as input to an MLP which produces a distribution over the output.
The inference network _enc_ uses the answer embedding _h_ _y_ and combines it with _x_ _i_ and ˜ _x_ to produce
the variational (categorical) distribution,


_q_ ( _z_ _i_ = 1) _∝_ exp( _u_ _[⊤]_ tanh( **U** 1 ( _x_ _i_ _⊙_ ReLU( **V** 1 _h_ _y_ )) + **U** 2 (˜ _x ⊙_ ReLU( **V** 2 _h_ _y_ ))))


where _⊙_ is the element-wise product. This parameterization worked better than alternatives. We did
not experiment with the relaxed case in VQA, as the object bounding boxes already give us the ability
to attend to larger portions of the image.


**Inference Alternatives** For categorical alignments we described maximizing a particular variational lower bound with REINFORCE. Note that other alternatives exist, and we briefly discuss them


6


here: 1) instead of the single-sample variational bound we can use a multiple-sample importance
sampling based approach such as Reweighted Wake-Sleep (RWS) [ 4 ] or VIMCO [ 52 ]; 2) instead of
REINFORCE we can approximate sampling from the discrete categorical distribution with GumbelSoftmax [30]; 3) instead of using an inference network we can directly apply Stochastic Variational
Inference (SVI) [28] to learn the local variational parameters in the posterior.


**Predictive Inference** At test time, we need to marginalize out the latent variables, i.e.
E _z_ [ _p_ ( _y | x,_ ˜ _x, z_ )] using _p_ ( _z | x,_ ˜ _x_ ) . In the categorical case, if speed is not an issue then enumerating alignments is preferable, which incurs a multiplicative cost of _O_ ( _T_ ) (but the enumeration is
parallelizable). Alternatively we experimented with a _K_ -max renormalization, where we only take
the top- _K_ attention scores to approximate the attention distribution (by re-normalizing). This makes
the multiplicative cost constant with respect to _T_ . For the relaxed case, sampling is necessary.


**5** **Experiments**


**Setup** For NMT we mainly use the IWSLT dataset [ 13 ]. This dataset is relatively small, but has
become a standard benchmark for experimental NMT models. We follow the same preprocessing as
in [ 21 ] with the same Byte Pair Encoding vocabulary of 14k tokens [ 65 ]. To show that variational
attention scales to large datasets, we also experiment on the WMT 2017 English-German dataset [ 8 ],
following the preprocessing in [ 74 ] except that we use newstest2017 as our test set. For VQA, we use
the VQA 2.0 dataset. As we are interested in intrinsic evaluation (i.e. log-likelihood) in addition to
the standard VQA metric, we randomly select half of the standard validation set as the test set (since
we need access to the actual labels). [7] (Therefore the numbers provided are not strictly comparable to
existing work.) While the preprocessing is the same as [ 2 ], our numbers are worse than previously
reported as we do not apply any of the commonly-utilized techniques to improve performance on
VQA such as data augmentation and label smoothing.


Experiments vary three components of the systems: (a) training objective and model, (b) training
approximations, comparing enumeration or sampling, [8] (c) test inference. All neural models have the
same architecture and the exact same number of parameters _θ_ (the inference network parameters _φ_
vary, but are not used at test). When training hard and variational attention with sampling both use
the same baseline, i.e the output from soft attention. The full architectures/hyperparameters for both
NMT and VQA are given in Appendix B.


**Results and Discussion** Table 1 shows the main results. We first note that hard attention underperforms soft attention, even when its expectation is enumerated. This indicates that Jensen’s inequality
alone is a poor bound. On the other hand, on both experiments, exact marginal likelihood outperforms
soft attention, indicating that when possible it is better to have latent alignments.


For NMT, on the IWSLT 2014 German-English task, variational attention with enumeration and
sampling performs comparably to optimizing the log marginal likelihood, despite the fact that it is
optimizing a lower bound. We believe that this is due to the use of _q_ ( _z_ ), which conditions on the
entire source/target and therefore potentially provides better training signal to _p_ ( _z | x,_ ˜ _x_ ) through the
KL term. Note that it is also possible to have _q_ ( _z_ ) come from a pretrained external model, such as
a traditional alignment model [ 20 ]. Table 3 (left) shows these results in context compared to the
best reported values for this task. Even with sampling, our system improves on the state-of-the-art.
On the larger WMT 2017 English-German task, the superior performance of variational attention
persists: our baseline soft attention reaches 24.10 BLEU score, while variational attention reaches
24.98. Note that this only reflects a reasonable setting without exhaustive tuning, yet we show that
we can train variational attention at scale. For VQA the trend is largely similar, and results for NLL
with variational attention improve on soft attention and hard attention. However the task-specific
evaluation metrics are slightly worse.


Table 2 (left) considers test inference for variational attention, comparing enumeration to _K_ -max with
_K_ = 5 . For all methods exact enumeration is better, however _K_ -max is a reasonable approximation.


7 VQA eval metric is defined as min _{_ # humans that said answer3 _,_ 1 _}_ . Also note that since there are sometimes

multiple answers for a given question, in such cases we sample (where the sampling probability is proportional
to the number of humans that said the answer) to get a single label.
8 Note that enumeration does not imply exact if we are enumerating an expectation on a lower bound.


7


NMT VQA
Model Objective E PPL BLEU NLL Eval


Soft Attention log _p_ ( _y |_ E[ _z_ ])    - 7.17 32.77 1.76 58.93
Marginal Likelihood log E[ _p_ ] Enum 6.34 33.29 1.69 60.33
Hard Attention E _p_ [log _p_ ] Enum 7.37 31.40 1.78 57.60
Hard Attention E _p_ [log _p_ ] Sample 7.38 31.00 1.82 56.30
Variational Relaxed Attention E _q_ [log _p_ ] _−_ KL Sample 7.58 30.05    -    Variational Attention E _q_ [log _p_ ] _−_ KL Enum 6.08 33.68 1.69 58.44
Variational Attention E _q_ [log _p_ ] _−_ KL Sample 6.17 33.30 1.75 57.52


Table 1: Evaluation on NMT and VQA for the various models. E column indicates whether the expectation
is calculated via enumeration (Enum) or a single sample (Sample) during training. For NMT we evaluate
intrinsically on perplexity (PPL) (lower is better) and extrinsically on BLEU (higher is better), where for BLEU
we perform beam search with beam size 10 and length penalty (see Appendix B for further details). For VQA
we evaluate intrinsically on negative log-likelihood (NLL) (lower is better) and extrinsically on VQA evaluation
metric (higher is better). All results except for relaxed attention use enumeration at test time.


PPL BLEU

Model Exact _K_ -Max Exact _K_ -Max


Marginal Likelihood 6.34 6.90 33.29 33.31
Hard + Enum 7.37 7.37 31.40 31.37
Hard + Sample 7.38 7.38 31.00 31.04
Variational + Enum 6.08 6.42 33.68 33.69
Variational + Sample 6.17 6.51 33.30 33.27


Table 2: (Left) Performance change on NMT from exact decoding to _K_ -Max decoding with _K_ = 5 . (see section
5 for definition of K-max decoding). (Right) Test perplexity of different approaches while varying _K_ to estimate
E _z_ [ _p_ ( _y|x,_ ˜ _x_ )]. Dotted lines compare soft baseline and variational with full enumeration.


Table 2 (right) shows the PPL of different models as we increase _K_ . Good performance requires
_K >_ 1, but we only get marginal benefits for _K >_ 5 . Finally, we observe that it is possible to _train_
with soft attention and _test_ using _K_ -Max with a small performance drop ( `Soft KMax` in Table 2
(right)). This possibly indicates that soft attention models are approximating latent alignment models.
On the other hand, training with latent alignments and testing with soft attention performed badly.


Table 3 (lower right) looks at the entropy of the prior distribution learned by the different models.
Note that hard attention has very low entropy (high certainty) whereas soft attention is quite high.
The variational attention model falls in between. Figure 3 (left) illustrates the difference in practice.


Table 3 (upper right) compares inference alternatives for variational attention. RWS reaches a
comparable performance as REINFORCE, but at a higher memory cost as it requires multiple
samples. Gumbel-Softmax reaches nearly the same performance and seems like a viable alternative;
although we found its performance is sensitive to its temperature parameter. We also trained a
non-amortized SVI model, but found that at similar runtime it was not able to produce satisfactory
results, likely due to insufficient updates of the local variational parameters. A hybrid method such as
semi-amortized inference [39, 34] might be a potential future direction worth exploring.


Despite extensive experiments, we found that variational relaxed attention performed worse than other
methods. In particular we found that when training with a Dirichlet KL, it is hard to reach low-entropy
regions of the simplex, and the attentions are more uniform than either soft or variational categorical
attention. Table 3 (lower right) quantifies this issue. We experimented with other distributions such
as Logistic-Normal and Gumbel-Softmax [ 31, 47 ] but neither fixed this issue. Others have also noted
difficulty in training Dirichlet models with amortized inference [69].


Besides performance, an advantage of these models is the ability to perform posterior inference, since
the _q_ function can be used directly to obtain posterior alignments. Contrast this with hard attention
where _q_ = _p_ ( _z | x,_ ˜ _x_ ), i.e. the variational posterior is independent of the future information. Figure 3
shows the alignments of _p_ and _q_ for variational attention over a fixed sentence (see Appendix C for
more examples). We see that _q_ is able to use future information to correct alignments. We note that
the inability of soft and hard attention to produce good alignments has been noted as a major issue
in NMT [ 38 ]. While _q_ is not used directly in left-to-right NMT decoding, it could be employed for
other applications such as in an iterative refinement approach [56, 42].


8


Figure 3: (Left) An example demonstrating the difference between the prior alignment (red) and the variational
posterior (blue) when translating from DE-EN (left-to-right). Note the improved blue alignments for `actually`
and `violent` which benefit from seeing the next word. (Right) Comparison of soft attention (green) with the _p_
of variational attention (red). Both models imply a similar alignment, but variational attention has lower entropy.


Inference Method #Samples PPL BLEU



IWSLT

Model BLEU


Beam Search Optimization [77] 26.36
Actor-Critic [5] 28.53
Neural PBMT + LM [29] 30.08
Minimum Risk Training [21] 32.84


Soft Attention 32.77
Marginal Likelihood 33.29
Hard Attention + Enum 31.40
Hard Attention + Sample 30.42
Variational Relaxed Attention 30.05

Variational Attention + Enum 33.69
Variational Attention + Sample 33.30



REINFORCE 1 6.17 33.30

RWS 5 6.41 32.96

Gumbel-Softmax 1 6.51 33.08


Entropy
Model NMT VQA


Soft Attention 1.24 2.70
Marginal Likelihood 0.82 2.66
Hard Attention + Enum 0.05 0.73
Hard Attention + Sample 0.07 0.58
Variational Relaxed Attention 2.02 
Variational Attention + Enum 0.54 2.07
Variational Attention + Sample 0.52 2.44



Table 3: (Left) Comparison against the best prior work for NMT on the IWSLT 2014 German-English test set.
(Upper Right) Comparison of inference alternatives of variational attention on IWSLT 2014. (Lower Right)
Comparison of different models in terms of implied discrete entropy (lower = more certain alignment).


**Potential Limitations** While this technique is a promising alternative to soft attention, there are
some practical limitations: (a) Variational/hard attention needs a good baseline estimator in the form
of soft attention. We found this to be a necessary component for adequately training the system. This
may prevent this technique from working when _T_ is intractably large and soft attention is not an
option. (b) For some applications, the model relies heavily on having a good posterior estimator. In
VQA we had to utilize domain structure for the inference network construction. (c) Recent models
such as the Transformer [ 74 ], utilize many repeated attention models. For instance the current best
translation models have the equivalent of 150 different attention queries per word translated. It is
unclear if this approach can be used at that scale as predictive inference becomes combinatorial.


**6** **Conclusion**


Attention methods are ubiquitous tool for areas like natural language processing; however they
are difficult to use as latent variable models. This work explores alternative approaches to latent
alignment, through variational attention with promising result. Future work will experiment with
scaling the method on larger-scale tasks and in more complex models, such as multi-hop attention
models, transformer models, and structured models, as well as utilizing these latent variables for
interpretability and as a way to incorporate prior knowledge.


9


**Acknowledgements**


We are grateful to Sam Wiseman and Rachit Singh for insightful comments and discussion, as well as
Christian Puhrsch for help with translations. This project was supported by a Facebook Research
Award (Low Resource NMT). YK is supported by a Google AI PhD Fellowship. YD is supported by
a Bloomberg Research Award. AMR gratefully acknowledges the support of NSF CCF-1704834 and
an Amazon AWS Research award.


**References**


[1] David Alvarez-Melis and Tommi S Jaakkola. A Causal Framework for Explaining the Predictions of
Black-Box Sequence-to-Sequence Models. In _Proceddings of EMNLP_, 2017.


[2] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei
Zhang. Bottom-up and Top-Down Attention for Image Captioning and Visual Question Answering. In
_Proceedings of CVPR_, 2018.


[3] Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. Multiple Object Recognition with Visual Attention.
In _Proceedings of ICLR_, 2015.


[4] Jimmy Ba, Ruslan R Salakhutdinov, Roger B Grosse, and Brendan J Frey. Learning Wake-Sleep Recurrent
Attention Models. In _Proceedings of NIPS_, 2015.


[5] Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron
Courville, and Yoshua Bengio. An Actor-Critic Algorithm for Sequence Prediction. In _Proceedings of_
_ICLR_, 2017.


[6] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural Machine Translation by Jointly Learning
to Align and Translate. In _Proceedings of ICLR_, 2015.


[7] Hareesh Bahuleyan, Lili Mou, Olga Vechtomova, and Pascal Poupart. Variational Attention for Sequenceto-Sequence Models. _arXiv:1712.08207_, 2017.


[8] Ondˇrej Bojar, Christian Buck, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow,
Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, and Julia Kreutzer. Proceedings of the second
conference on machine translation. In _Proceedings of the Second Conference on Machine Translation_ .
Association for Computational Linguistics, 2017.


[9] Jorg Bornschein, Andriy Mnih, Daniel Zoran, and Danilo J. Rezende. Variational Memory Addressing in
Generative Models. In _Proceedings of NIPS_, 2017.


[10] Peter F Brown, Vincent J Della Pietra, Stephen A Della Pietra, and Robert L Mercer. The Mathematics of
Statistical Machine Translation: Parameter Estimation. _Computational linguistics_, 19(2):263–311, 1993.


[11] Peter F. Brown, Vincent J. Della Pietra, Stephen A. Della Pietra, and Robert L. Mercer. The mathematics
of statistical machine translation: Parameter estimation. _Comput. Linguist._, 19(2):263–311, June 1993.


[12] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted Autoencoders. In _Proceedings_
_of ICLR_, 2015.


[13] Mauro Cettolo, Jan Niehues, Sebastian Stuker, Luisa Bentivogli, and Marcello Federico. Report on the
11th IWSLT evaluation campaign. In _Proceedings of IWSLT_, 2014.


[14] William Chan, Navdeep Jaitly, Quoc Le, and Oriol Vinyals. Listen, Attend and Spell. _arXiv:1508.01211_,
2015.


[15] Kyunghyun Cho, Aaron Courville, and Yoshua Bengio. Describing Multimedia Content using Attentionbased Encoder-Decoder Networks. In _IEEE Transactions on Multimedia_, 2015.


[16] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. AttentionBased Models for Speech Recognition. In _Proceedings of NIPS_, 2015.


[17] Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, and Yoshua Bengio. A
Recurrent Latent Variable Model for Sequential Data. In _Proceedings of NIPS_, 2015.


[18] Trevor Cohn, Cong Duy Vu Hoang, Ekaterina Vymolova, Kaisheng Yao, Chris Dyer, and Gholamreza
Haffari. Incorporating Structural Alignment Biases into an Attentional Neural Translation Model. In
_Proceedings of NAACL_, 2016.


10


[19] Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush. Image-to-Markup Generation with
Coarse-to-Fine Attention. In _Proceedings of ICML_, 2017.


[20] Chris Dyer, Victor Chahuneau, and Noah A. Smith. A Simple, Fast, and Effective Reparameterization of
IBM Model 2. In _Proceedings of NAACL_, 2013.


[21] Sergey Edunov, Myle Ott, Michael Auli, David Grangier, and Marc’Aurelio Ranzato. Classical Structured
Prediction Losses for Sequence to Sequence Learning. In _Proceedings of NAACL_, 2018.


[22] Marco Fraccaro, Soren Kaae Sonderby, Ulrich Paquet, and Ole Winther. Sequential Neural Models with
Stochastic Layers. In _Proceedings of NIPS_, 2016.


[23] Anirudh Goyal, Alessandro Sordoni, Marc-Alexandre Cote, Nan Rosemary Ke, and Yoshua Bengio.
Z-Forcing: Training Stochastic Recurrent Networks. In _Proceedings of NIPS_, 2017.


[24] Will Grathwohl, Dami Choi, Yuhuai Wu, Geoffrey Roeder, and David Duvenaud. Backpropagation through
the Void: Optimizing control variates for black-box gradient estimation. In _Proceedings of ICLR_, 2018.


[25] Jiatao Gu, Zhengdong Lu, Hang Li, and Victor OK Li. Incorporating Copying Mechanism in Sequence-toSequence Learning. 2016.


[26] Caglar Gulcehre, Sarath Chandar, Kyunghyun Cho, and Yoshua Bengio. Dynamic Neural Turing Machine
with Soft and Hard Addressing Schemes. _arXiv:1607.00036_, 2016.


[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition.
In _Proceedings of CVPR_, 2016.


[28] Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. _The_
_Journal of Machine Learning Research_, 14(1):1303–1347, 2013.


[29] Po-Sen Huang, Chong Wang, Sitao Huang, Dengyong Zhou, and Li Deng. Towards neural phrase-based
machine translation. In _Proceedings of ICLR_, 2018.


[30] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. _arXiv_
_preprint arXiv:1611.01144_, 2016.


[31] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. In
_Proceedings of ICLR_, 2017.


[32] Martin Jankowiak and Fritz Obermeyer. Pathwise Derivatives Beyond the Reparameterization Trick. In
_Proceedings of ICML_, 2018.


[33] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured Attention Networks. In
_Proceedings of ICLR_, 2017.


[34] Yoon Kim, Sam Wiseman, Andrew C Miller, David Sontag, and Alexander M Rush. Semi-amortized
variational autoencoders. _arXiv preprint arXiv:1802.02550_, 2018.


[35] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In _Proceedings of_
_ICLR_, 2015.


[36] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In _Proceedings of ICLR_, 2014.


[37] Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi,
Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, et al. Moses: Open source toolkit for statistical
machine translation. In _Proceedings of the 45th annual meeting of the ACL on interactive poster and_
_demonstration sessions_, pages 177–180. Association for Computational Linguistics, 2007.


[38] Philipp Koehn and Rebecca Knowles. Six Challenges for Neural Machine Translation. _arXiv:1706.03872_,
2017.


[39] Rahul G. Krishnan, Dawen Liang, and Matthew Hoffman. On the Challenges of Learning with Inference
Networks on Sparse, High-dimensional Data. In _Proceedings of AISTATS_, 2018.


[40] Rahul G. Krishnan, Uri Shalit, and David Sontag. Structured Inference Networks for Nonlinear State
Space Models. In _Proceedings of AAAI_, 2017.


[41] Dieterich Lawson, Chung-Cheng Chiu, George Tucker, Colin Raffel, Kevin Swersky, and Navdeep Jaitly.
Learning Hard Alignments in Variational Inference. In _Proceedings of ICASSP_, 2018.


11


[42] Jason Lee, Elman Mansimov, and Kyunghyun Cho. Deterministic Non-Autoregressive Neural Sequence
Modeling by Iterative Refinement. _arXiv:1802.06901_, 2018.


[43] Tao Lei, Regina Barzilay, and Tommi Jaakkola. Rationalizing Neural Rredictions. In _Proceedings of_
_EMNLP_, 2016.


[44] Yang Liu and Mirella Lapata. Learning Structured Text Representations. In _Proceedings of TACL_, 2017.


[45] Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective Approaches to Attention-based
Neural Machine Translation. In _Proceedings of EMNLP_, 2015.


[46] Xuezhe Ma, Yingkai Gao, Zhiting Hu, Yaoliang Yu, Yuntian Deng, and Eduard Hovy. Dropout with
Expectation-linear Regularization. In _Proceedings of ICLR_, 2017.


[47] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A Continuous Relaxation
of Discrete Random Variables. In _Proceedings of ICLR_, 2017.


[48] André F. T. Martins and Ramón Fernandez Astudillo. From Softmax to Sparsemax: A Sparse Model of
Attention and Multi-Label Classification. In _Proceedings of ICML_, 2016.


[49] Arthur Mensch and Mathieu Blondel. Differentiable Dynamic Programming for Structured Prediction and
Attention. In _Proceedings of ICML_, 2018.


[50] Andriy Mnih and Karol Gregor. Neural Variational Inference and Learning in Belief Networks. In
_Proceedings of ICML_, 2014.


[51] Andriy Mnih and Danilo J. Rezende. Variational Inference for Monte Carlo Objectives. In _Proceedings of_
_ICML_, 2016.


[52] Andriy Mnih and Danilo J Rezende. Variational inference for monte carlo objectives. _arXiv preprint_
_arXiv:1602.06725_, 2016.


[53] Volodymyr Mnih, Nicola Heess, Alex Graves, and Koray Kavukcuoglu. Recurrent Models of Visual
Attention. In _Proceedings of NIPS_, 2015.


[54] Vlad Niculae and Mathieu Blondel. A Regularized Framework for Sparse and Structured Neural Attention.
In _Proceedings of NIPS_, 2017.


[55] Vlad Niculae, André F. T. Martins, Mathieu Blondel, and Claire Cardie. SparseMAP: Differentiable Sparse
Structured Inference. In _Proceedings of ICML_, 2018.


[56] Roman Novak, Michael Auli, and David Grangier. Iterative Refinement for Machine Translation.
_arXiv:1610.06602_, 2016.


[57] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global Vectors for Word
Representation. In _Proceedings of EMNLP_, 2014.


[58] Colin Raffel, Minh-Thang Luong, Peter J Liu, Ron J Weiss, and Douglas Eck. Online and Linear-Time
Attention by Enforcing Monotonic Alignments. In _Proceedings of ICML_, 2017.


[59] Rajesh Ranganath, Sean Gerrish, and David M. Blei. Black Box Variational Inference. In _Proceedings of_
_AISTATS_, 2014.


[60] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks. In _Proceedings of NIPS_, 2015.


[61] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In _Proceedings of ICML_, 2014.


[62] Tim Rocktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomas Kocisky, and Phil Blunsom. Reasoning about Entailment with Neural Attention. In _Proceedings of ICLR_, 2016.


[63] Alexander M. Rush, Sumit Chopra, and Jason Weston. A Neural Attention Model for Abstractive Sentence
Summarization. In _Proceedings of EMNLP_, 2015.


[64] Philip Schulz, Wilker Aziz, and Trevor Cohn. A Stochastic Decoder for Neural Machine Translation. In
_Proceedings of ACL_, 2018.


[65] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural Machine Translation of Rare Words with
Subword Units. In _Proceedings of ACL_, 2016.


12


[66] Iulian Vlad Serban, Alessandro Sordoni, Laurent Charlin Ryan Lowe, Joelle Pineau, Aaron Courville, and
Yoshua Bengio. A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. In
_Proceedings of AAAI_, 2017.


[67] Shiv Shankar, Siddhant Garg, and Sunita Sarawagi. Surprisingly Easy Hard-Attention for Sequence to
Sequence Learning. In _Proceedings of EMNLP_, 2018.


[68] Bonggun Shin, Falgun H Chokshi, Timothy Lee, and Jinho D Choi. Classification of Radiology Reports
Using Neural Attention Models. In _Proceedings of IJCNN_, 2017.


[69] Akash Srivastava and Charles Sutton. Autoencoding Variational Inference for Topic Models. In _Proceed-_
_ings of ICLR_, 2017.


[70] Jinsong Su, Shan Wu, Deyi Xiong, Yaojie Lu, Xianpei Han, and Biao Zhang. Variational Recurrent Neural
Machine Translation. In _Proceedings of AAAI_, 2018.


[71] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-To-End Memory Networks. In
_Proceedings of NIPS_, 2015.


[72] Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. Modeling Coverage for Neural
Machine Translation. In _Proceedings of ACL_, 2016.


[73] George Tucker, Andriy Mnih, Chris J. Maddison, Dieterich Lawson, and Jascha Sohl-Dickstein. REBAR:
Low-variance, Unbiased Gradient Estimates for Discrete Latent Variable Models. In _Proceedings of NIPS_,
2017.


[74] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is All You Need. In _Proceedings of NIPS_, 2017.


[75] Stephan Vogel, Hermann Ney, and Christoph Tillmann. HMM-based Word Alignment in Statistical
Translation. In _Proceedings of COLING_, 1996.


[76] Ronald J. Williams. Simple Statistical Gradient-following Algorithms for Connectionist Reinforcement
Learning. _Machine Learning_, 8, 1992.


[77] Sam Wiseman and Alexander M. Rush. Sequence-to-Sequence learning as Beam Search Optimization. In
_Proceedings of EMNLP_, 2016.


[78] Shijie Wu, Pamela Shapiro, and Ryan Cotterell. Hard Non-Monotonic Attention for Character-Level
Transduction. In _Proceedings of EMNLP_, 2018.


[79] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun, Yuan Cao, Klaus Macherey Qin Gao, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu,
Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, Nishant Patil
George Kurian, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg
Corrado, Macduff Hughes, and Jeffrey Dean. Google’s Neural Machine Translation System: Bridging the
Gap between Human and Machine Translation. _arXiv:1609.08144_, 2016.


[80] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel,
and Yoshua Bengio. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. In
_Proceedings of ICML_, 2015.


[81] Zichao Yang, Kiaodong He, Jianfeng Gao, Li Deng, and Alex Smola. Stacked Attention Networks for
Image Question Answering. In _Proceedings of CVPR_, 2016.


[82] Lei Yu, Phil Blunsom, Chris Dyer, Edward Grefenstette, and Tomas Kocisky. The Neural Noisy Channel.
In _Proceedings of ICLR_, 2017.


[83] Lei Yu, Jan Buys, and Phil Blunsom. Online Segment to Segment Neural Transduction. In _Proceedings of_
_EMNLP_, 2016.


[84] Biao Zhang, Deyi Xiong, Jinsong Su, Hong Duan, and Min Zhang. Variational Neural Machine Translation.
In _Proceedings of EMNLP_, 2016.


[85] Chen Zhu, Yanpeng Zhao, Shuaiyi Huang, Kewei Tu, and Yi Ma. Structured Attentions for Visual Question
Answering. In _Proceedings of ICCV_, 2017.


13


## **Supplementary Materials for** **Latent Alignment and Variational Attention**

**Appendix A: Proof of Proposition 1**


**Proposition.** _Define_ _g_ _x,y_ ˆ : ∆ _[T][ −]_ [1] _�→_ [0 _,_ 1] _to be the function given by_ _g_ _x,y_ ˆ ( _z_ ) = _f_ ( _x, z_ ) _y_ ˆ _(i.e._
_g_ _x,y_ ˆ ( _z_ ) = _p_ ( _y_ = ˆ _y | x,_ ˜ _x, z_ )) _for a twice differentiable function_ _f_ _. Let_ _H_ _g_ _x,y_ ˆ ( _z_ ) _be the Hessian of_
_g_ _x,y_ ˆ ( _z_ ) _evaluated at_ _z_ _, and further suppose_ _∥H_ _g_ _x,y_ ˆ ( _z_ ) _∥_ 2 _≤_ _c_ _for all_ _z ∈_ ∆ _[T][ −]_ [1] _,_ ˆ _y ∈Y_ _, and_ _x_ _, where_
_∥· ∥_ 2 _is the spectral norm. Then for all_ ˆ _y ∈Y,_


_| p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) _−_ _p_ soft ( _y_ = ˆ _y | x,_ ˜ _x_ ) _| ≤_ _c_


_Proof._ We begin by performing Taylor’s expansion of _g_ _x,y_ ˆ at E[ _z_ ]:

E[ _g_ _x,y_ ˆ ( _z_ )] = E _g_ _x,y_ ˆ (E[ _z_ ]) + ( _z −_ E[ _z_ ]) _[⊤]_ _∇g_ _x,y_ ˆ (E[ _z_ ]) + [1]
� 2 [(] _[z][ −]_ [E][[] _[z]_ [])] _[⊤]_ _[H]_ _[g]_ _[x,][y]_ [ˆ] [(ˆ] _[z]_ [)(] _[z][ −]_ [E][[] _[z]_ [])] �

= _g_ _x,y_ ˆ (E[ _z_ ]) + 2 [1] [E][[(] _[z][ −]_ [E][[] _[z]_ [])] _[⊤]_ _[H]_ _[g]_ _[x,][y]_ [ˆ] [(ˆ] _[z]_ [)(] _[z][ −]_ [E][[] _[z]_ [])]]


for some ˆ _z_ = _λz_ + (1 _−_ _λ_ )E[ _z_ ] _, λ ∈_ [0 _,_ 1]. Then letting _u_ = _z −_ E[ _z_ ], we have


_u_ _[⊤]_ _u_
_|_ ( _z −_ E[ _z_ ]) _[⊤]_ _H_ _g_ _x,y_ ˆ (ˆ _z_ )( _z −_ E[ _z_ ]) _|_ = _| ∥u∥_ 2 [2] _∥u∥_ 2 _H_ _g_ _x,y_ ˆ (ˆ _z_ ) _∥u∥_ 2 _|_

_≤∥u∥_ 2 [2] _[c]_


where _c_ = max _{|λ_ max _|, |λ_ min _|}_ is the largest absolute eigenvalue of _H_ _g_ _x,y_ ˆ (ˆ _z_ ) . (Here _λ_ max and _λ_ min
are maximum/minimum eigenvalues of _H_ _g_ _X,q_ (ˆ _z_ ) ). Note that _c_ is also equal to the spectral norm
_∥H_ _g_ _X,q_ (ˆ _z_ ) _∥_ 2 since the Hessian is symmetric.


Then,


_|_ E[( _z −_ E[ _z_ ]) _[⊤]_ _H_ _g_ _x,y_ ˆ (ˆ _z_ )( _z −_ E[ _z_ ])] _| ≤_ E[ _|_ ( _z −_ E[ _z_ ]) _[⊤]_ _H_ _g_ _x,y_ ˆ (ˆ _z_ )( _z −_ E[ _z_ ]) _|_ ]

_≤_ E[ _∥u∥_ 2 [2] _[c]_ []]

_≤_ 2 _c_


Here the first inequality follows due to the convexity of the absolute value function and the last
inequality follows since


_∥u∥_ 2 [2] [= (] _[z][ −]_ [E][[] _[z]_ [])] _[⊤]_ [(] _[z][ −]_ [E][[] _[z]_ [])]

= _z_ _[⊤]_ _z_ + E[ _z_ ] _[⊤]_ E[ _z_ ] _−_ 2E[ _z_ ] _[⊤]_ _z_

_≤_ _z_ _[⊤]_ _z_ + E[ _z_ ] _[⊤]_ E[ _z_ ]

_≤_ 2


where the last two inequalities are due to the fact that _z,_ E[ _z_ ] _∈_ ∆ _[T][ −]_ [1] . Then putting it all together
we have,


_| p_ ( _y_ = ˆ _y | x,_ ˜ _x_ ) _−_ _p_ soft ( _y_ = ˆ _y | x,_ ˜ _x_ ) _|_ = _|_ E[ _g_ _x,y_ ˆ ( _z_ )] _−_ _g_ _x,y_ ˆ (E[ _z_ ]) _|_

= [1]

2 _[|]_ [ E][[(] _[z][ −]_ [E][[] _[z]_ [])] _[⊤]_ _[H]_ _[g]_ _[x,][y]_ [ˆ] [(ˆ] _[z]_ [)(] _[z][ −]_ [E][[] _[z]_ [])]] _[ |]_


_≤_ _c_


14


**Appendix B: Experimental Setup**


**Neural Machine Translation**


For data processing we closely follow the setup in [ 21 ], which uses Byte Pair Encoding over the
combined source/target training set to obtain a vocabulary size of 14,000 tokens. However, different
from [ 21 ] which uses maximum sequence length of 175, for faster training we only train on sequences
of length up to 125.


The encoder is a two-layer bi-directional LSTM with 512 units in each direction, and the decoder as
a two-layer LSTM with with 768 units. For the decoder, the convex combination of source hidden
states at each time step from the attention distribution is used as additional input at the next time step.
Word embedding is 512-dimensional.


The inference network consists of two bi-directional LSTMs (also two-layer and 512-dimensional
each) which is run over the source/target to obtain the hidden states at each time step. These hidden
states are combined using bilinear attention [ 45 ] to produce the variational parameters. (In contrast
the generative model uses MLP attention from [ 6 ], though we saw little difference between the two
parameterizations). Only the word embedding is shared between the inference network and the
generative model.


Other training details include: batch size of 6, dropout rate of 0.3, parameter initialization over a
uniform distribution _U_ [ _−_ 0 _._ 1 _,_ 0 _._ 1], gradient norm clipping at 5, and training for 30 epochs with Adam
(learning rate = 0.0003, _β_ 1 = 0.9, _β_ 2 = 0.999) [ 35 ] with a learning rate decay schedule which starts
halving the learning rate if validation perplexity does not improve. Most models converged well
before 30 epochs.


For decoding we use beam search with beam size 10 and length penalty _α_ = 1, from [ 79 ]. The length
penalty added about 0.5 BLEU points across all the models.


**Visual Question Answering**


The model first obtains object features by mean-pooling the pretrained ResNet-101 features [ 27 ]
(which are 2048-dimensional) over object regions given by Faster R-CNN [ 60 ].The ResNet features
are kept fixed and not fine-tuned during training. We fix the maximum number of possible regions to
be 36. For the question embedding we use a one-layer LSTM with 1024 units over word embeddings.
The word embeddings are 300-dimensional and initialized with GloVe [ 57 ]. The generative model
produces a distribution over the possible objects via applying MLP attention, i.e.


_p_ ( _z_ _i_ = 1 _| x,_ ˜ _x_ ) _∝_ exp( _w_ _[⊤]_ tanh( **W** 1 _x_ _i_ + **W** 2 _x_ ˜))


The selected image region is concatenated with the question embedding and fed to a one-layer MLP
with ReLU non-linearity and 1024 hidden units.


The inference network produces a categorical distribution over the image regions by interacting
the answer embedding _h_ _y_ (which are 256-dimensional and initialized randomly) with the question
embedding ˜ _x_ and the image regions _x_ _i_,


_q_ ( _z_ _i_ = 1) _∝_ exp( _u_ _[⊤]_ tanh( **U** 1 ( _x_ _i_ _⊙_ ReLU( **V** 1 _h_ _y_ )) + **U** 2 (˜ _x ⊙_ ReLU( **V** 2 _h_ _y_ ))))


where _⊙_ denotes element-wise multiplication. The generative/inference attention MLPs have 1024
hidden units each (i.e. _w, u ∈_ R [1024] ).


Other training details include: batch size of 512, dropout rate of 0.5 on the penultimate layer (i.e.
before affine transformation into answer vocabulary), and training for 50 epochs with with Adam
(learning rate = 0.0005, _β_ 1 = 0.9, _β_ 2 = 0.999) [35].


In cases where there is more than one answer for a given question/image pair, we randomly sample
the answer, where the sampling probability is proportional to the number of humans who gave the

answer.


15


**Appendix C: Additional Visualizations**


(a) (b)


(c) (d)


(e) (f)


Figure 4: (Left Column) Further examples highlighting the difference between the prior alignment (red) and
the variational posterior (blue) when translating from DE-EN (left-to-right). The variational posterior is able to
better handle reordering; in (a) the variational posterior successfully aligns ‘turning’ to ‘verwandelt’, in (c) we
see a similar pattern with the alignment of the clause ‘that’s my brand’ to ‘das ist meine marke’. In (e) the prior
and posterior both are confused by the ‘-ial’ in ‘territor-ial’, however the posterior still remains more accurate
overall and correctly aligns the rest of ‘revierverhalten’ to ‘territorial behaviour’. (Right Column) Additional
comparisons between soft attention (green) and the prior alignments of variational attention (red). Alignments
from both models are similar, but variational attention is lower entropy. Both soft and variational attention rely
on aligning the inserted English word ‘orientation’ to the comma in (b) since a direct translation does not appear
in the German source.


16


