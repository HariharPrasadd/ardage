## **Dropout Inference in Bayesian Neural Networks with Alpha-divergences**

**Yingzhen Li** [1] **Yarin Gal** [1 2]



**Abstract**


To obtain uncertainty estimates with real-world
Bayesian deep learning models, practical inference approximations are needed. Dropout variational inference (VI) for example has been used
for machine vision and medical applications,
but VI can severely underestimates model uncertainty. Alpha-divergences are alternative divergences to VI’s KL objective, which are able
to avoid VI’s uncertainty underestimation. But
these are hard to use in practice: existing techniques can only use Gaussian approximating distributions, and require existing models to be
changed radically, thus are of limited use for
practitioners. We propose a re-parametrisation
of the alpha-divergence objectives, deriving a
simple inference technique which, together with
dropout, can be easily implemented with existing models by simply changing the loss of the
model. We demonstrate improved uncertainty estimates and accuracy compared to VI in dropout
networks. We study our model’s epistemic uncertainty far away from the data using adversarial
images, showing that these can be distinguished
from non-adversarial images by examining our
model’s uncertainty.


**1. Introduction**


Deep learning models have been used to obtain state-ofthe-art results on many tasks (Krizhevsky et al., 2012;
Szegedy et al., 2014; Sutskever et al., 2014; Sundermeyer
et al., 2012; Mikolov et al., 2010; Kalchbrenner & Blunsom, 2013), and in many pipelines these models have replaced the more traditional _Bayesian probabilistic_ models
(Sennrich et al., 2016). But unlike deep learning models,
Bayesian probabilistic models can capture parameter uncertainty and its induced effects over predictions, capturing
the models’ ignorance about the world, and able to convey
their increased uncertainty on out-of-data examples. This
information can be used, for example, to identify when a vi

1 University of Cambridge, UK 2 The Alan Turing Institute,
UK. Correspondence to: Yingzhen Li _<_ yl494@cam.ac.uk _>_ .



sion model is given an adversarial image (studied below),
or to tackle many problems in AI safety (Amodei et al.,
2016). With model uncertainty at hand, applications as farreaching as safety in self-driving cars can be explored, using models which can propagate their uncertainty up the
decision making pipeline (Gal, 2016). With deterministic
deep learning models this invaluable uncertainty information is often lost.


Bayesian deep learning – an approach to combining
Bayesian probability theory together with deep learning –
allows us to use state-of-the-art models and at the same

time obtain model uncertainty (Gal, 2016; Gal & Ghahramani, 2016a). Originating in the 90s (Neal, 1995; MacKay,
1992; Denker & LeCun, 1991), Bayesian neural networks
(BNNs) in particular have started gaining in popularity
again (Graves, 2011; Blundell et al., 2015; HernandezLobato & Adams, 2015). BNNs are standard neural networks (NNs) with prior probability distributions placed
over their weights. Given observed data, inference is
then performed to find what are the more likely and less
likely weights to explain the data. But as easy it is to
formulate BNNs, is as difficult to perform inference in
them. Many approximations have been proposed over the
years (Denker & LeCun, 1991; Neal, 1995; Graves, 2011;
Blundell et al., 2015; Hernandez-Lobato & Adams, 2015;
Hern´andez-Lobato et al., 2016), some more practical and
some less practical. A practical approximation for inference in Bayesian neural networks should be able to scale
well to large data and complex models (such as convolutional neural networks (CNNs) (Rumelhart et al., 1985; LeCun et al., 1989)). Much more important perhaps, it would
be impractical to change existing model architectures that
have been well studied, and it is often impractical to work
with complex and cumbersome techniques which are difficult to explain to non-experts. Many existing approaches
to obtain model confidence often do not scale to complex
models or large amounts of data, and require us to develop
new models for existing tasks for which we already have
well performing tools (Gal, 2016).


One possible solution for practical inference in BNNs is
variational inference (VI) (Jordan et al., 1999), a ubiquitous
technique for approximate inference. Dropout variational
distributions in particular (a mixture of two Gaussians with
small standard deviations, and with one component fixed at


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



zero) can be used to obtain a practical inference technique
(Gal & Ghahramani, 2016b). These have been used for machine vision and medical applications (Kendall & Cipolla,
2016; Kendall et al., 2015; Angermueller & Stegle, 2015;
Yang et al., 2016). Dropout variational inference can be
implemented by adding dropout layers (Hinton et al., 2012;
Srivastava et al., 2014) before every weight layer in the NN
model. Inference is then carried out by Monte Carlo (MC)
integration over the variational distribution, in practice implemented by simulating stochastic forward passes through
the model at test time (referred to as MC dropout). Although dropout VI is a practical technique for approximate
inference, it also has some major limitations. Dropout VI
can severely underestimate model uncertainty (Gal, 2016,
Section 3.3.2) – a property many VI methods share (Turner
& Sahani, 2011). This can lead to devastating results in applications that _must rely_ on good uncertainty estimates such
as AI safety applications.


Alternative objectives to VI’s objective are therefore needed. Black-box _α_ -divergence minimisation
(Hern´andez-Lobato et al., 2016; Li & Turner, 2016; Minka,
2005) is a class of approximate inference methods extending on VI, approximating EP’s energy function (Minka,
2001) as well as the Hellinger distance (Hellinger, 1909).
These were proposed as a solution to some of the difficulties encountered with VI. However, the main difficulty
with _α_ -divergences is that the divergences are hard to use in
practice. Existing inference techniques only use Gaussian
approximating distributions, with the density over the approximation having to be evaluated explicitly many times.
The objective offers a limited intuitive interpretation which
is difficult to explain to non-experts, and of limited use
for engineers (Gal, 2016, Section 2.2.2). Perhaps more
important, current _α_ -divergence inference techniques require existing models and code-bases to be changed radically to perform inference in the Bayesian counterpart to
these models. To implement a complex CNN structure with
the inference and code of (Hern´andez-Lobato et al., 2016),
for example, one would be required to re-implement many
already-implemented software tools.


In this paper we propose a re-parametrisation of the induced _α_ -divergence objectives, and by relying on some
mild assumptions (which we justify below), derive a simple approximate inference technique which can easily be
implemented with existing models. Further, we rely on the
dropout approximate variational distribution and demonstrate how inference can be done in a practical way – requiring us to _only change the loss of the NN, L_ ( _θ_ ) _, and_
_to perform multiple stochastic forward passes at training_
_time_ . In particular, given _l_ ( _·, ·_ ) some standard NN loss such
as cross entropy or the Euclidean loss, and _{_ **f** _**[ω]**_ [�] _[k]_ ( _**x**_ _n_ ) _}_ _[K]_ _k_ =1
a set of _K_ stochastic dropout network outputs on input **x** _n_
with randomly masked weights � _**ω**_ _k_, our proposed objective



with _α_ a real number, _θ_ the set of network weights to
be optimised, and an _L_ 2 regulariser over _θ_ . By selecting
_α_ = 1 this objective directly optimises the per-point predictive log-likelihood, while picking _α →_ 0 would focus
on increasing the training accuracy, recovering VI.


Specific choices of _α_ will result in improved uncertainty estimates (and accuracy) compared to VI in dropout BNNs,
without slowing convergence time. We demonstrate this
through a myriad of applications, including an assessment
of fully connected NNs in regression and classification, and
an assessment of Bayesian CNNs. Finally, we study the
uncertainty estimates resulting from our approximate inference technique. We show that our models’ uncertainty
increases on adversarial images generated from the MNIST
dataset, suggesting that these lie outside of the training data
distribution. This in practice allows us to tell-apart such
adversarial images from non-adversarial images by examining epistemic model uncertainty.


**2. Background**


We review background in Bayesian neural networks and
approximate variational inference. In the next section we
discuss _α_ -divergences.


**2.1. Bayesian Neural Networks**


Given training inputs **X** = _{_ **x** 1 _, . . .,_ **x** _N_ _}_ and their corresponding outputs **Y** = _{_ **y** 1 _, . . .,_ **y** _N_ _}_, in parametric
Bayesian regression we would like to infer a distribution
over parameters _ω_ of a function **y** = **f** _[ω]_ ( **x** ) that could have
generated the outputs. Following the Bayesian approach, to
find parameters that could have generated our data, we put
some _prior_ distribution over the space of parameters _p_ 0 ( _ω_ ).
This distribution captures our prior belief as to which parameters are likely to have generated our outputs before
observing any data. We further need to define a probability distribution over the outputs given the inputs _p_ ( **y** _|_ **x** _, ω_ ).
For classification tasks we assume a softmax likelihood,


_p_ � _y|_ **x** _, ω_ � = Softmax ( **f** _[ω]_ ( **x** ))


or a Gaussian likelihood for regression. Given a dataset
**X** _,_ **Y**, we then look for the _posterior_ distribution over the
space of parameters: _p_ ( _ω|_ **X** _,_ **Y** ). This distribution captures
how likely the function parameters are, given our observed
data. With it we can predict an output for a new input point
**x** _[∗]_ by integrating


_p_ ( **y** _[∗]_ _|_ **x** _[∗]_ _,_ **X** _,_ **Y** ) = _p_ ( **y** _[∗]_ _|_ **x** _[∗]_ _, ω_ ) _p_ ( _ω|_ **X** _,_ **Y** )d _ω._ (1)
�



is:


_L_ ( _θ_ ) = _−_ [1]

_α_



� log-sum-exp � _−α · l_ ( _y_ _n_ _,_ **f** _**[ω]**_ [�] _[k]_ ( _**x**_ _n_ ))� + _L_ 2 ( _θ_ )


_n_


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



One way to define a distribution over a parametric set of
functions is to place a prior distribution over a _neural net-_
_work’s_ weights _ω_ = _{_ **W** _i_ _}_ _[L]_ _i_ =1 [, resulting in a] _[ Bayesian NN]_
(MacKay, 1992; Neal, 1995). Given weight matrices **W** _i_
and bias vectors **b** _i_ for layer _i_, we often place standard matrix Gaussian prior distributions over the weight matrices,
_p_ 0 ( **W** _i_ ) = _N_ ( **W** _i_ ; **0** _,_ **I** ) and often assume a point estimate
for the bias vectors for simplicity.


**2.2. Approximate Variational Inference in Bayesian**
**Neural Networks**


In approximate inference, we are interested in finding the
distribution of weight matrices (parametrising our functions) that have generated our data. This is the posterior
over the weights given our observables **X** _,_ **Y** : _p_ ( _ω|_ **X** _,_ **Y** ),
which is not tractable in general. Existing approaches to
approximate this posterior are through _variational infer-_
_ence_ (as was done in Hinton & Van Camp (1993); Barber
& Bishop (1998); Graves (2011); Blundell et al. (2015)).
We need to define an approximating variational distribution
_q_ _θ_ ( _ω_ ) (parametrised by variational parameters _θ_ ), and then
minimise w.r.t. _θ_ the KL divergence (Kullback & Leibler,
1951; Kullback, 1959) between the approximating distribution and the full posterior:


KL� _q_ _θ_ ( _ω_ ) _||p_ ( _ω|_ **X** _,_ **Y** )� _∝−_ _q_ _θ_ ( _ω_ ) log _p_ ( **Y** _|_ **X** _, ω_ )d _ω_
�

+ KL( _q_ _θ_ ( _ω_ ) _||p_ 0 ( _ω_ ))



cording to a mixture of two Gaussians with small variances
and the mean of one of the Gaussians fixed at zero. The uncertainty in the weights induces prediction uncertainty by
marginalising over the approximate posterior using Monte
Carlo integration:


_p_ ( _y_ = _c|_ **x** _,_ **X** _,_ **Y** ) = _p_ ( _y_ = _c|_ **x** _, ω_ ) _p_ ( _ω|_ **X** _,_ **Y** )d _ω_
�


_≈_ _p_ ( _y_ = _c|_ **x** _, ω_ ) _q_ _θ_ ( _ω_ )d _ω_
�



_≈_ [1]

_K_



_K_
� _p_ ( _y_ = _c|_ **x** _,_ � _**ω**_ _k_ )


_k_ =1



with � _**ω**_ _k_ _∼_ _q_ _θ_ ( _ω_ ), where _q_ _θ_ ( _ω_ ) is the Dropout distribution (Gal, 2016). Given its popularity, we concentrate on
the dropout stochastic regularisation technique throughout
the rest of the paper, although any other stochastic regularisation technique could be used instead (such as multiplicative Gaussian noise (Srivastava et al., 2014) or dropConnect
(Wan et al., 2013)).


Dropout VI is an example of practical approximate inference, but it also underestimates model uncertainty (Gal,
2016, Section 3.3.2). This is because minimising the KL divergence between _q_ ( _ω_ ) and _p_ ( _ω|_ **X** _,_ **Y** ) penalises _q_ ( _ω_ ) for
placing probability mass where _p_ ( _ω|_ **X** _,_ **Y** ) has no mass,
but does not penalise _q_ ( _ω_ ) for not placing probability mass
at locations where _p_ ( _ω|_ **X** _,_ **Y** ) _does have mass_ . We next
discuss _α_ -divergences as an alternative to the VI objective.


**3. Black-box** _α_ **-divergence minimisation**


In this section we provide a brief review of the _black box al-_
_pha_ (BB- _α_, Hern´andez-Lobato et al. (2016)) method upon
which the main derivation in this paper is based. Consider
approximating the following distribution:



= _−_



_N_
�


_i_ =1



_q_ _θ_ ( _ω_ ) log _p_ ( **y** _i_ _|_ **f** _[ω]_ ( **x** _i_ ))d _ω_
�



+ KL( _q_ _θ_ ( _ω_ ) _||p_ 0 ( _ω_ )) _,_ (2)


where _A ∝_ _B_ is slightly abused here to denote equality up
to an additive constant (w.r.t. variational parameters _θ_ ).


**2.3. Dropout Approximate Inference**


Given a (deterministic) neural network, stochastic regularisation techniques in the model (such as dropout (Hinton
et al., 2012; Srivastava et al., 2014)) can be interpreted
as variational Bayesian approximations in a Bayesian NN
with the same network structure (Gal & Ghahramani,
2016b). This is because applying a stochastic regularisation technique is equivalent to multiplying the NN weight
matrices **M** _i_ by some random noise _**ϵ**_ _i_ (with a new noise
realisation for each data point). The resulting stochastic
weight matrices **W** _i_ = _**ϵ**_ _i_ **M** _i_ can be seen as draws from the
approximate posterior over the BNN weights, replacing the
deterministic NN’s weight matrices **M** _i_ . Our set of variational parameters is then the set of matrices _θ_ = _{_ **M** _i_ _}_ _[L]_ _i_ =1 [.]
For example, dropout can be seen as an approximation to
Bayesian NN inference with _dropout approximating distri-_
_butions_, where the rows of the matrices **W** _i_ distribute ac


We provide details of _α_ -divergences and local approximation methods in the appendix, and in the rest of the paper
we consider three special cases in this rich family:



_p_ ( _ω_ ) = [1]



_f_ _n_ ( _ω_ ) _._

_n_



_Z_ _[p]_ [0] [(] _[ω]_ [)] �



In Bayesian neural networks context, these factors _f_ _n_ ( _ω_ )
represent the likelihood terms _p_ ( **y** _n_ _|_ **x** _n_ _, ω_ ), _Z_ = _p_ ( **Y** _|_ **X** ),
and the approximation target _p_ ( _ω_ ) is the exact posterior
_p_ ( _ω|_ **X** _,_ **Y** ). Popular methods of approximate inference include variational inference (VI) (Jordan et al., 1999) and
expectation propagation (EP) (Minka, 2001), where these
two algorithms are special cases of power EP (Minka,
2004) that minimises Amari’s _α_ -divergence (Amari, 1985)
D _α_ [ _p||q_ ] in a _local_ way:



1
D _α_ [ _p||q_ ] =
_α_ (1 _−_ _α_ )



1 _−_ _p_ ( _ω_ ) _[α]_ _q_ ( _ω_ ) [1] _[−][α]_ _dω_ _._
� � �


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



1. Exclusive KL divergence:


D 0 [ _p||q_ ] = KL[ _q||p_ ] = E _q_


2. Hellinger distance:



log _[q]_ [(] _[ω]_ [)] ;
� _p_ ( _ω_ ) �



distance seems to provide a good balance between zeroforcing and mass-covering, and empirically it has been
found to achieve the best performance.


Given the success of _α_ -divergence methods, it is a natural
idea to extend these algorithms to other classes of approximations such as dropout. However this task is non-trivial.
First, the original formulation of BB- _α_ energy is an ad hoc
adaptation of power-EP energy (see appendix), which applies to exponential family _q_ distributions only. Second,
the energy function offers a limited intuitive interpretation
to non-experts, thus of limited use for practitioners. Third
and most importantly, a naive implementation of BB- _α_ using dropout would bring in a prohibitive computational burden. To see this, we first review the BB- _α_ energy function
in the general case (Li & Turner, 2016) given _α ̸_ = 0:



D 0 _._ 5 [ _p||q_ ] = 4Hel [2] [ _q||p_ ] = 2
���


3. Inclusive KL divergence:



2
_q_ ( _ω_ ) _dω_ ;
�



_p_ ( _ω_ ) _−_ �



D 1 [ _p||q_ ] = KL[ _p||q_ ] = E _p_



log _[p]_ [(] _[ω]_ [)] _._
� _q_ ( _ω_ ) �



Since _α_ = 0 is used in VI and _α_ = 1 _._ 0 is used in EP, in
later sections we will also refer to these alpha settings as
the VI value, Hellinger value, and EP value, respectively.


Power-EP, though providing a generic variational framework, does not scale with big data. It maintains approximating factors attached to every likelihood term _f_ _n_ ( _ω_ ),
resulting in space complexity _O_ ( _N_ ) for the posterior approximation which is clearly undesirable. The recently proposed stochastic EP (Li et al., 2015) and BB- _α_ (Hern´andezLobato et al., 2016) inference methods reduce this memory
overhead to _O_ (1) by sharing these approximating factors.
Moreover, optimisation in BB- _α_ is done by descending the
so called BB- _α_ energy function, where Monte Carlo (MC)
methods and automatic differentiation are also deployed to
allow fast prototyping.


BB- _α_ has been successfully applied to Bayesian neural
networks for regression, classification (Hern´andez-Lobato
et al., 2016) and model-based reinforcement learning (Depeweg et al., 2016). They all found that using _α ̸_ = 0 often
returns better approximations than the VI case. The reasons
for the worse results of VI are two fold. From the perspective of inference, the zero-forcing behaviour of exclusive
KL-divergences enforces the _q_ distribution to be zero in the
region where the exact posterior has zero probability mass.
Thus VI often fits to a local mode of the exact posterior and
is over-confident in prediction. On hyper-parameter learning point of view, as the variational lower-bound is used as
a (biased) approximation to the maximum likelihood objective, the learned model could be biased towards oversimplified cases (Turner & Sahani, 2011). These problems
could potentially be addressed by using _α_ -divergences. For
example, inclusive KL encourages the coverage of the support set (referred as mass-covering), and when used in local divergence minimisation (Minka, 2005), it can fit an
approximation to a mode of _p_ ( _ω_ ) with better estimates of
uncertainty. Moreover the BB- _α_ energy provides a better
approximation to the marginal likelihood as well, meaning
that the learned model will be less biased and thus fitting
the data distribution better (Li & Turner, 2016). Hellinger



(4)

�
with _**ω**_ _k_ _∼_ _q_ ( _ω_ ). This is a biased approximation as the
expectation in (3) is computed before taking the logarithm.
But empirically Hern´andez-Lobato et al. (2016) showed
that the bias introduced by the MC approximation is often dominated by the variance of the samples, meaning that
the effect of the bias is negligible. When _α →_ 0 it returns
the _variational free energy_ (the VI objective)


_L_ 0 ( _q_ ) = _L_ VFE ( _q_ ) = KL[ _q||p_ 0 ] _−_ � E _q_ [log _f_ _n_ ( _ω_ )] _,_ (5)


_n_


and the corresponding MC approximation _L_ [MC] VFE [becomes]
an unbiased estimator of _L_ VFE . Also _L_ [MC] _α_ _→L_ [MC] VFE [as the]
number of samples _K →_ 1.


The original paper (Hern´andez-Lobato et al., 2016) proposed a naive implementation which directly evaluates the
MC estimation (4) with samples � _**ω**_ _k_ _∼_ _q_ ( _ω_ ). However
as discussed before, dropout implicitly samples different
masked weight matrices � _**ω**_ _∼_ _q_ for different data points.
This indicates that the naive approach, when applied to
dropout approximation, would gather all these samples for
all _M_ datapoints in a mini-batch (i.e. _MK_ sets of neural
network weight matrices in total), which brings prohibitive
cost if the network is wide and deep. Interestingly, the minimisation of the variational free energy ( _α_ = 0) with the
dropout approximation can be computed very efficiently.



_α_


_._ (3)

� �



_L_ _α_ ( _q_ ) = _−_ [1]

_α_



� log E _q_


_n_



�



1
_f_ _n_ ( _ω_ ) _p_ 0 ( _ω_ ) _N_


1

�� _q_ ( _ω_ ) _N_



One could verify that this is the same energy function as
presented in (Hern´andez-Lobato et al., 2016) by considering _q_ an exponential family distribution. In practice (3)
might be intractable, hence an MC approximation is introduced:



� _α_ �



_L_ [MC] _α_ [(] _[q]_ [) =] _[ −]_ [1]

_α_



log [1]

_K_

_n_



�



_K_



_f_ _n_ ( _**ω**_ � _k_ ) _p_ 0 ( _**ω**_ � _k_ ) _N_ 1


� 1

�� _q_ ( _**ω**_ _k_ ) _N_



�


_k_


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



The main reason for this success is due to the additive struc
ture of the variational free energy: no evaluation of _q_ density is required if the “regulariser” KL[ _q||p_ 0 ] can be computed/approximated efficiently. In the following section we
propose an improved version of BB- _α_ energy to allow applications with dropout and other flexible approximation

structures.


**4. A New Reparameterisation of BB-** _α_ **Energy**


We propose a reparamterisation of the BB- _α_ energy to reduce the computational overhead, which uses the so called
“cavity distributions”. First we denote ˜ _q_ ( _ω_ ) as a free-form
cavity distribution, and write the approximate posterior _q_

as



and by taking _α →_ 0 the variational free-energy (5) is again
recovered.


Given a loss function _l_ ( _·, ·_ ), e.g. _l_ 2 loss in regression or
cross entropy in classification, we can define the (unnormalised) likelihood term _f_ _n_ ( _ω_ ) _∝_ _p_ ( **y** _n_ _|_ _**x**_ _n_ _, ω_ ) _∝_
exp[ _−l_ ( **y** _n_ _,_ **f** _[ω]_ ( _**x**_ _n_ ))], e.g. see (LeCun et al., 2006) [1] .
Swapping _f_ _n_ ( _ω_ ) for this last expression, and approximating the expectation over _q_ using Monte Carlo sampling, we
obtain our proposed minimisation objective:


_L_ ˜ [MC] _α_ [(] _[q]_ [) = KL[] _[q][||][p]_ [0] [] +][ const] (7)



_−_ [1]


_α_



� log-sum-exp[ _−αl_ ( _y_ _n_ _,_ **f** _**[ω]**_ [�] _[k]_ ( _**x**_ _n_ ))]


_n_




[1] _q_ ˜( _ω_ ) _q_ ˜( _ω_ )

_Z_ _q_ � _p_ 0 ( _ω_ )



_q_ ( _ω_ ) = [1]



_p_ 0 ( _ω_ )



_α_
_N_ _−α_
_,_ (6)
�



where we assume _Z_ _q_ _<_ + _∞_ is the normalising constant
to ensure _q_ a valid distribution. When _α/N →_ 0, the unnormalised density in (6) converges to ˜ _q_ ( _ω_ ) for every _ω_,
and _Z_ _q_ _→_ 1 by the assumption of _Z_ _q_ _<_ + _∞_ (Van Erven & Harremo¨es, 2014). Hence _q →_ _q_ ˜ when _α/N →_ 0,
and this happens for example when we choose _α →_ 0, or
_N →_ + _∞_ as well as when _α_ grows sub-linearly to _N_ .
Now we rewrite the BB-alpha energy in terms of ˜ _q_ :



_L_ _α_ ( _q_ ) = _−_ [1]

_α_



1 ˜ _q_ ˜( _ω_ )
log _q_ ( _ω_ )
_n_ � [�] _Z_ _q_ � _p_ 0 ( _ω_ )



�



_p_ 0 ( _ω_ )



� _Nα−α_ [�] [1] _[−]_ _N_ _[α]_



_α_
_p_ 0 ( _ω_ ) _N_ _f_ _n_ ( _ω_ ) _[α]_ _dω_



= _[N]_



˜ _q_ ˜( _ω_ )

_[α]_ _q_ ( _ω_ )

_N_ [) log] � � _p_ 0 ( _ω_ )



_α_ [(1] _[ −]_ _N_ _[α]_



_p_ 0 ( _ω_ )



_α_
_N_ _−α_
_dω_
�



_−_ [1]


_α_



� log E _q_ ˜ [ _f_ _n_ ( _ω_ ) _[α]_ ]


_n_



with log-sum-exp being the log-sum-exp operator over _K_
samples from the approximate posterior � _**ω**_ _k_ _∼_ _q_ ( _ω_ ). This
objective function also approximates the marginal likelihood. Therefore, compared to the original formulation (3),
the improved version (7) is considerably simpler (both to
implement and to understand), has a similar form to standard objective functions used in deep learning research, yet
remains an approximate Bayesian inference algorithm.


To gain some intuitive understanding of this objective, we
observe what it reduces to for different _α_ and _K_ settings.
By selecting _α_ = 1 the per-point predictive log-likelihood
log E _q_ [ _p_ ( _y_ _n_ _|_ **x** _n_ _, ω_ )] is directly optimised. On the other
hand, picking the VI value ( _α →_ 0) would focus on increasing the training accuracy E _q_ [log _p_ ( _y_ _n_ _|_ **x** _n_ _, ω_ )]. The
Hellinger value could be used to achieve a balance between
reducing training error and improving predictive likelihood, which has been found to be desirable (Hern´andezLobato et al., 2016; Depeweg et al., 2016). Lastly, for
_K_ = 1 the log-sum-exp disappears, the _α_ ’s cancel out, and
the original (stochastic) VI objective is recovered.


In summary, our proposal modifies the loss function by
multiplying it by _α_ and then performing log-sum-exp with
a sum over multiple stochastic forward passes sampled
from the BNN approximate posterior. The remaining KLdivergence term (between _q_ and the prior _p_ ) can often be
approximated. It can be viewed as a regulariser added to
the objective function, and reduces to _L_ 2 -norm regulariser
for certain popular _q_ choices (Gal, 2016).


**4.1. Dropout BB-** _α_


We now provide a concrete example where the approximate
distribution is defined by dropout. With dropout VI, MC
samples are used to approximate the expectation w.r.t. _q_,
which in practice is implemented as performing _stochastic_
_forward passes_ through the dropout network – i.e. given an


1 We note that _f_ _n_ ( _ω_ ) does not need to be a normalised density of _y_ _n_ unless one would like to optimise the hyper parameters
associated with _f_ _n_ .



= R _β_ [˜ _q||p_ 0 ] _−_ [1]

_α_



_N_

� log E _q_ ˜ [ _f_ _n_ ( _ω_ ) _[α]_ ] _, β_ = _N −_ _α_ _[,]_

_n_



where R _β_ [˜ _q||p_ 0 ] represents the _R´enyi divergence_ (R´enyi
(1961), discussed in the appendix) of order _β_ . We note

_L_ again that when VFE (˜ _q_ ) as well as _N_ _[α]_ _q_ _[→]_ _→_ [0][ the new energy] _q_ ˜. More importantly, _[ L]_ _[α]_ [(˜] _[q]_ [)] R [ converges to] _β_ [˜ _q||p_ 0 ] _→_
KL[˜ _q||p_ 0 ] = KL[ _q||p_ 0 ] provided R _β_ [˜ _q||p_ 0 ] _<_ + _∞_ (which
holds when assuming _Z_ _q_ _<_ + _∞_ ) and _N_ _[α]_ _[→]_ [0][.]


This means that for a constant _α_ that scales sub-linearly
with _N_, in large data settings we can further approximate
the BB- _α_ energy as



_L_ _α_ ( _q_ ) _≈_ _L_ [˜] _α_ ( _q_ ) = KL[ _q||p_ 0 ] _−_ [1]

_α_



� log E _q_ [ _f_ _n_ ( _ω_ ) _[α]_ ] _._


_n_



Note that here we also use the fact that now _q ≈_ _q_ ˜. Critically, the proposed reparameterisation is continuous in _α_,


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



input **x**, the input is fed through the network and a new
dropout mask is sampled and applied at each dropout layer.
This gives a stochastic output – a sample from the dropout
network on the input **x** . A similar approximation is used
in our case as well, where to implement the MC sampling
in eq. (7) we perform multiple stochastic forward passes
through the network.


Recall the neural network **f** _[ω]_ ( **x** ) is parameterised by the
variable _ω_ . In classification, cross entropy is often used as
the loss function

� _l_ ( **y** _n_ _,_ **p** _[ω]_ ( **x** _n_ )) = � _−_ **y** _n_ _[T]_ [log] **[ p]** _[ω]_ [(] **[x]** _[n]_ [)] _[,]_ (8)



_l_ ( **y** _n_ _,_ **p** _[ω]_ ( **x** _n_ )) = �

_n_ _n_



_−_
**y** _n_ _[T]_ [log] **[ p]** _[ω]_ [(] **[x]** _[n]_ [)] _[,]_ (8)

_n_



**p** _[ω]_ ( **x** _n_ ) = Softmax( **f** _[ω]_ ( _**x**_ _n_ )) _,_


where the label **y** _n_ is a one-hot binary vector, and the network output Softmax( **f** _[ω]_ ( **x** _n_ )) encodes the probability vector of class assignments. Applying the re-formulated BB- _α_
energy (7) with a Bayesian equivalent of the network, we
arrive at the objective function



_p_ _i_ _||_ **M** _i_ _||_ 2 [2] _[−]_ [1]

_α_

_i_



_K_



_L_ ˜ [MC] _α_ [(] _[q]_ [) =] �



_α_



�( **p** _**[ω]**_ [�] _[k]_ ( **x** _n_ )) _[α]_


_k_



�



**y** _n_ _[T]_ [log] _K_ [1]
_n_



= [1]


_α_



� _l_


_n_



�



**y** _n_ _,_ [1]



**def** softmax_cross_ent_with_mc_logits(alpha):

**def** loss(y_true, mc_logits):

_# mc_logits: MC samples of shape MxKxD_
mc_log_softmax = mc_logits \

   - K.max(mc_logits, axis=2, keepdims=True)
mc_log_softmax = mc_log_softmax - \
logsumexp(mc_log_softmax, 2)
mc_ll = K.sum(y_true*mc_log_softmax,-1)
**return** -1./alpha * (logsumexp(alpha * \
mc_ll, 1) + K.log(1.0 / K_mc))

**return** loss


_Figure 1._ Code snippet for our induced classification loss.


**5. Experiments**


We test the reparameterised BB- _α_ on Bayesian NNs with
the dropout approximation. We assess the proposed inference in regression and classification tasks on standard
benchmarking datasets, comparing different values of _α_ .
We further assess the training time trade-off between our
technique and VI, and study the properties of our model’s
uncertainty on out-of-distribution data points. This last experiment leads us to propose a technique that could be used
to identify adversarial image attacks.


**5.1. Regression**


The first experiment considers Bayesian neural network regression with approximate posterior induced by dropout.
We use benchmark UCI datasets [2] that have been tested

in related literature. The model is a single-layer neural network with 50 ReLU units for all datasets except
for Protein and Year, which use 100 units. We consider
_α ∈{_ 0 _._ 0 _,_ 0 _._ 5 _,_ 1 _._ 0 _}_ in order to examine the effect of masscovering/zero-forcing behaviour in dropout. MC approximation with _K_ = 10 samples is also deployed to compute
the energy function. Other initialisation settings are largely
taken from (Li & Turner, 2016).


We summarise the test negative log-likelihood (LL) and
RMSE with standard error (across different random splits)
for selected datasets in Figure 2 and 3, respectively. The
full results are provided in the appendix. Although optimal _α_ may vary for different datasets, using non-VI values has significantly improved the test-LL performances,
while remaining comparable in test error metric. In particular, _α_ = 0 _._ 5 produced overall good results for both test
LL and RMSE, which is consistent with previous findings.
As a comparison we also include test performances of a
BNN with a Gaussian approximation (VI-G) (Li & Turner,
2016), a BNN with HMC, and a sparse Gaussian process
model with 50 inducing points (Bui et al., 2016). In testLL metric our best dropout model out-performs the Gaus

2 [http://archive.ics.uci.edu/ml/datasets.](http://archive.ics.uci.edu/ml/datasets.html)

[html](http://archive.ics.uci.edu/ml/datasets.html)



_K_



� **p** _**[ω]**_ [�] _[k]_ ( **x** _n_ ) _[α]_

_k_ �



+ � _L_ 2 ( **M** _i_ )


_i_



(9)


with _{_ **p** _**[ω]**_ [�] _[k]_ ( **x** _n_ ) _}_ _[K]_ _k_ =1 [being] _[ K]_ [ stochastic network outputs]
on input **x** _n_, _p_ _i_ equals to one minus the dropout rate of
the _i_ th layer, and the _L_ 2 regularization terms coming from
an approximation to the KL-divergence (Gal, 2016). I.e.
we raise network probability outputs to the power _α_ and
average them as an input to the standard cross entropy loss.
Taking _α ̸_ = 1 can be viewed as training the neural network
with an adjusted “power” loss, regularized by an _L_ 2 norm.
Implementing this induced loss with Keras (Chollet, 2015)
is as simple as a few lines of Python. A code snippet is
given in Figure 1, with more details in the appendix.


In regression problems, the loss function is defined as
_l_ ( _**y**_ _,_ **f** _[ω]_ ( _**x**_ )) = _[τ]_ 2 _[||]_ _**[y]**_ _[ −]_ **[f]** _[ ω]_ [(] _**[x]**_ [)] _[||]_ 2 [2] [and the likelihood term can]

be interpreted as **y** _∼N_ ( _**y**_ ; **f** _[ω]_ ( _**x**_ ) _, τ_ _[−]_ [1] _**I**_ ). Plugging this
into the energy function returns the following objective



_L_ ˜ [MC] _α_ [(] _[q]_ [) =] _[ −]_ [1]

_α_



_−_ _[ατ]_
log-sum-exp
� 2
_n_



�



2
2 _[||]_ **[y]** _[n]_ _[ −]_ **[f]** [ �] _**[ω]**_ _[k]_ [(] _**[x]**_ _[n]_ [)] _[||]_ [2] �



+ _[ND]_



_p_ _i_ _||_ _**M**_ _i_ _||_ 2 [2] _[,]_ (10)

_i_



2 log _τ_ + �



with _{_ **f** _**[ω]**_ [�] _[k]_ ( **x** _n_ ) _}_ _[K]_ _k_ =1 [being] _[ K]_ [ stochastic forward passes on]
input **x** _n_ . Again, this is reminiscent of the _l_ 2 objective in
standard deep learning, and can be implemented by simply passing the input through the dropout network multiple
times, collecting the stochastic outputs, and feeding the set
of outputs through our new BB-alpha loss function.


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



_Figure 2._ Negative test-LL results for Bayesian NN regression.
The lower the better. Best viewed in colour.


sian approximation method on almost all datasets, and for
some datasets is on par with HMC which is the current gold
standard for Bayesian neural works, and with the GP model
that is known to be superior in regression.


**5.2. Classification**


We further experiment with a classification task, comparing
the accuracy of the various _α_ values on the MNIST benchmark (LeCun & Cortes, 1998). We assessed a fully connect
NN with 2 hidden layers and 100 units in each layer. We
used dropout probability 0.5 and _α ∈{_ 0 _,_ 0 _._ 5 _,_ 1 _}_ . Again,
we use _K_ = 10 samples at training time for all _α_ values,
and _K_ test = 100 samples at test time. We use weight decay
10 _[−]_ [6], which is equivalent to prior lengthscale _l_ [2] = 0 _._ 1 (Gal
& Ghahramani, 2016b). We repeat each experiment three
times and plot mean and standard error. Test RMSE as well
as test log likelihood are given in Figure 4. As can be seen,
Hellinger value _α_ = 0 _._ 5 gives best test RMSE, with test
log likelihood matching that of the EP value _α_ = 1. The
VI value _α_ = 0 under-performs according to both metrics.


We next assess a convolutional neural network model



_Figure 3._ Test RMSE results for Bayesian NN regression. The
lower the better. Best viewed in colour.


(CNN). For this experiment we use the standard CNN example given in (Chollet, 2015) with 32 convolution filters,
100 hidden units at the top layer, and dropout probability
0 _._ 5 before each fully-connected layer. Other settings are as
before. Average test accuracy and test log likelihood are
given in Figure 5. In this case, VI value _α_ = 0 seems to
supersede the EP value _α_ = 1, and performs similarly to
the Hellinger value _α_ = 0 _._ 5 according to both metrics.


**5.3. Detecting Adversarial Examples**


The third set of experiments considers adversarial attacks
on dropout trained Bayesian neural networks. Bayesian
neural networks’ uncertainty increases on examples far
from the data distribution. We test the hypothesis that certain techniques for generating adversarial examples will
give images that lie outside of the image manifold, i.e. far
from the data distribution (note though that there exist techniques that will guarantee the images staying near the data
manifold, by minimising the perturbation used to construct
the adversarial example). By assessing our BNN uncertainty, we should see increased uncertainty for adversarial
images if they indeed lie outside of the training data distri














(a) Fully connected NN test ac
curacy



(b) Fully connected NN test log
likelihood





(a) CNN test accuracy





(b) CNN test log likelihood



_Figure 4._ MNIST test accuracy and test log likelihood for a fully
connected NN in a classification task.



_Figure 5._ MNIST test accuracy and test log likelihood for a convolutional neural network in a classification task.


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**


target class accuracy


original class accuracy



_Figure 6._ Un-targeted attack: classification accuracy results as a
function of perturbation stepsize. The adversarial examples are
shown for (from top to bottom) NN and BNN trained with dropout
and _α_ = 0 _._ 0 _,_ 0 _._ 5 _,_ 1 _._ 0.


bution. The tested model is a fully connected network with
3 hidden layers of 1000 units. The dropout trained models
are also compared to a benchmark NN with the same architecture but trained by maximum likelihood. The adversarial examples are generated on MNIST test data that is
normalised to be in the range [0 _,_ 1]. For the dropout trained
networks we perform MC dropout prediction at test time
with _K_ test = 10 MC samples.


The first attack in consideration is the Fast Gradient Sign
(FGS) method (Goodfellow et al., 2014). This is an untargeted attack, which attempts to reduces the maximum
value of the predicted class label probability


**x** adv = **x** _−_ _η ·_ sgn( _∇_ **x** max log _p_ ( _y|_ **x** )) _._
_y_


We use the single gradient step FGS implemented in Cleverhans (Papernot et al., 2016) with the stepsize _η_ varied
between 0.0 and 0.5. The left panel in Figure 6 demonstrates the classification accuracy on adversarial examples,
which shows that the dropout networks, especially the one
trained with _α_ = 1 _._ 0, are significantly more robust to adversarial attacks compared to the deterministic NN. More
interestingly, the test data examples and adversarial images
can be told-apart by investigating the uncertainty representation of the dropout models. In the right panel of Figure
6 we depict the predictive entropy computed on the neural
network output probability vector, and show example corresponding adversarial images below the axis for each corresponding stepsize. Clearly the deterministic NN model
produces over-confident predictions on adversarial samples, e.g. it predicts the wrong label very confidently even
when the input is still visually close to digit “7” ( _η_ = 0 _._ 2).
While dropout models, though producing wrong labels, are
very uncertain about their predictions. This uncertainty
keeps increasing as we move away from the data manifold. Hence the dropout networks are much more immu


_Figure 7._ Targeted attack: classification accuracy results as a
function of the number of iterative gradient steps. The adversarial examples are shown for (from top to bottom) NN and BNN
trained with dropout and _α_ = 0 _._ 0 _,_ 0 _._ 5 _,_ 1 _._ 0.


nised from noise-corrupted inputs, as they can be detected
using uncertainty estimates in this example.


The second attack we consider is a targeted version of FSG
(Carlini & Wagner, 2016), which maximises the predictive
probability of a selected class instead. As an example, we
fix class 0 as the target and apply the iterative gradientbase attack to all non-zero digits in test data. At step _t_, the
adversarial output is computed as


**x** _[t]_ adv [=] **[ x]** _[t]_ adv _[−]_ [1] [+] _[ η][ ·]_ [ sgn][(] _[∇]_ **[x]** [ log] _[ p]_ [(] _[y]_ [target] _[|]_ **[x]** _[t]_ adv _[−]_ [1] [))] _[,]_


where the stepsize _η_ is fixed at 0 _._ 01 in this case. Results are
presented in the left panel of Figure 7, and again dropout
trained models are more robust to this attack compared with
the deterministically trained NN. Similarly these adversarial examples could be detected by the Bayesian neural networks’ uncertainty, by examining the predictive entropy.
By visually inspecting the generated adversarial examples
in the right panel of Figure 7, it is clear that the NN overconfidently classifies a digit 7 to class 0. On the other hand,
the dropout models are still fairly uncertain about their predictions even after 40 gradient steps. More interestingly,
running this iterative attack on dropout models produces a
smooth interpolation between different digits, and when the
model is confident on predicting the target class, the corresponding adversarial images are visually close to digit zero.


These initial results suggest that assessing the epistemic
uncertainty of classification models can be used as a viable technique to identify adversarial examples. We would
note though that we used this experiment to demonstrate
our techniques’ uncertainty estimates, and much more research is needed to solve the difficulties faced with adversarial inputs.


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



**5.4. Run time trade-off**


We finish the experiments section by assessing the running
time trade-offs of using an increasing number of samples at
training time. Unlike VI, in our inference we rely on a large
number of samples to reduce estimator bias. When a small
number of samples is used ( _K_ = 1) our method collapses
to standard VI. In Figure 8 we see both test accuracy as well
as test log likelihood for a fully connected NN with four
layers of 1024 units trained on the MNIST dataset, with
_α_ = 1. The two metrics are shown as a function of wallclock run time for different values of _K ∈{_ 1 _,_ 10 _,_ 100 _}_ . As
can be seen, _K_ = 1 converges to test accuracy of 98 _._ 8%
faster than the other values of _K_, which converge to the
same accuracy. On the other hand, when assessing test log
likelihood, both _K_ = 1 and _K_ = 10 attain value _−_ 600
within 1000 seconds, but _K_ = 10 continues improving its
test log likelihood and converges to value _−_ 500 after 3000
seconds. _K_ = 100 converges to the same value but requires
much longer running time, possibly because of noise from
other processes.









(a) Test accuracy





(b) Test log likelihood



_Figure 8._ Run time experiment on the MNIST dataset for different
number of samples _K_ .


**6. Conclusions**


We presented a practical extension of the BB-alpha objective which allows us to use the technique with dropout approximating distributions. The technique often supersedes
existing approximate inference techniques (even sparse
Gaussian processes), and is easy to implement. A code
snippet for our induced loss is given in the appendix.


**Acknowledgements**


YL thanks the Schlumberger Foundation FFTF fellowship
for supporting her PhD study.


**References**


Amari, Shun-ichi. _Differential-Geometrical Methods in Statistic_ .
Springer, New York, 1985.



Amodei, Dario, Olah, Chris, Steinhardt, Jacob, Christiano, Paul,
Schulman, John, and Mane, Dan. Concrete problems in ai
safety. _arXiv preprint arXiv:1606.06565_, 2016.


Angermueller, C and Stegle, O. Multi-task deep neural network to
predict CpG methylation profiles from low-coverage sequencing data. In _NIPS MLCB workshop_, 2015.


Barber, David and Bishop, Christopher M. Ensemble learning in
Bayesian neural networks. _NATO ASI SERIES F COMPUTER_
_AND SYSTEMS SCIENCES_, 168:215–238, 1998.


Blundell, Charles, Cornebise, Julien, Kavukcuoglu, Koray, and
Wierstra, Daan. Weight uncertainty in neural network. In
_ICML_, 2015.


Bui, Thang D, Hern´andez-Lobato, Daniel, Li, Yingzhen,
Hern´andez-Lobato, Jos´e Miguel, and Turner, Richard E. Deep
gaussian processes for regression using approximate expectation propagation. In _Proceedings of The 33rd International_
_Conference on Machine Learning (ICML)_, 2016.


Carlini, Nicholas and Wagner, David. Towards evaluating the robustness of neural networks. _arXiv preprint arXiv:1608.04644_,
2016.


Chollet, Francois. Keras. [https://github.com/](https://github.com/fchollet/keras)
[fchollet/keras, 2015.](https://github.com/fchollet/keras)


Denker, John and LeCun, Yann. Transforming neural-net output
levels to probability distributions. In _Advances in Neural In-_
_formation Processing Systems 3_ . Citeseer, 1991.


Depeweg, Stefan, Hern´andez-Lobato, Jos´e Miguel, Doshi-Velez,
Finale, and Udluft, Steffen. Learning and policy search in
stochastic dynamical systems with bayesian neural networks.
_arXiv preprint arXiv:1605.07127_, 2016.


Gal, Yarin. _Uncertainty in Deep Learning_ . PhD thesis, University
of Cambridge, 2016.


Gal, Yarin and Ghahramani, Zoubin. Bayesian convolutional neural networks with Bernoulli approximate variational inference.
_ICLR workshop track_, 2016a.


Gal, Yarin and Ghahramani, Zoubin. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.
_ICML_, 2016b.


Goodfellow, Ian J, Shlens, Jonathon, and Szegedy, Christian. Explaining and harnessing adversarial examples. _arXiv preprint_
_arXiv:1412.6572_, 2014.


Graves, Alex. Practical variational inference for neural networks.
In _Advances in Neural Information Processing Systems_, pp.
2348–2356, 2011.


Hellinger, Ernst. Neue begr¨undung der theorie quadratischer formen von unendlichvielen ver¨anderlichen. _Journal f¨ur die reine_
_und angewandte Mathematik_, 136:210–271, 1909.


Hernandez-Lobato, Jose Miguel and Adams, Ryan. Probabilistic
backpropagation for scalable learning of Bayesian neural networks. In _ICML_, 2015.


Hern´andez-Lobato, Jos´e Miguel, Li, Yingzhen, Hern´andezLobato, Daniel, Bui, Thang, and Turner, Richard E. Black-box
alpha divergence minimization. In _Proceedings of The 33rd In-_
_ternational Conference on Machine Learning_, pp. 1511–1520,
2016.


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**



Hinton, Geoffrey E and Van Camp, Drew. Keeping the neural
networks simple by minimizing the description length of the
weights. In _COLT_, pp. 5–13. ACM, 1993.


Hinton, Geoffrey E, Srivastava, Nitish, Krizhevsky, Alex,
Sutskever, Ilya, and Salakhutdinov, Ruslan R. Improving neural networks by preventing co-adaptation of feature detectors.
_arXiv preprint arXiv:1207.0580_, 2012.


Jordan, Michael I, Ghahramani, Zoubin, Jaakkola, Tommi S, and
Saul, Lawrence K. An introduction to variational methods for
graphical models. _Machine learning_, 37(2):183–233, 1999.


Kalchbrenner, Nal and Blunsom, Phil. Recurrent continuous
translation models. In _EMNLP_, 2013.


Kendall, Alex and Cipolla, Roberto. Modelling uncertainty in
deep learning for camera relocalization. In _2016 IEEE Inter-_
_national Conference on Robotics and Automation (ICRA)_, pp.
4762–4769. IEEE, 2016.


Kendall, Alex, Badrinarayanan, Vijay, and Cipolla, Roberto.
Bayesian segnet: Model uncertainty in deep convolutional
encoder-decoder architectures for scene understanding. _arXiv_
_preprint arXiv:1511.02680_, 2015.


Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey E. Imagenet classification with deep convolutional neural networks.
In _Advances in neural information processing systems_, pp.
1097–1105, 2012.


Kullback, Solomon. _Information theory and statistics_ . John Wiley
& Sons, 1959.


Kullback, Solomon and Leibler, Richard A. On information and
sufficiency. _The annals of mathematical statistics_, 22(1):79–
86, 1951.


LeCun, Yann and Cortes, Corinna. The mnist database of handwritten digits, 1998.


LeCun, Yann, Boser, Bernhard, Denker, John S, Henderson,
Donnie, Howard, Richard E, Hubbard, Wayne, and Jackel,
Lawrence D. Backpropagation applied to handwritten zip code
recognition. _Neural Computation_, 1(4):541–551, 1989.


LeCun, Yann, Chopra, Sumit, Hadsell, Raia, Ranzato, M, and
Huang, F. A tutorial on energy-based learning. _Predicting_
_structured data_, 1:0, 2006.


Li, Yingzhen and Turner, Richard E. R´enyi divergence variational
inference. In _NIPS_, 2016.


Li, Yingzhen, Hern´andez-Lobato, Jos´e Miguel, and Turner,
Richard E. Stochastic expectation propagation. In _Advances_
_in Neural Information Processing Systems (NIPS)_, 2015.


MacKay, David JC. A practical Bayesian framework for backpropagation networks. _Neural Computation_, 4(3):448–472,
1992.


Mikolov, Tom´aˇs, Karafi´at, Martin, Burget, Luk´aˇs, Cernock`y, Jan, [ˇ]
and Khudanpur, Sanjeev. Recurrent neural network based language model. In _Eleventh Annual Conference of the Interna-_
_tional Speech Communication Association_, 2010.


Minka, Tom. Divergence measures and message passing. Technical report, Microsoft Research, 2005.



Minka, T.P. Expectation propagation for approximate Bayesian
inference. In _Conference on Uncertainty in Artificial Intelli-_
_gence (UAI)_, 2001.


Minka, T.P. Power EP. Technical Report MSR-TR-2004-149,
Microsoft Research, 2004.


Neal, Radford M. _Bayesian learning for neural networks_ . PhD
thesis, University of Toronto, 1995.


Papernot, Nicolas, Goodfellow, Ian, Sheatsley, Ryan, Feinman, Reuben, and McDaniel, Patrick. cleverhans v1.0.0:
an adversarial machine learning library. _arXiv preprint_
_arXiv:1610.00768_, 2016.


R´enyi, Alfr´ed. On measures of entropy and information. _Fourth_
_Berkeley symposium on mathematical statistics and probabil-_
_ity_, 1, 1961.


Rumelhart, David E, Hinton, Geoffrey E, and Williams, Ronald J.
Learning internal representations by error propagation. Technical report, DTIC Document, 1985.


Sennrich, Rico, Haddow, Barry, and Birch, Alexandra. Edinburgh
neural machine translation systems for wmt 16. In _Proceedings_
_of the First Conference on Machine Translation_, pp. 371–376,
Berlin, Germany, August 2016. Association for Computational
Linguistics.


Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex, Sutskever,
Ilya, and Salakhutdinov, Ruslan. Dropout: A simple way to
prevent neural networks from overfitting. _The Journal of Ma-_
_chine Learning Research_, 15(1):1929–1958, 2014.


Sundermeyer, Martin, Schl¨uter, Ralf, and Ney, Hermann. LSTM
neural networks for language modeling. In _INTERSPEECH_,
2012.


Sutskever, Ilya, Vinyals, Oriol, and Le, Quoc VV. Sequence to
sequence learning with neural networks. In _NIPS_, 2014.


Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet, Pierre,
Reed, Scott, Anguelov, Dragomir, Erhan, Dumitru, Vanhoucke, Vincent, and Rabinovich, Andrew. Going deeper with
convolutions. _arXiv preprint arXiv:1409.4842_, 2014.


Turner, RE and Sahani, M. Two problems with variational expectation maximisation for time-series models. _Inference and_
_Estimation in Probabilistic Time-Series Models_, 2011.


Van Erven, Tim and Harremo¨es, Peter. R´enyi divergence
and Kullback-Leibler divergence. _Information Theory, IEEE_
_Transactions on_, 60(7):3797–3820, 2014.


Wan, L, Zeiler, M, Zhang, S, LeCun, Y, and Fergus, R. Regularization of neural networks using dropconnect. In _ICML-13_,
2013.


Yang, Xiao, Kwitt, Roland, and Niethammer, Marc. Fast predictive image registration. _arXiv preprint arXiv:1607.02504_,
2016.


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**


**A. Code Example**


The following is a code snippet showing how our inference can be implemented with a few lines of Keras code (Chollet,
2015). We define a new loss function bbalpha ~~s~~ oftmax ~~c~~ ross ~~e~~ ntropy ~~w~~ ith ~~m~~ c ~~l~~ ogits, that takes MC sampled logits as an input. This is demonstrated for the case of classification. Regression can be implemented in a similar

way.


**def** bbalpha_softmax_cross_entropy_with_mc_logits(alpha):

**def** loss(y_true, mc_logits):

_# mc_logits: output of GenerateMCSamples, of shape M x K x D_
mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
mc_log_softmax = mc_log_softmax - logsumexp(mc_log_softmax, 2)
mc_ll = K.sum(y_true * mc_log_softmax, -1) _# M x K_
**return**   - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc))

**return** loss


MC samples for this loss can be generated using GenerateMCSamples, with layers being a list of Keras initialised
layers:


**def** GenerateMCSamples(inp, layers, K_mc=20):
output_list = []
**for** _ **in** xrange(K_mc):
output_list += [apply_layers(inp, layers)]
**def** pack_out(output_list):
output = K.pack(output_list) _# K_mc x nb_batch x nb_classes_
**return** K.permute_dimensions(output, (1, 0, 2)) _# nb_batch x K_mc x nb_classes_
**def** pack_shape(s):
s = s[0]
**return** (s[0], K_mc, s[1])
out = Lambda(pack_out, output_shape=pack_shape)(output_list)

**return** out


The above two functions rely on the following auxiliary functions:


**def** logsumexp(x, axis=None):
x_max = K.max(x, axis=axis, keepdims=True)
**return** K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max


**def** apply_layers(inp, layers):
output = inp
**for** layer **in** layers:
output = layer(output)
**return** output


**B. Alpha-divergence minimisation**


There are various available definitions of _α_ -divergences, and in this work we mainly used two of them: Amari’s definition
(Amari, 1985) adapted to EP context (Minka, 2005), and R´enyi divergence (R´enyi, 1961) which is more used in information
theory research.


  - Amari’s _α_ -divergence (Amari, 1985):



1
D _α_ [ _p||q_ ] =
_α_ (1 _−_ _α_ )


- R´enyi’s _α_ -divergence (R´enyi, 1961):



1 _−_ _p_ ( _ω_ ) _[α]_ _q_ ( _ω_ ) [1] _[−][α]_ _dω_ _._
� � �



1
R _α_ [ _p||q_ ] = _p_ ( _ω_ ) _[α]_ _q_ ( _ω_ ) [1] _[−][α]_ _dω._
_α −_ 1 [log] �


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**


These two divergence can be converted to each other, e.g. D _α_ [ _p||q_ ] = _α_ (11 _−α_ ) [(1] _[ −]_ [exp [(] _[α][ −]_ [1)R] _[α]_ [[] _[p][||][q]_ []])][. In power]
EP (Minka, 2004), this _α_ -divergence is minimised using projection-based updates. When the approximate posterior _q_
has an exponential family form, minimising D _α_ [ _p||q_ ] requires moment matching to the “tilted distribution” ˜ _p_ _α_ ( _ω_ ) _∝_
_p_ ( _ω_ ) _[α]_ _q_ ( _ω_ ) [1] _[−][α]_ . This projection update might be intractable for non-exponential family _q_ distributions, and instead BB- _α_
deploys a gradient-based update to search a local minimum. We will present the original derivation of the BB- _α_ energy
below and discuss how it relates to power EP.


**C. Original Derivation of BB-** _α_ **Energy**


Here we include the original formulation of the BB- _α_ energy for completeness. Consider approximating a distribution of
the following form



_p_ ( _ω_ ) = [1]

_Z_ _[p]_ [0] [(] _[ω]_ [)]



_p_ ( _ω_ ) = [1]



_N_
� _f_ _n_ ( _ω_ ) _,_


_n_



in which the prior distribution _p_ 0 ( _ω_ ) has an exponential family form _p_ 0 ( _ω_ ) _∝_ exp � _λ_ _[T]_ 0 _[φ]_ [(] _[ω]_ [)] �. Here _λ_ 0 is called natural
parameter or canonical parameter of the exponential family distribution, and _φ_ ( _ω_ ) is the sufficient statistic. As the factors
_f_ _n_ might not be conjugate to the prior, the exact posterior no longer belongs to the same exponential family as the prior,
and hence need approximations. EP construct such approximation by first approximating each complicated factor _f_ _n_ with
a simpler one _f_ [˜] _n_ ( _ω_ ) _∝_ exp � _λ_ _[T]_ _n_ _[φ]_ [(] _[ω]_ [)] �, then constructing the approximate distribution as



 _,_



1
_q_ ( _ω_ ) =
_Z_ ( _λ_ _q_ ) [exp]









_N_
�
� _n_ =0



� _λ_ _n_


_n_ =0



_T_
�



_φ_ ( _ω_ )







with _λ_ _q_ = _λ_ 0 + [�] _[N]_ _n_ =1 _[λ]_ _[n]_ [ and] _[ Z]_ [(] _[λ]_ _[q]_ [)][ the normalising constant/partition function. These] _[ local]_ [ parameters are updated]
using the following procedure (for _α ̸_ = 0):


1 compute cavity distribution _q_ _[\][n]_ ( _ω_ ) _∝_ _q_ ( _ω_ ) _/f_ _n_ ( _ω_ ), equivalently. _λ_ _[\][n]_ _←_ _λ_ _q_ _−_ _λ_ _n_ ;


2 compute the tilted distribution by inserting the likelihood term ˜ _p_ _n_ ( _ω_ ) _∝_ _q_ _[\][n]_ ( _ω_ ) _f_ _n_ ( _ω_ );


3 compute a projection update: _λ_ _q_ _←_ arg min _λ_ D _α_ [˜ _p_ _n_ _||q_ _λ_ ] with _q_ _λ_ an exponential family with natural parameter _λ_ ;


4 recover the site approximation by _λ_ _n_ _←_ _λ_ _q_ _−_ _λ_ _[\][n]_ and form the final update _λ_ _q_ _←_ [�] _n_ _[λ]_ _[n]_ [ +] _[ λ]_ [0] [.]


When converged, the solutions of _λ_ _n_ return a fixed point of the so called _power EP energy_ :



_L_ PEP ( _λ_ 0 _, {λ_ _n_ _}_ ) = log _Z_ ( _λ_ 0 ) + ( _[N]_



_α_




_[N]_

_α_ _[−]_ [1) log] _[ Z]_ [(] _[λ]_ _[q]_ [)] _[ −]_ _α_ [1]



_N_
� log _f_ _n_ ( _ω_ ) _[α]_ exp �( _λ_ _q_ _−_ _αλ_ _n_ ) _[T]_ _φ_ ( _ω_ )� _dω._ (11)

�
_n_ =1



But more importantly, before convergence all these local parameters _λ_ _n_ are maintained in memory. This indicates that
power EP does not scale with big data: consider Gaussian approximations which has _O_ ( _d_ [2] ) parameters with _d_ the dimensionality of _ω_ . Then the space complexity of power EP is _O_ ( _Nd_ [2] ), which is clearly prohibitive for big models like neural
networks that are typically applied to large datasets. BB- _α_ provides a simple solution of this memory overhead by sharing
the local parameters, i.e. defining _λ_ _n_ = _λ_ for all _n_ = 1 _, ..., N_ . Furthermore, under the mild condition that the exponential
family is regular, there exist a one-to-one mapping between _λ_ _q_ and _λ_ (given a fixed _λ_ 0 ). Hence we arrive at a “global”
optimisation problem in the sense that only one parameter _λ_ _q_ is optimised, where the objective function is the BB- _α_ energy



_α_

_._ (12)
� �



_L_ _α_ ( _λ_ 0 _, λ_ _q_ ) = log _Z_ ( _λ_ 0 ) _−_ log _Z_ ( _λ_ _q_ ) _−_ [1]

_α_



_N_
� log E _q_


_n_ =1



_f_ _n_ ( _ω_ )
�� exp [ _λ_ _[T]_ _φ_ ( _ω_ )]



One could verify that this is equivalent to the BB- _α_ energy function presented in the main text by considering exponential
family _q_ distributions.


Although empirical evaluations have demonstrated the superior performance of BB- _α_, the original formulation is difficult
to interpret for practitioners. First the local alpha-divergence minimisation interpretation is inherited from power EP, and


**Dropout Inference in Bayesian Neural Networks with Alpha-divergences**


the intuition of power EP itself might already pose challenges for practitioners. Second, the derivation of BB- _α_ from
power EP is ad hoc and lacks theoretical justification. It has been shown that power EP energy can be viewed as the dual
objective to a continuous version of Bethe free-energy, in which _λ_ _n_ represents the Lagrange multiplier of the constraints
in the primal problem. Hence tying the Lagrange multipliers would effectively changes the primal problem, thus losing
a number of nice guarantees. Nevertheless this approximation has been shown to work well in real-world settings, which
motivated our work to extend BB- _α_ to dropout approximation.


**D. Full Regression Results**


_Table 1._ Regression experiment: Average negative test log likelihood/nats
**Dataset** N D _α_ = 0 _._ 0 _α_ = 0 _._ 5 _α_ = 1 _._ 0 **HMC** **GP** **VI-G**

boston 506 13 2.42 _±_ 0.05 2.38 _±_ 0.06 2.50 _±_ 0.10 2.27 _±_ 0.03 2.22 _±_ 0.07 2.52 _±_ 0.03

concrete 1030 8 2.98 _±_ 0.02 2.88 _±_ 0.02 2.96 _±_ 0.03 2.72 _±_ 0.02 2.85 _±_ 0.02 3.11 _±_ 0.02
energy 768 8 1.75 _±_ 0.01 0.74 _±_ 0.02 0.81 _±_ 0.02 0.93 _±_ 0.01 1.29 _±_ 0.01 0.77 _±_ 0.02
kin8nm 8192 8 -0.83 _±_ 0.00 -1.03 _±_ 0.00 -1.10 _±_ 0.00 -1.35 _±_ 0.00 -1.31 _±_ 0.01 -1.12 _±_ 0.01
power 9568 4 2.79 _±_ 0.01 2.78 _±_ 0.01 2.76 _±_ 0.00 2.70 _±_ 0.00 2.66 _±_ 0.01 2.82 _±_ 0.01
protein 45730 9 2.87 _±_ 0.00 2.87 _±_ 0.00 2.86 _±_ 0.00 2.77 _±_ 0.00 2.95 _±_ 0.05 2.91 _±_ 0.00
red wine 1588 11 0.92 _±_ 0.01 0.92 _±_ 0.01 0.95 _±_ 0.02 0.91 _±_ 0.02 0.67 _±_ 0.01 0.96 _±_ 0.01
yacht 308 6 1.38 _±_ 0.01 1.08 _±_ 0.04 1.15 _±_ 0.06 1.62 _±_ 0.01 1.15 _±_ 0.03 1.77 _±_ 0.01
naval 11934 16 -2.80 _±_ 0.00 -2.80 _±_ 0.00 -2.80 _±_ 0.00 -7.31 _±_ 0.00 -4.86 _±_ 0.04 -6.49 _±_ 0.29
year 515345 90 3.59 _±_ NA 3.54 _±_ NA -3.59 _±_ NA NA _±_ NA 0.65 _±_ NA 3.60 _±_ NA


_Table 2._ Regression experiment: Average test RMSE
**Dataset** N D _α_ = 0 _._ 0 _α_ = 0 _._ 5 _α_ = 1 _._ 0 **HMC** **GP** **VI-G**

boston 506 13 2.85 _±_ 0.19 2.97 _±_ 0.19 3.04 _±_ 0.17 2.76 _±_ 0.20 2.43 _±_ 0.07 2.89 _±_ 0.17

concrete 1030 8 4.92 _±_ 0.13 4.62 _±_ 0.12 4.76 _±_ 0.15 4.12 _±_ 0.14 5.55 _±_ 0.02 5.42 _±_ 0.11
energy 768 8 1.02 _±_ 0.03 1.11 _±_ 0.02 1.10 _±_ 0.02 0.48 _±_ 0.01 1.02 _±_ 0.02 0.51 _±_ 0.01
kin8nm 8192 8 0.09 _±_ 0.00 0.09 _±_ 0.00 0.08 _±_ 0.00 0.06 _±_ 0.00 0.07 _±_ 0.00 0.08 _±_ 0.00
power 9568 4 4.04 _±_ 0.04 4.01 _±_ 0.04 3.98 _±_ 0.04 3.73 _±_ 0.04 3.75 _±_ 0.03 4.07 _±_ 0.04
protein 45730 9 4.28 _±_ 0.02 4.28 _±_ 0.04 4.23 _±_ 0.01 3.91 _±_ 0.02 4.83 _±_ 0.21 4.45 _±_ 0.02
red wine 1588 11 0.61 _±_ 0.01 0.62 _±_ 0.01 0.63 _±_ 0.01 0.63 _±_ 0.01 0.57 _±_ 0.01 0.63 _±_ 0.01
yacht 308 6 0.76 _±_ 0.05 0.85 _±_ 0.06 0.88 _±_ 0.06 0.56 _±_ 0.05 1.15 _±_ 0.09 0.81 _±_ 0.05
naval 11934 16 0.01 _±_ 0.00 0.01 _±_ 0.00 0.01 _±_ 0.00 0.00 _±_ 0.00 0.00 _±_ 0.00 0.00 _±_ 0.00
year 515345 90 8.66 _±_ NA 8.80 _±_ NA 8.97 _±_ NA NA _±_ NA 0.79 _±_ NA 8.88 _±_ NA


