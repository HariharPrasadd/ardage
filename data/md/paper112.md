Published as a conference paper at ICLR 2018

## G RADIENT E STIMATORS FOR I MPLICIT M ODELS


**Yingzhen Li & Richard E. Turner**
University of Cambridge
Cambridge, CB2 1PZ, UK
_{_ yl494,ret26 _}_ @cam.ac.uk


A BSTRACT


Implicit models, which allow for the generation of samples but not for point-wise
evaluation of probabilities, are omnipresent in real-world problems tackled by machine learning and a hot topic of current research. Some examples include data
simulators that are widely used in engineering and scientific research, generative
adversarial networks (GANs) for image synthesis, and hot-off-the-press approximate inference techniques relying on implicit distributions. The majority of existing approaches to learning implicit models rely on approximating the intractable
distribution or optimisation objective for gradient-based optimisation, which is
liable to produce inaccurate updates and thus poor models. This paper alleviates the need for such approximations by proposing the _Stein gradient estimator_,
which directly estimates the score function of the implicitly defined distribution.
The efficacy of the proposed estimator is empirically demonstrated by examples
that include gradient-free MCMC, meta-learning for approximate inference and
entropy regularised GANs that provide improved sample diversity.


1 I NTRODUCTION


Modelling is fundamental to the success of technological innovations for artificial intelligence. A
powerful model learns a useful representation of the observations for a specified prediction task,
and generalises to unknown instances that follow similar generative mechanics. A well established
area of machine learning research focuses on developing _prescribed probabilistic models_ (Diggle
& Gratton, 1984), where learning is based on evaluating the probability of observations under the
model. _Implicit probabilistic models_, on the other hand, are defined by a stochastic procedure that
allows for direct generation of samples, but not for the evaluation of model probabilities. These
are omnipresent in scientific and engineering research involving data analysis, for instance ecology,
climate science and geography, where simulators are used to fit real-world observations to produce
forecasting results. Within the machine learning community there is a recent interest in a specific
type of implicit models, generative adversarial networks (GANs) (Goodfellow et al., 2014), which
has been shown to be one of the most successful approaches to image and text generation (Radford
et al., 2016; Yu et al., 2017; Arjovsky et al., 2017; Berthelot et al., 2017). Very recently, implicit distributions have also been considered as approximate posterior distributions for Bayesian inference,
e.g. see Liu & Feng (2016); Wang & Liu (2016); Li & Liu (2016); Karaletsos (2016); Mescheder
et al. (2017); Husz´ar (2017); Li et al. (2017); Tran et al. (2017). These examples demonstrate the superior flexibility of implicit models, which provide highly expressive means of modelling complex
data structures.


Whilst prescribed probabilistic models can be learned by standard (approximate) maximum likelihood or Bayesian inference, implicit probabilistic models require substantially more severe approximations due to the intractability of the model distribution. Many existing approaches first
approximate the model distribution or optimisation objective function and then use those approximations to learn the associated parameters. However, for any finite number of data points there
exists an infinite number of functions, with arbitrarily diverse gradients, that can approximate perfectly the objective function at the training datapoints, and optimising such approximations can lead
to unstable training and poor results. Recent research on GANs, where the issue is highly prevalent, suggest that restricting the representational power of the discriminator is effective in stabilising
training (e.g. see Arjovsky et al., 2017; Kodali et al., 2017). However, such restrictions often intro

1


Published as a conference paper at ICLR 2018



true loss



true minimum



true minimum



approx. loss approx. loss minima


(a) approximate loss function



true gradient


approx. gradient



(b) approximate gradients



Figure 1: A comparison between the two approximation schemes. Since in practice the optimiser
only visits finite number of locations in the parameter space, it can lead to over-fitting if the neural network based functional approximator is not carefully regularised, and therefore the curvature
information of the approximated loss can be very different from that of the original loss (shown in
(a)). On the other hand, the gradient approximation scheme (b) can be more accurate since it only
involves estimating the sensitivity of the loss function to the parameters in a local region.


duce undesirable biases, responsible for problems such as mode collapse in the context of GANs,
and the underestimation of uncertainty in variational inference methods (Turner & Sahani, 2011).


In this paper we explore approximating the derivative of the log density, known as the score function, as an alternative method for training implicit models. An accurate approximation of the score
function then allows the application of many well-studied algorithms, such as maximum likelihood,
maximum entropy estimation, variational inference and gradient-based MCMC, to implicit models.
Concretely, our contributions include:


_•_ the _Stein gradient estimator_, a novel generalisation of the score matching gradient estimator
(Hyv¨arinen, 2005), that includes both parametric and non-parametric forms;

_•_ a comparison of the proposed estimator with the score matching and the KDE plug-in
estimators on performing gradient-free MCMC, meta-learning of approximate posterior
samplers for Bayesian neural networks, and entropy based regularisation of GANs.


2 L EARNING IMPLICIT PROBABILISTIC MODELS


Given a dataset _D_ containing i.i.d. samples we would like to learn a probabilistic model _p_ ( _**x**_ ) for the
underlying data distribution _p_ _D_ ( _**x**_ ). In the case of implicit models, _p_ ( _**x**_ ) is defined by a generative
process. For example, to generate images, one might define a generative model _p_ ( _**x**_ ) that consists of
sampling randomly a latent variable _**z**_ _∼_ _p_ 0 ( _**z**_ ) and then defining _**x**_ = _**f**_ _**θ**_ ( _**z**_ ). Here _**f**_ is a function
parametrised by _**θ**_, usually a deep neural network or a simulator. We assume _**f**_ to be differentiable
w.r.t. _**θ**_ . An extension to this scenario is presented by _conditional_ implicit models, where the addition
of a supervision signal _**y**_, such as an image label, allows us to define a conditional distribution _p_ ( _**x**_ _|_ _**y**_ )
implicitly by the transformation _**x**_ = _**f**_ _**θ**_ ( _**z**_ _,_ _**y**_ ). A related methodology, _wild variational inference_
(Liu & Feng, 2016; Li & Liu, 2016) assumes a tractable joint density _p_ ( _**x**_ _,_ _**z**_ ), but uses implicit
proposal distributions to approximate an intractable exact posterior _p_ ( _**z**_ _|_ _**x**_ ). Here the approximate
posterior _q_ ( _**z**_ _|_ _**x**_ ) can likewise be represented by a deep neural network, but also by a truncated
Markov chain, such as that given by Langevin dynamics with learnable step-size.


Whilst providing extreme flexibility and expressive power, the intractability of density evaluation also brings serious optimisation issues for implicit models. This is because many
learning algorithms, e.g. maximum likelihood estimation (MLE), rely on minimising a distance/divergence/discrepancy measure D[ _p||p_ _D_ ], which often requires evaluating the model density
(c.f. Ranganath et al., 2016; Liu & Feng, 2016). Thus good approximations to the optimisation
procedure are the key to learning implicit models that can describe complex data structure. In the
context of GANs, the Jensen-Shannon divergence is approximated by a variational lower-bound


2


Published as a conference paper at ICLR 2018


represented by a discriminator (Barber & Agakov, 2003; Goodfellow et al., 2014). Related work
for wild variational inference (Li & Liu, 2016; Mescheder et al., 2017; Husz´ar, 2017; Tran et al.,
2017) uses a GAN-based technique to construct a density ratio estimator for _q/p_ 0 (Sugiyama et al.,
2009; 2012; Uehara et al., 2016; Mohamed & Lakshminarayanan, 2016) and then approximates the
KL-divergence term in the variational lower-bound:


_L_ VI ( _q_ ) = E _q_ [log _p_ ( _**x**_ _|_ _**z**_ )] _−_ KL[ _q_ _**φ**_ ( _**z**_ _|_ _**x**_ ) _||p_ 0 ( _**z**_ )] _._ (1)


In addition, Li & Liu (2016) and Mescheder et al. (2017) exploit the additive structure of the KLdivergence and suggest discriminating between _q_ and an auxiliary distribution that is close to _q_,
making the density ratio estimation more accurate. Nevertheless all these algorithms involve a minimax optimisation, and the current practice of gradient-based optimisation is notoriously unstable.


The stabilisation of GAN training is itself a recent trend of related research (e.g. see Salimans et al.,
2016; Arjovsky et al., 2017). However, as the gradient-based optimisation only interacts with gradients, there is no need to use a discriminator if an accurate approximation to the intractable gradients
could be obtained. As an example, consider a variational inference task with the approximate posterior defined as _**z**_ _∼_ _q_ _**φ**_ ( _**z**_ _|_ _**x**_ ) _⇔_ _**ϵ**_ _∼_ _π_ ( _**ϵ**_ ) _,_ _**z**_ = _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ). Notice that the variational lower-bound
can be rewritten as
_L_ VI ( _q_ ) = E _q_ [log _p_ ( _**x**_ _,_ _**z**_ )] + H[ _q_ _**φ**_ ( _**z**_ _|_ _**x**_ )] _,_ (2)


the gradient of the variational parameters _**φ**_ can be computed by a sum of the path gradient of
the first term (i.e. E _π_ � _∇_ _**f**_ log _p_ ( _**x**_ _,_ _**f**_ ( _**ϵ**_ _,_ _**x**_ )) [T] _∇_ _**φ**_ _**f**_ ( _**ϵ**_ _,_ _**x**_ )�) and the gradient of the entropy term
_∇_ _**φ**_ H[ _q_ ( _**z**_ _|_ _**x**_ )]. Expanding the latter, we have


_∇_ _**φ**_ H[ _q_ _**φ**_ ( _**z**_ _|_ _**x**_ )] = _−∇_ _**φ**_ E _π_ ( _**ϵ**_ ) [log _q_ _**φ**_ ( _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ))]



= _−_ E _π_ ( _**ϵ**_ ) [ _∇_ _**φ**_ log _q_ _**φ**_ ( _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ))]

= _−_ E _π_ ( _**ϵ**_ ) [ _∇_ _**φ**_ log _q_ _**φ**_ ( _**z**_ _|_ _**x**_ ) _|_ _**z**_ = _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ) + _∇_ _**f**_ log _q_ _**φ**_ ( _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ) _|_ _**x**_ ) _∇_ _**φ**_ _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ )]

= _−_ E _q_ _**φ**_ ( _**z**_ _|_ _**x**_ ) [ _∇_ _**φ**_ log _q_ _**φ**_ ( _**z**_ _|_ _**x**_ )] _−_ E _π_ ( _**ϵ**_ ) [ _∇_ _**f**_ log _q_ _**φ**_ ( _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ ) _|_ _**x**_ ) _∇_ _**φ**_ _**f**_ _**φ**_ ( _**ϵ**_ _,_ _**x**_ )] _,_



(3)



in which the first term in the last line is zero (Roeder et al., 2017). As we typically assume the
tractability of _∇_ _**φ**_ _**f**_, an accurate approximation to _∇_ _**z**_ log _q_ ( _**z**_ _|_ _**x**_ ) would remove the requirement of
discriminators, speed-up the learning and obtain potentially a better model. Many gradient approximation techniques exist (Stone, 1985; Fan & Gijbels, 1996; Zhou & Wolfe, 2000; De Brabanter
et al., 2013), and in particular, in the next section we will review kernel-based methods such as
kernel density estimation (Singh, 1977) and score matching (Hyv¨arinen, 2005) in more detail, and
motivate the main contribution of the paper.


3 G RADIENT APPROXIMATION WITH THE S TEIN GRADIENT ESTIMATOR


We propose the _Stein gradient estimator_ as a novel generalisation of the score matching gradient estimator. Before presenting it we first set-up the notation. Column vectors and matrices are boldfaced.
The random variable under consideration is _**x**_ _∈X_ with _X_ = R _[d][×]_ [1] if not specifically mentioned. To
avoid misleading notation we use the distribution _q_ ( _**x**_ ) to derive the gradient approximations for general cases. As Monte Carlo methods are heavily used for implicit models, in the rest of the paper we
mainly consider approximating the gradient _**g**_ ( _**x**_ _[k]_ ) := _∇_ _**x**_ _k_ log _q_ ( _**x**_ _[k]_ ) for _**x**_ _[k]_ _∼_ _q_ ( _**x**_ ) _, k_ = 1 _, ..., K_ .
We use _x_ _[i]_ _j_ [to denote the] _[ j]_ [th element of the] _[ i]_ [th sample] _**[ x]**_ _[i]_ [. We also denote the matrix form of the col-]

T _K×d_
lected gradients as **G** := � _∇_ _**x**_ 1 log _q_ ( _**x**_ [1] ) _, · · ·, ∇_ _**x**_ _K_ log _q_ ( _**x**_ _[K]_ )� _∈_ R, and its approximation
ˆ ˆ T _k_ _k_
**G** := � _g_ ( _**x**_ [1] ) _, · · ·,_ ˆ _g_ ( _**x**_ _[K]_ )� with ˆ _g_ ( _**x**_ ) = _∇_ _**x**_ _k_ log ˆ _q_ ( _**x**_ ) for some ˆ _q_ ( _**x**_ ).


3.1 S TEIN GRADIENT ESTIMATOR : INVERTING S TEIN ’ S IDENTITY


We start from introducing _Stein’s identity_ that was first developed for Gaussian random variables
(Stein, 1972; 1981) then extended to general cases (Gorham & Mackey, 2015; Liu et al., 2016). Let
_**h**_ : R _[d][×]_ [1] _→_ R _[d]_ _[′]_ _[×]_ [1] be a differentiable multivariate test function which maps _**x**_ to a column vector
_**h**_ ( _**x**_ ) = [ _h_ 1 ( _**x**_ ) _, h_ 2 ( _**x**_ ) _, ..., h_ _d_ _′_ ( _**x**_ )] [T] . We further assume the _boundary condition_ for _**h**_ :


_q_ ( _**x**_ ) _**h**_ ( _**x**_ ) _|_ _∂X_ = **0** _,_ or lim (4)
_**x**_ _→_ _**∞**_ _[q]_ [(] _**[x]**_ [)] _**[h]**_ [(] _**[x]**_ [) = 0][ if] _[ X]_ [ =][ R] _[d]_ _[.]_


3


Published as a conference paper at ICLR 2018


This condition holds for almost any test function if _q_ has sufficiently fast-decaying tails (e.g. Gaussian tails). Now we introduce Stein’s identity (Stein, 1981; Gorham & Mackey, 2015; Liu et al.,
2016)
E _q_ [ _**h**_ ( _**x**_ ) _∇_ _**x**_ log _q_ ( _**x**_ ) [T] + _∇_ _**x**_ _**h**_ ( _**x**_ )] = **0** _,_ (5)

in which the gradient matrix term _∇_ _**x**_ _**h**_ ( _**x**_ ) = ( _∇_ _**x**_ _h_ 1 ( _**x**_ ) _, · · ·, ∇_ _**x**_ _h_ _d_ _′_ ( _**x**_ )) [T] _∈_ R _[d]_ _[′]_ _[×][d]_ _._ This identity
can be proved using _integration by parts_ : for the _i_ th row of the matrix _**h**_ ( _**x**_ ) _∇_ _**x**_ log _q_ ( _**x**_ ) [T], we have


E _q_ [ _h_ _i_ ( _**x**_ ) _∇_ _**x**_ log _q_ ( _**x**_ ) [T] ] = _h_ _i_ ( _**x**_ ) _∇_ _**x**_ _q_ ( _**x**_ ) [T] _d_ _**x**_
�



= _q_ ( _**x**_ ) _h_ _i_ ( _**x**_ ) _|_ _∂X_ _−_ _q_ ( _**x**_ ) _∇_ _**x**_ _h_ _i_ ( _**x**_ ) [T] _d_ _**x**_
�

= _−_ E _q_ [ _∇_ _**x**_ _h_ _i_ ( _**x**_ ) [T] ] _._



(6)



Observing that the gradient term _∇_ _**x**_ log _q_ ( _**x**_ ) of interest appears in Stein’s identity (5), we propose
the _Stein gradient estimator_ by inverting Stein’s identity. As the expectation in (5) is intractable, we
further approximate the above with Monte Carlo (MC):



_K_



1

_K_



_K_
�



� _−_ _**h**_ ( _**x**_ _[k]_ ) _∇_ _**x**_ _k_ log _q_ ( _**x**_ _[k]_ ) [T] + err = _K_ [1]

_k_ =1



_K_
� _∇_ _**x**_ _k_ _**h**_ ( _**x**_ _[k]_ ) _,_ _**x**_ _[k]_ _∼_ _q_ ( _**x**_ _[k]_ ) _,_ (7)


_k_ =1



with err _∈_ R _[d]_ _[′]_ _[×][d]_ the random error due to MC approximation, which has mean **0** and vanishes
as _K →_ + _∞_ . Now by temporarily denoting **H** = � _**h**_ ( _**x**_ [1] ) _, · · ·,_ _**h**_ ( _**x**_ _[K]_ )� _∈_ R _[d]_ _[′]_ _[×][K]_ _,_ _∇_ _**x**_ _**h**_ =
_K_ 1 � _Kk_ =1 _[∇]_ _**[x]**_ _[k]_ _**[h]**_ [(] _**[x]**_ _[k]_ [)] _[ ∈]_ [R] _[d]_ _[′]_ _[×][d]_ _[,]_ [ equation (7) can be rewritten as] _[ −]_ _K_ [1] **[HG]** [ +][ err][ =] _[ ∇]_ _**[x]**_ _**[h]**_ _[.]_ [ Thus we]

consider a ridge regression method (i.e. adding an _ℓ_ 2 regulariser) to estimate **G** :



ˆ
**G** [Stein] _V_ := arg min **G** ˆ _∈_ R _K×d_ _||∇_ _**x**_ _**h**_ + _K_ [1]



_F_ [+] _[η]_
_K_ **[H][G]** [ ˆ] _[||]_ [2] _K_



_K_ [2] _[||]_ **[G]** [ ˆ] _[||]_ _F_ [2] _[,]_ (8)



with _|| · ||_ _F_ the Frobenius norm of a matrix and _η ≥_ 0. Simple calculation shows that

ˆ
**G** [Stein] _V_ = _−_ ( **K** + _η_ _**I**_ ) _[−]_ [1] _⟨∇,_ **K** _⟩,_ (9)

where **K** := **H** [T] **H** _,_ **K** _ij_ = _K_ ( _**x**_ _[i]_ _,_ _**x**_ _[j]_ ) := _**h**_ ( _**x**_ _[i]_ ) [T] _**h**_ ( _**x**_ _[j]_ ) _, ⟨∇,_ **K** _⟩_ := _K_ **H** [T] _∇_ _**x**_ _**h**_ _,_ _⟨∇,_ **K** _⟩_ _ij_ =
� _Kk_ =1 _[∇]_ _x_ _[k]_ _j_ _[K]_ [(] _**[x]**_ _[i]_ _[,]_ _**[ x]**_ _[k]_ [)] _[.]_ [ One can show that the RBF kernel satisfies Stein’s identity (Liu et al., 2016).]
In this case _**h**_ ( _**x**_ ) = _K_ ( _**x**_ _, ·_ ) _, d_ _[′]_ = + _∞_ and by the reproducing kernel property (Berlinet & ThomasAgnan, 2011), _**h**_ ( _**x**_ ) [T] _**h**_ ( _**x**_ _[′]_ ) = _⟨K_ ( _**x**_ _, ·_ ) _, K_ ( _**x**_ _[′]_ _, ·_ ) _⟩_ _H_ = _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) _._


3.2 S TEIN GRADIENT ESTIMATOR MINIMISES THE KERNELISED S TEIN DISCREPANCY


In this section we derive the Stein gradient estimator again, but from a divergence/discrepancy minimisation perspective. Stein’s method also provides a tool for checking if two distributions _q_ ( _**x**_ )
and ˆ _q_ ( _**x**_ ) are identical. If the test function set _H_ is sufficiently rich, then one can define a Stein
discrepancy measure by
_S_ ( _q,_ ˆ _q_ ) := sup E _q_ � _∇_ _**x**_ log ˆ _q_ ( _**x**_ ) [T] _**h**_ ( _**x**_ ) + _⟨∇,_ _**h**_ _⟩_ � _,_ (10)
_**h**_ _∈H_


see Gorham & Mackey (2015) for an example derivation. When _H_ is defined as a unit ball in an
RKHS induced by a kernel _K_ ( _**x**_ _, ·_ ), Liu et al. (2016) and Chwialkowski et al. (2016) showed that
the supremum in (10) can be analytically obtained as (with _K_ _**xx**_ _′_ shorthand for _K_ ( _**x**_ _,_ _**x**_ _[′]_ )):

_S_ [2] ( _q,_ ˆ _q_ ) = E _**x**_ _,_ _**x**_ _′_ _∼q_ �(ˆ _**g**_ ( _**x**_ ) _−_ _**g**_ ( _**x**_ )) [T] _K_ _**xx**_ _′_ (ˆ _**g**_ ( _**x**_ _[′]_ ) _−_ _**g**_ ( _**x**_ _[′]_ ))� _,_ (11)
which is also named the _kernelised Stein discrepancy_ (KSD). Chwialkowski et al. (2016) showed that
for _C_ 0 -universal kernels satisfying the boundary condition, KSD is indeed a discrepancy measure:
_S_ [2] ( _q,_ ˆ _q_ ) = 0 _⇔_ _q_ = ˆ _q_ . Gorham & Mackey (2017) further characterised the power of KSD on
detecting non-convergence cases. Furthermore, if the kernel is twice differentiable, then using the
same technique as to derive (16) one can compute KSD by

ˆ T _′_ T T
_S_ [2] ( _q,_ ˆ _q_ ) = E _**x**_ _,_ _**x**_ _′_ _∼q_ � _**g**_ ( _**x**_ ) _K_ _**xx**_ _′_ ˆ _**g**_ ( _**x**_ ) + ˆ _**g**_ ( _**x**_ ) _∇_ _**x**_ _′_ _K_ _**xx**_ _′_ + _∇_ _**x**_ _K_ _**xx**_ _[′]_ [ ˆ] _**[g]**_ [(] _**[x]**_ _[′]_ [) +][ Tr][(] _[∇]_ _**[x]**_ _[,]_ _**[x]**_ _[′]_ _[K]_ _**[xx]**_ _[′]_ [)] � _._
(12)
In practice KSD is estimated with samples _{_ _**x**_ _[k]_ _}_ _[K]_ _k_ =1 _[∼]_ _[q]_ [, and simple derivations show that the V-]
statistic of KSD can be reformulated as _S_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] _K_ 1 [2] [Tr][( ˆ] **[G]** [T] **[K]** [ ˆ] **[G]** [ + 2 ˆ] **[G]** [T] _[⟨∇][,]_ **[ K]** _[⟩]_ [) +] _[ C]_ [. Thus the]
_l_ 2 error in (8) is equivalent to the V-statistic of KSD if _**h**_ ( _**x**_ ) = _K_ ( _**x**_ _, ·_ ), and we have the following:


4


Published as a conference paper at ICLR 2018


**Theorem 1.** **G** [ˆ] [Stein] _V_ _is the solution of the following KSD V-statistic minimisation problem_

**G** ˆ [Stein] _V_ = arg min **G** ˆ _∈_ R _K×d_ _S_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) +] _K_ _[η]_ [2] _[||]_ **[G]** [ ˆ] _[||]_ _F_ [2] _[.]_ (13)


One can also minimise the U-statistic of KSD to obtain gradient approximations, and a full derivation
of which, including the optimal solution, can be found in the appendix. In experiments we use Vstatistic solutions and leave comparisons between these methods to future work.


3.3 C OMPARISONS TO EXISTING KERNEL - BASED GRADIENT ESTIMATORS


There exist other gradient estimators that do not require explicit evaluations of _∇_ _**x**_ log _q_ ( _**x**_ ), e.g. the
denoising auto-encoder (DAE) (Vincent et al., 2008; Vincent, 2011; Alain & Bengio, 2014) which,
with infinitesimal noise, also provides an estimate of _∇_ _**x**_ log _q_ ( _**x**_ ) at convergence. However, applying such gradient estimators result in a double-loop optimisation procedure since the gradient
approximation is repeatedly required for fitting implicit distributions, which can be significantly
slower than the proposed approach. Therefore we focus on “quick and dirty” approximations and
only include comparisons to kernel-based gradient estimators in the following.


3.3.1 KDE GRADIENT ESTIMATOR : PLUG - IN ESTIMATOR WITH DENSITY ESTIMATION


A naive approach for gradient approximation would first estimate the intractable density ˆ _q_ ( _**x**_ ) _≈_
_q_ ( _**x**_ ) (up to a constant), then approximate the exact gradient by _∇_ _**x**_ log ˆ _q_ ( _**x**_ ) _≈∇_ _**x**_ log _q_ ( _**x**_ ). Specifically, Singh (1977) considered kernel density estimation (KDE) ˆ _q_ ( _**x**_ ) = _K_ 1 � _Kk_ =1 _[K]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[k]_ [)] _[ ×][ C.]_ [,]
then differentiated through the KDE estimate to obtain the gradient estimator:



_K_
� _K_ ( _**x**_ _[i]_ _,_ _**x**_ _[k]_ ) _._ (14)


_k_ =1



ˆ
**G** [KDE] =
_ij_



_K_
� _∇_ _x_ _ij_ _[K]_ [(] _**[x]**_ _[i]_ _[,]_ _**[ x]**_ _[k]_ [)] _[/]_

_k_ =1



Interestingly for translation invariant kernels _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _K_ ( _**x**_ _−_ _**x**_ _[′]_ ) the _KDE gradient estimator_
(14) can be rewritten as **G** [ˆ] [KDE] = _−_ diag ( **K1** ) _[−]_ [1] _⟨∇,_ **K** _⟩._ Inspecting and comparing it with the
Stein gradient estimator (9), one might notice that the Stein method uses the full kernel matrix
as the pre-conditioner, while the KDE method computes an averaged “kernel similarity” for the
denominator. We conjecture that this difference is key to the superior performance of the Stein
gradient estimator when compared to the KDE gradient estimator (see later experiments). The KDE
method only collects the similarity information between _**x**_ _[k]_ and other samples _**x**_ _[j]_ to form an estimate
of _∇_ _**x**_ _k_ log _q_ ( _**x**_ _[k]_ ), whereas for the Stein gradient estimator, the kernel similarity between _**x**_ _[i]_ and _**x**_ _[j]_
for all _i, j ̸_ = _k_ are also incorporated. Thus it is reasonable to conjecture that the Stein method can
be more sample efficient, which also implies higher accuracy when the same number of samples are
collected.


3.3.2 S CORE MATCHING GRADIENT ESTIMATOR : MINIMISING MSE


The KDE gradient estimator performs indirect approximation of the gradient via density estimation,
which can be inaccurate. An alternative approach directly approximates the gradient _∇_ _**x**_ log _q_ ( _**x**_ )
by minimising the expected _ℓ_ 2 error w.r.t. the approximation ˆ _**g**_ ( _**x**_ ) = (ˆ _g_ 1 ( _**x**_ ) _, · · ·,_ ˆ _g_ _d_ ( _**x**_ )) [T] :


_F_ (ˆ _**g**_ ) := E _q_ � _||_ _**g**_ ˆ( _**x**_ ) _−∇_ _**x**_ log _q_ ( _**x**_ ) _||_ 2 [2] � _._ (15)


It has been shown in Hyv¨arinen (2005) that this objective can be reformulated as



ˆ
_F_ (ˆ _**g**_ ) = E _q_ � _||_ _**g**_ ( _**x**_ ) _||_ 2 [2] [+ 2] _[⟨∇][,]_ [ ˆ] _**[g]**_ [(] _**[x]**_ [)] _[⟩]_ � + _C,_ _⟨∇,_ ˆ _**g**_ ( _**x**_ ) _⟩_ =



_d_
� _∇_ _x_ _j_ ˆ _g_ _j_ ( _**x**_ ) _._ (16)

_j_ =1



The key insight here is again the usage of integration by parts: after expanding the _ℓ_ 2 loss objective,
the cross term can be rewritten as E _q_ � _**g**_ ˆ( _**x**_ ) T _∇_ _**x**_ log _q_ ( _**x**_ )� = _−_ E _q_ [ _⟨∇,_ ˆ _**g**_ ( _**x**_ ) _⟩_ ] _,_ if assuming the
boundary condition (4) for ˆ _**g**_ (see (6)). The optimum of (16) is referred as the _score matching_
_gradient estimator_ . The _ℓ_ 2 objective (15) is also called _Fisher divergence_ (Johnson, 2004) which is
a special case of KSD (11) by selecting _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _δ_ _**x**_ = _**x**_ _′_ . Thus the Stein gradient estimator can be
viewed as a generalisation of the score matching estimator.


5


Published as a conference paper at ICLR 2018


The comparison between the two estimators is more complicated. Certainly by the Cauchy-Schwarz
inequality the Fisher divergence is stronger than KSD in terms of detecting convergence (Liu et al.,
2016). However it is difficult to perform direct gradient estimation by minimising the Fisher divergence, since (i) the Dirac kernel is non-differentiable so that it is impossible to rewrite the divergence
in a similar form to (12), and (ii) the transformation to (16) involves computing _∇_ _**x**_ _**g**_ ˆ( _**x**_ ). So one
needs to propose a _parametric_ approximation to **G** and then optimise the associated parameters accordingly, and indeed Sasaki et al. (2014) and Strathmann et al. (2015) derived a parametric solution
by first approximating the log density up to a constant as log ˆ _q_ ( _**x**_ ) := [�] _[K]_ _k_ =1 _[a]_ _[k]_ _[K]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[k]_ [) +] _[ C]_ [, then]
minimising (16) to obtain the coefficients ˆ _a_ [score] _k_ and constructing the gradient estimator as



ˆ
**G** [score] _i·_ =



_K_


ˆ

� _a_ [score] _k_ _∇_ _**x**_ _i_ _K_ ( _**x**_ _[i]_ _,_ _**x**_ _[k]_ ) _._ (17)


_k_ =1



Therefore the usage of parametric estimation can potentially remove the advantage of using a
stronger divergence. Conversely, the proposed Stein gradient estimator (9) is _non-parametric_ in
that it directly optimises over functions evaluated at locations _{_ _**x**_ _k_ _}_ _[K]_ _k_ =1 [. This brings in two key ad-]
vantages over the score matching gradient estimator: (i) it removes the _approximation error_ due to
the use of restricted family of parametric approximations and thus can be potentially more accurate;
(ii) it has a much simpler and _ubiquitous_ form that applies to _any kernel satisfying the boundary_
_condition_, whereas the score matching estimator requires tedious derivations for different kernels
repeatedly (see appendix).


In terms of computation speed, since in most of the cases the computation of the score matching
gradient estimator also involves kernel matrix inversions, both estimators are of the same order of
complexity, which is _O_ ( _K_ [3] + _K_ [2] _d_ ) (kernel matrix computation plus inversion). Low-rank approximations such as the Nystr¨om method (Smola & Sch¨okopf, 2000; Williams & Seeger, 2001) can
enable speed-up, but this is not investigated in the paper. Again we note here that kernel-based gradient estimators can still be faster than e.g. the DAE estimator since no double-loop optimisation is
required. Certainly it is possible to apply early-stopping for the inner-loop DAE fitting. However
the resulting gradient approximation might be very poor, which leads to unstable training and poorly
fitted implicit distributions.


3.4 A DDING PREDICTIVE POWER


Though providing potentially more accurate approximations, the non-parametric estimator (9) has
no predictive power as described so far. Crucially, many tasks in machine learning require predicting
gradient functions at samples drawn from distributions other than _q_, for example, in MLE _q_ ( _**x**_ )
corresponds to the model distribution which is learned using samples from the data distribution
instead. To address this issue, we derive two _predictive_ estimators, one generalised from the nonparametric estimator and the other minimises KSD using parametric approximations.



**Predictions using the non-parametric estimator.** Let us consider an unseen datum _**y**_ . If _**y**_ is sampled from _q_, then one can also apply the non-parametric estimator (9) for gradient approximation,
given the observed data **X** = _{_ _**x**_ [1] _, ...,_ _**x**_ _[K]_ _} ∼_ _q_ . Concretely, if writing ˆ _**g**_ ( _**y**_ ) _≈∇_ _**y**_ log _q_ ( _**y**_ ) _∈_ R _[d][×]_ [1]
then the non-parametric Stein gradient estimator computed on **X** _∪{_ _**y**_ _}_ is
� _**g**_ ˆ( **G** _**y**_ ˆ ) T � = _−_ ( **K** _[∗]_ + _η_ _**I**_ ) _[−]_ [1] � _∇_ _**y**_ _K_ ( _**y**_ _⟨∇,_ _**y**_ _,_ ) + **K** _⟩_ [�] + _∇_ _[K]_ _k_ =1 _**y**_ _K_ _[∇]_ ( _**[x]**_ _·,_ _[k]_ _**y**_ _[K]_ ) [(] _**[y]**_ _[,]_ _**[ x]**_ _[k]_ [)] � _,_ **K** _[∗]_ = � **KK** **X** _**yyy**_ **KK** _**y**_ **X** � _,_

with _∇_ _**y**_ _K_ ( _·,_ _**y**_ ) denoting a _K × d_ matrix with rows _∇_ _**y**_ _K_ ( _**x**_ _[k]_ _,_ _**y**_ ), and _∇_ _**y**_ _K_ ( _**y**_ _,_ _**y**_ ) only differentiates through the second argument. Then we demonstrate in the appendix that, by simple matrix
calculations and assuming a translation invariant kernel, we have (with column vector **1** _∈_ R _[K][×]_ [1] ):


_−_ 1
_∇_ _**y**_ log _q_ ( _**y**_ ) [T] _≈−_ � **K** _**yy**_ + _η −_ **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] **K** **X** _**y**_ �
(18)
� **K** _**y**_ **X** **G** [ˆ] [Stein] _V_ _−_ � **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] + **1** [T] [�] _∇_ _**y**_ _K_ ( _·,_ _**y**_ )� _._

In practice one would store the computed gradient **G** [ˆ] [Stein] _V_, the kernel matrix inverse ( **K** + _η_ **I** ) _[−]_ [1] and
_η_ as the “parameters” of the predictive estimator. For a new observation _**y**_ _∼_ _p_ in general, one can
“pretend” _**y**_ is a sample from _q_ and apply the above estimator as well. The approximation quality
depends on the similarity between _q_ and _p_, and we conjecture here that this similarity measure, if
can be described, is closely related to the KSD.


6



� = _−_ ( **K** _[∗]_ + _η_ _**I**_ ) _[−]_ [1] � _∇_ _**y**_ _K_ ( _**y**_ _⟨∇,_ _**y**_ _,_ ) + **K** _⟩_ [�] + _∇_ _[K]_ _k_ =1 _**y**_ _K_ _[∇]_ ( _**[x]**_ _·,_ _[k]_ _**y**_ _[K]_ ) [(] _**[y]**_ _[,]_ _**[ x]**_ _[k]_ [)]



� _,_ **K** _[∗]_ = � **KK** **X** _**yyy**_ **KK** _**y**_ **X**



_,_
�


Published as a conference paper at ICLR 2018


**Fitting a parametric estimator using KSD.** The non-parametric predictive estimator could be
computationally demanding. Setting aside the cost of fitting the “parameters”, in prediction the
time complexity for the non-parametric estimator is _O_ ( _K_ [2] + _Kd_ ). Also storing the “parameters”
needs _O_ ( _Kd_ ) memory for **G** [ˆ] [Stein] _V_ . These costs make the non-parametric estimator undesirable for
high-dimensional data, since in order to obtain accurate predictions it often requires _K_ scaling with
_d_ as well. To address this, one can also minimise the KSD using parametric approximations, in a
similar way as to derive the score matching estimator in Section 3.3.2. More precisely, we define
a parametric approximation in a similar fashion as (17), and in the appendix we show that if the
RBF kernel is used for both the KSD and the parametric approximation, then the linear coefficients
_**a**_ = ( _a_ 1 _, ..., a_ _K_ ) [T] can be calculated analytically: ˆ _**a**_ [Stein] _V_ = ( **Λ** + _η_ **I** ) _[−]_ [1] _**b**_, where


**Λ** =X _⊙_ ( **KKK** ) + **K** ( **K** _⊙_ X) **K** _−_ (( **KK** ) _⊙_ X) **K** _−_ **K** (( **KK** ) _⊙_ X) _,_
(19)
_**b**_ =( **K** diag(X) **K** + ( **KK** ) _⊙_ X _−_ **K** ( **K** _⊙_ X) _−_ ( **K** _⊙_ X) **K** ) **1** _,_


with X the “gram matrix” that has elements X _ij_ = ( _**x**_ _[i]_ ) [T] _**x**_ _[j]_ . Then for an unseen observation _**y**_ _∼_
_p_ the gradient approximation returns _∇_ _**y**_ log _q_ ( _**y**_ ) _≈_ (ˆ _**a**_ [Stein] _V_ ) [T] _∇_ _**y**_ _K_ ( _·,_ _**y**_ ). In this case one only
maintains the linear coefficients ˆ _**a**_ [Stein] _V_ and computes a linear combination in prediction, which takes
_O_ ( _K_ ) memory and _O_ ( _Kd_ ) time and therefore is computationally cheaper than the non-parametric
prediction model (27).


4 A PPLICATIONS


We present some case studies that apply the gradient estimators to implicit models. Detailed settings (architecture, learning rate, etc.) are presented in the appendix. Implementation is released at
[https://github.com/YingzhenLi/SteinGrad.](https://github.com/YingzhenLi/SteinGrad)


4.1 S YNTHETIC EXAMPLE : H AMILTONIAN FLOW WITH APPROXIMATE GRADIENTS


We first consider a simple synthetic example to demonstrate the accuracy of the proposed gradient
estimator. More precisely we consider the kernel induced Hamiltonian flow ( _not_ an exact sampler)
(Strathmann et al., 2015) on a 2-dimensional banana-shaped object: _**x**_ _∼B_ ( _**x**_ ; _b_ = 0 _._ 03 _, v_ =
100) _⇔_ _x_ 1 _∼N_ ( _x_ 1 ; 0 _, v_ ) _, x_ 2 = _ϵ_ + _b_ ( _x_ [2] 1 _[−]_ _[v]_ [)] _[, ϵ][ ∼N]_ [(] _[ϵ]_ [; 0] _[,]_ [ 1)][. The approximate Hamiltonian flow]
is constructed using the same operator as in Hamiltonian Monte Carlo (HMC) (Neal et al., 2011),
except that the exact score function _∇_ _**x**_ log _B_ ( _**x**_ ) is replaced by the approximate gradients. We still
use the exact target density to compute the rejection step as we mainly focus on testing the accuracy
of the gradient estimators. We test both versions of the predictive Stein gradient estimator (see
section 3.4) since we require the particles of parallel chains to be independent with each other. We
fit the gradient estimators on _K_ = 200 training datapoints from the target density. The bandwidth
of the RBF kernel is computed by the median heuristic and scaled up by a scalar between [1 _,_ 5].
All three methods are simulated for _T_ = 2 _,_ 000 iterations, share the same initial locations that are
constructed by target distribution samples plus Gaussian noises of standard deviation 2.0, and the
results are averaged over 200 parallel chains.


We visualise the samples and some MCMC statistics in Figure 2. In general all the resulting Hamiltonian flows are HMC-like, which give us the confidence that the gradient estimators extrapolate
reasonably well at unseen locations. However all of these methods have trouble exploring the extremes, because at those locations there are very few or even no training data-points. Indeed we
found it necessary to use large (but not too large) bandwidths, in order to both allow exploration
of those extremes, and ensure that the corresponding test function is not too smooth. In terms of
quantitative metrics, the acceptance rates are reasonably high for all the gradient estimators, and the
KSD estimates (across chains) as a measure of sample quality are also close to that computed on
HMC samples. The returned estimates of E[ _x_ 1 ] are close to zero which is the ground true value. We
found that the non-parametric Stein gradient estimator is more sensitive to hyper-parameters of the
dynamics, e.g. the stepsize of each HMC step. We believe a careful selection of the kernel (e.g. those
with long tails) and a better search for the hyper-parameters (for both the kernel and the dynamics)
can further improve the sample quality and the chain mixing time, but this is not investigated here.


7


Published as a conference paper at ICLR 2018


Figure 2: Kernel induced Hamiltonian flow compared with HMC. Top: samples generated from the
dynamics, training data (in cyan), an the trajectory of a particle for _T_ = 1 to 200 starting at the star
location (in yellow). Bottom: statistics computed during simulations. See main text for details.


4.2 M ETA - LEARNING OF APPROXIMATE POSTERIOR SAMPLERS FOR B AYESIAN NN S


One of the recent focuses on meta-learning has been on learning optimisers for training deep neural
networks, e.g. see (Andrychowicz et al., 2016). Could analogous goals be achieved for approximate
inference? In this section we attempt to learn an approximate posterior sampler for Bayesian neural
networks (Bayesian NNs, BNNs) that generalises to _unseen_ datasets and architectures. A more
detailed introduction of Bayesian neural networks is included in the appendix, and in a nutshell,
we consider a binary classification task: _p_ ( _y_ = 1 _|_ _**x**_ _,_ _**θ**_ ) = sigmoid(NN _**θ**_ ( _**x**_ )), _p_ 0 ( _**θ**_ ) = _N_ ( _**θ**_ ; **0** _,_ **I** ).
After observing the training data _D_ = _{_ ( _**x**_ _n_ _, y_ _n_ ) _}_ _[N]_ _n_ =1 [, we first obtain the approximate posterior]
_q_ _**φ**_ ( _**θ**_ ) _≈_ _p_ ( _**θ**_ _|D_ ) _∝_ _p_ 0 ( _**θ**_ ) [�] _[N]_ _n_ =1 _[p]_ [(] _[y]_ _[n]_ _[|]_ _**[x]**_ _[n]_ _[,]_ _**[ θ]**_ [)][, then approximate the predictive distribution for a]
new observation as _p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _, D_ ) _≈_ _K_ 1 � _Kk_ =1 _[p]_ [(] _[y]_ _[∗]_ [= 1] _[|]_ _**[x]**_ _[∗]_ _[,]_ _**[ θ]**_ _[k]_ [)] _[,]_ _**[ θ]**_ _[k]_ _[ ∼]_ _[q]_ _**[φ]**_ [(] _**[θ]**_ [)] _[.]_ [ In this task we]
define an implicit approximate posterior distribution _q_ _**φ**_ ( _**θ**_ ) as the following _stochastic_ normalising
flow (Rezende & Mohamed, 2015) _**θ**_ _t_ +1 = _**f**_ ( _**θ**_ _t_ _, ∇_ _t_ _,_ _**ϵ**_ _t_ ): given the current location _**θ**_ _t_ and the
mini-batch data _{_ ( _**x**_ _m_ _, y_ _m_ ) _}_ _[M]_ _m_ =1 [, the update for the next step is]


_**θ**_ _t_ +1 = _**θ**_ _t_ + _ζ_ ∆ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ ) + _**σ**_ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ ) _⊙_ _**ϵ**_ _t_ _,_ _**ϵ**_ _t_ _∼N_ ( _**ϵ**_ ; **0** _,_ **I** ) _,_



�



(20)

_._



_∇_ _t_ = _∇_ _**θ**_ _t_



_N_

_M_
�



_M_
�



� log _p_ ( _y_ _m_ _|_ _**x**_ _m_ _,_ _**θ**_ _t_ ) + log _p_ 0 ( _**θ**_ _t_ )


_m_ =1



The coordinates of the noise standard deviation _**σ**_ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ ) and the moving direction ∆ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ )
are parametrised by a _coordinate-wise_ neural network. If properly trained, this neural network will
learn the best combination of the current location and gradient information, and produce approximate posterior samples efficiently on different probabilistic modelling tasks. Here we propose using
the variational inference objective (2) computed on the samples _{_ _**θ**_ _t_ _[k]_ _[}]_ [ to learn the variational param-]
eters _**φ**_ . Since in this case the gradient of the log joint distribution can be computed analytically,
we only approximate the gradient of the entropy term H[ _q_ ] as in (3), with the exact score function replaced by the presented gradient estimators. We report the results using the non-parametric
Stein gradient estimator as we found it works better than the parametric version. The RBF kernel
is applied for gradient estimation, with the hyper-parameters determined by a grid search on the
bandwidth _σ_ [2] _∈{_ 0 _._ 25 _,_ 1 _._ 0 _,_ 4 _._ 0 _,_ 10 _._ 0 _,_ median trick _}_ and _η ∈{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _._ 0 _,_ 2 _._ 0 _}_ .


We briefly describe the test protocol. We take from the UCI repository (Lichman, 2013) six binary
classification datasets (australian, breast, crabs, ionosphere, pima, sonar), train an approximate sampler on crabs with a small neural network that has one 20-unit hidden layer with _ReLU_ activation,
and generalise to the remaining datasets with a bigger network that has 50 hidden units and uses
_sigmoid_ activation. We use ionosphere as the validation set to tune _ζ_ . The remaining 4 datasets
are further split into 40% training subset for simulating samples from the approximate sampler, and
60% test subsets for evaluating the sampler’s performance.


Figure 3 presents the (negative) test log-likelihood (LL), classification error, and an estimate of
the KSD U-statistic _S_ _U_ [2] [(] _[p]_ [(] _**[θ]**_ _[|D]_ [)] _[, q]_ [(] _**[θ]**_ [))][ (with data sub-sampling) over 5 splits of each test dataset.]
Besides the gradient estimators we also compare with two baselines: an approximate posterior sampler trained by maximum a posteriori (MAP), and stochastic gradient Langevin dynamics (SGLD)


8


Published as a conference paper at ICLR 2018


Figure 3: Generalisation performances for trained approximate posterior samplers.


(Welling & Teh, 2011) evaluated on the test datasets directly. In summary, SGLD returns best results in KSD metric. The Stein approach performs equally well or a little better than SGLD in
terms of test-LL and test error. The KDE method is slightly worse and is close to MAP, indicating
that the KDE estimator does not provide a very informative gradient for the entropy term. Surprisingly the score matching estimator method produces considerably worse results (except for breast
dataset), even after carefully tuning the bandwidth and the regularisation parameter _η_ . Future work
should investigate the usage of advanced recurrent neural networks such as an LSTM (Hochreiter &
Schmidhuber, 1997), which is expected to return better performance.


4.3 T OWARDS ADDRESSING MODE COLLAPSE IN GAN S USING ENTROPY REGULARISATION


GANs are notoriously difficult to train in practice. Besides the instability of gradient-based minimax
optimisation which has been partially addressed by many recent proposals (Salimans et al., 2016;
Arjovsky et al., 2017; Berthelot et al., 2017), they also suffer from mode collapse. We propose
adding an entropy regulariser to the GAN generator loss. Concretely, assume the generative model
_p_ _**θ**_ ( _**x**_ ) is implicitly defined by _**x**_ = _**f**_ _**θ**_ ( _**z**_ ) _,_ _**z**_ _∼_ _p_ 0 ( _**z**_ ), then the generator’s loss is defined by


˜
_J_ gen ( _**θ**_ ) = _J_ gen ( _**θ**_ ) _−_ _α_ H[ _p_ _**θ**_ ( _**x**_ )] _,_ (21)


where _J_ gen ( _**θ**_ ) is the original loss function for the generator from any GAN algorithm and _α_ is a
hyper-parameter. In practice (the gradient of) (21) is estimated using Monte Carlo.


We empirically investigate the entropy regularisation idea on the very recently proposed boundary
equilibrium GAN (BEGAN) (Berthelot et al., 2017) method using (continuous) MNIST, and we
refer to the appendix for the detailed mathematical set-up. In this case the non-parametric V-statistic
Stein gradient estimator is used. We use a convolutional generative network and a convolutional
auto-encoder and select the hyper-parameters of BEGAN _γ ∈{_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_, _α ∈_ [0 _,_ 1] and _λ_ =

_d_

0 _._ 001. The Epanechnikov kernel _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) := _d_ [1] � _j_ =1 [(1] _[−]_ [(] _[x]_ _[j]_ _[ −][x]_ _[′]_ _j_ [)] [2] [)][ is used as the pixel values lie]

in a unit interval (see appendix for the expression of the score matching estimator), and to ensure the
boundary condition we clip the pixel values into range [10 _[−]_ [8] _,_ 1 _−_ 10 _[−]_ [8] ]. The generated images are
visualised in Figure 4. BEGAN without the entropy regularisation fails to generate diverse samples
even when trained with learning rate decay. The other three images clearly demonstrate the benefit of
the entropy regularisation technique, with the Stein approach obtaining the highest diversity without
compromising visual quality.


We further consider four metrics to assess the trained models quantitatively. First 500 samples
are generated for each trained model, then we compute their nearest neighbours in the training set
using _l_ 1 distance, and obtain a probability vector **p** by averaging over these neighbour images’
label vectors. In Figure 5 we depict the entropy of **p** (top left), averaged _l_ 1 distances to the nearest
neighbour (top right), and the difference between the largest and smallest elements in **p** (bottom
right). The error bars are obtained by 5 independent runs. These results demonstrate that the Stein


9


Published as a conference paper at ICLR 2018


Figure 4: Visualisation of generated images from trained BEGAN models.


Figure 5: Quantitative evaluation on entropy regularised BEGAN. The higher the better for the LHS
panels and the other way around for the RHS ones. See main text for details.


approach performs significantly better than the other two, in that it learns a better generative model
not only faster but also in a more stable way. Interestingly the KDE approach achieves the lowest
average _l_ 1 distance to nearest neighbours, possibly because it tends to memorise training examples.
We next train a fully connected network _π_ ( _**y**_ _|_ _**x**_ ) on MNIST that achieves 98.16% text accuracy,
and compute on the generated images an empirical estimate of the inception score (Salimans et al.,
2016) E _p_ ( _**x**_ ) [KL[ _π_ ( _**y**_ _|_ _**x**_ ) _||π_ ( _**y**_ )]] with _π_ ( _**y**_ ) = E _p_ ( _**x**_ ) [ _π_ ( _**y**_ _|_ _**x**_ )] (bottom left panel). High inception
score indicates that the generate images tend to be both realistic looking and diverse, and again the
Stein approach out-performs the others on this metric by a large margin.


Concerning computation speed, all the three methods are of the same order: 10.20s/epoch for KDE,
10.85s/epoch for Score, and 10.30s/epoch for Stein. [1] This is because _K < d_ (in the experiments
_K_ = 100 and _d_ = 784) so that the complexity terms are dominated by kernel computations
( _O_ ( _K_ [2] _d_ )) required by all the three methods. Also for a comparison, the original BEGAN method
without entropy regularisation runs for 9.05s/epoch. Therefore the main computation cost is dominated by the optimisation of the discriminator/generator, and the proposed entropy regularisation
can be applied to many GAN frameworks with little computational burden.


5 C ONCLUSIONS AND FUTURE WORK


We have presented the Stein gradient estimator as a novel generalisation to the score matching gradient estimator. With a focus on learning implicit models, we have empirically demonstrated the
efficacy of the proposed estimator by showing how it opens the door to a range of novel learning
tasks: approximating gradient-free MCMC, meta-learning for approximate inference, and unsupervised learning for image generation. Future work will expand the understanding of gradient
estimators in both theoretical and practical aspects. Theoretical development will compare both
the V-statistic and U-statistic Stein gradient estimators and formalise consistency proofs. Practical
work will improve the sample efficiency of kernel estimators in high dimensions and develop fast
yet accurate approximations to matrix inversion. It is also interesting to investigate applications of
gradient approximation methods to training implicit generative models without the help of discriminators. Finally it remains an open question that how to generalise the Stein gradient estimator to
non-kernel settings and discrete distributions.


1 All the methods are timed on a machine with an NVIDIA GeForce GTX TITAN X GPU.


10


Published as a conference paper at ICLR 2018


A CKNOWLEDGEMENT


We thank Marton Havasi, Jiri Hron, David Janz, Qiang Liu, Maria Lomeli, Cuong Viet Nguyen and
Mark Rowland for their comments and helps on the manuscript. We also acknowledge the anonymous reviewers for their review. Yingzhen Li thanks Schlumberger Foundation FFTF fellowship.
Richard E. Turner thanks Google and EPSRC grants EP/M0269571 and EP/L000776/1.


R EFERENCES


Guillaume Alain and Yoshua Bengio. What regularized auto-encoders learn from the data-generating
distribution. _The Journal of Machine Learning Research_, 15(1):3563–3593, 2014.


Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul,
and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In _Advances in_
_Neural Information Processing Systems_, pp. 3981–3989, 2016.


Martin Arjovsky, Soumith Chintala, and Leon Bottou. Wasserstein gan. _arXiv preprint_
_arXiv:1701.07875_, 2017.


David Barber and Felix V Agakov. The im algorithm: A variational approach to information maximization. In _NIPS_, pp. 201–208, 2003.


Alain Berlinet and Christine Thomas-Agnan. _Reproducing kernel Hilbert spaces in probability and_
_statistics_ . Springer Science & Business Media, 2011.


David Berthelot, Tom Schumm, and Luke Metz. Began: Boundary equilibrium generative adversarial networks. _arXiv preprint arXiv:1703.10717_, 2017.


Kacper Chwialkowski, Heiko Strathmann, and Arthur Gretton. A kernel test of goodness of fit. In
_Proceedings of The 33rd International Conference on Machine Learning_, pp. 2606–2615, 2016.


Kris De Brabanter, Jos De Brabanter, Bart De Moor, and Ir`ene Gijbels. Derivative estimation with
local polynomial fitting. _The Journal of Machine Learning Research_, 14(1):281–301, 2013.


Peter J Diggle and Richard J Gratton. Monte carlo methods of inference for implicit statistical
models. _Journal of the Royal Statistical Society. Series B (Methodological)_, pp. 193–227, 1984.


Jianqing Fan and Irne Gijbels. _Local polynomial modelling and its applications_ . Chapman & Hall,
1996.


Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In _NIPS_, 2014.


Jackson Gorham and Lester Mackey. Measuring sample quality with stein’s method. In _NIPS_, 2015.


Jackson Gorham and Lester Mackey. Measuring sample quality with kernels. In _ICML_, 2017.


Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. _Neural_
_computation_, 14(8):1771–1800, 2002.


Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural computation_, 9(8):
1735–1780, 1997.


Ferenc Husz´ar. Variational inference using implicit distributions. _arXiv preprint arXiv:1702.08235_,
2017.


Aapo Hyv¨arinen. Estimation of non-normalized statistical models by score matching. _Journal of_
_Machine Learning Research_, 6(Apr):695–709, 2005.


Aapo Hyv¨arinen. Consistency of pseudolikelihood estimation of fully visible boltzmann machines.
_Neural Computation_, 18(10):2283–2292, 2006.


Oliver Thomas Johnson. _Information theory and the central limit theorem_ . World Scientific, 2004.


11


Published as a conference paper at ICLR 2018


Theofanis Karaletsos. Adversarial message passing for graphical models. _arXiv preprint_
_arXiv:1612.05048_, 2016.


Diederick P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International_
_Conference on Learning Representations (ICLR)_, 2015.


Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt Kira. How to train your dragan. _arXiv_
_preprint arXiv:1705.07215_, 2017.


Yingzhen Li and Qiang Liu. Wild variational approximations. _NIPS workshop on advances in_
_approximate Bayesian inference_, 2016.


Yingzhen Li, Richard E Turner, and Qiang Liu. Approximate inference with amortised mcmc. _arXiv_
_preprint arXiv:1702.08343_, 2017.


[M. Lichman. UCI machine learning repository, 2013. URL http://archive.ics.uci.edu/](http://archive.ics.uci.edu/ml)
[ml.](http://archive.ics.uci.edu/ml)


Qiang Liu and Yihao Feng. Two methods for wild variational inference. _arXiv preprint_
_arXiv:1612.00081_, 2016.


Qiang Liu, Jason D Lee, and Michael I Jordan. A kernelized stein discrepancy for goodness-of-fit
tests and model evaluation. In _ICML_, 2016.


Siwei Lyu. Interpretation and generalization of score matching. In _Proceedings of the Twenty-Fifth_
_Conference on Uncertainty in Artificial Intelligence_, pp. 359–366. AUAI Press, 2009.


Benjamin Marlin, Kevin Swersky, Bo Chen, and Nando Freitas. Inductive principles for restricted
boltzmann machine learning. In _Proceedings of the Thirteenth International Conference on Arti-_
_ficial Intelligence and Statistics_, pp. 509–516, 2010.


Lars Mescheder, Sebastian Nowozin, and Andreas Geiger. Adversarial variational bayes: Unifying
variational autoencoders and generative adversarial networks. _arXiv preprint arXiv:1701.04722_,
2017.


Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. _arXiv_
_preprint arXiv:1610.03483_, 2016.


Radford M Neal et al. Mcmc using hamiltonian dynamics. _Handbook of Markov Chain Monte_
_Carlo_, 2:113–162, 2011.


Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep
convolutional generative adversarial networks. In _ICLR_, 2016.


Rajesh Ranganath, Jaan Altosaar, Dustin Tran, and David M. Blei. Operator variational inference.
In _NIPS_, 2016.


Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In
_ICML_, 2015.


Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the landing: An asymptotically zerovariance gradient estimator for variational inference. _arXiv preprint arXiv:1703.09194_, 2017.


Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen.
Improved techniques for training gans. In _NIPS_, 2016.


Jaakko S¨arel¨a and Harri Valpola. Denoising source separation. _Journal of machine learning re-_
_search_, 6(Mar):233–272, 2005.


Hiroaki Sasaki, Aapo Hyv¨arinen, and Masashi Sugiyama. Clustering via mode seeking by direct
estimation of the gradient of a log-density. In _Joint European Conference on Machine Learning_
_and Knowledge Discovery in Databases_, pp. 19–34. Springer, 2014.


Radhey S Singh. Improvement on some known nonparametric uniformly consistent estimators of
derivatives of a density. _The Annals of Statistics_, pp. 394–399, 1977.


12


Published as a conference paper at ICLR 2018


Alex J Smola and Bernhard Sch¨okopf. Sparse greedy matrix approximation for machine learning.
In _Proceedings of the Seventeenth International Conference on Machine Learning_, pp. 911–918.
Morgan Kaufmann Publishers Inc., 2000.


Casper Kaae Sonderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Husz´ar. Amortised
map inference for image super-resolution. In _ICLR_, 2017.


Charles Stein. A bound for the error in the normal approximation to the distribution of a sum of
dependent random variables. In _Proceedings of the Sixth Berkeley Symposium on Mathematical_
_Statistics and Probability, Volume 2: Probability Theory_, pp. 583–602, 1972.


Charles M Stein. Estimation of the mean of a multivariate normal distribution. _The annals of_
_Statistics_, pp. 1135–1151, 1981.


Charles J Stone. Additive regression and other nonparametric models. _The annals of Statistics_, pp.
689–705, 1985.


Heiko Strathmann, Dino Sejdinovic, Samuel Livingstone, Zoltan Szabo, and Arthur Gretton.
Gradient-free hamiltonian monte carlo with efficient kernel exponential families. In _Advances_
_in Neural Information Processing Systems_, pp. 955–963, 2015.


Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Shohei Hido, Jun Sese, Ichiro Takeuchi, and
Liwei Wang. A density-ratio framework for statistical data processing. _Information and Media_
_Technologies_, 4(4):962–987, 2009.


Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density-ratio matching under the bregman divergence: a unified framework of density-ratio estimation. _Annals of the Institute of Sta-_
_tistical Mathematics_, 64(5):1009–1044, 2012.


Dustin Tran, Rajesh Ranganath, and David M Blei. Deep and hierarchical implicit models. _arXiv_
_preprint arXiv:1702.08896_, 2017.


R. E. Turner and M. Sahani. Two problems with variational expectation maximisation for time-series
models. In D. Barber, T. Cemgil, and S. Chiappa (eds.), _Bayesian Time series models_, chapter 5,
pp. 109–130. Cambridge University Press, 2011.


Masatoshi Uehara, Issei Sato, Masahiro Suzuki, Kotaro Nakayama, and Yutaka Matsuo. Generative
adversarial nets from a density ratio estimation perspective. _arXiv preprint arXiv:1610.02920_,
2016.


Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural compu-_
_tation_, 23(7):1661–1674, 2011.


Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and
composing robust features with denoising autoencoders. In _Proceedings of the 25th international_
_conference on Machine learning_, pp. 1096–1103. ACM, 2008.


Dilin Wang and Qiang Liu. Learning to draw samples: With application to amortized mle for
generative adversarial learning. _arXiv preprint arXiv:1611.01722_, 2016.


Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In
_Proceedings of the 28th International Conference on Machine Learning (ICML-11)_, pp. 681–688,
2011.


Christopher KI Williams and Matthias Seeger. Using the nystr¨om method to speed up kernel machines. In _Advances in neural information processing systems_, pp. 682–688, 2001.


Lantao Yu, Weinan Zhang, Jun Wang, and Yong Yu. Seqgan: sequence generative adversarial nets
with policy gradient. In _Thirty-First AAAI Conference on Artificial Intelligence_, 2017.


Shanggang Zhou and Douglas A Wolfe. On derivative estimation in spline regression. _Statistica_
_Sinica_, pp. 93–108, 2000.


13


Published as a conference paper at ICLR 2018


A S CORE MATCHING ESTIMATOR : REMARKS AND DERIVATIONS


In this section we provide more discussions and analytical solutions for the score matching estimator. More specifically, we will derive the linear coefficient _**a**_ = ( _a_ 1 _, ..., a_ _K_ ) for the case of the
Epanechnikov kernel.


A.1 S OME REMARKS ON SCORE MATCHING


**Remark.** It has been shown in S¨arel¨a & Valpola (2005); Alain & Bengio (2014) that de-noising autoencoders (DAEs) (Vincent et al., 2008), once trained, can be used to compute the score function
approximately. Briefly speaking, a DAE learns to reconstruct a datum˜ _**x**_ from a corrupted input
_**x**_ = _**x**_ + _σ_ _**ϵ**_ _,_ _**ϵ**_ _∼N_ ( **0** _,_ **I** ) by minimising the mean square error. Then the optimal DAE can be used to
1
approximate the score function as _∇_ _**x**_ log _p_ ( _**x**_ ) _≈_ _σ_ [2] [(][DAE] _[∗]_ [(] _**[x]**_ [)] _[−]_ _**[x]**_ [)][. Sonderby et al. (2017) applied]
this idea to train an implicit model for image super-resolution, providing some promising results
in some metrics. However applying similar ideas to variational inference can be computationally
expensive, because the estimation of _∇_ _**z**_ log _q_ ( _**z**_ _|_ _**x**_ ) is a sub-routine for VI which is repeatedly
required. Therefore in the paper we deploy kernel machines that allow analytical solutions to the
score matching estimator in order to avoid double loop optimisation.


**Remark.** As a side note, score matching can also be used to learn the parameters of an unnormalised density. In this case the target distribution _q_ would be the data distribution and ˆ _q_ is often
a Boltzmann distribution with intractable partition function. As a parameter estimation technique,
score matching is also related to contrastive divergence (Hinton, 2002), pseudo likelihood estimation (Hyv¨arinen, 2006), and DAEs (Vincent, 2011; Alain & Bengio, 2014). Generalisations of score
matching methods are also presented in e.g. Lyu (2009); Marlin et al. (2010).


A.2 T HE RBF KERNEL CASE


The derivations for the RBF kernel case is referred to (Strathmann et al., 2015), and for completeness we include the final solutions here. Assume the parametric approximation is defined as
log ˆ _q_ ( _**x**_ ) = [�] _[K]_ _k_ =1 _[a]_ _[k]_ _[K]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[k]_ [) +] _[ C]_ [, where the RBF kernel uses bandwidth parameter] _[ σ]_ [. then the]
optimal solution of the coefficients ˆ _**a**_ [score] = ( **Σ** + _η_ **I** ) _[−]_ [1] _**v**_, with



_,_
��



_**v**_ =



_d_
�


_i_ =1



_σ_ [2] **K1** _−_ **K** ( **x** _i_ _⊙_ **x** _i_ ) + diag( **x** _i_ ) **K1** _−_ 2diag( **x** _i_ ) **Kx** _i_
� �



**Σ** =



_d_
� [diag( **x** _i_ ) **K** _−_ **K** diag( **x** _i_ )] [ **K** diag( **x** _i_ ) _−_ diag( **x** _i_ ) **K** ] _,_


_i_ =1


**x** _i_ = ( _x_ [1] _i_ _[, x]_ [2] _i_ _[, ..., x]_ _[K]_ _i_ [)] [T] _[ ∈]_ [R] _[K][×]_ [1] _[.]_



A.3 T HE E PANECHNIKOV KERNEL CASE


The Epanechnikov kernel is defined as _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _d_ [1] � _di_ =1 [(1] _[ −]_ [(] _[x]_ _[i]_ _[ −]_ _[x]_ _i_ _[′]_ [)] [2] [)][, where the first and]

second order gradients w.r.t. _x_ _i_ is



_∇_ _x_ _i_ _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = [2]



_d_ [2] [(] _[x]_ _i_ _[′]_ _[−]_ _[x]_ _[i]_ [)] _[,]_ _∇_ _x_ _i_ _∇_ _x_ _i_ _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _−_ _d_ [2]



(22)
_d_ _[.]_



Thus the score matching objective with log ˆ _q_ ( _**x**_ ) = [�] _[K]_ _k_ =1 _[a]_ _[k]_ _[K]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[k]_ [) +] _[ C]_ [ is reduced to]



2

� _a_ _k_ _d_ _[d]_

_k_ =1



�



�



_F_ ( _**a**_ ) = [1]

_K_


= [4]

_K_



_K_
�

_j_ =1


_K_
�

_j_ =1



1

_d_ [2]
�



_K_
�


_k_ =1



_||_



_K_

2

� _a_ _k_ _d_ [(] _**[x]**_ _[k]_ _[ −]_ _**[x]**_ _[j]_ [)] _[||]_ 2 [2] _[−]_ [2]

_k_ =1



_K_
�



_K_
� _a_ _k_ _a_ _k_ _′_ ( _**x**_ _[k]_ _−_ _**x**_ _[j]_ ) [T] ( _**x**_ _[k]_ _[′]_ _−_ _**x**_ _[j]_ ) _−_ _**a**_ [T] **1**


_k_ _[′]_ =1



�



:= 4( _**a**_ [T] **Σ** _**a**_ _−_ _**a**_ [T] **1** ) _,_



14


Published as a conference paper at ICLR 2018


with the matrix elements



� _||_ _**x**_ _[j]_ _||_ 2 [2] _[−]_ [(] _**[x]**_ _[k]_ [ +] _**[ x]**_ _[k]_ _[′]_ [)] [T] _**[x]**_ _[j]_ [�]  _._





**Σ** _kk_ _′_ = [1]

_d_ [2]







( _**x**_ _[k]_ ) [T] _**x**_ _[k]_ _[′]_ + [1]
 _K_



_K_



_K_
�

_j_ =1



Define the “gram matrix” X _ij_ = ( _**x**_ _[i]_ ) [T] _**x**_ _[j]_, we write the matrix form of **Σ** as



**Σ** = [1]

_d_ [2]



X + [1] �Tr(X) _−_ 2X **11** [T] [��] _._
� _K_



Thus with an _l_ 2 regulariser, the fitted coefficients are



ˆ
_**a**_ [score] = _[d]_ [2]

2



_−_ 1

X + [1] �Tr(X) _−_ 2X **11** [T] [�] + _η_ **I** **1** _._
� _K_ �



B S TEIN GRADIENT ESTIMATOR : DERIVATIONS


B.1 D IRECT MINIMISATION OF KSD V- STATISTIC AND U- STATISTIC


The V-statistic of KSD is the following: given samples _**x**_ _[k]_ _∼_ _q, k_ = 1 _, ..., K_ and recall **K** _jl_ =
_K_ ( _**x**_ _[j]_ _,_ _**x**_ _[l]_ )



_K_
�


_l_ =1



1
_S_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] _K_ [2]



_K_
�

_j_ =1



� _**g**_ ˆ( _**x**_ _j_ ) T **K** _jl_ _**g**_ ˆ( _**x**_ _l_ ) + ˆ _**g**_ ( _**x**_ _j_ ) T _∇_ _**x**_ _l_ **K** _jl_ + _∇_ _**x**_ _j_ **K** T _jl_ _**[g]**_ [ˆ][(] _**[x]**_ _[l]_ [) +][ Tr][(] _[∇]_ _**x**_ _[j]_ _,_ _**x**_ _[l]_ **[K]** _[jl]_ [)] � _._



(23)
The last termnotations defined in the main text, readers can verify that the V-statistic can be computed as _∇_ _**x**_ _j_ _,_ _**x**_ _l_ **K** _jl_ will be ignored as it does not depend on the approximation ˆ _**g**_ . Using matrix


1
_S_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] _K_ [2] [Tr][(] **[K]** [ ˆ] **[G]** [ ˆ] **[G]** [T] [ + 2] _[⟨∇][,]_ **[ K]** _[⟩]_ **[G]** [ˆ] [T] [) +] _[ C.]_ (24)


Using the cyclic invariance of matrix trace leads to the desired result in the main text. The U-statistic
of KSD removes terms indexed by _j_ = _l_ in (23), in which the matrix form is


1
_S_ _U_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] (25)
_K_ ( _K −_ 1) [Tr][((] **[K]** _[ −]_ [diag][(] **[K]** [)) ˆ] **[G]** [ ˆ] **[G]** [T] [ + 2(] _[⟨∇][,]_ **[ K]** _[⟩−∇]_ [diag][(] **[K]** [)) ˆ] **[G]** [T] [) +] _[ C.]_


with the _j_ th row of _∇_ diag( **K** ) defined as _∇_ _**x**_ _j_ _K_ ( _**x**_ _[j]_ _,_ _**x**_ _[j]_ ). For most translation invariant kernels this
extra term _∇_ diag( **K** ) = **0**, thus the optimal solution of **G** [ˆ] by minimising KSD U-statistic is


ˆ
**G** [Stein] _U_ = _−_ ( **K** _−_ diag( **K** ) + _η_ _**I**_ ) _[−]_ [1] _⟨∇,_ **K** _⟩._ (26)


B.2 D ERIVING THE NON - PARAMETRIC PREDICTIVE ESTIMATOR



Let us consider an unseen datum _**y**_ . If _**y**_ is sampled from the _q_ distribution, then one can also
apply the non-parametric estimator (9) for gradient approximations, given the observed data **X** =
_{_ _**x**_ [1] _, ...,_ _**x**_ _[K]_ _} ∼_ _q_ . Concretely, if writing ˆ _**g**_ ( _**y**_ ) _≈∇_ _**y**_ log _q_ ( _**y**_ ) _∈_ R _[d][×]_ [1] then the non-parametric
Stein gradient estimator (using V-statistic) is
� _**g**_ ˆ( **G** _**y**_ ˆ ) T � = _−_ ( **K** _[∗]_ + _η_ _**I**_ ) _[−]_ [1] � _∇_ _**y**_ _K_ ( _**y**_ _⟨∇,_ _**y**_ _,_ ) + **K** _⟩_ [�] + _∇_ _[K]_ _k_ =1 _**y**_ _K_ _[∇]_ ( _**[x]**_ _·,_ _[k]_ _**y**_ _[K]_ ) [(] _**[y]**_ _[,]_ _**[ x]**_ _[k]_ [)] � _,_ **K** _[∗]_ = � **KK** **X** _**yyy**_ **KK** _**y**_ **X** � _,_


with _∇_ _**y**_ _K_ ( _·,_ _**y**_ ) denoting a _K_ _×d_ matrix with rows _∇_ _**y**_ _K_ ( _**x**_ _[k]_ _,_ _**y**_ ), and _∇_ _**y**_ _K_ ( _**y**_ _,_ _**y**_ ) only differentiates
through the second argument. Thus by simple matrix calculations, we have:



� = _−_ ( **K** _[∗]_ + _η_ _**I**_ ) _[−]_ [1] � _∇_ _**y**_ _K_ ( _**y**_ _⟨∇,_ _**y**_ _,_ ) + **K** _⟩_ [�] + _∇_ _[K]_ _k_ =1 _**y**_ _K_ _[∇]_ ( _**[x]**_ _·,_ _[k]_ _**y**_ _[K]_ ) [(] _**[y]**_ _[,]_ _**[ x]**_ _[k]_ [)]



� _,_ **K** _[∗]_ = � **KK** **X** _**yyy**_ **KK** _**y**_ **X**



_,_
�



_−_ 1
_∇_ _**y**_ log _q_ ( _**y**_ ) [T] _≈−_ � **K** _**yy**_ + _η −_ **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] **K** **X** _**y**_ �


_K_

_∇_ _**y**_ _K_ ( _**y**_ _,_ _**y**_ ) + � _∇_ _**x**_ _k_ _K_ ( _**y**_ _,_ _**x**_ _[k]_ ) + **K** _**y**_ **X**

� _k_ =1



�



_∇_ _**y**_ _K_ ( _**y**_ _,_ _**y**_ ) +



_K_
�



� _∇_ _**x**_ _k_ _K_ ( _**y**_ _,_ _**x**_ _[k]_ ) + **K** _**y**_ **X** **G** [ˆ] [Stein] _V_ _−_ **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] _∇_ _**y**_ _K_ ( _·,_ _**y**_ )


_k_ =1



_._



(27)



15


Published as a conference paper at ICLR 2018


For translation invariant kernels, typically _∇_ _**y**_ _K_ ( _**y**_ _,_ _**y**_ ) = **0**, and more conveniently,


_∇_ _**x**_ _k_ _K_ ( _**y**_ _,_ _**x**_ _[k]_ ) = _∇_ _**x**_ _k_ ( _**x**_ _[k]_ _−_ _**y**_ ) _∇_ ( _**x**_ _k_ _−_ _**y**_ ) _K_ ( _**x**_ _[k]_ _−_ _**y**_ ) = _−∇_ _**y**_ _K_ ( _**x**_ _[k]_ _,_ _**y**_ ) _._


Thus equation (27) can be further simplified to (with column vector **1** _∈_ R _[K][×]_ [1] )


_−_ 1
_∇_ _**y**_ log _q_ ( _**y**_ ) [T] _≈−_ � **K** _**yy**_ + _η −_ **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] **K** **X** _**y**_ �
(28)
� **K** _**y**_ **X** **G** [ˆ] [Stein] _V_ _−_ � **K** _**y**_ **X** ( **K** + _η_ **I** ) _[−]_ [1] + **1** [T] [�] _∇_ _**y**_ _K_ ( _·,_ _**y**_ )� _._


The solution for the U-statistic case can be derived accordingly which we omit here.


B.3 P ARAMETRIC S TEIN GRADIENT ESTIMATOR WITH THE RBF KERNEL


We define a parametric approximation in a similar way as for the score matching estimator:



2 _σ_ [2] _[||]_ _**[x]**_ _[ −]_ _**[x]**_ _[′]_ _[||]_ 2 [2]



_._ (29)
�



log ˆ _q_ ( _**x**_ ) :=



_K_
�



_k_ � =1 _a_ _k_ _K_ ( _**x**_ _,_ _**x**_ _[k]_ ) + _C,_ _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = exp � _−_ 2 _σ_ [1]



Now we show the optimal solution of _**a**_ = ( _a_ 1 _, ..., a_ _K_ ) [T] by minimising (23). To simplify derivations
we assume the approximation and KSD use the same kernel. First note that the gradient of the RBF
kernel is

_∇_ _**x**_ _K_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _σ_ [1] [2] _[K]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[′]_ [)(] _**[x]**_ _[′]_ _[ −]_ _**[x]**_ [)] _[.]_ (30)


Substituting (30) into (23):
_S_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] _[ C]_ [ +] _[ ♣]_ [+ 2] _[♠][,]_



_K_

1
_♣_ = _K_ [2] �

_k_ =1



_K_
�


_k_ _[′]_ =1



_K_
�

_j_ =1



_K_
�



� _a_ _k_ _a_ _k_ _′_ **K** _kj_ **K** _jl_ **K** _lk_ _′_ _σ_ [1]

_l_ =1



_σ_ [4] [(] _**[x]**_ _[k]_ _[ −]_ _**[x]**_ _[j]_ [)] [T] [(] _**[x]**_ _[k]_ _[′]_ _[ −]_ _**[x]**_ _[l]_ [)] _[,]_ (31)



_K_

1

� _a_ _k_ **K** _kj_ **K** _jl_ _σ_ [4] [(] _**[x]**_ _[k]_ _[ −]_ _**[x]**_ _[j]_ [)] [T] [(] _**[x]**_ _[j]_ _[ −]_ _**[x]**_ _[l]_ [)] _[.]_ (32)

_l_ =1



1
_♠_ =
_K_ [2]



_K_ _K_
� �

_k_ =1 _j_ =1



We first consider summing the _j_, _l_ indices in _♣_ . Recall the “gram matrix” X _ij_ = ( _**x**_ _[i]_ ) [T] _**x**_ _[j]_, the inner
product term in _♣_ can be expressed as X _kk_ _[′]_ + X _jl_ _−_ X _kl_ _−_ X _jk_ _[′]_ . Thus the summation over _j_, _l_ can
be re-written as



**Λ** :=



_K_
�

_j_ =1



_K_
� **K** _kj_ **K** _jl_ **K** _lk_ _′_ (X _kk_ _′_ + X _jl_ _−_ X _kl_ _−_ X _jk_ _′_ )


_l_ =1



_K_
�



= X _⊙_ ( **KKK** ) + **K** ( **K** _⊙_ X) **K** _−_ (( **KK** ) _⊙_ X) **K** _−_ **K** (( **KK** ) _⊙_ X) _._


And thus _♣_ = _σ_ 1 [4] _**[a]**_ [T] **[Λ]** _**[a]**_ [. Similarly the summation over] _[ j]_ [,] _[ l]_ [ in] _[ ♠]_ [can be simplified into]



_K_
� **K** _kj_ **K** _jl_ (X _kj_ + X _jl_ _−_ X _kl_ _−_ X _jj_ )


_l_ =1



_−_ _**b**_ :=



_K_
�

_j_ =1



= _−_ ( **K** diag(X) **K** + ( **KK** ) _⊙_ X _−_ **K** ( **K** _⊙_ X) _−_ ( **K** _⊙_ X) **K** ) **1** _,_


which leads to _♠_ = _−_ _σ_ [1] [4] _**[a]**_ [T] _**[b]**_ [. Thus minimising] _[ S]_ _V_ [2] [(] _[q,]_ [ ˆ] _[q]_ [)][ plus an] _[ l]_ [2] [ regulariser returns the Stein]

estimator ˆ _**a**_ [Stein] _V_ in the main text.


Similarly we can derive the solution for KSD U-statistic minimisation. The U statistic can also be
represented in quadratic form _S_ _U_ [2] [(] _[q,]_ [ ˆ] _[q]_ [) =] _[ C]_ [ + ˜] _[♣]_ [+ 2˜] _[♠]_ [, with][ ˜] _[♠]_ [=] _[ ♠]_ [and]



_K_
�


_k_ _[′]_ =1



_K_
�



� _a_ _k_ _a_ _k_ _′_ **K** _kj_ **K** _jj_ **K** _jk_ _′_ _σ_ [1]

_j_ =1



_σ_ [4] [(][X] _[kk]_ _[′]_ [ +][ X] _[jj]_ _[ −]_ [X] _[kj]_ _[ −]_ [X] _[jk]_ _[′]_ [)] _[.]_



˜
_♣_ = _♣−_ [1]

_K_ [2]



_K_
�


_k_ =1



16


Published as a conference paper at ICLR 2018


Summing over the _j_ indices for the second term, we have


_K_
� **K** _kj_ **K** _jj_ **K** _jk_ _[′]_ (X _kk_ _[′]_ + X _jj_ _−_ X _kj_ _−_ X _jk_ _[′]_ )

_j_ =1


= X _⊙_ ( **K** diag( **K** ) **K** ) + **K** diag( **K** _⊙_ X) **K** _−_ (( **K** diag( **K** )) _⊙_ X) **K** _−_ **K** ((diag( **K** ) **K** ) _⊙_ X) _._


Working through the analogous derivations reveals that ˆ _**a**_ [Stein] _U_ = ( **Λ** [˜] + _η_ **I** ) _[−]_ [1] _**b**_, with


**Λ** ˜ =X _⊙_ ( **K** ( **K** _−_ diag( **K** )) **K** ) + **K** (( **K** _⊙_ X) _−_ diag( **K** _⊙_ X)) **K**
_−_ (( **K** ( **K** _−_ diag( **K** ))) _⊙_ X) **K** _−_ **K** ((( **K** _−_ diag( **K** )) **K** ) _⊙_ X) _._


C M ORE DETAILS ON THE EXPERIMENTS


We describe the detailed experimental set-up in this section. All experiments use Adam optimiser
(Kingma & Ba, 2015) with standard parameter settings.


C.1 A PPROXIMATE POSTERIOR SAMPLER EXPERIMENTS


We start by reviewing Bayesian neural networks with binary classification as a running example.
In this task, a normal deep neural network is constructed to predict _y_ = _**f**_ _**θ**_ ( _**x**_ ), and the neural
network is parameterised by a set of weights (and bias vectors which we omit here for simplicity)
_**θ**_ = _{_ **W** _[l]_ _}_ _[L]_ _l_ =1 [. In the Bayesian framework these network weights are treated as random variables,]
and a prior distribution, e.g. Gaussian, is also attached to them: _p_ 0 ( _**θ**_ ) = _N_ ( _**θ**_ ; **0** _,_ **I** ). The likelihood
function of _**θ**_ is then defined as


_p_ ( _y_ = 1 _|_ _**x**_ _,_ _**θ**_ ) = sigmoid(NN _**θ**_ ( _**x**_ )) _,_


and _p_ ( _y_ = 0 _|_ _**x**_ _,_ _**θ**_ ) = 1 _−_ _p_ ( _y_ = 1 _|_ _**x**_ _,_ _**θ**_ ) accordingly. One can show that the usage of Bernoulli
distribution here corresponds to applying cross entropy loss for training.


After framing the deep neural network as a probabilistic model, a Bayesian approach would find the
posterior of the network weights _p_ ( _**θ**_ _|D_ ) and use the uncertainty information encoded in it for future
predictions. By Bayes’ rule, the exact posterior is



_p_ ( _**θ**_ _|D_ ) _∝_ _p_ 0 ( _**θ**_ )



_N_
� _p_ ( _y_ _n_ _|_ _**x**_ _n_ _,_ _**θ**_ ) _,_


_n_ =1



and the predictive distribution for a new input _**x**_ _[∗]_ is


_p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _, D_ ) = _p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _,_ _**θ**_ ) _p_ ( _**θ**_ _|D_ ) _d_ _**θ**_ _._ (33)
�


Again the exact posterior is intractable, and approximate inference would fit an approximate posterior distribution _q_ _**φ**_ ( _**θ**_ ) parameterised by the variational parameters _**φ**_ to the exact posterior, and then
use it to compute the (approximate) predictive distribution.


_p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _, D_ ) _≈_ _p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _,_ _**θ**_ ) _q_ _**φ**_ ( _**θ**_ ) _d_ _**θ**_ _._
�


Since in practice analytical integration for neural network weights is also intractable, the predictive
distribution is further approximated by Monte Carlo:



_p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _, D_ ) _≈_ [1]

_K_



_K_
� _p_ ( _y_ _[∗]_ = 1 _|_ _**x**_ _[∗]_ _,_ _**θ**_ _[k]_ ) _,_ _**θ**_ _[k]_ _∼_ _q_ _**φ**_ ( _**θ**_ ) _._


_k_ =1



Now it remains to fit the approximate posterior _q_ _**φ**_ ( _**θ**_ ), and in the experiment the approximate posterior is implicitly constructed by a stochastic flow. For the training task, we use a one hidden
layer neural network with 20 hidden units to compute the noise variance and the moving direction
of the next update. In a nutshell it takes the _i_ th coordinate of the current position and the gradient _**θ**_ _t_ ( _i_ ) _, ∇_ _t_ ( _i_ ) as the inputs, and output the corresponding coordinate of the moving direction


17


Published as a conference paper at ICLR 2018


∆ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ )( _i_ ) and the noise variance _**σ**_ _**φ**_ ( _**θ**_ _t_ _, ∇_ _t_ )( _i_ ). Softplus non-linearity is used for the hidden
layer and to compute the noise variance we apply ReLU activation to ensure non-negativity. The
step-size _ζ_ is selected as 1e-5 which is tuned on the KDE approach. For SGLD step-size 1e-5 also
returns overall good results.


The training process is the following. We simulate the approximate sampler for 10 transitions and
sum over the variational lower-bounds computed on the samples of every step. Concretely, the
maximisation objective is



_L_ ( _**φ**_ ) =



_T_
� _L_ VI ( _q_ _t_ ) _,_


_t_ =1



where _T_ = 100 and _q_ _t_ ( _**θ**_ ) is implicitly defined by the marginal distribution of _**θ**_ _t_ that is dependent
on _**φ**_ . In practice the variational lower-bound _L_ VI ( _q_ _t_ ) is further approximated by Monte Carlo and
data sub-sampling:



_L_ VI ( _q_ _t_ ) _≈_ _[N]_

_M_



_M_
� log _p_ ( _y_ _m_ _|_ _**x**_ _m_ _,_ _**θ**_ _t_ ) + log _p_ 0 ( _**θ**_ _t_ ) _−_ log _q_ _t_ ( _**θ**_ _t_ ) _._


_m_ =1



The MAP baseline considers an alternative objective function by removing the log _q_ _t_ ( _**θ**_ _t_ ) term from
the above MC-VI objective.


Truncated back-propagation is applied for every 10 steps in order to avoid vanishing/exploding
gradients. The simulated samples at time _T_ are stored to initialise the Markov chain for the next
iteration, and for every 50 iterations we restart the simulation by randomly sampling the locations
from the prior. Early stopping is applied using the validation dataset, and the learning rate is set to
0.001, the number of epochs is set to 500.


We perform hyper-parameter search for the kernel, i.e. a grid search on the bandwidth _σ_ [2] _∈_
_{_ 0 _._ 25 _,_ 1 _._ 0 _,_ 4 _._ 0 _,_ 10 _._ 0 _,_ median trick _}_ and _η ∈{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _._ 0 _,_ 2 _._ 0 _}_ . We found the median heuristic is
sufficient for the KDE and Stein approaches. However, we failed to obtain desirable results using the
score matching estimator with median heuristics, and for other settings the score matching approach
underperforms when compared to KDE and Stein methods.


C.2 BEGAN EXPERIMENTS


In this section we describe the experimental details of the BEGAN experiment, but first we introduce
the mathematical idea and discuss how the entropy regulariser is applied.


Assume the generator is implicitly defined: _**x**_ _∼_ _p_ _**θ**_ ( _**x**_ ) _↔_ _**x**_ = _**f**_ _**θ**_ ( _**z**_ ) _,_ _**z**_ _∼_ _p_ 0 ( _**z**_ ). In BEGAN the
discriminator is defined as an auto-encoder _D_ _**ϕ**_ ( _**x**_ ) that reconstructs the input _**x**_ . After selecting a
ratio parameter _γ >_ 0, a control rate _β_ 0 initialised at 0, and a “learning rate” _λ >_ 0 for the control
rate, the loss functions for the generator _**x**_ = _**f**_ _**θ**_ ( _**z**_ ) _,_ _**z**_ _∼_ _p_ 0 ( _**z**_ ) and the discriminator are:



_J_ ( _**x**_ ) = _||D_ _**ϕ**_ ( _**x**_ ) _−_ _**x**_ _||,_ _|| · ||_ = _|| · ||_ 2 [2] [or] _[ || · ||]_ [1] _[,]_
_J_ gen ( _**θ**_ ; _**ϕ**_ ) = _J_ ( _**f**_ _**θ**_ ( _**z**_ )) _,_ _**z**_ _∼_ _p_ 0 ( _**z**_ )
_J_ dis ( _**ϕ**_ ; _**θ**_ ) = _J_ ( _**x**_ ) _−_ _β_ _t_ _J_ gen ( _**θ**_ ; _**ϕ**_ ) _,_ _**x**_ _∼D_
_β_ _t_ +1 = _β_ _t_ + _λ_ ( _γJ_ ( _**x**_ ) _−J_ ( _**f**_ _**θ**_ ( _**z**_ ))) _._



(34)




_·_
The main idea behind BEGAN is that, as the reconstruction loss _J_ ( ) is approximately Gaussian
distributed, with _γ_ = 1 the discriminator loss _J_ dis is (approximately) proportional to the Wasserstein distance between loss distributions induced by the data distribution _p_ _D_ ( _**x**_ ) and the generator
_p_ _**θ**_ ( _**x**_ ). In practice it is beneficial to maintain the equilibrium _γ_ E _p_ _D_ [ _J_ ( _**x**_ )] = E _p_ _**θ**_ [ _J_ ( _**x**_ )] through
the optimisation procedure described in (34) that is motivated by proportional control theory. This
approach effectively stabilises training, however it suffers from catastrophic mode collapsing problem (see the left most panel in Figure 4). To address this issue, we simply subtract an entropy term
from the generator’s loss function, i.e.

˜
_J_ gen ( _**θ**_ ; _**ϕ**_ ) = _J_ gen ( _**θ**_ ; _**ϕ**_ ) _−_ _α_ H[ _p_ _**θ**_ ] _,_ (35)


where the rest of the optimisation objectives remains as in (34). This procedure would maintain
the equilibrium _γ_ E _p_ _D_ [ _J_ ( _**x**_ )] = E _p_ _**θ**_ [ _J_ ( _**x**_ )] _−_ _α_ H[ _p_ ]. We approximate the gradient _∇_ _**θ**_ H[ _p_ _**θ**_ ] using the estimators presented in the main text. For the purpose of updating the control rate _β_ _t_ two


18


Published as a conference paper at ICLR 2018


strategies are considered to approximate the contribution of the entropy term. Given _K_ samples
_**x**_ [1] _, ...,_ _**x**_ _[k]_ _∼_ _p_ _**θ**_ ( _**x**_ ), The first proposal considers a plug-in estimate of the entropy term with a KDE
estimate of _p_ _**θ**_ ( _**x**_ ), which is consistent with the KDE estimator but not necessary with the other
two (as they use kernels when representing log _p_ _**θ**_ ( _**x**_ ) or _∇_ _**x**_ log _p_ _**θ**_ ( _**x**_ )). The second one uses a

_K_

proxy of the entropy loss _−_ H [˜] [ _p_ ] _≈_ _K_ [1] � _k_ =1 _[∇]_ _**[x]**_ _[k]_ [ log] _[ p]_ _**[θ]**_ [(] _**[x]**_ _[k]_ [)] [T] _**[x]**_ _[k]_ [ with generated samples] _[ {]_ _**[x]**_ _[k]_ _[}]_ [ and]

_∇_ _**x**_ _k_ log _p_ _**θ**_ ( _**x**_ _[k]_ ) approximated by the gradient estimator in use.


In the experiment, we construct a deconvolutional net for the generator and a convolutional autoencoder for the discriminator. The convolutional encoder consists of 3 convolutional layers with
filter width 3, stride 2, and number of feature maps [32, 64, 64]. These convolutional layers are
followed by two fully connected layers with [512, 64] units. The decoder and the generative net have
a symmetric architecture but with stride convolutions replaced by deconvolutions. ReLU activation
function is used for all layers except the last layer of the generator, which uses sigmoid non-linearity.
The reconstruction loss in use is the squared _ℓ_ 2 norm _|| · ||_ 2 [2] [. The randomness] _[ p]_ [0] [(] _**[z]**_ [)][ is selected as]
uniform distribution in [-1, 1] as suggested in the original paper (Berthelot et al., 2017). The minibatch size is set to _K_ = 100. Learning rate is initialised at 0.0002 and decayed by 0.9 every 10
epochs, which is tuned on the KDE model. The selected _γ_ and _α_ values are: for KDE estimator
approach _γ_ = 0 _._ 3 _, αγ_ = 0 _._ 05, for score matching estimator approach _γ_ = 0 _._ 3 _, αγ_ = 0 _._ 1, and for
Stein approach _γ_ = 0 _._ 5 and _αγ_ = 0 _._ 3. The presented results use the KDE plug-in estimator for the
entropy estimates (used to tune _β_ ) for the KDE and score matching approaches. Initial experiments
found that for the Stein approach, using the KDE entropy estimator works slightly worse than the
proxy loss, thus we report results using the proxy loss. An advantage of using the proxy loss is
that it directly relates to the approximate gradient. Furthermore we empirically observe that the
performance of the Stein approach is much more robust to the selection of _γ_ and _α_ when compared
to the other two methods.


19


