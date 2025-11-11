## **Fast and Faster Convergence of SGD for Over-Parameterized Models** **(and an Accelerated Perceptron)**

**Sharan Vaswani** [1] **Francis Bach** [2] **Mark Schmidt** [1]

1 University of British Columbia 2 INRIA, ENS, PSL Research University



**Abstract**


Modern machine learning focuses on highly
expressive models that are able to fit or _inter-_
_polate_ the data completely, resulting in zero
training loss. For such models, we show that
the stochastic gradients of common loss functions satisfy a _strong growth condition_ . Under this condition, we prove that constant
step-size stochastic gradient descent (SGD)
with Nesterov acceleration matches the con
vergence rate of the deterministic accelerated
method for both convex and strongly-convex
functions. We also show that this condition

implies that SGD can find a first-order stationary point as efficiently as full gradient descent in non-convex settings. Under interpolation, we further show that all smooth loss
functions with a finite-sum structure satisfy a
_weaker growth condition_ . Given this weaker
condition, we prove that SGD with a constant step-size attains the deterministic convergence rate in both the strongly-convex and
convex settings. Under additional assumptions, the above results enable us to prove
an _O_ (1 _/k_ [2] ) mistake bound for _k_ iterations of
a stochastic perceptron algorithm using the
squared-hinge loss. Finally, we validate our
theoretical findings with experiments on synthetic and real datasets.


**1** **Introduction**


Modern machine learning models are typically trained
with iterative stochastic first-order methods [9, 41, 16,
31, 14, 8]. Stochastic gradient descent (SGD) and
related methods such as Adagrad [9] or Adam [16]


Proceedings of the 22 [nd] International Conference on Artificial Intelligence and Statistics (AISTATS) 2019, Naha,
Okinawa, Japan. PMLR: Volume 89. Copyright 2019 by
the author(s).



compute the gradient with respect to one or a minibatch of training examples in each iteration and take
a descent step using this gradient. Since these methods use only a small part of the data in each iteration, they are the preferred way for training models
on large datasets. However, in order to converge to
the solution, these methods require the step-size to
decay to zero in terms of the number of iterations.
This implies that the gradient descent procedure takes
smaller steps as the training progresses. Consequently,
these methods result in slow sub-linear rates of con
vergence. Specifically, if _k_ is the number of iterations,
then SGD-like methods achieve a convergence rate of
_O_ (1 _/k_ ) and _O_ (1 _/√k_ ) for strongly-convex and convex

functions respectively [23]. In practice, these methods
are augmented with some form of momentum or acceleration [27, 25] that results in faster empirical convergence [37]. Recently, there has been some theoretical
analysis for the use of such acceleration in the stochastic setting [7]. Other related work includes algorithms
specifically designed to achieve an accelerated rate of
convergence in the stochastic setting [1, 19, 10].


Another recent trend in the literature has been to
use variance-reduction techniques [31, 14, 8] that exploit the finite-sum structure of the loss function in
machine-learning applications. These methods do not
require the step-size to decay to zero and are able to
achieve the optimal rate of convergence. However,
they require additional bookkeeping [31, 8] or need
to compute the full gradient periodically [14], both of
which are difficult in the context of training complex
models on large datasets.


In this paper, we take further advantage of the optimization properties specific to modern machine learning models. In particular, we make use of the fact
that models such as non-parametric regression or overparameterized deep neural networks are expressive
enough to fit or _interpolate_ the training dataset completely [42, 22]. For an SGD-like algorithm, this implies that the gradient with respect to each training example converges to zero at the optimal solution. This
property of interpolation is also true for boosting [30]


**Fast and Faster Convergence of SGD for Over-Parameterized Models**



and for simple linear classifiers on separable data. For
example, the perceptron algorithm [29] was first shown
to converge to the optimal solution under a linear separability assumption on the data [26]. This assumption implies that the linear perceptron is able to fit the
complete dataset without errors.


There has been some related work that takes advan
tage of the interpolation property in order to obtain
faster rates of convergence for SGD [32, 22, 5]. Specifically, Schmidt and Le Roux [32] assume a _strong_
_growth condition_ on the stochastic gradients. This
condition relates the _ℓ_ 2 norms of the stochastic gradients to that of the full gradient. Under this assumption, they prove that constant step-size SGD can attain the same convergence rates as full gradient descent
in both the strongly-convex and convex cases. Other
related work has used the strong growth condition to
prove convergence rates for incremental gradient methods [34, 38]. Ma et al. [22] show that under weaker
conditions, SGD with constant step-size results in linear convergence for strongly-convex functions. They
also investigate the effect of batch-size on the convergence and theoretically justify the _linear-scaling rule_
used for training deep learning models in practice [12].
Recently, Cevher and V˜u showed the linear convergence of proximal stochastic gradient descent under a
weaker growth condition for restricted strongly convex functions [5]. They also analyse the effect of an
additive error term on the convergence rate.


In contrast to the above mentioned work, we first show
that the strong growth condition (SGC) [32] implies
that SGD with a constant step-size and Nesterov momentum [25] achieves the accelerated convergence rate
of the deterministic setting for both strongly-convex
and convex functions (Section 3). Our result gives
some theoretical justification behind the empirical success of using Nesterov acceleration with SGD [37]. Further, in Section 4 we consider non-convex objectives
and prove under the SGC that constant step-size SGD
is able to find a first-order stationary point as efficiently as deterministic gradient descent. To the best
of our knowledge, this is the first work to study accelerated and non-convex rates under the SGC.


After the release of the first version of this work, Liu
et al. [20] also considered minimizing strongly-convex
loss functions using a variant of Nesterov acceleration
assuming interpolation. In this setting they show accelerated rates for the squared loss, and under additional assumptions give accelerated rates for general
strongly-convex functions. However, it is not clear if
these additional assumptions are satisfied by common
loss functions. Indeed, these additional assumptions
imply the SGC (see Section 6.1) so the result presented
in Section 3 is more widely-applicable. Similarly, the



work of Jain et al. [13] uses tail-averaging to obtain
accelerated rates but only for the special case of the
squared loss under interpolation. Furthermore, unlike
these works, we show accelerated rates for convex functions (that are not strongly-convex) under the SGC.


Another work appearing after the release of the initial
version of this work is Bassily et al. [3], who considered minimizing non-convex functions satisfying the
Polyak-Lojasiewicz [28] (PL) inequality (a generalization of strong-convexity) under the interpolation condition. This is a much stronger assumption than we
make in Section 4 to analyze non-convex functions
(since it implies all local optima are global optima),
but under this condition they show that SGD can
achieve a linear convergence rate. However, the stepsize needed to achieve this rate is proportional to the
PL constant which is typically extremely small (and is
often is both unknown and difficult to estimate). By
exploiting the stronger SGC, in this version of the paper we have added a result under the PL inequality
(Section 4) that achieves a faster rate by using a stepsize that depends only on the smoothness properties
of the functions.


In this work, we also relax the strong growth condition
to a more practical _weak growth condition_ (WGC). In
Section 5, we prove that the weak growth condition
is sufficient to obtain the optimal convergence of constant step-size SGD for smooth strongly-convex and
convex functions. To demonstrate the applicability of
our growth conditions in practice, we first show that
for models interpolating the data, the WGC is satisfied for all smooth and convex loss functions with a
finite-sum structure (Section 6.1). Furthermore, we
prove that functions satisfying the WGC and the PL
condition also satisfy the SGC. Under additional assumptions, we show that it is also satisfied for the
squared-hinge loss. This result enables us to prove
an _O_ (1 _/k_ [2] ) mistake bound for _k_ iterations of an accelerated stochastic perceptron algorithm using the
squared-hinge loss (Section 7). Finally, in Section 8,
we evaluate our claims with experiments on synthetic
and real datasets.


**2** **Background**


In this section, we give the required background and
set up the necessary notation. Our aim is to minimize
a differentiable function _f_ ( _w_ ). Depending on the context, this function can be strongly-convex, convex or
non-convex. We assume that we have access to noisy
gradients for the function _f_ and use stochastic gradient descent (SGD) for _k_ iterations in order to minimize
it. The SGD update rule in iteration _k_ can be written
as: _w_ _k_ +1 = _w_ _k_ _−η_ _k_ _∇f_ ( _w_ _k_ _, z_ _k_ ). Here, _w_ _k_ +1 and _w_ _k_ are


**Sharan Vaswani, Francis Bach, Mark Schmidt**



the SGD iterates, _z_ _k_ is the gradient noise and _η_ _k_ is the
step-size at iteration _k_ . We assume that the gradients
_∇f_ ( _w, z_ ) are unbiased, implying that for all _w_ and _z_
that E _z_ [ _∇f_ ( _w, z_ )] = _∇f_ ( _w_ ).


While most of our results apply for general SGD
methods, a subset of our results rely on the function _f_ ( _w_ ) having a finite-sum structure meaning that
1 _n_
_f_ ( _w_ ) = _n_ � _i_ =1 _[f]_ _[i]_ [(] _[w]_ [).] In the context of supervised
machine learning, given a training dataset of _n_ points,
the term _f_ _i_ ( _w_ ) corresponds to the loss function for the
point ( _x_ _i_ _, y_ _i_ ) when the model parameters are equal
to _w_ . Here _x_ _i_ and _y_ _i_ refer to the feature vector
and label for point _i_ respectively. Common choices
of the loss function include the squared loss where
_f_ _i_ ( _w_ ) = [1] 2 [(] _[w]_ [T] _[x]_ _[i]_ _[ −]_ _[y]_ _[i]_ [)] [2] [, the hinge loss where] _[ f]_ _[i]_ [(] _[w]_ [) =]

max(0 _,_ 1 _−_ _y_ _i_ _w_ [T] _x_ _i_ ) or the squared-hinge loss where
_f_ _i_ ( _w_ ) = max (0 _,_ 1 _−_ _y_ _i_ _w_ [T] _x_ _i_ ) [2] . The finite sum setting
includes both simple models such as logistic regression
or least squares and more complex models like nonparametric regression and deep neural networks.


In the finite-sum setting, SGD consists of choosing
a point and its corresponding loss function (typically
uniformly) at random and evaluating the gradient with
respect to that function. It then performs a gradient
descent step: _w_ _k_ +1 = _w_ _k_ _−_ _η_ _k_ _∇f_ _k_ ( _w_ _k_ ) where _f_ _k_ ( _·_ ) is
the random loss function selected at iteration _k_ . The

unbiasedness property is automatically satisfied in this
case, i.e. E _i_ [ _∇f_ _i_ ( _w_ )] = _∇f_ ( _w_ ) for all _w_ . Note that in
this case, the random selection of points for computing
the gradient is the source of the noise _z_ _k_ . In order to
converge to the optimum, SGD requires the step-size
_η_ _k_ to decrease with _k_ ; specifically at a rate of ~~_√_~~ 1 _k_ [for]

convex functions and at a _k_ [1] [rate for strongly-convex]

functions. Decreasing the step-size with _k_ results in
sub-linear rates of convergence for SGD.


In order to derive convergence rates, we need to make
additional assumptions about the function _f_ [23]. Beyond differentiability, our results assume that the function _f_ ( _·_ ) satisfies some or all of the following common
assumptions. For all points _w_, _v_ and for constants _f_ _[∗]_,
_µ_, and _L_ ;


_f_ ( _w_ ) _≥_ _f_ _[∗]_ (Bounded below)


_f_ ( _v_ ) _≥_ _f_ ( _w_ ) + _⟨∇f_ ( _w_ ) _, v −_ _w⟩_ (Convexity)

_f_ ( _v_ ) _≥_ _f_ ( _w_ ) + _⟨∇f_ ( _w_ ) _, v −_ _w⟩_ + _[µ]_

2 _[∥][v][ −]_ _[w][∥]_ [2]

( _µ_ Strong-convexity)

_f_ ( _v_ ) _≤_ _f_ ( _w_ ) + _⟨∇f_ ( _w_ ) _, v −_ _w⟩_ + _[L]_

2 _[∥][v][ −]_ _[w][∥]_ [2]

( _L_ Smoothness)


Note that some of our results in Section 6 rely on the
finite-sum structure and we explicitly state when we
need this additional assumption.



In this paper, we consider the case where the model
is able to _interpolate_ or fit the labelled training data
completely. This is true for expressive models such as
non-parametric regression and over-parametrized deep
neural networks. For common loss functions that are

lower-bounded by zero, interpolating the data results
in zero training loss. Interpolation also implies that
the gradient with respect to each point converges to
zero at the optimum. Formally, in the finite-sum setting, if the function _f_ ( _·_ ) is minimized at _w_ _[∗]_, i.e., if
_∇f_ ( _w_ _[∗]_ ) = 0, then for all functions _f_ _i_ ( _·_ ), _∇f_ _i_ ( _w_ _[∗]_ ) = 0.


The _strong growth condition_ (SGC) used connects the
rates at which the stochastic gradients shrink relative
to the full gradient. Formally, for any point _w_ and the
noise random variable _z_, the function _f_ satisfies the
strong growth condition with constant _ρ_ if,


E _z_ _∥∇f_ ( _w, z_ ) _∥_ [2] _≤_ _ρ ∥∇f_ ( _w_ ) _∥_ [2] _._ (1)


Equivalently, in the finite-sum setting,


E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_ _ρ ∥∇f_ ( _w_ ) _∥_ [2] _._ (2)


For this inequality to hold, if _∇f_ ( _w_ ) = 0, then
_∇f_ _i_ ( _w_ ) = 0 for all _i_ . Thus, functions satisfying the
SGC necessarily satisfy the above interpolation property. Schmidt and Le Roux’s work [32] derives optimal
convergence rates for constant step-size SGD under the
above condition for both convex and strongly-convex
functions. In the next section, we show that the SGC
implies the accelerated rate of convergence for constant
step-size SGD with Nesterov momentum.


**3** **SGD with Nesterov acceleration**

**under the SGC**


We first describe constant step-size SGD with Nesterov acceleration. The algorithm consists of three
sequences ( _w_ _k_ _, ζ_ _k_ _, v_ _k_ ) updated in each iteration [24].
Specifically, it consists of the following update rules:


_w_ _k_ +1 = _ζ_ _k_ _−_ _η∇f_ ( _ζ_ _k_ _, z_ _k_ ) (3)


_ζ_ _k_ = _α_ _k_ _v_ _k_ + (1 _−_ _α_ _k_ ) _w_ _k_ (4)


_v_ _k_ +1 = _β_ _k_ _v_ _k_ + (1 _−_ _β_ _k_ ) _ζ_ _k_ _−_ _γ_ _k_ _η∇f_ ( _ζ_ _k_ _, z_ _k_ ) _._ (5)


Here, _η_ is the constant step-size for the SGD step and
_α_ _k_, _β_ _k_, _γ_ _k_ are tunable parameters to be set according
to the properties of _f_ .


In order to derive a convergence rate for the above algorithm under the SGC, we first observe that a form
of the SGC is satisfied in the case of coordinate descent [39]. In this case, we choose a coordinate (typically at random) and perform a gradient descent step
with respect to that coordinate. The notion of a coordinate in this case is analogous to that of an individual


**Fast and Faster Convergence of SGD for Over-Parameterized Models**



loss function in the finite sum case. For coordinate descent, a zero gradient at the optimal solution implies
that the partial derivative with respect to each coordinate is also equal to zero. This is analogous to the SGC
in the finite-sum case, although we note the results in
this section do not require the finite-sum assumption.


We use this analogy formally in order to extend the
proof of Nesterov’s accelerated coordinate descent [24]
to derive convergence rates for the above algorithm
when using the SGC. This enables us to prove the
following theorems (with proofs in Appendices B.1.1
and B.1.3) in both the strongly-convex and convex settings.


**Theorem 1** (Strongly convex) **.** _Under L-smoothness_
_and µ strong-convexity, if f satisfies the SGC with con-_
_stant ρ, then SGD with Nesterov acceleration with the_
_following choice of parameters,_


1 ~~_µη_~~
_γ_ _k_ = _;_ _β_ _k_ = 1 _−_
~~_√µηρ_~~ � _ρ_



rate of convergence up to a _ρ_ [2] factor for both stronglyconvex and convex functions.


In Appendix A, we consider the SGC with an extra additive error term, resulting in the following condition:
E _z_ _∥∇f_ ( _w, z_ ) _∥_ [2] _≤_ _ρ ∥∇f_ ( _w_ ) _∥_ [2] + _σ_ [2] . We analyse the
rate of convergence of the above algorithm under this
modified condition and obtain a similar dependence on
_σ_ as in Cohen et al. [7].


**4** **SGD for non-convex functions**

**satisfying the SGC**


In this section, we show that the SGC results in an
improvement over the _O_ 1 _/√k_ rate for SGD in the
� �

non-convex setting [11]. In particular, we show that
under the strong growth condition, constant step-size
SGD is able to find a first-order stationary point as efficiently as deterministic gradient descent. We prove the
following theorem (with the proof in Appendix B.2),


**Theorem 3** (Non-Convex) **.** _Under L-smoothness, if f_
_satisfies SGC with constant ρ, then SGD with a con-_
1
_stant step-size η_ = _ρL_ _[attains the following conver-]_
_gence rate:_



_√_ ~~_µ_~~
_b_ _k_ +1 =
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_



1
_a_ _k_ +1 =
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_




[ _f_ ( _w_ 0 ) _−_ _f_ _[∗]_ ] _._
�



_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_
_α_ _k_ = _;_ _η_ = [1]
_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_ [ +] _[ a]_ _k_ [2] _ρL_ _[,]_



2 _ρL_
_i_ =0 _,_ min 1 _,...k−_ 1 [E] � _∥∇f_ ( _w_ _i_ ) _∥_ [2] [�] _≤_ � _k_



_results in the following convergence rate:_


E _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _[∗]_ )



The above theorem shows that under the SGC, SGD
with a constant step-size can attain the optimal
_O_ (1 _/k_ ) rate for non-convex functions. To the best
of our knowledge, this is the first result for nonconvex functions under interpolation-like conditions.
Under these conditions, constant step-size SGD has a
better convergence rate than algorithms which have
recently been proposed to improve on SGD [2, 4].
Note that the above theorem applies to neural networks with a sigmoid activation function under the assumption that the strong-growth condition is satisfied.
Hence, our results also provide some theoretical justification for the effectiveness of SGD for non-convex
over-parameterized models like deep neural networks.


Under the additional assumption that the function satisfies the Polyak- Lojasiewicz condition [28] (a generalization of strong-convexity), we show that SGD can
obtain linear convergence. Specifically, we prove the
following theorem (with the proof in Appendix B.3),


**Theorem** **4** (Non-Convex + PL) **.** _Under_ _L-_
_smoothness, if f satisfies SGC with constant ρ and the_
_Polyak- Lojasiewicz inequality with constant µ, then_
1
_SGD with a constant step-size η_ = _ρL_ _[attains the fol-]_



~~_µ_~~
_≤_ 1 _−_
� � _ρ_ [2] _L_



~~_µ_~~
_≤_ 1 _−_
� � _ρ_ [2]



_k_
_f_ ( _w_ 0 ) _−_ _f_ ( _w_ _[∗]_ ) + _[µ]_ _._
� � 2 _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�]



**Theorem 2** (Convex) **.** _Under L-smoothness and con-_
_vexity, if f satisfies the SGC with constant ρ, then_
_SGD with Nesterov acceleration with the following_
_choice of parameters,_



1 1
_ρ_ [+] � _ρ_ [2] [+ 4] _[γ]_ _k_ [2] _−_ 1

_γ_ _k_ =

2

_a_ _k_ +1 = _γ_ _k_ _√_ ~~_ηρ_~~



_γ_ _k_ =



_ρ_ 1 [+] �



_γ_ _k_ _η_
_α_ _k_ = _;_ _η_ = [1]
_γ_ _k_ _η_ + _a_ [2] _k_ _ρL_ _[,]_



_results in the following convergence rate:_


E _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _[∗]_ ) _≤_ [2] _[ρ]_ [2] _[L]_ _∥w_ 0 _−_ _w_ _[∗]_ _∥_ [2] _._

_k_ [2]


The above theorems show that constant step-size SGD
with Nesterov momentum achieves the accelerated


_lowing convergence rate:_



**Sharan Vaswani, Francis Bach, Mark Schmidt**


**5.2** **SGD under the weak growth condition**



E [ _f_ ( _w_ _k_ +1 ) _−_ _f_ _[∗]_ ] _≤_ 1 _−_ _[µ]_
� _ρL_



_k_

[ _f_ ( _w_ 0 ) _−_ _f_ _[∗]_ ] _._
�



Note that the PL condition or a related notion of
restricted strong-convexity (RSI) [15] is satisfied by
numerous non-convex optimization problems of interest. These include neural networks [18, 17, 35], matrix
completion [36] and phase retrieval [6]. Under the additional SGC assumption, the above theorem implies
fast rates of convergence for SGD on these problems.
In contrast, Bassily et al. [3] do not assume the SGC

_k_

and achieve a rate of 1 _−_ _L_ _[µ]_ [2][2] using a much smaller
� �

step-size _η_ = _Lµ_ [2] [.]


**5** **Weak growth condition**


In this section, we relax the strong growth condition
to a more practical condition which we refer to as the
_weak growth condition_ (WGC). Formally, if the function _f_ ( _·_ ) is _L_ -smooth and has a minima at _w_ _[∗]_, then it
satisfies the WGC with constant _ρ_, if for all points _w_
and noise random variable _z_,


E _z_ _∥∇f_ ( _w, z_ ) _∥_ [2] _≤_ 2 _ρL_ [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )] _._ (6)


Equivalently, in the finite-sum setting,


E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_ 2 _ρL_ [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )] _._ (7)


In the above condition, notice that if _w_ = _w_ _[∗]_, then
_∇f_ _i_ ( _w_ _[∗]_ ) = 0 for all points _i_ . Thus, the WGC implies
the interpolation property explained in Section 2.


**5.1** **Relation between WGC and SGC**


In this section, we relate the two growth conditions.
We first prove that SGC implies WGC with the same _ρ_
without any additional assumptions, formally showing
that the WGC is indeed weaker than the corresponding
SGC. For the converse, a function satisfying the WGC
satisfies the SGC with a worse constant if it also satisfies the Polyak- Lojasiewicz (PL) inequality [28]. The
above relations are captured by the following proposition, proved in Appendix B.6


**Proposition 1.** _If f_ ( _·_ ) _is L-smooth, satisfies the_
_WGC with constant ρ and the PL inequality with con-_
_stant µ, then it satisfies the SGC with constant_ _[ρ]_ _µ_ _[L]_ _[.]_


_Conversely, if f_ ( _·_ ) _is L-smooth, convex and satisfies_
_the SGC with constant ρ, then it also satisfies the_
_WGC with the same constant ρ._



Using the WGC, we obtain the following convergence
rates for SGD with a constant step-size.


**Theorem 5** (Strongly-convex) **.** _Under L-smoothness_
_and µ strong-convexity, if f satisfies the WGC with_
1
_constant ρ, then SGD with a constant step-size η_ = _ρL_
_achieves the following rate:_



**Theorem 6** (Convex) **.** _Under L-smoothness and con-_
_vexity, if f satisfies the WGC with constant ρ, then_
1
_SGD with a constant step-size η_ = 4 _ρL_ _[and iterate]_
_averaging achieves the following rate:_


[(][1 +] _[ρ]_ [)] _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]
E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ [4] _[L]_ _._

_k_


_Here,_ ¯ _w_ _k_ = [[] � _ki_ =1 _k_ _[w]_ _[i]_ []] _is the averaged iterate after k_

_iterations._


The proofs for Theorems 5 and 6 are deferred to Appendices B.4 and B.5 respectively. In these cases, the
WGC is sufficient to show that constant step-size SGD
can attain the deterministic rates up to a factor of _ρ_ .
Since this condition is weaker than the corresponding strong growth condition, our results subsume the
SGC results [32]. Note that an alternative way to obtain the result in Theorem 5 would be to observe that
the WGC and strong convexity imply the SGC (with
a constant _[ρ]_ _µ_ _[L]_ [) (Proposition 1) and then use the result]

by Schmidt et al. [32]. This would result in an additional dependence on _ρLµ_ [which is worse than the rate]
in Theorem 5.


In the next section, we characterize the functions satisfying the growth conditions in practice.


**6** **Growth conditions in practice**


In this section, we give examples of functions that satisfy the weak and strong growth conditions. In Section 6.1, we first show that for models interpolating
the data, the WGC is satisfied by all smooth functions with a finite-sum structure. In section 6.2, we
show that the SGC is satisfied by the squared-hinge
loss under additional assumptions.


**6.1** **Functions satisfying WGC**


To characterize the functions satisfying the WGC, we
first prove the following proposition (with the proof in
Appendix B.7):



E _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] _≤_ 1 _−_ _[µ]_
� _ρL_



_k_
_∥w_ 0 _−_ _w_ _[∗]_ _∥_ [2] _._
�


**Fast and Faster Convergence of SGD for Over-Parameterized Models**



**Proposition 2.** _If the function f_ ( _·_ ) _is convex and has_
_a finite-sum structure for a model that interpolates the_
_data and L_ max _is the maximum smoothness constant_
_amongst the functions f_ _i_ ( _·_ ) _, then for all w,_


E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_ 2 _L_ _max_ [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )] _._ (8)


Comparing the above equation to Equation 7, we see
that any smooth finite-sum problem under interpolation satisfies the WGC with _ρ_ = _[L]_ _[max]_ _L_ . The WGC is

thus satisfied by common loss functions such as the
squared and squared-hinge loss. For these loss functions, if _L_ _i_ = _L_ for all _i_, then Theorem 5 implies
that SGD with _η_ = _L_ [1] [results in linear convergence for]

strongly-convex functions. This matches the recently
proved result of Ma et al. [22], whereas Theorem 6
allows us to generalize their result beyond stronglyconvex functions.


**6.2** **Functions satisfying SGC**


We now show that under additional assumptions on
the data, the squared-hinge loss also satisfies the SGC.
We first assume that the data is linearly separable
with a margin equal to _τ_, implying that for all _x_,
_τ_ = max _|w|_ =1 inf _x∈_ S _w_ _[⊤]_ _x_ . Here, S is the support of
the distribution of the features _x_ . Note that the above

assumption implies the existence of a classifier _w_ _[∗]_ such
that _||w_ _[∗]_ _||_ = _τ_ [1] [. In addition to this, we assume that]

the features have a finite support, meaning that the
set S is finite and has a cardinality equal to _c_ . Under
these assumptions, we prove the following lemma in
Appendix B.8,


**Lemma 1.** _For linearly separable data with margin τ_
_and a finite support of size c, the squared-hinge loss_

_c_
_satisfies the SGC with the constant ρ_ = _τ_ [2] _[.]_


In the next section, we use the above lemma to prove a
mistake bound for the perceptron algorithm using the
squared-hinge loss.


**7** **Implication for Faster Perceptron**


In this section, we use the strong growth property of
the squared-hinge function in order to prove a bound
on the number of mistakes made by the perceptron
algorithm [29] using a squared-hinge loss. The perceptron algorithm is used for training a linear classifier for binary classification and is guaranteed to
converge for linearly separable data [26]. It can be
considered as stochastic gradient descent on the loss
_f_ _i_ ( _w_ ) = max _{_ 0 _, y_ _i_ _x_ _[⊤]_ _i_ _[w][}]_ [.]


The common way to characterize the performance of
a perceptron is by bounding the number of mistakes



(in the binary classification setting) after _k_ iterations
of the algorithm. In other words, we care about the
quantity P( _yx_ _[⊤]_ _w_ _k_ ⩾ 0). Assuming linear separability
of the data and that _||x||_ = 1 for all points ( _x, y_ ), the
perceptron achieves a mistake bound of _O_ � _τ_ 1 [2] � [26].



(in the binary classification setting) after _k_ iterations
of the algorithm. In other words, we care about the
quantity P( _yx_ _[⊤]_ _w_ _k_ ⩾ 0). Assuming linear separability
of the data and that _||x||_ = 1 for all points ( _x, y_ ), the
perceptron achieves a mistake bound of _O_ � _τ_ 1 [2] � [26].


In this paper, we consider a modified perceptron algorithm using the squared-hinge function as the loss.
Note that since we assume the data to be linearly separable, a linear classifier is able to fit all the training
data. Since the squared-hinge loss function is smooth,
the conditions of Proposition 2 are satisfied, which
implies that it satisfies the WGC with _ρ_ = _L_ _max_ _L_ .

Also observe that since we assume that _||x||_ = 1,
_L_ _max_ = _L_ = 1. Using these facts with Theorem 6 and
assuming that we start the optimization with _w_ 0 = **0**,
we obtain the following convergence rate using SGD
with _η_ = 1 _/_ 4,



In this paper, we consider a modified perceptron algorithm using the squared-hinge function as the loss.
Note that since we assume the data to be linearly separable, a linear classifier is able to fit all the training
data. Since the squared-hinge loss function is smooth,
the conditions of Proposition 2 are satisfied, which
implies that it satisfies the WGC with _ρ_ = _L_ _max_ .



8
E[ _f_ ( _w_ _k_ +1 )] _≤_
_τ_ [2] _k_ _[.]_


To see this, recall that _||w_ _[∗]_ _||_ = _τ_ [1] [and the loss is equal]

to zero at the optima, implying that _f_ ( _w_ _[∗]_ ) = 0.


The above result gives us a bound on the training
loss. We use the following lemma (proved using the
Markov inequality in Appendix B.9) to relate the mistake bound to the training loss.

**Lemma 2.** _If f_ ( _w, x, y_ ) _represents the loss on the_
_point_ ( _x, y_ ) _, then_


P( _yx_ _[⊤]_ _w_ ⩽ 0) ⩽ E _x,y_ _f_ ( _w, x, y_ ) _._


Combining the above results, we obtain a mistake
bound of _O_ � _τ_ 1 [2] _k_ � when using the squared-hinge loss
on linearly separable data. We thus recover the standard results for the stochastic perceptron.


Note that for a finite amount of data (when the expectation is with respect to a discrete distribution),
if we use batch accelerated gradient descent (which is
not one of the stochastic gradient algorithms studied
in this paper, and for which no growth condition is
needed), we obtain a mistake bound that decreases as
1 _/k_ [2] . This improves on existing mistake bounds that
scale as 1 _/k_ [33, 40]. Note that both sets of algorithms
have the same dependence on the margin _τ_, but this
deterministic accelerated method would require evaluating _n_ gradients on each iteration.


From Lemma B.9, we know that the squared-hinge

_c_
loss satisfies the SGC with _ρ_ = _τ_ [2] [. Under the same]
conditions as above, this lemma along with the result
of Theorem 2 gives us the following bound:

E _f_ ( _w_ _k_ +1 ) _≤_ _τ_ [2] [6] _[c]_ _k_ [2][2] _[.]_


**Sharan Vaswani, Francis Bach, Mark Schmidt**



Using the result from Lemma 2, this results in a mistake bound of the order _O_ � _τ_ [6] 1 _k_ [2] � while only requiring one gradient per iteration. Hence, the use of acceleration leads to an improved novel dependence of
_O_ (1 _/k_ [2] ), but requires the additional assumptions of
Lemma B.9 and has a worse dependence on the margin _τ_ .


**8** **Experiments**


In this section, we empirically validate our theoretical results. For the first set of experiments (Figures 1(a)-1(d)), we generate a synthetic binary classification dataset with _n_ = 8000 and the dimension
_d_ = 100. We ensure that the data is linearly separable with a margin _τ_, thus satisfying the interpolation property for training a linear classifier. We
seek to minimize the finite-sum squared-hinge loss,
_f_ ( _w_ ) = [�] _[n]_ _i_ =1 [max (0] _[,]_ [ 1] _[ −]_ _[y]_ _[i]_ _[x]_ _i_ [T] _[w]_ [)] [2] [.] In Figure 1, we
vary the margin _τ_ and plot the logarithm of the loss
with the number of effective passes (one pass is equal
to _n_ iterations of SGD) over the data. In all of our
experiments, we estimate the value of the smoothness
parameter _L_ as the maximum eigenvalue of the Gram
matrix _X_ _[T]_ _X_ .


We evaluate the performance of constant step-size
SGD with and without acceleration. Since the
squared-hinge loss satisfies the WGC with _ρ_ = _[L]_ _[max]_ _L_

(Proposition 2), we use SGD with a constant step-size
_η_ = 1 _/L_ _max_ [1] (denoted as SGD in the plots). For using Nesterov acceleration, we experimented with the
dependence of the margin _τ_ on the constant _ρ_ in the
SGC. We found that setting _ρ_ = 1 _/τ_ results in consistently stable but fast convergence across different
choices of _τ_ . We thus use a step-size _η_ = _τ/L_ and set
the tunable parameters in the update Equations 3-5
as specified by Theorem 2. We denote this variant of
accelerated SGD as Acc-SGD in the subsequent plots.
In Appendix C, we propose a line-search heuristic to
dynamically estimate the value of _ρ_ .


In each of the Figures 1(a)-1(d), we make the following
observations: (i) SGD results in reasonably slow convergence. This observation is in line with other SGD
methods using 1 _/L_ as the step-size [31]. (ii) Acc-SGD
with _η_ = _τ/L_ is consistently stable and as suggested
by the theory, it results in faster convergence as compared to using SGD. (iii) For larger values of _τ_ (Figures 1(a)- 1(b)), the training loss becomes equal to
zero, verifying the interpolation property.


The next set of experiments (Figure 2) considers bi

1 Note that using _η_ = 1 _/L_ _max_ lead to consistently better
results as compared to using _η_ = 1 _/_ 4 _L_ _max_ as suggested by
Theorem 6.



nary classification on the CovType [2] and Protein [3]

datasets. For this, we train a linear classifer using the radial basis (non-parametric) features. Nonparametric regression models of this form are capable
of interpolating the data [22] and thus satisfy our assumptions. We subsample _n_ = 8000 random points
from the datasets and use the squared-hinge loss as
above. In this case, we perform a grid-search to obtain
a good estimate of _ρ_ . We choose _ρ_ = 1 for the CovType dataset and equal to 0 _._ 1 for the Protein dataset.


From Figures 2(a) and 2(b), we make the following
observations: (i) In Figure 2(a), both variants have
similar performance. (ii) In Figure 2(b), the Acc-SGD
leads to considerably faster convergence as compared
to SGD. These experiments show that in cases where
the interpolation property is satisfied, both SGD and
accelerated SGD with a constant step-size can result
in good empirical performance.


**9** **Conclusion**


In this paper, we showed that under interpolation,
the stochastic gradients of common loss functions satisfy specific growth conditions. Under these conditions, we proved that it is possible for constant stepsize SGD (with and without Nesterov acceleration) to
achieve the convergence rates of the corresponding deterministic settings. These are the first results achieving optimal rates in the accelerated and non-convex
settings under interpolation-like conditions. We used
these results to demonstrate the fast convergence of
the stochastic perceptron algorithm employing the
squared-hinge loss. We showed that both SGD and
accelerated SGD with a constant step-size can lead
to good empirical performance when the interpolation
property is satisfied. As opposed to determining the
step-size and the schedule for annealing it for current
SGD-like methods, our results imply that under interpolation, we only need to automatically determine
the constant step-size for SGD. In the future, we hope
to develop line-search techniques for automatically determining this step-size for both the accelerated and
non-accelerated variants.


2 `[http://osmot.cs.cornell.edu/kddcup](http://osmot.cs.cornell.edu/kddcup)`
3 `[http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)`
```
datasets

```

**Fast and Faster Convergence of SGD for Over-Parameterized Models**


(a) _τ_ = 0 _._ 1 (b) _τ_ = 0 _._ 05


(c) _τ_ = 0 _._ 01 (d) _τ_ = 0 _._ 005


Figure 1: Comparison of SGD and variants of accelerated SGD on a synthetic linearly separable dataset with
margin _τ_ . Accelerated SGD with _η_ = _τ/L_ leads to faster convergence as compared to SGD with _η_ = 1 _/L_ .


(a) CovType (b) Protein


Figure 2: Comparison of SGD and accelerated SGD for learning a linear classifier with RBF features on the (a)
CovType and (b) Protein datasets. Accelerated SGD leads to better performance as compared to SGD with
_η_ = 1 _/L_ .



**10** **Acknowledgements**


We acknowledge support from the European Research
Council (grant SEQUOIA 724063) and the CIFAR



program on Learning with Machines and Brains. We
also thank Nicolas Flammarion, Reza Babanezhad and
Adrien Taylor for discussions related to this work. We
also thank Kevin Scaman for discussions and insights


**Sharan Vaswani, Francis Bach, Mark Schmidt**



on using acceleration with multiplicative noise.


**References**


[1] Zeyuan Allen-Zhu. Katyusha: The first direct
acceleration of stochastic gradient methods. In
_Proceedings of the 49th Annual ACM SIGACT_
_Symposium on Theory of Computing_, pages 1200–
1205. ACM, 2017.


[2] Zeyuan Allen-Zhu. Natasha 2: Faster non-convex
optimization than sgd. In _Advances in Neural In-_
_formation Processing Systems_, pages 2680–2691,
2018.


[3] Raef Bassily, Mikhail Belkin, and Siyuan Ma.
On exponential convergence of sgd in non-convex
over-parametrized learning. _arXiv_ _preprint_
_arXiv:1811.02564_, 2018.


[4] Yair Carmon, John C Duchi, Oliver Hinder,
and Aaron Sidford. Convex until proven guilty:
Dimension-free acceleration of gradient descent
on non-convex functions. In _Proceedings of_
_the 34th International Conference on Machine_
_Learning-Volume 70_, pages 654–663. JMLR. org,
2017.


[5] Volkan Cevher and Bang Cˆong V˜u. On the linear convergence of the stochastic gradient method
with constant step-size. _Optimization Letters_,
pages 1–11, 2018.


[6] Yuxin Chen and Emmanuel Candes. Solving
random quadratic systems of equations is nearly
as easy as solving linear systems. In _Advances_
_in Neural Information Processing Systems_, pages
739–747, 2015.


[7] Michael Cohen, Jelena Diakonikolas, and Lorenzo
Orecchia. On acceleration with noise-corrupted
gradients. In _Proceedings of the 35th International_
_Conference on Machine Learning, ICML 2018,_
_Stockholmsm¨assan, Stockholm, Sweden, July 10-_
_15, 2018_, pages 1018–1027, 2018.


[8] Aaron Defazio, Francis Bach, and Simon LacosteJulien. Saga: A fast incremental gradient method
with support for non-strongly convex composite
objectives. In _Advances in neural information_
_processing systems_, pages 1646–1654, 2014.


[9] John Duchi, Elad Hazan, and Yoram Singer.
Adaptive subgradient methods for online learning
and stochastic optimization. _Journal of Machine_
_Learning Research_, 12(Jul):2121–2159, 2011.


[10] Roy Frostig, Rong Ge, Sham Kakade, and Aaron
Sidford. Un-regularizing: approximate proximal



point and faster stochastic algorithms for empirical risk minimization. In _International Con-_
_ference on Machine Learning_, pages 2540–2548,
2015.


[11] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex
stochastic programming. _SIAM Journal on Opti-_
_mization_, 23(4):2341–2368, 2013.


[12] Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter
Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: training imagenet in
1 hour. _arXiv preprint arXiv:1706.02677_, 2017.


[13] Prateek Jain, Sham M. Kakade, Rahul Kidambi,
Praneeth Netrapalli, and Aaron Sidford. Accelerating stochastic gradient descent for least squares
regression. In _Conference On Learning Theory,_
_COLT 2018, Stockholm, Sweden, 6-9 July 2018._,
pages 545–604, 2018.


[14] Rie Johnson and Tong Zhang. Accelerating
stochastic gradient descent using predictive variance reduction. In _Advances in neural informa-_
_tion processing systems_, pages 315–323, 2013.


[15] Hamed Karimi, Julie Nutini, and Mark Schmidt.
Linear convergence of gradient and proximalgradient methods under the polyak-�lojasiewicz
condition. In _Joint European Conference on_
_Machine Learning and Knowledge Discovery in_
_Databases_, pages 795–811. Springer, 2016.


[16] Diederik P Kingma and Jimmy Ba. Adam:
A method for stochastic optimization. _arXiv_
_preprint arXiv:1412.6980_, 2014.


[17] Robert Kleinberg, Yuanzhi Li, and Yang Yuan.
An alternative view: When does sgd escape local
minima? In _International Conference on Machine_
_Learning_, pages 2703–2712, 2018.


[18] Yuanzhi Li and Yang Yuan. Convergence analysis of two-layer neural networks with relu activation. In _Advances in Neural Information Process-_
_ing Systems_, pages 597–607, 2017.


[19] Hongzhou Lin, Julien Mairal, and Zaid Harchaoui. A universal catalyst for first-order optimization. In _Advances in Neural Information_
_Processing Systems_, pages 3384–3392, 2015.


[20] Chaoyue Liu and Mikhail Belkin. Mass: an accelerated stochastic method for over-parametrized
learning. _arXiv preprint arXiv:1810.13395_, 2018.


**Fast and Faster Convergence of SGD for Over-Parameterized Models**




[21] Jun Liu, Jianhui Chen, and Jieping Ye. Largescale sparse logistic regression. In _Proceedings of_
_the 15th ACM SIGKDD international conference_
_on Knowledge discovery and data mining_, pages
547–556. ACM, 2009.


[22] Siyuan Ma, Raef Bassily, and Mikhail Belkin.
The power of interpolation: Understanding the
effectiveness of SGD in modern over-parametrized
learning. In _Proceedings of the 35th International_
_Conference on Machine Learning, ICML 2018,_
_Stockholmsm¨assan, Stockholm, Sweden, July 10-_
_15, 2018_, pages 3331–3340, 2018.


[23] Arkadi Nemirovski, Anatoli Juditsky, Guanghui
Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. _SIAM Journal on optimization_,
19(4):1574–1609, 2009.


[24] Yu Nesterov. Efficiency of coordinate descent
methods on huge-scale optimization problems.
_SIAM Journal on Optimization_, 22(2):341–362,
2012.


[25] Yu Nesterov. Gradient methods for minimizing
composite functions. _Mathematical Programming_,
140(1):125–161, 2013.


[26] Albert B Novikoff. On convergence proofs for perceptrons. Technical report, 1963.


[27] Boris T Polyak. Some methods of speeding up the
convergence of iteration methods. _USSR Compu-_
_tational Mathematics and Mathematical Physics_,
4(5):1–17, 1964.


[28] Boris Teodorovich Polyak. Gradient methods for
minimizing functionals. _Zhurnal Vychislitel’noi_
_Matematiki i Matematicheskoi Fiziki_, 3(4):643–
653, 1963.


[29] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. _Psychological review_, 65(6):386,
1958.


[30] Robert E Schapire, Yoav Freund, Peter Bartlett,
Wee Sun Lee, et al. Boosting the margin: A new
explanation for the effectiveness of voting methods. _The annals of statistics_, 26(5):1651–1686,
1998.


[31] Mark Schmidt, Nicolas Le Roux, and Francis
Bach. Minimizing finite sums with the stochastic average gradient. _Mathematical Programming_,
162(1-2):83–112, 2017.




[32] Mark Schmidt and Nicolas Le Roux. Fast
convergence of stochastic gradient descent under a strong growth condition. _arXiv preprint_
_arXiv:1308.6370_, 2013.


[33] Negar Soheili and Javier Pena. A primal–dual
smooth perceptron–von neumann algorithm. In
_Discrete Geometry and Optimization_, pages 303–
320. Springer, 2013.


[34] Mikhail V Solodov. Incremental gradient algorithms with stepsizes bounded away from zero.
_Computational Optimization and Applications_,
11(1):23–35, 1998.


[35] Mahdi Soltanolkotabi, Adel Javanmard, and Jason D Lee. Theoretical insights into the optimization landscape of over-parameterized shallow neural networks. _IEEE Transactions on Information_
_Theory_, 2018.


[36] Ruoyu Sun and Zhi-Quan Luo. Guaranteed
matrix completion via non-convex factorization. _IEEE Transactions on Information Theory_,
62(11):6535–6579, 2016.


[37] Ilya Sutskever, James Martens, George Dahl, and
Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In _Inter-_
_national conference on machine learning_, pages
1139–1147, 2013.


[38] Paul Tseng. An incremental gradient (projection) method with momentum term and
adaptive stepsize rule. _SIAM Journal on Opti-_
_mization_, 8(2):506–531, 1998.


[39] Stephen J Wright. Coordinate descent algorithms. _Mathematical Programming_, 151(1):3–34,
2015.


[40] Adams Wei Yu, Fatma Kilinc-Karzan, and Jaime
Carbonell. Saddle points and accelerated perceptron algorithms. In _International Conference on_
_Machine Learning_, pages 1827–1835, 2014.


[41] Matthew D Zeiler. Adadelta: an adaptive learning rate method. _arXiv preprint arXiv:1212.5701_,
2012.


[42] Chiyuan Zhang, Samy Bengio, Moritz Hardt,
Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. _arXiv preprint arXiv:1611.03530_, 2016.


**Sharan Vaswani, Francis Bach, Mark Schmidt**


**A** **Incorporating additive error for Nesterov acceleration**


For this section, we assume an additive error in the the strong growth condition implying that the following
equation is satisfied for all _w_, _z_ .


E _z_ _∥∇f_ ( _w, z_ ) _∥_ [2] _≤_ _ρ ∥∇f_ ( _w_ ) _∥_ [2] + _σ_ [2]


In this case, we have the counterparts of Theorems 1 and 2 as follows:


**Theorem 7** (Strongly convex) **.** _Under L-smoothness and µ strongly-convexity, if f satisfies SGC with constant_
_ρ and an additive error σ, then SGD with Nesterov acceleration with the following choice of parameters,_


1 ~~_µη_~~
_γ_ _k_ = _;_ _β_ _k_ = 1 _−_
~~_√µηρ_~~ � _ρ_



_√_ ~~_µ_~~
_b_ _k_ +1 =
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_



1
_a_ _k_ +1 =
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_



_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_
_α_ _k_ = _;_ _η_ = [1]
_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_ [ +] _[ a]_ _k_ [2] _ρL_



_results in the following convergence rate:_


~~_µη_~~

[E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ )] _≤_ 1 _−_
� � _ρ_



_k_
_f_ ( _x_ 0 ) _−_ _f_ ( _w_ _[∗]_ ) + _[µ]_ + _[σ]_ [2] _[√]_ ~~_[η]_~~
� � 2 _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�] ~~_√ρµ_~~



**Theorem 8** (Convex) **.** _Under L-smoothness and convexity, if f satisfies SGC with constant ρ and an additive_
_error σ, then SGD with Nesterov acceleration with the following choice of parameters,_



1 1
_ρ_ [+] � _ρ_ [2] [+ 4] _[γ]_ _k_ [2] _−_ 1

_γ_ _k_ =

2

_a_ _k_ +1 = _γ_ _k_ _√_ ~~_ηρ_~~



_γ_ _k_ =



_ρ_ 1 [+] �



_γ_ _k_ _η_
_α_ _k_ = _;_ _η_ = [1]
_γ_ _k_ _η_ + _a_ [2] _k_ _ρL_



_results in the following convergence rate:_




[2] _[ρ]_

_k_ [2] _η_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ kσ]_ _ρ_ [2] _[η]_




[E _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _[∗]_ )] _≤_ [2] _[ρ]_



_ρ_



The above theorems are proved in appendices B.1.1 and B.1.3


**B** **Proofs**


**B.1** **Proofs for SGD with Nesterov Acceleration**


Recall the update equations for SGD with Nesterov acceleration as follows:


_w_ _k_ +1 = _ζ_ _k_ _−_ _η∇f_ ( _ζ_ _k_ _, z_ _k_ )


_ζ_ _k_ = _α_ _k_ _v_ _k_ + (1 _−_ _α_ _k_ ) _w_ _k_

_v_ _k_ +1 = _β_ _k_ _v_ _k_ + (1 _−_ _β_ _k_ ) _ζ_ _k_ _−_ _γ_ _k_ _η∇f_ ( _ζ_ _k_ _, z_ _k_ )


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


Since the stochastic gradients are unbiased, we obtain the following equation,


E _z_ [ _∇f_ ( _y, z_ )] = _∇f_ ( _y_ ) (9)


For the proof, we consider the more general strong-growth condition with an additive error _σ_ [2] .


E _z_ _∥∇f_ ( _w, z_ ) _∥_ [2] _≤_ _ρ ∥∇f_ ( _w_ ) _∥_ [2] + _σ_ [2] (10)


We choose the parameters _γ_ _k_, _α_ _k_, _β_ _k_, _a_ _k_, _b_ _k_ such that the following equations are satisfied:




[1] 1 + _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]

_ρ_ _[·]_ � _α_ _k_



(11)
�



_γ_ _k_ = [1]



_α_ _k_



_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_
_α_ _k_ = (12)
_γ_ _k_ _β_ _k_ _b_ [2] _k_ +1 _[η]_ [ +] _[ a]_ _k_ [2]

_β_ _k_ _≥_ 1 _−_ _γ_ _k_ _µη_ (13)

_a_ _k_ +1 = _γ_ _k_ _√_ ~~_ηρ_~~ _b_ _k_ +1 (14)


_b_ _k_
_b_ _k_ +1 _≤_ (15)
~~_√_~~ _β_ _k_


We now prove the following lemma assuming that the function _f_ ( _·_ ) is _L_ -smooth and _µ_ strongly-convex.

**Lemma 3.** _Assume that the function is L-smooth and µ strongly-convex and satisfies the strong-growth condition_
_in Equation 10. Then, using the updates in Equation 3-5 and setting the parameters according to Equations 11-_
_15, if η ≤_ _ρL_ 1 _[, then the following relation holds:]_



0 0
_b_ [2] _k_ +1 _[γ]_ _k_ [2] [[][E] _[f]_ [(] _[w]_ _[k]_ [+1] [)] _[ −]_ _[f]_ _[ ∗]_ []] _[ ≤]_ _ρη_ _[a]_ [2] [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] 2 _[b]_ _ρη_ [2] _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ _ρ_ [2] _[η]_



_k_
�[ _γ_ _i_ [2] _[b]_ [2] _i_ +1 []]


_i_ =0



_Proof._


Let _r_ _k_ +1 = _∥v_ _k_ +1 _−_ _w_ _[∗]_ _∥_, then using equation 5


_r_ _k_ [2] +1 [=] _[ ∥][β]_ _[k]_ _[v]_ _[k]_ [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[−]_ _[γ]_ _[k]_ _[η][∇][f]_ [(] _[ζ]_ _[k]_ _[, z]_ _[k]_ [)] _[∥]_ [2]

_r_ _k_ [2] +1 [=] _[ ∥][β]_ _[k]_ _[v]_ _[k]_ [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ ∥∇][f]_ [(] _[ζ]_ _[k]_ _[, z]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η][⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ _[, z]_ _[k]_ [)] _[⟩]_


Taking expecation wrt to _z_ _k_,


E[ _r_ _k_ [2] +1 [] =][ E][[] _[∥][β]_ _[k]_ _[v]_ _[k]_ [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [] +] _[ γ]_ _k_ [2] _[η]_ [2] [E] _[ ∥∇][f]_ [(] _[ζ]_ _[k]_ _[, z]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [][E] _[⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ _[, z]_ _[k]_ [)] _[⟩]_ []]

_≤∥β_ _k_ _v_ _k_ + (1 _−_ _β_ _k_ ) _ζ_ _k_ _−_ _w_ _[∗]_ _∥_ [2] + _γ_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [] _[⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ [)] _[⟩]_ [] +] _[ γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]

= _∥β_ _k_ ( _v_ _k_ _−_ _w_ _[∗]_ ) + (1 _−_ _β_ _k_ )( _ζ_ _k_ _−_ _w_ _[∗]_ ) _∥_ [2] + _γ_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [] _[⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ [)] _[⟩]_ [] +] _[ γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]

_≤_ _β_ _k_ _∥v_ _k_ _−_ _w_ _[∗]_ _∥_ [2] + (1 _−_ _β_ _k_ ) _∥ζ_ _k_ _−_ _w_ _[∗]_ _∥_ [2] + _γ_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [] _[⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ [)] _[⟩]_ [] +] _[ γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]

(By convexity of _∥·∥_ [2] )

= _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [] _[⟨][w]_ _[∗]_ _[−]_ _[β]_ _[k]_ _[v]_ _[k]_ _[−]_ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ [)] _[⟩]_ [] +] _[ γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]

= _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ [ [] _[⟨][β]_ _[k]_ [(] _[ζ]_ _[k]_ _[−]_ _[v]_ _[k]_ [) +] _[ w]_ _[∗]_ _[−]_ _[ζ]_ _[k]_ _[,][ ∇][f]_ [(] _[ζ]_ _[k]_ [)] _[⟩]_ [] +] _[ γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]

= _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ _⟨_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)] ( _w_ _k_ _−_ _ζ_ _k_ ) + _w_ _[∗]_ _−_ _ζ_ _k_ _, ∇f_ ( _ζ_ _k_ ) _⟩_ + _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2]
� _α_ _k_ �

(From equation 4)

= _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ _β_ _k_ (1 _−_ _α_ _k_ ) _⟨∇f_ ( _ζ_ _k_ ) _,_ ( _w_ _k_ _−_ _ζ_ _k_ ) _⟩_ + _⟨∇f_ ( _ζ_ _k_ ) _, w_ _[∗]_ _−_ _ζ_ _k_ _⟩_ + _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2]
� _α_ _k_ �

_≤_ _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2] [ + 2] _[γ]_ _[k]_ _[η]_ _β_ _k_ (1 _−_ _α_ _k_ ) ( _f_ ( _w_ _k_ ) _−_ _f_ ( _ζ_ _k_ )) + _⟨∇f_ ( _ζ_ _k_ ) _, w_ _[∗]_ _−_ _ζ_ _k_ _⟩_ + _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2]
� _α_ _k_ �

(By convexity)


**Sharan Vaswani, Francis Bach, Mark Schmidt**


By strong convexity,


E[ _r_ _k_ [2] +1 []] _[ ≤]_ _[β]_ _[k]_ _[r]_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ γ]_ _k_ [2] _[η]_ [2] _[ρ][ ∥∇][f]_ [(] _[ζ]_ _[k]_ [)] _[∥]_ [2]



_β_ _k_ (1 _−_ _α_ _k_ )
+ 2 _γ_ _k_ _η_
� _α_ _k_




_[µ]_ + _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2] (16)

2 _[∥][ζ]_ _[k]_ _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] �



_−_ _α_ _k_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _ζ_ _k_ )) + _f_ _[∗]_ _−_ _f_ ( _ζ_ _k_ ) _−_ _[µ]_

_α_ _k_ 2



By Lipschitz continuity of the gradient,


_f_ ( _w_ _k_ +1 ) _−_ _f_ ( _ζ_ _k_ ) _≤⟨∇f_ ( _ζ_ _k_ ) _, w_ _k_ +1 _−_ _ζ_ _k_ _⟩_ + _[L]_

2 _[∥][w]_ _[k]_ [+1] _[ −]_ _[ζ]_ _[k]_ _[∥]_ [2]

_≤−η⟨∇f_ ( _ζ_ _k_ ) _, ∇f_ ( _ζ_ _k_ _, z_ _k_ ) _⟩_ + _[L][η]_ [2] _∥∇f_ ( _ζ_ _k_ _, z_ _k_ ) _∥_ [2]

2


Taking expectation wrt _z_ _k_ and using equations 9, 10




_[ρη]_ [2] _∥∇f_ ( _ζ_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]

2 2



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _ζ_ _k_ )] _≤−η ∥∇f_ ( _ζ_ _k_ ) _∥_ [2] + _[L][ρη]_ [2]



2



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _ζ_ _k_ )] _≤_ _−η_ + _[L][ρη]_ [2]
� 2



_∥∇f_ ( _ζ_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]
� 2



If _η ≤_ _ρL_ 1 [,]



_−η_
E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _ζ_ _k_ )] _≤_
� 2



_∥∇f_ ( _ζ_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]
� 2



2
= _⇒∥∇f_ ( _ζ_ _k_ ) _∥_ [2] _≤_
� _η_


From equations 16 and 17,



E[ _f_ ( _ζ_ _k_ ) _−_ _f_ ( _w_ _k_ +1 )] + _Lησ_ [2] (17)
�



E[ _r_ _k_ [2] +1 []] _[ ≤]_ _[β]_ _[k]_ _[r]_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ + 2] _[γ]_ _k_ [2] _[ρη]_ [E][[] _[f]_ [(] _[ζ]_ _[k]_ [)] _[ −]_ _[f]_ [(] _[w]_ _[k]_ [+1] [)]]



_β_ _k_ (1 _−_ _α_ _k_ )
+ 2 _γ_ _k_ _η_
� _α_ _k_




_[µ]_ + _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2] [ +] _[ Lγ]_ _k_ [2] _[η]_ [3] _[ρσ]_ [2]

2 _[∥][ζ]_ _[k]_ _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] �



_−_ _α_ _k_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _ζ_ _k_ )) + _f_ _[∗]_ _−_ _f_ ( _ζ_ _k_ ) _−_ _[µ]_

_α_ _k_ 2



_≤_ _β_ _k_ _r_ _k_ [2] [+ (1] _[ −]_ _[β]_ _[k]_ [)] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ + 2] _[γ]_ _k_ [2] _[ηρ]_ [E][[] _[f]_ [(] _[ζ]_ _[k]_ [)] _[ −]_ _[f]_ [(] _[w]_ _[k]_ [+1] [)]]



_β_ _k_ (1 _−_ _α_ _k_ )
+ 2 _γ_ _k_ _η_
� _α_ _k_




_[µ]_ 2 _[∥][ζ]_ _[k]_ _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] � + 2 _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2] (Since _η ≤_ _ρL_ 1 [)]



_−_ _α_ _k_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _ζ_ _k_ )) + _f_ _[∗]_ _−_ _f_ ( _ζ_ _k_ ) _−_ _[µ]_

_α_ _k_ 2



= _β_ _k_ _r_ _k_ [2] [+] _[ ∥][ζ]_ _[k]_ _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ [(1] _[ −]_ _[β]_ _[k]_ [)] _[ −]_ _[γ]_ _[k]_ _[µη]_ [] +] _[ f]_ [(] _[ζ]_ _[k]_ [)] 2 _γ_ _k_ [2] _[ηρ][ −]_ [2] _[γ]_ _[k]_ _[η][ ·]_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)] _−_ 2 _γ_ _k_ _η_
� _α_ _k_ �



_−_ 2 _γ_ _k_ [2] _[ηρ]_ [E] _[f]_ [(] _[w]_ _[k]_ [+1] [) + 2] _[γ]_ _[k]_ _[ηf]_ _[ ∗]_ [+] 2 _γ_ _k_ _η ·_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]
� _α_ _k_



_f_ ( _w_ _k_ ) + 2 _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2]
�



_ρ_ [1] _[·]_ �1 + _[β]_ _[k]_ [(][1] _α_ _[−]_ _k_ _[α]_ _[k]_ [)]



,
�



Since _β_ _k_ _≥_ 1 _−_ _γ_ _k_ _µη_ and _γ_ _k_ = [1]



_α_ _k_



E[ _r_ _k_ [2] +1 []] _[ ≤]_ _[β]_ _[k]_ _[r]_ _k_ [2] _[−]_ [2] _[γ]_ _k_ [2] _[ηρ]_ [E] _[f]_ [(] _[w]_ _[k]_ [+1] [) + 2] _[γ]_ _[k]_ _[ηf]_ _[ ∗]_ [+] 2 _γ_ _k_ _η ·_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]
� _α_ _k_


Multiplying by _b_ [2] _k_ +1 [,]



E[ _r_ _k_ [2] +1 []] _[ ≤]_ _[β]_ _[k]_ _[r]_ _k_ [2] _[−]_ [2] _[γ]_ _k_ [2] _[ηρ]_ [E] _[f]_ [(] _[w]_ _[k]_ [+1] [) + 2] _[γ]_ _[k]_ _[ηf]_ _[ ∗]_ [+] 2 _γ_ _k_ _η ·_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]
� _α_ _k_



_f_ ( _w_ _k_ ) + 2 _γ_ _k_ [2] _[η]_ [2] _[σ]_ [2]
�



_b_ [2] _k_ +1 [E][[] _[r]_ _k_ [2] +1 []] _[ ≤]_ _[b]_ _k_ [2] +1 _[β]_ _[k]_ _[r]_ _k_ [2] _[−]_ [2] _[b]_ [2] _k_ +1 _[γ]_ _k_ [2] _[ηρ]_ [E] _[f]_ [(] _[w]_ _[k]_ [+1] [) + 2] _[b]_ [2] _k_ +1 _[γ]_ _[k]_ _[ηf]_ _[ ∗]_ [+] 2 _b_ [2] _k_ +1 _[γ]_ _[k]_ _[η][ ·]_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]
� _α_ _k_



_f_ ( _w_ _k_ ) + 2 _b_ [2] _k_ +1 _[γ]_ _k_ [2] _[η]_ [2] _[σ]_ [2]
�


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


Since _b_ [2] _k_ +1 _[β]_ _[k]_ _[ ≤]_ _[b]_ _k_ [2] [,] _[ b]_ [2] _k_ +1 _[γ]_ _k_ [2] _[ηρ]_ [ =] _[ a]_ [2] _k_ +1 [,] _[γ]_ _[k]_ _[ηβ]_ _[k]_ _α_ [(] _k_ [1] _[−][α]_ _[k]_ [)] = _b_ [2] _k_ _a_ +1 [2] _k_


_k_ +1 _[σ]_ [2] _[η]_
_b_ [2] _k_ +1 [E][[] _[r]_ _k_ [2] +1 []] _[ ≤]_ _[b]_ _k_ [2] _[r]_ _k_ [2] _[−]_ [2] _[a]_ [2] _k_ +1 [E] _[f]_ [(] _[w]_ _[k]_ [+1] [) + 2] _[b]_ _k_ [2] +1 _[γ]_ _[k]_ _[ηf]_ _[ ∗]_ [+ 2] _[a]_ _k_ [2] _[f]_ [(] _[w]_ _[k]_ [) +] [2] _[a]_ [2]

_ρ_

= _b_ [2] _k_ _[r]_ _k_ [2] _[−]_ [2] _[a]_ [2] _k_ +1 [[][E] _[f]_ [(] _[w]_ _[k]_ [+1] [)] _[ −]_ _[f]_ _[ ∗]_ [] + 2] _[a]_ [2] _k_ [[] _[f]_ [(] _[w]_ _[k]_ [)] _[ −]_ _[f]_ _[ ∗]_ [] + 2] � _b_ [2] _k_ +1 _[γ]_ _[k]_ _[η][ −]_ _[a]_ _k_ [2] +1 [+] _[ a]_ [2] _k_ � _f_ _[∗]_ + [2] _[a]_ _k_ [2] +1 _[σ]_ [2] _[η]_

_ρ_


Since � _b_ [2] _k_ +1 _[γ]_ _[k]_ _[η][ −]_ _[a]_ _k_ [2] +1 [+] _[ a]_ [2] _k_ � = 0,


_k_ +1 _[σ]_ [2] _[η]_
_b_ [2] _k_ +1 [E][[] _[r]_ _k_ [2] +1 []] _[ ≤]_ _[b]_ _k_ [2] _[r]_ _k_ [2] _[−]_ [2] _[a]_ [2] _k_ +1 [[][E] _[f]_ [(] _[w]_ _[k]_ [+1] [)] _[ −]_ _[f]_ _[ ∗]_ [] + 2] _[a]_ [2] _k_ [[] _[f]_ [(] _[w]_ _[k]_ [)] _[ −]_ _[f]_ _[ ∗]_ [] +] [2] _[a]_ [2]

_ρ_


Denoting E _f_ ( _w_ _k_ +1 ) as _φ_ _k_,

2 _a_ [2] _k_ +1 [[] _[φ]_ _[k]_ [+1] _[−]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] _k_ +1 [E][[] _[r]_ _k_ [2] +1 []] _[ ≤]_ [2] _[a]_ _k_ [2] [[] _[φ]_ _[k]_ _[−]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] _k_ [E][[] _[r]_ _k_ [2] [] +] [2] _[a]_ _k_ [2] +1 _[σ]_ [2] _[η]_

_ρ_


By recursion,



2 _a_ [2] _k_ +1 [[] _[φ]_ _[k]_ [+1] _[−]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] _k_ +1 [E][[] _[r]_ _k_ [2] +1 []] _[ ≤]_ [2] _[a]_ 0 [2] [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] 0 _[∥][x]_ [0] _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] [2] _[σ]_ [2] _[η]_

_ρ_


2 _a_ [2] _k_ +1 [[] _[φ]_ _[k]_ [+1] _[−]_ _[f]_ _[ ∗]_ []] _[ ≤]_ [2] _[a]_ [2] 0 [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] 0 _[∥][x]_ [0] _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ + 2] _[σ]_ [2] _[η]_

_ρ_



_k_
�[ _a_ [2] _i_ +1 []]


_i_ =0



_k_
�[ _a_ [2] _i_ +1 []]


_i_ =0



2 _b_ [2] _k_ +1 _[γ]_ _k_ [2] _[ρη]_ [ [] _[φ]_ _[k]_ [+1] _[−]_ _[f]_ _[ ∗]_ []] _[ ≤]_ [2] _[a]_ [2] 0 [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] _[ b]_ [2] 0 _[∥][x]_ [0] _[−]_ _[w]_ _[∗]_ _[∥]_ [2] [ + 2] _[σ]_ [2] _[η]_ [2] _[ρ]_



_k_
�[ _γ_ _i_ [2] _[b]_ [2] _i_ +1 []]


_i_ =0



0 0
_b_ [2] _k_ +1 _[γ]_ _k_ [2] [[][E] _[f]_ [(] _[w]_ _[k]_ [+1] [)] _[ −]_ _[f]_ _[ ∗]_ []] _[ ≤]_ _ρη_ _[a]_ [2] [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] 2 _[b]_ _ρη_ [2] _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ _ρ_ [2] _[η]_



_k_
�[ _γ_ _i_ [2] _[b]_ [2] _i_ +1 []]


_i_ =0



**Lemma 4.** _Under the parameter setting according to Equations 11- 15, the following relation is true:_



= _γ_ _k_ [2] _−_ 1
�



_γ_ _k_ [2] _[−]_ _[γ]_ _[k]_



1
� _ρ_ _[−]_ _[µηγ]_ _k_ [2] _−_ 1



_Proof._



(From equation 11)
�



_γ_ _k_ = [1]

_ρ_



1 + _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]
� _α_ _k_




_[k]_ _[γ]_ _[k]_ _[β]_ _[k]_ [(][1] _[ −]_ _[α]_ _[k]_ [)]

_ρ_ [=] _ρα_ _k_



_γ_ _k_ [2] _[−]_ _[γ]_ _[k]_



_ρα_ _k_



= [1] _a_ [2] _k_ (From equation 12)

_ηρ_ _b_ [2] _k_ +1

= _[β]_ _[k]_ _a_ [2] _k_ (From equation 15)

_ηρ_ _b_ [2] _k_

= [1] _[ −]_ _[γ]_ _[k]_ _[µη]_ _a_ [2] _k_ (From equation 13)

_ηρ_ _b_ [2] _k_


2

= [1] _[ −]_ _[γ]_ _[k]_ _[µη]_ ( _γ_ _k−_ 1 _√_ ~~_ηρ_~~ ~~)~~ (From equation 13)

_ηρ_



_a_ [2] _k_

[1] (From equation 12)

_ηρ_ _b_ [2] _k_ +1

_a_ [2] _k_

_[β]_ _[k]_ (From equation 15)

_ηρ_ _b_ [2] _k_




_[γ]_ _[k]_ _[µη]_ _a_ [2] _k_ (From equation 13)

_ηρ_ _b_ [2] _k_



= [1] _[ −]_ _[γ]_ _[k]_ _[µη]_



= (1 _−_ _γ_ _k_ _µη_ ) _γ_ _k_ [2] _−_ 1


= _γ_ _k_ [2] _−_ 1 (18)
�



= _⇒_ _γ_ _k_ [2] _[−]_ _[γ]_ _[k]_



1
� _ρ_ _[−]_ _[µηγ]_ _k_ [2] _−_ 1


**Sharan Vaswani, Francis Bach, Mark Schmidt**


**B.1.1** **Strongly-convex case**


We now consider the strongly-convex case,


Using Lemma 4,



= _γ_ _k_ [2] _−_ 1
�



_γ_ _k_ [2] _[−]_ _[γ]_ _[k]_


If _γ_ _k_ = _C_, then


If _b_ 0 = _[√]_ ~~_µ_~~ ~~,~~



1
� _ρ_ _[−]_ _[µηγ]_ _k_ [2] _−_ 1



1
_γ_ _k_ =
~~_√µηρ_~~


~~_µη_~~
_β_ _k_ = 1 _−_
� _ρ_



_b_ _k_ +1 = _b_ 0
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_



1 _b_ 0
_a_ _k_ +1 = ~~_ηρ_~~ _·_
~~_√µηρ_~~ _·_ _[√]_ 1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



_b_ 0 1

_[b]_ [0]

_µη_ ( _k_ +1) _/_ 2 [=] ~~_√µ_~~ _·_ 1 _−_ _µη_

_ρ_ ~~�~~ ~~�~~ ~~�~~ _ρ_



_µη_



_µη_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



1
_a_ _k_ +1 =
1 _−_ _µη_
~~�~~ ~~�~~ _ρ_



( _k_ +1) _/_ 2

_µη_

_ρ_ ~~�~~



_µη_



The above equation implies that _a_ 0 = 1. This gives us the parameter settings used in Theorem 1.


Using the result of Lemma 3 and the above relations, we obtain the following inequality. Note that _φ_ _k_ +1 =
E[ _f_ ( _w_ _k_ +1 )].



_µ_

1 _−_
~~�~~ ~~�~~



( _i_ +1)

_µη_

_ρ_ ~~�~~



1

_µη_ ( _k_ +1) _[·]_ _µηρ_ [[] _[φ]_ _[k]_ [+1] _[ −]_ _[f]_ _[ ∗]_ []] _[ ≤]_ [1]

_ρ_ ~~�~~




_[η]_ 1


_·_

_ρ_ _µηρ_



_k_
�


_i_ =0



_µη_



_µ_

1 _−_
~~�~~ ~~�~~



_µη_




[1] _µ_

_ρη_ [[] _[f]_ [(] _[x]_ [0] [)] _[ −]_ _[f]_ _[ ∗]_ [] +] 2 _ρη_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ _ρ_ [2] _[η]_



1

1 _−_
~~�~~ ~~�~~



( _i_ +1)

_µη_

_ρ_ ~~�~~



_k_
�


_i_ =0



_µη_



1

1 _−_
~~�~~ ~~�~~


1

1 _−_
~~�~~ ~~�~~



_µη_ _k_ [[] _[φ]_ _k_ +1 _[−]_ _[f]_ _[ ∗]_ []] _[ ≤]_ [[] _[f]_ [(] _[x]_ 0 [)] _[ −]_ _[f]_ _[ ∗]_ [] +] _[µ]_ 2

_ρ_ ~~�~~



_µη_




_[µ]_

2 _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ [2] _[η]_



_ρ_



_−k_

~~_µη_~~
1 _−_
� � _ρ_ �



_µη_ _k_ [[] _[φ]_ _k_ +1 _[−]_ _[f]_ _[ ∗]_ []] _[ ≤]_ � _f_ ( _x_ 0 ) _−_ _f_ _[∗]_ + _[µ]_ 2

_ρ_ ~~�~~



_µη_




_[µ]_ + _[σ]_ [2] _[√]_ ~~_[η]_~~

2 _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�] ~~_√ρµ_~~



~~_µη_~~

[ _φ_ _k_ +1 _−_ _f_ _[∗]_ ] _≤_ 1 _−_
� � _ρ_



_k_
_f_ ( _x_ 0 ) _−_ _f_ _[∗]_ + _[µ]_ + _[σ]_ [2] _[√]_ ~~_[η]_~~
� � 2 _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�] ~~_√ρµ_~~


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


**B.1.2** **Proof of Theorem 1**


1
We use the above relation to complete the proof for Theorem 1. Substituting _η_ = _ρL_ [and] _[ σ]_ [ = 0, we obtain the]
following:



~~_µη_~~

[E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ _[∗]_ ] _≤_ 1 _−_
� � _ρ_



_k_
_f_ ( _x_ 0 ) _−_ _f_ _[∗]_ + _[µ]_
� � 2 _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�]



**B.1.3** **Convex case**


We now use the above lemmas to first prove the convergence rate in the convex case. In this case, _µ_ = 0 and the
result of Lemma 4 can be written as:

_γ_ _k_ [2] _[−]_ _[γ]_ _ρ_ _[k]_ _[−]_ _[γ]_ _k_ [2] _−_ 1 [= 0]



= _⇒_ _γ_ _k_ =



1 1
_ρ_ [+] ~~�~~ _ρ_ [2] [+ 4] _[γ]_ _k_ [2] _−_ 1


2



Let _γ_ 0 = 0. From equation 13, for all _k_,


_β_ _k_ = 1


_b_ _k_ +1 = _b_ _k_ = _b_ 0 = 1 (From equation 15)

_a_ _k_ +1 = _γ_ _k_ _√_ ~~_ηρ_~~ _b_ 0 = _⇒_ _a_ _k_ +1 = _γ_ _k_ _√_ ~~_ηρ_~~ (From equation 14)


The above equation implies that _a_ 0 = 0. This gives us the parameter settings used in Theorem 2.


Using the result of Lemma 3 by setting _µ_ = 0 and the above relations, we obtain the following inequality. Note
that _φ_ _k_ +1 = E[ _f_ ( _w_ _k_ +1 )].



1
_γ_ _k_ [2] [[] _[φ]_ _[k]_ [+1] _[−]_ _[f]_ _[ ∗]_ []] _[ ≤]_ 2 _ρη_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ _ρ_ [2] _[η]_



By induction, _γ_ _i_ _≥_ 2 _iρ_ [,]



_k_ [2] 1

4 _ρ_ [2] [[] _[φ]_ _[k]_ [+1] _[ −]_ _[f]_ _[ ∗]_ []] _[ ≤]_ 2 _ρη_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ 4 _ρ_ [2] _[η]_ [3]



_k_ [2]



4 _ρ_ [3]




[2] _[ρ]_

_k_ [2] _η_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ σ]_ _k_ [2][2] _ρ_ _[η]_




[ _φ_ _k_ +1 _−_ _f_ _[∗]_ ] _≤_ [2] _[ρ]_



_k_ [2] _ρ_



_k−_ 1
�[ _γ_ _i_ [2] []]


_i_ =1


_k−_ 1
�[ _i_ [2] ]


_i_ =1


_k−_ 1
�[ _i_ [2] ]


_i_ =1




[2] _[ρ]_

_k_ [2] _η_ _[∥][x]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ +] _[ kσ]_ _ρ_ [2] _[η]_




[ _φ_ _k_ +1 _−_ _f_ _[∗]_ ] _≤_ [2] _[ρ]_



_ρ_



**B.1.4** **Proof of Theorem 2**


1
We use the above relation to complete the proof for Theorem 2. Substituting _η_ = _ρL_ [and] _[ σ]_ [ = 0, we obtain the]
following:


[E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ _[∗]_ ] _≤_ [2] _[ρ]_ [2] _[L]_ _∥x_ 0 _−_ _w_ _[∗]_ _∥_ [2]

_k_ [2]


**Sharan Vaswani, Francis Bach, Mark Schmidt**


**B.2** **Proof of Theorem 3**


_Proof._ Recall the stochastic gradient descent update,


_w_ _k_ +1 = _w_ _k_ _−_ _η∇f_ ( _w_ _k_ _, z_ _k_ ) (19)


By Lipschitz continuity of the gradient,


_f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ ) _≤⟨∇f_ ( _w_ _k_ ) _, w_ _k_ +1 _−_ _w_ _k_ _⟩_ + _[L]_

2 _[∥][w]_ _[k]_ [+1] _[ −]_ _[w]_ _[k]_ _[∥]_ [2]

_≤−η⟨∇f_ ( _w_ _k_ ) _, ∇f_ ( _w_ _k_ _, z_ _k_ ) _⟩_ + _[L][η]_ [2] _∥∇f_ ( _w_ _k_ _, z_ _k_ ) _∥_ [2]

2


Taking expectation wrt _z_ _k_ and using equations 9, 10




_[ρη]_ [2] _∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]

2 2



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤−η ∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[L][ρη]_ [2]



2



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤_ _−η_ + _[L][ρη]_ [2]
� 2



_∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]
� 2



If _η ≤_ _ρL_ 1 [,]



_−η_
E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤_
� 2



_∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]
� 2



2
= _⇒∥∇f_ ( _w_ _k_ ) _∥_ [2] _≤_
� _η_



E[ _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _k_ +1 )] + _Lησ_ [2] (20)
�



Taking expectation wrt _z_ 0 _, z_ 1 _, . . . z_ _t−_ 1 and summing from _k_ = 0 to _t −_ 1,



_t−_ 1
�



_η_



_−_ 2

_k_ � =0 E � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _η_



_t−_ 1
� E [ _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _k_ +1 )] + _Lηtσ_ [2]
� _k_ =0



_η_



= _⇒_



_t−_ 1
�



_−_ 2

_k_ � =0 _k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _η_



_t−_ 1
� E [ _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _k_ +1 )] + _Lησ_ [2]
� _k_ =0



2
_k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _ηt_



2
_k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _ηt_


2
_k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _ηt_



2
_k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _ηt_



If _σ_ = 0,



2
_k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _ηt_




[ _f_ ( _w_ 0 ) _−_ E[ _f_ ( _w_ _t_ )]] + _Lησ_ [2]
�


[ _f_ ( _w_ 0 ) _−_ _f_ ( _w_ _[∗]_ )] + _Lησ_ [2]
�


[ _f_ ( _w_ 0 ) _−_ _f_ ( _w_ _[∗]_ )]
�



2 _ρL_
= _⇒_ _k_ =0 min _,_ 1 _,...t−_ 1 [E] � _∥∇f_ ( _w_ _k_ ) _∥_ [2] [�] _≤_ � _t_


**B.3** **Proof of Theorem 4**



1

[ _f_ ( _w_ 0 ) _−_ _f_ ( _w_ _[∗]_ )] (Setting _η_ = _ρL_ [)]
�



_Proof._ Similar to the proof of Theorem 3, we can use the SGD update and Lipschitz continuity of the gradient
to obtain the following equation for the stepsize _η ≤_ _ρL_ 1 [:]



_−η_
E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤_
� 2



_∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[L][η]_ [2] _[σ]_ [2]
� 2


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


We now use the PL inequality with constant _µ_ as follows:


_∥∇f_ ( _w_ _k_ ) _∥_ [2] _≥_ 2 _µ_ [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]


Combining the above two inequalities,



If _σ_ = 0,


1
Substituting _η_ = _ρL_ [,]



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤−ηµ_ [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ] + _[L][η]_ [2] _[σ]_ [2]

2


E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ ( _w_ _k_ )] _≤−ηµ_ [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]


= _⇒_ E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ _[∗]_ ] _≤_ (1 _−_ _ηµ_ ) [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]



E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ _[∗]_ ] _≤_ 1 _−_ _[µ]_
� _ρL_


= _⇒_ E[ _f_ ( _w_ _k_ +1 ) _−_ _f_ _[∗]_ ] _≤_ 1 _−_ _[µ]_
� _ρL_


**B.4** **Proof of Theorem 5**


_Proof._


_∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] = _∥w_ _k_ _−_ _η∇f_ ( _w_ _k_ _, z_ ) _−_ _w_ _[∗]_ _∥_ [2]




[ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]
�


_k_

[ _f_ ( _w_ 0 ) _−_ _f_ _[∗]_ ]
�



(21)



= _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ 2 _η⟨∇f_ ( _w_ _k_ _, z_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_ + _η_ [2] _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2]

E _z_ [ _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] ] = _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ 2 _η_ E[ _⟨∇f_ ( _w_ _k_ _, z_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_ ] + _η_ [2] E[ _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] ]

= _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ 2 _η⟨∇f_ ( _w_ _k_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_ + _η_ [2] E[ _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] ]
(From the unbiasedness of stochastic gradients.)

_≤∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ 2 _η⟨∇f_ ( _w_ _k_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_ + 2 _ρη_ [2] _L_ [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ] (From equation 6)

_≤∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] + 2 _η_ _f_ _[∗]_ _−_ _f_ ( _w_ _k_ ) _−_ _[µ]_ + 2 _ρη_ [2] _L_ [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]
� 2 _[∥][w]_ _[k]_ _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [�]

(By strong convexity)

= (1 _−_ _µη_ ) _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] + �2 _η_ [2] _ρL −_ 2 _η_ � [ _f_ ( _w_ _k_ ) _−_ _f_ _[∗]_ ]



_∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] _≤_ 1 _−_ _[µ]_
� _ρL_


= _⇒∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] _≤_ 1 _−_ _[µ]_
� _ρL_


**B.5** **Proof of Theorem 6**


_Proof._


By convexity,



1
_∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] (Setting _η_ = _ρL_ [)]
�


_k_
_∥x_ 0 _−_ _w_ _[∗]_ _∥_ [2]
�



_f_ ( _w_ _k_ ) _≤_ _f_ ( _w_ _[∗]_ ) + _⟨∇f_ ( _w_ _k_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_


**Sharan Vaswani, Francis Bach, Mark Schmidt**


For any _β ≤_ 1,


_f_ ( _w_ _k_ ) _≤_ _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + (1 _−_ _β_ ) _⟨∇f_ ( _w_ _k_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩_


By Lipschitz continuity of _∇f_ ( _f_ ),


_f_ ( _w_ _k_ +1 ) _≤_ _f_ ( _w_ _k_ ) + _⟨∇f_ ( _w_ _k_ ) _, w_ _k_ +1 _−_ _w_ _k_ _⟩_ + _[L]_

2 _[∥][w]_ _[k]_ [+1] _[ −]_ _[w]_ _[k]_ _[∥]_ [2]

= _⇒_ _f_ ( _w_ _k_ +1 ) _≤_ _f_ ( _w_ _k_ ) _−_ _η⟨∇f_ ( _w_ _k_ ) _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_ + _[η]_ [2] _[L]_ _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2]

2


From the above equations,


_f_ ( _w_ _k_ +1 ) _≤_ _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + (1 _−_ _β_ ) _⟨∇f_ ( _w_ _k_ ) _, w_ _k_ _−_ _w_ _[∗]_ _⟩−_ _η⟨∇f_ ( _w_ _k_ ) _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_ + _[η]_ [2] _[L]_ _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2]

2


Note that,



1

2 _η_


1

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [�] = 2 [1] _η_

= [1]

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−∥w_ _k_ _−_ _η∇f_ ( _w_ _k_ _, z_ ) _−_ _w_ _[∗]_ _∥_ [2] [�]


_∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ _η_ [2] _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] + 2 _η⟨w_ _k_ _−_ _w_ _[∗]_ _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_
� �



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [�] = _[−]_ 2 _[η]_ _[∥∇][f]_ [(] _[w]_ _[k]_ _[, z]_ [)] _[∥]_ [2] [ +] _[ ⟨][w]_ _[k]_ _[ −]_ _[w]_ _[∗]_ _[,][ ∇][f]_ [(] _[w]_ _[k]_ _[, z]_ [)] _[⟩]_



= _⇒⟨w_ _k_ _−_ _w_ _[∗]_ _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_ = [1]

2 _η_


Taking expectation


E [ _⟨w_ _k_ _−_ _w_ _[∗]_ _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_ ] = [1]

2 _η_

= _⇒⟨w_ _k_ _−_ _w_ _[∗]_ _, ∇f_ ( _w_ _k_ ) _⟩_ = [1]

2 _η_


Using the above equations,



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [�] + _[η]_ 2 _[∥∇][f]_ [(] _[w]_ _[k]_ _[, z]_ [)] _[∥]_ [2]



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + _[η]_ 2 [E] � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + _[η]_ 2 [E] � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



_f_ ( _w_ _k_ +1 ) _≤_ _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + [(][1] _[ −]_ 2 _[β]_ [)(] _[η]_ [)] E � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



_−_ _η⟨∇f_ ( _w_ _k_ ) _, ∇f_ ( _w_ _k_ _, z_ ) _⟩_ + _[η]_ [2] _[L]_ _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2]

2



Taking expectation,


E[ _f_ ( _w_ _k_ +1 )] _≤_ _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + [(][1] _[ −]_ 2 _[β]_ [)(] _[η]_ [)] E � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



_−_ _η⟨∇f_ ( _w_ _k_ ) _,_ E [ _∇f_ ( _w_ _k_ _, z_ )] _⟩_ + _[η]_ [2] 2 _[L]_ [E] � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



= _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + [(][1] _[ −]_ 2 _[β]_ [)(] _[η]_ [)] E � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



_−_ _η ∥∇f_ ( _w_ _k_ ) _∥_ [2] + _[η]_ [2] 2 _[L]_ [E] � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


The term _−η ∥∇f_ ( _w_ _k_ ) _∥_ [2] _≤_ 0



= _⇒_ E[ _f_ ( _w_ _k_ +1 )] _≤_ _βf_ ( _w_ _k_ ) + (1 _−_ _β_ ) _f_ ( _w_ _[∗]_ ) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��]




_[β]_ [)(] _[η]_ [)]

2 E � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�] + _[η]_ [2] 2 _[L]_



+ [(][1] _[ −]_ _[β]_ [)(] _[η]_ [)]



2 [E] � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ _β_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ )) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��]



_β_ )( _η_ )

+ _[η]_ [2] _[L]_
2 2



� E � _∥∇f_ ( _w_ _k_ _, z_ ) _∥_ [2] [�]



(1 _−_ _β_ )( _η_ )
+
� 2



2



From equation 6,


E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ _β_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ )) + [1] _[ −]_ _[β]_

2 _η_



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��]



+ � _ρ_ (1 _−_ _β_ ) _ηL_ + _η_ [2] _ρL_ [2] [�] ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ ))


Let us choose 1 _−_ _β_ = _ηL_,



E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ _β_ ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ )) + [1] _[ −]_ _[β]_

2 _η_



_∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��] + 2 _ρη_ [2] _L_ [2] ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ ))
� �



E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ � _β_ + 2 _ρη_ [2] _L_ [2] [�] ( _f_ ( _w_ _k_ ) _−_ _f_ ( _w_ _[∗]_ )) + _[L]_

2



� _∥w_ _k_ _−_ _w_ _[∗]_ _∥_ [2] _−_ E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [��]



Let _δ_ _k_ +1 = E[ _f_ ( _w_ _k_ +1 )] _−_ _f_ ( _w_ _[∗]_ ) and ∆ _k_ +1 = E � _∥w_ _k_ +1 _−_ _w_ _[∗]_ _∥_ [2] [�]


= _⇒_ _δ_ _k_ +1 _≤_ � _β_ + 2 _ρη_ [2] _L_ [2] [�] _δ_ _k_ + _[L]_

2 [[∆] _[k]_ _[ −]_ [∆] _[k]_ [+1] []]


Summing from _i_ = 0 to _k −_ 1,



_k−_ 1
�



_k−_ 1
� _δ_ _i_ +1 _≤_ � _β_ + 2 _ρη_ [2] _L_ [2] [�] _[k]_ � _[−]_ [1]


_i_ =0 _i_ =0



2



_k−_ 1
� [∆ _i_ _−_ ∆ _i_ +1 ]


_i_ =0



� _δ_ _i_ + _[L]_ 2

_i_ =0


_[−]_ [1]
� _δ_ _i_ + _[L]_ 2

_i_ =0



2 [∆] [0]



= _⇒_



_k−_ 1
�



_k−_ 1
� _δ_ _i_ +1 _≤_ � _β_ + 2 _ρη_ [2] _L_ [2] [�] _[k]_ � _[−]_ [1]


_i_ =0 _i_ =0



= _⇒_



_k_
� _δ_ _i_ _≤_


_i_ =1



� _β_ + 2 _ρη_ [2] _L_ [2] [�] _δ_ 0 + _[L]_ 2 [∆] [0]

(1 _−_ _β −_ 2 _ρη_ [2] _L_ [2] )



Let ¯ _w_ _k_ = [[] � _ki_ =1 _k_ _[w]_ _[i]_ []] . By Jensen’s inequality,


_k_
E[ _f_ ( ¯ _w_ _k_ )] _≤_ � _i_ =1 [E][[] _[f]_ [(] _[w]_ _[i]_ [)]]

_k_



= _⇒_ E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_


= _⇒_ E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_


E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_



_k_
� _δ_ _i_


_i_ =1

� _β_ + 2 _ρη_ [2] _L_ [2] [�] _δ_ 0 + _[L]_ 2 [∆] [0]

(1 _−_ _β −_ 2 _ρη_ [2] _L_ [2] ) _k_



(Since 1 _−_ _β_ = _ηL_ )
( _ηL −_ 2 _ρη_ [2] _L_ [2] ) _k_




_[L]_

2 _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]



�1 _−_ _ηL_ + 2 _ρη_ [2] _L_ [2] [�] [ _f_ ( _w_ 0 ) _−_ _f_ ( _w_ _[∗]_ )] + _[L]_ 2


If _η_ = 4 _ρL_ 1 [,]


E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_



**Sharan Vaswani, Francis Bach, Mark Schmidt**


87 _ρ_ [[] _[f]_ [(] _[w]_ [0] [)] _[ −]_ _[f]_ [(] _[w]_ _[∗]_ [)] +] _[L]_ 2 _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]


1
8 _ρ_ _[k]_




_[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]
E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ [7 ][[] _[f]_ [(] _[w]_ [0] [)] _[ −]_ _[f]_ [(] _[w]_ _[∗]_ [)]][ + 4] _[ρ][L]_

_k_




_[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2] [ + 4] _[ρ][L]_ _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]
E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ [(][7] _[L][/]_ [2][)]

_k_




_[ρ]_ [)] _[∥][w]_ [0] _[ −]_ _[w]_ _[∗]_ _[∥]_ [2]
= _⇒_ E[ _f_ ( ¯ _w_ _k_ )] _−_ _f_ ( _w_ _[∗]_ ) _≤_ [4][(][1 +]

_k_



**B.6** **Proof for Proposition 1**


_Proof._


For the first part, we use the PL inequality which states the for all _w_,


2 [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )] _≤_ _µ_ [1] _[∥∇][f]_ [(] _[w]_ [)] _[∥]_ [2]


Combining this with the WGC gives us the desired result


For the converse, we use smoothness and the convexity of _f_ ( _·_ ). Specifically, for all points _a_, _b_,


_f_ ( _a_ ) _−_ _f_ ( _b_ ) _≥⟨f_ ( _b_ ) _, a −_ _b⟩_ + [1]

2 _L_ _[∥∇][f]_ [(] _[a]_ [)] _[ −∇][f]_ [(] _[b]_ [)] _[∥]_ [2]


Substituting _a_ = _w_ and _b_ = _w_ _[∗]_ and rearranging,


_∥∇f_ ( _w_ ) _∥_ [2] _≤_ 2 _L ·_ [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )]


Combining this with the SGC gives us the desired result.


**B.7** **Proof for Proposition 2**


_Proof._



E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] = [1]

_n_



_n_
� _∥∇f_ _i_ ( _w_ ) _∥_ [2] (22)


_i_ =1



By Lipschitz continuity of _∇f_ _i_ ( _w_ ) and convexity,


1
_f_ _i_ ( _w_ ) _−_ _f_ _i_ ( _w_ _[∗]_ ) _≥⟨∇f_ _i_ ( _w_ _[∗]_ ) _, w −_ _w_ _[∗]_ _⟩_ + _∥∇f_ _i_ ( _w_ ) _−∇f_ _i_ ( _w_ _[∗]_ ) _∥_ [2]
2 _L_ _i_


**Fast and Faster Convergence of SGD for Over-Parameterized Models**


For all _i_, _∇f_ _i_ ( _w_ _[∗]_ ) = _∇f_ ( _w_ _[∗]_ ) = 0. Hence,


1
_f_ _i_ ( _w_ ) _−_ _f_ _i_ ( _w_ _[∗]_ ) _≥_ _∥∇f_ _i_ ( _w_ ) _∥_ [2]
2 _L_ _i_

= _⇒∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_ 2 _L_ _i_ [ _f_ _i_ ( _w_ ) _−_ _f_ _i_ ( _w_ _[∗]_ )]


Using Equation 22,



E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_



_n_
�


_i_ =1



� 2 _nL_ _i_ [[] _[f]_ _[i]_ [(] _[w]_ [)] _[ −]_ _[f]_ _[i]_ [(] _[w]_ _[∗]_ [)]] �



_≤_ [2] _[L]_ _[max]_

_n_



_n_
� [ _f_ _i_ ( _w_ ) _−_ _f_ _i_ ( _w_ _[∗]_ )]


_i_ =1



E _i_ _∥∇f_ _i_ ( _w_ ) _∥_ [2] _≤_ 2 _L_ _max_ [ _f_ ( _w_ ) _−_ _f_ ( _w_ _[∗]_ )] (23)


**B.8** **Proof for Lemma 1**


_Proof._ Let _a_ = _y · x_ . For the squared-hinge loss, the strong growth condition is equivalent to


E�(1 _−_ _w_ _[⊤]_ _a_ ) [2] + � ⩽ _ρ_ ��E�(1 _−_ _w_ _[⊤]_ _a_ ) + _a_ � _∥_ [2]


1
��E�(1 _−_ _w_ _[⊤]_ _a_ ) + _a_ ��� ⩾ �(1 _−_ _w_ _[⊤]_ _a_ ) + _a_ _[⊤]_ _w_ _∗_ �
_∥w_ _∗_ _∥_ [E]

⩾ _τ_ E�(1 _−_ _w_ _[⊤]_ _a_ ) + �


2
We thus need to upper bound E�(1 _−_ _w_ _[⊤]_ _a_ ) [2] + � by a constant _c_ times �E�(1 _−_ _w_ _[⊤]_ _a_ ) + �� . We must have _c_ ⩾ 1 (as
a consequence of Jensen’s inequality). Then we have _ρ_ = _c/τ_ [2] . Next, we prove that if the distribution of _a_ is
uniform over _κ_ values, then _c_ = _κ_ .


Consider a random variable _A ∈_ R+ taking _κ_ values _a_ 1 _, . . ., a_ _κ_ with probabilities _p_ 1 _, . . ., p_ _κ_ . Then (E _A_ ) [2] =
� _[p]_ _[i]_ _[p]_ _[j]_ _[a]_ _[i]_ _[a]_ _[j]_ [ ⩾] [�] _i_ _[a]_ _i_ [2] _[p]_ [2] _i_ [⩾] [min] _[i]_ _[ p]_ _[i]_ � _i_ _[a]_ _i_ [2] _[p]_ _[i]_ [,]



_i_ _[a]_ _i_ [2] _[p]_ [2] _i_ [⩾] [min] _[i]_ _[ p]_ _[i]_ �



_i,j_ _[p]_ _[i]_ _[p]_ _[j]_ _[a]_ _[i]_ _[a]_ _[j]_ [ ⩾] [�]



_i_ _[a]_ _i_ [2] _[p]_ _[i]_ [,]



**B.9** **Proof for Lemma 2**


_Proof._ Let _a_ = _y · x_ .


P( _a_ _[⊤]_ _w_ ⩽ 0) ⩽ P((1 _−_ _a_ _[⊤]_ _w_ ) [2] + [⩾] [1)]

⩽ E(1 _−_ _a_ _[⊤]_ _w_ ) [2] +
= _⇒_ P( _a_ _[⊤]_ _w_ ⩽ 0) _≤_ E _f_ ( _w, a_ )


**C** **Additional experimental results**


In this section, we propose to use a line-search heuristic for both constant step-size SGD and its accelerated
variant. For SGD, we use the line-search proposed in SAG [31]: start with an initial estimate _L_ [ˆ] = 1 and in


1

each iteration, we double the estimate when the condition _f_ _k_ � _w_ _k_ _−_ _L_ [1] ˆ _[∇][f]_ _[k]_ [(] _[w]_ _[k]_ [)] � _≤_ _f_ _k_ ( _w_ _k_ ) _−_ 2 _L_ [ˆ] _[∥∇][f]_ _[k]_ [(] _[w]_ _[k]_ [)] _[∥]_ [2] [ is]

not satisfied. We denote this variant as SGD(LS) and the corresponding variant that uses a 1 _/L_ step-size as
SGD(T). For the accelerated case, we use the same line-search procedure as above, but search for an appropriate
value of _ρL_ . We denote the accelerated variant with and without line-search as Acc-SGD(LS) and Acc-SGD(T)
respectively.


We make the following observations: (i) Accelerated SGD in conjunction with our line-search heuristic is stable
across datasets. (ii) Acc-SGD(LS) either matches or outperforms Acc-SGD(T). (iii) In some cases, SGD(LS)
can result in faster empirical convergence as compared to the accelerated variants. We plan to investigate better
line-search methods for both SGD [31] and Acc-SGD [21] in the future.


**Sharan Vaswani, Francis Bach, Mark Schmidt**


(a) _τ_ = 0 _._ 1 (b) _τ_ = 0 _._ 05


(c) _τ_ = 0 _._ 01 (d) _τ_ = 0 _._ 005


Figure 3: Comparison of SGD and variants of accelerated SGD on a synthetic linearly separable dataset with
margin _τ_ .


(a) CovType (b) Protein


Figure 4: Comparison of SGD and accelerated SGD for learning a linear classifier with RBF features on the (a)
CovType and (b) Protein datasets.


