## Batched Gaussian Process Bandit Optimization via Determinantal Point Processes

Tarun Kathuria, Amit Deshpande, Pushmeet Kohli
Microsoft Research
_{_ `t-takat, amitdesh, pkohli` _}_ `@microsoft.com`


**Abstract**


Gaussian Process bandit optimization has emerged as a powerful tool for optimizing noisy black box
functions. One example in machine learning is hyper-parameter optimization where each evaluation of
the target function requires training a model which may involve days or even weeks of computation. Most
methods for this so-called “Bayesian optimization” only allow sequential exploration of the parameter space.
However, it is often desirable to propose _batches_ or sets of parameter values to explore simultaneously,
especially when there are large parallel processing facilities at our disposal. Batch methods require
modeling the interaction between the different evaluations in the batch, which can be expensive in
complex scenarios. In this paper, we propose a new approach for parallelizing Bayesian optimization by
modeling the diversity of a batch via Determinantal point processes (DPPs) whose kernels are learned
_automatically_ . This allows us to generalize a previous result as well as prove better regret bounds based
on DPP sampling. Our experiments on a variety of synthetic and real-world robotics and hyper-parameter
optimization tasks indicate that our DPP-based methods, especially those based on DPP sampling,
outperform state-of-the-art methods.

### **1 Introduction**


The optimization of an unknown function based on noisy observations is a fundamental problem in various
real world domains, e.g., engineering design [ 34 ], finance [ 37 ] and hyper-parameter optimization [ 30 ]. In recent
years, an increasingly popular direction has been to model smoothness assumptions about the function via a
Gaussian Process (GP), which provides an easy way to compute the posterior distribution of the unknown
function, and thereby uncertainty estimates that help to decide where to evaluate the function next, in
search of an optima. This _Bayesian optimization_ (BO) framework has received considerable attention in
tuning of hyper-parameters for complex models and algorithms in Machine Learning, Robotics and Computer
Vision [16, 32, 30, 12].
Apart from a few notable exceptions [ 9, 8, 11 ], most methods for Bayesian optimization work by exploring
one parameter value at a time. However, in many applications, it may be possible and, moreover, desirable
to run multiple function evaluations in parallel. A case in point is when the underlying function corresponds
to a laboratory experiment where multiple experimental setups are available or when the underlying function
is the result of a costly computer simulation and multiple simulations can be run across different processors
in parallel. By parallelizing the experiments, substantially more information can be gathered in the same
time-frame; however, future actions must be chosen without the benefit of intermediate results. One might
conceptualize these problems as choosing “batches” of experiments to run simultaneously. The key challenge
is to assemble batches (out of a combinatorially large set of batches) of experiments that both explore the
function and exploit by focusing on regions with high estimated value.


**1.1** **Our Contributions**


Given that functions sampled from GPs usually have some degree of smoothness, in the so-called _batch_
_Bayesian optimization_ (BBO) methods, it is desirable to choose batches which are diverse. Indeed, this is
the motivation behind many popular BBO methods like the BUCB [ 9 ], UCB-PE [ 8 ] and Local Penalization


1


[ 11 ]. Motivated by this long line of work in BBO, we propose a new approach that employs Determinantal
Point Processes (DPPs) to select diverse batches of evaluations. DPPs are probability measures over subsets
of a ground set that promote diversity, have applications in statistical physics and random matrix theory

[ 29, 22 ], and have efficient sampling algorithms [ 18, 19 ]. The two main ways for fixed cardinality subset
selection via DPPs are that of choosing the subset which maximizes the determinant [DPP-MAX, Theorem
3.3] and sampling a subset according to the determinantal probability measure [DPP-SAMPLE, Theorem
3.4]. Following UCB-PE [ 8 ], our methods also choose the first point via an acquisition function, and then
the rest of the points are selected from a relevance region using a DPP. Since DPPs crucially depend on the
choice of the DPP kernel, it is important to choose the right kernel. Our method allows the kernel to change
across iterations and automatically compute it based on the observed data. This kernel is intimately linked
to the GP kernel used to model the function; it is in fact exactly the posterior kernel function of the GP. The
acquisition functions we consider are EST [ 35 ], a recently proposed sequential MAP-estimate based Bayesian
optimization algorithm with regret bounds independent of the size of the domain, and UCB [ 31 ]. In fact, we
show that UCB-PE can be cast into our framework as just being DPP-MAX where the maximization is done
via a greedy selection rule.
Given that DPP-MAX is too greedy, it may be desirable to allow for uncertainty in the observations.
Thus, we define DPP-SAMPLE which selects the batches via sampling subsets from DPPs, and show that
the expected regret is smaller than that of DPP-MAX. To provide a fair comparison with an existing method,
BUCB, we also derive regret bounds for B-EST [Theorem 3.2]. Finally, for all methods with known regret
bounds, the key quantity is the information gain. In the appendix, we also provide a simpler proof of the
information gain for the widely-used RBF kernel which also improves the bound from _O_ (( log _T_ ) _[d]_ [+1] ) [ 27, 31 ]
to _O_ (( log _T_ ) _[d]_ ). We conclude with experiments on synthetic and real-world robotics and hyper-parameter
optimization for extreme multi-label classification tasks which demonstrate that our DPP-based methods,
especially the sampling based ones are superior or competitive with the existing baselines.


**1.2** **Related Work**


One of the key tasks involved in black box optimization is of choosing actions that both explore the
function and exploit our knowledge about likely high reward regions in the function’s domain. This
exploration-exploitation trade-off becomes especially important when the function is expensive to evaluate.
This exploration-exploitation trade off naturally leads to modeling this problem in the multi-armed bandit
paradigm [ 26 ], where the goal is to maximize cumulative reward by optimally balancing this trade-off. Srinivas
_et al_ . [ 31 ] analyzed the Gaussian Process Upper Confidence Bound (GP-UCB) algorithm, a simple and
intuitive Bayesian method [ 3 ] to achieve the first sub-linear regret bounds for Gaussian process bandit
optimization. These bounds however grow logarithmically in the size of the (finite) search space.
Recent work by Wang _et al_ . [ 35 ] considered an intuitive MAP-estimate based strategy (EST) which
involves estimating the maximum value of a function and choosing a point which has maximum probability
of achieving this maximum value. They derive regret bounds for this strategy and show that the bounds
are actually independent of the size of the search space. The problem setting for both UCB and EST
is of optimizing a particular _acquisition function_ . Other popular acquisition functions include expected
improvement (EI), probability of improvement over a certain threshold (PI). Along with these, there is also
work on Entropy search (ES) [ 13 ] and its variant, predictive entropy search (PES) [ 14 ] which instead aims
at minimizing the uncertainty about the location of the optimum of the function. All the fore-mentioned
methods, though, are inherently sequential in nature.
The BUCB and UCB-PE both depend on the crucial observation that the variance of the posterior
distribution does not depend on the actual values of the function at the selected points. They exploit this fact
by “hallucinating” the function values to be as predicted by the posterior mean. The BUCB algorithm chooses
the batch by sequentially selecting the points with the maximum UCB score keeping the mean function the
same and only updating the variance. The problem with this naive approach is that it is too “overconfident”
of the observations which causes the confidence bounds on the function values to shrink very quickly as we go
deeper into the batch. This is fixed by a careful initialization and expanding the confidence bounds which
leads to regret bounds which are worse than that of UCB by some multiplicative factor (independent of T
and B). The UCB-PE algorithm chooses the first point of the batch via the UCB score and then defines a
“relevance region” and selects the remaining points from this region greedily to maximize the _information_


2


_gain_, in order to focus on pure exploration (PE). This algorithm does not require any initialization like the
BUCB and, in fact, achieves better regret bounds than the BUCB.
Both BUCB and UCB-PE, however, are too greedy in their selection of batches which may be really
far from the optimal due to our “immediate overconfidence” of the values. Indeed this is the criticism of
these two methods by a recently proposed BBO strategy PPES [ 28 ], which parallelizes predictive entropy
search based methods and shows considerable improvements over the BUCB and UCB-PE methods. Another
recently proposed method is the Local Penalization (LP) [ 11 ], which assumes that the function is Lipschitz
continuous and tries to estimate the Lipschitz constant. Since assumptions of Lipschitz continuity naturally
allow one to place bounds on how far the optimum of _f_ is from a certain location, they work to smoothly
reduce the value of the acquisition function in a neighborhood of any point reflecting the belief about the
distance of this point to the maxima. However, assumptions of Lipschitzness are too coarse-grained and it
is unclear how their method to estimate the Lipschitz constant and modelling of local penalization affects
the performance from a theoretical standpoint. Our algorithms, in constrast, are general and do not assume
anything about the function other than it being drawn from a Gaussian Process.

### **2 Preliminaries**


**2.1** **Gaussian Process Bandit Optimization**


We address the problem of finding, in the lowest possible number of iterations, the maximum ( _m_ ) of an
unknown function _f_ : _X →_ R where _X ⊂_ R _[d]_, i.e.,


_m_ = _f_ ( _x_ _[∗]_ ) = max _x∈X_ _[f]_ [(] _[x]_ [)] _[.]_


We consider the domain to be discrete as it is well-known how to obtain regret bounds for continous, compact
domains via suitable discretizations [ 31 ]. At each iteration _t_, we choose a batch _{x_ _t,b_ _}_ 1 _≤b≤B_ of _B_ points and
then simultaneously observe the noisy values taken by _f_ at these points, _y_ _t,b_ = _f_ ( _x_ _t,b_ ) + _ϵ_ _t,b_, where _ϵ_ _t,k_ is
i.i.d. Gaussian noise _N_ (0 _, σ_ [2] ). The function is assumed to be drawn from a Gaussian process (GP), i.e.,
_f ∼_ _GP_ (0 _, k_ ), where _k_ : _X_ [2] _→_ R + is the kernel function. Given the observations _D_ _t_ = _{_ ( _x_ _τ_ _, y_ _τ_ ) _[t]_ _τ_ =1 _[}]_ [ up to time]
_t_, we obtain the posterior mean and covariance functions [ 25 ] via the kernel matrix _K_ _t_ = [ _k_ ( _x_ _i_ _, x_ _j_ )] _x_ _i_ _,x_ _j_ _∈D_ _t_ and
**k** **t** ( _x_ ) = [ _k_ ( _x_ _i_ _, x_ )] _x_ _i_ _∈D_ _t_ : _µ_ _t_ ( _x_ ) = **k** **t** ( _x_ ) _[T]_ ( _K_ _t_ + _σ_ [2] _I_ ) _[−]_ [1] **y** **t** and _k_ _t_ ( _x, x_ _[′]_ ) = _k_ ( _x, x_ _[′]_ ) _−_ **k** **t** ( _x_ ) _[T]_ ( _K_ _t_ + _σ_ [2] _I_ ) _[−]_ [1] **k** **t** ( _x_ _[′]_ ).
The posterior variance is given by _σ_ _t_ [2] [(] _[x]_ [) =] _[ k]_ _[t]_ [(] _[x, x]_ [). Define the Upper Confidence Bound (UCB)] _[ f]_ [ +] [ and]
Lower Confidence Bound (LCB) _f_ _[−]_ as


_f_ _t_ [+] [(] _[x]_ [) =] _[ µ]_ _[t][−]_ [1] [(] _[x]_ [) +] _[ β]_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ ) _f_ _t_ _[−]_ [(] _[x]_ [) =] _[ µ]_ _[t][−]_ [1] [(] _[x]_ [)] _[ −]_ _[β]_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ )


A crucial observation made in BUCB [ 9 ] and UCB-PE [ 8 ] is that the posterior covariance and variance
functions do not depend on the actual function values at the set of points. The EST algorithm in [ 35 ] chooses
at each timestep _t_,the point which has the maximum posterior probability of attaining the maximum value
_m_, i.e., the arg max _x∈X_ Pr ( _M_ _x_ _|m, D_ _t_ ) where _M_ _x_ is the event that point _x_ achieves the maximum value. This
turns out to be equal to arg min _x∈X_ � ( _m −_ _µ_ _t_ ( _x_ )) _/σ_ _t_ ( _x_ ) � . Note that this actually depends on the value of _m_
which, in most cases, is unknown. [ 35 ] get around this by using an approximation ˆ _m_ which, under certain
conditions specified in their paper, is an upper bound on _m_ . They provide two ways to get the estimate ˆ _m_,
namely ESTa and ESTn. We refer the reader to [ 35 ] for details of the two estimates and refer to ESTa as
EST.
Assuming that the horizon _T_ is unknown, a strategy has to be good at any iteration. Let _r_ _t,b_ denote
the _simple regret_, the difference between the value of the maxima and the point queried _x_ _t,k_, i.e., _r_ _t,b_ =
max _x∈X_ _f_ ( _x_ ) _−_ _f_ ( _x_ _t,b_ ). While, UCB-PE aims at minimizing a batched cumulative regret, in this paper we
will focus on the standard full cumulative regret defined as _R_ _T B_ = [�] _[T]_ _t_ =1 � _Bb_ =1 _[r]_ _[t,b]_ [. This models the case]
where all the queries in a batch should have low regret. The key quantity controlling the regret bounds of all
known BO algorithms is the maximum mutual information that can be gained about _f_ from _T_ measurements
1
: _γ_ _T_ = max _A⊆X_ _,|A|≤T_ _I_ ( _y_ _A_ _, f_ _A_ ) = max _A⊆X_ _,|A|≤T_ 2 [log det] [(] _[I]_ [ +] _[ σ]_ _[−]_ [2] _[K]_ _[A]_ [), where] _[ K]_ _[A]_ [ is the (square) submatrix]
of _K_ formed by picking the row and column indices corresponding to the set _A_ . The regret for both the UCB
and the EST algorithms are presented in the following theorem which is a combination of Theorem 1 in [ 31 ]
and Theorem 3.1 in [35].


3


**Algorithm 1** GP-BUCB/B-EST Algorithm


**Input:** Decision set _X_, GP prior _µ_ 0 _, σ_ 0, kernel function _k_ ( _·, ·_ ), feedback mapping _fb_ [ _·_ ]
**for** _t_ = 1 **to** TB **do**



Choose _β_ _t_ [1] _[/]_ [2] =



_C_ _[′]_ [�] 2 log( _|X|π_ [2] _t_ [2] _/_ 6) _δ_ � for BUCB
� _C_ _[′]_ [�] min _x∈X_ ( ˆ _m −_ _µ_ _fb_ [ _t_ ] ) _/σ_ _t−_ 1 ( _x_ )� for B-EST



Choose _x_ _t_ = arg max _x∈X_ [ _µ_ _fb_ [ _t_ ] ( _x_ ) + _β_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ )] and compute _σ_ _t_ ( _·_ )
**if** _fb_ [ _t_ ] _< fb_ [ _t_ + 1] **then**

Obtain _y_ _t_ _′_ = _f_ ( _x_ _t_ _′_ ) + _ϵ_ _t_ _′_ for _t_ _[′]_ _∈{fb_ [ _t_ ] + 1 _, . . ., fb_ [ _t_ + 1] _}_ and compute _µ_ _fb_ [ _t_ +1] ( _·_ )
**end if**

**end for**

**return** arg max
_t_ =1 _...T B_ _[y]_ _[t]_


**Theorem 2.1.** _Let_ _C_ = 2 _/_ _m_ ˆ log _−µ_ (1 + _t−_ 1 ( _x_ _σ_ ) _[−]_ [2] ) _and fix_ _δ >_ 0 _. For UCB, choose_ _β_ _t_ = 2 log ( _|X|t_ [2] _π_ [2] _/_ 6 _δ_ ) _and for_
_EST, choose_ _β_ _t_ = ( min _x∈X_ _σ_ _t−_ 1 ( _x_ ) ) [2] _and_ _ζ_ _t_ = 2 log ( _π_ [2] _t_ [2] _/δ_ ) _. With probability_ 1 _−_ _δ_ _, the cumulative regret_

_up to any time step T can be bounded as_



~~_√_~~ _CTβ_ _T_ _γ_ _T_ _for UCB_
� ~~_√_~~ _CTγ_ _T_ ( _β_ _t_ 1 _[∗]_ _/_ 2 + _ζ_ _T_ [1] _[/]_ [2] ) _for EST_ _[where][ t]_ _[∗]_ [= arg max] _t_ _β_ _t_ _._



_R_ _T_ =



_T_
� _r_ _t_ _≤_


_t_ =1



**2.2** **Determinantal Point Processes**


Given a DPP kernel _K ∈_ R _[m][×][m]_ of _m_ elements _{_ 1 _, . . ., m}_, the _k_ -DPP distribution defined on 2 _[Y]_ is defined
as picking _B_, a _k_ -subset of [ _m_ ] with probability proportional to det( _K_ _B_ ). Formally,


det( _K_ _B_ )
Pr( _B_ ) =
~~�~~ _|S|_ = _k_ [det(] _[K]_ _[S]_ [)]


The problem of picking a set of size _k_ which maximizes the determinant and sampling a set according to the
_k_ -DPP distribution has received considerable attention [ 23, 7, 6, 10, 1, 18 ]. The maximization problem in
general is NP-hard and furthermore, has a hardness of approximation result of 1 _/c_ _[k]_ for some _c >_ 1. The
best known approximation algorithm is by [ 23 ] with a factor of 1 _/e_ _[k]_, which almost matches the lower bound.
Their algorithm however is a complicated and expensive convex program. A simple greedy algorithm on the
other hand gives a 1 _/_ 2 _[k]_ [ log(] _[k]_ [)] -approximation. For sampling from _k_ -DPPs, an exact sampling algorithm exists
due to [ 10 ]. This, however, does not scale to large datasets. A recently proposed alternative is an MCMC
based method by [1] which is much faster.

### **3 Main Results**


In this section, we present our DPP-based algorithms. For a fair comparison of the various methods, we
first prove the regret bounds of the EST version of BUCB, i.e., B-EST. We then show the equivalence
between UCB-PE and UCB-DPP maximization along with showing regret bounds for the EST version of
PE/DPP-MAX. We then present the DPP sampling (DPP-SAMPLE) based methods for UCB and EST and
provide regret bounds. In Appendix 4, while borrowing ideas from [ 27 ], we provide a simpler proof with
improved bounds on the maximum information gain for the RBF kernel.


**3.1** **The Batched-EST algorithm**


The BUCB has a feedback mapping _fb_ which indicates that at any given time _t_ (just in this case we will
mean a total of _TB_ timesteps), the iteration upto which the actual function values are available. In the
batched setting, this is just _⌊_ ( _t −_ 1) _/B⌋B_ . The BUCB and B-EST, its EST variant algorithms are presented
in Algorithm 1. The algorithm mainly comes from the observation made in [ 35 ] that the point chosen by
EST is the same as a variant of UCB. This is presented in the following lemma.


4


**Lemma 3.1.** _(Lemma 2.1 in [_ _35_ _]) At any timestep_ _t_ _, the point selected by EST is the same as the point_
_selected by a variant of UCB with β_ _t_ [1] _[/]_ [2] = min _x∈X_ ( ˆ _m −_ _µ_ _t−_ 1 ( _x_ )) _/σ_ _t−_ 1 ( _x_ ) _._


This will be sufficient to get to B-EST as well by just running BUCB with the _β_ _t_ as defined in Lemma
3.1 and is also provided in Algorithm 1. In the algorithm, _C_ _[′]_ is chosen to be _exp_ (2 _C_ ), where _C_ is an upper
bound on the maximum conditional mutual information _I_ ( _f_ ( _x_ ); _y_ _fb_ [ _t_ ]+1: _t−_ 1 _|y_ 1: _fb_ [ _t_ ] ) (refer to [ 9 ] for details).
The problem with naively using this algorithm is that the value of _C_ _[′]_, and correspondingly the regret bounds,
usually has at least linear growth in _B_ . This is corrected in [ 9 ] by two-stage BUCB which first chooses an
initial batch of size _T_ _[init]_ by greedily choosing points based on the (updated) posterior variances. The values
are then obtained and the posterior GP is calculated which is used as the prior GP in Algorithm 1. The _C_ _[′]_

value can then be chosen independent of _B_ . We refer the reader to the Table 1 in [ 9 ] for values of _C_ _[′]_ and
_T_ _[init]_ for common kernels. Finally, the regret bounds of B-EST are presented in the next theorem.

**Theorem 3.2.** _Choose_ _α_ _t_ = � min _x∈X_ _m_ ˆ _−σ_ _t_ _µ_ _−f_ 1 _b_ ( [ _t_ _x_ ] )( _x_ ) � 2 _and_ _β_ _t_ = ( _C_ _′_ ) 2 _α_ _t_ _,_ _B ≥_ 2 _, δ >_ 0 _and the_ _C_ _′_ _and_ _T_ _init_

_values are chosen according to Table 1 in [_ _9_ _]. At any timestep_ _T_ _, let_ _R_ _T_ _be the cumulative regret of the_
_two-stage initialized B-EST algorithm. Then_


_Pr{R_ _T_ _≤_ _C_ _[′]_ _R_ _T_ _[seq]_ + 2 _∥f_ _∥_ _∞_ _T_ _[init]_ _, ∀T ≥_ 1 _} ≥_ 1 _−_ _δ_


_Proof._ The proof is presented in Appendix 1.


**3.2** **Equivalence of Pure Exploration (PE) and DPP Maximization**


We now present the equivalence between the Pure Exploration and a procedure which involves DPP
maximization based on the Greedy algorithm. For the next two sections, by an iteration, we mean all _B_
points selected in that iteration and thus, _µ_ _t−_ 1 and _k_ _t−_ 1 are computed using ( _t −_ 1) _B_ observations that are
available to us. We first describe a generic framework for BBO inspired by UCB-PE : At any iteration, the
first point is chosen by selecting the one which maximizes UCB or EST which can be seen as a variant of
UCB as per Lemma 3.1. A relevance region _R_ [+] _t_ [is defined which contains] [ arg max] _[x][∈X]_ _[f]_ _t_ [ +] +1 [(] _[x]_ [) with high]
probability. Let _y_ _t_ _[•]_ [=] _[ f]_ _t_ _[ −]_ [(] _[x]_ _[•]_ _t_ [),][ where] _[ x]_ _[•]_ _t_ [=] [ arg max] _[x][∈X]_ _[f]_ _t_ _[ −]_ [(] _[x]_ [). The relevance region is formally defined as]
_R_ [+] _t_ [=] _[ {][x][ ∈X|][µ]_ _[t][−]_ [1] [+ 2] � _β_ _t_ +1 _σ_ _t−_ 1 ( _x_ ) _≥_ _y_ _t_ _[•]_ _[}]_ [. The intuition for considering this region is that using] _[ R]_ [+] _t_

guarantees that the queries at iteration _t_ will leave an impact on the future choices at iteration _t_ + 1. The
next _B −_ 1 points for the batch are then chosen from _R_ [+] _t_ [, according to some rule. In the special case of]
UCB-PE, the _B −_ 1 points are selected greedily from _R_ [+] _t_ [by maximizing the (updated) posterior variance,]
while keeping the mean function the same. Now, at the _t_ _[th]_ iteration, consider the posterior kernel function
after _x_ _t,_ 1 has been chosen (say _k_ _t,_ 1 ) and consider the kernel matrix _K_ _t,_ 1 = _I_ + _σ_ _[−]_ [2] [ _k_ _t,_ 1 ( _p_ _i_ _, p_ _j_ )] _i,j_ over the
points _p_ _i_ _∈R_ [+] _t_ [. We will consider this as our DPP kernel at iteration] _[ t]_ [. Two possible ways of choosing] _[ B][ −]_ [1]
points via this DPP kernel is to either choose the subset of size _B −_ 1 of maximum determinant (DPP-MAX)
or sample a set from a ( _B −_ 1)-DPP using this kernel (DPP-SAMPLE). In this subsection, we focus on the
maximization problem. The proof of the regret bounds of UCB-PE go through a few steps but in one of
the intermediate steps (Lemma 5 of [ 8 ]), it is shown that the sum of regrets over a batch at an iteration _t_ is
upper bounded as



_B_
� _r_ _t,b_ _≤_


_b_ =1



_B_
�( _σ_ _t,b_ ( _x_ _t,b_ )) [2] _≤_


_b_ =1



_B_
�



_B_ _B_
� _C_ 2 _σ_ [2] log(1 + _σ_ _[−]_ [2] _σ_ _t,b_ ( _x_ _t,b_ )) = _C_ 2 _σ_ [2] log �

_b_ =1 � _b_ =1



�(1 + _σ_ _[−]_ [2] _σ_ _t,b_ ( _x_ _t,b_ )

_b_ =1 �



where _C_ 2 = _σ_ _[−]_ [2] _/_ log (1 + _σ_ _[−]_ [2] ). From the final log-product term, it can be seen (from Schur’s determinant
identity [ 5 ] and the definition of _σ_ _t,b_ ( _x_ _t,b_ ) ) that the product of the last _B −_ 1 terms is exactly the _B −_ 1
principal minor of _K_ _t,_ 1 formed by the indices corresponding to _S_ = _{x_ _t,b_ _}_ _[B]_ _b_ =2 [. Thus, it is straightforward to]
see that the UCB-PE algorithm is really just ( _B −_ 1)-DPP maximization via the greedy algorithm. This


connection will also be useful in the next subsection for DPP-SAMPLE. Thus, [�] _[B]_ _b_ =1 _[r]_ _[t,b]_ _[ ≤]_ _[C]_ [2] _[σ]_ [2] log (1 +
�


_σ_ _[−]_ [2] _σ_ _t,_ 1 ( _x_ _t,_ 1 )) + log det (( _K_ _t,_ 1 ) _S_ ) . Finally, for EST-PE, the proof proceeds like in the B-EST case by realising
�

that EST is just UCB with an adaptive _β_ _t_ . The final algorithm (along with its sampling counterpart; details
in the next subsection) is presented in Algorithm 2. The procedure kDPPMaxGreedy( _K, k_ ) picks a principal
submatrix of _K_ of size _k_ by the greedy algorithm. Finally, we have the theorem for the regret bounds for
(UCB/EST)-DPP-MAX.


5


**Algorithm 2** GP-(UCB/EST)-DPP-(MAX/SAMPLE) Algorithm


**Input:** Decision set _X_, GP prior _µ_ 0 _, σ_ 0, kernel function _k_ ( _·, ·_ )
**for** _t_ = 1 **to** T **do**


Compute _µ_ _t−_ 1 and _σ_ _t−_ 1 according to Bayesian inference.

Choose _β_ _t_ [1] _[/]_ [2] = 2 log( _|X|π_ [2] _t_ [2] _/_ 6) _δ_ � for UCB

��� min _x∈X_ ( ˆ _m −_ _µ_ _fb_ [ _t_ ] ) _/σ_ _t−_ 1 ( _x_ )� for EST
_x_ _t,_ 1 _←_ arg max _x∈X_ _µ_ _t−_ 1 ( _x_ ) + _[√]_ _β_ _t_ _σ_ _t−_ 1 ( _x_ )
Compute _R_ [+] _t_ [and construct the DPP kernel] _[ K]_ _[t,]_ [1]

kDPPMaxGreedy( _K_ _t,_ 1 _, B −_ 1) for DPP-MAX

_{x_ _t,b_ _}_ _[B]_ _b_ =2 _[←]_ �kDPPSample( _K_ _t,_ 1 _, B −_ 1) for DPP-SAMPLE

Obtain _y_ _t,b_ = _f_ ( _x_ _t,b_ ) + _ϵ_ _t,b_ for _b_ = 1 _, . . ., B_
**end for**


**Theorem 3.3.** _At iteration_ _t_ _, let_ _β_ _t_ = 2 log ( _|X|π_ [2] _t_ [2] _/_ 6 _δ_ ) _for UCB,_ _β_ _t_ = ( min _[m]_ [ˆ] _σ_ _[−]_ _t_ _[µ]_ _−_ _[t]_ 1 _[−]_ ( [1] _x_ [(] ) _[x]_ [)] [)] [2] _[ and]_ _[ ζ]_ _[t]_ [ =]

2 log ( _π_ [2] _t_ [2] _/_ 3 _δ_ ) _for EST,_ _C_ 1 = 36 _/_ log (1 + _σ_ _[−]_ [2] ) _and fix_ _δ >_ 0 _, then, with probability_ _≥_ 1 _−_ _δ_ _the full_
_cumulative regret_ _R_ _T B_ _incurred by UCB-DPP-MAX is_ _R_ _T B_ _≤_ _[√]_ _C_ 1 _TBβ_ _T_ _γ_ _T B_ _}_ _and that for EST-DPP-MAX_
_is R_ _T B_ _≤_ _[√]_ _C_ 1 _TBγ_ _T B_ ( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T_ [1] _[/]_ [2] ) _._


_Proof._ The proof is provided in Appendix 2. It should be noted that the term inside the logarithm in _ζ_ _t_ has
been multiplied by 2 as compared to the sequential EST, which has a union bound over just one point, _x_ _t_ .
This happens because we will need a union bound over not just _x_ _t,b_ but also _x_ _[•]_ _t_ [.]


**3.3** **Batch Bayesian Optimization via DPP Sampling**


In the previous subsection, we looked at the regret bounds achieved by DPP maximization. One natural
question to ask is whether the other subset selection method via DPPs, namely DPP sampling, gives us
equivalent or better regret bounds. Note that in this case, the regret would have to be defined as expected
regret. The reason to believe this is well-founded as indeed sampling from _k_ -DPPs results in better results,
in both theory and practice, for low-rank matrix approximation [ 10 ] and exemplar-selection for Nystrom
methods [ 20 ]. Keeping in line with the framework described in the previous subsection, the subset to be
selected has to be of size _B −_ 1 and the kernel should be _K_ _t,_ 1 at any iteration _t_ . Instead of maximizing, we
can choose to sample from a ( _B −_ 1)-DPP. The algorithm is described in Algorithm 2. The kDPPSample(K _,_ k)
procedure denotes sampling a set from the _k_ -DPP distribution with kernel _K_ . The question then to ask is
what is the expected regret of this procedure. In this subsection, we show that the expected regret bounds of
DPP-SAMPLE are less than the regret bounds of DPP-MAX and give a quantitative bound on this regret
based on entropy of DPPs. By entropy of a _k_ -DPP with kernel _K_, _H_ ( _k −_ DPP ( _K_ )), we simply mean the
standard definition of entropy for a discrete distribution. Note that the entropy is always non-negative in this
case. Please see Appendix 3 for details. For brevity, since we always choose _B −_ 1 elements from the DPP,
we denote _H_ ( _DPP_ ( _K_ )) to be the entropy of ( _B −_ 1)-DPP for kernel _K_ .


**Theorem 3.4.** _The regret bounds of DPP-SAMPLE are less than that of DPP-MAX. Furthermore, at_
_iteration_ _t_ _, let_ _β_ _t_ = 2 log ( _|X|π_ [2] _t_ [2] _/_ 6 _δ_ ) _for UCB,_ _β_ _t_ = ( min _[m]_ [ˆ] _σ_ _[−]_ _t_ _[µ]_ _−_ _[t]_ 1 _[−]_ ( [1] _x_ [(] ) _[x]_ [)] [)] [2] _[ and]_ _[ ζ]_ _[t]_ [ = 2] [ log] [(] _[π]_ [2] _[t]_ [2] _[/]_ [3] _[δ]_ [)] _[ for EST,]_

_C_ 1 = 36 _/_ log (1 + _σ_ _[−]_ [2] ) _and fix_ _δ >_ 0 _, then the expected full cumulative regret of UCB-DPP-SAMPLE satisfies_



_T_
� _H_ ( _DPP_ ( _K_ _t,_ 1 )) + _B_ log( _|X|_ )

_t_ =1 �



_R_ _T B_ [2] _[≤]_ [2] _[TBC]_ 1 _[β]_ _T_


_and that for EST-DPP-SAMPLE satisfies_



_γ_ _T B_ _−_
�



_R_ _T B_ [2] _[≤]_ [2] _[TBC]_ 1 [(] _[β]_ _t_ [1] _[/]_ [2] + _ζ_ _t_ [1] _[/]_ [2] ) [2] _γ_ _T B_ _−_
�


_Proof._ The proof is provided in Appendix 3.


6



_T_
� _H_ ( _DPP_ ( _K_ _t,_ 1 )) + _B_ log( _|X|_ )

_t_ =1 �


Figure 1: Immediate regret of the algorithms on two synthetic functions with B = 5 and 10


Note that the regret bounds for both DPP-MAX and DPP-SAMPLE are better than BUCB/B-EST due
to the latter having both an additional factor of _B_ in the log term and a regret multiplier constant _C_ _[′]_ . In
fact, for the RBF kernel, _C_ _[′]_ grows like _e_ _[d]_ _[d]_ which is quite large for even moderate values of _d_ .

### **4 Experiments**


In this section, we study the performance of the DPP-based algorithms, especially DPP-SAMPLE against
some existing baselines. In particular, the methods we consider are BUCB [ 9 ], B-EST, UCB-PE/UCBDPP-MAX [ 8 ], EST-PE/EST-DPP-MAX, UCB-DPP-SAMPLE, EST-DPP-SAMPLE and UCB with local
penalization (LP-UCB) [ 11 ]. We used the publicly available code for BUCB and PE [1] . The code was modified
to include the code for the EST counterparts using code for EST [2] . For LP-UCB, we use the publicly available
GPyOpt codebase [3] and implemented the MCMC algorithm by [ 1 ] for _k_ -DPP sampling with _ϵ_ = 0 _._ 01 as the
variation distance error. We were unable to compare against PPES as the code was not publicly available.
Furthermore, as shown in the experiments in [ 28 ], PPES is very slow and does not scale beyond batch sizes
of 4-5. Since UCB-PE almost always performs better than the simulation matching algorithm of [ 4 ] in all
experiments that we could find in previous papers [ 28, 8 ], we forego a comparison against simulation matching
as well to avoid clutter in the graphs. The performance is measured after _t_ batch evaluations using _immediate_

�
_regret_, _r_ _t_ = _|f_ ( _x_ _t_ ) _−_ _f_ ( _x_ _[∗]_ ) _|_, where _x_ _[∗]_ is a known optimizer of _f_ and � _x_ _t_ is the recommendation of an algorithm
after _t_ batch evaluations. We perform 50 experiments for each objective function and report the median of
the immediate regret obtained for each algorithm. To maintain consistency, the first point of all methods
is chosen to be the same (random). The mean function of the prior GP was the zero function while the
kernel function was the squared-exponential kernel of the form _k_ ( _x, y_ ) = _γ_ [2] exp [ _−_ 0 _._ 5 [�] _d_ [(] _[x]_ _[d]_ _[ −]_ _[y]_ _d_ [2] [)] _[/l]_ _d_ [2] []. The]

hyper-parameter _λ_ was picked from a broad Gaussian hyperprior and the the other hyper-parameters were
chosen from uninformative Gamma priors.


1 http://econtal.perso.math.cnrs.fr/software/
2 https://github.com/zi-w/EST
3 http://sheffieldml.github.io/GPyOpt/


7


Figure 2: Immediate regret of the algorithms for Prec@1 for FastXML on Bibtex and `Robot` with B = 5 and 10


Our first set of experiments is on a set of synthetic benchmark objective functions including Branin-Hoo

[ 21 ], a mixture of cosines [ 2 ] and the Hartmann-6 function [ 21 ]. We choose batches of size 5 and 10. Due to
lack of space, the results for mixture of cosines are provided in Appendix 5 while the results of the other two
are shown in Figure 1. The results suggest that the DPP-SAMPLE based methods perform superior to the
other methods. They do much better than their DPP-MAX and Batched counterparts. The trends displayed
with regards to LP are more interesting. For the Branin-Hoo, LP-UCB starts out worse than the DPP based
algorithms but takes over DPP-MAX relatively quickly and approaches the performance of DPP-SAMPLE
when the batch size is 5. When the batch size is 10, the performance of LP-UCB does not improve much but
both DPP-MAX and DPP-SAMPLE perform better. For Hartmann, LP-UCB outperforms both DPP-MAX
algorithms by a considerable margin. The DPP-SAMPLE based methods perform better than LP-UCB.
The gap, however, is more for the batch size of 10. Again, the performance of LP-UCB changes much lesser
compared to the performance gain of the DPP-based algorithms. This is likely because the batches chosen by
the DPP-based methods are more “globally diverse” for larger batch sizes. The superior performance of the
sampling based methods can be attributed to allowing for uncertainty in the observations by sampling as
opposed to greedily emphasizing on maximizing information gain.
We now consider maximization of real-world objective functions. The first function we consider, `robot`,
returns the walking speed of a bipedal robot [ 36 ]. The function’s input parameters, which live in [0 _,_ 1] [8], are
the robot’s controller. We add Gaussian noise with _σ_ = 0 _._ 1 to the noiseless function. The second function,
`Abalone` [4] is a test function used in [ 8 ]. The challenge of the dataset is to predict the age of a species of
sea snails from physical measurements. Similar to [ 8 ], we will use it as a maximization problem. Our final
experiment is on hyper-parameter tuning for extreme multi-label learning. In extreme classification, one
needs to deal with multi-class and multi-label problems involving a very large number of categories. Due to
the prohibitively large number of categories, running traditional machine learning algorithms is not feasible.
A recent popular approach for extreme classification is the FastXML algorithm [ 24 ]. The main advantage
of FastXML is that it maintains high accuracy while training in a fraction of the time compared to the


4 The Abalone dataset is provided by the UCI Machine Learning Repository at
```
http://archive.ics.uci.edu/ml/datasets/Abalone

```

8


previous state-of-the-art. The FastXML algorithm has 5 parameters and the performance depends on these
hyper-parameters, to a reasonable amount. Our task is to perform hyper-parameter optimization on these 5
hyper-parameters with the aim to maximize the Precision@k for _k_ = 1, which is the metric used in [ 24 ] to
evaluate the performance of FastXML compared to other algorithms as well. While the authors of [ 24 ] run
extensive tests on a variety of datasets, we focus on two small datasets : Bibtex [ 15 ] and Delicious[ 33 ]. As
before, we use batch sizes of 5 and 10. The results for Abalone and the FastXML experiment on Delicious
are provided in the appendix. The results for Prec@1 for FastXML on the Bibtex dataset and for the `robot`
experiment are provided in Figure 2. The blue horizontal line for the FastXML results indicates the maximum
Prec@k value found using grid search.
The results for `robot` indicate that while DPP-MAX does better than their Batched counterparts, the
difference in the performance between DPP-MAX and DPP-SAMPLE is much less pronounced for a small
batch size of 5 but is considerable for batch sizes of 10. This is in line with our intuition about sampling being
more beneficial for larger batch sizes. The performance of LP-UCB is quite close and slightly better than
UCB-DPP-SAMPLE. This might be because the underlying function is well-behaved (Lipschitz continuous)
and thus, the estimate for the Lipschitz constant might be better which helps them get better results. This
improvement is more pronounced for batch size of 10 as well. For Abalone (see Appendix 5), LP does better
than DPP-MAX but there is a reasonable gap between DPP-SAMPLE and LP which is more pronounced for
_B_ = 10.

The results for Prec@1 for the Bibtex dataset for FastXML are more interesting. Both DPP based
methods are much better than their Batched counterparts. For _B_ = 5, DPP-SAMPLE is only slightly better
than DPP-MAX. LP-UCB starts out worse than DPP-MAX but starts doing comparable to DPP-MAX after
a few iterations. For _B_ = 10, there is not a large improvement in the gap between DPP-MAX and DPPSAMPLE. LP-UCB however, quickly takes over UCB-DPP-MAX and comes quite close to the performance
of DPP-SAMPLE after a few iterations. For the Delicious dataset (see Appendix 5), we see a similar trend
of the improvement of sampling to be larger for larger batch sizes. LP-UCB displays an interesting trend
in this experiment by doing much better than UCB-DPP-MAX for _B_ = 5 and is in fact quite close to the
performance of DPP-SAMPLE. However, for _B_ = 10, its performance is much closer to UCB-DPP-MAX.
DPP-SAMPLE loses out to LP-UCB only on the `robot` dataset and does better for all the other datasets.
Furthermore, this improvement seems more pronounced for larger batch sizes. We leave experiments with
other kernels and a more thorough experimental evaluation with respect to batch sizes for future work.

### **5 Conclusion**


We have proposed a new method for batched Gaussian Process bandit (batch Bayesian) optimization based on
DPPs which are desirable in this case as they promote diversity in batches. The DPP kernel is automatically
figured out on the fly which allows us to show regret bounds for DPP maximization and sampling based
methods for this problem. We show that this framework exactly recovers a popular algorithm for BBO,
namely the UCB-PE when we consider DPP maximization using the greedy algorithm. We showed that the
regret for the sampling based method is always less than the maximization based method. We also derived
their EST counterparts and also provided a simpler proof of the information gain for RBF kernels which leads
to a slight improvement in the best bound known. Our experiments on a variety of synthetic and real-world
tasks validate our theoretical claims that sampling performs better than maximization and other methods.

### **References**


[1] N. Anari, S.O. Gharan, and A. Rezaei. Monte carlo markov chains algorithms for sampling strongly
rayleigh distributions and determinantal point processes. _COLT_, 2016.


[2] B.S. Anderson, A.W. Moore, and D. Cohn. A nonparametric approach to noisy and costly optimization.
_ICML_, 2000.


[3] P. Auer. Using confidence bounds for exploration-exploitation trade-offs. _JMLR_, 3:397–422, 2002.


[4] J. Azimi, A. Fern, and X. Fern. Batch bayesian optimization via simulation matching. 2010.


9


[5] R. Brualdi and H. Schneider. Determinantal identities: Gauss, schur, cauchy, sylvester, kronecker, jacobi,
binet, laplace, muir, and cayley. _Linear Algebra and its Applications_, 1983.


[6] A. C¸ ivril and M. Magdon-Ismail. On selecting a maximum volume sub-matrix of a matrix and related
problems. _Theor. Comput. Sci._, 410(47-49):4801–4811, 2009.


[7] A. C¸ ivril and M. Magdon-Ismail. Exponential inapproximability of selecting a maximum volume
sub-matrix. _Algorithmica_, 65(1):159–176, 2013.


[8] E. Contal, D. Buffoni, D. Robicquet, and N. Vayatis. Parallel gaussian process optimization with upper
confidence bound and pure exploration. _ECML_, 2013.


[9] T. Desautels, A. Krause, and J.W. Burdick. Parallelizing exploration-exploitation tradeoffs in gaussian
process bandit optimization. _JMLR_, 15:4053–4103, 2014.


[10] A. Deshpande and L. Rademacher. Efficient volume sampling for row/column subset selection. _FOCS_,
2010.


[11] J. Gonzalez, Z. Dai, P. Hennig, and N. Lawrence. Batch bayesian optimization via local penalization.
_AISTATS_, 2016.


[12] J. Gonz´alez, M. A. Osborne, and N. D. Lawrence. GLASSES: relieving the myopia of bayesian optimisation.
_AISTATS_, 2016.


[13] P. Hennig and C. Schuler. Entropy search for information-efficient global optimization. _JMLR_, 13, 2012.


[14] J.M. Hernandex-Lobato, M.W. Hoffman, and Z. Ghahramani. Predicitive entropy search for efficient
global optimization of black-box functions. _NIPS_, 2014.


[15] I. Katakis, G. Tsoumakas, and I. Vlahavas. Multilabel text classification for automated tag suggestion.
_ECML/PKDD Discovery Challenge_, 2008.


[16] A. Krause and C. S. Ong. Contextual gaussian process bandit optimization. _NIPS_, 2011.


[17] Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-Optimal Sensor Placements in Gaussian Processes:
Theory, Efficient Algorithms and Empirical Studies. _J. Mach. Learn. Res._, 9:235–284, 2008.


[18] Alex Kulesza and Ben Taskar. k-dpps: Fixed-size determinantal point processes. In _ICML_, 2011.


[19] Alex Kulesza and Ben Taskar. Determinantal Point Processes for Machine Learning. _Found. Trends_
_Mach. Learn._, (2-3):123–286, 2012.


[20] C. Li, S. Jegelka, and S. Sra. Fast dpp sampling for nystrm with application to kernel methods. _ICML_,
2016.


[21] D. Lizotte. Pratical bayesian optimization. _PhD thesis, University of Alberta_, 2008.


[22] R. Lyons. Determinantal probability measures. _Publications Math´ematiques de l’Institut des Hautes_
_´Etudes Scientifiques_, 98(1):167–212, 2003.


[23] A. Nikolov. Randomized rounding for the largest simplex problem. In _STOC_, pages 861–870, 2015.


[24] Y. Prabhu and M. Varma. Fastxml: A fast, accurate and stable tree-classifier for extreme multi-label
learning. _KDD_, 2014.


[25] C. Rasmussen and C. Williams. Gaussian processes for machine learning. _MIT Press_, 2008.


[26] H. Robbins. Some aspects of the sequential design of experiments. _Bul. Am. Math. Soc._, 1952.


[27] M. W. Seeger, S. M. Kakade, and D. P. Foster. Information consistency of nonparametric gaussian
process methods. _IEEE Tr. Inf. Theo._, 54(5):2376–2382, 2008.


[28] A. Shah and Z. Ghahramani. Parallel predictive entropy search for batch global optimization of expensive
objective functions. _NIPS_, 2015.


[29] T. Shirai and Y. Takahashi. Random point fields associated with certain fredholm determinants i:
fermion, poisson and boson point processes. _Journal of Functional Analysis_, 205(2):414 – 463, 2003.


[30] J. Snoek, H. Larochelle, and R.P. Adams. Practical bayesian optimization of machine learning. _NIPS_,
2012.


10


[31] N. Srinivas, A. Krause, S. Kakade, and M. Seeger. Information-theoretic regret bounds for gaussian
process optimization in the bandit setting. _IEEE Transactions on Information Theory_, 58(5):3250–3265,
2012.


[32] C. Thornton, F. Hutter, H. H. Hoos, and K. Leyton-Brown. Auto-weka : combined selection and
hyper-parameter optimization of classification algorithms. _KDD_, 2003.


[33] G. Tsoumakas, I. Katakis, and I. Vlahavas. Effective and efficient multilabel classification in domains
with large number of labels. _ECML/PKDD 2008 Workshop on Mining Multidimensional Data_, 2008.


[34] G. Wang and S. Shan. Review of metamodeling techniques in support of engineering design optimization.
_Journal of Mechanical Design_, 129:370–380, 2007.


[35] Z. Wang, B. Zhou, and S. Jegelka. Optimization as estimation with gaussian processes in bandit settings.
_AISTATS_, 2016.


[36] E. Westervelt and J. Grizzle. Feedback control of dynamic bipedal robot locomotion. _Control and_
_Automation Series_, 2007.


[37] W. Ziemba and R. Vickson. Stochastic optimization models in finance. _World Scientific Singapore_, 2006.

### **6 APPENDIX**


**6.1** **The Batched-EST Algorithm**


The proofs for B-EST are relatively straightforward which follow from combining the proofs of [ 9 ] and [ 35 ].
We provide them here for completeness. We first need a series of supporting lemmas which are variants of
the lemmas of UCB for EST. These require different bounds than the ones for BUCB.


**Lemma 6.1.** _(Lemma 3.2 in [_ _35_ _]) Pick_ _δ ∈_ (0 _,_ 1) _and set_ _ζ_ _t_ = 2 log ( _π_ [2] _t_ [2] _/_ 6 _δ_ ) _. Then, for an arbitrary_
_sequence of actions x_ 1 _, x_ 2 _, . . . ∈X_ _,_


Pr[ _|f_ ( _x_ _t_ ) _−_ _µ_ _t−_ 1 ( _x_ _t_ ) _| ≤_ _ζ_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ _t_ )] _≥_ 1 _−_ _δ, for all t ∈_ [1 _, T_ ] _._


The GP-UCB/EST decision rule is,

_x_ _t_ = arg max � _µ_ _t−_ 1 ( _x_ ) + _α_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ )�
_x∈X_


_m−µ_ _t−_ 1 ( _x_ ) 2
For EST, _α_ _t_ = � _σ_ _t−_ 1 ( _x_ ) � . Implicit in this definition of the decision rule is the corresponding confidence

interval for each _x ∈X_,

_C_ _t_ _[seq]_ ( _x_ ) _≡_ � _µ_ _t−_ 1 ( _x_ ) _−_ _α_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ ) _, µ_ _t−_ 1 ( _x_ ) + _α_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ )� _,_


where this confidence interval’s upper confidence bound is the value of the argument of the decision rule.
Furthermore, the width of any confidence interval is the difference between the uppermost and the lowermost
limits, here _w_ = 2 _α_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ ). In the case of BUCB/B-EST, the batched confidence rules are of the form,

_C_ _t_ _[batch]_ ( _x_ ) _≡_ � _µ_ _fb_ [ _t_ ] ( _x_ ) _−_ _β_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ ) _, µ_ _fb_ [ _t_ ] ( _x_ ) + _β_ _t_ [1] _[/]_ [2] _σ_ _t−_ 1 ( _x_ )�


**Lemma 6.2.** _(Similar to Lemma 12 in [_ _9_ _]) If_ _f_ ( _x_ _t_ ) _∈_ _C_ _t_ _[batch]_ ( _x_ _t_ ) _∀t ≥_ 1 _and given that actions are selected_
_using EST, it holds that,_



_R_ _T_ _≤_ �



_TC_ 1 _γ_ _T_ ( _ζ_ _T_ [1] _[/]_ [2] + _α_ _t_ [1] _[∗]_ _[/]_ [2] [)]



_Proof._ The proof of Lemma 12 in [ 9 ] just uses the fact that the sequential regret bounds of UCB are
2 _β_ _t_ [1] _[/]_ [2] _σ_ _t_ ( _x_ _t_ ). We follow their same proof but use the EST sequential bounds to get the desired result.


_Proof._ (of Theorem 3.2 in the main paper) The proof is similar to Theorem 5 in [ 9 ]. The sum of the regrets
over the T timesteps is split over the first _T_ _[init]_ timesteps and the remaining timesteps. The former term
_T_ _[init]_
� _t_ =1 _[m][ −]_ _[f]_ [(] _[x]_ _[t]_ [)] _[ ≤]_ [2] _[T]_ _[ init]_ _[∥][f]_ _[∥]_ _[∞]_ [. The latter term is treated as simple BUCB and from Lemma][ 6.2][, we]

get _R_ _T_ _init_ +1: _T_ _≤_ ~~�~~ ( _T −_ _T_ _[init]_ ) _C_ 1 _γ_ ( _T −T_ _init_ ) ( _ζ_ _T_ [1] _[/]_ [2] + _α_ _t_ [1] _[∗]_ _[/]_ [2] [)] _[ ≤√][TC]_ [1] _[γ]_ _[T]_ [(] _[ζ]_ _T_ [1] _[/]_ [2] + _α_ _t_ [1] _[∗]_ _[/]_ [2] [) as] _[ γ]_ _[T]_ [is a non-decreasing]

function. Combining the two terms gives us the desired result.


11


**6.2** **Batch Bayesian Optimization via DPP-Maximization**


In this section, we present the proof of the regret bounds for BBO via DPP-MAX. Since the GP-UCBDPP-MAX is the same as GP-UCB-PE, we focus on GP-EST-DPP-MAX. We first restate the EST part of
Theorem 3.3. Firstly, none of our proofs will depend on the order in which the batch was constructed but for
sake of clarity of exposition, whenever needed, we can consider any arbitrary ordering of the _B −_ 1 points
chosen by maximizing a ( _B −_ 1)-DPP or sampling from it.



**Theorem 6.3.** _At iteration_ _t_ _, fix_ _δ >_ 0 _and let_ _β_ _t_ = � min
_x∈X_



_m−µ_ _t−_ 1 ( _x_ )

_σ_ _t−_ 1 ( _x_ ) � _,_ _ζ_ _t_ = 2 log ( _π_ [2] _t_ [2] _/_ 3 _δ_ ) _and_ _C_ 1 =



36 _/_ log (1 + _σ_ _[−]_ [2] ) _. Then, with probability_ _≥_ 1 _−_ _δ_ _, the full cumulative regret incurred by EST-DPP-MAX is_
_R_ _t_ _≤_ _[√]_ _C_ 1 _TBγ_ _T B_ ( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T B_ [1] _[/]_ [2] [)] _[.]_


Notice that the logarithm term for _ζ_ _t_ in the above theorem is twice that of the one in Lemma 6.1. This
happens by considering the same proof as that for Lemma 6.1 but taking a union bound over _x_ _[•]_ _t_ [along with]
_x_ _t_ . We first prove some required lemmas.


**Lemma 6.4.** _The deviation of the first point, selected by either UCB or EST, is bounded by the deviation of_
_any point selected by the DPP-MAX or DPP-SAMPLE in the previous iteration with high probability, i.e.,_


_∀t < T, ∀_ 2 _≤_ _b ≥_ _B,_ _σ_ _t,_ 1 ( _x_ _t_ +1 _,_ 1 ) _≤_ _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )


_Proof._ The proof does not depend on the actual policy (UCB/EST) used for the first point of the batch
or whether it was DPP-MAX or DPP-SAMPLE (consider an arbitary ordering of points chosen in either
case). By the definition of _x_ _t_ +1 _,_ 1, we have _f_ _t_ [+] +1 [(] _[x]_ _[t]_ [+1] _[,]_ [1] [)] _[ ≥]_ _[f]_ _t_ [ +] +1 [(] _[x]_ _t_ _[•]_ [).] Also, from Lemma 6.1, we have
_f_ _t_ [+] +1 [(] _[x]_ _t_ _[•]_ [)] _[ ≥]_ _[f]_ _t_ _[ −]_ [(] _[x]_ _[•]_ _t_ [). This is different than Lemma 2 in [] [8] [] as it now only holds for] _[ x]_ _[•]_ _t_ [rather than all] _[ x][ ∈X]_ [.]
Thus, with high probability, _f_ _t_ [+] +1 [(] _[x]_ _[t]_ [+1] _[,]_ [1] [)] _[ ≥]_ _[y]_ _t_ _[•]_ [and thus,] _[ x]_ _[t]_ [+1] _[,]_ [1] _[∈R]_ [+] _t_ [. Now, from the definition of] _[ x]_ _[t,b]_ [, we]
have _σ_ _t−_ 1 _,b_ ( _x_ _t_ +1 _,_ 1 ) _≤_ _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ ) w.h.p. Using the “Information never hurts” principle [ 17 ], we know that the
entropy of _f_ ( _x_ ) for all locations _x_ can only decrease after observing a point _x_ _t,k_ . For GPs, the entropy is also
a non-decreasing function of the variance and thus, we have, _σ_ _t,_ 1 ( _x_ _t_ +1 _,_ 1 ) _≤_ _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ ) and we are done.


**Lemma 6.5.** _(Lemma 3 in [_ _8_ _]) The sum of deviations of the points selected by the UCB/EST policy is_
_bounded by the sum of deviations over all the selected points divided by B. Formally, with high probability,_



_T_
�



_B_



_B_
� _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ ) _._


_b_ =1



� _σ_ _t−_ 1 _,_ 1 ( _x_ _t,_ 1 ) _≤_ _B_ [1]

_t_ =1



_T_
�


_t_ =1



_Proof._ The proof is the same as that of Lemma 3 in [ 8 ] but we provide it here for completeness. Using
Lemma 6.4 and the definitions of _x_ _t,b_, we have _σ_ _t,_ 1 ( _x_ _t_ +1 _,_ 1 ) _≤_ _σ_ _t−_ 1 _,b_ for all _b ≥_ 2. Summing over all _b_, we get
for all _t ≥_ 1 _, σ_ _t−_ 1 _,_ 1 ( _x_ _t,_ 1 ) + ( _B −_ 1) _σ_ _t,_ 1 ( _x_ _t_ +1 _,_ 1 ) _≤_ [�] _[B]_ _b_ =1 _[σ]_ _[t][−]_ [1] _[,b]_ [(] _[x]_ _[t,b]_ [). Now, summing over] _[ t]_ [, we get the desired]
result.


**Lemma 6.6.** _(Lemma 4 in [_ _8_ _]) The sum of the variances of the selected points are bounded by a constant_



_T_
_factor times γ_ _T B_ _, i.e., ∃C_ 1 _[′]_ _[∈]_ [R] _[,]_ �

_t_ =1


We finally prove Theorem 6.3.



_B_

2

� ( _σ_ _t−_ 1 _,b_ ( _x_ _t−_ 1 _,b_ )) [2] _≤_ _C_ 1 _[′]_ _[γ]_ _[T B]_ _[. Here][ C]_ 1 _[′]_ [=] log(1+ _σ_ _[−]_ [2] ) _[.]_

_b_ =1



_Proof._ (of Theorem 6.3) Clearly, the proof of Lemma 6.1 holds even for the last _B −_ 1 points selected in a
batch. However, _t_ _[∗]_ only goes over the the _T_ iterations rather than _TB_ evaluations. Thus, the cumulative


12


regret is of the form


_R_ _T B_ =


_≤_



_T_
�


_t_ =1


_T_
�


_t_ =1



_B_
�( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T B_ [1] _[/]_ [2] [)] _[σ]_ _[t][−]_ [1] _[,b]_ [(] _[x]_ _[t,b]_ [)]


_b_ =1



_B_
�



_B_
� _r_ _t,b_


_b_ =1



_B_
�



~~�~~
~~�~~

_≤_ ( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T B_ [1] _[/]_ [2] [)] � _TB_

�



~~�~~
~~�~~
� _TB_
�



_T_
�


_t_ =1



_B_
�( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2] by Cauchy-Schwarz


_b_ =1



_≤_ � _TBC_ 1 _[′]_ _[γ]_ _[T B]_ [(] _[β]_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T B_ [1] _[/]_ [2] [)] by Lemma 6.6

_≤_ � _TBC_ 1 _γ_ _T B_ ( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _T B_ [1] _[/]_ [2] [)] since _C_ 1 _[′]_ _[≤]_ _[C]_ [1]



**6.3** **Batch Bayesian Optimization via DPP sampling**


In this section, we prove the expected regret bounds obtained by DPP-SAMPLE (Theorem 3.4 of the main
paper)


**Lemma 6.7.** _For the points chosen by (UCB/EST)-DPP-SAMPLE, the inequality,_



_T_

1

�( _σ_ _t−_ 1 _,_ 1 ( _x_ _t−_ 1 _,_ 1 )) [2] _≤_ _B −_ 1

_t_ =1



_T_ _B_
� �( _σ_ _t−_ 1 _,b_ ( _x_ _t−_ 1 _,b_ )) [2]

_t_ =1 _b_ =2



_holds with high probability._


_Proof._ Clearly, Lemma 6.5 holds in this case as well. Furthermore, it is easy to see that the inequality
obtained by replacing every term in every summation by its square in Lemma 6.5 is also true by a similar
proof. Thus, we have



_T_
�



_B_



_B_
�( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2]


_b_ =1



�( _σ_ _t−_ 1 _,_ 1 ( _x_ _t,_ 1 )) [2] _≤_ _B_ [1]

_t_ =1



_T_
�


_t_ =1



_B_



_B_
�( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2]


_b_ =2



= _⇒_ (1 _−_ [1]

_B_ [)]



_T_
�



�( _σ_ _t−_ 1 _,_ 1 ( _x_ _t,_ 1 )) [2] _≤_ _B_ [1]

_t_ =1



_T_
�


_t_ =1



_T_ _B_
� �( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2]

_t_ =1 _b_ =2



= _⇒_



_T_

1

�( _σ_ _t−_ 1 _,_ 1 ( _x_ _t,_ 1 )) [2] _≤_ _B −_ 1

_t_ =1



Hence, we are done.



=
Define _η_ _t_ [1] _[/]_ [2]



2 _β_ _t_ [1] _[/]_ [2] for UCB
�( _β_ _t_ [1] _[∗]_ _[/]_ [2] + _ζ_ _t_ [1] _[/]_ [2] ) for EST [.]



_Proof._ (of Theorem 3.4 in the main paper) The expectation here is taken over the last _B −_ 1 points in each
iteration being drawn from the ( _B −_ 1)-DPP with the posterior kernel at the _t_ _[th]_ iteration. Using linearity of


13


expectation, we get


_T_
E� �
� _t_ =1



_B_ _T_
� _r_ _t,b_ � [�] [2] = �

_b_ =1 � _t_ =1



_B_
�



_T_ _B_
� E� �

_t_ =1 _b_ =1



_B_
� _r_ _t,b_ � [�] [2]


_b_ =1



_T_ _B_
� _η_ _tB_ [1] _[/]_ [2] [E] � �

_t_ =1 _b_ =1



_T_

_≤_
�
�



_B_
� _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )� [�] [2]


_b_ =1



�( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2] [�] by Cauchy-Schwarz


_b_ =1



_≤_ _η_ _T B_ _TB_



_T_ _B_
� E� �

_t_ =1 _b_ =1



_T_
�



�( _σ_ _t−_ 1 _,b_ ( _x_ _t,b_ )) [2] [�] by Lemma 6.7


_b_ =2



_TB_ [2]
_≤_ _η_ _T B_

_B −_ 1



_T_ _B_
� E� �

_t_ =1 _b_ =2



_T_
�



It is easy to see by Schur’s identity and the definition of _σ_ _t−_ 1 _,b_ that the term inside the expectation is just
log det (( _K_ _t,_ 1 ) _S_ ), where _S_ is the set of _B −_ 1 points chosen in the _t_ _[th]_ iteration by DPP sampling with kernel
_K_ _t,_ 1 . Let _L_ = _B −_ 1. Thus,



_B −_ 1



_T_
E� �
� _t_ =1



_B_
�



_B_

_TB_ [2]

� _r_ _t,b_ � [�] [2] _≤_ _η_ _T B_ _B −_

_b_ =1



_T_
� E _S∼_ ( _L−DP P_ ( _K_ _t,_ 1 )) � log det(( _K_ _t,_ 1 ) _S_ )�


_t_ =1



Firstly, since the expectation is less than the maximum and _B/_ ( _B −_ 1) _≤_ 2, we get that the expected regret
has the same bound as the regret bounds for DPP-MAX. This bound is however, loose. We get a bound below
which may be worse but that is due to a loose analysis on our part and we can just choose the minimum of
below and the DPP-MAX regret bounds. Expanding the expectation, we get


E _S∼_ ( _L−DP P_ ( _K_ _t,_ 1 )) � log det(( _K_ _t,_ 1 ) _S_ )�



=
�

_|S|_ = _L_


=
�

_|S|_ = _L_



det(( _K_ _t,_ 1 ) _S_ ) log(det(( _K_ _t,_ 1 ) _S_ ))
~~�~~ det(( _K_ _t,_ 1 ) _S_ )

_|S|_ = _L_

det(( _K_ _t,_ 1 ) _S_ ) log( ~~�~~ det((det(( _K_ _t,_ _K_ 1 ) _t,S_ 1 )))



+

~~�~~ det(( _K_ _t,_ 1 ) _S_ )

_|S|_ = _L_



_t,_ 1 _S_
~~�~~ det(( _K_ _t,_ 1 ) _S_ ) [)]

_|S|_ = _L_



det(( _K_ _t,_ 1 ) _S_ )) log( [�] det(( _K_ _t,_ 1 ) _S_ ))

_|S|_ = _L_



~~�~~ det(( _K_ _t,_ 1 ) _S_ )

_|S|_ = _L_



~~�~~



= _−H_ (L-DPP( _K_ _t,_ 1 )) + log( � det(( _K_ _t,_ 1 ) _S_ ))

_|S|_ = _L_



_≤−H_ (L-DPP( _K_ _t,_ 1 )) + log( _|X|_ _[L]_ max det(( _K_ _t,_ 1 ) _S_ ))

_≤−H_ (L-DPP( _K_ _t,_ 1 )) + _L_ log( _|X|_ ) + log(max det(( _K_ _t,_ 1 ) _S_ ))


Plugging this into the summation and observing that the summation over last term is less than _C_ 1 _[′]_ _[γ]_ _[T B]_ [, we]
get the desired result.


**6.4** **Bounds on Information Gain for RBF kernels**


**Theorem 6.8.** _The maximum information gain for the RBF kernel after S timesteps is O_ �(log _|S|_ ) _[d]_ [�]


_Proof._ Let _x_ _S_ be the vector of points from subset _S_, that is, _x_ _S_ = ( _x_ ) _x∈S_, and the noisy evaluations of a
function _f_ at these points be denoted by a vector _y_ _S_ = _f_ _S_ + _ϵ_ _S_, where _f_ _S_ = ( _f_ ( _x_ )) _x∈S_ and _ϵ_ _S_ _∼_ _N_ (0 _, σ_ [2] _I_ ).
In Bayesian experimental design, the informativeness or the information gain of _S_ is given by the mutual
information between _f_ and these observations _I_ ( _y_ _S_ ; _f_ ) = _H_ ( _y_ _S_ ) _−_ _H_ ( _y_ _S_ _| f_ ). When _f_ is modeled by a


14


Gaussian process, it is specified by the mean function _µ_ ( _x_ ) = E _f_ ( _x_ ) and the covariance or kernel function
_k_ ( _x, x_ _[′]_ ) = E( _f_ ( _x_ ) _−_ _µ_ ( _x_ )) ( _f_ ( _x_ _[′]_ ) _−_ _µ_ ( _x_ _[′]_ )). In this case,


_I_ ( _y_ _S_ ; _f_ ) = _I_ ( _y_ _S_ ; _f_ _S_ ) = [1]

2 [log det(] _[I]_ [ +] _[ σ]_ _[−]_ [2] _[K]_ _[S]_ [)] _[,]_


where _K_ _S_ = ( _k_ ( _x, x_ _[′]_ )) _x,x_ _′_ _∈S_ . It is easy to see that



log det( _I_ + _σ_ _[−]_ [2] _K_ _S_ ) =



_|S|_
� log �1 + _σ_ _[−]_ [2] _λ_ _t_ ( _K_ _S_ )�


_t_ =1



Seeger _et al._ [ 27 ] showed that for a Gaussian RBF kernel in _d_ dimensions, _λ_ _t_ ( _K_ ) _≤_ _cB_ _[t]_ [1] _[/d]_, with _B <_ 1. Let

_d_
_T_ = �log 1 _/B_ _|S|_ � _≪|S|_ . Then for _t > T_, we have _λ_ _t_ _≤_ _c/T_, and for _t ≤_ _T_, we have _λ_ _t_ _≤_ _c_ . Therefore,



log det( _I_ + _σ_ _[−]_ [2] _K_ _S_ ) =



_|S|_
� log �1 + _σ_ _[−]_ [2] _λ_ _t_ ( _K_ _S_ )�


_t_ =1



=
�



� log �1 + _σ_ _[−]_ [2] _λ_ _t_ ( _K_ _S_ )� + �

_t≤T_ _t>T_



� log �1 + _σ_ _[−]_ [2] _λ_ _t_ ( _K_ _S_ )�


_t>T_



_≤_ _T_ log �1 + _σ_ _[−]_ [2] _c_ � + log 1 + _[cσ]_ _[−]_ [2]
� _|S|_

= _O_ ( _T_ )



� _T_



Thus, the maximum information gain for _S_ is upper bounded by _O_ �(log _|S|_ ) _[d]_ [�] .


15


**6.5** **Experiments**


**6.5.1** **Synthetic Experiments**


(a) Cosines Function. B=5 (b) Cosines Function. B=10


Figure 3: Immediate regret of the algorithms on the mixture of cosines synthetic function with B = 5 and 10


**6.5.2** **Real-World Experiments**


We first provide the results for the Abalone experiment and then provide the Prec@1 values for the FastXML
experiment on the Delicious experiment.


(a) Abalone Function. B=5 (b) Abalone Function. B=10


Figure 4: Immediate regret of the algorithms on the Abalone experiment with B = 5 and 10


(a) FastXML - Delicious Dataset. B=5 (b) FastXML - Delicious Dataset. B=10


Figure 5: Immediate regret of the algorithms on the FastXML experiment on the Delicious dataset with B = 5 and 10


16


