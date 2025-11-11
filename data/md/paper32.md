## **Classification with Valid and Adaptive Coverage**



**Emmanuel J. Candès**
Departments of Mathematics
and of Statistics
Stanford University



**Yaniv Romano** _[∗]_
Department of Statistics
Stanford University



**Matteo Sesia** _[∗]_
Department of Statistics
Stanford University



**Abstract**


Conformal inference, cross-validation+, and the jackknife+ are hold-out methods
that can be combined with virtually any machine learning algorithm to construct
prediction sets with guaranteed marginal coverage. In this paper, we develop
specialized versions of these techniques for categorical and unordered response
labels that, in addition to providing marginal coverage, are also fully adaptive to
complex data distributions, in the sense that they perform favorably in terms of
approximate conditional coverage compared to alternative methods. The heart of
our contribution is a novel conformity score, which we explicitly demonstrate to be
powerful and intuitive for classification problems, but whose underlying principle
is potentially far more general. Experiments on synthetic and real data demonstrate
the practical value of our theoretical guarantees, as well as the statistical advantages
of the proposed methods over the existing alternatives.


**1** **Introduction**


Imagine we have _n_ data samples _{_ ( _X_ _i_ _, Y_ _i_ ) _}_ _[n]_ _i_ =1 [with features] _[ X]_ _[i]_ _[ ∈]_ [R] _[p]_ [ and a discrete label] _[ Y]_ _[i]_ _[ ∈]_
_Y_ = _{_ 1 _,_ 2 _, . . ., C}_ . The samples are drawn exchangeably (e.g., i.i.d., although independence is
unnecessary) from some unknown distribution _P_ _XY_ . Given such data and a desired coverage level
1 _−_ _α ∈_ (0 _,_ 1), we seek to construct a prediction set _C_ [ˆ] _n,α_ _⊆Y_ for the _unseen_ label of a new data point
( _X_ _n_ +1 _, Y_ _n_ +1 ), also drawn exchangeably from _P_ _XY_, achieving marginal coverage; that is, obeying


P _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ ( _X_ _n_ +1 ) _≥_ 1 _−_ _α._ (1)
� �

The probability above is taken over all _n_ + 1 data points, and we ask that (1) holds for any fixed
_α_, _n_, and _P_ _XY_ . While marginal coverage has the advantage of being both desirable and practically
achievable, it unfortunately does not imply the stronger notion of conditional coverage:


P _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ ( _x_ ) _| X_ _n_ +1 = _x_ _≥_ 1 _−_ _α._ (2)
� �

The latter asks for valid coverage conditional on a specific observed value of the features _X_ . It
is already known that conditional coverage cannot be achieved in theory without strong modeling
assumptions [ 1, 23 ], which we are not willing to make in this paper. That said, it is undeniable
that conditional coverage would be preferable. We thus seek to develop classification methods that
are provably valid in the marginal sense (1) and also attempt to sensibly approximate conditional
coverage (2) . At the same time, we want powerful predictions, in the sense that the cardinality of _C_ [ˆ]
should be as small as possible.


**1.1** **The oracle classifier**


Imagine we have an _oracle_ with perfect knowledge of the conditional distribution _P_ _Y |X_ of _Y_ given
_X_ . This would of course give the problem away; to be sure, we would define optimal prediction sets


_∗_ Equal contribution.


Preprint. Under review.


_C_ _α_ [oracle] ( _X_ _n_ +1 ) with conditional coverage as follows: for any _x ∈_ R _[p]_, set _π_ _y_ ( _x_ ) = P[ _Y_ = _y | X_ = _x_ ]
for each _y ∈Y_ . Denote by _π_ (1) ( _x_ ) _≥_ _π_ (2) ( _x_ ) _≥_ _. . . ≥_ _π_ ( _C_ ) ( _x_ ) the order statistics for _π_ _y_ ( _x_ ) . For
simplicity, let us assume for now that there are no ties; we will relax this assumption shortly. For any
_τ ∈_ [0 _,_ 1], define the _generalized conditional quantile_ function [2]


_L_ ( _x_ ; _π, τ_ ) = min _{c ∈{_ 1 _, . . ., C}_ : _π_ (1) ( _x_ ) + _π_ (2) ( _x_ ) + _. . ._ + _π_ ( _c_ ) ( _x_ ) _≥_ _τ_ _},_ (3)


and the prediction set:


_C_ _α_ [oracle+] ( _x_ ) = _{_ ‘ _y_ ’ indices of the _L_ ( _x_ ; _π,_ 1 _−_ _α_ ) largest _π_ _y_ ( _x_ ) _} ._ (4)


Hence, (4) is the smallest deterministic set that contains a response with feature values _X_ = _x_ with
probability at least 1 _−_ _α_ . For example, if _π_ 1 ( _x_ ) = 0 _._ 3, _π_ 2 ( _x_ ) = 0 _._ 6, and _π_ 3 ( _x_ ) = 0 _._ 1, we have
_π_ (1) ( _x_ ) = 0 _._ 6, _π_ (2) ( _x_ ) = 0 _._ 3, and _π_ (3) ( _x_ ) = 0 _._ 1, with _L_ ( _x,_ 0 _._ 9) = 2, _C_ 0 [oracle] _._ 1 [(] _[x]_ [) =] _[ {]_ [1] _[,]_ [ 2] _[}]_ [, and]
_L_ ( _x,_ 0 _._ 5) = 1, _C_ 0 [oracle] _._ 5 [(] _[x]_ [) =] _[ {]_ [2] _[}]_ [. Furthermore, define a function] _[ S]_ [ with input] _[ x]_ [,] _[ u][ ∈]_ [[0] _[,]_ [ 1]] [,] _[ π]_ [, and] _[ τ]_ [,]
which can be seen as a _generalized inverse_ of (3):


_S_ ( _x, u_ ; _π, τ_ ) = ‘‘ _y_ ’ indices of the _L_ ( _x_ ; _π, τ_ ) _−_ 1 largest _π_ _y_ ( _x_ ) _,_ if _u ≤_ _V_ ( _x_ ; _π, τ_ ) _,_ (5)
� _y_ ’ indices of the _L_ ( _x_ ; _π, τ_ ) largest _π_ _y_ ( _x_ ) _,_ otherwise _,_


where



1
_V_ ( _x_ ; _π, τ_ ) =
_π_ ( _L_ ( _x_ ; _π,τ_ )) ( _x_ )









_L_ ( _x_ ; _π,τ_ )
�



� _π_ ( _c_ ) ( _x_ ) _−_ _τ_


_c_ =1



 _._



With this in place, by letting _u_ be the realization of a uniform random variable, we can see that the
oracle has access to tighter randomized prediction sets, namely,


_C_ _α_ [oracle] ( _x_ ) = _S_ ( _x, U_ ; _π,_ 1 _−_ _α_ ) _._ (6)


Above, _U ∼_ Uniform(0 _,_ 1) is independent of everything else. It is easy to verify that the sets in (6)
are the smallest randomized prediction sets with conditional coverage at level 1 _−_ _α_ . In the above
example, we would have _C_ 0 [oracle] _._ 5 [(] _[x]_ [) =] _[ ∅]_ [with probability] [ (0] _[.]_ [6] _[−]_ [0] _[.]_ [5)] _[/]_ [0] _[.]_ [6 = 1] _[/]_ [6] [ and] _[ C]_ 0 [oracle] _._ 5 [(] _[x]_ [) =] _[ {]_ [2] _[}]_
otherwise. Finally, if there are any ties among the class probabilities, the oracle could simply break
them at random and discard from _C_ _α_ [oracle] ( _x_ ) all labels with zero probability. Of course, we do not
have access to such an oracle since _P_ _Y |X_ is unknown.


**1.2** **Preview of our methods**


This paper uses classifiers trained on the available data to approximate the unknown conditional
distribution of _Y | X_ . A key strength of the proposed methods is their ability to work with any
black-box predictive model, including neural networks, random forests, support vector classifiers,
or any other currently existing or possible future alternatives. The only restriction on the training
algorithm is that it should treat all samples exchangeably; i.e., it should be invariant to their order.
Most off-the-shelf tools offer such suitable probability estimates ˆ _π_ _y_ ( _x_ ) that we can exploit, regardless
of whether they are well-calibrated, by imputing them into an algorithm inspired by the oracle from
Section 1.1 in order to obtain prediction sets with guaranteed coverage—as we shall see.


Our reader will understand that naively substituting _π_ _y_ ( _x_ ) with ˆ _π_ _y_ ( _x_ ) into the oracle procedure would
yield predictions lacking any statistical guarantees because ˆ _π_ _y_ ( _x_ ) may be a poor approximation of
_π_ _y_ ( _x_ ) . Fortunately, we can automatically account for errors in ˆ _π_ _y_ ( _x_ ) by adaptively choosing the
threshold _τ_ in (3) in such a way as to guarantee finite-sample coverage on future test points.


**1.3** **Related work**


We build upon conformal inference [ 12, 24, 26 ] and take inspiration from [ 3, 5, 8 – 10, 13, 18 ] which
made conformal prediction for regression problems adaptive to heteroscedasticity, thus bringing it
closer to conditional coverage [ 20 ]. Conformal inference has been applied before to classification
problems [ 7, 19, 24, 25 ] in order to attain marginal coverage; however, the idea of explicitly trying to
approximate the oracle from Section 1.1 is novel. We will see that our procedure empirically achieves


2 Recall that the conditional quantiles for continuous responses are: inf _{y ∈_ R : P[ _Y ≤_ _y | X_ = _x_ ] _≥_ _τ_ _}_ .


2


better conditional coverage than a direct application of conformal inference. While working on this
project, we became aware of the independent work of [ 2 ], which also seeks to improve the conditional
coverage of conformal classification methods. However, their approach differs substantially; see
Section 2.4. Finally, our method also naturally accommodates calibration through cross-validation+
and the jackknife+ [ 4 ], which had not yet been extended to classification, although the natural
generality of these calibration techniques has also been very recently noted by others [10].

A different but related line of work focuses on post-processing the output of black-box classification
algorithms to produce more accurate probability estimates [ 6, 11, 15, 16, 22, 27, 28 ], although without
achieving prediction sets with provable finite-sample coverage. These techniques are complementary
to our methods and may help further boost our performance by improving the accuracy of any given
black box; however, we have not tested them empirically in this paper for space reasons.


**2** **Methods**


**2.1** **Generalized inverse quantile conformity scores**


Suppose we have a black-box classifier ˆ _π_ _y_ ( _x_ ) that estimates the true unknown class probabilities
_π_ _y_ ( _x_ ) . Here, we only assume ˆ _π_ _y_ ( _x_ ) to be standardized: 0 _≤_ _π_ ˆ _y_ ( _x_ ) _≤_ 1, [�] _[C]_ _y_ =1 _[π]_ [ˆ] _[y]_ [ (] _[x]_ [) = 1] [,] _[ ∀][x, y]_ [.]
An example may be the output of the softmax layer of a neural network, after normalization. In
fact, almost any standard machine learning software, e.g., `sklearn`, can produce a suitable ˆ _π_, either
through random forests, k-nearest neighbors, or support vector machines, to name a few options.
Then, we plug ˆ _π_ into a modified version of the imaginary oracle procedure of Section 1.1 where the
threshold _τ_ needs to be carefully calibrated using hold-out samples independent of the training data.
We will present two alternative methods for calibrating _τ_ ; both are based on the following idea.

Define a _generalized inverse quantile_ conformity score function _E_ with input _x, y, u,_ ˆ _π_,


_E_ ( _x, y, u_ ; ˆ _π_ ) = min _{τ ∈_ [0 _,_ 1] : _y ∈S_ ( _x, u_ ; ˆ _π, τ_ ) _},_ (7)


where _S_ is our generalized notion of (estimated) conditional quantiles, defined in (5) . The conformity
score _E_ ( _·_ ) is the inverse of the smallest generalized quantile that contains the label _y_ conditional
on _X_ = _x_ . By construction, our scores evaluated on hold-out samples ( _X_ _i_ _, Y_ _i_ ), namely _E_ _i_ =
_E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˆ _π_ ), are uniformly distributed conditional on _X_ if ˆ _π_ = _π_ . (Each _U_ _i_ is a uniform random
variable in [0 _,_ 1] independent of everything else.) Therefore, one could also intuitively look at (7)
as a special type of _p-value_ . It is worth emphasizing that this property makes our scores naturally
comparable across different samples, in contrast with the scores found in the earlier literature on
adaptive conformal inference [ 18 ]. In fact, alternative conformity scores [ 2, 10, 12, 18 ] generally
have different distributions at different values of _X_, even in the ideal case where the base method
(our ˆ _π_ ) is a perfect oracle. Below, we shall see that, loosely speaking, we can construct prediction
sets with provable marginal coverage for future test points by applying (5) with a value of _τ_ close to
the 1 _−_ _α_ quantile of _{E_ _i_ _}_ _i∈I_ 2, where _I_ 2 is the set of hold-out data points not used to train ˆ _π_ .


**2.2** **Adaptive classification with split-conformal calibration**


Algorithm 1 implements the above idea with split-conformal calibration, from which we begin
because it is the easiest to explain. Later, we will consider alternative calibration methods based on
cross-validation+ and the jackknife+; we do not discuss full-conformal calibration in the interest of
space, and because it is often computationally prohibitive. For simplicity, we will apply Algorithm 1
by splitting the data into two sets of equal size; however, this is not necessary and using more data
points for training may sometimes perform better in practice [20].


**Theorem 1.** _If the samples_ ( _X_ _i_ _, Y_ _i_ ) _, for_ _i ∈{_ 1 _, . . ., n_ +1 _}_ _, are exchangeable and_ _B_ _from Algorithm 1_
_is invariant to permutations of its input samples, the output of Algorithm 1 satisfies:_

P � _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [sc] [(] _[X]_ _[n]_ [+1] [)] � _≥_ 1 _−_ _α._ (9)


_Furthermore, if the scores E_ _i_ _are almost surely distinct, the marginal coverage is near tight:_


1
P � _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [sc] [(] _[X]_ _[n]_ [+1] [)] � _≤_ 1 _−_ _α_ + _|I_ 2 _|_ + 1 _[.]_ (10)


3


**Algorithm 1:** Adaptive classification with split-conformal calibration

**1** **Input:** data _{_ ( _X_ _i_ _, Y_ _i_ ) _}_ _[n]_ _i_ =1 [,] _[ X]_ _[n]_ [+1] [, black-box learning algorithm] _[ B]_ [, level] _[ α][ ∈]_ [(0] _[,]_ [ 1)][.]

**2** Randomly split the training data into 2 subsets, _I_ 1 _, I_ 2 .

**3** Sample _U_ _i_ _∼_ Uniform(0 _,_ 1) for each _i ∈{_ 1 _, . . ., n_ + 1 _}_, independently of everything else.

**4** Train _B_ on all samples in _I_ 1 : ˆ _π ←B_ ( _{_ ( _X_ _i_ _, Y_ _i_ ) _}_ _i∈I_ 1 ).

**5** Compute _E_ _i_ = _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˆ _π_ ) for each _i ∈I_ 2, with the function _E_ defined in (7).

**6** Compute _Q_ [ˆ] 1 _−α_ ( _{E_ _i_ _}_ _i∈I_ 2 ) as the _⌈_ (1 _−_ _α_ )(1 + _|I_ 2 _|_ ) _⌉_ th largest value in _{E_ _i_ _}_ _i∈I_ 2 .

**7** Use the function _S_ defined in (5) to construct the prediction set at _X_ _n_ +1 as:


_C_ ˆ _n,α_ [sc] [(] _[X]_ _[n]_ [+1] [) =] _[ S]_ [(] _[X]_ _[n]_ [+1] _[, U]_ _[n]_ [+1] [; ˆ] _[π,]_ [ ˆ] _[Q]_ [1] _[−][α]_ [(] _[{][E]_ _[i]_ _[}]_ _[i][∈I]_ 2 [))] _[.]_ (8)


**8** **Output:** A prediction set _C_ [ˆ] _n,α_ [SC] [(] _[X]_ _[n]_ [+1] [)][ for the unobserved label] _[ Y]_ _[n]_ [+1] [.]


The proofs of this theorem and all other results are in Supplementary Section S2. Marginal coverage
holds regardless of the quality of the black-box approximation; however, one can intuitively expect
that if the black-box is consistent and a large amount of data is available, so that ˆ _π_ _y_ ( _x_ ) _≈_ _π_ _y_ ( _x_ ),
the output of our procedure will tend to be a close approximation of the output of the oracle,
which provides optimal conditional coverage. This statement could be made rigorous under some
additional technical assumptions besides the consistency of the black box [ 20 ]. However, we prefer
to avoid tedious technical details, especially since the intuition is already clear. If ˆ _π_ = _π_, the sets
_S_ ( _X_ _i_ _, U_ _i_ ; _π, τ_ ) in (5) will tend to contain the true labels for a fraction _τ_ of the points _i ∈I_ 2, as
long as _|I_ 2 _|_ is large. In this limit, _Q_ [ˆ] 1 _−α_ ( _{E_ _i_ _}_ _i∈I_ 2 ) becomes approximately equal to 1 _−_ _α_, and the
predictions in (8) will eventually approach those in (6).


**2.3** **Adaptive classification with cross-validation+ and jackknife+ calibration**


A limitation of Algorithm 1 is that it only uses part of the data to train the predictive algorithm.
Consequently, the estimate ˆ _π_ may not be as accurate as it could have been had we used all the
data for estimation purposes. This is especially true if the sample size _n_ is small. Algorithm 2
presents an alternative solution that replaces data splitting with a cross-validation approach, which is
computationally more expensive but often provides tighter prediction sets.


**Algorithm 2:** Adaptive classification with CV+ calibration

**1** **Input:** data _{_ ( _X_ _i_ _, Y_ _i_ ) _}_ _[n]_ _i_ =1 [,] _[ X]_ _[n]_ [+1] [, black-box] _[ B]_ [, number of splits] _[ K][ ≤]_ _[n]_ [, level] _[ α][ ∈]_ [(0] _[,]_ [ 1)][.]

**2** Randomly split the training data into _K_ disjoint subsets, _I_ 1 _, . . ., I_ _K_, each of size _n/K_ .

**3** Sample _U_ _i_ _∼_ Uniform(0 _,_ 1) for each _i ∈{_ 1 _, . . ., n_ + 1 _}_, independently of everything else.

**4** **for** _k ∈{_ 1 _, . . ., K}_ **do**

**5** Train _B_ on all samples except those in _I_ _k_ : ˆ _π_ _[k]_ _←B_ ( _{_ ( _X_ _i_ _, Y_ _i_ ) _}_ _i∈{_ 1 _,...,n}\I_ _k_ ).

**6** **end**

**7** Use the function _E_ defined in (7) to construct the prediction set _C_ [ˆ] _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)][ as:]


ˆ
_C_ _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [) =] _y ∈Y_ :
�

_n_ (11)
� _i_ =1 **1** � _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˆ _π_ _[k]_ [(] _[i]_ [)] ) _< E_ ( _X_ _n_ +1 _, y, U_ _n_ +1 ; ˆ _π_ _[k]_ [(] _[i]_ [)] )� _<_ (1 _−_ _α_ )( _n_ + 1)� _,_


where _k_ ( _i_ ) _∈{_ 1 _, . . ., K}_ is the fold containing the _i_ th sample.

**8** **Output:** A prediction set _C_ [ˆ] _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)][ for the unobserved label] _[ Y]_ _[n]_ [+1] [.]


In words, in Algorithm 2, we sweep over all possible labels _y ∈Y_ and include _y_ in the final prediction
set _C_ [ˆ] _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)] [ if the corresponding score] _[ E]_ [(] _[X]_ _[n]_ [+1] _[, y, U]_ _[n]_ [+1] [; ˆ] _[π]_ _[k]_ [(] _[i]_ [)] [)] [ is smaller than] [ (1] _[ −]_ _[α]_ [)(] _[n]_ [ + 1)]
hold-out scores _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˆ _π_ _[k]_ [(] _[i]_ [)] ) evaluated on the true labeled data. Note that we have assumed
_n/K_ to be an integer for simplicity; however, different splits can have different sizes. In the special


4


case where _K_ = _n_, we refer to the hold-out system in Algorithm 2 as jackknife+ rather than
cross-validation+, consistently with the terminology in [4].
**Theorem 2.** _Under the same assumptions of Theorem 1, the output of Algorithm 2 satisfies:_



2(1 _−_ 1 _/K_ )
P � _{Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)] � _≥_ 1 _−_ 2 _α −_ min � _n/K_ + 1



1 _−_ 1 _/K_ )

_n/K_ + 1 _[,]_ [ 1] _[ −]_ _K_ + 1 _[K][/][n]_



_._ (12)
�



_K_ + 1



_In the special case where K_ = _n, this bound simplifies to:_

P � _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [JK+] [(] _[X]_ _[n]_ [+1] [)] � _≥_ 1 _−_ 2 _α._ (13)


Note that this establishes that the coverage is slightly below 1 _−_ 2 _α_ . Therefore, to guarantee 1 _−_ _α_
coverage, we should replace the input _α_ in Algorithm 2 with a smaller value near _α/_ 2 . We chose not
to do so because our experiments show that the current implementation already typically covers at
level 1 _−_ _α_ (or even higher) in practice; this empirical observation is consistent with [ 4 ]. Furthermore,
there exists a conservative variation of Algorithm 2 for which we can prove 1 _−_ _α_ coverage without
modifying the input level; see Supplementary Section S1.1.


To see why everything above makes sense, consider what would happen if the black-box estimates
of conditional probabilities in Algorithm 2 were exact. In this case, the final prediction set in (11)
would become

ˆ
_C_ _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [) =] � _y ∈Y_ : _E_ ( _X_ _n_ +1 _, y, U_ _n_ +1 ; _π_ ) _<_ _Q_ [ˆ] 1 _−α_ ( _{E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; _π_ ) _}_ _i∈{_ 1 _,...,n}_ )� _,_ (14)

where _Q_ [ˆ] 1 _−α_ is defined as in Section 2.2. If _n_ is large, for any fixed threshold _τ_, we can
expect ˆ _S_ ( _X_ _i_ _, U_ _i_ ; _π, τ_ ) to contain _Y_ _i_ for approximately a fraction _τ_ of samples _i_ . Therefore,
_Q_ 1 _−α_ ( _{E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; _π_ ) _}_ _i∈{_ 1 _,...,n}_ ) _≈_ 1 _−_ _α_, and the decision rule becomes approximately:

_C_ ˆ _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)] _[ ≈{][y][ ∈Y]_ [ :] _[ E]_ [(] _[X]_ _[n]_ [+1] _[, y, U]_ _[n]_ [+1] [;] _[ π]_ [)] _[ ≤]_ [1] _[ −]_ _[α][}][,]_ (15)


which is equivalent to the oracle procedure from Section 1.1.


**2.4** **Comparison with alternative conformal methods**


Conformal prediction has been proposed before in the context of classification [ 24 ], through a very
general calibration rule of the form
_C_ ˆ( _x_ ; _t_ ) = _{y ∈Y_ : ˆ _f_ ( _y | x_ ) _≥_ _t},_


where the score _f_ [ˆ] is a function learned by a black-box classifier. However, to date it was not clear how
to best translate the output of the classifier into a powerful score _f_ [ˆ] for the above decision rule. In fact,
typical choices of _f_ [ˆ] ( _y | x_ ), e.g., the estimated probability of _Y_ = _y_ given _X_ = _x_, often lead to poor
conditional coverage because the same threshold _t_ is applied both to easy-to-classify samples (where
one label has probability close to 1 given _X_ ) and to hard-to-classify samples (where all probabilities
are close to 1 _/|Y|_ given _X_ ). Therefore, this _homogeneous_ conformal classification may significantly
underperform compared to the oracle from Section 1.1, even in the ideal case where the black-box
manages to learn the correct probabilities. This limitation has also been very recently noted in [ 2 ]
and is analogous to that addressed by [18] in problems with a continuous response variable [20].


The work of [ 2 ] addresses this problem by applying quantile regression [ 18 ] to hold-out scores _f_ [ˆ] .
However, their solution has two limitations. Firstly, it involves additional data splitting to avoid
overfitting, which tends to reduce power. Secondly, its theoretical asymptotic optimality is weaker
than ours because it requires the consistency of two black-boxes instead of one (this should be clear
even though we have explained consistency only heuristically). Practically, experiments suggest that
our method provides superior conditional coverage and often yields smaller prediction sets.


**3** **Experiments with simulated data**


**3.1** **Methods and metrics**


We compare the performances of Algorithms 1 (SC) and 2 (CV+, JK+), which are based on the
new generalized inverse quantile conformity scores in (7), to those of homogeneous conformal


5


classification (HCC) and conformal quantile classification (CQC) [ 2 ]. We focus on two different data
generating scenarios in which marginal coverage is not a good proxy for conditional coverage (the
second setting is discussed in Supplementary Section S3.3). In both cases, we explore 3 different
black-boxes: an _oracle_ that knows the true _π_ _y_ ( _x_ ) for all _y ∈Y_ and _x_ ; a support vector classifier
(SVC) implemented by the `sklearn` Python package; and a random forest classifier (RFC) also
implemented by `sklearn` — see Supplementary Section S3.1 for more details.

We fix _α_ = 0 _._ 1 and assess performance in terms of marginal coverage, conditional coverage, and
mean cardinality of the prediction sets. Conditional coverage is defined using an estimate of the
worst-slice (WS) coverage similar to that in [ 2 ], as explained in Supplementary Section S1.2. The
cardinality of the prediction sets is computed both marginally and conditionally on coverage; the
former is defined as E[ _|C_ [ˆ] ( _X_ _n_ +1 ) _|_ ] and the latter as E[ _|C_ [ˆ] ( _X_ _n_ +1 ) _| | Y_ _n_ +1 _∈_ _C_ [ˆ] ( _X_ _n_ +1 )] . Additional
coverage and size metrics defined by conditioning on the value of a given discrete feature, e.g., _X_ 1,
are discussed in Supplementary Section S3.


**3.2** **Experiments with multinomial model and inhomogeneous features**


We generate the features _X ∈_ R _[p]_, with _p_ = 10, as follows: _X_ 1 = 1 w.p. 1 _/_ 5 and _X_ 1 = _−_ 8 otherwise,
while _X_ 2 _, . . ., X_ 10 are independent standard normal. The conditional distribution of _Y ∈{_ 1 _, . . .,_ 10 _}_
given _X_ = _x_ is multinomial with weights _w_ _j_ ( _x_ ) defined as _w_ _j_ ( _x_ ) = _z_ _j_ ( _x_ ) _/_ [�] _[p]_ _j_ _[′]_ =1 _[z]_ _[j]_ _[′]_ [(] _[x]_ [)] [, where]
_z_ _j_ ( _x_ ) = exp( _x_ _[T]_ _β_ _j_ ) and each _β_ _j_ _∈_ R _[p]_ is sampled from an independent standard normal distribution.

Figure 1 confirms that our methods have valid conditional coverage if the true class probabilities
are provided by an oracle. If the probabilities are estimated by the RFC, the conditional coverage
appears to be only slightly below 1 _−_ _α_, and is near perfect with the SVC black box. By contrast,
the conditional coverage of the alternative methods is always significantly lower than 1 _−_ _α_, even
with the help of the oracle. Our methods produce slightly larger prediction sets when the oracle is
available, but our sets are typically smaller than those of CQC and only slightly larger than those of
HCC when the class probabilities are estimated. Finally, note that JK+ is the most powerful of our
methods, followed by CV+, although SC is computationally more affordable.






















|Oracle|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|**G**~~G~~|G||G|||


|RFC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
||~~G~~||G<br>~~G~~<br>G|||


|SVC|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||||Marginal|
||||||||
||||||||
||||||||
||||||||


|Oracle|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||||GG|||
|||~~G~~<br>||**G**<br>~~**G**~~||


|RFC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||||GG|||
|G<br>G<br>GGG|G<br>**G**<br>G|GG<br>G||G<br>GGG||
||||**G**|||


|SVC|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||||Marginal|
||||||||
|**G**<br>GG|~~G~~GG|~~G~~<br>**G**||G|||
|G|||~~**G**~~<br>G||||




|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|G||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||~~GG~~|~~G~~||||


|Col1|Col2|Col3|Col4|Col5|Col6|Marginal|
|---|---|---|---|---|---|---|
|||||||Conditional (W|
||G|G|||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||G<br>~~G~~|||


|Col1|Col2|Col3|G|Col5|Col6|
|---|---|---|---|---|---|
|G~~G~~G<br>**G**G|G<br>~~G~~**G**<br>G|GG~~G~~<br>G|~~GG~~|G<br>G**G**||


|Col1|Col2|Col3|Col4|Col5|Col6|Cond. on cov|
|---|---|---|---|---|---|---|
|**G**<br>~~G~~G|~~G~~GG|~~G~~**G**|**G**|G|||



























(a)



(b)



Figure 1: Several classification methods on simulated data with 10 classes, for different choices of
calibration and black-box models. SC, CV+, and JK+ are applied with our new generalized inverse
quantile conformity scores defined in (7) . The results correspond to 100 independent experiments
with 1000 training samples and 5000 test samples each. All methods have 90% marginal coverage.
(a): Marginal coverage and worst-slice conditional coverage. (b): Size of prediction sets.


6


**4** **Experiments with real data**


In this section, we compare the performance of our proposed methods (SC, CV+, and JK+) with the
new generalized inverse quantile conformity scores defined in (7) to those of HCC and CQC [ 2 ]. We
found that the original suggestion of [ 2 ] to fit a quantile neural network [ 21 ] on the class probability
score can be unstable and yield very wide predictions. Therefore, we offer a second variant of this
calibration method, denoted by CQC-RF, which replaces the quantile neural network estimator with
quantile random forests [14]; see Supplementary Section S4 for details.

The validity and statistical efficiency of each method is evaluated according to the same metrics as
in Section 3. In all experiments, we set _α_ = 0 _._ 1 and use the following base predictive models: (i)
kernel SVC, (ii) random forest classifier (RFC), and (iii) two-layer neural network classifier (NNet).
A detailed description of each algorithm and corresponding hyper-parameters is in Supplementary
Section S4. The methods are tested on two well-known data sets: the Mice Protein Expression
data set [3] and the MNIST handwritten digit data set. The supplementary material describes the
processing pipeline and discusses additional experiments on the Fashion-MNIST and CIFAR10 data
sets. Supplementary Tables S1–S4 summarize the results of our experiments in more detail and also
consider additional settings.


Figure 2 shows that all methods attain valid marginal coverage on the Mice Protein Expression data,
as expected. Here, HCC, CQC, and CQC-RF fail to achieve conditional coverage, in contrast to the
proposed methods (SC, CV+, JK+) based on our new conformity scores in (7) . Turning to efficiency,
we observe that the prediction sets of CV+ and JK+ are smaller than those of SC, and comparable in
size to those of HCC. Here, the original CQC algorithm performs poorly both in terms of conditional
coverage and cardinality. The CQC-RF variant is not as unstable as the original CQC, although it
does not perform much better than HCC.
















|NNet|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|||~~**G**~~|||
||||||
|**G**|~~G~~||~~G~~**G**~~G~~<br>~~**G**~~||


|SVC<br>G|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||G<br>~~G~~G|||Marginal|
|||||||
|GG<br>~~G~~|~~G~~||G**G**G<br>~~**G**~~|||










































|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|GG~~G~~|~~G~~||~~**G**~~**G**G<br>~~**G**~~||


|Col1|Col2|G G|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||||Cond. on cover|
|~~GG~~<br>~~**G**~~|~~G~~||**G**G**G**G~~G~~|||





Figure 2: Experiments on Mice Protein Expression data. 100 independent experiments with 500
randomly chosen training samples and 580 test samples each. Left: coverage. Right: size of prediction
sets (extremely large values for CQC not shown). Other details are as in Figure 1.


Figure 3 presents the results on the MNIST data. Here, the sample size is relatively large and hence
we exclude JK+ due to its higher computational cost. As in the previous experiments, all methods
achieve 90% marginal coverage. Unlike CQC, CQC-RF, and HCC, our methods also attain valid
conditional coverage when relying on the NNet or SVC as base predictive models. With the RFC, all
methods tend to undercover, suggesting that this classifier estimates the class probabilities poorly,
and our prediction sets are larger than those constructed by CQC-RF and HCC. By contrast, the
NNet enables our methods to achieve conditional coverage with prediction sets comparable in size to
those produced by CQC-RF and HCC. The bottom part of Figure 3 demonstrates that CV+ also has
conditional coverage given the true class label _Y_, while SC performs only slightly worse. In striking
contrast, both HCC, CQC, and CQC-RF fail to achieve 90% conditional coverage.


3 `[https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)`


7


|NNet RFC<br>1.000<br>0.975<br>0.950<br>0.925<br>G G<br>0.900 Coverage<br>0.875 G G GG<br>1.00<br>0.95<br>G<br>0.90<br>0.85<br>GG<br>00 .. 78 50 G G G G GG GG<br>CQC−RF CQC−RF<br>SC CV+ CQC HCC SC CV+ CQC<br>Method|SVC NNet RFC SVC<br>1.75<br>1.50 G G Marginal Marginal<br>1.25 G G<br>G G GGGG<br>1.00 GG G G<br>G Size<br>1.75 Conditional Cond.<br>G<br>1.50<br>G G on<br>1.25 G G GG 1.00 GG G G GGG G (WC) cover<br>CQC−RF CQC−RF CQC−RF CQC−RF<br>HCC SC CV+ CQC HCC SC CV+ CQC HCC SC CV+ CQC HCC SC CV+ CQC HCC<br>Method|
|---|---|
|G<br>~~G~~<br>~~G~~<br>G ~~G~~<br>~~G~~<br>G<br>~~G~~<br>~~G~~<br>G<br>~~G~~<br>G<br>G<br>GG~~G~~<br>G<br>~~G~~<br>G<br>G<br>GG<br>~~G~~<br>G<br>~~G ~~G G<br>~~G~~<br>~~G G~~<br>~~G~~<br>G<br>G<br>~~G~~<br>G<br>G<br>**G**~~G~~<br>G~~G~~<br>G<br>~~G~~<br>G<br>~~G~~<br>G<br>G<br>~~G~~<br>~~**G**~~**G**<br>~~**G**~~<br>G ~~**G**~~<br>G~~G~~G<br>~~G ~~~~**G**~~G~~**G**~~<br>G~~G~~<br>~~**G**~~<br>~~G~~<br>~~**G**~~<br>G<br>~~**G**~~<br>G<br>~~G~~~~**G**~~<br>G<br>~~**G**~~<br>~~**GG**~~<br>~~G~~<br>G~~G~~<br>~~**GG**~~<br>~~G~~<br>G<br>G G~~G~~~~**G** ~~~~**GG**~~<br>~~**G**~~<br>~~**G**~~<br>~~**G**~~<br>~~G~~<br>~~**G**~~G<br>~~**G**~~<br>G ~~**G**~~<br>~~G~~<br>**G**~~G~~<br>~~G~~<br>~~G~~G<br>G<br>G<br>~~G~~<br>~~G~~G<br>G G<br>G<br>~~G~~<br>~~G~~<br>~~G~~<br>G<br>G<br>~~G~~<br>G~~G~~<br>~~G~~<br>G<br>GG<br>~~G~~<br>**G**G<br>~~G~~<br>G<br>~~G~~<br>~~G~~<br>G ~~G~~<br>~~G~~<br>G<br>SC<br>CV+<br>CQC<br>CQC−RF<br>HCC<br>Coverage<br>Size cond. on cover<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0.80<br>0.85<br>0.90<br>0.95<br>1.00<br>1.00<br>1.05<br>1.10<br>1.15<br>1.20<br>True Y|G<br>~~G~~<br>~~G~~<br>G ~~G~~<br>~~G~~<br>G<br>~~G~~<br>~~G~~<br>G<br>~~G~~<br>G<br>G<br>GG~~G~~<br>G<br>~~G~~<br>G<br>G<br>GG<br>~~G~~<br>G<br>~~G ~~G G<br>~~G~~<br>~~G G~~<br>~~G~~<br>G<br>G<br>~~G~~<br>G<br>G<br>**G**~~G~~<br>G~~G~~<br>G<br>~~G~~<br>G<br>~~G~~<br>G<br>G<br>~~G~~<br>~~**G**~~**G**<br>~~**G**~~<br>G ~~**G**~~<br>G~~G~~G<br>~~G ~~~~**G**~~G~~**G**~~<br>G~~G~~<br>~~**G**~~<br>~~G~~<br>~~**G**~~<br>G<br>~~**G**~~<br>G<br>~~G~~~~**G**~~<br>G<br>~~**G**~~<br>~~**GG**~~<br>~~G~~<br>G~~G~~<br>~~**GG**~~<br>~~G~~<br>G<br>G G~~G~~~~**G** ~~~~**GG**~~<br>~~**G**~~<br>~~**G**~~<br>~~**G**~~<br>~~G~~<br>~~**G**~~G<br>~~**G**~~<br>G ~~**G**~~<br>~~G~~<br>**G**~~G~~<br>~~G~~<br>~~G~~G<br>G<br>G<br>~~G~~<br>~~G~~G<br>G G<br>G<br>~~G~~<br>~~G~~<br>~~G~~<br>G<br>G<br>~~G~~<br>G~~G~~<br>~~G~~<br>G<br>GG<br>~~G~~<br>**G**G<br>~~G~~<br>G<br>~~G~~<br>~~G~~<br>G ~~G~~<br>~~G~~<br>G<br>SC<br>CV+<br>CQC<br>CQC−RF<br>HCC<br>Coverage<br>Size cond. on cover<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0 1 2 3 4 5 6 7 8 9<br>0.80<br>0.85<br>0.90<br>0.95<br>1.00<br>1.00<br>1.05<br>1.10<br>1.15<br>1.20<br>True Y|


Figure 3: Experiments on MNIST data. 100 independent experiments with 10000 randomly chosen
training samples and 5000 test samples each. Top: coverage and size of prediction sets (large values
for CQC not shown). Bottom: coverage with neural network black box, conditional on the true _Y_, and
size of the corresponding prediction sets, conditional on coverage. Other details are as in Figure 1.


**5** **Conclusions**


This paper introduced a principled and versatile modular method for constructing prediction sets for
multi-class classification problems that enjoy provable finite-sample coverage, and also behave well
in terms of conditional coverage when compared to alternatives. Our approach leverages the power
of any black-box machine learning classifier that may be available to practitioners, and is easily
calibrated via various hold-out procedures; e.g., conformal splitting, CV+, or the jackknife+. This
flexibility makes our approach widely applicable and offers options to balance between computational
efficiency, data parsimony, and power.

Although this paper focused on classification, using conformity scores similar to those in (7) to
calibrate hold-out procedures for regression problems [ 4, 18 ] is tantalizing. In fact, previous work in
the regression setting focused on conformity scores that measure the distance of a data point from
its predicted interval on the scale of the _Y_ values (which makes sense for homoscedastic regression,
but may not be optimal otherwise), rather than by the amount one would need to relax the nominal
threshold (our _τ_ ) until the true value is covered. We leave it to future work to explore the performance
of our intuitive metrics in other settings.


The Python package at `[https://github.com/msesia/arc](https://github.com/msesia/arc)` implements our methods. This repository also contains code to reproduce our experiments.


8


**Broader Impact**


Machine learning algorithms are increasingly relied upon by decision makers. It is therefore crucial
to combine the predictive performance of such complex machinery with practical guarantees on the
reliability and uncertainty of their output. We view the calibration methods presented in this paper as
an important step towards this goal. In fact, uncertainty estimation is an effective way to quantify and
communicate the benefits and limitations of machine learning. Moreover, the proposed methodologies
provide an attractive way to move beyond the standard prediction accuracy measure used to compare
algorithms. For instance, one can compare the performance of two candidate predictors, e.g., random
forest and neural network (see Figure 3), by looking at the size of the corresponding prediction sets
and/or their their conditional coverage. Finally, the approximate conditional coverage that we seek in
this work is highly relevant within the broader framework of fairness, as discussed by [ 17 ] within a
regression setting. While our approximate conditional coverage already implicitly reduces the risk
of unwanted bias, an equalized coverage requirement [ 17 ] can also be easily incorporated into our
methods to explicitly avoid discrimination based on protected categories.


We conclude by emphasizing that the validity of our methods relies on the exchangeability of the data
points. If this assumption is violated (e.g., with time-series data), our prediction sets may not have the
right coverage. A general suggestion here is to always try to leverage specific knowledge of the data
and of the application domain to judge whether the exchangeability assumption is reasonable. Finally,
our data-splitting techniques in Section 4 offer a practical way to verify empirically the validity of the
predictions on any given data set.


**Acknowledgments and Disclosure of Funding**


E. C. was partially supported by Office of Naval Research grant N00014-20-12157, and by the Army
Research Office (ARO) under grant W911NF-17-1-0304. Y. R. was partially supported by ARO
under the same grant. Y. R. thanks the Zuckerman Institute, ISEF Foundation, the Viterbi Fellowship,
Technion, and the Koret Foundation, for providing additional research support. M. S. was suported
by NSF grant DMS 1712800.


**References**


[1] R. F. Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani. The limits of distribution-free
conditional predictive inference. _arXiv preprint arXiv:1903.04684_, 2019.


[2] M. Cauchois, S. Gupta, and J. Duchi. Knowing what you know: valid confidence sets in
multiclass and multilabel prediction. _arXiv preprint arXiv:2004.10181_, 2020.


[3] V. Chernozhukov, K. Wüthrich, and Y. Zhu. Distributional conformal prediction. _arXiv preprint_
_arXiv:1909.07889_, 2019.


[4] R. Foygel Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani. Predictive inference with the
jackknife+. _arXiv preprint arXiv:1905.02928_, 2019.


[5] L. Guan. Conformal prediction with localization. _arXiv preprint arXiv:1908.08558_, 2019.


[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
In _Proceedings of the 34th International Conference on Machine Learning-Volume 70_, pages
1321–1330. JMLR. org, 2017.


[7] Y. Hechtlinger, B. Póczos, and L. Wasserman. Cautious deep learning. _arXiv preprint_
_arXiv:1805.09460_, 2018.


[8] R. Izbicki, G. T. Shimizu, and R. B. Stern. Distribution-free conditional predictive bands using
density estimators. _arXiv preprint arXiv:1910.05575_, 2019.


[9] D. Kivaranovic, K. D. Johnson, and H. Leeb. Adaptive, distribution-free prediction intervals for
deep neural networks. _arXiv preprint arXiv:1905.10634_, 2019.


[10] A. K. Kuchibhotla and A. K. Ramdas. Nested conformal prediction and the generalized
jackknife+. _arXiv preprint arXiv:1910.10562_, 2019.


9


[11] A. Kumar, P. S. Liang, and T. Ma. Verified uncertainty calibration. In _Advances in Neural_
_Information Processing Systems_, pages 3787–3798, 2019.


[12] J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman. Distribution-free predictive
inference for regression. _Journal of the American Statistical Association_, 113(523):1094–1111,
2018.


[13] J. Lei and L. Wasserman. Distribution-free prediction bands for non-parametric regression.
_Journal of the Royal Statistical Society: Series B (Statistical Methodology)_, 76(1):71–96, 2014.


[14] N. Meinshausen. Quantile regression forests. _Journal of Machine Learning Research_, 7:983–
999, 2006.


[15] L. Neumann, A. Zisserman, and A. Vedaldi. Relaxed softmax: Efficient confidence autocalibration for safe pedestrian detection. 2018.


[16] J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized
likelihood methods. _Advances in large margin classifiers_, 10(3):61–74, 1999.


[17] Y. Romano, R. F. Barber, C. Sabatti, and E. Candès. With malice toward none: Assessing uncertainty via equalized coverage. _Harvard Data Science Review_, 4 2020.
https://hdsr.mitpress.mit.edu/pub/qedrwcz3.


[18] Y. Romano, E. Patterson, and E. J. Candès. Conformalized quantile regression. In _Advances in_
_Neural Information Processing Systems_, pages 3538–3548, 2019.


[19] M. Sadinle, J. Lei, and L. Wasserman. Least ambiguous set-valued classifiers with bounded
error levels. _Journal of the American Statistical Association_, 114(525):223–234, 2019.


[20] M. Sesia and E. J. Candès. A comparison of some conformal quantile regression methods. _Stat_,
9(1):e261, 2020.


[21] J. W. Taylor. A quantile regression neural network approach to estimating the conditional
density of multiperiod returns. _Journal of Forecasting_, 19(4):299–311, 2000.


[22] J. Vaicenavicius, D. Widmann, C. Andersson, F. Lindsten, J. Roll, and T. B. Schön. Evaluating
model calibration in classification. _arXiv preprint arXiv:1902.06977_, 2019.


[23] V. Vovk. Conditional validity of inductive conformal predictors. In _Asian conference on machine_
_learning_, pages 475–490, 2012.


[24] V. Vovk, A. Gammerman, and G. Shafer. _Algorithmic learning in a random world_ . Springer,
2005.


[25] V. Vovk, D. Lindsay, I. Nouretdinov, and A. Gammerman. Mondrian confidence machine.
Technical report, Royal Holloway, University of London, 2003. On-line Compression Modelling
project.


[26] V. Vovk, I. Nouretdinov, and A. Gammerman. On-line predictive linear regression. _The Annals_
_of Statistics_, 37(3):1566–1590, 2009.


[27] B. Zadrozny and C. Elkan. Obtaining calibrated probability estimates from decision trees and
naive bayesian classifiers. In _Icml_, volume 1, pages 609–616. Citeseer, 2001.


[28] B. Zadrozny and C. Elkan. Transforming classifier scores into accurate multiclass probability
estimates. In _Proceedings of the eighth ACM SIGKDD international conference on Knowledge_
_discovery and data mining_, pages 694–699, 2002.


10


## **Supplementary Material for** **Classification with Valid and Adaptive Coverage**



**Emmanuel J. Candès**
Departments of Mathematics
and of Statistics
Stanford University



**Yaniv Romano** _[∗]_
Department of Statistics
Stanford University



**Matteo Sesia** _[∗]_
Department of Statistics
Stanford University



**S1** **Supplementary methods**


**S1.1** **Adaptive classification with minimax jackknife+ calibration**


We can apply the minimax calibration technique of [ 2 ] to obtain a non-trivial variation of Algorithm 2
for which marginal coverage can be rigorously proved at level 1 _−_ _α_, without modifying the current
input level. Here, we consider the jackknife+—i.e., _K_ = _n_ —for simplicity. The only difference with
Algorithm 2 is that the prediction set in (11) is replaced by the following larger set:


ˆ
_C_ _n,α_ [J+mm] [(] _[X]_ _[n]_ [+1] [) =] _y ∈Y_ :
�

_n_ (S1)
� _i_ =1 **1** � _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˆ _π_ _[i]_ ) _<_ _j∈{_ min 1 _,...,n}_ _[E]_ [(] _[X]_ _[n]_ [+1] _[, y, U]_ _[n]_ [+1] [; ˆ] _[π]_ _[j]_ [)] � _<_ (1 _−_ _α_ )( _n_ + 1)� _._


**Theorem S1.** _Under the same assumptions of Theorem 1, the output of Algorithm 2 with_ _K_ = _n_ _,_
_and_ (11) _replaced by_ (S1) _, satisfies:_

P � _Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [J+][mm] ( _X_ _n_ +1 )� _≥_ 1 _−_ _α._ (S2)


**S1.2** **Quantifying conditional coverage in finite samples**


Similarly to the approach of [1], we measure coverage over a slab


_S_ _v,a,b_ = _{x ∈_ R _[p]_ : _a ≤_ _v_ _[T]_ _x ≤_ _b}_


of the feature space, where the values of _v ∈_ R _[p]_ and _a < b ∈_ R are chosen adversarially but
independently of the data. In particular, for any fixed classification prediction set _C_ [ˆ] and _δ ∈_ (0 _,_ 1),
we define



WSC( _C_ [ˆ] ; _δ_ ) = inf P[ _Y ∈_ _C_ [ˆ] ( _X_ ) _| X ∈_ _S_ _v,a,b_ ] s.t. P[ _X ∈_ _S_ _v,a,b_ ] _≥_ _δ_ ] _._
_v∈_ R _[p]_ _, a<b∈_ R � �



In practice, we estimate WSC for a particular _C_ [ˆ] by sampling 1000 independent vectors _v_ on the unit
sphere in R _[p]_ and optimizing the corresponding parameters _a, b_ through a grid search; we set _δ_ = 0 _._ 1 .
To avoid finite-sample negative bias, we partition the test data into two subsets (e.g., containing 25%
and 75% of the samples respectively); then, we use the first subset to estimate the optimal values
_v_ _[∗]_ _, a_ _[∗]_ _, b_ _[∗]_, and the second subset to evaluate conditional coverage:


P[ _Y ∈_ _C_ [ˆ] ( _X_ ) _| X ∈_ _S_ _v_ _∗_ _,a_ _∗_ _,b_ _∗_ ] _._ (S3)


_∗_ Equal contribution.


Preprint. Under review.


Therefore, regardless of the quality of our solution _v_ _[∗]_ _, a_ _[∗]_ _, b_ _[∗]_ to the above optimization problem,
the quantity in (S3) should be equal to the nominal coverage level 1 _−_ _α_ for any method with valid
conditional coverage. However, it is worth highlighting that controlling (S3) does not necessarily
imply that conditional coverage holds more generally, which is why we also look at alternative
measures of conditional coverage given either the value of certain features (e.g., _X_ 1 ), or that of the
true label _Y_ .


**S2** **Supplementary proofs**


_Proof of Theorem 1._ We begin by proving the lower bound on coverage. By construction of the
prediction set in (8), we know that


_Y_ _n_ +1 _∈_ _C_ [ˆ] _n,α_ [sc] [(] _[X]_ _[n]_ [+1] [)]


if and only if


min _{τ ∈_ [0 _,_ 1] : _Y_ _n_ +1 _∈S_ ( _X_ _n_ +1 _, U_ _n_ +1 ; ˆ _π, τ_ ) _} ≤_ _Q_ [ˆ] 1 _−α_ ( _{E_ _i_ _}_ _i∈I_ 2 ) _,_


or, equivalently, if and only if


_E_ _n_ +1 _≤_ _Q_ [ˆ] 1 _−α_ ( _{E_ _i_ _}_ _i∈I_ 2 ) _._ (S4)


Since all the conformity scores _E_ _n_ +1 and _{E_ _i_ _}_ _i∈I_ 2 are exchangeable, the probability of the event
in (S4) can be no larger than 1 _−_ _α_ . The formal proof of this statement is standard at this point,
so we simply refer to [ 3 ] for the remaining technical details. The proof for the upper bound also
immediately follows from (S4) by applying Lemma 2 in [3].


_Proof of Theorem 2._ The proof is essentially an application of the main result in [ 2 ]. This will
become apparent after we reduce our claim to the setting in the aforementioned paper. We now
examine this reduction.


Imagine that we have access to _m_ = _n/K_ test points


( _X_ _n_ +1 _, Y_ _n_ +1 _, U_ _n_ +1 ) _, . . .,_ ( _X_ _n_ + _m_ _, Y_ _n_ + _m_ _, U_ _n_ + _m_ )


as well as the training data; we will call this data set the _augmented_ data set. After partitioning the
training data into sets _I_ 1 _, . . ., I_ _K_ of size _m_, we define _I_ _K_ +1 = _{n_ + 1 _, . . ., n_ + _m}_ as the set of
test points. For any distinct _k, k_ _[′]_ _∈{_ 1 _, . . ., K_ + 1 _}_, let ˜ _π_ _[k,k]_ _[′]_ define the class probability estimator
obtained by fitting the black box on the data in _{_ 1 _, . . ., n_ + _m} \_ ( _I_ _k_ _∪I_ _k_ _′_ ) . Note that ˜ _π_ _[k,K]_ [+1] = ˆ _π_ _[k]_
for any _k_ .


Next, define the matrix _R ∈_ R [(] _[n]_ [+] _[m]_ [)] _[×]_ [(] _[n]_ [+] _[m]_ [)] with entries


+ _∞,_ if _k_ ( _i_ ) = _k_ ( _j_ ) _,_
_R_ _i,j_ = � _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˜ _π_ _[k]_ [(] _[i]_ [)] _[,k]_ [(] _[j]_ [)] ) _,_ if _k_ ( _i_ ) _̸_ = _k_ ( _j_ ) _,_


and the comparison matrix _A ∈{_ 0 _,_ 1 _}_ [(] _[n]_ [+] _[m]_ [)] _[×]_ [(] _[n]_ [+] _[m]_ [)] with entries


_A_ _ij_ = **1** [ _R_ _ij_ _> R_ _ji_ ] _._ (S5)


Note that


_Y_ _n_ +1 _/∈_ _C_ [ˆ] _n,α_ [CV+] [(] _[X]_ _[n]_ [+1] [)] _[ ⇐⇒]_ [(] _[n]_ [ + 1)] _[ ∈F]_ [(] _[A]_ [)] _[,]_


where the set _F_ ( _A_ ) is defined as in [2]:








(S6)
 _[.]_



_F_ ( _A_ ) =






 _[i][ ∈{]_ [1] _[, . . ., n]_ [ +] _[ m][}]_ [ :]



_n_ + _m_
�



� _A_ _i,j_ _≥_ (1 _−_ _α_ )( _n_ + 1)

_j_ =1



The rest of the proof follows directly by applying Lemma S1 below, which is established by the proof
of Theorem 4 in [ 2 ]. To invoke this lemma, we only need to check that _A_ = Σ _d_ _A_ Σ _⊤_, where _A_ is
defined as in (S5), and Σ is any permutation matrix that does not mix points assigned to different


2


folds (so that the unordered set of probability estimators _{π_ ˆ _[k]_ _}_ _[K]_ _k_ =1 [is invariant). This is easy to verify.]
Let _σ_ (1) _, . . ., σ_ ( _n_ + _m_ ) be the permutation of the data points corresponding to Σ, so that


(Σ _A_ Σ _[⊤]_ ) _ij_ = _A_ _σ_ ( _i_ ) _σ_ ( _j_ ) _._


Then, for any _i, j_ such that _k_ ( _i_ ) _̸_ = _k_ ( _j_ ),

_A_ _σ_ ( _i_ ) _σ_ ( _j_ ) = **1** � _E_ ( _X_ _σ_ ( _i_ ) _, Y_ _σ_ ( _i_ ) _, U_ _σ_ ( _i_ ) ; ˜ _π_ _[k]_ [(] _[σ]_ [(] _[i]_ [))] _[,k]_ [(] _[σ]_ [(] _[j]_ [))] ) _> E_ ( _X_ _σ_ ( _j_ ) _, Y_ _σ_ ( _j_ ) _, U_ _σ_ ( _j_ ) ; ˜ _π_ _[k]_ [(] _[σ]_ [(] _[i]_ [))] _[,k]_ [(] _[σ]_ [(] _[j]_ [))] )�

= **1** � _E_ ( _X_ _σ_ ( _i_ ) _, Y_ _σ_ ( _i_ ) _, U_ _σ_ ( _i_ ) ; ˜ _π_ _[k]_ [(] _[i]_ [)] _[,k]_ [(] _[j]_ [)] ) _> E_ ( _X_ _σ_ ( _j_ ) _, Y_ _σ_ ( _j_ ) _, U_ _σ_ ( _j_ ) ; ˜ _π_ _[k]_ [(] _[i]_ [)] _[,k]_ [(] _[j]_ [)] )�


_d_
= **1** _E_ ( _X_ _i_ _, Y_ _i_ _, U_ _i_ ; ˜ _π_ _[k]_ [(] _[i]_ [)] _[,k]_ [(] _[j]_ [)] ) _> E_ ( _X_ _j_ _, Y_ _j_ _, U_ _j_ ; ˜ _π_ _[k]_ [(] _[i]_ [)] _[,k]_ [(] _[j]_ [)] )
� �

= _A_ _ij_ _._


Above, the second equality holds because the black-box estimators ˆ _π_ _[k]_ are invariant to the ordering of
their input data points, and the third equality in distribution holds because the data points ( _X_ _i_ _, Y_ _i_ _, U_ _i_ )
are exchangeable. Finally, we also trivially know that _A_ _σ_ ( _i_ ) _σ_ ( _j_ ) = _A_ _ij_ for any _i, j_ such that
_k_ ( _i_ ) = _k_ ( _j_ ).


**Lemma S1** (Proved in [ 2 ]) **.** _Consider any partition of_ _{_ 1 _, . . ., n_ + _m}_ _points into_ _K_ + 1 _folds_

_I_ 1 _, . . ., I_ _K_ +1 _, with_ _m_ = _n/K_ _. If a random matrix_ _A ∈{_ 0 _,_ 1 _}_ [(] _[n]_ [+] _[m]_ [)] _[×]_ [(] _[n]_ [+] _[m]_ [)] _satisfies_ _A_ = Σ _d_ _A_ Σ _⊤_
_for any_ ( _n_ + _m_ ) _×_ ( _n_ + _m_ ) _permutation matrix_ Σ _that does not mix points assigned to different folds,_
_then, for any fixed α ∈_ (0 _,_ 1) _,_



2(1 _−_ 1 _/K_ )
P [( _n_ + 1) _∈F_ ( _A_ )] _≤_ 2 _α_ + min
� _n/K_ + 1



1 _−_ 1 _/K_ )

_n/K_ + 1 _[,]_ [ 1] _[ −]_ _K_ + 1 _[K][/][n]_



_,_ (S7)
�



_K_ + 1



_where the set_ _F_ ( _A_ ) _is defined as in_ (S6) _and depends on_ _α_ _. In the special case where_ _K_ = _n_ _, this_
_bound simplifies to:_


P [( _n_ + 1) _∈F_ ( _A_ )] _≤_ 2 _α._ (S8)


_Proof of Theorem S1._ The proof is effectively identical to that of Theorem 3 in [ 2 ], by the same
argument as in the proof of Theorem 2.


**S3** **Supplementary experiments with simulated data**


**S3.1** **Implementation details for black-box classifiers**


We have applied the following black-box classification methods to estimate label probabilities:


_•_ a support vector classifier ( SVC ) with linear kernel, as implemented by the `sklearn` Python
package with default parameters;

_•_ a random forest classifier ( RFC ) with 1000 estimators of maximum depth 5, as implemented
by the `sklearn` Python package with default parameters (except for the maximum number
of features considered at each split, which we set equal to _p_ ).


For the CQC method, we carry out quantile regression on the classification scores using the same
deep neural network employed in [3].


**S3.2** **Experiments with multinomial model and inhomogeneous features**


3


|Oracle|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**G**|**G**||||
||||||
||||||
|~~**G**~~<br>||**G**|||


|RFC|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||~~**G**~~||||
|~~**G**~~<br>**G**||~~**G**~~|~~**G**~~||


|SVC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~**G**~~|||||Marginal|
|||||||
|||||||
|~~**G**~~||**G**||||


|Oracle|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
|**G**~~**G**~~|**G**~~**G**~~|~~**G**~~<br>~~**G**~~**G**|~~**GG**~~||


|RFC|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|~~**G**~~**G**<br>**GG**|**G**~~**G**~~<br>**G**<br>~~**G**~~<br>~~**G**~~**G**<br>**G**||**G**<br>~~**G**~~<br>~~**G**~~<br>~~**G**~~||
||||||


|SVC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||||Marginal|
|||||||
|||||||
|**GG**~~**G**~~|**G**~~**G**~~|~~**G**~~**G**|~~**GG**~~|||


























|Col1|Col2|G GGG|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||~~**G**~~**G**|||||
|||||||


|Col1|Col2|G GGGGG|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|**G**|~~**G**~~|||||


|Col1|Col2|GGGGGG|Col4|Col5|Col6|Cond. on X1 = −8|
|---|---|---|---|---|---|---|
||||||||
||||||||
|**G**|**G**||||||
|~~**G**~~|~~**G**~~||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|~~**GGG**~~|~~**GG**~~|**G**<br>~~**G**~~<br>~~**G**~~**G**<br>~~**G**~~|~~**G**~~**G**||
||||||


|Col1|Col2|G|Col4|Col5|
|---|---|---|---|---|
||||||
|~~**G**~~<br>**G**<br><br>**G**|**G**<br><br><br>|~~**G**~~**G**|~~**G**~~||
|||**G**|||
||||||


|Col1|Col2|Col3|Col4|Col5|Cond. on X1 = −8|
|---|---|---|---|---|---|
|||||||
|||||||
|~~**GGG**~~|~~**GGG**~~|**G**~~**GG**~~<br>|~~**G**~~**G**|||
|||||||








































|Col1|G|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|||~~**G**~~<br>~~**G**~~**G**<br>~~**G**~~|||


|Col1|G<br>G|Col3|Col4|Col5|
|---|---|---|---|---|
|||~~**G**~~<br>~~**G**~~|||
|||~~**G**~~<br>~~**G**~~<br>|||


|Col1|Col2|Col3|Col4|Col5|Cond. on X1 = 1|
|---|---|---|---|---|---|
|||||||
|||~~**GG**~~||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**G**|**G**|~~**G**~~|||
||**G**||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||**GG**||
||||||


|Col1|Col2|G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G|G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G G|Col5|−8|
|---|---|---|---|---|---|
||||||Cond. on X1 = 1|
|~~**G**~~<br>|~~**G**~~<br>|**G**||||
|||||||













(a)



(b)



Figure S1: Different classification methods on simulated data with 10 classes, for different choices
of calibration and black-box models. The results correspond to 100 independent experiments with
10000 training samples and 5000 test samples each. JK+ is omitted for computational reasons. Other
details are as in Figure 1.


4


**S3.3** **Experiments with heteroscedastic decision-tree model and discrete features**


We set _p_ = 5 and generate each sample of features _X ∈_ R _[p]_ independently as follows: _X_ 1 = +1 w.p.
3 _/_ 4, and _X_ 1 = _−_ 1 w.p. 1 _/_ 4 ; _X_ 2 = +1 w.p. 3 _/_ 4, and _X_ 2 = _−_ 2 w.p. 1 _/_ 4 ; _X_ 3 = +1 w.p. 1 _/_ 4, and
_X_ 3 = _−_ 2 w.p. 1 _/_ 2 ; _X_ 4 is uniformly distributed on _{_ 1 _, . . .,_ 4 _}_ ; and _X_ 5 _∼N_ (0 _,_ 1) . The labels _Y_
belong to one of 4 possible classes, and their conditional distribution given _X_ = _x_ is given by the
decision tree shown in Figure S2, which only depends on the first four features.



( [1]



30 [1] _[,]_ [1]



30 [1] _[,]_ [1]



30 [1] _[,]_ [9]



10 [)]



( [1]



30 [1] _[,]_ [1]



30 [1] _[,]_ [9]



10 [9] _[,]_ [1]



30 [)]



10 [9] _[,]_ [1]



30 [1] _[,]_ [1]



30 [)]





( [1]



30 [1] _[,]_ [9]



_X_ 1 = +1


P[ _Y |X_ ]











10 [9] _[,]_ [1]

















( [9]



30 [1] _[,]_ [1]



30 [1] _[,]_ [1]



30 [)]



Figure S2: A toy model for _P_ _Y |X_ in a classification setting with 4 labels.


The performances of the different methods on data generated from this model are compared in
Figure S3. Here, the size of the training sample is equal to 10000 and the size of the test sample
is equal to 5000; all experiments are repeated 100 times. Since these training sets are fairly large,
for computational convenience we do not apply the JK+ method; see Figure S4 for a comparison
including JK+ with smaller sample sizes.

These results are qualitatively consistent with those from Section 3.2, confirming that our methods
have good approximate conditional coverage compared to the alternatives while not suffering from
a significant power loss. It is interesting to note that the conditional distribution of _Y | X_ is more
complicated here than in the previous example, hence the reason for a larger sample size. Despite this
large sample size, the SVC black-box is unable to learn good estimates of the class probabilities. This
is why methods with marginal coverage have relatively low power and poor conditional coverage. By
contrast, the RFC black-box can learn these class probabilities quite accurately, and thus it allows our
SC and CV+ methods to perform on par with the oracle (especially CV+, as expected). Again, the
alternative methods do not achieve conditional coverage even with the help of the oracle.


5


|Oracle|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||~~**G**~~||||
||~~**G**~~**G**||||


|RFC|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|**G**|||**G**||
||||||


|SVC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||~~**G**~~||||Marginal|
||||~~**G**~~<br>~~**G**~~|||
||||**G**|||


|Oracle|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|~~**G**~~|~~**GGGG**~~||||


|RFC|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|**G**||**GG**|||


|SVC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||||Marginal|
|~~**G**~~||**G**||||
|||||||




















|Col1|Col2|Col3|Col4|G|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||


|Col1|G|Col3|G|Col5|
|---|---|---|---|---|
|~~**G**~~|||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|Conditional (WS)|
|---|---|---|---|---|---|
|~~**G**~~||||||
|||||||
|||||||
||||**GG**|||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|~~**G**~~|~~**GGG**~~||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|**G**||~~**GG**~~|||
||||||


|Col1|Col2|Col3|Col4|Col5|Cond. on cover|
|---|---|---|---|---|---|
|~~**G**~~||~~**G**~~||||
|||||||
|||||||
|||||||
























|G<br>G|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||**GG**|**GG**||
||||||||


|G|Col2|Col3|G|Col5|
|---|---|---|---|---|
||||||
||||||


|Col1|G|Col3|Col4|Col5|Cond. on X0 = −1|
|---|---|---|---|---|---|
|~~**G**~~||||||
|||||||


|G|G<br>G|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||**G**|**G**||


|G|Col2|Col3|G|Col5|
|---|---|---|---|---|
||||||


|Col1|Col2|Col3|Col4|Col5|on cover|
|---|---|---|---|---|---|
||~~**G**~~||||Cond. on X1 = −1|
|||**G**||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||**G**||||**G**||
||||||||
||||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||**GG**||
||||||
||||||


|G|Col2|Col3|Col4|Col5|Cond. on X0 = 1|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||~~**GG**~~||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||


|Col1|Col2|G|Col4|Col5|Cond. on X0 = 1|
|---|---|---|---|---|---|
|||||||
|||||||



(a)



(b)



Figure S3: Performance of alternative classification methods on simulated data with 4 classes. Results
from 100 independent experiments with 10000 training samples and 5000 test samples each. Other
details as in Figure 1.


6


|Oracle|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


|RFC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||**G**||~~**G**~~|**G**||


|SVC|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||||Marginal|
|||**G**|~~**G**~~||||
|~~**G**~~||**G**|~~**G**~~<br>**G**||||


|Oracle|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|**GG**||||**G**<br>**G**||


|RFC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|~~**G**~~||||||
||**G**||**G**|~~**GGG**~~||


|SVC|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||||Marginal|
|||**G**|||||
||||**G**||||


|Col1|Col2|Col3|Col4|G|Col6|
|---|---|---|---|---|---|
|||||||


|Col1|GG|G|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||~~**G**~~|||


|Col1|Col2|G|Col4|Col5|Col6|Conditional (WS)|
|---|---|---|---|---|---|---|
|**G**||~~**GG**~~|**G**||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~**G**~~**G**||~~**G**~~|**G**|~~**G**~~||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||~~**G**~~|~~**G**~~|~~**GG**~~|**G**~~**GG**~~||


|GG<br>GG|G|G|Col4|Col5|Col6|Cond. on cover|
|---|---|---|---|---|---|---|
||||||||






















































|G|G|Col3|GG GGG|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||


|GG|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||**G**|**G**|||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Cond. on X0 = −1|
|---|---|---|---|---|---|---|
|||~~**G**~~|||||
||||||||


|GG|Col2|Col3|GG GGG|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||


|G|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~**G**~~|~~**G**~~||**G**|||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|on cover|
|---|---|---|---|---|---|---|
|||||||Cond. on X1 = −1|
||||||||
||||||||














|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||**G**||


|G|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~**G**~~|**G**|**G**||~~**GG**~~||


|G|G|Col3|Col4|Col5|Col6|−1|
|---|---|---|---|---|---|---|
|||**GG**||||Cond. on X0 =|
|~~**G**~~|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~**GG**~~||**GG**||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**G**|~~**GG**~~|**G**|~~**G**~~**G**|**GG**<br>**G**||


|Col1|Col2|Col3|Col4|Col5|Col6|−1|
|---|---|---|---|---|---|---|
|**G**<br>~~**G**~~|~~**G**~~|~~**G**~~<br><br>**G**||||Cond. on X0 =|
||||||||









(a)



(b)



Figure S4: Performance of alternative classification methods on simulated data with 4 classes. Results
from 100 independent experiments with 1000 training samples and 5000 test samples each. Other
details are as in Figure S3.


7


**S4** **Supplementary experiments with real data**


We compare the performance of our methods to that of HCC and CQC on four popular benchmark
data sets:


1. MNIST is a handwritten digit classification data set, containing 60000 grayscale images of
size 28 _×_ 28 pixels, each associated with one of _C_ = 10 classes. As a pre-processing step,
we apply Principal Component Analysis (PCA) to each image, resulting in a feature vector
_X_ of length _p_ = 50.
2. CIFAR10 is another image classification data set. The data includes 50000 RGB images,
each of size 32 _×_ 32 _×_ 3, belonging to one of _C_ = 10 classes. We also use PCA to reduce
the dimension to _p_ = 50.

3. Fashion-MNIST contains 60000 images associated with _C_ = 10 classes of clothes. We run
the same pre-processing step as in MNIST and CIFAR10, resulting in _p_ = 50 features.
4. The task in the Mice Protein Expression data set [2] is to identify the class of a mouse based
on genetic, behavioral and treatment covariates. After applying standard data cleaning, we
have 1080 samples, _p_ = 77 features, and _C_ = 8 classes.


We use the same baseline predictive algorithms as in Section S3.1, although with slightly different
RFC parameters—here the number of estimators is 100, and the minimum number of samples at
a leaf node is 3. Additionally, we also consider a neural network (NNet) with one hidden layer of
size 64 and ReLU activation function. We use the adam optimizer, with a minibatch of size 128, a
learning rate of 0 _._ 01, and a total number of epochs equal to 20. The CQC method is implemented as
described in Section S3.1. In the real data experiments we present a second variant of CQC, namely
CQC-RF, where we replace the quantile neural network algorithm with quantile random forest. To
this end, we use the default `skgarden` hyper-parameters for quantile random forest, except for the
number of estimators and the minimum number of samples required to split an internal node, which
we set to 100 and 3, respectively.


In the numerical experiments, we set the target coverage level to 90% and compare the coverage,
conditional coverage, and length of the different calibration methods combined with the above
predictive algorithms. The performance metrics are averaged over 100 experiments. The results
in Table S1 are obtained by randomly selecting _n_ train _∈{_ 500 _,_ 1000 _}_ training examples from the
Mice Protein Expression data set, used to fit and calibrate the predictive models. The remaining
_n_ test _∈{_ 580 _,_ 80 _}_ unseen samples formulate a test set, in which we evaluate the methods’ performance.
Tables S2, S3, and S4 correspond to MNIST, Fashion-MNIST, and CIFAR10 data sets. Each
experiment is conducted by randomly selecting _n_ train _∈{_ 500 _,_ 1000 _,_ 5000 _,_ 10000 _}_ training examples
as well as a disjoint set of 5000 unseen test samples, selected at random.


In sum, all the calibration methods achieve an exact 90% marginal coverage, as guaranteed by the
theory. CV+ and JK+ tend to achieve conditional coverage as well (green colored numbers), and
SC performs slightly worse. In contrast, in most cases CQC, CQC-RF, and HCC fail (red colored
numbers) to obtain the desired conditional coverage. As for the statistical efficiently, HCC often
results in the shortest prediction sets—while failing to attain conditional coverage. Here, our methods
are typically competitive and can even produce smaller prediction sets in some cases.


**S5** **Supplementary tables**


2 `https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression`


8


9


10


11


12


**References**


[1] M. Cauchois, S. Gupta, and J. Duchi. Knowing what you know: valid confidence sets in
multiclass and multilabel prediction. _arXiv preprint arXiv:2004.10181_, 2020.


[2] R. Foygel Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani. Predictive inference with the
jackknife+. _arXiv preprint arXiv:1905.02928_, 2019.


[3] Y. Romano, E. Patterson, and E. J. Candès. Conformalized quantile regression. In _Advances in_
_Neural Information Processing Systems_, pages 3538–3548, 2019.


13


