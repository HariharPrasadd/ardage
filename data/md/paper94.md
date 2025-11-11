## The Complexity of Constrained Min-Max Optimization



Constantinos Daskalakis

MIT

```
costis@csail.mit.edu

```


Stratis Skoulakis

SUTD

```
efstratios@sutd.edu.sg

```


Manolis Zampetakis
MIT

```
 mzampet@mit.edu

```


September 22, 2020


**Abstract**


Despite its important applications in Machine Learning, min-max optimization of objective
functions that are _nonconvex-nonconcave_ remains elusive. Not only are there no known firstorder methods converging even to approximate local min-max points, but the computational
complexity of identifying them is also poorly understood. In this paper, we provide a characterization of the computational complexity of the problem, as well as of the limitations of firstorder methods in constrained min-max optimization problems with nonconvex-nonconcave
objectives and linear constraints.
As a warm-up, we show that, even when the objective is a Lipschitz and smooth differentiable function, deciding whether a min-max point exists, in fact even deciding whether an
approximate min-max point exists, is NP-hard. More importantly, we show that an approximate local min-max point of large enough approximation is guaranteed to exist, but finding
one such point is PPAD-complete. The same is true of computing an approximate fixed point
of the (Projected) Gradient Descent/Ascent update dynamics.
An important byproduct of our proof is to establish an unconditional hardness result
in the Nemirovsky-Yudin [NY83] oracle optimization model. We show that, given oracle
access to some function _f_ : _P →_ [ _−_ 1, 1 ] and its gradient _∇_ _f_, where _P ⊆_ [ 0, 1 ] _[d]_ is a known
convex polytope, every algorithm that finds a _ε_ -approximate local min-max point needs to
make a number of queries that is exponential in at least one of 1/ _ε_, _L_, _G_, or _d_, where _L_
and _G_ are respectively the smoothness and Lipschitzness of _f_ and _d_ is the dimension. This
comes in sharp contrast to minimization problems, where finding approximate local minima
in the same setting can be done with Projected Gradient Descent using _O_ ( _L_ / _ε_ ) many queries.
Our result is the first to show an exponential separation between these two fundamental
optimization problems in the oracle model.


#### **Contents**

**1** **Introduction** **1**

1.1 Brief Overview of the Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.2 Local Minimization vs Local Min-Max Optimization . . . . . . . . . . . . . . . . . . 6
1.3 Further Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7


**2** **Preliminaries** **9**


**3** **Computational Problems of Interest** **12**
3.1 Mathematical Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
3.2 First-Order Local Optimization Computational Problems . . . . . . . . . . . . . . . . 14
3.3 Bonus Problems: Fixed Points of Gradient Descent/Gradient Descent-Ascent . . . . 15


**4** **Summary of Results** **16**


**5** **Existence of Approximate Local Min-Max Equilibrium** **18**


**6** **Hardness of Local Min-Max Equilibrium – Four-Dimensions** **19**
6.1 The 2D Bi-Sperner Problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
6.2 From 2D Bi-Sperner to Fixed Points of Gradient Descent/Ascent . . . . . . . . . . . 23


**7** **Hardness of Local Min-Max Equilibrium – High-Dimensions** **31**
7.1 The High Dimensional Bi-Sperner Problem . . . . . . . . . . . . . . . . . . . . . . . . 32
7.2 From High Dimensional Bi-Sperner to Fixed Points of Gradient Descent/Ascent . . 34


**8** **Smooth and Efficient Interpolation Coefficients** **40**
8.1 Smooth Step Functions – Toy Single Dimensional Example . . . . . . . . . . . . . . . 41
8.2 Construction of SEIC Coefficients in High-Dimensions . . . . . . . . . . . . . . . . . 43
8.3 Sketch of the Proof of Theorem 8.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45


**9** **Unconditional Black-Box Lower Bounds** **45**


**10 Hardness in the Global Regime** **47**


**A Proof of Theorem 4.1** **57**


**B** **Missing Proofs from Section 5** **58**
B.1 Proof of Theorem 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58

B.2 Proof of Theorem 5.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60


**C Missing Proofs from Section 8** **61**


**D Constructing the Turing Machine – Proof of Theorem 7.6** **74**


**E** **Convergence of PGD to Approximate Local Minimum** **81**


#### **1 Introduction**

_Min-Max Optimization_ has played a central role in the development of Game Theory [vN28],
Convex Optimization [Dan51, Adl13], and Online Learning [Bla56, CBL06, SS12, BCB12, SSBD14,
Haz16]. In its general constrained form, it can be written down as follows:


_**x**_ min _∈_ **R** _[d]_ [1] _**y**_ [max] _∈_ **R** _[d]_ [2] _[f]_ [ (] _**[x]**_ [,] _**[ y]**_ [)] [;] (1.1)


s.t. _g_ ( _**x**_, _**y**_ ) _≤_ 0.


Here, _f_ : **R** _[d]_ [1] _×_ **R** _[d]_ [2] _→_ [ _−_ _B_, _B_ ] with _B_ _∈_ **R** +, and _g_ : **R** _[d]_ [1] _×_ **R** _[d]_ [2] _→_ **R** is typically taken to be a
convex function so that the constraint set _g_ ( _**x**_, _**y**_ ) _≤_ 0 is convex. In this paper, we only use linear
functions _g_ so the constraint set is a polytope, thus projecting on this set and checking feasibility
of a point with respect to this set can both be done in polynomial time.
The goal in (1.1) is to find a feasible pair ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), i.e., _g_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _≤_ 0, that satisfies the following


_f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _≤_ _f_ ( _**x**_, _**y**_ _[⋆]_ ), for all _**x**_ s.t. _g_ ( _**x**_, _**y**_ _[⋆]_ ) _≤_ 0; (1.2)


_f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _≥_ _f_ ( _**x**_ _[⋆]_, _**y**_ ), for all _**y**_ s.t. _g_ ( _**x**_ _[⋆]_, _**y**_ ) _≤_ 0. (1.3)


It is well-known that, when _f_ ( _**x**_, _**y**_ ) is a convex-concave function, i.e., _f_ is convex in _**x**_ for
all _**y**_ and it is concave in _**y**_ for all _**x**_, then Problem (1.1) is guaranteed to have a solution, under
compactness of the constraint set [vN28, Ros65], while computing a solution is amenable to
convex programming. In fact, if _f_ is _L_ -smooth, the problem can be solved via first-order methods,
which are iterative, only access _f_ through its gradient, [1] and achieve an approximation error of
poly ( _L_, 1/ _T_ ) in _T_ iterations; see e.g. [Kor76, Nem04]. [2] When the function is strongly convexstrongly concave, the rate becomes geometric [FP07].
Unfortunately, our ability to solve Problem (1.1) remains rather poor in settings where our objective function _f_ is _not_ convex-concave. This is emerging as a major challenge in Deep Learning,
where min-max optimization has recently found many important applications, such as training Generative Adversarial Networks (see e.g. [GPM [+] 14, ACB17]), and robustifying deep neural
network-based models against adversarial attacks (see e.g. [MMS [+] 18]). These applications are
indicative of a broader deep learning paradigm wherein robustness properties of a deep learning
system are tested and enforced by another deep learning system. In these applications, it is very
common to encounter min-max problems with objectives that are nonconvex-nonconcave, and
thus evade treatment by the classical algorithmic toolkit targeting convex-concave objectives.
Indeed, the optimization challenges posed by objectives that are nonconvex-nonconcave are
not just theoretical frustration. Practical experience with first-order methods is rife with frustration as well. A common experience is that the training dynamics of first-order methods is unstable, oscillatory or divergent, and the quality of the points encountered in the course of training
can be poor; see e.g. [Goo16, MPPSD16, DISZ18, MGN18, DP18, MR18, MPP18, ADLH19]. This
experience is in stark contrast to minimization (resp. maximization) problems, where even for


1
In general, the access to the constraints _g_ by these methods is more involved, namely through an optimization
oracle that optimizes convex functions (in fact, quadratic suffices) over _g_ ( _**x**_, _**y**_ ) _≤_ 0. In the settings considered in this
paper _g_ is linear and these tasks are computationally straightforward.
2 In the stated error rate, we are suppressing factors that depend on the diameter of the feasible set. Moreover, the
stated error of _ε_ ( _L_, _T_ ) ≜ poly ( _L_, 1/ _T_ ) reflects that these methods return an approximate min-max solution, wherein
the inequalities on the LHS of (1.2) and (1.3) are satisfied to within an additive _ε_ ( _L_, _T_ ) .


1


nonconvex (resp. nonconcave) objectives, first-order methods have been found to efficiently converge to approximate local optima or stationary points (see e.g. [AAZB [+] 17, JGN [+] 17, LPP [+] 19]),
while practical methods such Stochastic Gradient Descent, Adagrad, and Adam [DHS11, KB14,
RKK18] are driving much of the recent progress in Deep Learning.


The goal of this paper is to _shed light on the complexity of min-max optimization problems_, and
_elucidate its difference to minimization and maximization problems_ —as far as the latter is concerned
without loss of generality we focus on minimization problems, as maximization problems behave
exactly the same; we will also think of minimization problems in the framework of (1.1), where
the variable _**y**_ is absent, that is _d_ 2 = 0. An important driver of our comparison between min-max
optimization and minimization is, of course, the nature of the objective. So let us discuss:


_▷_ _Convex-Concave Objective._ The benign setting for min-max optimization is that where the objective function is convex-concave, while the benign setting for minimization is that where the
objective function is convex. In their corresponding benign settings, the two problems behave
quite similarly from a computational perspective in that they are amenable to convex programming, as well as first-order methods which only require gradient information about the objective
function. Moreover, in their benign settings, both problems have guaranteed existence of a solution under compactness of the constraint set. Finally, it is clear how to define approximate
solutions. We just relax the inequalities on the left hand side of (1.2) and (1.3) by some _ε_ _>_ 0.


_▷_ _Nonconvex-Nonconcave Objective._ By contrapositive, the challenging setting for min-max optimization is that where the objective is _not_ convex-concave, while the challenging setting for
minimization is that where the objective is not convex. In these challenging settings, the behavior of the two problems diverges significantly. The first difference is that, while a solution to a
minimization problem is still guaranteed to exist under compactness of the constraint set even
when the objective is not convex, a solution to a min-max problem is _not_ guaranteed to exist
when the objective is not convex-concave, even under compactness of the constrained set. A trivial example is this: min _x_ _∈_ [ 0,1 ] max _y_ _∈_ [ 0,1 ] ( _x_ _−_ _y_ ) [2] . Unsurprisingly, we show that checking whether a
min-max optimization problem has a solution is NP-hard. In fact, we show that checking whether
there is an approximate min-max solution is NP-hard, even when the function is Lispchitz and
smooth and the desired approximation error is an absolute constant (see Theorem 10.1).


Since min-max solutions may not exist, what could we plausibly hope to compute? There are
two obvious targets:


(I) approximate stationary points of _f_, as considered e.g. by [ALW19]; and


(II) some type of approximate _local_ min-max solution.


Unfortunately, as far as (I) is concerned, it is still possible that (even approximate) stationary points may not exist, and we show that checking if there is one is NP-hard, even when
the constraint set is [ 0, 1 ] _[d]_, the objective has Lipschitzness and smoothness polynomial in _d_,
and the desired approximation is an absolute constant (Theorem 4.1). So we focus on (II),
i.e. (approximate) local min-max solutions. Several kinds of those have been proposed in the
literature [DP18, MR18, JNJ19]. We consider a generalization of the concept of local min-max
equilibria, proposed in [DP18, MR18], that also accommodates approximation.


2


**Definition 1.1** (Approximate Local Min-Max Equilibrium) **.** Given _f_, _g_ as above, and _ε_, _δ_ _>_ 0,
some point ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an ( _ε_, _δ_ ) _-local min-max solution of_ (1.1), or a ( _ε_, _δ_ ) _-local min-max equilibrium_,
if it is feasible, i.e. _g_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _≤_ 0, and satisfies:


_f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _<_ _f_ ( _**x**_, _**y**_ _[⋆]_ ) + _ε_, for all _**x**_ such that _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥≤_ _δ_ and _g_ ( _**x**_, _**y**_ _[⋆]_ ) _≤_ 0; (1.4)


_f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _>_ _f_ ( _**x**_ _[⋆]_, _**y**_ ) _−_ _ε_, for all _**y**_ such that _∥_ _**y**_ _−_ _**y**_ _[⋆]_ _∥≤_ _δ_ and _g_ ( _**x**_ _[⋆]_, _**y**_ ) _≤_ 0. (1.5)


In words, ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an ( _ε_, _δ_ ) -local min-max equilibrium, whenever the min player cannot update
_**x**_ to a feasible point within _δ_ of _**x**_ _[⋆]_ to reduce _f_ by at least _ε_, and symmetrically the max player
cannot change _**y**_ locally to increase _f_ by at least _ε_ .


We show that the existence and complexity of computing such approximate local min-max
equilibria depends on the relationship of _ε_ and _δ_ with the smoothness, _L_, and the Lipschitzness,
_G_, of the objective function _f_ . We distinguish the following regimes, also shown in Figure 1
together with a summary of our associated results.

- **Trivial Regime.** This occurs when _δ_ _<_ _G_ _[ε]_ [. This regime is trivial because the] _[ G]_ [-Lipschitzness of]

_f_ guarantees that all feasible points are ( _ε_, _δ_ ) -local min-max solutions.




- **Local Regime.** This occurs when _δ_ _<_
�



2 _ε_




- **Local Regime.** This occurs when _δ_ _<_ _Lε_ [, and it represents the interesting regime for min-]

max optimization. In this regime, we use the smoothness of _f_ to show that ( _ε_, _δ_ ) -local min-max
solutions always exist. Indeed, we show (Theorem 5.1) that computing them is computationally
equivalent to the following variant of (I) which is more suitable for the constrained setting:


(I’) (approximate) fixed points of the projected gradient descent-ascent dynamics (Section 3.3).


We show via an application of Brouwer’s fixed point theorem to the iteration map of the projected
gradient descent-ascent dynamics that (I)’ are guaranteed to exist. In fact, not only do they exist,
but computing them is in PPAD, as can be shown by bounding the Lipschitzness of the projected
gradient descent-ascent dynamics (Theorem 5.2).

- **Global Regime.** This occurs when _δ_ is comparable to the diameter of the constraint set. In
this case, the existence of ( _ε_, _δ_ ) -local min-max solutions is not guaranteed, and determining their
existence is NP-hard, even if _ε_ is an absolute constant (Theorem 10.1).


The main results of this paper, summarized in Figure 1, are to characterize the complexity of
computing local min-max solutions in the local regime. Our first main theorem is the following:


**Informal Theorem 1** (see Theorems 4.3, 4.4 and 5.1) **.** _Computing_ ( _ε_, _δ_ ) _-local min-max solutions of_
_Lipschitz and smooth objectives over convex compact domains in the local regime is_ PPAD _-complete. The_
_hardness holds even when the constraint set is a polytope that is a subset of_ [ 0, 1 ] _[d]_ _, the objective takes values_
_in_ [ _−_ 1, 1 ] _and the smoothness, Lipschitzness,_ 1/ _ε and_ 1/ _δ are polynomial in the dimension. Equivalently,_
_computing α-approximate fixed points of the Projected Gradient Descent-Ascent dynamics on smooth and_
_Lipschitz objectives is_ PPAD _-complete, and the hardness holds even when the the constraint set is a polytope_
_that is a subset of_ [ 0, 1 ] _[d]_ _, the objective takes values in_ [ _−_ _d_, _d_ ] _and smoothness, Lipschitzness, and_ 1/ _α are_
_polynomial in the dimension._


For the above complexity result we assume that we have “white box” access to the objective
function. An important byproduct of our proof, however, is to also establish an _unconditional_
_hardness result_ in the Nemirovsky-Yudin [NY83] oracle optimization model, wherein we are given
black-box access to oracles computing the objective function and its gradient. Our second main
2.
result is informally stated in Informal Theorem


3


Figure 1: Overview of the results proven in this paper and comparison between the complexity
of computing an ( _ε_, _δ_ ) -approximate local minimum and an ( _ε_, _δ_ ) -approximate local min-max
equilibrium of a _G_ -Lipschitz and _L_ -smooth function over a _d_ -dimensional polytope taking values
in the interval [ _−_ _B_, _B_ ] . We assume that _ε_ _<_ _G_ [2] / _L_, thus the trivial regime is a strict subset of
the local regime. Moreover, we assume that the approximation parameter _ε_ is provided in unary
representation in the input to these problems, which makes our hardness results stronger and
the comparison to the upper bounds known for finding approximate local minima fair, as these
require time/oracle queries that are polynomial in 1/ _ε_ . We note that the unary representation
is not required for our results proving inclusion in PPAD. The figure portrays a sharp contrast
between the computational complexity of approximate local minima and approximate local minmax equilibria in the local regime. Above the black lines, tracking the value of _δ_, we state our
“white box” results and below the black lines we state our “black-box” results. The main result
of this paper is the PPAD-hardness of approximate local min-max equilibrium for _δ_ _≥_ _[√]_ _ε_ / _L_ and



of this paper is the PPAD-hardness of approximate local min-max equilibrium for _δ_ _≥_ _[√]_ _ε_ / _L_ and

the corresponding query lower bound. In the query lower bound the function _h_ is defined as
_h_ ( _d_, _G_, _L_, _ε_ ) = �min ( _d_, _[√]_ _L_ / _ε_, _G_ / _ε_ ) � _p_ for some universal constant _p_ _∈_ **R** + . With _⋆_ we indicate our



_h_ ( _d_, _G_, _L_, _ε_ ) = �min ( _d_, _[√]_ _L_ / _ε_, _G_ / _ε_ ) � _p_ for some universal constant _p_ _∈_ **R** + . With _⋆_ we indicate our

PPAD-completeness result which directly follows from Theorems 4.3 and 4.4. The NP-hardess
results in the global regime are presented in Section 10. Finally, the folklore result showing the
tractability of finding approximate local minima is presented for completeness of exposition in
Appendix E. The claimed results for the trivial regime follow from the definition of Lipschitzness.



4


**Informal Theorem 2** (see Theorem 4.5) **.** _Assume that we have black-box access to an oracle computing_
_a G-Lipschitz and L-smooth objective function f_ : _P →_ [ _−_ 1, 1 ] _, where_ _P ⊆_ [ 0, 1 ] _[d]_ _is a known polytope,_
_and its gradient_ _∇_ _f. Then, computing an_ ( _ε_, _δ_ ) _-local min-max solution in the local regime (i.e., when_
_δ_ _<_ _[√]_ 2 _ε_ / _L) requires a number of oracle queries that is exponential in at least one of the following:_ 1/ _ε,_

_L, G, or d. In fact, exponential in d-many queries are required even when L, G,_ 1/ _ε and_ 1/ _δ are all_
_polynomial in d._


Importantly, the above lower bounds, in both the white-box and the black-box setting, come
in sharp contrast to minimization problems, given that finding approximate local minima of
smooth non-convex objectives ranging in [ _−_ _B_, _B_ ] in the local regime can be done using firstorder methods using _O_ ( _B_ _·_ _L_ / _ε_ ) time/queries (see Section E). Our results are the first to show an
exponential separation between these two fundamental problems in optimization in the blackbox setting, and a super-polynomial separation in the white-box setting assuming PPAD _̸_ = FP.


**1.1** **Brief Overview of the Techniques**


We very briefly outline some of the main ideas for the PPAD-hardness proof that we present
in Sections 6 and 7. Our starting point as in many PPAD-hardness results is a discrete analog
of the problem of finding Brouwer fixed points of a continuous map. Departing from previous
work, however, we do not use Sperner’s lemma as the discrete analog of Brouwer’s fixed point
theorem. Instead, we define a new problem, called BiSperner, which is useful for showing our
hardness results. BiSperner is closely related to the problem of finding panchromatic simplices
guaranteed by Sperner’s lemma except, roughly speaking, that the vertices of the simplicization
of a _d_ -dimensional hypercube are colored with 2 _d_ rather than _d_ + 1 colors, every point of the
simplicization is colored with _d_ colors rather than one, and we are seeking a vertex of the simplicization so that the union of colors on the vertices in its neighborhood covers the full set of
colors. The first step of our proof is to show that BiSperner is PPAD-hard. This step follows
from the hardness of computing Brouwer fixed points.


The step that we describe next is only implicitly done by our proof, but it serves as useful
intuition for reading and understanding it. We want to define a _discrete_ two-player zero-sum
game whose local equilibrium points correspond to solutions of a given BiSperner instance.
Our two players, called “minimizer” and “maximizer,” each choose a vertex of the simplicization
of the BiSperner instance. For every pair of strategies in our discrete game, i.e. vertices, chosen
by our players, we define a function value and gradient values. Note that, at this point, we
treat these values at different vertices of the simplicization as independent choices, i.e. are not
defining a function over the continuum whose function values and gradient values are consistent
with these choices. It is our intention, however, that in the _continuous_ two-player zero-sum game
that we obtain in the next paragraph via our interpolation scheme, wherein the minimizer and
maximizer may choose any point in the continuous hypercube, the function value determines
the payment of the minimizer to the maximizer, and the gradient value determines the direction
of the best-response dynamics of the game. Before getting to that continuous game in the next
paragraph, the main technical step of this discrete part of our construction is showing that every
local equilibrium of the discrete game corresponds to a solution of the BiSperner instance we are
reducing from. In order to achieve this we need to add some constraints to couple the strategies
of the minimizer and the maximizer player. This step is the reason that the constraints _g_ ( _**x**_, _**y**_ ) _≤_ 0
appear in the final min-max problem that we produce.


5


The third and quite challenging step of the proof is to show that we can interpolate in a
_smooth_ and _computationally efficient_ way the discrete zero-sum game of the previous step. In
low dimensions (treated in Section 6) such smooth and efficient interpolation can be done in
a relatively simple way using single-dimensional smooth step functions. In high dimensions,
however, the smooth and efficient interpolation becomes a challenging problem and to the best
of our knowledge no simple solution exists. For this reason we construct our novel _smooth and_
_efficient interpolation coefficients_ of Section 8. These are a technically involved construction that we
believe will prove to be very useful for characterizing the complexity of approximate solutions
of other optimization problems.


The last part of our proof is to show that all the previous steps can be implemented in
an efficient way both with respect to computational but also with respect to query complexity.
This part is essential for both our white-box and black-box results. Although this seems like a
relatively easy step, it becomes more difficult due to the complicated expressions in our smooth
and efficient interpolation coefficients used in our previous step.


Closing this section we mention that all our NP-hardness results are proven using a cute
application of Lovász Local Lemma [EL73], which provides a powerful rounding tool that can
drive the inapproximability all the way up to an absolute constant.


**1.2** **Local Minimization vs Local Min-Max Optimization**


Because our proof is convoluted, involving multiple steps, it is difficult to discern from it why
finding local min-max solutions is so much harder than finding local minima. For this reason, we
illustrate in this section a fundamental difference between local minimization and local min-max

optimization. This provides good intuition about why our hardness construction would fail if we
tried to apply it to prove hardness results for finding local minima (which we know don’t exist).
So let us illustrate a key difference between min-max problems that can be expressed in
the form min _x_ _∈X_ max _y_ _∈Y_ _f_ ( _x_, _y_ ), i.e. two-player zero-sum games wherein the players optimize
opposing objectives, and min-min problems of the form min _x_ _∈X_ min _y_ _∈Y_ _f_ ( _x_, _y_ ), i.e., two-player
coordination games wherein the players optimize the same objective. For simplicity, suppose
_X_ = _Y_ = **R** and let us consider long paths of best-response dynamics in the strategy space,
_X × Y_, of the two players; these are paths along which at least one of the players improves their
payoff. For our illustration, suppose that the derivative of the function with respect to either
variable is either 1 or _−_ 1. Consider a long path of best-response dynamics starting at a pair of
strategies ( _x_ 0, _y_ 0 ) in either a min-min problem or a min-max problem, and a specific point ( _x_, _y_ )
along that path. We claim that in min-min problems the function value at ( _x_, _y_ ) will have to
reveal how far from ( _x_ 0, _y_ 0 ) point ( _x_, _y_ ) lies within the path in _ℓ_ 1 distance. On the other hand,
in min-max problems the function value at ( _x_, _y_ ) may reveal very little about how far ( _x_, _y_ ) lies
from ( _x_ 0, _y_ 0 ) . We illustrate this in Figure 2. While in our min-min example the function value
must be monotonically decreasing inside the best-response path, in the min-max example the
function values repeat themselves in every straight line segment of length 3, without revealing
where in the path each segment is.
Ultimately a key difference between min-min and min-max optimization is that best-response
paths in min-max optimization problems can be closed, i.e., can form a cycle, as shown in Figure
2, Panel (b). On the other hand, this is impossible in min-min problems as the function value
must monotonically decrease along best-response paths, thus cycles may not exist.


6


(a) Min-min problem; the function values reveal the location of the points within best response path.



(b) Min-max problem; the function values do
not reveal the location of the points within
best response path.



Figure 2: Long paths of best-response dynamics in min-min problems (Panel (a)) and min-max
problems (Panel (b)), where horizontal moves correspond to one player (who is a minimizer in
both (a) and (b)) and vertical moves correspond to the other player (who is minimizer in (a) but a
maximizer in (b)). In Panels (a) and (b), we show the function value at a subset of discrete points
in a 2D grid along a long path of best-response dynamics, where for our illustration we assumed
that the derivative of the objective with respect to either variable always has absolute value 2.
As we see in Panel (a), the function value at some point along a long path of the best-response
dynamics in a min-min problem reveals information about where in the path that point lies.
This is in sharp contrast to min-max problems where only local information is revealed about
the objective as shown in Panel (b), due to the frequent turns of the path. In Panel (b) we also
show that the best-response dynamics in min-max problems can form closed paths. This cannot
happen in min-min problems as the function value must decrease along paths of best-response
dynamics, and hence it is impossible in min-min problems to build long best-response paths with
function values that can be computed locally.


The above discussion offers qualitative differences between min-min and min-max optimization, which lie in the heart of why our computational intractability results are possible to prove
for min-max but not min-min problems. For the precise step in our construction that breaks if
we were to switch from a min-max to a min-min problem we refer the reader to Remark 6.9.


**1.3** **Further Related Work**


There is a broad literature on the complexity of equilibrium computation. Virtually all these
results are obtained within the computational complexity formalism of _total search problems_ in
NP, which was spearheaded by [JPY88, MP89, Pap94b] to capture the complexity of search
problems that are guaranteed to have a solution. Some key complexity classes in this landscape are shown in Figure 3. We give a non-exhaustive list of intractability results for equilibrium computation: [FPT04] prove that computing pure Nash equilibria in congestion games
is PLS-complete; [DGP09] and later [CDT09] show that computing approximate Nash equilib

7


ria in normal-form games is PPAD-complete; [EY10] study the complexity of computing exact
Nash equilibria (which may use irrational probabilities), introducing the complexity class FIXP;

[VY11, CPY17] consider the complexity of computing Market equilibria; [Das13, Rub15, Rub16]
consider the complexity of computing approximate
Nash equilibria of constant approximation; [KM18]
establish a connection between approximate Nash
equilibrium computation and the SoS hierarchy;

[Meh14, DFS20] study the complexity of computing Nash equilibria in specially structured games.
A result that is particularly useful for our work is
the result of [HPV89] which shows black-box query
lower bounds for computing Brouwer fixed points
of a continuous function. We use this result in

Section 9 as an ingredient for proving our blackbox lower bounds for computing approximate local
min-max solutions.

Beyond equilibrium computation and its applications to Economics and Game Theory, the
study of total search problems has found profound connections to many scientific fields, including continuous optimization [DP11, DTZ18],

Figure 3: The complexity-theoretic land- combinatorial optimization [SY91], query complexNP.
scape of total search problems in ity [BCE [+] 95], topology [GH19], topological com
binatorics and social choice theory [FG18, FG19,
FRHSZ20b, FRHSZ20a], algebraic combinatorics [BIQ [+] 17, GKSZ19], and cryptography [Jeˇr16,
BPR15, SZZ18]. For a more extensive overview of total search problems we refer the reader to
the recent survey by Daskalakis [Das18].


As already discussed, min-max optimization has intimate connections to the foundations
of Game Theory, Mathematical Programming, Online Learning, Statistics, and several other
fields. Recent applications of min-max optimization to Machine Learning, such as Generative
Adversarial Networks and Adversarial Training, have motivated a slew of recent work targeting first-order (or other light-weight online learning) methods for solving min-max optimization problems for convex-concave, nonconvex-concave, as well as nonconvex-nonconcave objectives. Work on convex-concave and nonconvex-concave objectives has focused on obtaining
online learning methods with improved rates [KM19, LJJ19, TJNO19, NSH [+] 19, LTHC19, OX19,
Zha19, ADSG19, AMLJG20, GPDO20, LJJ20] and last-iterate convergence guarantees [DISZ18,
DP18, MR18, MPP18, RLLY18, HA18, ADLH19, DP19, LS19, GHP [+] 19, MOP19, ALW19], while
work on nonconvex-nonconcave problems has focused on identifying different notions of local
min-max solutions [JNJ19, MV20] and studying the existence and (local) convergence properties
of learning methods at these points [WZB19, MV20, MSV20].


8


#### **2 Preliminaries**

**Notation.** For any compact and convex _K_ _⊆_ **R** _[d]_ and _B_ _∈_ **R** +, we define _L_ ∞ ( _K_, _B_ ) to be the set of
all continuous functions _f_ : _K_ _→_ **R** such that max _**x**_ _∈_ _K_ _|_ _f_ ( _**x**_ ) _| ≤_ _B_ . When _K_ = [ 0, 1 ] _[d]_, we use _L_ ∞ ( _B_ )
instead of _L_ ∞ ([ 0, 1 ] _[d]_, _B_ ) for ease of notation. For _p_ _>_ 0, we define diam _p_ ( _K_ ) = max _**x**_, _**y**_ _∈_ _K_ _∥_ _**x**_ _−_ _**y**_ _∥_ _p_,
where _∥·∥_ _p_ is the usual _ℓ_ _p_ -norm of vectors. For an alphabet set Σ, the set Σ _[∗]_, called the Kleene
star of Σ, is equal to _∪_ _i_ [∞] = 0 [Σ] _[i]_ [. For any string] _**[ q]**_ _[ ∈]_ [Σ][ we use] _[ |]_ _**[q]**_ _[|]_ [ to denote the length of] _**[ q]**_ [. We use the]
symbol log ( _·_ ) for base 2 logarithms and ln ( _·_ ) for the natural logarithm. We use [ _n_ ] ≜ _{_ 1, . . ., _n_ _}_,

[ _n_ ] _−_ 1 ≜ _{_ 0, . . ., _n_ _−_ 1 _}_, and [ _n_ ] 0 ≜ _{_ 0, . . ., _n_ _}_ .


**Lipschitzness, Smoothness, and Normalization.** Our main objects of study are continuously
differentiable Lipschitz and smooth functions _f_ : _P →_ **R**, where _P ⊆_ [ 0, 1 ] _[d]_ is some polytope. A
continuously differentiable function _f_ is called _G-Lipschitz_ if _|_ _f_ ( _**x**_ ) _−_ _f_ ( _**y**_ ) _| ≤_ _G_ _∥_ _**x**_ _−_ _**y**_ _∥_ 2, for all
_**x**_, _**y**_, and _L-smooth_ if _∥∇_ _f_ ( _**x**_ ) _−∇_ _f_ ( _**y**_ ) _∥_ 2 _≤_ _L_ _∥_ _**x**_ _−_ _**y**_ _∥_ 2, for all _**x**_, _**y**_ .



_Remark_ 2.1 (Function Normalization) _. Note that the G-Lipschitzness of a function f_ : _P →_ **R** _, where_
_P ⊆_ [ 0, 1 ] _[d]_ _implies that for any_ _**x**_ _and_ _**y**_ _it holds that_ _|_ _f_ ( _**x**_ ) _−_ _f_ ( _**y**_ ) _| ≤_ _G_ _√_ _d. Whenever the range of a_



_P ⊆_ [ 0, 1 ] _implies that for any_ _**x**_ _and_ _**y**_ _it holds that_ _|_ _f_ ( _**x**_ ) _−_ _f_ ( _**y**_ ) _| ≤_ _G_ _√_ _d. Whenever the range of a_

_G-Lipschitz function is taken to be_ [ _−_ _B_, _B_ ] _, for some B, we always assume that B_ _≤_ _G_ _√_ _d. This can be_



_G-Lipschitz function is taken to be_ [ _−_ _B_, _B_ ] _, for some B, we always assume that B_ _≤_ _G_ _√_ _d. This can be_

_accomplished by setting_ _f_ [˜] ( _**x**_ ) = _f_ ( _**x**_ ) _−_ _f_ ( _**x**_ 0 ) _for some fixed_ _**x**_ 0 _in the domain of f. For all the problems_
_that we consider in this paper any solution for_ _f is also a solution for f and vice-versa._ [˜]



**Function Access.** We study optimization problems involving real-valued functions, considering
two access models to such functions.


 - **Black Box Model.** In this model we are given access to an oracle _O_ _f_ such that given a point
_**x**_ _∈_ [ 0, 1 ] _[d]_ the oracle _O_ _f_ returns the values _f_ ( _**x**_ ) and _∇_ _f_ ( _**x**_ ) . In this model we assume that
we can perform real number arithmetic operations. This is the traditional model used to
prove lower bounds in Optimization and Machine Learning [NY83].


 - **White Box Model.** In this model we are given the description of a polynomial-time Turing
machine _C_ _f_ that computes _f_ ( _**x**_ ) and _∇_ _f_ ( _**x**_ ) . More precisely, given some input _**x**_ _∈_ [ 0, 1 ] _[d]_,
described using _B_ bits, and some accuracy _ε_, _C_ _f_ runs in time upper bounded by some
polynomial in _B_ and log ( 1/ _ε_ ) and outputs approximate values for _f_ ( _**x**_ ) and _∇_ _f_ ( _**x**_ ), with
approximation error that is at most _ε_ in _ℓ_ 2 distance. We note that a running time upper
bound on a given Turing Machine can be enforced syntactically by stopping the computation and outputting a fixed output whenever the computation exceeds the bound. See
also Remark 2.6 for an important remark about how to formally study the computational
complexity of problems that take as input a polynomial-time Turing Machine.


**Promise Problems.** To simplify the exposition of our paper, make the definitions of our computational problems and theorem statements clearer, and make our intractability results stronger,
we choose to enforce the following constraints on our function access, _O_ _f_ or _C_ _f_, as a _promise_,
rather than enforcing these constraints in some syntactic manner.


1. **Consistency of Function Values and Gradient Values.** Given some oracle _O_ _f_ or Turing
machine _C_ _f_, it is difficult to determine by querying the oracle or examining the description
of the Turing machine whether the function and gradient values output on different inputs
are consistent with some differentiable function. In all our computational problems, we


9


will only consider instances where this is promised to be the case. Moreover, for all our
computational hardness results, the instances of the problems arising from our reductions
satisfy these constraints, which are guaranteed syntactically by our reduction.


2. **Lipschitzness, Smoothness and Boundedness.** Similarly, given some oracle _O_ _f_ or Turing
machine _C_ _f_, it is difficult to determine, by querying the oracle or examining the description
of the Turing machine, whether the function and gradient values output by _O_ _f_ or _C_ _f_ are
consistent with some Lipschitz, smooth and bounded function with some prescribed Lipschitzness, smoothness, and bound on its absolute value. In all our computational problems,
we only consider instances where the _G_ -Lipschitzness, _L_ -smoothness and _B_ -boundedness
of the function are promised to hold for the prescribed, in the input of the problem, parameters _G_, _L_ and _B_ . Moreover, for all our computational hardness results, the instances
of the problems arising from our reductions satisfy this constraint, which is guaranteed
syntactically by our reduction.


In summary, in the rest of this paper, whenever we prove _an upper bound_ for some computational problem, namely an upper bound on the number of steps or queries to the function
oracle required to solve the problem in the black-box model, or the containment of the problem
in some complexity class in the white-box model, we assume that the afore-described properties
are satisfied by the _O_ _f_ or _C_ _f_ provided in the input. On the other hand, whenever we prove a
_lower bound_ for some computational problem, namely a lower bound on the number of steps/queries required to solve it in the black-box model, or its hardness for some complexity class
in the white-box model, the instances arising in our lower bounds are guaranteed to satisfy
the above properties syntactically by our constructions. As such, our hardness results will not
exploit the difficulty in checking whether _O_ _f_ or _C_ _f_ satisfy the above constraints in order to infuse computational complexity into our problems, but will faithfully target the computational
problems pertaining to min-max optimization of smooth and Lipschitz objectives that we aim to
understand in this paper.


**2.1** **Complexity Classes and Reductions**


In this section we define the main complexity classes that we use in this paper, namely NP, FNP
and PPAD, as well as the notion of reduction used to show containment or hardness of a problem
for one of these complexity classes.


**Definition 2.2** (Search Problems, NP, FNP) **.** A binary relation _Q ⊆{_ 0, 1 _}_ _[∗]_ _× {_ 0, 1 _}_ _[∗]_ is in the class
FNP if (i) for every _**x**_, _**y**_ _∈{_ 0, 1 _}_ _[∗]_ such that ( _**x**_, _**y**_ ) _∈Q_, it holds that _|_ _**y**_ _| ≤_ poly ( _|_ _**x**_ _|_ ) ; and (ii) there
exists an algorithm that verifies whether ( _**x**_, _**y**_ ) _∈Q_ in time poly ( _|_ _**x**_ _|_, _|_ _**y**_ _|_ ) . The _search problem_
associated with a binary relation _Q_ takes some _**x**_ as input and requests as output some _**y**_ such
that ( _**x**_, _**y**_ ) _∈Q_ or outputting _⊥_ if no such _**y**_ exists. The _decision problem_ associated with _Q_ takes
some _**x**_ as input and requests as output the bit 1, if there exists some _**y**_ such that ( _**x**_, _**y**_ ) _∈Q_,
and the bit 0, otherwise. The class NP is defined as the set of decision problems associated with
relations _Q ∈_ FNP.


To define the complexity class PPAD we first define the notion of polynomial-time reductions
between search problems [3], and the computational problem End-of-a-Line [4] .


3 In this paper we only define and consider Karp-reductions between search problems.
4 This problem is sometimes called End-of-the-Line, but we adopt the nomenclature proposed by [Rub16] since
we agree that it describes the problem better.


10


**Definition 2.3** (Polynomial-Time Reductions) **.** A search problem _P_ 1 is _polynomial-time reducible_ to
a search problem _P_ 2 if there exist polynomial-time computable functions _f_ : _{_ 0, 1 _}_ _[∗]_ _→{_ 0, 1 _}_ _[∗]_

and _g_ : _{_ 0, 1 _}_ _[∗]_ _× {_ 0, 1 _}_ _[∗]_ _× {_ 0, 1 _}_ _[∗]_ _→{_ 0, 1 _}_ _[∗]_ with the following properties: (i) if _**x**_ is an input to
_P_ 1, then _f_ ( _**x**_ ) is an input to _P_ 2 ; and (ii) if _**y**_ is a solution to _P_ 2 on input _f_ ( _**x**_ ), then _g_ ( _**x**_, _f_ ( _**x**_ ), _**y**_ ) is
a solution to _P_ 1 on input _**x**_ .


**End-of-a-Line.**

Input: Binary circuits _C_ _S_ (for successor) and _C_ _P_ (for predecessor) with _n_ inputs and _n_ outputs.


Output: One of the following:

0. **0** if either both _C_ _P_ ( _C_ _S_ ( **0** )) and _C_ _S_ ( _C_ _P_ ( **0** )) are equal to **0**, or if they are both different than
**0**, where **0** is the all-0 string.
1. a binary string _**x**_ _∈{_ 0, 1 _}_ _[n]_ such that _**x**_ _̸_ = **0** and _C_ _P_ ( _C_ _S_ ( _**x**_ )) _̸_ = _**x**_ or _C_ _S_ ( _C_ _P_ ( _**x**_ )) _̸_ = _**x**_ .


To make sense of the above definition, we envision that the circuits _C_ _S_ and _C_ _P_ implicitly define
a directed graph, with vertex set _{_ 0, 1 _}_ _[n]_, such that the directed edge ( _**x**_, _**y**_ ) _∈{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_

belongs to the graph if and only if _C_ _S_ ( _**x**_ ) = _**y**_ and _C_ _P_ ( _**y**_ ) = _**x**_ . As such, all vertices in the implicitly
defined graph have in-degree and out-degree at most 1. The above problem permits an output
of **0** if **0** has equal in-degree and out-degree in this graph. Otherwise it permits an output _**x**_ _̸_ = **0**
such that _**x**_ has in-degree or out-degree equal to 0. It follows by the parity argument on directed
graphs, namely that in every directed graph the sum of in-degrees equals the sum of out-degrees,
that End-of-a-Line is a _total problem_, i.e. that for any possible binary circuits _C_ _S_ and _C_ _P_ there
exists a solution of the “0.” kind or the “1.” kind in the definition of our problem (or both).
Indeed, if **0** has unequal in- and out-degrees, there must exist another vertex _**x**_ _̸_ = **0** with unequal
in- and out-degrees, thus one of these degrees must be 0 (as all vertices in the graph have in- and
out-degrees bounded by 1).


We are finally ready to define the complexity class PPAD introduced by [Pap94b].


**Definition 2.4** ( **PPAD** ) **.** The complexity class PPAD contains all search problems that are polynomial time reducible to the End-of-a-Line problem.


The complexity class PPAD is of particular importance, since it contains lots of fundamental
problems in Game Theory, Economics, Topology and several other fields [DGP09, Das18]. A
particularly important PPAD-complete problem is finding fixed points of continuous functions,
whose existence is guaranteed by Brouwer’s fixed point theorem.


**Brouwer.**

Input: Scalars _L_ and _γ_ and a polynomial-time Turing machine _C_ _M_ evaluating a _L_ -Lipschitz
function _M_ : [ 0, 1 ] _[d]_ _→_ [ 0, 1 ] _[d]_ .

Output: A point _**z**_ _[⋆]_ _∈_ [ 0, 1 ] _[d]_ such that _∥_ _**z**_ _[⋆]_ _−_ _M_ ( _**z**_ _[⋆]_ ) _∥_ 2 _<_ _γ_ .


While not stated exactly in this form, the following is a straightforward implication of the results
presented in [CDT09].


**Lemma 2.5** ([CDT09]) **.** Brouwer _is_ PPAD _-complete even when d_ = 2 _. Additionally,_ Brouwer _is_
PPAD _-complete even when γ_ = poly ( 1/ _d_ ) _and L_ = poly ( _d_ ) _._


_Remark_ 2.6 (Respresentation of a polynomial-time Turing Machine) _. In the definition of the problem_
Brouwer _we assume that we are given in the input the description of a Turing Machine_ _C_ _M_ _that computes_


11


_the map M. In order for polynomial-time reductions to and from this problem to be meaningful we need to_
_have an upper bound on the running time of this Turing Machine which we want to be polynomial in the_
_input of the Turing Machine. The formal way to ensure this and derive meaningful complexity results is to_
_define a different problem, say k-_ Brouwer _, for every k_ _∈_ **N** _. In the problem k-_ Brouwer _the input Turing_
_Machine_ _C_ _M_ _has running time bounded by n_ _[k]_ _in the size n of its input. In the rest of the paper whenever_
_we say that a polynomial-time Turing Machine is required in the input to a computational problem_ Pr _, we_
_formally mean that we define a hierarchy of problems k-_ Pr _, k_ _∈_ **N** _, such that k-_ Pr _takes as input Turing_
_Machines with running time bounded by n_ _[k]_ _, and we interpret computational complexity results for_ Pr
_in the following way: whenever we prove that_ Pr _belongs to some complexity class, we prove that k-_ Pr
_belongs to the complexity class for all k_ _∈_ **N** _; whenever we prove that_ Pr _is hard for some complexity class,_
_we prove that, for some absolute constant k_ 0 _determined in the hardness proof, k-_ Pr _is hard for that class,_
_for all k_ _≥_ _k_ 0 _. For simplicity of exposition of our problems and results we do not repeat this discussion in_
_the rest of this paper._

#### **3 Computational Problems of Interest**


In this section, we define the computational problems that we study in this paper and discuss
our main results, postponing formal statements to Section 4. We start in Section 3.1 by defining
the mathematical objects of our study, and proceed in Section 3.2 to define our main computational problems, namely: (1) finding approximate stationary points; (2) finding approximate
local minima; and (3) finding approximate local min-max equilibria. In Section 3.3, we present
some bonus problems, which are intimately related, as we will see, to problems (2) and (3). As
discussed in Section 2, for ease of presentation, we define our problems as promise problems.


**3.1** **Mathematical Definitions**


We define the concepts of _stationary points_, _local minima_, and _local min-max equilibria_ of real valued functions, and make some remarks about their existence, as well as their computational
complexity. The formal discussion of the latter is postponed to Sections 3.2 and 4.
Before we proceed with our definitions, recall that the goal of this paper is to study constrained optimization. Our domain will be the hypercube [ 0, 1 ] _[d]_, which we might intersect with
the set _{_ _**x**_ _|_ _**g**_ ( _**x**_ ) _≤_ **0** _}_, for some convex (potentially multivariate) function _**g**_ . Although most
of the definitions and results that we explore in this paper can be extended to arbitrary convex
functions, we will focus on the case where _**g**_ is linear, and the feasible set is thus a polytope.
Focusing on this case avoids additional complications related to the representation of _**g**_ in the
input to the computational problems that we define in the next section, and avoids also issues
related to verifying the convexity of _**g**_ .


**Definition 3.1** (Feasible Set and Refutation of Feasibility) **.** Given _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and _**b**_ _∈_ **R** _[m]_, we
define the set of feasible solutions to be _P_ ( _**A**_, _**b**_ ) = _{_ _**z**_ _∈_ [ 0, 1 ] _[d]_ _|_ _**A**_ _[T]_ _**z**_ _≤_ _**b**_ _}_ . Observe that testing
whether _P_ ( _**A**_, _**b**_ ) is empty can be done in polynomial time in the bit complexity of _**A**_ and _**b**_ .


**Definition 3.2** (Projection Operator) **.** For a nonempty, closed, and convex set _K_ _⊂_ **R** _[d]_, we define
the projection operator Π _K_ : **R** _[d]_ _→_ _K_ as follows Π _K_ _**x**_ = argmin _**y**_ _∈_ _K_ _∥_ _**x**_ _−_ _**y**_ _∥_ 2 . It is well-known
that for any nonempty, closed, and convex set _K_ the argmin _**y**_ _∈_ _K_ _∥_ _**x**_ _−_ _**y**_ _∥_ 2 exists and is unique,
hence Π _K_ is well defined.


Now that we have defined the domain of the real-valued functions that we consider in this
paper we are ready to define a notion of approximate stationary points.


12


**Definition 3.3** ( _ε_ -Stationary Point) **.** Let _f_ : [ 0, 1 ] _[d]_ _→_ **R** be a _G_ -Lipschitz and _L_ -smooth function
and _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_, _**b**_ _∈_ **R** _[m]_ . We call a point _**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) a _ε_ - _stationary point_ of _f_ if _∥∇_ _f_ ( _**x**_ _[⋆]_ ) _∥_ 2 _<_ _ε_ .


It is easy to see that there exist continuously differentiable functions _f_ that do not have any
(approximate) stationary points, e.g. linear functions. As we will see later in this paper, deciding
whether a given function _f_ has a stationary point is NP-hard and, in fact, it is even NP-hard to
decide whether a function has an approximate stationary point of a very gross approximation.
At the same time, verifying whether a given point is (approximately) stationary can be done
efficiently given access to a polynomial-time Turing machine that computes _∇_ _f_, so the problem of
deciding whether an (approximate) stationary point exists lies in NP, as long as we can guarantee
that, if there is such a point, there will also be one with polynomial bit complexity. We postpone
a formal discussion of the computational complexity of finding (approximate) stationary points
or deciding their existence until we have formally defined our corresponding computational
problem and settled the bit complexity of its solutions.


For the definition of local minima and local min-max equilibria we need the notion of closed
_d_ -dimensional Euclidean balls.


**Definition 3.4** (Euclidean Ball) **.** For _r_ _∈_ **R** + we define the _closed Euclidean ball of radius r_ to be
the set B _d_ ( _r_ ) = � _**x**_ _∈_ **R** _[d]_ _| ∥_ _**x**_ _∥_ 2 _≤_ _r_ �. We also define the _closed Euclidean ball of radius r centered at_
_**z**_ _∈_ **R** _[d]_ to be the set B _d_ ( _r_ ; _**z**_ ) = � _**x**_ _∈_ **R** _[d]_ _| ∥_ _**x**_ _−_ _**z**_ _∥_ 2 _≤_ _r_ �.


**Definition 3.5** ( ( _ε_, _δ_ ) -Local Minimum) **.** Let _f_ : [ 0, 1 ] _[d]_ _→_ **R** be a _G_ -Lipschitz and _L_ -smooth function, _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_, _**b**_ _∈_ **R** _[m]_, and _ε_, _δ_ _>_ 0. A point _**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) is an ( _ε_, _δ_ ) - _local minimum_ of _f_ constrained on _P_ ( _**A**_, _**b**_ ) if and only if _f_ ( _**x**_ _[⋆]_ ) _<_ _f_ ( _**x**_ ) + _ε_ for every _**x**_ _∈P_ ( _**A**_, _**b**_ ) such that _**x**_ _∈_ B _d_ ( _δ_ ; _**x**_ _[⋆]_ ) .


To be clear, using the term “local minimum” in Definition 3.5 is a bit of a misnomer, since for
large enough values of _δ_ the definition captures global minima as well. As _δ_ ranges from large
to small, our notion of ( _ε_, _δ_ ) -local minimum transitions from being an _ε_ -globally optimal point
to being an _ε_ -locally optimal point. Importantly, unlike (approximate) stationary points, a ( _ε_, _δ_ ) local minimum is guaranteed to exist for all _ε_, _δ_ _>_ 0 due to the compactness of [ 0, 1 ] _[d]_ _∩P_ ( _**A**_, _**b**_ )
and the continuity of _f_ . Thus the problem of finding an ( _ε_, _δ_ ) -local minimum is _total_ for arbitrary
values of _ε_ and _δ_ . On the negative side, for arbitrary values of _ε_ and _δ_, there is no polynomial-size
and polynomial-time verifiable witness for certifying that a point _**x**_ _[⋆]_ is an ( _ε_, _δ_ ) -local minimum.
Thus the problem of finding an ( _ε_, _δ_ ) -local minimum is not known to lie in FNP. As we will
see in Section 4, this issue can be circumvented if we focus on particular settings of _ε_ and _δ_, in
relationship to the Lipschitzness and smoothness of _f_ and the dimension _d_ .

Finally we define ( _ε_, _δ_ ) -local min-max equilibrium as follows, recasting Definition 1.1 to the
constraint set _P_ ( _**A**_, _**b**_ ) .


**Definition 3.6** ( ( _ε_, _δ_ ) -Local Min-Max Equilibrium) **.** Let _f_ : [ 0, 1 ] _[d]_ [1] _×_ [ 0, 1 ] _[d]_ [2] _→_ **R** be a _G_ -Lipschitz
and _L_ -smooth function, _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and _**b**_ _∈_ **R** _[m]_, where _d_ = _d_ 1 + _d_ 2, and _ε_, _δ_ _>_ 0. A point
( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) is an ( _ε_, _δ_ ) - _local min-max equilibrium_ of _f_ if and only if the following hold:


 - _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _<_ _f_ ( _**x**_, _**y**_ _[⋆]_ ) + _ε_ for every _**x**_ _∈_ B _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ) with ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) ; and


 - _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _>_ _f_ ( _**x**_ _[⋆]_, _**y**_ ) _−_ _ε_ for every _**y**_ _∈_ B _d_ 2 ( _δ_ ; _**y**_ _[⋆]_ ) with ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) .


13


Similarly to Definition 3.5, for large enough values of _δ_, Definition 3.6 captures global min-max
equilibria as well. As _δ_ ranges from large to small, our notion of ( _ε_, _δ_ ) -local min-max equilibrium transitions from being an _ε_ -approximate min-max equilibrium to being an _ε_ -approximate
local min-max equilibrium. Moreover, in comparison to local minima and stationary points, the
problem of finding an ( _ε_, _δ_ ) -local min-max equilibrium is neither total nor can its solutions be
verified efficiently for all values of _ε_ and _δ_, even when _P_ ( _**A**_, _**b**_ ) = [ 0, 1 ] _[d]_ . Again, this issue can be
circumvented if we focus on particular settings of _ε_ and _δ_ values, as we will see in Section 4.


**3.2** **First-Order Local Optimization Computational Problems**


In this section, we define the search problems associated with our aforementioned definitions
of approximate stationary points, local minima, and local min-max equilibria. We state our
problems in terms of white-box access to the function _f_ and its gradient. Switching to the blackbox variants of our computational problems amounts to simply replacing the Turing machines
provided in the input of the problems with oracle access to the function and its gradient, as
discussed in Section 2. As per our discussion in the same section, we define our computational
problems as _promise problems_, the promise being that the Turing machine (or oracle) provided in
the input to our problems outputs function values and gradient values that are consistent with
a smooth and Lipschitz function with the prescribed in the input smoothness and Lipschitzness.
Besides making the presentation cleaner, as we discussed in Section 2, the motivation for doing
so is to prevent the possibility that computational complexity is tacked into our problems due
to the possibility that the Turing machines/oracles provided in the input do not output function
and gradient values that are consistent with a Lipschitz and smooth function. Importantly, all
our computational hardness results syntactically guarantee that the Turing machines/oracles
provided as input to our constructed hard instances satisfy these constraints.
Before stating our main computational problems below, we note that, for each problem, the
dimension _d_ (in unary representation) is also an implicit input, as the description of the Turing
machine _C_ _f_ (or the interface to the oracle _O_ _f_ in the black-box counterpart of each problem below) has size at least linear in _d_ . We also refer to Remark 2.6 for how we may formally study
complexity problems that take a polynomial-time Turing Machine in their input.


**StationaryPoint.**
Input: Scalars _ε_, _G_, _L_, _B_ _>_ 0 and a polynomial-time Turing machine _C_ _f_ evaluating a _G_ -Lipschitz
and _L_ -smooth function _f_ : [ 0, 1 ] _[d]_ _→_ [ _−_ _B_, _B_ ] and its gradient _∇_ _f_ : [ 0, 1 ] _[d]_ _→_ **R** _[d]_ ; a matrix
_**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and vector _**b**_ _∈_ **R** _[m]_ such that _P_ ( _**A**_, _**b**_ ) _̸_ = ∅.


Output: If there exists some point _**x**_ _∈P_ ( _**A**_, _**b**_ ) such that _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _<_ _ε_ /2, output some point
_**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) such that _∥∇_ _f_ ( _**x**_ _[⋆]_ ) _∥_ 2 _<_ _ε_ ; if, for all _**x**_ _∈P_ ( _**A**_, _**b**_ ), _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _>_ _ε_, output _⊥_ ;
otherwise, it is allowed to either output _**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) such that _∥∇_ _f_ ( _**x**_ _[⋆]_ ) _∥_ 2 _<_ _ε_ or to output _⊥_ .


It is easy to see that StationaryPoint lies in FNP. Indeed, if there exists some point _**x**_ _∈P_ ( _**A**_, _**b**_ )
such that _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _<_ _ε_ /2, then by the _L_ -smoothness of _f_ there must exist some point _**x**_ _[⋆]_ _∈_
_P_ ( _**A**_, _**b**_ ) of bit complexity polynomial in the size of the input such that _∥∇_ _f_ ( _**x**_ _[⋆]_ ) _∥_ 2 _<_ _ε_ . On the
other hand, it is clear that no such point exists if for all _**x**_ _∈P_ ( _**A**_, _**b**_ ), _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _>_ _ε_ . We note that
the looseness of the output requirement in our problem for functions _f_ that do not have points
_x_ _∈P_ ( _**A**_, _**b**_ ) such that _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _<_ _ε_ /2 but do have points _x_ _∈P_ ( _**A**_, _**b**_ ) such that _∥∇_ _f_ ( _**x**_ ) _∥_ 2 _≤_ _ε_ is
introduced for the sole purpose of making the problem lie in FNP, as otherwise we would not be
able to guarantee that the solutions to our search problem have polynomial bit complexity. As we


14


show in Section 4, StationaryPoint is also FNP-hard, even when _ε_ is a constant, the constraint
set is very simple, namely _P_ ( _**A**_, _**b**_ ) = [ 0, 1 ] _[d]_, and _G_, _L_ are both polynomial in _d_ .


Next, we define the computational problems associated with local minimum and local minmax equilibrium. Recall that the first is guaranteed to have a solution, because, in particular, a
global minimum exists due to the continuity of _f_ and the compactness of _P_ ( _**A**_, _**b**_ ) .


**LocalMin.**
Input: Scalars _ε_, _δ_, _G_, _L_, _B_ _>_ 0 and a polynomial-time Turing machine _C_ _f_ evaluating a _G_ Lipschitz and _L_ -smooth function _f_ : [ 0, 1 ] _[d]_ _→_ [ _−_ _B_, _B_ ] and its gradient _∇_ _f_ : [ 0, 1 ] _[d]_ _→_ **R** _[d]_ ; a
matrix _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and vector _**b**_ _∈_ **R** _[m]_ such that _P_ ( _**A**_, _**b**_ ) _̸_ = ∅.


Output: A point _**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) such that _f_ ( _**x**_ _[⋆]_ ) _<_ _f_ ( _**x**_ ) + _ε_ for all _**x**_ _∈_ B _d_ ( _δ_ ; _**x**_ _[⋆]_ ) _∩P_ ( _**A**_, _**b**_ ) .


**LocalMinMax.**
Input: Scalars _ε_, _δ_, _G_, _L_, _B_ _>_ 0; a polynomial-time Turing machine _C_ _f_ evaluating a _G_ -Lipschitz
and _L_ -smooth function _f_ : [ 0, 1 ] _[d]_ [1] _×_ [ 0, 1 ] _[d]_ [2] _→_ [ _−_ _B_, _B_ ] and its gradient _∇_ _f_ : [ 0, 1 ] _[d]_ [1] _×_ [ 0, 1 ] _[d]_ [2] _→_
**R** _[d]_ [1] [+] _[d]_ [2] ; a matrix _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and vector _**b**_ _∈_ **R** _[m]_ such that _P_ ( _**A**_, _**b**_ ) _̸_ = ∅, where _d_ = _d_ 1 + _d_ 2 .


Output: A point ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) such that

_▷_ _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _<_ _f_ ( _**x**_, _**y**_ _[⋆]_ ) + _ε_ for all _**x**_ _∈_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ) with ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) and

_▷_ _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _>_ _f_ ( _**x**_ _[⋆]_, _**y**_ ) _−_ _ε_ for all _**y**_ _∈_ _B_ _d_ 2 ( _δ_ ; _**y**_ _[⋆]_ ) with ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ),
or _⊥_ if no such point exists.


Unlike StationaryPoint the problems LocalMin and LocalMinMax exhibit vastly different
behavior, depending on the values of the inputs _ε_ and _δ_ in relationship to _G_, _L_ and _d_, as we
will see in Section 4 where we summarize our computational complexity results. This range of
behaviors is rooted at our earlier remark that, depending on the value of _δ_ provided in the input
to these problems, they capture the complexity of finding _global_ minima/min-max equilibria, for
large values of _δ_, as well as finding _local_ minima/min-max equilibria, for small values of _δ_ .


**3.3** **Bonus Problems: Fixed Points of Gradient Descent/Gradient Descent-Ascent**


Next we present a couple of bonus problems, GDFixedPoint and GDAFixedPoint, which respectively capture the computation of fixed points of the (projected) gradient descent and the
(projected) gradient descent-ascent dynamics, with learning rate = 1. As we see in Section 5,
these problems are intimately related, indeed equivalent under polynomial-time reductions, to
problems LocalMin and LocalMinMax respectively, in certain regimes of the approximation
parameters. Before stating problems GDFixedPoint and GDAFixedPoint, we define the mappings _F_ _GD_ and _F_ _GDA_ whose fixed points these problems are targeting.


**Definition 3.7** (Projected Gradient Descent) **.** For a closed and convex _K_ _⊆_ **R** _[d]_ and some continuously differentiable function _f_ : _K_ _→_ **R**, we define the _Projected Gradient Descent Dynamics with_
_learning rate_ 1 as the map _F_ _GD_ : _K_ _→_ _K_, where _F_ _GD_ ( _**x**_ ) = Π _K_ ( _**x**_ _−∇_ _f_ ( _**x**_ )) .


**Definition 3.8** (Projected Gradient Descent/Ascent) **.** For a closed and convex _K_ _⊆_ **R** _[d]_ [1] _×_ **R** _[d]_ [2] and
some continuously differentiable function _f_ : _K_ _→_ **R**, we define the _Unsafe Projected Gradient_
_Descent/Ascent Dynamic with learning rate_ 1 as the map _F_ _GDA_ : _K_ _→_ **R** _[d]_ [1] _×_ **R** _[d]_ [2] defined as follows



�



�



_F_ _GDA_ ( _**x**_, _**y**_ ) ≜



Π _K_ ( _**y**_ ) ( _**x**_ _−∇_ _**x**_ _f_ ( _**x**_, _**y**_ ))
�Π _K_ ( _**x**_ ) ( _**y**_ + _∇_ _**y**_ _f_ ( _**x**_, _**y**_ ))



≜ _F_ _GDAx_ ( _**x**_, _**y**_ )
� _F_ _GDAy_ ( _**x**_, _**y**_ )



for all ( _**x**_, _**y**_ ) _∈_ _K_, where _K_ ( _**y**_ ) = _{_ _**x**_ _[′]_ _|_ ( _**x**_ _[′]_, _**y**_ ) _∈_ _K_ _}_ and _K_ ( _**x**_ ) = _{_ _**y**_ _[′]_ _|_ ( _**x**_, _**y**_ _[′]_ ) _∈_ _K_ _}_ .


15


Note that _F_ _GDA_ is called “unsafe” because the projection happens individually for _**x**_ _−∇_ _**x**_ _f_ ( _**x**_, _**y**_ )
and _**y**_ + _∇_ _**y**_ _f_ ( _**x**_, _**y**_ ), thus _F_ _GDA_ ( _**x**_, _**y**_ ) may not lie in _K_ . We also define the “safe” version _F_ _sGDA_,
which projects the pair ( _**x**_ _−∇_ _**x**_ _f_ ( _**x**_, _**y**_ ), _**y**_ + _∇_ _**y**_ _f_ ( _**x**_, _**y**_ )) jointly onto _K_ . As we show in Section 5
(in particular inside the proof of Theorem 5.2), computing fixed points of _F_ _GDA_ and _F_ _sGDA_ are
computationally equivalent so we stick to _F_ _GDA_ which makes the presentation slightly cleaner.
We are now ready to define GDFixedPoint and GDAFixedPoint. As per earlier discussions,
we define these computational problems as _promise problems_, the promise being that the Turing
machine provided in the input to these problems outputs function values and gradient values
that are consistent with a smooth and Lipschitz function with the prescribed, in the input to these
problems, smoothness and Lipschitzness.


**GDFixedPoint.**
Input: Scalars _α_, _G_, _L_, _B_ _>_ 0 and a polynomial-time Turing machine _C_ _f_ evaluating a _G_ -Lipschitz
and _L_ -smooth function _f_ : [ 0, 1 ] _[d]_ _→_ [ _−_ _B_, _B_ ] and its gradient _∇_ _f_ : [ 0, 1 ] _[d]_ _→_ **R** _[d]_ ; a matrix
_**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and vector _**b**_ _∈_ **R** _[m]_ such that _P_ ( _**A**_, _**b**_ ) _̸_ = ∅.


Output: A point _**x**_ _[⋆]_ _∈P_ ( _**A**_, _**b**_ ) such that _∥_ _**x**_ _[⋆]_ _−_ _F_ _GD_ ( _**x**_ _[⋆]_ ) _∥_ 2 _<_ _α_, where _K_ = _P_ ( _**A**_, _**b**_ ) is the
projection set used in the definition of _F_ _GD_ .


**GDAFixedPoint.**
Input: Scalars _α_, _G_, _L_, _B_ _>_ 0 and a polynomial-time Turing machine _C_ _f_ evaluating a _G_ -Lipschitz
and _L_ -smooth function _f_ : [ 0, 1 ] _[d]_ [1] _×_ [ 0, 1 ] _[d]_ [2] _→_ [ _−_ _B_, _B_ ] and its gradient _∇_ _f_ : [ 0, 1 ] _[d]_ [1] _×_ [ 0, 1 ] _[d]_ [2] _→_
**R** _[d]_ [1] [+] _[d]_ [2] ; a matrix _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ and vector _**b**_ _∈_ **R** _[m]_ such that _P_ ( _**A**_, _**b**_ ) _̸_ = ∅, where _d_ = _d_ 1 + _d_ 2 .


Output: A point ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) such that _∥_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _−_ _F_ _GDA_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∥_ 2 _<_ _α_, where _K_ =
_P_ ( _**A**_, _**b**_ ) is the projection set used in the definition of _F_ _GDA_ .


In Section 5 we show that the problems GDFixedPoint and LocalMin are equivalent under
polynomial-time reductions, and the problems GDAFixedPoint and LocalMinMax are equivalent under polynomial-time reductions, in certain regimes of the approximation parameters.

#### **4 Summary of Results**


In this section we summarize our results for the optimization problems that we defined in the
previous section. We start with our theorem about the complexity of finding approximate stationary points, which we show to be FNP-complete even for large values of the approximation.


**Theorem 4.1** (Complexity of Finding Approximate Stationary Points) **.** _The computational problem_
StationaryPoint _is_ FNP _-complete, even when ε is set to any value_ _≤_ 1/24 _, and even when_ _P_ ( _**A**_, _**b**_ ) =

[ 0, 1 ] _[d]_ _, G_ = _√_ _d, L_ = _d, and B_ = 1 _._


It is folklore and easy to verify that approximate stationary points always exist and can be
found in time poly ( _B_, 1/ _ε_, _L_ ) when the domain of _f_ is unconstrained, i.e. it is the whole **R** _[d]_, and
the range of _f_ is bounded, i.e., when _f_ ( **R** _[d]_ ) _⊆_ [ _−_ _B_, _B_ ] . Theorem 4.1 implies that such a guarantee should not be expected in the bounded domain case, where the existence of approximate
stationary points is not guaranteed and must also be verified. In particular, it follows from our
theorem that any algorithm that verifies the existence of and computes approximate stationary
points in the constrained case should take time that is super-polynomial in at least one of _G_, _L_,
or _d_, unless P = NP. The proof of Theorem 4.1 is based on an elegant construction for converting
(real valued) stationary points of an appropriately constructed function to (binary) solutions of a


16


target Sat instance. This conversion involves the use of Lovász Local Lemma [EL73]. The details
A.
of the proof can be found in Appendix


The complexity of LocalMin and LocalMinMax is more difficult to characterize, as the
nature of these problems changes drastically depending on the relationship of _δ_ with with _ε_, _G_,
_L_ and _d_, which determines whether these problems ask for a _globally_ vs _locally_ approximately
optimal solution. In particular, there are two regimes wherein the complexity of both problems
is simple to characterize.


_▷_ **Global Regime.** When _δ_ _≥_ _√_ _d_ then both LocalMin and LocalMinMax ask for a _globally_

optimal solution. In this regime it is not difficult to see that both problems are FNP-hard to
solve even when _ε_ = Θ ( 1 ) and _G_, _L_ are _O_ ( _d_ ) (see Section 10).


_▷_ **Trivial Regime.** When _δ_ satisfies _δ_ _<_ _ε_ / _G_, then for every point _**z**_ _∈P_ ( _**A**_, _**b**_ ) it holds that
_|_ _f_ ( _**z**_ ) _−_ _f_ ( _**z**_ _[′]_ ) _| <_ _ε_ for every _**z**_ _[′]_ _∈_ B _d_ ( _δ_ ; _**z**_ ) with _**z**_ _[′]_ _∈P_ ( _**A**_, _**b**_ ) . Thus, every point _**z**_ in the
domain _P_ ( _**A**_, _**b**_ ) is a solution to both LocalMin and LocalMinMax.


It is clear from our discussion above, and in earlier sections, that, to really capture the complexity
of finding local as opposed to global minima/min-max equilibria, we should restrict the value
of _δ_ . We identify the following regime, which we call the “ _local regime_ .” As we argue shortly,
this regime is markedly different from the global regime identified above in that (i) a solution is
guaranteed to exist for both our problems of interest, where in the global regime only LocalMin
is guaranteed to have a solution; and (ii) their computational complexity transitions to lower
complexity classes.




_▷_ **Local Regime.** Our main focus in this paper is the regime defined by _δ_ _<_ _[√]_



**Local Regime.** Our main focus in this paper is the regime defined by _δ_ _<_ _[√]_ 2 _ε_ / _L_ . In

this regime it is well known that Projected Gradient Descent can solve LocalMin in
time _O_ ( _B_ _·_ _L_ / _ε_ ) (see Appendix E). Our main interest is understanding the complexity of
LocalMinMax, which is not well understood in this regime. We note that the use of the
constant 2 in the constraint _δ_ _<_ _[√]_ 2 _ε_ / _L_ which defines the local regime has a natural mo
tivation: consider a point _**z**_ where a _L_ -smooth function _f_ has _∇_ _f_ ( _**z**_ ) = 0; it follows from
the definition of smoothness that _**z**_ is both an ( _ε_, _δ_ ) -local min and an ( _ε_, _δ_ ) -local min-max
equilibrium, as long as _δ_ _<_ _[√]_ 2 _ε_ / _L_ .



2 _ε_ / _L_ .



The following theorems provide tight upper and lower bounds on the computational complexity
of solving LocalMinMax in the local regime. For compactness, we define the following problem:


**Definition 4.2** (Local Regime LocalMinMax) **.** We define the _local-regime local min-max equilib-_
_rium computation problem_, in short LR-LocalMinMax, to be the search problem LocalMinMax
restricted to instances in the local regime, i.e. satisfying _δ_ _<_ _[√]_ 2 _ε_ / _L_ .


**Theorem 4.3** (Existence of Approximate Local Min-Max Equilibrium) **.** _The computational problem_
LR-LocalMinMax _belongs to_ PPAD _. As a byproduct, if some function f is G-Lipschitz and L-smooth,_
_then an_ ( _ε_, _δ_ ) _-local min-max equilibrium is guaranteed to exist when δ_ _<_ _[√]_ 2 _ε_ / _L, i.e. in the local regime._


**Theorem 4.4** (Hardness of Finding Approximate Local Min-Max Equilibrium) **.** _The search problem_
LR-LocalMinMax _is_ PPAD _-hard, for any δ_ _≥_ _[√]_ _ε_ / _L, and even when it holds that_ 1/ _ε_ = poly ( _d_ ) _,_

_G_ = poly ( _d_ ) _, L_ = poly ( _d_ ) _, and B_ = _d._


17


Theorem 4.4 implies that any algorithm that computes an ( _ε_, _δ_ ) -local min-max equilibrium of a _G_ Lipschitz and _L_ -smooth function _f_ in the local regime should take time that is super-polynomial
in at least one of 1/ _ε_, _G_, _L_ or _d_, unless FP = PPAD. As such, the complexity of computing local
min-max equilibria in the local regime is markedly different from the complexity of computing
local minima, which can be found using Projected Gradient Descent in poly ( _G_, _L_, 1/ _ε_, _d_ ) time
and function/gradient evaluations (see Appendix E).


An important property of our reduction in the proof of Theorem 4.4 is that it is a _black-box_
_reduction_ . We can hence prove the following unconditional lower bound in the black-box model.


**Theorem 4.5** (Black-Box Lower Bound for Finding Approximate Local Min-Max Equilibrium) **.**
_Suppose_ _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ _and_ _**b**_ _∈_ **R** _[m]_ _are given together with an oracle_ _O_ _f_ _that outputs a G-Lipschtz and_
_L-smooth function f_ : _P_ ( _**A**_, _**b**_ ) _→_ [ _−_ 1, 1 ] _and its gradient_ _∇_ _f. Let also δ_ _≥_ _[√]_ _L_ / _ε, ε_ _≤_ _G_ [2] / _L, and let_

_all the parameters_ 1/ _ε,_ 1/ _δ, L, G be upper bounded by_ poly ( _d_ ) _. Then any algorithm that has access to_
_f only through_ _O_ _f_ _and computes an_ ( _ε_, _δ_ ) _-local min-max equilibrium has to make a number of queries to_
_O_ _f_ _that is exponential in at least one of the parameters:_ 1/ _ε, G, L or d even when_ _P_ ( _**A**_, _**b**_ ) _⊆_ [ 0, 1 ] _[d]_ _._


Our main goal in the rest of the paper is to provide the proofs of Theorems 4.3, 4.4 and 4.5.
In Section 5, we show how to use Brouwer’s fixed point theorem to prove the existence of approximate local min-max equilibrium in the local regime. Moreover, we establish an equivalence
between LocalMinMax and GDAFixedPoint, in the local regime, and show that both belong
to PPAD. In Sections 6 and 7, we provide a detailed proof of our main result, i.e. Theorem 4.4.
Finally, in Section 9, we show how our proof from Section 7 produces as a byproduct the blackbox, unconditional lower bound of Theorem 4.5. In Section 8, we outline a useful interpolation
technique which allows as to interpolate a function given its values and the values of its gradient
on a hypergrid, so as to enforce the Lipschitzness and smoothness of the interpolating function.
We make heavy use of this technically involved result in all our hardness proofs.

#### **5 Existence of Approximate Local Min-Max Equilibrium**


In this section, we establish the totality of LR-LocalMinMax, i.e. LocalMinMax for instances
satisfying _δ_ _<_ _[√]_ 2 _ε_ / _L_ as defined in Definition 4.2. In particular, we prove that every _G_ -Lipschitz



satisfying _δ_ _<_ _[√]_ 2 _ε_ / _L_ as defined in Definition 4.2. In particular, we prove that every _G_ -Lipschitz

and _L_ -smooth function admits an ( _ε_, _δ_ ) -local min-max equilibrium, as long as _δ_ _<_ _[√]_ 2 _ε_ / _L_ . A



and _L_ -smooth function admits an ( _ε_, _δ_ ) -local min-max equilibrium, as long as _δ_ _<_ _[√]_ 2 _ε_ / _L_ . A

byproduct of our proof is in fact that LR-LocalMinMax lies inside PPAD. Specifically the main
tool that we use to prove our result is a computational equivalence between the problem of finding fixed points of the Gradient Descent/Ascent dynamic, i.e. GDAFixedPoint, and the problem
LR-LocalMinMax. A similar equivalence between GDFixedPoint and LocalMin also holds,
but the details of that are left to the reader as a simple exercise. Next, we first present the equivalence between GDAFixedPoint and LR-LocalMinMax, and we then show that GDAFixedPoint
is in PPAD, which then also establishes that LR-LocalMinMax is in PPAD.



**Theorem 5.1.** _The search problems_ LR-LocalMinMax _and_ GDAFixedPoint _are equivalent under_
_polynomial-time reductions. That is, there is a polynomial-time reduction from_ LR-LocalMinMax _to_
GDAFixedPoint _and vice versa. In particular, given some_ _**A**_ _∈_ **R** _[d]_ _[×]_ _[m]_ _and_ _**b**_ _∈_ **R** _[m]_ _such that_ _P_ ( _**A**_, _**b**_ ) _̸_ =
∅ _, along with a G-Lipschitz and L-smooth function f_ : _P_ ( _**A**_, _**b**_ ) _→_ **R** _:_



_1. For arbitrary ε_ _>_ 0 _and_ 0 _<_ _δ_ _<_ _[√]_



2 _ε_ / _L, suppose that_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _∈P_ ( _**A**_, _**b**_ ) _is an α-approximate_



_√_
_fixed point of F_ _GDA_ _, i.e.,_ _∥_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _−_ _F_ _GDA_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _∥_ 2 _<_ _α, where α_ _≤_



( _G_ + _δ_ ) [2] + 4 ( _ε_ _−_ _[L]_ 2



_√_ ( _G_ + _δ_ ) [2] + 4 ( _ε_ _−_ 2 _[δ]_ [2] [)] _[−]_ [(] _[G]_ [+] _[δ]_ [)]
_fixed point of F_ _GDA_ _, i.e.,_ _∥_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _−_ _F_ _GDA_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _∥_ 2 _<_ _α, where α_ _≤_ 2 _._

_Then_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _is also a_ ( _ε_, _δ_ ) _-local min-max equilibrium of f._



18


_α_ [2] _·_ _L_
_2. For arbitary α_ _>_ 0 _, suppose that_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _is an_ ( _ε_, _δ_ ) _-local min-max equilibrium of f for ε_ = ( 5 _L_ + 2 ) [2]
_and δ_ = _[√]_ _ε_ / _L. Then_ ( _**x**_ _[∗]_, _**y**_ _[∗]_ ) _is also an α-approximate fixed point of F_ _GDA_ _._


The proof of Theorem 5.1 is presented in Appendix B.1. As already discussed, we use GDAFixedPoint as an intermediate step to establish the totality of LR-LocalMinMax and to show its
inclusion in PPAD. This leads to the following theorem.


**Theorem 5.2.** _The computational problems_ GDAFixedPoint _and_ LR-LocalMinMax _are both total_
_search problems and they both lie in_ PPAD _._


Observe that Theorem 4.3 is implied by Theorem 5.2 whose proof is presented in Appendix B.2.

#### **6 Hardness of Local Min-Max Equilibrium – Four-Dimensions**


In Section 5, we established that LR-LocalMinMax belongs to PPAD. Our proof is via the
intermediate problem GDAFixedPoint which we showed that it is computationally equivalent
to LR-LocalMinMax. Our next step is to prove the PPAD-hardness of LR-LocalMinMax using
again GDAFixedPoint as an intermediate problem.


In this section we prove that GDAFixedPoint is PPAD-hard in four dimensions. To establish
this hardness result we introduce a variant of the classical 2D-Sperner problem which we call
2D-BiSperner which we show is PPAD-hard. The main technical part of our proof is to show
that 2D-BiSperner with input size _n_ reduces to GDAFixedPoint, with input size poly ( _n_ ), _α_ =
exp ( _−_ poly ( _n_ )), _G_ = _L_ = exp ( poly ( _n_ )), and _B_ = 2. This reduction proves the hardness of
GDAFixedPoint. Formally, our main result of this section is the following theorem.


**Theorem 6.1.** _The problem_ GDAFixedPoint _is_ PPAD _-complete even in dimension d_ = 4 _and B_ = 2 _._
_Therefore,_ LR-LocalMinMax _is_ PPAD _-complete even in dimension d_ = 4 _and B_ = 2 _._


The above result excludes the existence of an algorithm for GDAFixedPoint whose running
time is poly ( log _G_, log _L_, log ( 1/ _α_ ), _B_ ) and, equivalently, the existence of an algorithm for the
problem LR-LocalMinMax with running time poly ( log _G_, log _L_, log ( 1/ _ε_ ), log ( 1/ _δ_ ), _B_ ), unless
FP = PPAD. Observe that it would not be possible to get a stronger hardness result for the four
dimensional GDAFixedPoint problem since it is simple to construct brute-force search algorithms with running time poly ( 1/ _α_, _G_, _L_, _B_ ) . We elaborate more on such algorithms towards the
end of this section. In order to prove the hardness of GDAFixedPoint for polynomially (rather
than exponentially) bounded (in the size of the input) values of 1/ _α_, _G_, and _L_ (See Theorem 4.4)
we need to consider optimization problems in higher dimensions. This is the problem that we
explore in Section 7. Beyond establishing the hardness of the problem for _d_ = 4 dimensions, the
purpose of this section is to provide a simpler reduction that helps in the understanding of our
main result in the next section.


**6.1** **The 2D Bi-Sperner Problem**


We start by introducing the 2D-BiSperner problem. Consider a coloring of the _N_ _×_ _N_, 2dimensional grid, where instead of coloring each vertex of the grid with a single color (as in
Sperner’s lemma), each vertex is colored via a combination of two out of four available colors.
The four available colors are 1 _[−]_, 1 [+], 2 _[−]_, 2 [+] . The five rules that define a proper coloring of the
_N_ _×_ _N_ grid are the following.


19


Figure 4: _Left_ : Summary of the rules from a proper coloring of the grid. The **gray** color on the
left and the right side can be replaced with either **blue** or **green** . Similarly the **gray** color on
the top and the bottom side can be replaced with either **red** or **yellow** . _Right:_ An example of a
proper coloring of a 9 _×_ 9 grid. The **brown** boxes indicate the two panchromatic cells, i.e., the
cells where all the four available colors appear.


1. The first color of every vertex is either 1 _[−]_ or 1 [+] and the second color is either 2 _[−]_ or 2 [+] .


.
2. The first color of all vertices on the left boundary of the grid is 1 [+]


.
3. The first color of all vertices on the right boundary of the grid is 1 _[−]_


.
4. The second color of all vertices on the bottom boundary of the grid is 2 [+]


.
5. The second color of all vertices on the top boundary of the grid is 2 _[−]_


Using similar proof ideas as in Sperner’s lemma it is not hard to establish via a combinatorial
argument that, in every proper coloring of the _N_ _×_ _N_ grid, there exists a square cell where each
of the four colors in _{_ 1 _[−]_, 1 [+], 2 _[−]_, 2 [+] _}_ appears in at least one of its vertices. We call such a cell a
_panchromatic square_ . In the 2D-BiSperner problem, defined formally below, we are given the description of some coloring of the grid and are asked to find either a panchromatic square or the violation of the proper coloring conditions. In this paper, we will not present a direct combinatorial
argument guaranteeing the existence of panchromatic squares under proper colorings of the grid,
since the existence of panchromatic squares will be implied by the totality of the 2D-BiSperner
problem, which will follow from our reduction from 2D-BiSperner to GDAFixedPoint as well
as our proofs in Section 5 establishing the totality of GDAFixedPoint. In Figure 4 we summarize
the five rules that define proper colorings and we present an example of a proper coloring of the
grid with 9 discrete points on each side.


20


In order to formally define the computational problem 2D-BiSperner in a way that is useful
for our reductions we need to allow for colorings of the _N_ _×_ _N_ grid described in a succinct
way, where the value _N_ can be exponentially large compared to the size of the input to the
problem. A standard way to do this, introduced by [Pap94b] in defining the computational
version of Sperner’s lemma, is to describe a coloring via a binary circuit _C_ _l_ that takes as input
the coordinates of a vertex in the grid and outputs the combination of colors that is used to color
this vertex. In the input, each one of the two coordinates of the input vertex is given via the
binary representation of a number in [ _N_ ] _−_ 1. Setting _N_ = 2 _[n]_ we have that the representation of
each coordinate belongs to _{_ 0, 1 _}_ _[n]_ . In the rest of the section we abuse the notation and we use a
coordinate _i_ _∈{_ 0, 1 _}_ _[n]_ both as a binary string and as a number in [ 2 _[n]_ ] _−_ 1 and it is clear from the
context which of the two we use. The output of _C_ _l_ should be a combination of one of the colors
_{_ 1 _[−]_, 1 [+] _}_ and one of the colors _{_ 2 _[−]_, 2 [+] _}_ . We represent this combination as a pair of _{−_ 1, 1 _}_ [2] . The
first coordinate of this pair refers to the choice of 1 _[−]_ or 1 [+] and the second coordinate refers to
the choice of 2 _[−]_ or 2 [+] .


In the definition of the computational problem 2D-BiSperner the input is a circuit _C_ _l_, as described above. One type of possible solutions to 2D-BiSperner is providing a pair of coordinates
( _i_, _j_ ) _∈{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ indexing a cell of the grid whose bottom left vertex is ( _i_, _j_ ) . For this
type of solution to be valid it must be that the output of _C_ _l_ when evaluated on all the vertices of
this square contains at least one negative and one positive value for each one of the two output
coordinates of _C_ _l_, i.e. the cell must be panchromatic. Another type of possible solution to 2DBiSperner is a vertex whose coloring violates the proper coloring conditions for the boundary,
namely 2–5 above. For notational convenience we refer to the first coordinate of the output of _C_ _l_
by _C_ _l_ [1] [and to the second coordinate by] _[ C]_ _l_ [2] [. The formal definition of the computational problem]
2D-BiSperner is then the following.


**2D-BiSperner.**
Input: A boolean circuit _C_ _l_ : _{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ _→{−_ 1, 1 _}_ [2] .


Output: A vertex ( _i_, _j_ ) _∈{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ such that one of the following holds

1. _i_ _̸_ = **1**, _j_ _̸_ = **1**, and



�

_i_ _[′]_ _−_ _i_ _∈{_ 0,1 _}_

_−_
_j_ _[′]_ _j_ _∈{_ 0,1 _}_



_C_ _l_ [1] [(] _[i]_ _[′]_ [,] _[ j]_ _[′]_ [) =] _[ {−]_ [1, 1] _[}]_ and �

_i_ _[′]_ _−_ _i_ _∈{_ 0,1 _}_

_−_
_j_ _[′]_ _j_ _∈{_ 0,1 _}_



_C_ _l_ [2] [(] _[i]_ _[′]_ [,] _[ j]_ _[′]_ [) =] _[ {−]_ [1, 1] _[}]_ [, or]



2. _i_ = **0** and _C_ _l_ [1] [(] _[i]_ [,] _[ j]_ [) =] _[ −]_ [1, or]
3. _i_ = **1** and _C_ _l_ [1] [(] _[i]_ [,] _[ j]_ [) = +] [1, or]
4. _j_ = **0** and _C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [) =] _[ −]_ [1, or]
5. _j_ = **1** and _C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [) = +] [1.]


Our next step is to show that the problem 2D-BiSperner is PPAD-hard. Thus our reduction
from 2D-BiSperner to GDAFixedPoint in the next section establishes both the PPAD-hardness

of GDAFixedPoint and the inclusion of 2D-BiSperner to PPAD.


**Lemma 6.2.** _The problem_ 2D _-_ BiSperner _is_ PPAD _-hard._


_Proof._ To prove this Lemma we will use Lemma 2.5. Let _C_ _M_ be a polynomial-time Turing machine
that computes a function _M_ : [ 0, 1 ] [2] _→_ [ 0, 1 ] [2] that is _L_ -Lipschitz. We know from Lemma 2.5 that


21


finding _γ_ -approximate fixed points of _M_ is PPAD-hard. We will use _C_ _M_ to define a circuit _C_ _l_ such
that a solution of 2D-BiSperner with input _C_ _l_ will give us a _γ_ -approximate fixed point of _M_ .

Consider the function _g_ ( _**x**_ ) = _M_ ( _**x**_ ) _−_ _**x**_ . Since _M_ is _L_ -Lipschitz, the function _g_ : [ 0, 1 ] [2] _→_

[ _−_ 1, 1 ] [2] is also ( _L_ + 1 ) -Lipschitz. Additionally _g_ can be easily computed via a polynomial-time
Turing machine _C_ _g_ that uses _C_ _M_ as a subroutine. We construct a proper coloring of a fine grid of

[ 0, 1 ] [2] using the signs of the outputs of _g_ . Namely we set _n_ = _⌈_ log ( _L_ / _γ_ ) + 2 _⌉_ and this defines
a 2 _[n]_ _×_ 2 _[n]_ grid over [ 0, 1 ] [2] that is indexed by _{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ . Let _g_ _η_ : [ 0, 1 ] [2] _→_ [ _−_ 1, 1 ] [2] be the
function that the Turing Machine _C_ _g_ evaluate when the requested accuracy is _η_ _>_ 0. Now we can
define the circuit _C_ _l_ as follows, [5]



1 _i_ = 0


_−_ 1 _i_ = 2 _[n]_ _−_ 1

1 _g_ _η_,1 2 ~~_[n]_~~ _−_ _i_ 1 [,] 2 ~~_[n]_~~ _−_ _j_ 1 _≥_ 0 and _i_ _̸_ = _−_ 1
� �

_−_ 1 _g_ _η_,1 2 ~~_[n]_~~ _−_ _i_ 1 [,] 2 ~~_[n]_~~ _−_ _j_ 1 _<_ 0 and _i_ _̸_ = 0
� �


1 _i_ = 0


_−_ 1 _i_ = 2 _[n]_ _−_ 1

1 _g_ _η_,2 2 ~~_[n]_~~ _−_ _i_ 2 [,] 2 ~~_[n]_~~ _−_ _j_ 1 _≥_ 0 and _i_ _̸_ = _−_ 1
� �

_−_ 1 _g_ _η_,2 2 ~~_[n]_~~ _−_ _i_ 2 [,] 2 ~~_[n]_~~ _−_ _j_ 1 _<_ 0 and _i_ _̸_ = 0
� �



_C_ _l_ [1] [(] _[i]_ [,] _[ j]_ [) =]


_C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [) =]




































,


,



where _g_ _i_ is the _i_ th output coordinate of _g_ . It is not hard then to observe that the coloring _C_ _l_ is
proper, i.e. it satisfies the boundary conditions due to the fact that the image of _M_ is always
inside [ 0, 1 ] [2] . Therefore the only possible solution to 2D-BiSperner with input _C_ _l_ is a cell that
contains all the colors _{_ 1 _[−]_, 1 [+], 2 _[−]_, 2 [+] _}_ . Let ( _i_, _j_ ) be the bottom-left vertex of this cell which we
denote by _R_, namely



.
��



_i_
_R_ = � _**x**_ _∈_ [ 0, 1 ] [2] _|_ _x_ 1 _∈_ � 2 _[n]_ _−_ 1 [,] 2 _[i]_ _[n]_ [ +] _−_ [ 1] 1



, _x_ 2 _∈_ _j_ _[j]_ [ +] [ 1]
� � 2 _[n]_ _−_ 1 [,] 2 _[n]_ _−_ 1



_γ_
**Claim 6.3.** _Let η_ = 2 ~~_√_~~



_γ_
2 _[, there exists]_ _**[ x]**_ _[ ∈]_ _[R such that]_ _[ |]_ _[g]_ [1] [(] _**[x]**_ [)] _[| ≤]_ 2 ~~_√_~~



2 _[and]_ _**[ y]**_ _[ ∈]_ _[R such that]_ _[ |]_ _[g]_ [2] [(] _**[y]**_ [)] _[| ≤]_ 2 ~~_√_~~ _γ_



2 _[.]_



_Proof of Claim 6.3._ We will prove the existence of _**x**_ and the existence of _**y**_ follows using an identical argument. If there exists a corner _**x**_ of _R_ such that _g_ 1 ( _**x**_ ) is in the range [ _−_ _η_, _η_ ] then the claim
follows. Suppose not. Using this together with the fact that the first color of one of the corners of
_R_ is 1 _[−]_ and also the first color of one of the corners of _R_ is 1 [+] we conclude that there exist points
_**x**_, _**x**_ _[′]_ such that _g_ _η_,1 ( _**x**_ ) _≥_ 0 and _g_ _η_,1 ( _**x**_ _[′]_ ) _≤_ 0 [6] . But we have that �� _g_ _η_ _−_ _g_ �� 2 _[≤]_ _[η]_ [. This together with]
the fact that _g_ 1 ( _**x**_ ) _̸∈_ [ _−_ _η_, _η_ ] and _g_ 1 ( _**x**_ _[′]_ ) _̸∈_ [ _−_ _η_, _η_ ] implies that _g_ 1 ( _**x**_ ) _≥_ 0 and also _g_ 1 ( _**x**_ _[′]_ ) _≤_ 0. But
because of the _L_ -Lipschitzness of _g_ and because the distance between _**x**_ and _**x**_ _[′]_ is at most _√_ 2 4 _[γ]_ _L_

_γ_
we conclude that _|_ _g_ 1 ( _**x**_ ) _−_ _g_ 1 ( _**x**_ _[′]_ ) _| ≤_ 2 ~~_√_~~ 2 [. Hence due to the signs of] _[ g]_ [1] [(] _**[x]**_ [)] [ and] _[ g]_ [1] [(] _**[x]**_ _[′]_ [)] [ we conclude]



2 _[γ]_



2 [. Hence due to the signs of] _[ g]_ [1] [(] _**[x]**_ [)] [ and] _[ g]_ [1] [(] _**[x]**_ _[′]_ [)] [ we conclude]



that _|_ _g_ 1 ( _**x**_ ) _| ≤_ 2 ~~_√_~~ _γ_



_γ_
2 [. The same way we can prove that] _[ |]_ _[g]_ [1] [(] _**[y]**_ [)] _[| ≤]_ 2 ~~_√_~~



2 [and the claim follows.]



5 We remind that we abuse the notation and we use a coordinate _i_ _∈{_ 0, 1 _}_ _n_ both as a binary string and as a number
in ([ 2 _[n]_ _−_ 1 ] _−_ 1 ) and it is clear from the context which of the two we use.
6 _n_
The latter is inaccurate for the cases where the vertex ( 0, _j_ ) belongs to either facets _i_ = 0 or _i_ = 2 _−_ 1. Notice that
the coloring in such vertices does not depend on the value of _g_ _η_ . However in case where the color of such a corner is
not consistent with the value of _g_ _η_, i.e. _g_ _η_,1 ( 0, _j_ ) _<_ 0 and _C_ _l_ [1] [(] [0,] _[ j]_ [) =] [ 1 then this means that] _[ |]_ _[g]_ [1] [(] [0,] _[ j]_ [)] _[| ≤]_ _[η]_ [. This is due]
to the fact that _g_ 1 ( 0, _j_ ) _≥_ 0 and _|_ _g_ 1 ( 0, _j_ ) _−_ _g_ 1, _η_ ( 0, _j_ ) _| ≤_ _η_ .


22


Using the Claim 6.3 and the _L_ -Lipschitzness of _g_ we get that for every _**z**_ _∈_ _R_



2 _·_ _L_ _·_ _[γ]_



, and
2



_|_ _g_ 1 ( _**z**_ ) _−_ _g_ 1 ( _**x**_ ) _| ≤_ _L_ _∥_ _**x**_ _−_ _**z**_ _∥_ 2 _≤_ _√_



2 _·_ _L_ _·_ _[γ]_



_|_ _g_ 2 ( _**z**_ ) _−_ _g_ 2 ( _**y**_ ) _| ≤_ _L_ _∥_ _**y**_ _−_ _**z**_ _∥_ 2 _≤_ _√_




_[γ]_

4 _L_ [=] _[⇒|]_ _[g]_ [1] [(] _**[z]**_ [)] _[| ≤]_ ~~_√_~~ _[γ]_

_[γ]_

4 _L_ [=] _[⇒|]_ _[g]_ [2] [(] _**[z]**_ [)] _[| ≤]_ ~~_√_~~ _[γ]_



2



where we have used also the fact that for any two points _**z**_, _**w**_ it holds that _∥_ _**z**_ _−_ _**w**_ _∥_ 2 _≤_ _√_



2 _[γ]_



, 2 4 _L_

which follows from the definition of the size of the grid. Therefore we have that _∥_ _g_ ( _**z**_ ) _∥_ 2 _≤_ _γ_ and
hence _∥_ _M_ ( _**z**_ ) _−_ _**z**_ _∥_ 2 _≤_ _γ_ which implies that any point _**z**_ _∈_ _R_ is a _γ_ -approximate fixed point of _M_
and the lemma follows.



Now that we have established the PPAD-hardness of 2D-BiSperner we are ready to present our
main result of this section which is a reduction from 2D-BiSperner to GDAFixedPoint.


**6.2** **From 2D Bi-Sperner to Fixed Points of Gradient Descent/Ascent**


We start with presenting a construction of a Lipschitz and smooth real-valued function _f_ :

[ 0, 1 ] [2] _×_ [ 0, 1 ] [2] _→_ **R** based on a given coloring circuit _C_ _l_ : _{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ _→{−_ 1, 1 _}_ [2] . Then
in Section 6.2.1 we will show that any solution to GDAFixedPoint with input the representation
_C_ _f_ of _f_ is also a solution to the 2D-BiSperner problem with input _C_ _l_ . Constructing Lipschitz
and smooth functions based on only local information is a surprisingly challenging task in highdimensions as we will explain in detail in Section 7. Fortunately in the low-dimensional case
that we consider in this section the construction is much more simple and the main ideas of our
reduction are more clear.


The basic idea of the construction of _f_ consists in interpreting the coloring of a given point
in the grid as the directions of the gradient of _f_ ( _**x**_, _**y**_ ) with respect to the variables _x_ 1, _y_ 1 and
_x_ 2, _y_ 2 respectively. More precisely, following the ideas in the proof of Lemma 6.2, we divide the

[ 0, 1 ] [2] square in _square-cells_ of length 1/ ( _N_ _−_ 1 ) = 1/ ( 2 _[n]_ _−_ 1 ) where the corners of these cells
correspond to vertices of the _N_ _×_ _N_ grid of the 2D-BiSperner instance described by _C_ _l_ . When _**x**_
is on a vertex of this grid, the first color of this vertex determines the direction of gradient with
respect to the variables _x_ 1 and _y_ 1, while the second color of this vertex determines the direction
of the gradient of the variables _x_ 2 and _y_ 2 . As an example, if _**x**_ = ( _x_ 1, _x_ 2 ) is on a vertex of the
_N_ _×_ _N_ grid, and the coloring of this vertex is ( 1 _[−]_, 2 [+] ), i.e. the output of _C_ _l_ on this vertex is
( _−_ 1, + 1 ), then we would like to have


_∂_ _f_ _∂_ _f_ _∂_ _f_ _∂_ _f_
( _**x**_, _**y**_ ) _≥_ 0, ( _**x**_, _**y**_ ) _≤_ 0, ( _**x**_, _**y**_ ) _≤_ 0, ( _**x**_, _**y**_ ) _≥_ 0.
_∂x_ 1 _∂y_ 1 _∂x_ 2 _∂y_ 2


The simplest way to achieve this is to define the function _f_ locally close to ( _**x**_, _**y**_ ) to be equal to


_f_ ( _**x**_, _**y**_ ) = ( _x_ 1 _−_ _y_ 1 ) _−_ ( _x_ 2 _−_ _y_ 2 ) .


Similarly, if _**x**_ is on a vertex of the _N_ _×_ _N_ grid, and the coloring of this vertex is ( 1 _[−]_, 2 _[−]_ ), i.e. the
output of _C_ _l_ on this vertex is ( _−_ 1, _−_ 1 ), then we would like to have


_∂_ _f_ _∂_ _f_ _∂_ _f_ _∂_ _f_
( _**x**_, _**y**_ ) _≥_ 0, ( _**x**_, _**y**_ ) _≤_ 0, ( _**x**_, _**y**_ ) _≥_ 0, ( _**x**_, _**y**_ ) _≤_ 0.
_∂x_ 1 _∂y_ 1 _∂x_ 2 _∂y_ 2


The simplest way to achieve this is to define the function _f_ locally close to ( _**x**_, _**y**_ ) to be equal to


_f_ ( _**x**_, _**y**_ ) = ( _x_ 1 _−_ _y_ 1 ) + ( _x_ 2 _−_ _y_ 2 ) .


23


Figure 5: The correspondence of the colors of the vertices of the _N_ _×_ _N_ grid with the directions
of the gradient of the function _f_ that we design.


In Figure 5 we show pictorially the correspondence of the colors of the vertices of the grid with
the gradient of the function _f_ that we design. As shown in the figure, any set of vertices that
share at least one of the colors 1 [+], 1 _[−]_, 2 [+], 2 _[−]_, agree on the direction of the gradient with respect
the horizontal or the vertical axis. This observation is one of the main ingredients in the proof of
correctness of our reduction that we present later in this section.
When _**x**_ is not on a vertex of the _N_ _×_ _N_ grid then our goal is to define _f_ via interpolating
the functions corresponding to the corners of the cell in which _**x**_ belongs. The reason that this
interpolation is challenging is that we need to make sure the following properties are satisfied


_▷_ the resulting function _f_ is both Lipschitz and smooth inside every cell,


_▷_ the resulting function _f_ is both Lipschitz and smooth even at the boundaries of every cell,
where two differect cells stick together,


_▷_ no solution to the GDAFixedPoint problem is created inside cells that are not solutions to
the 2D-BiSperner problem. In particular, it has to be true that if all the vertices of one cell
agree on some color then the gradient of _f_ inside that cell has large enough gradient in the
corresponding direction.


For the low dimensional case, that we explore in this section, satisfying the first two properties
is not a very difficult task, whereas for the third property we need to be careful and achieving
this property is the main technical contribution of this section. On the contrary, for the highdimensional case that we explore in Section 7 even achieving the first two properties is very
challenging and technical.
As we will see in Section 6.2.1, if we accomplish a construction of a function _f_ with the
aforementioned properties, then the fixed points of the projected Gradient Descent/Ascent can
only appear inside cells that have all of the colors _{_ 1 _[−]_, 1 [+], 2 _[−]_, 2 [+] _}_ at their corners. To see this
consider a cell that misses some color, e.g. 1 [+] . Then all the corners of this cell have as first
color 1 _[−]_ . Since _f_ is defined as interpolation of the functions in the corners of the cells, with the
aforementioned properties, inside that cell there is always a direction with respect to _x_ 1 and _y_ 1
for which the gradient is large enough. Hence any point inside that cell cannot be a fixed point
of the projected Gradient Descent/Ascent. Of course this example provides just an intuition of
our construction and ignores case where the cell is on the boundary of the grid. We provide a
detailed explanation of this case in Section 6.2.1.
The above neat idea needs some technical adjustments in order to work. At first, the interpolation of the function in the interior of the cell must be smooth enough so that the resulting


24


function is both Lipschitz and smooth. In order to satisfy this, we need to choose appropriate
coefficients of the interpolation that interpolate smoothly not only the value of the function but
also its derivatives. For this purpose we use the following smooth step function of order 1.


**Definition 6.4** (Smooth Step Function of Order 1) **.** We define _S_ 1 : [ 0, 1 ] _→_ [ 0, 1 ] to be the _smooth_
_step function of order_ 1 that is equal to _S_ 1 ( _x_ ) = 3 _x_ [2] _−_ 2 _x_ [3] . Observe that the following hold _S_ 1 ( 0 ) =
0, _S_ 1 ( 1 ) = 1, _S_ 1 _[′]_ [(] [0] [) =] [ 0, and] _[ S]_ 1 _[′]_ [(] [1] [) =] [ 0.]


As we have discussed, another issue is that since the interpolation coefficients depend on
the value of _**x**_ it could be that the derivatives of these coefficients overpower the derivatives of
the functions that we interpolate. In this case we could be potentially creating fixed points of
Gradient Descent/Ascent even in _non_ panchromatic squares. As we will see later the magnitude
of the derivatives from the interpolation coefficients depends on the differences _x_ 1 _−_ _y_ 1 and _x_ 2 _−_
_y_ 2 . Hence if we ensure that these differences are small then the derivatives of the interpolation
coefficients will have to remain small and hence they can never overpower the derivatives from
the corners of every cell. This is the place in our reduction where we add the constraints _**A**_ _·_
( _**x**_, _**y**_ ) _≤_ _**b**_ that define the domain of the function _f_ as we describe in Section 3.


Now that we have summarized the main ideas of our construction we are ready for the formal
definition of _f_ based on the coloring circuit _C_ _l_ .


**Definition 6.5** (Continuous and Smooth Function from Colorings of 2D-Bi-Sperner) **.** Given a
binary circuit _C_ _l_ : _{_ 0, 1 _}_ _[n]_ _× {_ 0, 1 _}_ _[n]_ _→{−_ 1, 1 _}_ [2], we define the function _f_ _C_ _l_ : [ 0, 1 ] [2] _×_ [ 0, 1 ] [2] _→_ **R** as
follows. For any _**x**_ _∈_ [ 0, 1 ] [2], let _A_ = ( _i_ _A_, _j_ _A_ ), _B_ = ( _i_ _B_, _j_ _B_ ), _C_ = ( _i_ _C_, _j_ _C_ ), _D_ = ( _i_ _D_, _j_ _D_ ) be the vertices
of the cell of the _N_ (= 2 _[n]_ ) _×_ _N_ grid which contains _**x**_ and _**x**_ _[A]_, _**x**_ _[B]_, _**x**_ _[C]_ and _**x**_ _[C]_ the corresponding
points in the unit square [ 0, 1 ] [2], i.e. _x_ 1 _[A]_ [=] _[ i]_ _[A]_ [/] [(] [2] _[n]_ _[ −]_ [1] [)] [,] _[ x]_ 2 _[A]_ [=] _[ j]_ _[A]_ [/] [(] [2] _[n]_ _[ −]_ [1] [)] [ etc. Let also] _[ A]_ [ be]
down-left corner of this cell and _B_, _C_, _D_ be the rest of the vertices in clockwise order, then we
define
_f_ _C_ _l_ ( _**x**_, _**y**_ ) = _α_ 1 ( _**x**_ ) _·_ ( _y_ 1 _−_ _x_ 1 ) + _α_ 2 ( _**x**_ ) _·_ ( _y_ 2 _−_ _x_ 2 )


where the coefficients _α_ 1 ( _**x**_ ), _α_ 2 ( _**x**_ ) _∈_ [ _−_ 1, 1 ] are defined as follows



�



�




_·_ _S_ 1
�




_· C_ _l_ _[i]_ [(] _[B]_ [)]
�



_D_
_x_ 2 _−_ _x_ 2
� _δ_



_D_
_x_ 1 _[−]_ _[x]_ [1]
� _δ_



_α_ _i_ ( _**x**_ ) = _S_ 1



_x_ _[C]_
1 _[−]_ _[x]_ [1]


_δ_

�




_·_ _S_ 1



_x_ _[C]_
2 _[−]_ _[x]_ [2]


_δ_

�




_· C_ _l_ _[i]_ [(] _[A]_ [) +] _[ S]_ [1]



�




_·_ _S_ 1
�




_· C_ _l_ _[i]_ [(] _[C]_ [) +] _[ S]_ [1]
�




_· C_ _l_ _[i]_ [(] _[D]_ [)]
�



_B_
_x_ 1 _−_ _x_ 1
� _δ_



_B_
_x_ 2 _[−]_ _[x]_ [2]
� _δ_



+ _S_ 1



_x_ 1 _−_ _x_ 1 _[A]_

_δ_
�




_·_ _S_ 1



_A_
_x_ 2 _−_ _x_ 2
� _δ_



where _δ_ ≜ 1/ ( _N_ _−_ 1 ) = 1/ ( 2 _[n]_ _−_ 1 ) .


In Figure 6 we present an example of the application of Definition 6.5 to a specific cell with some
given coloring on the corners.
An important property of the definition of the function _f_ _C_ _l_ is that the coefficients used in the
definition of _α_ _i_ have the following two properties



�



�




_·_ _S_ 1
�



_≥_ 0,
�



_D_
_x_ 2 _−_ _x_ 2
� _δ_



_S_ 1



_x_ _[C]_
1 _[−]_ _[x]_ [1]


_δ_

�




_·_ _S_ 1



_x_ _[C]_
2 _[−]_ _[x]_ [2]


_δ_

�



_≥_ 0, _S_ 1



_D_
_x_ 1 _[−]_ _[x]_ [1]
� _δ_



�




_·_ _S_ 1
�



_≥_ 0, _S_ 1
�



_≥_ 0, and
�



_B_
_x_ 1 _−_ _x_ 1
� _δ_



_B_
_x_ 2 _[−]_ _[x]_ [2]
� _δ_



_S_ 1



_x_ 1 _−_ _x_ 1 _[A]_

_δ_
�




_·_ _S_ 1



_A_
_x_ 2 _−_ _x_ 2
� _δ_



25


Figure 6: Example of the definition of the Lipschitz and smooth function _f_ on some cell given
the coloring on the corners of the cell. For details see Definition 6.5.



�



�



_D_
_x_ 2 _−_ _x_ 2
� _δ_ �



_x_ _[C]_
2 _[−]_ _[x]_ [2]


_δ_

�



+ _S_ 1



_D_
_x_ 1 _[−]_ _[x]_ [1] _·_ _S_ 1
� _δ_ �



_S_ 1



_x_ _[C]_
1 _[−]_ _[x]_ [1]


_δ_

�




_·_ _S_ 1



�



_B_
_x_ 1 _−_ _x_ 1
� _δ_



= 1.
�




_·_ _S_ 1
�



_B_
_x_ 2 _[−]_ _[x]_ [2]
� _δ_



_A_
_x_ 2 _−_ _x_ 2
� _δ_



+ _S_ 1
�



+ _S_ 1



_x_ 1 _−_ _x_ 1 _[A]_

_δ_
�




_·_ _S_ 1



Hence the function _f_ _C_ _l_ inside a cell is a smooth convex combination of the functions on the
corners of the cell, as is suggested from Figure 6. Of course there are many ways to define such
convex combination but in our case we use the smooth step function _S_ 1 to ensure the Lipschitz
continuous gradient of the overall function _f_ _C_ _l_ . We prove this formally in the next lemma.


**Lemma 6.6.** _Let f_ _C_ _l_ _be the function defined based on a coloring circuit_ _C_ _l_ _, as per Definition 6.5. Then_
_f_ _C_ _l_ _is continuous and differentiable at any point_ ( _**x**_, _**y**_ ) _∈_ [ 0, 1 ] [4] _. Moreover, f_ _C_ _l_ _is_ Θ ( 1/ _δ_ ) _-Lipschitz and_
Θ ( 1/ _δ_ [2] ) _-smooth in the whole 4-dimensional hypercube_ [ 0, 1 ] [4] _, where δ_ = 1/ ( _N_ _−_ 1 ) = 1/ ( 2 _[n]_ _−_ 1 ) _._


_Proof._ Clearly from Definition 6.5, _f_ _C_ _l_ is differentiable at any point ( _**x**_, _**y**_ ) _∈_ [ 0, 1 ] [4] in which _**x**_ lies
on the strict interior of its respective cell. In this case the derivative with respect to _x_ 1 is



_∂_ _f_ _C_ _l_ ( _**x**_, _**y**_ )




[1] [(] _**[x]**_ [)]

_·_ ( _y_ 1 _−_ _x_ 1 ) _−_ _α_ 1 ( _**x**_ ) + _[∂α]_ [2] [(] _**[x]**_ [)]
_∂x_ 1 _∂x_ 1



_l_ ( _**x**_, _**y**_ ) = _[∂α]_ [1] [(] _**[x]**_ [)]

_∂x_ 1 _∂x_ 1




_·_ ( _y_ 2 _−_ _x_ 2 ) .
_∂x_ 1



where for _∂α_ 1 ( _**x**_ ) / _∂x_ 1 we have that



_∂α_ 1 ( _**x**_ )



_δ_ _[S]_ 1 _[′]_




_·_ _S_ 1



1 ( _**x**_ )

= _−_ [1]
_∂x_ 1 _δ_



_x_ _[C]_
1 _[−]_ _[x]_ [1]


_δ_

�


_D_
_x_ 1 _[−]_ _[x]_ [1]
� _δ_

_x_ 1 _−_ _x_ 1 _[A]_

_δ_
�


_B_
_x_ 1 _−_ _x_ 1
� _δ_



�



_x_ _[C]_
2 _[−]_ _[x]_ [2]


_δ_

�


_D_
_x_ 2 _−_ _x_ 2
� _δ_



_x_ _[C]_
2 _[−]_ _[x]_ [2]


_δ_

�



�




_· C_ _l_ [1] [(] _[A]_ [)]



_−_ [1]

_δ_ _[S]_ 1 _[′]_




_·_ _S_ 1
�



+ [1] _δ_ _[S]_ 1 _[′]_



�



_A_
_x_ 2 _−_ _x_ 2
� _δ_




_·_ _S_ 1




_· C_ _l_ [1] [(] _[B]_ [)]
�


_· C_ _l_ [1] [(] _[C]_ [)]
�


_· C_ _l_ [1] [(] _[D]_ [)] [.]
�



_B_
_x_ 2 _[−]_ _[x]_ [2]
� _δ_



+ [1] _δ_ _[S]_ 1 _[′]_




_·_ _S_ 1
�



26


_∂α_ 1 ( _**x**_ )
Now since max _z_ _∈_ [ 0,1 ] _|_ _S_ 1 _[′]_ [(] _[z]_ [)] _[| ≤]_ [6, we can conclude that] ��� _∂x_ 1



_≤_ 24/ _δ_ . Similarly we can prove
���



_≤_ _O_ ( 1/ _δ_ ) . Using similar
���



_∂α_ 2 ( _**x**_ )
that
��� _∂x_ 1



��� _≤_ 24/ _δ_, which combined with _|_ _α_ 1 ( _**x**_ ) _| ≤_ 1 implies ��� _∂_ _f_ _C_ _∂_ _l_ ( _x_ _**x**_ 1, _**y**_ )



_≤_ 1 for _i_ = 1, 2. Hence
���



reasoning we can prove that ��� _∂_ _f_ _C_ _∂_ _l_ ( _x_ _**x**_ 2, _**y**_ )



��� _≤_ _O_ ( 1/ _δ_ ) and that ��� _∂_ _f_ _C_ _∂_ _l_ ( _y_ _**x**_ _i_, _**y**_ )



�� _∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� 2 _[≤]_ _[O]_ [(] [1/] _[δ]_ [)] [.]


The only thing we are missing to prove the Lipschitzness of _f_ _C_ _l_ is to prove its continuity on the
boundaries of the cells of our subdivision. Suppose _**x**_ lies on the boundary of some cell, e.g. let
_**x**_ lie on edge ( _C_, _D_ ) of one cell that is the same as the edge ( _A_ _[′]_, _B_ _[′]_ ) of the cell to the right of that
cell. Since _S_ 1 ( 0 ) = 0, _S_ 1 _[′]_ [(] [0] [) =] [ 0 and] _[ S]_ 1 _[′]_ [(] [1] [) =] [ 0 it holds that] _[ ∂α]_ [1] [(] _**[x]**_ [)] [/] _[∂][x]_ [1] [ =] [ 0 and the same for] _[ α]_ [2] [.]
Therefore the value of _∂_ _f_ _C_ _l_ / _∂x_ 1 remains the same no matter the cell according to which it was
calculated. As a result, _f_ _C_ _l_ is differentiable with respect to _x_ 1 even if _**x**_ belongs in the boundary
of its cell. Using the exact same reasoning for the rest of the variables, one can show that the
function _f_ _C_ _l_ is differentiable at any point ( _**x**_, _**y**_ ) _∈_ [ 0, 1 ] [4] and because of the aforementioned bound
on the gradient _∇_ _f_ _C_ _l_ we can conclude that _f_ _C_ _l_ is _O_ ( 1/ _δ_ ) -Lipschitz.


Using very similar calculations, we can compute the closed formulas of the second derivatives
of _f_ _C_ _l_ and using the bounds �� _f_ _C_ _l_ ( _·_ ) �� _≤_ 2, _|_ _S_ 1 ( _·_ ) _| ≤_ 1, _|_ _S_ 1 _′_ [(] _[·]_ [)] _[| ≤]_ [6, and] _[ |]_ _[S]_ 1 _[′′]_ [(] _[·]_ [)] _[| ≤]_ [6, we can prove]
that each entry of the Hessian _∇_ [2] _f_ _C_ _l_ ( _**x**_, _**y**_ ) is bounded by _O_ ( 1/ _δ_ [2] ) and thus

2
�� _∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� 2 _[≤]_ _[O]_ [(] [1/] _[δ]_ [2] [)]


which implies the Θ ( 1/ _δ_ [2] ) -smoothness of _f_ _C_ _l_ .


**6.2.1** **Description and Correctness of the Reduction – Proof of Theorem 6.1**


In this section, we present and prove the exact polynomial-time construction of the instance of
the problem GDAFixedPoint from an instance _C_ _l_ of the problem 2D-BiSperner.


**(+)** **Construction of Instance for Fixed Points of Gradient Descent/Ascent.**
Our construction can be described via the following properties.


 - The payoff function is the real-valued function _f_ _C_ _l_ ( _**x**_, _**y**_ ) from the Definition 6.5.


 - The domain is the polytope _P_ ( _**A**_, _**b**_ ) that we described in Section 3. The matrix _**A**_ and the
vector _**b**_ have constant size and they are computed so that the following inequalities hold


_x_ 1 _−_ _y_ 1 _≤_ ∆, _y_ 1 _−_ _x_ 1 _≤_ ∆, _x_ 2 _−_ _y_ 2 _≤_ ∆, and _y_ 2 _−_ _x_ 2 _≤_ ∆ (6.1)


where ∆ = _δ_ /12 and _δ_ = 1/ ( _N_ _−_ 1 ) = 1/ ( 2 _[n]_ _−_ 1 ) .


 - The parameter _α_ is set to be equal to ∆/3.


 - The parameters _G_ and _L_ are set to be equal to the upper bounds on the Lipschitzness and
the smoothness of _f_ _C_ _l_ respectively that we derived in Lemma 6.6. Namely we have that
_G_ = _O_ ( 1/ _δ_ ) = _O_ ( 2 _[n]_ ) and _L_ = _O_ ( 1/ _δ_ [2] ) = _O_ ( 2 [2] _[n]_ ) .


The first thing that is simple to observe in the above reduction is that it runs in polynomial
time with respect to the size of the the circuit _C_ _l_ which is the input to the 2D-BiSperner problem
that we started with. To see this, recall from the definition of GDAFixedPoint that our reduction


27


needs to output: (1) a Turing machine _C_ _f_ _C_ _l_ that computes the value and the gradient of the
function _f_ _C_ _l_ in time polynomial in the number of requested bits of accuracy; (2) the required
scalars _α_, _G_, and _L_ . For the first, we observe from the definition of _f_ _C_ _l_ that it is actually a piecewise polynomial function with a closed form that only depends on the values of the circuit _C_ _l_ on
the corners of the corresponding cell. Since the size of _C_ _l_ is the size of the input to 2D-BiSperner
we can easily construct a polynomial-time Turing machine that computes both function value and
the gradient of the piecewise polynomial function _f_ _C_ _l_ . Also, from the aforementioned description
of the reduction we have that log ( _G_ ), log ( _L_ ) and log ( 1/ _α_ ) are linear in _n_ and hence we can
construct the binary representation of all this scalars in time _O_ ( _n_ ) . The same is true for the
coefficients of _**A**_ and _**b**_ as we can see from their definition in (+) . Hence we conclude that our
reduction runs in time that is polynomial in the size of the circuit _C_ _l_ .


The next thing to observe is that, according to Lemma 6.6, the function _f_ _C_ _l_ is both _G_ -Lipschitz
and _L_ -smooth and hence the output of our reduction is a valid input for the promise problem
GDAFixedPoint. So the last step to complete the proof of Theorem 6.1 is to prove that the vector _**x**_ _[⋆]_ of every solution ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) of GDAFixedPoint with input _C_ _f_ _C_ _l_, lies in a cell that is either
panchromatic or violates the rules for proper coloring, in any of these cases we can find a solution to the 2D-BiSperner problem. This proves that our construction reduces 2D-BiSperner to
GDAFixedPoint.

We prove this last statement in Lemma 6.8, but before that we need the following technical
lemma that will be useful to argue about solution on the boundary of _P_ ( _**A**_, _**b**_ ) .


**Lemma 6.7.** _Let_ _C_ _l_ _be an input to the_ 2D _-_ BiSperner _problem, let f_ _C_ _l_ _be the corresponding G-Lipschitz_
_and L-smooth function defined in Definition 6.5, and let_ _P_ ( _**A**_, _**b**_ ) _be the polytope defined by_ (6.1) _. If_
( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _is any solution to the_ GDAFixedPoint _problem with inputs α, G, L,_ _C_ _f_ _C_ _l_ _,_ _**A**_ _, and_ _**b**_ _, defined in_
(+) _then the following statements hold, where recall that_ ∆ = _δ_ /12 _. For i_ _∈{_ 1, 2 _}_ _:_



_⋄_ _If x_ _i_ _[⋆]_ _[∈]_ [(] _[α]_ [, 1] _[ −]_ _[α]_ [)] _[ and x]_ _i_ _[⋆]_ _[∈]_ [(] _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] _[ then]_ ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ )


_⋄_ _If x_ _i_ _[⋆]_ _[≤]_ _[α][ or x]_ _i_ _[⋆]_ _[≤]_ _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥−_ _α._


_⋄_ _If x_ _i_ _[⋆]_ _[≥]_ [1] _[ −]_ _[α][ or x]_ _i_ _[⋆]_ _[≥]_ _[y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≤_ _α._


_The symmetric statements for y_ _i_ _[⋆]_ _[hold. For i]_ _[ ∈{]_ [1, 2] _[}]_ _[:]_

_⋄_ _If y_ _i_ _[⋆]_ _[∈]_ [(] _[α]_ [, 1] _[ −]_ _[α]_ [)] _[ and y]_ _i_ _[⋆]_ _[∈]_ [(] _[x]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ x]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] _[ then]_ ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ )

_⋄_ _If y_ _i_ _[⋆]_ _[≤]_ _[α][ or y]_ _i_ _[⋆]_ _[≤]_ _[x]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≤_ _α._



_≤_ _α._
���


_≤_ _α._
���



_⋄_ _If y_ _i_ _[⋆]_ _[≥]_ [1] _[ −]_ _[α][ or y]_ _i_ _[⋆]_ _[≥]_ _[x]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥−_ _α._


_Proof._ For this proof it is convenient to define ˆ _**x**_ = _**x**_ _[⋆]_ _−∇_ _x_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _K_ ( _**y**_ _[⋆]_ ) = _{_ _**x**_ _|_ ( _**x**_, _**y**_ _[⋆]_ ) _∈_
_P_ ( _**A**_, _**b**_ )) _}_, and _**z**_ = Π _K_ ( _**y**_ _⋆_ ) ˆ _**x**_ .
We first consider the first statement, so for the sake of contradiction let’s assume that _x_ _i_ _[⋆]_ _[∈]_



_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )
( _α_, 1 _−_ _α_ ), that _x_ _i_ _[⋆]_ _[∈]_ [(] _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] [, and that] ��� _∂x_ _i_



_>_ _α_ . Due to the defini���



tion of _P_ ( _**A**_, _**b**_ ) in (6.1) the set _K_ ( _**y**_ _[⋆]_ ) is an axes aligned box of **R** [2] and hence the projection
of any vector _**x**_ onto _K_ ( _**y**_ _[⋆]_ ) can be implemented independently for every coordinate _x_ _i_ of _**x**_ .


28


Therefore if it happens that ˆ _x_ _i_ _∈_ ( 0, 1 ) _∩_ ( _y_ _i_ _[⋆]_ _[−]_ [∆][,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] [)] [, then it holds that ˆ] _[x]_ _[i]_ [ =] _**[ z]**_ _[i]_ [.] Now
from the definition of ˆ _x_ _i_ and _z_ _i_, and the fact that _K_ ( _**y**_ _[⋆]_ ) is an axes aligned box, we get that



_|_ _x_ _i_ _[⋆]_ _[−]_ _[z]_ _[i]_ _[|]_ [ =] _[ |]_ _[x]_ _i_ _[⋆]_ _[−]_ _[x]_ [ˆ] _[i]_ _[|]_ [ =] ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ )



_>_ _α_ which contradicts the fact that ( _**x**_ _⋆_, _**y**_ _⋆_ ) is a solution to the
���



problem GDAFixedPoint. On the other hand if ˆ _x_ _i_ _̸∈_ ( _y_ _i_ _[⋆]_ _[−]_ [∆][,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] [)] _[ ∩]_ [(] [0, 1] [)] [ then] _[ z]_ _[i]_ [ has to be on]
the boundary of _K_ ( _**y**_ _[⋆]_ ) and hence _z_ _i_ has to be equal to either 0, or 1, or _y_ _i_ _[⋆]_ _[−]_ [∆][, or] _[ y]_ _i_ _[⋆]_ [+] [ ∆][. In any]
of these cases since we assumed that _x_ _i_ _[⋆]_ _[∈]_ [(] _[α]_ [, 1] _[ −]_ _[α]_ [)] [ and that] _[ x]_ _i_ _[⋆]_ _[∈]_ [(] _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] [ we]
conclude that _|_ _x_ _i_ _[⋆]_ _[−]_ _[z]_ _[i]_ _[|][ >]_ _[ α]_ [ and hence we get again a contradiction with the fact that] [ (] _**[x]**_ _[⋆]_ [,] _**[ y]**_ _[⋆]_ [)] [ is a]



solution to the problem GDAFixedPoint. Hence we have that ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ )



_≤_ _α_ .
���



_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )
For the second case, we assume for the sake of contradiction that _x_ _i_ _[⋆]_ _[≤]_ _[α]_ [ and] _∂x_ _i_ _< −_ _α_ .

These imply that ˆ _x_ _i_ _>_ _x_ _i_ _[⋆]_ [+] _[ α]_ [ and that] _[ z]_ _[i]_ [ =] [ min] [(] _[y]_ _i_ _[⋆]_ [+] [ ∆][, ˆ] _[x]_ _[i]_ [, 1] [)] _[ >]_ [ min] [(] [∆][, ˆ] _[x]_ _[i]_ [, 1] [)] _[ ≥]_ [min] [(] [3] _[α]_ [,] _[ x]_ _i_ _[⋆]_ [+] _[ α]_ [)] [.]
As a result, _|_ _x_ _i_ _[⋆]_ _[−]_ _[z]_ _[i]_ _[|]_ [ =] _[ z]_ _[i]_ _[ −]_ _[x]_ _i_ _[⋆]_ _[>]_ [ min] [(] [3] _[α]_ [, ˆ] _[x]_ _[i]_ [ +] _[ α]_ [)] _[ −]_ _[x]_ _i_ _[⋆]_ [which is greater than] _[ α]_ [. The latter is a]
contradiction with the assumption that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is a solution to the GDAFixedPoint problem.
Also if we assume that _x_ _i_ _[⋆]_ _[≤]_ _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [ using the same reasoning we get that] _[ z]_ _[i]_ [ =] [ min] [(] [ ˆ] _[x]_ _[i]_ [,] _[ y]_ _i_ _[⋆]_ [+]
∆ _−_ _α_, 1 ) . From this we can again prove that _|_ _x_ _i_ _[⋆]_ _[−]_ _[z]_ _[i]_ _[|][ >]_ _[ α]_ [ which contradicts the fact that] [ (] _**[x]**_ _[⋆]_ [,] _**[ y]**_ _[⋆]_ [)]
is a solution to GDAFixedPoint.

The third case can be proved using the same arguments as the second case. Then using the
corresponding arguments we can prove the corresponding statements for the _y_ variables.


We are now ready to prove that solutions of GDAFixedPoint can only occur in cells that are
either panchromatic or violate the boundary conditions of a proper coloring. For convenience in
the rest of this section we define _R_ ( _**x**_ ) to be the cell of the 2 _[n]_ _×_ 2 _[n]_ grid that contains _**x**_ .



_R_ ( _**x**_ ) = � 2 _[n]_ _−_ _i_ 1 [,] 2 _[i]_ _[n]_ [ +] _−_ [ 1] 1



_j_
_×_ _[j]_ [ +] [ 1]
� � 2 _[n]_ _−_ 1 [,] 2 _[n]_ _−_ 1



, (6.2)
�



2 _[i]_ ~~_[n]_~~ [+] _−_ [1] 1 � and _x_ 2 _∈_ 2 ~~_[n]_~~ _−_ _j_ 1 [,] 2 _[j]_ ~~_[n]_~~ [+] _−_ [1]
�



for _i_, _j_ such that _x_ 1 _∈_ � 2 ~~_[n]_~~ _−_ _i_ 1 [,] 2 _[i]_ ~~_[n]_~~ [+] _−_ [1]



2 _[j]_ ~~_[n]_~~ [+] _−_ [1] 1 if there are multiple _i_, _j_ that satisfy the
�



above condition then we choose _R_ ( _**x**_ ) to be the cell that corresponds to the _i_, _j_ such that the pair
( _i_, _j_ ) it the lexicographically first such that _i_, _j_ satisfy the above condition. We also define the
corners _R_ _c_ ( _**x**_ ) of _R_ ( _**x**_ ) as


_R_ _c_ ( _**x**_ ) = _{_ ( _i_, _j_ ), ( _i_, _j_ + 1 ), ( _i_ + 1, _j_ ), ( _i_ + 1 ), ( _j_ + 1 ) _}_ (6.3)



_j_
2 _[i]_ ~~_[n]_~~ [+] _−_ [1] 1 � _×_ 2 ~~_[n]_~~ _−_ 1 [,] 2 _[j]_ ~~_[n]_~~ [+] _−_ [1]
�



where _R_ ( _**x**_ ) = � 2 ~~_[n]_~~ _−_ _i_ 1 [,] 2 _[i]_ ~~_[n]_~~ [+] _−_ [1]



2 _[j]_ ~~_[n]_~~ [+] _−_ [1] 1 .
�



**Lemma 6.8.** _Let_ _C_ _l_ _be an input to the_ 2D _-_ BiSperner _problem, let f_ _C_ _l_ _be the corresponding G-Lipschitz_
_and L-smooth function defined in Definition 6.5, and let_ _P_ ( _**A**_, _**b**_ ) _be the polytope defined by_ (6.1) _. If_
( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _is any solution to the_ GDAFixedPoint _problem with inputs α, G, L,_ _C_ _f_ _C_ _l_ _,_ _**A**_ _, and_ _**b**_ _defined in_
(+) _then none of the following statements hold for the cell R_ ( _**x**_ _[⋆]_ ) _._


_1. x_ 1 _[⋆]_ _[≥]_ [1/] [(] [2] _[n]_ _[ −]_ [1] [)] _[ and, for all]_ _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ [1] [(] _**[v]**_ [) =] _[ −]_ [1] _[.]_


_2. x_ 1 _[⋆]_ _[≤]_ [(] [2] _[n]_ _[ −]_ [2] [)] [/] [(] [2] _[n]_ _[ −]_ [1] [)] _[ and, for all]_ _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ [1] [(] _**[v]**_ [) = +] [1] _[.]_


_3. x_ 2 _[⋆]_ _[≥]_ [1/] [(] [2] _[n]_ _[ −]_ [1] [)] _[ and, for all]_ _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ [2] [(] _**[v]**_ [) =] _[ −]_ [1] _[.]_


_4. x_ 2 _[⋆]_ _[≤]_ [(] [2] _[n]_ _[ −]_ [2] [)] [/] [(] [2] _[n]_ _[ −]_ [1] [)] _[ and, for all]_ _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ [2] [(] _**[v]**_ [) = +] [1] _[.]_


29


_Proof._ We prove that there is no solution ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) of GDAFixedPoint that satisfies the statement
1. and the fact that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) cannot satisfy the other statements follows similarly. It is convenient
for us to define ˆ _**x**_ = _**x**_ _[⋆]_ _−∇_ _x_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _K_ ( _**y**_ _[⋆]_ ) = _{_ _**x**_ _|_ ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ )) _}_, _**z**_ = Π _K_ ( _**y**_ _⋆_ ) ˆ _**x**_, and

ˆ
_**y**_ = _**y**_ _[⋆]_ + _∇_ _y_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _K_ ( _**x**_ _[⋆]_ ) = _{_ _**y**_ _|_ ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ )) _}_, _**w**_ = Π _K_ ( _**x**_ _⋆_ ) ˆ _**y**_ .
For the sake of contradiction we assume that there exists a solution of ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) such that
_x_ 1 _[⋆]_ _[≥]_ [1/] [(] [2] _[n]_ _[ −]_ [1] [)] [ and for all] _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] [ it holds that] _[ C]_ _l_ [1] [(] _**[v]**_ [) =] _[ −]_ [1. Using the fact that the first color]



of all the corners of _R_ ( _**x**_ _[⋆]_ ) is 1 _[−]_, we will prove that (1) _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _x_ _[⋆]_, _**y**_ _[⋆]_ )



= _−_ 1.
_∂y_ 1



_∂_ ( _**x**_ _x_ _[⋆]_ 1, _**y**_ _[⋆]_ ) _≥_ 1/2, and (2) _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _y_ _[⋆]_ 1, _**y**_ _[⋆]_ )



_i_ _j_
Let _R_ ( _**x**_ _[⋆]_ ) = � 2 ~~_[n]_~~ _−_ 1 [,] 2 _[i]_ ~~_[n]_~~ [+] _−_ [1] 1 � _×_ 2 ~~_[n]_~~ _−_ 1 [,] 2 _[j]_ ~~_[n]_~~ [+] _−_ [1] 1, then since all the corners _**v**_ _∈_ _R_ _c_ ( _**x**_ _[⋆]_ ) have _C_ _l_ [1] [(] _**[v]**_ [) =]

� �

_−_ 1, from the Definition 6.5 we have that



_j_
2 _[i]_ ~~_[n]_~~ [+] _−_ [1] 1 � _×_ 2 ~~_[n]_~~ _−_ 1 [,] 2 _[j]_ ~~_[n]_~~ [+] _−_ [1] 1
�



Let _R_ ( _**x**_ _[⋆]_ ) = � 2 ~~_[n]_~~ _−_ _i_ 1 [,] 2 _[i]_ ~~_[n]_~~ [+] _−_ [1]



_x_ _[C]_

_·_ _S_ 1 2 _[−]_ _[x]_ 2 _[⋆]_

_δ_
�



�



_f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) = ( _x_ 1 _[⋆]_ _[−]_ _[y]_ 1 _[⋆]_ [)] _[ −]_ [(] _[x]_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ _[ S]_ [1]



_x_ _[C]_
1 _[−]_ _[x]_ 1 _[⋆]_

_δ_
� �




_· C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [)]



_B_

_·_ _S_ 1 _x_ 2 _[−]_ _[x]_ 2 _[⋆]_
� � _δ_



_−_ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ _[ S]_ [1]


_−_ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ _[ S]_ [1]


_−_ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ _[ S]_ [1]



_D_
_x_
1 _[−]_ _[x]_ 1 _[⋆]_
� _δ_


_x_ _[⋆]_
1 _[−]_ _[x]_ 1 _[A]_

_δ_
�

_x_ 1 _⋆_ _[−]_ _[x]_ 1 _[B]_
� _δ_




_·_ _S_ 1
�



�



_x_ 2 _⋆_ _[−]_ _[x]_ 2 _[D]_
� _δ_

_x_ 2 _⋆_ _[−]_ _[x]_ 2 _[A]_
� _δ_




_·_ _S_ 1




_· C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [ +] [ 1] [)]
�


_· C_ _l_ [2] [(] _[i]_ [ +] [ 1,] _[ j]_ [ +] [ 1] [)]
�




_· C_ _l_ [2] [(] _[i]_ [ +] [ 1,] _[ j]_ [)]
�



where ( _x_ 1 _[A]_ [,] _[ x]_ 2 _[A]_ [) = (] _[i]_ [/] [(] [2] _[n]_ _[ −]_ [1] [)] [,] _[ j]_ [/] [(] [2] _[n]_ _[ −]_ [1] [))] [,] [ (] _[x]_ 1 _[B]_ [,] _[ x]_ 2 _[B]_ [) = (] _[i]_ [/] [(] [2] _[n]_ _[ −]_ [1] [)] [,] [ (] _[j]_ [ +] [ 1] [)] [/] [(] [2] _[n]_ _[ −]_ [1] [))] [,] [ (] _[x]_ 1 _[C]_ [,] _[ x]_ 2 _[C]_ [) =]
(( _i_ + 1 ) / ( 2 _[n]_ _−_ 1 ), ( _j_ + 1 ) / ( 2 _[n]_ _−_ 1 )), and ( _x_ 1 _[D]_ [,] _[ x]_ 2 _[D]_ [) = ((] _[i]_ [ +] [ 1] [)] [/] [(] [2] _[n]_ _[ −]_ [1] [)] [,] _[ j]_ [/] [(] [2] _[n]_ _[ −]_ [1] [))] [. If we differen-]

tiate this with respect to _y_ 1 we immediately get that _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _y_ _[⋆]_ 1, _**y**_ _[⋆]_ ) = _−_ 1. On the other hand if we

differentiate with respect to _x_ 1 we get



�




_· C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [)]
�



_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )



_δ_ _[·]_ _[ S]_ 1 _[′]_




_·_ _S_ 1



_**x**_, _**y**_ )

= 1 + ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ [ 1]
_∂x_ 1 _δ_



1 _[−]_ _[x]_ 1 _[A]_
1 _−_ _[x]_ _[⋆]_
_δ_

�



2 _[−]_ _[x]_ 2 _[A]_
1 _−_ _[x]_ _[⋆]_
� _δ_




_·_ _S_ 1

�



_x_ 2 _⋆_ _[−]_ _[x]_ 2 _[A]_
� _δ_




_· C_ _l_ [2] [(] _[i]_ [,] _[ j]_ [ +] [ 1] [)]
�



+ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ [ 1] 1

_δ_ _[·]_ _[ S]_ _[′]_



�



1 _[−]_ _[x]_ 1 _[A]_
1 _−_ _[x]_ _[⋆]_
_δ_



�

�



_−_ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ [ 1] 1

_δ_ _[·]_ _[ S]_ _[′]_



_x_ 2 _⋆_ _[−]_ _[x]_ 2 _[A]_
� _δ_




_· C_ _l_ [2] [(] _[i]_ [ +] [ 1,] _[ j]_ [ +] [ 1] [)]
�




_· C_ _l_ [2] [(] _[i]_ [ +] [ 1,] _[ j]_ [)]
�



_−_ ( _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ [)] _[ ·]_ [1] 1

_δ_ _[·]_ _[ S]_ _[′]_



_x_ _[⋆]_
1 _[−]_ _[x]_ 1 _[A]_

_δ_
�


_x_ _[⋆]_
1 _[−]_ _[x]_ 1 _[A]_

_δ_
�




_·_ _S_ 1


_·_ _S_ 1



2 _[−]_ _[x]_ 2 _[A]_
1 _−_ _[x]_ _[⋆]_
� _δ_



_≥_ 1 _−_ 4 _|_ _x_ 2 _[⋆]_ _[−]_ _[y]_ 2 _[⋆]_ _[| ·]_ [3]

2 _δ_



_≥_ 1 _−_ 6 _·_ [∆] _δ_ _[≥]_ [1/2] (6.4)



where the last inequality follows from the fact that _|_ _S_ 1 _[′]_ [(] _[·]_ [)] _[| ≤]_ [3/2 and the fact that, due to the]
constraints that define the polytope _P_ ( _**A**_, _**b**_ ), it holds that _|_ _x_ 2 _−_ _y_ 2 _| ≤_ ∆.
Hence we have established that if _x_ 1 _[⋆]_ _[≥]_ [1/] [(] [2] _[n]_ _[ −]_ [1] [)] [ and for all] _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] [ it holds that]



_C_ _l_ [1] [(] _**[v]**_ [) =] _[ −]_ [1 then it holds that that (1)] _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _x_ _[⋆]_, _**y**_ _[⋆]_ )



= _−_
_∂y_ 1 1. Now it is easy to



_∂_ ( _**x**_ _x_ _[⋆]_ 1, _**y**_ _[⋆]_ ) _≥_ 1/2, and (2) _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _y_ _[⋆]_ 1, _**y**_ _[⋆]_ )



see that the only way to satisfy both _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _x_ _[⋆]_ 1, _**y**_ _[⋆]_ ) _≥_ 1/2 and _|_ _z_ 1 _−_ _x_ 1 _[⋆]_ _[| ≤]_ _[α]_ [ is that either] _[ x]_ 1 _[⋆]_ _[≤]_ _[α]_ [ or]


30


_x_ _[⋆]_
1 _[≤]_ _[y]_ 1 _[⋆]_ _[−]_ [∆] [+] _[ α]_ [. The first case is excluded by the assumption in the first statement of our lemma]
and our choice of _α_ = ∆/3 = 1/ ( 36 _·_ ( 2 _[n]_ _−_ 1 )) thus it holds that _x_ 1 _[⋆]_ _[≤]_ _[y]_ 1 _[⋆]_ _[−]_ [∆] [+] _[ α]_ [. But then we]

can use the case 3 for the _y_ variables of Lemma 6.7 and we get that _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _y_ _[⋆]_ 1, _**y**_ _[⋆]_ ) _≥−_ _α_, which cannot

_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) = _−_
be true since we proved that _∂y_ 1 1. Therefore we have a contradiction and the first

statement of the lemma holds. Using the same reasoning we prove the rest of the statements.


_Remark_ 6.9 _._ The computations presented in (6.4) is the precise point where an attempt to prove
the hardness of minimization problems would fail. In particular, if our goal was to construct a
hard minimization instance then the function _f_ _C_ _l_ would need to have the terms _x_ _i_ + _y_ _i_ instead of
_x_ _i_ _−_ _y_ _i_ so that the fixed points of gradient descent coincide with approximate local minimum of
_f_ _C_ _l_ . In that case we cannot lower bound the gradient of (6.4) below from 1/2 because the term
_|_ _x_ 2 _[⋆]_ [+] _[ y]_ 2 _[⋆]_ _[|]_ [ will be the dominant one and hence the sign of the derivative can change depending on]
the value _|_ _x_ 2 _[⋆]_ [+] _[ y]_ 2 _[⋆]_ _[|]_ [. For a more intuitive explanation of the reason why we cannot prove hardness]
1.2.
of minimization problems we refer to the Introduction, at Section


We have now all the ingredients to prove Theorem 6.1.


_Proof of Theorem 6.1._ Let ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) be a solution to the GDAFixedPoint instance that we construct
based on the instance _C_ _l_ of 2D-BiSperner. Let also _R_ ( _**x**_ _[⋆]_ ) be the cell that contains _**x**_ _[⋆]_ . If the
corners _R_ _c_ ( _**x**_ _[⋆]_ ) contain all the colors 1 _[−]_, 1 [+], 2 _[−]_, 2 [+] then we have a solution to the 2D-BiSperner
instance and the Theorem 6.1 follows. Otherwise there is at least one color missing from _R_ _c_ ( _**x**_ _[⋆]_ ),
let’s assume without loss of generality that one of the missing colors is 1 _[−]_, hence for every
_**v**_ _∈_ _R_ _c_ ( _**x**_ _[⋆]_ ) it holds that _C_ _l_ ( _**v**_ ) = + 1. Now from Lemma 6.8 the only way for this to happen
is that _x_ 1 _[⋆]_ _[>]_ [ (] [2] _[n]_ _[ −]_ [2] [)] [/] [(] [2] _[n]_ _[ −]_ [1] [)] [ which implies that in] _[ R]_ _[c]_ [(] _**[x]**_ _[⋆]_ [)] [ there is at least one corner of the]
form _**v**_ = ( 2 _[n]_ _−_ 1, _j_ ) . But we have assumed that _C_ _l_ ( _**v**_ ) = + 1, hence _**v**_ is a violation of the
proper coloring rules and hence a solution to the 2D-BiSperner instance. We can prove the
corresponding statement if any other color from 1 [+], 2 _[−]_, 2 [+] is missing. Finally, we observe that
the function that we define has range [ _−_ 2, 2 ] and hence the Theorem 6.1 follows.

#### **7 Hardness of Local Min-Max Equilibrium – High-Dimensions**


Although the results of Section 6 are quite indicative about the computational complexity of
GDAFixedPoint and LR-LocalMinMax, we have not yet excluded the possibility of the existence of algorithms running in poly ( _d_, _G_, _L_, 1/ _ε_ ) time. In this section we present a, significantly
more challenging, high dimensional version of the reduction that we presented in Section 6.
The advantage of this reduction is that it rules out the existence even of algorithms running in
poly ( _d_, _G_, _L_, 1/ _ε_ ) steps unless FP = PPAD, for details see Theorem 4.4. An easy consequence
of our result is an unconditional lower bound on the _black-box model_ that states that the running
time of any algorithm for LR-LocalMinMax that has only oracle access to _f_ and _∇_ _f_ has to be
exponential in _d_, or _G_, or _L_, or 1/ _ε_, for details we refer to the Theorem 4.5 and Section 9.
The main reduction that we use to prove Theorem 4.4 is from the high dimensional generalization of the problem 2D-BiSperner, which we call HighD-BiSperner, to GDAFixedPoint.
Our reduction in this section resembles some of the ideas of the reductions of Section 6 but it has

many additional significant technical difficulties. The main difficulty that we face is how to define a function on a _d_ -dimensional simplex that is: (1) both Lipschitz and smooth, (2) interpolated
between some fixed functions at the _d_ + 1 corners of the simplex, and (3) remains Lipschitz and


31


smooth even if we glue together different simplices. It is well understood from previous works
how to construct such a function if we are interested only in achieving the Lipschitz continuity.
Surprisingly adding the smoothness requirement makes the problem very different and significantly more difficult. Our proof overcomes this technical difficulty by introducing a novel but
very technically involved way to define interpolation within a simplex of some fixed functions
on the corners of the simplex. We believe that our novel interpolation technique is of independent interest and we hope that it will be at the heart of other computational hardness results of
optimization problems in continuous optimization.


**7.1** **The High Dimensional Bi-Sperner Problem**


We start by presenting the HighD-BiSperner problem. The HighD-BiSperner is a straightforward _d_ -dimensional generalization of the 2D-BiSperner that we defined in the Section 6. Assume
that we have a _d_ -dimensional grid _N_ _× · · ·_ ( _d_ times ) _· · · ×_ _N_ . We assign to every vertex of this
grid a sequence of _d_ colors and we say that a coloring is _proper_ if the following rules are satisfied.


1. The _i_ th color of every vertex is either the color _i_ [+] or the color _i_ _[−]_ .


2. All the vertices whose _i_ th coordinate is 0, i.e. they are at the lower boundary of the _i_ th
direction, should have the _i_ th color equal to _i_ [+] .


3. All the vertices whose _i_ th coordinate is 1, i.e. they are at the higher boundary of the _i_ th
direction, should have the _i_ th color equal to _i_ _[−]_ .


Using proof ideas similar to the proof of the original Sperner’s Lemma it is not hard to prove
via a combinatorial argument that in every proper coloring of a _d_ -dimensional grid, there exists
a cubelet of the grid where all the 2 _·_ _d_ colors _{_ 1 _[−]_, 1 [+], . . ., _d_ _[−]_, _d_ [+] _}_ appear in some of its vertices,
we call such a cubelet _panchromatic_ . In the HighD-BiSperner problem we are asked to find such
a cubelet, or a violation of the rules of proper coloring. As in Section 6.1 we do not present this
combinatorial argument in this paper since the totality of the HighD-BiSperner problem will
follow from our reduction from HighD-BiSperner to GDAFixedPoint and our proofs in Section
5 that establish the totality of GDAFixedPoint.


As in the case of 2D-BiSperner, in order to formally define the computational problem
HighD-BiSperner we need to define the coloring of the _d_ -dimensional grid _N_ _× · · · ×_ _N_ in a
succinct way. The fundamental difference compared to the definition of 2D-BiSperner is that for
the HighD-BiSperner we assume that _N_ is only _polynomially large_ . This difference will enable
us to exclude algorithms for GDAFixedPoint that run in time poly ( _d_, 1/ _α_, _G_, _L_ ) . The input to
HighD-BiSperner is a coloring via a binary circuit _C_ _l_ that takes as input the coordinates of a
vertex of the grid and outputs the sequence of colors that are used to color this vertex. Each one
of _d_ coordinates is given via the binary representation of a number in [ _N_ ] _−_ 1. Setting _N_ = 2 _[ℓ]_,
where here _ℓ_ is a logarithmically in _d_ small number, we have that the representation of each
coordinate is a member of _{_ 0, 1 _}_ _[ℓ]_ . In the rest of the section we abuse the notation and we use
a coordinate _i_ _∈{_ 0, 1 _}_ _[ℓ]_ both as a binary string and as a number in �2 _[ℓ]_ [�] _−_ 1 and which of the
two we use it is clear from the context. The output of _C_ _l_ should be a sequence of _d_ colors, where
the _i_ th member of this sequence is one of the colors _{_ _i_ _[−]_, _i_ [+] _}_ . We represent this sequence as a
member of _{−_ 1, + 1 _}_ _[d]_, where the _i_ th coordinate refers to the choice of _i_ _[−]_ or _i_ [+] .
In the definition of the computational problem HighD-BiSperner the input is a circuit _C_ _l_, as
we described above. As we discussed above in the HighD-BiSperner problem we are asking for


32


a panchromatic cubelet of the grid. One issue with this high-dimensional setting is that in order
to check whether a cubelet is panchromatic or not we have to query all the 2 _[d]_ corners of this
cubelet which makes the verification problem inefficient and hence a containment to the PPAD
class cannot be proved. For this reason as a solution to the HighD-BiSperner we ask not just
for a cubelet but for 2 _·_ _d_ vertices _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[d]_ [)] _**u**_ [(] [1] [)], . . ., _**u**_ [(] _[d]_ [)], not necessarily different, such that
they all belong to the same cubelet and the _i_ th output of _C_ _l_ with input _**v**_ _i_ is _−_ 1, i.e. corresponds
to the color _i_ _[−]_, whereas the _i_ th output of _C_ _l_ with input _**u**_ _i_ is + 1, i.e. corresponds to the color
_i_ [+] . This way we have a certificate of size 2 _·_ _d_ that can be checked in polynomial time. Another
possible solution of HighD-BiSperner is a vertex whose coloring violates the aforementioned
boundary conditions 2. and 3.. of a proper coloring. For notational convenience we refer to the
_i_ th coordinate of _C_ _l_ by _C_ _l_ _[i]_ [. The formal definition of H][igh][D-B][i][S][perner][ is then the following.]


**HighD-BiSperner.**



Input: A boolean circuit _C_ _l_ : _{_ 0, 1 _}_ _[ℓ]_ _× · · · × {_ 0, 1 _}_ _[ℓ]_
� �� �
_d_ times

Output: One of the following:



_→{−_ 1, 1 _}_ _[d]_



1. Two sequences of _d_ vertices _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[d]_ [)] an _**u**_ [(] [1] [)], . . ., _**u**_ [(] _[d]_ [)] with _**v**_ [(] _[i]_ [)], _**u**_ [(] _[i]_ [)] _∈_ � _{_ 0, 1 _}_ _ℓ_ � _d_ such
that _C_ _[i]_
_l_ [(] _**[v]**_ [(] _[i]_ [)] [) =] _[ −]_ [1 and] _[ C]_ _l_ _[i]_ [(] _**[u]**_ [(] _[i]_ [)] [) = +] [1.]

_ℓ_ _d_ _i_
2. A vertex _**v**_ _∈_ � _{_ 0, 1 _}_ � with _v_ _i_ = **0** such that _C_ _l_ [(] _**[v]**_ [) =] _[ −]_ [1.]

3. A vertex _**v**_ _∈_ � _{_ 0, 1 _}_ _ℓ_ � _d_ with _v_ _i_ = **1** such that _C_ _li_ [(] _**[v]**_ [) = +] [1.]


Our first step is to establish the PPAD-hardness of HighD-BiSperner in Theorem 7.2. To prove
this we use a stronger version of the Brouwer problem that is called _γ_ -SuccinctBrouwer and
was first introduced in [Rub16].


_**γ**_ **-SuccinctBrouwer.**

Input: A polynomial-time Turing machine _C_ _M_ evaluating a 1/ _γ_ -Lipschitz continuous vectorvalued function _M_ : [ 0, 1 ] _[d]_ _→_ [ 0, 1 ] _[d]_ .

Output: A point _**x**_ _[⋆]_ _∈_ [ 0, 1 ] _[d]_ such that _∥_ _M_ ( _**x**_ _[⋆]_ ) _−_ _**x**_ _[⋆]_ _∥_ 2 _≤_ _γ_ .


**Theorem 7.1** ([Rub16]) **.** _γ_ -SuccinctBrouwer _is_ PPAD _-complete for any fixed constant γ_ _>_ 0 _._


**Theorem 7.2.** _There is a polynomial time reducton from any instance of the γ_ -SuccinctBrouwer _prob-_
_lem to an instance of_ HighD _-_ BiSperner _with N_ = Θ ( _d_ / _γ_ [2] ) _._


_Proof._ Consider the function _g_ ( _**x**_ ) = _M_ ( _**x**_ ) _−_ _**x**_ . Since _M_ is 1/ _γ_ -Lipschitz, _g_ : [ 0, 1 ] _[d]_ _→_ [ _−_ 1, 1 ] _[d]_ is
also ( 1 + 1/ _γ_ ) -Lipschitz. Additionally _g_ can be easily computed via a polynomial-time Turing
machine _C_ _g_ that uses _C_ _M_ as a subroutine. We construct the coloring sequences of every vertex of a
_d_ -dimensional grid with _N_ = Θ ( _d_ / _γ_ [2] ) points in every direction using _g_ . Let _g_ _η_ : [ 0, 1 ] [2] _→_ [ _−_ 1, 1 ] [2]

be the function that the Turing Machine _C_ _g_ evaluate when the requested accuracy is _η_ _>_ 0.
For each vertex _**v**_ = ( _v_ 1, . . ., _v_ _n_ ) _∈_ ([ _N_ ] _−_ 1 ) _[d]_ of the _d_ -dimensional grid its coloring sequence
_C_ _l_ ( _**v**_ ) _∈{−_ 1, 1 _}_ _[d]_ is constructed as follows: For each coordinate _j_ = 1, . . ., _d_,



1 _v_ _j_ = 0

_−_ 1 _v_ _j_ = 2 _[n]_ _−_ 1

sign � _g_ _j_ � _Nv_ _−_ 1 1 [, . . .,] _Nv_ _−_ _n_ 1 �� otherwise


33



_C_ _[j]_
_l_ [(] _**[v]**_ [) =]














,


where sign : [ _−_ 1, 1 ] _�→{−_ 1, 1 _}_ is the sign function and _g_ _η_, _j_ ( _·_ ) is the _j_ -th coordinate of _g_ _η_ .

Observe that since _M_ : [ 0, 1 ] _[d]_ _→_ [ 0, 1 ] _[d]_, for any vertex _**v**_ with _v_ _j_ = 0 it holds that _C_ _l_ _[j]_ [(] _**[v]**_ [) = +] [1 and]

respectively for any vertex _**v**_ with _v_ _j_ = _N_ _−_ 1 it holds that _C_ _l_ _[j]_ [(] _**[v]**_ [) =] _[ −]_ [1 due to the fact that the]
value of _M_ is always in [ 0, 1 ] _[d]_ and hence there are no vertices in the grid satisfying the possible
outputs 2. or 3. of the HighD-BiSperner problem. Thus the only possible solution of the above
HighD-BiSperner instance is a sequence of 2 _d_ vertices _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[d]_ [)], _**u**_ [(] [1] [)], . . ., _**u**_ [(] _[d]_ [)] on the same
cubelet that certify that the corresponding cubelet is panchromatic, as per possible output 1. of
the HighD-BiSperner problem. We next prove that any vertex _**v**_ of that cubelet it holds that


_**v**_ 2 _√_ _d_

_g_ _j_ _≤_ for all coordinates _j_ = 1, . . ., _d_ .
���� � _N_ _−_ 1 ����� _γN_


Let _**v**_ be any vertex on the same cubelet with the output vertices _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[d]_ [)], _**u**_ [(] [1] [)], . . ., _**u**_ [(] _[d]_ [)] .
From the guarantees of colors of the sequences _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[d]_ [)], _**u**_ [(] [1] [)], . . ., _**u**_ [(] _[d]_ [)] we have that either
_C_ _[j]_
_l_ [(] _**[v]**_ [)] _[ · C]_ _l_ _[j]_ [(] _**[v]**_ [(] _[j]_ [)] [) =] _[ −]_ [1 or] _[ C]_ _l_ _[j]_ [(] _**[v]**_ [)] _[ · C]_ _l_ _[j]_ [(] _**[u]**_ [(] _[j]_ [)] [) =] _[ −]_ [1, let] ~~_**[ v]**_~~ [(] _[j]_ [)] [ be the vertex] _**[ v]**_ [(] _[j]_ [)] [ or] _**[ u]**_ [(] _[j]_ [)] [ depending on which]

one the _j_ th color has product equal to _−_ 1 with _C_ _l_ _[j]_ [(] _**[v]**_ [)] [. Now let] _[ η]_ [ =] [2] _γ_ _√_ _Nd_ [if] _[ g]_ _[j]_ � _N_ _**v**_ _−_ 1 � _∈_ [ _−_ _η_, _η_ ]

_**v**_ _−_
then the wanted inequality follows. On the other hand if _g_ _j_ � _N_ _−_ 1 � _∈_ [ _η_, _η_ ] then using the fact
that �� _g_ � _N_ _**v**_ _−_ 1 � _−_ _g_ _η_ � _N_ _**v**_ _−_ 1 ��� ∞ _[≤]_ _[η]_ [ and that from the definition of the colors we have that either]



_**v**_
� _N_ _−_ 1



2 _√_
_≤_
����� _γN_



_√_ _d_

for all coordinates _j_ = 1, . . ., _d_ .
_γN_



_g_ _η_, _j_ � _N_ _**v**_ _−_ 1 � _≥_ 0, _g_ _η_, _j_ �



~~_**v**_~~ [(] _[j]_ [)] _**v**_ _**v**_ ˆ [(] _[j]_ [)] _**v**_
_N_ _−_ 1 � _<_ 0 or _g_ _η_, _j_ � _N_ _−_ 1 � _<_ 0, _g_ _η_, _j_ � _N_ _−_ 1 � _≥_ 0 we conclude that _g_ _j_ � _N_ _−_ 1 � _≥_ 0,



_g_ _j_ �



~~_**v**_~~ [(] _[j]_ [)] _**v**_ _**v**_ ˆ [(] _[j]_ [)]
_N_ _−_ 1 � _<_ 0 or _g_ _j_ � _N_ _−_ 1 � _<_ 0, _g_ _j_ � _N_ _−_ 1 � _≥_ 0 and thus,



_√_ _d_
_≤_ [2]

_γN_

����� 2



_≤_ 1 + [1]
� _γ_
������




_·_
�



_g_ _j_
����



_g_ _j_
�����



_**v**_
� _N_ _−_ 1



_≤_
�����



_**v**_
� _N_ _−_ 1



�



~~_**v**_~~ [(] _[j]_ [)]


_N_ _−_ 1



_**v**_ ~~_**v**_~~ [(] _[j]_ [)]
_N_ _−_ 1 _[−]_ _N_ _−_ 1
�����



_−_ _g_ _j_
�



where in the second inequality we have used the ( 1 + 1/ _γ_ ) -Lipschitzness of _g_ . As a result, the

ˆ ˆ
point ˆ _**v**_ = _**v**_ / ( _N_ _−_ 1 ) _∈_ [ 0, 1 ] _[d]_ satisfies _∥_ _M_ ( _**v**_ ) _−_ _**v**_ _∥_ 2 _≤_ 2 _d_ / ( _γN_ ) and thus for if we pick _N_ =
Θ ( _d_ / _γ_ [2] ) then any vertex _**v**_ of the panchromatic cell is a solution for _γ_ -SuccinctBrouwer.


Now that we have established the PPAD-hardness of HighD-BiSperner we are ready to present
our main result of this section which is a reduction from the problem HighD-BiSperner to the
problem GDAFixedPoint with the additional constraints that the scalars _α_, _G_, _L_ in the input
satisfy 1/ _α_ = poly ( _d_ ), _G_ = poly ( _d_ ), and _L_ = poly ( _d_ ) .


**7.2** **From High Dimensional Bi-Sperner to Fixed Points of Gradient Descent/Ascent**

Given the binary circuit _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→{−_ 1, + 1 _}_ _[d]_ that is an instance of HighD-BiSperner,
we construct a _G_ -Lipschitz and _L_ -smooth function _f_ _C_ _l_ : [ 0, 1 ] _[d]_ _×_ [ 0, 1 ] _[d]_ _→_ **R** . To do so, we divide
the [ 0, 1 ] _[d]_ hypercube into cubelets of length _δ_ = 1/ ( _N_ _−_ 1 ) . The corners of such cubelets have
coordinates that are integer multiples of _δ_ = 1/ ( _N_ _−_ 1 ) and we call them _vertices_ . Each vertex
can be represented by the vector _**v**_ = ( _v_ 1, . . ., _v_ _d_ ) _∈_ ([ _N_ ] _−_ 1 ) _[d]_ and admits a coloring sequence
defined by the boolean circuit _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→{−_ 1, + 1 _}_ _[d]_ . For every _**x**_ _∈_ [ 0, 1 ] _[d]_, we use _R_ ( _**x**_ ) to
denote the cubelet that contains _**x**_, formally



_R_ ( _**x**_ ) = _c_ 1 _[c]_ [1] [ +] [ 1] _× · · · ×_ _c_ _d_ _[c]_ _[d]_ [+] [ 1]
� _N_ _−_ 1 [,] _N_ _−_ 1 � � _N_ _−_ 1 [,] _N_ _−_ 1



�



_c_ 1
where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ such that _**x**_ _∈_ _N_ _−_ 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �




_[c]_ _N_ _[d]_ _−_ [+] 1 [1] and if there are multiple

�



corners _**c**_ that satisfy this condition then we choose _R_ ( _**x**_ ) to be the cell that corresponds to the _**c**_


34


that is lexicographically first among those that satisfy the condition. We also define _R_ _c_ ( _**x**_ ) to be
the set of vertices that are corners of the cublet _R_ ( _**x**_ ), namely


_R_ _c_ ( _**x**_ ) = _{_ _c_ 1, _c_ 1 + 1 _} × · · · × {_ _c_ _d_, _c_ _d_ + 1 _}_




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �



_c_ 1
where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ such that _R_ ( _**x**_ ) = _N_ _−_ 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�




_[c]_ _N_ _[d]_ _−_ [+] 1 [1] Every _**y**_ that belongs

�



to the cubelet _R_ ( _**x**_ ) can be written as a convex combination of the vectors _**v**_ / ( _N_ _−_ 1 ) where
_**v**_ _∈_ _R_ _c_ ( _**x**_ ) . The value of the function _f_ _C_ _l_ ( _**x**_, _**y**_ ) that we construct in this section is determined
by the coloring sequences _C_ _l_ ( _**v**_ ) of the vertices _**v**_ _∈_ _R_ _c_ ( _**x**_ ) . One of the main challenges that we
face though is that the size of _R_ _c_ ( _**x**_ ) is 2 _[d]_ and hence if we want to be able to compute the value
of _f_ _C_ _l_ ( _**x**_, _**y**_ ) efficiently then we have to find a consistent rule to pick a subset of the vertices of
_R_ _c_ ( _**x**_ ) whose coloring sequence we need to define the function value _f_ _C_ _l_ ( _**x**_, _**y**_ ) . Although there
are traditional ways to overcome this difficulty using the _canonical simplicization_ of the cubelet
_R_ ( _**x**_ ), these technique leads only to functions that are continuous and Lipschitz but they are not
enough to guarantee continuity of the gradient and hence the resulting functions are not smooth.


**7.2.1** **Smooth and Efficient Interpolation Coefficients**


The problem of finding a computationally efficient way to define a continuous function as an
interpolation of some fixed function in the corners of a cubelet so that the resulting function
is both Lischitz and smooth is surprisingly difficult to solve. For this reason we introduce in
this section the _smooth and efficient interpolation coefficients (SEIC)_ that as we will see in Section
7.2.2, is the main technical tool to implement such an interpolation. Our novel interpolation
coefficients are of independent interest and we believe that they will serve as a main technical
tool for proving other hardness results in continuous optimization in the future.
In this section we only give a high level description of the smooth and efficient interpolation
coefficients via their properties that we use in Section 7.2.2 to define the function _f_ _C_ _l_ . The actual
construction of the coefficients is very challenging and technical and hence we postpone a detail
exposition for Section 8.


**Definition 7.3** (Smooth and Efficient Interpolation Coefficients) **.** For every _N_ _∈_ **N** we define the
set of _smooth and efficient interpolation coefficients (SEIC)_ as the family of functions, called _coefficients_,
_I_ _d_, _N_ = P _**v**_ : [ 0, 1 ] _[d]_ _→_ **R** _|_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ [�] with the following properties.
�


(A) For all vertices _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_, the coefficient P _**v**_ ( _**x**_ ) is a twice-differentiable function and
satisfies









_∂_ P _**v**_ ( _**x**_ )
��� _∂x_ _i_

_∂_ 2 P _**v**_ ( _**x**_ )
��� _∂x_ _i_ _∂x_ _ℓ_



12
_≤_ Θ ( _d_ / _δ_ ) .
���



24 2
_≤_ Θ ( _d_ / _δ_ ) .
���



(B) For all _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_, it holds that P _**v**_ ( _**x**_ ) _≥_ 0 and ∑ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _d_ P _**v**_ ( _**x**_ ) = ∑ _**v**_ _∈_ _R_ _c_ ( _**x**_ ) P _**v**_ ( _**x**_ ) = 1.


(C) For all _**x**_ _∈_ [ 0, 1 ] _[d]_, it holds that all but _d_ + 1 of the coefficients P _**v**_ _∈I_ _d_, _N_ satisfy P _**v**_ ( _**x**_ ) = 0,
_∇_ P _**v**_ ( _**x**_ ) = 0 and _∇_ [2] P _**v**_ ( _**x**_ ) = 0. We denote this set of _d_ + 1 vertices by _R_ + ( _**x**_ ) . Furthermore,
it holds that _R_ + ( _**x**_ ) _⊆_ _R_ _c_ ( _**x**_ ) and given _**x**_ we can compute the set _R_ + ( _**x**_ ) it time poly ( _d_ ) .


(D) For all _**x**_ _∈_ [ 0, 1 ] _[d]_, if _x_ _i_ _≤_ 1/ ( _N_ _−_ 1 ) for some _i_ _∈_ [ _d_ ] then there exists _**v**_ _∈_ _R_ + ( _**x**_ ) such that
_v_ _i_ = 0. Respectively, if _x_ _i_ _≥_ 1 _−_ 1/ ( _N_ _−_ 1 ) then there exists _**v**_ _∈_ _R_ + ( _**x**_ ) such that _v_ _i_ = 1.


35


An intuitive explanation of the properties of the SEIC coefficients is the following


**(A) –** The coefficients P _**v**_ are both Lipschitz and smooth with Lipschitzness and smoothness
parameters that depends polynomially in _d_ and _N_ = 1/ _δ_ + 1.


**(B) –** The coefficients P _**v**_ ( _**x**_ ) define a convex combination of the vertices _R_ _c_ ( _**x**_ ) .


**(C) –** For every _**x**_ _∈_ [ 0, 1 ] _[d]_, out of the _N_ _[d]_ coefficients P _**v**_ only _d_ + 1 have non-zero value, or
non-zero gradient or non-zero Hessian when evaluated at the point _**x**_ . Moreover, given
_**x**_ _∈_ [ 0, 1 ] _[d]_ we can identify these _d_ + 1 coefficients efficiently.


**(D) –** For every _**x**_ _∈_ [ 0, 1 ] _[d]_ that is in a cubelet that touches the boundary there is at least one of
the vertices in _R_ + ( _**x**_ ) that is on the boundary of the continuous hypercube [ 0, 1 ] _[d]_ .


In Section 10 in the proof of Theorem 10.4 we present a simple application of the existence
of the SEIC coefficients for proving very simple black box oracle lower bounds for the global
minimization problem.
Based on the existence of these coefficients we are now ready to define the function _f_ _C_ _l_ which
is the main construction of our reduction.


**7.2.2** **Definition of a Lipschitz and Smooth Function Based on a BiSperner Instance**


In this section our goal is to formally define the function _f_ _C_ _l_ and prove its Lipschitzness and
smoothness properties in Lemma 7.5.


**Definition 7.4** (Continuous and Smooth Function from Colorings of Bi-Sperner) **.** Given a binary
circuit _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→{−_ 1, 1 _}_ _[d]_, we define the function _f_ _C_ _l_ : [ 0, 1 ] _[d]_ _×_ [ 0, 1 ] _[d]_ _→_ **R** as follows


_d_
### f C l ( x, y ) = ∑ ( x j − y j ) · α j ( x )

_j_ = 1


where _α_ _j_ ( _**x**_ ) = _−_ ∑ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _d_ P _**v**_ ( _**x**_ ) _· C_ _l_ _[j]_ [(] _**[v]**_ [)] [, and][ P] _**[v]**_ [ are the coefficients defined in Definition][ 7.3][.]


We first prove that the function _f_ _C_ _l_ constructed in Definition 7.4 is _G_ -Lipschitz and _L_ -smooth
for some appropriately selected parameters _G_, _L_ that are polynomial in the dimension _d_ and in
the discretization parameter _N_ . We use this property to establish that _f_ _C_ _l_ is a valid input to the
promise problem GDAFixedPoint.


**Lemma 7.5.** _The function f_ _C_ _l_ _of Definition 7.4 is O_ ( _d_ [15] / _δ_ ) _-Lipschitz and O_ ( _d_ [27] / _δ_ [2] ) _-smooth._


_Proof._ If we take the derivative with respect to _x_ _i_ and _y_ _i_ and using property (B) of the coefficients
P _**v**_ we get the following relations,



_d_
### l ( x, y ) = ∑ ( x j − y j ) · [∂α] [j] [(] [x] [)]

_∂x_ _i_ _j_ = 1 _∂x_ _i_



_∂_ _f_ _C_ _l_ ( _**x**_, _**y**_ )




_[j]_ [(] _**[x]**_ [)] _∂_ _f_ _C_ _l_ ( _**x**_, _**y**_ )

+ _α_ _i_ ( _**x**_ ) and
_∂x_ _i_ _∂y_ _i_



_l_ = _−_ _α_ _i_ ( _**x**_ )

_∂y_ _i_



where


### α i ( x ) = − ∑ P v ( x ) and ∂α j ( x ) = − ∑

_**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _∂x_ _i_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_


36



_∂_ P _**v**_ ( _**x**_ )

_· C_ _[j]_
_∂x_ _i_ _l_ [(] _**[v]**_ [)] [.]


Now by the property (C) of Definition 7.3 there are most _d_ + 1 vertices _**v**_ of _R_ _c_ ( _**x**_ ) with the



property _∇_ P _**v**_ ( _**x**_ ) _̸_ = 0. Then if we also use property (A) we get ��� _∂α∂_ _j_ _x_ ( _i_ _**x**_ )



_≤_ Θ ( _d_ 13 / _δ_ ) and using
���



_≤_ Θ ( _d_ ) . Therefore
���



the property (B) we get _|_ _α_ _i_ ( _**x**_ ) _| ≤_ 1. Thus ��� _∂_ _f_ _C_ _∂_ _l_ ( _x_ _**x**_ _i_, _**y**_ )



��� _≤_ Θ ( _d_ 14 / _δ_ ) and ��� _∂_ _f_ _C_ _∂_ _l_ ( _y_ _**x**_ _i_, _**y**_ )



we can conclude that �� _∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� 2 _[≤]_ [Θ] [(] _[d]_ [15] [/] _[δ]_ [)] [ and hence this proves that the function] _[ f]_ _[C]_ _[l]_ [is]
Lipschitz continuous with Lipschitz constant Θ ( _d_ [15] / _δ_ ) .
To prove the smoothness of _f_ _C_ _l_, we use the property (B) of the Definition 7.3 and we have



_d_
### f C l ( x, y ) = ∑ ( x j − y j ) · [∂] [2] [α] [j] [(] [x] [)]

_∂x_ _i_ _∂x_ _ℓ_ _j_ = 1 _∂x_ _i_ _∂x_ _ℓ_



_∂_ [2] _f_ _C_ _l_ ( _**x**_, _**y**_ )




_[∂]_ _[α]_ _[j]_ [(] _**[x]**_ [)] + _[∂α]_ _[ℓ]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂x_ _ℓ_ _∂x_ _i_




_[ℓ]_ [(] _**[x]**_ [)] _∂α_ _i_ ( _**x**_ )

+
_∂x_ _i_ _∂x_ _ℓ_



_∂x_ _ℓ_,



_∂_ [2] _f_ _C_ _l_ ( _**x**_, _**y**_ )



_∂_ _[ℓ]_ _x_ [(] _i_ _**[x]**_ [)], and _∂_ [2] _∂fy_ _C_ _i_ _l_ _∂_ ( _**x**_ _y_, _ℓ_ _**y**_ )



_f_ _C_ _l_ ( _**x**_, _**y**_ ) = _−_ _[∂α]_ _[ℓ]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂y_ _ℓ_ _∂x_ _i_



= 0
_∂y_ _i_ _∂y_ _ℓ_



where
_∂_ [2] _α_ _j_ ( _**x**_ ) = _−_
### ∑

_∂x_ _i_ _∂x_ _ℓ_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_



_∂_ [2] P _**v**_ ( _**x**_ )

_· C_ _[j]_
_l_ [(] _**[v]**_ [)]
_∂x_ _i_ _∂x_ _ℓ_



Again using the property (C) of Definition 7.3 we get that there are most _d_ + 1 vertices _**v**_ of _R_ _c_ ( _**x**_ )
such that _∇_ [2] P _**v**_ ( _**x**_ ) _̸_ = 0. This together with the property (A) of Definition 7.3 leads to the fact that
��� _∂∂_ [2] _xα_ _i_ _∂_ _j_ ( _**x**_ _x_ ) _ℓ_ ��� _≤_ Θ ( _d_ 25 / _δ_ 2 ) . Using the later together with the bounds that we obtained for ��� _∂α∂_ _j_ _x_ ( _i_ _**x**_ ) ��� in

2
the beginning of the proof we get that �� _∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� _F_ _[≤]_ [Θ] [(] _[d]_ [27] [/] _[δ]_ [2] [)] [, where with] _[ ∥·∥]_ _[F]_ [we denote]
the Frobenious norm. Since the bound on the Frobenious norm is a bound to the spectral norm
too, we get that the function _f_ _C_ _l_ is Θ ( _d_ [27] / _δ_ [2] ) -smooth.


**7.2.3** **Description and Correctness of the Reduction – Proof of Theorem 4.4**


We start with a description of the reduction from HighD-BiSperner to GDAFixedPoint. Suppose we have an instance of HighD-BiSperner given by boolean circuit _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→_
_{−_ 1, 1 _}_ _[d]_, we construct an instance of GDAFixedPoint according to the following set of rules.


**(** _⋆_ ) **Construction of Instance for Fixed Points of Gradient Descent/Ascent.**


 - The payoff function is the real-valued function _f_ _C_ _l_ ( _**x**_, _**y**_ ) from the Definition 7.4.


 - The domain is the polytope _P_ ( _**A**_, _**b**_ ) that we described in Section 3. The matrix _**A**_ and the
vector _**b**_ are computed so that the following inequalities hold


_x_ _i_ _−_ _y_ _i_ _≤_ ∆, _y_ _i_ _−_ _x_ _i_ _≤_ ∆ for all _i_ _∈_ [ _d_ ] (7.1)



��� _≤_ Θ ( _d_ 25 / _δ_ 2 ) . Using the later together with the bounds that we obtained for ��� _∂α∂_ _j_ _x_ ( _i_ _**x**_ )



in
���



_∂x_ _i_



where ∆ = _t_ _·_ _δ_ / _d_ [14], with _t_ _∈_ **R** + be a constant such that ��� _∂_ P _∂_ _**v**_ _x_ ( _i_ _**x**_ )




_·_ _δ_
��� _d_ [12] _[t]_ _[ ≤]_ [1] 2 [, for all] _**[ v]**_ _[ ∈]_



([ _N_ ] _−_ 1 ) _[d]_ and _**x**_ _∈_ [ 0, 1 ] _[d]_ . The fact that such a constant _t_ exists follows from the property
(A) of the smooth and efficient coefficients.


- The parameter _α_ is set to be equal to ∆/3.


- The parameters _G_ and _L_ are set to be equal to the upper bounds on the Lipschitzness and
the smoothness of _f_ _C_ _l_ respectively that we derived in Lemma 7.5. Namely we have that
_G_ = _O_ ( _d_ [15] / _δ_ ) and _L_ = _O_ ( _d_ [27] / _δ_ [2] ) .


37


The first thing to observe is that the afore-described reduction is polynomial-time. For this
observe that all of _α_, _G_, _L_, _**A**_, and _**b**_ have representation that is polynomial in _d_ even if we use
unary instead of binary representation. So the only thing that remains is the existence of a Turing
machine _C_ _f_ _C_ _l_ that computes the function and the gradient value of _f_ _C_ _l_ in time polynomial to the
size of _C_ _l_ and the requested accuracy. To prove this we need a detailed description of the SEIC
coefficients and for this reason we postpone the proof of this to the Appendix D. Here we state
the formally the result that we prove in the Appendix D which together with the discussion
above proves that our reduction is indeed polynomial-time.


**Theorem 7.6.** _Given a binary circuit_ _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→{−_ 1, 1 _}_ _[d]_ _that is an input to the_ HighD _-_
BiSperner _problem. Then, there exists a polynomial-time Turing machine_ _C_ _f_ _C_ _l_ _, that can be constructed_

_in polynomial-time from the circuit_ _C_ _l_ _such that for all vector_ _**x**_, _**y**_ _∈_ [ 0, 1 ] _[d]_ _and accuracy ε_ _>_ 0 _,_ _C_ _f_ _C_ _l_
_computes both z_ _∈_ **R** _and_ _**w**_ _∈_ **R** _[d]_ _such that_

�� _z_ _−_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� _≤_ _ε_, �� _**w**_ _−∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) �� 2 _[≤]_ _[ε]_ [.]


_Moreover the running time of_ _C_ _f_ _C_ _l_ _is polynomial in the binary representation of_ _**x**_ _,_ _**y**_ _, and_ log ( 1/ _ε_ ) _._


We also observe that according to Lemma 7.5, the function _f_ _C_ _l_ is both _G_ -Lipschitz and _L_  smooth and hence the output of our reduction is a valid input for the constructed instance of
the promise problem GDAFixedPoint. The next step is to prove that the vector _**x**_ _[⋆]_ of every
solution ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) of GDAFixedPoint with input as we described above, lies in a cubelet that
is either panchromatic according to _C_ _l_ or is a violation of the rules for proper coloring of the
HighD-BiSperner problem.


**Lemma 7.7.** _Let_ _C_ _l_ _be an input to the_ HighD _-_ BiSperner _problem, let f_ _C_ _l_ _be the corresponding G-_
_Lipschitz and L-smooth function defined in Definition 7.4, and let_ _P_ ( _**A**_, _**b**_ ) _be the polytope defined by_
(7.1) _. If_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _is any solution to the_ GDAFixedPoint _problem with input α, G, L,_ _C_ _f_ _C_ _l_ _,_ _**A**_ _, and_ _**b**_ _,_

_defined in_ ( _⋆_ ) _then the following statements hold, where we remind that_ ∆ = _t_ _·_ _δ_ / _d_ [14] _._



_⋄_ _If x_ _i_ _[⋆]_ _[∈]_ [(] _[α]_ [, 1] _[ −]_ _[α]_ [)] _[ and x]_ _i_ _[⋆]_ _[∈]_ [(] _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] _[ then]_ ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ )


_⋄_ _If x_ _i_ _[⋆]_ _[≤]_ _[α][ or x]_ _i_ _[⋆]_ _[≤]_ _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥−_ _α._


_⋄_ _If x_ _i_ _[⋆]_ _[≥]_ [1] _[ −]_ _[α][ or x]_ _i_ _[⋆]_ _[≥]_ _[y]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≤_ _α._


_The symmetric statements for y_ _i_ _[⋆]_ _[hold.]_

_⋄_ _If y_ _i_ _[⋆]_ _[∈]_ [(] _[α]_ [, 1] _[ −]_ _[α]_ [)] _[ and y]_ _i_ _[⋆]_ _[∈]_ [(] _[x]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [,] _[ x]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α]_ [)] _[ then]_ ��� _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ )


_⋄_ _If y_ _i_ _[⋆]_ _[≤]_ _[α][ or y]_ _i_ _[⋆]_ _[≤]_ _[x]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≤_ _α._



_≤_ _α._
���


_≤_ _α._
���



_⋄_ _If y_ _i_ _[⋆]_ _[≥]_ [1] _[ −]_ _[α][ or y]_ _i_ _[⋆]_ _[≥]_ _[x]_ _i_ _[⋆]_ [+] [ ∆] _[−]_ _[α][ then]_ _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥−_ _α._


_Proof._ The proof of this lemma is identical to the proof of Lemma 6.7 and for this reason we skip
the details of the proof here.


38


**Lemma 7.8.** _Let_ _C_ _l_ _be an input to the_ HighD _-_ BiSperner _problem, let f_ _C_ _l_ _be the corresponding G-_
_Lipschitz and L-smooth function defined in Definition 7.4, and let_ _P_ ( _**A**_, _**b**_ ) _be the polytope defined by_
(7.1) _. If_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _is any solution to the_ GDAFixedPoint _problem with input α, G, L,_ _C_ _f_ _C_ _l_ _,_ _**A**_ _, and_ _**b**_ _,_
_defined in_ ( _⋆_ ) _, then none of the following statements hold for the cubelet R_ ( _**x**_ _[⋆]_ ) _._


_1. x_ _i_ _[⋆]_ _[≥]_ [1/] [(] _[N]_ _[ −]_ [1] [)] _[ and for any]_ _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ _[i]_ [(] _**[v]**_ [) =] _[ −]_ [1] _[.]_


_2. x_ _i_ _[⋆]_ _[≤]_ [1] _[ −]_ [1/] [(] _[N]_ _[ −]_ [1] [)] _[ and for any]_ _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] _[, it holds that]_ _[ C]_ _l_ [1] [(] _**[v]**_ [) = +] [1] _[.]_


_Proof._ We prove that there is no solution ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) of GDAFixedPoint that satisfies the statement
1. and the fact that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) cannot satisfy the statement 2. follows similarly. It is convenient
for us to define ˆ _**x**_ = _**x**_ _[⋆]_ _−∇_ _x_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _K_ ( _**y**_ _[⋆]_ ) = _{_ _**x**_ _|_ ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ )) _}_, _**z**_ = Π _K_ ( _**y**_ _⋆_ ) ˆ _**x**_, and

ˆ
_**y**_ = _**y**_ _[⋆]_ _−∇_ _y_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _K_ ( _**x**_ _[⋆]_ ) = _{_ _**y**_ _|_ ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ )) _}_, _**w**_ = Π _K_ ( _**x**_ _⋆_ ) ˆ _**y**_ .
For the sake of contradiction we assume that there exists a solution of ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) such that
_x_ 1 _[⋆]_ _[≥]_ [1/] [(] _[N]_ _[ −]_ [1] [)] [ and for any] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] [ it holds that] _[ C]_ _l_ _[i]_ [(] _**[v]**_ [) =] _[ −]_ [1. Using this fact, we will prove]


_̸_


_̸_


_̸_



that (1) _∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )


_̸_


_̸_


_̸_



= _−_ 1.
_∂y_ _i_


_̸_


_̸_


_̸_



( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥_ 1/2, and (2) _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ )


_̸_


_̸_


_̸_



Let _R_ ( _**x**_ _[⋆]_ ) = _Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�


_̸_


_̸_


_̸_




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �


_̸_


_̸_


_̸_




_[c]_ _N_ _[d]_ _−_ [+] 1 [1], then since all the corners _**v**_ _∈_ _R_ + ( _**x**_ _[⋆]_ ) have

�


_̸_


_̸_


_̸_



_C_ _[i]_
_l_ [(] _**[v]**_ [) =] _[ −]_ [1, from the Definition][ 7.4][ we have that]


_d_
### f C l ( x [⋆], y [⋆] ) = ( x i [⋆] [−] [y] i [⋆] [) +] ∑ ( x [⋆] j [−] [y] [⋆] j [)] [ ·] [ α] [j] [(] [x] [)]

_j_ = 1, _j_ _̸_ = _i_


If we differentiate this with respect to _y_ _i_ we immediately get that _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _y_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) = _−_ 1. On the other

hand if we differentiate with respect to _x_ _i_ we get


_̸_


_̸_



_̸_


_d_

_**x**_ _[⋆]_, _**y**_ _[⋆]_ )
### = 1 + ∑ ( x j − y j ) · [∂α] [j] [(] [x] [)]

_∂x_ _i_ _j_ = 1, _j_ _̸_ = _i_ _∂x_ _i_


_̸_



_̸_


_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )


_̸_


_̸_



_̸_


_̸_ _∂x_ _i_


_̸_



_̸_


_̸_

_̸_ ����



_̸_


_̸_

### ≥ 1 − ∑

_j_ _̸_ = _i_



_̸_


_̸_

�� _x_ _j_ _−_ _y_ _j_ �� _·_ _∂α_ _j_ ( _**x**_ )

_̸_ ���� _∂x_ _i_



_̸_


_̸_


_̸_


_d_ 13
_≥_ 1 _−_ ∆ _·_ _d_ _·_ Θ
� _δ_

_≥_ 1/2



_̸_


_̸_


_̸_

�



_̸_


_̸_


_̸_


where the above follows from the following facts: (1) that ��� _∂α∂_ _j_ _x_ ( _l_ _**x**_ )



_̸_


_̸_


_̸_


_≤_ Θ ( _d_ 13 / _δ_ ), which is proved in
���



_̸_


_̸_


_̸_


the proof of Lemma 7.5, (2) �� _x_ _j_ _−_ _y_ _j_ �� _≤_ ∆, and (3) the definition of ∆. Now it is easy to see that the

only way to satisfy both _∂_ _f_ _C_ _l_ ( _∂_ _**x**_ _x_ _[⋆]_ _i_, _**y**_ _[⋆]_ ) _≥_ 1/2 and _|_ _z_ _i_ _−_ _x_ _i_ _[⋆]_ _[| ≤]_ _[α]_ [ is that either] _[ x]_ _i_ _[⋆]_ _[≤]_ _[α]_ [ or] _[ x]_ _i_ _[⋆]_ _[≤]_ _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [.]

The first case is excluded by the assumption of the first statement of our lemma and our choice
of _α_ = ∆/3 _<_ 1/ ( _N_ _−_ 1 ), thus it holds that _x_ _i_ _[⋆]_ _[≤]_ _[y]_ _i_ _[⋆]_ _[−]_ [∆] [+] _[ α]_ [. But then we can use the case 3.]

for the _y_ variables of Lemma 6.7 and we get that _∂_ _f_ _C_ _l_ _∂_ ( _**x**_ _y_ _[⋆]_ 1, _**y**_ _[⋆]_ ) _≥−_ _α_, which cannot be true since

_∂_ _f_ _C_ _l_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) = _−_
we proved that _∂y_ _i_ 1. Therefore we have a contradiction and the first statement of the

lemma holds. Using the same reasoning we prove the second statement too.


We are now ready to complete the proof that the our reduction from HighD-BiSperner to
GDAFixedPoint is correct and hence we can prove Theorem 4.4.


39


_Proof of Theorem 4.4._ Let ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) be a solution to the GDAFixedPoint problem with input a Turing machine that represents the function _f_ _C_ _l_, _α_ = ∆/3, where ∆ = _t_ _·_ _δ_ / _d_ [14], _G_ = Θ ( _d_ [15] / _δ_ ),
_L_ = Θ ( _d_ [27] / _δ_ [2] ), and _**A**_, _**b**_ as described in ( _⋆_ ).
For each coordinate _i_, there exist the following three mutually exclusive cases,


**1** **1**

_▷_ _**N**_ _**−**_ **1** _**[≤]**_ _**[x]**_ _**i**_ _**[⋆]**_ _**[≤]**_ **[1]** _**[ −]**_ _**N**_ _**−**_ **1** [: Since] _[ |]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] _[| ≥]_ [1, it follows directly from Lemma][ 7.8][ that there]
exists _**v**_ _∈_ _R_ + ( _**x**_ _[⋆]_ ) such that _C_ _l_ _[i]_ [(] _**[v]**_ [) =] _[ −]_ [1 and] _**[ v]**_ _[′]_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] [ such that] _[ C]_ _l_ _[i]_ [(] _**[v]**_ [) = +] [1.]


**1**

_▷_ **0** _**≤**_ _**x**_ _**i**_ _**[⋆]**_ _**[<]**_ _**N**_ _**−**_ **1** [: Let] _[ C]_ _l_ _[i]_ [(] _**[v]**_ [) =] _[ −]_ [1 for all] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] [.] By the property (D) of the SEIC
coefficients, we have that there exists _**v**_ _∈_ _R_ + ( _x_ _[⋆]_ ) with _v_ _i_ = 0. This node is hence a solution
of type 2. for the HighD-BiSperner problem.


**1**

_▷_ **1** _**−**_ _**N**_ _**−**_ **1** _**[<]**_ _**[ x]**_ _**i**_ _**[⋆]**_ _**[≤]**_ **[1]** [: Let] _[ C]_ _l_ _[i]_ [(] _**[v]**_ [) = +] [1 for all] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ _[⋆]_ [)] [. By the property][ (D)][ of the SEIC]
coefficients, we have that there exists _**v**_ _∈_ _R_ + ( _x_ _[⋆]_ ) with _v_ _i_ = 1. This node is hence a solution
of type 3. for the HighD-BiSperner problem.


Since **R** + ( _**x**_ _[⋆]_ ) computable in polynomial time given _**x**_ _[⋆]_, we can easily check for every _i_ _∈_ [ _d_ ]
whether any of the above cases hold. If at least for some _i_ _∈_ [ _d_ ] the 2nd or the 3rd case from
above hold, then the corresponding vertex gives a solution to the HighD-BiSperner problem
and therefore our reduction is correct. Hence we may assume that for every _i_ _∈_ [ _d_ ] the 1st of
the above cases holds. This implies that the cubelet _R_ ( _**x**_ _[⋆]_ ) is pachromatic and therefore it is a
solution to the problem HighD-BiSperner. Finally, we observe that the function that we define
has range [ _−_ _d_, _d_ ] and hence the Theorem 4.4 follows using Theorem 5.1.

#### **8 Smooth and Efficient Interpolation Coefficients**


In this section we describe the construction of the smooth and efficient interpolation coefficients
(SEIC) that we introduced in Section 7.2.1. After the description of the construction we present
the statements of the lemmas that prove the properties (A) - (D) of their Definition 7.3 and we
refer to the Appendix C. We first remind the definition of the SEIC coefficients.


**Definition 7.3** (Smooth and Efficient Interpolation Coefficients) **.** For every _N_ _∈_ **N** we define the
set of _smooth and efficient interpolation coefficients (SEIC)_ as the family of functions, called _coefficients_,
_I_ _d_, _N_ = P _**v**_ : [ 0, 1 ] _[d]_ _→_ **R** _|_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ [�] with the following properties.
�


(A) For all vertices _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_, the coefficient P _**v**_ ( _**x**_ ) is a twice-differentiable function and
satisfies









_∂_ P _**v**_ ( _**x**_ )
��� _∂x_ _i_

_∂_ 2 P _**v**_ ( _**x**_ )
��� _∂x_ _i_ _∂x_ _ℓ_



12
_≤_ Θ ( _d_ / _δ_ ) .
���



24 2
_≤_ Θ ( _d_ / _δ_ ) .
���



(B) For all _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_, it holds that P _**v**_ ( _**x**_ ) _≥_ 0 and ∑ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _d_ P _**v**_ ( _**x**_ ) = ∑ _**v**_ _∈_ _R_ _c_ ( _**x**_ ) P _**v**_ ( _**x**_ ) = 1.


(C) For all _**x**_ _∈_ [ 0, 1 ] _[d]_, it holds that all but _d_ + 1 of the coefficients P _**v**_ _∈I_ _d_, _N_ satisfy P _**v**_ ( _**x**_ ) = 0,
_∇_ P _**v**_ ( _**x**_ ) = 0 and _∇_ [2] P _**v**_ ( _**x**_ ) = 0. We denote this set of _d_ + 1 vertices by _R_ + ( _**x**_ ) . Furthermore,
it holds that _R_ + ( _**x**_ ) _⊆_ _R_ _c_ ( _**x**_ ) and given _**x**_ we can compute the set _R_ + ( _**x**_ ) it time poly ( _d_ ) .


(D) For all _**x**_ _∈_ [ 0, 1 ] _[d]_, if _x_ _i_ _≤_ 1/ ( _N_ _−_ 1 ) for some _i_ _∈_ [ _d_ ] then there exists _**v**_ _∈_ _R_ + ( _**x**_ ) such that
_v_ _i_ = 0. Respectively, if _x_ _i_ _≥_ 1 _−_ 1/ ( _N_ _−_ 1 ) then there exists _**v**_ _∈_ _R_ + ( _**x**_ ) such that _v_ _i_ = 1.


40


Our main goal in this section is to prove the following theorem.


**Theorem 8.1.** _For every d_ _∈_ **N** _and every N_ = poly ( _d_ ) _there exist a family of functions_ _I_ _d_, _N_ _that_
_satisfies the properties (A) - (D) of Definition 7.3._


One important component of the construction of the SEIC coefficients is the _smooth-step func-_
_tions_ which we introduce in Section 8.1. These functions also provide a toy example of smooth
and efficient interpolation coefficients in 1 dimension. Then in Section 8.2 we present the construction of the SEIC coefficients in multiple dimensions and in Section 8.3 we state the main
lemmas that lead to the proof of Theorem 8.1.


**8.1** **Smooth Step Functions – Toy Single Dimensional Example**


Smooth step functions are real-valued function _g_ : **R** _→_ **R** of a single real variable with the
following properties


**Step Value.** For every _x_ _≤_ 0 it holds that _g_ ( _x_ ) = 0, for every _x_ _≥_ 1 it holds that _g_ ( _x_ ) = 1 and for
every _x_ _∈_ [ 0, 1 ] it holds that _S_ ( _x_ ) _∈_ [ 0, 1 ] .


**Smoothness.** For some _k_ it holds that _g_ is _k_ times continuously differentiable and its _k_ th derivative satisfies _g_ [(] _[k]_ [)] ( 0 ) = 0 and _g_ [(] _[k]_ [)] ( 1 ) = 0.


The largest number _k_ such that the smoothness property from above holds is characterizes the
_order of smoothness_ of the smooth step function _g_ .
In Section 6 we have already defined and used the smooth step function of order 1. For the
construction of the SEIC coefficients we use the smooth step function of order 2 and the smooth
step function of order ∞ defined as follows.


**Definition 8.2.** We define the smooth step function _S_ : **R** _→_ **R** of order 2 as the following function



_S_ ( _x_ ) =



6 _x_ [5] _−_ 15 _x_ [4] + 10 _x_ [3] _x_ _∈_ ( 0, 1 )


0 _x_ _≤_ 0




1 _x_ _≥_ 1



.



We also define the smooth step function _S_ ∞ : **R** _→_ **R** of order ∞ as the following function



_S_ ∞ ( _x_ ) =














2 _[−]_ [1/] _[x]_

2 _[−]_ [1/] _[x]_ + 2 _[−]_ [1/] [(] [1] _[−]_ _[x]_ [)] _x_ _∈_ ( 0, 1 )


0 _x_ _≤_ 0


1 _x_ _≥_ 1



.



We note that we use the notation _S_ instead of _S_ 2 for the smooth step function of order 2 for
simplicitly of the exposition of the paper.


We present a plot of these step function in Figure 7, and we summarize some of their properties in Lemma 8.3. A more detailed lemma with additional properties of _S_ ∞ that are useful for
the proof of Theorem 8.1 is presented in Lemma C.5 in the Appendix C.


**Lemma 8.3.** _Let S and S_ ∞ _be the smooth step functions defined in Definition 8.2. It holds both S and_
_S_ ∞ _are monotone increasing functions and that S_ ( 0 ) = 0 _, S_ ( 1 ) = 1 _and also S_ _[′]_ ( 0 ) = _S_ _[′]_ ( 1 ) = _S_ _[′′]_ ( 0 ) =
_S_ _[′′]_ ( 1 ) = 0 _. It also holds that S_ ∞ ( 0 ) = 0 _, S_ ∞ ( 1 ) = 1 _and also S_ ∞ [(] _[k]_ [)] [(] [0] [) =] _[ S]_ ∞ [(] _[k]_ [)] [(] [1] [) =] [ 0] _[ for every]_
_k_ _∈_ **N** _. Additionally it holds for every x that_ _|_ _S_ _[′]_ ( _x_ ) _| ≤_ 2 _, and_ _|_ _S_ _[′′]_ ( _x_ ) _| ≤_ 6 _whereas_ _|_ _S_ ∞ _[′]_ [(] _[x]_ [)] _[| ≤]_ [16] _[, and]_
_|_ _S_ ∞ _[′′]_ [(] _[x]_ [)] _[| ≤]_ [32] _[.]_


41


(b) The function _P_ 3 from Example 8.4.



P 3



1.0


0.8


0.6


0.4


0.2



S S ∞


(a) Functions _S_ and _S_ ∞ .



1.0


0.8


0.6


0.4


0.2



5



5



45 1



5



Figure 7: (a) The smooth step function _S_ of order 1 and the smooth step function _S_ ∞ of order ∞.
As we can see both _S_ and _S_ ∞ are continuous and continuously differentiable functions but _S_ ∞ is
much more flat around 0 and 1 since it has all its derivatives equal to 0 both at the point 0 and at
the point 1. This makes the _S_ ∞ function infinitely many times differentiable. (b) The constructed
_P_ 3 function of the family of SEIC coefficients for single dimensional case with _N_ = 5. For details
we refer to the Example 8.4.


_Proof._ For the function _S_ we compute _S_ _[′]_ ( _x_ ) = 30 _x_ [4] _−_ 60 _x_ [3] + 30 _x_ [2] for _x_ _∈_ [ 0, 1 ] and _S_ _[′]_ ( _x_ ) = 0
for _x_ _̸∈_ [ 0, 1 ] . Therefore we can easily get that _|_ _S_ _[′]_ ( _x_ ) _| ≤_ 2 for all _x_ _∈_ **R** . We also have that
_S_ _[′′]_ ( _x_ ) = 120 _x_ [3] _−_ 180 _x_ [2] + 60 _x_ for _x_ _∈_ ( 0, 1 ) and _S_ _[′′]_ ( _x_ ) = 0 for _x_ _̸∈_ [ 0, 1 ] hence we can conclude
that _|_ _S_ _[′′]_ ( _x_ ) _| ≤_ 6.
The calculations for _S_ ∞ are more complicated. We have that



ln ( 2 )
exp � _x_ ( 1 _−_ _x_ ) � ( 1 _−_ 2 _x_ ( 1 _−_ _x_ ))
_S_ ∞ _[′]_ [(] _[x]_ [) =] [ ln] [(] [2] [)] 2
ln ( 2 ) ln ( 2 )
exp _x_ + exp 1 _−_ _x_ ( 1 _−_
~~�~~ ~~�~~ ~~�~~ ~~�~~ ~~��~~



2 .

ln ( 2 )

1 _−_ _x_ ( 1 _−_ _x_ ) [2] _x_ [2]
~~��~~



( 2 ) ln ( 2 )

_x_ + exp 1 _−_ _x_
~~�~~ ~~�~~



We set _h_ ( _x_ ) ≜ exp ln _x_ ( 2 )
� �



( 2 ) ln ( 2 )

_x_ + exp 1 _−_ _x_
� �



ln ( 2 )

1 _−_ _x_ ( 1 _−_ _x_ ) [2] _x_ [2] for _x_ _∈_ [ 0, 1 ] and doing simple calculations
��




[1] 4 [exp] ln _x_ ( 2 )

�



we get that for _x_ _≤_ 1/2 it holds that _h_ ( _x_ ) _≥_ [1] 4



( 2 )

_x_ _x_ [2] . But the later can be easily lower
�



bounded by 1/4. Applying the same argument for _x_ _≥_ 1/2 we get that in general _h_ ( _x_ ) _≥_ 1/4.

ln ( 2 ) ln ( 2 )
Also it is not hard to see that for _x_ _≤_ 1/2 it holds that exp � _x_ ( 1 _−_ _x_ ) � _≤_ 4 exp � _x_ �, whereas for

ln ( 2 ) ln ( 2 )
_x_ _≥_ 1/2 it holds that exp � _x_ ( 1 _−_ _x_ ) � _≤_ 4 exp � 1 _−_ _x_ �. Combining all these we can conclude that

_|_ _S_ ∞ _[′]_ [(] _[x]_ [)] _[| ≤]_ [16. Using similar argument we can prove that] _[ |]_ _[S]_ ∞ _[′′]_ [(] _[x]_ [)] _[| ≤]_ [32. For all the derivatives of]
_S_ ∞ we can inductively prove that


_k_ _−_ 1
### S ∞ [(] [k] [)] [(] [x] [) =] ∑ h i ( x ) · S ∞ [(] [i] [)] [(] [x] [)] [,]

_i_ = 0


where _h_ 0 ( 1 ) = 0 and all the functions _h_ _i_ ( _x_ ) are bounded. Then the fact that all the derivatives of
_S_ ∞ vanish at 0 and at 1 follows by a simple inductive argument.


_Example_ 8.4 (Single Dimensional Smooth and Efficient Interpolation Coefficients) _._ Using the
smooth step functions that we described above we can get a construction of SEIC coefficients for


42


the single dimensional case. Unfortunately the extension to multiple dimensions is substantially
harder and invokes new ideas that we explore later in this section. For the single dimensional
problem of this example we have the interval [ 0, 1 ] divided with _N_ discrete points and our goal
is to design _N_ functions P 1 - P _N_ that satisfy the properties (A) - (D) of Definition 7.3. A simple
construction of such functions is the following



P _i_ ( _x_ ) =



_i_
_S_ ∞ ( _N_ _·_ _x_ _−_ ( _i_ _−_ 1 )) _x_ _≤_ _N_ _−_ 1
_i_ .
� _S_ ∞ ( _i_ + 1 _−_ _N_ _·_ _x_ ) _x_ _>_ _N_ _−_ 1



Based on Lemma 8.3 it is not hard then to see that P _i_ is twice differentiable and it has bounded
first and second derivatives, hence it satisfies property (A) of Definition 8. Using the fact that
1 _−_ _S_ ∞ ( _x_ ) = _S_ ∞ ( 1 _−_ _x_ ) we can also prove property (B). Finally properties (C) and (D) can be
proved via the definition of the coefficient P _i_ from above. In Figure 7 we can see the plot of P 3
for _N_ = 5. We leave the exact proofs of this example as an exercise for the reader.


**8.2** **Construction of SEIC Coefficients in High-Dimensions**


The goal of this section is to present the construction of the family _I_ _d_, _N_ of smooth and efficient
interpolation coefficients for every number of dimensions _d_ and any discretization parameter _N_ .
Before diving into the details of our construction observe that even the 2-dimensional case with
_N_ = 2 is not trivial. In particular, the first attempt would be to define the SEIC coefficients based
on the simple split of the square [ 0, 1 ] [2] to two triangles divided by the diagonal of [ 0, 1 ] [2] . Then
using any soft-max function that is twice continuously differentiable we define a convex combination at every triangle. Unfortunately this approach cannot work since the resulting coefficients
have discontinuous gradients along the diagonal of [ 0, 1 ] [2] . We leave the presice calculations of
this example as an exercise to the reader.
We start with some definitions about the orientation and the representation of the cubelets of
the grid ([ _N_ ] _−_ 1 ) _[d]_ . Then we proceed with the definition of the _Q_ _**v**_ functions in Definition 8.7.
Finally using _Q_ _**v**_ we can proceed with the construction of the SEIC coefficients.



**Definition 8.5** (Source and Target of Cubelets) **.** Each cubelet _Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �




_[c]_ _N_ _[d]_ _−_ [+] 1 [1],

�



where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ admits a **source vertex** _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _d_ ) _∈_ ([ _N_ ] _−_ 1 ) _[d]_ and a **target vertex**
_**t**_ _**[c]**_ = ( _t_ 1, . . ., _t_ _d_ ) _∈_ ([ _N_ ] _−_ 1 ) _[d]_ defined as follows,


_c_ _j_ + 1 _c_ _j_ is odd _c_ _j_ _c_ _j_ is odd
_s_ _j_ = and _t_ _j_ =
� _c_ _j_ _c_ _j_ is even � _c_ _j_ + 1 _c_ _j_ is even


Notice that the source _**s**_ _**[c]**_ and the target _**t**_ _**[c]**_ are vertices of the cubelet whose down-left corner is _**c**_ .



**Definition 8.6.** (Canonical Representation) Let _**x**_ _∈_ [ 0, 1 ] _[d]_ and _R_ ( _**x**_ ) = _Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�




_[c]_ _N_ _[d]_ _−_ [+] 1 [1]

�




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �



where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ . The _**canonical representation**_ **of** _**x**_ **under cubelet with down-left cor-**
**ner** _**c**_, denoted by _**p**_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _[d]_ [)] [ is defined as follows,]


_p_ _j_ = _[x]_ _[j]_ _[ −]_ _[s]_ _[j]_

_t_ _j_ _−_ _s_ _j_


where _**t**_ _**[c]**_ = ( _t_ 1, . . ., _t_ _d_ ) and _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _d_ ) are respectively the _target_ and the _source_ of _R_ ( _**x**_ ) .


43


**Definition 8.7** (Defining the functions _Q_ _**v**_ ( _**x**_ ) ) **.** Let _**x**_ _∈_ [ 0, 1 ] _[d]_ lying in the cublet


_[̸]_



_R_ ( _**x**_ ) = _c_ 1 _[c]_ [1] [ +] [ 1]
� _N_ _−_ 1 [,] _N_ _−_ 1


_[̸]_



_× · · · ×_ _c_ _d_ _[c]_ _[d]_ [+] [ 1]
� � _N_ _−_ 1 [,] _N_ _−_ 1


_[̸]_



,
�


_[̸]_



with corners _R_ _c_ ( _**x**_ ) = _{_ _c_ 1, _c_ 1 + 1 _} × · · · × {_ _c_ _d_, _c_ _d_ + 1 _}_, where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ . Let also _**s**_ _**[c]**_ =
( _s_ 1, . . ., _s_ _d_ ) be the source vertex of _R_ ( _**x**_ ) and _**p**_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _[d]_ [)] [ be the canonical representation of]
_**x**_ . Then for each vertex _**v**_ _∈_ _R_ _c_ ( _**x**_ ) we define the following partition of the set of coordinates [ _d_ ],


_A_ _**[c]**_ _**v**_ [=] _[ {]_ _[j]_ [ :] _|_ _v_ _j_ _−_ _s_ _j_ _|_ = 0 _}_ and _B_ _**v**_ _**[c]**_ [=] _[ {]_ _[j]_ [ :] _|_ _v_ _j_ _−_ _s_ _j_ _|_ = 1 _}_


If there exist _j_ _∈_ _A_ _**[c]**_ _**v**_ [and] _[ ℓ]_ _[∈]_ _[B]_ _**v**_ _**[c]**_ [such that] _[ p]_ _j_ _[≥]_ _[p]_ _ℓ_ [then] _[ Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0. Otherwise we define] [7]


_[̸]_




_[̸]_

_Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =]



∏ _j_ _∈_ _A_ _**cv**_ ∏ _ℓ∈_ _B_ _**cv**_ _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) _A_ _**[c]**_ _**v**_ [,] _[ B]_ _**v**_ _**[c]**_ _[̸]_ [=][ ∅]

∏ _[d]_ _ℓ_ = 1 _[S]_ [∞] [(] [1] _[ −]_ _[S]_ [(] _[p]_ _[ℓ]_ [))] _B_ _**v**_ _**[c]**_ [=][ ∅]




∏ _[d]_ _j_ = 1 _[S]_ [∞] [(] _[S]_ [(] _[p]_ _[j]_ [))] _A_ _**[c]**_ _**v**_ [=][ ∅]




_[̸]_


where _S_ ∞ ( _x_ ) and _S_ ( _x_ ) are the smooth step function defined in Definition 8.2.


To provide a better understanding of the Definitions 8.5, 8.6, and 8.7 we present the following
3-dimensional example.


_Example_ 8.8 _._ We consider a case where _d_ = 3 and _N_ = 3. Let _**x**_ = ( 1.3/3, 2.5/3, 0.3/3 ) lying
1 2
in the cubelet _R_ ( _**x**_ ) = � 3 [,] [2] 3 � _×_ � 3 [, 1] � _×_ �0, 3 [1] �, and let _**c**_ = ( 1, 2, 0 ) . Then the source of _R_ ( _**x**_ ) is




_[̸]_


1 [2]

3 [,] 3




_[̸]_


2

3 [, 1] � _×_ �0, 3 [1]




_[̸]_


2

[2] 3 � _×_ � 3




_[̸]_


1 2
in the cubelet _R_ ( _**x**_ ) = � 3 [,] [2] 3 � _×_ � 3 [, 1] � _×_ �0, 3 [1] �, and let _**c**_ = ( 1, 2, 0 ) . Then the source of _R_ ( _**x**_ ) is

_**s**_ _**[c]**_ = ( 2, 2, 0 ) and the target _**t**_ _**[c]**_ = ( 1, 3, 1 ) (Definition 8.5). The canonical representation of _**x**_ is
_**p**_ _**[c]**_ _**x**_ [= (] [0.7, 0.5, 0.3] [)] [ (Definition][ 8.6][). The only vertices with no-zero coefficients] _[ Q]_ _**v**_ _**[c]**_ [(] _**[x]**_ [)] [ are those]
belonging in the set _R_ + ( _**x**_ ) = _{_ ( 1, 3, 1 ), ( 1, 3, 0 ), ( 1, 2, 0 ), ( 2, 2, 0 ) _}_ and again by Definition 8.7 we
have that




_[̸]_


_▷_ _Q_ ( 1,3,1 ) ( _**x**_ ) = _S_ ∞ ( _S_ ( 0.3 )) _·_ _S_ ∞ ( _S_ ( 0.5 )) _·_ _S_ ∞ ( _S_ ( 0.7 )),


_▷_ _Q_ ( 1,3,0 ) ( _**x**_ ) = _S_ ∞ ( _S_ ( 0.5 ) _−_ _S_ ( 0.3 )) _·_ _S_ ∞ ( _S_ ( 0.7 ) _−_ _S_ ( 0.3 )),


_▷_ _Q_ ( 1,2,0 ) ( _**x**_ ) = _S_ ∞ ( _S_ ( 0.7 ) _−_ _S_ ( 0.3 )) _·_ _S_ ∞ ( _S_ ( 0.7 ) _−_ _S_ ( 0.5 )),


_▷_ _Q_ ( 2,2,0 ) ( _**x**_ ) = _S_ ∞ ( 1 _−_ _S_ ( 0.3 )) _·_ _S_ ∞ ( 1 _−_ _S_ ( 0.5 )) _·_ _S_ ∞ ( 1 _−_ _S_ ( 0.7 )) .


Now based on the Definitions 8.5, 8.6, and 8.7 we are ready to present the construction of the
smooth and efficient interpolation coefficients.


**Definition 8.9** (Construction of SEIC Coefficients) **.** Let _**x**_ _∈_ [ 0, 1 ] _[d]_ lying in the cubelet _R_ ( _**x**_ ) =
_Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1] . Then for each vertex _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ the coefficient P _**v**_ ( _**x**_ ) is
� � � �




_[̸]_


_Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1] . Then for each vertex _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ the coefficient P _**v**_ ( _**x**_ ) is
� � � �

defined as follows,




_[̸]_


_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �




_[̸]_


P _**v**_ ( _**x**_ ) =




_[̸]_


_Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [/] [(] [∑] _**v**_ _∈_ _R_ _c_ ( _**x**_ ) _[Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [))] if _**v**_ _∈_ _R_ _c_ ( _**x**_ )
0 if _**v**_ / _∈_ _R_ _c_ ( _**x**_ )
�




_[̸]_


where the functions _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ ≥]_ [0 are defined in Definition][ 8.7][ for any] _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _**[x]**_ [)] [.]


7 We note that in the following expression ∏ denotes the product symbol and should not be confused with the
projection operator used in the previous sections.


44


**8.3** **Sketch of the Proof of Theorem 8.1**


First it is necessary to argue that P _**v**_ ( _**x**_ ) is a continuous function since it could be the case that
_Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [/] [(] [∑] _**v**_ _∈_ _R_ _**c**_ ( _**x**_ ) _[Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [))] _[ ̸]_ [=] _[ Q]_ _**v**_ _**[c]**_ _[′]_ [(] _**[x]**_ [)] [/] [(] [∑] _**v**_ _∈_ _V_ _**c**_ _′_ _[Q]_ _**v**_ _**[c]**_ _[′]_ [(] _**[x]**_ [))] [ for some point] _**[ x]**_ [ that lies in the boundary of]
two adjacent cubelets with down-left corners _**c**_ and _**c**_ _[′]_ respectively. We specifically design the
coefficients _Q_ _**[c]**_ _v_ [(] _**[x]**_ [)] [ such as the latter does not occur and this is the main reason that the definition]
of the function _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ is slightly complicated. For this reason we prove the following lemma.]


**Lemma 8.10.** _For any vertex_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _,_ P _**v**_ ( _**x**_ ) _is a continuous and twice differentiable function_
_and for any_ _**v**_ / _∈_ _R_ _c_ ( _**x**_ ) _it holds that_ P _**v**_ ( _**x**_ ) = _∇_ P _**v**_ ( _**x**_ ) = _∇_ [2] P _**v**_ ( _**x**_ ) = 0 _. Moreover, for every_ _**x**_ _∈_ [ 0, 1 ] _[d]_

_the set R_ + ( _**x**_ ) _of vertices_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _such that_ P _**v**_ ( _**x**_ ) _>_ 0 _satisfies_ _|_ _R_ + ( _**x**_ ) _|_ = _d_ + 1 _._


Based on Lemma 8.10 and the expression of P _**v**_ we can prove that the P _**v**_ coefficients defined in
Definition 8.9 satisfy the properties (B) and (C) of the definition 7.3. To prove the properties (A)
and (D) we also need the following two lemmas.


**Lemma 8.11.** _For any vertex_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _, it holds that_



_≤_ Θ ( _d_ 12 / _δ_ ) _,_
���



_1._


_2._



_∂_ P _**v**_ ( _**x**_ )
��� _∂x_ _i_



_∂_ 2 P _**v**_ ( _**x**_ )
��� _∂x_ _i_ _∂x_ _j_



24 2
_≤_ Θ ( _d_ / _δ_ ) _._
���



**Lemma 8.12.** _Let a point_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _and R_ + ( _**x**_ ) _the set of vertices with_ P _**v**_ ( _**x**_ ) _>_ 0 _, then we have that_


_1. If_ 0 _≤_ _x_ _i_ _<_ 1/ ( _N_ _−_ 1 ) _then there always exists a vertex_ _**v**_ _∈_ _R_ + ( _**x**_ ) _such that v_ _i_ = 0 _._


_2. If_ 1 _−_ 1/ ( _N_ _−_ 1 ) _<_ _x_ _i_ _≤_ 1 _then there always exists a vertex_ _**v**_ _∈_ _R_ + ( _**x**_ ) _such that v_ _i_ = 1 _._


The proofs of Lemmas 8.10, 8.11, and 8.12 can be found in Appendix C. Based on Lemmas 8.10,
8.11, and 8.12 we are now ready to prove Theorem 8.1.


_Proof of Theorem 8.1._ The fact that the coefficients P _**v**_ satisfy the property (A) follows directly
from Lemma 8.11. Property (B) follows directly from the definition of P _**v**_ in Definition 8.9 and
the simple fact that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ ≥]_ [0. Property][ (C)][ follows from the second part of Lemma][ 8.10][. Finally]
Property (D) follows directly from Lemma 8.12.

#### **9 Unconditional Black-Box Lower Bounds**


In this section our goal is to prove Theorem 4.5 based on the Theorem 4.4 that we proved in
Section 7 and the known black box lower bounds that we know for PPAD by [HPV89]. In this
section we assume that all the real number operation are performed with infinite precision.


**Theorem 9.1** ([HPV89]) **.** _Assume that there exists an algorithm A that has black-box oracle access to the_
_value of a function M_ : [ 0, 1 ] _[d]_ _→_ [ 0, 1 ] _[d]_ _and outputs_ _**w**_ _[⋆]_ _∈_ [ 0, 1 ] _[d]_ _. There exists a universal constant c_ _>_ 0
_such that if M is_ 2 _-Lipschitz and_ _∥_ _M_ ( _**w**_ _[⋆]_ ) _−_ _**w**_ _[⋆]_ _∥_ 2 _≤_ 1/ ( 2 _c_ ) _, then A has to make at least_ 2 _[d]_ _different_
_oracle calls to the function value of M._


It is easy to observe in the reduction in the proof of Theorem 7.2 is a black-box reduction and
in every evaluation of the constructed circuit _C_ _l_ only requires one evaluation of the input function
_M_ . Therefore the proof of Theorem 7.2 together with the Theorem 9.1 imply the following
corollary.


45


Figure 8: Pictorial representation on the way the black box lower bound follows from the white
box PPAD-completeness that presented in Section 7 and the known black box lower bounds for
the Brouwer problem by [HPV89]. In the figure we can see the four dimensional case of Section
6 that corresponds to the 2D-BiSperner and the 2-dimensional Brouwer. As we can see, in that
case 1 query to _O_ _f_ can be implemented with 3 queries to 2D-BiSperner and each of these can
be implemented with 1 query to 2-dimensional Brouwer. In the high dimensional setting of
Section 7, every query ( _**x**_, _**y**_ ) to the oracle _O_ _f_ to return the values _f_ ( _**x**_, _**y**_ ) and _∇_ _f_ ( _**x**_, _**y**_ ) can be
implemented via _d_ + 1 oracles to an HighD-BiSperner instance. Each of these oracles to HighDBiSperner can be implemented via 1 oracle to a Brouwer instance. Therefore an _M_ _[d]_ query
lower bound for Brouwer implies an _M_ _[d]_ query lower bound for HighD-BiSperner which in
turn implies an _M_ _[d]_ / ( _d_ + 1 ) query lower bound for our GDAFixedPoint and LR-LocalMinMax
problems.


**Corollary 9.2** (Black-Box Lower Bound for Bi-Sperner) **.** _Let_ _C_ _l_ : ([ _N_ ] _−_ 1 ) _[d]_ _→{−_ 1, 1 _}_ _[d]_ _be an_
_instance of the_ HighD _-_ BiSperner _problem with N_ = _O_ ( _d_ ) _. Then any algorithm that has black-box_
_oracle access to_ _C_ _l_ _and outputs a solution to the corresponding_ HighD _-_ BiSperner _problem, needs_ 2 _[d]_

_different oracle calls to the value of_ _C_ _l_ _._


Based on Corollary 9.2 and the reduction that we presented in Section 7, we are now ready
to prove Theorem 4.5.


_Proof of Theorem 4.5._ This proof follows the steps of Figure 8. The last part of that figure is established in Corollary 9.2. So what is left to prove Theorem 4.5 is that for every instance of
HighD-BiSperner we can construct a function _f_ such that the oracle _O_ _f_ can be implemented via
_d_ + 1 queries to the instance of HighD-BiSperner and also every solution of GDAFixedPoint
with oracle access _O_ _f_ to _f_ and _∇_ _f_ reveals one solution of the starting HighD-BiSperner instance.
To construct this oracle _O_ _f_ we follow exactly the reduction that we described in Section
7. The correctness of the reduction that we provide in Section 7 suffices to prove that every
solution of the GDAFixedPoint with oracle access _O_ _f_ to _f_ and _∇_ _f_ gives a solution to the initial
HighD-BiSperner instance. So the only thing that remains is to bound the number of queries
to the HighD-BiSperner instance that we need in order to implement the oracle _O_ _f_ . To do
this consider the following definition of _f_ based on an instance _C_ _l_ of HighD-BiSperner from


46


Definition 7.4 with a scaling factor to make sure that the range of the function is [ _−_ 1, 1 ]

### f C l ( x, y ) = [1] ∑ d ( x j − y j ) · α j ( x )

_d_ _[·]_ _j_ = 1


where _α_ _j_ ( _**x**_ ) = _−_ ∑ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _d_ P _**v**_ ( _**x**_ ) _· C_ _l_ _[j]_ [(] _**[v]**_ [)] [, and][ P] _**[v]**_ [ are the coefficients defined in Definition][ 7.3][.]
From the property (C) of the coefficients P _**v**_ we have that to evaluate _a_ _j_ ( _**x**_ ) we only need the

values _C_ _[j]_
_l_ [(] _**[v]**_ [)] [ for] _[ d]_ [ +] [ 1 coefficients] _**[ v]**_ [ and the same coefficients are needed to evaluate] _[ a]_ _[j]_ [(] _**[x]**_ [)] [ for]
every _j_ . This implies that for every ( _**x**_, _**y**_ ) we need _d_ + 1 oracle calls to the instance _C_ _l_ of HighDBiSperner so that _O_ _f_ returns the value of _f_ _C_ _l_ ( _**x**_, _**y**_ ) . If we take the gradient of _f_ _C_ _l_ with respect to
( _**x**_, _**y**_ ) then an identical argument implies that the same set of _d_ + 1 queries to HighD-BiSperner
are needed so that _O_ _f_ returns the value of _∇_ _f_ _C_ _l_ ( _**x**_, _**y**_ ) too. Therefore every query to the oracle _O_ _f_
can be implemented via _d_ + 1 queries to _C_ _l_ . Now we can use Corollary 9.2 to get that the number
of queries that we need in order to solve the GDAFixedPoint with oracle access _O_ _f_ to _f_ and _∇_ _f_
is at least 2 _[d]_ / ( _d_ + 1 ) . Finally observe that the proof of the Theorem 5.1 applies in th the black box
model too. Hence finding solution of GDAFixedPoint in when we have black box access _O_ _f_ to _f_
and _∇_ _f_ is equivalent to finding solutions of LR-LocalMinMax when we have exactly the same
black box access _O_ _f_ to _f_ and _∇_ _f_ . Therefore to find solutions of LR-LocalMinMax with black
box access _O_ _f_ to _f_ and _∇_ _f_ we need at least 2 _[d]_ / ( _d_ + 1 ) queries to _O_ _f_ and the theorem follows by
observing that in our proof the only parameters that depend on _d_ are _L_, _G_, _ε_, and possibly _δ_ but
1/ _δ_ = _O_ ( _[√]_ _L_ / _ε_ ) and hence the dependence of _δ_ can be replaced by dependence on _L_ and _ε_ .

#### **10 Hardness in the Global Regime**


In this section our goal is to prove that the complexity of the problems LocalMinMax and
LocalMin is significantly increased when _ε_, _δ_ lie outside the local regime, in the global regime.
We start with the following theorem where we show that FNP-hardness of LocalMinMax.


**Theorem 10.1.** LocalMinMax _is_ FNP _-hard even when ε is set to any value_ _≤_ 1/384 _, δ is set to any_
_value_ _≥_ 1 _, and even when_ _P_ ( _**A**_, _**b**_ ) = [ 0, 1 ] _[d]_ _, G_ = _√_ _d, L_ = _d, and B_ = _d._


_Proof._ We now present a reduction from 3-SAT(3) to LocalMinMax that proves Theorem 10.1.
First we remind the definition of the problem 3-SAT(3).


**3-SAT(3).**
Input: A boolean CNF-formula _φ_ with boolean variables _x_ 1, . . ., _x_ _n_ such that every clause of _φ_
has at most 3 boolean variables and every boolean variable appears to at most 3 clauses.

Output: An assignment _**x**_ _∈{_ 0, 1 _}_ _[n]_ that satisfies _φ_, or _⊥_ if no such assignment exists.


Given an instance of 3-SAT(3) we first construct a polynomial _P_ _j_ ( _**x**_ ) for each clause _φ_ _j_ as
follows: for each boolean variable _x_ _i_ (there are _n_ boolean variables _x_ _i_ ) we correspond a respective
real-valued variable _x_ _i_ . Then for each clause _φ_ _j_ (there are _m_ such clauses), let _ℓ_ _i_, _ℓ_ _k_, _ℓ_ _m_ denote the
literals participating in _φ_ _j_, _P_ _j_ ( _**x**_ ) = _P_ _ji_ ( _**x**_ ) _·_ _P_ _jk_ ( _**x**_ ) _·_ _P_ _jm_ ( _**x**_ ) where


1 _−_ _x_ _i_ if _ℓ_ _i_ = _x_ _i_
_P_ _ji_ ( _**x**_ ) =
� _x_ _i_ if _ℓ_ _i_ = ~~_x_~~ _i_


47


Then the overall constructed function is


_m_
### f ( x, w, z ) = ∑ P j ( x ) · ( w j − z j ) [2]

_j_ = 1


where each _w_ _j_, _z_ _j_ are additional variables associated with clause _φ_ _j_ . The player that wants to
minimize _f_ controls _**x**_, _**w**_ vectors while the maximizing player controls the _**z**_ variables.


**Lemma 10.2.** _The formula φ admits a satisfying assignment if and only if there exist an_ ( _ε_, _δ_ ) _-local_
_min-max equilibrium of f_ ( _**x**_, _**w**_ ) _with ε_ _≤_ 1/384 _, δ_ = 1 _and_ ( _**x**_, _**w**_ ) _∈_ [ 0, 1 ] _[n]_ [+] [2] _[m]_ _._


_Proof._ Let us assume that there exists a satisfying assignment. Given such a satisfying assignment
we will construct (( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ), _**z**_ _[⋆]_ ) that is a ( 0, 1 ) -local min-max equilibrium of _f_ . We set each
variable _x_ _i_ _[⋆]_ [≜] [1 if and only if the respective boolean variable is true. Observe that this implies that]
_P_ _j_ ( _**x**_ _[⋆]_ ) = 0 for all _j_, meaning that the strategy profile (( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ), _**z**_ _[⋆]_ ) is a global Nash equilibrium
no matter the values of _**w**_ _[⋆]_, _**z**_ _[⋆]_ .
On the opposite direction, let us assume that there exists an ( _ε_, _δ_ ) -local min-max equilibrium
of _f_ with _ε_ = 1/384 and _δ_ = 1. In this case we first prove that for each _j_ = 1, . . ., _m_


_P_ _j_ ( _**x**_ _[⋆]_ ) _≤_ 16 _·_ _ε_ .


Fix any clause _j_ . In case ��� _w_ _⋆_ _j_ _[−]_ _[z]_ _[⋆]_ _j_ ��� _≥_ 1/4 then the minimizing player can further decrease _f_ by at

least _P_ _j_ ( _x_ ) /16 by setting _w_ _[⋆]_ _j_ [≜] _[z]_ _[⋆]_ _j_ [. On the other hand in case] ��� _w_ _⋆_ _j_ _[−]_ _[z]_ _[⋆]_ _j_ ��� _≤_ 1/4 then the maximizing

player can increase _f_ by at least _P_ _j_ ( _x_ _[⋆]_ ) /16 by moving _z_ _[⋆]_ _j_ [either to 0 or to 1. We remark that both]
of the options are feasible since _δ_ = 1.
Now consider the probability distribution over the boolean assignments where each boolean
variable _x_ _i_ is independently selected to be true with probability _x_ _i_ _[⋆]_ [. Then,]


_⋆_
**P** �clause _φ_ _j_ is not satisfied� = _P_ _j_ ( _**x**_ ) _≤_ 16 _·_ _ε_ = 1/24


Since each _φ_ _j_ shares variables with at most 6 other clauses, the event of _φ_ _j_ not being satisfied
is dependent with at most 6 other events. By the Lovász Local Lemma [EL73], we get that the
probability none of these events occur is positive. As a result, there exists a satisfying assignment.



Hence the formula _φ_ is satisfiable if and only if _f_ has a ( 1/384, 1 ) -local min-max equilibrium
point. What is left to prove the FNP-hardness is to show how we can find a satisfying assignment of _φ_ given an approximate stationary point of _f_ . This can be done using the celebrated
results that provide constructive proofs of the Lovász Local Lemma [Mos09, MT10]. Finally
to conclude the proof observe that since the _f_ that we construct is a polynomial of degree 6
which can efficiently be described as a sum of monomials, we can trivially construct a Turing
machine that computes the values of both _f_ and _∇_ _f_ in the polynomial time in the requested
number of bits accuracy. The constructed function _f_ is _√_ _d_ -Lipschitz and _d_ -smooth, where _d_ is



number of bits accuracy. The constructed function _f_ is _√_ _d_ -Lipschitz and _d_ -smooth, where _d_ is

the number of variables that is equal to _n_ + 2 _m_ . More precisely since each variable _x_ _i_ participates in at most 3 clauses, the real-valued variable _x_ _i_ appears in at most 3 monomials _P_ _j_ . Thus
_−_ 3 _≤_ _∂_ _f_ ( _∂_ _**x**_, _x_ _**w**_ _i_, _**x**_ ) _≤_ 3. Similarly it is not hard to see that _−_ 2 _≤_ _∂_ _f_ ( _∂_ _**x**_ _w_, _**w**_ _j_, _**x**_ ), _[∂]_ _[f]_ [(] _**[x]**_ _∂_ [,] _z_ _**[w]**_ _j_ [,] _**[x]**_ [)] _≤_ 2. All the

latter imply that _∥∇_ _f_ ( _**x**_, _**w**_, _**z**_ ) _∥_ 2 _≤_ Θ ( _[√]_ _n_ + _m_ ), meaning that _f_ ( _**x**_, _**w**_, _**z**_ ) is Θ ( _n_ + _m_ ) -Lipschitz.


48



_**x**_, _**w**_, _**x**_ ) _∂_ _f_ ( _**x**_, _**w**_, _**x**_ )

_∂x_ _i_ _≤_ 3. Similarly it is not hard to see that _−_ 2 _≤_ _∂w_



_∂_ _**x**_ _w_, _**w**_ _j_, _**x**_ ), _[∂]_ _[f]_ [(] _**[x]**_ _∂_ [,] _z_ _**[w]**_ _j_ [,] _**[x]**_ [)]



_∂_ [,] _z_ _j_ [,] _≤_ 2. All the


Using again the fact that each _x_ _i_ participates in at most 3 monomials _P_ _j_ ( _**x**_ ), we get that all



_∂w_ _j_ _∂_ [,] _z_ [,] _j_ _∈_ [ _−_ 6, 6 ] . Thus the absolute



the terms _[∂]_ [2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [,] _**[z]**_ [)]



_∂_ [(] _**[x]**_ [2] [,] _x_ _**[w]**_ _i_ [,] _**[z]**_ [)], _[∂]_ [2] _[f]_ _∂_ [(] [2] _**[x]**_ _w_ [,] _**[w]**_ [,] _**[z]**_ [)]



_∂_ [(] [2] _**[x]**_ _w_ [,] _**[w]**_ _j_ [,] _**[z]**_ [)], _[∂]_ [2] _[f]_ [(] _∂_ _**[x]**_ [2] [,] _z_ _**[w]**_ _j_ [,] _**[z]**_ [)]




[(] _∂_ _**[x]**_ [2] [,] _z_ _**[w]**_ _j_ [,] _**[z]**_ [)], _[∂]_ [2] _∂_ _[f]_ _x_ [(] _i_ _**[x]**_ _∂_ [,] _**[w]**_ _w_ [,] _j_ _**[z]**_ [)]




_[f]_ [(] _**[x]**_ [,] _**[w]**_ [,] _**[z]**_ [)] _[∂]_ [2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [,] _**[z]**_ [)]

_∂x_ _i_ _∂w_ _j_ [,] _∂x_ _i_ _∂z_ _j_



_∂_ _[f]_ _x_ [(] _i_ _**[x]**_ _∂_ [,] _**[w]**_ _z_ [,] _j_ _**[z]**_ [)], _[∂]_ [2] _∂_ _[f]_ _w_ [(] _**[x]**_ _j_ _∂_ [,] _**[w]**_ _z_ [,] _j_ _**[z]**_ [)]



2
value of each entry of _∇_ [2] _f_ ( _**x**_, _**w**_, _**z**_ ) is bounded by 6 and thus �� _∇_ _f_ ( _**x**_, _**w**_, _**z**_ ) �� 2 _[≤]_ [Θ] [(] _[n]_ [ +] _[ m]_ [)] [,]
which implies the Θ ( _n_ + _m_ ) -smoothness. Therefore our reduction produces a valid instance of
LocalMinMax and hence the theorem follows.


Next we show the FNP-hardness of LocalMin. As we can see there is a gap between Theorem
10.1 and Theorem 10.3. In particular, the FNP-hardness result of LocalMinMax is stronger since
it holds for any _δ_ _≥_ 1 whereas for the FNP-hardness of LocalMin our proof needs _δ_ _≥_ _√_ _d_ when

the rest of the parameters remain the same.



**Theorem 10.3.** LocalMin _is_ FNP _-hard even when ε is set to any value_ _≤_ 1/24 _, δ is set to any value_
_≥_ _√_ _d, and even when_ _P_ ( _**A**_, _**b**_ ) = [ 0, 1 ] _[d]_ _, G_ = _√_ _d, L_ = _d, and B_ = _d._



_d, and even when_ _P_ ( _**A**_, _**b**_ ) = [ 0, 1 ] _[d]_ _, G_ = _√_



_d, L_ = _d, and B_ = _d._



_Proof._ We follow the same proof as in the proof of Theorem 10.1 but we instead set _f_ ( _**x**_ ) =
∑ _[m]_ _j_ = 1 _[P]_ _[j]_ [(] _**[x]**_ [)] [ where] _**[ x]**_ _[ ∈]_ [[] [0, 1] []] _[n]_ [ (the number of variables is] _[ d]_ [ :] [=] _[ n]_ [). We then get that if the initial]
formula is satisfiable then there exist _**x**_ _∈P_ ( _**A**_, _**b**_ ), such that _f_ ( _**x**_ ) = 0. On the other hand if there
exist _**x**_ _∈P_ ( _**A**_, _**b**_ ) such that _f_ ( _**x**_ ) _≤_ 1/24 then the formula is satisfiable due to the Lovász Local
Lemma [EL73]. Therefore the FNP-hardness follows again from the constructive proof of the
Lovász Local Lemma [Mos09, MT10]. Setting _δ_ _≥_ _[√]_ _n_ which equals the diameter of the feasibility
set implies that in case there exists ˆ _**x**_ with _f_ ( _**x**_ ˆ ) = 0 then all ( _ε_, _δ_ ) -LocalMin _**x**_ _[∗]_ must admit value
_f_ ( _**x**_ _[∗]_ ) _≤_ 1/24 and thus a satisfying assignment is implied.


Next we prove a black box lower bound for minimization in the global regime. The proof
of following lower bound illustrates the strength of the SEIC coefficients presented in Section 8.
The next Theorem can also be used to prove the FNP-hardness of LocalMin in the global regime
but with worse Lipschitzness and smoothness parameters than the once at Theorem 10.3 and for
this reason we present both of them.


**Theorem 10.4.** _In the worst case,_ Ω �2 _[d]_ / _d_ � _value/gradient black-box queries are needed to determine a_
( _ε_, _δ_ ) _-_ LocalMin _for functions f_ ( _**x**_ ) : [ 0, 1 ] _[d]_ _→_ [ 0, 1 ] _with G_ = Θ ( _d_ [15] ) _, L_ = Θ ( _d_ [22] ) _, ε_ _<_ 1 _, δ_ = _√_ _d._


_Proof._ The proof is based on the fact that given just _black-box access_ to a boolean formula _φ_ :
_{_ 0, 1 _}_ _[d]_ _�→{_ 0, 1 _}_, at least Ω ( 2 _[d]_ ) queries are needed in order to determine whether _φ_ admits a
satisfying assignment. The term _black-box access_ refers to the fact that the clauses of the formula
are not given and the only way to determine whether a specific boolean assignment is satisfying
is by quering the specific binary string.
Given such a black-box oracle for a satisfying assignment _d_, we construct the function _f_ _φ_ ( _**x**_ ) :

[ 0, 1 ] _[d]_ _�→_ [ 0, 1 ] as follows:


1. for each corner _**v**_ _∈_ _V_ of the [ 0, 1 ] _[d]_ hypercube, i.e. _**v**_ _∈{_ 0, 1 _}_ _[d]_, we set _f_ _φ_ ( _**v**_ ) : = 1 _−_ _φ_ ( _**v**_ ) .


2. for the rest of the points _**x**_ _∈_ [ 0, 1 ] _[d]_ / _V_, _f_ _φ_ ( _**x**_ ) : = ∑ _**v**_ _∈_ _V_ _P_ _**v**_ ( _**x**_ ) _·_ _f_ _φ_ ( _**v**_ ) where _P_ _**v**_ are the coefficients of Definition 8.9.


2
We remind that by Lemma 8.11, we get that �� _∇_ _f_ _φ_ ( _**x**_ ) �� 2 _[≤]_ [Θ] [(] _[d]_ [12] [)] [ and] �� _∇_ _f_ _φ_ ( _**x**_ ) �� 2 _[≤]_ [Θ] [(] _[d]_ [25] [)] [,]
meaning that _f_ _φ_ ( _·_ ) is Θ ( _d_ [12] ) -Lipschitz and Θ ( _d_ [25] ) -smooth. Moreover by Lemma 8.7, for any


49


_**x**_ _∈_ [ 0, 1 ] _[n]_ the set _V_ ( _x_ ) = _{_ _**v**_ _∈_ _V_ : _P_ _**v**_ ( _**x**_ ) _̸_ = 0 _}_ has cardinality at most _d_ + 1, while at the same
time ∑ _**v**_ _∈_ _V_ _P_ _**v**_ ( _**x**_ ) = 1.
In case _φ_ is not satisfiable then _f_ _φ_ ( _**x**_ ) = 1 for all _**x**_ _∈_ [ 0, 1 ] _[d]_ since _f_ _φ_ ( _**v**_ ) = 1 for all _**v**_ _∈_ _V_ . In
case there exists a satisfying assignment _**v**_ _[∗]_ then _f_ _φ_ ( _**v**_ _[∗]_ ) = 0. Since _δ_ _≥_ _√_ _d_ that is the diameter

of [ 0, 1 ] _[d]_, any ( _ε_, _δ_ ) -LocalMin _**x**_ _[∗]_ must have _f_ _φ_ ( _**x**_ ) _≤_ _ε_ _<_ 1. Since _f_ _φ_ ( _**x**_ _[∗]_ ) ≜ ∑ _**v**_ _∈_ _V_ ( _**x**_ _∗_ ) _P_ _**v**_ ( _**x**_ _[∗]_ ) _·_
_f_ _φ_ ( _**v**_ _[∗]_ ) _<_ 1, there exists at least one vertex ˆ _**v**_ _∈_ _V_ ( _**x**_ ) with _f_ _φ_ ( _**v**_ ˆ ) = 0, meaning that _φ_ ( _**v**_ _[∗]_ ) = 1.
As a result, given an ( _ε_, _δ_ ) -LocalMin _**x**_ _[∗]_ with _f_ _φ_ ( _**x**_ _[∗]_ ) _<_ 1, we can find a satisfying ˆ _**v**_ by querying
_φ_ ( _**v**_ ) for each vertex _**v**_ _∈_ _V_ ( _**x**_ _[∗]_ ) . Since _|_ _V_ ( _**x**_ _[∗]_ ) _| ≤_ _d_ + 1, this will take at most _d_ + 1 additional
queries.
Up next, we argue that in case an ( _ε_, _δ_ ) -LocalMin could be determined with less than
_O_ ( 2 _[d]_ / _d_ ) value/gradient queries, then determining whether _φ_ admits a satisfying assignment
could be done with less that _O_ ( 2 _[d]_ ) queries on _φ_ (the latter is obviously impossible). Notice that
any value/gradient query both _f_ _φ_ ( _**x**_ ) and _∇_ _f_ _φ_ ( _**x**_ ) can be computed by querying the value _f_ _φ_ ( _**v**_ )
of the vertices _**v**_ _∈_ _V_ ( _**x**_ ) . Since _|_ _V_ ( _**x**_ ) _| ≤_ _d_ + 1, any value/gradient query of _f_ _φ_ can be simulated
by _d_ + 1 queries on _φ_ .

#### **Acknowledgements**


This work was supported by NSF Awards IIS-1741137, CCF-1617730 and CCF-1901292, by a
Simons Investigator Award, by the DOE PhILMs project (No. DE-AC05-76RL01830), and by the
DARPA award HR00111990021. M.Z. was also supported by Google Ph.D. Fellowship. S.S. was
supported by NRF 2018 Fellowship NRF-NRFF2018-07.

#### **References**


[AAZB [+] 17] Naman Agarwal, Zeyuan Allen-Zhu, Brian Bullins, Elad Hazan, and Tengyu Ma.
Finding approximate local minima faster than gradient descent. In _Proceedings of_
_the 49th Annual ACM SIGACT Symposium on Theory of Computing_, pages 1195–1199,
2017.


[ACB17] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In _Proceedings of the 34th International Conference on Machine_
_Learning-Volume 70_, pages 214–223, 2017.


[Adl13] Ilan Adler. The equivalence of linear programs and zero-sum games. _International_
_Journal of Game Theory_, 42(1):165–177, 2013.


[ADLH19] Leonard Adolphs, Hadi Daneshmand, Aurelien Lucchi, and Thomas Hofmann. Local saddle point optimization: A curvature exploitation approach. In _The 22nd_
_International Conference on Artificial Intelligence and Statistics_, pages 486–495, 2019.


[ADSG19] Mohammad Alkousa, Darina Dvinskikh, Fedor Stonyakin, and Alexander Gasnikov. Accelerated methods for composite non-bilinear saddle point problem. _arXiv_
_preprint arXiv:1906.03620_, 2019.


[ALW19] Jacob Abernethy, Kevin A Lai, and Andre Wibisono. Last-iterate convergence rates
for min-max optimization. _arXiv preprint arXiv:1906.02027_, 2019.


50


[AMLJG20] Waïss Azizian, Ioannis Mitliagkas, Simon Lacoste-Julien, and Gauthier Gidel. A
tight and unified analysis of extragradient for a whole spectrum of differentiable
games. In _Proceedings of the 23rd International Conference on Artificial Intelligence and_
_Statistics (AISTATS)_, 2020.


[BCB12] Sébastien Bubeck and Nicolo Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. _Foundations and Trends in Machine Learning_,
5(1):1–122, 2012.


[BCE [+] 95] Paul Beame, Stephen A. Cook, Jeff Edmonds, Russell Impagliazzo, and Toniann
Pitassi. The relative complexity of NP search problems. In _Proceedings of the Twenty-_
_Seventh Annual ACM Symposium on Theory of Computing, 29 May-1 June 1995, Las_
_Vegas, Nevada, USA_, pages 303–314, 1995.


[BIQ [+] 17] Aleksandrs Belovs, Gábor Ivanyos, Youming Qiao, Miklos Santha, and Siyi Yang.
On the polynomial parity argument complexity of the combinatorial nullstellensatz.
In _Proceedings of the 32nd Computational Complexity Conference_, pages 1–24, 2017.


[Bla56] David Blackwell. An analog of the minimax theorem for vector payoffs. _Pacific J._
_Math._, 6(1):1–8, 1956.


[BPR15] Nir Bitansky, Omer Paneth, and Alon Rosen. On the cryptographic hardness of
finding a nash equilibrium. In _Proceedings of the 56th Annual Symposium on Founda-_
_tions of Computer Science, (FOCS)_, 2015.


[Bre76] Richard P Brent. Fast multiple-precision evaluation of elementary functions. _Journal_
_of the ACM (JACM)_, 23(2):242–251, 1976.


[CBL06] Nikolo Cesa-Bianchi and Gabor Lugosi. _Prediction, Learning, and Games_ . Cambridge
University Press, 2006.


[CDT09] Xi Chen, Xiaotie Deng, and Shang-Hua Teng. Settling the complexity of computing
two-player nash equilibria. _Journal of the ACM (JACM)_, 56(3):1–57, 2009.


[CPY17] Xi Chen, Dimitris Paparas, and Mihalis Yannakakis. The complexity of nonmonotone markets. _J. ACM_, 64(3):20:1–20:56, 2017.


[Dan51] George B. Dantzig. A proof of the equivalence of the programming problem and
the game problem. In _Koopmans, T. C., editor(s), Activity Analysis of Production and_
_Allocation_ . Wiley, New York, 1951.


[Das13] Constantinos Daskalakis. On the complexity of approximating a nash equilibrium.
_ACM Transactions on Algorithms (TALG)_, 9(3):1–35, 2013.


[Das18] Constantinos Daskalakis. Equilibria, Fixed Points, and Computational Complexity

       - Nevanlinna Prize Lecture. _Proceedings of the International Congress of Mathematicians_
_(ICM)_, 1:147–209, 2018.


[DFS20] Argyrios Deligkas, John Fearnley, and Rahul Savani. Tree polymatrix games are
ppad-hard. _CoRR_, abs/2002.12119, 2020.


51


[DGP09] Constantinos Daskalakis, Paul W Goldberg, and Christos H Papadimitriou. The
complexity of computing a nash equilibrium. _SIAM Journal on Computing_, 39(1):195–
259, 2009.


[DHS11] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for
online learning and stochastic optimization. _Journal of machine learning research_,
12(Jul):2121–2159, 2011.


[DISZ18] Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng.
Training gans with optimism. In _International Conference on Learning Representations_
_(ICLR 2018)_, 2018.


[DP11] Constantinos Daskalakis and Christos Papadimitriou. Continuous local search. In
_Proceedings of the twenty-second annual ACM-SIAM symposium on Discrete Algorithms_,
pages 790–804. SIAM, 2011.


[DP18] Constantinos Daskalakis and Ioannis Panageas. The limit points of (optimistic) gradient descent in min-max optimization. In _Advances in Neural Information Processing_
_Systems_, pages 9236–9246, 2018.


[DP19] Constantinos Daskalakis and Ioannis Panageas. Last-iterate convergence: Zero-sum
games and constrained min-max optimization. _Innovations in Theoretical Computer_
_Science_, 2019.


[DTZ18] Constantinos Daskalakis, Christos Tzamos, and Manolis Zampetakis. A converse
to banach’s fixed point theorem and its CLS-completeness. In _Proceedings of the 50th_
_Annual ACM SIGACT Symposium on Theory of Computing (STOC)_, 2018.


[EL73] Paul Erd˝os and László Lovász. Problems and results on 3-chromatic hypergraphs
and some related questions. In _Colloquia Mathematica Societatis Janos Bolyai 10. Infinite_
_and Finite Sets, Keszthely (Hungary)_ . Citeseer, 1973.


[EY10] Kousha Etessami and Mihalis Yannakakis. On the complexity of nash equilibria
and other fixed points. _SIAM Journal on Computing_, 39(6):2531–2597, 2010.


[FG18] Aris Filos-Ratsikas and Paul W. Goldberg. Consensus halving is ppa-complete.
In _Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing_
_(STOC)_, 2018.


[FG19] Aris Filos-Ratsikas and Paul W. Goldberg. The complexity of splitting necklaces
and bisecting ham sandwiches. In _Proceedings of the 51st Annual ACM SIGACT_
_Symposium on Theory of Computing (STOC)_, 2019.


[FP07] Francisco Facchinei and Jong-Shi Pang. _Finite-dimensional variational inequalities and_
_complementarity problems_ . Springer Science & Business Media, 2007.


[FPT04] Alex Fabrikant, Christos H. Papadimitriou, and Kunal Talwar. The complexity of
pure nash equilibria. In _Proceedings of the 36th Annual ACM Symposium on Theory of_
_Computing (STOC)_, 2004.


52


[FRHSZ20a] Aris Filos-Ratsikas, Alexandros Hollender, Katerina Sotiraki, and Manolis Zampetakis. Consenus-halving: Does it ever get easier? _arXiv preprint arXiv:2002.11437_,
2020.


[FRHSZ20b] Aris Filos-Ratsikas, Alexandros Hollender, Katerina Sotiraki, and Manolis Zampetakis. A topological characterization of modulo-p arguments and implications
for necklace splitting. _arXiv preprint arXiv:2003.11974_, 2020.


[GH19] Paul W. Goldberg and Alexandros Hollender. The hairy ball problem is ppadcomplete. In _Proceedings of the 46th International Colloquium on Automata, Languages,_
_and Programming (ICALP)_, 2019.


[GHP [+] 19] Gauthier Gidel, Reyhane Askari Hemmat, Mohammad Pezeshki, Rémi Le Priol,
Gabriel Huang, Simon Lacoste-Julien, and Ioannis Mitliagkas. Negative momentum for improved game dynamics. In _The 22nd International Conference on Artificial_
_Intelligence and Statistics_, pages 1802–1811, 2019.


[GKSZ19] Mika Göös, Pritish Kamath, Katerina Sotiraki, and Manolis Zampetakis. On the
complexity of modulo-q arguments and the chevalley-warning theorem. _arXiv_
_preprint arXiv:1912.04467_, 2019.


[Goo16] Ian Goodfellow. Nips 2016 tutorial: Generative adversarial networks. _arXiv preprint_
_arXiv:1701.00160_, 2016.


[GPDO20] Noah Golowich, Sarath Pattathil, Constantinos Daskalakis, and Asuman E.
Ozdaglar. Last iterate is slower than averaged iterate in smooth convex-concave
saddle point problems. _CoRR_, abs/2002.00057, 2020.


[GPM [+] 14] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative Adversarial Nets.
In _Advances in Neural Information Processing Systems 27: Annual Conference on Neural_
_Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada_,
pages 2672–2680, 2014.


[HA18] Erfan Yazdandoost Hamedani and Necdet Serhat Aybat. A primal-dual algorithm
for general convex-concave saddle point problems. _arXiv preprint arXiv:1803.01401_,
2018.


[Haz16] Elad Hazan. Introduction to online convex optimization. _Foundations and Trends in_
_Optimization_, 2(3-4):157–325, 2016.


[HPV89] M. D. Hirsch, C. H. Papadimitriou, and S. A. Vavasis. Exponential lower bounds
for finding brouwer fixed points. _Journal of Complexity_, 5:379–416, 1989.


[Jeˇr16] Emil Jeˇrábek. Integer factoring and modular square roots. _Journal of Computer and_
_System Sciences_, 82(2):380–394, 2016.


[JGN [+] 17] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How
to escape saddle points efficiently. In _Proceedings of the 34th International Conference_
_on Machine Learning-Volume 70_, pages 1724–1732. JMLR. org, 2017.


53


[JNJ19] Chi Jin, Praneeth Netrapalli, and Michael I Jordan. What is local optimality in
nonconvex-nonconcave minimax optimization? _arXiv preprint arXiv:1902.00618_,
2019.


[JPY88] David S Johnson, Christos H Papadimitriou, and Mihalis Yannakakis. How easy is
local search? _Journal of computer and system sciences_, 37(1):79–100, 1988.


[KB14] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization.
_arXiv preprint arXiv:1412.6980_, 2014.


[KM18] Pravesh K. Kothari and Ruta Mehta. Sum-of-squares meets Nash: lower bounds for
finding any equilibrium. In _Proceedings of the 50th Annual ACM SIGACT Symposium_
_on Theory of Computing (STOC)_, 2018.


[KM19] Weiwei Kong and Renato DC Monteiro. An accelerated inexact proximal
point method for solving nonconvex-concave min-max problems. _arXiv preprint_
_arXiv:1905.13433_, 2019.


[Kor76] GM Korpelevich. The extragradient method for finding saddle points and other
problems. _Matecon_, 12:747–756, 1976.


[LJJ19] Tianyi Lin, Chi Jin, and Michael I Jordan. On gradient descent ascent for nonconvexconcave minimax problems. _arXiv preprint arXiv:1906.00331_, 2019.


[LJJ20] Tianyi Lin, Chi Jin, and Michael Jordan. Near-optimal algorithms for minimax
optimization. _arXiv preprint arXiv:2002.02417_, 2020.


[LPP [+] 19] Jason D. Lee, Ioannis Panageas, Georgios Piliouras, Max Simchowitz, Michael I.
Jordan, and Benjamin Recht. First-order methods almost always avoid strict saddle
points. _Math. Program._, 176(1-2):311–337, 2019.


[LS19] Tengyuan Liang and James Stokes. Interaction matters: A note on non-asymptotic
local convergence of generative adversarial networks. In _The 22nd International Con-_
_ference on Artificial Intelligence and Statistics_, pages 907–915, 2019.


[LTHC19] Songtao Lu, Ioannis Tsaknakis, Mingyi Hong, and Yongxin Chen. Hybrid block
successive approximation for one-sided non-convex min-max problems: algorithms
and applications. _arXiv preprint arXiv:1902.08294_, 2019.


[Meh14] Ruta Mehta. Constant rank bimatrix games are ppad-hard. In _Proceedings of the 46th_
_Symposium on Theory of Computing (STOC)_, 2014.


[MGN18] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods
for gans do actually converge? In _International Conference on Machine Learning_, pages
3481–3490, 2018.


[MMS [+] 18] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and
Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In
_International Conference on Learning Representations_, 2018.


54


[MOP19] Aryan Mokhtari, Asuman Ozdaglar, and Sarath Pattathil. A unified analysis of
extra-gradient and optimistic gradient methods for saddle point problems: Proximal point approach. _arXiv preprint arXiv:1901.08511_, 2019.


[Mos09] Robin A Moser. A constructive proof of the lovász local lemma. In _Proceedings of the_
_forty-first annual ACM symposium on Theory of computing_, pages 343–350, 2009.


[MP89] N Meggido and CH Papadimitriou. A note on total functions, existence theorems,
and computational complexity. Technical report, Tech. report, IBM, 1989.


[MPP18] Panayotis Mertikopoulos, Christos H. Papadimitriou, and Georgios Piliouras. Cycles in adversarial regularized learning. In _Proceedings of the Twenty-Ninth Annual_
_ACM-SIAM Symposium on Discrete Algorithms (SODA)_, 2018.


[MPPSD16] Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein. Unrolled generative
adversarial networks. _arXiv preprint arXiv:1611.02163_, 2016.


[MR18] Eric Mazumdar and Lillian J Ratliff. On the convergence of gradient-based learning
in continuous games. _arXiv preprint arXiv:1804.05464_, 2018.


[MSV20] Oren Mangoubi, Sushant Sachdeva, and Nisheeth K Vishnoi. A provably convergent and practical algorithm for min-max optimization with applications to gans.
_arXiv preprint arXiv:2006.12376_, 2020.


[MT10] Robin A Moser and Gábor Tardos. A constructive proof of the general lovász local
lemma. _Journal of the ACM (JACM)_, 57(2):1–15, 2010.


[MV20] Oren Mangoubi and Nisheeth K Vishnoi. A second-order equilibrium in
nonconvex-nonconcave min-max optimization: Existence and algorithm. _arXiv_
_preprint arXiv:2006.12363_, 2020.


[Nem04] Arkadi Nemirovski. Interior point polynomial time methods in convex programming. _Lecture notes_, 2004.


[NSH [+] 19] Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, Jason D Lee, and Meisam Razaviyayn. Solving a class of non-convex min-max games using iterative first order
methods. In _Advances in Neural Information Processing Systems_, pages 14905–14916,
2019.


[NY83] Arkadi˘ı Semenovich Nemirovsky and David Borisovich Yudin. _Problem complexity_
_and method efficiency in optimization._ Chichester: Wiley, 1983.


[OX19] Yuyuan Ouyang and Yangyang Xu. Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems. _Mathematical Programming_,
pages 1–35, 2019.


[Pap94a] C Papadimitriou. _Computational Complexity_ . Addison Welsey, 1994.


[Pap94b] Christos H Papadimitriou. On the complexity of the parity argument and other
inefficient proofs of existence. _Journal of Computer and system Sciences_, 48(3):498–532,
1994.


55


[RKK18] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and
beyond. In _Proceedings of the 6th International Conference on Learning Representations_
_(ICLR)_, 2018.


[RLLY18] Hassan Rafique, Mingrui Liu, Qihang Lin, and Tianbao Yang. Non-convex minmax optimization: Provable algorithms and applications in machine learning. _arXiv_
_preprint arXiv:1810.02060_, 2018.


[Ros65] J Ben Rosen. Existence and uniqueness of equilibrium points for concave n-person
games. _Econometrica: Journal of the Econometric Society_, pages 520–534, 1965.


[Rub15] Aviad Rubinstein. Inapproximability of nash equilibrium. In _Proceedings of the Forty-_
_Seventh Annual ACM on Symposium on Theory of Computing (STOC)_, 2015.


[Rub16] Aviad Rubinstein. Settling the complexity of computing approximate two-player
nash equilibria. In _2016 IEEE 57th Annual Symposium on Foundations of Computer_
_Science (FOCS)_, pages 258–265. IEEE, 2016.


[SS12] Shai Shalev-Shwartz. Online learning and online convex optimization. _Foundations_
_and Trends in Machine Learning_, 4(2):107–194, 2012.


[SSBD14] Shai Shalev-Shwartz and Shai Ben-David. _Understanding machine learning: From_
_theory to algorithms_ . Cambridge university press, 2014.


[SY91] Alejandro A. Schäffer and Mihalis Yannakakis. Simple local search problems that
are hard to solve. _SIAM J. Comput._, 20(1):56–87, 1991.


[SZZ18] Katerina Sotiraki, Manolis Zampetakis, and Giorgos Zirdelis. Ppp-completeness
with connections to cryptography. In _Proceddings of the 59th IEEE Annual Symposium_
_on Foundations of Computer Science ( FOCS)_, 2018.


[TJNO19] Kiran K Thekumparampil, Prateek Jain, Praneeth Netrapalli, and Sewoong Oh. Efficient algorithms for smooth minimax optimization. In _Advances in Neural Information_
_Processing Systems_, pages 12659–12670, 2019.


[vN28] John von Neumann. Zur Theorie der Gesellschaftsspiele. In _Math. Ann._, pages
295–320, 1928.


[VY11] Vijay V. Vazirani and Mihalis Yannakakis. Market equilibrium under separable,
piecewise-linear, concave utilities. _J. ACM_, 58(3):10:1–10:25, 2011.


[WZB19] Yuanhao Wang, Guodong Zhang, and Jimmy Ba. On solving minimax optimization locally: A follow-the-ridge approach. In _International Conference on Learning_
_Representations_, 2019.


[Zha19] Renbo Zhao. Optimal algorithms for stochastic three-composite convex-concave
saddle point problems. _arXiv preprint arXiv:1903.01687_, 2019.


56


#### **A Proof of Theorem 4.1**

We first remind the definition of the 3-SAT(3) problem that we will use for our reduction.


**3-SAT(3).**
Input: A boolean CNF-formula _φ_ with boolean variables _x_ 1, . . ., _x_ _n_ such that every clause of _φ_
has at most 3 boolean variables and every boolean variable appears to at most 3 clauses.

Output: An assignment _**x**_ _∈{_ 0, 1 _}_ _[n]_ that satisfies _φ_, or _⊥_ if no such assignment exists.


It is well known that 3-SAT(3) is FNP-complete, for details see §9.2 of [Pap94a]. To prove
Theorem 4.1, we reduce 3-SAT(3) to _ε_ -StationaryPoint.


Given an instance of 3-SAT(3) we construct the function _f_ : [ 0, 1 ] _[n]_ [+] _[m]_ _→_ [ 0, 1 ], where _m_ is the
number of clauses of _φ_ . For each literal _x_ _i_ we assign a real-valued variable which by abuse of
notation we also denote _x_ _i_ and it would be clear from the context whether we refer to the literal
or the real-valued variable. Then for each clause _φ_ _j_ of _φ_, we construct a polynomial _P_ _j_ ( _x_ ) as
follows: if _ℓ_ _i_, _ℓ_ _k_, _ℓ_ _m_ are the literals participating in _φ_ _j_, then _P_ _j_ ( _x_ ) = _P_ _ji_ ( _x_ ) _·_ _P_ _jk_ ( _x_ ) _·_ _P_ _jm_ ( _x_ ) where


1 _−_ _x_ _i_ if _ℓ_ _i_ = _x_ _i_
_P_ _ji_ ( _x_ ) =
� _x_ _i_ if _ℓ_ _i_ = ~~_x_~~ _i_


The overall constructed function is _f_ ( _**x**_, _**w**_ ) = ∑ _[m]_ _j_ = 1 _[w]_ _[j]_ _[·]_ _[ P]_ _[j]_ [(] _**[x]**_ [)] [, where each] _[ w]_ _[j]_ [is an additional]




[(] _**[x]**_ [,] _**[w]**_ [)] [(] _**[x]**_ [,] _**[w]**_ [)]

_∂w_ _j_ _≤_ 1 and _−_ 3 _≤_ _[∂]_ _[f]_ _∂x_ _i_




[(] _**[x]**_ [,] _**[w]**_ [)]
variable associated with clause _φ_ _j_ . Notice that 0 _≤_ _[∂]_ _[f]_ _∂w_



_∂x_ [,] _i_ _≤_ 3 since the



boolean variable _x_ _i_ participates in at most 3 clauses. As a result, _∥∇_ _f_ ( _**x**_, _**w**_ ) _∥_ 2 _≤_ Θ ( _[√]_ _n_ + _m_ ),
meaning that _f_ ( _**x**_, _**w**_ ) is _G_ -Lipschitz with _G_ = Θ ( _[√]_ _n_ + _m_ ) . Also notice that all the entries of
_∇_ [2] _f_ ( _**x**_, _**w**_ ), i.e. _[∂]_ [2] _[f]_ _∂_ [(] [2] _**[x]**_ _x_ [,] _i_ _**[w]**_ [)] = _[∂]_ [2] _∂_ _[f]_ [2] [(] _w_ _**[x]**_ [,] _j_ _**[w]**_ [)], _[∂]_ _∂_ [2] _x_ _[f]_ _i_ [(] _∂_ _**[x]**_ [,] _w_ _**[w]**_ _j_ [)] [,] _[∂]_ _∂_ [2] _x_ _[f]_ _i_ [(] _∂_ _**[x]**_ [,] _x_ _**[w]**_ _m_ [)] [,] _[∂]_ _∂_ [2] _w_ _[f]_ _k_ [(] _∂_ _**[x]**_ [,] _**[w]**_ _w_ _j_ [)] _[∈]_ [[] _[−]_ [3, 3] []] [. As a result,] �� _∇_ 2 _f_ ( _**x**_, _**w**_ ) �� 2 _[≤]_

Θ ( _n_ + _m_ ), meaning that _f_ ( _**x**_, _**w**_ ) is _L_ -smooth with _L_ = Θ ( _n_ + _m_ ) .


**Lemma A.1.** _There exists a satisfying assignment for the clauses φ_ 1, . . ., _φ_ _m_ _if and only if there solution_
_of the constructed_ StationaryPoint _with ε_ = 1/24 _a admits solution_ ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) _∈_ [ 0, 1 ] _[n]_ [+] _[m]_ _such that_
_∥∇_ _f_ ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) _∥_ 2 _<_ 1/24 _._


_Proof._ By the definition of StationaryPoint, in case there exists a pair of points ( _**x**_ ˆ, ˆ _**w**_ ) _∈_ [ 0, 1 ] _[n]_ [+] _[m]_


ˆ
with _∥∇_ _f_ ( _**x**_, ˆ _**w**_ ) _∥_ 2 _<_ _ε_ /2 = 1/48, then a pair of points ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) with _∥∇_ _f_ ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) _∥_ 2 _<_ _ε_ = 1/24
must be returned. In case _∥∇_ _f_ ( _**x**_, _**w**_ ) _∥_ 2 _>_ _ε_ = 1/24 for all ( _**x**_, _**w**_ ) _∈_ [ 0, 1 ] _[n]_ [+] _[m]_, the null symbol _⊥_
is returned.
Let us assume that there exists a satisfying assignment of _φ_ . Consider the solution ( _**x**_ ˆ, ˆ _**w**_ )
constructed as follows: each variable ˆ _x_ _i_ is set to 1 iff the respective boolean variable is true and

ˆ
_w_ _j_ = 0 for all _j_ = 1, . . ., _m_ . Since the assignment satisfies the CNF-formula _φ_, there exists at
least one true literal in each clause _φ_ _j_ which means that _P_ _j_ ( _x_ ) = 0 for all _j_ = 1, . . ., _m_ . As a




_[f]_ [(] _**[x]**_ [,] _**[w]**_ [)] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [)]

= _[∂]_ [2]
_∂_ [2] _x_ _i_ _∂_ [2] _w_



_∂_ _[f]_ [2] [(] _w_ _**[x]**_ [,] _j_ _**[w]**_ [)], _[∂]_ _∂_ [2] _x_ _[f]_ _i_ [(] _∂_ _**[x]**_ [,] _w_ _**[w]**_ _j_ [)]




[2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [)] _[∂]_ [2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [)]

_∂x_ _i_ _∂w_ _j_ [,] _∂x_ _i_ _∂x_ _m_




[2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [)] _[∂]_ [2] _[f]_ [(] _**[x]**_ [,] _**[w]**_ [)]

_∂x_ _i_ _∂x_ _m_ [,] _∂w_ _k_ _∂w_




_**[x]**_ [,] _**[w]**_ 2

_∂w_ _k_ _∂w_ _j_ _[∈]_ [[] _[−]_ [3, 3] []] [. As a result,] �� _∇_ _f_ ( _**x**_, _**w**_ ) �� 2 _[≤]_



result _[∂]_ _[f]_ [(] _**[x]**_ [ˆ] [,][ ˆ] _**[w]**_ [)]




[(] _**[x]**_ [ˆ] [,][ ˆ] _**[w]**_ [)] ˆ [(] _**[x]**_ [ˆ] [,][ ˆ] _**[w]**_ [)]

_∂w_ _j_ = _P_ _j_ ( _**x**_ ) = 0 for all _j_ = 1, . . ., _m_ . At the same time, _[∂]_ _[f]_ _∂x_ _i_



_∂x_ [,] _i_ = 0 since ˆ _w_ _j_ = 0 for



ˆ
all _j_ = 1, . . ., _m_ . Overall we have that _∇_ _f_ ( _**x**_, ˆ _**w**_ ) = 0 _<_ 1/48 = _ε_ /2. As a result, the constructed
StationaryPoint instance must return a solution ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) with _∥∇_ _f_ ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) _∥_ 2 _<_ 24 [1] [=] _[ ε]_ [.]

On the opposite direction, the existence of a pair of points ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) with _∥∇_ _f_ ( _**x**_ _[⋆]_, _**w**_ _[⋆]_ ) _∥_ 2 _<_
1/24 implies _P_ _j_ ( _**x**_ _[∗]_ ) _<_ 1/24 for all _j_ = 1 . . . _m_ . Consider the probability distribution over the
boolean assignments in which _each boolean variable x_ _i_ _is independently selected to be true with proba-_
_bility x_ _i_ _[⋆]_ _[.]_ [ Then,]

_⋆_
**P** �clause _φ_ _j_ is not satisfied� = _P_ _j_ ( _**x**_ ) _<_ 1/24


57


Since _φ_ _j_ shares variables with at most 6 other clauses, the bad event of _φ_ _j_ not being satisfied is
dependent with at most 6 other bad events. By Lovász Local Lemma [EL73], we get that the
probability none of the events occurs is positive. As a result, there exists a satisfying assignment.


Using Lemma A.1 we can conclude that _φ_ is satisfiable if and only if _f_ has a 1/24-approximate
stationary point. What is left to prove the FNP-hardness is to show how we can find a satisfying assignment of _φ_ given an approximate stationary point of _f_ . This can be done using the
celebrated results that provide constructive proofs of the Lovász Local Lemma [Mos09, MT10].

Finally, we remind that the constructed function _f_ is Θ _√_ _d_ -Lipschitz and Θ ( _d_ ) -smooth, where
� �

_d_ is the number of variables that is equal to _n_ + _m_ .

#### **B Missing Proofs from Section 5**


In this section we give proofs for the statements presented in Section 5. These statements establish
the totality and inclusion to PPAD of LR-LocalMinMax and GDAFixedPoint.


**B.1** **Proof of Theorem 5.1**


We start with establishing claim “1.” in the statement of the theorem. It will be clear that our
proof will provide a polynomial-time reduction from LR-LocalMinMax to GDAFixedPoint.
Suppose that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an _α_ -approximate fixed point of _F_ _GDA_, where _α_ is the specified in the
theorem statement function of _δ_, _G_ and _L_ . To simplify our proof, we abuse notation and define
_f_ ( _**x**_ ) ≜ _f_ ( _**x**_, _**y**_ _[⋆]_ ), _∇_ _f_ ( _**x**_ ) ≜ _∇_ _x_ _f_ ( _**x**_, _**y**_ _[⋆]_ ), _K_ ≜ _{_ _**x**_ _|_ ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) _}_ and ˆ _**x**_ ≜ Π _K_ ( _**x**_ _[⋆]_ _−∇_ _f_ ( _**x**_ _[⋆]_ )) .

ˆ
Because ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an _α_ -approximate fixed point of _F_ _FDA_, it follows that _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥_ 2 _<_ _α_ .


**Claim B.1.** _⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _⟩_ _<_ ( _G_ + _δ_ + _α_ ) _·_ _α_, _for all_ _**x**_ _∈_ _K_ _∩_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ) _._


_Proof._ Using the fact that ˆ _**x**_ = Π _K_ ( _**x**_ _[⋆]_ _−∇_ _f_ ( _**x**_ _[⋆]_ )) and that _K_ is a convex set we can apply Theorem
1.5.5 (b) of [FP07] to get that


ˆ ˆ
_⟨_ _**x**_ _[⋆]_ _−∇_ _f_ ( _**x**_ _[⋆]_ ) _−_ _**x**_, _**x**_ _−_ _**x**_ _⟩≤_ 0, _∀_ _**x**_ _∈_ _K_ . (B.1)


Next, we do some simple algebra to get that, for all _**x**_ _∈_ _K_ _∩_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ),


_⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _⟩_ = _⟨_ _**x**_ _[⋆]_ _−∇_ _f_ ( _**x**_ _[⋆]_ ) _−_ _**x**_ ˆ, _**x**_ _−_ _**x**_ ˆ _⟩_ + _⟨_ _**x**_ _−_ _**x**_ ˆ _−∇_ _f_ ( _**x**_ _[⋆]_ ), ˆ _**x**_ _−_ _**x**_ _[⋆]_ _⟩_


(B.1) ˆ
_≤⟨_ _**x**_ _−_ _**x**_ _−∇_ _f_ ( _**x**_ _[⋆]_ ), ˆ _**x**_ _−_ _**x**_ _[⋆]_ _⟩_


ˆ ˆ
_≤_ ( _∥_ _**x**_ _−_ _**x**_ _∥_ 2 + _∥∇_ _f_ ( _**x**_ _[⋆]_ ) _∥_ 2 ) _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥_ 2 _<_ ( _G_ + _δ_ + _α_ ) _·_ _α_,


where the second to last inequality follows from Cauchy–Schwarz inequality and the triangle
inequality, and the last inequality follows from the triangle inequality and the following facts: (1)
_∥_ _**x**_ _[⋆]_ _−_ _**x**_ ˆ _∥_ 2 _<_ _α_, (2) _**x**_ _∈_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ), and (3) _∥∇_ _f_ ( _**x**_, _**y**_ ) _∥_ 2 _≤_ _G_ for all ( _**x**_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) .


For all _**x**_ _∈_ _K_ _∩_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ), from the _L_ -smoothness of _f_ we have that


_|_ _f_ ( _**x**_ ) _−_ ( _f_ ( _**x**_ _[⋆]_ ) + _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _−_ _**x**_ _[⋆]_ _⟩_ ) _| ≤_ _[L]_ 2 [.] (B.2)

2 _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2]


We distinguish two cases:


58


1. _f_ ( _**x**_ _[⋆]_ ) _≤_ _f_ ( _**x**_ ) : In this case we stop, remembering that


_f_ ( _**x**_ _[⋆]_ ) _≤_ _f_ ( _**x**_ ) . (B.3)


2. _f_ ( _**x**_ _[⋆]_ ) _>_ _f_ ( _**x**_ ) : In this case, we consider two further sub-cases:


(a) _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _−_ _**x**_ _[⋆]_ _⟩≥_ 0: in this sub-case, Eq (B.2) gives


_f_ ( _**x**_ _[⋆]_ ) _−_ _f_ ( _**x**_ ) + _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _−_ _**x**_ _[⋆]_ _⟩≤_ _[L]_ 2

2 _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2]


Thus



_f_ ( _**x**_ _[⋆]_ ) _≤_ _f_ ( _**x**_ ) + _[L]_




_[L]_ _[L]_

2 _[≤]_ _[f]_ [ (] _**[x]**_ [) +]
2 _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2] 2



2 _[δ]_ [2] _[ <]_ _[ f]_ [ (] _**[x]**_ [) +] _[ ε]_ [,] (B.4)



where for the last inequality we used that _**x**_ _∈_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ), and that _δ_ _<_ _[√]_


(b) _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _−_ _**x**_ _[⋆]_ _⟩_ _<_ 0: in this sub-case, Eq (B.2) gives


_f_ ( _**x**_ _[⋆]_ ) _−_ _f_ ( _**x**_ ) _−⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _⟩≤_ _[L]_ 2 [.]

2 _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2]


Thus


_f_ ( _**x**_ _[⋆]_ ) _≤_ _f_ ( _**x**_ ) + _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _⟩_ + _[L]_ 2

2 _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2]

_≤_ _f_ ( _**x**_ ) + _⟨∇_ _f_ ( _**x**_ _[∗]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _⟩_ + _[L]_

2 _[·]_ _[ δ]_ [2]



2 _ε_ / _L_ .



_<_ _f_ ( _**x**_ ) + ( _G_ + _δ_ + _α_ ) _·_ _α_ + _[L]_

2 _[·]_ _[ δ]_ [2]

_≤_ _f_ ( _**x**_ ) + _ε_, (B.5)


where the second inequality follows from the fact that _**x**_ _∈_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ), the third inequality follows from Claim B.1, and the last inequality follows from the constraints



( _G_ + _δ_ ) [2] + 4 ( _ε_ _−_ _[L]_ 2 _[δ]_ [2] [)] _[−]_ [(] _[G]_ [+] _[δ]_ [)]

2 .



_δ_ _<_ _[√]_



_√_
2 _ε_ / _L_ and _α_ _≤_



In all cases, we get from (B.3), (B.4) and (B.5) that _f_ ( _**x**_ _[⋆]_ ) _<_ _f_ ( _**x**_ ) + _ε_, for all _x_ _∈_ _K_ _∩_ _B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ) .
Thus, lifting our abuse of notation, we get that _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _<_ _f_ ( _**x**_, _**y**_ _[⋆]_ ) + _ε_, for all _**x**_ _∈{_ _**x**_ _|_ _**x**_ _∈_
_B_ _d_ 1 ( _δ_ ; _**x**_ _[⋆]_ ) and ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) _}_ . Using an identical argument we can also show that _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _>_
_f_ ( _**x**_ _[⋆]_, _**y**_ ) _−_ _ε_ for all _**y**_ _∈{_ _**y**_ _|_ _**y**_ _∈_ _B_ _d_ 2 ( _δ_ ; _**y**_ _[⋆]_ ) and ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) _}_ . The first part of the theorem
follows.


Now let us establish claim “2.” in the theorem statement. It will be clear that our proof will
provide a polynomial-time reduction from GDAFixedPoint to LR-LocalMinMax. For the choice
of parameters _ε_ and _δ_ described in the theorem statement, we will show that, if ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an ( _ε_, _δ_ ) 
_⋆_ _⋆_ _⋆_
local min-max equilibrium of _f_, then _∥_ _F_ _GDAx_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _−_ _**x**_ _[⋆]_ _∥_ 2 _<_ _α_ /2 and �� _F_ _GDAy_ ( _**x**_, _**y**_ ) _−_ _**y**_ �� 2 _[<]_
_α_ /2. The second part of the theorem will then follow. We only prove that _∥_ _F_ _GDAx_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _−_ _**x**_ _[⋆]_ _∥_ 2 _<_
_α_ /2, as the argument for _**y**_ _[⋆]_ is identical. In the argument below we abuse notation in the same

ˆ
way we described earlier. With that notation we will show that _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥_ 2 _<_ _α_ /2.


**ˆ**
**Proof that** _**∥**_ _**x**_ _**−**_ _**x**_ _**[⋆]**_ _**∥**_ _**<**_ _**α**_ **/2.** From our choice of _ε_ and _δ_, it is easy to see that _δ_ = _α_ / ( 5 _L_ + 2 ) _<_

ˆ ˆ
_α_ /2. Thus, if _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥_ _<_ _δ_, then we automatically get _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥_ _<_ _α_ /2. So it remains to handle


59


ˆ _**x**_ ˆ _−_ _**x**_ _[⋆]_
the case _∥_ _**x**_ _−_ _**x**_ _[⋆]_ _∥≥_ _δ_ . We choose _**x**_ _c_ ≜ _**x**_ _[⋆]_ + _δ_ _∥_ _**x**_ ˆ _−_ _**x**_ ~~_[⋆]_~~ _∥_ 2 [. It is easy to see that] _**[ x]**_ _[c]_ _[ ∈]_ _[B]_ _[d]_ [1] [(] _[δ]_ [;] _**[ x]**_ _[⋆]_ [)] [ and]
hence we get that

_f_ ( _**x**_ _[⋆]_ ) _−_ _ε_ _<_ _f_ ( _**x**_ _c_ ) _≤_ _f_ ( _**x**_ _[⋆]_ ) + _⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _c_ _−_ _**x**_ _[⋆]_ _⟩_ + _[L]_ 2 _[∥]_ _**[x]**_ _[c]_ _[ −]_ _**[x]**_ _[⋆]_ _[∥]_ [2]

_≤_ _f_ ( _**x**_ _[⋆]_ ) + _⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _c_ _−_ _**x**_ _[⋆]_ _⟩_ + _[ε]_

2 [,]


where the first inequality follows from the fact that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is an ( _ε_, _δ_ ) -local min-max equilibrium,
the second inequality follows from the _L_ -smoothness of _f_, and the third inequality follows from
_∥_ _**x**_ _c_ _−_ _**x**_ _[⋆]_ _∥≤_ _δ_ and our choice of _δ_ = _[√]_ _ε_ / _L_ . The above implies:


_⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _[⋆]_ _−_ _**x**_ _c_ _⟩_ _<_ 3 _ε_ /2.


Since ˆ _**x**_ _−_ _**x**_ _[⋆]_ = ( _**x**_ _c_ _−_ _**x**_ _[⋆]_ ) _· ∥_ _**x**_ ˆ _−_ _**x**_ _[⋆]_ _∥_ 2 / _δ_ we get that _⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _[⋆]_ _−_ _**x**_ ˆ _⟩_ _<_ 2 [3] _δ_ _[ε]_ _[∥]_ _**[x]**_ _[⋆]_ _[−]_ _**[x]**_ [ˆ] _[∥]_ 2 [. Therefore]


_∥_ _**x**_ _[⋆]_ _−_ _**x**_ ˆ _∥_ [2] 2 = _⟨_ _**x**_ _[⋆]_ _−∇_ _f_ ( _**x**_ _[⋆]_ ) _−_ _**x**_ ˆ, _**x**_ _[⋆]_ _−_ _**x**_ ˆ _⟩_ + _⟨∇_ _f_ ( _**x**_ _[⋆]_ ), _**x**_ _[⋆]_ _−_ _**x**_ ˆ _⟩_


3 _ε_
_<_ 2 _δ_ _[∥]_ _**[x]**_ _[⋆]_ _[−]_ _**[x]**_ [ˆ] _[∥]_ [2]


where in the above inequality we have also used (B.1). As a result, _∥_ _**x**_ _[⋆]_ _−_ _**x**_ ˆ _∥_ 2 _<_ 2 [3] _δ_ _[ε]_ _[<]_ _[ α]_ [/2.]


**B.2** **Proof of Theorem 5.2**


We provide a polynomial-time reduction from GDAFixedPoint to Brouwer. This establishes
both the totality of GDAFixedPoint and its inclusion to PPAD, since Brouwer is both total
and lies in PPAD, as per Lemma 2.5. It also establishes the totality and inclusion to PPAD of
LR-LocalMinMax, since LR-LocalMinMax is polynomial-time reducible to GDAFixedPoint,
as shown in Theorem 5.1.

We proceed to describe our reduction. Suppose that _f_ is the _G_ -Lipschitz and _L_ -smooth function provided as input to GDAFixedPoint. Suppose also that _α_ is the approximation parameter
provided as input to GDAFixedPoint. Given _f_ and _α_, we define function _M_ : _P_ ( _**A**_, _**b**_ ) _→P_ ( _**A**_, _**b**_ ),
which serves as input to Brouwer, as follows:


_M_ ( _**x**_, _**y**_ ) = Π _P_ ( _**A**_, _**b**_ ) � ( _**x**_ _−∇_ _x_ _f_ ( _**x**_, _**y**_ ), _**y**_ + _∇_ _y_ _f_ ( _**x**_, _**y**_ )) � .



Given that _f_ is _L_ -smooth, it follows that _M_ is ( _L_ + 1 ) -Lipschitz. We set the approximation
parameter provided as input to Brouwer be _γ_ = _α_ [2] /4 ( _G_ + 2 _√_ _d_ ) .


To show the validity of the afore-described reduction, we prove that every feasible point
( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ ) that is a _γ_ -approximate fixed point of _M_, i.e. _∥_ _M_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _−_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _∥_ 2 _<_ _γ_ is
also an _α_ -approximate fixed point of _F_ _GDA_ . Observe that since _P_ ( _**A**_, _**b**_ ) _⊆_ [ 0, 1 ] _[d]_ it holds that
_∥_ ( _**x**_, _**y**_ ) _−_ ( _**x**_ _[′]_, _**y**_ _[′]_ ) _∥_ 2 _≤_ _√_ _d_ for all ( _**x**_, _**y**_ ), ( _**x**_ _[′]_, _**y**_ _[′]_ ) _∈P_ ( _**A**_, _**b**_ ) . Hence, if _γ_ _>_ _√_ _d_, then finding _γ_ 


_d_ for all ( _**x**_, _**y**_ ), ( _**x**_ _[′]_, _**y**_ _[′]_ ) _∈P_ ( _**A**_, _**b**_ ) . Hence, if _γ_ _>_ _√_



_∥_ ( _**x**_, _**y**_ ) _−_ ( _**x**_ _[′]_, _**y**_ _[′]_ ) _∥_ 2 _≤_ _√_ _d_ for all ( _**x**_, _**y**_ ), ( _**x**_ _[′]_, _**y**_ _[′]_ ) _∈P_ ( _**A**_, _**b**_ ) . Hence, if _γ_ _>_ _√_ _d_, then finding _γ_ 
approximate fixed points of _M_ is trivial and the same is true for fiding _α_ -approximate fixed
points of _F_ _GDA_, since _γ_ = _α_ [2] /4 ( _G_ + 2 _√_ _d_ ) which implies that, if _γ_ _>_ _√_ _d_, then _α_ _>_ _√_ _d_ . Thus, we



_d_ ) which implies that, if _γ_ _>_ _√_



_d_, then _α_ _>_ _√_



points of _F_ _GDA_, since _γ_ = _α_ /4 ( _G_ + 2 _√_ _d_ ) which implies that, if _γ_ _>_ _√_ _d_, then _α_ _>_ _√_ _d_ . Thus, we

may assume that _γ_ _≤_ _√_ _d_ .

Next, to simplify notation we define ( _**x**_ ∆, _**y**_ ∆ ) = ( _x_ _[⋆]_ _−∇_ _x_ _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), _**y**_ _[⋆]_ + _∇_ _y_ _f_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )) and

ˆ
( _**x**_, ˆ _**y**_ ) = argmin ( _**x**_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) _∥_ ( _**x**_ ∆, _**y**_ ∆ ) _−_ ( _**x**_, _**y**_ ) _∥_ 2 . Given that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) is a _γ_ -approximate fixed point
of _M_, we have that


ˆ
_∥_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _−_ ( _**x**_, ˆ _**y**_ ) _∥_ 2 _<_ _γ_ . (B.6)


60



_d_ .


Using Theorem 1.5.5 (b) of [FP07], we get that


ˆ ˆ
_⟨_ ( _**x**_ ∆, _**y**_ ∆ ) _−_ ( _**x**_, ˆ _**y**_ ), ( _**x**_, _**y**_ ) _−_ ( _**x**_, ˆ _**y**_ ) _⟩≤_ 0 for all ( _**x**_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) . (B.7)


Next we show the following:



**Claim B.2.** _For all_ ( _**x**_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ ) _,_ _⟨_ ( _**x**_ ∆, _**y**_ ∆ ) _−_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ), ( _**x**_, _**y**_ ) _−_ ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ ) _⟩_ _<_ ( _G_ + 2 _√_ _d_ ) _·_ _γ._


_Proof._ We have that:


ˆ
_⟨_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ), ( _**x**_, _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _⟩_ = _⟨_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_, ˆ _y_ ), ( _**x**_, _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _⟩_


ˆ
+ _⟨_ ( _**x**_, ˆ _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ), ( _**x**_, _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _⟩_


ˆ ˆ
= _⟨_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_, ˆ _y_ ), ( _**x**_, _y_ ) _−_ ( _**x**_, ˆ _y_ ) _⟩_


ˆ ˆ
+ _⟨_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_, ˆ _y_ ), ( _**x**_, ˆ _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _⟩_


ˆ
+ _⟨_ ( _**x**_, ˆ _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ), ( _**x**_, _y_ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _⟩_



ˆ
_<_ _∥_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_, ˆ _y_ ) _∥_ 2 _γ_ + _γ_ _·_ _√_ _d_

_≤_ _∥_ ( _**x**_ ∆, _y_ ∆ ) _−_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _∥_ 2 _γ_ + _γ_ [2] + _γ_ _·_ _√_



_d_



= _∥∇_ _f_ ( _**x**_ _[⋆]_, _y_ _[⋆]_ ) _∥_ 2 _γ_ + _γ_ [2] + _γ_ _·_ _√_ _d_



_≤_ ( _G_ + 2 _√_



_d_ ) _·_ _γ_,



where (1) for the first inequality we use (B.6), (B.7), the Cauchy-Schwarz inequality, and the fact
that the _ℓ_ 2 diameter of _P_ ( _**A**_, _**b**_ ) is at most _√_ _d_ ; (2) for the second inquality we use the triangle



that the _ℓ_ 2 diameter of _P_ ( _**A**_, _**b**_ ) is at most _√_ _d_ ; (2) for the second inquality we use the triangle

inequality and (B.6); (3) for the equality that follows we use the definition of ( _**x**_ ∆, _y_ ∆ ) ; and (4) for
the last inequality we use that _G_, the Lipschitzness of _f_, bounds the magnitude of its gradient,
and that _γ_ _≤_ _√_ _d_ .



_d_ .



Now let _**x**_ _[′]_ = argmin _**x**_ _∈_ _K_ ( _y_ _⋆_ ) _∥_ _**x**_ _−_ _**x**_ ∆ _∥_ 2 where _K_ ( _**y**_ _[⋆]_ ) = _{_ _**x**_ _|_ ( _**x**_, _**y**_ _[⋆]_ ) _∈P_ ( _**A**_, _**b**_ )) _}_ . Using Theorem
1.5.5 (b) of [FP07] for _**x**_ _[′]_ we get that _⟨_ _**x**_ ∆ _−_ _**x**_ _[′]_, _**x**_ _[⋆]_ _−_ _**x**_ _[′]_ _⟩≤_ 0. Using Claim B.2 for vector ( _**x**_ _[′]_, _y_ _[⋆]_ ) _∈_
_P_ ( _**A**_, _**b**_ ) we get that _⟨_ _**x**_ _[⋆]_ _−_ _**x**_ ∆, _**x**_ _[⋆]_ _−_ _**x**_ _[′]_ _⟩_ _<_ ( _G_ + 2 _√_ _d_ ) _γ_ . Adding the last two inequalities and using



_P_ ( _**A**_, _**b**_ ) we get that _⟨_ _**x**_ _[⋆]_ _−_ _**x**_ ∆, _**x**_ _[⋆]_ _−_ _**x**_ _[′]_ _⟩_ _<_ ( _G_ + 2 _√_ _d_ ) _γ_ . Adding the last two inequalities and using

the fact that _γ_ = _α_ [2] /4 ( _G_ + 2 _√_ _d_ ) we get the following




[2] /4 ( _G_ + 2 _√_ _d_ ) we get the following

��� _**x**_ _⋆_ _−_ Π _K_ ( _y_ _⋆_ ) ( _**x**_ _⋆_ _−∇_ _x_ _f_ ( _**x**_ _⋆_, _y_ _⋆_ )) ��� 2 _[<]_ �



( _G_ + 2 ~~_√_~~



_d_ ) _·_ _γ_ = _α_ /2.



Using the exact same reasoning we can also prove that
��� _**y**_ _⋆_ _−_ Π _K_ ( _x_ _⋆_ ) ( _**y**_ _⋆_ _−∇_ _y_ _f_ ( _**x**_ _⋆_, _y_ _⋆_ )) ��� 2 _[<]_ _[ α]_ [/2]


where _K_ ( _**x**_ _[⋆]_ ) = _{_ _**y**_ _|_ ( _**x**_ _[⋆]_, _**y**_ ) _∈P_ ( _**A**_, _**b**_ )) _}_ . Combining the last two inequalities we get that ( _**x**_ _[⋆]_, _**y**_ _[⋆]_ )
is an _α_ -approximate fixed point of _F_ _GDA_ .

#### **C Missing Proofs from Section 8**


In this section we present the missing proofs from Section 8 and more precisely in the following
sections we prove the Lemmas 8.10, 8.11, and 8.12. For the rest of the proofs in this section we
define _L_ ( _**c**_ ) to be the cubelet which has the down-left corner equal to _**c**_, formaly



_L_ ( _**c**_ ) = _c_ 1 _[c]_ [1] [ +] [ 1]
� _N_ _−_ 1 [,] _N_ _−_ 1



_c_ _d_ [+] [ 1]
_× · · · ×_ _[c]_ _[d]_
� � _N_ _−_ 1 [,] _N_ _−_ 1



�



and we also define _L_ _c_ ( _**c**_ ) to be the set of corners of the cubelet _L_ ( _**c**_ ), or more formally


_L_ _c_ ( _**c**_ ) = _{_ _c_ 1, _c_ 1 + 1 _} × · · · × {_ _c_ _d_, _c_ _d_ + 1 _}_ .


61


**C.1** **Proof of Lemma 8.10**


We start with a lemma about the differentiability properties of the functions _Q_ _**[c]**_ _**v**_ [which we defined]
in Definition 8.7.


_[̸]_



**Lemma C.1.** _Let_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _lying in cublet R_ ( _**x**_ ) = _Nc_ _−_ 1 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1]
�


_[̸]_




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �


_[̸]_




_[c]_ _N_ _[d]_ _−_ [+] 1 [1] _, where_ _**c**_ _∈_

�


_[̸]_



([ _N_ ] _−_ 1 ) _[d]_ _. Then for any vertex_ _**v**_ _∈_ _R_ _c_ ( _**x**_ ) _, the function Q_ _**v**_ _**[c]**_ [(] _**[x]**_ [)] _[ is continuous and twice differentiable.]_

_Moreover if Q_ _**v**_ _**[c]**_ [(] _**[x]**_ [) =] [ 0] _[ then also]_ _[d][Q]_ _dx_ _**v**_ _**[c]**_ [(] _i_ _**[x]**_ [)] = 0 _and_ _[d]_ _dx_ [2] _[Q]_ _i_ _dx_ _**v**_ _**[c]**_ [(] _**[x]**_ _j_ [)] [=] [ 0] _[.]_


_Proof._ **1st order differentiability:** We remind from the Definition 8.7 that if we let _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _d_ )
be the source vertex of _R_ ( _**x**_ ) and _**p**_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _[d]_ [)] [ be the canonical representation of] _**[ x]**_ [. Then for]
each vertex _**v**_ _∈_ _R_ _c_ ( _**x**_ ) we define the following partition of the set of coordinates [ _d_ ],


_A_ _**[c]**_ _**v**_ [=] _[ {]_ _[j]_ [ :] _|_ _v_ _j_ _−_ _s_ _j_ _|_ = 0 _}_ and _B_ _**v**_ _**[c]**_ [=] _[ {]_ _[j]_ [ :] _|_ _v_ _j_ _−_ _s_ _j_ _|_ = 1 _}_ .


Now in case _B_ _**v**_ _**[c]**_ [=][ ∅] [, which corresponds to] _**[ v]**_ [ being the source node] _**[ s]**_ _**[c]**_ [ then] _[ Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ ∏] _[d]_ _j_ = 1 _[S]_ [∞] [(] [1] _[ −]_
_S_ ( _p_ _j_ )) which is clearly differentiable as product of compositions of differentiable functions. The
exact same holds for _A_ _**[c]**_ _**v**_ [=][ ∅] [which corresponds to] _**[ v]**_ [ being the target vertex] _**[ t]**_ _**[c]**_ [ of the cubelet]
_R_ ( _**x**_ ) . We thus focus on the case where _A_ _**[c]**_ _**v**_ [,] _[ B]_ _**v**_ _**[c]**_ _[̸]_ [=][ ∅] [. To simplify notation we denote] _[ Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ by]
_Q_ ( _**x**_ ), _A_ _**[c]**_ _**v**_ [by] _[ A]_ [ and] _[ B]_ _**v**_ _**[c]**_ [by] _[ B]_ [ for the rest of this proof. We prove that in case] _[ i]_ _[ ∈]_ _[B]_ [ then] _[∂][Q]_ _∂x_ [(] _i_ _**[x]**_ [)]

always exits. The case _i_ _∈_ _A_ follows then symmetrically. We have the following cases


 - Let _j_ _∈_ _A_ and _ℓ_ _∈_ _B_ _\ {_ _i_ _}_ such that _p_ _j_ _≥_ _p_ _ℓ_ . By Definition 8.7, if _ε_ is sufficiently small then

_Q_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = _Q_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) = _Q_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0. Thus _[∂][Q]_ _∂x_ [(] _i_ _**[x]**_ [)] exists and equals 0.


 - Let _p_ _ℓ_ _>_ _p_ _j_ for all _ℓ_ _∈_ _B_ _\ {_ _i_ _}_ and _j_ _∈_ _A_ . In this case we have the following subcases.


_▷_ _p_ _i_ _>_ _p_ _j_ for all _j_ _∈_ _A_ : Then _[∂][Q]_ _∂x_ [(] _i_ _**[x]**_ [)] exists since both _S_ ∞ ( _·_ ) and _S_ ( _·_ ) are differentiable.

_▷_ _p_ _i_ _<_ _p_ _j_ for some _j_ _∈_ _A_ : By Definition 8.7, if _ε_ is sufficiently small then _Q_ ( _x_ _i_ _−_ _ε_, _x_ _−_ _i_ ) =

_Q_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) = _Q_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0. Thus _[∂][Q]_ _∂x_ [(] _i_ _**[x]**_ [)] exists and equals 0.

_▷_ _p_ _i_ = _p_ _j_ for some _j_ _∈_ _A_ and _p_ _i_ _≥_ _p_ _j_ _′_ for all _j_ _[′]_ _∈_ _A_ _\ {_ _j_ _}_ : By Definition 8.7, if _ε_ is
sufficiently small then _Q_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = 0 and also _Q_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0, thus


_Q_ ( _x_ _i_, _**x**_ _−_ _i_ ) _−_ _Q_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ )
lim = 0.
_ε_ _→_ 0 [+] _ε_


At the same time

_Q_ ( _x_ _i_ + _ε_, _x_ _−_ _i_ ) _−_ _Q_ ( _x_ _i_, _x_ _−_ _i_ )
lim = 0
_ε_ _→_ 0 [+] _ε_


since both _S_ ∞ ( _·_ ) and _S_ ( _·_ ) are differentiable functions, _S_ ∞ ( _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ )) = _S_ ∞ ( 0 ) = 0,
and _S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _i_ [)] _[ −]_ _[S]_ [(] _[p]_ _j_ [)) =] _[ S]_ ∞ _[′]_ [(] [0] [) =] [ 0.]


**2nd order differentiability:** Let _Q_ _[′]_ ( _**x**_ ) be equal to _[∂][Q]_ _∂x_ [(] _k_ _**[x]**_ [)] for convenience. As in the previous

analysis in case _A_ _**[c]**_ _**v**_ [=][ ∅] [or] _[ B]_ _**v**_ _**[c]**_ [=][ ∅] [then] _[ Q]_ _[′]_ [(] _[x]_ [)] [ is differentiable with respect to] _[ x]_ _i_ [since] _[ S]_ [(] _[·]_ [)] [,] _[ S]_ [∞] [(] _[·]_ [)]
are twice differentiable. Thus we again focus in the case where _A_, _B_ _̸_ = ∅ . Notice that by the
previous analysis _Q_ _[′]_ ( _**x**_ ) = 0 if there exists _ℓ_ _∈_ _B_ and _j_ _∈_ _A_ such that _p_ _ℓ_ _≥_ _p_ _j_ . Without loss of




_[̸]_


_[′]_ [(] _**[x]**_ [)] ≜ _[∂]_ [2] _[Q]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂x_ _i_ _∂x_ _k_




_[̸]_


generality we assume that _i_ _∈_ _B_ and we prove that _[∂][Q]_ _∂_ _[′]_ _x_ [(] _**[x]**_ [)]




_[̸]_


_∂x_ _i_ _∂x_ _k_ [always exists.]




_[̸]_


62


- Let _j_ _∈_ _A_ and _ℓ_ _∈_ _B_ _\ {_ _i_ _}_ such that _p_ _j_ _≥_ _p_ _ℓ_ . By Definition 8.7, _Q_ _[′]_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = _Q_ _[′]_ ( _x_ _i_ +




_[′]_ [(] _**[x]**_ [)] ≜ _[∂]_ [2] _[Q]_ _[′]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂x_ _i_ _∂x_ _k_



_ε_, _**x**_ _−_ _i_ ) = _Q_ _[′]_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0. Thus _[∂][Q]_ _∂_ _[′]_ [(] _**[x]**_ [)]



_∂x_ _i_ _∂x_ _k_ [exists and equals 0.]




- Let _p_ _ℓ_ _>_ _p_ _j_ for all _ℓ_ _∈_ _B_ _\ {_ _i_ _}_ and _j_ _∈_ _A_ .




_[′]_ [(] _**[x]**_ [)] ≜ _[∂]_ [2] _[Q]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂x_ _i_ _∂x_ _k_




_▷_ _p_ _i_ _>_ _p_ _j_ for all _j_ _∈_ _A_ : Then _[∂][Q]_ _∂_ _[′]_ _x_ [(] _**[x]**_ [)]



_p_ _i_ _>_ _p_ _j_ for all _j_ _∈_ _A_ : Then _∂x_ _i_ _**[x]**_ ≜ _∂x_ _i_ _∂x_ _**[x]**_ _k_ [exists since both] _[ S]_ [∞] [(] _[·]_ [)] [ and] _[ S]_ [(] _[·]_ [)] [ are twice]

differentiable.




_▷_ _p_ _i_ _<_ _p_ _j_ for some _j_ _∈_ _A_ . By Definition 8.7, _Q_ _[′]_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = _Q_ _[′]_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) =




_[′]_ [(] _**[x]**_ [)] ≜ _[∂]_ [2] _[Q]_ [(] _**[x]**_ [)]

_∂x_ _i_ _∂x_ _i_ _∂x_ _k_



_Q_ _[′]_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0. Thus _[∂][Q]_ _∂_ _[′]_ [(] _**[x]**_ [)]



_∂x_ _i_ _∂x_ _k_ [exists and equals 0.]




_▷_ _p_ _i_ = _p_ _j_ for some _j_ _∈_ _A_ and _p_ _i_ _>_ _p_ _j_ _′_ for all _j_ _[′]_ _∈_ _A_ _\ {_ _j_ _}_ . By Definition 8.7, if _ε_ is
sufficiently small then _Q_ _[′]_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = 0 and thus


_Q_ _[′]_ ( _x_ _i_, _**x**_ _−_ _i_ ) _−_ _Q_ _[′]_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ )
lim = 0.
_ε_ _→_ 0 [+] _ε_


At the same time lim _ε_ _→_ 0 + _[Q]_ _[′]_ [(] _[x]_ _[i]_ [+] _[ε]_ [,] _**[x]**_ _[−]_ _[i]_ [)] _ε_ _[−]_ _[Q]_ _[′]_ [(] _[x]_ _[i]_ [,] _**[x]**_ _[−]_ _[i]_ [)] exists since both _S_ ∞ ( _·_ ) and _S_ ( _·_ ) are twice

differentiable. Moreover equals 0 since _S_ ∞ ( _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ )) = _S_ ∞ ( 0 ) = 0 and _S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _i_ [)] _[ −]_
_S_ ( _p_ _j_ )) = _S_ ∞ _[′]_ [(] [0] [) =] _[ S]_ ∞ _[′′]_ [(] [0] [) =] _[ S]_ [(] [0] [) =] [ 0.]


In every step of the above proof where we use properties of _S_ ∞ and _S_ we use Lemma 8.3.


So far we have established the fact that the functions _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ are twice differentiable when] _**[ x]**_ [ moves]
within the same cubelet. Next we will show that when _**x**_ moves from one cubelet to another then

the corresponding _Q_ _**[c]**_ _**v**_ [functions changes value smoothly.]



**Lemma C.2.** _Let_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _such that there exists a coordinate i_ _∈_ [ _d_ ] _with the property R_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) =
_c_ 1 _c_ _d_ _c_ 1 _[′]_ 1 [+] [1] _c_ _[′]_ _d_ _d_ [+] [1]
_N_ _−_ 1 [,] _[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _N_ _−_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1] _and R_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = _N_ _−_ 1 [,] _[c]_ _N_ _[′]_ _−_ 1 _× · · · ×_ _N_ _−_ 1 [,] _[c]_ _N_ _[′]_ _−_ 1 _, with_ _**c**_, _**c**_ _[′]_ _∈_
� � � � � � � �

([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ _and ε sufficiently small, i.e._ _**x**_ _lies in the boundary of two cubelets. Then the following_
_statements hold._


_1. For all vertices_ _**v**_ _∈_ _R_ _c_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) _∩_ _R_ _c_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) _, it holds that_


_(a) Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[ Q]_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] _[,]_

_(b)_ _∂Q∂_ _**[c]**_ _**v**_ _x_ [(] _j_ _**[x]**_ [)] = _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] _for all i_ _∈_ [ _d_ ] _, and_


_∂_ [2] _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _**v**_ [(] _**[x]**_ [)]
_(c)_ _∂x_ _i_ _∂x_ _j_ [=] _[∂]_ _∂_ _[Q]_ _x_ _i_ _**[c]**_ _∂_ _[′]_ _x_ _j_ _[for all i]_ [,] _[ j]_ _[ ∈]_ [[] _[d]_ []] _[.]_


_2. For all vertices_ _**v**_ _∈_ _R_ _c_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) _\_ _R_ _c_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) _, it holds that Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0] _[.]_

_3. for all vertices_ _**v**_ _∈_ _R_ _c_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) _\_ _R_ _c_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) _, it holds that Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [) =] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0] _[.]_


Lemma C.2 is crucial since it establishes that P _**v**_ ( _**x**_ ) is a continuous and twice differentiable even
when _**x**_ moves from one cubelet to another. Since the proof of Lemma C.2 is very long and
contains the proof of some sublemmas, we postpone it for the end of this section in Section C.1.1.
We now proceed with the proof of Lemma 8.10.


63




_[c]_ _N_ [1] _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _d_ 1 [,] _[c]_ _N_ _[d]_ _−_ [+] 1 [1]

� �




_[c]_ _N_ _[d]_ _−_ [+] 1 [1] _and R_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = _Nc_ _−_ 1 _[′]_ 1 [,] _[c]_ _N_ 1 _[′]_ _−_ [+] 1 [1] _× · · · ×_ _Nc_ _−_ _[′]_ _d_ 1 [,] _[c]_ _N_ _d_ _[′]_ _−_ [+] 1 [1] _, with_ _**c**_, _**c**_ _[′]_ _∈_

� � � � �


_Proof of Lemma 8.10._ We first prove that P _**v**_ ( _**x**_ ) is a continuous function. Let _**x**_ _∈_ [ 0, 1 ] _[d]_ lying on
the boundary of the following cubelets

_c_ 1 [(] [1] [)] _[c]_ 1 [(] [1] [)] + 1 _× · · · ×_ _c_ [(] _d_ [1] [)] _[c]_ [(] _d_ [1] [)] + 1
_N_ _−_ 1 [,] _N_ _−_ 1 _N_ _−_ 1 [,] _N_ _−_ 1
� � � �



�



_× · · · ×_



_c_ [(] _d_ [1] [)] _d_ + 1

_[c]_ [(] [1] [)]
_N_ _−_ 1 [,] _N_ _−_ 1
�



�




_· · ·_

_c_ [(] _[i]_ [)]
1 1 [+] [ 1]

_[c]_ [(] _[i]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
� �



�



_× · · · ×_



_c_ [(] _[i]_ [)]
_d_ _d_ [+] [ 1]

_[c]_ [(] _[i]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
� �




_· · ·_
_c_ 1 [(] _[m]_ [)] 1 + 1

_[c]_ [(] _[m]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
� �



�



_× · · · ×_



_c_ [(] _d_ _[m]_ [)] _d_ + 1

_[c]_ [(] _[m]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
� �



.



where _**c**_ [(] [1] [)], . . ., _**c**_ [(] _[m]_ [)] _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ . This means that for every _i_ _∈_ [ _m_ ] there exists a coordinate
_j_ _i_ _∈_ [ _d_ ] and a value _η_ _i_ _∈_ **R** with sufficiently small absolute value such that



�



_R_ ( _x_ _j_ _i_ + _η_ _i_, _**x**_ _−_ _j_ _i_ ) =



_c_ [(] _[i]_ [)]
1 1 [+] [ 1]

_[c]_ [(] _[i]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
�



_× · · · ×_



_c_ [(] _[i]_ [)]
_d_ _d_ [+] [ 1]

_[c]_ [(] _[i]_ [)]
_N_ _−_ 1 [,] _N_ _−_ 1
� �



.



We then consider the following cases.


 - _**v**_ / _∈∪_ _i_ _[m]_ = 1 _[R]_ _[c]_ [(] _[x]_ _[j]_ _i_ [+] _[ η]_ _[i]_ [,] _**[ x]**_ _[−]_ _[j]_ _i_ [)] [.] By Definition 8.9, in all the _m_ aforementioned cubelets, the
coefficient P _**v**_ takes value 0 and hence it is continuous in this part of the space.


 - _**v**_ _∈∩_ _j_ _∈_ _U_ _R_ _c_ ( _x_ _j_ _i_ + _η_ _i_, _**x**_ _−_ _j_ _i_ ) and _**v**_ / _∈∪_ _i_ _∈_ _U_ _R_ _c_ ( _x_ _j_ _i_ + _η_ _i_, _**x**_ _−_ _j_ _i_ ), for some _U_ _⊆_ [ _m_ ] with _U_ = [ _m_ ] _\_ _U_ .
In this case P _**v**_ ( _x_ _j_ _i_ + _η_ _i_, _**x**_ _j_ _i_ ) was computed according to a cubelet with _**v**_ _∈_ _R_ _c_ ( _x_ _j_ _i_ + _η_ _i_, _**x**_ _−_ _j_ _i_ ) .

Then Lemma C.2 implies that _Q_ _**[c]**_ _**v**_ [(] _[i]_ [)] [(] _**[x]**_ [) =] [ 0 since] _**[ v]**_ _[ ∈]_ _[R]_ _[c]_ [(] _[x]_ _j_ _i_ [+] _[ η]_ _i_ [,] _**[ x]**_ _−_ _j_ _i_ [)] _[ \]_ _[ R]_ _[c]_ [(] _[x]_ _j_ _i_ _′_ [+] _[ η]_ _i_ _[′]_ [,] _**[ x]**_ _−_ _j_ _i_ _′_ [)]
where _i_ _[′]_ _∈_ [ _m_ ] and _i_ _̸_ = _i_ _[′]_ . Therefore we conclude that P _**v**_ ( _**x**_ ) = 0 and


_η_ lim _i_ _→_ 0 [P] _**[v]**_ [(] _[x]_ _[j]_ _[i]_ [ +] _[ η]_ _[i]_ [,] _**[ x]**_ _[−]_ _[i]_ [) =] [ 0.]


 - _**v**_ _∈∩_ _i_ _[m]_ = 1 _[R]_ _[c]_ [(] _[x]_ _[j]_ _i_ [+] _[ η]_ _[i]_ [,] _**[ x]**_ _[−]_ _[j]_ _i_ [)] [. By Lemma][ C.2][ for all] _[ i]_ _[ ∈]_ [[] _[m]_ []] [ it holds that]



_Q_ _**[c]**_ _**v**_ [(] _[i]_ [)] [(] _**[x]**_ [)]



_Q_ _**[c]**_ _**v**_ [(] _[i]_ [)] [(] _**[x]**_ [)] _Q_ _**[c]**_ _**v**_ [(] _[i]_ [)] [(] _**[x]**_ [)]

∑ _**v**_ _∈_ _R_ _c_ ( _x_ _ji_ + _η_ _i_, _**x**_ _−_ _ji_ ) _Q_ _**[c]**_ _**v**_ [(] _[i]_ [)] [(] _**[x]**_ [)] [=] ∑ _**v**_ _∈∩_ _i_ _[m]_ = 1 _[R]_ _[c]_ [(] _[x]_ _[j]_ _i_ [+] _[η]_ _[i]_ [,] _**[x]**_ _[−]_ _[j]_



∑ _**v**_ _∈∩_ _i_ _[m]_ = 1 _[R]_ _[c]_ [(] _[x]_ _[j]_ _i_ [+] _[η]_ _[i]_ [,] _**[x]**_ _[−]_ _[j]_ _i_ [)] _[ Q]_ _**v**_ _**[c]**_ [(] _[i]_ [)] [(] _**[x]**_ [)]



_Q_ _**[c]**_ _**v**_ [(] _[i]_ _[′]_ [)] ( _**x**_ )
=



_Q_ _**[c]**_ _**v**_ [(] _[i]_ _[′]_ [)] ( _**x**_ ) _Q_ _**[c]**_ _**v**_ [(] _[i]_ _[′]_ [)] ( _**x**_ )

=
∑ _**v**_ _∈∩_ _i_ _[m]_ = 1 _[R]_ _[c]_ [(] _[x]_ _[j]_ _i_ [+] _[η]_ _[i]_ [,] _**[x]**_ _[−]_ _[j]_ _i_ [)] _[ Q]_ _**v**_ _**[c]**_ [(] _[i]_ _[′]_ [)] ( _**x**_ ) ∑ _**v**_ _∈_ _R_ _c_ ( _x_ _ji_ + _η_ _i_, _**x**_ _−_ _ji_ )



∑ _**v**_ _∈_ _R_ _c_ ( _x_ _ji_ + _η_ _i_, _**x**_ _−_ _ji_ ) _Q_ _**[c]**_ _**v**_ [(] _[i]_ _[′]_ [)] ( _**x**_ )



which again implies the continuity of P _**v**_ ( _**x**_ ) at _**x**_ .



Next we prove that P _**v**_ ( _**x**_ ) is differentiable for all _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ . Fix some _i_ _∈_ [ _d_ ] we will
prove that _[∂]_ [P] _∂x_ [(] _**[x]**_ [)] always exists. Let _C_ [+] be the set of down-left corners of the cubelets in which



prove that _∂x_ _**[x]**_ _i_ always exists. Let _C_ [+] be the set of down-left corners of the cubelets in which

lim _ε_ _→_ 0 + ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) belongs to and _C_ _[−]_ be the set of down-left corners of the cubelets in which
lim _ε_ _→_ 0 + ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) belongs to. It easy to see that _C_ [+] and _C_ _[−]_ are non-empty and fixed for _ε_ _>_ 0
and sufficiently small.

_[∂]_ [P] _**[v]**_ [(] _**[x]**_ [)]
To prove that _∂_ always exists, we consider the following 3 mutually exclusive cases.



_∂_ _**[v]**_ _x_ _i_ always exists, we consider the following 3 mutually exclusive cases.



64


- _**v**_ _∈_ _L_ _c_ ( _**c**_ [(] [1] [)] ) for _**c**_ [(] [1] [)] _∈_ _C_ [+] and _**v**_ _∈_ _L_ _c_ ( _**c**_ [(] [2] [)] ) for _**c**_ [(] [2] [)] _∈_ _C_ _[−]_ . Since the coefficient P _**v**_ ( _**x**_ ) is a continuous function, we have that



_∂Q_ _**[c]**_ _**v**_ [(] _[′]_ [1] [)] ( _**x**_ )




_▷_ lim _ε_ _→_ 0 + [P] _**[v]**_ [(] _[x]_ _[i]_ [+] _[ε]_ [,] _**[x]**_ _[−]_ _[i]_ [)] _ε_ _[−]_ [P] _**[v]**_ [(] _[x]_ _[i]_ [,] _**[x]**_ _[−]_ _[i]_ [)] =



_∂Q_ _**[c]**_ _**v**_ _∂_ [(] _x_ [1] _i_ [)] ( _**x**_ ) ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 1 ) ) _Q_ _**v**_ _**[c]**_ [(] _[′]_ [1] [)] ( _**x**_ ) _−_ _Q_ _**[c]**_ _**v**_ [(] [1] [)] ( _**x**_ ) ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 1 ) )



) ) _**v**_ _[′]_ _**v**_ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 1 ) ) _∂xi_

2
~~�~~ ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 1 )) _Q_ _**v**_ _**[c]**_ [(] _[′]_ [1] [)] ( _**x**_ ) ~~�~~



_∂Q_ _**[c]**_ _**v**_ [(] _[′]_ [2] [)] ( _**x**_ )




_▷_ lim _ε_ _→_ 0 + [P] _**[v]**_ [(] _[x]_ _[i]_ [,] _**[x]**_ _[−]_ _[i]_ [)] _[−]_ [P] _ε_ _**[v]**_ [(] _[x]_ _[i]_ _[−]_ _[ε]_ [,] _**[x]**_ _[−]_ _[i]_ [)] =



_∂Q_ _**[c]**_ _**v**_ _∂_ [(] _x_ [2] _i_ [)] ( _**x**_ ) ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 2 ) ) _Q_ _**v**_ _**[c]**_ [(] _[′]_ [2] [)] ( _**x**_ ) _−_ _Q_ _**[c]**_ _**v**_ [(] [2] [)] ( _**x**_ ) ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 2 ) )



) ) _**v**_ _[′]_ _**v**_ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 2 ) ) _∂xi_

2
~~�~~ ∑ _**v**_ _′∈_ _Lc_ ( _**c**_ ( 2 )) _Q_ _**v**_ _**[c]**_ [(] _[′]_ [2] [)] ( _**x**_ ) ~~�~~



Both of the above limits exists due to the fact that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ is differentiable (Lemma][ C.1][).]
Moreover, since _**v**_ _∈_ _L_ _c_ ( _**c**_ [(] [1] [)] ) _∩_ _L_ _c_ ( _**c**_ [(] [2] [)] ), Case 1 of Lemma C.2 implies that the two limits
above have exactly the same value and hence P _**v**_ is differentiable at _**x**_ .


- _**v**_ / _∈_ _L_ _c_ ( _**c**_ [(] [1] [)] ) for all _**c**_ [(] [1] [)] _∈_ _C_ [+] . In the case where _**v**_ / _∈_ _L_ _c_ ( _**c**_ ) for all the down-left corners
_**c**_ of the cubelets at which _**x**_ lies, then by Definition 8.9 P _**v**_ ( _x_ _i_, _**x**_ _−_ _i_ ) = P _**v**_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) =
P _**v**_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ ) = 0. Thus _[∂]_ [P] _∂_ _**[v]**_ _x_ [(] _i_ _**[x]**_ [)] exists and equals 0. Therefore we may assume that _**v**_ _∈_ _L_ _c_ ( _**c**_ )

for some down-left corner _**c**_ of a cubelet at which _**x**_ lies. Due to the fact that P _**v**_ ( _**x**_ ) is a
continuous function and that _**v**_ / _∈_ _L_ _c_ ( _**c**_ [(] [1] [)] ) for all _**c**_ [(] [1] [)] _∈_ _C_ [+], we get that


P _**v**_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) = 0 and P _**v**_ ( _x_ _i_, _**x**_ _−_ _i_ ) = 0.


We also have that _**v**_ _∈_ _L_ _c_ ( _**c**_ ) / _L_ _c_ _**c**_ [(] [1] [)] where _**c**_, _**c**_ [(] [1] [)] are down-left corners of cubelets at which
_**x**_ lies and ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) lies respectively. Therefore we get by Case 1 of Lemma C.2 that
_Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0 implying that][ P] _**[v]**_ [(] _[x]_ _i_ [,] _**[ x]**_ _−_ _i_ [) =] [ 0. As a result,]


P _**v**_ ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) _−_ P _**v**_ ( _x_ _i_, _x_ _−_ _i_ )
lim = 0
_ε_ _→_ 0 [+] _ε_


We now need to argue that lim _ε_ _→_ 0 + [P] _**[v]**_ [(] _[x]_ _[i]_ [,] _**[x]**_ _[−]_ _[i]_ [)] _[−]_ [P] _ε_ _**[v]**_ [(] _[x]_ _[i]_ _[−]_ _[ε]_ [,] _[x]_ _[−]_ _[i]_ [)] exists and equals 0. At first observe

that 0 _≤_ _x_ _i_ _−_ _c_ _i_ _≤_ _δ_ since _**x**_ lies in the cubelet with down-left corner _**c**_ . In case _x_ _i_ _−_ _c_ _i_ _<_ _δ_
then ( _x_ _i_ + _ε_, _**x**_ _−_ _i_ ) lies in _**c**_ for arbitrarily small _ε_, meaning that _**c**_ _∈_ _C_ [+] . The latter contradicts
the fact that _**v**_ / _∈_ _L_ _c_ _**c**_ [(] [1] [)] for all _**c**_ [(] [1] [)] _∈_ _C_ [+] . As a result, _x_ _i_ _−_ _c_ _i_ = _δ_ which implies that _**c**_ _∈_ _C_ _[−]_

and hence



P _**v**_ ( _x_ _i_, _**x**_ _−_ _i_ ) _−_ P _**v**_ ( _x_ _i_ _−_ _ε_, _**x**_ _−_ _i_ )
lim =
_ε_ _→_ 0 [+] _ε_



_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _∂Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)]
_∂x_ _i_ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] _[ −]_ _[Q]_ _**v**_ _**[c]**_ [(] _**[x]**_ [)] [ ∑] _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ ) _∂x_ _i_

2 .
~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] ~~�~~



The above limit equals to 0 since _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ [(] _i_ _**[x]**_ [)] = 0 by applying Lemma C.2 due to the

fact that _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _\_ _L_ _c_ ( _**c**_ [(] [1] [)] ) .


 - _**v**_ / _∈_ _L_ _c_ ( _**c**_ [(] [2] [)] ) for all _**c**_ [(] [2] [)] _∈_ _C_ _[−]_ . Symmetrically with the previous case.


The second order differentiability of P _**v**_ ( _**x**_ ) can be established using exactly the same arguments
for computing the following limit


P _**v**_ ( _x_ _i_ + _ε_, _x_ _j_ + _ε_ _[′]_, _**x**_ _−_ _i_, _j_ ) _−_ P _**v**_ ( _**x**_ )
lim .
_ε_, _ε_ _[′]_ _→_ 0 _ε_ [2]


65


The last thing that we need to show to prove Lemma 8.10 is that the set _R_ + ( _**x**_ ) has cardinality
at most _d_ + 1 and that it can be computed in poly ( _d_ ) time. Let _p_ _**[c]**_ _**x**_ _[∈]_ [[] [0, 1] []] _[d]_ [ be the canonical]
representation of _**x**_ with the respect to a cubelet _L_ ( _**c**_ ) in which _**x**_ belongs to. We define the source
vertex _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _d_ ) and the target vertex _**t**_ _**[c]**_ = ( _t_ 1, . . ., _t_ _d_ ) of _L_ ( _**c**_ ) . Once this is done the vertices
in _R_ + ( _**v**_ ) are exactly the vertices of _L_ _c_ ( _**c**_ ) for which it holds that


_p_ _ℓ_ _>_ _p_ _j_ for all _ℓ_ _∈_ _A_ _**[c]**_ _**v**_ [,] _[ j]_ _[ ∈]_ _[B]_ _**v**_ _**[c]**_


since for all the others _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ it holds that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0,] _[ ∇]_ _[Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0, and] _[ ∇]_ [2] _[Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0.]
These vertices _**v**_ _∈_ _R_ + ( _**x**_ ) can be computed in polynomial time as follows: **i)** the coordinates
_p_ 1, . . ., _p_ _d_ are sorted in increasing order, and **ii)** for each _m_ = 0, . . ., _d_ compute the vertex _**v**_ [(] _[m]_ [)] _∈_
_L_ _c_ ( _**c**_ ),


_s_ _j_ if coordinate _j_ belongs in the first _m_ coordinates wrt the order of _**p**_ _**[c]**_ _**x**_
_**v**_ _[m]_ _j_ [=] � _t_ _j_ if coordinate _j_ belongs in the last _d_ _−_ _m_ coordinates wrt the order of _**p**_ _**[c]**_ _**x**_


By Definition 8.7 it immediately follows that _R_ + ( _**x**_ ) _⊆{_ _**v**_ [(] [1] [)], . . ., _**v**_ [(] _[m]_ [)] _}_ from which we get that
_|_ _R_ + ( _**x**_ ) _| ≤_ _d_ + 1 and also they can be computed in poly ( _d_ ) time.


To finish the proof of Lemma 8.10 we only need the proof of Lemma C.2 which we present in the
following section.


**C.1.1** **Proof of Lemma C.2**


**Lemma C.3.** _Let a point_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _lying in the boundary of the cubelets with down-left corners_
_**c**_ = ( _c_ 1, . . ., _c_ _m_ _−_ 1, _c_ _m_, _c_ _m_ + 1, . . ., _c_ _d_ ) _and_ _**c**_ _[′]_ = ( _c_ 1, . . ., _c_ _m_ _−_ 1, _c_ _m_ + 1, _c_ _m_ + 1, . . ., _c_ _d_ ) _. Then the canoni-_
_cal representation of_ _**x**_ _in the cubelet L_ ( _**c**_ ) _is the same with the the canonical representation of_ _**x**_ _in the_
_cubelet L_ ( _**c**_ _[′]_ ) _. More precisely,_ _**p**_ _**[c]**_ _**x**_ [=] _**[ p]**_ _**[c]**_ _**x**_ _[′]_ _[.]_


_Proof._ Let _c_ _m_ be even. By the definition of the canonical representation in Definition 8.6, the
source and target of the cubelets _L_ ( _**c**_ ) and _L_ ( _**c**_ _[′]_ ) are respectively,


_⋄_ _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _m_ _−_ 1, _c_ _m_, _s_ _m_ + 1, . . ., _s_ _d_ ),


_⋄_ _**t**_ _**[c]**_ = ( _t_ 1, . . ., _s_ _m_ _−_ 1, _c_ _m_ + 1, _t_ _m_ + 1, . . ., _t_ _d_ ),


_⋄_ _**s**_ _**[c]**_ _[′]_ = ( _s_ 1, . . ., _s_ _m_ _−_ 1, _c_ _m_ + 2, _s_ _m_ + 1, . . ., _s_ _d_ ),


_⋄_ _**t**_ _**[c]**_ _[′]_ = ( _t_ 1, . . ., _t_ _m_ _−_ 1, _c_ _m_ + 1, _t_ _m_ + 1, . . ., _t_ _d_ ) .


Hence we get that _p_ _j_ = _p_ _[′]_ _j_ [for] _[ j]_ _[ ̸]_ [=] _[ m]_ [. Since] _**[ x]**_ [ belongs to the boundary of both cublets] _[ L]_ [(] _**[c]**_ [)] [ and]
_L_ ( _**c**_ _[′]_ ) we get that _x_ _m_ = _c_ _m_ + 1 which implies that _p_ _m_ = _p_ _m_ _[′]_ [=] [ 1. In case] _[ c]_ _[m]_ [is odd we get that]
_**p**_ _**[c]**_ _**x**_ [=] _**[ p]**_ _**[c]**_ _**x**_ _[′]_ [but with] _[ p]_ _[m]_ [ =] _[ p]_ _m_ _[′]_ [=] [ 0.]


**Lemma C.4.** _Let_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _lying at the intersection of the cubelets L_ ( _**c**_ ) _, L_ ( _**c**_ _[′]_ ) _with down-left corners_
_**c**_ = ( _c_ 1, . . ., _c_ _m_ _−_ 1, _c_ _m_, _c_ _m_ + 1, . . ., _c_ _d_ ) _, and_ _**c**_ _[′]_ = ( _c_ 1, . . ., _c_ _m_ _−_ 1, _c_ _m_ + 1, _c_ _m_ + 1, . . ., _c_ _d_ ) _. Then the following_

_statements are true._


_1. For all vertices_ _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _∩_ _L_ _c_ ( _**c**_ _[′]_ ) _it holds that_


_(a) Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[ Q]_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] _[,]_


66


_(b)_ _∂Q∂_ _**[c]**_ _**v**_ _x_ [(] _i_ _**[x]**_ [)] = _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] _,_


_∂_ [2] _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _**v**_ [(] _**[x]**_ [)]
_(c)_ _∂x_ _i_ _∂x_ _j_ [=] _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**[c]**_ _[′]_ _x_ _j_ _[.]_

_2. For all vertices_ _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _\_ _L_ _c_ ( _**c**_ _[′]_ ) _it holds that Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0] _[.]_

_3. For all vertices_ _**v**_ _∈_ _L_ _c_ ( _**c**_ _[′]_ ) / _L_ _c_ ( _**c**_ ) _it holds that Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [) =] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0] _[.]_


_Proof._ 1. Let _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _∩_ _L_ _c_ ( _**c**_ _[′]_ ) then we have that


(a) _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)][ =] _[Q]_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] [. By Lemma][ C.3][ we get that the canonical representation] _**[ p]**_ _**[c]**_ _**x**_ [=] _**[ p]**_ _**[c]**_ _**x**_ _[′]_ [.]
Since _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ is a function of the canonical representation] _**[ p]**_ _**[c]**_ _**x**_ [(see Definition][ 8.9][), it]
holds that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] _[ Q]_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] [ for all vertices] _**[ v]**_ _[ ∈]_ _[L]_ _[c]_ [(] _**[c]**_ [)] _[ ∩]_ _[L]_ _[c]_ [(] _**[c]**_ _[′]_ [)] [.]



(b) _∂Q∂_ _**[c]**_ _**v**_ _x_ [(] _i_ _**[x]**_ [)] = _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] . For _i_ _̸_ = _m_, we get that _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ [(] _i_ _**[x]**_ [)] = _t_ _i_ _−_ 1 _s_ _i_ _∂Q∂_ _**[c]**_ _**v**_ _p_ [(] _i_ _**[x]**_ [)] = _t_ _i_ ~~_[′]_~~ _[−]_ 1 _[s]_ _i_ ~~_[′]_~~



_∂Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] = _[∂][Q]_ _**v**_ _**[c]**_ _[′]_ [(] _**[x]**_ [)] since
_∂p_ _i_ ~~_[′]_~~ _∂x_ _i_



_t_ _i_ = _t_ _i_ _[′]_ [and] _[ s]_ _[i]_ [ =] _[ s]_ _i_ _[′]_ [for all] _[ i]_ _[ ̸]_ [=] _[ m]_ [. The latter argument cannot be applied for the] _[ m]_ [-th]
coordinate since _t_ _m_ _−_ _s_ _m_ = _−_ ( _t_ _[′]_ _m_ _[−]_ _[s]_ _[′]_ _m_ [)] [. However since] _**[ x]**_ [ belongs to the boundary of]
both the cubelets _L_ ( _**c**_ ) and _L_ ( _**c**_ _[′]_ ) it is implied that _p_ _m_ = _p_ _m_ _[′]_ [is either 0 or 1, meaning]

that _[∂][Q]_ _∂x_ _**v**_ _**[c]**_ _m_ [(] _**[x]**_ [)] = _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ _m_ [(] _**[x]**_ [)] = 0 since _S_ _[′]_ ( 0 ) = _S_ _[′]_ ( 1 ) = 0 from Lemma 8.3.



(c) _∂∂_ [2] _xQ_ _i_ _∂_ _**[c]**_ _**v**_ [(] _x_ _**[x]**_ _j_ [)] [=] _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [. For] _[ i]_ [,] _[ j]_ _[ ̸]_ [=] _[ m]_ [, we get that] _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ [(] _x_ _**[x]**_ _j_ [)] [=] _t_ _i_ _−_ 1 _s_ _i_ _t_ _j_ _−_ 1 _s_ _j_ _∂∂_ [2] _pQ_ _i_ _∂_ _**[c]**_ _**v**_ [(] _p_ _**[x]**_ _j_ [)] [=] _t_ _i_ ~~_[′]_~~ _[−]_ 1 _[s]_ _i_ ~~_[′]_~~ _t_ ~~_[′]_~~ _j_ _[−]_ 1 _[s]_ ~~_[′]_~~ _j_



_∂Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)]
_∂p_ _i_ ~~_[′]_~~ _[∂][p]_ ~~_[′]_~~ _j_ [=]



_∂∂_ [2] _xQ_ _i_ _∂_ _**[c]**_ _**v**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [since] _[ t]_ _[i]_ [ =] _[ t]_ _i_ _[′]_ [and] _[ s]_ _[i]_ [ =] _[ s]_ _i_ _[′]_ [for all] _[ i]_ _[ ̸]_ [=] _[ m]_ [. As in the previous case,] _[ p]_ _[m]_ [ =] _[ p]_ _m_ _[′]_ [equals]

_**v**_ [(] _**[x]**_ [)] _**v**_ [(] _**[x]**_ [)]
either 0 or 1. As a result, _[∂]_ _∂_ [2] _x_ _[Q]_ _m_ _∂_ _**[c]**_ _x_ _j_ [=] _[∂]_ _∂_ [2] _x_ _[Q]_ _m_ _**[c]**_ _∂_ _[′]_ _x_ _j_ [=] [ 0 since] _[ S]_ _[′]_ [(] [0] [) =] _[ S]_ _[′]_ [(] [1] [) =] _[ S]_ _[′′]_ [(] [0] [) =] _[ S]_ _[′′]_ [(] [1] [) =] [ 0]
by Lemma 8.3.


2. Since _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _\_ _L_ _c_ ( _**c**_ _[′]_ ), we get that _v_ _m_ = _c_ _m_ . In case _c_ _m_ is even, we get that _s_ _m_ = _c_ _m_ = _v_ _m_
and thus the coordinate the coordinate _m_ belongs in the set _A_ _**[c]**_ _**v**_ [. Since] _**[ x]**_ [ coincides with one]
of the corners in _L_ _c_ ( _**c**_ ) _\_ _L_ _c_ ( _**c**_ _[′]_ ) we get that _p_ _m_ = 1 which combined with the fact that _m_ _∈_ _A_ _**[c]**_ _**v**_
implies that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0 (see Definition][ 8.7][). Then by Lemma][ C.1][,] _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0. In]
case is odd, we get that _s_ _m_ = _c_ _m_ + 1. The latter combined with the fact that _v_ _m_ = _c_ _m_ implies
that the _m_ -th coordinate belongs in _B_ _**v**_ _**[c]**_ [. Now] _[ p]_ _[m]_ [=] [ 0 and by Definition][ 8.7][,] _[ Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0. Then]

again by Lemma C.1, _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ _[′]_ [(] _i_ _**[x]**_ [)] = _[∂]_ _∂_ [2] _x_ _[Q]_ _i_ _∂_ _**v**_ _**[c]**_ _[′]_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0.]


3. This case follows with the same reasoning with previous case 2.


We are now ready to prove Lemma C.2.


_Proof of Lemma C.2._ 1. Let _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _∩_ _L_ _c_ ( _**c**_ _[′]_ ) . There exists a sequence of corners


_**c**_ = _**c**_ [(] [1] [)], . . ., _**c**_ [(] _[m]_ [)] = _**c**_ _[′]_


such that _**c**_ ( _j_ ) _−_ _**c**_ ( _j_ + 1 )
��� ��� 1 [=] [ 1 and] _**[ v]**_ _[ ∈]_ _[L]_ _[c]_ [(] _**[c]**_ _[j]_ [)] [ for all] _[ j]_ _[ ∈]_ [[] _[m]_ []] [. By Lemma][ C.4][ we get that,]


(a) _Q_ _**[c]**_ _**v**_ [(] _[j]_ [)] [(] _**[x]**_ [) =] _[ Q]_ _**v**_ _**[c]**_ [(] _[j]_ [+] [1] [)] ( _**x**_ ) .

(b) _∂Q_ _**[c]**_ _**v**_ _∂_ [(] _x_ _[j]_ [)] _i_ ( _**x**_ ) = _[∂][Q]_ _**[c]**_ _**v**_ [(] _∂_ _[j]_ [+] _x_ [1] _i_ [)] ( _**x**_ ) .


67


(c) _∂_ [2] _∂Qx_ _i_ _**[c]**_ _**v**_ _∂_ [(] _[j]_ [)] _x_ ( _j_ _**x**_ ) = _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _x_ [(] _[j]_ _i_ _∂_ [+] [1] [)] _x_ ( _j_ _**x**_ ) .


which implies Case 1 of Lemma C.2.


2. Let _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _\_ _L_ _c_ ( _**c**_ _[′]_ ) . There exists a sequence of corners _**c**_ = _**c**_ [(] [1] [)] . . ., _**c**_ [(] _[i]_ [)] such that
_**c**_ ( _j_ ) _−_ _**c**_ ( _j_ + 1 )
��� ��� 1 [=] [ 1 and] _**[ v]**_ [ /] _[∈]_ _[L]_ _[c]_ _**[c]**_ [(] _[i]_ [)] [ and] _**[ v]**_ _[ ∈]_ _[L]_ _[c]_ [(] _**[c]**_ [(] _[j]_ [)] [)] [ for all] _[ j]_ _[ <]_ _[ i]_ [. By case 2 of Lemma]

C.4 we get that _Q_ _**v**_ _**[c]**_ [(] _[i]_ _[−]_ [1] [)] ( _**x**_ ) = _[∂][Q]_ _**v**_ _**[c]**_ [(] _∂_ _[i]_ _[−]_ _x_ [1] _i_ [)] ( _**x**_ ) = _[∂]_ [2] _[Q]_ _∂x_ _**v**_ _**[c]**_ [(] _i_ _[i]_ _∂_ _[−]_ [1] _x_ [)] _j_ ( _**x**_ ) = 0. Then case 2 of Lemma C.2 follows
by case 1 of Lemma C.4.


3. Similarly with case 2.


**C.2** **Proof of Lemma 8.11**


We start this section with some fundamental properties of the smooth step function _S_ ∞ that are
more fine-grained than the properties we presented in Lemma 8.3.


**Lemma C.5.** _For d_ _≥_ 10 _there exists a universal constant c_ _>_ 0 _such that the following statements hold._


_1. If x_ _≥_ 1/ _d then S_ ∞ ( _x_ ) _≥_ _c_ _·_ 2 _[−]_ _[d]_ _._


_2. If x_ _≤_ 1/ _d then S_ ∞ _[′]_ [(] _[x]_ [)] _[ ≤]_ _[c]_ _[ ·]_ _[ d]_ [2] _[ ·]_ [ 2] _[−]_ _[d]_ _[.]_


_3. If x_ _≥_ 1/ _d then_ _S_ _[S]_ ∞∞ _[′]_ ( [(] _x_ _[x]_ ) [)] _[≤]_ _[c]_ _[ ·]_ _[ d]_ [2] _[.]_


_4. If x_ _≤_ 1/ _d then_ _|_ _S_ ∞ _[′′]_ [(] _[x]_ [)] _[| ≤]_ _[c]_ _[ ·]_ _[ d]_ [4] _[ ·]_ [ 2] _[−]_ _[d]_ _[.]_


_5. If x_ _≥_ 1/ _d then_ _[|]_ _S_ _[S]_ ∞∞ _[′′]_ ( [(] _x_ _[x]_ ) [)] _[|]_ _[≤]_ _[c]_ _[ ·]_ _[ d]_ [4] _[.]_


_Proof._ We compute the derivative of _S_ ∞ and we have that



1 1
_S_ ∞ _[′]_ [(] _[x]_ [) =] [ ln] [(] [2] [)] _[S]_ [∞] [(] _[x]_ [)] _[S]_ [∞] [(] [1] _[ −]_ _[x]_ [)] [+]
� _x_ [2] ( 1 _−_ _x_ ) [2]



�



from which we immediately get _S_ ∞ _[′]_ [(] _[x]_ [)] _[ ≥]_ [0. Then we can compute the second derivative of] _[ S]_ [∞]
as follows
_S_ ∞ _[′′]_ [(] _[x]_ [) =] [ ln] [(] [2] [)] _[S]_ [∞] [(] _[x]_ [)] _[S]_ [∞] [(] [1] _[ −]_ _[x]_ [)] _[·]_



��



1 1

_[−]_
� _x_ [3] ( 1 _−_ _x_ ) [3]



2

_−_ 2
�




_·_



�



1 1
ln ( 2 ) ( _S_ ∞ ( 1 _−_ _x_ ) _−_ _S_ ∞ ( _x_ )) [+]
� _x_ [2] ( 1 _−_ _x_ ) [2]



.



We next want to prove that _S_ ∞ _[′′]_ [(] _[x]_ [)] _[ ≥]_ [0 for] _[ x]_ _[ ≤]_ [1/10. To see this observe that 1] _[ −]_ [2] _[ ·]_ _[ S]_ [∞] [(] _[x]_ [)] _[ ≥]_ [1/2]
for _x_ _≤_ 1/ _d_ and therefore



_S_ ∞ _[′′]_ [(] _[x]_ [)] _[ ≥]_ [ln] [(] [2] [)]




[(] [2] [)] ln ( 2 )

_S_ ∞ ( _x_ ) _S_ ∞ ( 1 _−_ _x_ )
_x_ [3] � 2 _x_



( 2 ) _−_ 2

2 _x_ �



hence for _x_ _≤_ 4/ ln ( 2 ) it holds that _S_ ∞ _[′′]_ [(] _[x]_ [)] _[ ≥]_ [0. By similar but more tedious calculations we can]
conclude that _S_ ∞ _[′′′]_ [(] _[x]_ [)] _[ ≥]_ [0 for] _[ x]_ _[ ≤]_ [1/10. Hence in the interval] _[ x]_ _[ ∈]_ [[] [0, 1/10] []] [ all the functions] _[ S]_ [∞] [,]
_S_ ∞ _[′]_ [,] _[ S]_ ∞ _[′′]_ [are all increasing functions of] _[ x]_ [.]


68


Next we show that the function _h_ ( _x_ ) = 2 _[−]_ [1/] _[x]_ + 2 _[−]_ [1/] [(] [1] _[−]_ _[x]_ [)] is upper and lower bounded. First
observe that _h_ ( _x_ ) _≥_ max _{_ 2 _[−]_ [1/] _[x]_, 2 _[−]_ [1/] [(] [1] _[−]_ _[x]_ [)] _}_ . Now if we set _t_ ( _x_ ) = 2 _[−]_ [1/] _[x]_ then _t_ _[′]_ ( _x_ ) = ln ( 2 ) _t_ ( _x_ ) / _x_ [2]

and hence _t_ ( _x_ ) _≥_ _t_ ( 1/2 ) = 1/4 for _x_ _≥_ 1/2. The same way we can prove that 2 _[−]_ [1/] [(] [1] _[−]_ _[x]_ [)] _≥_ 1/4
for _x_ _≤_ 1/2. Therefore _h_ ( _x_ ) _≥_ 1/4 for all _x_ _∈_ [ 0, 1 ] . Also it is not hard to see that 2 _[−]_ [1/] _[x]_ _≤_ 1/2
and 2 _[−]_ [1/] [(] [1] _[−]_ _[x]_ [)] _≤_ 1/2 which implies _h_ ( _x_ ) _≤_ 1. Hence overall we have that _h_ ( _x_ ) _∈_ [ 1/4, 1 ] for all
_x_ _∈_ [ 0, 1 ] . We are now ready to prove the statements.


1. We have shown that _S_ ∞ _[′]_ [(] _[x]_ [)] _[ ≥]_ [0 for all] _[ x]_ _[ ∈]_ [[] [0, 1] []] [. Hence] _[ S]_ [∞] [is an increasing function and]
therefore _S_ ∞ ( _x_ ) _≥_ _S_ ∞ ( 1/ _d_ ) for _x_ _≥_ 1/ _d_ . Now we have that _S_ ∞ ( 1/ _d_ ) = 2 _[−]_ _[d]_ / _h_ ( 1/ _d_ ) _≥_ 2 _[−]_ _[d]_ .


2. Since _S_ ∞ _[′]_ [(] _[x]_ [)] [ is increasing for] _[ x]_ _[ ∈]_ [[] [0, 1/10] []] [, we have that] _[ S]_ ∞ _[′]_ [(] _[x]_ [)] _[ ≤]_ _[S]_ ∞ _[′]_ [(] [1/] _[d]_ [)] [ for] _[ x]_ _[ ≤]_ [1/] _[d]_ [ and]

therefore



1
_d_ [2] +

2

~~�~~ 1 _−_ [1] _d_ ~~�~~



�



_S_ ∞ _[′]_ [(] _[x]_ [)] _[ ≤]_ [ln] [(] [2] [)] _[S]_ [∞] [(] [1] _[ −]_ [1/] _[d]_ [)] _[S]_ [∞] [(] [1/] _[d]_ [)]



�



2 _[−]_ _[d]_
_≤_ 2 ln ( 2 ) _h_ ( 1/ _d_ ) _[≤]_ [8 ln] [(] [2] [)] [2] _[−]_ _[d]_ [.]



3. We have that for _x_ _≤_ 1/ _d_


_S_ ∞ _[′]_ [(] _[x]_ [)] 1 1
_S_ ∞ ( _x_ ) [=] [ ln] [(] [2] [)] _[S]_ [∞] [(] [1] _[ −]_ _[x]_ [)] � _x_ [2] [+] ( 1 _−_ _x_ ) [2]



_≤_ 2 ln ( 2 ) [1] _[≤]_ [2 ln] [(] [2] [)] _[d]_ [2] [.]
� _x_ [2]



4. Follows directly from the statement 1., the fact that _S_ ∞ _[′′]_ [(] _[x]_ [)] [ is increasing for] _[ x]_ _[ ∈]_ [[] [0, 1/10] []] [ and]
the above expression of _S_ ∞ _[′′]_ [this statement follows.]


5. This statement follows using the same reasoning with statement 3.


In this section we establish the bounds on the gradient and the hessian of P _**v**_ ( _**x**_ ) . These
bounds are formally stated in Lemma 8.11 the proof of which is the main goal of the section.


**Lemma 8.11.** _For any vertex_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _, it holds that_



_≤_ Θ ( _d_ 12 / _δ_ ) _,_
���



_1._


_2._



_∂_ P _**v**_ ( _**x**_ )
��� _∂x_ _i_



_∂_ 2 P _**v**_ ( _**x**_ )
��� _∂x_ _i_ _∂x_ _j_



24 2
_≤_ Θ ( _d_ / _δ_ ) _._
���



In order to prove Lemma 8.11. We first introduce several technical lemmas.


**Lemma C.6.** _Let_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _lying in cublet L_ ( _**c**_ ) _, with_ _**c**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _and let_ _**p**_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _[d]_ [)] _[ be the]_
_canonical representation of_ _**x**_ _. Then for all vertices_ _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _, it holds that_

_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] 11
### ≤ Θ ( d ) · ∑ Q [c] v [(] [x] [)] [.]
���� _∂p_ _i_ ���� _**v**_ _∈_ _V_ _c_


69


_Proof._ To simplify notation we use _Q_ _**v**_ ( _**x**_ ) instead of _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [,] _[ A]_ [ instead of] _[ A]_ _**[c]**_ _**v**_ [and] _[ B]_ [ instead of] _[ B]_ _**v**_ _**[c]**_
for the rest of the proof. Without loss of generality we assume that for all _j_ _∈_ _A_ and _ℓ_ _∈_ _B_,
_p_ _ℓ_ _>_ _p_ _j_ since otherwise _[∂][Q]_ _∂_ _**v**_ _**[c]**_ _p_ [(] _i_ _**[x]**_ [)] = 0 trivially by the Definition 8.7. Let _i_ _∈_ _B_ (symmetrically for
_i_ _∈_ _A_ ) then,

_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]

=

���� _∂p_ _i_ ����


_̸_


_̸_


_̸_


_̸_


_̸_


_̸_


_̸_


### = ∏ S ∞ ( S ( p ℓ ) − S ( p j )) · ℓ̸ = i j [∏] ∈ A

_̸_


_̸_


_̸_


_̸_


_̸_


_̸_




### ∑

_̸_ _j_ _∈_ _A_




_̸_


_̸_


_̸_


_̸_


_̸_


_̸_



_′_
### �� S ∞ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� ∏ S ∞ ( S ( p i ) − S ( p j ′ ))

_̸_ _j_ _[′]_ _∈_ _A_ / _{_ _j_ _}_


_̸_


_̸_


_̸_


_̸_


_̸_


_̸_



 _S_ _[′]_ ( _p_ _i_ )

_̸_




_̸_


_̸_


_̸_


_̸_


_̸_


_̸_



_̸_

### ≤ 6 ∑

_j_ _∈_ _A_ _̸_


_̸_


_̸_


_̸_


_̸_


_̸_



_̸_

### �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� · ∏ S ∞ ( S ( p ℓ ) − S ( p j ′ ))

( _j_ _[′]_, _ℓ_ ) _̸_ =( _j_, _i_ )


_̸_


_̸_


_̸_


_̸_


_̸_



_̸_


_̸_


where the last inequality follows by the fact that _|_ _S_ _[′]_ ( _·_ ) _| ≤_ 6. Since _|_ _A_ _| ≤_ _d_ the proof of the lemma
will be completed if we are able to show that for any _j_ _∈_ _A_, it holds that

### �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� · ∏ S ∞ ( S ( p ℓ ) − S ( p j ′ )) ≤ Θ ( d [10] ) · ∑ Q v ′ ( x )

( _j_ _[′]_, _ℓ_ ) _̸_ =( _j_, _i_ ) _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


_′_
In case _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ ) _≥_ 1/ _d_ [5] then by case 3. of Lemma C.5 we get that �� _S_ ∞ [(] _[S]_ [(] _[p]_ _i_ [)] _[ −]_ _[S]_ [(] _[p]_ _j_ [))] �� _≤_
_c_ _·_ _d_ [10] _·_ _S_ ∞ ( _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ )), which implies gthe following

### �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� · ∏ S ∞ ( S ( p ℓ ) − S ( p j ′ )) ≤

( _j_ _[′]_, _ℓ_ ) _̸_ =( _j_, _i_ )
### ≤ c · d [10] · S ∞ ( S ( p i ) − S ( p j )) · ∏ S ∞ ( S ( p ℓ ) − S ( p j ′ ))

( _j_ _[′]_, _ℓ_ ) _̸_ =( _j_, _i_ )

= _c_ _·_ _d_ [10] _·_ _Q_ _**v**_ ( _**x**_ )
### ≤ c · d [10] · ∑ Q v ′ ( x )

_**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


Now consider the case where _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ ) _≤_ 1/ _d_ [5] . Using case 2. of Lemma C.5, we have that

### �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� · ∏ S ∞ ( S ( p ℓ ) − S ( p j ′ )) ≤ �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� ≤ Θ ( d 10 · 2 − d 5 )

( _j_ _[′]_, _ℓ_ ) _̸_ =( _j_, _i_ )


Consider the sequence of points in the [ 0, 1 ] interval 0, _p_ 1, . . ., _p_ _d_, 1. There always exist two consecutive points with distance greater that 1/ ( _d_ + 1 ) . As a result, there exists _**v**_ _[∗]_ _∈_ _L_ _c_ ( _**c**_ ) such that
_p_ _ℓ_ _−_ _p_ _j_ _≥_ 1/ ( _d_ + 1 ) for all _ℓ_ _∈_ _B_ _**v**_ _[∗]_ and _j_ _∈_ _A_ _**v**_ _[∗]_ . Then _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ ) _≥_ 1/ ( _d_ + 1 ) [2] and by case 1.
of Lemma C.5, _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) _≥_ _c_ 2 _[−]_ [(] _[d]_ [+] [1] [)] [2] . If we also use the fact that _|_ _A_ _**v**_ _[∗]_ _| · |_ _B_ _**v**_ _[∗]_ _| ≤_ _d_ [2], we
get that
_Q_ _**v**_ _[∗]_ ( _**x**_ ) _≥_ ( _c_ _·_ 2 _[−]_ [(] _[d]_ [+] [1] [)] [2] ) _[d]_ [2] = _c_ _[d]_ [2] 2 _[−]_ [(] _[d]_ [+] [1] [)] [2] _[·]_ _[d]_ [2] .


Then it holds that
### Q v [∗] 1 ( x ) [·] �� S ∞ ′ [(] [S] [(] [p] i [)] [ −] [S] [(] [p] j [))] �� · ( j [′], ℓ ∏ ) ̸ =( j, i ) S ∞ ( S ( p ℓ ) − S ( p j ′ )) ≤


_≤_ Θ _d_ [10] _·_ ( 1/ _c_ ) _·_ 2 _[−]_ _[d]_ [3] [+(] _[d]_ [+] [1] [)] [2] [�] _[d]_ [2] [�] _≤_ Θ ( _d_ [10] ) .
�
�


Combining the later with the discussion in the rest of the proof the lemma follows.


70


_∂_ P _**v**_ ( _**x**_ )
**Lemma C.7.** _For any vertex_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _it holds that_ ��� _∂x_ _i_


_̸_


_̸_


_̸_



_≤_ Θ � _d_ [12] / _δ_ � _._
���


_̸_


_̸_


_̸_



_Proof._ To simplify notation we use _Q_ _**v**_ ( _**x**_ ) instead of _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [ for the rest of the proof. Without loss]
of generality we assume that _**x**_ lies on a cubelet _L_ ( _**c**_ ) with _**c**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ and _**v**_ _∈_ _L_ _c_ ( _**c**_ ), since
otherwise _[∂]_ [P] _∂_ _**[v]**_ _x_ [(] _i_ _**[x]**_ [)] = 0. Let _**p**_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _[d]_ [)] [ be the canonical representation of] _**[ x]**_ [ in the cubelet]

_L_ ( _**c**_ ) . Then it holds that


_̸_


_̸_


_̸_



����


_̸_


_̸_


_̸_



_∂_ P _**v**_ ( _**x**_ )
���� _∂p_ _i_


_̸_


_̸_


_̸_



=
����


_≤_


_̸_


_̸_


_̸_



�� _∂Q_ _**v**_ ( _**x**_ )
� _∂p_ _i_


_̸_


_̸_


_̸_



_∂Q_ _**v**_ ( _**x**_ ) _∂Q_ _**v**_ _′_ ( _**x**_ )
��� _∂p_ _i_ ��� ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) ��� _∂p_ _i_

∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) [+] ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ )


_̸_


_̸_


_̸_



_∂_ _**v**_ _p_ ( _i_ _**x**_ ) _·_ �∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) � _−_ _Q_ _**v**_ ( _**x**_ ) _·_ �∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _∂Q∂_ _**v**_ _p_ _′_ ( _i_ _**x**_ )


_̸_


_̸_


_̸_



( ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _′_ ( _**x**_ )) [2]


_̸_


_̸_


_̸_



∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ )


_̸_


_̸_


_̸_



_∂p_ _i_


_̸_


_̸_


_̸_



�� _∂Q_ _**v**_ ( _**x**_ )
� _∂p_ _i_


_̸_


_̸_


_̸_



��
�


_̸_


_̸_


_̸_



_∂p_ _i_


_̸_


_̸_


_̸_



_∂p_ _i_


_̸_


_̸_


_̸_



��
�


_̸_


_̸_


_̸_



_≤_ ( _d_ + 2 ) _·_ Θ ( _d_ [11] ) = Θ ( _d_ [12] )


where the last inequality follows by Lemma C.6 and the fact that at most _d_ + 1 vertices _**v**_ of _L_ _c_ ( _**c**_ )
have non-zero gradient as we have proved in Lemma 8.10. Then the proof of Lemma C.7 follows
by the fact that _p_ _i_ = _[x]_ _t_ _i_ _[i]_ _−_ _[−]_ _s_ _[s]_ _i_ _[i]_ [.]


_̸_


_̸_


_̸_



_∂_ 2 Q _**cv**_ [(] _**[x]**_ [)]
**Lemma C.8.** _Let_ _**c**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _and_ _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _then it holds that_ ��� _∂p_ _i_ _∂p_ _j_


_̸_


_̸_


_̸_



22 _**c**_
��� _≤_ Θ ( _d_ ) _·_ ∑ _**v**_ _∈_ _R_ _**c**_ ( _**x**_ ) _Q_ _**v**_ [(] _**[x]**_ [)] _[.]_


_̸_


_̸_


_̸_



_Proof._ To simplify the notation we use _CS_ ( _p_ _ℓ_ _−_ _p_ _m_ ) to denote _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _m_ )), _CS_ _[′]_ ( _p_ _ℓ_ _−_ _p_ _m_ )
to denote _|_ _S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _ℓ_ [)] _[ −]_ _[S]_ [(] _[p]_ _[m]_ [))] _[|]_ [,] _[ A]_ [ to denote] _[ A]_ _**[c]**_ _**v**_ [and] _[ B]_ [ to denote] _[ B]_ _**v**_ _**[c]**_ [for the rest of the proof. As]

in Lemma C.7, we assume that _p_ _ℓ_ _>_ _p_ _m_ for all _ℓ_ _∈_ _B_ and _m_ _∈_ _A_ since otherwise _[∂]_ _∂_ [2] _p_ [Q] _i_ _∂_ _**[v]**_ [(] _p_ _**[x]**_ _j_ [)] [=] [ 0. We]

have the following cases for the indices _i_ and _j_


_̸_


_̸_


_̸_




- If _i_, _j_ _∈_ _B_ then

_∂_ [2] Q _**v**_ ( _**x**_ )
���� _∂p_ _i_ _∂p_ _j_


_̸_


_̸_


_̸_



=
����


_̸_


_̸_


_̸_


### = ∑ CS [′] ( p i − p m 1 ) CS [′] ( p j − p m 2 ) · ∏ CS ( p ℓ − p m ) · S [′] ( p i ) S [′] ( p j )

_m_ 1, _m_ 2 _∈_ _A_ ( _m_, _ℓ_ ) _̸_ = _{_ ( _m_ 1, _i_ ), ( _m_ 2, _j_ ) _}_


_̸_


_̸_



_̸_
### ≤ 36 ∑ CS [′] ( p i − p m 1 ) CS [′] ( p j − p m 2 ) · ∏ CS ( p ℓ − p m )

_m_ 1, _m_ 2 _∈_ _A_ ( _m_, _ℓ_ ) _̸_ = _{_ ( _m_ 1, _i_ ), ( _m_ 2, _j_ ) _}_

� �� �
≜ _U_ ( _i_, _j_ )


_̸_



_̸_


.


_̸_


_̸_



_̸_


_̸_


If additionally it holds that _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _m_ 1 ) _≤_ 1/ _d_ [5] or _S_ ( _p_ _j_ ) _−_ _S_ ( _p_ _m_ 2 ) _≤_ 1/ _d_ [5], then by the
case 2. of Lemma C.5, we have that


_U_ ( _i_, _j_ ) _≤_ _CS_ _[′]_ ( _p_ _i_ _−_ _p_ _m_ 1 ) _·_ _CS_ _[′]_ ( _p_ _j_ _−_ _p_ _m_ 2 ) _≤_ Θ ( _d_ [10] _e_ _[−]_ _[d]_ [5] ) .


The latter follows from the fact that the function _S_ ∞ _[′]_ [(] _[·]_ [)] [ is bounded in the] [ [] [0, 1] []] [ interval and]
that _CS_ ( _p_ _ℓ_ _−_ _p_ _m_ ) _≤_ 1. With the exact same arguments as in Lemma C.6, we hence get that

### CS [′] ( p i − p m 1 ) CS [′] ( p j − p m 2 ) · Π ( m, ℓ ) ̸ = { ( m 1, i ), ( m 2, j ) } CS ( p ℓ − p m ) ≤ Θ ( d [10] ) ∑ Q [c] v [′] [(] [x] [)] [.]

_**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



_̸_


_̸_


_̸_


_∂_ 2 _Q_ _**v**_ ( _**x**_ )
Thus
��� _∂p_ _i_ _∂p_ _j_



_̸_


_̸_


_̸_


12 _**c**_
��� _≤_ Θ ( _d_ ) ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ [(] _**[x]**_ [)] [.]



_̸_


_̸_


_̸_


71


On the other hand if _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _m_ 1 ) _≥_ 1/ _d_ [5] and _S_ ( _p_ _j_ ) _−_ _S_ ( _p_ _m_ 2 ) _≥_ 1/ _d_ [5] then by case 1. of
Lemma C.5, _CS_ _[′]_ ( _p_ _i_ _−_ _p_ _m_ 1 ) _≤_ _c_ _·_ _d_ [10] _·_ _CS_ ( _p_ _i_ _−_ _p_ _m_ 1 ) and _CS_ _[′]_ ( _p_ _j_ _−_ _p_ _m_ 2 ) _≤_ _c_ _·_ _d_ [10] _·_ _CS_ ( _p_ _j_ _−_ _p_ _m_ 2 )


_̸_


_̸_


_̸_


_̸_


_̸_



_∂_ 2 _Q_ _**v**_ ( _**x**_ )
and thus _U_ ( _i_, _j_ ) _≤_ Θ ( _d_ [20] ) _·_ _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] [. Overall we get that] ��� _∂p_ _i_ _∂p_ _j_


_̸_


_̸_


_̸_


_̸_


_̸_



22 _**c**_
��� _≤_ Θ ( _d_ ) _·_ ∑ _**v**_ _′_ _∈_ _R_ _**c**_ ( _**x**_ ) _Q_ _**v**_ _[′]_ [(] _**[x]**_ [)] [.]


_̸_


_̸_


_̸_


_̸_


_̸_




- If _i_ _∈_ _B_ and _j_ _∈_ _A_ then

_∂_ [2] Q _**v**_ ( _**x**_ )

_≤_

���� _∂p_ _i_ _∂p_ _j_ ����


_̸_


_̸_


_̸_


_̸_


_̸_



_∂p_ _i_ _∂p_ _j_


_̸_


_̸_


_̸_


_̸_


_̸_



_≤_
����


_̸_


_̸_


_̸_


_̸_


_̸_


### ≤ ∑ CS [′] ( p i − p m 1 ) CS [′] ( p ℓ 2 − p j ) · ∏ CS ( p ℓ − p m ) · S [′] ( p i ) S [′] ( p j )

_m_ 1 _∈_ _A_, _ℓ_ 2 _∈_ _B_ ( _m_, _ℓ_ ) _̸_ = _{_ ( _i_, _m_ 1 ), ( _ℓ_ 2, _j_ ) _}_


_̸_


_̸_


_̸_


_̸_



_̸_


_̸_

������


_̸_


_̸_


_̸_



_̸_


+


_̸_


_̸_


_̸_


_̸_



_̸_


_′′_
### CS ( p i − p j ) · ∏ CS ( p ℓ − p m ) · S [′] ( p i ) S [′] ( p j )
������ ( _m_, _ℓ_ ) _̸_ =( _i_, _j_ )


_̸_


_̸_


_̸_



_̸_


_̸_

### ≤ Θ ( d [22] ) ∑ Q [c] v [(] [x] [) +] [ 36]

_**v**_ _∈_ _L_ _c_ ( _**c**_ ) _̸_


_̸_


_̸_



_̸_


_̸_


_′′_
### CS ( p i − p j ) · ∏ CS ( p ℓ − p m )
������ ( _m_, _ℓ_ ) _̸_ =( _i_, _j_ ) ������

� �� �
_Q_ _[′′]_ ( _**x**_ )


_̸_


_̸_



_̸_


_̸_


.


_̸_


_̸_


_̸_



_̸_


_̸_


_̸_


_′′_
In case _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ ) _≥_ 1/ _d_ [5] then by case 4. of Lemma C.5, we get that _CS_ ( _p_ _i_ _−_ _p_ _j_ ) _≤_
��� ���

_cd_ [20] _·_ _CS_ ( _p_ _i_ _−_ _p_ _j_ ) which implies that _Q_ _′′_ _≤_ Θ ( _d_ 20 ) _·_ _Q_ _**cv**_ [(] _**[x]**_ [)] [.]

On the other hand if _S_ ( _p_ _i_ ) _−_ _S_ ( _p_ _j_ ) _≤_ 1/ _d_ [5] then by case 5. of Lemma C.5, we get that

_′′_ _′′_ 20 _−_ _d_ 5
_Q_ _≤_ _CS_ ( _p_ _i_ _−_ _p_ _j_ ) _≤_ _c_ _·_ _d_ _e_ . As in the proof of Lemma C.6, there exists a vertex
��� ���

_**v**_ _[∗]_ _∈_ _R_ _**c**_ ( _**x**_ ) such that _Q_ _**[c]**_ _**v**_ _[∗]_ [(] _**[x]**_ [)] _[ ≥]_ _[c]_ _[d]_ [2] _[e]_ _[−]_ [(] _[d]_ [+] [1] [)] [2] _[d]_ [2] [ and thus] _[ Q]_ _′′_ _≤_ Θ ( _d_ 20 ) ∑ _**v**_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**cv**_ [(] _**[x]**_ [)] [. Overall]
we get that
### ∂ [2] Q v ( x ) ≤ Θ ( d 22 ) ∑ Q [c] v [(] [x] [)] [.]
���� _∂p_ _i_ _∂p_ _j_ ����


_̸_


_̸_



_̸_


_̸_


_̸_


_∂p_ _i_ _∂p_ _j_


_̸_


_̸_



_̸_


_̸_


_̸_

### ≤ Θ ( d 22 ) ∑ Q [c] v [(] [x] [)] [.]

���� _**v**_ _∈_ _L_ _c_ ( _**c**_ )


_̸_


_̸_



_̸_


_̸_


_̸_


- If _i_ = _j_ _∈_ _B_ then

_∂_ [2] Q _**v**_ ( _**x**_ )
���� _∂_ [2] _p_ _i_ ����


_̸_


_̸_



_̸_


_̸_


_̸_


_∂_ [2] _p_ _i_


_̸_


_̸_



_̸_


_̸_


_̸_


_≤_
����


_̸_


_̸_



_̸_


_̸_


_̸_


_̸_ �����


_̸_



_̸_


_̸_


_̸_

### ≤ ∑

_m_ 1, _m_ 2 _∈_ _A_ _̸_

### + ∑

_m_ 1 _∈_ _A_ _̸_



_̸_


_̸_


_̸_

### CS ′ ( p i − p m 1 ) CS ′ ( p i − p m 2 ) · ∏ CS ( p ℓ − p m ) · S [′] ( p i ) S [′] ( p i )

����� ( _m_, _ℓ_ ) _̸_ = _{_ ( _m_ 1, _i_ ), ( _m_ 2, _i_ ) _}_


_̸_



_̸_


_̸_


_̸_


_̸_

### CS ′′ ( p i − p m 1 ) · ∏ CS ( p ℓ − p m ) S [′] ( p i ) S [′] ( p i )

����� ( _m_, _ℓ_ ) _̸_ =( _m_ 1, _ℓ_ )



_̸_


_̸_


_̸_


_̸_

_̸_ �����



_̸_


_̸_


_̸_


_̸_


_̸_
### ≤ Θ ( d [22] + d · d [20] ) · ∑ Q [c] v [(] [x] [)] [.]

_**v**_ _∈_ _L_ _c_ ( _**c**_ )


If we combine all the above cases then the Lemma follows.


_∂_ 2 P _**v**_ ( _**x**_ )
**Lemma C.9.** _For any vertex_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _, it holds that_ ��� _∂x_ _i_ _∂x_ _j_


72



_̸_


_̸_


_̸_


_̸_


_̸_


24 2
_≤_ Θ ( _d_ / _δ_ ) _._
���


_Proof._ Without loss of generality we assume that _**v**_ _∈_ _L_ _c_ ( _**c**_ ), where _**c**_ _∈_ ([ _N_ _−_ 1 ] _−_ 1 ) _[d]_ such that

_**x**_ _∈_ _L_ ( _**c**_ ), since otherwise _[∂]_ _∂_ [2] _x_ [P] _i_ _∂_ _**[v]**_ [(] _x_ _**[x]**_ _j_ [)] [=] [ 0.]



_∂_ [2] P _**v**_ ( _**x**_ )



_∂_ [2] P _**v**_ ( _**x**_ ) _∂_ [2] _Q_ _**v**_ ( _**x**_ )

=
_∂p_ _i_ _∂p_ _j_ _∂p_ _i_ _∂p_ _j_



3
1
### ∑ Q v ′ ( x ) · 4

� _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ ) � ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~



_∂p_ _i_ _∂p_ _j_



_∂Q_ _**v**_ ( _**x**_ )
### + ∑

_∂p_ _i_ _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


### ∑ Q v ′ ( x )

� _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


### ∑ Q v ′ ( x )

� _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



2
1

_·_ 4
� ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~


2
1

_·_ 4
� ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~



_−_ _∂Q_ _**v**_ _′_ ( _**x**_ )
### ∑

_∂p_ _j_ _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



_∂Q_ _**v**_ _′_ ( _**x**_ )

_∂p_ _j_


_∂Q_ _**v**_ _′_ ( _**x**_ )

_∂p_ _i_



2
1

_·_ 4
� ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~


### ∑ Q v ′ ( x )

� _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


### − Q v ( x ) ∑

_**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



_∂_ [2] _Q_ _**v**_ _′_ ( _**x**_ )

_∂p_ _i_ _∂p_ _j_


### − ∂Q v ( x ) ∑ Q v ′ ( x ) · 2 ∑ Q v ′ ( x ) ∑

_∂p_ _i_ _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ ) _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ ) _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



_∂Q∂_ _**v**_ _p_ _′_ ( _j_ _**x**_ ) _·_ ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ 1 ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~ 4

_∂Q∂_ _**v**_ _p_ _′_ ( _j_ _**x**_ ) _·_ ~~�~~ ∑ _**v**_ _′_ _∈_ _L_ _c_ ( _**c**_ 1 ) _Q_ _**v**_ _[′]_ ( _**x**_ ) ~~�~~ 4


### + Q v ( x ) ∑

_**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )


### ∂Q v ′ ( x ) · 2 ∑ Q v ′ ( x ) ∑

_∂p_ _i_ _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ ) _**v**_ _[′]_ _∈_ _L_ _c_ ( _**c**_ )



Using Lemma C.8 and Lemma C.6 we can bound every term in the above expression and hence



_∂_ 2 P _**v**_ ( _**x**_ )
we get that ��� _∂p_ _i_ _∂p_ _j_



��� _≤_ Θ ( _d_ 24 ) . Then the lemma follows from the fact that _∂∂xp_ _ii_ [=] [ 1/] _[δ]_ [.]



Finally using Lemma C.7 and Lemma C.9 we get the proof of Lemma 8.11.


**C.3** **Proof of Lemma 8.12**


Let 0 _≤_ _x_ _i_ _<_ 1/ ( _N_ _−_ 1 ) and _**c**_ = ( _c_ 1, . . ., _c_ _i_, . . ., _c_ _d_ ) denote down-left corner of the cubelet _R_ ( _**x**_ )
at which _**x**_ _∈_ [ 0, 1 ] _[d]_ lies, i.e. _**x**_ _∈_ _L_ ( _**c**_ ) . Since _**x**_ _≤_ 1/ ( _N_ _−_ 1 ), this means that _c_ _i_ = 0. By the
definition of _sources and targets_ in Definition 8.6, we have that _s_ _i_ = 0 and _t_ _i_ = 1/ ( _N_ _−_ 1 ), where _s_ _i_,
_t_ _i_ are respectively the _i_ -th coordinate of the source _**s**_ _**c**_ and the target _**t**_ _**c**_ vertex. Let the canonical
representation _p_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _d_ [)] [ of] _**[ x]**_ [ in the cubelet] _[ L]_ [(] _**[c]**_ [)] [. Now partition the coordinates] [ [] _[d]_ []] [ in the]
following sets
_A_ = � _j_ _|_ _p_ _j_ _≤_ _p_ _i_ � and _B_ = � _j_ _|_ _p_ _i_ _<_ _p_ _j_ � .


If _B_ = ∅ then notice that P _**s**_ _**c**_ ( _**x**_ ) _>_ 0, since _p_ _i_ _<_ 1, by the fact that _x_ _i_ _<_ 1/ ( _N_ _−_ 1 ) . Thus the
lemma follows since _s_ _i_ = 0. So we may assume that _B_ _̸_ = ∅ . In this case consider the corner
_**v**_ = ( _v_ 1, . . ., _v_ _d_ ) defined as follows


_v_ _j_ = _s_ _j_ _j_ _∈_ _A_ .
� _t_ _j_ _j_ _∈_ _B_


Observe that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ >]_ [ 0 and thus] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ [)] [. Moreover the coordinate] _[ i]_ _[ ∈]_ _[A]_ [ and therefore it]
holds that _v_ _i_ = _s_ _i_ = 0. This proves the first statement of the Lemma.
For the second statement let 1 _−_ 1/ ( _N_ _−_ 1 ) _≤_ _x_ _i_ _≤_ 1/ ( _N_ _−_ 1 ) and _**c**_ = ( _c_ 1, . . ., _c_ _i_, . . ., _c_ _d_ )
denote down-left corner of the cubelet _R_ ( _**x**_ ) at which _**x**_ _∈_ [ 0, 1 ] _[d]_ lies, i.e. _**x**_ _∈_ _L_ ( _**c**_ ) . This means
that _c_ _i_ = _N_ _[N]_ _−_ _[−]_ 1 [2] [.]


73


 - Let _N_ be odd. In this case by the definition of sources and targets in Definition 8.6, we have
that _s_ _i_ = 1 _−_ 1/ ( _N_ _−_ 1 ) and _t_ _i_ = 1, where _s_ _i_, _t_ _i_ are respectively the _i_ -th coordinate of the
source and target vertex. Let _p_ _**[c]**_ _**x**_ [= (] _[p]_ [1] [, . . .,] _[ p]_ _d_ [)] [ be the canonical representation of] _**[ x]**_ [ under]
in the cubelet _L_ ( _**c**_ ) . Now partition the coordinates [ _d_ ] as follows,


_A_ = � _j_ _|_ _p_ _j_ _<_ _p_ _i_ � and _B_ = � _j_ _|_ _p_ _i_ _≤_ _p_ _j_ �


If _A_ = ∅ then notice that for the target vertex _**t**_ _**c**_, P _**t**_ _**c**_ ( _**x**_ ) _>_ 0, since _p_ _i_ _>_ 0, by the fact that
_x_ _i_ _>_ 1 _−_ 1/ ( _N_ _−_ 1 ) . Thus the lemma follows since _t_ _i_ = 1. So we may assume that _A_ _̸_ = ∅ .
In this case consider the corner _**v**_ = ( _v_ 1, . . ., _v_ _d_ ) defined as follows,


_s_ _j_ _j_ _∈_ _A_
_v_ _j_ =
� _t_ _j_ _j_ _∈_ _B_


Observe that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ >]_ [ 0 and thus] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ [)] [. Moreover the coordinate] _[ i]_ _[ ∈]_ _[B]_ [ and thus]
_v_ _i_ = _t_ _i_ = 1.


 - Let _N_ be even. In this case we have that _t_ _i_ = 1 _−_ 1/ ( _N_ _−_ 1 ) and _s_ _i_ = 1. Now partition the
coordinates [ _d_ ] as follows,


_A_ = � _j_ _|_ _p_ _j_ _≤_ _p_ _i_ � and _B_ = � _j_ _|_ _p_ _i_ _<_ _p_ _j_ �


If _B_ = ∅ then notice that for the source vertex _**s**_ _**c**_, P _**s**_ _**c**_ ( _**x**_ ) _>_ 0, since _p_ _i_ _<_ 1, by the fact that
_x_ _i_ _>_ 1 _−_ 1/ ( _N_ _−_ 1 ) . Thus the lemma follows since _s_ _i_ = 1. In case _B_ _̸_ = ∅ consider the
corner _**v**_ = ( _v_ 1, . . ., _v_ _d_ ) defined as follows,


_s_ _j_ _j_ _∈_ _A_
_v_ _j_ =
� _t_ _j_ _j_ _∈_ _B_


Observe that _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ >]_ [ 0 and thus] _**[ v]**_ _[ ∈]_ _[R]_ [+] [(] _**[x]**_ [)] [. Moreover the coordinate] _[ i]_ _[ ∈]_ _[A]_ [ and thus]
_v_ _i_ = _s_ _i_ = 1.


If we put together the last two cases then this implies the second statement of the lemma.

#### **D Constructing the Turing Machine – Proof of Theorem 7.6**


In this section we prove Theorem 7.6 establishing that both the function _f_ _C_ _l_ ( _**x**_, _**y**_ ) of Definition 7.4
and its gradient, is computable by a polynomial-time Turing Machine. We prove Theorem 7.6
through a series of Lemmas. To simplify notation we set _b_ ≜ log 1/ _ε_ .


**Definition D.1.** For a _x_ _∈_ **R**, we denote by [ _x_ ] _b_ _∈_ **R**, a value represented by the _b_ bits such that


_|_ [ _x_ ] _b_ _−_ _x_ _| ≤_ 2 _[−]_ _[b]_ .


**Lemma D.2.** _There exist Turing Machines M_ _S_ ∞ _, M_ _S_ ∞ _′_ _that given input x_ _∈_ [ 0, 1 ] _and ε in binary form,_
_compute_ [ _S_ ∞ ( _x_ )] _b_ _and_ [ _S_ ∞ _[′]_ [(] _[x]_ [)]] _b_ _[in time polynomial in b]_ [ =] [ log] [(] [1/] _[ε]_ [)] _[ and the binary representation of x.]_


_Proof._ The Turing Machine _M_ _S_ ∞ outputs the fist _b_ bits of the following quantity,



_b_ _[′]_



_W_ ( _x_ ) =







1

1 + 2 [[] _[−]_ [1] _x_

 ~~�~~



1 + 2 [[] _[−]_ [1] _x_
~~�~~




[1] 1

_x_ [+] _x_ _−_ 1 []] _b_ _′_
~~�~~



_b_ _[′]_









74


where _b_ _[′]_ will be selected sufficiently large. Notice it is possible to compute the above quantity
due to the fact that all functions [1] [+] 1 [, 2] _[γ]_ [ and] 1 _[−]_ _[b]_ _[′]_ [ in]



due to the fact that all functions _γ_ [1] [+] _γ_ _−_ 1 1 [, 2] _[γ]_ [ and] 1 + 1 _γ_ [can be computed with accuracy 2] _[−]_ _[b]_ _[′]_ [ in]

polynomial time with respect to _b_ _[′]_ and the binary representation of _γ_ [Bre76]. Moreover,


 1  1

1 _−_ 1

1 + 2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 []] _b_ _′_ 1 + 2 _[−]_ [1] _x_ [+] _x_ _−_ 1

������� ~~�~~ ~~�~~ _b_ _[′]_  _b_ _[′]_ �������







1

_−_



1

1 + 2 [[] _[−]_ [1] _x_

 ~~�~~



1 + 2 [[] _[−]_ [1] _x_
~~�~~




[1] 1

_x_ [+] _x_ _−_ 1 []] _b_ _′_
~~�~~



_b_ _[′]_









1 + 2 _[−]_ [1] _x_




[1] 1

_x_ [+] _x_ _−_ 1



�������



_b_ _[′]_



_≤_

�������



_b_ _[′]_







1

1 + 2 [[] _[−]_ [1] _x_

 ~~�~~




[1] 1

_x_ [+] _x_ _−_ 1 []] _b_ _′_
~~�~~



1 + 2 [[] _[−]_ [1] _x_
~~�~~



_b_ _[′]_









�������



1

_−_ 1

1 + 2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 []] _b_ _′_
~~�~~ ~~�~~



1



1

_−_

1 + 2 [[] _[−]_ [1] _x_

_b_ _[′]_



+


+



�������



1
����� 1 + 2 [[] _[−]_ [1] _x_



1 + 2 [[] _[−]_ [1] _x_



1


[1] 1

_x_ [+] _x_ _−_ 1 []] _b_ _′_ _[−]_ 1 + 2 _[−]_



1 + 2 _[−]_ [1] _x_




[1] 1

_x_ [+] _x_ _−_ 1



1

1 + 2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 []] _b_ _′_
~~�~~ ~~�~~




[1] 1

_x_ [+] _x_ _−_ 1 []] _b_ _′_



_b_ _[′]_

�������



1 + 2 [[] _[−]_ [1] _x_



�����



1

_≤_ 2 _[−]_ _[b]_ _[′]_ + 2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 []] _b_ _′_
���� �



_≤_ 2 _[−]_ _[b]_ _[′]_ + 2 [[] _[−]_ [1] _x_
����



1
_x_ [+] _x_ _−_ 1 []] _b_ _′_

_b_ _[′]_ _[ −]_ [2] [[] _[−]_ [1] ���



_x_

_b_ _[′]_ _[ −]_ [2] [[] _[−]_ [1]



+ ln 2
����

_≤_ 4 _·_ 2 _[−]_ _[b]_ _[′]_



1

_−_ [1]
� _x_ [+] _x_ _−_ 1



_−_ [1]
� _x_



�



1

_−_ [1]
_b_ _[′]_ _[ −]_ � _x_ [+] _x_ _−_ 1



_−_ [1]
_b_ _[′]_ _[ −]_ � _x_



�����



where the first inequality follows from triangle inequality and the second follows from the facts
that 1/ ( 1 + _γ_ ) is a 1-Lipschitz function of _γ_ for _γ_ _≥_ 0, and 1/ ( 1 + 2 _[γ]_ ) is an ln ( 2 ) -Lipschitz
function of _γ_ for _γ_ _≥_ 0. The last inequality follows from the definition of [ _·_ ] _b_ _′_ . Hence _W_ ( _x_ ) is
indeed equal to [ _S_ ∞ ( _x_ )] _b_ if we choose _b_ _[′]_ = _b_ + 2.

Next we explain how _M_ _S_ ∞ _′_ computes [ _S_ ∞ _[′]_ [(] _[x]_ [)]] _b_ [. First notice that] _[ S]_ ∞ _[′]_ [(] _[x]_ [)] [ is equal to]



_x_ _−_ 1 1 _−_ ( _x_ _−_ 11 ) [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1

1 2

2 _[−]_ [1] _x_ + 2 _x_ _−_ 1
~~�~~ ~~�~~



_S_ ∞ _[′]_ [(] _[x]_ [) =] [ ln 2] _[ ·]_



_x_ 1 [2] [2] _[−]_ [1] _x_




[1] _x_ [+] _x_ _−_ 1 1 _−_ ( _x_ _−_ 11 ) [2] [2] _[−]_ [1] _x_




[1] _x_ + 2 _x_ _−_ 1 1 2 .

~~�~~



To describe how to compute _S_ ∞ _[′]_ [(] _[x]_ [)] [ we first assume that we have computed the following quan-]
tities. Then based on these quantities we show how _S_ ∞ _[′]_ [(] _[x]_ [)] [ can be computed and finally we]
consider the computation of these quantities.


_▷_ [ ln 2 ] _b_ _′_,




_▷_ _A_ _←_ � _x_ 1 [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 �



_b_ _[′]_ [,]




_▷_ _B_ _←_ � ( _x_ _−_ 11 ) [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 �



_b_ _[′]_ [,]



1 2 [�]

_▷_ _C_ _←_ 2 _[−]_ [1] _x_ + 2 _x_ _−_ 1

�

�� _b_ _[′]_ [.]



75


_A_ + _B_
Then _M_ _S_ ∞ _′_ outputs the fist _b_ bits of the quantity � [ ln 2 ] _b_ _′_ _·_ � _C_



_b_ _[′]_ �



_C_ + _B_ �



_b_ bits of the quantity � [ ln 2 ] _b_ _′_ _·_ � _C_ � _b_ _[′]_ � _b_ _[′]_ [. We now prove that]


_A_ + _B_

[ ln 2 ] _b_ _′_ _[A]_ [ +] _[ B]_ _≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�]
� _C_ � _b_ _[′]_ _[ −]_ [ln 2] � �� _C_ � �
�������� _S_ _[′]_ ( _x_ ) ��������



_C_
� �� �
_S_ ∞ _[′]_ ( _x_ )



_≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�]
�
��������



_C_



�




_[A]_ [ +] _[ B]_

_C_

_b_ _[′]_ _[ −]_ [ln 2]



Consider the function _g_ ( _α_, _β_, _γ_ ) = _[α]_ [+] _γ_ _[β]_ where _|_ _α_ _|_, _|_ _β_ _| ≤_ _c_ 1 and _|_ _γ_ _| ≥_ _c_ 2 where _c_ 1, _c_ 2 are universal



constants. Notice that _g_ ( _α_, _β_, _γ_ ) is _c_ -Lipschitz for _c_ =
�



2
_c_ [2] 2 [+] [ 2] _c_ _[c]_ [2] 2 [1] [. Since for sufficiently large] _[ b]_ _[′]_



all the quantities _|_ _A_ _|_, _|_ _B_ _|_, ��� _x_ 1 [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 ���, ��� ( _x_ _−_ 11 ) [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 ��� _≤_ _c_ 1 and _|_ _C_ _|_, �2 _[−]_ [1] _x_ + 2 _x_ _−_ 1 1 � 2 _≥_ _c_ 2 where

_c_ 1, _c_ 2 are universal constants we get that


_A_ + _B_

_≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�] .

����� _C_ � _b_ _[′]_ _[ −]_ _[A]_ [ +] _C_ _[ B]_ ���� �



all the quantities _|_ _A_ _|_, _|_ _B_ _|_, ��� _x_ 1




[1] _x_ [+] _x_ _−_ 1 1 1 _x_

���, ��� ( _x_ _−_ 1 ) [2] [2] _[−]_ [1]



_x_ 1 [2] [2] _[−]_ [1] _x_




[1] 1

_x_ [+] _x_ _−_ 1 _≤_ _c_ 1 and _|_ _C_ _|_, 2 _[−]_ [1] _x_
��� �



_A_ + _B_
� _C_



�



_C_

_b_ _[′]_ _[ −]_ _[A]_ [ +] _[ B]_



_≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�] .
�
����



_C_



_C_



Now consider the function _g_ ( _α_, _β_ ) = _α_ _·_ _β_ where _|_ _α_ _|_, _|_ _β_ _| ≤_ _c_ where _c_ is a universal constant.
In this case _g_ ( _α_, _β_ ) is _√_ 2 _c_ -Lipschitz continuous. Since for _b_ _[′]_ sufficiently large all the quantities



In this case _g_ ( _α_, _β_ ) is _√_ 2 _c_ -Lipschitz continuous. Since for _b_ _[′]_ sufficiently large all the quantities

_A_ + _B_ _A_ + _B_
_|_ [ ln 2 ] _b_ _′_ _|_, ��� _C_ � _b_ _[′]_ ��, ln 2, �� _C_ �� are bounded by a universal constant _c_, we have that,



_C_ + _B_ �



_b_ _[′]_ ��, ln 2, �� _AC_ + _B_



+ _B_

_C_ �� are bounded by a universal constant _c_, we have that,

_A_ + _B_

[ ln 2 ] _b_ _′_ _[A]_ [ +] _[ B]_ _≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�]
���� � _C_ � _b_ _[′]_ _[ −]_ [ln 2] _C_ ���� �



_C_



�




_[A]_ [ +] _[ B]_

_C_

_b_ _[′]_ _[ −]_ [ln 2]



_C_



_≤_ Θ 2 _[−]_ _[b]_ _[′]_ [�]
�
����



Next we explain how the values _A_, _B_ and _C_ are computed while [ ln ( 2 )] _[′]_ _b_ [can easily be computed]
via standard techniques [Bre76].




- **Computation of** _**A**_ **.** The Turing Machine _M_ _S_ ∞ _′_ will compute _A_ by taking the first _b_ _[′]_ bits of
the following quantity,

1

�2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 [+] [2 ln] _[ x]_ [/ ln 2] []] _b_ _′′_ � _b_ _[′′]_




[1] 1

_x_ [+] _x_ _−_ 1 [+] [2 ln] _[ x]_ [/ ln 2] []] _b_ _′′_
�



_b_ _[′′]_
where _b_ _[′′]_ will be taken sufficiently large. We remark that both where both the exponentiation and the natural logarithm can be computed in polynomial-time with respect to the
number of accuracy bits and the binary representation of the input [Bre76]. The function
_x_ 1 [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 = 2 _[−]_ [1] _x_ [+] _x_ _−_ 1 1 [+] [2 ln] _[ x]_ [/ ln 2] is _c_ -Lipschitz where _c_ is a universal constant. Thus,




[1] 1

_x_ [+] _x_ _−_ 1 = 2 _[−]_ [1] _x_




[1] 1

_x_ [+] _x_ _−_ 1 [+] [2 ln] _[ x]_ [/ ln 2] is _c_ -Lipschitz where _c_ is a universal constant. Thus,
�����2 [[] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 [+] [2 ln] _[ x]_ [/ ln 2] []] _b_ _′′_ � _b_ _[′′]_ _[ −]_ _x_ [1] [2] [2] _[−]_ [1] _x_ [+] _x_ _−_ 1 1 ���� _≤_ Θ ( 2 _−_ _b_ _′′_ ) .



_b_ _[′′]_ _[ −]_ _x_ [1]




[1] _x_ [+] _x_ _−_ 1 1 _≤_ Θ ( 2 _−_ _b_ _′′_ ) .

����




[1] 1

_x_ [+] _x_ _−_ 1 [+] [2 ln] _[ x]_ [/ ln 2] []] _b_ _′′_
�




[2] _[−]_ [1] _x_
_x_ [2]




- **Computation of** _**B**_ **.** Using the same arguments as for _A_ .




- **Computation of** _**C**_ **.** To compute _C_ we first compute _b_ _[′′]_ bits of the following quantity,


2

 1 



2



_b_ _[′′]_


2
_≤_ Θ 2 _[−]_ _[b]_ _[′′]_ [�]
�
�
�������



1
 ~~�~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]




[1] _x_ []] _b_ _′′_

~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_









We first argue that




2

������� ~~�~~







2


_b_ _[′′]_



1
 ~~�~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]




[1] _x_ []] _b_ _′′_

~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_



1

_−_
� 2 _[−]_ [1] _x_




[1] _x_ + 2 _x_ _−_ 1 1



2 _[−]_ [1] _x_









The latter follows by applying the triangle inequality and the following 3 inequalities.


76


1.


2.


3.



�������











2 [�]

_≤_ Θ ( 2 _[−]_ _[b]_ _[′′]_ )
������



1
 ~~�~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]







_−_



1

 ~~�~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]




[1] _x_ []] _b_ _′′_

~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_









2


_b_ _[′′]_




[1] _x_ []] _b_ _′′_

~~�~~










this holds since for _b_ _[′′]_ _>_ 1 we have

 1







1
 ~~��~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]




[1] _x_ []] _b_ _′′_

~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_ ~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_ ~~�~~









_b_ _[′′]_



1
and
~~��~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]




[1] _x_ []] _b_ _′′_

~~�~~



are both upper-bounded by 2 while the function _g_ ( _α_ ) = _α_ [2] is 4-Lipschitz for _|_ _α_ _| ≤_ 2.



�������



2







1

 ~~�~~ 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ ~~�~~ _b_ _[′′]_ [ +]



_−_



1
� 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ +




[1] _x_ []] _b_ _′′_

~~�~~



1
_b_ _[′′]_ [ +] ~~�~~ 2 [[] _x_ _−_ 1 []] _b_ _′′_ ~~�~~



_b_ _[′′]_



2 _[−]_ [[] [1] _x_




[1] _x_ []] _b_ _′′_ + 2 [[] _x_ _−_ 1 1 []] _b_ _′′_



2
_≤_ Θ 2 _[−]_ _[b]_ _[′′]_ [�]
�
�
�������











[1] _x_ []] _b_ _′′_

�



The latter follows since for _b_ _[′′]_ larger than a universal constant, both 2 _[−]_ [[] [1] _x_
�



The latter follows since for _b_ larger than a universal constant, both 2 _x_ _b_ _′′_

_b_ _[′′]_ [ +]
2 [[] _x_ _−_ 1 1 []] _b_ _′′_ [1] _x_ []] _b_ _′′_ + 2 [[] _x_ _−_ 1 1 []] _b_ _′′_ are greater than a universal constant _c_, while the
� � _b_ _[′′]_ [ and 2] _[−]_ [[]




[1] _x_ []] _b_ _′′_ + 2 [[] _x_ _−_ 1 1 []] _b_ _′′_ are greater than a universal constant _c_, while the




[1] _x_
_b_ _[′′]_ [ and 2] _[−]_ [[]



function _g_ ( _α_, _β_ ) = 1/ ( _α_ + _β_ ) [2] is Θ � _c_ [3] [�] -Lipschitz for _α_ + _β_ _≥_ _c_ .



������



1
� 2 _[−]_ [[] [1] _x_ []] _b_ _′′_ +



2 _[−]_ [[] [1] _x_




[1] _x_ []] _b_ _′′_ + 2 [[] _x_ _−_ 1 1 []] _b_ _′′_



2
1

_−_
� � 2 _[−]_ [1] _x_ +




[1] _x_ + 2 _x_ _−_ 1 1



2
_≤_ Θ 2 _[−]_ _[b]_ _[′′]_ [�]
�
�
������



2 _[−]_ [1] _x_



The latter follows since for _b_ _[′′]_ larger than a universal constant it holds that both the
quantities in the left hand side are greater than a positive universal constant _c_, while
the function _g_ ( _α_, _β_ ) = 1/ ( 2 _[−]_ _[α]_ + 2 _[β]_ ) for 2 _[−]_ _[α]_ + 2 _[β]_ _≥_ _c_, _α_ _≥_ 0, and _β_ _≤_ 0 is Θ �1/ _c_ [3] [�]      Lipschitz.


This concludes the proof of the lemma.


**Lemma D.3.** _There exist Turing Machines M_ _Q_ _and M_ _Q_ _′_ _that given_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _and ε_ _>_ 0 _in binary form,_
_respectively compute_ [ _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]] _b_ _[and]_ [ [] _[∇]_ _[Q]_ _**v**_ _**[c]**_ [(] _**[x]**_ [)]] _b_ _[for all vertices]_ _**[ v]**_ _[ ∈]_ [([] _[N]_ []] _[ −]_ [1] [)] _[d]_ _[ with Q]_ _**v**_ _**[c]**_ [(] _**[x]**_ [)] _[ >]_ [ 0] _[, where]_
_b_ = log ( 1/ _ε_ ) _. These vertices are most d_ + 1 _. Moreover both M_ _Q_ _and M_ _Q_ _′_ _run in polynomial time with_
_respect to b, d and the binary representation of_ _**x**_ _._


_Proof._ Both _M_ _Q_, _M_ _Q_ _′_ firsts compute the canonical representation _p_ _**[c]**_ _**x**_ _[∈]_ [[] [0, 1] []] _[d]_ [ with the respect to]
the cell _R_ ( _**x**_ ) in which _**x**_ lies. Such a cell _R_ ( _**x**_ ) can be computed by taking the first ( log _N_ + 1 ) -bits
at each coordinate of _**x**_ . The source vertex _**s**_ _**[c]**_ = ( _s_ 1, . . ., _s_ _d_ ) and the target vertex _**t**_ _**[c]**_ = ( _t_ 1, . . ., _t_ _d_ )
with respect to _R_ ( _**x**_ ) are also computed. Once this is done we are only interested in vertices
_**v**_ _∈_ _R_ _**c**_ ( _**x**_ ) for which
_p_ _ℓ_ _>_ _p_ _j_ for all _ℓ_ _∈_ _A_ _**[c]**_ _**v**_ [,] _[ j]_ _[ ∈]_ _[B]_ _**v**_ _**[c]**_


77


since for all the other _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ both _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0 and] _[ ∇]_ _[Q]_ _**[c]**_ _**v**_ [(] _**[x]**_ [) =] [ 0. These vertices, that are]
denoted by _R_ + ( _**x**_ ), are at most _d_ + 1 and can be computed in polynomial time.
The vertices _**v**_ _∈_ _R_ + ( _**x**_ ) can be computed in polynomial time as follows: **(i)** the coordinates
_p_ 1, . . ., _p_ _d_ are sorted in increasing order **ii)** for each _m_ = 0, . . ., _d_ compute the vertex _**v**_ _[m]_ _∈_ _R_ _**c**_ ( _**x**_ ),


_s_ _j_ if coordinate _j_ belongs in the first _m_ coordinates wrt the order of _**p**_ _**[c]**_ _**x**_
_**v**_ _[m]_ _j_ [=] � _t_ _j_ if coordinate _j_ belongs in the last _d_ _−_ _m_ coordinates wrt the order of _**p**_ _**[c]**_ _**x**_



By Definition 8.7 it immediately follows that _R_ + ( _**x**_ ) _⊆_ [�] _[d]_ _m_ = 0 _[{]_ _**[v]**_ _[m]_ _[}]_ [ which also establish that]
_|_ _R_ + ( _**x**_ ) _| ≤_ _d_ + 1.
Once _R_ + ( _**x**_ ) is computed, _M_ _Q_ computes for each pair ( _ℓ_, _j_ ) _∈_ _B_ _**v**_ _**[c]**_ _[×]_ _[ A]_ _**[c]**_ _**v**_ [the value of the number]
� _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) � _b_ _[′]_ [ for some accuracy] _[ b]_ _[′]_ [ that we determine later but depends polynomially on]



� _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) � _b_ _[′]_ [ for some accuracy] _[ b]_ _[′]_ [ that we determine later but depends polynomially on]

_b_, _d_ and the input accuracy of _**x**_ . Then each _**v**_ _∈_ _R_ + ( _**x**_ ), _M_ _Q_ outputs as [ _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]] _b_ [the fist] _[ b]_ [ bits of]
the following quantity

### ∏ � S ∞ ( S ( p ℓ ) − S ( p j )) � b [′]
� _ℓ∈_ _B_ _**v**_ _**[c]**_, _j_ _∈_ _A_ _**[c]**_ _**v**_ � _b_ _[′]_


where _b_ _[′]_ is selected sufficiently large. We next prove that this computation indeed outputs

[ _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]] _b_ [accurately.]

To simplify notation let _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) be denoted by _S_ _ℓ_ _j_, _A_ _**[c]**_ _**v**_ [denoted by] _[ A]_ [ and] _[ B]_ _**v**_ _**[c]**_ [denoted]
by _B_ . Then,

����Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ � _b_ _[′]_ � _b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_ ��� _≤_ ����Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ � _b_ _[′]_ � _b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ � _S_ _ℓ_ _j_ � _b_ _[′]_ ���



� _S_ ∞ ( _S_ ( _p_ _ℓ_ ) _−_ _S_ ( _p_ _j_ )) �



_b_ _[′]_



�



_b_ _[′]_



_b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_ ��� _≤_ ����Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ �



_b_ _[′]_ �



_b_ _[′]_ �



� _S_ _ℓ_ _j_ �
_b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_



_b_ _[′]_ ���



+ ���Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ � _b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_ ���



_≤_ 2 _[−]_ _[b]_ _[′]_ + ���Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ � _b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_ ���



Consider the function _g_ ( _**y**_ ) = ∏ _ℓ∈_ _B_, _j_ _∈_ _A_ _y_ _ℓ_ _j_ . For _**y**_ _∈_ [ 0, 1 + 1/ _d_ [2] ] _[|]_ _[A]_ _[|×|]_ _[B]_ _[|]_, _∥∇_ _g_ ( _**y**_ ) _∥_ 2 _≤_ Θ ( _d_ ) . As a
result, for all _**y**_, _**z**_ _∈_ [ 0, 1 + 1/ _d_ [2] ] _[|]_ _[A]_ _[|×|]_ _[B]_ _[|]_,



� 1/2



_|_ _g_ ( _**y**_ ) _−_ _g_ ( _**z**_ ) _| ≤_ Θ ( _d_ ) _·_


### ∑ ( y ℓ j − z ℓ j )

� _ℓ∈_ _B_, _j_ _∈_ _A_



In case the accuracy _b_ _[′]_ _≥_ Θ ( log _d_ ) then � _S_ _ℓ_ _j_ � _b_ _[′]_ _[ ≤]_ _[S]_ _[ℓ]_ _[j]_ [ +] [ 1/] _[d]_ [2] _[ ≤]_ [1] [ +] [ 1/] _[d]_ [2] [ and the above inequality]

applies. Thus,


### ∑

� _ℓ∈_ _B_, _j_ _∈_ _A_


### ∏

����� _ℓ∈_ _B_, _j_ _∈_ _A_



_B_ _[′]_ _[ −]_ _[S]_ _[ℓ]_ _[j]_ � [�] [1/2]



� _S_ _ℓ_ _j_ � _B_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_



_≤_ Θ ( _d_ )
�����



�� _S_ _ℓ_ _j_ �



Overall, ����Π _ℓ∈_ _B_, _j_ _∈_ _A_ � _S_ _ℓ_ _j_ �



_≤_ Θ ( _d_ [2] ) _·_ 2 _[−]_ _[b]_ _[′]_


2 _−_ _b_ _′_

_b_ _[′]_ � _b_ _[′]_ _[ −]_ [Π] _[ℓ][∈]_ _[B]_ [,] _[j]_ _[∈]_ _[A]_ _[S]_ _[ℓ]_ _[j]_ ��� _≤_ Θ ( _d_ ) _·_ 2 which concludes the proofof the cor


rected of [ _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]] _b_ [by selecting] _[ b]_ _[′]_ [ =] _[ b]_ [ +] [ Θ] [(] [log] _[ d]_ [)] [.]


78


_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]
In order to compute _∂x_ _ℓ_ where _ℓ_ _∈_ _B_ _**v**_ _**[c]**_ [(symmetrically for] _[ j]_ _[ ∈]_ _[A]_ _**[c]**_ _**v**_ [),] _[ M]_ _Q_ _[′]_ [ additionally]
computes the � _S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _ℓ_ [)] _[ −]_ _[S]_ [(] _[p]_ _j_ [))] � _b_ _[′]_ [ with accuracy] _[ b]_ _[′]_ [.] To simplify notation we denote with



computes the � _S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _ℓ_ [)] _[ −]_ _[S]_ [(] _[p]_ _j_ [))] � _b_ _[′]_ [ with accuracy] _[ b]_ _[′]_ [.] To simplify notation we denote with

_S_ ∞ _[′]_ [(] _[S]_ [(] _[p]_ _ℓ_ [)] _[ −]_ _[S]_ [(] _[p]_ _j_ [))] [ with] _[ S]_ _ℓ_ _[′]_ _j_ [and] _[ S]_ _[′]_ [(] _[p]_ _[i]_ [)] [ by] _[ S]_ _i_ _[′]_ [. Then] _[ M]_ _[Q]_ _[′]_ [ outputs,]
� _∂Q∂_ _**cv**_ _x_ [(] _i_ _**[x]**_ [)] � _b_ _[′]_ _[ ←]_ � _t_ _i_ _−_ 1 _s_ _i_ _·_ � _∂Q∂_ _**cv**_ _p_ [(] _i_ _**[x]**_ [)] � _b_ _[′]_ � _b_ _[′]_



�



1 _·_ _∂Q_ _**cv**_ [(] _**[x]**_ [)]
_b_ _[′]_ _[ ←]_ � _t_ _i_ _−_ _s_ _i_ � _∂p_ _i_



�



_b_ _[′]_



�



_b_ _[′]_



_b_ _[′]_



_∂Q_ _**cv**_ [(] _**[x]**_ [)]
where
� _∂p_ _i_ �



_b_ _[′]_ _[ ←]_



� _S_ _i_ _[′]_ � _b_ _[′]_ [ Π] _[m]_ _[∈]_ _[A]_ [/] _[j]_ [,] _[ℓ][∈]_ _[B]_ [ [] _[S]_ _[ℓ]_ _[m]_ []] _[b]_ _[′]_
_b_ _[′]_ _[ ·]_



�


### ∑

� _j_ _∈_ _A_



_S_ _[′]_
� _ij_ �



Observe that _t_ _i_ _−_ _s_ _i_ = [si][g][n] _N_ [(] _−_ _[t]_ _[i]_ _[−]_ 1 _[s]_ _[i]_ [)] and thus _t_ _i_ _−_ 1 _s_ _i_ _[·]_ � _∂Q∂_ _**[c]**_ _**v**_ _p_ [(] _i_ _**[x]**_ [)]



�



_b_ _[′]_ [ can be exactly computed. We next prove]



�



_b_ _[′]_ [ are correct.]



_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]
that these computations of � _∂x_ _i_



_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]

� _b_ _[′]_ [ and] � _∂p_ _i_



We first bound ���� _S_ _ij_ _[′]_ �



_b_ _[′]_ _[ ·]_ [ [] _[S]_ _i_ _[′]_ []] _b_ _[′]_ _[ ·]_ [ Π] _[m]_ _[∈]_ _[A]_ [/] _[{]_ _[j]_ _[}]_ [,] _[ℓ][∈]_ _[B]_ [[] _[S]_ _[ℓ]_ _[m]_ []] _b_ _[′]_ _[ −]_ _[S]_ _ij_ _[′]_ _[·]_ _[ S]_ _i_ _[′]_ _[·]_ [ Π] _[m]_ _[∈]_ _[A]_ [/] _[{]_ _[j]_ _[}]_ [,] _[ℓ][∈]_ _[B]_ _[S]_ _[ℓ]_ _[m]_ ���.



Consider the function _g_ ( _y_ 1, _y_ 2, _**y**_ ) = _y_ 1 _·_ _y_ 2 _·_ ∏ _m_ _∈_ _A_ / _{_ _j_ _}_, _ℓ∈_ _B_ _y_ _ℓ_ _m_ . As previously done, for _y_ 1, _y_ 2 _∈_

[ 0, 6 ] and _**y**_ _∈_ [ 0, 1 + 1/ _d_ [2] ] _[|]_ _[A]_ _[|×|]_ _[B]_ _[|−]_ [1] we have that, _∥∇_ _g_ ( _y_ 1, _y_ 2, _**y**_ ) _∥_ 2 _≤_ Θ ( _d_ ) . If _b_ _[′]_ _≤_ Θ ( log _d_ ) then
��� _S_ _ij_ _′_ ���, _S_ _i_ _′_ _[≤]_ [6 and] _[ S]_ _[ℓ]_ _[m]_ _[ ∈]_ [[] [0, 1] [ +] [ 1/] _[d]_ [2] []] [. As a result,]



_S_ _[′]_
���� _ij_ �



_b_ _[′]_ _[ ·]_ [ Π] _[m]_ _[∈]_ _[A]_ [/] _[{]_ _[j]_ _[}]_ [,] _[ℓ][∈]_ _[B]_ [ [] _[S]_ _[ℓ]_ _[m]_ []] _[b]_ _[′]_ _[ −]_ _[S]_ _ij_ _[′]_ _[·]_ _[ S]_ _i_ _[′]_ _[·]_ [ Π] _m_ _∈_ _A_ / _{_ _j_ _}_, _ℓ∈_ _B_ _[S]_ _[ℓ]_ _[m]_ ��� _≤_ Θ ( _d_ [2] ) _·_ 2 _[−]_ _[b]_ _[′]_ .



� _S_ _i_ _[′]_ �
_b_ _[′]_ _[ ·]_



_∂Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)]
We can now use the above inequality to bound ���� _∂p_ _i_



_**v**_ [(] _**[x]**_ [)]
_b_ _[′]_ _[ −]_ _[∂][Q]_ _∂_ _**[c]**_ _p_ _i_



�



. More precisely,
���



����



_∂Q_ _**cv**_ [(] _**[x]**_ [)]
� _∂p_ _i_



_**v**_ [(] _**[x]**_ [)]
_b_ _[′]_ _[ −]_ _[∂][Q]_ _∂_ _**[c]**_ _p_ _i_



�



����


### b [′] [ ·] � S i [′] � b [′] [ ·] ∏ [ S ℓ m ] b ′ − ∑ S ij [′] [·] [ S] i [′] [·] ∏ S ℓ m

_m_ _∈_ _A_ / _{_ _j_ _}_, _ℓ∈_ _B_ _j_ _∈_ _A_ _m_ _∈_ _A_ / _{_ _j_ _}_, _ℓ∈_ _B_



_≤_ 2 _[−]_ _[b]_ +


### ∑

_j_ _∈_ _A_
������



_S_ _[′]_
� _ij_ �



������



_≤_ Θ ( _d_ [3] ) _·_ 2 _[−]_ _[b]_ _[′]_



We finally get that


_∂Q_ _**cv**_ [(] _**[x]**_ [)]

����� _∂x_ _i_



_**v**_ [(] _**[x]**_ [)]
_b_ _[′]_ _[ −]_ _[∂][Q]_ _∂_ _**[c]**_ _x_ _i_



_∂Q_ _**cv**_ [(] _**[x]**_ [)]
� _∂p_ _i_



_**v**_ [(] _**[x]**_ [)]

� _b_ _[′]_ _[ −]_ _[∂][Q]_ _∂_ _**[c]**_ _p_ _i_



3 _−_ _b_ _′_
_≤_ Θ ( _Nd_ ) _·_ 2 .
����



�



_≤_ 2 _−_ _b_ _′_ + _N_
���� ����



Thus the analysis is completed by selecting _b_ _[′]_ = _b_ + Θ ( log _d_ ) + Θ ( log _N_ ) .


**Lemma D.4.** _There exist Turing Machines M_ _P_ _and M_ _P_ _′_ _that given_ _**x**_ _∈_ [ 0, 1 ] _[d]_ _and ε_ _>_ 0 _in binary form_
_compute_ [ P _**v**_ ( _**x**_ )] _b_ _and_ [ _∇_ P _**v**_ ( _**x**_ )] _b_ _respectively for all vertices_ _**v**_ _∈_ ([ _N_ ] _−_ 1 ) _[d]_ _with_ P _**v**_ ( _**x**_ ) _>_ 0 _, where_
_b_ = log ( 1/ _ε_ ) _. These vertices are most d_ + 1 _. Moreover both M_ _P_ _and M_ _P_ _′_ _run in polynomial time with_
_respect to b, d and the binary representation of_ _**x**_ _._



_Proof. M_ _P_ first runs _M_ _Q_ of Lemma D.3 to find the coefficients _Q_ _**[c]**_ _**v**_ [(] _**[x]**_ [)] _[ >]_ [ 0. We remind that these]
vertices are denoted with _R_ + ( _**x**_ ) and _|_ _R_ + ( _**x**_ ) _| ≤_ _d_ + 1. Then for each _**v**_ _∈_ _R_ + ( _**x**_ ), _M_ _P_ outputs as

[ P _**v**_ ( _**x**_ )] _b_ the fist _b_ bits of the quantity,

[ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_
� ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~ _b_ _[′]_ � _b_ _[′]_



�



_b_ _[′]_



_b_ _[′]_



79


where we determine the value of _b_ _[′]_ later in the proof but it is chosen to be polynomial in _b_ and
_d_ . We next present the proof that the above expression correctly computes [ P _**v**_ ( _**x**_ )] _b_ .

For accuracy _b_ _[′]_ _≥_ Θ ( _d_ [2] log _d_ ) we get that,

### ∑ [ Q [c] v [′] [(] [x] [)]] b [′] [ ≥] ∑ Q [c] v [′] [(] [x] [)] [ −] [Θ] [(] [d] [)] [ ·] [ 2] [−] [b] [′]

_**v**_ _[′]_ _∈_ _R_ + ( _**x**_ ) _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ )


=
### ∑ Q [c] v [′] [(] [x] [)] [ −] [Θ] [(] [d] [)] [ ·] [ 2] [−] [b] [′]

_**v**_ _[′]_ _∈_ _R_ _**c**_ ( _**x**_ )

_≥_ Θ 1/ _d_ ) _[d]_ [2] [�] _−_ Θ ( _d_ ) _·_ 2 _[−]_ _[b]_ _[′]_
�

_≥_ Θ � ( 1/ _d_ ) _[d]_ [2] [�]


Consider the function _g_ ( _**y**_ ) = _y_ _i_ / ( ∑ _[d]_ _j_ = [+] 1 [1] _[y]_ _[j]_ [)] [. Notice that for] _**[ y]**_ _[ ∈]_ [[] [0, 1] []] _[d]_ [+] [1] [ and][ ∑] _[d]_ _j_ = [+] 1 [1] _[y]_ _[j]_ _[ ≥]_ _[µ]_ [ then]

_∥∇_ _g_ ( _**y**_ ) _∥_ 2 _≤_ Θ ( _d_ [3/2] / _µ_ [2] ) . The latter implies that for _**y**_, _**z**_ _∈_ [ 0, 1 ] _[d]_ [+] [1] such that ∑ _[d]_ _j_ = [+] 1 [1] _[y]_ _[j]_ _[ ≥]_ _[µ]_ [ and]

that ∑ _[d]_ _j_ = [+] 1 [1] _[z]_ _[j]_ _[ ≥]_ _[µ]_ [, it holds that]



_y_ _i_ _−_ _z_ _i_
����� ∑ _[d]_ _j_ = [+] 1 [1] _[y]_ _[j]_ ∑ _[d]_ _j_ = [+] 1 [1] _[z]_ _[j]_



_d_ 3/2
_≤_ Θ
� _µ_ [2]
�����




_· ∥_ _**y**_ _−_ _**z**_ _∥_ 2 .
�



Since there are at most _d_ + 1 vertices _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ ) while both the term ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) � _Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] �



_b_ _[′]_ [ and]



the term ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _**[x]**_ [)] [ are greater than][ Θ] � ( 1/ _d_ ) _[d]_ [2] [�], we can apply the above inequality with

_µ_ = Θ ( 1/ _d_ ) _[d]_ [2] [�] and we get the following
�




[ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_
����� ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~



_−_ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]
_b_ _[′]_ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)]



�����



1/2
� [ _Q_ _**cv**_ _[′]_ [(] _[x]_ [)]] _b_ _[′]_ _[ −]_ _[Q]_ _**v**_ _**[c]**_ _[′]_ [(] _[x]_ [)] � 2
�



_≤_ Θ _d_ [2] _[d]_ [2] [+] [3/2] [�] _·_
�


### ∑

� _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ )



_≤_ Θ _d_ [2] _[d]_ [2] [+] [2] [�] _·_ 2 _[−]_ _[b]_ _[′]_
�



Overall, we have that


[ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_

������ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~



_−_ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]
_b_ _[′]_ ∑ _**v**_ _′_ _∈_ _R_ _**c**_ ( _**x**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)]



�����



_b_ _[′]_



�



_−_ [ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_
_b_ _[′]_ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~



_≤_




[ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_

������ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~



_b_ _[′]_



�



_b_ _[′]_



�����



�����



+




[ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]] _b_ _[′]_
����� ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) ~~�~~ _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)] ~~�~~



_−_ _Q_ _**[c]**_ _**v**_ [(] _[x]_ [)]
_b_ _[′]_ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) _Q_ _**[c]**_ _**v**_ _[′]_ [(] _[x]_ [)]



_≤_ Θ _d_ [2] _[d]_ [2] [+] [1] [�] 2 _[−]_ _[b]_ _[′]_
�


The proof is completed via selecting _b_ _[′]_ = _b_ + Θ ( _d_ [2] log _d_ ) .


80


In order to compute _[∂]_ [P] _∂_ _**[v]**_ _x_ [(] _i_ _**[x]**_ [)] the Turing machine _M_ _P_ _′_ computes all vertices _R_ + ( _**x**_ ) the coefficients

_∂Q∂_ _**[c]**_ _**v**_ _x_ [(] _i_ _**[x]**_ [)] with accuracy _b_ _[′]_ . Then for each _**v**_ _∈_ _R_ + ( _**x**_ ) the Turing Machine _M_ _P_ _′_ outputs,



_∂_ P _**v**_ ( _**x**_ )
� _∂x_ _i_



�



1 _·_ _∂_ P _**v**_ ( _**x**_ )
_b_ _[′]_ _[ ←]_ � _t_ _i_ _−_ _s_ _i_ � _∂p_ _i_



�



_b_ _[′]_



_b_ _[′]_



�



_b_ _[′]_



_∂Q_ _**v**_ _′_ ( _**x**_ )
_b_ _[′]_ _[ ·]_ [ ∑] _**[v]**_ _[′]_ _[∈]_ _[R]_ [+] [(] _**[x]**_ [)] [ [] _[Q]_ _**[v]**_ _[′]_ [(] _**[x]**_ [)]] _[b]_ _[′]_ _[ −]_ [[] _[Q]_ _**[v]**_ [(] _**[x]**_ [)]] _[b]_ _[′]_ _[ ·]_ [ ∑] _**[v]**_ _[′]_ _[∈]_ _[R]_ [+] [(] _**[x]**_ [)] � _∂p_ _i_



_i_ _b_ _[′]_


2
~~�~~ ∑ _**v**_ _′_ _∈_ _R_ + ( _**x**_ ) [ _Q_ _**v**_ _[′]_ ( _**x**_ )] _b_ _′_ ~~�~~



�



�



_∂_ P _**v**_ ( _**x**_ )
where
� _∂p_ _i_



�



_b_ _[′]_ _[ ←]_



_∂p_ _i_















_∂Q_ _**v**_ ( _**x**_ )
� _∂p_ _i_



Similarly as above and as in Lemma D.3 we can prove that if _b_ _[′]_ _≥_ _b_ + Θ ( _d_ [2] log _d_ ) + Θ ( log _N_ ),
_∂_ P _**v**_ ( _**x**_ ) _−_ _b_
���� _∂p_ _i_ � _b_ _[′]_ _[ −]_ _[∂]_ [P] _∂_ _**[v]**_ _p_ [(] _i_ _**[x]**_ [)] ��� _≤_ 2 .



_b_ _[′]_ _[ −]_ _[∂]_ [P] _∂_ _**[v]**_ _p_ [(] _i_ _**[x]**_ [)]



_≤_ 2 _−_ _b_ .
���



_∂p_ _i_



�



_∂p_ _i_



_Proof of Theorem 7.6._ Let _R_ ( _**x**_ ) be the cell at which _**x**_ lies. The Turing Machine _M_ _f_ _C_ _l_ initially calculates the vertices _**v**_ _∈_ _R_ _**c**_ ( _**x**_ ) with coefficient P _**v**_ ( _**x**_ ) _>_ 0. We remind that this set is denoted by
_R_ + ( _**x**_ ) and _|_ _R_ + ( _**x**_ ) _| ≤_ _d_ + 1. Then _M_ _f_ _C_ _l_ outputs the first _b_ bits of the following quantity,



� _f_ _C_ _l_ ( _**x**_, _**y**_ ) �



_d_
### ∑ [ α ( x, j )] b ′ · ( x j − y j ) where [ α ( x, j )] b ′ = ∑ C l ( v, j ) · [ P v ( x )] b ′
_b_ _[′]_ [ =] _j_ = 1 _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ )



we next prove that the above computation is correct.



�����



���� _f_ _C_ _l_ ( _**x**_, _**y**_ ) �



=
_b_ _[′]_ _[ −]_ _[f]_ _[C]_ _[l]_ [(] _**[x]**_ [,] _**[y]**_ [)] ���



�����



_d_ _d_
### ∑ [ α ( x, j )] b ′ · ( x j − y j ) − ∑ α ( x, j ) · ( x j − y j )

_j_ = 1 _j_ = 1



_d_
### ≤ ∑ | [ α ( x, j )] − α ( x, j ) |

_j_ = 1



_d_

=
### ∑

_j_ = 1


### ∑ C l ( v, j ) · [ P v ( x )] b ′ − ∑ C l ( v, j ) · P v ( x )

����� _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ ) _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ ) �����



_d_
### ≤ ∑ ∑ | [ P v ( x )] b ′ − P v ( x ) |

_j_ = 1 _**v**_ _[′]_ _∈_ _R_ + ( _**x**_ )

_≤_ _d_ _·_ ( _d_ + 1 ) _·_ 2 _[−]_ _[b]_ _[′]_



Setting _b_ _[′]_ = _b_ + Θ ( log _d_ ) we get the desired result. Similarly for _∂_ _f_ _C_ _∂_ _l_ _x_ ( _**x**_, _**y**_ )



_∂y_ _i_ .



_l_ ( _**x**_, _**y**_ ) and _∂_ _f_ _C_ _l_ ( _**x**_, _**y**_ )

_∂x_ _i_ _∂y_ _i_


#### **E Convergence of PGD to Approximate Local Minimum**

In this section we present for completeness the folklore result that the Projected Gradient Descent
with convex projection set converges fast to a first order stationary point. Using the same ideas
that we presented in Section 5 this result implies that Projected Gradient Descent solves the
LocalMin problem in time poly ( 1/ _ε_, _L_, _G_, _d_ ) when ( _ε_, _δ_ ) in the input are in the local regime.
Also observe that although the following proof assumes access to the exact value of the gradient
_∇_ _f_ it is very simple to adapt the proof to the case where we only have access to _∇_ _f_ with accuracy
_ε_ [3] . We leave this as an exercise to the reader.


81


**Theorem E.1.** _Let f_ : _K_ _→_ **R** _be an L-smooth function and K_ _⊆_ **R** _[d]_ _be a convex set. The projected_

_[f]_ [(] _**[x]**_ [0] [)] _[−]_ _[f]_ [(] _**[x]**_ _[⋆]_ [))]
_gradient descent algorithm started at_ _**x**_ 0 _, with step size η, after at most T_ _≥_ [2] _[L]_ [(] _ε_ [2] _steps outputs_

_a point_ ˆ _**x**_ _such that_

ˆ ˆ ˆ
_∥_ _**x**_ _−_ Π _K_ ( _**x**_ _−_ _η_ _∇_ _f_ ( _**x**_ )) _∥_ 2 _≤_ _η_ _·_ _ε_


_where η_ = 1/ _L and_ _**x**_ _[⋆]_ _is a global minimum of f._


_Proof._ If we run the Projected Gradient Descent algorithm on _f_ then we have


_**x**_ _t_ + 1 _←_ Π _K_ ( _**x**_ _t_ _−_ _η_ _∇_ _f_ ( _**x**_ _t_ ))


then due to the _L_ -smoothness of _f_ we have that


_f_ ( _**x**_ _t_ + 1 ) _≤_ _f_ ( _**x**_ _t_ ) + _⟨∇_ _f_ ( _**x**_ _t_ ), _**x**_ _t_ + 1 _−_ _**x**_ _t_ _⟩_ + _[L]_ 2 [.]

2 _[∥]_ _**[x]**_ _[t]_ [+] [1] _[ −]_ _**[x]**_ _[t]_ _[∥]_ [2]


We can now apply Theorem 1.5.5 (b) of [FP07] to get that


_⟨_ _η_ _· ∇_ _f_ ( _**x**_ _t_ ), _**x**_ _t_ + 1 _−_ _**x**_ _t_ _⟩≤−∥_ _**x**_ _t_ + 1 _−_ _**x**_ _t_ _∥_ [2] 2 [=] _[⇒]_


_⟨∇_ _f_ ( _**x**_ _t_ ), _**x**_ _t_ + 1 _−_ _**x**_ _t_ _⟩≤−_ [1] 2

_[· ∥]_ _**[x]**_ _[t]_ [+] [1] _[ −]_ _**[x]**_ _[t]_ _[∥]_ [2]
_η_


If we combine these then we have that



1
_f_ ( _**x**_ _t_ + 1 ) _≤_ _f_ ( _**x**_ _t_ ) _−_
� _η_



1

_η_ _[−]_ _[L]_ 2



2



_∥_ _**x**_ _t_ + 1 _−_ _**x**_ _t_ _∥_ [2] 2 [.]
�



So if we pick _η_ = 1/ _L_ then we get


_f_ ( _**x**_ _t_ + 1 ) _≤_ _f_ ( _**x**_ _t_ ) _−_ _[L]_ 2 [.]

2 _[∥]_ _**[x]**_ _[t]_ [+] [1] _[ −]_ _**[x]**_ _[t]_ _[∥]_ [2]


If sum all the above inequalities and divide by _T_ then we get



1

_T_



_T_ _−_ 1 2
### ∑ ∥ x t + 1 − x t ∥ [2] 2 [≤]
_t_ = 0 _T_ _·_ _L_ [(] _[ f]_ [ (] _**[x]**_ [0] [)] _[ −]_ _[f]_ [ (] _**[x]**_ _[T]_ [))]



which implies that



min
0 _≤_ _t_ _≤_ _T_ _−_ 1 _[∥]_ _**[x]**_ _[t]_ [+] [1] _[ −]_ _**[x]**_ _[t]_ _[∥]_ [2] _[ ≤]_



�



2
_T_ _·_ _L_ [(] _[ f]_ [ (] _**[x]**_ [0] [)] _[ −]_ _[f]_ [ (] _**[x]**_ _[T]_ [))]




_[f]_ [(] _**[x]**_ [0] [)] _[−]_ _[f]_ [(] _**[x]**_ _[⋆]_ [))]
Therefore for _T_ _≥_ [2] _[L]_ [(] _ε_ [2] we have that


min
0 _≤_ _t_ _≤_ _T_ _−_ 1 _[∥]_ _**[x]**_ _[t]_ [+] [1] _[ −]_ _**[x]**_ _[t]_ _[∥]_ [2] _[ ≤]_ _[η]_ _[ ·]_ _[ ε]_ [ =] _[ ε]_ [/] _[L]_ [.]


82


