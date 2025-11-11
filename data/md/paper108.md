1


# Non-convex Min-Max Optimization: Applications, Challenges, and Recent Theoretical Advances

Meisam Razaviyayn, Tianjian Huang, Songtao Lu, Maher Nouiehed, Maziar Sanjabi, Mingyi Hong



_**Abstract**_ **—The** **min-max** **optimization** **problem,** **also**
**known as the saddle point problem, is a classical opti-**
**mization problem which is also studied in the context of**
**zero-sum games. Given a class of objective functions, the**
**goal is to find a value for the argument which leads to a**
**small objective value even for the worst-case function in the**
**given class. Min-max optimization problems have recently**
**become very popular in a wide range of signal and data**
**processing applications such as fair beamforming, training**
**generative adversarial networks (GANs), and robust ma-**
**chine learning (ML), to just name a few. The overarching**
**goal of this article is to provide a survey of recent advances**
**for an important subclass of min-max problem in which**
**the minimization and maximization problems can be non-**
**convex and/or non-concave. In particular, we first present**
**a number of applications to showcase the importance of**
**such min-max problems; then, we discuss key theoretical**
**challenges, and provide a selective review of some exciting**
**recent theoretical and algorithmic advances in tackling**
**non-convex min-max problems. Finally, we point out open**
**questions and future research directions.** [1]


I. I NTRODUCTION


Recently, the class of non-convex min-max optimization problems has attracted significant attention across
signal processing, optimization, and ML communities.
The overarching goal of this paper is to provide a
selective survey of the applications of such a new class of
problem, discuss theoretical and algorithmic challenges,
and present some recent advances in various directions.
To begin our discussion, let us consider the following
generic problem formulation:


min _f_ ( **x** _,_ **y** ) (Min-Max)
**x** [max] **y**

s.t. **x** _∈X ⊆_ R _[d]_ _,_ **y** _∈Y ⊆_ R _[b]_ _,_


where _f_ ( _·, ·_ ) : R _[d]_ _×_ R _[b]_ _→_ R is differentiable with
Lipschitz continuous gradient in ( **x** _,_ **y** ), possibly nonconvex in **x** and possibly non-concave in **y** ; **x** _∈_ R _[d]_

and **y** _∈_ R _[b]_ are the optimization variables; _X_ and _Y_ are
the feasible sets, which are assumed to be closed and
convex. Notice that, while we present this article around


1 The manuscript is accepted in IEEE Signal Processing Magazine.



the above min-max formulation, extending the ideas and
discussions to max-min problems is straight forward.


When problem (Min-Max) is convex in **x** and concave in **y**, the corresponding variational inequality (VI)
becomes monotone, and a wide range of algorithms have
been proposed for solving this problem; see, e.g., [1]–[4],
and the references therein. However, as we will discuss
in this article, solving min-max problems is challenging in non-convex setting. Such non-convex min-max
optimization problems appear in different applications
in signal processing (e.g., robust transceiver design,fair
resource allocation [5], communication in the presence
of jammers [6]), distributed signal processing [7], [8]),
and ML (e.g., robust training of neural networks [9],
training generative adversarial networks (GANs) [10],

[11], and fair inference [12], [13]. More generally, any
design problem in the presence of model uncertainty
or adversary can be modeled as an optimization of
the form (Min-Max). In this setup, **x** is the design
parameter that should be optimized, while **y** is the
uncertainty/adversary parameter which is not accurately
measured, or may be adjusted by an adversary. In such
scenarios, the goal in formulation (Min-Max) is to find a
solution **x** = ¯ **x** that has a robust performance against all
uncertainty/adversary values of **y** _∈Y_ . Such a robustness
requirement has long been deemed important in signal
processing community, and it has recently played a
crucial role in designing modern ML tools.


Despite the rising interests in nonconvex min-max
problems, seldom have they been rigorously analyzed
in either classical optimization or signal processing literature. In this article, we first present a number of applications to showcase the importance of such min-max
problems, then we discuss key theoretical challenges,
and provide a selective review of some recent theoretical
and algorithmic advances in tackling the class of nonconvex min-max problems. Finally, we will point out
open questions and future research directions.


II. A PPLICATIONS OF NON  - CONVEX MIN  - MAX

PROBLEMS


To appreciate the importance of problem (Min-Max),
let us first present a number of key applications of nonconvex min-max optimization problems.


_1) Generative_ _adversarial_ _networks_ _(GANs):_
GANs [10] have recently gained tremendous popularity
due to their unique ability to learn complex distributions
and generate realistic samples, e.g., high resolution fake
images. In the absence of labels, GANs aim at finding
a mapping from a known distribution, e.g. Gaussian, to
an unknown data distribution, which is only represented
by empirical samples [10].
GANs consist of two neural networks: the generator
and the discriminator. The goal of the generator is to
generate _fake_ samples which look like _real_ samples
in the distribution of interest. This process is done by
taking i.i.d. samples from a known distribution such as
Gaussian and transform it to samples similar to real
ones via trained neural network. On the other hand, the
discriminator’s objective is to correctly classify the fake
samples generated by the generator and the real samples
drawn from the distribution of interest. The two-player
game between the generator and the discriminator can
be modeled as a min-max optimization problem [10]:


min max _V_ ( **w** _g_ _,_ **w** _d_ ) _,_ (1)
**w** _g_ **w** _d_


where **w** _g_ is the generator’s parameter; **w** _d_ is the
discriminator’s parameter; and _V_ ( _·, ·_ ) shows the cost
function of the generator (which is equal to the negative of the discriminator’s cost function). This minmax objective can be also justified as minimizing some
distance between the distribution of real samples and
the distribution of generated samples. In this interpretation, the distance between the two distributions is
computed by solving a maximization (dual) problem;
and the goal is to minimize the distance between the
distribution of generated samples and the distribution of
real samples. Various distance measures have been used
for training GANs such as Jensen-Shannon divergence

[10], _f_ -divergence, and Wasserstein distance [11]. All
these distances lead to non-convex non-concave min-max

formulations for training GANs.


_2) Fair ML:_ The past few years have witnessed
several reported instances of ML algorithms suffering
from systematic discrimination against individuals of
certain protected groups; see, e.g., [13]–[15], and the
references therein. Such instances stimulate strong interest in the field of fairness in ML which in addition to
the typical goal of having an accurate learning model,
brings fairness to the learning task. Imposing fairness on
ML models can be done through three main approaches:



2


preprocessing approaches, in-processing approaches, and
postprocessing approaches.
To understand these three approaches, consider a ML
task over a given random variables **X** _∈_ R _[d]_ representing
the non-sensitive data attributes and **S** _∈_ R _[k]_ representing
the sensitive attributes (such as age, gender, ethnicity,
etc.). Pre-processing approaches tend to hinder discrimination by masking the training data before passing it
to the decision making process. Among these methods, recent works [14], [15] have used an adversarial
approach which seeks to learn a data representation
**Z** = _ζ_ ( **X** _,_ **S** ) capable of minimizing the loss over the
classifier _g_ ( **Z** ), and protecting the sensitive attributes
**S** from an adversary _h_ ( **Z** ) that tries to reconstruct **S**
from **Z** . This requires solving the following min-max
optimization problem:


min E **X** _,_ **S** _{L_ ( _ζ, g, h_ ) _}._
_ζ,g_ [max] _h_


Realizing the functions as neural networks, this formulation leads to non-convex min-max optimization problem.
Contrary to pre-processing methods, in-processing approaches impose fairness during training procedure. For
example, they impose fairness by adding a regularization
term that penalizes statistical dependence between the
learning model output and the sensitive attributes **S** . Let
_g_ _**θ**_ ( **X** _,_ **S** ) be a certain output of the learning model. One
can balance the learning accuracy and fairness by solving
the following optimization problem:


min E _{L_ ( _**θ**_ _,_ **X** ) _}_ + _λ ρ_ � _g_ _**θ**_ ( **X** _,_ **S** ) _,_ **S** � _,_ (2)
_**θ**_


where _ρ_ ( _·, ·_ ) is a statistical independence measure and
_L_ ( _·, ·_ ) denotes the training loss function. For example,
in the classification task in which **X** contains both the
input feature and the target variable, the function _L_ ( _·, ·_ )
measures the classification error of the trained classifier. Here, the parameter _λ_ is a positive scalar balancing fairness and accuracy of the output model. When
_λ →∞_, this optimization problem focuses more on
making _g_ _**θ**_ ( **X** _,_ **S** ) and **S** independent, resulting in a fair
inference. However, when _λ_ = 0, no fairness is imposed
and the focus is to maximize the accuracy of the model

output.
Various statistical dependence measures have been
proposed for use in this formulation. For example, [13]
proposed using R´enyi correlation to impose fairness.
The R´enyi correlation between two random variables **A**
and **B** is defined as _ρ_ ( **A** _,_ **B** ) ≜ sup _k,ℓ_ _ρ_ _p_ ( _k_ ( **A** ) _, ℓ_ ( **B** ))
where _ρ_ _p_ is the Pearson correlation coefficient and the
supremum is over the set of measurable functions _k_ ( _·_ )
and _ℓ_ ( _·_ ). Plugging the definition of R´enyi correlation
in (2) leads to a natural min-max formulation, which
is the focus of this article.


_3) Adversarial ML:_ The formulation (Min-Max) is
also instrumental to model the dynamic process of _ad-_
_versarial learning_, where the model training process involves some kind of “adversary”. Depending on whether
the goal is to break the ML model or to make it more robust, one can formulate different min-max optimization
problems, as we briefly discuss in the following sections.

**Adversarial attacks.** First, let us take the viewpoint of
the adversary, who would like to break a ML model so
that it is more likely to produce wrong predictions. In
this scenario, the adversary tries to increase the error
of a well-trained ML model; therefore its behavior is
modeled as the _outer_ optimization problem, aiming to
reduce the performance of the trained model. On the
other hand, the training process is modeled as the _inner_
optimization problem aiming to minimize the training

error.

To be more specific, take the poisoning attack [16]
as an example. Let _D_ := _{_ **u** _i_ _, t_ _i_ _}_ _[N]_ _i_ =1 [denote the training]
dataset, where **u** _i_ and _t_ _i_ represent the features and target
labels of sample _i_ respectively. Each data sample **u** _i_ can
be corrupted by a perturbation vector _**δ**_ _i_ to generate a
“poisoned” sample **u** _i_ + _**δ**_ _i_ . Let _**δ**_ := ( _**δ**_ 1 _, . . .,_ _**δ**_ _N_ ) be the
collection of all poisoning attacks. Then, the poisoning
attack problem is formulated as


_̸_



3


Note that compared with (3), the roles of minimization
and maximization have been switched. Clearly, this
optimization problem is of the form (Min-Max).


_4) Distributed processing:_ Some constrained nonconvex optimization problems could also be formulated
as a min-max saddle point problem by leveraging the
primal dual approach or the method of Lagrange multipliers. An example of that appears in distributed data
processing over networks. Consider a network of _N_
nodes in a graph _G_ = _{V, E}_ with _|V|_ = _N_ vertices.
The nodes can communicate with their neighbors, and
their goal is to jointly solve the optimization problem:


_̸_



min

_z_


_̸_



_N_
� _g_ _i_ ( _z_ ) _,_


_i_ =1


_̸_



where each _g_ _i_ ( _·_ ) is a smooth function only known by
node _i_ . Further, for simplicity of presentation, assume
that _z ∈_ R.

Such a distributed optimization setting has been
widely studied in the optimization and signal processing
communities over the past few decades. Let _x_ _i_ be node
_i_ ’s local copy of _z_ . A standard first step in distributed
optimization is to rewrite the above problem as:


_̸_



_N_
� _g_ _i_ ( _x_ _i_ ) s.t. **Ax** = 0 _,_ (4)


_i_ =1


_̸_



where **A** _∈_ R _[|E|×][N]_ is the incidence matrix for graph
_G_ and **x** = ( _x_ 1 _, . . ., x_ _N_ ) is the concatenation of all
copies of the decision variable. The linear constraint in
(4) enforces _x_ _i_ = _x_ _j_ _,_ if _i, j_ are neighbors. Problem (4)
can be rewritten as [2]


_̸_



min
_x∈_ R _[N]_ _[ g]_ [(] **[x]** [) :=]


_̸_



max
_**δ**_ : _∥_ _**δ**_ _i_ _∥≤ε_ [min] **w**


_̸_



_N_
� _ℓ_ ( _p_ ( **u** _i_ + _**δ**_ _i_ ; **w** ) _, t_ _i_ ) (3)


_i_ =1


_̸_



where **w** is the weight of the neural network; _p_ ( _·_ ) is the
predicted output of the neural network; and _ℓ_ ( _·_ ) is the
loss function. The constraint _∥_ _**δ**_ _i_ _∥≤_ _ε_ indicates that the
poisoned samples should not be too different from the
original ones, so that the attack is not easily detectable.
Note that the “max-min” problem (3) can be written
equivalently in the form of (Min-Max) by adding a minus
sign to the objective.

**Defense against adversarial attacks.** It has been widely
observed that ML models, especially neural networks,
are highly vulnerable to adversarial attacks, including the
poisoning attack discussed in the previous subseciton,
or other popular attackes such as Fast Gradient Sign
Method (FGSM) attack [17] and Projected Gradient
Descent (PGD) attack [18]. These adversarial attacks
show that a small perturbation in the data input can
significantly change the output of a neural network and
deceive different neural network architectures in a wide

range of applications. To make ML models robust against
adversarial attacks, one popular approach is to solve
the following robust training problem [9] (using similar _̸_
notations as in (3)):



where **y** is the Lagrangian multiplier. Clearly, (5) is in
the form of (Min-Max), where the coupling between
**x** and **y** is _linear_ . A number of algorithms have been
developed for it; see a recent survey [20].


_5) Max-Min fair transceiver design:_ Consider the
problem of resource allocation in a wireless communication system, where _N_ transmitter-receiver pairs are communicating. The goal is to maximize the minimum rate
among all users. To be specific, consider a setting with
_K_ parallel channels. User _i_ transmits with power **p** _i_ :=

[ _p_ [1] _i_ [;] _[ · · ·][, p]_ _[K]_ _i_ []][, and its rate is given by:] _[ r]_ _[i]_ [(] **[p]** [1] _[, . . .,]_ **[ p]** _[N]_ [) =]
� _Kk_ =1 [log] 1+ _a_ _[k]_ _ii_ _[p]_ _[k]_ _i_ (assuming Gaus� _σ_ _i_ [2] [+] ~~[ �]~~ _[N]_ _j_ =1 _,j_ = _̸_ _i_ _[a]_ _[k]_ _ji_ _[p]_ _[k]_ _j_ �

sian signaling), which is a non-convex function in **p** .
Here _a_ _[k]_ _ji_ [denotes the channel gain between transmitter] _[ j]_


2 It can be shown that finding a stationary solution of (5) is equivalent
to finding a stationary solution for (4); see [19].



max
_y∈_ R _[|E|]_ [ min] _x∈_ R _[N]_


_̸_



_N_
� _g_ _i_ ( _x_ _i_ ) + **y** _[T]_ **Ax** (5)


_i_ =1


_̸_



_̸_


min

**w**



_̸_


_N_


max

� _**δ**_ : _∥_ _**δ**_ _i_ _∥≤ε_ _[ℓ]_ [(] _[p]_ [(] **[u]** _[i]_ [ +] _**[ δ]**_ _[i]_ [;] **[ w]** [)] _[, t]_ _[i]_ [)] _[.]_

_i_ =1


and receiver _i_ on the _k_ -th channel, and _σ_ _i_ [2] [is the noise]
power of user _i_ . Let **p** := [ **p** 1 ; _· · ·_ ; **p** _N_ ], then the maxmin fair power control problem is given by [5]


max _{r_ _i_ ( **p** ) _}_ _[N]_ _i_ =1 _[,]_ (6)
**p** _∈P_ [min] _i_


where _P_ denotes the set of feasible power allocations.
While the inside minimization is over a discrete vari
able _i_, we can reformulate it as a minimization over
continuous variables using transformation:


_̸_



max **p** _∈P_ **y** [min] _∈_ ∆


_̸_



_N_
� _r_ _i_ ( **p** 1 _, · · ·,_ **p** _N_ ) _× y_ _i_ _,_ (7)


_i_ =1


_̸_



where ∆ ≜ _{_ **y** _|_ **y** _≥_ **0** ; [�] _[N]_ _i_ =1 _[y]_ _[i]_ [ = 1] _[}]_ [ is the probability]
simplex. Notice that the inside minimization problem
is linear in **y** . Hence, there always _exists_ a solution at
one of the extreme points of the simplex ∆. Thus, the
formulation (7) is equivalent to the formulation (6). By
multiplying the objective by the negative sign, we can
transform the above “max-min” formulation to “min
max” form consistent with (Min-Max), i.e.,


_̸_



min **p** _∈P_ [max] **y** _∈_ ∆


_̸_



_N_
� _−r_ _i_ ( **p** 1 _, · · ·,_ **p** _N_ ) _× y_ _i_


_i_ =1


_̸_



_6) Communication in the presence of jammers:_
Consider a variation of the above problem, where _M_
jammers participate in an _N_ -user _K_ -channel interference channel transmission. The jammers’ objective is to
reduce the sum rate of the system by transmitting noises,
while the goal for the regular users is to transmit as much
information as possible. We use _p_ _[k]_ _i_ [(resp.] _[ q]_ _j_ _[k]_ [) to denote]
the _i_ -th regular user’s (resp. _j_ -th jammer’s) power on
the _k_ -th channel. The corresponding sum-rate max-min
problem can be formulated as:


_̸_



_̸_ �



1 + _a_ _[k]_ _ii_ _[p]_ _[k]_ _i_

� _σ_ _i_ [2] [+] ~~[ �]~~ _[N]_ _ℓ_ =1 _,j_ = _̸_ _i_ _[a]_ _[k]_ _ℓi_ _[p]_ _[k]_ _ℓ_ [+] _[ b]_ _[k]_ _ji_ _[q]_ _j_ _[k]_



max min

**p** **q**

_̸_



� log

_k,i,j_ _̸_



_,_


_̸_



4


non-convexity of the objective (which prevents us from
finding global optima), but also is due to aiming for
finding a min-max solution. To see the challenges of
solving non-convex min-max problems, let us compare
and contrast the optimization problem (Min-Max) with
the regular smooth non-convex optimization problem:


min _h_ ( **z** ) _._ (9)
**z** _∈Z_


where the gradient of function _h_ is Lipschitz continuous.
While solving general non-convex optimization problem (9) to global optimality is hard, one can apply simple
iterative algorithms such as projected gradient descent
(PGD) to (9) by running the iterates


**z** _[r]_ [+1] = _P_ _Z_ ( **z** _[r]_ _−_ _α∇h_ ( **z** _[r]_ )) _,_


where _r_ is the iteration count; _P_ _Z_ is projection to the set
_Z_ ; and _α_ is the step-size. Algorithms like PGD enjoy
two properties:

i) The quality of the iterates improve over time, i.e.,
_h_ ( **z** _[r]_ [+1] ) _≤_ _h_ ( **z** _[r]_ ), where _r_ is the iteration number.
ii) These algorithms are guaranteed to converge to
(first-order) stationary points with global iteration
complexity guarantees [21] under a mild set of
assumptions.
The above two properties give enough confidence to
researchers to apply projected gradient descent to many
non-convex problems of the form (9) and expect to
find “reasonably good” solutions in practice. In contrast,
_there is no widely accepted optimization tool_ for solving
general non-convex min-max problem (Min-Max). A
simple extension of the PGD to the min-max setting
is the gradient-descent ascent algorithm (GDA). This
popular algorithm simply alternates between a gradient
descent step on **x** and a gradient ascent step on **y** through
the update rules


**x** _[r]_ [+1] = _P_ _X_ ( **x** _[r]_ _−_ _α∇_ _x_ _f_ ( **x** _[r]_ _,_ **y** _[r]_ )) _,_

_̸_ **y** _[r]_ [+1] = _P_ _Y_ ( **y** _[r]_ + _α∇_ _y_ _f_ ( **x** _[r]_ _,_ **y** _[r]_ )) _,_


where _P_ _X_ and _P_ _Y_ are the projections to the sets _X_
and _Y_, respectively. The update rule of **x** and **y** can
be done alternatively as well [i.e., **y** _[r]_ [+1] = _P_ _Y_ ( **y** _[r]_ +
_α∇_ _y_ _f_ ( **x** _[r]_ [+1] _,_ **y** _[r]_ ))]. Despite popularity of this algorithm,
it fails in many practical instances. Moreover, it is not
hard to construct very simple examples for which this
algorithm fails to converge to any meaningful point; see
Fig. 1 for an illustration.


IV. R ECENT DEVELOPMENTS FOR SOLVING

NON    - CONVEX MIN    - MAX PROBLEMS

To understand some of the recently developed algorithms for solving non-convex min-max problems, we
first need to review and discuss stationarity and optimality conditions for such problems. Then, we highlight
some of the ideas leading to algorithmic developments.



_̸_


s.t. **p** _∈P,_ **q** _∈Q,_ (8)


where _a_ _[k]_ _ℓi_ [and] _[ b]_ _[k]_ _ji_ [represent the] _[ k]_ [-th channels between]
the regular user pairs ( _ℓ, i_ ) and regular and jammer
pair ( _i, j_ ), respectively. Here _P_ and _Q_ denote the set
of feasible power allocation constraints for the users and
the jammers. Many other related formulations have been
considered, mostly from the game theory perspective [6].
Similar to the previous example, by multiplying the
objective by a negative sign, we obtain an optimization
problem of the form (Min-Max).


III. C HALLENGES


Solving min-max problems even up to simple notions
of stationarity could be extremely challenging in the
non-convex setting. This is not only because of the


_̸_



_̸_



_̸_



_̸_

Fig. 1: GDA trajectory for the function _f_ ( _x, y_ ) = _xy_ . The iterates
of GDA diverge even in this simple scenario. The GDA algorithm
starts from the red point and moves away from the origin (which is
the optimal solution).


_A. Optimality Conditions_


Due to the non-convex nature of problem (Min-Max),
finding the global solution is NP-hard in general [22].
Hence, the developed algorithms in the literature aimed
at finding _“stationary solutions”_ to this optimization
problem. One approach for defining such stationarity
concepts is to look at problem (Min-Max) as a game.
In particular, one may ignore the order of minimization and maximization in problem (Min-Max) and view
it as a zero-sum game between two players. In this
game, one player is interested in solving the problem:
min **x** _∈X_ _f_ ( **x** _,_ **y** ), while the other player is interested
in solving: max **y** _∈Y_ _f_ ( **x** _,_ **y** ) _._ Since the objective functions of both players are non-convex in general, finding a global Nash Equilibrium is not computationally
tractable [22]. Hence, we may settle for finding a
point satisfying first-order optimality conditions for each
player’s objective function, i.e., finding a point (¯ **x** _,_ ¯ **y** )
satisfying


_⟨∇_ **x** _f_ (¯ **x** _,_ ¯ **y** ) _,_ **x** _−_ **x** ¯ _⟩≥_ 0 _, ∀_ **x** _∈X_


and (Game-Stationary)


_⟨∇_ **y** _f_ (¯ **x** _,_ ¯ **y** ) _,_ **y** _−_ **y** ¯ _⟩≤_ 0 _, ∀_ **y** _∈Y._


This condition, which is also referred to as “quasiNash Equilibrium” condition in [23] or “First-order Nash
Equilibrium” condition in [24], is in fact the solution of
the VI corresponding to the min-max game. Moreover,
one can use fixed point theorems and show existence of
a point satisfying (Game-Stationary) condition under a
mild set of assumptions; see, e.g., [25, Proposition 2]. In
addition to existence, it is always easy to _check_ whether
a given point satisfies the condition (Game-Stationary).
The ease of checkability and the game theoretic interpretation of the above (Game-Stationary) condition



5


have attracted many researchers to focus on developing
algorithms for finding a point satisfying this notion; see,
e.g., [23]–[25], and the references therein.
A potential drawback of the above stationarity notation is its ignorance to the order of the minimization
and maximization players. Notice that the Sion’s minmax theorem shows that when _f_ ( **x** _,_ **y** ) is convex in **x**
and concave in **y** the minimization and maximization
can interchange in (Min-Max), under the mild additional
assumption that either _X_ or _Y_ is compact. However, for
the general non-convex problems, the minimization and
maximization cannot interchange, i.e.,


min = _̸_ max
**x** _∈X_ [max] **y** _∈Y_ _[f]_ [(] **[x]** _[,]_ **[ y]** [)] **y** _∈Y_ **x** [min] _∈X_ _[f]_ [(] **[x]** _[,]_ **[ y]** [)] _[.]_


Moreover, the two problems may have different solutions. Therefore, the (Game-Stationary) notion might not
be practical in applications in which the minimization
and maximization order is important, such as defense
against adversarial attacks to neural networks (as discussed in the previous section). To modify the definition
and considering the minimization and maximization order, one can define the stationarity notion by rewriting
the optimization problem (Min-Max) as


min _g_ ( **x** ) (10)
**x** _∈X_


where _g_ ( **x** ) ≜ max **y** _∈Y_ _f_ ( **x** _,_ **y** ) when **x** _∈X_ and _g_ ( **x** ) =
+ _∞_ when **x** _/∈X_ . Using this viewpoint, we can define
a point ¯ **x** as a stationary point of (Min-Max) if ¯ **x** is a
first-order stationary point of the non-convex non-smooth
optimization (10). In other words,


**0** _∈_ _∂g_ (¯ **x** ) _,_ (Optimization-Stationary)


where _∂g_ (¯ **x** ) is Fr´echet sub-differential of a function _g_ ( _·_ )
at the point ¯ **x**, i.e., _∂g_ (¯ **x** ) ≜ _{_ **v** _|_ lim inf **x** _′_ _�→_ **x** � _g_ ( **x** _[′]_ ) _−_
_g_ ( **x** ) _−⟨_ **v** _,_ **x** _[′]_ _−_ **x** _⟩_ � _/_ � _∥_ **x** _[′]_ _−_ **x** _∥_ � _≥_ 0 _}_ . It is
again not hard to show existence of a point satisfying (Optimization-Stationary) under a mild set of assumptions such as compactness of the feasible set and
continuity of the function _f_ ( _·, ·_ ). This is because of
the fact that any continuous function on a compact set
attains its minimum. Thus, at least the global minimum
of the optimization problem (10) satisfies the optimality
condition (Optimization-Stationary). This is in contrast
to the (Game-Stationary) notion in which even the global
minimum of (10) may not satisfy (Game-Stationary)
condition. The following example, which is borrowed
from [26], illustrates this fact.


_Example 1:_ Consider the optimization problem
(Min-Max) where the function _f_ ( _x, y_ ) = 0 _._ 2 _xy_ _−_ cos( _y_ )
in the region [ _−_ 1 _,_ 1] _×_ [ _−_ 2 _π,_ 2 _π_ ]. It is not hard to check
that this min-max optimization problem has two global
solutions ( _x_ _[∗]_ _, y_ _[∗]_ ) = (0 _, −π_ ) and (0 _, π_ ). However,


neither of these two points satisfy the condition
(Game-Stationary). One criticism of the (Optimization-Stationary) notion is the high computational cost of its evaluation for general non-convex problems. More precisely, unlike the (Game-Stationary) notion, checking (Optimization-Stationary) for a given point **x** ¯
could be computationally intractable for general nonconvex function _f_ ( **x** _,_ **y** ). Finally, it is worth mentioning that, although the two stationary notions
(Optimization-Stationary) and (Game-Stationary) lead
to different definitions of stationarity (as illustrated in
Example 1), the two notions could coincide in special
cases such as when the function _f_ ( **x** _,_ **y** ) is concave in
**y** and its gradient is Lipschitz continuous; see [27] for
more detailed discussion.


_B. Algorithms Based on Potential Reduction_


Constructing a potential and developing an algorithm
to optimize the potential function is a popular way
to solve different types of games. A natural potential
to minimize is the function _g_ ( **x** ), defined in the previous section. In order to solve (10) using standard
first-order algorithms, one needs to have access to the
(sub-)gradients of the function _g_ ( _·_ ). While presenting
the function _g_ ( _·_ ) in closed-form may not be possible,
calculating its gradient at a given point may still be
feasible via Danskin’s theorem stated below.


**Danskin’s Theorem [28]:** Assume the function _f_ ( **x** _,_ **y** )
is differentiable in **x** for all **x** _∈X_ . Furthermore,
assume that _f_ ( **x** _,_ **y** ) is strongly concave in **y** and that
_Y_ is compact. Then, the function _g_ ( **x** ) is differentiable
in **x** . Moreover, for any **x** _∈X_, we have _∇g_ ( **x** ) =
_∇_ **x** _f_ ( **x** _,_ **y** _[∗]_ ( **x** )) _,_ where **y** _[∗]_ ( **x** ) = arg max **y** _∈Y_ _f_ ( **x** _,_ **y** ).
This theorem states that one can compute the gradient
of the function _g_ ( **x** ) through the gradient of the function
_f_ ( _·, ·_ ) when the inner problem is strongly concave, i.e.
_f_ ( **x** _, ·_ ) is strongly concave for all **x** . Therefore, under
the assumptions of Danskin’s theorem, to apply gradient
descent algorithm to (10), one needs to run the following
iterative procedure:


**y** _[r]_ [+1] = arg max **y** _∈Y_ _[f]_ [(] **[x]** _[r]_ _[,]_ **[ y]** [)] (11a)

**x** _[r]_ [+1] = _P_ _X_ ( **x** _[r]_ _−_ _α∇f_ ( **x** _[r]_ _,_ **y** _[r]_ [+1] )) _._ (11b)


More precisely, the dynamics obtained in (11) is equivalent to the gradient descent dynamics **x** _[r]_ [+1] = _P_ _X_ ( **x** _[r]_ _−_
_α∇g_ ( **x** _[r]_ )) _,_ according to Danskin’s theorem. Notice that
computing the value of **y** _[r]_ [+1] in (11) requires finding
the exact solution of the optimization problem in (11a).
In practice, finding such an exact solution may not
be computationally possible. Luckily, even an inexact
version of this algorithm is guaranteed to converge as



6


long as the point **y** _[r]_ [+1] is computed accurately enough
in (11a). In particular, [26] showed that the iterative
algorithm

Find **y** _[r]_ [+1] s.t. _f_ ( **x** _[r]_ _,_ **y** _[r]_ [+1] ) _≥_ max **y** _∈Y_ _[f]_ [(] **[x]** _[r]_ _[,]_ **[ y]** [)] _[ −]_ _[ϵ]_

(12a)

**x** _[r]_ [+1] = _P_ _X_ ( **x** _[r]_ _−_ _α∇f_ ( **x** _[r]_ _,_ **y** _[r]_ [+1] )) _._ (12b)


is guaranteed to find an “approximate stationary point”
where the approximation accuracy depends on the value
of _ϵ_ . Interestingly, even the strong concavity assumption
could be relaxed for convergence of this algorithm as
long as step (12a) is computationally affordable (see [26,
Theorem 35]). The rate of convergence of this algorithm
is accelerated for the case in which the function _f_ ( **x** _,_ **y** )
is concave in **y** (and general nonconvex in **x** ) in [24]
and further improved in [29] through a proximal-based
acceleration procedure. These works (locally) create
strongly convex approximation of the function by adding
proper regularizers, and then apply accelerated iterative
first-order procedures for solving the approximation. The
case in which the function _f_ ( **x** _,_ **y** ) is concave in **y** has
also applications in solving finite max problems of the
form:

min (13)
**x** _∈X_ [max] _[ {][f]_ [1] [(] **[x]** [)] _[, . . ., f]_ _[n]_ [(] **[x]** [)] _[}][.]_


This is due to the fact that this optimization problem can
be rewritten as



min
**x** _∈X_ [max] **y** _∈_ ∆



_n_
� _y_ _i_ _f_ _i_ ( **x** ) _,_ (14)


_i_ =1



where ∆ ≜ _{_ **y** _|_ **y** _≥_ **0** ; [�] _[n]_ _i_ =1 _[y]_ _[i]_ [ = 1] _[}]_ [. Clearly, this]
optimization problem is concave in **y** .
Finally, it is worth emphasizing that, the algorithms
developed based on Danskin’s theorem (or its variations)
can only be applied to problems where step (12a) can
be computed efficiently. While non-convex non-concave
min-max problems do not satisfy this assumption in
general, one may be able to approximate the objective
function with another objective function for which this
assumption is satisfied. The following example illustrates
this possibility in a particular application.


_Example 2:_ Consider the defense problem against adversarial attacks explained in the previous section where
the training of a neural network requires solving the
optimization problem (using similar notations as in (3)):



min

**w**



_N_
� _∥_ max _**δ**_ _i_ _∥≤ε_ _[ℓ]_ [(] _[p]_ [(] **[u]** _[i]_ [ +] _**[ δ]**_ _[i]_ [;] **[ w]** [)] _[, t]_ _[i]_ [)] _[.]_ (15)

_i_ =1



Clearly, the objective function is non-convex in **w**
and non-concave in _**δ**_ . Although finding the strongest
adversarial attacker _**δ**_ _i_, that maximizes the inner problem
in (15), might be intractable, it is usually possible


A [9] B [31] Proposed [24]


Natural 98.58% 97.21% 98.20%



7


_C. Algorithms Based on Solving VI_

Another perspective that leads to the development of
algorithms is the game theoretic perspective. To present
the ideas, first notice that the (Game-Stationary) notion
defined in the previous section can be summarized as


_⟨F_ (¯ **z** ) _,_ **z** _−_ **z** ¯ _⟩≥_ 0 _, ∀_ **z** _∈Z_ (17)



FGSM [17]


PGD [18]



_ε_ = 0 _._ 2 96.09% 96.19% 97.04%

_ε_ = 0 _._ 3 94.82% 96.17% 96.66%

_ε_ = 0 _._ 4 89.84% 96.14% 96.23%


_ε_ = 0 _._ 2 94.64% 95.01% 96.00%

_ε_ = 0 _._ 3 91.41% 94.36% 95.17%

_ε_ = 0 _._ 4 78.67% 94.11% 94.22%



, and _F_ ( _·_ ) is a mapping
�



TABLE I: The performance of different defense algorithms for
training neural network on MNIST dataset. The first row is the
accuracy when no attack is present. The second and the third row
show the performance of different defense algorithms under “FGSM”
and “PGD” attack, respectively. Different _ε_ values show the magnitude
of the attack. Different columns show different defense strategies. The
defense method A (proposed in [9]) and the defense mechanism B
(proposed in [31]) are compared against the proposed method in [24].
More details on the experiment can be found in [24].


to obtain a finite set of weak attackers. In practice,
these attackers could be obtained using heuristics, e.g.
projected gradient ascent or its variants [24]. Thus, [24]
proposes to approximate the above problem with the
following more tractable version:



**x**
where _Z_ ≜ _X × Y_, **z** = � **y**



induced by the objective function _f_ ( **x** _,_ **y** ) in (Min-Max),
defined by



_._
�



**x** ¯
_F_ (¯ **z** ) ≜ _F_ ¯
�� **y**



= _∇_ **x** _f_ (¯ **x** _,_ ¯ **y** )
�� � _−∇_ **y** _f_ (¯ **x** _,_ ¯ **y** )



_k_ =1 _[,]_



1
min
**w** _N_



_N_
�



_K_

� max � _ℓ_ ( _p_ ( **u** _i_ + _**δ**_ _k_ ( **u** _i_ _,_ **w** ); **w** ) _, t_ _i_ ) � _k_

_i_ =1



(16)


where _{_ _**δ**_ _k_ ( **u** _i_ _,_ **w** ) _}_ _[K]_ _k_ =1 [is a set of] _[ K]_ [-weak attackers to]
data point **u** _i_ using the neural network’s weight **w** . Now
the maximization is over a finite number of adversaries
and hence can be transformed to a concave inner max
imization problem using the transformation described
in (13) and (14). The performance of this simple reformulation of the problem is depicted in Table I. As can
be seen in this table, the proposed reformulation yields
comparable results against state-of-the-art algorithms. In
addition, unlike the other two algorithms, the proposed
method enjoys theoretical convergence guarantees. 
In the optimization problem (15), the inner optimization problem is non-concave in _**δ**_ . We approximate
this non-concave function with a concave function by
generating a finite set of adversarial instances in (16) and
used the transformation described in (13),(14) to obtain a
concave inner maximization probelm. This technique can
be useful in solving other general optimization problems
of the form (Min-Max) by approximating the inner
problem with a finite set of points in the set _Y_ . Another
commonly used technique to approximate the inner
maximization in (Min-Max) with a concave problem is to
add a proper regularizer in _y_ . This technique has been
used in [30] to obtain a stable training procedure for
generative adversarial networks.



This way of looking at the min-max optimization problem naturally leads to the design of algorithms that
compute the solution of the (Stampacchia) VI (17).
When problem (Min-Max) is (strongly) convex in
**x** and (strongly) concave in **y**, the mapping _F_ ( **z** )
is (strongly) monotone [3], therefore classical methods
for solving variational inequalities (VI) such as extragradient can be applied [1]. However, in the non-convex
and/or non-concave setting of interest to this article,
the strong monotonicity property no longer holds; and
hence the classical algorithms cannot be used in this
setting. To overcome this barrier, a natural approach is to
approximate the mapping _F_ ( _·_ ) with a series of strongly
monotone mappings and solve a series of strongly monotone VIs. The work [32] builds upon this idea and creates
a series of strongly monotone VIs using proximal point
algorithm, and proposes an iterative procedure named
inexact proximal point (IPP) method, as given below:


Let _F_ **z** _[γ]_ _[r]_ [(] **[z]** [) =] _[ F]_ [(] **[z]** [) +] _[ γ]_ _[−]_ [1] [(] **[z]** _[ −]_ **[z]** _[r]_ [)] (18a)

Let **z** _[r]_ [+1] be the (approx) solution of VI _F_ **z** _[γ]_ _[r]_ [(] _[·]_ [)] _[.]_ [ (18b)]


In (18), _γ >_ 0 is chosen to be small enough so that
the mapping _F_ **z** _[γ]_ _[r]_ [(] **[z]** [)][ becomes strongly monotone (in] **[ z]** [).]
The strongly monotone mapping _F_ **z** _[γ]_ _[r]_ [(] **[z]** [)][ can be solved]
using another iterative procedure such as extra gradient
method, or the iterative procedure


**z** _[t]_ [+1] = _P_ _Z_ ( **z** _[t]_ _−_ _βF_ ( **z** _[t]_ )) (19)


where _β_ denotes the stepsize and _P_ _Z_ is the projection
to the set _Z_ .

Combining the dynamics in (18) with the iterative procedure in (19) leads to a natural double-loop algorithm.
This double-loop algorithm is not always guaranteed to
solve the VI in (17). Instead, it has been shown that
this double-loop procedure computes a solution **z** _[∗]_ to
the following _Minty VI_ :


_⟨F_ ( **z** ) _,_ **z** _−_ **z** _[∗]_ _⟩≥_ 0 _, ∀_ **z** _∈Z._ (20)


3 A strongly monotone mapping _F_ ( _·_ ) satisfies the following _⟨F_ ( **z** ) _−_
_F_ ( **v** ) _,_ **z** _−_ **v** _⟩≥_ _σ∥_ **v** _−_ **z** _∥_ [2] _, ∀_ **v** _,_ **z** _∈Z_, for some constant _σ >_ 0.
If it satisfies this inequality for _σ_ = 0, we say the VI is monotone.


Notice that this solution concept is different than the
solution ¯ **z** in (17) as it has _F_ ( **z** ) instead of _F_ (¯ **z** ) in
the left hand side. While these two solution concepts
are different in general, it is known that if _Z_ is a
convex set, then any **z** _[∗]_ satisfying (20) also satisfies
(17). Furthermore, if _F_ ( _·_ ) is monotone (or when _f_ ( _·, ·_ )
is convex in **x** and concave in **y** ), then any solution to
(17) is also a solution to (20). While such monotonicity
requirement can be slightly relaxed to cover a wider
range of non-convex problems (see e.g. [33]), it is
important to note that for generic non-convex, and/or
non-convave function _f_ ( _·, ·_ ), there may not exist **z** _[∗]_ that
satisfies (20); see Example 3.


_Example 3:_ Consider the following function which is
non-convex in _x_, but concave in _y_ :


_f_ ( _x, y_ ) = _x_ [3] + 2 _xy −_ _y_ [2] _, X × Y_ = [ _−_ 1 _,_ 1] _×_ [ _−_ 1 _,_ 1] _._


One can verify that there are only two points, (0 _,_ 0) and
( _−_ 1 _, −_ 1) that satisfy (17). However, none of the above
solutions satisfies (20). To see this, one can verify that
_⟨F_ ( _z_ ) _, z −_ _z_ _[∗]_ _⟩_ = _−_ 4 _<_ 0 for _z_ = (0 _, −_ 1) and _z_ _[∗]_ =
( _−_ 1 _, −_ 1) and that _⟨F_ ( _z_ ) _, z −_ _z_ _[∗]_ _⟩_ = _−_ 3 _<_ 0 for _z_ =
( _−_ 1 _,_ 0) and _z_ _[∗]_ = (0 _,_ 0). Since any **z** _[∗]_ satisfying (20) will
satisfy (17), we conclude that there is no point satisfying
(20) for this min-max problem. In conclusion, the VIs (17) and (20) offer new perspectives to analyze problem (Min-Max); but the existing
algorithms such as (18) cannot deal with many problems
covered by the potential based methods discussed in
Sec. IV-B (for example when _f_ ( **x** _,_ **y** ) is non-convex in
_x_ and concave in **y** or when a point satisfying (20) does
not exists as we explained in Example 3). Moreover,
the VI-based algorithms completely ignore the order
of maximization and minimization in (Min-Max) and,
hence, cannot be applied to problems in which the order
of min and max is crucial.


_D. Algorithms Using Single-Loop Update_


The algorithms discussed in the previous two subsections are all _double loop_ algorithms, in which one
variable (e.g. **y** ) is updated in a few consecutive iterations before another variable gets updated. In many
practical applications, however, _single loop_ algorithms
which update **x** and **y** either alternatingly or simultaneously are preferred. For example, in problem (8),
the jammer often pretends to be the regular user, so
it updates simultaneously with the regular users [6].
However, it is challenging to design and analyze single
loop algorithms for problem (Min-Max) — even for the
simplest linear problem where _f_ ( **x** _,_ **y** ) = _⟨_ **x** _,_ **y** _⟩_, the
single-loop algorithm GDA diverges; see the discussion
in Sec. III.



8


To overcome the above challenges, [19] proposes a
single loop algorithm called Hybrid Block Successive
Approximation (HiBSA), whose iterations are given by


**x** _[r]_ [+1] = _P_ _X_ ( **x** _[r]_ _−_ _β_ _[r]_ _∇_ **x** _f_ ( **x** _[r]_ _,_ **y** _[r]_ )) _,_ (21a)

**y** _[r]_ [+1] = _P_ _Y_ �(1 + _γ_ _[r]_ _ρ_ ) **y** _[r]_ + _ρ∇_ **y** _f_ � **x** _[r]_ [+1] _,_ **y** _[r]_ [��] _,_ (21b)


where _β_ _[r]_ _, ρ >_ 0 are the step sizes; _γ_ _[r]_ _>_ 0 is some
perturbation parameter. This algorithm can be further
generalized to optimize certain approximation functions
of _x_ and _y_, similarly to the successive convex approximation strategies used in min-only problems [34], [35].
The “hybrid” in the name refers to the fact that this
algorithm contains both the descent and ascent steps.
The HiBSA iteration is very similar to the GDA algorithm mentioned previously, in which gradient descent
and ascent steps are performed alternatingly. The key
difference here is that the **y** update includes an additional
term _γ_ _[r]_ _ρ_ **y** _[r]_, so that at each iteration the **y** update
represents a “perturbed” version of the original gradient
ascent step. The idea is that after the perturbation,
the new iteration **y** _[r]_ [+1] is “closer” to the old iteration
**y** _[r]_, so it can avoid the divergent patterns depicted in
Fig. 1. Intuitively, as long as the perturbation eventually
goes to zero, the algorithm will still converge to the
desired solutions. Specifically, it is shown in [19] that,
if _f_ ( **x** _,_ **y** ) is strongly concave in **y**, then one can simply
remove the perturbation term (by setting _γ_ _[r]_ = 0 for all
_r_ ), and the HiBSA will converge to a point satisfying
condition (Game-Stationary). Further, if _f_ ( **x** _,_ **y** ) is only
concave in **y**, then one needs to choose _β_ _[r]_ = _O_ (1 _/r_ [1] _[/]_ [2] ),
and _γ_ _[r]_ = _O_ (1 _/r_ [1] _[/]_ [4] ) to converge to a point satisfying
condition (Game-Stationary).


_Example 4:_ We apply the HiBSA to the power control
problem (8). It is easy to verify that the jammer’s
objective is strongly concave over the feasible set.
We compare HiBSA with two classic algorithms:
interference pricing [36], and the WMMSE [37], both of
which are designed for power control problem _without_
assuming the presence of the jammer. Our problem
is tested using the following setting. We construct a
network with 10 regular user and a single jammer. The
interference channel among the users and the jammer is
generated using uncorrelated fading channel model with
channel coefficients generated from the complex zeromean Gaussian distribution with unit covariance.

From Fig. 2 (top), we see that the pricing algorithm
monotonically increases the sum rate (as is predicted by
theory), while HiBSA behaves differently: after some
initial oscillation, the algorithm converges to a value that
has lower sum-rate. Further in Fig. 2 (bottom), we do see
that by using the proposed algorithm, the jammer is able
to effectively reduce the total sum rate of the system. ■


Fig. 2: The convergence curves and total averaged system performance comparing three algorithms: WMMSE, Interference Pricing and
HiBSA. All users’ power budget is fixed at _P_ = 10 [SNR] _[/]_ [10] . For test
cases without a jammer, we set _σ_ _k_ [2] [= 1][ for all] _[ k]_ [. For test cases with]
a jammer, we set _σ_ _k_ [2] [= 1] _[/]_ [2][ for all] _[ k]_ [ and let the jammer have the rest]
of the noise power, i.e., _p_ 0 _,_ max = _N/_ 2. Figure taken from [19].


_E. Extension to Zeroth-order Based Algorithms_


Up to now, all the algorithms reviewed require firstorder (gradient) information. In this subsection, we disucss a useful extension when only _zeroth-order_ (ZO)
information is available. That is, we only have access
to the objective values _f_ ( **x** _,_ **y** ) at a given point ( **x** _,_ **y** )
at every iteration. This type of algorithm is useful, for
example, in practical adversarial attack scenario where
the attacker only has access to the output of the ML
model [16].
To design algorithms in the ZO setting, one typically
replaces the gradient _∇h_ ( **x** ) with some kind of _gradient_
_estimate_ . One popular estimate is given by



�
_∇_ **x** _h_ ( **x** ) = [1]

_q_



_q_
�


_i_ =1



_d_ [ _h_ ( **x** + _µ_ **u** _i_ ) _−_ _h_ ( **x** )]

**u** _i_ _,_
_µ_



where _{_ **u** _i_ _}_ _[q]_ _i_ =1 [are] _[ q][ i.i.d.]_ [ random direction vectors]
drawn uniformly from the unit sphere, and _µ >_ 0 is
a smoothing parameter. We note that the ZO gradient
estimator involves the random direction sampling w.r.t.
**u** _i_ . It is known that _∇_ [�] **x** _h_ ( **x** ) provides an unbiased
estimate of the gradient of the smoothing function of
_f_ rather than the true gradient of _f_ . Here the smoothing
function of _f_ is defined by _h_ _µ_ ( **x** ) = E **v** [ _h_ ( **x** + _µ_ **v** )],
where **v** follows the uniform distribution over the unit



9


Euclidean ball. Such a gradient estimate is used in [16] to
develop a ZO algorithm for solving min-max problems.


_Example 5:_ To showcase the performance comparison
between ZO based and first-order (FO) based algorithm,
we consider applying HiBSA and its ZO version to
the data poisoning problem (3). In particular, let us
consider attacking the data set used to train a logistic
regression model. We first set the poisoning ratio, i.e.,
the percentage of the training samples attacked, to 15%.
Fig. 3 (top) demonstrates the testing accuracy (against
iterations) of the model learnt from poisoned training
data, where the poisoning attack is generated by ZO minmax (where the adversarial only has access to victim
model outputs) and FO min-max (where the adversarial
has access to details of the victim model). As we can
see from Fig. 3, the poisoning attack can significantly
reduce the testing accuracy compared to the clean model.
Further, the ZO min-max yields promising attacking performance comparable to the FO min-max. Additionally,
in Fig. 3 (bottom), we present the testing accuracy of
the learned model under different data poisoning ratios.
As we can see, only 5% poisoned training data can
significantly break the testing accuracy of a well-trained
model. The details of this experiment can be found in

[16]. 

Fig. 3: Empirical performance of ZO/FO-Min-Max in poisoning
attacks, where the ZO/FO-Min-Max algorithm refers to HiBSA with
either ZO or FO oracle: (top) testing accuracy versus iterations (the
shaded region represents variance of 10 random trials), and (bottom)
testing accuracy versus data poisoning ratio. Figure taken from [16].


V. C ONNECTIONS AMONG ALGORITHMS AND

OPTIMALITY CONDITIONS


In this section, we summarize our discussion on
various optimality conditions as well as algorithm
performance. First, in Fig. 4, we describe the relationship between the Minty condition (20), the
(Optimization-Stationary) and (Game-Stationary). Second, we compare the properties of different algorithms,
such as their convergence conditions and optimality
criteria in Table II. Despite the possible equivalence between the optimality conditions, we still keep the column
“optimality criteria” because these are the criteria based
on which the algorithms are originally designed.


Non-convex min-max optimization problems appear
in a wide range of applications. Despite the recent
developments in solving these problems, the available
tool sets and theories are still very limited. In particular,
as discussed in section IV, these algorithms require at
least one of the two following assumptions:


i) The objective of one of the two player is easy
to optimize. For example, the object function in
(Min-Max) is concave in **y** .
ii) The Minty solutions satisfying (20) for the min-max
game are the solutions to the Stampchia VI (17).

While the first assumption is mostly easy to check,
the second assumption might not be easy to verify.
Nevertheless, these two conditions do not imply each
other. Moreover, there is a wide range of non-convex
min-max problem instances that does not satisfy either of
these assumptions. For solving those problems, it might
be helpful to approximate them with a min-max problem
satisfying one of these two assumptions.
As future research directions, a natural first open
direction is toward the development of algorithms that



10


can work under a more relaxed set of assumptions. We
emphasize that the class of problems that are provably
solvable (to satisfy either (Optimization-Stationary) or
(Game-Stationary)) is still very limited, so it is important
to extend solvable problems to a more general set of nonconvex non-concave functions. One possible first step to
address this is to start from algorithms that converge to
desired solutions when initialized close enough to them,
i.e. _local_ convergence; for recent developments on this
topic see [38]. Another natural research direction is about
the development of the algorithms in the absence of
smoothness assumptions. When the objective function
of the players are non-smooth but “proximal-gradient
friendly”, many of the results presented in this review
can still be used by simply using proximal gradient
instead of gradient [39]. These scenarios happen, for
example, when the objective function of the players
is a summation of a smooth function and a convex

non-smooth function. Additionally, it is of interest to
customize the existing generic non-convex min-max algorithms to practical applications, for example to rein
One of the main applications of non-convex minmax problems, as mentioned in section II, is to design
systems that are robust against uncertain parameters or
the existence of adversaries. A major question in these
applications is whether one can provide “robustness
certificate” after solving the optimization problem. In
particular, can we guarantee or measure the robustness
level in these applications? The answer to this question is
closely tied to the development of algorithms for solving
non-convex min-max problems.
Another natural research direction is about the rate

of convergence of the developed algorithms. For example, while we know solving min-max problems to
(Optimization-Stationary) is easy when (Min-Max) is
concave in **y** (and possibly non-convex in **x** ), the optimal
rate of convergence (using gradient information) is still
not known. Moreover, it is natural to ask whether knowing the Hessian or higher order derivatives of the objective function could improve the performance of these
algorithms. So far, most of the algorithms developed
for non-convex min-max problems rely only on gradient
information.


R EFERENCES


[1] A. Nemirovski, “Prox-method with rate of convergence o(1/t)
for variational inequalities with lipschitz continuous monotone
operators and smooth convex-concave saddle point problems,”
_SIAM Journal on Optimization_, vol. 15, no. 1, pp. 229–251, 2004.

[2] A. Juditsky and A. Nemirovski, “Solving variational inequalities
with monotone operators on domains given by linear minimization oracles,” _Mathematical Programming_, vol. 156, no. 1-2, pp.
221–256, 2016.

[3] P. Tseng, “On accelerated proximal gradient methods for convexconcave optimization,” _submitted to SIAM Journal on Optimiza-_
_tion_, vol. 2, p. 3, 2008.


11













|Algorithm|Optimality Criterion|Oracles|Assumptions|Col5|Other Comments|
|---|---|---|---|---|---|
|_Multi-Step GDA_ [24]|Game-Stationary|FO|_f_ NC in_ x_, concave in_ y_|_f_ NC in_ x_, concave in_ y_|Det. & DL, & Acc.|
|_CDIAG_ [29]|Optimization-Stationary|FO|_f_ NC in_ x_, concave in_ y_|_f_ NC in_ x_, concave in_ y_|Det. & TL, & Acc.|
|_PG-SMD/ PGSVRG_ [33]|Optimization-Stationary|FO||_f_ concave in_ y_|St. & DL|
|_PG-SMD/ PGSVRG_ [33]|Optimization-Stationary|FO|_f_ = 1<br>_n_<br>~~P~~_n_<br>_i_=1 _fi_<br>_fi_(_x, y_) NC in_ x_|_fi_ concave in_ y_|_fi_ concave in_ y_|
|_IPP_ for VI [32]|Minty-Condition|FO|NC in x, y|NC in x, y|Det. & DL|
|_GDA_ [27]|Optimization-Stationary|FO|_f_ NC in_ x_, concave in_ y_|_f_ NC in_ x_, concave in_ y_|Det. & St. & DL<br>_y_-step has small stepsize|
|_HiBSA_ [19] [16]|Game-Stationary|FO & ZO|_f_ NC in_ x_, concave in_ y_|_f_ NC in_ x_, concave in_ y_|Det. & SL|


TABLE II: Summary of algorithms for the min-max optimization problem (Min-Max) along with their convergence guarantees. Note that in


the third column, we characterize the type of the oracle used, i.e., FO or ZO. In the last column are other comments about the algorithms, i.e

deterministic (Det.) or stochastic (St.), single loop (SL) or double loop (DL) or triple loop (TL), Acceleration (Acc.) or not. Moreover, we use

the abbreviations NC for non-convex.




[4] Y. Nesterov, “Dual extrapolation and its applications to solving
variational inequalities and related problems,” _Mathematical Pro-_
_gramming_, vol. 109, no. 2-3, pp. 319–344, 2007.

[5] Y.-F. Liu, Y.-H. Dai, and Z.-Q. Luo, “Max-min fairness linear
transceiver design for a multi-user MIMO interference channel,”
in _Proc. of International Conference on Communications_, 2011.

[6] R. H. Gohary, Y. Huang, Z.-Q. Luo, and J.-S. Pang, “Generallized
iterative water-filling algorithm for distributed power control
in the presence of a jammer,” _IEEE Transactions On Signal_
_Processing_, vol. 57, no. 7, pp. 2660–2674, 2009.

[7] G. B. Giannakis, Q. Ling, G. Mateos, I. D. Schizas, and H. Zhu,
“Decentralized learning for wireless communications and networking,” in _Splitting Methods in Communication and Imaging_ .
Springer New York, 2015.

[8] A. Nedi´c, A. Olshevsky, and M. G. Rabbat, “Network topology
and communication-computation tradeoffs in decentralized optimization,” _Proceedings of the IEEE_, vol. 106, no. 5, pp. 953–976,
May 2018.

[9] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu,
“Towards deep learning models resistant to adversarial attacks,”
in _International Conference on Learning Representations_, 2018.

[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, “Generative
adversarial nets,” in _Proc. of Advances in Neural Information_
_Processing systems_, 2014, pp. 2672–2680.

[11] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative
adversarial networks,” in _Proc. of Interantional Conference on_
_Machine Learning_, 2017.

[12] B. H. Zhang, B. Lemoine, and M. Mitchell, “Mitigating unwanted
biases with adversarial learning,” in _Proceedings of the 2018_
_AAAI/ACM Conference on AI, Ethics, and Society_ . ACM, 2018,
pp. 335–340.

[13] S. Baharlouei, M. Nouiehed, A. Beirami, and M. Razaviyayn,
“R´enyi fair inference,” in _Proc. of International Conference on_
_Learning Representation_, 2020.

[14] D. Xu, S. Yuan, L. Zhang, and X. Wu, “Fairgan: Fairness-aware
generative adversarial networks,” in _2018 IEEE International_
_Conference on Big Data (Big Data)_ . IEEE, 2018, pp. 570–575.

[15] D. Madras, E. Creager, T. Pitassi, and R. S. Zemel, “Learning
adversarially fair and transferable representations,” in _Proc. of_
_International Conference on Machine Learning_, 2018, pp. 3381–
3390.

[16] S. Liu, S. Lu, X. Chen, Y. Feng, K. Xu, A. Al-Dujaili, M. Hong,
and U.-M. Obelilly, “Min-max optimization without gradients:
convergence and applications to black-box evasion and poison


ing attacks,” in _Proc. of International Conference on Machine_
_Learning_, 2020.

[17] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and
harnessing adversarial examples,” in _Proc. of International Con-_
_ference on Learning Representations_, 2015.

[18] A. Kurakin, I. Goodfellow, and S. Bengio, “Adversarial machine
learning at scale,” in _Proc. of International Conference on Learn-_
_ing Representations_, 2017.

[19] S. Lu, I. Tsaknakis, M. Hong, and Y. Chen, “Hybrid block
successive approximation for one-sided non-convex min-max
problems: algorithms and applications,” _IEEE Transactions on_
_Signal Processing_, 2020, accepted for publication.

[20] T. Chang, M. Hong, H. Wai, X. Zhang, and S. Lu, “Distributed
learning in the nonconvex world: From batch data to streaming
and beyond,” _IEEE Signal Processing Magazine_, vol. 37, no. 3,
pp. 26–38, 2020.

[21] Y. Nesterov, _Introductory lectures on convex optimization: A_
_basic course_ . Springer Science & Business Media, 2013, vol. 87.

[22] K. G. Murty and S. N. Kabadi, “Some NP-complete problems
in quadratic and nonlinear programming,” _Mathematical_
_Programming_, vol. 39, no. 2, pp. 117–129, Jun 1987.

[[Online]. Available: http://dx.doi.org/10.1007/BF02592948](http://dx.doi.org/10.1007/BF02592948)

[23] J.-S. Pang and G. Scutari, “Nonconvex games with side constraints,” _SIAM Journal on Optimization_, vol. 21, no. 4, pp. 1491–
1522, 2011.

[24] M. Nouiehed, M. Sanjabi, T. Huang, J. D. Lee, and M. Razaviyayn, “Solving a class of non-convex min-max games using
iterative first order methods,” in _Advances in Neural Information_
_Processing Systems 32_, 2019, pp. 14 905–14 916.

[25] J.-S. Pang and M. Razaviyayn, “A unified distributed algorithm
for non-cooperative games,” _Big Data over Networks_, p. 101,
2016.

[26] C. Jin, P. Netrapalli, and M. I. Jordan, “What is local optimality
in nonconvex-nonconcave minimax optimization?” in _Proc. of_
_Interantional Conference on Machine Learning_, 2020.

[27] T. Lin, C. Jin, and M. I. Jordan, “On gradient descent ascent for
nonconvex-concave minimax problems,” in _Proc. of International_
_Conference on Machine Learning_, 2020.

[28] J. M. Danskin, “The theory of max-min, with applications,” _SIAM_
_Journal on Applied Mathematics_, vol. 14, no. 4, pp. 641–664,
1966.

[29] K. K. Thekumparampil, P. Jain, P. Netrapalli, and S. Oh, “Efficient algorithms for smooth minimax optimization,” in _Advances_
_in Neural Information Processing Systems 32_, 2019, pp. 12 659–
12 670.


12




[30] M. Sanjabi, J. Ba, M. Razaviyayn, and J. D. Lee, “On the
convergence and robustness of training gans with regularized
optimal transport,” in _Advances in Neural Information Processing_
_Systems_, 2018, pp. 7091–7101.

[31] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. E. Ghaoui, and M. Jordan,
“Theoretically principled trade-off between robustness and accuracy,” in _International Conference on Machine Learning_, 2019,
pp. 7472–7482.

[32] Q. Lin, M. Liu, H. Rafique, and T. Yang, “Solving
weakly-convex-weakly-concave saddle-point problems as
weakly-monotone variational inequality,” _arXiv_ _preprint_
_arXiv:1810.10207_, 2018.

[33] H. Rafique, M. Liu, Q. Lin, and T. Yang, “Non-convex min-max
optimization: Provable algorithms and applications in machine
learning,” _Smooth Games Optimization and Machine Learning_
_Workshop (NIPS 2018), arXiv preprint arXiv:1810.02060_, 2018.

[34] M. Hong, M. Razaviyayn, Z.-Q. Luo, and J.-S. Pang, “A unified
algorithmic framework for block-structured optimization involving big data,” _IEEE Signal Processing Magazine_, vol. 33, no. 1,
pp. 57 – 77, 2016.

[35] M. Razaviyayn, M. Hong, and Z.-Q. Luo, “A unified convergence
analysis of block successive minimization methods for nonsmooth optimization,” _SIAM Journal on Optimization_, vol. 23,
no. 2, pp. 1126–1153, 2013.

[36] D. A. Schmidt, C. Shi, R. A. Berry, M. L. Honig, and
W. Utschick, “Comparison of distributed beamforming algorithms for mimo interference networks,” _IEEE Transactions on_
_Signal Processing_, vol. 61, no. 13, pp. 3476–3489, July 2013.

[37] Q. Shi, M. Razaviyayn, Z.-Q. Luo, and C. He, “An iteratively
weighted MMSE approach to distributed sum-utility maximization for a MIMO interfering broadcast channel,” _IEEE Transac-_
_tions on Signal Processing_, vol. 59, no. 9, pp. 4331–4340, 2011.

[38] L. Adolphs, H. Daneshmand, A. Lucchi, and T. Hofmann, “Local
saddle point optimization: A curvature exploitation approach,” in
_The 22nd International Conference on Artificial Intelligence and_
_Statistics_, 2019, pp. 486–495.

[39] B. Barazandeh and M. Razaviyayn, “Solving non-convex nondifferentiable min-max games using proximal gradient method,”
in _ICASSP 2020-2020 IEEE International Conference on Acous-_
_tics, Speech and Signal Processing (ICASSP)_ . IEEE, 2020, pp.
3162–3166.

[40] H.-T. Wai, M. Hong, Z. Yang, Z. Wang, and K. Tang, “Variance
reduced policy evaluation with smooth function approximation,”
in _Advances in Neural Information Processing Systems 32_, 2019,
pp. 5784–5795.


