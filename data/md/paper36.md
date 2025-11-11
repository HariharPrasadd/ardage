## **Melding the Data-Decisions Pipeline: Decision-Focused Learning for** **Combinatorial Optimization**

**Bryan Wilder, Bistra Dilkina, Milind Tambe**
Center for Artificial Intelligence in Society, University of Southern California
_{_ bwilder, dilkina, tambe _}_ @usc.edu



**Abstract**


Creating impact in real-world settings requires artificial intelligence techniques to span the full pipeline from data, to
predictive models, to decisions. These components are typically approached separately: a machine learning model is
first trained via a measure of predictive accuracy, and then its
predictions are used as input into an optimization algorithm
which produces a decision. However, the loss function used to
train the model may easily be misaligned with the end goal,
which is to make the best decisions possible. Hand-tuning
the loss function to align with optimization is a difficult and
error-prone process (which is often skipped entirely).

We focus on combinatorial optimization problems and introduce a general framework for decision-focused learning,
where the machine learning model is directly trained in conjunction with the optimization algorithm to produce highquality decisions. Technically, our contribution is a means
of integrating common classes of discrete optimization problems into deep learning or other predictive models, which are
typically trained via gradient descent. The main idea is to use
a continuous relaxation of the discrete problem to propagate
gradients through the optimization procedure. We instantiate
this framework for two broad classes of combinatorial problems: linear programs and submodular maximization. Experimental results across a variety of domains show that decisionfocused learning often leads to improved optimization performance compared to traditional methods. We find that standard
measures of accuracy are not a reliable proxy for a predictive
model’s utility in optimization, and our method’s ability to
specify the true goal as the model’s training objective yields
substantial dividends across a range of decision problems.


**Introduction**


The goal in many real-world applications of artificial intelligence is to create a pipeline from data, to predictive
models, to decisions. Together, these steps enable a form
of evidence-based decision making which has transformative potential across domains such as healthcare, scientific
discovery, transportation, and more (Horvitz and Mitchell
2010; Horvitz 2010). This pipeline requires two technical
components: machine learning models and optimization algorithms. Machine learning models use the data to predict


Copyright c _⃝_ 2019, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.



unknown quantities; optimization algorithms use these predictions to arrive at a decision which maximizes some objective. Our concern here is combinatorial optimization, which
is ubiquitous in real-world applications of artificial intelligence, ranging from matching applicants to public housing
to selecting a subset of movies to recommend. We focus
on common classes of combinatorial problems which have
well-structured continuous relaxations, e.g., linear programs
and submodular maximization. A vast literature has been devoted to combinatorial optimization (Korte et al. 2012). Importantly though, optimization is often insufficient without
the broader pipeline because the objective function is unknown and must predicted via machine learning.
While machine learning has witnessed incredible growth
in recent years, the two pieces of the pipeline are treated
entirely separately by typical training approaches. That is,
a system designer will first train a predictive model using
some standard measure of accuracy, e.g., mean squared error for a regression problem. Then, the model’s predictions
are given as input to the optimization algorithm to produce a
decision. Such _two-stage_ approaches are extremely common
across many domains (Wang et al. 2006; Fang et al. 2016;
Mukhopadhyay et al. 2017; Xue et al. 2016). This process
is justified when the predictive model is perfect, or near-so,
since completely accurate predictions also produce the best
decisions. However, in complex learning tasks, all models
will make errors and the training process implicitly trades
off where these errors will occur. When prediction and optimization are separate, this tradeoff is divorced from the goal
of the broader pipeline: to make the best decision possible.
We propose a _decision-focused learning_ framework
which melds the data-decisions pipeline by integrating prediction and optimization into a single end-to-end system.
That is, the predictive model is trained using the quality
of the decisions which it induces via the optimization algorithm. Similar ideas have recently been explored in the
context of convex optimization (Donti, Amos, and Kolter
2017), but to our knowledge ours is the first attempt to train
machine learning systems for performance on _combinato-_
_rial_ decision-making problems. Combinatorial settings raise
new technical challenges because the optimization problem
is discrete. However, machine learning systems (e.g., deep
neural networks) are often trained via gradient descent.
Our first contribution is a general framework for training


machine learning models via their performance on combinatorial problems. The starting point is to relax the combinatorial problem to a continuous one. Then, we analytically differentiate the optimal solution to the continuous problem as
a function of the model’s predictions. This allows us to train
using a continuous proxy for the discrete problem. At test
time, we round the continuous solution to a discrete point.
Our second contribution is to instantiate this framework
for two broad classes of combinatorial problems: linear programs and submodular maximization problems. Linear programming encapsulates a number of classical problems such
as shortest path, maximum flow, and bipartite matching.
Submodular maximization, which reflects the intuitive phenomena of diminishing returns, is also ubiquitous; applications range from social networks (Kempe, Kleinberg, and
Tardos 2003) to recommendation systems (Viappiani and
Boutilier 2010). In each case, we resolve a set of technical
challenges to produce well-structured relaxations which can
be efficiently differentiated through.
Finally, we give an extensive empirical investigation,
comparing decision-focused and traditional methods on a
series of domains. Decision-focused methods often improve
performance for the pipeline as a whole (i.e., decision quality) despite worse predictive accuracy according to standard
measures. Intuitively, the predictive models trained via our
approach focus specifically on qualities which are important
for making good decisions. By contrast, more generic methods produce predictions where error is distributed in ways
which are not aligned with the underlying task.


**Problem description**


We consider combinatorial optimization problems of the
form max _x∈X_ _f_ ( _x, θ_ ), where _X_ is a discrete set enumerating the feasible decisions. Without loss of generality, _X ⊆_
_{_ 0 _,_ 1 _}_ _[n]_ and the decision variable _x_ is a binary vector. The
objective _f_ depends on a parameter _θ ∈_ Θ. If _θ_ were known
exactly, a wide range of existing techniques could be used
to solve the problem. In this paper, we consider the challenging (but prevalent) case where _θ_ is unknown and must
be inferred from data. For instance, in bipartite matching, _x_
represents whether each pair of nodes were matched and _θ_
contains the reward for matching each pair. In many applications, these affinities are learned from historical data.
Specifically, the decision maker observes a feature vector
_y ∈Y_ which is correlated with _θ_ . This introduces a learning problem which must be solved prior to optimization. As
in classical supervised learning, we formally model _y_ and _θ_
as drawn from a joint distribution _P_ . Our algorithm will observe training instances ( _y_ 1 _, θ_ 1 ) _..._ ( _y_ _N_ _, θ_ _N_ ) drawn iid from
_P_ . At test time, we are give a feature vector _y_ corresponding to an _unobserved θ_ . Our algorithm will use _y_ to predict
a parameter value _θ_ [ˆ] . Then, we will solve the optimization
problem max _x_ _f_ ( _x,_ _θ_ [ˆ] ) to obtain a decision _x_ _[∗]_ . Our utility is
the objective value that _x_ _[∗]_ obtains with respect to the _true_
_but unknown_ parameter _θ_, _f_ ( _x_ _[∗]_ _, θ_ ).
Let _m_ : _Y →_ Θ denote a model mapping observed
features to parameters. Our goal is to (using the training
data) find a model _m_ which maximizes expected perfor


mance on the underlying optimization task. Define _x_ _[∗]_ ( _θ_ ) =
arg max _x∈X_ _f_ ( _x, θ_ ) to be the optimal _x_ for a given _θ_ . The
end goal of the data-decisions pipeline is to maximize


E (1)
_y,θ∼P_ [[] _[f]_ [(] _[x]_ _[∗]_ [(] _[m]_ [(] _[y]_ [))] _[, θ]_ [)]]


The classical approach to this problem is a _two-stage_
method which first learns a model using a task-agnostic
loss function (e.g., mean squared error) and then uses
the learned model to solve the optimization problem. The
model class will have its own parameterization, which we
denote by _m_ ( _y, ω_ ). For instance, the model class could
consist of deep neural networks where _ω_ denotes the
weights. The two-stage approach first solves the problem
min _ω_ E _y,θ∼P_ [ _L_ ( _θ, m_ ( _y, ω_ ))], where _L_ is a loss function.
Such a loss function measures the overall “accuracy” of the
model’s predictions but does not specifically consider how
_m_ will fare when used for decision making. The question we
address is whether it is possible to do better by specifically
training the model to perform well on the decision problem.


**Previous work**

There is a growing body of research at the interface of
machine learning and discrete optimization (Vinyals, Fortunato, and Jaitly 2015; Bertsimas and Dunn 2017; Khalil
et al. 2017b; Khalil et al. 2017a). However, previous work
largely focuses on either using discrete optimization to find
an accuracy-maximizing predictive model or using machine
learning to speed up optimization algorithms. Here, we pursue a deeper synthesis; to our knowledge, this work is the
first to train predictive models using combinatorial optimization performance with the goal of improving decision making.
The closest work to ours in motivation is (Donti, Amos,
and Kolter 2017), who study task-based convex optimization. Their aim is to optimize a convex function which depends on a learned parameter. As in their work, we use the
idea of differentiating through the KKT conditions. However, their focus is entirely on continuous problems. Our
discrete setting raises new technical challenges, highlighted
below. Elmachtoub and Grigas (2017) also propose a means
of integrating prediction and optimization; however, their
method applies strictly to linear optimization and focuses
on linear predictive models while our framework applies to
nonlinear problems with more general models (e.g., neural networks). Finally, some work has noted that two-stage
methods lead to poor optimization performance in specific
domains (Beygelzimer and Langford 2009; Ford et al. 2015).
Our work is also related to recent research in structured
prediction (Belanger, Yang, and McCallum 2017; Tu and
Gimpel 2018; Niculae et al. 2018; Djolonga and Krause
2017). which aims to make a prediction lying in a discrete
set. This is fundamentally different than our setting since
their goal is to _predict_ an external quantity, not to _optimize_
and find the best decision possible. However, structured prediction sometimes integrates a discrete optimization problem as a module within a larger neural network. The closest such work technically to ours is (Tschiatschek, Sahin,


and Krause 2018), who design a differentiable algorithm for
submodular maximization in order to predict choices made
by users. Their approach is to introduce noise into the standard greedy algorithm, making the probability of outputting
a given set differentiable. There are two key differences between our approaches. First, their approach does not apply to
the decision-focused setting because it maximizes the likelihood of a _fixed_ set but cannot optimize for finding the best
set. Second, exactly computing gradients for their algorithm
requires marginalizing over the _k_ ! possible permutations of
the items, forcing a heuristic approximation to the gradient.
Our approach allows closed-form differentiation.

Some deep learning architectures differentiate through
gradient descent steps, related to our approach in the submodular setting. Typically, previous approaches explicitly
unroll _T_ iterations of gradient descent in the computational
graph (Domke 2012). However, this approach is usually employed for _unconstrained_ problems where each iteration is
a simple gradient step. By contrast, our combinatorial problems are constrained, requiring a projection step to enforce
feasibility. Unrolling the projection step may be difficult,
and would incur a large computational cost. We instead exploit the fact that gradient ascent converges to a local optimum and analytically differentiate via the KKT conditions.


**General framework**


Our goal is to integrate combinatorial optimization into the
loop of gradient-based training. That is, we aim to directly
train the predictive model _m_ by running gradient steps on the
objective in Equation 1, which integrates both prediction and
optimization. The immediate difficulty is the dependence on
_x_ _[∗]_ ( _m_ ( _y, ω_ )). This term is problematic for two reasons. First,
it is a discrete quantity since _x_ _[∗]_ is a decision from a binary
set. This immediately renders the output nondifferentiable
with respect to the model parameters _ω_ . Second, even if _x_ _[∗]_
were continuous, it is still defined as the solution to an optimization problem, so calculating a gradient requires us to
differentiate through the argmax operation.

We resolve both difficulties by considering a continuous
relaxation of the combinatorial decision problem. We show
that for a broad class of combinatorial problems, there are
appropriate continuous relaxations such that we can analytically obtain derivatives of the continuous optimizer with respect to the model parameters. This allows us to train any
differentiable predictive model via gradient descent on a
continuous surrogate to Equation 1. At test time, we solve
the true discrete problem by rounding the continuous point.

More specifically, we relax the discrete constraint _x ∈X_
to the continuous one _x ∈_ _conv_ ( _X_ ) where _conv_ denotes the
convex hull. Let _x_ ( _θ_ ) = arg max _x∈conv_ ( _X_ ) _f_ ( _x, θ_ ) denote
the optimal solution to the continuous problem. To train our
predictive model, we would like to compute gradients of the
whole-pipeline objective given by Equation 1, replacing the
discrete quantity _x_ _[∗]_ with the continuous _x_ . We can obtain a
stochastic gradient estimate by sampling a single ( _y, θ_ ) from
the training data. On this sample, the chain rule gives



By solving this system of linear equations, we can obtain the desired term _[dx]_ _dθ_ [. However, the above approach is]

a general framework; our main technical contribution is to
instantiate it for specific classes of combinatorial problems.
Specifically, we need (1) an appropriate continuous relaxation, along with a means of solving the continuous optimization problem and (2) efficient access to the terms in
Equation 2 which are needed for the backward pass (i.e.,
gradient computation). We provide both ingredients for two
broad classes of problems: linear programming and submodular maximization. In each setting, the high-level challenge
is to ensure that the continuous relaxation is differentiable, a
feature not satisfied by naive alternatives. We also show how
to efficiently compute terms needed for the backward pass,
especially for the more intricate submodular case.


**Linear programming**


The first setting that we consider is combinatorial problems
which can be expressed as a linear program with equality
and inequality constraints in the form


max _θ_ _[T]_ _x_ s.t. _Ax_ = _b, Gx ≤_ _h_ (3)


Example problems include shortest path, maximum flow,
bipartite matching, and a range of other domains. For instance, in a shortest path problem _θ_ contains the cost for
traversing each edge, and we are interested in problems
where the true costs are unknown and must be predicted.
Since the LP can be regarded as a continuous problem (it



_df_ ( _x_ ( _θ_ [ˆ] ) _, θ_ )



( _θ_ [ˆ] ) _, θ_ ) = _[d][f]_ [(] _[x]_ [(] _[θ]_ [ˆ][)] _[,][ θ]_ [)]

_dω_ [ˆ]



_dx_ ( _θ_ [ˆ] )

_dθ_ [ˆ]



_dθ_ [ˆ]

_dω_



_dx_ ( _θ_ [ˆ] )



The first term is just the gradient of the objective with
respect to the decision variable _x_, and the last term is the
gradient of the model’s predictions with respect to its own
internal parameterization.
The key is computing the middle term, which measures
how the optimal decision changes with respect to the prediction _θ_ [ˆ] . For continuous problems, the optimal continuous
decision _x_ must satisfy the KKT conditions (which are sufficient for convex problems). The KKT conditions define a
system of linear equations based on the gradients of the objective and constraints around the optimal point. Is is known
that by applying the implicit function theorem, we can differentiate the solution to this linear system (Gould et al.
2016; Donti, Amos, and Kolter 2017). In more detail, recall that our continuous problem is over _conv_ ( _X_ ), the convex hull of the discrete feasible solutions. This set is a polytope, which can be represented via linear equalities as the
set _{x_ : _Ax ≤_ _b}_ for some matrix _A_ and vector _b_ . Let ( _x, λ_ )
be pair of primal and dual variables which satisfy the KKT
conditions. Then differentiating the conditions yields that



(2)
�



_dx_

_dθ_
_dλ_

�� _dθ_



_∇_ [2] _x_ _[f]_ [(] _[x, θ]_ [)] _A_ _[T]_
� _diag_ ( _λ_ ) _A_ _diag_ ( _Ax −_ _b_ )



_dθ_
_dλ_
_dθ_



_d∇_ _x_ _f_ ( _x,θ_ )
= _dθ_
� � 0


just happens that the optimal solutions in these example domains are integral), we could attempt to apply Equation 2
and differentiate the solution. This approach runs into an immediate difficulty: the optimal solution to an LP may not be
differentiable (or even continuous) with respect to _θ_ . This
is because the optimal solution may “jump” to a different
vertex. Formally, the left-hand side matrix in Equation 2 becomes singular since _∇_ [2] _x_ _[f]_ [(] _[x, θ]_ [)][ is always zero. We resolve]
this challenge by instead solving the regularized problem


max _θ_ _[T]_ _x −_ _γ||x||_ 2 [2] [s.t.] _[ Ax]_ [ =] _[ b, Gx][ ≤]_ _[h]_ (4)


which introduces a penalty proportional to the squared
norm of the decision vector. This transforms the LP into a
strongly concave quadratic program (QP). The Hessian is
given by _∇_ [2] _x_ _[f]_ [(] _[x, θ]_ [) =] _[ −]_ [2] _[γI]_ [ (where] _[ I]_ [ is the identity ma-]
trix), which renders the solution differentiable under mild
conditions (see supplement for proof):


**Theorem 1.** _Let x_ ( _θ_ ) _denote the optimal solution of Prob-_
_lem 4. Provided that the problem is feasible and all rows of A_
_are linearly independent, x_ ( _θ_ ) _is differentiable with respect_
_to θ almost everywhere. If A has linearly dependent rows,_
_removing these rows yields an equivalent problem which is_
_differentiable almost everywhere. Wherever x_ ( _θ_ ) _is differen-_
_tiable, it satisfies the conditions in Equation 2._


Moreover, we can control the loss that regularization can
cause on the original, linear problem:

**Theorem 2.** _Define D_ = max _x,y∈conv_ ( _X_ ) _||x −_ _y||_ [2] _as the_
_squared diameter of the feasible set and OPT to be the op-_
_timal value for Problem 3. We have θ_ _[⊤]_ _x_ ( _θ_ ) _≥_ _OPT −_ _γD._


Together, these results give us a differentiable surrogate
which still enjoys an approximation guarantee relative to the
integral problem. Computing the backward pass via Equation 2 is now straightforward since all the relevant terms are
easily available. Since _∇_ _x_ _θ_ _[⊤]_ _x_ = _θ_, we have _[d][∇]_ _[x]_ _dθ_ _[f]_ [(] _[x][,][θ]_ [)] = _I_ .

All other terms are easily computed from the optimal primaldual pair ( _x, λ_ ) which is output by standard QP solvers. We
can also leverage a recent QP solver (Amos and Kolter 2017)
which maintains a factorization of the KKT matrix for a
faster backward pass. At test time, we simply set _γ_ = 0
to produce an integral decision.


**Submodular maximization**


We consider problems where the underlying objective to
maximize a set function _f_ : 2 _[V]_ _→_ _R_, where _V_ is a ground
set of items. A set function is _submodular_ if for any _A ⊆_ _B_
and any _v ∈_ _V \B_, _f_ ( _A∪{v}_ ) _−f_ ( _A_ ) _≥_ _f_ ( _B∪{v}_ ) _−f_ ( _B_ ).
We will restrict our consideration to submodular functions
which are _monotone_ ( _f_ ( _A ∪{v}_ ) _−_ _f_ ( _A_ ) _≥_ 0 _∀A, v_ ) and
_normalized f_ ( _∅_ ) = 0. This class of functions contains
many combinatorial problems which have been considered
in machine learning and artificial intelligence (e.g., influence maximization, facility location, diverse subset selection, etc.). We focus on the cardinality-constrained optimization problem max _|S|≤k_ _f_ ( _S_ ), though our framework easily
accommodates more general matroid constraints.



**Continuous relaxation:** We employ the canonical continuous relaxation for submodular set functions, which associates each set function _f_ with its _multilinear extension_
_F_ (Calinescu et al. 2011). We can view a set function as
defined on the domain _{_ 0 _,_ 1 _}_ _[|][V][ |]_, where each element is an
indicator vector which the items contained in the set. The extension _F_ is a continuous function defined on the hypercube

[0 _,_ 1] _[|][V][ |]_ . We interpret a given fraction vector _x ∈_ [0 _,_ 1] _[|][V][ |]_
as giving the marginal probability that each item is included
in the set. _F_ ( _x_ ) is the expected value of _f_ ( _S_ ) when each
item _i_ is included in _S_ independently with probability _x_ _i_ .
In other words, _F_ ( _x_ ) = [�] _S⊆V_ _[f]_ [(] _[S]_ [)][ �] _i∈S_ _[x]_ _[i]_ � _i̸∈S_ [1] _[ −]_ _[x]_ _[i]_ [.]

While this definition sums over exponentially many terms,
arbitrarily close approximations can be obtained via random sampling. Further, closed forms are available for many
cases of interest (Iyer, Jegelka, and Bilmes 2014). Importantly, well-known rounding algorithms (Calinescu et al.
2011) can convert a fractional point _x_ to a set _S_ satisfying
E[ _f_ ( _S_ )] _≥_ _F_ ( _x_ ); i.e., the rounding is lossless.
As a proxy for the discrete problem max _|S|≤k_ _f_ ( _S_ ), we
can instead solve max _x∈conv_ ( _X_ ) _F_ ( _x_ ), where _X_ = _{x ∈_
_{_ 0 _,_ 1 _}_ _[|][V][ |]_ : [�] _i_ _[x]_ _[i]_ _[ ≤]_ _[k][}]_ [. Unfortunately,] _[ F]_ [ is not in general]

concave. Nevertheless, many first-order algorithms still obtain a constant factor approximation. For instance, a variant
of the Frank-Wolfe algorithm solves the continuous maximization problem with the optimal approximation ratio of
(1 _−_ 1 _/e_ ) (Calinescu et al. 2011; Bian et al. 2017).
However, non-concavity complicates the problem of differentiating through the continuous optimization problem.
Any polynomial-time algorithm can only be guaranteed to
output a _local_ optimum, which need not be unique (compared to strongly convex problems, where there is a single
global optimum). Consequently, the algorithm used to select _x_ ( _θ_ ) might return a _different_ local optimum under an
infinitesimal change to _θ_ . For instance, the Frank-Wolfe algorithm (the most common algorithm for continuous submodular maximization) solves a linear optimization problem
at each step. Since (as noted above), the solution to a linear
problem may be discontinuous in _θ_, this could render the
output of the optimization problem nondifferentiable.
We resolve this difficulty through a careful choice of optimization algorithm for the forward pass. Specifically, we use
apply projected stochastic gradient ascent (SGA), which has
recently been shown to obtain a [1] 2 [-approximation for con-]

tinuous submodular maximization (Hassani, Soltanolkotabi,
and Karbasi 2017). Although SGA is only guaranteed to find
a local optimum, each iteration applies purely differentiable
computations (a gradient step and projection onto the set
_conv_ ( _X_ )), and so the final output after _T_ iterations will be
differentiable as well. Provided that _T_ is sufficiently large,
this output will converge to a local optimum, which must
satisfy the KKT conditions. Hence, we can apply our general approach to the local optimum returned by SGA. The
following theorem shows that the local optima of the multilinear extension are differentiable:


**Theorem 3.** _Suppose that x_ _[∗]_ _is a local maximum of the mul-_
_tilinear extension, i.e,., ∇_ _x_ _F_ ( _x_ _[∗]_ _, θ_ ) = 0 _and ∇_ [2] _x_ _[F]_ [(] _[x]_ _[∗]_ _[, θ]_ [)] _[ ≻]_
0 _. Then, there exists a neighborhood I around x_ _[∗]_ _such that_



_S⊆V_ _[f]_ [(] _[S]_ [)][ �]



_i∈S_ _[x]_ _[i]_ �


_the maximizer of F_ ( _·, θ_ ) _within I∩conv_ ( _X_ ) _is differentiable_
_almost everywhere as a function of θ, with_ _[dx]_ _dθ_ [(] _[θ]_ [)] _satisfying_

_the conditions in Equation 2._

_̸_

We remark that Theorem 3 requires a local maximum,
while gradient ascent may in theory find saddle points. How- _̸_
ever, recent work shows that random perturbations ensure
that gradient ascent quickly escapes saddle points and finds
an approximate local optimum (Jin et al. 2017).
**Efficient backward pass:** We now show how the terms
needed to compute gradients via Equation 2 can be efficiently obtained. In particular, we need access to the optimal dual variable _λ_ as well as the term _[d][∇]_ _[x]_ _[F]_ [(] _[x][,][θ]_ [)] . These

_dθ_
were easy to obtain in the LP setting but the submodular
setting requires some additional analysis. Nevertheless, we
show that both can be obtained efficiently.
**Optimal dual variables:** SGA only produces the optimal primal variable _x_, not the corresponding dual variable _λ_
which is required to solve Equation 2 in the backward pass.
We show that for cardinality-constrained problems, we can
obtain the optimal dual variables analytically given a primal
solution _x_ . Let _λ_ _[L]_ _i_ [be the dual variable associated with the]
constraint _x_ _i_ _≥_ 0, _λ_ _[U]_ _i_ [with] _[ x]_ _[i]_ _[ ≤]_ [1][ and] _[ λ]_ _[S]_ [ with][ �] _i_ _[x]_ _[i]_ _[ ≤]_ _[k]_ [.]

By differentiating the Lagrangian, any optimum satisfies


_∇_ _x_ _i_ _f_ ( _x_ ) _−_ _λ_ _[L]_ _i_ [+] _[ λ]_ _[U]_ _i_ [+] _[ λ]_ _[S]_ _i_ [= 0] _∀i_


where complementary slackness requires that _λ_ _[L]_ _i_ [= 0][ if]
_x_ _i_ _>_ 0 and _λ_ _[U]_ _i_ = 0 if _x_ _i_ _<_ 1. Further, it is easy to see
that for all _i_ with 0 _< x_ _i_ _<_ 1, _∇_ _x_ _i_ _f_ ( _x_ ) must be equal.
Otherwise, _x_ could not be (locally) optimal since we could
increase the objective by finding a pair _i, j_ with _∇_ _x_ _i_ _f_ ( _x_ ) _>_
_∇_ _x_ _j_ _f_ ( _x_ ), increasing _x_ _i_, and decreasing _x_ _j_ . Let _∇_ _∗_ denote
the shared gradient value for fractional entries. We can solve
the above equation and express the optimal dual variables as


_λ_ _[S]_ = _−∇_ _∗_ _,_ _λ_ _[L]_ _i_ [=] _[ λ]_ _[S]_ _[ −∇]_ _[x]_ _i_ _[f,]_ _λ_ _[U]_ _i_ [=] _[ ∇]_ _[x]_ _i_ _[f][ −]_ _[λ]_ _[S]_


where the expressions for _λ_ _[L]_ _i_ [and] _[ λ]_ _[U]_ _i_ [apply only when]
_x_ _i_ = 0 and _x_ _i_ = 1 respectively (otherwise, complementary
slackness requires these variables be set to 0).
**Computing** **dd** _θ_ _[∇]_ **[x]** **[F]** [(] **[x]** _[, θ]_ [)] **[:]** [ We show that this term can]
be obtained in closed form for the case of probabilistic coverage functions, which includes many cases of practical interest (e.g. budget allocation, sensor placement, facility location, etc.). However, our framework can be applied to arbitrary submodular functions; we focus here on coverage functions just because they are particularly common in applications. A coverage function takes the following form. There
a set of items _U_, and each _j ∈_ _U_ has a weight _w_ _j_ . The algorithm can choose from a ground set _V_ of actions. Each
action _a_ _i_ covers each item _j_ independently with probability
_θ_ _ij_ . We consider the case where the probabilities _θ_ are be unknown and must be predicted from data. For such problems,
the multilinear extension has a closed form



_̸_


_̸_


**Experiments**
We conduct experiments across a variety of domains in order to compare our decision-focused learning approach with
traditional two stage methods. We start out by describing
the experimental setup for each domain. Then, we present
results for the complete data-decisions pipeline in each domain (i.e., the final solution quality each method produces
on the optimization problem). We find that decision-focused
learning almost always outperforms two stage approaches.
To investigate this phenomenon, we show more detailed results about what each model learns. Two stage approaches
typically learn predictive models which are more accurate
according to standard measures of machine learning accuracy. However, decision-focused methods learn qualities
which are important for optimization performance even if
this leads to lower accuracy in an overall sense.
**Budget allocation:** We start with a synthetic domain
which allows us to illustrate how our methods differ from
traditional approaches and explore when improved decision
making is achievable. This example concerns budget allocation, a submodular maximization problem which models
an advertiser’s choice of how to divide a finite budget _k_ between a set of channels. There is a set of customers _R_ and
the objective is _f_ ( _S_ ) = [�] _v∈R_ [1] _[ −]_ [�] _u∈S_ [(1] _[ −]_ _[θ]_ _[uv]_ [)][, where]

_θ_ _uv_ is the probability that advertising on channel _u_ will
reach customer _v_ . This is the expected number of customers
reached. Variants on this problem have been the subject of a
great deal of research (Alon, Gamzu, and Tennenholtz 2012;
Soma et al. 2014; Miyauchi et al. 2015).
In our problem, the matrix _θ_ is not known in advance and
must be learned from data. The ground truth matrices were
generated using the Yahoo webscope (Yahoo 2007) dataset
which logs bids placed by advertisers on a set of phrases. In
our problem, the phrases are channels and the accounts are
customers. Each instance samples a random subset of 100
channels and 500 customers. For each edge ( _u, v_ ) present in
the dataset, we sample _θ_ _uv_ uniformly at random in [0,0.2].
For each channel _u_, we generate a feature vector from that
channel’s row of the matrix, _θ_ _u_ via complex nonlinear function. Specifically, _θ_ _u_ is passed through a 5-layer neural network with random weight matrices and ReLU activations to
obtain a feature vector _y_ _u_ . The learning task is to reconstruct
_θ_ _u_ from _y_ _u_ . The optimization task is to select _k_ channels in
order to maximize the number of customers reached.
**Bipartite matching:** This problem occurs in many domains; e.g., bipartite matching has been used to model the
problem of a public housing programs matching housing resources to applicants (Benabbou et al. 2018) or platforms
matching advertisers with users (Bellur and Kulkarni 2007).
In each of these cases, the reward to matching any two nodes
is not initially known, but is instead predicted from the features available for both parties. Bipartite matching can be
formulated as a linear program, allowing us to apply our



and we can obtain the expression


_̸_


_̸_



_d_
_dθ_ _kj_ _∇_ _x_ _i_ _F_ ( _x, θ_ ) = _̸_ _̸_



_−θ_ _ij_ _x_ _k_ � _̸_
�� _k_ = _̸_ _i_ [1] _[ −]_



_−θ_ _ij_ _x_ _k_ � _ℓ_ = _̸_ _i,k_ [1] _[ −]_ _[x]_ _[ℓ]_ _[θ]_ _[ℓj]_ if _k ̸_ = _i_

� _k_ = _̸_ _i_ [1] _[ −]_ _[x]_ _[k]_ _[θ]_ _[kj]_



_̸_

_k_ = _̸_ _i_ [1] _[ −]_ _[x]_ _[k]_ _[θ]_ _[kj]_ otherwise _._



_̸_


_̸_


_F_ ( _x, θ_ ) = � _w_ _j_

_j∈U_



_̸_


_̸_


�



_̸_


_̸_


1 _−_
�



_̸_


_̸_


� 1 _−_ _x_ _ij_ _θ_ _ij_


_i∈V_



_̸_


_̸_


�


Table 1: Solution quality of each method for the full data-decisions pipeline.


Budget allocation Matching Diverse recommendation


_k_ = 5 10 20 _−_ 5 10 20


NN1-Decision **49.18** _±_ **0.24** **72.62** _±_ **0.33** **98.95** _±_ **0.46** 2.50 _±_ 0.56 **15.81** _±_ **0.50** **29.81** _±_ **0.85** **52.43** _±_ **1.23**
NN2-Decision 44.35 _±_ 0.56 67.64 _±_ 0.62 93.59 _±_ 0.77 **6.15** _±_ **0.38** 13.34 _±_ 0.77 26.32 _±_ 1.38 47.79 _±_ 1.96
NN1-2Stage 32.13 _±_ 2.47 45.63 _±_ 3.76 61.88 _±_ 4.10 2.99 _±_ 0.76 4.08 _±_ 0.16 8.42 _±_ 0.29 19.16 _±_ 0.57
NN2-2Stage 9.69 _±_ 0.05 18.93 _±_ 0.10 36.16 _±_ 0.18 3.49 _±_ 0.32 11.63 _±_ 0.43 22.79 _±_ 0.66 42.37 _±_ 1.02
RF-2Stage **48.81** _±_ **0.32** **72.40** _±_ **0.43** **98.82** _±_ **0.63** 3.66 _±_ 0.26 7.71 _±_ 0.18 15.73 _±_ 0.34 31.25 _±_ 0.64
Random 9.69 _±_ 0.04 18.92 _±_ 0.09 36.13 _±_ 0.14 2.45 _±_ 0.64 8.19 _±_ 0.19 16.15 _±_ 0.35 31.68 _±_ 0.71



decision-focused approach. The learning problem is to use
node features to predict whether each edge is present or absent (a classification problem). The optimization problem is
to find a maximum matching in the predicted graph.
Our experiments use the cora dataset (Sen et al. 2008).
The nodes are scientific papers and edges represent citation.
Each node’s feature vector indicating whether each word in
a vocabulary appeared in the paper (there are 1433 such features). The overall graph has 2708 nodes. In order to construct instances for the decision problem, we partitioned the
complete graph into 27 instances, each with 100 nodes, using metis (Karypis and Kumar 1998). We divided the nodes
in each instance into the sides of a bipartite graph (of 50
nodes each) such that the number of edges crossing sides
was maximized. The learning problem is much more challenging than before: unlike in budget allocation, the features
do not contain enough information to reconstruct the citation
network. However, a decision maker may still benefit from
leveraging whatever signal is available.
**Diverse recommendation:** One application of submodular optimization is to select diverse sets of item, e.g. for
recommendation systems or document summarization. Suppose that each item _i_ is associated with a set of topics _t_ ( _i_ ).
Then, we aim to select a set of _k_ items which collectively
cover as many topics as possible: _f_ ( _S_ ) = ��� _i∈S_ _[t]_ [(] _[i]_ [)] ��. Such

formulations have been used across recommendation systems (Ashkan et al. 2015), text summarization (Takamura
and Okumura 2009), web search (Agrawal et al. 2009) and
image segmentation (Prasad, Jegelka, and Batra 2014).
In many applications, the item-topic associations _t_ ( _i_ ) are
not known in advance. Hence, the learning task is to predict a binary matrix _θ_ where _θ_ _ij_ is 1 if item _i_ covers topic
_j_ and 0 otherwise. The optimization task is to find a set of
_k_ items maximizing the number of topics covered according
to _θ_ . We consider a recommendation systems problem based
on the Movielens dataset (GroupLens 2011) in which 2113
users rate 10197 movies (though not every user rated every
movie). The items are the movies, while the topics are the
top 500 actors. In our problem, the movie-actor assignments
are unknown, and must be predicted only from user ratings.
This is a _multilabel classification problem_ where we attempt
to predict which actors are associated with each movie. We
randomly divided the movies into 101 problem instances,
each with 100 movies. The feature matrix _y_ contains the ratings given by each of the 2113 users for the 100 movies in
the instance (with zeros where no rating is present).
**Algorithms and experimental setup:** In each domain,



we randomly divided the instances into 80% training and
20% test. All results are averaged over 30 random splits. Our
decision-focused framework was instantiated using feedforward, fully connected neural networks as the underlying
predictive model. All networks used ReLU activations. We
experimented with networks with 1 layer, representing a restricted class of models, and 2-layer networks, where the
hidden layer (of size 200) gives additional expressive power.
We compared two training methods. First, the decisionfocused approach proposed above. Second, a two stage approach that uses a machine learning loss function (mean
squared error for regression tasks and cross-entropy loss
for classification). _This allows us to isolate the impact of_
_the training method since both use the same underlying ar-_
_chitecture._ We experimented with additional layers but observed little benefit for either method. All networks were
trained using Adam with learning rate 10 _[−]_ [3] . We refer to
the 1-layer decision focused network as _NN1-Decision_ and
the 1-layer two stage network as _NN1-2Stage_ (with analogous names for the 2-layer networks). We also compared to a
random forest ensemble of 100 decisions trees ( _RF-2Stage_ ).
Gradient-based training cannot be applied to random forests,
so benchmark represents a strong predictive model which
can be used by two stage approaches but not by our framework. Lastly, we show performance for a random decision.

**Solution quality:** Table 1 shows the solution quality that
each approaches obtains on the full pipeline; i.e., the objective value of its decision evaluated using the true parameters. Each value is the mean (over the 30 iterations) and
a bootstrapped 95% confidence interval. For the budget allocation and diverse recommendation tasks, we varied the
budget _k_ . The decision-focused methods obtain the highestperformance across the board, tied with random forests on
the synthetic budget allocation task.

We now consider each individual domain, starting with
budget allocation. Both decision-focused methods substantially outperform the two-stage neural networks, obtaining
at least 37% greater objective value. This demonstrates that
with fixed predictive architecture, decision-focused learning can greatly improve solution quality. NN1-Decision performs somewhat better than NN2-Decision, suggesting that
the simpler class of models is easier to train. However, NN12Stage performs significantly worse than NN1-Decision,
indicating that alignment between training and the decision problem is highly important for simple models to succeed. RF-2Stage performs essentially equivalently to NN1Decision. This is potentially surprising since random for

Table 2: Accuracy of each method according to standard measures.

Budget allocation Matching Diverse recommendation


MSE CE AUC CE AUC


NN1-Decision 0.8673e-02 _±_ 1.83e-04 0.994 _±_ 0.002 0.501 _±_ 0.011 1.053 _±_ 0.005 0.593 _±_ 0.003
NN2-Decision 1.7118e-02 _±_ 2.65e-04 0.689 _±_ 0.004 **0.560** _±_ **0.006** 1.004 _±_ 0.022 0.577 _±_ 0.008
NN1-2Stage 0.0501e-02 _±_ 2.67e-06 0.696 _±_ 0.001 0.499 _±_ 0.013 0.703 _±_ 0.001 0.389 _±_ 0.003
NN2-2Stage 0.0530e-02 _±_ 2.27e-06 **0.223** _±_ **0.005** 0.498 _±_ 0.007 0.690 _±_ 0.000 **0.674** _±_ **0.004**
RF-2Stage **0.0354e-02** _±_ **4.17e-06** 0.693 _±_ 0.000 0.500 _±_ 0.000 **0.689** _±_ **0.000** 0.500 _±_ 0.000





















Figure 1: (a) ground truth (b) NN1-2Stage (c) NN1-Decision


est are a much more expressive model class. As we will
see later, much of the random forest’s success is due to the
fact that the features in this synthetic domain are very highsignal; indeed, they suffice for near-perfect reconstruction.
The next two domains, both based on real data, explore lowsignal settings where highly accurate recovery is impossible.
In bipartite matching, NN2-Decision obtains the highest overall performance, making nearly _over 70% more_
_matches_ than the next best method (RF-2Stage, followed
closely by NN2-2Stage). Both 1-layer models perform extremely poorly, indicating that the more complex learning
problem requires a more expressive model class. However,
the highly expressive RF-2Stage does only marginally better
than NN2-2Stage, demonstrating the critical role of aligning
training and decision making.

In the diverse recommendation domain, NN1-Decision
has the best performance, followed closely by NN2Decision. NN2-2Stage trails by 23%, and NN1-2Stage performs extremely poorly. This highlights the importance of
the training method within the same class of models: NN1Decision obtains approximately 2.7 times greater objective
value than NN1-2Stage. RF-2Stage also performs poorly in
this domain, and is seemingly unable to extract any signal
which boosts decision quality above that of random.
**Exploration of learned models:** We start out by showing the accuracy of each method according to standard measures, summarized in Table 2. For classification domains (diverse recommendation, matching), we show cross-entropy
loss (which is directly optimized by the two stage networks)
and AUC. For regression (the budget allocation domain), we



Figure 2: Left: our method’s predicted total out-weight for
each item. Right: predictions from two stage method.


show mean squared error (MSE). For budget allocation and
diverse recommendation, we fixed _k_ = 10.
The two-stage methods are, in almost all cases, significantly more accurate than the decision-focused networks despite their worse solution quality. Moreoever, no accuracy
measure is well-correlated with solution quality. On budget allocation, the two decision-focused networks have the
worst MSE but the best solution quality. On bipartite matching, NN2-2Stage has better cross-entropy loss but much
worse solution quality than NN2-Decision. On diverse recommendation, NN2-2Stage has the best AUC but worse solution quality than either decision-focused network.
This incongruity raises the question of what differentiates
the predictive models learned via decision-focused training. We now show more a more detailed exploration of
each model’s predictions. Due to space constraints, we focus on the simpler case of the synthetic budget allocation
task, comparing NN1-Decision and NN1-2Stage. However,
the higher-level insights generalize across domains (see the
supplement for more detailed visualizations).
Figure 1 shows each model’s predictions on an example
instance. Each heat map shows a predicted matrix _θ_, where
dark entries correspond to a high prediction and light entries
to low. The first matrix is the ground truth. The second matrix is the prediction made by NN1-2Stage, which matches
the overall sparsity of the true _θ_ but fails to recover almost all
of the true connections. The last matrix corresponds to NN1Decision and appears completely dissimilar to the ground
truth. Nevertheless, these seemingly nonsensical predictions
lead to the best quality decisions.
To investigate the connection between predictions and
decision, Figure 2 aggregates each model’s predictions at
the channel level. Formally, we examine the predicted outweight for each channel _u_, i.e., the sum of the row _θ_ _u_ . This
is a coarse measure of _u_ ’s importance for the optimization problem; channels with connections to many customers


are more likely to be good candidates for the optimal set.
Surprisingly, NN1-Decision’s predicted out-weights are extremely well correlated with the ground truth out-weights
( _r_ [2] = 0 _._ 94). However, the absolute magnitude of its predictions are skewed: the bulk of channels have low outweight
(less than 1), but NN1-Decision’s predictions are all at least
13. By contrast NN1-2Stage has poorer correlation, making
it less useful for identifying the outliers which comprise the
optimal set. However, it better matches the values of low
out-weight channels and hence attains better MSE. This illustrates how aligning the model’s training with the optimization problem leads it to focus on qualities which are
specifically important for decision making, even if this compromises accuracy elsewhere.
**Acknowledgments:** This work was supported by the
Army Research Office (MURI W911NF1810208) and a National Science Foundation Graduate Research Fellowship.


**References**


[Agrawal et al. 2009] Agrawal, R.; Gollapudi, S.; Halverson,
A.; and Ieong, S. 2009. Diversifying search results. In
_WSDM_, 5–14. ACM.

[Alon, Gamzu, and Tennenholtz 2012] Alon, N.; Gamzu, I.;
and Tennenholtz, M. 2012. Optimizing budget allocation
among channels and influencers. In _WWW_ .

[Amos and Kolter 2017] Amos, B., and Kolter, J. Z. 2017.
Optnet: Differentiable optimization as a layer in neural networks. In _ICML_ .

[Ashkan et al. 2015] Ashkan, A.; Kveton, B.; Berkovsky, S.;
and Wen, Z. 2015. Optimal greedy diversity for recommendation. In _IJCAI_, 1742–1748.

[Belanger, Yang, and McCallum 2017] Belanger, D.; Yang,
B.; and McCallum, A. 2017. End-to-end learning for structured prediction energy networks. In _ICML_ .

[Bellur and Kulkarni 2007] Bellur, U., and Kulkarni, R.
2007. Improved matchmaking algorithm for semantic web
services based on bipartite graph matching. In _ICWS_, 86–93.
IEEE.

[Benabbou et al. 2018] Benabbou, N.; Chakraborty, M.; Ho,
X.-V.; Sliwinski, J.; and Zick, Y. 2018. Diversity constraints
in public housing allocation. In _AAMAS_, 973–981.

[Bertsimas and Dunn 2017] Bertsimas, D., and Dunn, J.
2017. Optimal classification trees. _Machine Learning_
106(7):1039–1082.

[Beygelzimer and Langford 2009] Beygelzimer, A., and
Langford, J. 2009. The offset tree for learning with partial
labels. In _KDD_ .

[Bian et al. 2017] Bian, A. A.; Mirzasoleiman, B.; Buhmann,
J. M.; and Krause, A. 2017. Guaranteed non-convex optimization: Submodular maximization over continuous do
mains. In _AISTATS_ .

[Calinescu et al. 2011] Calinescu, G.; Chekuri, C.; P´al, M.;
and Vondr´ak, J. 2011. Maximizing a monotone submodular
function subject to a matroid constraint. _SIAM Journal on_
_Computing_ 40(6):1740–1766.




[Djolonga and Krause 2017] Djolonga, J., and Krause, A.
2017. Differentiable learning of submodular models. In
_NIPS_, 1013–1023.

[Domke 2012] Domke, J. 2012. Generic methods for
optimization-based modeling. In _Artificial Intelligence and_
_Statistics_, 318–326.

[Donti, Amos, and Kolter 2017] Donti, P.; Amos, B.; and
Kolter, J. Z. 2017. Task-based end-to-end model learning
in stochastic optimization. In _NIPS_ .

[Elmachtoub and Grigas 2017] Elmachtoub, A. N., and Grigas, P. 2017. Smart “predict, then optimize”. _arXiv preprint_
_arXiv:1710.08005_ .

[Fang et al. 2016] Fang, F.; Nguyen, T. H.; Pickles, R.; Lam,
W. Y.; Clements, G. R.; An, B.; Singh, A.; Tambe, M.;
Lemieux, A.; et al. 2016. Deploying paws: Field optimization of the protection assistant for wildlife security. In _IAAI_,
3966–3973.

[Ford et al. 2015] Ford, B.; Nguyen, T.; Tambe, M.; Sintov,
N.; and Delle Fave, F. 2015. Beware the soothsayer: From
attack prediction accuracy to predictive reliability in security
games. In _International Conference on Decision and Game_
_Theory for Security_ .

[Gould et al. 2016] Gould, S.; Fernando, B.; Cherian, A.;
Anderson, P.; Cruz, R. S.; and Guo, E. 2016. On differentiating parameterized argmin and argmax problems
with application to bi-level optimization. _arXiv preprint_
_arXiv:1607.05447_ .

[GroupLens 2011] GroupLens. 2011. Movielens dataset.

[Hassani, Soltanolkotabi, and Karbasi 2017] Hassani, H.;
Soltanolkotabi, M.; and Karbasi, A. 2017. Gradient
Methods for Submodular Maximization. In _NIPS_ .

[Horvitz and Mitchell 2010] Horvitz, E., and Mitchell, T.
2010. From data to knowledge to action: A global enabler
for the 21st century. _Computing Community Consortium_ 1.

[Horvitz 2010] Horvitz, E. 2010. From data to predictions
and decisions: Enabling evidence-based healthcare. _Com-_
_puting Community Consortium_ 6.

[Iyer, Jegelka, and Bilmes 2014] Iyer, R. K.; Jegelka, S.; and
Bilmes, J. A. 2014. Monotone closure of relaxed constraints
in submodular optimization. In _UAI_ .

[Jin et al. 2017] Jin, C.; Ge, R.; Netrapalli, P.; Kakade, S. M.;
and Jordan, M. I. 2017. How to escape saddle points efficiently. In _ICML_ .

[Karypis and Kumar 1998] Karypis, G., and Kumar, V. 1998.
A fast and high quality multilevel scheme for partitioning
irregular graphs. _SIAM Journal on Scientific Computing_
20(1):359–392.

[Kempe, Kleinberg, and Tardos 2003] Kempe, D.; Kleinberg, J.; and Tardos, E. 2003. Maximizing the Spread of [´]
Influence Through a Social Network. In _KDD_ .

[Khalil et al. 2017a] Khalil, E. B.; Dai, H.; Zhang, Y.; Dilkina, B.; and Song, L. 2017a. Learning combinatorial optimization algorithms over graphs. In _NIPS_ .

[Khalil et al. 2017b] Khalil, E. B.; Dilkina, B.; Nemhauser,


G. L.; Ahmed, S.; and Shao, Y. 2017b. Learning to run
heuristics in tree search. In _IJCAI_ .

[Korte et al. 2012] Korte, B.; Vygen, J.; Korte, B.; and Vygen, J. 2012. _Combinatorial optimization_, volume 2.
Springer.

[Miyauchi et al. 2015] Miyauchi, A.; Iwamasa, Y.; Fukunaga, T.; and Kakimura, N. 2015. Threshold influence model
for allocating advertising budgets. In _ICML_, 1395–1404.

[Mukhopadhyay et al. 2017] Mukhopadhyay, A.; Vorobeychik, Y.; Dubey, A.; and Biswas, G. 2017. Prioritized allocation of emergency responders based on a continuous-time
incident prediction model. In _AAMAS_, 168–177.

[Niculae et al. 2018] Niculae, V.; Martins, A. F.; Blondel,
M.; and Cardie, C. 2018. Sparsemap: Differentiable sparse
structured inference. In _ICML_ .

[Prasad, Jegelka, and Batra 2014] Prasad, A.; Jegelka, S.;
and Batra, D. 2014. Submodular meets structured: Finding
diverse subsets in exponentially-large structured item sets.
In _NIPS_ .

[Sen et al. 2008] Sen, P.; Namata, G.; Bilgic, M.; Getoor, L.;
Galligher, B.; and Eliassi-Rad, T. 2008. Collective classification in network data. _AI magazine_ 29(3):93.

[Soma et al. 2014] Soma, T.; Kakimura, N.; Inaba, K.; and
Kawarabayashi, K.-i. 2014. Optimal budget allocation: Theoretical guarantee and efficient algorithm. In _ICML_, 351–
359.

[Takamura and Okumura 2009] Takamura, H., and Okumura, M. 2009. Text summarization model based on maximum coverage problem and its variant. In _EACL_ .

[Tschiatschek, Sahin, and Krause 2018] Tschiatschek, S.;
Sahin, A.; and Krause, A. 2018. Differentiable submodular
maximization. In _IJCAI_ .

[Tu and Gimpel 2018] Tu, L., and Gimpel, K. 2018. Learning approximate inference networks for structured prediction. In _ICLR_ .

[Viappiani and Boutilier 2010] Viappiani, P., and Boutilier,
C. 2010. Optimal bayesian recommendation sets and myopically optimal choice query sets. In _NIPS_ .

[Vinyals, Fortunato, and Jaitly 2015] Vinyals, O.; Fortunato,
M.; and Jaitly, N. 2015. Pointer networks. In _NIPS_ .

[Wang et al. 2006] Wang, H.; Xie, H.; Qiu, L.; Yang, Y. R.;
Zhang, Y.; and Greenberg, A. 2006. Cope: traffic engineering in dynamic networks. In _Sigcomm_, volume 6, 194.

[Xue et al. 2016] Xue, Y.; Davies, I.; Fink, D.; Wood, C.; and
Gomes, C. P. 2016. Avicaching: A two stage game for bias
reduction in citizen science. In _AAMAS_, 776–785.

[Yahoo 2007] Yahoo. 2007. Yahoo! webscope dataset ydataysm-advertiser-bids-v1 0. http://research.yahoo.
com/Academic_Relations.


**Proofs**


_Proof of Theorem 1._ We start with the case where all rows
of _A_ are linearly independent. Here, the result follows easily from Theorem 1 of (Amos and Kolter 2017) since the



Hessian matrix is _γI_ and hence guaranteed to be positive
definite.
When _A_ has linearly dependent rows, we argue that these
rows can be removed without changing the feasible region.
Consider two rows _a_ _i_ and _a_ _j_ such that for all _x_, _a_ _[⊤]_ _i_ _[x]_ [ =]
_ca_ _[⊤]_ _j_ _[x]_ [ for some scalar] _[ c]_ [. We are guaranteed that the prob-]
lem is feasible, meaning that there exists an _x_ which satisfies both constraints simultaneously. For this _x_, we have
_a_ _[⊤]_ _i_ _[x]_ [ =] _[ b]_ _[i]_ [ and] _[ a]_ _[⊤]_ _j_ _[x]_ [ =] _[ b]_ _[j]_ [. But since] _[ a]_ _[⊤]_ _i_ _[x]_ [ =] _[ ca]_ _[⊤]_ _j_ _[x]_ [, we must]
have _b_ _i_ = _cb_ _j_ . Accordingly, constraint _i_ is satisfied if and
only if constraint _j_ is satisfied, and so removing one of the
constraints leaves the feasible set unchanged. Applying this
argument inductively yields the theorem.


_Proof of Theorem 2._ Let _x_ _max_ = arg max _y∈conv_ ( _X_ ) _||y||_ [2] .
We have that


_θ_ _[⊤]_ _x_ ( _θ_ ) = max � _θ_ _[⊤]_ _y −_ _γ||y||_ [2] [�] + _||x_ ( _θ_ ) _||_ [2]
_y_

_≥_ max � _θ_ _[⊤]_ _y_ � _−_ _γ||x_ _max_ _||_ [2] + _γ||x_ ( _θ_ ) _||_ [2]
_y_

= max � _θ_ _[⊤]_ _y_ � + _γ_ � _||x_ ( _θ_ ) _||_ [2] _−||x_ _max_ _||_ [2] [�]
_y_

_≥_ _OPT −_ _γ||x_ ( _θ_ ) _−_ _x_ _max_ _||_ [2]

_≥_ _OPT −_ _γD_


where the second inequality uses the reverse triangle inequality.



_Proof of Theorem 3._ Since _X_ = _{x ∈{_ 0 _,_ 1 _}_ _[|][V][ |]_ : [�]



_Proof of Theorem 3._ Since _X_ = _{x ∈{_ 0 _,_ 1 _}_ : [�] _i_ _[x]_ _[i]_ _[ ≤]_

_k}_, _conv_ ( _X_ ) is described by the two inequality constraints
_−Ix ≤_ 0 and 1 _[⊤]_ _x ≤_ _k_ . It is easy to see that the corresponding constraint matrix _A_ has full row rank. Even though _F_
is not concave, any stationary point ( _x, λ_ ) must satisfy the
KKT conditions. By applying the implicit function theorem
to differentiate these equations, we get the form

_∇_ [2] _x_ _[F]_ [(] _[x, θ]_ [)] _A_ _[T]_ _dxdθ_ = _d∇_ _x_ _dθf_ ( _x,θ_ )
� _diag_ ( _λ_ ) _A_ _diag_ ( _Ax −_ _b_ )�� _dλdθ_ � � 0 �


So long as the right hand side matrix is invertible almost everywhere, the implicit function theorem guarantees
that _dxdθ_ [exists in a neighborhood of] _[ x]_ [ and satisfies the]
above conditions. Note that at a local maximum, we have
_∇_ [2] _x_ _[F]_ [(] _[x, θ]_ [)] _[ ≻]_ [0][, implying that the Hessian matrix must be]
invertible. Accordingly, it is easy to show that the RHS matrix is nonsingular by applying the same logic as (Amos and
Kolter 2017) (Theorem 1).


**Visualizations**

We now show more detailed analysis of the predictions made
by each model in the other two domains: diverse recommendation and bipartite matching. The general trends are similar to those observed in the main paper for budget allocation (although the results are somewhat messier for the realdata domains). We see that the decision-focused neural network makes apparently nonsensical predications. However,
the out-weight that it predicts for each item is better correlated with the ground truth than for the two stage method.



_dx_

_dθ_
_dλ_

�� _dθ_



_dθ_
_dλ_
_dθ_



_d∇_ _x_ _f_ ( _x,θ_ )
= _dθ_
� � 0



_dθ_
0



�


Figure 3: Diverse recommendation predictions. Top to
bottom: ground truth, our method’s prediction (by NN2Decision), two stage prediction (by NN2-2Stage)










|10.0<br>th|r2 = 0.34<br>50 300 350 400|
|---|---|
|0.0<br>2.5<br>5.0<br>7.5<br>Ground tru|0.0<br>2.5<br>5.0<br>7.5<br>Ground tru|



Figure 4: Diverse recommendation predicted outweight according to NN2-Decision (right) and NN2-2Stage (left).


Figure 5: Bipartite matching predictions. Left to right:
ground truth adjacency matrix, our method’s prediction
(NN2-Decision), two stage prediction (NN2-2Stage).











Figure 6: Bipartite matching predicted outweight according
to NN2-Decision (right) and NN2-2Stage (left).


