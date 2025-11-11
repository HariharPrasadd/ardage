## **Zeroth-Order Online Alternating Direction Method of Multipliers:** **Convergence Analysis and Applications**

**Sijia Liu** **Jie Chen** **Pin-Yu Chen** **Alfred O. Hero**
University of Michigan Northwestern Polytechnical IBM Research, University of Michigan
IBM Research, Cambridge University, China Yorktown Heights



**Abstract**



In this paper, we design and analyze a
new zeroth-order online algorithm, namely,
the zeroth-order online alternating direction
method of multipliers (ZOO-ADMM), which
enjoys dual advantages of being gradientfree operation and employing the ADMM to
accommodate complex structured regularizers. Compared to the first-order gradientbased online algorithm, we show that ZOOADMM requires _[√]_ ~~_m_~~ times more iterations,
leading to a convergence rate of _O_ ( _[√]_ ~~_m_~~ _/√T_ ),

where _m_ is the number of optimization
variables, and _T_ is the number of iterations. To accelerate ZOO-ADMM, we propose two minibatch strategies: gradient sample averaging and observation averaging, resulting in an improved convergence rate of
_O_ (�1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the mini


1 + _q_ _[−]_ [1] _m/√_



_O_ (�1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the mini
batch size. In addition to convergence analysis, we also demonstrate ZOO-ADMM to applications in signal processing, statistics, and
machine learning.



**1** **Introduction**


Online convex optimization (OCO) performs sequential inference in a data-driven adaptive fashion, and
has found a wide range of applications (Hall and Willett, 2015; Hazan, 2016; Hosseini et al., 2016). In this
paper, we focus on regularized convex optimization in
the OCO setting, where a cumulative empirical loss is
minimized together with a fixed regularization term.
Regularized loss minimization is a common learning
paradigm, which has been very effective in promotion
of sparsity through _ℓ_ 1 or mixed _ℓ_ 1 / _ℓ_ 2 regularization


Proceedings of the 21 _[st]_ International Conference on Artificial Intelligence and Statistics (AISTATS) 2018, Lanzarote, Spain. PMLR: Volume 84. Copyright 2018 by the
author(s).



(Bach et al., 2012), low-rank matrix completion via
nuclear norm regularization (Cand`es and Recht, 2009),
graph signal recovery via graph Laplacian regularization (Chen and Liu, 2017), and constrained optimization by imposing indicator functions of constraint sets
(Parikh and Boyd, 2014).


Several OCO algorithms have been proposed for regularized optimization, e.g., composite mirror descent,
namely, proximal stochastic gradient descent (Duchi
et al., 2010), regularized dual averaging (Xiao, 2010),
and adaptive gradient descent (Duchi et al., 2011).
However, the complexity of the aforementioned algorithms is dominated by the computation of the proximal operation with respect to the regularizers (Parikh
and Boyd, 2014). An alternative is to use online alternating direction method of multipliers (O-ADMM)
(Ouyang et al., 2013; Suzuki, 2013; Wang and Banerjee, 2013). Different from the algorithms in (Duchi
et al., 2010, 2011; Xiao, 2010), the ADMM framework offers the possibility of splitting the optimization
problem into a sequence of easily-solved subproblems.
It was shown in (Ouyang et al., 2013; Suzuki, 2013;
Wang and Banerjee, 2013) that the online variant of
ADMM has convergence rate of _O_ (1 _/√T_ ) for convex

loss functions and _O_ (log _T/T_ ) for strongly convex loss
functions, where _T_ is the number of iterations.


One limitation of existing O-ADMM algorithms is the
need to compute and repeatedly evaluate the gradient
of the loss function over the iterations. In many practical scenarios, an explicit expression for the gradient
is difficult to obtain. For example, in bandit optimization (Agarwal et al., 2010), a player receives partial
feedback in terms of loss function values revealed by
her adversary, and making it impossible to compute
the gradient of the full loss function. In adversarial
black-box machine learning models, only the function
values (e.g., prediction results) are provided (Chen
et al., 2017). Moreover, in some high dimensional settings, acquiring the gradient information may be difficult, e.g., involving matrix inversion (Boyd and Vandenberghe, 2004). This motivates the development of
gradient-free (zeroth-order) optimization algorithms.


**ZOO-ADMM: Convergence Analysis and Applications**



Zeroth-order optimization approximates the full gradient via a randomized gradient estimate (Agarwal
et al., 2010; Duchi et al., 2015; Ghadimi and Lan,
2013; Hajinezhad et al., 2017; Nesterov and Spokoiny,
2015; Shamir, 2017). For example, in (Agarwal et al.,
2010; Shamir, 2017), zeroth-order algorithms were developed for bandit convex optimization with multipoint bandit feedback. In (Nesterov and Spokoiny,
2015), a zeroth-order gradient descent algorithm was
proposed that has _O_ ( _m/√T_ ) convergence rate, where

_m_ is the number of variables in the objective function.
A similar convergence rate was found in (Ghadimi and
Lan, 2013) for nonconvex optimization. This slowdown (proportional to the problem size _m_ ) in convergence rate was further improved to _O_ ( _[√]_ ~~_m_~~ _/√T_ )

(Duchi et al., 2015), whose optimality was proved under the framework of mirror descent algorithms. A
more recent relevant paper is (Gao et al., 2017), where
a variant of the ADMM algorithm that uses gradient
estimation was introduced. However, the ADMM algorithm presented in (Gao et al., 2017) was not customized for OCO. Furthermore, it only ensured that
the linear equality constraints are satisfied in expectation; hence, a particular instance of the proposed
solution could violate the constraints.



In this paper, we propose a zeroth-order online
ADMM (called ZOO-ADMM) algorithm, and analyze
its convergence rate under different settings, including
stochastic optimization, learning with strongly convex
loss functions, and minibatch strategies for convergence acceleration. We summarize our contributions
as follows.


_•_ We integrate the idea of zeroth-order optimization
with online ADMM, leading to a new gradient-free
OCO algorithm, ZOO-ADMM.


_•_ We prove ZOO-ADMM yields a _O_ ( _[√]_ ~~_m_~~ _/√T_ ) con
vergence rate for smooth+nonsmooth composite objective functions.


_•_ We introduce a general hybrid minibatch strategy for
acceleration of ZOO-ADMM, leading to an improved
convergence rate _O_ (�1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the



where **x** _∈_ R _[m]_ and **y** _∈_ R _[d]_ are optimization variables,
_X_ and _Y_ are closed convex sets, _f_ ( _·_ ; **w** _t_ ) is a convex
and smooth cost/loss function parameterized by **w** _t_ at
time _t_, _φ_ is a convex regularization function (possibly
nonsmooth), and **A** _∈_ R _[l][×][m]_, **B** _∈_ R _[l][×][d]_, and **c** _∈_ R _[l]_

are appropriate coefficients associated with a system
of _l_ linear constraints.


In problem (21), the use of time-varying cost functions _{f_ ( **x** ; **w** _t_ ) _}_ _[T]_ _t_ =1 [captures possibly time-varying en-]
vironmental uncertainties that may exist in the online
setting (Hazan, 2016; Shalev-Shwartz, 2012). We can
also write the online cost as _f_ _t_ ( **x** ) when it cannot be
explicitly parameterized by **w** _t_ . One interpretation of
_{f_ ( **x** ; **w** _t_ ) _}_ _[T]_ _t_ =1 [is the empirical approximation to the]
stochastic objective function E **w** _∼P_ [ _f_ ( **x** ; **w** )]. Here _P_
is an empirical distribution with density [�] _t_ _[δ]_ [(] **[w]** _[,]_ **[ w]** _[t]_ [),]

where _{_ **w** _t_ _}_ _[T]_ _t_ =1 [is a set of i.i.d. samples, and] _[ δ]_ [(] _[·][,]_ **[ w]** _[t]_ [)]
is the Dirac delta function at **w** _t_ . We also note that
when _Y_ = _X_, _l_ = _m_, **A** = **I** _m_, **B** = _−_ **I** _m_, **c** = **0** _m_,
the variable **y** and the linear constraint in (21) can be
eliminated, leading to a standard OCO formulation.
Here **I** _m_ denotes the _m × m_ identity matrix, and **0** _m_
is the _m ×_ 1 vector of all zeros [1] .


**2.1** **Background on O-ADMM**


O-ADMM (Ouyang et al., 2013; Suzuki, 2013; Wang
and Banerjee, 2013) was originally proposed to extend
batch-type ADMM methods to the OCO setting. For
solving (21), a widely-used algorithm was developed by
(Suzuki, 2013), which combines online proximal gradient descent and ADMM in the following form:



**x** _t_ +1 = arg min
**x** _∈X_



� **g** _t_ _[T]_ **[x]** _[ −]_ _**[λ]**_ _[T]_ _t_ [(] **[Ax]** [ +] **[ By]** _[t]_ _[−]_ **[c]** [)]




_[ρ]_ 2 [+] [1]

2 _[∥]_ **[Ax]** [ +] **[ By]** _[t]_ _[ −]_ **[c]** _[∥]_ [2] 2



+ _[ρ]_



2 _η_ _t_ _∥_ **x** _−_ **x** _t_ _∥_ [2] **G** _t_



_,_ (2)
�



**y** _t_ +1 = arg min
**y** _∈Y_



� _φ_ ( **y** ) _−_ _**λ**_ _[T]_ _t_ [(] **[Ax]** _[t]_ [+1] [+] **[ By]** _[ −]_ **[c]** [)]



+ _[ρ]_ 2 _,_ (3)

2 _[∥]_ **[Ax]** _[t]_ [+1] [ +] **[ By]** _[ −]_ **[c]** _[∥]_ [2] �



1 + _q_ _[−]_ [1] _m/√_



convergence rate _O_ (�1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the

minibatch size.




_•_ We illustrate the practical utility of ZOO-ADMM in
machine leanring, signal processing and statistics.


**2** **ADMM: from First to Zeroth Order**


In this paper, we consider the regularized loss minimization problem over a time horizon of length _T_



1
minimize
**x** _∈X_ _,_ **y** _∈Y_ _T_



_T_
� _f_ ( **x** ; **w** _t_ ) + _φ_ ( **y** )


_t_ =1



(1)



_**λ**_ _t_ +1 = _**λ**_ _t_ _−_ _ρ_ ( **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** ) _,_ (4)


where _t_ is the iteration number (possibly the same as
the time step), **g** _t_ is the gradient of the cost function
_f_ ( **x** ; **w** _t_ ) at **x** _t_, namely, **g** _t_ = _∇_ **x** _f_ ( **x** ; **w** _t_ ) _|_ **x** = **x** _t_, _**λ**_ _t_ is a
Lagrange multiplier (also known as the dual variable),
_ρ_ is a positive weight to penalize the augmented term
associated with the equality constraint of (21), _∥· ∥_ 2
denotes the _ℓ_ 2 norm, _η_ _t_ is a non-increasing sequence of
positive step sizes, and _∥_ **x** _−_ **x** _t_ _∥_ [2] **G** _t_ [= (] **[x]** _[−]_ **[x]** _[t]_ [)] _[T]_ **[ G]** _[t]_ [(] **[x]** _[−]_
**x** _t_ ) is a Bregman divergence generated by the strongly
convex function (1 _/_ 2) **x** _[T]_ **G** _t_ **x** with a known symmetric
positive definite coefficient matrix **G** _t_ .


1
In the sequel we will omit the dimension index _m_,
which can be inferred from the context.



subject to **Ax** + **By** = **c** _,_


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**



1

_T_
�



_T_
� ( _f_ ( **x** _t_ ; **w** _t_ ) + _φ_ ( **y** _t_ ))


_t_ =1



Similar to batch-type ADMM algorithms, the subproblem in (23) is often easily solved via the proximal
operator with respect to _φ_ (Boyd et al., 2011). However, one limitation of O-ADMM is that it requires the
gradient **g** _t_ in (2). We will develop the gradient-free
(zeroth-order) O-ADMM algorithm below that relaxes
this requirement.


**2.2** **Motivation of ZOO-ADMM**


To avoid explicit gradient calculations in (2), we adopt
a random gradient estimator to estimate the gradient
of a smooth cost function (Duchi et al., 2015; Ghadimi
and Lan, 2013; Nesterov and Spokoiny, 2015; Shamir,
2017). The gradient estimate of _f_ ( **w** ; **w** _t_ ) is given by


ˆ _[β]_ _[t]_ **[z]** _[t]_ [;] **[ w]** _[t]_ [)] _[ −]_ _[f]_ [(] **[x]** _[t]_ [;] **[ w]** _[t]_ [)]
**g** _t_ = _[f]_ [(] **[x]** _[t]_ [ +] **z** _t_ _,_ (5)

_β_ _t_


where **z** _t_ _∈_ R _[m]_ is a random vector drawn independently at each iteration _t_ from a distribution **z** _∼_ _µ_
with E _µ_ [ **zz** _[T]_ ] = **I**, and _{β_ _t_ _}_ is a non-increasing sequence of small positive smoothing constants. Here
for notational simplicity we replace _{}_ _[T]_ _t_ =1 [with] _[ {}]_ [. The]
rationale behind the estimator (5) is that ˆ **g** _t_ becomes
an unbiased estimator of **g** _t_ when the smoothing parameter _β_ _t_ approaches zero (Duchi et al., 2015).


After replacing **g** _t_ with ˆ **g** _t_ in (5), the resulting algorithm (2)-(24) can be implemented without explicit
gradient computation. This extension is called zerothorder O-ADMM (ZOO-ADMM) that involves a modification of step (2) :



2016)


Regret _T_ ( **x** _t_ _,_ **y** _t_ _,_ **x** _[∗]_ _,_ **y** _[∗]_ ) :=E



�



_−_ [1]

_T_



_T_
�



� ( _f_ ( **x** _[∗]_ ; **w** _t_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



_,_ (8)



**x** _t_ +1 = arg min
**x** _∈X_



ˆ _T_
� **g** _t_ **[x]** _[ −]_ _**[λ]**_ _[T]_ _t_ [(] **[Ax]** [ +] **[ By]** _[t]_ _[−]_ **[c]** [)]



where ( **x** _[∗]_ _,_ **y** _[∗]_ ) denotes the best batch offline solution.


**3** **Algorithm and Convergence**
**Analysis of ZOO-ADMM**


In this section, we begin by stating assumptions used
in our analysis. We then formally define the ZOOADMM algorithm and derive its convergence rate.


We assume the following conditions in our analysis.

_• Assumption A:_ In problem (21), _X_ and _Y_ are
bounded with finite diameter _R_, and at least one of
**A** and **B** in **Ax** + **By** = **c** is invertible.


_• Assumption B:_ _f_ ( _·_ ; **w** _t_ ) is convex and Lipschitz continuous with �E[ _∥∇_ **x** _f_ ( **x** ; **w** _t_ ) _∥_ [2] 2 []] _[ ≤]_ _[L]_ [1] [ for all] _[ t]_ [ and]

**x** _∈X_ .


_•_ Assumption C: _f_ ( _·_ ; **w** _t_ ) is _L_ _g_ ( **w** _t_ )-smooth with _L_ _g_ =
�E[( _L_ _g_ ( **w** _t_ ) [2] )].


_• Assumption D: φ_ is convex and _L_ 2 -Lipschitz continuous with _∥∂φ_ ( **y** ) _∥_ 2 _≤_ _L_ 2 for all **y** _∈Y_, where _∂φ_ ( **y** )
denotes the subgradient of _φ_ .


_• Assumption E:_ In (5), given **z** _∼_ _µ_, the quantity
_M_ ( _µ_ ) := ~~�~~ E[ _∥_ **z** _∥_ [6] 2 [] is finite, and there is a function]

_s_ : N _→_ R + satisfying E[ _∥⟨_ **a** _,_ **z** _⟩_ **z** _∥_ 2 [2] []] _[ ≤]_ _[s]_ [(] _[m]_ [)] _[∥]_ **[a]** _[∥]_ [2] 2 [for]
all **a** _∈_ R _[m]_, where _⟨·, ·⟩_ denotes the inner product of
two vectors.


We remark that Assumptions A-D are standard for
stochastic gradient-based and ADMM-type methods
(Boyd et al., 2011; Hazan, 2016; Shalev-Shwartz, 2012;
Suzuki, 2013). We elaborate on the rationale behind
them in Sec. 8.1. Assumption E places moment constraints on the distribution _µ_ that will allow us to derive the necessary concentration bounds for our convergence analysis. If _µ_ is uniform on the surface of the
Euclidean-ball of radius _[√]_ ~~_m_~~, we have _M_ ( _µ_ ) = _m_ [1] _[.]_ [5]

and _s_ ( _m_ ) = _m_ . And if _µ_ = _N_ ( **0** _,_ **I** _m×m_ ), we have
_M_ ( _µ_ ) _≈_ _m_ [1] _[.]_ [5] and _s_ ( _m_ ) _≈_ _m_ (Duchi et al., 2015). For
ease of representation, we restrict our attention to the
case that _s_ ( _m_ ) = _m_ in the rest of the paper. It is also
worth mentioning that the convex and strongly convex
conditions of _f_ ( _·_ ; **w** _t_ ) can be described as


_f_ ( **x** ; **w** _t_ ) _≥f_ (˜ **x** ; **w** _t_ ) + ( **x** _−_ **x** ˜) _[T]_ _∇_ **x** _f_ (˜ **x** ; **w** _t_ )

+ _[σ]_ (9)

2 _[∥]_ **[x]** _[ −]_ **[x]** [˜] _[∥]_ [2] _[,][ ∀]_ **[x]** _[,]_ [ ˜] **[x]** _[,]_


where _σ ≥_ 0 is a parameter controlling convexity. If




_[ρ]_ 2 [+] [1]

2 _[∥]_ **[Ax]** [ +] **[ By]** _[t]_ _[ −]_ **[c]** _[∥]_ [2] 2



+ _[ρ]_



2 _η_ _t_ _∥_ **x** _−_ **x** _t_ _∥_ [2] **G** _t_



_._ (6)
�



In (22), we can specify the matrix **G** _t_ in such a way
as to cancel the term _∥_ **Ax** _∥_ 2 [2] [. This technique has been]
used in the linearized ADMM algorithms (Parikh and
Boyd, 2014; Zhang et al., 2011) to avoid matrix inversions. Defining **G** _t_ = _α_ **I** _−_ _ρη_ _t_ **A** _[T]_ **A**, the update rule
(22) simplifies to a projection operator



**x** _t_ +1 = arg min
**x** _∈X_



_∥_ **x** _−_ _**ω**_ _∥_ [2] 2 with (7)
� �



_**ω**_ := � _ηα_ _t_ � _−_ **g** ˆ _t_ + **A** _[T]_ ( _**λ**_ _t_ _−_ _ρ_ ( **Ax** _t_ + **By** _t_ _−_ **c** ))� + **x** _t_ � _,_


where _α >_ 0 is a parameter selected to ensure **G** _t_ _⪰_ **I** .
Here **X** _⪰_ **Y** signifies that **X** _−_ **Y** is positive semidefinite.


To evaluate the convergence behavior of ZOO-ADMM,
we will derive its expected average regret (Hazan,


**ZOO-ADMM: Convergence Analysis and Applications**



_σ >_ 0, then _f_ ( _·_ ; **w** _t_ ) is strongly convex with parameter
_σ_ . Otherwise ( _σ_ = 0), (9) implies convexity of _f_ ( _·_ ; **w** _t_ ).


The ZOO-ADMM iterations are given as Algorithm 1.
Compared to O-ADMM in (Suzuki, 2013), we only require querying two function values for the generation
of gradient estimate at step 3. Also different from (Gao
et al., 2017), steps 7-11 of Algorithm 1 imply that the
equality constraint of problem (21) is always satisfied
at _{_ **x** _t_ _,_ **y** _t_ _[′]_ _[}]_ [ or] _[ {]_ **[x]** _[′]_ _t_ _[,]_ **[ y]** _[t]_ _[}]_ [. The average regret of ZOO-]
ADMM is bounded in Theorem 1.


**Theorem 1** _Suppose_ **B** _is invertible in problem_ (21) _._
_For {_ **x** _t_ _,_ **y** _t_ _[′]_ _[}][ generated by ZOO-ADMM, the expected]_
_average regret is bounded as_


Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)]



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



_T_
� _η_ _t_


_t_ =1



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_




_[σ]_ _[mL]_ 1 [2]

2 _[,]_ [ 0] _[}][R]_ [2] [ +] _T_



(10)
_T_ _[,]_



_g_
+ _[M]_ [(] _[µ]_ [)] [2] _[L]_ [2]
4 _T_



_T_
�



� _η_ _t_ _β_ _t_ [2] [+] _[K]_ _T_

_t_ =1



_where α is introduced in_ (7) _, R, L_ 1 _, L_ _g_ _, s_ ( _m_ ) _and_
_M_ ( _µ_ ) _are defined in Assumptions A-E, and K denotes_
_a constant term that depends on α, R, η_ 1 _,_ **A** _,_ **B** _,_ _**λ**_ _,_
_ρ and L_ 2 _. Suppose_ **A** _is invertible in problem_ (21) _._
_For {_ **x** _[′]_ _t_ _[,]_ **[ y]** _[t]_ _[}][, the regret]_ [ Regret] _T_ [(] **[x]** _[′]_ _t_ _[,]_ **[ y]** _[t]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)] _[ obeys]_
_the same bounds as_ (47) _._


**Proof:** See Sec. 8.2. 

In Theorem 1, if the step size _η_ _t_ and the smoothing
parameter _β_ _t_ are chosen as



_C_ 1
_η_ _t_ =
_m_ ~~_√_~~



_C_ 2 (11)
_t_ _[, β]_ _[t]_ [ =] _M_ ( _µ_ ) _t_



for some constant _C_ 1 _>_ 0 and _C_ 2 _>_ 0, then the regret
bound (47) simplifies to



Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)] _[ ≤]_ _[αR]_ [2]

2 _C_ 1



_√_ ~~_m_~~
~~_√_~~ _T_



+ 2 _C_ 1 _L_ [2] 1



_√_ ~~_m_~~
~~_√_~~ _T_



~~_m_~~ + [5] _[C]_ [1] _[C]_ 2 [2] _[L]_ [2] _g_ 1

_T_ 12 _T_ [+] _[ K]_ _T_



(12)
_T_ _[.]_



**Algorithm 1** ZOO-ADMM for solving problem (21)


1: Input: **x** 1 _∈X_, **y** 1 _∈Y_, _**λ**_ 1 = **0**, _ρ >_ 0, step
sizes _{η_ _t_ _}_, smoothing constants _{β_ _t_ _}_, distribution
_µ_, and _α ≥_ _ρη_ _t_ _λ_ max ( **A** _[T]_ **A** ) + 1 so that **G** _t_ _⪰_ **I**,
where _λ_ max ( _·_ ) denotes the maximum eigenvalue of
a symmetric matrix
2: **for** _t_ = 1 _,_ 2 _, . . ., T_ **do**
3: sample **z** _t_ _∼_ _µ_ to generate ˆ **g** _t_ using (5)
4: update **x** _t_ +1 via (7) under ˆ **g** _t_ and ( **x** _t_ _,_ **y** _t_ _,_ _**λ**_ _t_ )
5: update **y** _t_ +1 via (23) under ( **x** _t_ +1 _,_ _**λ**_ _t_ )
6: update _**λ**_ _t_ +1 via (24) under ( **x** _t_ +1 _,_ **y** _t_ +1 _,_ _**λ**_ _t_ )
7: **if B** is invertible **then**
8: compute **y** _t_ _[′]_ +1 [:=] **[ B]** _[−]_ [1] [(] **[c]** _[ −]_ **[Ax]** _[t]_ [+1] [)]
9: **else**
10: compute **x** _[′]_ _t_ +1 [:=] **[ A]** _[−]_ [1] [(] **[c]** _[ −]_ **[By]** _[t]_ [+1] [)]
11: **end if**

12: **end for**
13: output: _{_ **x** _t_ _,_ **y** _t_ _[′]_ _[}]_ [ or] _[ {]_ **[x]** _[′]_ _t_ _[,]_ **[ y]** _[t]_ _[}]_ [, running average]

_T_

(¯ **x** _T_ _,_ ¯ **y** _T_ _[′]_ [) or (¯] **[x]** _[′]_ _T_ _[,]_ [ ¯] **[y]** _[T]_ [ ), where ¯] **[x]** _[T]_ [ =] _T_ [1] � _k_ =1 **[x]** _[k]_ [.]


That is because the second moment of the gradient
estimate also depends on the number of optimization
variables. In the next section, we will propose two
minibatch strategies that can be used to reduce the
variance of the gradient estimate and to improve the
convergence speed of ZOO-ADMM.


**4** **Convergence for Special Cases**


In this section, we specialize ZOO-ADMM to three
cases: a) stochastic optimization, b) strongly convex
cost function in (21), and c) the use of minibatch
strategies for evaluation of gradient estimates. Without loss of generality, we restrict analysis to the case
that **B** is invertible in (21).


The stochastic optimization problem is a special case
of the OCO problem (21). If the objective function
becomes _F_ ( **x** _,_ **y** ) := E **w** [ _f_ ( **x** ; **w** )] + _φ_ ( **y** ) then we can
link the regret with the optimization error at the running average ¯ **x** _T_ and ¯ **y** _T_ under the condition that _F_ is
convex. We state our results as Corollary 1.


**Corollary 1** _Consider_ _the_ _stochastic_ _optimization_
_problem_ _with_ _the_ _objective_ _function_ _F_ ( **x** _,_ **y** ) :=
E **w** [ _f_ ( **x** ; **w** )] + _φ_ ( **y** ) _, and set η_ _t_ _and β_ _t_ _using_ (11) _. For_
_{_ **x** ¯ _t_ _,_ ¯ **y** _t_ _[′]_ _[}][ generated by ZOO-ADMM, the optimization]_
_error_ E [ _F_ (¯ **x** _T_ _,_ ¯ **y** _T_ _[′]_ [)] _[ −]_ _[F]_ [(] **[x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)]] _[ obeys the same bound]_
_as_ (12) _._


**Proof:** See Sec. 8.4. 

We recall from (9) that _σ_ controls the convexity of _f_ _t_,
where _σ >_ 0 if _f_ _t_ is strongly convex. In Corollary 2, we
show that _σ_ affects the average regret of ZOO-ADMM.


**Corollary 2** _Suppose f_ ( _·_ ; **w** _t_ ) _is strongly convex, and_



The above simplification is derived in Sec. 8.3.


It is clear from (12) that ZOO-ADMM converges
at least as fast as _O_ ( _[√]_ ~~_m_~~ _/√T_ ), which is similar to

the convergence rate of O-ADMM found by (Suzuki,
2013) but involves an additional factor _[√]_ ~~_m_~~ ~~.~~ Such a
dimension-dependent effect on the convergence rate
has also been reported for other zeroth-order optimization algorithms (Duchi et al., 2015; Ghadimi and
Lan, 2013; Shamir, 2017), leading to the same convergence rate as ours. In (12), even if we set _C_ 2 = 0
(namely, _β_ _t_ = 0) for an unbiased gradient estimate (5),
the dimension-dependent factor _[√]_ ~~_m_~~ is not eliminated.


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**



_the step size η_ _t_ _and the smoothing parameter β_ _t_ _are_
_chosen as η_ _t_ = _σtα_ _[and][ β]_ _[t]_ [ =] _MC_ ( _µ_ 2 ) _t_ _[for][ C]_ [2] _[ >]_ [ 0] _[. Given]_
_{_ **x** _t_ _,_ **y** _t_ _[′]_ _[}][ generated by ZOO-ADMM, the expected aver-]_
_age regret can be bounded as_


1 _m_ log _T_
Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)] _[ ≤]_ _[αL]_ [2]
_σ_ _T_

2 _[L]_ [2] _g_ 1
+ [3] _[αC]_ [2] (13)
8 _σ_ _T_ [+] _[ K]_ _T_ _[.]_


**Proof:** See Sec. 8.5. 

Corollary 2 implies that when the cost function is
strongly convex, the regret bound of ZOO-ADMM
could achieve _O_ ( _m/T_ ) up to a logarithmic factor log _T_ .
Compared to the regret bound _O_ ( _[√]_ ~~_m_~~ _/√T_ ) in the gen
eral case (12), the condition of strong convexity improves the regret bound in terms of the number of
iterations _T_, but the dimension-dependent factor now
becomes linear in the dimension _m_ due to the effect of
the second moment of gradient estimate.


The use of a gradient estimator makes the convergence
rate of ZOO-ADMM dependent on the dimension _m_,
i.e., the number of optimization variables. Thus, it is
important to study the impact of minibatch strategies
on the acceleration of the convergence speed (Cotter
et al., 2011; Duchi et al., 2015; Li et al., 2014; Suzuki,
2013). Here we present two minibatch strategies: gradient sample averaging and observation averaging. In
the first strategy, instead of using a single sample as
in (5), the average of _q_ sub-samples _{_ **z** _t,i_ _}_ _[q]_ _i_ =1 [are used]
for gradient estimation



**Corollary 3** _Consider the hybrid minibatch strategy_
(51) _in ZOO-ADMM, and set η_ _t_ = ~~_√_~~ 1+ _C_ _q_ 11 ~~_m_~~ _q_ 2 ~~_√_~~ _t_ _[and]_

_β_ _t_ = _MC_ ( _µ_ 2 ) _t_ _[. The expected average regret is bounded as]_



_q_ 1 _q_ 2
~~_√_~~ _T_



Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)] _[ ≤]_ _[αR]_ [2]

2 _C_ 1



�



1 + _[s]_ [(] _[m]_ [)]



_T_



_q_ 1 _q_ 2
~~_√_~~ _T_



_q_ 1 _q_ 2 + [5] _[C]_ [1] _[C]_ 2 [2] _[L]_ [2] _g_ 1

_T_ 6 _T_ [+] _[ K]_ _T_



(17)
_T_ _[,]_



1 + _[s]_ [(] _[m]_ [)]



+ 2 _C_ 1 _L_ [2] 1



�



_where q_ 1 _and q_ 2 _are number of sub-samples {_ **z** _t,i_ _} and_
_{_ **w** _t,i_ _}, respectively._


**Proof:** See Sec. 8.6. 


It is clear from Corollary 3 that the use of minibatch
strategies can alleviate the dimension dependency,
leading to the regret bound _O_ ( ~~�~~ 1 + _m/_ ( _q_ 1 _q_ 2 ) _/√T_ ).



1 + _m/_ ( _q_ 1 _q_ 2 ) _/√_



leading to the regret bound _O_ ( ~~�~~ 1 + _m/_ ( _q_ 1 _q_ 2 ) _/√T_ ).

The regret bound in (17) also implies that the convergence behavior of ZOO-ADMM is similar using either
gradient sample averaging minibatch (14) or observation averaging minibatch (15). If _q_ 1 = 1 and _q_ 2 = 1,
the regret bound (17) reduces to _O_ ( _[√]_ ~~_m_~~ _/√T_ ), which

is the general case in (12). If _q_ 1 _q_ 2 = _O_ ( _m_ ), we obtain
the regret error _O_ (1 _/√T_ ) as in the case where an ex
plicit expression for the gradient is used in the OCO
algorithms.



**g** ˆ _t_ = [1]

_q_



_q_
�


_i_ =1



_f_ ( **x** _t_ + _β_ _t_ **z** _t,i_ ; **w** _t_ ) _−_ _f_ ( **x** _t_ ; **w** _t_ ) **z** _t,i_ _,_ (14)

_β_ _t_



where _q_ is called the batch size. The use of (14) is analogous to the use of an average gradient in incremental
gradient (Blatt et al., 2007) and stochastic gradient
(Roux et al., 2012). In the second strategy, we use a
subset of observations _{_ **w** _t,i_ _}_ _[q]_ _i_ =1 [to reduce the gradient]
variance,



**g** ˆ _t_ = [1]

_q_



_q_
�


_i_ =1



_f_ ( **x** _t_ + _β_ _t_ **z** _t_ ; **w** _t,i_ ) _−_ _f_ ( **x** _t_ ; **w** _t,i_ ) **z** _t_ _._ (15)

_β_ _t_



We note that in the online setting, the subset of
observations _{_ **w** _t,i_ _}_ _[q]_ _i_ =1 [can be obtained via a sliding]
time window of length _q_, namely, **w** _i,t_ = **w** _t−i_ +1 for
_i_ = 1 _,_ 2 _, . . ., q_ .


Combination of (14) and (15) yields a hybrid strategy



_q_ 2
�


_i_ =1



**5** **Applications of ZOO-ADMM**


In this section, we demonstrate several applications of
ZOO-ADMM in signal processing, statistics and machine learning.


**5.1** **Black-box optimization**


In some OCO problems, explicit gradient calculation is
impossible due to the lack of a mathematical expression for the loss function. For example, commercial
recommender systems try to build a representation of
a customer’s buying preference function based on a discrete number of queries or purchasing history, and the
system never has access to the gradient of the user’s
preference function over their product line, which may
even be unknown to the user. Gradient-free methods are therefore necessary. A specific example is the
Yahoo! music recommendation system (Dror et al.,
2012), which will be further discussed in the Sec. 6.
In these examples, one can consider each user as a
black-box model that provides feedback on the value
of an objective function, e.g., relative preferences over
all products, based on an online evaluation of the objective function at discrete points on its domain. Such
a system can benefit from ZOO-ADMM.



ˆ 1
**g** _t_ =
_q_ 1 _q_ 2



_q_ 1
�

_j_ =1



_f_ ( **x** _t_ + _β_ _t_ **z** _t,j_ ; **w** _t,i_ ) _−_ _f_ ( **x** _t_ ; **w** _t,i_ )

**z** _t,j_ _._
_β_ _t_



(16)


In Corollary 3, we demonstrate the convergence behavior of the general hybrid ZOO-ADMM.


**ZOO-ADMM: Convergence Analysis and Applications**



**5.2** **Sensor selection**


Sensor selection for parameter estimation is a fundamental problem in smart grids, communication systems, and wireless sensor networks (Hero and Cochran,
2011; Liu et al., 2016). The goal is to seek the optimal
tradeoff between sensor activations and the estimation
accuracy. The sensor selection problem is also closely
related to leader selection (Lin et al., 2014) and experimental design (Boyd and Vandenberghe, 2004).


For sensor selection, we often solve a (relaxed) convex
program of the form (Joshi and Boyd, 2009)



to cancer recurrence or death) (Sohn et al., 2009). Let
_{_ **a** _i_ _∈_ R _[m]_ _, δ_ _i_ _∈{_ 0 _,_ 1 _}, t_ _i_ _∈_ R + _}_ _[n]_ _i_ =1 [be] _[ n]_ [ triples of] _[ m]_
covariates, where **a** _i_ is a vector of covariates or factors
for subject _i_, _δ_ _i_ is a censoring indicator variable taking
1 if an event (e.g., death) is observed and 0 otherwise,
and _t_ _i_ denotes the censoring time.


This sparse regression problem can be formulated as
the solution to an _ℓ_ 1 penalized optimization problem
(Park and Hastie, 2007; Sohn et al., 2009), which yields



1
minimize

**x** _n_









_i_ **[x]** [ + log]



 _[−]_ **[a]** _[T]_



 _j_ [�] _∈R_



_e_ **[a]** _j_ _[T]_ **[x]**
_j_ [�] _∈R_ _i_



_n_
� _δ_ _i_


_i_ =1











��



�



1
minimize
**x** _T_



_T_
�


_t_ =1



_−_
logdet



_m_
� _x_ _i_ **a** _i,t_ **a** _[T]_ _i,t_
� _i_ =1



(18)



subject to **1** _[T]_ **x** = _m_ 0 _,_ **0** _≤_ **x** _≤_ 1 _,_


where **x** _∈_ R _[m]_ is the optimization variable, _m_ is the
number of sensors, **a** _i,t_ _∈_ R _[n]_ is the observation coefficient of sensor _i_ at time _t_, and _m_ 0 is the number of
selected sensors. The objective function of (18) can be
interpreted as the log determinant of error covariance
associated with the maximum likelihood estimator for
parameter estimation (Rao, 1973). The constraint
**0** _≤_ **x** _≤_ **1** is a relaxed convex hull of the Boolean
constraint **x** _∈{_ 0 _,_ 1 _}_ _[m]_, which encodes whether or not
a sensor is selected.


Conventional methods such as projected gradient
(first-order) and interior-point (second-order) algorithms can be used to solve problem (18). However,
both of them involve calculation of inverse matrices
necessary to evaluate the gradient of the cost function.
By contrast, we can rewrite (18) in a form amenable
to ZOO-ADMM that avoids matrix inversion,



1
minimize
**x** _,_ **y** _T_



_T_
� _f_ ( **x** ; **w** _t_ ) + _I_ 1 ( **x** ) + _I_ 2 ( **y** )


_t_ =1



(19)



+ _γ∥_ **x** _∥_ 1 (20)


where **x** _∈_ R _[m]_ is the vector of covariates coefficients
to be designed, _R_ _i_ is the set of subjects at risk at
time _t_ _i_, namely, _R_ _i_ = _{j_ : _t_ _j_ _≥_ _t_ _i_ _}_, and _γ >_ 0 is
a regularization parameter. In the objective function
of (20), the first term corresponds to the (negative)
log partial likelihood for the Cox proportional hazards
model (Cox, 1972), and the second term encourages
sparsity of the covariate coefficients.


By introducing a new variable **y** _∈_ R _[m]_ together with
the constraint **x** _−_ **y** = **0**, problem (20) can be cast as
the canonical form (21) amenable to the ZOO-ADMM
algorithm. This helps us to avoid the gradient calculation for the involved objective function in Cox regression. We specify the ZOO-ADMM algorithm for
solving (20) in Sec. 8.8.


**6** **Experiments**


In this section, we demonstrate the effectiveness of
ZOO-ADMM, and validate its convergence behavior
for the applications introduced in Sec. 5. In Algorithm 1, we set **x** 1 = **0**, **y** 1 = **0**, _**λ**_ 1 = **0**, _ρ_ = 10,
_η_ _t_ = 1 _/√mt_, _β_ _t_ = 1 _/_ ( _m_ [1] _[.]_ [5] _t_ ), _α_ = _ρη_ _t_ _λ_ max ( **A** _[T]_ **A** ) + 1,

and the distribution _µ_ is chosen to be uniform on the
surface of the Euclidean-ball of radius _[√]_ ~~_m_~~ . Unless
specified otherwise, we use the gradient sample averaging minibatch of size 30 in ZOO-ADMM. Through
this section, we compare ZOO-ADMM with the conventional O-ADMM algorithm in (Suzuki, 2013) under the same parameter settings. Our experiments are
performed on a synthetic dataset for sensor selection,
and on real datasets for black-box optimization and
Cox regression. Experiments were conducted by Matlab R2016 on a machine with 3.20 GHz CPU and 8

GB RAM.


**Black-box optimization:** We consider prediction of
users’ ratings in the Yahoo! music system (Dror et al.,
2012). Our dataset, provided by (Lian et al., 2016),
include _n_ _[′]_ = 131072 true music ratings **r** _∈_ R _[n]_ _[′]_, and
the predicted ratings of _m_ = 237 individual models



subject to **x** _−_ **y** = **0** _,_


where **y** _∈_ R _[m]_ is an auxiliary variable, _f_ ( **x** ; **w** _t_ ) =

_−_
logdet( [�] _[m]_ _i_ =1 _[x]_ _[i]_ **[a]** _[i,t]_ **[a]** _[T]_ _i,t_ [) with] **[ w]** _[t]_ [ =] _[ {]_ **[a]** _[i,t]_ _[}]_ _i_ _[m]_ =1 [, and] _[ {I]_ _[i]_ _[}]_
are indicator functions


0 **0** _≤_ **x** _≤_ **1** 0 **1** _[T]_ **y** = _m_ 0
_I_ 1 ( **x** ) = _I_ 2 ( **y** ) =
� _∞_ otherwise _,_ � _∞_ otherwise _._


We specify the ZOO-ADMM algorithm for solving (59)
in Sec. 8.7.


**5.3** **Sparse Cox regression**


In survival analysis, Cox regression (also known as proportional hazards regression) is a method to investigate effects of variables of interest upon the amount of
time that elapses before a specified event occurs, e.g.,
relating gene expression profiles to survival time (time


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**



ZOO-ADMM: no minibatch

|ZOO-ADMM: minibatch over {𝒘𝑡}, size = 5<br>ZOO-ADMM: minibatch over {𝒛𝑡}, size = 5<br>O-ADMM: minibatch over {𝒘𝑡}, size = 50|Col2|Col3|Col4|
|---|---|---|---|
|||||



Iteration


(a)

|Col1|ZOO-ADMM<br>O-ADMM|Col3|ZOO-ADMM|Col5|
|---|---|---|---|---|
||||||



Iteration


(b)


**Figure 1:** Convergence of ZOO-ADMM: a) RMSE under
different minibatch strategies, b) update error with minibatch size equal to 50.



Number of selected sensors, 𝑚 0


(a)





Number of optimization variables

(b)


**Figure 2:** ZOO-ADMM for sensor selection: a) MSE versus number of selected sensors _m_ 0, b) computation time
versus number of optimization variables.



created from the NTU KDD-Cup team (Chen et al.,
2011). Let **C** _∈_ R _[n][×][m]_ represent a matrix of each models’ predicted ratings on Yahoo! music data sample.
We split the dataset ( **C** _,_ **r** ) into two equal parts, leading to the training dataset ( **C** 1 _∈_ R _[n][×][m]_ _,_ **r** 1 _∈_ R _[n]_ )
and the test dataset ( **C** 2 _∈_ R _[n][×][m]_ _,_ **r** 2 _∈_ R _[n]_ ), where
_n_ = _n_ _[′]_ _/_ 2.


Our goal is to find the optimal coefficients **x** to blend
_m_ individual models such that the mean squared error


_n_ _n_

_f_ ( **x** ) := [1] � _i_ =1 _[f]_ [(] **[x]** [;] **[ w]** _[i]_ [) =] [1] � _i_ =1 [([] **[C]** [1] []] _i_ _[T]_ **[x]** _[ −]_ [[] **[r]** [1] []] _[i]_ [)] [2] [ is]



_n_ [1] � _ni_ =1 _[f]_ [(] **[x]** [;] **[ w]** _[i]_ [) =] _n_ [1]



_n_ _n_

_f_ ( **x** ) := _n_ [1] � _i_ =1 _[f]_ [(] **[x]** [;] **[ w]** _[i]_ [) =] _n_ [1] � _i_ =1 [([] **[C]** [1] []] _i_ _[T]_ **[x]** _[ −]_ [[] **[r]** [1] []] _[i]_ [)] [2] [ is]

minimized, where **w** _i_ = ([ **C** 1 ] _i_ _,_ [ **r** 1 ] _i_ ), [ **C** 1 ] _i_ is the _i_ th
row vector of **C** 1, and [ **r** 1 ] _i_ is the _i_ th entry of **r** 1 . Since
( **C** _,_ **r** ) includes predicted ratings on Yahoo! Music data
using NTU KDD-Cup team’s models, it is private information known only to other users. Therefore, the
information ( **C** _,_ **r** ) cannot be accessed directly (Lian
et al., 2016), and explicit gradient calculation for _f_ is
not possible. We thus treat the loss function as a black
box, where it is evaluated at individual points **x** in its
domain but not over any open region of its domain.


As discussed in Sec. 5.1, we can apply ZOO-ADMM



to solve the proposed linear blending problem, and
the prediction accuracy can be measured by the root
mean squared error (RMSE) of the test data RMSE =
� _∥_ **r** 2 _−_ **C** 2 **x** _∥_ [2] 2 _[/n]_ [, where an update of] **[ x]** [ is obtained at]

each iteration.


In Fig. 1, we compare the performance of ZO-ADMM
with O-ADMM and the optimal solution provided by
(Lian et al., 2016). In Fig. 1-(a), we present RMSE
as a function of iteration number under different minibatch schemes. As we can see, both gradient sample
averaging (over _{_ **z** _t_ _}_ ) and observation averaging (over
_{_ **w** _t_ _}_ ) significantly accelerate the convergence speed of
ZOO-ADMM. In particular, when the minibatch size
_q_ is large enough (50 in our example), the dimensiondependent slowdown factor of ZOO-ADMM can be
mitigated. We also observe that ZOO-ADMM reaches
the best RMSE in (Lian et al., 2016) after 10000 iterations. In Fig. 1-(b), we show the convergence error _∥_ **x** _t_ +1 _−_ **x** _t_ _∥_ 2 versus iteration number using gradient sample averaging minibatch of size 50. Compared
to O-ADMM, ZOO-ADMM has a larger performance
gap in its first few iterations, but it thereafter con

20


15


10


5


0



**ZOO-ADMM: Convergence Analysis and Applications**


**Table 1:** Percentage of common genes found using ZOOADMM and Cox scores (Witten and Tibshirani, 2010).




|# selected genes|γ = 1.5<br>19|γ = 0.05<br>56|γ = 0.001<br>93|
|---|---|---|---|
|Overlapping (%)|80.1%|87.5%|92.3%|



100


50


0

10 1.5 0.25 0.05 0.005 0.001
Sparsity promoting parameter, γ


**Figure 3:** Partial likelihood and number of selected genes
versus sparsity promoting parameter _γ_ .


verges quickly resulting in comparable performance to
O-ADMM.


**Sensor selection:** We consider an example of estimating a spatial random field based on measurements
of the field at a discrete set of sensor locations. Assume that _m_ = 100 sensors are randomly deployed
over a square region to monitor a vector of field intensities (e.g., temperature values). The objective is
to estimate the field intensity at _n_ = 5 locations over
a time period of _T_ = 1000 secs. In (18), the observation vectors _{_ **a** _i,t_ _}_ are chosen randomly, and independently, from a distribution _N_ ( _µ_ _i_ **1** _n_ _,_ **I** _n_ ). Here _µ_ _i_ is
generated by an exponential model (Liu et al., 2016),
_µ_ _i_ = 5 _e_ � _nj_ =1 _[∥]_ [ˆ] **[s]** _[j]_ _[−]_ [˜] **[s]** _[i]_ _[∥]_ [2] _[/n]_, where ˆ **s** _j_ is the _j_ -th spatial location at which the field intensity is to be estimated
and ˜ **s** _i_ is the spatial location of the _i_ sensor.


In Fig. 2, we present the performance of ZOO-ADMM
for sensor selection. In Fig. 2-(a), we show the mean
squared error (MSE) averaged over 50 random trials
for different number of selected sensors _m_ 0 in (18).
We compare our approach with O-ADMM and the
method in (Joshi and Boyd, 2009). The figure shows
that ZOO-ADMM yields almost the same MSE as OADMM. The method in (Joshi and Boyd, 2009) yields
slightly better estimation performance, since it uses
the second-order optimization method for sensor selection. In Fig. 2-(b), we present the computation time of
ZOO-ADMM versus the number of optimization variables _m_ . The figure shows that ZOO-ADMM becomes
much more computationally efficient as _m_ increases
since no matrix inversion is required.


**Sparse Cox regression:** We next employ ZOOADMM to solve problem (20) for building a sparse predictor of patient survival using the Kidney renal clear
cell carcinoma dataset [2] . The aforementioned dataset
includes clinical data (survival time and censoring information) and gene expression data for 606 patients


2 Available at `[http://gdac.broadinstitute.org/](http://gdac.broadinstitute.org/)`



(534 with tumor and 72 without tumor). Our goal
is to seek the best subset of genes (in terms of optimal sparse covariate coefficients) that make the most
significant impact on the survival time.


In Fig. 3, we show the partial likelihood and number
of selected genes as functions of the regularization parameter _γ_ . The figure shows that ZOO-ADMM nearly
attains the accuracy of O-ADMM. Furthermore, the
likelihood increases as the number of selected genes
increases. There is thus a tradeoff between the (negative) log partial likelihood and the sparsity of covariate
coefficients in problem (20). To test the significance of
our selected genes, we compare our approach with the
significance analysis based on univariate Cox scores
used in (Witten and Tibshirani, 2010). The percentage of overlap between the genes identified by each
method is shown in Table 1 under different values of
_γ_ . Despite its use of a zeroth order approximation to
the gradient, the ZOO-ADMM selects at least 80% of
the genes selected by the gradient-based Cox scores of
(Witten and Tibshirani, 2010).


**7** **Conclusion**



In this paper, we proposed and analyzed a
gradient-free (zeroth-order) online optimization algorithm, ZOO-ADMM. We showed that the regret bound of ZOO-ADMM suffers an additional
dimension-dependent factor in convergence rate over
gradient-based online variants of ADMM, leading to
_O_ ( _[√]_ ~~_m_~~ _/√T_ ) convergence rate, where _m_ is the num
ber of optimization variables. To alleviate the dimension dependence, we presented two minibatch
strategies that yield an improved convergence rate of
_O_ ( ~~�~~ 1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the minibatch size.



1 + _q_ _[−]_ [1] _m/√_



_O_ ( ~~�~~ 1 + _q_ _[−]_ [1] _m/√T_ ), where _q_ is the minibatch size.

We illustrated the effectiveness of ZOO-ADMM via
multiple applications using both synthetic and realworld datasets. In the future, we would like to relax
the assumptions on smoothness and convexity of the
cost function in ZOO-ADMM.



**Acknowledgements**


This work was partially supported by grants from the
US Army Research Office, grant numbers W911NF-151-0479 and W911NF-15-1-0241. The work of J. Chen
was supported in part by the Natural Science Foundation of China under Grant 61671382.


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**



**8** **Supplementary Material**


**8.1** **Assumptions and Key Notations**


Recall that we consider the regularized loss minimization problem over a time horizon of length _T_,



**8.2** **Proof of Theorem 1**


Since the sequences _{_ **x** _t_ _}_, _{_ **y** _t_ _}_ and _{_ _**λ**_ _t_ _}_ produced
from (22)-(24) have the same structure as the
ADMM/O-ADMM steps, the property of ADMM
given by Theorem 4 of (Suzuki, 2013) is directly applicable to our case, yielding



1
minimize
**x** _∈X_ _,_ **y** _∈Y_ _T_



_T_
� _f_ _t_ ( **x** ; **w** _t_ ) + _φ_ ( **y** )


_t_ =1



(21)



_T_
�( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ )) _−_


_t_ =1



_T_
�( _f_ _t_ ( **x** ) + _φ_ ( **y** ))


_t_ =1



subject to **Ax** + **By** = **c** _._


ZOO-ADMM is given by



_T_
� _t_ =1 (˜ **v** _t_ _−_ **v** ) _[T]_ _H_ (˜ **v** _t_ ) _≤_ _[∥]_ **[x]** [1] _[ −]_ 2 _η_ **[x]** 1 _[∥]_ **G** [2] 1



_T_
�


_t_ =2



**x** _t_ +1 = arg min
**x** _∈X_



ˆ _T_
� **g** _t_ **[x]** _[ −]_ _**[λ]**_ _[T]_ _t_ [(] **[Ax]** [ +] **[ By]** _[t]_ _[−]_ **[c]** [)]



�



_∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t_ _−_ _∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t−_ 1
2 _η_ _t_ 2 _η_ _t−_ 1
�



+


+




_[ρ]_ 2 [+] [1]

2 _[∥]_ **[Ax]** [ +] **[ By]** _[t]_ _[ −]_ **[c]** _[∥]_ [2] 2



+ _[ρ]_



2 _η_ _t_ _∥_ **x** _−_ **x** _t_ _∥_ [2] **G** _t_



_,_ (22)
�



**y** _t_ +1 = arg min
**y** _∈Y_



� _φ_ ( **y** ) _−_ _**λ**_ _[T]_ _t_ [(] **[Ax]** _[t]_ [+1] [+] **[ By]** _[ −]_ **[c]** [)]



+ _[ρ]_ 2 _,_ (23)

2 _[∥]_ **[Ax]** _[t]_ [+1] [ +] **[ By]** _[ −]_ **[c]** _[∥]_ [2] �



+ _⟨_ _**λ**_ _,_ **A** ( **x** _T_ +1 _−_ **x** 1 ) _⟩_ + _[ρ]_ _[∥]_ _**[λ]**_ [1] _[ −]_ _**[λ]**_ _[∥]_ 2 [2]

2 _[∥]_ **[y]** [1] _[ −]_ **[y]** _[∥]_ **[B]** _[T]_ **[ B]** [ +] 2 _ρ_

_−_ _[∥]_ _**[λ]**_ _[T]_ [ +1] _[ −]_ _**[λ]**_ _[∥]_ 2 [2] + _⟨_ **B** ( **y** _−_ **y** _T_ +1 ) _,_ _**λ**_ _T_ +1 _−_ _**λ**_ _⟩_
2 _ρ_



_**λ**_ _t_ +1 = _**λ**_ _t_ _−_ _ρ_ ( **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** ) _,_ (24)


where **G** _t_ = _α_ **I** _−_ _ρη_ _t_ **A** _[T]_ **A** .


We first elaborate on our assumptions.


_•_ Assumption A implies that _∥_ **x** _−_ **x** _[′]_ _∥_ 2 _≤_ _R_ and
_∥_ **y** _−_ **y** _[′]_ _∥_ 2 _≤_ _R_ for all **x** _,_ **x** _[′]_ _∈X_ and for all **y** _,_ **y** _[′]_ _∈_
_Y_ .


_•_ Based on Jensen’s inequality, Assumptions B implies that _∥_ E[ _∇_ **x** _f_ ( **x** ; **w** _t_ )] _∥_ 2 _≤_ _L_ 1 .


_•_ Assumption C implies a Lipschitz condition over
the gradient _∇_ **x** _f_ ( **x** ; **w** _t_ ) with constant _L_ _g_ ( **w** _t_ )
(Bubeck et al., 2015; Hazan, 2016). Also based
on Jensen’s inequality, we have _|_ E[ _L_ _g_ ( **w** _t_ )] _| ≤_ _L_ _g_ .


We next introduce key notations used in our analysis.
Given the primal-dual variables **x**, **y** and _**λ**_ of problem
(21), we define **v** := [ **x** _[T]_ _,_ **y** _[T]_ _,_ _**λ**_ _[T]_ ], and a primal-dual
mapping _H_



_T_

_−⟨_ **B** ( **y** _−_ **y** 1 ) _,_ _**λ**_ 1 _−_ _**λ**_ _⟩−_ �


_t_ =1



_∥λ_ _t_ _−_ _λ_ _t_ +1 _∥_ 2 [2]

2 _ρ_



_T_
�


_t_ =1



_η_ _t_

2 _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ **G** [2] _[−]_ _t_ [1] _[.]_ (27)



_−_



_T_
�


_t_ =1



_σ_

2 [+]
2 _[∥]_ **[x]** _[t]_ _[ −]_ **[x]** _[∥]_ [2]



Here for notational simplicity we have used, and henceforth will continue to use, _f_ _t_ ( **x** _t_ ) instead of _f_ ( **x** _t_ ; **w** _t_ ).


In (27), based on **G** _t_ = _α_ **I** _−_ _ρη_ _t_ **A** _[T]_ **A**, we have


_∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t_ _−_ _∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t−_ 1
2 _η_ _t_ 2 _η_ _t−_ 1



_α_ _α_

= _−_
� 2 _η_ _t_ 2 _η_ _t−_ 1


which yields



_∥_ **x** _t_ _−_ **x** _∥_ 2 [2] _[,]_
�



�



_∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t_ _−_ _∥_ **x** _t_ _−_ **x** _∥_ [2] **G** _t−_ 1
2 _η_ _t_ 2 _η_ _t−_ 1
�



 _,_ **C** :=







 00 00 _−−_ **AB** _[T][T]_


**A** **B** 0





_T_
�


_t_ =2





_,_ (25)




_H_ ( **v** ) := **Cv** _−_



00


**c**




0 0 _−_ **B** _[T]_



**A** **B** 0



_−_



_T_
�


_t_ =1



_σ_

2 _[≤]_
2 _[∥]_ **[x]** _[t]_ _[ −]_ **[x]** _[∥]_ [2]



where **C** is skew symmetric, namely, **C** _[T]_ = _−_ **C** . An
important property of the affine mapping _H_ is that
_⟨_ **v** 1 _−_ **v** 2 _, H_ ( **v** 1 ) _−_ _H_ ( **v** 2 ) _⟩_ = 0 for every **v** 1 and **v** 2 .
Supposing the sequence _{_ **v** _t_ _}_ is generated by an algorithm, we introduce the auxiliary sequence


**v** ˜ _t_ := [ **x** _[T]_ _t_ _[,]_ **[ y]** _t_ _[T]_ _[,]_ [ ˜] _**[λ]**_ _[T]_ _t_ []] _[T]_ _[,]_ (26)


where _**λ**_ [˜] _t_ := _**λ**_ _t_ _−_ _ρ_ ( **Ax** _t_ +1 + **By** _t_ _−_ **c** ).



We also note that the terms 2 _η_ 1 1 _[∥]_ **[x]** [1] _[ −]_ **[x]** _[∥]_ **G** [2] 1 [,]
_⟨_ _**λ**_ _,_ **A** ( **x** _T_ +1 _−_ **x** 1 ) _⟩_, _ρ_ 2 _[∥]_ **[y]** [1] _[ −]_ **[y]** _[∥]_ **[B]** _[T]_ **[ B]** [,] 21 _ρ_ [(] _[∥]_ _**[λ]**_ [1] _[ −]_ _**[λ]**_ _[∥]_ 2 [2] _[−]_
_∥_ _**λ**_ _T_ +1 _−_ _**λ**_ _∥_ 2 [2] [),] _[ ⟨]_ **[B]** [(] **[y]** _[ −]_ **[y]** _[T]_ [ +1] [)] _[,]_ _**[ λ]**_ _[T]_ [ +1] _[−]_ _**[λ]**_ _[⟩]_ [, and] _[ ⟨]_ **[B]** [(] **[y]** _[ −]_
**y** 1 ) _,_ _**λ**_ 1 _−_ _**λ**_ _⟩_ are _independent_ of time _t_ . In particular,



_T_
�



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



(28)
2 _[,]_ [ 0] _[}][R]_ [2] _[.]_



� _t_ =2 max _{_ 2 _[α]_ _η_


we have



**ZOO-ADMM: Convergence Analysis and Applications**


where the last equality holds due to (32).



_∥_ **x** 1 _−_ **x** _∥_ [2] **G** 1 _[≤]_ _[αR]_ [2] _[,]_
_⟨_ _**λ**_ _,_ **A** ( **x** _T_ +1 _−_ **x** 1 ) _⟩≤_ _R∥_ _**λ**_ _∥_ 2 _∥_ **A** _∥_ _F_ _,_

( _∥_ _**λ**_ 1 _−_ _**λ**_ _∥_ 2 [2] _[−∥]_ _**[λ]**_ _[T]_ [ +1] _[−]_ _**[λ]**_ _[∥]_ [2] 2 [)] _[ ≤∥]_ _**[λ]**_ _[∥]_ [2] 2 _[,]_
_⟨_ **B** ( **y** _−_ **y** 1 ) _,_ _**λ**_ _−_ _**λ**_ 1 _⟩≤_ _R∥_ **B** _∥_ _F_ _∥_ _**λ**_ _∥_ 2 _,_ (29)


where _∥· ∥_ _F_ denotes the Frobenius norm of a matrix,
and we have used the facts that **G** _t_ _⪯_ _α_ **I** and _**λ**_ 1 = **0** .


Based on the optimality condition of **y** _t_ +1 in (23), we
have _⟨∂φ_ ( **y** _t_ +1 ) _−_ **B** _[T]_ _**λ**_ _t_ + _ρ_ **B** _[T]_ ( **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** ) _,_ **y** _−_
**y** _t_ +1 _⟩≥_ 0 _, ∀_ **y** _∈Y_, which is equivalent to _⟨∂φ_ ( **y** _t_ +1 ) _−_
**B** _[T]_ _**λ**_ _t_ +1 _,_ **y** _−_ **y** _t_ +1 _⟩≥_ 0. And thus, we obtain


_⟨_ _**λ**_ _t_ +1 _,_ **B** ( **y** _−_ **y** _t_ +1 ) _⟩−⟨_ _**λ**_ _,_ **B** ( **y** _−_ **y** _t_ +1 ) _⟩_
_≤⟨∂φ_ ( **y** _t_ +1 ) _,_ **y** _−_ **y** _t_ +1 _⟩−⟨_ _**λ**_ _,_ **B** ( **y** _−_ **y** _t_ +1 ) _⟩,_


which yields


_⟨_ **B** ( **y** _−_ **y** _t_ +1 ) _,_ _**λ**_ _t_ +1 _−_ _**λ**_ _⟩_

_≤⟨_ **y** _−_ **y** _t_ +1 _, ∂φ_ ( **y** _t_ +1 ) _−_ **B** _[T]_ _**λ**_ _⟩_

_≤R_ ( _L_ 2 + _∥_ **B** _[T]_ _**λ**_ _∥_ 2 ) _,_ (30)


where we have used the fact that _∥∂φ_ ( **y** _t_ +1 ) _∥_ 2 _≤_ _L_ 2 .


Substituting (28)-(30) into (27), we then obtain



Let ( **x** _[∗]_ _,_ **y** _[∗]_ ) be the optimal solution (implying **Ax** _[∗]_ +
**By** _[∗]_ _−_ **c** = **0** ). For any dual variable _**λ**_ _[∗]_ and ˜ **v** _t_ =

[ **x** _[T]_ _t_ _[,]_ **[ y]** _t_ _[T]_ _[,]_ [ ˜] _**[λ]**_ _[T]_ _t_ []] _[T]_ [, we have]


(˜ **v** _t_ _−_ **v** _[∗]_ ) _[T]_ _H_ (˜ **v** _t_ ) = _H_ ( **v** _[∗]_ ) _[T]_ (˜ **v** _t_ _−_ **v** _[∗]_ )




[]

**x** _t_ _−_ **x** _[∗]_
 _**λ**_ **y** ˜ _tt_ _− −_ **y** _**λ**_ _[∗][∗]_



_−_ **B** _[T]_ _**λ**_ _[∗]_









_T_ []



=







 _−−_ **AB** _[T][T]_ _**λλ**_ _[∗][∗]_

 **Ax** _[∗]_ + **By** _[∗]_



**Ax** _[∗]_ + **By** _[∗]_ _−_ **c**









_**λ**_ **y** ˜ _tt_ _− −_ **y** _**λ**_ _[∗][∗]_



= _⟨_ _**λ**_ _[∗]_ _,_ **c** _−_ **Ax** _t_ _−_ **By** _t_ _⟩_ = [1] (34)

_ρ_ _[⟨]_ _**[λ]**_ _[∗]_ _[,]_ _**[ λ]**_ _[t]_ _[ −]_ _**[λ]**_ _[t][−]_ [1] _[⟩]_



where **v** _[∗]_ := [( **x** _[∗]_ ) _[T]_ _,_ ( **y** _[∗]_ ) _[T]_ _,_ ( _**λ**_ _[∗]_ ) _[T]_ ] _[T]_, and the affine
mapping _H_ ( _·_ ) is given by (25).


Setting _**λ**_ _[∗]_ = ( **B** _[−]_ [1] ) _[T]_ _∂φ_ ( **y** _t_ _[′]_ [), based on (33) and (34)]
we have


_f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ _[′]_ [)] _[ −]_ [(] _[f]_ _[t]_ [(] **[x]** _[∗]_ [) +] _[ φ]_ [(] **[y]** _[∗]_ [))]

_≤f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ ) + (˜ **v** _t_ _−_ **v** _[∗]_ ) _[T]_ _H_ (˜ **v** _t_ )
_−_ ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ )) _._ (35)


Combining (31) and (35) yields



_T_



1

_T_



_T_
�



� ( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ _[′]_ [))] _[ −]_ _T_ [1]

_t_ =1



_T_
� ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



_T_



_T_
� ( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ ))


_t_ =1



_T_



_T_
�


_t_ =1


_T_
�



_T_
�


_t_ =1



_∥_ _**λ**_ _t_ +1 _−_ _**λ**_ _t_ _∥_ 2 [2]
_≤_ [1]
2 _ρ_ _T_



_∥_ _**λ**_ _t_ +1 _−_ _**λ**_ _t_ _∥_ 2 [2]

2 _ρ_



1

_T_



_T_
�



� ( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ )) _−_ _T_ [1]

_t_ =1



_T_
� ( _f_ _t_ ( **x** ) + _φ_ ( **y** ))


_t_ =1



_T_



+ [1]

_T_


_−_ [1]

_T_


+ [1]

_T_



� ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ )) + _T_ [1]

_t_ =1



+ [1]

_T_



_T_
�



�(˜ **v** _t_ _−_ **v** ) _[T]_ _H_ (˜ **v** _t_ ) + _T_ [1]

_t_ =1



_T_
�(˜ **v** _t_ _−_ **v** _[∗]_ ) _[T]_ _H_ (˜ **v** _t_ )


_t_ =1



_T_
�


_t_ =1



_∥_ _**λ**_ _t_ +1 _−_ _**λ**_ _t_ _∥_ 2 [2]

2 _ρ_



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



+ [1]

_T_



_T_
�


_t_ =1



_η_ _t_

2 _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ [2] [ +] _[ K]_ _T_



_η_ _t_



(31)
_T_ _[,]_



_≤_ [1]

_T_



_T_
�


_t_ =1



_η_ _t_

2 [+] _[ K]_
2 _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ [2] _T_



(36)
_T_ _[.]_



where _K_ is a constant term related to _α_, _R_, _η_ 1, **A**,
**B**, _**λ**_, _ρ_ and _L_ 2, _K_ = _[αR]_ [2] [+] _[ R][∥]_ _**[λ]**_ _[∥]_ [2] _[∥]_ **[A]** _[∥]_ _[F]_ [ +] [1] _[∥]_ _**[λ]**_ _[∥]_ 2 [2] [+]



_η_ _t_




_[αR]_ [1]

2 _η_ 1 [+] _[ R][∥]_ _**[λ]**_ _[∥]_ [2] _[∥]_ **[A]** _[∥]_ _[F]_ [ +] 2



**B**, _**λ**_, _ρ_ and _L_ 2, _K_ = _[αR]_ 2 _η_ 1 [+] _[ R][∥]_ _**[λ]**_ _[∥]_ [2] _[∥]_ **[A]** _[∥]_ _[F]_ [ +] 2 [1] _ρ_ _[∥]_ _**[λ]**_ _[∥]_ 2 [2] [+]

_R∥_ **B** _∥_ _F_ _∥_ _**λ**_ _∥_ 2 + _R_ ( _L_ 2 + _∥_ **B** _[T]_ _**λ**_ _∥_ 2 ), and we have used the
fact that _∥_ **g** ˆ _t_ _∥_ [2] **G** _[−]_ _t_ [1] _≤∥_ **g** ˆ _t_ _∥_ 2 [2] [(due to] **[ G]** _[−]_ _t_ [1] _⪯_ **I** ).


Based on (31) we continue to prove Theorem 1. When
**B** is invertible and **y** _t_ _[′]_ [=] **[ B]** _[−]_ [1] [(] **[c]** _[ −]_ **[Ax]** _[t]_ [), we obtain]



+ [1]

_T_



1

_T_



Since _**λ**_ _t_ +1 _−_ _**λ**_ _t_ = _ρ_ ( **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** ), from (36)
we have



_T_



_T_
�



� ( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ _[′]_ [))] _[ −]_ _T_ [1]

_t_ =1



_T_
� ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



**B** ( **y** _t_ _[′]_ _[−]_ **[y]** _[t]_ [) =] [1] (32)

_ρ_ [(] _**[λ]**_ _[t]_ _[ −]_ _**[λ]**_ _[t][−]_ [1] [)] _[.]_


Based on the convexity of _f_ and _φ_, we obtain


_f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ _[′]_ [)] _[ ≤]_ _[f]_ _[t]_ [(] **[x]** _[t]_ [) +] _[ φ]_ [(] **[y]** _[t]_ [) +] _[ ⟨][∂φ]_ [(] **[y]** _t_ _[′]_ [)] _[,]_ **[ y]** _t_ _[′]_ _[−]_ **[y]** _[t]_ _[⟩]_

= _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ ) + [1] _t_ [)] _[,]_ _**[ λ]**_ _[t]_ _[−]_ _**[λ]**_ _[t][−]_ [1] _[⟩][,]_

_ρ_ _[⟨]_ [(] **[B]** _[−]_ [1] [)] _[T]_ _[ ∂φ]_ [(] **[y]** _[′]_

(33)



+ _[ρ]_

2 _T_



_T_
� _∥_ **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** _∥_ 2 [2]


_t_ =1



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



+ [1]

_T_



_T_
�


_t_ =1



_η_ _t_

2 [+] _[ K]_
2 _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ [2] _T_



_η_ _t_



(37)
_T_ _[.]_


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**



Taking expectations for both sides of (37) with respect
to its randomness, we have



Setting _**λ**_ _[∗]_ = ( **A** _[−]_ [1] ) _[T]_ _∇f_ _t_ ( **x** _[′]_ _t_ [), based on (43) and (34)]
we have


_f_ _t_ ( **x** _[′]_ _t_ [) +] _[ φ]_ [(] **[y]** _[t]_ [)] _[ −]_ [(] _[f]_ _[t]_ [(] **[x]** _[∗]_ [) +] _[ φ]_ [(] **[y]** _[∗]_ [))] _[ ≤]_ _[f]_ _[t]_ [(] **[x]** _[t]_ [)]

+ _φ_ ( **y** _t_ ) + (˜ **v** _t_ _−_ **v** _[∗]_ ) _[T]_ _H_ (˜ **v** _t_ ) _−_ ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ )) _._
(44)


Since the right hand side (RHS) of (44) and
RHS of (35) are same, we can then mimic the
aforementioned procedure to prove that the regret
Regret _T_ ( **x** _[′]_ _t_ _[,]_ **[ y]** _[t]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [) obeys the same bounds as (42).]


**8.3** **Simplification of Regret Bound**



_T_



�



E



1

_T_
�



_T_
�



� ( _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ _[′]_ [))] _[ −]_ _T_ [1]

_t_ =1



_T_
�



� ( _f_ _t_ ( **x** _[∗]_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



_ρ_

+ E

2 _T_
�



_T_
� _∥_ **Ax** _t_ +1 + **By** _t_ +1 _−_ **c** _∥_ 2 [2]


_t_ =1



�



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



_T_
�


_t_ =1



+ [1]

_T_



+ [1]



_η_ _t_

2 [] +] _[ K]_
2 [E][[] _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ [2] _T_



_η_ _t_



(38)
_T_ _[.]_



Based on (Duchi et al., 2015, Lemma 1), the secondorder statistics of the gradient estimate ˆ **g** _t_ is given by


E **z** _t_ [ˆ **g** _t_ ] = **g** _t_ + _β_ _t_ _L_ _g_ ( **w** _t_ ) _ν_ ( **x** _t_ _, β_ _t_ ) _,_ (39)

E **z** _t_ [ _∥_ **g** ˆ _t_ _∥_ 2 [2] []] _[ ≤]_ [2] _[s]_ [(] _[m]_ [)] _[∥]_ **[g]** _[t]_ _[∥]_ 2 [2] [+ 1] 2 _[β]_ _t_ [2] _[L]_ _[g]_ [(] **[w]** _[t]_ [)] [2] _[M]_ [(] _[µ]_ [)] [2] _[,]_ [ (40)]


where **g** _t_ = _∇_ **x** _f_ ( **x** ; **w** _t_ ) _|_ **x** = **x** _t_, _∥ν_ ( **x** _t_ _, β_ _t_ ) _∥_ 2 _≤_
1
2 [E] **[z]** [[] _[∥]_ **[z]** _[∥]_ 2 [3] [],] _[ L]_ _[g]_ [(] **[w]** _[t]_ [) is defined in Assumption C, and]
_s_ ( _m_ ) and _M_ ( _µ_ ) are introduced in Assumption E. According to (40), we have

E[ _∥_ **g** ˆ _t_ _∥_ 2 [2] [] =][ E] �E **z** [ _∥_ **g** ˆ _t_ _∥_ 2 [2] []] �

_≤_ E �2 _s_ ( _m_ ) _∥_ **g** _t_ _∥_ 2 [2] [+ 1] 2 _[β]_ _t_ [2] _[L]_ [2] _g,t_ _[M]_ [(] _[µ]_ [)] [2] �

_≤_ 2 _s_ ( _m_ ) _L_ [2] 1 [+ 1] 2 _[β]_ _t_ [2] _[L]_ [2] _g_ _[M]_ [(] _[µ]_ [)] [2] _[,]_ (41)


where for ease of notation, we have replaced _L_ _g_ ( **w** _t_ )
with _L_ _g,t_, and the last inequality holds due to Assumptions B and C.


Substituting (41) into (38), the expected average regret can be bounded as


Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)]



Consider terms in right hand side (RHS) of (42) together with _η_ _t_ = ~~_√_~~ _s_ ( _Cm_ 1 ) ~~_√_~~ _t_ [and] _[ β]_ _[t]_ [ =] _MC_ ( _µ_ 2 ) _t_ [, we have]



_C_ 2
_t_ [and] _[ β]_ _[t]_ [ =] _M_ ( _µ_ ) _t_ [, we have]



_s_ ( _m_ ) ~~_√_~~



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



1

_T_


_≤_ [1]

_T_



_T_
�



� _t_ =2 ( 2 _[α]_ _η_



� _t_ =2 max _{_ 2 _[α]_ _η_



_T_
�



_α_ 1

_[α]_ _−_ ) _R_ [2] _≤_

2 _η_ _t_ 2 _η_ _t−_ 1 ~~_√_~~



_T_



_αR_ [2] ~~[�]~~ _s_ ( _m_ )

2 _C_ 1 _,_



_s_ ( _m_ ) _L_ [2] 1
~~_√_~~ _T_



_T_ _,_



_s_ ( _m_ ) _L_ [2] 1

_T_



_T_
�



�

� _η_ _t_ _≤_ [2] _[C]_ [1]


_t_ =1



_s_ ( _m_ ) _T_



_M_ ( _µ_ ) [2] _L_ [2] _g_

4 _T_



_T_
�



2 _[L]_ [2] _g_

� _η_ _t_ _β_ _t_ [2] [=] _[C]_ [1] _[C]_ [2]

_t_ =1 4 ~~�~~ _s_ ( _m_ )



_T_
�


_t_ =1



1

_t_ [5] _[/]_ [2]



_≤_ [5] _[C]_ [1] _[C]_ 2 [2] _[L]_ [2] _g_ _,_ (45)
12 _T_



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



where we have used the facts that [�] _[T]_ _t_ =1 ~~_√_~~ 1 _t_ _[≤]_ [2] _√_



_T_,



_T_
� _η_ _t_


_t_ =1



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_




_[σ]_ 1

2 _[,]_ [ 0] _[}][R]_ [2] [ +] _[ s]_ [(] _[m]_ _T_ [)] _[L]_ [2]



_T_
�(1 _/t_ _[a]_ )


_t_ =2



(42)
_T_ _[.]_



_g_
+ _[M]_ [(] _[µ]_ [)] [2] _[L]_ [2]
4 _T_



_T_
�



� _η_ _t_ _β_ _t_ [2] [+] _[K]_ _T_

_t_ =1



_T_
�(1 _/t_ _[a]_ ) = 1 +


_t_ =1



On the other hand, when **A** is invertible and **x** _[′]_ _t_ [=]
**A** _[−]_ [1] ( **c** _−_ **By** _t_ ), we obtain

**A** ( **x** _[′]_ _t_ _[−]_ **[x]** _[t]_ [) =] [1]

_ρ_ [(] _**[λ]**_ _[t]_ _[ −]_ _**[λ]**_ _[t][−]_ [1] [)] _[.]_


Based on the convexity of _f_ and _φ_, we obtain


_f_ _t_ ( **x** _[′]_ _t_ [) +] _[ φ]_ [(] **[y]** _[t]_ [)]
_≤f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ ) + _⟨∇f_ _t_ ( **x** _[′]_ _t_ [)] _[,]_ **[ x]** _[′]_ _t_ _[−]_ **[x]** _[t]_ _[⟩]_

= _f_ _t_ ( **x** _t_ ) + _φ_ ( **y** _t_ ) + [1] _t_ [)] _[,]_ _**[ λ]**_ _[t]_ _[−]_ _**[λ]**_ _[t][−]_ [1] _[⟩][.]_

_ρ_ _[⟨]_ [(] **[A]** _[−]_ [1] [)] _[T]_ _[ ∇][f]_ _[t]_ [(] **[x]** _[′]_

(43)



_∞_
_≤_ 1 + (1 _/t_ _[a]_ ) = _a/_ ( _a −_ 1) _, ∀a >_ 1 _,_ (46)
� 1


and we recall that _s_ ( _m_ ) = _m ≥_ 1. Substituting (45)
into RHS of (42), we conclude that the expected average regret Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [) is upper bounded]
by



(47)



_s_ ( _m_ ) _L_ [2] 1
~~_√_~~ _T_



( _m_ ) _L_ [2] 1 2 _[L]_ [2] _g_

+ [5] _[C]_ [1] _[C]_ [2] + _[K]_
_T_ 12 _T_ _T_



1

~~_√_~~ _T_



_αR_ [2] ~~[�]~~




~~[�]~~

_s_ ( _m_ ) �
+ [2] _[C]_ [1]
2 _C_ 1



_T_ _[.]_


**ZOO-ADMM: Convergence Analysis and Applications**



**8.4** **Proof of Corollary 1**


Given i.i.d. samples _{_ **w** _t_ _}_ drawn from the probability
distribution _P_, from Theorem 1 we have



**8.6** **Proof of Corollary 3**


We consider the hybrid minibatch strategy



_q_ 2
�


_i_ =1



_q_ 1
�

_j_ =1



_f_ ( **x** _t_ + _β_ _t_ **z** _t,j_ ; **w** _t,i_ ) _−_ _f_ ( **x** _t_ ; **w** _t,i_ )

**z** _t,j_
_β_ _t_



ˆ 1
**g** _t_ =
_q_ 1 _q_ 2



E



1

_T_
�



_T_
� ( _f_ ( **x** _t_ ; **w** _t_ ) + _φ_ ( **y** _t_ _[′]_ [))]


_t_ =1



_T_
�



�



_−_ [1]

_T_



_T_
�



� ( _f_ ( **x** _[∗]_ ; **w** _t_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



_αR_ [2] ~~[�]~~



_s_ ( _m_ ) _L_ [2] 1
~~_√_~~ _T_



_≤_ [1]
~~_√_~~ _T_




~~[�]~~

_s_ ( _m_ ) + [2] _[C]_ [1] �
2 _C_ 1



_T_



2 _[L]_ [2] _g_ 1
+ [5] _[C]_ [1] _[C]_ [2] _[K]_ (48)
12 _T_ [+] _T_ _[.]_



(51)


with ˆ **g** _t,ij_ := _f_ ( **x** _t_ + _β_ _t_ **z** _t,j_ ; **w** _β_ _tt,i_ ) _−f_ ( **x** _t_ ; **w** _t,i_ ) **z** _t,j_ . Based on

(39) and i.i.d. samples _{_ **w** _t,i_ _}_ and _{_ **z** _t,j_ _}_, we have


**g** ¯ _t_ := E[ˆ **g** _t,ij_ ] = E[ **g** _t_ ] + _β_ _t_ E[ _L_ _g,t_ _ν_ ( **x** _t_ _, β_ _t_ )] _, ∀i, j._ (52)


where for ease of notation we have replaced _L_ _g_ ( **w** _t_ )
with _L_ _g,t_, _∥ν_ ( **x** _t_ _, β_ _t_ ) _∥_ 2 _≤_ 12 [E][[] _[∥]_ **[z]** _[∥]_ 2 [3] []] _[ ≤]_ _[M]_ [(] _[µ]_ [) due to]
Assumption E. From (51), we obtain



Based on _F_ ( **x** _,_ **y** ) = E **w** [ _f_ ( **x** ; **w** )] + _φ_ ( **y** ), from (48) we
have


E [ _F_ (¯ **x** _t_ _,_ ¯ **y** _t_ ) _−_ _F_ ( **x** _[∗]_ _,_ **y** _[∗]_ )]



2



_q_ 2


¯

�(ˆ **g** _t,ij_ _−_ **g** _t_ ) + ¯ **g** _t_

_j_ =1



������



2



_q_ 1
�


_i_ =1



�









������



1

_q_ 1 _q_ 2



E[ _∥_ **g** ˆ _t_ _∥_ 2 [2] [] =][E]



_≤_ E



1

_T_
�



_T_
�



� _F_ ( **x** _t_ _,_ **y** _t_ ) _−_ _F_ ( **x** _[∗]_ _,_ **y** _[∗]_ )


_t_ =1









1

_T_
�










2



_q_ 2
�



¯

�(ˆ **g** _t,ij_ _−_ **g** _t_ )

_j_ =1



������



_q_ 1
�


_i_ =1



=E **z** 1: _T_


_−_ [1]

_T_



E **w** 1: _T_

�



� ( _f_ ( **x** _[∗]_ ; **w** _t_ ) + _φ_ ( **y** _[∗]_ ))


_t_ =1



_T_
� ( _f_ ( **x** _t_ ; **w** _t_ ) + _φ_ ( **y** _t_ _[′]_ [))]


_t_ =1



= _∥_ **g** ¯ _t_ _∥_ 2 [2] [+][ E]



������



1

_q_ 1 _q_ 2



2









_T_
�



��



_s_ ( _m_ ) _L_ [2] 1
~~_√_~~ _T_



_≤_ [1]
~~_√_~~ _T_



_αR_ [2] ~~[�]~~




~~[�]~~

_s_ ( _m_ ) + [2] _[C]_ [1] �
2 _C_ 1



_T_



2 _[L]_ [2] _g_ 1
+ [5] _[C]_ [1] _[C]_ [2] _[K]_ (49)
12 _T_ [+] _T_ _[,]_



where the first inequality holds due to the convexity of
_F_, and the second equality holds since **x** _t_ and **y** _t_ are
implicit functions of i.i.d. random variables _{_ **w** _k_ _}_ _[t]_ _k_ _[−]_ =1 [1]
and _{_ **z** _k_ _}_ _[t]_ _k_ _[−]_ =1 [1] [, and] _[ {]_ **[w]** _[t]_ _[}]_ [ and] _[ {]_ **[z]** _[t]_ _[}]_ [ are independent of]
each other.


**8.5** **Proof of Corollary 2**


Substituting _η_ _t_ = _σt_ _[α]_ [and] _[ β]_ _[t]_ [ =] _MC_ ( _µ_ 2 ) _t_ [into RHS of (42),]

we have



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2] [ = 0] _[,]_



1

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



_T_

1 [lo][g] _[ T]_

� _η_ _t_ _≤_ _[αs]_ [(] _[m]_ [)] _σT_ _[L]_ [2] _,_

_t_ =1



_T_
�



_s_ ( _m_ ) _L_ [2] 1

_T_



_T_
�


_t_ =1



1 2 _[L]_ [2] _g_
_t_ [3] _[≤]_ [3] _[αC]_ 8 _σT_ [2] _,_


(50)



_M_ ( _µ_ ) [2] _L_ [2] _g_

4 _T_



_T_

2 _[L]_ [2] _g_

� _η_ _t_ _β_ _t_ [2] [=] _[ αC]_ 4 _σT_ [2]

_t_ =1



¯ 1 ˆ ¯
= _∥_ **g** _t_ _∥_ 2 [2] [+] E[ _∥_ **g** _t,_ 11 _−_ **g** _t_ _∥_ [2] 2 [] =] _[ ∥]_ **[g]** [¯] _[t]_ _[∥]_ [2]
_q_ 1 _q_ 2

+ 1 E[ _∥_ **g** ˆ _t,_ 11 _∥_ [2] ] _−_ 1 _∥_ **g** ¯ _t_ _∥_ [2] _,_ (53)
_q_ 1 _q_ 2 _q_ 1 _q_ 2


where we have used the fact that E[ˆ **g** _t,ij_ ] = E[ˆ **g** _t,_ 11 ] for
any _i_ and _j_ .


The definition of ¯ **g** _t_ in (52) yields


_∥_ **g** ¯ _t_ _∥_ [2] _≤_ 2 _∥_ E[ **g** _t_ ] _∥_ 2 [2] [+ 2] _[∥][β]_ _[t]_ [E][[] _[L]_ _[g,t]_ _[ν]_ [(] **[x]** _[t]_ _[, β]_ _[t]_ [)]] _[∥]_ 2 [2]
_≤_ 2E[ _∥_ **g** _t_ _∥_ 2 [2] [] + 2] _[β]_ _t_ [2] [E][[] _[L]_ [2] _g,t_ []][E][[] _[∥][ν]_ [(] **[x]** _[t]_ _[, β]_ _[t]_ [)] _[∥]_ 2 [2] []]

_≤_ 2E[ _∥_ **g** _t_ _∥_ 2 [2] [] + 1] 2 _[β]_ _t_ [2] _[L]_ [2] _g_ _[M]_ [(] _[µ]_ [)] [2] _[,]_ (54)


where the first inequality holds due to Cauchy-Schwarz
inequality, and the second inequality holds due to
Jensen’s inequality. From (40), we obtain


E[ _∥_ **g** ˆ _t,_ 11 _∥_ [2] ] _≤_ 2 _s_ ( _m_ )E[ _∥_ **g** _t_ _∥_ 2 [2] [] + 1] 2 _[β]_ _t_ [2] _[L]_ [2] _g_ _[M]_ [(] _[µ]_ [)] [2] _[.]_ (55)


Substituting (54) and (55) into (53), we obtain


E[ _∥_ **g** ˆ _t_ _∥_ 2 [2] []] _[ ≤∥]_ **[g]** [¯] _[t]_ _[∥]_ [2] 2 [+] 1 E[ _∥_ **g** ˆ _t,_ 11 _∥_ [2] 2 []]
_q_ 1 _q_ 2



_≤_ 2(1 + _[s]_ [(] _[m]_ [)]




[(] _[m]_ [)]

)E[ _∥_ **g** _t_ _∥_ 2 [2] [] +] _[q]_ [1] _[q]_ [2] [ + 1]
_q_ 1 _q_ 2 2 _q_ 1 _q_ 2



2 _q_ 1 _q_ 2 _β_ _t_ [2] _[L]_ [2] _g_ _[M]_ [(] _[µ]_ [)] [2] _[.]_ [ (56)]



where we have used the facts that [�] _[T]_ _t_ =1 1 _t_ _[≤]_ [1 + log] _[ T]_
and (46). Based on (50) and (47), we complete the
proof.



Similar to proof of Theorem 1, substituting (56) into


(38), we obtain



**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**


are given by



Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)]



_∥_ **x** _−_ **d** _t_ _∥_ [2] 2 _,_ (60)
� �


_∥_ **y** _−_ ( **x** _t_ +1 _−_ (1 _/ρ_ ) _**λ**_ _t_ ) _∥_ [2] 2 _,_ (61)
� �



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



**x** _t_ +1 = arg min
**0** _≤_ **x** _≤_ **1**


**y** _t_ +1 = arg min
**1** _[T]_ **y** = _m_ 0



+ [1]

_T_



_T_
�


_t_ =1



_η_ _t_ 2 [] +] _[K]_

2 [E][[] _[∥]_ **[g]** [ˆ] _[t]_ _[∥]_ [2] _T_



_T_



_η_ _t_



_α_

_[α]_ _−_ _−_ _[σ]_

2 _η_ _t_ 2 _η_ _t−_ 1 2



2 _[,]_ [ 0] _[}][R]_ [2]



_≤_ [1]

_T_



_T_
�



� _t_ =2 max _{_ 2 _[α]_ _η_



where **g** ˆ _t_ is the gradient estimate, and **d** _t_ :=
_ηα_ _t_ [(] _[−]_ **[g]** [ˆ] _[t]_ [ +] _**[ λ]**_ _[t]_ _[ −]_ _[ρ]_ **[x]** _[t]_ [ +] _[ ρ]_ **[y]** _[t]_ [)+] **[x]** _[t]_ [. Sub-problems (60) and]

(61) yield closed-form solutions as below (Parikh and
Boyd, 2014)



+ [(] _[q]_ [1] _[q]_ [2] [ +] _[ s]_ [(] _[m]_ [))] _[L]_ 1 [2]
_q_ 1 _q_ 2 _T_



_T_
� _η_ _t_


_t_ =1




[ **x** _t_ +1 ] _i_ =










0 [ **d** _t_ ] _i_ _<_ 0

[ **d** _t_ ] _i_ [ **d** _t_ ] _i_ _∈_ [0 _,_ 1] and (62)
1 [ **d** _t_ ] _i_ _>_ 1 _,_



_T_
�



(57)
_T_ _[.]_



**1** _m_ _,_
_m_



_g_ _[M]_ [(] _[µ]_ [)] [2]
+ [(] _[q]_ [1] _[q]_ [2] [ + 1)] _[L]_ [2]

4 _q_ 1 _q_ 2 _T_



_g_ _[M]_ [(] _[µ]_ [)] [2]
+ [(] _[q]_ [1] _[q]_ [2] [ + 1)] _[L]_ [2]



� _η_ _t_ _β_ _t_ [2] [+] _[ K]_ _T_

_t_ =1



(63)


where [ **x** ] _i_ denote the _i_ th entry of **x** .


**8.8** **ZOO-ADMM for Sparse Cox Regression**


This sparse regression problem can formulated as



**y** _t_ +1 = **x** _t_ +1 _−_ [1]




[1] [(] **[x]** _[t]_ [+1] _[ −]_ _**[λ]**_ _[t]_ _[/ρ]_ [)]

_ρ_ _**[λ]**_ _[t]_ [ +] _[ m]_ [0] _[ −]_ **[1]** _[T]_ _m_



Substituting _η_ _t_ = _C_ 1
~~�~~ 1+ _[s]_ _q_ [(] 1



1+ _[s]_ [(] _[m]_ [)]



_C_ 2
_t_ [and] _[ β]_ _[t]_ [ =] _M_ ( _µ_ ) _t_ [into]




_[s]_ _q_ 1 _[m]_ _q_ 2 ~~_√_~~



(57), we obtain


Regret _T_ ( **x** _t_ _,_ **y** _t_ _[′]_ _[,]_ **[ x]** _[∗]_ _[,]_ **[ y]** _[∗]_ [)]



_q_ 1 _q_ 2
~~_√_~~ _T_



_q_ 1 _q_ 2 + 2 _C_ 1 _L_ [2] 1

_T_



_q_ 1 _q_ 2
~~_√_~~ _T_



_≤_ _[αR]_ [2]

2 _C_ 1



�



1 + _[s]_ [(] _[m]_ [)]



~~�~~



1 + _[s]_ [(] _[m]_ [)]



1
minimize
**x** _,_ **y** _n_



**x** _,_ **y** _n_ _i_ =1 (64)

subject to **x** _−_ **y** = **0** _,_



_T_



_n_
�



� _f_ ( **x** ; **w** _i_ ) + _γ∥_ **y** _∥_ 1


_i_ =1



+ [5] _[C]_ [1] _[C]_ 2 [2] _[L]_ [2] _g_ _q_ 1 _q_ 2 + 1
12 _T_

_[s]_



+ _[K]_

_T_



_q_ 1 _q_ 2
~~�~~



+ _[K]_



1 + _[s]_ [(] _[m]_ [)]



_q_ 1 _q_ 2



_q_ 1 _q_ 2
~~_√_~~ _T_



_q_ 1 _q_ 2 + 2 _C_ 1 _L_ [2] 1

_T_



_q_ 1 _q_ 2
~~_√_~~ _T_



_≤_ _[αR]_ [2]

2 _C_ 1



�



1 + _[s]_ [(] _[m]_ [)]



~~�~~



1 + _[s]_ [(] _[m]_ [)]



_T_



2 _[L]_ [2] _g_ 1
+ [5] _[C]_ [1] _[C]_ [2] _[K]_ (58)
6 _T_ [+] _T_ _[,]_



where _f_ ( **x** ; **w** _i_ ) = _δ_ _i_ � _−_ **a** _[T]_ _i_ **[x]** [ + log (][�] _j∈R_ _i_ _[e]_ **[a]** _j_ _[T]_ **[x]** )� with

**w** _i_ = **a** _i_ . By using the ZOO-ADMM algorithm, we
can avoid the gradient calculation for the involved objective function in Cox regression. The two key steps
of ZOO-ADMM (22)-(23) at iteration _i_ become



which then completes the proof.


**8.7** **ZOO-ADMM for Sensor Selection**


We recall that the sensor selection problem can be cast

as



**x** _i_ +1 = _[η]_ _α_ _[t]_ [(] _[−]_ **[g]** [ˆ] _[i]_ [ +] _**[ λ]**_ _[i]_ _[ −]_ _[ρ]_ **[x]** _[i]_ [ +] _[ ρ]_ **[y]** _[i]_ [) +] **[ x]** _[i]_ _[,]_ (65)



_,_ (66)
�



**y** _i_ +1 = arg min

**y**



� _∥_ **y** _∥_ 1 + 2 _[ρ]_ _γ_ _[∥]_ **[y]** _[ −]_ **[d]** _[i]_ _[∥]_ 2 [2]



_∥_ **y** _∥_ 1 + _[ρ]_
� 2



1
minimize
**x** _,_ **y** _T_



_T_
� _f_ ( **x** ; **w** _t_ ) + _I_ 1 ( **x** ) + _I_ 2 ( **y** )


_t_ =1



(59)



ˆ
where **g** _i_ is the gradient estimate, **d** _i_ =
( **x** _i_ +1 _−_ (1 _/ρ_ ) _**λ**_ _i_ ), and the solution of sub-problem
(66) is given by the soft-thresholding operator at the
point **d** _i_ with parameter _ρ/γ_ (Parikh and Boyd, 2014,
Sec. 6)



subject to **x** _−_ **y** = **0** _,_


where **y** _∈_ R _[m]_ is an auxiliary variable, _f_ ( **x** ; **w** _t_ ) =

_−_
logdet( [�] _[m]_ _i_ =1 _[x]_ _[i]_ **[a]** _[i,t]_ **[a]** _[T]_ _i,t_ [) with] **[ w]** _[t]_ [ =] _[ {]_ **[a]** _[i,t]_ _[}]_ _i_ _[m]_ =1 [, and] _[ {I]_ _[i]_ _[}]_
are indicator functions


0 **0** _≤_ **x** _≤_ **1** 0 **1** _[T]_ **y** = _m_ 0
_I_ 1 ( **x** ) = _I_ 2 ( **y** ) =
� _∞_ otherwise _,_ � _∞_ otherwise _._


Based on (59), two key steps of ZOO-ADMM (22)-(23)



(1 _−_ _ρ|_ [ **d** _γ_ _i_ ] _k_ _|_ [)[] **[d]** _[i]_ []] _[k]_ [ **d** _i_ ] _k_ _>_ _[γ]_ _ρ_

[ **y** _i_ +1 ] _k_ =
� 0 [ **d** _i_ ] _k_ _≤_ _[γ]_



_ρ_ _i_ _k_ _ρ_

0 [ **d** _i_ ] _k_ _≤_ _[γ]_



_ρ_ _[,]_



for _k_ = 1 _,_ 2 _, . . ., m_ .


**References**


A. Agarwal, O. Dekel, and L. Xiao. Optimal algorithms for online convex optimization with multipoint bandit feedback. In _COLT_, pages 28–40, 2010.


**ZOO-ADMM: Convergence Analysis and Applications**



F. Bach, R. Jenatton, J. Mairal, and G. Obozinski. Optimization with sparsity-inducing penalties.
_Foundations and Trends⃝_ R _in Machine Learning_, 4
(1):1–106, 2012.


D. Blatt, A. O. Hero, and H. Gauchman. A convergent
incremental gradient method with a constant step
size. _SIAM Journal on Optimization_, 18(1):29–51,
2007.


S. Boyd and L. Vandenberghe. _Convex optimization_ .
Cambridge university press, 2004.


S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. _Foundations and Trends⃝_ R _in Machine Learn-_
_ing_, 3(1):1–122, 2011.


S. Bubeck et al. Convex optimization: Algorithms and
complexity. _Foundations and Trends⃝_ R _in Machine_
_Learning_, 8(3-4):231–357, 2015.


E. J. Cand`es and B. Recht. Exact matrix completion
via convex optimization. _Foundations of Computa-_
_tional mathematics_, 9(6):717, 2009.


P.-L. Chen, C.-T. Tsai, Y.-N. Chen, K.-C. Chou, C.L. Li, C.-H. Tsai, K.-W. Wu, Y.-C. Chou, C.-Y. Li,
W.-S. Lin, et al. A linear ensemble of individual
and blended models for music rating prediction. In
_Proceedings of the 2011 International Conference on_
_KDD Cup_, pages 21–60. JMLR. org, 2011.


P.-Y. Chen and S. Liu. Bias-variance tradeoff of graph
laplacian regularizer. _IEEE Signal Processing Let-_
_ters_, 2017.


P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.J. Hsieh. Zoo: Zeroth order optimization based
black-box attacks to deep neural networks without training substitute models. _arXiv preprint_
_arXiv:1708.03999_, 2017.


A. Cotter, O. Shamir, N. Srebro, and K. Sridharan.
Better mini-batch algorithms via accelerated gradient methods. In _Advances in neural information_
_processing systems_, pages 1647–1655, 2011.


D. R. Cox. Regression models and life-tables. _Journal_
_of the Royal Statistical Society. Series B (Method-_
_ological)_, 34(2):187–220, 1972.


G. Dror, N. Koenigstein, Y. Koren, and M. Weimer.
The yahoo! music dataset and kdd-cup11. In _Pro-_
_ceedings of KDD Cup 2011_, pages 3–18, 2012.


J. Duchi, S. Shalev-Shwartz, Y. Singer, and A. Tewari.
Composite objective mirror descent. In _COLT_,
pages 14–26, 2010.


J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. _Journal of Machine Learning Research_,
12(Jul):2121–2159, 2011.



J. C. Duchi, M. I. Jordan, M. J. Wainwright, and
A. Wibisono. Optimal rates for zero-order convex
optimization: The power of two function evaluations. _IEEE Transactions on Information Theory_,
61(5):2788–2806, 2015.


X. Gao, B. Jiang, and S. Zhang. On the informationadaptive variants of the admm: An iteration complexity perspective. _Journal of Scientific Comput-_
_ing_, Dec 2017. ISSN 1573-7691.


S. Ghadimi and G. Lan. Stochastic first-and zerothorder methods for nonconvex stochastic programming. _SIAM Journal on Optimization_, 23(4):2341–
2368, 2013.


D. Hajinezhad, M. Hong, and A. Garcia. Zenith: A
zeroth-order distributed algorithm for multi-agent
nonconvex optimization. 2017.


E. C. Hall and R. M. Willett. Online convex optimization in dynamic environments. _IEEE Journal of_
_Selected Topics in Signal Processing_, 9(4):647–662,
June 2015. ISSN 1932-4553.


E. Hazan. Introduction to online convex optimization.
_Foundations and Trends⃝_ R _in Optimization_, 2(3-4):
157–325, 2016.


A. O. Hero and D. Cochran. Sensor management:
Past, present, and future. _IEEE Sensors Journal_,
11(12):3064–3075, 2011.


S. Hosseini, A. Chapman, and M. Mesbahi. Online distributed convex optimization on dynamic networks.
_IEEE Transactions on Automatic Control_, 61(11):
3545–3550, 2016.


S. Joshi and S. Boyd. Sensor selection via convex optimization. _IEEE Transactions on Signal Processing_,
57(2):451–462, 2009.


M. Li, T. Zhang, Y. Chen, and A. J. Smola. Efficient
mini-batch training for stochastic optimization. In
_Proceedings of the 20th ACM SIGKDD international_
_conference on Knowledge discovery and data mining_,
pages 661–670. ACM, 2014.


X. Lian, H. Zhang, C.-J. Hsieh, Y. Huang, and
J. Liu. A comprehensive linear speedup analysis for
asynchronous stochastic parallel optimization from
zeroth-order to first-order. In _Advances in Neural_
_Information Processing Systems_, pages 3054–3062,
2016.


F. Lin, M. Fardad, and M. R. Jovanovic. Algorithms
for leader selection in stochastically forced consensus
networks. _IEEE Transactions on Automatic Con-_
_trol_, 59(7):1789–1802, 2014.


S. Liu, S. P. Chepuri, M. Fardad, E. Ma¸sazade,
G. Leus, and P. K. Varshney. Sensor selection
for estimation with correlated measurement noise.
_IEEE Transactions on Signal Processing_, 64(13):
3509–3522, 2016.


**Sijia Liu, Jie Chen, Pin-Yu Chen, Alfred O. Hero**


Y. Nesterov and V. Spokoiny. Random gradient-free
minimization of convex functions. _Foundations of_
_Computational Mathematics_, 2(17):527–566, 2015.


H. Ouyang, N. He, L. Tran, and A. Gray. Stochastic
alternating direction method of multipliers. In _In-_
_ternational Conference on Machine Learning_, pages
80–88, 2013.


N. Parikh and S. Boyd. Proximal algorithms. _Founda-_
_tions and Trends⃝_ R _in Optimization_, 1(3):127–239,
2014.


M. Y. Park and T. Hastie. L1-regularization path algorithm for generalized linear models. _Journal of_
_the Royal Statistical Society: Series B (Statistical_
_Methodology)_, 69(4):659–677, 2007.


C. R. Rao. _Linear statistical inference and its applica-_
_tions_, volume 2. Wiley New York, 1973.


N. L. Roux, M. Schmidt, and F. R. Bach. A stochastic
gradient method with an exponential convergence

~~r~~ ate for finite training sets. In _Advances in Neural_
_Information Processing Systems_, pages 2663–2671,
2012.


S. Shalev-Shwartz. Online learning and online convex optimization. _Foundations and Trends_ R _⃝_ _in Ma-_
_chine Learning_, 4(2):107–194, 2012.


O. Shamir. An optimal algorithm for bandit and zeroorder convex optimization with two-point feedback.
_Journal of Machine Learning Research_, 18(52):1–11,
2017.


I. Sohn, J. Kim, S.-H. Jung, and C. Park. Gradient
lasso for cox proportional hazards model. _Bioinfor-_
_matics_, 25(14):1775–1781, 2009.


T. Suzuki. Dual averaging and proximal gradient
descent for online alternating direction multiplier
method. In _International Conference on Machine_
_Learning_, pages 392–400, 2013.


H. Wang and A. Banerjee. Online alternating direction method (longer version). _arXiv preprint_
_arXiv:1306.3721_, 2013.


D. M. Witten and R. Tibshirani. Survival analysis with
high-dimensional covariates. _Statistical methods in_
_medical research_, 19(1):29–51, 2010.


L. Xiao. Dual averaging methods for regularized
stochastic learning and online optimization. _Journal_
_of Machine Learning Research_, 11(Oct.):2543–2596,
2010.


X. Zhang, M. Burger, and S. Osher. A unified primaldual algorithm framework based on bregman iteration. _Journal of Scientific Computing_, 46(1):20–46,
2011.


