## Fast and Complete: Enabling Complete Neural Network Veri- fication with Rapid and Massively Parallel Incomplete Verifiers

**Kaidi Xu** **[*,]** [1] **Huan Zhang** **[*,]** [2] **Shiqi Wang** [3] **Yihan Wang** [2]

**Suman Jana** [3] **Xue Lin** [1] **Cho-Jui Hsieh** [2]


1 Northeastern University 2 UCLA 3 Columbia University


xu.kaid@northeastern.edu, huan@huan-zhang.com, tcwangshiqi@cs.columbia.edu,

wangyihan617@gmail.com, suman@cs.columbia.edu, xue.lin@northeastern.edu,

chohsieh@cs.ucla.edu


A BSTRACT

Formal verification of neural networks (NNs) is a challenging and important problem. Existing efficient complete solvers typically require the branch-and-bound
(BaB) process, which splits the problem domain into sub-domains and solves each
sub-domain using faster but weaker incomplete verifiers, such as Linear Programming (LP) on linearly relaxed sub-domains. In this paper, we propose to use the
backward mode linear relaxation based perturbation analysis (LiRPA) to replace
LP during the BaB process, which can be efficiently implemented on the typical machine learning accelerators such as GPUs and TPUs. However, unlike LP,
LiRPA when applied naively can produce much weaker bounds and even cannot
check certain conflicts of sub-domains during splitting, making the entire procedure incomplete after BaB. To address these challenges, we apply a fast gradient
based bound tightening procedure combined with batch splits and the design of
minimal usage of LP bound procedure, enabling us to effectively use LiRPA on
the accelerator hardware for the challenging complete NN verification problem
and significantly outperform LP-based approaches. On a single GPU, we demonstrate an order of magnitude speedup compared to existing LP-based approaches.


1 I NTRODUCTION


Although neural networks (NNs) have achieved great success on various complicated tasks, they
remain susceptible to adversarial examples (Szegedy et al., 2013): imperceptible perturbations of
test samples might unexpectedly change the NN predictions. Therefore, it is crucial to conduct
formal verification for NNs such that they can be adopted in safety or security-critical settings.


Formally, the neural network verification problem can be cast into the following decision problem:


_Given a neural network f_ ( _·_ ) _, an input domain C, and a property P. ∀x ∈C, does f_ ( _x_ ) _satisfy P?_


The property _P_ is typically a set of desirable outputs of the NN conditioned on the inputs. Typically,
consider a binary classifier _f_ ( _x_ ) and a positive example _x_ 0 ( _f_ ( _x_ 0 ) _≥_ 0), we can set _P_ to be nonnegative numbers R [+] and _x_ is bounded within an _l_ _∞_ norm ball _C_ = _{x|∥x −_ _x_ 0 _∥_ _∞_ _≤_ _ϵ}_ . The
success of verification guarantees that the label of _x_ 0 cannot flip for any perturbed inputs within _C_ .


In this paper we study the _complete verification_ setting, where given sufficient time, the verifier
should give a definite “yes/no” answer for a property under verification. In the above setting, it
must solve the non-convex optimization problem min _x∈C_ _f_ ( _x_ ) to a global minimum. Complete NN
verification is generally a challenging NP-Hard problem (Katz et al., 2017) which usually requires
expensive formal verification methods such as SMT (Katz et al., 2017) or MILP solvers (Tjeng
et al., 2019b). On the other hand, incomplete solvers such as convex relaxations of NNs (Salman
et al., 2019) can only provide a sound analysis, i.e., they can only approximate the lower bound of
min _x_ _∈_ _C_ _f_ ( _x_ ) as _f_ and verify the property when _f ≥_ 0. No conclusion can be drawn when _f <_ 0.


Recently, a Branch and Bound (BaB) style framework (Bunel et al., 2018; 2020b) has been adopted
for efficient complete verification. BaB solves the optimization problem min _x∈C_ _f_ ( _x_ ) to a global


   - Equal Contribution.


1


minimum by branching into multiple sub-domains recursively and bounding the solution for each
sub-domain using incomplete verifiers. BaB typically uses a Linear Programming (LP) bounding
procedure as an incomplete verifier to provide feasibility checking and relatively tight bounds for
each sub-domain. However, the relatively high solving cost of LPs and incapability of parallelization (especially on massively parallel hardware accelerators like GPUs or TPUs) greatly limit the
performance and scalability of the existing complete BaB based verifiers.


In this paper, we aim to use fast and typically weak incomplete verifiers for complete verification.
Specifically, we focus on a class of incomplete verifiers using efficient bound propagation operations, referred to as linear relaxation based perturbation analysis (LiRPA) algorithms (Xu et al.,
2020). Representative algorithms in this class include convex outer adversarial polytope (Wong
& Kolter, 2018), CROWN (Zhang et al., 2018) and DeepPoly (Singh et al., 2019b). LiRPA algorithms exhibit high parallelism as the bound propagation process is similar to forward or backward
propagation of NNs, which can fully exploit machine learning accelerators (e.g., GPUs and TPUs).


Although LiRPA bounds are very efficient for incomplete verification, especially in training certified
adversarial defenses (Wong et al., 2018; Mirman et al., 2018; Wang et al., 2018a; Zhang et al., 2020),
they are generally considered too loose to be useful compared to LPs in the complete verification
settings with BaB. As we will demonstrate later, using LiRPA bounds naively in BaB cannot even
guarantee the completeness when splitting ReLU nodes, and thus we need additional measures to
make them useful for complete verification. In fact, LiRPA methods have been used to get upper and
lower bounds for each ReLU neuron in constructing tighter LPs (Bunel et al., 2018; Lu & Kumar,
2020). It was also used in (Wang et al., 2018c) for verifying small-scale problems with relatively low
dimensional input domains using input splits, but splitting the input space can be quite ineffective
(Bunel et al., 2018) and is unable to scale to high dimensional input case like CIFAR-10. Except
one concurrent work (Bunel et al., 2020a), most complete verifiers are based on relatively expensive
solvers like LP and cannot fully take benefit from massively parallel hardware (e.g., GPUs) to obtain
tight bounds for accelerating large-scale complete verification problems. Our main contributions are:


- We show that LiRPA bounds, when improved with fast gradient optimizers, can potentially outperform bounds obtained by LP verifiers. This is because LiRPA allows joint optimization of both
intermediate layer bounds of ReLU neurons (which determine the tightness of relaxation) and output
bounds, while LP can only optimize output bounds with fixed relaxations on ReLU neurons.

- We show that BaB purely using LiRPA bounds is insufficient for complete verification due to the
lack of feasibility checking for ReLU node splits. To address this issue, we design our algorithm to
only invoke LP when absolutely necessary and exploits hardware parallelism when possible.

- To fully exploit the hardware parallelism on the machine learning accelerators, we use a batch
splitting approach that splits on multiple neurons simultaneously, further improving our efficiency.

- On a few standard and representative benchmarks, our proposed NN verification framework
can outperform previous baselines significantly, with a speedup of around 30X compared to basic
BaB+LP baselines, and up to 3X compared to recent state-of-the-art complete verifiers.


2 B ACKGROUND


**2.1** **Formal definition of Neural network (NN) verification**


**Notations of NN.** For illustration, we define an _L_ -layer feedforward NN _f_ : R _[|][x][|]_ _→_ R with _L_
weights **W** [(] _[i]_ [)] ( _i ∈{_ 1 _, · · ·, L}_ ) recursively as _h_ [(] _[i]_ [)] ( _x_ ) = **W** [(] _[i]_ [)] _g_ [(] _[i][−]_ [1)] ( _x_ ), hidden layer _g_ [(] _[i]_ [)] ( _x_ ) =
ReLU( _h_ [(] _[i]_ [)] ( _x_ )), input layer _g_ [(0)] ( _x_ ) = _x_, and final output _f_ ( _x_ ) = _h_ [(] _[L]_ [)] ( _x_ ). For simplicity we ignore
biases. We sometimes omit _x_ and use _h_ [(] _j_ _[i]_ [)] to represent the _pre-activation_ of the _j_ -th ReLU neuron

in _i_ -th layer for _x ∈C_, and we use _g_ _j_ [(] _[i]_ [)] to represent the post-activation value. We focus on verifying
ReLU based NNs, but our method is generalizable to other activation functions supported by LiRPA.


**NN Verification Problem.** Given an input _x_, its bounded input domain _C_, and a feedforward NN
_f_ ( _·_ ), the aim of formal verification is to prove or disprove certain properties _P_ of NN outputs. Since
most properties studied in previous works can be expressed as a Boolean expression over a linear
equation of network output, where the linear property can be merged into the last layer weights of a
NN, the ultimate goal of complete verification reduces to prove or disprove:


_∀x ∈C, f_ ( _x_ ) _≥_ 0 (1)


2


One way to prove Eq. 1 is to solve min _x∈C_ _f_ ( _x_ ). Due to the non-convexity of NNs, finding the
exact minimum of _f_ ( _x_ ) over _x ∈C_ is challenging as the optimization process is generally NPcomplete (Katz et al., 2017). However, in practice, a _sound_ approximation of the lower bound for
_f_ ( _x_ ), denoted as _f_, can be more easily obtained and is sufficient to verify the property. Thus, a good
verification strategy to get a tight approximation _f_ can save significant time cost. Note that _f_ must
be sound, i.e., _∀x ∈C, f ≤_ _f_ ( _x_ ), proving _f ≥_ 0 is sufficient to prove the property _f_ ( _x_ ) _≥_ 0.


**2.2** **The Branch and Bound (BaB) framework for Neural Network Verification**


Branch and Bound (BaB), an effective strategy in solving traditional combinatorial optimization
problems, has been customized and widely adopted for NN verification (Bunel et al., 2018; 2020b).
Specifically, BaB based verification framework is a recursive process, consisting of two main steps:
branching and bounding. For _branching_, BaB based methods will divide the bounded input domain
_C_ into sub-domains _{C_ _i_ _|C_ = [�] _i_ _[C]_ _[i]_ _[}]_ [, each defined as a new independent verification problem. For]

instance, it can split a ReLU unit _g_ _j_ [(] _[k]_ [)] = ReLU( _h_ [(] _j_ _[k]_ [)] [)][ to be negative and positive cases as] _[ C]_ [0] [ =]

_C ∩_ � _h_ [(] _j_ _[k]_ [)] _≥_ 0� and _C_ 1 = _C ∩_ � _h_ [(] _j_ _[k]_ [)] _<_ 0� for a ReLU-based network; for each sub-domain _C_ _i_,
BaB based methods perform _bounding_ to obtain a relaxed but sound lower bound _f_ ~~_C_~~ _i_ . A tightened
global lower bound over _C_ can then be obtained by taking the minimum values of the sub-domain
lower bounds from all the sub-domains: _f_ = min _i_ _f_ ~~_C_~~ _i_ . Branching and bounding will be performed
recursively to tighten the approximated global lower bound over _C_ until either (1) the global lower
bound _f_ becomes larger than 0 and prove the property or (2) a violation (e.g., adversarial example)
is located in a sub-domain to disprove the property. Essentially, we build a search tree where each
leaf is a sub-domain, and the property _P_ can be proven only when it is valid on all leaves.


**Soundness of BaB** We say the verification process is sound if we can always trust the “yes” ( _P_
is verified) answer given by the verifier. It is straightforward to see that the whole BaB based
verification process is sound as long as the bounding method used for each sub-domain _C_ _i_ is sound.


**Completeness of BaB** The completeness of the BaB-based NN verification process, which was
usually assumed true in some previous works (Bunel et al., 2020b; 2018), in fact, is not always
true even if all possible sub-domains are considered with a sound bounding method. Additional
requirements for the bounding method are required - we point out that a key factor for completeness
involves feasibility checking in the bounding method which we will discuss in Section 3.2.


**Branching in BaB** Since branching step determines the shape of the search tree, the main challenge is to efficiently choose a good leaf to split, which can significantly reduce the total number of
branches and running time. In this work we focus on branching on activation (ReLU) nodes. BaBSR
(Bunel et al., 2018) includes a simple branching heuristic which assigns each ReLU node a score to
estimate the improvement for tightening _f_ by splitting it, and splits the node with the highest score.


**Bounding with Linear Programming (LP)** A typical bounding method used in BaB based verification is the _Linear Programming bounding procedure_ (sometimes simply referred to as “LP” or
“LP verifier” in our paper). Specifically, we transform the original verification problem into a linear
programming problem by relaxing every activation unit as a convex (linear) domain (Ehlers, 2017)
and then get the lower bound _f_ _C_ _i_ with a linear solver given domain _C_ _i_ . For instance, as shown in

Figure 1a, _g_ _j_ [(] _[i]_ [)] = ReLU( _h_ [(] _j_ _[i]_ [)] [)][ can be linearly relaxed with the following 3 constraints:] [ (1)] _[ g]_ _j_ [(] _[i]_ [)] _≥_ _h_ [(] _j_ _[i]_ [)] [;]

_**u**_ [(] _[i]_ [)]
(2) _g_ _j_ [(] _[i]_ [)] _≥_ 0; (3) _g_ _j_ [(] _[i]_ [)] _≤_ _**u**_ [(] _[i]_ [)] _j_ _−_ _**l**_ [(] _[i]_ [)] ( _h_ [(] _j_ _[i]_ [)] _−_ _**l**_ _j_ [(] _[i]_ [)] [)] [. Note that the lower bound] _**[ l]**_ _j_ [(] _[i]_ [)] and upper bound _**u**_ [(] _j_ _[i]_ [)] for

_j_ _j_

each activation node _h_ _j_ [(] _[i]_ [)] are required in the LP construction given _C_ _i_ . They are typically computed
by the existing cheap bounding methods like LiRPA variants (Wong & Kolter, 2018) with low cost.
The tighter the intermediate bounds ( _**l**_ _j_ [(] _[i]_ [)] [,] _**[ u]**_ [(] _j_ _[i]_ [)] [) are, the tighter] _[f]_ [ approximated by LP is.]


**2.3** **Linear Relaxation based Perturbation Analysis (LiRPA)**


**Bound propagation in LiRPA** We used Linear Relaxation based Perturbation Analysis (LiRPA) [1]
as bound procedure in BaB to get linear upper and lower bounds of NN output w.r.t input _x ∈C_ :
**A** _x_ + **b** _≤_ _f_ ( _x_ ) _≤_ **A** _x_ + **b** _,_ _x ∈C_ (2)


1 We only use the backward mode LiRPA bounds (e.g., CROWN and DeepPoly) in this paper.


3


_j_



_h_ [(] _[i]_ [)]



_h_ [(] _[i]_ [)]



_h_ [(] _[i]_ [)]

_j_



_h_ [(] _[i]_ [)]











_j_



_j_



_j_












|Col1|Col2|Col3|Col4|
|---|---|---|---|
|~~(~~~~_i_)~~<br>_j_|~~(~~~~_i_)~~<br>_j_|**u**|**u**|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|~~(~~_i_~~)~~<br>_j_<br>**u**~~(~~_i_~~)~~<br>_j_|~~(~~_i_~~)~~<br>_j_<br>**u**~~(~~_i_~~)~~<br>_j_|~~(~~_i_~~)~~<br>_j_<br>**u**~~(~~_i_~~)~~<br>_j_||


|Col1|Col2|Col3|h|
|---|---|---|---|
||**l**(_i_)<br>_j_<br>**u**(_i_)<br>_j_|**l**(_i_)<br>_j_<br>**u**(_i_)<br>_j_|**l**(_i_)<br>_j_<br>**u**(_i_)<br>_j_|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|~~**l**(~~~~_i_)~~<br>_j_|~~**l**(~~~~_i_)~~<br>_j_|**u**|**u**|



(a) (b) (c) (d)
Figure 1: Relaxations of a ReLU: (a) “triangle” relaxation in LP; (b)(c) No relaxation when **u** [(] _j_ _[i]_ [)] _≤_ 0 (always

inactive) or **l** [(] _j_ _[i]_ [)] _≥_ 0 (always active); (d) linear relaxation in LiRPA when **l** [(] _j_ _[i]_ [)] _<_ 0 _,_ **u** [(] _j_ _[i]_ [)] _>_ 0 (unstable).


A lower bound _f_ can then be simply obtained by taking the lower b ~~ound of the linear equa~~ tion
**A** _x_ + **b** w.r.t input _x ∈C_, which can be obtained via H¨older’s inequality when _C_ is a _ℓ_ _p_ norm ball.


To get the coefficients **A**, **A**, **b**, **b**, LiRPA propagates bounds of _f_ ( _x_ ) as a linear function to the
output of each layer, in a backward manner. At the output layer _h_ [(] _[L]_ [)] ( _x_ ) we simply have:

|u|(i) > 0 (<br>j|unstabl|
|---|---|---|
|h|~~ of the~~<br>en_ C_ is|~~linear~~<br>a_ ℓp_ no|
|s<br>|a linear<br>|functi<br>|


**I** _h_ [(] _[L]_ [)] ( _x_ ) _≤_ _f_ ( _x_ ) _≤_ **I** _h_ [(] _[L]_ [)] ( _x_ ) _,_ _x ∈C_ (3)
Then, the next step is to backward propagate the identity linear relationship through a linear layer
_h_ [(] _[L]_ [)] ( _x_ ) = **W** [(] _[L]_ [)] _g_ [(] _[L][−]_ [1)] ( _x_ ) to get the linear bounds of _f_ ( _x_ ) w.r.t _g_ [(] _[L][−]_ [1)] :

**W** [(] _[L]_ [)] _g_ [(] _[L][−]_ [1)] ( _x_ ) _≤_ _f_ ( _x_ ) _≤_ **W** [(] _[L]_ [)] _g_ [(] _[L][−]_ [1)] ( _x_ ) _,_ _x ∈C_ (4)

To get the linear relationship of _h_ [(] _[L][−]_ [1)] w.r.t _f_ ( _x_ ), we need to backward propagate through ReLU
layer _g_ [(] _[L][−]_ [1)] ( _x_ ) = ReLU( _h_ [(] _[L][−]_ [1)] ( _x_ )). Since it is nonlinear, we perform linear relaxations. For
illustration, considering the _j_ -th ReLU neuron at _i_ -th layer, _g_ _j_ [(] _[i]_ [)] [(] _[x]_ [) =][ ReLU][(] _[h]_ [(] _j_ _[i]_ [)] [(] _[x]_ [))][, we can]

~~(~~ _i_ )
linearly upper and lower bound it by _a_ [(] ~~_j_~~ _[i]_ [)] _[h]_ [(] _j_ _[i]_ [)] [(] _[x]_ [) +] _[b]_ [(] ~~_j_~~ _[i]_ [)] _≤_ _g_ _j_ [(] _[i]_ [)] [(] _[x]_ [)] _[ ≤]_ ~~_[a]_~~ ~~[(]~~ _j_ _[i]_ [)] _[h]_ [(] _j_ _[i]_ [)] [(] _[x]_ [) +] _[ b]_ _j_ [, where]



~~(~~ _i_ )
_a_ [(] ~~_j_~~ _[i]_ [)] _[, ]_ ~~_[a]_~~ [(] _j_ _[i]_ [)] _[,][ b]_ [(] ~~_j_~~ _[i]_ [)] _[, b]_ _j_ are:
 ~~~~ _a_ [(] ~~_j_~~ _[i]_ [)] = ~~_a_~~ ~~[(]~~ _j_ _[i]_ [)] = 0 _, b_ [(] ~~_j_~~ _[i]_

_a_ [(] ~~_j_~~ _[i]_ [)] = ~~_a_~~ ~~[(]~~ _j_ _[i]_ [)] = 1 _, b_ [(] ~~_j_~~ _[i]_

~~~~



**u** [(] _[i]_ **u** [)][(] _j_ _−_ _[i]_ [)] **l** [(] _[i]_ [)] _, b_ [(] ~~_j_~~ _[i]_ [)] = 0 _, b_ ~~(~~ _ji_ ) = _−_ **uu** [(] _[i]_ [(] _j_ [)] _[i]_ [)] _−_ **l** [(] _j_ **l** _[i]_ [(][)] _[i]_
_j_ _j_ _j_ _j_



~~(~~ _i_ )
_a_ [(] ~~_j_~~ _[i]_ [)] = ~~_a_~~ ~~[(]~~ _j_ _[i]_ [)] = 0 _, b_ [(] ~~_j_~~ _[i]_ [)] = _b_ _j_ = 0 **u** [(] _j_ _[i]_ [)] _≤_ 0 (always inactive for _x ∈C_ )



~~(~~ _i_ )
_a_ [(] ~~_j_~~ _[i]_ [)] = ~~_a_~~ ~~[(]~~ _j_ _[i]_ [)] = 1 _, b_ [(] ~~_j_~~ _[i]_ [)] = _b_ _j_ = 0 **l** [(] _j_ _[i]_ [)] _≥_ 0 (always active for _x ∈C_ )



(5)



**u** [(] _[i]_ [)]
_a_ [(] ~~_j_~~ _[i]_ [)] = _α_ _j_ [(] _[i]_ [)] _[, ]_ ~~_[a]_~~ ~~[(]~~ _j_ _[i]_ [)] = [(] _[i]_ [)] _j_



**uu** [(] _[i]_ _j_ [)] _−_ _j_ **l** [(] _[i]_ [)] **l** [(] _j_ _[i]_ [)] _<_ 0 _,_ **u** [(] _j_ _[i]_ [)] _>_ 0 (unstable for _x ∈C_ )
_j_ _j_







Here **l** [(] _j_ _[i]_ [)] _≤_ _h_ [(] _j_ _[i]_ [)] [(] _[x]_ [)] _[ ≤]_ **[u]** [(] _j_ _[i]_ [)] are intermediate pre-activation bounds for _x ∈C_, and _α_ _j_ [(] _[i]_ [)] is an arbitrary

value between 0 and 1. The pre-activation bounds **l** [(] _j_ _[i]_ [)] and **u** [(] _j_ _[i]_ [)] can be computed by treating _h_ [(] _j_ _[i]_ [)] [(] _[x]_ [)]
as the output neuron with LiRPA. Figure 1(b,c,d) illustrate the relaxation for each state of ReLU
neuron. With these linear relaxations, we can get the linear equation of _h_ [(] _[L][−]_ [1)] w.r.t output _f_ ( _x_ ):


~~(~~ _L−_ 1) ( _L_ )
**W** [(] _[L]_ [)] **D** [(] _α_ _[L][−]_ [1)] _h_ [(] _[L][−]_ [1)] ( _x_ ) + **b** [(] _[L]_ [)] _≤_ _f_ ( _x_ ) _≤_ **W** [(] _[L]_ [)] **D** _α_ _h_ [(] _[L][−]_ [1)] ( _x_ ) + **b** _,_ _x ∈C_



_b_ ~~_j_~~ [(] _[L]_ [)] _,_ **W** _j_ [(] _[L]_ [)] _≥_ 0

~~(~~ _L_ )

� _b_ _j_ _,_ **W** _j_ [(] _[L]_ [)] _<_ 0



**D** [(] _α,_ _[L]_ ( [)] _j,j_ ) [=]



~~�~~ ~~_a_~~ _a_ [(] ~~[(]~~ _jj_ _[L][L]_ [)][)] _,,_ **WW** _jj_ [(][(] _[L][L]_ [)][)] _≥<_ 00 _[,]_ **b** [(] _[L]_ [)] = **b** _[′]_ [(] _[L]_ [)] _[⊤]_ **W** [(] _[L]_ [)] _,_ where **b** _[′]_ _j_ [(] _[L]_ [)] =



(6)



~~(~~ _L−_ 1)
The diagonal matrices **D** [(] _α_ _[L][−]_ [1)], **D** _α_ and biases reflects the linear relaxations and also considers
the signs in **W** [(] _[L]_ [)] to maintain the lower and upper bounds. The definitions for _j_ -th diagonal element
~~(~~ _L_ ) ~~(~~ _L_ ) ( _L_ )
**D** _α,_ ( _j,j_ ) [and bias] **[ b]** are similar, with the conditions for checking the signs of **W** _j_ swapped.

Importantly, **D** [(] _α_ _[L][−]_ [1)] has free variables _α_ _j_ [(] _[i]_ [)] _∈_ [0 _,_ 1] which do not affect correctness of the bounds.
We can continue backward propagating these bounds layer by layer (e.g., _g_ [(] _[L][−]_ [2)] ( _x_ ), _h_ [(] _[L][−]_ [2)] ( _x_ ),
etc) until reaching _g_ [(0)] ( _x_ ) = _x_, getting the eventual linear equations of _f_ ( _x_ ) in terms of input _x_ :


**L** ( _x,_ _**α**_ ) _≤_ _f_ ( _x_ ) _≤_ **U** ( _x,_ _**α**_ ) _,_ _∀x ∈C,_ where

~~(~~ _L−_ 1) ~~(~~ 1) (7)
**L** ( _x,_ _**α**_ ) = **W** [(] _[L]_ [)] **D** [(] _α_ _[L][−]_ [1)] _· · ·_ **D** [(1)] ~~_α_~~ **[W]** [(1)] _[x]_ [ +] **[ b]** _[,]_ **U** ( _x,_ _**α**_ ) = **W** [(] _[L]_ [)] **D** _α_ _· · ·_ **D** _α_ **[W]** [(1)] _[x]_ [ +] **[ b]**

Here _**α**_ denotes _α_ _j_ [(] _[i]_ [)] for all unstable ReLU neurons in NN. The obtained bounds ( **L** ( _x,_ _**α**_ ) _,_ **U** ( _x,_ _**α**_ ))
of _f_ ( _x_ ) are linear functions in terms of _x_ . Beyond the simple feedforward NN presented here, LiRPA
can support more complicated NN architectures like DenseNet and Transformers by computing **L**
and **U** automatically and efficiently on general computational graphs (Xu et al., 2020).


**Soundness of LiRPA** The above backward bound propagation process guarantees that **L** ( _x, α_ ) and
**U** ( _x, α_ ) soundly bound _f_ ( _x_ ) for all _x ∈C_ . Detailed proofs can be found in (Zhang et al., 2018;
Singh et al., 2019b) for feedforward NNs and (Xu et al., 2020) for general networks.


4


Unoptimized LiRPA: lower bound 𝑓 = −1.94























































Optimized LiRPA: lower bound 𝑓 = −0.83

















































Figure 2: Illustration of our optimized LiRPA bounds and the BaB process. Given a two-layer neural network,
we aim to verify output _f_ ( _x_ ) _≥_ 0. Optimized LiRPA chooses optimized slopes for ReLU lower bounds,
allowing tightening the intermediate layer bounds _**l**_ _j_ [(] _[i]_ [)] and _**u**_ [(] _j_ _[i]_ [)] and also the output layer lower bound _f_ . BaB

splits two unstable neurons _h_ [(2)] 2 and _h_ [(2)] 1 to improve _f_ and verify all sub-domains ( _f ≥_ 0 for all cases).


3 P ROPOSED A LGORITHM


**Overview** In this section, we will first introduce our proposed efficient optimization of LiRPA
bounds on GPUs that can allow us to achieve tight approximation on par with LP or even tighter for
some cases but in a much faster manner. In Fig. 2, we provide a two-layer NN example to illustrate
how our optimized LiRPA can improve the performance of BaB verification. In Section 3.2, we
then show that feasibility checking is important to guarantee the completeness of BaB, and BaB
using LiRPA without feasibility checking will end up to be incomplete. To ensure completeness,
we design our algorithm with minimal usage of LP for checking feasibility of splits. Finally, we
propose a batch split design by solving a batch of sub-domains in a massively parallel manner on
GPUs to fully leverage the benefits of cheap and parallelizable LiRPA. We further improve BaBSR
in a parallel fashion for branching and we summarize the detailed proposed algorithm in Section 3.4.


**3.1** **Optimized LiRPA for Complete Verification**


**Concrete outer bounds with optimizable parameters** We propose to use LiRPA as the bounding
step in BaB. A pair of sound and concrete lower bound and upper bound ( _f, f_ ) to _f_ ( _x_ ) can be
obtained according to Eq. 7 given fixed _**α**_ = _**α**_ 0 :


_f_ ( _**α**_ 0 ) = min _x∈C_ **[L]** [(] _[x,]_ _**[ α]**_ [0] [)] _[,]_ _f_ ( _**α**_ 0 ) = max _x∈C_ **[U]** [(] _[x,]_ _**[ α]**_ [0] [)] (8)


Because **L**, **U** are linear functions w.r.t _x_ when _**α**_ 0 is fixed, it is easy to solve Eq. 8 using H¨older’s
inequality when _C_ is a _ℓ_ _p_ norm ball (Xu et al., 2020). In incomplete verification settings, _**α**_ can be
set via certain heuristics (Zhang et al., 2018). Salman et al. (2019) showed that, the variable _**α**_ is
equivalent to dual variables in the LP relaxed verification problem (Wong & Kolter, 2018). Thus, an
optimal selection of _**α**_ given the _same_ pre-activation bounds **l** [(] _j_ _[i]_ [)] and **u** [(] _j_ _[i]_ [)] can in fact, lead to the the
same optimal solution for _f_ and _f_ as in LP.


Previous complete verifiers typically use LiRPA variants to obtain intermediate layer bounds to
construct an LP problem (Bunel et al. (2018); Lu & Kumar (2020)) and solve the LP to obtain
bounds at output layer. The main reason for using LP is that it typically produces much tighter
bounds than LiRPA when _**α**_ is not optimized. We use optimized LiRPA, which is fast, acceleratorfriendly, and can produce tighter bounds, well outperforming LP for complete verification:

_f_ = min _**α**_ [min] _x∈C_ **[L]** [(] _[x,]_ _**[ α]**_ [)] _[,]_ _f_ = max _**α**_ max _x∈C_ **[U]** [(] _[x,]_ _**[ α]**_ [)] _[,]_ _α_ _j_ [(] _[i]_ [)] _∈_ [0 _,_ 1] (9)

The inner minimization or maximization has closed form solutions (Xu et al., 2020) based on
H¨older’s inequality, so we only need to optimize on _**α**_ . Since we use a differentiable framework (Xu
et al., 2020) to compute the LiRPA bound functions **L** and **U**, the gradients _∂_ _[∂]_ _**α**_ **[L]** [and] _[∂]_ _∂_ **[U]** _**α**_ [can be]

obtained easily. Optimization over _**α**_ can be done via projected gradient descent (each coordinate of
_**α**_ is constrained in [0 _,_ 1]). Since the gradient computation and optimization are done on GPUs, the
bounding process is still very fast and can be one or two magnitudes faster than solving an LP.


**Optimized LiRPA bounds can be tighter than LP** Solving Eq. 9 using gradient descent cannot
guarantee to converge to the global optima, so it seems the bounds must be looser than LP. Counterintuitively, by optimizing _**α**_, we can potentially _obtain tighter bounds_ than LP. When a “triangle”


5



_∂_ _[∂]_ _**α**_ **[L]** [and] _[∂]_ _∂_ **[U]** _**α**_


relaxation is constructed for LP, intermediate pre-activation bounds **l** [(] _j_ _[i]_ [)] [,] **[ u]** [(] _j_ _[i]_ [)] must be fixed for the
_j_ -th ReLU in layer _i_ . During the LP optimization process, only the output bounds are optimized;
intermediate bounds stay unchanged. However, in the LiRPA formulation, **L** ( _x,_ _**α**_ ) and **U** ( _x,_ _**α**_ ) are
complex functions of _**α**_ : since intermediate bounds are also computed by LiRPA, they depend on all
_α_ [(] _[i]_ _[′]_ [)] _[∂]_ **[L]** _[∂]_ **[U]**
_j_ _[′]_ [ (0] _[ < i]_ _[′]_ _[ < i]_ [)][ in previous layers. Thus, the gradients] _∂_ _**α**_ [and] _∂_ _**α**_ [can tighten output layer bounds]



_∂_ _[∂]_ _**α**_ **[L]** [and] _[∂]_ _∂_ **[U]** _**α**_



_α_ _j_ _[′]_ [ (0] _[ < i]_ _[′]_ _[ < i]_ [)][ in previous layers. Thus, the gradients] _∂_ _**α**_ [and] _∂_ _**α**_ [can tighten output layer bounds]

_f_ and _f_ indirectly by tightening intermediate layer bounds, forming a tighter convex relaxation for

the next iteration. An LP solver cannot achieve this because adding **l** [(] _j_ _[i]_ [)] and **u** [(] _j_ _[i]_ [)] as optimization
variables makes the problem non-linear. This is the key to our success of applying LiRPA based
bounds for the complete verification setting, where tighter bounds are essential.


In Figure 3, we illustrate our optimized LiRPA bounds and
the LP solution. Initially, we use LiRPA with _**α**_ set via a fast
heuristic to compute intermediate layer bounds **l** [(] _j_ _[i]_ [)] and **u** [(] _j_ _[i]_ [)] [,]
and then use them to build a relaxed LP problem. The solution
to this initial LP problem (red line) is much tighter than the
LiRPA solution with the heuristically set _**α**_ (the left-most point
of the blue line). Then, we optimize _**α**_ with gradient decent,
and LiRPA quickly outperforms this initial LP solution due to
optimized tighter intermediate layer bounds. We can create a
new LP with optimized intermediate bounds (light blue line),

|2<br>4<br>6<br>8<br>10|Col2|Col3|Col4|
|---|---|---|---|
|10<br>8<br>6<br>4<br>2||||
|10<br>8<br>6<br>4<br>2||||
|10<br>8<br>6<br>4<br>2||||
|10<br>8<br>6<br>4<br>2|||LP initial <br>|
|10<br>8<br>6<br>4<br>2|||optimized<br>LP optimi|
|0<br>50<br>100<br>iteratio<br>12|0<br>50<br>100<br>iteratio<br>12|0<br>50<br>100<br>iteratio<br>12|0<br>50<br>100<br>iteratio<br>12|


Figure 3: Optimized LiRPA bound (0

producing a slightly tighter bound than LiRPA with optimized

to 200 iterations) vs LP bounds.

_**α**_ . The LP bounds in most existing complete verifiers all use
intermediate layer bounds obtained from unoptimized LiRPA bounds or even weaker methods like
interval arithmetic, ending up to the solution close to or lower than the red line in Figure 3. Instead, our optimized LiRPA bounds can produce tight bounds, and also exploit parallel acceleration
from machine learning accelerators, leading to huge improvements in verification time compared to
existing baselines.

**ReLU Split Constraints** In the BaB process, when a ReLU _h_ [(] _j_ _[i]_ [)] is split into two sub-domains

( _h_ [(] _j_ _[i]_ [)] _≥_ 0 and _h_ [(] _j_ _[i]_ [)] _<_ 0), we simply set _**l**_ _j_ [(] _[i]_ [)] _≥_ 0 and _**u**_ [(] _j_ _[i]_ [)] _<_ 0 in bounding step. It tighten the
LiRPA bounds by forcing the split ReLU linear, reducing relaxation errors. However, when splits
are added, LiRPA and LP are not equivalent even under fixed **l** [(] _j_ _[i]_ [)] [,] **[ u]** [(] _j_ _[i]_ [)] and optimal _**α**_ . After splits,
LiRPA cannot check certain constraints where LP is capable to, as we will discuss in the next section.



















Figure 3: Optimized LiRPA bound (0
to 200 iterations) vs LP bounds.



**3.2** **Completeness with Minimal Usage of LP Bounding Procedure**


Even though our optimized LiRPA can bring us huge speed improvement over LP for BaB based
verification, we observe that it may end up to be incomplete due to the lack of feasibility checking: it
cannot detect some conflicting settings of ReLU splits. We state such an observation in Theorem 3.1:


**Theorem 3.1 (Incompleteness without feasibility checking)** _When using LiRPA variants de-_
_scribed in Section 2.3 as the bounding procedure, BaB based verification is incomplete._


We prove the theorem by giving a counter-example in Appendix A.1 where _all_ ReLU neurons are
split and thus LiRPA runs on a linear network for each sub-domain. As a result, LiRPA can still
be indecisive for the verification problem. The main reason is that LiRPA variants will lose the
feasibility information encoded by the sub-domain constraints. For illustration, consider a subdomain _C_ _i_ = _C ∩_ ( _h_ [(] _j_ _[i]_ 1 [1] [)] _<_ 0) _∩_ ( _h_ [(] _j_ _[i]_ 2 [2] [)] _≥_ 0), LiRPA will force _g_ _j_ [(] 1 _[i]_ [1] [)] [(] _[x]_ [) = 0][ (inactive ReLU, a zero]

function) and _g_ _j_ [(] 2 _[i]_ [2] [)] [(] _[x]_ [) =] _[ h]_ [(] _j_ _[i]_ 2 [2] [)] [(] _[x]_ [)][ (active ReLU, an identity function) respectively and propagate]
these bounds to get the approximated lower bound _f_ ~~_C_~~ _i_ . However, the split feasibility constraint

( _h_ [(] _j_ _[i]_ 1 [1] [)] _<_ 0) _∩_ ( _h_ [(] _j_ _[i]_ 2 [2] [)] _≥_ 0) is ignored, so two conflict splits may be conducted (e.g., when _h_ [(] _j_ _[i]_ 1 [1] [)] _<_ 0,

_h_ [(] _j_ _[i]_ 2 [2] [)] cannot be _≥_ 0). On the contrary, LP can fully preserve such feasibility information due to the
linear solver involved and detect the infeasible sub-domains. Then, in Theorem 3.2 we show that the
minimal usage of feasibility checking with LP can guarantee the completeness of BaB with LiRPA.


6


**Algorithm 1** Parallel BaB with optimized LiRPA bounding (we highlight the differences between
our algorithm and regular BaB (Bunel et al., 2018) in blue. Comments are in brown.)



1: **Inputs** : _f_, _C_, _n_ (batch size), _η_ (threshold to switch to LP)
2: ( _f, f_ ) _←_ `optimized` ~~`L`~~ `iRPA` ( _f,_ [ _C_ ])
3: P _←_ �( _f, f, C_ )� _▷_ P is the set of all unverified sub-domains
4: **while** _f <_ 0 **and** _f ≥_ 0 **do**
5: ( _C_ 1 _, . . ., C_ _n_ ) _←_ `batch` ~~`p`~~ `ick` ~~`o`~~ `ut` (P _, n_ ) _▷_ Pick sub-domains to split and removed them from P
6: � _C_ 1 _[l]_ _[,][ C]_ 1 _[u]_ _[, . . .,][ C]_ _n_ _[l]_ _[,][ C]_ _n_ _[u]_ � _←_ `batch` ~~`s`~~ `plit` ( _C_ 1 _, . . ., C_ _n_ ) _▷_ Each _C_ _i_ splits into two sub-domains _C_ _i_ _[l]_ [and] _[ C]_ _i_ _[u]_

7: _[l]_ _[u]_ _[l]_ _←_ ~~`L`~~ `iRPA` _C_ _[l]_ _[ C]_ _[u]_ _[ C]_ _[l]_ _[ C]_ _[u]_ _▷_



� _f_ ~~_C_~~ 1 _l_ _[, f]_ _[ C]_ 1 _[l]_ _[, ][f]_ ~~_[C]_~~ 1 _[u]_ _[, f]_ _[ C]_ 1 _[u]_ _[, . . ., ][f]_ ~~_C_~~ _n_ _[l]_ _[, f]_ _[ C]_ _n_ _[l]_ _[, ][f]_ ~~_C_~~ _n_ _[u]_ _[, f]_ _[ C]_ _n_ _[u]_



� _←_ `optimized` ~~`L`~~ `iRPA` ( _f,_ � _C_ 1 _[l]_ _[,][ C]_ 1 _[u]_ _[, . . .,][ C]_ _n_ _[l]_ _[,][ C]_ _n_ _[u]_ �) _▷_



Compute lower and upper bounds using LiRPA for each sub-domain on GPUs in a batch
8: P _←_ P [�] `Domain` ~~`F`~~ `ilter` �[ _f_ _C_ 1 _l_ _[, f]_ _[ C]_ 1 _[l]_ _[,][ C]_ 1 _[l]_ []] _[,]_ [[] _[f]_ ~~_C_~~ 1 _u_ _[, f]_ _[ C]_ 1 _[u]_ _[,][ C]_ 1 _[u]_ []] _[, . . .,]_ [[] _[f]_ ~~_C_~~ _ln_ _[, f]_ _C_ _n_ [1] _[,][ C]_ _n_ _[l]_ []] _[,]_ [[] _[f]_ ~~_C_~~ _nu_ _[, f]_ _[ C]_ _n_ _[u]_ _[,][ C]_ _n_ _[u]_ []] � _▷_

Filter out verified sub-domains, insert the left domains back to P
9: _f ←_ min _{f_ _C_ _i_ _|_ ( _f_ ~~_C_~~ _i_ _, C_ _i_ ) _∈_ P _}_, _i_ = 1 _, . . ., n_ _▷_ To ease notation, _C_ _i_ here indicates both _C_ _i_ _[u]_ [and] _[ C]_ _i_ _[l]_
10: _f ←_ min _{f_ _C_ _i_ _|_ ( _f_ _C_ _i_ _, C_ _i_ ) _∈_ P _}_, _i_ = 1 _, . . ., n_
11: **if** `length` (P) _> η_ **then** _▷_ Fall back to LP for completeness



12:



� _f_ _C_ 1 _l_ _[, f]_ _[ C]_ 1 _[l]_ _[, ][f]_ ~~_[C]_~~ 1 _[u]_ _[, f]_ _[ C]_ 1 _[u]_ _[, . . ., ][f]_ _C_ _n_ _[l]_ _[, f]_ _[ C]_ _n_ _[l]_ _[, ][f]_ ~~_C_~~ _n_ _[u]_ _[, f]_ _[ C]_ _n_ _[u]_ _[,]_ � _←_ `compute` ~~`b`~~ `ound` ~~`L`~~ `P` ( _f,_ � _C_ 1 _[l]_ _[,][ C]_ 1 _[u]_ _[, . . .,][ C]_ _n_ _[l]_ _[,][ C]_ _n_ _[u]_ �)



13: P _←_ P [�] `Domain` ~~`F`~~ `ilter` �[ _f_ _C_ 1 _l_ _[, f]_ _[ C]_ 1 _[l]_ _[,][ C]_ 1 _[l]_ []] _[,]_ [[] _[f]_ _C_ 1 _u_ _[, f]_ _[ C]_ 1 _[u]_ _[,][ C]_ 1 _[u]_ []] _[, . . .,]_ [[] _[f]_ ~~_C_~~ _ln_ _[, f]_ _C_ _n_ [1] _[,][ C]_ _n_ _[l]_ []] _[,]_ [[] _[f]_ ~~_C_~~ _nu_ _[, f]_ _[ C]_ _n_ _[u]_ _[,][ C]_ _n_ _[u]_ []] �

14: **Outputs:** _f_, _f_


**Theorem 3.2 (Minimal feasibility checking for completeness)** _When using LiRPA variants de-_
_scribed in Section 2.3 as the bounding procedure, BaB based verification is complete if all infeasible_
_leaf sub-domains (i.e., sub-domains cannot be further split) are detected by linear programming._


We prove the theorem in Appendix A.2, where we show that by checking the feasibility of splits
with LP, we can eliminate the cases where incompatible splits are chosen in the LiRPA BaB process.
Since LP is slow while LiRPA is highly efficient, we propose to only use LP when the LiRPA based
bounding process is stuck, either (1) when partitioning and bounding new sub-domains with LiRPA
cannot further improve the bounds, or (2) when all unstable neurons have been split. In this way, the
infeasible sub-domains can be eventually detected by occasional usage of LP while the advantage of
massive parallel LiRPA on GPUs is fully enjoyed. We will describe our full algorithm in Sec. 3.4.


**3.3** **Batch Splits**

SOTA BaB methods (Bunel et al., 2020b; Lu & Kumar, 2020) only split one sub-domain during
each branching step. Since we use cheap and GPU-friendly LiRPA bounds, we can select a batch of
sub-domains to split and propagate their LiRPA bounds in a batch. Such a batch splitting design can
greatly improve hardware efficiency on GPUs. Given a batch size _n_ that allows us to fully use the
GPU memory available, we can obtain _n_ bounds simultaneously. It grows the search tree on a single
leaf by a depth of log 2 _n_, or split _n/_ 2 leaf nodes at the same time, accelerating by up to _n_ times.


**3.4** **Our Complete Verification Framework**

Our LiRPA based complete verification framework is presented in Alg. 1. The algorithm takes a
target NN function _f_ and a domain _C_ as inputs. We run optimized LiRPA to get initial bounds
( _f, f_ ) for _x ∈C_ (Line 2). Then we utilize the power of GPUs to split in parallel and maintain
a global set P storing all the sub-domains which cannot be verified with optimized LiRPA (Line
5-10). Specifically, `batch` ~~`p`~~ `ick` ~~`o`~~ `ut` improves BaBSR (Bunel et al., 2018) in a parallel manner to
select _n_ sub-domains in P and determine the corresponding ReLU neuron to split for each of them. If
the length of P is less than _n_, then we reduce _n_ to the length of P. `batch` ~~`s`~~ `plit` splits each selected
_C_ _i_ to two sub-domains _C_ _i_ _[l]_ [and] _[ C]_ _i_ _[u]_ [by forcing the selected unstable ReLU neuron to be positive and]
negative, respectively. `optimize` ~~`L`~~ `iRPA` runs optimized LiRPA in parallel as a batch and returns
the lower and upper bounds for _n_ selected sub-domains simultaneously. `Domain` ~~`F`~~ `ilter` filters out
verified sub-domains (proved with _f_ _C_ _i_ _≥_ 0) and we insert the remaining ones to P. The loop breaks

if the property is proved ( _f ≥_ 0) or a counter-example is found in any sub-domain ( _f <_ 0).


To avoid excessive splits, we set the maximum length of the sub-domains to _η_ (Line 12). Once the
length of P reaches this threshold, `compute` ~~`b`~~ `ound` ~~`L`~~ `P` will be called. It solves these _η_ sub-domains
by LP (one by one in a loop, or in parallel if using multiple CPUs is allowed) with optimized LiRPA


7


computed intermediate layer bounds. If a sub-domain _C_ _i_ _∈_ P (which previously cannot be verified
by LiRPA) is proved or detected to be infeasible by LP, as an effective heuristic, we will backtrack
and prioritize to check its parent node with LP. If the parent sub-domain is also proved or infeasible,
we can prune all its child nodes to greatly reduce the size of the search tree.


**Completeness of our framework** Our algorithm is _complete_, because we follow Theorem 3.2 and
check feasibility of all split sub-domains that have deep BaB search tree depth (length of P reaches
threshold _η_ ), forming a superset of the worst case where all ReLU neurons are split.


4 E XPERIMENTS


In this section, we compare our verifier against the state-of-the-art ones to illustrate the effectiveness
of our proposed framework. Overall, our verifier is about 10X, 4X and 20X faster than the best
LP-based verifier (Lu & Kumar, 2020) on the Base, Wide and Deep models, respectively.


**Setup** We follow the most challenging experimental setup used in the state-of-the-art verifiers
GNN- ONLINE (Lu & Kumar, 2020) and B A BSR (Bunel et al., 2020b). Specifically, we evaluate on CIFAR10 dataset on three NNs: Base, Wide and Deep. The dataset is categorized into
three difficulty levels: Easy, Medium, and Hard, which is generated according to the performance
of BaBSR. The verification task is defined as given a _l_ _∞_ norm perturbation less than _ϵ_, the classifier will not predict a specific (predefined) wrong label for each image _x_ (see Appendix B).
We set batch size _n_ = 400 _,_ 200 _,_ 200 for Base, Wide and Deep model respectively and threshold
_η_ = 12000. More details on experimental setup are provided in Appendix B. Our code is available
[at https://github.com/kaidixu/LiRPA](https://github.com/kaidixu/LiRPA_Verify) ~~V~~ erify.


**Comparisons against state-of-the-art verifiers** We include five different methods for comparison:
(1) B A BSR (Bunel et al., 2020b), a BaB and LP based verifier using a simple ReLU split heuristic;
(2) MIPP LANET (Ehlers, 2017), a customized MIP solver for NN verification where unstable ReLU
neurons are randomly selected for split; (3) GNN (Lu & Kumar, 2020) and (4) GNN-O NLINE (Lu
& Kumar, 2020) are BaB and LP based verifiers using a learned graph neural network (GNN) to
guide the ReLU split. (5) BDD+ B A BSR (Bunel et al., 2020a) is a very recently proposed verification framework based on Lagrangian decomposition which also supports GPU acceleration without
solving LPs. All methods use 1 CPU with 1 GPU. The timeout threshold is 3,600 seconds.


For the Base model in different difficulty levels, Easy, Medium and Hard, Table 1 shows that we are
around 5 _∼_ 40X faster than baseline BaBSR and around 2 _∼_ 20X faster than GNN split baselines.
The accumulative solved properties with increasing runtime are shown in Figure 4. In all our experiments, we use the basic heuristic in BaBSR for branching and do not use GNNs, so our speedup
comes purely from the faster LiRPA based bounding procedure. We are also competitive against
Lagrangian decomposition on GPUs.

Table 1: Performance of various methods on different models. We compare each method’s avg. solving time,
the avg. number of branches required, and the percentage of timed out (TO) properties.


Base - Easy Base - Medium Base - Hard Wide Deep


Method time(s) branches %TO time(s) branches %TO time(s) branches %TO time(s) branches %TO time(s) branches %TO


B A BSR 522.5 585 0.0 1335.4 1471 0.0 2875.2 1843 35.2 3325.7 455 50.3 2855.2 365 54.0

MIP PLANET 1462.2  - 16.5 1912.2  - 43.5 2172.2  - 46.2 3088.4  - 79.4 2842.5  - 73.6

GNN 312.9 301 0.0 624.1 635 0.9 1468.7 931 15.6 1791.5 375 19.0 1870.6 198 18.4

GNN- ONLINE 207.43 269 0.0 638.15 546 0.4 1255.4 968 15.6 1642.0 389 19.0 1845.7 196 18.4

BDD+ B A BSR 15.68 1371 0.0 51.88 6482 0.4 **627.96** 91880 13.4 510.55 45855 11.4 230.06 6721 4.4

O URS **11.86** 2589 0.0 **42.04** 9233 **0.0** 633.85 96755 **13.0** **375.23** 53481 **8.5** **81.55** 1439 **1.6**


Figure 4: Cactus plots for our method and other baselines in Base (Easy, Medium and Hard ), Wide and Deep
models. We plot the percentage of solved properties with growing running time.




















|BaBSR<br>MIPplanet<br>GNN<br>GNN-Online<br>Prox. BaBSR<br>Ours|verifi<br>80<br>60 properties<br>40<br>20<br>f|Col3|Col4|Col5|80<br>60<br>40<br>20|Col7|Col8|Col9|80<br>60<br>40<br>20|Col11|Col12|Col13|80<br>60<br>40<br>20|Col15|Col16|Col17|80<br>60<br>40<br>20|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|100<br>101<br>Runn<br>0<br>% o<br>|100<br>101<br>Runn<br>0<br>% o<br>|100<br>101<br>Runn<br>0<br>% o<br>|102<br>ng time (in|103<br>s)<br>100<br>101<br>Runni<br>0|103<br>s)<br>100<br>101<br>Runni<br>0|103<br>s)<br>100<br>101<br>Runni<br>0|102<br>ng time (in|103<br> s)<br>100<br>101<br>Runni<br>0|103<br> s)<br>100<br>101<br>Runni<br>0|103<br> s)<br>100<br>101<br>Runni<br>0|102<br>ng time (i|103<br> s)<br>100<br>101<br>Runn<br>0|103<br> s)<br>100<br>101<br>Runn<br>0|103<br> s)<br>100<br>101<br>Runn<br>0|102<br>ng time (i|103<br> s)<br>100<br>101<br>Runni<br>0|103<br> s)<br>100<br>101<br>Runni<br>0|103<br> s)<br>100<br>101<br>Runni<br>0|102<br>ng time (i|103<br> s)|



(d) Wide





(c) Base-Hard



(a) Base-Easy (b) Base-Medium (c) Base-Hard (d) Wide (e) Deep


**Performance on Larger Models** In Table 1, we show that our verifier is more scalable on larger
(wider or deeper) NNs compared to other state-of-the-art verifiers. Our method enjoys efficient GPU
acceleration particularly on Deep model and can achieve 30X speedup compared to B A BSR, and we



(a) Base-Easy



(b) Base-Medium



8


are also significantly faster than Lagrangian decomposition based GPU verifier (BDD+ B A BSR).
When compared to the state-of-the-art LP based BaB, GNN-O NLINE, our method can save 20X
running time on Deep model. In Appendix C, we analyze the effectiveness of optimized LiRPA and
batch splits separately, and find that optimized LiRPA is crucial for NN verification. Performance
comparisons of our proposed framework on CPU cores without GPU acceleration are included in
Appendix D.


5 R ELATED W ORK


**Complete verifiers** Early complete verifiers rely on satisfiability modulo theory (SMT) (Katz et al.,
2017; Huang et al., 2017; Ehlers, 2017) and mixed integer linear programming (MILP) (Tjeng et al.,
2019a; Dutta et al., 2018), and they typically do not scale well. Higher order logic provers such
as proof assistant (Bentkamp et al., 2018) can also be potentially used for NN verification, but
their scalability to the NN setting has not been demonstrated. Recently, Bunel et al. (2018) unified
many approaches used in various complete verifiers into a BaB framework. An LP based bounding
procedure is used in most of the existing BaB framework (Bunel et al., 2018; Wang et al., 2018c;
Royo et al., 2019; Lu & Kumar, 2020). For branching, two categories of branching strategies were
proposed: (1) input node branching (Wang et al., 2018c; Bunel et al., 2020b; Royo et al., 2019;
Anderson et al., 2019) where input features are divided into sub-domains, and (2) activation node
(especially, ReLU) branching (Katz et al., 2017; Bunel et al., 2018; Wang et al., 2018b; Ehlers, 2017;
Lu & Kumar, 2020) where hidden layer activations are split into sub-domains. Bunel et al. (2018)
found that input node branching cost is exponential to input dimension. Thus, many state-of-the-art
verifiers use activation node branching instead, focusing on heuristics to select good nodes to split.
BaBSR (Bunel et al., 2018) prioritizes ReLUs for splitting based on their pre-activation bounds; Lu
& Kumar (2020) used a graph neural network (GNN) to learn good splitting heuristics. Our work
focuses on improving bounding and can use better branching heuristics to achieve further speedup.


Two mostly relevant concurrent works using GPUs for accelerating NN verification are: (1)
GPUPoly (M¨uller et al., 2020), an extension of DeepPoly on CUDA, is _still an incomplete verifier_ .
Also, it is implemented in CUDA C++, requiring manual effort for customization and gradient computation, so it is not easy to get the gradients for optimizing bounds as we have done in Section 3.1.
(2) Lagrangian Decomposition (Bunel et al., 2020a) is a GPU-accelerated BaB based complete verifier that iteratively tightens the bounds based on a Lagrangian decomposition optimization formulation and does not reply on LP. However, it solves a much more complicated optimization problem
than LiRPA, and typically requires hundreds of iterations to converge for a single sub-domain.


**Incomplete verifiers** Many incomplete verification methods rely on convex relaxations of NN,
replacing nonlinear activations like ReLUs with linear constraints (Wong & Kolter, 2018; Wang
et al., 2018b; Zhang et al., 2018; Weng et al., 2018; Gehr et al., 2018; Singh et al., 2018a;b; 2019b;a)
or semidefinite constraints (Raghunathan et al., 2018; Dvijotham et al., 2020; Dathathri et al., 2020).
Tightening the relaxation for incomplete verification was discussed in (Dvijotham et al., 2018; Singh
et al., 2019a; Lyu et al., 2019; Tjandraatmadja et al., 2020). Typically, tight relaxations require
more computation and memory in general. We refer the readers to (Salman et al., 2019) for a
comprehensive survey. Recently, Xu et al. (2020) categorized the family of linear relaxation based
incomplete verifiers into LiRPA framework, allowing efficient implementation on machine learning
accelerators. Our work uses LiRPA as the bounding procedure for complete verification and exploits
its computational efficiency to accelerate, and our main contribution is to show that we can use fast
but weak incomplete verifiers as the main driver for complete verification when strategically applied.


6 C ONCLUSION


We use a LiRPA based incomplete NN verifier to accelerate the bounding procedure in branch and
bound (BaB) for complete NN verification on massively parallel accelerators. We use a fast gradient
based procedure to tighten LiRPA bounds. We study the completeness of BaB with LiRPA, and
show up to 5X speedup compared to state-of-the-art verifiers across multiple models and properties.


A CKNOWLEDGMENTS


This work is supported by NSF grant CNS18-01426; an ARL Young Investigator (YIP) award; an
NSF CAREER award; a Google Faculty Fellowship; a Capital One Research Grant; and a J.P. Mor

9


gan Faculty Award; Air Force Research Laboratory under FA8750-18-2-0058; NSF IIS-1901527,
NSF IIS-2008173 and NSF CAREER-2048280.


R EFERENCES


Greg Anderson, Shankara Pailoor, Isil Dillig, and Swarat Chaudhuri. Optimization and abstraction:
A synergistic approach for analyzing neural network robustness. In _Proceedings of the 40th_
_ACM SIGPLAN Conference on Programming Language Design and Implementation_, pp. 731–
744, 2019.


Alexander Bentkamp, Jasmin Christian Blanchette, Simon Cruanes, and Uwe Waldmann. Superposition for lambda-free higher-order logic. In _International Joint Conference on Automated_
_Reasoning_, pp. 28–46. Springer, 2018.


Rudy Bunel, Alessandro De Palma, Alban Desmaison, Krishnamurthy Dvijotham, Pushmeet Kohli,
Philip H. S. Torr, and M. Pawan Kumar. Lagrangian decomposition for neural network verification. _Conference on Uncertainty in Artificial Intelligence (UAI) 2020_, 2020a.


Rudy Bunel, Jingyue Lu, Ilker Turkaslan, P Kohli, P Torr, and P Mudigonda. Branch and bound for
piecewise linear neural network verification. _Journal of Machine Learning Research_, 21(2020),
2020b.


Rudy R Bunel, Ilker Turkaslan, Philip Torr, Pushmeet Kohli, and Pawan K Mudigonda. A unified
view of piecewise linear neural network verification. In _Advances in Neural Information Process-_
_ing Systems_, pp. 4790–4799, 2018.


Sumanth Dathathri, Krishnamurthy Dvijotham, Alexey Kurakin, Aditi Raghunathan, Jonathan Uesato, Rudy R Bunel, Shreya Shankar, Jacob Steinhardt, Ian Goodfellow, Percy S Liang, et al.
Enabling certification of verification-agnostic networks via memory-efficient semidefinite programming. _Advances in Neural Information Processing Systems_, 33, 2020.


Souradeep Dutta, Susmit Jha, Sriram Sankaranarayanan, and Ashish Tiwari. Output range analysis for deep feedforward neural networks. In _NASA Formal Methods Symposium_, pp. 121–138.
Springer, 2018.


Krishnamurthy Dvijotham, Sven Gowal, Robert Stanforth, Relja Arandjelovic, Brendan
O’Donoghue, Jonathan Uesato, and Pushmeet Kohli. Training verified learners with learned verifiers. _arXiv preprint arXiv:1805.10265_, 2018.


Krishnamurthy Dj Dvijotham, Robert Stanforth, Sven Gowal, Chongli Qin, Soham De, and Pushmeet Kohli. Efficient neural network verification with exactness characterization. In _Uncertainty_
_in Artificial Intelligence_, pp. 497–507. PMLR, 2020.


Ruediger Ehlers. Formal verification of piece-wise linear feed-forward neural networks. In _Interna-_
_tional Symposium on Automated Technology for Verification and Analysis_, pp. 269–286. Springer,
2017.


Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin Vechev. Ai2: Safety and robustness certification of neural networks with abstract interpretation. In _2018 IEEE Symposium on Security and Privacy (SP)_, pp. 3–18. IEEE, 2018.


Xiaowei Huang, Marta Kwiatkowska, Sen Wang, and Min Wu. Safety verification of deep neural
networks. In _International Conference on Computer Aided Verification_, pp. 3–29. Springer, 2017.


Guy Katz, Clark Barrett, David L Dill, Kyle Julian, and Mykel J Kochenderfer. Reluplex: An
efficient smt solver for verifying deep neural networks. In _International Conference on Computer_
_Aided Verification_, pp. 97–117. Springer, 2017.


Jingyue Lu and M Pawan Kumar. Neural network branching for neural network verification. _Inter-_
_national Conference on Learning Representation (ICLR)_, 2020.


Zhaoyang Lyu, Ching-Yun Ko, Zhifeng Kong, Ngai Wong, Dahua Lin, and Luca Daniel. Fastened
crown: Tightened neural network robustness certificates. _arXiv preprint arXiv:1912.00574_, 2019.


10


Matthew Mirman, Timon Gehr, and Martin Vechev. Differentiable abstract interpretation for provably robust neural networks. In _International Conference on Machine Learning_, pp. 3575–3583,
2018.


Christoph M¨uller, Gagandeep Singh, Markus P¨uschel, and Martin Vechev. Neural network robustness verification on gpus. _arXiv preprint arXiv:2007.10868_, 2020.


Aditi Raghunathan, Jacob Steinhardt, and Percy S Liang. Semidefinite relaxations for certifying
robustness to adversarial examples. In _Advances in Neural Information Processing Systems_, pp.
10877–10887, 2018.


Vicenc Rubies Royo, Roberto Calandra, Dusan M Stipanovic, and Claire Tomlin. Fast neural network verification via shadow prices. _arXiv preprint arXiv:1902.07247_, 2019.


Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, and Pengchuan Zhang. A convex relaxation
barrier to tight robustness verification of neural networks. In _Advances in Neural Information_
_Processing Systems 32_, pp. 9832–9842, 2019.


Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus P¨uschel, and Martin Vechev. Fast and
effective robustness certification. In _Advances in Neural Information Processing Systems_, pp.
10825–10836, 2018a.


Gagandeep Singh, Timon Gehr, Markus P¨uschel, and Martin Vechev. Boosting robustness certification of neural networks. In _International Conference on Learning Representations_, 2018b.


Gagandeep Singh, Rupanshu Ganvir, Markus P¨uschel, and Martin Vechev. Beyond the single neuron
convex barrier for neural network certification. In _Advances in Neural Information Processing_
_Systems_, pp. 15072–15083, 2019a.


Gagandeep Singh, Timon Gehr, Markus P¨uschel, and Martin Vechev. An abstract domain for certifying neural networks. _Proceedings of the ACM on Programming Languages_, 3(POPL):41,
2019b.


Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow,
and Rob Fergus. Intriguing properties of neural networks. In _ICLR_, 2013.


Christian Tjandraatmadja, Ross Anderson, Joey Huchette, Will Ma, Krunal Patel, and Juan Pablo
Vielma. The convex relaxation barrier, revisited: Tightened single-neuron relaxations for neural
network verification. _arXiv preprint arXiv:2006.14076_, 2020.


Vincent Tjeng, Kai Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed
integer programming. _ICLR_, 2019a.


Vincent Tjeng, Kai Y. Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed
integer programming. In _7th International Conference on Learning Representations, ICLR 2019,_
_New Orleans, LA, USA, May 6-9, 2019_ [. OpenReview.net, 2019b. URL https://openreview.net/](https://openreview.net/forum?id=HyGIdiRqtm)
[forum?id=HyGIdiRqtm.](https://openreview.net/forum?id=HyGIdiRqtm)


Shiqi Wang, Yizheng Chen, Ahmed Abdou, and Suman Jana. Mixtrain: Scalable training of formally
robust neural networks. _arXiv preprint arXiv:1811.02625_, 2018a.


Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, and Suman Jana. Efficient formal safety
analysis of neural networks. In _Advances in Neural Information Processing Systems_, pp. 6367–
6377, 2018b.


Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, and Suman Jana. Formal security analysis
of neural networks using symbolic intervals. In _27th {USENIX} Security Symposium ({USENIX}_
_Security 18)_, pp. 1599–1614, 2018c.


Tsui-Wei Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Luca Daniel, Duane Boning, and Inderjit Dhillon. Towards fast computation of certified robustness for relu networks. In
_International Conference on Machine Learning_, pp. 5273–5282, 2018.


11


Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer
adversarial polytope. In _International Conference on Machine Learning_, pp. 5283–5292, 2018.


Eric Wong, Frank Schmidt, Jan Hendrik Metzen, and J Zico Kolter. Scaling provable adversarial
defenses. In _NIPS_, 2018.


Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya
Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified
robustness and beyond. _Advances in Neural Information Processing Systems_, 33, 2020.


Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In _Advances in neural information_
_processing systems_, pp. 4939–4948, 2018.


Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards
stable and efficient training of verifiably robust neural networks. In _International Conference on_
_Learning Representations_, 2020.


12


A P ROOFS


**A.1** **Proof of Theorem 3.1**


We prove Theorem 3.1 by providing a simple counterexample, and we illustrate the necessity of
feasibility checking for the completeness of BaB based verification. Consider an NN with only two
ReLU units _g_ 1 [(2)] = ReLU( _h_ [(2)] 1 [)][ and] _[ g]_ 2 [(2)] = ReLU( _h_ [(2)] 2 [)][ where they share the same one dimension]
input _h_ [(2)] 1 = _h_ [(2)] 2 = _x_ . The final output function of NN is defined as _f_ = _g_ 1 [(2)] _−_ _g_ 2 [(2)] [. As a]
verification problem, we want to verify the property _f ≥_ 0 where _x_ = [ _−_ 1 _,_ 1]. Since hidden nodes
_h_ 1 and _h_ 2 are exactly the same, the ground-truth output range is _f_ _[∗]_ ( _x_ ) _∈_ [0 _,_ 0]. A complete BaB
based verifier is expected to obtain that optimal bound and prove the property after splitting _h_ 1 and
_h_ 2 together while BaB with only LiPRA cannot guarantee that completeness. Specifically, BaB with
only LiRPA will split the original domain _x ∈_ [ _−_ 1 _,_ 1] into four sub-domains and approximate the
bound with LiRPA respectively:


(1) (feasible) sub-domain _x ∈_ [ _−_ 1 _,_ 1] _, h_ [(2)] 1 _≥_ 0 _, h_ [(2)] 2 _≥_ 0 with output _f_ = [ _x, x_ ] _−_ [ _x, x_ ] _∈_ [0 _,_ 0]


(2) (feasible) sub-domain _x ∈_ [ _−_ 1 _,_ 1] _, h_ [(2)] 1 _<_ 0 _, h_ [(2)] 2 _<_ 0 with output _f_ = [0 _,_ 0] _−_ [0 _,_ 0] _∈_ [0 _,_ 0]


(3) (infeasible) sub-domain _x ∈_ [ _−_ 1 _,_ 1] _, h_ [(2)] 1 _<_ 0 _, h_ [(2)] 2 _≥_ 0 with output _f_ = [0 _,_ 0] _−_ [ _x, x_ ] _∈_ [ _−_ 1 _,_ 1]


(4) (infeasible) sub-domain _x ∈_ [ _−_ 1 _,_ 1] _, h_ [(2)] 1 _≥_ 0 _, h_ [(2)] 2 _<_ 0 with output _f_ = [ _x, x_ ] _−_ [0 _,_ 0] _∈_ [ _−_ 1 _,_ 1]


Only the first two split sub-domains are feasible and therefore the ground-truth lower bound 0 can be
obtained by taking the minimum of the estimated bounds from sub-domains (1) and (2). However,
pure LiRPA is not able to tell the infeasibility of sub-domains (3) and (4) and thus BaB with pure
LiRPA will report the minimum _−_ 1 got from all these four sub-domains as the global lower bound
for the original input domain, ending up not being able to verify the property, i.e., incomplete.


**A.2** **Proof of Theorem 3.2**


We prove Theorem 3.2 by considering the worst case where all unstable ReLU neurons are split.


Given a neural network function _f_ with input domain _C_, assume there are _N_ unstable ReLU neurons
_{g_ _i_ = ReLU( _h_ _i_ ) _|i_ = 1 _, · · ·, N_ _}_ in total. In the worst case, we have 2 _[N]_ leaf sub-domains _S_ =
_{C_ _i_ _|i_ = 1 _, · · ·,_ 2 _[N]_ _}_, where each _C_ _i_ corresponds to one assignment of unstable ReLU neuron splits.
For example, we can have
_C_ 1 = _C ∩_ ( _h_ 1 _≥_ 0) _∩_ ( _h_ 2 _≥_ 0) _∩· · · ∩_ ( _h_ _N_ _≥_ 0)
_C_ 2 = _C ∩_ ( _h_ 1 _<_ 0) _∩_ ( _h_ 2 _≥_ 0) _∩· · · ∩_ ( _h_ _N_ _≥_ 0)
_C_ 3 = _C ∩_ ( _h_ 1 _≥_ 0) _∩_ ( _h_ 2 _<_ 0) _∩· · · ∩_ ( _h_ _N_ _≥_ 0)
_C_ 4 = _C ∩_ ( _h_ 1 _<_ 0) _∩_ ( _h_ 2 _<_ 0) _∩· · · ∩_ ( _h_ _N_ _≥_ 0)

_· · ·_


Note that by definition the original input domain _C_ = _∪_ _C_ _′_ _∈S_ _C_ _[′]_ ; in other words, all the 2 _[N]_ split
sub-domains combined will be the same as the original input domain.


Not all of the sub-domains are actually feasible, due to the consistency requirements between neurons. For example, in our proof in Section A.1, _h_ [(2)] 1 and _h_ [(2)] 2 cannot be both _≥_ 0 or both _<_ 0.
We can divide the sub-domains _S_ into two mutually exclusive sub-sets, _S_ [feas] for all the feasible
sub-domains, and _S_ [infeas] for all the infeasible sub-domains. We have _C_ = _∪_ _C_ _′_ _∈S_ feas _C_ _[′]_ since these
infeasible sub-domains are empty sets.


We first show that linear programming (LP) can be used to effectively detect these infeasible subdomains. For some _C_ _[′]_ _∈S_ [infeas], because all the ReLU neurons are fixed to be positive or negative,
no relaxation is needed and the network is essentially linear; thus, the input value of every hidden
neuron _h_ _i_ can be written as a linear equation w.r.t. input _x_ . We add all the Boolean predicates on _h_ _i_
to a LP problem as linear constraints w.r.t _x_ . If this LP is feasible, then we can find some input _x_ 0
that assigns compatible values to all _h_ _i_ ; otherwise, the LP is infeasible.


Due to the lack of feasibility checking in LiRPA, the computed global lower (or upper) bounds
from LiRPA is _f_ ~~L~~ iRPA = min _C_ _[′]_ _∈S_ _f_ _C_ _′_ = min �min _C_ _′_ _∈S_ feas _f_ ~~_C_~~ _′_ _,_ min _C_ _′_ _∈S_ infeas _f_ ~~_C_~~ _′_ �. With feasibility


13


checking from LP, we can remove all infeasible sub-domains from this min such that they do not
contribute to the global lower bound: _f_ = min _C_ _′_ _∈S_ feas _f_ ~~_C_~~ _′_ .


To prove the whole BaB verification is complete, it is sufficient to prove this lower bound _f_ is the
exact minimum of _f_ bounded in _C_ . Since any sub-domain _C_ _[′]_ _∈S_ [feas] is a leaf sub-domain with
no unstable ReLU neurons, the neural network bounded within _C_ _[′]_ is a linear function. LiRPA can
give an exact minimum of _f_ within sub-domain _C_ _[′]_ . Since _C_ = _∪_ _C_ _′_ _∈S_ feas _C_ _[′]_ (in other words, _S_ [feas]
covers all the feasible sub-domains within _C_ ), the minimal value for all of them _f_ = min _C_ _′_ _∈S_ feas _f_ ~~_C_~~ _′_
forms the exact minimum of _f_ within the input domain _C_ . Thus, BaB with LiRPA based bounding
procedure is complete when feasibility checking is applied.


B E XPERIMENTAL S ETUP


We use the same set of models and benchmark examples used in the state-of-the-art verifiers GNNONLINE (Lu & Kumar, 2020) and B A BSR (Bunel et al., 2020b). Specifically, we evaluate on the
most challenging CIFAR-10 dataset with the same standard robustly trained convolutional neural
networks: Base, Wide, and Deep. These model structures are also used in (Lu & Kumar, 2020;
Bunel et al., 2020a). The Base model contains 2 convolution layers with 8 and 16 filters as well
as two linear layers with 100 and 10 hidden units, respectively. In total, the Base model has 3,172
ReLU activation units. The Wide model contains 2 convolution layers with 16 and 32 filters and
two linear layers with 100 and 10 hidden units, respectively, which contains 6,244 ReLU activation
units in total. The Deep model contains 4 convolution layers and all of them have 8 filters and
two linear layers with 100 and 10 hidden units, respectively, with 3,756 ReLU activation units
in total. The source code of B A BSR, MIP PLANET, GNN and GNN-O NLINE are available at
[https://github.com/oval-group/GNN](https://github.com/oval-group/GNN_branching) ~~b~~ ranching. The source code of BDD+ B A BSR is available at
[https://github.com/verivital/vnn-comp by replacing the dataset to the same one we used here.](https://github.com/verivital/vnn-comp)


Given an correctly classified image _x_ with label _y_ _c_, and another wrong label _y_ _c_ _′_ _̸_ = _y_ _c_ (pre-defined
in this benchmark) and _ϵ_, the verifier needs to prove:
( _**e**_ [(] _[c]_ [)] _−_ _**e**_ [(] _[c]_ _[′]_ [)] ) _[T]_ _f_ ( _x_ _[′]_ ) _>_ 0 s.t _∀x_ _[′]_ _∥x −_ _x_ _[′]_ _∥_ _∞_ _≤_ _ϵ_ (10)
where _f_ ( _·_ ) is the logit-layer output of a multi-class classifier, _**e**_ [(] _[c]_ [)] and _**e**_ [(] _[c]_ _[′]_ [)] are one-hot encoding
vectors for labels _y_ _c_ and _y_ _c_ _′_ . We want to verify that for a given _ϵ_, the trained classifier will not
predict wrong label _y_ _c_ _′_ for image _x_ . All properties including _x_, _ϵ_, and _c_ _[′]_ are provided by (Lu &
Kumar, 2020). Specifically, they categorize verification properties solved by B A BSR within 800s
as easy, between 800s and 2400s as medium and more than 2400s as hard.


Our experiments are conducted on one Intel I7-7700K CPU and one Nvidia GTX 1080 Ti GPU. The
parallel batch size _n_ is set to 400, 200 and 200 for base, wide and deep model respectively and the
_η_ is set to 12,000 due to GPU memory constraint. To make a fair comparison, we use one CPU core
for all methods. Also, we use one GPU for GNN, GNN-O NLINE, BDD+ B A BSR and our method.
When optimizing the LiRPA bounds, we apply 100 steps gradient decent for obtaining the initial _f_
(Line 2 in Algorithm 1). After that, we use 10 steps gradient decent (Line 7) and early stop once
_f >_ 0 or _f_ has no improvement.


C A BLATION STUDY


Our efficient framework leverages two powerful components: (1) optimized LiRPA bounds and
(2) batch splits on GPUs. In this section, we conduct breakdown experiments to show how each
individual technique can help with complete verification. As we can see in Table 2, using batched
split with unoptimized LiRPA is not very successful and cannot beat B A BSR. We observe that,
without optimized LiRPA, the bounds are very loose and cannot quickly improve the global lower
bound. In contrast, using optimized LiRPA bounds without batch splits (splitting a single node at
a time and running a batch size of 1 on GPU) can still significantly speed up complete verification,
around 2 _∼_ 10X compared to B A BSR. Finally, combining batch splits and optimized LiRPA allows
us to gain up to 44X speedup compared to B A BSR.


D C OMPLETE VERIFICATION WITH LiRPA ON CPU VS GPU


For a fair comparison, we only use one CPU core and one GPU (the same as GNN and GNNO NLINE ) in our experimental results in Section 4. In this section, we investigate the performance of


14


Table 2: Ablation study for different components of our algorithm. The speedup rate is computed based on
running time of B A BSR baseline: speedup = Time of BaBSR _/_ Time of our method.


Easy Medium Hard Wide Deep


Method time(s) speedup time(s) speedup time(s) speedup time(s) speedup time(s) speedup


B A BSR baseline 522.48 – 1335.40 – 2875.16 – 3325.65 – 2855.19 –
Batch Splits (unoptimized LiRPA) 587.10 0.89 1470.02 0.91 3013.57 0.95 3457.30 0.96 2998.50 0.95
Optimized LiRPA (no batch splits) 94.08 5.58 361.53 3.70 1384.22 2.07 736.56 4.51 287.33 9.94
Optimized LiRPA & Batch Splits **11.86** 44.05 **42.04** 31.80 **633.85** 4.53 **375.23** 8.86 **81.55** 35.01


our algorithm for the cases where one or multiple CPU cores are available without GPU acceleration. Note that existing baselines such as B A BSR and MIP PLANET can only effectively utilize one
CPU core subject to the Gurobi solver. GNN and GNN-O NLINE can utilize one GPU to run the
GNN during branching while the rest of the verification processes all perform on one CPU core. In
contrast, our method is much more flexible, and we are not limited by the number of CPU cores or
GPUs. When running on multi-core CPUs, LiRPA can be automatically accelerated by the underlying linear algebra library (e.g., Intel MKL or OpenBLAS) since the main computation of LiRPA is
just matrix multiplications.


In Figure 5, we show the performance of our algorithm on a single CPU core and multiple CPU
cores (in blue), and compare it to our main results with one CPU core plus one GPU (in red). As
we can see, the running time decreases when the number of CPU cores increases, but the speedup
is not linear due to the limitation of the underlying linear algebra library and hardware. There is
a big gap between the running time on 8 CPU cores and the time on one CPU core + one GPU,
and the performance gap is more obvious on Wide and Deep models. Thus, the speedup of LiRPA
computation on GPUs is significant. However, surprisingly, even when using only one CPU core,
we are still significantly faster than baseline B A BSR and also get very competitive performance
when compared to GNN-O NLINE which needs one GPU additionally. This shows the efficiency of
LiRPA based verification algorithms.














|200<br>175<br>150<br>125 time(s)<br>100<br>75 n CPU core(s)<br>50 One CPU core + one GPU<br>GNN Online<br>25<br>1 2 4 8<br>number of CPU core|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|1<br>2<br>4<br>8<br>number of CPU core<br>25<br>50<br>75<br>100<br>125<br>150<br>175<br>200<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>GNN Online|||||
|1<br>2<br>4<br>8<br>number of CPU core<br>25<br>50<br>75<br>100<br>125<br>150<br>175<br>200<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>GNN Online||~~n CP~~<br>One <br>|~~U core(s)~~<br>CPU core + one GPU<br>||
|1<br>2<br>4<br>8<br>number of CPU core<br>25<br>50<br>75<br>100<br>125<br>150<br>175<br>200<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>GNN Online||GNN|Online||


|1600<br>1400 n CPU core(s)<br>One CPU core + one GPU<br>1200 GNN Online<br>time(s)<br>1000<br>800<br>600<br>400<br>1 2 4<br>number of CPU core|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|1<br>2<br>4<br><br>number of CPU core<br>400<br>600<br>800<br>1000<br>1200<br>1400<br>1600<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||~~n CPU~~<br>One C<br>~~GNN ~~|~~core(s)~~<br>PU core + one GPU<br>~~Online~~|
|1<br>2<br>4<br><br>number of CPU core<br>400<br>600<br>800<br>1000<br>1200<br>1400<br>1600<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||||
|1<br>2<br>4<br><br>number of CPU core<br>400<br>600<br>800<br>1000<br>1200<br>1400<br>1600<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||||


|1750<br>1500 n CPU core(s)<br>One CPU core + one GPU<br>1250 GNN Online<br>time(s)<br>1000<br>750<br>500<br>250<br>0<br>1 2 4<br>number of CPU core|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|1<br>2<br>4<br><br>number of CPU core<br>0<br>250<br>500<br>750<br>1000<br>1250<br>1500<br>1750<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||~~n CPU~~<br>One C<br>~~GNN ~~|~~core(s)~~<br>PU core + one GPU<br>~~Online~~|
|1<br>2<br>4<br><br>number of CPU core<br>0<br>250<br>500<br>750<br>1000<br>1250<br>1500<br>1750<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||||
|1<br>2<br>4<br><br>number of CPU core<br>0<br>250<br>500<br>750<br>1000<br>1250<br>1500<br>1750<br>time(s)<br>~~n CPU core(s)~~<br>One CPU core + one GPU<br>~~GNN Online~~|||||



Base Model (easy instances) Wide Model Deep Model
(B A BSR: 522.48s) (B A BSR: 3325.65s) (B A BSR: 2855.19s)


Figure 5: Running time of our method on the Base, Wide, and Deep networks when using 1, 2, 4
and 8 CPU cores without a GPU (blue), and our method using 1 CPU core + 1 GPU (red) and a
strong baseline method, GNN-O NLINE (green). We report the baseline B A BSR verification time in
captions because they are out of range on the figures.


15


