**Efficient nonparametric** _**n**_ **-body force fields from machine learning**


Aldo Glielmo, [1,] _[ ∗]_ Claudio Zeni, [1,] _[ †]_ and Alessandro De Vita [1, 2]

1 _Department of Physics, King’s College London, Strand, London WC2R 2LS, United Kingdom_
2 _Dipartimento di Ingegneria e Architettura, Università di Trieste, via A. Valerio 2, I-34127 Trieste, Italy_


We provide a definition and explicit expressions for _n_ -body Gaussian Process (GP) kernels, which
can learn any interatomic interaction occurring in a physical system, up to _n_ -body contributions, for
any value of _n_ . The series is complete, as it can be shown that the “universal approximator” squared
exponential kernel can be written as a sum of _n_ -body kernels. These recipes enable the choice of
optimally efficient force models for each target system, as confirmed by extensive testing on various
materials. We furthermore describe how the _n_ -body kernels can be “mapped” on equivalent representations that provide database-size-independent predictions and are thus crucially more efficient.
We explicitly carry out this mapping procedure for the first non-trivial (3-body) kernel of the series,
and we show that this reproduces the GP-predicted forces with meV/Å accuracy while being orders
of magnitude faster. These results pave the way to using novel force models (here named “M-FFs”)
that are computationally as fast as their corresponding standard parametrised _n_ -body force fields,
while retaining the nonparametric character, the ease of training and validation, and the accuracy
of the best recently proposed machine learning potentials.



**I.** **INTRODUCTION**


Since their conception, first-principles molecular dynamics (MD) simulations [1] based on density functional theory (DFT) [2, 3] have proven extremely useful to investigate complex physical processes that require
quantum accuracy. These simulations are computationally expensive, and thus still typically limited to hundreds of atoms and the picosecond timescale. For larger
systems that are non-uniform and thus intractable using periodic boundary conditions, multiscale embedding
(“QM/MM”) approaches can sometimes be used successfully. This is possible if full quantum accuracy is only
needed in a limited (QM) zone of the system, while a
simpler molecular mechanics (MM) description suffices
everywhere else. Very often, however, target problems
require minimal model system sizes and simulation times
so large that the calculations must be exclusively based
on classical force fields i.e., force models that use the position of atoms as the only explicitly represented degrees
of freedom.
In the remainder of this introductory section we briefly
review the relative strengths and weaknesses of standard
parametrized (P-FFs) and machine learning force fields
(ML-FFs). We then consider how accurate P-FFs are
hard to develop but eventually fully exploit useful knowledge on the systems, while GP-based ML-FFs offer a
general mathematical framework for handling training
and validation, but are significantly slower (Section I A).
These shortcomings motivate an analysis of how prior
knowledge such as symmetry has been so far incorporated in GP kernels (Section I B) and points to features
still missing in ML kernels, which are commonplace in
the more standard, highly efficient P-FFs based on truncated _n_ -body expansions (Section I C). This suggests the


_∗_ [aldo.glielmo@kcl.ac.uk](mailto:aldo.glielmo@kcl.ac.uk)

_†_ [claudio.zeni@kcl.ac.uk](mailto:claudio.zeni@kcl.ac.uk)



possibility of defining a series of _n_ -body GP kernels (Section II B), providing a scheme to construct them (Section
II C and D) and, after the best value of _n_ for the given
target system has been identified with appropriate testing (Section II E), exploiting their dimensionally-reduced
feature spaces to massively boost the execution speed of
force prediction (Section III).


**A.** **Parametrized and machine learning force fields**


Producing accurate and fully transferable force fields is
a remarkably difficult task. The traditional way to do this
involves adjusting the parameters of carefully chosen analytic functions in the hope of matching extended reference
data sets obtained from experiments or quantum calculations [4, 5]. The descriptive restrictiveness of the parametric functions used is both a drawback and a strength
of this methodology. The main difficulty is that developing a good parametric function requires a great deal
of chemical intuition and patient effort, guided by trial
and error steps with no guarantee of success [6]. However, for systems and processes in which the approach
is fruitful, the development effort is amply rewarded by
the opportunity to provide extremely fast and accurate
force models [7–10]. The identified functional forms will
in these cases contain valuable knowledge on the target
system, encoded in a compact formulation that still accurately captures the relevant physics. Such knowledge
is furthermore often transferable to novel (while similar)
systems as a “prior” piece of information, i.e., it constitutes a good working hypothesis on how these systems
will behave. When QM data on the novel system become available, this can be simply used to fine-tune the
parameters of the functional form to a new set of best-fit
values that maximise prediction accuracy.
Following a different approach, “nonparametric” ML
force fields can be constructed, whose dependence on
the atomic position is not constrained to a particular


analytic form. An implementation and tests exploring the feasibility of ML to describe atomic interactions
can be found, e.g., in pioneering work by Skinner and
Broughton [11] that proposed using ML models to reproduce first-principles potential energy surfaces. More
recent works implementing this general idea have been
based on Neural Networks [12], Gaussian Process (GP)
regression [13] or linear regression on properly defined
bases [14]. Current work aims at making these learning
algorithms both faster and more accurate [15–20].
As processing power and data communication bandwidth increase, and the cost of data storage decreases,
modeling based on ML and direct inference promises
to become an increasingly attractive option, compared
with more traditional classical force field approaches.
However, although ML schemes are general and have
been shown to be remarkably accurate interpolators in
specific systems, so far they have not become as widespread as it might have been expected. This is mainly
because “standard” classical potentials are still orders
of magnitude faster than their ML counterpart [21].
Moreover, ML-FFs also involve a more complex mathematical and algorithmic machinery than the traditional
compact functional forms of P-FFs, whose arguments are
physically descriptive features that remain easier to visualize and interpret.


**B.** **Prior knowledge and GP kernels**


These shortcomings provide motivation for the present
work. The high computational cost of many ML models is a direct consequence of the general inverse relation
between the sought flexibility and the measured speed of
any algorithm capable of learning. Highly flexible ML
algorithms by definition assume very little or no prior
knowledge of the target systems. In a Bayesian context,
this involves using a general prior kernel, typically aspiring to preserve the full universal approximator properties
of e.g., the square exponential kernel [22, 23]. The price
of such a kernel choice is that the ML algorithm will
require large training databases [24], slowing down computations as the prediction time grows linearly with the
database size.

Large database sizes are not, however, unavoidable,
and any data-intensive and fully flexible scheme to potential energy fitting is suboptimal by definition, as it exploits no prior knowledge of the system. This completely
“agnostic” approach is at odds with the general lesson
from classical potential development, indicating that it is
essential for efficiency to incorporate in the force prediction model as much prior knowledge of the target system
as can be made available. In this respect, GP kernels _can_
be tailored to bring some form of prior knowledge to the
algorithm.
For example, it is possible to include any symmetry
information of the system. This can be done by using
descriptors that are independent of rotations, transla


2


tions and permutations [15, 25–27]. Alternatively, one
can construct scalar-valued GP kernels that are made invariant under rotation (see e.g., [16, 28]) or matrix-valued
GP kernels made covariant under rotation ([16], an idea
that can be extended to higher-order tensors [29, 30]). Invariance or covariance are in these cases obtained starting from a non-invariant representation by appropriate
integration over the _SO_ (3) rotation group [16, 28].
Symmetry aside, progress can be made by attempting
to use kernels based on simple, descriptive features corresponding to low-dimensional feature spaces. Taking inspiration from parametrized force fields, these descriptors
could e.g., be chosen to be interatomic distances taken
singularly or in triplets, yielding kernels based on 2- or
3-body interactions [16, 31, 32]. Since low-dimensional
feature spaces allow efficient learning (convergence is
reached using small databases), to the extent that simple
descriptors capture the correct physics, the GP process
will be a relatively fast, while still very accurate, interpolator.


**C.** **Scope of the present work**


There are, however, two important aspects that have
not as yet been fully explored while trying to develop
efficient kernels based on dimensionally reduced feature
spaces. Both aspects will be addressed in the present
work.

First, a systematic classification of rotationally invariant (or covariant, if matrix valued) kernels, representative of the feature spaces corresponding to _n_ -body interactions is to date still missing. Namely, no definition or
general recipe has been proposed for constructing _n_ -body
kernels, or for identifying the actual value (or effective
interval of values) of _n_ associated with already available
kernels. This would be clearly useful, however, as the
discussion above strongly suggests that the kernel corresponding to the lowest value of _n_ compatible with the
physics of the target system will be the most informationally efficient one for carrying out predictions: striking the
right balance between speed and accuracy.
Second, for any ML approach based on a GP kernel and
a fixed database, the GP predictions for any target configuration are also fixed once and for all. For an _n_ -body
kernel, these predictions do not need, however, to be explicitly carried out as sums over the training dataset, as
they could be approximated with arbitrary precision by
“mapping” the GP prediction on a new representation
based on the underlying _n_ -body feature space. We note
that this approximation step would make the final prediction algorithm independent of the database size, and
thus in principle as fast as any classical _n_ -body potential
based on functional forms, while still parameter free. The
remainder of this work explores these two issues, and it
is structured as follows.

In the next Section II, after introducing the terminology and the notation (II A), we provide a definition of


an _n_ -body kernel (II B) and we propose a systematic way
of constructing _n_ -body kernels of any order _n_, showing
how previously proposed approaches can be reinterpreted
within this scheme (II C and D). We furthermore show,
by extensive testing on a range realistic materials, how
the optimal interaction order can be chosen as the lowest
_n_ compatible with the required accuracy and the available computational power (II E). In the following Section
III we describe how the predictions of _n_ -body GP kernels
can be recast (mapped) with arbitrary accuracy into very
fast nonparametric force fields based on machine learning
(M-FFs) which fully retain the _n_ -body character of the
GP process from which they were derived. The procedure is carried out explicitly for a 3-body kernel, and we
find that evaluating atomic forces is orders of magnitude
faster than the corresponding GP calculation.


**II.** _**n**_ **-BODY EXPANSIONS WITH** _**n**_ **-BODY**

**KERNELS**


**A.** **Notation and terminology**


GP-based potentials are usually constructed by assigning an energy _ε_ to a given atomic configuration _ρ_, typically including a central atom and all its neighbors up to a
suitable cutoff radius. The existence of a corresponding
local energy function _ε_ ( _ρ_ ) is generally assumed, in order
to provide a total energy expression and guarantee a linear scaling of the predictions with the total number of
atoms in the system. Within GP regression this function
is calculated from a database _D_ = _{ρ_ _d_ _, ε_ _d_ _,_ **f** _d_ _}_ _[N]_ _d_ =1 [of ref-] _̸_
erence data, typically obtained by quantum mechanical
simulations, and usually consisting of a set of _N_ atomic
configurations _{ρ_ _d_ _}_ together with their relative energies
_{ε_ _d_ _}_ and/or forces _{_ **f** _d_ _}_ .
It is worth noting here that although there is no well
defined local atomic energy in a reference quantum simulation, one can always use gradient information (atomic
forces, which are well defined local physical quantities)
to machine-learn a potential energy function. This can
be done straightforwardly using derivative kernels (cf.,
e.g., Ref. [22] or Ref. [33]) to learn and predict forces.
Alternatively, one can learn forces directly without an intermediate energy expression, as done in Refs. [15, 34] or
more recently in Ref. [16]. A necessary condition for any
of these approaches to produce energy-conserving force
fields (i.e., fields that make zero work on any closed trajectory loop) is that the database is constructed once
and for all, and never successively updated. After training on the given fixed database, the GP prediction on a
target configuration _ρ_ consists of a linear combination of
the kernel function values measuring the similarity of the
target configuration with each database entry:



3


where the coefficients _α_ _d_ are obtained by means of inversion of the _covariance matrix_ [22] and can be shown
to minimise the regularised quadratic error between GP
predictions and reference calculations.


**B.** **Definition of an** _**n**_ **-body kernel**


Classical interatomic potentials are often characterized by the number of atoms (“bodies”) they let interact simultaneously (cf. e.g., Refs. [9, 10]). To translate
this concept into the realm of GP regression, we assume
that the target configuration _ρ_ ( _{_ **r** _i_ _}_ ) represents the local
atomic environment of an atom fixed at the origin of a
Cartesian reference frame, expressed in terms of the relative positions **r** _i_ of the surrounding atoms. We define
the order of a kernel _k_ _n_ ( _ρ, ρ_ _[′]_ ) as the smallest integer _n_
for which the following property holds true:


_∂∂_ **r** _[n]_ _i_ 1 _k · · ·_ _n_ ( _ρ, ∂ρ_ **r** _[′]_ _i_ ) _n_ = 0 _∀_ **r** _i_ 1 _̸_ = **r** _i_ 2 _̸_ = _· · · ̸_ = **r** _i_ _n_ _,_ (2)


where **r** _i_ 1 _, . . .,_ **r** _i_ _n_ are the positions of any choice of a
set of _n_ different surrounding atoms. By virtue of linearity, the local energy in Eq. (1) will also satisfy the
same property if _k_ _n_ does. Thus, Eq. (2) implies that the
central atom in a local configuration interacts with up
to _n −_ 1 other atoms simultaneously, making the interaction energy term _n_ -body. For instance, using a 2-body
kernel, the force on the central atom due to atom **r** _j_ will
not depend on the position of any other atom **r** _l_ = _̸_ _j_ belonging to the target configuration _ρ_ ( _{_ **r** _i_ _}_ ). Eq. (2) can
be used directly to check through either numeric or symbolic differentiation if a given kernel is of order _n_, a fact
that might be far from obvious from its analytic form,
depending on how the kernel is built.


**C.** **Building** _**n**_ **-body kernels I: SO(3) integration**


Following a standard route [16, 28, 35], we begin by
representing each local atomic configuration as a sum of
Gaussian functions _N_ with a given variance _σ_ [2], centered
on the _M_ atoms of the configuration:



_̸_


_ρ_ ( **r** _, {_ **r** _i_ _}_ ) =



_̸_


_M_
� _N_ ( **r** _|_ **r** _i_ _, σ_ [2] ) _,_ (3)


_i_ =1



_̸_


where **r** and _{_ **r** _i_ _}_ _[M]_ _i_ =1 [are position vectors relative to the]
central atom of the configuration. This representation
guarantees by construction invariance with respect to
translations and permutations of atoms (here assumed
to be of a single chemical species). As described in [16],
a covariant 2-body force kernel can be constructed from
the non-invariant scalar (“base”) kernel obtained as a dot



_̸_


_ε_ ( _ρ_ ) =



_̸_


_N_
� _k_ ( _ρ, ρ_ _d_ ) _α_ _d_ _,_ (1)


_d_ =1


4



product overlap integral of the two configurations:


_k_ 2 ( _ρ, ρ_ _[′]_ ) = _d_ **r** _ρ_ ( **r** ) _ρ_ _[′]_ ( **r** )
�



= _L_ � e _[−]_ [(] **[r]** _[i]_ _[−]_ **[r]** _j_ _[′]_ [)] [2] _[/]_ [4] _[σ]_ [2] _,_ (4)

_i∈ρ,j∈ρ_ _[′]_



1E00


1E −01


1E −02


1E −03


1E −04


1E −05


1E −06



where _L_ is an unessential constant factor, omitted for
convenience from now on. That (4) is a 2-body kernel consistent with the definition of Eq. (2) can be
checked straightforwardly by explicit differentiation (see
Appendix A). Its 2-body structure is also readable from
the fact that _k_ 2 is a sum of contributions comparing pairs
of atoms in the two configurations, the first pair located
at the two ends of vector **r** _i_ in the target configuration _ρ_,
and consisting of the central atom and atom _i_, and the
second pair similarly represented by the vector **r** _[′]_ _j_ [in the]
database configuration _ρ_ _[′]_ . A rotation-covariant matrixvalued force kernel can at this point be constructed by
Haar integration [36, 37] as an integral over the _SO_ (3)
manifold [16]:







**K** _[s]_ 2 [(] _[ρ, ρ]_ _[′]_ [) =] _dR_ **R** _k_ 2 ( _ρ, Rρ_ _[′]_ ) _._ (5)
� _SO_ (3)



2 3 4 5

Interaction order


Figure 1. GP relative error as a function of the interaction
order (2- to 5-body), using _n_ -body kernels with increasing
_n_ . Learning energies within baseline precision (black dashed
line) requires an _n_ -body kernel with _n_ at least as high as the
particles’ interaction order.


simple 1D model consisting of _n_ _[′]_ particles interacting via
an _ad-hoc n_ _[′]_ -body potential (see Appendix B.). We first
let the particles interact to generate a configuration database, and then attempt to machine-learn these interactions using the kernels just described. Figure 1 illustrates
the average prediction errors on the local energies of this
system incurred by the GP regression based on four different kernels as a function of the interaction order _n_ _[′]_ .
It is clear from the graph that a force field that lets the
_n_ _[′]_ particles interact simultaneously can only be learned
accurately with a ( _n ≥_ _n_ _[′]_ )-body kernel (6), or with the
many-body exponential kernel (7) which contains all interaction orders.
To construct _n_ -body kernels useful for applications to
real 3D systems we need to include rotational symmetry
by averaging over the rotation group. For our present
scopes, it is sufficient to discuss the case of rotationinvariant _n_ -body scalar energy kernels, for which the
integral (formally a _transformation integration_ [38]) is
readily obtained from Eq. (5) by simply dropping the **R**
matrix in the integrand:



This kernel can be used to infer forces on atoms using a
GP regression vector formula analogous to Eq. (1) (see
Ref. [16]). These forces belong to a 2-body force field
purely as a consequence of the base kernel property in
Eq. (2). It is interesting to notice that there is no use or
need for an intermediate energy expression to construct
this 2-body force field, which is automatically energyconserving.
Higher order _n_ -body base kernels can be constructed
as finite powers of the 2-body base kernel (4):


_k_ _n_ ( _ρ, ρ_ _[′]_ ) = _k_ 2 ( _ρ, ρ_ _[′]_ ) _[n][−]_ [1] (6)


where the _n_ -body property (Eq. (2)) can once more be
checked by explicit differentiation (see Appendix A). Furthermore, taking the exponential of the kernel in Eq. (4)
gives rise to a fully many-body base kernel, as all powers
of _k_ 2 are contained in the exponential formal series expansion:


_k_ _MB_ ( _ρ, ρ_ _[′]_ ) = e _[k]_ [2] [(] _[ρ,ρ]_ _[′]_ [)] _[/θ]_ [2]



= 1 + _θ_ [1] [2] _[k]_ [2] [ +] 2!1 _θ_ [4] _[k]_ [3] [ +] _[ . . . .]_ (7)



The use of this integral in the context of potential energy learning was originally proposed in [28], where it
was carried out using appropriate functional expansions.
Alternatively, one can exploit the Gaussian nature of
the configuration expansion (3) to obtain an analytically exact formula, as done further below. The resulting
symmetrized _n_ -body kernel _k_ _n_ _[s]_ [will learn faster than its]
non-symmetrized counterpart _k_ _n_, as the rotational degrees of freedom have been integrated out. This is because a non-symmetrized _n_ -body kernel ( _k_ _n_ ) must learn
functions of 3 _n −_ 3 variables (translations are taken into
account by the local representation based on relative po


_k_ _n_ _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] _dR k_ _n_ ( _ρ, Rρ_ _[′]_ ) _._ (10)
� _SO_ (3)



One can furthermore check that the simple exponential
many-body kernel _k_ _MB_ defined above is, up to normalisation, equivalent to the squared exponential kernel [22]
on the natural distance induced by the dot product kernel
_k_ 2 ( _ρ, ρ_ _[′]_ ):


e _[−][d]_ [2] [(] _[ρ,ρ]_ _[′]_ [)] _[/]_ [2] _[θ]_ [2] = _N_ ( _ρ_ ) _N_ ( _ρ_ _[′]_ ) _k_ _MB_ ( _ρ, ρ_ _[′]_ ) (8)

_d_ [2] ( _ρ, ρ_ _[′]_ ) = _k_ 2 ( _ρ, ρ_ ) + _k_ 2 ( _ρ_ _[′]_ _, ρ_ _[′]_ ) _−_ 2 _k_ 2 ( _ρ, ρ_ _[′]_ ) _._ (9)


To check on these ideas, we next test the accuracy of
these kernels in learning the interactions occurring in a


5



sition in Eq. (3)). After integration, the new kernel _k_ _n_ _[s]_
defines a smaller and more physically-based space of functions of 3 _n −_ 6 variables, which is the rotation-invariant
functional domain of _n_ interacting particles.
The symmetrization integral in Eq. (10) can be written
down for the many-body base kernel _k_ _MB_ (Eq. (7)), to
define a new many-body kernel _k_ _MB_ _[s]_ [invariant under all]
physical symmetries:



1.0 n = 2 n = 3


0.5


0.0




0.0 0.5 1.0



_k_ _MB_ _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] _dR k_ _MB_ ( _ρ, Rρ_ _[′]_ ) _._ (11)
� _SO_ (3)



0.5


0.0

0.0 0.5 1.0



By virtue of the universal approximation theorem [22, 39]
this kernel would be able to learn arbitrary physical interactions with arbitrary accuracy, if provided with sufficient data.
Unfortunately, the exponential kernel (7) has to date
resisted all attempts to carry out the analytic integration over rotations (11), leaving as the only open options
numerical integration, or discrete summation over a relevant point group of the system [16]. On the other hand,
the analytic integration of 2- or 3-body kernels to give
symmetric _n_ -body kernels can be carried out in different

ways.
For example, one could use an intermediate step that
was introduced during the construction of the widely
used SOAP kernel [28, 31, 40–42]. This kernel has a
full many-body character [28], ensured by the prescribed
normalisation step defined by Eqs. (31-36) of the standard Ref. [28]), which made it possible to use it e.g., to
augment to full many-body the descriptive power of a 2and 3-body explicit kernel expansion [26]. However, the
Haar integral over rotations introduced in [28] as an intermediate kernel construction step could also be seen, if
taken on its own, as a transformation integration procedure [38] yielding a symmetrized _n_ -body kernel as defined
in Eq. (10) above, which would in turn become a higher
finite-order kernel if raised to integer powers _ζ ≥_ 2 (see
next subsection).
Carrying out Haar integrals is not, in general, an easy
task. In the example above, computing a general rotation
invariant _n_ -body kernel via the exact, suitably truncated
spherical harmonics expansion procedure of Ref. [28] becomes challenging for _n >_ 3. Significant difficulties likewise arise if attempting a “covariant” integration over rotations, for which we found an exact analytic expression
only for 2- and 3-body matrix-valued kernels [16], with a
technique that becomes unviable for _n >_ 3. Fortunately,
the Haar integration can be avoided altogether, following the simple route of constructing symmetric _n_ -kernels
directly using symmetry-invariant descriptors, as we will
see in the next section. The problem of obtaining an
analytic Haar integral expression for the general _n_ case
remains, however, an interesting one, which we tackle in
the remainder of this section following a novel analytic

route.
We first write the _n_ -body base kernel of Eq. (6) as an
explicit product of ( _n −_ 1) 2-body kernels. The Haar



where now for each of the two configurations _ρ, ρ_ _[′]_, the
sum runs over all _n_ -plets of atoms that include the central
atom (whose indices _i_ 0 and _j_ 0 are thus omitted). Expanding the exponents as ( **r** _i_ _−_ **Rr** _[′]_ _j_ [)] [2] [ =] _[ r]_ _i_ [2] [+] _[r]_ _j_ _[′]_ [2] _[−]_ [2][Tr][(] **[Rr]** _[′]_ _j_ **[r]** _i_ [T] [)]
allows us to extract from the integral (13) a rotation
independent constant _C_ **i** _,_ **j**, and to express the rotationdependent scalar products sum as a trace of a matrix
product:


_k_ ˜ **i** _,_ **j** = _C_ **i** _,_ **j** _I_ **i** _,_ **j** (14)

_C_ **i** _,_ **j** = e _[−]_ [(] _[r]_ _i_ [2] 1 [+] _[r]_ _j_ _[′]_ [2] 1 [+] _[...r]_ _in_ [2] _−_ 1 [+] _[r]_ _jn_ _[′]_ [2] _−_ 1 [)] _[/]_ [4] _[σ]_ [2] (15)


_I_ **i** _,_ **j** = _dR_ e [Tr(] **[RM]** **[i]** _[,]_ **[j]** [)] (16)
�


where the matrix **M** **i** _,_ **j** is the sum of the outer products
of the ordered vector couples in the two configurations:
**M** **i** _,_ **j** = ( **r** _[′]_ _j_ 1 **[r]** _i_ [T] 1 [+] _[ · · ·]_ [ +] **[ r]** _[′]_ _j_ _n−_ 1 **[r]** _i_ [T] _n−_ 1 [)] _[/]_ [2] _[σ]_ [2] [. The integral (16)]
occurs in the context of multivariate statistics as the generating function of the non-central Wishart distribution

[43]. As shown in [44], it can be expressed as a power
series in the symmetric polynomials ( _α_ 1 = [�] [3] _i_ _[µ]_ _[i]_ _[, α]_ [2] [ =]
� 3 _i<j_ _[µ]_ _[i]_ _[µ]_ _[j]_ _[, α]_ [3] [ =] _[ µ]_ [1] _[µ]_ [2] _[µ]_ [3] [) of the eigenvalues] _[ {][µ]_ _[i]_ _[}]_ _i_ [3] =1 [of]



Numerical integral


Figure 2. Scatter plots showing the values of the integral
in (13) (on a random selection of configurations) computed
either by numerical integration or via the analytic expression
(Eqs. (14, 15, 17)). Interaction orders from _n_ = 2 to _n_ = 5
are considered.


integral (10) can then be written as



_k_ _n_ _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] �



_k_ ˜ **i** _,_ **j** (12)



**i** =( _i_ 1 _,...,i_ _n−_ 1 ) _∈ρ_
**j** =( _j_ 1 _,...,j_ _n−_ 1 ) _∈ρ_ _[′]_



**Rr** _[′]_ _j_ 1 _[∥]_ [2] _∥_ **r** _in−_ 1 _−_ **Rr** _[′]_ _jn−_ 1 _[∥]_ [2]

4 _σ_ ~~[2]~~ _. . ._ e _[−]_ 4 _σ_ ~~[2]~~



˜ _∥_ **r** _i_ 1 _−_ **Rr** _[′]_ _j_ 1 _[∥]_ [2]
_k_ **i** _,_ **j** = _dR_ e _[−]_ 4 _σ_ ~~[2]~~
�



4 _σ_ ~~[2]~~ (13)


6



the symmetric matrix **M** [T] **i** _,_ **j** **[M]** **[i]** _[,]_ **[j]** [:]





_I_ **i** _,_ **j** = � _A_ _p_ 1 _p_ 2 _p_ 3 _α_ 1 _[p]_ [1] _[α]_ 2 _[p]_ [2] _[α]_ 3 _[p]_ [3] (17)


_p_ 1 _,p_ 2 _,p_ 3



0.2


0.18



_π_ 2 _[−]_ [(1+2] _[p]_ [1] [+4] _[p]_ [2] [+6] _[p]_ [3] [)] ( _p_ 1 + 2 _p_ 2 + 4 _p_ 3 )!
_A_ _p_ 1 _p_ 2 _p_ 3 = _p_ 1 ! _p_ 2 ! _p_ 3 !Γ( [3] 2 [+] _[ p]_ [1] [ + 2] _[p]_ [2] [ + 3] _[p]_ [3] [)Γ(1 +] _[ p]_ [2] [ + 2] _[p]_ [3] [)]



1
_×_ (18)
Γ( [1] 2 [+] _[ p]_ [3] [)(] _[p]_ [1] [ + 2] _[p]_ [2] [ + 3] _[p]_ [3] [)!] _[.]_



0.16


0.14


0.12


0.1


0.08

10 100 1000

Training points


Figure 3. Learning curves for 2 and 3-body kernels obtained
either via a Haar integration (Eqs. (12-18)), or directly specifying a similarity kernel function of the effective degrees of
freedom (Eqs. (19, 20)).


within the two configurations. Since these kernels learn
functions of low-dimensional spaces, their exact analytic
form is not essential for performance, as any fully nonlinear function _k_ [˜] will give equivalent converged results in
the rapidly reached large-database limit. This equivalence can be neatly observed in Figure 3, which reports the
performance of 2- and 3-body kernels built either directly
over the set of distances (Eqs. (19) and (20)) or via the
exact Haar integral (Eqs. (12-18)). As the test system is
crystalline Silicon, 3-body kernels are better performing.
However, since convergence of the 2- and 3-body feature
space is quickly achieved (at about _N_ = 50 and _N_ = 100
respectively), there is no significant performance difference between _SO_ (3)-integrated _n_ -body kernels and physically motivated ones. Consequently, for low interaction
orders, simple and computationally fast kernels like the
ones in Eqs. (19, 20) are always preferable to more complex (and heavier) alternatives obtained via integration
over rotations (e.g., the one defined by Eqs. (12-18) or
those found in Refs. [16, 28].
We note at this point that Eq. (19) can be generalized
to construct a symmetric _n_ -body kernel



Remarkably, in this result (whose exactness is checked
numerically in Figure 2) the integral over rotations does
not depend on the order _n_ of the base kernel, once the
matrix **M** **i** _,_ **j** is computed. This is not the case for previous approaches to integrating over rotations [16, 28] that
need to be reformulated with increasing and eventually
prohibitive difficulty each time the order _n_ needs to be
increased.
However, the final expression given by Eqs. (14-18)
is still a relatively complex and computationally heavy
function of the atomic positions. To alleviate its evaluation cost, it would be interesting to see whether it is possible to recast it as an explicit scalar product in a given
feature space. This would allow e.g., to transfer most
of the computational burden to the pre-computation of
the corresponding basis functions. Fortunately such complexity can be largely avoided altogether if equally accurate kernels can be built by physical intuition at least for
the most relevant lowest _n_ orders, as discussed in the
next section.


**D.** **Building** _**n**_ **-body kernels II:** _**n**_ **-body feature**
**spaces and uniqueness issues**


The practical effect of the Haar integration (10) is
the elimination of the three spurious rotational degrees
of freedom. The same result can always be achieved
by selecting a group of symmetry- invariant degrees
of freedom for the system, typically including the distances and/or bond angles found in local atomic environments, or simple functions of these. Appropriate symmetrized kernels can then simply be obtained by defining
a similarity measure _directly_ on these invariant quantities

[15, 26, 27, 45]. To construct symmetry invariant _n_ -body
kernels with _n_ = 2 and _n_ = 3 we can choose these degrees
of freedom to be just interparticle distances:



_k_ _n_ _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] �



_k_ ˜ _n_ ( **q** _i_ 1 _,...,i_ _n−_ 1 _,_ **q** _[′]_ _j_ 1 _,...,j_ _n−_ 1 [)] _[,]_ (21)



_k_ 2 _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] �

_i∈ρ_
_j∈ρ_ _[′]_



_k_ ˜ 2 ( _r_ _i_ _, r_ _j_ ) (19)



_i_ 1 _,...,i_ _n−_ 1 _∈ρ_
_j_ 1 _,...,j_ _n−_ 1 _∈ρ_ _[′]_


where the components of the feature vectors **q** are the
chosen symmetry-invariant degrees of freedom describing
the _n_ -plet of atoms.
The **q** feature vectors are required to be (3 _n −_ 6) dimensional for all _n_, except for _n_ = 2, where they become
scalars. In practice, for _n >_ 3 selecting a suitable set of
invariant degrees of freedom is not trivial. For instance,
for _n_ = 4 the set of six unordered distances between
four particles does not specify their relative positions unambiguously, while for _n >_ 4 the number of distances
associated with _n_ atoms exceeds the target feature space



_k_ 3 _[s]_ [(] _[ρ, ρ]_ _[′]_ [) =] �



_k_ ˜ 3 (( _r_ _i_ 1 _, r_ _i_ 2 _, r_ _i_ 1 _i_ 2 ) _,_ ( _r_ _j_ _[′]_ 1 _[, r]_ _j_ _[′]_ 2 _[, r]_ _j_ _[′]_ 1 _j_ 2 [))]



_i_ 1 _,i_ 2 _∈ρ_
_j_ 1 _,j_ 2 _∈ρ_ _[′]_

(20)


where the _k_ [˜] are kernel functions that directly specify the
correlation of distances, or triplets of distances, found


Unique Non Unique


Figure 4. Unique interaction (left panel) associated with the
3-body kernel _k_ 3 _[s]_ [(20) compared with the non-unique 3-body]
interaction (right panel) associated with the kernel _k_ 3 _[¬][u]_ =
( _k_ 2 _[s]_ [)] [2] [ (22), which is a function of two distances only (see text).]


dimension 3 _n −_ 6. Meanwhile, the computational cost of
evaluating the full sum in Eq. (21) very quickly becomes
prohibitively large as the number of elements in the sum
grows exponentially with _n_ .
The order of an _already symmetric n_ -body kernel can
however be augmented with no computational overhead
by generating a derived kernel through simple exponentiation to an integer power, at the cost of losing the _unique-_
_ness_ [28, 46, 47] of the representation. This can be easily
understood by means of an example (graphically illustrated in Figure 4). Let us consider the 2-body symmetric kernel _k_ 2 _[s]_ [(Eq. (19)) which learns a function of just]
a single distance, and therefore treats the _r_ _i_ distances
between the central atom and its neighbors independently. Its square is the kernel


|kernel|order|symm.|unique|name|
|---|---|---|---|---|
|_ks_<br>2|2|||2-body|
|_k¬u_<br>3|3|<br>|<br>_×_|3-body, non-unique|
|_ks_<br>3|3|<br>||3-body|
|_k¬u_<br>5|5|<br>|<br>_×_|5-body, non-unique|
|_kds_<br>_MB_|_∞_|<br>_∼_||many-body, approx. symmetric|



_k_ 3 _[¬][u]_ [(] _[ρ, ρ]_ _[′]_ [) =] �

_i_ 1 _,i_ 2 _∈ρ_
_j_ 1 _,j_ 2 _∈ρ_ _[′]_



_k_ ˜ 2 ( _r_ _i_ 1 _, r_ _j_ _[′]_ 1 [)˜] _[k]_ [2] [(] _[r]_ _[i]_ 2 _[, r]_ _j_ _[′]_ 2 [)] (22)



7


Table I. Some of the kernels presented and their properties.


simple calculation reveals that, also in the general case,
the number of variables on which _k_ _n_ _[¬][u]_ is implicitly built
is (3 _n_ _[′]_ _−_ 6) _ζ_, always smaller than the full dimension of
_n_ -body feature space (3 _n_ _[′]_ _−_ 3) _ζ −_ 3 (as expected, the two
become equal only for the trivial exponent _ζ_ = 1).
None of the kernels obtained as finite powers of some
symmetric lower-order kernels is a many-body one (they
will all satisfy Eq. (2) for some finite _n_ ). However, an attractive immediate generalization consists of substituting
any squaring or cubing with full exponentiation. For instance, exponentiating a symmetrized 3-body kernel we
obtain the many-body kernel _k_ _MB_ = exp[ _k_ 3 _[s]_ [(] _[ρ, ρ]_ _[′]_ [)]][. It]
is clear from the infinite expansion in Eq. (7) that this
kernel is a many-body one in the sense of Eq. (2), and is
also fully symmetric [1] . As is also the case for all finitepower kernels, the computational cost of this many body
kernel will depend on the order _n_ _[′]_ of the input kernel (3
in the present example) as the sum in Eq. (21) only runs
on the atomic _n_ _[′]_ -plets (here, triplets) in _ρ_ and _ρ_ _[′]_ . This
new kernel is not a priori known to neglect any order
of interaction that might occur in a physical system and
thus be encoded in a reference QM training database.
To summarise, we provided a definition for an _n_ -body
kernel, and proposed a general formalism for building _n_ body kernels by exact Haar integration over rotations.
We then defined a class of simpler kernels based on rotation invariant features that are also _n_ -body according to
the previous definition. As both approaches become computationally expensive for high values of _n_, we pointed
out that _n_ -body kernels can be built as powers of lowerorder input _n_ _[′]_ -body kernels, with no additional computational overhead. While such a procedure in our case
comes at the cost of sacrificing the unicity property of the
descriptor, it also suggests how to build, by full exponentiation, a many-body symmetric kernel. For many applications, however, using a finite-order kernel will provide
the best option.


1 One could also inexpensively obtain a many body kernel by
normalisation of an explicit finite order one, for instance, as
_k_ _MB_ = _k_ 3 _[s]_ [(] _[ρ, ρ]_ _[′]_ [)] _[/]_ ~~�~~ _k_ 3 ~~_[s]_~~ [(] _[ρ, ρ]_ [)] _[k]_ 3 ~~_[s]_~~ [(] _[ρ]_ ~~_[′]_~~ _[, ρ]_ ~~_[′]_~~ [)][.] The denominator makes

this many-body in the sense of Eq. (2) (as is also the case for
the SOAP kernel, see discussion in Section II C, while no Haar
integration is needed here).



which will be able to learn functions of two distances
_r_ _i_ 1 _, r_ _i_ 2 from the central atom of the target configuration _ρ_ (see Figure 4) and thus will be a 3-body kernel
in the sense of Eq. (2). However, this kernel cannot
resolve angular information, as rotating the atoms in _ρ_
around the origin by independent, arbitrary angles will
yield identical predictions.
Extending this line of reasoning, it is easy to show that
squaring a symmetric 3-body kernel yields a kernel that
can capture interactions up to 5-body, although again
non-uniquely. This has often been done in practice by
squaring the SOAP integral [26, 42]. In general, raising a
3-body “input” kernel to an arbitrary integer power _ζ ≥_ 2
yields an _n_ -body output kernel of order 2 _ζ_ +1, _k_ _n_ _[¬]_ =2 _[u]_ _ζ_ +1 [=]
_k_ 3 _[s]_ [(] _[ρ, ρ]_ _[′]_ [)] _[ζ]_ [. This kernel is also non-unique as it will learn]
a function of only 3 _ζ_ variables, while the total number of
relevant _n_ -body degrees of freedom (3 _n −_ 6 = 6 _ζ −_ 3) is
always larger than this. Substituting 3 with any _n_ _[′]_ order
of the symmetrized input kernel will similarly generate a
_k_ _n_ _[¬][u]_ = _k_ _n_ _[s]_ _[′]_ [(] _[ρ, ρ]_ _[′]_ [)] _[ζ]_ [ kernel of order] _[ n]_ [ = (] _[n]_ _[′]_ _[ −]_ [1)] _[ζ]_ [ + 1][. A]


8



0.4


0.3


0.2


0.1


0









0.4


0.3


0.2


0.1


0





10 100 1000

Training points


(a)



10 100 1000

Training points


(b)



1.5


1


0.5


0



1.5


1


0.5


0













10 100 1000

Training points


(c)



10 100 1000

Training points


(d)



Figure 5. Learning curves reporting the mean generalization error (measured as the modulus of the difference between target
and predicted force vectors) as a function of the training set size, for different materials and kernels of increasing order. The
insets in (a) and (d) report the converged error achieved by a given kernel as a function of the kernel’s order. The systems
considered are: (a) Crystalline nickel, 500 K (compared to a nickel nanocluster in the inset); (b) iron with a vacancy, 500 K; (c)
diamond and graphite, mixed temperatures and pressures; and (d) amorphous silicon, 650 K (compared to crystalline silicon in
the inset). For extra details on the datasets and kernels used, and on the experimental methodology, see Appendixes C and D.



**E.** **Optimal** _**n**_ **-kernel choice**


In general, choosing a higher order _n_ -body kernel will
improve accuracy at the expense of speed. The optimal
kernel choice for a given application will correspond to
the best tradeoff between computational cost and representation power, which will depend on the physical system investigated. The properties of some of the kernels
discussed above are summarized in Table I, while their
performance is tested on a range of materials in Figure 5.

The figure reveals some general trends. 2-body kernels
can be trained very quickly, as good convergence can be
attained already with _∼_ 100 training configurations. The
2-body representation is a very good descriptor for a few
materials under specific conditions, while their overall
accuracy is ultimately limited. This will yield e.g., excellent force accuracy for a close-packed bulk system like



crystalline Nickel (inset (a)), and reasonable accuracy for
a defected _α_ -Fe system whose bcc structure is however
metastable if just pair potentials are used (inset (b)). Accuracy improves dramatically once angular information
is acquired by training 3-body kernels. These can accurately describe forces acting on iron atoms in the bulk
_α_ -Fe system containing a vacancy (inset (b)) and those
acting on carbon atoms in both diamond and graphite
(inset (c)). However, 3-body GPs need larger training
databases. Also, atoms participate in many more triplets
than simple bonds in their standard environments contained in the database, which will make 3-body kernels
slower than 2-body ones for making predictions by GP
regression. Both problems would extend, getting worse,
to higher values of _n_, as summing over all database configurations and all feature _n_ -plets in each database configuration will make GP predictions progressively slower.


9



However, complex materials where high-order interactions presumably play a significant role should be expected to be well described by ML-FF based on a many-body
kernel. This is verified here in the case of amorphous Silicon (inset (d)).


Identifying the _n_ value best suited for the description
of a given material system can also be done in practice by
monitoring how the converged error varies as a function
of the kernel order. Plots illustrating this behaviour are
provided in insets (a) and (d) for nickel and silicon systems, respectively. In each plot the more complex system
(a Ni cluster and an amorphous Si system, respectively)
display a high accuracy gain (larger negative slope) when
the kernel order is increased, while the relatively simpler
cristalline Ni and Si systems show a practically constant
trend on the same scale.


Figure 5 (b) also shows the performance of some nonunique kernels. As discussed above, these are options to
increase the order of an input kernel avoiding the need to
sum over the correspondingly higher order _n_ -plets. Our
tests indicate that the ML-FFs generated by non-unique
kernels sometimes improve appreciably on the input kernels’ performance: e.g., the error incurred by the 2-body
kernel of Eq. (19) in the Fe-vacancy system is higher than
that associated with its square, the non-unique 3-body
kernel of Eq. (22). Unfortunately, but not surprisingly,
the improvement can be in other cases modest or nearly
absent, as exemplified by comparing the errors associated
with the 3-body kernel and its square -the non-unique 5body kernel-, in the same system.


Overall, the analysis of Figure 5 suggests that an optimal kernel can be chosen by comparing the learning
curves of the various _n_ -body kernels and the many-body
kernel over the available QM database: the comparison will reveal the simplest (most informative, lowest
_n_ ) description that is still compatible with the error level
deemed acceptable in the simulation.


Trading transferability for accuracy by training the
kernels on a QM database appropriately tailored for the
target system (e.g., restricted to just bulk or simplydefected system configurations sampled at the relevant
temperatures as done in the Ni and Fe-systems of Figure 5) will enable surprisingly good accuracy even for
low _n_ values. This should be expected to systematically
improve on the accuracy performance of classical potentials involving non-linear parameter fitting, as exemplified by comparing the errors associated with _n_ -body kernel models and the average errors of state-of-the-art embedded atom model (EAM) P-FFs [7, 48] (insets (a) and
(b)). The next section further explores the performance
of GP-based force prediction, to address the final issue
of what execution speed can be expected for ML-based
force fields, once the optimally accurate choice of kernel
has been made.



0

0 1 2 3 4 5 6 7 8 9 10
Error on force [meV/Å]


Figure 6. nser grids. Inset: standard deviation of the distributions as a function of the number of interpolation grid
points, on a log-log scale (each distribution in the main panel
corresponds to a dot of same color in the insert).


**III.** **MAPPED FORCE FIELDS (M-FFS)**


Once a GP kernel is recognized as being _n_ -body, it
automatically defines an _n_ -body force field corresponding
to it, for any given choice of training set. This will be an
_n_ -body function of atomic positions satisfying Eq. (2),
whose values _can_ be computed by GP regression sums
over the training set as done by standard ML-FF implementations, but _do not have to_ be computed this way.
In particular, the execution speed of a machine learningderived _n_ -body force field might be expected to depend
on its order _n_ (e.g., it will involve sums over all atomic
triplets, like any 3-body P-FF, if _n_ =3), but should otherwise be independent of the training set size. It should
therefore be possible to construct a mapping procedure
yielding a machine learning-derived, nonparametric force
field (an efficient “M-FF”) that allows a very significant
speed-up over calculating forces by direct GP regression.
We note that non-unique kernels obtained as powers of
_n_ _[′]_ -body input kernels exceed their reference _n_ _[′]_ -body feature space and thus could not be similarly sped up by
mapping their predictions onto an M-FF of equal order
_n_ _[′]_, while mapping onto an M-FF of the higher output
order _n_ would still be feasible.


For convenience, we will analyze a 3-body kernel case,
show that a 3-body GP exactly corresponds to a classical 3-body M-FF, and show how the mapping yielding
the M-FF can be carried out in this case, using a 3Dspline approximator. The generalization to any order _n_
is straightforward provided that a good approximator can
be identified and implemented. We begin by inserting the
general form of a 3-body kernel (Eq. (21)) into the GP
prediction expression (Eq. (1)), to obtain



1


0.8


0.6


0.4


0.2






10



GP 3-body Mapped 3-body



Ratio



0.25


0.00


-0.25


0.20


0.10


0.00



10 [4]

10 [3]

10 [2]

10 [1]

10 [0]

10 [−1]

10 [−2]

10 [−3]



M = 24 _N_ = 500





2.0 2.5 3.0 3.5 4.0 4.5
Distance [ _Å_ ]



1 10 100 1000

_N_



10

_M_



20 30



Figure 7. Computational cost of evaluating the 3-body energy (Eq. (24)) as a function of the database size _N_ and the
number of atoms _M_ located within the cutoff radius. Left:
time taken (s) for a single energy prediction using the GP
(blue dots and solid line) and the mapped potential (orange
dots and solid line), as a function of _N_, for _M_ =24. The speedup ratio is also provided as a dotted line. Right: scaling of
the same quantities as a function of _M_ for a training set of
_N_ = 500 configurations.



�

_i_ 1 _,i_ 2 _∈ρ_
_j_ 1 _,j_ 2 _∈ρ_ _d_



_ε_ ( _ρ_ ) =



_N_
�


_d_ =1



_k_ ˜ 3 ( **q** _i_ 1 _,i_ 2 _,_ **q** _[d]_ _j_ 1 _,j_ 2 [)] _[α]_ _[d]_ _[.]_ (23)



Inverting the order of the sums over the database and
atoms in the target configurations yields a general expression for the 3-body potential:



_ε_ ( _ρ_ ) = � _ε_ ˜( **q** _i_ 1 _,i_ 2 ) (24)

_i_ 1 _,i_ 2 _∈ρ_



�

_j_ 1 _,j_ 2 _∈ρ_ _d_



80 90 100 110 120 130 140
Angle [deg]


Figure 8. Energy profiles of the 3-body M-FF, trained for the
a-Si system at 650K. Upper panel: 2-body interaction term.
Lower panel: 3-body interaction energy for an atomic triplet,
angular dependence when the two distances from the central
atoms are both equal to 2.4Å.


distributed grid of points within its domain. This procedure effectively maps the GP predictions on the relevant
3-body feature space: once completed, the value of the
triplet energy at any new target point can be calculated
via a local interpolation, using just a subset of nearest
tabulated grid points. If the number of grid points _N_ _g_
is made sufficiently high, the mapped function will be
essentially identical to the original one but, by virtue of
the locality of the interpolation, the cost of evaluating it
will not depend on _N_ _g_ .
The 3-body configuration energy of Eq. (24) also includes 2-body contributions coming from the terms in
the sum for which the indices _i_ 1 and _i_ 2 are equal. When
_i_ 1 = _i_ 2 = _i_ the term _ε_ ( **q** _i,i_ ) can be interpreted as the pairwise energy associated with the central atom and atom _i_ .
The term can consequently be mapped onto a 1D 2-body
feature space whose coordinate is the single independent component of the **q** _i,i_ feature vector, typically the
distance between atom _i_ and the central atom. In the
same way, an _n_ -body kernel naturally defines a set of _n_ body energy terms of order comprised between 2 and _n_,
depending on the number of repeated indices.
Figure 6 shows the convergence of the mapped forces
derived from the 3-body kernel in Eq. (20) for a database of DFTB atomic forces for the a-Si system. The
interpolation is carried out using a 3D cubic spline for
different 3D mesh sizes. Comparison with the reference
forces produced by the GP allows to generate, for each
mesh size, the distribution of the absolute deviation of
the force components from their GP-predicted values.
The standard deviation of the interpolation error distribution is shown in the insert on a log-log scale, as
a function of _N_ _g_ . Depending on the specific reference
implementation, the speed-up in calculating the local energy (Eq. (24)) provided by the mapping procedure can
vary widely, while it will always grow linearly with _N_
and quadratically with _M_ (see Figure 7), and it will be
always substantial: in typical testing scenarios we found
this to be of the order of 10 [3] _−_ 10 [4] .



_ε_ ˜( **q** _i_ 1 _,i_ 2 ) =



_N_
�


_d_ =1



_k_ ˜ 3 ( **q** _i_ 1 _,i_ 2 _,_ **q** _[d]_ _j_ 1 _,j_ 2 [)] _[α]_ _[d]_ _[.]_ (25)



Eq. (24) reveals that the GP implicitly defines the local
energy of a configuration as a sum over all triplets containing the central atom, where the function ˜ _ε_ represents
the energy associated with each triplet in the physical
system. The triplet energy is calculated by three nested
sums, one over the _N_ database entries and two running
over the _M_ atoms of each database configuration ( _M_ may
slightly vary over configurations, but can be assumed to
be constant for the present purpose). The computational
cost of a single evaluation of the triplet energy (25) scales
consequently as _O_ ( _NM_ [2] ). Clearly, improving the GP
prediction accuracy by increasing _N_ and _M_ will make
the prediction slower. However, such a computational
burden can be avoided, bringing the complexity of the
triplet energy calculation (25) to _O_ (1).
Since the triplet energy ˜ _ε_ is a function of just three variables (the effective symmetry-invariant degrees of freedom associated with three particles in three dimensions),
we can calculate and store its values on an appropriately


An M-FF example, obtained for a-Si with _n_ = 3 is
shown in Figure 8. As the energy profile is not prescribed
by any particular functional form, it is free to optimally
adapt to the information contained in the QM training
set, to best reproduce the quantum interactions that produced it. Figure 8 contains some expected features e.g., a
radial minimum at about _r ≃_ 2 _._ 4Å in the 2-body section
(upper panel), the corresponding angular minimum at
_θ_ 0 _≃_ 110 _[◦]_ (lower panel), which is approximately equal to
the _sp_ [3] hybridization angle of 109.47 _[◦]_, and rapid growth
for small radii (upper panel) and angles (lower panel).
Less intuitive features are also visible, which however
contribute to the best representation of the bulk system’s
interactions that a 3-body expansion can achieve for the
given database. An example is the shallow maximum in
the 2-body section at _r ≃_ 3 _._ 1Å, which would of course
disappear if we fitted our model on QM forces calculated
for a Si dimer, that do not contain a hump. The resulting Si force field, appropriate for a Si dimer, would
however inevitably reproduce the QM bulk interactions
less accurately. More generally, training on the aggregate dataset could be a sensible compromise, producing a
more transferable, but locally less accurate force field.


**IV.** **CONCLUDING REMARKS**


The results presented in this work exemplify how physical priors built in the GP kernels restrict their descriptiveness, while improving their convergence speed as a
function of training dataset size. This provides a framework to optimise efficiency. Comparing the performance
of _n_ -body kernels allows us to identify the lowest order _n_
that is compatible with the required accuracy as a consequence of the physical properties of the target system.
As a result, accuracy can in each case be achieved using
the most efficient kernel e.g., a 2-body kernel for bulk
Ni, or a 3-body kernel for carbon and graphite, for a _∼_
0.1 eV/Å target force accuracy, see Figure 5. As should
be reasonably expected, relying on low-dimensional feature spaces will limit the maximum achievable accuracy if
higher-order interactions, missing in the representation,
occur in the target system.
On the other hand, we find that once the optimal order
_n_ has been identified and the _n_ -kernel has been trained
(whichever its form e.g., whether defined as a function
of invariant descriptors as in (21), or constructed as a
power of such a function, or derived as an analytic integral over rotations using Eqs. (12-18)), it becomes possible
to map its prediction on the appropriate _n_ -dimensional
domain, and thus generate a _n_ -body M-FF machinelearned atomic force field that predicts atomic forces at
essentially the same computational cost of a classical _n_ body P-FF parametric force field. The GP predictions
allow for a natural intrinsic measure of uncertainty -the
GP predicted variance-, and the same mapping procedure used for the former can also be applied to the latter. Thus, like their ML-FF counterparts, and unlike



11


P-FFs, M-FFs offer a tool which could be used to monitor whether any extrapolation is taking place that might
involve large prediction errors.
In general, our results suggest a possible three-step
procedure to build fast nonparametric M-FFs whenever
a series of kernels _k_ _n_ can be defined with progressively
higher complexity/descriptive power and well-defined
feature spaces with _n_ -dependent dimensionality. However the series is constructed (and whether or not it converges to a universal descriptor) this will involve (i) GP
training of _n_ -kernels for different values of _n_, in each case
using as many database entries relevant to the target system as needed to achieve convergence of the _n_ -dependent
prediction error; (ii) identification of the optimal order _n_,
yielding the simplest viable description of the system’s
interactions - this could be e.g., the minimal value of
_n_ compatible with the target accuracy or, within a GP
statistical learning framework, the result of a Bayesian
model selection procedure [22]; (iii) mapping of the resulting (GP-predicted) ML-FF onto an efficient M-FF, using a suitably fast approximator function defined over the
relevant feature space.
A major limitation of the M-FFs obtained this way is
that, similar to P-FFs, each of them can be used only
in “interpolation mode”, that is when the target configurations are all well represented in the fixed database
used. This is not the case in molecular dynamics simulations potentially revealing new chemical reaction paths,
or whenever online learning or the use of dynamicallyadjusted database subsets are necessary to avoid unvalidated extrapolations and maximise efficiency. In such
cases, “learn on the fly” (LOTF) algorithms can be deployed, which have the ability to incorporate novel QM
data into the database used for force prediction. In such
schemes, the new data are either developed at runtime
by new QM calculations, or they are adaptably retrieved
as the most relevant subset of a much larger available
QM database [15]. The availability of an array of _n_ -body
kernels is very useful for this class of algorithms, which
provides further motivation for their development. In
particular, distributing the use of _n_ -body kernels nonuniformly in both space and time along the system’s trajectory has the potential to provide an optimally efficient
approach to accurate MD simulations using the LOTF
scheme. Finally, while the complication of continuously
mapping the GP predictions to reflect a dynamically updated training database makes on the fly M-FF generation a less attractive option, a strategy to produce the
MD trajectory with classical force field efficiency might
involve using (concurrently across the system, and at any
given time) a locally optimal choice built from a comprehensive set of pre-computed low-order M-FFs.


**ACKNOWLEDGEMENTS**


The authors acknowledge funding by the Engineering and Physical Sciences Research Council (EPSRC)


through the Centre for Doctoral Training “Cross Disciplinary Approaches to Non-Equilibrium Systems”
(CANES, Grant No. EP/L015854/1) and by the Office
of Naval Research Global (ONRG Award No. N6290915-1-N079). ADV acknowledges further support by the
EPSRC HEmS Grant No. EP/L014742/1 and by the
European Union’s Horizon 2020 research and innovation
program (Grant No. 676580, The NOMAD Laboratory,
a European Centre of Excellence). We are grateful to the


[1] R. Car and M. Parrinello, Physical Review Letters **55**,
2471 (1985).

[2] P. Hohenberg and W. Kohn, Physical review **136** (1964).

[3] W. Kohn and L. J. Sham, Physical review **140**, A1133
(1965).

[4] F. H. Stillinger and T. A. Weber, Physical review **B31**,
5262 (1985).

[5] J. Tersoff, Physical Review B **37**, 6991 (1988).

[6] D. W. Brenner, physica status solidi(b) **217**, 23 (2000).

[7] Y. Mishin, Acta Materialia **52**, 1451 (2004).

[8] A. C. T. van Duin, S. Dasgupta, F. Lorant, and W. A.
Goddard, The Journal of Physical Chemistry A **105**,
9396 (2001).

[9] G. A. Cisneros, K. T. Wikfeldt, L. Ojamäe, J. Lu, Y. Xu,
H. Torabifard, A. P. Bartók, G. Csányi, V. Molinero, and
F. Paesani, Chemical Reviews **116**, 7501 (2016).

[10] S. K. Reddy, S. C. Straight, P. Bajaj, C. Huy Pham,
M. Riera, D. R. Moberg, M. A. Morales, C. Knight, A. W.
Götz, and F. Paesani, The Journal of Chemical Physics
**145**, 194504 (2016).

[11] A. J. Skinner and J. Q. Broughton, Modelling and Simulation in Materials Science and Engineering **3**, 317 (1995).

[12] J. Behler and M. Parrinello, Physical Review Letters **98**,
146401 (2007).

[13] A. P. Bartók, M. C. Payne, R. Kondor, and G. Csányi,
Physical Review Letters **104**, 136403 (2010).

[14] A. V. Shapeev, Multiscale Modeling & Simulation **14**,
1153 (2016).

[15] Z. Li, J. R. Kermode, and A. De Vita, Physical Review
Letters **114**, 096405 (2015).

[16] A. Glielmo, P. Sollich, and A. De Vita, Physical Review
B **95**, 214302 (2017).

[17] V. Botu and R. Ramprasad, Physical Review B **92**,
094306 (2015).

[[18] G. Ferré, T. Haut, and K. Barros, (2016), 1612.00193v1.](http://arxiv.org/abs/1612.00193v1)

[19] E. V. Podryabinkin and A. V. Shapeev, Computational
Materials Science **140**, 171 (2017).

[20] A. Takahashi, A. Seko, and I. Tanaka, (2017),
[1710.05677.](http://arxiv.org/abs/1710.05677)

[21] J. R. Boes, M. C. Groenenboom, J. A. Keith, and J. R.
Kitchin, International Journal of Quantum Chemistry
**116**, 979 (2016).

[22] C. K. I. Williams and C. E. Rasmussen, the MIT Press
(2006).

[23] C. M. Bishop, _Pattern Recognition and Machine Learn-_
_ing_, Information Science and Statistics (Springer, New
York, NY, 2006).

[24] M. J. Kearns and U. V. Vazirani, _An Introduction to_
_Computational Learning Theory_ (MIT Press, 1994).



12


UK Materials and Molecular Modelling Hub for computational resources, which is partially funded by EPSRC
(EP/P020194/1). We furthermore thank Gábor Csányi,
from the Engineering Department, University of Cambridge, for the Carbon database and Samuel Huberman,
from the MIT Nano-Engineering Group for the initial
geometry used in the a-Si simulations. Finally, we want
to thank Ádám Fekete for stimulating discussions and
precious technical help.


[25] M. Rupp, R. Ramakrishnan, and O. A. von Lilienfeld, The Journal of Physical Chemistry Letters **6**, 3309
(2015).

[26] V. L. Deringer and G. Csányi, Physical Review B **95**,
094203 (2017).

[27] F. A. Faber, A. S. Christensen, B. Huang, and O. A. von
Lilienfeld, The Journal of Chemical Physics **148**, 241717
(2018).

[28] A. P. Bartók, R. Kondor, and G. Csányi, Physical Review B **87**, 184115 (2013).

[29] T. Bereau, R. A. DiStasio Jr, A. Tkatchenko, and O. A.
[von Lilienfeld, arXiv.org (2017), 1710.05871v1.](http://arxiv.org/abs/1710.05871v1)

[30] A. Grisafi, D. M. Wilkins, G. Csányi, and M. Ceriotti,
[arXiv.org (2017), 1709.06757v1.](http://arxiv.org/abs/1709.06757v1)

[31] W. J. Szlachta, A. P. Bartók, and G. Csányi, Physical
Review B **90**, 104108 (2014).

[[32] H. Huo and M. Rupp, arXiv.org (2017), 1704.06439v1.](http://arxiv.org/abs/1704.06439v1)

[33] I. Macêdo and R. Castro, _Learning divergence-free and_
_curl-free vector fields with matrix-valued kernels_ (Instituto Nacional de Matematica Pura e Aplicada, 2008).

[34] V. Botu and R. Ramprasad, International Journal of
Quantum Chemistry **115**, 1075 (2015).

[35] G. Ferré, J. B. Maillet, and G. Stoltz, The Journal of
Chemical Physics **143**, 104114 (2015).

[36] L. Nachbin, _The Haar integral_ (Van Nostrand, Princeton,
1965).

[37] H. Schulz-Mirbach, in _Proceedings of the 12th IAPR In-_
_ternational Conference on Pattern Recognition, Vol. 3 -_
_Conference C: Signal Processing (Cat. No.94CH3440-5_
(IEEE, 1994) pp. 387–390 vol.2.

[38] B. Haasdonk and H. Burkhardt, Machine Learning **68**,
35 (2007).

[39] K. Hornik, Neural networks **6**, 1069 (1993).

[40] A. P. Thompson, L. P. Swiler, C. R. Trott, S. M. Foiles,
and G. J. Tucker, Journal of Computational Physics **285**,
316 (2015).

[41] S. De, A. P. Bartók, G. Csányi, and M. Ceriotti, Physical
Chemistry Chemical Physics **18**, 13754 (2016).

[42] P. Rowe, G. Csányi, D. Alfè, and A. Michaelides,
[arXiv.org (2017), 1710.04187v2.](http://arxiv.org/abs/1710.04187v2)

[43] T. W. Anderson, The Annals of Mathematical Statistics
**17**, 409 (1946).

[44] A. T. James, Proc. R. Soc. Lond. A **229**, 367 (1955).

[45] M. Rupp, A. Tkatchenko, K. R. Müller, and O. A. von
Lilienfeld, Physical Review Letters **108**, 058301 (2012).

[46] B. Huang and O. A. von Lilienfeld, The Journal of Chemical Physics **145**, 161102 (2016).

[47] O. A. von Lilienfeld, R. Ramakrishnan, M. Rupp, and
A. Knoll, International Journal of Quantum Chemistry
**115**, 1084 (2015).


[48] M. I. Mendelev, S. Han, D. J. Srolovitz, G. J. Ackland,
D. Y. Sun, and M. Asta, Philosophical Magazine **83**,
3977 (2003).

[49] J. P. Perdew, K. Burke, and M. Ernzerhof, Physical
Review Letters **77**, 3865 (1996).

[50] C. Zeni, K. Rossi, A. Glielmo, N. Gaston, F. Baletto,
[and A. De Vita, arXiv.org (2018), 1802.01417v1.](http://arxiv.org/abs/1802.01417v1)


**APPENDIX**


**A.** **Kernel order by explicit differentiation**


We first prove that the kernel given in Eq. (4) is 2-body
in the sense of Eq. (2). For this it is sufficient to show that
its second derivative with respect to the relative position
of two different atoms of the target configuration _ρ_ always
vanishes. The first derivative is



_∂k_ 2 ( _ρ,_ _ρ_ _[′]_ )



2


=
_∂_ **r** _i_ 1 �



_∂_
e _[−∥]_ **[r]** _[i]_ _[−]_ **[r]** _j_ _[′]_ _[∥]_ [2] _[/]_ [4] _[σ]_ [2]
_∂_ **r** _i_ 1



_ij_



= � e _[−∥]_ **[r]** _[i]_ _[−]_ **[r]** _j_ _[′]_ _[∥]_ [2] _[/]_ [4] _[σ]_ [2] [ (] **[r]** _[i]_ 2 _[ −]_ _σ_ [2] **[r]** _j_ _[′]_ [)] _δ_ _ii_ 1

_ij_

= � e _[−∥]_ **[r]** _[i]_ [1] _[−]_ **[r]** _j_ _[′]_ _[∥]_ [2] _[/]_ [4] _[σ]_ [2] [ (] **[r]** _[i]_ [1] 2 _[ −]_ _σ_ [2] **[r]** _j_ _[′]_ [)] _._

_j_



This depends only on the atom located at **r** _i_ 1 of the configuration _ρ_ . Thus, differentiating with respect to the
relative position **r** _i_ 2 of any other atom of the configuration gives the relation in Eq. (2) for 2-body kernels:


_∂_ [2] _k_ 2 ( _ρ,_ _ρ_ _[′]_ )

= 0 _._
_∂_ **r** _i_ 1 _∂_ **r** _i_ 2


We next show that the kernel defined in Eq. (6) is an _n_  body in the sense of Eq. (2). This follows naturally from
the result above, given that _k_ _n_ is defined as _k_ _n_ = _k_ 2 _[n][−]_ [1] .
We can thus write down its first derivative as


_∂k_ _n_ _∂k_ 2
= ( _n −_ 1) _k_ 2 _[n][−]_ [2] _._
_∂_ **r** _i_ 1 _∂_ **r** _i_ 1


Since the second derivative of _k_ 2 is null, the second derivative of _k_ _n_ is simply



_∂_ [2] _k_ _n_ _∂k_ 2
= ( _n −_ 2)( _n −_ 1) _k_ 2 _[n][−]_ [3]
_∂_ **r** _i_ 1 _∂_ **r** _i_ 2 _∂_ **r** _i_ 1



_∂k_ 2

_∂_ **r** _i_ 2



and after _n −_ 1 derivations we similarly obtain


_∂_ [2] _k_ 2 _[n][−]_ [1] _∂k_ 2 _∂k_ 2
= ( _n −_ 1)! _k_ 2 [0] _. . ._ _._
_∂_ **r** _i_ 1 _· · · ∂_ **r** _i_ _n_ _∂_ **r** _i_ 1 _∂_ **r** _i_ _n−_ 1


Since _k_ 2 [0] [= 1][, the final derivative with respect to the] _[ n]_ _[th]_
particle position **r** _i_ _n_ is zero as required by Eq. (2).



13


**B.** **1D** _**n**_ _**[′]**_ **-body model**


To test the ideas behind the _n_ -body kernels, we used a
1D _n_ _[′]_ -particle model reference system where a (“central”)
particle is kept fixed at the coordinate axis origin (consistent with the local configuration convention of Eq. (3)).
The energy of the central particle is defined as


_f_ = � _J x_ _i_ 1 _. . . x_ _i_ _n′−_ 1

_i_ 1 _...i_ _n′−_ 1


where _{x_ _i_ _p_ _}_ _p_ _[n]_ =1 _[′]_ _[−]_ [1] are the relative positions of _n_ _[′]_ _−_ 1
particles, and _J_ is an interaction constant.


To generate Figure 1 a large set of configurations was
generated by uniformly and independently sampling each
relative position _x_ _i_ _p_ within the range ( _−_ 0 _._ 5 _,_ 0 _._ 5). The
energy of the central particle of each configuration was
then given by the above equation, with the interaction
constant _J_ set to 0 _._ 5. In order to analyse the converged
properties of the _n_ -body kernels presented, large training
sets ( _N_ = 1000) were used.


**C.** **Databases details**


The bulk Ni and Fe databases were obtained from simulations using a 4 _×_ 4 _×_ 4 periodically repeated unit cell,
modelling the electronic exchange and correlation interactions via the PBE/GGA approximation [49], and controlling the temperature (set at 500K) by means of a
weakly-coupled Langevin thermostat (the DFT trajectories are available from the KCL research data management
[system at the link http://doi.org/10.18742/RDM01-92).](http://doi.org/10.18742/RDM01-92)
The C database comprises bulk diamond and AB- and
ABC-stacked graphene layer structures. These structures were obtained from DFT simulations at varying
temperatures and pressures, using a fixed 3 _×_ 3 _×_ 2 periodic cell geometry for graphite, and simulation cells
ranging from 1 _×_ 1 _×_ 1 to 2 _×_ 2 _×_ 2 unitary cells for diamond, the relative DFT trajectories can be found in
the “libAtoms” data repository via the following link
[http://www.libatoms.org/Home/DataRepository. Crys-](http://www.libatoms.org/Home/DataRepository)
talline and amorphous Si database was obtained from a
microcanonical DFTB 64-atom simulation carried out in
periodic boundaries, with average kinetic energy corresponding to a temperature of _T_ = 650 _K_ . The results
presented for the Ni cluster are reported from Ref. [50]
and correspond to constant temperature DFT MD runs
( _T_ = 300 _K_ ) of a particular Ni 19 geometry (named
“4HCP” in the article). The radial cutoffs used to create
the local environments for the four elements considered
are: 4.0 Å (Ni), 4.45 Å (Fe), 3.7 Å (C) and 4.5 Å (Si).


**D.** **Details on the kernels used and on the**

**experimental methodology**


All energy kernels presented in the work can be used
to learn/predict forces after generating a standard derivative kernel (Ref. [22], section 9.4, also cf. Section II A
of the main text.) In particular, for each scalar energy
kernel _k_ a matrix-valued force kernel **K** can be readily
obtained by double differentiation with respect to the
positions ( **r** 0 and **r** _[′]_ 0 [) of the central atoms in the target]
and database configurations _ρ_ and _ρ_ _[′]_ :


_[ρ]_ _[′]_ [)]
**K** ( _ρ, ρ_ _[′]_ ) = _[∂]_ [2] _[k]_ [(] _[ρ,]_ _._

_∂_ **r** 0 _∂_ **r** _[′]_ 0 [T]


The kernels _k_ [˜] 2 and _k_ [˜] 3 (Eqs. (19,20)) were chosen as
simple squared exponentials in the tests shown. Noting
as **q** (or _q_ ) the vector (or scalar) containing the effective
degrees of freedom of the atomic _n_ -plet considered (see
Eq. (21)), the two kernels read:


˜
_k_ 2 ( _q_ _i_ _, q_ _j_ _[′]_ [) = e] _[−]_ [(] _[q]_ _[i]_ _[−][q]_ _j_ _[′]_ [)] [2] _[/]_ [2] _[σ]_ [2]


˜
_k_ 3 ( **q** _i_ 1 _,i_ 2 _,_ **q** _[′]_ _j_ 1 _,j_ 2 [) =] � e _[−∥]_ **[q]** _[i]_ [1] _[,i]_ [2] _[−]_ **[Pq]** _j_ _[′]_ 1 _,j_ 2 _[∥]_ [2] _[/]_ [2] _[σ]_ [2] _,_

**P** _∈P_ _c_



14


where _P_ _c_ (| _P_ _c_ _|_ = 3) is the set of cyclic permutations of
three elements. Summing over the permutation group is
needed to guarantee permutation symmetry of the energy. As discussed in the main text, the exact form of
these low-order kernels is not essential as the large database limit is quickly reached.


The many-body force kernel referred to in Fig. 5 was
built as a covariant discrete summation of the many-body
energy base-kernel (7) over the _O_ 48 crystallographic point
group, using the procedure of Ref. [16]. This procedure
yields an approximation to the full covariant integral of
the many-body kernel (7) given in Eq. (5).


In order to obtain Figure 5, repeated (randomised)
realisations of the same learning curves were performed.
The points (and error bars) plotted are the means (and
standard deviations) of the generated data . The kernel
hyperparameters were independently optimised by cross
validation for each dataset.


