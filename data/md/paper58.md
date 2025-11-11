## Discovery of Algebraic Reynolds-Stress Models Using Sparse Symbolic Regression.

Martin Schmelzer Richard P. Dwight Paola Cinnella


This article is published (open-access). Please, use the original
article instead of this one. For citation:


**Schmelzer, M., Dwight, R.P. & Cinnella, P.**
**Discovery of Algebraic Reynolds-Stress Models Using**
**Sparse Symbolic Regression.**
**Flow Turbulence Combustion 104, 579603 (2020).**
```
https://doi.org/10.1007/s10494-019-00089-x

```

**Abstract**


A novel deterministic symbolic regression method SpaRTA is introduced to infer algebraic stress models for the closure of RANS equations
directly from high-fidelity LES or DNS data. The models are written
as tensor polynomials and are built from a library of candidate functions. The machine-learning method is based on elastic net regularisation which promotes sparsity of the inferred models. By being datadriven the method relaxes assumptions commonly made in the process of
model development. Model-discovery and cross-validation is performed for
three cases of separating flows, i.e. periodic hills ( _Re_ =10595), convergingdiverging channel ( _Re_ =12600) and curved backward-facing step ( _Re_ =13700).
The predictions of the discovered models are significantly improved over
the _k_    - _ω_ SST also for a _true_ prediction of the flow over periodic hills at
_Re_ =37000. This study shows a systematic assessment of SpaRTA for
rapid machine-learning of robust corrections for standard RANS turbulence models.

Keywords: Turbulence Modelling, Machine Learning, Sparse Regression, Symbolic Regression

### **1 Introduction**


The capability of Computational Fluid Dynamics (CFD) to deliver reliable prediction is limited by the unsolved closure problem of turbulence modelling.


1


The workhorse for turbulence modelling in industry are the Reynolds-Averaged
Navier-Stokes (RANS) equations using linear eddy viscosity models (LEVM)

[1]. The lower computational costs compared to high-fidelity approaches, e.g.
Large-Eddy (LES) or Direct Numerical Simulations (DNS), come at the price
of uncertainty especially for flows with separation, adverse pressure gradients or
high streamline curvature. Data-driven methods for turbulence modelling based
on supervised machine learning have been introduced to leverage RANS for improved predictions [2, 3, 4]. In [5], the source terms of the Spalart-Allmaras were
learnt from data using a single hidden layer neural network, which served as a
first feasibility study. In [6], a factor was introduced to correct the turbulent
production in the _k_ -equation of the _k_ - _ω_ model. This term was found via inverse
modelling and served to train a Gaussian process. While this approach has been
extended and applied to industrially relevant flows such as airfoils in [7, 8] it still
relies on the Boussinesq assumption. In [9], a deep neural network was trained
to predict _a_ _ij_ given input only from a baseline linear eddy viscosity simulation
and thus replacing the turbulence model instead of augmenting it. The network
was designed to embed Galilean invariance of the predicted _a_ _ij_ . This concept
of physics-informed machine learning was extended, e.g., in [10] using random
forest regression. Despite the success of the data-driven approaches a drawback
is their black box nature, which hampers the understanding of the physics of
the resulting models in order to derive new modelling ideas from it.
Recently, a method has been introduced using genetic-programming (GEP)
based symbolic regression to derive Explicit Algebraic Reynolds-stress Models
(EARSM) directly from high-fidelity data [11, 12]. EARSM, first introduced by

[13] and further developed by [14], are nonlinear extensions of LEVM and are
commonly derived by projecting Reynolds-stress models (RSM) onto a set of
tensorial polynomials [15, 16]. These models are numerically more robust than
RSM at similar computational costs as LEVM [17], but do not show superior
predictive capabilities for all kinds of flows [15]. The data-driven GEP method
retains the input quantities used to derive EARSM, but replaces the commonly
used projection method to find the formal structure of the model by an evolutionary process, which makes it an open-box machine learning approach. The
advantage of such a data-driven method is that instead of relying on assumptions made during the development of an EARSM, a model is inferred directly
from data. While such a model might not provide an universal approach for all
kinds of flows as commonly aimed for in physical modelling, it serves as a pragmatic tool to correct the flow at hand. For cases exhibiting similar flow physics,
e.g. separation, it has also been shown that the discovered models provide suitable corrections indicating the predictive potential of a data-driven approach.
Due to the non-deterministic nature of GEP it discovers for each run another model with a different mathematical form, e.g. other terms and/or other
values for coefficients, with varying complexity. It is reported that the models using only a few nonlinear terms show a low training and prediction error as well as high numerical robustness for industrially relevant flow cases

[18, 19]. Therefore, we instead introduce a new deterministic symbolic regression method SpaRTA (Sparse Regression of Turbulent Stress Anisotropy), for


2


Figure 1: Technical flow diagram of SpaRTA (Sparse Regression of Turbulent
Stress Anisotropy).


3


which we constrain the search towards sparse algebraic models using sparsitypromoting regression techniques [20, 21]. SpaRTA combines functions from a
predefined library of candidates without any random recombination. It consists
of four steps: (i) building a library of candidate functions, (ii) model selection
using sparse-regression techniques, (iii) inference of model coefficients and (iv)
cross-validation of the resulting models, see Figure 1. The first three steps are
computationally very cheap also for high-dimensional problems and allow for
rapid model discovery.
The present study provides several novel concepts for data-driven modelling,
which are organised as follows. In Section 2 we define additive model-form error terms within the _k_ - _ω_ SST LEVM model and use _k_ -corrective-frozen-RANS,
which is an extension of the method introduced in [12], to compute the modelform error from high-fidelity data. The novelty in this work is that we identify
not only a correction of the stress-strain relation, but also one for the turbulent
transport equations and thereby achieve excellent agreement with mean-fields
of high-fidelity data. We also validate that the model-form error is successfully
captured by adding the two terms to the solver and performing a CFD simulation. The _k_ -corrective-frozen-RANS does not require any iterative optimisation
procedure as compared to [6] and is therefore very efficient, but also limited to
full-field data. In Section 3 we introduce the steps of SpaRTA. The details of
the test cases, the CFD setup and the sources of the high-fidelity data are given
in Section 4. In Section 5 SpaRTA is applied to the test cases, the discovered
models are presented and the best models are chosen using cross-validation.
Finally, conclusions are drawn in Section 6.

### **2 Model-form error of RANS equations**


In the following, we augment the baseline model, i.e. the linear eddy viscosity assumption and the turbulence transport equations of the _k_ - _ω_ SST, with
additive terms accounting for the error due to the model-form. We introduce _k_ corrective-frozen-RANS, which is an extension of the method in [12], to extract
these two types of error from high-fidelity data sources efficiently. Finally, we
validate that the extracted terms reduce the error for given test cases.


**2.1** **Identification of additive model-form error from data**


The incompressible and constant-density RANS equations read


_∂_ _i_ _U_ _i_ = 0 _,_



_,_ (1)
�



_U_ _j_ _∂_ _j_ _U_ _i_ = _∂_ _j_



_−_ [1]
� _ρ_ _[P]_ [ +] _[ ν∂]_ _[j]_ _[U]_ _[i]_ _[ −]_ _[τ]_ _[ij]_



where _U_ _i_ is the mean velocity, _ρ_ is the constant density, _P_ is the mean pressure and _ν_ is the kinematic viscosity. The Reynolds-stress _τ_ _ij_ is the subject of
modelling. This symmetric, second-order tensor field can be decomposed into


4


an anisotropic _a_ _ij_ = 2 _kb_ _ij_ and isotropic part [2] 3 _[kδ]_ _[ij]_



_τ_ _ij_ = 2 _k_ _b_ _ij_ + [1]
� 3 _[δ]_ _[ij]_



_,_ (2)
�



in which the baseline model, _b_ _[o]_ _ij_ [=] _[ −]_ _[ν]_ _k_ _[t]_ _[S]_ _[ij]_ [, forms a linear relation between]

anisotropy and the mean-strain rate tensor _S_ _ij_ via the scalar eddy viscosity _ν_ _t_ .
Commonly, _ν_ _t_ is computed using a transport model such as _k_ - _ω_ SST [15], in
which _k_ is the turbulent kinetic energy and _ω_ the specific dissipation rate.
In order to extract the model-form error in these models from high-fidelity
data sources, we compute the residuals of the baseline turbulence model given
the data. The residual for the constitutive relation is equivalent to an additive
term _b_ [∆] _ij_ [leading to an augmented constitutive relation]

_b_ _ij_ = _−_ _[ν]_ _k_ _[t]_ _[S]_ _[ij]_ [ +] _[ b]_ _ij_ [∆] _[.]_ (3)


To evaluate _b_ [∆] _ij_ [it is necessary to estimate] _[ ν]_ _[t]_ [, therefore also] _[ ω]_ [ needs to be]
specified. In [12, 22], _ω_ was efficiently obtained by passively solving the _ω_
transport equation given high-fidelity data for _U_ _i_, _k_ and _b_ _ij_ . The associated _ν_ _t_
was then used to compute _b_ [∆] _ij_ [with (3). This method is named frozen-RANS as]
only one equation is solved iteratively while the remaining variables are frozen.
Despite the fact that _b_ [∆] _ij_ [also alters the production of turbulent kinetic energy]
_P_ _k_, it is not evident that solving the _k_ equation given the data and the frozen
_ω_ should lead to the same _k_ as present in the data. Therefore, we introduce _k_ corrective-frozen-RANS for which we also compute the residual of the _k_ equation
alongside the computation of the frozen _ω_ . The residual is equivalent to an
additive correction term, which we define as _R_, leading to an augmented _k_ - _ω_
SST model


_∂_ _t_ _k_ + _U_ _j_ _∂_ _j_ _k_ = _P_ _k_ + _R −_ _β_ _[∗]_ _ωk_ + _∂_ _j_ [( _ν_ + _σ_ _k_ _ν_ _t_ ) _∂_ _j_ _k_ ] _,_ (4)

_∂_ _t_ _ω_ + _U_ _j_ _∂_ _j_ _ω_ = _[γ]_ ( _P_ _k_ + _R_ ) _−_ _βω_ [2] + _∂_ _j_ [( _ν_ + _σ_ _ω_ _ν_ _t_ ) _∂_ _j_ _ω_ ] + _CD_ _kω_ _,_ (5)

_ν_ _t_


in which the production of turbulent kinetic energy is augmented by _b_ [∆] _ij_ [to]
_P_ _k_ = 2 _k_ ( _b_ _[o]_ _ij_ [+] _[ b]_ [∆] _ij_ [)] _[∂]_ _[j]_ _[U]_ _[i]_ [. The corresponding eddy viscosity is] _[ ν]_ _[t]_ [ =] max( _aa_ 11 _ω,SFk_ 2 ) [.]
The other standard terms of _k_ - _ω_ SST read


1
_CD_ _kω_ = max 2 _σ_ _ω_ 2 _,_
� _ω_ [(] _[∂]_ _[i]_ _[k]_ [)(] _[∂]_ _[i]_ _[ω]_ [)] _[,]_ [ 10] _[−]_ [10] �



�



2 _√_


_β_ _[∗]_

�



_√_


_β_ _[∗]_

�



4 []


_,_

�� 



_F_ 1 = tanh


_F_ 2 = tanh













�

�



min



max



_√k_

_β_ _[∗]_ _ωy_ _[,]_ [ 500] _y_ [2] _ω_ _[ν]_



_y_ [2] _ω_



�



_,_ _CD_ [4] _[σ]_ _[ω]_ _kω_ [2] _[k]_ _y_ [2]



max



2 _√k_

_β_ _[∗]_ _ωy_ _[,]_ [ 500] _y_ [2] _ω_ _[ν]_



_y_ [2] _ω_



2 []


_,_

�� 



Φ = _F_ 1 Φ 1 + (1 _−_ _F_ 1 )Φ 2 _,_



(6)



5


Table 1: Mean-squared error _ϵ_ of reconstructed velocity _U_ _i_ and Reynolds-stress
_τ_ _ij_ for different test cases with _b_ [∆] _ij_ [and] _[ R]_ [ added as static fields to the solver.]
Normalisation with _ϵ_ of the baseline _k_ - _ω_ SST results _U_ _i_ _[o]_ [and] _[ τ]_ _[ o]_ _ij_ [. Description of]
cases in Section 4.


Case _ϵ_ ( _U_ _i_ ) _·_ 10 _[−]_ [5] _ϵ_ ( _U_ _i_ ) _/ϵ_ ( _U_ _i_ _[o]_ [)] _ϵ_ ( _τ_ _ij_ ) _·_ 10 _[−]_ [6] _ϵ_ ( _τ_ _ij_ ) _/ϵ_ ( _τ_ _ij_ _[o]_ [)]


PH 10595 1 _._ 74 0 _._ 00165 36 _._ 7 0 _._ 1495
CD 12600 31 _._ 4 0 _._ 0229 7 _._ 21 0 _._ 4781
CBFS 13700 59 _._ 6 0 _._ 22703 1 _._ 34 0 _._ 4949


in which the latter blends the coefficients Φ _→_ (Φ 1 _,_ Φ 2 )


_α_ = (5 _/_ 9 _,_ 0 _._ 44) _, β_ = (3 _/_ 40 _,_ 0 _._ 0828) _, σ_ _k_ = (0 _._ 85 _,_ 1 _._ 0) _, σ_ _ω_ = (0 _._ 5 _,_ 0 _._ 856) _._ (7)



The remaining terms are _β_ _[∗]_ = 0 _._ 09 _, a_ 1 = 0 _._ 31 and _S_ = �2 _S_ _ij_ _S_ _ij_ . During

the iterative computation of the frozen _ω_ the residual of the _k_ equation is fed
back into the _ω_ equation until convergence is achieved. In order to validate
that the resulting fields compensate the model-form error, _b_ [∆] _ij_ [and] _[ R]_ [ are added]
as static fields to a modified OpenFOAM solver [23] and a CFD simulation is
performed starting from the baseline solution for the flow configurations described in Section 4, for which high-quality data is available. The mean-squared
error between the high-fidelity data and the reconstructed velocity _U_ _i_ as well
as the Reynolds-stress _τ_ _ij_ is low, see Table 1. Also the stream-wise velocity
profiles shown in Figure 2 demonstrate that the high-fidelity mean-flow data is
essentially reproduced given _b_ [∆] _ij_ [and] _[ R]_ [. The] _[ k]_ [-corrective-frozen-RANS approach]
requires full-field data, but is not based on an inversion procedure, e.g. using
adjoint-based optimisation as in [6, 8], which makes it very cost-efficient.


**2.2** **Nonlinear eddy-viscosity models for** _b_ [∆] _ij_ **[and]** _[ R]_


In order to discover corrections for the model-form error _b_ [∆] _ij_ [and] _[ R]_ [, we need to]
decide on a modelling ansatz. Within this mathematical framework the symbolic
regression targets to find specific expressions as corrections models. In [13], a
nonlinear generalisation of the linear eddy viscosity concept was proposed. This
concept has been used in several works on data-driven turbulence modelling

[2, 3]. The fundamental assumption is made that the anisotropy of the Reynoldsstress _b_ _ij_ not only depends on the strain rate tensor _S_ _ij_ = _τ_ [1] 2 [(] _[∂]_ _[j]_ _[U]_ _[i]_ [ +] _[ ∂]_ _[i]_ _[U]_ _[j]_ [) but]



stress _b_ _ij_ not only depends on the strain rate tensor _S_ _ij_ = _τ_ 2 [(] _[∂]_ _[j]_ _[U]_ _[i]_ [ +] _[ ∂]_ _[i]_ _[U]_ _[j]_ [) but]

also on the rotation rate tensor Ω _ij_ = _τ_ [1] 2 [(] _[∂]_ _[j]_ _[U]_ _[i]_ _[ −]_ _[∂]_ _[i]_ _[U]_ _[j]_ [) with the timescale]



also on the rotation rate tensor Ω _ij_ = _τ_ 2 [(] _[∂]_ _[j]_ _[U]_ _[i]_ _[ −]_ _[∂]_ _[i]_ _[U]_ _[j]_ [) with the timescale]

_τ_ = 1 _/ω_ . The Cayley-Hamilton theorem then dictates that the most general
form of the anisotropic part of the Reynolds-stress can be expressed as



_b_ _ij_ ( _S_ _ij_ _,_ Ω _ij_ ) =



_N_
� _T_ _ij_ [(] _[n]_ [)] _[α]_ _[n]_ [(] _[I]_ [1] _[, ..., I]_ [5] [)] _[,]_ (8)


_n_ =1


6


|bΔ and R k-ω SST LES (Breuer et al., 2009)<br>ij|Col2|
|---|---|
|||



(a) PH 10595





|bΔ and R k-ω SST DNS (Laval and Marquillie, 2010)<br>ij|Col2|
|---|---|
|0.5<br>1.0<br>1.5<br>2.0<br>y/H||


(b) CD 12600





|bΔ and R k-ω SST<br>ij<br>3<br>2<br>y/H<br>1|Col2|LES (Bentaleb et al., 2012)|
|---|---|---|
|1<br>2<br>3<br>y/H<br>bΔ<br>ij and R<br>k~~-~~ω SST|||


(c) CBFS 13700


Figure 2: Stream-wise velocity component for propagated model-form error
acquired using _k_ -corrective-frozen-RANS.


7


with ten nonlinear base tensors _T_ _ij_ [(] _[n]_ [)] and five corresponding invariants _I_ _m_ . Only
the first four base tensors and the first two invariants are used in this work, which

are


_T_ _ij_ [(1)] = _S_ _ij_ _,_

_T_ _ij_ [(2)] = _S_ _ik_ Ω _kj_ _−_ Ω _ik_ _S_ _kj_ _,_

_T_ _ij_ [(3)] = _S_ _ik_ _S_ _kj_ _−_ [1] 3 _[δ]_ _[ij]_ _[S]_ _[mn]_ _[S]_ _[nm]_ _[,]_

_T_ _ij_ [(4)] = Ω _ik_ Ω _kj_ _−_ 3 [1] _[δ]_ _[ij]_ [Ω] _[mn]_ [Ω] _[nm]_ (9)


_I_ 1 = _S_ _mn_ _S_ _nm_ _, I_ 2 = Ω _mn_ Ω _nm_ _._ (10)


Using this set for (8) we have an ansatz, which only requires functional expressions for the coefficients _α_ _n_, to model _b_ [∆] _ij_ [. However, computing] _[ b]_ [∆] _ij_ [using]
(3) requires a correct _k_ as discussed in Section 2.1. This aspect is taken into
account in the modelling ansatz for _R_, for which we take a closer look at the
eddy viscosity concept.
Both linear and nonlinear eddy viscosity models provide expressions for the
anisotropy _b_ _ij_ based on a local relation between stress and strain. Due to the
restriction of this local closure only the normal stresses 3 [2] _[kδ]_ _[ij]_ [ can account for]

nonlocal effects by transport equations for the turbulent quantities using convection and diffusion terms [15, 24]. The term _R_ provides local information
to correct the transport equations. Depending on the local sign of _R_ it either
increases or decreases the net production _P_ _k_ locally. Hence, it acts as an additional production or dissipation term, which can overcome the error in _k_ . We
model it in a similar way to the turbulent production


_R_ = 2 _kb_ _[R]_ _ij_ _[∂]_ _[j]_ _[U]_ _[i]_ _[,]_ (11)


which has the additional benefit that we can also use the framework of nonlinear
eddy viscosity models to model _R_ .
Since the general modelling framework is the same for both _b_ [∆] _ij_ [and] _[ R]_ [, a]
natural next step would be to combine both in order to find a single model
accounting for the sources of model-form error on the level of the constitutive
relation as well as within the turbulent transport equations. For example in

[12] models identified using genetic programming were modified such that any
additional contribution of the first base tensor _T_ _ij_ [(1)] in (8) was added with a
positive sign for the computation of _P_ _k_ . This ad-hoc correction was established
based on physical reasoning to avoid very low production close to walls and
led to significantly improved predictions. However, in contrast to [12] we have
extracted two target terms _b_ [∆] _ij_ [and] _[ R]_ [ using] _[ k]_ [-corrective-frozen-RANS, which also]
make it possible to systematically study (i) how to obtain corrections models
for each target individually and (ii) their combined effect on the predictions.
Given the polynomial model (8) and the set of base tensors (9) and invariants
(10) we are now left with the task of providing suitable expressions for _α_ _n_ ( _I_ 1 _, I_ 2 )


8


for _n_ = 1 _, ...,_ 4 to overcome the model-form error. This is the purpose of the
deterministic symbolic regression technique detailed in the following section.

### **3 Model discovery methodology**


Deterministic symbolic regression constructs a large library of nonlinear candidate functions to regress data. It identifies the relevant candidates by adopting
a sparsity constraint. Two fundamental methods have been proposed: Sparse
identification of nonlinear dynamics (SINDy) [20, 25] and fast function extraction (FFX) [26]. Both methods were applied in several areas of physical modelling. In the following, we introduce the steps of the model discovery methodology SpaRTA based on FFX, for which a library is constructed using a set of raw
input variables and mathematical operations. The model selection uses elastic
net regression. Finally, for the inference of the model coefficients the stability
requirements of a CFD solver are considered. An overview of SpaRTA is given
in Figure 1.


**3.1** **Building a library of candidate functions**


The deterministic symbolic regression requires a library of candidate functions,
from which a model is deduced by building a linear combination of the candidates. Hence, the library is an essential element of the entire methodology and
needs to accommodate relevant candidates explaining the data. We rely on the
nonlinear eddy viscosity concept and aim to find models for _α_ _n_ in (8) given as
primitive input features the invariants _I_ 1 and _I_ 2 . For the present work we focus
on a library, in which the primitive input features are squared and the resulting
candidates are multiplied by each other leading to a maximum degree of 6. In
addition to the two invariants we also include a constant function _c_ to the set

of raw input features. The resulting vector _**B**_ reads


_**B**_ =� _c, I_ 1 _, I_ 2 _, I_ 1 [2] _[, I]_ 2 [2] _[, I]_ 1 [2] _[I]_ 2 [3] _[, I]_ 1 [4] _[I]_ 2 [2] _[, I]_ [1] _[I]_ 2 [2] _[, I]_ [1] _[I]_ 2 [3] _[,]_


_T_
_I_ 1 _I_ 2 [4] _[, I]_ 1 [3] _[I]_ [2] _[, I]_ 1 [2] _[I]_ 2 [4] _[, I]_ 1 [2] _[I]_ [2] _[, I]_ [1] _[I]_ [2] _[, I]_ 1 [3] _[I]_ 2 [2] _[, I]_ 1 [2] _[I]_ 2 [2] (12)
�


with the cardinality of _**B**_, _|_ _**B**_ _|_ = 16.
For the library to regress models for _b_ [∆] _ij_ [each function of] _**[ B]**_ [ is multiplied with]

each base tensor _T_ _ij_ [(] _[n]_ [)] [, leading to the library of tensorial candidate functions]


_T_
_**C**_ _b_ ∆ _ij_ [=] � _cT_ _ij_ [(1)] _[, cT]_ _ij_ [ (2)] _[, . . ., I]_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _ij_ � _._ (13)


In order to regress models for _R_ the double dot product of each function in _**C**_ _b_ ∆
_ij_
with the mean velocity gradient tensor _∂_ _j_ _U_ _i_ is computed, leading to


_T_
_**C**_ _R_ = � _cT_ _ij_ [(1)] _[∂]_ _[j]_ _[U]_ _[i]_ _[, . . ., I]_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _ij_ _[∂]_ _[j]_ _[U]_ _[i]_ � _._ (14)


9


The two libraries _**C**_ _b_ ∆ _ij_ [and] _**[ C]**_ _[R]_ [ are evaluated given the high-fidelity validation]
data for each test case and stored column-wise in matrices



_**C**_ _b_ ∆ _ij_ [=]


_**C**_ _R_ =













_cT_ _xx_ [(1)] _[|]_ _k_ =0 _cT_ _xx_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _xx_ _[|]_ _k_ =0
_cT_ _xy_ [(1)] _[|]_ _k_ =0 _cT_ _xy_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _xy_ _[|]_ _k_ =0
_cT_ _xz_ [(1)] _[|]_ _k_ =0 _cT_ _xz_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _xz_ _[|]_ _k_ =0
_cT_ _yy_ [(1)] _[|]_ _k_ =0 _cT_ _yy_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _yy_ _[|]_ _k_ =0
_cT_ _xz_ [(1)] _[|]_ _k_ =0 _cT_ _yz_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _yz_ _[|]_ _k_ =0
_cT_ _zz_ [(1)] _[|]_ _k_ =0 _cT_ _zz_ [(2)] _[|]_ _k_ =0 _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _zz_ _[|]_ _k_ =0

... ... ...
_cT_ _zz_ [(1)] _[|]_ _k_ = _K_ _cT_ _zz_ [(2)] _[|]_ _k_ = _K_ _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _zz_ _[|]_ _k_ = _K_



_cT_ _ij_ [(1)] _[∂]_ _[j]_ _[U]_ _[i]_ _[|]_ _[k]_ [=0] _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _ij_ _[∂]_ _[j]_ _[U]_ _[i]_ _[|]_ _[k]_ [=0]

... ...
_cT_ _ij_ [(1)] _[∂]_ _[j]_ _[U]_ _[i]_ _[|]_ _[k]_ [=] _[K]_ _. . ._ _I_ 1 [2] _[I]_ 2 [2] _[T]_ [ (4)] _ij_ _[∂]_ _[j]_ _[U]_ _[i]_ _[|]_ _[k]_ [=] _[K]_









_∈_ R 6 _K×|_ _**C**_ _b_ ∆ _ij_ _[|]_ _,_ (15)





 _∈_ R _K×|_ _**C**_ _R_ _|_ _,_ (16)



in which _K_ is the number of mesh points of the test case at hand. The corresponding target data _b_ [∆] _ij_ [and] _[ R]_ [ are stacked to vectors]


_**b**_ **[∆]** = � _b_ [∆] _xx_ _[|]_ _[k]_ [=0] _[, b]_ [∆] _xy_ _[|]_ _[k]_ [=0] _[, ..., b]_ _zz_ [∆] _[|]_ _[k]_ [=] _[K]_ � _T_ _∈_ R 6 _K_ _,_ (17)

_**R**_ = [ _R|_ _k_ =0 _, R|_ _k_ =1 _, ..., R|_ _k_ = _K_ ] _[T]_ _∈_ R _[K]_ _._ (18)


**3.2** **Model selection using sparsity-promoting regression**


Given the above defined libraries the task is to form a linear model to regress
the target data **∆** = _**b**_ **[∆]** or _**R**_ by finding the coefficient vector **Θ**


**∆** = _**C**_ ∆ **Θ** _,_ (19)


which represents a large, overdetermined system of equations. When using ordinary least-squares regression a dense coefficient vector **Θ** is obtained, resulting in
overly complex models, which are potentially overfitting the data given the large
libraries (13) and (14). Due to multi-collinearity between the candidates, _**C**_ ∆
can be ill-conditioned, so that the coefficients may also display large differences
in magnitude expressed in a large _l_ 1 -norm of **Θ** . Such models are unsuitable to
be implemented in a CFD solver as they increase the numerical stiffness of the
problem and impede convergence of the solution.
Following the idea of parsimonious models we constrain the search to models
which optimally balance error and complexity and are not overfitting the data

[25]. In principle, given a library a combinatoric study can be carried out, by
performing an ordinary least-squares regression for each possible subset of candidates. Starting from each single candidate function individually, proceeding
with all possible pairs up to more complex combinations. As the number of possible models grows exponentially with the number of candidates _I_ = 2 _[|]_ _**[C]**_ [∆] _[|]_ _−_ 2


10


this approach becomes already infeasible for the simple libraries (13) and (14)
with _|_ _**C**_ ∆ _| ≈_ 64.
Hence, we follow [25, 26] and engage sparsity-promoting regularisation of the
underlying least-squares optimisation problem. The model-discovery procedure
is divided into two parts: (i) model selection and (ii) model inference, see Figure
1. For the first step, the model selection, we use the elastic net formulation



**Θ** = arg min **Θ** ˆ



ˆ 2 ˆ
_**C**_ ∆ **Θ** _−_ **∆** **Θ**
��� ��� 2 [+] _[ λρ]_ ��� ��� 1



ˆ 2
+ 0 _._ 5 _λ_ (1 _−_ _ρ_ ) **Θ** (20)
��� ��� 2 _[,]_


which blends the _l_ 1 - and _l_ 2 -norm regularisation given the mixing parameter _ρ ∈_

[0 _,_ 1] and the regularisation weight _λ_, to promote the sparsity of **Θ** [26, 27]. On
its own, the _l_ 1 -norm, known as Lasso-regression, promotes sparsity by allowing
only a few nonzero coefficients while shrinking the rest to zero. The _l_ 2 -norm,
known as Ridge-regression, enforces relatively small coefficients without setting
them to zero, but is able to identify also correlated candidate functions instead of
picking a single one. By combining both methods, the elastic net can find sparse
models with a good predictive performance. Besides the mixing parameter, also
the regularisation parameter _λ_ shapes the form of the model: For a very large _λ_
the vector **Θ** will only contain zeros independent of _ρ_ . The amount of nonzero
coefficients increases for smaller _λ_ values making the discovery of sparse models
possible.
Given the elastic net regularisation method we need to specify suitable combinations of the weight _λ_ and type of the regularisation _ρ_, for which the optimisation problem (20) is solved. Most commonly the optimal ( _λ, ρ_ ) combination
is found based on a strategy to avoid overfitting of the resulting models, e.g.
using cross-validation [25], for which the data is split into a training and a test
set. While the optimisation problem given a grid ( _**λ**_ _,_ _**ρ**_ ) is solved on the former,
only the model with the best performance evaluated on the latter survives. For
the purpose of CFD a true validation of the models can only be performed once
they are implemented in a solver and applied to a test case. In order to not
overcharge the role of the training data from _k_ -corrective-frozen-RANS at this
stage of the methodology, we select a wide spectrum of models varying in accuracy and complexity using (20) instead of a single one. The validation task will
be performed later using a CFD solver.
Following [26] we use


_**ρ**_ = [0 _._ 01 _,_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 5 _,_ 0 _._ 7 _,_ 0 _._ 9 _,_ 0 _._ 95 _,_ 0 _._ 99 _,_ 1 _._ 0] _[T]_ _,_ (21)


which ensures that we cover a substantial range of different regularisation types.
The upper limit of the regularisation weight is defined as _λ_ max = max( _|_ _**C**_ ∆ _[T]_ **[∆]** _[|]_ [)] _[/]_ [(] _[Kρ]_ [),]
because for any _λ > λ_ max all elements in **Θ** will be equal to zero. The entire

vector


_**λ**_ = [ _λ_ 0 _, ..., λ_ max ] _[T]_ (22)


11


is defined of having 100 entries between _λ_ 0 = _ξλ_ max with _ξ_ = 10 _[−]_ [3] uniformly
spaced using a log-scale as defined in [26]. This provides a search space ( _**λ**_ _,_ _**ρ**_ ),
the elastic net, which is large enough and has an appropriate resolution. At
each grid point ( _λ_ _i_ _, ρ_ _j_ ) a vector **Θ** [(] ∆ _[i,j]_ [)] as a solution of (20) is found using the
coordinate descent algorithm. The duration for the model selection step given
the number of data points _K ∼_ 15000 is of the order of a minute on a standard
consumer laptop.
Solving (20) for different ( _λ_ _i_ _, ρ_ _j_ ) might produce **Θ** [(] ∆ _[i,j]_ [)] with the same abstract
model form **Θ** **[¯]**, which means that the same entries are equal to zero. As the
specific values of the coefficients will be defined in the next step, the selection
step of SpaRTA concludes with filtering out the set of _D_ unique abstract model
forms _**D**_ ∆ = � **¯Θ** _d_ ∆ �� _d_ = 1 _, ..., D_ �.


**3.3** **Model inference for CFD**


The abstract models _**D**_ ∆ are found using standardised candidates, because the
relevance of each candidate should not be determined by its magnitude during
the model selection step. With the aim of defining a model with the correct
units, we need to perform an additional regression using the unstandardised
candidate functions for each subset determined by the abstract model forms in
_**D**_ ∆, which is the purpose of the model inference step outlined in the following.
In [25, 28, 29] this was done using ordinary least-squares regression for problems in the domains of dynamical systems and biological networks. As mentioned above, the ability of the CFD solver, in which the models will be implemented, to produce a converged solution is sensitive to large coefficients, which
has been reported in [11, 12, 22]. We take this additional constraint into account
by performing a Ridge regression



(23)
2 _[,]_



**Θ** _[s,d]_ ∆ [= arg min] **Θ** ˆ _[s,d]_ ∆



2
_**C**_ ∆ _s_ **[Θ]** [ˆ] _[s,d]_ ∆ _[−]_ **[∆]**
��� ��� 2



2 **Θ** ˆ _s,d_ ∆ 2

2 [+] _[ λ]_ _[r]_ ��� ��� 2



in which _λ_ _r_ is the Tikhonov-regularisation parameter. The index _s_ denotes the
submatrix of _**C**_ ∆ and the subvector of **Θ** _[d]_ ∆ [consisting of the selected columns]
or elements respectively as defined in _**D**_ ∆ . The elements of **Θ** _[d]_ ∆ [associated with]
the inactive candidates are zero and are not modified during this step.
By using the _l_ 2 -norm regularisation the magnitude of the nonzero coefficients is shrunk [25, 30]. In general, low values for _λ_ _r_ reduce the bias introduced through regularisation, but lead to larger coefficient values, and vice
versa. Since shrinkage of the coefficients also reduces the influence of candidate
functions with a lower magnitude compared to others, we need to find a tradeoff between error of the model on the target data **∆** and the likelihood that the
model will deliver converged solutions when used in a CFD solver. The problem of finding such an optimum is that the latter aspect can only be answered
retrospectively. Recently, this problem has been addressed in [31] by embedding CFD simulations in the search for correction models guided by genetic
programming. While this increases the costs of the model search drastically,


12


it also significantly increases the chance of delivering models with better convergence properties. Even though this procedure provides a strong indication,
the identified models are also not guaranteed to converge _a priori_ for any other
test case outside the training set. Via testing using the cases in Section 4, we
have identified 0 _._ 1 _< λ_ _r_ _<_ 0 _._ 01 able to deliver coefficients in a range balancing
the error on the target data ∆and the likelihood to produce converged CFD
solutions.

Our efforts are based on an empirical observation, but do not guarantee a
well-behaving numerical setup under all conditions. However, we have identified
corrections of _b_ [∆] _ij_ [as the only contribution which can do harm to the convergence]
properties for the given test cases. Therefore, if a model does not converge, we
further decrease the coefficients by a factor _ξ_ = 0 _._ 1, for the model correcting
_b_ [∆] _ij_ [only. This ad-hoc intervention is sufficient to achieve convergence for the]
studied cases.
Finally, the resulting coefficient vector **Θ** _[d]_ ∆ [is used to retrieve the symbolic]
expression of the models by a dot product with the library of candidate functions
_**C**_ ∆ in (13) and (14)


_M_ ∆ _[d]_ [:=] _**[ C]**_ ∆ _[T]_ **[Θ]** ∆ _[d]_ _[,]_ (24)


which are implemented in the open-source finite-volume code OpenFOAM [23].
The divergence terms of the equations are discretised with linear upwinding and
turbulent diffusion with 2nd order central differencing. In summary, the model
discovery step of SpaRTA selects models utilising elastic net regression in (20)
and further infers the coefficients of the selected models in (23). The latter
process is guided by the aim to discover models complying with the restrictions
of a CFD solver.

### **4 Test cases and high-fidelity data**


In order to apply SpaRTA we need full-field data of _U_ _i_, _k_ and _τ_ _ij_, which we take
from LES and DNS studies conducted by other researchers. We have selected
three test cases of separating flows over curved surfaces in two-dimensions with
similar Reynolds-numbers. For each case fine meshes are selected, which ensure that the discretisation error is much smaller compared to the error due to
turbulence modelling.
**Periodic hills (PH)**, for which the flow is over a series of hills in a channel.
Initially proposed by [32] this case has been studied both experimentally as well
as numerically in detail. We use LES data from [33] for _Re_ = 10595 (PH 10595 )
to apply SpaRTA and test the performance of the resulting models. In addition,
we also use experimental data from [34] at a much larger _Re_ = 37000 (PH 37000 )
in order to test the models outside the range of the training data. The numerical
mesh consists of 120 _×_ 130 cells. Cyclic boundary conditions are used at the
inlet and outlet. The flow is driven by a volume forcing defined to produce a
constant bulk velocity.


13


**Converging-diverging channel (CD)** . A DNS study of the flow within a
channel, in which an asymmetric bump is placed, exposed to an adverse pressure
gradient was performed by [35] for _Re_ = 12600 (CD 12600 ). The flow shows a
small separation bubble on the lee-side of the bump, which is challenging for
RANS to predict. The numerical mesh consists of 140 _×_ 100 cells. The inlet
profile was obtained from a channel-flow simulation at equivalent _Re_ .
**Curved backward-facing step (CBFS)** . In [36] a LES simulation of a
flow over a gently-curved backward-facing step was performed at _Re_ = 13700
(CBFS 13700 ). Similar to PH also for this flow the mean effect of separation
and reattachment dynamics is the objective. The numerical mesh consists of
140 _×_ 150 cells. The inlet was obtained from a fully-developed boundary layer
simulation.

Despite the simple geometries, the mean effect of the separation and reattachment dynamics of a flow on a curved surface is a challenging problem for
steady-RANS approaches. Especially, PH serves as an important testbed for
classical and data-driven approaches for turbulence modelling, e.g. [2, 37], but
also the other two have been introduced with the purpose of closure investigation.

### **5 Results**


The method SpaRTA introduced in Section 3 is applied to the three test cases
of Section 4. The models resulting from the model-discovery are presented and
their mean-squared error on the training data is evaluated. In order to identify
the models with the best predictive capabilities, we carry out cross-validation
of the resulting models using CFD [30]: Models identified given training data
of one case are used for CFD simulations of the remaining two case. For each
case a single model is chosen as the best-performing one. Finally, the three
resulting models are tested in a _true_ prediction for the flow over periodic hills
at _Re_ = 37000.


**5.1** **Discovery of models and their error on training data**


The goal of the model-discovery is to identify an ensemble of diverse models with
small coefficients, varying in model-structure (complexity) and accuracy. Such
an ensemble is better-suited for the cross-validation on unseen test cases, than a
selection of the best models given only the training data. The sparse-regression
for _b_ [∆] _ij_ [applied to the three test cases PH] [10595] [, CBFS] [13700] [ and CD] [12600] [ resulted]
in 52, 114 and 136 distinct models respectively, see Figure 3a for the results
for PH 10595 . It can be observed that, in general, an increase in complexity of
a model leads to a reduction of the error. But, the bias introduced through
the ridge regression of the inference step in SpaRTA, see Section 3.3, shrinks
the model coefficients. If a coefficient is associated with a candidate function
with a much lower magnitude compared to others, due to shrunk coefficients it
becomes less relevant. The result is a staircase structure of the error: Models


14


Table 2: Best-predictive models with rank (index _i, j, k_ in Figure 7) and normalised error on velocity _ϵ_ ( _U_ ) _/ϵ_ ( _U_ _[o]_ ) for different cases.


PH 10595 CD 12600 CBFS 13700
Model index _i_ _ϵ_ ( _U_ ) _/ϵ_ ( _U_ _[o]_ ) index _j_ _ϵ_ ( _U_ ) _/ϵ_ ( _U_ _[o]_ ) index _k_ _ϵ_ ( _U_ ) _/ϵ_ ( _U_ _[o]_ )


_M_ [(1)] (1.) 0.22287 (19.) 0.21146 - 0.30413
_M_ [(2)] - 0.38867 (1.) 0.20828 (26.) 0.40154
_M_ [(3)] (3.) 0.22744 - 0.22422 (1.) 0.30655


show a different form but have similar error. This pattern can also be observed
for the other cases and becomes even more prominent for the models regressing
_R_ . For this target the model discovery resulted in 18 and 19 distinct model
forms for CD 12600 and PH 10595 respectively, see Figure 3b for results of case
PH 10595 . For CBFS 13700 only three models have been found. We identify _T_ _ij_ [(1)] [,]

_I_ 1 _T_ _ij_ [(1)] and _I_ 2 _T_ _ij_ [(1)] as the relevant candidates to regress _R_, and models combining
all three give the lowest error per test case.
In order to reduce the redundancy within the ensemble of models regressing
_b_ [∆] _ij_ [and] _[ R]_ [ we select only a representative subset of models.] This ensemble
needs to acknowledge the hierarchical structure of diverse model-forms and their
accuracy. In an ad-hoc way, we hand-select 5 models for _b_ [∆] _ij_ [and 3 for] _[ R]_ [, except]
for CBFS 13700 only 1. The ensembles of selected models are shown in Figure
4 and Figure 5. The result is a hierarchical spectrum of models regressing the
training data varying in complexity and error. Given this ensemble we study
the performance of each model for predictions in the next section.


**5.2** **Cross-validation using CFD**


Cross-validation tests how well models identified on training data perform on
unseen test cases [30]. This assessment allows to determine the best-predictive
models from a set. As stated above, the role of the frozen, training data should
not be overcharged, so that we cross-validate using CFD. By doing so, we can
assess the validity of SpaRTA as a tool for model discovery as well as the predictive performance of the identified models outside of their training set.
The selected correction models regress _b_ [∆] _ij_ [and] _[ R]_ [ individually and can also be]
applied individually for predictions when implemented in the solver, i.e. a model
correcting _b_ [∆] _ij_ [can be used without a correction of] _[ R]_ [ and vice-versa. This gives]
us 8 models per training data for PH 10595 and CD 12600 and 6 for CBFS 13700 .
Also, we can study their combined effect. With 5 models for _b_ [∆] _ij_ [and 3 for] _[ R]_
we have additional 15 possible combinations, which makes in total 23 distinct
models for the training data of PH 10595 and CD 12600 and 11 distinct model
combinations for the training data of CBFS 13700 . For the cross-validation in
the following, we conduct in total 35 for test cases PH 10595 and CD 12600 and 47
simulations for test case CBFS 13700 including the baseline simulation with the
uncorrected _k_ - _ω_ SST.


15


(a) _b_ [∆] _ij_


(b) _R_


Figure 3: Model-structure of all discovered models using SpaRTA and meansquared error on training data for PH 10595 . The matrix (l.) shows the values of
the active (coloured) candidate functions (x-axis) for each model _M_ _i_ with model
index _i_ (y-axis). The mean-squared error between the frozen data _b_ [∆] _ij_ [and the]
model is also shown (r.).


16


(a) PH 10595


(b) CD 12600


(c) CBFS 13700


Figure 4: Selected models on frozen data _b_ [∆] _ij_ [.]


17


Figure 5: Selected models on frozen data _R_ .









|2|Col2|Col3|Col4|
|---|---|---|---|
|PH<br>0<br>2<br>4<br>6<br>8<br>||||
|PH<br>0<br>2<br>4<br>6<br>8<br>|CBFS|CBFS|CBFS|


Figure 6: Mean-squared error of velocity vector of each correction model normalised by the mean-squared error of the baseline _k_ - _ω_ SST. The colour indicates
on which high-fidelity data the models have been identified. Full circles represent simulations using both corrections, while left-/right-filled circles represent
simulations using only correction for _R_ or _b_ [∆] _ij_ [respectively.]


18


(a) PH 10595


(b) CD 12600


19

(c) CBFS 13700


Figure 7: The two matrices (l.) show the models _M_ _i_ for _b_ [∆] _ij_ [and] _[ R]_ [. The mean-]
squared error in velocity _U_, production _P_ _k_ and Reynolds-stress _τ_ _ij_ normalised
by the mean-squared error of the baseline _k_ - _ω_ SST model is also shown (mid to
right).


In Figure 6 the mean-squared error of each model on the velocity field _ϵ_ ( _U_ )
normalised with the mean-squared error of the baseline _ϵ_ ( _U_ _[o]_ ) is shown. The
type of model, whether it is providing a correction both for _b_ [∆] _ij_ [and] _[ R]_ [ or for each]
one individually, and from which training data it originated from, is emphasised
by a unique marker form and color combination. Whether the correction for
_b_ [∆] _ij_ [needs to be scaled with] _[ ξ]_ [ = 0] _[.]_ [1 to achieve convergence, see Section 3.3, is]
indicated by a black marker edge. Most of the models show a good or even
substantial improvement over the baseline. But, for the set of models, only
providing a correction for _b_ [∆] _ij_ [, most but not all lead to an improvement of the]
resulting velocity field. In contrast to that, if only a correction for _R_ is deployed,
the result is a consistent, substantial improvement across all test cases. Using
both a model for _b_ [∆] _ij_ [and] _[ R]_ [ leads to a further improvement, except for test case]
CBFS 13700 . Surprisingly, the best model per test case is not always identified
on the associated training data. While this expectation holds for the cases
CBFS 13700 and CD 12600 it is not true for PH 10595, for which the other two
training sets deliver significantly better performing models. In general, the
data of CD 12600 and CBFS 13700 provide models, which are well performing on
all test cases presented.
In Figure 7, both the error and the model structure for the correction of
_b_ [∆] _ij_ [as well as for] _[ R]_ [ is shown. The models are ordered according to the mean-]
squared error on the stream-wise velocity _U_ . In line of the discussion of Figure
6 three groups can be identified: a few models, which lead to an increased
error compared to the baseline; a small group of models per test case, which
are equal or similar to the baseline; and the great majority of models, which
result in an improvement. It can be observed how the error in the velocity is
significantly reduced once a correction of _R_ is used. The two other error plots in
Figure 7 give an indication of the relative performance of the models compared
to the baseline. The first is the mean-squared error of the total production
_P_ _k_ within the _k_ equation and the second one is the mean-squared error of the
Reynolds-stress _τ_ _ij_ normalised by the baseline result. Following the rationale
of correcting terms with the baseline model, improving these terms should lead
to an improved velocity. For the cases CD 12600 and CBFS 13700, we see that the
error for _U_ and _P_ _k_ reduce simultaneously for most of the models. Also the error
in _τ_ _ij_ shows a reduction when the error in _U_ decreases, but not as significant
as the error in _P_ _k_ . For a group of models between model index 5 _< j <_ 20
for case CD 12600, we see a jump in the error in _P_ _k_ and _τ_ _ij_, while the error in
the velocity is not changing compared to the neighbouring models. For case
PH 10595, we also observe a strong reduction of the _P_ _k_ error for an active _R_
correction. Surprisingly, for these the error in _τ_ _ij_ increases. When the error in
_U_ is further reduced the error in _τ_ _ij_ also decreases, but the error in _P_ _k_ increases
again. It can be observered, that the best models correct the velocity up to
5 times better in mean-squared error than the _k_ - _ω_ SST baseline model. This
leaves still room for further improvement compared to the error using the frozen
data sets, see Table 1. But, especially for case CBFS 13700 the result is already
very close to the possible correction provided by the frozen data at least for _U_ .


20


Given this cross-validation assessment we select models _**M**_ [(] _[i]_ [)] = ( _M_ _b_ [(] [∆] _[i]_ [)] _[, M]_ _R_ [ (] _[i]_ [)] [)] _[T]_

based on the lowest _ϵ_ ( _U_ ) per case


_M_ _b_ [(1)] [∆] [=] �24 _._ 94 _I_ 1 [2] [+ 2] _[.]_ [65] _[I]_ 2 [1] � _T_ _ij_ [(1)] + 2 _._ 96 _T_ _ij_ [(2)] _[,]_

+ �2 _._ 49 _I_ 2 [1] [+ 20] _[.]_ [05] � _T_ _ij_ [(3)] + �2 _._ 49 _I_ 1 [1] [+ 14] _[.]_ [93] � _T_ _ij_ [(4)] _[,]_

_M_ _R_ [(1)] = 0 _._ 4 _T_ _ij_ [(1)] _[,]_ (25)


_M_ _b_ [(2)] [∆] [=] _[ T]_ [ (1)] _ij_ �0 _._ 46 _I_ 1 [2] [+ 11] _[.]_ [68] _[I]_ 2 [1] _[−]_ [0] _[.]_ [30] _[I]_ 2 [2] [+ 0] _[.]_ [37] �

+ _T_ _ij_ [(2)] � _−_ 12 _._ 25 _I_ 1 [1] _[−]_ [0] _[.]_ [63] _[I]_ 2 [2] [+ 8] _[.]_ [23] �

+ _T_ _ij_ [(3)] � _−_ 1 _._ 36 _I_ 2 [1] _[−]_ [2] _[.]_ [44] �

+ _T_ _ij_ [(4)] � _−_ 1 _._ 36 _I_ 1 [1] [+ 0] _[.]_ [41] _[I]_ 2 [1] _[−]_ [6] _[.]_ [52] � _,_

_M_ _R_ [(2)] = 1 _._ 4 _T_ _ij_ [(1)] _[,]_ (26)


_M_ _b_ [(3)] [∆] [=] _[ T]_ [1] �0 _._ 11 _I_ 1 [1] _[I]_ 2 [1] [+ 0] _[.]_ [27] _[I]_ 1 [1] _[I]_ 2 [2] _[−]_ [0] _[.]_ [13] _[I]_ 1 [1] _[I]_ 2 [3] [+ 0] _[.]_ [07] _[I]_ 1 [1] _[I]_ 2 [4]

+ 17 _._ 48 _I_ 1 [1] [+ 0] _[.]_ [01] _[I]_ 1 [2] _[I]_ 2 [1] [+ 1] _[.]_ [251] _[I]_ 1 [2] [+ 3] _[.]_ [67] _[I]_ 2 [1] [+ 7] _[.]_ [52] _[I]_ 2 [2] _[−]_ [0] _[.]_ [3] �

+ _T_ 2 �0 _._ 17 _I_ 1 [1] _[I]_ 2 [2] _[−]_ [0] _[.]_ [16] _[I]_ 1 [1] _[I]_ 2 [3] _[−]_ [36] _[.]_ [25] _[I]_ 1 [1] _[−]_ [2] _[.]_ [39] _[I]_ 1 [2] [+ 19] _[.]_ [22] _[I]_ 2 [1] [+ 7] _[.]_ [04] �

+ _T_ 3 � _−_ 0 _._ 22 _I_ 1 [2] [+ 1] _[.]_ [8] _[I]_ 2 [1] [+ 0] _[.]_ [07] _[I]_ 2 [2] [+ 2] _[.]_ [65] �

+ _T_ 4 �0 _._ 2 _I_ 1 [2] _[−]_ [5] _[.]_ [23] _[I]_ 2 [1] _[−]_ [2] _[.]_ [93] � _,_

_M_ _R_ [(3)] = 0 _._ 93 _T_ _ij_ [(1)] _[,]_ (27)


for which further details on the corresponding training data and the rank of the
model on each test case are given in Table 2. Especially model _M_ [(3)] performs
very well both on CBFS 13700 (rank 1.) and PH 10595 (3.). While the rank of
the others varies more between the test cases, they are still within the set of
well-performing models with _ϵ_ ( _U_ ) _/ϵ_ ( _U_ _[o]_ ) _<_ 0 _._ 5. Their predictions of stream-wise
velocity _U_, _k_, the Reynolds-stress component _τ_ _xy_ and the skin-friction coefficient
_C_ _f_ are shown in Figure 8 to 11 for the three test cases. As already stated for
the error evaluated on the entire domain discussed above, these three models
show an improvement of the spatial distribution of the predicted quantities in
comparison to the baseline prediction of _k_ - _ω_ SST. Especially the velocity is wellcaptured for all three. While _k_ is better identified compared to the baseline, we
still observe a discrepancy between the predictions and the data. For PH 10595
the three models do not fit the complex spatial structure especially in the shearlayer, but together encapsulate the data for most of the profiles. For CD 12600 the
models are underestimating _k_ for _x <_ 7 and overestimate it further downstream.
For CBFS 13700 the models also underestimate on the curved surface, but fit the
data better than the baseline for 3 _< x <_ 5. The magnitude of the Reynoldsstress component _τ_ _xy_ is underestimated on the curved surfaces of all test cases.


21


For PH 10595 the models fail to fit the complex spatial structure especially within
the separated shear-layer behind the hill and on the hill itself. The skin friction
coefficient _C_ _f_ and the associated separation and reattachment points are best
captured by _M_ [(1)] and _M_ [(3)] for PH 10595 and CBFS 13700 and systematically
under-estimated with _M_ [(2)], i.e. a shorter recirculation zone. For CD 12600, we
observe a small recirculation zone as reported in the literature, but too far downstream. However, the baseline _k_ - _ω_ SST drastically over-predicts this zone and
the model _M_ [(2)] ignores it entirely.
Overall, the models _M_ [(1)] and _M_ [(3)] agree best with the data, which is in
line with the global error on _U_ in Table 2. The models are different in their
form, but show similar error values and spatial structure across the test cases.
Model _M_ [(2)] tends to overestimate the magnitude of the quantities _U_, _k_ and
_τ_ _xy_ and therefore predicts smaller or no separation bubbles. This model was
identified using PH 10595 as training data and, ignoring the specific structure for
_b_ [∆]
_ij_ [, has the largest coefficient for correcting] _[ R]_ [, see (26), which leads to larger] _[ k]_
compared to the others, which is the reason for the systematic over-prediction.
In order to test how the models extrapolate to cases of larger _Re_, we predict
the flow over periodic hills at _Re_ = 37000, see Figure 12. Due to an increase
of turbulence this case has a significantly shorter recirculation zone. For this
_true_ prediction throughout the domain the three models improve significantly
compared to the baseline. Interestingly, the model _M_ [(2)] is delivering the best
fit of the data and the others tend to slightly underestimate it. Thus, taking
the results of the cross-validation on the low- _Re_ cases into account, the models
show a weak _Re_ -dependence, but overall robustness between the cases.

### **6 Conclusion and extension**


In this work SpaRTA was introduced to discover algebraic models in order to
correct the model-form error within the _k_ - _ω_ SST. For this novel machine learning
method two additive terms, on the level of the stress-strain relation _b_ [∆] _ij_ [and]
within the turbulent transport equations _R_, were identified by means of _k_ corrective-frozen-RANS, for which the governing equations are evaluated given
high-fidelity data of three cases of separating flows. It was validated that the
computed terms are compensating the model-form error and reproduce the highfidelity LES or DNS mean-flow data. Hence, _k_ -corrective-frozen-RANS is a costefficient way to distill useful information directly from full-field data without the
need of an inversion procedure.
Cross-validation of the discovered models using CFD was carried out to
rank the models. While using both corrections for _R_ as well es for _b_ [∆] _ij_ [lead to a]
systematic improvement of the predictions over the baseline, a correction only
for _R_ can already be enough to achieve sufficient results for the velocity field.
For the best performing models on each case both the global error on _U_ as well as
the spatial structure on _U_, _k_ and _τ_ _xy_ was coherent. The models also performed
well for the periodic hills flow at a much larger _Re_ -number ( _Re_ = 37000). As
the sparse regression is computationally inexpensive, SpaRTA allows for rapid


22


|3|Col2|
|---|---|
|2<br>||
|||



(a) PH 10595







(b) CD 12600

|M(1) M(2) M(3)<br>3<br>2<br>y/H<br>1<br>0<br>0 1 2 3 4|Col2|M(1) M(2) M(3)|k-ω SST LES|
|---|---|---|---|
|0<br>1<br>2<br>3<br>4<br>0<br>1<br>2<br>3<br>y/H<br>M(1)<br>M(2)<br>M(3)||||
|0<br>1<br>2<br>3<br>4<br>0<br>1<br>2<br>3<br>y/H<br>M(1)<br>M(2)<br>M(3)|||5<br>6<br>7<br>8|



(c) CBFS 13700


Figure 8: Predicted stream-wise velocity.


23




|3|Col2|Col3|
|---|---|---|
|2<br>|||
||||



(a) PH 10595











(b) CD 12600











(c) CBFS 13700


Figure 9: Predicted turbulent kinetic energy.


24


|2.0|Col2|Col3|Col4|
|---|---|---|---|
|6<br>7<br>8<br>9<br>10<br>11<br>12<br>0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>y/H||||
|6<br>7<br>8<br>9<br>10<br>11<br>12<br>0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>y/H||10<br>11|12|


(b) CD 12600

|3<br>2<br>y/H<br>1<br>0<br>0|Col2|Col3|Col4|Col5|LES|
|---|---|---|---|---|---|
|0<br>0<br>1<br>2<br>3<br>y/H|M(1)<br>M|(2)<br>M(3)|k~~-~~ω|SST|LES|
|0<br>0<br>1<br>2<br>3<br>y/H||||||
|0<br>0<br>1<br>2<br>3<br>y/H||||||
|0<br>0<br>1<br>2<br>3<br>y/H|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|



(c) CBFS 13700


Figure 10: Predicted shear stress.


25



(a) PH 10595




(a) PH 10595

|M(1) M(2) M(3)<br>3<br>2<br>1<br>0<br>0 2 4 6|M(1) M(2) M(3)|k-ω SST DNS<br>8 10 12|
|---|---|---|
|0<br>2<br>4<br>6<br>0<br>1<br>2<br>3<br>M(1)<br>M(2)<br>M(3)|||
|0<br>2<br>4<br>6<br>0<br>1<br>2<br>3<br>M(1)<br>M(2)<br>M(3)|||



(b) CD 12600

|M(1) M(2) M(<br>0.005<br>f<br>0.000<br>−5.0 −2.5 0.0 2.5 5.0|M(1) M(2) M(|3) k-ω SST LES<br>7.5 10.0 12.5 15.0|
|---|---|---|
|~~−~~5.0<br>~~−~~2.5<br>0.0<br>2.5<br>5.0<br>0.000<br>0.005<br>f<br>M(1)<br>M(2)<br>M(|||
|~~−~~5.0<br>~~−~~2.5<br>0.0<br>2.5<br>5.0<br>0.000<br>0.005<br>f<br>M(1)<br>M(2)<br>M(|||



(c) CBFS 13700


Figure 11: Predicted skin friction coefficient.


Figure 12: Flow over periodic hills at _Re_ = 37000 using correction models
compared to baseline _k_ - _ω_ SST and experimental data of [34].


26


discovery of robust models, i.e. a model trained for one flow may perform well
for flows outside of the training range, but with similar features.
Overall, the present systematic study has shown the capabilities of SpaRTA
to discover effective corrections to _k_ - _ω_ SST. Further work will focus on making
the model-filtering and the inference step of SpaRTA more systematic and datadriven. We will also apply SpaRTA to a larger variety of flow cases in order
to show its potential for rapid model discovery of corrections for industrial

purposes.

### **acknowledgements**


The authors wish to thank Richard Sandberg for sharing OpenFOAM code for
comparison of implementations and Michael Breuer for providing the full-field
LES and DNS simulation data for the periodic hill flow case.

### **Compliance with Ethical Standards**


**Funding** This research has received funding from the European Unions Seventh
Framework Programme under grant number ACP3-GA-2013-605036, UMRIDA project.
**Conflict of Interest** The authors declare that they have no conflict of interest.

### **References**


[1] J. Slotnick, A. Khodadoust, J. Alonso, D. Darmofal, W. Gropp, E. Lurie,
D. Mavriplis, CFD Vision 2030 Study: A Path to Revolutionary Computational
Aerosciences. Tech. Rep. March (2014). DOI 10.1017/CBO9781107415324.004


[2] H. Xiao, P. Cinnella, Progress in Aerospace Sciences (2019). DOI 10.1016/j.
paerosci.2018.10.001


[3] K. Duraisamy, G. Iaccarino, H. Xiao, Annual Review of Fluid Mechanics **51** (1)
(2019). DOI 10.1146/annurev-fluid-010518-040547


[4] P.A. Durbin, Annual Review of Fluid Mechanics **50** (1) (2018). DOI 10.1146/
annurev-fluid-122316-045020


[5] B.D. Tracey, K. Duraisamy, J.J. Alonso, in _AIAA SciTech Forum 53rd AIAA_
_Aerospace Sciences Meeting_ (2015), January, pp. 1–23. DOI 10.2514/6.2015-1287


[6] E.J. Parish, K. Duraisamy, Journal of Computational Physics **305** (2016). DOI
10.1016/j.jcp.2015.11.012


[7] A.P. Singh, K. Duraisamy, Physics of Fluids **28** (045110) (2016). DOI 10.1063/1.
4947045


[8] A.P. Singh, K. Duraisamy, Z.J. Zhang, in _55th AIAA Aerospace Sciences Meeting_
(2017). DOI 10.2514/6.2017-0993


[9] J. Ling, A. Kurzawski, J. Templeton, Journal of Fluid Mechanics **807** (2016).
DOI 10.1017/jfm.2016.615


[10] J.L. Wu, H. Xiao, E. Paterson, (January) (2018). URL `[http://arxiv.org/abs/](http://arxiv.org/abs/1801.02762)`

```
  1801.02762

```

27


[11] J. Weatheritt, R. Sandberg, Journal of Computational Physics **325**, 22 (2016).
DOI 10.1016/j.jcp.2016.08.015


[12] J. Weatheritt, R.D. Sandberg, International Journal of Heat and Fluid Flow **68**
(2017). DOI 10.1016/j.ijheatfluidflow.2017.09.017


[13] S.B. Pope, Journal of Fluid Mechanics **72** (2), 331 (1975). DOI 10.1017/
S0022112075003382


[14] T.B. Gatski, C.G. Speziale, Journal of Fluid Mechanics **254**, 59 (1993). DOI
10.1017/S0022112093002034


[15] M. Leschziner, _Statistical Turbulence Modelling for Fluid Dynamics - Demystified:_
_An Introductory Text for Graduate Engineering Students_ (Imperial College Press,
2015)


[16] S.B. Pope, _Turblent Flows_ (Cambridge University Press, 2000)


[17] S. Wallin, Engineering turbulence modelling for CFD with a focus on explicit
algebraic Reynolds stress models by. Phd thesis, Royal Institute of Technology
Stockholm (2000)


[18] J. Weatheritt, R.D. Sandberg, in _Conference: 11th International ERCOFTAC_
_Symposium on Engineering Turbulence Modelling and Measurements_, vol. 2
(2017), vol. 2, pp. 2–7


[19] H.D. Akolekar, J. Weatheritt, N. Hutchins, R.D. Sandberg, G. Laskowski,
V. Michelassi, in _Proceedings of ASME Turbo Expo 2018_ (Oslo, Norway, 2018),
pp. 1–13


[20] S.L. Brunton, J.L. Proctor, J.N. Kutz, Proceedings of the National Academy of
Sciences **113** (15) (2016). DOI 10.1073/pnas.1517384113


[21] S.H. Rudy, S.L. Brunton, J.L. Proctor, J.N. Kutz, Science Advances **3** (2017).
DOI 10.1126/sciadv.1602614


[22] J. Weatheritt, R.D. Sandberg, Journal of Ship Research (2019). DOI 10.5957/
josr.09180053


[23] H.G. Weller, G. Tabor, H. Jasak, C. Fureby, Computers in Physics **12** (6), 620
(1998). DOI 10.1063/1.168744


[24] D.C. Wilcox, _Turbulence Modeling for CFD_, 3rd edn. (DCW Industries, Inc.,
2006)


[25] S.L. Brunton, J.N. Kutz, _Data-Driven Science and Engineering: Machine Learn-_
_ing, Dynamical Systems, and Control_ (Cambridge University Press, Cambridge,
2019). DOI 10.1017/9781108380690


[26] T. McConaghy, in _Genetic Programming Theory and Practice IX. Genetic and_
_Evolutionary Computation._ (Springer, New York, NY, 2011)


[27] H. Zou, T. Hastie, Journal of the Royal Statistical Society. Series B: Statistical
Methodology **67** (2), 301 (2005). DOI 10.1111/j.1467-9868.2005.00503.x


[28] M. Quade, M. Abel, J. Nathan Kutz, S.L. Brunton, Chaos **28** (6) (2018). DOI
10.1063/1.5027470


[29] N.M. Mangan, J.N. Kutz, S.L. Brunton, J.L. Proctor, Proceedings of the Royal
Society A: Mathematical, Physical and Engineering Sciences **473** (2203) (2017).
DOI 10.1098/rspa.2017.0009


28


[30] Bishop, Christopher M, _Pattern Recognition and Machine Learning_ (Springer,
2006)


[31] Y. Zhao, H.D. Akolekar, J. Weatheritt, V. Michelassi, R.D. Sandberg, (2019)


[32] C.P. Mellen, J. Fr¨ohlich, W. Rodi, in _16th IMACS World Congress,_ (2000)


[33] M. Breuer, N. Peller, C. Rapp, M. Manhart, Computers and Fluids **38** (2), 433
(2009). DOI 10.1016/j.compfluid.2008.05.002


[34] C. Rapp, M. Manhart, Experiments in Fluids **51** (1), 247 (2011). DOI 10.1007/
s00348-011-1045-y


[35] J.P. Laval, M. Marquillie, in _Progress in Wall Turbulence: Understanding and_
_Modeling_, vol. 14, ed. by M. Stanislas, J. Jimenez, I. Marusic (ERCOFTAC Series,
2011), vol. 14, pp. 203–209. DOI 10.1007/978-90-481-9603-6


[36] Y. Bentaleb, S. Lardeau, M.A. Leschziner, Journal of Turbulence **13** (4) (2012).
DOI 10.1080/14685248.2011.637923


[37] S. Jakirlic, Extended excerpt related to the test case: ”Flow over a periodical
arrangement of 2D hills”. Tech. Rep. June (2012)


29


