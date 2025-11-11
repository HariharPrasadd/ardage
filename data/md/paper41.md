**ERROR ESTIMATES FOR DEEPONETS: A DEEP LEARNING**

**FRAMEWORK IN INFINITE DIMENSIONS**


SAMUEL LANTHALER


_Seminar for Applied Mathematics (SAM),_
_Eidgen¨ossische Technische Hochschule Z¨urich (ETHZ),_
_R¨amistrasse 101, 8092 Z¨urich, Switzerland_


SIDDHARTHA MISHRA


_Seminar for Applied Mathematics (SAM),_
_Eidgen¨ossische Technische Hochschule Z¨urich (ETHZ),_
_R¨amistrasse 101, 8092 Z¨urich, Switzerland_


GEORGE EM KARNIADAKIS


_Division of Applied Mathematics and School of Engineering,_
_Brown University,_
_Providence, RI 02912, USA_


Abstract. DeepONets have recently been proposed as a framework for learning nonlinear operators mapping between infinite dimensional Banach spaces.
We analyze DeepONets and prove estimates on the resulting approximation
and generalization errors. In particular, we extend the universal approximation property of DeepONets to include measurable mappings in non-compact
spaces. By a decomposition of the error into encoding, approximation and reconstruction errors, we prove both lower and upper bounds on the total error,
relating it to the spectral decay properties of the covariance operators, associated with the underlying measures. We derive almost optimal error bounds
with very general affine reconstructors and with random sensor locations as
well as bounds on the generalization error, using covering number arguments.
We illustrate our general framework with four prototypical examples of nonlinear operators, namely those arising in a nonlinear forced ODE, an elliptic
PDE with variable coefficients and nonlinear parabolic and hyperbolic PDEs.
While the approximation of _arbitrary_ Lipschitz operators by DeepONets to accuracy _ϵ_ is argued to suffer from a “curse of dimensionality” (requiring a neural
networks of exponential size in 1 _/ϵ_ ), in contrast, for all the above _concrete_ examples of interest, we rigorously prove that DeepONets can _break this curse_
_of dimensionality_ (achieving accuracy _ϵ_ with neural networks of size that can
grow algebraically in 1 _/ϵ_ ). Thus, we demonstrate the efficient approximation
of a potentially large class of operators with this machine learning framework.


_Date_ : January 14, 2022.
_E-Mail_ : samuel.lanthaler@math.ethz.ch .


1


2 ERROR ESTIMATES FOR DEEPONETS


1. Introduction


_Deep neural networks_ (Goodfellow et al. 2016) have been very successfully used
for a diverse range of regression and classification learning tasks in science and
engineering in recent years (LeCun et al. 2015). These include image and text
classification, computer vision, text and speech recognition, natural language processing, autonomous systems and robotics, game intelligence and protein folding
(Evans et al. 2018).
As deep neural networks are _universal approximators_, i.e., they can approximate
any continuous (even measurable) finite-dimensional function to arbitrary accuracy
(Barron 1993, Hornik et al. 1989, Cybenko 1989, Tianping Chen & Liu 1990), it is
natural to use them as ansatz spaces for the solutions of partial differential equations
(PDEs). They have been used for solving high-dimensional parabolic PDEs by
emulating explicit representations such as the Feynman-Kac formula as in E et al.
(2017), Han et al. (2018), Beck et al. (2021) and references therein, and as _physics_
_informed neural networks_ (PINNs) for solving both forward problems (Raissi &
Karniadakis 2018, Raissi et al. 2019, Mao et al. 2020, Mishra & Molinaro 2020,
2021 _b_ ), as well as inverse problems (Raissi et al. 2019, 2018, Mishra & Molinaro
2021 _a_, Lu et al. 2021) for a variety of linear and non-linear PDEs.
Deep neural networks are also being widely used in the context of _many query_
problems for PDEs, such as uncertainty quantification (UQ) (see e.g. O’LearyRoseberry et al. 2022, Zhu & Zabaras 2018, Lye et al. 2020), optimal control (design), deterministic and Bayesian inverse problems (Adler & Oktem 2017, Khoo & [¨]
Ying 2019) and PDE constrained optimization (Guo et al. 2016, Lye et al. 2021). In
such _many query_ problems, the inputs are functions such as the initial and boundary data, source terms and/or coefficients in the underlying differential operators.
The outputs are either the solution field (in space-time or at fixed time instances)
or possibly _observables_ (functionals of the solution field). Thus, the input to output
map is, in general, a (possibly) non-linear _operator_, mapping one function space to
another.

Currently, it is standard to approximate the underlying input function with a
finite, but possibly very high-dimensional, _parametric_ representation. Similarly, the
resulting output function is approximated by a finite dimensional representation,
for instance, values on a grid or coefficients of a suitable basis. Thus, the underlying
_operator, mapping infinite dimensional spaces,_ is approximated by a function that
maps a finite but high dimensional input spaces into another finite-dimensional
output space. Consequently, this finite-dimensional map for the resulting _paramet-_
_ric PDE_ can be _learned_ with _standard_ deep neural networks, as for elliptic and
parabolic PDEs in Schwab & Zech (2019), Opschoor et al. (2019, 2020), Kutyniok
et al. (2021), for transport PDEs (Laakmann & Petersen 2021) and for hyperbolic
and related PDEs (DeRyck & Mishra 2021, Lye et al. 2020, 2021, and references
therein).
However, this finite dimensional parametrization of the underlying infinite dimensional problem is subject to the inherent and non-vanishing error both at the
input end, due to the finite dimensional representation as well as at the output end,
on account of numerical errors at finite resolution. More fundamentally, a parametric representation requires explicit knowledge of the underlying measure on input
space such that a finite dimensional approximation of inputs can be performed.
Such explicit knowledge may not always be available. Finally, the parametric approach does not cover a large number of situations where the underlying physics, in
the form of governing PDEs, may not even be known explicitly, yet large amounts
of (possibly noisy) data for the input-output mapping is available. It is not obvious
how such a learning task can be performed with standard neural networks.


ERROR ESTIMATES FOR DEEPONETS 3


Hence, _operator learning_, i.e. learning nonlinear operators mapping one infinitedimensional Banach space to another, from data, is increasingly being investigated
in the contexts of PDEs and possibly other fields. One research direction has
focused largely on operators which can be expressed as solution operators to a suitable PDE/ODE; examples of this approach include the identification of individual
terms of the underlying differential equation from data, expressed in terms of nonlocal integral operators (Patel et al. 2021, You et al. 2021, and references therein),
the identification of suitable closure models for turbulent flows (Duraisamy et al.
2019, Ahmed et al. 2021, and references therein) or the discovery of the governing equations of an underlying dynamical system, expressed in terms of an ODE
(or PDE), (Brunton et al. 2016, and references therein). In a different research
direction, the aim is to use deep neural networks to directly learn the _underlying_
_(solution–)operator_, itself. Several frameworks have been proposed for this task; we
refer to Li, Kovachki, Azizzadenesheli, Liu, Stuart, Bhattacharya & Anandkumar
(2020) and Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar (2020) for graph kernel operators, Bhattacharya et al. (2021) for a recent
approach based on principal component analysis, and Li et al. (2021) and references therein on _Fourier neural operators_ . A different approach was proposed by
Chen & Chen (1995), where they presented a neural network architecture, termed
as _operator nets_, to approximate a non-linear operator _G_ : _K →_ _K_ _[′]_, where _K, K_ _[′]_

are compact subsets of infinite dimensional Banach spaces, _K ⊂_ _C_ ( _D_ ), _K_ _[′]_ _⊂_ _C_ ( _U_ )
with _D, U_ compact domains in R _[d]_, R _[n]_, respectively. Then, an operator net can be
formulated in terms of two shallow, i.e., one hidden layer, neural networks. The
first is the so-called _branch net_ _**β**_ ( _u_ ) = ( _β_ 1 ( _u_ ) _, . . ., β_ _p_ ( _u_ )), defined for 1 _≤_ _k ≤_ _p_ as,



� _ξ_ _ki_ _[j]_ _[u]_ [(] _[x]_ _[j]_ [) +] _[ θ]_ _k_ _[i]_

_j_ =1







_m_
�
 _j_ =1



 _._ (1.1)





_β_ _k_ ( _u_ ) =



_ℓ_
� _c_ _[i]_ _k_ _[σ]_


_i_ =1



Here, _{x_ _j_ _}_ 1 _≤j≤m_ _⊂_ _D_, are the so-called _sensors_ and _c_ _[i]_ _k_ _[, ξ]_ _ki_ _[j]_ [are weights and] _[ θ]_ _k_ _[i]_ [are]
biases of the neural network.
The second neural network is the so-called _trunk net_ _**τ**_ ( _y_ ) = ( _τ_ 1 ( _y_ ) _, . . ., τ_ _p_ ( _y_ )),
defined as,


_τ_ _k_ ( _y_ ) = _σ_ ( _w_ _k_ _· y_ + _ζ_ _k_ ) _,_ 1 _≤_ _k ≤_ _p,_ (1.2)


for any _y ∈_ _U_ and with weights _w_ _k_ and biases _ζ_ _k_ . Here, _σ_ is a non-linear activation
function in the branch net (1.1) and (a possibly different one) in the trunk net (1.2).
The branch and trunk nets are then combined to approximate the underlying nonlinear operator in the _operator net_



_G_ ( _u_ )( _y_ ) _≈_



_p_
� _β_ _k_ ( _u_ ) _τ_ _k_ ( _y_ ) _,_ _u ∈_ _K, y ∈_ _U._ (1.3)


_k_ =1



More recently, Lu et al. (2019) replace the shallow branch and trunk nets in the
operator net (1.3) with deep neural networks to propose _deep operator nets_ ( _Deep-_
_ONets_ in short), which are expected to be more expressive than shallow operator
nets and have already been successfully applied to a variety of problems with differential equations. These include learning linear and non-linear dynamical systems
and reaction-diffusion PDEs with source terms (Lu et al. 2019), learning the PDEs
governing electro-convection (Mao et al. 2021), Navier-Stokes equations in hypersonics with chemistry (Cai et al. 2021) and the dynamics of bubble growth (Lin
et al. 2021), among others. A simple example that illustrates DeepONets (1.3) and
their ability to learn an operator efficiently is included in Appendix D (cf. Figure
2).


4 ERROR ESTIMATES FOR DEEPONETS


Why are DeepONets able to approximate operators mapping infinite dimensional
spaces, efficiently? A first answer to this question lies in a remarkable _universal_
_approximation theorem_ for the operator network (1.3) first proved by Chen & Chen
(1995), and extended to DeepONets by Lu et al. (2019), where it is shown that
as long as the underlying operator _G_ is _continuous_ and maps a _compact subset_ of
the infinite-dimensional space into another Banach space, there always exists an
operator network of the form (1.3), that approximates _G_ to arbitrary precision,
i.e. to any given error tolerance. However, the assumptions on continuity and in
particular, compactness of the input space, in the universal approximation theorem
do not cover most examples of practical interest, such as many of the operators
considered by Lu et al. (2019). Moreover, this universal approximation property
does not provide any explicit information on the computational complexity of the
operator network, i.e. no explicit knowledge of the number of sensors _m_, number
of branch and trunk nets _p_ and the sizes (number of weights and biases) as well as
depths (for DeepONets) of these neural networks can be inferred from the universal
approximation property.
Given the infinite dimensional setting, it could easily happen that the computational complexity of the DeepONet for attaining a given tolerance _ϵ_ scales _expo-_
_nentially_ in 1 _/ϵ_ . In fact, we will provide a heuristic argument strongly suggesting
that such exponential scaling can not be overcome in the approximation of _general_
Lipschitz continuous operators (cp. Remark 3.4; and Thm. 2.2 of Mhaskar & Hahm
(1997) for related work on rigorous lower bounds). This (worst-case) scaling will be
referred to as the _curse of dimensionality_ and can severely inhibit the efficiency of
DeepONets at realistic learning tasks. Although numerical experiments presented
in Lu et al. (2019), Cai et al. (2021), Mao et al. (2021), Lin et al. (2021) strongly
indicate that DeepONets may not suffer from this curse of dimensionality for many
cases of interest, no rigorous results to this end are available currently. Moreover,
no rigorous results on the DeepONet _generalization error_, i.e., the error due to finite
sampling of the input space, are available currently.
The above considerations motivate our current paper where we seek to provide
rigorous and explicit bounds on the error incurred by DeepONets in approximating
nonlinear operators on Banach spaces. As a first step, we extend the universal
approximation theorem from continuous to _measurable_ operators, while removing
the compactness requirements of Chen & Chen (1995). Next, using a very natural
decomposition of DeepONets (cf Figure 1) into an _encoder_ that maps the infinitedimensional input space into a finite-dimensional space, an _approximator_ neural
network that maps one finite-dimensional space into another and a trunk net induced affine _reconstructor_, that maps a finite dimensional space into the infinite
dimensional output space, we decompose the total DeepONet approximation error
in terms of the resulting _encoding, approximation and reconstruction errors_ and
estimate each part separately. This allows us to derive rigorous _upper_ as well as
_lower_ bounds on the DeepONet error, under very general hypotheses on the underlying nonlinear operator and underlying measures on input Hilbert spaces. In
particular, optimal bounds on the encoding and reconstruction errors stem from
a careful analysis of the eigensystem for the covariance operators, associated with
the underlying input measure.
A similar error decomposition has been employed in Bhattacharya et al. (2021),
to analyze an operator learning architecture combining principal components analysis (PCA) autoencoders for the encoding and reconstruction with a neural network
for the non-linear approximation step. In particular, the authors derive a quantitative error estimate for the _empirical_ PCA autoencoder, which is based on a _finite_


ERROR ESTIMATES FOR DEEPONETS 5


_number_ of input/output samples ( _u_ _j_ _, G_ ( _u_ _j_ )), _j_ = 1 _, . . ., N_ . However, we need significant additional efforts to translate ”PCA” based ideas into quantitative error
and complexity estimates for the point-evaluation encoder and the neural network
reconstruction of DeepONets (even in the limit of infinite data). Another key distinction of the present work with Bhattacharya et al. (2021) is a detailed discussion
of the _efficiency_ of the DeepONet approximation, providing quantitative error and
complexity bounds not only for the encoding and reconstruction steps, but also for
the approximator network.
In addition to our analysis of the encoding, approximation and reconstruction
errors, we illustrate these abstract error estimates with four prototypical differential
equations, namely a nonlinear ODE with a forcing term, a linear elliptic PDE with
variable diffusion coefficients, a semi-linear parabolic PDE (Allen-Cahn equation)
and a quasi-linear hyperbolic PDE (scalar conservation law), thus covering a wide
spectrum of differential equations with different types of inputs and different levels
of Sobolev regularity of the resulting solutions. For each of these four problems, we
rigorously prove that the underlying operators possess additional structure, through
which DeepONets can achieve an approximation accuracy _ϵ_ with a size that scales
only algebraically in 1 _/ϵ_, i.e. _DeepONets can break the curse of dimensionality_, associated with the approximation of the infinite-dimensional input-to-output map.
Thus providing the first rigorous proofs of their possible efficiency at operator approximation. An appropriate notion of the curse of dimensionality in the DeepONet
context will be given in Definition 3.5 (cp. also Remark 3.4). Finally, we also provide a rigorous bound for the _generalization error_ of DeepONets and show that,
despite the underlying infinite dimensional setting, the estimate on generalization
error scales (asymptotically) as 1 _/√N_ with _N_ being the number of training samples

(up to log terms), which is consistent with the standard finite dimensional bound
with statistical learning theory techniques.
The rest of the paper is organized as follows: In section 2, we formulate the
underlying operator learning problem and introduce DeepONets. The abstract error estimates are presented in section 3 and are illustrated on four concrete model
problems in section 4. Sections 3 and 4 focus on quantitative bounds for the _best-_
_approximation error_ that is achievable, in principle, by the given DeepONet architecture; additional (generalization) errors due to the availability of only a finite
number of training samples, are discussed in section 5, where estimates on the
DeepONet generalization error are derived. The proofs of our theoretical results
are presented in the appendix. Other error sources, e.g. due to (imperfect) training
algorithms such as stochastic gradient descent, errors due to uncertain and noisy
data, or errors due to a mismatch between the training and evaluation data, will
not be discussed in the present work. The analysis of such errors represent avenues
for extensive future work. Moreover, for simplicity of the exposition, the results of
the present work are formulated for neural networks with _ReLU activation function_ ;
this particular choice of activation function is, however, not essential to reach the
main conclusions.


2. Deep Operator Networks


Our main aim in this section is to follow Lu et al. (2019) and introduce DeepONets, i.e., deep version of the shallow operator network (1.3) for approximating
operators. To this end, we start with a brief recapitulation of what a neural network
is.


2.1. **Neural Networks.** Let R _[d]_ _[in]_ and R _[d]_ _[out]_ denote the input and output spaces,
respectively. Given any input vector _z ∈_ R _[d]_ _[in]_, a feedforward neural network (also
termed as a multi-layer perceptron), transforms it to an output through layers of


6 ERROR ESTIMATES FOR DEEPONETS


units (neurons) consisting of either affine-linear maps between units (in successive layers) or scalar non-linear activation functions within units Goodfellow et al.
(2016), resulting in the representation,


L _θ_ ( _y_ ) = _C_ _K_ _◦_ _σ ◦_ _C_ _K−_ 1 _. . . . . . . . . ◦_ _σ ◦_ _C_ 2 _◦_ _σ ◦_ _C_ 1 ( _y_ ) _._ (2.1)


Here, _◦_ refers to the composition of functions and _σ_ is a scalar (non-linear) activation function. A large variety of activation functions have been considered in the
machine learning literature Goodfellow et al. (2016), including adaptive activation
functions in Jagtap et al. (2020). Popular choices for the activation function _σ_
in (2.1) include the sigmoid function, the tanh function and the _ReLU_ function
defined by,
_σ_ ( _z_ ) = max( _z,_ 0) _._ (2.2)

In the present work, we will only consider neural networks with ReLU activation
function, i.e., the term “neural network” should be understood synonymous with
“ReLU neural network”.

For any 1 _≤_ _k ≤_ _K_, we define

_C_ _k_ _z_ _k_ = _W_ _k_ _z_ _k_ + _b_ _k_ _,_ for _W_ _k_ _∈_ R _[d]_ _[k]_ [+1] _[×][d]_ _[k]_ _, z_ _k_ _∈_ R _[d]_ _[k]_ _, b_ _k_ _∈_ R _[d]_ _[k]_ [+1] _._ (2.3)


For consistency of notation, we set _d_ 1 = _d_ _in_ and _d_ _K_ +1 = _d_ _out_ .
Thus in the terminology of machine learning, the neural network (2.1) consists
of an input layer, an output layer and ( _K −_ 1) hidden layers for some 1 _< K ∈_ N.
The _k_ -th hidden layer (with _d_ _k_ +1 neurons) is given an input vector _z_ _k_ _∈_ R _[d]_ _[k]_
and transforms it first by an affine linear map _C_ _k_ (2.3) and then by a nonlinear
(component wise) activation _σ_ . A straightforward addition shows that our network



_K_
contains _d_ _in_ + _d_ _out_ + � _d_ _k_
� _k_ =2



neurons. We also denote,
�


_θ_ = _{W_ _k_ _, b_ _k_ _},_ (2.4)



to be the concatenated set of (tunable) weights and biases for our network. It is
straightforward to check that _θ ∈_ Θ _⊂_ R _[M]_ with



_M_ =



_K_
�( _d_ _k_ + 1) _d_ _k_ +1 _._ (2.5)


_k_ =1



We also introduce the following nomenclature for a deep neural network L,


size(L _θ_ ) := _∥θ∥_ _ℓ_ 0 _,_ depth(L _θ_ ) = _K −_ 1 _,_ (2.6)


with _∥θ∥_ _ℓ_ 0 = # _{θ_ _k_ _̸_ = 0 _}_ denoting the total number of _non-zero_ tuning parameters
(weights and biases) of the neural network and _K −_ 1 being the number of hidden layers of the network. Henceforth, the explicit _θ_ -dependence is suppressed for
notational convenience and we denote the neural network (2.1) as L.


2.2. **DeepONets.** A DeepONet, as proposed in Lu et al. (2019) is a deep neural
network extension of the operator network (1.3). Roughly speaking, the shallow
branch and trunk nets in (1.1) and (1.2) are replaced by deep neural networks of the
form (2.1). However, we present a slightly more general form of DeepONets in this
paper, as compared to the DeepONets of Lu et al. (2019). To this end, we recall
that _D ⊂_ R _[d]_ and _U ⊂_ R _[n]_ are compact domains (e.g. with Lipschitz boundary)
and introduce the following _operators_ (cp. Figure 1):

_•_ **Encoder.** Given a set of _sensor_ points _x_ _j_ _∈_ _D_, for 1 _≤_ _j ≤_ _m_, we define
the linear mapping,


_E_ : _C_ ( _D_ ) _→_ R _[m]_ _,_ _E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) _,_ (2.7)


as the _encoder_ mapping. Note that the encoder _E_ is well-defined as one can
evaluate continuous functions pointwise.


ERROR ESTIMATES FOR DEEPONETS 7


Figure 1. Schematic illustration of the decomposition of a DeepONet into the encoder _E_, approximator _A_ and reconstructor _R_ .


_•_ **Approximator.** Given the above sensor points _{x_ _j_ _}_ for 1 _≤_ _j ≤_ _m_, the
_approximator_ is a deep neural network of the form (2.1) and defined as,


_A_ : R _[m]_ _→_ R _[p]_ _, {u_ _j_ _}_ _[m]_ _j_ =1 _[�→{A]_ _[k]_ _[}]_ _[p]_ _k_ =1 _[,]_ (2.8)


Note that _d_ _in_ = _m_ and _d_ _out_ = _p_ in the approximator neural network _A_ of
form (2.1). Given the encoder and approximator, we define the _branch net_
_**β**_ : _C_ ( _D_ ) _→_ R _[p]_ as the composition _**β**_ ( _u_ ) = _A ◦E_ ( _u_ ).

_•_ **Reconstructor.** First, we denote a _trunk net_ _**τ**_ as a neural network

_**τ**_ : R _[n]_ _→_ R _[p]_ [+1] _, y_ = ( _y_ 1 _, . . ., y_ _n_ ) _�→{τ_ _k_ ( _y_ ) _}_ _[p]_ _k_ =0 _[,]_


with each _τ_ _k_ of the form (2.1), with _d_ _in_ = _n_ and _d_ _out_ = 1 and for any
_y ∈_ _U ⊂_ R _[n]_ .
Then, we define a _**τ**_ -induced _reconstructor_ as



_R_ = _R_ _**τ**_ : R _[p]_ _→_ _C_ ( _U_ ) _,_ _R_ _**τ**_ ( _α_ _k_ ) := _τ_ 0 ( _y_ ) +



_p_
� _α_ _k_ _τ_ _k_ ( _y_ ) _._ (2.9)


_k_ =1



Henceforth for notational convenience, we will suppress the _**τ**_ -dependence
of the _**τ**_ -induced reconstructor and simply label it as _R_ . Note that the reconstructor is well-defined as the activation function _σ_ in (2.1) is at least
continuous.

Given the above ingredients, we combine them into a DeepONet as,


_N_ : _C_ ( _D_ ) _→_ _C_ ( _U_ ) _,_ _N_ ( _u_ ) = ( _R ◦A ◦E_ )( _u_ ) _._ (2.10)


I.e. a DeepONet is composed of three components:

(1) Encoding: The encoder mapping _E_ : _C_ ( _D_ ) _→_ R _[m]_, _u �→{u_ ( _x_ _j_ ) _}_ _[m]_ _j_ =1 [,]
(2) Approximation: The encoded (finite-dimensional) data is approximated by
a neural network mapping _A_ : R _[m]_ _→_ R _[p]_,
(3) Reconstruction: The result is decoded by _R_ : R _[p]_ _→_ _C_ ( _U_ ), _{A_ _k_ _}_ _[p]_ _k_ =1 _[�→]_
_τ_ 0 + [�] _[p]_ _k_ =1 _[A]_ _[k]_ _[τ]_ _[k]_ [, with] _**[ τ]**_ [ being the trunk net.]
A graphical depiction of the constituent parts of a DeepONet is shown in figure
1. The only difference between our version of the DeepONet (2.10) and the version
presented in the recent paper Lu et al. (2019), lies in the fact that we use a more
general affine reconstruction step. In other words, setting _τ_ 0 ( _y_ ) _≡_ _τ_ 0, for some
_τ_ 0 _∈_ R in (2.9) recovers the DeepONet of Lu et al. (2019). We remark in passing
that although the above formulation assumes a mapping between (scalar) functions,
_N_ : _C_ ( _D_ ) _→_ _C_ ( _U_ ), all results in this work extend trivially to the more general case
of DeepONet approximations for _systems N_ : _C_ ( _D_ ; R _[d]_ _[u]_ ) _→_ _C_ ( _U_ ; R _[d]_ _[v]_ ). For clarity
of the exposition and simplicity of notation, we will focus on the case _d_ _u_ = _d_ _v_ = 1,
in the following.


8 ERROR ESTIMATES FOR DEEPONETS


We recall that the DeepONet (2.10) contains parameters corresponding to the
weights and biases of the approximator neural network _A_ and the trunk net _**τ**_,
that need to be tuned ( _trained_ ) such that the DeepONet (2.10) approximates the
underlying operator _G_ : _X →_ _Y_ . To this end, we need to define a distance between
_G_ and the DeepONet _N_ . A natural way to do this, is to fix a probability measure
_µ ∈P_ ( _X_ ), and to consider the following error, measured in the _L_ [2] ( _µ_ )- _norm_ :



_|G_ ( _u_ )( _y_ ) _−N_ ( _u_ )( _y_ ) _|_ [2] _dy dµ_ ( _u_ )



1 _/_ 2









�
_E_ =







ˆ
 _X_



_,_ (2.11)



_X_



ˆ

_U_



with _N_ being the DeepONet (2.10). Note that we have replaced the function spaces
_C_ ( _D_ ) and _C_ ( _U_ ) by more general function spaces _X_ and _Y_, for which we will assume
that there exists an embedding _X�→L_ [2] ( _D_ ), _Y �→L_ [2] ( _U_ ). In particular, for the error
(2.11) to be well-defined, it suffices that

_•_ there exists a Borel set _A ⊂_ _X_, such that _µ_ ( _A_ ) = 1, and _A ⊂_ _C_ ( _D_ ) so that
_E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) is well-defined on _A_,

_•_ The mapping
_G_ : _X →_ _Y,_
maps given data _u ∈_ _X_ to a _L_ [2] ( _U_ ) function _v_ ( _y_ ) = _G_ ( _u_ )( _y_ ) defined on _U_,
and _G ∈_ _L_ [2] ( _µ_ ) := _L_ [2] ( _µ_ ; _∥· ∥_ _L_ 2 ( _U_ ) ), in the sense that


_∥G_ ( _u_ ) _∥_ [2] _L_ [2] ( _U_ ) _[dµ]_ [(] _[u]_ [)] _[ <][ ∞][.]_

ˆ _X_


We formalize these concepts with the following definition:



_**Definition**_ **2.1** (Data for DeepONet approximation) _**.**_ Let _D ⊂_ R _[d]_, _U ⊂_ R _[n]_ be
bounded domains. Let _X_, _Y_ be separable Banach spaces with a continuous embedding _ι_ : _X�→L_ [2] ( _D_ ) and ~~_ι_~~ : _Y �→L_ [2] ( _U_ ). We call _µ_, _G_ **data for the DeepONet**
**approximation problem**, provided _µ ∈P_ 2 ( _X_ ) is a Borel probability measure on
_X_, there exists a Borel set _A ⊂_ _X_, such that _µ_ ( _A_ ) = 1 and _A_ consists of continuous
functions, and _G_ : _X →_ _Y_ is a Borel measurable mapping, such that _G ∈_ _L_ [2] ( _µ_ ), i.e.
´ _X_ _[∥G]_ [(] _[u]_ [)] _[∥]_ _L_ [2] [2] _U_ _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [. Here,] _[ P]_ [2] [(] _[X]_ [) is the set of probability measures with]



´ _X_ _[∥G]_ [(] _[u]_ [)] _[∥]_ _L_ [2] [2] ( _U_ ) _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [. Here,] _[ P]_ [2] [(] _[X]_ [) is the set of probability measures with]

finite second moments ´ _X_ _[∥][u][∥]_ _X_ [2] _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [.]



_X_ _[∥][u][∥]_ _X_ [2] _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [.]



_**Remark**_ **2.2** _**.**_ The setting considered in Definition 2.1 corresponds to a “perfect
data setting”, in which the input/output pairs ( _u, G_ ( _u_ )), _u ∼_ _µ_ are provided exactly, i.e. in the absence of measurement noise and uncertainty. Furthermore, the
main focus of this work will be on the problem of finding complexity bounds on
the best-approximation provided by the DeepONet architecture (2.7)–(2.9). Concerning the _finite data setting_, i.e. when only a finite number of input/output pairs
( _u_ 1 _, G_ ( _u_ 1 )) _, . . .,_ ( _u_ _N_ _, G_ ( _u_ _N_ )), _u_ _j_ _∼_ _µ_ are available, first results on the generalization
error will be presented in Section 5.


In the framework of nonlinear operators that arise in differential equations, the
Banach spaces _X_ and _Y_ will be function spaces on _D_ and _U_, respectively; a typical
example is _X_, _Y_ = _H_ _[s]_ ( _D_ ) for some _s ≥_ 0, where _H_ _[s]_ ( _D_ ) denotes the _L_ [2] -based
Sobolev space on _D_ . The embeddings _ι_ : _X�→L_ [2] ( _D_ ), ~~_ι_~~ : _Y �→L_ [2] ( _U_ ) will thus be
canonical, and henceforth, we will identify _X ≃_ _ι_ ( _X_ ), _Y ≃_ ~~_ι_~~ ~~(~~ _Y_ ) as subsets of _L_ [2] ( _D_ )
and _L_ [2] ( _U_ ), respectively.


_**Remark**_ **2.3** _**.**_ A technical difficulty associated with DeepONets arises due to the
specific form of the encoder (2.7), which is defined via point-wise evaluations. In
principle, the components of this encoder could easily be replaced by more general


ERROR ESTIMATES FOR DEEPONETS 9


functionals of _u_, and in fact, this might be more natural in certain settings (e.g. to
model physical measurements; or for mathematical reasons, see Section 4.4). For
our general discussion, we will focus instead on encoders of the particular form
(2.7). The main reason for this choice is the possibility for direct comparison with
the numerical experiments of Lu et al. (2019), which are based on the DeepONet
architecture (2.7)–(2.9). Furthermore, fixing a particular choice will allow us to
analyse the encoding error associated with _E_ in great detail in Section 3.5 (see
Section 3.2.3 for an overview).



Since the point-wise encoder _E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) is not well-defined on
spaces such as _L_ [2] ( _D_ ), we first show that (2.11) is nevertheless well-defined. From
Lemma B.1, we infer that if _A ⊂_ _X_ is a Borel measurable set such that _A ⊂_ _C_ ( _D_ ),
then _N_ = _R ◦A ◦E_ : _A →_ _L_ [2] ( _U_ ) is measurable, and possesses a measurable
extension _R ◦A ◦E_ : _L_ [2] ( _D_ ) _→_ _L_ [2] ( _U_ ). Clearly, since _µ_ ( _A_ ) = 1, we then have


2
�� _G_ ( _u_ )( _y_ ) _−_ ( _R ◦A ◦E_ )( _u_ )( _y_ )�� _dy dµ_ ( _u_ )

ˆ _X_ ˆ _U_



2
�� _G_ ( _u_ )( _y_ ) _−_ ( _R ◦A ◦E_ )( _u_ )( _y_ )�� _dy dµ_ ( _u_ )



_X_



ˆ



_U_



=
ˆ _A_



_|G_ ( _u_ )( _y_ ) _−_ ( _R ◦A ◦E_ )( _u_ )( _y_ ) _|_ [2] _dy dµ_ ( _u_ ) _,_

ˆ _U_



for any extension _E_ . This allows us to define the error (2.11) uniquely and allows
to formulate the following precise definition of DeepONets,


_**Definition**_ **2.4** (DeepONet) _**.**_ Let _µ_, _G_ be given data for the DeepONet approximation problem (see Definition 2.1). A **DeepONet** _N_, approximating the nonlinear operator _G_, is a mapping _N_ : _C_ ( _D_ ) _→_ _L_ [2] ( _U_ ) of the form _N_ = _R ◦_
_A ◦E_, where _E_ : ( _X, ∥· ∥_ _X_ ) _→_ (R _[m]_ _, ∥· ∥_ _ℓ_ 2 ) denotes the encoder given by (2.7),
_A_ : (R _[m]_ _, ∥· ∥_ _ℓ_ 2 ) _→_ (R _[p]_ _, ∥· ∥_ _ℓ_ 2 ) denotes the approximator network (2.8), and _R_ :
(R _[p]_ _, ∥· ∥_ _ℓ_ 2 ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ( _U_ ) ) denotes the reconstruction of the form (2.9), induced by the trunk net _**τ**_ .


3. Error bounds for DeepONets


Our aim in this section is to derive bounds on the error (2.11) incurred by the
DeepONet (2.10) in approximating the underlying nonlinear operator _G_ .


3.1. **A universal approximation theorem.** As a first step in showing that the
DeepONet error (2.11) can be small, we have the following _universal approximation_
_theorem_, that generalizes the universal approximation property of Chen & Chen
(1995) to significantly more general nonlinear operators,


_**Theorem**_ **3.1** _**.**_ Let _µ ∈P_ ( _C_ ( _D_ )) be a probability measure on _C_ ( _D_ ). Let _G_ :
_C_ ( _D_ ) _→_ _L_ [2] ( _D_ ) be a Borel measurable mapping, with _G ∈_ _L_ [2] ( _µ_ ), then for every
_ϵ >_ 0, there exists an operator network _N_ = _R ◦A ◦E_, such that


1 _/_ 2

_∥G −N∥_ _L_ 2 ( _µ_ ) = _∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ [2] _L_ [2] ( _U_ ) _[dµ]_ [(] _[u]_ [)] _< ϵ._
�ˆ _X_ �


The proof of this theorem is based on an application of Lusin’s Theorem to
approximate measurable maps by continuous maps on compact subsets and then
using the universal approximation theorem of Chen & Chen (1995). It is presented
in detail in Appendix C.1.


_**Remark**_ **3.2** _**.**_ The universal approximation theorem of Chen & Chen (1995) states
that DeepONets can approximate continuous operators _G_ uniformly over compact
subsets _K ⊂_ _X_ . In contrast, the above theorem removes both the compactness


10 ERROR ESTIMATES FOR DEEPONETS


constraint, as well as the continuity assumption on _G_ and paves the way for the
theorem to be applied in realistic settings, for instance in the approximation of
nonlinear operators that arise when considering differential equations such as those
of Lu et al. (2019) and later in this paper. However, this extension comes at the
expense of considering a weaker distance ( _L_ [2] ( _µ_ _⊗_ _dy_ ) vs. _L_ _[∞]_ ( _µ_ _⊗_ _dy_ )) than in Chen
& Chen (1995). In practice, it is indeed the _L_ [2] -distance that is minimized during
the training process. Moreover, the above theorem also allows us to consider cases
of practical interest where _µ_ is supported on an _unbounded_ subset, as is e.g. the case
when _µ_ is a non-degenerate Gaussian measure. Indeed, in most of the numerical
examples in Lu et al. (2019), the underlying measure _µ_ is a Gaussian measure given
by the law of a Gaussian random field.


The universal approximation theorem 3.1 shows that for any given tolerance _ϵ_,
there exists a DeepONet of the form (2.10) such that the resulting approximation
error (2.11) is smaller than this tolerance. However, this theorem does not provide
any explicit information about the number of sensors _m_, the number of branch and
trunk net outputs _p_ or the hyperparameters of the approximator neural network _A_
and the trunk net _**τ**_ . As discussed in the introduction, these numbers specify the
complexity of a DeepONet and we would like to obtain explicit bounds (information) on the computational complexity of a DeepONet for achieving a given error
tolerance and ascertain whether DeepONets are efficient at approximating a given
nonlinear operator _G_ . In practice, we are thus interested in deriving _quantitative_
error and complexity bounds for the DeepONet approximation of operators. This
will be the focus of the remainder of the present section.


3.2. **Overview of quantitative error bounds.** We will first provide an overview
of the main results on quantitative error bounds derived in the present work. An
extended discussion of these results can be found in the following subsections, which
include detailed derivations and proofs.


3.2.1. _Error decomposition and the curse of dimensionality._ Given the decomposition of the DeepONet (2.10) into an encoder _E_, approximator _A_ and reconstructor
_R_, it is natural to expect that the total error (2.11) also decomposes into errors
associated with them. For a given encoder _E_ and reconstructor _R_, we can define
(approximate) inverses _D_ (the _decoder_ ) and _P_ (the _projector_ ), which are required
to satisfy the following relations exactly


_E ◦D_ = Id : R _[m]_ _→_ R _[m]_ _,_ _P ◦R_ = Id : R _[p]_ _→_ R _[p]_ _,_


and should satisfy


_D ◦E ≈_ Id : _X →_ _X,_ _R ◦P ≈_ Id : _Y →_ _Y ._


We note that _D_ and _P_ are not necessarily unique, and need to be chosen. All
mappings are illustrated in the following diagram:


_G_

_L_ [2] ( _D_ ) _L_ [2] ( _U_ )



_E_ _D_ _P_



_R_



_A_

R _[m]_ R _[p]_


Given choices for the decoder _D_ and the projector _P_, we can now define the **en-**
**coding error** _E_ [�] _E_, the **approximation error** _E_ [�] _A_, and the **reconstruction error**


ERROR ESTIMATES FOR DEEPONETS 11


�
_E_ _R_, respectively, as follows:



2

_∥D ◦E_ ( _u_ ) _−_ _u∥_ _X_ [2] _[dµ]_ [(] _[u]_ [)]
_X_ � [1]



�
_E_ _E_ :=
�ˆ


�
_E_ _A_ :=
�ˆ



2
_,_ (3.1)



2

_ℓ_ [2] (R _[p]_ ) _[d]_ [(] _[E]_ [#] _[µ]_ [)(] _**[u]**_ [)]
R _[m]_ _[ ∥A]_ [(] _**[u]**_ [)] _[ −P ◦G ◦D]_ [(] _**[u]**_ [)] _[∥]_ [2] � [1]



2
(3.2)



� 2 [1]



�
_E_ _R_ :=



ˆ
�



_∥R ◦P_ ( _u_ ) _−_ _u∥_ [2] _L_ [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[u]_ [)]
_L_ [2] ( _U_ )



2

_._ (3.3)



We could also have written these errors as _E_ [�] _E_ = _∥D ◦E −_ Id _∥_ _L_ 2 ( _µ_ ), _E_ [�] _A_ = _∥A −_

_P ◦G ◦D∥_ _L_ 2 ( _E_ # _µ_ ), and _E_ [�] _R_ = _∥R ◦P −_ Id _∥_ _L_ 2 ( _G_ # _µ_ ) . Intuitively, the encoding and

reconstruction errors, _E_ [�] _E_ and _E_ [�] _R_, measure the loss of information by the DeepONet’s finite-dimensional encoding of the underlying infinite-dimensional spaces;
these error sources are weighted by the input measure _µ_ on the input side, and the
push-forward measure _G_ # _µ_ on the output side, respectively. The approximation error _E_ [�] _A_ measures the error due to the approximation by the approximator network
_A_ : R _[m]_ _→_ R _[p]_ of the “encoded/projected” operator, _G_ = _P ◦G ◦D_ : R _[m]_ _→_ R _[p]_ .
To state our main result on the error decomposition of _E_ [�] in terms of _E_ [�] _E_, _E_ [�] _A_
and _E_ [�] _R_, we first recall the following notation for a mapping _F_ : _X →_ _Y_ between
arbitrary Banach spaces _X, Y_ :



Lip _α_ ( _F_ : _X →_ _Y_ ) := sup
_u,u_ _[′]_ _∈X_



_∥F_ ( _u_ ) _−F_ ( _u_ _[′]_ ) _∥_ _Y_

_,_ Lip( _F_ ) := Lip 1 ( _F_ ) _._
_∥u −_ _u_ _[′]_ _∥_ _[α]_ _X_



We then have the following error estimate, whose proof can be found in Appendix
C.2:


_**Theorem**_ **3.3** _**.**_ Consider the setting of Definition 2.1. Let the nonlinear operator
_G_ : _A ⊂_ _X →_ _Y_ be _α_ -H¨older continuous (or Lipschitz continuous if _α_ = 1),
where _X�→L_ [2] ( _D_ ), _Y �→L_ [2] ( _U_ ). Choose an arbitrary encoder _E_ : _C_ ( _D_ ) _→_ R _[m]_,
approximator _A_ : R _[m]_ _→_ R _[p]_ and an arbitrary reconstruction _R_ : R _[p]_ _→_ _L_ [2] ( _U_ ), of
the form (2.9). Then the error (2.11) associated with the DeepONet _N_ = _R ◦A ◦E_
satisfies the following upper bound,
_E_ � _≤_ Lip _α_ ( _G_ )Lip( _R ◦P_ )( _E_ � _E_ ) _[α]_ + Lip( _R_ ) _E_ � _A_ + � _E_ _R_ _._ (3.4)


� The last theorem shows that the total error _E_ [�] is indeed controlled by _E_ [�] _E_, _E_ [�] _A_ and
_E_ _R_ . Furthermore, this theorem provides us with a clear strategy for estimating the
DeepONet error (2.11) for a concrete operator _G_ of interest:


_•_
First, we will bound the encoding/reconstruction errors, providing suitable
estimates for

_D ◦E ≈_ Id _,_ _R ◦P ≈_ Id _._

In this step, we need to choose _D_, _E_, _R_, _P_ in order to minimize the resulting
encoding and reconstruction errors.

_•_ In a second step, we estimate the approximation error


_∥A −P ◦G ◦D∥_ _L_ 2 ( _E_ # _µ_ ) _,_


for fixed projector _P_ and decoder _D_ . The second step boils down to the
conventional approximation of a function _G_ : R _[m]_ _→_ R _[p]_ by neural networks.
Our goal will be to analyze the total error _E_ [�] in terms of this decomposition, with
the aim of showing the _efficiency_ of the DeepONet approximation for a wide range
of operators of interest. To this end, we will first need to discuss a suitable notion
of “efficiency”, which is motivated by the following remark.


12 ERROR ESTIMATES FOR DEEPONETS


_**Remark**_ **3.4** _**.**_ As shown above, the error introduced by the approximation step in
the DeepONet decomposition is naturally related to the error in the approximation
of a high-dimensional mapping _G_ = _P ◦G ◦D_ : R _[m]_ _→_ R _[p]_ by the neural network
_A_ : R _[m]_ _→_ R _[p]_ . The relevant function _G_ can be thought of as a finite-dimensional
projection of the operator _G_ . In particular, _G_ inherits regularity properties of
_G_ such as Lipschitz continuity. As shown in (Yarotsky 2018, Theorem 1), the
approximation of a _general_ Lipschitz continuous function to accuracy _∼_ _ϵ_, requires
a ReLU network of size ≳ _ϵ_ _[−][m/]_ [2], and hence suffers from the _curse of dimensionality_
in high dimensions, _m ≫_ 1. In the context of DeepONets, we recall that _m_ is the
number of sensors used in the encoding step _u �→E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) of
the DeepONet architecture. Achieving a small error of order _∼_ _ϵ_ in this encoding
step requires _m_ = _m_ ( _ϵ_ ) to depend on _ϵ_, with _m_ ( _ϵ_ ) _→∞_ as _ϵ →_ 0. Therefore,
general neural network approximation results indicate that the required DeepONet
complexity for the approximation _N ≈G_ of an _arbitrary_ Lipschitz continuous
operator _G_ to a given accuracy _ϵ_ requires at least size( _N_ ) ≳ _ϵ_ _[−][m]_ [(] _[ϵ]_ [)] _[/]_ [2] . In particular,
this scaling is faster than _any algebraic rate_ in _ϵ_ _[−]_ [1] . This connection between the
curse of dimensionality, as commonly understood in the literature (Chkifa et al.
2015, Cohen et al. 2011, 2010), and this worse-than-algebraic asymptotic growth of
DeepONet size in _ϵ_ _[−]_ [1] provides an appropriate notion of the curse of dimensionality
in the infinite-dimensional DeepONet context.


Given Remark 3.4 above, we can say that “DeepONets break the curse of dimensionality” in the approximation of a given operator _G_, if, for any accuracy _ϵ >_ 0,
there exists a DeepONet which achieves an approximation error _E_ [�] _≤_ _ϵ_ (cp. (2.11)),
with a complexity which scales _at most algebraically_ in _ϵ_ _[−]_ [1] . This key concept with
respect to computational complexity and efficiency of DeepONet approximation is
made precise below:


_**Definition**_ **3.5** (Curse of Dimensionality for DeepONets) _**.**_ Given a DeepONet _N_
(2.10), we define the _size_ of the DeepONet as the sum of the sizes of the approximator neural network _A_ and the trunk net _**τ**_, i.e. size( _N_ ) = size( _A_ ) + size( _**τ**_ ) (cp.
(2.6)). For a given tolerance _ϵ >_ 0, let _N_ _ϵ_ be a DeepONet such that the error _E_ [�]
(2.11) is less than _ϵ_, and

size ( _N_ _ϵ_ ) _∼O_ � _ϵ_ _[−][ϑ]_ _[ϵ]_ [�] _,_ (3.5)


for some _ϑ_ _ϵ_ _≥_ 0. Note that the universal approximation theorem 3.1 guarantees
the existence of such a _N_ _ϵ_ and _ϑ_ _ϵ_ for every _ϵ >_ 0.
The DeepONet approximation of a nonlinear operator _G_, with underlying measure _µ_ (check from Definition 2.1) is said to incur a _curse of dimensionality_, if


_ϵ_ lim _→_ 0 _[ϑ]_ _[ϵ]_ [ = +] _[∞][.]_ (3.6)


On the other hand, the DeepONet approximation is said to _break the curse of_
_dimensionality_ if there exist DeepONets _N_ _ϵ_ such,


_ϵ_ lim _→_ 0 _[ϑ]_ _[ϵ]_ [ =] _[ ϑ <]_ [ +] _[∞][.]_ (3.7)


This definition emphasizes the fundamental role played by bounds on the size
of the DeepONet for obtaining a certain level of error tolerance. We provide such
explicit bounds later in this paper. In the following subsections, we survey our main
results on the reconstruction error _E_ [�] _R_, the encoding error _E_ [�] _E_ and the approximation
error _E_ [�] _A_ .


ERROR ESTIMATES FOR DEEPONETS 13


3.2.2. _Reconstruction error._ The reconstruction error _E_ [�] _R_ (2.9) is intimately related
to the eigenfunctions and eigenvalues of the covariance operator Γ _G_ # _µ_ of the pushforward measure _G_ # _µ_,


Γ _G_ # _µ_ = ( _v −_ E[ _v_ ]) _⊗_ ( _v −_ E[ _v_ ]) _d_ ( _G_ # _µ_ )( _v_ ) _,_ (3.8)
ˆ _Y_



where E[ _v_ ] = ´ _Y_ _[v d]_ [(] _[G]_ [#] _[µ]_ [)(] _[v]_ [) denotes the mean of] _[ G]_ [#] _[µ]_ [. General results on the]

relation between the eigenstructure of covariance operators and optimal projections onto finite-dimensional linear and affine subspace are presented in Section
3.3, below. Based on these results, it will be shown that

(1) for any affine reconstruction _R_ : R _[p]_ _→_ _L_ [2] ( _U_ ), there exists a (unique)
optimal projection _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_, and that this optimal _P_ is itself _affine_
(cp. Lemma 3.13),
(2) among all affine reconstructions _R_ : R _[p]_ _→_ _L_ [2] ( _U_ ), there exists an optimal
choice _R_ opt : R _[p]_ _→_ _L_ [2] ( _U_ ), which achieves the minimum reconstruction
error: _E_ [�] _R_ opt = min _R_ _E_ [�] _R_ (cp. Theorem 3.14).
As a consequence of the analysis of the optimal reconstruction, we can then derive
the following _lower bound_ on the reconstruction and total approximation errors:


_**Theorem**_ **3.6** _**.**_ Consider the setting of Definition 2.1, let _G_ : _X →_ _L_ [2] ( _U_ ) be an
operator. Let _N_ = _R◦A◦E_ : _X →_ _L_ [2] ( _U_ ) be an arbitrary DeepONet approximation
of _G_, with encoder _E_ : _C_ ( _D_ ) _→_ R _[m]_, approximator _A_ : R _[m]_ _→_ R _[p]_ and reconstruction
_R_ : R _[p]_ _→_ _L_ [2] ( _U_ ). Let _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_ be the optimal affine projection associated
with _R_ . Then the total error _E_ [�] and the reconstruction error _E_ [�] _R_ can be estimated
_from below_ by

~~�~~ _λ_ _k_ _≤_ _E_ [�] _R_ _≤_ _E_ [�] _,_ (3.9)
� _k>p_


in terms of the eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ of the covariance operator Γ _G_ # _µ_ associated
with the push-forward measure _G_ # _µ_ .


Theorem 3.6 provides a definite, a priori limitation on the best error which can
be achieved by a DeepONet approximation. The proof of this theorem can be found
on page 20.
The next goal is to provide _upper_ bounds on _E_ [�] _R_ for a suitably chosen trunk net
reconstruction _R_ (2.9). As mentioned in point (2) above, in principle, there exists
a provably _optimal_ choice _R_ opt among all affine reconstructions. However, given
that the trunk net basis functions _τ_ 0 _, . . ., τ_ _p_ are represented by _neural networks_,
this optimal choice _R_ opt cannot in general be represented exactly by the trunk net
reconstruction _R_, leading to an additional contribution to the reconstruction error,
which depends on how well the eigenfunctions of the covariance operator can be
approximated by neural networks (cp. Proposition 3.15).
As will be discussed in Section 3.4.1, due to the distortion by _G_, the eigenstructure of the push-forward measure _G_ # _µ_ can be very different from that of _µ_ . In
fact, even if _µ_ has exponentially decaying spectrum of the covariance operator Γ _µ_,
the push-foward under _G_ can destroy such high rates of decay of the eigenvalues
of Γ _G_ # _µ_ (cp. Proposition 3.19). Thus, with the exception of _linear_ operators (cp.
Proposition 3.20), the eigenstructure of _G_ # _µ_ may to depend in a very complicated
way on both _G_ and _µ_, for _non-linear G_, making it difficult to analyze Γ _G_ # _µ_ .
As it can be very difficult to obtain the necessary information on the eigenfunctions needed to _quantify_ their approximability by the trunk net _**τ**_, we will discuss an
alternative way to obtain estimates on _E_ [�] _R_, in Section 3.4. This alternative relies on


14 ERROR ESTIMATES FOR DEEPONETS


a comparison principle with a given (non-optimal) reconstructor _R_ [�] : R _[p]_ _→_ _L_ [2] ( _U_ )
(cp. Lemma 3.16). As one concrete application, this general comparison principle
is applied to the reconstructor _R_ [�] = _R_ Fourier, obtained by expansion in the standard
Fourier basis. This allows us to derive the following quantitative upper reconstruction error and complexity estimate, which depends only on the _average smoothness_
of the output functions _G_ ( _u_ ) _∈_ _L_ [2] ( _U_ ) (stated in the periodic setting, for simplicity):


_**Theorem**_ **3.7** _**.**_ If _G_ defines a Lipschitz mapping _G_ : _X →_ _H_ _[s]_ (T _[n]_ ), for some _s >_ 0,
with


_∥G_ ( _u_ ) _∥_ _H_ [2] _[s]_ _[ dµ]_ [(] _[u]_ [)] _[ ≤]_ _[M <][ ∞][,]_

ˆ _X_

then there exists a constant _C_ = _C_ ( _n, s, M_ ) _>_ 0, such that for any _p ∈_ N, there
exists a trunk net _**τ**_ : R _[n]_ _→_ R _[p]_ (with bias term _τ_ 0 _≡_ 0), with

size( _**τ**_ ) _≤_ _Cp_ (1 + log( _p_ ) [2] ) _,_

(3.10)
depth( _**τ**_ ) _≤_ _C_ (1 + log( _p_ ) [2] ) _,_


and such that the associated reconstruction _R_ : R _[p]_ _→_ _L_ [2] (T _[n]_ ), _R_ ( _α_ ) = [�] _[p]_ _k_ =1 _[α]_ _[k]_ _[τ]_ _[k]_
satisfies

�
_E_ _R_ _≤_ _Cp_ _[−][s/n]_ _._ (3.11)


Furthermore, the reconstruction _R_ and the associated optimal projection _P_ satisfy
Lip( _R_ ) _,_ Lip( _P_ ) _≤_ 2.


Theorem 3.7 follows from the discussion in Section 3.4.3. Thus, (3.11) provides
us with a quantitative algebraic rate of decay for the reconstruction error (3.3) as
long as the size of the trunk net scales as in (3.10) and the nonlinear operator
_G_ maps onto the Sobolev space _H_ _[s]_ . Such nonlinear operators arise frequently in
PDEs as we will see in Section 4. For further details and extended analysis of the
reconstruction error, we refer to Section 3.2.2.


3.2.3. _Encoding error._ Next, we aim to bound the encoding error (3.1), associated
with the DeepONet (2.10). Full details and an extended discussion will be given in
Section 3.5.
We observe from (3.1) that the encoding error _does not depend_ on the nonlinear
operator _G_, but only depends on the underlying probability measure _µ_ . Following
the architecture used by Lu et al. (2019), we have fixed the form of the encoder _E_
(2.7) to be the point-wise evaluation of the input functions at _sensors_, i.e. points
_{x_ _j_ _} ⊂_ _D_ with 1 _≤_ _j ≤_ _m_ . Thus, key objectives of our analysis are to determine
suitable choices of sensors for a fixed _m_, as well as to find the appropriate form of
a decoder _D_ in order to minimize the encoding error (3.1). To this end, we start
with a result that provides a lower bound on the encoding error:



_**Theorem**_ **3.8** _**.**_ Let _µ_ be a probability measure on _X_ = _L_ [2] ( _D_ ) with ´ _X_ _[∥][u][∥]_ _L_ [2] [2] _x_ _[dµ]_ [(] _[u]_ [)] _[ <]_

_∞_ and ´ _X_ _[u dµ]_ [(] _[u]_ [) = 0. If] _[ E]_ [ :] _[ X][ →]_ [R] _[m]_ [ and] _[ D]_ [ :][ R] _[m]_ _[ →]_ _[X]_ [ are any encoder/decoder]



_**Theorem**_ **3.8** _**.**_ Let _µ_ be a probability measure on _X_ = _L_ [2] ( _D_ ) with ´



_∞_ and ´ _X_ _[u dµ]_ [(] _[u]_ [) = 0. If] _[ E]_ [ :] _[ X][ →]_ [R] _[m]_ [ and] _[ D]_ [ :][ R] _[m]_ _[ →]_ _[X]_ [ are any encoder/decoder]

pair with a _linear_ decoder _D_, then


_∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ][ ≥]_ � _λ_ _k_ _._

ˆ _X_



_X_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ][ ≥]_ �



� _λ_ _k_ _._

_k>m_



Here, _λ_ _k_ refers the _k_ -th eigenvalue of the covariance operator Γ _µ_ associated with
the measure _µ_ . In particular, we then have the lower bound


�
_E_ _E_ _≥_ _λ_ _k_ _._ (3.12)


_k>m_

~~��~~


ERROR ESTIMATES FOR DEEPONETS 15


The proof is presented in Appendix C.10. The bound (3.12) provides a lower
bound on the encoding error and connects this error to the spectral decay of the
underlying covariance operator, at least for linear decoders.
Our next aim is to derive upper bounds on the encoding error. To illustrate our
main ideas, we will restrict our discussion to the case _X_ = _L_ [2] ( _D_ ). The results are
readily extended to more general spaces, such as Sobolev spaces _X_ = _H_ _[s]_ ( _D_ ) for
_s >_ 0. We fix a probability measure _µ_ on _L_ [2] ( _D_ ), and write the covariance operator
Γ _µ_ as an eigenfunction decomposition,



Γ _µ_ =



_∞_
� _λ_ _ℓ_ ( _φ_ _ℓ_ _⊗_ _φ_ _ℓ_ ) _,_


_ℓ_ =1



where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ are the decreasing eigenvalues, and such that the _φ_ _ℓ_ are an
orthonormal basis of _L_ [2] ( _D_ ). We will assume that all _φ_ _ℓ_ are continuous functions,
so that point-wise evaluation of _φ_ _ℓ_ makes sense. In this case, it can be shown
(cp. Lemma 3.22) that the encoding error is composed of the optimal lower bound
(3.12), and an additional _aliasing_ contribution, due to the encoding in terms of
point-evaluations:
( _E_ [�] _E_ ) [2] = � _λ_ _ℓ_ + ( _E_ [�] aliasing ) [2] _._

_ℓ>m_

An explicit expression for _E_ [�] aliasing can be given (cp. (3.40)), but finding sensor
locations _x_ 1 _, . . ., x_ _m_ which minimize this aliasing error contribution appears to be
very difficult, in general. We therefore propose to replace the sensors _x_ 1 _, . . ., x_ _m_ by
_random_ sensor locations _X_ 1 _, . . ., X_ _M_ (iid, uniformly distributed over the domain
_D_ ), and study the corresponding _random encoder_,


_E_ ( _u_ ) = ( _u_ ( _X_ 1 ) _, . . ., u_ ( _X_ _M_ )) _._


Surprisingly, it can be shown that this random encoder can be close to optimal, as
is made precise by the following theorem:


_**Theorem**_ **3.9** _**.**_ If the eigenbasis of the uncentered covariance operator Γ (3.14),
associated with the underlying measure _µ_, is bounded in _L_ _[∞]_, then there exists a
constant _C ≥_ 1, depending only on sup _ℓ∈_ N _∥φ_ _ℓ_ _∥_ _L_ _[∞]_ and _|D|_, such that the encoding
error (3.1) with _M_ random sensors satisfies (for almost all _M ∈_ N):


�
_E_ _E_ ( _X_ 1 _, . . ., X_ _M_ ) _≤_ _C_ ~~�~~ _λ_ _ℓ_ _,_ (as _M →∞_ ) _,_
~~�~~ _ℓ>M/C_ log( _M_ )


with probability 1 in the iid random sensors _X_ 1 _, X_ 2 _, X_ 3 _, · · · ∼_ Unif( _D_ ).


Thus, for an underlying measure _µ_ whose covariance operator has a bounded
eigenbasis, then as the number of randomly chosen sensors increases, the resulting
encoding error goes to zero almost surely. The rate of decay only depends on the
spectral decay of the covariance operator. Moreover, given the lower bound (3.12)
on the encoding error, we have the surprising result that _randomly chosen sensor_
_points_ lead to an optimal (up to a log) decay of the encoding error (3.1), corresponding to the DeepONet (2.10). We refer the interested reader to the discussion
leading to Lemma 3.24, page 29, for precise details of the derivation of Theorem
3.9.

In addition to the above general results, we also consider the specific case of an
input measure _µ ∈P_ ( _L_ [2] (T _[d]_ )), which is given as the _law_ of a random field of the
form
_u_ ( _x_ ; _Y_ ) = ~~_u_~~ ~~(~~ _x_ ) + � _Y_ _k_ _α_ _k_ **e** _k_ ( _x_ ) _,_

_k∈_ Z _[d]_


16 ERROR ESTIMATES FOR DEEPONETS


where _{_ **e** _k_ ( _x_ ) _}_ _k∈_ Z _d_ denotes the trigonometric basis (cp. Appendix A), _Y_ _k_ _∈_ [ _−_ 1 _,_ 1]
are centered random variables, and the coefficients _α_ _k_ _≥_ 0 satisfy a decay of the
form _α_ _k_ ≲ exp( _−ℓ|k|_ _∞_ ), for all _k ∈_ Z _[d]_, for a fixed “length-scale” _ℓ>_ 0. In this case,
we study the encoder _E_ obtained by evaluation at sensor locations on an _equidistant_
_grid_ on T _[d]_ . Given this setting, we show that there exists a decoder _D_, such that the
corresponding encoding error� _E_ [�] _E_ can be estimated by an exponential upper bound,
_E_ _E_ ≲ exp( _−cℓm_ [1] _[/d]_ ) (cp. Theorem 3.28). We refer to Section 3.5.3 for the precise
details.


3.2.4. _Approximation error._ Given a particular choice of encoder/decoder and reconstruction/projection pairs ( _E, D_ ) and ( _R, P_ ), the approximation error _E_ [�] _A_ (3.2)
for the approximator _A_ : R _[m]_ _→_ R _[p]_ in the DeepONet (2.10) is a measure for the
non-commutativity of the following diagram:


_G_

_L_ [2] ( _D_ ) _L_ [2] ( _U_ )



_E_ _D_ _P_



_R_



_A_

R _[m]_ R _[p]_


I.e., it measures the error in the approximation _A ≈_ _G_ = _P ◦G ◦D_ . Thus, bounding
the approximation error _E_ [�] _A_ can be viewed as a special instance of the general
problem of the neural network approximation of a high-dimensional mapping _G_ :
R _[m]_ _→_ R _[p]_ . As already pointed out in Remark 3.4, relying on “mild” regularity
properties of _G_ (or _G_ ), such as Lipschitz continuity, leads to complexity bounds
which suffer from the curse of dimensionality (cp. Definition 3.5, and Section 3.6.1,
below).
Given this possible curse of dimensionality in bounding the approximation error
(3.2) for Lipschitz continuous maps, we seek to find a class of nonlinear operators
_G_ for which this curse of dimensionality can be avoided:

_•_ One possible class is the class of _holomorphic mappings_ [ _−_ 1 _,_ 1] [N] _→_ _V_, _**y**_ _�→_
_F_ ( _**y**_ ), with _V_ an arbitrary Banach space: Such operators have been shown
in recent papers to be efficiently approximated by ReLU neural networks,
_breaking the curse of dimensionality_ .

_•_ Another strategy relies on the use of additional structure of the underlying
operator _G_, which is _not captured_ by its smoothness properties. In this
direction, we show that many PDE operators may possess such internal
structure, making them not only amenable to approximation by classical
numerical methods, but also enabling DeepONets to _break the curse of_
_dimensionality_ .


The first approach relying on holomorphy is discussed at an abstract level in
Section 3.6.2, where we apply the main results of Schwab & Zech (2019), Opschoor
et al. (2019, 2020) to DeepONets. The discussion of this specific parametrized
setting ultimately leads to Theorem 3.35, which provides quantitative estimates on
the approximation error for the DeepONet approximation of holomorphic operators.
This abstract result is applied to two concrete examples of holomorphic operators
in Sections 4.1 and 4.2.

In contrast to the holomorphic case, generally applicable results which rely on
internal structure of _G other than smoothness_, appear to be much more difficult to
state at an abstract level; therefore, in this case, we instead focus on a case-by-case
discussion of the approximation error for concrete operators of interest, which we
defer to Section 4.3, for a parabolic PDE, and in Section 4.4, for a hyperbolic PDE.
The remaining subsections of the present Section 3 provide the details as well
as an extended discussion of the error associated with projections onto linear and


ERROR ESTIMATES FOR DEEPONETS 17


affine subspaces (Section 3.3), bounds on the reconstruction error (Section 3.4),
bounds on the encoding error (Section 3.5), and bounds on the approximation
error (Section 3.6).


3.3. **On the error due to projections of Hilbert spaces onto linear and**
**affine subspaces.** We start with the observation that the encoder _E_ is a linear
mapping from _X_ to R _[m]_ . As long as we choose the decoder _D_ to also be a linear
mapping from R _[m]_ to _X_, we see that the encoding error _E_ [�] _E_ (3.1) can be bounded
from below by the following _projection error_ :

_E_ � Proj ( _V_ � ; _µ_ ) = ˆ _X_ _v_ � inf _∈V_ [�] _∥v −_ _v_ � _∥_ _X_ [2] _[dµ]_ [(] _[v]_ [)] _[,]_ (3.13)


where _V_ [�] = Im( _D_ ).
Hence, we need to study general properties of the projection error _E_ [�] Proj ( _V_ [�] ; _ν_ )
onto a finite-dimensional linear subspace _V_ [�] _⊂_ _H_, for an arbitrary Hilbert space _H_,
and given a probability measure _ν ∈P_ ( _H_ ) with finite second moment ´ _H_ _[∥][v][∥]_ _H_ [2] _[dν]_ [(] _[v]_ [).]

This study results in the following theorem that characterizes _optimal finite-dimensional_
_subspaces_ of the Hilbert space _H_,


_**Theorem**_ **3.10** _**.**_ Let _ν ∈P_ 2 ( _H_ ) be a probability measure on a separable Hilbert
space _H_ . For any _p ∈_ N, there exists an optimal _p_ -dimensional subspace _V_ _p_ _⊂_ _H_,
such that

� � �
_E_ Proj ( _V_ _p_ ; _ν_ ) = � inf _E_ Proj ( _V_ ; _ν_ ) _._
_V ⊂V,_
dim( _V_ [�] )= _p_


Furthermore, we can characterize the set of optimal subspaces _V_ [�] _⊂_ _H_ as follows:
Let _λ_ 1 _> λ_ 2 _> . . ._ denote the distinct eigenvalues of the operator


Γ = ( _v ⊗_ _v_ ) _dν_ ( _v_ ) _._ (3.14)
ˆ _H_


Let _E_ _k_ = � _v ∈_ _H_ �� Γ _v_ = _λ_ _k_ _v_ �, _k ∈_ N denote the corresponding eigenspaces. Choose
_n ∈_ N, such that



_n−_ 1
� dim( _E_ _k_ ) _< p ≤_


_k_ =1



_n_
� dim( _E_ _k_ ) _._


_k_ =1



A _p_ -dimensional subspace _V_ [�] _⊂_ _H_ is optimal for _E_ [�] Proj ( _V_ [�] ; _ν_ ), if and only if,



_n−_ 1
� _E_ _k_ _⊂_ _V_ [�] _⊂_


_k_ =1



_n_
� _E_ _k_ _._


_k_ =1



For any _n ∈_ N, there exists a _unique_ optimal subspace _V_ _p_ _n_ _⊂_ _H_, of dimension
_p_ _n_ = [�] _[n]_ _k_ =1 [dim(] _[E]_ _[k]_ [). For any optimal subspace][ �] _[V][ ⊂]_ _[H]_ [, the resulting projection]
error is given by
_E_ � Proj ( _V_ � ) = � _λ_ _j_ _,_ (3.15)

_j>p_


where



_λ_ 1 = _· · ·_ = _λ_ _n_ 1
� ~~��~~ �

= _λ_ 1




_> λ_ _n_ 1 +1 = _· · ·_ = _λ_ _n_ 1 + _n_ 2
� ~~��~~ �

= _λ_ 2




_> . . .,_



are the eigenvalues of Γ repeated according to multiplicity, with _n_ _j_ = dim( _E_ _j_ ).


18 ERROR ESTIMATES FOR DEEPONETS


The proof of this theorem is based on a series of highly technical lemmas and
is presented in detail in Appendix C.3. The essence of the above theorem is the
connection between optimal linear subspaces (that minimize projection errors) of
a Hilbert space and the eigensystem of its (uncentered) covariance operator (3.14).
Thus given (3.15), the study of projection errors with respect to finite-dimensional
linear subspaces requires a careful investigation into the decay of the eigenvalues of
the operator (3.14) and will be instrumental in providing bounds on the encoding
error (3.1) and enable us to identify suitable sensors _{x_ _j_ _}_ for defining the encoder
_E_ .


_**Remark**_ **3.11** _**.**_ We would like to point out that the main observations of Theorem
3.10, and in particular, the important identity (3.15) for the minimal projection
error have previously been observed in Bhattacharya et al. (2021). In the finitedimensional case, the underlying ideas are well-known in principal component analysis. While the basic ideas are not new, we nevertheless include Theorem 3.10 in
the present work for completeness, due to its central importance to our discussion.


Similarly, we observe that the trunk-net induced reconstructor _R_ (2.9) is an
affine mapping between R _[p]_ and the output Banach space _Y_ . As will be shown in
Lemma 3.13, the reconstruction error (3.3) for any _R_ can be bounded from below
by the error with respect to projection onto affine subspaces of the output Hilbert
space. We formalize this notion below.
Given a separable Hilbert space _H_, let _V_ [�] 0 now denote an affine subspace of the
form



_α_ 1 _, . . ., α_ _p_ _∈_ R
������






 _[.]_



�
_V_ 0 =






 _[v]_ [�][ =][ �] _[v]_ [0] [ +]



_p_


�

� _α_ _j_ _v_ _j_

_j_ =1



for � _v_ 0 _, . . .,_ � _v_ _p_ _∈_ _H_ . Note that for _any_ � _v_ _[′]_ _∈_ _V_ [�] 0, the set


� � � �
_V_ 0 _−_ _v_ _[′]_ = _v −_ _v_ _[′]_ [ ��] _v ∈_ _V_ [�] _,_
� �� �

is a vector space _V_ �, associated with the affine space _V_ [�], spanned by � _v_ 1 � _, . . .,V_ 0 is unique and only depends on � _v_ _p_ . It is easy to see that the vector space � _V_ 0, and not
on a particular choice of the � _v_ 0 _, . . .,_ � _v_ _p_ .
The following theorem provides a complete characterization of finite-dimensional
optimal affine subspaces of _H_ and the resulting projection error.


_**Theorem**_ **3.12** _**.**_ Let _H_ be a separable Hilbert space and _ν ∈P_ 2 ( _H_ ) be a probability
measure with finite second moment. Let _p ∈_ N. Let _V_ [�] 0 be an affine subspace with
associated vector space _V_ [�] such that dim( _V_ [�] ) = _p_ . Then there exists a unique
element � _v_ 0 _∈_ _V_ [�] 0 such that


_∥_ E[ _v_ ] _−_ _v_ � 0 _∥_ = inf _v_ � _∈V_ [�] 0 _∥_ E[ _v_ ] _−_ _v_ � _∥,_


and the projection error given by,

_E_ � Proj ( _V_ � 0 ) = ˆ _H_ _v_ � inf _∈V_ [�] 0 _∥v −_ _v_ � _∥_ [2] _dν_ ( _v_ ) (3.16)


can be written as,

_E_ � Proj ( _V_ � ) = _∥_ E[ _v_ ] _−_ _v_ � 0 _∥_ [2] + ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ ) _,_ (3.17)


ERROR ESTIMATES FOR DEEPONETS 19


Furthermore, the affine space _V_ [�] 0 is a minimizer of _E_ [�] Proj ( _V_ [�] 0 ) (in the class of affine
subspaces of dimension _p_ ), if and only if,



_n_
� _E_ _j_ _._

_j_ =1



_v_ � 0 = E[ _v_ ] _,_ and



_n−_ 1
� _E_ _j_ _⊂_ _V_ [�] _⊂_

_j_ =1



Here _E_ _j_ denote the eigenspaces of the covariance operator


Γ = ( _v −_ E[ _v_ ]) _⊗_ ( _v −_ E[ _v_ ]) _dν_ ( _v_ ) _,_ (3.18)
ˆ _H_



associated with the distinct eigenvalues _λ_ 1 _> λ_ 2 _> . . ._, and _n ∈_ _N_ is chosen such
that
_n−_ 1 _n_
� dim( _E_ _j_ ) _< p ≤_ � dim( _E_ _j_ ) _._



� dim( _E_ _j_ ) _< p ≤_

_j_ =1



_n_
�



� dim( _E_ _j_ ) _._

_j_ =1



In this case, the projection error is given by

_E_ � Proj ( _V_ � 0 ) = � _λ_ _j_ _,_

_j>p_



where

_λ_ 1 = _· · ·_ = _λ_ _n_ 1
� ~~��~~ �

= _λ_ 1




_> λ_ _n_ 1 +1 = _· · ·_ = _λ_ _n_ 1 + _n_ 2
� ~~��~~ �

= _λ_ 2




_> . . .,_



are the eigenvalues of Γ repeated according to multiplicity, with _n_ _j_ = dim( _E_ _j_ ).


The above theorem is proved in Appendix C.3.2. It is the extension of Theorem
3.10 to affine subspaces of a Hilbert space. It serves to relate the projection error
to the decay of eigenvalues of the associated covariance operator (3.18) and will be
the key to proving bounds on the reconstruction error (3.3) and in the identification
of the optimal trunk network for the DeepONet (2.10).


3.4. **Bounds on the reconstruction error** (3.3) **.** In this section, we will apply
results from the previous sub-section to bound the reconstruction error _E_ [�] _R_ (3.3).
We start by recalling that the reconstructor _R_ (2.9) in the DeepONet (2.10) is
affine. The following lemma, whose proof is provided in Appendix C.4, identifies
the optimal projector _P_ for a given reconstructor _R_ .


_**Lemma**_ **3.13** _**.**_ Let _R_ = _R_ _**τ**_ : R _[p]_ _→_ _L_ [2] ( _U_ ) be an affine reconstructor of the
form (2.9), for _τ_ 0 _∈_ _L_ [2] ( _U_ ) and linearly independent _τ_ 1 _, . . ., τ_ _p_ _∈_ _L_ [2] ( _U_ ). Let
_ν ∈P_ 2 ( _L_ [2] ( _U_ )) be a probability measure with finite second moments. Then, the
reconstruction error (3.3) is minimized in the class of Borel measurable projectors
_P_ : _L_ [2] ( _U_ ) _→_ R _[p]_, for

_P_ ( _u_ ) := � _⟨u −_ _τ_ 0 _, τ_ 1 _[∗]_ _[⟩][, . . .,][ ⟨][u][ −]_ _[τ]_ [0] _[, τ]_ _p_ _[ ∗]_ _[⟩]_ � _,_ (3.19)

where _τ_ 1 _[∗]_ _[, . . ., τ]_ _p_ _[ ∗]_ _[∈]_ [span(] _[τ]_ [1] _[, . . ., τ]_ _[p]_ [) denotes the dual basis of] _[ τ]_ [1] _[, . . ., τ]_ _[p]_ [, i.e. such]
that
_⟨τ_ _ℓ_ _, τ_ _k_ _[∗]_ _[⟩]_ [=] _[ δ]_ _[ℓk]_ _[,]_ _∀_ _k, ℓ_ _∈{_ 1 _, . . ., p}._

In this case, we have



_R ◦P_ ( _u_ ) = _τ_ 0 _[⊥]_ [+]



_p_
� _⟨u, τ_ _k_ _[∗]_ _[⟩][τ]_ _[k]_ (3.20)


_k_ =1



where



_τ_ 0 _[⊥]_ [=] _[ τ]_ [0] _[−]_



_p_
� _⟨τ_ 0 _, τ_ _k_ _[∗]_ _[⟩][τ]_ _[k]_ _[,]_


_k_ =1



is the projection of _τ_ 0 onto the orthogonal complement of span( _τ_ 1 _, . . ., τ_ _p_ ) _⊂_ _L_ [2] ( _U_ ).


20 ERROR ESTIMATES FOR DEEPONETS


Next, we can directly apply Theorem 3.12 to identify an _optimal_ reconstructor
_R_, and apply Lemma 3.13 to identify the associated _optimal_ projector _P_, in the
following theorem,


_**Theorem**_ **3.14** _**.**_ Denote _ν_ := _G_ # _µ_ . If the reconstruction _R_ = _R_ _**τ**_ : R _[p]_ _→_ _L_ [2] ( _U_ ) is
fixed to be of the form (2.9) with _τ_ _j_ _∈_ _L_ [2] ( _U_ ) for _j_ = 0 _, . . ., p_, then the reconstruction error (3.3), in the class of affine reconstructions _R_ and _arbitrary measurable_
projections _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_, is minimized by the choice



_R_ opt = _R_ _**τ**_ � ( _α_ 1 _, . . ., α_ _p_ ) = � _τ_ 0 +



_p_


�

� _α_ _j_ _τ_ _j_ _,_ (3.21)

_j_ =1



where

_τ_ � 0 = E _ν_ [ _v_ ] = _v dν,_
ˆ _L_ [2] ( _U_ )

is the mean and � _τ_ _j_, _j_ = 1 _, . . ., p_ are _p_ eigenvectors of the covariance operator


Γ = ( _v −_ E _ν_ [ _v_ ]) _⊗_ ( _v −_ E _ν_ [ _v_ ]) _dν_ ( _v_ ) _,_
ˆ _Y_


corresponding to the _p_ largest eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥· · · ≥_ _λ_ _p_ _≥_ _. . ._ . Furthermore,
the optimal reconstruction error satisfies the lower bound


�
_E_ _R_ opt _≥_ _λ_ _k_ _,_ (3.22)
~~��~~ _k>p_


in terms of the spectrum _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ of Γ (eigenvalues repeated according to
their multiplicity). Given _R_ opt, the corresponding optimal measurable projection
_P_ : _L_ [2] ( _U_ ) _→_ R _[p]_ is affine and is given by the orthogonal projection


� �
_P_ ( _v_ ) = ( _⟨_ ( _v −_ _τ_ 0 ) _,_ � _τ_ 1 _⟩, . . ., ⟨_ ( _v −_ _τ_ 0 ) _,_ � _τ_ _p_ _⟩_ ) _._ (3.23)


Given the existence of an _optimal_ affine reconstructor (3.21) and the associated
optimal projector (3.23), the proof of Theorem 3.6 is now straightforward.


_Proof of Theorem 3.6._ Let _N_ = _R ◦A ◦E_ be a DeepONet approximation of _G_ . We
aim to show that
~~�~~ _λ_ _k_ _≤_ _E_ [�] _R_ _≤_ _E_ [�] _,_
� _k>p_

where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote the eigenvalues of Γ _G_ # _µ_ .
By Lemma 3.13, the optimal projector _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_ for a the given reconstructor _R_ is such that _R ◦P_ = Π _V_ 0 : _L_ [2] ( _U_ ) _→_ _L_ [2] ( _U_ ) is the orthogonal projection
onto the affine subspace _V_ 0 = Im( _R_ ) _⊂_ _L_ [2] ( _U_ ). To prove the lower bound _E_ [�] _R_ _≤_ _E_ [�],
we observe that for any _u ∈_ _X_, we have


_∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) = _∥G_ ( _u_ ) _−R ◦A ◦E_ ( _u_ ) _∥_ _L_ 2 ( _U_ )

_≥_ inf
_v∈V_ 0 _[∥G]_ [(] _[u]_ [)] _[ −]_ _[v][∥]_ _[L]_ [2] [(] _[U]_ [)]

= _∥G_ ( _u_ ) _−_ Π _V_ 0 _G_ ( _u_ ) _∥_ _L_ 2 ( _U_ )
= _∥G_ ( _u_ ) _−R ◦P ◦G_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) _,_


and hence,

�
_E_ = _∥G −R ◦A ◦E∥_ _L_ 2 ( _µ_ ) _≥∥G −R ◦P ◦G∥_ _L_ 2 ( _µ_ ) = _∥_ Id _−R ◦P∥_ _L_ 2 ( _G_ # _µ_ ) = � _E_ _R_ _._


Furthermore, by Theorem 3.14, we thus have


_λ_ _k_ = _E_ [�] _R_ opt _≤_ _E_ [�] _R_ _≤_ _E_ [�] _,_ (3.24)

~~��~~ _k>p_


ERROR ESTIMATES FOR DEEPONETS 21


where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote the eigenvalues of the covariance operator Γ _G_ # _µ_ of the
push-forward measure _G_ # _µ_ . This proves the claim. 

The lower bound in (3.24) is fundamental, as it reveals that the spectral decay
rate for the operator Γ _G_ # _µ_ of the push-forward measure essentially determines how
low the approximation error of DeepONets can be for a given output dimension _p_
of the trunk nets.

After establishing the lower bound on the reconstruction error, we seek to derive
upper bounds on this error. We observe that the optimal reconstructor is given by
eigenfunctions of the covariance operator Γ (3.18). In general, these eigenfunctions
are not neural networks of the form (2.1). However, given the fact that neural
networks are universal approximators of functions on finite dimensional spaces,
one can expect that the trunk nets in (2.9) will approximate these underlying
eigenfunctions to high accuracy. This is indeed established in the following,


_**Proposition**_ **3.15** _**.**_ Let _ν_ = _G_ # _µ ∈P_ 2 ( _Y_ ) be a probability measure with finite
second moments. Write the covariance operator in the form



Γ _ν_ =



_∞_
� _λ_ _k_ ( _φ_ _k_ _⊗_ _φ_ _k_ ) _,_


_k_ =1



with _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ and orthonormal eigenbasis _φ_ _k_ . Let � _τ_ _k_, _k_ = 0 _,_ 1 _, . . ., p_, denote
the optimal choice for an affine reconstruction _R_ opt, as in Theorem 3.14, more
precisely, let � _τ_ 0 = E _ν_ [ _v_ ], � _τ_ _k_ = _φ_ _k_ for _k_ = 1 _, . . ., p_ . Let _τ_ 0 _, τ_ 1 _, . . ., τ_ _p_ be the trunk
net functions of an arbitrary DeepONet (2.10). The reconstruction error for the
reconstruction _R_ = _R_ _**τ**_ :



_R_ ( _α_ 1 _, . . ., α_ _p_ ) = _τ_ 0 +



_p_
� _α_ _k_ _τ_ _k_ _,_


_k_ =1



with corresponding (unique optimal) projection _P_ (3.19) satisfies



�
_E_ _R_ _≤_ �



1 + Tr(Γ _ν_ ) max _k_ =0 _,...,p_ _[∥][τ]_ [�] _[k]_ _[ −]_ _[τ]_ _[k]_ _[∥]_ _[L]_ [2] _[y]_ [ +] _λ_ _k_ _._ (3.25)
~~��~~ _k>p_



This proposition is proved in Appendix C.5. The estimate (3.25) shows that an
upper bound on the reconstruction error has two contributions, one of them arises
from the decay rate of the eigenvalues of the covariance operator associated with
the push-forward measure _ν_ = _G_ # _µ_ and does not depend on the underlying neural
networks. On the other hand, the second contribution to (3.25) depends on the
choice and approximation properties of the trunk net in the DeepONet (2.10).
Thus, to bound the reconstruction error, we need some explicit information about
the optimal reconstruction in terms of the eigensystem of the covariance operator.
However in practice, one does not have access to the form of the nonlinear operator
_G_, but only to measurements of _G_ on a finite number of training samples. Hence, it
may not be possible to determine what the optimal reconstructor for a particular
nonlinear operator _G_ is. Therefore, we have the following lemma that compares
the reconstruction error of the DeepONet (2.10) with the reconstruction error that
arises from another, possibly non-optimal, choice of affine reconstructor.


_**Lemma**_ **3.16** _**.**_ Let _ν ∈P_ 2 ( _Y_ ) be a probability measure with finite second moment.
Let _**τ**_ = (0 _, τ_ 1 _, . . ., τ_ _p_ ) denote the trunk net functions of a DeepONet _without bias_
( _τ_ 0 _≡_ 0), and with associated reconstruction _R_ = _R_ _**τ**_ :



_R_ ( _α_ 1 _, . . ., α_ _p_ ) =



_p_
� _α_ _k_ _τ_ _k_ _._


_k_ =1


22 ERROR ESTIMATES FOR DEEPONETS


Let � _**τ**_ = (0 _,_ � _τ_ 1 _, . . .,_ � _τ_ _p_ ) denote the basis functions for a reconstruction _R_ [�] = _R_ [�] _**τ**_ � :
R _[p]_ _→_ _Y_ without bias ( _τ_ � 0 _≡_ 0), of the form



�
_R_ ( _α_ 1 _, . . ., α_ _p_ ) =



_p_


�

� _α_ _k_ _τ_ _k_ _._


_k_ =1



Assume that the functions � _τ_ 1 _, . . .,_ � _τ_ _p_ _∈_ _Y_ are _orthonormal_ . Let _P,_ _P_ [�] : _Y →_ R _[p]_

denote the corresponding projection mappings (3.19), and let _E_ [�] _R_, _E_ [�] _R_ � [denote the]
reconstruction errors of ( _R, P_ ) and ( _R_ [�] _,_ _P_ [�] ), respectively. Let _ϵ ∈_ (0 _,_ 1 _/_ 2), _p ≥_ 1 be
given. If


_ϵ_
_k_ =1 max _,...,p_ _[∥][τ]_ _[k]_ _[ −]_ _[τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] _[ ≤]_ _p_ [3] _[/]_ [2] _[,]_ (3.26)


then we have

� �
_E_ _R_ _≤_ _E_ _R_ � [+] _[ Cϵ.]_ (3.27)



where _C ≥_ 1 depends only on ´



_L_ [2] _[ ∥][u][∥]_ [2] _[ dν]_ [(] _[u]_ [). Furthermore, we have the estimate]



Lip( _P_ ) _,_ Lip( _R_ ) _≤_ 2 _,_ (3.28)


for the Lipschitz constant of the projection _P_ (cp. (3.19)) and _R_, respectively,
where


Lip( _P_ ) = Lip � _P_ : ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ) _→_ (R _[p]_ _, ∥· ∥_ _ℓ_ 2 )� _,_

Lip( _R_ ) = Lip � _R_ : (R _[p]_ _, ∥· ∥_ _ℓ_ 2 ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 )� _._


The proof of this lemma is provided in Appendix C.6 and we would like to point
out that for simplicity of exposition, we have set the _biases τ_ 0 = � _τ_ 0 _≡_ 0 in the
above. One can readily incorporate these bias terms to derive an analogous version
of the bound (3.27).
We illustrate the comparison principle elucidated in Lemma 3.16 with an example. To this end, we set the target space as the _n_ -dimensional torus, i.e. _U_ = [0 _,_ 2 _π_ ] _[n]_

(or _U_ = T _[n]_ ). With respect to this _U_, a possible _canonical_ reconstructor is given
by the _n_ -dimensional Fourier reconstruction;



_R_ Fourier ( _α_ 1 _, . . ., α_ _p_ ) =



_p_
� _α_ _j_ **e** _j_ ( _x_ ) _,_ (3.29)

_j_ =1



where we use the notation introduced in Appendix A. With respect to this Fourier
reconstructor, we prove in Appendix C.7 the following estimate on approximation
by trunk neural networks,


_**Lemma**_ **3.17** _**.**_ Let _n, p ∈_ N, and consider the Fourier reconstruction _R_ Fourier on

[0 _,_ 2 _π_ ] _[n]_ _≃_ T _[n]_ . There exists a constant _C >_ 0, independent of _p_, such that for any
_ϵ ∈_ (0 _,_ 1 _/_ 2), there exists a trunk net _**τ**_ : R _[n]_ _→_ R _[p]_, with

size( _**τ**_ ) _≤_ _Cp_ (1 + log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _,_

depth( _**τ**_ ) _≤_ _C_ (1 + log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _,_


and such that


_p_ [3] _[/]_ [2] max (3.30)
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ **[e]** _[j]_ _[∥]_ _[L]_ [2] [([0] _[,]_ [2] _[π]_ []] _[n]_ [)] _[ ≤]_ _[ϵ,]_


where **e** 1 _, . . .,_ **e** _p_ denote the first _p_ elements of the Fourier basis.
Furthermore, the Lipschitz norm of _R_ = _R_ _**τ**_ and of the linear projection _P_ :
_L_ [2] ( _U_ ) _→_ R _[p]_ associated with the _**τ**_ -induced reconstruction _R_ _**τ**_ via (3.19) can be
estimated by
Lip � _R_ : (R _[p]_ _, ∥· ∥_ _ℓ_ 2 ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 )� _≤_ 2 _,_


ERROR ESTIMATES FOR DEEPONETS 23


and
Lip � _P_ : ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ) _→_ (R _[p]_ _, ∥· ∥_ _ℓ_ 2 )� _≤_ 2 _._


Now combining the results of the above Lemma with the estimate (3.27) yields
the following result,


_**Lemma**_ **3.18** _**.**_ Let _n ∈_ N and fix _ν ∈P_ 2 ( _L_ [2] (T _[n]_ )). There exists a constant _C >_ 0,
depending only on _n_ and ´ _L_ [2] (T _[n]_ ) _[∥][u][∥]_ [2] _[ dν]_ [(] _[u]_ [), such that for any] _[ ϵ][ ∈]_ [(0] _[,]_ [ 1] _[/]_ [2), there]

exists a trunk net _**τ**_ : R _[n]_ _→_ R _[p]_, with


size( _**τ**_ ) _≤_ _C_ (1 + _p_ log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _,_

depth( _**τ**_ ) _≤_ _C_ (1 + log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _,_


and such that the reconstruction



_R_ : R _[p]_ _→_ _L_ [2] (T _[n]_ ) _,_ _R_ ( _α_ 1 _, . . ., α_ _p_ ) =



_p_
� _α_ _k_ _τ_ _k_ _,_


_k_ =1



satisfies � �
_E_ _R_ _≤_ _E_ _R_ Fourier + _Cϵ._ (3.31)

Furthermore, _R_ and the associated projection _P_ : ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 _→_ (R _[p]_ _, ∥· ∥_ _ℓ_ 2 )
(cp. (3.19)) satisfy Lip( _R_ ) _,_ Lip( _P_ ) _≤_ 2.


The significance of bound (3.31) lies in the fact that it reduces the problem of
estimating the reconstruction error (3.3) for a DeepONet to estimating the reconstruction error for a Fourier reconstructor (3.29), which might be much easier to
derive in concrete examples. Our next goal will be to obtain further insight into
the decay of the eigenvalues of Γ _G_ # _µ_ (in terms of _G_ and _µ_ ), and – in view of the
explicit complexity estimate provided by Lemma 3.18 – to estimate _E_ [�] _R_ Fourier .


3.4.1. _On the decay of spectrum for the push-forward measure._ From the bound
(3.4), it is clear that the decay of eigenvalues of the covariance operator, associated
with the push forward measure _G_ # _µ_ plays a crucial role in estimating the total
error (2.11), and in particular, the reconstruction error (3.3).
If _G_ is at least Lipschitz continuous, then we can write



Tr(Γ _G_ # _µ_ ) =
ˆ _Y_



2
�� _v −_ E _G_ # _µ_ [ _v_ ]�� _L_ [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[v]_ [)]



= inf
_v_ � _∈L_ [2] ( _U_ )


= inf
_v_ � _∈L_ [2] ( _U_ )



�
_∥v −_ _v∥_ [2] _L_ [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[v]_ [)]

ˆ _L_ [2] ( _U_ )


�
_∥G_ ( _u_ ) _−_ _v∥_ [2] _L_ [2] ( _U_ ) _[dµ]_ [(] _[u]_ [)] _[.]_

ˆ _X_



Making the particular (sub-optimal) choice � _v_ := _G_ (E _µ_ [ _u_ ]) _∈_ _L_ [2] ( _U_ ) and utilizing
the Lipschitz continuity of _G_, we can estimate the last expression as


_≤_ _∥G_ ( _u_ ) _−G_ (E _µ_ [ _u_ ]) _∥_ [2] _L_ [2] ( _U_ ) _[dµ]_ [(] _[u]_ [)]
ˆ _X_


_≤_ Lip( _G_ ) [2] _∥u −_ E _µ_ [ _u_ ] _∥_ [2] _X_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_

= Lip( _G_ ) [2] Tr (Γ _µ_ ) _._


Thus, Γ _G_ # _µ_ is a trace-class operator for Lipschitz continuous _G_, implying that

� _λ_ _k_ _→_ 0 _,_ (as _p →∞_ ) _,_

_k>p_


24 ERROR ESTIMATES FOR DEEPONETS


However, the efficiency of DeepONets in approximating the nonlinear operator _G_
relies on the _precise rate_ of this spectral decay. In particular, an exponential decay
of the eigenvalues would facilitate efficient approximation by DeepONets.
Clearly, the spectral decay of Γ _G_ # _µ_ depends on both _G_ and _µ_ (possibly in a
complicated manner). If the eigenvalues of Γ _µ_ decay rapidly, e.g. exponentially,
one might hope that the same is true for the eigenvalues of Γ _G_ # _µ_, under relatively
mild conditions on _G_ . The following lemma, proved in Appendix C.8 shows that
this is unfortunately not the case under just the assumption that the operator _G_ is
only Lipschitz continuous,


_**Proposition**_ **3.19** _**.**_ Let _µ ∈P_ 2 ( _X_ ) be any non-degenerate Gaussian measure; in
particular, the spectrum of Γ _µ_ may have arbitrarily fast spectral decay. Given any
sequence ( _γ_ _k_ ) _k∈_ N such that



_γ_ _k_ _≥_ 0 _,_ _∀_ _k ∈_ N _,_



_∞_
� _k_ [2] _γ_ _k_ _< ∞,_


_k_ =1



there exists a Lipschitz continuous map _G_ : _X →_ _L_ [2] ( _U_ ), such that the spectrum of
covariance operator Γ _G_ # _µ_ of the push-forward measure _G_ # _µ_ is given by ( _γ_ _k_ ) _k∈_ N .


Thus, the above proposition clearly shows that, in general, a nonlinear operator
_G_ can possibly destroy high rates of spectral decay for the covariance operator
associated with a push-forward measure, even if the eigenvalues of the covariance
operator associated with the underlying measure _µ_, decay exponentially rapidly.
However, there are special cases where one can indeed obtain fast rates of spectral
decay for the covariance operator associated with a push-forward measure. We
identify two of these special cases, with wide ranging applicability, below.


3.4.2. _Reconstruction error for linear operators G_ : _X →_ _Y ._ If the operator _G_ :
_X →_ _Y_ is a bounded _linear_ operator, then the spectrum of the push-forward
measure _G_ # _µ_ can be bounded above by the spectrum of _µ_ . More precisely, we have



_**Proposition**_ **3.20** _**.**_ Let _X_ be a separable Hilbert space. Let _µ ∈P_ ( _X_ ) be a
probability measure with finite second moment ´ _X_ _[∥][u][∥]_ _X_ [2] _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [. Let] _[ λ]_ [1] _[ ≥]_ _[λ]_ [2] _[ ≥]_



probability measure with finite second moment ´ _X_ _[∥][u][∥]_ _X_ [2] _[dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [. Let] _[ λ]_ [1] _[ ≥]_ _[λ]_ [2] _[ ≥]_

_. . ._ denote the eigenvalues of the covariance operator Γ _µ_ of _µ_, repeated according
to multiplicity. If _G_ : _X →_ _L_ [2] ( _U_ ) is a bounded linear operator, then for any _p ∈_ N,
there exists a _p_ -dimensional affine subspace _W_ 0 _⊂_ _L_ [2] ( _U_ ), such that
ˆ _L_ [2] _U_ _w_ inf _∈W_ 0 _[∥][w][ −]_ _[u][∥]_ _L_ [2] [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[u]_ [)] _[ ≤∥G∥]_ [2] [ �] _λ_ _k_ _._ (3.32)



_L_ [2] ( _U_ ) _w_ inf _∈W_ 0 _[∥][w][ −]_ _[u][∥]_ _L_ [2] [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[u]_ [)] _[ ≤∥G∥]_ [2] [ �]



_λ_ _k_ _._ (3.32)

_k>p_



Here _∥G∥_ (= Lip( _G_ )) denotes the operator norm of _G_ : _X →_ _L_ [2] ( _U_ ). If the affine
reconstruction/projection pair is chosen such that _R ◦P_ : _L_ [2] ( _U_ ) _→_ _L_ [2] ( _U_ ) is the
orthogonal projection onto _W_ 0, then the reconstruction error _E_ [�] _R_ is bounded by


�
_E_ _R_ _≤∥G∥_ ~~�~~ _λ_ _k_ _._
� _k>p_


The proof of this proposition is presented in Appendix C.9.


ERROR ESTIMATES FOR DEEPONETS 25


3.4.3. _Reconstruction error for operators with_ smooth _image._ Another class of operators for which we can readily estimate the spectral decay of the covariance operator associated with the push-forward measure are those operators that map into
smoother (more regular) subspaces of the target space.
As a concrete example, we set _U_ = T _[n]_ as the periodic torus. Let _N ∈_ N.
We denote by _P_ _N_ : _L_ [2] (T _[n]_ ) _→_ _L_ [2] (T _[n]_ ) the orthogonal Fourier projection onto the
Fourier basis


_P_ _N_ _u_ = � _u_ � _k_ **e** _k_ ( _x_ ) _,_

_|k|_ _∞_ _≤N_


where the sum is over _k_ = ( _k_ 1 _, . . ., k_ _n_ ) _∈_ Z _[n]_, such that


_|k|_ _∞_ := max
_j_ =1 _,...,n_ _[|][k]_ _[j]_ _[| ≤]_ _[N.]_


Note that the size of this set is _|{k ∈_ Z _[n]_ _| |k|_ _∞_ _≤_ _N_ _}|_ = (2 _N_ + 1) _[n]_ . Hence, with _p_
degrees of freedom, we can represent _P_ _N_ for



( _p_ 1 _/n_ _−_ 1)
_p ≥_ (2 _N_ + 1) _[n]_ _⇒_ _N ≤_
� 2



_≤_ _p_ [1] _[/n]_ _._ (3.33)
�



Given _p ∈_ N, let _P_ : _L_ [2] (T _[n]_ ) _→_ R _[p]_ be a mapping encoding all Fourier coefficients
with _|k|_ _∞_ _≤_ _N_, where _N ≤_ _p_ [1] _[/n]_ is the largest integer satisfying (3.33). Let
_R_ Fourier : R _[p]_ _→_ _L_ [2] (T _[n]_ ) denote the corresponding Fourier reconstruction (3.29), so
that _R_ Fourier _◦P_ : _L_ [2] (T _[n]_ ) _→_ _L_ [2] (T _[N]_ ) satisfies _R_ Fourier _◦P_ = _P_ _N_ . It is well-known
that for _u ∈_ _H_ _[s]_ (T _[n]_ ), we have


1
_∥P_ _N_ _u −_ _u∥_ _L_ 2 (T _n_ ) _≤_ _N_ _[s]_ _[∥][u][∥]_ _[H]_ _[s]_ [(][T] _[n]_ [)] _[.]_



Hence the resulting reconstruction error _E_ [�] _R_ _F ourier_ is given by,


1

ˆ _Y_ _∥R_ Fourier _◦P −_ Id _∥_ [2] _L_ [2] (T) _[d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))] _[ ≤]_ _N_ [2] _[s]_ ˆ _Y_ _∥u∥_ [2] _H_ _[s]_



1
_Y_ _∥R_ Fourier _◦P −_ Id _∥_ [2] _L_ [2] (T) _[d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))] _[ ≤]_ _N_ [2] _[s]_



ˆ



_∥u∥_ [2] _H_ _[s]_ (T) _[d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))]
_Y_



1

=
_N_ [2] _[s]_



_∥G_ ( _u_ ) _∥_ [2] _H_ _[s]_ (T) _[dµ]_ [(] _[u]_ [)] _[.]_

ˆ _X_



This elementary calculation leads to the following result,


_**Proposition**_ **3.21** _**.**_ If _G_ defines a Lipschitz mapping _G_ : _X →_ _H_ _[s]_ (T _[n]_ ), for some
_s >_ 0, with


_∥G_ ( _u_ ) _∥_ _H_ [2] _[s]_ _[ dµ]_ [(] _[u]_ [)] _[ ≤]_ _[M <][ ∞][,]_

ˆ _X_


then we have the following estimate on the reconstruction error:


�
_E_ _R_ Fourier _≤_ _CMp_ _[−][s/n]_ _,_ (3.34)


where _C_ = _C_ ( _n, s_ ) _>_ 0 depends on _n_, _s_, but is independent of _p_ .


Combining (3.34) and (3.31), and setting _ϵ_ = _p_ _[−][s/n]_ in (3.31) immediately implies
Theorem 3.7, which has already been stated in the overview Section 3.2.2. Theorem
3.7 provides a complexity and error estimate for the trunk net approximation under
the assumptions of the previous proposition.


26 ERROR ESTIMATES FOR DEEPONETS


3.5. **Bounds on the encoding error** (3.1) **.** Our aim in this section is to bound
the encoding error (3.1), associated with the DeepONet (2.10). This error _does not_
_depend_ on the nonlinear operator _G_, but only on the underlying probability measure
_µ_ . Key objectives of our analysis are to determine suitable choices of sensors for a
fixed _m_ as well as to find the appropriate form of a decoder _D_ in order to minimize
the encoding error (3.1). We first recall the lower bound of Theorem 3.8, on the
encoding error:



( _E_ [�] _E_ ) [2] =
ˆ



� _λ_ _k_ _,_ (3.35)

_k>m_



_∥D ◦E_ ( _u_ ) _−_ _u∥_ [2] _L_ [2] ( _D_ ) _[dµ]_ [(] _[u]_ [)] _[ ≥]_ �
_X_



if _D_ : R _[m]_ _→_ _X_ is a _linear_ decoder, _µ_ is mean-zero and _λ_ _k_ refers to the _k_ -th
eigenvalue of the covariance operator Γ _µ_ = ´ _X_ _[u][ ⊗]_ _[u dµ]_ [(] _[u]_ [). The proof presented in]

Appendix C.10 relies on Theorem 3.10 and from this proof, we can readily see that
the restriction on the zero mean of the measure _µ_ can be relaxed by using an affine
decoder and Theorem 3.12. We note in passing that the above bound (3.35) in fact
holds for any encoder (not necessarily linear) as long as the decoder is linear.
Our next aim is to derive _upper_ bounds on the encoding error. To illustrate our
main ideas, we will restrict our discussion to the case _X_ = _L_ [2] ( _D_ ). The results are
readily extended to more general spaces, such as Sobolev spaces _X_ = _H_ _[s]_ ( _D_ ) for
_s >_ 0. We fix a probability measure _µ_ on _L_ [2] ( _D_ ), and write the covariance operator
Γ as an eigenfunction decomposition



Γ =



_∞_
� _λ_ _ℓ_ ( _φ_ _ℓ_ _⊗_ _φ_ _ℓ_ ) _,_


_ℓ_ =1



where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ are the decreasing eigenvalues, and such that the _φ_ _ℓ_ are an
orthonormal basis of _L_ [2] ( _D_ ). We will assume that all _φ_ _ℓ_ are continuous functions, so
that point-wise evaluation of _φ_ _ℓ_ makes sense. Note that since the _φ_ _ℓ_ are orthonormal
in _L_ [2] ( _D_ ), they are also linearly independent as elements in _C_ ( _D_ ).
Assume now that for some _M ∈_ N, there exist sensors _X_ 1 _, . . ., X_ _M_ _∈_ _D_, such
that the matrix Φ _M_ _∈_ R _[m][×][M]_ with entries


[Φ _M_ ] _ij_ = [ _φ_ _i_ ( _X_ _j_ )] _,_ for _i_ = 1 _, . . ., m_, _j_ = 1 _, . . ., M_, (3.36)


has full rank, i.e.


det(Φ _M_ Φ _[T]_ _M_ [)] _[ ̸]_ [= 0] _[.]_ (3.37)


Then we can define a “ _projection_ ” onto _φ_ 1 _, . . ., φ_ _m_ by



_E_ _D_
_u_ ( _x_ ) _→_ ( _u_ ( _X_ 1 ) _, . . ., u_ ( _X_ _m_ )) _→_



_m_
� [Φ _[†]_ _M_ []] _[kj]_ _[u]_ [(] _[X]_ _[j]_ [)] _[φ]_ _[k]_ [(] _[x]_ [)] _[,]_ (3.38)

_j,k_ =1



where


_−_ 1
Φ _[†]_ _M_ [:=] �Φ _M_ Φ _[T]_ _M_ � Φ _M_ _,_ (3.39)


denotes the pseudo-inverse of Φ _M_ _∈_ R _[m][×][M]_ . Written in somewhat simpler matrix/vectormultiplication notation, we might write this projection as



_u_ ( _x_ ) _�→_ Φ _[†]_ _M_ _[u]_ [(] _**[X]**_ [)] _[,]_ _**[ ϕ]**_ [(] _[x]_ [)]
� � _ℓ_ [2] (R _[m]_ ) _[,]_



where _**X**_ := ( _X_ 1 _, . . ., X_ _M_ ), _u_ ( _**X**_ ) = ( _u_ ( _X_ 1 ) _, . . ., u_ ( _X_ _M_ )) and _**ϕ**_ ( _x_ ) = ( _φ_ 1 ( _x_ ) _, . . ., φ_ _m_ ( _x_ )).
Note that for _u_ ( _x_ ) = _φ_ _i_ ( _x_ ), 1 _≤_ _i ≤_ _m_, we have



_M_
�[Φ _M_ ] _ℓ,j_ _φ_ _i_ ( _X_ _j_ ) =

_j_ =1



_M_
�[Φ _M_ ] _ℓ,j_ [Φ _M_ ] _i,j_ = [Φ _M_ Φ _[T]_ _M_ []] _[ℓ,i]_ _[.]_

_j_ =1


ERROR ESTIMATES FOR DEEPONETS 27



Hence



_k,ℓ_ _[φ]_ _[k]_ [(] _[x]_ [)]



_M_
�



_φ_ _i_ ( _x_ ) _�→_


=


=



_m_
�

_k,ℓ_ =1



�[Φ _M_ ] _ℓ,j_ _φ_ _i_ ( _X_ _j_ ) �(Φ _M_ Φ _[T]_ _M_ [)] _[−]_ [1] [�]

_j_ =1



_m_
�



_k,ℓ_ _[φ]_ _[k]_ [(] _[x]_ [)]



� [Φ _M_ Φ _[T]_ _M_ []] _[ℓ,i]_ �(Φ _M_ Φ _[T]_ _M_ [)] _[−]_ [1] [�]

_k,ℓ_ =1



_m_
� _δ_ _ik_ _φ_ _k_ ( _x_ ) = _φ_ _i_ ( _x_ ) _._


_k_ =1



_m_
�



So the map (3.38) clearly provides a projection onto span( _φ_ _j_ ; _j_ = 1 _, . . ., m_ ).
Next, we have the following Lemma, provided in Appendix C.11, which characterizes the encoding error (3.1), associated with the encoder/decoder pair given by
(3.38).


_**Lemma**_ **3.22** _**.**_ Let _µ ∈P_ 2 ( _L_ [2] ( _D_ )) be a measure with finite second moments, i.e.
such that ´ _L_ [2] ( _D_ ) _[∥][u][∥]_ _L_ [2] [2] _[ dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [. Let Φ] _[M]_ [ be given by (3.36), and assume that]

the non-singularity condition (3.37) holds. Then the encoding error _E_ [�] _E_ for the pair
_E_, _D_, defined by (3.38) can be written as

( _E_ [�] _E_ ) [2] = ( _E_ [�] aliasing ) [2] + ( _E_ [�] _⊥_ ) [2] _,_


where



( _E_ [�] _⊥_ ) [2] =
ˆ



� _λ_ _ℓ_ _,_

_ℓ>m_



_∥P_ _m_ _[⊥]_ _[u][∥]_ [2] _L_ [2] _[ dµ]_ [(] _[u]_ [) =] �
_X_



with _P_ _m_ _[⊥]_ [:] _[ L]_ [2] _[ →]_ _[L]_ [2] [ the orthogonal projection onto the orthogonal complement of]
span( _φ_ 1 _, . . ., φ_ _m_ ), where _φ_ 1 _, φ_ 2 _, . . ._ denote a basis of orthonormal eigenfunctions of
the covariance operator Γ _µ_ of _µ_, with corresponding eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._,
and
( _E_ [�] aliasing ) [2] = � _λ_ _ℓ_ _∥_ Φ _[†]_ _M_ _[φ]_ _[ℓ]_ [(] _**[X]**_ [)] _[∥]_ _ℓ_ [2] [2] _[.]_ (3.40)

_ℓ>m_


Thus, the above Lemma 3.22 provides us a strategy to bound the encoding error
as long as the non-singularity condition (3.37) on the matrix (3.36) is satisfied.
Moreover, one component of the encoding error is completely specified in terms
of the spectral decay of the associated covariance operator. However, the other
component measures the error due to _aliasing_, with the notation being motivated
by an example from Fourier analysis, that is considered in the following subsection.
Hence, bounding the aliasing error and checking the validity of the non-singularity
condition (3.37) require us to specify the location of sensors.
Given the form of the matrix (3.36), it would make sense to relate the sensor
locations with the eigenfunctions of the covariance operator Γ. However in general,
we may not have any information on these eigenfunctions _φ_ 1 _, . . ., φ_ _m_, and hence,
finding suitable sensors _X_ 1 _, . . ., X_ _M_ satisfying the non-singularity condition (3.37)
might be a very difficult task. Instead, we propose to choose them iid randomly in
_D_, allowing for _M ≥_ _m_, and study the corresponding random encoder


_E_ ( _u_ ) = ( _u_ ( _X_ 1 ) _, . . ., u_ ( _X_ _M_ )) _,_


with associated decoder _D_ given by (3.38). Note that the decoder _D_ is merely
used in the analysis of the encoding error, but its explicit form is not needed,
when DeepONets are used in practice. In fact, _the use of random sensors and the_
_simplicity of the resulting encoder could constitute one of the key benefits of the_
_DeepONets_ .


28 ERROR ESTIMATES FOR DEEPONETS


More precisely, in the following we fix a probability space (Ω _,_ Prob), and a sequence of iid random variables _{X_ _k_ _}_ _k∈_ N :


_ω �→_ ( _X_ 1 ( _ω_ ) _, X_ 2 ( _ω_ ) _, X_ 3 ( _ω_ ) _, . . ._ ) _,_ ( _ω ∈_ Ω) _,_


such that _X_ _k_ _∼_ Unif( _D_ ) for all _k ∈_ N. For any _M ∈_ N, and _ω ∈_ Ω, we can then
study the corresponding encoder _E_ : _C_ ( _D_ ) _→_ R _[M]_,


_E_ ( _u_ ; _X_ 1 ( _ω_ ) _, . . ., X_ _M_ ( _ω_ )) = ( _u_ ( _X_ 1 ( _ω_ )) _, . . ., u_ ( _X_ _M_ ( _ω_ ))) _._


As is common in probability theory, we will usually suppress the argument _ω_, and
write, e.g., _X_ 1 instead of _X_ 1 ( _ω_ ).
Hence, the matrix Φ _M_ = [ _φ_ _i_ ( _X_ _j_ )] _i,j_ _∈_ R _[m][×][M]_ is a random matrix, depending
on _X_ 1 _, . . ., X_ _M_ . In order for the resulting decoder in (3.38) to be well-defined, we
need to show that there is a non-zero probability that Φ _M_ Φ _[T]_ _M_ [is non-singular, for]
sufficiently large _M_ .
Moreover, by Lemma 3.22, the aliasing error (3.40) depends on


_∥_ Φ _[†]_ _M_ _[φ]_ _[ℓ]_ [(] _**[X]**_ [)] _[∥]_ _ℓ_ [2] [2] _[ ≤∥]_ [Φ] _[†]_ _M_ _[∥]_ _ℓ_ [2] [2] _→ℓ_ [2] _[∥][φ]_ _[ℓ]_ [(] _**[X]**_ [)] _[∥]_ [2] _ℓ_ [2] _[.]_


We note that



_|D|_
_∥_ Φ _[†]_ _M_ _[∥]_ _ℓ_ [2] [2] _→ℓ_ [2] [ =]



_|D|_
_Mσ_ min _M_
~~�~~



_D|_ _,_

_M_ [Φ] _[M]_ [Φ] _M_ _[T]_
~~�~~



_|D|_
can be written in terms of the smallest singular value, i.e. _σ_ min _M_ [Φ] _[M]_ [Φ] _M_ _[T]_, of
� �



the rescaled matrix _[|][D][|]_



the rescaled matrix _M_ [Φ] _[M]_ [Φ] _M_ _[T]_ [(the reason for introducing the rescaling] _[ |][D][|][/M]_ [ will]

be explained below). Hence, to bound the aliasing error and the overall encoding
error, we will need to provide a lower bound (with high probability) on this smallest
singular value. We investigate these two issues in the following.
First, we note that for iid random sensors _X_ 1 _, . . ., X_ _M_, by our definition of Φ _M_,
we have


_M_

� _|MD|_ [Φ] _[M]_ [Φ] _M_ _[T]_ � = _[|]_ _M_ _[D][|]_ � _φ_ _k_ ( _X_ _j_ ) _φ_ _ℓ_ ( _X_ _j_ ) _,_



_M_ [Φ] _[M]_ [Φ] _M_ _[T]_



�



= _[|][D][|]_

_M_

_k,ℓ_



_M_



_M_
�



� _φ_ _k_ ( _X_ _j_ ) _φ_ _ℓ_ ( _X_ _j_ ) _,_

_j_ =1



and the last sum can be interpreted as a Monte-Carlo estimate of


_|D|_ E[ _φ_ _k_ ( _X_ ) _φ_ _ℓ_ ( _X_ )] = _φ_ _k_ ( _x_ ) _φ_ _ℓ_ ( _x_ ) _dx_ = _δ_ _kℓ_ _._
ˆ _D_


For such Monte-Carlo estimates, we can rely on well-known bounds from probability
theory and have the following lemma on the bounds for the smallest singular value:


_**Lemma**_ **3.23** _**.**_ For any _m ∈_ N, denote


_ω_ _m_ := max
_k≤m_ _[∥][φ]_ _[k]_ _[∥]_ _[L]_ _[∞]_ _[.]_


Let _X_ 1 _, . . ., X_ _M_ _∼_ Unif( _D_ ) be iid random variables. Define Φ _M_ by (3.36). Then
we have



� 2 [�]



Prob _σ_ min
�



_|D|_
� _M_ [Φ] _[M]_ [Φ] _M_ _[T]_



_<_ 1 _−_ [1]
� ~~_√_~~ 2



_≤_ 2 _m_ [2] exp
�



_M_

� _−_ � _|D|ω_ _m_ [2] _m_



_._ (3.41)



This lemma is proved using the well-know Hoeffding’s inequality and the proof
is presented in Appendix C.12. A direct application of the bound (3.41) allows us
to bound the aliasing error (3.40) in the following,


ERROR ESTIMATES FOR DEEPONETS 29



_**Lemma**_ **3.24** _**.**_ Let _X_ 1 _, . . ., X_ _M_ _∼_ Unif( _D_ ) be iid random variables, uniform on _D_ .
Let _µ ∈P_ ( _L_ [2] ( _D_ )) be a probability measure concentrated on continuous functions.
Let _λ_ 1 _, λ_ 2 _, . . .,_ denote the eigenvalues of the covariance operator Γ _µ_ (3.14) of _µ_, with
associated orthonormal eigenbasis _φ_ 1 _, φ_ 2 _, . . ._ . The aliasing error (3.40), resulting
from this random choice of sensors, is bounded by


_M_

1
ˆ _∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _[ dµ]_ [(] _[u]_ [)] _[ ≤]_ _|D|_ _[T]_ _M_ � �� _λ_ _ℓ_ _|φ_ _ℓ_ ( _X_ _j_ ) _|_ [2] � _._

_[|][D][|]_



�� _ℓ>m_



_λ_ _ℓ_ _|φ_ _ℓ_ ( _X_ _j_ ) _|_ [2]
_ℓ>m_ �



1


_D|_ _M_

_M_ [Φ] _[M]_ [Φ] _M_ _[T]_
~~�~~ _[|][D][|]_



_|D|_
_σ_ min _M_
~~�~~



_._



_M_



_M_
�

_j_ =1



Denote _ω_ _m_ := max _k≤m_ _∥φ_ _k_ _∥_ _L_ _∞_ . Then, with probability



2 [�]


_,_

�



Prob _≥_ 1 _−_ 2 _m_ [2] exp



_M_

� _−_ � _|D|ω_ _m_ [2] _m_



we have

_∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _[ dµ]_ [(] _[u]_ [)] _[ ≤]_
ˆ



_√_ 2 _|D|_

~~_√_~~ 2 _−_ 1



_√_



2 _−_ 1



� _λ_ _ℓ_ _∥φ_ _ℓ_ _∥_ _L_ [2] _[∞]_ _[.]_

_ℓ>m_



Moreover, let _κ ∈_ N, and define _M_ = _M_ ( _m_ ) = _⌈κ|D|mω_ _m_ [2] [log(] _[m]_ [)] _[⌉]_ [. Then, with]
probability

2
Prob _≥_ 1 _−_ _m_ _[κ][−]_ [2] _[,]_

it holds that for _M_ iid uniformly chosen random sensors _X_ 1 _, . . ., X_ _M_ _∈_ _D_, the
encoder
_E_ : _C_ ( _D_ ) _→_ R _[M]_ _,_ _u_ ( _x_ ) _�→_ ( _u_ ( _X_ 1 ) _, . . ., u_ ( _X_ _M_ )) _,_
possesses a decoder _D_ given by (3.38), such that



( _E_ [�] _E_ ) [2] _≤_



_√_



_√_ 2 _|D|_

~~_√_~~ 2 _−_ 1



2 _−_ 1



� _λ_ _ℓ_ (1 + _∥φ_ _ℓ_ _∥_ _L_ [2] _[∞]_ [)] _[.]_ (3.42)

_ℓ>m_



A direct consequence of this lemma is Theorem 3.9, stated in the overview Section 3.2.3, and whose proof is provided in appendix C.13. Theorem 3.9 shows the
remarkable result that _randomly chosen sensor points_ can lead to an optimal (up
to a log) decay of the encoding error (3.1), corresponding to the DeepONet (2.10).


3.5.1. _Examples._ In this section, we seek to illustrate the estimates on the encoding
error for concrete prototypical examples. We recall that the encoding error (3.1)
is independent of the operator _G_, depending only on the probability measure _µ ∈_
_P_ ( _X_ ). We assume that _X_ = _L_ [2] ( _D_ ) in the following. Assuming furthermore that
_µ_ has finite second moments ´ _L_ [2] ( _D_ ) _[∥][u][∥]_ [2] _[ dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [, then it is well-known Stuart]

(2010) that by the Karhunen-Lo`eve expansion, we can write _µ_ as the law of a
random variable _u_ = _u_ ( _·_ ; _**Z**_ ), of the form



_λ_ _ℓ_ _Z_ _ℓ_ _φ_ _ℓ_ _,_ _**Z**_ = ( _Z_ 1 _, Z_ 2 _, . . ._ ) _,_ (3.43)



_u_ ( _·_ ; _**Z**_ ) = ~~_u_~~ +



_∞_
�


_ℓ_ =1



�



where ~~_u_~~ _∈_ _L_ [2] ( _D_ ) is the mean, _φ_ 1 _, φ_ 2 _, · · · ∈_ _L_ [2] ( _D_ ) are an orthonormal basis consisting of eigenfunctions of the covariance operator Γ _µ_ of _µ_, _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote the
corresponding eigenvalues, and _Z_ 1 _, Z_ 2 _, . . ._ are real-valued random variables satisfying
E[ _Z_ _ℓ_ ] = 0 _,_ E[ _Z_ _k_ _Z_ _ℓ_ ] = _δ_ _kℓ_ _,_ _∀_ _k, ℓ_ _∈_ N _._

In practice, the probability measure _µ_ is often specified as the law of an expansion
of the form (3.43) Stuart (2010) (see e.g. examples 5 and 6 of Lu et al. (2019)).
This provides a very convenient method to sample from the measure _µ_, which is
defined on an infinite dimensional space _X_ .


30 ERROR ESTIMATES FOR DEEPONETS



A particularly important class of measures on infinite-dimensional spaces are the
so-called _Gaussian measures_ Stuart (2010). A Gaussian measure _µ ∈P_ ( _L_ [2] ( _D_ )) is
uniquely characterized by its mean ~~_u_~~ = ´ _L_ [2] _D_ _[u dµ]_ [(] _[u]_ [)] _[ ∈]_ _[L]_ [2] [(] _[D]_ [), and its covariance]



uniquely characterized by its mean ~~_u_~~ = ´ _L_ [2] ( _D_ ) _[u dµ]_ [(] _[u]_ [)] _[ ∈]_ _[L]_ [2] [(] _[D]_ [), and its covariance]

operator Γ = ´ _L_ [2] _D_ [(] _[u][ ⊗]_ _[u]_ [)] _[ dµ]_ [(] _[u]_ [), which may, e.g., be expressed in terms of a]



operator Γ = ´ _L_ [2] ( _D_ ) [(] _[u][ ⊗]_ _[u]_ [)] _[ dµ]_ [(] _[u]_ [), which may, e.g., be expressed in terms of a]

covariance integral kernel _k_ ( _x, x_ _[′]_ ) _∈_ _L_ [2] ( _D × D_ ), and through which the covariance
operator Γ is defined by integration against _k_ ( _x, x_ _[′]_ ):



Γ : _L_ [2] ( _D_ ) _→_ _L_ [2] ( _D_ ) _,_ _u_ ( _x_ ) _�→_ _k_ ( _x, x_ _[′]_ ) _u_ ( _x_ _[′]_ ) _dx_ _[′]_ _._ (3.44)
ˆ _D_



3.5.2. _Encoding error for a particular Gaussian measure._ For this concrete example, we consider a Gaussian measure that was used in the context of DeepONets
in Lu et al. (2019). For definiteness and simplicity of exposition, we consider the
one-dimensional periodic case by setting _D_ = T = [0 _,_ 2 _π_ ] and consider a Gaussian
measure _µ_, defined on it with the periodization,



_−|x −_ _x_ _′_ _−_ _h|_ 2
_h_ � _∈_ 2 _π_ Z exp � 2 _ℓ_ [2]



_k_ _p_ ( _x, x_ _[′]_ ) := �



2 _ℓ_ [2]



_._ (3.45)
�



of the frequently used covariance kernel

_−|x −_ _x_ _′_ _|_ 2
_k_ ( _x, x_ _[′]_ ) = exp
� 2 _ℓ_ [2]



_._
�



We have the following result, proved in Appendix C.14 on the encoder and the
encoding error for this Gaussian measure,


_**Lemma**_ **3.25** _**.**_ Let _µ_ be given by the law of the Gaussian process with covariance
kernel _k_ _p_ ( _x, x_ _[′]_ ) (3.45). Let

_x_ _j_ = [2] _[π]_ [(] _[j][ −]_ [1][)] _,_

_m_

denote equidistant points on [0 _,_ 2 _π_ ] for _m_ = 2 _K_ + 1, _K ∈_ N. Define the _pseudo-_
_spectral_ encoder _E_ : _L_ [2] (T) _→_ R _[m]_ by _E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )), then a decoder
_D_ : R _[m]_ _→_ _L_ [2] (T) is given via the discrete Fourier transform:



_K_

_D_ ( _u_ 1 _, . . ., u_ _m_ )( _x_ ) = � _u_ � _k_ _e_ _[ikx]_ _,_ (3.46)


_k_ = _−K_



where



�
_u_ _k_ := [1]

_m_



_m_
� _u_ _j_ _e_ _[−]_ [2] _[πijk]_ _._

_j_ =1



The encoding error (3.1) for the resulting DeepONet (2.10) satisfies,



�
_E_ _E_ _≤_
�



2 ~~�~~ _λ_ _k_ _≤_

_|k|>⌊m/_ 2 _⌋_



2 ~~�~~



_⌊m/_ 2 _⌋ℓ_
4 _π_ erfc

� ~~�~~ ~~_√_~~ 2



_._ (3.47)
~~�~~



Here _λ_ _k_ = _√_ 2 _π ℓ_ exp( _−_ ( _ℓk_ ) [2] _/_ 2) are the eigenvalues of the covariance operator (3.44)

for the Gaussian measure and erfc denotes the complementary error function, i.e.



2
erfc( _x_ ) :=
~~_√π_~~



_∞_

_e_ _[−][t]_ [2] _dt_

ˆ _x_



By an asymptotic expansion of erfc( _x_ ), the above lemma implies that there exists
a constant _C >_ 0, such that


�
_E_ _E_ _≤_ _C_ exp � _−_ ( _⌊m/_ 2 _⌋ℓ_ ) [2] _/_ 4� ≲ _C_ exp � _−γm_ [2] _ℓ_ [2] [�] _,_ _∀_ _γ <_ [1] (3.48)

16 _[,]_


i.e., the encoding error decays _super-exponentially_ in this case.


ERROR ESTIMATES FOR DEEPONETS 31


Thus, we have a very fast decay of the encoding error with a _pseudo-spectral en-_
_coder_ for this Gaussian measure. In view of the earlier discussion in this subsection,
it is natural to examine what happens if the encoder was a _random encoder_, i.e.
based on pointwise evaluation at uniformly distributed random points in [0 _,_ 2 _π_ ].
Given Lemma 3.24, one would expect a similar super-exponential decay (modulo a
logarithmic correction). This is indeed the case as shown in the following Lemma
(proved in appendix C.15),


_**Lemma**_ **3.26** _**.**_ Let _µ_ be a Gaussian measure on the one-dimensional periodic torus
_D_ = T = [0 _,_ 2 _π_ ], characterized by the covariance kernel (3.45). Let _X_ 1 _, · · ·, X_ _M_
be uniformly distributed random sensors on _D_ . There exists a constant _γ >_ 0,
such that with probability 1, the encoding error (3.1) corresponding to the random
encoder (pointwise evaluations at random sensors) is bounded by,



�
_E_ _E_ ≲ exp _−_ _[γ][M]_ [ 2] _[ℓ]_ [2]
� log( _M_ ) [2]



_._ (3.49)
�



Comparing the bounds (3.48) and (3.49), we see that the random encoder also
decays super-exponentially (up to a log). However, as remarked before, no information about the underlying measure is used in defining the random encoder.


3.5.3. _Encoding error for a parametrized measure._ As a second example, we define
the underlying measure _µ ∈P_ ( _L_ [2] ( _D_ )), as the _law_ of its Karhunen-Loeve expansion
(3.43) with the following ansatz on the resulting random field,



_u_ ( _x_ ; _Y_ ) = ~~_u_~~ ~~(~~ _x_ ) +



_∞_
� _Y_ _ℓ_ _α_ _ℓ_ _ψ_ _ℓ_ ( _x_ ) _,_ (3.50)


_ℓ_ =1



with _α_ _ℓ_ _>_ 0, ~~_u_~~ _, ψ_ _ℓ_ _∈_ _L_ [2] ( _D_ ) are such that [�] _[∞]_ _ℓ_ =1 _[α]_ _[ℓ]_ _[∥][ψ]_ _[ℓ]_ _[∥]_ _[L]_ [2] _[ <][ ∞]_ [is bounded, and]
where _Y_ _ℓ_ _∈_ [ _−_ 1 _,_ 1] are mean-zero random variables, distributed according to some
measure _dρ_ ( _**y**_ ) on _**y**_ _∈_ [ _−_ 1 _,_ 1] [N] . In this case, the series in (3.50) converges uniformly
in _L_ [2] ( _D_ ) for any _**y**_ = ( _y_ _j_ ) _j∈_ N _∈_ [ _−_ 1 _,_ 1] [N], and



_∥u_ ( _·_ ; _**y**_ ) _∥_ _L_ 2 _≤_ ~~_∥u∥_~~ _L_ 2 +



_∞_
� _α_ _ℓ_ _∥ψ_ _ℓ_ _∥_ _L_ 2 _._


_ℓ_ =1



For the sake of definiteness, we shall only discuss a prototypical case, where the
(spatial) domain _D_ is either periodic, i.e. _D_ = T _[d]_, or rectangular [1] _D_ = [0 _,_ 2 _π_ ] _[d]_,
and the expansion functions are given by the trigonometric basis _{_ **e** _k_ _}_ _k∈_ Z _d_, indexed
by _k ∈_ Z _[d]_ (check notation from Appendix A). The underlying ideas readily extend
to more general choices of basis functions _ψ_ _ℓ_ . To be definite, we will assume that
the random field _u_ ( _x_ ; _**y**_ ) is expanded as

_u_ ( _x_ ; _**y**_ ) = ~~_u_~~ ~~(~~ _x_ ) + � _y_ _k_ _α_ _k_ **e** _k_ ( _x_ ) _,_ (3.51)

_k∈_ Z _[d]_


where ~~_u_~~ _∈_ _C_ _[∞]_ ( _D_ ), and such that there exist constants _C_ _α_ _>_ 0, _ℓ>_ 0, such that

_|α_ _k_ _| ≤_ _C_ _α_ exp( _−ℓ|k|_ _∞_ ) _,_ _∀_ _k ∈_ Z _[d]_ _._ (3.52)


We furthermore assume that the _Y_ _k_ _∈_ [ _−_ 1 _,_ 1], _k ∈_ Z _[d]_, are centered random variables, implying that E[ _u_ ] = ~~_u_~~ ~~.~~ We let _µ ∈P_ ( _L_ [2] ( _D_ )) denote the law of the random
variable _u_ ( _·_ ; ( _Y_ _k_ ) _k∈_ Z _d_ ). By the assumed decay (3.52), we have supp( _µ_ ) _⊂_ _C_ _[∞]_ ( _D_ ).
We remark that one can also readily consider the setup where the coefficients _α_ _k_
decay at an algebraic rate. Note that the expansion (3.51) appears to be similar to
that of the Karhunen-Loeve expansion of the Gaussian measure, considered in the


1 Generalization to a more general domain of the form _D_ = [�] _di_ =1 [[] _[α]_ _[i]_ _[, β]_ _[i]_ [], for] _[ α]_ _[i]_ _[ < β]_ _[i]_ [ is straight-]
forward.


32 ERROR ESTIMATES FOR DEEPONETS


last section. The main difference lies in the fact that _Y_ _k_ are no longer assumed to
be normally distributed, nor necessarily iid.
Given _µ_ as the law of random field (3.51) as the underlying measure, we need to
construct a suitable encoder and decoder and then estimate the resulting encoding
error (3.1). To this end, we adapt the pseudo-spectral encoder and the resulting
discrete Fourier transform based decoder (3.46) to the current setup. For multiindices _i_ = ( _i_ 1 _, . . ., i_ _d_ ) _∈{_ 0 _, . . .,_ 2 _N_ _}_ _[d]_, let



2 _π_ _**i**_ 2 _πi_ 1 2 _πi_ 2 2 _πi_ _d_
_x_ _i_ :=
2 _N_ + 1 [=] � 2 _N_ + 1 _[,]_ 2 _N_ + 1 _[, . . .,]_ 2 _N_ + 1



_,_
�



denote the (2 _N_ + 1) _[d]_ points on [0 _,_ 2 _π_ ] _[d]_ on an equidistant cartesian grid with grid
size 2 _π/_ (2 _N_ + 1). In the following, we will denote by _I_ _N_ the index set


_I_ _N_ := _{i_ = ( _i_ 1 _, . . ., i_ _d_ ) _| i_ _r_ _∈{_ 0 _, . . .,_ 2 _N_ _}, ∀r_ = 1 _, . . ., d}._


Define the encoder _E_ : _C_ ( _D_ ) _→_ R _[m]_, for _m_ = (2 _N_ + 1) _[d]_ = _|I_ _N_ _|_, by


_E_ ( _u_ ) = ( _u_ ( _x_ _i_ )) _i∈I_ _N_ _._ (3.53)


To construct a suitable decoder _D_ : R _[m]_ _→_ _L_ [2] ( _D_ ) corresponding to the encoder _E_ above, we will first recover (an approximation of) the coefficients _y_ _k_ from
the encoded values _E_ ( _u_ ( _·_ ; _**y**_ )). This can be achieved by the following sequence of
mappings:

(1) Subtract the mean ~~_u_~~ : Define


_M_ : R _[m]_ _→_ R _[m]_ _,_ ( _u_ _i_ ) _i∈I_ _N_ _�→_ ( _u_ _i_ _−_ ~~_u_~~ ~~(~~ _x_ _i_ )) _i∈I_ _N_ _._ (3.54)


(2) Discrete Fourier transform: Define



� _u_ � _i_ **e** _k_ ( _x_ _i_ )

_i∈I_ _N_



�



_k∈K_ _N_



_,_ (3.55)



_FT_ : R _[m]_ _→_ R _[m]_ _,_ ( _u_ � _i_ ) _i∈I_ _N_ _�→_



(2 _π_ ) _[d]_

_|I_ _N_ _|_

�



�



where the set of Fourier wavenumbers _k ∈K_ _N_ is given by

_K_ _N_ = � _k_ = ( _k_ 1 _, . . ., k_ _d_ ) �� _k ∈_ Z _d_ _, −N ≤_ _k_ _j_ _≤_ _N ∀j_ � _._



Note that we have _FT_ ( **e** _k_ ) = **e** _k_ for all _k ∈K_ _N_ .
(3) Approximation of ( _y_ 1 _, . . ., y_ _m_ ): Given the discrete Fourier coefficients � _u_ _k_ =

�
_FT_ ( _u_ ) _k_, _k ∈K_ _N_, we define (with _J_ = Z _[d]_ )

 _Y_ : R _[m]_ _→_ [ _−_ 1 _,_ 1] _[J]_ _,_

� � (3.56)

 _Y_ [�] shrink _Y_



_Y_ : R _[m]_ _→_ [ _−_ 1 _,_ 1] _[J]_ _,_



( _u_ � _k_ ) _k∈K_ _N_ _�→_ ( _Y_ [�] _j_ ) _j∈J_ := �shrink � _Y_ � _j_ ��



(3.56)
_j∈J_ _[,]_







where



_Y_ � _j_ :=



�
_u_ _k_ _/α_ _k_ _,_ ( _j_ = _k ∈K_ _N_ ) _,_
(3.57)
�0 _,_ (otherwise) _,_



and the shrink-operator shrink : R _→_ [ _−_ 1 _,_ 1] ensures that _Y_ [�] _k_ _∈_ [ _−_ 1 _,_ 1] for
all _k_ :



shrink( _Y_ ) =



_Y,_ _|Y | ≤_ 1 _,_
� _Y/|Y |,_ _|Y | >_ 1 _._



Finally, the decoder _D_ : R _[m]_ _→_ _L_ [2] (T _[d]_ ) is defined by the composition

_D_ ( _u_ _i_ ) := _u_ ( _·,_ _Y_ [�] ( _u_ _i_ )) _,_ (3.58)


where _u_ ( _·_ ; _Y_ ) is given by (3.51) and


�
_Y_ ( _u_ _i_ ) := ( _Y ◦FT ◦M_ )( _u_ _i_ ) _._ (3.59)


ERROR ESTIMATES FOR DEEPONETS 33


It is easy to see that the main difference between the above encoder/decoder pair
and the pseudo-spectral encoder/decoder of (3.46) is the non-linear shrink operation
in (3.56), which we introduce to ensure that _Y_ [�] _∈_ [ _−_ 1 _,_ 1] [N] . As a consequence of this
observation, we find that the usual error estimates for pseudo-spectral methods
imply similar error estimates for the encoding error _∥D ◦E_ ( _u_ ) _−_ _u∥_ _L_ 2 . For instance,
we have the following proposition (proved in Appendix C.16),


_**Proposition**_ **3.27** _**.**_ Let _s > d/_ 2, and assume that _u ∈_ _H_ _[s]_ ( _D_ ). There exists a
constant _C_ = _C_ ( _s, d_ ) _>_ 0, such that the encoder/decoder pair ( _E, D_ ) defined by
(3.53) and (3.58) satisfy the estimate


_∥D ◦E_ ( _u_ ) _−_ _u∥_ _L_ 2 _≤_ _CN_ _[−][s]_ _∥u∥_ _H_ _s_ _._ (3.60)


On the other hand, if the coefficients _α_ _k_ decay exponentially as in the random
field (3.50), one can expect exponential decay rates for the encoding error as in the
following Theorem,


_**Theorem**_ **3.28** _**.**_ Let _µ ∈P_ ( _L_ [2] ( _D_ )), with _D_ = T _[d]_ or _D_ = [0 _,_ 2 _π_ ] _[d]_, denote the law of
the random field _u_ ( _·_ ; _Y_ ) defined by (3.51), with random variables _Y_ = ( _Y_ _j_ ) _j∈J_ _∈_

[ _−_ 1 _,_ 1] _[J]_, _J_ = Z _[d]_, and with _α_ _k_ satisfying the decay assumption (3.52). Given
_N ∈_ N, consider the encoder/decoder pair ( _E, D_ ) based on the discrete Fourier
transformation on a regular grid with grid size _m_ = (2 _N_ + 1) _[d]_ on _D_ . Then there
exists constants _C, c >_ 0, independent of _m_, such that the encoding error _E_ [�] _E_ for
the encoder/decoder pair _E_, _D_ defined by (3.53) and (3.58), can be bounded by


�
_E_ _E_ _≤_ _C_ exp( _−cℓm_ [1] _[/d]_ ) _._ (3.61)


Furthermore, if _G_ : _L_ [2] ( _D_ ) _→_ _L_ [2] ( _U_ ) is an operator, and _F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _L_ [2] ( _U_ ),
_**y**_ _�→F_ ( _**y**_ ) is defined by
_F_ ( _**y**_ ) := _G_ ( _u_ ( _·_ ; _**y**_ )) _,_


then we have the identity


_G ◦D_ (( _u_ _i_ ) _i∈I_ _N_ ) = _F_ ( _Y_ [�] (( _u_ _i_ ) _i∈I_ _N_ ) _,_ _∀_ ( _u_ _i_ ) _i∈I_ _N_ _∈_ R _[I]_ _[N]_ _≃_ R _[m]_ _._ (3.62)


This theorem, proved in Appendix C.17, shows that the encoding error for a
very general form of the underlying measure _µ_, decays exponentially in the number
of sensors and suggests that DeepONets will have a small encoding error with a
few sensors. The map _**u**_ _�→_ _Y_ [�] ( _**u**_ ) defined by (3.59), plays a key role in defining the
decoder (3.58) as well as the action of operator _G_ on the decoder. It turns out that
this map can be efficiently approximated by a neural network of moderate size as
follows:


_**Lemma**_ **3.29** _**.**_ Let _N ∈_ N, and denote _m_ := (2 _N_ +1) _[d]_ = _|K_ _N_ _|_ . Let _κ_ : _{_ 1 _, . . ., m} →_
_K_ _N_ be a bijection. There exists a constant _C >_ 0, independent of _N_, _m_, such that
for every _N_ there exists a ReLU neural network _N_ : R _[m]_ _→_ R _[m]_, with


size( _N_ ) _≤_ _C_ (1 + _m_ log( _m_ )) _,_ depth( _N_ ) _≤_ _C_ (1 + log( _m_ )) _,_

and such that _N_ ( _**u**_ ) = ( _Y_ [�] _κ_ (1) ( _**u**_ ) _, . . .,_ _Y_ [�] _κ_ ( _m_ ) ( _**u**_ )), for all _**u**_ _∈_ R _[m]_ .


The proof is based on a simple observation that, _Y_ [�] = _Y ◦FT ◦M_, with


(1) _M_ an affine mapping, introducing a bias,
(2) _FT_ a linear mapping, implementing the discrete Fourier transform
(3) _Y_ a linear scaling followed by a shrink operation.


34 ERROR ESTIMATES FOR DEEPONETS


The map _M_ can evidently be represented by a neural network of depth = _O_ (1),
and size = _O_ ( _m_ ). The discrete Fourier transform can be efficiently computed
using the fast Fourier transform (FFT) algorithm in _O_ ( _N_ _[d]_ log( _N_ )) = _O_ ( _m_ log( _m_ ))
operations. We note that each step of this recursive algorithm is linear, and requires
_O_ ( _m_ ) multiplications and _O_ (log( _m_ )) recursive steps to compute the FFT. Each step
in the recursion can be represented exactly by a finite number _O_ (1) of ReLU neural
network layers of size _O_ ( _m_ ). The _O_ (log( _m_ )) steps in the recursion can thus be
represented by a composition of _O_ (log( _m_ )) neural network layers. Thus, the whole
algorithm can be represented by a neural network of size = _O_ ( _m_ log( _m_ )) and depth
_O_ (log( _m_ )). Finally, the linear scaling step can clearly be represented by a neural
network of depth = _O_ (1) and size = _O_ ( _m_ ), and the shrink operation can be written
in the form


shrink( _Y_ ) = 1 _−_ max(0 _,_ 2 _−_ max(0 _,_ 1 + _Y_ )) = 1 _−_ _σ_ (2 _−_ _σ_ (1 + _Y_ )) _,_


where _σ_ ( _x_ ) = max(0 _, x_ ) denotes the ReLU activation function. Hence, _Y_ can be
represented by a neural network of size = _O_ ( _m_ ) and depth = _O_ (1). Combining
these three steps, we conclude that also the composition _Y_ [�] can be represented by
a neural network of size = _O_ ( _m_ log( _m_ )) and depth = _O_ (log( _m_ )). Similarly, any
smooth activation function such as sigmoid or tanh can be used to define a neural
network of the same size to approximate _Y_ [�] .


3.6. **Bounds on the approximation error** (3.2) **.** Given a particular choice of
encoder/decoder and reconstruction/projection pairs ( _E, D_ ) and ( _R, P_ ), the approximation error _E_ [�] _A_ (3.2) for the approximator _A_ : R _[m]_ _→_ R _[p]_ in the DeepONet
(2.10) is a measure for the non-commutativity of the following diagram:


_G_

_L_ [2] ( _D_ ) _L_ [2] ( _U_ )



_E_ _D_ _P_



_R_



_A_

R _[m]_ R _[p]_


i.e., it measures the error in the approximation _A ≈P ◦G ◦D_ . Thus, bounding
the approximation error _E_ [�] _A_ can be viewed as a special instance of the general
problem of the neural network approximation of high-dimensional mappings R _[m]_ _→_
R _[p]_ . Our aim in this section is to review some of the available results on neural

network approximation in a finite-dimensional setting and relate it to the problem
of deriving bounds on the approximation error (3.2) for the DeepONet (2.10). We
start by considering the neural network approximation for regular high-dimensional
mappings.


3.6.1. _Regular high-dimensional mappings._ Evidently, the neural network approximation of a mapping _G_ : R _[m]_ _→_ R _[p]_, _x �→_ _G_ ( _x_ ) = ( _G_ 1 ( _x_ ) _, . . ., G_ _p_ ( _x_ )) can be carried
out by independently approximating each of the components _G_ _j_ : R _[m]_ _→_ R by a
neural network _A_ _j_ : R _[m]_ _→_ R, and then combining these individual approximations
to a single neural network _A_ : R _[m]_ _→_ R _[p]_, _x �→A_ ( _x_ ) = ( _A_ 1 ( _x_ ) _, . . ., A_ _p_ ( _x_ )), with



size( _A_ ) =



_p_
� size( _A_ _j_ ) _≤_ _p_ max _j_ =1 _,...,p_ [size(] _[A]_ _[j]_ [)] _[.]_

_j_ =1



The approximation of _G_ _j_ : R _[m]_ _→_ R (over a bounded domain _K ⊂_ R _[m]_ ) by neural
networks is a fundamental approximation theoretic problem. One approach for
deriving general approximation results relies on the Sobolev regularity of _G_ _j_ . As a
prototype of the available results in this direction, we cite the following result due
to Yarotsky Yarotsky (2017), for _G_ _j_ _∈_ _W_ _[k,][∞]_ ([0 _,_ 1] _[m]_ ),


ERROR ESTIMATES FOR DEEPONETS 35


_**Theorem**_ **3.30** ((Yarotsky 2017, Theorem 1)) _**.**_ Let _m_, _k ∈_ N be given. There exists
a constant _C_ = _C_ ( _m, k_ ) _>_ 0, such that for any _ϵ ∈_ (0 _,_ 1) and _G_ _j_ : [0 _,_ 1] _[m]_ _→_ R with
_∥G_ _j_ _∥_ _W_ _k,∞_ _≤_ 1, there exists a ReLU neural network _A_ _j_ : [0 _,_ 1] _[m]_ _→_ R, with

depth( _A_ _j_ ) _≤_ _C_ (1 + log( _ϵ_ _[−]_ [1] )) _,_ size( _A_ _j_ ) _≤_ _Cϵ_ _[−][m/k]_ (1 + log( _ϵ_ _[−]_ [1] )) _,_


such that
_∥G_ _j_ ( _x_ ) _−A_ _j_ ( _x_ ) _∥_ _L_ _∞_ ([0 _,_ 1] _m_ ) _≤_ _ϵ._


In the particular case of a Lipschitz mapping _G_ : R _[m]_ _→_ R _[p]_, the upper limit
on the required size of the approximating neural network _A_ is thus _O_ ( _pϵ_ _[−][m]_ ). In
particular, this size scales exponentially in the input dimension (=number of DeepONet sensors) _m_ . By Definition 3.5, this constitutes a _curse of dimensionality_ as
the number of sensors _m_ need to grow as _ϵ →_ 0 from bounds such as (3.60) on the
encoding error (3.1). Thus, the above Theorem of Yarotsky may not suffice to find
optimal sizes of the approximator network _A_ in the DeepONet (2.10).


3.6.2. _Holomorphic infinite-dimensional mappings._ Given this possible curse of dimensionality in bounding the approximation error (3.2) for Lipschitz continuous
maps, we seek to find a class of nonlinear operators _G_, mapping infinite-dimensional
Banach spaces, for which this curse of dimensionality can be avoided. One possible
class is the class of _holomorphic mappings_ _**y**_ _�→F_ ( _**y**_ ), which has been shown in
recent papers Schwab & Zech (2019), Opschoor et al. (2019, 2020) to be efficiently
approximated by ReLU neural networks, _breaking the curse of dimensionality_ . For
instance, the class of mappings considered in Opschoor et al. (2020) are infinite
dimensional mappings

_F_ : [ _−_ 1 _,_ 1] [N] _→_ _V,_ _**y**_ = ( _y_ _j_ ) _j∈_ N _→F_ ( _**y**_ ) _,_


where _V_ is a Banach space. To simplify notation, we shall replace N by an arbitrary
countable index set _J_ and consider mappings

_F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _V,_ _**y**_ = ( _y_ _j_ ) _j∈J_ _→F_ ( _**y**_ ) _,_ (3.63)


in the following. In this context, we require the following definition Opschoor et al.
(2020):


_**Definition**_ **3.31** (( _b, ϵ_ )-admissibility) _**.**_ Let _V_ be a Banach space. Let _**b**_ = ( _b_ _j_ ) _j∈_ N
be a given sequence of monotonically decreasing positive reals _b_ _j_ _>_ 0 such that
_**b**_ _∈_ _ℓ_ _[p]_ (N) for some _p ∈_ (0 _,_ 1]. Let _κ_ : N _→J_ be an enumeration of the index set _J_ .
A poly-radius _**ρ**_ = ( _ρ_ _j_ ) _j∈J_ _∈_ (1 _, ∞_ ) _[J]_ is called ( _**b**_ _, ϵ_ ; _κ_ )-admissible for some _ϵ >_ 0, if
� _b_ _j_ ( _ρ_ _κ_ ( _j_ ) _−_ 1) _≤_ _ϵ._

_j∈_ N


We further recall that for a radius _ρ >_ 1, the Bernstein ellipse _E_ _ρ_ _⊂_ C is defined
by



_−_ 1
_z_ + _z_
_E_ _ρ_ := � 2



0 _≤|z| < ρ_ _._ (3.64)
���� �



We define holomorphy, following (Opschoor et al. 2020, Definition 3.3) as:


_**Definition**_ **3.32** (( _b, ϵ_ )-holomorphy) _**.**_ Let _V_ be a Banach space. Let _κ_ : N _→J_
be an enumeration of the index set _J_ . A continuous mapping _F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _V_ is
called ( _**b**_ _, ϵ, κ_ ) **-holomorphic**, if there exists a constant _C_ = _C_ ( _F_ ), such that the
following holds: For every ( _**b**_ _, ϵ, κ_ )-admissible _**ρ**_ with _E_ _**ρ**_ := [�] _j∈J_ _[E]_ _[ρ]_ _j_ _[⊂]_ [C] _[J]_ [, there]

exists an extension _F_ [�] : _E_ _**ρ**_ _→_ _V_ C, such that

_**z**_ _�→_ _F_ [�] ( _**z**_ ) is holomorphic (3.65)


36 ERROR ESTIMATES FOR DEEPONETS


as a function of each _z_ _j_ _∈_ _E_ _ρ_ _j_, _j ∈J_, and such that



sup
_**z**_ _∈E_ _**ρ**_



_F_ ( _**z**_ ) _≤_ _C_ ( _F_ ) _._ (3.66)
���� ��� _V_ C



Here _V_ C denotes the complexification of the (real) Banach space _V_ .


We remark that such _holomorphic operators_ arise naturally in the context of
elliptic and parabolic PDEs, for instance as the data to solution map of diffusion
equations with random coefficients Schwab & Zech (2019) and references therein.
We see further examples of these operators in the next section.
We can now state the following result which follows from (Opschoor et al. 2020,
Theorem 4.11) (our statement here is closer to the formulation of (Schwab & Zech
2019, Theorem 3.9)):


_**Proposition**_ **3.33** _**.**_ Let _V_ be a Banach space. Let _F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _V_ be a ( _**b**_ _, ϵ, κ_ )holomorphic map for some _**b**_ _∈_ _ℓ_ _[q]_ (N) and _q ∈_ (0 _,_ 1), and an enumeration _κ_ : N _→J_ .
Then there exists a constant _C >_ 0, such that for every _N ∈_ N, there exists an
index set

Λ _N_ _⊂_ � _**ν**_ = ( _ν_ 1 _, ν_ 2 _, . . ._ ) _∈_ [�] _j∈J_ [N] [0] ��� _ν_ _j_ _̸_ = 0 for finitely many _j ∈J_ � _,_


with _|_ Λ _N_ _|_ = _N_, a finite set of coefficients _{c_ _**ν**_ _}_ _**ν**_ _∈_ Λ _N_ _⊂_ _V_, and a ReLU network
_N_ : R _[N]_ _→_ R [Λ] _[N]_, _y �→{N_ _**ν**_ ( _y_ ) _}_ _**ν**_ _∈_ Λ _N_ with


size( _N_ ) _≤_ _C_ (1 + _N_ log( _N_ ) log log( _N_ )) _,_


depth( _N_ ) _≤_ _C_ (1 + log( _N_ ) log log( _N_ )) _,_


and such that



_≤_ _CN_ [1] _[−]_ [1] _[/q]_ _._ (3.67)
����� _V_



sup
_**y**_ _∈_ [ _−_ 1 _,_ 1] _[J]_



_F_ ( _**y**_ ) _−_ �
����� _**ν**_ _∈_ Λ



� _c_ _**ν**_ _N_ _**ν**_ ( _y_ _κ_ (1) _, . . ., y_ _κ_ ( _N_ ) )

_**ν**_ _∈_ Λ _N_



The following corollary of the above proposition, proved in Appendix C.18, enables us to apply the result of the theorem to the specific structure of the approximator neural network _A_ and the resulting approximation error (3.2) for our
DeepONet (2.10).


_**Corollary**_ **3.34** _**.**_ Let _V_ be a Banach space. Let _F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _V_ be a ( _**b**_ _, ϵ, κ_ )holomorphic map for some _**b**_ _∈_ _ℓ_ _[q]_ (N) and _q ∈_ (0 _,_ 1), where _κ_ : N _→J_ is an
enumeration of _J_ . In particular, it is assumed that _{b_ _j_ _}_ _j∈_ N is a monotonically
decreasing sequence. If _P_ : _V →_ R _[p]_ is a continuous linear mapping, then there
exists a constant _C >_ 0, such that for every _m ∈_ N, there exists a ReLU network
_N_ : R _[m]_ _→_ R _[p]_, with


size( _N_ ) _≤_ _C_ (1 + _pm_ log( _m_ ) log log( _m_ )) _,_


depth( _N_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ )) _,_


and such that


sup
_**y**_ _∈_ [ _−_ 1 _,_ 1] _[J]_ _[ ∥P ◦F]_ [(] _**[y]**_ [)] _[ −N]_ [(] _[y]_ _[κ]_ [(1)] _[, . . ., y]_ _[κ]_ [(] _[m]_ [)] [)] _[∥]_ _[ℓ]_ [2] [(][R] _[p]_ [)] _[ ≤]_ _[C][∥P∥]_ _[m]_ _[−][s]_ _[,]_


where _s_ := _q_ _[−]_ [1] _−_ 1 _>_ 0 and _∥P∥_ = _∥P∥_ _V →ℓ_ 2 denotes the operator norm.


With the above setup, we recall that maps of the form (3.63) arise naturally
from general nonlinear operators _G_ : _X →_ _L_ [2] ( _U_ ), _u �→G_ ( _u_ ), when the probability
measure _µ ∈P_ ( _X_ ) is given in the parametrized form (3.50). As in the subsection
3.5.3, we will focus on the case where either _D_ = T _[d]_ is periodic, or _D_ = [0 _,_ 2 _π_ ] _[d]_ is a


ERROR ESTIMATES FOR DEEPONETS 37


rectangular domain and an expansion (3.50) in terms of the standard Fourier basis
_{_ **e** _k_ _}_ _k∈_ Z _d_ . If the underlying measure _µ_ can be written as the law of a random field in
the parametrized form (3.50) with exponentially decaying coefficients (3.52), then
we can define a “parametrized version” of the operator _G_ by the following mapping
_F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _L_ [2] ( _U_ ), _J_ = Z _[d]_, defined by


_F_ ( _**y**_ ) := _G_ ( _u_ ( _·_ ; _**y**_ )) _,_ (3.68)


for _**y**_ = ( _y_ _k_ ) _k∈_ Z _d_ _∈_ [ _−_ 1 _,_ 1] _[J]_ .
To show how the neural network approximation of such _F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _L_ [2] ( _U_ ) relates to the approximator network _A_ : R _[m]_ _→_ R _[p]_, we recall that the encoder/decoder
pair ( _E, D_ ) of section 3.5.3 was constructed in terms of an approximate sensor datato-parameter mapping _**u**_ := ( _u_ _i_ ) _i∈I_ _N_ _�→_ _Y_ [�] ( _**u**_ ) such that (cf. Theorem 3.28)

( _G ◦D_ )( _**u**_ ) = _F_ ( _Y_ [�] ( _**u**_ )) _,_ _∀_ _**u**_ _∈_ R _[m]_ _._ (3.69)


Let _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_ be the projection mapping corresponding to a reconstruction
_R_ : R _[p]_ _→_ _L_ [2] ( _U_ ), and let the neural network _N_ be constructed as in Corollary
3.34, corresponding to an enumeration _κ_ : N _→_ Z _[d]_, which enumerates _k ∈_ Z _[d]_ with
increasing _|k|_ _∞_, i.e., such that _j �→|κ_ ( _j_ ) _|_ _∞_ is monotonically increasing. Then,
it is straightforward to see that for _m_ = (2 _N_ + 1) _[d]_, the Fourier wavenumbers
_κ_ (1) _, . . ., κ_ ( _m_ ) _∈_ Z _[d]_ correspond precisely to the Fourier wavenumbers in


_K_ _N_ = � _k ∈_ Z _[d]_ [ ��] _|k|_ _∞_ _≤_ _N_ � _._


In this case, we define

_A_ ( _**u**_ ) := _N_ ( _Y_ [�] _κ_ (1) ( _**u**_ ) _, . . .,_ _Y_ [�] _κ_ ( _m_ ) ( _**u**_ )) _,_ (3.70)


which we shall also denote more compactly as _A_ ( _**u**_ ) = _N_ ( _Y_ [�] _κ_ ( _**u**_ )), where _Y_ [�] _κ_ =
( _Y_ [�] _κ_ (1) _, . . .,_ _Y_ [�] _κ_ ( _m_ ) ). We now note the following theorem, proved in Appendix C.19,
on the approximation of _A_ :


_**Theorem**_ **3.35** _**.**_ Let _G_ : _X →_ _L_ [2] ( _U_ ) be a non-linear operator. Assume that the
parametrized mapping _F_ given by (3.68), defines a ( _**b**_ _, ϵ, κ_ )-holomorphic mapping
_F_ : [ _−_ 1 _,_ 1] _[J]_ _→_ _L_ [2] ( _U_ ), with _**b**_ _∈_ _ℓ_ _[q]_ (N) and _κ_ : N _→J_ an enumeration. Assume that
_µ ∈P_ ( _X_ ) is given as the law of the random field (3.50). Let the encoder/decoder
pair be constructed as in section 3.5.3, so that (3.69) holds. Given an affine reconstruction _R_ : R _[p]_ _→_ _L_ [2] ( _U_ ), let _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_ denote the corresponding optimal
linear projection (3.19). Then given _k ∈_ N, there exists a constant _C_ _k_ _>_ 0, independent of _m_, _p_, such that the approximator _A_ : R _[m]_ _→_ R _[p]_ defined by (3.70) can
be represented by a neural network with


size( _A_ ) _≤_ _C_ _k_ (1 + _mp_ log( _m_ ) log log( _m_ )) _,_


depth( _A_ ) _≤_ _C_ _k_ (1 + _m_ log( _m_ ) log log( _m_ )) _._


and such that the approximation error _E_ [�] _A_ can be estimated by


�
_E_ _A_ _≤_ _C_ _k_ _∥P∥_ _m_ _[−][k]_ _,_


where _∥P∥_ = _∥P∥_ _L_ 2 ( _U_ ) _→_ R _p_ is the operator norm of _P_ .


Thus, the above theorem shows that an approximator neural network _A_ of loglinear size in the product of the number of sensors _m_ and number of trunk nets
_p_ can lead to very small approximation error if the underlying operator _G_ yields
a holomorphic reduction (3.68). This will allow us to overcome the curse of dimensionality for the approximation error (3.2) for the DeepONet (2.10) in many

cases.


38 ERROR ESTIMATES FOR DEEPONETS


4. Error bounds on DeepONets in concrete examples.


In the last section, we decomposed the error (2.11) that a DeepONet (2.10)
incurs in approximating a nonlinear operator _G_ : _X →_ _Y_, with an underlying
measure _µ ∈P_ ( _X_ ), into the encoding error (3.1), the reconstruction error (3.3)
and approximation error (3.2). We provided explicit bounds on each of these three
errors. In particular, the encoding error was estimated in terms of the spectral
decay of the underlying covariance operator of the measure _µ_ and it was shown
that under a boundedness assumption on the eigenfunctions, even a random choice
of sensor points provided an optimal encoding error (modulo a log term). More
judicious choices of sensor points, for instance with the pseudo-spectral encoder
(3.46), allowed us to recover optimal (up to constants) encoding error. Similarly,
we showed that the reconstruction error (3.3) relies on the spectral decay properties
of the covariance operator with respect to the push-forward measure _G_ # _µ_ and can be
bounded by using the smoothness of the operator _G_ as in (3.11). Finally, estimating
the approximation error (3.2) boils down to a neural network approximation of
finite, but high, dimensional mappings and one can use either Sobolev regularity or
if available, holomorphy, of the map _P ◦G ◦D_ to bound this component. The above
discussion provides the following workflow to bound the DeepONet error (2.11) in
concrete cases, i.e., for concrete instances of the operator _G_ and the underlying

measure _µ_ :


_•_ For a given measure _µ_, estimate the spectral decay rate for the associated
covariance operator. Use random sensors in the case of no further information about the measure to obtain almost optimal bounds on the encoding
error. If more information is available, one can use bespoke sensor points
to obtain optimal bounds on the encoding error.

_•_ For a given operator _G_, use smoothness of the operator to estimate the
reconstruction error (3.3) as in (3.11).

_•_
For the approximation error (3.2), use the regularity, in particular possible
holomorphy, of the operator _G_ to estimate this error.

The simplest examples for _G_ correspond to those cases where it is a _bounded linear_
_operator_ . In this case, the above workflow is carried out in Appendix D, where we
show that DeepONet approximation of these general linear operators depends on
the spectral decay of the underlying measure _µ_ and on the approximation property
of the trunk net _τ_ . However for nonlinear operators _G_, one has to carry out the
above workflow in each specific case. To this end, we will illustrate this workflow
for four concrete examples of _nonlinear operators_, that are chosen to represent
different types of differential equations, namely, a nonlinear ODE, an elliptic PDE,
a nonlinear parabolic and a nonlinear hyperbolic PDE. Within each class, we choose
a concrete example that is widely agreed as a prototype for this class of problems.


4.1. **A nonlinear ODE: Gravity pendulum with external force.**



4.1.1. _Problem formulation._ We consider the following nonlinear ODE system, already considered in the context of approximation by DeepONets in Lu et al. (2019):
 _dvdt_ 1 [=] _[ v]_ [2] _[,]_

(4.1)

 _dv_



_dv_ 1



_dt_ [=] _[ v]_ [2] _[,]_







(4.1)

_dv_ 2

_dt_ [=] _[ −][γ]_ [ sin(] _[v]_ [1] [) +] _[ u]_ [(] _[t]_ [)] _[.]_



_dv_ 2



with initial condition _v_ (0) = 0 and where _γ >_ 0 is a parameter. Let us denote
_v_ = ( _v_ 1 _, v_ 2 ),



_v_ 2
_g_ ( _v_ ) := � _−γ_ sin( _v_ 1 )



0
_,_ _U_ ( _t_ ) :=
� � _u_ ( _t_ )



_,_
�


ERROR ESTIMATES FOR DEEPONETS 39


so that equation (4.1) can be written in the form


_dv_

_v_ (0) = 0 _._ (4.2)
_dt_ [=] _[ g]_ [(] _[v]_ [) +] _[ U,]_


In (4.2), _v_ 1 _, v_ 2 are the angle and angular velocity of the pendulum and the constant
_γ_ denotes a frequency parameter. The dynamics of the pendulum is driven by an
external force _u_ = _u_ ( _t_ ). It is straightforward to see that for each _r >_ 0, there exists
a constant _C_ _r_ _>_ 0, such that

_∥g_ [(] _[r]_ [)] _∥_ _L_ _∞_ (R) _≤_ _C_ _r_ _,_ (4.3)


where _g_ _[r]_ denotes the _r_ -th derivative.
With the external force _u_ as the input, the output of the system is the solution
vector _v_ ( _t_ ) and the underlying nonlinear operator is given by _G_ : _L_ [2] ([0 _, T_ ]) _→_
_L_ [2] ([0 _, T_ ]), _u �→G_ ( _u_ ) = _v_ . The following Lemma, proved in Appendix E.1, provides
a precise characterization of this operator.


_**Lemma**_ **4.1** _**.**_ There exists a constant _C_ = _C_ ( _∥g_ [(1)] _∥_ _L_ _∞_ _, T_ ) _>_ 0, such that for any
two _u, u_ _[′]_ _∈_ _L_ [2] ([0 _, T_ ]), we have


_∥G_ ( _u_ ) _−G_ ( _u_ _[′]_ ) _∥_ _L_ 2 ([0 _,T_ ]) _≤_ _C∥u −_ _u_ _[′]_ _∥_ _L_ 2 ([0 _,T_ ]) _._


In particular, _G_ : _L_ [2] ([0 _, T_ ]) _→_ _L_ [2] ([0 _, T_ ]), mapping _u_ ( _t_ ) _→_ _v_ ( _t_ ), with _v_ being the
solution of the ODE (4.2), is Lipschitz continuous.


Next, in order to define the data for the DeepONet approximation (see Definition
2.1), we need to specify an underlying measure _µ ∈P_ ( _L_ [2] ([0 _, T_ ])). Following the
discussion in the previous section and for the sake of definiteness, we choose a
parametrized measure _µ_, as considered in section 3.5.3, as a law of a random field
_u_, that can be expanded in the form



_,_ _t ∈_ [0 _, T_ ] _,_ (4.4)
�



_u_ ( _t_ ; _Y_ ) = � _Y_ _k_ _α_ _k_ **e** _k_

_k∈_ Z



2 _πt_
� _T_



where **e** _k_ ( _x_ ), _k ∈_ Z, denotes the one-dimensional standard Fourier basis on [0 _,_ 2 _π_ ]
(with notation of Appendix A) and the coefficients _α_ _k_ _≥_ 0 decay to zero. We will
assume that there exist constants _C_ _α_ _, ℓ>_ 0, such that


_α_ _k_ _≤_ _C_ _α_ exp( _−|k|ℓ_ ) _._


Furthermore, we assume that the _{Y_ _k_ _}_ _k∈_ Z are iid random variables on [ _−_ 1 _,_ 1].
With the above data for the DeepONet approximation problem, we will provide
explicit bounds on the total error (2.11) for a DeepONet (2.10) approximating
the operator _G_ . Following the workflow outlined above, we proceed to bound the
following sources of error.


4.1.2. _Bounds on the encoding error_ (3.1) _._ Given the underlying measure _µ_ defined
as the law of the random field (4.4), we choose the encoder-decoder pair as used
in section 3.5.3, i.e., the encoder is the pointwise evaluation (3.53) on equidistant
points _t_ 1 _, . . ., t_ _m_ on [0 _, T_ ] and the corresponding decoder is given by (3.58). A direct
application of Theorem 3.28 yields the following bound on the encoding error,


_**Proposition**_ **4.2** _**.**_ Let _µ ∈P_ ( _L_ [2] ([0 _, T_ ])) denote the law of the random field _u_ ( _·_ ; _Y_ )
defined by (4.4). Given _N ∈_ N consider the encoder/decoder pair ( _E, D_ ) with
_m_ = 2 _N_ + 1 grid points and given by (3.53) and (3.58), respectively. Then, there
exists a constant _C >_ 0, independent of _m_, such that the encoding error _E_ [�] _E_ (3.1)
can be bounded by

�
_E_ _E_ _≤_ _C_ exp( _−ℓ⌊m/_ 2 _⌋_ ) _._


40 ERROR ESTIMATES FOR DEEPONETS


Furthermore, denoting by _F_ : [ _−_ 1 _,_ 1] [Z] _→_ _L_ [2] ([0 _, T_ ]), _**y**_ _�→F_ ( _**y**_ ), the mapping


_F_ ( _**y**_ ) := _G_ ( _u_ ( _·_ ; _**y**_ )) _,_ (4.5)


we have the identity _G ◦D_ ( _**u**_ ) = _F_ ( _Y_ [�] ( _**u**_ )), for all _**u**_ _∈_ R _[m]_ .


Moreover, as in Lemma 3.29, _Y_ [�] in (4.5) can be represented by a neural network,
in the sense that _Y_ [�] _k_ _≡_ 0, for _|k| > N_, and there exists a neural network _N_ with


size( _N_ ) = _O_ ( _m_ log( _m_ )) _,_ depth( _N_ ) = _O_ (log( _m_ )) _,_

and _N_ ( _**u**_ ) = ( _Y_ [�] _−N_ ( _**u**_ ) _, . . .,_ _Y_ [�] 0 ( _**u**_ ) _, . . .,_ _Y_ [�] _N_ ( _**u**_ )), for all _**u**_ _∈_ R _[m]_ .


4.1.3. _Bounds on the reconstruction error_ (3.3) _._ Following our program outlined
above, we will bound the reconstruction error (3.3) for a DeepONet approximation
the operator _G_ for the forced pendulum by appealing to the smoothness of the
image Im( _G_ ) of _G_ . To this end, we have the following lemma, proved in Appendix
E.2,


_**Lemma**_ **4.3** _**.**_ Let _T >_ 0, and consider the solution _v_ ( _t_ ) of (4.2) for _t ∈_ [0 _, T_ ],
where _g_ ( _v_ ) satisfies _L_ _[∞]_ -bound (4.3) for all _k ∈_ N, _k ≥_ 1. Then for any _k ∈_ N,
there exists a constant _A_ _k_ _>_ 0 (possibly depending on _g_ and _T_, in addition to _k_,
but independent of _u_ ), such that

_∥v_ [(] _[k]_ [)] _∥_ _L_ _∞_ _≤_ _A_ _k_ �1 + _∥u∥_ _[k]_ _H_ _[k]_ � _._


Here



_∥u∥_ _H_ _k_ :=



_k_
� _∥u_ [(] _[ℓ]_ [)] _∥_ _L_ 2 ([0 _,T_ ]) _._


_ℓ_ =0



Given the desired smoothness of the image of the operator _G_, we need to find a
suitable reconstructor (2.9). To this end, we will use Legendre polynomials to build
our reconstructor and have the following result, proved in Appendix E.3,


_**Lemma**_ **4.4** _**.**_ If � _τ_ _k_, _k_ = 1 _, . . ., p_ are the first _p_ Legendre polynomials, then the
reconstruction error _E_ [�] _R_ � [for the reconstruction mapping]



_α_ = ( _α_ 1 _, . . ., α_ _p_ ) _�→_ _R_ [�] ( _α_ ) =



_p_
� _α_ _k_ _τ_ � _k_ _,_


_k_ =1



induced by the trunk net � _**τ**_ = (0 _,_ � _τ_ 1 _, . . .,_ � _τ_ _p_ ), satisfies



�
_E_ _R_ � _[≤]_ _[C]_

_p_ _[k]_



1 _/_ 2
(1 + _∥u∥_ _H_ _k_ ([0 _,T_ ]) ) [2] _[k]_ _dµ_ ( _u_ ) _._

�ˆ _X_ �



For some constant _C_ = _C_ ( _k, T_ ) _>_ 0. In particular, if _µ ∈P_ ( _L_ [2] ( _D_ )) is concentrated on _H_ _[k]_ ( _D_ ) and ´ _L_ [2] ( _D_ ) _[∥][u][∥]_ _H_ [2] _[k]_ _[k]_ _[ dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [, then there exists a constant]

_C_ = _C_ ( _T, k, µ_ ) _>_ 0 independent of _p_, such that


�
_E_ _R_ � _[≤]_ _[Cp]_ _[−][k]_ _[.]_


From (Opschoor et al. 2019, Proposition 2.10), we have the following result; for
any _p ∈_ N, _δ ∈_ (0 _,_ 1), there exists a ReLU neural network _**L**_ _δ_ : [0 _, T_ ] _→_ R _[p]_, _t �→_
_**L**_ _δ_ ( _t_ ) = ( _L_ 1 _,δ_ ( _t_ ) _, . . ., L_ _p,δ_ ( _t_ )), which approximates the first _p_ Legendre polynomials
_L_ 1 ( _t_ ) _, . . ., L_ _p_ ( _t_ ) with

max
_j_ =1 _,...,p_ _[∥][L]_ _[j]_ _[ −]_ _[L]_ _[j,δ]_ _[∥]_ _[L]_ _[∞]_ _[≤]_ _[δ,]_

and for a constant _C >_ 0, independent of _δ_ and _p_, it holds

size( _**L**_ _δ_ ) _≤_ _Cp_ [3] + _p_ [2] log( _δ_ _[−]_ [1] ) _._ depth( _**L**_ _δ_ ) _≤_ _C_ (1 + log( _p_ ))( _p_ + log( _δ_ _[−]_ [1] )) _,_


ERROR ESTIMATES FOR DEEPONETS 41


We will leverage the above neural network to find a suitable trunk net for the
DeepONet approximation of the operator _G_ for the forced pendulum. To this end,
let _R_ [�] denote the reconstruction



�
_R_ ( _α_ ) =



_p_
� _α_ _k_ _τ_ � _k_ _,_


_k_ =1



where � _τ_ _k_ = _L_ _k_ ( _t_ ) denotes the _k_ -th Legendre polynomial. By Lemma 4.4, for any
_r_ � _∈_ N, there exists _C >_ 0, depending on _r_ but independent of _p_, such that we have
_E_ _R_ � _[≤]_ _[Cp]_ _[−][r]_ [, for all] _[ p][ ∈]_ [N][. Choosing] _**[ τ]**_ [ =] _**[ L]**_ _[δ]_ [ as above with] _[ δ]_ [ =] _[ p]_ _[−][r][−]_ [3] _[/]_ [2] [, it follows]
that for any _p ∈_ N, there exists a trunk net _**τ**_ with

_p_ [3] _[/]_ [2] max
_k_ =1 _,...,p_ _[∥][τ]_ _[k]_ _[ −]_ _[τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] [([0] _[,T]_ [ ])] _[ ≤]_ _[p]_ _[−][r]_ _[,]_


and
size( _**τ**_ ) = _O_ ( _p_ [3] ) _._ depth( _**τ**_ ) = _O_ ( _p_ log( _p_ )) _._
From Lemma 3.16, it follows that for the reconstruction _R_ : R _[p]_ _→_ _L_ [2] ([0 _, T_ ]),



_R_ ( _α_ ) =



_p_
� _α_ _k_ _τ_ _k_ _,_


_k_ =1



induced by the trunk net _**τ**_ = (0 _, τ_ 1 _, . . ., τ_ _p_ ), we have


� �
_E_ _R_ _≤_ _E_ _R_ � [+] _[ p]_ [3] _[/]_ [2] _k_ =1 max _,...,p_ _[∥][τ]_ _[k]_ _[ −]_ _[τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] [([0] _[,T]_ [ ])] _[ ≤]_ _[Cp]_ _[−][r]_ _[.]_


We summarize these observations in the following proposition,


_**Proposition**_ **4.5** _**.**_ Let _µ_ denote the law of the random field _u_ ( _·_ ; _Y_ ) given by (4.4).
For any _r ∈_ N, there exists a constant _C >_ 0, depending on _T_, _µ_ and _r_, but
independent of _p_, such that for any _p ∈_ N, there exists a trunk neural network
_**τ**_ : [0 _, T_ ] _→_ R _[p]_ [+1], _s �→_ ( _τ_ 0 ( _s_ ) _, . . ., τ_ _p_ ( _s_ )), with

size( _**τ**_ ) _≤_ _C_ (1 + _p_ [3] ) _,_ depth( _**τ**_ ) _≤_ _C_ (1 + _p_ log( _p_ ))) _,_

such that the reconstruction error _E_ [�] _R_ with corresponding projection _P_, satisfies
the bound

�
_E_ _R_ _≤_ _Cp_ _[−][r]_ _,_ ( _p ∈_ N) _._ (4.6)

Furthermore, for any _p ∈_ N, the reconstruction _R_ and the projection satisfy the
uniform bound Lip( _R_ ) _,_ Lip( _P_ ) _≤_ 2, where

Lip( _R_ ) = Lip � _R_ : (R _[p]_ _, ∥· ∥_ _ℓ_ 2 ) _→_ ( _L_ [2] ([0 _, T_ ]) _, ∥· ∥_ _L_ 2 )�

Lip( _P_ ) = Lip � _P_ : ( _L_ [2] ([0 _, T_ ]) _, ∥· ∥_ _L_ 2 ) _→_ (R _[p]_ _, ∥· ∥_ _ℓ_ 2 )� _._


Thus, we observe from (4.6) that the reconstruction error for a DeepONet can be
made very small for a moderate number of trunk nets. Moreover, the corresponding
size of the trunk net is moderate in the number of trunk nets.


4.1.4. _Bounds on the approximation error_ (3.2) _._ Following our workflow, we will
derive bounds on the approximation error (3.2) for a DeepONet (2.10) approximating the operator _G_ for the forced gravity pendulum by showing that the corresponding operator is holomorphic in the sense of Definition 3.32.
To this end, we assume that



_u_ ( _t_ ) = _u_ ( _t_ ; _**y**_ ) =



_∞_
� _y_ _k_ _α_ _k_ **e** _k_ ( _t_ ) _,_ ( _**y**_ = ( _y_ _j_ ) _j∈_ N ) _,_ (4.7)


_k_ =1



can be expanded by the Fourier basis functions **e** _k_ _∈_ _L_ [2] ([0 _, T_ ]), such that

_α_ _k_ _∥_ **e** _k_ _∥_ _L_ _∞_ _≤_ _C_ _α_ _e_ _[−|][k][|][ℓ]_ _≤_ 1 _,_ _∀_ _k ∈_ Z _._ (4.8)


42 ERROR ESTIMATES FOR DEEPONETS


Letting _κ_ : N _→_ Z denote the enumeration of appendix A, with _j �→|κ_ ( _j_ ) _|_ increasing, we define a monotonically decreasing sequence ( _b_ _j_ ) _j∈_ N by

_b_ _j_ := _C_ _α_ _e_ _[−|][κ]_ [(] _[j]_ [)] _[|][ℓ]_ _._ (4.9)


We will show that that the mapping

_F_ : [ _−_ 1 _,_ 1] [N] _→_ _L_ [2] ([0 _, T_ ]) _,_ _**y**_ _�→_ _v_ ( _**y**_ ) _,_


where _v_ ( _**y**_ ) solves the gravity pendulum equation (4.2) with forcing _U_ ( _t_ ; _**y**_ ) =
(0 _, u_ ( _t_ ; _**y**_ )) can be extended to a holomorphic mapping on suitable (admissible)
poly-ellipses _E_ _**ρ**_ = [�] _[∞]_ _j_ =1 _[E]_ _[ρ]_ _j_ _[⊂]_ [C] [N] [,]



_F_ :



_∞_
� _E_ _ρ_ _j_ _→_ _L_ [2] C [([0] _[, T]_ [])] _[,]_ _**z**_ _�→_ _v_ ( _**z**_ ) _,_

_j_ =1



where _**ρ**_ = ( _ρ_ _j_ ) _j∈_ N, _ρ_ _j_ _>_ 1 for all _j ∈_ N. For _ρ >_ 1, _E_ _ρ_ _⊂_ C denotes the (interior of
the) Bernstein ellipse (cp. (3.64)). It is straightforward to see that,


_E_ _ρ_ _⊂{z ∈_ C _| |_ Re( _z_ ) _| < ρ, |_ Im( _z_ ) _| < ρ −_ 1 _}._ (4.10)


To prove the existence of an analytic continuation to suitable poly-ellipses _E_ _**ρ**_,
our first goal is to show that there exists a _δ >_ 0, such that the ODE-system (4.2)
has a complex-valued solution _v_ : [0 _, T_ ] _→_ C [2] for any forcing _U_ : [0 _, T_ ] _→_ C [2] with
_|_ Im( _U_ ( _t_ )) _| ≤_ _δ_ . To this end, we first note that the complex-valued ODE (4.2) is
equivalent to the following system for _v_ ( _t_ ) = _v_ _r_ ( _t_ )+ _iv_ _i_ ( _t_ ), with _v_ _r_ = ( _v_ _r,_ 1 _, v_ _r,_ 2 ) _, v_ _i_ =
( _v_ _i,_ 1 _, v_ _i,_ 2 ) : [0 _, T_ ] _→_ R [2] :
 _dv_ _r_ _v_ _r_ (0) = 0 _,_

_dt_ [=] _[ g]_ _[r]_ [(] _[v]_ _[r]_ _[, v]_ _[i]_ [) + Re(] _[U]_ [)] _[,]_

(4.11)

 _dv_



_dv_ _r_



_v_ _r_ (0) = 0 _,_
_dt_ [=] _[ g]_ _[r]_ [(] _[v]_ _[r]_ _[, v]_ _[i]_ [) + Re(] _[U]_ [)] _[,]_







_dv_ _i_



(4.11)

_dv_ _i_

_v_ _i_ (0) = 0 _,_
_dt_ [=] _[ g]_ _[i]_ [(] _[v]_ _[r]_ _[, v]_ _[i]_ [) + Im(] _[U]_ [)] _[,]_



where _U_ ( _t_ ) = (0 _, u_ ( _t_ )), _u_ : [0 _, T_ ] _→_ C and



_v_ _r,_ 2
_g_ _r_ ( _v_ _r_ _, v_ _i_ ) = � _−γ_ sin( _v_ _r,_ 1 ) cosh( _v_ _i,_ 1 )



_v_ _i,_ 2
� _,_ _g_ _i_ ( _v_ _r_ _, v_ _i_ ) = � _−γ_ cos( _v_ _r,_ 1 ) sinh( _v_ _i,_ 1 )



_._
�



We note that the second equation of (4.11) implies that



_t_
_|v_ _i_ ( _t_ ) _| ≤_
ˆ 0



_t_ _t_

_|g_ _i_ ( _v_ _r_ ( _s_ ) _, v_ _i_ ( _s_ )) _| ds_ +
0 ˆ 0



_|_ Im( _U_ ( _s_ )) _| ds_
0



_t_

_≤_
ˆ 0



_|_ Im( _u_ ( _s_ )) _| ds._
0



_t_ _t_

( _|v_ _i_ ( _s_ ) _|_ + _γ|_ sinh( _v_ _i,_ 1 ( _s_ )) _|_ ) _ds_ +
0 ˆ 0



Furthermore, we have


_|_ sinh( _x_ ) _|_ =
����



_x_

cosh( _ξ_ ) _dξ_ _≤|x|_ sup cosh( _ξ_ ) _≤|x|e_ _[|][x][|]_ _,_
0 ���� _|ξ|≤|x|_



_x_

cosh( _ξ_ ) _dξ_ _≤|x|_ sup cosh( _ξ_ ) _≤|x|e_ _[|][x][|]_ _,_

ˆ 0 ���� _|ξ|≤|x|_



for any _x ∈_ R. We thus conclude that



_t_
_|v_ _i_ ( _t_ ) _| ≤_
ˆ 0



_t_ _t_

_|v_ _i_ ( _s_ ) _|_ 1 + _γe_ _[|][v]_ _[i]_ [(] _[s]_ [)] _[|]_ [�] _ds_ +
0 � ˆ 0



_|_ Im( _u_ ( _s_ )) _| ds,_ (4.12)
0



for all _t ∈_ [0 _, T_ ].
The above estimate paves the way for the following lemma, proved in Appendix E.4 on the boundedness of the imaginary part of the solution of the complex
extension of the forced gravity pendulum (4.11):


_**Lemma**_ **4.6** _**.**_ Let _u_ ( _t_ ) _∈_ _L_ _[∞]_ ([0 _, T_ ]). Assume that the solution of (4.11) exists
on [0 _, T_ 0 ] for some 0 _< T_ 0 _≤_ _T_ . There exists a constant _δ >_ 0, such that if
sup _s∈_ [0 _,T_ ] _|_ Im( _u_ ( _s_ )) _| ≤_ _δ_, then sup _s∈_ [0 _,T_ 0 ] _|v_ _i_ ( _s_ ) _| ≤_ 1.


ERROR ESTIMATES FOR DEEPONETS 43


This lemma allows us to prove the following lemma (see the detailed proof in
Appendix E.5), which establishes global solutions for the complex extension of the
forced pendulum (4.11),


_**Lemma**_ **4.7** _**.**_ If _δ_ is chosen as in Lemma 4.6, and if _∥_ Im( _u_ ) _∥_ _L_ _∞_ ([0 _,T_ ]) _≤_ _δ_, then the
maximal existence interval of the solution of the ODE system (4.11) contains [0 _, T_ ],
and sup _t∈_ [0 _,T_ ] _|v_ _i_ ( _t_ ) _| ≤_ 1.


Now assume that _u_ ( _t_ ) = _u_ ( _t_ ; _**z**_ ) is parametrized as in (4.7). If _**z**_ _∈_ _E_ _**ρ**_ belongs to
a poly-ellipse in C [N], then clearly



_∥_ Im( _u_ ) _∥_ _L_ _∞_ _≤_



_∞_

_k_ � =1 _|_ Im( _z_ _k_ ) _| α_ _≤_ ~~�~~ _bk_ _j_ _∥_ ( **e** _k_ ~~�~~ _k_ = � _∥_ _κL_ ( _j_ _∞_ )) ~~�~~



_≤_



_∞_
�( _ρ_ _κ_ ( _j_ ) _−_ 1) _b_ _j_ _._ (4.13)

_j_ =1



We recall (cp. Definition 3.31), that a sequence _**ρ**_ = ( _ρ_ _j_ ) _j∈_ N, with _ρ_ _j_ _>_ 1 for all
_j ∈_ N is called ( _**b**_ _, δ, κ_ )-admissible, if


_∞_
� _b_ _j_ ( _ρ_ _κ_ ( _j_ ) _−_ 1) _< δ,_ (4.14)

_j_ =1


where _b_ _j_ is defined by (4.9), and where _δ >_ 0 is chosen as in Lemma 4.7. It is now
clear from Lemma 4.7, that for any _**z**_ _∈_ _E_ _**ρ**_ with ( _**b**_ _, δ, κ_ )- _admissible_ _**ρ**_, the solution
_v_ ( _**z**_ ) _∈_ _C_ ([0 _, T_ ]; C [2] ) of (4.11) is well-defined. Hence, we can define a mapping



_F_ : _E_ _**ρ**_ =



_∞_
� _E_ _ρ_ _j_ _→_ _L_ [2] C [([0] _[, T]_ [])] _[,]_ _F_ ( _**z**_ ) := _v_ ( _**z**_ ) _,_ (4.15)

_j_ =1



where _v_ ( _**z**_ ) is the solution of (4.11) with forcing _U_ ( _t_ ) = ( _u_ ( _t_ ) _,_ 0), where _u_ ( _t_ ) =
_u_ ( _t_ ; _**z**_ ) is given by (4.7). This allows us to prove in Appendix E.6 the following
lemma on the holomorphy of the map _F_ (4.15),


_**Lemma**_ **4.8** _**.**_ The mapping _F_ defined by (4.15) is ( _**b**_ _, δ, κ_ )-holomorphic, according
to Definition 3.32; i.e., for each index _j ∈_ N, the componentwise mapping


_E_ _ρ_ _j_ _�→_ _L_ [2] C [([0] _[, T]_ [])] _[,]_ _z_ _j_ _�→F_ ( _**z**_ ) _,_


where the _z_ _k_ _∈_ _E_ _ρ_ _k_ for _k ̸_ = _j_ are held fixed, is (complex-)differentiable. Moreover,
there exists a constant _C >_ 0, such that for any admissible _**ρ**_, we have


sup _∥F_ ( _**z**_ ) _∥_ _L_ 2C [([0] _[,T]_ [ ])] _[ ≤]_ _[C.]_
_**z**_ _∈E_ _**ρ**_


As a consequence of Lemma 4.8, we can now state the following approximation
result, which follows from the general approximation result for ( _**b**_ _, ϵ, κ_ )-holomorphic
mappings, Theorem 3.35:


_**Proposition**_ **4.9** _**.**_ Let ( _E, D_ ) denote the encoder/decoder pair (3.53), (3.58) with
_m_ sensors, let ( _R, P_ ) denote the reconstruction/projection pair, constructed in
Proposition 4.5, for a given _p ∈_ N. For any _k ∈_ N, there exists a constant _C >_ 0,
depending on _k_, the final time _T_ and the probability measure _µ_, but independent
of _m_ and _p_, such that there exists a neural approximator network _A_ : R _[m]_ _→_ R _[p]_

with


size( _A_ ) _≤_ _C_ (1 + _pm_ log( _m_ ) log log( _m_ )) _,_


depth( _A_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ )) _,_


44 ERROR ESTIMATES FOR DEEPONETS


and such that the approximation error


� 1 _/_ 2
_E_ _A_ = _ℓ_ [2] _[ d]_ [(] _[E]_ [#] _[µ]_ [)(] _**[u]**_ [)] _,_
�ˆ R _[m]_ _[ ∥A]_ [(] _**[u]**_ [)] _[ −P ◦G ◦D]_ [(] _**[u]**_ [)] _[∥]_ [2] �


can be estimated by

�
_E_ _A_ _≤_ _Cm_ _[−][k]_ _._


The proof follows from a direct application of Theorem 3.35, with the observation
that _∥P∥_ = Lip( _P_ ) _≤_ 2 in Proposition 4.5 is bounded independently of _m_ and _p_ .


4.1.5. _Bounds on the DeepONet approximation error_ (2.11) _._ We combine propositions 4.2, 4.5, 4.9 to state the following theorem on the DeepONet error (2.11) for
the forced gravity pendulum;


_**Theorem**_ **4.10** _**.**_ Consider the DeepONet approximation problem for the gravity
pendulum (4.1), where the forcing _u_ ( _t_ ) is distributed according to a probability
measure _µ ∈P_ ( _L_ [2] ([0 _, T_ ])) given as the law of the random field (4.4). For any
_k, r ∈_ N, there exists a constant _C_ = _C_ ( _k, r_ ) _>_ 0, and a constant _c >_ 0, independent
of _m_, _p_, such that for any _m, p ∈_ N, there exists a DeepONet (2.10) with trunk net
_**τ**_ and branch net _**β**_, such that


size( _**τ**_ ) _≤_ _C_ (1 + _p_ [3] ) _,_ depth( _**τ**_ ) _≤_ _C_ (1 + _p_ log( _p_ )) _,_


and


size( _**β**_ ) _≤_ _C_ (1 + _pm_ log( _m_ ) log log( _m_ )) _,_


depth( _**β**_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ )) _,_


and such that the DeepONet approximation error (2.11) is bounded by


�
_E ≤_ _Ce_ _[−][cℓm]_ + _Cm_ _[−][k]_ + _Cp_ _[−][r]_ _._ (4.16)


_**Remark**_ **4.11** _**.**_ Theorem 4.10 guarantees that for _ϵ >_ 0, a DeepONet approximation error of _E_ [�] _∼_ _ϵ_ can be achieved provided that _m_ ≳ max( _ℓ_ _[−]_ [1] log( _ϵ_ _[−]_ [1] ) _, ϵ_ _[−]_ [1] _[/k]_ ) and
_p_ ≳ _ϵ_ _[−]_ [1] _[/r]_ . As long as the intuitively obvious restriction that _mℓ_ _≫_ 1 is satisfied,
i.e., that the sensors can resolve the typical length scale _ℓ_, these requirements can
be achieved provided that _m_ ≳ _ϵ_ _[−]_ [1] _[/k]_ and _p_ ≳ _ϵ_ _[−]_ [1] _[/r]_ . In this case, an error _E_ [�] ≲ _ϵ_
can be achieved by a DeepONet with size of the order of


size( _**τ**_ _,_ _**β**_ ) _∼_ _ϵ_ _[−]_ [1] _[/r]_ [ �] _ϵ_ _[−]_ [2] _[/r]_ + _ϵ_ _[−]_ [1] _[/k]_ log( _ϵ_ _[−]_ [1] ) log log( _ϵ_ _[−]_ [1] ) _._ (4.17)
�


As the _k, r ∈_ N were arbitrary, this shows that the required size only grows _sub-_
_algebraically_ with _ϵ_ _[−]_ [1] _→∞_, i.e. the required size scales asymptotically _≪_ _ϵ_ _[−]_ [1] _[/s]_, as
_ϵ →_ 0, for any _s >_ 0. Given Definition 3.5, this clearly implies that the DeepONet
approximation for this problem does not suffer from the curse of dimensionality.
We however point out that the implied constants in these asymptotic estimates
depend on _k, r_ . In particular, these constants might deteriorate as _k, r →∞_ .
On the other hand, we can fix _k, r_ to be large and conclude from the complexity
estimate (4.17) that the complexity of the DeepONet in this case can only grow
algebraically with the error tolerance _ϵ_ . Given Definition 3.5 (cp. equation (3.7)),
this suffices to prove that a DeepONet can approximate the operator _G_ in the
forced gravity pendulum problem with the measure _µ_ (4.4) by _breaking the curse_
_of dimensionality_ .


ERROR ESTIMATES FOR DEEPONETS 45


_**Remark**_ **4.12** _**.**_ In Lu et al. (2019), the authors used a Gaussian measure, with a
covariance kernel (3.45), as the underlying measure _µ_ for the forced gravity pendulum. An estimate, analogous to (4.16) can be proved in this case, but with a
super-exponential decay of the encoding error with respect to the number of sensors
_m_, given the bound (3.48). However, the overall complexity still has sub-algebraic
asymptotic growth in _ϵ_ _[−]_ [1], as the other error terms scale exactly as in (4.17).
A different underlying measure _µ_ results from choosing an algebraic decay of the
coefficients _α_ _k_ in (4.4). Clearly, the bound (4.16) will also hold in this case but
with an algebraic decay in the encoding error. Nevertheless, the total complexity
of the problem still scales algebraically (polynomially) with the error tolerance _ϵ_
and shows that the resulting DeepONet breaks the curse of dimensionality also in
this case.


_**Remark**_ **4.13** _**.**_ The estimates on the sizes of the trunk and branch nets rely on
expressivity results for ReLU deep neural networks in approximating holomorphic
functions Opschoor et al. (2019). These in turn, depend on the results of Yarotsky
(2017) for approximation of functions using ReLU networks. These approximation
results may not necessarily be optimal and might suggest networks of larger size
as well as depth, than what is needed in practice Lu et al. (2019). Moreover, the
constants in complexity estimate (4.17) depend on the final time _T_ and can grow
exponentially in _T_ (with a familiar argument based on the Gr¨onwall’s inquality).


4.2. **An elliptic PDE: Multi-d diffusion with variable coefficients.**


4.2.1. _Problem formulation._ We will consider the following very popular model
problem for elliptic PDEs with unknown diffusion coefficients Cohen et al. (2011)
and references therein. For the sake of definiteness and simplicity, we shall assume
a periodic domain _D_ = T _[d]_ in the following. We consider an elliptic PDE with
variable coefficients _a_ :


_−∇·_ ( _a_ ( _x_ ) _∇u_ ( _x_ )) = _f_ ( _x_ ) _,_ (4.18)


for _u ∈_ _H_ [1] ( _D_ ) with suitable boundary conditions, and for fixed _f ∈_ _H_ _[−]_ [1] ( _D_ ).
We also fix a probability measure _µ_ on the coefficients _a_ on _L_ [2] ( _D_ ), such that
supp( _µ_ ) _⊂_ _L_ _[∞]_ ( _D_ ). To ensure coercivity of the problem (4.18), we will assume that


_µ_ (� _a ∈_ _L_ [2] ( _D_ ) �� 0 _< λ_ ( _a_ ) _≤_ Λ( _a_ ) _< ∞_ �) = 1 _,_ (4.19)


where


_λ_ ( _a_ ) := ess inf (4.20)
_x∈D_ _[a]_ [(] _[x]_ [)] _[,]_

Λ( _a_ ) := ess sup _a_ ( _x_ ) (4.21)
_x∈D_


denote the essential infimum and supremum of _a_, respectively. To ensure uniqueness of solutions to (4.18), we require that _u ∈_ _H_ 0 [1] [(][T] _[d]_ [) have zero mean, i.e., that]
´ T _[d]_ _[ u]_ [(] _[x]_ [)] _[ dx]_ [ = 0. We note that the condition (4.19) on] _[ µ]_ [ is for example satisfied by]

the log-Gaussian measures commonly employed in hydrology (Charrier 2012).
We note that the variable coefficient _a_ can model the rock permeability in a
Darcy flow with _u_ modeling the pressure and _f_ a source/injection term. Similarly, _a_
may model variable conductivity in a medium with _u_ modeling the temperature. In
many applications, one is interested in inferring the solution _u_ for a given coefficient
_a_ as the input. Thus, the _nonlinear_ operator _G_ maps the input coefficient _a_ into the
solution field _u_ of the PDE (4.18). The well-definedness of this operator is given in
the following Lemma (proved in appendix E.7),


46 ERROR ESTIMATES FOR DEEPONETS


_**Lemma**_ **4.14** _**.**_ Assume that the coefficients _a, a_ _[′]_ _∈_ _C_ (T _[d]_ ) satisfy the uniform coercivity assumption

0 _< λ ≤_ inf
_x∈_ T _[d]_ _[ a]_ [(] _[x]_ [)] _[,]_ [ inf] _x∈_ T _[d]_ _[ a]_ _[′]_ [(] _[x]_ [)] _[.]_


Let _u, u_ _[′]_ _∈_ _H_ 0 [1] [(][T] _[d]_ [) denote the solution to (4.18) with coefficients] _[ a, a]_ _[′]_ [, respectively,]
and with the same right-hand side _f ∈_ _L_ [2] (T _[d]_ ). There exists a constant _C_ =
_C_ ( _λ,_ T _[d]_ _, ∥f_ _∥_ _L_ 2 (T _d_ ) ), such that


_∥u −_ _u_ _[′]_ _∥_ _L_ 2 (T _d_ ) _≤_ _C∥a −_ _a_ _[′]_ _∥_ _L_ _∞_ (T _d_ ) _._


In particular the operator _G_ : _X →_ _L_ [2] (T _[d]_ ), _a �→_ _u_, is Lipschitz continuous for any
set _X ⊂_ _C_ (T _[d]_ ), and satisfying the coercivity assumption.


By Sobolev embedding, Lemma 4.14 ensures the well-definedness, and Lipschitz
continuity, of the operator _G_ : _H_ _[s]_ (T _[d]_ ) _→_ _L_ [2] (T _[d]_ ) for any _s > d/_ 2. We complete
the data for a DeepONet approximation problem (Definition 2.1) by specifying an
underlying measure _µ_ . Following standard practice Cohen et al. (2011) and references therein, and aided by the fact that we enforce periodic boundary conditions,
the underlying measure _µ_ is the law of a random field _a_, that is expanded in terms
of the Fourier basis. More precisely, we assume that we can write _a_ in the following
form


_a_ ( _x, Y_ ) = ~~_a_~~ ~~(~~ _x_ ) + � _α_ _k_ _Y_ _k_ **e** _k_ ( _x_ ) _,_ (4.22)

_k∈_ Z _[d]_


with notation from Appendix A, and where for simplicity ~~_a_~~ ~~(~~ _x_ ) _≡_ 1 is assumed to be
constant. Furthermore, we will consider the case of smooth coefficients _x �→_ _a_ ( _x_ ; _Y_ ),
which is ensured by requiring that there exist constants _C_ _α_ _>_ 0 and _ℓ>_ 1, such
that


_|α_ _k_ _| ≤_ _C_ _α_ exp( _−ℓ|k|_ _∞_ ) _,_ _∀_ _k ∈_ Z _[d]_ _._ (4.23)


Let us define _**b**_ = ( _b_ 1 _, b_ 2 _, . . ._ ) _∈_ _ℓ_ [1] (N) by


_b_ _j_ := _C_ _α_ exp( _−ℓ|κ_ ( _j_ ) _|_ _∞_ ) _,_ (4.24)


where _κ_ : N _→_ Z _[d]_ is the enumeration for the standard Fourier basis, defined
in appendix A. Note that by assumption on the enumeration _κ_, we have that
_b_ 1 _≥_ _b_ 2 _≥_ _. . ._ is a monotonically decreasing sequence. In the following, we will assume throughout that _∥_ _**b**_ _∥_ _ℓ_ 1 _<_ 1, ensuring the uniform coercivity condition _λ_ ( _a_ ) _≥_ _λ_
for some _λ >_ 0 and all random coefficients _a_ = _a_ ( _·_ ; _Y_ ) in (4.18). In (4.23), the
parameter _ℓ>_ 0 can be interpreted as the correlation length scale of the random
coefficients. We furthermore assume that the _Y_ _j_ _∈_ [ _−_ 1 _,_ 1] are centered random variables, implying that E[ _a_ ] = ~~_a_~~ ~~.~~ We let _µ ∈P_ ( _L_ [2] (T _[d]_ )) denote the law of the random
coefficient (4.22). By the assumed decay (4.23), we have supp( _µ_ ) _⊂_ _C_ _[∞]_ (T _[d]_ ).
Given the above setup, our aim is to find a DeepONet (2.10) which approximates
the underlying _G_, corresponding to the elliptic PDE (4.18) efficiently. To this end,
we will follow the program outlined at the beginning of this section, and bound
the encoding, reconstruction and approximation errors, separately in the following
sections.


4.2.2. _Bounds on the encoding error_ (3.1) _._ We are almost in the setup that was
already considered in section 3.5.3, with the only difference that we now consider
_X_ = _H_ _[s]_ (T _[d]_ ) for a fixed _s > d/_ 2, instead of _X_ = _L_ [2] (T _[d]_ ). We can again consider the
Fourier based encoder/decoder pair ( _E, D_ ) given by (3.53) and (3.58), respectively.
Applying a straightforward extension of Theorem 3.28 to _H_ _[s]_ (T _[d]_ ), we observe that


ERROR ESTIMATES FOR DEEPONETS 47


due to the exponential decay of the (Fourier) coefficients _α_ _k_, the exponential decay of the pseudo-spectral projection continues to hold also in the _H_ _[s]_ (T _[d]_ )-norm,
yielding the following error estimate for the encoding error _E_ [�] _E_ :


_**Proposition**_ **4.15** _**.**_ Given _m ∈_ N, let ( _E, D_ ) denote the Fourier based encoder/decoder
pair ( _E, D_ ) given by (3.53) and (3.58), respectively. There exists a constant _C >_ 0,
depending on _C_ _α_ _, ℓ>_ 0 and on _s > d/_ 2, but independent of _m_, and a universal
constant _c >_ 0, independent of _m_ such that


� 1 _/_ 2
_E_ _E_ = _∥D ◦E_ ( _u_ ) _−_ _u∥_ [2] _H_ _[s]_ (T _[d]_ ) _[dµ]_ [(] _[u]_ [)] _≤_ _C_ exp( _−cℓm_ [1] _[/d]_ ) _._ (4.25)
�ˆ _X_ �


4.2.3. _Bounds on the reconstruction error_ (3.3) _._ We follow the program outlined
at the beginning of this section and bound the reconstruction error via smoothness
of the image of the operator _G_ for the elliptic PDE (4.18). To this end, we have
the following Lemma (proved in Appendix E.8),


_**Lemma**_ **4.16** _**.**_ Let _k ∈_ N. Let _u_ be a solution of (4.18), with coefficient _a ∈_
_C_ _[∞]_ (T _[d]_ ), right-hand side _f ∈_ _H_ _[k]_ (T _[d]_ ) and _λ_ = min _x∈_ T _d_ _a_ ( _x_ ) _>_ 0. Then for any
_k ∈_ N, there exists a constant _C >_ 0, depending only on _k_ and _λ_, such that


_∥u∥_ [2] _H_ _[k]_ [+1] _[ ≤]_ _[C][∥][f]_ _[∥]_ _H_ [2] _[k]_ �1 + _∥a∥_ [2] _C_ _[k]_ _[k]_ � _._


Given the above smoothness estimate, one can directly apply Proposition 3.21
and Theorem 3.7 to obtain the following bound on the reconstruction error,


_**Proposition**_ **4.17** _**.**_ Let _µ ∈P_ ( _L_ [2] (T _[d]_ )) be a probability measure with supp( _µ_ ) _⊂_
_C_ _[k]_ (T _[d]_ ) _∩_ _L_ [2] (T _[d]_ ). Assume that there exists _λ_ 0 _>_ 0, such that _µ_ ( _λ_ ( _a_ ) _≥_ _λ_ 0 ) = 1,
and that ´ _L_ [2] _[ ∥][a][∥]_ [2] _C_ _[k]_ _[k]_ (T _[d]_ ) _[dµ]_ [(] _[a]_ [)] _[ <][ ∞]_ [. Define an operator] _[ G]_ [ :] _[ C]_ [(][T] _[d]_ [)] _[ →]_ _[L]_ [2] [(][T] _[d]_ [) by]

_a �→_ _u_ = _G_ ( _a_ ), where _u_ is the solution of (4.18) with a smooth right-hand side
_f ∈_ _C_ _[k]_ (T _[d]_ ). Then there exists a constant _C_ ( _k, ∥f_ _∥_ _C_ _k_ _, µ_ ) _>_ 0, depending only on
_∥f_ _∥_ _C_ _k_, _k_, and _µ_, such that for any _p ∈_ N, there exists a trunk net _**τ**_ : _U ⊂_ R _[n]_ _→_ R _[p]_,
_y �→_ _**τ**_ ( _y_ ) = (0 _, τ_ 1 ( _y_ ) _, . . ., τ_ _p_ ( _y_ )) with


size( _**τ**_ ) _≤_ _Cp_ (1 + log( _p_ ) [2] ) _,_ depth( _**τ**_ ) _≤_ _C_ (1 + log( _p_ ) [2] ) _,_


such that the corresponding reconstruction



_R_ : R _[p]_ _→_ _L_ [2] (T _[d]_ ) _≃_ _L_ [2] ([0 _,_ 2 _π_ ] _[d]_ ) _,_ _R_ ( _α_ ) :=


satisfies the following reconstruction error bound:



_p_
� _α_ _j_ _τ_ _j_ _,_

_j_ =1



�
_E_ _R_ _≤_ _C_ ( _k, ∥f_ _∥_ _C_ _k_ _, µ_ ) _p_ _[−][k/d]_ _._ (4.26)


Furthermore, the reconstruction _R_ and the associated projection _P_ : _L_ [2] (T _[d]_ ) _→_ R _[p]_

given by (3.19) satisfy Lip( _R_ ) _,_ Lip( _P_ ) _≤_ 2.


4.2.4. _Bounds on the approximation error_ (3.2) _._ For choices of the encoder _E_ in
Proposition 4.15, and the reconstructor _R_ in Proposition 4.17, the approximation
error with approximator network _A_ is given by
ˆ R _[m]_ _[ ∥A]_ [(] _[a]_ _[i]_ [)] _[ −P ◦G ◦D]_ [(] _[a]_ _[i]_ [)] _[∥]_ [2] _[ d]_ [(] _[E]_ [#] _[µ]_ [)(] _[a]_ _[i]_ [)] _[,]_


where the decoder _D_ is given by (3.58), and the projector _P_ has a Lipschitz constant
bounded by Lip( _P_ ) _≤_ 2.


48 ERROR ESTIMATES FOR DEEPONETS


From (Cohen et al. 2011, Theorem 1.3) (see also (Schwab & Zech 2019, Example
2.2) for a more detailed discussion relevant to the present setting), it follows that
the mapping
_F_ : [ _−_ 1 _,_ 1] [N] _→_ _L_ [2] (T _[d]_ ) _,_ _Y �→G_ ( _a_ ( _·_ ; _Y_ )) _,_
is ( _**b**_ _, ϵ, κ_ )-holomorphic according to our Definition 3.32, with _**b**_ defined by (4.24),
provided that _ϵ <_ 1 _−∥_ _**b**_ _∥_ _ℓ_ 1, and where _κ_ : N _→_ Z _[d]_ denotes the enumeration of the
standard Fourier basis (cp. appendix A). Following the discussion in section 3.6.2,
such _F_ can be efficiently approximated by neural networks. In particular, we can
directly apply Theorem 3.70, together with the observation that _∥P∥_ = Lip( _P_ ) _≤_ 2
is bounded independently of _m_ and _p_, to conclude the following bound on the
approximation error (3.2),


_**Proposition**_ **4.18** _**.**_ Let the operator _G_ be defined as mapping the coefficient _a_ to
the solution _u_ of the elliptic PDE (4.18) and the measure _µ_ be the law of the random
field (4.22). Let the encoder _E_ : _C_ (T _[d]_ ) _→_ R _[m]_ be given by (3.53) and the decoder
_D_ : R _[m]_ _→_ _L_ [2] (T _[d]_ ) be given by (3.58). Let the reconstruction/projection pair ( _R, P_ )
be given as in Proposition 4.17 for given _p ∈_ N. Then for any _k ∈_ N, there exists
a constant _C >_ 0, depending on _k_, but independent of the trunk net size _p_ and
number of sensors _m_, such that for any _m, p ∈_ N, there exists an approximator
network _A_, with


size( _A_ ) _≤_ _C_ (1 + _pm_ log( _m_ ) log log( _m_ )) _,_

(4.27)
depth( _A_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ ))


and approximation error


_E_ _A_ _≤_ _Cm_ _[−][k]_ _._ (4.28)


4.2.5. _Bounds on the DeepONet approximation error_ (2.11) _._ Finally, combining the
results of Propositions 4.15, 4.17, 4.18, we conclude that


_**Theorem**_ **4.19** _**.**_ For any _k, r ∈_ N, there exists a constant _C >_ 0, such that for any
_m, p ∈_ N, there exists a DeepONet _N_ = _R ◦A ◦E_ with _m_ sensors, a trunk net
_**τ**_ = (0 _, τ_ 1 _, . . ., τ_ _p_ ) with _p_ outputs and branch net _**β**_ = (0 _, β_ 1 _, . . ., β_ _p_ ), such that


size( _**β**_ ) _≤_ _C_ (1 + _pm_ log( _m_ ) log log( _m_ )) _,_


depth( _**β**_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ )) _,_


and


size( _**τ**_ ) _≤_ _C_ (1 + _p_ log( _p_ ) [2] )

depth( _**τ**_ ) _≤_ _C_ (1 + log( _p_ ) [2] )


such that the DeepONet approximation error (2.11) satisfies


� _d_ 1 _−k_ _−r_
_E ≤_ _Ce_ _[−][cℓm]_ + _Cm_ + _Cp_ _._ (4.29)


As the bound (4.29) is very similar to the bound (4.16), we can directly apply
the discussion in Remark 4.11 to derive the following complexity estimate

size( _**τ**_ _,_ _**β**_ ) _∼_ _ϵ_ _[−]_ [1] _[/r]_ [ �] _ϵ_ _[−]_ [2] _[/r]_ + _ϵ_ _[−]_ [1] _[/k]_ log( _ϵ_ _[−]_ [1] ) log log( _ϵ_ _[−]_ [1] ) _._ (4.30)
�


To derive (4.30), we choose _m ∼_ _ϵ_ _[−]_ [1] _[/k]_ sensors, and we assume that we are in the
regime, where _e_ _[−][cℓm]_ [1] _[/d]_ _≪_ _m_ _[−][k]_ _∼_ _ϵ_ . This clearly requires that _m ≫_ _ℓ_ _[−][d]_ _|_ log( _ϵ_ ) _|_ _[d]_,
i.e., that the _m_ sensors resolve scales of length _ℓ>_ 0 in the _d_ -dimensional domain
_D_ = [0 _,_ 2 _π_ ] _[d]_ . This is a reasonable assumption in low dimensions _d ∈{_ 1 _,_ 2 _,_ 3 _}_ of the
domain _D_, where this is practically feasible.


ERROR ESTIMATES FOR DEEPONETS 49


As the _k, r ∈_ N were arbitrary, this shows that the required size only grows
_sub-algebraically_ with _ϵ_ _[−]_ [1] _→∞_ . Thus from Definition 3.5, we can conclude that
there exists a DeepONet (2.10), which breaks the curse of dimensionality in approximating the nonlinear operator _G_ mapping the coefficient _a_ to the solution _u_
of the elliptic PDE (4.18).


_**Remark**_ **4.20** _**.**_ We emphasize that the above operator provides a mapping _G_ :
_C_ (T _[d]_ ) _→_ _L_ [2] (T _[d]_ ) in _low spatial dimensions d ∈{_ 1 _,_ 2 _,_ 3 _}_, in which case we show
that _G_ can be efficiently approximated by a DeepONet _N_ . Thus, the “curse of
dimensionality” here refers to the approximation problem for the high-dimensional
(in fact, _∞_ -dimensional) input-to-output mapping _G_, in accordance with our Definition 3.5 (cp. also Remark 3.4). Whether DeepONets also provide an efficient
approximation for similar operators with large _spatial dimension d ≫_ 1 will be the
subject of future research.



4.3. **Nonlinear parabolic PDE: A reaction-diffusion equation.** As a prototype for nonlinear reaction-diffusion parabolic type PDEs, we consider the following
version of the well-known Allen-Cahn equation which arises in the study of phase
transitions in materials such as alloys,

 _∂v_

_∂t_ [= ∆] _[v]_ [ +] _[ f]_ [(] _[v]_ [)] _[,]_ (4.31)





_∂v_



_∂t_ [= ∆] _[v]_ [ +] _[ f]_ [(] _[v]_ [)] _[,]_








_[,]_ (4.31)

_v_ ( _t_ = 0) = _u,_



where the non-linearity is given by _f_ ( _v_ ) = _v −_ _v_ [3] . For the sake of simplicity,
the Allen-Cahn PDE (4.31) is supplemented with periodic boundary conditions
on the space-time domain [0 _, T_ ] _×_ T _[d]_ . For initial data _u_ = _u_ ( _x_ ) drawn from a
probability measure _µ ∈P_ ( _L_ [2] (T _[d]_ )), our aim in this section will be to consider the
DeepONet approximation of the data-to-solution mapping _G_ : _L_ [2] (T _[d]_ ) _→_ _L_ [2] (T _[d]_ ),
_u �→G_ ( _u_ ) := _v_ ( _t_ = _T_ ), where _v_ solves (4.31).
As a first step, we need to show that the operator _G_ is well-defined. To this end,
we recall some well-known existence and boundedness results for the Allen-Cahn
equation (4.31),


_**Theorem**_ **4.21** (see e.g. (Yang et al. 2018, Cor. 1)) _**.**_ Let _v_ ( _x, t_ ) solve (4.31), with
initial data _u_ . If the initial data satisfies _∥u∥_ _L_ _[∞]_ _x_ _[≤]_ [1, then the solution of (4.31)]
satisfies _∥v_ ( _t_ ) _∥_ _L_ _∞x_ _[≤]_ [1 for all] _[ t][ ∈]_ [[0] _[, T]_ [].]


Note that the Allen-Cahn equation (4.31) is the _L_ [2] -gradient flow of a GinzburgLandau energy functional with a double-well potential. Normalizing the wells, it
makes sense to consider the initial data such that _−_ 1 _≤_ _u_ ( _x_ ) _≤_ 1 and the above
theorem guarantees that the maximum principle holds. Using standard parabolic
regularity theory, in appendix E.11, we prove the following regularity result for the
Allen-Cahn equation,


_**Theorem**_ **4.22** _**.**_ There exists an increasing function _η_ : [0 _, ∞_ ) _→_ [0 _, ∞_ ), _s �→_ _η_ ( _s_ ),
with the following property: If _u ∈_ _C_ [4] _[,α]_ (T _[d]_ ), _α ∈_ (0 _,_ 1), is initial data for the
Allen-Cahn equation (4.31) with _α_ -H¨older continuous 4th derivatives, and such that
_∥u∥_ _L_ _∞x_ _[≤]_ [1, then the solution] _[ v]_ [ of (4.31) has H¨older continuous partial derivatives]


_∂_ _[k]_ [+] _[ℓ]_ _v_
_,_ _∀_ _i_ 1 _, . . ., i_ _ℓ_ _∈{_ 1 _, . . ., d},_ 2 _k_ + _ℓ_ _≤_ 4 _,_
_∂t_ _[k]_ _∂x_ _i_ 1 _. . . ∂x_ _i_ _ℓ_

and _∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) defined by

_∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) := max _x∈_ T _[d]_ _[ ∥][v]_ [(] _[ ·][, x]_ [)] _[∥]_ _[C]_ [2] [([0] _[,T]_ [ ])] [ + max] _t∈_ [0 _,T_ ] _[∥][v]_ [(] _[t,][ ·]_ [ )] _[∥]_ _[C]_ [4] [(][T] _[d]_ [)] _[,]_ (4.32)


50 ERROR ESTIMATES FOR DEEPONETS


can be bounded from above


_∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) _≤_ _η_ � _∥u∥_ _C_ 4 _,α_ (T _d_ ) � _._ (4.33)


As a corollary of the above theorems, we can show in Appendix E.12, the following Lipschitz continuity of the solution mapping at time _T_ : _u �→_ _v_ ( _T_ ).


_**Corollary**_ **4.23** _**.**_ Let _α ∈_ (0 _,_ 1). Let _u, u_ _[′]_ _∈_ _C_ [4] _[,α]_ (T _[d]_ ) be such that _∥u∥_ _L_ _∞_ _, ∥u_ _[′]_ _∥_ _L_ _∞_ _≤_
1. Let _v, v_ _[′]_ _∈_ _C_ [(2] _[,]_ [4)] (T) denote the solution of (4.31) with initial data _u, u_ _[′]_, respectively. There exists a constant _C_ = _C_ ( _T_ ) _>_ 0, such that

_∥v_ ( _T_ ) _−_ _v_ _[′]_ ( _T_ ) _∥_ _L_ 2 (T _d_ ) _≤_ _C∥u −_ _u_ _[′]_ _∥_ _L_ 2 (T _d_ ) _._


It follows from Corollary 4.23 that the mapping _C_ [4] _[,α]_ (T _[d]_ ) _∩_ _B_ 1 _[∞]_ _→_ _L_ [2] (T _[d]_ ),
_u �→_ _v_ ( _T_ ), where _B_ 1 _[∞]_ = � _u ∈_ _C_ [4] _[,α]_ (T _[d]_ ) �� _∥u∥_ _L_ _∞_ _≤_ 1� admits a unique Lipschitz
continuous extension


_G_ : _L_ [2] (T _[d]_ ) _∩_ _B_ 1 _[∞]_ _[→]_ _[L]_ [2] [(][T] _[d]_ [)] _[,]_ _u �→G_ ( _u_ ) = _v_ ( _T_ ) _,_

with Lip( _G_ ) _≤_ _C_ = _e_ [4] _[T]_ . In the following, we will discuss the approximation of
this mapping _G_ by DeepONets (2.10). To this end, we assume that the underlying
measure _µ_ satisfies

supp( _µ_ ) _⊂_ � _u ∈_ _L_ [2] (T _[d]_ ) �� _∥u∥_ _L_ _∞_ _≤_ 1� _⊂_ _L_ [2] _x_ _[∩]_ _[L]_ _[∞]_ _x_ _[.]_ (4.34)


Motivated by the a priori estimates of Theorems 4.21 and 4.22, we shall assume
that the initial measure _µ_ is the law of a random field _u_ of the form

_u_ ( _x_ ) = � _α_ _k_ _Y_ _k_ **e** _k_ ( _x_ ) _,_ (4.35)

_k∈_ Z _[d]_


in terms of the Fourier basis **e** _k_ (cf. Appendix A) and with _Y_ _k_ _∈_ [ _−_ 1 _,_ 1] are iid
random variables. To ensure sufficient smoothness of the solutions, we further
assume that


_|α_ _k_ _| ≤_ _C_ exp ( _−ℓ|k|_ _∞_ ) _,_ _∀_ _k ∈_ Z _[d]_ _,_ (4.36)


for some _C >_ 0, and length scale _ℓ>_ 0; Note that (4.36) in particular implies that
_u ∈_ _C_ [4] _[,α]_ (T _[d]_ ) for some _α >_ 0. Furthermore, we shall assume that
� _|α_ _k_ _| ≤_ 1 _,_ (4.37)

_k∈_ Z _[d]_


so that _|u_ ( _x_ ) _| ≤_ 1 for all _x ∈_ T _[d]_ .


4.3.1. _Bounds on the encoding error_ (3.1) _._ With measure _µ_ defined by (4.35), we
can readily apply Theorem 3.28 to obtain the following bounds on the encoding

error:


_**Proposition**_ **4.24** _**.**_ Let _d ∈{_ 2 _,_ 3 _}_ . Let _µ ∈P_ ( _L_ [2] (T _[d]_ )) denote the law of the
random field (4.35) with coefficients _α_ _k_ satisfying the decay and boundedness assumptions (4.36), (4.37). For _N ∈_ N, let _x_ _i_, _i_ = 1 _, . . ., m_ = (2 _N_ + 1) _[d]_ be an
enumeration of the grid points of a regular cartesian grid on T _[d]_ . Define the encoder _E_ : _C_ (T _[d]_ ) _→_ R _[m]_ by


_E_ ( _u_ ) = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) _,_

and define the corresponding decoder _D_ : R _[m]_ _→_ _L_ [2] (T _[d]_ ) by Fourier interpolation
onto Fourier modes _|k|_ _∞_ _≤_ _N_ . Then the encoding error for the encoder/decoder
pair ( _E, D_ ) can be bounded by

�
_E_ _E_ _≤_ _C_ exp( _−cℓm_ [1] _[/d]_ ) _,_


for some constants _C, c >_ 0, independent of _N_ .


ERROR ESTIMATES FOR DEEPONETS 51


4.3.2. _Bounds on the reconstruction error_ (3.3) _._ To bound the reconstruction error,
we recall that for _u ∈_ _C_ [4] _[,α]_ (T _[d]_ ), we have _G_ ( _u_ ) = _v_ ( _T_ ) _∈_ _C_ [4] (T _[d]_ ), by Theorem 4.22,
with



�



_∥G_ ( _u_ ) _∥_ _C_ 4 (T _d_ ) _≤_ _η_ � _∥u∥_ _C_ 4 _,α_ (T _d_ ) � _≤_ _η_



sup _∥u∥_ _C_ 4 _,α_ (T _d_ )

� _u∈_ supp( _µ_ )



_< ∞,_



uniformly bounded from above. It follows that _∥G_ ( _u_ ) _∥_ _H_ 4 _≤_ _M_ is uniformly bounded,
and by (3.34), the reconstruction error for the Fourier reconstructor/projector pair
onto the standard Fourier basis **e** 1 _, . . .,_ **e** _p_, satisfies


�
_E_ _R_ Fourier _≤_ _Cp_ _[−]_ [4] _[/d]_ _._


From Lemma 3.17, we obtain:


_**Proposition**_ **4.25** _**.**_ There exists a constant _C >_ 0, independent of _p_, such that for
any _p ∈_ N, there exists a trunk net _**τ**_ : _U_ = T _[d]_ _⊂_ R _[d]_ _→_ R _[p]_, with


size( _**τ**_ ) _≤_ _Cp_ (1 + log( _p_ ) [2] ) _,_

depth( _**τ**_ ) _≤_ _C_ (1 + log( _p_ ) [2] ) _,_


and such that the reconstruction error for _R_ : R _[p]_ _→_ _L_ [2] (T _[d]_ ), _R_ ( _α_ 1 _, . . ., α_ _p_ ) =
� _pk_ =1 _[α]_ _[k]_ _[τ]_ _[k]_ [, can be bounded by]


�
_E_ _R_ _≤_ _Cp_ _[−]_ [4] _[/d]_ _._ (4.38)


Furthermore, we have Lip( _R_ ) _,_ Lip( _P_ ) _≤_ 2, where _P_ : _L_ [2] (T _[n]_ ) _→_ R _[p]_ denotes the
optimal projection (3.19) associated with _R_ .


4.3.3. _Bounds on the approximation error_ (3.2) _._ For the last two concrete examples, bounds on the approximation error (3.2) leveraged the fact that the underlying
operator _G_ was _holomorphic_ in an appropriate sense. However, it is unclear if the
operator _G_ for the Allen-Cahn equation (4.31) is holomorphic. In fact, we have
only provided that it is Lipschitz continuous. Hence, the approximator network
_A ≈P ◦G ◦D_ : R _[m]_ _→_ R _[p]_ approximates a _m_ -dimensional Lipschitz function. General neural network approximation results for such mappings Yarotsky (2017) imply
that for a fixed _M >_ 0, there exists a neural network _A_ with size( _A_ ) = _O_ ( _pϵ_ _[−]_ [1] _[/m]_ ),
such that _∥A −P ◦G ◦D∥_ _L_ _∞_ ([ _−M,M_ ] _m_ ) _≤_ _ϵ_ .
Recall that the total error (2.11) for the DeepONet is given by (3.4) with _α_ = 1
as _G_ is Lipschitz. In this case from Proposition 4.24, we have that


�
_E_ _E_ ≲ exp( _−cm_ ) _,_


we require at least _m ∼_ log( _ϵ_ _[−]_ [1] ) sensors to achieve an encoding error _E_ [�] _E_ _≤_ _ϵ_ .
Therefore, the general results of Yarotsky (2017) would suggest that the required
size of the approximator network _A_ scales at best like _ϵ_ _[−|]_ [ log(] _[ϵ]_ [)] _[|]_, where we note
that the exponent _|_ log( _ϵ_ ) _| →∞_ as _ϵ →_ 0. Hence, by our Definition 3.5, such a
DeepONet would incur the curse of dimensionality.
Is it possible for us to break this curse of dimensionality for the Allen-Cahn equation? It turns out that the approach of a recent paper DeRyck & Mishra (2021)
might suggest a way around the obstacle discussed above. Following DeRyck &
Mishra (2021), we will leverage the fact that neural networks can _emulate_ conventional numerical methods for approximating a PDE. To this end, we will consider
the following finite difference scheme:


52 ERROR ESTIMATES FOR DEEPONETS


4.3.4. _A convergence finite difference scheme for the Allen-Cahn equation._ In Tang
& Yang (2016), it has been shown that an implicit-explicit finite difference scheme
of the following form


_U_ _[n]_ [+1] _−_ _U_ _[n]_

= _D_ ∆ _x_ _U_ _[n]_ + _f_ ( _U_ _[n]_ ) _,_
∆ _t_


converges to the exact solution as ∆ _t,_ ∆ _h →_ 0. Here _U_ _[n]_ = ( _U_ 1 _[n]_ _[, . . ., U]_ _m_ _[ n]_ [),] _[ m][ ∼]_
(∆ _x_ ) _[−][d]_ are approximate values of the solution at time _t_ = _t_ _n_, i.e. _U_ _i_ _[n]_ _[≈]_ _[u]_ [(] _[x]_ _[i]_ _[, t]_ _[n]_ [),]
with _x_ _i_, _i_ = 1 _, . . ., m_ = (2 _N_ + 1) _[d]_ an enumeration of a cartesian grid with grid size
∆ _x_ on T _[d]_ . The values of _U_ [0] at _t_ = 0 are initialized as


_U_ _i_ [0] [:=] _[ u]_ [(] _[x]_ _[i]_ [)] _[.]_


The evaluation of the nonlinearity is carried out pointwise, _f_ ( _U_ _[n]_ ) = ( _f_ ( _U_ 1 _[n]_ [)] _[, . . ., f]_ [(] _[U]_ _m_ _[ n]_ [)).]
_D_ ∆ _x_ denotes the discrete matrix of the Laplace operator, whose one-dimensional
analogue in the presence of periodic boundary conditions is given by














1
Λ ∆ _x_ = ∆ _x_ [2]














_−_ 2 1 1

1 _−_ 2 1

... ... ...

1 _−_ 2 1

1 1 _−_ 2



_N_ _×N_

For _d_ = 2 dimensions, we can write



_D_ ∆ _x_ = Λ ∆ _x_ _⊗_ _I_ + _I ⊗_ Λ ∆ _x_ _,_


where _I_ is the _m × m_ unit-matrix _⊗_ denotes the Kronecker product. For _d_ = 3, we
have

_D_ ∆ _x_ = Λ ∆ _x_ _⊗_ _I ⊗_ _I_ + _I ⊗_ Λ ∆ _x_ _⊗_ _I_ + _I ⊗_ _I ⊗_ Λ ∆ _x_ _._

For our purposes, we simply note that the update rule of the numerical scheme
of Tang & Yang (2016) can be written in the form

_U_ _[n]_ [+1] = _R_ ∆ _t,_ ∆ _x_ ( _U_ _[n]_ + ∆ _tf_ ( _U_ _[n]_ )) _,_ _U_ _i_ [0] [=] _[ u]_ [(] _[x]_ _[i]_ [)] _[, i]_ [ = 1] _[, . . ., m.]_ (4.39)


where _R_ ∆ _t,_ ∆ _x_ = ( _I −_ ∆ _tD_ ∆ _x_ ) _[−]_ [1] is a _m × m_ matrix. We furthermore note the
following result of Tang & Yang (2016):


_**Theorem**_ **4.26** ((Tang & Yang 2016, Thm. 2.1)) _**.**_ Consider the Allen-Cahn problem (4.31) with periodic boundary conditions. If the initial value is bounded by 1,
i.e. max _x∈_ T _d_ _|u_ ( _x_ ) _| ≤_ 1, then the numerical solution of the fully discrete scheme
(4.39) is also bounded by 1 in the sense that _∥U_ _[n]_ _∥_ _ℓ_ _∞_ _≤_ 1 for all _n >_ 0, provided
that the stepsize satisfies 0 _<_ ∆ _t ≤_ 2 [1] [.]


We also prove in Appendix E.11, the following convergence result for the scheme
(4.39),


_**Theorem**_ **4.27** _**.**_ Consider the Allen-Cahn problem (4.31) with periodic boundary
conditions. Assume that the solution _v ∈_ _C_ [(2] _[,]_ [4)] ([0 _, T_ ] _×_ T _[d]_ ), there exists a constant
_C >_ 0 independent of _v_, ∆ _t_ and ∆ _x_, such that the error _E_ _i_ _[n]_ [:=] _[ |][U]_ _i_ _[ n]_ _[−]_ _[v]_ [(] _[x]_ _[i]_ _[, t]_ _[n]_ [)] _[|]_ [,]
_i_ = 1 _, . . ., m_ is bounded by

_∥E_ _[n]_ _∥_ _ℓ_ _∞_ _≤_ (∆ _t_ + ∆ _x_ [2] ) exp � _Ct_ _n_ _∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) � _._


Next, we proceed to bound the approximation error (3.2). To provide an upper
bound on the size of the approximator network _A_, we next show that the numerical
scheme (4.39) can be efficiently approximated by a suitable neural network. By
“efficient”, we imply that the required size of the neural network _A_ increases at


ERROR ESTIMATES FOR DEEPONETS 53


most polynomially with the number of sensor points _m_, rather than exponentially.
We begin with the following observation, proved in Appendix E.12,


_**Lemma**_ **4.28** _**.**_ Let _f_ ( _v_ ) = _v −_ _v_ [3] be the nonlinearity in the Allen-Cahn equation
(4.31). There exist constants _C, M >_ 0, such that for any _ϵ ∈_ (0 _,_ 1), there exists
a ReLU neural network _g_ _ϵ_ : R _→_ R, _ξ �→_ _g_ _ϵ_ ( _ξ_ ) with size( _g_ _ϵ_ ) _≤_ _C_ (1 + _|_ log( _ϵ_ ) _|_ ),
depth( _g_ _ϵ_ ) _≤_ _C_ (1 + _|_ log( _ϵ_ ) _|_ ), and such that


sup _|f_ ( _η_ ) _−_ _g_ _ϵ_ ( _η_ ) _| < ϵ,_
_η∈_ [ _−_ 1 _,_ 1]


and Lip( _g_ _ϵ_ ) _≤_ _M_ .


Thus, the above lemma shows that there exists a small neural network with ReLU
activation function that approximates the nonlinearity in the Allen-Cahn equations
to high accuracy. In fact, if we use a smooth, i.e. _C_ [3] activation function, one can
see from Remark E.4 in Appendix E.12 that an even smaller network (with size
that does not need to increase with increasing accuracy) suffices to represent this
nonlinearity accurately. However, we stick to ReLU activation functions here for
definiteness. Given _ϵ >_ 0, let _g_ _ϵ_ : R _→_ R be a ReLU neural network as in Lemma
4.28. It is now clear that if _g_ _ϵ_ ( _η_ ) is represented by a (small) neural network, then
there exists a larger neural network _N_, such that


_N_ ( _U_ [�] [0] ) = _U_ [�] _[n]_ _,_



where _U_ [�] _[k]_, _k_ = 1 _, . . ., n_ is determined by the recursion relation

 _U_ � _[k]_ [+1] = _R_ ∆ _x,_ ∆ _t_ _U_ � _[k]_ + ∆ _tg_ _ϵ_ ( _U_ � _[k]_ )

� �


�



[0]



� � �
_U_ _[k]_ [+1] = _R_ ∆ _x,_ ∆ _t_ _U_ _[k]_ + ∆ _tg_ _ϵ_ ( _U_ _[k]_ )
� �







_U_ � [0] _∈_ R _[m]_ _._ (4.40)



More precisely, we have the following lemma, proved in Appendix E.13, for the _em-_
_ulation_ of the finite difference scheme (4.40) with a suitable ReLU neural network,


_**Lemma**_ **4.29** _**.**_ There exists a constant _C >_ 0, such that for any _m ∈_ N, _ϵ >_ 0, there
exists a neural network _N_ with size( _N_ ) _≤_ _Cn_ ( _m_ [2] + _m|_ log( _ϵ_ ) _|_ ), and depth( _N_ ) _≤_
_C_ (1 + _n|_ log _ϵ|_ ), such that _N_ ( _U_ [�] [0] ) = _U_ [�] _[n]_, maps any initial data _U_ [�] [0] _∈_ R _[m]_ to the
solution _U_ [�] _[n]_ of the recursion (4.40).


Finally, we estimate the _error_ i.e. difference between the result _U_ _[n]_ of the exact
update rule for the numerical scheme (4.39), and the approximate neural network
version (4.40) _U_ [�] _[n]_ in the following Lemma (proved in Appendix E.14).


_**Lemma**_ **4.30** _**.**_ Let _U_ _[n]_, _U_ [�] _[n]_ be obtained by (4.39) and (4.40), respectively, with
initial data _U_ [0] = _U_ [�] [0] such that _∥U_ [0] _∥_ _ℓ_ _∞_ _≤_ 1, and Lip( _g_ _ϵ_ ) _≤_ _M_ . Then


_∥U_ _[n]_ _−_ _U_ [�] _[n]_ _∥_ _ℓ_ _∞_ _≤_ _Te_ _[MT]_ _ϵ,_


where _T_ = _n_ ∆ _t_ .


After having emulated the finite difference scheme (4.39) with a ReLU neural
network and estimated the error in doing so, we are in a position to state the bounds
on the approximation error in the following proposition, proved in Appendix E.15,


54 ERROR ESTIMATES FOR DEEPONETS


_**Proposition**_ **4.31** _**.**_ Let _µ_ be given as the law of (4.35). Let the encoder/decoder
pair _E_, _D_ be given as in Proposition 4.24 for some _m ∈_ N. Let _R_ be given as in
Proposition 4.25, for _p ∈_ N. There exists a constant _C >_ 0, independent of _m_ and
_p_, such that there exists a neural network _A_ : R _[m]_ _→_ R _[p]_, such that

size( _A_ ) _≤_ _C_ (1 + _m_ [2+2] _[/d]_ + _mp_ ) _,_ depth( _A_ ) _≤_ _C_ (1 + _m_ [2] _[/d]_ log( _m_ )) _,_


and

�
_E_ _A_ _≤_ _Cm_ _[−]_ [1] _[/d]_ _._


_**Remark**_ **4.32** _**.**_ Choosing the number of trunk nets _p_ ≲ _m_ [1+2] _[/d]_, it follows from
the previous proposition that an approximation error _A_ of order _ϵ_ can be achieved,
with _m ∼_ _ϵ_ _[−][d]_, i.e. a neural network of size

size( _A_ ) = _O_ ( _ϵ_ _[−]_ [2(] _[d]_ [+1)] ) _,_ depth( _A_ ) = _O_ ( _ϵ_ _[−]_ [2] _|_ log( _ϵ_ ) _|_ ) _._


This is polynomial in _ϵ_ _[−]_ [1] (recall that _d_ = 2 _,_ 3 is a fixed constant independent of
the accuracy _ϵ_ ), and hence does not suffer from the curse of dimensionality for
the infinite-dimensional approximation problem (cp. Definition 3.5). We would
nevertheless expect that sharper error estimates could be obtained by basing the
neural network emulation result on higher-order finite difference methods. In the
present work, our main objective is to show that neural networks can break the
curse of dimensionality, rather than attempting to establish optimal complexity
bounds.


4.3.5. _Bounds on the DeepONet approximation error_ (2.11) _._ Combining Propositions 4.24, 4.25, 4.31, we can now state the following theorem


_**Theorem**_ **4.33** _**.**_ Consider the DeepONet approximation problem for the AllenCahn equation (4.31), where the initial data _u_ is distributed according to a probability measure _µ ∈P_ ( _L_ [2] (T _[d]_ )) is the law of the random field (4.35). There exist
constants _C, c >_ 0, such that for any _m, p ∈_ N, there exists a DeepONet (2.10)
with trunk net _**τ**_ and branch net _**β**_, such that


size( _**τ**_ ) _≤_ _C_ (1 + _p_ log( _p_ ) [2] ) _,_ depth( _**τ**_ ) _≤_ _C_ (1 + log( _p_ ) [2] ) _,_


and


size( _**β**_ ) _≤_ _C_ (1 + _m_ [2+2] _[/d]_ + _pm_ ) _,_


depth( _**β**_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ )) _,_


and such that the DeepONet approximation (2.11) is bounded by


�
_E ≤_ _C_ exp( _−cℓm_ [1] _[/d]_ ) + _Cp_ _[−]_ [4] _[/d]_ + _Cm_ _[−]_ [1] _[/d]_ _._ (4.41)


_**Remark**_ **4.34** _**.**_ The bound (4.41) guarantees that for _ϵ >_ 0, a DeepONet approximation error of _E_ [�] _∼_ _ϵ_ can be achieved provided that _m_ ≳ max( _ℓ_ _[−][d]_ log( _ϵ_ _[−]_ [1] ) _[d]_ _, ϵ_ _[−][d]_ )
and _p_ ≳ _ϵ_ _[−][d/]_ [4] . Assuming that _m_ [1] _[/d]_ _ℓ_ _≫_ 1 is satisfied, i.e., that the sensors can
resolve the typical correlation length scale _ℓ_ in the _d_ -dimensional domain _D_ = T _[d]_,
this can be ensured provided that� _m_ ≳ _ϵ_ _[−][d]_ and _p_ ≳ _ϵ_ _[−][d/]_ [4] . In this case, an error
_E_ ≲ _ϵ_ can thus be achieved with an overall DeepONet size of order

size( _**τ**_ _,_ _**β**_ ) ≲ _ϵ_ _[−][d/]_ [4] log( _ϵ_ _[−]_ [1] ) [2] + _ϵ_ _[−]_ [2(] _[d]_ [+1)] ≲ _ϵ_ _[−]_ [2(] _[d]_ [+1)] _._ (4.42)


For the cases of interest, _d_ = 2 _,_ 3, this upper bound on the required size of the
DeepONet thus scales as size ≲ _ϵ_ _[−]_ [6], and size ≲ _ϵ_ _[−]_ [8], respectively. This shows that
the required size scales at worst algebraically in _ϵ_ _[−]_ [1], thus _breaking the curse of_
_dimensionality_, as per Definition 3.5. As already pointed out in Remark 4.32, the
explicit exponents _−_ 6 and _−_ 8 may be considerably improved if the emulation result


ERROR ESTIMATES FOR DEEPONETS 55


were based on a _higher-order order_ numerical scheme in place of the low-order finite
difference scheme (4.39).


4.4. **Nonlinear hyperbolic PDE: Scalar conservation laws.** As a final concrete example in this paper, we will consider this prototypical example of nonlinear
hyperbolic PDEs. We remark that the image of the underlying operator _G_ in the
previous three examples consisted of smooth functions. On the other hand, it is
well known that solutions of conservation laws can be discontinuous, on account
of the formation of shock waves. Thus, one of our objectives in this section is to
show that DeepONets can even approximate nonlinear operators, that map into
discontinuous functions, efficiently.


4.4.1. _Problem formulation._ We consider a scalar conservation law on a one-dimensional
domain _D ⊂_ R:

_∂_ _t_ _v_ + _∂_ _x_ ( _f_ ( _v_ )) = 0 _,_

(4.43)

� _v_ ( _t_ = 0) = _u,_


with initial data _u_ drawn from some underlying measure _µ_, and with a given flux
function _f ∈_ _C_ [2] (R). For simplicity, we will assume that _D_ = T = [0 _,_ 2 _π_ ], and
periodic boundary conditions for (4.43). Our results are however readily generalized
to other boundary conditions and to several space dimensions. Solutions of (4.43)
are interpreted in the weak sense, imposing the entropy conditions


_∂_ _t_ _η_ ( _v_ ) + _∂_ _x_ _q_ ( _v_ ) _≤_ 0 _,_ (in the distributional sense)


for all entropy/entropy flux pairs ( _η, q_ ), consisting of a convex _C_ [1] -function _η_ : R _→_
R and the corresponding flux _q_ : R _→_ R with derivative satisfying _q_ _[′]_ ( _u_ ) = _f_ _[′]_ ( _u_ ) _η_ _[′]_ ( _u_ )
for all _u ∈_ R.
Under these conditions, it is well known Godlewski & Raviart (1991) that the
solution _v_ of (4.43) is unique for any initial data _u ∈_ _L_ [1] (T) _∩_ _L_ _[∞]_ (T), and the
solution operator _S_ _t_ is contractive as a function _L_ [1] (T) _→_ _L_ [1] (T) for any _t ∈_ [0 _, ∞_ ):


_∥S_ _t_ ( _u_ ) _−S_ _t_ ( _u_ _[′]_ ) _∥_ _L_ 1 (T) _≤∥u −_ _u_ _[′]_ _∥_ _L_ 1 (T) _,_ _∀_ _u, u_ _[′]_ _∈_ _L_ [1] (T) _∩_ _L_ _[∞]_ (T) _._ (4.44)


We note that _S_ _t_ : BV(T) _→_ BV(T), maps functions of bounded variation to functions of bounded variation, in fact we have


_∥S_ _t_ ( _u_ ) _∥_ BV _≤∥u∥_ BV _._ (4.45)


Here, we denote _∥· ∥_ BV = _∥· ∥_ _L_ 1 +TV( _·_ ), as the BV-norm with TV( _w_ ) representing
the total variation of a function _w_ Godlewski & Raviart (1991). Furthermore, _S_ _t_ ( _u_ )
satisfies the maximum principle, so that


_∥S_ _t_ ( _u_ ) _∥_ _L_ _∞_ (T) _≤∥u∥_ _L_ _∞_ _._


Next, in order to specify the DeepONet approximation problem (cf. Definition
2.1), we take the nonlinear operator _G_ : _L_ [1] ( _D_ ) _→_ _L_ [1] ( _D_ ) as _G_ ( _u_ ) := _S_ _T_ ( _u_ ), mapping
the initial data _u_ of the scalar conservation law (4.43) to the solution _S_ _T_ ( _u_ ) =
_v_ ( _·, t_ = _T_ ) at the final time _T >_ 0. Clearly, given the _L_ [1] -contractivity (4.44), the
operator _G_ is well-defined and Lipschitz continuous.
Defining the set


BV _M_ := _{u ∈_ BV(T) _| ∥u∥_ BV _≤_ _M_ _},_ (4.46)


we will consider any measure _µ_ that satisfies _µ_ (BV _M_ ) = 1, as the underlying
measure for the DeepONet approximation problem.


56 ERROR ESTIMATES FOR DEEPONETS


4.4.2. _DeepONet approximation in the Banach space L_ [1] ( _D_ ) _._ So far, we have only
considered the DeepONet approximation of operators, defined on the Hilbert spaces.
However, given the fact that the solution operator _S_ _t_ and the resulting operator
_G_ are contractive on the Banach space _L_ [1] ( _D_ ), it is very natural to consider the
DeepONet approximation problem in this function space. Thus, we need to modify
the definition of the DeepONet error (2.11) and define its _L_ [1] -version by,


�
_E_ _L_ 1 = _∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ _L_ 1 _dµ_ ( _u_ ) _,_ (4.47)
ˆ _L_ [1] ( _D_ )


with the DeepONet _N_ (2.10) approximating the operator _G_ . It is of the form,


_N_ = _R ◦A ◦E,_ (4.48)


with _R_ : R _[p]_ _→_ _C_ ( _D_ ) is the usual affine reconstruction defined in (2.9) based on
the trunk network _**τ**_, _A_ : R _[m]_ _→_ R _[p]_ is the approximator neural network used
to define the branch network _**β**_, and we have introduced a generalized encoder
_E_ : _L_ [1] ( _D_ ) _→_ R _[m]_, which is defined by taking local averages in cells _C_ 1 _, . . ., C_ _m_ _⊂_ _D_ :



_E_ ( _u_ ) = _u_ ( _x_ ) _dx, . . .,_
� _C_ 1


with cells _C_ 1 _, . . ., C_ _m_ given by



_u_ ( _x_ ) _dx_ _,_ (4.49)
_C_ _m_ �



_C_ _j_ := [ _x_ _j_ _−_ ∆ _x/_ 2 _, x_ _j_ + ∆ _x/_ 2] _,_ (for _j_ = 1 _, . . ., m_ ) _,_ (4.50)


where _x_ 1 _, . . ., x_ _m_ denote _m_ equidistant sensors on the periodic domain _D_ = T, with
∆ _x_ = 2 _π/m_ . Note that the encoder _E_ is well-defined (and in fact continuous) for
any _u ∈_ _L_ [1] ( _D_ ). In particular, this encoding allows us to consider discontinuous
initial data _u_, and thus we do not need to assume that the underlying measure _µ_
is concentrated on _C_ ( _D_ ). This choice of encoder constitutes a key difference with
pointwise encoder _E_, considered in section 2 and all the previous examples.
Given this architecture for the DeepONet (4.48), we aim to bound the resulting approximation error (4.47). As we are no longer in the Hilbert space setting,
we cannot directly appeal to the abstract error estimates of section 3, nor follow
the program outlined at the beginning of this section. Nevertheless, we will still
follow a key idea from the last subsection, namely emulating numerical schemes
approximating the underlying PDEs with neural networks. To this end, we recall
a straightforward adaptation of the results of a recent paper DeRyck & Mishra
(2021);


_**Theorem**_ **4.35** _**.**_ Let _M >_ 0 be given. Consider initial data _u ∈_ BV _M_ for the
scalar conservation law (4.43) with flux function _f ∈_ _C_ [2] (R). Let the operator
_G_ ( _u_ ) = _S_ _T_ ( _u_ ) be given by mapping _u �→_ _v_ ( _T, ·_ ), where _v_ solves (4.43). There
exists a constant _C_ = _C_ ( _T, ∥f_ _∥_ _C_ 2 ) _>_ 0, such that for any _m ∈_ N, there exists a
neural network _N_ : R _[m]_ _→_ R _[m]_ with


size( _N_ ) _≤_ _Cm_ [5] _[/]_ [2] _,_ depth( _N_ ) _≤_ _Cm,_



such that

_G_ ( _u_ ) _−_
������



_m_
�



������ _L_ [1] ( _D_ )



� _N_ _j_ ( _E_ ( _u_ )) 1 _C_ _i_ ( _·_ )

_j_ =1



_≤_ _m_ _[C]_ _[α]_ _[.]_



Here _α >_ 0 is the convergence rate of the well-known Lax-Friedrichs scheme
Godlewski & Raviart (1991) and 1 _C_ _j_ ( _·_ ) denotes the indicator function of the cell
_C_ _i_ (cp. (4.50)). For the neural network _N_ we furthermore have


_∥N_ ( _E_ ( _u_ )) _∥_ _ℓ_ _∞_ _≤_ _M,_ for all _u ∈_ BV _M_ .


ERROR ESTIMATES FOR DEEPONETS 57


_**Remark**_ **4.36** _**.**_ The worst-case estimate for the convergence rate _α_ of the LaxFriedrichs scheme guarantees that _α ≥_ 2 [1] [Godlewski & Raviart (1991). However, in]

practice, the observed convergence rate is often higher, _α ≈_ 1.


The proof of Theorem 4.35 is based on the fact that neural networks can emulate
the Lax-Friedrichs difference scheme for (4.43). In fact, the neural network _N_
constructed in DeRyck & Mishra (2021) is an exact form of the Lax-Friedrichs
scheme applied to a scalar conservation law (4.43), but with a neural network
approximated flux _f_ [�] _≈_ _f_, and satisfying a suitable CFL condition. The bound on
_∥N_ ( _E_ ( _u_ )) _∥_ _ℓ_ _[∞]_ _≤_ _M_ therefore follows from the fact that _∥u∥_ _L_ _[∞]_ _≤∥u∥_ BV _≤_ _M_ and
the fact that the Lax-Friedrichs scheme satisfies the maximum principle.
To obtain a DeepONet approximation result from 4.35, we have the following
simple Lemma (proved in Appendix E.16), on the approximation of characteristic
functions by ReLU neural networks,


_**Lemma**_ **4.37** _**.**_ There exists a constant _C >_ 0, such that for any _a, b ∈_ R, _a < b_,
and _ϵ >_ 0, there exists a ReLU neural network _χ_ _[ϵ]_ [ _a,b_ ] [:][ R] _[ →]_ [R][, with]

size( _χ_ _[ϵ]_ [ _a,b_ ] [)] _[ ≤]_ _[C,]_ depth( _χ_ _[ϵ]_ [ _a,b_ ] [) = 1] _[,]_


and
_∥χ_ _[ϵ]_ [ _a,b_ ] _[−]_ [1] [[] _[a,b]_ []] _[∥]_ _[L]_ [1] [(][R][)] _[≤]_ _[ϵ.]_


This allows us to prove in Appendix E.17, our main approximation result for the
DeepONet approximation problem (4.47) for scalar conservation laws,


_**Theorem**_ **4.38** _**.**_ Let _µ ∈P_ ( _L_ [1] ( _D_ )) be a probability measure such that there exists
_M >_ 0 such that _µ_ (BV _M_ ) = 1. Let the underlying operator _G_ map the initial data
_u_ to the solution (at final time) _v_ ( _., T_ ) of the scalar conservation law (4.43). Let
the encoder _E_ of the DeepONet (4.48) be given by (4.49) with _m_ equidistant cells
_C_ _j_ (4.50). There exists a constant _C >_ 0, independent of _m_ and _p_, such that for
any _m, p ∈_ N, there exists a DeepONet ( _**τ**_ _,_ _**β**_ ) with trunk net _**τ**_ : R _[p]_ _→_ _C_ ( _D_ ),
and branch net _**β**_ : _L_ [1] ( _D_ ) _→_ R _[p]_, of the form _**β**_ ( _u_ ) = _A_ ( _E_ ( _u_ )) for a neural network
_A_ : R _[m]_ _→_ R _[p]_, with

size( _**β**_ ) _≤_ _Cm_ [5] _[/]_ [2] _,_ depth( _**β**_ ) _≤_ _Cm,_


and

size( _**τ**_ ) _≤_ _Cp,_ depth( _**τ**_ ) = _C,_

such that

�
_E_ _L_ 1 _≤_ _C_ max 1 _−_ _[p]_ + _Cm_ _[−][α]_ _,_ (4.51)
� _m_ _[,]_ [ 0] �

with the DeepONet approximation error _E_ [�] _L_ 1 defined in (4.47).


_**Remark**_ **4.39** _**.**_ To achieve an error _E_ [�] _L_ 1 _∼_ _ϵ_, the estimate (4.51) provided by Theorem 4.38 shows that it is sufficient that _p ≥_ (1 _−_ _ϵ_ ) _m_, i.e. _p ∼_ _m_, and _m_ ≳ _ϵ_ _[−]_ [1] _[/α]_ .
Thus, a DeepONet approximation error of order _ϵ_ can be achieved with a DeepONet
( _**β**_ _,_ _**τ**_ ) of size
size( _**β**_ ) _∼_ _ϵ_ _[−]_ [5] _[/]_ [2] _[α]_ _,_ size( _**τ**_ ) _∼_ _ϵ_ _[−]_ [1] _[/α]_ _._ (4.52)
For the worst-case rate of _α_ = 1 _/_ 2, this yields a total DeepONet size of order _∼_ _ϵ_ _[−]_ [5] .
For more realistic values of _α ≈_ 1, we require a size of order _∼_ _ϵ_ _[−]_ [2] _[.]_ [5] . Clearly, this
scales polynomially in _ϵ_ _[−]_ [1], and thus DeepONets can efficiently approximate the solution operator for scalar conservation laws _by breaking the curse of dimensionality_,
see Definition 3.5.


58 ERROR ESTIMATES FOR DEEPONETS


4.4.3. _A lower bound on the DeepONet approximation error._ Theorem 4.38 shows
that with the natural scaling, _p_ = _m_, we have _E_ [�] _L_ 1 _≤_ _Cp_ _[−][α]_, where _α >_ 0 is the
convergence rate of the Lax-Friedrichs scheme, and _p ∈_ N is the output dimension
of the branch/trunk net _**β**_ _,_ _**τ**_ . The goal of this section is to present an example of
a measure _µ ∈_ BV _M_, for which we also have a lower bound of the form _E_ [�] _L_ 1 ≳ _p_ _[−]_ [1]

and demonstrate that the DeepONet approximation, as described above, _is almost_
_optimal_ .
To this end, we will rely on the lower bound (3.9) for the ( _L_ [2] -based) error

� �
_E ≥_ _E_ _R_ _≥_ � ~~�~~ _k>p_ _[λ]_ _[k]_ [, of Theorem 3.6. Recall][ �] _[E]_ [ denotes the DeepONet error (2.11)]

in the _L_ [2] norm, _E_ [�] _R_ denotes the reconstruction error (3.3) and _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote
the eigenvalues of the covariance operator Γ _G_ # _µ_ = ´ ( _v−_ E[ _v_ ]) _⊗_ ( _v−_ E[ _v_ ]) _d_ ( _G_ # _µ_ )( _v_ ),
associated with the push-forward measure _G_ # _µ_ .
To provide a concrete example which exhibits the decay _E_ [�] _L_ 1 ≲ _Cp_ _[−]_ [1] in terms of
the output dimension _p_ of the trunk net, we now consider the measure _µ ∈P_ ( _L_ [2] (T))
given as the law


�
_µ_ = law _{−_ sin( _· −_ _x_ ) _|_ � _x ∼_ Unif(T) _},_ (4.53)


for the Burgers’ equation, i.e., scalar conservation law (4.43),

_∂_ _t_ _v_ + _∂_ _x_ � _v_ [2] _/_ 2� = 0 _,_

(4.54)

� _v_ ( _t_ = 0) = _u._


Let _v_ 0 ( _x, t_ ) denote the solution of (4.54) with initial data


_v_ 0 ( _x, t_ = 0) = _u_ 0 ( _x_ ) := _−_ sin( _x_ ) _._


The characteristic starting at _x_ 0 for this data is given by


_x_ ( _t_ ; _x_ 0 ) = _x_ 0 _−_ sin( _x_ 0 ) _t._


By the method of characteristics, the solution _v_ ( _x, t_ ) can be expressed in the form


_v_ ( _x_ ( _t_ ) _, t_ ) = _u_ 0 ( _x_ 0 ) _,_


for any _t ≥_ 0 which is sufficiently small such that the mapping _x_ 0 _�→_ _x_ ( _t_ ; _x_ 0 ) is
one-to-one. Since
_∂x_
= 1 _−_ cos( _x_ 0 ) _t,_
_∂x_ 0

it follows that a classical solution exists for any _t <_ 1, but the characteristics cross
at time _t_ = 1, corresponding to the formation of a stationary shock wave at _x_ 0 = 0.
The size of the jump of _v_ ( _x, t_ ) at _x_ = 0 at the particular _t_ = _π/_ 2 is given by


_v_ (0+ _, π/_ 2) _−_ _v_ (0 _−, π/_ 2) = 2 _,_


corresponding to the time _t_, at which the characteristics emanating from _x_ 0 = _±π/_ 2
reach the stationary shock at the origin. Furthermore, the function _v_ ( _x, t_ ) is smooth
on T _\{_ 0 _}_ . From basic properties of the Fourier coefficients of functions with jumpdiscontinuities, we can conclude


_**Lemma**_ **4.40** _**.**_ The solution _v_ _t_ ( _x_ ) := _v_ ( _x, t_ ) at time _t_ = _π/_ 2 of the Burgers equation
(4.54) with initial data _u_ 0 ( _x_ ) = _−_ sin( _x_ ), has Fourier coefficients with asymptotic
decay



1

_[−][i]_

_πk_ [+] _[ o]_ � _k_



�
_v_ _t_ ( _k_ ) = [1]

2 _π_



ˆ 0 2 _π_



_v_ _t_ ( _x_ ) _e_ _[−][ikx]_ _dx_ = _[−][i]_
0 _πk_



_k_



_,_ ( _k →∞_ ) _._
�



Based on this lemma, we can now estimate the spectral decay of the covariance
operator Γ _G_ # _µ_ of the push-forward measure _G_ # _µ_ :


ERROR ESTIMATES FOR DEEPONETS 59


_**Lemma**_ **4.41** _**.**_ Let _G_ ( _u_ ) := _v_ _t_, where _v_ ( _x, t_ ) = _v_ _t_ ( _x_ ) is the solution of the inviscid
Burgers equation with initial data _u_, evaluated at time _t_ = _π/_ 2. Let _µ_ be given as
the law (4.53). Then the eigenfunctions of Γ _G_ # _µ_ is given by the standard Fourier
basis **e** _k_, _k ∈_ N, with eigenvalues



1 1
_λ_ _k_ = _π_ [2] _k_ [2] [+] _[ o]_ � _k_ [2]



_,_ ( _k →∞_ ) _._
�



The proof of this lemma relies on the observation that _G_ # _µ_ is a translationinvariant measure on _L_ [2] (T), and hence, the integral kernel representing its covariance operator Γ _G_ # _µ_ is _stationary_ . In particular, this implies that the eigenfunctions
of Γ _G_ # _µ_ are given by the standard Fourier basis. Furthermore, the asymptotics of
the eigenvalues _λ_ _k_ _∝_ _k_ _[−]_ [2], as _k →∞_ can in this case be determined explicitly, based
on Lemma 4.40. The details of the argument are provided in appendix E.18.
As a consequence of Lemma 4.41, we can now state the following result:


_**Theorem**_ **4.42** _**.**_ Let _µ ∈P_ ( _L_ [2] (T)) be given by the law (4.53). Let _u �→G_ ( _u_ )
denote the operator, mapping initial data _u_ ( _x_ ) to the solution _v_ _t_ ( _x_ ) = _v_ ( _x, t_ ) at
time _t_ = _π/_ 2, where _v_ solves the inviscid Burgers equation (4.54). Then there
exists a universal constant _C >_ 0 (depending only on _µ_, but independent of the
neural network architecture), such that the DeepONet approximation error _E_ [�] for
any trunk net of size _p_ is bounded from below by


�
_E ≥_ _[C]_ _._
~~_√p_~~


_Proof._ By Theorem 3.6, the DeepONet error _E_ [�] satisfies the lower bound


�
_E ≥_ ~~�~~ _λ_ _k_ _,_
� _k>p_


where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote the ordered (repeated) eigenvalues of the covariance
operator Γ _G_ # _µ_, corresponding to a complete orthonormal eigenbasis _φ_ 1 _, φ_ 2 _, . . ._ of
Γ _G_ # _µ_ . By Lemma 4.41, the asymptotic decay of the eigenvalues can be estimated

from below by _λ_ _k_ _≥_ � _kc_ � 2, for a suitable constant _c >_ 0. It follows that



_λ_ _k_ _≥_ _c_ ~~�~~
_k>p_ � _k>p_



�
_E ≥_
~~��~~ _k>p_



_k>p_



1
_k_ [2] _[≥]_ ~~_√_~~ _[C]_ ~~_p_~~ _,_



for some _C >_ 0, as claimed. 

� � �
_**Remark**_ **4.43** _**.**_ Note that since _−_ sin( _x −_ _x_ ) = sin( _x_ ) cos( _x_ ) _−_ cos( _x_ ) sin( _x_ ), the
probability measure _µ_ of Theorem 4.42 is supported on a (compact subset of a) _two-_
_dimensional_ subspace span(cos( _x_ ) _,_ sin( _x_ )) _⊂_ _L_ [2] (T). In particular, the spectrum of
Γ _µ_ decays _at an arbitrarily fast rate_, asymptotically (since almost all eigenvalues
are zero), and the encoding error can be made to vanish, _E_ [�] _E_ = 0, with a suitable
choice of only two sensors _x_ 1 _, x_ 2 . Nevertheless, Theorem 4.42 shows that the pushforward measure _G_ # _µ_ under the inviscid Burgers equation is sufficiently complex
that the reconstruction error _E_ [�] _R_ cannot decay faster than _p_ _[−]_ [1] _[/]_ [2] in the dimension
of the reconstruction space _p_ . In particular, the fast spectral decay of _µ_ does not
imply a similarly fast spectral decay of _G_ # _µ_ under the inviscid Burgers dynamics.


We use the above result to claim that the _L_ [1] -error _E_ [�] _L_ 1 satisfies the following
lower bound:


60 ERROR ESTIMATES FOR DEEPONETS


_**Theorem**_ **4.44** _**.**_ Let _µ ∈P_ (BV _M_ ) denote the probability measure (4.53). Let _E_ [�] _L_ 1
be the DeepONet approximation error given by (4.47). Let _M >_ 0. Then there
exists a constant _C_ = _C_ ( _M_ ) _>_ 0, independent of _p_, such that



_∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ _L_ 1 (T) _dµ_ ( _u_ ) _≥_ _[C]_
BV _M_ _p_



�
_E_ _L_ 1 =
ˆ



_p_ _[,]_



for any DeepONet _N_ = _R ◦A ◦E_, such that sup _u∈_ supp( _µ_ ) _∥N_ ( _u_ ) _∥_ _L_ _∞_ _≤_ _M_ .


_**Remark**_ **4.45** _**.**_ Although we do not have a proof, it appears reasonable to conjecture that the _L_ [1] -optimal DeepONet approximations of _G_ satisfy a uniform bound of
the form sup _u∈_ supp( _µ_ ) _∥N_ ( _u_ ) _∥_ _L_ _∞_ (T) _≤_ _M_ for some _M >_ 0, independent of _p_ . If this
is indeed the case, then Theorem 4.44 is valid without the additional assumption
on sup _u∈_ supp( _µ_ ) _∥N_ ( _u_ ) _∥_ _L_ _∞_ _≤_ _M_ .
Clearly, the DeepONet constructed in Theorem 4.38 belongs to the class of DeepONets, satsifying a bound sup _u∈µ_ _∥N_ ( _u_ ) _∥_ _L_ _∞_ (T) _≤_ _M_ (in fact, with _M_ = _M_ ). Thus,
we cannot expect to improve the upper bound (4.51) to a convergence rate _α >_ 1.



_Proof._ Let _X_ denote the support of _µ_, let _R_ be any reconstruction. Following the
argument of Theorem 3.6, we have

� _λ_ _k_ _≤_ ( _E_ [�] _R_ ) [2] _≤_ _∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ [2] _L_ [2] (T) _[dµ]_ [(] _[u]_ [)] _[.]_

ˆ



� _λ_ _k_ _≤_ ( _E_ [�] _R_ ) [2] _≤_

ˆ
_k>p_



_∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ [2] _L_ [2] (T) _[dµ]_ [(] _[u]_ [)] _[.]_
_X_



By Theorem 4.42, we have [�] _k>p_ _[λ]_ _[k]_ _[ ≥]_ _[Cp]_ _[−]_ [1] [, for an absolute constant] _[ C >]_ [ 0. We]

now note the following interpolation inequality


_∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ [2] _L_ [2] (T) _[≤∥G]_ [(] _[u]_ [)] _[ −N]_ [(] _[u]_ [)] _[∥]_ _[L]_ [1] [(][T][)] _[∥G]_ [(] _[u]_ [)] _[ −N]_ [(] _[u]_ [)] _[∥]_ _[L]_ _[∞]_ [(][T][)] _[.]_


If _u ∈_ BV _M_, then _∥u∥_ _L_ _∞_ _≤_ _M_ and hence also _∥G_ ( _u_ ) _∥_ _L_ _∞_ (T) _≤_ _M_ . Furthermore, by
assumption, we have _∥N_ ( _u_ ) _∥_ _L_ _∞_ (T) _≤_ _M_ . Thus, we conclude that



_C_



_∥G_ ( _u_ ) _−N_ ( _u_ ) _∥_ _L_ 1 (T) _dµ_ ( _u_ ) _,_
_X_



_C_

_p_ _[≤]_ [(] _[M]_ [ +] _[ M]_ [)] ˆ



for any such DeepONet _N_ ( _u_ ). This clearly implies that claimed lower bound. 

5. On the generalization error for DeepONets


In section 3 and with the concrete examples in section 4, we have shown that
there exists a DeepONet _N_ of the form (2.10) which approximates an underlying
operator _G_ efficiently, i.e., the DeepONet approximation error (2.11) can be made
small without incurring the curse of dimensionality in terms of the complexity of
the DeepONet (2.10). However, in practice, one needs to _train_ the DeepONet (2.10)
by using a gradient descent algorithm to find neural network parameters (weights
and biases for the trunk and branch nets) that minimize the _loss function_



_L_ �( _N_ ) :=
ˆ _L_ [2] ( _D_ )



_|G_ ( _u_ )( _y_ ) _−N_ ( _u_ )( _y_ ) _|_ [2] _dy dµ_ ( _u_ ) _,_ (5.1)

ˆ _U_



which is related to the approximation error _E_ [�] by _E_ [�] = ~~�~~ _L_ � (cf. (2.11)).

However the loss function _L_ [�] cannot be computed exactly, and is usually approximated by sampling in both the target space _y ∈_ _U_ and the input function space
_u ∈_ _L_ [2] . As is standard in deep learning Goodfellow et al. (2016), one can follow Lu
et al. (2019) and take _N_ _u_ iid samples _U_ 1 _, . . ., U_ _N_ _u_ _∼_ _µ_ (with underlying measure
_µ_ for the DeepONet approximation problem), and for each sample _U_ _j_ to evaluate


ERROR ESTIMATES FOR DEEPONETS 61


_G_ ( _U_ _j_ ) : _U →_ R at _N_ _y_ points _Y_ _j_ [1] _[, . . ., Y]_ _j_ _[ N]_ _[y]_ with corresponding weights _w_ _j_ _[k]_ _[>]_ [ 0, lead-]
ing to the following _empirical loss_ _L_ [�] _N_ _u_ _,N_ _y_ _≈_ _L_ [�], for the DeepONet _N_ = _R ◦A ◦E_
approximation of _G_ :



_N_ _y_ 2
� _w_ _k_ _[j]_ ��� _G_ ( _U_ _j_ )( _Y_ _k j_ [)] _[ −N]_ [(] _[U]_ _[j]_ [)(] _[Y]_ _k_ _[ j]_ [)] ��� _._ (5.2)

_k_ =1



� 1
_L_ _N_ _u_ _,N_ _y_ ( _N_ ) := _N_ _u_



_N_ _u_
�

_j_ =1



If we denote ∆( _u, y_ ) := _|G_ ( _u_ )( _y_ ) _−N_ ( _u_ )( _y_ ) _|_ [2], then (5.2) can be written in the form



� 1
_L_ _N_ _u_ _,N_ _y_ ( _N_ ) = _N_ _u_



_N_ _u_
� _I_ _j_ _[N]_ _[y]_ (∆( _U_ _j_ _, ·_ )) _,_ (5.3)

_j_ =1



_I_ _j_ _[N]_ _[y]_ (∆( _U_ _j_ _, ·_ ) =



_N_ _y_
� _w_ _k_ _[j]_ [∆(] _[U]_ _[j]_ _[, Y]_ _j_ _[ k]_ [)] _[.]_ (5.4)


_k_ =1



So far, we have not specified how the _Y_ _j_ _[k]_ [,] _[ k]_ [ = 1] _[, . . ., N]_ _[y]_ [,] _[ j]_ [ = 1] _[, . . ., N]_ _[u]_ [ are to be]
chosen. As we want the innermost sum to be an approximation


_I_ _j_ _[N]_ _[y]_ (∆( _U_ _j_ _, ·_ )) _≈_ ∆( _U_ _j_ _, y_ ) _dy,_
ˆ _U_


we propose two intuitive options:

(1) Choose _Y_ _j_ _[k]_ [to be random variables, independent of all] _[ U]_ [�] _j_ [, and drawn iid]
uniform on _U_, i.e.

_Y_ _j_ _[k]_ _[∼]_ [Unif(] _[U]_ [) iid] _[,]_ _w_ _j_ _[k]_ [=] _[|]_ _N_ _[U]_ _y_ _[|]_ _,_ (5.5)


for _j_ = 1 _, . . ., N_ _u_, _k_ = 1 _, . . ., N_ _y_ .
(2) Let _y_ _k_ _∈_ _U_, _w_ _k_ _>_ 0, for _k_ = 1 _, . . ., N_ _y_, be the evaluation points and weights
of a suitable quadrature rule on _U_ . Then choose


_Y_ _j_ _[k]_ [=] _[ y]_ _[k]_ _[,]_ _w_ _j_ _[k]_ [=] _[ w]_ _[k]_ for _k_ = 1 _, . . ., N_ _y_ _._ (5.6)


5.1. **Deterministic choice of** _Y_ _k_ _[j]_ **[.]** [ If the] _[ G]_ [(] _[u]_ [) and] _[ N]_ [(] _[u]_ [) have bounded] _[ k]_ [-th deriva-]
tives in _y_, uniformly for all _u ∈_ supp( _µ_ ), then a suitable choice of the quadrature
points and weights _y_ _k_, _w_ _k_ in (5.6) can lead to a quadrature error

_I_ _N_ _y_ (∆( _U_ _j_ _, ·_ )) _−_ ∆( _U_ _j_ _, y_ ) _dy_ _≤_ _C∥_ ∆( _U_ _j_ _, ·_ ) _∥_ _C_ _k_ ( _U_ ) _N_ _y −k/n_ _._ (5.7)
���� ˆ _U_ ����


Thus, for small dimensions (e.g. _n_ = 1 _,_ 2 _,_ 3), and sufficiently regular _G_ ( _u_ ), one
might expect a deterministic quadrature rule to significantly outperform a random
sampling in _y_ . Note that if _y �→_ ∆( _U, y_ ) possesses a complex analytic extension to
a neighbourhood of _U_ for suitable _U_ and quadrature (e.g. _U_ is a torus, and the
quadrature is given by the trapezoidal rule), then the error in (5.7) in fact decays
_exponentially_ in _N_ _y_ . This might be of relevance for problems such as the forced
gravity pendulum in section 4.1 and the elliptic PDE considered in section 4.2.
We will be interested in the difference _L_ [�] _N_ _u_ _,N_ _y_ ( _N_ ) _−_ _L_ [�] ( _N_ ) in the following. We
denote



� 1
_L_ _N_ _u_ _,∞_ ( _N_ ) := _N_ _u_



_N_ _u_
�

_j_ =1



_|G_ ( _U_ _j_ )( _y_ ) _−N_ ( _U_ _j_ )( _y_ ) _|_ [2] _dy._

ˆ _U_



We can decompose the difference _L_ [�] _N_ _u_ _,N_ _y_ ( _N_ ) _−_ _L_ [�] ( _N_ ) as follows:


� � � � � �
_L_ _N_ _u_ _,N_ _y_ ( _N_ ) _−_ _L_ ( _N_ ) = � _L_ _N_ _u_ _,N_ _y_ ( _N_ ) _−_ _L_ _N_ _u_ _,∞_ ( _N_ )� + � _L_ _N_ _u_ _,∞_ ( _N_ ) _−_ _L_ ( _N_ )� _._


62 ERROR ESTIMATES FOR DEEPONETS


Then, for any parametrized network _N_ _θ_ = [�] _[p]_ _k_ =1 _[β]_ _[k]_ [(] _[u]_ [;] _[ θ]_ [)] _[τ]_ _[k]_ [(] _[y]_ [;] _[ θ]_ [) with parameter] _[ θ]_ [,]
we have (for a suitable choice of quadrature points)



� 1
_L_ _N_ _u_ _,N_ _y_ ( _N_ _θ_ ) _−_ _L_ _N_ _u_ _,∞_ ( _N_ _θ_ ) _≤_
���� ��� _N_ _u_


1
_≤_
_N_ _u_



_N_ _u_
�

_j_ =1



_N_ _u_
� _C∥_ ∆ _θ_ ( _U_ _j_ _, ·_ ) _∥_ _C_ _k_ ( _U_ ) _N_ _y_ _[−][k/n]_

_j_ =1



_I_ _N_ _y_ (∆ _θ_ ( _U_ _j_ _, ·_ )) _−_ ∆ _θ_ ( _U_ _j_ _, y_ ) _dy_
���� ˆ _U_ ����



�



_≤_ _C_



_∥G_ ( _u_ ) _∥_ [2] _C_ _[k]_ ( _U_ ) [+]

�



_p_
� _|β_ _k_ ( _u_ ; _θ_ ) _|_ [2] _∥τ_ _k_ ( _·_ ; _θ_ ) _∥_ [2] _C_ _[k]_ ( _U_ )


_k_ =1



_N_ _y_ _[−][k/n]_ _._



Thus, if




sup _∥τ_ _k_ ( _·_ ; _θ_ ) _∥_ _C_ _k_ ( _U_ ) _≤_ _C_ 0 _,_
_θ_







sup _|β_ _k_ ( _·_ ; _θ_ ) _|_ [2] _≤_ Ψ( _u_ ) _,_
_θ_

_∥G_ ( _u_ ) _∥_ [2] _C_ _[k]_ ( _U_ ) _[≤]_ [Ψ(] _[u]_ [)] _[,]_



for some integrable Ψ( _u_ ) _≥_ 0, such that E _µ_ [Ψ( _u_ )] _< ∞_, then we can estimate



E sup
� _θ_



�
���� _L_ _N_ _u_ _,N_ _y_ ( _N_ _θ_ ) _−_ _L_ _N_ _u_ _,∞_ ( _N_ _θ_ )���� _≤_ _CN_ _y_ _[−][k/n]_ _,_



for some _C >_ 0 depending on _k_, the quadrature points, Ψ, the upper bound _C_ 0 and
_µ_, but independent of _N_ _y_ .


5.2. **Random choice of** _U_ _j_ **,** _Y_ _k_ _[j]_ **[.]** [ However, in addition to the sampling in the]
target space _Y_ _k_ _[j]_ _[∈]_ _[U]_ [, one also needs to sample from the underlying input function]
space _U_ _j_ _∈_ _X ⊂_ _L_ [2] ( _D_ ). Random sampling is the only viable option for this infinite
dimensional space. For the sake of simplicity of notation and exposition, we will
choose _N ∈_ N, and _N_ _u_ = _N_, _N_ _y_ = 1 in (5.5): i.e., choose mutually independent
random variables _U_ _j_, _Y_ _j_, _j_ = 1 _, . . ., N_, such that


_U_ 1 _, . . ., U_ _N_ _∼_ _µ_ are iid _,_ _Y_ 1 _, . . ., Y_ _N_ _∼_ Unif( _U_ ) are iid _._ (5.8)


Let _L_ [�] _N_ := _L_ [�] _N_ _u_ _,_ 1, i.e.



�
_L_ _N_ ( _N_ ) := _[|][U]_ _[|]_

_N_



_N_
� _|G_ ( _U_ _j_ )( _Y_ _j_ ) _−N_ ( _U_ _j_ )( _Y_ _j_ ) _|_ [2] _,_ (5.9)

_j_ =1



be the corresponding empirical loss. We note that the random variables ( _U_ _j_ _, Y_ _j_ ),
_j_ = 1 _, . . ., N_ are iid random variables with joint distribution ( _U_ _j_ _, Y_ _j_ ) _∼_ _µ⊗_ Unif( _U_ ).
Fix a DeepONet neural network architecture of the form (2.10) with parameters
(weights and biases in the corresponding trunk and branch nets) _θ �→N_ _θ_ . We
assume that the weights and biases are bounded, _θ ∈_ [ _−B, B_ ] _[d]_ _[θ]_ for _B >_ 0, and
some large _d_ _θ_ _∈_ N, representing the number of tuning parameters in the DeepONet
_N_ _θ_ . Let _N_ [�] _N_ = _N_ _θ_ � denote an optimizer of the empirical loss _L_ [�] _N_ among all choices
of the weights and biases _θ_, let _N_ [�] be an optimizer of the loss function _L_ [�] := ( _E_ [�] ) [2]

(5.1). The quantity


� �
( _E_ [�] gen ) [2] = _L_ [�] _N_ _N_ _−_ _L_ [�] _N_ _,_ (5.10)
� � � �

with _L_ [�] _N_ � _N_ denoting the empirical loss (5.2), is referred to as the _generalization_
� �

_error_ . It provides a measure of how far the empirical optimizer _N_ [�] _N_ of _L_ [�] _N_ is from
being an optimizer of _L_ [�] . The generalization error has been studied in detail for
conventional neural networks defined on _finite-dimensional_ spaces. In the present


ERROR ESTIMATES FOR DEEPONETS 63


section, we consider the extension of these results to the setting of DeepONets,
which are defined on _infinite-dimensional_ spaces.
Let us make the following assumptions:


_**Assumption**_ **5.1** (Boundedness) _**.**_ We assume that there exists a function Ψ :
_L_ [2] ( _D_ ) _→_ [0 _, ∞_ ), ( _u_ ) _�→_ Ψ( _u_ ), such that


_|G_ ( _u_ )( _y_ ) _| ≤_ Ψ( _u_ ) _,_ sup _|N_ _θ_ ( _u_ )( _y_ ) _| ≤_ Ψ( _u_ ) _,_
_θ∈_ [ _−B,B_ ] _[dθ]_


for all _u ∈_ _L_ [2] ( _D_ ), _y ∈_ _U_, and there exist constants _C, κ >_ 0, such that


Ψ( _u_ ) _≤_ _C_ (1 + _∥u∥_ _L_ 2 ) _[κ]_ _._ (5.11)


_**Assumption**_ **5.2** (Lipschitz continuity) _**.**_ There exists a function Φ : _L_ [2] ( _D_ ) _→_

[0 _, ∞_ ), _u �→_ Φ( _u_ ), such that


_|N_ _θ_ ( _u_ )( _y_ ) _−N_ _θ_ _′_ ( _u_ )( _y_ ) _| ≤_ Φ( _u_ ) _∥θ −_ _θ_ _[′]_ _∥_ _ℓ_ _∞_ _,_


for all _u ∈_ _L_ [2] ( _D_ ), _y ∈_ _U_, and


Φ( _u_ ) _≤_ _C_ (1 + _|u∥_ _L_ 2 ) _[κ]_ _,_


for the same constants _C, κ >_ 0 as in (5.11).


Note that both the boundedness and Lipschitz continuity assumptions are satisfied for the concrete examples of operators _G_ considered in section 4.
To simplify the notation for the following estimates, we denote by _Z_ _j_ = ( _U_ _j_ _, Y_ _j_ )
a (joint) random variable on _L_ [2] ( _D_ ) _× U_, and by a slight abuse of notation we write
_G_ ( _Z_ _j_ ) = _G_ ( _U_ _j_ )( _Y_ _j_ ), _N_ _θ_ ( _Z_ _j_ ) = _N_ _θ_ ( _U_ _j_ )( _Y_ _j_ ). Recall that with the random choice
(5.8), the _Z_ _j_ are iid random variables. We denote



_S_ _θ_ _[N]_ [=] _N_ [1]



_N_
� _|G_ ( _Z_ _j_ ) _−N_ _θ_ ( _Z_ _j_ ) _|_ [2] _._ (5.12)

_j_ =1



We have the following bound on the generalization error (5.10),


_**Theorem**_ **5.3** _**.**_ Let _N_ [�] and _N_ [�] _N_ denote the minimizer of the loss (5.1) and empirical
loss (5.9), respectively. If the assumptions 5.1 and 5.2 hold, then the generalization
error (5.10) is bounded by



_C_
E _L_ ( _N_ � _N_ ) _−_ _L_ [�] ( _N_ [�] ) _≤_
����� ���� ~~_√_~~ _N_



1 + _Cd_ _θ_ log( _CB√_
�



_N_ ) [2] _[κ]_ [+1] _[/]_ [2] [�] _,_ (5.13)



where _C_ = _C_ ( _µ,_ Ψ _,_ Φ) is a constant independent of _B_, _d_ _θ_ and _N_ and _κ_ is specified
in (5.11).


The proof of Theorem 5.3 relies on a series of technical lemmas and is detailed
in Appendix F.


_**Remark**_ **5.4** _**.**_ The generalization error bound (5.13) shows that even if the underlying approximation problem is in infinite dimensions, the DeepONet generalization
error (5.10), at worst, scales (up to a log) with the standard Monte Carlo scaling
of 1 _/√N_ in the number of samples _N_ from the infinite dimensional input space.

Thus, one can reduce the generalization error by increasing the number of samples,
even in this infinite dimensional setting. In particular, the curse of dimensionality
is also overcome for the generalization error.


64 ERROR ESTIMATES FOR DEEPONETS


_**Remark**_ **5.5** _**.**_ A careful observation of the bound (5.13) on the generalization error
(5.10) reveals that the bound depends explicitly on the number of parameters _d_ _θ_ of
the DeepONet (2.10). As we have seen in the previous sections, one might need a
large number of parameters in order to reduce the approximation error (2.11) to the
desired tolerance. Thus, this estimate, like all estimates based on covering number
and other statistical learning theory techniques Cucker & Smale (2002), applies in
the underparametrized regime i.e _N ≫_ _d_ _θ_ .


_**Remark**_ **5.6** _**.**_ The bound (5.13) can also blow up if the bound on weights _B →∞_ .
However, we note that this blowup is modulated by a log and the weights can indeed
be unbounded asymptotically i.e _B ∼_ _e_ _[N]_ _[ r]_ for some _r <_ 1 _/_ 2 will still result in a
decay of the error with increasing _N_ . Given that such an exponential blowup may
not occur in practice, it is reasonable to assume that the explicit dependence on
the weight bounds may not affect the decay rate of the generalization error. This
also holds if the bound on weights _B_, blows up as _B ∼_ _d_ _[r]_ _θ_ _[θ]_ [, for some] _[ r]_ _[θ]_ _[ >]_ [ 0. Given]
the log-term in (5.13), this blow up of weights will only translate into a _d_ _θ_ log( _d_ _θ_ )dependence of the generalization error. As long as one is in the under-parametrized
regime, this dependence does not affect the overall decay of the generalization error
as _N →∞_ .


6. Discussion


Operators, mapping infinite-dimensional Banach spaces, arise naturally in the
study of differential equations. _Learning_ such operators from data using neural
networks can be very challenging on account of the underlying infinite-dimensional
setup. In this paper, we analyze a neural network architecture termed _DeepONets_
for approximating such operators. DeepONets are a recent extension Lu et al.
(2019) of operator networks, first considered in Chen & Chen (1995) and have
been recently successfully applied in many different contexts Lu et al. (2019), Mao
et al. (2021), Cai et al. (2021), Lin et al. (2021) and references therein. However,
apart from the universal approximation result of Chen & Chen (1995) and its
extension to DeepONets in Lu et al. (2019), very few rigorous results for DeepONets
are available currently. In particular, given the underlying infinite-dimensional
setup, it is essential to demonstrate that DeepONets can overcome the _curse of_
_dimensionality_, associated with the _∞_ -dimensional input-to-output mapping (see
Definition 3.5).
Our main aim in this article has been to analyze a form of DeepONets (2.10)
and prove rigorous bounds on the error (2.11), incurred by a DeepONet in approximating a nonlinear operator _G_, with an underlying measure _µ_ (see Definition 2.1).
To this end, we have presented the following results in the paper:


_•_
We extend the universal approximation theorem of Chen & Chen (1995) to
Theorem 3.1, where we show that given any _measurable_ operator _G_ : _X →_
_Y_, for _X_ = _C_ ( _D_ ), _Y_ = _L_ [2] ( _U_ ), with respect to an underlying measure _µ ∈_
_P_ ( _X_ ), there exists a DeepONet of the form (2.10), which can approximate
it to arbitrary accuracy. In particular, we remove the continuity (of _G_ ) and
compactness (of subsets of _X_ ) assumptions of Chen & Chen (1995) and
pave the way for the application of DeepONets to approximate operators
that arise in applications of PDEs to fields such as hypersonics Mao et al.
(2021).

_•_
We provide an upper bound (Theorem 3.3) on the DeepONet error (2.11)
by decomposing it into three parts, i.e., an encoding error (3.1) stemming
from the encoder _E_, an approximation error that arises from the approximator neural network _A_ that maps between finite-dimensional spaces and


ERROR ESTIMATES FOR DEEPONETS 65


a reconstruction error (3.3), corresponding to the trunk net induced affine
reconstructor _R_ (2.9).

_•_ In Theorem 3.14, we prove _lower bounds_ on the reconstruction error (3.3)
by utilizing _optimal errors_ for projections on finite dimensional affine subspaces of separable Hilbert spaces (Theorem 3.12). This allows us in Theorems 3.3 and 3.6 to prove _two-sided bounds_ on the DeepONet error (2.11).
In particular, the lower bound is explicitly given in terms of the decay
of the eigenvalues of the covariance operator (3.18), associated with the
push-forward measure _G_ # _µ_ (3.9). Moreover, this construction also allows
us to infer the number of trunk nets _p_ and that these trunk nets should
approximate the eigenfunctions of the covariance operator in-order to obtain optimal reconstruction errors. Furthermore, we also provide bounds
(3.11) on the reconstruction error that leverage the Sobolev regularity of
the image of the nonlinear operator _G_ .

_•_ To control the encoding error (3.1) corresponding to the encoder _E_, which
is a pointwise evaluation of the input at _m sensor locations_, we construct a
_decoder D_ (approximate inverse of the encoder) (3.38). We show in Theorem 3.9 that sensors chosen _at random_ on the underlying domain _D_ suffice
to provide an _almost optimal_ (optimal modulo a log) bound on the encoding
error. This further highlights the fact that DeepONets allow for a general
approximation framework, i.e., no explicit information is needed about the
location of sensor points and they can be chosen randomly.

_•_
Finally, estimating the approximation error (3.2) reduces to deriving bounds
on a neural network _A_ that maps one finite (but possibly very-high) dimensional space to another. Hence, standard neural network approximation
results such as from Yarotsky (2017) can be applied. In particular, approximation results for holomorphic maps, such as those derived in Schwab &
Zech (2019), Opschoor et al. (2019, 2020) are important in this context.


The above results provide a workflow for deriving bounds on DeepONet approximation for general nonlinear operators _G_ with underlying measures _µ_, which is
outlined at the beginning of Section 4. We illustrate this program with very general bounded linear operators and with the following four concrete examples of
nonlinear operators, each corresponding to a differential equation that serves as a
model for a large class of related problems:


_•_ First, we consider the forced gravity pendulum (4.1), with the operator _G_
mapping the forcing term into the solution of the ODE and a parametrized
random field defining the underlying measure. We bound the reconstruction error by smoothness of the image of _G_ and the encoding error decays
exponentially in the number of sensors on account of the decay of eigenvalues of the covariance operator associated with the underlying measure. The
approximation error is bounded by showing that the operator allows for a
complex analytic extension. Combining these ingredients in Theorem 4.10,
we prove an error bound (4.16) on the DeepONet approximation error. In
particular, it is shown in (4.17) that the size of the DeepONet (total number of parameters in the trunk and branch nets) only grow sub-algebraically
with respect to the error tolerance.

_•_
As a second example, we consider the standard elliptic PDE (4.18) with a
variable coefficient _a_, which, for instance, arises in the modeling of groundwater Darcy flow. The nonlinear operator _G_ in this case maps the coefficient
to the solution of the elliptic PDE and the underlying measure is the law of
the random field (4.22). Again, we utilize the spectral properties of the underlying covariance operator, smoothness of the image of _G_ and holomorphy


66 ERROR ESTIMATES FOR DEEPONETS


of an associated map to prove the bound (4.29) on the DeepONet approximation error. As in the case of the forced gravity pendulum, we show that
the size of the DeepONet only grows _sub-algebraically_ with decreasing error
tolerance.

_•_
As a third example, we consider the Allen-Cahn equation (4.31) that models
phase transitions, as a model for nonlinear parabolic PDEs of the reactiondiffusion type. The operator _G_ maps the initial data into the solution (at
a given time) for the Allen-Cahn equation. In this case, no holomorphic
extension of the underlying mapping is available. Nevertheless, we use
a novel strategy to _emulate_ a convergent finite difference scheme (4.39)
by neural networks and derive an upper bound (4.41) on the DeepONet
approximation error. In particular, the size of the DeepONet only grows
polynomially (4.42) with respect to decreasing error tolerance.

_•_
In the final example, we consider a scalar conservation law (4.43) as a
prototype for nonlinear hyperbolic PDEs. In this case, the operator _G_ is
defined as the mapping between the initial data and the entropy solution
of the conservation law at a given time. This example differs from the
other three in two crucial respects. First, the underlying solutions are
discontinuous and thus a pointwise evaluation based encoder cannot be
used and is replaced by local averages (4.49). Second, the operator _G_ is
contractive (hence Lipschitz continuous) in _L_ [1] . Thus, the usual Hilbert
space setup, will a priori, will lead to sub-optimal error bounds. Hence,
we adapt our theory to an _L_ [1] -Banach space version and are able to prove
the upper bound (4.51) on the resulting DeepONet approximation error
(4.47). This bound also allows us to conclude in (4.52) that the size of
the DeepONet only grows polynomially with respect to the error tolerance.
Moreover, we also construct an explicit example to show a lower bound in
Theorem 4.44 on the DeepONet error in this case. This shows that the
derived upper bounds are almost optimal for scalar conservation laws.


Hence, in all the four concrete examples which cover a large spectrum of nonlinear
operators arising in the study of differential equations, _we prove that there exist_
_DeepONets, which break the curse of dimensionality_ in approximating the underlying operators. These examples and the underlying abstract theory provide a
comprehensive study of the approximation error (2.11) for DeepONets.
Finally, we also study the generalization error (5.10) that arises from replacing the loss function (population risk) (5.1) with its sampled version, the so-called
empirical loss (empirical risk) (5.2) that is used during training. Under very general assumptions on the underlying operator _G_ and the approximating branch and
trunk nets, we apply covering number estimates to prove the bound (5.13) on the
generalization error. In spite of the overall infinite-dimensional setup, this bound
shows that the generalization errors decays (up to a log) with the reciprocal of the
square-root of the total number of samples in the input function space, thus also
overcoming the curse of dimensionality in this respect.
Thus, the analysis and results of this paper clearly prove that DeepONets can
efficiently approximate operators in very general settings that include many examples of PDEs. The analysis also reveals some reasons for why DeepONets can work
so well in practical applications and as building blocks for complex multi-physics
systems, such as in the DeepM&Mnet architectures introduced in Mao et al. (2021),
Cai et al. (2021). The main reason is the generality and flexibility of DeepONets.
In particular _no a priori information about the underlying measure and operator_
_are necessary at an algorithmic level_, apart from being able to sample from the


ERROR ESTIMATES FOR DEEPONETS 67


underlying measure. Given our analysis, one can even use a small number of _ran-_
_domly distributed sensors_ to achieve almost optimal encoding error. Similarly, a
small number of trunk-nets, with a very general neural network architecture, will
be able to learn the eigenfunctions of the underlying covariance operator such that
an optimal reconstruction error is attained with the resulting affine reconstructor (2.9). Finally, the branch nets can be trained simultaneously to minimize the
approximation error.
At this point, we contrast DeepONets with other recently proposed frameworks
for operator learning. In particular, we focus on a recent paper Bhattacharya et al.
(2021), where the authors present an operator learning framework based on a principal component analysis (PCA) autoencoder for both the encoding and reconstruction steps. Thus, in that approach, one has to explicitly construct an approximate
eigenbasis of the empirical covariance operator for the input measure and its pushforward with respect to the underlying operator. Neural networks are only used to
approximate the operator on PCA projected finite-dimensional spaces. In contrast,
DeepONets do not require any explicit knowledge of the covariance operator. In
fact, our analysis shows that DeepONets _implicitly_ and _concurrently_ learn a suitable basis in output space along with an approximation of the projected operator.
Although, many elements of our analysis overlap with that of Bhattacharya et al.
(2021), we provide significantly more general results, including the alleviation of
the curse of dimensionality for DeepONets. Moreover, our analysis can be readily
extended to the framework of Bhattacharya et al. (2021) to prove the mitigation of
the curse of dimensionality in that context.
It is also instructive to compare our error bounds with the numerical results of Lu
et al. (2019), Mao et al. (2021), Cai et al. (2021), Lin et al. (2021). In particular for
the forced gravity pendulum, the authors of Lu et al. (2019) considered a Gaussian
random field with covariance kernel, similar to (3.45), as the underlying measure
and observed an exponential decay of the test error with respect to the number
of sensors (see Figure 2 (B) for a simpler example). Indeed, this observation is
consistent with both the exponential decay of the encoding error (3.48) and the
spectral decay of the overall error (4.16), as long as the correlation scale is resolved,
i.e. _m ∼_ 1 _/ℓ_, which is also observed in the numerical experiments of Lu et al. (2019).
On the other hand, the decay of the generalization error with respect to the number
of training samples, both in examples considered in Lu et al. (2019) as well as in
figure 2 (C,D) shows a very interesting behavior. For a small number of samples,
the training error decays exponentially enabling fast training for DeepONets. Only
for a relatively large number of training samples, the generalization error decays
algebraically with respect to the number of training samples, at a rate consistent
with the error bound (5.13). This bi-phasic behavior of the generalization error is
certainly not explained by the bound (5.13) and will be a topic of future work.
The methods and results of this paper can be extended in different directions.
We can apply the abstract framework presented in section 3 to other examples of
differential equations, for instance the Navier-Stokes equations of fluid dynamics.
Although we showed that the curse of dimensionality is broken by DeepONets for
all the examples that we consider, it is unclear if our bounds on computational
complexity of DeepONets are sharp. We show almost sharpness for scalar conservation laws and given the sub-algebraic decay of DeepONet size, we believe that
the results for the pendulum and elliptic PDE are also close to optimal. However,
there is certainly room for a sharp estimate for the Allen-Cahn equation. Finally,
one can readily extend DeepONets, for instance by endowing them with a recurrent
structure, to approximate the whole time-series for a time-parametrized operator,


68 ERROR ESTIMATES FOR DEEPONETS


such as the solution operators of time-dependent PDEs. Extending the rigorous results of this paper to cover this recurrent case will also be considered in the future.
Another possible avenue of future work is the extension of the approximation results
in this paper to the case of multiple nonlinear operators (MNOs), already considered in Back & Chen (2002), where the authors prove an universal approximation
property, similar to Chen & Chen (1995) for these operators.


Acknowledgements


The research of Samuel Lanthaler and Siddhartha Mishra is partially supported
by the European Research Council Consolidator grant ERC-CoG 770880 COMANFLO. George Karniadakis acknowledges partial support from MURI-AFOSR FA955020-1-0358: ”Learning and Meta-Learning of Partial Differential Equations via PhysicsInformed Neural Networks: Theory, Algorithms, and Applications”


References


Adler, J. & Oktem, O. (2017), ‘Solving ill-posed inverse problems using iterative [¨]
deep neural networks’, _Inverse Problems_ **33** (12), 124007.
**URL:** _https://doi.org/10.1088/1361-6420/aa9581_
Ahmed, S. E., Pawar, S., San, O., Rasheed, A., Iliescu, T. & Noack, B. R. (2021),
‘On closures for reduced order models—a spectrum of first-principle to machinelearned avenues’, _Physics of Fluids_ **33** (9), 091301.
**URL:** _https://doi.org/10.1063/5.0061577_
Back, A. D. & Chen, T. (2002), ‘Universal approximation for multiple nonlinear
operators by neural networks’, _Neural Computation_ **14**, 2561–2566.
Barron, A. R. (1993), ‘Universal approximation bounds for superpositions of a
sigmoidal function’, _IEEE Trans. Inform. Theory._ **39** (3), 930–945.
Beck, C., Becker, S., Grohs, P., Jaafari, N. & Jentzen, A. (2021), ‘Solving the
Kolmogorov PDE by means of deep learning’, _Journal of Scientific Computing_
**88** (3).
**URL:** _http://dx.doi.org/10.1007/s10915-021-01590-0_
Berner, J., Grohs, P. & Jentzen, A. (2020), ‘Analysis of the generalization error:
Empirical risk minimization over deep artificial neural networks overcomes the
curse of dimensionality in the numerical approximation of black–scholes partial
differential equations’, _SIAM Journal on Mathematics of Data Science_ **2** (3), 631–
657.
Bhattacharya, K., Hosseini, B., Kovachki, N. B. & Stuart, A. M. (2021), ‘Model
reduction and neural networks for parametric PDEs’, _The SMAI journal of com-_
_putational mathematics_ **7**, 121–157.
**URL:** _https://smai-jcm.centre-mersenne.org/articles/10.5802/smai-jcm.74/_
Bogachev, V. I. (2007), _Measure theory_, Springer.
Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016), ‘Discovering governing equations
from data by sparse identification of nonlinear dynamical systems’, _Proceedings_
_of the National Academy of Sciences_ **113** (15), 3932–3937.
**URL:** _https://www.pnas.org/content/113/15/3932_
Cai, S., Wang, Z., Lu, L., Zaki, T. A. & Karniadakis, G. E. (2021), ‘DeepM&Mnet:
Inferring the electroconvection multiphysics fields based on operator approximation by neural networks’, _Journal of Computational Physics_ **436**, 110296.
Canuto, C. & Quarteroni, A. (1982), ‘Approximation results for orthogonal polynomials in Sobolev spaces’, _Mathematics of Computation_ **38** (157), 67–86.
**URL:** _http://www.jstor.org/stable/2007465_
Charrier, J. (2012), ‘Strong and weak error estimates for elliptic partial differential equations with random coefficients’, _SIAM Journal on Numerical Analysis_


ERROR ESTIMATES FOR DEEPONETS 69


**50** (1), 216–246.
**URL:** _https://doi.org/10.1137/100800531_
Chen, T. & Chen, H. (1995), ‘Universal approximation to nonlinear operators by
neural networks with arbitrary activation functions and its application to dynamical systems’, _IEEE Transactions on Neural Networks_ **6** (4), 911–917.
Chkifa, A., Cohen, A. & Schwab, C. (2015), ‘Breaking the curse of dimensionality in sparse polynomial approximation of parametric PDEs’, _Journal de_
_Math´ematiques Pures et Appliqu´ees_ **103** (2), 400–428.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0021782414000580_
Cohen, A., DeVore, R. & Schwab, C. (2010), ‘Convergence rates of best N-term
Galerkin approximations for a class of elliptic sPDEs’, _Foundations of Computa-_
_tional Mathematics_ **10** (6), 615–646.
Cohen, A., Devore, R. & Schwab, C. (2011), ‘Analytic regularity and polynomial
approximation of parametric and stochastic elliptic PDE’s’, _Analysis and Appli-_
_cations_ **9** (01), 11–47.
Cucker, F. & Smale, S. (2002), ‘On the mathematical foundations of learning’,
_Bulletin of the American Mathematical Society_ **39** (1), 1–49.
Cybenko, G. (1989), ‘Approximations by superpositions of sigmoidal functions’,
_Approximation theory and its applications_ **9** (3), 17–28.
DeRyck, T. & Mishra, S. (2021), Error analysis for deep neural network approximations of parametric hyperbolic conservation laws. Preprint, available from
arXiv:.
Duraisamy, K., Iaccarino, G. & Xiao, H. (2019), ‘Turbulence modeling in the age
of data’, _Annual Review of Fluid Mechanics_ **51** (1), 357–377.
**URL:** _https://doi.org/10.1146/annurev-fluid-010518-040547_
E, W., Han, J. & Jentzen, A. (2017), ‘Deep learning-based numerical methods for
high-dimensional parabolic partial differential equations and backward stochastic
differential equations’, _Communications in Mathematics and Statistics_ **5** (4), 349–
380.
Elbr¨achter, D., Perekrestenko, D., Grohs, P. & B¨olcskei, H. (2021), ‘Deep neural network approximation theory’, _IEEE Transactions on Information Theory_
**67** (5), 2581–2623.
Evans, R., Jumper, J., Kirkpatrick, J., Sifre, L., Green, T., Qin, C., Zidek, A., Nelson, A., Bridgland, A., Penedones, H. et al. (2018), ‘De novo structure prediction
with deep-learning based scoring’, _Annual Review of Biochemistry_ **77** (363-382), 6.
Godlewski, E. & Raviart, P. A. (1991), _Hyperbolic systems of conservation laws_,
Ellipsis.
Goodfellow, I., Bengio, Y. & Courville, A. (2016), _Deep learning_, MIT press.
Guo, X., Li, W. & Iorio, F. (2016), Convolutional neural networks for steady flow
approximation, _in_ ‘Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining’, KDD ’16, Association for
Computing Machinery, New York, NY, USA, p. 481–490.
**URL:** _https://doi.org/10.1145/2939672.2939738_
Han, J., Jentzen, A. & E, W. (2018), ‘Solving high-dimensional partial differential
equations using deep learning’, _Proceedings of the National Academy of Sciences_
**115** (34), 8505–8510.
Hornik, K., Stinchcombe, M. & White, H. (1989), ‘Multilayer feedforward networks
are universal approximators’, _Neural networks_ **2** (5), 359–366.
Jagtap, A. D., Kawaguchi, K. & Karniadakis, G. E. (2020), ‘Adaptive activation
functions accelerate convergence in deep and physics-informed neural networks’,
_Journal of Computational Physics_ **404**, 109136.
**URL:** _http://www.sciencedirect.com/science/article/pii/S0021999119308411_


70 ERROR ESTIMATES FOR DEEPONETS


Khoo, Y. & Ying, L. (2019), ‘Switchnet: A neural network model for forward and inverse scattering problems’, _SIAM Journal on Scientific Computing_ **41** (5), A3182–
A3201.
**URL:** _https://doi.org/10.1137/18M1222399_
Kutyniok, G., Petersen, P., Raslan, M. & Schneider, R. (2021), ‘A theoretical analysis of deep neural networks and parametric PDEs’, _Constructive Approximation_
pp. 1–53.
Laakmann, F. & Petersen, P. (2021), ‘Efficient approximation of solutions of parametric linear transport equations by relu dnns’, _Advances in Computational_
_Mathematics_ **47** (1), 1–32.
LeCun, Y., Bengio, Y. & Hinton, G. (2015), ‘Deep learning’, _Nature_
**521** (7553), 436–444.
Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart,
A. & Anandkumar, A. (2021), Fourier neural operator for parametric partial
differential equations, _in_ ‘International Conference on Learning Representations’.
**URL:** _https://openreview.net/forum?id=c8P9NQVtmnO_
Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart,
A. M. & Anandkumar, A. (2020), ‘Neural operator: Graph kernel network for
partial differential equations’, _CoRR_ **abs/2003.03485** .
Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Stuart, A. M., Bhattacharya,
K. & Anandkumar, A. (2020), Multipole graph neural operator for parametric
partial differential equations, _in_ H. Larochelle, M. Ranzato, R. Hadsell, M. F.
Balcan & H. Lin, eds, ‘Advances in Neural Information Processing Systems
(NeurIPS)’, Vol. 33, Curran Associates, Inc., pp. 6755–6766.
Lieberman, G. M. (1996), _Second order parabolic differential equations_, World scientific.
Lin, C., Li, Z., Lu, L., Cai, S., Maxey, M. & Karniadakis, G. E. (2021), ‘Operator learning for predicting multiscale bubble growth dynamics’, _The Journal of_
_Chemical Physics_ **154** (10), 104118.
Lu, L., Jin, P. & Karniadakis, G. E. (2019), ‘DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation
theorem of operators’, _arXiv preprint arXiv:1910.03193_ .
Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F. & Johnson, S. G. (2021),
‘Physics-informed neural networks with hard constraints for inverse design’, _arXiv_
_e-prints_ pp. arXiv–2102.
Lye, K. O., Mishra, S. & Ray, D. (2020), ‘Deep learning observables in computational fluid dynamics’, _Journal of Computational Physics_ **410**, 109339.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0021999120301133_
Lye, K. O., Mishra, S., Ray, D. & Chandrashekar, P. (2021), ‘Iterative surrogate
model optimization (ISMO): An active learning algorithm for PDE constrained
optimization with deep neural networks’, _Computer Methods in Applied Mechan-_
_ics and Engineering_ **374**, 113575.
**URL:** _https://www.sciencedirect.com/science/article/pii/S004578252030760X_
Mao, Z., Jagtap, A. D. & Karniadakis, G. E. (2020), ‘Physics-informed neural
networks for high-speed flows’, _Computer Methods in Applied Mechanics and_
_Engineering_ **360**, 112789.
Mao, Z., Lu, L., Marxen, O., Zaki, T. A. & Karniadakis, G. E. (2021), ‘DeepMandMnet for hypersonics: Predicting the coupled flow and finite-rate chemistry
behind a normal shock using neural-network approximation of operators’, _Jour-_
_nal of Computational Physics_ **447**, 110698.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0021999121005933_


ERROR ESTIMATES FOR DEEPONETS 71


Mhaskar, H. N. & Hahm, N. (1997), ‘Neural networks for functional approximation
and system identification’, _Neural Computation_ **9** (1), 143–159.
Mishra, S. & Molinaro, R. (2020), ‘Estimates on the generalization error of
physics informed neural networks (pinns) for approximating pdes’, _arXiv preprint_
_arXiv:2006.16144_ .
Mishra, S. & Molinaro, R. (2021 _a_ ), ‘Estimates on the generalization error of physicsinformed neural networks for approximating a class of inverse problems for PDEs’,
_IMA Journal of Numerical Analysis_ . drab032.
**URL:** _https://doi.org/10.1093/imanum/drab032_
Mishra, S. & Molinaro, R. (2021 _b_ ), ‘Physics informed neural networks for simulating
radiative transfer’, _Journal of Quantitative Spectroscopy and Radiative Transfer_
**270**, 107705.
Opschoor, J. A. A., Schwab, C. & Zech, J. (2019), Exponential ReLU DNN expression of holomorphic maps in high dimension, Technical Report 2019-35, Seminar
for Applied Mathematics, ETH Z¨urich, Switzerland.
Opschoor, J. A. A., Schwab, C. & Zech, J. (2020), Deep learning in high dimension:
ReLU network expression rates for bayesian PDE inversion, Technical Report
2020-47, Seminar for Applied Mathematics, ETH Z¨urich, Switzerland.
O’Leary-Roseberry, T., Villa, U., Chen, P. & Ghattas, O. (2022), ‘Derivativeinformed projected neural networks for high-dimensional parametric maps governed by PDEs’, _Computer Methods in Applied Mechanics and Engineering_
**388**, 114199.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0045782521005302_
Patel, R. G., Trask, N. A., Wood, M. A. & Cyr, E. C. (2021), ‘A physics-informed
operator regression framework for extracting data-driven continuum models’,
_Computer Methods in Applied Mechanics and Engineering_ **373**, 113500.
**URL:** _https://www.sciencedirect.com/science/article/pii/S004578252030685X_
Pinelis, I. & Molzon, R. (2016), ‘Optimal-order bounds on the rate of convergence
to normality in the multivariate delta method’, _Electron. J. Statist._ **10** (1), 1001–
1063.
**URL:** _https://doi.org/10.1214/16-EJS1133_
Pinkus, A. (1999), ‘Approximation theory of the MLP model in neural networks’,
_Acta numerica_ **8** (1), 143–195.
Raissi, M. & Karniadakis, G. E. (2018), ‘Hidden physics models: Machine learning
of nonlinear partial differential equations’, _Journal of Computational Physics_
**357**, 125–141.
Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019), ‘Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations’, _Journal of Computational_
_Physics_ **378**, 686–707.
Raissi, M., Yazdani, A. & Karniadakis, G. E. (2018), ‘Hidden fluid mechanics: A
Navier-Stokes informed deep learning framework for assimilating flow visualization data’, _arXiv preprint arXiv:1808.04327_ .
Schwab, C. & Zech, J. (2019), ‘Deep learning in high dimension: Neural network
expression rates for generalized polynomial chaos expansions in uq’, _Analysis and_
_Applications_ **17** (01), 19–55.
Stuart, A. M. (2010), ‘Inverse problems: a Bayesian perspective’, _Acta numerica_
**19**, 451–559.
Tang, T. & Yang, J. (2016), ‘Implicit-explicit scheme for the allen-cahn equation preserves the maximum principle’, _Journal of Computational Mathematics_
**34** (5), 451–461.


72 ERROR ESTIMATES FOR DEEPONETS


Tianping Chen, H. C. & Liu, R.-W. (1990), A constructive proof of Cybenko’s
approximation theorem and its extensions, _in_ ‘Computing Science and Statistics (edited by LePage and Page) Proc. of the 22nd Symposium on Interface’,
Springer-Verlag, pp. 163–168.
Welti, T. (2020), _High-dimensional stochastic approximation: algorithms and con-_
_vergence rates_, ETH Dissertation N. 26805.
Yang, J., Du, Q. & Zhang, W. (2018), ‘Uniform l p-bound of the allen-cahn equation
and its numerical discretization’, _International Journal of Numerical Analysis &_
_Modeling_ **15** .
Yarotsky, D. (2017), ‘Error bounds for approximations with deep ReLU networks’,
_Neural Networks_ **94**, 103–114.
Yarotsky, D. (2018), Optimal approximation of continuous functions by very deep
relu networks, _in_ ‘Conference on Learning Theory’, PMLR, PMLR, pp. 639–649.
You, H., Yu, Y., Trask, N., Gulian, M. & D’Elia, M. (2021), ‘Data-driven learning of
nonlocal physics from high-fidelity synthetic data’, _Computer Methods in Applied_
_Mechanics and Engineering_ **374**, 113553.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0045782520307386_
Zhu, Y. & Zabaras, N. (2018), ‘Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification’, _Journal of Com-_
_putational Physics_ **366**, 415–447.
**URL:** _https://www.sciencedirect.com/science/article/pii/S0021999118302341_


ERROR ESTIMATES FOR DEEPONETS 73


Appendix A. Notation for Standard Fourier basis


In several instances in this paper, we employ the following “standard” real
Fourier basis _{_ **e** _k_ _}_ _k∈_ Z _d_ in _d_ dimensions: For _k_ = ( _k_ 1 _, . . ., k_ _d_ ) _∈_ Z _[d]_, we define



**e** _k_ := _C_ _k_



cos( _k · x_ ) _,_ ( _k_ 1 _≥_ 0) _,_
(A.1)
�sin( _k · x_ ) _,_ ( _k_ 1 _<_ 0) _,_



where the factor _C_ _k_ _>_ 0 ensures that **e** _k_ is properly normalized, i.e. that _∥_ **e** _k_ _∥_ _L_ 2 (T _d_ ) =
1, or explicitly,



1
_C_ _k_ =
(2 _π_ ) _[d]_



2 _,_ ( _k ̸_ = 0) _,_
�1 _,_ ( _k_ = 0) _._



We note that the basis _{_ **e** _k_ _}_ _k∈_ Z _d_ simply consists of the real and imaginary parts of
the complex Fourier basis _{e_ _[ik][·][x]_ _}_ _k∈_ Z _d_ .
On occasion, it will also be convenient to write the standard Fourier basis in
the form _{_ **e** _j_ _}_ _j∈_ N (indexed by integer _j ∈_ N, rather than _k ∈_ Z _[d]_ ). In this case, we
identify


**e** _j_ ( _x_ ) := **e** κ( _j_ ) ( _x_ ) _,_ ( _j ∈_ N) _,_


where κ : N _→_ Z _[d]_ is a fixed enumeration of Z _[d]_, with the property that _j �→|_ κ( _j_ ) _|_ _∞_
is monotonically increasing, i.e. such that _j ≤_ _j_ _[′]_ implies that _|_ κ( _j_ ) _|_ _∞_ _≤|_ κ( _j_ _[′]_ ) _|_ _∞_,
where


_|k|_ _∞_ := max _k_ = ( _k_ 1 _, . . ., k_ _d_ ) _∈_ Z _[d]_ _._ (A.2)
_ℓ_ =1 _,...,d_ _[|][k]_ _[ℓ]_ _[|][,]_


Appendix B. On the definition of Error (2.11)


We need the following lemma in order to conclude that the error (2.11) is welldefined on _L_ [2] ( _D_ ), even if the encoder _E_ is only well-defined on continuous functions.


_**Lemma**_ **B.1** _**.**_ Let _E_ : _C_ ( _D_ ) _→_ R _[m]_ denote the point-wise encoder _u �→E_ ( _u_ ) =
( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )), for some _x_ 1 _, . . ., x_ _m_ _∈_ _D_ . There exists a Borel measurable
extension _E_ : _L_ [2] ( _D_ ) _→_ R _[m]_, such that _E_ ( _u_ ) = _E_ ( _u_ ) for any _u ∈_ _C_ ( _D_ ) _∩_ _L_ [2] ( _D_ ).


_Proof._ It suffices to consider the case _m_ = 1. In this case, we note that _E_ ( _u_ ) =
lim sup _k→∞_ _E_ _k_ ( _u_ ) for any _u ∈_ _C_ ( _D_ ), where



_E_ _k_ : _L_ [2] ( _D_ ) _→_ R _,_ _E_ _k_ ( _u_ ) =



_u_ ( _y_ ) _dy,_
_B_ 1 _/k_ ( _x_ 1 )



is continuous for any _k ∈_ N. In particular, it follows that for _u ∈_ _L_ [2] ( _D_ ), the
functional

� �
_E_ : _L_ [2] ( _D_ ) _→_ [ _−∞, ∞_ ] _,_ _E_ ( _u_ ) = lim sup _E_ _k_ ( _u_ ) _,_
_k→∞_


is Borel measurable. We can now define a measurable extension _E_ of _E_ by e.g.
setting



_E_ ( _u_ ) =



_E_ ( _u_ ) _,_ ( _E_ [�] ( _u_ ) _∈_ R) _,_
��0 _,_ ( _E_ [�] ( _u_ ) = _±∞_ ) _._






74 ERROR ESTIMATES FOR DEEPONETS


Appendix C. Proofs of Results in Section 3


C.1. **Proof of Theorem 3.1.** The proof of the universal approximation theorem
will be based on an application of the following well-known version of Lusin’s theorem (see e.g. Bogachev (2007), Thm. 7.1.13), which we will state for the special
case of probability measures on Polish spaces, below:


_**Theorem**_ **C.1** (Lusin’s theorem) _**.**_ Let _X, Y_ be separable and complete metric
spaces. Let _µ ∈P_ ( _X_ ) be a probability measure, and let _G_ : _X →_ _Y_ be a Borel
measurable mapping. Then for any _ϵ >_ 0, there exists a compact set _K ⊂_ _X_, such
that _µ_ ( _X \ K_ ) _< ϵ_, and such that the restriction _G|_ _K_ : _K →_ _Y_ is continuous.


In addition to Lusin’s theorem, the following clipping lemma will be used in the
proof of Theorem 3.1:


_**Lemma**_ **C.2** (Clipping lemma) _**.**_ Let _ϵ >_ 0, and fix 0 _< R_ 1 _< R_ 2 . There exists a
ReLU neural network _γ_ : R _[p]_ _→_ R _[p]_, such that

_∥γ_ ( _x_ ) _−_ _x∥_ _ℓ_ 2 _< ϵ,_ if _∥x∥_ _ℓ_ 2 _≤_ _R_ 1 _,_
� _∥γ_ ( _x_ ) _∥_ _ℓ_ 2 _≤_ _R_ 2 _,_ _∀_ _x ∈_ R _[p]_ _._


_Proof._ Without loss of generality, we may assume that _ϵ ≤_ _R_ 2 _−_ _R_ 1 . We first note
that _x �→_ _σ_ _R_ 1 ( _x_ ) := min(max( _x, −R_ 1 ) _, R_ 1 ) maps R _[p]_ _→_ [ _−R_ 1 _, R_ 1 ] _[p]_, and _σ_ _R_ 1 can be
represented exactly by a (two-layer) ReLU neural network. Furthermore, for any
_x ∈_ [ _−R_ 1 _, R_ 1 ] _[p]_, we have _σ_ _R_ 1 ( _x_ ) = _x_ . Define a continuous function _φ_ : R _[p]_ _→_ R _[p]_ by



_φ_ ( _x_ ) :=



 _x,_ ( _∥x∥_ _ℓ_ 2 _≤_ _R_ 1 ) _,_

_x_

 _R_ 1 ( _∥x∥_ _ℓ_ 2 _> R_ 1 ) _._

 _∥x∥_ _ℓ_ 2 _[,]_



By the universal approximation theorem, there exists a ReLU network � _γ_ : [ _−R_ 1 _, R_ 1 ] _[p]_ _→_
R _[p]_, such that

�
_∥γ_ ( _x_ ) _−_ _φ_ ( _x_ ) _∥_ _ℓ_ 2 _< ϵ,_ _∀_ _x ∈_ [ _−R_ 1 _, R_ 1 ] _[p]_ _._
Define now _γ_ : R _[p]_ _→_ R _[p]_, by _γ_ ( _x_ ) := � _γ ◦_ _σ_ _R_ 1 ( _x_ ). Then, for _∥x∥_ _ℓ_ 2 _≤_ _R_ 1, we have


�
_∥γ_ ( _x_ ) _−_ _x∥_ _ℓ_ 2 = _∥γ_ ( _x_ ) _−_ _φ_ ( _x_ ) _∥_ _ℓ_ 2 _< ϵ,_


and


sup sup
_x∈_ R _[p]_ _[ ∥][γ]_ [(] _[x]_ [)] _[∥]_ _[ℓ]_ [2] [ =] _ξ∈_ [ _−R_ 1 _,R_ 1 ] _[p]_ _[ ∥][γ]_ [�][(] _[ξ]_ [)] _[∥]_ _[ℓ]_ [2]


_≤_ sup
_ξ∈_ [ _−R_ 1 _,R_ 1 ] _[p]_ _[ {∥][γ]_ [�][(] _[ξ]_ [)] _[ −]_ _[φ]_ [(] _[ξ]_ [)] _[∥]_ _[ℓ]_ [2] [ +] _[ ∥][φ]_ [(] _[ξ]_ [)] _[∥]_ _[ℓ]_ [2] _[}]_


_≤_ _ϵ_ + _R_ 1 _≤_ _R_ 2 _._


                       

Using the above clipping lemma, we can now prove the universal approximation
theorem.


_Proof of Theorem 3.1._ Let _ϵ >_ 0 be given. By assumption _G_ : _C_ ( _D_ ) _→_ _L_ [2] ( _U_ ) is a
measurable mapping, _G ∈_ _L_ [2] ( _µ_ ), where _µ ∈P_ ( _C_ ( _D_ )). We have to show that there
exists a DeepONet _N_ = _R ◦A ◦E_, such that


_∥G −N∥_ _L_ 2 ( _µ_ ) _< ϵ._


Given _M >_ 0, define _G_ _M_ by clipping _G_ at size _M >_ 0, i.e. define



_G_ _M_ ( _u_ ) :=



 _G_ ( _u_ ) _,_ ( _∥G_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) _≤_ _M_ ) _,_

_G_ ( _u_ )

 _M_ _,_ ( _∥G_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) _> M_ ) _._

 _∥G_ ( _u_ ) _∥_ _L_ 2 ( _U_ )


ERROR ESTIMATES FOR DEEPONETS 75


so that _∥G_ _M_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) _≤_ _M_ for all _u ∈_ _C_ ( _D_ ). Then


_∥G −N∥_ _L_ 2 ( _µ_ ) _≤∥G −G_ _M_ _∥_ _L_ 2 ( _µ_ ) + _∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ) _,_ (C.1)


and the first term goes to 0 by the dominated convergence theorem and the fact
that _G_ _M_ ( _u_ ) _→G_ ( _u_ ) pointwise, as _M →∞_ . We can thus choose _M > ϵ_, such that


_∥G −G_ _M_ _∥_ _L_ 2 ( _µ_ ) _< ϵ/_ 3 _._ (C.2)


It now remains to approximate _G_ _M_ by a suitable DeepONet _N_, where _G_ _M_ is
bounded.
Next, we note that _C_ ( _D_ ) and _L_ [2] ( _U_ ) are Polish spaces (separable, complete
metric spaces). This allows us to invoke Lusin’s theorem, Theorem C.1, which
shows that there exists a compact set _K ⊂_ _C_ ( _D_ ), such that the restriction _G_ _M_ _|_ _K_
of _G_ _M_ to _K_,
_G_ _M_ _|_ _K_ : _K →_ _L_ [2] ( _U_ ) _,_

is continuous, and such that _µ_ ( _C_ ( _D_ ) _\ K_ ) _<_ ( _ϵ/_ 9 _M_ ) [2] .
Next, fix an orthonormal basis _φ_ 1 _, φ_ 2 _, φ_ 3 _, · · · ⊂_ _L_ [2] ( _U_ ) consisting of continuous
functions. For _κ ∈_ N, let _P_ _κ_ : _L_ [2] ( _U_ ) _→_ _C_ ( _U_ ) denote the projection onto _φ_ 1 _, . . ., φ_ _κ_,
i.e.



_P_ _κ_ ( _v_ ) =



_κ_
� _⟨v, φ_ _k_ _⟩φ_ _k_ _._


_k_ =1



We note that _P_ _κ_ : _L_ [2] ( _U_ ) _→_ _C_ ( _U_ ) is continuous for any fixed _κ_ . Let now _K_ _[′]_ :=
_G_ _M_ ( _K_ ) denote the image of the compact set _K_ under _G_ _M_ . Since _G_ _M_ _|_ _K_ : _K →_ _L_ [2] ( _U_ )
is continuous, _K_ _[′]_ is compact as a subset of _L_ [2] ( _U_ ). By the compactness of _K_ _[′]_,
there exists _κ ∈_ N, such that max _v∈K_ _′_ _∥v −_ _P_ _κ_ ( _v_ ) _∥_ _< ϵ/_ 6. We conclude that the
composition _G_ [�] := _P_ _κ_ _◦G_ _M_ : _K →_ _C_ ( _U_ ), with _K ⊂_ _C_ ( _D_ ) a compact subset, is
continuous, and



max
_u∈K_



�
��� _G_ _M_ ( _u_ ) _−_ _G_ ( _u_ )��� _L_ [2] ( _U_ ) [= max] _u∈K_ _[∥G]_ _[M]_ [(] _[u]_ [)] _[ −]_ _[P]_ _[κ]_ _[ ◦G]_ _[M]_ [(] _[u]_ [)] _[∥]_ _L_ [2] [2] ( _U_ ) (C.3)


= max
_v∈K_ _[′]_ _[ ∥][v][ −]_ _[P]_ _[κ]_ [(] _[v]_ [)] _[∥]_ _[L]_ [2] [(] _[U]_ [)] _[ < ϵ/]_ [6] _[.]_



We can apply the universal approximation theorem of continuous operators on
compact subsets of Chen & Chen (1995) to the continuous mapping _G_ [�] : _K ⊂_
_C_ ( _D_ ) _→_ _C_ ( _U_ ) to conclude that there exists an operator network _N_ [�] with a single
hidden layer in the approximator network and a single hidden layer in the trunk
network (and with _τ_ 0 _≡_ 0), such that


sup _∥G_ [�] ( _u_ ) _−_ _N_ [�] ( _u_ ) _∥_ _L_ 2 ( _U_ ) _< ϵ/_ 12 _._
_u∈K_


Note that this implies in particular that


_∥N_ [�] ( _u_ ) _∥_ _L_ 2 ( _U_ ) _≤∥G_ _M_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) + _∥G_ _M_ ( _u_ ) _−_ _G_ [�] ( _u_ ) _∥_ _L_ 2 ( _U_ ) + _∥G_ [�] ( _u_ ) _−_ _N_ [�] ( _u_ ) _∥_ _L_ 2 ( _U_ )
_≤_ _M_ + _ϵ/_ 6 + _ϵ/_ 6 _<_ 2 _M,_


for all _u ∈_ _K_ . We note that after suitably modifying [2] the (linear) output layers of
the branch and trunk nets of _N_ [�], we can write _N_ [�] in the form



�
_N_ ( _u_ )( _y_ ) =



_p_
� _β_ _k_ ( _u_ ) _τ_ _k_ ( _y_ ) _,_


_k_ =1



� 2 If _A ∈_ R _p×p_ is an invertible matrix, then the transformed trunk and branch nets � _τ_ := _A·τ_ and
_β_ := _A_ _[−][T]_ _· β_ represent the same DeepONet, i.e. we have [�] _[p]_ _k_ =1 _[β]_ [�] _[k]_ [(] _[u]_ [)] _[τ]_ [�] _[k]_ [(] _[y]_ [) =][ �] _[p]_ _k_ =1 _[β]_ _[k]_ [(] _[u]_ [)] _[τ]_ _[k]_ [(] _[y]_ [)]
for all _u, y_ .


76 ERROR ESTIMATES FOR DEEPONETS


with _orthonormal_ trunk net functions _{τ_ 1 _, . . ., τ_ _p_ _} ⊂_ _L_ [2] ( _U_ ). In particular, we then
have


_∥N_ [�] ( _u_ ) _∥_ _L_ 2 = _∥β_ ( _u_ ) _∥_ _ℓ_ 2 _,_ _∀_ _u ∈_ _C_ ( _D_ ) _,_


and


_∥β_ ( _u_ ) _∥_ _ℓ_ 2 _≤_ _M_ + _ϵ/_ 3 _<_ 2 _M,_ _∀_ _u ∈_ _K._


Applying Lemma C.2 with _R_ 1 = _M_ + _ϵ/_ 3, _R_ 2 = 2 _M_, we conclude that there exists
a ReLU neural network _γ_ : R _[p]_ _→_ R _[p]_, such that

_∥γ_ ( _x_ ) _−_ _x∥_ _ℓ_ 2 _< ϵ/_ 12 _,_ if _∥x∥_ _ℓ_ 2 _≤_ _R_ 1 _,_
� _∥γ_ ( _x_ ) _∥_ _ℓ_ 2 _≤_ 2 _M,_ _∀_ _x ∈_ R _[p]_ _._


We now define a “clipped” DeepONet _N_ : _C_ ( _D_ ) _→_ _L_ [2] ( _U_ ), by



_N_ ( _u_ ) :=



_p_
� _γ_ _k_ ( _β_ ( _u_ )) _τ_ _k_ ( _y_ ) _._


_k_ =1



Then _N_ ( _u_ ) satisfies


max _N_ ( _u_ ) _∥_ _L_ 2 + max _N_ ( _u_ ) _−_ _G_ [�] ( _u_ ) _∥_ _L_ 2
_u∈K_ _[∥N]_ [(] _[u]_ [)] _[ −]_ _[G]_ [�][(] _[u]_ [)] _[∥]_ _[L]_ [2] _[ ≤]_ [max] _u∈K_ _[∥N]_ [(] _[u]_ [)] _[ −]_ [�] _u∈K_ _[∥]_ [�]

= max _N_ ( _u_ ) _−_ _G_ [�] ( _u_ ) _∥_ _L_ 2
_u∈K_ _[∥][γ]_ [(] _[β]_ [(] _[u]_ [))] _[ −]_ _[β]_ [(] _[u]_ [)] _[∥]_ _[ℓ]_ [2] [ + max] _u∈K_ _[∥]_ [�]

_≤_ max _N_ ( _u_ ) _−_ _G_ [�] ( _u_ ) _∥_ _L_ 2
_∥x∥_ _ℓ_ 2 _≤R_ 1 _[∥][γ]_ [(] _[x]_ [)] _[ −]_ _[x][∥]_ _[ℓ]_ [2] [ + max] _u∈K_ _[∥]_ [�]

_≤_ _ϵ/_ 12 + _ϵ/_ 12 = _ϵ/_ 6 _,_


and _N_ ( _u_ ) is bounded from above by


_∥N_ ( _u_ ) _∥_ _L_ 2 = _∥γ_ ( _β_ ( _u_ )) _∥_ _ℓ_ 2 _≤_ 2 _M,_ _∀_ _u ∈_ _C_ ( _D_ ) _._


It follows that for this clipped DeepONet _N_, we have


_∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ) _≤∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ; _K_ ) + _∥G_ _M_ _∥_ _L_ 2 ( _µ_ ; _X\K_ ) + _∥N∥_ _L_ 2 ( _µ_ ; _X\K_ )

_≤∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ; _K_ ) + 3 _Mµ_ ( _X \ K_ ) [1] _[/]_ [2]



(C.4)



_≤∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ; _K_ ) + _ϵ/_ 3 _,_ (C.5)


and, using (C.3) and (C.4), we find



_∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ; _K_ ) _≤_ max _u∈K_



� _∥G_ _M_ ( _u_ ) _−_ _G_ [�] ( _u_ ) _∥_ _L_ 2 ( _U_ ) + _∥G_ [�] ( _u_ ) _−N_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) �



_u∈K_ (C.6)

_< ϵ/_ 3 _._



Hence, by (C.5) and (C.6), we have


_∥G_ _M_ _−N∥_ _L_ 2 ( _µ_ ) _<_ 2 _ϵ/_ 3 _._ (C.7)


Combining (C.7) and (C.2), we conclude that the DeepONet _N_ satisfies


_∥G −N∥_ _L_ 2 ( _µ_ ) _<_ 2 _ϵ/_ 3 + _ϵ/_ 3 = _ϵ._


                       

ERROR ESTIMATES FOR DEEPONETS 77


C.2. **Proof of Theorem 3.3.**


_Proof._ The upper estimate follows by a suitable decomposition of the difference.
We write


_N −G_ = _R ◦A ◦E −G_ = [ _R ◦A ◦E −R ◦P ◦G_ ] + [ _R ◦P ◦G −G_ ]


= [ _R ◦A ◦E −R ◦P ◦G ◦D ◦E_ ]


+ [ _R ◦P ◦G ◦D ◦E −R ◦P ◦G_ ]


+ [ _R ◦P ◦G −G_ ]


=: _T_ 1 + _T_ 2 + _T_ 3 _._


We can now estimate the norm of each of the three terms, as follows:


1 _/_ 2

_∥T_ 1 _∥_ _L_ 2 ( _µ_ ) = _∥R ◦A ◦E −R ◦P ◦G ◦D ◦E∥_ [2] _L_ [2] ( _U_ ) _[dµ]_
�ˆ _X_ �


1 _/_ 2

_≤_ Lip( _R_ ) _∥A ◦E −P ◦G ◦D ◦E∥_ [2] _ℓ_ [2] (R _[p]_ ) _[dµ]_
�ˆ _X_ �


1 _/_ 2

= Lip( _R_ ) _ℓ_ (R _[p]_ ) _[d]_ [(] _[E]_ [#] _[µ]_ [)]
�ˆ R _[m]_ _[ ∥A −P ◦G ◦D∥]_ [2] �

= Lip( _R_ ) _∥A −P ◦G ◦D∥_ _L_ 2 ( _E_ # _µ_ ) _._


In the second line above, we denote


Lip( _R_ ) = Lip � _R_ : (R _[p]_ _, ∥· ∥_ _ℓ_ 2 (R _p_ ) ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ( _U_ ) )� _._


For the second term, we obtain


1 _/_ 2

_∥T_ 2 _∥_ _L_ 2 ( _µ_ ) = _∥R ◦P ◦G ◦D ◦E −R ◦P ◦G∥_ [2] _L_ [2] ( _U_ ) _[dµ]_
�ˆ _X_ �


1 _/_ 2

_≤_ Lip( _R ◦P_ ) _∥G ◦D ◦E −G∥_ [2] _L_ [2] ( _U_ ) _[dµ]_
�ˆ _X_ �



where



1 _/_ 2

_≤_ Lip( _R ◦P_ )Lip _α_ ( _G_ ) _∥D ◦E −_ Id _∥_ _X_ [2] _[α]_ _[dµ]_ _,_
�ˆ _X_ �


Lip( _R ◦P_ ) = Lip( _R ◦P_ : ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ( _U_ ) ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ( _U_ ) )) _,_

Lip _α_ ( _G_ ) = Lip _α_ ( _G_ : ( _A ⊂_ _X, ∥· ∥_ ) _→_ ( _L_ [2] ( _U_ ) _, ∥· ∥_ _L_ 2 ( _U_ ) )) _,_



and _α ∈_ (0 _,_ 1]. Since _α ∈_ (0 _,_ 1], we can estimate the last term by Jensen’s inequality
to obtain:


1 _/_ 2

_∥T_ 2 _∥_ _L_ 2 ( _µ_ ) _≤_ Lip( _R ◦P_ )Lip _α_ ( _G_ ) _∥D ◦E −_ Id _∥_ _X_ [2] _[α]_ _[dµ]_
�ˆ _X_ �


_α/_ 2

_≤_ Lip( _R ◦P_ )Lip _α_ ( _G_ ) _∥D ◦E −_ Id _∥_ _X_ [2] _[dµ]_
�ˆ _X_ �

= Lip( _R ◦P_ )Lip _α_ ( _G_ ) _∥D ◦E −_ Id _∥_ _[α]_ _L_ [2] ( _µ_ ) _[.]_


78 ERROR ESTIMATES FOR DEEPONETS


Finally for the third term, we have (by the definition of the push-forward)


1 _/_ 2

_∥T_ 3 _∥_ _L_ 2 ( _µ_ ) = _∥R ◦P ◦G −G∥_ [2] _L_ [2] ( _U_ ) _[dµ]_
�ˆ _X_ �



� 1 _/_ 2



=



ˆ
�



_∥R ◦P −_ Id _∥_ [2] _L_ [2] ( _U_ ) _[d]_ [(] _[G]_ [#] _[µ]_ [)]
_L_ [2] ( _U_ )



= _∥R ◦P −_ Id _∥_ _L_ 2 ( _G_ # _µ_ ) _._


                       

C.3. **Proof of Theorem 3.10.** The proof of Theorem 3.10 is a consequence of the
following series of lemmas.


_**Lemma**_ � **C.3** _**.**_ Given a separable Hilbert space _H_ and a _p_ -dimensional subspace
_V ⊂_ _H_, we have



� �
_E_ Proj ( _V_ ) = _∥v∥_ [2] _dµ_ ( _v_ ) _−_
ˆ _H_



_p_
�


_i_ =1



_|⟨v,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ ) _,_ (C.8)

ˆ _H_



where� � _v_ _i_, _i_ = 1 _, . . ., p_ is any orthonormal basis of _V_ [�] . In particular, minimizing
_E_ Proj is equivalent to maximizing



_p_
�


_i_ =1



_|⟨v,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ ) =

ˆ _X_



_p_
� � _|⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] + _⟨v_ � _i_ _,_ Γ _v_ � _i_ _⟩_ � _,_ (C.9)

_i_ =1



where



Γ = ( _v −_ E[ _v_ ]) _⊗_ ( _v −_ E[ _v_ ]) _dµ_ ( _v_ ) _._
ˆ _X_



_Proof._ Let � _v_ _i_, _i_ = 1 _, . . ., p_ be any orthonormal basis of _V_ [�] . Then



����� 2



�
_v_ � inf _∈V_ [�] _∥v −_ _v∥_ [2] =



_v −_
�����



_p_
� _⟨v,_ � _v_ _i_ _⟩_ _v_ � _i_


_k_ =1



_p_
�



_,_



and since _v −_ [�] _k_ _[⟨][v,]_ [ �] _[v]_ _[i]_ _[⟩]_ _[v]_ [�] _[i]_ [ is perpendicular to all][ �] _[v]_ _[i]_ [, we have]



����� 2



2



����� 2



2



_∥v∥_ [2] =



_v −_
�����



_p_
� _⟨v,_ � _v_ _i_ _⟩_ _v_ � _i_


_k_ =1



_p_
�



+



�����



_p_
�



� _⟨v,_ � _v_ _i_ _⟩_ _v_ � _i_


_k_ =1



= inf _v_ � _∈V_ [�] _∥v −_ _v_ � _∥_ [2] +



_p_
� _|⟨v,_ � _v_ _i_ _⟩|_ [2] _._


_k_ =1



This is equivalent to (C.8). Furthermore, for any _i_ = 1 _, . . ., p_, we have


_|⟨v,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ ) = _|⟨v −_ E[ _v_ ] _,_ � _v_ _i_ _⟩_ + _⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ )

ˆ _H_ ˆ _H_



_|⟨v,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ ) =
_H_ ˆ



_|⟨v −_ E[ _v_ ] _,_ � _v_ _i_ _⟩_ + _⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] _dµ_ ( _v_ )
_H_



=
ˆ _H_



=
ˆ _H_



� _|⟨v −_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] + 2 _⟨v −_ E[ _v_ ] _,_ � _v_ _i_ _⟩⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩_ + _|⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] [�] _dµ_ ( _v_ )


� _|⟨v −_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] + _|⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] [�] _dµ_ ( _v_ )



� �
= _⟨v_ _i_ _,_ Γ _v_ _i_ _⟩_ + _|⟨_ E[ _v_ ] _,_ � _v_ _i_ _⟩|_ [2] _._






ERROR ESTIMATES FOR DEEPONETS 79


_**Lemma**_ **C.4** _**.**_ The covariance operator Γ of a probability measure _ν ∈P_ 2 ( _H_ ) is a
compact, self-adjoint operator. In particular, there exists a discrete set of (unique)
eigenvalues _λ_ 1 _> λ_ 2 _> . . ._, and orthogonal subspaces _E_ 1 _, E_ 2 _, . . ._, such that



_∞_
�

_j_ =1



_H_ =



_∞_
� _E_ _j_ _,_ and Γ =

_j_ =1



_λ_ _j_ _P_ _E_ _j_ _._



Here _P_ _E_ : _H →_ _E_ denotes the orthogonal projection onto the subspace _E ⊂_ _H_ .


_Proof._ The proof can e.g. be found in (Pinelis & Molzon 2016, Appendix E,F). 

As an immediate corollary, we also obtain


_**Lemma**_ **C.5** _**.**_ For any _ν ∈P_ 2 ( _H_ ), the operator


Γ = ( _v ⊗_ _v_ ) _dν_ ( _v_ ) _,_
ˆ _H_


is a self-adjoint, compact operator, and hence it also possesses an eigendecomposition as in Lemma C.4.


_Proof._ We can write Γ = Γ + E[ _v_ ] _⊗_ E[ _v_ ], where E[ _v_ ] := ´ _H_ _[v dν]_ [(] _[v]_ [) is the expected]

value under _ν_ . It is now immediate that Γ is self-adjoint, since both Γ and E[ _v_ ] _⊗_ E[ _v_ ]
are self-adjoint. Furthermore, Γ is also compact, since Γ = Γ + E[ _v_ ] _⊗_ E[ _v_ ] is the
sum of a compact operator Γ and the finite-rank operator E[ _v_ ] _⊗_ E[ _v_ ]. 

The next lemma shows that the quantity (C.9) to be maximized can be written
in an equivalent form, which only involves the orthogonal projection onto _V_ _p_ .


_**Lemma**_ **C.6** _**.**_ Let Γ = Γ + E[ _v_ ] _⊗_ E[ _v_ ]. Let _φ_ _k_, ( _k ∈_ N), be an orthonormal
basis of eigenvectors of Γ with corresponding eigenvalues _λ_ _k_ . Let � _v_ 1 _, . . .,_ � _v_ _p_ be any
orthonormal basis of _V_ _p_ . Then



_p_


� �

� _⟨v_ _i_ _,_ Γ _v_ _i_ _⟩_ =

_j_ =1



_∞_
� _λ_ _k_ _∥P_ _V_ _p_ _φ_ _k_ _∥_ [2] _._ (C.10)


_k_ =1



Here _P_ _V_ _p_ : _H →_ _V_ _p_ denotes the orthogonal projection onto _V_ _p_ .


_Proof._ Let _u_ 1 _, . . ., u_ _p_ _∈_ _V_ _p_ be an orthonormal basis _V_ _p_ . Let _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ denote
the eigenvalues of Γ with corresponding orthonormal eigenbasis _φ_ 1 _, φ_ 2 _, . . ._ . Then



_u_ _j_ =



_∞_
� _⟨u_ _j_ _, φ_ _k_ _⟩φ_ _k_ _,_


_k_ =1


80 ERROR ESTIMATES FOR DEEPONETS


and hence



_p_
�

_j_ =1



� _u_ _j_ _,_ Γ _u_ _j_ � =


=


=


=


=



_p_
�

_j_ =1


_p_
�

_j_ =1


_p_
�

_j_ =1



_u_ _j_ _,_ Γ

�



�



_∞_
� _⟨u_ _j_ _, φ_ _k_ _⟩φ_ _k_


_k_ =1



_∞_
�



_∞_
� _⟨u_ _j_ _, φ_ _k_ _⟩_ � _u_ _j_ _,_ Γ _φ_ _k_ �


_k_ =1


_∞_
� _λ_ _k_ _|⟨u_ _j_ _, φ_ _k_ _⟩|_ [2]


_k_ =1



_∞_
� _λ_ _k_


_k_ =1









_p_
�



� _|⟨u_ _j_ _, φ_ _k_ _⟩|_ [2]

_j_ =1









_∞_

2

� _λ_ _k_ �� _P_ _V_ _p_ _φ_ _k_ �� _._


_k_ =1




                       

_**Lemma**_ � **C.7** _**.**_ If _V_ _p_ is a minimizing subspace, i.e. a subspace which minimizes
_E_ Proj, then



_n−_ 1
� _E_ _j_ _⊂_ _V_ _p_ _⊂_

_j_ =1



_n_
� _E_ _j_ _,_ (C.11)

_j_ =1



_n_
�



where _n_ is chosen such that



dim � _[n]_ � _[−]_ [1] _E_ _j_ � =

_j_ =1



_n−_ 1
� dim( _E_ _j_ ) _< p ≤_

_j_ =1



_n_ _n_
� dim( _E_ _j_ ) = dim � �

_j_ =1 _j_ =1



_n_
�



� _E_ _j_ � _._

_j_ =1



Here, the _E_ _j_ denote the eigenspaces of


Γ = Γ + E[ _v_ ] _⊗_ E[ _v_ ] =



_∞_
�

_j_ =1



_λ_ _j_ _P_ _E_ _j_ _,_



corresponding to the (distinct) eigenvalues _λ_ 1 _> λ_ 2 _> λ_ 3 _> . . ._ of Γ.


_Proof._ Let _V_ _p_ _⊂_ _H_ be a minimizer of _E_ [�] Proj with respect to _ν_ of dimension dim( _V_ _p_ ) =
_p_ . For each eigenspace _E_ _j_ of Γ, denote


_p_ _j_ = dim( _E_ _j_ ) _,_


and define _p_ 0 := 0. Now, we choose an orthonormal eigenbasis _φ_ 1 _, φ_ 2 _, . . ._ of Γ, with
corresponding eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._, such that


_E_ _j_ = span( _φ_ _p_ _j−_ 1 +1 _, . . ., φ_ _p_ _j_ ) _,_ ( _j ∈_ N) _._


Note that the eigenvalues thus satisfy



_λ_ 1 = _· · ·_ = _λ_ _p_ 1 _> λ_ _p_ 1 +1 = _· · ·_ = _λ_ _p_ 2
~~�~~ ~~�~~ � ~~�~~ ~~�~~ ~~��~~ �

= _λ_ 1 = _λ_ 2




_> . . ._



By Lemma C.3 and C.6, _V_ _p_ is a maximizer of the mapping



_∞_

� 2
_V �→_ � _λ_ _k_ �� _P_ _V_ � _φ_ _k_ �� _,_


_k_ =1


ERROR ESTIMATES FOR DEEPONETS 81


among all _p_ -dimensional subspaces _V_ [�] _⊂_ _X_ . Given a _p_ -dimensional subspace _V_ [�] _⊂_ _X_,
let _u_ 1 _, . . ., u_ _p_ be an orthonormal basis of _V_ [�] . We note that



_p_
� _∥u_ _j_ _∥_ [2] = _p,_

_j_ =1



_∞_

2

� �� _P_ _V_ � _φ_ _k_ �� =


_k_ =1



_∞_ _p_
� � _|⟨u_ _j_ _, φ_ _k_ _⟩|_ [2] =

_k_ =1 _j_ =1



And for all _k ∈_ N, we have
2
0 _≤_ �� _P_ _V_ � _φ_ _k_ �� _≤_ 1 _._
Thus, for any _p_ -dimensional subspace, the coefficients

_α_ _k_ = _α_ _k_ ( _V_ [�] ) := _∥P_ _V_ � _φ_ _k_ _∥_ [2] _,_ ( _k ∈_ N) _,_


belong to the set



_α_ = ( _α_ _k_ ) _k∈_ N

�



�



_α_ _k_ _∈_ [0 _,_ 1] _∀_ _k ∈_ N _,_
�����



_._

�



_._

�



_A_ _p_ :=



_∞_
� _α_ _k_ = _p_


_k_ =1



_∞_
�



And we are interested in the maximizer _V_ [�] = _V_ _p_ of



( _α_ _k_ ( _V_ [�] )) _k∈_ N _�→_


We now make the following claim:


_**Claim.**_ For any ( _α_ _k_ ) _k∈_ N _∈A_ _p_, we have



_∞_
� _λ_ _k_ _α_ _k_ ( _V_ [�] ) _._


_k_ =1



_∞_
� _λ_ _k_ _α_ _k_ _≤_


_k_ =1



_p_
� _λ_ _k_ _,_ (C.12)


_k_ =1



with equality, if and only if


_α_ _k_ =


where _n_ is chosen such that



1 _,_ �if _k ≤_ [�] _j_ _[n]_ =1 _[−]_ [1] _[p]_ _[j]_ � _,_
0 _,_ �if _k >_ [�] _[n]_ _j_ =1 _[p]_ _[j]_ � _,_



_n−_ 1
� _p_ _j_ _≤_ _p <_

_j_ =1



_n_
� _p_ _j_ _._

_j_ =1



Before proving the above claim, we show that it implies (C.11). Indeed, for the
subspace _V_ [�] = span( _φ_ 1 _, . . ., φ_ _p_ ), we clearly have



_α_ _k_ ( _V_ [�] ) = _∥P_ _V_ � _φ_ _k_ _∥_ [2] =



1 _,_ ( _k ≤_ _p_ ) _,_
�0 _,_ ( _k > p_ ) _,_



and hence

_∞_
� _λ_ _k_ _α_ _k_ ( _V_ [�] ) =


_k_ =1



_p_
� _λ_ _k_ _,_


_k_ =1



achieves the upper bound in (C.12). If _V_ _p_ is another optimizer, then we must thus
also have

_∞_ _p_
� _λ_ _k_ _α_ _k_ ( _V_ _p_ ) = � _λ_ _k_ _._



� _λ_ _k_ _α_ _k_ ( _V_ _p_ ) =


_k_ =1



_p_
�



� _λ_ _k_ _._


_k_ =1



The claim then implies that


_∥P_ _V_ _p_ _φ_ _k_ _∥_ [2] =



1 _,_ �if _k ≤_ [�] _[n]_ _j_ =1 _[−]_ [1] _[p]_ _[j]_ � _,_
0 _,_ �if _k >_ [�] _[n]_ _j_ =1 _[p]_ _[j]_ � _._


82 ERROR ESTIMATES FOR DEEPONETS


The latter is equivalent to the statement that



_φ_ _k_ _∈_ _V_ _p_ _,_ for _k ≤_



_n−_ 1
� _p_ _j_ _,_ and _φ_ _k_ _⊥_ _V_ _p_ _,_ for _k >_

_j_ =1



_n_
� _p_ _j_ _,_

_j_ =1



i.e. that

_n−_ 1
� _E_ _j_ _⊂_ _V_ _p_ _,_ and

_j_ =1



_∞_
� _E_ _j_ _⊥_ _V_ _p_ _._

_j_ = _n_ +1



Thus, assuming the claim (C.12), it follows that for any optimal subspace _V_ _p_, we
must have

_n−_ 1 _n_
� _E_ _j_ _⊂_ _V_ _p_ _⊂_ � _E_ _j_ _._



� _E_ _j_ _⊂_ _V_ _p_ _⊂_

_j_ =1



_n_
�



� _E_ _j_ _._

_j_ =1



We finally need to **prove the claim** : To prove the inequality (C.12), we simply
note that



_∞_
� _λ_ _k_ _α_ _k_ _−_


_k_ =1



_p_
� _λ_ _k_ (1 _−_ _α_ _k_ )


_k_ =1



_∞_
� _λ_ _k_ _α_ _k_ _−_


_k_ =1



_p_
� _λ_ _k_ =


_k_ =1



_∞_
�



_p_
� _λ_ _k_ _α_ _k_ _−_


_k_ =1



_p_
�



= � _λ_ _k_ _α_ _k_ _−_

_k>p_



_p_
� _λ_ _k_ (1 _−_ _α_ _k_ ) _._


_k_ =1



Since the sequence of eigenvalues _λ_ _k_ is monotonically decreasing and _α_ _k_ _≥_ 0, 1 _−_
_α_ _k_ _≥_ 0, we have



_p_
� _λ_ _p_ (1 _−_ _α_ _k_ ) _,_


_k_ =1



� _λ_ _k_ _α_ _k_ _≤_ �

_k>p_ _k>p_



�



� _λ_ _p_ _α_ _k_ _,_ and

_k>p_



_p_
� _λ_ _k_ (1 _−_ _α_ _k_ ) _≥_


_k_ =1



and hence

� _λ_ _k_ _α_ _k_ _−_

_k>p_



and hence

�



_p_
�



� _λ_ _k_ (1 _−_ _α_ _k_ ) _≤_ �


_k_ =1



� _λ_ _p_ _α_ _k_ _−_

_k>p_



_p_
� _λ_ _p_ (1 _−_ _α_ _k_ )


_k_ =1



= � _λ_ _p_ _α_ _k_ _−_ _pλ_ _p_ +

_k>p_



_p_
� _λ_ _p_ _α_ _k_


_k_ =1



�



= _λ_ _p_


= 0 _._



_∞_
� _α_ _k_ _−_ _p_
� _k_ =1



_∞_
�
� _k_ =1



The last line follows from the fact that [�] _[∞]_ _k_ =1 _[α]_ _[k]_ [ =] _[ p]_ [, for any (] _[α]_ _[k]_ [)] _[k][∈]_ [N] _[ ∈A]_ _[p]_ [. We]
thus conclude that

_∞_ _p_
� _λ_ _k_ _α_ _k_ _≤_ � _λ_ _k_ _,_



� _λ_ _k_ _α_ _k_ _≤_


_k_ =1



_p_
�



� _λ_ _k_ _,_


_k_ =1



for all ( _α_ _k_ ) _k∈_ N _∈A_ _p_ . Furthermore, if [�] _[∞]_ _k_ =1 _[λ]_ _[k]_ _[α]_ _[k]_ [ =][ �] _[p]_ _k_ =1 _[λ]_ _[k]_ [, then in all the above]
estimates, we must have equality. In particular, we must have



_p_
� _λ_ _p_ (1 _−_ _α_ _k_ ) _._


_k_ =1



� _λ_ _k_ _α_ _k_ = �

_k>p_ _k>p_



�



� _λ_ _p_ _α_ _k_ _,_

_k>p_



_p_
� _λ_ _k_ (1 _−_ _α_ _k_ ) =


_k_ =1



The first equality is only possible, if _α_ _k_ = 0, for any _k ∈_ N, such that _λ_ _k_ _< λ_ _p_, i.e.
we must have



_α_ _k_ = 0 _,_ if _k >_



_n_
� _p_ _j_ _._

_j_ =1


ERROR ESTIMATES FOR DEEPONETS 83


The second equality is only possible, if _α_ _k_ = 1, for any _k ∈_ N, such that _λ_ _k_ _> λ_ _p_,
i.e. we must have



_α_ _k_ = 1 _,_ if _k ≤_



_n−_ 1
� _p_ _j_ _._

_j_ =1



This concludes the proof. 

_**Remark**_ **C.8** _**.**_ It is also straight-forward to check that if _V_ [�] _⊂_ _H_ is a _p_ -dimensional
subspace, such that



_n_
� dim( _E_ _k_ ) _,_


_k_ =1



_n_
� _E_ _k_ _,_


_k_ =1



where


then



_n−_ 1
� _E_ _k_ _⊂_ _V_ [�] _⊂_


_k_ =1


_n−_ 1
� dim( _E_ _k_ ) _< p ≤_


_k_ =1



_∞_
� _λ_ _k_ _∥P_ _V_ � _φ_ _k_ _∥_ [2] =


_k_ =1



_p_
� _λ_ _k_ _,_


_k_ =1



i.e. _V_ [�] is optimal for _E_ [�] Proj ( _V_ [�] ; _ν_ ), in this case.


From the last lemma, we immediately have the following corollary.


_**Corollary**_ **C.9** _**.**_ For any _n ∈_ N, the minimizing subspace _V_ _p_ of _E_ [�] Proj ( _V_ _p_ ; _ν_ ) of
dimension



_p_ =



_n_
� dim( _E_ _k_ ) _,_


_k_ =1



is unique.


_Proof._ By Lemma C.7, if _V_ _p_ is any minimizer of _E_ [�] Proj, then



_V_ _p_ _⊂_



_n_
� _E_ _k_ _._


_k_ =1



Since dim( _V_ _p_ ) = _p_ = [�] _[n]_ _k_ =1 [dim(] _[E]_ _[k]_ [), by assumption, it follows that] _[ V]_ _[p]_ [ =][ �] _[n]_ _k_ =1 _[E]_ _[k]_
is uniquely determined by the eigenspaces _E_ 1 _, . . ., E_ _n_ . 

C.3.1. _Proof of Theorem 3.10._


_Proof._ The existence and characterization of optimal subspaces _V_ [�] is a consequence
of Lemma C.7 and Remark C.8. Uniqueness of the optimal subspace _V_ _p_ _⊂_ _H_ for
_p_ _n_ = [�] _[n]_ _k_ =1 [dim(] _[E]_ _[k]_ [) is proved in Corollary C.9. Finally, the identity for the pro-]
jection error _E_ [�] Proj ( _V_ [�] ) = [�] _k>p_ _[λ]_ _[k]_ [ for the optimal subspace] _[ V]_ _[p]_ [ = span(] _[φ]_ [1] _[, . . ., φ]_ _[p]_ [),]


84 ERROR ESTIMATES FOR DEEPONETS


with _φ_ _k_ denote the eigenfunctions of the uncentered covariance operator Γ corresponding to decreasing eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._, follows from



(C.8)
_↓_

=
ˆ



_E_ � Proj ( _V_ � )



_∥v∥_ [2] _dµ_ ( _u_ ) _−_
_H_



_p_
�


_i_ =1



_|⟨v, φ_ _i_ _⟩|_ [2] _dµ_ ( _v_ )

ˆ _H_



_p_
� _⟨φ_ _i_ _,_ Γ _φ_ _i_ _⟩_


_i_ =1



_p_
�



= Tr(Γ) _−_



(C.10)
_↓_

=



_∞_
�



_∞_
� _λ_ _k_ _∥P_ _V_ _p_ _φ_ _k_ _∥_ [2]


_k_ =1



_∞_
�



� _λ_ _k_ _._

_k>p_



_p_
�



� _λ_ _k_ = �


_k_ =1



=



� _λ_ _k_ _−_


_k_ =1


_∞_
� _λ_ _k_ _−_


_k_ =1



_∞_
�




                       

C.3.2. _Proof of Theorem 3.12._


_Proof._ Since _V_ 0 is a finite-dimensional affine subspace, the existence and uniqueness
of
_v_ � 0 _∈_ argmin _∥_ E[ _v_ ] _−_ _v_ � _∥_ [2] _,_
_v_ � _∈V_ [�] 0

is straight-forward. We also note that E[ _v_ ] _−_ _v_ � 0 _⊥_ _V_ [�] . We can now write

_E_ � Proj ( _V_ � 0 ) = ˆ _H_ _v_ � inf _∈V_ [�] 0 _∥v −_ _v_ � _∥_ [2] _dν_ ( _v_ )


� �

= ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ _v_ 0 _−_ _v∥_ [2] _dν_ ( _v_ )


� �

= ˆ _H_ _v_ � inf _∈V_ [�] _∥_ [ _v −_ E[ _v_ ] _−_ _v_ ] _−_ [E[ _v_ ] _−_ _v_ 0 ] _∥_ [2] _dν_ ( _v_ )



= inf
ˆ _H_ _v_ � _∈V_ [�]



� � �
� _∥_ E[ _v_ ] _−_ _v_ 0 _∥_ [2] _−_ 2 _⟨v −_ E[ _v_ ] _−_ _v,_ E[ _v_ ] _−_ _v_ 0 _⟩_


�
+ _∥v −_ E[ _v_ ] _−_ _v∥_ [2] [�] _dν_ ( _v_ )



Since E[ _v_ ] _−_ _v_ � 0 _⊥_ _v_ � for all � _v ∈_ _V_ [�], we have


2 _⟨v −_ E[ _v_ ] _−_ _v,_ � E[ _v_ ] _−_ _v_ � 0 _⟩_ = 2 _⟨v −_ E[ _v_ ] _,_ E[ _v_ ] _−_ _v_ � 0 _⟩,_


and thus



_E_ � Proj ( _V_ � 0 ) = ˆ _H_ _v_ � inf _∈V_ [�]



� �
� _∥_ E[ _v_ ] _−_ _v_ 0 _∥_ [2] _−_ 2 _⟨v −_ E[ _v_ ] _,_ E[ _v_ ] _−_ _v_ 0 _⟩_


�
+ _∥v −_ E[ _v_ ] _−_ _v∥_ [2] [�] _dν_ ( _v_ )



�

= _∥_ E[ _v_ ] _−_ _v_ 0 _∥_ [2] _dν_ ( _v_ )
ˆ _H_


�

_−_ 2 _⟨v −_ E[ _v_ ] _,_ E[ _v_ ] _−_ _v_ 0 _⟩_ _dν_ ( _v_ )
ˆ _H_

+ ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ ) _._


The first term is independent of _v_, the second term averages to zero. Hence, we
finally obtain

_E_ � Proj ( _V_ � 0 ) = _∥_ E[ _v_ ] _−_ _v_ � 0 _∥_ [2] + ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ ) _._


ERROR ESTIMATES FOR DEEPONETS 85


Note now if _V_ [�] 0 is an affine subspace with associated vector space _V_ [�] for which the
first term is not equal to 0, then the affine subspace _V_ [�] 0 _[′]_ [:=][ E][[] _[v]_ [] +][ �] _[V]_ [ satisfies]

_E_ � Proj ( _V_ � 0 _[′]_ [) =] ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ )

_< ∥_ E[ _v_ ] _−_ _v_ � 0 _∥_ [2] + ˆ _H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ )

= _E_ [�] Proj ( _V_ [�] 0 ) _._



Thus, if _V_ [�] 0 is a minimizer among affine subspaces, then we must have � _v_ 0 = E[ _v_ ].
Next, define a measure ~~_ν_~~ _∈P_ ( _H_ ) by


Φ( _v_ ) ~~_dν_~~ ~~(~~ _v_ ) := Φ( _v −_ E[ _v_ ]) _dν_ ( _v_ ) _,_ _∀_ Φ _∈_ _L_ _[∞]_ ( _H_ ) _._

ˆ _H_ ˆ _H_



Φ( _v_ ) ~~_dν_~~ ~~(~~ _v_ ) :=
_H_ ˆ



Φ( _v −_ E[ _v_ ]) _dν_ ( _v_ ) _,_ _∀_ Φ _∈_ _L_ _[∞]_ ( _H_ ) _._
_H_



Then ´



_H_ _[v ]_ ~~_[dν]_~~ ~~[(]~~ _[v]_ [) = 0, and]
ˆ _H_ _v_ � inf _∈V_ [�] _∥v_



_H_ _v_ � inf _∈V_ [�] _∥v −_ _v_ � _∥_ [2] ~~_dν_~~ ~~(~~ _v_ ) _,_



_H_ _v_ � inf _∈V_ [�] _∥v −_ E[ _v_ ] _−_ _v_ � _∥_ [2] _dν_ ( _v_ ) = ˆ



is the projection error _E_ [�] Proj ( _V_ [�] ; ~~_ν_~~ ~~)~~ of the _p_ -dimensional vector space _V_ [�] with respect
_E_ to the measure� Proj ( _V_ � ) among ~~_ν_~~ _p_ ~~.~~ Since-dimensional subspaces. By Lemma C.7, it follows that _V_ [�] 0 is a minimizer, we must have that _V_ [�] is a minimizer of



_n−_ 1
� _E_ _j_ _⊂_ _V_ [�] _⊂_

_j_ =1



_n_
� _E_ _j_ _,_

_j_ =1



where _E_ _j_ denote the eigenspaces of the covariance operator of ~~_ν_~~ ~~,~~ given by


_v ⊗_ _v_ ~~_dν_~~ ~~(~~ _v_ ) = ( _v −_ E[ _v_ ]) _⊗_ ( _v −_ E[ _v_ ]) _dν_ ( _v_ ) = Γ _,_

ˆ _H_ ˆ _H_



_v ⊗_ _v_ ~~_dν_~~ ~~(~~ _v_ ) =
_H_ ˆ



( _v −_ E[ _v_ ]) _⊗_ ( _v −_ E[ _v_ ]) _dν_ ( _v_ ) = Γ _,_
_H_



and corresponding to distinct eigenvalues _λ_ 1 _> λ_ 2 _> . . ._ of Γ, as claimed. This
concludes the proof of this theorem. 

C.4. **Proof of Lemma 3.13.**


_Proof._ Let _V_ = span( _τ_ 1 _, . . ., τ_ _p_ ). We first note that since we have a direct sum
_L_ [2] ( _U_ ) = _V_ _[⊥]_ _⊕_ _V_, we can decompose any _u ∈_ _L_ [2] ( _U_ ) uniquely as



_u_ = _u_ _[⊥]_ +



_p_
� _α_ _k_ _τ_ _k_ _,_


_k_ =1



for coefficients _α_ 1 _, . . ., α_ _p_ _∈_ R. Let _τ_ 1 _[∗]_ _[, . . ., τ]_ _p_ _[ ∗]_ _[∈]_ [span(] _[τ]_ [1] _[, . . ., τ]_ _[p]_ [) denote the dual]
basis of _τ_ 1 _, . . ., τ_ _p_ . Taking the inner product of the last identity with _τ_ _ℓ_ _[∗]_ [, it follows]
that



_⟨τ_ _ℓ_ _[∗]_ _[, u][⟩]_ [=] _[ ⟨][τ]_ _[ ∗]_ _ℓ_ _[, u]_ _[⊥]_ _[⟩]_ [+]



_p_
� _α_ _k_ _⟨τ_ _ℓ_ _[∗]_ _[, τ]_ _[k]_ _[⟩]_ [=] _[ α]_ _[ℓ]_ _[,]_


_k_ =1



where we have used that _u_ _[⊥]_ _⊥_ _V ∋_ _τ_ _ℓ_ _[∗]_ [, and] _[ ⟨][τ]_ _[ ∗]_ _ℓ_ _[, τ]_ _[k]_ _[⟩]_ [=] _[ δ]_ _[kℓ]_ [. Applying this identity,]
we now note that



_R ◦P_ ( _u_ ) = _τ_ 0 +



_p_ _p_
�[ _P_ ( _u_ )] _k_ _τ_ _k_ = _τ_ 0 _[⊥]_ [+] �


_k_ =1 _k_ =1



�[ _P_ ( _u_ )] _k_ + _⟨τ_ _k_ _[∗]_ _[, τ]_ [0] _[⟩]_ � _τ_ _k_ _,_



and



_u_ = _u_ _[⊥]_ +



_p_
� _⟨τ_ _k_ _[∗]_ _[, u][⟩][τ]_ _[k]_ _[.]_


_k_ =1


86 ERROR ESTIMATES FOR DEEPONETS


Hence



_R ◦P_ ( _u_ ) _−_ _u_ = _τ_ 0 _[⊥]_ _[−]_ _[u]_ _[⊥]_ [+]



_p_
�


_k_ =1



�[ _P_ ( _u_ )] _k_ + _⟨τ_ _k_ _[∗]_ _[, τ]_ [0] _[−]_ _[u][⟩]_ � _τ_ _k_ _._



The norm of the last term is clearly minimized if the sum over _k_ = 1 _, . . ., p_ vanishes.
This is the case, provided that


[ _P_ ( _u_ )] _k_ = _⟨τ_ _k_ _[∗]_ _[, u][ −]_ _[τ]_ [0] _[⟩][,]_ ( _k_ = 1 _, . . ., p_ ) _._


The claim follows. 

C.5. **Proof of Proposition 3.15.**


_Proof._ Denote _V_ [�] 0 := Im( _R_ ) the affine image of _R_ = _R_ _**τ**_ . Taking into account
Theorem 3.12, we obtain

( _E_ [�] _R_ ) [2] = ˆ _Y_ _∥R ◦P −_ Id _∥_ [2] _L_ [2] _y_ _[dν]_



_∞_
� _λ_ _k_ _∥R ◦Pφ_ _k_ _−_ _φ_ _k_ _∥_ [2] _L_ [2] _y_

_k_ =1



= inf _v_ � _∈V_ [�] 0 _∥_ E _ν_ [ _v_ ] _−_ _v_ � _∥_ [2] _L_ [2] _y_ [+]


_≤∥_ E _ν_ [ _v_ ] _−_ _τ_ 0 _∥_ [2] _L_ [2] _y_ [+] �



� _λ_ _k_ _∥R ◦Pφ_ _k_ _−_ _φ_ _k_ _∥_ [2] _L_ [2] _y_

_k>p_



� _λ_ _k_ _∥τ_ _k_ _−_ _φ_ _k_ _∥_ [2] _L_ [2] _y_ [+] �

_k≤p_ _k>p_



� _λ_ _k_ _∥R ◦Pφ_ _k_ _−_ _φ_ _k_ _∥_ [2] _L_ [2] _y_

_k>p_



�
= _∥τ_ 0 _−_ _τ_ 0 _∥_ [2] _L_ [2] _y_ [+]


�
_≤∥τ_ 0 _−_ _τ_ 0 _∥_ [2] _L_ [2] _y_ [+]



_p_
�



�

� _λ_ _k_ _∥τ_ _k_ _−_ _τ_ _k_ _∥_ [2] _L_ [2] _y_ [+] �

_k_ =1



_∞_
� _λ_ _k_
� _k_ =1



�



�
_k_ =1 sup _,...,p_ _∥τ_ _k_ _−_ _τ_ _k_ _∥_ [2] _L_ [2] _y_ [+] _k>p_ � _λ_ _k_



�
_≤_ [1 + Tr(Γ _ν_ )] _k_ =0 sup _,_ 1 _,...,p_ _∥τ_ _k_ _−_ _τ_ _k_ _∥_ [2] _L_ [2] _y_ [+] _k>p_ � _λ_ _k_ _._



Taking square roots of both sides, the claimed estimate (3.25) now follows from the
trivial inequality _√a_ + _b ≤_ _[√]_ ~~_a_~~ + _√b_ for _a, b ≥_ 0. 


_a_ + _b ≤_ _[√]_ ~~_a_~~ + _√_



_b_ for _a, b ≥_ 0. 


C.6. **Proof of Lemma 3.16.**


_Proof._ By the triangle inequality, we have


� � 1 _/_ 2
_E_ _R_ _≤_ _E_ _R_ � [+] _L_ [2] _[ dν]_ [(] _[u]_ [)] _._
�ˆ _L_ [2] _[ ∥R ◦P]_ [(] _[u]_ [)] _[ −]_ _[R ◦]_ [�] _[P]_ [�][(] _[u]_ [)] _[∥]_ [2] �


Note that by the assumed orthonormality of the � _τ_ _k_, the projection can be written
as follows (cp. (3.20))



� �
_R ◦_ _P_ ( _u_ ) =



_p_


� �

� _⟨τ_ _k_ _, u⟩τ_ _k_ _._


_k_ =1



Furthermore, in terms of the dual basis _τ_ 1 _[∗]_ _[, . . ., τ]_ _p_ _[ ∗]_ _[∈]_ _[Y]_ [, we have]



_R ◦P_ ( _u_ ) =



_p_
� _⟨τ_ _k_ _[∗]_ _[, u][⟩][τ]_ _[k]_ _[.]_


_k_ =1



In terms of this expansion we can again use the triangle inequality to obtain


_L_ [2] _[ dν]_ [(] _[u]_ [))] [1] _[/]_ [2]

�ˆ _L_ [2] _[ ∥R ◦P]_ [(] _[u]_ [)] _[ −]_ _[R ◦]_ [�] _[P]_ [�][(] _[u]_ [)] _[∥]_ [2]



1 _/_ 2 (C.13)
_k_ _[, u][⟩][τ]_ _[k]_ _[−⟨][τ]_ [�] _[k]_ _[, u][⟩][τ]_ [�] _[k]_ _[∥]_ [2] _L_ [2] _[ dν]_ [(] _[u]_ [)]

�ˆ _L_ [2] _[ ∥⟨][τ]_ _[ ∗]_ �



_≤_



_p_
�


_k_ =1


ERROR ESTIMATES FOR DEEPONETS 87


Since

_∥⟨τ_ _k_ _[∗]_ _[, u][⟩][τ]_ _[k]_ _[−⟨][τ]_ [�] _[k]_ _[, u][⟩][τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[ ≤∥⟨][τ]_ _[ ∗]_ _k_ _[, u][⟩]_ [(] _[τ]_ _[k]_ _[−]_ _[τ]_ [�] _[k]_ [)] _[∥]_ _L_ [2] [ +] _[ ∥⟨][τ]_ _[ ∗]_ _k_ _[−]_ _[τ]_ [�] _[k]_ _[, u][⟩][τ]_ [�] _[k]_ _[∥]_ _L_ [2]


_≤∥τ_ _k_ _[∗]_ _[∥]_ _L_ [2] _[∥][u][∥]_ _L_ [2] _[∥][τ]_ _[k]_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] [ +] _[ ∥][τ]_ _[ ∗]_ _k_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[∥][u][∥]_ _L_ [2] _[∥][τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[,]_
(C.14)


we next wish to establish that under the assumptions of this lemma, we can bound


_∥τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[ ≤]_ _[C][√]_ ~~_[p]_~~ [max]
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[,]_


for some absolute constant _C >_ 0, independent of _k_ and _p_ . To see this, we note
that for any _k, j_ = 1 _, . . ., p_, we have _⟨τ_ _k_ _[∗]_ _[, τ]_ _[j]_ _[⟩]_ [=] _[ δ]_ _[kj]_ [, and hence]


_⟨_ ( _τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ [)] _[,]_ [ �] _[τ]_ _[j]_ _[⟩]_ [=] _[ ⟨][τ]_ _[ ∗]_ _k_ _[,]_ [ �] _[τ]_ _[j]_ _[⟩−]_ _[δ]_ _[kj]_ [=] _[ ⟨][τ]_ _[ ∗]_ _k_ _[,]_ [ �] _[τ]_ _[j]_ _[⟩−⟨][τ]_ _[ ∗]_ _k_ _[, τ]_ _[j]_ _[⟩]_ [=] _[ ⟨][τ]_ _[ ∗]_ _k_ _[,]_ [ (] _[τ]_ [�] _[j]_ _[−]_ _[τ]_ _[j]_ [)] _[⟩][.]_


If _u ∈_ _Y_ is arbitrary, then we obtain from the above identity



������



_|⟨_ ( _τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ [)] _[, u][⟩|]_ [ =]


=


_≤_



������

������









_p_
�



� _⟨τ_ � _j_ _, u⟩⟨_ ( _τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ [)] _[,]_ [ �] _[τ]_ _[j]_ _[⟩]_

_j_ =1



_p_
� _|⟨τ_ � _j_ _, u⟩|_ [2]

_j_ =1



_p_
�



_p_
�



������



� _⟨τ_ � _j_ _, u⟩⟨τ_ _k_ _[∗]_ _[,]_ [ �] _[τ]_ _[j]_ _[−]_ _[τ]_ _[j]_ _[⟩]_

_j_ =1



1 _/_ 2 []







1 _/_ 2









� _|⟨τ_ _k_ _[∗]_ _[,]_ [ �] _[τ]_ _[j]_ _[−]_ _[τ]_ _[j]_ _[⟩|]_ [2]

_j_ =1









_≤∥u∥_ _L_ 2 _∥τ_ _k_ _[∗]_ _[∥]_ _L_ [2] _[ √]_ ~~_[p]_~~ [max] (C.15)
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[.]_


By assumption, we have that

_√_ ~~_p_~~ max
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[ ≤]_ _[ϵ][ ≤]_ [1] 2 _[.]_


It follows that for any _k_ = 1 _, . . ., p_, we have


_∥τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] [ = sup] _|⟨τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[, u][⟩| ≤]_ _[ϵ][ ∥][τ]_ _[ ∗]_ _k_ _[∥]_ _L_ [2] _[ ≤]_ _[ϵ][ ∥][τ]_ _[ ∗]_ _k_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] [ +] _[ ϵ][ ∥][τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[,]_
_∥u∥≤_ 1


and hence
_ϵ_ _ϵ_
_∥τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[ ≤]_ 1 _−_ _ϵ_ _[∥][τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] [ =] 1 _−_ _ϵ_ _[≤]_ [2] _[ϵ.]_

Note also that this implies


_∥τ_ _k_ _[∗]_ _[∥]_ _L_ [2] _[ ≤∥][τ]_ _[ ∗]_ _k_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2] [ +] _[ ∥][τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[ ≤]_ [2] _[ϵ]_ [ + 1] _[ ≤]_ [2] _[.]_ (C.16)


It now follows from (C.14), that


_∥⟨τ_ _k_ _[∗]_ _[, u][⟩][τ]_ _[k]_ _[−⟨][τ]_ [�] _[k]_ _[, u][⟩][τ]_ [�] _[k]_ _[∥]_ _L_ [2] _[ ≤]_ [2] _[∥][u][∥]_ _L_ [2] ( _[√]_ ~~_p_~~ + 1) max
� _j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[.]_ �

_≤_ 4 _∥u∥_ _L_ 2 _[√]_ ~~_p_~~ max
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[.]_


Substitution in (C.13), finally yields


� � 1 _/_ 2
_E_ _R_ _≤_ _E_ _R_ � [+ 4] �ˆ _L_ [2] _[ ∥][u][∥]_ _L_ [2] [2] _[ dν]_ [(] _[u]_ [)] � _p_ [3] _[/]_ [2] _j_ =1 max _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[.]_


Letting

1 _/_ 2

_C_ := 4 _L_ [2] _[ dν]_ [(] _[u]_ [)] _,_
�ˆ _L_ [2] _[ ∥][u][∥]_ [2] �


we conclude that

� �
_E_ _R_ _≤_ _E_ _R_ � [+] _[ Cp]_ [3] _[/]_ [2] _j_ =1 max _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[ ≤]_ _[E]_ [�] _R_ [ �] [+] _[ Cϵ,]_


88 ERROR ESTIMATES FOR DEEPONETS


as claimed.
To prove the Lipschitz bound on _P_ : _L_ [2] ( _U_ ) _→_ R _[p]_, let _u, u_ _[′]_ _∈_ _L_ [2] ( _U_ ) be given,
and denote _w_ = _u −_ _u_ _[′]_ . Then



_∥P_ ( _u_ ) _−P_ ( _u_ _[′]_ ) _∥_ _ℓ_ 2 =


_≤_



_p_ 1 _/_ 2
� _|⟨τ_ _k_ _[∗]_ _[, w][⟩|]_ [2]
� _k_ =1 �


_p_ 1 _/_ 2
� _|⟨τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[, w][⟩|]_ [2]
� _k_ =1 �



+



_p_ 1 _/_ 2
� _|⟨τ_ � _k_ _, w⟩|_ [2]
� _k_ =1 �



By (C.15) and (C.16), we can bound each term in the first sum by


_|⟨τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[, w][⟩| ≤]_ [2] _[√]_ ~~_[p]_~~ [max]
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[ ∥][u][∥]_ _[L]_ [2] _[.]_



Hence,
_p_ 1 _/_ 2
� _|⟨τ_ _k_ _[∗]_ _[−]_ _[τ]_ [�] _[k]_ _[, w][⟩|]_ [2]
� _k_ =1 �



_≤_ 2 _p_ max
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] _[∥][u][∥]_ _[L]_ [2] _[.]_



As the � _τ_ _k_ are orthonormal by assumption, the second sum can be estimated simply
by
_p_ 1 _/_ 2


�

� _|⟨τ_ _k_ _, w⟩|_ [2] _≤∥w∥_ _L_ 2 _._
� _k_ =1 �


Recalling that _w_ = _u −_ _u_ _[′]_, it follows that


_∥P_ ( _u_ ) _−P_ ( _u_ _[′]_ ) _∥_ _ℓ_ 2 _≤_ 1 + 2 _p_ max _∥u −_ _u_ _[′]_ _∥_ _L_ 2 _._
� _j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ _[τ]_ [�] _[j]_ _[∥]_ _[L]_ [2] �

_≤_ (1 + 2 _ϵ_ ) _∥u −_ _u_ _[′]_ _∥_ _L_ 2 _≤_ 2 _∥u −_ _u_ _[′]_ _∥_ _L_ 2 _._


As _u, u_ _[′]_ _∈_ _L_ [2] ( _U_ ) were arbitrary, the claimed estimate for Lip( _P_ ) follows. Similarly,
we can estimate for the reconstruction:



_p_ 1 _/_ 2
� _|⟨τ_ � _k_ _, w⟩|_ [2]

_k_ =1 �



_≤∥w∥_ _L_ 2 _._



_∥R_ ( _α_ ) _−R_ ( _α_ _[′]_ ) _∥_ _L_ 2 =


_≤_



�����

�����



_p_
�



�( _α_ _k_ _−_ _α_ _k_ _[′]_ [)] _[τ]_ _[k]_


_k_ =1



_p_
�



�( _α_ _k_ _−_ _α_ _k_ _[′]_ [)] _[τ]_ [�] _[k]_


_k_ =1



����� _L_ [2]



����� _L_ [2]


+
����� _L_ [2]



�����



_p_
�



�( _α_ _k_ _−_ _α_ _k_ _[′]_ [)(] _[τ]_ _[k]_ _[−]_ _[τ]_ [�] _[k]_ [)]


_k_ =1



_≤∥α −_ _α_ _[′]_ _∥_ _ℓ_ 2 +



_p_
� _|α_ _k_ _−_ _α_ _k_ _[′]_ _[|∥][τ]_ _[k]_ _[−]_ _[τ]_ [�] _[k]_ _[∥]_ _L_ [2]


_k_ =1



_≤_ 1 + _p_ [1] _[/]_ [2] max _∥α −_ _α_ _[′]_ _∥_ _ℓ_ 2
� _k_ =1 _,...,p_ _[∥][τ]_ _[k]_ _[ −]_ _[τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] �

_≤_ (1 + _ϵ_ ) _∥α −_ _α_ _[′]_ _∥_ _ℓ_ 2 _≤_ 2 _∥α −_ _α_ _[′]_ _∥_ _ℓ_ 2 _._


Thus, Lip( _R_ ) _≤_ 2. 

C.7. **Proof of Lemma 3.17.**


_Proof._ We note that each element in the (real) trigonometric basis **e** 1 _, . . .,_ **e** _p_ can
be expressed in the form


**e** _j_ ( _x_ ) = cos( _κ · x_ ) _,_ or **e** _j_ ( _x_ ) = sin( _κ · x_ ) _,_


for _κ_ = _κ_ ( _j_ ) _∈_ Z _[n]_ with _|k|_ _∞_ _≤_ _N_, where _N_ is chosen as the smallest natural
number such that _p ≤_ (2 _N_ + 1) _[n]_ . It follows from


_|κ · x| ≤_ _n|κ|_ _∞_ _|x|_ _∞_ _≤_ 2 _πnN,_


ERROR ESTIMATES FOR DEEPONETS 89


that if _C_ _ϵ_ _, S_ _ϵ_ : R _→_ R are neural networks, such that


sup _|C_ _ϵ_ ( _ξ_ ) _−_ cos( _ξ_ ) _|, |S_ _ϵ_ ( _ξ_ ) _−_ sin( _ξ_ ) _| ≤_ _ϵ,_ (C.17)
_ξ∈_ [0 _,_ 2 _πnN_ ]


then the map


_x �→_ (( _κ_ ( _j_ ) _· x_ )) _j_ =1 _,...,p_ _�→_ ( _C_ _ϵ_ ( _κ_ ( _j_ ) _· x_ ) _, S_ _ϵ_ ( _κ_ ( _j_ ) _· x_ )) _j_ =1 _,...,p_


can be represented by a neural network _T_ with a size bounded by


size( _T_ ) = _O_ ( _p_ size( _C_ _ϵ_ ) + _p_ size( _S_ _ϵ_ )) _,_


depth( _N_ ) = _O_ (max (depth( _C_ _ϵ_ ) _,_ depth( _S_ _ϵ_ ))) _._


By (Elbr¨achter et al. 2021, Theorem III.9), there exist _C_ _ϵ_, _S_ _ϵ_, satisfying (C.17),
with
size( _C_ _ϵ_ ) _,_ size( _S_ _ϵ_ ) = _O_ (log( _ϵ_ _[−]_ [1] ) [2] + log( _N_ )) _,_

depth( _C_ _ϵ_ ) _,_ depth( _S_ _ϵ_ ) = _O_ (log( _ϵ_ _[−]_ [1] ) [2] + log( _N_ )) _._


Finally, we note that log( _N_ ) _∼_ 1 _/n_ log( _p_ ) = _O_ (log( _p_ )). We thus conclude that for
any _ϵ >_ 0, there exists a neural network _T_ _ϵ_ = ( _T_ _ϵ,_ 1 _, . . ., T_ _ϵ,p_ ), such that


_∥T_ _ϵ,j_ _−_ **e** _j_ _∥_ _L_ _∞_ ([0 _,_ 2 _π_ ] _n_ ) _≤_ _ϵ,_


for all _j_ = 1 _, . . ., p_, and


size( _T_ _ϵ_ ) = _O_ ( _p_ (log( _p_ ) + log( _ϵ_ _[−]_ [1] ) [2] ) _,_

depth( _T_ _ϵ_ ) = _O_ (log( _p_ ) + log( _ϵ_ _[−]_ [1] ) [2] ) _._


To satisfy the estimate (3.30), we set _**τ**_ = _T_ _ϵ/p_ 3 _/_ 2, for which we have


size( _**τ**_ ) = _O_ ( _p_ log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _,_

depth( _**τ**_ ) = _O_ (log( _ϵ_ _[−]_ [1] _p_ ) [2] ) _._


and

max
_j_ =1 _,...,p_ _[∥][τ]_ _[j]_ _[ −]_ **[e]** _[j]_ _[∥]_ _[L]_ _[∞]_ [([0] _[,]_ [2] _[π]_ []] _[n]_ [)] _[ ≤]_ _[ϵ.]_


                       

C.8. **Proof of Proposition 3.19.**


_Proof._ Proposition 3.19 follows form the following two claims:


_**Claim.**_ If _µ_ is a non-degenerate Gaussian, then there exists a Lipschitz continuous
mapping _P_ : _X →_ [0 _,_ 1], such that _P_ # _µ_ = _dx_ is the uniform measure on [0 _,_ 1].


_**Claim.**_ If _Y_ is an infinite-dimensional Hilbert space with orthonormal basis _{e_ _k_ _}_ _k∈_ N,
and _γ_ _k_ _≥_ 0, _k ∈_ N, a sequence, such that


_∞_
� _k_ [2] _γ_ _k_ _< ∞,_


_k_ =1


then there exists a Lipschitz continuous mapping _G_ : [0 _,_ 1] _→_ _Y_, such that the
covariance operator of _G_ # _dx_ is given by



Γ _G_ # _dx_ =



_∞_
� _γ_ _k_ ( _e_ _k_ _⊗_ _e_ _k_ ) _._


_k_ =1


90 ERROR ESTIMATES FOR DEEPONETS


The sought-after map _G_ can then be defined as _G_ = _G ◦_ _P_ : _X →_ _Y_, where
_P_ : _X →_ [0 _,_ 1] and _G_ : [0 _,_ 1] _→_ _Y_ are defined as in the above claims.
To prove the **first claim**, we simply note that since _µ_ is a non-degenerate Gaussian measure, there exists a one-dimensional projection _L_ : _X →_ R, such that
_L_ # _µ_ = _N_ ( _m, σ_ [2] ) is a non-degenerate Gaussian with mean _m ∈_ R, and variance
_σ_ [2] _>_ 0. Upon performing a translation by _µ_ and scaling by 1 _/σ_, we obtain an
affine mapping _L_ [�] : _X →_ R, such that _L_ [�] # _µ_ = _N_ (0 _,_ 1). We finally note that the
error function



1
erf( _x_ ) :=
~~_√_~~ 2 _π_



_x_

_e_ _[−][u]_ [2] _[/]_ [2] _du,_

ˆ _−∞_



maps the standard Gaussian distribution to the uniform distribution on [0 _,_ 1]. And
hence, we have (erf _◦_ _L_ [�] ) # _µ_ = _dx_ on [0 _,_ 1].
To prove the **second claim**, we show that the function



_G_ : [0 _,_ 1] _→_ _Y,_ _G_ ( _x_ ) :=



_∞_
�


_k_ =1



~~�~~ 2 _γ_ _k_ cos(2 _πkx_ ) _e_ _k_ _,_



possesses the desired properties: We first note that



_∞_
_∥∂_ _x_ _G_ ( _x_ ) _∥_ _Y_ [2] [= (2] _[π]_ [)] [2] �



_∞_ _∞_
� 2 _γ_ _k_ _k_ [2] _|_ sin(2 _πkx_ ) _|_ [2] _≤_ (2 _π_ ) [2] �


_k_ =1 _k_ =1



� 2 _γ_ _k_ _k_ [2] _<_ + _∞,_


_k_ =1



is uniformly bounded in _x_ by assumption on _γ_ _k_ . In particular, it follows that
_x �→_ _G_ ( _x_ ) is Lipschitz continuous. Furthermore, we have



cos(2 _πkx_ ) cos(2 _πk_ _[′]_ _x_ )( _e_ _k_ _⊗_ _e_ _k_ _′_ ) _dx_
0



_u ⊗_ _u d_ ( _G_ # _dx_ ) =

ˆ _Y_


=



_∞_ 1
� 2 _[√]_ ~~_γ_~~ _k_ ~~_γ_~~ _k_ _′_

_k,k_ _[′]_ =1 ˆ 0



_∞_
� _γ_ _k_ ( _e_ _k_ _⊗_ _e_ _k_ ) _._


_k_ =1



_∞_
�




                       

C.9. **Proof of Proposition 3.20.**


_Proof._ Let _v_ 1 _, . . ., v_ _p_ _∈_ _X_ denote pairwise orthogonal eigenvectors corresponding
to the first _p_ eigenvalues _λ_ 1 _, . . ., λ_ _p_ of the covariance operator Γ _µ_ . It follows from
Theorem 3.12, that the _p_ -dimensional affine subspace _V_ 0, given by _V_ 0 = _v_ 0 + _V_,
where _v_ 0 = E _µ_ [ _u_ ] and _V_ = span( _v_ 1 _, . . ., v_ _p_ ), satisfies



ˆ



� _λ_ _j_ _._

_j>p_



_X_ _v_ inf _∈V_ 0 _[∥][v][ −]_ _[u][∥]_ [2] _[ dµ]_ [(] _[u]_ [) =] �



Let _W_ 0 := _G_ ( _V_ 0 ). Since _G_ : _X →_ _Y_ is linear, _W_ 0 is an affine subspace of dimension
at most _p_ . If dim( _W_ 0 ) _< p_, we extend _W_ 0 to an affine subspace _W ⊂_ _Y_, such that
_W_ 0 _⊂_ _W_ and dim( _W_ ) = _p_, else set _W_ := _W_ 0 . Then, irrespective of the choice of


ERROR ESTIMATES FOR DEEPONETS 91



the extension, we have
ˆ _Y_ _w_ inf _∈W_ _[∥][w]_



inf _Y_ _[d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))]
_Y_ _w∈W_ 0 _[∥][w][ −]_ _[u][∥]_ [2]



_Y_ _w_ inf _∈W_ _[∥][w][ −]_ _[u][∥]_ _Y_ [2] _[d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))] _[ ≤]_ ˆ



= inf _Y_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_ _w∈W_ 0 _[∥][w][ −G]_ [(] _[u]_ [)] _[∥]_ [2]



= inf _Y_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_ _v∈V_ 0 _[∥G]_ [(] _[v]_ [)] _[ −G]_ [(] _[u]_ [)] _[∥]_ [2]



_≤_ inf _Y_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_ _v∈V_ 0 _[∥G∥]_ [2] _[∥][v][ −]_ _[u][∥]_ [2]



= _∥G∥_ [2] [ �] _λ_ _j_ _,_


_j>p_



as claimed. Finally, we note that if ( _R, P_ ) are chosen such that _R ◦P_ : _Y →_ _Y_ is
the orthogonal projection onto _W_, then



( _E_ [�] _R_ ) [2] =
ˆ



_∥R ◦P_ ( _u_ ) _−_ _u∥_ [2] _d_ ( _G_ # _µ_ ( _u_ )) =
_Y_ ˆ



_Y_ _w_ inf _∈W_ _[∥][w][ −]_ _[u][∥]_ [2] _[ d]_ [(] _[G]_ [#] _[µ]_ [(] _[u]_ [))] _[.]_




                       

C.10. **Proof of Theorem 3.8.**


_Proof._ Since _D_ : R _[m]_ _→_ _X_ is assumed to be linear, there exist _ψ_ 1 _, . . ., ψ_ _m_ _∈_ _X_,
such that



_D_ ( _u_ 1 _, . . ., u_ _m_ ) =



_m_
� _u_ _k_ _ψ_ _k_ _._


_k_ =1



Let _U_ := span( _ψ_ 1 _, . . ., ψ_ _m_ ) _⊂_ _X_ . Since _U_ is a subspace of _X_ of dimension at most
_m_, we have
ˆ _X_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ][ ≥]_ ˆ _X_ _u_ � inf _∈U_ _[∥][u][ −]_ _[u]_ [�] _[∥]_ _L_ [2] [2] _x_ _[dµ]_ [(] _[u]_ [)]



_∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ][ ≥]_
_X_ ˆ



_X_ _u_ � inf _∈U_ _[∥][u][ −]_ _[u]_ [�] _[∥]_ _L_ [2] [2] _x_ _[dµ]_ [(] _[u]_ [)]



_≥_ _U_ � inf _⊂X_ ;
dim( _U_ )= _m_



ˆ _X_ _u_ � inf _∈U_ [�] _∥u −_ _u_ � _∥_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)] _[.]_



By Theorem 3.10, the infimum on the last line is attained for the subspace _U_ [�]
spanned by the orthonormal eigenfunctions _φ_ 1 _, . . ., φ_ _m_ of the uncentered covariance
operator Γ _µ_ of _µ_ . For this space, we have


_m_


�

ˆ _X_ _u_ � inf _∈U_ [�] _∥u −_ _u∥_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [) =] ˆ _X_ _∥u∥_ [2] _dµ_ ( _u_ ) _−_ � ˆ _X_ _|⟨u, φ_ _k_ _⟩|_ [2] _dµ_ ( _u_ )



_X_ _u_ � inf _∈U_ [�] _∥u −_ _u_ � _∥_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [) =] ˆ



_∥u∥_ [2] _dµ_ ( _u_ ) _−_
_X_



_m_
�


_k_ =1



ˆ



_|⟨u, φ_ _k_ _⟩|_ [2] _dµ_ ( _u_ )
_X_



_m_
� _⟨φ_ _k_ _,_ Γ _φ_ _k_ _⟩_


_k_ =1



_m_
�



= Tr(Γ) _−_



=



_∞_
� _λ_ _k_ _−_


_k_ =1



_m_
� _λ_ _k_


_k_ =1



_m_
�



= � _λ_ _k_ _._

_k>m_



Thus, we conclude that
ˆ



� _λ_ _k_ _._

_k>m_



_X_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ][ ≥]_ �






92 ERROR ESTIMATES FOR DEEPONETS


C.11. **Proof of Lemma 3.22.**


_Proof._ Since _D ◦E_ is the identity on span( _φ_ 1 _, . . ., φ_ _m_ ), we have


_D ◦E −_ Id = _D ◦E ◦_ _P_ _m_ _[⊥]_ _[−]_ _[P]_ _[ ⊥]_ _m_ _[.]_


Furthermore, since Im( _D_ ) _⊂_ span( _φ_ 1 _, . . ., φ_ _m_ ), we also have _D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[ ⊥]_ _[P]_ _[ ⊥]_ _m_ _[u]_ [,]
for all _u ∈_ _L_ [2] _x_ [. Hence]


( _E_ [�] _E_ ) [2] = _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_


= _∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[ −]_ _[P]_ _[ ⊥]_ _m_ _[u][∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_



= _∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_

~~�~~ ~~��~~ ~~�~~
= ( _E_ [�] aliasing ) [2]



+ _∥P_ _m_ _[⊥]_ _[u][∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)]
ˆ _X_

~~�~~ � ~~�~~ ~~�~~
= ( _E_ [�] _⊥_ ) [2]



_._



The aliasing error


( _E_ [�] aliasing ) [2] = _∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)] _[,]_
ˆ _X_


can be re-written as follows [3] : Let _φ_ 1 _, φ_ 2 _, . . ._ denote the eigenfunctions of the covariance operator of _µ_, Γ _φ_ _k_ = _λ_ _k_ _φ_ _k_, where _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._ . First, we note that
there exist random variables _Z_ _ℓ_ (not necessarily iid), with


E[ _Z_ _ℓ_ _Z_ _ℓ_ _′_ ] = _δ_ _ℓℓ_ _′_ _,_


such that the random variable



_∞_
�


_ℓ_ =1



~~�~~ _λ_ _k_ _Z_ _k_ _φ_ _k_ _∼_ _µ_



is distributed according to _µ_ . Then, we have



= E








( _E_ [�] _⊥_ ) [2] = E









�����



_∞_
�


_ℓ_ =1



�



_λ_ _ℓ_ _Z_ _ℓ_ _P_ _m_ _[⊥]_ _[φ]_ _[ℓ]_



�����



2 []



� _λ_ _ℓ_ _Z_ _ℓ_ _φ_ _ℓ_



�����



2 []









�����



�

_ℓ>m_



=
�

_ℓ,ℓ_ _[′]_ _>m_



~~�~~ _λ_ _ℓ_ _λ_ _ℓ_ _′_ E [ _Z_ _ℓ_ _Z_ _ℓ_ _′_ ] _⟨φ_ _ℓ_ _, φ_ _ℓ_ _′_ _⟩_ = � �

_ℓ,ℓ_ _[′]_ _>m_



_λ_ _ℓ_ _λ_ _ℓ_ _′_ _δ_ _ℓ,ℓ_ _′_



3
we assume ~~´~~



= � _λ_ _ℓ_ _._

_ℓ>m_


_X_ _[u dµ]_ [(] _[u]_ [) = 0, but this is not essential here]


ERROR ESTIMATES FOR DEEPONETS 93


With Φ _M_ := ( _φ_ _i_ ( _X_ _j_ )) _∈_ R _[m][×][M]_ a matrix such that det(Φ _M_ Φ _[T]_ _M_ [)] _[ ̸]_ [= 0, the aliasing]
error is given by







_∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [) =][ E]

ˆ _X_









������



_m_
�



� _φ_ _k_


_k_ =1



_λ_ _ℓ_ _Z_ _ℓ_ _φ_ _ℓ_ ( _X_ _j_ )



������



�



2 []



_m_
�



�[Φ _[†]_ ] _kj_ �

_j_ =1 _ℓ>m_



_ℓ>m_



� [Φ _[†]_ ] _kj_ [Φ _[†]_ ] _kj_ _′_ _×_

_j,j_ _[′]_ =1



=


=



_m_ _m_
� _∥φ_ _k_ _∥_ [2] �

_k_ =1 _[′]_ =1



_m_
�



_m_
�



_m_ _m_
� _∥φ_ _k_ _∥_ [2] �

_k_ =1 _[′]_ =1



�

_ℓ,ℓ_ _[′]_ _>m_



~~�~~ _λ_ _ℓ_ _λ_ _ℓ_ _′_ E [ _Z_ _ℓ_ _Z_ _ℓ_ _′_ ] _φ_ _ℓ_ ( _X_ _j_ ) _φ_ _ℓ_ _′_ ( _X_ _j_ )



_′_

� [Φ _[†]_ ] _kj_ [Φ _[†]_ ] _kj_ �

_j,j_ _[′]_ =1 _ℓ>m_



� _λ_ _ℓ_ _φ_ _ℓ_ ( _X_ _j_ ) _φ_ _ℓ_ ( _X_ _j_ )

_ℓ>m_



� 2



�[Φ _[†]_ ] _kj_ _φ_ _ℓ_ ( _X_ _j_ )

_j_ =1



= � _λ_ _ℓ_

_ℓ>m_



=
�



_m_
�


_k_ =1



_m_
�
� _j_ =1



Denote _φ_ _ℓ_ ( _**X**_ ) := ( _φ_ _ℓ_ ( _X_ 1 ) _, . . ., φ_ _ℓ_ ( _X_ _m_ )). Then we can write the last line equivalently in the form



( _E_ [�] aliasing ) [2] =
ˆ



� _λ_ _ℓ_ _∥_ Φ _[†]_ _φ_ _ℓ_ ( _**X**_ ) _∥_ [2] _ℓ_ [2] _[,]_

_ℓ>m_



_X_ _∥D ◦E_ ( _P_ _m_ _[⊥]_ _[u]_ [)] _[∥]_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [) =] �



as claimed. 

C.12. **Proof of Lemma 3.23.**


_Proof._ The minimal singular value of _A_ _M_ = _[|]_ _M_ _[D][|]_ [Φ] _[M]_ [Φ] _M_ _[T]_ [is given by]


_σ_ min ( _A_ _M_ ) = inf
_∥v∥_ _ℓ_ 2 =1 _[⟨][v, A]_ _[M]_ _[v][⟩]_


= inf
_∥v∥_ _ℓ_ 2 =1 [[] _[⟨][v,]_ **[ 1]** _[v][⟩−⟨][v,]_ [ (] **[1]** _[ −]_ _[A]_ _[M]_ [)] _[v][⟩]_ []]

= 1 _−_ sup _⟨v,_ ( **1** _−_ _A_ _M_ ) _v⟩_
_∥v∥_ _ℓ_ 2 =1

= 1 _−_ _σ_ max ( **1** _−_ _A_ _M_ ) _._


The maximal singular value (=spectral norm) of **1** _−_ _A_ _M_ _∈_ R _[m][×][m]_ can be estimated
from above by
_σ_ max ( **1** _−_ _A_ _M_ ) _≤∥_ **1** _−_ _A_ _M_ _∥_ F _,_


where we denote, for any matrix _A_ = ( _a_ _ij_ ), by _∥A∥_ F the Frobenius norm,







1 _/_ 2



_m_
�
 _i,j_ =1









_∥A∥_ F =



� _|a_ _ij_ _|_ [2]

_i,j_ =1



_._



Note that for all _i, j ∈{_ 1 _, . . ., m}_, we have that the ( _i, j_ ) entry of **1** _−_ _A_ _M_ is given
by



_m_
� _φ_ _i_ ( _X_ _k_ ) _φ_ _j_ ( _X_ _k_ ) _._


_k_ =1



_δ_ _ij_ _−_ _[|][D][|]_

_M_



_δ_ _ij_ _−_ _[|][D][|]_



With _Y_ _k_ = _−|D|φ_ _i_ ( _X_ _k_ ) _φ_ _j_ ( _X_ _k_ ), this can be written in the form



1

_M_



_m_
� _Y_ _k_ _−_ E[ _Y_ _k_ ] _,_


_k_ =1


94 ERROR ESTIMATES FOR DEEPONETS


where the _Y_ _k_ are iid random variables bounded by _|D|ω_ _m_ [2] [. It follows from Hoeffd-]
ing’s inequality that for any _δ >_ 0, we have



_≥_ _δ_
�����



�



_≤_ 2 exp _−_ [2] _[M]_ [ 2] _[δ]_ [2] _._
� _|D|_ [2] _ω_ _m_ [4] �



Prob



1

_M_
������



_m_
�



� _Y_ _k_ _−_ E[ _Y_ _k_ ]


_k_ =1



Equivalently, for any _i, j ∈{_ 1 _, . . ., m}_ we have:

Prob � _|_ [ **1** _−_ _A_ _M_ ] _ij_ _| ≥_ _δ_ � _≤_ 2 exp � _−_ _|_ [2] _D_ _[M]_ _|_ [2][ 2] _ω_ _[δ]_ _m_ [4][2]


It now follows that



_._
�









Prob [ _∥_ **1** _−_ _A_ _M_ _∥_ _F_ _≥_ _δ_ ] = Prob







_m_
�
 _i,j_ =1



� _|_ ( **1** _−_ _A_ _M_ ) _ij_ _|_ [2] _≥_ _δ_ [2]

_i,j_ =1



_≤_ Prob max
� _i,j_ =1 _,...,m_ _[|]_ [(] **[1]** _[ −]_ _[A]_ _[M]_ [)] _[ij]_ _[| ≥]_ _m_ _[δ]_



�



_≤_ _m_ [2] max _|_ ( **1** _−_ _A_ _M_ ) _ij_ _| ≥_ _[δ]_
_i,j_ =1 _,...,m_ [Prob] � _m_



_≤_ _m_ [2] max _|_ ( **1** _−_ _A_ _M_ ) _ij_ _| ≥_ _[δ]_
_i,j_ =1 _,...,m_ [Prob] � _m_



� 2 [�]



_≤_ 2 _m_ [2] exp



_Mδ_
_−_ 2

� � _|D|ω_ _m_ [2] _m_



Choosing _δ_ = 1 _/√_



2, we have



�


_,_



Prob _σ_ max ( **1** _−_ _A_ _M_ ) _≥_ 1 _/√_
�



2 _≤_ 2 _m_ [2] exp
�



_M_

� _−_ � _|D|ω_ _m_ [2] _m_



�



and thus, from _σ_ min ( _A_ _M_ ) = 1 _−_ _σ_ max ( **1** _−_ _A_ _M_ ), also



� 2 [�]


� 2 [�]



Prob _σ_ min ( _A_ _M_ ) _<_ 1 _−_ 1 _/√_
�


C.13. **Proof of Theorem 3.9.**


_Proof._ Let



2 _≤_ 2 _m_ [2] exp
�



_M_

� _−_ � _|D|ω_ _m_ [2] _m_



_._







2 max( _|D|,_ 1)

sup(1 + _∥φ_ _ℓ_ _∥_ _L_ [2] _[∞]_ [)] _[.]_

~~_√_~~ 2 _−_ 1 _ℓ∈_ N



_C_ =



_√_



By assumption, we have _C < ∞_, and we also note that _C ≥_ 1. Let (Ω _, P_ ) be
the probability space Ω= [�] _[∞]_ _ℓ_ =1 _[D]_ [, with probability measure] _[ P]_ [ =][ �] _[∞]_ _ℓ_ =1 [Unif(] _[D]_ [),]
such that the iid random variables _X_ 1 _, X_ 2 _, . . ._ are given by projection onto the
corresponding factor
_X_ _ℓ_ : Ω _→_ _D,_ _X_ _ℓ_ ( _ω_ ) = _ω_ _ℓ_ _,_
where _ω_ = ( _ω_ 1 _, ω_ 2 _, . . .,_ ) _∈_ Ω.
By Lemma 3.24, if we choose _M_ ( _m_ ) = _⌈Cκm_ log( _m_ ) _⌉_, with _κ_ = 4, then we have

Prob �� _ω_ ���� _E_ _E_ ( _X_ 1 ( _ω_ ) _, . . ., X_ _M_ ( _ω_ )) _≤_ _C_ � _ℓ>m_ _[λ]_ _[ℓ]_ �� _≥_ 1 _−_ 2 _m_ _[−]_ [2] _._ (C.18)

where _E_ [�] _E_ = _E_ [�] _E_ ( _X_ 1 _, . . ., X_ _M_ ) is the random encoding error based on the sensors
_X_ 1 _, . . ., X_ _M_ . We note that asymptotically as _m →∞_, we have


_M ∼_ _Cκm_ log( _m_ ) ≳ _Cκm,_


and hence log( _M_ ) ≳ log( _Cκ_ ) + log( _m_ ) _≥_ log( _m_ ), where we used that _C >_ 1 and
_κ_ = 4, so that log( _Cκ_ ) _≥_ 0 in the last estimate. In particular, this implies that


_M/_ log( _M_ ) ≲ [ _Cκm_ log( _m_ )] _/_ log( _m_ ) = _Cκm._


ERROR ESTIMATES FOR DEEPONETS 95



It follows that, by possibly enlarging the constant _C >_ 1, we have
� _λ_ _ℓ_ _≥_ � _λ_ _ℓ_ _,_



� _λ_ _ℓ_ _≥_ �

_ℓ>M/C_ log( _M_ ) _ℓ>m_



� _λ_ _ℓ_ _,_

_ℓ>m_



and using also (C.18), we conclude that for sufficiently large _C_, we have

Prob �� _ω_ ���� _E_ _E_ ( _X_ 1 _, . . ., X_ _M_ ) _≤_ _C_ � _ℓ>M/C_ log( _M_ ) _[λ]_ _[ℓ]_ �� _≥_ 1 _−_ _C_ [lo][g(] _M_ _[M]_ [2] [)] [2] _,_


for all _M ∈_ N. In particular, the probability that


�
_E_ _E_ ( _X_ 1 _, . . ., X_ _M_ ) _> C_ � _λ_ _ℓ_ _,_

_ℓ>M/C_ log( _M_ )


for infinitely many _M_, can be bounded by


�
Prob� _E_ _E_ ( _X_ 1 _, . . ., X_ _M_ ) _> C_ [�] _ℓ>M/C_ log( _M_ ) _[λ]_ _[ℓ]_ [infinitely often] �






 _[ω]_








_≤_ lim sup Prob
_M_ 0 _→∞_







 _M>M_ [�]



_M>M_ 0














 _[ω]_



_≤_ lim sup
_M_ 0 _→∞_



� Prob

_M>M_ 0



�
_E_ _E_ ( _X_ 1 ( _ω_ ) _, . . ., X_ _M_ ( _ω_ )) _> C_ � _λ_ _ℓ_
������ _ℓ>M/C_ log( _M_ )


�
_E_ _E_ ( _X_ 1 ( _ω_ ) _, . . ., X_ _M_ ( _ω_ )) _> C_ � _λ_ _ℓ_
������ _ℓ>M/C_ log( _M_ )















log( _M_ ) [2]

_M_ [2]



_≤_ _C_ lim sup
_M_ 0 _→∞_


= 0 _._



�

_M>M_ 0



Thus, for almost all _ω ∈_ Ω, we have


�
_E_ _E_ ( _X_ 1 ( _ω_ ) _, . . ., X_ _M_ ( _ω_ )) _≤_ _C_ � _λ_ _ℓ_ _,_

_ℓ>M/C_ log( _M_ )


for all sufficiently large _M >_ 0. 

C.14. **Proof of Lemma 3.25.**


_Proof._ The starting point is the claim that if the covariance operator Γ be given by
(3.45), the eigenfunctions and eigenvalues ( _φ_ _k_ _, λ_ _k_ ) of Γ are given by



_φ_ _k_ ( _x_ ) = _e_ _[−][ikx]_ _,_ _λ_ _k_ = _√_



To see this, we note that

2 _π_

_k_ _p_ ( _x, x_ _[′]_ ) _φ_ _k_ ( _x_

ˆ 0



_h∈_ 2 _π_ Z



_k_ _p_ ( _x, x_ _[′]_ ) _φ_ _k_ ( _x_ _[′]_ ) _dx_ _[′]_ = �
0



2 _πℓe_ _[−]_ [(] _[ℓk]_ [)] [2] _[/]_ [2] _,_ ( _k ∈_ Z) _._ (C.19)


2 _π_

_e_ _[−]_ [(] _[x][−][x]_ _[′]_ _[−][h]_ [)] [2] _[/]_ [(2] _[ℓ]_ [2] [)] _e_ _[−][ikx]_ _[′]_ _dx_ _[′]_

ˆ 0



_∞_
= _e_ _[−]_ [(] _[x][−][x]_ _[′]_ [)] [2] _[/]_ [(2] _[ℓ]_ [2] [)] _e_ _[−][ikx]_ _[′]_ _dx_ _[′]_
ˆ _−∞_



= _√_



2 _πℓe_ _[−]_ [(] _[ℓk]_ [)] [2] _e_ _[−][ikx]_ _,_



where we used that the Fourier transform on the real line R


_∞_
_F_ [ _u_ ]( _k_ ) = _u_ ( _x_ _[′]_ ) _e_ _[−][ikx]_ _[′]_ _dx_ _[′]_ _,_
ˆ _−∞_


satisfies
_F_ [ _u_ ( _· −_ _x_ )][ _k_ ] = _F_ [ _u_ ]( _k_ ) _e_ _[−][ikx]_ _,_

and we recall that for _α >_ 0, the Fourier transform of a Gaussian is


_π_
_F_ [exp( _−αx_ [2] )]( _k_ ) = � _α_ [exp(] _[−][k]_ [2] _[/]_ [4] _[α]_ [)] _[.]_


96 ERROR ESTIMATES FOR DEEPONETS


For simplicity, assume _m_ = 2 _K_ + 1 for _K ∈_ N. Recall that the decoder _D_ for
encoder _E_ is the discrete Fourier transform (3.46), it is straightforward to check that
_E ◦D_ = Id, so ( _E, D_ ) is an admissible encoder/decoder pair, and by the definition
(3.1) of _E_ [�] _E_ :


( _E_ [�] _E_ ) [2] _≤_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ]_ [(] _[u]_ [)] _[.]_
ˆ _X_


Let _P_ _K_ : _L_ [2] _x_ _[→]_ _[L]_ [2] _x_ [denote the orthogonal projection onto span(] _[e]_ _[ikx]_ [;] _[ |][k][| ≤]_ _[K]_ [),]
and denote by _P_ _K_ _[⊥]_ [:] _[ L]_ _x_ [2] _[→]_ _[L]_ [2] _x_ [the orthogonal projection onto the orthogonal com-]
plement, so that Id = _P_ _K_ + _P_ _K_ _[⊥]_ [.] We note that _D ◦E_ is _linear_ and we have
( _D ◦E_ )( _e_ _[ikx]_ ) = _e_ _[ikx]_ for all _|k| ≤_ _K_ (where _m_ = 2 _K_ + 1). From this, it follows that


_D ◦E −_ Id = [ _D ◦E ◦_ _P_ _K_ _−_ _P_ _K_ ] + � _D ◦E ◦_ _P_ _K_ _[⊥]_ _[−]_ _[P]_ _[ ⊥]_ _K_ �

= _D ◦E ◦_ _P_ _K_ _[⊥]_ _[−]_ _[P]_ _[ ⊥]_ _K_
= _P_ _K_ _◦D ◦E ◦_ _P_ _K_ _[⊥]_ _[−]_ _[P]_ _[ ⊥]_ _K_ _[,]_


where we used that _D_ = _P_ _K_ _◦D_ . The two terms on the last line are obviously
perpendicular to each other. Hence


_∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ [=] _[ ∥D ◦E ◦]_ _[P]_ _K_ _[ ⊥]_ _[∥]_ [2] _L_ [2] _x_ [+] _[ ∥][P]_ _K_ _[ ⊥]_ _[∥]_ [2] _L_ [2] _x_ _[.]_ (C.20)


We note that the non-vanishing of the first term in C.20 is due to _aliasing_, i.e. the
fact that for any point on the grid _x_ _j_, _j_ = 1 _, . . ., m_, we have


_e_ _[ikx]_ _[j]_ = _e_ _[i]_ [(] _[k]_ [+] _[qm]_ [)] _[x]_ _[j]_ _,_ for all _q ∈_ Z _._


Therefore, the higher-order modes _e_ _[ikx]_ for _|k| > K_ map under _E_ to


_E_ ( _e_ _[ikx]_ ) = _E_ ( _e_ _[ik]_ [0] _[x]_ ) _,_


where _k_ 0 _∈{−K, . . ., K}_ is the unique value such that there exists _q ∈_ Z with
_k_ 0 = _k_ + _qm_ .
As the functions _x �→_ _e_ _[ikx]_, _k ∈_ Z, are the eigenfunctions of the covariance
operator of _µ_ . Let _λ_ _k_, _k ∈_ Z, denote the corresponding eigenvalues. By the
Karhunen-Loeve expansion for the Gaussian measure _µ_, we can now write



ˆ _X_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ]_ [ =][ E] � _∥D ◦E_ ( _X_ ) _−_ _X∥_ [2] _L_ [2] _x_ �



= E � _∥D ◦E_ � _P_ _K_ _[⊥]_ _[X]_ � _∥_ [2] _L_ [2] _x_ � + E � _∥P_ _K_ _[⊥]_ [(] _[X]_ [)] _[∥]_ [2] _L_ [2] _x_



_,_
�



where



_X_ =
�

_k∈_ Z



�



_λ_ _k_ _X_ _k_ _e_ _[ikx]_ _,_



and the _X_ _k_ _∼N_ (0 _,_ 1) are iid Gaussian random variables with unit variance. Due
to aliasing, we have



�



_λ_ _k_ 0 + _qm_ _X_ _k_ 0 + _qm_







 _q∈_ [�] Z _\{_



_q∈_ Z _\{_ 0 _}_



_D ◦E_ � _P_ _K_ _[⊥]_ _[X]_ � =



_K_
�

_k_ 0 = _−K_



 _e_ _[ik]_ [0] _[x]_ _._




ERROR ESTIMATES FOR DEEPONETS 97



And


E � _∥D ◦E_ � _P_ _K_ _[⊥]_ _[X]_ � _∥_ [2] _L_ [2] _x_



= E
�









�



_λ_ _k_ 0 + _qm_ _X_ _k_ 0 + _qm_



2 []










 _q∈_ [�] Z _\{_



_K_
�













_k_ 0 = _−K_



_q∈_ Z _\{_ 0 _}_



~~�~~ _λ_ _k_ 0 + _qm_ ~~�~~



~~�~~



_λ_ _k_ 0 + _q_ _′_ _m_ _X_ _k_ 0 + _qm_ _X_ _k_ 0 + _q_ _′_ _m_







�
 _q,q_ _[′]_ _∈_ Z



_q,q_ _[′]_ _∈_ Z _\{_ 0 _}_









=


=



_K_
� E

_k_ 0 = _−K_



_K_
�

_k_ 0 = _−K_



_K_
�



�

_q,q_ _[′]_ _∈_ Z _\{_ 0 _}_



~~�~~ _λ_ _k_ 0 + _qm_ ~~�~~



~~�~~



_λ_ _k_ 0 + _q_ _′_ _m_ E [ _X_ _k_ 0 + _qm_ _X_ _k_ 0 + _q_ _′_ _m_ ] _._



As the _X_ _k_ are iid with zero mean and unit variance, we have



E[ _X_ _k_ _X_ _k_ _′_ ] = _δ_ _k,k_ _′_ =



1 _,_ _k_ = _k_ _[′]_
�0 _,_ _k ̸_ = _k_ _[′]_ _[ .]_



Thus,



� _λ_ _k_ 0 + _qm_

_q∈_ Z _\{_ 0 _}_



E � _∥D ◦E_ � _P_ _K_ _[⊥]_ _[X]_ � _∥_ [2] _L_ [2] _x_ � =


=



_K_
�

_k_ 0 = _−K_


_K_
�

_k_ 0 = _−K_



�

_q,q_ _[′]_ _∈_ Z _\{_ 0 _}_



~~�~~ _λ_ _k_ 0 + _qm_ ~~�~~



~~�~~



_λ_ _k_ 0 + _q_ _′_ _m_ _δ_ _q,q_ _′_



= � _λ_ _k_ _._

_|k|>K_



On the other hand, it is easy to see that

E � _∥P_ _K_ _[⊥]_ _[X][∥]_ [2] _L_ [2] _x_



� = � _λ_ _k_ _._

_|k|>K_



=
� �



We thus conclude that
ˆ _X_ _∥D ◦E −_ Id _∥_ [2] _L_ [2] _x_ _[dµ]_ [ =][ E] � _∥D ◦E_ � _P_ _K_ _[⊥]_ _[X]_ � _∥_ [2] _L_ [2] _x_

= 2 � _λ_ _k_ _._

_|k|>K_


Recalling that by (C.19), we have



� + E � _∥P_ _K_ _[⊥]_ [(] _[X]_ [)] _[∥]_ [2] _L_ [2] _x_



�



_λ_ _k_ = _√_ 2 _πℓe_ _[−]_ [(] _[ℓk]_ [)] [2] _[/]_ [2] _,_



we finally obtain



2 � _λ_ _k_ = 2 _√_

_|k|>K_



2 �



2 _π_ � _ℓe_ _[−]_ [(] _[ℓk]_ [)] [2] _[/]_ [2]

_|k|>K_



� _ℓe_ _[−]_ [(] _[ℓk]_ [)] [2] _[/]_ [2]

_|k|>K_



_≤_ 2 _π√_ 2 [2]

~~_√π_~~



_∞_

_e_ _[−]_ [(] _[ℓx]_ [)] [2] _[/]_ [2] _ℓdx_

ˆ _K_



2
= 4 _π_
� ~~_√π_~~



ˆ _K ∞_



2) [2] _d_ _ℓx/√_
�



~~_√_~~
_e_ _[−]_ [(] _[ℓx/]_

_K_



2
� [�]



_ℓK_
= 4 _π_ erfc _._
� ~~_√_~~ 2 �






98 ERROR ESTIMATES FOR DEEPONETS


C.15. **Proof of Lemma 3.26.**


_Proof._ By Lemma 3.25, the eigenfunctions of the covariance operator are _φ_ _k_ =
**e** _k_, the standard Fourier basis. By Theorem 3.9, there exists a constant _C ≥_ 1,
depending on _|D|_ and sup _k∈_ N _∥φ_ _k_ _∥_ _L_ _∞_ _≤_ 2(2 _π_ ) _[−][d]_, such that with probability 1 in
the random sensors _X_ 1 _, X_ 2 _, X_ 3 _, · · · ∈_ _D_, we have for almost all _M ∈_ N:


�
_E_ _E_ _≤_ _C_ ~~�~~ _λ_ _k_ _._
� _k>M/C_ log( _M_ )



By Lemma 3.25 (cp. (3.47), (3.48)), we have


�

~~�~~ _k>M/C_ ~~�~~ log( _M_ ) _λ_ _k_ ≲ exp � _−γ_ _C_



� _M_ [2] _ℓ_ [2]
_k>M/C_ ~~�~~ log( _M_ ) _λ_ _k_ ≲ exp � _−γ_ _C_ [2] log(



_C_ [2] log( _M_ ) [2]



�
_,_ _∀γ <_ [1]
� 16 _[.]_



The implied constant here only depends on the value of � _γ_ and on _|D|_ . Thus,
choosing e.g. � _γ_ = 1 _/_ 20 and noting that _|D|_ = 2 _π_ is a fixed constant, we conclude
that



�
_E_ _E_ _≤_ _C_ exp _−γ_ _[M]_ [ 2] _[ℓ]_ [2]
� log( _M_ ) [2]



_,_
�



for _γ_ = 1 _/_ (20 _C_ [2] ), where the constant _C >_ 0 is independent of _ℓ_ and _M_ . 

C.16. **Proof of Proposition 3.27.**


_Proof._ Fix _u_ = _u_ ( _·_ ; _Y_ ) for some _Y ∈_ [ _−_ 1 _,_ 1] _[J]_ . Let _Y_ [�] = _Y_ [�] ( _E_ ( _u_ )) be given by (3.59).
Let _Y_ [�] be given by (3.57), such that


� �
_Y_ _j_ = shrink( _Y_ _j_ ) _,_ _∀_ _j ∈J_ = Z _[d]_ _._


We note that since _Y_ _j_ _∈_ [ _−_ 1 _,_ 1] for all _j ∈J_, we have

� � �
_Y_ _j_ _−_ _Y_ _j_ = _Y_ _j_ _−_ shrink( _Y_ _j_ ) _≤_ _Y_ _j_ _−_ _Y_ _j_ _,_ _∀_ _j ∈J_ = Z _[d]_ _._
��� ��� ��� ��� ��� ���


But, by definition of _Y_ [�] _j_, we have that



_u �→_ _u_ ( _·_ ; _Y_ [�] ( _E_ ( _u_ ))) = �



� _Y_ � _j_ ( _E_ ( _u_ )) _α_ _j_ **e** _j_ = �

_j∈_ Z _[d]_ _j∈_ Z



� _u_ � _j_ **e** _j_ _,_

_j∈_ Z _[d]_



is the pseudo-spectral Fourier projection of _u_ onto _{_ **e** _j_ _}_ _j∈K_ _N_ . In particular, it
follows from standard estimates for the pseudo-spectral projection that for _u ∈_
_H_ _[s]_ (T _[d]_ ), with _s > n/_ 2, we have

_u_ ( _·_ ; _Y_ ) _−_ _u_ ( _·_ ; � _Y_ ( _E_ ( _u_ ))
��� ��� _L_ [2] _[ ≤]_ _[C][∥][u][∥]_ _[H]_ _[s]_ _[ N]_ _[ −][s]_ _[,]_



for some _C_ = _C_ ( _s_ ) _>_ 0, and hence also

2
��� _u_ ( _·_ ; _Y_ ) _−_ _u_ ( _·_ ; � _Y_ )��� _L_ [2] [ =] �

_j∈_ Z _[d]_


_≤_
�

_j∈_ Z _[d]_



� 2
��� _Y_ _j_ _−_ _Y_ _j_ ��� _α_ _j_ [2]


��� _Y_ _j_ _−_ _Y_ � _j_ ��� 2 _α_ _j_ [2]



2
= _u_ ( _·_ ; _Y_ ) _−_ _u_ ( _·_ ; � _Y_
��� ��� _L_ [2]

_≤_ _C∥u∥_ _H_ _s_ _N_ _[−][s]_ _._






ERROR ESTIMATES FOR DEEPONETS 99


C.17. **Proof of Theorem 3.28.**


_Proof._ We note that since _m_ = (2 _N_ + 1) _[d]_, we have _N_ ≳ _m_ [1] _[/d]_ . Thus, it suffices
to show that _E_ [�] _E_ _≤_ _C_ exp( _−cN_ ) for some constants _c, C_ independent of _N_ . We
now note that if _Y_ = ( _Y_ _j_ ) _j∈J_ are distributed according to the probability measure
_ρ ∈P_ ([ _−_ 1 _,_ 1] _[J]_ ), and if _Y_ [�] = _Y_ [�] ( _E_ ( _u_ ) and _Y_ [�] = _Y_ [�] ( _E_ ( _u_ )) are given by (3.59), (3.57),
respectively, then


( _E_ [�] _E_ ) [2] = _∥D ◦E_ ( _u_ ) _−_ _u∥_ [2] _L_ [2] _[ dµ]_ [(] _[u]_ [)]
ˆ _L_ [2] ( _D_ )


= _∥u_ ( _·_ ; _Y_ [�] ) _−_ _u_ ( _·_ ; _Y_ ) _∥_ [2] _L_ [2] _[ dρ]_ [(] _[Y]_ [ )]
ˆ _L_ [2] ( _D_ )


As in the proof of Proposition 3.27 in appendix C.16, we see that – due to the
exponential decay ≲ exp( _−|k|ℓ_ ) of the Fourier coefficients of _u_ ( _·_ ; _Y_ ) – there exist
_C, c >_ 0, independent of _N_ and _Y ∈_ [ _−_ 1 _,_ 1] _[J]_, such that the last term can be
estimated by


_≤_ [ _C_ exp( _−cℓN_ )] [2] _dρ_ ( _Y_ )
ˆ _L_ [2] ( _D_ )

= [ _C_ exp( _−cℓN_ )] [2] _._


This shows that _E_ [�] _E_ _≤_ _C_ exp( _−cℓN_ ). And we recall that _N_ ≲ _m_ [1] _[/d]_, by definition
of _m_ = (2 _N_ + 1) _[d]_ .
Finally, if
_F_ ( _**y**_ ) = _G_ ( _u_ ( _·_ ; _**y**_ )) _,_

then for any _**u**_ = ( _u_ _i_ ) _i∈I_ _N_, we have by the definition of _D_ ( _**u**_ ) = _u_ ( _·_ ; _Y_ [�] ( _**u**_ )) (cp.
(3.58))

_G ◦D_ ( _**u**_ ) = _G_ _u_ ( _·,_ _Y_ [�] ( _**u**_ )) = _F_ ( _Y_ [�] ( _**u**_ )) _._
� �


                       

C.18. **Proof of Corollary 3.34.**


_Proof._ Let _N_ [�] be a neural network satisfying the estimates of Proposition 3.33, with
_N_ := _m_ . Then



_N_ := _P ◦_ _N_ [�] = � (� _P_ ~~���~~ _c_ _**ν**_ )

_**ν**_ _∈_ Λ _N_ _∈_ R _[p]_



_N_ � _**ν**_ ( _y_ _κ_ (1) _, . . ., y_ _κ_ ( _N_ ) )



is a linear combination of the neural network output of _N_ [�], with coefficients in R _[p]_ .
In particular, by adding a linear output layer of size _O_ ( _pN_ ) to the network _N_ [�], we
can represent _N_ as a neural network with


size( _N_ ) _≤_ size( _N_ [�] ) + _pN,_ depth( _N_ ) _≤_ depth( _N_ [�] ) + 1 _._


The claimed bounds on the size of _N_ thus readily follow from the corresponding
bounds for _N_ [�] . Furthermore, we have for any _**y**_ _∈_ [ _−_ 1 _,_ 1] _[J]_ :

�
�� _P ◦F_ ( _**y**_ ) _−N_ ( _y_ _κ_ (1) _, . . ., y_ _κ_ ( _N_ ) �� _ℓ_ [2] [ =] ��� _P ◦F_ ( _**y**_ ) _−P ◦_ _N_ ( _y_ _κ_ (1) _, . . ., y_ _κ_ ( _N_ ) ��� _ℓ_ [2]


�
_≤∥P∥_ _F_ ( _**y**_ ) _−_ _N_ ( _y_ _κ_ (1) _, . . ., y_ _κ_ ( _N_ )
��� ��� _V_ _[.]_


The claimed error estimate thus follows from the error estimate in Proposition
3.33. 

100 ERROR ESTIMATES FOR DEEPONETS


C.19. **Proof of Theorem 3.35.**


_Proof._ The proof of this theorem requires the following simple Lemma on Neural
Network Calculus,


_**Lemma**_ **C.10** _**.**_ Let _N_ 1 : R _[n]_ [0] _→_ R _[n]_ [1], _N_ 2 : R _[n]_ [1] _→_ R _[n]_ [2] be two neural networks.
Then the composition _N_ = _N_ 2 _◦N_ 1 : R _[n]_ [0] _→_ R _[n]_ [2] can be represented by a neural
network _N_ with


size( _N_ ) = size( _N_ 1 ) + size( _N_ 2 ) _,_ depth( _N_ ) = depth( _N_ 1 ) + depth( _N_ 2 ) _._


C.19.1. _Proof of the Theorem 3.35._ From the network size bounds of Corollary
3.34, _N_ is a neural network of size


size( _N_ ) _≤_ _C_ (1 + _mp_ log( _m_ ) log log( _m_ ))) _,_


depth( _N_ ) _≤_ _C_ (1 + log( _m_ ) log log( _m_ ))) _._


with a constant _C >_ 0, independent of _m_, _p_ . By Lemma 3.29, the map _**u**_ _�→_ _Y_ [�] _κ_ ( _**u**_ )
can be represented by a neural network of size


size( _Y_ [�] _κ_ ) _≤_ _C_ (1 + _m_ log( _m_ )) _,_ depth( _Y_ [�] _κ_ ) _≤_ _C_ (1 + log( _m_ )) _,_


for a constant _C >_ 0, independent of _m_ . Since the definition of _Y_ [�] _κ_ does not involve
the projection _P_ at all, the constant is also independent of _p_ . By the composition
Lemma C.10 it follows that _**u**_ _�→A_ ( _**u**_ ) = ( _N ◦_ _Y_ [�] _κ_ )( _**u**_ ) can be represented by a
neural network with


size( _A_ ) = size( _N_ ) + size( _Y_ [�] _κ_ ) _,_ depth( _A_ ) = depth( _N_ ) + depth( _Y_ [�] _κ_ ) _._


We also note that the following estimate for the approximation error _E_ [�] _A_ :


( _E_ [�] _A_ ) [2] = _∥P ◦G ◦D_ ( _**u**_ ) _−A_ ( _**u**_ ) _∥_ [2] _ℓ_ [2] _[ d]_ [(] _[E]_ [#] _[µ]_ [)(] _**[u]**_ [)]
ˆ _L_ [2] ( _D_ )

_≤_ sup _∥P ◦G ◦D_ ( _**u**_ ) _−A_ ( _**u**_ ) _∥_ [2] _ℓ_ [2]
_**u**_ _∈_ supp( _µ_ )

= sup _∥P ◦F_ ( _Y_ [�] ( _**u**_ )) _−N_ ( _Y_ [�] ( _**u**_ )) _∥_ [2] _ℓ_ [2]
_**u**_ _∈_ supp( _µ_ )


By Corollary 3.34 we can further estimate the last term by


_≤_ sup _ℓ_ [2]
_**y**_ _∈_ [ _−_ 1 _,_ 1] _[J]_ _[ ∥P ◦F]_ [(] _**[y]**_ [)] _[ −N]_ [(] _[y]_ _[κ]_ [(1)] _[, . . ., y]_ _[κ]_ [(] _[m]_ [)] [)] _[∥]_ [2]

_≤_ _C∥P∥_ _m_ _[−]_ [2] _[s]_ _,_ where _s_ := [1]

_q_ _[−]_ [1] _[ >]_ [ 0] _[,]_


provided that _α_ _k_ _∈_ _ℓ_ _[q]_ (Z _[d]_ ) with _q ∈_ (0 _,_ 1). But by the exponential decay assumption
(3.52), we have _α_ _k_ _∈_ _ℓ_ _[q]_ (Z _[d]_ ) for any _q ∈_ (0 _,_ 1). The claim follows. 

Appendix D. DeepONet approximation of linear operators


In this section, we will illustrate the ability of a DeepONet to approximate
bounded linear operators efficienty. A simple numerical example of the DeepONet
approximation of a linear functional _G_ : _C_ ([0 _,_ 1] [2] ) _→_ R is presented in Figure 2,
below.


ERROR ESTIMATES FOR DEEPONETS 101


(a) (b)


(c) (d)


Figure 2. Illustration of a DeepONet ((1.1)-(1.3), with _p_ = 1 and
_σ_ ( _x_ ) = _x_ ) approximating the operator _G_ : _C_ ([0 _,_ 1] [2] ) _→_ R, defined
as _G_ ( _u_ ) := ´ ∆ _[u]_ [(] _[x]_ [)] _[ dx]_ [, with integration domain ∆(shown in red]

in (A)), with underlying measure being the law of a Gaussian random field with covariance kernel _k_ ( _x, y_ ) = exp( _−|x−y|_ [2] _/_ 2 _ℓ_ [2] ), with
_ℓ_ = 0 _._ 1. _m_ sensor points _x_ 1 _, . . ., x_ _m_ are drawn at random from the
uniform distribution on [0 _,_ 1] [2] and the DeepONet is trained by minimizing the mean square loss function, with respect to _N_ _u_ samples,
drawn from the underlying measure. (A) Illustration of a typical
sample drawn from the Gaussian random field. (B) Convergence
of the test mean squared error (computed with respect to 10 _[′]_ 240
test samples) as _m →∞_, for different numbers of training samples
_N_ _u_ . We observe a clear exponential decay of error wrt _m_ as well
as “resonances” at _m_ = _N_ _u_ (double descent). (C) Convergence of
the test MSE as _N_ _u_ _→∞_, for different numbers of random sensor
points _x_ 1 _, . . ., x_ _m_ . (D) semilog plot of the MSE loss as _N_ _u_ _→∞_
where one observes that the test error decays as exp( _−_ _[√]_ _N_ _u_ ) when
_N_ _u_ _≤_ _m_ . Note that similar behavior of the error is seen for more
complicated operators in Lu et al. (2019).


Of particular interest is the observed exponential decay in the DeepONet approximation error as the number of sensors _m →∞_, which has also been observed
for other (linear and even non-linear) problems in Lu et al. (2019). Can the theoretical framework developed in the present work explain such behaviour? To answer
this question, a general error estimate for the DeepONet approximation of linear
operators is derived in Theorem D.2. The estimate is then applied to a prototypical
elliptic PDE (cf. example D.4). In the following, we consider the following setup:


_**Setup**_ **D.1** _**.**_ We consider data for the DeepONet approximation problem _µ_, _G_ (cp.
Definition 2.1), where

_• G_ : _L_ [2] ( _D_ ) _→_ _L_ [2] ( _U_ ) is a bounded linear mapping,


102 ERROR ESTIMATES FOR DEEPONETS


_• µ ∈P_ 2 ( _L_ [2] ( _D_ )) is a probability measure with mean 0, and with uniformly
bounded eigenfunctions of the covariance operator Γ _µ_,

_•_ for _m ∈_ N, the sensors _x_ 1 _, . . ., x_ _m_ _∼_ Unif( _D_ ) are drawn iid random,

_•_ for _p ∈_ N, we denote by � _τ_ _k_, _k_ = 0 _, . . ., p_, the optimal choice for an affine
reconstruction _R_ _**τ**_ � = _R_ opt for the push-forward measure _G_ # _µ_, as in Theorem 3.14; i.e., in the present case, � _τ_ 0 _≡_ 0, and � _τ_ _k_, _k_ = 1 _, . . ., p_, are the first
_p_ eigenfunctions of the covariance operator Γ _G_ # _µ_ .


Under the above assumptions, we then have


_**Theorem**_ **D.2** _**.**_ Consider the setup D.1. Let _m, p ∈_ N denote the number of sensors
and the output dimension of the branch/trunk nets ( _**β**_ _,_ _**τ**_ ), respectively. Let _**τ**_ be a
trunk-net approximation of � _**τ**_, such that the associated reconstruction _R_ = _R_ _**τ**_ and
projection _P_ satisfy Lip( _R_ ) _,_ Lip( _R ◦P_ ) _≤_ 2. For _m ∈_ N, we define the (random)
encoder _E_ : _u �→_ ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )). Then, with probability 1 in the choice of the
random sensor points, there exists a constant _C >_ 0, depending only on _µ_ and the
measure of the domain _|D|_, such that for any _m, p ∈_ N there exists a shallow ReLU
approximator net _A_ : R _[m]_ _→_ R _[p]_ with


size( _A_ ) _≤_ 2(2 + _m_ ) _p,_ depth( _A_ ) _≤_ 1 _,_


and such that the DeepONet _N_ = _R_ _**τ**_ _◦A◦E_, with branch net _**β**_ = _A◦E_ and trunk
net _**τ**_ satisfies the following asymptotic DeepONet approximation error estimate



_λ_ _ℓ_ + ~~�~~
_ℓ>p_ � _ℓ>_ _C_ log( _m_



~~�~~ _λ_ _ℓ_

_ℓ>_ _C_ log( _m_ _m_ )



�
_E ≤_ _C_ ~~�~~ 1 + _∥G∥_ [2]



�
_E ≤_ _C_ ~~�~~








 _k_ =0 _,...,p_ _[∥][τ]_ _[k]_ _[ −]_ _[τ]_ [�] _[k]_ _[∥]_ _[L]_ [2] [(] _[U]_ [)] [ +]

 [max] ~~��~~ _ℓ>p_





(D.1)

~~~~

 _[,]_



for almost all _m_, as _m →∞_ .


_Proof._ By the DeepONet error decomposition of Theorem 3.3 and the assumed
Lipschitz bounds Lip( _R_ ) _,_ Lip( _R ◦P_ ) _≤_ 2, we have


� �
_E ≤_ 2 _∥G∥E_ _E_ + 2 � _E_ _A_ + � _E_ _R_ _._


We first observe that for any choice of the (affine) encoder/decoder and reconstruction/projection pairs ( _E, D_ ), ( _R, P_ ), and for a linear mapping _G_ there exists an
_exact, affine approximator A_ : R _[m]_ _→_ R _[p]_, such that _A_ ( _**u**_ ) = _P ◦G ◦D_ ( _**u**_ ) for all
_**u**_ _∈_ R _[m]_ . Furthermore, _A_ can be represented by a shallow ReLU neural net of the
claimed size, on account of the fact that _Ax_ + _b_ = _σ_ ( _Ax_ + _b_ ) _−_ _σ_ ( _−_ ( _Ax_ + _b_ )) has
an exact representation for the ReLU activation function _σ_ ( _x_ ) = max( _x,_ 0). Thus,
the approximation error _E_ [�] _A_ can be made to vanish in this case, _E_ [�] _A_ = 0.
Under the assumptions of this theorem, the random encoding error has been
estimated in Theorem 3.9, where it is shown that there exists a constant _C ≥_ 1,
depending only on the uniform upper bound of the eigenfunctions of Γ _µ_, and on
_|D|_, such that (for almost all _m ∈_ N):


�
_E_ _E_ _≤_ _C_ ~~�~~ _λ_ _ℓ_ _,_ (D.2)
� _ℓ>m/C_ log( _m_ )


as _m →∞_, with probability 1 in the iid random sensors _x_ 1 _, x_ 2 _, · · · ∼_ Unif( _D_ ).
Here, _C_ = _C_ ( _|D|, µ_ ) is a constant.
Finally, we estimate the reconstruction error _E_ [�] _R_ : By Proposition 3.15, we have



1 + Tr(Γ _G_ # _µ_ ) max _k_ =0 _,...,p_ _[∥][τ]_ [�] _[k]_ _[ −]_ _[τ]_ _[k]_ _[∥]_ _[L]_ [2] [(] _[U]_ [)] [ +] ~~�~~
� _k>p_



�
_E_ _R_ _≤_
~~�~~



~~�~~ _λ_ _[G]_ _k_ [#] _[µ]_ _,_

_k>p_


ERROR ESTIMATES FOR DEEPONETS 103



where _λ_ _[G]_ 1 [#] _[µ]_ _≥_ _λ_ _[G]_ 2 [#] _[µ]_ _≥_ _. . ._ denote the eigenvalues of the covariance operator Γ _G_ # _µ_
of the push-forward measure _G_ # _µ_ . By Proposition 3.20, we can estimate
� _λ_ _k_ _[G]_ [#] _[µ]_ _≤∥G∥_ [2] [ �] _λ_ _k_ _,_



� _λ_ _k_ _[G]_ [#] _[µ]_ _≤∥G∥_ [2] [ �]

_k>p_ _k>p_



_λ_ _k_ _,_

_k>p_



in terms of the eigenvalues _λ_ 1 _≥_ _λ_ 2 _≥_ _. . ._, of the covariance operator Γ _µ_ of _µ_, and
in particular Tr(Γ _G_ # _µ_ ) _≤∥G∥_ [2] Tr(Γ _µ_ ) _≤_ _C∥G∥_ [2], where _C_ = _C_ ( _µ_ ) depends only on
_µ_ . It thus, follows that there exists a constant _C_ = _C_ ( _µ_ ), such that



_λ_ _ℓ_

_ℓ>p_



�
_E_ _R_ _≤_ _C_ ~~�~~ 1 + _∥G∥_ [2]








 _k_ =0 _,...,p_ _[∥][τ]_ [�] _[k]_ _[ −]_ _[τ]_ _[k]_ _[∥]_ _[L]_ [2] [(] _[U]_ [)] [ +]

 [max] ~~��~~ _ℓ>p_





(D.3)



 _[.]_



Combining (D.2) and (D.3) with the error decomposition, and taking into account
that _E_ [�] _A_ = 0, yields the desired result. 

_**Remark**_ **D.3** _**.**_ From the proof of Theorem D.2, it is clear that to achieve the error
bound (D.1), we can replace the shallow ReLU approximator net _A_, by a single-layer
affine approximator _A_ : R _[m]_ _→_ R _[p]_, _A_ ( _**u**_ ) := _A ·_ _**u**_ + _b_, where _A ∈_ R _[p][×][m]_, _b ∈_ R _[m]_ .
This clearly corresponds to the use of the linear activation function _σ_ ( _x_ ) = _x_ in
the branch net. In contrast, the activation function in the trunk net must be kept
non-linear to be able to approximate the optimal trunk net � _**τ**_ .



_**Example**_ **D.4** _**.**_ To illustrate Theorem D.2, we consider the following example of
a linear operator _G_ : _L_ [2] (T _[d]_ ) _→_ _L_ [2] (T _[d]_ ), _f �→_ _v_, mapping the source term _f_ to the
solution of the PDE ∆ _v_ = _f −_ ffl T _[d]_ _[ f dx]_ [, with periodic boundary conditions and]



solution of the PDE ∆ _v_ = _f −_ T _[d]_ _[ f dx]_ [, with periodic boundary conditions and]

imposing ffl T _[d]_ _[ v dx]_ [ = 0. It is well-known that this operator is bounded; in fact, we]



imposing T _[d]_ _[ v dx]_ [ = 0. It is well-known that this operator is bounded; in fact, we]

have _∥G∥≤_ 1. We fix the initial measure _µ ∈P_ ( _L_ [2] (T _[d]_ )) as a Gaussian random
field, with Karhunen-Loeve expansion



_f_ = � _α_ _k_ _X_ _k_ **e** _k_ _,_

_k∈_ Z _[d]_



where _|α_ _k_ _| ≤_ exp( _−ℓ|k|_ ) have exponential decay with typical length scale _ℓ>_ 0,
_{_ **e** _k_ _}_ _k∈_ Z _d_ is the standard Fourier basis on T _[d]_, and the _X_ _k_ _∼N_ (0 _,_ 1) are iid Gaussian
random variables. In this case, the eigenfunctions of the associated covariance
operator Γ _µ_ of _µ_ are given by the **e** _k_, with corresponding eigenvalues _α_ _k_, _k ∈_ Z _[d]_ .
In particular, it follows that there exists constants _C, c >_ 0, depending only on _d_
and _ℓ_, such that the last two terms in (D.1) can be bounded from above by



_c m_ [1] _[/d]_
_≤_ _C_ exp _−c p_ [1] _[/d]_ [�] + _C_ exp _−_
� � log( _m_ ) [1] _[/d]_



_._
�



In particular, Theorem D.2 implies that for any fixed _σ >_ 0, and with _p ∼_ log( _ϵ_ _[−]_ [1] ) _[d]_,
_m ∼_ log( _ϵ_ _[−]_ [1] ) _[d]_ [(1+] _[σ]_ [)], we can achieve an overall DeepONet approximation error

�
_E_ ≲ _ϵ_ + max _j_ =0 _,...,p_ _[∥]_ **[e]** _[j]_ _[ −]_ _[τ]_ _[j]_ _[∥]_ _[L]_ [2] [(] _[U]_ [)] _[,]_


where we have taken into account that the Fourier basis **e** _j_ is an eigenbasis also for
the push-forward measure _G_ # _µ_ . Furthermore, it follows from Lemma 3.17, that for
any fixed _σ >_ 0, there exists a trunk net _**τ**_ with asymptotic size (as _ϵ →_ 0)

size( _**τ**_ ) ≲ log( _ϵ_ _[−]_ [1] ) _[d]_ [+2+] _[σ]_ _,_ depth( _**τ**_ ) ≲ log( _ϵ_ _[−]_ [1] ) [2+] _[σ]_ _,_ (D.4)


and such that max _k_ =0 _,...,p_ _∥_ **e** _j_ _−_ _τ_ _j_ _∥_ _L_ 2 ( _U_ ) _< ϵ_ . Thus, for the present example, an
overall DeepONet error _E_ [�] ≲ _ϵ_ can be achieved with a DeepONet ( _**β**_ _,_ _**τ**_ ) with trunk
net _**τ**_ satisfying the size bounds (D.4), and a branch net _**β**_ of size

size( _**β**_ ) ≲ log( _ϵ_ _[−]_ [1] ) [2] _[d]_ [+] _[σ]_ _,_ depth( _**β**_ ) ≲ 1 _,_ (D.5)


104 ERROR ESTIMATES FOR DEEPONETS


for any fixed _σ >_ 0. The implied constants here depend on the length scale _ℓ>_ 0,
the dimension _d_ and the additional parameter _σ >_ 0, which was introduced to avoid
the appearance of multiple logarithms.


Appendix E. Proofs of Results in Section 4


E.1. **Proof of Lemma 4.1.**


_Proof._ Let _v, v_ _[′]_ solve (4.2) with forcing _u, u_ _[′]_, respectively. Then we can write

_d_ ( _v −_ _v_ _[′]_ ) = _G_ ( _v, v_ _[′]_ )( _v −_ _v_ _[′]_ ) + ( _u −_ _u_ _[′]_ ) _,_ (E.1)

_dt_


where


1
_G_ ( _v, v_ _[′]_ ) = _g_ [(1)] ( _sv_ + (1 _−_ _s_ ) _v_ _[′]_ ) _ds,_
ˆ 0

so that _|G_ ( _v, v_ _[′]_ ) _| ≤∥g_ [(1)] _∥_ _L_ _∞_ for all _v, v_ _[′]_ . It follows readily from (E.1) that


_d_
_dt_ _[|][v][ −]_ _[v]_ _[′]_ _[|]_ [2] _[ ≤]_ _[C][|][v][ −]_ _[v]_ _[′]_ _[|]_ [2] [ +] _[ |][u][ −]_ _[u]_ _[′]_ _[|]_ [2] _[,]_


for some _C >_ 0 depending only on _∥g_ [(1)] _∥_ _L_ _∞_ . Gronwall’s inequality then implies
that


_t_
_|v −_ _v_ _[′]_ _|_ [2] ( _t_ ) _≤_ _|u −_ _u_ _[′]_ _|_ [2] _ds e_ _[Ct]_ _≤∥u −_ _u_ _[′]_ _∥_ [2] _L_ [2] ([0 _,T_ ]) _[e]_ _[CT]_ _[ .]_
ˆ 0


The claim follows by integration over _t ∈_ [0 _, T_ ]. 

E.2. **Proof of Lemma 4.3.**


_Proof._ We first show that there exists a constant _C >_ 0, depending only on the
final time _T_, such that
_∥v∥_ _L_ _∞_ _≤_ _C∥u∥_ _L_ 2 _._


To this end, we simply note that integrating (4.2) from 0 to _T_ and taking into
account that _v_ (0) = 0, we have



_t_
_|v_ ( _t_ ) _| ≤_
ˆ 0



_|u_ ( _s_ ) _| ds_
0



_t_ _t_

_|g_ ( _v_ ( _s_ )) _| ds_ +
0 ˆ 0



_t_
_≤∥g_ [(1)] _∥_ _L_ _∞_ _|v_ ( _s_ ) _| ds_ + _√_
ˆ 0



_T_ _∥u∥_ _L_ 2 ([0 _,T_ ])



_≤_ _C_ 1



_t_

_|v_ ( _s_ ) _| ds_ + _√_

ˆ 0



_T_ _∥u∥_ _L_ 2 ([0 _,T_ ]) _._



Gronwall’s inequality implies that



_|v_ ( _t_ ) _| ≤_ _√_



_T_ _∥u∥_ _L_ 2 ([0 _,T_ ]) _e_ _[C]_ [1] _[T]_ _,_ _∀_ _t ∈_ [0 _, T_ ] _._



Thus, _∥v∥_ _L_ _∞_ ([0 _,T_ ]) _≤_ _C∥u∥_ _L_ 2 ([0 _,T_ ], where _C_ = _√T_ exp( _C_ 1 _T_ ) depends only on _T_ .

For general _k ∈_ N, we take _k_ derivatives of (4.2) to find:


_d_
_{g_ [(] _[ℓ]_ [)] _}_ _[k]_ _ℓ_ =1 _[−]_ [1] _[,][ {][v]_ _[ℓ]_ _[}]_ _ℓ_ _[k]_ =1 _[−]_ [1] + _u_ [(] _[k]_ [)] _._ (E.2)
_dt_ _[v]_ [(] _[k]_ [)] [ =] _[ g]_ [(] _[k]_ [)] [(] _[v]_ [)] _[ v]_ [(] _[k]_ [)] [ +] _[ P]_ _[k]_ � �


Here, _P_ _k_ = _P_ _k_ � _{g_ [(] _[ℓ]_ [)] _}_ _[k]_ _ℓ_ =1 _[−]_ [1] _[,][ {][v]_ _[ℓ]_ _[}]_ _[k]_ _ℓ_ =1 _[−]_ [1] � is a polynomial of the following form



_C_ _ℓ,_ _**γ**_

_**γ**_



 _._







 [�] _**γ**_



_P_ _k_ _{g_ [(] _[ℓ]_ [)] _}_ _[k]_ _ℓ_ =1 _[−]_ [1] _[,][ {][v]_ _[ℓ]_ _[}]_ _[k]_ _ℓ_ =1 _[−]_ [1] =
� �



_k−_ 1
� _g_ [(] _[ℓ]_ [)] ( _v_ )


_ℓ_ =1



_N_ _**γ**_
� _v_ [(] _[γ]_ _[j]_ [)]

_j_ =1


ERROR ESTIMATES FOR DEEPONETS 105


The sum in the parentheses is over the (finite) set of _**γ**_ = ( _γ_ 1 _, . . ., γ_ _N_ _**γ**_ ), _γ_ _j_ _∈_ N,
satisfying


_N_ _**γ**_
� _γ_ _j_ = _k,_ 1 _≤_ _γ_ _j_ _< k,_ _∀_ _j_ = 1 _, . . ., N_ _**γ**_ _._

_j_ =1


The coefficients _C_ _ℓ,_ _**γ**_ are combinatorial coefficients that depend only on _k_, and
which can in principle be determined for any given _k_ . We will prove the claimed
estimate on _∥v_ [(] _[k]_ [)] _∥_ _L_ _∞_ by induction on _k_ = 1 _,_ 2 _, . . ._ . We will first estimate the
size of the polynomial _P_ _k_ : To this end, note that for _k_ = 1, the sum defining _P_ _k_
is necessarily empty and thus, we have _P_ _k_ _≡_ 0. For _k >_ 1, we assume that the
claimed inequality for _∥v_ [(] _[k]_ [)] _∥_ _L_ _∞_ has already been proven to hold for derivatives
_v_ [(] _[γ]_ _[j]_ [)] of order _γ_ _j_ _≤_ _k −_ 1. In this case, we can estimate



_|C_ _ℓ,_ _**γ**_ _|_

_**γ**_





 _,_











 [�] _**γ**_



_|P_ _k_ _| ≤_



_k−_ 1
� _∥g_ [(] _[ℓ]_ [)] _∥_ _L_ _∞_


_ℓ_ =1



_N_ _**γ**_
� _∥v_ [(] _[γ]_ _[j]_ [)] _∥_ _L_ _∞_

_j_ =1



_N_ _**γ**_
�



and



_N_ _**γ**_
� _∥v_ [(] _[γ]_ _[j]_ [)] _∥_ _L_ _∞_ _≤_

_j_ =1


_≤_


=



_N_ _**γ**_
� _A_ _γ_ _j_ (1 + _∥u∥_ _H_ _[γ]_ _j_ ) _[γ]_ _[j]_

_j_ =1









_N_ _**γ**_
� _A_ _γ_ _j_

_j_ =1



_N_ _**γ**_
� _A_ _γ_ _j_ (1 + _∥u∥_ _H_ _k_ ) _[γ]_ _[j]_

_j_ =1



 (1 + _∥u∥_ _H_ _k_ ) _[k]_ _._





Thus, taking into account that the derivatives _g_ [(] _[ℓ]_ [)] ( _v_ ) of order _ℓ_ _≤_ _k_ _−_ 1 are assumed
to be uniformly bounded, _|g_ [(] _[ℓ]_ [)] ( _v_ ) _| ≤_ _C_ _k_, we can now estimate


_|P_ _k_ _| ≤_ _C_ (1 + _∥u∥_ _[k]_ _H_ _[k]_ [)] _[,]_ ( _k >_ 1) _,_


where _C_ = _C_ ( _k, g_ ) _>_ 0 is a constant depending only on _k_ and _g_ . As pointed out
above, this inequality for _P_ _k_ holds trivially also for _k_ = 1, since _P_ _k_ _≡_ 0, in this
case. Inserting the above estimate in (E.2), and integrating over [0 _, t_ ], we obtain



_t_ _T_

_|v_ [(] _[k]_ [)] ( _s_ ) _| ds_ + _C_ (1 + _∥u∥_ _H_ _k_ ) _[k]_ +
0 ˆ 0



_t_
_|v_ [(] _[k]_ [)] ( _t_ ) _| ≤∥g_ [(] _[k]_ [)] _∥_ _L_ _∞_
ˆ 0



_|u_ [(] _[k]_ [)] ( _s_ ) _| ds_
0



_≤_ _C_ _k_



_t_

_|v_ [(] _[k]_ [)] ( _s_ ) _| ds_ + _C_ (1 + _∥u∥_ _H_ _k_ ) _[k]_ + _√_

ˆ 0



_T_ _∥u∥_ _H_ _k_ _._



Increasing the constant _C_ if necessary, we can absorb the last term in the second
term and conclude that there exists a constant _C_ = _C_ ( _k, g, T_ ) _>_ 0, such that



_|v_ [(] _[k]_ [)] ( _t_ ) _| ≤_ _C_ _k_



_t_

_|v_ [(] _[k]_ [)] ( _s_ ) _| ds_ + _C_ (1 + _∥u∥_ _H_ _k_ ) _[k]_ _._

ˆ 0



Gronwall’s inequality now yields


_|v_ [(] _[k]_ [)] ( _t_ ) _| ≤_ _Ce_ _[C]_ _[k]_ _[T]_ (1 + _∥u∥_ _H_ _k_ ) _[k]_ _,_


for all _t ∈_ [0 _, T_ ]. Hence, for _A_ _k_ = _Ce_ _[C]_ _[k]_ _[T]_, we have


_∥v_ [(] _[k]_ [)] _∥_ _L_ _∞_ ([0 _,T_ ]) _≤_ _A_ _k_ (1 + _∥u∥_ _H_ _k_ ) _[k]_ _,_


where _A_ _k_ is independent of _u_ . 

106 ERROR ESTIMATES FOR DEEPONETS


E.3. **Proof of Lemma 4.4.**


_Proof._ The Legendre polynomials form an orthonormal basis of _L_ [2] ([0 _, T_ ]). For _p ∈_
N, let _P_ _p_ : _L_ [2] ([0 _, T_ ]) _→_ _L_ [2] ([0 _, T_ ]) denote the orthogonal projection onto the span
of the first _p_ Legendre polynomials, span( _τ_ � 1 _, . . .,_ � _τ_ _p_ ). By (Canuto & Quarteroni
1982, Theorem 2.3), for any _k ∈_ N, there exists a constant _C_ = _C_ ( _T, k_ ) _>_ 0, such
that

_∥u −_ _P_ _p_ _u∥_ _L_ 2 ([0 _,T_ ]) _≤_ _Cp_ _[−][k]_ _∥u∥_ _H_ _k_ ([0 _,T_ ]) _._


Thus, if _R_ = _R_ _**τ**_ � is the reconstruction with trunk net � _**τ**_ = (0 _,_ � _τ_ 1 _, . . .,_ � _τ_ _p_ ), and if _P_
denotes the corresponding optimal projection (3.19), then _P_ _p_ = _R ◦P_, and


� 1 _/_ 2
_E_ _R_ = _∥R ◦P_ ( _v_ ) _−_ _v∥_ [2] _L_ [2] ([0 _,T_ ]) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[v]_ [)]
�ˆ _X_ �



_≤_ _[C]_

_p_ _[k]_



= _[C]_

_p_ _[k]_



1 _/_ 2
_∥v∥_ [2] _H_ _[k]_ ([0 _,T_ ]) _[d]_ [(] _[G]_ [#] _[µ]_ [)(] _[v]_ [)]

�ˆ _X_ �


1 _/_ 2
_∥G_ ( _u_ ) _∥_ [2] _H_ _[k]_ ([0 _,T_ ]) _[dµ]_ [(] _[u]_ [)]

�ˆ _X_ �



By Lemma 4.3, we can furthermore estimate _∥v∥_ _H_ _k_ _≤_ _C_ (1 + _∥u∥_ _H_ _k_ ) _[k]_, for some
constant _C >_ 0, depending on _T_ and _k_ . Hence, we find



1 _/_ 2
2 _k_
�1 + _∥u∥_ _H_ _k_ ([0 _,T_ ]) � _dµ_ ( _u_ ) _,_
�



�
_E_ _R_ _≤_ _[C]_

_p_ _[k]_



�ˆ _X_



where _C_ = _C_ ( _k, T_ ) _>_ 0 is a constant independent of _p_ . 

E.4. **Proof of Lemma 4.6.**


_Proof._ We have by (4.12)


_t_
_|v_ _i_ ( _t_ ) _| ≤_ _α_ + _|v_ _i_ ( _s_ ) _|β_ ( _s_ ) _ds,_
ˆ 0


where _α_ = _∥_ Im( _u_ ) _∥_ _L_ _∞_ _T_, and _β_ ( _s_ ) = _γ_ (1 + exp( _|v_ _i_ ( _s_ ) _|_ )). By Gronwall’s inequality,
it follows that


_t_
_|v_ _i_ ( _t_ ) _| ≤_ _α_ exp _β_ ( _s_ ) _ds_
�ˆ 0 �



Let



_t_
= _∥_ Im( _u_ ) _∥_ _L_ _∞_ _T_ exp _γ_ 1 + _e_ _[|][v]_ _[i]_ [(] _[s]_ [)] _[|]_ [�] _ds_ _._
�ˆ 0 � �


1
_δ_ :=
2 _T_ exp( _γ_ (1 + _e_ ) _T_ ) _[.]_



We claim that if _∥_ Im( _u_ ) _∥_ _L_ _∞_ _≤_ _δ_, then _|v_ _i_ ( _t_ ) _| <_ 1 for all _t ∈_ [0 _, T_ ]. Suppose this
was not the case. If there exists _t ∈_ [0 _, T_ ] such that _|v_ _i_ ( _t_ ) _| ≥_ 1, then the set


_B_ := _{t ∈_ [0 _, T_ ] _| |v_ _i_ ( _t_ ) _| ≥_ 1 _}_


is nonempty. Let _t_ 0 be given by


_t_ 0 = inf _B._


By the continuity of _t �→_ _v_ _i_ ( _t_ ), _B_ is a closed set. In particular, this implies that
_t_ 0 _∈_ _B_ . Since _v_ _i_ (0) = 0, we must have _t_ 0 _>_ 0, and _|v_ _i_ ( _t_ ) _| <_ 1 for all _t ∈_ [0 _, t_ 0 ). But


ERROR ESTIMATES FOR DEEPONETS 107



then,



_t_ 0
_|v_ _i_ ( _t_ 0 ) _| ≤∥_ Im( _u_ ) _∥_ _L_ _∞_ _T_ exp _γ_ 1 + _e_ _[|][v]_ _[i]_ [(] _[s]_ [)] _[|]_ [�] _ds_
�ˆ 0 � �

_≤∥_ Im( _u_ ) _∥_ _L_ _∞_ _T_ exp( _γ_ (1 + _e_ ) _T_ )


_≤_ _δ T_ exp( _γ_ (1 + _e_ ) _T_ )

_≤_ [1]

2 _[<]_ [ 1] _[,]_


leads to a contradiction to the assumption that _|v_ _i_ ( _t_ 0 ) _| ≥_ 1. Thus, we conclude
that the set _B_ must in fact be empty, i.e. that


_|v_ _i_ ( _s_ ) _| <_ 1 _,_ for all _s ∈_ [0 _, T_ ] _._


                       

E.5. **Proof of Lemma 4.7.**


_Proof._ We argue by contradiction. Suppose the claim is not true. Then there exists
_u_ : [0 _, T_ ] _→_ C, _u ∈_ _L_ _[∞]_ ([0 _, T_ ]), and 0 _< T_ 1 _< T_ such that the solution [4] of (4.11)
is defined on [0 _, T_ 1 ), but lim _t↗T_ 1 _|v_ ( _t_ ) _|_ = _∞_ . Since the right-hand side of (4.11) is
uniformly Lipschitz continuous in _v_ _r_, this can only be the case, if lim _t↗T_ 1 _|v_ _i_ ( _t_ ) _|_ =
_∞_ . In particular, there exists _T_ 0 _< T_ 1, such that _|v_ _i_ ( _T_ 0 ) _| >_ 1. But then, this would
imply the existence of a solution of (4.11), which is defined on [0 _, T_ 0 ], for which
sup _s∈_ [0 _,T_ ] _|_ Im( _u_ ( _s_ )) _| ≤_ _δ_, and such that we have sup _s∈_ [0 _,T_ 0 ] _|v_ _i_ ( _s_ ) _| ≥|v_ _i_ ( _T_ 0 ) _| >_ 1.
This is clearly in contradiction with Lemma 4.6. By contradiction, it thus follows
that we must have _T_ 0 = _T_, and hence the solution of (4.11) exists on [0 _, T_ ] for any
_u ∈_ _L_ _[∞]_ ([0 _, T_ ]), with _∥_ Im( _u_ ) _∥_ _L_ _∞_ ([0 _,T_ ]) _≤_ _δ_ . Furthermore, it follows from Lemma
4.6 that sup _s∈_ [0 _,T_ ] _|v_ _i_ ( _s_ ) _| ≤_ 1 in this case. 

E.6. **Proof of Lemma 4.8.**


_Proof._ Fix _z_ _k_ _∈_ _E_ _ρ_ _k_ for _k ̸_ = _j_ . For _z_ _j_ _∈_ _E_ _ρ_ _j_, and by slight abuse of notation,
let us denote the forcing by _u_ ( _t, z_ _j_ ) = _u_ ( _t,_ _**z**_ ), and the corresponding solution of
the pendulum equations (4.11) by _v_ ( _t, z_ _j_ ) = _F_ ( _**z**_ )( _t_ ), where _**z**_ = ( _z_ _ℓ_ ) _ℓ∈_ N . Then
the claim is that _z_ _j_ _�→_ _v_ ( _t, z_ _j_ ) is complex-differentiable in _z_ _j_ _∈_ _E_ _ρ_ _j_ . We note that
_v_ ( _t, z_ _j_ ) is the unique fixed point of


_t_
_v_ ( _t, z_ _j_ ) = _G_ ( _s, z_ _j_ _, v_ ( _s, z_ _j_ )) _ds,_
ˆ 0



where _G_ ( _t, z_ _j_ _, v_ ) := _g_ ( _v_ ) + _u_ ( _t, z_ _j_ ). To show boundedness, we note that by (4.13),
the assumed ( _**b**_ _, δ_ )-admissibility of _**ρ**_ and Lemma 4.7, the solution _t �→_ _v_ ( _t, z_ _j_ ) exists
for any _z_ _j_ _∈_ _E_ _ρ_ _j_, and that sup _t∈_ [0 _,T_ ] _|_ Im( _v_ ( _t, z_ _j_ )) _| ≤_ 1. But then, also the real part
_v_ _r_ ( _t_ ) := Re( _v_ ( _t, z_ _j_ )) is bounded, because from (4.11), we conclude that

_dv_ _r_ ( _t_ )

_≤|g_ _r_ ( _v_ _r_ ( _t_ ) _, v_ _i_ ( _t_ )) _|_ + _|_ Re( _U_ ( _t_ )) _|_

���� _dt_ ����

_≤|v_ _r_ ( _t_ ) _|_ + _k_ cosh( _|v_ _i_ ( _t_ ) _|_ ) + _|_ Re( _U_ ( _t_ )) _|_


_≤|v_ _r_ ( _t_ ) _|_ + _k_ cosh(1) + _∥_ Re( _U_ ) _∥_ _L_ _∞_ ([0 _,T_ ]) _,_


which implies by Gronwall’s inequality that


_|v_ _r_ ( _t_ ) _| ≤_ � _k_ cosh(1) + _∥_ Re( _U_ ) _∥_ _L_ _∞_ ([0 _,T_ ]) � _Te_ _[T]_ _,_


4 We note that short-time existence follows from the fact that the right-hand side is locally
Lipschitz continuous in _v_, so that _T_ 1 _>_ 0.



_dt_



_≤|g_ _r_ ( _v_ _r_ ( _t_ ) _, v_ _i_ ( _t_ )) _|_ + _|_ Re( _U_ ( _t_ )) _|_
����


108 ERROR ESTIMATES FOR DEEPONETS


for all _t ∈_ [0 _, T_ ]. We furthermore note that



_∞_
� _|_ Re( _z_ _k_ ) _|b_ _k_


_k_ =1



_∥_ Re( _U_ ) _∥_ _L_ _∞_ ([0 _,T_ ]) _≤_


_≤_



_∞_
� _|_ Re( _z_ _k_ ) _|α_ _k_ _∥ψ_ _k_ _∥_ _L_ _∞_ =


_k_ =1



_∞_
� _ρ_ _k_ _b_ _k_ =


_k_ =1



_∞_
� _b_ _k_ +


_k_ =1



_∞_
�( _ρ_ _k_ _−_ 1) _b_ _k_


_k_ =1



_≤∥_ _**b**_ _∥_ _ℓ_ 1 (N) + _δ < ∞,_


is uniformly bounded for all _**z**_ _∈_ _E_ _**ρ**_ for any admissible _**ρ**_ . This shows that



sup _∥v∥_ _L_ 2C [([0] _[,T]_ [ ])] _[ ≤]_ [sup]
_**z**_ _∈E_ _**ρ**_ _**z**_ _∈E_ _**ρ**_


_≤_ sup
_**z**_ _∈E_ _**ρ**_



_√T_ _∥v∥_ _L_ _∞_ ([0 _,T_ ])


_√T_ _∥v_ _r_ _∥_ _L_ _∞_ ([0 _,T_ ]) + sup

_**z**_ _∈E_ _**ρ**_



_√_



_T_ _∥v_ _i_ _∥_ _L_ _∞_ ([0 _,T_ ])



_≤_ _T_ _√Te_ _[T]_ [ �] _γ_ cosh(1) + _∥_ _**b**_ _∥_ _ℓ_ 1 (N) + _δ_ � + _√_

=: _C,_



_T_



is uniformly bounded for all ( _**b**_ _, ϵ_ )-admissible _**ρ**_ .
Finally, we prove the holomorphy of _z_ _j_ _�→_ _v_ ( _t, z_ _j_ ): By definition, _z_ _j_ _�→_ _u_ ( _t, z_ _j_ )
is an affine function. Hence _z_ _j_ _�→_ _G_ ( _t, z_ _j_ _, v_ ) is also an affine function of _z_ _j_, and in
particular differentiable in _z_ _j_ . Since _v_ ( _t, z_ _j_ ) exists and is bounded for all _z_ _j_ _∈_ _E_ _ρ_ _j_,
it then follows from the general theory of parametric ODEs that _∂_ _z_ _j_ _v_ ( _t, z_ _j_ ) exists
and that



_t_
_∂_ _z_ _j_ _v_ ( _t, z_ _j_ ) =
ˆ 0



� _∂_ _z_ _j_ _G_ ( _s, z_ _j_ _, v_ ( _s, z_ _j_ )) + _∂_ _v_ _G_ ( _s, z_ _j_ _, v_ ( _s, z_ _j_ )) _∂_ _z_ _j_ _v_ ( _s, z_ _j_ )� _ds._



This implies that _z_ _j_ _�→_ _v_ ( _t, z_ _j_ ) is a holomorphic mapping. 

E.7. **Proof of Lemma 4.14.**


_Proof._ The difference _w_ = _u −_ _u_ _[′]_ is a solution of the equation


_∇·_ ( _a∇w_ ) = _∇·_ ( _a∇u_ ) _−∇·_ ( _a∇u_ _[′]_ )


= _f −∇·_ ( _a∇u_ _[′]_ )


= _∇·_ ( _a_ _[′]_ _∇u_ _[′]_ ) _−∇·_ ( _a∇u_ _[′]_ )


= _∇·_ (( _a_ _[′]_ _−_ _a_ ) _∇u_ _[′]_ ) _._


By elliptic theory, we thus have


_∥u −_ _u_ _[′]_ _∥_ _L_ 2 ( _D_ ) = _∥w∥_ _L_ 2 ( _D_ ) _≤∥∇·_ (( _a_ _[′]_ _−_ _a_ ) _∇u_ _[′]_ ) _∥_ _H_ _−_ 1 ( _D_ )
_≤∥_ ( _a_ _[′]_ _−_ _a_ ) _∇u_ _[′]_ _∥_ _L_ 2 ( _D_ ) _≤∥a_ _[′]_ _−_ _a∥_ _L_ _∞_ ( _D_ ) _∥∇u_ _[′]_ _∥_ _L_ 2 ( _D_ )
_≤∥a_ _[′]_ _−_ _a∥_ _L_ _∞_ ( _D_ ) _∥u_ _[′]_ _∥_ _H_ 01 [(] _[D]_ [)] _[ ≤∥][a]_ _[′]_ _[ −]_ _[a][∥]_ _[L]_ _[∞]_ [(] _[D]_ [)] _[C][∥][f]_ _[∥]_ _[L]_ [2] [(] _[D]_ [)] _[.]_


                       

E.8. **Proof of Lemma 4.16.**


_Proof._ It is well-known that if _u_ is a solution of (4.18), with smooth coefficient _a_ ( _x_ )
and right-hand side _f ∈_ _H_ _[k]_, then _u ∈_ _H_ _[k]_ [+1] . The main point of this lemma is
the explicit dependence on the norm of _a_, which will be required to estimate the
reconstruction error. Let _**k**_ = ( _k_ 1 _, . . ., k_ _m_ ) _∈_ N _[m]_ 0 [denote any multi-index. Then by]
differentiation of (4.18), we find



_∂_ _x_ _**[k][−][ℓ]**_ _a_ ( _x_ ) _∇∂_ _x_ _**[ℓ]**_ _[u]_
� �



_∇·_ � _a_ ( _x_ ) _∇∂_ _x_ _**[k]**_ _[u]_ � = _−∇·_



�� _**ℓ**_ _<_ _**k**_



_**k**_
� _**ℓ**_



+ _∂_ _x_ _**[k]**_ _[f.]_ (E.3)


ERROR ESTIMATES FOR DEEPONETS 109



where _ℓ_ runs over all indices _**ℓ**_ = ( _ℓ_ 1 _, . . ., ℓ_ _m_ ) _∈_ N 0, such that _ℓ_ _j_ _≤_ _k_ _j_ for all
_j_ = 1 _, . . ., m_, with a strict inequality for at least one _j_ . We also write

_**k**_ _m_ _k_ _j_

:= _._

� _**ℓ**_ � � � _ℓ_ _j_ �



:=
�



_m_
�

_j_ =1



_k_ _j_
� _ℓ_ _j_



_._
�



Integrating (E.3) against _∂_ _x_ _**[k]**_ _[u]_ [, with] _[ k]_ [ :=] _[ |]_ _**[k]**_ _[|]_ [ =] _[ k]_ [1] [+] _[ · · ·]_ [ +] _[ k]_ _[m]_ [, it follows that]


_λ∥∇∂_ _x_ _**[k]**_ _[u][∥]_ [2] _L_ [2] _x_ _[≤]_ _x_ [(] _[x]_ [)] _[|]_ [2] _[ dx]_
ˆ T _[n]_ _[ a]_ [(] _[x]_ [)] _[|∇][∂]_ _**[k]**_



_≤_ _C_ _k_ �

_**ℓ**_ _<_ _**k**_



ˆ T _[n]_



_**k−ℓ**_ _**ℓ**_ _**k**_
�� _∂_ _x_ _a_ ( _x_ )���� _∇∂_ _x_ _[u]_ [(] _[x]_ [)] ���� _∇∂_ _x_ _[u]_ [(] _[x]_ [)] �� _dx_



+ _x_ _[f]_ _[||][∂]_ _x_ _**[k]**_ _[u][|][ dx]_
ˆ T _[n]_ _[ |][∂]_ _**[k]**_

_≤_ _C_ _k_ _∥a∥_ _C_ _k_ _∥u∥_ _H_ _k_ _∥∇∂_ _x_ _**[k]**_ _[u][∥]_ _L_ [2] [ +] _[ ∥][∂]_ _x_ _**[k]**_ _[f]_ _[∥]_ _L_ [2] _[∥][u][∥]_ _H_ _[k]_ _[.]_


Using the inequality _ab ≤_ 2 _[−]_ [1] _ϵa_ [2] + (2 _ϵ_ ) _[−]_ [1] _b_ [2] for _a, b, ϵ >_ 0, we find



_k_ _[∥][a][∥]_ [2] _C_ _[k]_
_λ∥∇∂_ _x_ _**[k]**_ _[u][∥]_ [2] _L_ [2] _x_ _[≤]_ _[C]_ [2] 2 _λ_ _∥u∥_ [2] _H_ _[k]_ [ +] _[ λ]_ 2



_x_ _[u][∥]_ [2] _L_ [2] [ +] _[ λ]_
2 _[∥∇][∂]_ _**[k]**_ 2



2 _λ_ _[∥][u][∥]_ _H_ [2] _[k]_ _[,]_



_x_ _[f]_ _[∥]_ [2] _L_ [2] [ +] [1]
2 _[∥][∂]_ _**[k]**_ 2



and hence

_∥∇∂_ _x_ _**[k]**_ _[u][∥]_ [2] _L_ [2] _x_ _[≤]_ [(] _[C]_ _k_ [2] _[∥][a][∥]_ [2] _C_ _[k]_ [ + 1)] _[∥][u]_ _λ_ _[∥]_ [2] _H_ [2] _[k]_ + _∥∂_ _x_ _**[k]**_ _[f]_ _[∥]_ [2] _L_ [2] _[.]_

Summing the last estimate over all _|_ _**k**_ _|_ = _k_, and increasing the constant _C_ _k_, if
necessary, we obtain

_∥u∥_ [2] _H_ _[k]_ [+1] _[ ≤]_ _[C]_ _[k]_ [(] _[∥][a][∥]_ [2] _C_ _[k]_ [ + 1)] _[∥][u][∥]_ _H_ [2] _[k]_ [ +] _[ ∥][f]_ _[∥]_ [2] _H_ _[k]_ _[,]_


where the new constant _C_ _k_ = _C_ _k_ ( _k, λ_ ) now depends on both _k_ and _λ_ . Repeating
the same argument for any _ℓ_ _≤_ _k_, we also find

_∥u∥_ [2] _H_ _[ℓ]_ [+1] _[ ≤]_ _[C]_ _[ℓ]_ [(] _[∥][a][∥]_ [2] _C_ _[ℓ]_ [+ 1)] _[∥][u][∥]_ _H_ [2] _[ℓ]_ [+] _[ ∥][f]_ _[∥]_ [2] _H_ _[ℓ]_

_≤_ _C_ _[′]_ ( _∥a∥_ [2] _C_ _[k]_ [ + 1)] _[∥][u][∥]_ _H_ [2] _[ℓ]_ [+] _[ ∥][f]_ _[∥]_ _H_ [2] _[k]_ _[,]_


with _C_ _[′]_ := max _ℓ≤k_ _C_ _ℓ_ a fixed constant depending only on _k_ and _λ_ . Writing the
last inequality in the form

_∥u∥_ [2] _H_ _[ℓ]_ [+1] _[ ≤]_ _[A][∥][u][∥]_ [2] _H_ _[ℓ]_ [+] _[ ∥][f]_ _[∥]_ _L_ [2] [2] _[,]_


it follows by induction on _ℓ_ = 0 _, . . ., k_, that



_∥u∥_ [2] _H_ _[k]_ [+1] _[ ≤]_ _[A]_ _[k]_ _[∥][u][∥]_ _H_ [2] [1] [ +] _[ ∥][f]_ _[∥]_ _L_ [2] [2]



_k−_ 1

_A_ _[ℓ]_ _._

�


_ℓ_ =0



For _k_ = 0, we have the well-known bound _∥u∥_ [2] _H_ [1] _[ ≤]_ _[C]_ _[′′]_ _[∥][f]_ _[∥]_ [2] _L_ [2] [ with] _[ C]_ _[′′]_ [ =] _[ C]_ _[′′]_ [(] _[λ]_ [).]
We may wlog assume _C_ _[′′]_ _≥_ 1. In particular, we then conclude that



_∥u∥_ [2] _H_ _[k]_ [+1] _[ ≤]_ _[C]_ _[′′]_ _[∥][f]_ _[∥]_ _L_ [2] [2]



_k_
� _A_ _[ℓ]_ _,_


_ℓ_ =0



where _A_ := _C_ _[′]_ ( _∥a∥_ _C_ _k_ + 1). Finally, for fixed _k_, we note that since _C_ _[′]_ = _C_ _[′]_ ( _k, λ_ ),
_C_ _[′′]_ = _C_ _[′′]_ ( _λ_ ), there exists a constant _C_ depending only on _k_ and _λ_, but independent
of _∥a∥_ _C_ _k_, such that



_k_
_C_ _[′′]_ �



_k_ _k_
� _A_ _[ℓ]_ = _C_ _[′′]_ _C_ _[′]_ �


_ℓ_ =0 _ℓ_ =0



�(1 + _∥a∥_ [2] _C_ _[k]_ [)] _[ℓ]_ _[≤]_ _[C]_ [(1 +] _[ ∥][a][∥]_ [2] _C_ _[k]_ _[k]_ [)] _[.]_


_ℓ_ =0



For such _C_ = _C_ ( _k, λ_ ) _>_ 0, we conclude that

_∥u∥_ [2] _H_ _[k]_ [+1] _[ ≤]_ _[C][∥][f]_ _[∥]_ [2] _H_ _[k]_ [(1 +] _[ ∥][a][∥]_ [2] _C_ _[k]_ _[k]_ [)] _[,]_


110 ERROR ESTIMATES FOR DEEPONETS


as claimed. 

E.9. **Proof of Theorem 4.22.** To prove Theorem 4.22, we recall some notions
and results from the theory of second order parabolic equations (cp. Lieberman
(1996)). First, let _U_ = [0 _, T_ ] _×_ T _[d]_ denote the domain of the solution of the parabolic
equation (4.31). We denote by (cp. (Lieberman 1996, p. 46))


_C_ _[k,α]_ ( _U_ ) := � _v ∈_ _C_ ( _U_ ) �� _|v|_ ( _k,α_ ) _< ∞_ � _,_ _k ∈_ N _, α ∈_ (0 _,_ 1) _,_


the parabolic H¨older space on _U_, where


_|v|_ ( _k,α_ ) := � sup _|∂_ _x_ _[β]_ _[∂]_ _t_ _[j]_ _[v][|]_ [ + [] _[v]_ []] ( _k,α_ ) _[,]_

_U_
_β_ +2 _j≤k_


and

[ _v_ ] ( _k,α_ ) := � [ _∂_ _x_ _[β]_ _[∂]_ _t_ _[j]_ _[v]_ []] _[α]_ _[,]_

_β_ +2 _j_ = _k_


and where the parabolic H¨older semi-norm [ _w_ ] _α_ of a function _w_ is defined by


_̸_




[ _w_ ] _α_ := sup sup
( _t,x_ ) _∈U_ ( _t_ _[′]_ _,x_ _[′]_ ) _̸_ =( _t,x_ )



_|w_ ( _t, x_ ) _−_ _w_ ( _t_ _[′]_ _, x_ _[′]_ ) _|_

_[.]_

_̸_ ~~�~~ _|x −_ _x_ _[′]_ _|_ + _|t −_ _t_ _[′]_ _|_ [1] _[/]_ [2] ~~[�]~~ ~~_[α]_~~



_̸_


We then have the following Schauder estimate:


_**Lemma**_ **E.1** (Schauder estimate) _**.**_ Let _α ∈_ (0 _,_ 1). For any _k ∈_ N, there exists a
constant _C >_ 0, such that if _v_ is a solution to

_∂_ _t_ _v −_ ∆ _v_ = _f,_


_v_ ( _t_ = 0) = 0 _,_

�


with _f ∈C_ [(] _[k,α]_ [)] ( _U_ ), then


_|v|_ ( _k_ +2 _,α_ ) _≤_ _C|f_ _|_ ( _k,α_ ) + _∥u∥_ _C_ _k_ +2 _,α_ (T _d_ ) _._


_Proof._ For _k_ = 0, this follows e.g. from (Lieberman 1996, Theorem 4.28). For
_k >_ 0, we note that the coefficients in equation _∂_ _t_ _v−_ ∆ _v_ = _f_ are constant, and hence
we can apply the base-case to partial derivatives of the equation (more precisely,
finite-difference approximations thereof, and take the limit). 

We also note the following strong _L_ _[p]_ estimate for parabolic equations:


_**Lemma**_ **E.2** _**.**_ Let _p ∈_ [2 _, ∞_ ). There exists a constant _C >_ 0, such that if _v ∈_
_L_ _[∞]_ ([0 _, T_ ] _×_ T _[d]_ ) is a weak solution of _∂_ _t_ _v −_ ∆ _v_ = _f_, for _f ∈_ _L_ _[∞]_ (T _[d]_ ), and with
initial data _v_ ( _t_ = 0) = _u_, then


_∥v∥_ _W_ 1 _,p_ ([0 _,T_ ] _×_ T _d_ ) _≤_ _C_ � _∥f_ _∥_ _L_ _p_ + _∥u∥_ _W_ 2 _,p_ (T _d_ ) � _._


_Proof._ Theorem 7.32 of Lieberman (1996) provides a sharper estimate, from which
the claim readily follows. 

_**Corollary**_ **E.3** _**.**_ Let _d ∈{_ 2 _,_ 3 _}_ . There exists a constant _C >_ 0, and _α ∈_ (0 _,_ 1),
such that if _f ∈_ _L_ _[∞]_, and _u ∈_ _C_ [1] ([0 _, T_ ] _×_ T _[d]_ ), and if _v_ solves _∂_ _t_ _v −_ ∆ _v_ = _f_, with
initial data _v_ ( _t_ = 0) = _u_, then


_∥v∥_ (0 _,α_ ) _≤_ _C_ � _∥f_ _∥_ _L_ _∞_ + _∥u∥_ _C_ 2 (T _d_ ) � _._


_Proof._ This follows directly from Lemma E.2 and the fact that by Sobolev embedding, we have _W_ [1] _[,p]_ _�→C_ _[α]_, for _α ≤_ 1 _−_ ( _d_ + 1) _/p_ for _p > d_ + 1. 

ERROR ESTIMATES FOR DEEPONETS 111


_Proof of Theorem 4.22._ The claim follows from a bootstrap argument: From the
a priori estimate of Theorem 4.21 for the Allen-Cahn equation, we know that for
initial data _∥u∥_ _L_ _[∞]_ _≤_ 1, we have _∥v_ ( _t_ ) _∥_ _L_ _[∞]_ _≤_ 1 for all _t ∈_ [0 _, T_ ]. In particular, it
follows that _|f_ _|_ = _|f_ ( _v_ ) _| ≤_ _C_ is bounded in _L_ _[∞]_ . Thus, _v_ solves

_∂_ _t_ _v −_ ∆ _v_ = _f,_

(E.4)
_v_ ( _t_ = 0) = _u,_

�


with source term _f ∈_ _L_ _[∞]_ . By the strong _L_ _[p]_ estimate (cp. Corollary E.3), and
the assumed smoothness of _u ∈_ _C_ [4] _[,α]_, it follows that _v ∈C_ _[α]_ ([0 _, T_ ] _×_ T _[d]_ ) for some
_α ∈_ (0 _,_ 1). In turn this implies the following chain of improved regularity based on
the Schauder estimate of Lemma E.1, and using also the fact that _u ∈_ _C_ _[k,α]_ (T _[d]_ ) for
all _k ≤_ 4:


_v ∈C_ _[α]_ ( _U_ ) = _⇒_ _f_ ( _v_ ) _∈C_ _[α]_ ( _U_ ) = _⇒_ _v ∈C_ [2] _[,α]_ ( _U_ )

= _⇒_ _f_ ( _v_ ) _∈C_ [2] _[,α]_ ( _U_ ) = _⇒_ _v ∈C_ [4] _[,α]_ ( _U_ ) _._


Thus, we conclude that if _u ∈_ _C_ [4] _[,α]_ (T _[d]_ ), then we must have _v ∈C_ [4] _[,α]_ ( _U_ ). Furthermore, it follows from the estimates of Corollary E.3, Lemma E.1, that there in fact
exists a constant _σ_ = _σ_ ( _∥u∥_ _C_ 4 _,α_ ) _>_ 0, depending on the _C_ [4] _[,α]_ -norm of the initial
data _u_, such that
_|v|_ (4 _,α_ ) _≤_ _σ_ ( _∥u∥_ _C_ 4 _,α_ (T _d_ ) ) _._

Clearly, we have _∥v∥_ _C_ (4 _,_ 2) ( _U_ ) _≤|v|_ (4 _,α_ ) for any _α >_ 0. The claimed estimate thus
follows. 

E.10. **Proof of Corollary 4.23.**


_Proof._ Since we have _∥v_ ( _t_ ) _∥_ _L_ _∞_ (T _d_ ) _, ∥v_ _[′]_ ( _t_ ) _∥_ _L_ _∞_ (T _d_ ) for all _t ∈_ [0 _, T_ ], the difference
_w_ = _v −_ _v_ _[′]_, solves the following equation


_∂_ _t_ _w_ = ∆ _w_ + _F_ ( _v, v_ _[′]_ ) _w,_


where _F_ ( _a, b_ ) := 1 _−_ _a_ [2] _−_ _ab −_ _b_ [2] . It follows from the uniform boundedness of
_v, v_ _[′]_ that _|F_ ( _v, v_ _[′]_ ) _| ≤_ 4 is uniformly bounded. Multiplying the equation by _w_ and
integrating over _x_, we obtain



_d_

_dt_



ˆ



T _[d]_ _[ w]_ [2] _[ dx][ ≤]_ [8] ˆ



T _[d]_ _[ w]_ [2] _[ dx.]_



And hence, by Gronwall’s inequality, we must have _∥w_ ( _t_ ) _∥_ [2] _L_ [2] _[ ≤∥][w]_ [(0)] _[∥]_ [2] _L_ [2] _[e]_ [8] _[t]_ [ for all]
_t ∈_ [0 _, T_ ]. We conclude that


_∥v_ ( _T_ ) _−_ _v_ _[′]_ ( _T_ ) _∥_ _L_ 2 (T _d_ ) _≤_ _e_ [4] _[T]_ _∥u −_ _u_ _[′]_ _∥_ _L_ 2 (T _d_ ) _._


                       

E.11. **Proof of Theorem 4.27.**


_Proof._ Our main observation is that the local truncation error for the scheme (4.39),
given by


_T_ _j_ _[n]_ [:= (] _[I][ −]_ [∆] _[tD]_ [∆] _[x]_ [)] _[v]_ [(] _[t]_ _[n]_ _[, x]_ _[j]_ [)] _[ −]_ [(] _[v]_ [(] _[t]_ _[n]_ _[, x]_ _[j]_ [) + ∆] _[t]_ � _v_ ( _t_ _n−_ 1 _, x_ _j_ ) _−_ _v_ ( _t_ _n−_ 1 _, x_ _j_ ) [3] [�] _,_


has a Taylor expansion

_T_ _j_ _[n]_ [:= ∆] _[t]_ � _∂v_ ( _t_ _n_ _∂t_ _−_ 1 _, x_ _j_ ) _−_ ∆ _v_ ( _t_ _n−_ 1 _, x_ _j_ ) _−_ ( _v_ ( _t_ _n−_ 1 _, x_ _j_ ) _−_ _v_ ( _t_ _n−_ 1 _, x_ _j_ ) [3] )�

+ ∆ _t R_ (∆ _t,_ ∆ _x_ ) _,_


112 ERROR ESTIMATES FOR DEEPONETS


where similar to Tang & Yang (2016), the remainder term _R_ (∆ _t,_ ∆ _x_ ) can be estimated by



���� _L_ _[∞]_



_d_
�
���� _L_ _[∞]_ [+ ∆] _[x]_ [2] _k_ =1



_∂_ [4] _v_
���� _∂x_ [4] _k_



�



_|R_ (∆ _t,_ ∆ _x_ ) _| ≤_ _C_



�



_∂v_
∆ _t_
���� _∂t_



_∂t_



_∂_ [2] _v_
���� _L_ _[∞]_ [+ ∆] _[t]_ ���� _∂t_ [2]



_≤_ _C_ �∆ _t_ + ∆ _x_ [2] [�] _∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) _,_


where _∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) is defined by (4.32). We note that in contrast to our estimate, the remainder term was bounded in Tang & Yang (2016) by the larger norm
_∥v∥_ _C_ 2 ([0 _,T_ ]; _C_ 4 (T _d_ )) . In view of the available a priori estimates for parabolic equations, the parabolic norm _∥v∥_ _C_ (2 _,_ 4) ([0 _,T_ ] _×_ T _d_ ) appears better adapted to the problem,
and provides a less restrictive convergence result for the scheme. The remainder of
the proof is the same as in (Tang & Yang 2016, Theorem 4.1). 

E.12. **Proof of Lemma 4.28.**


_Proof._ By a result of Yarotsky (Yarotsky 2017, Prop. 2, 3), there exists a constant
_C_ _[′]_ _>_ 0, such that for any _ϵ >_ 0, there exists a ReLU network _×_ [�] : [ _−_ 2 _,_ 2] _→_ R, such
that size( _×_ [�] ) _≤_ _C_ _[′]_ ( _|_ log( _ϵ_ ) _|_ + 1), and


sup _|×_ [�] ( _ξ, η_ ) _−_ _ξη| < ϵ/_ 2 _._
_ξ,η∈_ [ _−_ 2 _,_ 2]


In fact, the mapping constructed in Yarotsky (2017) is based on the identity
_ξη_ = 12 �( _ξ_ + _η_ ) [2] _−_ _ξ_ [2] _−_ _η_ [2] [�], and finding a suitable neural network approximation _f_ _m_ ( _x_ ) _≈_ _x_ [2], of the form (Yarotsky 2017, paragraph above eq. (3)) _f_ _m_ ( _x_ ) =
_x −_ [�] _[m]_ _s_ =1 [2] _[−]_ [2] _[s]_ _[g]_ _[s]_ [(] _[x]_ [), with] _[ m]_ [ =] _[ O]_ [(] _[|]_ [ log(] _[ϵ]_ [)] _[|]_ [) and with] _[ g]_ _[s]_ [ a] _[ s]_ [-fold iteration of the]
following sawtooch function _g_ ( _x_ ):



_g_ ( _x_ ) =



2 _x,_ _x <_ [1] 2
�2(1 _−_ _x_ ) _,_ _x ≥_ [1] 2



2 _[,]_
2(1 _−_ _x_ ) _,_ _x ≥_ [1] _[,]_



2 _[,]_

_g_ _s_ ( _x_ ) = ( _g ◦· · · ◦_ _g_ )

[1] 2 _[,]_ ~~�~~ ~~��~~ ~~�~~

_s_ times



( _x_ ) _._



It is then immediate that Lip( _g_ _s_ ) _≤_ Lip( _g_ ) _[s]_ _≤_ 2 _[s]_, and hence Lip( _f_ _m_ ) _≤_ 1 +
� _ms_ =1 [2] _[−]_ [2] _[s]_ [Lip(] _[g]_ _[s]_ [)] _[ ≤]_ [1 +][ �] _[m]_ _s_ =1 [2] _[−][s]_ _[ ≤]_ [2. It is then readily seen that there exists a]
constant _M >_ 0, independent of _ϵ_, such that


Lip( _×_ [�] : [ _−_ 2 _,_ 2] [2] _→_ ) _≤_ _M._


in addition to the approximation property


sup
_ξ,η∈_ [ _−_ 1 _,_ 1] [2] _[ |][×]_ [�] [(] _[ξ, η]_ [)] _[ −]_ _[ξη][|][ < ϵ/]_ [2] _[.]_


In particular, with the mapping constructed above, we then have sup _η∈_ [ _−_ 1 _,_ 1] _|×_ [�] ( _η, η_ ) _−_
_η_ [2] _| < ϵ_, and from the assumption that _ϵ <_ 1, it follows that _×_ [�] ( _η, η_ ) _−_ 1 _∈_ [ _−_ 2 _,_ 2]
for all _η ∈_ [ _−_ 1 _,_ 1]. Observing that _ξ_ [3] _−_ _ξ_ = _ξ_ ( _ξ_ [2] _−_ 1), we find for any _η ∈_ [ _−_ 1 _,_ 1]

��� _×_ � _η,_ ( _×_ [�] ( _η, η_ ) _−_ 1)� _−_ ( _η_ [3] _−_ _η_ )�� _≤_ ��� _×_ � _η,_ ( _×_ [�] ( _η, η_ ) _−_ 1)� _−_ _η_ ( _×_ [�] ( _η, η_ ) _−_ 1)��


� 2
+ �� _η_ ( _×_ ( _η, η_ ) _−_ 1) _−_ _η_ ( _η_ _−_ 1)��



_≤_ sup
_ξ∈_ [ _−_ 2 _,_ 2]



��� _×_ ( _η, ξ_ ) _−_ _ηξ_ ��



2
+ _|η|_ ��� _×_ ( _η, η_ ) _−_ _η_ ��



_≤_ 2 sup
_ξ,η∈_ [ _−_ 1 _,_ 1]


_< ϵ._



��� _×_ ( _η, ξ_ ) _−_ _ηξ_ ��


ERROR ESTIMATES FOR DEEPONETS 113


To finish the proof, we note that if ( _ξ, η_ ) _�→_ _×_ [�] ( _ξ, η_ ) is represented by a ReLU neural
network of size _≤_ _C_ _[′]_ ( _|_ log( _ϵ_ ) _|_ + 1), then the function

_g_ _ϵ_ ( _η_ ) := max � _−_ 1 _,_ min �1 _,_ _×_ [�] � _η,_ ( _×_ [�] ( _η, η_ ) _−_ 1)���


can be represented by a ReLU neural network of size _≤_ _C_ ( _|_ log( _ϵ_ ) _|_ + 1), where _C_ is
a constant multiple of _C_ _[′]_, and we have _g_ _ϵ_ ( _η_ ) _∈_ [ _−_ 1 _,_ 1] for all _η ∈_ R. Furthermore,
since _η_ [3] _−_ _η ∈_ [ _−_ 1 _,_ 1] for all _η ∈_ [ _−_ 1 _,_ 1], it also follows from the above estimate that

sup _|g_ _ϵ_ ( _η_ ) _−_ ( _η_ [3] _−_ _η_ ) _| ≤_ _ϵ._
_η∈_ [ _−_ 1 _,_ 1]


                       

_**Remark**_ **E.4** _**.**_ If we consider neural networks _g_ with _any smooth_ (e.g. _σ ∈_ _C_ [3] (R))
non-linear activation function _σ_ : R _→_ R, then the previous approximation result
can be considerably improved (Pinkus 1999, Prop. 3.4): Indeed, by assumption on
_σ_ there exists a point _x_ 0 _∈_ R, such that _σ_ _[′′]_ ( _x_ 0 ) _̸_ = 0. Then for _h >_ 0, _η ∈_ [ _−_ 1 _,_ 1],
we have by Taylor expansion
_σ_ ( _x_ 0 + _ηh_ ) _−_ 2 _σ_ ( _x_ 0 ) + _σ_ ( _x_ 0 _−_ _ηh_ ) = _η_ [2] + _O_ ( _h_ ) _,_

_σ_ _[′′]_ ( _x_ 0 ) _h_ [2]


as _h →_ 0, uniformly in _η ∈_ [ _−_ 1 _,_ 1]. In particular, it follows that for smooth _σ_ there
exists a neural network architecture _of fixed size_, such that and for any _h >_ 0, there
is neural network _g_ : [ _−_ 1 _,_ 1] _→_ R, _η �→_ _g_ ( _η_ ), such that _|g_ ( _η_ ) _−_ _η_ [2] _| ≤_ _Ch_ for all
_η ∈_ [ _−_ 1 _,_ 1]. As a consequence, using the representation Yarotsky (2017)

_xy_ = [1]

2 [((] _[x]_ [ +] _[ y]_ [)] [2] _[ −]_ _[x]_ [2] _[ −]_ _[y]_ [2] [)] _[,]_


it follows that there exists a constant _C >_ 0, such that for any _ϵ >_ 0, there exists
a neural network _×_ _ϵ_ : [ _−_ 1 _,_ 1] [2] _→_ R of size size( _×_ _ϵ_ ) _≤_ _C_, such that


sup _| ×_ _ϵ_ ( _x, y_ ) _−_ _xy| < ϵ._
_x,y∈_ [ _−_ 1 _,_ 1]


But then, arguing as in the proof of Lemma 4.28, we find that for any _ϵ >_ 0, the
function _g_ _ϵ_ ( _x_ ) = _×_ _ϵ_ ( _×_ _ϵ_ ( _x, x_ ) _, x_ ) _−_ _x_ can be represented by a neural network, with
a neural network size, size( _g_ _ϵ_ ) _≤_ _C_ _[′]_, which is uniformly bounded in _ϵ >_ 0, and such
that sup _η∈_ [ _−_ 1 _,_ 1] _|g_ _ϵ_ ( _η_ ) _−_ ( _η_ [3] _−_ _η_ ) _| < ϵ_ .


E.13. **Proof of Lemma 4.29.**


_Proof._ By Lemma 4.28, the non-linearity in (4.40) can be represented by a neural
network with size bounded by


size( _g_ _ϵ_ ) _≤_ _C_ (1 + _|_ log( _ϵ_ ) _|_ ) _,_ depth( _g_ _ϵ_ ) _≤_ _C_ (1 + _|_ log( _ϵ_ ) _|_ ) _._


It follows that the mapping

_U_ � 1 _[k]_ _[, . . .,]_ [ �] _[U]_ _m_ _[ k]_ _�→_ _g_ _ϵ_ _U_ � 1 _[k]_ _, . . ., g_ _ϵ_ _U_ � _m_ _[k]_ =: _G_ _ϵ_ ( _U_ [�] _[k]_ ) _,_
� � � � � � ��


can be represented by a neural network _G_ _ϵ_ : R _[m]_ _→_ R _[m]_, with


size( _G_ _ϵ_ ) = _O_ ( _m|_ log( _ϵ_ ) _|_ ) _,_ depth( _G_ _ϵ_ ) = _O_ ( _|_ log( _ϵ_ ) _|_ ) _._


Since the identity mapping _U_ [�] _[k]_ _�→_ _U_ [�] _[k]_ can be represented by a ReLU network with
size = _O_ ( _m_ ), depth = _O_ (1), there exists a neural network with size = _O_ ( _m|_ log( _ϵ_ ) _|_ ),
depth = _O_ ( _|_ log _ϵ|_ ), which represents

_U_ � _[k]_ _�→_ _U_ � _[k]_ + ∆ _tG_ _ϵ_ ( _U_ � _[k]_ ) _._


Finally, we note that
_U_ � _[k]_ + ∆ _tG_ _ϵ_ ( _U_ � _[k]_ ) _�→_ _R_ ∆ _x,_ ∆ _t_ ( _U_ � _[k]_ + ∆ _tG_ _ϵ_ ( _U_ � _[k]_ )) _,_


114 ERROR ESTIMATES FOR DEEPONETS


is simply a matrix-vector multiplication of a _fixed_ matrix _R_ ∆ _x,_ ∆ _t_ = ( _I−_ ∆ _t D_ ∆ _x_ ) _[−]_ [1] _∈_
R _[m][×][m]_ (independent of _U_ [�] _[k]_ ), with a vector in R _[m]_ . This operation can be represented
by a ReLU neural network layer with size = _O_ ( _m_ [2] ), depth = _O_ (1). We conclude
that the composition


� � � � �
_U_ _[k]_ _�→_ _U_ _[k]_ + ∆ _t G_ _ϵ_ ( _U_ _[k]_ ) _�→_ _R_ ∆ _x,_ ∆ _t_ _U_ _[k]_ + ∆ _t G_ _ϵ_ ( _U_ _[k]_ ) = _U_ [�] _[k]_ [+1] _,_
� �

can be represented by a neural network _N_ [�] : R _[m]_ _→_ R _[m]_ with

size( _N_ [�] ) = _O_ ( _m_ [2] + _m|_ log _ϵ|_ ) _,_ depth( _N_ [�] ) = _O_ ( _|_ log _ϵ|_ ) _._


Since _U_ [�] [0] _�→_ _U_ [�] [1] _�→· · · �→_ _U_ [�] _[n]_ involves the composition of _n_ such steps, we
conclude that _U_ [�] [0] _�→_ _U_ [�] _[n]_ can be represented by a neural network _N_ = _N ◦_ [�] _N ◦· · ·◦_ [�] _N_ [�] :
R _[m]_ _→_ R _[m]_ ( _n_ iterations) with

size( _N_ ) = _O_ ( _n_ ( _m_ [2] + _m|_ log _ϵ|_ )) _,_ depth( _N_ ) = _O_ ( _n|_ log( _ϵ_ ) _|_ ) _._


                       

E.14. **Proof of Lemma 4.30.**


_Proof._ We note that


_U_ _[k]_ [+1] _−_ _U_ [�] _[k]_ [+1] = _R_ ∆ _x,_ ∆ _t_ _U_ _[k]_ _−_ _U_ [�] _[k]_ + ∆ _t_ _g_ _ϵ_ ( _U_ _[k]_ ) _−_ _g_ _ϵ_ ( _U_ [�] _[k]_ )
� � ��

+ ∆ _tR_ ∆ _x,_ ∆ _t_ �� _g_ _ϵ_ ( _U_ _[k]_ ) _−_ _f_ ( _U_ _[k]_ )�� _._


We note that _∥U_ _[k]_ _∥_ _ℓ_ _∞_ _≤_ 1 for all _k >_ 0, by Theorem 4.26. Furthermore, it has been
shown in Tang & Yang (2016) that _∥R_ ∆ _x,_ ∆ _t_ _∥_ _ℓ_ _∞_ _→ℓ_ _∞_ _≤_ 1. It follows that
�� _R_ ∆ _x,_ ∆ _t_ �� _g_ _ϵ_ ( _U_ _[k]_ ) _−_ _f_ ( _U_ _[k]_ )���� _ℓ_ _[∞]_ _[≤]_ sup _|g_ _ϵ_ ( _η_ ) _−_ _f_ ( _η_ ) _| ≤_ _ϵ._
_η∈_ [ _−_ 1 _,_ 1]


Denote now _E_ _[k]_ = _∥U_ _[k]_ _−_ _U_ [�] _[k]_ _∥_ _ℓ_ _∞_, for _k_ = 0 _, . . ., n_ . Then

_E_ _[k]_ [+1] _≤∥R_ ∆ _t,_ ∆ _x_ _∥_ _ℓ_ _∞_ _→ℓ_ _∞_ [�] _E_ _[k]_ + ∆ _t_ Lip( _g_ _ϵ_ ) _E_ _[k]_ [�] + ∆ _t ϵ_

_≤_ (1 + ∆ _t_ Lip( _g_ _ϵ_ )) _E_ _[k]_ + ∆ _t ϵ._


Summing over _k_ = 1 _, . . ., ℓ_, and taking into account that _E_ [0] = 0, we find for any
_ℓ_ _∈{_ 1 _, . . ., n}_ that



_E_ _[ℓ]_ = _E_ [0] +



_ℓ−_ 1
�[ _E_ _[k]_ [+1] _−_ _E_ _[k]_ ]


_k_ =0



_≤_ _Tϵ_ + ∆ _t_



_ℓ−_ 1
� Lip( _g_ _ϵ_ ) _E_ _[k]_ _._


_k_ =1



By the Gronwall inequality, it follows that


_E_ _[n]_ _≤_ _ϵ T_ exp(Lip( _g_ _ϵ_ ) _T_ ) _._


The result follows from Lip( _g_ _ϵ_ ) _≤_ _M_ . 

E.15. **Proof of Proposition 4.31.**


_Proof._ By definition, the approximator neural network _A_ should provide an approximation _A ≈P ◦G ◦D_ . Our goal is to construct _A_ based on the neural network approximation (4.40) of the convergent numerical scheme (4.39). To
this end, we first note that by Lemmas 4.29, 4.30, there exists a constant _C_ =
_C_ (sup _u∈_ supp( _µ_ ) _∥u∥_ _C_ (2 _,_ 4) _, T_ ) _>_ 0, independent of _m_, _n_ and _ϵ_, and a neural network
_N_ with


size( _N_ ) _≤_ _C_ (1 + _n_ ( _m_ [2] + _m|_ log( _ϵ_ ) _|_ )) _,_ depth( _N_ ) _≤_ _C_ (1 + _n|_ log( _ϵ_ ) _|_ ) _,_


ERROR ESTIMATES FOR DEEPONETS 115


such that for any _u ∈_ supp( _µ_ ), we have


max
_j_ =1 _,...,m_ _[|N]_ _[j]_ [(] _**[u]**_ [)] _[ −]_ _[v]_ [(] _[x]_ _[j]_ _[, T]_ [)] _[| ≤]_ _[C]_ [(] _[ϵ]_ [ + ∆] _[x]_ [2] [ +] _[ T/n]_ [)] _[,]_


where _**u**_ = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) = _E_ ( _u_ ). We note that for the present choice of the
_x_ _j_ as nodes on an equidistant grid, we have ∆ _x ∼_ _m_ _[−]_ [2] _[/d]_ . Thus, we choose _n_ =
_⌈Tm_ [2] _[/d]_ _⌉_, _ϵ_ = _m_ _[−]_ [2] _[/d]_ to conclude that there exists a neural network _N_ : R _[m]_ _→_ R _[m]_,
such that


size( _N_ ) _≤_ _C_ (1 + _m_ [2+2] _[/d]_ ) _,_ depth( _N_ ) _≤_ _C_ (1 + _m_ [2] _[/d]_ log( _m_ )) _,_ (E.5)


and


max (E.6)
_j_ =1 _,...,m_ _[|N]_ _[j]_ [(] _**[u]**_ [)] _[ −]_ _[v]_ [(] _[x]_ _[j]_ _[, T]_ [)] _[| ≤]_ _[Cm]_ _[−]_ [2] _[/d]_ _[.]_


Given the grid points _x_ _j_, we note that we can define a linear mapping


_L_ : R _[m]_ _→_ _C_ ( _D_ ) _,_ _**v**_ = ( _v_ 1 _, . . ., v_ _m_ ) _�→L_ ( _**v**_ )( _x_ ) _,_


by employing local linear interpolation of the values _v_ _j_ on the mesh _x_ _j_, i.e. such
that� _L_ ( _**v**_ )( _x_ _j_ ) = _v_ _j_ . Given the projection operator _P_, we define a linear mapping�
_L_ : R _[m]_ _→_ R _[p]_ by � _L_ ( _**v**_ ) := _P ◦L_ ( _**v**_ ). With these definitions, we now set _A_ := � _L◦N_ =
_P ◦L ◦_ � _N_ [�] . We note that since _L_ [�] = _P ◦L_ is a linear mapping R _[m]_ _→_ R _[p]_, and since
_N_ is a neural network with size bounded by (E.5), we can represent _A_ as a neural
network with


size( _A_ ) = _O_ ( _m_ [2+2] _[/d]_ + _mp_ ) _,_

depth( _A_ ) = _O_ ( _m_ [2] _[/d]_ log( _m_ ) + 1) = _O_ ( _m_ [2] _[/d]_ log( _m_ )) _._


Furthermore, for this choice of _A_, we have


( _E_ [�] _A_ ) [2] = _ℓ_ [2] _[ d]_ [(] _[E]_ [#] _[µ]_ [)(] _**[u]**_ [)]
ˆ R _[m]_ _[ ∥A]_ [(] _**[u]**_ [)] _[ −P ◦G ◦D]_ [(] _**[u]**_ [)] _[∥]_ [2]


= _∥A ◦E_ ( _u_ ) _−P ◦G ◦D ◦E_ ( _u_ ) _∥_ [2] _ℓ_ [2] _[ dµ]_ [(] _[u]_ [)]
ˆ supp( _µ_ )


= _∥P ◦L ◦_ _N ◦E_ [�] ( _u_ ) _−P ◦G ◦D ◦E_ ( _u_ ) _∥_ [2] _ℓ_ [2] _[ dµ]_ [(] _[u]_ [)]
ˆ supp( _µ_ )


Furthermore, we can estimate the integrand for any _u ∈_ supp( _µ_ ):


_∥P ◦L ◦_ _N ◦E_ [�] ( _u_ ) _−P ◦G ◦D ◦E_ ( _u_ ) _∥_ _ℓ_ 2

_≤∥P∥_ _L_ 2 ( _U_ ) _→ℓ_ 2 (R _p_ ) _∥L ◦_ _N ◦E_ [�] ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ 2 ( _U_ )

_≤∥L ◦_ _N ◦E_ [�] ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) _._


If we denote by _x �→_ _v_ ( _x, T_ ) = _G_ ( _u_ ) the solution at _t_ = _T_ of the Allen-Cahn
equation (4.31) with initial data _u_, and _**u**_ = ( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )) = _E_ ( _u_ ), then we
have


_∥L ◦_ _N ◦E_ [�] ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ 2 ( _U_ )

_≤∥L ◦_ _N_ [�] ( _**u**_ ) _−G_ ( _u_ ) _∥_ _L_ 2 ( _U_ ) + _∥G_ ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ 2 ( _U_ )

_≤_ _C∥L ◦_ _N_ [�] ( _**u**_ ) _−G_ ( _u_ ) _∥_ _L_ _∞_ ( _U_ ) + _C∥G_ ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ _∞_ ( _U_ ) _._
(E.7)


To estimate the first term, we note that since _L ◦_ _N_ [�] ( _**u**_ ) and _v_ ( _·, T_ ) = _G_ ( _u_ ) are
Lipschitz continuous functions with Lipschitz constant that can be bounded by


116 ERROR ESTIMATES FOR DEEPONETS


_∥u∥_ _C_ 4 _,α_ for a fixed _α ∈_ (0 _,_ 1), as in Theorem 4.22, we have


_∥L ◦_ _N_ [�] ( _**u**_ ) _−G_ ( _u_ ) _∥_ _L_ _∞_ ( _U_ ) _≤_ _C_ ( _∥u∥_ _C_ 4 _,α_ )∆ _x_


+ max _N_ ( _**u**_ )( _x_ _j_ ) _−G_ ( _u_ )( _x_ _j_ ) _|_
_j_ =1 _,...,m_ _[|L ◦]_ [�]


= _C_ ( _∥u∥_ _C_ 4 _,α_ )∆ _x_ + max _N_ _j_ ( _**u**_ ) _−_ _v_ ( _x_ _j_ _, T_ ) _|_
_j_ =1 _,...,m_ _[|]_ [ �]

_≤_ _Cm_ _[−]_ [1] _[/d]_ _._


In the last step, we used the fact that ∆ _x_ ≲ _m_ _[−]_ [1] _[/d]_ and (E.6).
To estimate the other second term in (E.7), we note that the numerical scheme
(4.39) applied to _u_ and _D ◦E_ ( _u_ ) starts from the same discrete initial data _E_ ( _u_ ) =
( _u_ ( _x_ 1 ) _, . . ., u_ ( _x_ _m_ )), since _E ◦D_ = Id, by assumption. It then follows from the error
estimate of Theorem 4.27, and the fact that the Lipschitz constant of _G_ ( _u_ ) and
_G ◦D ◦E_ ( _u_ ) are bounded in terms of _∥u∥_ _C_ 4 _,α_, that


_∥G_ ( _u_ ) _−G ◦D ◦E_ ( _u_ ) _∥_ _L_ _∞_ _≤_ _Cm_ _[−]_ [1] _[/d]_ + max
_j_ =1 _,...,m_ _[|G]_ [(] _[u]_ [)(] _[x]_ _[j]_ [)] _[ −G ◦D ◦E]_ [(] _[u]_ [)(] _[x]_ _[j]_ [)] _[|]_

_≤_ _Cm_ _[−]_ [1] _[/d]_ + _C_ ∆ _x_ [2] _≤_ _Cm_ _[−]_ [1] _[/d]_ _._


The constant _C_ = _C_ ( _u_ ) _>_ 0 depends on _u_ only through _∥u∥_ _C_ 4 _,α_ . In particular,
there exists a constant _C_ _[′]_ _>_ 0, such that _C_ ( _∥u∥_ _C_ 4 _,α_ _≤_ _C_ _[′]_ for all _u ∈_ supp( _µ_ ).
Combining the above estimates, we can now estimate


�
_E_ _A_ _≤_ _C_ _[′]_ _m_ _[−]_ [1] _[/d]_ _._


To conclude, we have shown that there exists an approximator network _A_ : R _[m]_ _→_
R _[p]_, for _p_ = _m_, such that


size( _A_ ) = _O_ ( _m_ [2+2] _[/d]_ + _mp_ ) _,_ depth( _A_ ) = _O_ ( _m_ [2] _[/d]_ log( _m_ )) _,_


and _E_ [�] _A_ = _O_ ( _m_ _[−]_ [1] _[/d]_ ). 

E.16. **Proof of Lemma 4.37.**


_Proof._ Let _σ_ ( _x_ ) = max( _x,_ 0) be the ReLU activation function. We assume wlog
that _ϵ < b −_ _a_ (otherwise decrease _ϵ_ ). We note that

_χ_ _[ϵ]_ [ _a,b_ ] [(] _[x]_ [) :=] _[σ]_ [(] _[x][ −]_ _[a]_ [)] _[ −]_ _[σ]_ [(] _[x][ −]_ _[a][ −]_ _[ϵ][/]_ [2][)] _ϵ/_ [ +] 2 _[ σ]_ [(] _[x][ −]_ _[b]_ [)] _[ −]_ _[σ]_ [(] _[x][ −]_ _[b][ −]_ _[ϵ][/]_ [2][)] _,_


is continuous, satisfies



_χ_ _[ϵ]_ [ _a,b_ ] [(] _[x]_ [) =]



0 _,_ _x /∈_ [ _a, b_ ] _,_
�1 _,_ _x ∈_ [ _a_ + _ϵ/_ 2 _, b −_ _ϵ/_ 2] _,_



and is linear on [ _a, a_ + _ϵ/_ 2] and [ _b −_ _ϵ/_ 2 _, b_ ]. In particular, it follows that


_∥χ_ _[ϵ]_ [ _a,b_ ] _[−]_ [1] [[] _[a,b]_ []] _[∥]_ _[L]_ [1] [(][R][)] _[≤|]_ [[] _[a, a]_ [ +] _[ ϵ/]_ [2]] _[|]_ [ +] _[ |]_ [[] _[b][ −]_ _[ϵ/]_ [2] _[, b]_ []] _[|]_ [ =] _[ ϵ.]_


Furthermore, _χ_ _[ϵ]_ [ _a,b_ ] [is represented by the same neural network architecture for any]
choice of _a, b, ϵ_ . 

E.17. **Proof of Theorem 4.38.**


_Proof._ By Theorem 4.35, there exists a constant _C >_ 0, and a neural network
_N_ : R _[m]_ _→_ R _[m]_, with


size( _N_ ) _≤_ _Cm_ _[−]_ [5] _[/]_ [2] _,_ depth( _N_ ) _≤_ _Cm,_ (E.8)


ERROR ESTIMATES FOR DEEPONETS 117



such that for any _u ∈_ BV _M_, we have


_m_

_G_ ( _u_ ) _−_ � _N_ _j_ (
������ _j_ =1



������ _L_ [1] ( _D_ )



_m_
�



� _N_ _j_ ( _E_ ( _u_ )) 1 _C_ _j_ ( _·_ )

_j_ =1



_≤_ _m_ _[C]_ _[α]_ _[.]_



We now define _A_ : R _[m]_ _→_ R _[p]_ by _A_ ( _**u**_ ) := ( _N_ 1 ( _**u**_ ) _, . . ., N_ _p_ ( _**u**_ )), where we formally
set _N_ _j_ _≡_ 0, if _j > m_ . For each _j_ = 1 _, . . ., m_, let _τ_ _j_ ( _y_ ) := _χ_ _[ϵ]_ _C_ _j_ [, where] _[ χ]_ _C_ _[ϵ]_ _j_ [is a neural]
network approximation of 1 _C_ _j_ as in Lemma 4.37, i.e. such that


size( _τ_ _j_ ) _≤_ _C,_ depth( _τ_ _j_ ) = 1 _,_ (E.9)


such that
_∥τ_ _j_ _−_ 1 _C_ _j_ _∥_ _L_ 1 _≤_ _ϵ,_ for _j_ = 1 _, . . ., m_ .
For _j > m_, we define _τ_ _j_ _≡_ 0, corresponding to an approximation of _C_ _j_ := _∅_ for
_j > m_ . Note that we clearly obtain from (E.9) for the trunk net _**τ**_ = ( _τ_ 1 _, . . ., τ_ _p_ ):


size( _**τ**_ ) _≤_ _Cp,_ depth( _**τ**_ ) = 1 _._ (E.10)


Similarly, we define for _j_ = 1 _, . . ., p_ the branch net _**β**_ = ( _β_ 1 _, . . ., β_ _p_ ) by:



_β_ _j_ ( _u_ ) :=



_N_ _j_ ( _E_ ( _u_ )) _,_ ( _j ≤_ _m_ )
�0 _,_ ( _j > m_ ) _._



By (E.8), we note that


size( _**β**_ ) _≤_ _Cm_ _[−]_ [5] _[/]_ [2] _,_ depth( _**β**_ ) _≤_ _Cm,_ (E.11)



Then, clearly we have


_p_

_G_ ( _u_ ) _−_ �
������ _j_ =1



_G_ ( _u_ ) _−_
������



_p_
�



_≤_
������ _L_ [1]



������ _L_ [1]



� _β_ _j_ ( _u_ ) _τ_ _j_

_j_ =1



_m_
� _N_ _j_ ( _E_ ( _u_ ))1 _C_ _j_

_j_ =1



+


+



_p_
� _|N_ _j_ ( _E_ ( _u_ )) _|_ ��1 _C_ _j_ _−_ _τ_ _j_ �� _L_ [1]

_j_ =1


_m_
� _|N_ _j_ ( _E_ ( _u_ )) _|_ ��1 _C_ _j_ �� _L_ [1]

_j_ = _p_ +1



The first term can be estimated by _Cm_ _[−][α]_ . Each term in the sum for _j_ = 1 _, . . ., p_
can be estimated by



�



_∥_ 1 _C_ _j_ _−_ _τ_ _j_ _∥_ _L_ 1 =



_∥_ 1 _C_ _j_ _−_ _τ_ _j_ _∥_ _L_ 1 _,_ ( _j_ = 1 _, . . ., m_ ) _,_
�0 _,_ ( _j > m_ )



_≤_ _ϵ,_



by choice of the _τ_ _j_ . We also note that _|N_ _j_ ( _E_ ( _u_ )) _| ≤_ _M_ for all _u ∈_ BV _M_ (cp.
Theorem 4.35). Hence, we have


_p_
� _|N_ _j_ ( _E_ ( _u_ )) _|_ ��1 _C_ _j_ _−_ _τ_ _j_ �� _L_ [1] _[ ≤]_ _[Mpϵ.]_

_j_ =1


Finally, the last sum over _j_ = _p_ + 1 _, . . ., m_, if non-empty, can be estimated by



_m_
� _|N_ _j_ ( _E_ ( _u_ )) _|_ ��1 _C_ _j_ �� _L_ [1] _[ ≤]_

_j_ = _p_ +1



_m_
�

_j_ = _p_ +1



2 _πM_


_m_



= max( _m −_ _p,_ 0) [2] _[πM]_

_m_



= 2 _πM_ max 1 _−_ _[p]_ _._
� _m_ _[,]_ [ 0] �


118 ERROR ESTIMATES FOR DEEPONETS



In particular, we conclude that there exists a constant _C >_ 0, independent of _p_ and
_m_, such that for any _ϵ >_ 0, there exists a DeepONet ( _**β**_ _,_ _**τ**_ ) with size bounded by
(E.11) and (E.10), such that


_p_

_G_ ( _u_ ) _−_ � _β_ _j_ ( _u_ ) _τ_ _j_ _≤_ _Cm_ _[−][α]_ + _Cpϵ_ + _C_ max �1 _−_ _m_ _[p]_ _[,]_ [ 0] � _,_ _∀u ∈_ BV _M_ _._
������ _j_ =1 ������ _L_ [1]


Since _ϵ >_ 0 is arbitrary, we may set _ϵ_ = _m_ _[−][α]_ _p_ _[−]_ [1], and absorb the second term in
the first. Integrating against _µ_ with _µ_ (BV _M_ ) = 1, we find that
ˆ _L_ [1] _D_ �� _G_ ( _u_ ) _−R ◦A ◦E_ ( _u_ )�� _L_ [1] _[ dµ]_ [(] _[u]_ [)] _[ ≤]_ _[Cm]_ _[−][α]_ [ +] _[ C]_ [ max] �1 _−_ _m_ _[p]_ _[,]_ [ 0] � _,_



_p_
�



� _β_ _j_ ( _u_ ) _τ_ _j_

_j_ =1



_≤_ _Cm_ _[−][α]_ + _Cpϵ_ + _C_ max 1 _−_ _[p]_
�
������ _L_ [1]




_[p]_ _,_ _∀u ∈_ BV _M_ _._

_m_ _[,]_ [ 0] �



_L_ [1] ( _D_ )



�� _G_ ( _u_ ) _−R ◦A ◦E_ ( _u_ )�� _L_ [1] _[ dµ]_ [(] _[u]_ [)] _[ ≤]_ _[Cm]_ _[−][α]_ [ +] _[ C]_ [ max] �1 _−_ _[p]_




_[p]_ _,_

_m_ _[,]_ [ 0] �



where we recall that, by definition, we have



_R ◦A ◦E_ ( _u_ ) _≡_



_p_
� _β_ _j_ ( _u_ ) _τ_ _j_ _._

_j_ =1



The claimed estimate on _E_ [�] _L_ 1 thus follows with trunk and branch net complexity
bounds (E.10) and (E.11). 

E.18. **Proof of Lemma 4.41.**


_Proof._ We first note that the covariance operator Γ _G_ # _µ_ can be represented in the
form

2 _π_
(Γ _G_ # _µ_ _u_ )( _x_ ) = _k_ ( _x, x_ _[′]_ ) _u_ ( _x_ _[′]_ ) _dx_ _[′]_ _,_
ˆ 0


where


_k_ ( _x, x_ _[′]_ ) = _u_ ( _x_ ) _u_ ( _x_ _[′]_ ) _d_ ( _G_ # _µ_ )( _u_ )
ˆ _L_ [2] (T)


= _G_ ( _u_ )( _x_ ) _G_ ( _u_ )( _x_ _[′]_ ) _dµ_ ( _u_ ) _._
ˆ _L_ [2] (T)



Next, we note that for any functional _F ∈_ _L_ [1] ( _µ_ ), we have

_F_ ( _u_ ) _dµ_ ( _u_ ) = [1] 2 _π_ _F_ ( _u_ ( _· −_ _x_

ˆ _L_ [2] T 2 _π_ ˆ 0



_F_ ( _u_ ) _dµ_ ( _u_ ) = [1]
_L_ [2] (T) 2 _π_



2 _π_



ˆ 0 2 _π_



� �
_F_ ( _u_ ( _· −_ _x_ )) _dx,_
0



� �
and the solution with initial data _u_ 0 ( _x −_ _x_ ) at _t_ = _π/_ 2 is given by _G_ ( _u_ ( _· −_ _x_ )) =

�
_v_ _t_ ( _x −_ _x_ ). It follows that


2 _π_


� � �

_G_ ( _u_ )( _x_ ) _G_ ( _u_ )( _x_ _[′]_ ) _dµ_ ( _u_ ) = [1] _G_ ( _u_ ( _· −_ _x_ ))( _x_ ) _G_ ( _u_ ( _· −_ _x_ ))( _x_ _[′]_ ) _dx_

ˆ _L_ [2] T 2 _π_ ˆ 0



_G_ ( _u_ )( _x_ ) _G_ ( _u_ )( _x_ _[′]_ ) _dµ_ ( _u_ ) = [1]
_L_ [2] (T) 2 _π_



2 _π_



ˆ 0 2 _π_



� � �
_G_ ( _u_ ( _· −_ _x_ ))( _x_ ) _G_ ( _u_ ( _· −_ _x_ ))( _x_ _[′]_ ) _dx_
0



= [1]

2 _π_


By a change of variables, we thus find



2 _π_


� � �
_v_ _t_ ( _x −_ _x_ ) _v_ _t_ ( _x_ _[′]_ _−_ _x_ ) _dx._

ˆ 0



_k_ ( _x, x_ _[′]_ ) = [1]

2 _π_


= [1]

2 _π_



2 _π_


� � �
_v_ _t_ ( _x −_ _x_ ) _v_ _t_ ( _x_ _[′]_ _−_ _x_ ) _dx_

ˆ 0


2 _π_

_v_ _t_ ( _x −_ _x_ _[′]_ + _ξ_ ) _v_ _t_ ( _ξ_ ) _dξ_

ˆ 0



= _g_ ( _x −_ _x_ _[′]_ ) _,_



where



_g_ ( _x_ ) := [1]

2 _π_



2 _π_

_v_ _t_ ( _x_ + _ξ_ ) _v_ _t_ ( _ξ_ ) _dξ,_

ˆ 0


ERROR ESTIMATES FOR DEEPONETS 119


is written as a convolution. In particular, _k_ ( _x, x_ _[′]_ ) = _g_ ( _x_ _−_ _x_ _[′]_ ) is a _stationary_ kernel.
From the stationarity of _k_ ( _x, x_ _[′]_ ), it follows that the eigenfunctions of _k_ ( _x, x_ _[′]_ ) are
given by the Fourier basis _{_ **e** _k_ _}_ _k∈_ Z, with corresponding eigenvalues


�
_λ_ _k_ = (2 _π_ ) _g_ ( _k_ ) _,_


where � _g_ ( _k_ ) denotes the _k_ -th Fourier coefficient of _g_ . Finally, we note that



� � 1 1
_λ_ _k_ = (2 _π_ ) _g_ ( _k_ ) = _|v_ _t_ ( _k_ ) _|_ [2] = _π_ [2] _k_ [2] [+] _[ o]_ � _k_ [2]



_._
�




                       

Appendix F. Proof of Theorem 5.3


We note that, since _N_ [�] is a minimizer of _L_ [�], and _N_ [�] _N_ is a minimizer of _L_ [�] _N_, we
have the following well-known bound:

� � � �
_L_ _N_ _N_ _−_ _L_ [�] _N_ = � _L_ _N_ _N_ _−_ _L_ [�] _N_
����� � � ���� � � � �


� �
_≤_ _L_ [�] _N_ _N_ _−_ _L_ [�] _N_ _N_ _N_
� � � �


� �
+ _L_ [�] _N_ _N_ _−_ _L_ [�] _N_
� � � �



_≤_ 2 sup
_θ_



�
_L_ _N_ ( _N_ _θ_ ) _−_ _L_ ( _N_ _θ_ ) _,_
���� ���



where the supremum is taken over all admissible _θ ∈_ [ _−B, B_ ] _[d]_ _[θ]_ . Starting from this
bound, the proof of Theorem 5.3 relies on the following lemmas, which follow very
closely the argument in (Welti 2020, Chapter 5.3) (see also Cucker & Smale (2002),
Berner et al. (2020)).


_**Lemma**_ **F.1** _**.**_ Under assumptions 5.1 and 5.2, we have



� Ψ( _Z_ _j_ )Φ( _Z_ _j_ )

_j_ =1







_∥θ −_ _θ_ _[′]_ _∥_ _ℓ_ _∞_ _._




�� _S_ _θN_ _[−]_ _[S]_ _θ_ _[N]_ _[′]_ �� _≤_ 4

_N_







_N_



�
 _j_ =1



_Proof._ We have


_|S_ _θ_ _[N]_ _[−]_ _[S]_ _θ_ _[N]_ _[′]_ _[ | ≤]_ [1]

_N_


_≤_ [1]

_N_


_≤_ [1]

_N_


= [4]

_N_



_N_



�
 _j_ =1



_N_
�

_j_ =1



2 2
�� _|G_ ( _Z_ _j_ ) _−N_ _θ_ ( _Z_ _j_ ) _|_ _−|G_ ( _Z_ _j_ ) _−N_ _θ_ _′_ ( _Z_ _j_ ) _|_ ��



_N_
� (2 _|G_ ( _Z_ _j_ ) _|_ + _|N_ _θ_ ( _Z_ _j_ ) _|_ + _|N_ _θ_ _′_ ( _Z_ _j_ ) _|_ ) _|N_ _θ_ ( _Z_ _j_ ) _−N_ _θ_ _′_ ( _Z_ _j_ ) _|_

_j_ =1


_N_
� 4 _|_ Ψ( _Z_ _j_ ) _||_ Φ( _Z_ _j_ ) _|∥θ −_ _θ_ _[′]_ _∥_ _ℓ_ _∞_

_j_ =1







� _|_ Ψ( _Z_ _j_ ) _||_ Φ( _Z_ _j_ ) _|_

_j_ =1







_∥θ −_ _θ_ _[′]_ _∥_ _ℓ_ _∞_ _,_




as claimed. 

_**Lemma**_ **F.2** _**.**_ If _θ_ 1 _, . . ., θ_ _K_ are such that for all _θ ∈_ [ _−B, B_ ] _[d]_, there exists _j_ with
_∥θ −_ _θ_ _j_ _∥_ _ℓ_ _∞_ _≤_ _ϵ_, then



_p_ [�] [1] _[/p]_
_N_
��� _S_ _θ_ _j_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ _j_ []] ��� _._



E



�



sup
_θ∈_ [ _−B,B_ ] _[d]_



1 _/p_
�� _S_ _θN_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] �� _p_
�



_≤_ 8 _ϵ_ E [ _|_ ΨΦ _|_ _[p]_ ] [1] _[/p]_ + E max
� _j_ =1 _,...,K_


120 ERROR ESTIMATES FOR DEEPONETS


_Proof._ Fix _ϵ >_ 0. Define a mapping _j_ : [ _−B, B_ ] _[d]_ _→_ N, by _j_ ( _θ_ ) = min _{j ∈{_ 1 _, . . ., K} | |θ −_ _θ_ _j_ _| ≤_ _ϵ}_ .
Then, we have



_≤_ E


_≤_ E



1 _/p_
sup _θ_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] _[|]_ _[p]_
_θ∈_ [ _−B,B_ ] _[d]_ _[ |][S]_ _[N]_ �



E



�



��



��



sup
_θ∈_ [ _−B,B_ ] _[d]_



max
_j_ =1 _,...,K_



��� _S_ _θN_ _[−]_ _[S]_ _θ_ _[N]_ _j_ ( _θ_ ) ��� + ��� _S_ _jN_ ( _θ_ ) _[−]_ [E][[] _[S]_ _j_ _[N]_ ( _θ_ ) []] ���



+ ���E[ _S_ _θN_ _j_ ( _θ_ ) []] _[ −]_ [E][[] _[S]_ _θ_ _[N]_ []] ���



� _p_ � 1 _/p_



��� _S_ _jN_ ( _θ_ ) _[−]_ [E][[] _[S]_ _j_ _[N]_ ( _θ_ ) []] ���







1 _/p_









+ [8] _[ϵ]_

_N_







_N_



�
 _j_ =1



� _p_ [] 



_≤_ E max
� _j_ =1 _,...,K_



� _|_ Ψ( _Z_ _j_ ) _||_ Φ( _Z_ _j_ ) _|_

_j_ =1


_p_ [�] [1] _[/p]_
��� _S_ _θN_ _j_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ _j_ []] ���



+ [8] _[ϵ]_

_N_



_N_
� E [ _|_ Ψ( _Z_ _j_ )Φ( _Z_ _j_ ) _|_ _[p]_ ] [1] _[/p]_

_j_ =1



= E max
� _j_ =1 _,...,K_



_p_ [�] [1] _[/p]_
��� _S_ _θN_ _j_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ _j_ []] ���



+ 8 _ϵ_ E [ _|_ Ψ( _Z_ 1 )Φ( _Z_ 1 ) _|_ _[p]_ ] [1] _[/p]_ _,_


as claimed. 

_**Lemma**_ **F.3** _**.**_ Let _K ∈_ N and let _θ_ 1 _, . . ., θ_ _K_ _∈_ [ _−B, B_ ] _[d]_ _[θ]_ be given. Then, for any
_p ≥_ 1, we have



E max
� _j_ =1 _,...,K_



_p_ [�] [1] _[/p]_ _p_ [�] 1 _/p_
_N_ _N_
��� _S_ _θ_ _j_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ _j_ []] ��� _≤_ _K_ [1] _[/p]_ _j_ =1 max _,...,K_ [E] ���� _S_ _θ_ _j_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ _j_ []] ��� _._



_Proof._ This follows readily from the fact that for any measurable _X_ 1 _, . . ., X_ _K_, we
have



� _|X_ _j_ _|_ _[p]_

_j_ =1



E max _≤_ E
� _j_ =1 _,...,K_ _[|][X]_ _[j]_ _[|]_ _[p]_ �







_K_



�
 _j_ =1



 =



_K_
� E[ _|X_ _j_ _|_ _[p]_ ] _≤_ _K_ _j_ =1 max _,...,K_ [E][[] _[|][X]_ _[j]_ _[|]_ _[p]_ []] _[.]_

_j_ =1




                       

_**Lemma**_ **F.4** _**.**_ Let 2 _≤_ _p < ∞_ . For any _θ ∈_ [ _−B, B_ ] _[d]_ _[θ]_, we have

_N_ _p_ [�] 1 _/p_ � _|_ Ψ _|_ [2] _[p]_ [�] [1] _[/p]_
E ��� _S_ _θ_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] �� _≤_ [16] _[√][p][ −]_ [1] ~~_√_~~ [E] _N_ _._


_Proof._ It follows from (Welti 2020, Corollary 5.18), with _S_ _θ_ _[N]_ [=] _N_ [1] � _Nj_ =1 _[X]_ _[j]_ [,] _[ X]_ _[j]_ [ :=]

_|G_ ( _Z_ _j_ ) _−N_ _θ_ ( _Z_ _j_ ) _|_ [2], that



E � _|S_ _θ_ _[N]_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] _[|]_ _[p]_ [�] [1] _[/p]_ _[ ≤]_ [2] _[√][p][ −]_ [1]
~~_√_~~ _N_



max
� _j_ =1 _,...,N_ [E][[] _[|][X]_ _[j]_ _[ −]_ [E][[] _[X]_ _[j]_ []] _[|]_ _[p]_ []] [1] _[/p]_ _[.]_ �



But by the boundedness assumption 5.1, we have


_X_ _j_ _≤_ 2 _|G_ ( _Z_ _j_ ) _|_ [2] + 2 _|N_ _θ_ ( _Z_ _j_ ) _|_ [2] _≤_ 4 _|_ Ψ( _Z_ _j_ ) _|_ [2] _._


Hence



ERROR ESTIMATES FOR DEEPONETS 121


E[ _|X_ _j_ _−_ E[ _X_ _j_ ] _|_ _[p]_ ] [1] _[/p]_ _≤_ 8E[ _|_ Ψ( _Z_ _j_ ) _|_ [2] _[p]_ ] [1] _[/p]_ = 8E[ _|_ Ψ _|_ [2] _[p]_ ] [1] _[/p]_ _,_



where the last equality follows from the fact that the _Z_ _j_ are iid. Hence


_N_ _p_ [�] [1] _[/p]_ � _|_ Ψ _|_ [2] _[p]_ [�] [1] _[/p]_
E ��� _S_ _θ_ _[−]_ [E] � _S_ _θ_ _[N]_ ��� _≤_ [16] _[√][p][ −]_ [1] ~~_√_~~ [E] _N_ _._


_**Lemma**_ **F.5** _**.**_ Let _ϵ >_ 0 be given. Let _p ≥_ 2. Then we have







�



E



sup
_θ∈_ [ _−B,B_ ] _[d]_



1 _/p_
�� _S_ _θN_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] �� _p_
�



_√_ ~~_p_~~ _∥_ Ψ _∥_ _L_ 2 _p_
_≤_ 16 _∥_ Ψ _∥_ _L_ 2 _p_ _ϵ∥_ Φ _∥_ _L_ 2 _p_ + _K_ ( _ϵ_ ) [1] _[/p]_ _,_
� ~~_√_~~ _N_ �



where _K_ ( _ϵ_ ) denotes the _ϵ_ -covering number of [ _−B, B_ ] _[d]_ _[θ]_ .


_Proof._ Denote _K_ := _K_ ( _ϵ_ ) the covering number of [ _−B, B_ ] _[d]_ _[θ]_ . Then, by the definition of a covering number, there exist _θ_ 1 _, . . ., θ_ _K_, such that for any _θ ∈_ [ _−B, B_ ] _[d]_ _[θ]_,
there exists _j ∈{_ 1 _, . . ., K}_, such that _|θ −_ _θ_ _j_ _| ≤_ _ϵ_ . By Lemma F.2, we have



1 _/p_
�� _S_ _θN_ _[−]_ [E] � _S_ _θ_ _[N]_ ��� _p_
�



_p_ [�] [1] _[/p]_
_N_
��� _S_ _θ_ _j_ _[−]_ [E] � _S_ _θ_ _[N]_ _j_ ���� _._



E



sup

� _θ∈_ [ _−B,B_ ] _[dθ]_



_≤_ 8 _ϵ_ E[ _|_ ΨΦ _|_ _[p]_ ] [1] _[/p]_ + E max
� _j_ =1 _,...,K_



We estimate the first term by


8 _ϵ_ E[ _|_ ΨΦ _|_ _[p]_ ] [1] _[/p]_ _≤_ 8 _ϵ_ E[ _|_ Ψ _|_ [2] _[p]_ ] [1] _[/]_ [2] _[p]_ E[ _|_ Φ _|_ [2] _[p]_ ] [1] _[/]_ [2] _[p]_ = 8 _ϵ∥_ Ψ _∥_ _L_ 2 _p_ _∥_ Φ _∥_ _L_ 2 _p_ _._


By Lemma F.3 and Lemma F.4, we can estimate the last term



E max
� _j_ =1 _,...,K_



_p_ [�] [1] _[/p]_ _p_ [�] 1 _/p_
��� _S_ _θN_ _j_ _[−]_ [E] � _S_ _θ_ _[N]_ _j_ ���� _≤_ _K_ [1] _[/p]_ _j_ =1 max _,...,K_ [E] ���� _S_ _θN_ _j_ _[−]_ [E] � _S_ _θ_ _[N]_ _j_ ����




[E] � _|_ Ψ _|_ [2] _[p]_ [�] [1] _[/p]_
_≤_ [16] _[K]_ [1] _[/p]_ _[√]_ ~~_[p]_~~
~~_√_~~ _N_


_[∥]_ [Ψ] _[∥]_ [2]
= [16] _[K]_ [1] _[/p]_ ~~_√_~~ _[√]_ _N_ ~~_[p]_~~ _L_ [2] _[p]_ _._


Substitution of these upper bounds now yields



E max
� _j_ =1 _,...,K_



_p_ [�] [1] _[/p]_ _[∥]_ [Ψ] _[∥]_ [2]
_N_ _L_ [2] _[p]_
��� _S_ _θ_ _j_ _[−]_ [E] � _S_ _θ_ _[N]_ _j_ ���� _≤_ 8 _ϵ∥_ Ψ _∥_ _L_ 2 _p_ _∥_ Φ _∥_ _L_ 2 _p_ + [16] _[K]_ [1] _[/p]_ ~~_√_~~ _[√]_ _N_ ~~_[p]_~~ _._



The claimed bound follows. 

We also remark the following well-known fact:


_**Lemma**_ **F.6** _**.**_ The covering number of [ _−B, B_ ] _[d]_ satisfies



_CB_
_K_ ( _ϵ_ ) _≤_
� _ϵ_



_d_

_,_
�



for some constant _C >_ 0, independent of _ϵ_, _B_ and _d_ .


_Proof._ For a proof, see e.g. (Welti 2020, Lemma 5.11). 

122 ERROR ESTIMATES FOR DEEPONETS


Often, one can prove a bound of the form


_κ_
_G_ ( _u_ ) _≤_ Ψ( _u, y_ ) _≤_ _C_ �1 + _∥u∥_ _L_ 2 _x_ � _,_


and for Gaussian measures _µ_, we note that there exists _α >_ 0, such that

_e_ _α∥u∥_ [2] _L_ [2] _x_ _dµ_ ( _u_ ) _< ∞._

ˆ _L_ [2] _x_


We next want to derive some estimates on _∥_ Ψ _∥_ _L_ _p_, as a function of _p_ .


_**Lemma**_ **F.7** _**.**_ Let _p ≥_ 1, _α >_ 0. The mapping


[0 _, ∞_ ) _→_ R _,_ _x �→_ _p_ log(1 + _x_ ) _−_ _αx_ [2] _,_


satisfies the upper bound



_p_ log(1 + _x_ ) _−_ _αx_ [2] _≤_ _p_ log 1 + _[p]_
� 2 _α_



_._
�



_**Lemma**_ **F.8** _**.**_ If _A_ = ´ _L_ [2] _x_ [exp(] _[α][∥][u][∥]_ _L_ [2] [2] _x_ [)] _[ dµ]_ [(] _[u]_ [)] _[ <][ ∞]_ [, then]



�ˆ _L_ [2] _x_



1 _/p_


_κp_
�1 + _∥u∥_ _L_ 2 _x_ � _dµ_ ( _u_ )

�



_≤_ _A_ 1 + _[κ][p]_
� 2 _α_



_κ_

_._
�



_Proof._ We have


(1 + _∥u∥_ _L_ 2 ) _[κp]_ = exp ( _κp_ log(1 + _∥u∥_ _L_ 2 ))

= exp � _κp_ log(1 + _∥u∥_ _L_ 2 ) _−_ _α∥u∥_ [2] [�] _e_ _[α][∥][u][∥]_ [2]



_≤_ exp sup _κp_ log(1 + _x_ ) _−_ _αx_ [2]

� _x∈_ [0 _,∞_ ) �



_e_ _[α][∥][u][∥]_ [2] _._



From Lemma F.7, it follows that


(1 + _∥u∥_ ) _[κp]_ _≤_ 1 + _[κ][p]_
� 2 _α_



_κp_
� _e_ _[α][∥][u][∥]_ [2] _._



Thus, we conclude that
�ˆ _L_ [2] [(1 +] _[ ∥]_



1 _/p_
_≤_ _A_ [1] _[/p]_ [ �] 1 + _[κ][p]_
_L_ [2] [(1 +] _[ ∥][u][∥]_ _[L]_ [2] [)] _[κp]_ � 2 _α_



2 _α_



_κ_
_≤_ _A_ 1 + _[κ][p]_
� � 2 _α_



_κ_
_≤_ _A_ 1 + _[κ][p]_
� � 2 _α_



_κ_

_,_
�



for all _p ≥_ 1, where in the last step, we have used that


_A_ = _L_ [2] [)] _[ dµ]_ [(] _[u]_ [)] _[ ≥]_ [1] _[,]_
ˆ _L_ [2] [ exp(] _[α][∥][u][∥]_ [2]


for any _α >_ 0. 


_Proof of Theorem 5.3._ Note that

_L_ ( _N_ � _N_ ) _−_ _L_ [�] ( _N_ [�] ) _≤_ 2 sup
���� ��� _θ∈_ [ _−B,B_ ] _[dθ]_


We now claim that



�
_L_ _N_ ( _N_ _θ_ ) _−_ _L_ ( _N_ _θ_ ) _._
���� ���



2 _κ_ +1 _/_ 2

1 + _d_ _θ_ log( _CB√N_ ) _,_ (F.1)
� �



�



_C_
_≤_
~~_√_~~ _N_



E



�



sup
_θ∈_ [ _−B,B_ ] _[dθ]_



�
_L_ _N_ ( _N_ _θ_ ) _−_ _L_ ( _N_ _θ_ )
���� ���



for _C_ = _C_ ( _α, κ,_ Ψ _,_ Φ), from which the claimed bound on the generalization error
follows. To prove the claimed inequality (F.1), we note that


� �
_L_ _N_ ( _N_ _θ_ ) _−_ _L_ ( _N_ _θ_ ) = _S_ _θ_ _[N]_ _[−]_ [E][[] _[S]_ _θ_ _[N]_ []] _[.]_


ERROR ESTIMATES FOR DEEPONETS 123


By Lemma F.5 and F.6, we have for any _p ≥_ 2 and _ϵ >_ 0:



�



1 _/p_
�� _S_ _θN_ _[−]_ [E] � _S_ _θ_ _[N]_ ��� _p_
�



�



_CB_
_ϵ∥_ Φ _∥_ _L_ 2 _p_ +
� _ϵ_



E



sup

� _θ∈_ [ _−B,B_ ] _[dθ]_



_≤_ 16 _∥_ Ψ _∥_ _L_ 2 _p_



_._



_ϵ_



_d_ _θ_ _/p_ _√_ ~~_p_~~ _∥_ Ψ _∥_ _L_ 2 _p_
� ~~_√_~~ _N_



By assumption on Ψ _,_ Φ, there exist constants _C >_ 0, _κ >_ 0, such that


_|_ Ψ( _u, y_ ) _|, |_ Φ( _u, y_ ) _| ≤_ _C_ (1 + _∥u∥_ _L_ 2 ) _[κ]_ _._ (F.2)


By Lemma F.8, we can thus estimate


_∥_ Ψ _∥_ _L_ 2 _p_ _, ∥_ Φ _∥_ _L_ 2 _p_ _≤_ _C_ (1 + _γκp_ ) _[κ]_ _,_


for constants _C, γ >_ 0, depending only the measure _µ_ and the constant _C_ appearing
in the upper bound (F.2). In particular, we have



_≤_ 16 _C_ [2] (1 + _γκp_ ) [2] _[κ]_
�



E

�



�



_d_ _θ_ _/p_
_√_ ~~_p_~~
� ~~_√_~~ _N_



sup
_θ∈_ [ _−B,B_ ] _[dθ]_



1 _/p_
�� _S_ _θN_ _[−]_ [E] � _S_ _θ_ _[N]_ ��� _p_
�



_CB_
_ϵ_ +
� _ϵ_



_,_



_ϵ_



for some constants _C, γ >_ 0, independent of _κ_, _µ_, _B_, _d_ _θ_, _N_, _ϵ >_ 0 and _p ≥_ 2.
1
We now choose _ϵ_ =
~~_√_~~ _N_ [, so that]



_CB_
_ϵ_ +
� _ϵ_



_d_ _θ_ _/p_
_√_ ~~_p_~~
� ~~_√_~~ _N_



~~_p_~~ 1


=
_N_ ~~_√_~~



_N_



_d_ _θ_ _/p_

�1 + � _CB√N_ � _√_ ~~_p_~~ � _._



Next, let _p_ = _d_ _θ_ log( _CB√_



Next, let _p_ = _d_ _θ_ log( _CB√N_ ). We may wlog assume that _p ≥_ 2 (otherwise, increase

the constant _C_ ). Then,
� _CB√N_ � _d_ _θ_ _/p_ _√_ ~~_p_~~ = exp log( _CB√N_ ) _d_ _θ_ _√_ ~~_p_~~ = _e_ � _d_ _θ_ log( _CB_ ~~_√_~~ _N_ ) _,_



_√_ ~~_p_~~ = _e_ �
�



_d_ _θ_ _/p_
_N_ � _√_ ~~_p_~~ = exp



log( _CB√_


_p_

�



_√N_ ) _d_ _θ_


_p_



_d_ _θ_ log( _CB_ ~~_√_~~



_N_ ) _,_



and thus we conclude that



_d_ _θ_ log( _CB_ ~~_√_~~ _N_ ) _._ _._

�



_CB_
_ϵ_ +
� _ϵ_



_d_ _θ_ _/p_
_√_ ~~_p_~~
� ~~_√_~~ _N_



~~_p_~~ 1

_≤_
_N_ ~~_√_~~



_N_



1 + _e_
� �



On the other hand, we have


2 _κ_

(1 + _γκp_ ) [2] _[κ]_ = 1 + _γκd_ _θ_ log( _CB√N_ ) _._
� �



Increasing the constant _C >_ 0, if necessary, we can further estimate


2 _κ_ [�]

�1 + _γκd_ _θ_ log( _CB√N_ )� 1 + _e_ � _d_ _θ_ log( _CB_ ~~_√_~~ _N_ ) _._ � _≤_ _C_ �1 + _d_ _θ_



2 _κ_ +1 _/_ 2
_N_ ) _,_
�



2 _κ_ [�]
_N_ )� 1 + _e_ �



_N_ ) _._ _≤_ _C_ 1 + _d_ _θ_ log( _CB√_
� �



_d_ _θ_ log( _CB_ ~~_√_~~



where _C >_ 0 depends on _κ_, _γ_, _µ_ and the constant appearing in (F.2), but is
independent of _d_ _θ_, _B_ and _N_ . We can express this dependence in the form _C_ =
_C_ ( _µ,_ Ψ _,_ Φ) _>_ 0, as the constants _κ_ and _γ_ depend on the Gaussian tail of _µ_ and the
upper bound on Ψ, Φ.
To conclude, we have shown that there exists a constant _C_ = _C_ ( _µ,_ Ψ _,_ Φ) _>_ 0,
such that for any _d_ _θ_, _B_ and _N_, we have



sup

� _θ∈_ [ _−B,B_ ] _[dθ]_


sup

� _θ∈_ [ _−B,B_ ] _[dθ]_



�� _S_ _θN_ _[−]_ [E] � _S_ _θ_ _[N]_ ���
�


1 _/p_
�� _S_ _θN_ _[−]_ [E] � _S_ _θ_ _[N]_ ��� _p_
�



= E

�


_≤_ E



E



�



sup
_θ∈_ [ _−B,B_ ] _[dθ]_



�
_L_ _N_ ( _N_ _θ_ ) _−_ _L_ ( _N_ _θ_ )
���� ���



_C_
_≤_
~~_√_~~ _N_



2 _κ_ +1 _/_ 2

1 + _d_ _θ_ log( _CB√N_ ) _._
� �



This is the claimed inequality (F.2). 

