## Interpretable Scientific Discovery with Symbolic Regression: A Review

Nour Makke Sanjay Chawla


Qatar Computing Research Institute, HBKU, Doha


May 3, 2023


**Abstract**


Symbolic regression is emerging as a promising machine learning method for learning succinct underlying
interpretable mathematical expressions directly from data. Whereas it has been traditionally tackled with
genetic programming, it has recently gained a growing interest in deep learning as a data-driven model
discovery method, achieving significant advances in various application domains ranging from fundamental
to applied sciences. In this survey, we present a structured and comprehensive overview of symbolic regression
methods and discuss their strengths and limitations.


1


### **Contents**

**1** **Introduction** **3**


**2** **Problem Definition** **4**
2.1 Class of Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

2.2 Expression representation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4


**3** **Symbolic regression methods overview** **5**


**4** **Linear symbolic regression** **6**
4.1 Unidimensional case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

4.1.1 Univariate function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

4.1.2 Multivariate function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

4.2 Multidimensional case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11


**5** **Nonlinear symbolic regression** **12**


**6** **Tree expression** **13**
6.1 Genetic Programming . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
6.2 Transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

6.3 Reinforcement learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16


**7** **Applications** **17**


**8** **Datasets** **23**


**9** **Discussion** **24**


**10 Conclusion** **25**


**A Datasets Benchmarks Equations** **29**


2


### **1 Introduction**

Symbolic Regression (SR) is a rapidly growing subfield within machine learning (ML) to infer symbolic mathematical expressions from data [1, 2]. Interest in SR is being driven by the observation that it is not sufficient to
only have accurate predictive models; however, it is often necessary that the learned models be interpretable [3].
A model is interpretable if the relationship between the input and output of the model can be logically or
mathematically traced in a succinct manner. In other words, learnable models are interpretable if expressed as
mathematical equations. As “disciplines” become increasingly data-rich and adopt ML techniques, the demand
for interpretable models is likely to grow. For example, in the natural sciences (e.g., physics), mathematical
models derived from first principles make it possible to reason about the underlying phenomenon in a way
that is not possible with predictive models like deep neural networks. In critical disciplines like healthcare,
non-interpretable models may never be allowed to be deployed - however accurate they maybe [4].



**Example:** Consider a data set consisting of samples ( _q_ 1 _, q_ 2 _, r, F_ ), where _q_ 1 and _q_ 2 are the charges of two
particles, _r_ is the distance between them and _F_ is the measured force between the particles. Assume _q_ 1 _, q_ 2,
and _r_ are the input variables, and _F_ is the output variable. Suppose we model the input-output relationship
as _F_ = _θ_ 0 + _θ_ 1 _q_ 1 + _θ_ 2 _q_ 2 + _θ_ 3 _r_ . Then, using the data set, we can infer the model’s parameters ( _θ_ _i_ ). The model
will be interpretable because we will know the impact of each variable on the output. For example, if _θ_ 3 is
negative, then that implies that as _r_ increases, the force _F_ will decrease. From physics, we know that this form
of the model is unlikely to be accurate. On the other hand, we could model the relationship using a neural network _F_ = _NN_ ( _q_ 1 _, q_ 2 _, r, θ_ ). We expect the model to be highly accurate and predictive because neural networks
(NNs) are universal function approximators. However, the model is uninterpretable because the input-output
relationship is not easily apparent. The input feature vector subsequently undergoes several layers of nonlinear
transformations, i.e., _y_ = _σ_ ( [�] _i_ _[W]_ _[i]_ _[ σ]_ [(][�] _j_ _[W]_ _[j]_ _[ σ]_ [(][�] _k_ _[W]_ _[k]_ _[ σ]_ [(] _[· · ·]_ [ �] _ℓ_ _[W]_ _[ℓ]_ [x)))), where] _[ σ]_ [ is a nonlinear activation]

function, and _W_ _idx_ are the learnable parameters of NN layer of index _idx_ . Such models, called _“blackbox”_, do
not have an internal logic to let users understand how inputs are mathematically mapped to outputs. Explainability is the application of other methods to explain model predictions and to understand how it is learned. It
refers to why the model makes the decision that way. What distinguishes explainability from interpretability
is that interpretable models are transparent [3]. For example, the linear regression model predictions can be
interpreted by evaluating the relative contribution of individual features to the predictions using their weights.
An ideal SR model will return the relationship as _k_ _[q]_ _r_ [1] _[q]_ [2][2] [, which is the definition of the Coulomb force between]

two charged particles with a constant [1] _k_ = 8 _._ 98 _×_ 10 [9] . However, learning the SR model is highly non-trivial as
it involves searching over a large space of mathematical operations and identifying the right constant ( _k_ ) that
will fit the data. SR models can be directly inferred from data or can be used to _“whitebox”_ a _“blackbox”_ model
such as a neural network.



_i_ _[W]_ _[i]_ _[ σ]_ [(][�]



_j_ _[W]_ _[j]_ _[ σ]_ [(][�]



_k_ _[W]_ _[k]_ _[ σ]_ [(] _[· · ·]_ [ �]



The ultimate goal of SR is to bridge data and observations following the Keplerian trial and error approach [5].
Kepler developed a data-driven model for planetary motion using the most accurate astronomical measurements
of the era, which resulted in elliptic orbits described by a power law. In contrast, Newton developed a dynamic
relationship between physical variables that described the underlying process at the origin of these elliptic orbits. Newton’s approach [6] led to three laws of motion later verified by experimental observations. Whereas
both methods fit the data well, Newton’s approach could be generalized to predict behavior in regimes where no
data were available. Although SR is regarded as a data-driven model discovery tool, it aims to find a symbolic
model that simultaneously fits data well and could be generalized to uncovered regimes.


SR is deployed as an interpretable and predictive ML model or a data-driven scientific discovery method. SR
was investigated as early as 1970 in research works [7, 8, 9] aiming to rediscover empirical laws. Such works
iteratively apply a set of data-driven heuristics to formulate mathematical expressions. The first AI system
meant to automate scientific discovery is called BACON [10, 11]. It was developed by Patrick Langley in
the late 1970s and was successful in rediscovering versions of various physical laws, such as Coulomb’s law
and Galileo’s laws for the pendulum and constant acceleration, among many others. SR was later studied by
Koza [12, 13, 1] who proposed that genetic programming (GP) can be used to discover symbolic models by
encoding mathematical expressions as computational trees, where GP is an evolutionary algorithm that iteratively evolves an initial population of individuals via biology-inspired operations. SR was since then tackled
with GP-based methods [1, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]. Moreover, it was popularized as a datadriven scientific discovery tool with the commercial software Eureqa [26] based on a research work [2]. Whereas
GP-based methods achieve high prediction accuracy, they do not scale to high dimensional data sets and are
sensitive to hyperparameters [19]. More recently, SR has been addressed with deep learning-based methods

[27, 28, 19, 29, 30, 31, 32, 33] which leverage neural networks (NNs) to learn accurate symbolic models. SR has


1 _k_ is the electric force (or Coulomb) constant, _k_ = 8 _._ 9875517923 _×_ 10 9 kg _·_ m 3 _·_ s _−_ 4 _·_ A _−_ 2 in SI base units.


3


been applied in fundamental and applied sciences such as astrophysics [34], chemistry [35, 36], materials science [37, 38], semantic similarity measurement [39], climatology [40], medicine [41], among many others. Many
of these applications are promising, showing the potential of SR. A recent SR benchmarking platform _SRBench_
is introduced by La Cava _et al._ [42]. It comprises 14 SR methods (among which ten are GP-based), applied on
252 data sets. The goal of _SRBench_ was to provide a benchmark for rigorous evaluation and comparison of SR
methods.


This survey aims to help researchers effectively and comprehensively understand the SR problem and how it
could be solved, as well as to present the current status of the advances made in this growing subfield. The
survey is structured as follows. First, we define the SR problem, present a structured and comprehensive review of methods, and discuss their strengths and limitations. Furthermore, we discuss the adoption of these
SR methods across various application domains and assess their effectiveness. Along with this survey, a living
review [25] aims to group state-of-the-art SR methods and applications and track advances made in the SR
field. The objective is to update this list often to incorporate new research works.


This paper is organized as follows. The SR problem definition is presented in Section 2. We present an overview
of methods deployed to solve the SR problem in Section 3, and the methods are discussed in detail in Sections 4, 5
and 6. Selected applications are described and discussed in Section 7. Section 8 presents an overview of existing
benchmark data sets. Finally, we summarize our conclusions and discuss perspectives in Section 10.

### **2 Problem Definition**


The problem of symbolic regression can be defined in terms of classical Empirical Risk Minimization (ERM) [43].


**Data:** Given a data set _D_ = _{_ ( **x** _i_ _, y_ _i_ ) _}_ _[n]_ _i_ =1 [, where] **[ x]** _[i]_ _[ ∈]_ [R] _[d]_ [ is the input vector and] _[ y]_ _[i]_ _[ ∈]_ [R][ is a scalar output.]


**Function Class:** Let _F_ be a function class consisting of mappings _f_ : R _[d]_ _→_ R.


**Loss Function:** Define the loss function for every candidate _f ∈F_ :



_l_ ( _f_ ) :=



_n_
� _l_ ( _f_ ( _x_ _i_ ) _, y_ _i_ ) (1)


_i_ =1



A common choice is the squared difference between the output and prediction, i.e. _l_ ( _f_ ) = [�] _i_ [(] _[y]_ _[i]_ _[ −]_ _[f]_ [(] _[x]_ _[i]_ [))] [2] [.]


**Optimization:** The optimization task is to find the function ( _f_ ) over the set of functions _F_ that minimizes
the loss function:


_f_ _[∗]_ = arg min _l_ ( _f_ ) (2)
_f_ _∈F_


As stated below, what distinguishes SR from conventional regression problems is the discrete nature of the
function class _F_ . Different methods for solving the SR problem reduce to characterizing the function class.


**2.1** **Class of Function**


In SR, to define _F_, we specify a library of elementary arithmetic operations and mathematical functions and
variables, and an element _f ∈F_ is the set of all functions that can be obtained by function composition in the
library [44]. For example, consider a library:


_L_ = _{_ id( _·_ ) _,_ add( _·, ·_ ) _,_ sub( _·, ·_ ) _,_ mul( _·, ·_ ) _,_ +1 _, −_ 1 _}_ (3)


Then the set of all of the polynomials (in one variable _x_ ) with integer coefficients can be derived from _L_ using
function composition.


**2.2** **Expression representation**


It is convenient to express symbolic expressions in a sequential form using either a unary-binary expression
tree or the polish notation [45]. For example, the expression _f_ (x) = _x_ 1 _x_ 2 _−_ 2 _x_ 3 can be derived using function
composition from _L_ (Eq. 3) and represented as a tree-like structure illustrated in Figure 1a. By traversing the
(binary) tree top to bottom and left to right in a depth-first manner, we can represent the same expression as
a unique sequence called the polish form, as illustrated in Figure 1b.


4


(a)

_−_ _×_ _x_ 1 _x_ 2 _×_ + 1 1 _x_ 3


(b)


Figure 1: (a) Example of a unary-binary tree that encodes _f_ (x) = _x_ 1 _x_ 2 _−_ 2 _x_ 3 . (b) Sequence representation of
the tree-like structure of _f_ (x).


In practice, the library _L_ includes many other common elementary mathematical functions, including the basic
trigonometric functions like sine, cosine, logarithm, exponential, square root, power low, etc. Prior domain
knowledge is advantageous for library definition because it reduces the search space to only include the most
relevant mathematical operations to the studied problem. Furthermore, a large range of possible numeric
constants should be possible to express. For example, numbers in base-10 floating point notation rounded up to
four significant digits can be represented as triple of ( _sign_, _mantissa_, _exponent_ ) [31]. The function sin(3 _._ 456 _x_ ),
for example, can be represented as [sin _,_ mul _,_ 3456 _, E −_ 3 _, x_ ].

### **3 Symbolic regression methods overview**


In this survey, we categorize SR methods in the following manner: regression-based methods, expression treebased methods, physics-inspired and mathematics-inspired methods, as presented in Figure 2. For each category,
a summary of the mathematical tool, the expression form, the set of unknowns, and the search space, is presented in Table 1.


The linear method defines the functional form as a linear combination of nonlinear functions of _x_ that are
comprised in the predefined library _L_ . Linear models are expressed as:


_f_ (x _, θ_ ) = � _θ_ _j_ _h_ _j_ (x) (4)

_j_


where _j_ spans the base functions of _L_ . The optimization problem reduces to find the set of parameters _{θ}_ that
minimizes the loss function defined over a continuous parameter space Θ = R _[M]_ as follows:



_θ_ _[∗]_ = arg min
_θ∈_ Θ



� _l_ ( _f_ ( _x_ _i_ _, θ_ ) _, y_ _i_ ) (5)


_i_



This method is advantageous for being deterministic and disadvantageous because it imposes a single model
structure which is fixed during training when the model’s parameters are learned.


The nonlinear method defines the model structure by a neural network. Nonlinear models can thus be expressed

as:



_f_ (x _, W_ ) = _σ_ (�



_W_ _i_ _σ_ (�

_i_



_W_ _j_ _σ_ ( _· · ·_ �
_j_ _ℓ_



_W_ _ℓ_ x)))) (6)

_ℓ_



where _σ_ is a nonlinear activation function, and _W_ _idx_ are the learnable parameters of NN layer of index _idx_ .
Similarly to the linear method, the optimization problem reduces finding the set of parameters _{W, b}_ of neural
network layers, which minimizes the loss function over the space of real values.


Expression tree-based methods treat mathematical expressions as unary-binary trees whose internal nodes are
operators and terminals are operands (variables or constants). This category comprises GP-based, deep neural
transformers, and reinforcement learning-based methods. In GP-based methods, a set of transition rules (e.g.,
mutation, crossover) is defined over the tree space and applied to an initial population of trees throughout


5


Figure 2: Taxonomy based on the type of symbolic regression methods. _φ_ denotes a neural network function,
_W_ denotes the set of learnable parameters in NN. **x** denotes the input data, **z** denotes a reduced representation
of **x**, and **x** _[′]_ denotes a new representation of **x**, e.g., by defining new features based on the original ones. _T_
represents the final population of selected expression trees in genetic programming.


many iterations until the loss function is minimized. Transformers [46] represent a novel architecture of neural
network (encoder and decoder) that uses attention mechanism. The latter was primarily used to capture longrange dependencies in a sentence. Transformers were designed to operate on sequential data and to perform
sequence-to-sequence (seq2seq) tasks. For their use in SR, input data points ( **x** _, y_ ) and symbolic expressions
( _f_ ) are encoded as sequences and transformers perform set-to-sequence tasks. The unknowns are the weight
parameters of the encoder and the decoder. Reinforcement learning (RL) is a machine learning method that
seeks to learn a policy _π_ ( _x|θ_ ) by training an agent to perform a task by interacting with its environment
in discrete time steps. An RL setting requires four components: state space, action space, state transition
probabilities, and reward. The agent selects an action that is sent to the environment. A reward and a new
state are sent back to the agent from its environment and used by the agent to improve its policy at the next
time step. In the context of SR, symbolic expression (sequence) represents a state, predicting an element in a
sequence represents an action, the parent and sibling represent the environment, and the reward is commonly
chosen as the mean square error (MSE). RL-based SR methods are commonly hybrid and use various ML tools
(e.g., NN, RNN, etc.) in a joint manner with RL.

### **4 Linear symbolic regression**


The linear approach assumes, by definition, that the target symbolic expression ( _f_ ( _x_ )) is a linear combination
of nonlinear functions of feature attributes:


_f_ (x) = � _θ_ _j_ _h_ _j_ (x) (7)

_j_


Here x denotes the input features vector, _θ_ _j_ denotes a weight coefficient, and _h_ _j_ ( _·_ ) denotes a unary operator
of the library _L_ . This approach predefines the model’s structure and reduces the SR problem to learn only
the model’s parameters by solving a system of linear equations. The particular case where _f_ ( _x_ ) is a linear
combination of degree-one monomial reduces to a conventional linear regression problem, i.e., _f_ ( _x_ ) = [�] _j_ _[θ]_ _[j]_ _[x]_ _[j]_ [ =]

_θ_ 0 + _θ_ 1 _x_ + _θ_ 2 _x_ [2] + _· · ·_ . There exist two cases for this problem: (1) a unidimensional case defined by _f_ : R _[d]_ _→_ R;
and (2) a multidimensional case defined by _f_ : R _[d]_ _→_ R _[m]_, with _d_ the number of input features and _m_ the
number of variables required for a complete description of a system; for example, the Lorenz system for fluid
flow is defined in terms of three physical variables which depend on time.


6


Table 1: Table summarizing symbolic regression methods. The mathematical tool, the expression form, the set
of unknowns, and the search space are specified for each method. Set2seq abbreviates “set-to-sequence”.


Method Tool Expression form Unkown Search space


Linear SR Uni-D linear system _y_ = ~~[�]~~ _i_ _[θ]_ _[i]_ _[f]_ _[i]_ [(] **[x]** [)] _{θ}_ _i_ R

Multi-D linear system _y_ _i_ = [�] _j_ _[θ]_ _[j]_ _[f]_ _[j]_ [(] **[x]** [)] ( _{θ}_ _i_ ) _j_ R


Nonlinear SR Neural Network _y_ = _f_ ( _W ·_ **x** + _b_ ) _{W, b}_ R


Genetic Programming Expression tree trees
Expression-tree search

Transformers set2seq mapping _{W_ _q_ _, W_ _k_ _, W_ _v_ _}_ R


Reinforcement learning set2seq mapping _π_ ( _θ_ ) R


Physics-inspired AI-Feynman _y_ = _f_ ( **x** _, θ_ ) _−_ _−_


Mathematics-inspired Symbolic metamodels _G_ ( **x** _, θ_ ) _θ_ R


**4.1** **Unidimensional case**


Given a data set _D_ = _{_ ( _x_ _i_ _, y_ _i_ ) _}_ _[n]_ _i_ =1 [, the mathematical expression could be either univariate (] _[x]_ _[i]_ _[ ∈]_ [R] _[, y]_ _[i]_ [ =] _[ f]_ [(] _[x]_ _[i]_ [))]
or multivariate ( **x** _i_ _∈_ R _[d]_ _, y_ _i_ = _f_ ( **x** _i_ )). The methodology of linear SR is presented in detail for the univariate
case in Secion 4.1.1 for simplicity and is extended for the multivariate case in Section 4.1.2.


**4.1.1** **Univariate function**


**Data set:** _D_ = _{x_ _i_ _∈_ R; _y_ _i_ = _f_ ( _x_ _i_ ) _}_ .


**Library:** _L_ can include any number of mathematical operators such that the dimension of the data set is always
greater than the dimension of the library matrix (see discussion below).


In this approach, a coefficient _θ_ _j_ is assigned to each candidate function ( _f_ _j_ ( _·_ ) _∈_ _L_ ) as an activeness criterion
such that:


_y_ = � _θ_ _j_ _f_ _j_ ( _x_ ) (8)

_j_


Applying Eq. 8 to input-output pairs ( _x_ _i_ _, y_ _i_ ) yields a system of linear equations as follows:



_y_ 1 = _θ_ 0 + _θ_ 1 _f_ 1 ( _x_ 1 ) + _θ_ 2 _f_ 2 ( _x_ 1 ) + _· · ·_ + _θ_ _k_ _f_ _k_ ( _x_ 1 )
_y_ 2 = _θ_ 0 + _θ_ 1 _f_ 1 ( _x_ 2 ) + _θ_ 2 _f_ 2 ( _x_ 2 ) + _· · ·_ + _θ_ _k_ _f_ _k_ ( _x_ 2 )

...
_y_ _n_ = _θ_ 0 + _θ_ 1 _f_ 1 ( _x_ _n_ ) + _θ_ 2 _f_ 2 ( _x_ _n_ ) + _· · ·_ + _θ_ _k_ _f_ _k_ ( _x_ _n_ )



(9)



which can be represented in a matrix form as:

 _y_ 1  1 _f_ 1









_θ_ 0

 _θ_ 1


...

 _θ_ _k_





(10)




_y_ 1

_y_ 2

...

_y_ _n_







=














1 _f_ 1 ( _x_ 1 ) _f_ 2 ( _x_ 1 ) _· · ·_ _f_ _k_ ( _x_ 1 )
1 _f_ 1 ( _x_ 2 ) _f_ 2 ( _x_ 2 ) _· · ·_ _f_ _k_ ( _x_ 2 )

...
1 _f_ 1 ( _x_ _n_ ) _f_ 2 ( _x_ _n_ ) _· · ·_ _f_ _k_ ( _x_ _n_ )



Equation 10 can then be presented in a compact form:


Y = U(X) _·_ Θ (11)


7


where Θ _∈_ R [(] _[k]_ [+1)] is the sparse vector of coefficients, and U _∈_ R _[n][×]_ [(] _[k]_ [+1)] is the library matrix which can be
represented as a function of the input vector X as follows:







(12)




U(X) =


**Example:** For a library defined as:







 1 _|_ _f_ 1 (X) _|_ _f_ 2 (X) _|_ _· · ·_ _f_ _k_ (X) _|_

 _|_ _|_ _|_ _|_



_L_ = _{_ 1 _, x,_ ( _·_ ) [2] _,_ sin( _·_ ) _,_ cos( _·_ ) _,_ exp( _·_ ) _}_ (13)



The matrix U becomes:









U(X) =







 1 _|_ X _|_ _|_ X [2] sin(X) _|_ cos(X) _|_ exp(X) _|_

 _|_ _|_ _|_ _|_ _|_ _|_



Each row (of index _i_ ) in Eq. 12 is a vector of ( _k_ + 1) functions of _x_ _i_ . The vector of coefficients, i.e., the model’s
parameters, is obtained by solving Eq. 11 as follows: [2] :


Θ = (U [T] U) _[−]_ [1] U [T] Y (14)


The magnitude of a coefficient _θ_ _k_ effectively measures the size of the contribution of the associated function
_f_ _k_ ( _·_ ) to the final prediction. Finally, the prediction vector Y can be evaluated using Eq. 11. [ˆ]


An exemplary schematic is illustrated in Figure 3 for the univariate function _f_ ( _x_ ) = 1 + _αx_ [3] . Only coefficients
associated with functions _{_ 1 _, x_ [3] _}_ of the library are non-zero, with values equal to 1 and _α_, respectively.



Y



1 _[x ]_ _x_ [2] _x_ [3]



Θ



=




_· · ·_



Figure 3: Schematic of the system of linear equations of Eq. 11 for _f_ ( _x_ ) = 1 + _αx_ [3] . A library matrix U(X)
of nonlinear functions of the input is constructed, where _L_ = _{_ 1 _, x, x_ [2] _, x_ [3] _, · · · }_ . The marked entries in the Θ
vector denote the non-zero coefficients determining which functions of the library are active.


In the following, linear SR is tested on synthetic data. In each experiment, training and test data sets are
generated. Each set consists of twenty data points randomly sampled from a uniform distribution U( _−_ 1 _,_ 1),
and _y_ is evaluated using a univariate function, i.e., _D_ = _{_ ( _x_ _i_ _, f_ ( _x_ _i_ )) _}_ _[n]_ _i_ =1 [. Two libraries are considered in these]
experiments: _L_ 1 = _{x,_ ( _·_ ) [2] _,_ ( _·_ ) [3] _, · · ·,_ ( _·_ ) [9] _}_ and _L_ 2 = _L_ 1 _∪{_ sin( _·_ ) _,_ cos( _·_ ) _,_ tan( _·_ ) _,_ exp( _·_ ) _,_ sigmoid( _·_ ) _}_ . The results are
reported in terms of the output expression (Equation 7) and the coefficient of determination _R_ [2] . SR problems
are grouped into (i) pure polynomial functions and (ii) mixed polynomial and trigonometric functions. In each
experiment, parameters are learned using the training data set, and results are reported for the test data set in
Table 2.
For polynomial functions, an exact output is obtained using _L_ 1 with an _R_ [2] = 1 _._ 0, whereas only approximate
output is obtained using _L_ 2 . In the latter case, the quality of the fit depends on the size of the training data set.
An exemplary result is shown in Figure 4 for _f_ ( _x_ ) = _x_ + _x_ [2] + _x_ [3] . Points represent the (test) data of the input
file, i.e., X; the red curve represents _f_ ( _x_ ) as a function of _x_, and the blue and black dashed curves represent
the predicted function _f_ [ˆ] ( _x_ ) obtained using _L_ 1 and _L_ 2 respectively. An exact match between the ground-truth
function and the predicted one is found using _L_ 1, whereas a significant discrepancy is obtained using _L_ 2 . This
discrepancy could be explained by the fact that various functions in _L_ 2 exhibit the same _x_ -dependence over the
covered _x_ -range.


2 Technically the pseudo-inverse, _U_ +


8


Table 2: Results of linear SR in the case of univariate functions. _D_ = _{_ ( _x_ _i_ ; _y_ _i_ ) _}_ ; _x_ _i_ _∈_ U( _−_ 1 _,_ 1 _,_ 20) and _y_ _i_ = _f_ ( _x_ _i_ ).
_L_ 1 = _{x,_ ( _·_ ) [2] _, · · ·,_ ( _·_ ) [9] _}_ and _L_ 2 = _L_ 1 _∪{_ sin( _·_ ) _,_ cos( _·_ ) _,_ tan( _·_ ) _,_ exp( _·_ ) _,_ sigmoid( _·_ ) _}_ . T denotes True, and F denotes
False.


Benchmark Expression _L_ 1 _L_ 2


Exp _R_ [2] Exp _R_ [2]


Nguyen-2 _x_ [4] + _x_ [3] + _x_ [2] + _x_ T 1.0 F 0.886
Nguyen-3 _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ T 1.0 F 0.867
Livermore-21 _x_ [8] + _x_ [7] + _x_ [6] + _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ T 1.0 F 0.869
Livermore-9 _x_ [9] + _x_ [8] + _x_ [7] + _x_ [6] + _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ T 1.0 F 0.882
Livermore-6 _x_ + 2 _x_ [2] + 3 _x_ [3] + 4 _x_ [4] T 1.0 F 0.417
Livermore-19 _x_ + _x_ [2] + _x_ [4] + _x_ [5] T 1.0 F -0.079


Livermore-14 ~~_[∗]_~~ _x_ + _x_ ~~[2]~~ + _x_ ~~[3]~~ + sin( _x_ ) F 1.0 F -0.857
Nguyen-5 sin( _x_ [2] ) cos( _x_ ) _−_ 1 F 0.999 F -3.97
Nguyen-6 sin( _x_ ) + sin( _x_ + _x_ [2] ) F 0.999 F 0.564


Figure 4: Result of linear SR for the Nguyen-1 benchmark, i.e., _f_ ( _x_ ) = _x_ + _x_ [2] + _x_ [3] . Red points represent (test)
data set. The red curve represents the true function. The blue and black dashed curves represent the learned
functions using _L_ 1 and _L_ 2, respectively.


For mixed polynomial and trigonometric expressions, both library choices do not produce the exact expression.
However, a better _R_ [2] -coefficient is obtained using _L_ 1 . In the case of Nguyen-5 benchmark for example, i.e.,
_f_ ( _x_ ) = sin( _x_ [2] ) cos( _x_ ) _−_ 1, the resulting function is the Taylor expansion of _f_ :


ˆ
_y_ ( _x_ ) _≈−_ 1 + 0 _._ 9 _x_ [2] _−_ 0 _._ 5 _x_ [4] _−_ 0 _._ 13 _x_ [6] + _O_ ( _x_ [8] )


In conclusion, this approach can not learn the ground-truth function when the latter is a multiplication of
two functions (i.e., _f_ ( _x_ ) = _f_ 1 ( _x_ ) _∗_ _f_ 2 ( _x_ )) or when it has a multiplicative or an additive factor to the variable
(e.g., sin( _α_ + _x_ ) _,_ exp( _λ ∗_ _x_ ), etc.). In the best case, it outputs an approximation of the ground-truth function.
Furthermore, this approach fails to predict the correct mathematical expression when the library is extended
to include a mixture of polynomial, trigonometric, exponential, and logarithmic functions.


**4.1.2** **Multivariate function**


For a given data set _D_ = _{x_ _i_ _∈_ R _[d]_ ; _y_ _i_ = _f_ ( _x_ 1 _, · · ·, x_ _d_ ) _}_, where _d_ is the number of features, the same equations
presented in Section 4.1.1 are applicable. However, the dimension of the library matrix U changes to consider
the features vector dimension. For example, for the same library shown in Eq. 13 and a two dimensional features


9


vector, i.e., X _∈_ R [2], U(X) becomes:









U(X) =


=







 1 _|_ X _|_ X _|_ _[P]_ [ 2] sin(X) _|_ cos(X) _|_ exp(X) _|_

 _|_ _|_ _|_ _|_ _|_ _|_



 1 _|_ _|x_ 1 _x|_ 2 _|x_ [2] 1 _x_ 1 _|x_ 2 _x|_ [2] 2 sin( _|_ _x_ 1 ) sin( _|x_ 2 ) _· · ·_

 _|_ _|_ _|_ _|_ _|_ _|_ _|_ _|_













(15)



Here, **X** _[P]_ _[q]_ denotes polynomials in X of the order _q_ .


Table 3 presents the results of the experiments performed on two-variables dependent functions, i.e., _f_ ( _x_ 1 _, x_ 2 ).
Similarly to Section 4.1.1, training and test data sets are generated by randomly sampling twenty pairs of points
( _x_ 1 _, x_ 2 ) from a uniform distribution U(-1,1) such that _D_ = _{_ ( _x_ 1 _i_ _, x_ 2 _i_ _, f_ ( _x_ 1 _i_ _, x_ 2 _i_ )) _}_ _[n]_ _i_ =1 [. The same choices for the]
library are considered: _L_ 1 = _{x,_ ( _·_ ) [2] _, · · ·,_ ( _·_ ) [9] _}_ and _L_ 2 = _L_ 1 _∪{_ sin( _·_ ) _,_ cos( _·_ ) _,_ tan( _·_ ) _,_ exp( _·_ ) _,_ sigmoid( _·_ ) _}_ . An exact
match between the ground-truth and predicted function is obtained using _L_ 1 for any polynomial function,
whereas only approximate solutions are obtained for trigonometric functions. The results are approximate of
the ground-truth function using _L_ 2 .


Table 3: Results for multivariate functions using linear SR. _D_ = _{_ ( _x_ 1 _, x_ 2 ) _∈_ U( _−_ 1 _,_ 1 _,_ 20); _y_ = _f_ ( _x_ 1 _, x_ 2 ) _}_ .
_L_ 1 = _{x,_ ( _·_ ) [2] _, · · ·,_ ( _·_ ) [9] _}_ and _L_ 2 = _L_ 1 _∪{_ sin( _·_ ) _,_ cos( _·_ ) _,_ tan( _·_ ) _,_ exp( _·_ ) _,_ sigmoid( _·_ ) _}_ . T and F refer to True and False.


Benchmark Expression _L_ 1 _L_ 2


Result _R_ [2] Result _R_ [2]


Nguyen-12 _x_ [4] 1 _[−]_ _[x]_ [3] 1 [+] [1] 2 _[x]_ 2 [2] _[−]_ _[x]_ [2] T 1.0 F _≈_ 1

Livermore-5 _x_ [4] 1 _[−]_ _[x]_ [3] 1 [+] _[ x]_ [2] 1 _[−]_ _[x]_ [2] T 1.0 F _≈_ 1
Nguyen-9 sin( _x_ 1 ) + sin( _x_ [2] 2 [)] F _≈_ 1 F _≈_ 1
Nguyen-10 2 sin( _x_ 1 ) cos( _x_ 2 ) F _≈_ 1 F _≈_ 1


Furthermore, linear SR is tested on a dataset generated using a two-dimensional multivariate normal distribution
_N_ ( _µ,_ **Σ** ), as shown in Fig. 5. Different analytic expressions for _f_ ( _x_ 1 _, x_ 2 ) were tested with different library bases
that are summarized in Table 4, including pure polynomial basis functions, polynomial and trigonometric basis
functions, and a mixed library.


Figure 5: Two-dimensional multivariate normal distribution used in test applications.


The function _y_ 1 = cos( _x_ 1 ) + sin( _x_ 2 ) is explored with all three bases. In the case of a pure polynomial basis, the
correct terms of the Taylor expansion of both cos( _x_ 1 ) and sin( _x_ 2 ) are identified with only approximate values
of their coefficients, i.e., ˆ _y_ 1 = (0 _._ 88 _−_ 0 _._ 3 _x_ [2] 1 [+ 0] _[.]_ [01] _[x]_ [4] 1 [) + (0] _[.]_ [97] _[ −]_ [0] _[.]_ [2] _[x]_ [3] 2 [), which is reflected in the significantly]
high reconstruction error of the order of 30%. In both bases where trigonometric functions are enlisted, the
correct terms cos( _x_ 1 ) and sin( _x_ 2 ) are identified with an excellent reconstruction error, that is _≥_ 10 _[−]_ [7] . Note
that the lowest reconstruction error is obtained for the library U2, which has the least number of operations
and, consequently, the lowest number of coefficients.


10


Table 4: Library bases used in test problems of Sec. 4.1.2


Name List of functions


U1 1 _,_ X _,_ X _[P]_ [ 2] _,_ X _[P]_ [ 3] _,_ X _[P]_ [ 4]
Library U2 1 _,_ X _,_ X _[P]_ [ 2] _,_ cos(X) _,_ sin(X)


U3 1 _,_ X _,_ X _[P]_ [ 2] _,_ cos(X) _,_ sin(X) _,_ tan(X) _,_ exp(X) _,_ sigmoid(X)


The function _y_ 2 = _x_ [2] 1 [+ cos(] _[x]_ [2] [) is also tested.] For the pure polynomial basis, the reconstructed function
_y_ ˆ 2 = _x_ [2] 1 [+ (0] _[.]_ [83 + 0] _[.]_ [49] _[x]_ [2] _[−]_ _[x]_ [2] 2 [) predicts approximate values with a reconstruction error of] _[ ≤]_ [1%. An excellent]
prediction is made for the other bases, which enlist both operations in _y_ 2 ( _x_ 1 _, x_ 2 ).


In the same exercise, a more complicated function form is tested that includes mixed terms, i.e., _y_ 3 = _x_ 1 (1 +
_x_ 2 ) + cos( _x_ 1 ) _∗_ sin( _x_ 2 ). The difference between the true and the predicted function is illustrated in Fig. 6. The
linear approach performs similarly for all three library bases. A low reconstruction error is obtained because
the operation term cos( _x_ 1 ) _∗_ sin( _x_ 2 ) in _y_ 3 is not enlisted in any of the libraries, showing an important limitation
of the current approach.


Figure 6: Difference between true ( _y_ ) and predicted (ˆ _y_ ) values of the function _y_ = _x_ 1 (1+ _x_ 2 )+cos( _x_ 1 ) _∗_ sin( _x_ 2 ),
for the three libraries defined in Table 4: U1 (left), U2 (center), U3 (right).


**4.2** **Multidimensional case**


The target mathematical expression comprises _m_ components, i.e., Y = [ _y_ 1 _, · · ·, y_ _m_ ], and the goal is to learn
the coefficients of a system of linear equations rather than one mathematical expression. Each component ( _y_ _j_ )
is described by:


y _j_ = _f_ _j_ (x) = � _θ_ _jk_ _h_ _k_ (x) (16)


_k_



In this case, there exist _m_ sparse vectors of coefficients, i.e., Θ = [ _θ_ 1 _· · · θ_ m ]. Consider the Lorenz system,
which is a set of ordinary differential equations that captures nonlinearities in the dynamics of fluid convection.
It consists of three variables _{x_ 1 _, x_ 2 _, x_ 3 _}_ and their first-order derivatives with respect to time _{_ [d] d _[x]_ _t_ [1] _[,]_ [d] d _[x]_ _t_ [2] _[,]_ [d] d _[x]_ _t_ [3] _[}]_ [,]

which we will refer to as _{y_ 1 _, y_ 2 _, y_ 3 _}_ . Using the library of Eq. 13, the system of linear equations is represented
in a matrix form as follows:

 _y_ 1 _y_ 2 _y_ 3  1 _x_ 1 _x_ 2 _x_ [2] 1 _x_ 1 _x_ 2 _x_ [2] 2 exp( _x_ 2 )  _θ_ 1 _θ_ 2 _θ_ 3 




_[x]_ [1] [d] _[x]_ [2]

d _t_ _[,]_ d _t_




_[x]_ [2] [d] _[x]_ [3]

d _t_ _[,]_ d _t_



_θ_ 1 _θ_ 2 _θ_ 3

... ... ...

... ... ...



















 (17)



_y_ 1 _y_ 2 _y_ 3

... ... ...

... ... ...







=










1 _x_ 1 _x_ 2 _x_ [2] 1 _x_ 1 _x_ 2 _x_ [2] 2 exp( _x_ 2 )

... ... ... ... ... ... _· · ·_ ...

... ... ... ... ... ... ...







11


Here, Y _∈_ R _[n][×]_ [3], U(X) _∈_ R _[n][×][k]_ and Θ _∈_ R _[k][×]_ [3], where _n_ is the size of the input data and _k_ is the number of
columns in the library matrix U. The _j_ _[th]_ -component of the Y vector is given by:


_y_ _j_ = _θ_ _j,_ 0 + _θ_ _j,_ 1 _x_ 1 + _θ_ _j,_ 2 _x_ 2 + _θ_ _j,_ 3 _x_ [2] 1 [+] _[ · · ·]_ [ +] _[ θ]_ _[j,k]_ [exp(] _[x]_ [2] [)] (18)


Equation 17 can be written in a compact form as:


y _k_ = U(x _[T]_ ) _θ_ _k_ (19)


The application presented in [33] uses this approach, where the authors aim to learn differential equations that
govern the dynamics of a given system, such as a nonlinear pendulum and the Lorenz system. The approach
successfully learned the exact weights, allowing them to recover the correct governing equations.


An exemplary schematic is illustrated in Figure 7 for the Lorenz system defined by ˙ _x_ = _σ_ ( _y_ _−x_ ), ˙ _y_ = _x_ ( _ρ−z_ ) _−y_,

˙
_z_ = _xy −_ _βz_ . Here _x_, _y_, and _z_ are physical variables and ˙ _x_, ˙ _y_, and ˙ _z_ are their respective time-derivatives. Only
coefficients associated with functions _{x_ 1 _, x_ 1 _x_ 2 _, }_ should be non-zero and equal to the factors shown in the
Lorenz system’s set of equations.



_θ_ 1 _θ_ 2 _θ_ 3



y 1 y 2 y 3



1 _[x]_ 1 _[x]_ 2 _[x]_ 3 _[x]_ [2] 1



=




_· · ·_



Figure 7: Schematic of the system of Eq. 11 for the Lorenz system defined by _y_ 1 = _σ_ ( _x_ 2 _−x_ 1 ), _y_ 2 = _x_ 1 ( _ρ−x_ 3 ) _−x_ 2,
_y_ 3 = _x_ 1 _x_ 2 _−_ _βx_ 3 . A library U(X) of nonlinear functions of the input is constructed. The marked entries in the
_θ_ s vectors denote the non-zero coefficients determining which library functions are active for each of the three
variables _{y_ 1 _, y_ 2 _, y_ 3 _}_ .


In summary, the linear approach is only successful in particular cases and can not be generalized. Its main
limitation is in predefining the model’s structure as a linear combination of nonlinear functions, reducing the
SR problem to solve a system of linear equations. In contrast, the main mission of SR is to learn the model’s
structure and parameters. A direct consequence of this limitation is that the linear approach fails to learn
expressions in many cases: (i) composition of functions (e.g., _f_ ( _x_ ) = _f_ 1 ( _x_ ) _∗_ _f_ 2 ( _x_ )); (ii) multivariate functions
(e.g., exp( _x∗y_ ) _,_ tan( _x_ + _y_ ), etc.); and (iii) functions including multiplicative or additive factors to their arguments
(e.g., exp( _λx_ )). Finally the dimension of the library matrix can be challenging in computing resources for
extended libraries and high-dimensional data sets.

### **5 Nonlinear symbolic regression**


The nonlinear method uses deep neural networks (DNN), known for their great ability to detect and learn
complex patterns directly from data.


DNN has the advantage of being fully differentiable in its free parameters allowing end-to-end training using
back-propagation. This approach searches the target expression by replacing the standard activation functions
in a neural network with elementary mathematical operations. Figure 8 shows an NN-based architecture for
SR called the Equation Learner (EQL) network proposed by Martius and Lampert [47] in comparison with a
standard NN. Only two hidden layers are shown for simple visualization, but the network’s deepness is controlled
as per the case study.


The EQL network uses a multi-layer feed-forward NN with one output node. A linear transformation _z_ [[] _[l]_ []] is
applied at every hidden layer ( _l_ ), followed by a nonlinear transformation _a_ [[] _i_ _[l]_ []] using unary (i.e., one argument)
and binary (i.e., two arguments) activation functions as follows


_z_ [[] _[l]_ []] = _W_ [[] _[l]_ []] _· a_ [[] _[l][−]_ [1]] + _b_ [[] _[l]_ []]

(20)
_a_ [[] _i_ _[l]_ []] [=] _[ f]_ _[i]_ [(] _[z]_ _i_ [[] _[l]_ []] [)]


12


(a) (b)


Figure 8: Exemplary setup of a standard NN (8a) and EQL-NN (8b) with input **x**, output ˆ _y_ and two hidden
layers. In (a), _f_ denotes the activation function usually chosen among _{_ RELU, tanh, sigmoid _}_ while in EQL
each node has a specific activation function drawn from the function class _F_ .


where _{W, b}_ denote the weight parameters and _f_ _i_ denotes individual activation function from the library
_L_ = _{_ identity _,_ ( _·_ ) [n] _,_ cos _,_ sin _,_ exp _,_ log _,_ sigmoid _}_ . In a standard NN, the same activation function is applied
to all hidden units and is typically chosen among _{_ RELU, tanh, sigmoid, softmax, etc. _}_ .


The problem reduces to learn the correct weight parameters _{W_ [[] _[l]_ []] _, b_ [[] _[l]_ []] _}_, whereas the operators of the target
mathematical expression are selected during training. To overcome the interpretability limitation of neural
network-based architectures and to promote simple over complex solutions as a typical formula describing a
physical process, sparsity is enforced by adding a regularization term _l_ 1 to the _l_ 2 loss function such that,



_L_
� _|W_ [[] _[l]_ []] _|_ 1 (21)


_l_ =1



_ℓ_ = [1]

_N_



_N_


ˆ

� _∥y_ ( _x_ _i_ ) _−_ _y_ _i_ _∥_ [2] + _λ_


_i_ =1



Where _N_ denotes the number of data entries and _L_ denotes the number of layers. Whereas this method is endto-end differentiable in NN parameters and scales well to high dimensional problems, back-propagation through
activation functions such as division or logarithm requires simplifications to the search space, thus limiting
its ability to produce simple expressions involving divisions (e.g., [sin ][(] _x_ _[x][/y]_ [)] ). An extended version EQL _[÷]_ [48]

includes only the division, whereas exponential and logarithm activation functions are not included because of
numerical issues.

### **6 Tree expression**


This section discusses SR methods in which a mathematical expression is regarded as a unary-binary tree consisting of internal nodes and terminals. Every tree node represents a mathematical operation (e.g., + _, −, ×,_ sin _,_ log,
etc.) that is drawn from a pre-defined function class _F_ (Section 2.1) and every tree terminal node (or leaf)
represents an operand, i.e., variable or constant, as illustrated for the example shown in Figure 9. Expression
tree-based methods include genetic programming, transformers, and reinforcement learning.


**6.1** **Genetic Programming**


Genetic programming (GP) is an evolutionary algorithm in computer science that searches the space of computer programs to solve a given problem. Starting with a “population” (set) of “individuals” (trees) that is
randomly generated, GP evolves the initial population _T_ _GP_ [(0)] [using a set of evolutionary “transition rules” (op-]
erations) _{r_ _i_ : _f →_ _f | i ∈_ N _}_ that is defined over the tree space. GP evolutionary operations include mutation,
crossover, and selection. The mutation operation introduces random variations to an individual by replacing
one subtree with another randomly generated subtree (Figure 10-right). The crossover operation involves exchanging content between two individuals, for example, by swapping one random subtree of one individual with
another random subtree of another individual (Figure 10-left). Finally, the selection operation is used to select
which individuals from the current population persist onto the next population. A common selection operator
is tournament selection, in which a set of _k_ candidate individuals are randomly sampled from the population,


13


(a) (b)


Figure 9: (a) Expression-tree structure of _f_ ( _x_ ) = _x_ [2] _−_ cos( _x_ ). (b) _f_ ( _x_ ) as a function of _x_ (blue curve) and data
points (red points) generated using _f_ ( _x_ ).


and the individual with the highest fitness i.e., a minimum loss is selected. In a GP algorithm, a single iteration
corresponds to one generation. The application of one generation of GP on a population _T_ _GP_ [(] _[i]_ [)] [produces a new,]
augmented population _T_ _GP_ [(] _[i]_ [+1)] . In each generation, each individual has a probability of undergoing a mutation
operation and a probability of undergoing a crossover operation. The selection is applied when the dimension
of the current population is the same as the previous one. Throughout _M_ _k_ iterations, the following steps are
undertaken: (1) transition rules are applied to the function set _F_ _[k]_ = _{f_ 1 _[k]_ _[,][ · · ·][, f]_ _M_ _[ k]_ _k_ _[}]_ [ such that] _[ f]_ _[ k]_ [+1] [ =] _[ r]_ _[i]_ [(] _[f]_ _[ k]_ [)]
where _k_ denotes the iteration index; (2) the loss function _ℓ_ ( _F_ _[k]_ ) is evaluated for the set; and (3) an elite set
of individuals is selected for the next iteration step. The GP algorithm repeats this procedure until a predetermined accuracy level is achieved.


Figure 10: Crossover (left) and mutation (right) operations on exemplary expression trees in genetic programming.


Whereas GP allows for large variations in the population resulting in improved performance for out-of-distribution
data, GP-based methods do not scale well to high dimensional data sets and are highly sensitive to hyperparameters [19].


**6.2** **Transformers**


Transformer neural network (TNN) is a novel NN architecture introduced by Vaswani _et al._ [46] in natural
language processing (NLP) to model sequential data. TNN is based on the attention mechanism that aims to
model long-range dependencies in a sequence. Consider the English-to-French translation of the two following

sentences:


En: The kid did not go to school because **it** was closed.


14


Fr: L’enfant n’est pas all´e `a l’´ecole parce qu’elle ´etait ferm´ee.


En: The kid did not go to school because **it** was cold.
Fr: L’enfant n’est pas all´e `a l’´ecole parce qu’il faisait froid.


The two sentences are identical except for the last word, which refers to the school in the first sentence (i.e.,
“ **closed** ”) and to the weather in the second one (i.e., “ **cold** ”). Transformers create a context-dependent word
embedding that it pays particular attention to the terms (of the sequence) with high weights. In this example,
the noun that the adjective of each sentence refers to has a significant weight and is therefore considered for
translating the word “it”. Technically, an embedding _x_ _i_ is assigned to each element of the input sequence, and
a set of _m_ key-value pairs is defined, i.e., _S_ = _{_ ( _k_ 1 _, v_ 1 ) _, · · ·,_ ( _k_ _m_ _, v_ _m_ ) _}_ . For each query, the attention mechanism
computes a linear combination of values [�] _j_ _[ω]_ _[j]_ _[v]_ _[j]_ [, where the attention weights (] _[ω]_ _[j]_ _[ ∝]_ _[q][ ·][ k]_ _[j]_ [) are derived using]

the dot product between the query ( _q_ ) and all keys ( _k_ _j_ ), as follows:


Attention( _q, S_ ) = � _σ_ ( _q · k_ _j_ ) _v_ _j_ (22)

_j_


Here, _q_ = _xW_ _q_ is a query, _k_ _i_ = _x_ _i_ _W_ _k_ is a key, _v_ _i_ = _x_ _i_ _W_ _v_ is a value, and _W_ _q_, _W_ _k_, _W_ _v_ are the learnable
parameters. The architecture of the self-attention mechanism is illustrated in Figure 11.


Figure 11: Evaluation of Attention( _q, S_ ) (Eq. 22) for a query _q_ _i_, computed using the input vector embedding

_x_ _i_ .


In the context of SR, both input data points _{_ ( **x** _i_ _, y_ _i_ ) _|_ **x** _i_ _∈_ R _[d]_ _, y_ _i_ _∈_ R _, i ∈_ N _n_ _}_ and mathematical expressions
_f_ are encoded as sequences of symbolic representations as discussed in Section 2.2. The role of the transformer
is to create the dependencies at two levels, first between numerical and symbolic sequences and between tokens
of symbolic sequence. Consider the mathematical expression _f_ ( _x, y, z_ ) = sin( _x/y_ ) _−_ sin( _z_ ), which can be written
as a sequence of tokens following the polish notation:


_−_ sin _÷_ _x_ _y_ sin _z_


Each symbol is associated with an embedding such that:


_x_ 1 : _−_ _x_ 2 : sin _x_ 3 : _÷_ _x_ 4 : _x_ _x_ 5 : _y_ _x_ 6 : sin _x_ 7 : _z_


15


In this particular example, for query ( _x_ 7 : _z_ ), the attention mechanism will give a higher weight for the binary
operator ( _x_ 1 : _−_ ) than for the variable ( _x_ 5 : _y_ ) or the division operator ( _x_ 3 : _÷_ ).


Transformers consist of an encoder-decoder structure; each block comprises a self-attention layer and a feedforward neural network. TNN inputs a sequence of embeddings _{x_ _i_ _}_ and outputs a “context-dependent” sequence of embeddings _{y_ _i_ _}_ one at a time, through a latent representation _z_ _i_ . TNN is an auto-regressive model,
i.e., sampling each symbol is conditioned by the previously sampled symbols and the latent sequence. An example of a TNN encoder is shown in Figure 12.


Figure 12: Structure of a TNN encoder [46]. It comprises an attention layer and a feed-forward neural network.


In symbolic regression case, the encoder and the decoder do not share the same vocabulary because the decoder
has a mixture of symbolic and numeric representations, while the encoder has only numeric representations.
There exist two approaches to solving SR problems using transformers. First is the skeleton approach [32, 49]
where the transformer conducts the two-steps procedure: (1) the decoder predicts a skeleton _f_ _e_, a parametric
function that defines the general shape of the target expression up to a choice of constants, using the function
class _F_ and (2) the constants are fitted using optimization techniques such as the non-linear optimization solver
BFGS. For example, if _f_ = cos(2 _x_ 1 ) _−_ 0 _._ 1 exp( _x_ 2 ), then the decoder predicts _f_ _e_ = cos( _◦_ _x_ 1 ) _−◦_ exp( _x_ 2 ) where

_◦_
denotes an unknown constant. The second is an end-to-end (E2E) approach [31] where both the skeleton and
the numerical values of the constants are simultaneously predicted. Both approaches are further discussed in
Section 7.


**6.3** **Reinforcement learning**


Reinforcement learning provides a framework for learning and decision-making by trial and error [50]. An RL
Setting consists of four components ( _S, A, P, R_ ) in a Markov decision process. In this setting, an agent observes
a state _s ∈S_ of the environment and, based on that, takes action _a ∈A_, which results in a reward _r_ = _R_ ( _s, a_ ),
and the environment then transitions to a new state _s_ _[′]_ _∈S_ . The interaction goes on in time steps until a
terminal state is reached. The aim of the agent is to learn the policy _P_ (also called transition dynamics), which
is a mapping from states to actions that maximize the expected cumulative reward. An exemplary sketch of an
RL-based SR method is illustrated in Figure 13.
SR problem can be framed in RL as follows: the agent (NN) observes the environment (parent and sibling in
a tree) and, based on the observation, takes an action (predict the next token of the sequence) and transitions


16


Figure 13: Exemplary sketch of a general RL-based SR method. _s_ _t_, _a_ _t_, and _r_ _t_ = _R_ ( _s_ _t_ _, a_ _t_ ) denote the state,
action, and reward at time step _t_ . ( _t_ + 1) denotes the next time step.


into a new state. In this view, the NN model is like a policy, the parent and sibling are like observations, and
sampled symbols are like actions.

### **7 Applications**


Most existing algorithms for solving SR are GP-based, whereas many others, and more recent, are deep learning (DL)-based. There exist two different strategies to solve SR problems, as illustrated in the taxonomy of
Figure 14.


Figure 14: Strategies for solving SR problem. An SR algorithm has three types of input: data (x), a new or
reduced representation of the data (x _[′]_ or z), or a model ( _f_ (x)) learned from the data.


The first is a one-step approach, where data points are directly fed into an SR algorithm. A second is a two-step
approach involving a process which either learns a new representation of data or learns a _“blackbox”_ model,
which will be then fed into SR algorithm as described below:


1. Learn a new representation of the original data set by defining new features (reducing the number of independent variables) or a reduced representation using specific NN architectures such as principal component
analysis and autoencoders.


2. Learn a _“blackbox”_ model either using regular NN or using conceptual NN such as graph neural network
(GNN). In this case, an SR algorithm is applied to the learned model or parts of it.


We group the applications based on the categories presented in Section 3, and we summarize them in Table 5.


17


Table 5: Table summarizing symbolic regression applications. D refers to data input, 1 refers to data representation input, and 2 refers to model input to SR.


Application Ref Name/description(code) Year Method Strategy


SINDY [51] Sparse Identification of Nonlinear Dynamics link 2016 Linear D



SINDY-AE [33] Data-driven discovery of coordinates and governing equations ( `[https://github.com/kpchamp/](https://github.com/kpchamp/SindyAutoencoders)`
`[SindyAutoencoders](https://github.com/kpchamp/SindyAutoencoders)` )



2019 Linear 1



EQL [47] Equation learner ( `[https://github.com/](https://github.com/KristofPusztai/EQL)` 2016 non linear D
`[KristofPusztai/EQL](https://github.com/KristofPusztai/EQL)` )


EQL _÷_ [48] Equation learner division ( `[https://github.com/](https://github.com/martius-lab/EQL)` 2018 Non linear D
`[martius-lab/EQL](https://github.com/martius-lab/EQL)` )


Eureqa [26] Commercial software 2011 GP D


FFX [20] Fast function extraction ( `[https://github.com/](https://github.com/natekupp/ffx/tree/master/ffx)` 2011 GP D
`[natekupp/ffx/tree/master/ffx](https://github.com/natekupp/ffx/tree/master/ffx)` )


ITEA [22] Interaction-Transformation Evolutionary Algorithm 2019 GP D
for SR ( `[https://github.com/folivetti/ITEA/](https://github.com/folivetti/ITEA/)` )


MRGP [23] Multiple Regression GP ( `[https://github.com/](https://github.com/flexgp/gp-learners)` 2014 GP D
`[flexgp/gp-learners](https://github.com/flexgp/gp-learners)` )



E2ESR [31] End-to-end SR with transformers
( `[https://github.com/facebookresearch/](https://github.com/facebookresearch/symbolicregression)`
`[symbolicregression](https://github.com/facebookresearch/symbolicregression)` )

NeSymReS [32] Neural SR that scales ( `[https://](https://github.com/SymposiumOrganization/ NeuralSymbolicRegressionThatScales)`
```
         github.com/SymposiumOrganization/
```

`[NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/ NeuralSymbolicRegressionThatScales)` )



2022 TNN D


2021 TNN D



DSR [19] Deep symbolic regression ( `[https://github.com/](https://github.com/brendenpetersen/deep-symbolic-regression)` 2019 RNN,RL D
`[brendenpetersen/deep-symbolic-regression](https://github.com/brendenpetersen/deep-symbolic-regression)` )



NGPPS [29] SR via Neural-Guided GP population seeding ( `[https://github.com/brendenpetersen/](https://github.com/brendenpetersen/deep-symbolic-regression)`
`[deep-symbolic-regression](https://github.com/brendenpetersen/deep-symbolic-regression)` )



2021 RNN,GP,RL D



AIFeynman [27] Physics-inspired method for SR ( `[https://github.](https://github.com/SJ001/AI-Feynman)` 2019 Physics-informed 1
`[com/SJ001/AI-Feynman](https://github.com/SJ001/AI-Feynman)` )


SM [30] Symbolic Metamodel ( `[https://bitbucket.org/](https://bitbucket.org/mvdschaar/mlforhealthlabpub)` 2019 Mathematics 2
`[mvdschaar/mlforhealthlabpub](https://bitbucket.org/mvdschaar/mlforhealthlabpub)` )



GNN [52] Discovering Symbolic Models from DL with Inductive Biases ( `[https://github.com/MilesCranmer/](https://github.com/MilesCranmer/symbolic_deep_learning)`
`[symbolic_deep_learning](https://github.com/MilesCranmer/symbolic_deep_learning)` )


HEAL [53] Heuristic and Evolutionary Algorithms Laboratory ( `[https://github.com/heal-research/](https://github.com/heal-research/HeuristicLab)`
`[HeuristicLab](https://github.com/heal-research/HeuristicLab)` )


18



2020 GNN 2


_−_ Heuristic D


GP-based applications will not be reviewed here; they are listed in the living review [25], along with DL-based
applications. State-of-the-art GP-based methods are discussed in detail in [54]. Among GP-based applications
is the commercial software Eureqa [26], the most well-known GP-based method that uses the algorithm proposed by Schmidt and Lipson in [2]. Eureqa is used as a baseline SR method in several research works.


**SINDY-AE** [33] is a hybrid SR method that combines autoencoder network [55] with linear SR [51]. The
novelty of this approach is in simultaneously learning sparse dynamical models and reduced representations of
coordinates that define the model using snapshot data. Given a data set **x** ( _t_ ) _∈_ R _[n]_, this method seeks to learn
coordinate transformations from original to intrinsic coordinates **z** = _φ_ ( **x** ) (encoder) and back via **x** = _ψ_ ( **z** )
(decoder), along with the dynamical model associated with the set of reduced coordinates **z** ( _t_ ) _∈_ R _[d]_ ( _d ≪_ _n_ ):


_d_
(23)
_dt_ **[z]** [(] _[t]_ [) =] **[ g]** [ (] **[z]** [(] _[t]_ [))]


through a customized loss function _L_, defined as a sum of four terms:



+ _λ_ 2 _∥_ **˙x** _−_ **˙x** pred _∥_ 2 [2]
� ~~�~~ � ~~�~~

decoder loss



+ _λ_ 3 _∥_ Θ _∥_ 1
~~��~~ � ~~�~~
regularizer loss



_L_ = _∥_ **x** _−_ _ψ_ ( _φ_ ( **x** )) _∥_ 2 [2]
~~�~~ ~~�~~ � ~~�~~

reconstruction error



+ _λ_ 1 _∥_ **˙z** _−_ **˙z** pred _∥_ 2 [2]
~~�~~ ~~��~~ ~~�~~

encoder loss



(24)



Here the derivative of the reduced variables **z** are computed using the derivatives of the original variable **x**,
i.e. **˙z** = _∇_ **x** _φ_ ( **x** ) ˙ _x_ . Predicted coordinates denoted as **a** pred represent NN outputs and are expressed in terms of
coefficient vector Θ and library matrix **U** ( **x** ) following Eq. 19, i.e., **z** rec = **U** ( **z** _[T]_ )Θ = **U** ( _φ_ ( **x** ) _[T]_ )Θ. The library
is specified before training, and the coefficients Θ are learned with the NN parameters as part of the training
procedure.


A case study is the nonlinear pendulum motion whose dynamics are governed by a second-order differential equation given by ¨ _x_ = _−_ sin( _x_ ). The data set is generated as a series of snapshot images from a simulated video of a
nonlinear pendulum. After training, the SINDY autoencoder correctly identified the equation ¨ _z_ = _−_ 0 _._ 99 sin _z_,
which is the dynamical model of a nonlinear pendulum in the reduced representation. This approach is particularly efficient when the dynamical model may be dense in terms of functions of the original measurement
coordinates **x** . This method and similar works [56] make the path to “Gopro physics” where researchers point
a camera on an event and get back an equation capturing the underlying phenomenon using an algorithm.


Despite successful applications involving partial differential equations, still, one main limitation of this method
is in its linear SR part. For example, a model expressed as _f_ (x) = _x_ 1 _x_ 2 _−_ 2 _x_ 2 exp( _−x_ 3 ) + [1] 2 [exp(] _[−]_ [2] _[x]_ [1] _[x]_ [3] [) is]

discovered only if each term of this expression is comprised in the library, e.g., exp( _−_ 2 _x_ 1 _x_ 2 ). The presence of
the exponential function, i.e., exp( _x_ ), is insufficient to discover the second and the third terms.


**Symbolic metamodel** [30] (SM) is a _model-of-a-model_ method for interpreting “blackbox” model predictions.
It inputs a learned _“blackbox”_ ) model and outputs a symbolic expression. Available post-hoc methods aim to
explain ML model predictions, i.e., they can explain some aspects of the prediction but can not offer a full
model interpretation. In contrast, SM is interpretable because it uncovers the functional form that underlies
the learned model. The symbolic metamodel is based on Meijer _G_ -function [57, 58], which is a special univariate function characterized by a set of indices, i.e., _G_ _[m,n]_ _p,q_ [(] **[a]** _[p]_ _[,]_ **[ b]** _[q]_ _[|][x]_ [), where] **[ a]** [ and] **[ b]** [ are two sets of real-values]
parameters. An instance of the Meijer _G_ -function is specified by ( **a** _,_ **b** ), for example the function _G_ [1] 2 _[,]_ _,_ [2] 2 [(] _[a,a]_ _a,b_ _[|][x]_ [)]
takes different forms for different settings of the parameters _a_ and _b_, as illustrated in Figure 15.


In the context of SR problem solving, the target mathematical expression is defined as a parameterization
of the Meijer function, i.e., _{g_ ( _x_ ) = _G_ ( _θ,_ **x** ) _| θ_ = ( **a** _,_ **b** ) _}_, thus reducing the optimization task to a standard parameter optimization problem that can be efficiently solved using gradient descent algorithms _θ_ _[k]_ [+1] :=
_θ_ _[k]_ _−γ_ [�] _i_ _[l]_ [(] _[G]_ [(] **[x]** _[i]_ _[, θ]_ [)] _[, f]_ [(] **[x]** _[i]_ [))] _[|]_ _[θ]_ [=] _[θ]_ _[k]_ [. The parameters] **[ a]** [ and] **[ b]** [ are learned during training, and the indices (] _[m, n, p, q]_ [)]

are regarded as hyperparameters of the model. SM was tested on both synthetic and real data and was deployed in two modes spanning (1) only polynomial expressions (SM _[p]_ ) and (2) closed-form expressions (SM _[c]_ ),
in comparison to a GP-based SR method. SM _[p]_ produces accurate polynomial expressions for three out of four
tested functions (except the Bessel function), whereas SM _[c]_ produces the correct ground-truth expression for all
four functions and significantly outperforms GP-based SR.


More generally, consider a problem in a critical discipline such as healthcare. Assuming a feature vector comprising (age, gender, weight, blood pressure, temperature, disease history, profession, etc.) with the aim to
predict the risk of a given disease. Predictions made by a _“blackbox”_ could be highly accurate. However, the
learned model does not provide insights into why the risk is high or low for a patient and what parameter is the
most critical or weightful in the prediction. Applying the symbolic metamodel to the learned model outputs


19


Figure 15: Example of a Meijer G-function _G_ [2] 1 _[,]_ _,_ [2] 1 [(] _[a,a]_ _a,b_ _[|][x]_ [) for different values of] _[ a]_ [ and] _[ b]_ [ [30].]


a symbolic expression, e.g., _f_ ( _x_ 1 _, x_ 2 ) = _x_ 1 (1 _−_ exp( _−x_ 2 )), where _x_ 1 is the blood pressure and _x_ 2 is the age.
Here, we can learn that only two features (out of many others) are crucial for the prediction and that the risk
increases with high blood pressure and decreases with age. This is an ideal example showing the difference between _“blackbox”_ and interpretable models. In addition, it is worth mentioning that methods applied for model
interpretation only exploit part of the prediction and can not unveil how the model captures nonlinearities in
the data. Thus model interpretation methods are insufficient to provide full insights into why and how model
predictions are made and are not by any means equivalent to interpretable models.


**End-to-end symbolic regression** [31] (E2ESR) is a transformer-based method that uses end-to-end learning to solve SR problems. It is made up of three components: (1) an embedder that maps each input point
( _x_ _i_ _, y_ _i_ ) to a single embedding, (2) a fully-connected feedforward network, and (3) a transformer that outputs a
mathematical expression. What distinguishes E2ESR from other transformer-based applications is the use of an
end-to-end approach without resorting to skeletons, thus using both symbolic representations for the operators
and the variables and numeric representations for the constants. Both input data points _{_ ( **x** _i_ _, y_ _i_ ) _| i ∈_ N _n_ _}_ and
mathematical expressions _f_ are encoded as sequences of symbolic representations following the description in
Section 2.2. E2ESR is tested and compared to several GP-based and DL-based applications on SR benchmarks.
Results are reported in terms of mean accuracy, formula complexity, and inference time, and it was shown
E2ESR achieves very competitive results for SR and outperforms previous applications.


**AIFeynman** [27] is a physics-inspired SR method that recursively applies a set of solvers, i.e., dimensional
analysis [3], polynomial fit, and brute-force search to solve an SR problem. If the problem is not solved, the
algorithm searches for simplifying intrinsic properties in data (e.g. invariance, factorization) using NN and
deploys them to recursively simplify the dataset into simpler sub-problems with fewer independent variables.
Each sub-problem is then tackled by a symbolic regression method of choice. The authors created the Feynman
SR database (see Section 8) to test their approach. All the basic equations and 90% of the bonus equations
were solved by their algorithm, outperforming Eureqa.


**Deep Symbolic Regression** (DSR) [19] is an RL-based search method for symbolic regression that uses a
generative recurrent neural network (RNN). RNN defines a probability distribution ( _p_ ( _θ_ )) over mathematical
expressions ( _τ_ ), and batches of expressions _T_ = _{τ_ [(] _[i]_ [)] _}_ _[N]_ _i_ =1 [are stochastically generated. An exemplary sketch]
of how RNN generates an expression (e.g., _x_ [2] _−_ cos( _x_ )) is shown in Figure 16. Starting with the first node
following the pre-order traversal (Section 2.2) of an expression tree, RNN is initially fed with empty placeholders
tokens (a parent and a sibling) and produces a categorical distribution, i.e., outputs the probability of selecting
every token from the defined library _L_ = _{_ + _, −, ×, ÷,_ sin _,_ cos _,_ log _,_ etc _.}_ . The sampled token is fed into the
first node, and the number of siblings is determined based on whether the operation is unary (one sibling)
or binary (two siblings). The second node is then selected, and the RNN is fed with internal weights along
with the first token and outputs a new (and potentially different) categorical distribution. This procedure is
repeated until the expression is complete. Expressions are then evaluated with a reward function _R_ ( _τ_ ) to test
the goodness of the fit to the data _D_ for each candidate expression ( _f_ ) using normalized root-mean-square error,



1 _n_
_n_ ~~�~~ _i_ =1 [(] _[y]_ _[i]_ _[ −]_ _[f]_ [(X] _[i]_ [))] [2] ~~�~~ .



1
_R_ ( _τ_ ) = 1 _/_ �1 + _σ_ _y_



�



3 Dimensional analysis is a well-known technique in physics that uses set of units of measurements to solve an equation and/or
to check the correctness of a given equation.


20


Figure 16: Exemplary sketch of RNN generating a mathematical expression _x_ [2] _−_ cos( _x_ ).


To generate better expressions ( _f_ ), the probability distribution _p_ ( _τ_ _|θ_ ) needs to be optimized. Using a gradientbased approach for optimization requires the reward function _R_ ( _τ_ ) to be differentiable with respect to the RNN
parameter _θ_, which is not the case. Instead, the learning objective is defined as the expectation of the reward
under expressions from the policy, i.e., _J_ ( _θ_ ) = E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ )], and reinforcement learning is used to maximize
_J_ ( _θ_ ) by means of the “standard policy gradient”:


_∇_ _θ_ _J_ ( _θ_ ) = _∇_ _θ_ E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ )] = E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ ) _∇_ _θ_ log _p_ ( _τ_ _|θ_ )] (25)


This reinforcement learning trick, called REINFORCE [59], can be derived using the definition of the expectation
E[ _·_ ] and the derivative of log( _·_ ) function as follows:



_∇_ _θ_ E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ )] = _∇_ _θ_



_R_ ( _τ_ ) _p_ ( _τ_ _|θ_ ) _dθ_
�



= _R_ ( _τ_ ) _∇_ _θ_ _p_ ( _τ_ _|θ_ ) _dθ_
�

= _R_ ( _τ_ ) _[∇]_ _[θ]_ _[p]_ [(] _[τ]_ _[|][θ]_ [)]
� _p_ ( _τ_ _|θ_ ) _[p]_ [(] _[τ]_ _[|][θ]_ [)] _[dθ]_


= _R_ ( _τ_ ) log( _p_ ( _τ_ _|θ_ ) _p_ ( _τ_ _|θ_ ) _dθ_
�

= E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ ) _∇_ _θ_ log _p_ ( _τ_ _|θ_ )]



(26)



The importance of this result is that it allows estimating the expectation using samples from the distribution.
More explicitly, the gradient of _J_ ( _θ_ ) is estimated by computing the mean over a batch of N sampled expressions
as follows:



_∇_ _θ_ _J_ ( _θ_ ) = [1]

_N_



_N_
� _R_ ( _τ_ [(] _[i]_ [)] ) _∇_ _θ_ log _p_ ( _τ_ [(] _[i]_ [)] _|θ_ ) (27)


_i_ =1



The standard policy gradient (Eq. 25) permits optimizing a policy’s average performance over all samples from
the distribution. Since SR requires maximizing best-case performance, i.e., to optimize the gradient over the
top _ϵ_ fraction of samples from the distribution found during training, a new learning objective is defined as a
conditional expectation of rewards above the (1 _−_ _ϵ_ )-quantile of the distribution of rewards, as follows:


_J_ risk ( _θ, ϵ_ ) = E _τ_ _∼p_ ( _τ_ _|θ_ ) [ _R_ ( _τ_ ) _| R_ ( _τ_ ) _≥_ _R_ _ϵ_ ( _θ_ )] (28)


21


where _R_ _ϵ_ ( _θ_ ) represent the samples from the distribution below the _ϵ_ -threshold. The gradient of the new learning
objective is given by:


_∇_ _θ_ _J_ risk ( _θ_ ) = E _τ_ _∼p_ ( _τ_ _|θ_ ) [( _R_ ( _τ_ ) _−_ _R_ _ϵ_ ( _θ_ )) _· ∇_ _θ_ log _p_ ( _τ_ _|θ_ ) _| R_ ( _τ_ ) _≥_ _R_ _ϵ_ ( _θ_ )] (29)


DSR was essentially evaluated on the Nguyen SR benchmark and several additional variants of this benchmark.
An excellent recovery rate was reported for each set, and DSR solved all mysteries except the Nguyen-12 benchmark given by _x_ [4] _−_ _x_ [3] + 2 [1] _[y]_ [2] _[ −]_ _[y]_ [. More details on SR data benchmarks can be found in Section 8.]


**Neural-guided genetic programming population seeding** [29] (NGPPS) is a hybrid method that combines GP and RNN [19] and leverages the strengths of each of the two components. Whereas GP begins with
random starting populations, the authors in [29] propose to use the batch of expressions sampled by RNN as a
staring population for GP: _T_ _GP_ [(0)] [=] _[ T]_ _[RNN]_ [. Each iteration of the proposed algorithm consists of 4 steps: (1) The]
batch of expressions sampled by RNN is passed as a starting population to GP, (2) _S_ generations of GP are
performed and result in a final GP population _T_ _GP_ _[S]_ [, (3) An elite set of top-performing GP samples is selected]
_T_ _GP_ _[E]_ [and passed to the gradient update of RNN.]


Figure 17: Neural-guided genetic programming population seeding method overview [29].


**Neural symbolic regression that scales** [32] (NeSymReS) is a transformer-based algorithm that emphasizes
large-scale pre-training. It comprises a pre-training and test phase. Pre-training includes data generation and
model training. Hundreds of millions of training examples are generated for every minibatch in pre-training.
Each training example consists of a symbolic equation _f_ _e_ and a set of _n_ input-output pairs _{_ **x** _i_ _, y_ _i_ = _f_ ( **x** _i_ ) _}_
where _n_ can vary across examples, and the number of independent input variables is at most three. In the test
phase, a set of input-output pairs _{x_ _i_ _, y_ _i_ _}_ is fed into the encoder that maps it into a latent vector _z_, and the
decoder iteratively samples candidates’ skeletons. What distinguishes this method is the learning task, i.e., it
improves over time with experience, and there is no need to be retrained from scratch on each new experiment.
It was shown that NeSymReS outperforms selected baselines (including DSR) in time and accuracy by a large
margin on all datasets (AI-Feynman, Nguyen, and strictly out-of-sample equations (SOOSE) with and without
constants). NeSymReS is more than three orders of magnitudes faster at reaching the same maximum accuracy
as GP while only running on CPU.


**GNN** [52] is a hybrid scheme performing SR by training a Graph Neural Network (GNN) and applying SR
algorithms on GNN components to find mathematical equations.


A case study is Newtonian dynamics which describes the dynamics of particles in a system according to Newton’s laws of motion. _D_ consists of an N-body system with known interaction (force law _F_ such as electric,
gravitation, spring, etc.), where particles (nodes) are characterized by their attributes (mass, charge, position,
velocity, and acceleration) and their interaction (edges) are assigned the attribute of dimension 100. The GNN
functions are trained to predict instantaneous acceleration for each particle using the simulated data and then
applied to a different data sample. The study shows that the most significant edge attributes, say _{e_ 1 _, e_ 2 _}_,
fit to a linear combination of the true force components, _{F_ 1 _, F_ 2 _}_, which were used in the simulation showing


22


that edge attributes can be interpreted as force laws. The most significant edge attributes were then passed
into Eureqa to uncover analytical expressions that are equivalent to the simulated force laws. The proposed
approach was also applied to datasets in the field of cosmology, and it discovered an equation that fits the data
better than the existing hand-designed equation.


The same group has recently succeeded in inferring Newton’s law for gravitational force using GNN and _PySR_
for symbolic regression task [34]. GNN was trained using observed trajectories (position) of the Sun, planets, and
moons of the solar system collected during 30 years. The SR algorithm could correctly infer Newton’s formula
that describes the interaction between masses, i.e., _F_ = _−GM_ 1 _M_ 2 _/r_ [2], and the masses and the gravitational
constant as well.

### **8 Datasets**


For symbolic regression purposes, there exist several benchmark data sets that can be categorized into two main
groups: (1) ground-truth problems (or synthetic data) and (2) real-world problems (or real data), as summarized in Figure 18. In this section, we describe each category and discuss its main strength and limitations.













Figure 18: Taxonomy based on the type of SR benchmark problems.


Ground-truth regression problems are characterized by known mathematical equations, they are listed in Table 6.
These include (1) physics-inspired equations [27, 60] and (2) real-valued symbolic equations [1, 14, 15, 16, 17,
18, 19, 61].


Table 6: Table summarizing ground-truth problems for symbolic regression.


Type Benchmark Number of problems Year Reference


Feynman Database 119 2019 [27]
Physics-related

Strogatz Repository 10 2011 [60]



Mathematics-related



Koza 3 1994 [1]
Keijzer 15 2003 [14]
Vladislavleva 8 2009 [15]
Nguyen 12 2011 [17]
Korns 15 2011 [16]
R 3 2013 [61]
Jin 6 2019 [18]
Livermore 22 2021 [19]



The Feynman Symbolic Regression Database [62] is the largest SR database that originates from Feynman
lectures on Physics series [63, 64] and is proposed in [27]. It consists of 119 [4] physics-inspired equations that
describe static physical systems and various physics processes. The proposed equations depend on at least
one variable and, at most, nine variables. Each benchmark (corresponding to one equation) is generated by
randomly sampling one million entries. Each entry is a row of randomly generated input variables, which are
sampled uniformly between 1 and 5. This range of sampling was slightly adjusted for some equations to avoid
unphysical results (e.g., division by zero or the square root of a negative number). The output is evaluated
using function _f_, e.g. _D_ = _{_ **x** _i_ _∈_ R _[d]_ _, y_ _i_ = _f_ ( _x_ 1 _, · · ·, x_ _d_ ) _}_ .


4 The equation number II.11.17 is missing in the benchmark repository.


23


This benchmark is rich in proposing various theoretical formulae. Still, it suffers a few limitations: (1) there
is no distinction between variables and constants, i.e., constants are randomly sampled and, in some cases,
in domains extremely far from physical values. For example, the speed of light is sampled from a uniform
distribution _U_ (1 _,_ 20) whereas its physical value is orders of magnitude higher, i.e., _c_ = 2 _._ 988 _×_ 10 [8] m/s, and
the gravitational constant is sampled from _U_ (1 _,_ 2) whereas its physical value is orders of magnitude smaller,
_G_ = 6 _._ 6743 _×_ 10 _[−]_ [11] m [3] kg _[−]_ [1] s _[−]_ [2], among others (e.g., vacuum permittivity _ϵ ∼_ 10 _[−]_ [12], Boltzmann constant
_k_ _b_ _∼_ 10 _[−]_ [23], Planck constant _h ∼_ 10 _[−]_ [34] ). (2) Some variables are sampled in nonphysical ranges. For example,
the gravitational force is defined between two masses distant by _r_ as _F_ = _Gm_ 1 _m_ 2 _/r_ [2] . This force is weak unless
defined between significantly massive objects (e.g., the mass of the earth is _M_ _e_ = 5 _._ 9722 _×_ 10 [24] kg) whereas _m_ 1
and _m_ 2 are sampled in _U_ (1 _,_ 5) in the Feynman database. (3) Some variables are treated as floats while they are
integers, and (4) many equations are duplicates of each other (e.g., a multiplicative function of two variables
_f_ ( _x, y_ ) = _x ∗_ _y_ ) or have similar functional forms.
The ODE-Strogatz repository [60] consists of ten physics equations that describe the behavior of dynamical
systems which can exhibit chaotic and/or non-linear behavior. Each dataset is one state of a two-state system
of ordinary differential equations.


Within the same category, there exist several benchmarks [1, 14, 15, 16, 17, 18, 19] consisting of real-valued
symbolic functions. The majority of these benchmarks are proposed for GP-based methods and grouped into
four categories: polynomial, trigonometric, logarithmic, exponential, and square-root functions, and a combination of univariate and bivariate functions. The suggested functions do not have any physical meaning, and most
depend either on one or two independent variables. Datasets are generally generated by randomly sampling
either 20 or 100 points in narrow ranges. The most commonly known is the so-called Nguyen benchmark, which
consists of 12 symbolic functions taken from [65, 66, 67]. Only four equations have the scalars _{_ 1,2,1/2 _}_ as
constants therein. Each benchmark is defined by a ground-truth expression, training, and test datasets. The
equations proposed in these benchmarks can not be found in a single repository. Therefore we list them in the
appendix in Tables [7-9] and Tables [10-11] for completeness and for easy comparison.


Real-world problems are characterized by an unknown model that underlies data. This category comprises two
groups: observations and measurements. Data sets in the observations category can originate from any domain,
such as health informatics, environmental science, business, commerce, etc. Data could be collected online or
offline from reports or studies. A wide range of problems can be assessed from the following repositories: the
PMLB [68], the OpenML [69], and the UCI [70]. An exemplary application in this category is wind speed
forecasting [71]. Measurements represent sets of data points that are collected (and sometimes analyzed) in
physics experiments. Here the target model is either an underlying theory than can be derived from first
principles or not. In the first case, symbolic regression would either infer the correct model structure and
parameters or contribute to the theory development of the studied process, whereas in the second case, the
symbolic regression output could be the awaited theory.

### **9 Discussion**


SR is a growing area of ML and is gaining more attention as interpretability is increasingly promoted [3] in AI
applications. SR is propelled by the fact that ML models are becoming very big in parameters at the expense of
making accurate predictions. An exemplary application is the chatGPT-4, a large language model comprising
hundreds of billions of parameters and trained on hundreds of terabytes of textual data. Such big models are
very complicated networks. ChatGPT-4, for example, is accomplishing increasingly complicated and intelligent
tasks to the point that it is showing emergent properties [72]. However, it is not straightforward to understand
when it works and, more importantly, when it does not. In addition, its performance improves with increasing
the number of parameters, highlighting that its prediction accuracy depends on the size of the training data
set. Therefore, a new paradigm is needed, especially in scientific disciplines, such as physical sciences, where
problems are of causal hypothesis-driven nature. SR is by far the most potential candidate to fulfill the interpretability requirements and is expected to play a central role in the future of ML.


Despite the significant advances made in this subfield and the high performance of most deep learning-based
SR methods proposed in the literature, still, SR methods fail to recover relatively simple relationships. A case
in point is the Nguyen-12 expression, i.e., _f_ ( _x, y_ ) = _x_ [4] _−_ _x_ [3] + _y_ [2] _/_ 2 _−_ _y_, where _x_ and _y_ are uniformly sampled
in the range [0 _,_ 1]. The NGPPS method could not recover this particular expression using the library basis
_L_ = _{_ + _, −, ×, ÷,_ sin _,_ cos _,_ exp _,_ log _, x, y}_ . A variant of this expression, Nguyen-12 _[⋆]_, consisting of the same equation but defined over a larger domain, i.e., data points sampled in [0 _,_ 10], was successfully covered using the same
library, with a recovery rate of 12%. This result is significantly below the perfect performance on all other Nguyen
expressions. A similar observation is made for the Livermore-5 whose expression is _f_ ( _x, y_ ) = _x_ [4] _−_ _x_ [3] + _x_ [2] _−_ _y_ .


24


We ran NGPPS on Nguyen-12 with two libraries, a pure polynomial basis _L_ 1 = _{_ + _, −, ×, ÷,_ ( _·_ ) [2] _,_ ( _·_ ) [3] _,_ ( _·_ ) [4] _, x, y}_
and a mixed basis _L_ 2 = _L_ 1 _∪{_ sin _,_ cos _,_ exp _,_ log _,_ sqrt _,_ expneg _}_ . The algorithm succeeds in recovering Nguyen-12
only using a pure polynomial basis with a recovery rate of 3%. The same observation is made by applying
linear SR on Nguyen-12. This highlights how strongly the predicted expression depends on the set of allowable
mathematical operations. A practical way to encounter this limitation is to implement basic domain knowledge
in SR applications whenever possible. For example, astronomical data collected by detecting the light curves of
astronomical objects exhibit periodic behavior. In such cases, periodic functions such as trigonometric functions
should be part of the library basis.


Most SR methods are only applied to synthetic data for which the input-output relationship is known. This is
justified because the methods must be cross-checked, and their performance must be evaluated using groundtruth expressions. However, the reported results are for synthetic data only. To the best of our knowledge, only
one physics application [34] succeeded in extracting New’s laws of gravitation by applying SR to astronomical
data. The absence of such applications leads us to state that SR is still a relatively nascent area with the
potential to make a big impact. Physics in general, and physical sciences in particular, represent a very broad
field for SR development purposes and are very rich both in data and expressions, e.g., areas such as astronomy
and high-energy physics are very rich in data. In addition, lots of our acquired knowledge in physics can be
used for SR methods test purposes because underlying phenomena and equations are well known. All that is
needed is greater effort and investment.

### **10 Conclusion**


This work presents an in-depth introduction to the symbolic regression problem and an expansive review of its
methodologies and state-of-the-art applications. Also, this work highlights a number of conclusions that can
be made about symbolic regression methods, including (1) linear symbolic regression suffer many limitations,
all originating from predefining the model structure, (2) neural network-based methods lead to numerical issues and the library can not include all mathematical operations, (3) expression tree-based methods are yet
the most powerful in terms of model performance on synthetic data, in particular transformer-based ones, (4)
model predictions strongly depend on the set of allowable operations in the library basis, and (5) generally,
deep learning-based methods are performing better than other ML-based methods.


Symbolic regression represents a powerful tool for learning interpretable models in a data-driven manner. Its
application is likely to grow in the future because it balances prediction accuracy and interpretability. Despite
the limited SR application to real data, the few existing ones are very promising. A potential path to boost
progress in this subfield is to apply symbolic regression to experimental data in physics.

### **References**


[1] J. R. Koza, “Genetic programming as a means for programming computers by natural selection,” _Proceed-_
_ings of the National Academy of Sciences_, vol. 4, no. 2, pp. 87–112, 1994.


[2] M. Schmidt and H. Lipson, “Distilling free-form natural laws from experimental data,” _Science_, vol. 324,
no. 5923, pp. 81–85, 2009.


[3] C. Rudin, “Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,” _Nature Machine Intelligence_, vol. 1, no. 5, pp. 206–215, 2019.


[4] M. Mozaffari-Kermani, S. Sur-Kolay, A. Raghunathan, and N. K. Jha, “Systematic poisoning attacks on
and defenses for machine learning in healthcare,” _IEEE Journal of Biomedical and Health Informatics_,
vol. 19, no. 6, pp. 1893–1905, 2015.


[5] J. Kepler, _Epitome astronomiae Copernicanae_ . in: Noscemus Wiki.


[6] I. Newton, A. Motte, and J. Machin, _The Mathematical Principles of Natural Philosophy_ . No. Volume 1
in The Mathematical Principles of Natural Philosophy, London: B. Motte, 1729.


[7] D. Gerwin, “Information processing, data inferences, and scientific generalization,” _Systems Research and_
_Behavioral Science_, vol. 19, pp. 314–325, 1974.


[8] P. Langley, “Data-driven discovery of physical laws,” _Cognitive Science_, vol. 5, no. 1, pp. 31–54, 1981.


25


[9] B. C. Falkenhainer and R. S. Michalski, “Integrating quantitative and qualitative discovery: The abacus
system,” _Mach. Learn._, vol. 1, p. 367–401, mar 1986.


[10] L. P.W., “Bacon: A production system that discovers empirical laws,” 1979.


[11] P. Langley, H. A. Simon, G. L. Bradshaw, and J. M. Zytkow, _Scientific Discovery: Computational Explo-_
_rations of the Creative Process_ . Cambridge, MA, USA: MIT Press, 1987.


[12] J. R. Koza, “Hierarchical genetic algorithms operating on populations of computer programs,” in _Pro-_
_ceedings of the 11th International Joint Conference on Artificial Intelligence - Volume 1_, IJCAI’89, (San
Francisco, CA, USA), p. 768–774, Morgan Kaufmann Publishers Inc., 1989.


[13] J. R. Koza, “Genetic programming: A paradigm for genetically breeding populations of computer programs
to solve problems,” tech. rep., Stanford, CA, USA, 1990.


[14] M. Keijzer, “Improving symbolic regression with interval arithmetic and linear scaling,” in _Genetic Pro-_
_gramming_ (C. Ryan, T. Soule, M. Keijzer, E. Tsang, R. Poli, and E. Costa, eds.), (Berlin, Heidelberg),
pp. 70–82, Springer Berlin Heidelberg, 2003.


[15] E. Vladislavleva, G. Smits, and D. den Hertog, “Order of nonlinearity as a complexity measure for models
generated by symbolic regression via pareto genetic programming,” _IEEE Transactions on Evolutionary_
_Computation_, vol. 13, pp. 333–349, 2009.


[16] M. F. Korns, _Accuracy in Symbolic Regression_, pp. 129–151. New York, NY: Springer New York, 2011.


[17] N. Q. Uy, N. X. Hoai, M. O’Neill, R. I. McKay, and E. G. L´opez, “Semantically-based crossover in genetic programming: application to real-valued symbolic regression,” _Genetic Programming and Evolvable_
_Machines_, vol. 12, pp. 91–119, 2010.


[18] Y. Jin, W. Fu, J. Kang, J. Guo, and J. Guo, “Bayesian symbolic regression,” 2019.


[19] B. K. Petersen, “Deep symbolic regression: Recovering mathematical expressions from data via policy
gradients,” _CoRR_, vol. abs/1912.04871, 2019.


[20] T. McConaghy, _FFX: Fast, Scalable, Deterministic Symbolic Regression Technology_, pp. 235–260. New
York, NY: Springer New York, 2011.


[21] M. Virgolin, T. Alderliesten, C. Witteveen, and P. A. N. Bosman, “A model-based genetic programming
approach for symbolic regression of small expressions,” _CoRR_, vol. abs/1904.02050, 2019.


[22] F. O. de Fran¸ca and G. S. I. Aldeia, “Interaction-transformation evolutionary algorithm for symbolic
regression,” _CoRR_, vol. abs/1902.03983, 2019.


[23] I. Arnaldo, K. Krawiec, and U.-M. O’Reilly, “Multiple regression genetic programming,” in _Proceedings_
_of the 2014 Annual Conference on Genetic and Evolutionary Computation_, GECCO ’14, (New York, NY,
USA), p. 879–886, Association for Computing Machinery, 2014.


[24] W. G. L. Cava, T. R. Singh, J. Taggart, S. Suri, and J. H. Moore, “Stochastic optimization approaches to
learning concise representations,” _CoRR_, vol. abs/1807.00981, 2018.


[25] N. Makke and S. Chawla, “A living review of symbolic regression,” _https://github.com/nmakke/SR-_
_LivingReview_, 2022.


[26] R. Dubcakova, “Eureqa: software review,” _Genetic Programming and Evolvable Machines_, vol. 12, pp. 173–
178, June 2011.


[27] S.-M. Udrescu and M. Tegmark, “Ai feynman: a physics-inspired method for symbolic regression,” 2019.


[28] G. Martius and C. H. Lampert, “Extrapolation and learning equations,” _CoRR_, vol. abs/1610.02995, 2016.


[29] T. N. Mundhenk, M. Landajuela, R. Glatt, C. P. Santiago, D. M. Faissol, and B. K. Petersen, “Symbolic
regression via neural-guided genetic programming population seeding,” _CoRR_, vol. abs/2111.00053, 2021.


[30] A. M. Alaa and M. van der Schaar, “Demystifying black-box models with symbolic metamodels,” in
_Advances in Neural Information Processing Systems_ (H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´eBuc, E. Fox, and R. Garnett, eds.), vol. 32, (New York), Curran Associates, Inc., 2019.


[31] P.-A. Kamienny, S. d’Ascoli, G. Lample, and F. Charton, “End-to-end symbolic regression with transformers,” 2022.


26


[32] L. Biggio, T. Bendinelli, A. Neitz, A. Lucchi, and G. Parascandolo, “Neural symbolic regression that
scales,” _CoRR_, vol. abs/2106.06427, 2021.


[33] K. Champion, B. Lusch, J. N. Kutz, and S. L. Brunton, “Data-driven discovery of coordinates and governing
equations,” _Proceedings of the National Academy of Sciences_, vol. 116, no. 45, pp. 22445–22451, 2019.


[34] P. Lemos, N. Jeffrey, M. Cranmer, S. Ho, and P. Battaglia, “Rediscovering orbital mechanics with machine
learning,” 2022.


[35] R. Batra, L. Song, and R. Ramprasad, “Emerging materials intelligence ecosystems propelled by machine
learning,” _Nature Reviews Materials_, vol. 6, pp. 655–678, Nov. 2020.


[36] A. Hernandez, A. Balasubramanian, F. Yuan, S. Mason, and T. Mueller, “Fast, accurate, and transferable
many-body interatomic potentials by symbolic regression,” 2019.


[37] Y. Wang, N. Wagner, and J. M. Rondinelli, “Symbolic regression in materials science,” _MRS Communica-_
_tions_, vol. 9, pp. 793–805, sep 2019.


[38] B. Weng, Z. Song, R. Zhu, Q. Yan, Q. Sun, C. G. Grice, Y. Yan, and W.-J. Yin, “Simple descriptor derived
from symbolic regression accelerating the discovery of new perovskite catalysts,” _Nature Communications_,
vol. 11, 2020.


[39] J. Martinez-Gil and J. M. Chaves-Gonzalez, “A novel method based on symbolic regression for interpretable
semantic similarity measurement,” _Expert Systems with Applications_, vol. 160, p. 113663, 2020.


[40] I. A. Abdellaoui and S. Mehrkanoon, “Symbolic regression for scientific discovery: an application to wind
speed forecasting,” _2021 IEEE Symposium Series on Computational Intelligence (SSCI)_, pp. 01–08, 2021.


[41] M. Virgolin, Z. Wang, T. Alderliesten, and P. A. N. Bosman, “Machine learning for the prediction of pseudorealistic pediatric abdominal phantoms for radiation dose reconstruction,” _Journal of Medical Imaging_,
vol. 7, no. 4, p. 046501, 2020.


[42] W. G. L. Cava, P. Orzechowski, B. Burlacu, F. O. de Fran¸ca, M. Virgolin, Y. Jin, M. Kommenda,
and J. H. Moore, “Contemporary symbolic regression methods and their relative performance,” _CoRR_,
vol. abs/2107.14351, 2021.


[43] V. Vapnik, “Principles of risk minimization for learning theory,” in _Advances in Neural Information Pro-_
_cessing Systems_ (J. Moody, S. Hanson, and R. Lippmann, eds.), vol. 4, (Cambridge, Massachusetts),
Morgan-Kaufmann, 1991.


[44] M. Virgolin and S. P. Pissis, “Symbolic regression is np-hard,” 2022.


[45] R. Robinson, “Jan �Lukasiewicz: Aristotle’s syllogistic from the standpoint of modern formal logic. second
edition enlarged. pp. xvi 222. oxford: Clarendon press, 1957. cloth, 305. net.,” _The Classical Review_, vol. 8,
no. 3-4, p. 282–282, 1958.


[46] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin,
“Attention is all you need,” _CoRR_, vol. abs/1706.03762, 2017.


[47] G. Martius and C. H. Lampert, “Extrapolation and learning equations,” _CoRR_, vol. abs/1610.02995, 2016.


[48] S. S. Sahoo, C. H. Lampert, and G. Martius, “Learning equations for extrapolation and control,” _CoRR_,
vol. abs/1806.07259, 2018.


[49] M. Valipour, B. You, M. Panju, and A. Ghodsi, “Symbolicgpt: A generative transformer model for symbolic
regression,” _CoRR_, vol. abs/2106.14131, 2021.


[50] R. S. Sutton and A. G. Barto, _Reinforcement Learning: An Introduction_ . Cambridge, MA, USA: A Bradford
Book, 2018.


[51] S. L. Brunton, J. L. Proctor, and J. N. Kutz, “Discovering governing equations from data by sparse
identification of nonlinear dynamical systems,” _Proceedings of the National Academy of Sciences_, vol. 113,
no. 15, pp. 3932–3937, 2016.


[52] M. D. Cranmer, A. Sanchez-Gonzalez, P. W. Battaglia, R. Xu, K. Cranmer, D. N. Spergel, and S. Ho,
“Discovering symbolic models from deep learning with inductive biases,” _CoRR_, vol. abs/2006.11287, 2020.


[53] Heuristic and E. A. Laboratory.


27


[54] W. La Cava, K. Danai, and L. Spector, “Inference of compact nonlinear dynamic models by epigenetic
local search,” _Engineering Applications of Artificial Intelligence_, vol. 55, pp. 292–306, 2016.


[55] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, _Learning Internal Representations by Error Propaga-_
_tion_, p. 318–362. Cambridge, MA, USA: MIT Press, 1986.


[56] B. Chen, K. Huang, S. Raghupathi, I. Chandratreya, Q. Du, and H. Lipson, “Discovering state variables
hidden in experimental data,” 2021.


[57] C. Meijer, _On the g-function_ . Netherlands: North-Holland, 1946.


[58] R. Beals and J. Szmigielski, “Meijer g-functions: a gentle introduction,” _Notices of the American Mathe-_
_matical Society_, vol. 60, pp. 866–873, 2013.


[59] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,”
_Mach. Learn._, vol. 8, p. 229–256, may 1992.


[60] W. La Cava, K. Danai, and L. Spector, “Inference of compact nonlinear dynamic models by epigenetic
local search,” _Engineering Applications of Artificial Intelligence_, vol. 55, pp. 292–306, 2016.


[61] K. Krawiec and T. Pawlak, “Approximating geometric crossover by semantic backpropagation,” in _Pro-_
_ceedings of the 15th Annual Conference on Genetic and Evolutionary Computation_, GECCO ’13, (New
York, NY, USA), p. 941–948, Association for Computing Machinery, 2013.


[62] M. Tegmark, “The feynman symbolic regression database,” May 2019.


[63] R. Feynman, R. Leighton, and M. Sands, _The Feynman Lectures on Physics, Vol. I: The New Millennium_
_Edition: Mainly Mechanics, Radiation, and Heat_ . The Feynman Lectures on Physics, New York: Basic
Books, 2011.


[64] R. Feynman, R. Leighton, M. Sands, and M. Gottlieb, _The Feynman Lectures on Physics_ . No. Volume 2
in The Feynman Lectures on Physics, Boston: Pearson/Addison-Wesley, 2006.


[65] M. Keijzer, “Improving symbolic regression with interval arithmetic and linear scaling,” in _Genetic Pro-_
_gramming_ (C. Ryan, T. Soule, M. Keijzer, E. Tsang, R. Poli, and E. Costa, eds.), (Berlin, Heidelberg),
pp. 70–82, Springer Berlin Heidelberg, 2003.


[66] N. Hoai, R. McKay, D. Essam, and R. Chau, “Solving the symbolic regression problem with tree-adjunct
grammar guided genetic programming: the comparative results,” in _Proceedings of the 2002 Congress on_
_Evolutionary Computation. CEC’02 (Cat. No.02TH8600)_, vol. 2, pp. 1326–1331 vol.2, 2002.


[67] C. G. Johnson, “Genetic programming crossover: Does it cross over?,” in _Genetic Programming_ (L. Vanneschi, S. Gustafson, A. Moraglio, I. De Falco, and M. Ebner, eds.), (Berlin, Heidelberg), pp. 97–108,
Springer Berlin Heidelberg, 2009.


[68] R. S. Olson, W. La Cava, P. Orzechowski, R. J. Urbanowicz, and J. H. Moore, “Pmlb: a large benchmark
suite for machine learning evaluation and comparison,” _BioData Mining_, vol. 10, pp. 1–13, Dec 2017.


[69] J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo, “Openml: networked science in machine learning,”
_SIGKDD Explorations_, vol. 15, no. 2, pp. 49–60, 2013.


[70] D. Dua and C. Graff, “UCI machine learning repository,” 2017.


[71] I. A. Abdellaoui and S. Mehrkanoon, “Symbolic regression for scientific discovery: an application to wind
speed forecasting,” 2021.


[72] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou,
D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, “Emergent abilities of
large language models,” 2022.


[73] U.-M. O’Reilly, “Genetic programming ii: Automatic discovery of reusable programs.,” _Artificial Life_,
vol. 1, no. 4, pp. 439–441, 1994.


28


### **A Datasets Benchmarks Equations**

Table 7: Ground-truth expressions for Koza [73], Nguyen [17], Jin [18], Keijzer [14] and R [61] benchmarks.


Dataset Expression Variables Data range


Koza-1 [5] _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U[-1, 1, 20]
Koza-2 _x_ [5] _−_ 2 _x_ [3] + _x_ 1 U[-1, 1, 20]
Koza-3 _x_ [6] _−_ 2 _x_ [4] + _x_ [2] 1 U[-1, 1, 20]


Nguyen-1 _x_ [3] + _x_ [2] + _x_ 1 U(-1,1,20)
Nguyen-2 _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U(-1,1,20)
Nguyen-3 _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U(-1,1,20)
Nguyen-4 _x_ [6] + _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U(-1,1,20)
Nguyen-5 sin( _x_ [2] ) cos( _x_ ) _−_ 1 1 U(-1,1,20)
Nguyen-6 sin( _x_ ) + sin( _x_ + _x_ [2] ) 1 U(-1,1,20)
Nguyen-7 log( _x_ + 1) + log( _x_ [2] + 1) 1 U(0,2,20)
Nguyen-8 _√_ ~~_x_~~ 1 U(0,4,20)
Nguyen-9 sin( _x_ ) + sin( _y_ [2] ) 2 U(-1,1,100)
Nguyen-10 2 sin( _x_ ) cos( _y_ ) 2 U(-1,1,100)
Nguyen-11 _x_ _[y]_ 2
Nguyen-12 _x_ [4] _−_ _x_ [3] + [1] 2 _[y]_ [2] _[ −]_ _[y]_ 2

Jin-1 2 _._ 5 _x_ [4] _−_ 1 _._ 3 _x_ [3] + 0 _._ 5 _y_ [2] _−_ 1 _._ 7 _y_ 2 U(-3,3,100)
Jin-2 8 _._ 0 _x_ [2] + 8 _._ 0 _y_ [3] _−_ 15 _._ 0 2 U(-3,3,100)
Jin-3 0 _._ 2 _x_ [3] + 1 _._ 5 _y_ [3] _−_ 1 _._ 2 _y −_ 0 _._ 5 _x_ 2 U(-3,3,100)
Jin-4 1 _._ 5 exp( _x_ ) + 5 _._ 0 cos( _y_ ) 2 U(-3,3,100)
Jin-5 6 _._ 0 sin( _x_ ) cos( _y_ ) 2 U(-3,3,100)
Jin-6 1 _._ 35 _xy_ + 5 _._ 5 sin(( _x −_ 1 _._ 0)( _y −_ 1 _._ 0) 2 U(-3,3,100)


Keijzer-1 0 _._ 3 _x_ sin(2 _πx_ ) 1 E[-1, 1, 0.1]
Keijzer-2 0 _._ 3 _x_ sin(2 _πx_ ) 1 E[-2, 2, 0.1]
Keijzer-3 0 _._ 3 _x_ sin(2 _πx_ ) 1 E[-3, 3, 0.1]
Keijzer-4 _x_ [3] _e_ _[−][x]_ cos( _x_ ) sin( _x_ )(sin [2] ( _x_ ) cos( _x_ ) _−_ 1) 1 E[0, 10, 0.05]

_x, z_ : U[-1,1,1000]
Keijzer-5 30 _xz/_ ( _x −_ 10) _y_ [2] 3
_y_ : U[1,2,1000]
Keijzer-6 � _x_ 1 _[i]_ 1 E[1, 50, 1]
Keijzer-7 log _x_ 1 E[1, 100, 1]
Keijzer-8 _√_ ~~_x_~~ 1 E[0, 100, 1]
Keijzer-9 arcsinh( _x_ ) = log( _x_ + _√x_ [2] + 1) 1 E[0, 100, 1]

Keijzer-10 _x_ _[y]_ 2 U[0, 1, 100]
Keijzer-11 _xy_ + sin(( _x −_ 1)( _y −_ 1)) 2 U[-3, 3, 20]
Keijzer-12 _x_ [4] _−_ _x_ [3] + _y_ [2] _/_ 2 _−_ _y_ 2 U[-3, 3, 20]
Keijzer-13 6 sin( _x_ ) cos( _y_ ) 2 U[-3, 3, 20]
Keijzer-14 8 _/_ (2 + _x_ [2] + _y_ [2] ) 2 U[-3, 3, 20]
Keijzer-15 _x_ [3] _/_ 5 + _y_ [3] _/_ 2 _−_ _y −_ _x_ 2 U[-3, 3, 20]


R1 ( _x_ + 1) [3] _/_ ( _x_ [2] _−_ _x_ + 1) 1 E[-1,1,20]
R2 ( _x_ [5] _−_ 3 _x_ [3] + 1) _/_ ( _x_ [2] + 1) 1 E[-1,1,20]
R3 ( _x_ [6] + _x_ [5] ) _/_ ( _x_ [4] + _x_ [3] + _x_ [2] + _x_ + 1) 1 E[-1,1,20]


29


Table 8: Ground-truth expressions for Korns [16] and Livermore [19] benchmarks.


Dataset Expression Variables Data range



Korns-1 1 _._ 57 + (24 _._ 3 _v_ ) 1 U[-50, 50, 10000]
Korns-2 0 _._ 23 + 14 _._ 2 _[v]_ 3 [+] _ω_ _[y]_ 3 U[-50, 50, 10000]

Korns-3 _−_ 5 _._ 41 + 4 _._ 9 _[v][−][x]_ 3 [+] _ω_ _[y/][w]_ 4 U[-50, 50, 10000]

Korns-4 _−_ 2 _._ 3 + 0 _._ 13 sin( _z_ ) 1 U[-50, 50, 10000]
Korns-5 3 + 2 _._ 13 ln( _ω_ ) 1 U[-50, 50, 10000]
Korns-6 1 _._ 3 + 0 _._ 13 _[√]_ ~~_x_~~ 1 U[-50, 50, 10000]
Korns-7 213 _._ 80940889(1 _−_ _e_ _[−]_ [0] _[.]_ [54723748542] _[x]_ ) 1 U[-50, 50, 10000]
Korns-8 6 _._ 87 + 11 _√_ 7 _._ 23 _x v ω_ 3 U[-50, 50, 10000]

~~_√x_~~ _e_ _[z]_
Korns-9 ln( _y_ ) _v_ [2] 4 U[-50, 50, 10000]

Korns-10 0 _._ 81 + 24 _._ 3 4 [2] _v_ _[y]_ [3] [+] +5 [3] _[z]_ _ω_ [2][4] 4 U[-50, 50, 10000]



Korns-11 6 _._ 87 + 11 cos(7 _._ 23 _x_ [3] ) 1 U[-50, 50, 10000]
Korns-12 2 _−_ 2 _._ 1 cos(9 _._ 8 _x_ ) sin(1 _._ 3 _ω_ ) 2 U[-50, 50, 10000]

tan( _z_ )

Korns-13 32 _−_ 3 [tan] tan( [(] _[x]_ _y_ ) [)] tan( _v_ ) 4 U[-50, 50, 10000]



Korns-11 6 _._ 87 + 11 cos(7 _._ 23 _x_ [3] ) 1 U[-50, 50, 10000]
Korns-12 2 _−_ 2 _._ 1 cos(9 _._ 8 _x_ ) sin(1 _._ 3 _ω_ ) 2 U[-50, 50, 10000]

tan( _z_ )

Korns-13 32 _−_ 3 [tan] tan( [(] _[x]_ _y_ ) [)] tan( _v_ ) 4 U[-50, 50, 10000]

Korns-14 22 _−_ 4 _._ 2(cos( _x_ ) _−_ tan( _y_ )) [tanh] sin( [(] _v_ _[z]_ ) [)] 4 U[-50, 50, 10000]



Korns-14 22 _−_ 4 _._ 2(cos( _x_ ) _−_ tan( _y_ )) [tanh][(] _[z]_ [)]



Korns-15 12 _−_ 6 [tan] _e_ _[y]_ [(] _[x]_ [)] (ln( _z_ ) _−_ tan( _v_ )) 4 U[-50, 50, 10000]

Livermore-1 1 _/_ 3 + _x_ + sin( _x_ [2] ) 1 U[-10,10,1000]
Livermore-2 sin( _x_ [2] ) cos( _x_ ) _−_ 2 1 U[-1,1,20]
Livermore-3 sin( _x_ [3] ) cos( _x_ [2] ) _−_ 1 1 U[-1,1,20]
Livermore-4 log( _x_ + 1) + log( _x_ [2] + 1) + log( _x_ ) 1 U[0,2,20]
Livermore-5 _x_ [4] _−_ _x_ [3] + _x_ [2] _−_ _y_ 2 U[0,1,20]
Livermore-6 4 _x_ [4] + 3 _x_ [3] + 2 _x_ [2] + _x_ 1 U[-1,1,20]
Livermore-7 sinh( _x_ ) 1 U[-1,1,20]
Livermore-8 cosh( _x_ ) 1 U[-1,1,20]
Livermore-9 _x_ [9] + _x_ [8] + _x_ [7] + _x_ [6] + _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U[-1,1,20]
Livermore-10 6 sin( _x_ ) cos( _y_ ) 2 U[0,1,20]
Livermore-11 _x_ [2] _y_ [2] _/_ ( _x_ + _y_ ) 2 U[-1,1,50]
Livermore-12 _x_ [5] _/y_ [3] 2 U[-1,1,50]
Livermore-13 _x_ [1] _[/]_ [3] 1 U[0,4,20]
Livermore-14 _x_ [3] + _x_ [2] + _x_ + sin( _x_ ) + sin( _x_ [2] ) 1 U[-1,1,20]
Livermore-15 _x_ [1] _[/]_ [5] 1 U[0,4,20]
Livermore-16 _x_ [2] _[/]_ [5] 1 U[0,4,20]
Livermore-17 4 sin( _x_ ) cos( _y_ ) 2 U[0,1,20]
Livermore-18 sin( _x_ [2] ) cos( _x_ ) _−_ 5 1 U[-1,1,20]
Livermore-19 _x_ [5] + _x_ [4] + _x_ [2] + _x_ 1 U[-1,1,20]
Livermore-20 exp( _−x_ [2] ) 1 U[-1,1,20]
Livermore-21 _x_ [8] + _x_ [7] + _x_ [6] + _x_ [5] + _x_ [4] + _x_ [3] + _x_ [2] + _x_ 1 U[-1,1,20]
Livermore-22 exp( _−_ 0 _._ 5 _x_ [2] ) 1 U[-1,1,20]


Table 9: Ground-truth expressions for Vladislavleva [15] benchmark.


Dataset Expression Variables Data range



_e_ _[−]_ [(] _[x][−]_ [1)2]
Vladislavleva-1



Vladislavleva-1 1 _._ 2+( _e_ _y−_ 2 _._ 5) [2] 1 U[0.3, 4, 100]

Vladislavleva-2 _e_ _[−][x]_ _x_ [3] (cos _x_ sin _x_ )(cos _x_ sin [2] _x −_ 1) 2 E[0.5, 10, 0.1]
Vladislavleva-3 _e_ _[−][x]_ _x_ [3] (cos _x_ sin _x_ )(cos _x_ sin [2] _x −_ 1)( _y −_ 5) 2 _x_ :E[0.05,10,0.1]
_y_ :E[0.05,10.05,2]
Vladislavleva-4 5+ ~~[�]~~ [5] _i_ =1 10 [(] _[x]_ _[i]_ _[−]_ [3)] [2] 5 U[0.05, 6.05, 1024]



( _z−_ 1)
Vladislavleva-5 30( _x −_ 1) _y_ [2] ( _x−_ 10) 3 _x_ : U[0.05, 2, 300]
_y_ : U[1, 2, 300]
_z_ : U[0.05, 2, 300]
Vladislavleva-6 6 sin( _x_ ) cos( _y_ ) 2 U[0.1, 5.9, 30]
Vladislavleva-7 ( _x −_ 3)( _y −_ 3) + 2 sin(( _x −_ 4)( _y −_ 4)) 2 U[0.05, 6.05, 300]
( _x−_ 3) [4] +( _y−_ 3) [3] _−_ ( _y−_ 3)
Vladislavleva-8 2



( _y−y_ 2) _−_ [4] +10 _−_ _y−_ 2 U[0.05, 6.05, 50]



30


Table 10: Feynman physics equation [27].



Function form #v



Function form #v



_f_ = exp( _−θ_ [2] _/_ 2) _/_ ~~�~~



_f_ = exp( _−θ_ [2] _/_ 2) _/_ ~~�~~ (2 _π_ ) 1

_f_ = exp( _−_ ( _θ/σ_ ) [2] _/_ 2) _/_ (�(2 _π_ ) _σ_ ) 2



_f_ = exp( _−_ ( _θ/σ_ ) [2] _/_ 2) _/_ (�(2 _π_ ) _σ_ ) 2

_f_ = exp( _−_ (( _θ −_ _θ_ 1 ) _/σ_ ) [2] _/_ 2) _/_ ( ~~�~~ (2 _π_ ) _σ_ ) 3



_f_ = exp( _−_ (( _θ −_ _θ_ 1 ) _/σ_ ) [2] _/_ 2) _/_ ( ~~�~~ (2 _π_ ) _σ_ ) 3

_d_ = ~~�~~ ( _x_ 2 _−_ _x_ 1 ) [2] + ( _y_ 2 _−_ _y_ 1 ) [2] 4



_d_ = ~~�~~ ( _x_ 2 _−_ _x_ 1 ) [2] + ( _y_ 2 _−_ _y_ 1 ) [2] 4

_F_ = (( _x_ 2 _−x_ 1 ) [2] +( _Gmy_ 2 _−_ 1 _my_ 1 2) [2] +( _z_ 2 _−z_ 1 ) [2] 9
_m_ = _m_ 0 3
~~_√_~~ 1 _−v_ [2] _/c_ [2]

_A_ = _x_ 1 _y_ 1 + _x_ 2 _y_ 2 + _x_ 3 _y_ 3 6
_F_ = _µN_ _n_ 2
_F_ = _q_ 1 _q_ 2 _/_ (4 _πϵr_ [2] ) 4
_E_ _f_ = _q_ 1 _r/_ (4 _πϵr_ [3] ) 3
_F_ = _q_ 2 _E_ _f_ 2
_F_ = _q_ ( _E_ _f_ + _Bv_ sin( _θ_ )) 5
_K_ = 1 _/_ 2 _m_ ( _v_ [2] + _u_ [2] + _w_ [2] ) 4
_U_ = _Gm_ 1 _m_ 2( _r_ [1] 2 _[−]_ _r_ [1] 1 [)] 5

_U_ = _mgz_ 3
_U_ = [1] 2

2 _[k]_ _[spring]_ _[x]_ [2]
_x_ _[′]_ = ( _x −_ _ut_ ) _/_ ~~�~~ 1 _−_ _u_ [2] _/c_ [2] 4



_m_ 0 3

1 _−v_ [2] _/c_ [2]



_n_ = _n_ 0 exp( _−mgx/_ ( _k_ _b_ _T_ )) 6
_L_ _rad_ = _hω_ [¯] [3] _/_ ( _π_ [2] _c_ [2] (exp( _hω/_ [¯] ( _k_ _b_ _T_ )) _−_ 1)) 5
_v_ = _mu_ _drift_ _qV_ _e_ _/d_ 4
_D_ = _µ_ _e_ _k_ _b_ _T_ 3
_κ_ = 1 _/_ ( _γ −_ 1) _k_ _b_ _v/A_ 4
_E_ = _nk_ _b_ _T_ ln( _V_ _[V]_ 1 [ 2] [)] 5

_c_ = �( _γpr/ρ_ ) 3



_c_ = �( _γpr/ρ_ ) 3

_E_ = _mc_ [2] _/_ �1 _−_ _v_ [2] _/c_ [2] 3




[1]

_r_ 2 _[−]_ _r_ [1]



_E_ = _mc_ [2] _/_ �1 _−_ _v_ [2] _/c_ [2] 3

_x_ = _x_ 1 (cos( _ωt_ ) + _α_ cos( _ωt_ ) [2] ) 4
_P_ = _κ_ ( _T_ 2 _−_ _T_ 1 ) _A/d_ 5
_F_ _E_ = _Pwr/_ (4 _πr_ [2] ) 2
_V_ _e_ = _q/_ (4 _πϵr_ ) 3
_V_ _e_ = 4 _πϵ_ 1 _[p]_ _[d]_ [ cos(] _[θ]_ [)] _[/][r]_ [2] 4
_E_ _f_ = 4 _πϵ_ 3 _[p]_ _[d]_ _[z/r]_ [5] [�] _x_ [2] + _y_ [2] 6



_x_ _[′]_ = ( _x −_ _ut_ ) _/_ ~~�~~ 1 _−_ _u_ [2] _/c_ [2] 4

_t_ _[′]_ = ( _t −_ _ux/c_ [2] ) _/_ �1 _−_ _u_ [2] _/c_ [2] 4



_t_ _[′]_ = ( _t −_ _ux/c_ [2] ) _/_ �1 _−_ _u_ [2] _/c_ [2] 4

_p_ = _m_ 0 _v/_ �1 _−_ _v_ [2] _/c_ [2] 3



_p_ = _m_ 0 _v/_ �1 _−_ _v_ [2] _/c_ [2] 3

_v_ _[′]_ = ( _u_ + _v_ ) _/_ (1 + _uv/c_ [2] ) 3
_r_ = ( _m_ 1 _r_ 1 + _m_ 2 _r_ 2 ) _/_ ( _m_ 1 + _m_ 2) 4
_τ_ = _rF_ sin( _θ_ ) 3
_L_ = _mrv_ sin( _θ_ ) 4
_E_ = 4 [1] _[m]_ [(] _[ω]_ [2] [ +] _[ ω]_ 0 [2] [)] _[x]_ [2] 4

_V_ _e_ = _q/C_ 2
_θ_ 1 = arcsin( _n_ sin( _θ_ 2)) 2
_f_ _f_ = 1 _/_ ( _d_ [1] [+] _d_ _[n]_ [)] 3



_E_ _f_ = 4 _πϵ_ _[p]_ _[d]_ _[z/r]_ [5] [�] _x_ [2] + _y_ [2] 6

_E_ _f_ = 4 _πϵ_ 3 _[p]_ _[d]_ [ cos(] _[θ]_ [) sin(] _[θ]_ [)] _[/r]_ [3] 4
_E_ = [3] 3

5 _[q]_ [2] _[/]_ [(4] _[πϵd]_ [)]
_E_ _den_ = _ϵEf_ [2] _/_ 2 2
_E_ _f_ = _σ_ _den_ _/ϵ_ 1 _/_ (1 + _χ_ ) 3
_x_ = _qE_ _f_ _/_ ( _m_ ( _ω_ 0 [2] _[−]_ _[ω]_ [2] [))] 5
_n_ = _n_ 0 (1 + _p_ _d_ _E_ _f_ cos( _θ_ ) _/_ ( _k_ _b_ _T_ )) 6
_P_ _⋆_ = _n_ _rho_ _p_ [2] _d_ _[E]_ _[f]_ _[/]_ [(3] _[k]_ _[b]_ _[T]_ [)] 5
_P_ _⋆_ = _nα/_ (1 _−_ ( _nα/_ 3)) _ϵE_ _f_ 4
_θ_ = 1 + _nα/_ (1 _−_ ( _nα/_ 3)) 2
_B_ = 1 _/_ (4 _πϵc_ [2] )2 _I/r_ 4
_ρ_ _c_ = _ρ_ _c_ 0 _/_ ~~�~~ 1 _−_ _v_ [2] _/c_ [2] 3



_ρ_ _c_ = _ρ_ _c_ 0 _/_ ~~�~~ 1 _−_ _v_ [2] _/c_ [2] 3

_j_ = _ρ_ _c_ 0 _v/_ ~~�~~ 1 _−_ _v_ [2] _/c_ [2] 3



_f_ _f_ = 1 _/_ ( _d_ 1 [+] _d_ _[n]_ 2 [)] 3

_k_ = _ω/c_ 2
_x_ = � _x_ [2] 1 [+] _[ x]_ [2] 2 _[−]_ [2] _[x]_ [1] _[x]_ [2] [ cos(] _[θ]_ [1] _[ −]_ _[θ]_ [2] [)] 4




_[n]_
_d_ 1 [+] _d_



_x_ = � _x_ [2] 1 [+] _[ x]_ [2] 2 _[−]_ [2] _[x]_ [1] _[x]_ [2] [ cos(] _[θ]_ [1] _[ −]_ _[θ]_ [2] [)] 4

_I_ _⋆_ = _I_ 0 _⋆_ sin [2] ( _nθ/_ 2) _/_ sin [2] ( _θ/_ 2) 3
_θ_ = arcsin( _λ/nd_ ) 3
_P_ = _q_ [2] _a_ [2] _/_ (6 _πϵc_ [3] ) 4
_P_ = (1 _/_ 2 _ϵcE_ _f_ [2] [)(8] _[πr]_ [2] _[/]_ [3)(] _[ω]_ [4] _[/]_ [(] _[ω]_ [2] _[ −]_ _[ω]_ 0 [2] [)] [2] [)] 6
_ω_ = _qvB/p_ 4
_ω_ = _ω_ 0 _/_ (1 _−_ _v/c_ ) 3
_ω_ = (1 + _v/c_ ) _/_ �1 _−_ _v_ [2] _/c_ [2] _ω_ 0 3



_j_ = _ρ_ _c_ 0 _v/_ ~~�~~ 1 _−_ _v_ [2] _/c_ [2] 3

_E_ = _−µ_ _M_ _B_ cos( _θ_ ) 3
_E_ = _−p_ _d_ _E_ _f_ cos( _θ_ ) 3
_V_ _e_ = _q/_ (4 _πϵr_ (1 _−_ _v/c_ )) 5
_k_ = ~~�~~ _ω_ [2] _/c_ [2] _−_ _π_ [2] _/d_ [2] 3



_ω_ = (1 + _v/c_ ) _/_ �1 _−_ _v_ [2] _/c_ [2] _ω_ 0 3

_E_ = _hω_ [¯] 2
_I_ _⋆_ = _I_ 1 + _I_ 2 + 2 _[√]_ _I_ 1 _I_ 2 cos( _δ_ ) 3
_r_ = 4 _πϵh_ [¯] [2] _/_ ( _mq_ [2] ) 4
_E_ = [3] 2

2 _[p]_ _[F]_ _[ V]_
_E_ = 1 _/_ ( _γ −_ 1) _p_ _F_ _V_ 3
_P_ _F_ = _nkb_ _T_ _/V_ 4



_k_ = ~~�~~ _ω_ [2] _/c_ [2] _−_ _π_ [2] _/d_ [2] 3

_F_ _E_ = _ϵcE_ _f_ [2] 3
_E_ _den_ = _ϵE_ _f_ [2] 2
_I_ = _qv/_ (2 _πr_ ) 3
_µ_ _M_ = _qvr/_ 2 3
_ω_ = _gqB/_ (2 _m_ ) 4
_µ_ _M_ = _qh/_ [¯] (2 _m_ ) 3
_E_ = _gµ_ _M_ _BJ_ _z_ _/h_ [¯] 5
_M_ = _n_ _rho_ _µ_ _M_ tanh( _µ_ _M_ _B/_ ( _k_ _b_ _T_ )) 5
_f_ = _µ_ _m_ _B/_ ( _k_ _b_ _T_ ) + ( _µ_ _m_ _α_ ) _/_ ( _ϵc_ [2] _k_ _b_ _T_ ) _M_ 8
_E_ = _µ_ _M_ (1 + _χ_ ) _B_ 6
_F_ = _Y A_ _x_ _/d_ 4
_µ_ _S_ = _Y/_ (2(1 + _σ_ )) 2
_E_ = _hω/_ [¯] (exp( _hω/_ [¯] ( _k_ _b_ _T_ )) _−_ 1) 4



31


Table 11: Feynman physics equation [27].


Function form # variables

_n_ = 1 _/_ (exp( _hω/_ [¯] ( _k_ _b_ _T_ )) _−_ 1) 4
_n_ = _n_ 0 _/_ (exp( _µ_ _m_ _B/_ ( _k_ _b_ _T_ )) + exp( _−µ_ _m_ _B/_ ( _k_ _b_ _T_ )))
_ω_ = 2 _µ_ _M_ _B/h_ [¯] 3
_p_ _γ_ = sin( _E_ _n_ _t/h_ [¯] ) [2] 3
_p_ _γ_ = ( _p_ _d_ _E_ _f_ _t/h_ [¯] ) sin(( _ω −_ _ω_ 0 ) _t/_ 2) [2] _/_ (( _ω −_ _ω_ 0 ) _t/_ 2) [2] 6



_E_ = _µ_ _M_
~~�~~



_B_ _x_ [2] + _B_ _y_ [2] + _B_ _z_ [2] 3



_L_ = _nh_ [¯] 2
_v_ = 2 _E_ _n_ _d_ [2] _k/h_ [¯] 4
_I_ = _I_ 0 (exp( _qV_ _e_ _/_ ( _k_ _b_ _T_ )) _−_ 1) 5
_E_ = 2 _U_ (1 _−_ cos( _kd_ )) 3
_m_ = _h_ [¯] [2] _/_ (2 _E_ _n_ _d_ [2] ) 3
_k_ = 2 _πα/_ ( _nd_ ) 3
_f_ = _β_ (1 + _α_ cos( _θ_ )) 3
_E_ = _−mq_ [4] _/_ (2(4 _πϵ_ ) [2] _h_ [¯] [2] )(1 _/n_ [2] ) 4
_j_ = _−ρ_ _c_ 0 _qA_ _vec_ _/m_ 4



32


