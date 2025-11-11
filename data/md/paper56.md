# Training of Quantum Circuits on a Hybrid Quantum Computer

#### D. Zhu, [1] [∗] N. M. Linke, [1] M. Benedetti, [3] [,] [4] K. A. Landsman [1], N. H. Nguyen [1], C. H. Alderete [1], A. Perdomo-Ortiz [3] [,] [7], N. Korda [5], A. Garfoot [5], C. Brecque [5], L. Egan [1], O. Perdomo [6], C. Monroe [1] [,] [2]

1 Joint Quantum Institute, Department of Physics,


and Joint Center for Quantum Information and Computer Science,


University of Maryland, College Park, MD 20742, USA

2 IonQ, Inc., College Park, MD 20740

3 Department of Computer Science, University College London,


WC1E 6BT London, UK,


4 Cambridge Quantum Computing Limited, CB2 1UB Cambridge, UK

5 Mind Foundry Limited, OX2 7DD Oxford, UK

6 Department of Mathematics, Central Connecticut State University,


New Britain, CT 06050, USA


7 Zapata Computing Inc., 439 University Avenue,


Office 535, Toronto, ON, M5G 1Y8


_∗_ To whom correspondence should be addressed; E-mail: daiwei@terpmail.umd.edu.


**Generative modeling is a flavor of machine learning with applications ranging**


**from computer vision to chemical design. It is expected to be one of the tech-**


**niques most suited to take advantage of the additional resources provided by**


**near-term quantum computers. Here we implement a data-driven quantum**


**circuit training algorithm on the canonical Bars-and-Stripes data set using a**


**quantum-classical hybrid machine. The training proceeds by running param-**


**eterized circuits on a trapped ion quantum computer, and feeding the results**


1


**to a classical optimizer. We apply two separate strategies, Particle Swarm and**


**Bayesian optimization to this task. We show that the convergence of the quan-**


**tum circuit to the target distribution depends critically on both the quantum**


**hardware and classical optimization strategy. Our study represents the first**


**successful training of a high-dimensional universal quantum circuit, and high-**


**lights the promise and challenges associated with hybrid learning schemes.**

### **One Sentence Summary**


We train generative modeling circuits on a quantum-classical hybrid computer showing opti

mization strategy and resource trade-off.

### **Introduction**


Hybrid quantum algorithms ( _1_ ) use both classical and quantum resources to solve potentially


difficult problems. This approach is particularly promising for current quantum computers of


limited size and power ( _2_ ). Several variants of hybrid quantum algorithms have recently been


demonstrated, such as the Variational Quantum Eigensolver (VQE) for quantum chemistry and


related applications ( _3–7_ ), and the Quantum Approximate Optimization Algorithm (QAOA) for


graph or other optimization problems ( _8, 9_ ). Hybrid quantum algorithms can also be used for


generative models, which aim to learn representations of data in order to make subsequent tasks


easier. Applications of generative modeling include computer vision ( _10_ ), speech synthesis


( _11_ ), the inference of missing text ( _12_ ), de-noising of images ( _13_ ), and chemical design ( _14_ ).


Here, we apply a hybrid quantum learning scheme on a trapped ion quantum computer ( _15_ ) to


accomplish a generative modeling task.


Data-driven quantum circuit learning (DDQCL) is a hybrid framework for generative mod

eling of classical data where the model consists of a parameterized quantum circuit ( _16_ ). The


2


model is trained by sampling the output of a quantum computer and updating the circuit param

eters using a classical optimizer. After convergence, the optimal circuit produces a quantum


state that captures the correlations in the training data sets. Hence the trained circuit serves as a


generative model for the training data. Theoretical results suggest that such generative models


have more expressive power than widely used classical neural networks ( _17, 18_ ). This is be

cause instantaneous quantum polynomial circuits – special cases of the parameterized quantum


circuits used for generative modeling – cannot be efficiently simulated by classical means.


The Bars-and-Stripes (BAS) data set is a canonical body of synthetic data for generative


modeling ( _19_ ). It can be easily visualized in terms of images containing horizontal bars or


vertical stripes, where each pixel represents a qubit. Here, we use the uniformly distributed


2-by-2 BAS shown in Fig.1 in a proof-of-principle generative modeling task on a trapped-ion


quantum computer. This is the first successful demonstration of generative quantum circuits


trained on multi-qubit quantum hardware. We note that there has been a single-qubit experiment


in this context ( _20_ ). We compare the performance of different classical optimization algorithms


and conclude that Bayesian optimization shows significant advantages over Particle Swarm


Optimization for this task.


The experiment is performed on four qubits within a seven-qubit fully programmable trapped


ion quantum computer ( _21_ ) (see Method). With individual addressing and readout of all qubits,


the system can perform sequences of gates from a universal gates set, composed of Ising gates


and arbitrary rotations ( _15_ ). In order to run the large number of variational circuit instances nec

essary for the data-driven learning, we calibrate single- and two-qubit gates and execute lists of


circuits in an automated fashion.


The training pipeline is illustrated in Fig. 1. The quantum circuits are structured as layers


of parameterized gates. We use two types of layers, involving single-qubit rotations and two

qubit entangling gates. A single-qubit layer sandwiches an X-rotation between two Z-rotations


3


Figure 1: Data-driven quantum circuit learning (DDQCL) is a hybrid quantum algorithm
scheme that can be used for generative modeling, illustrated here by the example of 2-by-2
Bars and Stripes (BAS) data. From top left, clockwise: A parametrized circuit is initialized at
random. Then at each iteration, the circuit is executed on a trapped ion quantum computer. The
probability distribution of measurement is compared on a classical computer against the BAS
target data set. Next, the quantified difference is used to optimize the parametrized circuit. This
learning process is iterated until convergence.


4


on each qubit _i_, or _R_ _z_ [(] _[i]_ [)] [(] _[α]_ _i_ [)] _[R]_ _x_ [(] _[i]_ [)] [(] _[β]_ _i_ [)] _[R]_ _z_ [(] _[i]_ [)] [(] _[γ]_ _i_ [)][, involving twelve rotation parameters for the four]


qubits (see Fig. 2). An entangling layer applies Ising or XX gates between all pairs of qubits


according to any imposed connectivity graph. This is expressed as a sequence of _XX_ _[i,j]_ ( _χ_ _i,j_ )


operations as shown in Fig. 2), with up to six entangling parameters ( _15_ ) for four qubits. Due


to the universality of this gate set, a sufficiently long sequence of layers of these two types can


produce arbitrary unitaries.





















Figure 2: Connectivity graphs and corresponding training circuits. Top: Fully-connected training circuit layer, with layers of rotations (square boxes) and entanglement gates (rounded boxes)
between any pair of the four qubits. Bottom: Star-connectivity training circuit layer, with restricted entangling gates. In either case, each rotation (denoted by X or Z) and each entanglement gate (denoted by XX) includes a distinct control parameter, for a total of 18 parameters
for the fully-connected circuit layer and 15 parameters for the star-connected circuit layer. We
remove the first Z rotation (dashed square box) acting on the initial state _|_ 0 _⟩_, resulting in 14 and
11 parameters, respectively. The connectivity figures on the left define the mapping between
the four qubits and the pixels of the BAS images (see Fig.1).

.


At the start of DDQCL, all the rotation and entangling parameters are initialized with ran

dom values. Next the circuit is repeatedly executed on the trapped ion quantum computer in


order to reconstruct the state distribution. A classical computer then compares the measured


distribution with the target distribution and quantifies the difference using a cost function (see


Method for details). A classical optimization algorithm then varies the parameters. We iterate


5


the entire process until convergence.


We impose two distinct connectivity graphs in a four-qubit circuit: all-to-all and star, as


shown in Fig.2. With star connectivity, entanglement between certain qubit-pairs cannot occur


within a single gate layer, which means more layers are necessary for certain target distributions.


Comparing the training process between circuits of different connectivity provides insight into


the performance of DDQCL algorithms on platforms with more limited interaction graphs.


For each connectivity graph, we add layers until the goal of reproducing the BAS data with


the trained model is achieved. The match between training data and model is limited by noise,


experimental throughput rate (how fast the system can process circuits), and sampling errors.


The cost function used in optimization scores the result, but a successful training process must


be able to generate data that can be qualitatively recognized as a BAS pattern to ensure that the


system provides usable results in the spirit of generative modeling in machine learning ( _22_ ).


We now describe the classical optimization strategies for the training algorithm. Although


gradient-based approaches were recently proposed for DDQCL ( _23_ ), we employ gradient-free


optimization schemes that appear less sensitive to noise and experimental throughput. We ex

plore two such schemes: Particle Swarm Optimization (PSO) ( _24_ ) and Bayesian Optimization


(BO) ( _25_ ). PSO is a stochastic optimization scheme commonly used in machine learning that


works by creating many “particles” randomly distributed across parameter space that explore


the landscape collaboratively. We limit the number of particles to twice the number of pa

rameters. BO is a global optimization paradigm that can handle the expensive sampling of


many-parameter functions. It works by maintaining a surrogate model of the underlying cost


function and, at each iteration, updates the model to guide the search for the global minimum.


Essentially, the problem of optimizing the real cost is replaced with that of optimizing the sur

rogate model, which is designed to be a much easier optimization problem. We use OPTaaS, a


BO software package developed by Mind Foundry and adapted for this work.


6


### **Results**

Results from PSO optimization are shown in Fig. 3. We first simulate the training procedure


using a classical simulator in place of the quantum processor (orange plots in Fig. 3). Since


the PSO method is sensitive to the initial ”seed” values of the particles, we simulate the conver

gence for many different random seeds (see Fig.3). We choose a seed that converges quickly


and reliably under simulated sampling error to start the training procedure on the trapped ion


quantum computer illustrated in Fig.1. We iterate the training until it converges (blue plots in


Fig.3). In practice, which seeds are successful is unknown, and different seeds need to be tried


experimentally until a good model is obtained. This incurs an additional cost in the form of


multiple independent DDQCL training rounds.


For all-to-all connectivity, we find that a circuit with one rotation gate layer and one entan

gling gate layer is able to produce the desired BAS distribution (Fig. 3a). This is not the case


for the star-connected circuit, with the closest state having two additional components in the


superposition (states 6 and 9 in Fig. 3b). With two additional layers, the star-connected circuit


is able to model the BAS distribution (orange plots of Fig. 3c). In the experiment however (blue


plots in Fig. 3c), the PSO is unable to converge to an acceptable solution even using the best


pre-screened seed value and sufficient sample statistics. We conclude that PSO fails because the


throughput rate is too low for effectively training the circuit in the face of gate imperfections.


For these reasons, we instead employ a Bayesian optimization scheme for the circuit training


procedure. We find that all circuits experimentally converge in agreement with the simulations,


as shown in Fig. 4. Moreover, even the star-connected circuit with four layers now produces


a recognizable BAS distribution (Fig. 4c). In contrast to PSO, BO dramatically reduces the


number of samples needed for training and does not require any pre-selection of random seeds


or other prior knowledge of the cost-function landscape.


7


BO updates the surrogate model using the experimental result of every iteration. Therefore,


the classical part of each BO iteration consumes more time than with PSO, where the time


cost on the classical optimizer is negligible. However, the BO procedure converges faster to


the desired BAS distribution. More generally, these examples highlight the need to balance


quantum and classical resources in order to produce acceptable performance and run time in a


hybrid quantum algorithm.


As a measure of the performance of the various training procedures, we compute the Kullback

Leibler (KL) divergence ( _26_ ) and the qBAS score (an alternative performance measure sug

gested in ( _16_ )) of the experimental results at the end of each DDQCL training run, shown in


Table 1. We also compute the entanglement entropy (S) averaged over all two plus two qubit


partitions assuming a pure state ( _27_ ), estimated via simulation of the quantum state from the


trained circuits. The entanglement entropy quantifies the level of entanglement of a state, thus


indicates how difficult it is to produce such state. This metric shows that the successfully trained


circuits generate states that are consistent with a high level of entanglement. As a reference, the


entanglement entropy of a GHZ state over any partition is _S_ = 1.


8


#### (a) (b) (c)

Figure 3: Quantum circuit training results with Particle Swarm optimization (PSO), with simulations (orange) and trapped ion quantum computer results (blue). Column (a) corresponds to
a circuit with one layer of single qubit rotations (square boxes) and one layer of entanglement
gates (rounded boxes) of all-to-all connectivity. The circuit converges well to produce the barsand-stripes (BAS) distribution. Columns (b) and (c) correspond to a circuit with two and four
layers and star-connectivity, respectively. In (b), the simulation shows imperfect convergence
with two extra state components (6 and 9), due to the limited connectivity, and the experimental
results follow the simulation. In (c), the simulation shows convergence to the BAS distribution,
but the experiment fails to converge despite performing 1,400 quantum circuits. The optimization is sensitive to the choice of initialization seeds. To illustrate the convergence behavior, the
shaded regions span the 5th-95th percentile range of random seeds (500 for (a) and (b), 1000 for
(c), and the orange curve shows the median. The two-layer circuits have 14 and 11 parameters
for (a) all-to-all- and (b) star-connectivity, while the (c) star-connectivity circuit with four layers
has 26 parameters. The number of PSO particles used is twice the number of parameters, and
each training sample is repeated 5000 times. Including circuit compilation, controller-upload
time, and classical PSO optimization, each circuit instance takes about 1 min to be processed,
in addition to periodic interruptions for the re-calibration of gates.


9


#### (a) (b) (c)

Figure 4: Quantum circuit training results with Bayesian optimization (BO), with simulations
(orange) and trapped ion quantum computer results (blue). Column (a) corresponds to a circuit
with two layers of gates and all-to-all connectivity. Columns (b) and (c) correspond to a circuit
with two and four layers and star-connectivity, respectively. Convergence is much faster than
with PSO (Fig. 3). Unlike the PSO results, the four-layer star-connected circuit in (c) is trained
successfully, and no prior knowledge enters BO process. As before, the two-layer circuits have
14 and 11 parameters for (a) all-to-all- and (b) star-connectivity, while the (c) star-connectivity
circuit with four layers has 26 parameters. We use a batch of 5 circuits per iteration, and each
training sample is repeated 5000 times. Including circuit compilation, controller-upload time,
and BO classical optimization, each circuit instance takes 2-5 minutes, depending on the amount
of accumulated data.


10


|circuits|optimizer<br>PSO|D<br>KL<br>0.116|qBAS score<br>0.91|S<br>1.628|
|---|---|---|---|---|
||BO<br>PSO|0.094<br>0.357|0.91<br>0.74|1.659<br>0.9950|
||BO<br>PSO|0.328<br>0.646|0.77<br>0.59|0.9999<br>0.8867|
||BO|0.100|0.91|1.709|


Table 1: KL divergence ( _D_ _KL_, see Materials and Methods), qBAS score, and entanglement
entropy ( _S_ ) for the state obtained at the end of each of the DDQCL training on hardware, for
various circuits and classical optimizers used.

### **Discussion**


This demonstration of generative modeling using reconfigurable quantum circuits of up to 26


parameters represents one of the most powerful hybrid quantum applications to date. With


ongoing engineering improvements ( _28_ ), we expect the system to grow in both qubit number


and gate quality. This approach can be scaled up to handle larger data sets with increased qubit


number by adapting the cost function for sparser sampling ( _16_ ). Moreover, this procedure can


be adapted for other types of hybrid quantum algorithms.


Classical optimization techniques for hybrid quantum algorithms on intermediate-scale quan

tum computer do not always succeed ( _29_ ). Recent work suggests that typical cost functions for


medium to large scale variational quantum circuits landscape resemble “barren plateaus” ( _30_ ),


making optimization hard. As quantum computers scale up for larger problems, the cost of


classical optimization such as BO must be weighed against the quantum algorithmic advantage.


11


### **Materials and Methods**

#### **Trapped Ion Quantum Computer**

The trapped ion quantum computer used for this study consists of a chain of seven single [171] Yb [+]


ions confined in a Paul trap and laser cooled close to their motional ground state. Each ion


provides one physical qubit in the form of a pair of states in the hyperfine-split [2] _S_ 1 _/_ 2 ground


level with an energy difference of 12.642821 GHz, which is insensitive to magnetic fields to first


order. The qubits are collectively initialized into _|_ 0 _⟩_ through optical pumping, and state readout


is accomplished by state-dependent fluorescence detection ( _31_ ). Qubit operations are realized


via pairs of Raman beams, derived from a single 355-nm mode-locked laser ( _15_ ). These optical


controllers consist of an array of individual addressing beams and a counter-propagating global


beam that illuminates the entire chain. Single qubit gates are realized by driving resonant Rabi


rotations of defined phase, amplitude, and duration. Single-qubit rotations about the z-axis, are


performed by classically advancing/regarding the phase of the optical beatnote applied to the


particular qubit. Two-qubit gates are achieved by illuminating two selected ions with beat-note


frequencies near motional sidebands and creating an effective Ising spin-spin interaction via


transient entanglement between the two qubits and the motion in the trap ( _32–34_ ). Since our


particular scheme involves multiple modes of motion, we use an amplitude modulation scheme


to disentangle the qubit state from the motional state at the end of the interaction ( _35_ ). Typical


single-qubit gate fidelities are 99 _._ 5(2)%. Typical two-qubit gate fidelities are 98 _−_ 99%, with


fidelity mainly limited by residual entanglement of the qubit states to the motional state of the


ions, coherent crosstalk and driving intensity noise from classical imperfections in our optical


controllers.


In our experiment, the effect of the gate errors is seen as an offset in the cost function af

ter convergence. An improvement in gate fidelity will reduce this offset. But the convergence


12


behavior of an ideal system (as shown in the simulations in Fig.3 and Fig.4) is not significantly


faster than the actual experimental system. This is because it is limited by the classical opti

mization routine.


The trapped ion quantum architecture is scalable to a much larger number of qubits, as


atomic clock qubits are perfectly replicable and do not suffer idle errors (T1 and T2 times are


essentially infinite). All of the errors in scaling arise from the classical controllers, such as


applied noise on the trap electrodes and laser beam intensity fluctuations. Fundamental errors


(like spontaneous scattering from the control laser beams) are not expected to play a role until


our gates approach 99 _._ 99% fidelity. However, as the qubit number grows beyond about 20

30, we expect to sacrifice full connectivity, as gates will only be performed with high fidelity


between any qubit and its 15-20 nearest neighbors.


Another limitation is the sampling rate on the quantum computer. This is limited by techni

cal issues on the current experiment, and can be improved, e.g. by increasing the upload speed


of the experimental control system.

#### **Classical Optimizers: PSO and BO**


We explore two different classical optimizer in this study: Particle Swarm Optimization(PSO)


and Bayesian Optimization(BO).


PSO is a gradient-free optimization method inspired by the social behaviour of some an

imals. Each particle represents a candidate solution and moves within the solution space ac

cording to its current performance and the performance of the swarm. Three hyper-parameters


control the dynamics of the swarm: a cognition coefficient _c_ 1, a social coefficient _c_ 2, and an


inertia coefficient _w_ ( _24_ ).


Concretely, each particle consists of a position vector _θ_ _i_ and a velocity vector _v_ _i_ . At iteration


_t_ of the algorithm, the velocity of particle _i_ for the coordinate _d_ is updated as


13


_v_ _i,d_ [(] _[t]_ [+1)] = _wv_ _i,d_ [(] _[t]_ [)] [+] _[ c]_ [1] _[r]_ 1 [(] _[t]_ _,d_ [)] [(] _[p]_ [(] _i,d_ _[t]_ [)] _[−]_ _[θ]_ _i,d_ [(] _[t]_ [)] [) +] _[ c]_ [2] _[r]_ 2 [(] _[t]_ _,d_ [)] [(] _[g]_ _d_ [(] _[t]_ [)] _[−]_ _[θ]_ _i,d_ [(] _[t]_ [)] [)] _[,]_ (1)


where _r_ [(] _[t]_ [)]
1 _,d_ [and] _[ r]_ 2 [(] _[t]_ _,d_ [)] [are random numbers sampled from the uniform distribution in [0,1] for]

every dimension and every iteration, _p_ [(] _i_ _[t]_ [)] is the particle’s best position, _g_ [(] _[t]_ [)] is the swarm’s best


position. The position is then updated as


_θ_ _i_ [(] _[t]_ [+1)] = _θ_ _i_ [(] _[t]_ [)] + _v_ _i_ [(] _[t]_ [)] _[,]_ (2)


In our problem, each particle corresponds to a point in parameter space of the quantum


circuit. For example, in the fully connected circuit with two layers, each particle consists of


an instance of the 14 parameters. Recall, however, that parameters are angles and are therefore


periodic; We customized the PSO updates above to use this information. In Eq. (1), _p_ [(] _i,d_ _[t]_ [)] [and]


_θ_ [(] _[t]_ [)]
_i,d_ [can be thought of as two points on a circle. Instead of using the standard displacement]

_p_ [(] _i,d_ _[t]_ [)] _[−]_ _[θ]_ _i,d_ [(] _[t]_ [)] [, we use the angular displacement, that is the signed length of the minor arc on the unit]

circle. We use the same definition of displacement for the swarm’s best position _g_ _i,d_ [(] _[t]_ [)] [. Finally, in]


Eq. (2), we make sure to express angles always using their principal values.


In our experiments, we set the number of particles to twice the number of parameters of the


circuit. Position and velocity vectors of each particle are initialized from the uniform distribu

tion. For the coefficients we use _c_ 1 = _c_ 2 = 1 and _w_ = 0 _._ 5.


Bayesian Optimisation is a powerful global optimisation paradigm. It is best suited to find

ing optima of multi-modal objective functions that are expensive to evaluate. There are two


main features that characterize the a BO process: the surrogate model and an acquisition func

tion.


The surrogate model is non-parametric model of the objective function. At each iteration,


the surrogate model is updated using the sampled points in parameter space. The package used


in this study is OPTaaS by MindFoundry. It implements the surrogate model as regression using


14


Gaussian Process ( _36_ ). A kernel (or correlation function) characterizes the Gaussian process,


we use a Matern 5/2 as it provides the most flexibility.


The acquisition function is computed from the surrogate model. It is used to select points


for evaluation during the optimization. It trades off exploration against exploitation. The ac

quisition function of a point has a high value if the cost function is expected to give a signif

icant improvement over historically sampled points, or if the uncertainty of the point is high,


according to the surrogate model. A simple and well known acquisition function, Expected


Improvement ( _37_ ), is employed here.


In our case, OPTaaS also leverages the cyclic symmetry of the angles by embedding the


parameter space into a metric space with the appropriate topology, effectively allowing the


Gaussian Process surrogate model to be placed over a hyper-torus, rather than a hyper-cube.


This greatly alleviates the so-called curse of dimensionality ( _38_ ), and allows for much more


efficient use of samples of the objective function.


It is key in Bayesian Optimisation to adequately optimise the acquisition function during


each iteration. OPTaaS puts considerable computational resources towards this non-convex


optimisation problem.


There are two major reasons why the BO out performs PSO in our specific case. First,


PSO spends significant amount of computation resource exploring trajectories far from opti

mal, while BO mitigates it by the use of acquisition function. Second, the maintenance of


the surrogate model enable us to make much better use of the information from the historical


exploration of the parameter space.

#### **Cost Functions**


We use a cost function to quantify the difference between the target BAS distribution and the


experimental measurements of the circuit. The cost functions used to implement the training


15


are variants of the original Kullback-Leibler Divergence ( _D_ _KL_ ) ( _26_ ):



_p_ ( _i_ ) log _[q]_ [(] _[i]_ [)]
_i_ _p_ ( _i_ )



_D_ _KL_ ( _p, q_ ) = _−_ �



(3)
_p_ ( _i_ ) _[.]_



Here _p_ and _q_ are two distributions.


_D_ _KL_ ( _p, q_ ) is an information theoretic measure of how two probability distribution differ.


If base 2 for the logarithm is used, it quantifies the expected number of extra bits required to


store samples from p when an optimal code designed for q is used instead. It can be shown


that _D_ _KL_ ( _p, q_ ) is non-negative, and is zero if and only if p=q. However, it is asymmetric in the


arguments and does not satisfy the triangle inequality. Therefore _D_ _KL_ ( _p, q_ ) is not a metric.


The KL divergence is a very general measure, but it is not always well-defined, e.g. if an


element of the domain is supported by _p_ and not by _q_, the measure will diverge. This problem


may occur quite often if _D_ _KL_ ( _p, q_ ) is estimated from samples and if the dimensionality of the


domain is large. For PSO, we use the clipped negative log-likelihood cost function ( _16_ ),


_C_ _nll_ = _−_ � _p_ ( _i_ ) log _{_ max[ _ϵ, q_ ( _i_ )] _}._ (4)


_i_


Here we set _p_ as the target distribution. Thus Eq.4 is equivalent to Eq.3 up to a constant offset,


so the optimization of these two functions is equivalent. _ϵ_ is a small number (0.0001 here) used


to avoid a numerical singularity when _q_ ( _i_ ) is measured to be zero.


For BO, we use the clipped symmetrized Kullback-Leibler (KL) divergence as the cost


function


_D_ ˜ _KL_ ( _p, q_ ) = _D_ _KL_ [max( _ϵ, p_ ) _,_ max( _ϵ, q_ )] + _D_ _KL_ [max( _ϵ, q_ ) _,_ max( _ϵ, p_ )] _._ (5)


This is found to be the most reliable variant of _D_ _KL_ for BO.


16


### **Acknowledgments**

We thank C. Figgatt for helpful discussion. This work was supported by the ARO with funds


from the Intelligence Advanced Research Projects Activity (IARPA) LogiQ program (Grant


Number W911NF16-1-0082), the Army Research Office (ARO) MURI program on Modular


Quantum Circuits (Grant Number W911NF1610349), the AFOSR MURI program on Opti

mal Quantum Measurements (Grant Number 5710003628), the NSF STAQ Practical Fully

Connected Quantum Computer Project, and the NSF Physics Frontier Center at JQI (Grant


Number PHY0822671). L. Egan is additionally funded by NSF award DMR-1747426.

#### **Authors’ contributions**


D. Z, N. M. L, M. B, K. A. L, A. P and C. M designed the research. D. Z, N. M. L, M. B, K.


A. L, N. H. N, C. H. A, A. P, L. E, and O. P collected and analyzed data. D. Z, M. B, A. P, N.


K, A. G and C. B contributed to the software used in this study. All authors contributed to this


manuscript.

#### **Competing interests**


C.M. is a founding scientist of IonQ, Inc. All other authors declare that they have no competing


interests.

#### **Data availability**


All data needed to evaluate the conclusions in the paper are present in the paper and/or the


Supplementary Materials. Additional data related to this paper may be requested from the


corresponding author upon request.


17


### **References**

1. J. R. McClean, J. Romero, R. Babbush, A. Aspuru-Guzik, The theory of variational hybrid


quantum-classical algorithms. _New Journal of Physics_ **18**, 023023 (2016).


2. J. Preskill, Quantum Computing in the NISQ era and beyond. _Quantum_ **2**, 79 (2018).


3. A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, J. M. Gambetta,


Hardware-efficient variational quantum eigensolver for small molecules and quantum mag

nets. _Nature_ **549**, 242 (2017).


4. A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru

Guzik, J. L. O’Brien, A variational eigenvalue solver on a photonic quantum processor.


_Nature communications_ **5**, 4213 (2014).


5. C. Hempel, C. Maier, J. Romero, J. McClean, T. Monz, H. Shen, P. Jurcevic, B. P. Lanyon,


P. Love, R. Babbush, A. Aspuru-Guzik, R. Blatt, C. F. Roos, Quantum chemistry calcula

tions on a trapped-ion quantum simulator. _Phys. Rev. X_ **8**, 031022 (2018).


6. P. OMalley, R. Babbush, I. Kivlichan, J. Romero, J. McClean, R. Barends, J. Kelly,


P. Roushan, A. Tranter, N. Ding, _et al._, Scalable quantum simulation of molecular ener

gies. _Physical Review X_ **6**, 031007 (2016).


7. C. Kokail, C. Maier, R. van Bijnen, T. Brydges, M. Joshi, P. Jurcevic, C. Muschik, P. Silvi,


R. Blatt, C. Roos, _et al._, Self-verifying variational quantum simulation of lattice models.


_Nature_ **569**, 355 (2019).


8. E. Farhi, J. Goldstone, S. Gutmann, A quantum approximate optimization algorithm. _MIT-_


_CTP/4610_ (2014).


18


9. J. Otterbach, R. Manenti, N. Alidoust, A. Bestwick, M. Block, B. Bloom, S. Caldwell,


N. Didier, E. S. Fried, S. Hong, _et al._, Unsupervised machine learning on a hybrid quantum


computer. _[arXiv preprint arXiv:1712.05771](http://arxiv.org/abs/1712.05771)_ (2017).


10. J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, _Proceedings of the IEEE international conference_


_on computer vision_ (2017), pp. 2223–2232.


11. A. Van Den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalch

brenner, A. Senior, K. Kavukcuoglu, Wavenet: A generative model for raw audio. _CoRR_


_abs/1609.03499_ (2016).


12. S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Jozefowicz, S. Bengio, Generating sen

tences from a continuous space. _SIGNLL Conference on Computational Natural Language_


_Learning (CONLL), 2016_ (2016).


13. Y. Bengio, L. Yao, G. Alain, P. Vincent, _Advances in Neural Information Processing Sys-_


_tems_ (2013), pp. 899–907.


14. R. G´omez-Bombarelli, J. N. Wei, D. Duvenaud, J. M. Hern´andez-Lobato, B. S´anchez

Lengeling, D. Sheberla, J. Aguilera-Iparraguirre, T. D. Hirzel, R. P. Adams, A. Aspuru

Guzik, Automatic chemical design using a data-driven continuous representation of


molecules. _ACS central science_ **4**, 268–276 (2018).


15. S. Debnath, N. M. Linke, C. Figgatt, K. A. Landsman, K. Wright, C. Monroe, Demon

stration of a small programmable quantum computer with atomic qubits. _Nature_ **536**, 63


(2016).


16. M. Benedetti, D. Garcia-Pintos, O. Perdomo, V. Leyton-Ortega, Y. Nam, A. Perdomo-Ortiz,


A generative modeling approach for benchmarking and training shallow quantum circuits.


_npj Quantum Information_ **5**, 45 (2019).


19


17. Y. Du, M.-H. Hsieh, T. Liu, D. Tao, The expressive power of parameterized quantum cir

cuits. _[arXiv preprint arXiv:1810.11922](http://arxiv.org/abs/1810.11922)_ (2018).


18. X. Gao, Z.-Y. Zhang, L.-M. Duan, A quantum machine learning algorithm based on gener

ative models. _Science Advances_ **4** (2018).


19. D. J. MacKay, D. J. Mac Kay, _Information theory, inference and learning algorithms_ (Cam

bridge university press, 2003).


20. L. Hu, S.-H. Wu, W. Cai, Y. Ma, X. Mu, Y. Xu, H. Wang, Y. Song, D.-L. Deng, C.-L.


Zou, _et al._, Quantum generative adversarial learning in a superconducting quantum circuit.


_Science advances_ **5**, eaav2761 (2019).


21. K. A. Landsman, C. Figgatt, T. Schuster, N. M. Linke, B. Yoshida, N. Y. Yao, C. Monroe,


Verified quantum information scrambling. _Nature_ **567**, 61 (2019).


22. L. Theis, A. v. d. Oord, M. Bethge, A note on the evaluation of generative models. _arXiv_


_[preprint arXiv:1511.01844](http://arxiv.org/abs/1511.01844)_ (2015).


23. J.-G. Liu, L. Wang, Differentiable learning of quantum circuit born machines. _Physical_


_Review A_ **98**, 062324 (2018).


24. J. Kennedy, R. Eberhart, _Proc. IEEE International Conference on Neural Networks, Perth,_


_Australia_ (1995), pp. 1942–1948.


25. P. I. Frazier, A tutorial on bayesian optimization. _[arXiv preprint arXiv:1807.02811](http://arxiv.org/abs/1807.02811)_ (2018).


26. S. Kullback, R. A. Leibler, On information and sufficiency. _The annals of mathematical_


_statistics_ **22**, 79–86 (1951).


20


27. A. Higuchi, A. Sudbery, How entangled can two couples get? _Physics Letters A_ **273**,


213–217 (2000).


28. K. Wright, K. Beck, S. Debnath, J. Amini, Y. Nam, N. Grzesiak, J.-S. Chen, N. Pisenti,


M. Chmielewski, C. Collins, _et al._, Benchmarking an 11-qubit quantum computer. _arXiv_


_[preprint arXiv:1903.08181](http://arxiv.org/abs/1903.08181)_ (2019).


29. K. E. Hamilton, E. F. Dumitrescu, R. C. Pooser, Generative model benchmarks for super

conducting qubits. _[arXiv preprint arXiv:1811.09905](http://arxiv.org/abs/1811.09905)_ (2018).


30. J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven, Barren plateaus in


quantum neural network training landscapes. _Nature communications_ **9**, 4812 (2018).


31. S. Olmschenk, K. C. Younge, D. L. Moehring, D. N. Matsukevich, P. Maunz, C. Monroe,


Manipulation and detection of a trapped yb [+] hyperfine qubit. _Phys. Rev. A_ **76**, 052314


(2007).


32. K. Mølmer, A. Sørensen, Multiparticle entanglement of hot trapped ions. _Phys. Rev. Lett._


**82**, 1835–1838 (1999).


33. E. Solano, R. L. de Matos Filho, N. Zagury, Deterministic bell states and measurement of


the motional state of two trapped ions. _Phys. Rev. A_ **59**, R2539–R2543 (1999).


34. G. Milburn, S. Schneider, D. James, Ion trap quantum computing with warm ions.


_Fortschritte der Physik_ **48**, 801–810 (2000).


35. T. Choi, S. Debnath, T. A. Manning, C. Figgatt, Z.-X. Gong, L.-M. Duan, C. Monroe,


Optimal quantum control of multimode couplings between trapped ion qubits for scalable


entanglement. _Phys. Rev. Lett._ **112**, 190502 (2014).


36. C. E. Rasmussen, _Summer School on Machine Learning_ (Springer, 2003), pp. 63–71.


21


37. E. Brochu, V. M. Cora, N. De Freitas, A tutorial on bayesian optimization of expensive cost


functions, with application to active user modeling and hierarchical reinforcement learning.


_[arXiv preprint arXiv:1012.2599](http://arxiv.org/abs/1012.2599)_ (2010).


38. R. E. Bellman, _Adaptive control processes: a guided tour_, vol. 2045 (Princeton university


press, 2015).


22


