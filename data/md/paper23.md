**A rigorous and robust quantum speed-up in supervised machine learning**


Yunchao Liu, [1, 2,] _[ ∗]_ Srinivasan Arunachalam, [2,] _[ †]_ and Kristan Temme [2,] _[ ‡]_

1 _Department of Electrical Engineering and Computer Sciences,_
_University of California, Berkeley, CA 94720_
2 _IBM Quantum, T.J. Watson Research Center, Yorktown Heights, NY 10598_
(Dated: December 1, 2020)


Over the past few years several quantum machine learning algorithms were proposed that
promise quantum speed-ups over their classical counterparts. Most of these learning algorithms either assume quantum access to data – making it unclear if quantum speed-ups still
exist without making these strong assumptions, or are heuristic in nature with no provable
advantage over classical algorithms. In this paper, we establish a rigorous quantum speed-up
for supervised classification using a general-purpose quantum learning algorithm that only
requires classical access to data. Our quantum classifier is a conventional support vector
machine that uses a fault-tolerant quantum computer to estimate a kernel function. Data
samples are mapped to a quantum feature space and the kernel entries can be estimated
as the transition amplitude of a quantum circuit. We construct a family of datasets and
show that no classical learner can classify the data inverse-polynomially better than random
guessing, assuming the widely-believed hardness of the discrete logarithm problem. Meanwhile, the quantum classifier achieves high accuracy and is robust against additive errors in
the kernel entries that arise from finite sampling statistics.


Finding potential applications for quantum computing which demonstrate quantum speed-ups
is a central goal of the field. Much attention has been drawn towards establishing a quantum
advantage in machine learning due to its wide applicability [1–5]. In this direction there have been
several quantum algorithms for machine learning tasks that promise polynomial and exponential
speed-ups. A family of such quantum algorithms assumes that classical data is encoded in amplitudes of a quantum state, which uses a number of qubits that is only logarithmic in the size of
the dataset. These quantum algorithms are therefore able to achieve exponential speed-ups over

–
classical approaches [6 16]. However, it is not known whether data can be efficiently provided this
way in practically relevant settings. This raises the question of whether the advantage comes from
the quantum algorithm, or from the way data is provided [17]. Indeed, recent works have shown
that if classical algorithms have an analogous sampling access to data, then some of the proposed

–
exponential speed-ups do no longer exist [18 23].
Consequently a different class of quantum algorithms has been developed which only assumes
classical access to data. Most of these algorithms use variational circuits for learning, where a
candidate circuit is selected from a parameterized circuit family via classical optimization [24–
28]. Although friendly to experimental implementation, these algorithms are heuristic in nature
since no formal evidence has been provided which shows that they have a genuine advantage over
classical algorithms. An important challenge is therefore to find one example of such a heuristic
quantum machine learning algorithm, which given _classical access_ to data can _provably_ outperform
all classical learners for some learning problem.
In this paper, we answer this in the affirmative. We show that an exponential quantum speedup can be obtained via the use of a quantum-enhanced feature space [29, 30], where each data
point is mapped non-linearly to a quantum state and then classified by a linear classifier in the
high dimensional Hilbert space. To efficiently learn a linear classifier in feature space from training
data, we use the standard kernel method in _support vector machines_ (SVMs), a well-known family


_∗_ [yunchaoliu@berkeley.edu](mailto:yunchaoliu@berkeley.edu)

_†_ [Srinivasan.Arunachalam@ibm.com](mailto:Srinivasan.Arunachalam@ibm.com)

_‡_ [kptemme@ibm.com](mailto:kptemme@ibm.com)


2


of supervised classification algorithms [31, 32]. We obtain the kernel matrix by measuring the
pairwise inner product of the feature states on a quantum computer, a procedure we refer to as
_quantum kernel estimation_ (QKE). This kernel matrix is then given to a classical optimizer that
efficiently finds the linear classifier that optimally separates the training data in feature space by
running a convex quadratic program.
The advantage of our quantum learner stems from its ability to recognize classically intractable
complex patterns using the feature map. We prove an end-to-end quantum advantage based on this
intuition, where our quantum classifier is guaranteed to achieve high accuracy for a classically hard
classification problem. We show that under a suitable quantum feature map, the classical data
points, which are indistinguishable from having random labels by efficient classical algorithms, are
linearly separable with a large margin in high-dimensional Hilbert space. Based on this property,

–
we then combine ideas from classic results on the generalization of soft margin classifiers [33 36] to
rigorously bound the misclassification error of the SVM-QKE algorithm. The optimization for large
margin classifiers in the SVM program is crucial in our proof, as it allows us to learn the optimal
separator in the exponentially large feature space, while also making our quantum classifier robust
against additive sampling errors.
Our classification problem that shows the exponential quantum speed-up is constructed based on
the discrete logarithm problem (DLP). We prove that no efficient classical algorithm can achieve an
accuracy that is inverse-polynomially better than random guessing, assuming the widely-believed
classical hardness of DLP. In computational learning theory, the use of one-way functions for
constructing classically hard learning problems is a well-known technique [37]. Rigorous separations
between quantum and classical learnability have been established using this idea in the quantum
oracular and PAC model [2, 38], as well as in the classical generative setting [39]. There the
quantum algorithms are constructed specifically to solve the problems for showing a separation,
and in general are not applicable to other learning problems. Based on different complexitytheoretic assumptions, evidences of an exponential quantum speed-up were shown for a quantum
generative model [40], where the overall performance is not guaranteed.


**A classically intractable learning problem**


The task of supervised classification is to assign a label _y ∈{−_ 1 _,_ 1 _}_ to a datum _x ∈X_ from data
space _X_ according to some unknown decision rule _f_ (usually referred to as a _concept_ ), by learning
from labeled examples _S_ = _{_ ( _x_ _i_ _, y_ _i_ ) _}_ _i_ =1 _,...,m_ that are generated from this concept, _y_ _i_ = _f_ ( _x_ _i_ ). Given
the training set _S_, an efficient learner needs to compute a classifier _f_ _[∗]_ in time that is polynomial
in the size of _S_, with the goal of achieving high _test accuracy_,


acc _f_ ( _f_ _[∗]_ ) = Pr (1)
_x∼X_ [[] _[f]_ _[∗]_ [(] _[x]_ [) =] _[ f]_ [(] _[x]_ [)]] _[,]_


the probability of agreeing with _f_ on unseen examples. Here we assume that the datum _x_ is
sampled uniformly random from _X_, in both training and testing, and the size of _S_ is polynomial
in the data dimension.

An important ingredient of machine learning is prior knowledge, i.e., additional information
given to the learning algorithm besides the training set. In standard computational learning theory [37, 41], this is modeled as a concept class – a (often exponentially large) set of labeling rules,
and the target concept is promised to be chosen from the concept class. A concept class _C_ is
efficiently learnable if for every _f ∈C_, an efficient algorithm can achieve 0.99 test accuracy by
learning from examples labeled according to _f_ with high success probability. See Appendix A for
detailed definitions.
Our concept class that separates quantum and classical learnability is based on the discrete
logarithm problem (DLP). For a large prime number _p_ and a generator _g_ of Z _[∗]_ _p_ [=] _[ {]_ [1] _[,]_ [ 2] _[, . . ., p][ −]_ [1] _[}]_ [,]


3


a b c


FIG. 1. Learning the concept class _C_ by a quantum feature map. ( **a** ) After taking the discrete log of the data
samples, they become separated in log space by the concept _s_ . ( **b** ) However, in the original data space, the
data samples look like randomly labeled and cannot be learned by an efficient classical algorithm. ( **c** ) Using
the quantum feature map, each _x ∈_ Z _[∗]_ _p_ [is mapped to a quantum state] _[ |][φ]_ [(] _[x]_ [)] _[⟩]_ [, which corresponds to a uniform]
superposition of an interval in log space starting with log _g_ _x_ . This feature map creates a large margin, as
the +1 labeled example (red interval) has high overlap with a separating hyperplane (green interval), while
the _−_ 1 labeled example (blue interval) has zero overlap.


it is a widely-believed conjecture that no classical algorithm can compute log _g_ _x_ on input _x ∈_ Z _[∗]_ _p_ [,]
in time polynomial in _n_ = _⌈_ log 2 ( _p_ ) _⌉_, the number of bits needed to represent _p_ . Meanwhile, DLP
can be solved by Shor’s quantum algorithm [42] in polynomial time.
Based on DLP, we define our concept class _C_ = _{f_ _s_ _}_ _s∈_ Z _∗p_ over the data space _X_ = Z _[∗]_ _p_ [as follows,]



_f_ _s_ ( _x_ ) =



+1 _,_ if log _g_ _x ∈_ [ _s, s_ + _[p][−]_ 2 [3] []] _[,]_ (2)
� _−_ 1 _,_ else.



Each concept _f_ _s_ : Z _[∗]_ _p_ _[→{−]_ [1] _[,]_ [ 1] _[}]_ [ maps half the elements in][ Z] _[∗]_ _p_ [to +1 and half of them to] _[ −]_ [1.]
To see why the discrete logarithm is important in our definition, note that if we change log _g_ _x_
to _x_ in Eq. (2), then learning the concept class _C_ is a trivial problem. Indeed, if we imagine the
elements of Z _[∗]_ _p_ [as lying on a circle, then each concept] _[ f]_ _[s]_ [corresponds to a direction for cutting the]
circle in two halves (Fig. 1a). Therefore, the training set of labeled examples can be separated as
two distinct clusters, where one cluster is labeled +1 and the other labeled _−_ 1. To classify a new
example, a learning algorithm can simply decide based on which cluster is closer to the example.
This intuition also explains why the original concept class _C_ is learnable by a quantum learner,
since the learner can use Shor’s algorithm to compute the discrete log for every data sample it
receives [38], and then solve the above trivial learning problem.
On the other hand, due to the classical intractability of DLP, the training samples look like
randomly labeled from the viewpoint of a classical learner (Fig. 1b). In fact, we can prove that the
best a classical learner can do is randomly guess the label for new examples, which achieves 50%
test accuracy. These results are summarized below.


**Theorem 1.** _Assuming the classical hardness of_ DLP _, no efficient classical algorithm can achieve_
1 1
2 [+] poly( _n_ ) _[test accuracy for][ C][.]_


Our proof, c.f. Appendix B, of classical hardness of learning _C_ is based on an average-case
hardness result for discrete log by Blum and Micali [43]. They showed that computing the most


4



significant bit of log _g_ _x_ for [1] 2 [+] poly1( _n_ ) [fraction of] _[ x][ ∈]_ [Z] _[∗]_ _p_ [is as hard as solving][ DLP][. We then reduce]

our concept class learning problem to DLP using this result, by showing that if an efficient learner
can achieve [1] [+] 1 _[ C]_



significant bit of log _g_ _x_ for [1] 2



can achieve [1] 1

2 [+] poly( _n_ ) [test accuracy for] _[ C]_ [, then it can be used to construct an efficient classical]
algorithm for DLP, which proves Theorem 1.
In addition to establishing a separation between quantum and classical learnability for binary
classification, we also note that this separation can be efficiently verified by a classical verifier in an
interactive setting. This follows from a nice property of our concept class. We show that for every
concept _f ∈C_, we can _efficiently generate_ labeled examples ( _x, f_ ( _x_ )) classically where _x ∼_ Z _[∗]_ _p_ [is]
uniformly distributed, despite _f_ ( _x_ ) being hard to compute by definition. To test if a prover can
learn _C_, a classical verifier can pick a random concept and efficiently generate two sets of data
( _S, T_ ), where _S_ is a training set of labeled examples and _T ⊆_ Z _[∗]_ _p_ [is a test set of examples with]
labels removed. The verifier then sends ( _S, T_ ) to the prover and asks for labels for _T_, and finally
accepts or rejects based on the accuracy of these labels. As a corollary of Theorem 1, an efficient
quantum learner can pass this challenge, while no efficient classical learner can pass it assuming
the classical hardness of DLP.



**Efficient learnability with** QKE


We now turn our attention to general-purpose quantum learning algorithms which only requires
classical access to data and in principle can be applied to a wide range of learning problems.
Examples include quantum neural networks [24–28], generative models [40], and kernel methods [29,
30]. The main challenge of proving a quantum advantage for these algorithms is that they may
not be able to utilize the full power of quantum computers, and it is unclear if the set of problems
that they can solve is beyond the capability of classical algorithms. Indeed, previous analysis of
these quantum algorithms only establish evidence that parts of the algorithm cannot be efficiently
simulated classically [29, 30, 40], which does not guarantee that the algorithms can solve classically
hard learning problems.
Recall that we have constructed a learning problem that is as hard as the discrete log problem.
This implies classical intractability assuming the hardness of discrete log, while also assuring that
the problem is within the power of quantum computers due to Shor’s algorithm. This provides
the basis to solving our main challenge – we now show that the concept class _C_ can be efficiently
learned by our support vector machine algorithm with quantum kernel estimation (SVM-QKE).
This formally establishes our intuition that quantum feature maps can recognize patterns that are
unrecognizable by classical algorithms, even when the quantum classifier is inherently noisy due
to finite sampling statistics. We have therefore established an end-to-end quantum advantage for
quantum kernel methods.
The core component in our algorithm that leads to its ability to outperform classical learners
is the _quantum feature map_ . For learning the concept class _C_, the feature map is constructed prior
to seeing the training samples and has the following form,



1
_x �→|φ_ ( _x_ ) _⟩_ =
~~_√_~~ 2 _[k]_



2 _[k]_ _−_ 1
�


_i_ =0



_i_
�� _x · g_ � _,_ (3)



which maps a classical data point _x ∈_ Z _[∗]_ _p_ [to a] _[ n]_ [-qubit quantum state] _[ |][φ]_ [(] _[x]_ [)] _[⟩⟨][φ]_ [(] _[x]_ [)] _[|]_ [, and] _[ k]_ [ =] _[ n][−][t]_ [ log] _[ n]_
for some constant _t_ . This family of states was first introduced in [44] to study the complexity of
quantum state generation and statistical zero knowledge, where it is shown that _|φ_ ( _x_ ) _⟩_ = _U_ ( _x_ ) _|_ 0 _[n]_ _⟩_
can be efficiently prepared on a fault tolerant quantum computer by a circuit _U_ ( _x_ ) which uses
Shor’s algorithm as a subroutine, c.f. Appendix D 1.
In learning algorithms, feature maps play the role of pattern recognition: the intrinsic labeling
patterns for data, which are hard to recognize in the original space (Fig. 1b), become easy to


5


FIG. 2. Quantum kernel estimation. A quantum feature map _x �→|φ_ ( _x_ ) _⟩_ is represented by a circuit,
_|φ_ ( _x_ ) _⟩_ = _U_ ( _x_ ) _|_ 0 _[n]_ _⟩_ . Each kernel entry _K_ ( _x_ _i_ _, x_ _j_ ) is obtained using a quantum computer by running the
circuit _U_ _[†]_ ( _x_ _j_ ) _U_ ( _x_ _i_ ) on input _|_ 0 _[n]_ _⟩_, and then estimate �� _⟨_ 0 _n_ _|U_ _†_ ( _x_ _j_ ) _U_ ( _x_ _i_ ) _|_ 0 _n_ _⟩_ �� 2 by counting the frequency of
the 0 _[n]_ output.


identify once mapped to the feature space. Our feature map indeed achieves this by mapping low
dimensional data vectors to the Hilbert space with exponentially large dimension. For each concept
_f_ _s_ _∈C_, we show that there exists a separating hyperplane _w_ _s_ = _|φ_ _s_ _⟩⟨φ_ _s_ _|_ in feature space. That
is, for +1 labeled examples, we have Tr[ _w_ _s_ _|φ_ ( _x_ ) _⟩⟨φ_ ( _x_ ) _|_ ] = _| ⟨φ_ _s_ _|φ_ ( _x_ ) _⟩|_ [2] = 1 _/_ poly( _n_ ), while for _−_ 1
labeled examples we have _| ⟨φ_ _s_ _|φ_ ( _x_ ) _⟩|_ [2] = 0 with probability 1 _−_ 1 _/_ poly( _n_ ) (Fig. 1c). If we think of _w_ _s_
as the normal vector of a hyperplane in feature space associated with the Hilbert-Schmidt inner
product, then this property suggests that +1/-1 labeled examples are separated by this hyperplane
by a large margin. This _large margin property_ is the fundamental reason that our algorithm can
succeed: it suggests that to correctly classify the data samples, it suffices to find a good linear
classifier in feature space, while also guaranteeing that such a good linear classifier exists.
The idea of applying a high-dimensional feature map to reduce a complex pattern recognition
problem to linear classification is not new, and has been the foundation of a family of supervised
learning algorithms called support vector machines (SVMs) [31, 32]. Consider a general feature
map _φ_ : _X →H_ that maps data to a feature space _H_ associated with an inner product _⟨·, ·⟩_ . To
find a linear classifier in _H_, we consider the convex quadratic program



1
min 2 [+] _[ λ]_
_w,ξ_ 2 _[∥][w][∥]_ [2] 2



_m_
� _ξ_ _i_ [2] s.t. _y_ _i_ _· ⟨φ_ ( _x_ _i_ ) _, w⟩≥_ 1 _−_ _ξ_ _i_ (4)


_i_ =1



where _ξ_ _i_ _≥_ 0. Here _λ >_ 0 is a constant, _w_ is a hyperplane in _H_ which defines a linear classifier
_y_ = sign ( _⟨φ_ ( _x_ ) _, w⟩_ ), and _ξ_ _i_ are slack variables used in the soft margin constraints. Intuitively, this
program optimizes for the hyperplane that maximally separates +1/-1 labeled data. Note that (4)
is efficient in the dimension of _H_ . However, once we map to a high-dimensional feature space, it
takes exponential time to find the optimal hyperplane. The main insight which leads to the success
of SVMs is that this problem can be solved by running the dual program of Eq. (4)



2



2 _λ_



max
_α≥_ 0



_m_
�



_α_ _i_ _−_ [1]

� 2

_i_ =1



_m_
� _α_ _i_ [2] _[,]_ (5)


_i_ =1



_m_
�



� _α_ _i_ _α_ _j_ _y_ _i_ _y_ _j_ _K_ ( _x_ _i_ _, x_ _j_ ) _−_ 2 [1]

_i,j_ =1



where _K_ ( _x_ _i_ _, x_ _j_ ) = _⟨φ_ ( _x_ _i_ ) _, φ_ ( _x_ _j_ ) _⟩_ is the _kernel matrix_ . This dual program, which returns a linear
classifier defined as _y_ = sign ( [�] _[m]_ _i_ =1 _[α]_ _[i]_ _[y]_ _[i]_ _[K]_ [(] _[x, x]_ _[i]_ [)), is equivalent to the original program as guaran-]
teed by strong duality. Effectively, this means that we can do optimization in the high-dimensional
feature space efficiently, as long as the kernel _K_ ( _x_ _i_ _, x_ _j_ ) can be efficiently computed.


6


The same insight can be applied to our quantum feature map: to utilize the full power of the
quantum feature space, it suffices to compute the inner products _| ⟨φ_ ( _x_ _i_ ) _|φ_ ( _x_ _j_ ) _⟩|_ [2] between the
feature states. To estimate such an inner product using a quantum computer, we simply apply
_U_ _[†]_ ( _x_ _j_ ) _U_ ( _x_ _i_ ) on input _|_ 0 _[n]_ _⟩_, and measure the probability of the 0 _[n]_ output (see Fig. 2). We call such
a procedure quantum kernel estimation (QKE). The overall procedure for learning with quantum
feature map is now clear. On input a set of _m_ labeled training examples _S_, run QKE to obtain
the _m × m_ kernel matrix, then run the dual SVM Eq. (5) on a classical computer to obtain the
solution _α_ . To classify a new example _x_, run QKE to obtain _K_ ( _x, x_ _i_ ) for each _i_ = 1 _, . . ., m_,
then return



�



_._ (6)



_y_ = sign



_m_
�
� _i_ =1



� _α_ _i_ _y_ _i_ _K_ ( _x, x_ _i_ )


_i_ =1



Throughout the entire SVM-QKE algorithm, QKE is the only subroutine that requires a quantum
computer, while all other optimization steps can be performed classically. See Appendix C for a
detailed description of the algorithm.
Despite the seemingly direct analogy between quantum and classical feature maps, one important aspect of QKE makes the analysis of our quantum algorithm fundamentally different from
classical SVMs. Note that estimating the output probability of a quantum computer is inherently
noisy due to finite sampling statistics, even when the quantum computer is fully error corrected. In
QKE, this finite sampling error can be modeled as i.i.d. additive errors for each kernel entry, with
mean 0 and variance [1]

_R_ [, where] _[ R]_ [ is the number of measurement shots for each kernel estimation]
circuit. Our main result rigorously establishes the performance guarantee of SVM-QKE, which
remains robust under this noise model.


**Theorem 2.** _The concept class C is efficiently learnable by_ SVM _-_ QKE _. More specifically, for any_
_concept f ∈_ _C, the_ SVM _-_ QKE _algorithm returns a classifier with test accuracy at least 0.99 in_
_polynomial time, with probability at least_ 2 _/_ 3 _over the choice of random training samples and over_
_noise in_ QKE _estimation._


The main idea in our proof, c.f. Appendix D and E, is to connect the large margin property to

–
existing results on the generalization of soft margin classifiers [33 36, 45]. There, it is shown that
if the learning algorithm finds a hyperplane _w_ that has a large margin on the training set, then
the linear classifier _y_ = sign ( _⟨φ_ ( _x_ ) _, w⟩_ ) has high accuracy with high probability. To see how we
can apply these results, recall that in the large margin property, we have established that there
exists a hyperplane _w_ _[∗]_ with a large margin on the training set. Therefore, as the SVM program
optimizes for the hyperplane with largest margin, it is guaranteed to find a good hyperplane _w_,
although not necessarily the same as _w_ _[∗]_ . Applying the generalization bounds to this _w_ gives us
the desired result.

As discussed above, the missing piece in the proof sketch is to show that the performance
of SVM-QKE remains robust with additive noise in the kernel. In the following we prove noise
robustness by introducing two additional results. First, we show that the dual SVM program
(Eq. (5)) is robust, i.e., when the kernel used in (5) has a small additive perturbation, then the
solution returned by the program also has a small perturbation. This follows from strong convexity
of (5) and standard perturbation analysis of positive definite quadratic programs [46]. This result
implies that the hyperplane _w_ _[′]_ obtained by the noisy kernel is close to the noiseless solution _w_ with
high probability. Second, we show that when _w_ _[′]_ is close to _w_, the linear classifier obtained by _w_ _[′]_

has high accuracy. This seemingly simple statement is not trivial, as the sign function is sensitive
to noise. That is, if _⟨φ_ ( _x_ ) _, w⟩_ is very close to 0, then a small perturbation in _w_ could change its
sign. We provide a solution to this problem by proving a stronger generalization bound. We show


7


that if a hyperplane _w_ has a large margin on the training set, then not only does _⟨φ_ ( _x_ ) _, w⟩_ have
the correct sign, it is also bounded away from 0 with high probability. Therefore, when the noisy
solution _w_ _[′]_ is close to _w_, _⟨φ_ ( _x_ ) _, w_ _[′]_ _⟩_ also has the correct sign with high probability. Combining
these two results with the proof sketch, we have the full proof of Theorem 2.


**Conclusions and outlook**


We show that learning with quantum feature maps provides a way to harness the computational
power of quantum mechanics in machine learning problems. This idea leads to a simple quantum
machine learning algorithm that makes no additional assumptions on data access and has rigorous
and robust performance guarantees. While the learning problem we have presented here that
demonstrates an exponential quantum speed-up is not practically motivated, our result sets a
positive theoretical foundation for the search of practical quantum advantage in machine learning.
An important future direction is to construct quantum feature maps that can be applied to practical
machine learning problems that are classically challenging. The results we have established here
can be useful for the theoretical analysis of such proposals.
An important advantage of the SVM-QKE algorithm, which only uses quantum computers to

–
estimate kernel entries, is that error-mitigation techniques can be applied [47 49] when the feature
map circuit is sufficiently shallow. Our robustness analysis gives hope that an error-mitigated
quantum feature map can still maintain its computational power. Finding quantum feature maps
that are sufficiently powerful and shallow is therefore the stepping stone towards obtaining a
quantum advantage in machine learning on near-term devices.


**ACKNOWLEDGMENTS**


We thank Sergey Bravyi and Robin Kothari for helpful comments and discussions. Y.L. was supported by Vannevar Bush faculty fellowship N00014-17-1-3025 and DOE QSA grant #FP00010905.
Part of this work was done when Y.L. was a research intern at IBM. S.A. and K.T. acknowledge
support from the MIT-IBM Watson AI Lab under the project _Machine Learning in Hilbert Space_,
the IBM Research Frontiers Institute and the ARO Grant W911NF-20-1-0014.


[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, Nature **549** [, 195 (2017).](https://doi.org/10.1038/nature23474)

[2] S. Arunachalam and R. de Wolf, SIGACT News **[48](https://doi.org/10.1145/3106700.3106710)**, 41–67 (2017).

[[3] V. Dunjko and H. J. Briegel, Reports on Progress in Physics](https://doi.org/10.1088/1361-6633/aab406) **81**, 074001 (2018).

[[4] C. Ciliberto, M. Herbster, A. D. Ialongo, M. Pontil, A. Rocchetto, S. Severini, and L. Wossnig, Proceed-](https://doi.org/10.1098/rspa.2017.0551)
[ings of the Royal Society A: Mathematical, Physical and Engineering Sciences](https://doi.org/10.1098/rspa.2017.0551) **474**, 20170551 (2018).

[5] G. Carleo, I. Cirac, K. Cranmer, L. Daudet, M. Schuld, N. Tishby, L. Vogt-Maranto, and L. Zdeborov´a,

Rev. Mod. Phys. **[91](https://doi.org/10.1103/RevModPhys.91.045002)**, 045002 (2019).

[6] A. W. Harrow, A. Hassidim, and S. Lloyd, Phys. Rev. Lett. **[103](https://doi.org/10.1103/PhysRevLett.103.150502)**, 150502 (2009).

[7] N. Wiebe, D. Braun, and S. Lloyd, Phys. Rev. Lett. **[109](https://doi.org/10.1103/PhysRevLett.109.050505)**, 050505 (2012).

[8] S. Lloyd, M. Mohseni, and P. Rebentrost, Quantum algorithms for supervised and unsupervised machine
[learning (2013), arXiv:1307.0411 [quant-ph].](https://arxiv.org/abs/1307.0411)

[9] S. Lloyd, M. Mohseni, and P. Rebentrost, Nature Physics **[10](https://doi.org/10.1038/nphys3029)**, 631 (2014).

[10] P. Rebentrost, M. Mohseni, and S. Lloyd, Phys. Rev. Lett. **[113](https://doi.org/10.1103/PhysRevLett.113.130503)**, 130503 (2014).

[11] S. Lloyd, S. Garnerone, and P. Zanardi, Quantum algorithms for topological and geometric analysis of
[big data (2014), arXiv:1408.3106 [quant-ph].](https://arxiv.org/abs/1408.3106)

[[12] I. Cong and L. Duan, New Journal of Physics](https://doi.org/10.1088/1367-2630/18/7/073011) **18**, 073011 (2016).

[[13] I. Kerenidis and A. Prakash, Quantum recommendation systems (2016), arXiv:1603.08675 [quant-ph].](https://arxiv.org/abs/1603.08675)


8


[14] F. G. S. L. Brand˜ao, A. Kalev, T. Li, C. Y.-Y. Lin, K. M. Svore, and X. Wu, in _[46th International Col-](https://doi.org/10.4230/LIPIcs.ICALP.2019.27)_
_[loquium on Automata, Languages, and Programming (ICALP 2019)](https://doi.org/10.4230/LIPIcs.ICALP.2019.27)_, Leibniz International Proceedings
in Informatics (LIPIcs), Vol. 132 (2019) pp. 27:1–27:14.

[15] P. Rebentrost, A. Steffens, I. Marvian, and S. Lloyd, Phys. Rev. A **[97](https://doi.org/10.1103/PhysRevA.97.012327)**, 012327 (2018).

[16] Z. Zhao, J. K. Fitzsimons, and J. F. Fitzsimons, Phys. Rev. A **[99](https://doi.org/10.1103/PhysRevA.99.052331)**, 052331 (2019).

[17] S. Aaronson, Nature Physics **[11](https://doi.org/10.1038/nphys3272)**, 291 (2015).

[18] E. Tang, in _[Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing](https://doi.org/10.1145/3313276.3316310)_, STOC
(2019) p. 217–228.

[19] E. Tang, Quantum-inspired classical algorithms for principal component analysis and supervised clus[tering (2018), arXiv:1811.00414 [cs.DS].](https://arxiv.org/abs/1811.00414)

[20] A. Gily´en, S. Lloyd, and E. Tang, Quantum-inspired low-rank stochastic regression with logarithmic
[dependence on the dimension (2018), arXiv:1811.04909 [cs.DS].](https://arxiv.org/abs/1811.04909)

[21] N.-H. Chia, H.-H. Lin, and C. Wang, Quantum-inspired sublinear classical algorithms for solving low[rank linear systems (2018), arXiv:1811.04852 [cs.DS].](https://arxiv.org/abs/1811.04852)

[22] C. Ding, T.-Y. Bao, and H.-L. Huang, Quantum-inspired support vector machine (2019),
[arXiv:1906.08902 [cs.LG].](https://arxiv.org/abs/1906.08902)

[23] N.-H. Chia, A. Gily´en, T. Li, H.-H. Lin, E. Tang, and C. Wang, in _[Proceedings of the 52nd Annual](https://doi.org/10.1145/3357713.3384314)_
_[ACM SIGACT Symposium on Theory of Computing](https://doi.org/10.1145/3357713.3384314)_, STOC (2020) p. 387–400.

[24] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, Physical Review A **98**, 032309 (2018).

[25] E. Farhi and H. Neven, arXiv preprint arXiv:1802.06002 (2018).

[26] E. Grant, M. Benedetti, S. Cao, A. Hallam, J. Lockhart, V. Stojevic, A. G. Green, and S. Severini, npj
Quantum Information **4**, 1 (2018).

[27] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, Physical Review A **101**, 032308 (2020).

[[28] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, Quantum Science and Technology](https://doi.org/10.1088/2058-9565/ab4eb5) **4**, 043001 (2019).

[29] V. Havl´ıˇcek, A. D. C´orcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta,

Nature **567** [, 209 (2019).](https://doi.org/10.1038/s41586-019-0980-2)

[30] M. Schuld and N. Killoran, Phys. Rev. Lett. **[122](https://doi.org/10.1103/PhysRevLett.122.040504)**, 040504 (2019).

[31] B. E. Boser, I. M. Guyon, and V. N. Vapnik, in _[Proceedings of the Fifth Annual Workshop on Compu-](https://doi.org/10.1145/130385.130401)_
_[tational Learning Theory](https://doi.org/10.1145/130385.130401)_, COLT (1992) p. 144–152.

[32] V. Vapnik, _The nature of statistical learning theory_ (Springer science & business media, 2013).

[[33] M. Anthony and P. L. Bartlett, Combinatorics, Probability and Computing](https://doi.org/10.1017/S0963548300004247) **9**, 213–225 (2000).

[34] J. Shawe-Taylor, P. L. Bartlett, R. C. Williamson, and M. Anthony, IEEE Transactions on Information
Theory **44**, 1926 (1998).

[35] P. Bartlett and J. Shawe-Taylor, Generalization performance of support vector machines and other
pattern classifiers, in _Advances in Kernel Methods: Support Vector Learning_ (MIT Press, Cambridge,
MA, USA, 1999) p. 43–54.

[36] J. Shawe-Taylor and N. Cristianini, IEEE Transactions on Information Theory **48**, 2721 (2002).

[37] M. J. Kearns, _The computational complexity of machine learning_ (MIT press, 1990).

[38] R. A. Servedio and S. J. Gortler, SIAM J. Comput. **[33](https://doi.org/10.1137/S0097539704412910)**, 1067–1092 (2004).

[39] R. Sweke, J.-P. Seifert, D. Hangleiter, and J. Eisert, On the quantum versus classical learnability of
[discrete distributions (2020), arXiv:2007.14451 [quant-ph].](https://arxiv.org/abs/2007.14451)

[40] X. Gao, Z.-Y. Zhang, and L.-M. Duan, Science Advances **4** [, 10.1126/sciadv.aat9004 (2018).](https://doi.org/10.1126/sciadv.aat9004)

[41] M. J. Kearns and U. V. Vazirani, _An introduction to computational learning theory_ (MIT press, 1994).

[[42] P. W. Shor, SIAM Journal on Computing](https://doi.org/10.1137/S0097539795293172) **26**, 1484 (1997).

[43] M. Blum and S. Micali, SIAM J. Comput. **[13](https://doi.org/10.1137/0213053)**, 850–864 (1984).

[[44] D. Aharonov and A. Ta-Shma, SIAM Journal on Computing](https://doi.org/10.1137/060648829) **37**, 47 (2007).

[[45] P. L. Bartlett and P. M. Long, Journal of Computer and System Sciences](https://doi.org/https://doi.org/10.1006/jcss.1997.1557) **56**, 174 (1998).

[[46] J. W. Daniel, Mathematical Programming](https://doi.org/10.1007/BF01580110) **5**, 41 (1973).

[47] K. Temme, S. Bravyi, and J. M. Gambetta, Phys. Rev. Lett. **[119](https://doi.org/10.1103/PhysRevLett.119.180509)**, 180509 (2017).

[48] Y. Li and S. C. Benjamin, Phys. Rev. X **7** [, 021050 (2017).](https://doi.org/10.1103/PhysRevX.7.021050)

[[49] A. Kandala, K. Temme, A. D. C´orcoles, A. Mezzacapo, J. M. Chow, and J. M. Gambetta, Nature](https://doi.org/10.1038/s41586-019-1040-7) **567**,
[491 (2019).](https://doi.org/10.1038/s41586-019-1040-7)

[[50] M. Mosca and C. Zalka, International Journal of Quantum Information](https://doi.org/10.1142/S0219749904000109) **02**, 91 (2004).

[[51] C. J. Burges, Data Mining and Knowledge Discovery](https://doi.org/10.1023/A:1009715923555) **2**, 121 (1998).

[[52] A. J. Smola and B. Sch¨olkopf, Statistics and Computing](https://doi.org/10.1023/B:STCO.0000035301.49549.88) **14**, 199 (2004).


9


[53] C. Gidney and M. Eker˚a, How to factor 2048 bit rsa integers in 8 hours using 20 million noisy qubits
[(2019), arXiv:1905.09749 [quant-ph].](https://arxiv.org/abs/1905.09749)

[54] Note that here the halfspace learning problem is defined in feature space; our original concept class
is not a halfspace learning problem. Also, in our case the noise is defined in terms of additive error
in the kernel, which is different from the well-studied question of _learning halfspaces with noise_ in
computational learning theory.

[55] A. Grønlund, L. Kamma, and K. G. Larsen, Proceedings of International Conference on Machine
Learning, ICML (2020).


**Appendix A: Supervised learning and the discrete log problem**


We work in the same setting as in standard computational learning theory [37, 41]. The _data_
_space X ⊆_ R _[d]_ is a fixed subset of _d_ -dimensional Euclidean space. In this paper we will be concerned
with _distribution-dependent_ learning, i.e., we fix our _data distribution_ to be the uniform distribution
over _X_ . A _concept class C_ is a set of functions that maps data vectors to binary labels, i.e., every
_f ∈C_ is a function _f_ : _X →{−_ 1 _,_ 1 _}_ .
A learning algorithm is given a set of _training samples S_ = _{_ ( _x_ _i_ _, f_ ( _x_ _i_ )) _}_ _[m]_ _i_ =1 [, where each] _[ x]_ _[i]_ [ is]
independently drawn from the uniform distribution over _X_, and _f ∈C_ is an unknown concept
in the concept class. The goal of the learning algorithm is to return a classifier _f_ _[∗]_ that runs in
polynomial time. The _test accuracy_ of the learned classifier is defined as the probability of agreeing
with the unknown concept,


acc _f_ ( _f_ _[∗]_ ) = Pr (A1)
_x∼X_ [[] _[f]_ _[∗]_ [(] _[x]_ [) =] _[ f]_ [(] _[x]_ [)]] _[ .]_


**Definition 3** (Efficient learning of _C_ ) **.** _Let X ⊆_ R _[d]_ _. A concept class C ⊆{f_ : _X →{−_ 1 _,_ 1 _}} is_
efficiently learnable _, if there exists a learning algorithm A that satisfies the following: for every_
_f ∈C, algorithm A takes as input_ poly( _d_ ) _many training samples S and with probability at least_ 2 _/_ 3
_(over the choice of random training samples and randomness of the algorithm), outputs a classifier_
_in time_ poly( _d_ ) _that achieves 99% test accuracy._


The concept class we construct for showing our quantum advantage is based on the _discrete log_
_problem_ (DLP) which we define first:


DLP: given a prime _p_, a primitive element _g_ of Z _[∗]_ _p_ [=] _[ {]_ [1] _[,]_ [ 2] _[, . . ., p][ −]_ [1] _[}]_ [, and] _[ y][ ∈]_ [Z] _[∗]_ _p_ [, find]
_x ∈_ Z _[∗]_ _p_ [such that] _[ g]_ _[x]_ _[ ≡]_ _[y]_ [ (mod] _[ p]_ [).]


For a fixed _p, g_, we let DLP( _p, g_ ) be the discrete log problem with inputs _y ∈_ Z _[∗]_ _p_ [. The input to the]
DLP problem can be described by _n_ = _⌈_ log 2 _p⌉_ bits. It is shown [43] that DLP is reducible to the
following _decision problem_ DLP 1

2 [:]


  - **Input:** prime _p_, generator _g_ of Z _[∗]_ _p_ [,] _[ y][ ∈]_ [Z] _[∗]_ _p_ [.]


  - **Output:** 1 if log _g_ _y ≤_ _[p][−]_ 2 [1] and 0 otherwise.



**Lemma 4** ([43, Theorem 3]) **.** _For every prime p and generator g, if there exists a polynomial time_
_algorithm that correctly decides_ DLP 1 [(] _[p, g]_ [)] _[ for at least]_ [1] [+] 1 _[fraction of the inputs][ y][ ∈]_ [Z] _[∗]_ _[,]_



_algorithm that correctly decides_ DLP 12 [(] _[p, g]_ [)] _[ for at least]_ [1] 2 [+] poly1( _n_ ) _[fraction of the inputs][ y][ ∈]_ [Z] _[∗]_ _p_ _[,]_

_then there exists a polynomial time algorithm for_ DLP( _p, g_ ) _._



12 [(] _[p, g]_ [)] _[ for at least]_ [1] 2



Furthermore, [43] showed that DLP can be further reduced to the following promise discrete
logarithm problem DLP _c_ for poly1( _n_ ) _[≤]_ _[c][ ≤]_ 2 [1] [:]


  - **Input:** prime _p_, generator _g_ of Z _[∗]_ _p_ [,] _[ y][ ∈]_ [Z] _[∗]_ _p_ [.]


10




- **Promise:** log _g_ _y ∈_ [1 _, c_ ( _p −_ 1)] or [ _[p][−]_ 2 [1]




_[−]_ 2 [1] + 1 _,_ _[p][−]_ 2 [1]




_[−]_

2 + _c_ ( _p −_ 1)]




- **Output:** _−_ 1 if log _g_ _y ∈_ [1 _, c_ ( _p −_ 1)] and +1 if log _g_ _y ∈_ [ _[p][−]_ 2 [1]




_[−]_ 2 [1] + 1 _,_ _[p][−]_ 2 [1]




_[−]_

2 + _c_ ( _p −_ 1)].



1
**Lemma 5** ([43]) **.** _For every prime p, generator g and_ poly( _n_ ) _[≤]_ _[c][ ≤]_ 2 [1] _[, if there exists a polynomial]_

_time algorithm for_ DLP _c_ ( _p, g_ ) _, then there exists a polynomial time algorithm for_ DLP( _p, g_ ) _._


The proof of this fact is implicitly implied by the proof of Lemma 3 and Theorem 3 in [43].


**Appendix B: A concept class reducible to discrete log**


In this section we construct a concept class, wherein learning the concept class is as hard as
solving DLP 1

2 [(] _[p, g]_ [). Therefore, assuming the hardness of][ DLP][(] _[p, g]_ [), no classical polynomial time]
algorithm can learn this concept class. On the other hand, the learning problem is a simple
clustering problem in 1D after taking the discrete logarithm, and therefore is easy to learn using a

quantum computer.
We work in the setting introduced in the previous section, where we use standard definitions (see
Definition 3) from computational learning theory, and we assume a fixed _p, g_ such that computing
discrete log in Z _[∗]_ _p_ [is classically hard. Our concept class is defined as follows.]


**Definition 6** (Concept class) **.** _We define a concept class over the data space X_ = Z _[∗]_ _p_ _[⊆{]_ [0] _[,]_ [ 1] _[}]_ _[n]_ _[,]_
_where n_ = _⌈_ log 2 _p⌉. For any s ∈_ Z _[∗]_ _p_ _[, define a concept][ f]_ _[s]_ [:][ Z] _[∗]_ _p_ _[→{−]_ [1] _[,]_ [ 1] _[}][ as]_



_f_ _s_ ( _x_ ) =



+1 _,_ _if_ log _g_ _x ∈_ [ _s, s_ + _[p][−]_ 2 [3] []] _[,]_ (B1)
� _−_ 1 _,_ _else._



_Note that in interval_ [ _s, s_ + _[p][−]_ 2 [3] []] _[,][ s]_ [ +] _[ i][ denotes addition within]_ [ Z] _[∗]_ _p_ _[. By definition,][ f]_ _[s]_ _[maps half the]_

_elements in_ Z _[∗]_ _p_ _[to]_ [ +1] _[ and half of them to][ −]_ [1] _[. The concept class is defined as][ C]_ [ =] _[ {][f]_ _[s]_ _[}]_ _[s][∈]_ [Z] _[∗]_ _p_ _[.]_


A target concept in _C_ can be specified by choosing an element _s ∈_ Z _[∗]_ _p_ [, which can be understood]
as the “secret key” for the concept _f_ _s_ . We can also efficiently generate training samples for

every concept.


**Lemma 7.** _For every concept f ∈C, there exists an efficient classical algorithm that can generate_
_samples_ ( _x, f_ ( _x_ )) _, where x is uniformly random in_ Z _[∗]_ _p_ _[.]_


_Proof._ To generate a sample ( _x, f_ _s_ ( _x_ )) for a concept _f_ _s_, first generate a random _y ∼_ Z _[∗]_ _p_ [, and let]



_b_ =



+1 _,_ if _y ∈_ [ _s, s_ + _[p][−]_ 2 [3] []] _[,]_ (B2)
� _−_ 1 _,_ else.



Then return ( _g_ _[y]_ _, b_ ). Since _g_ _[y]_ is also uniformly distributed in Z _[∗]_ _p_ [, this procedure correctly generates]
a sample from the data distribution.


Using the quantum algorithm for discrete logarithm problem [42, 50], the concept class _C_ is
polynomially learnable in BQP (in fact with probability 1). On the other hand, the result of Blum
and Micali [43] implies that no efficient classical algorithm can do better than random guessing.


**Theorem 8.** _The concept class C is efficiently learnable in_ BQP _. On the other hand, suppose there_
_exists an efficient classical algorithm that, for every concept f ∈C, can achieve_ 2 [1] [+] poly1( _n_ ) _[test]_

_accuracy, with probability at least_ 2 _/_ 3 _over the choice of random training samples and randomness_
_of the algorithm. Then there exists an efficient classical algorithm for_ DLP _._


11



**Remark 1.** _In the following we prove a stronger statement for classical hardness. We show that_
_assuming the classical hardness of_ DLP _, no efficient classical algorithm can achieve_ [1] [+] 1 _[test]_



_assuming the classical hardness of_ DLP _, no efficient classical algorithm can achieve_ [1] 2 [+] poly1( _n_ ) _[test]_

_accuracy with probability_ [2] _[for]_ [ any] _[ concept in the concept class][ C][.]_



3 _[for]_ [ any] _[ concept in the concept class][ C][.]_



_Proof._ We first show quantum learnability. For every concept _f_ _s_ _∈C_, use a quantum computer to
take the discrete logarithm of the classical training data samples. Then, after taking the discrete
logarithm, the training data samples are clustered in two intervals: [ _s, s_ + _[p][−]_ 2 [3] [] with label +1, and]



logarithm, the training data samples are clustered in two intervals: [ _s, s_ + _[p][−]_ 2 [] with label +1, and]

[ _s_ + _[p][−]_ 2 [1] _[, s]_ [ +] _[ p][ −]_ [2] with label] _[ −]_ [1, for the unknown] _[ s][ ∈]_ [Z] _[∗]_ [(which defines the unknown concept] _[ f]_ _[s]_ [).]




[ _s_ + _[p][−]_ 2 _[, s]_ [ +] _[ p][ −]_ [2] with label] _[ −]_ [1, for the unknown] _[ s][ ∈]_ [Z] _[∗]_ _p_ [(which defines the unknown concept] _[ f]_ _[s]_ [).]

For a new data sample _x_, use a quantum computer to take its discrete log. Then, compute the
average distance _d_ + _/d_ _−_ between log _g_ _x_ and the +1/ _−_ 1 labeled clusters, respectively. Assign label
to _x_ based on which cluster is closer to log _g_ _x_ . This algorithm can achieve 99% accuracy for any
concept _f_ _s_, with high probability over random training samples. We omit the detailed proof here.
To show classical hardness as stated in Remark 1, consider an arbitrary concept _f_ _s_ and polynomially many training samples. By Lemma 7 this can be generated classically in polynomial time.
By assumption, an efficient classical algorithm _A_ can learn this concept with [1] [+] 1 [accuracy]



By assumption, an efficient classical algorithm _A_ can learn this concept with [1] 2 [+] poly1( _n_ ) [accuracy]

(call _A_ a good classifier if it satisfies this), with probability at least 2 _/_ 3. We use this learned
classifier to solve DLP 1 [(] _[p, g]_ [).]



classifier to solve DLP 1

2 [(] _[p, g]_ [).]
**Algorithm** _A_ _[′]_ **for** DLP 1 [(]



1

2 [(] _[p, g]_ [)] **[:]**



1. On input _y ∈_ Z _[∗]_ _p_ [such that log] _g_ _[y][ ∈]_ [[1] _[,]_ _[p][−]_ 2 [1]




_[−]_ [1]

2 [] or [] _[p][−]_ 2 [1]




_[−]_

2 + 1 _, p −_ 1].



2. Send _y_ _·g_ _[s][−]_ [1] to the classifier _A_, decide log _g_ _y ∈_ [1 _,_ _[p][−]_ 2 [1]



_y_ _·g_ _[s][−]_ to the classifier _A_, decide log _g_ _y ∈_ [1 _,_ _[p][−]_ 2 [] if the classifier returns +1, and decide]

log _g_ _y ∈_ [ _[p][−]_ 2 [1] + 1 _, p −_ 1] if the classifier returns _−_ 1.




_[−]_

2 + 1 _, p −_ 1] if the classifier returns _−_ 1.



To see that this procedure correctly decides DLP 1

2 [(] _[p, g]_ [) with a non-trivial bias, for a good classi-]
fier _A_ we have



Pr
_y∼_ Z _[∗]_ _p_



_A_ _[′]_ correctly decides DLP 1
� 2 [(] _[p, g]_ [) on input] _[ y]_ �



= Pr
_y∼_ Z _[∗]_ _p_



� _A_ correctly classifies _y · g_ _[s][−]_ [1] for concept _f_ _s_ �



= Pr [ _A_ correctly classifies _y_ for concept _f_ _s_ ]
_y∼_ Z _[∗]_ _p_


1

= [1]

2 [+] poly( _n_ ) _[.]_



(B3)



12 [(] _[p, g]_ [) on] 2 [1]



By Lemma 4, once we have an algorithm that can correctly decide DLP 12 [(] _[p, g]_ [) on] 2 [1] [+] poly1( _n_ ) [fraction]

of the inputs, it can be used to solve DLP( _p, g_ ) with high success probability. Finally by a simple
union bound, we have a polynomial time algorithm that with high probability solves DLP( _p, g_ ).



By Lemma 4, once we have an algorithm that can correctly decide DLP 1



An advantage of our supervised learning task is that it is efficiently verifiable by a classical
verifier. Consider the following challenge:


1. A classical verifier picks a random concept _f_ _s_ _∼C_ (it can do so by choosing a uniformly
random _s ∼_ Z _[∗]_
_p_ [). Then, generate polynomial-sized samples (] _[S, T]_ [) where the data labels in] _[ T]_
are removed.


2. The verifier sends ( _S, T_ ) to a prover, and the prover returns a set of _{−_ 1 _,_ 1 _}_ labels for _T_ .


3. The verifier accepts if more than 99% of the labels returned by the prover are correct.


12


Say a prover passes the challenge if the verifier accepts with probability at least 2 _/_ 3. Our
hardness result implies the following Corollary:


**Corollary 9.** _There exists a_ BQP _prover that can pass the above challenge. Assuming the classical_
_hardness of_ DLP _, no polynomial-time classical prover can pass the above challenge._


**Appendix C: Support vector machine and the quantum kernel estimation algorithm**


**1.** **Support vector machines**


We give a brief overview of support vector machine and the quantum kernel estimation algorithm [29, 30]. Along the way, we also establish properties that are useful for the analysis of our
algorithm in the next section. We refer to Ref. [51, 52] for a more detailed introduction to support
vector machines.

A support vector machine (SVM) is a classification algorithm that takes as input a set of training
samples _S_ = _{_ ( _x_ 1 _, y_ 1 ) _, . . .,_ ( _x_ _m_ _, y_ _m_ ) _}_ where _x_ _i_ _∈_ R _[d]_, _y ∈{−_ 1 _,_ 1 _}_ and in time poly( _d, m_ ) (assume
that the training set has polynomial size, _m_ = poly( _d_ )) returns a set of parameters ( _w, b_ ) _∈_ R _[d]_ _×_ R
which define a linear classifier _f_ _[∗]_ : _X →{−_ 1 _,_ 1 _}_ as follows


_y_ pred = _f_ _[∗]_ ( _x_ ) = sign ( _⟨w, x⟩_ + _b_ ) _,_ (C1)


where _⟨w, x⟩_ = [�] _i_ _[x]_ _[i]_ _[w]_ _[i]_ [. For a data vector] _[ x]_ [ with true label] _[ y]_ [, it is easy to see that the classifier is]
correct on this point _if and only if y_ ( _⟨w, x⟩_ + _b_ ) _>_ 0. We say a training set _S_ is _linearly separable_
if there exists ( _w, b_ ) such that


_y_ _i_ ( _⟨w, x_ _i_ _⟩_ + _b_ ) _>_ 0 _,_ for every ( _x_ _i_ _, y_ _i_ ) _∈_ _S,_ (C2)


and such a ( _w, b_ ) is called a _separating hyperplane_ for _S_ in R _[d]_ . When the training set is linearly
separable, the SVM algorithm can efficiently find a separating hyperplane by running the so-called
_hard margin primal program_


1
min _w,b_ 2 _[∥][w][∥]_ 2 [2] (C3)

s.t. _y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _≥_ 1 _,_


a convex quadratic program whose optimal solution can be obtained in polynomial time. One
important property of SVM is that it is a maximum margin classifier. For a general unnormalized
hyperplane ( _w, b_ ), define its _normalized margin_ on a training data ( _x, y_ ) as


_γ_ ˆ ( _w,b_ ) ( _x, y_ ) = 1 _y_ ( _⟨w, x⟩_ + _b_ ) (C4)
_∥w∥_ 2


and let _γ_ ( _w,b_ ) ( _x, y_ ) = _y_ ( _⟨w, x⟩_ + _b_ ) denote the unnormalized margin. It is easy to see that Eq. (C3)
returns a classifier that maximizes


( _x,y_ min ) _∈S_ _[γ]_ [ˆ] [(] _[w,b]_ [)] [(] _[x, y]_ [)] _[,]_ (C5)


which is the minimum distance from any training point to the hyperplane. The general intuition
that SVM maximizes the margin is useful for understanding the generalization bounds that we will
prove in the next section.


13


However, for most “practical” purposes, assuming _S_ is linearly separable is a strong assumption.
Additionally, when the training set _S_ is not linearly separable, Eq. (C3) does not have a feasible
solution. To find a good linear classifier with the presence of outliers, we introduce the _soft margin_
_primal program_



min 1 2 [+] _[λ]_
_w,ξ,b_ 2 _[∥][w][∥]_ [2] 2



� _ξ_ _i_ _[p]_


_i_



s.t. _y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _≥_ 1 _−_ _ξ_ _i_

_ξ_ _i_ _≥_ 0 _,_



(C6)



where _ξ_ _i_ are the _slack variables_ introduced to relax the margin constraints, with an additional
penalty term _[λ]_ 2 � _i_ _[ξ]_ _i_ _[p]_ [. For any positive integer] _[ p]_ [, Eq. (][C6][) is a convex program and is feasible. In]

this work, we focus on choosing _p_ = 2, which becomes a quadratic program. In practice it is also
common to use _p_ = 1.
For the _p_ = 2 case, we further derive the Wolfe dual program of (C6) based on Lagrangian
duality, resulting in the _L_ 2 _soft margin dual program_



2



2 _λ_



max

_α_



�



_α_ _i_ _−_ [1]

2

_i_



� _α_ _i_ [2]


_i_



�



� _α_ _i_ _α_ _j_ _y_ _i_ _y_ _j_ _⟨x_ _i_ _, x_ _j_ _⟩−_ 2 [1]

_i,j_



s.t. _α_ _i_ _≥_ 0
� _α_ _i_ _y_ _i_ = 0 _._


_i_



(C7)



Here the primal Lagrangian is given by




[1] 2 [+] _[ λ]_

2 _[∥][w][∥]_ [2] 2



_L_ _P_ = [1]



2



_α_ _i_ ( _y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _−_ 1 + _ξ_ _i_ ) _−_ �

_i_ _i_



_µ_ _i_ _ξ_ _i_ _,_ (C8)

_i_



�



_ξ_ _i_ [2] _[−]_ �

_i_ _i_



and the primal and dual optimal solutions can be connected via the Karush-Kuhn-Tucker (KKT)
conditions


_w_ = � _α_ _i_ _y_ _i_ _x_ _i_


_i_



� _α_ _i_ _y_ _i_ = 0


_i_


_λξ_ _i_ _−_ _α_ _i_ _−_ _µ_ _i_ = 0


_α_ _i_ ( _y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _−_ 1 + _ξ_ _i_ ) = 0


_µ_ _i_ _ξ_ _i_ = 0


_α_ _i_ _≥_ 0


_µ_ _i_ _≥_ 0


_ξ_ _i_ _≥_ 0


_y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _−_ 1 + _ξ_ _i_ _≥_ 0 _._


An immediate corollary of Eq. (C9) is



(C9)



_α_ _i_ = _λξ_ _i_ (C10)


at the optimal solution. This means that when _λ_ is a constant, the Lagrangian multipliers _α_ _i_ is
proportional to the slack variables _ξ_ _i_ . This is a useful property for our analysis later. In addition,
the bias parameter _b_ can be determined by the equality _y_ _i_ ( _⟨x_ _i_ _, w⟩_ + _b_ ) _−_ 1 + _ξ_ _i_ = 0 for any _α_ _i_ _̸_ = 0.


14


Finally, it will be convenient for us to work with optimizations without the bias parameter
_b ∈_ R. We show that we can assume this is without loss of generality, as we can add one extra
dimension ˜ _x_ = ( _x,_ 1) _/√_ 2 to the data vectors so that the bias parameter is absorbed into _w_ . In this

case, the _L_ 2 soft margin dual program becomes



4



2 _λ_



(C11)



max

_α_



�



_α_ _i_ _−_ [1]

4

_i_



� _α_ _i_ [2]


_i_



�



� _α_ _i_ _α_ _j_ _y_ _i_ _y_ _j_ ( _⟨x_ _i_ _, x_ _j_ _⟩_ + 1) _−_ 2 [1]

_i,j_



s.t. _α_ _i_ _≥_ 0 _._


Here we used the fact that _⟨x_ ˜ _i_ _,_ ˜ _x_ _j_ _⟩_ = 21 [(] _[⟨][x]_ _[i]_ _[, x]_ _[j]_ _[⟩]_ [+ 1), and notice that the equality constraint]
� _i_ _[α]_ _[i]_ _[y]_ _[i]_ [ = 0 is removed. This is because of the new KKT conditions]



_w_ = � _α_ _i_ _y_ _i_ _x_ _i_


_i_


_λξ_ _i_ _−_ _α_ _i_ _−_ _µ_ _i_ = 0


_α_ _i_ ( _y_ _i_ ( _⟨x_ _i_ _, w⟩_ ) _−_ 1 + _ξ_ _i_ ) = 0


_µ_ _i_ _ξ_ _i_ = 0


_α_ _i_ _≥_ 0


_µ_ _i_ _≥_ 0


_ξ_ _i_ _≥_ 0


_y_ _i_ ( _⟨x_ _i_ _, w⟩_ ) _−_ 1 + _ξ_ _i_ _≥_ 0 _,_


where _α_ _i_ = _λξ_ _i_ still hold at optimality.


**2.** **Non-linear classification**



(C12)



In this section we generalize support vector machines for _non-linear_ classification, i.e., we map
the _d_ -dimensional data vectors into a _n_ -dimensional feature space ( _n ≫_ _d_ ) via a _feature map_ :


_φ_ : _X →_ R _[n]_ _,_ (C13)


where we assume that _φ_ is normalized, i.e., it maps a unit vector to a unit vector. The feature map
is chosen prior to seeing the training data. Notice that in the dual program, training data is only
accessed via the _kernel_ matrix _K ∈_ R _[m][×][m]_ (recall that _m_ denotes the number of training samples)
defined as


_K_ ( _x_ _i_ _, x_ _j_ ) = _⟨φ_ ( _x_ _i_ ) _, φ_ ( _x_ _j_ ) _⟩._ (C14)



Therefore, it’s possible to work with an exponentially large (or even infinite-dimensional) feature
space (i.e., when _n_ is large), as long as the kernel is computable in poly( _d_ ) time.
In addition, for any feature map _φ_ 0 : _X →_ R _[n]_, we can always use a new feature map _φ_ : _X →_
R _[n]_ [+1] such that _φ_ ( _x_ ) = ( _φ_ 0 ( _x_ ) _,_ 1) _/√_ 2, which allows us to remove the bias parameter _b_ . This can



R _[n]_ such that _φ_ ( _x_ ) = ( _φ_ 0 ( _x_ ) _,_ 1) _/√_ 2, which allows us to remove the bias parameter _b_ . This can

be done via changing the kernel as _K_ ( _x_ _i_ _, x_ _j_ ) = [1] 2 [(] _[K]_ [0] [(] _[x]_ _[i]_ _[, x]_ _[j]_ [) + 1) as shown in the previous section.]



be done via changing the kernel as _K_ ( _x_ _i_ _, x_ _j_ ) = 2 [(] _[K]_ [0] [(] _[x]_ _[i]_ _[, x]_ _[j]_ [) + 1) as shown in the previous section.]

Therefore, with a suitable kernel transformation, we can run the following dual program without
loss of generality:



2



2 _λ_



(C15)



max

_α_



�



_α_ _i_ _−_ [1]

2

_i_



� _α_ _i_ [2]


_i_



�



_α_ _i_ _α_ _j_ _y_ _i_ _y_ _j_ _K_ ( _x_ _i_ _, x_ _j_ ) _−_ [1]

2

_i,j_



s.t. _α_ _i_ _≥_ 0 _._


15


Let _Q ∈_ R _[m][×][m]_ be a matrix such that _Q_ _ij_ = _y_ _i_ _y_ _j_ _K_ ( _x_ _i_ _, x_ _j_ ). Then we can write the dual program
in vectorized form




[1] _Q_ + [1]

2 _[α]_ _[T]_ � _λ_



max 1 _[T]_ _α −_ [1]
_α_ 2




[1] _α_

_λ_ [I] �



_α_ 2 _λ_ (C16)


s.t. _α ≥_ 0 _._



It is easy to see that for every _λ >_ 0, Eq. (C16) is a strongly convex quadratic program and has
a unique optimal solution. After training is finished (i.e., we obtain training samples, compute
the kernel _K_ and solve Eq. (C16)), when a learner is presented with a new test example _x_, the
classifier returns



�



_y_ pred = sign ( _⟨w, φ_ ( _x_ ) _⟩_ ) = sign



�� _i_



_α_ _i_ _y_ _i_ _K_ ( _x_ _i_ _, x_ )

_i_



_,_ (C17)



where we used the KKT condition _w_ = [�] _i_ _[α]_ _[i]_ _[y]_ _[i]_ _[φ]_ [(] _[x]_ _[i]_ [) (which can be derived by simply replacing]
_x_ _i_ with _φ_ ( _x_ _i_ ) in Eq. (C12)). So a classifier needs to evaluate the kernel function on the new test
example and output _y_ pred .


**3.** **Quantum kernel estimation**


Different from the above standard approaches, the kernel used in our quantum algorithm is
constructed by a quantum feature map. The main idea in the _quantum kernel estimation_ algorithm [29, 30] is to map classical data vectors into quantum states:


_x →|φ_ ( _x_ ) _⟩⟨φ_ ( _x_ ) _|,_ (C18)


where we use the density matrix representation to avoid global phase. Then, the kernel function
is the Hilbert-Schmidt inner product between density matrices,


_K_ ( _x_ _i_ _, x_ _j_ ) = Tr � _|φ_ ( _x_ _i_ ) _⟩⟨φ_ ( _x_ _i_ ) _| · |φ_ ( _x_ _j_ ) _⟩⟨φ_ ( _x_ _j_ ) _|_ � = _|⟨φ_ ( _x_ _i_ ) _|φ_ ( _x_ _j_ ) _⟩|_ [2] _._ (C19)


This _quantum feature map_ is implemented via a quantum circuit parameterized by _x_,


_|φ_ ( _x_ ) _⟩_ = _U_ ( _x_ ) _|_ 0 _[n]_ _⟩_ _,_ (C20)


where we assume the feature map uses _n_ qubits. Therefore, to obtain the kernel function


2
_K_ ( _x_ _i_ _, x_ _j_ ) = _⟨_ 0 _n_ _| U_ _†_ ( _x_ _i_ ) _U_ ( _x_ _j_ ) _|_ 0 _n_ _⟩_ _,_ (C21)
��� ���


we can run the quantum circuit _U_ _[†]_ ( _x_ _i_ ) _U_ ( _x_ _j_ ) on input _|_ 0 _[n]_ _⟩_, and measure the probability of
the 0 _[n]_ output.


16


**Algorithm 1** Support vector machine with quantum kernel estimation (SVM-QKE training)
**Input:** a training set _S_ = _{_ ( _x_ _i_ _, y_ _i_ ) _}_ _[m]_ _i_ =1
**Output:** _α_ 1 _, . . ., α_ _m_ (solution to the _L_ 2 soft margin dual program (C16))


1: **for** _i_ = 1 _. . . m_ **do**
2: _K_ 0 ( _x_ _i_ _, x_ _i_ ) := 1
3: **end for**

4: **for** _i_ = 1 _. . . m_ **do** _▷_ quantum kernel estimation
5: **for** _j_ = _i_ + 1 _. . . m_ **do**
6: Apply _U_ _[†]_ ( _x_ _i_ ) _U_ ( _x_ _j_ ) on input _|_ 0 _[n]_ _⟩_
7: Measure the output probability of 0 _[n]_ with _R_ shots, denoted as _p_
8: _K_ 0 ( _x_ _i_ _, x_ _j_ ) = _K_ 0 ( _x_ _j_ _, x_ _i_ ) := _p_
9: **end for**

10: **end for**
11: _K_ := [1] 2 [(] _[K]_ [0] [ +] **[ 1]** _[m][×][m]_ [)] _▷_ SVM training

12: Run the dual program (C16), record solution as _α_
13: **Return** _α_


**Algorithm 2** Support vector machine with quantum kernel estimation (SVM-QKE testing)
**Input:** a new example _x ∈X_, a training set _S_ = _{_ ( _x_ _i_ _, y_ _i_ ) _}_ _[m]_ _i_ =1 [, training parameters] _[ α]_ [1] _[, . . ., α]_ _[m]_
**Output:** _y ∈{−_ 1 _,_ 1 _}_


1: _t_ := 0

2: **for** _i_ = 1 _. . . m_ **do** _▷_ quantum kernel estimation
3: Apply _U_ _[†]_ ( _x_ ) _U_ ( _x_ _i_ ) on input _|_ 0 _[n]_ _⟩_
4: Measure the output probability of 0 _[n]_ with _R_ shots, denoted as _p_
5: _t_ := _t_ + _α_ _i_ _y_ _i_ _·_ _[p]_ [+] 2 [1]

6: **end for**
7: **Return** sign( _t_ )


We describe the full support vector machine algorithm with quantum kernel estimation (SVMQKE), with training (Algorithm 1) and testing (Algorithm 2) phases. Here, **1** denotes the all-one
matrix, and _R_ denotes the number of measurement shots for each kernel estimation circuit.
The main differences between QKE and classical kernels are two-fold:


  - On the one hand, quantum feature maps are more expressive than classical feature maps.
Therefore, a SVM trained with a quantum kernel may achieve better performance than with
classical kernels.


  - On the other hand, a fundamental feature in QKE is that we only have a noisy estimate
of the quantum kernel entries in both training and testing, due to finite sampling error in
experiment. More specifically, for each _K_ 0 ( _x_ _i_ _, x_ _j_ ) as defined in Algorithm 1, we have access
to a noisy estimator _p_ with mean equals _K_ 0 ( _x_ _i_ _, x_ _j_ ) and variance _R_ [1] [.]


Therefore, noise robustness is an important property for provable quantum advantage with QKE.
We will formally prove this in the next section.


**Appendix D: Efficient learnability with quantum kernel estimation**


**1.** **Quantum feature map**


We now define our quantum feature map for learning the concept class _C_ based on the discrete
logarithm problem (recall Definition 6). This family of states, whose construction is based on the


17


discrete logarithm problem, was first introduced in Ref. [44] to study the complexity of quantum
state generation and statistical zero knowledge.
For a prime _p_, let _n_ = _⌈_ log 2 _p⌉_ be the number of bits needed to represent _{_ 0 _,_ 1 _, . . ., p −_ 1 _}_ . For
_y ∈_ Z _[∗]_ _p_ [,] _[ k][ ∈{]_ [1] _[,]_ [ 2] _[, . . ., n][ −]_ [1] _[}]_ [, define a polynomial-sized classical circuit family] _[ {][C]_ _[y,k]_ _[}]_ [ as follows:]


_C_ _y,k_ : _{_ 0 _,_ 1 _}_ _[k]_ _→{_ 0 _,_ 1 _}_ _[n]_ _,_

(D1)
_C_ _y,k_ ( _i_ ) = _y · g_ _[i]_ (mod _p_ ) _, ∀i ∈{_ 0 _,_ 1 _}_ _[k]_ _._


It’s easy to see that _C_ _y,k_ is injective, i.e., for all _i ̸_ = _j_, we have _C_ _y,k_ ( _i_ ) _̸_ = _C_ _y,k_ ( _j_ ). Furthermore,
_C_ _y,k_ ( _i_ ) can be computed using _O_ ( _n_ ) multiplications within Z _[∗]_ _p_ [. We now show how to prepare a]
uniform superposition over the elements of _C_ _y,k_ on a quantum computer, which we refer to as the
_n_ -qubit feature state



1
_|C_ _y,k_ _⟩_ =
~~_√_~~ 2 _[k]_



�

_i∈{_ 0 _,_ 1 _}_ _[k]_



_i_
�� _y · g_ � _._ (D2)



First construct the reversible extension of _C_ _y,k_ as


_C_ ˜ _y,k_ : _{_ 0 _,_ 1 _}_ [2] _[n]_ _→{_ 0 _,_ 1 _}_ [2] _[n]_ _,_
˜ (D3)
_C_ _y,k_ _|i⟩|_ 0 _[n]_ _⟩_ = _|i⟩|C_ _y,k_ ( _i_ ) _⟩_ _,_


where _|i⟩_ uses the least significant bits of the _n_ -bit register. Then, construct a quantum circuit _U_ _y_
using the quantum algorithm for discrete log [42, 50, 53] which uses _O_ [˜] ( _n_ [3] ) gates (we use _O_ [˜] ( _·_ ) to
hide polylog factors),


_U_ _y_ _|C_ _y,k_ ( _i_ ) _⟩|_ 0 _⟩_ = _|i⟩|C_ _y,k_ ( _i_ ) _⟩_ _._ (D4)


The overall procedure for preparing _|C_ _y,k_ _⟩_ is as follows, up to adding/discarding auxiliary qubits:



1
_|_ 0 _[n]_ _⟩_ _−−−→_ _[H]_ _[⊗][k]_
~~_√_~~ 2 _[k]_



_C_ ˜ _y,k_ 1

� _|i⟩_ _−−−→_

~~_√_~~
_i∈{_ 0 _,_ 1 _}_ _[k]_



�



2 _[k]_



2 _[k]_



� _|i⟩|C_ _y,k_ ( _i_ ) _⟩_ _−−→_ _U_ _y_ _[†]_ 1

~~_√_~~
_i∈{_ 0 _,_ 1 _}_ _[k]_



�



� _|C_ _y,k_ ( _i_ ) _⟩_ _._ (D5)

_i∈{_ 0 _,_ 1 _}_ _[k]_



**Definition 10.** _Define the family of_ feature states _via the map_ ( _y, k_ ) _→|C_ _y,k_ _⟩_ _that takes classical_
_data y ∈_ Z _[∗]_ _p_ _[, k][ ∈{]_ [1] _[,]_ [ 2] _[, . . ., n][ −]_ [1] _[}][ and maps it to a][ n][-qubit feature state]_



1
_|C_ _y,k_ _⟩_ =
~~_√_~~ 2 _[k]_



�

_i∈{_ 0 _,_ 1 _}_ _[k]_



_i_
�� _y · g_ � _._ (D6)



_Such a procedure can be implemented in_ BQP _using_ _O_ [˜] ( _n_ [3] ) _gates._


In Ref. [44], it was proven that constructing these feature states is as hard as solving discrete
log, where they show that DLP 1 _/_ 6 can be reduced to estimating the inner product between _|C_ _y,k_ _⟩_
with different _y_ and _k_ . In this work, our quantum kernel is constructed via the feature map
with a fixed _k_, which is chosen prior to running the quantum kernel estimation algorithm. More
specifically, for different training samples _y, y_ _[′]_ _∈_ Z _[∗]_ _p_ [, the corresponding kernel entry is given by]


2
_K_ 0 ( _y, y_ _[′]_ ) = ��� _C_ _y,k_ �� _C_ _y_ _′_ _,k_ ��� _._ (D7)


We note that these feature states have a special structure: after taking the discrete log for
each basis state, the feature state becomes the superposition of an interval. Therefore, computing
the inner product between feature states is equivalent to computing the intersection between their
corresponding intervals. We provide more details in the following definition.


18


**Definition 11** (Interval states) **.** _For a fixed g, p, suppose y_ = _g_ _[x]_ _. The feature state can be written_
_as |C_ _y,k_ _⟩_ = ~~_√_~~ 12 _[k]_ � 2 _i_ =0 _k_ _−_ 1 �� _g_ _x_ + _i_ � _. This can be understood as “interval states” in the log space, where_

_the exponent spans an interval_ [ _x, . . ., x_ + 2 _[k]_ _−_ 1] _of length_ 2 _[k]_ _. As a consequence, since y_ = _g_ _[x]_ _is a_
_one-to-one mapping, computing the inner products between feature states is equivalent to computing_
_intersection of the corresponding intervals._


By definition, our kernel _K_ 0 is constructed using interval states with a fixed length. In order for
our quantum algorithm to solve a classically hard problem, a necessary condition is that the kernel
entries _K_ 0 ( _y, y_ _[′]_ ) cannot be efficiently estimated up to additive error. Otherwise, the quantum
kernel estimation procedure can be efficiently simulated classically. Next we show that estimating
the kernel entries is as hard as solving the discrete log problem. Although this is implied by our
main results (Theorem 8 and 22), here we give a direct proof which is a generalization of Ref. [44].


**Lemma 12.** _For an arbitrary (fixed) prime p and generator g, if there exists a polynomial time_
_algorithm such that, on input y, y_ _[′]_ _∈_ Z _[∗]_ _p_ _[, computes][ K]_ [0] [(] _[y, y]_ _[′]_ [)] _[ up to 0.01 additive error, then there]_
_exists a polynomial time algorithm for_ DLP( _p, g_ ) _._



_Proof._ We show this lemma by using an algorithm that estimates the kernel entries well to solve
DLP 116 [(] _[p, g]_ [) which in turn (by Lemma][ 5][) implies an efficient algorithm for][ DLP][(] _[p, g]_ [).] In the

following we assume _k_ = _n −_ 3, but the proof can be generalized to any _k_ = _n −_ _t_ log _n_ for some

constant _t_ .
Consider an input _y_ = _g_ _[x]_ for the problem DLP 116 [(] _[p, g]_ [), where we are promised that either]



_x ∈_ [1 _,_ _[p][−]_ [1]




_[−]_ 2 [1] + 1 _,_ _[p][−]_ 2 [1]



16 _[−]_ [1] [] or] _[ x][ ∈]_ [[] _[p][−]_ 2 [1]




_[−]_ 2 [1] + _[p]_ 16 _[−]_ [1]



_x ∈_ [1 _,_ _[p]_ 16 _[−]_ [1] [] or] _[ x][ ∈]_ [[] _[p][−]_ 2 [1] + 1 _,_ _[p][−]_ 2 [1] + _[p]_ 16 _[−]_ [1] [].] Let _y_ _[′]_ = _g_ [(] _[p]_ [+1)] _[/]_ [2] and consider feature states _|C_ _y_ _⟩_

and �� _C_ _y_ _′_ ~~�~~ . Then for the two cases,



1. If _x ∈_ [1 _,_ _[p][−]_ [1]



16 _[−]_ [1] [],] _[ |][C]_ _[y]_ _[⟩]_ [corresponds to a subinterval of [1] _[,]_ _[p][−]_ 2 [1]




_[−]_

2 [] and therefore] _[ K]_ [0] [(] _[y, y]_ _[′]_ [) = 0.]




_[−]_ 2 [1] + 1 _,_ _[p][−]_ 2 [1]



2. If _x ∈_ [ _[p][−]_ 2 [1]




_[−]_ 2 [1] + _[p]_ 16 _[−]_ [1]



_x ∈_ [ _[p][−]_ 2 + 1 _,_ _[p][−]_ 2 + _[p]_ 16 _[−]_ [], the intersection of the corresponding intervals is at least] 16 _p_ [, so]

_K_ 0 ( _y, y_ _[′]_ ) _≥_ [1] [.]



16 [.]



Therefore, an algorithm that can approximate _K_ 0 ( _y, y_ _[′]_ ) within 0 _._ 01 additive error can decide the
promise problem DLP 116 [(] _[p, g]_ [). Lemma][ 5][ now shows the lemma statement.]


**2.** **Mapping to high dimensional Euclidean space**


Now we are ready to apply the feature map in our support vector machine algorithm using
quantum kernel estimation. We recall the definition of _C_ (Definition 6) here for convenience:
_C_ = _{f_ _s_ _}_ _s∈_ Z _∗p_, where



_f_ _s_ ( _x_ ) =



+1 _,_ if log _g_ _x ∈_ [ _s, s_ + _[p][−]_ 2 [3] []] _[,]_ (D8)
� _−_ 1 _,_ else.



Consider the mapping from _x ∈_ Z _[∗]_ _p_ [to the quantum feature states described in the previous section]
(renamed here as _|φ_ ( _x_ ) _⟩_ ),



1
_x →|φ_ ( _x_ ) _⟩_ =
~~_√_~~ 2 _[k]_



2 _[k]_ _−_ 1
�


_i_ =0



_i_
�� _x · g_ � _,_ (D9)


19



where _k_ = _n −_ _t_ log _n_ for some constant _t_ to be specified later (recall that _n_ = _⌈_ log 2 _p⌉_ ). It was
shown in Definition 10 that _|φ_ ( _x_ ) _⟩_ can be prepared in BQP. Let ∆= [2] _[k]_ [+1] = _O_ ( _n_ _[−][t]_ ). Then the



shown in Definition 10 that _|φ_ ( _x_ ) _⟩_ can be prepared in BQP. Let ∆= [2] _p_ = _O_ ( _n_ _[−][t]_ ). Then the

feature states span a [∆] [=] _[ O]_ [(] _[n]_ _[−][t]_ [) fraction of the elements in][ Z] _[∗]_ [.]



feature states span a [∆] 2 [=] _[ O]_ [(] _[n]_ _[−][t]_ [) fraction of the elements in][ Z] _[∗]_ _p_ [.]

Also define the _halfspace state |φ_ _s_ _⟩_ corresponding to every concept _c_ _s_ _∈C_ as follows



1
_|φ_ _s_ _⟩_ =
~~�~~ ( _p −_ 1) _/_ 2



( _p−_ 3) _/_ 2
�


_i_ =0



�� _g_ _s_ + _i_ � _,_ for every _s ∈_ Z _[∗]_ _p_ _[.]_ (D10)



Observe that the halfspace state spans a [1] 2 [-fraction of the full space][ Z] _[∗]_ _p_ [. The following property]

shows that _|φ_ _s_ _⟩_ is a separating hyperplane in Hilbert space.


  - _| ⟨φ_ _s_ _|φ_ ( _x_ ) _⟩|_ [2] = ∆, for 1 _−_ ∆fraction of _x_ in _{x_ : _f_ _s_ ( _x_ ) = +1 _}_ .


  - _| ⟨φ_ _s_ _|φ_ ( _x_ ) _⟩|_ [2] = 0, for 1 _−_ ∆fraction of _x_ in _{x_ : _f_ _s_ ( _x_ ) = _−_ 1 _}_ .


Notice that _|φ_ _s_ _⟩_ has a _large margin_ property: it separates training samples with label +1 from
those with label _−_ 1. The probability of having an outlier (a data point that lies inside the margin
or on the wrong side of the hyperplane) is small, which equals ∆= 1 _/_ poly( _n_ ). Recall that the goal
of an SVM algorithm is to find a hyperplane that maximizes the margin on the training set, and
the above property shows that such a good hyperplane exists.

_−_
In general, learning a separating hyperplane that separates +1 _/_ 1 examples is called a _halfspace_
_learning_ problem. Rigorously speaking, our data vectors are represented by quantum states with
the Hilbert-Schmidt inner product. For simplicity, we now show that our learning problem is
equivalent to learning a halfspace in 4 _[n]_ -dimensional Euclidean space. For that, we first express a
Hermitian matrix _W ∈_ C [2] _[n]_ _[×]_ [2] _[n]_ uniquely in terms of the orthonormal Pauli basis as follows



1
_W_ =
~~_√_~~



� _w_ _p_ _σ_ _p_ _,_ (D11)

_p∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ _[n]_



�
2 _[n]_



where _σ_ _p_ _∈{_ I _,_ X _,_ Y _,_ Z _}_ _[⊗][n]_ are _n_ -qubit Pauli operators and _w_ _p_ = ~~_√_~~ 12 _[n]_ [Tr[] _[σ]_ _[p]_ _[W]_ []] _[ ∈]_ [R][ are the] _[ Fourier]_

_coefficients_ . We can use the 4 _[n]_ dimensional _Pauli vector w_ = ( _w_ _p_ ) to represent _W_, as the HilbertSchmidt inner product _⟨W, W_ _[′]_ _⟩_ = _⟨w, w_ _[′]_ _⟩_ is equivalent to the Euclidean inner product. Also note
that a pure quantum state in Hilbert space corresponds to a unit Pauli vector in Euclidean space.
The large margin property can be recast in Euclidean space with the Pauli basis representation.



**Lemma 13.** _For any concept f_ _s_ _∈C, let w_ _s_ _be the Pauli vector of |φ_ _s_ _⟩⟨φ_ _s_ _|, b_ = _−_ [∆] 2 _[, and]_ [ ˆ] _[x][ be the]_

_Pauli vector of |φ_ ( _x_ ) _⟩⟨φ_ ( _x_ ) _|. Then_




_• ⟨w_ _s_ _,_ ˆ _x⟩_ + _b_ = [∆] 2 _[, for]_ [ 1] _[ −]_ [∆] _[fraction of][ x][ in][ {][x]_ [ :] _[ f]_ _[s]_ [(] _[x]_ [) = +1] _[}][,]_




_• ⟨w_ _s_ _,_ ˆ _x⟩_ + _b_ = _−_ [∆] 2 _[, for]_ [ 1] _[ −]_ [∆] _[fraction of][ x][ in][ {][x]_ [ :] _[ f]_ _[s]_ [(] _[x]_ [) =] _[ −]_ [1] _[}][,]_



_where ⟨·, ·⟩_ _denotes Euclidean inner product._


Our SVM algorithm that uses the kernel _K_ 0 ( _x, x_ _[′]_ ) = _|⟨φ_ ( _x_ ) _|φ_ ( _x_ _[′]_ ) _⟩|_ [2] can be equivalently understood as using the kernel _K_ 0 ( _x, x_ _[′]_ ) = _⟨x,_ ˆ ˆ _x_ _[′]_ _⟩_ based on a feature map that maps _x ∈_ Z _[∗]_ _p_ [to ˆ] _[x]_ [, a 4] _[n]_

dimensional vector in Euclidean space. Finally, recall that we only have access to a noisy estimate
of _K_ 0 ( _x, x_ _[′]_ ) using quantum kernel estimation. The noisy estimator that we obtain from a quantum
computer has mean _K_ 0 ( _x, x_ _[′]_ ) and variance _R_ [1] [, where] _[ R]_ [ denotes the number of measurement shots]

for each kernel estimation circuit.


20


To summarize, here we have shown that the original problem of learning the concept class _C_ can
be mapped to a noisy halfspace learning problem [54] in high dimensional Euclidean space with the
following properties. For simplicity, below we do not specify the concept, since everything holds
equivalently for each concept in _C_ . From now on our analysis is restricted to the high dimensional
Euclidean space, and for notation simplicity we overload _x_ to represent the Pauli vector ˆ _x_ .


**Properties of noisy halfspace learning.**


1. _Data space_ : _X ⊆_ R [4] _[n]_ with unit length _∥x∥_ 2 = 1 for every _x ∈X_ . Each _x_ is associated with
a label _y ∈{−_ 1 _,_ 1 _}_ .


2. _Separability_ : the data points lie outside a margin of [∆] 2 [with high probability over the uniform]

distribution on _X_ . That is, there exists a hyperplane ( _w, b_ ) where _w ∈_ R [4] _[n]_, _∥w∥_ 2 = 1, and
_b ∈_ R, such that



= 1 _−_ ∆ _._ (D12)
�



Pr
_x∼X_



_y_ ( _⟨w, x⟩_ + _b_ ) _≥_ [∆]
� 2



3. _Bounded distance_ : all data points are close to the above hyperplane:


_|y_ ( _⟨w, x⟩_ + _b_ ) _| ≤_ [∆] for every _x ∈X_ _._ (D13)

2 _[,]_


4. _Noisy kernel_ : instead of having the ideal kernel _K_ 0 ( _x_ _i_ _, x_ _j_ ) = _⟨x_ _i_ _, x_ _j_ _⟩_, we have access to a
noisy kernel _K_ 0 _[′]_ [, where] _[ K]_ 0 _[′]_ [(] _[x]_ _[i]_ _[, x]_ _[j]_ [) =] _[ K]_ [0] [(] _[x]_ _[i]_ _[, x]_ _[j]_ [) +] _[ e]_ _[ij]_ [. Here,] _[ e]_ _[ij]_ [ are independent random]
variables satisfying


   - _e_ _ij_ _∈_ [ _−_ 1 _,_ 1]


   - E[ _e_ _ij_ ] = 0

   - Var[ _e_ _ij_ ] _≤_ _R_ [1] [, where] _[ R]_ [ denotes the number of measurement shots.]



These properties are simple corollaries of the definition of _|φ_ ( _x_ ) _⟩_, _|φ_ _s_ _⟩_ and Lemma 13.
In order to further simplify our analysis, we perform an additional transform which allows us
to remove the bias parameter _b_ without loss of generality. This will help us simplify our notations
in later proofs. More specifically, we replace _x_ with ( _x,_ 1) _/√_ 2 and _w_ with ( _w, b_ ) _/√w_ [2] + _b_ [2] . This



2 and _w_ with ( _w, b_ ) _/√_



in later proofs. More specifically, we replace _x_ with ( _x,_ 1) _/√_ 2 and _w_ with ( _w, b_ ) _/√w_ [2] + _b_ [2] . This

corresponds to replacing the original kernel _K_ 0 with a new kernel _K_ = [1] [(] _[K]_ [0] [ +] **[ 1]** _[m][×][m]_ [) where] **[ 1]**



corresponds to replacing the original kernel _K_ 0 with a new kernel _K_ = 2 [(] _[K]_ [0] [ +] **[ 1]** _[m][×][m]_ [) where] **[ 1]**

denotes the all-one matrix. These steps are explained in more detail in Section C. The final form
of our halfspace learning problem is given below.



**Lemma 14.** _We have mapped the original problem of learning the concept class C into the following_
_noisy halfspace learning problem. Below we do not specify the concept, as these properties hold for_
_every concept in C._


_1._ Data space: _X ⊆_ R [4] _[n]_ [+1] _with unit length ∥x∥_ 2 = 1 _, ∀x ∈X_ _. Each x is associated with a label_
_y ∈{−_ 1 _,_ 1 _}._


_2._ Separability: _the data points lie outside a O_ (∆) _margin with high probability over the uniform_
_distribution. That is, there exists a hyperplane w where w ∈_ R [4] _[n]_ [+1] _, ∥w∥_ 2 = 1 _, such that_



= 1 _−_ ∆ _._ (D14)
�



Pr
_x∼X_



∆
_y⟨w, x⟩≥_
� ~~_√_~~ 8 + 2∆ [2]


21


_3._ Bounded distance: _all data points are close to the above hyperplane:_


∆
_|y⟨w, x⟩| ≤_ ~~_√_~~ 8 + 2∆ [2] _[,]_ _for every x ∈X_ _._ (D15)


_4._ Noisy kernel: _let K_ _ij_ = 12 [(1 +] _[ K]_ [0] [(] _[x]_ _[i]_ _[, x]_ _[j]_ [))] _[. We have access to a noisy kernel][ K]_ _[′]_ _[, where]_
_K_ _[′]_
_ij_ [=] _[ K]_ _[ij]_ [ +] _[ e]_ _[ij]_ _[. Here,][ e]_ _[ij]_ _[ are independent random variables satisfying]_


_• e_ _ij_ _∈_ [ _−_ 1 _/_ 2 _,_ 1 _/_ 2]


_•_ E[ _e_ _ij_ ] = 0

_•_ Var[ _e_ _ij_ ] _≤_ _R_ [1] _[, where][ R][ denotes the number of measurement shots.]_


The hyperplane specified in the above lemma is particularly useful for our analysis later. We
define its unnormalized version as the “ground truth hyperplane” as follows.


**Definition 15.** _Consider the halfspace learning problem defined in Lemma 14. Define w_ _[∗]_ _∈_ R [4] _[n]_ [+1]

_as the (unnormalized) ground truth hyperplane as given in Lemma 14, that satisfies the follow-_
_ing properties:_


Pr
_x∼X_ [[] _[y][⟨][w]_ _[∗]_ _[, x][⟩≥]_ [1] = 1] _[ −]_ [∆] _[,]_ (D16)

_|y⟨w_ _[∗]_ _, x⟩| ≤_ 1 _,_ _for every x ∈X_ _._


_Note that the norm of w_ _[∗]_ _is ∥w_ _[∗]_ _∥_ 2 = _O_ (∆ _[−]_ [1] ) _._


**3.** **Generalization of the noisy classifier**


Next, we focus on the noisy halfspace learning problem as given by Lemma 14. We show that the
four properties established in Lemma 14 are sufficient for formally proving the efficient learnability
of the concept class _C_ using our quantum algorithm.
Consider the primal optimization problem in the support vector machine used by Algorithm 1:



1
min 2 [+] _[ λ]_
_w,ξ_ 2 _[∥][w][∥]_ [2] 2



� _ξ_ _i_ [2]


_i_



s.t. _y_ _i_ _⟨x_ _i_ _, w⟩≥_ 1 _−_ _ξ_ _i_

_ξ_ _i_ _≥_ 0



with the dual form



2



2 _λ_



(D17)


(D18)



max

_α_



�



_α_ _i_ _−_ [1]

2

_i_



� _α_ _i_ [2]


_i_



�



_α_ _i_ _α_ _j_ _y_ _i_ _y_ _j_ _K_ _ij_ _−_ [1]

2

_i,j_



s.t. _α_ _i_ _≥_ 0 _._


The above duality follows from the KKT conditions _w_ = [�] _i_ _[α]_ _[i]_ _[y]_ _[i]_ _[x]_ _[i]_ [ and] _[ α]_ _[i]_ [ =] _[ λξ]_ _[i]_ [. The kernel]
matrix _K_ is a positive semidefinite matrix. Let _Q_ be a matrix such that _Q_ _ij_ = _y_ _i_ _y_ _j_ _K_ _ij_, which
is also positive semidefinite. Then the dual program (D18) is equivalent to the following convex
quadratic program:



1
min _Q_ + [1]
_α_ 2 _[α]_ _[T]_ � _λ_




[1] _α −_ 1 _[T]_ _α_

_λ_ [I] �



_α_ 2 _λ_ (D19)


s.t. _α ≥_ 0 _._


22


Recall that we have small additive perturbations in _K_, which in turn gives additive perturbations
in _Q_ . One useful property of the dual program (D19) is that it is robust to perturbations in _Q_ .
More specifically, we use the following lemma from standard perturbation analysis.


**Lemma 16** ([46, Theorem 2.1]) **.** _Let x_ 0 _be the solution to the quadratic program_


1
min
_x_ 2 _[x]_ _[T]_ _[ Kx][ −]_ _[c]_ _[T]_ _[ x]_



_s.t._ _Gx ≤_ _g_


_Dx_ = _d,_



(D20)



_where K is positive definite with smallest eigenvalue λ >_ 0 _. Let K_ _[′]_ _be a positive definite matrix_
_such that ∥K_ _[′]_ _−_ _K∥_ _F_ _≤_ _ε < λ. Let x_ _[′]_ 0 _[be the solution to]_ [ (][D20][)] _[ with][ K][ replaced by][ K]_ _[′]_ _[. Then]_


_ε_
_∥x_ _[′]_ 0 _[−]_ _[x]_ [0] _[∥]_ [2] _[≤]_ (D21)
_λ −_ _ε_ _[∥][x]_ [0] _[∥]_ [2] _[.]_


Before going into the analysis of robustness against noise, notice that the bound in Lemma 16
is multiplicative. Therefore, it is useful to establish an upper bound on _∥α∥_ 2, the norm of the
solution to the noiseless quadratic program (D18). Recall from the KKT conditions that _α_ _i_ = _λξ_ _i_,
where the dual variables are directly related to the slack variables in the primal program (D17).
The following lemma establishes a useful property for _ξ_ _i_ _[∗]_ [for the ground truth hyperplane.]


**Lemma 17.** _For the ground truth hyperplane w_ _[∗]_ _as defined in Definition 15, the corresponding_
_slack variables ξ_ _i_ _[∗]_ _[in the primal program]_ [ (][D17][)] _[ satisfy]_


E � _∥ξ_ _[∗]_ _∥_ 2 [2] � _≤O_ ( _m_ ∆) _,_ (D22)


_where the expectation is taken over the training set._


_Proof._ We can write the slack variables as


_ξ_ _i_ _[∗]_ [= max] _[{]_ [1] _[ −]_ _[y]_ _[i]_ _[⟨][x]_ _[i]_ _[, w]_ _[∗]_ _[⟩][,]_ [ 0] _[}][.]_ (D23)


By the properties given in Definition 15, we have



Therefore,



Pr [ _ξ_ _i_ _[∗]_ [= 0] = 1] _[ −]_ [∆] _[,]_ (D24)

_ξ_ _i_ _[∗]_ _[≤]_ [2] _[.]_


E � _∥ξ_ _[∗]_ _∥_ 2 [2] � = _m_ E � _ξ_ 1 _[∗]_ [2] � _≤_ 4 _m_ ∆ _._ (D25)



Now we are ready to bound the norm of the dual variables _α_ _i_ . Intuitively, we can do so because
_∥ξ∥_ [2] 2 [is part of the training loss in the primal program (][D17][). The loss of the solution returned by]
the program can only be smaller than the loss of the ground truth hyperplane, which is guaranteed
to be small.


**Lemma 18.** _Let α_ 0 _be the solution returned by the dual program_ (D18) _. We have_


1
E � _∥α_ 0 _∥_ 2 [2] � = _O_ [+] _[ m]_ [∆] _,_ (D26)
� ∆ [2] �


_where the expectation is over the training set._


23


_Proof._ Let _w_ 0 = [�] _i_ _[α]_ [0] _[i]_ _[y]_ _[i]_ _[x]_ _[i]_ [ be the hyperplane which corresponds to] _[ α]_ [0] [, and let] _[ ξ]_ [0] [ be the corre-]
sponding slack variable. Then


_∥α_ 0 _∥_ 2 [2] [=] _[ λ]_ [2] _[∥][ξ]_ [0] _[∥]_ [2] 2 _[≤]_ _[λ]_ � _∥w_ 0 _∥_ 2 [2] [+] _[ λ][∥][ξ]_ [0] _[∥]_ [2] 2 � _≤_ _λ_ � _∥w_ _[∗]_ _∥_ 2 [2] [+] _[ λ][∥][ξ]_ _[∗]_ _[∥]_ [2] 2 � _._ (D27)


Here, the first line follows from the KKT condition _α_ _i_ = _λξ_ _i_, and the third line is because _w_ 0 is
the optimal solution to (D17). Therefore by Lemma 17, E � _∥α_ 0 _∥_ [2] 2 � = _O_ � ∆1 [2] [+] _[ m]_ [∆] �.


**Remark 2.** _Recall that we have the freedom to choose_ ∆= _O_ ( _n_ _[−][t]_ ) _for any constant t. Suppose_
_we have polynomially many training samples m ≈_ _n_ _[c]_ _, and let t_ = _c/_ 3 _. Then the above bound gives_
E � _∥α_ 0 _∥_ [2] 2 � = _O_ � _m_ [2] _[/]_ [3] [�] _._


Having established the above lemmas, now we are ready to prove our key result for noise
robustness (Lemma 19). Let _Q_ _[′]_ be the noisy kernel measured by a quantum computer, and let
_λ ∈_ (0 _,_ 1) be a constant. Here we briefly recall the steps in Algorithm 1 and 2. The classifier
is constructed in two steps. First, use a classical computer to run the dual program (D19) with
_Q_ replaced by the experimental estimate _Q_ _[′]_, and let _α_ _[′]_ be the solution returned by the program.
Second, given a new data sample _x_, use a quantum computer to obtain noisy estimates _K_ _[′]_ ( _x, x_ _i_ )
for all _i_, and output



�



_y_ pred = sign



�� _i_



_α_ _i_ _[′]_ _[y]_ _[i]_ _[K]_ _[′]_ [(] _[x, x]_ _[i]_ [)]

_i_



_._ (D28)



Let _h_ ( _x_ ) = [�] _i_ _[α]_ _[i]_ _[y]_ _[i]_ _[K]_ [(] _[x, x]_ _[i]_ [) and] _[ h]_ _[′]_ [(] _[x]_ [) =][ �] _i_ _[α]_ _i_ _[′]_ _[y]_ _[i]_ _[K]_ _[′]_ [(] _[x, x]_ _[i]_ [), which corresponds to the value of the]
noiseless/noisy classifier before taking the sign. We will prove the following result which establishes
the noise robustness of _h_ .


**Lemma 19** (Noise robustness) **.** _Suppose we take R_ = _O_ ( _m_ [4] ) _measurement shots for each quantum_
_kernel estimation circuit. Then, with probability at least 0.99 (over the choice of random training_
_samples and measurement noise), for every x ∈X we have_

_′_
�� _h_ ( _x_ ) _−_ _h_ ( _x_ )�� _≤_ 0 _._ 01 _._ (D29)


_Proof._ Consider the (noisy) quadratic program (D18). The Frobenius norm is given by


_∥Q_ _[′]_ _−_ _Q∥_ _F_ [2] [= 2] � _e_ [2] _ij_ _[,]_ (D30)

_i<j_


where _e_ _ij_ are independent random variables satisfying E[ _e_ _ij_ ] = 0 and E � _e_ [2] _ij_ � _≤_ _R_ 1 [.] Therefore

E � _∥Q_ _[′]_ _−_ _Q∥_ [2] _F_ � _≤O_ _mR_ [2] . Now we invoke Lemma 16 and Lemma 18 (see Remark 2). Using
� �

Markov’s inequality: with probability at least 0.999 (over the choice of training samples and measurement noise), we have that



2
_m_

_−_
_∥Q_ _[′]_ _Q∥_ _F_ [2] _[≤O]_
� _R_



_,_ and _∥α∥_ 2 [2] [=] _[ O]_ � _m_ [2] _[/]_ [3] [�] _._ (D31)
�



Let _δ_ _i_ = _α_ _i_ _[′]_ _[−]_ _[α]_ _[i]_ [. Since] _[ λ]_ [min] � _Q_ + _λ_ [1]



_λ_ [is lower bounded by a constant, Lemma][ 16][ gives]



_λ_ [1] [I] � _≥_ _λ_ [1]



_∥δ∥_ 2 _≤O_



_m_ [4] _[/]_ [3]


~~_√_~~ _R_

�



~~_√_~~



_R_



�



_._ (D32)


24



Then, let _ν_ _i_ = _K_ _[′]_ ( _x, x_ _i_ ) _−_ _K_ ( _x, x_ _i_ ) for _i_ = 1 _, . . ., m_ . Similarly, _ν_ _i_ are independent random variables
1 _√_ ~~_m_~~
satisfying E[ _ν_ _i_ ] = 0 and E � _ν_ _i_ [2] � _≤_ _R_ [.] By Markov’s inequality, we have _∥ν∥_ 2 _≤O_ � ~~_√_~~ _R_ � with



1 _√_ ~~_m_~~
satisfying E[ _ν_ _i_ ] = 0 and E � _ν_ _i_ [2] � _≤_ _R_ [.] By Markov’s inequality, we have _∥ν∥_ 2 _≤O_ � ~~_√_~~ _R_ � with

probability at least 0.999. Overall for any _x ∈X_, the error bound gives



_R_



_α_ _i_ _y_ _i_ _K_ ( _x, x_ _i_ )

_i_



�����



_′_
�� _h_ ( _x_ ) _−_ _h_ ( _x_ )�� =



�����



�



( _α_ _i_ + _δ_ _i_ ) _y_ _i_ ( _K_ ( _x, x_ _i_ ) + _ν_ _i_ ) _−_ �

_i_ _i_



_≤_ � _|α_ _i_ _ν_ _i_ + _δ_ _i_ _K_ ( _x, x_ _i_ ) + _δ_ _i_ _ν_ _i_ _|_


_i_

_≤∥α∥_ 2 _· ∥ν∥_ 2 + _[√]_ _m∥δ∥_ 2 + _∥δ∥_ 2 _· ∥ν∥_ 2



(D33)



�



_≤O_



_m_ [11] _[/]_ [6]


~~_√_~~ _R_

�



~~_√_~~



_R_



_,_



where the third line uses Cauchy–Schwarz inequality. Therefore, _R_ = _O_ ( _m_ [4] ) measurement shots
is sufficient for achieving _|h_ ( _x_ ) _−_ _h_ _[′]_ ( _x_ ) _| ≤_ 0 _._ 01, and by a simple union bound this holds with
probability at least 0.99.


Having established noise robustness, it remains to prove a generalization error bound for the
noisy classifier: if the classifier has small training error/loss, it should also have small test error, which is referred to as _generalization error_ in learning theory. The main idea is a two-step ar
gument:


1. The noiseless classifier _y_ = sign ( _h_ ( _x_ )) (we have defined _h_ ( _x_ ) = [�] _i_ _[α]_ _[i]_ _[y]_ _[i]_ _[K]_ [(] _[x, x]_ _[i]_ [) =] _[ ⟨][w, x][⟩]_ [)]
has small generalization error, which follows from standard generalization bounds for soft
margin classifiers.


2. We have established that the noisy classifier is close to the noiseless classifier. Therefore, the
noisy classifier should also have small generalization error.


For the first step, we refer to standard results on the generalization of soft margin classifiers (see,
for example [33–36, 45, 55]). Recall that a hyperplane _w_ correctly classifies a data point ( _x, y_ ) if and
only if _y⟨w, x⟩_ _>_ 0. Therefore for a specific concept _f ∈C_, the test accuracy of _f_ _[∗]_ ( _x_ ) = sign ( _⟨w, x⟩_ )
is given by


acc _f_ ( _f_ _[∗]_ ) = Pr (D34)
_x∼X_ [[] _[f]_ _[∗]_ [(] _[x]_ [) =] _[ f]_ [(] _[x]_ [)] = 1] _[ −]_ _x_ [Pr] _∼X_ [[] _[y][⟨][w, x][⟩]_ _[<]_ [ 0]] _[,]_


where we have used _y_ = _f_ ( _x_ ). Our results will be given in the form of an upper bound on
Pr _x∼X_ [ _y⟨w, x⟩_ _<_ 0]. The following result gives a generalization bound that coincides with our
_L_ 2 training loss up to polylog factors, as indicated by the _O_ [˜] notation, and therefore is directly
applicable to the noiseless classifier.


**Lemma 20** ([36, Theorem VII.11]) **.** _For any hyperplane w satisfying the constraints of the primal_
_program_ (D17) _, with probability_ 1 _−_ _δ over randomly drawn training set S of size m, the general-_
_ization error is bounded by_



˜

[1] _O_ _∥w∥_ 2 [2] [+] _[ ∥][ξ][∥]_ [2] 2 [+ log 1]

_m_ � _δ_



Pr
_x∼X_ [[] _[y][⟨][w, x][⟩]_ _[<]_ [ 0]] _[ ≤]_ _m_ [1]



_δ_



_._ (D35)
�



However, although this result establishes step 1, it cannot be directly applied to step 2: our
noise robustness result, which states that _h_ _[′]_ ( _x_ ) is close to _h_ ( _x_ ), does not guarantee that sign( _h_ _[′]_ ( _x_ ))
agrees well with sign( _h_ ( _x_ )). The above lemma implies that _h_ ( _x_ ) is on the correct side of the origin


25


with high probability, but it could still be very close to the origin, which may lead to a bad noisy
classifier sign( _h_ _[′]_ ( _x_ )).
A simple solution to this problem is to show a stronger generalization bound, which in addition
to _h_ ( _x_ ) being correct, also shows that _h_ ( _x_ ) is bounded away from the origin. We indeed prove such
a result, by combining ideas from the aforementioned references. Notice that the only difference
between the following lemma and the previous lemma is that we replaced 0 with 0.1.


**Lemma 21.** _For any hyperplane w satisfying the constraints of the primal program_ (D17) _, with_
_probability_ 1 _−δ over randomly drawn training set S of size m, the generalization error is bounded by_



˜

[1] _O_ _∥w∥_ 2 [2] [+] _[ ∥][ξ][∥]_ 2 [2] [+ log 1]

_m_ � _δ_



Pr
_x∼X_ [[] _[y][⟨][w, x][⟩]_ _[<]_ [ 0] _[.]_ [1]] _[ ≤]_ [1]



_δ_



_._ (D36)
�



The proof is presented in Section E. Combining Lemma 19 with Lemma 21, we arrive at our
main theorem for the learnability of _C_ with our quantum algorithm.


**Theorem 22.** _For any concept f_ _s_ _∈C, the_ SVM _-_ QKE _algorithm returns a classifier with test_
_accuracy at least 0.99 in polynomial time, with probability at least_ 2 _/_ 3 _over the choice of random_
_training samples and over noise._


_Proof._ Below we do not specify the concept, as the proof works equivalently for every concept
_f_ _s_ _∈C_ . Let _w_ _[∗]_ be the ground truth hyperplane as in Definition 15. Note that _∥w_ _[∗]_ _∥_ 2 = _O_ (∆ _[−]_ [1] ).
Using Lemma 17, we have that with probability at least 0.99 over the choice of training samples,
the _L_ 2 training loss of _w_ _[∗]_ satisfies




[1] 2 [+] _[ λ]_

2 _[∥][w]_ _[∗]_ _[∥]_ [2] 2



1

[+] _[ m]_ [∆] _._ (D37)
∆ [2] �



Loss( _w_ _[∗]_ ) := [1]



1
2 _[≤O]_
2 _[∥][ξ]_ _[∗]_ _[∥]_ [2] � ∆



Let _w_ 0 be the optimal solution of the primal program (D17), and let _h_ ( _x_ ) = _⟨w_ 0 _, x⟩_ . Let _h_ _[′]_ be the
noisy classifier obtained by the dual program (D18). By Lemma 19, for any _x ∈X_ we have


_|yh_ _[′]_ ( _x_ ) _−_ _yh_ ( _x_ ) _| ≤_ 0 _._ 01 _,_ (D38)


with probability at least 0.99 over the choice of training samples and noise. Therefore, by a simple
union bound, with probability at least 2 _/_ 3, the test error of the noisy classifier is upper bounded by


Pr
_x∼X_ [[] _[yh]_ _[′]_ [(] _[x]_ [)] _[ <]_ [ 0]] _[ ≤]_ _x_ [Pr] _∼X_ [[] _[yh]_ _[′]_ [(] _[x]_ [)] _[ <]_ [ 0] _[.]_ [09]]


_≤_ Pr
_x∼X_ [[] _[y][⟨][w]_ [0] _[, x][⟩]_ _[<]_ [ 0] _[.]_ [1]]



˜

_≤_ [1] _O_ (Loss( _w_ 0 ))

_m_



(D39)



˜ ˜ 1

_≤_ [1] _O_ (Loss( _w_ _[∗]_ )) _≤_ _O_ [+ ∆] _,_

_m_ � _m_ ∆ [2] �



where in the third line we use Lemma 21, and the fourth line is because _w_ 0 is the optimal solution
to (D17). Finally, notice that the above bound holds for arbitrary ∆= _O_ � _n_ _[−][t]_ [�] for constant _t_ . In
order to optimize this bound, we can choose _t_ = _c/_ 3 for _m_ = _n_ _[c]_ (also see Remark 2). This gives
the final bound

Pr _m_ _[−]_ [1] _[/]_ [3] [�] _._ (D40)
_x∼X_ [[] _[yh]_ _[′]_ [(] _[x]_ [)] _[ <]_ [ 0]] _[ ≤]_ _[O]_ [˜] �


Therefore, polynomially many training samples are sufficient for learning the concept class _C_ with
high accuracy.


26


**Appendix E: Generalization bound for soft margin SVM**


In this section we prove Lemma 21, a generalization bound for the L2 soft margin SVM in
Eq. (D17) (restated below for convenience).



1
min 2 [+] _[ λ]_
_w,ξ_ 2 _[∥][w][∥]_ [2] 2



� _ξ_ _i_ [2]


_i_



(E1)



s.t. _y_ _i_ _⟨x_ _i_ _, w⟩≥_ 1 _−_ _ξ_ _i_

_ξ_ _i_ _≥_ 0 _._



Let _w_ be an unnormalized feasible solution to Eq. (E1). The first step is to use a trick developed
by [36], that converts the soft margin problem to a hard margin problem by mapping to a larger
space. Let _S_ = _{_ ( _x_ 1 _, y_ 1 ) _, . . .,_ ( _x_ _m_ _, y_ _m_ ) _}_ be the training set. Consider the mapping


_x �→_ _x_ ˜ = ( _x, δ_ _x_ )



� _yδ_ _x_ _· ξ_ ( _x, y, w_ )

( _x,y_ ) _∈S_







(E2)




˜
_w �→_ _w_ =







_w,_ �
 ( _x,y_ )



where _δ_ _x_ : _X →{_ 0 _,_ 1 _}_ is a function defined as _δ_ _x_ ( _x_ _[′]_ ) = 1 if and only if _x_ _[′]_ = _x_ and _ξ_ ( _x, y, w_ ) =
max _{_ 0 _,_ 1 _−_ _y⟨w, x⟩}_ are the slack variables used in Eq. (E1). Denote the enlarged space by _L_ _X_ and
_∥· ∥_ its induced norm. The following useful properties hold for this transform:


1. If ( _x, y_ ) _∈_ _S_, _y⟨w,_ ˜ ˜ _x⟩≥_ 1.


˜
2. If ( _x, y_ ) _/∈_ _S_, _⟨w,_ ˜ _x⟩_ = _⟨w, x⟩_ .


˜
3. _∥w∥_ [2] = _∥w∥_ [2] 2 [+] _[ ∥][ξ][∥]_ 2 [2] [.]


4. _∥x_ ˜ _∥_ [2] = 2.


In the following, we can assume that the training data does not appear when testing the classifier,
which is the case with high probability. By property 2, to bound the generalization performance
of the hyperplane _w_, we only need to bound the generalization performance of ˜ _w_ in the enlarged
space, which corresponds to a hard margin problem.
For the generalization error of hard margin classifiers, it is well-known that the generalization
error bound is captured by the VC-dimension which characterizes the complexity of the classifier
family. Intuitively, a hard margin classifier corresponds to a “thick” hyperplane in the data space,
which reduces its complexity compared with margin-less hyperplanes. The relevant complexity
measure in our results is the so-called _fat-shattering dimension_ which we define now.


**Definition 23** ([36, Definition III.4]) **.** _Let F ⊆{f_ : _X →_ R _} be a set of real-valued functions. We_
_say that a set of points_ _X_ [ˆ] _⊆_ _X is γ-shattered by F, if there are real numbers r_ _x_ _indexed by x ∈_ _X_ [ˆ]
_such that for all binary vectors b_ _x_ _indexed by x ∈_ _X_ [ˆ] _, there is a function f_ _b_ _∈F satisfying_



_f_ _b_ ( _x_ )



_≥_ _r_ _x_ + _γ,_ _if b_ _x_ = 1 _,_
(E3)
� _≤_ _r_ _x_ _−_ _γ,_ _otherwise._



_The γ-fat-shattering dimension of F, denoted_ fat _F_ ( _γ_ ) _, is the size of the largest set_ _X_ [ˆ] _that is γ-_
_shattered by F, if this is finite or infinity otherwise._


27


Let _H_ be a set of linear functions that map from _L_ _X_ to R, such that their norm equals to _∥w_ ˜ _∥_ .
Since the data vectors ˜ _x_ have bounded norm, the fat-shattering dimension of _H_ was shown [36] to
be bounded by



˜ 2
_∥w∥_
fat _H_ ( _γ_ ) _≤O_
� _γ_ [2]



_._ (E4)
�



Next we invoke the following lemma from Ref. [33] (also see [45]) which uses fat-shattering dimension to understand generalization bounds for learning real-valued concept classes.


**Lemma 24** ([33, Corollary 3.3]) **.** _Let C, H be sets of functions that map from a set X to_ [0 _,_ 1] _. Then_
_for all η, γ, δ ∈_ (0 _,_ 1) _, for every f ∈C and for every probability measure D on X_ _, with probability_
_at least_ 1 _−_ _δ (over the choice of S_ = _{x_ 1 _, . . ., x_ _m_ _} where x_ _i_ _∼D), if h ∈H and |h_ ( _x_ _i_ ) _−_ _f_ ( _x_ _i_ ) _| ≤_ _η_
_for_ 1 _≤_ _i ≤_ _m, then_



_γ_

[1] fat _H_

_m_ _[·]_ [ ˜] _[O]_ � � 8



_._ (E5)
�



Pr � _|h_ ( _x_ ) _−_ _f_ ( _x_ ) _| ≥_ _η_ + _γ_ � _≤_ [1]
_x∼D_



8



� + log [1] _δ_



1 1
To apply this lemma, consider the function _h_ (˜ _x_ ) = _∥w_ ˜ _∥_ _[⟨][w,]_ [˜] [ ˜] _[x][⟩]_ [. Let] _[ γ]_ [0] [ =] _∥w_ ˜ _∥_ [and] _[ γ]_ [ = 0] _[.]_ [9] _[γ]_ [0] [. Let]
_y_ = _f_ (˜ _x_ ) _∈{−_ 1 _,_ 1 _}_ be any labeling rule, and _S_ = _{_ (˜ _x_ 1 _, y_ 1 ) _, . . .,_ (˜ _x_ _m_ _, y_ _m_ ) _}_ be a training set. By the
properties of the mapping, for all ˜ _x ∈_ _S_ we have _yh_ (˜ _x_ ) _≥_ _γ_ 0, which means that


_|h_ (˜ _x_ ) _−_ _f_ (˜ _x_ ) _|_ = _|yh_ (˜ _x_ ) _−_ 1 _| ≤_ 1 _−_ _γ_ 0 _._ (E6)


Applying Lemma 24, we have with probability at least 1 _−_ _δ_,




[1] _O_ ˜ _∥w_ ˜ _∥_ [2] + log [1]

_m_ � _δ_



Pr � _|h_ (˜ _x_ ) _−_ _f_ (˜ _x_ ) _| ≥_ 1 _−_ 0 _._ 1 _γ_ 0 � _≤_ [1]



_δ_



_._ (E7)
�



Finally, note that _yh_ (˜ _x_ ) _≤_ 0 _._ 1 _γ_ 0 implies that _|h_ (˜ _x_ ) _−_ _f_ (˜ _x_ ) _| ≥_ 1 _−_ 0 _._ 1 _γ_ 0, therefore




[1] _O_ ˜ _∥w_ ˜ _∥_ [2] + log [1]

_m_ � _δ_



Pr [ _yh_ (˜ _x_ ) _≤_ 0 _._ 1 _γ_ 0 ] _≤_ [1]



_δ_



_._ (E8)
�


(E9)



This concludes the proof of Lemma 21, as


Pr
_x∼X_ [[] _[y][⟨][w, x][⟩]_ _[<]_ [ 0] _[.]_ [1] = Pr] _x∼X_ [[] _[y][⟨][w,]_ [˜] [ ˜] _[x][⟩]_ _[<]_ [ 0] _[.]_ [1]]


= Pr
_x∼X_ [[] _[yh]_ [(˜] _[x]_ [)] _[ <]_ [ 0] _[.]_ [1] _[γ]_ [0] []]



˜ ˜

[1] _O_ _∥w∥_ [2] + log [1]

_m_ � _δ_



_≤_ [1]



_δ_



� = [1]



˜

[1] _O_ _∥w∥_ 2 [2] [+] _[ ∥][ξ][∥]_ [2] 2 [+ log 1]

_m_ � _δ_



_δ_



_._
�


