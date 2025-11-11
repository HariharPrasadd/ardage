**A Neural Decoder for Topological Codes**


Giacomo Torlai and Roger G. Melko
_Department of Physics and Astronomy, University of Waterloo, Ontario N2L 3G1, Canada_
_Perimeter Institute of Theoretical Physics, Waterloo, Ontario N2L 2Y5, Canada_
(Dated: October 17, 2016)


We present an algorithm for error correction in topological codes that exploits modern machine
learning techniques. Our decoder is constructed from a stochastic neural network called a Boltzmann
machine, of the type extensively used in deep learning. We provide a general prescription for the
training of the network and a decoding strategy that is applicable to a wide variety of stabilizer codes
with very little specialization. We demonstrate the neural decoder numerically on the well-known
two dimensional toric code with phase-flip errors.



_Introduction_ : Much of the success of modern machine
learning stems from the flexibility of a given neural network architecture to be employed for a multitude of different tasks. This generalizability means that neural networks can have the ability to infer structure from vastly
different data sets with only a change in optimal hyperparameters. For this purpose, the machine learning community has developed a set of standard tools, such as
fully-connected feed forward networks [1] and Boltzmann
machines [2]. Specializations of these underlie many of
the more advanced algorithms, including convolutional
networks [3] and deep learning [4, 5], encountered in
real-world applications such as image or speech recognition [6].
These machine learning techniques may be harnessed
for a multitude of complex tasks in science and engineering [7–16]. An important application lies in quantum
computing. For a quantum logic operation to succeed,
noise sources which lead to decoherence in a qubit must
be mitigated. This can be done through some type of
quantum error correction – a process where the logical
state of a qubit is encoded redundantly so that errors can
be corrected before they corrupt it. A leading candidate
for this is the implementation of fault-tolerant hardware
through _surface codes_, where a logical qubit is stored as a
topological state of an array of _physical_ qubits [17]. Random errors in the states of the physical qubits can be
corrected before they proliferate and destroy the logical
state. The quantum error correction protocols that perform this correction are termed “decoders”, and must be
implemented by classical algorithms running on conventional computers [18].
In this paper we demonstrate how one of the simplest
stochastic neural networks for unsupervised learning, the
restricted Boltzmann machine [19], can be used to construct a general error-correction protocol for stabilizer
codes. Give a _syndrome_, defined by a measurement of
the end points of an (unknown) chain of physical qubit
errors, we use our Boltzmann machine to devise a protocol with the goal of correcting errors without corrupting
the logical bit. Our decoder works for generic degenerate stabilizers codes that have a probabilistic relation
between syndrome and errors, which does not have to be


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||_↵_|_β_|**_r_**_0_|||
||**_e_**|_γ_||||
|||||||
||**_r_**_00_|||||



FIG. 1. Several operations on a 2 _D_ toric code. Logical operators _Z_ [ˆ] _L_ [(1)] and _Z_ [ˆ] _L_ [(1)] (orange) are non-trivial cycles on the real
lattice. A physical error chain _**e**_ (purple) and its syndrome
_S_ ( _**e**_ ) (black squares). A recovery chain _**r**_ _[′]_ (green), with the
combined operator on the cycle _**e**_ _⊕_ _**r**_ _[′]_ being a product of
stabilizers _Z_ [ˆ] _α_ _Z_ [ˆ] _β_ _Z_ [ˆ] _γ_ (recovery success). A recovery chain _**r**_ _[′′]_

(red) whose cycle has non-trivial homology and acts on the
code state as _Z_ [ˆ] _L_ [(1)] (logical failure).


_a priori_ known. Importantly, it is very simple to implement, requiring no specialization regarding code locality,
dimension, or structure. We test our decoder numerically
on a simple two-dimensional surface code with phase-flip

errors.

_The_ 2D _Toric Code._ Most topological codes can be
described in terms of the stabilizer formalism [20]. A stabilizer code is a particular class of error-correcting code
characterized by a protected subspace _C_ defined by a stabilizer group _S_ . The simplest example is the 2 _D_ toric
code, first introduced by Kitaev [21]. Here, the quantum information is encoded into the homological degrees
of freedom, with topological invariance given by the first
homology group [22]. The code features _N_ qubits placed
on the links of a _L × L_ square lattice embedded on a
torus. The stabilizers group is _S_ = _{Z_ [ˆ] _p_ _,_ _X_ [ˆ] _v_ _}_, where the
plaquette and vertex stabilizers are defined respectively
as _Z_ [ˆ] _p_ = [�] _ℓ∈p_ _[σ]_ [ˆ] _ℓ_ _[z]_ [and ˆ] _[X]_ _[v]_ [ =][ �] _ℓ∈v_ _[σ]_ [ˆ] _ℓ_ _[x]_ [, with ˆ] _[σ]_ _ℓ_ _[z]_ [and ˆ] _[σ]_ _ℓ_ _[x]_
acting respectively on the links contained in the plaque


_Z_ ˆ _L_ [(2)]












2



tte _p_ and the links connected to the vertex _v_ . There are
two encoded logical qubits, manipulated by logical operators _Z_ [ˆ] _L_ [(1] _[,]_ [2)] as ˆ _σ_ _[z]_ acting on the non-contractible loops on
the real lattice and logical _X_ [ˆ] _L_ [(1] _[,]_ [2)] as the non-contractible
loops on the dual lattice (Fig 1).


Given a reference state _|ψ_ 0 _⟩∈C_, let us consider the
simple phase-flip channel described by a Pauli operator
where ˆ _σ_ _[z]_ is applied to each qubit with probability _p_ _err_ .
This operator can be efficiently described by a mapping
between the links and Z 2, called an error chain _**e**_, whose
boundary is called a syndrome _**S**_ ( _**e**_ ). In a experimental implementation, only the syndrome (and not the error chain) can be measured. Error correction (decoding)
consists of applying a recovery operator whose chain _**r**_
generates the same syndrome, _**S**_ ( _**e**_ ) = _**S**_ ( _**r**_ ). The recovery succeeds only if the combined operation is described
by a _cycle_ (i.e. a chain with no boundaries) _**e**_ _⊕_ _**r**_ that
belongs to the trivial homology class _h_ 0, describing contractable loops on the torus. On the other hand, if the
cycle belongs to a non-trivial homology class (being noncontractible on the torus), the recovery operation directly
manipulates the encoded logical information, leading to
a logical failure (Fig 1).


Several decoders have been proposed for the 2 _D_ toric
code, based on different strategies [23–27]. Maximum
likelihood decoding consists of finding a recovery chain
_**r**_ with the most likely homology class [28, 29]. A different recovery strategy, designed to reduce computational
complexity, consists of generating the recovery chain _**r**_
compatible with the syndrome simply by using the minimum number of errors. Such a procedure, called Minimum Weight Perfect Matching [30] (MWPM), has the
advantage that can be performed without the knowledge
of the error probability _p_ _err_ . This algorithm is however
sub-optimal (with lower threshold probability [22]) since
it does not take into account the high degeneracy of the
error chains given a syndrome.


_The Neural Decoder._ Neural networks are commonly
used to extract features from raw data in terms of probability distributions. In order to exploit this for error
correction, we first build a dataset made of error chains
and their syndromes _D_ = _{_ _**e**_ _,_ _**S**_ _}_, and train a neural network to model the underlying probability distribution
_p_ _data_ ( _**e**_ _,_ _**S**_ ). Our goal is to then generate error chains
to use for the recovery. We use a generative model called
a Boltzmann machine, a powerful stochastic neural network widely used in the pre-training of the layers of deep
neural networks [31, 32]. The network architecture features three layers of stochastic binary neurons, the syndrome layer _**S**_ _∈{_ 0 _,_ 1 _}_ _[N/]_ [2], the error layer _**e**_ _∈{_ 0 _,_ 1 _}_ _[N]_,
and one hidden layer _**h**_ _∈{_ 0 _,_ 1 _}_ _[n]_ _[h]_ (Fig. 2). Symmetric edges connect both the syndrome and the error layer
with the hidden layer. We point out the this network is
equivalent to a traditional bilayer restricted Boltzmann
machine, where we have here divided the visible layer








##### **_d_**





FIG. 2. The neural decoder architecture. The hidden layer _**h**_
is fully-connected to the syndrome and error layers _**S**_ and _**e**_


into two separate layers for clarity. The weights on the
edges connecting the network layers are given by the ma
trices _**U**_ and _**W**_ with zero diagonal. Moreover, we also
add external fields _**b**_, _**c**_ and _**d**_ coupled to the every neuron in each layer. The probability distribution that the
probabilistic model associates to this graph structure is
the Boltzmann distribution [33]


_p_ _**λ**_ ( _**e**_ _,_ _**S**_ _,_ _**h**_ ) = [1] e _[−][E]_ _**[λ]**_ [(] _**[e]**_ _[,]_ _**[S]**_ _[,]_ _**[h]**_ [)] (1)

_Z_ _**λ**_


where _Z_ _**λ**_ = Tr _{_ _**h**_ _,_ _**S**_ _,_ _**E**_ _}_ e _[−][E]_ _**[λ]**_ [(] _**[e]**_ _[,]_ _**[S]**_ _[,]_ _**[h]**_ [)] is the partition function, _**λ**_ = _{_ _**U**_ _,_ _**W**_ _,_ _**b**_ _,_ _**c**_ _,_ _**d**_ _}_ is the set of parameters of the
model, and the energy is



_E_ _**λ**_ ( _**e**_ _,_ _**S**_ _,_ _**h**_ ) = _−_ �



_U_ _ik_ _h_ _i_ _S_ _k_ _−_ �
_ik_ _ij_



_W_ _ij_ _h_ _i_ _e_ _j_ +

_ij_



(2)
_d_ _k_ _S_ _k_ _._

_k_



_−_
�



� _b_ _j_ _e_ _j_ _−_ �

_j_ _i_



_c_ _i_ _h_ _i_ _−_ �

_i_ _k_



The joint probability distribution over ( _**e**_ _,_ _**S**_ ) is obtained
after integrating out the hidden variables from the full
distribution



_p_ _**λ**_ ( _**e**_ _,_ _**S**_ _,_ _**h**_ ) = [1]

_Z_

_**h**_



_p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) = �



e _[−E]_ _**[λ]**_ [(] _**[e]**_ _[,]_ _**[S]**_ [)] (3)
_Z_ _**λ**_



where the effective energy _E_ _**λ**_ ( _**e**_ _,_ _**S**_ ) can be computed
exactly. Moreover, given the structure of the network, the conditional probabilities _p_ _**λ**_ ( _**e**_ _|_ _**h**_ ), _p_ _**λ**_ ( _**S**_ _|_ _**h**_ )
and _p_ _**λ**_ ( _**h**_ _|_ _**e**_ _,_ _**S**_ ) are also known exactly. The training of
the machine consists of tuning the parameters _**λ**_ until the
model probability _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) becomes close to the target
distribution _p_ _data_ ( _**E**_ _,_ _**S**_ ) of the dataset. This translates
into solving an optimization problem over the parameters
_**λ**_ by minimizing the distance between the two distribution, defined as the Kullbach-Leibler (KL) divergence,
KL _∝−_ [�] ( _**e**_ _,_ _**S**_ ) _∈D_ [log] _[ p]_ _**[λ]**_ [(] _**[e]**_ _[,]_ _**[ S]**_ [). Details about the Boltz-]

mann machine and its training algorithm are reported in
the Supplementary Materials.


3



We now discuss the decoding algorithm, which proceeds assuming that we successfully learned the distribution _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ). Given an error chain _**e**_ 0 with syndrome
_**S**_ 0 we wish to use the Boltzmann machine to generate
an error chain compatible with _**S**_ 0 to use for the recovery. To achieve this goal we separately train networks on
different datasets obtained from different error regimes
_p_ _err_ . Assuming we know the error regimes that generated _**e**_ 0, the recovery procedure consists of sampling
a recovery chain from the distribution _p_ _**λ**_ ( _**e**_ _|_ _**S**_ 0 ) given
by the network trained at the same probability _p_ _err_ of
_**e**_ 0 . Although the Boltzmann machine does not learn
this distribution directly, by sampling the error and hidden layers while keeping the syndrome layer fixed to _**S**_ 0,
since _p_ _**λ**_ ( _**e**_ _,_ _**S**_ 0 ) = _p_ _**λ**_ ( _**e**_ _|_ _**S**_ 0 ) _p_ ( _**S**_ 0 ), we are enforcing sampling from the desired conditional distribution. An advantage of this procedure over decoders that employ conventional Monte Carlo [25, 26] on specific stabilizer codes
is that specialized sampling algorithms tied to the stabilizer structure, or multi-canonical methods such as parallel tempering, are not required.
An error correction procedure can be defined as follows (Alg. 1): we first initialize the machine into a random state of the error and hidden layers (see Fig. 2) and
to _**S**_ 0 for the syndrome layer. We then let the machine
equilibrate by repeatedly performing block Gibbs sampling. After a some amount of equilibration steps, we
begin checking the syndrome of the error state _**e**_ in the
machine and, as soon as _**S**_ ( _**e**_ ) = _**S**_ 0 we select it for the
recovery operation.


**Algorithm 1** Neural Decoding Strategy


1: _**e**_ 0 : physical error chain
2: _**S**_ 0 = _**S**_ ( _**e**_ 0 ) _▷_ Syndrome Extraction
3: RBM = _{_ _**e**_ _,_ _**S**_ = _**S**_ 0 _,_ _**h**_ _}_ _▷_ Network Initialization
4: **while** _**S**_ ( _**e**_ ) _̸_ = _**S**_ 0 **do** _▷_ Sampling
5: Sample _**h**_ _∼_ _p_ ( _**h**_ _|_ _**e**_ _,_ _**S**_ 0 )
6: Sample _**e**_ _∼_ _p_ ( _**e**_ _|_ _**h**_ )
7: **end while**

8: _**r**_ = _**e**_ _▷_ Decoding


_Results._ We train neural networks in different error
regimes by building several datasets _D_ _p_ = _{_ _**e**_ _k_ _,_ _**S**_ _k_ _}_ _[M]_ _k_ =1
at elementary error probabilities _p_ = _{_ 0 _._ 5 _,_ 0 _._ 6 _, . . .,_ 0 _._ 15 _}_
of the phase-flip channel. For a given error probability,
the network hyper-parameters are individually optimized
via a grid search (for details see the Supplementary Material). Once training is complete, we perform decoding
following the procedure laid out in Alg. 1. We generate a
test set _T_ _p_ = _{_ _**e**_ _k_ _}_ _[M]_ _k_ =1 [and for each error chain] _**[ e]**_ _[k]_ _[ ∈T]_ _[p]_ [, af-]
ter a suitable equilibration time (usually _N_ _eq_ _∝_ 10 [2] sampling steps), we collect the first error chain _**e**_ compatible
with the original syndrome, _**S**_ ( _**e**_ ) = _**S**_ ( _**e**_ _k_ ). We use this
error chain for the recovery, _**r**_ [(] _[k]_ [)] = _**e**_ . Importantly, error
recovery with _**r**_ [(] _[k]_ [)] chosen from the first compatible chain
means that the cycle _**e**_ _k_ + _**r**_ [(] _[k]_ [)] is sampled from a distri



### _P_

_fail_

### _p err_


FIG. 3. Logical failure probability as a function of elementary
error probability for MWPM (lines) and the neural decoder
(markers) of size _L_ = 4 (red) and _L_ = 6 (green).


bution that includes all homology classes. By computing
the Wilson loops on the cycles we can measure their homology class. This allows us to gauge the accuracy of the
decoder in term of the logical failure probability, defined
as _P_ _fail_ = _[n]_ _[f]_ _M_ _[ail]_ where _n_ _fail_ is the number of cycles with

non-trivial homology. Because of the fully-connected architecture of the network, and the large complexity of
the probability distribution arising from the high degeneracy of error chains given a syndrome, we found that
the dataset size required to accurately capture the underlying statistics must be relatively large ( _|D_ _p_ _| ∝_ 10 [5] ).
In Fig. 3 we plot the logical failure probability _P_ _fail_ as a
function of the elementary error probability for the neural decoding scheme.
To compare our numerical results we also perform
error correction using the recovery scheme given by
MWPM [34]. This algorithm creates a graph whose vertices corresponds to the syndrome and the edges connect
each vertex with a weight equal to the Manhattan distance (the number of links connecting the vertices in the
original square lattice). MWPM then finds an optimal
matching of all the vertices pairwise using the minimum
weight, which corresponds to the minimum number of
edges in the lattice [35]. Fig. 3 displays the comparison between a MWPM decoder (line) and our neural
decoder (markers). As is evident, the neural decoder has
an almost identical logical failure rate for error probabilities below the threshold ( _p_ _err_ _≈_ 10 _._ 9 [22]), yet a significant higher probability above. Note that by training
the Boltzmann machine on different datasets we have enforced in the neural decoder a dependence on the error
probability. This is in contrast to MWPM which is performed without such knowledge. Another key difference
is that the distributions learned by the Boltzmann machine contain the entropic contribution from the high degeneracy of error chains, which is directly encoded into
the datasets. It will be instructive to explore this fur

|Col1|p = 0.05<br>err|Col3|p = 0.08<br>err|
|---|---|---|---|
|||||
|||||


|Col1|p = 0.12<br>err|Col3|p = 0.15<br>err|
|---|---|---|---|
|||||
|||||


_h_ 0 _h_ 1 _h_ 2 _h_ 3 _h_ 0 _h_ 1 _h_ 2 _h_ 3


FIG. 4. Histogram of the homology classes returned by our
neural decoder for various elementary error probabilities _p_ _err_ .
The green bars represent the trivial homology class _h_ 0 corresponding to contractable loops on the torus. The other three
classes correspond respectively to the logical operations _Z_ ˆ _L_ [(2)] and _Z_ [ˆ] _L_ [(1)] _Z_ ˆ _L_ [(2)] _[.]_ _Z_ [ˆ] _L_ [(1)] [,]


ther, to determine whether the differences in Fig. 3 come
from inefficiencies in the training, the different decoding
model of the neural network, or both. Finite-size scaling
on larger _L_ will allow calculation of the threshold defined
by the neural decoder.

In the above algorithm, which amounts to a simple
and practical implementation of the neural decoder, our
choice to use the first compatible chain for error correction means that the resulting logical operation is sampled
from a distribution that includes all homology classes.
This is illustrated in Fig. 4, where we plot the histogram
of the homology classes for several different elementary
error probabilities. Accordingly, our neural decoder can
easily be modified to perform Maximum Likelihood (ML)
optimal decoding. For a given syndrome, instead of obtaining only one error chain to use in decoding, one could
sample many error chains and build up the histogram
of homology classes with respect to any reference error state. Then, choosing the recovery chain from the
largest histogram bin will implement, by definition, ML
decoding. Although the computational cost of this procedure will clearly be expensive using the current fullyconnected restricted Boltzmann machine, it would be interesting to explore specializations of the neural network
architecture in the future to see how its performance may
compare to other ML decoding algorithms [28]
_Conclusions._ We have presented a decoder for topological codes using a simple algorithm implemented with a
restricted Boltzmann machine, a common neural network
used in many machine learning applications. Our neural
decoder is easy to program using standard machine learning software libraries and training techniques, and relies
on the efficient sampling of error chains distributed over
all homology classes. Numerical results show that our



4


decoder has a logical failure probability that is close to
MWPM, but not identical, a consequence of our neural
network being trained separately at different elementary
error probabilities. This leads to the natural question of
the relationship between the neural decoder and optimal
decoding, which could be explored further by a variation
of our algorithm that implements maximum likelihood
decoding.


In its current implementation, the Boltzmann machine
is restricted within a given layer of neurons, but fullyconnected between layers. This means that our decoder
does not depend on the specific geometry used to implement the code, nor on the structure of the stabilizer
group; it is trained simply using a raw data input vector,
with no information on locality or dimension. In order
to scale up our system sizes on the 2 _D_ toric code (as
required e.g. to calculate the threshold), one could relax
some of the general fully-connected structure of the network, and specialize it to accommodate the specific details of the code. This specialization should be explored
in detail, before comparisons of computational efficiency
can be made between our neural decoder, MWPM, and
other decoding schemes. Note that, even with moderate
specialization, the neural decoder as we have presented
above can immediately be extended to other choices of
error models [36], such as the more realistic case of imperfect syndrome measurement [37], or transferred to other
topological stabilizer codes, such as color codes [38, 39].


Finally, it would be interesting to explore the improvements in performance obtained by implementing standard tricks in machine learning, such as convolutions,
adaptive optimization algorithms, or the stacking of multiple Boltzmann machines into a network with deep structure. Given the rapid advancement of machine learning
technology within the world’s information industry, we
expect that such tools will be the obvious choice for the
real-world implementation of decoding schemes on future
topologically fault-tolerant qubit hardware.


_Acknowledgements._ The authors thank J. Carrasquilla,
D. Gottesman, M. Hastings, C. Herdmann, B. Kulchytskyy, and M. Mariantoni for enlightening discussions.
This research was supported by NSERC, the CRC program, the Ontario Trillium Foundation, the Perimeter
Institute for Theoretical Physics, and the National Science Foundation under Grant No. NSF PHY-1125915.

Simulations were performed on resources provided by
SHARCNET. Research at Perimeter Institute is supported through Industry Canada and by the Province of
Ontario through the Ministry of Research & Innovation.


[[1] K. Hornik, M. Stinchcombe, and H. White, Neural Net-](http://www.sciencedirect.com/science/article/pii/0893608089900208)
works **2** [, 359 (1989).](http://www.sciencedirect.com/science/article/pii/0893608089900208)

[[2] R. Salakhutdinov, Technical Report UTML, Dep. Comp.](http://www.cs.toronto.edu/~rsalakhu/papers/bm.pdf)


[Sc., University. of Toronto, 002 (2008).](http://www.cs.toronto.edu/~rsalakhu/papers/bm.pdf)

[3] A. Krizhevsky, I. Sutskever, [and G. Hinton, Proc. Ad-](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
[vances in Neural Information Processing Systems](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **25**,
[1090 (2012).](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[[4] G. Hinton, Trends in Cognitive Science](http://www.cs.toronto.edu/~fritz/absps/tics.pdf) **10**, 428 (2007).

[5] Y. LeCun, Y. Bengio, and G. Hinton, Nature **[521](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html)**, 436
[(2008).](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html)

[6] G. Hinton and _et al_ [, IEEE Signal Processing Magazine](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38131.pdf)
**29** [, 82 (2012).](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38131.pdf)

[[7] G. Torlai and R. Melko, arXiv:1606.02718 (2016).](https://arxiv.org/abs/1606.02718)

[[8] L. Wang, arXiv:1606.00318 (2016).](https://arxiv.org/abs/1606.00318)

[[9] J. Carrasquilla and R. G. Melko, arXiv:1605.01735](https://arxiv.org/abs/1605.01735)
[(2016).](https://arxiv.org/abs/1605.01735)

[10] P. Broecker, J. Carrasquilla, R. G. Melko, and S. Trebst,

[arXiv:1608.07848 (2016).](https://arxiv.org/abs/1608.07848)

[11] K. Ch’ng, J. Carrasquilla, R. G. Melko, and E. Khatami,

[arXiv:1609.02552 (2016).](https://arxiv.org/abs/1609.02552)

[[12] G. Carleo and M. Troyer, arXiv:1606.02318 (2016).](https://arxiv.org/abs/1606.02318)

[13] D.-L. Deng, X. Li, [and S. D. Sarma, arXiv:1609.09060](https://arxiv.org/abs/1609.09060)
[(2016).](https://arxiv.org/abs/1609.09060)

[[14] L. Huang and L. Wang, arXiv:1610.02746 (2016).](https://arxiv.org/abs/1610.02746)

[[15] J. Liu, Y. Qi., Z. Y. Meng, and L. Fu, arXiv:1610.03137](https://arxiv.org/abs/1610.03137)
[(2016).](https://arxiv.org/abs/1610.03137)

[[16] E. M. Stoudenmire and D. J. Schwab, arXiv:1605.05775](https://arxiv.org/abs/1605.05775)
[(2016).](https://arxiv.org/abs/1605.05775)

[[17] H. Bombin, Quantum Error Correction (2013).](http://www.cambridge.org/us/academic/subjects/physics/quantum-physics-quantum-information-and-quantum-computation/quantum-error-correction)

[18] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. C.
Cleland, Phys. Rev. A **[86](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.180501)**, 032324 (2012).

[[19] G. Hinton, Neural Networks: Tricks of the Trade, 599](http://link.springer.com/chapter/10.1007/978-3-642-35289-8_32)
[(2012).](http://link.springer.com/chapter/10.1007/978-3-642-35289-8_32)

[[20] D. Gottesman, arXiv:quant-ph/9705052 (1997).](https://arxiv.org/abs/quant-ph/9705052)

[[21] A. Y. Kitaev, Annals of Physics](http://www.sciencedirect.com/science/article/pii/S0003491602000180) **1**, 2 (2003).

[[22] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, Jour-](http://scitation.aip.org/content/aip/journal/jmp/43/9/10.1063/1.1499754)
[nal of Mathematical Physics](http://scitation.aip.org/content/aip/journal/jmp/43/9/10.1063/1.1499754) **43**, 4452 (2002).

[[23] G. Duclos-Cianci and D. Poulin, Phys. Rev. Lett.](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.050504) **104**,
[050504 (2010).](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.050504)

[[24] G. Duclos-Cianci and D. Poulin, Quant. Inf. Comp.](https://arxiv.org/abs/1304.6100) **14**,
[0721 (2014).](https://arxiv.org/abs/1304.6100)

[[25] J. R. Wootton and D. Loss, Phys. Rev. Lett.](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.160503) **109**, 160503
[(2012).](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.160503)

[26] A. Hutter, J. R. Wootton, [and D. Loss, Phys. Rev. A](http://dx.doi.org/10.1103/PhysRevA.89.022326)
**89** [, 022326 (2014).](http://dx.doi.org/10.1103/PhysRevA.89.022326)

[[27] A. Fowler, arXiv:1310.0863 .](https://arxiv.org/abs/1310.0863)

[[28] S. Bravyi, M. Suchara, and A. Vargo, Phys. Rev. A](http://journals.aps.org/pra/abstract/10.1103/PhysRevA.90.032326) **90**,
[032326 (2014).](http://journals.aps.org/pra/abstract/10.1103/PhysRevA.90.032326)

[29] B. Heim, K. M. Svore, and M. B. Hastings,
[arXiv:1609.06373 .](https://arxiv.org/abs/1609.06373)

[30] J. Edmonds, Canadian Journal of Mathematics **17**, 449
(1997).

[[31] G. Hinton, S. Osindero, and Y. Teh, Neural computation](http://www.mitpressjournals.org/doi/abs/10.1162/neco.2006.18.7.1527#.VzSfdWamvEY)
**18** [, 1527 (2006).](http://www.mitpressjournals.org/doi/abs/10.1162/neco.2006.18.7.1527#.VzSfdWamvEY)

[[32] R. Salakhutdinov and I. Murray, ICML’08 Proceedings](http://dl.acm.org/citation.cfm?id=1390266)
[of the 25th international conference on machine learning](http://dl.acm.org/citation.cfm?id=1390266)
[, 872 (2008).](http://dl.acm.org/citation.cfm?id=1390266)

[[33] A. Fischer and C. Igel, Progress in Pattern Recognition,](http://dx.doi.org/10.1007/978-3-642-33275-3_2)
[Image Analysis, Computer Vision, and Applications, 14](http://dx.doi.org/10.1007/978-3-642-33275-3_2)
[(2012).](http://dx.doi.org/10.1007/978-3-642-33275-3_2)

[[34] V. Kolmogorov, Math. Prog. Comp.](http://link.springer.com/article/10.1007/s12532-009-0002-8) **1**, 43 (2002).

[35] A. G. Fowler, A. C. Whiteside, and L. C. L. Hollemberg,

Phys. Rev. Lett. **[108](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.180501)**, 180501 (2012).

[[36] E. Novais and E. R. Mucciolo, Phys. Rev. Lett.](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.010502) **110**,
[010502 (2013).](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.010502)

[37] C. Wang, J. Harrington, [and J. Preskill, Annals of](http://www.sciencedirect.com/science/article/pii/S0003491602000192)
Physics **[1](http://www.sciencedirect.com/science/article/pii/S0003491602000192)**, 31 (2003).



5


[38] H. G. Katzgraber, H. Bombin, and M. A. MartinDelgado, Phys. Rev. Lett **[103](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.090501)**, 090501 (2009).

[[39] B. J. Brown, N. H. Nickerson, and D. E. Browne, Nature](http://www.nature.com/articles/ncomms12302)
[Communications](http://www.nature.com/articles/ncomms12302) **7** (2016).

[[40] G. Hinton, Neural computation](http://www.mitpressjournals.org/doi/abs/10.1162/089976602760128018#.VyfQIT9kCHs) **14**, 1771 (2002).

[[41] A. Krogh and J. A. Hertz, Advances in neural networks](http://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf)
[information processing systems](http://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf) **4**, 950 (1992).


**SUPPLEMENTARY MATERIAL**


_Training the Boltzmann Machine._ We have seen that
the training of the neural network consists in finding a
set of parameters _**λ**_ which minimizes the distance between the dataset distribution and the model distribution _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ). This probability distribution is obtained
from the distribution over the full graph by integrating
out the hidden degrees of freedom



The gradient of the KL divergence thus reduces to the
gradient of the log probability


_∇_ _**λ**_ _j_ log _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) = _−∇_ _**λ**_ _j_ _E_ _**λ**_ ( _**e**_ _,_ _**S**_ )+

+ � log _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) _∇_ _**λ**_ _j_ _E_ _**λ**_ ( _**e**_ _,_ _**S**_ ) (7)

_**e**_ _,_ _**S**_


For instance, for the case of the derivative with respect
to the weights matrix _**W**_, the gradient of the effective
energy is equal to the correlation matrix averaged over
the conditional probability of the hidden layer


_∇_ _**W**_ _E_ _**λ**_ ( _**e**_ _,_ _**S**_ ) = _−_ � _p_ _**λ**_ ( _**h**_ _|_ _**e**_ _,_ _**S**_ ) _**e h**_ _[⊤]_ (8)


_**h**_


Therefore the gradient of the KL divergence with respect
to _**W**_ can be written as


_∇_ _**W**_ KL = _−⟨_ _**e h**_ _[⊤]_ _⟩_ _p_ _**λ**_ ( _**h**_ _|_ _**e**_ _,_ _**S**_ ) + _⟨_ _**e h**_ _[⊤]_ _⟩_ _p_ _**λ**_ ( _**h**_ _,_ _**e**_ _,_ _**S**_ ) (9)


We note now that, due to the restricted nature of
the Boltzmann machine (no intra-layer connections), all



_p_ _**λ**_ ( _**e**_ _,_ _**S**_ _,_ _**h**_ ) = [1]

_Z_

_**h**_



_p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) = �



e _[−E]_ _**[λ]**_ [(] _**[e]**_ _[,]_ _**[S]**_ [)] (4)
_Z_ _**λ**_



Such marginalization can be carried out exactly, leading
to an effective energy



_E_ _**λ**_ ( _**e**_ _,_ _**S**_ ) = _−_ �



_b_ _j_ _e_ _j_ _−_ �
_j_ _k_



_d_ _k_ _S_ _k_ +

_k_



_−_
� log �1 + e _[c]_ _[i]_ [+][�]

_i_



_k_ _[U]_ _[ik]_ _[S]_ _[k]_ [+][�]



(5)
_j_ _[W]_ _[ij]_ _[e]_ _[j]_ [�]



The function to minimize is the average of the KL divergence on the dataset samples, which can be written, up
to constant entropy term, as



KL( _p_ _data_ _| p_ _**λ**_ ) = _−_ [1]

_|D|_



� log _p_ _**λ**_ ( _**e**_ _,_ _**S**_ ) (6)

_{_ _**e**_ _,_ _**S**_ _}_


the conditional probabilities factorize over the nodes
of the corresponding layer and can be exactly calculated using Bayes theorem. For instance, the hidden
layer conditional distribution factorize as _p_ _**λ**_ ( _**h**_ _|_ _**e**_ _,_ _**S**_ ) =
� _i_ _[p]_ _**[λ]**_ [(] _[h]_ _[i]_ _[ |]_ _**[ e]**_ _[,]_ _**[ S]**_ [), where each hidden nodes is activated]

with probability



6


entire dataset, we divide _D_ into mini-batches _D_ [[] _[b]_ []] and
update _**λ**_ for each mini-batch _b_



_η_
_**λ**_ _j_ _←_ _**λ**_ _j_ _−_ _|D_ [[] _[b]_ []] _|_



� _∇_ _**λ**_ _j_ KL ( _p_ _data_ _|| p_ _**λ**_ ) (12)

_{_ _**E**_ _,_ _**S**_ _}∈D_ [[] _[b]_ []]



where the step _η_ of the gradient descent is called learning rate. The initial values of the parameters are drawn
from an uniform distribution centered around zero with

some width _w_ . We also note that a common issue arising
in the training of neural networks is the overfitting, i.e.
the network reproducing very well the distribution contained in the training dataset but being unable to properly generalize the learned features. To avoid overfitting
we employ weight-decay regularization, by adding an extra penalty term to the KL divergence, proportional to
the square weights times a coefficient _L_ 2 [41]. It is now
clear that, in addition to the network parameters _**λ**_, there
are several hyper-parameters dictating the performance
of the training. Specifically, the hyper-parameters are
the learning rate _η_, the size of the mini-batches _b_ _S_, the
width _w_ of the distribution for the initial values of the

weights, the order _κ_ in the contrastive divergence algorithm, the amplitude _L_ 2 of weight decay regularization
and the number of hidden units _n_ _h_ . To find a suitable
choice of these external hyper-parameters, we perform a
grid search where we train different networks for several
combinations of such parameters. We then select the network with higher performance in terms of logical failure
probability evaluated over a reference set of error chains.



_p_ _**λ**_ ( _h_ _i_ = 1 _|_ _**e**_ _,_ _**S**_ ) = 1 + e _[c]_ _[i]_ [+][�]
�



_k_ _[U]_ _[ik]_ _[S]_ _[k]_ [+][�]



_j_ _[W]_ _[ij]_ _[e]_ _[j]_ [�] _[−]_ [1] (10)



In computing the gradient of the KL divergence, the first
average of the correlation matrix in Eq. 9 is trivial since,
given the state of the error layer _**e**_ _∈D_, we can easily
sample the hidden state _**h**_ with the above conditional
probability. On the other hand, the second term involves an average over the full probability distribution
_p_ _**λ**_ ( _**h**_ _,_ _**e**_ _,_ _**S**_ ), whose partition function is not know and
thus inaccessible. To calculate such average correlations
we instead run a Markov chain Monte Carlo for _κ_ steps


_{_ _**e**_ _,_ _**S**_ _}_ [(0)] _→_ _**h**_ [(0)] _→· · · →{_ _**e**_ _,_ _**S**_ _}_ [(] _[κ]_ [)] _→_ _**h**_ [(] _[κ]_ [)] (11)


Because _{_ _**e**_ _,_ _**S**_ _}_ [(0)] _∈D_ and thus already belongs to the
distribution we are sampling from, there is no need of
running a long chain. This algorithm, given the number
of steps _κ_ of the Markov chain, is called contrastive divergence (CD _κ_ ) [40]. The optimization algorithm used to
update the parameters _**λ**_ is called stochastic gradient descent. Instead of evaluating the average gradient on the


