**Experimental neural network enhanced quantum tomography** _[∗]_


Adriano Macarone Palmieri, [1] Egor Kovlakov, [2] Federico Bianchi, [3] Dmitry
Yudin, [1] Stanislav Straupe, [2] Jacob D. Biamonte, [1] and Sergei Kulik [2]

1 _Deep Quantum Labs, Skolkovo Institute of Science and Technology, Moscow 121205, Russia_
2 _Quantum Technologies Centre, Faculty of Physics,_
_M. V. Lomonosov Moscow State University, Moscow 119991, Russia_
3 _Department of Computer Science, University of Milan-Bicocca, Milan 20133, Italy_


Quantum tomography is currently ubiquitous for testing any implementation of a quantum information processing device. Various sophisticated procedures for state and process reconstruction
from measured data are well developed and benefit from precise knowledge of the model describing
state preparation and the measurement apparatus. However, physical models suffer from intrinsic limitations as actual measurement operators and trial states cannot be known precisely. This
scenario inevitably leads to state-preparation-and-measurement (SPAM) errors degrading reconstruction performance. Here we develop and experimentally implement a machine learning based
protocol reducing SPAM errors. We trained a supervised neural network to filter the experimental
data and hence uncovered salient patterns that characterize the measurement probabilities for the
original state and the ideal experimental apparatus free from SPAM errors. We compared the neural
network state reconstruction protocol with a protocol treating SPAM errors by process tomography,
as well as to a SPAM-agnostic protocol with idealized measurements. The average reconstruction
fidelity is shown to be enhanced by 10% and 27%, respectively. The presented methods apply to
the vast range of quantum experiments which rely on tomography.



_Introduction._ Rapid experimental progress realizing
quantum enhanced technologies places an increased demand on methods for validation and testing. As
such, various approaches to augment state- and processtomography have recently been proposed. A persistent problem faced by these contemporary approaches
are systematic errors in state preparation and measurements (SPAM). Such notoriously challenging errors are
inevitable in any experimental realization [2–12]. Here
we develop a data-driven, deep-learning based approach
to augment state- and detector-tomography that successfully minimized SPAM error on quantum optics experimental data.
Several prior approaches have been developed to circumvent the SPAM problem. One line of thought
leads to the so-called _randomized benchmarking_ protocols

[3, 13, 14], which were designed for quality estimation of
quantum gates in the quantum circuit model. The idea is
to average the error over a large set of randomly chosen
gates, thus effectively minimizing the average influence
of SPAM. Randomized benchmarking in its initial form
however, only allowed to estimate an average fidelity for
the set of gates, so more elaborate and informative procedures were developed [4, 15]. Another example is _gate_
_set tomography_ [5, 16, 17]. Therein the experimental
apparatus is treated as a black box with external controls allowing for (i) state preparation, (ii) application
of gates and (iii) measurement. These unknown components (i)-(iii) are inferred from measurement statistics.
Both approaches require long sequences of gates and are
not suited for a simple prepare-and-measure scenario in
quantum communication applications. Indeed, in such a


_∗_ All data and source code are available online at [1].



scenario the experimenter faces careful calibration of the
measurement setup, or in other words _quantum detec-_
_tor tomography_ [6, 7, 18], which works reliably if known
probe states can be prepared [19–22].
As (imperfect) quantum tomography is a data-driven
technique, recent proposals suggest a natural benefit offered by machine learning methods. Bayesian models
were used to optimise the data collection process by
adaptive measurements in state reconstruction [8, 9, 23],
process tomography [24], Hamiltonian learning [25] and
other problems in experimental characterisation of quantum devices [26]. Neural networks were proposed to
facilitate quantum tomography in high-dimensions. In
such approaches neural networks of different architectures, such as restricted Boltzmann machines [10, 11, 27],
variational autoencoders [12] and other architectures [28]
are used for efficient state reconstruction; interestingly,
a model for tackling a more realistic scenario of mixed
quantum states has been proposed [29].
Our framework differs significantly and is based on supervised learning, specifically tailored to address SPAM
errors. Our method hence compensates for measurement
errors of the specific experimental apparatus employed,
as we demonstrate on real experimental data from highdimensional quantum states of single photons encoded in
spatial modes. The success of our approach bootstraps
the well-known _noise filtering_ class of techniques in machine learning.
_Quantum tomography._ Performing quantum state estimation implies the reconstruction of the density matrix _ρ_ of an unknown quantum state given the outcomes
of known measurements [30–32]. In general, a measurement is characterized by a set of positive operator valued
measures (POVM’s) _{_ M _a_ _}_ with index _α ∈A_ the different configurations of the experimental apparatus (set _A_ ).


Given the configuration _α_, the probability of observing
an outcome _γ_ is given:


P( _γ|α, ρ_ ) = Tr( _M_ _αγ_ _ρ_ ) _,_ (1)


where _M_ _αγ_ _∈_ M _α_ are POVM elements, i.e. positive operators satisfying the completeness relation [�] _γ_ _[M]_ _[αγ]_ [ =][ I][.]

A statistical estimator maps the set of all observed outcomes _D_ _N_ = _{γ_ _n_ _}_ _[N]_ _n_ =1 [onto an estimate of the unknown]
quantum state ˆ _ρ_ . A more general concept of quantum
process tomography stands for a protocol dealing with
estimation of an unknown quantum operation acting on
quantum states [33, 34]. Process tomography uses measurements on a set of known test states _{ρ_ _α_ _}_ to recover
the description of an unknown operation. [1]

The reconstruction procedure requires knowledge of
the measurement operators _{M_ _αγ_ _}_, as well as the test
states _{ρ_ _α_ _}_ in the case of process tomography. However,
both tend to deviate from the experimenter’s expectations due to stochastic noise and systematic errors. While
stochastic noise may to some extent be circumvented by
increasing the sample size, systematic errors are notoriously hard to correct. The only known way to make
tomography reliable is to explicitly incorporate these errors in (1). Thus, trial states and measurements should
be considered as acted upon by some SPAM processes:
_ρ_ ˜ _α_ = _R_ ( _ρ_ _α_ ) and _M_ [˜] _αγ_ = _M_ ( _M_ _αγ_ ), and the models for
these processes should be learned independently from a
calibration procedure. Such calibration is essentially tomography on its right. For example, the reconstruction
of measurement operators is known as detector tomography [6, 7, 18, 35, 36] and requires ideal preparation of
calibration states. The most straightforward approach is
calibration of the measurement setup with some closeto-ideal and easy to prepare test states, or calibration
of the preparation setup with known and close-to-ideal
measurements. In this case, one may then infer the processes _R_ and/or _M_ explicitly – for example – in the form
of the corresponding operator elements, and incorporate
this knowledge in the reconstruction procedure. Ideally,
this procedure should produce an estimator free from bias
caused by systematic SPAM errors. [2]

_Denoising by deep learning._ The problem of fighting
SPAM is essentially a denoising problem. Given the estimates of raw probabilities inferred from the experimental dataset P [˜] ( _γ|α,_ ˜ _ρ_ ) = Tr( _M_ [˜] _αγ_ ˜ _ρ_ ) _,_ one wants to establish a one-to-one correspondence with the ideal probabilities P [˜] ( _γ|α,_ ˜ _ρ_ ) _↔_ P( _γ|α, ρ_ ) for the measurement setup
free from systematic SPAM errors. We use a deep neural
network (DNN) in the form of an overcomplete autoencoder trained on a dataset _D_ _N_ to approximate the map
from P [˜] to P.


1 See Supplemental Material (Section 3) for the thorough discussion of quantum process tomography and its application for calibration of the measurement setup (Section 1).
2 See Supplemental Material (Section 3) for the detailed description of this procedure applied to our experiment (Section 2).



2


400 200


FIG. 1. The DNN architecture for an overcomplete autoencoder, employed in our simulations for denoising. Input and
output layers constitute of 36 neurons each and two hidden
layers of 400 and 200 neurons respectively. The DNN modifies its internal parameters to find a function _F_ : P [˜] ( _γ|α,_ ˜ _ρ_ ) _→_
P( _γ|α, ρ_ ) which translates between the experimentally estimated probabilities P [˜] ( _γ|α,_ ˜ _ρ_ ), subjected to SPAM errors, at
the input and ideal P( _γ|α, ρ_ ) at the output. To achieve this
goal the network is forced to reduce the Kullback-Leibler divergence amongst pairs of distributions. An early stopper is
applied in order to avoid overfitting during the training phase.


To train and test the DNN we prepare a dataset of
_N_ Haar-random pure states _D_ _N_ = _{|ψ_ _i_ _⟩}_ _[N]_ _i_ =1 [.] For a
_d_ -dimensional Hilbert space, reconstruction of a Hermitian density matrix with unit trace requires at least
_d_ [2] different measurements. The network is trained on
the dataset, consisting of _d_ [2] _× N_ frequencies experimentally obtained by performing the same _d_ [2] measurements
_{M_ [˜] _γ_ _}_ _[d]_ _γ_ [2] =1 [for all] _[ N]_ [ states (in our experiments] _[ d]_ [ = 6, i.e.]
we deal with a six-dimensional Hilbert space). These frequencies are fed to the input layer of the feed-forward
network consisting of _d_ [2] = 36 neurons. [3]

We use a DNN with two hidden layers as shown in
Fig. 1. The first hidden layer is chosen to consist of four
hundred neurons, whilst the second contains two hundred. To prevent overfitting we applied dropout between
the two hidden layers with drop probability equal to 0.2,
i.e. at each iteration we randomly drop 20% neurons of
the first hidden layer in such a way that the network
becomes more robust to variations. We use a _rectified_
_linear unit_ as an activation function after both hidden
layers, while in the final output _d_ [2] -dimensional layer we
use a _softmax_ function to transform the predicted values
to valid normalized probability distributions. Following
the standard paradigm of statistical learning, we divided
our dataset of overall _N_ = 10500 states (represented by
their density matrix elements) into 7000 states for training, 1500 states for validation and 2000 for testing. The


3 See Supplemental Material (Section 5) for DNN architecture and
the details of training process.


validation set is an independent set and is used to stop
the network training as soon as the error evaluated for
this set stops decreasing (generally, this is referred to
as early stopping: we examine validation loss every 100
epochs). Our loss function is computed over mini-batches
of data of size 40.
_Kullback-Leibler divergence._ Training is performed by
minimization of the loss function, defined as the sum of
Kullback-Leibler divergences between the distributions of
_predicted probabilities {p_ _[i]_ _γ_ _[}]_ _[d]_ _γ_ [2] =1 [at the output layer of the]
network and the ideally _expected probabilities {_ P _[i]_ _γ_ _[}]_ _[d]_ _γ_ [2] =1 [,]
which are calculated for the test states as P _[i]_ _γ_ [= Tr(] _[M]_ _[γ]_ _[ρ]_ _[i]_ [)]
assuming errorless projectors _M_ _γ_ :



�



_d_ [2]
� P _[i]_ _γ_ [log]

_γ_ =1



P _[i]_
_γ_
� _p_ _[i]_ _γ_



_N_
�


_i_ =1



3


FIG. 2. Experimental setup for preparation and measurement
of spatial qudit states. In the generation part, single photons
from a heralded source are beam-shaped by a single mode
fiber (SMF) and then transformed by a hologram displayed
on a spatial light modulator. Analogously, the detection part
consists of a hologram corresponding to the chosen detection
mode, followed by a single mode fiber and a single photon
counter. The hologram in the generation part produces highquality HG modes with the use of amplitude modulation,
while a phase-only hologram at the detection part sacrifices
projection quality for efficiency.


two quantum numbers, associated with orthogonal optical modes, and radial degree of freedom of LaguerreGaussian beams [39, 40] as well as full set of HermiteGaussian (HG) modes [41] offer viable alternatives for
increasing the accessible Hilbert space dimensionality.
One of the troubles with using the full set of orthogonal modes for encoding is the poor quality of projective
measurements. Existing methods to remedy the situation [42] trade reconstruction quality for efficiency, significantly reducing the latter. Complex high-dimensional
projectors are especially vulnerable to measurement errors and fidelities of state reconstruction are typically at
most _∼_ 0 _._ 9 in high-dimensional tomographic experiments

[43]. That provides a challenging experimental scenario
for our machine-learning-enhanced methods.
Our experiment is schematically illustrated in Fig. 2.
We use phase holograms displayed on the spatial light
modulator as spatial mode transformers. At the preparation stage an initially Gaussian beam is modulated both
in phase and in amplitude to an arbitrary superposition
of HG modes, which are chosen as the basis in the Hilbert
space. At the detection phase the beam passes through a
phase-only mode-transforming hologram and is focused
to a single mode fiber, filtering out a single Gaussian
mode. This sequence corresponds to a projective measurement in mode space, where the projector _M_ [˜] _γ_ is determined by the phase hologram. [4] In dimension _d_ = 6, we
are able to prepare an arbitrary superposition expressed


4 See Supplemental Material for the details of the experimental



_L_ =



_N_
� _D_ _KL_ ( _{_ P _[i]_ _}||{p_ _[i]_ _}_ ) =


_i_ =1



_._ (2)



The minimization of KL divergence of Eq. (2) is achieved
by virtue of gradient descent with respect to the parameters _{θ_ _k_ _}_ of the DNN for updating its internal weights.
The KL divergence for a pair (P _[i]_ _, p_ _[i]_ ) can be expressed in
terms of cross-entropy _H_ (P _[i]_ _, p_ _[i]_ ) = [�] _[d]_ _γ_ [2] =1 [P] _[i]_ _γ_ [log] _[ p]_ _[i]_ _γ_ [which]

has to be minimized. For this purpose, we utilized the
_RMSprop_ [37] algorithm, in which the learning rate is
adapted for each of the parameters _{θ_ _k_ _}_, dividing the
learning rate for a weight by a running average of the
magnitudes of recent gradients for that weight according

to


_v_ _t_ = _αv_ _t−_ 1 + (1 _−_ _α_ )( _∇_ _i_ _L_ [2] ( _{θ}_ )) (3)


where _α_ = 0 _._ 1. While the parameters are updated as



_θ_ _t_ = _θ_ _t−_ 1 _−_ _η_ (4)
~~_√_~~ _v_ ( _t_ ) _[∇]_ _[i]_ _[L]_ [(] _[{][θ][}]_ [)]



with _η_ standing for the learning rate.
_Experimental dataset._ We fix the set of tomographicaly complete measurements _{_ M _α_ _}_ = M to estimate all
matrix elements of _ρ_ using (1) and an appropriate estimator. We will assume that our POVM M consists of _d_ [2]

one-dimensional projectors _M_ _γ_ = _|ϕ_ _γ_ _⟩⟨ϕ_ _γ_ _|_ . These projectors are transformed by systematic SPAM errors into
some positive operators _M_ [˜] _γ_ . Experimental data consists
of frequencies _f_ _γ_ = _n_ _γ_ _/n_, where _n_ _γ_ is the number of
times an outcome _γ_ was observed in a series of _n_ measurements with identically prepared state _ρ_ . For the time
being, we assume, that all the SPAM errors can be attributed to the measurement part of the setup, and the
state preparation may be performed reliably. This is indeed the case in our experimental implementation (see
Supplemental Material).
We reconstruct high-dimensional quantum states encoded in the spatial degrees of freedom of photons. The
most prominent example of such encoding uses photonic
states with orbital angular momentum (OAM) [38] as relevant to numerous experiments in quantum optics and
quantum information. However, OAM is only one of


4


(a) Fidelity (b) Purity (c) Fidelity (pure)


FIG. 3. Results of experimental state reconstruction vs phase only holograms. (a) Fidelity of the experimentally reconstructed
states with ideal _F_ = _⟨ψ_ _[i]_ _|ρ_ ˆ _[i]_ ( _raw/nn_ ) _[|][ψ]_ _[i]_ _[⟩]_ [for 2000 test states reconstructed from raw data (orange bars) and reconstructed after]
neural network processing of the data (blue bars). (b) A similar diagram for purity of the reconstructed states, _π_ = Tr ˆ _ρ_ [2] .
(c) Fidelity histogram for the case, when the state is reconstructed to be pure. The results of the filtering process are clearly
witnessed by the modification of data histogram shapes. Besides the shifting towards higher values, that shows average gain
over our experimental data, the reduction of FWHM indicates filtering task by the neural network.



in the basis of HG modes as _|ψ⟩_ = [�] [2] _i,j_ =0 _[c]_ _[ij]_ _[|]_ [HG] _[ij]_ _[⟩]_ [. In]
the measurement phase we used a SIC (symmetric informationally complete) POVM, which is close to optimal
for state reconstruction and may be relatively easily realized for spatial modes [43].
_Experimental results._ We performed state reconstruction using maximum likelihood estimation [44] for both
raw experimental data and DNN-processed data. [5] In
the former case, the log-likelihood function to be maximized with respect to _ρ_ has been chosen as _L_ ( _f_ _γ_ _[i]_ _[|][ρ]_ [)] _[ ∝]_
36
� _γ_ =1 _[f]_ _γ_ _[ i]_ [log [Tr(] _[M]_ _[γ]_ _[ρ]_ [)], with frequencies] _[ f]_ _[γ]_ [=] _[ n]_ _[γ]_ _[/n]_ [ and]
_i_ numbering the test set states. Whereas in the latter
case, these frequencies have been replaced with predicted
probabilities _p_ _γ_ . The results for ˆ _ρ_ _[i]_ ( _raw_ ) [= argmax] _[ L]_ [(] _[f]_ _γ_ _[ i]_ _[|][ρ]_ [)]
and ˆ _ρ_ _[i]_ ( _nn_ ) [= argmax] _[ L]_ [(] _[p]_ _[i]_ _γ_ _[|][ρ]_ [) with the prepared states]
_|ψ_ _[i]_ _⟩_ are shown in Fig. 3. Interestingly, the average reconstruction fidelity increases from _F_ ( _raw_ ) = (0 _._ 82 _±_ 0 _._ 05)
to _F_ ( _nn_ ) = (0 _._ 91 _±_ 0 _._ 03) and this increase is uniform
over the entire test set. Similar behavior is observed
for the purity — since we did not force the state to be
pure in the reconstruction, the average purity of the estimate is less then unity: _π_ ( _raw_ ) = (0 _._ 78 _±_ 0 _._ 07), whereas
_π_ ( _nn_ ) = (0 _._ 88 _±_ 0 _._ 04). If the restriction to pure states is
explicitly imposed in the reconstruction procedure, the
fidelity increase is even more significant, as shown in
Fig. 8c. In this case the initially relatively high fidelity of
_F_ ( _raw_ ) = (0 _._ 94 _±_ 0 _._ 03) increases to _F_ ( _nn_ ) = (0 _._ 98 _±_ 0 _._ 02)


setup (Section 1) and state preparation and detection methods
(Section 2).
5 See also Supplemental Material (Section 4) for extra information
on spatial probability distribution of reconstructed states.



— a very high value, given the states dimensionality.


_Conclusion._ Our results were obtained with analytical correction for some known SPAM errors already performed. In particular, we have explicitly taken into account the Gouy phase-shifts acquired by the modes of different order during propagation (see Supplemental Material). This correction is however unnecessary for neuralnetwork post-processing. The DNN has been trained
without any need of data _preprocessing_ over the experimental dataset, as to say without introducing any phase
correction in our initial data, wherein considering the effect of a channel process _E_ . However, we have achieved
average estimation fidelities of _F_ ( _nn_ ) = (0 _._ 81 _±_ 0 _._ 19)
as compared to _F_ ( _raw_ ) = (0 _._ 54 _±_ 0 _._ 12) for this _com-_
_pletely agnostic_ scenario, showing a dramatic improvement by straightforward application of a learning approach. To conclude, our results unambiguously demonstrate that a use of neural-network-architecture on experimental data can provide a reliable tool for quantum
state-and-detector tomography.


_Acknowledgements._ The authors acknowledge financial
support under the Russian National Quantum Technologies Initiative and thank Timur Tlyachev and Dmitry
Dylov for helpful suggestions on an early version of this
study, and Yuliia Savenko (illustrator) for producing the
neural network illustration. E.K. acknowledges support
from the BASIS Foundation.


[1] Experimental data and source code. `[https://](https://github.com/Quantum-Machine-Learning-Initiative/dnnquantumtomography)`
```
  github.com/Quantum-Machine-Learning-Initiative/
```

`[dnnquantumtomography](https://github.com/Quantum-Machine-Learning-Initiative/dnnquantumtomography)` .

[2] D. Rosset, R. Ferretti-Sch¨obitz, J.-D. Bancal, N. Gisin,
and Y.-C. Liang. Imperfect measurement settings: Implications for quantum state tomography and entanglement


witnesses. _Physical Review A_, 86:062325, Dec 2012.

[3] E. Knill, D. Leibfried, R. Reichle, J. Britton, R. B.
Blakestad, J. D. Jost, C. Langer, R. Ozeri, S. Seidelin,
and D. J. Wineland. Randomized benchmarking of quantum gates. _Physical Review A_, 77:012307, Jan 2008.

[4] S. T. Merkel, J. M. Gambetta, J. A. Smolin, S. Poletto,
A. D. C´orcoles, B. R. Johnson, C. A. Ryan, and M. Steffen. Self-consistent quantum process tomography. _Phys-_
_ical Review A_, 87:062119, Jun 2013.

[5] R. Blume-Kohout, J. K. Gamble, E. Nielsen, K.
Rudinger, J. Mizrahi, K. Fortier, and P. Maunz. Demonstration of qubit operations below a rigorous fault tolerance threshold with gate set tomography. _Nature Com-_
_munications_, 8:14485, 2017.

[6] J. S. Lundeen, A. Feito, H. Coldenstrodt-Ronge, K. L.
Pregnell, C. Silberhorn, T. C. Ralph, J. Eisert, M. B.
Plenio, and I. A. Walmsley. Tomography of quantum
detectors. _Nature Physics_, 5(1):27, 2009.

[7] G. Brida, L. Ciavarella, I. P. Degiovanni, M. Genovese, A.
Migdall, M. G. Mingolla, M. G. A. Paris, F. Piacentini,
and S. V. Polyakov. Ancilla-assisted calibration of a measuring apparatus. _Physical Review Letters_, 108:253601,
Jun 2012.

[8] F. Husz´ar and N. M. T. Houlsby. Adaptive Bayesian
quantum tomography. _Physical Review A_, 85:052120,
May 2012.

[9] K. S. Kravtsov, S. S. Straupe, I. V. Radchenko, N. M. T.
Houlsby, F. Husz´ar, and S. P. Kulik. Experimental adaptive Bayesian tomography. _Physical Review A_, 87:062122,
Jun 2013.

[10] G. Torlai, G. Mazzola, J. Carrasquilla, M. Troyer, R.
Melko, and G. Carleo. Neural-network quantum state
tomography. _Nature Physics_, 14(5):447, 2018.

[11] G. Carleo, Y. Nomura, and M. Imada. Constructing exact representations of quantum many body systems with
deep neural network. _Nature Communications_, 9:5322,
2018.

[12] A. Rocchetto, E. Grant, S. Strelchuk, G. Carleo, and S.
Severini. Learning hard quantum distributions with variational autoencoders. _npj Quantum Information_, 4(1):28,
2018.

[13] E. Magesan, J. M. Gambetta, and J. Emerson. Scalable
and robust randomized benchmarking of quantum processes. _Physical Review Letters_, 106:180504, May 2011.

[14] J. J. Wallman and S. T. Flammia. Randomized benchmarking with confidence. _New Journal of Physics_,
16(10):103032, oct 2014.

[15] I. Roth, R. Kueng, S. Kimmel, Y.-K. Liu, D. Gross,
J. Eisert, and M. Kliesch. Recovering quantum gates
from few average gate fidelities. _Physical Review Letters_,
121:170502, Oct 2018.

[16] R. Blume-Kohout, J. K. Gamble, E. Nielsen, J. Mizrahi,
J. D. Sterk, and P. Maunz. Robust, self-consistent,
closed-form tomography of quantum logic gates on a
trapped ion qubit. _[arXiv preprint arXiv:1310.4492](http://arxiv.org/abs/1310.4492)_, 2013.

[17] J. P. Dehollain, J. T. Muhonen, R. Blume-Kohout, K. M.
Rudinger, J. K. Gamble, E. Nielsen, A. Laucht, S. Simmons, R. Kalra, A. S. Dzurak, and A. Morello. Optimization of a solid-state electron spin qubit using gate
set tomography. _New Journal of Physics_, 18(10):103018,
oct 2016.

[18] I. B. Bobrov, E. V. Kovlakov, A. A. Markov, S. S.
Straupe, and S. P. Kulik. Tomography of spatial mode
detectors. _Optics Express_, 23(2):649–654, Jan 2015.



5


[19] D. Mogilevtsev, J. ˇReh´aˇcek, and Z. Hradil. Selfcalibration for self-consistent tomography. _New Journal_
_of Physics_, 14(9):095001, sep 2012.

[20] A. M. Bra´nczyk, D. H. Mahler, L. A. Rozema, A. Darabi,
A. M. Steinberg, and D. F. V. James. Self-calibrating
quantum state tomography. _New Journal of Physics_,
14(8):085003, aug 2012.

[21] S. S. Straupe, D. P. Ivanov, A. A. Kalinkin, I. B. Bobrov,
S. P. Kulik, and D. Mogilevtsev. Self-calibrating tomography for angular schmidt modes in spontaneous parametric down-conversion. _Physical Review A_, 87:042109,
Apr 2013.

[22] C. Jackson and S. J. van Enk. Detecting correlated errors in state-preparation-and-measurement tomography.
_Physical Review A_, 92:042312, Oct 2015.

[23] C. Granade, C. Ferrie, and S. T. Flammia. Practical
adaptive quantum tomography. _New Journal of Physics_,
19(11):113017, 2017.

[24] I. A. Pogorelov, G. I. Struchalin, S. S. Straupe, I. V.
Radchenko, K. S. Kravtsov, and S. P. Kulik. Experimental adaptive process tomography. _Physical Review_
_A_, 95:012302, Jan 2017.

[25] C. E. Granade, C. Ferrie, N. Wiebe, and D. G. Cory.
Robust online Hamiltonian learning. _New Journal of_
_Physics_, 14(10):103013, 2012.

[26] D. T. Lennon, H. Moon, L. C. Camenzind, L. Yu, D. M.
Zumb¨uhl, G. A. D. Briggs, M. A. Osborne, E. A. Laird,
and N. Ares. Efficiently measuring a quantum device using machine learning. _[arXiv preprint arXiv:1810.10042](http://arxiv.org/abs/1810.10042)_,
2018.

[27] J. Carrasquilla, G. Torlai, R. G. Melko, and L. Aolita.
Reconstructing quantum states with generative models.
_Nature Machine Intelligence_, 1(3):155, 2019.

[28] T. Xin, S. Lu, N. Cao, G. Anikeeva, D. Lu, J. Li, G.
Long, and B. Zeng. Local-measurement-based quantum
state tomography via neural networks. _arXiv preprint_
_[arXiv:1807.07445](http://arxiv.org/abs/1807.07445)_, 2018.

[29] G. Torlai and R. G. Melko. Latent space purification
via neural density operator. _Physical Review Letters_,
120:240503, 2018.

[30] K. Banaszek, G. M. D’Ariano, M. G. A. Paris, and M. F.
Sacchi. Maximum-likelihood estimation of the density
matrix. _Physical Review A_, 61:010304, Dec 1999.

[31] D. F. V. James, P. G. Kwiat, W. J. Munro, and A. G.
White. Measurement of qubits. _Physical Review A_,
64:052312, Oct 2001.

[32] M. Paris and J. Reh´aˇcek, editors. [ˇ] _Quantum State Estima-_
_tion_, volume 649 of _Lecture Notes in Physics_ . SpringerVerlag, 2004.

[33] I. L. Chuang and M. A. Nielsen. Prescription for experimental determination of the dynamics of a quantum
black box. _Journal of Modern Optics_, 44(11-12):2455–
2467, 1997.

[34] J. F. Poyatos, J. I. Cirac, and P. Zoller. Complete characterization of a quantum process: The two-bit quantum
gate. _Physical Review Letters_, 78:390–393, Jan 1997.

[35] J. Fiur´aˇsek. Maximum-likelihood estimation of quantum
measurement. _Physical Review A_, 64:024102, Jul 2001.

[36] G. M. D’Ariano, L. Maccone, and P. Lo Presti. Quantum calibration of measurement instrumentation. _Phys-_
_ical Review Letters_, 93:250407, Dec 2004.

[37] S. Ruder. An overview on gradient descent optimization
algorithm. _[arXiv preprint arXiv:1609.04747](http://arxiv.org/abs/1609.04747)_, 2016.

[38] G. Molina-Terriza, J. P. Torres, and L. Torner. Twisted


photons. _Nature Physics_, 3(5):305, 2007.

[39] V. D. Salakhutdinov, E. R. Eliel, and W. L¨offler. Fullfield quantum correlations of spatially entangled photons.
_Physical Review Letters_, 108:173604, Apr 2012.

[40] M. Krenn, M. Huber, R. Fickler, R. Lapkiewicz, S.
Ramelow, and A. Zeilinger. Generation and confirmation of a (100 _×_ 100)-dimensional entangled quantum system. _Proceedings of the National Academy of Sciences_,
111(17):6243–6247, 2014.

[41] E. V. Kovlakov, I. B. Bobrov, S. S. Straupe, and S. P.
Kulik. Spatial bell-state generation without transverse
mode subspace postselection. _Physical Review Letters_,
118:030503, Jan 2017.

[42] F. Bouchard, N. Herrera Valencia, F. Brandt, R. Fickler,
M. Huber, and M. Malik. Measuring azimuthal and radial
modes of photons. _Optics Express_, 26(24):31925–31941,
Nov 2018.

[43] N. Bent, H. Qassim, A. A. Tahir, D. Sych, G. Leuchs,
L. L. S´anchez-Soto, E. Karimi, and R. W. Boyd. Ex


6


perimental realization of quantum tomography of photonic qudits via symmetric informationally complete positive operator-valued measures. _Physical Review X_,
5(4):041006, 2015.

[44] Z. Hradil. Quantum-state estimation. _Physical Review_
_A_, 55:R1561–R1564, Mar 1997.

[45] A. Mair, A. Vaziri, G. Weihs, and A. Zeilinger. Entanglement of the orbital angular momentum states of photons.
_Nature_, 412(6844):313, 2001.

[46] E. Bolduc, N. Bent, E. Santamato, E. Karimi, and R. W.
Boyd. Exact solution to simultaneous intensity and phase
encryption with a single phase-only hologram. _Optics_
_Letters_, 38(18):3546–3549, 2013.

[47] I. Goodfellow, Y. Bengio, and A. Courville. _Deep Learn-_
_ing_ . MIT Press, 2016.

[48] L. Prechelt. Early stopping-but when? In _Neural Net-_
_works: Tricks of the trade_, pages 55–69. Springer, 1998.



**SUPPLEMENTAL MATERIAL**


**1.** **Experimental setup**


We use spatial degrees of freedom of photons to produce high-dimensional quantum states. The corresponding
continuous Hilbert space is typically discretized using the basis of transverse modes, for this purpose we chose HermiteGaussian (HG) modes HG _nm_ ( _x, y_ ), which are the solutions of the Helmholtz equation in Cartesian coordinates ( _x, y_ )
and form a complete orthonormal basis. The HG modes are separable in _x_ - and _y_ -coordinates, so that HG _nm_ ( _x, y_ ) =
HG _n_ ( _x_ ) _×_ HG _m_ ( _y_ ). Each mode is characterized by indices _n_ and _m_ which indicate the orders of corresponding Hermite
polynomials _H_ _n_ ( _x_ ) and _H_ _m_ ( _y_ ):



HG _n_ ( _z_ ) _∝_ _H_ _n_ ( _√_



2 _z/w_ ) exp( _−z_ [2] _/w_ [2] ) _,_ (5)



where _w_ is the mode waist. We limited the dimensionality of the Hilbert space to 6 by using only the beams with
_n_ + _m ≤_ 2. The basis of HG modes is fully equivalent to a commonly used basis of Laguerre-Gaussian (LG) modes
which are also the solutions of the Helmholtz equation but in cylindrical coordinates. Most commonly, only the
azimuthal part of LG basis, associated with orbital angular momentum (OAM) of photons, is considered in the
experiments, primarily due to simplicity of detection [43]. Here we use the full two-dimensional mode spectrum of
HG modes, which is equivalent to including the radial degree of freedom in addition to OAM. This is rarely done in
quantum experiments, and one of the reasons is poor quality of projective measurements. Thus, this choice of physical
system nicely fits the purpose of our demonstration.
The experimental setup is presented in Fig. 4. Two light sources were used: an attenuated 808 nm diode laser and a
heralded single photon source. Heralded single photons were obtained from spontaneous parametric down conversion
in a 15 mm periodically-poled KTP crystal pumped by a 405 nm volume-Bragg-grating-stabilized diode laser. Beams
from both sources were filtered by a single-mode fiber (SMF) and then collimated by an aspheric lens L2 (11 mm).
We used one half (right) of the SLM (Holoeye Pluto) to generate the desired mode in the first diffraction order of the
displayed hologram. Since SLM’s working polarization is vertical, the half-wave plate (HWP) was inserted into the
optical path to let the beam pass through the polarizing beamsplitter (PBS). The combination of lenses L3 and L4
with equal focal lengths (100 mm) separated by a 200 mm distance was used to cut off the zero diffraction order with
a pinhole in the focal plane. After the double pass through this telescope and a quarter-wave plate (QWP) the beam
was reflected by the PBS and directed back to the SLM. Using the hologram displayed on the left half of the SLM
and a single mode fiber followed by a single photon counting module (SPCM) we realized a well-known technique of
projective measurements in the spatial mode space [45]. To focus the first diffraction order of the reflected beam on
the tip of the fiber we used an aspheric lens L5 with the same focal length (11 mm) as L1.
All data used for the neural network training and evaluation were taken for an attenuated laser source due to much
higher data acquisition rate. When the NN trained on the attenuated laser was applied to a dataset taken with the
heralded single photon source, the reconstruction fidelity slightly degraded — we observed _F_ ( _nn_ ) = 0 _._ 86 _±_ 0 _._ 04 vs.
_F_ ( _raw_ ) = 0 _._ 81 _±_ 0 _._ 05, while _π_ ( _nn_ ) = 0 _._ 84 _±_ 0 _._ 04 vs. _π_ ( _raw_ ) = 0 _._ 75 _±_ 0 _._ 07. The most likely reason for this is some


7


_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



FIG. 4. Experimental setup for preparation and measurement of spatial qudit states. In the generation part single photons
from a heralded source or a diode laser at the same wavelength of 810 nm are beam-shaped by a single mode fiber (SMF) and
then transformed by a hologram displayed on a spatial light modulator. The detection part analogously consists of a hologram,
corresponding to the chosen detection mode followed by a single mode fiber and a single photon counter. The hologram in the
generation part uses amplitude modulation to produce high-quality HG modes, while a phase-only hologram at the detection
part sacrifices projection quality for efficiency.


non-uniformity of the datasets caused by experimental drifts — the data for heralded single photons were taken after
some period of time. We believe, the performance may be recovered if we use heralded photons data for training as
well, using a much larger amount of data.


**2.** **State generation and detection methods**


To generate the beams with an arbitrary phase and amplitude profiles with a phase-only SLM, we calculated
hologram patterns _F_ ( _i, j_ ), which can be described as a superposition of a desired phase profile Φ( _i, j_ ) and a blazed
grating pattern with a period Λ modulated by the corresponding amplitude mask _M_ ( _i, j_ ):


_F_ ( _i, j_ ) = _A_ ( _i, j_ )Mod(Φ( _i, j_ ) + 2 _πij/_ Λ _,_ 2 _π_ ) _,_ (6)


where _i_ and _j_ are the pixel coordinates. A spatially dependent blazing function allows one to control the intensity in
the first diffraction order by changing the phase depth of the hologram. The presence of an amplitude mask _A_ ( _i, j_ )
significantly decreases the diffraction efficiency, but at the same time corrects the alterations caused by diffraction.
Bolduc et al. showed that with the modification Φ( _i, j_ ) _→_ Φ( _i, j_ ) _−_ _πA_ ( _i, j_ ) this technique guarantees accurate
conversion of the plane wave into a beam of arbitrary spatial profile [46]. This allows one to safely assume that the
preparation errors in our setup are small. Considering the states generated with amplitude modulation as ideal, we
compared the quality of detection with and without such modulation ( _A_ ( _i, j_ ) = 1 for the latter case). The result is


˜
illustrated in Fig. 5 where the experimentally measured probabilities _P_ _j_ _[i]_ [= Tr] � _M_ _j_ _|ϕ_ _i_ _⟩⟨ϕ_ _i_ _|_ � = _|⟨ϕ_ _i_ _|ϕ_ ˜ _j_ _⟩|_ [2] are shown.

The projectors _M_ _j_ = _|ϕ_ _i_ _⟩⟨ϕ_ _j_ _|_ were chosen to be elements of the SIC (symmetric informationally complete) POVM for
_d_ = 6 dimensional Hilbert space, and _M_ [˜] _j_ are their SPAM-corrupted counterparts. Thus the ratio between _P_ _j_ _[i]_ = _i_ [and]
_P_ _j_ _[i]_ = _̸_ _i_ [was expected to be close to 36. To quantify the deviation between ˜][M][ and][ M][ we used the similarity parameter]

_S_ = ( [�] _i,j_ ~~�~~ _P_ _j_ _[i]_ [P] _[i]_ _j_ [)] [2] _[/]_ [(][�] _i,j_ _[P]_ _[ i]_ _j_ � _i,j_ [P] _[i]_ _j_ [), where] _[ P]_ _[ i]_ _j_ [and][ P] _[i]_ _j_ [stand for the experimentally measured and theoretically]

expected probabilities, respectively. We found that the value of the similarity parameter decreased from 0 _._ 99 to



_̸_

_i,j_ ~~�~~



_̸_


_S_ = ( [�]



_̸_


_P_ _[i]_
_j_ [P] _[i]_ _j_ [)] [2] _[/]_ [(][�]



_̸_

_i,j_ _[P]_ _[ i]_ _j_ �


8


0 _._ 96 after switching off the amplitude modulation in the detection holograms. At the same time, the total amount
of observed counts rose from 6 _._ 2 _×_ 10 [6] to 40 _._ 9 _×_ 10 [6] due to the increased diffraction efficiency of the hologram.
This illustrates a known tradeoff between the projection measurement quality and detection efficiency. One of the
applications of the results developed here is in increasing the detection efficiency for complex measurements of spatial
states of photons without sacrificing quality.
There is a simple way to understand, why projection measurement quality is lower, than that of state preparation.
The orthogonality condition for the detection of HG mode HG _nm_ ( _x, y_ ) with a hologram corresponding to HG _n_ _′_ _m_ _′_ ( _x, y_ )
can be written as

_∞_

HG _[∗]_ _n_ _[′]_ _m_ _[′]_ [(] _[x, y]_ [)] _[ ×]_ [ HG] _[nm]_ [(] _[x, y]_ [)] _[ dxdy]_ [ =] _[ δ]_ _[n]_ _[′]_ _[n]_ _[δ]_ _[m]_ _[′]_ _[m]_ _[,]_ (7)

� _−∞_


but since the aforementioned hologram calculation method is designed for the plain wave input and not a Gaussian
beam, one has to introduce an additional Gaussian term with a waist _w_ _f_ corresponding to the detection mode waist,
which breaks the orthogonality

_∞_

HG _[∗]_ _n_ _[′]_ _m_ _[′]_ [(] _[x, y]_ [)] _[ ×]_ [ HG] _[nm]_ [(] _[x, y]_ [)] _[ ×]_ [ exp[] _[−]_ [(] _[x]_ [2] [ +] _[ y]_ [2] [)] _[/w]_ _f_ [2] []] _[ dxdy][ ̸]_ [=] _[ δ]_ _[n]_ _[′]_ _[n]_ _[δ]_ _[m]_ _[′]_ _[m]_ _[.]_ (8)

� _−∞_


One possible way to fix the problem is to increase the waist _w_ _f_ of the detection mode as in the experimental work

[42]. Unfortunately, it leads to the reduction of the detection efficiency to the level of a few percent. Thus, we
used a different approach, modifying the HG mode equation (5) used for the holograms calculation with the second
independent width parameter ˜ _w_ in the following way



HG˜ _a_ ( _z_ ) _∝_ _H_ _a_ ( _√_



2 _z/w_ ) exp( _−z_ [2] _/w_ ˜ [2] ) _,_ (9)



where the parameter ˜ _w_ was chosen to satisfy the relation


1 _/_ ˜ _w_ [2] + 1 _/w_ _f_ [2] [= 1] _[/w]_ [2] (10)


to compensate for the SMF term in (7).


**3.** **Gouy phases reconstruction by process tomography**


Importantly, as the lengths of optical paths in our setup were comparable with the Rayleigh lengths of the collimated
beams, the generated states suffered from additional Gouy phase shifts, which depend on the mode orders. In order
to avoid the related difficulties we reconstructed these phases with a standard process tomography procedure and
took them into account during state generation.
Quantum process tomography is a protocol dealing with estimation of unknown quantum operation _E_ acting on
quantum states. The most general form of such an operation in the absence of loss is a CPTP map, which can be


9


FIG. 6. Experimentally reconstructed first operator element _E_ 1 of the process _E_ associated with the spatial state evolution
between the preparation and measurement stages. The matrix elements are expressed in Hermite-Gaussian modes basis. Ideally,
it should be an identity matrix, but additional phase-shifts, known as Gouy phase-shifts were observed in our setup.


written in the following form



_ρ_ _[′]_ = _E_ ( _ρ_ ) =



_K_
� _E_ _k_ _ρE_ _k_ _[†]_ _[,]_ (11)


_k_ =1



with _K ≤_ _d_ [2] in a _d_ -dimensional state space, known as an operator-sum representation. The problem of quantum
process tomography boils down to reconstruction of the operators _{E_ _k_ _}_ given the observed outcomes of measurements
performed on some test states _ρ_ _α_ with probabilities


P( _γ|α, E_ ) = Tr( _M_ _αγ_ _E_ ( _ρ_ _α_ )) = Tr�� _[K]_ _M_ _αγ_ _E_ _k_ _ρ_ _α_ _E_ _k_ _[†]_ � _._ (12)

_k_ =1


We have reconstructed the operator elements _E_ _k_ for the process associated with the spatial state evolution between
the preparation and measurement. In this case masks with amplitude modulation were used both for state preparation

_K_
and measurement. The process _E_ ( _ρ_ ) = � _E_ _k_ _ρE_ _k_ _[†]_ [turned out to be close to a rank-one process with a single dominating]

_k_ =1

operator element _E_ 1 . As one can see from Fig. 6, the reconstructed _E_ 1 is close to a diagonal matrix with pure phases
at the diagonal. These phase-shifts are naturally interpreted as Gouy phase shifts, since they are almost equal for the
modes of a particular order _n_ + _m_ = const. The inferred Gouy phase shifts were found to be equal to 0 _._ 92 _±_ 0 _._ 02 and
1 _._ 97 _±_ 0 _._ 03 radians for mode orders of _n_ + _m_ = 1 and _n_ + _m_ = 2 correspondingly. Only when this additional phase-shifts
were taken into account, the fidelities above 0.8 were achieved without the neural-network-enhanced post-processing.
The Gouy phase-shift elimination case is a nice example of the situation where the machine-learning-based approach
helps even if the correct model of the detector is unknown to the experimenter. Indeed, instead of performing the full
process reconstruction to find out the relevant phase-shifts for the modes of different order, one may stay agnostic
of these shifts and consider them as just another contribution to systematic SPAM errors. We have tested the
performance of the neural network trained on the states for which no correction of the Gouy phase shifts were made.
The impression of how dramatically these phase-shifts affect measurement, we show the crosstalk probabilities forthe projectors of the SIC POVM, with no phase correction, i.e. modified as _M_ ˜ _j_ = _E_ 1 _M_ _j_ _E_ 1 _[†]_ [in Fig.][ 7a][. Without]
phase correction the state reconstruction of the 2000 test states gives the average fidelity of _F_ ( _raw_ ) = 0 _._ 54 _±_ 0 _._ 12
only and the average purity of the estimate _π_ ( _raw_ ) = (0 _._ 77 _±_ 0 _._ 07). When the state is reconstructed as a pure one,
the value of average fidelity increases to _F_ [˜] ( _raw_ ) = 0 _._ 60 _±_ 0 _._ 13. At the same time DNN-trained on the same dataset
without any information about the Gouy phase-shifts gives the corresponding fidelities of _F_ ˜ ( _nn_ ) = 0 _._ 89 _±_ 0 _._ 22 (see Fig. 7b and Fig. 7c). _F_ ( _nn_ ) = 0 _._ 81 _±_ 0 _._ 19 and


10


500


400


300


200


100


0
0.0 0.2 0.4 0.6 0.8 1.0


(c) Fidelity (pure)





0.0 0.2 0.4 0.6 0.8 1.0


(b) Fidelity



FIG. 7. (a) Experimentally measured cross-talk probabilities _P_ _j_ _[i]_ [=] _[ |⟨][ϕ]_ _[i]_ _[|][ϕ]_ [ ˜] _[j]_ _[⟩|]_ [2] [ for the projectors from the SIC POVM without the]
Gouy phase correction. (b) Fidelity of the experimentally reconstructed states with the ideal for 2000 test states reconstructed
from the raw data (orange bars) and reconstructed after the neural network processing of the data (blue bars) without Gouy
phase correction. (c) Fidelity histogram for the case of no phase correction when the states are reconstructed as pure ones.


**4.** **State Reconstruction**


(a) Raw reconstruction (b) NN-enhanced reconstruction (c) Prepared state


FIG. 8. Spatial probability distributions for the exemplary reconstructed state _|ψ⟩_ = (0 _._ 04 _−_ 0 _._ 21 _i_ ) _|_ HG 00 _⟩_ +(0 _._ 07+0 _._ 17 _i_ ) _|_ HG 01 _⟩_ +
( _−_ 0 _._ 14 + 0 _._ 29 _i_ ) _|_ HG 10 _⟩_ + (0 _._ 21 _−_ 0 _._ 05 _i_ ) _|_ HG 02 _⟩_ + (0 _._ 09 + 0 _._ 02 _i_ ) _|_ HG 11 _⟩_ + (0 _._ 68 _−_ 0 _._ 55 _i_ ) _|_ HG 20 _⟩_ . (a) Reconstruction from the raw
data, (b) reconstruction from the predicted probabilities, (c) the prepared state. Fidelities with the prepared states are
_F_ ( _raw_ ) = (0 _._ 75 _±_ 0 _._ 02) and _F_ ( _nn_ ) = 0 _._ 91, respectively.


**5.** **Neural Network**


Throughout the paper we consider a feed-forward neural network [47] with two hidden layers of 400 and 200 neurons
respectively, which maps input probabilities to the ideal ones and can be regarded as an autoencoder. To prevent
overfitting in our model we use dropout between the two hidden layers with drop probability equal to 0.2; this means
that at each iteration we randomly drop 20% of the neurons of the first hidden layer in such a way that the network
becomes more robust to variations in the input data. After both hidden layers we use the Rectified Linear Unit
(ReLU) as activation function, while in the final output layer of 36 dimensions we use a softmax function to transform
predicted values in probabilities. The network is trained considering the Kullback-Leibler divergence (KLD) between
predicted values and the real target probabilities. Thus, we aim at minimizing the distance between the predicted
distribution and the objective one according to



�



_N_
� _D_ _KL_ ( _{_ P _[i]_ _}||{p_ _[i]_ _}_ ) =


_i_ =1



_N_
�


_i_ =1



_d_ [2]
� P _[i]_ _γ_ [log]

_γ_ =1



P _[i]_
_γ_
� _p_ _[i]_ _γ_



_._ (13)



In the following, we address the performance of DNN with respect to varying the size of the dataset, constituted by
10500 states, and the performance for the reconstruction task.


11


FIG. 9. Dependence of the KLD loss function on the training set size (upper graph) and learning rate of the neural network
quantified by classical fidelity or Bhattacharyya distance (lower graph). As expected, the quality of the prediction offered by
our model increases with increasing size of the training set.


At each iteration we select 2000 states from the dataset that we consider as testing samples. Then, from the
remaining dataset of _K_ = 8500 states we sample a percentage of data in the range from _η_ = 0 _._ 1 to _η_ = 1 with steps of
0.1 (i.e., 10% of the data, or 850 samples). We train the network over 200 epochs and compute both the loss function
and Bhattacharyya distance (classical fidelity)



_F_ _c_ ( _η_ ) =



_ηK_
�


_i_ =1



�



P _[i]_ _γ_ _p_ _[i]_ _γ_ (14)



on the test data sample. We do this 5 times and average the results to report a stable value as shown in Fig. 9.
Interestingly, little data and few epochs are necessary to learn how to generate probabilities that are close to the ideal
ones. Fidelity for _η_ = 0 _._ 1, i.e. training set consisting of 10% of the data is already equal to 0 _._ 9720.
The experimental setting used to obtain the result described in the paper is as follows: we divide our initial dataset
into three subsets, namely training, validation and testing set, which is in line with commonly accepted ratio of 80%
(or 60%) for training, 10% (or 20%) for validation and 10% (or 20%) for testing. We hereby approach the problem
considering roughly 20% of the dataset (2000 samples) for testing and we use 15% as validation (1500 samples) and
7500 samples for training. This division ensures that there is enough data to train the network and that we can test
on a sample that is almost 20% of the original data.
The training set is used to train the model while the validation set is an independent set to stop the network
training as soon as the error no longer decreases on the validation set. This technique is generally referred to as _early_
_stopping_ [48], we stop training if the error does not decrease within 100 epochs. At the end of the early stopping we
restore the weights of the network that had the best validation loss during training. Finally, we test the model on
the test set. This last step allows us to have a more unbiased estimation about the value of the loss on a completely
unseen set of data, since the model has been chosen using a validation set and it is biased on this set.


