**Incorporating long-range physics in atomic-scale machine learning**


Andrea Grisafi and Michele Ceriotti
_Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne, Switzerland_ _´_ _Laboratory of Computational Science and Modeling, IMX,_


The most successful and popular machine learning models of atomic-scale properties derive their
transferability from a locality ansatz. The properties of a large molecule or a bulk material are
written as a sum over contributions that depend on the configurations within finite atom-centered
environments. The obvious downside of this approach is that it cannot capture non-local, non-additive
effects such as those arising due to long-range electrostatics or quantum interference. We propose a
solution to this problem by introducing non-local representations of the system that are remapped
as feature vectors that are defined locally and are equivariant in _O_ (3). We consider in particular one
form that has the same asymptotic behavior as the electrostatic potential. We demonstrate that this
framework can capture non-local, long-range physics by building a model for the electrostatic energy
of randomly distributed point-charges, for the unrelaxed binding curves of charged organic molecular
dimers, and for the electronic dielectric response of liquid water. By combining a representation of
the system that is sensitive to long-range correlations with the transferability of an atom-centered
additive model, this method outperforms current state-of-the-art machine-learning schemes, and
provides a conceptual framework to incorporate non-local physics into atomistic machine learning.



**INTRODUCTION**


In recent years, atomistic machine learning models
have become increasingly popular as a way to perform
fast predictions of molecular and material properties with
the accuracy of first-principle quantum mechanical calculations [ 1 ], but a much reduced cost. The success of
these methods has gone hand-in-hand with the progress in
constructing representations for molecular and materials
configurations that are flexible enough to be transferred
across a wide spectrum of different atomic arrangements,
while satisfying, at the same time, stringent symmetry
constraints [2–6].


At the core of the vast majority of transferable machinelearning model for physical properties lies the local nature
of the underlying atomistic representation. This is usually
constructed by considering the set of atomic coordinates
that are included within spherical environments of a given
radial cutoff around any arbitrary atomic center [ 7 – 9 ].
The prediction of a given physical property is therefore
formally decomposed in the sum of atom-centered contributions that effectively incorporate information associated
with many-body structural correlations between atoms
in each local environment. This locality assumption is
very convenient, as it keeps at bay the dimensionality
of the regression problem that is modelled by ML, and
is physically justified by the nearsigthedness principle
of electronic matter [ 10 ]. The major drawback is that
it neglects long-range physical effects. Long-range electrostatic interactions, for example, are known to play a
fundamental role in the description of ionic systems [ 11 ],
macroscopically polarized interfaces [ 12 ], electrode surfaces [ 13 ] and nano-science in general [ 14 ]. In all these
cases, the pathologically slow decay _∼_ 1 _/r_ of the Coulomb
interaction makes it virtually impossible to reach convergence while using a local machine-learning scheme, which



is reflected in an effective limit to the accuracy that can
be reached by these models.


The problem of incorporating long-range effects in electronic energy predictions is usually tackled by explicitly
separating the local many-body contribution to the total
energy from a classical electrostatic term approximated
via pairwise Coulomb interactions. This can be done
either by direct subtraction of the Ewald-like electrostatic
energy of the system [ 15, 16 ], or by machine learning, in
turn, the partial charges and the atomic multipoles that
determine the long-range electrostatics [ 17 – 22 ]. Other
more sophisticated models, specifically designed for ionic
systems, rely on a charge equilibration scheme [23, 24].


Beyond electrostatic energies, the breakdown of a local
machine learning model is particularly pronounced when
dealing with intrinsically non-local quantities like the dielectric response of a condensed-phase medium [ 6 ]. This
non-locality has to do both with the effect of the far-field
electrostatics [ 25 ], and to the topological quantum nature
of the macroscopic polarization of an infinitely extended
material [ 26, 27 ]. In this case, the problem can possibly
be bypassed by adopting specific physical prescriptions.
Examples of this can be found in Ref. [ 6 ], where the dielectric tensor _**ε**_ _∞_ of liquid water is learned indirectly by
building a model for an effective molecular polarizability that is mapped to _**ε**_ _∞_ through the Clausius-Mossotti
relationship [ 25 ]. In the context of reproducing the autocorrelation function of the macroscopic polarization of
liquid water, another strategy has recently been adopted,
where the selected learning targets are the positions of
the Wannier centers that are used to recast the electron
density of the system into a set of point-charges [28].


By and large, the learning models previously described
tackle the problem of including long-range phenomena
by making use of an _ad hoc_ definition of the electrostatic
energy, or dielectric response, in terms of local atomic


quantities. Although successful, these kind of approaches
have the downside of being very system dependent and,
as such, hardly transferable across systems that have a
different nature, e.g., those related to charge transfer, or
to charge polarizability in (near)-metallic systems [ 29 ].
Capturing long-range effects without any prior assumption on the nature of the learning target is a difficult task
to accomplish with the methods currently available. Most
of the approaches that have explicitly attempted to do
so, such as Coulomb kernels [ 30 ], many-body tensor representations [ 31 ], or multi-scale invariants [ 32 ], are built
upon a global representation of the system rather than
on an additive atom-centred model.

Here we propose a simple, yet elegant, solution to this
problem, where the non-local character of the target property is incorporated in a symmetry-equivariant fashion
into an atom-centered representation. In doing so, we
construct a formalism that ensures that the resulting features exhibit the correct asymptotic dependence on the
distribution of atoms in the far-field. This representation
can be incorporated straightforwardly into conventional,
additive machine learning models. While the idea is very
general, we present as an example a model that has an
asymptotic behavior consistent with electrostatic interactions. We show that it can be used successfully to build a
local machine learning model that accurately reproduces
Coulomb interactions between point particles, the binding
curves of charged organic fragments, and the electronic
dielectric response of bulk water.


**LONG-DISTANCE EQUIVARIANT**
**REPRESENTATION**


Let us start from the same formal definition of a ML
representation of a structure _A_ that was introduced in
Ref. [ 33 ], written in the position basis as a decorated
atom density


_⟨_ **r** _|A⟩_ = � _g_ ( **r** _−_ **r** _i_ ) _|α_ _i_ _⟩_ _,_ (1)


_i_


where the index _i_ runs over all the atoms in the structure,
_g_ is a Gaussian (or another localized function) peaked
at each atom’s position **r** _i_, and _|α_ _i_ _⟩_ is an abstract vector
that encodes the chemical nature of the atom. We now

introduce an atom-density potential representation



2


asymptotically as _|_ **r** _−_ **r** _i_ _|_ _[−][p]_ [ 34 ]. The physical significance
of _|V_ _[p]_ _⟩_ is obvious, if one considers typical forms of the
interactions between atoms and molecules. For instance,
if we had a single species and interpreted (1) as a charge
1
density, � **r** �� _V_ � would correspond to the electrostatic potential generated by such charge density. Analogously, the
_p_ = 6 case would provide the formally correct asymptotic
limit of the energy per particle associated with dispersion
interactions [ 35 ], which has inspired previous representations of local environmets such as aSLATM [36].
Proceeding as in Ref. 33, we can symmetrize the representation over the continuous translation group, taking a
tensor product with the density representation to preserve
structural information. One obtains the symmetrized ket


_⟨_ **r** _|AV_ _[p]_ _⟩_ _t_ ˆ = d _t_ [ˆ] _⟨_ **0** _|_ _t_ [ˆ] _|A⟩⟨_ **r** _|_ _t_ [ˆ] _|V_ _[p]_ _⟩_ = � _|α_ _j_ _⟩_ � **r** �� _V_ _jp_ � _,_
�

_j_

(3)
where we introduced the shorthand notation (see the SI
for a full derivation)




[(] _[g][ ⋆]_ _[g]_ [)(] **[r]** _[′]_ _[ −]_ [(] **[r]** _[i]_ _[ −]_ **[r]** _[j]_ [))]
_i_ _|α_ _i_ _⟩_ � d **r** _[′]_ _|_ **r** _[′]_ _−_ **r** _|_ ~~_[p]_~~



� **r** �� _V_ _jp_ � = �



_|_ **r** _[′]_ _−_ **r** _|_ ~~_[p]_~~ _._ (4)



_i_ _|α_ _i_ _⟩_ � d **r** _[′]_ _[ g]_ _|_ [(] **r** **[r]** _[′]_ _[′]_ _−_ _[ −]_ **r** **[r]** _|_ ~~_[p]_~~ _[i]_ [)]



_⟨_ **r** _|V_ _[p]_ _⟩_ = �



_|_ **r** _[′]_ _−_ **r** _|_ ~~_[p]_~~ _[.]_ (2)



Modulo the re-definition of the atom density function
_p_
as the auto-correlation of _g_, � **r** �� _V_ _j_ � is just the atomdensity potential (2) computed using **r** _j_ as the origin of
the reference frame.

Symmetrization over the translation group leads naturally to a structural representation that amounts to a sum
over atom-centred descriptors – foreshadowing an additive
property model built on such feature vector. Particularly
for low values of the potential exponent _p_, however, the
integral in Eq. (4) introduces a substantially non-local
_p_
behavior. The value of � **r** �� _V_ _j_ � in the vicinity of the central atom _j_ can in principle depend on the position of
atoms that are very far from it, _even if one introduces a_
_p_
_cutoff function that restricts the range of_ � **r** �� _V_ _j_ � _around_
_the central atom, and hence its complexity_ . One can then
symmetrize further over the rotation group and over inversion symmetry. We will refer from now on to the resulting
class of atomistic representations that capture long-range
interactions based on the local value of an atom-density
potential as the _long-distance equivariant_ (LODE) framework. In the following we will focus on the case of _p_ = 1,
that corresponds to electrostatic interactions.
It is instructive to first consider the case of the first
order spherical invariant, and to take the limit in which
the atom density is represented by Dirac- _δ_ distributions.
It is easy to see that in this limit



The rationale for performing this transformation (that can
be seen as the action of a linear integral operator on _|A⟩_ )
is that, whereas _⟨_ **r** _|A⟩_ contains information only about
the atoms in the vicinity of **r**, _⟨_ **r** _|V_ _[p]_ _⟩_ contains information
about the position of _all_ atoms in the structure, with a
dependence on the position of the _i_ -th atom that decays



where **r** _ij_ = **r** _i_ _−_ **r** _j_ . Integrating over the SO(3) group



� _α_ **r** �� _V_ _j_ 1 � = �


_i∈α_



1
(5)
_|_ **r** _−_ **r** _ij_ _|_ _[,]_


yields the first invariant



1

� _i∈α_ min � _r_



� _αr_ ��� _V_ _j_ 1(1) [�] = � d _R_ [ˆ] _⟨αr_ ˆ **r** _|_ _R_ [ˆ] �� _V_ _j_ 1 � = �



_r_ _ij_



1 [1]

_r_ _[,]_ _r_



_,_
�



(6)
that simply sums up 1 _/r_ _ij_ terms for all atoms _outside_ the
region over which the LODE representation is computed.
Ignoring the contribution from the atoms within the cutoff,
that can be better characterized by other atomic structure
representations, a linear model built on these features is
equivalent to a fixed point-charge electrostatic model.
In other words, in this limit the radial dependence of
the regression weights is integrated out, and the weights
associated with each pair of central atom type _α_ _[′]_ and
neighbor type _α_ corresponds to the product of the atomic
charges _q_ _α_ _[′]_ and _q_ _α_ .
While this construction is very revealing, it is clear that
its descriptive power is limited. Non-linear kernel models
can provide a more flexible functional form, but higherorder invariants provide a systematic way of incorporating
more information on structural features. As in the SOAP
framework for the atom density [ 3, 33, 37 ], the most convenient way to compute such invariants involves writing
the scalar field associated with the species _α_ on a basis
of radial functions _R_ _n_ ( _r_ ) and spherical harmonics _Y_ _m_ _[l]_ [(] [ˆ] **[r]** [),]


� _αnlm_ �� _V_ _jp_ � = d **r** _R_ _n_ ( _r_ ) _Y_ _m_ _[l]_ [(ˆ] **[r]** [)] _[⋆]_ [�] _α_ **r** �� _V_ _jp_ � (7)
�


and then computing the appropriate spherically-covariant
combinations. For example, for rotationally invariant
representations of order _ν_ = 2 (the form that is equivalent
to the SOAP power spectrum and that we will use in
applications)



_αnα_ _[′]_ _n_ _[′]_ _l_ _V_ _p_ (2) [�] =
� ��� _j_ �

_|m|≤l_



� _αnlm_ �� _V_ _jp_ � _⋆_ � _α_ _[′]_ _n_ _[′]_ _lm_ �� _V_ _jp_ �



~~_√_~~



2 _l_ + 1 _._



(8)
The extension to higher orders in spatial correlations _ν >_ 2
and/or to rotationally covariant representations of a given
spherical-tensor order _λ >_ 0 is straightforward based on
the analogous density-based counterparts [ 6, 33, 38 ]. Note
that it is also possible to compute representations that
combine different values of _p_, and even _p_ = 0, corresponding to the atom-density field. A systematic investigations
of the various combinations, and their physical meaning,
is left for future work.


**Efficient evaluation of the LODE representation**


As discussed in the SI, for molecules and clusters the
expansion (7) can be computed conveniently in real space,
by numerical integration on appropriate atom-centred
grids. For a bulk system, described by a periodicallyrepeated supercell, the long-range nature of the integral



3


kernel that appears in (4) would make computing the
expansion prohibitive. This is exactly the same problem
one faces when evaluating electrostatic interactions in the
condensed phase, and fortunately it has long been solved,
e.g., with the many techniques based on the use of a planewaves auxiliary basis [ 39, 40 ]. Consider the plane-wave
definition as _⟨_ **r** _|_ **k** _⟩_ = _e_ [i] **[k]** _[·]_ **[r]**, with _{_ **k** _}_ representing a set of
wave-vectors that are compatible with the simulation box.
The fact we start from a smooth, Gaussian atom density,
means that in practice one needs only a manageable
number of plane waves. In particular, the width _σ_ of the
Gaussian density determines the minimum wavelength
that should be introduced in the the plane-wave expansion,
so that **k** -vectors only need to be generated within a
sphere of radius _k_ max of the order of 2 _π/σ_ . In order to
evaluate the local potential projections, it is then enough
to include the identity resolution [�] **k** _[|]_ **[k]** _[⟩⟨]_ **[k]** _[|]_ [ within the]

braket of Eq. (7), i.e.

� _αnlm_ �� _V_ _jp_ � = � _⟨nlm|_ **k** _⟩_ � _α_ **k** �� _V_ _jp_ � _._ (9)


**k**


As detailed in the SI, _⟨nlm|_ **k** _⟩_ corresponds to the expansion in plane waves of the basis of the local environment
representation, and can be computed analytically once
and for all if the radial functions are taken to be Gaussian
type orbitals [ 41 ]. Conversely, � _α_ **k** �� _V_ _jp_ � represents the
Fourier components of the potential generated by the
Gaussian density of element _α_ for the entire system, and
can be readily computed analytically [ 42 ]. As a result,
the geometric local nature of the representation of Eq. (9)
is formally factorized from its system-dependent global
character. It has not escaped our attention that Eqn. (9)
could also be used to compute efficiently the coefficients of
the density expansion that enter, for instance, the SOAP
framework. In the context of electrostatic interactions,
one should note that although the fictitious charge density
distribution of Eq. (1) does not satisfy charge neutrality,
one can avoid a divergence of the potential by ignoring
the **k** = **0** component from the sum of Eq. (9) . Similarly,
divergences in the potential for _p >_ 1 can be eliminated
by appropriately regularizing the 1 _/r_ _[p]_ divergence in reciprocal space.


**RESULTS**


We now proceed to test the performance of the LODE
representation in the context of predicting scalar electrostatic properties. In all cases we use Gaussian process
regression using simple polynomial kernels, to emphasize
the role of the features - as opposed to the regression
scheme - on the performance of the model. Details of
the parameters used in each example are reported in
the SI. We use the SOAP framework as the baseline

for a comparison, which is appropriate given the close
relation between the two approaches, and the excellent


4


reaches an accuracy of about 20% RMSE when using the
maximum number of training structures. A linear model
built using the LODE( _ν_ = 1) representation, on the
other hand yields an error below 1% by using a handful of
training points. As discussed above, this model represents
exactly Coulomb interactions between fixed point charges,
and the only reason the error does not converge to zero is
the fact we use a Gaussian smearing in the definition of
LODE, rather than _δ_ distributions. This is apparent in the
dramatic reduction of the error when halving the value of
_σ_ . A LODE( _ν_ = 2) model, although initially less effective,
possesses sufficient descriptive power to reach, and then
overcome, the accuracy of the linear _ν_ = 1 _, σ_ = 1 [˚] A
model. This simple example highlights how difficult it
is to incorporate long-range physics with a conventional
local structure representation, and demonstrates that the
LODE features can, on their own, be used as a very
efficient description to predict the electrostatic energy of
a system of fixed point charges.


**Binding curves of charged dimers**


We now consider a more realistic scenario, namely the
problem of predicting the binding curves of a dataset of
organic molecular dimers that carry an electric charge.
We extract 661 different dimers containing H, C, N and
O atoms from the BioFragment Database (BFDb) [ 44 ],
where at least one of the two monomers in each dimer

configuration has a net charge. This choice ensures that
we focus the exercise on a problem for which permanent
electrostatic interactions play a prominent role. Contrary
to the NaCl toy system, however, one cannot expect that
a fixed point-charge model would suffice to predict the
binding curves. The dataset contains a multitude of chemical moieties, including neutral polar fragments, highly
polarizable groups, and provides a realistic assessment of
how well a LODE model can perform in practice. For
each of the 661 dimers, we consider 13 configurations
where the reciprocal distance between the two monomers,
defined as the distance between their geometric centers,
spans an interval that can go from a minimum of _∼_ 3 [˚] A to
a maximum of _∼_ 8 [˚] A. For each of these configurations, unrelaxed binding curves are computed at the DFT/B3LYP
level using the FHI-aims quantum-chemistry package [ 45 ].
The training dataset is defined by considering the binding curves of the first 600 dimers out of the total of 661,
while predictions are tested on the remaining 61. We
also include the isolated monomers in the training set,
so that the ML model has knowledge of the dissociation
limit, and compute a few additional reference energies
at larger separations, which are however not used for
training. SOAP and LODE representations are defined
within spherical environments of _r_ cut = 3 _._ 0 [˚] A, while the
Gaussian width of the density field is chosen to be _σ_ =0.3
and 1.0 [˚] A respectively.



FIG. 1. Learning curves for the electrostatic energy of an
idealized random gas of point charges. The model is trained
on 1500 randomly selected configurations and tested on other
500 independent configurations. ( _black full and dashed lines_ )
Local ML (SOAP) results at environment cutoffs of 3, 6 and
9 [˚] A. ( _red lines_ ) LODE( _ν_ = 1) results at an environment
cutoff of 2 [˚] A and Gaussian smearing of 0.5 and 1.0 [˚] A, and
LODE( _ν_ = 2) results with a cutoff of 3A. [˚]


performances demonstrated by SOAP-based models. It is
important however to stress that _any_ local model with
a finite cutoff will exhibit similar behavior as what we
observe with SOAP. We also benchmark the combination

of SOAP and LODE, that incorporates the advantages of
both short-range and long-range models, realizing a kind
of range-separated machine learning framework.


**A gas of point charges**


We begin by considering a toy system made of randomly distributed point-charges in a cubic box that is
infinitely repeated in the three dimensions using periodic
boundary conditions. The number of positive charges
is equal to the number of negative charges, so that the
system is overall neutral. To limit the amplitude of energy fluctuations, we discard configurations in which two
charges are closer together than 2.5 [˚] A. Following these
prescriptions, we generate a total of 2000 configurations,
each of which contains 64 atoms in cubic boxes spanning
a broad range of densities, with side lengths between 12
and 20 [˚] A. For each of these configurations, we compute
the electrostatic energy using the Ewald method, as implemented in LAMMPS [ 43 ]. Fig. 1 compares the learning
performance obtained using a local SOAP representation
with different cutoffs, to the one obtained by direct application of the LODE representation. In both cases, a
Gaussian width of _σ_ =1.0 [˚] A has been used to construct
the density distribution of Eq. (1).
The figure clearly demonstrate the inefficiency of a
local model when attempting to learn a property that is
dominated by long-range effects. Given that the training
set contains few configurations with atoms closer than 3 [˚] A,
the model with _r_ cut =3 [˚] A is almost completely ineffective.
Even increasing the cutoff up to 9 [˚] A, a SOAP model barely


5



DFT local SOAP + LODE(1) SOAP + LODE(2)



a) b) 0.00 c)



0.2



b)



0.00







0.2


0.1


0.0


d)

0.10


0.05


0.00





0.05


0.10



10 [1]



R [ _Å_ ]





R [ _Å_ ]





10 [1]



e) 0.00 f)


0.05



0.1


0.0


0.15


0.10


0.05


0.00









0.05

10 [1] R [ _Å_ ]



0.10


0.15


0.20



10 [1]



R [ _Å_ ]



R [ _Å_ ]



10 [1]



FIG. 2. Comparison of reference and predicted binding curves of six molecular dimers. ( _black dots_ ) DFT reference calculations,
( _red lines_ ) local SOAP predictions, ( _green lines_ ) combined SOAP and LODE(1) predictions, ( _blue lines_ ) combined SOAP and
LODE(2) predictions. Full lines and shaded background represent the range of distances that is comparable to the geometries
included in the training set. Dashed lines refer to predictions carried out in an extrapolative (long-range) regime.



Before carrying out the learning exercise, the reference
DFT energies are baselined with respect to the monomer
energies, so that the model only has to reproduce the
interaction energies between the two fragments. Upon
this baselining, we find that optimal SOAP performances
correspond to a RMSE _∼_ 20%, whereas a suitable combination between SOAP and LODE( _ν_ = 2) allows us to
bring the error down to _∼_ 4%. This substantial improvement can be justified by the large discrepancy between
the SOAP and SOAP+LODE accuracy in representing
the interaction between the monomers at intermediate

and large distance. To clarify the issue further, we plot
in Fig. 2 the predicted binding curves of 6 test dimers,
against the reference DFT calculations. We observe that
a SOAP-based local description is overall able to capture
the short-range interactions with good accuracy. However,
it becomes less and less effective as the distance between
the monomers increases, to the point of being completely
blind to changes in interatomic distances when the environments cutoff distance is overcome. Note that the
performance of the local model at small separations is
degraded substantially by the inclusion of fully dissociated dimers in the training set, because the representation
cannot distinguish these configurations from those barely
beyond the cutoff distance, that correspond to a non-zero
value of the binding curve. The SOAP+LODE multiscale description, in contrast, can recognize the changes
in separation between the monomers, leading to a smooth
asymptotic behavior of the predicted binding curve. Although a linear model incorporating LODE( _ν_ = 1) allows
us to halve the error made by SOAP down to _∼_ 10%, it
is not sufficiently expressive to achieve predictive accuracy - particularly for binding curves that involve neutral



monomers that do not have a 1 _/r_ asymptotic behavior.
This limitation can be addressed using a non-linear
kernel based on SOAP+LODE( _ν_ = 2). The resulting
model is able to accurately predict the binding curves in
the entire domain of distances, demonstrating its transferability across a vast spectrum of different chemical species
and intermolecular configurations. This is particularly
remarkable, as the SOAP+LODE( _ν_ = 2) model does not
only predict accurately systems that are dominated by
monopole electrostatics (Fig.3-(a,b,c,e)), but also systems
in which only one of the molecules is charged, and so
interactions involve polarization as well as charge-dipole
electrostatics (Fig.3-(d,f)). It should be noted, however,
that the current scheme cannot transparently describe
the physics of polarization or charge transfer. While the
use of a composite SOAP+LODE kernel can describe how
the environment of an atom affects its response to an
external field, there is no explicit provision to represent
how the field generated by far-away atoms depends on
their neighboring structure.


**Dielectric response of liquid water**


As a final example, we revisit the problem of constructing a model of the infinite-frequency dielectric response
tensor _**ε**_ _∞_ of liquid water. Details about the dataset
generation and the computation of the dielectric tensors
are reported in Ref. [ 6 ]. In that work, we argued that a
local model was inefficient in learning dielectric response
because of its collective nature, and showed that using the
Clausius-Mossotti relationship to map _**ε**_ _∞_ to more local
quantities was greatly improving the model. Here, LODE


10 1

local r _cut_ =3 _Å_


local r _cut_ =4 _Å_


local r _cut_ =6 _Å_


LODE r _cut_ =3 _Å_


SOAP + LODE


10 2


10 [1] 10 [2] 10 [3]


training structures


FIG. 3. Learning curves for the isotropic component of the
dielectric response tensor _**ε**_ _∞_ of liquid water. The model is
trained on up to 800 randomly selected configurations and
tested on other 200 independent configurations. ( _black full and_
_dashed lines_ ) SOAP results with _r_ cut = 3, 4 and 6 [˚] A. ( _red line_ )
LODE results with _r_ cut =3 [˚] A. ( _blue line_ ) combined results of
SOAP and LODE, both using _r_ cut =3 A. [˚]


learning performances are only tested for the isotropic
component of the tensor _ε_ 0 = Tr [ _**ε**_ _∞_ ], which was shown
to be most sensitive to the collective nature of the physics
of dielectrics. Similarly to the case of th BFDb, we use a
non-linear kernel that combines a SOAP representations
computed using an optimal Gaussian width of _σ_ =0.3 [˚] A,
and LODE( _ν_ = 2) features constructed starting from a
Gaussian density of _σ_ =1.0 [˚] A. Figure 3 reports results obtained when learning on 800 randomly selected structures
and predicting on other 200 independent configurations.


Similarly to what has been observed in the previous
example, LODE performs much better than SOAP when
relying upon a local description of _r_ cut =3 [˚] A. In this case,
however, we observe a substantial improvement of the
performance of SOAP when increasing the size of the
local environments, eventually overcoming the LODE accuracy with a radial cutoff of _r_ cut =6 [˚] A. This might be
a consequence of a less pronounced contribution of longrange tails, or - likely - of the fact that a cutoff of 6 [˚] A
encompasses the entirety of the supercell, and therefore
effectively provides a complete description of the input
space of this specific dataset. Optimal ML predictions
can be obtained when combining the fine-grained local
description of SOAP at _r_ cut =3 [˚] A with the coarse-grained
and non-local description of LODE at the same cutoff.
This behaviour highlights the multiscale character of _ε_ 0,
meaning that both the local many-body information and
the long-range electrostatic effects need to be considered
to get accurate predictions. It is also important to stress
that a combination of SOAP and LODE is not only beneficial in terms of learning performance, but can also reduce
the computational effort in evaluating the feature vector –
much like efficient methods for evaluating empirical potentials often treat separately short-range and long-range
interactions.



6


**CONCLUSIONS**


Machine-learning of atomic-scale properties that are
dominated by short-range interactions has reached a stage
of maturity, with a substantial consensus about the ingredients of a successful model. The most commonly
used frameworks incorporate symmetries and physical
principles into the representation of atomic configurations, and achieve transferability by building additive
property models. Furthermore, there is a growing understanding of the deep connections that exist between many
of these methods, which is reflected in the fact that in
most applications they reach similar levels of accuracy.
In this paper we show how to extend these schemes in
a way that makes it possible to incorporate long-range
physics, without sacrificing the transferability of additive
property models and the general applicability of rather
abstract measures of atomic structure correlations. The

crux lies in the definition of an atom-density potential
that folds global information on the structure and composition of a system into a local representation, that (1) has
a physically-motivated asymptotic behavior with interatomic separation and (2) can be efficiently computed
in a symmetry-consistent fashion using similar ideas as
those that underlie the SOAP framework and related

approaches.


We apply this long-distance equivariant (LODE) representation focusing on the version that is based on a
Coulomb-like atom-density potential. We demonstrate
that, alone or in combination with SOAP, it outperforms
local machine-learning methods in capturing long-range
physics, for tasks that involve learning the electrostatic
energy of a point-charge model, the binding curve of
dimers of electrically charged organic fragments, and the
dielectric constant of bulk water.


These examples are little more than an assay that proves
that this scheme can incorporate efficiently long-range
information in atomistic machine learning. More work is
needed to draw a systematic, formal connection between
a ML model built on LODE features and long-range interatomic potentials, much like a connection has been shown
between linear models built on density-based features and
short-range many-body potentials [ 33, 46, 47 ]; whether
choosing other exponents in _⟨_ **r** _|V_ _[p]_ _⟩_ can improve models
of dispersion and of long-range effects that do not imply
a characteristic asymptotic behavior; whether equivariant
local features can be obtained by combining the expansion
of the density and that of _⟨_ **r** _|V_ _[p]_ _⟩_ ; whether the combination of SOAP and LODE can be used to improve the
accuracy and the computational efficiency of existing ML
forcefields; whether it is possible to incorporate polarizable atoms physics into the LODE framework. Future
investigation will address these and many other questions,
and unearth the full potential of this physics-inspired
approach to atomistic machine learning.


**ACKNOWLEDGMENTS**


The Authors would like to thank Clemence Cormin
boeuf and G´abor Cs´any for insightful comments on an
early version of the manuscript. M.C and A.G. were
supported by the European Research Council under the
European Union’s Horizon 2020 research and innovation
programme (grant agreement no. 677013-HBMAP), and
by the NCCR MARVEL, funded by the Swiss National
Science Foundation. A.G. acknowledges funding by the
MPG-EPFL Center for Molecular Nanoscience and Tech
nology. We thank CSCS for providing CPU time under
project id s843.


[1] J. Behler, Angewandte Chemie International Edition **56**,
12828 (2017).

[2] [J. Behler and M. Parrinello, Phys. Rev. Lett.](http://dx.doi.org/10.1103/PhysRevLett.98.146401) **98**, 146401
[(2007).](http://dx.doi.org/10.1103/PhysRevLett.98.146401)

[3] A. P. Bart´ok, R. Kondor, and G. Cs´anyi, Phys. Rev. B
**87**, 184115 (2013).

[4] A. Shapeev, Multiscale Model. Sim. **14**, 1153 (2016).

[5] A. Glielmo, P. Sollich, and A. De Vita, Phys. Rev. B **95**,
214302 (2017).

[6] A. Grisafi, D. M. Wilkins, G. Cs´anyi, and M. Ceriotti,
Phys. Rev. Lett. **120**, 036002 (2018).

[7] A. P. Bart´ok, S. De, C. Poelking, N. Bernstein, J. R.
Kermode, G. Cs´anyi, and M. Ceriotti, Sci. Adv. **3** (2017).

[8] S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky,
K. T. Sch¨utt, and K.-R. M¨uller, Sci. Adv. **3** (2017).

[9] L. Zhang, J. Han, H. Wang, R. Car, and W. E, Phys.
Rev. Lett. **120**, 143001 (2018).

[10] E. Prodan and W. Kohn, Proceedings of the National
Academy of Sciences **102**, 11635 (2005).

[11] R. Kjellander, The Journal of Chemical Physics **148**,
193701 (2018).

[12] Z. Guo, F. Ambrosio, W. Chen, P. Gono, and
A. Pasquarello, Chemistry of Materials **30**, 94 (2018).

[13] R. Jorn, R. Kumar, D. P. Abraham, and G. A. Voth,
The Journal of Physical Chemistry C **117**, 3747 (2013).

[14] R. H. French, V. A. Parsegian, R. Podgornik, R. F. Rajter,
A. Jagota, J. Luo, D. Asthagiri, M. K. Chaudhury, Y.-m.
Chiang, S. Granick, S. Kalinin, M. Kardar, R. Kjellander,
D. C. Langreth, J. Lewis, S. Lustig, D. Wesolowski, J. S.
Wettlaufer, W.-Y. Ching, M. Finnis, F. Houlihan, O. A.
von Lilienfeld, C. J. van Oss, and T. Zemb, Rev. Mod.
Phys. **82**, 1887 (2010).

[15] A. P. Bart´ok, M. C. Payne, R. Kondor, and G. Cs´anyi,
Phys. Rev. Lett. **104**, 136403 (2010).

[16] Z. Deng, C. Chen, X.-G. Li, and S. P. Ong, npj Computational Materials **5**, 75 (2019).

[17] [N. Artrith, T. Morawietz, and J. Behler, Phys. Rev. B](http://dx.doi.org/10.1103/PhysRevB.83.153101)
**83** [, 153101 (2011).](http://dx.doi.org/10.1103/PhysRevB.83.153101)

[18] T. Bereau, D. Andrienko, and O. A. von Lilienfeld, J.
Chem. Theory Comput. **11**, 3225 (2015).

[19] T. Bereau, R. A. DiStasio, A. Tkatchenko, and O. A.
von Lilienfeld, J. Chem. Phys. **148**, 241706 (2018).



7


[20] P. Bleiziffer, K. Schaller, and S. Riniker, Journal of
Chemical Information and Modeling **58**, 579 (2018).

[21] B. Nebgen, N. Lubbers, J. S. Smith, A. E. Sifain,
A. Lokhov, O. Isayev, A. E. Roitberg, K. Barros, and
S. Tretiak, J. Chem. Theory Comput. **14**, 4687 (2018).

[22] K. Yao, J. E. Herr, D. Toth, R. Mckintyre, and J. Parkhill,
Chem. Sci. **9** [, 2261 (2018).](http://dx.doi.org/ 10.1039/C7SC04934J)

[23] S. A. Ghasemi, A. Hofstetter, S. Saha, and S. Goedecker,
Phys. Rev. B **[92](http://dx.doi.org/ 10.1103/PhysRevB.92.045131)**, 045131 (2015).

[24] S. Faraji, S. A. Ghasemi, S. Rostami, R. Rasoulkhani,
B. Schaefer, S. Goedecker, and M. Amsler, Phys. Rev. B
**95**, 104105 (2017).

[25] C. B¨ottcher, O. van Belle, P. Bordewijk, and A. Rip,
_Theory of electric polarization_ (Elsevier Scientific Pub.
Co., 1978).

[26] R. Resta, Rev. Mod. Phys. **66**, 899 (1994).

[27] [R. Resta, Journal of Physics: Condensed Matter](http://stacks.iop.org/0953-8984/22/i=12/a=123201) **22**,
[123201 (2010).](http://stacks.iop.org/0953-8984/22/i=12/a=123201)

[28] L. Zhang, M. Chen, X. Wu, H. Wang, W. E, and R. Car,
arXiv:1906.11434 (2019).

[29] D. M. Wilkins, A. Grisafi, Y. Yang, K. U. Lao, R. A.
[DiStasio, and M. Ceriotti, Proc. Natl. Acad. Sci.](http://dx.doi.org/10.1073/pnas.1816132116) **116**,
[3401 (2019).](http://dx.doi.org/10.1073/pnas.1816132116)

[30] M. Rupp, A. Tkatchenko, K.-R. M¨uller, and O. A. von
Lilienfeld, Phys. Rev. Lett. **108**, 058301 (2012).

[31] H. Huo and M. Rupp, arXiv:1704.06439 (2017).

[32] M. Hirn, S. Mallat, and N. Poilvert, Multiscale Modeling
& Simulation **15**, 827 (2017).

[33] M. J. Willatt, F. Musil, and M. Ceriotti, The Journal of
Chemical Physics **150**, 154110 (2019).

[34] Evaluation of the integral for _p >_ 1 require some form
of regularization or short-distance cutoff to remove the
singularity for **r** _→_ **r** _i_ .

[35] R. Dreizler and E. Gross, _Density Functional Theory: An_
_Approach to the Quantum Many-Body Problem_ (Springer
Berlin Heidelberg, 2012).

[36] B. Huang and O. A. von Lilienfeld, arXiv:1707.04146
(2017).

[37] [S. De, A. A. P. Bart´ok, G. Cs´anyi, and M. Ceriotti, Phys.](http://dx.doi.org/10.1039/C6CP00415F)
[Chem. Chem. Phys.](http://dx.doi.org/10.1039/C6CP00415F) **18**, 13754 (2016).

[38] A. Grisafi, D. M. Wilkins, M. J. Willatt, and M. Ceriotti,
arXiv:1904.01623 (2019).

[39] P. P. Ewald, Annalen der Physik **369**, 253 (1921).

[40] U. Essmann, L. Perera, M. L. Berkowitz, T. Darden,
H. Lee, and L. G. Pedersen, The Journal of Chemical
Physics **103**, 8577 (1995).

[41] K. Cahill, _Physical Mathematics_ (Cambridge University
Press, 2013).

[42] M. P. Allen and D. J. Tildesley, _Computer Simulation of_
_Liquids_ (Clarendon Press, 1989).

[43] S. Plimpton, J. Comp. Phys. **117**, 1 (1995).

[44] L. A. Burns, J. C. Faver, Z. Zheng, M. S. Marshall, D. G.
Smith, K. Vanommeslaeghe, A. D. MacKerell, K. M. Merz,
and C. D. Sherrill, J. Chem. Phys. **[147](http://dx.doi.org/ 10.1063/1.5001028)**, 161727 (2017).

[45] V. Blum, R. Gehrke, F. Hanke, P. Havu, V. Havu, X. Ren,
K. Reuter, and M. Scheffler, Computer Physics Communications **180**, 2175 (2009).

[46] A. Glielmo, C. Zeni, and A. De Vita, Phys. Rev. B **97**,
184307 (2018).

[47] R. Drautz, Phys. Rev. B **99**, 014104 (2019).


