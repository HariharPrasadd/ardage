**Comparison of permutationally invariant polynomials, neural networks, and Gaussian**


**approximation potentials in representing water interactions through many-body**


**expansions**


Thuong T. Nguyen, [1, 2] Eszter Sz´ekely, [3] Giulio Imbalzano, [4] J¨org Behler, [5] G´abor Cs´anyi, [3]


Michele Ceriotti, [4] Andreas W. G¨otz, [2] and Francesco Paesani [1, 2,][ a)]


1) _Department of Chemistry and Biochemistry, University of California, San Diego,_


_La Jolla, California 92093, United States_


2) _San Diego Supercomputer Center, University of California, San Diego, La Jolla,_


_California 92093, United States_


3) _Engineering Department, University of Cambridge, Trumpington Street,_


_Cambridge CB2 1PZ, United Kingdom_


4) _Laboratory of Computational Science and Modeling, Institute of Materials,_

_´Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne,_


_Switzerland_


5) _Universit¨at G¨ottingen, Institut f¨ur Physikalische Chemie,_


_Theoretische Chemie, Tammannstr._ _6, 37077 G¨ottingen,_


_Germany_


1


The accurate representation of multidimensional potential energy surfaces is a nec

essary requirement for realistic computer simulations of molecular systems. The


continued increase in computer power accompanied by advances in correlated elec

tronic structure methods nowadays enable routine calculations of accurate interaction


energies for small systems, which can then be used as references for the development


of analytical potential energy functions (PEFs) rigorously derived from many-body


expansions. Building on the accuracy of the MB-pol many-body PEF, we investigate


here the performance of permutationally invariant polynomials, neural networks, and


Gaussian approximation potentials in representing water two-body and three-body


interaction energies, denoting the resulting potentials PIP-MB-pol, BPNN-MB-pol,


and GAP-MB-pol, respectively. Our analysis shows that all three analytical rep

resentations exhibit similar levels of accuracy in reproducing both two-body and


three-body reference data as well as interaction energies of small water clusters ob

tained from calculations carried out at the coupled cluster level of theory, the current


gold standard for chemical accuracy. These results demonstrate the synergy between


interatomic potentials formulated in terms of a many-body expansion, such as MB

pol, that are physically sound and transferable, and machine-learning techniques that


provide a flexible framework to approximate the short-range interaction energy terms.


a) [Electronic mail: fpaesani@ucsd.edu](mailto:fpaesani@ucsd.edu)


2


**I.** **INTRODUCTION**


Since the first Monte Carlo (MC) [1,2] and molecular dynamics (MD) [3,4] simulations of molec

ular systems, computer simulations have become a powerful tool for molecular sciences,


complementing experimental measurements and often providing insights that are difficult to


obtain by other means. Although the first simulations were performed for idealized molecu

lar systems, it was recognized since the beginning that both realism and predictive power of


a computer simulation are directly correlated with the accuracy with which the underlying


molecular interactions are described.


In this context, computer modeling of water is perhaps the most classic example. Given


its role as life’s matrix, [5] it is not surprising that numerous molecular models of water have


been developed (see Refs. 6–9 for recent reviews) since the first simulations performed by


Barker and Watts, [10] and Rahman and Stillinger. [11] However, despite almost 50 years have


passed since these pioneering studies, the development of a molecular model that correctly


reproduces the behavior of water from the gas to the condensed phase still represents a


formidable challenge.


From a theoretical standpoint, the energy of a system containing _N_ water molecules can


be formally expressed through the many-body expansion of the interaction energy (MBE)


as [12]



_N_
� _V_ [2B] ( _i, j_ ) +

_i<j_



_N_
� _V_ [3B] ( _i, j, k_ ) + _· · ·_ + _V_ [NB] (1 _, . . ., N_ ) _._ (1)

_i<j<k_



_E_ N (1 _, . . ., N_ ) =



_N_
� _V_ [1B] ( _i_ ) +


_i_ =1



where _V_ [1B] ( _i_ ) = _E_ ( _i_ ) _−_ _E_ eq ( _i_ ) corresponds to the one-body (1B) energy required to deform


an individual water molecule (monomer) from its equilibrium geometry. All higher-order


terms _V_ [nB] in Eq. 1 describe _n_ -body (nB) interactions defined recursively as



_V_ [nB] (1 _, . . ., n_ ) = _E_ _n_ (1 _, . . ., n_ ) _−_ �



_V_ [1B] ( _i_ ) _−_ �

_i_



� _V_ [2B] ( _i, j_ ) _−_ _. . ._

_i<j_



(2)

� _V_ [(n-1)B] ( _i, j, . . .,_ ( _n −_ 1)) _._

_i<j<···<n−_ 1



_−_
�



Most popular molecular models of water are pairwise additive (i.e., they truncate Eq. 1 at the


2B term) and use an effective _V_ [2B] to account for many-body contributions in an empirical


fashion. [13–24] Although in the early times of computer simulations this simplification was a


necessity dictated by computational efficiency, the importance of many-body effects in water


3


was already recognized in the 1950s by Frank and Wen who introduced a molecular model of


liquid water consisting of “flickering clusters of hydrogen-bonded molecules”, emphasizing


the “co-operative nature” of hydrogen bonding. [25] It also became soon apparent that “pair


potentials do not realistically reproduce both gas and condensed phase water properties”. [26]


The first attempts to derive potential energy functions (PEFs) for aqueous systems which


could rigorously represent the individual terms of the MBE were made in the late 1970s


and 1980s. [27–32] In particular, Clementi and coworkers developed a series of analytical PEFs


for water which were fitted to _ab initio_ reference data obtained at the fourth-order Møller

Plesset (MP4) and Hartree-Fock levels of theory for the 2B and 3B terms, respectively,


and represented many-body effects through a classical polarization term. [31,32] Stillinger and


David developed a polarizable model for water in which H [+] and O [2] _[−]_ moieties were considered


as the basic dynamical and structural elements. [33] Building upon these pioneering studies,


several polarizable models have been proposed over the years, most notably the Dang

Chang model, [34] the TTM models [35–40], and AMOEBA. [41,42] The interested reader is referred


to Refs. 8,9 for recent reviews. Finally, in recent years, also machine learning potentials


have been applied to water [43,44], which are able to include high order many body terms in


the PEFs in form of structural descriptions of the atomic environments.


The development of efficient algorithms for correlated electronic structure methods along


with continued improvements in computer performance has recently made it possible to eval

uate the individual terms of Eq. 1, with chemical accuracy. In parallel, tremendous progress


has been made in constructing multidimensional mathematical functions that are capable


to reproduce interaction energies in generic N-molecule systems, with high fidelity. [45–47]


By combining these three approaches, it has been realized that the MBE provides a rig

orous and efficient framework for the development of full-dimensional PEFs entirely from


first principles, in which low-order terms are accurately determined from correlated electronic


structure data, e.g., using coupled cluster theory with single, double, and perturbative triple


excitations, CCSD(T), in the complete basis set, CBS, limit, the current “gold standard” for


chemical accuracy, and higher-order terms are represented by classical many-body induction.


Along these lines, several many-body PEFs for water have been proposed in the last decade,


the most notable of which are CC-pol, [48] WHBB, [49] HBB2-pol, [50] and MB-pol. [51–53] When em

ployed in computer simulations that allow for explicit treatment of nuclear quantum effects,


these many-body PEFs have been shown to correctly predict structural, thermodynamic,


4


dynamical, and spectroscopic properties of water, from the dimer in the gas phase to liquid


water and ice (see Ref. 9 for a recent review).


Among the existing many-body PEFs, MB-pol (PIP-MB-pol in the present nomencla

ture) has been shown to correctly predict the properties of water across different phases, [54]


reproducing the vibration-rotation tunneling spectrum of the water dimer, [51] the energetics,


quantum equilibria, and infrared spectra of small clusters, [52,55–57] the structural, thermody

namic, and dynamical properties of liquid water, [58,59] including subtle quantum effects such


as equilibrium isotope fractionation [60], the energetics of the ice phases, [61] the infrared and


Raman spectra of liquid water, [62,63] the sum-frequency generation spectrum of the air/wa

ter interface at ambient conditions, [64] and the infrared and Raman spectra of ice I _h_ . [65] It


has been shown that the accuracy of PIP-MB-pol in reproducing the properties of water


depends primarily on its ability to correctly represent each individual term of the MBE at


both short- and long-range.


Briefly, within MB-pol, _V_ [1B] in Eq. 1 is represented by the 1B PEF developed by Par

tridge and Schwenke [66], which reproduces intramolecular distortion with spectroscopic accu

racy. _V_ [2B] includes a term describing 2B dispersion, which is derived from the asymptotic


expansion of the interaction energy, as well as a term describing electrostatic interactions


associated with both permanent and induced molecular moments. At short-range, within


the original PIP-MB-pol, _V_ [2B] is supplemented by a 4 _[th]_ -degree permutationally invariant


polynomial (PIP) [45] that smoothly switches to zero as the distance between the two oxygen


atoms in the dimer approaches 6.5 A. [˚] [51] Similarly, _V_ [3B] includes a 3B induction term that


is supplemented by a short-range 4 _[th]_ -degree PIP that smoothly switches to zero once the


oxygen-oxygen distance between two pairs of water molecules within the trimer approaches


4.5 A. [˚] [52] All higher-body terms are implicitly represented by classical many-body induc

tion according to a modified Thole-type scheme originally adopted by the TTM4-F water


model. [40] The PIP 2B and 3B terms, which were derived from CCSD(T) calculations carried


out in the complete basis set limit for large sets of water dimers and trimers, correct for


deficiencies associated with a purely classical description of intermolecular interactions by


effectively representing quantum-mechanical interactions that arise from the overlap of the


monomer electron densities (e.g., charge transfer and penetration, and Pauli repulsion).


In this study, we investigate the application of Behler-Parrinello neural networks [67,68]


(BPNN) and Gaussian approximation potentials (GAP) as alternatives for the original PIP


5


representations of MB-pol short-range 2B and 3B terms. Using the same training, validation,


and test sets, two additional (BPNN- and GAP-based) analytical expressions of MB-pol are


derived, which effectively exhibit the same accuracy as the original, PIP-based expression.


This study provides further evidence for the ability of the MBE in combination with machine


learning techniques to serve as a rigorous and efficient route for the development of accurate


potential energy functions such as MB-pol in the case of water. The article is organized as


follows: In Section II, we provide an overview of the computational framework associated


with the many-body formalism adopted by MB-pol, while in Section III we describe the


three different models (PIP-MB-pol, BPNN-MB-pol, and GAP-MB-pol) used to represent


water two-body and three-body interactions. The results are presented in Section IV, and


the conclusions along with an outlook are given in Section V.


**II.** **MB-POL FUNCTIONAL FORM AND COMPUTATIONAL DETAILS**


We are employing the MB-pol framework for water, which is based on the MBE of Eq.


(1) and contains explicit terms for the 1B, 2B, and 3B terms, in combination with classi

cal N-body polarization that accounts for all higher-body contributions to the interaction


energy [51,52] . In MB-pol, the 2B term is divided into long-range interactions that are well


described using classical expressions for electrostatics, induction, and dispersion, and short

range interactions that include complex quantum-mechanical effects due to the overlap of


the monomer electron densities.


_V_ [2B] ( _i, j_ ) = _V_ short [2B] [(] _[i, j]_ [) +] _[ V]_ long [ 2B] [(] _[i, j]_ [)] (3)


with


_V_ long [2B] [(] _[i, j]_ [) =] _[ V]_ TTM,elec [ 2B] [(] _[i, j]_ [) +] _[ V]_ TTM,ind [ 2B] [(] _[i, j]_ [) +] _[ V]_ disp [ 2B] [(] _[i, j]_ [)] _[,]_ (4)


where _V_ TTM,elec [2B] [and] _[ V]_ TTM,ind [ 2B] [are electrostatic and induction energies represented by a slightly]

modified version of the Thole-type TTM4-F model [40,51,52], and the dispersion energy _V_ disp [2B] [is]


modeled by a _C_ 6 term that is dampened at short range [51] . Similarly, the 3B term in MB-pol


is decomposed into classical 3B induction that captures essentially all of the 3B interaction


energy at long range and an expression for the highly complex interactions at short range,


_V_ [3B] ( _i, j, k_ ) = _V_ short [3B] [(] _[i, j, k]_ [) +] _[ V]_ TTM,ind [ 3B] [(] _[i, j, k]_ [)] _[.]_ (5)


6


Because corrections to the underlying classical baseline potentials _V_ long [2B] [and] _[ V]_ long [ 3B] [=] _[ V]_ TTM,ind [ 3B]

are only required at short range, and in order to obtain a smooth, differentiable potential


energy surface, MB-pol employs switching functions that smoothly turn off the short-range


potentials _V_ short [2B] [and] _[ V]_ short [ 3B] [once the separation between the oxygen atoms of the water]


molecules exceeds a preset cutoff.


The MB-pol short-range 2B and 3B potentials [51,52] are written as


_V_ short [2B] [(] _[i, j]_ [) =] _[ s]_ [(] _[i, j]_ [)] _[V]_ ML [ 2B] [(] _[i, j]_ [)] (6)


and


_V_ short [3B] [(] _[i, j, k]_ [) =] _[ s]_ [(] _[i, j, k]_ [)] _[V]_ ML [ 3B] [(] _[i, j, k]_ [)] _[,]_ (7)


where


_s_ ( _i, j, k_ ) = _s_ ( _i, j_ ) _s_ ( _i, k_ ) + _s_ ( _i, j_ ) _s_ ( _j, k_ ) + _s_ ( _i, k_ ) _s_ ( _j, k_ ) _._ (8)


The switching function was chosen as



1 if _t_ _ij_ _<_ 0



_s_ ( _i, j_ ) =


















cos [2] [ �] _[π]_

2




_[π]_ 2 _[t]_ _[ij]_ � if 0 _≤_ _t_ _ij_ _<_ 1



_,_ (9)



0 if 1 _≤_ _t_ _ij_



where


_−_
_ij_ _R_ low
_t_ _ij_ = _[R]_ [OO] (10)
_R_ high _−_ _R_ low


is a scaled and shifted oxygen-oxygen distance for water molecules _i_ and _j_ . The MB-pol 2B


and 3B cutoff values are _R_ low [2B] [= 4] _[.]_ [5 ˚A,] _[ R]_ high [2B] [= 6] _[.]_ [5 ˚A,] _[ R]_ low [3B] [= 0] _[.]_ [0 ˚A, and] _[ R]_ high [3B] [= 4] _[.]_ [5 ˚A.]


An accurate description of both the 2B and the 3B short-range interactions requires


flexible multi-dimensional functions, for which the original PIP-MB-pol model employs per

mutationally invariant polynomials [45,69] (PIPs). In this work we investigate the performance


of alternative machine learning (ML) frameworks to represent these 2B and 3B short range


interactions in water, by comparing PIPs to Behler-Parrinello neural networks (BPNN) and


Gaussian approximation potentials (GAP) for _V_ ML [2B] [and] _[ V]_ ML [ 3B] [. We employ the original MB-pol]


switching functions and cutoff values with the PIP and BPNN potentials, while GAP uses


slightly different cutoff values and switching functions. [70] In the context of MBE and neural


networks, it should be noted that a neural network representation of the many-body expan

sion of the interaction energy, truncated at the 3B term, has been reported for methanol. [71]


7


**A.** **Training sets and reference energies**


We employ the original MB-pol 2B and 3B data sets [51,52], which sample regions of the


2B and 3B water PES, respectively, that are most relevant for simulations of water at


normal to moderate temperature and pressure. The 2B training set consists of 42,508 water


dimer structures with center-of-mass separations ranging from 1.6 to 8 A that include the [˚]


global dimer minimum geometry, several saddle points, compressed geometries with positive


interaction energies, and dimers extracted from path-integral molecular dynamics (PIMD)


simulations of liquid water at ambient temperature and pressure. Similarly, the 3B training


set contains 12,347 water trimer structures that include the global minimum and trimers


extracted from a range of MD and PIMD simulations of small water clusters, liquid water,


and water ice phases at varying temperatures and pressures. Both the 2B and the 3B QM


reference energies of these data sets were obtained at the complete basis set (CBS) limit


of coupled cluster theory with single, double and iterative triple excitations, CCSD(T). For


details see the original publications [51,52] . The short-range training set energies _V_ short [ref] [employed]


in this work were obtained from the QM reference data by subtracting the MB-pol baseline


long-range 2B and 3B potentials _V_ long [2B] [and] _[ V]_ TTM,ind [ 3B] [, respectively.]


The original 2B dataset includes a few dimer structures with extremely high binding


energy. Those high energy structures are not only physically unimportant, but also sparsely


distributed, which can lead to difficulties for machine learning techniques to make effective


predictions for structures in this regime because of insufficient information. Therefore, we


have retained only configurations with binding energies below 60 kcal/mol in this work. In


addition, we have removed all configurations with oxygen-oxygen separations larger than the


MB-pol 2B short-range cutoff of 6.5 [˚] A, leading to a total of 42,069 configurations in the final


2B training set. In contrast, the trimer dataset with 12,347 configurations is fully employed.


Each dataset is then randomly divided into three separate sets, training, validation, and


test sets with a ratio of 0.81:0.09:0.1. The first two are used during training and for model


selection while the last one is kept completely isolated from the training procedure and is


employed for the final evaluation only.


8


**B.** **Water cluster test sets**


Reference interaction energies of (H 2 O) _n_ clusters with _n_ = 4 _−_ 6 (see Fig. 4) are based on


geometries optimized with MP2 and RI-MP2 [72,73] and were taken from Ref. 59. The energies


were obtained using the MBE of the interaction energy [74] with both 2B and 3B interaction


energies computed at the same level as the MB-pol 2B and 3B training sets [51,52], that is,


effectively at the CBS limit of CCSD(T). All higher order contributions to the interaction


energy ( _>_ 3B) were obtained from explicitly correlated CCSD(T)-F12b [75] calculations with


the VTZ-F12 basis set [76], which yields results close to the CBS.


**III.** **MANY-BODY MODELS**


**A.** **Permutationally invariant polynomials**


The permutationally invariant polynomials are functions of the distances between pairs


involving both the physical atoms (H and O) and two additional sites L 1 and L 2 that are


located symmetrically along the directions of the oxygen lone pairs of a water molecule,


**r** [(] L _[±]_ [)] = **r** O + [1] 2 _[γ]_ _[∥]_ [(] **[r]** [OH] [1] [ +] **[ r]** [OH] [2] [)] _[ ±][ γ]_ _[⊥]_ [(] **[r]** [OH] [1] _[ ×]_ **[ r]** [OH] [2] [)] _[,]_ (11)


where _γ_ _∥_ and _γ_ _⊥_ are fitting parameters and **r** OH 1 _,_ 2 are the O-H bond vectors. Exponential

functions of the type _ξ_ _i_ = _e_ _[−][kd]_ _[i]_ or _ξ_ _i_ = _e_ _[−][k]_ [(] _[d]_ _[i]_ _[−][d]_ [(0)] [)] and Coulomb-type functions _ξ_ _i_ [Coul] =


_e_ _[−][kd]_ _[i]_ _/d_ _i_ are built for the set _{d_ _i_ _}_ of these distances and are used as basis for the PIPs. The


PIP _V_ ML,PIP = [�] _l_ _[c]_ _[l]_ _[η]_ _[l]_ [ is then constructed from the set] _[ {][ξ]_ _[i]_ _[}]_ [ of these functions, where] _[ {][η]_ _[l]_ _[}]_


are symmetrized monomials up to a given degree. The symmetrization is carried out such


that the monomials, and hence the PIP, are invariant with respect to the permutations of


the water molecules as well as to the permutations of equivalent H and L sites within each


molecule. The polynomial coefficients _c_ _l_ and the exponential coefficients _k_ and distances


_d_ [(0)] are linear and non-linear fitting parameters, respectively. For details see the original


publications [51,52] .


For the 2B PIP we are using 31 basis functions: 6 exponential functions for all intra

molecular HH and OH pairs with exponential coefficients _k_ HH [intra] and _k_ OH [intra] [; 9 Coulomb-type]


functions for all inter-molecular HH, OH and OO pairs with exponential coefficients _k_ HH [inter] [,]


_k_ OH [inter] [, and] _[ k]_ OO [inter] [; 15 exponential functions for all inter-molecular LH, LO and LL pairs with]


9


exponential coefficients _k_ LH [inter] [,] _[ k]_ LO [inter] [, and] _[ k]_ LL [inter] [.] A total of 1153 symmetrized monomials


form _V_ [2B]
ML,PIP [: 6 first-degree monomials using only intermolecular] _[ ξ]_ _[i]_ [ variables, 63 second-]


degree monomials with at most a linear dependence on intramolecular variables, 491 third

degree ones containing at most quadratic intramolecular variables, 593 fourth-degree terms


involving only quadratic intramolecular variables, as in the original paper [51] .


For the 3B PIP we are using 36 exponential functions for each of the intra- and inter

molecular distances between all real (O and H) atoms with exponential coefficients and

distances _k_ HH [intra] [,] _[ k]_ OH [intra] [,] _[ k]_ HH [inter] [,] _[ k]_ OH [inter] [,] _[ k]_ OO [inter] [,] _[ d]_ HH [intra] _[,]_ [(0)], _d_ OH [intra] _[,]_ [(0)], _d_ [inter] HH _[,]_ [(0)], _d_ [inter] OH _[,]_ [(0)], and _d_ OO [inter] _[,]_ [(0)] .


A total of 1163 symmetrized monomials form _V_ ML,PIP [3B] [: 13 second-degree monomials with]


only intermolecular exponential variables, 202 third-degree monomials with at most a linear


dependence on intramolecular variables, 948 fourth-degree monomials containing at most


a linear dependence on intramolecular variables or intermolecular ones involving oxygen

oxygen and hydrogen-hydrogen distances, as in the original paper [52] .


The linear and nonlinear parameters were optimized using a singular value decomposition


and the simplex algorithm, respectively, by minimizing the regularized sum of squared errors


_χ_ [2] for the corresponding training set _S_, commonly referred to as Tikhonov regularization or


ridge regression [77],



_χ_ [2] = �



�[ _V_ short ( _n_ ) _−_ _V_ short [ref] [(] _[n]_ [)]] [2] [ + Γ] [2] [ �]

_n∈S_ _l_



_c_ [2] _l_ _[.]_ (12)

_l_



The regularization parameter Γ was set to 5 _×_ 10 _[−]_ [4] for 2B and 1 _×_ 10 _[−]_ [4] for 3B in order to


reduce the variation of the linear parameters without spoiling the overall accuracy of the


fits.


**B.** **Behler-Parrinello neural networks**


Based on the assumption that the total energy of a system can be written as a sum of


atomic energy contributions, a BPNN consists of a set of fully connected feed-forward neural


networks, each of which provides an atomic energy [67,68] . Each atomic network takes as its


input a set of atom-centered symmetry functions [78] that encode the atomic positions and at


the same time are invariant with respect to overall rotation and translation, as well as to


permutations of like atoms. The invariance of the total energy is assured by enforcing that


all atomic networks of the same species are identical, thus having the same structure and


weights. As a result, for the water systems considered here, there are two sub-networks, one


10


for all H atoms and the other for all O atoms, which need to be trained simultaneously. Aim

ing at smoothly disabling the short-range interaction energy contribution at long distances,


described in Eqs. 6-7, the sum of all atomic energies from the last layer of the sub-networks


is multiplied by the switching function to produce a final output for a BPNN. The network


weights are determined with respect to the values of the reference short-range interaction


energies.


The following modified radial and angular symmetry functions, which lack the cut-off


functions of the original BPNN approach, have been chosen for each atom _i_


_G_ _[rad]_ _i_ = � _e_ _[−][η]_ [(] _[R]_ _[ij]_ _[−][R]_ _[s]_ [)] [2] _,_ (13)

_j_ = _̸_ _i_


_̸_ _̸_



_̸_


_G_ _[ang]_ _i_ = 2 [1] _[−][ζ]_ [ �]


_j_ = _̸_ _i_ _̸_



_̸_


� (1 + _λ_ cos _θ_ _ijk_ ) _[ζ]_ _e_ _[−][η]_ _[′]_ [(] _[R]_ _[ij]_ [+] _[R]_ _[ik]_ [+] _[R]_ _[jk]_ [)] [2] _,_ (14)

_̸_ _k_ = _̸_ _i,j_



_̸_


_̸_ _̸_


resulting in an input vector **G** _i_ = _{G_ _[rad/ang]_ _i_ _}_ for the atomic network. _θ_ _ijk_ denotes the


angle enclosed by two interatomic distances _R_ _ij_ and _R_ _ik_ . Each summation above takes into


account only same combination of atomic species and the set of parameters, _{_ ( _η, R_ _s_ ) _}_ and


_{_ ( _ζ, λ, η_ _[′]_ ) _}_, is the same for each type of species grouping. We have removed the cut-off


function from the original forms of the symmetry functions used in Ref. 67 and 68 since we


apply the MB-pol 2B and 3B switching functions, thus never feeding any structures to the


2B and 3B BPNNs that are beyond the cut-off region.


The dimension of the input vector should reflect a balance between giving an effective


resolution of the local environment and the computational cost of training and inference


with a large input vector neural network. After carefully examining different parameter


sets, we have come up with the final set as follows. For the 2B term, there are 24 radial


Gaussian-shape filters, Eq. (13), whose centers _R_ _s_ are placed evenly between 0.8 A and [˚]


8 [˚] A, which are relatively close to the smallest and the largest interatomic distances in the


training set. For O-O distances the two smallest centers are excluded because the O-O


separation is well beyond the space covered by these two filters. The width of those filters is


proportional to their centers’ position, 1 _/_ _[√]_ 2 _η_ = 0 _._ 2 _R_ _s_ . The angular probe in Eq. 14 takes


_ζ_ = [1 _,_ 4 _,_ 16] for different filter widths, _λ_ = _±_ 1 for switching the filter’s center between 0 and

_π_, and _η_ _[′]_ = [0 _._ 001 _,_ 0 _._ 01 _,_ 0 _._ 05] (A [˚] _[−]_ [2] ) for various levels of the separation dependence. As for


3B BPNN, a similar scheme is applied with few adjustments, which include 16 radial filters


with centers arranged in the same range, between 0.8 A and 8 [˚] [˚] A, and two levels of separation


11


dependence attached to the angular filter, _η_ _[′]_ = [0 _._ 001 _,_ 0 _._ 03] (A [˚] _[−]_ [2] ). Moreover, to reduce the


redundancy and computational cost, for the angular probe for hydrogen atoms, we consider


only two types of triplet of atoms, a hydrogen atom with other two hydrogen atoms or with


an oxygen and another hydrogen. In total, a set of 82 and 84 symmetry functions for O and


H is formed for the 2B BPNN while another set of 66 and 56 functions for O and H is used


for the 3B BPNN. The complete set of the symmetry function parameters can be found in


the SI.


The neural network training encounters various hyperparameters and different techniques


for initialization of these parameters, which are mostly found by trial and error. Following


is our final network architecture and set-up for the network training. The atomic network


consists of one input layer, three hidden layers, and one single output layer. The input


layer takes as its input the preprocessed symmetry functions, each of which is obtained by


rescaling the symmetry function with its corresponding maximum value in the training and


validation sets. Furthermore, the numbers of units in each hidden layer are chosen to be the


same for both atomic networks for O and H. Overall, with 34 and 22 units per hidden layer,


the final 2B and 3B BPNN models contain 10542 and 4798 weight and bias parameters,


respectively. For the continuity of the energy functional, the activation function for each


unit is chosen to be a hyperbolic tangent for the hidden layers and a linear function for the


output layer. Besides, the reference energies for the 2B training are converted to energy


per atom in eV unit so that the network targets a similar range of values as given by the


activation functions.


We build the network models using Keras [79] with Theano [80] backend and choose the Adam


optimizer with a batch size of 64 for training. The Nguyen-Widrow method [81] is employed to


initialize the network weights and biases. For a stable and effective training, the optimiza

tion process is continuously carried out five times with descending starting learning rates


[10 _[−]_ [3] _,_ 2 _·_ 10 _[−]_ [4] _,_ 6 _·_ 10 _[−]_ [5] _,_ 9 _·_ 10 _[−]_ [6] _,_ 10 _[−]_ [6] ] and corresponding numbers of iterations, or epochs,


[1500 _,_ 1500 _,_ 1000 _,_ 1000 _,_ 1000]. Furthermore, we apply an additional decay rate _α_ = 10 _[−]_ [5] to


each learning rate such that at a given epoch _k_ the leaning rate is _lr_ _k−_ 1 _/_ (1 + _α · k_ ) based


on the value at the previous epoch _lr_ _k−_ 1 . The training is to optimize the mean squared


error of the modeled energies compared to the reference data in the training set. To avoid


overfitting, on each epoch, the quality of the model is monitored on the validation set such


that only the model that gives the highest accuracy over this set is ultimately kept. Finally,


12


the trained model is then evaluated on the test set to quantify its capability of generalization


to unseen data. For the systems considered here, the training processes generally take three


hours and one hour on a Tesla K40 GPU with the GPU-accelerated cuDNN library for 2B


and 3B sets, respectively.


**C.** **Gaussian Approximation Potentials**


The Gaussian Approximation Potential (GAP) [82,83] framework, available in the QUIP


program package [84], is an implementation of Gaussian process regression (GPR) interpola

tion for the atomic energy as a function of the geometry of the neighbouring atoms. The


functional form representing a function _f_ that is to be interpolated is identical to that of


kernel ridge regression,

_f_ ( _**R**_ ) = � _b_ _k_ _K_ ( _**R**_ _,_ _**R**_ _k_ ) _,_ (15)


_k_


where the high dimensional vector _**R**_ represents the complete geometry of neighbouring


atoms, _k_ indexes a set of representative data points _{_ _**R**_ _k_ _}_, _K_ is the kernel function, and


_{b_ _k_ _}_ are fitting coefficients. In the GPR formalism, _K_ corresponds to an estimate of the


covariance of the unknown function, and the linear system is solved in the least squares


sense using Tikhonov regularisation, but the regularisation parameters are now interpreted


as estimates of data and model error. In the present case, the regularisation was chosen


to be 0.00115 kcal/mol for the 2B term, and 0.0231 kcal/mol for the 3B term after manual


exploration of the data.


The success of the GAP fit depends on choosing an appropriate kernel, one that captures


the structure of the input data and as much as possible about the function to be fitted. Here


we use the ”Smooth Overlap of Atomic Positions” (SOAP), a kernel that is the rotationally


integrated overlap of the neighbour densities, which was shown to be equivalent to the scalar


product of the spherical Fourier spectrum [83] . The atomic environment of atom _i_ is described


by a set of neighbour densities, one for each atomic species, which are represented as the


sum of Gaussians each centred on one of the neighbouring atoms _j_ [85] :



_−_ _[|]_ _**[r]**_ _[ −]_ _**[r]**_ _[i][j]_ _[|]_ [2]
exp
� 2 _σ_ _at_ [2]
_j_



_ρ_ _[α]_ _i_ [(] _**[r]**_ [) =] �



2 _σ_ _at_ [2]



_f_ _cut_ ( _**r**_ _ij_ ) (16)
�



where _j_ ranges over neighbours with atomic species _α_, _**r**_ _ij_ are the positions relative to _i_,


and _σ_ _at_ [2] [is a smoothing parameter. We included the switching function] _[ f]_ _[cut]_ [which smoothly]


13


goes to zero beyond a specified radial value. This local atomic neighbour density can be


expanded in terms of spherical harmonics, _Y_ _lm_ (ˆ _**r**_ ) and orthogonal radial functions, _g_ _n_ ( _|_ _**r**_ _|_ ) :


_ρ_ _[α]_ _i_ [(] _**[r]**_ [) =] � _c_ _[α]_ _nlm_ _[g]_ _[n]_ [(] _[|]_ _**[r]**_ _[|]_ [)] _[Y]_ _[lm]_ [(ˆ] _**[r]**_ [)] (17)


_nlm_


The expansion coefficients are then combined to form the rotationally invariant power spec

trum:



�( _c_ _[α]_ _n_ 1 _lm_ [)] _[†]_ [(] _[c]_ _[β]_ _n_ 2 _lm_ [)] (18)


_m_



_p_ _[αβ]_ _n_ 1 _n_ 2 _l_ [(] _**[R]**_ _[i]_ [) =] _[ π]_



~~�~~



8

2 _l_ + 1



where we have emphasized the functional dependence on the complete neighbour geometry.


The complete SOAP kernel can be written as:


_ζ_

_K_ ( _**R**_ _,_ _**R**_ _**[′]**_ ) = _p_ _[αβ]_ _n_ 1 _n_ 2 _l_ [(] _**[R]**_ [)] _[p]_ _[αβ]_ _n_ 1 _n_ 2 _l_ [(] _**[R]**_ _**[′]**_ [)] _,_ (19)
�� _αβn_ 1 _n_ 2 _l_ �


where we have allowed for a small integer exponent _ζ_ (here set to 2). The kernel is also


normalised so that the kernel of each environment with itself is unity. Separate fits are made


to the atomic energy function corresponding to each atomic species taken as the center of an


atomic environment. The key free parameters are the radial cutoff in _f_ _cut_, and the smoothing


parameter _σ_ _at_ . In the present cases here, atomic energy functions are represented by the


sum of two kernels [86], one with a smaller radial cutoff (4.5 [˚] A) and smaller smoothing (0.4 [˚] A),


and one with a larger cutoff (6.5 A for the 2B and 7.0 [˚] [˚] A for the 3B fit) and larger smoothing


(1.0 [˚] A). The RMSE is only weakly sensitive to these, and some manual optimisation was


carried out. Each fit uses 10 radial basis functions and a spherical harmonics basis band


limit of 10. The representative environments for the fit are chosen using CUR matrix


decomposition [87] . The number of representative points are 9000 in the 2B fit and 10000


in the 3B fit. The full command lines of the fits are given in the Supporting Information.


Note that although formally the GAP construction corresponds to a decomposition of the


total energy into atomic energies, similarly to BPNN above, the cutoffs are sufficiently large


to encompass all atoms in the water dimer and trimers in the dataset, and therefore the


decomposition does not represent an approximation.


14


**IV.** **RESULTS**


**A.** **2B and 3B interactions, and the structure of the training data**


The root mean squared errors (RMSEs) obtained with PIPs, BPNNs, and GAPs for the


2B and 3B datasets are reported in Table I. For the 2B term, all three methods achieve


TABLE I. RMSE (in kcal/mol) per isomer on the provided training, validation, and test sets in


the PIP, BPNN, GAP short range interaction two-body (2B) and three-body (3B) energy fitting.


2B 3B


training validation test training validation test


PIP 0.0349 0.0449 0.0494 0.0262 0.0463 0.0465


BPNN 0.0493 0.0784 0.0792 0.0318 0.0658 0.0634


GAP 0.0176 0.0441 0.0539 0.0052 0.0514 0.0517


similar accuracy: the error on the training set is less than 0.050 kcal/mol per dimer while


the errors on validation and test sets are less than 0.080 kcal/mol per dimer. These errors


demonstrate a high level of accuracy since the average value of the target energies in the


dataset is 3 kcal/mol. Among the three, the 2B PIP model appears to perform better on


the validation and test sets and suffers less from overfitting. The difference in RMSEs for


the training set and the test set are below 0.02 kcal/mol with PIP, but around 0.03 kcal/mol


with BPNN and 0.04 kcal/mol with GAP. The GAP model gets a slightly lower error for


the training set, but overfitting prevents to achieve a similar accuracy for the test set.


In order to investigate in more detail the performance of the different regression schemes


for predicting the 2B and 3B energies over the MB-pol dimer and trimer data sets, we used a


dimensionality reduction scheme to obtain a 2D representation of the structure of the train


set. We followed a procedure similar to that used in Ref. 88 to map a database of oligopeptide


conformers. A metric based on SOAP descriptors [85] was used to assess the similarity between


reference conformations of dimers or trimers. A 2D map that best preserved the similarity


between 1000 reference configurations selected by farthest point sampling [89] was obtained


using the sketch-map algorithm [90,91] . All other configurations (training and testing) were


then assigned 2D coordinates ( _x_ _i_ _, y_ _i_ ) by projecting them on the same reference sketch-map.


We could then compute the histogram of configurations _h_ ( _x, y_ ), the averages of the properties


15


FIG. 1. (a) Sketch-map representation for the training data set for dimer configurations. Points


are colored according to O-O distance, and a few reference configurations are also shown. (b)


Histogram of the training point positions on the sketch-map. The train set density is also reported


on other plots as a reference for comparison. (c) Conditional average of the 2B energies for different


parts of the train set. (d-f) Conditional average RMSE for the PIP, BPNN, GAP fits of the 2B


energy in different parts of the test set.


of the different configurations, and of the test RMSE for the various methods, conditional


on the position on the 2D map, e.g.


_h_ ( _x, y_ ) = _⟨δ_ ( _x −_ _x_ _i_ ) _δ_ ( _y −_ _y_ _i_ ) _⟩_



_V_ short [2B] [(] _[x, y]_ [) =]



� _V_ short [2B] [(] _[i]_ [)] _[δ]_ [(] _[x][ −]_ _[x]_ _[i]_ [)] _[δ]_ [(] _[y][ −]_ _[y]_ _[i]_ [)] � (20)

_h_ ( _x, y_ ) _._



Figure 1 demonstrates the application of this analysis to the dimer dataset. One of


the sketch-map coordinates correlates primarily with O-O distance, while different relative


16


orientations and internal monomer deformations are mixed in the other direction. Conforma

tional space is very non-uniformly sampled (Fig. 1b), with a large number of configurations


at large O-O distance – which correspond to _V_ short [2B] [of less than 0.01 kcal/mol – and at in-]


termediate distances, with sparser sampling in the high-energy, repulsive region (Fig. 1c).


It is interesting to see that the three regression schemes we considered exhibit very similar


performance in the various regions, with tiny errors _<_ 0 _._ 01 kcal/mol for far-away molecules,


and much larger errors, as large as 1 kcal/mol, for configurations in the repulsive region.


These large errors are not only due to the high energy scale of _V_ short [2B] [in this region: the]


largest errors appear in the portion of the map which is characterized by both large _V_ short [2B]

and low density of sample points.


10





1


0.1


100 1000 10000

n. training structures


FIG. 2. TEST RMSE as a function of the size of the train set for the 2B energy contribution,


using a BPNN for the regression. Training configurations were selected at random (5 independent


selections, average and standard deviation shown) or by farthest point sampling.


The non-uniform sampling of the dimer space configuration means that there is room to


improve it. Figure 2 compares the test RMSE obtained by BPNN fits constructed on subsets


of the overall training set. The error can be reduced by up to a factor of five by choosing


the subset with a FPS strategy, rather than at random. This observation is consistent


with recent observations made using SOAP-GAP in a variety of systems [86,92] . Selecting


training configurations from a larger database of potential candidates using FPS gives a


viable strategy to reduce the number of high-end calculations that have to be performed to


describe accurately interactions in the construction of a MB potential.


Figure 3 shows a similar analysis for the case of the trimer data and _V_ short [3B] [. 3B energies]


span a smaller range than the 2B component, that includes most of the core repulsion. The


17


FIG. 3. (a) Sketch-map representation for the training data set for trimer configurations. Points


are colored according to the root mean square of the three O-O distances; trimer geometries are


also represented as triangles, together with a few structures for which a snapshot is shown. (b)


Histogram of the training point positions on the sketch-map. The train set density is also reported


on other plots as a reference for comparison. (c) Conditional average of the 3B energies for different


parts of the train set. (d-f) Conditional average RMSE for the PIP, BPNN, GAP fits of the 3B


energy in different parts of the test set.


higher dimensionality of the problem, however, makes this a harder regression problem, as


is apparent from the irregular correlations between energy and position on the map, that


reveals an alternation of regions of positive and negative contributions.


As a result, the absolute RMSE accuracy of the regression models is comparable to


that for the 2B terms, with PIP and GAP yielding comparable accuracy (RMSE _≈_ 0 _._ 05


kcal/mol), followed closely by BPNN (RMSE _≈_ 0 _._ 06 kcal/mol). As in the case of 2B energy


18


contributions, an analysis of the error distribution shows that improving the sampling density


and uniformity for the train set is likely to be the most effective strategy to further improve


the model. Errors are concentrated at the periphery of the data set. The good performance


of the GAP model can be traced to the fact that it provides a very good description of the


short RMS _d_ OO region, even if only a few reference structures are available, even though it


performs less well than PIP or NN for configurations that involve far away molecules.


**B.** **Water clusters**


Isomers of water clusters (H 2 O) _n_ with _n_ = 4 _,_ 5 _,_ 6 (see Fig. 4 for the structures) serve as


larger test systems to investigate the performance of MB-pol with PIP, BPNN, and GAP


representations of the short-range 2B and 3B energies and the corresponding effect on the


total interaction energies of the clusters.


(H 2 O) 4 :


Isomer 1 Isomer 2 Isomer 3


(H 2 O) 5 :


Isomer 1 Isomer 2 Isomer 3 Isomer 4


Isomer 5 Isomer 6 Isomer 7


(H 2 O) 6 :


Isomer 1 - prism Isomer 2 - cage Isomer 3 - book1 Isomer 4 - book2



cyclic-boat2



Isomer 5 - bag Isomer 6 cyclic-chair



cyclic-boat1



FIG. 4. Isomers of water clusters (H 2 O) _n_, _n_ = 4 _,_ 5 _,_ 6, used for the analysis of the performance of


PIP, BPNN and GAP representations of 2B and 3B energies. Reproduced from Ref. 59.


19


An analysis of the 2B and 3B contributions to the total interaction energy of the water


clusters is shown in Fig. 5. MB-pol errors with respect to the CCSD(T) reference values


are smaller than 0.3 kcal/mol in all cases, independent of the cluster size and geometry and


independent of the approach that is used to represent the short-range 2B and 3B energies.


The errors increase somewhat with cluster size as the individual errors for the larger number


of 2B and 3B terms can start to add up for cluster configurations that contain repeating


dimer and trimer units. This is mostly pronounced for 2B interaction energies. While similar


errors in 2B interaction energies are seen with the three potentials, GAP-MB-pol exhibits


smaller errors in 3B interaction energies than PIP-MB-pol and BPNN-MB-pol.


(H 2 0) 4 (H 2 0) 5 (H 2 0) 6

0.4


0.2


0.0


~~−~~ 0.2


0.4



0.2


0.0


~~−~~ 0.2


0.4


0.2


0.0


~~−~~ 0.2

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||

2B 3B



2B 3B 2B 3B





FIG. 5. Errors (in kcal/mol) in the 2B and 3B interaction energies calculated with PIP, BPNN and


GAP short-range potentials with respect to reference CCSD(T) values for water clusters (H 2 O) _n_,


_n_ = 4 _,_ 5 _,_ 6.


Fig. 6 compares the total interaction energies of all water cluster isomers as obtained with


MB-pol using PIP, BPNN, and GAP representations of short-range 2B and 3B energies in


comparison to the CCSD(T)/CCSD(T)-F12b reference values. In correspondence with the


2B and 3B contributions, the error in the total interaction energy increases with cluster


size. Due to extended hydrogen bonding and symmetry, the ring-type isomers also have


relatively large higher-body contributions that can be non-negligible and that can exhibit


20


errors of similar magnitude as the 2B and 3B terms as has been shown in previous work [59,93] .


The error for this type of isomers is thus particularly large. However, the deviation in the


computed interaction energies never exceeds 0.8 kcal/mol and, most importantly, the relative


order of the total interaction energies for the different isomers of each cluster is retained in all


cases. Overall we conclude that any of the investigated approaches to represent short-range


2B and 3B interaction energies within the MB-pol model is suitable to predict accurate


interaction energies of water clusters.

















FIG. 6. Interaction energies of the low-lying isomers of water clusters (H 2 O) _n_, _n_ = 4 _,_ 5 _,_ 6, obtained


using MB-pol with PIP, BPNN and GAP short-range 2B and 3B potentials in comparison to


CCSD(T)/CCSD(T)-F12b reference values.


**V.** **CONCLUSIONS**


We have explored different representations of MB-pol short-range two-body (2B) and


three-body (3B) interaction energies using permutationally invariant polynomials (PIP),


Behler-Parrinello neural networks (BPNN), and Gaussian approximation potentials (GAP).


The accuracy of the three models has been assessed by comparing their ability to reproduce


large datasets of CCSD(T)/CBS 2B and 3B interaction energies as well as in predicting the


energetics of small water clusters, which are always found to be within chemical accuracy


(1 kcal/mol). These results demonstrate that the three models are effectively equivalent,


consistently exhibiting similar performance in representing many-body interactions in water


within the MB-pol framework. The most promising approach to further increase the accu

21


racy for both the 2B and 3B terms involves increasing the number of reference calculations


and optimizing the training set to cover more uniformly the relevant configuration space.


Our analysis of the 2B and 3B contributions to the MB-pol interaction energies can be


taken as a case study for the general problem of the systematic construction of potentials


derived from the many-body expansion. The combination between an accurate machine

learning representation of the short-range terms in combination with a physically sound


form of long-range contributions provides a promising route to the development of accurate,


efficient and transferable potential energy surfaces.


**VI.** **ACKNOWLEDGEMENTS**


This work was supported by the National Science Foundation through grant no. ACI

1642336 (to F.P. and A.W.G.). This work used the Extreme Science and Engineering Dis

covery Environment (XSEDE), which is supported by National Science Foundation grant no.


ACI-1548562. J.B. is grateful for a Heisenberg professorship funded by the DFG (Be3264/11

2). E.Sz. would like to acknowledge the support of the Peterhouse Research Studentship and


the support of BP International Centre for Advanced Materials (ICAM). M.C. was supported


by the European Research Council under the European Union’s Horizon 2020 research and


innovation programme (grant agreement no. 677013-HBMAP). G.I. acknowledges funding


from the Fondazione Zegna.


**REFERENCES**


1 N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller, J. Chem.


Phys. **21**, 1087 (1953).


2 W. Wood and F. Parker, J. Chem. Phys. **27**, 720 (1957).


3 B. J. Alder and T. E. Wainwright, J. Chem. Phys. **31**, 459 (1959).


4 B. Alder and T. Wainwright, J. Chem. Phys. **33**, 1439 (1960).


5 P. Ball, Chem. Rev. **108**, 74 (2008).


6 B. Guillot, J. Mol. Liq. **101**, 219 (2002).


7 C. Vega and J. L. Abascal, Phys. Chem. Chem. Phys. **13**, 19663 (2011).


8 I. Shvab and R. J. Sadus, Fluid Phase Equilib. **407**, 7 (2016).


22


9 G. A. Cisneros, K. T. Wikfeldt, L. Ojam¨ae, J. Lu, Y. Xu, H. Torabifard, A. P. Bart´ok,


G. Cs´anyi, V. Molinero, and F. Paesani, Chem. Rev. **[116](http://dx.doi.org/ 10.1021/acs.chemrev.5b00644)**, 7501 (2016).


10 J. Barker and R. Watts, Chem. Phys. Lett. **3**, 144 (1969).


11 A. Rahman and F. H. Stillinger, J. Chem. Phys. **55**, 3336 (1971).


12 J. Mayer and M. Mayer, _Statistical Mechanics_ (John Wiley, New York, 1940).


13 H. J. Berendsen, J. P. Postma, W. F. van Gunsteren, and J. Hermans, in _Intermolecular_


_forces_ (Springer, 1981) pp. 331–342.


14 W. L. Jorgensen, J. Chandrasekhar, J. D. Madura, R. W. Impey, and M. L. Klein, J.


Chem. Phys. **79**, 926 (1983).


15 H. Berendsen, J. Grigera, and T. Straatsma, J. Phys. Chem. **91**, 6269 (1987).


16 L. X. Dang and B. M. Pettitt, J. Phys. Chem. **91**, 3349 (1987).


17 D. M. Ferguson, J. Comp. Chem. **16**, 501 (1995).


18 M. W. Mahoney and W. L. Jorgensen, J. Chem. Phys. **112**, 8910 (2000).


19 H. W. Horn, W. C. Swope, J. W. Pitera, J. D. Madura, T. J. Dick, G. L. Hura, and


T. Head-Gordon, J. Chem. Phys. **120**, 9665 (2004).


20 Y. Wu, H. L. Tepper, and G. A. Voth, J. Chem. Phys. **124**, 024503 (2006).


21 F. Paesani, W. Zhang, D. A. Case, T. E. Cheatham III, and G. A. Voth, J. Chem. Phys.


**125**, 184507 (2006).


22 S. Habershon, T. E. Markland, and D. E. Manolopoulos, J. Chem. Phys. **131**, 024501


(2009).


23 K. Park, W. Lin, and F. Paesani, J. Phys. Chem. B **116**, 343 (2011).


24 I. S. Joung and T. E. Cheatham III, The journal of physical chemistry B **112**, 9020 (2008).


25 H. S. Frank and W.-Y. Wen, Discuss. Faraday Soc. **24**, 133 (1957).


26 T. P. Lybrand and P. A. Kollman, J. Chem. Phys. **83**, 2923 (1985).


27 O. Matsuoka, E. Clementi, and M. Yoshimine, J. Chem. Phys. **64**, 1351 (1976).


28 G. Lie and E. Clementi, Phys. Rev. A **33**, 2679 (1986).


29 A. Lyubartsev and A. Laaksonen, Chem. Phys. Lett. **325**, 15 (2000).


30 K. Honda and K. Kitaura, Chem. Phys. Lett. **140**, 53 (1987).


31 U. Niesar, G. Corongiu, M.-J. Huang, M. Dupuis, and E. Clementi, International Journal


of Quantum Chemistry **36**, 421 (1989).


32 U. Niesar, G. Corongiu, E. Clementi, G. Kneller, and D. Bhattacharya, Journal of Physical


Chemistry **94**, 7949 (1990).


23


33 F. H. Stillinger and C. W. David, J. Chem. Phys. **69**, 1473 (1978).


34 L. X. Dang and T.-M. Chang, J. Chem. Phys. **106**, 8149 (1997).


35 C. J. Burnham and S. S. Xantheas, J. Chem. Phys. **116**, 1479 (2002).


36 S. S. Xantheas, C. J. Burnham, and R. J. Harrison, J. Chem. Phys. **116**, 1493 (2002).


37 C. J. Burnham and S. S. Xantheas, J. Chem. Phys. **116**, 1500 (2002).


38 C. J. Burnham and S. S. Xantheas, J. Chem. Phys. **116**, 5115 (2002).


39 G. S. Fanourgakis and S. S. Xantheas, J. Chem. Phys. **128**, 074506 (2008).


40 [C. J. Burnham, D. J. Anick, P. K. Mankoo, and G. F. Reiter, J. Chem. Phys.](http://scitation.aip.org/content/aip/journal/jcp/128/15/10.1063/1.2895750) **128**, 154519


[(2008).](http://scitation.aip.org/content/aip/journal/jcp/128/15/10.1063/1.2895750)


41 P. Ren and J. W. Ponder, J. Phys. Chem. B **107**, 5933 (2003).


42 L.-P. Wang, T. Head-Gordon, J. W. Ponder, P. Ren, J. D. Chodera, P. K. Eastman, T. J.


Martinez, and V. S. Pande, J. Phys. Chem. B **117**, 9956 (2013).


43 A. P. Bart´ok, M. J. Gillan, F. R. Manby, and G. Cs´anyi, Phys. Rev. B **[88](http://dx.doi.org/10.1103/PhysRevB.88.054104)**, 054104 (2013).


44 [T. Morawietz, A. Singraber, C. Dellago, and J. Behler, Proc. Natl. Acad. Sci.](http://dx.doi.org/10.1073/pnas.1602375113) **113**, 8368


[(2016).](http://dx.doi.org/10.1073/pnas.1602375113)


45 B. J. Braams and J. M. Bowman, Int. Rev. Phys. Chem. **28**, 577 (2009).


46 [J. Behler, Angew. Chem. Int. Ed.](http://dx.doi.org/10.1002/anie.201703114) **56**, 12828 (2017).


47 A. P. Bart´ok, M. J. Gillan, F. R. Manby, and G. Cs´anyi, Physical Review B **88**, 054104


(2013).


48 R. Bukowski, K. Szalewicz, G. C. Groenenboom, and A. van der Avoird, Science **315**,


1249 (2007).


49 Y. Wang, X. Huang, B. C. Shepler, B. J. Braams, and J. M. Bowman, J. Chem. Phys.


**134**, 094509 (2011).


50 V. Babin, G. R. Medders, and F. Paesani, J. Phys. Chem. Lett. **3**, 3765 (2012).


51 [V. Babin, C. Leforestier, and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct400863t) **9**, 5395 (2013).


52 [V. Babin, G. R. Medders, and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct500079y) **10**, 1599 (2014).


53 [G. R. Medders, V. Babin, and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct5004115) **10**, 2906 (2014).


54 F. Paesani, Acc. Chem. Res. **49**, 1844 (2016).


55 J. O. Richardson, C. P´erez, S. Lobsiger, A. A. Reid, B. Temelso, G. C. Shields, Z. Kisiel,


D. J. Wales, B. H. Pate, and S. C. Althorpe, Science **351**, 1310 (2016).


56 W. T. S. Cole, J. D. Farrell, D. J. Wales, and R. J. Saykally, Science **352**, 1194 (2016).


24


57 S. E. Brown, A. W. G¨otz, X. Cheng, R. P. Steele, V. A. Mandelshtam, and F. Paesani,


J. Am. Chem. Soc. **139**, 7082 (2017).


58 [G. R. Medders, V. Babin, and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct300913g) **9**, 1103 (2013).


59 S. K. Reddy, S. C. Straight, P. Bajaj, C. Huy Pham, M. Riera, D. R. Moberg, M. A.


Morales, C. Knight, A. W. G¨otz, and F. Paesani, J. Chem. Phys. **145**, 194504 (2016).


60 B. Cheng, J. Behler, and M. Ceriotti, J. Phys. Chem. Letters **7**, 2210 (2016).


61 [C. H. Pham, S. K. Reddy, K. Chen, C. Knight, and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/ 10.1021/acs.jctc.6b01248)


**13** [, 1778 (2017).](http://dx.doi.org/ 10.1021/acs.jctc.6b01248)


62 [G. R. Medders and F. Paesani, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct501131j) **11**, 1145 (2015).


63 S. C. Straight and F. Paesani, J. Phys. Chem. B **120**, 8539 (2016).


64 [G. R. Medders and F. Paesani, J. Am. Chem. Soc.](http://dx.doi.org/10.1021/jacs.6b00893) **138**, 3912 (2016).


65 D. R. Moberg, S. C. Straight, C. Knight, and F. Paesani, J. Phys. Chem. Lett. **8**, 2579


(2017).


66 H. Partridge and D. W. Schwenke, J. Chem. Phys. **[106](http://dx.doi.org/http://dx.doi.org/10.1063/1.473987)**, 4618 (1997).


67 J. Behler and M. Parrinello, Phys. Rev. Lett. **[98](http://dx.doi.org/10.1103/PhysRevLett.98.146401)**, 146401 (2007).


68 [J. Behler, International Journal of Quantum Chemistry](http://dx.doi.org/10.1002/qua.24890) **115**, 1032 (2015).


69 [Z. Xie and J. M. Bowman, J. Chem. Theory Comput.](http://dx.doi.org/10.1021/ct9004917) **6**, 26 (2010).


70 A. P. Bart´ok and G. Cs´anyi, International Journal of Quantum Chemistry **115**, 1051


(2015).


71 K. Yao, J. E. Herr, and J. Parkhill, J. Chem. Phys. **146**, 014106 (2017).


72 [D. M. Bates and G. S. Tschumper, J. Phys. Chem. A](http://dx.doi.org/10.1021/jp8105919) **113**, 3555 (2009).


73 B. Temelso, K. A. Archer, and G. C. Shields, J. Phys. Chem. A **[115](http://dx.doi.org/10.1021/jp2069489)**, 12034 (2011).


74 U. G´ora, R. Podeszwa, W. Cencek, and K. Szalewicz, J. Chem. Phys. **[135](http://dx.doi.org/ http://dx.doi.org/10.1063/1.3664730)**, 224102 (2011).


75 T. B. Adler, G. Knizia, and H. J. Werner, J. Chem. Phys. **[127](http://dx.doi.org/10.1063/1.2817618)**, 221106 (2007).


76 K. A. Peterson, T. B. Adler, and H.-J. Werner, J. Chem. Phys. **[128](http://dx.doi.org/http://dx.doi.org/10.1063/1.2831537)**, 084102 (2008).


77 A. Tikhonov, in _Soviet Math. Dokl._, Vol. 5 (1963) pp. 1035–1038.


78 J. Behler, J. Chem. Phys. **[134](http://dx.doi.org/10.1063/1.3553717)**, 074106 (2011).


79 F. Chollet _et al._, “Keras,” `[https://github.com/keras-team/keras](https://github.com/keras-team/keras)` (2015).


80 Theano Development Team, arXiv e-prints **[abs/1605.02688](http://arxiv.org/abs/1605.02688)** (2016).


81 D. Nguyen and B. Widrow, in _[1990 IJCNN International Joint Conference on Neural](http://dx.doi.org/10.1109/IJCNN.1990.137819)_


_[Networks](http://dx.doi.org/10.1109/IJCNN.1990.137819)_, Vol. 3 (1990) pp. 21–26.


25


82 A. P. Bart´ok, M. C. Payne, R. Kondor, and G. Cs´anyi, Phys. Rev. Lett. **104**, 136403


(2010).


83 A. P. Bart´ok, R. Kondor, and G. Cs´anyi, Phys. Rev. B **87**, 184115 (2013).


84 A. Bart´ok-P´artay, S. Cereda, G. Cs´anyi, J. Kermode, I. Solt, W. Szlachta, C. V´arnai, and


S. Winfield, `[http://www.libatoms.org](http://www.libatoms.org)` .


85 S. De, A. P. Bart´ok, G. Cs´anyi, and M. Ceriotti, Phys. Chem. Chem. Phys. **18**, 13754


(2016).


86 A. P. Bartok, S. De, C. Poelking, N. Bernstein, J. Kermode, G. Csanyi, and M. Ceriotti,


Science Advances **3**, e1701816 (2017).


87 M. W. Mahoney and P. Drineas, Proc. Natl. Acad. Sci. USA **106**, 697 (2009).


88 S. De, F. Musil, T. Ingram, C. Baldauf, and M. Ceriotti, Journal of Cheminformatics **9**,


6 (2017).


89 D. J. Rosenkrantz, R. E. Stearns, and P. M. Lewis, II, SIAM Journal on Computing **6**,


563 (1977).


90 M. Ceriotti, G. A. Tribello, and M. Parrinello, Proc. Natl. Acad. Sci. USA **108**, 13023


(2011).


91 M. Ceriotti, G. A. Tribello, and M. Parrinello, J. Chem. Theory Comput. **9**, 1521 (2013).


92 F. Musil, S. De, J. Yang, J. E. Campbell, G. M. Day, and M. Ceriotti, Chemical Science


(2018).


93 G. R. Medders, A. W. G¨otz, M. A. Morales, and F. Paesani, J. Chem. Phys. **143**, 104102


(2015).


26


