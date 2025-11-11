_[Submitted to the Annals of Statistics](http://www.imstat.org/aos/)_
arXiv: `[arXiv:1605.00353](http://arxiv.org/abs/arXiv:1605.00353)`


**RATE-OPTIMAL PERTURBATION BOUNDS FOR**

**SINGULAR SUBSPACES WITH APPLICATIONS TO**

**HIGH-DIMENSIONAL STATISTICS**


By T. Tony Cai _[∗]_ and Anru Zhang


_University of Pennsylvania and University of Wisconsin-Madison_


Perturbation bounds for singular spaces, in particular Wedin’s
sin Θ theorem, are a fundamental tool in many fields including highdimensional statistics, machine learning, and applied mathematics.
In this paper, we establish separate perturbation bounds, measured
in both spectral and Frobenius sin Θ distances, for the left and right
singular subspaces. Lower bounds, which show that the individual
perturbation bounds are rate-optimal, are also given.
The new perturbation bounds are applicable to a wide range of
problems. In this paper, we consider in detail applications to low-rank
matrix denoising and singular space estimation, high-dimensional
clustering, and canonical correlation analysis (CCA). In particular,
separate matching upper and lower bounds are obtained for estimating the left and right singular spaces. To the best of our knowledge,
this is the first result that gives different optimal rates for the left
and right singular spaces under the same perturbation.


**1. Introduction.** Singular value decomposition (SVD) and spectral methods have been widely used in statistics, probability, machine learning, and
applied mathematics as well as many applications. Examples include lowrank matrix denoising (Shabalin and Nobel, 2013; Yang et al., 2014; Donoho
and Gavish, 2014), matrix completion (Cand`es and Recht, 2009; Cand`es and
Tao, 2010; Keshavan et al., 2010; Gross, 2011; Chatterjee, 2014), principle
component analysis (Anderson, 2003; Johnstone and Lu, 2009; Cai et al.,
2013, 2015b), canonical correlation analysis (Hotelling, 1936; Hardoon et al.,
2004; Gao et al., 2014, 2015), community detection (von Luxburg et al., 2008;
Rohe et al., 2011; Balakrishnan et al., 2011; Lei and Rinaldo, 2015). Specific
applications include collaborative filtering (the Netflix problem) (Goldberg
et al., 1992), multi-task learning (Argyriou et al., 2008), system identification (Liu and Vandenberghe, 2009), and sensor localization (Singer and


_∗_ The research of Tony Cai was supported in part by NSF Grant DMS-1208982 and
DMS-1403708, and NIH Grant R01 CA127334.
_MSC 2010 subject classifications:_ Primary 62H12, 62C20; secondary 62H25
_Keywords and phrases:_ canonical correlation analysis, clustering, high-dimensional
statistics, low-rank matrix denoising, perturbation bound, singular value decomposition,
sin Θ distances, spectral method


1


2 T. T. CAI AND A. ZHANG


Cucuringu, 2010; Candes and Plan, 2010), among many others. In addition,
the SVD is often used to find a “warm start” for more delicate iterative
algorithms, see, e.g., Cai et al. (2016); Sun and Luo (2015).
Perturbation bounds, which concern how the spectrum changes after a
small perturbation to a matrix, often play a critical role in the analysis of
the SVD and spectral methods. To be more specific, for an approximately
low-rank matrix _X_ and a perturbation matrix _Z_, it is crucial in many applications to understand how much the left or right singular spaces of _X_ and
_X_ + _Z_ differ from each other. This problem has been widely studied in the
literature (Davis and Kahan, 1970; Wedin, 1972; Weyl, 1912; Stewart, 1991,
2006; Yu et al., 2015). Among these results, the sin Θ theorems, established
by Davis and Kahan (1970) and Wedin (1972), have become fundamental
tools and are commonly used in applications. While Davis and Kahan (1970)
focused on eigenvectors of symmetric matrices, Wedin’s sin Θ theorem studies the more general singular vectors for asymmetric matrices and provides
a uniform perturbation bound for both the left and right singular spaces in
terms of the singular value gap and perturbation level.
Several generalizations and extensions have been made in different settings after the seminal work of Wedin (1972). For example, Vu (2011), Shabalin and Nobel (2013), O’Rourke et al. (2013), Wang (2015) considered the
rotations of singular vectors after random perturbations; Fan et al. (2016)
gave an _ℓ_ _∞_ eigenvector perturbation bound and used the result for robust
covariance estimation. See also Dopico (2000); Stewart (2006).
Despite its wide applicability, Wedin’s perturbation bound is not sufficiently precise for some analyses, as the bound is uniform for both the left
and right singular spaces. It clearly leads to sub-optimal result if the left
and right singular spaces change in different orders of magnitude after the
perturbation. In a range of applications, especially when the row and column dimensions of the matrix differ significantly, it is even possible that
one side of the singular space can be accurately recovered, while the other
side cannot. The numerical experiment given in Section 2.3 provides a good
illustration for this point. It can be seen from the experiment that the left
and right singular perturbation bounds behave distinctly when the row and
column dimensions are significantly different. Furthermore, for a range of
applications, the primary interest only lies in one of the singular spaces.
For example, in the analysis of bipartite network data, such as the Facebook user-public-page-subscription network, the interest is often focused on
grouping the public pages (or grouping the users). This is the case for many
clustering problems. See Section 4 for further discussions.
In this paper, we establish separate perturbation bounds for the left and


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 3


right singular subspaces. The bounds are measured in both the spectral and
Frobenius sin Θ distances, which are equivalent to several widely used losses
in the literature. We also derive lower bounds that are within a constant

factor of the corresponding upper bounds. These results together show that
the obtained perturbation bounds are rate-optimal.
The newly established perturbation bounds are applicable to a wide range
of problems in high-dimensional statistics. In this paper, we discuss in detail
the applications of the perturbation bounds to the following high-dimensional
statistical problems.


1. _Low-rank matrix denoising and singular space estimation:_ Suppose one
observes a low-rank matrix with random additive noise and wishes to

estimate the mean matrix or its left or right singular spaces. Such a
problem arises in many applications. We apply the obtained perturbation bounds to study this problem. Separate matching upper and
lower bounds are given for estimating the left and right singular spaces.
These results together establish the optimal rates of convergence. Our
analysis shows an interesting phenomenon that in some settings it is
possible to accurately estimate the left singular space but not the right
one and vice versa. To the best of our knowledge, this is the first result that gives different optimal rates for the left and right singular
spaces under the same perturbation. Another fact we observe is that
in certain class of low-rank matrices, one can stably recover the original matrix if and only if one can accurately recover both its left and
right singular spaces.
2. _High-dimensional clustering:_ Unsupervised learning is an important
problem in statistics and machine learning with a wide range of applications. We apply the perturbation bounds to the analysis of clustering for high-dimensional Gaussian mixtures. Particularly in a highdimensional two-class clustering setting, we propose a simple PCAbased clustering method and use the obtained perturbation bounds to
prove matching upper and lower bounds for the misclassification rates.
3. _Canonical correlation analysis (CCA):_ CCA is a commonly used tools
in multivariate analysis to identify and measure the associations among
two sets of random variables. The perturbation bounds are also applied to analyze CCA. Specifically, we develop sharper upper bounds
for estimating the left and right canonical correlation directions. To
the best of our knowledge, this is the first result that captures the
phenomenon that in some settings it is possible to accurately estimate
one side of canonical correlation directions but not the other side.


4 T. T. CAI AND A. ZHANG


In addition to these applications, the perturbation bounds can also be applied to the analysis of _community detection in bipartite networks, multi-_
_dimensional scaling, cross-covariance matrix estimation, and singular space_
_estimation for matrix completion_, and other problems to yield better results
than what are known in the literature. These applications demonstrate the
usefulness of the newly established perturbation bounds.
The rest of the paper is organized as follows. In Section 2, after basic notation and definitions are introduced, the perturbation bounds are presented
separately for the left and right singular subspaces. Both the upper bounds
and lower bounds are provided. We then apply the newly established perturbation bounds to low-rank matrix denoising and singular space estimation,
high-dimensional clustering, and canonical correlation analysis in Sections

–
3 5. Section 6 presents some numerical results and other potential applications are briefly discussed in Section 7. The main theorems are proved in
Section 8 and the proofs of some additional technical results are given in the
supplementary materials.


**2. Rate-Optimal Perturbation Bounds for Singular Subspaces.**
We establish in this section rate-optimal perturbation bounds for singular
subspaces. We begin with basic notation and definitions that will be used in
the rest of the paper.


2.1. _Notation and Definitions._ For _a, b ∈_ R, let _a_ _∧_ _b_ = min( _a, b_ ), _a_ _∨_ _b_ =
max( _a, b_ ). Let O _p,r_ = _{V ∈_ R _[p][×][r]_ : _V_ [⊺] _V_ = _I_ _r_ _}_ be the set of all _p×r_ orthonormal columns and write O _p_ for O _p,p_, the set of _p_ -dimensional orthogonal matrices. For a matrix _A ∈_ R _[p]_ [1] _[×][p]_ [2], write the SVD as _A_ = _U_ Σ _V_ [⊺], where Σ =
diag _{σ_ 1 ( _A_ ) _, σ_ 2 ( _A_ ) _, · · · }_ with the singular values _σ_ 1 ( _A_ ) _≥_ _σ_ 2 ( _A_ ) _≥· · · ≥_ 0 in
descending order. In particular, we use _σ_ min ( _A_ ) = _σ_ min( _p_ 1 _,p_ 2 ) ( _A_ ) _, σ_ max ( _A_ ) =
_σ_ 1 ( _A_ ) as the smallest and largest non-trivial singular values of _A_ . Several
matrix norms will be used in the paper: _∥A∥_ = _σ_ 1 ( _A_ ) is the spectral norm;

_∥A∥_ _F_ = _i_ _[σ]_ _i_ [2] [(] _[A]_ [) is the Frobenius norm; and] _[ ∥][A][∥]_ _[∗]_ [=][ �] _i_ _[σ]_ _[i]_ [(] _[A]_ [) is the]
~~��~~
nuclear norm. We denote P _A_ _∈_ R _[p]_ [1] _[×][p]_ [1] as the projection operator onto the
column space of _A_, which can be written as P _A_ = _A_ ( _A_ [⊺] _A_ ) _[†]_ _A_ [⊺] . Here ( _·_ ) _[†]_

represents the Moore-Penrose pseudoinverse. Given the SVD _A_ = _U_ Σ _V_ [⊺]

with Σ non-singular, a simpler form for P _A_ is P _A_ = _UU_ [⊺] . We adopt the R
convention to denote the submatrix: _A_ [ _a_ : _b,c_ : _d_ ] represents the _a_ -to- _b_ -th row,
_c_ -to- _d_ -th column of matrix _A_ ; we also use _A_ [ _a_ : _b,_ :] and _A_ [: _,c_ : _d_ ] to represent
_a_ -to- _b_ -th full rows of _A_ and _c_ -to- _d_ -th full columns of _A_, respectively. We use
_C, C_ 0 _, c, c_ 0 _, · · ·_ to denote generic constants, whose actual values may vary
from time to time.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 5


We use the _sinΘ distance_ to measure the difference between two _p × r_
orthogonal columns _V_ and _V_ [ˆ] . Suppose the singular values of _V_ [⊺] _V_ [ˆ] are _σ_ 1 _≥_
_σ_ 2 _≥· · · ≥_ _σ_ _r_ _≥_ 0. Then we call


Θ( _V,_ _V_ [ˆ] ) = diag(cos _[−]_ [1] ( _σ_ 1 ) _,_ cos _[−]_ [1] ( _σ_ 2 ) _, · · ·,_ cos _[−]_ [1] ( _σ_ _r_ ))


as the principle angles. A quantitative measure of distance between the
column spaces of _V_ and _V_ [ˆ] is then _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ or _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ . Some
more convenient characterizations and properties of the sin Θ distances will
be given in Lemma 1 in Section 8.1.


2.2. _Perturbation Upper Bounds and Lower Bounds._ We are now ready
to present the perturbation bounds for the singular subspaces. Let _X ∈_
R _[p]_ [1] _[×][p]_ [2] be an approximately low-rank matrix and let _Z ∈_ R _[p]_ [1] _[×][p]_ [2] be a “small”
perturbation matrix. Our goal is to provide separate and rate-sharp bounds
for the sin Θ distances between the left singular subspaces of _X_ and _X_ + _Z_
and between the right singular subspaces of _X_ and _X_ + _Z_ .
Suppose _X_ is approximately rank- _r_ with the SVD _X_ = _U_ Σ _V_ [⊺], where a
significant gap exists between _σ_ _r_ ( _X_ ) and _σ_ _r_ +1 ( _X_ ). The leading _r_ left and
right singular vectors of _X_ are of particular interest. We decompose _X_ as
follows,



Σ 1 0
(2.1) _X_ = � _U_ _U_ _⊥_ � _·_ � 0 Σ 2



_V_ ⊺

_·_ _V_ [⊺]
� � _⊥_



_,_
�



where _U ∈_ O _p_ 1 _,r_ _, V ∈_ O _p_ 2 _,r_, Σ 1 = diag( _σ_ 1 ( _X_ ) _, · · ·, σ_ _r_ ( _X_ )) _∈_ R _[r][×][r]_ _,_ Σ 2 =
diag( _σ_ _r_ +1 ( _X_ ) _, · · ·_ ) _∈_ R [(] _[p]_ [1] _[−][r]_ [)] _[×]_ [(] _[p]_ [2] _[−][r]_ [)], [ _U U_ _⊥_ ] _∈_ O _p_ 1 _,_ [ _V V_ _⊥_ ] _∈_ O _p_ 2 are orthogonal matrices.
Let _Z_ be a perturbation matrix and let _X_ [ˆ] = _X_ + _Z_ . Partition the SVD
of _X_ [ˆ] in the same way as in (2.1),



ˆ ˆ ˆ ˆΣ 1 0
(2.2) _X_ = _X_ + _Z_ = � _U_ _U_ _⊥_ � _·_
� 0 ˆΣ 2



_V_ ˆ ⊺
� _·_ � _V_ ˆ _⊥_ [⊺]



_,_
�



while _U,_ [ˆ] _U_ [ˆ] _⊥_ _,_ Σ [ˆ] 1 _,_ Σ [ˆ] 2 _,_ _V_ [ˆ] and _V_ [ˆ] _⊥_ have the same structures as _U, U_ _⊥_ _,_ Σ 1 _,_ Σ 2 _, V_,
and _V_ _⊥_ . Decompose the perturbation _Z_ into four blocks


(2.3) _Z_ = _Z_ 11 + _Z_ 12 + _Z_ 21 + _Z_ 22 _,_


where


_Z_ 11 = P _U_ _Z_ P _V_ _,_ _Z_ 21 = P _U_ _⊥_ _Z_ P _V_ _,_ _Z_ 12 = P _U_ _Z_ P _V_ _⊥_ _,_ _Z_ 22 = P _U_ _⊥_ _Z_ P _V_ _⊥_ _._


6 T. T. CAI AND A. ZHANG


Define
_z_ _ij_ := _∥Z_ _ij_ _∥_ for _i, j_ = 1 _,_ 2 _._


Theorem 1 below provides separate perturbation bounds for the left and
right singular subspaces in terms of both spectral and Frobenius sin Θ dis
tances.


Theorem 1 (Perturbation bounds for singular subspaces). _Let X,_ _X_ [ˆ] _,_
_and Z be given as_ (2.1) _-_ (2.3) _. Denote_


_α_ := _σ_ min ( _U_ [⊺] _XV_ [ˆ] ) _and_ _β_ := _∥U_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥][.]_


_If α_ [2] _> β_ [2] + _z_ 12 [2] _[∧]_ _[z]_ 21 [2] _[, then]_



_αz_ 12 + _βz_ 21
_∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥≤_ _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ 1 _,_



(2.4)

_∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ _F_ _≤_ _[α]_ _α_ _[∥]_ [2] _[Z]_ _−_ [12] _β_ _[∥]_ _[F]_ [2] [ +] _−_ _z_ _[β]_ 21 [2] _[∥][Z][∧]_ [21] _[z][∥]_ 12 [2] _[F]_ _∧_ _[√]_ _r._



_∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ _F_ _≤_ _[α][∥][Z]_ [12] _[∥]_ _[F]_ [ +] _[β][∥][Z]_ [21] _[∥]_ _[F]_



_αz_ 21 + _βz_ 12
_∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥≤_ _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ 1 _,_



(2.5)

_∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _F_ _≤_ _[α]_ _α_ _[∥]_ [2] _[Z]_ _−_ [21] _β_ _[∥]_ _[F]_ [2] [ +] _−_ _z_ _[β]_ 21 [2] _[∥][Z][∧]_ [12] _[z][∥]_ 12 [2] _[F]_ _∧_ _[√]_ _r._



_∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _F_ _≤_ _[α][∥][Z]_ [21] _[∥]_ _[F]_ [ +] _[β][∥][Z]_ [12] _[∥]_ _[F]_



One can see the respective effects of the perturbation on the left and right
singular spaces. In particularly, if _z_ 12 _≥_ _z_ 21 (which is typically the case when
_p_ 2 _≫_ _p_ 1 ), then Theorem 3 gives a smaller bound for _∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ than for
_∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ .


Remark 1. The assumption _α_ [2] _> β_ [2] + _z_ 12 [2] _[∧]_ _[z]_ 21 [2] [in Theorem][ 1][ ensures]
that the amplitude of _U_ [⊺] _XV_ [ˆ] = Σ 1 + _U_ [⊺] _ZV_ dominates those of _U_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ [=]
Σ 2 + _U_ _⊥_ [⊺] _[ZV]_ _[⊥]_ [,] _[ U]_ [⊺] _[ZV]_ _[⊥]_ [, and] _[ U]_ _⊥_ [⊺] _[ZV]_ [, so that ˆ] _[U]_ [ and ˆ] _[V]_ [ can be close to] _[ U]_
and _V_, respectively. This assumption essentially means that there exists
significant gap between the _r_ -th and ( _r_ + 1)-st singular values of _X_ and the
perturbation term _Z_ is bounded. We will show in Theorem 2 that _U_ [ˆ] and _V_ [ˆ]
might be inconsistent when this condition fails to hold.


Remark 2. Consider the setting where _X ∈_ R _[p]_ [1] _[×][p]_ [2] is a fixed rank- _r_
matrix with _r ≤_ _p_ 1 _≪_ _p_ 2, and _Z ∈_ R _[p]_ [1] _[×][p]_ [2] is a random matrix with i.i.d.
standard normal entries. In this case, _Z_ 11 _, Z_ 12 _, Z_ 21 _,_ and _Z_ 22 are all i.i.d.
standard normal matrices of dimensions _r × r_, _r ×_ ( _p_ 2 _−_ _r_ ), ( _p_ 1 _−_ _r_ ) _× r_,
and ( _p_ 1 _−_ _r_ ) _×_ ( _p_ 2 _−_ _r_ ), respectively. By random matrix theory (see, e.g.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 7


Vershynin (2012); Tao (2012)), _α ≥_ _σ_ _r_ ( _X_ ) _−∥Z_ 11 _∥≥_ _σ_ _r_ ( _X_ ) _−C_ ( _[√]_ ~~_p_~~ 1 + _[√]_ ~~_p_~~ 2 ~~)~~,
_β ≤_ _C_ ( _[√]_ ~~_p_~~ 1 + _[√]_ ~~_p_~~ 2 ~~)~~, _z_ 12 _≤_ _C_ _[√]_ ~~_p_~~ 2 ~~_,_~~ and _z_ 21 _≤_ _C_ _[√]_ ~~_p_~~ 1 for some constant _C >_ 0
with high probability. When _σ_ _r_ ( _X_ ) _≥_ _C_ gap _p_ 2 _/_ _[√]_ ~~_p_~~ 1 for some large constant
_C_ gap, Theorem 3 immediately implies



_∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥≤_ _[C][√]_ ~~_[p]_~~ [2]




_[C][√]_ ~~_[p]_~~ [2] _∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥≤_ _[C][√]_ ~~_[p]_~~ [1]

_σ_ _r_ ( _X_ ) _[,]_ _σ_ _r_ ( _X_



_σ_ _r_ ( _X_ ) _[.]_



Further discussions on perturbation bounds for general sub-Gaussian perturbation matrix with matching lower bounds with be given in Section 3.


Theorem 1 gives upper bounds for the perturbation effects. We now establish lower bounds for the differences as measured by the sin Θ distances.
Theorem 2 first states that _U_ [ˆ] and _V_ [ˆ] might be inconsistent when the condition _α_ [2] _> β_ [2] + _z_ 12 [2] _[∧][z]_ 21 [2] [fails to hold, and then provides the lower bounds that]
match those in (2.11) and (2.12), proving that the results given in Theorem
1 is essentially sharp. Theorem 2 also provides the worst-case matrix pair
( _X, Z_ ) that nearly achieves the supremum in (2.9) and (2.7). The matrix pair
shows where the lower bound is “close” to the upper bound, which is useful
in understanding the fundamentals about singular subspace perturbations.
Before stating the lower bounds, we define the following class of ( _X, Z_ )
pairs of _p_ 1 _× p_ 2 matrices and perturbations,


_F_ _r,α,β,z_ 21 _,z_ 12 = �( _X, Z_ ) : _X, U, V_ [ˆ] are given as (2.1) and (2.2) _,_

(2.6)
_σ_ min ( _U_ [⊺] _XV_ [ˆ] ) _≥_ _α, ∥U_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥≤]_ _[β,][ ∥][Z]_ [12] _[∥≤]_ _[z]_ [12] _[,][ ∥][Z]_ [21] _[∥≤]_ _[z]_ [21] � _._


In addition, we also define


˜ ˜
_G_ _α,β,z_ 21 _,z_ 12 _,z_ ˜ 21 _,z_ ˜ 12 = �( _X, Z_ ) : _∥Z_ 21 _∥_ _F_ _≤_ _z_ 21 _, ∥Z_ 12 _∥_ _F_ _≤_ _z_ 12 _,_

(2.7)
( _X, Z_ ) _∈F_ _r,α,β,z_ 21 _,z_ 12 � _._


Theorem 2 (Perturbation Lower Bound). _If α_ [2] _≤_ _β_ [2] + _z_ 12 [2] _[∧]_ _[z]_ 21 [2] _[and]_
_r ≤_ _[p]_ [1] _[∧]_ 2 _[p]_ [2] _, then_



1
(2.8) inf ˜ sup _∥_ sin Θ( _V,_ _V_ [˜] ) _∥≥_
_V_ ( _X,Z_ ) _∈F_ 2 ~~_√_~~



2 _[.]_




_• Provided that α_ [2] _> β_ [2] + _z_ 12 [2] [+] _[ z]_ 21 [2] _[,][ r][ ≤]_ _[p]_ [1] _[∧]_ 2 _[p]_ [2] _we have the following lower_

_bound for all estimate_ _V_ [˜] _∈_ _O_ _p_ 2 _×r_ _based on the observations_ _X_ [ˆ] _,_



1
(2.9) inf ˜ sup _∥_ sin Θ( _V,_ _V_ [˜] ) _∥≥_
_V_ ( _X,Z_ ) _∈F_ 8 ~~_√_~~ 10



_αz_ 12 + _βz_ 21
_∧_ 1 _._
� _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] �


8 T. T. CAI AND A. ZHANG


_In particular, if X_ = _αUV_ [⊺] + _βU_ _⊥_ _V_ _⊥_ [⊺] _[and][ Z]_ [ =] _[ z]_ [12] _[UV]_ [ ⊺] _⊥_ [+] _[ z]_ [21] _[U]_ _[⊥]_ _[V]_ [ ⊺] _[, then]_
( _X, Z_ ) _∈F and_



1

~~_√_~~ 10



_αz_ 12 + _βz_ 21 _αz_ 12 + _βz_ 21
� _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ 1� _≤∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥≤_ � _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ 1� _,_



_when_ _U,_ [ˆ] _V_ [ˆ] _are the leading r left and right singular vectors of_ _X_ [ˆ] _._

_• Provided that α_ [2] _> β_ [2] + _z_ 12 [2] [+] _[ z]_ 21 [2] _[,]_ [ ˜] _[z]_ 21 [2] _[≤]_ _[rz]_ 21 [2] _[,]_ [ ˜] _[z]_ 12 [2] _[≤]_ _[rz]_ 12 [2] _[,][ r][ ≤]_ _[p]_ [1] _[∧]_ 2 _[p]_ [2] _, we_

_have the following lower bound for all estimator_ _V_ [˜] 1 _∈_ O _p_ 2 _×r_ _based on the_
_observations_ _X_ [ˆ] _,_



1
(2.10) inf ˜ sup _∥_ sin Θ( _V,_ _V_ [˜] ) _∥_ _F_ _≥_
_V_ 1 ( _X,Z_ ) _∈G_ 8 ~~_√_~~ 10



_αz_ ˜ 12 + _βz_ ˜ 21
_∧_ _[√]_ _r_ _._
� _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] �



_In particular, if X_ = _αUV_ [⊺] + _βU_ _⊥_ _V_ _⊥_ [⊺] _[,][ Z]_ [ = ˜] _[z]_ [12] _[UV]_ [ ⊺] _⊥_ [+ ˜] _[z]_ [21] _[U]_ _[⊥]_ _[V]_ [ ⊺] _[, then]_
( _X, Z_ ) _∈G and_



1

~~_√_~~ 10



_αz_ ˜ 12 + _βz_ ˜ 21 _αz_ ˜ 12 + _βz_ ˜ 21
� _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ _[√]_ _r_ � _≤∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥≤_ � _α_ [2] _−_ _β_ [2] _−_ _z_ 21 [2] _[∧]_ _[z]_ 12 [2] _∧_ _[√]_ _r_ � _,_



_Xwhere_ ˆ _._ _U,_ [ˆ] _V_ [ˆ] _are respectively the leading r left and right singular vectors of_


The following Proposition 1, which provides upper bounds for the sin Θ
distances between leading singular vectors of a matrix _A_ and arbitrary orthogonal columns _W_, can be viewed as another version of Theorem 1. For
some applications, applying Proposition 1 might be more convenient than
using Theorem 1 directly.


Proposition 1. _Suppose A ∈_ R _[p]_ [1] _[×][p]_ [2] _,_ _V_ [˜] = [ _V V_ _⊥_ ] _∈_ O _p_ 2 _are right_
_singular vectors of A, V ∈_ O _p_ 2 _,r_ _, V_ _⊥_ _∈_ O _p_ 2 _,p_ 2 _−r_ _correspond to the first r_
_and last_ ( _p_ 2 _−_ _r_ ) _singular vectors respectively._ _W_ [˜] = [ _W W_ _⊥_ ] _∈_ O _p_ 2 _is any_
_orthogonal matrix with W ∈_ O _p_ 2 _,r_ _, W_ _⊥_ _∈_ O _p_ 2 _,p_ 2 _−r_ _. Given that σ_ _r_ ( _AW_ ) _>_
_σ_ _r_ +1 ( _A_ ) _, we have_


(2.11) _∥_ sin Θ( _V, W_ ) _∥≤_ _[σ]_ _[r]_ [(] _[AW]_ [)] _[∥]_ [P] [(] _[AW]_ [)] _[AW]_ _[⊥]_ _[∥]_ _∧_ 1 _,_

_σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)]


(2.12) _∥_ sin Θ( _V, W_ ) _∥_ _F_ _≤_ _[σ]_ _[r]_ [(] _[AW]_ [)] _[∥]_ [P] [(] _[AW]_ [)] _[AW]_ _[⊥]_ _[∥]_ _[F]_ _∧_ _[√]_ _r._

_σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)]


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 9


It is also of practical interest to provide perturbation bounds for a given
subset of singular vectors and in particular for a given singular vector. The
following Corollary 1 provides the one-sided perturbation bound for _U_ [ˆ] [: _,i_ : _j_ ]
and _V_ [ˆ] [: _,i_ : _j_ ] when there are significant gaps between the ( _i −_ 1)-st and _i_ -th
and between the _j_ -th and ( _j_ + 1)-st singular values and the perturbation is
bounded. Particularly when _i_ = _j_, Corollary 1 provides the upper bound for
the perturbation of the _i_ -th left and right singular vectors of _X_ [ˆ], ˆ _u_ _i_ and ˆ _v_ _i_ .


Corollary 1 (Perturbation bounds for individual singular vectors).
_Suppose X,_ _X_ [ˆ] _, and Z are given as_ (2.1) _-_ (2.3) _. For any k ≥_ 1 _, let U_ ( _k_ ) =
_U_ [: _,_ 1: _k_ ] _∈_ O _p_ 1 _,k_ _, V_ ( _k_ ) = _V_ [: _,_ 1: _k_ ] _∈_ O _p_ 2 _,k_ _, and U_ ( _k_ ) _⊥_ _∈_ O _p_ 1 _,p_ 1 _−k_ _, V_ ( _k_ ) _⊥_ _∈_ O _p_ 2 _,p_ 2 _−k_
_be the orthogonal complements. Denote_


_α_ [(] _[k]_ [)] = _σ_ min ( _U_ ( [⊺] _k_ ) _[XV]_ [ˆ] [(] _[k]_ [)] [)] _[,]_ _β_ [(] _[k]_ [)] = _∥U_ ( [⊺] _k_ ) _⊥_ _[XV]_ [ˆ] [(] _[k]_ [)] _[⊥]_ _[∥][,]_


⊺ ⊺
_z_ 12 [(] _[k]_ [)] [=] ��� _U_ ( _k_ ) _[ZV]_ [(] _[k]_ [)] _[⊥]_ ��� _,_ _z_ 21 [(] _[k]_ [)] [=] ��� _U_ ( _k_ ) _⊥_ _[ZV]_ [(] _[k]_ [)] ��� _,_

_for k_ = 1 _, . . ., p_ 1 _∧_ _p_ 2 _. We further define α_ [(0)] = _∞, β_ [(0)] = _∥X_ [ˆ] _∥, z_ 12 [(0)] [=] _[ z]_ 21 [(0)] [=]
0 _. For_ 1 _≤_ _i ≤_ _j ≤_ _p_ 1 _∧_ _p_ 2 _, provided that_ ( _α_ [(] _[i][−]_ [1)] ) [2] _>_ ( _β_ [(] _[i][−]_ [1)] ) [2] + ( _z_ 12 [(] _[i][−]_ [1)] ) [2] _∧_
( _z_ 21 [(] _[i][−]_ [1)] ) [2] _and_ ( _α_ [(] _[j]_ [)] ) [2] _>_ ( _β_ [(] _[j]_ [)] ) [2] + ( _z_ 12 [(] _[j]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 21 [(] _[j]_ [)] [)] [2] _[, we have]_



2 []





















1 _/_ 2



_∧_ 1 _,_



sin Θ( ˆ _V_ [: _,i_ : _j_ ] _, V_ [: _,i_ : _j_ ] ) _≤_
��� ���


sin Θ( ˆ _U_ [: _,i_ : _j_ ] _, U_ [: _,i_ : _j_ ] ) _≤_
��� ���





















�













_k∈{i−_ 1 _,j}_



_α_ [(] _[k]_ [)] _z_ [(] _[k]_ [)]
12 [+] _[ β]_ [(] _[k]_ [)] _[z]_ 21 [(] _[k]_ [)]
� �

( _α_ [(] _[k]_ [)] ) [2] _−_ ( _β_ [(] _[k]_ [)] ) [2] _−_ ( _z_ 21 [(] _[k]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 12 [(] _[k]_ [)] [)] [2]



2 []











1 _/_ 2









_∧_ 1 _._



�













_k∈{i−_ 1 _,j}_



_α_ [(] _[k]_ [)] _z_ [(] _[k]_ [)]
21 [+] _[ β]_ [(] _[k]_ [)] _[z]_ 12 [(] _[k]_ [)]
� �

( _α_ [(] _[k]_ [)] ) [2] _−_ ( _β_ [(] _[k]_ [)] ) [2] _−_ ( _z_ 21 [(] _[k]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 12 [(] _[k]_ [)] [)] [2]



_In particular, for any integer_ 1 _≤_ _i ≤_ _p_ 1 _∧_ _p_ 2 _, if_ ( _α_ [(] _[i][−]_ [1)] ) [2] _>_ ( _β_ [(] _[i][−]_ [1)] ) [2] +
( _z_ 12 [(] _[i][−]_ [1)] ) [2] _∧_ ( _z_ 21 [(] _[i][−]_ [1)] ) [2] _and_ ( _α_ [(] _[i]_ [)] ) [2] _>_ ( _β_ [(] _[i]_ [)] ) [2] + ( _z_ 12 [(] _[i]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 21 [(] _[i]_ [)] [)] [2] _[,][ u]_ _[i]_ _[,]_ [ ˆ] _[u]_ _[i]_ _[, v]_ _[i]_ _[,]_ [ ˆ] _[v]_ _[i]_ _[, i.e.]_
_the i-th singular vectors of X and_ _X_ [ˆ] _, are different by_









2 []





















1 _/_ 2



~~�~~ 1 _−_ ( _v_ _i_ [⊺] _[v]_ [ˆ] _[i]_ [)] [2] _[ ≤]_












_i_
�



_k_ = _i−_ 1









_∧_ 1 _,_


_∧_ 1 _._



_α_ [(] _[k]_ [)] _z_ [(] _[k]_ [)]
12 [+] _[ β]_ [(] _[k]_ [)] _[z]_ 21 [(] _[k]_ [)]
� �

( _α_ [(] _[k]_ [)] ) [2] _−_ ( _β_ [(] _[k]_ [)] ) [2] _−_ ( _z_ 21 [(] _[k]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 12 [(] _[k]_ [)] [)] [2]


_α_ [(] _[k]_ [)] _z_ [(] _[k]_ [)]
21 [+] _[ β]_ [(] _[k]_ [)] _[z]_ 12 [(] _[k]_ [)]
� �

( _α_ [(] _[k]_ [)] ) [2] _−_ ( _β_ [(] _[k]_ [)] ) [2] _−_ ( _z_ 21 [(] _[k]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 12 [(] _[k]_ [)] [)] [2]



2 []











1 _/_ 2



�1 _−_ ( _u_ [⊺] _i_ _[u]_ [ˆ] _[i]_ [)] [2] _[ ≤]_












_i_
�



_k_ = _i−_ 1








10 T. T. CAI AND A. ZHANG


Remark 3. The upper bound given in Corollary 1 is rate-optimal over
the following set of ( _X, Z_ ) pairs,


_H_ _α_ ( _i−_ 1) _,β_ ( _i−_ 1) _,z_ 12( _i−_ 1) _,z_ 21 [(] _[i][−]_ [1)] _,α_ [(] _[j]_ [)] _,β_ [(] _[j]_ [)] _,z_ 12 [(] _[j]_ [)] _[,z]_ 21 [(] _[j]_ [)]



=



_σ_ min ( _U_ ( [⊺] _k_ ) _[XV]_ [ˆ] [(] _[k]_ [)] [)] _[ ≥]_ _[α]_ [(] _[k]_ [)] _[,][ ∥][U]_ ( [⊺] _k_ ) _⊥_ _[XV]_ [ˆ] [(] _[k]_ [)] _[⊥]_ _[∥≤]_ _[β]_ [(] _[k]_ [)] _[,]_
( _X, Z_ ) : ⊺ ( _k_ ) ⊺ ( _k_ ) _, k ∈{i −_ 1 _, j}_

� ��� _U_ ( _k_ ) _[ZV]_ [(] _[k]_ [)] _[⊥]_ ��� _≤_ _z_ 12 _[,]_ ��� _U_ ( _k_ ) _⊥_ _[ZV]_ [(] _[k]_ [)] ��� _≤_ _z_ 21 �



_._



The detailed analysis can be carried out similarly to the one for Theorem 2.


2.3. _Comparisons with Wedin’s sin_ **Θ** _Theorem._ Theorems 1 and 2 together establish separate rate-optimal perturbation bounds for the left and
right singular subspaces. We now compare the results with the well-known
Wedin’s sin Θ Theorem, which gives uniform upper bounds for the singular
subspaces on both sides. Specifically, using the same notation as in Section
2.2, Wedin’s sin Θ Theorem states that if _σ_ min (Σ [ˆ] 1 ) _−_ _σ_ max (Σ 2 ) = _δ >_ 0, then


max _∥ZV_ [ˆ] _∥, ∥U_ [ˆ] [⊺] _Z∥_
� �
max _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥, ∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _≤_ _,_
� � _δ_


max _∥ZV_ [ˆ] _∥_ _F_ _, ∥U_ [ˆ] [⊺] _Z∥_ _F_
� �
max _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ _F_ _, ∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _F_ _≤_ _._
� � _δ_


When _X_ are _Z_ are symmetric, Theorem 1, Proposition 1, and Wedin’s sin Θ
theorem provide similar upper bound for singular subspace perturbation.
As mentioned in the introduction, the uniform bound on both left and
right singular subspaces in Wedin’s sin Θ Theorem might be sub-optimal
in some cases when _X_ or _Z_ are asymmetric. For example, in the setting
discussed in Remark 2, applying Wedin’s theorem leads to


_[√]_ ~~_[p]_~~ [2] ~~_[}]_~~
max _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥, ∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _≤_ _[C]_ [ max] _[{√]_ ~~_[p]_~~ [1] ~~_[,]_~~ _,_
� � _σ_ _r_ ( _X_ )


which is sub-optimal for _∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ if _p_ 2 _≫_ _p_ 1 .


**3. Low-rank Matrix Denoising and Singular-Space Estimation.**
In this section, we apply the perturbation bounds given in Theorem 1 for
low-rank matrix denoising. It can be seen that the new perturbation bounds
are particularly powerful when the matrix dimensions differ significantly. We
also establish a matching lower bound for low-rank matrix denoising which
shows that the results are rate-optimal.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 11


As mentioned in the introduction, accurate recovery of a low-rank matrix based on noisy observations has a wide range of applications, including
magnetic resonance imaging (MRI) and relaxometry. See, e.g., Candes et al.
(2013); Shabalin and Nobel (2013) and the reference therein. This problem is also important in the context of dimensional reduction. Suppose one
observes a low-rank matrix with additive noise,


_Y_ = _X_ + _Z,_


where _X_ = _U_ Σ _V_ [⊺] _∈_ R _[p]_ [1] _[×][p]_ [2] is a low-rank matrix with _U ∈_ O _p_ 1 _,r_ _, V ∈_ O _p_ 2 _,r_ _,_
and Σ = diag _{σ_ 1 ( _X_ ) _, . . ., σ_ _r_ ( _X_ ) _} ∈_ R _[r][×][r]_, and _Z ∈_ R _[p]_ [1] _[×][p]_ [2] is an i.i.d. meanzero sub-Gaussian matrix. The goal is to estimate the underlying low-rank
matrix _X_ or its singular values or singular vectors.
This problem has been actively studied. For example, Bura and Pfeiffer
(2008), Capitaine et al. (2009), Benaych-Georges and Nadakuditi (2012),
Shabalin and Nobel (2013) focused on the asymptotic distributions of single
singular value and vector when _p_ 1, _p_ 2 and the singular values grows proportionally. Vu (2011) discussed the squared matrix perturbed by i.i.d Bernoulli
matrix and derived an upper bound on the rotation angle of singular vectors. O’Rourke et al. (2013) further generalized the results in Vu (2011)
and proposed a trio-concentrated random matrix perturbation setting. Recently, Wang (2015) provides the _ℓ_ _∞_ distance under relatively complicated
settings when matrix is perturbed by i.i.d. Gaussian noise. Donoho and
Gavish (2014); Gavish and Donoho (2014); Candes et al. (2013) studied the
algorithm for recovering _X_, where singular value thresholding (SVT) and
hard singular value thresholding (HSVT), stated as



_,_
�



_SV T_ ( _Y_ ) _λ_ = arg min
_X_



1
� 2



2 _[∥][Y][ −]_ _[X][∥]_ _F_ [2] [+] _[ λ][∥][X][∥]_ _[∗]_



(3.1)

1

_HSV T_ ( _Y_ ) _λ_ = arg min _F_ [+] _[ λ]_ [rank(] _[X]_ [)]
_X_ � 2 _[∥][Y][ −]_ _[X][∥]_ [2] �



_HSV T_ ( _Y_ ) _λ_ = arg min
_X_



1
� 2



were proposed. The optimal choice of thresholding level _λ_ _[∗]_ was further discussed in Donoho and Gavish (2014) and Gavish and Donoho (2014). Especially, Donoho and Gavish (2014) proves that



inf _X_ ˆ _X∈_ sup R _[p]_ [1] _[×][p]_ [2]
rank( _X_ ) _≤r_



E _X_ ˆ _−_ _X_ 2
��� ��� _F_ _[≍]_ _[r]_ [(] _[p]_ [1] [ +] _[ p]_ [2] [)] _[,]_



when _Z_ is i.i.d. standard normal random matrix. If one defines the class
of rank- _r_ matrices, _F_ _r,t_ = _{X ∈_ R _[p]_ [1] _[×][p]_ [2] : _σ_ _r_ ( _X_ ) _≥_ _t}_, the following upper


12 T. T. CAI AND A. ZHANG


bound for the relative error is an immediate consequence of our results,


_[p]_ [2] [)]
(3.2) sup E _[∥][X]_ [ˆ] _[ −]_ _[X][∥]_ _F_ [2] _≤_ _[C]_ [(] _[p]_ [1] [ +] _∧_ 1 _,_
_X∈F_ _r,t_ _∥X∥_ [2] _F_ _t_ [2]


where
ˆ _SV T_ ( _Y_ ) _λ_ _∗_ _,_ if _t_ [2] _≥_ _C_ ( _p_ 1 + _p_ 2 ) _,_
_X_ =
� 0 _,_ if _t_ [2] _< C_ ( _p_ 1 + _p_ 2 ) _._


In the following discussion, we assume that the entries of _Z_ = ( _Z_ _ij_ ) have
unit variance (which can be simply achieved by normalization). To be more
precise, we define the class of distributions _G_ _τ_ for some _τ >_ 0 as follows.


(3.3) If _Z ∼G_ _τ_ _,_ then E _Z_ = 0 _,_ Var( _Z_ ) = 1 _,_ E exp( _tZ_ ) _≤_ exp( _τt_ ) _, ∀t ∈_ R _._


The distribution of the entries of _Z_, _Z_ _ij_, is assumed to satisfy


_iid_
_Z_ _ij_ _∼G_ _τ_ _,_ 1 _≤_ _i ≤_ _p_ 1 _,_ 1 _≤_ _j ≤_ _p_ 2 _._


Suppose _U_ [ˆ] and _V_ [ˆ] are respectively the first _r_ left and right singular vectors
of _Y_ . We use _U_ [ˆ] and _V_ [ˆ] as the estimators of _U_ and _V_ respectively. Then the
perturbation bounds for singular spaces yield the following results.


Theorem 3 (Upper Bound). _Suppose X_ = _U_ Σ _V_ [⊺] _∈_ R _[p]_ [1] _[×][p]_ [2] _is of rank-r._
_There exists constants C >_ 0 _that only depends on τ such that_


_r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
E _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ [2] _≤_ _[C][p]_ [2] [(] _[σ]_ [2] _∧_ 1 _,_
_σ_ _r_ [4] ( _X_ )

_r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
E _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ _F_ [2] _[≤]_ _[C][p]_ [2] _[r]_ [(] _[σ]_ [2] _∧_ _r._
_σ_ _r_ [4] ( _X_ )


_r_ [(] _[X]_ [)][ +] _[p]_ [2] [)]
E _∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ [2] _≤_ _[C][p]_ [1] [(] _[σ]_ [2] _∧_ 1 _,_
_σ_ _r_ [4] ( _X_ )

_r_ [(] _[X]_ [)][ +] _[p]_ [2] [)]
E _∥_ sin Θ( _U,_ _U_ [ˆ] ) _∥_ _F_ [2] _[≤]_ _[C][p]_ [1] _[r]_ [(] _[σ]_ [2] _∧_ _r._
_σ_ _r_ [4] ( _X_ )


Theorem 3 provides a non-trivial perturbation upper bound for sin Θ( _V,_ _V_ [ˆ] )
(or sin Θ( _U,_ _U_ [ˆ] )) if there exists a constant _C_ gap _>_ 0 such that


1
_σ_ _r_ [2] _[≥]_ _[C]_ [gap] [((] _[p]_ [1] _[p]_ [2] [)] 2 + _p_ 2 )


1
(or _σ_ _r_ [2] _[≥]_ _[C]_ [gap] [((] _[p]_ [1] _[p]_ [2] [)] 2 + _p_ 1 )). In contrast, Wedin’s sin Θ Theorem requires
the singular value gap _σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] [(] _[p]_ [1] [+] _[p]_ [2] [), which shows the power of the]
proposed unilateral perturbation bound.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 13


Furthermore, the upper bounds in Theorem 3 are rate-sharp in the sense
that the following matching lower bounds hold. To the best of our knowledge,
this is the first result that gives different optimal rates for the left and right
singular spaces under the same perturbation.


Theorem 4 (Lower Bound). _Define the following class of low-rank ma-_

_trices_


(3.4) _F_ _r,t_ = � _X ∈_ R _[p]_ [1] _[×][p]_ [2] : _σ_ _r_ ( _X_ ) _≥_ _t_ � _._



_If r ≤_ _[p]_ [1]



16 _[p]_ [1] _[∧]_ _[p]_ 2 [2]




[2]

2 _[, then]_



_p_ 2 ( _t_ 2 + _p_ 1 )
inf ˜ sup E _∥_ sin Θ( _V,_ _V_ [˜] ) _∥_ [2] _≥_ _c_ _∧_ 1 _,_
_V_ _X∈F_ _r,t_ � _t_ [4] �



2
_p_ 2 _r_ ( _t_ + _p_ 1 )
inf ˜ sup E _∥_ sin Θ( _V,_ _V_ [˜] ) _∥_ _F_ [2] _[≥]_ _[c]_ _∧_ _r_ _._
_V_ _X∈F_ _r,t_ � _t_ [4] �



_p_ 1 ( _t_ 2 + _p_ 2 )
inf ˜ sup E _∥_ sin Θ( _U,_ _U_ [˜] ) _∥_ [2] _≥_ _c_ _∧_ 1 _,_
_V_ _X∈F_ _r,t_ � _t_ [4] �



2
_p_ 1 _r_ ( _t_ + _p_ 2 )
inf ˜ sup E _∥_ sin Θ( _U,_ _U_ [˜] ) _∥_ _F_ [2] _[≥]_ _[c]_ _∧_ _r_ _._
_V_ _X∈F_ _r,t_ � _t_ [4] �



Remark 4. Using similar technical arguments, we can also obtain the
following lower bound for estimating the low-rank matrix _X_ over _F_ _r,t_ under
a relative error loss,


(3.5) inf ˜ sup E _[∥][X]_ [˜] _[ −]_ _[X][∥]_ _F_ [2] _≥_ _c_ _p_ 1 + _p_ 2 _∧_ 1 _._
_X_ _X∈F_ _r,t_ _∥X∥_ [2] _F_ � _t_ [2] �


Combining equations (3.2) and (3.5) yields the minimax optimal rate for
relative error in matrix denoising,


_F_ _p_ 1 + _p_ 2
inf _X_ ˜ _X_ sup _∈F_ _r,t_ E _[∥][X]_ [˜] _∥_ _[ −]_ _X∥_ _[X]_ [2] _F_ _[∥]_ [2] _≍_ _c_ � _t_ [2] _∧_ 1� _._


An interesting fact is that



+ _p_ 2 _p_ 2 ( _t_ 2 + _p_ 1 )

_∧_ 1 _≍_ _c_
_t_ [2] � � _t_ [4]



+ _p_ 2 )

_t_ [4] _∧_ 1� _,_



_p_ 1 + _p_ 2
_c_
� _t_ [2]



+ _p_ 1 ) _p_ 1 ( _t_ 2 + _p_ 2 )

_∧_ 1 + _c_
_t_ [4] � � _t_ [4]


14 T. T. CAI AND A. ZHANG


which yields directly


_F_
inf ˜ sup E _∥_ sin Θ( _U,_ _U_ [˜] ) _∥_ [2] +inf ˜ sup E _∥_ sin Θ( _V,_ _V_ [˜] ) _∥_ [2] _≍_ inf ˜ sup E _[∥][X]_ [˜] _[ −]_ _[X][∥]_ [2] _._
_U_ _X∈F_ _r,t_ _V_ _X∈F_ _r,t_ _X_ _X∈F_ _r,t_ _∥X∥_ _F_


Hence, for the class of _F_ _r,t_, one can stably recover _X_ in relative Frobenius
norm loss if and only if one can stably recover both _U_ and _V_ in spectral
sin Θ norm.


Another interesting aspect of Theorems 3 and 4 is that, when _p_ 1 _≫_ _p_ 2,
1
( _p_ 1 _p_ 2 ) 2 _≪_ _t_ [2] _≪_ _p_ 1, there is no stable algorithm for recovery of either the left
singular space _U_ or whole matrix _X_ in the sense that there exists uniform
constant _c >_ 0 such that


2
inf ˜ sup E sin Θ( _U,_ ˜ _U_ ) _≥_ _c,_ inf ˜ sup E _[∥][X]_ [˜] _[ −]_ _[X][∥]_ _F_ [2] _≥_ _c._
_U_ _X∈F_ _r,t_ ��� ��� _X_ _X∈F_ _r,t_ _∥X∥_ [2] _F_


In fact, for _X_ = _tUV_ [⊺] _∈F_ _r,t_, if we simply apply SVT or HSVT algorithms
with optimal choice of _λ_ as proposed in Donoho and Gavish (2014) and Gavish and Donoho (2014), with high probability, _SV T_ _λ_ ( _X_ [ˆ] ) = _HSV T_ _λ_ ( _X_ [ˆ] ) = 0 _._
On the other hand, the spectral method does provide a consistent recovery
of the right singular-space according to Theorem 3.


2
E sin Θ( _V,_ ˆ _V_ ) _→_ 0 _._
��� ���


This phenomenon is well demonstrated by the simulation result (Table 6)
provided in Section 1.


**4. High-dimensional Clustering.** Unsupervised learning, or clustering, is an ubiquitous problem in statistics and machine learning (Hastie
et al., 2009). The perturbation bounds given in Theorem 1 as well as the
results in Theorems 3 and 4 have a direct implication in high-dimensional
clustering. Suppose the locations of _n_ points, _X_ = [ _X_ 1 _· · · X_ _n_ ] _∈_ R _[p][×][n]_, which
lie in a certain _r_ -dimensional subspace _S_ in R _[p]_, are observed with noise


_Y_ _i_ = _X_ _i_ + _ε_ _i_ _,_ _i_ = 1 _, · · ·, n._


Here _X_ _i_ _∈S ⊆_ R _[p]_ are fixed coordinates, _ε_ _i_ _∈_ R _[p]_ are random noises. The goal
is to cluster the observations _Y_ . Let the SVD of _X_ be given by _X_ = _U_ Σ _V_ [⊺],
where _U ∈_ O _p,r_, _V ∈_ O _n,r_, and Σ _∈_ R _[r][×][r]_ . When _p ≫_ _n_, applying the
standard algorithms (e.g. _k_ -means) directly to the coordinates _Y_ may lead
to sub-optimal results with expensive computational costs due to the highdimensionality. A better approach is to first perform dimension reduction


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 15


by computing the SVD of _Y_ directly or on its random projections, and then
carry out clustering based on the first _r_ right singular vectors _V_ [ˆ] _∈_ O _n,r_ .
See, e.g. Feldman et al. (2013) and Boutsidis et al. (2015), and the reference
therein. It is important to note that the left singular space _U_ are not directly
used in the clustering procedure. Thus Theorem 3 is more suitable for the
analysis of the clustering method than Wedin’s sin Θ theorem as the method
main depends on the accuracy of _V_ [ˆ] as an estimate of _V_ .
Let us consider the following two-class clustering problem in more detail
(see Hastie et al. (2009); Azizyan et al. (2013); Jin and Wang (2016); Jin
et al. (2015)). Suppose _l_ _i_ _∈{−_ 1 _,_ 1 _}_, _i_ = 1 _, ..., n_, are indicators representing
the class label of the _n_ -th nodes and let _µ ∈_ R _[p]_ be a fixed vector. Suppose
one observes _Y_ = [ _Y_ 1 _, · · ·, Y_ _n_ ] where


_iid_
_Y_ _i_ = _l_ _i_ _µ_ + _Z_ _i_ _,_ _Z_ _i_ _∼_ _N_ (0 _, I_ _p_ ) _,_ 1 _≤_ _i ≤_ _n,_


where neither the labels _l_ _i_ nor the mean vector _µ_ are observable. The goal is
to cluster the data into two groups. The accuracy of any clustering algorithm
is measured by the misclassification rate



(4.1) _M_ ( _l,_ [ˆ] _l_ ) := [1]

_n_ [min] _π_



_i_ : _l_ _i_ _̸_ = _π_ ( [ˆ] _l_ _i_ ) _._
���� ����



Here _π_ is any permutations on _{−_ 1 _,_ 1 _}_, as any permutation of the labels
_{−_ 1 _,_ 1 _}_ does not change the clustering outcome.
In this case, _EY_ _i_ is either _µ_ or _−µ_, which lies on a straight line. A simple
PCA-based clustering method is to set


(4.2) ˆ _l_ = sgn(ˆ _v_ ) _,_


where ˆ _v ∈_ R _[n]_ is the first right singular vector of _Y_ . We now apply the sin Θ
upper bound in Theorem 3 to analyze the performance guarantees of this
clustering method. We are particularly interested in the high-dimensional
case where _p ≥_ _n_ . The case where _p < n_ can be handled similarly.


Theorem 5. _Suppose p ≥_ _n, π is any permutation on {−_ 1 _,_ 1 _}. When_
1
_∥µ∥_ 2 _≥_ _C_ gap ( _p/n_ ) 4 _for some large constant C_ gap _>_ 0 _, then for some other_
_constant C >_ 0 _the mis-classification rate for the PCA-based clustering_
_method_ [ˆ] _l given in_ (4.2) _satisfies_


2 [+] _[p]_
E _M_ ( [ˆ] _l, l_ ) _≤_ _C_ _[n][∥][µ][∥]_ [2] _._
_n∥µ∥_ [4] 2


16 T. T. CAI AND A. ZHANG


It is intuitively clear that the clustering accuracy depends on the signal strength _∥µ∥_ 2 . The stronger the signal, the easier to cluster. In particular, Theorem 5 requires the minimum signal strength condition _∥µ∥_ 2 _≥_
1
_C_ gap ( _p/n_ ) 4 . The following lower bound result shows that this condition is
1
necessary for consistent clustering: When the condition _∥µ∥_ 2 _≥_ _C_ gap ( _p/n_ ) 4
does not hold, it is not possible to essentially do better than random guessing.


Theorem 6. _Suppose p ≥_ _n, there exists c_ _gap_ _, C_ _n_ _>_ 0 _such that if n ≥_
_C_ _n_ _,_



inf ˜ _l_ sup 1

_µ_ : _∥µ∥_ 2 _≤c_ _gap_ ( _p/n_ ) ~~4~~
_l∈{−_ 1 _,_ 1 _}_ _[n]_



E _M_ ( [˜] _l, l_ ) _≥_ [1]

8 _[.]_



Remark 5. Azizyan et al. (2013) considered a similar setting when
_n ≥_ _p_, _l_ _i_ ’s are i.i.d. Rademacher variables and derived rates of convergence
for both the upper and lower bounds with a logarithmic gap between the
upper and lower bounds. In contrast, with the help of the newly obtained
perturbation bounds, we are able to establish the optimal misclassification
rate for high-dimensional setting when _n ≤_ _p_ .
Moreover, Jin and Wang (2016) and Jin et al. (2015) considered the sparse
and highly structured setting, where the contrast mean vector _µ_ is assumed
to be sparse and the nonzero coordinates are all equal. Their method is based
on feature selection and PCA. Our setting is close to the “less sparse/weak
signal” case in Jin et al. (2015). In this case, they introduced a simple
aggregation method with


ˆ _l_ [(] _[sa]_ [)] = sgn( _X_ ˆ _µ_ ) _,_


where ˆ _µ_ = arg max _µ∈{−_ 1 _,_ 0 _,_ 1 _}_ _p_ _∥Xµ∥_ _q_ for some _q >_ 0. The statistical limit, i.e.
the necessary condition for obtaining correct labels for most of the points,
is _∥µ∥_ 2 _> C_ in their setting, which is smaller than the boundary _∥µ∥_ 2 _>_
1 1
_C_ ( _p/n_ ) 4 in Theorem 5. As shown in Theorem 6 the bound _∥µ∥_ 2 _> C_ ( _p/n_ ) 4
is necessary. The reason for this difference is that they focused on highly
structured contrast mean vector _µ_ which only takes two values _{_ 0 _, ν}_ . In
contrast, we considered the general _µ ∈_ R _[p]_, which leads to stronger condition
and larger statistical limit. Moreover, the simple aggregation algorithm is
computational difficult for a general signal _µ_, thus the PCA-based method
considered in this paper is preferred under the general dense _µ_ setting.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 17


**5. Canonical Correlation Analysis.** In this section, we consider an
application of the perturbation bounds given in Theorem 1 to the canonical correlation analysis (CCA), which is one of the most important tools
in multivariate analysis in exploring the relationship between two sets of
variables (Hotelling, 1936; Anderson, 2003; Witten et al., 2009; Gao et al.,
2014, 2015; Ma and Li, 2016). Given two random vectors _X_ and _Y_ with
a certain joint distribution, the CCA first looks for the pair of vectors
_α_ [(1)] _∈_ R _[p]_ [1] _, β_ [(2)] _∈_ R _[p]_ [2] that maximize corr(( _α_ [(1)] ) [⊺] _X,_ ( _β_ [(1)] ) [⊺] _Y_ ). After obtaining the first pair of canonical directions, one can further obtain the
second pair _α_ [(2)] _∈_ R _[p]_ [1] _, β_ [(2)] _∈_ R _[p]_ [2] such that Cov(( _α_ [(1)] ) [⊺] _X,_ ( _α_ [(2)] ) [⊺] _X_ ) =
Cov(( _β_ [(1)] ) [⊺] _Y,_ ( _β_ [(2)] ) [⊺] _Y_ ) = 0, and
Corr(( _α_ [(2)] ) [⊺] _X,_ ( _β_ [(2)] ) [⊺] _Y_ ) is maximized. The higher order canonical directions can be obtained by repeating this process. If ( _X, Y_ ) is further assumed
to have joint covariance, say



_X_ Σ _X_ Σ _XY_
Cov = Σ =
� _Y_ � �Σ _Y X_ Σ _Y Y_



_,_
�



the population canonical correlation directions can be inductively defined as
the following optimization problem. For _k_ = 1 _,_ 2 _, · · ·_,

( _α_ [(] _[k]_ [)] _, β_ [(] _[k]_ [)] ) = arg max _a_ [⊺] Σ _XY_ _b,_
_a∈_ R _[p]_ [1] _,b∈_ R _[p]_ [2]


subject to _a_ [⊺] Σ _X_ _a_ = _b_ [⊺] Σ _Y_ _b_ = 1 _,_

_a_ [⊺] Σ _X_ _α_ [(] _[l]_ [)] = _b_ [⊺] Σ _Y_ _β_ [(] _[l]_ [)] = 0 _, ∀_ 1 _≤_ _l ≤_ _k −_ 1 _._


A more explicit form for the canonical correlation directions is given in



1 1

Hotelling (1936): (Σ _X_ 2 _[α]_ [(] _[k]_ [)] _[,]_ [ Σ] _Y_ 2 _[β]_ [(] _[k]_ [)] [) is the] _[ k]_ [-th pair of singular vectors of]



_−_ [1] _−_ [1]

2 2
_X_ [Σ] _[XY]_ [ Σ] _Y_



_−_ [1]
Σ 2



Σ 2 2

_X_ [Σ] _[XY]_ [ Σ] _Y_ [. We combine the leading] _[ r]_ [ population canonical correlation]

directions and write

_A_ = � _α_ [(1)] _· · · α_ [(] _[r]_ [)] [�] _,_ _B_ = � _β_ [(1)] _· · · β_ [(] _[r]_ [)] [�] _._


Suppose one observes i.i.d.samples ( _X_ _i_ [⊺] _[, Y]_ _i_ [ ⊺] [)] [⊺] _[∼]_ _[N]_ [(0] _[,]_ [ Σ). Then the sample]
covariance and cross-covariance for _X_ and _Y_ can be calculated as



_n_



_n_



ˆΣ _X_ = [1]

_n_



_n_
�



� _X_ _i_ _X_ _i_ [⊺] _[,]_ ˆΣ _Y_ = _n_ [1]

_i_ =1



_n_
�



� _Y_ _i_ _Y_ _i_ [⊺] _[,]_ ˆΣ _XY_ = _n_ [1]

_i_ =1



_n_
� _X_ _i_ _Y_ _i_ [⊺] _[.]_


_i_ =1



The standard approach to estimate the canonical correlation directions _α_ [(] _[k]_ [)],



_−_ [1]
_β_ [(] _[k]_ [)] is via the SVD of Σ [ˆ] 2



_−_ [1] _−_ [1]

2 2
_X_ [ˆΣ] _[XY]_ [ ˆΣ] _Y_



2
_Y_



_−_ [1]
ˆΣ 2



_−_ [1] _−_ [1]

2 2
_X_ [ˆΣ] _[XY]_ [ ˆΣ] _Y_



_−_ 2

= _U_ [ˆ] _S_ [ˆ] V = [ˆ]
_Y_



_p_ 1 _∧p_ 2
� _U_ ˆ [: _,k_ ] ˆ _S_ _kk_ ˆ _U_ [: [⊺] _,k_ ] _[.]_

_k_ =1


18 T. T. CAI AND A. ZHANG


Then the leading _r_ sample canonical correlation directions can be calculated

as



_A_ ˆ = ˆΣ _−_ [1] 2

_X_

_B_ ˆ = ˆΣ _−_ 2 [1]



_−X_ 2 _[U]_ [ˆ] [[:] _[,]_ [1:] _[r]_ []] _[,]_ _A_ ˆ = [ˆ _α_ [(1)] _,_ ˆ _α_ [(2)] _, · · ·,_ ˆ _α_ [(] _[r]_ [)] ] _,_



(5.1) _B_ ˆ = ˆΣ _Y−_ 2 [1] _V_ ˆ [: _,_ _[,]_ 1: _r_ ] _,_ _B_ ˆ = [ˆ _β_ [(1)] _,_ ˆ _β_ [(2)] _, · · ·,_ ˆ _β_ [(] _[r]_ [)] ] _._



_A,_ ˆ ˆ _B_ are consistent estimators for the first _r_ left and right canonical directions in the classical fixed dimension case.
Let _X_ _[∗]_ _∈_ R _[p]_ [1] be an independent copy of the original sample _X_, we define
the following two losses to measure the accuracy of the estimator of the
canonical correlation directions,


(5.2) _L_ sp ( _A, A_ [ˆ] ) = min max ( _AOv_ [ˆ] ) [⊺] _X_ _[∗]_ _−_ ( _Av_ ) [⊺] _X_ _[∗]_ [�] [2] _,_
_O∈_ O _r_ _v∈_ R _[r]_ _,∥v∥_ 2 =1 [E] _[X]_ _[∗]_ �

(5.3) _L_ _F_ ( _A, A_ [ˆ] ) = min 2 _[.]_
_O∈_ O _r_ [E] _[X]_ _[∗]_ _[∥]_ [( ˆ] _[AO]_ [)] [⊺] _[X]_ _[∗]_ _[−]_ _[A]_ [⊺] _[X]_ _[∗]_ _[∥]_ [2]


These two losses quantify how well the estimator ( _AO_ [ˆ] ) [⊺] _X_ _[∗]_ can predict the
values of the canonical variables _A_ [⊺] _X_ _[∗]_, where _O ∈_ O _r_ is a rotation matrix
as the objects of interest here are the directions.
The following theorem gives the upper bound for one side of the canonical
correlation directions. The main technical tool is the perturbation bounds
given in Section 2.


Theorem 7. _Suppose_ ( _X_ _i_ _, Y_ _i_ ) _∼_ _N_ (0 _,_ Σ) _, i_ = 1 _, · · ·, n, where S_ =



_−_ [1] _−_ [1]

2 2
_X_ [Σ] _[XY]_ [ Σ] _Y_



_−_ [1]
Σ 2



Σ _X−_ 2 [Σ] _[XY]_ [ Σ] _−Y_ 2 _is of rank-r. Suppose_ _A_ [ˆ] _∈_ R _[p]_ [1] _[×][r]_ _is given by_ (5.1) _. Then_

_there exist uniform constants C_ gap _, C, c >_ 0 _such that whenever σ_ _r_ ( _S_ ) [2] _≥_



1
_C_ gap (( _p_ 1 _p_ 2 ) ~~2~~ + _p_ 1 + _p_ [3] 2 _[/]_ [2] _n_ _[−]_ ~~2~~ [1] )

_n_ _,_



P _L_ sp ( _A, A_ [ˆ] ) _≤_ _[C][p]_ [1] [(] _[nσ]_ _r_ [2] [(] _[S]_ [)][ +] _[p]_ [2] [)] _≥_ 1 _−_ _C_ exp( _−cp_ 1 _∧_ _p_ 2 ) _,_
� _n_ [2] _σ_ _r_ [4] ( _S_ ) �



P _L_ _F_ ( _A, A_ [ˆ] ) _≤_ _[C][p]_ [1] _[r]_ [(] _[nσ]_ _r_ [2] [(] _[S]_ [)][ +] _[p]_ [2] [)] _≥_ 1 _−_ _C_ exp( _−cp_ 1 _∧_ _p_ 2 ) _._
� _n_ [2] _σ_ _r_ [4] ( _S_ ) �


_The results for_ _B_ [ˆ] _can be stated similarly._


Remark 6. Chen et al. (2013) and Gao et al. (2014, 2015) considered
sparse CCA, where the canonical correlation directions _A_ and _B_ are assumed
to be jointly sparse. In particular, Chen et al. (2013) and Gao et al. (2015)
proposed estimators under different settings and provided a unified rateoptimal bound for jointly estimating left and right canonical correlations.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 19


Gao et al. (2014) proposed another computationally feasible estimators _A_ [ˆ] _[∗]_

and _B_ [ˆ] _[∗]_ and provided a minimax rate-optimal bound for _L_ _F_ ( _A_ [ˆ] _[∗]_ _, A_ ) under
regularity conditions that can also be used to prove the consistency of _B_ [ˆ] _[∗]_ .

1

Now consider the setting where _p_ 2 _≫_ _p_ 1, _[p]_ _n_ [2] _[≫]_ _[σ]_ _r_ [2] [(] _[S]_ [) =] _[ t]_ [2] _[ ≫]_ [(] _[p]_ [1] _[p]_ _n_ [2] [)] ~~2~~ . The

lower bound result in Theorem 3.3 by Gao et al. (2014) implies that there
is no consistent estimator for the right canonical correlation directions _B_ .
While Theorem 7 given above shows that the left sample canonical correlation directions _A_ [ˆ] are a consistent estimator of _A_ . This interesting phenomena
again shows the merit of our proposed unilateral perturbation bound.
It is also interesting to develop the lower bounds for _A_ [ˆ] and _B_ [ˆ] . The best
known result, given in Theorem 3.2 in Gao et al. (2014), is the following
two-sided lower bound for both _A_ [ˆ] and _B_ [ˆ] in Frobenius norm loss:


ˆ ˆ _r_ ( _p_ 1 + _p_ 2 )
inf ˆ sup P max _L_ _F_ _A, A_ _, L_ _F_ _B, B_ _≥_ _c_ _≥_ 0 _._ 8 _._
_A,B_ [ˆ] _A,B_ � � � � � �� � _nσ_ min [2] [(] _[S]_ [)] _[∧]_ [1] ��


Establishing the matching one-sided lower bound for Theorem 7 is technical
challenging. We leave it for future research.


**6. Simulations.** In this section, we carry out numerical experiments
to further illustrate the advantages of the separate bounds for the left and
right singular subspaces over the uniform bounds. As mentioned earlier, in
a range of cases, especially when the numbers of rows and columns of the
matrix differ significantly, it is even possible that the singular space on one
side can be stably recovered, while the other side cannot. To illustrates this
point, we specifically perform simulation studies in matrix denoising, highdimensional clustering, and canonical correlation analysis.
We first consider the matrix denoising model discussed in Section 3. Let
_X_ = _tUV_ [⊺] _∈_ R _[p]_ [1] _[×][p]_ [2], where _t ∈_ R, _U_ and _V_ are _p_ 1 _× r_ and _p_ 2 _× r_ random
uniform orthonormal columns with respect to the Haar measure. Let the
_iid_
perturbation _Z_ = ( _Z_ _ij_ ) _p_ 1 _×p_ 2 be randomly generated with _Z_ _ij_ _∼_ _N_ (0 _,_ 1). We
calculate the SVD of _X_ + _Z_ and form the first _r_ left and right singular vectors
as _U_ [ˆ] and _V_ [ˆ] . The average losses in Frobenius and spectral sin Θ distances
for both the left and right singular space estimates with 1,000 repetitions
are given in Table 6 for various values of ( _p_ 1 _, p_ 2 _, r, t_ ). It can be easily seen
from this experiment that the left and right singular perturbation bounds
behave very distinctly when _p_ 1 _≫_ _p_ 2 .
We then consider the high-dimensional clustering model studied in Sec
˜
tion 4. Let ˜ _µ∼N_ (0 _, I_ _p_ ) and _µ_ = _t_ ( _p/n_ ) [1] _[/]_ [4] _·_ ˜ _µ/∥µ∥_ 2 _∈_ R _[p]_, where _t_ = _∥µ∥_ 2
essentially represents the signal strength. The group label _l ∈_ R _[n]_ is randomly


20 T. T. CAI AND A. ZHANG


( _p_ 1 _,_ _p_ 2 _, r, t_ ) _∥_ sin Θ( _U_ ~~[ˆ]~~ _, U_ ) _∥_ [2] _∥_ sin Θ( _V_ ~~[ˆ]~~ _, V_ ) _∥_ [2] _∥_ sin Θ( _U_ ~~[ˆ]~~ _, U_ ) _∥_ _F_ [2] _∥_ sin Θ( _V_ ~~[ˆ]~~ _, V_ ) _∥_ _F_ [2]
(100, 10, 2, 15) 0.3512 0.0669 0.6252 0.0934
(100, 10, 2, 30) 0.1120 0.0139 0.1984 0.0196
(100, 20, 5, 20) 0.2711 0.0930 0.9993 0.2347
(100, 20, 5, 40) 0.0770 0.0195 0.2835 0.0508
(1000, 20, 5, 30) 0.5838 0.0699 2.6693 0.1786
(1000, 20, 10, 100) 0.1060 0.0036 0.9007 0.0109
(1000, 200, 10, 50) 0.3456 0.0797 2.9430 0.4863
(1000, 200, 50, 100) 0.1289 0.0205 4.3614 0.2731


Table 1

_Average losses in Frobenius and spectral_ sin Θ _distances for both the left and right_
_singular space changes after Gaussian noise perturbations._


generated as

_l_ _i_ _iid_ _∼_ � _−_ 1 _,_ 1 _,_ with probabilitywith probability 1 _ρ, −_ _ρ._


_iid_
Based on _n_ i.i.d. observations: _Y_ _i_ = _l_ _i_ _µ_ + _Z_ _i_ _, Z_ _i_ _∼_ _N_ (0 _, I_ _p_ ) _, i_ = 1 _, . . ., n_, we
apply the proposed estimator (4.2) to estimate _l_ . The results for different
values of ( _n, p, t, ρ_ ) are provided in Table 6. It can be seen that the numerical results match our theoretical analysis – the proposed [ˆ] _l_ achieves good
performance roughly when _t ≥_ _C_ ( _p/n_ ) [1] _[/]_ [4] .

|P n<br>P<br>P<br>(p, t, ρP) P<br>PP|5 10 20 50 100 200|
|---|---|
|(100, 1, 1/2)<br>(100, 1, 3/4)<br>(100, 3, 1/2)<br>(100, 3, 3/4)<br>(1000, 1, 1/2)<br>(1000, 1, 3/4)<br>(1000, 3, 1/2)<br>(1000, 3, 3/4)|0.2100<br>0.1485<br>0.0690<br>0.0494<br>0.0440<br>0.0333<br>0.2150<br>0.1590<br>0.0680<br>0.0468<br>0.0422<br>0.0290<br>0.0019<br>0.0005<br>0.0000<br>0.0000<br>0.0000<br>0.0000<br>0.0020<br>0.0005<br>0.0000<br>0.0000<br>0.0000<br>0.0000<br>0.3260<br>0.3510<br>0.3594<br>0.2855<br>0.2691<br>0.1364<br>0.3610<br>0.3610<br>0.3462<br>0.3057<br>0.2696<br>0.1410<br>0.1370<br>0.0485<br>0.0066<br>0.0019<br>0.0013<br>0.0003<br>0.1160<br>0.0425<br>0.0046<br>0.0019<br>0.0018<br>0.0006|



Table 2

_Average misclassification rate for different settings._


We finally investigate the numerical performance of canonical correlation
analysis particularly when the dimensions of two samples differ significantly.
Suppose Σ _X_ = _I_ _p_ 1 + 2 _∥Z_ _p_ 1 1+ _Z_ _p_ ~~[⊺]~~ 1 _[∥]_ [(] _[Z]_ _[p]_ [1] [ +] _[ Z]_ _p_ [⊺] 1 [)] _[,]_ [ Σ] _Y_ [=] _[ I]_ _p_ 2 [+] 2 _∥Z_ _p_ 2 1+ _Z_ _p_ ~~[⊺]~~ 2 _[∥]_ [(] _[Z]_ _[p]_ [2] [ +] _[ Z]_ _p_ [⊺] 2 [),]



Σ _XY_ = Σ _X_ [1] _[/]_ [2] _·_ ( _tUV_ [⊺] ) Σ [1] _Y_ _[/]_ [2] [, where] _[ Z]_ _[p]_ [1] [and] _[ Z]_ _[p]_ [2] [are i.i.d. Gaussian matrices;]
_U ∈_ O _p_ 1 _,r_ _, V ∈_ O _p_ 2 _,r_ are random orthogonal matrices. With _n_ pairs of observations

_X_ _i_ _iid_ Σ _X_ Σ _XY_

_∼_ _N_ (0 _,_ Σ) _,_ Σ = _,_ _i_ = 1 _, . . ., n,_

� _Y_ _i_ � �Σ [⊺] _XY_ Σ _Y_ �



_iid_ Σ _X_ Σ _XY_
_∼_ _N_ (0 _,_ Σ) _,_ Σ =
� �Σ [⊺] _XY_ Σ _Y_



_,_ _i_ = 1 _, . . ., n,_
�


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 21


we apply the procedure discussed in Section 5 to obtain _A_ [ˆ] and _B_ [ˆ], i.e. the
estimates for left and right canonical correlation directions. Since the exact
losses in _L_ sp ( _·, ·_ ), _L_ _F_ ( _·, ·_ ) metrics (5.2) involves difficult optimization, we
instead measure the losses in

sin Θ( ˆ _U, U_ ) _,_ sin Θ( ˆ _U, U_ ) sin Θ( ˆ _V, V_ ) _,_ and sin Θ( ˆ _V, V_ )
��� ��� ��� ��� _F_ _[,]_ ��� ��� ��� ��� _F_ _[.]_


Here _U, V,_ _U,_ [ˆ] _V_ [ˆ] are the first _r_ left and right singular vectors of Σ _[−]_ _X_ [1] _[/]_ [2] Σ _XY_ Σ _[−]_ _Y_ [1] _[/]_ [2]
and Σ [ˆ] _X_ _[−]_ [1] _[/]_ [2] ˆΣ _XY_ ˆΣ _[−]_ _Y_ [1] _[/]_ [2], respectively. It is shown in Step 1 of the proof for Theorem 7 that these measures are equivalent to _L_ sp and _L_ _F_ . The results under
various choices of ( _p_ 1 _, p_ 2 _, n, t_ ) are collected in Table 3. It can be easily seen
that the performance of the right canonical direction estimation is much
better than the left ones when _p_ 1 is much larger than _p_ 2, which is consistent
with the theoretical results in Theorem 7 and illustrates the power of the
newly proposed perturbation bound results.


( _p_ 1 _,_ _p_ 2 _, r, t_ ) _∥_ sin Θ( _U_ ~~[ˆ]~~ _S_ _, U_ _S_ ) _∥_ _∥_ sin Θ( _U_ ~~[ˆ]~~ _S_ _, U_ _S_ ) _∥_ _F_ _∥_ sin Θ( _V_ ~~[ˆ]~~ _S_ _, V_ _S_ ) _∥_ _∥_ sin Θ( _V_ ~~[ˆ]~~ _S_ _, V_ _S_ ) _∥_ _F_
(30, 10, 100, .8) 0.3194 0.6609 0.1571 0.2530
(30, 10, 200, .5) 0.5348 1.1111 0.3343 0.5256
(100, 10, 200, .8) 0.4103 1.0145 0.1120 0.1825
(100, 10, 500, .5) 0.5183 1.2821 0.1614 0.2606
(200, 20, 500, .8) 0.3239 0.8428 0.0746 0.1442
(200, 20, 800, .5) 0.5834 1.5155 0.2423 0.4605
(500, 50, 1000, .8) 0.3875 1.0515 0.1091 0.2472
(500, 50, 2000, .5) 0.5677 1.5467 0.2216 0.4910


Table 3
_Average losses in L_ sp ( _·, ·_ ) _and L_ _F_ ( _·, ·_ ) _metrics for the left and right canonical directions._


**7. Discussions.** We have established in the present paper new and
rate-optimal perturbation bounds, measured in both spectral and Frobenius
sin Θ distances, for the left and right singular subspaces separately. These
perturbation bounds are widely applicable to the analysis of many highdimensional problems. In particular, we applied the perturbation bounds
to study three important problems in high-dimensional statistics: low-rank
matrix denoising and singular space estimation, high-dimensional clustering
and CCA. As mentioned in the introduction, in addition to these problems
and possible extensions discussed in the previous sections, the obtained perturbation bounds can be used in a range of other applications including
_community detection in bipartite networks, multidimensional scaling, cross-_
_covariance matrix estimation, and singular space estimation for matrix com-_
_pletion_ . We briefly discuss these problems here.


22 T. T. CAI AND A. ZHANG


An interesting application of the perturbation bounds given in Section 2 is
_community detection in bipartite graphs_ . Community detection in networks
has attracted much recent attention. The focus of the current community
detection literature has been mainly on unipartite graph (i.e., there are only
one type of nodes). However in some applications, the nodes can be divided
into different types and only the interactions between the different types of
nodes are available or of interest, such as people vs. committees, Facebook
users vs. public pages (see Melamed (2014); Alzahrani and Horadam (2016)).
The observations on the connectivity of the network between two types of
nodes can be described by an adjacency matrix _A_, where _A_ _ij_ = 1 if the _i_ -th
Type 1 node and _j_ -th Type 2 node are connected, and _A_ _ij_ = 0 otherwise.
The spectral method is one of the most commonly used approaches in the
literature with theoretical guarantees (Rohe et al., 2011; Lei and Rinaldo,
2015). In a bipartite network, the left and right singular subspaces could
behave very differently from each other. Our perturbation bounds can be
used for community detection in bipartite graph and potentially lead to
sharper results in some settings.
Another possible application lies in _multidimensional scaling (MDS)_ with
distance matrix between two sets of points. MDS is a popular method of
visualizing the data points embedded in low-dimensional space based on the
distance matrices (Borg and Groenen, 2005). Traditionally MDS deals with
unipartite distance matrix, where all distances between any pairs of points
are observed. In some applications, the data points are from two groups
and one is only able to observe its biparitite distance matrix formed by the
pairwise distances between points from different groups. As the SVD is a
commonly used technique for dimension reduction in MDS, the perturbation
bounds developed in this paper can be potentially used for the analysis of
MDS with bipartite distance matrix.
In some applications, the _cross-covariance matrix_, not the overall covariance matrix, is of particular interest. (Cai et al., 2015a) considered multiple
testing of cross-covariances in the context of the phenome-wide association
studies (PheWAS). Suppose _X ∈_ R _[p]_ [1] and _Y ∈_ R _[p]_ [2] are jointly distributed
with covariance matrix Σ. Given _n_ i.i.d. samples ( _X_ _i_ _, Y_ _i_ ), _i_ = 1 _, · · ·, n_, from
the joint distribution, one wishes to make statistical inference for the crosscovariance matrix Σ _XY_ . If Σ _XY_ has low-rank structure, the perturbation
bounds established in Section 2 could be potentially applied to make statistical inference for Σ _XY_ .
_Matrix completion,_ whose central goal is to recover a large low-rank matrix based on a limited number of observable entries, has been widely studied
in the last decade. Among various methods for matrix completion, spectral


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 23


method is fast, easy to implement and achieves good performance (Keshavan et al., 2010; Chatterjee, 2014; Cho et al., 2015). The new perturbation
bounds can be potentially used for singular space estimation under the matrix completion setting to yield better results.
In addition to the aforementioned problems, _high-dimensional clustering_
_with correlated features_ is an important extension of the problem of clustering with independent features considered in the present paper. Specifically,
_iid_
based on _n_ observations _Y_ _i_ = _l_ _i_ _µ_ + _Z_ _i_ _∈_ R _[p]_, _i_ = 1 _, . . ., n_, where _Z_ _i_ _∼_ _N_ (0 _,_ Σ),
one aims to recover the unknown labels _{l_ _i_ _}_ _[n]_ _i_ =1 [. When Σ is known or can be]
well estimated, one can transform _Y_ [˜] _i_ = Σ _[−]_ [1] _[/]_ [2] _Y_ _i_ _, i_ = 1 _, . . ., n_ and perform
the spectral method on _Y_ ˜ _i_ _n_
� � _i_ =1 [. It would be an interesting and challenging]

problem to consider the general setting where Σ is unknown. We leave this
for future research.


**8. Proofs.** We prove the main results in Sections 2, 3 and 4 in this
section. The proofs for CCA and the additional technical results are given
in the supplementary materials.


8.1. _Proofs of General Unilateral Perturbation Bounds._ Some technical
tools are needed to prove Theorems 1, 2 and Proposition 1. In particular,
we need a few useful properties of sin Θ distances given below in Lemma
1. Specifically, Lemma 1 provides some more convenient expressions than
the definitions for the sin Θ distances. It also shows that they are indeed
distances as they satisfy triangle inequality. Some other widely used metrics
for orthogonal spaces, including


(8.1) _D_ sp ( _V, V_ [ˆ] ) = inf _D_ _F_ ( _V, V_ [ˆ] ) = inf
_O∈_ O _r_ _[∥][V]_ [ˆ] _[ −]_ _[V O][∥][,]_ _O∈_ O _r_ _[∥][V]_ [ˆ] _[ −]_ _[V O][∥]_ _[F]_ _[,]_


ˆ ˆ
(8.2) _V_ ˆ _V_ ⊺ _−_ _V V_ ⊺ _,_ _V_ ˆ _V_ ⊺ _−_ _V V_ ⊺
��� ��� ��� ��� _F_ _[.]_


are shown to be equivalent to the sin Θ distances.


Lemma 1 (Properties of the sin Θ Distances). _The following properties_
_hold for the_ sin Θ _distances._


_1. (Equivalent Expressions) Suppose V,_ _V_ [ˆ] _∈_ O _p,r_ _. If V_ _⊥_ _is an orthogonal_
_extension of V, namely_ [ _V V_ _⊥_ ] _∈_ O _p_ _, we have the following equivalent_
_forms for ∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _and ∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ _,_



=
(8.3) _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ �



1 _−_ _σ_ [2]
min [( ˆ] _[V]_ [ ⊺] _[V]_ [ ) =] _[ ∥][V]_ [ˆ] [ ⊺] _[V]_ _[⊥]_ _[∥][,]_


24 T. T. CAI AND A. ZHANG


(8.4) _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ = ~~�~~ _r −∥V_ [⊺] _V_ [ˆ] _∥_ [2] _F_ [=] _[ ∥][V]_ [ˆ] [ ⊺] _[V]_ _[⊥]_ _[∥]_ _[F]_ _[ .]_


_2. (Triangle Inequality) For any V_ 1 _, V_ 2 _, V_ 3 _∈_ O _p,r_ _,_


(8.5) _∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥≤∥_ sin Θ( _V_ 1 _, V_ 2 ) _∥_ + _∥_ sin Θ( _V_ 1 _, V_ 3 ) _∥,_


(8.6) _∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥_ _F_ _≤∥_ sin Θ( _V_ 1 _, V_ 2 ) _∥_ _F_ + _∥_ sin Θ( _V_ 1 _, V_ 3 ) _∥_ _F_ _._


_3. (Equivalence with Other Metrics) The metrics defined as_ (8.1) _and_
(8.2) _are equivalent to_ sin Θ _distances as the following inequalities hold_



_∥_ sin Θ( _V, V_ [ˆ] ) _∥≤_ _D_ sp ( _V, V_ [ˆ] ) _≤_ _√_



2 _∥_ sin Θ( _V, V_ [ˆ] ) _∥,_



_∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ _≤_ _D_ _F_ ( _V, V_ [ˆ] ) _≤_ _√_ 2 _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ _,_


ˆ
sin Θ( ˆ _V, V_ ) _≤_ _V_ ˆ _V_ ⊺ _−_ _V V_ ⊺ _≤_ 2 sin Θ( ˆ _V, V_ ) _,_
��� ��� ��� ��� ��� ���

ˆ
_V_ ˆ _V_ ⊺ _−_ _V V_ ⊺ _√_ 2 sin Θ( ˆ _V, V_ )
��� ��� _F_ [=] ��� ��� _F_ _[.]_



_∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ _≤_ _D_ _F_ ( _V, V_ [ˆ] ) _≤_ _√_



2 sin Θ( ˆ _V, V_ )
��� ��� _F_ _[.]_



Proof of Proposition 1. First, we can rotate the right singular space
by right multiplying the whole matrices _A, V_ [⊺] _, W_ [⊺] by [ _W W_ _⊥_ ] without
changing the singular values and left singular vectors. Thus without loss
of generality, we assume that


[ _W W_ _⊥_ ] = _I_ _p_ 2 _._


Next, we further calculate the SVD: _AW_ = _A_ [: _,_ 1: _r_ ] := _U_ [¯] Σ [¯] _V_ [¯] [⊺], where _U_ [¯] _∈_
O _p_ 1 _,r_ _,_ Σ [¯] _∈_ R _[r][×][r]_, _V_ [¯] _∈_ O _r_, and rotate the left singular space by left multiplying the whole matrix _A_ by [ _U_ [¯] _U_ [¯] _⊥_ ] [⊺], then rotate the right singular space by
right multiplying _A_ [: _,_ 1: _r_ ] by _V_ [¯] . After this rotation, the singular structure of
_A_, _AW_ are unchanged. Again without loss of generality, we can assume that

[ _U_ [¯] _U_ [¯] _⊥_ ] [⊺] = _I_ _p_ 1 _,_ _V_ [¯] = _I_ _r_ _._ After these two steps of rotations, the formation of
_A_ is much simplified,



_σ_ 1 ( _AW_ )






_,_





(8.7) _A_ =



_r_ ... _U_ ¯ [⊺] _AW_ _⊥_
_σ_ _r_ ( _AW_ ) ¯
_p_ 1 _−_ _r_  0 _U_ _⊥_ [⊺] _[AW]_ _[⊥]_



_r_ _p_ 2 _−_ _r_

_σ_ 1 ( _AW_ )


¯








PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 25


while the problem we are considering is still without loss of generality. For
convenience, denote


(8.8) � _U_ ¯ ⊺ _AW_ _⊥_ � ⊺ = [ _y_ (1) _y_ (2) _· · · y_ ( _r_ ) ] _,_ _y_ [(1)] _, · · ·, y_ [(] _[r]_ [)] _∈_ R _[p]_ [2] _[−][r]_ _._


We can further compute that



_σ_ 1 [2] [(] _[AW]_ [)] _σ_ 1 ( _AW_ ) _y_ [(1)][⊺]






_._





(8.9) _A_ [⊺] _A_ =



_r_ ... ...
_σ_ _r_ [2] [(] _[AW]_ [)] _σ_ _r_ ( _AW_ ) _y_ [(] _[r]_ [)][⊺]



_r_ _p_ 2 _−_ _r_
 _σ_ 1 [2] [(] _[AW]_ [)] _σ_ 1 ( _AW_ ) _y_







_p_ 2 _−_ _r_ _σ_ 1 ( _AW_ ) _y_ [(1)] _· · ·_ _σ_ _r_ ( _AW_ ) _y_ [(] _[r]_ [)] ( _AW_ _⊥_ ) [⊺] _AW_ _⊥_



By basic theory in algebra, the _i_ -th eigenvalue of _A_ [⊺] _A_ is equal to _σ_ _i_ [2] [(] _[A]_ [), and]
the _i_ -th eigenvector of _A_ [⊺] _A_ is equal to the _i_ -th right singular vector of _A_
(up-to-sign). Suppose the singular vectors of _A_ are _V_ [˜] = [ _v_ [(1)] _, v_ [(2)] _, · · ·, v_ [(] _[p]_ [2] [)] ],
where the singular values can be further decomposed into two parts as
(8.10)



_v_ [(] _[k]_ [)] =



_r_ _α_ [(] _[k]_ [)]

_p_ 2 _−_ _r_ � _β_ [(] _[k]_ [)] _[,]_ � or equivalently _,_ _α_ [(] _[k]_ [)] = _W_ [⊺] _v_ [(] _[k]_ [)] _, β_ [(] _[k]_ [)] = _W_ _⊥_ [⊺] _[v]_ [(] _[k]_ [)] _[.]_



By observing the _i_ -th entry of _A_ [⊺] _Av_ [(] _[k]_ [)] = _σ_ _k_ [2] [(] _[A]_ [)] _[v]_ [(] _[k]_ [)] [, we know for 1] _[ ≤]_ _[i][ ≤]_ _[r]_ [,]
_r_ + 1 _≤_ _k ≤_ _p_ 2,

� _σ_ _i_ [2] [(] _[AW]_ [)] _[ −]_ _[σ]_ _k_ [2] [(] _[A]_ [)] � _α_ _i_ [(] _[k]_ [)] + _σ_ _i_ ( _AW_ ) _y_ [(] _[i]_ [)][⊺] _β_ [(] _[k]_ [)] = 0 _,_

(8.11) _−σ_ _i_ ( _AW_ )
_⇒_ _α_ [(] _[k]_ [)] =
_i_
_σ_ _i_ [2] [(] _[AW]_ [)] _[ −]_ _[σ]_ _k_ [2] [(] _[A]_ [)] _[y]_ [(] _[i]_ [)][⊺] _[β]_ [(] _[k]_ [)] _[.]_


Recall the assumption that


(8.12) _σ_ 1 ( _AW_ ) _≥· · · ≥_ _σ_ _r_ ( _AW_ ) _> σ_ _r_ +1 ( _A_ ) _≥· · · σ_ _p_ 2 ( _A_ ) _≥_ 0 _._


Also _x_ [2] _−x_ _y_ [2] [=] _x−y_ 1 [2] _/x_ [is a decreasing function for] _[ x]_ [ and a increasing function]
for _y_ when _x > y ≥_ 0, so
(8.13)
_σ_ _i_ ( _AW_ ) _σ_ _r_ ( _AW_ )
1 _≤_ _i ≤_ _r, r_ + 1 _≤_ _k ≤_ _p_ 2 _._
_σ_ _i_ [2] [(] _[AW]_ [)] _[ −]_ _[σ]_ _k_ [2] [(] _[A]_ [)] _[ ≤]_ _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] _[,]_


Since [ _β_ [(] _[r]_ [+1)] _· · · β_ [(] _[p]_ [2] [)] ] is the submatrix of the orthogonal matrix _V_,


(8.14) [ _β_ ( _r_ +1) _· · · β_ ( _p_ 2 ) ] _≤_ 1 _._
��� ���


26 T. T. CAI AND A. ZHANG


Now we can give an upper bound for the Frobenius norm of [ _α_ [(] _[r]_ [+1)] _· · · α_ [(] _[p]_ [2] [)] ]



_p_ 2
�

_k_ = _r_ +1



2
���� _α_ [(] _[r]_ [+1)] _· · · α_ [(] _[p]_ [2] [)] [���] � _F_ [=]



_r_
�


_i_ =1



2
_α_ [(] _[k]_ [)]
_i_
� �



(8.13) _σ_ _r_ [2] [(] _[AW]_ [)]
_≤_ 2
~~�~~ _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] ~~�~~



_r_
�


_i_ =1



_p_ 2
�

_k_ = _r_ +1



_y_ [(] _[i]_ [)][⊺] _β_ [(] _[k]_ [)] [�] [2]
�



_≤_ _σ_ _r_ [2] [(] _[AW]_ [)] 2 _[∥]_ [[] _[y]_ 1 _[· · ·][ y]_ _r_ []] [⊺] _[∥]_ [2] _F_ [ _β_ ( _r_ +1) _· · · β_ ( _p_ 2 ) ] 2
~~�~~ _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] ~~�~~ ��� ���

(8.8)( _≤_ 8.14) _σ_ _r_ [2] [(] _[AW]_ [)] 2 �� _U_ ¯ ⊺ _AW_ _⊥_ �� 2 _F_ _[.]_
~~�~~ _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] ~~�~~


It is more complicated to give a upper bound for the spectral norm of

[ _α_ [(] _[r]_ [+1)] _· · · α_ [(] _[p]_ [2] [)] ]. Suppose _s_ = ( _s_ _r_ +1 _, · · ·, s_ _p_ 2 ) _∈_ R _[p]_ [2] _[−][r]_ is any vector with
_∥s∥_ 2 = 1. Based on (8.11),


_p_ 2
� _s_ _k_ _α_ _i_ [(] _[k]_ [)]

_k_ = _r_ +1



_p_ 2
�

_k_ = _r_ +1



=


(8.12)
=



_p_ 2
�

_k_ = _r_ +1


_p_ 2
�

_k_ = _r_ +1



_−s_ _k_ _σ_ _i_ ( _AW_ ) _y_ [(] _[i]_ [)][⊺] _β_ [(] _[k]_ [)]

=
_σ_ _i_ [2] [(] _[AW]_ [)] _[ −]_ _[σ]_ _k_ [2] [(] _[A]_ [)]



_∞_
�


_l_ =0



_−s_ _k_ _σ_ _k_ [2] _[l]_ [(] _[A]_ [)] =

_y_ [(] _[i]_ [)][⊺] _β_ [(] _[k]_ [)]
_σ_ _i_ [2] _[l]_ [+1] ( _AW_ )



_−s_ _k_ 1

_[y]_ [(] _[i]_ [)][⊺] _[β]_ [(] _[k]_ [)]
_σ_ _i_ ( _AW_ ) 1 _−_ _σ_ _k_ [2] [(] _[A]_ [)] _[/σ]_ _[i]_ [(] _[AW]_ [)] [2]



_p_ 2
� _s_ _k_ _σ_ _k_ [2] _[l]_ [(] _[A]_ [)] _[β]_ [(] _[k]_ [)]
� _k_ = _r_ +1 �



_−y_ [(] _[i]_ [)][⊺]

_σ_ _i_ [2] _[l]_ [+1] ( _AW_ )



_∞_
�


_l_ =0



_._



Hence,



_y_ [(1)][⊺] _/σ_ 1 [2] _[l]_ [+1] ( _AW_ )

...
_y_ [(] _[r]_ [)][⊺] _/σ_ _r_ [2] _[l]_ [+1] ( _AW_ )







 _·_



�����



_p_ 2

_s_ _k_ _α_ [(] _[k]_ [)] _≤_

�

_k_ = _r_ +1 ����� 2



_p_ 2
�



�������









_p_ 2
� _s_ _k_ _σ_ _k_ [2] _[l]_ [(] _[A]_ [)] _[β]_ [(] _[k]_ [)]
� _k_ = _r_ +1 � [�]

������ 2



_≤_


(8.8)(8.14)(8.12)
_≤_



_∞_
�


_l_ =0


_∞_
�


_l_ =0



_∞_
�


_l_ =0



��[ _y_ (1) _y_ (2) _· · · y_ ( _r_ ) ]�� _·_ [ _β_ ( _r_ +1) _β_ ( _r_ +2) _· · · β_ ( _p_ 2 ) ]

_σ_ _r_ [2] _[l]_ [+1] ( _AW_ ) ��� ���




_·_ ���� _s_ _r_ +1 _σ_ _r_ [2] +1 _[l]_ [(] _[A]_ [)] _[,][ · · ·][, s]_ _[p]_ 2 _[σ]_ _p_ [2] 2 _[l]_ [(] _[A]_ [)] ���� 2



_∥U_ [˜] [⊺] _AW_ _⊥_ _∥_ _· σ_ _r_ [2] +1 _[l]_ [(] _[A]_ [)] _[∥][s][∥]_ [2]
_σ_ _r_ [2] _[l]_ [+1] ( _AW_ )



= _[∥][U]_ [˜] [⊺] _[AW]_ _[⊥]_ _[∥][σ]_ _[r]_ [(] _[AW]_ [)]

_σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] _[,]_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 27


which implies


_∥_ [ _α_ [(] _[r]_ [+1)] _· · · α_ [(] _[p]_ [2] [)] ] _∥≤_ _[∥][U]_ [˜] [⊺] _[AW]_ _[⊥]_ _[∥][σ]_ _[r]_ [(] _[AW]_ [)]

_σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] _[.]_


Note the definition of _α_ [(] _[i]_ [)] in (8.10), we know


[ _α_ [(] _[r]_ [+1)] _α_ [(] _[r]_ [+2)] _· · · α_ [(] _[p]_ [2] [)] ] = _V_ [˜] [1: _r,_ ( _r_ +1): _p_ 2 ] = ( _V_ _⊥_ ) [1: _r,_ :] _._


Thus,


_∥_ sin Θ( _V, W_ ) _∥_ (8.3 = _∥_ ) _W_ [⊺] _V_ _⊥_ _∥_ = [ _α_ ( _r_ +1) _· · · α_ ( _p_ 2 ) ] _≤_ _∥U_ ˜ ⊺ _AW_ _⊥_ _∥σ_ _r_ ( _AW_ )
��� ��� _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] _[,]_


(8.4)
_∥_ sin Θ( _V, W_ ) _∥_ [2] _F_ = _∥W_ [⊺] _V_ _⊥_ _∥_ _F_ [2]

(8.15) = _∥_ [ _α_ [(] _[r]_ [+1)] _· · · α_ [(] _[p]_ [2] [)] ] _∥_ _F_ [2] _[≤]_ _∥U_ [˜] [⊺] _AW_ _⊥_ _∥_ [2] _F_ _[σ]_ _r_ [2] [(] _[AW]_ [)] 2 _[.]_
~~�~~ _σ_ _r_ [2] ( _AW_ ) _−_ _σ_ _r_ [2] +1 [(] _[A]_ [)] ~~�~~


Finally, since _U_ [¯] is the left singular vectors of _AW_,


(8.16) _∥U_ [¯] [⊺] _AW_ _⊥_ _∥_ = _∥_ P ( _AW_ ) _AW_ _⊥_ _∥,_ _∥U_ [¯] [⊺] _AW_ _⊥_ _∥_ _F_ = _∥_ P ( _AW_ ) _AW_ _⊥_ _∥._


The upper bounds 1 in (2.11) and _[√]_ ~~_r_~~ on (2.12) are trivial. Therefore, we
have finished the proof of Proposition 1.


Proof of Theorem 1. Before proving this theorem, we introduce the
following lemma on the inequalities of the singular values in the perturbed

matrix.


Lemma 2. _Suppose X ∈_ R _[p][×][n]_ _, Y ∈_ R _[p][×][n]_ _, rank_ ( _X_ ) = _a, rank_ ( _Y_ ) = _b,_


_1. σ_ _a_ + _b_ +1 _−r_ ( _X_ + _Y_ ) _≤_ min( _σ_ _a_ +1 _−r_ ( _X_ ) _, σ_ _b_ +1 _−r_ ( _Y_ )) _for r ≥_ 1 _;_
_2. if we further have X_ [⊺] _Y_ = 0 _or XY_ [⊺] = 0 _, we must have a_ + _b ≤_ _n ∧_ _p,_
_and_
_σ_ _r_ [2] [(] _[X]_ [ +] _[ Y]_ [ )] _[ ≥]_ [max(] _[σ]_ _r_ [2] [(] _[X]_ [)] _[, σ]_ _r_ [2] [(] _[Y]_ [ ))]


_for any r ≥_ 1 _. Also,_


_σ_ 1 [2] [(] _[X]_ [ +] _[ Y]_ [ )] _[ ≤]_ _[σ]_ 1 [2] [(] _[X]_ [) +] _[ σ]_ 1 [2] [(] _[Y]_ [ )] _[.]_


28 T. T. CAI AND A. ZHANG


The proof of Lemma 2 is provided in the supplementary materials. Applying Lemma 2, we get


_σ_ min [2] [( ˆ] _[XV]_ [ ) =] _[ σ]_ _r_ [2] [( ˆ] _[XV]_ [ ) =] _[ σ]_ _r_ [2] [(] _[UU]_ [⊺] _[XV]_ [ˆ] [ +] _[ U]_ _[⊥]_ _[U]_ _⊥_ [⊺] _[XV]_ [ˆ] [ )]

(8.17)
_≥σ_ _r_ [2] [(] _[UU]_ [⊺] _[XV]_ [ˆ] [ ) =] _[ α]_ [2] _[,]_ (by Lemma 2 Part 2 _._ )


Since _U, V_ have _r_ columns, rank( _XV V_ [ˆ] [⊺] ) _,_ rank( _UU_ [⊺] _X_ [ˆ] ) _≤_ _r_ . Also since _X_ [ˆ] =
_U_ _⊥_ _U_ _⊥_ [⊺] _[X]_ [ˆ][ +] _[ UU]_ [⊺] _[X]_ [ˆ][ = ˆ] _[XV]_ _[⊥]_ _[V]_ [ ⊺] _⊥_ [+ ˆ] _[XV V]_ [ ⊺] [, we have]


_σ_ _r_ [2] +1 [( ˆ] _[X]_ [)] _[ ≤]_ [min] _σ_ 1 [2] [(] _[U]_ _[⊥]_ _[U]_ _⊥_ [⊺] _[X]_ [ˆ] [)] _[, σ]_ 1 [2] [( ˆ] _[XV]_ _[⊥]_ _[V]_ [ ⊺] _⊥_ [)] (by Lemma 2 Part 1.)
� �

= min _σ_ 1 [2] [(] _[Z]_ [21] [+] _[ U]_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ [)] _[, σ]_ 1 [2] [(] _[Z]_ [12] [+] _[ U]_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ [)]
� �

_≤_ ( _β_ [2] + _z_ 12 [2] [)] _[ ∧]_ [(] _[β]_ [2] [ +] _[ z]_ 21 [2] [)] (by Lemma 2 Part 2.)

= _β_ [2] + _z_ 12 [2] _[∧]_ _[z]_ 21 [2] _[.]_


We shall also note the fact that for any matrix _A ∈_ R _[p][×][r]_ with _r ≤_ _p_, denote
the SVD as _A_ = _U_ _A_ Σ _A_ _V_ _A_ [⊺] [, then]


(8.18) _A_ ( _A_ ⊺ _A_ ) _†_ = _U_ _A_ Σ _A_ _V_ ⊺ _A_ � _V_ _A_ Σ [2] _A_ _[V]_ [ ⊺] _A_ � _†_ [�] = _U_ _A_ Σ _†A_ _[V]_ [ ⊺] _A_ _≤_ _σ_ min _−_ 1 [(] _[A]_ [)] _[.]_
��� ��� ��� �� ��� ���


Thus,


_∥_ P ( ˆ _XV_ ) _[XV]_ [ˆ] _[⊥]_ _[∥]_ [=] _[ ∥]_ [P] ( _XV_ [ˆ] ) [P] _[U]_ [ ˆ] _[XV]_ _[⊥]_ [+][ P] ( _XV_ [ˆ] ) [P] _[U]_ _[⊥]_ _[XV]_ [ˆ] _[⊥]_ _[∥]_

_≤∥_ P ( ˆ _XV_ ) _[UU]_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥]_ [+] _[ ∥]_ [P] ( _XV_ [ˆ] ) _[U]_ _[⊥]_ _[U]_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥]_

_≤∥U_ [⊺] _XV_ [ˆ] _⊥_ _∥_ + _∥XV_ [ˆ] [( _XV_ [ˆ] ) [⊺] ( _XV_ [ˆ] )] _[−]_ [1] ( _XV_ [ˆ] ) [⊺] _U_ _⊥_ _U_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥]_

_≤∥U_ [⊺] _XV_ [ˆ] _⊥_ _∥_ + _∥XV_ [ˆ] [( _XV_ [ˆ] ) [⊺] ( _XV_ [ˆ] )] _[−]_ [1] _∥· ∥U_ _⊥_ [⊺] _[XV]_ [ˆ] _[ ∥· ∥][U]_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥]_


(8.18) 1
_≤∥U_ [⊺] _ZV_ _⊥_ _∥_ + _∥U_ _⊥_ [⊺] _[ZV][ ∥· ∥][U]_ _⊥_ [⊺] _[XV]_ [ˆ] _[⊥]_ _[∥]_

_σ_ min ( _XV_ [ˆ] )



(8.17 _≤_ ) _z_ 12 + _[β]_



_._

_α_




_[β][z]_ [21]

_[β]_

_α_ _[z]_ [21] [ =] _[ αz]_ [12] [ +] _α_



Similarly,




_[β][∥][Z]_ [21] _[∥]_ _[F]_
_∥P_ ( ˆ _XV_ ) _[XV]_ [ˆ] _[⊥]_ _[∥]_ _[F]_ _[ ≤]_ _[α][∥][Z]_ [12] _[∥]_ _[F]_ [ +] _α_ _._



Next, applying Proposition 1 by setting _A_ = _X_ [ˆ], _W_ [˜] = [ _V V_ _⊥_ ], _V_ [˜] = [ _V_ [ˆ] _V_ [ˆ] _⊥_ ],
we could obtain (2.4).


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 29


**References.**


Alzahrani, T. and Horadam, K. (2016). Community detection in bipartite networks: Algorithms and case studies. In _Complex Systems and Networks_, pages 25–50. Springer.
Anderson, T. W. (2003). _An Introduction to Multivariate Statistical Analysis_ . Wiley,
Hoboken, NJ., 3rd ed. edition.
Argyriou, A., Evgeniou, T., and Pontil, M. (2008). Convex multi-task feature learning.
_Machine Learning_, 73(3):243–272.
Azizyan, M., Singh, A., and Wasserman, L. (2013). Minimax theory for high-dimensional
gaussian mixtures with sparse mean separation. In _Advances in Neural Information_
_Processing Systems_, pages 2139–2147.
Balakrishnan, S., Xu, M., Krishnamurthy, A., and Singh, A. (2011). Noise thresholds
for spectral clustering. In _Advances in Neural Information Processing Systems_, pages
954–962.
Benaych-Georges, F. and Nadakuditi, R. R. (2012). The singular values and vectors of
low rank perturbations of large rectangular random matrices. _Journal of Multivariate_
_Analysis_, 111:120–135.
Borg, I. and Groenen, P. J. (2005). _Modern multidimensional scaling: Theory and appli-_
_cations_ . Springer Science & Business Media.
Boutsidis, C., Zouzias, A., Mahoney, M. W., and Drineas, P. (2015). Randomized dimensionality reduction for _k_ -means clustering. _Information Theory, IEEE Transactions on_,
61(2):1045–1062.
Bura, E. and Pfeiffer, R. (2008). On the distribution of the left singular vectors of a
random matrix and its applications. _Statistics & Probability Letters_, 78(15):2275–2280.
Cai, T., Cai, T. T., Liao, K., and Liu, W. (2015a). Large-scale simultaneous testing of
cross-covariance matrix with applications to phewas. technical report.
Cai, T. T., Li, X., and Ma, Z. (2016). Optimal rates of convergence for noisy sparse phase
retrieval via thresholded wirtinger flow. _The Annals of Statistics_, 44:2221–2251.
Cai, T. T., Ma, Z., and Wu, Y. (2013). Sparse PCA: Optimal rates and adaptive estimation. _The Annals of Statistics_, 41(6):3074–3110.
Cai, T. T., Ma, Z., and Wu, Y. (2015b). Optimal estimation and rank detection for sparse
spiked covariance matrices. _Probability Theory and Related Fields_, 161:781–815.
Candes, E., Sing-Long, C. A., and Trzasko, J. D. (2013). Unbiased risk estimates for singular value thresholding and spectral estimators. _Signal Processing, IEEE Transactions_
_on_, 61(19):4643–4657.
Candes, E. J. and Plan, Y. (2010). Matrix completion with noise. _Proceedings of the_
_IEEE_, 98(6):925–936.
Cand`es, E. J. and Recht, B. (2009). Exact matrix completion via convex optimization.
_Foundations of Computational Mathematics_, 9(6):717–772.
Cand`es, E. J. and Tao, T. (2010). The power of convex relaxation: Near-optimal matrix
completion. _Information Theory, IEEE Transactions on_, 56(5):2053–2080.
Capitaine, M., Donati-Martin, C., and F´eral, D. (2009). The largest eigenvalues of finite rank deformation of large wigner matrices: convergence and nonuniversality of the
fluctuations. _The Annals of Probability_, 37:1–47.
Chatterjee, S. (2014). Matrix estimation by universal singular value thresholding. _The_
_Annals of Statistics_, 43(1):177–214.
Chen, M., Gao, C., Ren, Z., and Zhou, H. H. (2013). Sparse CCA via precision adjusted
iterative thresholding. _arXiv preprint arXiv:1311.6186_ .
Cho, J., Kim, D., and Rohe, K. (2015). Asymptotic theory for estimating the singular
vectors and values of a partially-observed low rank matrix with noise. _arXiv preprint_


30 T. T. CAI AND A. ZHANG


_arXiv:1508.05431_ .
Davis, C. and Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. iii.
_SIAM J. Numer. Anal._, 7:1–46.
Donoho, D. and Gavish, M. (2014). Minimax risk of matrix denoising by singular value
thresholding. _The Annals of Statistics_, 42(6):2413–2440.
Dopico, F. M. (2000). A note on sin _θ_ theorems for singular subspace variations. _BIT_
_Numerical Mathematics_, 40(2):395–403.
Fan, J., Wang, W., and Zhong, Y. (2016). An _ℓ_ _∞_ eigenvector perturbation bound and its
application to robust covariance estimation. _arXiv preprint arXiv:1603.03516_ .
Feldman, D., Schmidt, M., and Sohler, C. (2013). Turning big data into tiny data:
Constant-size coresets for k-means, PCA and projective clustering. In _Proceedings_
_of the Twenty-Fourth Annual ACM-SIAM Symposium on Discrete Algorithms_, pages
1434–1453. SIAM.
Gao, C., Ma, Z., Ren, Z., and Zhou, H. H. (2015). Minimax estimation in sparse canonical
correlation analysis. _The Annals of Statistics_, 43(5):2168–2197.
Gao, C., Ma, Z., and Zhou, H. H. (2014). Sparse CCA: Adaptive estimation and computational barriers. _arXiv preprint arXiv:1409.8565_ .
Gavish, M. and Donoho, D. L. (2014). The optimal hard threshold for singular values is
4 _/_ ~~_√_~~ 3. _Information Theory, IEEE Transactions on_, 60(8):5040–5053.

Goldberg, D., Nichols, D., Oki, B. M., and Terry, D. (1992). Using collaborative filtering
to weave an information tapestry. _Communications of the ACM_, 35(12):61–70.
Gross, D. (2011). Recovering low-rank matrices from few coefficients in any basis. _Infor-_
_mation Theory, IEEE Transactions on_, 57(3):1548–1566.
Hardoon, D. R., Szedmak, S., and Shawe-Taylor, J. (2004). Canonical correlation analysis:
An overview with application to learning methods. _Neural computation_, 16(12):2639–
2664.
Hastie, T., Tibshirani, R., and Friedman, J. (2009). The elements of statistical learning
2nd edition.
Hotelling, H. (1936). Relations between two sets of variates. _Biometrika_, 28(3/4):321–377.
Jin, J., Ke, Z. T., and Wang, W. (2015). Phase transitions for high dimensional clustering
and related problems. _arXiv preprint arXiv:1502.06952_ .
Jin, J. and Wang, W. (2016). Important feature pca for high dimensional clustering (with
discussion). _Annals of Statistics_, 44:2323–2359.
Johnstone, I. M. and Lu, A. Y. (2009). On consistency and sparsity for principal components analysis in high dimensions. _Journal of the American Statistical Association_,
104(486):682–693.
Keshavan, R. H., Montanari, A., and Oh, S. (2010). Matrix completion from noisy entries.
_J. Mach. Learn. Res._, 11(1):2057–2078.
Lei, J. and Rinaldo, A. (2015). Consistency of spectral clustering in stochastic block
models. _The Annals of Statistics_, 43(1):215–237.
Liu, Z. and Vandenberghe, L. (2009). Interior-point method for nuclear norm approximation with application to system identification. _SIAM Journal on Matrix Analysis and_
_Applications_, 31(3):1235–1256.
Ma, Z. and Li, X. (2016). Subspace perspective on canonical correlation analysis: Dimension reduction and minimax rates. _arXiv preprint arXiv:1605.03662_ .
Melamed, D. (2014). Community structures in bipartite networks: A dual-projection
approach. _PloS one_, 9(5):e97823.
O’Rourke, S., Vu, V., and Wang, K. (2013). Random perturbation of low rank matrices:
Improving classical bounds. _arXiv preprint arXiv:1311.2657_ .
Rohe, K., Chatterjee, S., and Yu, B. (2011). Spectral clustering and the high-dimensional


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 31


stochastic blockmodel. _The Annals of Statistics_, 39:1878–1915.
Rudelson, M. and Vershynin, R. (2013). Hanson-wright inequality and sub-gaussian concentration. _Electron. Commun. Probab_, 18(0).
Shabalin, A. A. and Nobel, A. B. (2013). Reconstruction of a low-rank matrix in the
presence of gaussian noise. _Journal of Multivariate Analysis_, 118:67–76.
Singer, A. and Cucuringu, M. (2010). Uniqueness of low-rank matrix completion by rigidity
theory. _SIAM Journal on Matrix Analysis and Applications_, 31(4):1621–1641.
Stewart, G. W. (1991). Perturbation theory for the singular value decomposition. _SVD_
_and Signal Processing, II: Algorithms, Analysis and Applications_, pages 99–109.
Stewart, M. (2006). Perturbation of the svd in the presence of small singular values.
_Linear algebra and its applications_, 419(1):53–77.
Sun, R. and Luo, Z.-Q. (2015). Guaranteed matrix completion via nonconvex factorization.
In _Foundations of Computer Science (FOCS), 2015 IEEE 56th Annual Symposium on_,
pages 270–289. IEEE.
Tao, T. (2012). _Topics in random matrix theory_, volume 132. American Mathematical
Society Providence, RI.
Vershynin, R. (2011). Spectral norm of products of random and deterministic matrices.
_Probability theory and related fields_, 150(3-4):471–509.
Vershynin, R. (2012). Introduction to the non-asymptotic analysis of random matrices.
In _Compressed sensing_, pages 210–268. Cambridge Univ. Press, Cambridge.
von Luxburg, U., Belkin, M., and Bousquet, O. (2008). Consistency of spectral clustering.
_The Annals of Statistics_, pages 555–586.
Vu, V. (2011). Singular vectors under random perturbation. _Random Structures & Algo-_
_rithms_, 39(4):526–538.
Wang, R. (2015). Singular vector perturbation under gaussian noise. _SIAM Journal on_
_Matrix Analysis and Applications_, 36(1):158–177.
Wedin, P. (1972). Perturbation bounds in connection with singular value decomposition.
_BIT_, 12:99–111.
Weyl, H. (1912). Das asymptotische verteilungsgesetz der eigenwerte linearer partieller
differentialgleichungen (mit einer anwendung auf die theorie der hohlraumstrahlung).
_Mathematische Annalen_, 71(4):441–479.
Witten, D. M., Tibshirani, R., and Hastie, T. (2009). A penalized matrix decomposition,
with applications to sparse principal components and canonical correlation analysis.
_Biostatistics_, pages 515–534.
Yang, D., Ma, Z., and Buja, A. (2014). Rate optimal denoising of simultaneously sparse
and low rank matrices. _arXiv preprint arXiv:1405.0338_ .
Yu, B. (1997). Assouad, fano, and le cam. In _Festschrift for Lucien Le Cam_, pages 423–435.
Springer.
Yu, Y., Wang, T., and Samworth, R. J. (2015). A useful variant of the davis-kahan theorem
for statisticians. _Biometrika_, 102:315–323.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 1

## **SUPPLEMENT TO “RATE-OPTIMAL PERTURBATION** **BOUNDS FOR SINGULAR SUBSPACES WITH APPLICATIONS** **TO HIGH-DIMENSIONAL STATISTICS”**


BY T. TONY CAI [1] and ANRU ZHANG


University of Pennsylvania and University of Wisconsin-Madison


In this supplementary material, we provide the proofs for Theorem 2,
Corollary 1, matrix denoising, high-dimensional clustering, canonical correlation analysis and all the technical lemmas.


**1. Additional Proofs.**


Proof of Theorem 2. The construction of lower bound relies on the

following design of 2-by-2 blocks.


Lemma 3 (SVD of 2-by-2 matrices).


_1. Suppose 2-by-2 matrix B satisfies_


_a_ _b_
_B_ = 0 _d_ _,_ _a, b, d ≥_ 0 _,_ _a_ [2] _≤_ _b_ [2] + _d_ [2] _._
� �



_v_ 11 _v_ 12
_Suppose V_ =
� _v_ 21 _v_ 22



_is the right singular vectors of B, then_
�



1
(1.1) _|v_ 12 _|_ = _|v_ 21 _| ≥_
~~_√_~~ 2 _[.]_



_2. Suppose 2-by-2 matrix A satisfies_


_a_ _b_
(1.2) _A_ = _,_ _a, b, c, d ≥_ 0 _,_ _a_ [2] _> d_ [2] + _b_ [2] + _c_ [2] _._
_c_ _d_
� �



_v_ 11 _v_ 12
_Suppose V_ =
� _v_ 21 _v_ 22



_is the right singular vectors of A, then_
�



1
_|v_ 12 _|_ = _|v_ 21 _| ≥_
~~_√_~~ 10



_ab_ + _cd_

_._
� _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] _[∧]_ [1] �



1 The research was supported in part by NSF FRG Grant DMS-0854973, NSF Grant
DMS-1208982 and NIH Grant R01 CA127334-05.


2 T. T. CAI AND A. ZHANG


The proof of Lemma 3 is provided later.




_•_ Now we consider the situation when _α_ [2] _≤_ _β_ [2] + _z_ 12 [2] _[∧][z]_ 21 [2] [. Clearly,] _[ α]_ [2] _[ ≤]_ _[β]_ [2] [+]
_z_ 12 [2] [under this setting. We can write down the singular value decomposition]
for the following matrix,


⊺

_α_ _z_ 12 _u_ 11 _u_ 12 _σ_ 1 0 _v_ 11 _v_ 12

= _·_ _·_ _,_

�0 _β_ � � _u_ 21 _u_ 22 � � 0 _σ_ 2 � � _v_ 21 _v_ 22 �



_u_ 11 _u_ 12
=
� � _u_ 21 _u_ 22



_σ_ 1 0

_·_
� � 0 _σ_ 2



_v_ 11 _v_ 12

_·_
� � _v_ 21 _v_ 22



⊺

_,_
�



By the first part of Lemma 3, we have


1
(1.3) _|v_ 12 _|_ = _|v_ 21 _| ≥_
~~_√_~~ 2 _[.]_


We construct the following matrices



_X_ 1 =


_Z_ 1 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _σ_ 1 _u_ 11 _v_ 11 _I_ _r_ _σ_ 1 _u_ 11 _v_ 21 _I_ _r_ 0
_r_ _σ_ 1 _u_ 21 _v_ 11 _I_ _r_ _σ_ 1 _u_ 21 _v_ 21 _I_ _r_ 0 _,_
_p_ 1 _−_ 2 _r_ � 0 0 0 �


_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _σ_ 2 _u_ 12 _v_ 12 _I_ _r_ _σ_ 2 _u_ 12 _v_ 22 _I_ _r_ 0
_r_ _σ_ 2 _u_ 22 _v_ 12 _I_ _r_ _σ_ 2 _u_ 22 _v_ 22 _I_ _r_ 0 ;
_p_ 1 _−_ 2 _r_ � 0 0 0 �



(1.4)


_X_ 2 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _αI_ _r_ 0 0
_r_ 0 0 0 _,_ _Z_ 2 =
_p_ 1 _−_ 2 _r_ � 0 0 0 �



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ 0 _z_ 12 _I_ _r_ 0
_r_ 0 _βI_ _r_ 0 _._
_p_ 1 _−_ 2 _r_ �0 0 0 �



We can see rank( _X_ 1 ) = rank( _X_ 2 ) = _r_,



_X_ 1 + _Z_ 1 = _X_ 2 + _Z_ 2 = _X_ [ˆ] =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _αI_ _r_ _z_ 12 _I_ _r_ 0
_r_ 0 _βI_ _r_ 0 _._
_p_ 1 _−_ 2 _r_ � 0 0 0 �



It is easy to check that both ( _X_ 1 _, Z_ 1 ) and ( _X_ 2 _, Z_ 2 ) are both in _F_ _r,α,β,z_ 12 _,z_ 21 .
Assume _V_ 1, _V_ 2 are the first _r_ singular vectors of _X_ 1 and _X_ 2, respectively.


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 3


Based on the structure of _X_ 1 _, X_ 2, we know



_V_ 1 =



_r_ _v_ 11 _I_ _r_
_p_ 2 _−r_ 2 _r_ � _v_ 21 0 _I_ _r_ � _,_ _V_ 2 =



_r_ _I_ _r_
_p_ 2 _−_ _r_ � 0 � _[.]_



Now based on the observations _X_ [ˆ], for any estimator _V_ [˜] for the right singular space, we have


max _∥_ sin Θ( _V, V_ [˜] 1 ) _∥, ∥_ sin Θ( _V, V_ [˜] 2 ) _∥_
� �



(1.5)



_≥_ [1]

2



_∥_ sin Θ( _V, V_ [˜] 1 ) _∥_ + _∥_ sin Θ( _V, V_ [˜] 2 ) _∥_
� �



Lemma 1 1 (1.3) 1

_≥_ _≥_
2 _[∥]_ [sin Θ(] _[V]_ [1] _[, V]_ [2] [)] _[∥]_ 2 ~~_√_~~



2 _[,]_



which gives us (2.8).

_•_ Next, we consider the situation when _α_ [2] _> β_ [2] + _z_ 12 [2] [+] _[z]_ 21 [2] [. We first assume]
_αz_ 12 _≥_ _βz_ 21 . Since _α_ [2] _> β_ [2] + _z_ 12 [2] [+] _[ z]_ 21 [2] [, we have]


_αz_ 12 _≥_ ( _αz_ 12 + _βz_ 21 ) _/_ 2 _,_ _α_ [2] _−_ _β_ [2] _≤_ 2( _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] [)] _[.]_



Suppose we have the following singular value decomposition for 2-by-2

matrix

⊺

_α_ _z_ 12 _u_ 11 _u_ 12 _σ_ 1 0 _v_ 11 _v_ 12

= _·_ _·_ _,_

�0 _β_ � � _u_ 21 _u_ 22 � � 0 _σ_ 2 � � _v_ 21 _v_ 22 �



_v_ 11 _v_ 12

_·_
� � _v_ 21 _v_ 22



⊺

_,_
�



_u_ 11 _u_ 12
=
� � _u_ 21 _u_ 22



_σ_ 1 0

_·_
� � 0 _σ_ 2



by Lemma 3, we have


(1.6)



( _αz_ 12 + _βz_ 21 ) _/_ 2
� 2( _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] [)] _[ ∧]_ [1] �



_|v_ 12 _|_ = _|v_ 21 _| ≥_ [1]
~~_√_~~ 10



_αz_ 12 1
� _α_ [2] _−_ _β_ [2] _[∧]_ [1] � _≥_ ~~_√_~~ 10



1
_≥_
4 ~~_√_~~ 10



_αz_ 12 + _βz_ 21
_∧_ 1 _._
� _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] �



We construct the following matrices



_X_ 1 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _σ_ 1 _u_ 11 _v_ 11 _I_ _r_ _σ_ 1 _u_ 11 _v_ 21 _I_ _r_ 0
_r_ _σ_ 1 _u_ 21 _v_ 11 _I_ _r_ _σ_ 1 _u_ 21 _v_ 21 _I_ _r_ 0 _,_
_p_ 1 _−_ 2 _r_ � 0 0 0 �


4 T. T. CAI AND A. ZHANG



_Z_ 1 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _σ_ 2 _u_ 12 _v_ 12 _I_ _r_ _σ_ 2 _u_ 12 _v_ 22 _I_ _r_ 0
_r_ _σ_ 2 _u_ 22 _v_ 12 _I_ _r_ _σ_ 2 _u_ 22 _v_ 22 _I_ _r_ 0 ;
_p_ 1 _−_ 2 _r_ � 0 0 0 �



(1.7)


_X_ 2 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _αI_ _r_ 0 0
_r_ 0 0 0 _,_ _Z_ 2 =
_p_ 1 _−_ 2 _r_ � 0 0 0 �



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ 0 _z_ 12 _I_ _r_ 0
_r_ 0 _βI_ _r_ 0 _._
_p_ 1 _−_ 2 _r_ �0 0 0 �



We can see rank( _X_ 1 ) = rank( _X_ 2 ) = _r_,



_X_ 1 + _Z_ 1 = _X_ 2 + _Z_ 2 = _X_ [ˆ] =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _αI_ _r_ _z_ 12 _I_ _r_ 0
_r_ 0 _βI_ _r_ 0 _._
_p_ 1 _−_ 2 _r_ � 0 0 0 �



It is easy to check that both ( _X_ 1 _, Z_ 1 ) and ( _X_ 2 _, Z_ 2 ) are both in _F_ _r,α,β,z_ 12 _,z_ 21 .
Assume _V_ 1, _V_ 2 are the first _r_ singular vectors of _X_ 1 and _X_ 2, respectively.
Based on the structure of _X_ 1 _, X_ 2, we know



_V_ 1 =



_r_ _v_ 11 _I_ _r_
_p_ 2 _−r_ 2 _r_ � _v_ 21 0 _I_ _r_ � _,_ _V_ 2 =



_r_ _I_ _r_
_p_ 2 _−_ _r_ � 0 � _[.]_



Now based on the observations _X_ [ˆ], for any estimator _V_ [˜] for the right singular space, we have


max _∥_ sin Θ( _V, V_ [˜] 1 ) _∥, ∥_ sin Θ( _V, V_ [˜] 2 ) _∥_
� �



(1.8)



_≥_ [1]

2



_∥_ sin Θ( _V, V_ [˜] 1 ) _∥_ + _∥_ sin Θ( _V, V_ [˜] 2 ) _∥_
� �



1 1 (1.6) 1

_≥_ _≥_
2 _[∥]_ [sin Θ(] _[V]_ [1] _[, V]_ [2] [)] _[∥]_ 8 ~~_√_~~



Lemma 1



10



_αz_ 12 + _βz_ 21
_∧_ 1 _,_
� _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] �



which gives us (2.9).
For the situation where _αz_ 12 _≤_ _βz_ 21, we consider the singular decomposition of the 2-by-2 matrix


⊺

_α_ 0 _u_ 11 _u_ 12 _σ_ 1 0 _v_ 11 _v_ 12

= _·_ _·_ _,_

� _z_ 21 _β_ � � _u_ 21 _u_ 22 � � 0 _σ_ 2 � � _v_ 21 _v_ 22 �



_u_ 11 _u_ 12
=
� � _u_ 21 _u_ 22



_σ_ 1 0

_·_
� � 0 _σ_ 2



_v_ 11 _v_ 12

_·_
� � _v_ 21 _v_ 22



⊺

_,_
�


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 5


we can construct _X_ 1 _, Z_ 1 _, X_ 2 _, Z_ 2 similarly as the previous case, and derive
(2.9).
Then we consider the worst-case matrices. Define



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ 0 _z_ 12 _I_ _r_ 0
_r_ _z_ 21 _I_ _r_ _βI_ _r_ 0 _,_
_p_ 1 _−_ 2 _r_ � 0 0 0 �



_X_ 3 =



_r_ _r_ _p_ 2 _−_ 2 _r_
_r_ _αI_ _r_ 0 0
_r_ 0 0 0 _,_ _Z_ 3 =
_p_ 1 _−_ 2 _r_ � 0 0 0 �



and consider the singular value decomposition of _X_ 3 + _Z_ 3, which is essentially the singular value decomposition of the following 2-by-2 matrix,


⊺

_α_ _z_ 12 _u_ 11 _u_ 12 _σ_ 1 0 _v_ 11 _v_ 12

= _·_ _·_ _,_

� _z_ 21 _β_ � � _u_ 21 _u_ 22 � � 0 _σ_ 2 � � _v_ 21 _v_ 22 �



_v_ 11 _v_ 12

_·_
� � _v_ 21 _v_ 22



⊺

_,_
�



_u_ 11 _u_ 12
=
� � _u_ 21 _u_ 22



_σ_ 1 0

_·_
� � 0 _σ_ 2



by Lemma 3, we have



1 _αz_ 12 + _βz_ 21

(1.9) _|v_ 12 _|_ = _|v_ 21 _| ≥_ ~~_√_~~ 10 � _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] _∧_ 1� _._



1
_|v_ 12 _|_ = _|v_ 21 _| ≥_
~~_√_~~



10



If _V_ 3 and _V_ [ˆ] 3 are the leading _r_ right singular vectors of _X_ 3 and _X_ 3 + _Z_ 3
respectively, then



1
sin Θ( ˆ _V_ 3 _, V_ 3 ) = _|v_ 12 _| ≥_
��� ��� ~~_√_~~ 10



_αz_ 12 + _βz_ 21
_∧_ 1 _,_
� _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] �



and by upper bound result in Theorem 1,

_αz_ 12 + _βz_ 21
���sin Θ( ˆ _V_ 3 _, V_ 3 )��� _≤_ _α_ [2] _−_ _β_ [2] _−_ _z_ 12 [2] _[∧]_ _[z]_ 21 [2] _∧_ 1 _._


_•_ For the proof of the Frobenius norm loss lower bound (2.7), the construction of pairs ( _X_ 1 _, Z_ 1 ) _,_ ( _X_ 2 _, Z_ 2 ) is essentially the same. We just need to
construct similar pairs of ( _X_ 1 _, Z_ 1 ) and ( _X_ 2 _, Z_ 2 ) by replacing _z_ 12 _, z_ 21 by
_z_ ˜ 12 _/_ _[√]_ ~~_r_~~ _,_ ˜ _z_ 21 _/_ _[√]_ ~~_r_~~ ~~.~~ Based on the similar calculation, we can finish the proof
of Theorem 2.


1.1. _Proof of Corollary 1._ By Theorem 1, we have
(1.10)
sin Θ( ˆ _V_ ( _i−_ 1) _, V_ ( _i−_ 1) ) _≤_ _α_ [(] _[i][−]_ [1)] _z_ 12 [(] _[i][−]_ [1)] + _β_ [(] _[i][−]_ [1)] _z_ 21 [(] _[i][−]_ [1)]
��� ��� ( _α_ [(] _[i][−]_ [1)] ) [2] _−_ ( _β_ [(] _[i][−]_ [1)] ) [2] _−_ ( _z_ 12 [(] _[i][−]_ [1)] ) [2] _∧_ ( _z_ 21 [(] _[i][−]_ [1)] ) [2] _[∧]_ [1] _[,]_


6 T. T. CAI AND A. ZHANG


_α_ [(] _[j]_ [)] _z_ [(] _[j]_ [)] _[β]_ [(] _[j]_ [)] _[z]_ [(] _[j]_ [)]
(1.11) sin Θ( ˆ _V_ ( _j_ ) _, V_ ( _j_ ) ) _≤_ 12 [+] 21
��� ��� ( _α_ [(] _[j]_ [)] ) [2] _−_ ( _β_ [(] _[j]_ [)] ) [2] _−_ ( _z_ 12 [(] _[j]_ [)] [)] [2] _[ ∧]_ [(] _[z]_ 21 [(] _[j]_ [)] [)] [2] _[∧]_ [1] _[.]_


Next, based on Lemma 1, we know
���sin Θ( ˆ _V_ [: _,i_ : _j_ ] _, V_ [: _,i_ : _j_ ] )��� = ��� _V_ ⊺[: _,i_ : _j_ ] _[V]_ [ˆ] [[:] _[,i]_ [:] _[j]_ []] _[⊥]_ ���



_≤_ � _∥V_ [: [⊺] _,i_ : _j_ ] _[V]_ [ˆ] [[:] _[,]_ [1:(] _[i][−]_ [1)]] _[∥]_ [2] [ +] _[ ∥][V]_ [ ⊺] [: _,i_ : _j_ ] _[V]_ [ˆ] [[:] _[,]_ [(] _[j]_ [+1):] _[p]_ [2] []] _[∥]_ [2] [�] [1] _[/]_ [2]
������



=



_V_ [⊺]

[: _,i_ : _j_ ] _[V]_ [ˆ] [[:] _[,]_ [1:(] _[i][−]_ [1)]]
_V_ [⊺]

������ [: _,i_ : _j_ ] _[V]_ [ˆ] [[:] _[,]_ [(] _[j]_ [+1):] _[p]_ [2] []]



_≤_ � _∥V_ [: [⊺] _,i_ : _p_ 2 ] _[V]_ [ˆ] [[:] _[,]_ [1:(] _[i][−]_ [1)]] _[∥]_ [2] [ +] _[ ∥][V]_ [ ⊺] [: _,_ 1: _j_ ] _[V]_ [ˆ] [[:] _[,]_ [(] _[j]_ [+1):] _[p]_ [2] []] _[∥]_ [2] [�] [1] _[/]_ [2]

= � _∥V_ [: [⊺] _,_ 1:( _i−_ 1)] _⊥_ _[V]_ [ˆ] [[:] _[,]_ [1:(] _[i][−]_ [1)]] _[∥]_ [2] [ +] _[ ∥][V]_ [ ⊺] [: _,_ 1: _j_ ] _[V]_ [ˆ] [[:] _[,]_ [1:] _[j]_ []] _[⊥]_ _[∥]_ [2] [�] [1] _[/]_ [2]

= � _∥_ sin Θ( _V_ [ˆ] ( _i−_ 1) _, V_ ( _i−_ 1) ) _∥_ [2] + _∥_ sin Θ( _V_ [ˆ] ( _j_ ) _, V_ ( _j_ ) ) _∥_ [2] [�] [1] _[/]_ [2]


Particularly when _j_ = _i_, by Lemma 1 we have _∥_ sin Θ( _v_ _i_ _,_ ˆ _v_ _i_ ) _∥_ = ~~�~~ 1 _−_ (ˆ _v_ _i_ ~~[⊺]~~ _[v]_ _[i]_ [)] [2] _[.]_

Combining (1.10), (1.11), and the inequality above, we have finished the
proof for this corollary.


1.2. _Proofs for Matrix Denoising._ We prove all results in Section 3 in
this section.


Proof of Theorem 3. We need some technical results for the proof of
this theorem. Specifically, the following lemma relating to random matrix
theory plays an important part.


Lemma 4 (Properties related to Random Matrix Theory). _Suppose X ∈_
R _[p]_ [1] _[×][p]_ [2] _is a rank-r matrix with right singular space as V_ _∈_ O _n,r_ _, Z ∈_

R _[p]_ [1] _[×][p]_ [2] _, Z_ _[iid]_ _∼G_ _τ_ _is an i.i.d. sub-Gaussian random matrix. Y_ = _X_ + _Z. Then_
_there exists constants C, c only depending on τ such that for any x >_ 0 _,_
(1.12)
P � _σ_ _r_ [2] [(] _[Y V]_ [ )] _[ ≥]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)(1] _[ −]_ _[x]_ [)] � _≤_ _C_ exp � _Cr −_ _c_ � _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] � _x_ [2] _∧_ _x_ � _,_


(1.13) P � _σ_ _r_ [2] +1 [(] _[Y]_ [ )] _[ ≤]_ _[p]_ [1] [(1 +] _[ x]_ [)] � _≤_ _C_ exp � _Cp_ 2 _−_ _cp_ 1 _· x_ [2] _∧_ _x_ � _._


_Moreover, there exists C_ gap _, C, c which only depends on τ_ _, such that whenever_
_σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] _[p]_ [2] _[, for any][ x >]_ [ 0] _[ we have]_


(1.14)

P ( _∥_ P _Y V_ _Y V_ _⊥_ _∥≤_ _x_ )


_≥_ 1 _−_ _C_ exp _Cp_ 2 _−_ _c_ min( _x_ [2] _,_ ~~�~~ _σ_ _r_ [2] ( _X_ ) + _p_ 1 _x_ ) _−_ _C_ exp � _−c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] � _._
� �


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 7


The following lemma provides an upper bound for matrix spectral norm
based on an _ε_ -net argument.


Lemma 5 ( _ε_ -net Argument for Unit Ball). _For any p ≥_ 1 _, denote_ B _[p]_ =
_{x_ : _x ∈_ R _[p]_ _, ∥x∥_ 2 _≤_ 1 _} as the p-dimensional unit ball. Suppose K ∈_ R _[p]_ [1] _[×][p]_ [2]

_is a random matrix. Then we have for t >_ 0 _,_


(1.15) P ( _∥K∥≥_ 3 _t_ ) _≤_ 7 _[p]_ [1] [+] _[p]_ [2] _·_ _u∈_ B max _[p]_ [1] _,v∈_ B _[p]_ [2] [P][ (] _[|][u]_ [⊺] _[Kv][| ≥]_ _[t]_ [)] _[ .]_


Now we start to prove Theorem 3. We only need to focus on the losses
of _V_ [ˆ] since the results for _U_ [ˆ] are symmetric. Besides, we only need to prove
the spectral norm loss, as sin Θ( _V, V_ [ˆ] ) is a _r × r_ matrix _∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ [2] _F_ _[≤]_
_r∥_ sin Θ( _V,_ _V_ [ˆ] ) _∥_ [2] . Throughout the proof, we use _C_ and _c_ to represent generic
“large” and “small” constants, respectively. These constants _C, c_ are uniform
and only relying on _τ_, while the actual values may vary in different formulas.
1
Next, we focus on the scenario that _σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] [((] _[p]_ [1] _[p]_ [2] [)] 2 + _p_ 2 ) for some
large constant _C_ gap _>_ 0 only relying on _τ_ . The other case will be considered
later. By Lemma 4, there exists constants _C, c_ only depending on _τ_ such
that



_r_ [(] _[X]_ [)]
P _σ_ _r_ [2] [(] _[Y V]_ [ )] _[ ≤]_ _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] _[−]_ _[σ]_ [2]
� 3



�



_σ_ _r_ [4] [(] _[X]_ [)]
_≤C_ exp � _Cr −_ _c_ min � _σ_ _r_ [2] [(] _[X]_ [)] _[,]_ _σ_ _r_ [2] ( _X_ ) + _p_ 1



_,_
��



3 [1] _[σ]_ _r_ [2] [(] _[X]_ [)] � _≤_ _C_ exp � _Cp_ 2 _−_ _c_ min � _σ_ _r_ [2] [(] _[X]_ [)] _[,]_ _[σ]_ _r_ [4] _p_ [(] 1 _[X]_ [)]



P _σ_ _r_ [2] +1 [(] _[Y]_ [ )] _[ ≥]_ _[p]_ [1] [+] [1]
� 3



_p_ 1



_,_
��



(1.16)

P _{∥_ P _Y V_ _Y V_ _⊥_ _∥≥_ _x}_


_≤C_ exp _Cp_ 2 _−_ _c_ min _x_ [2] _, x_ ~~�~~ _σ_ _r_ [2] ( _X_ ) + _p_ 1 + _C_ exp � _−c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] � _._
� � ��


When _C_ gap is large enough, it holds that _σ_ _r_ [4] [(] _[X]_ [)] _[/]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[ ≥]_ _[Cp]_ [2] [. Then]


_c_ min _σ_ _r_ [4] [(] _[X]_ [)] _, σ_ _r_ [2] [(] _[X]_ [)] _−_ _Cr ≥_ _σ_ _r_ [4] [(] _[X]_ [)] _−_ _Cr ≥_ _[c]_ _σ_ _r_ [4] [(] _[X]_ [)] ;
� _σ_ _r_ [2] ( _X_ ) + _p_ 1 � _σ_ _r_ [2] ( _X_ ) + _p_ 1 2 _σ_ _r_ [2] ( _X_ ) + _p_ 1



_r_ [(] _[X]_ [)]
_c_ min _σ_ _r_ [2] [(] _[X]_ [)] _[,]_ _[σ]_ [4]
� _p_ 1



_−_ _Cp_ 2 _≥_ _c_ _σ_ _r_ [4] [(] _[X]_ [)] _−_ _Cp_ 2 _≥_ _[c]_ _σ_ _r_ [4] [(] _[x]_ [)] _._
� _σ_ _r_ [2] ( _X_ ) + _p_ 1 2 _σ_ _r_ [2] ( _X_ ) + _p_ 1


8 T. T. CAI AND A. ZHANG



_σ_ _r_ [2] ( _X_ ) + _p_ 1 _x_ ) _−_ _Cp_ 2 = _c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[ −]_ _[Cp]_ [2] _[≥]_ _[c]_



_c_ min( _x_ [2] _,_ ~~�~~



� _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] � _,_
2



_σ_ _r_ [4] [(] _[X]_ [)]

_≥_ _[c]_ if _x_ = �

2 _σ_ _r_ [2] ( _X_ ) + _p_ 1



_σ_ _r_ [2] ( _X_ ) + _p_ 1 .



To sum up, if we denote the event _Q_ as



_r_ [(] _[X]_ [)]
_Q_ = � _σ_ _r_ [2] [(] _[Y V]_ [ )] _[ ≥]_ _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] _[−]_ _[σ]_ [2] 3



3 _[σ]_ _r_ [2] [(] _[X]_ [)] _[,]_




_[X]_ [)] _, σ_ _r_ [2] +1 [(] _[Y]_ [ )] _[ ≤]_ _[p]_ [1] [+] [1]

3 3



_∥_ P _Y V_ _Y V_ _⊥_ _∥≤_ ~~�~~ _σ_ _r_ [2] ( _X_ ) + _p_ 1 _,_

�



1
when _σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] [((] _[p]_ [1] _[p]_ [2] [)] 2 + _p_ 2 ) for some large constant _C_ gap _>_ 0,


(1.17) P ( _Q_ _[c]_ ) _≤_ _C_ exp _−c_ _σ_ _r_ [4] [(] _[X]_ [)] _._

� _σ_ _r_ [2] ( _X_ ) + _p_ 1 �


Under event _Q_, we can apply Proposition 1 and obtain



_r_ [(] _[Y V]_ [)] _[∥]_ [P] _[Y V]_ _[Y V]_ _[⊥]_ _[∥]_ [2]
_∥_ sin Θ( _V, V_ [ˆ] ) _∥_ [2] _≤_ _[σ]_ [2]



_σ_ _r_ [4] ( _X_ ) _._




_[σ]_ _r_ [2] [(] _[Y V]_ [)] _[∥]_ [P] _[Y V]_ _[Y V]_ _[⊥]_ _[∥]_ [2] _[≤]_ _[C]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [)][ +] _[p]_ [1] [)] _[∥]_ [P] _[Y V]_ _[Y V]_ _[⊥]_ _[∥]_ [2]

( _σ_ _r_ [2] ( _Y V_ ) _−_ _σ_ _r_ +1 ( _Y_ )) [2] _σ_ _r_ [4] ( _X_ )



_x_ [2]
Here we used the fact that ( _x_ [2] _−y_ [2] ) [2] [is a decreasing function of] _[ x]_ [ and increas-]
ing function of _y_ when _x > y ≥_ 0.
Next we shall note that _∥_ sin Θ( _V, V_ [ˆ] ) _∥≤_ 1 for any _V, V_ [ˆ] _∈_ O _p_ 2 _,r_ . Therefore,



2 2 2
E sin Θ( ˆ _V, V_ ) = E sin Θ( ˆ _V, V_ ) 1 _Q_ + E sin Θ( ˆ _V, V_ ) 1 _Q_ _c_
��� ��� ��� ��� ��� ���



(1.18)

_r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
_≤_ _[C]_ [(] _[σ]_ [2] E _∥_ P _Y V_ _Y V_ _⊥_ _∥_ [2] 1 _Q_ + P( _Q_ _[c]_ ) _._

_σ_ _r_ [4] ( _X_ )



_r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
_≤_ _[C]_ [(] _[σ]_ [2]



By basic property of exponential function,
(1.19)



P( _Q_ _[c]_ ) (1.17 _≤_ ) exp _−c_ _σ_ _r_ [4] [(] _[X]_ [)]
� _σ_ _r_ [2] ( _X_ ) + _p_ 1



_≤_ _C_ _[σ]_ _r_ [2] [(] _[X]_ [)][ +] _[p]_ [1] _≤_ _C_ _[p]_ [2] [(] _[σ]_ _r_ [2] [(] _[X]_ [)][ +] _[p]_ [1] [)] _._
� _σ_ _r_ [4] ( _X_ ) _σ_ _r_ [4] ( _X_ )



It remains to consider E _∥_ P _Y V_ _Y V_ _⊥_ _∥_ [2] 1 _Q_ . Denote _T_ = _∥_ P _Y V_ _Y V_ _⊥_ _∥_ . Applying
Lemma 4 again, we have for some constant _C_ _x_ to be determined a little


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 9


while later that


_∞_
E _T_ [2] 1 _Q_ _≤_ E _T_ [2] 1 _{T_ 2 _≤σ_ _r_ 2 ( _X_ )+ _p_ 1 _}_ = P � _T_ [2] 1 _{T_ 2 _≤σ_ _r_ 2 ( _X_ )+ _p_ 1 _}_ _≥_ _t_ � _dt_
� 0

_σ_ _r_ 2 [(] _[X]_ [)+] _[p]_ [1]
_≤C_ _x_ _p_ 2 + P � _T_ [2] 1 _{T_ 2 _≤σ_ _r_ 2 ( _X_ )+ _p_ 1 _}_ _≥_ _t_ � _dt_
� _C_ _x_ _p_ 2

(1.16) _σ_ _r_ 2 [(] _[X]_ [)+] _[p]_ [1]
_≤_ _C_ _x_ _p_ 2 + _C_ �exp _{Cp_ 2 _−_ _ct} dt_ + exp � _−c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] �� _dt_
� _C_ _x_ _p_ 2

_≤C_ _x_ _p_ 2 + _C_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [) exp] � _−c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] �

+ _C_ exp( _Cp_ 2 ) _·_ exp ( _−cC_ _x_ _p_ 2 ) [1]

_c_

_≤C_ _x_ _p_ 2 + _C_ + _[C]_

_c_ [exp((] _[C][ −]_ _[cC]_ _[x]_ [)] _[p]_ [2] [)] _[.]_


As we could see we can choose _C_ _x_ large enough, but only relying on other
constants _C, c_ in the inequalities above, to ensure that


(1.20) E _T_ [2] 1 _Q_ _≤_ _Cp_ 2


for large constant _C >_ 0 as long as _p_ 2 _≥_ 1. Now, combining (1.18), (1.19),
(1.20) as well as the trivial upper bound 1, we obtain


2 _r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
E sin Θ( ˆ _V, V_ ) _≤_ _[C][p]_ [2] [(] _[σ]_ [2] _∧_ 1 _._
��� ��� _σ_ _r_ [4] ( _X_ )


1
as long as _σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] ( _p_ 1 _p_ 2 ) 2 + _p_ 2 for some large enough _C_ gap .
� �


1
Finally when _σ_ _r_ [2] [(] _[X]_ [)] _[ < C]_ [gap] ( _p_ 1 _p_ 2 ) 2 + _p_ 2, we have
� �



1
_p_ 2 ( _σ_ _r_ [2] _σ_ [(] _r_ [4] _[X]_ ( _X_ [)][ +] ) _[p]_ [1] [)] _≥_ _[p]_ [2] [(] _[p]_ [2][1] [ +] _[ C]_ [g][a][p] [(] _[p]_ [1] _[p]_ [2][2] [)] 2 + 21 _C_ g [3] a _[/]_ p [2] _p_ 2 )



(1.21)


then



_C_ gap [2] ( _p_ 1 _p_ 2 + _p_ [2] 2 [+ 2] _[p]_



1
1 2 _[p]_ [3] 2 _[/]_ [2] )



1
= _[p]_ _C_ [1] gap [2] [ +] _[ C]_ ~~�~~ _p_ [g][a] 1 [p] + 2( [(] _[p]_ [1] _[p]_ _p_ [2] 1 [)] _p_ 2 2 +) 12 _C_ + g _p_ ap2 _p_ ~~�~~ 2 _≥_ min �1 _,_ _C_ gap 1



_,_
�



2 _r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
E sin _θ_ ( ˆ _V, V_ ) _≤_ 1 _≤_ _[C][p]_ [2] [(] _[σ]_ [2] _∧_ 1
��� ��� _σ_ _r_ [4] ( _X_ )

when _C ≥_ min _[−]_ [1] (1 _,_ 1 _/C_ gap ). In summary, no matter what value _σ_ _r_ [2] [(] _[X]_ [)]
takes, we always have


2 _r_ [(] _[X]_ [)][ +] _[p]_ [1] [)]
E sin Θ( ˆ _V, V_ ) _≤_ _[C][p]_ [2] [(] _[σ]_ [2] _∧_ 1 _._
��� ��� _σ_ _r_ [4] ( _X_ )


10 T. T. CAI AND A. ZHANG


Proof of Theorem 4.. Since _∥_ sin Θ( _V, V_ [ˆ] ) _∥≥_ ~~_√_~~ 1 ~~_r_~~ _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_, we
only need to prove the Frobenius norm lower bound. Particularly we focus
_iid_
on the case with Gaussian noise with mean 0 and variance 1, i.e. _Z_ _ij_ _∼_
_N_ (0 _,_ 1). The technical tool we use to develop such lower bound include the
generalized Fano’s Lemma.
One interesting aspect of this singular space estimation problem is that,
multiple sampling distribution _P_ _X_ correspond to one target parameter _V_ .
In order to proceed, we select one “representative”, either a distribution _P_ _V_
or a mixture distribution _P_ [¯] _V,t_ for each _V_, and consider the estimation with
samples from “representative” distribution. In order to bridge the representative estimation and the original estimation problem, we introduce the
following lower bound lemma.


Lemma 6. _Let {P_ _θ_ : _θ ∈_ Θ _} be a set of probability measures in Euclidean_
_space_ R _[p]_ _, let T_ : Θ _→_ Φ _be a function which maps θ into another metric_
_space φ ∈_ (Φ _, d_ ) _. For each φ ∈_ Φ _, denote co_ ( _P_ _φ_ ) _as the convex hull of the set_
_of probability measures P_ _φ_ = _{P_ _θ_ : _T_ ( _θ_ ) = _φ}. If we choose a representative_
_P_ _φ_ _in each co_ ( _P_ _φ_ ) _, then for any estimator_ _φ_ [ˆ] _based on the sample generated_
_from P_ _θ_ _with φ_ = _T_ ( _θ_ ) _, and any estimator_ _φ_ [ˆ] _[′]_ _based on the samples generated_
_from P_ _θ_ _φ_ _, we have_


(1.22) inf ˆ sup E _P_ _θ_ [ _d_ [2] ( _T_ ( _θ_ ) _,_ _φ_ [ˆ] )] _≥_ inf ˆ sup E _P_ _φ_ [ _d_ [2] ( _φ,_ _φ_ [ˆ] )] _._
_φ_ _θ∈_ Θ _φ_ _φ∈_ Φ


We will prove this theorem separately in two cases according to _t_ : _t_ [2] _≤_
_p_ 1 _/_ 4 or _t_ [2] _> p_ 1 _/_ 4.


_•_ First we consider when _t_ [2] _≤_ _p_ 1 _/_ 4. For each _V ∈_ O _p_ 2 _,r_, we define the
following class of density _P_ _Y_, where _Y_ = _X_ + _Z_ and the right singular
space of _X_ is _V_ .
(1.23)



�



_P_ _V,t_ =



_P_ _Y_ : _[Y][ ∈]_ [R] _[p]_ [1] _[×][p]_ [2] _[, Y]_ [ =] _[ X]_ [ +] _[ Z, X]_ [ is fixed] _[, Z]_ _[ iid]_ _∼_ _N_ (0 _,_ 1) _,_

� _X ∈F_ _r,t_ _,_ the right singular vectors of _X_ is _V O, O ∈_ O _r_



_._



For each _V ∈_ O _p_ 2 _,r_, we construct the following Gaussian mixture mea

PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 11


sure _P_ [¯] _V,t_ _∈_ R _[p]_ [1] _[×][p]_ [2] .


(1.24)


_̸_



_P_ ¯ _V,t_ ( _Y_ ) = _C_ _V,t_


_̸_



� _W_ _∈_ R _[p]_ [1] _[×][r]_ : _σ_ min ( _W_ ) _≥_ 1 _/_ 2


_̸_



1

[exp(] _[−∥][Y][ −]_ [2] _[tWV]_ [ ⊺] _[∥]_ _F_ [2] _[/]_ [2)]
(2 _π_ ) _[p]_ [1] _[p]_ [2] _[/]_ [2]


_̸_




_·_ ( _[p]_ [1] _F_ _[/]_ [2)] _[dW.]_

2 _π_ [)] _[p]_ [1] _[r/]_ [2] [ exp(] _[−][p]_ [1] _[∥][W]_ _[∥]_ [2]


Here _C_ _V,t_ is the constant which normalizes the integral and makes _P_ [¯] _V,t_
a valid probability density. To be specific

_C_ _V,t_ _[−]_ [1] [=] � _Y_ _P_ _V,t_ ( _Y_ ) _dY_ = P � _σ_ min ( _W_ ) _≥_ 1 _/_ 2��� _W ∈_ R _p_ 1 _×r_ _, W_ _iid_ _∼_ _N_ (0 _,_ 1 _/p_ 1 )� _._


Moreover, since 2 _tWV_ [⊺] is rank- _r_ with the least singular value no less
than _t_ in the event that _σ_ min ( _W_ ) _≥_ 1 _/_ 2, _P_ [¯] _V,t_ is a mixture density of
_P_ _V,t_, i.e. _P_ [¯] _V,t_ _∈_ _co_ ( _P_ _V,t_ ).
The following lemma, whose proof is provided in the supplementary
materials, gives a upper bound for the KL divergence between _P_ [¯] _V,t_
and _P_ [¯] _V_ _′_ _,t_ .


Lemma 7. _Under the assumption of Theorem 4 and t_ [2] _≤_ _p_ 1 _/_ 4 _, for_
_any V, V_ _[′]_ _∈_ O _p_ 2 _,r_ _, we have_


8 _t_ [4]
(1.25) _D_ ( _P_ [¯] _V,t_ _||P_ [¯] _V_ _′_ _,t_ ) _≤_ _∥_ sin Θ( _V, V_ _[′]_ ) _∥_ _F_ [2] [+] _[ C]_ _[KL]_ _[,]_

4 _t_ [2] + _p_ 1


_where C_ _KL_ _is some uniform constant._


We then consider the metric (O _p_ 2 _,r_ _,_ _∥_ sin Θ( _·, ·_ ) _∥_ _F_ ). Especially we consider the ball with radius 0 _< ε <_ _√_ 2 _r_ and center _V ∈_ O _p_ 2 _,r_


_B_ ( _V, ε_ ) = � _V_ _[′]_ _∈_ O _p_ 2 _,r_ : ��sin Θ( _V_ _′_ _, V_ )�� _F_ _[≤]_ _[ε]_ � _._


_̸_



Note that _r ≤_ _p_ 2 _/_ 2, _∥_ sin Θ( _V_ _[′]_ _, V_ ) _∥_ _F_ = 1
~~_√_~~


_̸_



Note that _r ≤_ _p_ 2 _/_ 2, _∥_ sin Θ( _V_ _[′]_ _, V_ ) _∥_ _F_ = ~~_√_~~ 2 _[∥][V]_ _[ ′]_ _[V]_ [ ⊺] _[−]_ _[V V]_ [ ⊺] _[∥]_ _[F]_ [, based]

on Lemma 1 in Cai et al. (2013), one can show for any _α ∈_ (0 _,_ 1),
_ε ∈_ (0 _,_ _√_ 2 _r_ ), there exists _V_ 1 _, · · ·, V_ _m_ _⊆_ _B_ ( _V, ε_ ), such that


_̸_



2 _r_ ), there exists _V_ 1 _, · · ·, V_ _m_ _⊆_ _B_ ( _V, ε_ ), such that


_̸_



_c_
_m ≥_
� _α_ _̸_



_r_ ( _p_ 2 _−r_ )
_,_ min
� 1 _≤i_ = _̸_ _j≤m_ _[∥]_ [sin Θ(] _[V]_ _[i]_ _[, V]_ _[j]_ [)] _[∥]_ _[F]_ _[ ≥]_ _[αε.]_



_̸_


By _{V_ 1 _, · · ·, V_ _m_ _} ⊆_ _B_ ( _V, ε_ ), _∥_ sin Θ( _V_ _i_ _, V_ _j_ ) _∥≤_ 2 _ε_, along with (1.25),


¯
_D_ � _P_ _V_ _i_ _,t_ _||P_ ¯ _V_ _j_ _,t_ � _≤_ 4 [32] _t_ [2] _[ε]_ + [2] _[t]_ _p_ [4] 1 + _C_ _KL_ _._


12 T. T. CAI AND A. ZHANG


Fano’s Lemma (Yu, 1997) leads to the following lower bound
(1.26)



inf ˆ sup E _P_ ¯ _V,t_
_V_ _V ∈_ Θ



2
sin Θ( ˆ _V, V_ )
��� ���



2

_F_ _[≥]_ _[α]_ [2] _[ε]_ [2]
�



1 _−_



32 _ε_ [2] _t_ [4]
4 _t_ [2] + _p_ 1 [+] _[ C]_ _[KL]_ [ + log 2]



_r_ ( _p_ 2 _−_ _r_ ) log _α_ ~~_[c]_~~



�



_._



We particularly select


_α_ = _c_ exp( _−_ (1 + _C_ _KL_ + log(2))) _, ε_ =


Note that _r ≤_ _p_ 2 _/_ 2, we further have



~~�~~



_r_ ( _p_ 2 _−_ _r_ )(4 _t_ [2] + _p_ 1 )

_∧_ _√_
32 _t_ [4]



_r_ ( _p_ 2 _−_ _r_ )(4 _t_ [2] + _p_ 1 )



2 _r._



2
sin Θ( ˆ _V, V_ )
��� ���



2 + _p_ 1 )

_∧_ _r_ _._
_t_ [4] �



(1.27) inf ˆ sup E _P_ ¯ _V,t_
_V_ _V ∈_ O _p_ 2 _,r_



2 _rp_ 2 ( _t_ 2 + _p_ 1 )

_F_ _[≥]_ _[c]_ � _t_ [4]



Finally, note that _P_ [¯] _V,t_ is a mixture distribution from _P_ _V,t_ defined in
(1.23). Lemma 6 implies



2
(1.28) inf _V_ ˆ _X_ sup _∈F_ _r,t_ E ���sin Θ( ˆ _V, V_ )��� _F_ _[≥]_ [inf] _V_ ˆ _V_ sup _∈_ Θ E _P_ ¯ _V,t_



2
sin Θ( ˆ _V, V_ )
��� ��� _F_ _[.]_



The two inequalities above together imply the desired lower bound.

_•_ Then we consider when _t_ [2] _> p_ 1 _/_ 4. This case is simpler than the previous as we do not have to mix the multivariate Gaussian measures.

Suppose



_U_ 0 = _I_ _r_
0
�



_∈_ O _p_ 1 _,r_ _._
�



We introduce


(1.29) _X_ _V_ = _tU_ 0 _V_ [⊺] _,_ _V ∈_ O _p_ 2 _,r_ _,_


and denote _P_ _V_ as the probability measure of _Y_ when _Y_ = _X_ _V_ + _Z, Z ∈_
R _[p]_ [1] _[×][p]_ [2], _Z_ _[iid]_ _∼_ _N_ (0 _,_ 1). Based on (1.82), we have



(1.30) _D_ ( _P_ _V_ _||P_ _V_ _′_ ) = [1]

2



2
�� _tU_ 0 _V_ ⊺ _−_ _tU_ 0 ( _V_ _′_ ) ⊺ �� _F_ [=] _[ t]_ [2]

2



2
�� _V −_ _V_ _′_ �� _F_ _[.]_



Based on the same procedure Step 5 in the case _t_ [2] _< p_ 1 _/_ 4, one can
construct the ball of radius _ε_ centered at _V_ 0 _∈_ O _p_ 2 _,r_,


_B_ ( _V_ 0 _, ε_ ) = � _V_ _[′]_ : _∥_ sin Θ( _V_ _[′]_ _, V_ 0 ) _∥_ _F_ _≤_ _ε_ � _._


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 13


and for 0 _< α <_ 1, there exist _{V_ 1 _[′]_ _[,][ · · ·][, V]_ _m_ _[ ′]_ _[} ⊆]_ _[B]_ [(] _[V]_ [0] _[, ε]_ [) such that]


_̸_


_̸_ _̸_


_̸_



_c_
_m ≥_
� _α_ _̸_


_̸_ _̸_


_̸_



_r_ ( _p_ 2 _−r_ )
_,_ max
� 1 _≤i_ = _̸_ _j≤m_


_̸_ _̸_


_̸_



��sin _θ_ ( _V_ _′i_ _[, V]_ _j_ _[ ′]_ [)] �� _F_ _[≥]_ _[εα.]_

_̸_


_̸_ _̸_


_̸_



_̸_


By basic property of sin Θ distances (Lemma 1 in this paper), we can
find _O_ _i_ _∈_ O _r_ such that

�� _V_ 0 _−_ _V_ _′i_ _[O]_ _[i]_ �� _F_ _[≤]_ _√_ 2 ��sin Θ( _V_ 0 _, V_ _i ′_ [)] �� _≤_ _√_ 2 _ε_


_̸_ _̸_


_̸_



_̸_


2 ��sin Θ( _V_ 0 _, V_ _i ′_ [)] �� _≤_ _√_


_̸_ _̸_


_̸_



_̸_


2 _ε_


_̸_ _̸_


_̸_



_̸_


Denote _V_ _i_ = _V_ _i_ _[′]_ _[O]_ _[i]_ _[ ∈]_ [O] _[p]_ 2 _[,r]_ [, then]


(1.30) _t_ [2]
1 _≤_ max _i_ = _̸_ _j≤m_ _[D]_ � _P_ _V_ _i_ _||P_ _V_ _j_ � = 1 _≤_ max _i_ = _̸_ _j≤m_ 2 _[∥][V]_ _[i]_ _[ −]_ _[V]_ _[j]_ _[∥]_ _F_ [2]

_≤_ _[t]_ [2] max _∥V_ 0 _−_ _V_ _i_ _∥_ [2] _F_ [+] _[ ∥][V]_ [0] _[ −]_ _[V]_ _[i]_ _[∥]_ [2] _F_ _≤_ 4 _t_ [2] _ε_ [2] _._

2 1 _≤i_ = _̸_ _j≤m_ [2] � �


Follow the same procedure as Step 5 in the case _t_ [2] _< p_ 1 _/_ 4, we have
(1.31)



_̸_


_̸_ _̸_


_̸_


2

1 _−_ [4] _[ε]_ [2] _[t]_ [2] [ + lo][g][ 2]
_F_ _[≥]_ _[α]_ [2] _[ε]_ [2] � _r_ ( _p_ 2 _−_ _r_ ) log ~~_[c]_~~



_̸_


_̸_ _̸_


_̸_


2
inf ˆ sup E _P_ _V_ sin Θ( ˆ _V, V_ )
_V_ _V ∈_ O _p_ 2 _,r_ ��� ���



_̸_


_̸_ _̸_


_̸_


_r_ ( _p_ 2 _−_ _r_ ) log ~~_[c]_~~



_̸_


_̸_ _̸_


_̸_


_α_



_̸_


_̸_ _̸_


_̸_


_._
�



_̸_


_̸_ _̸_


_̸_


for any _α ∈_ (0 _,_ 1), _ε <_ _√_



_̸_


_̸_ _̸_


_̸_


_[c]_

4 [,] _[ ε]_ [2] [ =] ~~�~~



_̸_


_̸_ _̸_


_̸_


2 _r_ . By selecting _α_ = _[c]_



_̸_


_̸_ _̸_


_̸_


_r_ ( _p_ 2 _−r_ )



_̸_


_̸_ _̸_


_̸_


2 2 _t−_ [2] _r_ _∧_ _√_



_̸_


_̸_ _̸_


_̸_


[ =]

for any _α ∈_ (0 _,_ 1), _ε <_ 2 _r_ . By selecting _α_ = 4 _[c]_ [,] _[ ε]_ 2 _t_ [2] _∧_ 2 _r_,

we have

2 _p_ 2 _r_

inf _V_ ˆ _V_ sup _∈_ Θ E _P_ ¯ _V_ ���sin Θ( ˆ _V, V_ )��� _F_ _[≥]_ _[c]_ � _t_ [2] _[∧]_ _[r]_ � _._



_̸_


_̸_ _̸_


_̸_


2
sin Θ( ˆ _V, V_ )
��� ���



_̸_


_̸_ _̸_


_̸_


2 _r_


_._
_t_ [2] _[∧]_ _[r]_ �



_̸_


_̸_ _̸_


_̸_


_F_ _[≥]_ _[c]_ � _pt_ 2 [2] _r_



_̸_


_̸_ _̸_


_̸_


Based on the assumption that _t_ [2] _> p_ 1 _/_ 4,



_̸_


_̸_ _̸_


_̸_


2 _r_ _[p]_ [1] _[/]_ [8][)]

_[≥]_ _[p]_ [2] _[r]_ [(] _[t]_ [2] _[/]_ [2 +]
_t_ [2] _t_ [4]



_̸_


_̸_ _̸_


_̸_


8



_̸_


_̸_ _̸_


_̸_


_._
�



_̸_


_̸_ _̸_


_̸_


_p_ 2 _r_



_̸_


_̸_ _̸_


_̸_


[2 +] _[p]_ [1] _[/]_ [8][)]

_≥_ [1]
_t_ [4] 8



_̸_


_̸_ _̸_


_̸_


2
_p_ 2 _r_ ( _t_ + _p_ 1 )
� _t_ [4]



_̸_


_̸_ _̸_


_̸_


Similarly by Lemma 6, we have



_̸_


_̸_ _̸_


_̸_


2 _p_ 2 _r_ ( _t_ 2 + _p_ 1 )

_F_ _[≥]_ _[c]_ � _t_ [4]



_̸_


_̸_ _̸_


_̸_


2
inf ˆ sup E sin Θ( ˆ _V, V_ )
_V_ _X∈F_ _r,t_ ��� ��� _F_



_̸_


_̸_ _̸_


_̸_


2 + _p_ 1 )

_∧_ _r_ _._
_t_ [4] �



_̸_


_̸_ _̸_


_̸_


Finally, combining these two cases for _t_ [2] _< p_ 1 _/_ 4 and _t_ [2] _> p_ 1 _/_ 4, we have
finished the proof of Theorem 4.


14 T. T. CAI AND A. ZHANG


1.3. _Proofs in High-dimensional Clustering._


Proof of Theorem 5. Since _EY_ = _µl_ [⊺] = _[√]_ ~~_n∥_~~ _µ∥_ 2 _·_ ~~_√_~~ _l_ ~~_n_~~, by Theorem 3,
we know


_̸_



E


_̸_



2 [�]

_l_
sin Θ _,_ ˆ _v_
� ~~_√n_~~ �����
�����


_̸_



�( _[√]_ ~~_n∥_~~ _µ∥_ 2 ) [2] + _p_ � 2 [+] _[p]_ [)]
_≤_ _[Cn]_ _≤_ _[C]_ [(] _[n][∥][µ][∥]_ [2] _._

( ~~_[√]_~~ ~~_n∥_~~ _µ∥_ 2 ) [4] _n∥µ∥_ [4] 2


_̸_



In addition, the third part of Lemma 1 implies


_̸_



�


_̸_



E min 2
� _o∈{−_ 1 _,_ 1 _}_ _[∥][v]_ [ˆ] _[ −]_ _[o][ ·][ l/][√][n][∥]_ [2]


_̸_



_≤_ 2E
�


_̸_



2

_l_
sin Θ _,_ ˆ _v_ _l_
� ~~_√n_~~ ����� ���
�����


_̸_



2 [+] _[p]_ [)]
_≤_ _[C]_ [(] _[n][∥][µ][∥]_ [2] _._
_n∥µ∥_ [4] 2


_̸_



Finally, by definition that [ˆ] _l_ = sgn(ˆ _v_ ) and _l_ _i_ _/_ _[√]_ ~~_n_~~ = _±_ 1 _/_ _[√]_ ~~_n_~~ ~~,~~ we can obtain


_̸_



_i_ : _[l]_ _[i]_ = _̸_ _π_ (sgn(ˆ _v_ _i_ )) _/_ _[√]_ _n_
� ~~_√n_~~ �����



1

_̸_

_n_ [E][ min] _π_



1
_i_ : _l_ _i_ _̸_ = _π_ ( [ˆ] _l_ _i_ ) _≤_ _̸_
���� ���� _n_ [E][ min] _π_



_̸_

����



_̸_


_l_ _i_ ˆ
_i_ : _−_ _ov_ _i_
� ���� ~~_√n_~~



_̸_


1
_≥_ 1 _/√n_ _≤_ min
���� ����� _n_ [E] _o∈{−_ 1 _,_ 1 _}_



_̸_


1
_≥_ 1 _/√n_ _≤_
���� ����� _n_



_̸_


_n_
�


_i_ =1



_̸_


_≤_ [1] min

_n_ [E] _o∈{−_ 1 _,_ 1 _}_



_̸_


����



_̸_


ˆ 2
� _l_ _i_ _/_ _[√]_ _n −_ _ov_ _i_ � _n_



_̸_


2 [+] _[p]_ [)]

=E min 2 _≤_ _[C]_ [(] _[n][∥][µ][∥]_ [2] _._
� _o∈{−_ 1 _,_ 1 _}_ _[∥][v]_ [ˆ] _[ −]_ _[o][ ·][ l/][√][n][∥]_ [2] � _n∥µ∥_ [4] 2


which has finished the proof of this theorem.


Proof of Theorem 6. We first consider the metric space _{−_ 1 _,_ 1 _}_ _[n]_, where
each pair of _l_ and _−l_ are considered as the same element. For any _l_ 1 _, l_ 2 _∈_
_{−_ 1 _,_ 1 _}_ _[n]_, it is easy to see that

_M_ ( _l_ 1 _, l_ 2 ) = [1]

2 _n_ [min] _[{∥][l]_ [1] _[ −]_ _[l]_ [2] _[∥]_ [1] _[,][ ∥][l]_ [1] [ +] _[ l]_ [2] _[∥]_ [1] _[}][,]_


where _M_ is defined in (4.1). For any three elements _l_ 1 _, l_ 2 _, l_ 3 _∈{−_ 1 _,_ 1 _}_ _[n]_,
since


_∥l_ 1 _−_ _l_ 2 _∥_ 1 _≤_ min _{∥l_ 1 _−_ _l_ 3 _∥_ 1 + _∥l_ 3 _−_ _l_ 2 _∥_ 1 _, ∥l_ 1 + _l_ 3 _∥_ 1 + _∥l_ 3 + _l_ 2 _∥_ 1 _},_


_∥l_ 1 + _l_ 2 _∥_ 1 _≤_ min _{∥l_ 1 _−_ _l_ 3 _∥_ 1 + _∥l_ 3 + _l_ 2 _∥_ 1 _, ∥l_ 1 + _l_ 3 _∥_ 1 + _∥l_ 3 _−_ _l_ 2 _∥_ 1 _},_


we have

_M_ ( _l_ 1 _, l_ 2 ) = [1]

2 _n_ [min] _[ {∥][l]_ [1] _[ −]_ _[l]_ [2] _[∥]_ [1] _[,][ ∥][l]_ [1] [ +] _[ l]_ [2] _[∥]_ [1] _[} ≤]_

_≤_ [1]

2 _n_ [min] _[{∥][l]_ [1] _[ −]_ _[l]_ [3] _[∥]_ [1] [ +] _[ ∥][l]_ [3] _[ −]_ _[l]_ [2] _[∥]_ [1] _[,][ ∥][l]_ [1] [ +] _[ l]_ [3] _[∥]_ [1] [ +] _[ ∥][l]_ [3] [ +] _[ l]_ [2] _[∥]_ [1] _[,]_

+ _∥l_ 1 _−_ _l_ 3 _∥_ 1 + _∥l_ 3 + _l_ 2 _∥_ 1 _, ∥l_ 1 + _l_ 3 _∥_ 1 + _∥l_ 3 _−_ _l_ 2 _∥_ 1 _}_ = _M_ ( _l_ 1 _, l_ 3 ) + _M_ ( _l_ 2 _, l_ 3 ) _._


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 15


In other words, _M_ is a metric in the space of _{−_ 1 _,_ 1 _}_ _[n]_ . Similarly to Lemma 4
in Yu (1997), we can show that there exists universal positive constant _c_ 0 _, C_ _n_ _[′]_ [,]
such that when _n ≥_ _C_ _n_ _[′]_ [, there exists a subset] _[ A]_ [ of] _[ {−]_ [1] _[,]_ [ +1] _[}]_ _[n]_ [ satisfying]
(1.32)
_|A| ≥_ exp( _c_ 0 _n_ ) _,_ and _∀l, l_ _[′]_ _∈{−_ 1 _,_ +1 _}, l ̸_ = _l_ _[′]_ _,_ we have _M_ ( _l, l_ _[′]_ ) _≥_ 1 _/_ 3 _._


1
Let _C_ _KL_ be the uniform constant in Lemma 7. Now, we set _c_ _gap_ = ( _c_ 0 _/_ 300) 4 _, C_ _n_ =
_c_ 0 log 236 _[∨]_ [12] _[C]_ _c_ 0 _[KL]_ _∨_ _C_ _n_ _[′]_ [and denote] _[ t]_ [2] [ =] _[ c]_ _[gap]_ [(] _[p/n]_ [)] 14 . Suppose _n ≥_ _C_ _n_ . Next, we

need to discuss separately according to the value of _t_ [2] .


_•_ When _nt_ [2] _≤_ _p/_ 4, for each _l ∈{−_ 1 _,_ 1 _}_ _[n]_, similarly to the proof to Theorem

6, we define the following class of density _P_ _Y_, where _Y_ = _X_ + _Z_ and the
right singular space of _X_ is _l/_ _[√]_ ~~_n_~~ ~~,~~


_P_ _l,t_ = _P_ _Y_ : _Y ∈_ R _[p][×][n]_ _, Y_ = _X_ + _Z, X_ is fixed _,_
�

_Z_ _[iid]_ _∼_ _N_ (0 _,_ 1) _, X_ = _µl_ [⊺] _, ∥µ∥_ 2 _≥_ _t_ _._
�


We construct the following Gaussian mixture measure _P_ [¯] _l,t_,



_P_ ¯ _l,t_ ( _Y_ ) = _C_ _l,t_



� _µ_ 0 _∈_ R _[p]_ : _∥µ_ 0 _∥≥_ 1 _/_ 2



1

[exp(] _[−∥][Y][ −]_ [2] _[tµ]_ [0] _[l]_ [⊺] _[∥]_ _F_ [2] _[/]_ [2)]
(2 _π_ ) _[pn/]_ [2]



(1.33)



_p_

_·_
� 2 _π_



_p_ 1 _r/_ 2
� exp( _−p∥µ_ 0 _∥_ 2 [2] _[/]_ [2)] _[dµ]_ [0] _[.]_



Here _C_ _l,t_ is the constant which normalize the integral and make _P_ [¯] _l,t_ a
valid probability density. To be specific, _C_ _l,t_ _[−]_ [1] [=] � _Y_ _[P]_ _[l,t]_ [(] _[Y]_ [ )] _[dY.]_ [ Moreover,]

since _∥_ 2 _tµ_ 0 _∥≥_ _t_ in the event that _∥µ_ 0 _∥_ 2 _≥_ 1 _/_ 2, _P_ [¯] _l,t_ is a mixture density
in _P_ _l,t_, i.e. _P_ [¯] _l,t_ _∈_ _co_ ( _P_ _l,t_ ). By Lemma 7, for any two _l, l_ _[′]_ _∈{−_ 1 _,_ 1 _}_ _[n]_, the
KL-divergence between _P_ [¯] _l,t_ and _P_ [¯] _l_ _′_ _,t_ have the following upper bound



8 _n_ [2] _t_ [4]
_D_ ( _P_ [¯] _l,t_ _||P_ [¯] _l_ _′_ _,t_ ) _≤_

4 _nt_ [2] + _p_



_′_ 2
��sin Θ( _l/√n, l_ _/√n_ )�� _F_ [+] _[ C]_ _[KL]_



_gap_ [(] _[p/n]_ [)]

[8] _[n]_ [2] _[t]_ [4]

4 _nt_ [2] + _p_ [+] _[ C]_ _[KL]_ _[ ≤]_ [8] _[n]_ [2] _[c]_ [4] _p_



_≤_ [8] _[n]_ [2] _[t]_ [4]



_p_

+ _C_ _KL_ = 8 _nc_ [4] _gap_ [+] _[ C]_ _[KL]_
_p_



Then the generalized Fano’s lemma (Lemma 3 in Yu (1997)) along with
inequality (1.32) and Lemma 6 lead to



E _M_ ( [ˆ] _l, l_ )



E _M_ ( [ˆ] _l, l_ ) _≥_ inf ˆ _l_ _∥µ∥_ 2 _≤_ max _c_ _gap_ ( _p/n_ ) ~~4~~ 1
_l∈A_



(1.34)



inf ˆ _l_ _∥µ∥_ 2 _≤_ max _c_ _gap_ ( _p/n_ ) ~~4~~ 1
_l∈{−_ 1 _,_ 1 _}_ _[n]_



�



_≥_ [1]

8 _[,]_



(1.32)
_≥_ [1]

6



�



1 _−_ [8] _[nc]_ _g_ [4] _ap_ [+] _[ C]_ _[KL]_ [+ log 2]



_c_ 0 _n_


16 T. T. CAI AND A. ZHANG


where the last inequality holds since _n ≥_ 36 _/_ ( _c_ 0 log 2) _, n ≥_ 36 _C_ _KL_ _/c_ 0, and
1
_c_ _gap_ _≤_ ( _c_ 0 _/_ 300) 4 .

_•_ When _nt_ [2] _> p/_ 4, we fix _µ_ = ( _t,_ 0 _, . . .,_ 0) [⊺] _∈_ R _[p]_ _,_ introduce


_X_ _l_ = _µl_ [⊺] _,_ _l ∈{−_ 1 _,_ 1 _}_ _[n]_ _,_


and denote _P_ _l_ as the probability of _Y_ if _Y_ = _X_ _l_ + _Z_ with _Z_ _[iid]_ _∼_ _N_ (0 _,_ 1).
Based on the calculation in Theorem 4, the KL-divergence between _P_ _l_ and
_P_ _l_ _′_ satisfies

_D_ ( _P_ _l_ _||P_ _l_ _′_ ) = _[t]_ [2] 2 _[≤]_ [2] _[t]_ [2] _[n.]_

2 _[∥][l][ −]_ _[l]_ _[′]_ _[∥]_ [2]


Applying the generalized Fano’s lemma on _A_, we have



_._
�



inf ˆ _l_ max _l∈A_ [E] _[M]_ [(ˆ] _[l, l]_ [)] _[ ≥]_ 6 [1]



1 _−_ [2] _[t]_ [2] _[n]_ [ + lo][g][ 2]
� _c_ 0 _n_



1
When _nt_ [2] _> p/_ 4, _t_ = _c_ _gap_ ( _p/n_ ) 4, we know


1
1 1 1 4 ) [2]
2 [(] _[p/n]_ [)] 2 _≤_ _t_ = _c_ _gap_ ( _p/n_ ) 4 _,_ thus _t ≤_ [(] _[c]_ _[g][a]_ 1 _[p]_ [(] _[p/][n]_ [)] 12 _≤_ 2 _c_ [2] _gap_ _[.]_

2 [(] _[p/n]_ [)]


1
Provided that _n ≥_ 36 _/_ ( _c_ 0 log 2) _, c_ _gap_ _≤_ ( _c_ 0 _/_ 300) 4, we have



_≥_ [1]
� 6



�



1 _−_ [16] _[c]_ _g_ [4] _ap_ _−_ [lo][g][ 2]
_c_ 0 _c_ 0 _n_



1

6



1 _−_ [2] _[t]_ [2] _[n]_ [ + lo][g][ 2]
� _c_ 0 _n_



_c_ 0 _n_



�



_≥_ [1]

8 _[,]_



which implies



E _M_ ( [ˆ] _l, l_ ) _≥_ [1]

8 _[.]_



E _M_ ( [ˆ] _l, l_ ) _≥_ [1]



inf ˆ _l_ _∥µ∥_ 2 _≤_ max _c_ _gap_ ( _p/n_ ) 1 ~~4~~
_l∈{−_ 1 _,_ 1 _}_ _[n]_



To sum up, we have finished the proof of this theorem.


_Proofs in Canonical Correlation Analysis._


Proof of Theorem 7. The proof of Theorem 7 is relatively complicated, which we shall divide into steps.


1. _(Analysis of Loss)_ Suppose the SVD of _S_ [ˆ] and _S_ are



(1.35)



_S_ ˆ = ˆ _U_ _S_ ˆΣ _S_ ˆ _V_ _S_ [⊺] _[,]_ _U_ ˆ _S_ _∈_ O _p_ 1 _,_ ˆΣ _S_ _∈_ R _[p]_ [1] _[×][p]_ [2] _,_ ˆ _V_ _S_ _∈_ O _p_ 2
_S_ = _U_ _S_ Σ _S_ _V_ _S_ [⊺] _[,]_ _U_ _S_ _∈_ O _p_ 1 _,r_ _,_ Σ _S_ _∈_ R _[r][×][r]_ _, V_ _S_ _∈_ O _p_ 2 _,r_
(we shall note _rank_ ( _S_ ) = _r_ .)


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 17



_−_ [1] _−_ [1]

2 2
_X_ _[U]_ _[S]_ [, ˆ] _[A]_ [ = ˆΣ] _X_



_−_ [1]
respectively. Recall that _A_ = Σ 2



respectively. Recall that _A_ = Σ _−X_ 2 _[U]_ _[S]_ [, ˆ] _[A]_ [ = ˆΣ] _−X_ 2 _[U]_ [ˆ] _[S,]_ [[:] _[,]_ [1:] _[r]_ []] [. Invertible]

multiplication to all _X_ ’s or _Y_ ’s does not change the loss of the procedure, thus without loss of generality we could assume that Σ _X_ = _I_ _p_ 1 .
Under this assumption, we have following expansions for the loss _L_ _F_
and _L_ .
sp



_L_ _F_ ( _A, A_ [ˆ] )


= min 2
_O∈_ O _r_ [E] _[X]_ _[∗]_ _[∥]_ [( ˆ] _[AO][ −]_ _[A]_ [)] [⊺] _[X]_ _[∗]_ _[∥]_ [2]


= min ( _AO_ [ˆ] _−_ _A_ ) [⊺] _X_ _[∗]_ ( _X_ _[∗]_ ) [⊺] ( _AO_ [ˆ] _−_ _A_ )
_O∈_ O _r_ [E] _[X]_ _[∗]_ [tr] � �



_−_ [1]
= min (Σ [ˆ] _X_ 2
_O∈_ O _[r]_ [ tr] �



_−_ [1]

2
_X_ _[U]_ [ˆ] _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O][ −]_ _[U]_ _[S]_ [)]
�



_−_ [1] _−_ [1]

2 2
_X_ _[U]_ [ˆ] _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O][ −]_ _[U]_ _[S]_ [)] [⊺] _[I]_ _[p]_ [1] [(ˆΣ] _X_



����



2



_F_



= min
_O∈_ O _r_



_−_ [1]
ˆΣ 2

_X_ _[U]_ [ˆ] _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O][ −]_ _[U]_ _[S]_

����



2



_F_



_≤_ 2 min
_O∈_ O _r_



ˆ 2
_U_ _S,_ [: _,_ 1: _r_ ] _O −_ _U_ _S_
��� _F_
����



2 _−_ [1]

(ˆΣ _X_ 2
_F_ [+]
����



_−_ [1]

2
_X_ _[−]_ _[I]_ [) ˆ] _[U]_ _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O]_
����



�



2 _−_ [1]

(ˆΣ _X_ 2
_F_ [+ 2]
����



2



(by Lemma 1).

_F_



2
_≤_ 4 sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ )
��� ��� _F_



_−_ [1]

2
_X_ _[−]_ _[I]_ [) ˆ] _[U]_ _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O]_
����



Similarly,

_L_ sp ( _A, A_ [ˆ] ) = min max ( _AOv_ [ˆ] ) [⊺] _X_ _[∗]_ _−_ ( _Av_ ) [⊺] _X_ _[∗]_ [�] [2]
_O∈_ O _r_ _v∈_ R _[r]_ _,∥v∥_ 2 =1 [E] _[X]_ _[∗]_ �


= min max
_O∈_ O _r_ _v∈_ R _[r]_ _,∥v∥_ 2 =1 [( ˆ] _[AOv][ −]_ _[Av]_ [)] [⊺] [Σ] _[X]_ [( ˆ] _[AOv][ −]_ _[Av]_ [)]



(1.36)



����



_−_ [1]
ˆΣ 2

_X_ _[U]_ [ˆ] _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O][ −]_ _[U]_ _[S]_

����



= min
_O∈_ O _r_



ˆ 2
_AO −_ _A_ = min
��� ��� _O∈_ O _r_



2 _−_ [1]
_≤_ 4 sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ) + 2 ˆΣ _X_ 2
��� ���
����



2

_._



_−_ [1]

2
_X_ _[−]_ _[I]_
����



We use the bold symbols **X** _∈_ R _[p]_ [1] _[×][n]_ _,_ **Y** _∈_ R _[p]_ [2] _[×][n]_ to denote those
_n_ samples. Since Σ = [ˆ] _n_ [1] **[XX]** [⊺] [, where] **[ X]** _[ ∈]_ [R] _[p]_ [1] _[×][n]_ [ is a i.i.d. Gaussian]

matrix, by random matrix theory (Corollary 5.35 in Vershynin (2012)),
there exists constant _C, c_ such that



_p_ 1 _/n ≤_ _σ_ min (Σ [ˆ] _X_ ) _≤_ _σ_ max (Σ [ˆ] _X_ ) _≤_ 1 + _C_ ~~�~~



(1.37) 1 _−_ _C_ ~~�~~



_p_ 1 _/n._



with probability at least 1 _−_ _C_ exp( _cp_ 1 ). Since


1 property of correlation matrix _≥_ _σ_ _r_ [2] [(] _[S]_ [)] problem assumption _≥_ _C_ gap _p_ 1 _,_

_n_


18 T. T. CAI AND A. ZHANG


we can find large _C_ gap _>_ 0 to ensure _n ≥_ _Cp_ 1 for any large _C >_ 0. In
addition, we could have



�



(1.38) P



_−_ [1]
ˆΣ 2

_X_

�����



_−_ [1]

2
_X_ _[−]_ _[I]_
����



2 _Cp_ 1
_≤_ _Cp_ 1 _/n ≤_
_nσ_ _r_ [2] ( _S_ )



_≥_ 1 _−_ _C_ exp( _−cp_ 1 ) _._



Moreover, as _U_ [ˆ] _S,_ [: _,_ 1: _r_ ] _O ∈_ O _p_ 1 _,r_, we have



2



_,_

_F_



_−_ [1]
_r_ ˆΣ _X_ 2
����



_−_ [1]

2
_X_ _[−]_ _[I]_
����



2
_−_ [1]
_≥_ (ˆΣ _X_ 2
����



_−_ [1]

2
_X_ _[−]_ _[I]_ [) ˆ] _[U]_ _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O]_
����



thus
(1.39)



2



�



P



_−_ [1]
(ˆΣ _X_ 2
�����



_−_ [1]

2
_X_ _[−]_ _[I]_ [) ˆ] _[U]_ _[S,]_ [[:] _[,]_ [1:] _[r]_ []] _[O]_
����



_Cp_ 1 _r_
_≤_ _Cp_ 1 _r/n ≤_
_F_ _nσ_ _r_ [2] ( _S_ )



_≥_ 1 _−C_ exp( _−cp_ 1 ) _._



Now the central goal in our analysis moves to bound

2 2
sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ) sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ) _,_
��� ��� _F_ [and] ��� ���


namely to compare the singular spectrum between the population



_−_ [1]
_S_ = Σ 2



_−_ [1] _−_ [1]

2 2
_X_ [Σ] _[XY]_ [ Σ] _Y_



_−_ [1] _−_ [1]

2 2
_Y_ and the sample version _S_ [ˆ] = Σ [ˆ] _X_



_−_ [1] _−_ [1]

2 2
_X_ [ˆΣ] _[XY]_ [ ˆΣ] _Y_



2 in sin Θ
_Y_



2
distance. Since sin Θ( _U_ [ˆ] _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ) is _r × r_, sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ )
��� ��� _F_ _[≤]_

2
sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ) . We only need to consider the sin Θ spectral
��� ���
distance below.
2. _(Reformulation of the Model Set-up)_ In this step, we transform _X, Y_ to
a better formulation to simplify the further analysis. First, any invertible affine transformation on **X** and **Y** separately will not essentially
change the problem. We specifically do transformation as follows



_−_ [1]
**X** _→_ [ _U_ _S_ _U_ _S⊥_ ] [⊺] Σ _X_ 2




[1] 2 [ _V_ _S_ _V_ _S⊥_ ] [⊺] Σ _−Y_ [1] 2



_−X_ 2 **[X]** _[,]_ **Y** _→_ ( _I_ _p_ 2 _−_ Σ [2] _S_ [)] _[−]_ [1] 2



2
_Y_ **[Y]** _[.]_



Simple calculation shows that after transformation, Var( _X_ ), Var( _Y_ ) _,_ Cov( _X, Y_ )
become _I_ _p_ 1, ( _I_ _p_ 2 _−_ Σ [2] _S_ [)] _[−]_ [1] [, Σ] _[S]_ [(] _[I]_ _[p]_ 2 _[−]_ [Σ] [2] _S_ [)] _[−]_ 2 [1] respectively. Therefore, with
out loss of generality we can assume that


(1.40) Σ _X_ = _I_ _p_ 1 _,_ Σ _Y_ = ( _I_ _p_ 2 _−_ Σ [2] _S_ [)] _[−]_ [1] _[.]_


(1.41)

_σ_ _i_ ( _S_ ) _,_ _i_ = _j_ = 1 _, · · ·, r_
_S ∈_ R _[p]_ [1] _[×][p]_ [2] is diagonal, such that _S_ _ij_ = � 0 _,_ otherwise


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 19


throughout the proof. (It will be explained a while later why we want
to transform Σ _Y_ into this form.)
Since _X, Y_ are jointly Gaussian, we can relate them as _Y_ = _W_ [⊺] _X_ + _Z_,
where _W ∈_ R _[p]_ [1] _[×][p]_ [2] is a fixed matrix, _X ∈_ R _[p]_ [1] and _Z ∈_ R _[p]_ [2] are
independent random vectors. Based on simple calculation, Σ _XY_ =
E _XY_ [⊺] = Σ _X_ _W_, Σ _Y_ = _W_ [⊺] Σ _X_ _W_ + Var( _Z_ ). Combining (1.40), we
can calculate that



1

_W_ = Σ _[−]_ _X_ [1] [Σ] _[XY]_ [ =] _[ S]_ [Σ] _Y_ 2 _[,]_ _W_ is diagonal _,_



(1.42)



_W_ _ij_ = _S_ _ii_ (1 _−_ _S_ _ii_ [2] [)] _[−]_ 2 [1] _,_ _i_ = _j_
� 0 otherwise



1

2
_Y_ [=] _[ I]_ _[p]_ [2] _[.]_



(1.43) Var( _Z_ ) = Σ _Y_ _−_ Σ



1

2
_Y_ _[S]_ [⊺] _[S]_ [Σ]



1 1

2 2
_Y_ [= Σ] _Y_ [(] _[I]_ _[p]_ [2] _[−]_ _[S]_ [⊺] _[S]_ [)Σ]



1

2
_Y_ [= Σ]



In other words, _Z_ is i.i.d. standard normal and _W_ is diagonal. By
rescaling, the analysis could be much more simplified and easier to
read.
3. _(Expression for_ _S_ [ˆ] _)_ In this step, we move from the population to the
samples and find out a useful expression for _S_ [ˆ] . We use bold symbols
**X** _∈_ R _[p]_ [1] _[×][n]_, **Y** _∈_ R _[p]_ [2] _[×][n]_, **Z** _∈_ R _[p]_ [2] _[×][n]_ to denote the compiled data such
that **Y** = _W_ [⊺] **X** + **Z** . Denote the singular decomposition for **X** and **Y**

are
**X** = _U_ [ˆ] _X_ Σ [ˆ] _X_ _V_ [ˆ] _X_ [⊺] _[,]_ **Y** = _U_ [ˆ] _Y_ Σ [ˆ] _Y_ _V_ [ˆ] _Y_ [⊺] _[,]_


here _U_ [ˆ] _X_ _∈_ O _p_ 1 _,_ Σ [ˆ] _X_ _∈_ R _[p]_ [1] _[×][p]_ [1] _,_ _V_ [ˆ] _X_ _∈_ O _n,p_ 1, _U_ [ˆ] _Y_ _∈_ O _p_ 2 _,_ Σ [ˆ] _Y_ _∈_ R _[p]_ [2] _[×][p]_ [2] _,_ _V_ [ˆ] _Y_ _∈_
O _n,p_ 2 . Thus,


ˆΣ _X_ = ˆ _U_ _X_ ˆΣ [2] _X_ _[U]_ [ˆ] _X_ [⊺] _[,]_ ˆΣ _Y_ = ˆ _U_ _Y_ ˆΣ [2] _Y_ _[U]_ [ˆ] _Y_ [⊺] _[,]_ ˆΣ _XY_ = ˆ _U_ _X_ ˆΣ _X_ ˆ _V_ _X_ [⊺] _[V]_ [ˆ] _[Y]_ [ ˆΣ] _[Y]_ [ ˆ] _[U]_ _Y_ [⊺] _[.]_



Additionally,
_S_ ˆ = ˆΣ _−_ 2 [1]



_−_ 2

_Y_ = _U_ [ˆ] _X_ _V_ [ˆ] _X_ [⊺] _[V]_ [ˆ] _[Y]_ [ ˆ] _[U]_ _Y_ [⊺] _[.]_



_−_ [1] _−_ [1]

2 2
_X_ [ˆΣ] _[XY]_ [ ˆΣ] _Y_



4. _(Useful Characterization of the left Singular Space of_ _S_ [ˆ] _)_ Since **X** is a
i.i.d. Gaussian matrix at this moment, from the random matrix theory,
we know _U_ [ˆ] _X_, _V_ [ˆ] _X_ are randomly distributed with Haar measure on O _p_ 1
and O _n,p_ 1 respectively and _U_ [ˆ] _X_ _,_ Σ [ˆ] _X_ _,_ _V_ [ˆ] _X_ _, Z_ are independent. We can
extend _V_ [ˆ] _X_ _U_ [ˆ] _X_ [⊺] _[∈]_ [O] _[n,p]_ [1] [to orthogonal matrix ˜] _[R]_ _[X]_ [ = [ ˆ] _[V]_ _[X]_ [ ˆ] _[U]_ _X_ [⊺] _[,]_ [ ˆ] _[R]_ _[X][⊥]_ []] _[ ∈]_ _[O]_ _[n]_
such that _R_ [ˆ] _X⊥_ _∈_ O _n,n−p_ 1 . Introduce **Z** [˜] = **Z** _·_ _R_ [˜] _X_, **Y** [˜] = **Y** _·_ _R_ [˜] _X_,


20 T. T. CAI AND A. ZHANG


**X** ˜ = **X** ˜ _R_ _X_ . ˜ **Y** [⊺] can be explicitly written as the following clean form


(1.44)



ˆ
**Y** ˜ [⊺] = ˜ **X** [⊺] _W_ + ˜ **Z** [⊺] = _U_ ˆ _X_ ˆ _V_ ⊺ _X_ **[X]** [⊺] _[W]_
� _R_ _X_ [⊺] _⊥_ **[X]** [⊺] _[W]_


:= **G** + **Z** [˜] [⊺] _,_



+ _Z_ [˜] [⊺] =
�



_p_ 2
_p_ 1 _U_ ˆ _X_ ˆΣ _X_ ˆ _U_ _X_ [⊺] _[W]_ + **Z** [˜] [⊺]
_n −_ _p_ 1 � 0 �



˜ ˜
**Z** _[iid]_ _∼_ _N_ (0 _,_ 1) _,_ **Z** _,_ ˆ _U_ _X_ _,_ ˆΣ _X_ are independent; **G** _,_ ˜ **Z** are independent.


Meanwhile, since _V_ [ˆ] _Y_ is the left singular vectors for **Y** [⊺], we have




[ _V_ [ˆ] _X_ _U_ [ˆ] _X_ [⊺] _[,]_ [ ˆ] _[R]_ _[X][⊥]_ []] [⊺] _[V]_ [ˆ] _[Y]_ [ =] _U_ ˆˆ _X_ ˆ _V_ ⊺ _X_ _[V]_ [ˆ] _[Y]_
� _R_ _X_ [⊺] _⊥_ _[V]_ [ˆ] _[Y]_



�



is the left singular vectors of **Y** [˜] [⊺] = [ _V_ [ˆ] _X_ _U_ [ˆ] _X_ [⊺] _[,]_ [ ˆ] _[R]_ _[X][⊥]_ []] [⊺] **[Y]** [⊺] [, which yields the]
following important characterization for _S_ [ˆ] .
(1.45) _S_ ˆ ˆ _U_ _Y_ = ˆ _U_ _X_ ˆ _V_ _X_ [⊺] _[V]_ [ˆ] _[Y]_ [ is the first] _[ p]_ [1] [ rows of the left singular vectors of ˜] **[Y]** [⊺] [.]


Suppose the SVD for **Y** [˜] is


**Y** ˜ [⊺] = ˜ _U_ _Y_ ˜Σ _Y_ ˜ _V_ _Y_ [⊺] _[,]_ _U_ ˜ _Y_ _∈_ O _n,p_ 2 _,_ ˜Σ _Y_ _∈_ R _[p]_ [2] _[×][p]_ [2] _,_ ˜ _V_ _Y_ _∈_ O _p_ 2 _._


Further assume the SVD for the first _p_ 1 rows of _U_ [˜] _Y_ is

_U_ ˜ _Y_ _Y_ 2 _[,]_ _U_ ˜ _Y_ 2 _∈_ O _p_ 1 _,_ ˜Σ _Y_ 2 _∈_ R _[p]_ [1] _[×][p]_ [2] _,_ ˜ _V_ 2 _∈_ O _p_ 2 _._
� � [1: _p_ 1 _,_ :] [= ˜] _[U]_ _[Y]_ [ 2] [ ˜Σ] _[Y]_ [ 2] [ ˜] _[V]_ [ ⊺]

By characterization ( _U_ ˆ _Y_ _∈_ O _p_ 2 does not change left singular vectors, we have1.45) and the fact that right multiplication of




[1: _p_ 1 _,_ ] _[.]_



(1.46)



_U_ ˆ _S_ = ˜ _U_ _Y_ 2 _,_ where
_U_ ˆ _S_ is the left singular vectors of ˆ _S_,


_U_ ˜ _Y_ 2 is the left singular space of _U_ ˜ _Y_
� �



The characterization above is the baseline we shall use later to compare
the spectrum of _S_ [ˆ] and _S_ .
5. _(Split of_ sin Θ _Norm Distance)_ Recall Step 1, the central goal of analysis now is to find the sin Θ distance between the leading _r_ left singular


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 21


vectors of _S_ [ˆ] and _S_ . It is easy to see that the left singular space of _S_ is







_∈_ O _p_ 1 _,r_




(1.47) _U_ _S_ =









1

...

1

0 _· · ·_ 0



By the traingle inequality of sin Θ distance (Lemma 1),
(1.48)
���sin Θ � _U_ ˆ _S,_ [: _,_ 1: _r_ ] _, U_ _S_ ���� _≤∥_ sin Θ ( _U_ _G_ _, U_ _S_ ) _∥_ + ���sin Θ � _U_ _G_ _,_ _U_ [ˆ] _S,_ [: _,_ 1: _r_ ] ���� _._


where _U_ _G_ is defined a little while later in (1.49). For the next Steps 6
and 7, we try to bound the two sin Θ distances respectively.
6. _(Left Singular Space of_ **G** [1: _p_ 1 _,_ 1: _r_ ] _)_ Recall the definition of **G** in (1.44),
since _W_ is with only first _r_ diagonal entries non-zero, only the submatrix **G** [1: _p_ 1 _,_ 1: _r_ ] of **G** is non-zero. Suppose


(1.49) **G** [1: _p_ 1 _,_ 1: _r_ ] = _U_ _G_ Σ _G_ _V_ _G_ [⊺] _[,]_ _U_ _G_ _∈_ O _p_ 1 _,r_ _,_ Σ _G_ _∈_ R _[r][×][r]_ _, V_ _G_ _∈_ O _r_ _._


In this section we aim to show the following bound on the sin Θ distance from _U_ _G_ to _U_ _S_, i.e. there exists _C_ 0 _>_ 0 such that whenever
_n > Cp_ 1,


(1.50) P � _∥_ sin Θ( _U_ _G_ _, U_ _S_ ) _∥≥_ _C_ ~~�~~ _p_ 1 _/n_ � _≤_ _C_ exp( _−cp_ 1 )


and also



(1.51) P


Recall



�



_cnσ_ _r_ [2] [(] _[S]_ [)]
_σ_ min [2] [(] **[G]** [1: _p_ 1 _,_ 1: _r_ ] [)] _[ ≥]_
~~�~~ 1 _−_ _σ_ _r_ [2] (



1 _−_ _σ_ _r_ [2] ( _S_ )



�



_≥_ 1 _−_ _C_ exp( _−cp_ 1 ) _._






_._




(1.52) _G_ [1: _p_ 1 _,_ 1: _r_ ] = _U_ [ˆ] _X_ Σ [ˆ] _X_ _U_ [ˆ] _X_ [⊺] _[W]_ [[:] _[,]_ [1:] _[r]_ []] _[,]_ _W_ [: _,_ 1: _r_ ] =


Thus if we split _U_ [ˆ] _X_ as



_W_ 11



...

_W_ _rr_

0





_r_

ˆ
_U_ ˆ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ [⊺] 1 _r_ _._
� _U_ _X_ 2 ˆΣ _X_ ˆ _U_ _X_ [⊺] 1 � _p_ 1 _−_ _r_



_U_ ˆ _X_ =



_p_ 1

ˆ
_U_ ˆ _X_ 1 _r_ _,_ then _U_ ˆ _X_ ˆΣ _X_ _U_ ˆ _X_ [⊺]
� _U_ _X_ 2 � _p_ 1 _−_ _r_ � � [: _,_ 1: _r_ ] [=]


22 T. T. CAI AND A. ZHANG


Furthermore, due to the diagonal structure of _W_, **G** [1: _p_ 1 _,_ 1: _r_ ] has the



same left singular space as _U_ [ˆ] _X_ Σ [ˆ] _X_ _U_ ˆ _X_ [⊺]
� �




[: _,_ 1: _r_ ] [. Since ˆΣ] _[X]_ [ is the singular]



values of i.i.d. Gaussian matrix _X ∈_ R _[p]_ [1] _[×][n]_, by random matrix theory
(Vershynin, 2012),
(1.53)
P _√n_ + _C√_ ~~_p_~~ 1 _≥∥_ ˆΣ _X_ _∥≥_ _σ_ min ˆΣ _X_ _≥_ _[√]_ _n −_ _C_ _[√]_ ~~_p_~~ 1 _≥_ 1 _−C_ exp( _−cp_ 1 ) _._
� � � �


Under the event that _[√]_ ~~_n_~~ + _C_ _[√]_ ~~_p_~~ 1 _≥∥_ Σ [ˆ] _X_ _∥≥_ _σ_ min ˆΣ _X_ _≥_ _[√]_ ~~_n_~~ _−C_ _[√]_ ~~_p_~~ 1
� �
hold and _n ≥_ _C_ gap _p_ 1 for some large _C_ gap, we have


ˆ
_σ_ min _U_ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ [⊺] 1 _≥_ _σ_ min (Σ [ˆ] _X_ ) _≥_ _[√]_ _n −_ _C_ _[√]_ ~~_p_~~ 1 ~~_,_~~ (since _U_ [ˆ] _X_ 1 is orthogonal.)
� �

_U_ ˆ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ ⊺ 2 = _U_ ˆ _X_ 1 ˆΣ _X_ _−_ _[√]_ _nI_ _U_ ˆ _X_ [⊺] 2 _≤_ _C√_ ~~_p_~~ 1 ~~_._~~
��� ��� ��� � � ���



By Lemma 9, the left singular space of _U_ [ˆ] _X_ Σ [ˆ] _X_ _U_ ˆ _X_ [⊺]
� �

the left singular vectors of _G_ [1: _p_ 1 _,_ 1: _r_ ] satisfies




[: _,_ 1: _r_ ] [, which is also]



_σ_ max [2] � _U_ _G,_ [( _r_ +1): _n,_ 1: _r_ ] � _≤_



� _X_ 1 _X_ _X_ 2 �

ˆ ~~�~~ ˆ ⊺ ~~�~~ 2 _[≤]_ _[C]_ _n_ _[p]_ [1]
_σ_ min [2] _U_ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ [⊺] 1 + _U_ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ 2
~~�~~ ~~�~~ �� ��



��� _U_ ˆ _X_ 1 ˆΣ _X_ ˆ _U_ _X_ ⊺ 2 ��� 2



_n_



when _n ≥_ _C_ gap _p_ 1 for some large _C_ gap _>_ 0. By the characterization
of sin Θ distance in Lemma 1, we have finally proved the statement
(1.50).
Since


ˆ
_σ_ min ( **G** [1: _p_ 1 _,_ 1: _r_ ] ) = _σ_ _r_ � _U_ _X_ ˆΣ _X_ ˆ _U_ _X_ [⊺] _[W]_ [[:] _[,]_ [1:] _[r]_ []] �

(1.54) _≥σ_ _r_ � _U_ ˆ _X_ ˆΣ _X_ ˆ _U_ _X_ [⊺] 1 � _· σ_ min � _W_ [: _,_ 1: _r_ ] � _≥_ _σ_ min (Σ [ˆ] _X_ ) _·_ ~~�~~ 1 _σ −_ _r_ ( _σS_ _r_ [2] )( _S_ ) _,_


by (1.53), (1.51) holds.
7. In this step we try to prove the following statement: there exists constant _C_ gap, such that whenever


1
( _p_ 1 _p_ 2 ) 2 + _p_ 1 + _p_ [3] 2 _[/]_ [2] _n_ _[−]_ 2 [1]
(1.55) _σ_ _r_ [2] [(] _[S]_ [)] _[ ≥]_ _[C]_ [gap] _,_

_n_


we have
(1.56)



�



P



2 � _nσ_ _r_ [2] [(] _[S]_ [)][ +] _[ p]_ [2] �
sin Θ( ˆ _U_ _S,_ [1: _r,_ :] _, U_ _G_ ) _≤_ _[Cp]_ [1]
��� _n_ [2] _σ_ _r_ [4] ( _S_ )
����



_≥_ 1 _−C_ exp( _−cp_ 1 _∧p_ 2 ) _._


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 23


We shall again note that by 1 _≥_ _σ_ _r_ [2] [(] _[S]_ [), the condition (][1.55][) also implies]
_n ≥_ _C_ gap _p_ 1 . Recall (1.51), with probability at least 1 _−_ _C_ exp( _−cp_ 1 ),
(1.57)


1
_r_ [(] _[S]_ [)]
_σ_ _r_ [2] [(] **[G]** [)] _[ ≥]_ _[cnσ]_ [2] _r_ [(] _[S]_ [)] _[ ≥]_ _[cC]_ [gap] ( _p_ 1 _p_ 2 ) 2 + _p_ 1 + _p_ [3] 2 _[/]_ [2] _n_ _[−]_ 2 [1] _._
1 _−_ _σ_ _r_ [2] ( _S_ ) _[≥]_ _[cnσ]_ [2] � �


Conditioning on _G_ with _σ_ _r_ [2] [(] _[G]_ [) satisfies (][1.57][), set] _[ L]_ [ = ˜] _[Y]_ [ ⊺] [,] _[ n]_ [ =] _[ n]_ [,]
_p_ = _p_ 2, _d_ = _p_ 1, _r_ = _r_, Lemma 8 leads to the following result



**G**
�����



�



P



2 _r_ [(] **[G]** [)][ +] _[p]_ [2] [)(][1 +] _[ σ]_ _r_ [2] [(] **[G]** [)] _[/][n]_ [)]
sin Θ( ˆ _U_ _S,_ [: _,_ 1: _r_ ] _, U_ _G_ ) _≤_ _[C][p]_ [1] [(] _[σ]_ [2]
��� _σ_ _r_ [4] ( **G** )
����



_≥_ 1 _−_ _C_ exp( _−cp_ 1 _∧_ _p_ 2 ) _._


1
whenever _σ_ _r_ [2] [(] **[G]** [)] _[ ≥]_ _[C]_ [gap] ( _p_ 1 _p_ 2 ) 2 + _p_ 1 + _p_ [3] 2 _[/]_ [2] _n_ _[−]_ 2 [1] for uniform con� �

stant _C_ gap large enough. Then we shall also note that


_∥S∥≤_ 1 _, ⇒_ _σ_ _r_ [2] [(] _[S]_ [)] _[ ≤]_ [1]

� _nσ_ _r_ [2] [(] _[S]_ [)][ +] _[ p]_ [2] � � _nσ_ _r_ [2] [(] _[S]_ [)][ +] _[ p]_ [2] � (1 + _nσ_ _r_ [2] [(] _[S]_ [)] _[/][n]_ [)]
_⇒_ _[Cp]_ [1] _≥_ _[Cp]_ [1]

_n_ [2] _σ_ _r_ [4] ( _S_ ) _n_ [2] _σ_ _r_ [4] ( _S_ )

(1.57) _Cp_ 1 � _nσ_ _r_ [2] [(] _[S]_ [)][ +] _[ p]_ [2] � _r_ [(] **[G]** [)][ +] _[p]_ [2] [)(][1 +] _[ σ]_ _r_ [2] [(] **[G]** [)] _[/][n]_ [)]
_⇒_ _≥_ _[C][p]_ [1] [(] _[σ]_ [2] _._

_n_ [2] _σ_ _r_ [4] ( _S_ ) _σ_ _r_ [4] ( **G** )


The two inequalities above together implies the statement (1.56) holds
for true.


Finally, combining (1.48), (1.50), (1.56), (1.38), (1.39) and (1.36), we have
completed the proof of Theorem 7.


The following Lemmas 8, 9 and 10 are used in the proof of Theorem 7. To
be specific, Lemma 8 provides a sin Θ upper bound for the singular vectors of
a sub-matrix; Lemma 10 gives both upper and lower bounds for the singular
values of a sub-matrix; Lemma 10 propose a spectral norm upper bound for
any matrix after a random projection.


Lemma 8. _Suppose L ∈_ R _[n][×][p]_ _(n > p) is a non-central random matrix_

_such that L_ = _G_ + _Z, Z_ _[iid]_ _∼_ _N_ (0 _,_ 1) _, G is all zero except the top left d × r_
_block (d ≥_ _r). Suppose the SVD for G_ [1: _d,_ 1: _r_ ] _, L are_


_G_ [1: _d,_ 1: _r_ ] = _U_ _G_ Σ _G_ _V_ _G_ [⊺] _[,]_ _U_ _G_ _∈_ O _d,r_ _,_ Σ _G_ _∈_ R _[r][×][r]_ _, V_ _G_ _∈_ O _r_ ;


24 T. T. CAI AND A. ZHANG


_L_ = _U_ [ˆ] Σ [ˆ] _V_ [ˆ] [⊺] _,_ _U_ ˆ _∈_ O _n,p_ _,_ ˆΣ _∈_ R _[p][×][p]_ _,_ ˆ _V ∈_ O _p_ _._

_In addition, suppose r < d < n, the SVD for_ _U_ [ˆ] [1: _d,_ :] _is_ _U_ [ˆ] [1: _d,_ :] = _U_ [ˆ] 2 Σ [ˆ] 2 _V_ [ˆ] 2 [⊺] _[.]_
_There exists C_ gap _, C_ 0 _>_ 0 _such that whenever_


1
(1.58) _σ_ _r_ [2] [(] _[G]_ [) =] _[ t]_ [2] _[ > C]_ [gap] [((] _[pd]_ [)] 2 + _d_ + _p_ [3] _[/]_ [2] _n_ _[−]_ 2 [1] ) _,_ _n ≥_ _C_ 0 _p,_


_we have_



~~�~~
(1.59) _∥_ sin Θ( _U_ [ˆ] 2 _,_ [: _,_ 1: _r_ ] _, U_ _G_ ) _∥≤_ _[C]_



_t_ [2]



_d_ ( _t_ [2] + _p_ )(1 + _t_ [2] _/n_ )



_with probability at least_ 1 _−_ _C_ exp( _−cd ∧_ _p_ ) _._


Lemma 9 (Spectral Bound for Partial Singular Vectors). _Suppose L ∈_
R _[n][×][p]_ _with n > p, L_ = _U_ Σ _V_ [⊺] _is the SVD with U ∈_ O _n×p_ _,_ Σ _∈_ R _[p][×][p]_ _, V ∈_ O _p_ _._
_Then for any subset_ Ω _⊆{_ 1 _, · · ·, n},_


_σ_ max [2] [(] _[L]_ [Ω _,_ :] [)]
(1.60) _σ_ max [2] [(] _[U]_ [Ω _,_ :] [)] _[ ≤]_
_σ_ max [2] ( _L_ [Ω _,_ :] ) + _σ_ min [2] [(] _[L]_ [[Ω] _[c]_ _[,]_ [:]] [)] _[,]_


_σ_ [2]
min [(] _[L]_ [[][Ω] _[,]_ [:][]] [)]
(1.61) _σ_ min [2] [(] _[U]_ [Ω _,_ :] [)] _[ ≥]_
_σ_ min [2] [(] _[L]_ [[Ω] _[,]_ [:]] [) +] _[ σ]_ max [2] [(] _[L]_ [[Ω] _[c]_ _[,]_ [:]] [)] _[.]_


Lemma 10 (Spectral Norm Bound of a Random Projection). _Suppose_
_X ∈_ R _[n][×][m]_ _is a fixed matrix_ rank( _X_ ) = _p, suppose R ∈_ O _n×d_ _(n > d) is_
_with orthogonal columns with Haar measure. There exists uniform constant_
_C_ 0 _>_ 0 _such that whenever n ≥_ _C_ 0 _d, the following bound probability hold for_
_uniform constant C, c >_ 0 _,_


(1.62) P _∥R_ [⊺] _X∥_ [2] _≥_ _[p]_ [ +] _[ C][√][p][d]_ [ +] _[ Cd]_ _∥X∥_ _≤_ 1 _−_ _C_ exp( _−cd_ ) _._
� _n_ �


_Proofs of Technical Lemmas._


Proof of Lemma 1.


_•_ (Expressions) Suppose _V_ [⊺] _V_ [ˆ] = _A_ Σ _B_ [⊺] is the SVD, where Σ = diag( _σ_ 1 _, · · ·, σ_ _r_ ),
_A, B ∈_ O _r_ . By definition of sin Θ distances,


(1.63)



1 _−_ _σ_ _r_ [2] = ~~�~~



_∥_ sin Θ( _V, V_ [ˆ] ) _∥_ = sin(cos _[−]_ [1] ( _σ_ _r_ )) = ~~�~~



1 _−_ _σ_ [2]
min [(] _[V]_ [ ⊺] _[V]_ [ˆ][ )] _[,]_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 25



_r_
� sin [2] (cos _[−]_ [1] ( _σ_ _r_ ))


_i_ =1


_r_
� _σ_ _i_ [2] [=] _[ r][ −∥][V]_ [ˆ] [ ⊺] _[V][ ∥]_ _F_ [2] _[.]_


_i_ =1



(1.64)



_∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ [2] [=]


_r_

= �(1 _−_ _σ_ _r_ [2] [) =] _[ r][ −]_


_i_ =1



On the other hand, since _V_ [ˆ], _V_, _V_ _⊥_ are orthonormal,


(1.65)

_∥V_ [ˆ] [⊺] _V x∥_ [2] 2
_σ_ min [2] [( ˆ] _[V]_ [ ⊺] _[V]_ [ ) = min]
_x∈_ R _[r]_ _∥x∥_ [2] 2

_∥V x∥_ [2] 2 _[−]_ _[∥][V]_ [ˆ] [ ⊺] _⊥_ _[V x][∥]_ 2 [2] _∥V_ [ˆ] _⊥_ _V x∥_ [2] 2
= min = 1 _−_ max = 1 _−∥V_ [ˆ] _⊥_ [⊺] _[V][ ∥]_ [2] _[,]_
_x∈_ R _[r]_ _∥x∥_ [2] 2 _x∈_ R _[r]_ _∥x∥_ [2] 2

_∥V_ [ˆ] [⊺] _V ∥_ _F_ [2] [= tr] � _V_ ˆ [⊺] _V V_ [⊺] _V_ ˆ � = tr � _V_ ˆ ˆ _V_ [⊺] _V V_ [⊺] [�]

(1.66)
=tr � _V V_ [⊺] _−_ _V_ [ˆ] _⊥_ _V_ [ˆ] _⊥_ [⊺] _[V V]_ [ ⊺] [�] = _r −∥V_ [ˆ] _⊥_ [⊺] _[V][ ∥]_ _F_ [2] _[,]_

we conclude that _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ = _∥V_ [ˆ] _⊥_ [⊺] _[V][ ∥]_ [,] _[ ∥]_ [sin Θ( ˆ] _[V, V]_ [ )] _[∥]_ _[F]_ [ =] _[ ∥][V]_ [ˆ] [ ⊺] _⊥_ _[V][ ∥]_ _[F]_ [ .]

_•_
(Triangle Inequality) Next we consider the triangle inequality under
spectral norm (8.5). Denote


_x_ = _∥_ sin Θ( _V_ 1 _, V_ 2 ) _∥,_ _y_ = _∥_ sin Θ( _V_ 1 _, V_ 3 ) _∥._


For each _i_ = 1 _,_ 2 _,_ 3, we can expand _V_ _i_ to the full orthogonal matrix as

[ _V_ _i_ _V_ _i⊥_ ] _∈_ O _p_ . Thus,


(1.63) (1.65)
_σ_ min ( _V_ 1 [⊺] _[V]_ [2] [)] = ~~�~~ 1 _−_ _x_ [2] _,_ _σ_ max [2] [(] _[V]_ [ ⊺] 1 _⊥_ _[V]_ [2] [)] = 1 _−_ _σ_ min [2] [(] _[V]_ [ ⊺] 1 _[V]_ [2] [) =] _[ x]_ [2] _[.]_



Similarly
_σ_ min ( _V_ 1 [⊺] _[V]_ [3] [) =] �


Thus,



1 _−_ _y_ [2] _,_ _σ_ 1 [2] [(] _[V]_ [ ⊺] 1 _⊥_ _[V]_ [3] [) =] _[ y]_ [2] _[.]_



_σ_ min ( _V_ 2 [⊺] _[V]_ [3] [) =] _[σ]_ [min] [(] _[V]_ [ ⊺] 2 _[V]_ [1] _[V]_ [ ⊺] 1 _[V]_ [3] [ +] _[ V]_ [ ⊺] 2 _[V]_ [1] _[⊥]_ _[V]_ [ ⊺] 1 _⊥_ _[V]_ [3] [)]

_≥σ_ min ( _V_ 2 [⊺] _[V]_ [1] _[V]_ [ ⊺] 1 _[V]_ [3] [)] _[ −]_ _[σ]_ [max] [(] _[V]_ [ ⊺] 2 _[V]_ [1] _[⊥]_ _[V]_ [ ⊺] 1 _⊥_ _[V]_ [3] [)]

_≥σ_ min ( _V_ 2 [⊺] _[V]_ [1] [)] _[σ]_ [min] [(] _[V]_ [ ⊺] 1 _[V]_ [3] [)] _[ −]_ _[σ]_ [max] [(] _[V]_ [ ⊺] 2 _[V]_ [1] _[⊥]_ [)] _[ ·][ σ]_ [max] [(] _[V]_ [ ⊺] 1 _⊥_ _[V]_ [3] [)]

_≥_ ~~�~~ (1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _−_ _xy._



Therefore,


(1.67) _∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥≤_



~~�~~



1 _−_
~~��~~



2
(1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _−_ _xy_
~~�~~



+ _[.]_



Now, we discuss under two situations.


26 T. T. CAI AND A. ZHANG


1. If ~~�~~ (1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _≥_ _xy_, we have



2
(1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _−_ _xy_
�



1 _−_ � ~~�~~



+
= _x_ [2] + _y_ [2] _−_ _x_ [2] _y_ [2] + 2 _xy_ ~~�~~ (1 _−_ _x_ [2]



(1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _−_ _x_ [2] _y_ [2] _≤_ ( _x_ + _y_ ) [2] _._



Thus, _∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥≤_ _x_ + _y_ .

2. If ~~�~~ (1 _−_ _x_ [2] )(1 _−_ _y_ [2] ) _> xy_, we have _x_ [2] + _y_ [2] _>_ 1. Provided that

0 _≤_ _x, y ≤_ 1, this implies _x_ + _y >_ 1. Thus, _∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥≤_ 1 _≤_

_x_ + _y_ .


To sum up, we always have (8.5). The proof of the triangle inequality
under Frobenius norm is slightly simpler.


(1.68)

_∥_ sin Θ( _V_ 2 _, V_ 3 ) _∥_ _F_


(1.64) _,_ (1.66)
= _∥V_ 2 [⊺] _⊥_ _[V]_ [3] _[∥]_ _[F]_ [ =] _[ ∥][V]_ [ ⊺] 2 _⊥_ [(][P] _[V]_ [1] [+][ P] _[V]_ 1 _⊥_ [)] _[V]_ [3] _[∥]_ _[F]_
_≤∥V_ 2 [⊺] _⊥_ _[V]_ [1] _[V]_ [ ⊺] 1 _[V]_ [3] _[∥]_ _[F]_ [ +] _[ ∥][V]_ [ ⊺] 2 _⊥_ _[V]_ [1] _[⊥]_ _[V]_ [ ⊺] 1 _⊥_ _[V]_ [3] _[∥]_ _[F]_
_≤∥V_ 2 [⊺] _⊥_ _[V]_ [1] _[∥]_ _[F]_ _[ · ∥][V]_ [ ⊺] 1 _[V]_ [3] _[∥]_ [+] _[ ∥][V]_ [ ⊺] 2 _⊥_ _[V]_ [1] _[⊥]_ _[∥· ∥][V]_ [ ⊺] 1 _⊥_ _[V]_ [3] _[∥]_ _[F]_
_≤∥V_ 2 [⊺] _⊥_ _[V]_ [1] _[∥]_ _[F]_ [ +] _[ ∥][V]_ [ ⊺] 1 _⊥_ _[V]_ [3] _[∥]_ _[F]_ _[ ≤∥]_ [sin Θ(] _[V]_ [1] _[, V]_ [3] [)] _[∥]_ _[F]_ [ +] _[ ∥]_ [sin Θ(] _[V]_ [1] _[, V]_ [2] [)] _[∥]_ _[F]_ _[ .]_


_•_
(Equivalence with Other Metrics) Since all metrics mentioned in Lemma

1 is rotation invariant, i.e. for any _J ∈_ O _p_, sin Θ( _V, V_ [ˆ] ) = sin Θ( _JV, JV_ [ˆ] ),
so does the other metrics. Without loss of generality, we can assume
that



_I_ _r_
_V_ =
�0 ( _p−r_ ) _×r_ � _p×r_



In this case,


_D_ sp ( _V, V_ [ˆ] ) = inf _O∈_ O _r_ _[∥][V]_ [ˆ] _[ −]_ _[V O][∥≥∥][V]_ [ˆ] [[(] _[r]_ [+1):] _[p,]_ [:]] _[∥]_ [=] _[ ∥][V]_ [ ⊺] _⊥_ _[V]_ [ˆ] _[ ∥]_ [=] _[ ∥]_ [sin Θ( ˆ] _[V, V]_ [ )] _[∥][.]_


Recall _V_ [⊺] _V_ [ˆ] = _A_ Σ _B_ [⊺] is the singular decomposition.


_D_ sp ( _V, V_ [ˆ] ) = inf
_O∈_ O _r_ _[∥][V]_ [ˆ] _[ −]_ _[V O][∥≤∥][V]_ [ˆ] _[ −]_ _[V AB]_ [⊺] _[∥]_


~~�~~ ˆ ~~�~~ 2 ~~�~~ ˆ ~~�~~ 2

_≤_ � _V_ ⊺ _V −_ _V AB_ ⊺ � + � _V_ ⊺ _⊥_ _V −_ _V AB_ ⊺ �

~~�~~ � ~~�~~ ~~�~~ � � ~~�~~ ~~�~~ �



= ~~�~~ _∥A_ (Σ _−_ _I_ _r_ ) _B_ [⊺] _∥_ [2] + _∥V_ [ˆ] _⊥_ [⊺] _[V]_ [ˆ] _[ ∥]_ [2] _[ ≤]_ �


(1.63)

_≤_ ~~�~~ 1 _−_ _σ_ _r_ [2] + _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ [2] _≤_ _√_



(1 _−_ _σ_ _r_ ) [2] + _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ [2]



2 _∥_ sin Θ( _V, V_ [ˆ] ) _∥._


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 27


Similarly, we can show _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ _≤_ _D_ _F_ ( _V, V_ [ˆ] ) _≤_ _√_ 2 _∥_ sin Θ( _V, V_ [ˆ] ) _∥_ _F_ .

For _∥V_ [ˆ] _V_ [ˆ] [⊺] _−_ _V V_ [⊺] _∥_, one can show

ˆ
��� _V_ ˆ _V_ ⊺ _−_ _V V_ ⊺ ��� _≥_ ��� _V_ ⊺ _⊥_ _[V]_ [ˆ][ ˆ] _[V]_ [ ⊺] _[−]_ _[V]_ [ ⊺] _⊥_ _[V V]_ [ ⊺] [���]

= _∥_ ( _V_ _⊥_ [⊺] _[V]_ [ˆ][ ) ˆ] _[V]_ [ ⊺] _[∥]_ [=] _[ ∥][V]_ [ ⊺] _⊥_ _[V]_ [ˆ] _[ ∥]_ [=] _[ ∥]_ [sin Θ( ˆ] _[V, V]_ [ )] _[∥]_ since _V_ is orthonormal _._


Besides,


ˆ
_V_ ˆ _V_ [⊺] _−_ _V V_ [⊺] = (P _V_ + P _V_ _⊥_ ) ˆ _V_ ˆ _V_ [⊺] (P _V_ + P _V_ _⊥_ ) _−_ _V V_ [⊺]


= _V_ ( _V_ [⊺] _V_ [ˆ] )( _V_ [⊺] _V_ [ˆ] ) [⊺] _−_ _I_ _r_ _V_ [⊺] + _V_ _⊥_ _V_ _⊥_ [⊺] _[V]_ [ˆ][ ˆ] _[V]_ [ ⊺] _[V V]_ [ ⊺]
� �

+ _V V_ [⊺] _V_ [ˆ] _V_ [ˆ] [⊺] _V_ _⊥_ _V_ _⊥_ [⊺] [+] _[ V]_ _[⊥]_ _[V]_ [ ⊺] _⊥_ _[V]_ [ˆ][ ˆ] _[V]_ [ ⊺] _[V]_ _[⊥]_ _[V]_ [ ⊺] _⊥_ _[.]_


For any vector _x ∈_ R _[p]_, we denote _x_ 1 = _V_ [⊺] _x, x_ 2 = _V_ _⊥_ [⊺] _[x]_ [. We also denote]
_t_ = _∥V_ _⊥_ [⊺] _[V]_ [ˆ] _[ ∥]_ [. Recall that we have proved in part 1 that] _[ σ]_ min [2] [(] _[V]_ [ ⊺] _[V]_ [ˆ][ ) =]
1 _−_ _σ_ max ( _V_ _⊥_ _V_ [ˆ] ), then 1 _≥_ _σ_ max ( _V_ [⊺] _V_ [ˆ] ) _≥_ _σ_ min ( _V_ [⊺] _V_ [ˆ] ) _≥_ _√_ 1 _−_ _t_ [2] . Thus,


_x_ ⊺ _V_ ˆ ˆ _V_ [⊺] _−_ _V V_ [⊺] [�] _x_
��� � ���


= _x_ ⊺1 ( _V_ [⊺] _V_ [ˆ] )( _V_ [⊺] _V_ [ˆ] ) [⊺] _−_ _I_ _r_ _x_ 1 + _x_ [⊺] 2 _V_ _⊥_ [⊺] _[V]_ [ˆ] _V_ ˆ [⊺] _V x_ 1

� � � �
�����



+ _x_ [⊺] 1 _V_ [⊺] _V_ [ˆ] _V_ ˆ [⊺] _V_ _⊥_ _x_ 2 + _x_ [⊺] 2 _V_ _⊥_ [⊺] _[V]_ [ˆ] _V_ ˆ [⊺] _V_ _⊥_ _x_ 2
� � � �� �


_≤t_ [2] _∥x_ 1 _∥_ 2 [2] [+ 2] _[t][∥][x]_ [2] _[∥]_ [2] _[∥][x]_ [1] _[∥]_ [2] [+] _[ t]_ [2] _[∥][x]_ [2] _[∥]_ [2] 2
_≤_ � _t_ [2] + _t_ �� _∥x_ 1 _∥_ 2 [2] [+] _[ ∥][x]_ [2] _[∥]_ [2] 2 � _≤_ 2 _t∥x∥_ 2 [2] _[,]_


which implies

ˆ
_V_ ˆ _V_ ⊺ _−_ _V V_ ⊺ _≤_ 2 _t_ = 2 sin Θ( ˆ _V, V_ ) _._
��� ��� ��� ���



�����



This has established the equivalence between _∥_ sin _θ_ ( _V, V_ [ˆ] ) _∥_ and _∥V_ [ˆ] _V_ [ˆ] [⊺] _−_
_V V_ [⊺] _∥_ . Finally for _∥V_ [ˆ] _V_ [ˆ] [⊺] _−_ _V V_ [⊺] _∥_ _F_, one has


_∥V_ [ˆ] _V_ [ˆ] [⊺] _−_ _V V_ [⊺] _∥_ _F_ [2] [=tr] �( _V_ [ˆ] _V_ [ˆ] [⊺] _−_ _V V_ [⊺] ) [2] [�]


ˆ ˆ ˆ ˆ
=tr _V_ ˆ _V_ [⊺] _V_ ˆ _V_ [⊺] + _V V_ [⊺] _V V_ [⊺] _−_ _V_ ˆ _V_ [⊺] _V V_ [⊺] _−_ _V V_ [⊺] _V_ ˆ _V_ [⊺] [�]
�

= _r_ + _r −_ 2 _∥V_ [ˆ] [⊺] _V ∥_ _F_ [2] [= 2] _[∥]_ [sin Θ( ˆ] _[V, V]_ [ )] _[∥]_ [2] _F_ _[.]_


28 T. T. CAI AND A. ZHANG


Proof of Lemma 2. Before starting the proof, we introduce some useful notations. For any matrix _M ∈_ R _[p]_ [1] _[×][p]_ [2] with the SVD _M_ = [�] _[p]_ _i_ =1 [1] _[∧][p]_ [2] _u_ _i_ _σ_ _i_ ( _M_ ) _v_ _i_ [⊺]
we use _M_ max( _r_ ) to denote its leading _r_ principle components, i.e. _M_ max( _r_ ) =
� _ri_ =1 _[u]_ _[i]_ _[σ]_ _[i]_ [(] _[M]_ [)] _[v]_ _i_ [⊺] [;] _[ M]_ _[−]_ [max(] _[r]_ [)] [denotes the remainder, i.e.]



_M_ _−_ max( _r_ ) =



_p_ 1 _∧p_ 2
� _u_ _i_ _σ_ _i_ ( _M_ ) _v_ _i_ [⊺] _[.]_

_i_ = _r_ +1



1. First, by a well-known fact about best low-rank matrix approximation,


_σ_ _a_ + _b_ +1 _−r_ ( _X_ + _Y_ ) = min
_M_ _∈_ R _[p][×][n]_ _,_ rank( _M_ ) _≤a_ + _b−r_ _[∥][X]_ [ +] _[ Y][ −]_ _[M]_ _[∥][.]_


Hence,


_σ_ _a_ + _b_ +1 _−r_ ( _X_ + _Y_ ) _≤∥X_ + _Y −_ ( _X_ max( _a−r_ ) + _Y_ ) _∥_

= _∥X_ _−_ max( _a−r_ ) _∥_ = _σ_ _a_ +1 _−r_ ( _X_ );


similarly _σ_ _a_ + _b_ +1 _−r_ ( _X_ + _Y_ ) _≤_ _σ_ _b_ +1 _−r_ ( _Y_ ).
2. When we further have _X_ [⊺] _Y_ = 0 or _XY_ [⊺] = 0, without loss of generality
we can assume _X_ [⊺] _Y_ = 0. Then the column space of _X_ and _Y_ are
orthogonal, and rank( _X_ + _Y_ ) = rank( _X_ ) + rank( _Y_ ) = _a_ + _b_, which
means _a_ + _b ≤_ _n_ . Next, note that


( _X_ + _Y_ ) [⊺] ( _X_ + _Y_ ) = _X_ [⊺] _X_ + _Y_ [⊺] _Y_ + _X_ [⊺] _Y_ + _Y_ [⊺] _X_ = _X_ [⊺] _X_ + _Y_ [⊺] _Y,_


if we note _λ_ _i_ ( _·_ ) as the _r_ -th largest eigenvalue of the matrix, then we
have


_σ_ _i_ [2] [(] _[X]_ [ +] _[ Y]_ [ )]


= _λ_ _i_ (( _X_ + _Y_ ) [⊺] ( _X_ + _Y_ )) = _λ_ _i_ ( _X_ [⊺] _X_ + _Y_ [⊺] _Y_ )


_≥_ max( _λ_ _i_ ( _X_ [⊺] _X_ ) _, λ_ _i_ ( _Y_ [⊺] _Y_ )) (since _X_ [⊺] _X, Y_ [⊺] _Y_ are semi-positive definite)

= max( _σ_ _i_ [2] [(] _[X]_ [)] _[, σ]_ _i_ [2] [(] _[Y]_ [ ))] _[.]_


_σ_ 1 [2] [(] _[X]_ [ +] _[ Y]_ [ ) =] _[λ]_ [1] [((] _[X]_ [ +] _[ Y]_ [ )] [⊺] [(] _[X]_ [ +] _[ Y]_ [ )) =] _[ ∥][X]_ [⊺] _[X]_ [ +] _[ Y]_ [ ⊺] _[Y][ ∥]_

_≤∥X_ [⊺] _X∥_ + _∥Y_ [⊺] _Y ∥_ = _σ_ 1 [2] [(] _[X]_ [) +] _[ σ]_ 1 [2] [(] _[Y]_ [ )] _[.]_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 29


_a_ _b_
Proof of Lemma 3. 1. If _A_ = 0 _d_ and _a_ [2] _≤_ _b_ [2] + _d_ [2],
� �



2
_a_ _ab_
_A_ [⊺] _A_ =
� _ab_ _b_ [2] + _d_ [2]



�



We can solve the two eigenvalues of _A_ [⊺] _A_ are



�
_{λ_ 1 _, λ_ 2 _}_ = _[a]_ [2] [ +] _[ b]_ [2] [ +] _[ d]_ [2] _[ ±]_



( _b_ [2] + _d_ [2] _−_ _a_ [2] ) [2] + 4 _a_ [2] _b_ [2]

2 _._



_λ_ 2 _≥_ _[a]_ [2] [ +] _[ b]_ [2] [ +] _[ d]_ [2] _[ −]_ [(] _[b]_ [2] [ +] _[ d]_ [2] _[ −]_ _[a]_ [2] [)] _[ −]_ [2] _[ab]_ = _a_ ( _a −_ _b_ ) _._

2



By the definition of singular vectors, we have ( _A_ [⊺] _A −_ _λ_ 2 _I_ 2 ) _v_ 12
� _v_ 22

Thus,
( _a_ [2] _−_ _λ_ 2 ) _v_ 12 + _abv_ 22 = 0


Given _v_ 12 [2] [+] _[ v]_ 22 [2] [= 1, we have]


_ab_
_|v_ 12 _|_ =
~~�~~ ( _a_ [2] _−_ _λ_ 2 ) [2] + _a_ [2] _b_ [2]



= 0.
�



_ab_ _ab_

( _a_ [2] _−_ _a_ ( _a −_ _b_ )) [2] + _a_ [2] _b_ [2] [=] ~~_√_~~ 2 _a_



_ab_
_≥_
~~�~~ ( _a_ [2] _−_ _a_ ( _a −_



_ab_ 1

[=]
2 _a_ [2] _b_ [2] ~~_√_~~



2 _[.]_



_a_ _b_
2. If _A_ =,
_c_ _d_
� �



2 2
_a_ + _c_ _ab_ + _cd_
_A_ [⊺] _A_ = � _ab_ + _cd_ _b_ [2] + _d_ [2] � _,_



If its eigenvalues are _λ_ 1 _≥_ _λ_ 2, clearly _λ_ 1 + _λ_ 2 = _a_ [2] + _b_ [2] + _c_ [2] + _d_ [2],
_λ_ 1 _λ_ 2 = ( _ad_ _−_ _bc_ ) [2] . We can solve that its eigen-values _λ_ 1 _, λ_ 2 of _A_ [⊺] _A_ are


(1.69)



~~�~~
_{λ_ 1 _, λ_ 2 _}_ = _[a]_ [2] [ +] _[ b]_ [2] [ +] _[ c]_ [2] [ +] _[ d]_ [2] _[ ±]_



2



( _a_ [2] + _b_ [2] + _c_ [2] + _d_ [2] ) [2] _−_ 4( _ad −_ _bc_ ) [2]



= _[a]_ [2] [ +] _[ b]_ [2] [ +] _[ c]_ [2] [ +] _[ d]_ [2] _[ ±]_ ~~�~~ ( _a_ [2] + _c_ [2] _−_ _b_ [2] _−_ _d_ [2] ) [2] + 4( _ab −_ _cd_ ) [2]

2



Thus, _λ_ 2 = _[a]_ [2] [+] _[b]_ [2] [+] _[c]_ [2] [+] _[d]_ [2] _[−]_ _[√]_



_c_ [2] _−b_ [2] _−d_ [2] ) [2] +4( _ab−cd_ ) [2]

2 _≤_ _[a]_ [2] [+] _[b]_ [2] [+] _[c]_ [2] [+] _[d]_ [2] _[−]_ 2 [(] _[a]_ [2] [+] _[c]_ [2] _[−][b]_ [2] _[−][d]_ [2] [)]



( _a_ [2] + _c_ [2] _−b_ [2] _−d_ [2] ) [2] +4( _ab−cd_ ) [2]



2 = 2 2 =

_b_ [2] + _d_ [2] . Also,



_λ_ 2 _≥_ _[a]_ [2] [ +] _[ b]_ [2] [ +] _[ c]_ [2] [ +] _[ d]_ [2] _[ −]_ [(] _[a]_ [2] [ +] _[ c]_ [2] _[ −]_ _[b]_ [2] _[ −]_ _[d]_ [2] [)] _[ −]_ [2] _[|][ab][ −]_ _[cd][|]_ _≥_ _b_ [2] + _d_ [2] _−|ab−cd|._

2


30 T. T. CAI AND A. ZHANG


Thus,


_a_ [2] + _c_ [2] _−_ _λ_ 2 _≤_ _a_ [2] + _c_ [2] _−_ _b_ [2] _−_ _d_ [2] + _|ab −_ _cd|_

_≤_ 2( _a_ [2] _−_ _b_ [2] _−_ _d_ [2] ) _−_ ( _a_ [2] _−_ _b_ [2] _−_ _c_ [2] _−_ _d_ [2] ) + _|ab_ + _cd|_

_≤_ 2( _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] ) + _|ab_ + _cd|_


By the definition of singular vectors, we have ( _A_ [⊺] _A −_ _λ_ 2 _I_ 2 ) _v_ 12
� _v_ 22

Thus,
( _a_ [2] + _c_ [2] _−_ _λ_ 2 ) _v_ 12 + ( _ab_ + _cd_ ) _v_ 22 = 0

Given _v_ 12 [2] [+] _[ v]_ 22 [2] [= 1, we have]


_ab_ + _cd_
_|v_ 12 _|_ =
~~�~~ ( _a_ [2] + _c_ [2] _−_ _λ_ 2 ) [2] + ( _ab_ + _cd_ ) [2]

_ab_ + _cd_
_≥_
~~�~~ (2( _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] ) + _|ab_ + _cd|_ ) [2] + ( _ab_ + _cd_ ) [2]



= 0.
�



_ab_ + _cd_
_≥_
~~�~~ 4( _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] ) [2] + 4( _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] ) _|ab_ + _cd|_ + 2( _ab_ + _cd_ ) [2]

_ab_ + _cd_
_≥_
~~�~~ 10 max _{_ ( _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] ) [2] _,_ ( _ab_ + _cd_ ) [2] _}_



_≥_ [1]
~~_√_~~ 10



_ab_ + _cd_

_._
� _a_ [2] _−_ _d_ [2] _−_ _b_ [2] _∧_ _c_ [2] _[∧]_ [1] �



This finished the proof of the second part of Lemma 3.


Proof of Lemma 4. Suppose the SVD of _X_ is _X_ = _U_ Σ _V_ [⊺], where _U ∈_
O _p_ 1 _,r_ _, V ∈_ O _p_ 2 _,r_ _,_ Σ = diag( _σ_ 1 _, · · ·, σ_ _r_ ) _∈_ R _[r][×][r]_ . We can extend _V ∈_ O _p_ 2 _,r_
into the full _p_ 2 _× p_ 2 orthogonal matrix


[ _V V_ _⊥_ ] = _V_ 0 _∈_ O _p_ 2 _,_ _V_ _⊥_ _∈_ O _p_ 2 _,p_ 2 _−r_ _._


For convenience, denote _Y V_ ≜ _Y_ 1 . Since,


(1.70) _EY_ [⊺] _Y_ = _X_ [⊺] _X_ + _p_ 1 _I_ _p_ 2 = _V_ Σ [2] _V_ [⊺] + _p_ 1 _I_ _p_ 2 _,_ _EV_ [⊺] _Y_ [⊺] _Y V_ = Σ [2] + _p_ 1 _I_ _p_ 2 _,_


we introduce fixed normalization matrix _M ∈_ R _[p]_ [2] _[×][r]_ as



( _σ_ 1 [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ 2 [1]



( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ 2 [1]



_M_ =









2

...



_._





 _r×r_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 31


By (1.70), this design yields


_M_ [⊺] _EY_ [⊺]
1 _[Y]_ [1] _[M]_ [ =] _[ I]_ _[r]_ _[.]_


In other words, by right multiplying _M_ to _Y_ 1, we can normalize its second
moment. Now we are ready to show (1.12), (1.13) and (1.14).


1. We target on (1.12) in this step. Note that the maximum diagonal
entry of _M_ is ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ 2 [1], thus


_σ_ _r_ [2] [(] _[Y]_ [1] [)] _[ ≥]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[σ]_ _r_ [2] [(] _[Y]_ [1] _[M]_ [)] _[ ≥]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[σ]_ _[r]_ [(] _[M]_ [⊺] _[Y]_ 1 [ ⊺] _[Y]_ [1] _[M]_ [)]

_≥_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[ −]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[ ∥][M]_ [⊺] _[Y]_ 1 [ ⊺] _[Y]_ [1] _[M][ −]_ _[I]_ _[r]_ _[∥]_ _[,]_


(1.12) could be implied by
(1.71)
P ( _∥M_ [⊺] _Y_ 1 [⊺] _[Y]_ [1] _[M][ −]_ _[I]_ _[r]_ _[∥≤]_ _[x]_ [)] _[ ≥]_ [1] _[ −]_ _[C]_ [ exp] � _Cr −_ _c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[x][ ∧]_ _[x]_ [2] [�] _._


The main idea to proceed is to use _ε_ -net to split the spectral norm
deviation control to single random variable deviation control. Then
use Hanson-Wright inequality to control the single random variable.
To be specific, for any unit vector _u ∈_ R _[r]_, by expansion _Y_ 1 = _Y V_ =
_XV_ + _ZV_, we have


_u_ [⊺] _M_ [⊺] _Y_ [⊺]
1 _[Y]_ [1] _[Mu][ −]_ _[u]_ [⊺] _[I]_ _[r]_ _[u]_


= _u_ [⊺] _M_ [⊺] _Y_ [⊺]
1 _[Y]_ [1] _[Mu][ −]_ _[Eu]_ [⊺] _[M]_ [⊺] _[Y]_ 1 [ ⊺] _[Y]_ [1] _[Mu]_



(1.72)



= ( _u_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] _XV Mu −_ _Eu_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] _XV Mu_ )


+ (2 _u_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] _ZV Mu −_ _E_ 2 _u_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] _ZV Mu_ )


+ ( _u_ [⊺] _M_ [⊺] _V_ [⊺] _Z_ [⊺] _ZV Mu −_ _Eu_ [⊺] _M_ [⊺] _V_ [⊺] _Z_ [⊺] _ZV Mu_ )

=2( _XV Mu_ ) [⊺] _ZV_ ( _Mu_ ) + ( _V Mu_ ) [⊺] ( _Z_ [⊺] _Z −_ _p_ 1 _I_ _p_ 2 )( _V Mu_ ) _._



We shall emphasize that the only random variable in the equation
above is _Z ∈_ R _[p]_ [1] _[×][p]_ [2] . Our plan is to bound the two terms in (1.72)
separately as follows.


_•_ For fixed unit vector _u ∈_ R _[r]_, we vectorize _Z ∈_ R _[p]_ [1] _[×][p]_ [2] into _⃗_ **z** _∈_
R _[p]_ [1] _[p]_ [2] as follows,


_⃗_
**z** = ( _z_ 11 _, z_ 12 _, · · ·, z_ 1 _p_ 2 _, z_ 21 _, · · · z_ 2 _p_ 2 _, · · ·, z_ _p_ 1 1 _, · · ·, z_ _p_ 1 _p_ 2 ) [⊺] _._


We also repeat the ( _V Mu_ )( _V Mu_ ) [⊺] block for _p_ 1 times and introduce



 _∈_ R ( _p_ 1 _p_ 2 ) _×_ ( _p_ 1 _p_ 2 ) _._



_⃗_
**D** =



( _V Mu_ )( _V Mu_ ) [⊺]

...

 ( _Mu_ )( _Mu_ ) [⊺]


32 T. T. CAI AND A. ZHANG


It is obvious that ( _V Mu_ ) [⊺] ( _Z_ [⊺] _Z −_ _I_ _p_ 1 )( _V Mu_ ) = _⃗_ **z** [⊺] **D** _[⃗]_ _⃗_ **z** _−_ _E⃗_ **z** [⊺] **D** _[⃗]_ _⃗_ **z** .
Besides,


_∥_ **D** _[⃗]_ _∥_ = _∥_ ( _V Mu_ )( _V Mu_ ) [⊺] _∥_ = _∥Mu∥_ 2 [2] _[≤∥][M]_ _[∥]_ [2] _[∥][u][∥]_ 2 [2] [= (] _[σ]_ _r_ [2] [(] _[X]_ [)+] _[p]_ [1] [)] _[−]_ [1] _[,]_


_∥_ **D** _[⃗]_ _∥_ _F_ [2] [=] _[ p]_ [1] _[∥]_ [(] _[V Mu]_ [)(] _[V Mu]_ [)] [⊺] _[∥]_ _F_ [2] [=] _[ p]_ [1] _[∥][Mu][∥]_ 2 [4] _[≤]_ _[p]_ [1] [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ [2] _[.]_


By Hanson-Wright Inequality (Theorem 1 in Rudelson and Vershynin (2013)),


P _{|_ ( _V Mu_ ) [⊺] ( _Z_ [⊺] _Z −_ _p_ 1 _I_ _p_ 2 )( _V Mn_ ) _| > x}_


=P _⃗_ **z** ⊺ **D** _⃗_ _⃗_ **z** _−_ _E⃗_ **z** ⊺ **D** _⃗_ _⃗_ **z** _> x_
���� ��� �


2 2
_≤_ 2 exp _−c_ min _x_ ( _σ_ _r_ [(] _[X]_ [)][ +] _[p]_ [1] [)] [2] _, x_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _,_
� � _p_ 1 ��


where _c_ only depends on _τ_ .


_•_ Next, we bound


( _XV Mu_ ) [⊺] _Z_ ( _V Mu_ ) = tr( _Z_ ( _V Mu_ )( _XV Mu_ ) [⊺] )


_⃗_
= **z** [⊺] vec( _V Mu_ ( _XV Mu_ ) [⊺] ) _,_


where _⃗_ **z**, vec( _V Mu_ ( _XV Mu_ ) [⊺] ) are the vectorized _Z_, ( _V Mu_ ( _XV Mu_ ) [⊺] ).
Since



_σ_ 1 ( _X_ )( _σ_ 1 [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ 2 [1]



_σ_ _r_ ( _X_ )( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[−]_ 2 [1]



_XV M_ = _U_









2

...






_,_




we know _∥XV M_ _∥≤_ 1, and


_∥_ vec( _V Mu_ ( _XV Mu_ ) [⊺] ) _∥_ 2 [2] [=] _[ ∥]_ [(] _[MV u]_ [)(] _[XV Mu]_ [)] [⊺] _[∥]_ [2] _F_
= _∥Mu∥_ 2 [2] _[· ∥][XV Mu][∥]_ [2] 2 _[≤∥][M]_ _[∥]_ [2] _[ ≤]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [)] _[−]_ [1] _[.]_


By the basic property of i.i.d. sub-Gaussian random variables, we
have


P ( _|_ ( _XV Mu_ ) [⊺] _Z_ ( _V Mu_ ) _| > x_ ) _≤_ _C_ exp � _−cx_ [2] ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] � _,_


where _C, c_ only depends on _τ_ .


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 33


The two bullet points above and (1.72) implies for any fixed unit vector
_u ∈_ R _[r]_,


(1.73)
P ( _|u_ [⊺] _M_ [⊺] _Y_ 1 [⊺] _[Y]_ [1] _[Mu][ −]_ _[u]_ [⊺] _[I]_ _[r]_ _[u][|][ > x]_ [)] _[ ≤]_ _[C]_ [ exp] � _−c_ � _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] � _x_ [2] _∧_ _x_ � _,_


for all _x >_ 0. Here the _C, c_ above only depends on _τ_ . Next, the _ε_ -net
argument (Lemma 5) leads to
(1.74)
P ( _∥M_ [⊺] _Y_ 1 [⊺] _[Y]_ [1] _[M][ −]_ _[I]_ _[r]_ _[∥]_ _[> x]_ [)] _[ ≤]_ _[C]_ [ exp] � _Cr −_ _c_ � _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] � _x_ [2] _∧_ _x_ � _._


In other words, (1.71) holds, which implies (1.12).
2. In order to prove (1.13), we use the following fact about best rank- _r_
approximation of _Y_,


_σ_ _r_ ( _Y_ ) = max
rank( _B_ ) _≤r_ _[∥][Y][ −]_ _[B][∥≥∥][Y][ −]_ _[Y][ ·]_ [ [] _[V]_ [ 0]] _[∥]_ [=] _[ σ]_ [max] [(] _[Y V]_ _[⊥]_ [)] _[.]_


to switch our focus from _σ_ _r_ ( _Y_ ) to _σ_ max ( _Y V_ _⊥_ ). Next,


_σ_ max [2] [(] _[Y V]_ _[⊥]_ [) =] _[ σ]_ [max] [(] _[V]_ [ ⊺] _⊥_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ [)] _[ ≤]_ _[p]_ [1] [ +] �� _V_ ⊺ _⊥_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ _[−]_ _[EV]_ [ ⊺] _⊥_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ �� _._


Note that _EV_ [⊺]
_⊥_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ [=] _[ p]_ [1] _[I]_ _[p]_ [2] _[−][r]_ [, based on essentially the same pro-]
cedure as the proof for (1.12), one can show that


_−_ 1
P ��� _p_ 1 _[V]_ _[⊥]_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ _[−]_ _[Ep]_ _[−]_ 1 [1] _[V]_ _[⊥]_ _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ �� _≥_ _x_ � _≤_ _C_ exp � _Cp_ 2 _−_ _cp_ 1 � _x_ [2] _∧_ _x_ �� _._


Then we obtain (1.13) by combining the two inequalities above.
3. Finally, we consider _∥_ P _Y V_ _Y V_ _⊥_ _∥_ . Since


_∥_ P _Y V_ _Y V_ _⊥_ _∥_ = _∥_ P _Y V M_ _Y V_ _⊥_ _∥_



(1.75)



= _∥_ ( _Y V M_ )(( _Y V M_ ) [⊺] ( _Y V M_ )) _[−]_ [1] ( _Y V M_ ) [⊺] _Y V_ _⊥_ _∥_


(8.18)
_≤_ _σ_ min _[−]_ [1] [(] _[Y V M]_ [)] _[∥][M]_ [⊺] _[V]_ [ ⊺] _[Y]_ [ ⊺] _[Y V]_ _[⊥]_ _[∥]_



We analyze _σ_ min ( _Y V M_ ) and _∥M_ [⊺] _V_ [⊺] _Y_ [⊺] _V_ _⊥_ _∥_ separately below.
Since


_σ_ min [2] [(] _[Y V M]_ [) =] _[ σ]_ [min] [(] _[M]_ [⊺] _[V]_ [ ⊺] _[Y]_ [ ⊺] _[Y V M]_ [)] _[ ≥]_ [1] _[ −∥][M]_ [⊺] _[V]_ [ ⊺] _[Y]_ [ ⊺] _[Y V M][ −]_ _[I]_ _[r]_ _[∥][,]_


by (1.71), we know there exists _C, c_ only depending on _τ_ such that


P � _σ_ min [2] [(] _[Y V M]_ [)] _[ ≥]_ [1] _[ −]_ _[x]_ � _≥_ 1 _−_ exp � _Cr −_ _c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] _[x][ ∧]_ _[x]_ [2] [�] _._


34 T. T. CAI AND A. ZHANG


Set _x_ = 1 _/_ 2, we could choose _C_ gap large enough, but only depends
on _τ_, such that whenever _σ_ _r_ [2] [(] _[X]_ [)] _[ ≥]_ _[C]_ [gap] _[p]_ [2] _[≥]_ _[C]_ [gap] _[r]_ [,] _[ Cr][ −]_ _[c]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +]
_p_ 1 ) _x ∧_ _x_ [2] _≤−_ 8 _[c]_ [(] _[σ]_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [). Under such setting,]



(1.76) P _σ_ min [2] [(] _[Y V M]_ [)] _[ ≥]_ [1]
� 2



_≥_ 1 _−_ _C_ exp � _−c_ ( _σ_ _r_ [2] [(] _[X]_ [) +] _[ p]_ [1] [)] � _._
�



For _∥M_ [⊺] _V_ [⊺] _Y_ [⊺] _Y V_ _⊥_ _∥_, since _XV_ _⊥_ = 0, we have the following decomposition,


_M_ [⊺] _V_ [⊺] _Y_ [⊺] _Y V_ _⊥_ = _M_ [⊺] _V_ [⊺] ( _X_ + _Z_ ) [⊺] ( _X_ + _Z_ ) _V_ _⊥_

= _M_ [⊺] _V_ [⊺] _X_ [⊺] _ZV_ _⊥_ + _M_ [⊺] _V_ [⊺] _Z_ [⊺] _ZV_ _⊥_ _._


Follow the similar idea of the proof for (1.12), we can show for any
unit vectors _u ∈_ R _[r]_ _, v ∈_ R _[p]_ [2] _[−][r]_,


P ( _|u_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] _ZV_ _⊥_ _v| ≥_ _x_ ) _≤_ _C_ exp � _−cx_ [2] _/∥_ ( _V_ _⊥_ _v_ )( _u_ [⊺] _M_ [⊺] _V_ [⊺] _X_ [⊺] ) _∥_ _F_ [2] �

_≤C_ exp( _−cx_ [2] ) _._


P ( _|u_ [⊺] _M_ [⊺] _V_ [⊺] _Z_ [⊺] _ZV_ _⊥_ _v| ≥_ _x_ ) = P ( _|u_ [⊺] _M_ [⊺] _V_ [⊺] ( _Z_ [⊺] _Z −_ _p_ 1 _I_ _p_ 2 ) _V_ _⊥_ _v| ≥_ _x_ )



_≤C_ exp( _−c_ min( _x_ [2] _,_ �



_σ_ _r_ [2] + _p_ 1 _x_ )) _._



By the _ϵ_ -net argument again (Lemma 4), we have
(1.77)
P ( _∥M_ [⊺] _V_ [⊺] _Y_ [⊺] _Y V_ _⊥_ _∥≥_ _x_ ) _≤_ _C_ exp _Cp_ 2 _−_ _c_ min( _x_ [2] _,_ �
�


Combining (1.75), (1.76) and (1.77), we obtain (1.14).



_σ_ _r_ [2] + _p_ 1 _x_ ) _._
�



Proof of Lemma 5. First, based on Lemma 2.5 in Vershynin (2011),
there exists _ε_ -nets _W_ _L_ in B _[p]_ [1], _W_ _R_ in B _[p]_ [2], namely for any _u ∈_ B _[p]_ [1], _v ∈_ B _[p]_ [2],
there exists _u_ 0 _∈_ _W_ _L_, _v_ 0 _∈_ _W_ _R_ such that _∥u_ 0 _−_ _u∥_ 2 _≤_ _ε_, _∥v_ 0 _−_ _v∥_ 2 _≤_ _ε_ and
_|W_ _L_ _| ≤_ (1+2 _/ε_ ) _[p]_ [1] _, |W_ _R_ _| ≤_ (1+2 _/ε_ ) _[p]_ [2] . Especially we choose _ε_ = 1 _/_ 3. Under
the event that
_Q_ = _{|u_ [⊺] _Kv| ≥_ _t, ∀u ∈_ _W_ _L_ _, v ∈_ _W_ _R_ _},_


denote ( _u_ _[∗]_ _, v_ _[∗]_ ) = arg max _u∈_ B _p_ 1 _,v∈_ B _p_ 2 _|u_ [⊺] _Kv|_, _α_ = max _u∈_ B _[p]_ 1 _,v∈_ B _[p]_ 2 _|u_ [⊺] _Kv|_,
1
then _α_ = _∥K∥_ . According to the definition of 3 [-net, there exists] _[ u]_ 0 _[∗]_ _[∈]_
_W_ _L_ _, v_ 0 _[∗]_ _[∈]_ _[W]_ _[R]_ [ such that] _[ ∥][u]_ _[∗]_ _[−]_ _[u]_ _[∗]_ 0 _[∥]_ [2] _[ ≤]_ [1] _[/]_ [3] _[,][ ∥][v]_ _[∗]_ _[−]_ _[v]_ 0 _[∗]_ _[∥]_ [2] _[ ≤]_ [1] _[/]_ [3. Then,]


_α_ = _|_ ( _u_ _[∗]_ ) [⊺] _Kv_ _[∗]_ _| ≤|_ ( _u_ _[∗]_ 0 [)] [⊺] _[Kv]_ 0 _[∗]_ _[|]_ [ +] _[ |]_ [(] _[u]_ _[∗]_ _[−]_ _[u]_ [0] [)] [⊺] _[Kv]_ 0 _[∗]_ _[|]_ [ +] _[ |]_ [(] _[u]_ _[∗]_ [)] [⊺] _[K]_ [(] _[v]_ _[∗]_ _[−]_ _[v]_ [0] [)] _[|]_

_≤t_ + _∥u_ _[∗]_ _−_ _u_ 0 _∥_ 2 _· ∥K∥· ∥v_ 0 _[∗]_ _[∥]_ [2] [+] _[ ∥][u]_ _[∗]_ _[∥]_ [2] _[· ∥][K][∥· ∥][v]_ _[∗]_ _[−]_ _[v]_ [0] _[∥]_ [2] _[≤]_ _[t]_ [ + 2]

3 _[α]_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 35


Thus, _α ≤_ 3 _t_ when _Q_ happens. Finally, since



P( _Q_ ) _≤_ �

_u∈W_ _L_



� P ( _|u_ [⊺] _Kv| ≥_ _t_ ) _≤_ 7 _[p]_ [1] _·_ 7 _[p]_ [2] _·_ max _u∈_ B _[n]_ [ P][ (] _[|][u]_ [⊺] _[Ku][| ≥]_ _[t]_ [)] _[,]_

_v∈W_ _R_



which finished the proof of this lemma.


Proof of Lemma 6. For each representative _P_ _φ_ _∈_ _co_ ( _P_ _φ_ ), we suppose
_m_ [(] _[φ]_ [)] is a measure on _P_ _φ_ such that


(1.78) _P_ _φ_ = _P_ _θ_ _dm_ [(] _[φ]_ [)] _._
� _P_ _φ_


Thus,


sup E _P_ _θ_ _d_ [2] ( _T_ ( _θ_ ) _,_ _φ_ [ˆ] ) = sup sup
_θ∈_ Θ � � _φ∈_ Φ _θ∈P_ _φ_



=E _P_ _θ_ _d_ [2] ( _φ,_ _φ_ [ˆ] ) _≥_ sup
� � _φ∈_ Φ



E _P_ _θ_ _d_ [2] ( _φ,_ _φ_ [ˆ] ) _dm_ [(] _[φ]_ [)]

� _P_ _φ_ � �



_d_ [2] ( _φ,_ _φ_ [ˆ] ) _dP_ _θ_ _dm_ [(] _[φ]_ [)]
� � [�]

_P_ _φ_



� R _[p]_



_d_ [2] ( _φ,_ _φ_ [ˆ] ) _dP_ _θ_ _dm_ [(] _[φ]_ [)] = sup
� � _φ∈_ Φ



� R _[p]_



= sup
_φ∈_ Φ


(1.78)
= sup
_φ∈_ Φ



� _P_ _φ_

� R _[p]_



� _d_ [2] ( _φ,_ _φ_ [ˆ] )� _dP_ _φ_ = sup _φ∈_ Φ E _P_ _φ_ [ _d_ [2] ( _φ,_ _φ_ [ˆ] )] _._



Proof of Lemma 7. The direct calculation for _D_ ( _P_ [¯] _V,t_ _||P_ [¯] _V_ _′_ _,t_ ) is relatively difficult, thus we detour by introducing the similar density to _P_ [¯] _V,t_ as
follows,



_P_ ˜ _V,t_ ( _Y_ ) =
�



1

[exp(] _[−∥][Y][ −]_ [2] _[tWV]_ [ ⊺] _[∥]_ _F_ [2] _[/]_ [2)]
R _[p]_ [2] _[×][r]_ (2 _π_ ) _[p]_ [1] _[p]_ [2] _[/]_ [2]



(1.79) R _[p]_ [2] _[×][r]_

_·_ ( _[p]_ [1] _F_ _[/]_ [2)] _[dW.]_

2 _π_ [)] _[p]_ [1] _[r/]_ [2] [ exp(] _[−][p]_ [1] _[∥][W]_ _[∥]_ [2]




_·_ ( _[p]_ [1]



We can see _P_ [˜] _V,t_ is another mixture of Gaussian distributions, thus it is
indeed a density which sums up to 1. Since _V ∈_ O _n,r_,


(1.80) _V_ [⊺] _V_ = _I_ _r_ _,_


36 T. T. CAI AND A. ZHANG


Denote _Y_ _i·_ as the _i_ -th row of _Y_ . Note that _P_ [˜] _V,t_ can be simplified as


(1.81)



_P_ ˜ _V,t_ ( _Y_ ) (1.80 = ) _p_ _[p]_ 1 [1] _[r/]_ [2]
(2 _π_ ) _[p]_ [1] [(] _[p]_ [2] [+] _[r]_ [)] _[/]_ [2]



�



R _[n][×][r]_ [ exp] � _−_ 2 [1]



_Y Y_ [⊺] _−_ 2 _tY V W_ [⊺] _−_ 2 _tWV_ [⊺] _Y_ [⊺]
2 [tr] �



+ 4 _t_ [2] _WW_ [⊺] + _p_ 1 _WW_ [⊺] [��] _dW_



= _p_ _[p]_ 1 [1] _[r/]_ [2]
(2 _π_ ) _[p]_ [1] [(] _[p]_ [2] [+] _[r]_ [)] _[/]_ [2]



�




[1] _Y_ ( _I_ _p_ 2 _−_ 4 _t_ [2]

2 [tr] � 4 _t_ [2] +



_V V_ [⊺] ) _Y_ [⊺]
4 _t_ [2] + _p_ 1



R _[n][×][r]_ [ exp] � _−_ 2 [1]



2 _t_ 2 _t_
+ (4 _t_ [2] + _p_ 1 )( _W −_ _Y V_ )( _W −_ _Y V_ ) [⊺] [��] _dW._
4 _t_ [2] + _p_ 1 4 _t_ [2] + _p_ 1



�



4 _t_ [2] + _p_ 1 _V V_ [⊺] ) _Y_ _i_ [⊺] _·_



�



=



_p_ 1 _r/_ 2
� _p_ 1 _/_ (4 _t_ [2] + _p_ 1 )�

exp
(2 _π_ ) _[p]_ [1] _[p]_ [2] _[/]_ [2]



_−_ [1]



_._



2



_p_ 1
�



1 4 _t_ [2]
� _Y_ _i·_ ( _I −_ 4 _t_ [2] +

_i_ =1



From the calculation above we can see _P_ [˜] _V,t_ is actually joint normal, i.e. when
_Y ∼_ _P_ [˜] _V,t_,



_−_ 1 [�]

4 _t_ [2]

_V V_ [⊺]
4 _t_ [2] + _p_ 1 �



= _N_ 0 _, I_ _p_ 2 + [4] _[t]_ [2] _V V_ [⊺] _,_ _i_ = 1 _· · ·, p_ 1 _._
� _p_ 1 �



_iid_
_Y_ _i·_ _∼_ _N_



�



4 _t_ [2]
0 _,_ � _I_ _p_ 2 _−_ 4 _t_ [2] +



It is widely known that the KL-divergence between two _p_ -dimensional multivariate Gaussians is


_D_ ( _N_ ( _µ_ 0 _,_ Σ 0 ) _||N_ ( _µ_ 1 _,_ Σ 1 ))



(1.82) det Σ 1

= [1] tr �Σ _[−]_ 0 [1] [Σ] [1] � + ( _µ_ 1 _−_ _µ_ 0 ) [⊺] Σ _[−]_ 1 [1] [(] _[µ]_ [1] _[ −]_ _[µ]_ [0] [)] _[ −]_ _[p]_ [ + log] _._

2 � � det Σ 0 ��



= [1]



2



det Σ 1
tr �Σ _[−]_ 0 [1] [Σ] [1] � + ( _µ_ 1 _−_ _µ_ 0 ) [⊺] Σ _[−]_ 1 [1] [(] _[µ]_ [1] _[ −]_ _[µ]_ [0] [)] _[ −]_ _[p]_ [ + log]
� � det Σ 0



det Σ 0



We can calculate that for any two _V, V_ _[′]_ _∈_ O _p_ 1 _,r_,


_D_ ( _P_ [˜] _V,t_ _||P_ [˜] _V_ _′_ _,t_ )



�



= _[p]_ [1]

2


= _[p]_ [1]

2



4 _t_ [2]
�tr �� _I_ _p_ 2 _−_ 4 _t_ [2] +



4 _t_ [2]

tr( _V V_ [⊺] ) + [4] _[t]_ [2]
4 _t_ [2] + _p_ 1 _p_ 1



4 _t_ [2]

4 _t_ [2] + _p_ 1 _V V_ [⊺] � � _I_ _p_ 2 + [4] _p_ _[t]_ 1 [2]




_[t]_ [2]

_V_ _[′]_ ( _V_ _[′]_ ) [⊺] _−_ _p_ 2
_p_ 1 ��



4 _t_ [2]

_−_
� 4 _t_ [2] +



tr( _V_ _[′]_ ( _V_ _[′]_ ) [⊺] )
_p_ 1



16 _t_ [4]

_−_

_p_ 1 (4 _t_ [2] + _p_ 1 ) [tr(] _[V V]_ [ ⊺] [(] _[V]_ _[ ′]_ [)(] _[V]_ _[ ′]_ [)] [⊺] [)] �



(1.80) 16 _t_ [4]
= � _r −∥V_ [⊺] _V_ _[′]_ _∥_ _F_ [2] �

2(4 _t_ [2] + _p_ 1 )



Lemma 1 16 _t_ [4]
= _F_

2(4 _t_ [2] + _p_ 1 ) _[∥]_ [sin Θ(] _[V, V]_ _[ ′]_ [)] _[∥]_ [2]


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 37


Next, we show that _P_ [¯] _V,t_ ( _Y_ ) and _P_ [˜] _V,t_ ( _Y_ ) are very close in terms of calculating KL-divergence. To be specific, we show when _C_ _r_ is large enough but
with a uniform choice, there exists a uniform constant _c_ such that


(1.83) 1 _−_ 2 exp( _−cp_ 1 ) _≤_ _P_ ¯˜ _V,t_ ( _Y_ ) _≤_ 1 + 2 exp( _−cp_ 1 ) _,_ _∀Y ∈_ R _[p]_ [1] _[×][p]_ [2] _._
_P_ _V,t_ ( _Y_ )


According to (1.24) and (1.81), we know for any fixed _Y_,


(1.84) _P_ ¯ _V,t_ ( _Y_ )

_P_ ˜ _V,t_ ( _Y_ )



exp � _−_ tr( _Y Y_ [⊺] _−_ 2 _tY V W_ [⊺] _−_ 2 _tWV_ [⊺] _Y_ [⊺] + (4 _t_ [2] + _p_ 1 ) _WW_ [⊺] ) _/_ 2�
2 [1]



= _C_ _V,t_



�



_σ_ min ( _W_ ) _≥_ [1]



4 _t_ [2]

_·_ exp tr( _Y_ ( _I −_
� 4 _t_ [2] +



4 _t_ [2] 4 _t_ 2 + _p_ 1

_V V_ [⊺] ) _Y_ [⊺] ) _/_ 2 _·_
4 _t_ [2] + _p_ 1 � � 2 _π_



_p_ 1 _r/_ 2
_dW_
�



2 _π_



�



2 _t_
_−_ (4 _t_ [2] + _p_ 1 ) _W −_ _Y V_
���� 4 _t_ [2] + _p_ 1 ����



_/_ 2

_F_



�



_p_ 1 _r/_ 2

exp
�



2



2



= _C_ _V,t_



�



_σ_ min ( _W_ ) _≥_ [1]



_dW_



4 _t_ 2 + _p_ 1
� 2 _π_



�



= _C_ _V,t_ P



_σ_ min ( _W_ [˜] ) _≥_ [1]



2



_._



˜ 2 _t_ _I_ ( _p_ 1 _r_ )
_W ∈_ R _[p]_ [1] _[×][r]_ _,_ ˜ _W ∼_ _N_ _Y V,_
� 4 _t_ [2] + _p_ 1 4 _t_ [2] + _p_ 1
�����



� [�]



For fixed _Y ∈_ R _[p]_ [1] _[×][p]_ [2], _Y V ∈_ R _[p]_ [1] _[×][r]_, we can find _Q ∈_ O _p_ 1 _,p_ 1 _−r_ which is
orthogonal to _Y V_, i.e. _Q_ [⊺] _Y V_ = 0. Then _Q_ [⊺] _W_ [˜] _∈_ R [(] _[p]_ [1] _[−][r]_ [)] _[×][r]_ and _Q_ [⊺] _W_ [˜] are
i.i.d. normal distributed with mean 0 and variance 1 _/_ (4 _t_ [2] + _p_ 1 ). By standard
result in random matrix (e.g. Corollary 5.35 in Vershynin (2012)), we have



1
_σ_ min ( _W_ [˜] ) = _σ_ _r_ ( _W_ [˜] ) _≥_ _σ_ _r_ ( _Q_ [⊺] _W_ [˜] ) =
~~�~~ 4 _t_ [2]



4 _t_ [2] + _p_ 1 _σ_ _r_ � ~~�~~



4 _t_ [2] + _p_ 1 _Q_ [⊺] _W_ [˜] �



1
_≥_
~~�~~ 4 _t_ [2] + _p_ 1



� _√p_ 1 _−_ _r −√r −_ _x_ �



with probability at least 1 _−_ 2 exp( _−x_ [2] _/_ 2). Since _t_ [2] _≤_ _p_ 1 _/_ 4, _p_ 1 _≥_ 16 _r_, we
further set _x_ = 0 _._ 078 _[√]_ ~~_p_~~ 1 ~~,~~ the inequality above further yields



_p_ 1 _/_ 16 _−_ 0 _._ 078 _[√]_ ~~_p_~~ 1
_≥_ [1]
~~_√_~~ 2 _p_ 1 2



1
_σ_ min ( _W_ [˜] ) _≥_ ( _[√]_ _p_ 1 _−_ _r−_ _[√]_ _r−x_ ) _≥_
~~�~~ 4 _t_ [2] + _p_ 1


with probability at least 1 _−_ exp( _−cp_ 1 ).



~~�~~



15 _p_ 1 _/_ 16 _−_ ~~�~~



2


38 T. T. CAI AND A. ZHANG


Thus, for fixed _Y_, _p_ 1 _≥_ 16 _r_,
(1.85)



P



�



_σ_ min ( _W_ [˜] ) _≥_ [1]



2



˜ 2 _t_ _I_ ( _p_ 1 _r_ )
����� _W ∈_ R _[p]_ [1] _[×][r]_ _,_ ˜ _W ∼_ _N_ � 4 _t_ [2] + _p_ 1 _Y V,_ 4 _t_ [2] + _p_ 1 � [�]



_≥_ 1 _−_ exp( _−cp_ 1 ) _._



Recall the definition of _C_ _V,t_, we have


_C_ _V,t_ _[−]_ [1] [=][ P] � _σ_ min ( _W_ ) _≥_ 1 _/_ 2��� _W ∈_ R _p_ 1 _×r_ _, W_ _iid_ _∼_ _N_ (0 _,_ 1 _/p_ 1 )� _._


Also recall the assumption that _r ≤_ 16 _p_ 1, Corollary 5.35 in Vershynin (2012)
yields
P( _σ_ min ( _W_ ) _<_ 1 _/_ 2) _≤_ exp( _−cp_ 1 ) _._


Thus,


(1.86) 1 _< C_ _V,t_ _<_ 1 + 2 exp( _−cp_ 1 )


Combining (1.84), (1.86) and (1.85) we have proved (1.83).
Finally, we can bound the KL divergence for _P_ [¯] _V,t_ and _P_ [¯] _V_ _′_ _,t_ based on the
previous steps.


_D_ ( _P_ [¯] _V,t_ _||P_ [¯] _V_ _′_ _,t_ )



¯
¯ _P_ _V,t_ ( _Y_ )
_P_ _V,t_ ( _Y_ ) log ~~¯~~
_Y ∈_ R _[p]_ [1] _[×][p]_ [2] � _P_ _V_ _′_ _,t_ ( _Y_ )



=
�



_dY_
�



�



��



¯

= _P_ _V,t_ ( _Y_ )
� _Y ∈_ R _[p]_ [1] _[×][p]_ [2]



�



log



_P_ ~~¯~~ _V_ _′_ _,t_ ( _Y_ )


_P_ ¯ _V,t_ ( _Y_ )
� _P_ ˜ _V,t_ ( _Y_ )



_P_ ¯ _V,t_ ( _Y_ )
� _P_ ˜ _V,t_ ( _Y_ )



�



+ log



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



+ log



_P_ ˜ _V_ _′_ _,t_ ( _Y_ )
� _P_ ~~¯~~ _V_ _′_ _,t_ ( _Y_ )



_dY_



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



�



+ 2 log(1 + exp( _−cp_ 1 ))

�



�



_≤_ _P_ ¯ _V,t_ ( _Y_ )
� _Y_



log

�



_dY_



˜

_≤C_ exp( _−cp_ 1 ) + _P_ _V,t_ ( _Y_ ) log
� _Y_



_≤C_ exp( _−cp_ 1 ) +
�



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )

_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



�



_dY_



_dY_
������



+
� _Y_



( ˜ _P_ _V,t_ ( _Y_ ) _−_ _P_ ¯ _V,t_ ( _Y_ )) log
�����



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



16 _t_ [4]
_≤_ _F_ [+] _[ C]_ [ exp(] _[−][cp]_ [1] [)]

2(4 _t_ [2] + _p_ 1 ) _[∥]_ [sin Θ(] _[V, V]_ _[ ′]_ [)] _[∥]_ [2]



_dY_
������



+ exp( _−cp_ 1 )
� _Y_



_P_ ˜ _V,t_ ( _Y_ ) log
�����



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 39



For the last term in the formula above, we can calculate accordingly as


˜

˜ _P_ _V,t_ ( _Y_ )
_P_ _V,t_ ( _Y_ ) log ˜ _dY_

� _Y_ ����� � _P_ ( _Y_ ) ������



_dY_
������



_Y_



_P_ ˜ _V,t_ ( _Y_ ) log
�����



_P_ ˜ _V,t_ ( _Y_ )
� _P_ ˜ _V_ _′_ _,t_ ( _Y_ )



_P_ ˜ _V_ _′_ _,t_ ( _Y_ )



_dY_
�����



˜

= _P_ _V,t_ ( _Y_ ) _·_
� _Y_



4 _t_ [2]


4 _t_ [2] + _p_ 1

�����



_p_
� _Y_ _i·_ � _V V_ [⊺] _−_ _V_ _[′]_ ( _V_ _[′]_ ) [⊺] [�] _Y_ _i_ [⊺] _·_ _[/]_ [2]


_i_ =1



4 _t_ [2]


4 _t_ [2] + _p_ 1

�



4 _t_ [2]


4 _t_ [2] +

�



_Y_ _i·_ _∼_ _N_ 0 _,_ _I_ _p_ 2 + [4] _[t]_ [2] _V V_ [⊺]
����� � � _p_ 1 �� [�]



_≤E_



_p_ 1
�


_i_ =1



1
� _V V_ [⊺] + _V_ _[′]_ ( _V_ _[′]_ ) [⊺] [�] _Y_ _i_ [⊺] _·_
2 _[Y]_ _[i][·]_



4 _t_ [2]

2(4 _t_ [2] + _p_ 1 ) [tr] �� _V V_ [⊺] + _V_ _[′]_ ( _V_ _[′]_ ) [⊺] [�] _·_ � _I_ _p_ 2 + [4] _p_ _[t]_ 1 [2]



4 _t_ [2]
= _p_ 1 _·_




_[t]_ [2]

_V V_ [⊺]
_p_ 1 ��



4 _t_ [2]
_≤p_ 1 _·_



4 _t_ [2]

2(4 _t_ [2] + _p_ 1 ) [tr] �2 _V_ [⊺] � _I_ _p_ 2 + [4] _p_ _[t]_ 1 [2]




_[t]_ [2]

_V V_ [⊺] _V_
_p_ 1 � �



=4 _t_ [2] _r ≤_ _p_ [2] 1 _[.]_


Since exp( _−cp_ 1 ) _p_ [2] 1 [is upper bounded by some constant for all] _[ p]_ [1] _[ ≥]_ [1. To]
sum up there exists uniform constant _C_ _KL_ such that for all _V, V_ _[′]_ _∈_ O _p_ 2 _,r_,


16 _t_ [4]
_D_ ( _P_ [¯] _V,t_ _||P_ [¯] _V_ _′_ _,t_ ) _≤_ _F_ [+] _[ C]_ _[KL]_ _[,]_

2(4 _t_ [2] + _p_ 1 ) _[∥]_ [sin Θ(] _[V, V]_ _[ ′]_ [)] _[∥]_ [2]


which has finished the proof of this lemma.


Proof of Lemma 8. First of all, since left and right multiplication for
_G_ [1: _r,_ :] does not change the essence of the problem, without loss of generality
we can assume that



_∈_ O _d,r_ _._








_U_ _G_ =









1

...

1

0 _· · ·_ 0



In such case, all non-zero entries of _G_ are zero except the top left _r_ _×r_ block.
Based on the random matrix theory (Lemma 4), we know


1
(1.87) P � _σ_ min [2] [(] _[L]_ [1: _r,_ :] [)] _[ ≥]_ _[t]_ [2] [ +] _[ p][ −]_ _[C]_ �( _dp_ ) 2 + _d_ �� _≥_ 1 _−_ _C_ exp( _−cd_ ) _,_


2 1
(1.88) P ��� _L_ [( _r_ +1): _n,_ :] �� _≤_ _n_ + _C_ �( _pn_ ) 2 + _p_ �� _≥_ 1 _−_ _C_ exp( _−cp_ ) _,_


2
(1.89) P ��� _L_ [( _r_ +1): _d,_ :] �� _≤_ _Cd_ � _≥_ 1 _−_ _C_ exp( _−cd_ ) _._


40 T. T. CAI AND A. ZHANG


1. First we consider _σ_ min ( _U_ [ˆ] [1: _r,_ :] ). By Lemma 9 and (1.87), (1.88), we
know with probability at least 1 _−_ _C_ exp( _−cd ∧_ _p_ ).


_σ_ [2]
min [(] _[L]_ [[][1:] _[r][,]_ [:][]] [)]
_σ_ min [2] [( ˆ] _[U]_ [1: _r,_ :] [)] _[ ≥]_
_σ_ min [2] [(] _[L]_ [[1:] _[r,]_ [:]] [) +] _[ σ]_ max [2] [(] _[L]_ [[(] _[r]_ [+1):] _[p,]_ [:]] [)]


1
_t_ [2] + _p −_ _C_ ( _dp_ ) 2 + _d_
� �
_≥_ 1 1 _._

_t_ [2] + _p −_ _C_ ( _dp_ ) 2 + _d_ + _n_ + _C_ ( _pn_ ) 2 + _p_
~~�~~ ~~�~~ ~~�~~ ~~�~~


We target on showing under the statement given in the lemma,



1
_t_ [2] + _p −_ _C_ ( _dp_ ) 2 + _d_
� �
(1.90)



_t_ [2] + _n_ _[.]_



+ _p −_ 2 +

_t_ [2] + _p −_ _C_ ( _dp_ ) 12 + _d_ + _n_ + _C_ ( _pn_ ) 12 + _p_ _≥_ _[p]_ _t_ [ +] [2] + _[ t]_ [2] _n_ _[/]_ [2]
~~�~~ ~~�~~ ~~�~~ ~~�~~



The inequality above is implied by

1 1
_t_ [2] + _p −_ _C_ (( _dp_ ) 2 + _d_ ) ( _t_ [2] + _n_ ) _≥_ _t_ [2] + _n_ + _C_ ( _pn_ ) 2 + _p_ _p_ + _t_ [2] _/_ 2�
� � � � ���


2
_n_ + _t_ 1 1
_⇐t_ [2] _−_ _C_ ( _pn_ ) 2 + _p_ + ( _dp_ ) 2 + _d_
� 2 � � [�]


1 1
_−_ _C_ _p_ [3] _[/]_ [2] _n_ 2 + _p_ [2] + ( _dp_ ) 2 _n_ + _dn_ _≥_ 0 _._
� �


1
In fact, whenever _t_ [2] _> C_ gap ( _dp_ ) 2 + _d_ + _p_ [3] _[/]_ [2] _n_ _[−]_ 2 [1], _n > C_ 0 _p_, we have
� �


2
_n_ + _t_ 1 1
_t_ [2] _−_ _C_ ( _pn_ ) 2 + _p_ + ( _dp_ ) 2 + _d_
� 2 � � [�]


1 1
_−_ _C_ _p_ [3] _[/]_ [2] _n_ 2 + _p_ [2] + ( _dp_ ) 2 _n_ + _dn_
� �



1 1
_≥t_ [2] ( _C_ gap _/_ 2 _−_ _C_ ) ( _dp_ ) 2 + _d_ + _n_
� � � � 2



1

2 _[−]_ _C_ _[C]_



_C_ _[C]_ 0 _−_ ~~_√_~~ _CC_ 0



_C_
�� _−_ _C_ gap



� _nt_ [2] [�]



From the inequality above we can see, there exists large but uniform
choices of constants _C_ 0 _, C_ gap _>_ 0 such that the term above is nonnegative, additionally implies (1.90). In another word, there exists
_C_ 0 _, C_ gap _>_ 0, such that whenever (1.58) holds, then



(1.91) P � _σ_ min [2] � _U_ ˆ [1: _r,_ :] � _≥_ _[p]_ _n_ [ +] + _[ t]_ _t_ [2] _[/]_ [2] [2]



_≥_ 1 _−_ _C_ exp( _−cd ∧_ _p_ ) _._
�



2. Next we consider _σ_ max ( _U_ [ˆ] [( _r_ +1): _p,_ :] ). Suppose we randomly generate _R_ [˜] _∈_
O _n−r_ as a unitary matrix of ( _n −_ _r_ ) dimension, which is independent


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 41


of _L_ . Also, _R ∈_ O _n−r,p−r_ as the first _p −_ _r_ columns of _R_ [˜] . Clearly,

_I_ _r_
˜ _· L_ and _L_ have the same distribution.
_R_ [⊺]
� �



This implies

_I_ _r_ ˆ
˜ _·_ _U,_ [ˆ] and _U_ have the same distribution.
_R_ [⊺]
� �


When we focus on the [( _r_ + 1) : _d_ ]-th rows, we get


_R_ [⊺] _U_ [ˆ] [( _r_ +1): _n,_ :] and _U_ ˆ [( _r_ +1): _d,_ :] have the same distribution.


Thus, we can turn to consider _R_ [⊺] _U_ [ˆ] [( _r_ +1): _n,_ :] rather than _U_ [ˆ] [( _r_ +1): _p,_ :] .
_U_ Conditioning on firstˆ [( _r_ +1): _n,_ :] is a random matrix with spectral norm no more than 1. _r_ rows of _U_ [ˆ], i.e. _U_ [ˆ] [1: _r,_ :], the rest part of _U_ [ˆ], i.e.
Applying Lemma 10, we get for any given _U_ [ˆ] [1: _r,_ :] we have the following
conditioning probability when _n ≥_ _C_ 0 _d_ for some large _C_ 0,



_U_ ˆ

[1: _r,_ :]
�����



�



P



ˆ 2
_R_ ⊺ _U_ [( _r_ +1): _n,_ :] _≥_ _[p]_ [ +] _[ C]_ �
���
����



( _d −_ _r_ ) _p_ + _C_ ( _d −_ _r_ )


_n −_ _r_



_≤_ _C_ exp( _−cp_ ) _._



Therefore,
(1.92)



_≤_ _C_ exp( _−cp_ ) _._



P



ˆ 2
~~�~~
_U_ [( _r_ +1): _d,_ :] _≥_ _[p]_ [ +] _[ C]_
���
����



( _d −_ _r_ ) _p_ + _C_ ( _d −_ _r_ )



_n −_ _r_



�



Next, under essentially the same argument as the proof in Part 1, one
can show that there exists _C_ gap _, C_ 0 _>_ 0 such that whenever (1.58)
holds, we have



_p_ + _C_ ~~�~~



_r_ ) _p_ + _C_ ( _d −_ _r_ )

_≤_ _[p]_ [ +] _[ t]_ [2] _[/]_ [4]
_n −_ _r_ _n_ + _t_ [2]



( _d −_ _r_ ) _p_ + _C_ ( _d −_ _r_ )




_[,]_
_n_ + _t_ [2]



additionally we also have

(1.93) P _U_ ˆ [( _r_ +1): _d,_ :] 2 _≥_ _[p]_ [ +] _[ t]_ [2] _[/]_ [4]
���� ��� _n_ + _t_ [2]



_≤_ _C_ exp( _−cp_ ) _._
�



3. In this step we consider _∥U_ [ˆ] [( _r_ +1): _p,_ :] _P_ _U_ ˆ [1: _r,_ :] _∥_ . The idea to proceed is

similar to the part for _∥U_ [ˆ] [( _r_ +1): _p,_ :] _∥_ . Conditioning on _U_ [ˆ] [1: _r,_ :],


_U_ ˆ [( _r_ +1): _p,_ :] _P_ ˆ _U_ [1: _r,_ :] and _R_ [⊺] _U_ [ˆ] [( _r_ +1): _n,_ :] _P_ ˆ _U_ [1: _r,_ :] have the same distribution _._


42 T. T. CAI AND A. ZHANG


Also, _∥U_ [ˆ] [ _r_ +1: _n,_ :] _P_ ˆ _U_ [1: _r,_ :] _∥≤_ 1, rank( _U_ [ˆ] [ _r_ +1: _n,_ :] _P_ ˆ _U_ [1: _r,_ :] ) _≤_ _r_ . By Lemma 10,
there exists uniform constants _C, c >_ 0 such that



P ���� _U_ ˆ [( _r_ +1): _p,_ :] _P_ ˆ _U_ [1: _r,_ :]



1

_> C_ ( _d/n_ ) 2
��� �



(1.94) ˆ _r,_ 1

=P ���� _R_ ⊺ _U_ [( _r_ +1): _n,_ :] _P_ _U_ ˆ [1: _r,_ :] ��� _> C_ ( _d/n_ ) 2 � _≤_ _C_ exp( _−cd_ ) _._



=P ���� _R_ ⊺ _U_ ˆ [( _r_ +1): _n,_ :] _P_ _U_ ˆ [1: _r,_ :]



4. Combine (1.91), (1.93), (1.94), we know there exists _C_ gap _, C_ 0 _>_ 0 such
that whenever (1.58) holds, then with probability at least 1 _−C_ exp( _p∧_
_d_ ),

ˆ 2
_σ_ min [2] [( ˆ] _[U]_ [1: _r,_ :] [)] _[ >]_ _U_ [( _r_ +1): _d,_ :] _≥_ _σ_ _r_ [2] +1 [( ˆ] _[U]_ [1: _d,_ :] [)] _[.]_
��� ���



ˆ
_σ_ min ( _U_ [ˆ] [1: _r,_ :] ) _·_ ��� _U_ [( _r_ +1): _p,_ :] _P_ ˆ _U_ [1: _r,_ :]



��
�



_p_ + _t_ [2] _/_ 4



~~�~~



_pn_ ++ _t_ [2] _t/_ [2] 4 _[·][ C]_ [(] _[d/n]_ [)] 12



_p_ + _t_ [2] _/_ 2



(1.95)



~~�~~ ˆ ~~�~~ 2 _≤_
_σ_ min [2] [( ˆ] _[U]_ [[1:] _[r,]_ [:]] [)] _[ −]_ � _U_ [( _r_ +1): _d,_ :]
� ��



+ _t_ [2] _/_ 2

_n_ + _t_ [2] _[−]_ _[p]_ _n_ [+] + _[t]_ [2] _t_ _[/]_ [2] [4]



_n_ + _t_ [2]



~~�~~
_≤_ _[C]_



_t_ [2] _._



_d_ ( _p_ + _t_ [2] )(1 + _t_ [2] _/n_ )



By Proposition 1, we have finished the proof of Lemma 9.


Proof of Lemma 9.. Based on the setting, _U_ = _LV_ Σ _[−]_ [1] . Thus,



� _U_ _v_ � 2
� [Ω _,_ :] � 2

_σ_ max [2] [(] _[U]_ [Ω _,_ :] [) = max] _≤_ max
_v∈_ R _[p]_ _∥Uv∥_ [2] 2 _v∈_ R _[p]_



� _L_ _V_ Σ _−_ 1 _v_ � 2
� [Ω _,_ :] � 2
~~�~~ _L_ _V_ Σ _−_ 1 _v_ ~~�~~ 2 ~~�~~ _L_ _c_ _V_ Σ _−_ 1 _v_ ~~�~~ 2
� [Ω _,_ :] � 2 [+] � [Ω _,_ :] � 2



_σ_ max [2] [(] _[L]_ [Ω _,_ :] [)] �� _V_ Σ _−_ 1 _v_ �� 22
_≤_ max
_v∈_ R _[p]_ _σ_ max [2] ( _L_ [Ω _,_ :] ) _∥V_ Σ _[−]_ [1] _v∥_ [2] 2 [+] _[ σ]_ min [2] [(] _[L]_ [[Ω] _[c]_ _[,]_ [:]] [)] _[ ∥][V]_ [ Σ] _[−]_ [1] _[v][∥]_ 2 [2]

= _σ_ max [2] [(] _[L]_ [Ω _,_ :] [)]
_σ_ max [2] ( _L_ [Ω _,_ :] ) + _σ_ min [2] [(] _[L]_ [[Ω] _[c]_ _[,]_ [:]] [)] _[.]_


The other inequality in the lemma can be proved in the same way.


Proof of Lemma 10. Suppose _α_ = _∥X∥_ . Since left and right multiply
orthogonal matrix to _X_ does not essentially change the problem, without
loss of generality we can assume that _X ∈_ R _[n][×][m]_ is diagonal, such that
_X_ = diag( _σ_ 1 ( _X_ ) _, σ_ 2 ( _X_ ) _, · · ·, σ_ _p_ ( _X_ ) _,_ 0 _, · · ·_ ). Clearly _σ_ _i_ ( _X_ ) _≤_ _α_ . Now





������� _≤_ _α ·_ �� _R_ [1: _p,_ :] �� _._



_∥X_ [⊺] _R∥_ =



�������









_σ_ 1 ( _X_ ) _R_ 11 _· · ·_ _σ_ 1 ( _X_ ) _R_ 1 _d_

... ...
_σ_ _p_ ( _X_ ) _R_ _p_ 1 _· · ·_ _σ_ _p_ ( _X_ ) _R_ _pd_


PERTURBATION BOUNDS FOR SINGULAR SUBSPACES 43


In order to finish the proof, we only need to bound the spectral norm for
_∥R_ [1: _p,_ :] _∥_ . For any unit vector _v ∈_ R _[d]_, _Rv_ is randomly distributed on O _n,_ 1
with Haar measure. Thus
(1.96)

_∥R_ [1: _p,_ :] _v∥_ 2 [2] [has the same distribution as] _[x]_ 1 [2] [+] _[ · · ·]_ [ +] _[ x]_ [2] _p_ _,_ _x_ 1 _, · · ·, x_ _n_ _iid_ _∼_ _N_ (0 _,_ 1) _._
_x_ [2] 1 [+] _[ · · ·]_ [ +] _[ x]_ _n_ [2]


By the tail bound for _χ_ [2] -distribution,



_≤_ 2 exp( _−t_ ) _,_


_≤_ 2 exp( _−t_ ) _,_



�



P


P



�



_n −_ _√_

�



�



_p −_ ~~�~~



2 _nt ≤_



2 _pt ≤_



_p_
�



� _x_ [2] _k_ _[≤]_ _[p]_ [ +] ~~�~~


_k_ =1



2 _pt_ + 2 _t_



2 _nt_ + 2 _t_

�



_n_
� _x_ [2] _k_ _[≤]_ _[n]_ [ +] _√_


_k_ =1



which means



2 exp( _−t_ ) _≥_ P � _v_ [⊺] [�] _R_ [1: [⊺] _p,_ :] _[R]_ [[1:] _[p,]_ [:]] _[ −]_ _n_ _[p]_




_[√]_ ~~[2]~~ ~~_[p][t]_~~ [+ 2] _[t]_
_n_ _[p]_ _[I]_ _[r]_ � _v ≥_ _[p]_ [ +] 2 _nt_



_n_



_−_ _[p]_

2 _nt_ _n_



_n −_ ~~_√_~~



2 exp( _−t_ ) _≥_ P � _v_ [⊺] [�] _R_ [1: [⊺] _p,_ :] _[R]_ [[1:] _[p,]_ [:]] _[ −]_ _n_ _[p]_



_p −_ _[√]_ 2 _pt_
_n_ _[p]_ _[I]_ _[r]_ � _v ≤_ _n_ + ~~_√_~~ 2 _nt_



_p_

2 _nt_ + 2 _t_ _[−]_ _n_ _[p]_



_n_



_,_
�


_._
�



We set _t_ = _Cd_ for large enough _C >_ 0 and apply _ε_ -net method (Lemma 5),
the following result hold for true.



_C_ exp( _−cd_ ) _≥_ P _R_ 1:⊺ _d._ : _[R]_ [1:] _[d,]_ [:] _[ −]_ _[d]_
� [�] ��� _n_ _[I]_ _[r]_ ����



_p_ + _C√pd_ + _Cd_
_≥_ 3 max
� _n −_ _C_ ~~_√_~~ _nd_



_−_ _[p]_

_nd_ _n_




_[p]_ _p −_ _C_ _[√]_ _pd_

_n_ _[−]_ _n_ + _C_ ~~_√_~~ _nd_ +



_nd_ + _Cd_



_n −_ _C_ ~~_√_~~




_[p]_ _[p]_

_n_ _[,]_ _n_



_._
��



Note that



_p_ + _C√pd_ + _Cd_
max
� _n −_ _C_ ~~_√_~~ _nd_



�




_[p]_ _p −_ _C_ _[√]_ _pd_

_n_ _[−]_ _n_ + _C_ ~~_√_~~ _nd_ +



_nd_ + _Cd_



_n −_ _C_ ~~_√_~~



_−_ _[p]_

_nd_ _n_




_[p]_ _[p]_

_n_ _[,]_ _n_



_d/n_
�



_d/n_ + _pd/n_ + _[√]_ _pd_
�



_nd_ + _Cd_



_C_ � _√pd_ + _d_ + _p_ �



_≤_ max


_≤_ max










_C_ �2 _[√]_ _pd_ + _d_ �

_n −_ _C_ ~~_√_~~ _nd_

�



_n −_ _C_ ~~_√_~~



_nd_ + _Cd_



_n −_ _C_ ~~_√_~~



_p_ � _d/n_ � _C_ � _p_ ~~�~~

_nd_ _,_ _n_



_n_ + _C_ ~~_√_~~






 _[.]_








�



_d_ � _C_ �3 _[√]_ _pd_ �

_nd_ _,_ _n_ + _C_ ~~_√_~~ _nd_ +



_n_ + _C_ ~~_√_~~


44 T. T. CAI AND A. ZHANG


Thus there exists _C_ 0 _>_ 0 such that when _n > C_ 0 _r_, _n −_ _C_ _[√]_ ~~_nr_~~ _> n/_ 2, and
additionally,



1
_C_ ( _pd_ ) 2 + _d_
� �
_≤_ _._

_n_




_[d]_ _d −_ _C√_

_n_ _[−]_ _n_ + _C_ ~~_nr_~~



_d −_ _C√dr_


_n_ + _C_ ~~_[√]_~~ ~~_nr_~~ + _Cr_



�



max



_d_ + _C√_


_n −_

�



_C√dr_ + _Cr_

_−_ _[d]_
_n −_ _C_ ~~_[√]_~~ ~~_nr_~~ _n_




_[d]_ _[d]_

_n_ _[,]_ _n_



To sum up, we have finished the proof of Lemma 10.



Department of Statistics

The Wharton School

University of Pennsylvania
Philadelphia, PA, 19104.
[E-mail: tcai@wharton.upenn.edu](mailto:tcai@wharton.upenn.edu)
[URL: http://www-stat.wharton.upenn.edu/](http://www-stat.wharton.upenn.edu/{\raise .17ex\hbox {$\scriptstyle \sim $}}tcai/) _∼_ tcai/



Department of Statistics

University of Wisconsin-Madison

Madison, WI, 53706.
[E-mail: anruzhang@stat.upenn.edu](mailto:anruzhang@stat.upenn.edu)
[URL: http://www.stat.wisc.edu/](http://www.stat.wisc.edu/{\raise .17ex\hbox {$\scriptstyle \sim $}}anruzhang/) _∼_ anruzhang/


