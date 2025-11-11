Low-Rank Tensor Networks for

Dimensionality Reduction and Large-Scale
Optimization Problems: Perspectives and
Challenges PART 1 [1]


A. Cichocki N. Lee,
I.V. Oseledets, A-H. Phan,
Q. Zhao, D. Mandic


Andrzej CICHOCKI
RIKEN Brain Science Institute (BSI), Japan and SKOLTECH, Russia
cia@brain.riken.jp


Namgil LEE
RIKEN BSI, namgil.lee@riken.jp


Ivan OSELEDETS
Skolkovo Institute of Science and Technology (SKOLTECH), and
Institute of Numerical Mathematics of Russian Academy of Sciences,
Russia

i.oseledets@skolkovotech.ru


Anh-Huy PHAN
RIKEN BSI, phan@brain.riken.jp


Qibin ZHAO
RIKEN BSI, qbzhao@brain.riken.jp


Danilo P. MANDIC
Imperial College, UK
d.mandic@imperial.ac.uk


1 Copyright A.Cichocki _et al._ Please make reference to: A. Cichocki, N. Lee,
I. Oseledets, A.-H. Phan, Q. Zhao and D.P. Mandic (2016), “Tensor Networks for
Dimensionality Reduction and Large-scale Optimization: Part 1 Low-Rank Tensor
Decompositions”, Foundations and Trends in Machine Learning: Vol. 9: No. 4-5,
pp 249-429.


**Abstract**


Machine learning and data mining algorithms are becoming
increasingly important in analyzing large volume, multi-relational
and multi–modal datasets, which are often conveniently represented as
multiway arrays or tensors. It is therefore timely and valuable for the
multidisciplinary research community to review tensor decompositions
and tensor networks as emerging tools for large-scale data analysis and
data mining. We provide the mathematical and graphical representations
and interpretation of tensor networks, with the main focus on the
Tucker and Tensor Train (TT) decompositions and their extensions or
generalizations.
To make the material self-contained, we also address the concept of
tensorization which allows for the creation of very high-order tensors from
lower-order structured datasets represented by vectors or matrices. Then,
in order to combat the curse of dimensionality and possibly obtain linear
or even sub-linear complexity of storage and computation, we address
super-compression of tensor data through low-rank tensor networks.
Finally, we demonstrate how such approximations can be used to solve a
wide class of huge-scale linear/ multilinear dimensionality reduction and
related optimization problems that are far from being tractable when using
classical numerical methods.
The challenge for huge-scale optimization problems is therefore to
develop methods which scale linearly or sub-linearly (i.e., logarithmic
complexity) with the size of datasets, in order to benefit from the well–
understood optimization frameworks for smaller size problems. However,
most efficient optimization algorithms are convex and do not scale well
with data volume, while linearly scalable algorithms typically only apply
to very specific scenarios. In this review, we address this problem through
the concepts of low-rank tensor network approximations, distributed
tensor networks, and the associated learning algorithms. We then elucidate
how these concepts can be used to convert otherwise intractable huge-scale
optimization problems into a set of much smaller linked and/or distributed
sub-problems of affordable size and complexity. In doing so, we highlight
the ability of tensor networks to account for the couplings between the
multiple variables, and for multimodal, incomplete and noisy data.
The methods and approaches discussed in this work can be considered
both as an alternative and a complement to emerging methods for


1


huge-scale optimization, such as the random coordinate descent (RCD)
scheme, subgradient methods, alternating direction method of multipliers
(ADMM) methods, and proximal gradient descent methods. This is PART1
which consists of Sections 1-4.


Keywords: Tensor networks, Function-related tensors, CP
decomposition, Tucker models, tensor train (TT) decompositions,
matrix product states (MPS), matrix product operators (MPO), basic
tensor operations, multiway component analysis, multilinear blind
source separation, tensor completion, linear/ multilinear dimensionality
reduction, large-scale optimization problems, symmetric eigenvalue
decomposition (EVD), PCA/SVD, huge systems of linear equations,
pseudo-inverse of very large matrices, Lasso and Canonical Correlation
Analysis (CCA).


2


**Chapter 1**


**Introduction and Motivation**


This monograph aims to present a coherent account of ideas and
methodologies related to tensor decompositions (TDs) and tensor networks
models (TNs). Tensor decompositions (TDs) decompose principally data
tensors into factor matrices, while tensor networks (TNs) decompose
higher-order tensors into sparsely interconnected small-scale low-order
core tensors. These low-order core tensors are called “components”,
“blocks”, “factors” or simply “cores”. In this way, large-scale data can be
approximately represented in highly compressed and distributed formats.
In this monograph, the TDs and TNs are treated in a unified way,
by considering TDs as simple tensor networks or sub-networks; the
terms “tensor decompositions” and “tensor networks” will therefore be
used interchangeably. Tensor networks can be thought of as special
graph structures which break down high-order tensors into a set of
sparsely interconnected low-order core tensors, thus allowing for both
enhanced interpretation and computational advantages. Such an approach
is valuable in many application contexts which require the computation
of eigenvalues and the corresponding eigenvectors of extremely highdimensional linear or nonlinear operators. These operators typically
describe the coupling between many degrees of freedom within realworld physical systems; such degrees of freedom are often only weakly
coupled. Indeed, quantum physics provides evidence that couplings
between multiple data channels usually do not exist among all the
degrees of freedom but mostly locally, whereby “relevant” information,
of relatively low-dimensionality, is embedded into very large-dimensional
measurements [148,156,183,214].
Tensor networks offer a theoretical and computational framework for


3


the analysis of computationally prohibitive large volumes of data, by
“dissecting” such data into the “relevant” and “irrelevant” information,
both of lower dimensionality. In this way, tensor network representations
often allow for super-compression of datasets as large as 10 [50] entries, down
to the affordable levels of 10 [7] or even less entries [22,68,69,110,112,120,133,
161,215].
With the emergence of the big data paradigm, it is therefore both
timely and important to provide the multidisciplinary machine learning
and data analytic communities with a comprehensive overview of tensor
networks, together with an example-rich guidance on their application in
several generic optimization problems for huge-scale structured data. Our
aim is also to unify the terminology, notation, and algorithms for tensor
decompositions and tensor networks which are being developed not only
in machine learning, signal processing, numerical analysis and scientific
computing, but also in quantum physics/ chemistry for the representation
of, e.g., quantum many-body systems.


**1.1** **Challenges in Big Data Processing**


The volume and structural complexity of modern datasets are becoming
exceedingly high, to the extent which renders standard analysis methods
and algorithms inadequate. Apart from the huge Volume, the other
features which characterize big data include Veracity, Variety and Velocity
(see Figures 1.1(a) and (b)). Each of the “V features” represents a research
challenge in its own right. For example, high volume implies the need for
algorithms that are scalable; high Velocity requires the processing of big
data streams in near real-time; high Veracity calls for robust and predictive
algorithms for noisy, incomplete and/or inconsistent data; high Variety
demands the fusion of different data types, e.g., continuous, discrete,
binary, time series, images, video, text, probabilistic or multi-view. Some
applications give rise to additional “V challenges”, such as Visualization,
Variability and Value. The Value feature is particularly interesting and
refers to the extraction of high quality and consistent information, from
which meaningful and interpretable results can be obtained.
Owing to the increasingly affordable recording devices, extremescale volumes and variety of data are becoming ubiquitous across the
science and engineering disciplines. In the case of multimedia (speech,
video), remote sensing and medical / biological data, the analysis also
requires a paradigm shift in order to efficiently process massive datasets


4


(a)


(b)











**VOLUME**

































**VARIETY**


Figure 1.1: A framework for extremely large-scale data analysis. (a) The 4V
challenges for big data. (b) A unified framework for the 4V challenges and the
potential applications based on tensor decomposition approaches.

5


within tolerable time (velocity). Such massive datasets may have billions
of entries and are typically represented in the form of huge block
matrices and/or tensors. This has spurred a renewed interest in the
development of matrix / tensor algorithms that are suitable for very
large-scale datasets. We show that tensor networks provide a natural
sparse and distributed representation for big data, and address both
established and emerging methodologies for tensor-based representations
and optimization. Our particular focus is on low-rank tensor network
representations, which allow for huge data tensors to be approximated
(compressed) by interconnected low-order core tensors.


**1.2** **Tensor Notations and Graphical Representations**


Tensors are multi-dimensional generalizations of matrices. A matrix (2ndorder tensor) has two modes, rows and columns, while an _N_ th-order tensor
has _N_ modes (see Figures 1.2–1.7); for example, a 3rd-order tensor (with
three-modes) looks like a cube (see Figure 1.2). Subtensors are formed
when a subset of tensor indices is fixed. Of particular interest are _fibers_
which are vectors obtained by fixing every tensor index but one, and _matrix_
_slices_ which are two-dimensional sections (matrices) of a tensor, obtained
by fixing all the tensor indices but two. It should be noted that block
matrices can also be represented by tensors, as illustrated in Figure 1.3 for
4th-order tensors.
We adopt the notation whereby tensors (for _N_ ě 3) are denoted by
bold underlined capital letters, e.g., **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ . For simplicity, we
assume that all tensors are real-valued, but it is, of course, possible to define
tensors as complex-valued or over arbitrary fields. Matrices are denoted
by boldface capital letters, e.g., **X** P **R** _[I]_ [ˆ] _[J]_, and vectors (1st-order tensors)
by boldface lower case letters, e.g., **x** P **R** _[J]_ . For example, the columns of
the matrix **A** = [ **a** 1, **a** 2, . . ., **a** _R_ ] P **R** _[I]_ [ˆ] _[R]_ are the vectors denoted by **a** _r_ P **R** _[I]_,
while the elements of a matrix (scalars) are denoted by lowercase letters,
e.g., _a_ _ir_ = **A** ( _i_, _r_ ) (see Table 1.1).
A specific entry of an _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ is denoted
by _x_ _i_ 1, _i_ 2,..., _i_ _N_ = **X** ( _i_ 1, _i_ 2, . . ., _i_ _N_ ) P **R** . The order of a tensor is the number
of its “modes”, “ways” or “dimensions”, which can include space, time,
frequency, trials, classes, and dictionaries. The term _‘_ ‘size” stands for
the number of values that an index can take in a particular mode. For
example, the tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ is of order _N_ and size _I_ _n_ in all modes- _n_
( _n_ = 1, 2, . . ., _N_ ) . Lower-case letters e.g, _i_, _j_ are used for the subscripts in


6


Horizontal Slices Lateral Slices Frontal Slices


**X** [(] _[i]_ [,:,:)] **X** [(:, ] _[j]_ [,:)]





Column (Mode-1) Row (Mode-2) Tube (Mode-3)
Fibers Fibers Fibers



**X**



(1,3,:)



**X**



(1,:,3)



**X**



(:,3,1)



Figure 1.2: A 3rd-order tensor **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_, with entries _x_ _i_, _j_, _k_ = **X** ( _i_, _j_, _k_ ), and
its subtensors: slices (middle) and fibers (bottom). All fibers are treated as
column vectors.


running indices and capital letters _I_, _J_ denote the upper bound of an index,
i.e., _i_ = 1, 2, . . ., _I_ and _j_ = 1, 2, . . ., _J_ . For a positive integer _n_, the shorthand
notation ă _n_ ą denotes the set of indices t1, 2, . . ., _n_ u.
Notations and terminology used for tensors and tensor networks differ
across the scientific communities (see Table 1.2); to this end we employ
a unifying notation particularly suitable for machine learning and signal
processing research, which is summarized in Table 1.1.
Even with the above notation conventions, a precise description of
tensors and tensor operations is often tedious and cumbersome, given


7


Table 1.1: Basic matrix/tensor notation and symbols.


**X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ _N_ th-order tensor of size _I_ 1 ˆ _I_ 2 ˆ ¨ ¨ ¨ ˆ _I_ _N_


_x_ _i_ 1, _i_ 2,..., _i_ _N_ = **X** ( _i_ 1, _i_ 2, . . ., _i_ _N_ ) ( _i_ 1, _i_ 2, . . ., _i_ _N_ ) th entry of **X**


_x_, **x**, **X** scalar, vector and matrix


**G**, **S**, **G** [(] _[n]_ [)], **X** [(] _[n]_ [)] core tensors


_N_ th-order diagonal core tensor with
**Λ** P **R** _[R]_ [ˆ] _[R]_ [ˆ¨¨¨ˆ] _[R]_
nonzero entries _λ_ _r_ on the main
diagonal


transpose, inverse and Moore–
**A** [T], **A** [´][1], **A** [:] Penrose pseudo-inverse of a matrix
**A**


**A** = [ **a** 1, **a** 2, . . ., **a** _R_ ] P **R** _[I]_ [ˆ] _[R]_ [ matrix with] _[ R]_ [ column vectors] **[ a]** _r_ [P]
**R** _[I]_, with entries _a_ _ir_


**A**, **B**, **C**, **A** [(] _[n]_ [)], **B** [(] _[n]_ [)], **U** [(] _[n]_ [)] component (factor) matrices


**X** ( _n_ ) P **R** _[I]_ _[n]_ [ˆ] _[I]_ [1] [¨¨¨] _[I]_ _[n]_ [´][1] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ mode- _n_ matricization of **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_


**X** ă _n_ ą P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ mode-(1, . . ., _n_ ) matricization of **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_


**X** ( :, _i_ 2, _i_ 3, . . ., _i_ _N_ ) P **R** _[I]_ [1] mode-1 fiber of a tensor **X** obtained by
fixing all indices but one (a vector)


**X** ( :, :, _i_ 3, . . ., _i_ _N_ ) P **R** _[I]_ [1] [ˆ] _[I]_ [2] slice (matrix) of a tensor **X** obtained
by fixing all indices but two


**X** ( :, :, :, _i_ 4, . . ., _i_ _N_ ) subtensor of **X**, obtained by fixing
several indices


_R_, ( _R_ 1, . . ., _R_ _N_ ) tensor rank _R_ and multilinear rank



˝, d, b


b _L_, |b|



outer, Khatri–Rao, Kronecker products


Left Kronecker, strong Kronecker products



**x** = vec ( **X** ) vectorization of **X**


tr ( ‚ ) trace of a square matrix


diag ( ‚ ) diagonal matrix


8


Table 1.2: Terminology used for tensor networks across the machine
learning / scientific computing and quantum physics / chemistry
communities.

|Machine Learning|Quantum Physics|
|---|---|
|_N_th-order tensor<br>high/low-order tensor<br>ranks of TNs<br>unfolding, matricization<br>tensorization<br>core<br>variables<br>ALS Algorithm<br>MALS Algorithm<br>column vector** x** P** R**_I_ˆ1<br>row vector** x**T P** R**1ˆ_I_<br>inner product x**x**,** x**y =** x**T**x** <br>Tensor Train (TT)<br>Tensor Chain (TC)<br>Matrix TT<br>Hierarchical Tucker (HT)|rank-_N_ tensor<br>tensor of high/low dimension<br>bond dimensions of TNs<br>grouping of indices<br>splitting of indices<br>site<br>open (physical) indices<br>one-site DMRG or DMRG1<br>two-site DMRG or DMRG2<br>ket |Ψy<br>bra xΨ|<br>xΨ|Ψy<br>Matrix Product State (MPS) (with Open<br>Boundary Conditions (OBC))<br>MPS with Periodic Boundary Conditions<br>(PBC)<br>Matrix Product Operators (with OBC)<br>Tree Tensor Network State (TTNS) with<br>rank-3 tensors|



9


|G<br>11|G<br>12|
|---|---|
|**G**21|**G**22|



Figure 1.3: A block matrix and its representation as a 4th-order tensor,
created by reshaping (or a projection) of blocks in the rows into lateral slices
of 3rd-order tensors.


|...|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||





|Scalar|Vector|Matrix|3rd-order Tensor|4th-order Tensor<br>...|
|---|---|---|---|---|
|...<br>One-way|...<br>2-way|...<br>3-way|...<br>4-way|...<br>...<br>...<br>...<br>...<br>...<br>...<br>5-way|
|Univariate||Multivariate<br>Multiway Analysis (High-order tensors)|Multivariate<br>Multiway Analysis (High-order tensors)|Multivariate<br>Multiway Analysis (High-order tensors)|


Figure 1.4: Graphical representation of multiway array (tensor) data of
increasing structural complexity and “Volume” (see [155] for more detail).


the multitude of indices involved. To this end, in this monograph,
we grossly simplify the description of tensors and their mathematical
operations through diagrammatic representations borrowed from physics
and quantum chemistry (see [156] and references therein). In this way,
tensors are represented graphically by nodes of any geometrical shapes
(e.g., circles, squares, dots), while each outgoing line (“edge”, “leg”,“arm”)
from a node represents the indices of a specific mode (see Figure 1.5(a)).
In our adopted notation, each scalar (zero-order tensor), vector (first-order


10


(a)


(b)































=

_J_ _I_





_I_ _J_


|Col1|A|x|
|---|---|---|
||||
||||


|Col1|b=A|
|---|---|
|||
|||



=

_J_ _K_ _I_



**C** = **AB**





_I_ _J_



_K_ _I_ _K_


|Col1|A|B|Col4|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|
|---|---|---|
||||















=



_J_



_L_



_L_



_J_



_K_

Σ

_k_ =1 _[a]_ _[i,j,k]_ _[ b]_ _[k,l,m,p]_ [ =] _[ c]_ _[i,j,l,m,p]_


Figure 1.5: Graphical representation of tensor manipulations. (a) Basic
building blocks for tensor network diagrams. (b) Tensor network diagrams
for matrix-vector multiplication (top), matrix by matrix multiplication
(middle) and contraction of two tensors (bottom). The order of reading
of indices is anti-clockwise, from the left position.


tensor), matrix (2nd-order tensor), 3rd-order tensor or higher-order tensor
is represented by a circle (or rectangular), while the order of a tensor is
determined by the number of lines (edges) connected to it. According
to this notation, an _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ is represented by a
circle (or any shape) with _N_ branches each of size _I_ _n_, _n_ = 1, 2, . . ., _N_ (see
Section 2). An interconnection between two circles designates a contraction


11


4th-order tensor







5th-order tensors



|6th-order tensor|Col2|
|---|---|
|=|=|
|=||
|=||
|=||
|||
|||
|||


Figure 1.6: Graphical representations and symbols for higher-order block
tensors. Each block represents either a 3rd-order tensor or a 2nd-order
tensor. The outer circle indicates a global structure of the block tensor (e.g.
a vector, a matrix, a 3rd-order block tensor), while the inner circle reflects
the structure of each element within the block tensor. For example, in the
top diagram a vector of 3rd order tensors is represented by an outer circle
with one edge (a vector) which surrounds an inner circle with three edges (a
3rd order tensor), so that the whole structure designates a 4th-order tensor.


of tensors, which is a summation of products over a common index (see
Figure 1.5(b) and Section 2).
Block tensors, where each entry (e.g., of a matrix or a vector) is an
individual subtensor, can be represented in a similar graphical form, as
illustrated in Figure 1.6. Hierarchical (multilevel block) matrices are also
naturally represented by tensors and vice versa, as illustrated in Figure 1.7
for 4th-, 5th- and 6th-order tensors. All mathematical operations on tensors
can be therefore equally performed on block matrices.
In this monograph, we make extensive use of tensor network
diagrams as an intuitive and visual way to efficiently represent tensor
decompositions. Such graphical notations are of great help in studying and
implementing sophisticated tensor operations. We highlight the significant


12


(a)


(b)


(c)




















|Col1|Col2|Col3|...|Col5|
|---|---|---|---|---|
||||||







Figure 1.7: Hierarchical matrix structures and their symbolic representation
as tensors. (a) A 4th-order tensor representation for a block matrix **X** P
**R** _[R]_ [1] _[I]_ [1] [ˆ] _[R]_ [2] _[I]_ [2] (a matrix of matrices), which comprises block matrices **X** _r_ 1, _r_ 2 P
**R** _[I]_ [1] [ˆ] _[I]_ [2] . (b) A 5th-order tensor. (c) A 6th-order tensor.


advantages of such diagrammatic notations in the description of tensor
manipulations, and show that most tensor operations can be visualized
through changes in the architecture of a tensor network diagram.


13


**1.3** **Curse** **of** **Dimensionality** **and** **Generalized**
**Separation of Variables for Multivariate Functions**


**1.3.1** **Curse of Dimensionality**


The term _curse of dimensionality_ was coined by [18] to indicate that the
number of samples needed to estimate an arbitrary function with a given
level of accuracy grows exponentially with the number of variables, that
is, with the dimensionality of the function. In a general context of
machine learning and the underlying optimization problems, the “curse
of dimensionality” may also refer to an exponentially increasing number
of parameters required to describe the data/system or an extremely large
number of degrees of freedom. The term “curse of dimensionality”, in
the context of tensors, refers to the phenomenon whereby the number
of elements, _I_ _[N]_, of an _N_ th-order tensor of size ( _I_ ˆ _I_ ˆ ¨ ¨ ¨ ˆ _I_ ) grows
exponentially with the tensor order, _N_ . Tensor volume can therefore
easily become prohibitively big for multiway arrays for which the
number of dimensions (“ways” or “modes”) is very high, thus requiring
enormous computational and memory resources to process such data.
The understanding and handling of the inherent dependencies among the
excessive degrees of freedom create both difficult to solve problems and
fascinating new opportunities, but comes at a price of reduced accuracy,
owing to the necessity to involve various approximations.
We show that the curse of dimensionality can be alleviated or even fully
dealt with through tensor network representations; these naturally cater
for the excessive volume, veracity and variety of data (see Figure 1.1) and
are supported by efficient tensor decomposition algorithms which involve
relatively simple mathematical operations. Another desirable aspect of
tensor networks is their relatively small-scale and low-order _core tensors_,
which act as “building blocks” of tensor networks. These core tensors are
relatively easy to handle and visualize, and enable super-compression of
the raw, incomplete, and noisy huge-scale datasets. This also suggests a
solution to a more general quest for new technologies for processing of
exceedingly large datasets within affordable computation times.
To address the curse of dimensionality, this work mostly focuses
on approximative low-rank representations of tensors, the so-called
low-rank tensor approximations (LRTA) or low-rank tensor network
decompositions.


14


**1.4** **Separation of Variables and Tensor Formats**


A tensor is said to be in a _full format_ when it is represented as an original
(raw) multidimensional array [118], however, distributed storage and
processing of high-order tensors in their full format is infeasible due to the
curse of dimensionality. The _sparse format_ is a variant of the full tensor
format which stores only the nonzero entries of a tensor, and is used
extensively in software tools such as the Tensor Toolbox [8] and in the
sparse grid approach [25,80,91].
As already mentioned, the problem of huge dimensionality can be
alleviated through various distributed and compressed tensor network
formats, achieved by low-rank tensor network approximations. The
underpinning idea is that by employing tensor networks formats, both
computational costs and storage requirements may be dramatically
reduced through distributed storage and computing resources. It is
important to note that, except for very special data structures, a tensor
cannot be compressed without incurring some compression error, since
a low-rank tensor representation is only an approximation of the original

tensor.
The concept of compression of multidimensional large-scale data
by tensor network decompositions can be intuitively explained as
follows. Consider the approximation of an _N_ -variate function _f_ ( **x** ) =
_f_ ( _x_ 1, _x_ 2, . . ., _x_ _N_ ) by a finite sum of products of individual functions, each
depending on only one or a very few variables [16, 34, 67, 206]. In the
simplest scenario, the function _f_ ( **x** ) can be (approximately) represented in
the following separable form


_f_ ( _x_ 1, _x_ 2, . . ., _x_ _N_ ) – _f_ [(] [1] [)] ( _x_ 1 ) _f_ [(] [2] [)] ( _x_ 2 ) ¨ ¨ ¨ _f_ [(] _[N]_ [)] ( _x_ _N_ ) . (1.1)


In practice, when an _N_ -variate function _f_ ( **x** ) is discretized into an _N_ thorder array, or a tensor, the approximation in (1.1) then corresponds to
the representation by rank-1 tensors, also called elementary tensors (see
Section 2). Observe that with _I_ _n_, _n_ = 1, 2, . . ., _N_ denoting the size of
each mode and _I_ = max _n_ t _I_ _n_ u, the memory requirement to store such
a full tensor is [ś] _n_ _[N]_ = 1 _[I]_ _[n]_ [ ď] _[ I]_ _[N]_ [, which grows exponentially with] _[ N]_ [.] On
the other hand, the separable representation in (1.1) is completely defined
by its factors, _f_ [(] _[n]_ [)] ( _x_ _n_ ), ( _n_ = 1, 2, . . ., _N_ ), and requires only [ř] _n_ _[N]_ = 1 _[I]_ _[n]_ [ !]
_I_ _[N]_ storage units. If _x_ 1, _x_ 2, . . ., _x_ _N_ are statistically independent random
variables, their joint probability density function is equal to the product
of marginal probabilities, _f_ ( **x** ) = _f_ [(] [1] [)] ( _x_ 1 ) _f_ [(] [2] [)] ( _x_ 2 ) ¨ ¨ ¨ _f_ [(] _[N]_ [)] ( _x_ _N_ ), in an exact


15


analogy to outer products of elementary tensors. Unfortunately, the form
of separability in (1.1) is rather rare in practice.
The concept of tensor networks rests upon generalized (full or partial)
separability of the variables of a high dimensional function. This can be
achieved in different tensor formats, including:


_•_
The Canonical Polyadic (CP) format (see Section 3.2), where



_f_ ( _x_ 1, _x_ 2, . . ., _x_ _N_ ) –



_R_
ÿ _f_ _r_ [(] [1] [)] ( _x_ 1 ) _f_ _r_ [(] [2] [)] ( _x_ 2 ) ¨ ¨ ¨ _f_ _r_ [(] _[N]_ [)] ( _x_ _N_ ), (1.2)

_r_ = 1



in an exact analogy to (1.1). In a discretized form, the above CP format
can be written as an _N_ th-order tensor



**F** –



_R_
ÿ **f** [(] _r_ [1] [)] [˝] **[ f]** [(] _r_ [2] [)] [˝ ¨ ¨ ¨ ˝] **[ f]** [(] _r_ _[N]_ [)] P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, (1.3)

_r_ = 1



where **f** [(] _r_ _[n]_ [)] P **R** _[I]_ _[n]_ denotes a discretized version of the univariate
function _f_ _r_ [(] _[n]_ [)] ( _x_ _n_ ), symbol ˝ denotes the outer product, and _R_ is the
tensor rank.


_•_
The Tucker format, given by



_R_ _N_
ÿ _g_ _r_ 1,..., _r_ _N_ _f_ _r_ [(] 1 [1] [)] [(] _[x]_ 1 [)] [ ¨ ¨ ¨] _[ f]_ [ (] _r_ _N_ _[N]_ [)] [(] _[x]_ _N_ [)] [,] (1.4)

_r_ _N_ = 1



_f_ ( _x_ 1, . . ., _x_ _N_ ) –



_R_ 1


¨ ¨ ¨

ÿ

_r_ 1 = 1



and its distributed tensor network variants (see Section 3.3),


_•_ The Tensor Train (TT) format (see Section 4.1), in the form



_R_ _N_ ´1
ÿ _f_ _r_ [(] 1 [1] [)] [(] _[x]_ 1 [)] _[ f]_ [ (] _r_ 1 [2] _r_ [)] 2 [(] _[x]_ 2 [)] [ ¨ ¨ ¨]

_r_ _N_ ´1 = 1



_f_ ( _x_ 1, _x_ 2, . . ., _x_ _N_ ) –



_R_ 1
ÿ

_r_ 1 = 1



_R_ 2


¨ ¨ ¨

ÿ

_r_ 2 = 1



¨ ¨ ¨ _f_ _r_ [(] _N_ _[N]_ ´ [´] 2 _r_ [2] [)] _N_ ´1 [(] _[x]_ _N_ ´1 [)] _[ f]_ [ (] _r_ _N_ _[N]_ ´ [)] 1 [(] _[x]_ _N_ [)] [,] (1.5)


with the equivalent compact matrix representation


_f_ ( _x_ 1, _x_ 2, . . ., _x_ _N_ ) – **F** [(] [1] [)] ( _x_ 1 ) **F** [(] [2] [)] ( _x_ 2 ) ¨ ¨ ¨ **F** [(] _[N]_ [)] ( _x_ _N_ ), (1.6)


where **F** [(] _[n]_ [)] ( _x_ _n_ ) P **R** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_, with _R_ 0 = _R_ _N_ = 1.


16


_•_ The Hierarchical Tucker (HT) format (also known as the Hierarchical
Tensor format) can be expressed via a hierarchy of nested separations
in the following way. Consider nested nonempty disjoint subsets _u_,
_v_, and _t_ = _u_ Y _v_ Ă t1, 2, . . ., _N_ u, then for some 1 ď _N_ 0 ă _N_, with
_u_ 0 = t1, . . ., _N_ 0 u and _v_ 0 = t _N_ 0 + 1, . . ., _N_ u, the HT format can be
expressed as



_R_ _v_ 0
ÿ _g_ _r_ [(] _u_ [12] 0, [¨¨¨] _r_ _v_ _[N]_ 0 [)] _f_ _r_ [(] _u_ _[u]_ 0 [0] [)] [(] **[x]** _u_ 0 [)] _[ f]_ [ (] _r_ _v_ _[v]_ 0 [0] [)] [(] **[x]** _v_ 0 [)] [,]

_r_ _v_ 0 = 1



_f_ ( _x_ 1, . . ., _x_ _N_ ) –


_f_ _r_ [(] _t_ _[t]_ [)] [(] **[x]** _t_ [)] –



_R_ _u_ 0
ÿ

_r_ _u_ 0 = 1



_R_ _u_
ÿ

_r_ _u_ = 1



_R_ _v_
ÿ _g_ _r_ [(] _u_ _[t]_ [)], _r_ _v_, _r_ _t_ _[f]_ [ (] _r_ _u_ _[u]_ [)] [(] **[x]** _u_ [)] _[ f]_ [ (] _r_ _v_ _[v]_ [)] [(] **[x]** _v_ [)] [,]

_r_ _v_ = 1



where **x** _t_ = t _x_ _i_ : _i_ P _t_ u. See Section 2.3 for more detail.


**Example.** In a particular case for _N_ =4, the HT format can be
expressed by



_f_ ( _x_ 1, _x_ 2, _x_ 3, _x_ 4 ) –


_f_ _r_ [(] 12 [12] [)] [(] _[x]_ 1 [,] _[ x]_ 2 [)] –


_f_ _r_ [(] 34 [34] [)] [(] _[x]_ 3 [,] _[ x]_ 4 [)] –



_R_ 12 _R_ 34
ÿ ÿ _g_ _r_ [(] 12 [1234], _r_ 34 [)] _[f]_ [ (] _r_ 12 [12] [)] [(] _[x]_ 1 [,] _[ x]_ 2 [)] _[ f]_ [ (] _r_ 34 [34] [)] [(] _[x]_ 3 [,] _[ x]_ 4 [)] [,]

_r_ 12 = 1 _r_ 34 = 1



_R_ 1
ÿ

_r_ 1 = 1


_R_ 3
ÿ

_r_ 3 = 1



_R_ 2
ÿ _g_ _r_ [(] 1 [12], _r_ [)] 2, _r_ 12 _[f]_ [ (] _r_ 1 [1] [)] [(] _[x]_ 1 [)] _[ f]_ [ (] _r_ 2 [2] [)] [(] _[x]_ 2 [)] [,]

_r_ 2 = 1


_R_ 4
ÿ _g_ _r_ [(] 3 [34], _r_ [)] 4, _r_ 34 _[f]_ [ (] _r_ 3 [3] [)] [(] _[x]_ 3 [)] _[ f]_ [ (] _r_ 4 [4] [)] [(] _[x]_ 4 [)] [.]

_r_ 4 = 1



The Tree Tensor Network States (TTNS) format, which is an extension
of the HT format, can be obtained by generalizing the two subsets,
_u_, _v_, into a larger number of disjoint subsets _u_ 1, . . ., _u_ _m_, _m_ ě 2. In
other words, the TTNS can be obtained by more flexible separations
of variables through products of larger numbers of functions at each
hierarchical level (see Section 2.3 for graphical illustrations and more
detail).


All the above approximations adopt the form of “sum-of-products” of
single-dimensional functions, a procedure which plays a key role in all
tensor factorizations and decompositions.
Indeed, in many applications based on multivariate functions, very
good approximations are obtained with a surprisingly small number


17


of factors; this number corresponds to the tensor rank, _R_, or tensor
network ranks, t _R_ 1, _R_ 2, . . ., _R_ _N_ u (if the representations are exact and
minimal). However, for some specific cases this approach may fail to obtain
sufficiently good low-rank TN approximations. The concept of generalized
separability has already been explored in numerical methods for highdimensional density function equations [34, 133, 206] and within a variety
of huge-scale optimization problems (see Part 2 of this monograph).
To illustrate how tensor decompositions address excessive volumes of
data, if all computations are performed on a CP tensor format in (1.3) and
not on the raw _N_ th-order data tensor itself, then instead of the original,
_exponentially growing_, data dimensionality of _I_ _[N]_, the number of parameters
in a CP representation reduces to _NIR_, which _scales linearly_ in the tensor
order _N_ and size _I_ (see Table 4.4). For example, the discretization of a
5-variate function over 100 sample points on each axis would yield the

=
difficulty to manage 100 [5] 10, 000, 000, 000 sample points, while a rank-2
CP representation would require only 5 ˆ 2 ˆ 100 = 1000 sample points.
Although the CP format in (1.2) effectively bypasses the curse of
dimensionality, the CP approximation may involve numerical problems
for very high-order tensors, which in addition to the intrinsic uncloseness
of the CP format (i.e., difficulty to arrive at a canonical format), the
corresponding algorithms for CP decompositions are often ill-posed [63].
As a remedy, greedy approaches may be considered which, for enhanced
stability, perform consecutive rank-1 corrections [135]. On the other hand,
many efficient and stable algorithms exist for the more flexible Tucker
format in (1.4), however, this format is not practical for tensor orders _N_ ą 5
because the number of entries of both the original data tensor and the core
tensor (expressed in (1.4) by elements _g_ _r_ 1, _r_ 2,..., _r_ _N_ ) scales exponentially in the
tensor order _N_ (curse of dimensionality).
In contrast to CP decomposition algorithms, TT tensor network formats
in (1.5) exhibit both very good numerical properties and the ability
to control the error of approximation, so that a desired accuracy of
approximation is obtained relatively easily. The main advantage of the
TT format over the CP decomposition is the ability to provide stable
quasi-optimal rank reduction, achieved through, for example, truncated
singular value decompositions (tSVD) or adaptive cross-approximation

[16, 116, 162]. This makes the TT format one of the most stable and simple
approaches to separate latent variables in a sophisticated way, while the
associated TT decomposition algorithms provide full control over low-rank


18


.
TN approximations [1] In this monograph, we therefore make extensive
use of the TT format for low-rank TN approximations and employ the TT
toolbox software for efficient implementations [160]. The TT format will
also serve as a basic prototype for high-order tensor representations, while
we also consider the Hierarchical Tucker (HT) and the Tree Tensor Network
States (TTNS) formats (having more general tree-like structures) whenever
advantageous in applications.
Furthermore, we address in depth the concept of tensorization
of structured vectors and matrices to convert a wide class of hugescale optimization problems into much smaller-scale interconnected
optimization sub-problems which can be solved by existing optimization
methods (see Part 2 of this monograph).
The tensor network optimization framework is therefore performed
through the two main steps:


_•_
Tensorization of data vectors and matrices into a high-order tensor,
followed by a distributed approximate representation of a cost
function in a specific low-rank tensor network format.


_•_
Execution of all computations and analysis in tensor network formats
(i.e., using only core tensors) that scale linearly, or even sub-linearly
(quantized tensor networks), in the tensor order _N_ . This yields
both the reduced computational complexity and distributed memory
requirements.


**1.5** **Advantages of Multiway Analysis via Tensor**
**Networks**


In this monograph, we focus on two main challenges in huge-scale data
analysis which are addressed by tensor networks: (i) an approximate
representation of a specific cost (objective) function by a tensor network
while maintaining the desired accuracy of approximation, and (ii) the
extraction of physically meaningful latent variables from data in a
sufficiently accurate and computationally affordable way. The benefits of
multiway (tensor) analysis methods for large-scale datasets then include:


1 Although similar approaches have been known in quantum physics for a long time,
their rigorous mathematical analysis is still a work in progress (see [156,158] and references
therein).


19


_•_
Ability to perform all mathematical operations in tractable tensor
network formats;


_•_
Simultaneous and flexible distributed representations of both the
structurally rich data and complex optimization tasks;


_•_ Efficient compressed formats of large multidimensional data
achieved via tensorization and low-rank tensor decompositions into
low-order factor matrices and/or core tensors;


_•_
Ability to operate with noisy and missing data by virtue of numerical
stability and robustness to noise of low-rank tensor / matrix
approximation algorithms;


_•_ A flexible framework which naturally incorporates various
diversities and constraints, thus seamlessly extending the standard,
flat view, Component Analysis (2-way CA) methods to multiway
component analysis;


_•_
Possibility to analyze linked (coupled) blocks of large-scale matrices
and tensors in order to separate common / correlated from
independent / uncorrelated components in the observed raw data;


_•_
Graphical representations of tensor networks allow us to express
mathematical operations on tensors (e.g., tensor contractions and
reshaping) in a simple and intuitive way, and without the explicit use
of complex mathematical expressions.


In that sense, this monograph both reviews current research in this area
and complements optimisation methods, such as the Alternating Direction
Method of Multipliers (ADMM) [23].
Tensor decompositions (TDs) have been already adopted in widely
diverse disciplines, including psychometrics, chemometrics, biometric,
quantum physics / information, quantum chemistry, signal and image
processing, machine learning, and brain science [42, 43, 79, 91, 119, 124,
190, 202]. This is largely due to their advantages in the analysis of data
that exhibit not only large volume but also very high variety (see Figure
1.1), as in the case in bio- and neuroinformatics and in computational
neuroscience, where various forms of data collection include sparse tabular
structures and graphs or hyper-graphs.
Moreover, tensor networks have the ability to efficiently
parameterize, through structured compact representations, very


20


general high-dimensional spaces which arise in modern applications

[19, 39, 50, 116, 121, 136, 229]. Tensor networks also naturally account
for intrinsic multidimensional and distributed patterns present in data,
and thus provide the opportunity to develop very sophisticated models
for capturing multiple interactions and couplings in data – these are
more physically insightful and interpretable than standard pair-wise
interactions.


**1.6** **Scope and Objectives**


Review and tutorial papers [7, 42, 54, 87, 119, 137, 163, 189] and books

[43, 91, 124, 190] dealing with TDs and TNs already exist, however, they
typically focus on standard models, with no explicit links to very largescale data processing topics or connections to a wide class of optimization
problems. The aim of this monograph is therefore to extend beyond the
standard Tucker and CP tensor decompositions, and to demonstrate the
perspective of TNs in extremely large-scale data analytics, together with
their role as a mathematical backbone in the discovery of hidden structures
in prohibitively large-scale data. Indeed, we show that TN models provide
a framework for the analysis of linked (coupled) blocks of tensors with
millions and even billions of non-zero entries.
We also demonstrate that TNs provide natural extensions of 2way (matrix) Component Analysis (2-way CA) methods to multi-way
component analysis (MWCA), which deals with the extraction of desired
components from multidimensional and multimodal data. This paradigm
shift requires new models and associated algorithms capable of identifying
core relations among the different tensor modes, while guaranteeing linear

.
/ sub-linear scaling with the size of datasets [2]
Furthermore, we review tensor decompositions and the associated
algorithms for very large-scale linear / multilinear dimensionality
reduction problems. The related optimization problems often involve
structured matrices and vectors with over a billion entries (see [67, 81, 87]
and references therein). In particular, we focus on Symmetric Eigenvalue
Decomposition (EVD/PCA) and Generalized Eigenvalue Decomposition
(GEVD) [70, 120, 123], SVD [127], solutions of overdetermined and
undetermined systems of linear algebraic equations [71, 159], the Moore–
Penrose pseudo-inverse of structured matrices [129], and Lasso problems


2 Usually, we assume that huge-scale problems operate on at least 10 7 parameters.


21


[130]. Tensor networks for extremely large-scale multi-block (multiview) data are also discussed, especially TN models for orthogonal
Canonical Correlation Analysis (CCA) and related Partial Least Squares
(PLS) problems. For convenience, all these problems are reformulated
as constrained optimization problems which are then, by virtue of lowrank tensor networks reduced to manageable lower-scale optimization subproblems. The enhanced tractability and scalability is achieved through
tensor network contractions and other tensor network transformations.
The methods and approaches discussed in this work can be considered
a both an alternative and complementary to other emerging methods
for huge-scale optimization problems like random coordinate descent
(RCD) scheme [150,180], sub-gradient methods [151], alternating direction
method of multipliers (ADMM) [23], and proximal gradient descent
methods [165] (see also [30,98] and references therein).
This monograph systematically introduces TN models and the
associated algorithms for TNs/TDs and illustrates many potential
applications of TDs/TNS. The dimensionality reduction and optimization
frameworks (see Part 2 of this monograph) are considered in detail, and we
also illustrate the use of TNs in other challenging problems for huge-scale
datasets which can be solved using the tensor network approach, including
anomaly detection, tensor completion, compressed sensing, clustering, and
classification.


22


**Chapter 2**


**Tensor Operations and Tensor**
**Network Diagrams**


Tensor operations benefit from the power of multilinear algebra which
is structurally much richer than linear algebra, and even some basic
properties, such as the rank, have a more complex meaning. We next
introduce the background on fundamental mathematical operations in
multilinear algebra, a prerequisite for the understanding of higher-order
tensor decompositions. A unified account of both the definitions and
properties of tensor network operations is provided, including the outer,
multi-linear, Kronecker, and Khatri–Rao products. For clarity, graphical
illustrations are provided, together with an example rich guidance for
tensor network operations and their properties. To avoid any confusion
that may arise given the numerous options on tensor reshaping, both
mathematical operations and their properties are expressed directly in their
native multilinear contexts, supported by graphical visualizations.


**2.1** **Basic Multilinear Operations**


The following symbols are used for most common tensor multiplications:
b for the Kronecker product, d for the Khatri–Rao product, f for the
Hadamard (componentwise) product, ˝ for the outer product and ˆ _n_ for
the mode- _n_ product. Basic tensor operations are summarized in Table 2.1,
and illustrated in Figures 2.1–2.13. We refer to [43,119,128] for more detail
regarding the basic notations and tensor operations. For convenience,

¨ ¨
general operations, such as vec ( ) or diag ( ), are defined similarly to the
MATLAB syntax.


23


Table 2.1: Basic tensor/matrix operations.


Mode- _n_ product of a tensor **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_
**C** = **A** ˆ _n_ **B** and a matrix **B** P **R** _[J]_ [ˆ] _[I]_ _[n]_ yields a tensor
**C** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] _[J]_ [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with entries
_c_ _i_ 1,..., _i_ _n_ ´1, _j_, _i_ _n_ + 1,..., _i_ _N_ = [ř] _i_ _[I]_ _n_ _[n]_ = 1 _[a]_ _[i]_ 1 [,...,] _[i]_ _n_ [,...,] _[i]_ _N_ _[b]_ _[j]_ [,] _[ i]_ _n_


Multilinear (Tucker) product of a core tensor,
**C** = � **G** ; **B** [(] [1] [)], . . ., **B** [(] _[N]_ [)] � **G**, and factor matrices **B** [(] _[n]_ [)], which gives


**C** = **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)]



**C** = **A** ˆ [¯] _n_ **b**


**C** = **A** ˆ [1] _N_ **[B]** [=] **[ A]** [ˆ] [1] **[ B]**


**C** = **A** ˝ **B**



Mode- _n_ product of a tensor **A** P
**R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and vector **b** P **R** _[I]_ _[n]_ yields
a tensor **C** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_,
with entries _c_ _i_ 1,..., _i_ _n_ ´1, _i_ _n_ + 1,..., _i_ _N_ =
_I_ _n_
ř _i_ _n_ = 1 _[a]_ _[i]_ 1 [,...,] _[i]_ _n_ ´1 [,] _[i]_ _n_ [,] _[i]_ _n_ + 1 [,...,] _[i]_ _N_ _[b]_ _[i]_ _n_


Mode- ( _N_, 1 ) contracted product of tensors
**A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_,
with _I_ _N_ = _J_ 1, yields a tensor
**C** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [´][1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_ with entries
_c_ _i_ 1,..., _i_ _N_ ´1, _j_ 2,..., _j_ _M_ = [ř] _i_ _[I]_ _N_ _[N]_ = 1 _[a]_ _[i]_ 1 [,...,] _[i]_ _N_ _[b]_ _[i]_ _N_ [,] _[j]_ 2 [,...,] _[j]_ _M_


Outer product of tensors **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_
and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_ yields an ( _N_ + _M_ ) thorder tensor **C**, with entries _c_ _i_ 1,..., _i_ _N_, _j_ 1,..., _j_ _M_ =
_a_ _i_ 1,..., _i_ _N_ _b_ _j_ 1,..., _j_ _M_



**X** = **a** ˝ **b** ˝ **c** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_ Outer product of vectors **a**, **b** and **c** forms a
rank-1 tensor, **X**, with entries _x_ _ijk_ = _a_ _i_ _b_ _j_ _c_ _k_



**C** = **A** b _L_ **B**



(Left) Kronecker product of tensors **A** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[N]_ yields
a tensor **C** P **R** _[I]_ [1] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ _[ J]_ _[N]_, with entries
_c_ _i_ 1 _j_ 1,..., _i_ _N_ _j_ _N_ = _a_ _i_ 1,..., _i_ _N_ _b_ _j_ 1,..., _j_ _N_



(Left) Khatri–Rao product of matrices **A** =
**C** = **A** d _L_ **B** [ **a** 1, . . ., **a** _J_ ] P **R** _[I]_ [ˆ] _[J]_ and **B** = [ **b** 1, . . ., **b** _J_ ] P
**R** _[K]_ [ˆ] _[J]_ yields a matrix **C** P **R** _[IK]_ [ˆ] _[J]_, with
columns **c** _j_ = **a** _j_ b _L_ **b** _j_ P **R** _[IK]_


24


=




|Col1|Col2|...|Col4|
|---|---|---|---|
|||||
|Tensor<br>Data|ensor<br>|ensor<br>||





Figure 2.1: Tensor reshaping operations: Matricization, vectorization and
tensorization. Matricization refers to converting a tensor into a matrix,
vectorization to converting a tensor or a matrix into a vector, while
tensorization refers to converting a vector, a matrix or a low-order tensor
into a higher-order tensor.


**Multi–indices:** By a multi-index _i_ = _i_ 1 _i_ 2 ¨ ¨ ¨ _i_ _N_ we refer to an index which
takes all possible combinations of values of indices, _i_ 1, _i_ 2, . . ., _i_ _N_, for _i_ _n_ =
1, 2, . . ., _I_ _n_, _n_ = 1, 2, . . ., _N_ and in a specific order. Multi–indices can be
defined using two different conventions [71]:


1. Little-endian convention (reverse lexicographic ordering)


_i_ 1 _i_ 2 ¨ ¨ ¨ _i_ _N_ = _i_ 1 + ( _i_ 2 ´ 1 ) _I_ 1 + ( _i_ 3 ´ 1 ) _I_ 1 _I_ 2 + ¨ ¨ ¨ + ( _i_ _N_ ´ 1 ) _I_ 1 ¨ ¨ ¨ _I_ _N_ ´1 .


2. Big-endian (colexicographic ordering)


_i_ 1 _i_ 2 ¨ ¨ ¨ _i_ _N_ = _i_ _N_ + ( _i_ _N_ ´1 ´ 1 ) _I_ _N_ + ( _i_ _N_ ´2 ´ 1 ) _I_ _N_ _I_ _N_ ´1 +

¨ ¨ ¨ + ( _i_ 1 ´ 1 ) _I_ 2 ¨ ¨ ¨ _I_ _N_ .


The little-endian convention is used, for example, in Fortran and MATLAB,
while the big-endian convention is used in C language. Given the complex
and non-commutative nature of tensors, the basic definitions, such as
the matricization, vectorization and the Kronecker product, should be


25


consistent with the chosen convention [1] . In this monograph, unless
otherwise stated, we will use little-endian notation.


**Matricization.** The matricization operator, also known as the unfolding
or flattening, reorders the elements of a tensor into a matrix (see Figure
2.2). Such a matrix is re-indexed according to the choice of multi-index
described above, and the following two fundamental matricizations are
used extensively.


**The mode-** _n_ **matricization.** For a fixed index _n_ P t1, 2, . . ., _N_ u, the mode_n_ matricization of an _N_ th-order tensor, **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, is defined as the
(“short” and “wide”) matrix


**X** ( _n_ ) P **R** _[I]_ _[n]_ [ˆ] _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[n]_ [´][1] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_, (2.1)


with _I_ _n_ rows and _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _n_ ´1 _I_ _n_ + 1 ¨ ¨ ¨ _I_ _N_ columns, the entries of which are


( **X** ( _n_ ) ) _i_ _n_, _i_ 1 ¨¨¨ _i_ _n_ ´1 _i_ _n_ + 1 ¨¨¨ _i_ _N_ = _x_ _i_ 1, _i_ 2,..., _i_ _N_ .


Note that the columns of a mode- _n_ matricization, **X** ( _n_ ), of a tensor **X** are the
mode- _n_ fibers of **X** .


**The mode-** t _n_ u **canonical matricization.** For a fixed index _n_ P
t1, 2, . . ., _N_ u, the mode- ( 1, 2, . . ., _n_ ) matricization, or simply mode- _n_
canonical matricization, of a tensor **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ is defined as the matrix


**X** ă _n_ ą P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_, (2.2)


with _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _n_ rows and _I_ _n_ + 1 ¨ ¨ ¨ _I_ _N_ columns, and the entries


( **X** ă _n_ ą ) _i_ 1 _i_ 2 ¨¨¨ _i_ _n_, _i_ _n_ + 1 ¨¨¨ _i_ _N_ = _x_ _i_ 1, _i_ 2,..., _i_ _N_ .


The matricization operator in the MATLAB notation (reverse
lexicographic) is given by


**X** ă _n_ ą = reshape ( **X**, _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _n_, _I_ _n_ + 1 ¨ ¨ ¨ _I_ _N_ ) . (2.3)


As special cases we immediately have (see Figure 2.2)


**X** ă1ą = **X** ( 1 ), **X** ă _N_ ´1ą = **X** [T] ( _N_ ) [,] **X** ă _N_ ą = vec ( **X** ) . (2.4)


1 Note that using the colexicographic ordering, the vectorization of an outer product of
two vectors, **a** and **b**, yields their Kronecker product, that is, vec ( **a** ˝ **b** ) = **a** b **b**, while
using the reverse lexicographic ordering, for the same operation, we need to use the Left
Kronecker product, vec ( **a** ˝ **b** ) = **b** b **a** = **a** b _L_ **b** .


26


(a)



_I_ 3


_I_ 1


_I_ 3


_I_ 1


_I_ 3



**A** _I_ 2 **A** (1)


_I_ 1



∈



_I_ 1 × _I_ 2 _I_ 3

R



_I_ 2


**A**


_I_ 2


**A**



_I_ 2


_I_ 3



_I_ 3



_I_ 1



**A** (2) ∈ R



_I_ 2 × _I_ 1 _I_ 3



×



_I_ 3

_I_ 1





**A** (2) ∈ R





_I_ 1


_I_ 2


(b)


|3 I I1 I2 3 × ∈R|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||





_I_ 2

















(c)



**A** _<n>_


_J_





_I_



_I_ 1


_I_ _n_





_I_ _n_ +1



Figure 2.2: Matricization (flattening, unfolding) used in tensor reshaping. (a)
Mode-1, mode-2, and mode-3 matricizations of a 3rd-order tensor, from the top
to the bottom panel. (b) Tensor network diagram for the mode- _n_ matricization
of an _N_ th-order tensor, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, into a short and wide matrix, **A** ( _n_ ) P
**R** _[I]_ _[n]_ [ ˆ] _[ I]_ [1] [¨¨¨] _[I]_ _[n]_ [´][1] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ . (c) Mode-t1, 2, . . ., _n_ uth (canonical) matricization of an _N_ thorder tensor, **A**, into a matrix **A** ă _n_ ą = **A** ( _i_ 1 ¨¨¨ _i_ _n_ ; _i_ _n_ + 1 ¨¨¨ _i_ _N_ ) P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[n]_ [ ˆ] _[ I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ .


27


4th-order tensor



**X** ∊ [I] R _[K]_ [×] [2] [×] [2] [×] [2]



Matrix

8 _K_ 4 _K_



2 _K_ × 2 × 2
**X** ∊IR
3



Vector



IR **X** ∊IR **X** ∊IR **X** ∊ 4 [I]



3rd-order tensor

4 _K_ × 2 2 _K_ × 2 × 2



**x** ∊



**X** ∊



Figure 2.3: Tensorization of a vector into a matrix, 3rd-order tensor and
4th-order tensor.


The tensorization of a vector or a matrix can be considered as a reverse
process to the vectorization or matricization (see Figures 2.1 and 2.3).


**Kronecker, strong Kronecker, and Khatri–Rao products of matrices and**
**tensors.** For an _I_ ˆ _J_ matrix **A** and a _K_ ˆ _L_ matrix **B**, the standard (Right)
Kronecker product, **A** b **B**, and the Left Kronecker product, **A** b _L_ **B**, are the
following _IK_ ˆ _JL_ matrices









**A** _b_ 1,1 ¨ ¨ ¨ **A** _b_ 1, _L_



... ... ... .
**A** _b_ _K_,1 ¨ ¨ ¨ **A** _b_ _K_, _L_ 





, **A** b _L_ **B** =



**A** b **B** =









_a_ 1,1 **B** ¨ ¨ ¨ _a_ 1, _J_ **B**

... ... ...

_a_ _I_,1 **B** ¨ ¨ ¨ _a_ _I_, _J_ **B**



Observe that **A** b _L_ **B** = **B** b **A**, so that the Left Kronecker product will
be used in most cases in this monograph as it is consistent with the littleendian notation.
Using Left Kronecker product, the strong Kronecker product of two block
matrices, **A** P **R** _[R]_ [1] _[I]_ [ˆ] _[R]_ [2] _[J]_ and **B** P **R** _[R]_ [2] _[K]_ [ˆ] _[R]_ [3] _[L]_, given by











, **B** =







,




**A** =









**A** 1,1 ¨ ¨ ¨ **A** 1, _R_ 2

... ... ...

**A** _R_ 1,1 ¨ ¨ ¨ **A** _R_ 1, _R_ 2



**B** 1,1 ¨ ¨ ¨ **B** 1, _R_ 3

... ... ...

**B** _R_ 2,1 ¨ ¨ ¨ **B** _R_ 2, _R_ 3



can be defined as a block matrix (see Figure 2.4 for a graphical illustration)


**C** = **A** |b| **B** P **R** _[R]_ [1] _[IK]_ [ˆ] _[R]_ [3] _[JL]_, (2.5)


28


**B**



**C** **=** **A** **B**









=








|A|Col2|Col3|
|---|---|---|
||||
|**A**11|**A**12|**A**13|
|**A**21|**A**22|**A**23|
||||


|A B<br>11 L 11<br>+A B<br>12 L 21<br>+A B<br>13 L 31|A B<br>11 L 12<br>+A B<br>12 L 22<br>+A B<br>13 L 32|
|---|---|
|**A**21<br>**B**<br>**+A**22<br>**+A**23<br>L<br>11<br>**B**<br>L<br>21<br>**B**<br>L<br>31|**A**21<br>**B**<br>**+A**22<br>**+A**23<br>L<br>12<br>**B**<br>L<br>22<br>**B**<br>L<br>32|



Figure 2.4: Illustration of the strong Kronecker product of two block
matrices, **A** = [ **A** _r_ 1, _r_ 2 ] P **R** _[R]_ [1] _[I]_ [1] [ˆ] _[R]_ [2] _[J]_ [1] and **B** = [ **B** _r_ 2, _r_ 3 ] P **R** _[R]_ [2] _[I]_ [2] [ˆ] _[R]_ [3] _[J]_ [2], which
is defined as a block matrix **C** = **A** |b| **B** P **R** _[R]_ [1] _[I]_ [1] _[I]_ [2] [ˆ] _[R]_ [3] _[J]_ [1] _[J]_ [2], with the blocks
**C** _r_ 1, _r_ 3 = [ř] _r_ _[R]_ 2 [2] = 1 **[A]** _[r]_ 1 [,] _[r]_ 2 [b] _[L]_ **[ B]** _[r]_ 2 [,] _[r]_ 3 [P] **[ R]** _[I]_ [1] _[I]_ [2] [ˆ] _[J]_ [1] _[J]_ [2] [, for] _[ r]_ [1] [ =] [ 1, . . .,] _[ R]_ [1] [,] _[ r]_ [2] [ =] [ 1, . . .,] _[ R]_ [2]
and _r_ 3 = 1, . . ., _R_ 3 .


with blocks **C** _r_ 1, _r_ 3 = [ř] _r_ _[R]_ 2 [2] = 1 **[A]** _[r]_ 1 [,] _[r]_ 2 [b] _[L]_ **[ B]** _[r]_ 2 [,] _[r]_ 3 [P] **[ R]** _[IK]_ [ˆ] _[JL]_ [, where] **[ A]** _[r]_ 1 [,] _[r]_ 2 [P] **[ R]** _[I]_ [ˆ] _[J]_

and **B** _r_ 2, _r_ 3 P **R** _[K]_ [ˆ] _[L]_ are the blocks of matrices within **A** and **B**,
respectively [62, 112, 113]. Note that the strong Kronecker product is
similar to the standard block matrix multiplication, but performed using
Kronecker products of the blocks instead of the standard matrix-matrix
products. The above definitions of Kronecker products can be naturally
extended to tensors [174] (see Table 2.1), as shown below.


**The Kronecker product of tensors.** The (Left) Kronecker product of two
_N_ th-order tensors, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[N]_, yields a tensor
**C** = **A** b _L_ **B** P **R** _[I]_ [1] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ _[ J]_ _[N]_ of the same order but enlarged in size, with
entries _c_ _i_ 1 _j_ 1,..., _i_ _N_ _j_ _N_ = _a_ _i_ 1,..., _i_ _N_ _b_ _j_ 1,..., _j_ _N_ as illustrated in Figure 2.5.


**The mode-** _n_ **Khatri–Rao product of tensors.** The Mode- _n_ Khatri–
Rao product of two _N_ th-order tensors, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [ˆ¨¨¨ˆ] _[I]_ _[N]_
and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[n]_ [ˆ¨¨¨ˆ] _[J]_ _[N]_, for which _I_ _n_ = _J_ _n_, yields a tensor
**C** = **A** d _n_ **B** P **R** _[I]_ [1] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] _[J]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [+] [1] _[J]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ _[ J]_ _[N]_, with subtensors
**C** ( :, . . . :, _i_ _n_, :, . . ., : ) = **A** ( :, . . . :, _i_ _n_, :, . . ., : ) b **B** ( :, . . . :, _i_ _n_, :, . . ., : ) .


**The mode-** 2 **and mode-1 Khatri–Rao product of matrices.** The above
definition simplifies to the standard Khatri–Rao (mode-2) product of two
matrices, **A** = [ **a** 1, **a** 2, . . ., **a** _R_ ] P **R** _[I]_ [ˆ] _[R]_ and **B** = [ **b** 1, **b** 2, . . ., **b** _R_ ] P **R** _[J]_ [ˆ] _[R]_, or in
other words a “column-wise Kronecker product”. Therefore, the standard


29


2 2 2 3 3 3


Figure 2.5: The left Kronecker product of two 4th-order tensors, **A** and
**B**, yields a 4th-order tensor, **C** = **A** b _L_ **B** P **R** _[I]_ [1] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ [4] _[J]_ [4], with entries
_c_ _k_ 1, _k_ 2, _k_ 3, _k_ 4 = _a_ _i_ 1,..., _i_ 4 _b_ _j_ 1,..., _j_ 4, where _k_ _n_ = _i_ _n_ _j_ _n_ ( _n_ = 1, 2, 3, 4). Note that the
order of tensor **C** is the same as the order of **A** and **B**, but the size in every
mode within **C** is a product of the respective sizes of **A** and **B** .


Right and Left Khatri–Rao products for matrices are respectively given by [2]


**A** d **B** = [ **a** 1 b **b** 1, **a** 2 b **b** 2, . . ., **a** _R_ b **b** _R_ ] P **R** _[IJ]_ [ˆ] _[R]_, (2.6)

**A** d _L_ **B** = [ **a** 1 b _L_ **b** 1, **a** 2 b _L_ **b** 2, . . ., **a** _R_ b _L_ **b** _R_ ] P **R** _[IJ]_ [ˆ] _[R]_ . (2.7)


Analogously, the mode-1 Khatri–Rao product of two matrices **A** P **R** _[I]_ [ˆ] _[R]_

and **B** P **R** _[I]_ [ˆ] _[Q]_, is defined as



 P **R** _I_ ˆ _RQ_ . (2.8)



**A** d 1 **B** =









**A** ( 1, : ) b **B** ( 1, : )

...
**A** ( _I_, : ) b **B** ( _I_, : )



**Direct sum of tensors.** A direct sum of _N_ th-order tensors **A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_
and **B** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[N]_ yields a tensor **C** = **A** ‘ **B** P **R** [(] _[I]_ [1] [+] _[J]_ [1] [)] [ˆ¨¨¨ˆ] [(] _[I]_ _[N]_ [+] _[J]_ _[N]_ [)],
with entries **C** ( _k_ 1, . . ., _k_ _N_ ) = **A** ( _k_ 1, . . ., _k_ _N_ ) if 1 ď _k_ _n_ ď _I_ _n_, @ _n_,
**C** ( _k_ 1, . . ., _k_ _N_ ) = **B** ( _k_ 1 ´ _I_ 1, . . ., _k_ _N_ ´ _I_ _N_ ) if _I_ _n_ ă _k_ _n_ ď _I_ _n_ + _J_ _n_, @ _n_,
and **C** ( _k_ 1, . . ., _k_ _N_ ) = 0, otherwise (see Figure 2.6(a)).


**Partial (mode-** _n_ **) direct sum of tensors.** A partial direct sum of tensors
**A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[N]_, with _I_ _n_ = _J_ _n_, yields a tensor
**C** = **A** ‘ ~~_n_~~ **B** P **R** [(] _[I]_ [1] [+] _[J]_ [1] [)] [ˆ¨¨¨ˆ] [(] _[I]_ _[n]_ [´][1] [+] _[J]_ _[n]_ [´][1] [)] [ˆ] _[I]_ _[n]_ [ˆ] [(] _[I]_ _[n]_ [+] [1] [+] _[J]_ _[n]_ [+] [1] [)] [ˆ¨¨¨ˆ] [(] _[I]_ _[N]_ [+] _[J]_ _[N]_ [)], where


2
For simplicity, the mode 2 subindex is usually neglected, i.e., **A** d 2 **B** = **A** d **B** .


30


**C** ( :, . . ., :, _i_ _n_, :, . . ., : ) = **A** ( :, . . ., :, _i_ _n_, :, . . ., : ) ‘ **B** ( :, . . ., :, _i_ _n_, :, . . ., : ), as
illustrated in Figure 2.6(b).


**Concatenation of** _N_ **th-order tensors.** A concatenation along mode_n_ of tensors **A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[N]_, for which _I_ _m_ = _J_ _m_,
@ _m_ ‰ _n_ yields a tensor **C** = **A** ‘ _n_ **B** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [)] [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] [(] _[I]_ _[N]_ [)],
with subtensors **C** ( _i_ 1, . . ., _i_ _n_ ´1, :, _i_ _n_ + 1, . . ., _i_ _N_ ) = **A** ( _i_ 1, . . ., _i_ _n_ ´1, :
, _i_ _n_ + 1, . . ., _i_ _N_ ) ‘ **B** ( _i_ 1, . . ., _i_ _n_ ´1, :, _i_ _n_ + 1, . . ., _i_ _N_ ), as illustrated in Figure
2.6(c). For a concatenation of two tensors of suitable dimensions along
mode- _n_, we will use equivalent notations **C** = **A** ‘ _n_ **B** = **A** " _n_ **B** .


**3D Convolution.** For simplicity, consider two 3rd-order tensors
**A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ] _[J]_ [3] . Their 3D Convolution yields a tensor
**C** = **A** ˚ **B** P **R** [(] _[I]_ [1] [+] _[J]_ [1] [´][1] [)] [ˆ] [(] _[I]_ [2] [+] _[J]_ [2] [´][1] [)] [ˆ] [(] _[I]_ [3] [+] _[J]_ [3] [´][1] [)], with entries:
**C** ( _k_ 1, _k_ 2, _k_ 3 ) = ř _j_ 1 ř _j_ 2 ř _j_ 3 **B** ( _j_ 1, _j_ 2, _j_ 3 ) **A** ( _k_ 1 ´ _j_ 1, _k_ 2 ´ _j_ 2, _k_ 3 ´ _j_ 3 ) as
illustrated in Figure 2.7 and Figure 2.8.


**Partial (mode-** _n_ **) Convolution.** For simplicity, consider two 3rd-order
tensors **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ] _[J]_ [3] . Their mode-2 (partial) convolution
yields a tensor **C** = **A** d 2 **B** P **R** _[I]_ [1] _[J]_ [1] [ˆ] [(] _[I]_ [2] [+] _[J]_ [2] [´][1] [)] [ˆ] _[I]_ [3] _[J]_ [3], the subtensors (vectors) of
which are **C** ( _k_ 1, :, _k_ 3 ) = **A** ( _i_ 1, :, _i_ 3 ) ˚ **B** ( _j_ 1, :, _j_ 3 ) P **R** _[I]_ [2] [+] _[J]_ [2] [´][1], where _k_ 1 = _i_ 1 _j_ 1,
and _k_ 3 = _i_ 3 _j_ 3 .


**Outer product.** The central operator in tensor analysis is the outer or tensor
product, which for the tensors **A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[M]_ gives
the tensor **C** = **A** ˝ **B** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[M]_ with entries _c_ _i_ 1,..., _i_ _N_, _j_ 1,..., _j_ _M_ =
_a_ _i_ 1,..., _i_ _N_ _b_ _j_ 1,..., _j_ _M_ .
Note that for 1st-order tensors (vectors), the tensor product reduces to
the standard outer product of two nonzero vectors, **a** P **R** _[I]_ and **b** P **R** _[J]_,
which yields a rank-1 matrix, **X** = **a** ˝ **b** = **ab** [T] P **R** _[I]_ [ˆ] _[J]_ . The outer product
of three nonzero vectors, **a** P **R** _[I]_, **b** P **R** _[J]_ and **c** P **R** _[K]_, gives a 3rd-order
rank-1 tensor (called pure or elementary tensor), **X** = **a** ˝ **b** ˝ **c** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_,
with entries _x_ _ijk_ = _a_ _i_ _b_ _j_ _c_ _k_ .
**Rank-1 tensor.** A tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, is said to be of rank-1 if it can
be expressed exactly as the outer product, **X** = **b** [(] [1] [)] ˝ **b** [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **b** [(] _[N]_ [)]

of nonzero vectors, **b** [(] _[n]_ [)] P **R** _[I]_ _[n]_, with the tensor entries given by
_x_ _i_ 1, _i_ 2,..., _i_ _N_ = _b_ _i_ [(] 1 [1] [)] _[b]_ _i_ [(] 2 [2] [)] [¨ ¨ ¨] _[ b]_ _i_ [(] _N_ _[N]_ [)] [.]


**Kruskal tensor, CP decomposition.** For further discussion, it is important


31


(a)
_I_
3


_I_
1





_J_

3




|Col1|J B<br>1|Col3|Col4|
|---|---|---|---|
|||||
|**A**||||
|**A**||||
|**A**||**B**||
|**A**||||



_J_



2



( _I_ 1 _J_ _**+**_ ) 1 × ( _I_ 2 _**+**_ ) _J_ 2 × ( _I_ 3 _**+**_ ) _J_ 3
**A** **B** ∈ IR


|Col1|B|
|---|---|
|||



(b)


_I_


_I_
1


(c)

|J 3|Col2|Col3|Col4|
|---|---|---|---|
|3<br><br>3||||
|3<br><br>3||||
|**A**<br>||||



_I_ 3 _**=**_ _J_ 3


_I_
1


|3<br>A<br>B|Col2|Col3|
|---|---|---|
|**A**|**A**|**A**|
|**A**|**A**||
|**A**|||
|**B**<br>|**B**|**B**|


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**A**|**A**||||
|**A**|**A**||||
|||**B**<br>|**B**<br>|**B**<br>|





_I_



_I_ 3 _**=**_ _J_ 3





**A** _I_ **A**



_J_ _**=**_ _I_
1 1



_J_
1



_I_
1







_I_ 2 _J_ 2 _I_ 2 _**=**_ _J_ 2 _I_ 2 _J_ 2

**A** **B** **A** **B** **A** **B**
1 2 3



_I_ 3 _**=**_ _J_ 3



_I_



_J_ 1 **B** _I_ 1 _**=**_ _J_ **A** **B** _I_ 1 _**=**_ _J_ 1 **A**



**B** _I_ 1 _**=**_ _J_ 1 **A** **B**



_I_ 1 _**=**_ _J_ 1



_I_ 1 _**=**_ _J_ 1


|3<br>A<br>B|Col2|
|---|---|
|**A**|**A**|
|**A**||
|**B**|**B**|


|3|Col2|
|---|---|
|||
|**A**|**B**|


|J<br>3 B<br>3<br>A|Col2|
|---|---|
|**B**<br>_J_3<br>3<br>**A**|**A**|
|**A**|**A**|



_I_ 2 _**=**_ _J_ 2



_I_
2



2



_J_



_I_ 2 _**=**_ _J_ 2



**A** **B** **A** **B** **A** **B**
1 2 3


Figure 2.6: Illustration of the direct sum, partial direct sum and
concatenation operators of two 3rd-order tensors. (a) Direct sum. (b) Partial
(mode-1, mode-2, and mode-3) direct sum. (c) Concatenations along mode1,2,3.


32


**A** **B**



**C**






|1|2|3|4|
|---|---|---|---|
|0|3|2|1|
|5|0|1|4|
|3|1|0|2|


|0|-1|0|
|---|---|---|
|~~-~~1|4|~~-~~1|
|0|~~-~~1|0|


|0|-1|-2|-3|-4|0|
|---|---|---|---|---|---|
|-1|2|1|4|12|-4|
|0|-9|8|0|-6|-1|
|-5|17|-10|-2|12|-4|
|-3|6|1|-4|4|-2|
|0|-3|-1|0|-2|0|





- ~~-~~ 1 4 ~~-~~ 1 =











|0<br>0|0<br>-1|0<br>0|Col4|Col5|
|---|---|---|---|---|
|~~-~~1<br>0|~~1~~<br>4|~~2~~<br><br>~~-~~|3<br><br>|4|
|0<br>~~0~~|~~-~~<br>~~0~~|0<br>~~3~~|~~2~~<br><br>|~~1~~<br>|
||~~5~~<br><br>|~~0~~<br><br><br>|~~1~~<br><br>|~~4~~<br>|
||~~3~~|~~1~~|~~0~~|~~2~~|


1･4+2･(-1)=2


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||2|||||
|||||||
|||||||
|||||||
|||||||

















|1<br>0|2<br>-1|33<br>0|4|
|---|---|---|---|
|~~-~~1<br>~~0~~|4<br>~~3~~|~~2~~<br><br>~~-~~1|~~1~~|
|~~5~~<br><br>0|~~0~~<br><br><br>~~-~~|~~1~~<br><br><br><br>0|~~4~~<br>|
|~~3~~<br><br>|~~1~~<br><br>|~~0~~<br><br><br>|~~2~~<br>|


2･(-1)+3･4+2･(-1)=8


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||8||||
|||||||
|||||||
|||||||



















|1|2<br>0|33<br>-1|4<br>0|
|---|---|---|---|
|~~0~~|~~-~~1<br>~~3~~|~~2~~<br><br>4|~~-~~1<br>~~1~~|
|~~5~~|~~0~~<br><br>0|~~1~~<br><br><br>~~-~~1|0<br>~~4~~|
|~~3~~<br>|~~1~~<br><br>|~~0~~<br><br>|~~2~~<br><br>|


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||||0|||
|||||||
|||||||
|||||||


3･(-1)+3･(-1)+2･4+1･(-1)+1･(-1)=0


Figure 2.7: Illustration of the 2D convolution operator, performed through
a sliding window operation along both the horizontal and vertical index.


33


**A** **B** ( _I_ + _J_ -1) **C**



( _I_ + _J_ -1)



_I_ 3 + _J_ 3 -1)



_I_


_I_
1





_J_



( _I_ 1 + _J_ 1 -1)





 - =

_J_
1






|A I|Col2|Col3|
|---|---|---|
|3<br>_I_2|3<br>_I_2|3<br>_I_2|
||||
||||
||||
||||
||||


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||




|4 0 3<br>2 3<br>2 1|Col2|Col3|Col4|Col5|2<br>6<br>4|3|3|
|---|---|---|---|---|---|---|---|
|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|2<br>6<br>4<br>|2<br>|4<br>|
|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|2<br>6<br>4<br>|2|5|
|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4<br>|0<br>|0<br>|0<br>|0<br>|
|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|2<br>|3<br>|3<br>|3<br>|3<br>|
|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|4 0 3<br>2 3<br>2 1<br>|2|1|1|1|1|
|0<br>|3<br>|2<br>|2<br>|2<br>|2<br>|2<br>|2<br>|
|2<br>|3<br>|1<br>|1<br>|1<br>|1<br>|1<br>|1<br>|
|1|0|5|5|5|5|5|5|


|-2 -1 0<br>-1 1 1<br>0 1 2|Col2|Col3|Col4|Col5|Col6|0|-1|0|
|---|---|---|---|---|---|---|---|---|
|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-1<br>|5<br>|-1<br>|
|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|0|-1|0|
|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2<br>|-1<br>|0<br>|0<br>|0<br>|0<br>|
|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-1<br>|1<br>|1<br>|1<br>|1<br>|1<br>|
|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|-2 -1 0<br>-1 1 1<br>0 1 2<br>|0|1|2|2|2|2|
|0<br>|-1<br>|0<br>|0<br>|0<br>|0<br>|0<br>|0<br>|0<br>|
|-1<br>|4<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|
|0|-1|0|0|0|0|0|0|0|


|Col1|Col2|Col3|Col4|Col5|Col6|0|-3|0|
|---|---|---|---|---|---|---|---|---|
|||||||-6<br>|10<br>|-4<br>|
|||||||0|-2|0|
||||-8<br>|0<br>|0<br>|0<br>|0<br>|0<br>|
||||-2<br>|3<br>|5<br>|5<br>|5<br>|5<br>|
||||0|1|4|4|4|4|
|0<br>|-3<br>|0<br>|0<br>|0<br>|0<br>|0<br>|0<br>|0<br>|
|-2<br>|12<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|-1<br>|
|0|-1|0|0|0|0|0|0|0|



Hadamard product


Figure 2.8: Illustration of the 3D convolution operator, performed through
a sliding window operation along all three indices.


to highlight that any tensor can be expressed as a finite sum of rank-1
tensors, in the form



_R_
ÿ

_r_ = 1



, **b** _r_ [(] _[n]_ [)] P **R** _[I]_ _[n]_, (2.9)
�



**X** =



_R_
ÿ **b** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **b** _r_ [(] _[N]_ [)] =

_r_ = 1



_N_
˝ _r_
� _n_ = 1 **[b]** [(] _[n]_ [)]



which is exactly the form of the Kruskal tensor, illustrated in Figure 2.9,
also known under the names of CANDECOMP / PARAFAC, Canonical
Polyadic Decomposition (CPD), or simply the CP decomposition in (1.2).
We will use the acronyms CP and CPD.


**Tensor rank.** The tensor rank, also called the CP rank, is a natural extension
of the matrix rank and is defined as a minimum number, _R_, of rank-1 terms
in an exact CP decomposition of the form in (2.9).
Although the CP decomposition has already found many practical
applications, its limiting theoretical property is that the best rank- _R_
approximation of a given data tensor may not exist (see [63] for more


34


Figure 2.9: The CP decomposition for a 4th-order tensor **X** of rank _R_ .
Observe that the rank-1 subtensors are formed through the outer products
of the vectors **b** _r_ [(] [1] [)] [, . . .,] **[ b]** _r_ [(] [4] [)] [,] _[ r]_ [ =] [ 1, . . .,] _[ R]_ [.]


detail). However, a rank- _R_ tensor can be approximated arbitrarily well
_R_ .
by a sequence of tensors for which the CP ranks are strictly less than
For these reasons, the concept of border rank was proposed [21], which
is defined as the minimum number of rank-1 tensors which provides the
approximation of a given tensor with an arbitrary accuracy.


**Symmetric tensor decomposition.** A symmetric tensor (sometimes called
a super-symmetric tensor) is invariant to the permutations of its indices. A
symmetric tensor of _N_ th-order has equal sizes, _I_ _n_ = _I_, @ _n_, in all its modes,
and the same value of entries for every permutation of its indices. For
example, for vectors **b** [(] _[n]_ [)] = **b** P **R** _[I]_, @ _n_, the rank-1 tensor, constructed
by _N_ outer products, ˝ _n_ _[N]_ = 1 **[b]** [(] _[n]_ [)] [ =] **[ b]** [ ˝] **[ b]** [ ˝ ¨ ¨ ¨ ˝] **[ b]** [ P] **[ R]** _[I]_ [ˆ] _[I]_ [ˆ¨¨¨ˆ] _[I]_ [, is symmetric.]
Moreover, every symmetric tensor can be expressed as a linear combination
of such symmetric rank-1 tensors through the so-called symmetric CP
decomposition, given by



**X** =



_R_
ÿ _λ_ _r_ **b** _r_ ˝ **b** _r_ ˝ ¨ ¨ ¨ ˝ **b** _r_, **b** _r_ P **R** _[I]_, (2.10)

_r_ = 1



where _λ_ _r_ P **R** are the scaling parameters for the unit length vectors **b** _r_,
while the symmetric tensor rank is the minimal number _R_ of rank-1 tensors
that is necessary for its exact representation.


**Multilinear products.** The mode- _n_ (multilinear) product, also called the
tensor-times-matrix product (TTM), of a tensor, **A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, and a
matrix, **B** P **R** _[J]_ [ˆ] _[I]_ _[n]_, gives the tensor


**C** = **A** ˆ _n_ **B** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] _[J]_ [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, (2.11)


35


(a)


(b)



_I_ 2



_I_ 3 _I_ _I_



_I_ 1



**A**
(1)







_I_ 1



_J_







**C** _J_ **B**



_J_





3

|I ..<br>3|Col2|Col3|A|Col5|Col6|
|---|---|---|---|---|---|
|..<br>_I_3<br>|..<br>_I_3<br>||**A**|**A**|**A**|
|..<br>_I_3<br>|||**A**|**A**|**A**|
|..<br>_I_3<br>||||||
|**B**<br>|**C**|**C**|**C**|||
|**B**<br>|**C**|**C**|**C**|||

_I_ 1 _I_ 2



_I_ 1



**C**

|2 3<br>A A ... A<br>1 2 I<br>3|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|2<br>3<br><br><br>...<br>**A**1<br>**A**2<br>**A**_I_3<br>|**A**1|**A**2|...|**A**_I_3|
|**B**<br>|**BA**1|**BA**2|...|**BA**_I_3|
|_I_|_I_|_I_|_I_|_I_|

(1)



**C** = **A** × 1 **B** **C** (1) = **B A** (1)















Figure 2.10: Illustration of the multilinear mode- _n_ product, also known as
the TTM (Tensor-Times-Matrix) product, performed in the tensor format
(left) and the matrix format (right). (a) Mode-1 product of a 3rd-order
tensor, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3], and a factor (component) matrix, **B** P **R** _[J]_ [ˆ] _[I]_ [1], yields
a tensor **C** = **A** ˆ 1 **B** P **R** _[J]_ [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] . This is equivalent to a simple matrix
multiplication formula, **C** ( 1 ) = **BA** ( 1 ) . (b) Graphical representation of a
mode- _n_ product of an _N_ th-order tensor, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, and a factor
matrix, **B** P **R** _[J]_ [ˆ] _[I]_ _[n]_ .


with entries



_c_ _i_ 1, _i_ 2,..., _i_ _n_ ´1, _j_, _i_ _n_ + 1,..., _i_ _N_ =



_I_ _n_
ÿ _a_ _i_ 1, _i_ 2,..., _i_ _N_ _b_ _j_, _i_ _n_ . (2.12)

_i_ _n_ = 1



From (2.12) and Figure 2.10, the equivalent matrix form is **C** ( _n_ ) = **BA** ( _n_ ),
which allows us to employ established fast matrix-by-vector and
matrix-by-matrix multiplications when dealing with very large-scale
tensors. Efficient and optimized algorithms for TTM are, however, still
emerging [11,12,131].


36


**Full multilinear (Tucker) product.** A full multilinear product, also called
the Tucker product, of an _N_ th-order tensor, **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ¨¨¨ˆ] _[R]_ _[N]_, and a
set of _N_ factor matrices, **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ for _n_ = 1, 2, . . ., _N_, performs the
multiplications in all the modes and can be compactly written as (see Figure
2.11(b))


**C** = **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)] (2.13)
= � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] � P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ .


Observe that this format corresponds to the Tucker decomposition

[119,209,210] (see Section 3.3).


**Multilinear product of a tensor and a vector (TTV).** In a similar way, the
mode- _n_ multiplication of a tensor, **A** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, and a vector, **b** P **R** _[I]_ _[n]_
(tensor-times-vector, TTV) yields a tensor


**C** = **A** ˆ [¯] _n_ **b** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, (2.14)


with entries



_c_ _i_ 1,..., _i_ _n_ ´1, _i_ _n_ + 1,..., _i_ _N_ =



_I_ _n_
ÿ _a_ _i_ 1,..., _i_ _n_ ´1, _i_ _n_, _i_ _n_ + 1,..., _i_ _N_ _b_ _i_ _n_ . (2.15)

_i_ _n_ = 1



Note that the mode- _n_ multiplication of a tensor by a matrix does not change
the tensor order, while the multiplication of a tensor by vectors reduces its
order, with the mode _n_ removed (see Figure 2.11).
Multilinear products of tensors by matrices or vectors play a key role
in deterministic methods for the reshaping of tensors and dimensionality
reduction, as well as in probabilistic methods for randomization /
sketching procedures and in random projections of tensors into matrices
or vectors. In other words, we can also perform reshaping of a tensor
through random projections that change its entries, dimensionality or
size of modes, and/or the tensor order. This is achieved by multiplying
a tensor by random matrices or vectors, transformations which preserve
its basic properties. [72, 126, 132, 137, 168, 192, 199, 223] (see Section 3.5 for
more detail).


**Tensor contractions.** Tensor contraction is a fundamental and the most
important operation in tensor networks, and can be considered a higherdimensional analogue of matrix multiplication, inner product, and outer
product.


37


(a)













































(b) (c)


























|I<br>5<br>(5)<br>B<br>I 1 B(1) R 5 B(4)<br>R<br>4 I<br>4<br>R G<br>1<br>R<br>B(2) 2 R B(3)<br>3<br>I<br>I 3<br>2|Col2|Col3|
|---|---|---|
|_R_1<br>_R_2<br>_R_4<br>**B**<br>(1)_ R_5<br>**B**<br>(5)<br>**B**<br>(4)<br>**B**<br>(3)<br>**B**<br>(2)<br>_I_1<br>_I_2<br>_I_3<br>_I_4<br>_I_5<br>_R_3<br>**G**|_R_1<br>_R_2<br>_R_4<br>**B**<br>(1)_ R_5<br>**B**<br>(5)<br>**B**<br>(4)<br>**B**<br>(3)<br>**B**<br>(2)<br>_I_1<br>_I_2<br>_I_3<br>_I_4<br>_I_5<br>_R_3<br>**G**|_R_4<br>_R_1<br>_R_2<br>~~_R_~~3<br>**b**3<br>**b**2<br>**b**1<br>**G**|



Figure 2.11: Multilinear tensor products in a compact tensor network
notation. (a) Transforming and/or compressing a 4th-order tensor, **G** P
**R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[R]_ [3] [ˆ] _[R]_ [4], into a scalar, vector, matrix and 3rd-order tensor, by
multilinear products of the tensor and vectors. Note that a mode- _n_
multiplication of a tensor by a matrix does not change the order of a
tensor, while a multiplication of a tensor by a vector reduces its order by
one. For example, a multilinear product of a 4th-order tensor and four
vectors (top diagram) yields a scalar. (b) Multilinear product of a tensor,
**G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ¨¨¨ˆ] _[R]_ [5], and five factor (component) matrices, **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ ( _n_ =
1, 2, . . ., 5), yields the tensor **C** = **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ˆ 3 **B** [(] [3] [)] ˆ 4 **B** [(] [4] [)] ˆ 5 **B** [(] [5] [)] P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ [5] . This corresponds to the Tucker format. (c) Multilinear product
of a 4th-order tensor, **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[R]_ [3] [ˆ] _[R]_ [4], and three vectors, **b** _n_ P **R** _[R]_ _[n]_
( _n_ = 1, 2, 3 ), yields the vector **c** = **G** ˆ [¯] 1 **b** 1 ˆ [¯] 2 **b** 2 ˆ [¯] 3 **b** 3 P **R** _[R]_ [4] .


38


In a way similar to the mode- _n_ multilinear product [3], the mode- ( _n_ _[m]_ [)]
product (tensor contraction) of two tensors, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P
**R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_, with common modes, _I_ _n_ = _J_ _m_, yields an ( _N_ + _M_ ´ 2 ) -order
tensor, **C** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [+] [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[m]_ [´][1] [ˆ] _[J]_ _[m]_ [+] [1] [ˆ¨¨¨ˆ] _[J]_ _[M]_, in the form (see
Figure 2.12(a))


**C** = **A** ˆ _[m]_ _n_ **[B]** [,] (2.16)


for which the entries are computed as


_c_ _i_ 1, ..., _i_ _n_ ´1, _i_ _n_ + 1, ..., _i_ _N_, _j_ 1, ..., _j_ _m_ ´1, _j_ _m_ + 1, ..., _j_ _M_ =



=



_I_ _n_
ÿ _a_ _i_ 1,..., _i_ _n_ ´1, _i_ _n_, _i_ _n_ + 1, ..., _i_ _N_ _b_ _j_ 1, ..., _j_ _m_ ´1, _i_ _n_, _j_ _m_ + 1, ..., _j_ _M_ . (2.17)

_i_ _n_ = 1



This operation is referred to as a _contraction of two tensors in single common_
_mode_ .

Tensors can be contracted in several modes or even in all modes, as
illustrated in Figure 2.12. For convenience of presentation, the super- or
sub-index, e.g., _m_, _n_, will be omitted in a few special cases. For example, the
multilinear product of the tensors, **A** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_,
with common modes, _I_ _N_ = _J_ 1, can be written as


**C** = **A** ˆ [1] _N_ **[B]** [=] **[ A]** [ˆ] [1] **[ B]** [=] **[ A]** [‚] **[ B]** [P] **[ R]** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [´][1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[M]_ [,] (2.18)


for which the entries



_c_ _i_ 1, _i_ 2,..., _i_ _N_ ´1, _j_ 2, _j_ 3,..., _j_ _M_ =



_I_ _N_
ÿ _a_ _i_ 1, _i_ 2,..., _i_ _N_ _b_ _i_ _N_, _j_ 2,..., _j_ _M_ .

_i_ _N_ = 1



In this notation, the multiplications of matrices and vectors can be
written as, **A** ˆ [1] 2 **[B]** [ =] **[ A]** [ ˆ] [1] **[ B]** [ =] **[ AB]** [,] **[ A]** [ ˆ] [2] 2 **[B]** [ =] **[ AB]** [T] [,] **[ A]** [ ˆ] [1,2] 1,2 **[B]** [ =] **[ A]** [ ¯][ˆ] **[B]** [ =]
x **A**, **B** y, and **A** ˆ [1] 2 **[x]** [ =] **[ A]** [ ˆ] [1] **[ x]** [ =] **[ Ax]** [.]
Note that tensor contractions are, in general not associative or
commutative, since when contracting more than two tensors, the order has
to be precisely specified (defined), for example, **A** ˆ _[b]_ _a_ [(] **[B]** [ˆ] _[d]_ _c_ **[C]** [)] [ for] _[ b]_ [ ă] _[ c]_ [.]
It is also important to note that a matrix-by-vector product, **y** =
**Ax** P **R** _[I]_ [1] [¨¨¨] _[I]_ _[N]_, with **A** P **R** _[I]_ [1] [¨¨¨] _[I]_ _[N]_ [ˆ] _[J]_ [1] [¨¨¨] _[J]_ _[N]_ and **x** P **R** _[J]_ [1] [¨¨¨] _[J]_ _[N]_, can be expressed
in a tensorized form via the contraction operator as **Y** = **A** ˆ [¯] **X**, where


3
In the literature, sometimes the symbol ˆ _n_ is replaced by ‚ _n_ .


39


Figure 2.12: Examples of contractions of two tensors. (a) Multilinear
product of two tensors is denoted by **A** ˆ _[m]_ _n_ **B** . (b) Inner product of two
3rd-order tensors yields a scalar _c_ = x **A**, **B** y = **A** ˆ [1,2,3] 1,2,3 **[B]** [=] **[ A]** [ˆ][¯] **[ B]** [=]
ř _i_ 1, _i_ 2, _i_ 3 _[a]_ _[i]_ 1 [,] _[i]_ 2 [,] _[i]_ 3 _[b]_ _[i]_ 1 [,] _[i]_ 2 [,] _[i]_ 3 [. (c) Tensor contraction of two 4th-order tensors, along]
mode-3 in **A** and mode-2 in **B**, yields a 6th-order tensor, **C** = **A** ˆ [2] 3 **[B]** [P]
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [4] [ˆ] _[J]_ [1] [ˆ] _[J]_ [3] [ˆ] _[J]_ [4], with entries _c_ _i_ 1, _i_ 2, _i_ 4, _j_ 1, _j_ 3, _j_ 4 = [ř] _i_ 3 _a_ _i_ 1, _i_ 2, _i_ 3, _i_ 4 _b_ _j_ 1, _i_ 3, _j_ 3, _j_ 4 . (d)
Tensor contraction of two 5th-order tensors along the modes 3, 4, 5 in **A**
and 1, 2, 3 in **B** yields a 4th-order tensor, **C** = **A** ˆ [1,2,3] 5,4,3 **[B]** [P] **[ R]** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[J]_ [4] [ˆ] _[J]_ [5] [.]


the symbol ˆ [¯] denotes the contraction of all modes of the tensor **X** (see
Section 4.5).
Unlike the matrix-by-matrix multiplications for which several efficient
parallel schemes have been developed, (e.g. BLAS procedure) the
number of efficient algorithms for tensor contractions is rather limited. In
practice, due to the high computational complexity of tensor contractions,
especially for tensor networks with loops, this operation is often performed
approximately [66,107,138,167].


**Tensor trace.** Consider a tensor with partial self-contraction modes, where
the outer (or open) indices represent physical modes of the tensor, while
the inner indices indicate its contraction modes. The Tensor Trace operator
performs the summation of all inner indices of the tensor [89]. For example,
a tensor **A** of size _R_ ˆ _I_ ˆ _R_ has two inner indices, modes 1 and 3 of size


40


_R_, and one open mode of size _I_ . Its tensor trace yields a vector of length _I_,
given by
**a** = Tr ( **A** ) = ÿ **A** ( _r_, :, _r_ ),


_r_

the elements of which are the traces of its lateral slices **A** _i_ P **R** _[R]_ [ˆ] _[R]_ ( _i_ =
1, 2, . . ., _I_ ), that is, (see bottom of Figure 2.13)


**a** = [ tr ( **A** 1 ), . . ., tr ( **A** _i_ ), . . ., tr ( **A** _I_ )] [T] . (2.19)


A tensor can have more than one pair of inner indices, e.g., the tensor **A**
of size _R_ ˆ _I_ ˆ _S_ ˆ _S_ ˆ _I_ ˆ _R_ has two pairs of inner indices, modes 1 and
6, modes 3 and 4, and two open modes (2 and 5). The tensor trace of **A**
therefore returns a matrix of size _I_ ˆ _I_ defined as



Tr ( **A** ) = ÿ


_r_



ÿ **A** ( _r_, :, _s_, _s_, :, _r_ ) .


_s_



A variant of Tensor Trace [128] for the case of the partial tensor selfcontraction considers a tensor **A** P **R** _[R]_ [ˆ] _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[R]_ and yields a reducedorder tensor **A** [r] = Tr ( **A** ) P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with entries



r
**A** ( _i_ 1, _i_ 2, . . ., _i_ _N_ ) =



_R_
ÿ **A** ( _r_, _i_ 1, _i_ 2, . . ., _i_ _N_, _r_ ), (2.20)

_r_ = 1



Conversions of tensors to scalars, vectors, matrices or tensors with
reshaped modes and/or reduced orders are illustrated in Figures 2.11– 2.13.


**2.2** **Graphical Representation of Fundamental Tensor**
**Networks**


Tensor networks (TNs) represent a higher-order tensor as a set of sparsely
interconnected lower-order tensors (see Figure 2.14), and in this way
provide computational and storage benefits. The lines (branches, edges)
connecting core tensors correspond to the contracted modes while their
weights (or numbers of branches) represent the rank of a tensor network [4],
whereas the lines which do not connect core tensors correspond to the
“external” physical variables (modes, indices) within the data tensor. In
other words, the number of free (dangling) edges (with weights larger than
one) determines the order of a data tensor under consideration, while set
of weights of internal branches represents the TN rank.


4 Strictly speaking, the minimum set of internal indices t _R_ 1, _R_ 2, _R_ 3, . . .u is called the rank
(bond dimensions) of a specific tensor network.


41


_c_ tr( ) **A**



= _aii_
_i_





**A**
_I_


**A** 2



**A** 3



_c_ tr ( **A** 1 **A** 2 **A** 3 **A** 4 )



**x** **A** **y**





**X** **A** **X**


_K_





_R_ =



tr( **Ayx** T ) **xAy** T


tr( **X AX** T )


[ _a_ 1 _a_, 2,..., _a_ _I_ []] [T]


_i_ _r_ **[A]** [(] _[r,i,r]_ [)]







Figure 2.13: Tensor network notation for the traces of matrices (panels 14 from the top), and a (partial) tensor trace (tensor self-contraction) of a
3rd-order tensor (bottom panel). Note that graphical representations of
the trace of matrices intuitively explain the permutation property of trace
operator, e.g., tr ( **A** 1 **A** 2 **A** 3 **A** 4 ) = tr ( **A** 3 **A** 4 **A** 1 **A** 2 ) .


**2.3** **Hierarchical** **Tucker** **(HT)** **and** **Tree** **Tensor**

**Network State (TTNS) Models**


Hierarchical Tucker (HT) decompositions (also called hierarchical tensor
representation) have been introduced in [92] and also independently in

[86], see also [7, 91, 122, 139, 211] and references therein [5] . Generally, the
HT decomposition requires splitting the set of modes of a tensor in a
hierarchical way, which results in a binary tree containing a subset of
modes at each branch (called a dimension tree); examples of binary trees
are given in Figures 2.15, 2.16 and 2.17. In tensor networks based on binary


5 The HT model was developed independently, from a different perspective, in the
chemistry community under the name MultiLayer Multi-Configurational Time-Dependent
Hartree method (ML-MCTDH) [220]. Furthermore, the PARATREE model, developed
independently for signal processing applications [181], is quite similar to the HT model [86].


42


Figure 2.14: Illustration of the decomposition of a 9th-order tensor, **X** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ [9], into different forms of tensor networks (TNs). In general, the
objective is to decompose a very high-order tensor into sparsely (weakly)
connected low-order and small size tensors, typically 3rd-order and 4thorder tensors called cores. Top: The Tensor Chain (TC) model, which
is equivalent to the Matrix Product State (MPS) with periodic boundary
conditions (PBC). Middle: The Projected Entangled-Pair States (PEPS), also
with PBC. Bottom: The Tree Tensor Network State (TTNS).


trees, all the cores are of order of three or less. Observe that the HT model
does not contain any cycles (loops), i.e., no edges connecting a node with
itself. The splitting operation of the set of modes of the original data tensor
by binary tree edges is performed through a suitable matricization.
**Choice of dimension tree.** The dimension tree within the HT format
is chosen _a priori_ and defines the topology of the HT decomposition.
Intuitively, the dimension tree specifies which groups of modes are
“separated” from other groups of modes, so that a sequential HT
decomposition can be performed via a (truncated) SVD applied to a
suitably matricized tensor. One of the simplest and most straightforward
choices of a dimension tree is the linear and unbalanced tree, which gives
rise to the tensor-train (TT) decomposition, discussed in detail in Section 2.4
and Section 4 [158,161].
Using mathematical formalism, a dimension tree is a binary tree _T_ _N_,


43


Figure 2.15: The standard Tucker decomposition of an 8th-order tensor into
a core tensor (red circle) and eight factor matrices (green circles), and its
transformation into an equivalent Hierarchical Tucker (HT) model using
interconnected smaller size 3rd-order core tensors and the same factor

matrices.


_N_ ą 1, which satisfies that


(i) all nodes _t_ P _T_ _N_ are non-empty subsets of _{_ 1, 2,..., N _}_,


(ii) the set _t_ _root_ = t1, 2, . . ., _N_ u is the root node of _T_ _N_, and


(iii) each non-leaf node has two children _u_, _v_ P _T_ _N_ such that _t_ is a disjoint
union _t_ = _u_ Y _v_ .


The HT model is illustrated through the following Example.


**Example.** Suppose that the dimension tree _T_ 7 is given, which gives the
HT decomposition illustrated in Figure 2.17. The HT decomposition of a
tensor **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ [7] with given set of integers t _R_ _t_ u _t_ P _T_ 7 can be expressed
in the tensor and vector / matrix forms as follows. Let intermediate
tensors **X** [(] _[t]_ [)] with _t_ = t _n_ 1, . . ., _n_ _k_ u Ă t1, . . ., 7u have the size _I_ _n_ 1 ˆ _I_ _n_ 2 ˆ
¨ ¨ ¨ ˆ _I_ _n_ _k_ ˆ _R_ _t_ . Let **X** _r_ [(] _t_ _[t]_ [)] ” **X** [(] _[t]_ [)] ( :, . . ., :, _r_ _t_ ) denote the subtensor of **X** [(] _[t]_ [)] and

**X** [(] _[t]_ [)] ” **X** [(] _[t]_ [)]
ă _k_ ą [P] **[ R]** _[I]_ _[n]_ [1] _[ I]_ _[n]_ [2] [¨¨¨] _[I]_ _[nk]_ [ˆ] _[R]_ _[t]_ [ denote the corresponding unfolded matrix.]
Let **G** [(] _[t]_ [)] P **R** _[R]_ _[u]_ [ˆ] _[R]_ _[v]_ [ˆ] _[R]_ _[t]_ be core tensors where _u_ and _v_ denote respectively the
left and right children of _t_ .


44


Order 3: Order 4:


Order 5:


Order 6:


Order 7:


Order 8:


Figure 2.16: Examples of HT/TT models (formats) for distributed Tucker
decompositions with 3rd-order cores, for different orders of data tensors.
Green circles denote factor matrices (which can be absorbed by core
tensors), while red circles indicate cores. Observe that the representations
are not unique.


45


**G**



(12 ・・・7)



















_I_











Figure 2.17: Example illustrating the HT decomposition for a 7th-order data

tensor.


The HT model shown in Figure 2.17 can be then described
mathematically in the vector form as


vec ( **X** ) – ( **X** [(] [123] [)] b _L_ **X** [(] [4567] [)] ) vec ( **G** [(] [12][¨¨¨][7] [)] ),


**X** [(] [123] [)] – ( **B** [(] [1] [)] b _L_ **X** [(] [23] [)] ) **G** ă [(] [123] 2ą [)] [,] **X** [(] [4567] [)] – ( **X** [(] [45] [)] b _L_ **X** [(] [67] [)] ) **G** ă [(] [4567] 2ą [)] [,]


**X** [(] [23] [)] – ( **B** [(] [2] [)] b _L_ **B** [(] [3] [)] ) **G** ă [(] [23] 2ą [)] [,] **X** [(] [45] [)] – ( **B** [(] [4] [)] b _L_ **B** [(] [5] [)] ) **G** ă [(] [45] 2ą [)] [,]


**X** [(] [67] [)] – ( **B** [(] [6] [)] b _L_ **B** [(] [7] [)] ) **G** ă [(] [67] 2ą [)] [.]


An equivalent, more explicit form, using tensor notations becomes



_R_ 4567
ÿ _g_ _r_ [(] 123 [12][¨¨¨], _r_ 4567 [7] [)] **[X]** _r_ [(] 123 [123] [)] ˝ **X** _r_ [(] 4567 [4567] [)] [,]

_r_ 4567 = 1



**X** –



_R_ 123
ÿ

_r_ 123 = 1



_R_ 23
ÿ _g_ _r_ [(] 1 [123], _r_ 23 [)], _r_ 123 **[b]** _r_ [(] 1 [1] [)] [˝] **[ X]** _r_ [(] 23 [23] [)] [,]

_r_ 23 = 1


46



**X** _r_ [(] 123 [123] [)] –



_R_ 1
ÿ

_r_ 1 = 1


**X** _r_ [(] 4567 [4567] [)] –



_R_ 45
ÿ

_r_ 45 = 1



_R_ 67
ÿ _g_ _r_ [(] 45 [4567], _r_ 67 [)], _r_ 4567 **[X]** _r_ [(] 45 [45] [)] [˝] **[ X]** _r_ [(] 67 [67] [)] [,]

_r_ 67 = 1



_R_ 3
ÿ _g_ _r_ [(] 2 [23], _r_ [)] 3, _r_ 23 **[b]** _r_ [(] 2 [2] [)] [˝] **[ b]** _r_ [(] 3 [3] [)] [,]

_r_ 3 = 1



**X** _r_ [(] 23 [23] [)] –


**X** _r_ [(] 45 [45] [)] –


**X** _r_ [(] 67 [67] [)] –



_R_ 2
ÿ

_r_ 2 = 1


_R_ 4
ÿ

_r_ 4 = 1


_R_ 6
ÿ

_r_ 6 = 1



_R_ 7
ÿ _g_ _r_ [(] 6 [67], _r_ [)] 7, _r_ 67 **[b]** _r_ [(] 6 [6] [)] [˝] **[ b]** _r_ [(] 7 [7] [)] [.]

_r_ 7 = 1



_R_ 5
ÿ _g_ _r_ [(] 4 [45], _r_ [)] 5, _r_ 45 **[b]** _r_ [(] 4 [4] [)] [˝] **[ b]** _r_ [(] 5 [5] [)] [,]

_r_ 5 = 1



The TT/HT decompositions lead naturally to a distributed Tucker
decomposition, where a single core tensor is replaced by interconnected
cores of lower-order, resulting in a distributed network in which only some
cores are connected directly with factor matrices, as illustrated in Figure
2.15. Figure 2.16 illustrates exemplary HT/TT structures for data tensors of
various orders [122,205]. Note that for a 3rd-order tensor, there is only one
HT tensor network representation, while for a 5th-order we have 5, and for
a 10th-order tensor there are 11 possible HT architectures.
A simple approach to reduce the size of a large-scale core tensor in the
standard Tucker decomposition (typically, for _N_ ą 5) would be to apply
the concept of distributed tensor networks (DTNs). The DTNs assume two
kinds of cores (blocks): (i) the internal cores (nodes) which are connected
only to other cores and have no free edges and (ii) external cores which
do have free edges representing physical modes (indices) of a given data
tensor (see also Section 2.6). Such distributed representations of tensors are
not unique.
The tree tensor network state (TTNS) model, whereby all nodes are of
3rd-order or higher, can be considered as a generalization of the TT/HT
decompositions, as illustrated by two examples in Figure 2.18 [149]. A more
detailed mathematical description of the TTNS is given in Section 3.3.


47


Figure 2.18: The Tree Tensor Network State (TTNS) with 3rd-order and 4thorder cores for the representation of 24th-order data tensors. The TTNS
can be considered both as a generalization of HT/TT format and as a
distributed model for the Tucker- _N_ decomposition (see Section 3.3).


**2.4** **Tensor Train (TT) Network**


The Tensor Train (TT) format can be interpreted as a special case of the
HT format, where all nodes (TT-cores) of the underlying tensor network
are connected in cascade (or train), i.e., they are aligned while factor
matrices corresponding to the leaf modes are assumed to be identities and
thus need not be stored. The TT format was first proposed in numerical
analysis and scientific computing in [158, 161]. Figure 2.19 presents the
concept of TT decomposition for an _N_ th-order tensor, the entries of which
can be computed as a cascaded (multilayer) multiplication of appropriate
matrices (slices of TT-cores). The weights of internal edges (denoted by
t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u) represent the TT-rank. In this way, the so aligned
sequence of core tensors represents a “tensor train” where the role of
“buffers” is played by TT-core connections. It is important to highlight that
TT networks can be applied not only for the approximation of tensorized
vectors but also for scalar multivariate functions, matrices, and even largescale low-order tensors, as illustrated in Figure 2.20 (for more detail see
Section 4).
In the quantum physics community, the TT format is known as the
Matrix Product State (MPS) representation with the Open Boundary
Conditions (OBC) and was introduced in 1987 as the ground state of the


48


(a)



**G** [(1)] **G** [(2)] **G** [( )] _n_ **G** [( )] _N_

_R_ 1 _R_ 2 _R_ _n_ -1 _R_ _n_ _R_ _N_ -1



_I_ 1 _I_ 2 _I_ _n_ _I_ _N_



(1) (2)
**G** **G**



( ) _n_ ( _N_ )
**G** **G**



_i_ 2 _i_ _n_ _[i]_ _N_





_I_ _n_


... ...


_R_ _n_ -1

|i<br>...<br>..|Col2|
|---|---|
|||
|||
|||
|||



_R_ _n_



_I_ 1


(b)



_R_ 1

_R_ 1

|...<br>2|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|...<br>2<br><br>|...<br>2<br><br>|...<br>2<br><br>||||
|||||||

_R_ 2



_R_ _N_





(1) (2)
**G** _i_ 1 **G** _i_ 2



( ) _n_
**G**



( ) _N_
**G** _i_ _N_



_i_ 2 _i_ _n_





_R_ 1



_I_ _n_


_R_ _n_ -1



_R_ _N_ -1



_R_ _N_


|1 ...|Col2|
|---|---|
||..|


|I 2 ...|Col2|Col3|
|---|---|---|
|...<br>_I_2<br>|||
|||..|


|i<br>...<br>N<br>...|Col2|Col3|
|---|---|---|
|...<br>...<br>_N_<br>|||
||||



_R_ _n_



_R_ 1



_R_ 2



_R_ _N_



Figure 2.19: Concepts of the tensor train (TT) and tensor chain (TC)
decompositions (MPS with OBC and PBC, respectively) for an _N_ th-order data
tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ . (a) Tensor Train (TT) can be mathematically

described as _x_ _i_ 1, _i_ 2,..., _i_ _N_ = **G** _i_ [(] 1 [1] [)] **G** _i_ [(] 2 [2] [)] ¨ ¨ ¨ **G** _i_ [(] _N_ _[N]_ [)] [, where (bottom panel) the slice]

matrices of TT-cores **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ are defined as **G** _i_ [(] _n_ _[n]_ [)] = **G** [(] _[n]_ [)] ( :, _i_ _n_, :
) P **R** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ with _R_ 0 = _R_ _N_ = 1. (b) For the Tensor Chain (TC), the
entries of a tensor are expressed as _x_ _i_ 1, _i_ 2,..., _i_ _N_ = tr ( **G** _i_ [(] 1 [1] [)] **G** _i_ [(] 2 [2] [)] ¨ ¨ ¨ **G** _i_ [(] _N_ _[N]_ [)] [) =]



_R_ 1
ÿ

_r_ 1 = 1



_R_ 2


¨ ¨ ¨

ÿ

_r_ 2 = 1



_R_ _N_
ÿ _g_ [(] _r_ [1] _N_ [)], _i_ 1, _r_ 1 _[g]_ [(] _r_ [2] 1 [)], _i_ 2, _r_ 2 [¨ ¨ ¨] _[ g]_ [(] _r_ _[N]_ _N_ [)] ´1, _i_ _N_, _r_ _N_ [, where (bottom panel) the lateral]

_r_ _N_ = 1



slices of the TC-cores are defined as **G** _i_ [(] _n_ _[n]_ [)] = **G** [(] _[n]_ [)] ( :, _i_ _n_, : ) P **R** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ and

_g_ [(] _r_ _[n]_ _n_ [)] ´1, _i_ _n_, _r_ _n_ [=] **[ G]** [(] _[n]_ [)] [(] _[r]_ _[n]_ [´] [1] [,] _[ i]_ _[n]_ [,] _[ r]_ _[n]_ [)] [ for] _[ n]_ [ =] [ 1, 2, . . .,] _[ N]_ [, with] _[ R]_ [0] [ =] _[ R]_ _[N]_ [ ą][ 1. Notice]
that TC/MPS is effectively a TT with a single loop connecting the first and the last
core, so that all TC-cores are of 3rd-order.


49


**a**


**A**



_I_ = _I_ 1 _I_ 2 … _I_ _N_


_J_ = _J_ 1 _J_ 2 … _J_ _N_


_I_ = _I_ _I_ … _I_
1 2 _N_


_K K_ = 1 _K_ 2 … _K_ _N_


_J_


|I I I<br>1 2 3|Col2|Col3|Col4|
|---|---|---|---|
|_I_1<br>_I_2<br>_I_3|_I_1|_I_2|_I_3|


|Col1|J<br>1|J<br>2|J<br>3|
|---|---|---|---|
||_I_1|_I_2|_I_3|









_N_





_J_
_N_



Figure 2.20: Forms of tensor train decompositions for a vector, **a** P **R** _[I]_,
matrix, **A** P **R** _[I]_ [ˆ] _[J]_, and 3rd-order tensor, **A** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_ (by applying a suitable
tensorization).


1D AKLT model [2]. It was subsequently extended by many researchers [6]

(see [102,156,166,183,214,216,224] and references therein).


**Advantages of TT formats.** An important advantage of the TT/MPS
format over the HT format is its simpler practical implementation, as no
binary tree needs to be determined (see Section 4). Another attractive
property of the TT-decomposition is its simplicity when performing basic
mathematical operations on tensors directly in the TT-format (that is,
employing only core tensors). These include matrix-by-matrix and matrixby-vector multiplications, tensor addition, and the entry-wise (Hadamard)
product of tensors. These operations produce tensors, also in the TTformat, which generally exhibit increased TT-ranks. A detailed description
of basic operations supported by the TT format is given in Section 4.5.
Moreover, only TT-cores need to be stored and processed, which makes
the number of parameters to scale linearly in the tensor order, _N_, of a data
tensor and all mathematical operations are then performed only on the loworder and relatively small size core tensors.


6 In fact, the TT was rediscovered several times under different names: MPS, valence
bond states, and density matrix renormalization group (DMRG) [224]. The DMRG usually
refers not only to a tensor network format but also the efficient computational algorithms
(see also [101,182] and references therein). Also, in quantum physics the ALS algorithm is
called the one-site DMRG, while the Modified ALS (MALS) is known as the two-site DMRG
(for more detail, see Part 2).


50


Figure 2.21: Class of 1D and 2D tensor train networks with open boundary
conditions (OBC): the Matrix Product State (MPS) or (vector) Tensor Train
(TT), the Matrix Product Operator (MPO) or Matrix TT, the Projected
Entangled-Pair States (PEPS) or Tensor Product State (TPS), and the
Projected Entangled-Pair Operators (PEPO).


The TT rank is defined as an ( _N_ ´ 1 ) -tuple of the form


rank TT ( **X** ) = **r** _TT_ = t _R_ 1, . . ., _R_ _N_ ´1 u, _R_ _n_ = rank ( **X** ă _n_ ą ), (2.21)


where **X** ă _n_ ą P **R** _[I]_ [1] [¨¨¨] _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [´][1] [¨¨¨] _[I]_ _[N]_ is an _n_ th canonical matricization of the tensor

**X** .
Since the TT rank determines memory requirements of a tensor train,
it has a strong impact on the complexity, i.e., the suitability of tensor train
representation for a given raw data tensor.
The number of data samples to be stored scales linearly in the tensor
order, _N_, and the size, _I_, and quadratically in the maximum TT rank bound,
_R_, that is


_N_
ÿ _R_ _n_ ´1 _R_ _n_ _I_ _n_ „ _O_ ( _NR_ [2] _I_ ), _R_ : = max _n_ [t] _[R]_ _[n]_ [u][,] _I_ : = max _n_ [t] _[I]_ _[n]_ [u][.] (2.22)

_n_ = 1


.
This is why it is crucially important to have low-rank TT approximations [7]
A drawback of the TT format is that the ranks of a tensor train
decomposition depend on the ordering (permutation) of the modes,


7 In the worst case scenario the TT ranks can grow up to _I_ ( _N_ /2 ) for an _N_ th-order tensor.


51


which gives different size of cores for different ordering. To solve this
challenging permutation problem, we can estimate mutual information
between individual TT cores pairwise (see [13, 73]). The procedure can be
arranged in the following three steps: (i) Perform a rough (approximate) TT
decomposition with relative low TT-rank and calculate mutual information
between all pairs of cores, (ii) order TT cores in such way that the mutual
information matrix is close to a diagonal matrix, and finally, (iii) perform
TT decomposition again using the so optimised order of TT cores (see also
Part 2).


**2.5** **Tensor Networks with Cycles: PEPS, MERA and**
**Honey-Comb Lattice (HCL)**


An important issue in tensor networks is the rank-complexity trade-off in
the design. Namely, the main idea behind TNs is to dramatically reduce
computational cost and provide distributed storage and computation
through low-rank TN approximation. However, the TT/HT ranks, _R_ _n_,
of 3rd-order core tensors sometimes increase rapidly with the order of a
data tensor and/or increase of a desired approximation accuracy, for any
choice of a tree of tensor network. The ranks can be often kept under
control through hierarchical two-dimensional TT models called the PEPS
(Projected Entangled Pair States [8] ) and PEPO (Projected Entangled Pair
Operators) tensor networks, which contain cycles, as shown in Figure 2.21.
In the PEPS and PEPO, the ranks are kept considerably smaller at a cost
of employing 5th- or even 6th-order core tensors and the associated higher
computational complexity with respect to the order [76,184,214].
Even with the PEPS/PEPO architectures, for very high-order tensors,
the ranks (internal size of cores) may increase rapidly with an increase in
the desired accuracy of approximation. For further control of the ranks,
alternative tensor networks can be employed, such as: (1) the HoneyComb Lattice (HCL) which uses 3rd-order cores, and (2) the Multi-scale
Entanglement Renormalization Ansatz (MERA) which consist of both 3rdand 4th-order core tensors (see Figure 2.22) [83, 143, 156]. The ranks are
often kept considerably small through special architectures of such TNs,
at the expense of higher computational complexity with respect to tensor


8 An “entangled pair state” is a tensor that cannot be represented as an elementary rank1 tensor. The state is called “projected” because it is not a real physical state but a projection
onto some subspace. The term “pair” refers to the entanglement being considered only for
maximally entangled state pairs [94,156].


52


(a) (b)


Figure 2.22: Examples of TN architectures with loops. (a) Honey-Comb
Lattice (HCL) for a 16th-order tensor. (b) MERA for a 32th-order tensor.


contractions due to many cycles.
Compared with the PEPS and PEPO formats, the main advantage of the
MERA formats is that the order and size of each core tensor in the internal
tensor network structure is often much smaller, which dramatically reduces
the number of free parameters and provides more efficient distributed
storage of huge-scale data tensors. Moreover, TNs with cycles, especially
the MERA tensor network allow us to model more complex functions and
interactions between variables.


**2.6** **Concatenated (Distributed) Representation of TT**
**Networks**


Complexity of algorithms for computation (contraction) on tensor
networks typically scales polynomially with the rank, _R_ _n_, or size, _I_ _n_, of
the core tensors, so that the computations quickly become intractable with
the increase in _R_ _n_ . A step towards reducing storage and computational
requirements would be therefore to reduce the size (volume) of core tensors
by increasing their number through distributed tensor networks (DTNs),
as illustrated in Figure 2.22. The underpinning idea is that each core
tensor in an original TN is replaced by another TN (see Figure 2.23 for TT
networks), resulting in a distributed TN in which only some core tensors
are associated with physical (natural) modes of the original data tensor

[100]. A DTN consists of two kinds of relatively small-size cores (nodes),


53


Figure 2.23: Graphical representation of a large-scale data tensor via its
TT model (top panel), the PEPS model of the TT (third panel), and its
transformation to a distributed 2D (second from bottom panel) and 3D
(bottom panel) tensor train networks.


54


Table 2.2: Links between tensor networks (TNs) and graphical models used
in Machine Learning (ML) and Statistics. The corresponding categories are
not exactly the same, but have general analogies.






|Tensor Networks|Neural Networks and Graphical Models in<br>ML/Statistics|
|---|---|
|TT/MPS<br>HT/TTNS<br>PEPS<br>MERA<br>ALS, DMRG/MALS<br>Algorithms|Hidden Markov Models (HMM)<br>Deep Learning Neural Networks, Gaussian<br>Mixture Model (GMM)<br>Markov Random Field (MRF), Conditional<br>Random Field (CRF)<br>Wavelets, Deep Belief Networks (DBN)<br>Forward-Backward<br>Algorithms,<br>Block<br>Nonlinear Gauss-Seidel Methods|



internal nodes which have no free edges and external nodes which have
free edges representing natural (physical) indices of a data tensor.
The obvious advantage of DTNs is that the size of each core tensor in the
internal tensor network structure is usually much smaller than the size of
the initial core tensor; this allows for a better management of distributed
storage, and often in the reduction of the total number of network
parameters through distributed computing. However, compared to initial
tree structures, the contraction of the resulting distributed tensor network
becomes much more difficult because of the loops in the architecture.


**2.7** **Links** **between** **TNs** **and** **Machine** **Learning**
**Models**


Table 2.2 summarizes the conceptual connections of tensor networks with
graphical and neural network models in machine learning and statistics

[44, 45, 52, 53, 77, 110, 146, 154, 226]. More research is needed to establish
deeper and more precise relationships.


55


**2.8** **Changing the Structure of Tensor Networks**


An advantage of the graphical (graph) representation of tensor networks is
that the graphs allow us to perform complex mathematical operations on
core tensors in an intuitive and easy to understand way, without the need
to resort to complicated mathematical expressions. Another important
advantage is the ability to modify (optimize) the topology of a TN, while
keeping the original physical modes intact. The so optimized topologies
yield simplified or more convenient graphical representations of a higherorder data tensor and facilitate practical applications [94, 100, 230]. In
particular:


_•_
A change in topology to a HT/TT tree structure provides reduced
computational complexity, through sequential contractions of core
tensors and enhanced stability of the corresponding algorithms;


_•_
Topology of TNs with cycles can be modified so as to completely
eliminate the cycles or to reduce their number;


_•_
Even for vastly diverse original data tensors, topology modifications
may produce identical or similar TN structures which make it easier
to compare and jointly analyze block of interconnected data tensors.
This provides opportunity to perform joint group (linked) analysis of
tensors by decomposing them to TNs.


It is important to note that, due to the iterative way in which tensor
contractions are performed, the computational requirements associated
with tensor contractions are usually much smaller for tree-structured
networks than for tensor networks containing many cycles. Therefore,
for stable computations, it is advantageous to transform a tensor network
with cycles into a tree structure.


**Tensor Network transformations.** In order to modify tensor network
structures, we may perform sequential core contractions, followed by the
unfolding of these contracted tensors into matrices, matrix factorizations
(typically truncated SVD) and finally reshaping of such matrices back into
new core tensors, as illustrated in Figures 2.24.
The example in Figure 2.24(a) shows that, in the first step a contraction
of two core tensors, **G** [(] [1] [)] P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[R]_ and **G** [(] [2] [)] P **R** _[R]_ [ˆ] _[I]_ [3] [ˆ] _[I]_ [4], is performed to
give the tensor


**G** [(] [1,2] [)] = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] [ˆ] _[I]_ [4], (2.23)


56


(a)


(b)



Contraction Matricization SVD Reshaping

_I_ 1 _I_ 4





_I_ 2



_I_ 1





**U**


**V**



_I_ 4



**G** [(1,2)]
_I_ 4 _I_ _I_ 4





**G** (1,2)





_I_ 4


(1)


**G** (2)


_I_ 3



_I_ 3 _I_ 2



_I_ 2 _I_ 3 _I_ 2 _I_ 3





_I_ 2 _I_ 3





Figure 2.24: Illustration of basic transformations on a tensor network. (a)
Contraction, matricization, matrix factorization (SVD) and reshaping of
matrices back into tensors. (b) Transformation of a Honey-Comb lattice
into a Tensor Chain (TC) via tensor contractions and the SVD.


with entries _g_ _i_ [(] 1 [1,2], _i_ 2 [)], _i_ 3, _i_ 4 [=] [ ř] _r_ _[R]_ = 1 _[g]_ _i_ [(] 1 [1], [)] _i_ 2, _r_ _[g]_ _r_ [(], [2] _i_ [)] 3, _i_ 4 [. In the next step, the tensor] **[ G]** [(] [1,2] [)] [ is]
transformed into a matrix via matricization, followed by a low-rank matrix
factorization using the SVD, to give

**G** _i_ [(] 1 [1,2] _i_ 4, [)] _i_ 2 _i_ 3 [–] **[ USV]** [T] [ P] **[ R]** _[I]_ [1] _[I]_ [4] [ˆ] _[I]_ [2] _[I]_ [3] [.] (2.24)



In the final step, the factor matrices, **US** [1/2] P **R** _[I]_ [1] _[I]_ [4] [ˆ] _[R]_ [1] and **VS** [1/2] P **R** _[R]_ [1] [ˆ] _[I]_ [2] _[I]_ [3],
are reshaped into new core tensors, **G** 1 ( 1 ) P **R** _I_ 1 ˆ _R_ 1 ˆ _I_ 4 and **G** 1 ( 2 ) P **R** _R_ 1 ˆ _I_ 2 ˆ _I_ 3 .



are reshaped into new core tensors, **G** 1 ( 1 ) P **R** _I_ 1 ˆ _R_ 1 ˆ _I_ 4 and **G** 1 ( 2 ) P **R** _R_ 1 ˆ _I_ 2 ˆ _I_ 3 .

The above tensor transformation procedure is quite general, and is
applied in Figure 2.24(b) to transform a Honey-Comb lattice into a tensor
chain (TC), while Figure 2.25 illustrates the conversion of a tensor chain
(TC) into TT/MPS with OBC.



1 ( 1 ) P **R** _I_ 1 ˆ _R_ 1 ˆ _I_ 4 and **G**



57


**G**





(1)
**G**


(4)
**G**





**G** (3)





**G** (1,2)









**G** (3)


(3)


**G** (2)





(1)
**G**











**G** (4) 4 _R_ 2 **G** (3) **G** (3,4) _R R_ 2 4









Figure 2.25: Transformation of the closed-loop Tensor Chain (TC) into the
open-loop Tensor Train (TT). This is achieved by suitable contractions,
reshaping and decompositions of core tensors.


To convert a TC into TT/MPS, in the first step, we perform a contraction
of two tensors, **G** [(] [1] [)] P **R** _[I]_ [1] [ˆ] _[R]_ [4] [ˆ] _[R]_ [1] and **G** [(] [2] [)] P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[I]_ [2], as


**G** [(] [1,2] [)] = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] P **R** _[I]_ [1] [ˆ] _[R]_ [4] [ˆ] _[R]_ [2] [ˆ] _[I]_ [2],


for which the entries _g_ _i_ [(] 1 [1,2], _r_ 4 [)], _r_ 2, _i_ 2 [=] [ ř] _r_ _[R]_ 1 [1] = 1 _[g]_ _i_ [(] 1 [1], [)] _r_ 4, _r_ 1 _[g]_ _r_ [(] 1 [2], [)] _r_ 2, _i_ 2 [. In the next step, the]

tensor **G** [(] [1,2] [)] is transformed into a matrix, followed by a truncated SVD


**G** [(] [1,2] [)] – **USV** [T] P **R** _[I]_ [1] [ˆ] _[R]_ [4] _[R]_ [2] _[I]_ [2] .
( 1 )



Finally, the matrices, **U** P **R** _[I]_ [1] [ˆ] _[R]_ 1 [1] and **VS** P **R** _[R]_ 1 [1] [ˆ] _[R]_ [4] _[R]_ [2] _[I]_ [2], are reshaped back
into the core tensors, **G** 1 ( 1 ) = **U** P **R** 1ˆ _I_ 1 ˆ _R_ 11 and **G** 1 ( 2 ) P **R** _R_ 11 [ˆ] _[R]_ [4] [ˆ] _[R]_ [2] [ˆ] _[I]_ [2] .



into the core tensors, **G** 1 ( 1 ) = **U** P **R** 1ˆ _I_ 1 ˆ _R_ 11 and **G** 1 ( 2 ) P **R** _R_ 11 [ˆ] _[R]_ [4] [ˆ] _[R]_ [2] [ˆ] _[I]_ [2] .

The procedure is repeated all over again for different pairs of cores, as
illustrated in Figure 2.25.



1 ( 1 ) = **U** P **R** 1ˆ _I_ 1 ˆ _R_ 11 and **G** 1



58


**A** 1 **b** (3)1 + **A** ~~2~~ **b** (3)2 + [. . .] + **A** ~~_R_~~



**b** (3)
_R_



**b** (1)1 **b** (2)1 **b** (1)2 **b** (2)2

|Col1|Col2|Col3|
|---|---|---|
||||
||||
|**X**|**X**||



**X** **A** **A** **A**
~~1~~ 2 ~~_R_~~


+ + . . . +



**b** (1) **b** (2) _R_
_R_



( **b** (1)1 **b** (2)1 **b** (3)1 ) ( **b** (1)2 **b** (2)2 **b** (3)2 ) ( **b** (1) _R_ **b** (2) _R_ **b** (3) _R_ )


Figure 2.26: Block term decomposition (BTD) of a 6th-order block tensor,

to yield **X** = [ř] _r_ _[R]_ = 1 **[A]** _r_ [˝] **b** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝ **b** _r_ [(] [3] [)] (top panel), for more detail see
� �

[57, 193]. BTD in the tensor network notation (bottom panel). Therefore,
the 6th-order tensor **X** is approximately represented as a sum of _R_ terms,
each of which is an outer product of a 3rd-order tensor, **A** _r_, and another a
3rd-order, rank-1 tensor, **b** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝ **b** _r_ [(] [3] [)] (in dashed circle), which itself is
an outer product of three vectors.


**2.9** **Generalized Tensor Network Formats**


The fundamental TNs considered so far assume that the links between
the cores are expressed by tensor contractions. In general, links between
the core tensors (or tensor sub-networks) can also be expressed via other
mathematical linear/multilinear or nonlinear operators, such as the outer
(tensor) product, Kronecker product, Hadamard product and convolution
operator. For example, the use of the outer product leads to Block Term
Decomposition (BTD) [57,58,61,193] and use the Kronecker products yields
to the Kronecker Tensor Decomposition (KTD) [174, 175, 178]. Block term
decompositions (BTD) are closely related to constrained Tucker formats
(with a sparse block Tucker core) and the Hierarchical Outer Product
Tensor Approximation (HOPTA), which be employed for very high-order
data tensors [39].
Figure 2.26 illustrates such a BTD model for a 6th-order tensor, where
the links between the components are expressed via outer products, while
Figure 2.27 shows a more flexible Hierarchical Outer Product Tensor
Approximation (HOPTA) model suitable for very high-order tensors.


59


Figure 2.27: Conceptual model of the HOPTA generalized tensor network,
illustrated for data tensors of different orders. For simplicity, we use
the standard outer (tensor) products, but conceptually nonlinear outer
products (see Eq. (2.25) and other tensor product operators (Kronecker,
Hadamard) can also be employed. Each component (core tensor), **A** _r_, **B** _r_
and/or **C** _r_, can be further hierarchically decomposed using suitable outer
products, so that the HOPTA models can be applied to very high-order

tensors.


60


Observe that the fundamental operator in the HOPTA generalized
tensor networks is outer (tensor) product, which for two tensors **A** P
**R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **B** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[M]_, of arbitrary orders _N_ and _M_, is defined as
an ( _N_ + _M_ ) th-order tensor **C** = **A** ˝ **B** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[M]_, with entries
_c_ _i_ 1,..., _i_ _N_, _j_ 1,..., _j_ _M_ = _a_ _i_ 1,..., _i_ _N_ _b_ _j_ 1,..., _j_ _M_ . This standard outer product of two tensors
can be generalized to a nonlinear outer product as follows
� **A** ˝ _f_ **B** � _i_ 1,..., _i_ _N_, _j_ 1,..., _J_ _M_ [=] _[ f]_ � _a_ _i_ 1,..., _i_ _N_, _b_ _j_ 1,..., _j_ _M_ �, (2.25)


where _f_ ( ¨, ¨ ) is a suitably designed nonlinear function with associative
and commutative properties. In a similar way, we can define other
nonlinear tensor products, for example, Hadamard, Kronecker or Khatri–
Rao products and employ them in generalized nonlinear tensor networks.
The advantage of the HOPTA model over other TN models is its flexibility
and the ability to model more complex data structures by approximating
very high-order tensors through a relatively small number of low-order

cores.
The BTD, and KTD models can be expressed mathematically, for
example, in simple nested (hierarchical) forms, given by



�



_i_ 1,..., _i_ _N_, _j_ 1,..., _J_ _M_ [=] _[ f]_ � _a_ _i_ 1,..., _i_ _N_, _b_ _j_ 1,..., _j_ _M_ �, (2.25)



BTD : **X** –


KTD : **X** [˜] –



_R_
ÿ ( **A** _r_ ˝ **B** _r_ ), (2.26)

_r_ = 1


_R_
ÿ ( **A** _r_ b **B** _r_ ), (2.27)

_r_ = 1



where, e.g., for BTD, each factor tensor can be represented recursively as
**A** _r_ – [ř] _r_ _[R]_ 1 [1] = 1 [(] **[A]** _r_ [(] 1 [1] [)] [˝] **[ B]** _r_ [(] 1 [1] [)] [)] [ or] **[ B]** _r_ [–][ ř] _r_ _[R]_ 2 [2] = 1 **[A]** _r_ [(] 2 [2] [)] [˝] **[ B]** _r_ [(] 2 [2] [)] [.]
Note that the 2 _N_ th-order subtensors, **A** _r_ ˝ **B** _r_ and **A** _r_ b **B** _r_, have the same
elements, just arranged differently. For example, if **X** = **A** ˝ **B** and **X** [1] =
**A** b **B**, where **A** P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[N]_ and **B** P **R** _[K]_ [1] [ˆ] _[K]_ [2] [ˆ¨¨¨ˆ] _[K]_ _[N]_, then
_x_ _j_ 1, _j_ 2,..., _j_ _N_, _k_ 1, _k_ 2,..., _k_ _N_ = _x_ _k_ [1] 1 + _K_ 1 ( _j_ 1 ´1 ),..., _k_ _N_ + _K_ _N_ ( _j_ _N_ ´1 ) [.]
The definition of the tensor Kronecker product in the KTD model
assumes that both core tensors, **A** _r_ and **B** _r_, have the same order. This is
not a limitation, given that vectors and matrices can also be treated as
tensors, e.g, a matrix of dimension _I_ ˆ _J_ as is also a 3rd-order tensor of
dimension _I_ ˆ _J_ ˆ 1. In fact, from the BTD/KTD models, many existing
and new TDs/TNs can be derived by changing the structure and orders of
factor tensors, **A** _r_ and **B** _r_ . For example:


_•_ If **A** _r_ are rank-1 tensors of size _I_ 1 ˆ _I_ 2 ˆ ¨ ¨ ¨ ˆ _I_ _N_, and **B** _r_ are scalars,
@ _r_, then (2.27) represents the rank- _R_ CP decomposition;


61


_•_ If **A** _r_ are rank- _L_ _r_ tensors of size _I_ 1 ˆ _I_ 2 ˆ ¨ ¨ ¨ ˆ _I_ _R_ ˆ 1 ˆ ¨ ¨ ¨ ˆ 1, in the
Kruskal (CP) format, and **B** _r_ are rank-1 tensors of size 1 ˆ ¨ ¨ ¨ ˆ 1 ˆ
_I_ _R_ + 1 ˆ ¨ ¨ ¨ ˆ _I_ _N_, @ _r_, then (2.27) expresses the rank-( _L_ _r_ ˝ 1) BTD;


_•_ If **A** _r_ and **B** _r_ are expressed by KTDs, we arrive at the Nested
Kronecker Tensor Decomposition (NKTD), a special case of which is
the Tensor Train (TT) decomposition. Therefore, the BTD model in
(2.27) can also be used for recursive TT-decompositions.


The generalized tensor network approach caters for a large variety of
tensor decomposition models, which may find applications in scientific
computing, signal processing or deep learning (see, eg., [37,39,45,58,177]).
In this monograph, we will mostly focus on the more established
Tucker and TT decompositions (and some of their extensions), due to their
conceptual simplicity, availability of stable and efficient algorithms for their
computation and the possibility to naturally extend these models to more
complex tensor networks. In other words, the Tucker and TT models are
considered here as simplest prototypes, which can then serve as building
blocks for more sophisticated tensor networks.


62


**Chapter 3**


**Constrained Tensor**

**Decompositions: From**
**Two-way to Multiway**
**Component Analysis**


The component analysis (CA) framework usually refers to the application
of constrained matrix factorization techniques to observed mixed signals in
order to extract components with specific properties and/or estimate the
mixing matrix [40, 43, 47, 55, 103]. In the machine learning practice, to aid
the well-posedness and uniqueness of the problem, component analysis
methods exploit prior knowledge about the statistics and diversities of
latent variables (hidden sources) within the data. Here, by the diversities,
we refer to different characteristics, features or morphology of latent
variables which allow us to extract the desired components or features, for
example, sparse or statistically independent components.


**3.1** **Constrained Low-Rank Matrix Factorizations**


Two-way Component Analysis (2-way CA), in its simplest form, can be
formulated as a constrained matrix factorization of typically low-rank, in
the form



**X** = **AΛB** [T] + **E** =



_R_


_λ_ _r_ **a** _r_ ˝ **b** _r_ + **E** =

ÿ

_r_ = 1


63



_R_
ÿ _λ_ _r_ **a** _r_ **b** [T] _r_ [+] **[ E]** [,] (3.1)

_r_ = 1


where **Λ** = diag ( _λ_ 1, . . ., _λ_ _R_ ) is an optional diagonal scaling matrix.
The potential constraints imposed on the factor matrices, **A** and/or **B**,
include orthogonality, sparsity, statistical independence, nonnegativity or
smoothness. In the bilinear 2-way CA in (3.1), **X** P **R** _[I]_ [ˆ] _[J]_ is a known
matrix of observed data, **E** P **R** _[I]_ [ˆ] _[J]_ represents residuals or noise, **A** =

[ **a** 1, **a** 2, . . ., **a** _R_ ] P **R** _[I]_ [ˆ] _[R]_ is the unknown mixing matrix with _R_ basis vectors
**a** _r_ P **R** _[I]_, and depending on application, **B** = [ **b** 1, **b** 2, . . ., **b** _R_ ] P **R** _[J]_ [ˆ] _[R]_, is
the matrix of unknown components, factors, latent variables, or hidden
sources, represented by vectors **b** _r_ P **R** _[J]_ (see Figure 3.2).
It should be noted that 2-way CA has an inherent symmetry. Indeed,
Eq. (3.1) could also be written as **X** [T] « **BA** [T], thus interchanging the roles of
sources and mixing process.
Algorithmic approaches to 2-way (matrix) component analysis are well
established, and include Principal Component Analysis (PCA), Robust
PCA (RPCA), Independent Component Analysis (ICA), Nonnegative
Matrix Factorization (NMF), Sparse Component Analysis (SCA) and
Smooth Component Analysis (SmCA) [6, 24, 43, 47, 109, 228]. These
techniques have become standard tools in blind source separation (BSS),
feature extraction, and classification paradigms. The columns of the matrix
**B**, which represent different latent components, are then determined by
specific chosen constraints and should be, for example, (i) as statistically
mutually independent as possible for ICA; (ii) as sparse as possible for
SCA; (iii) as smooth as possible for SmCA; (iv) take only nonnegative
values for NMF.
Singular value decomposition (SVD) of the data matrix **X** P **R** _[I]_ [ˆ] _[J]_ is a
special, very important, case of the factorization in Eq. (3.1), and is given
by



_R_
ÿ _σ_ _r_ **u** _r_ **v** [T] _r_ [,] (3.2)

_r_ = 1



**X** = **USV** [T] =



_R_


_σ_ _r_ **u** _r_ ˝ **v** _r_ =

ÿ

_r_ = 1



where **U** P **R** _[I]_ [ˆ] _[R]_ and **V** P **R** _[J]_ [ˆ] _[R]_ are column-wise orthogonal matrices and
**S** P **R** _[R]_ [ˆ] _[R]_ is a diagonal matrix containing only nonnegative singular values
_σ_ _r_ in a monotonically non-increasing order.
According to the well known Eckart–Young theorem, the truncated
SVD provides the optimal, in the least-squares (LS) sense, low-rank
matrix approximation [1] . The SVD, therefore, forms the backbone of
low-rank matrix approximations (and consequently low-rank tensor
approximations).


1 [145] has generalized this optimality to arbitrary unitarily invariant norms.


64


Another virtue of component analysis comes from the ability to perform
simultaneous matrix factorizations


**X** _k_ « **A** _k_ **B** [T] _k_ [,] ( _k_ = 1, 2, . . ., _K_ ), (3.3)


on several data matrices, **X** _k_, which represent linked datasets, subject to
various constraints imposed on linked (interrelated) component (factor)
matrices. In the case of orthogonality or statistical independence
constraints, the problem in (3.3) can be related to models of group
PCA/ICA through suitable pre-processing, dimensionality reduction and
post-processing procedures [38, 75, 88, 191, 239]. The terms “group
component analysis” and “joint multi-block data analysis” are used
interchangeably to refer to methods which aim to identify links
(correlations, similarities) between hidden components in data. In other
words, _the objective of group component analysis is to analyze the correlation,_
_variability, and consistency of the latent components across multi-block datasets_ .
The field of 2-way CA is maturing and has generated efficient algorithms
for 2-way component analysis, especially for sparse/functional PCA/SVD,
ICA, NMF and SCA [6,40,47,103,236].
The rapidly emerging field of tensor decompositions is the next
important step which naturally generalizes 2-way CA/BSS models and
algorithms. Tensors, by virtue of multilinear algebra, offer enhanced
flexibility in CA, in the sense that not all components need to be
statistically independent, and can be instead smooth, sparse, and/or nonnegative (e.g., spectral components). Furthermore, additional constraints
can be used to reflect physical properties and/or diversities of spatial
distributions, spectral and temporal patterns. We proceed to show how
constrained matrix factorizations or 2-way CA models can be extended
to multilinear models using tensor decompositions, such as the Canonical
Polyadic (CP) and the Tucker decompositions, as illustrated in Figures 3.1,
3.2 and 3.3.


**3.2** **The CP Format**


The CP decomposition (also called the CANDECOMP, PARAFAC, or
Canonical Polyadic decomposition) decomposes an _N_ th-order tensor, **X** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, into a linear combination of terms, **b** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝¨ ¨ ¨˝ **b** _r_ [(] _[N]_ [)], which


65


are rank-1 tensors, and is given by [29,95,96]



**X** –



_R_
ÿ _λ_ _r_ **b** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **b** _r_ [(] _[N]_ [)]

_r_ = 1



= **Λ** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)]

= � **Λ** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �,



(3.4)



where _λ_ _r_ are non-zero entries of the diagonal core tensor **Λ** P **R** _[R]_ [ˆ] _[R]_ [ˆ¨¨¨ˆ] _[R]_

and **B** [(] _[n]_ [)] = [ **b** 1 [(] _[n]_ [)] [,] **[ b]** 2 [(] _[n]_ [)] [, . . .,] **[ b]** [(] _R_ _[n]_ [)] []] [ P] **[ R]** _[I]_ _[n]_ [ˆ] _[R]_ [ are factor matrices (see Figure 3.1]
and Figure 3.2).
Via the Khatri–Rao products (see Table 2.1), the CP decomposition can
be equivalently expressed in a matrix/vector form as


**X** ( _n_ ) – **B** [(] _[n]_ [)] **Λ** ( **B** [(] _[N]_ [)] d ¨ ¨ ¨ d **B** [(] _[n]_ [+] [1] [)] d **B** [(] _[n]_ [´][1] [)] d ¨ ¨ ¨ d **B** [(] [1] [)] ) [T] (3.5)

= **B** [(] _[n]_ [)] **Λ** ( **B** [(] [1] [)] d _L_ ¨ ¨ ¨ d _L_ **B** [(] _[n]_ [´][1] [)] d _L_ **B** [(] _[n]_ [+] [1] [)] d _L_ ¨ ¨ ¨ d _L_ **B** [(] _[N]_ [)] ) [T]


and


vec ( **X** ) – [ **B** [(] _[N]_ [)] d **B** [(] _[N]_ [´][1] [)] d ¨ ¨ ¨ d **B** [(] [1] [)] ] _**λ**_ (3.6)

– [ **B** [(] [1] [)] d _L_ **B** [(] [2] [)] d _L_ ¨ ¨ ¨ d _L_ **B** [(] _[N]_ [)] ] _**λ**_,


where _**λ**_ = [ _λ_ 1, _λ_ 2, . . ., _λ_ _R_ ] [T] and **Λ** = diag ( _λ_ 1, . . ., _λ_ _R_ ) is a diagonal matrix.
The rank of a tensor **X** is defined as the smallest _R_ for which the CP
decomposition in (3.4) holds exactly.


**Algorithms to compute CP decomposition.** In real world applications, the
signals of interest are corrupted by noise, so that the CP decomposition is
rarely exact and has to be estimated by minimizing a suitable cost function.
Such cost functions are typically of the Least-Squares (LS) type, in the form
of the Frobenius norm


_J_ 2 ( **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] ) = } **X** ´ � **Λ** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �} [2] _F_ [,] (3.7)


or Least Absolute Error (LAE) criteria [217]


_J_ 1 ( **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] ) = } **X** ´ � **Λ** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �} 1 . (3.8)


The Alternating Least Squares (ALS) based algorithms minimize the
cost function iteratively by individually optimizing each component (factor
matrix, **B** [(] _[n]_ [)] )), while keeping the other component matrices fixed [95,119].


66


(a) Standard block diagram for CP decomposition of a 3rd-order tensor































(b) CP decomposition for a 4th-order tensor in the tensor network notation































Figure 3.1: Representations of the CP decomposition. The objective of
the CP decomposition is to estimate the factor matrices **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ and
scaling coefficients t _λ_ 1, _λ_ 1, . . ., _λ_ _R_ u. (a) The CP decomposition of a 3rdorder tensor in the form, **X** – **Λ** ˆ 1 **A** ˆ 2 **B** ˆ 3 **C** = [ř] _r_ _[R]_ = 1 _[λ]_ _[r]_ **[ a]** _[r]_ [ ˝] **[ b]** _[r]_ [ ˝] **[ c]** _[r]_ [ =]
**G** _c_ ˆ 1 **A** ˆ 2 **B**, with **G** _c_ = **Λ** ˆ 3 **C** . (b) The CP decomposition for a 4th-order
tensor in the form **X** – **Λ** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ˆ 3 **B** [(] [3] [)] ˆ 4 **B** [(] [4] [)] = [ř] _r_ _[R]_ = 1 _[λ]_ _[r]_ **[ b]** _r_ [(] [1] [)] ˝
**b** _r_ [(] [2] [)] ˝ **b** _r_ [(] [3] [)] ˝ **b** _r_ [(] [4] [)] [.]


67


Figure 3.2: Analogy between a low-rank matrix factorization, **X** – **AΛB** [T] =
_R_
ř _r_ = 1 _[λ]_ _[r]_ **a** _r_ ˝ **b** _r_ (top), and a simple low-rank tensor factorization (CP
decomposition), **X** – **Λ** ˆ 1 **A** ˆ 2 **B** ˆ 3 **C** = [ř] _r_ _[R]_ = 1 _[λ]_ _[r]_ **[ a]** _[r]_ [ ˝] **[ b]** _[r]_ [ ˝] **[ c]** _[r]_ [ (bottom).]


To illustrate the ALS principle, assume that the diagonal matrix **Λ**
has been absorbed into one of the component matrices; then, by taking
advantage of the Khatri–Rao structure in Eq. (3.5), the component matrices,
**B** [(] _[n]_ [)], can be updated sequentially as



:

. (3.9)
�



**B** [(] _[n]_ [)] Ð **X** ( _n_ )



ä
� _k_ ‰ _n_



ä **B** [(] _[k]_ [)] æ

_k_ ‰ _n_ �� _k_ ‰ _n_



æ ( **B** [(] _[k]_ [)] [ T] **B** [(] _[k]_ [)] )

_k_ ‰ _n_



The main challenge (or bottleneck) in implementing ALS and Gradient
Decent (GD) techniques for CP decomposition lies therefore in multiplying
a matricized tensor and Khatri–Rao product (of factor matrices) [35, 171]
and in the computation of the pseudo-inverse of ( _R_ ˆ _R_ ) matrices (for the
basic ALS see Algorithm 1).


The ALS approach is attractive for its simplicity, and often provides
satisfactory performance for well defined problems with high SNRs
and well separated and non-collinear components. For ill-conditioned
problems, advanced algorithms are required which typically exploit
the rank-1 structure of the terms within CP decomposition to perform
efficient computation and storage of the Jacobian and Hessian of the cost
function [172, 176, 193]. Implementation of parallel ALS algorithm over
distributed memory for very large-scale tensors was proposed in [35,108].


**Multiple random projections, tensor sketching and Giga-Tensor.** Most of
the existing algorithms for the computation of CP decomposition are based


68


**Algorithm 1** : **Basic ALS for the CP decomposition of a 3rd-order**

**tensor**

**Input:** Data tensor **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_ and rank _R_
**Output:** Factor matrices **A** P **R** _[I]_ [ˆ] _[R]_, **B** P **R** _[J]_ [ˆ] _[R]_, **C** P **R** _[K]_ [ˆ] _[R]_, and scaling
vector _**λ**_ P **R** _[R]_


1: Initialize **A**, **B**, **C**
2: **while** not converged or iteration limit is not reached **do**
3: **A** Ð **X** ( 1 ) ( **C** d **B** )( **C** [T] **C** f **B** [T] **B** ) [:]

4: Normalize column vectors of **A** to unit length (by computing the
norm of each column vector and dividing each element of a
vector by its norm)
5: **B** Ð **X** ( 2 ) ( **C** d **A** )( **C** [T] **C** f **A** [T] **A** ) [:]

6: Normalize column vectors of **B** to unit length
7: **C** Ð **X** ( 3 ) ( **B** d **A** )( **B** [T] **B** f **C** [T] **C** ) [:]

8: Normalize column vectors of **C** to unit length,
store the norms in vector _**λ**_

9: **end while**

10: **return A**, **B**, **C** and _**λ**_ .


on the ALS or GD approaches, however, these can be too computationally
expensive for huge tensors. Indeed, algorithms for tensor decompositions
have generally not yet reached the level of maturity and efficiency of lowrank matrix factorization (LRMF) methods. In order to employ efficient
LRMF algorithms to tensors, we need to either: (i) reshape the tensor at
hand into a set of matrices using traditional matricizations, (ii) employ
reduced randomized unfolding matrices, or (iii) perform suitable random
multiple projections of a data tensor onto lower-dimensional subspaces.
The principles of the approaches (i) and (ii) are self-evident, while the
approach (iii) employs a multilinear product of an _N_ th-order tensor and
( _N_ ´ 2 ) random vectors, which are either chosen uniformly from a unit
sphere or assumed to be i.i.d. Gaussian vectors [126].
For example, for a 3rd-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3], we can use the set
of random projections, **X** ¯3 = **X** ˆ [¯] 3 _**ω**_ 3 P **R** _[I]_ [1] [ˆ] _[I]_ [2], **X** ¯2 = **X** ˆ [¯] 2 _**ω**_ 2 P **R** _[I]_ [1] [ˆ] _[I]_ [3]
and **X** ¯1 = **X** ˆ [¯] 1 _**ω**_ 1 P **R** _[I]_ [2] [ˆ] _[I]_ [3], where the vectors _**ω**_ _n_ P **R** _[I]_ _[n]_, _n_ = 1, 2, 3,
are suitably chosen random vectors. Note that random projections in such
a case are non-typical – instead of using projections for dimensionality
reduction, they are used to reduce a tensor of any order to matrices and
consequently transform the CP decomposition problem to constrained
matrix factorizations problem, which can be solved via simultaneous (joint)
matrix diagonalization [31, 56]. It was shown that even a small number of


69


random projections, such as _O_ ( log _R_ ) is sufficient to preserve the spectral
information in a tensor. This mitigates the problem of the dependence on
the eigen-gap [2] that plagued earlier tensor-to-matrix reductions. Although
a uniform random sampling may experience problems for tensors with
spiky elements, it often outperforms the standard CP-ALS decomposition
algorithms.
Alternative algorithms for the CP decomposition of huge-scale tensors
include tensor sketching – a random mapping technique, which exploits
kernel methods and regression [168, 223], and the class of distributed
algorithms such as the DFacTo [35] and the GigaTensor which is based on
Hadoop / MapReduce paradigm [106].


**Constraints.** Under rather mild conditions, the CP decomposition is
generally unique by itself [125, 188]. It does not require additional
constraints on the factor matrices to achieve uniqueness, which makes
it a powerful and useful tool for tensor factroization. Of course, if
the components in one or more modes are known to possess some
properties, e.g., they are known to be nonnegative, orthogonal,
statistically independent or sparse, such prior knowledge may be
incorporated into the algorithms to compute CPD and at the same time
relax uniqueness conditions. More importantly, such constraints may
enhance the accuracy and stability of CP decomposition algorithms
and also facilitate better physical interpretability of the extracted
components [65,117,134,187,195,234].


**Applications.** The CP decomposition has already been established as an
advanced tool for blind signal separation in vastly diverse branches of
signal processing and machine learning [1, 3, 119, 147, 189, 207, 223]. It is
also routinely used in exploratory data analysis, where the rank-1 terms
capture essential properties of dynamically complex datasets, while in
wireless communication systems, signals transmitted by different users
correspond to rank-1 terms in the case of line-of-sight propagation and
therefore admit analysis in the CP format. Another potential application
is in harmonic retrieval and direction of arrival problems, where real or
complex exponentials have rank-1 structures, for which the use of CP
decomposition is quite natural [185,186,194].


2 In linear algebra, the eigen-gap of a linear operator is the difference between two
successive eigenvalues, where the eigenvalues are sorted in an ascending order.


70


**3.3** **The Tucker Tensor Format**


Compared to the CP decomposition, the Tucker decomposition provides
a more general factorization of an _N_ th-order tensor into a relatively small
size core tensor and factor matrices, and can be expressed as follows:



_R_ _N_
ÿ _g_ _r_ 1 _r_ 2 ¨¨¨ _r_ _N_ � **b** _r_ [(] 1 [1] [)] [˝] **[ b]** _r_ [(] 2 [2] [)] [˝ ¨ ¨ ¨ ˝] **[ b]** _r_ [(] _N_ _[N]_ [)] �

_r_ _N_ = 1



**X** –



_R_ 1


¨ ¨ ¨

ÿ

_r_ 1 = 1



= **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)]

= � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �, (3.10)


where **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ is the given data tensor, **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ¨¨¨ˆ] _[R]_ _[N]_ is
the core tensor, and **B** [(] _[n]_ [)] = [ **b** 1 [(] _[n]_ [)] [,] **[ b]** 2 [(] _[n]_ [)] [, . . .,] **[ b]** [(] _R_ _[n]_ _n_ [)] []] [ P] **[ R]** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ [ are the mode-]
_n_ factor (component) matrices, _n_ = 1, 2, . . ., _N_ (see Figure 3.3). The core
tensor (typically, _R_ _n_ ăă _I_ _n_ ) models a potentially complex pattern of mutual
interaction between the vectors in different modes. The model in (3.10) is
often referred to as the Tucker- _N_ model.
The CP and Tucker decompositions have long history. For recent
surveys and more detailed information we refer to [42,46,87,119,189].
Using the properties of the Kronecker tensor product, the Tucker- _N_
decomposition in (3.10) can be expressed in an equivalent matrix and
vector form as


**X** ( _n_ ) – **B** [(] _[n]_ [)] **G** ( _n_ ) ( **B** [(] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[n]_ [´][1] [)] b _L_ **B** [(] _[n]_ [+] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[N]_ [)] ) [T]

= **B** [(] _[n]_ [)] **G** ( _n_ ) ( **B** [(] _[N]_ [)] b ¨ ¨ ¨ b **B** [(] _[n]_ [+] [1] [)] b **B** [(] _[n]_ [´][1] [)] b ¨ ¨ ¨ b **B** [(] [1] [)] ) [T], (3.11)

**X** ă _n_ ą – ( **B** [(] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[n]_ [)] ) **G** ă _n_ ą ( **B** [(] _[n]_ [+] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[N]_ [)] ) [T]

= ( **B** [(] _[n]_ [)] b ¨ ¨ ¨ b **B** [(] [1] [)] ) **G** ă _n_ ą ( **B** [(] _[N]_ [)] b **B** [(] _[N]_ [´][1] [)] b ¨ ¨ ¨ b **B** [(] _[n]_ [+] [1] [)] ) [T],


(3.12)

vec ( **X** ) – [ **B** [(] [1] [)] b _L_ **B** [(] [2] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[N]_ [)] ] vec ( **G** )

= [ **B** [(] _[N]_ [)] b **B** [(] _[N]_ [´][1] [)] b ¨ ¨ ¨ b **B** [(] [1] [)] ] vec ( **G** ), (3.13)


where the multi-indices are ordered in a reverse lexicographic order (littleendian).
Table 3.1 and Table 3.2 summarize fundamental mathematical
representations of CP and Tucker decompositions for 3rd-order and _N_ thorder tensors.
The Tucker decomposition is said to be in an _independent Tucker format_
if all the factor matrices, **B** [(] _[n]_ [)], are full column rank, while a Tucker format


71


(a) Standard block diagrams of Tucker (top) and Tucker-CP (bottom)
decompositions for a 3rd-order tensor



**B** [(3)]



( _K_ _R_ 3 )





_K_





**B** [(2) T]



=



_r_ 1 2, _r r_, 3



_R_ 3

_R_ 2



_r_ 1 2, _r r_, 3



_I_



**X** **B** [(1)]



= **B** = _r_ 1 2, _r r_, 3 (2)

**B** [(1)] _R_ 2 _r_ 1 2, _r r_, 3 (1) **b** _r_ 2





,,





_J_



( _I R_ 1 ) ( _R_ 1 _R_ 2 _R_ 3 )



( _R_ 2 _J_ )


|r3 b (3)|Col2|
|---|---|
|3<br>|3<br>|
||_r_1<br>**b**<br>_r_2<br>**b**<br>(2)<br>(1)|



**B** [(3)]



( _K_ _R_ 3 )



**c** 1



**B** [(1)]



**c** 1


**+** [. . .]
# (





= **+** [. . .] **+**



**c** _R_



**B** [(2) T]



(b) The TN diagram for the Tucker and Tucker/CP decompositions of a 4th-order

tensor













































Figure 3.3: Illustration of the Tucker and Tucker-CP decompositions, where the
objective is to compute the factor matrices, **B** [(] _[n]_ [)], and the core tensor, **G** . (a) Tucker
decomposition of a 3rd-order tensor, **X** – **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ˆ 3 **B** [(] [3] [)] . In some
applications, the core tensor can be further approximately factorized using the
CP decomposition as **G** – [ř] _r_ _[R]_ = 1 **[a]** _[r]_ [ ˝] **[ b]** _[r]_ [ ˝] **[ c]** _[r]_ [ (bottom diagram), or alternatively]
using TT/HT decompositions. (b) Graphical representation of the Tucker-CP
decomposition for a 4th-order tensor, **X** – **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ˆ 3 **B** [(] [3] [)] ˆ 4 **B** [(] [4] [)] =
� **G** ; **B** [(] [1] [)], **B** [(] [2] [)], **B** [(] [3] [)], **B** [(] [4] [)] � – ( **Λ** ˆ 1 **A** [(] [1] [)] ˆ 2 **A** [(] [2] [)] ˆ 3 **A** [(] [3] [)] ˆ 4 **A** [(] [4] [)] ) ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ˆ 3
**B** [(] [3] [)] ˆ 4 **B** [(] [4] [)] = � **Λ** ; **B** [(] [1] [)] **A** [(] [1] [)], **B** [(] [2] [)] **A** [(] [2] [)], **B** [(] [3] [)] **A** [(] [3] [)], **B** [(] [4] [)] **A** [(] [4] [)] �.


72


is termed an _orthonormal format_, if in addition, all the factor matrices,
**B** [(] _[n]_ [)] = **U** [(] _[n]_ [)], are orthogonal. The standard Tucker model often has
orthogonal factor matrices.


**Multilinear rank.** The multilinear rank of an _N_ th-order tensor **X** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ corresponds to the _N_ -tuple ( _R_ 1, _R_ 2, . . ., _R_ _N_ ) consisting of the
dimensions of the different subspaces. If the Tucker decomposition (3.10)
holds exactly it is mathematically defined as


rank _ML_ ( **X** ) = trank ( **X** ( 1 ) ), rank ( **X** ( 2 ) ), . . ., rank ( **X** ( _N_ ) ) u, (3.14)


with **X** ( _n_ ) P **R** _[I]_ _[n]_ [ˆ] _[I]_ [1] [¨¨¨] _[I]_ _[n]_ [´][1] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ for _n_ = 1, 2, . . ., _N_ . Rank of the Tucker
decompositions can be determined using information criteria [227], or
through the number of dominant eigenvalues when an approximation
accuracy of the decomposition or a noise level is given (see Algorithm 8).
The independent Tucker format has the following important properties
if the equality in (3.10) holds exactly (see, e.g., [105] and references therein):


1. The tensor (CP) rank of any tensor, **X** = � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] � P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, and the rank of its core tensor, **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ¨¨¨ˆ] _[R]_ _[N]_, are
exactly the same, i.e.,


rank _CP_ ( **X** ) = rank _CP_ ( **G** ) . (3.15)


2. If a tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ = � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �, admits an
independent Tucker format with multilinear rank t _R_ 1, _R_ 2, . . ., _R_ _N_ u,
then



_R_ _n_ ď



_N_
ź _R_ _p_ @ _n_ . (3.16)

_p_ ‰ _n_



Moreover, without loss of generality, under the assumption _R_ 1 ď
_R_ 2 ď ¨ ¨ ¨ ď _R_ _N_, we have


_R_ 1 ď rank _CP_ ( **X** ) ď _R_ 2 _R_ 3 ¨ ¨ ¨ _R_ _N_ . (3.17)


3. If a data tensor is symmetric and admits an independent Tucker
format, **X** = � **G** ; **B**, **B**, . . ., **B** � P **R** _[I]_ [ˆ] _[I]_ [ˆ¨¨¨ˆ] _[I]_, then its core tensor, **G** P
**R** _[R]_ [ˆ] _[R]_ [ˆ¨¨¨ˆ] _[R]_, is also symmetric, with rank _CP_ ( **X** ) = rank _CP_ ( **G** ).


73


4. For the orthonormal Tucker format, that is, **X** =
� **G** ; **U** [(] [1] [)], **U** [(] [2] [)], . . ., **U** [(] _[N]_ [)] � P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with **U** [(] _[n]_ [)] [T] **U** [(] _[n]_ [)] = **I**, @ _n_,
the Frobenius norms and the Schatten _p_ -norms [3] of the data tensor, **X**
, and its core tensor, **G**, are equal, i.e.,


} **X** } _F_ = } **G** } _F_,

} **X** } _S_ _p_ = } **G** } _S_ _p_, 1 ď _p_ ă 8.


Thus, the computation of the Frobenius norms can be performed with
an _O_ ( _R_ _[N]_ ) complexity ( _R_ = maxt _R_ 1, . . ., _R_ _N_ u ), instead of the usual
order _O_ ( _I_ _[N]_ ) complexity (typically _R_ ! _I_ ).


Note that the CP decomposition can be considered as a special case
of the Tucker decomposition, whereby the cube core tensor has nonzero
elements only on the main diagonal (see Figure 3.1). In contrast to
the CP decomposition, the unconstrained Tucker decomposition is not
unique. However, constraints imposed on all factor matrices and/or core
tensor can reduce the indeterminacies inherent in CA to only column-wise
permutation and scaling, thus yielding a unique core tensor and factor
matrices [235].
The Tucker- _N_ model, in which ( _N_ ´ _K_ ) factor matrices are identity
matrices is called the Tucker- ( _K_, _N_ ) model. In the simplest scenario, for
a 3rd-order tensor **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_, the Tucker-(2,3) or simply Tucker-2 model,
can be described as [4]


**X** – **G** ˆ 1 **A** ˆ 2 **B** ˆ 3 **I** = **G** ˆ 1 **A** ˆ 2 **B**, (3.18)


or in an equivalent matrix form


**X** _k_ = **AG** _k_ **B** [T], ( _k_ = 1, 2, . . ., _K_ ), (3.19)


where **X** _k_ = **X** ( :, :, _k_ ) P **R** _[I]_ [ˆ] _[J]_ and **G** _k_ = **G** ( :, :, _k_ ) P **R** _[R]_ [1] [ˆ] _[R]_ [2] are respectively
the frontal slices of the data tensor **X** and the core tensor **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[R]_ [3],
and **A** P **R** _[I]_ [ˆ] _[R]_ [1], **B** P **R** _[J]_ [ˆ] _[R]_ [2] .


3 The Schatten _p_ -norm of an _N_ th-order tensor **X** is defined as the average of the Schatten
norms of mode- _n_ unfoldings, i.e., } **X** } _S_ _p_ = ( 1/ _N_ ) [ř] _n_ _[N]_ = 1 [}] **[X]** ( _n_ ) [}] _S_ _p_ [and][ }] **[X]** [}] _S_ _p_ [= (] [ř] _r_ _[σ]_ _r_ _[p]_ [)] [1/] _[p]_ [,]
where _σ_ _r_ is the _r_ th singular value of the matrix **X** . For _p_ = 1, the Schatten norm of a matrix
**X** is called the nuclear norm or the trace norm, while for _p_ = 0 the Schatten norm is the
rank of **X**, which can be replaced by the surrogate function log det ( **XX** [T] + _ε_ **I** ), _ε_ ą 0.
4 For a 3rd-order tensor, the Tucker-2 model is equivalent to the TT model. The case
where the factor matrices and the core tensor are non-negative is referred to as the NTD-2
(Nonnegative Tucker-2 decomposition).


74


Table 3.1: Different forms of CP and Tucker representations of a 3rdorder tensor **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_, where _**λ**_ = [ _λ_ 1, _λ_ 2, . . ., _λ_ _R_ ] [T], and **Λ** =
diagt _λ_ 1, _λ_ 2, . . ., _λ_ _R_ u.


CP Decomposition Tucker Decomposition


Scalar representation



_R_ _R_ 1
ř _λ_ _r_ _a_ _i r_ _b_ _j r_ _c_ _k r_ _x_ _ijk_ = ř

_r_ = 1 _r_ =



_R_ 3
ř _g_ _r_ 1 _r_ 2 _r_ 3 _a_ _i r_ 1 _b_ _j r_ 2 _c_ _k r_ 3

_r_ 3 = 1



_R_
_x_ _ijk_ = ř



_r_ 1 = 1



_R_ 2
ř

_r_ 2 = 1



Tensor representation, outer products



_R_ _R_ 1
ř _λ_ _r_ **a** _r_ ˝ **b** _r_ ˝ **c** _r_ **X** = ř

_r_ = 1 _r_ =



_R_ 3
ř _g_ _r_ 1 _r_ 2 _r_ 3 **a** _r_ 1 ˝ **b** _r_ 2 ˝ **c** _r_ 3

_r_ 3 = 1



_R_
**X** = ř



_r_ 1 = 1



_R_ 2
ř

_r_ 2 = 1



Tensor representation, multilinear products


**X** = **Λ** ˆ 1 **A** ˆ 2 **B** ˆ 3 **C** **X** = **G** ˆ 1 **A** ˆ 2 **B** ˆ 3 **C**
= � **Λ** ; **A**, **B**, **C** � = � **G** ; **A**, **B**, **C** �


Matrix representations


**X** ( 1 ) = **A Λ** ( **B** d _L_ **C** ) [T] **X** ( 1 ) = **A G** ( 1 ) ( **B** b _L_ **C** ) [T]

**X** ( 2 ) = **B Λ** ( **A** d _L_ **C** ) [T] **X** ( 2 ) = **B G** ( 2 ) ( **A** b _L_ **C** ) [T]

**X** ( 3 ) = **C Λ** ( **A** d _L_ **B** ) [T] **X** ( 3 ) = **C G** ( 3 ) ( **A** b _L_ **B** ) [T]


Vector representation


vec ( **X** ) = ( **A** d _L_ **B** d _L_ **C** ) _**λ**_ vec ( **X** ) = ( **A** b _L_ **B** b _L_ **C** ) vec ( **G** )


Matrix slices **X** _k_ = **X** ( :, :, _k_ )



�



**X** _k_ = **A** diag ( _λ_ 1 _c_ _k_,1, . . ., _λ_ _R_ _c_ _k_, _R_ ) **B** [T] **X** _k_ = **A**


75



_R_ 3
ř _c_ _kr_ 3 **G** ( :, :, _r_ 3 )
� _r_ 3 = 1



**B** [T]


Table 3.2: Different forms of CP and Tucker representations of an _N_ th-order
tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ .

































































76


**Generalized Tucker format and its links to TTNS model.** For high-order
tensors, **X** P **R** _[I]_ [1,1] [ˆ¨¨¨ˆ] _[I]_ [1,] _[K]_ [1] [ˆ] _[I]_ [2,1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [,] _[KN]_, the Tucker- _N_ format can be naturally
generalized by replacing the factor matrices, **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, by higher-order
tensors **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [,1] [ˆ] _[I]_ _[n]_ [,2] [ˆ¨¨¨ˆ] _[I]_ _[n]_ [,] _[Kn]_ [ˆ] _[R]_ _[n]_, to give


**X** – � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] �, (3.20)


where the entries of the data tensor are computed as



_R_ _N_
ÿ **G** ( _r_ 1, . . ., _r_ _N_ ) **B** [(] [1] [)] ( **i** 1, _r_ 1 ) ¨ ¨ ¨ **B** [(] _[N]_ [)] ( **i** _N_, _r_ _N_ ),

_r_ _N_ = 1



**X** ( **i** 1, . . ., **i** _N_ ) =



_R_ 1


¨ ¨ ¨

ÿ

_r_ 1 = 1



and **i** _n_ = ( _i_ _n_,1 _i_ _n_,2 . . . _i_ _n_, _K_ _n_ ) [128].
Furthermore, the nested (hierarchical) form of such a generalized
Tucker decomposition leads to the Tree Tensor Networks State (TTNS)
model [149] (see Figure 2.15 and Figure 2.18), with possibly a varying order
of cores, which can be formulated as


**X** = � **G** ~~1~~ ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [1] [)] �

**G** ~~1~~ = � **G** ~~2~~ ; **A** [(] [1,2] [)], **A** [(] [2,2] [)], . . ., **A** [(] _[N]_ [2] [,2] [)] �.


¨ ¨ ¨

**G** _P_ = � **G** _P_ + 1 ; **A** [(] [1,] _[P]_ [+] [1] [)], **A** [(] [2,] _[P]_ [+] [1] [)], . . ., **A** [(] _[N]_ _[P]_ [+] [1] [,] _[P]_ [+] [1] [)] �, (3.21)


where **G** _p_ P **R** _R_ 1 [(] _[p]_ [)] [ˆ] _[R]_ 2 [(] _[p]_ [)] [ˆ¨¨¨ˆ] _[R]_ [(] _Np_ _[p]_ [)] and **A** ( _n_ _p_, _p_ ) P **R** _R_ _lnp_ [(] _[p]_ [´][1] [)] ˆ¨¨¨ˆ _R_ _mnp_ [(] _[p]_ [´][1] [)] [ˆ] _[R]_ [(] _np_ _[p]_ [)], with _p_ =
2, . . ., _P_ + 1.
Note that some factor tensors, **A** [(] _[n]_ [,1] [)] and/or **A** [(] _[n]_ _[p]_ [,] _[p]_ [)], can be identity
tensors which yield an irregular structure, possibly with a varying order
of tensors. This follows from the simple observation that a mode- _n_ product
may have, e.g., the following form


**X** ˆ _n_ **B** [(] _[n]_ [)] = � **X** ; **I** 1, . . ., **I** _I_ _n_ ´1, **B** [(] _[n]_ [)], **I** _I_ _n_ + 1, . . ., **I** _I_ _N_ �.


The efficiency of this representation strongly relies on an appropriate
choice of the tree structure. It is usually assumed that the tree structure
of TTNS is given or assumed _a priori_, and recent efforts aim to find an
optimal tree structure from a subset of tensor entries and without any _a_
_priori_ knowledge of the tree structure. This is achieved using so-called
rank-adaptive cross-approximation techniques which approximate a
tensor by hierarchical tensor formats [9,10].


77


**Operations in the Tucker format.** If large-scale data tensors admit an
exact or approximate representation in their Tucker formats, then most
mathematical operations can be performed more efficiently using the so
obtained much smaller core tensors and factor matrices. Consider the _N_ thorder tensors **X** and **Y** in the Tucker format, given by


**X** = � **G** _X_ ; **X** [(] [1] [)], . . ., **X** [(] _[N]_ [)] � and **Y** = � **G** ~~_Y_~~ ; **Y** [(] [1] [)], . . ., **Y** [(] _[N]_ [)] �, (3.22)


for which the respective multilinear ranks are t _R_ 1, _R_ 2, . . ., _R_ _N_ u and
t _Q_ 1, _Q_ 2, . . ., _Q_ _N_ u, then the following mathematical operations can be
performed directly in the Tucker format [5], which admits a significant
reduction in computational costs [128,175,177]:


_•_ **The addition** of two Tucker tensors of the same order and sizes


**X** + **Y** = � **G** _X_ ‘ **G** ~~_Y_~~ ; [ **X** [(] [1] [)], **Y** [(] [1] [)] ], . . ., [ **X** [(] _[N]_ [)], **Y** [(] _[N]_ [)] ] �, (3.23)


where ‘ denotes a direct sum of two tensors, and [ **X** [(] _[n]_ [)], **Y** [(] _[n]_ [)] ] P
**R** _[I]_ _[n]_ [ˆ] [(] _[R]_ _[n]_ [+] _[Q]_ _[n]_ [)], **X** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ and **Y** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[Q]_ _[n]_, @ _n_ .


_•_
**The Kronecker product** of two Tucker tensors of arbitrary orders and
sizes


**X** b **Y** = � **G** _X_ b **G** ~~_Y_~~ ; **X** [(] [1] [)] b **Y** [(] [1] [)], . . ., **X** [(] _[N]_ [)] b **Y** [(] _[N]_ [)] �. (3.24)


_•_ **The Hadamard** or element-wise product of two Tucker tensors of the
same order and the same sizes


**X** f **Y** = � **G** _X_ b **G** ~~_Y_~~ ; **X** [(] [1] [)] d 1 **Y** [(] [1] [)], . . ., **X** [(] _[N]_ [)] d 1 **Y** [(] _[N]_ [)] �, (3.25)


where d 1 denotes the mode-1 Khatri–Rao product, also called the
transposed Khatri–Rao product or row-wise Kronecker product.


_•_ **The inner product** of two Tucker tensors of the same order and
sizes can be reduced to the inner product of two smaller tensors by
exploiting the Kronecker product structure in the vectorized form, as


5 Similar operations can be performed in the CP format, assuming that the core tensors
are diagonal.


78


follows



x **X**, **Y** y = vec ( **X** ) [T] vec ( **Y** ) (3.26)



_N_
= vec ( **G** _X_ ) [T] â
� _n_ = 1



_N_ _N_
â **X** [(] _[n]_ [)] [ T] â

_n_ = 1 �� _n_ = 1



_N_
â **Y** [(] _[n]_ [)]

_n_ = 1 �



vec ( **G** ~~_Y_~~ )



_N_
= vec ( **G** _X_ ) [T] â **X** [(] _[n]_ [)] [ T] **Y** [(] _[n]_ [)]
� _n_ = 1 �



vec ( **G** ~~_Y_~~ )



= x� **G** _X_ ; ( **X** [(] [1] [)] [ T] **Y** [(] [1] [)] ), . . ., ( **X** [(] _[N]_ [)] [ T] **Y** [(] _[N]_ [)] ) �, **G** ~~_Y_~~ y.


_•_ **The Frobenius norm** can be computed in a particularly simple
way if the factor matrices are orthogonal, since then all products
**X** [(] _[n]_ [)] [ T] **X** [(] _[n]_ [)], @ _n_, become the identity matrices, so that


} **X** } _F_ = x **X**, **X** y


T
= vec � **G** _X_ ; ( **X** [(] [1] [)] [ T] **X** [(] [1] [)] ), . . ., ( **X** [(] _[N]_ [)] [ T] **X** [(] _[N]_ [)] ) � vec ( **G** _X_ )
� �

= vec ( **G** _X_ ) [T] vec ( **G** _X_ ) = } **G** _X_ } _F_ . (3.27)


_•_ **The** _N_ **-D discrete convolution** of tensors **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and **Y** P
**R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[N]_ in their Tucker formats can be expressed as


**Z** = **X** ˚ **Y** = � **G** _Z_ ; **Z** [(] [1] [)], . . ., **Z** [(] _[N]_ [)] � (3.28)

P **R** [(] _[I]_ [1] [+] _[J]_ [1] [´][1] [)] [ˆ¨¨¨ˆ] [(] _[I]_ _[N]_ [+] _[J]_ _[N]_ [´][1] [)] .


If t _R_ 1, _R_ 2, . . ., _R_ _N_ u is the multilinear rank of **X** and t _Q_ 1, _Q_ 2, . . ., _Q_ _N_ u
the multilinear rank **Y**, then the core tensor **G** _Z_ = **G** _X_ b **G** ~~_Y_~~ P
**R** _[R]_ [1] _[Q]_ [1] [ˆ¨¨¨ˆ] _[R]_ _[N]_ _[Q]_ _[N]_ and the factor matrices


**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] d 1 **Y** [(] _[n]_ [)] P **R** [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)] [ˆ] _[R]_ _[n]_ _[Q]_ _[n]_, (3.29)


where **Z** [(] _[n]_ [)] ( :, _s_ _n_ ) = **X** [(] _[n]_ [)] ( :, _r_ _n_ ) ˚ **Y** [(] _[n]_ [)] ( :, _q_ _n_ ) P **R** [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)] for
_s_ _n_ = ~~_r_~~ _n_ ~~_q_~~ _n_ = 1, 2, . . ., _R_ _n_ _Q_ _n_ .


_•_ **Super Fast discrete Fourier transform** (MATLAB functions fftn ( **X** )
and fft ( **X** [(] _[n]_ [)], [], 1 ) ) of a tensor in the Tucker format


_F_ ( **X** ) = � **G** _X_ ; _F_ ( **X** [(] [1] [)] ), . . ., _F_ ( **X** [(] _[N]_ [)] ) �. (3.30)


79


Note that if the data tensor admits low multilinear rank
approximation, then performing the FFT on factor matrices of
relatively small size **X** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, instead of a large-scale data tensor,
decreases considerably computational complexity. This approach is
referred to as the super fast Fourier transform in Tucker format.


**3.4** **Higher Order SVD (HOSVD) for Large-Scale**
**Problems**


The MultiLinear Singular Value Decomposition (MLSVD), also called the
higher-order SVD (HOSVD), can be considered as a special form of the
constrained Tucker decomposition [59, 60], in which all factor matrices,
**B** [(] _[n]_ [)] = **U** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[I]_ _[n]_, are orthogonal and the core tensor, **G** = **S** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, is all-orthogonal (see Figure 3.4).
The orthogonality properties of the core tensor are defined through the
following conditions:


1. _All orthogonality._ The slices in each mode are mutually orthogonal,
e.g., for a 3rd-order tensor and its lateral slices


x **S** :, _k_,: **S** :, _l_,: y = 0, for _k_ ‰ _l_, (3.31)


2. _Pseudo-diagonality._ The Frobenius norms of slices in each mode are
decreasing with the increase in the running index, e.g., for a 3rd-order
tensor and its lateral slices


} **S** :, _k_,: } _F_ ě } **S** :, _l_,: } _F_, _k_ ě _l_ . (3.32)


These norms play a role similar to singular values in standard matrix
SVD.


In practice, the orthogonal matrices **U** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, with _R_ _n_ ď _I_ _n_, can be
computed by applying both the randomized and standard truncated SVD
to the unfolded mode- _n_ matrices, **X** ( _n_ ) – **U** [(] _[n]_ [)] **S** _n_ **V** [(] _[n]_ [)] [T] P **R** _[I]_ _[n]_ [ˆ] _[I]_ [1] [¨¨¨] _[I]_ _[n]_ [´][1] _[I]_ _[n]_ [+] [1] [¨¨¨] _[I]_ _[N]_ .
After obtaining the orthogonal matrices **U** [(] _[n]_ [)] of left singular vectors of **X** ( _n_ ),
for each _n_, the core tensor **G** = **S** can be computed as


**S** = **X** ˆ 1 **U** [(] [1] [)] [ T] ˆ 2 **U** [(] [2] [)] [ T] ¨ ¨ ¨ ˆ _N_ **U** [(] _[N]_ [)] [ T], (3.33)


so that


**X** = **S** ˆ 1 **U** [(] [1] [)] ˆ 2 **U** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **U** [(] _[N]_ [)] . (3.34)


80


(a)


(b)


(c)


















|Col1|Col2|
|---|---|
|||




















|Col1|U|Col3|
|---|---|---|
||**U**||
||**U**||
||**U**||
||**U**||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**V**<br>T|**V**<br>T|**V**<br>T|**V**<br>T|**V**<br>T|
|||0|||


























|U(1)|Col2|
|---|---|
|**U**<br>(1)<br>||
|**U**<br>(1)<br>||























Figure 3.4: Graphical illustration of the truncated SVD and HOSVD. (a) The
exact and truncated standard matrix SVD, **X** – **USV** [T] . (b) The truncated
(approximative) HOSVD for a 3rd-order tensor calculated as **X** – **S** _t_ ˆ 1 **U** [(] [1] [)] ˆ 2
**U** [(] [2] [)] ˆ 3 **U** [(] [3] [)] . (c) Tensor network notation for the HOSVD of a 4th-order tensor
**X** – **S** _t_ ˆ 1 **U** [(] [1] [)] ˆ 2 **U** [(] [2] [)] ˆ 3 **U** [(] [3] [)] ˆ 4 **U** [(] [4] [)] . All the factor matrices, **U** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, and
the core tensor, **S** _t_ = **G** P **R** _[R]_ [1] [ˆ¨¨¨ˆ] _[R]_ _[N]_, are orthogonal.


81


Due to the orthogonality of the core tensor **S**, its slices are also mutually
orthogonal.
Analogous to the standard truncated SVD, a large-scale data tensor, **X**,
can be approximated by discarding the multilinear singular vectors and
slices of the core tensor corresponding to small multilinear singular values.
Figure 3.4 and Algorithm 2 outline the truncated HOSVD, for which any
optimized matrix SVD procedure can be applied.
For large-scale tensors, the unfolding matrices, **X** ( _n_ ) P **R** _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [¯] ( _I_ _n_ ¯ =
_I_ 1 ¨ ¨ ¨ _I_ _n_ _I_ _n_ + 1 ¨ ¨ ¨ _I_ _N_ ) may become prohibitively large (with _I_ _n_ ¯ " _I_ _n_ ), easily
exceeding the memory of standard computers. Using a direct and
simple divide-and-conquer approach, the truncated SVD of an unfolding
matrix, **X** ( _n_ ) = **U** [(] _[n]_ [)] **S** _n_ **V** [(] _[n]_ [)] [T], can be partitioned into _Q_ slices, as **X** ( _n_ ) =

[ **X** 1, _n_, **X** 2, _n_, . . ., **X** _Q_, _n_ ] = **U** [(] _[n]_ [)] **S** _n_ [ **V** [T] 1, _n_ [,] **[ V]** [T] 2, _n_ [, . . .,] **[ V]** [T] _Q_, _n_ []] [. Next, the orthogonal]
matrices **U** [(] _[n]_ [)] and the diagonal matrices **S** _n_ can be obtained from the
eigenvalue decompositions **X** ( _n_ ) **X** [T] ( _n_ ) [=] **[ U]** [(] _[n]_ [)] **[S]** [2] _n_ **[U]** [(] _[n]_ [)] [T] [ =] [ ř] _q_ **[X]** _[q]_ [,] _[n]_ **[X]** [T] _q_, _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [,]

allowing for the terms **V** _q_, _n_ = **X** [T] _q_, _n_ **[U]** [(] _[n]_ [)] **[S]** [´] _n_ [1] to be computed separately. This
enables us to optimize the size of the _q_ th slice **X** _q_, _n_ P **R** _[I]_ _[n]_ [ˆ] [(] _[I]_ _[n]_ [¯] [/] _[Q]_ [)] so as to
match the available computer memory. Such a simple approach to compute
matrices **U** [(] _[n]_ [)] and/or **V** [(] _[n]_ [)] does not require loading the entire unfolding
matrices at once into computer memory; instead the access to the datasets is
sequential. For current standard sizes of computer memory, the dimension
_I_ _n_ is typically less than 10,000, while there is no limit on the dimension
_I_ _n_ ¯ = [ś] _k_ ‰ _n_ _[I]_ _[k]_ [.]
For very large-scale and low-rank matrices, instead of the standard
truncated SVD approach, we can alternatively apply the randomized SVD
algorithm, which reduces the original data matrix **X** to a relatively small
matrix by random sketching, i.e. through multiplication with a random
sampling matrix **Ω** (see Algorithm 3). Note that we explicitly allow the
rank of the data matrix **X** to be overestimated (that is, _R_ [˜] = _R_ + _P_, where
_R_ is a true but unknown rank and _P_ is the over-sampling parameter)
because it is easier to obtain more accurate approximation of this form.
Performance of randomized SVD can be further improved by integrating
multiple random sketches, that is, by multiplying a data matrix **X** by a set
of random matrices **Ω** _p_ for _p_ = 1, 2, . . ., _P_ and integrating leading lowdimensional subspaces by applying a Monte Carlo integration method [33].
Using special random sampling matrices, for instance, a sub-sampled
random Fourier transform, substantial gain in the execution time can
be achieved, together with the asymptotic complexity of _O_ ( _IJ_ log ( _R_ )) .
Unfortunately, this approach is not accurate enough for matrices for which


82


**Algorithm 2** : **Sequentially Truncated HOSVD [212]**

**Input:** _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and approximation

accuracy _ε_
**Output:** HOSVD in the Tucker format **X** [ˆ] = � **S** ; **U** [(] [1] [)], . . ., **U** [(] _[N]_ [)] �,
such that } **X** ´ **X** [ˆ] } _F_ ď _ε_
1: **S** Ð **X**

2: **for** _n_ = 1 to _N_ **do**
3: [ **U** [(] _[n]_ [)], **S**, **V** ] = `truncated` ~~`s`~~ `vd` ( **S** ( _n_ ), ~~?~~ _εN_ [)]

4: **S** Ð **VS**

5: **end for**
6: **S** Ð `reshape` ( **S**, [ _R_ 1, . . ., _R_ _N_ ])
7: **return** Core tensor **S** and orthogonal factor matrices
**U** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ .


**Algorithm 3** : **Randomized SVD (rSVD) for large-scale and low-rank**
**matrices with single sketch [93]**

**Input:** A matrix **X** P **R** _[I]_ [ˆ] _[J]_, desired or estimated rank _R_, and
oversampling parameter _P_ or overestimated rank _R_ [r] = _R_ + _P_,
exponent of the power method _q_ ( _q_ = 0 or _q_ = 1)
**Output:** An approximate rank- _R_ [r] SVD, **X** – **USV** [T],

i.e., orthogonal matrices **U** P **R** _[I]_ [ˆ] _[R]_ [r], **V** P **R** _[J]_ [ˆ] _[R]_ [r]

and diagonal matrix of singular values **S** P **R** _[R]_ [r][ˆ] _[R]_ [r]

1: Draw a random Gaussian matrix **Ω** P **R** _[J]_ [ˆ] _[R]_ [r],

2: Form the sample matrix **Y** = ( **XX** [T] ) _[q]_ **XΩ** P **R** _[I]_ [ˆ] _[R]_ [r]

3: Compute a QR decomposition **Y** = **QR**

4: Form the matrix **A** = **Q** [T] **X** P **R** _[R]_ [r][ˆ] _[J]_

5: Compute the SVD of the small matrix **A** as **A** = **USV** [p] [T]

6: Form the matrix **U** = **QU** [p] .


the singular values decay slowly [93].
The truncated HOSVD can be optimized and implemented in several
alternative ways. For example, if _R_ _n_ ! _I_ _n_, the truncated tensor **Z** Ð
**X** ˆ 1 **U** [(] [1] [)] [T] yields a smaller unfolding matrix **Z** ( 2 ) P **R** _[I]_ [2] [ˆ] _[R]_ [1] _[I]_ [3] [¨¨¨] _[I]_ _[N]_, so that the
multiplication **Z** ( 2 ) **Z** [T] ( 2 ) [can be faster in the next iterations [5,212].]

Furthermore, since the unfolding matrices **Y** [T] ( _n_ ) [are typically very “tall]
and skinny”, a huge-scale truncated SVD and other constrained low-rank
matrix factorizations can be computed efficiently based on the Hadoop /
MapReduce paradigm [20,48,49].


83


**Algorithm 4** : **Higher Order Orthogonal Iteration (HOOI) [5,60]**

**Input:** _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ (usually in Tucker/HOSVD
format)
**Output:** Improved Tucker approximation using ALS approach, with
orthogonal factor matrices **U** [(] _[n]_ [)]

1: Initialization via the standard HOSVD (see Algorithm 2)

2: **repeat**

3: **for** _n_ = 1 to _N_ **do**

4: **Z** Ð **X** ˆ _p_ ‰ _n_ t **U** [(] _[p]_ [)] [ T] u

5: **C** Ð **Z** ( _n_ ) **Z** [T] ( _n_ ) [P] **[ R]** _[R]_ [ˆ] _[R]_

6: **U** [(] _[n]_ [)] Ð leading _R_ _n_ eigenvectors of **C**

7: **end for**

8: **G** Ð **Z** ˆ _N_ **U** [(] _[N]_ [)] [ T]

9: **until** the cost function ( } **X** } [2] _F_ [´ ][}] **[G]** [}] [2] _F_ [)] [ ceases to decrease]

10: **return** � **G** ; **U** [(] [1] [)], **U** [(] [2] [)], . . ., **U** [(] _[N]_ [)] �


_Low multilinear rank approximation is always well-posed_, however, in
contrast to the standard truncated SVD for matrices, _the truncated HOSVD_
_does not yield the best multilinear rank approximation_, but satisfies the quasibest approximation property [59]



} **X** ´ � **S** ; **U** [(] [1] [)], . . ., **U** [(] _[N]_ [)] �} ď ?



_N_ } **X** ´ **X** ~~B~~ est}, (3.35)



where **X** ~~B~~ est is the best multilinear rank approximation of **X**, for a specific
tensor norm } ¨ }.
When it comes to the problem of finding the best approximation, the
ALS type algorithm called the Higher Order Orthogonal Iteration (HOOI)
exhibits both the advantages and drawbacks of ALS algorithms for CP
decomposition. For the HOOI algorithms, see Algorithm 4 and Algorithm
5. For more sophisticated algorithms for Tucker decompositions with
orthogonality and nonnegativity constraints, suitable for large-scale data
tensors, see [49,104,169,236].
When a data tensor **X** is very large and cannot be stored in computer
memory, another challenge is to compute a core tensor **G** = **S** directly,
using the formula (3.33). Such computation is performed sequentially
by fast matrix-by-matrix multiplications [6], as illustrated in Figure 3.5(a)
and (b).


6 Efficient and parallel (state of the art) algorithms for multiplications of such very largescale matrices are proposed in [11,131].


84


Table 3.3: Basic multiway component analysis (MWCA)/Low-Rank Tensor
Approximations (LRTA) and related multiway dimensionality reduction
models. The symbol **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ denotes a noisy data tensor, while
**Y** = **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)] is the general constrained Tucker model
with the latent factor matrices **B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ and the core tensor **G** P
**R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ¨¨¨ˆ] _[R]_ _[N]_ . In the special case of a CP decomposition, the core tensor
is diagonal, **G** = **Λ** P **R** _[R]_ [ˆ¨¨¨ˆ] _[R]_, so that **Y** = [ř] _r_ _[R]_ = 1 _[λ]_ _[r]_ [(] **[b]** _r_ [(] [1] [)] ˝ **b** _r_ [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **b** _r_ [(] _[N]_ [)] ) .



|Cost Function|Constraints|
|---|---|
|Multilinear (sparse) PCA (MPCA)<br>max**u**(_n_)<br>_r_<br>**X** ¯ˆ1**u**(1)<br>_r_<br>¯ˆ2**u**(2)<br>_r_<br>¨ ¨ ¨ ¯ˆ_N_**u**(_N_)<br>_r_<br>+_ γ_ ř_N_<br>_n_=1 }**u**(_n_)<br>_r_<br>}1|**u**(_n_) T<br>_r_<br>**u**(_n_)<br>_r_<br>= 1, @(_n_,_ r_)<br>**u**(_n_) T<br>_r_<br>**u**(_n_)<br>_q_<br>= 0 for_ r_ ‰_ q_|
|HOSVD/HOOI<br>min**U**(_n_) }**X** ´** G** ˆ1** U**(1) ˆ2** U**(2) ¨ ¨ ¨ ˆ_N_** U**(_N_)}2<br>_F_|**U**(_n_) T **U**(_n_) =** I**_Rn_, @_n_|
|Multilinear ICA<br>min**B**(_n_) }**X** ´** G** ˆ1** B**(1) ˆ2** B**(2) ¨ ¨ ¨ ˆ_N_** B**(_N_)}2<br>_F_|Vectors of** B**(_n_) statistically<br>as independent as possible|
|Nonnegative CP/Tucker decomposition<br>(NTF/NTD) [43]<br>min**B**(_n_) }**X** ´** G** ˆ1** B**(1) ¨ ¨ ¨ ˆ_N_** B**(_N_)}2<br>_F_<br>+_γ_ ř_N_<br>_n_=1<br>ř_Rn_<br>_rn_=1 }**b**(_n_)<br>_rn_ }1|Entries of** G** and** B**(_n_), @_n_<br>are nonnegative|
|Sparse CP/Tucker decomposition<br>min**B**(_n_) }**X** ´** G** ˆ1** B**(1) ¨ ¨ ¨ ˆ_N_** B**(_N_)}2<br>_F_<br>+_γ_ ř_N_<br>_n_=1<br>ř_Rn_<br>_rn_=1 }**b**(_n_)<br>_rn_ }1|Sparsity constraints<br>imposed on** B**(_n_)|
|Smooth CP/Tucker decomposition<br>(SmCP/SmTD) [228]<br>min**B**(_n_) }**X** ´** Λ** ˆ1** B**(1) ¨ ¨ ¨ ˆ_N_** B**(_N_)}2<br>_F_<br>+_γ_ ř_N_<br>_n_=1<br>ř_R_<br>_r_=1 }**Lb**(_n_)<br>_r_<br>}2|Smoothness imposed<br>on vectors** b**(_n_)<br>_r_<br>of** B**(_n_) P** R**_In_ˆ_R_, @_n_<br>via a difference operator** L**|


85




**Algorithm 5** : **HOOI using randomization for large-scale data [238]**

**Input:** _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and multilinear rank
t _R_ 1, _R_ 2, . . ., _R_ _N_ u
**Output:** Approximative representation of a tensor in Tucker format,
with orthogonal factor matrices **U** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_

1: Initialize factor matrices **U** [(] _[n]_ [)] as random Gaussian matrices
Repeat steps (2)-(6) only two times:

2: **for** _n_ = 1 to _N_ **do**

3: **Z** = **X** ˆ _p_ ‰ _n_ t **U** [(] _[p]_ [)] [ T] u

4: Compute **Z** [˜] [(] _[n]_ [)] = **Z** ( _n_ ) **Ω** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, where **Ω** [(] _[n]_ [)] P **R** ś _p_ ‰ _n_ _[R]_ _[p]_ [ˆ] _[R]_ _[n]_

is a random matrix drawn from Gaussian distribution

5: Compute **U** [(] _[n]_ [)] as an orthonormal basis of **Z** [˜] [(] _[n]_ [)], e.g., by using QR
decomposition

6: **end for**


7: Construct the core tensor as
**G** = **X** ˆ 1 **U** [(] [1] [)] [ T] ˆ 2 **U** [(] [2] [)] [ T] ¨ ¨ ¨ ˆ _N_ **U** [(] _[N]_ [)] [ T]

8: **return X** – � **G** ; **U** [(] [1] [)], **U** [(] [2] [)], . . ., **U** [(] _[N]_ [)] �


**Algorithm 6** : **Tucker decomposition with constrained factor**
**matrices via 2-way CA /LRMF**

**Input:** _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, multilinear rank
t _R_ 1, . . ., _R_ _N_ u and desired constraints imposed on factor matrices
**B** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_
**Output:** Tucker decomposition with constrained factor matrices **B** [(] _[n]_ [)]

using LRMF and a simple unfolding approach

1: Initialize randomly or via standard HOSVD (see Algorithm 2)

2: **for** _n_ = 1 to _N_ **do**

3: Compute specific LRMF or 2-way CA (e.g., RPCA, ICA, NMF) of
unfolding **X** [T] ( _n_ ) [–] **[ A]** [(] _[n]_ [)] **[B]** [(] _[n]_ [)] [ T] [ or] **[ X]** [(] _[n]_ [)] [ –] **[ B]** [(] _[n]_ [)] **[A]** [(] _[n]_ [)] [ T]

4: **end for**

5: Compute core tensor **G** = **X** ˆ 1 [ **B** [(] [1] [)] ] [:] ˆ 2 [ **B** [(] [2] [)] ] [:] ¨ ¨ ¨ ˆ _N_ [ **B** [(] _[N]_ [)]] [:]

6: **return** Constrained Tucker decomposition **X** – � **G**, **B** [(] [1] [)], . . ., **B** [(] _[N]_ [)] �


We have shown that for very large-scale problems, it is useful to divide
a data tensor **X** into small blocks **X** [ _k_ 1, _k_ 2,..., _k_ _N_ ] . In a similar way, we can
partition the orthogonal factor matrices **U** [(] _[n]_ [)] [T] into the corresponding blocks


86


of matrices **U** [(] _[n]_ [)] [T]

[ _k_ _n_, _p_ _n_ ] [, as illustrated in Figure 3.5(c) for 3rd-order tensors]

[200, 221]. For example, the blocks within the resulting tensor **G** [(] _[n]_ [)] can
be computed sequentially or in parallel, as follows:



**G** [(] _[n]_ [)]

[ _k_ 1, _k_ 2,..., _q_ _n_,..., _k_ _N_ ] [=]



_K_ _n_
ÿ **X** [ _k_ 1, _k_ 2,..., _k_ _n_,..., _k_ _N_ ] ˆ _n_ **U** [(] [ _k_ _[n]_ _n_ [)], [ T] _q_ _n_ ] [.] (3.36)

_k_ _n_ = 1



**Applications.** We have shown that the Tucker/HOSVD decomposition
may be considered as a multilinear extension of PCA [124]; it therefore
generalizes signal subspace techniques and finds application in areas
including multilinear blind source separation, classification, feature
extraction, and subspace-based harmonic retrieval [90, 137, 173, 213]. In
this way, a low multilinear rank approximation achieved through Tucker
decomposition may yield higher Signal-to-Noise Ratio (SNR) than the SNR
for the original raw data tensor, which also makes Tucker decomposition a
natural tool for signal compression and enhancement.
It was recently shown that HOSVD can also perform simultaneous
subspace selection (data compression) and K-means clustering, both
unsupervised learning tasks [99, 164]. This is important, as a combination
of these methods can both identify and classify “relevant” data, and in
this way not only reveal desired information but also simplify feature
extraction.


**Anomaly detection using HOSVD.** Anomaly detection refers to the
discrimination of some specific patterns, signals, outliers or features that
do not conform to certain expected behaviors, trends or properties [32,
78]. While such analysis can be performed in different domains, it is
most frequently based on spectral methods such as PCA, whereby high
dimensional data are projected onto a lower-dimensional subspace in
which the anomalies may be identified more easier. The main assumption
within such approaches is that the normal and abnormal patterns, which
may be difficult to distinguish in the original space, appear significantly
different in the projected subspace. When considering very large datasets,
since the basic Tucker decomposition model generalizes PCA and SVD,
it offers a natural framework for anomaly detection via HOSVD, as
illustrated in Figure 3.6. To handle the exceedingly large dimensionality,
we may first compute tensor decompositions for sampled (pre-selected)
small blocks of the original large-scale 3rd-order tensor, followed by the
analysis of changes in specific factor matrices **U** [(] _[n]_ [)] . A simpler form


87


(a) Sequential computation



_I_ 3

_R_ 1 _R_ 2



_R_ 1



_I_ 3



_I_ 2



_R_ 3



**G** (1) _I_ 3 **G** (2)

_R_ 2







_R_ 1



**U** (1)T **G** (1) _R_ 2 **U** (2)T **G** (2) _R_ 3 **U** (3)T **G**



_R_ 3



_I_ 1 _I_ 2 _I_ 3



_I_ 2 _I_ 3 _R_ 1


|3 .<br>I<br>1<br>U(1)T|..<br>X<br>G(1) ...|Col3|Col4|
|---|---|---|---|
|_I_1<br><br>.<br>**U**<br>(1)T<br>|**X**<br>|**X**<br>|**X**<br>|
|_I_1<br><br>.<br>**U**<br>(1)T<br>|**X**|**X**|**X**|
|_I_1<br><br>.<br>**U**<br>(1)T<br>|**X**|||
|_I_1<br><br>.<br>**U**<br>(1)T<br>|**G**<br>(1)|||
|_I_1<br><br>.<br>**U**<br>(1)T<br>|**G**<br>(1)|||
|_I_1<br><br>.<br>**U**<br>(1)T<br>||||


|...<br>1<br>I G(1)<br>2<br>U(2)T G(2) ...|Col2|G(1)|Col4|Col5|
|---|---|---|---|---|
|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1|**G**<br>(1)|**G**<br>(1)|**G**<br>(1)|
|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1||**G**<br>(1)|**G**<br>(1)|**G**<br>(1)|
|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1|||||
|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1|**G**<br>(2)<br>|(2)<br>|||
|_I_2<br><br>...<br>**G**<br>(1)<br>**U**<br>(2)T<br>**G**<br>(2)<br>...<br>1|**G**<br>(2)<br>|**G**<br><br>|||


|R ...<br>2<br>I G(2)<br>3<br>U(3)T G ...|Col2|(2)|Col4|Col5|
|---|---|---|---|---|
|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|(2)|(2)|(2)|
|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|**G**|(2)|(2)|(2)|
|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|**G**||||
|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|**G**<br>||||
|_I_3<br><br>...<br>**G**<br>(2)<br>**G**<br>**U**<br>(3)T<br>...<br>_R_2|**G**<br>||||



**G** (( **X** 1 **U** (1)T ) 2 **U** (2)T ) 3 **U** (3)T


(b) Fast matrix-by-matrix approach



_I_ 3 _R_ 1 _[R]_ [2]


_I_ _I_
2 3



_I_ 3



_I_ 2



_I_ 1 **X** **X** (1)



_R_ 1



_I_ 1


_R_ 1



_I_ 1 _I_ 2 _I_ 3

|3 ...<br>I X<br>1<br>U(1)T G(1) ...|Col2|X|Col4|Col5|
|---|---|---|---|---|
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...||**X**<br>|**X**<br>|**X**<br>|
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...||**X**|**X**|**X**|
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...|||||
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...|||||
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...|**G**<br>(1)<br>|(1)<br>|(1)<br>|(1)<br>|
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...|**G**<br>(1)<br>|(1)<br>|||
|_I_1<br><br><br>**X**<br>...<br>**U**<br>(1)T<br>**G**<br>(1)<br>3<br>...|||||



(c) Divide-and-conquer approach


**X**



_R_ 1 **U** (1)T **G** (1)(1)

_I_ 1



**U** (1)T **Z G** = (1)





=




|X X<br>[1,1,2] [1,2,2]<br>X X<br>[1,1,1] [1,2,1]<br>X X<br>[2,1,1] [2,2,1]<br>X X<br>[3,1,1] [3,2,1]|Col2|
|---|---|
|**X**~~[~~1,1,1] <br>**X**~~[~~2,1,1]|**X**~~[~~1,2,<br>**X**~~[~~2,2,1|
|**X**~~[~~3,1,1]|**X**~~[~~3,2,|


|U<br>[1,1]|U<br>[2,1]|U<br>[3,1]|
|---|---|---|
|**U**[1,2]|**U**[2,2]|**U**[3,2]|


|Z G = (1)|Col2|
|---|---|
|**Z**[1,1,1]<br>**Z**[1,2,1]<br>**Z**[2,1,1]<br>**Z**[2,2,1]<br>**Z**[1,1,2]<br>**Z**[1,2,2]<br>**z**[<br>]<br>2**,**2,2|**Z**[1,1,1]<br>**Z**[1,2,1]<br>**Z**[2,1,1]<br>**Z**[2,2,1]<br>**Z**[1,1,2]<br>**Z**[1,2,2]<br>**z**[<br>]<br>2**,**2,2|
|**Z**[1,1,1]<br>|**Z**[1,2,1]|
|**Z**[2,1,1]|**Z**[2,2,1]|



( _I_ 1 _I_ 2 _I_ 3 ) ( _R_ 1 _I_ 1 ) ( _R_ 1 _I_ 2 _I_ 3 )


Figure 3.5: Computation of a multilinear (Tucker) product for large-scale
HOSVD. (a) Standard sequential computing of multilinear products (TTM) **G** =
**S** = ((( **X** ˆ 1 **U** [(] [1] [)] [T] ) ˆ 2 **U** [(] [2] [)] [T] ) ˆ 3 **U** [(] [3] [)] [T] ) . (b) Distributed implementation through
fast matrix-by-matrix multiplications. (c) An alternative method for large-scale
problems using the “divide and conquer” approach, whereby a data tensor, **X**,
and factor matrices, **U** [(] _[n]_ [)] [T], are partitioned into suitable small blocks: Subtensors

**X** [ _k_ 1, _k_ 2, _k_ 3 ] and block matrices **U** [(] [ _k_ [1] 1 [)], [T] _p_ 1 ] [. The blocks of a tensor,] **[ Z]** [=] **[ G]** [(] [1] [)] [ =] **[ X]** [ˆ] [1] **[ U]** [(] [1] [)] [T] [,]

are computed as **Z** [ _q_ 1, _k_ 2, _k_ 3 ] = [ř] _k_ _[K]_ 1 [1] = 1 **[X]** [[] _[k]_ 1 [,] _[k]_ 2 [,] _[k]_ 3 []] [ ˆ] [1] **[ U]** [(] [ _k_ [1] 1 [)], [T] _q_ 1 ] [(see Eq. (3.36) for a general]
case).


88


( _I_ 3 _R_ 3 )


|I X|Col2|Col3|
|---|---|---|
|3<br>|3<br>|3<br>|
||**X**~~**_k_**~~<br>||



( _I_ 1 _R_ 1 ) ( _R_ 1 _R_ 2 _R_ 3 ) ( _R_ 2 _I_ 2 )



(2)
**U**



_I_



_I_
2



**U**



(2)T
**U** _**k**_



_I_
1



(1)
**U**



_R_ 3 **U** (2)T
_R_ 1 **G** **U** (2)T _**k**_

_R_
2





Figure 3.6: Conceptual model for performing the HOSVD for a very largescale 3rd-order data tensor. This is achieved by dividing the tensor into
blocks **X** _k_ – **G** ˆ 1 **U** [(] [1] [)] ˆ 2 **U** [(] _k_ [2] [)] ˆ 3 **U** [(] [3] [)], ( _k_ = 1, 2 . . ., _K_ ) . It assumed that
the data tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] is sampled by sliding the block **X** _k_ from left
to right (with an overlapping sliding window). The model can be used
for anomaly detection by fixing the core tensor and some factor matrices
while monitoring the changes along one or more specific modes (in our case
mode two). Tensor decomposition is then first performed for a sampled
(pre-selected) small block, followed by the analysis of changes in specific
smaller–dimensional factor matrices **U** [(] _[n]_ [)] .


is straightforwardly obtained by fixing the core tensor and some factor
matrices while monitoring the changes along one or more specific modes,
as the block tensor moves from left to right as shown in Figure 3.6.


**3.5** **Tensor Sketching Using Tucker Model**


The notion of sketches refers to replacing the original huge matrix or tensor
by a new matrix or tensor of a significantly smaller size or compactness, but
which approximates well the original matrix/tensor. Finding such sketches
in an efficient way is important for the analysis of big data, as a computer
processor (and memory) is often incapable of handling the whole data-set
in a feasible amount of time. For these reasons, the computation is often
spread among a set of processors which for standard “all-in-one” SVD
algorithms, are unfeasible.
Given a very large-scale tensor **X**, a useful approach is to compute a
sketch tensor **Z** or set of sketch tensors **Z** _n_ that are of significantly smaller
sizes than the original one.
There exist several matrix and tensor sketching approaches:
sparsification, random projections, fiber subset selections, iterative
sketching techniques and distributed sketching. We review the main


89


sketching approaches which are promising for tensors.
**1. Sparsification** generates a sparser version of the tensor which, in general,
can be stored more efficiently and admit faster multiplications by factor
matrices. This is achieved by decreasing the number on non-zero entries
and quantizing or rounding up entries. A simple technique is elementwise sparsification which zeroes out all sufficiently small elements (below
some threshold) of a data tensor, keeps all sufficiently large elements,
and randomly samples the remaining elements of the tensor with sample
probabilities proportional to the square of their magnitudes [152].
**2.** **Random Projection** based sketching randomly combines fibers of a
data tensor in all or selected modes, and is related to the concept of a
randomized subspace embedding, which is used to solve a variety of
numerical linear algebra problems (see [208] and references therein).
**3. Fiber subset selection**, also called tensor cross approximation (TCA),
finds a small subset of fibers which approximates the entire data tensor.
For the matrix case, this problem is known as the Column/Row Subset
Selection or CUR Problem which has been thoroughly investigated and for
which there exist several algorithms with almost matching lower bounds

[64,82,140].


**3.6** **Tensor** **Sketching** **via** **Multiple** **Random**
**Projections**


The random projection framework has been developed for computing
structured low-rank approximations of a data tensor from (random) linear
projections of much lower dimensions than the data tensor itself [28, 208].
Such techniques have many potential applications in large-scale numerical
multilinear algebra and optimization problems.
Notice that for an _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, we can compute
the following sketches


**Z** = **X** ˆ 1 **Ω** 1 ˆ 2 **Ω** 2 ¨ ¨ ¨ ˆ _N_ **Ω** _N_ (3.37)


and

**Z** _n_ = **X** ˆ 1 **Ω** 1 ¨ ¨ ¨ ˆ _n_ ´1 **Ω** _n_ ´1 ˆ _n_ + 1 **Ω** _n_ + 1 ¨ ¨ ¨ ˆ _N_ **Ω** _N_, (3.38)


for _n_ =, 1, 2, . . ., _N_, where **Ω** _n_ P **R** _[R]_ _[n]_ [ˆ] _[I]_ _[n]_ are statistically independent
random matrices with _R_ _n_ ! _I_ _n_, usually called test (or sensing) matrices.
A sketch can be implemented using test matrices drawn from various
distributions. The choice of a distribution leads to some tradeoffs [208],


90


(a)


(b)





_I_ 3





1



3 **Z**

_I_ 1 _R_ _R_



**Ω** 1



**Z**
_R_ 3



2



_R_ 1



= _I_ 1 _I_ 1 ~~**X**~~ **X** _I_ 3 _R_ 3 =





_R_ 1

_I_ 2 _R_ 3



_I_ 1 **X** _I_ 3 [3] = _I_ 1









_R_ 2



_R_ 2



_R_ 1 **Ω** 1



**Z**



_R_ 1 **Ω** 1



3



_I_ 1





_R_ 1

_R_ 2



_I_ 3 3 **Z**



=



~~**X**~~ **X** = _R_ ~~**X**~~ **X**



**X** **X**





_R_ 1
_R_ 2 _[R]_ 3











_R_ 2





_R_ 2


**Z** _n_ **X** 1 **Ω** 1 _n_ -1 **Ω** _n_ -1 _n_ +1 **[Ω]** _n_ +1 _N_ **Ω** _N_













_R_ _n_ +1



1 **Ω** 1 2 **Ω** 2 [...] _N_ **Ω** _N_



















**Ω**







=


**Z** **X**


=



















_R_ _N_













Figure 3.7: Illustration of tensor sketching using random projections of a
data tensor. (a) Sketches of a 3rd-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3] given by **Z** ~~1~~ =
**X** ˆ 2 **Ω** 2 ˆ 3 **Ω** 3 P **R** _[I]_ [1] [ˆ] _[R]_ [2] [ˆ] _[R]_ [3], **Z** ~~2~~ = **X** ˆ 1 **Ω** 1 ˆ 3 **Ω** 3 P **R** _[R]_ [1] [ˆ] _[I]_ [2] [ˆ] _[R]_ [3], **Z** ~~3~~ = **X** ˆ 1
**Ω** 1 ˆ 2 **Ω** 2 P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[I]_ [3], and **Z** = **X** ˆ 1 **Ω** 1 ˆ 2 **Ω** 2 ˆ 3 **Ω** 3 P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[R]_ [3] . (b)
Sketches for an _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ .


91


especially regarding (i) the costs of randomization, computation, and
communication to generate the test matrices; (ii) the storage costs for the
test matrices and the sketch; (iii) the arithmetic costs for sketching and
updates; (iv) the numerical stability of reconstruction algorithms; and (v)
the quality of a priori error bounds. The most important distributions of
random test matrices include:


_•_
**Gaussian random projections** which generate random matrices
with standard normal distribution. Such matrices usually provide
excellent performance in practical scenarios and accurate a priori
error bounds.


_•_ **Random matrices with orthonormal columns** that span uniformly
distributed random subspaces of dimensions _R_ _n_ . Such matrices
behave similar to Gaussian case, but usually exhibit even better
numerical stability, especially when _R_ _n_ are large.


_•_ **Rademacher and super-sparse Rademacher random projections** that
have independent Rademacher entries which take the values ˘1 with
equal probability. Their properties are similar to standard normal
test matrices, but exhibit some improvements in the cost of storage
and computational complexity. In a special case, we may use ultra
sparse Rademacher test matrices, whereby in each column of a test
matrix independent Rademacher random variables are placed only
in very few uniformly random locations determined by a sampling
parameter _s_ ; the remaining entries are set to zero. In an extreme case
of maximum sparsity, _s_ = 1, and each column of a test matrix has
exactly only one nonzero entry.


_•_ **Subsampled randomized Fourier transforms** based on test matrices
take the following form


**Ω** _n_ = **P** _n_ **F** _n_ **D** _n_, (3.39)


where **D** _n_ are diagonal square matrices with independent
Rademacher entries, **F** _n_ are discrete cosine transform (DCT) or
discrete Fourier transform (DFT) matrices, and entries of the matrix
**P** _n_ are drawn at random from a uniform distribution.


**Example.** The concept of tensor sketching via random projections is
illustrated in Figure 3.7 for a 3rd-order tensor and for a general case of
_N_ th-order tensors. For a 3rd-order tensor with volume (number of entries)


92


_I_ 1 _I_ 2 _I_ 3 we have four possible sketches which are subtensors of much smaller
sizes, e.g., _I_ 1 _R_ 2 _R_ 3, with _R_ _n_ ! _I_ _n_, if the sketching is performed along mode-2
and mode-3, or _R_ 1 _R_ 2 _R_ 3, if the sketching is performed along all three modes
(Figure 3.7(a) bottom right). From these subtensors we can reconstruct any
huge tensor if it has low a multilinear rank (lower than t _R_ 1, _R_ 2, . . ., _R_ _n_ u).
In more general scenario, it can be shown [28] that the _N_ th order tensor
data tensor **X** with sufficiently low-multilinear rank can be reconstructed
perfectly from the sketch tensors **Z** _n_, for _n_ = 1, 2, . . ., _N_, as follows


ˆ
**X** = **Z** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)], (3.40)


where **B** [(] _[n]_ [)] = [ **Z** _n_ ] ( _n_ ) **Z** [:] ( _n_ ) [for] _[ n]_ [ =] [ 1, 2, . . .,] _[ N]_ [ (for more detail see the next]
section).


**3.7** **Matrix/Tensor Cross-Approximation (MCA/TCA)**


Huge-scale matrices can be factorized using the Matrix CrossApproximation (MCA) method, which is also known under
the names of Pseudo-Skeleton or CUR matrix decompositions

[16, 17, 84, 85, 116, 141, 142, 162]. The main idea behind the MCA is to
provide reduced dimensionality of data through a linear combination of
only a few “meaningful” components, which are exact replicas of columns
and rows of the original data matrix. Such an approach is based on the
fundamental assumption that large datasets are highly redundant and
can therefore be approximated by low-rank matrices, which significantly
reduces computational complexity at the cost of a marginal loss of
information.
The MCA method factorizes a data matrix **X** P **R** _[I]_ [ˆ] _[J]_ as [84, 85] (see
Figure 3.8)


**X** = **CUR** + **E**, (3.41)


where **C** P **R** _[I]_ [ˆ] _[C]_ is a matrix constructed from _C_ suitably selected columns
of the data matrix **X**, matrix **R** P **R** _[R]_ [ˆ] _[J]_ consists of _R_ appropriately selected
rows of **X**, and matrix **U** P **R** _[C]_ [ˆ] _[R]_ is calculated so as to minimize the norm
of the error **E** P **R** _[I]_ [ˆ] _[J]_ .
A simple modification of this formula, whereby the matrix **U** is
absorbed into either **C** or **R**, yields the so-called CR matrix factorization
or Column/Row Subset selection:


**X** – **CR** [˜] = **CR** [˜] (3.42)


93


Figure 3.8: Principle of the matrix cross-approximation which decomposes
a huge matrix **X** into a product of three matrices, whereby only a small-size
core matrix **U** needs to be computed.


for which the bases can be either the columns, **C**, or rows, **R**, while **R** [˜] = **UR**
and **C** [˜] = **CU** .
For dimensionality reduction, _C_ ! _J_ and _R_ ! _I_, and the columns
and rows of **X** should be chosen optimally, in the sense of providing a
high “statistical leverage” and the best low-rank fit to the data matrix,
while at the same time minimizing the cost function } **E** } [2] _F_ [. For a given]
set of columns, **C**, and rows, **R**, the optimal choice for the core matrix
is **U** = **C** [:] **X** ( **R** [:] ) [T] . This requires access to all the entries of **X** and is not
practical or feasible for large-scale data. In such cases, a pragmatic choice
for the core matrix would be **U** = **W** [:], where the matrix **W** P **R** _[R]_ [ˆ] _[C]_ is
composed from the intersections of the selected rows and columns. It
should be noted that for rank ( **X** ) ď mint _C_, _R_ u the cross-approximation is
exact. For the general case, it has been proven that when the intersection
submatrix **W** is of maximum volume [7], the matrix cross-approximation is
close to the optimal SVD solution. The problem of finding a submatrix
with maximum volume has exponential complexity, however, suboptimal
matrices can be found using fast greedy algorithms [4,144,179,222].
The concept of MCA can be generalized to tensor cross-approximation
(TCA) (see Figure 3.9) through several approaches, including:


_•_
Applying the MCA decomposition to a matricized version of the
tensor data [142];


_•_
Operating directly on fibers of a data tensor which admits a lowrank Tucker approximation, an approach termed the Fiber Sampling


7 The volume of a square submatrix **W** is defined as | det ( **W** ) |.


94


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||**C**||||
|||**C**||||
|||**C**||||
|||||||
||**T**<br>|||||







Figure 3.9: The principle of the tensor cross-approximation (TCA)
algorithm, illustrated for a large-scale 3rd-order tensor **X** – **U** ˆ 1 **C** ˆ 2
**R** ˆ 3 **T** = � **U** ; **C**, **R**, **T** �, where **U** = **W** ˆ 1 **W** [:] ( 1 ) [ˆ] [2] **[ W]** [:] ( 2 ) [ˆ] [3] **[ W]** [:] ( 3 ) =

� **W** ; **W** [:] ( 1 ) [,] **[ W]** [:] ( 2 ) [,] **[ W]** [:] ( 3 ) [�] [P] **[ R]** _[P]_ [2] _[P]_ [3] [ˆ] _[P]_ [1] _[P]_ [3] [ˆ] _[P]_ [1] _[P]_ [2] [ and] **[ W]** [P] **[ R]** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ] _[P]_ [3] [. For simplicity]
of illustration, we assume that the selected fibers are permuted, so as
to become clustered as subtensors, **C** P **R** _[I]_ [1] [ˆ] _[P]_ [2] [ˆ] _[P]_ [3], **R** P **R** _[P]_ [1] [ˆ] _[I]_ [2] [ˆ] _[P]_ [3] and
**T** P **R** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ] _[I]_ [3] .


Tucker Decomposition (FSTD) [26–28].


Real-life structured data often admit good low-multilinear rank
approximations, and the FSTD provides such a low-rank Tucker
decomposition which is practical as it is directly expressed in terms of a
relatively small number of fibers of the data tensor.
For example, for a 3rd-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ] _[I]_ [3], for which an exact
rank- ( _R_ 1, _R_ 2, _R_ 3 ) Tucker representation exists, the FSTD selects _P_ _n_ ě _R_ _n_,
_n_ = 1, 2, 3, indices in each mode; this determines an intersection subtensor,
**W** P **R** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ] _[P]_ [3], so that the following exact Tucker representation can be
obtained (see Figure 3.10)



**X** = � **U** ; **C**, **R**, **T** �, (3.43)


where the core tensor is computed as **U** = **G** = � **W** ; **W** [:] ( 1 ) [,] **[ W]** [:] ( 2 ) [,] **[ W]** [:] ( 3 ) [�][, while]

the factor matrices, **C** P **R** _[I]_ [1] [ˆ] _[P]_ [2] _[P]_ [3], **R** P **R** _[I]_ [2] [ˆ] _[P]_ [1] _[P]_ [3], **T** P **R** _[I]_ [3] [ˆ] _[P]_ [1] _[P]_ [2], contain the
fibers which are the respective subsets of the columns **C**, rows **R** and tubes
**T** .
An equivalent Tucker representation is then given by


**X** = � **W** ; **CW** [:] ( 1 ) [,] **[ RW]** [:] ( 2 ) [,] **[ TW]** [:] ( 3 ) [�][.] (3.44)


Observe that for _N_ = 2, the TCA model simplifies into the MCA for a


95


(a)


(b)






















|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||**W**||R<br>**R**|R<br>**R**|
||||||
|**C**<br>|**C**|**C**|**C**|**C**|













































Figure 3.10: The Tucker decomposition of a low multilinear rank 3rdorder tensor using the cross-approximation approach. (a) Standard block
diagram. (b) Transformation from the TCA in the Tucker format, **X** –
**U** ˆ 1 **C** ˆ 2 **R** ˆ 3 **T**, into a standard Tucker representation, **X** – **W** ˆ 1 **B** [(] [1] [)] ˆ 2
**B** [(] [2] [)] ˆ 3 **B** [(] [3] [)] = � **W** ; **CW** [:] ( 1 ) [,] **[ RW]** [:] ( 2 ) [,] **[ TW]** [:] ( 3 ) [�][, with a prescribed core tensor] **[ W]** [.]


96


matrix case, **X** = **CUR**, for which the core matrix is **U** = � **W** ; **W** [:] ( 1 ) [,] **[ W]** [:] ( 2 ) [�] [=]

**W** [:] **WW** [:] = **W** [:] .

For a general case of an _N_ th-order tensor, we can show [26] that a
tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with a low multilinear rank t _R_ 1, _R_ 2, . . ., _R_ _N_ u,
where _R_ _n_ ď _I_ _n_, @ _n_, can be fully reconstructed via the TCA FSTD, **X** =
� **U** ; **C** [(] [1] [)], **C** [(] [2] [)], . . ., **C** [(] _[N]_ [)] �, using only _N_ factor matrices **C** [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[P]_ _[n]_ ( _n_ =
1, 2, . . ., _N_ ), built up from the fibers of the data and core tensors, **U** =
**G** = � **W** ; **W** [:] ( 1 ) [,] **[ W]** [:] ( 2 ) [, . . .,] **[ W]** [:] ( _N_ ) [�][, under the condition that the subtensor] **[ W]** [P]

**R** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ¨¨¨ˆ] _[P]_ _[N]_ with _P_ _n_ ě _R_ _n_, @ _n_, has the multilinear rank t _R_ 1, _R_ 2, . . ., _R_ _N_ u.
The selection of a minimum number of suitable fibers depends upon
a chosen optimization criterion. A strategy which requires access to
only a small subset of entries of a data tensor, achieved by selecting
the entries with maximum modulus within each single fiber, is given in

[26]. These entries are selected sequentially using a deflation approach,
thus making the tensor cross-approximation FSTD algorithm suitable for
the approximation of very large-scale but relatively low-order tensors
(including tensors with missing fibers or entries).
It should be noted that an alternative efficient way to estimate
subtensors **W**, **C**, **R** and **T** is to apply random projections as follows


**W** = **Z** = **X** ˆ 1 **Ω** 1 ˆ 2 **Ω** 2 ˆ 3 **Ω** 3 P **R** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ] _[P]_ [3],

**C** = **Z** ~~1~~ = **X** ˆ 2 **Ω** 2 ˆ 3 **Ω** 3 P **R** _[I]_ [1] [ˆ] _[P]_ [2] [ˆ] _[P]_ [3],

**R** = **Z** ~~2~~ = **X** ˆ 1 **Ω** 1 ˆ 3 **Ω** 3 P **R** _[P]_ [1] [ˆ] _[I]_ [2] [ˆ] _[P]_ [3],

**T** = **Z** ~~3~~ = **X** ˆ 1 **Ω** 1 ˆ 2 **Ω** 2 P **R** _[P]_ [1] [ˆ] _[P]_ [2] [ˆ] _[I]_ [3], (3.45)


where **Ω** _n_ P **R** _[P]_ _[n]_ [ˆ] _[I]_ _[n]_ with _P_ _n_ ě _R_ _n_ for _n_ = 1, 2, 3 are independent random
matrices. We explicitly assume that the multilinear rank t _P_ 1, _P_ 2, . . ., _P_ _N_ u of
approximated tensor to be somewhat larger than a true multilinear rank
t _R_ 1, _R_ 2, . . ., _R_ _N_ u of target tensor, because it is easier to obtain an accurate
approximation in this form.


**3.8** **Multiway Component Analysis (MWCA)**


**3.8.1** **Multilinear** **Component** **Analysis** **Using** **Constrained**
**Tucker Decomposition**


The great success of 2-way component analyses (PCA, ICA, NMF, SCA)
is largely due to the existence of very efficient algorithms for their
computation and the possibility to extract components with a desired


97


physical meaning, provided by the various flexible constraints exploited
in these methods. Without these constraints, matrix factorizations would
be less useful in practice, as the components would have only mathematical
but not physical meaning.
Similarly, to exploit the full potential of tensor
factorization/decompositions, it is a prerequisite to impose suitable
constraints on the desired components. In fact, there is much more
flexibility for tensors, since different constraints can be imposed on
the matrix factorizations in every mode _n_ a matricized tensor **X** ( _n_ ) (see
Algorithm 6 and Figure 3.11).
Such physically meaningful representation through flexible modewise constraints underpins the concept of multiway component analysis
(MWCA). The Tucker representation of MWCA naturally accommodates
such diversities in different modes. Besides the orthogonality, alternative
constraints in the Tucker format include statistical independence, sparsity,
smoothness and nonnegativity [42,43,213,235] (see Table 3.3).
The multiway component analysis (MWCA) based on the Tucker- _N_
model can be computed directly in two or three steps:


1. For each mode _n_ ( _n_ = 1, 2, . . ., _N_ ) perform model reduction and
matricization of data tensors sequentially, then apply a suitable set
of 2-way CA/BSS algorithms to the so reduced unfolding matrices, **X** ˜ ( _n_ ) . In each mode, we can apply different constraints and a different
2-way CA algorithms.


2. Compute the core tensor using, e.g., the inversion formula, **G** [ˆ] =
**X** ˆ 1 **B** [(] [1] [)] [:] ˆ 2 **B** [(] [2] [)] [:] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)] [:] . This step is quite important because
core tensors often model the complex links among the multiple
components in different modes.


3. Optionally, perform fine tuning of factor matrices and the core tensor
by the ALS minimization of a suitable cost function, e.g., } **X** ´
� **G** ; **B** [(] [1] [)], . . ., **B** [(] _[N]_ [)] �} [2] _F_ [, subject to specific imposed constraints.]


**3.9** **Analysis of Coupled Multi-block Matrix/Tensors –**
**Linked Multiway Component Analysis (LMWCA)**


We have shown that TDs provide natural extensions of blind source
separation (BSS) and 2-way (matrix) Component Analysis to multi-way
component analysis (MWCA) methods.


98


**X** (:,:, ) _k_

|...|Col2|Col3|
|---|---|---|
||||
||||



**X** _i_ (,:,:)



_K_ **X** (2) _K_





_J_ **X** (1) _J_


_I_ _..._



**S** 1



1





1



_R_
1


_R_
2

|A<br>3|Col2|
|---|---|
|**A**3<br>|T<br>|



_R_
3



(ICA)



_K_



_I_





_J_



_J_ _..._


**X** (:, :) _j,_ _I_ **X** (3) _I_


_K_ _..._



T

**B** 2



(SCA)



...



3



Figure 3.11: Multiway Component Analysis (MWCA) for a third-order
tensor via constrained matrix factorizations, assuming that the components
are: orthogonal in the first mode, statistically independent in the second
mode and sparse in the third mode.


In addition, TDs are suitable for the coupled multiway analysis of
multi-block datasets, possibly with missing values and corrupted by noise.
To illustrate the simplest scenario for multi-block analysis, consider the
block matrices, **X** [(] _[k]_ [)] P **R** _[I]_ [ˆ] _[J]_, which need to be approximately jointly
factorized as


**X** [(] _[k]_ [)] – **AG** [(] _[k]_ [)] **B** [T], ( _k_ = 1, 2, . . ., _K_ ), (3.46)


where **A** P **R** _[I]_ [ˆ] _[R]_ [1] and **B** P **R** _[J]_ [ˆ] _[R]_ [2] are common factor matrices and **G** [(] _[k]_ [)] P
**R** _[R]_ [1] [ˆ] _[R]_ [2] are reduced-size matrices, while the number of data matrices _K_ can
be huge (hundreds of millions or more matrices). Such a simple model is
referred to as the Population Value Decomposition (PVD) [51]. Note that
the PVD is equivalent to the unconstrained or constrained Tucker-2 model,
as illustrated in Figure 3.12. In a special case with square diagonal matrices,
**G** [(] _[k]_ [)], the model is equivalent to the CP decomposition and is related to joint
matrix diagonalization [31, 56, 203]. Furthermore, if **A** = **B** then the PVD
model is equivalent to the RESCAL model [153].
Observe that the PVD/Tucker-2 model is quite general and flexible,
since any high-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ (with _N_ ą 3), can be reshaped
and optionally permuted into a “skinny and tall” 3rd-order tensor, **X** [r] P
**R** _[J]_ [ ˆ] _[ J]_ [ ˆ] _[ K]_, with e.g., _I_ = _I_ 1, _J_ = _I_ 2 and _K_ = _I_ 3 _I_ 4 ¨ ¨ ¨ _I_ _N_, for which
PVD/Tucker-2 Algorithm 8 can be applied.


99


(a)


(b)



( _I_ × _J_ ) ( _I_ × _R_ 1 ) ( _R  R_ 1 × 2 ) ( _R_ 2 × _J_ )









(2) **A** **G** (2)







( _K_ ) **A** **G** ( _K_ )














|K<br>G BT|Col2|
|---|---|
|**G**|T<br>**B**|



( _I_ × _J_ × _K_ )



_R_ 2


( _I_ × _R_ 1 ) ( _R  R_ 1   2 × × _K_ ) ( _R_ 2 [×] _J_ )



Figure 3.12: Concept of the Population Value Decomposition (PVD). (a)
Principle of simultaneous multi-block matrix factorizations. (b) Equivalent
representation of the PVD as the constrained or unconstrained Tucker-2
decomposition, **X** – **G** ˆ 1 **A** ˆ 2 **B** . The objective is to find the common
factor matrices, **A**, **B** and the core tensor, **G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[K]_ .


As previously mentioned, various constraints, including sparsity,
nonnegativity or smoothness can be imposed on the factor matrices, **A** and
**B**, to obtain physically meaningful and unique components.
A simple SVD/QR based algorithm for the PVD with orthogonality
constraints is presented in Algorithm 7 [49,51,219]. However, it should be
noted that this algorithm does not provide an optimal solution in the sense


100


**Algorithm 7** : **Population Value Decomposition (PVD) with**
**orthogonality constraints**

**Input:** A set of matrices **X** _k_ P **R** _[I]_ [ˆ] _[J]_, for _k_ = 1, . . ., _K_ (typically, _K_ " maxt _I_, _J_ u)
**Output:** Factor matrices **A** P **R** _[I]_ [ˆ] _[R]_ [1], **B** P **R** _[J]_ [ˆ] _[R]_ [2] and **G** _k_ P **R** _[R]_ [1] [ˆ] _[R]_ [2],
with orthogonality constraints **A** [T] **A** = **I** _R_ 1 and **B** [T] **B** = **I** _R_ 2
1: **for** _k_ = 1 to _K_ **do**
2: Perform truncated SVD, **X** _k_ = **U** _k_ **S** _k_ **V** [T] _k_ [, using] _[ R]_ [ largest singular]
values

3: **end for**

4: Construct short and wide matrices:
**U** = [ **U** 1 **S** 1, . . ., **U** _K_ **S** _K_ ] P **R** _[I]_ [ˆ] _[KR]_ and **V** = [ **V** 1 **S** 1, . . ., **V** _K_ **S** _K_ ] P **R** _[J]_ [ˆ] _[KR]_

5: Perform SVD (or QR) for the matrices **U** and **V**
Obtain common orthogonal matrices **A** and **B** as left-singular
matrices of **U** and **V**, respectively
6: **for** _k_ = 1 to _K_ **do**
7: Compute **G** _k_ = **A** [T] **X** _k_ **B**
8: **end for**


**Algorithm 8** : **Orthogonal Tucker-2 decomposition with a prescribed**
**approximation accuracy [170]**

**Input:** A 3rd-order tensor **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_ (typically, _K_ " maxt _I_, _J_ u)
and estimation accuracy _ε_
**Output:** A set of orthogonal matrices **A** P **R** _[I]_ [ˆ] _[R]_ [1], **B** P **R** _[J]_ [ˆ] _[R]_ [2] and core tensor
**G** P **R** _[R]_ [1] [ˆ] _[R]_ [2] [ˆ] _[K]_, which satisfies the constraint } **X** ´ **G** ˆ 1 **A** ˆ **B** } [2] _F_ [ď] _[ ε]_ [2] [, s.t,]
**A** [T] **A** = **I** _R_ 1 and **B** [T] **B** = **I** _R_ 2 .
1: Initialize **A** = **I** _I_ P **R** _[I]_ [ˆ] _[I]_, _R_ 1 = _I_
2: **while** not converged or iteration limit is not reached **do**

3: Compute the tensor **Z** [(] [1] [)] = **X** ˆ 1 **A** [T] P **R** _[R]_ [1] [ˆ] _[J]_ [ˆ] _[K]_

4: Compute EVD of a small matrix **Q** 1 = **Z** [(] ( [1] 2 [)] ) **[Z]** [(] ( 2 [1] [)] ) [ T] P **R** _[J]_ [ˆ] _[J]_ as

**Q** 1 = **B** diag � _λ_ 1, ¨ ¨ ¨, _λ_ _R_ 2 � **B** [T], such that
_R_ 2
ř _r_ 2 = 1 _[λ]_ _[r]_ 2 [ě ][}] **[X]** [}] [2] _F_ [´] _[ ε]_ [2] [ ě][ ř] _r_ _[R]_ 2 [2] = [´] 1 [1] _[λ]_ _[r]_ 2
5: Compute tensor **Z** [(] [2] [)] = **X** ˆ 2 **B** [T] P **R** _[I]_ [ˆ] _[R]_ [2] [ˆ] _[K]_

6: Compute EVD of a small matrix **Q** 2 = **Z** [(] ( [2] 1 [)] ) **[Z]** [(] ( 1 [2] [)] ) [ T] P **R** _[I]_ [ˆ] _[I]_ as

**Q** 2 = **A** diag � _λ_ 1, . . ., _λ_ _R_ 1 � **A** [T], such that
_R_ 1
ř _r_ 1 = 1 _[λ]_ _[r]_ 1 [ě ][}] **[X]** [}] [2] _F_ [´] _[ ε]_ [2] [ ě][ ř] _r_ _[R]_ 1 [1] = [´] 1 [1] _[λ]_ _[r]_ 1
7: **end while**
8: Compute the core tensor **G** = **X** ˆ 1 **A** [T] ˆ 2 **B** [T]

9: **return A**, **B** and **G** .


101


Figure 3.13: Linked Multiway Component Analysis (LMWCA) for coupled
3rd-order data tensors **X** [(] [1] [)], . . ., **X** [(] _[K]_ [)] ; these can have different dimensions
in every mode, except for the mode-1 for which the size is _I_ 1 for all
**X** [(] _[k]_ [)] .
Linked Tucker-1 decompositions are then performed in the form
**X** [(] _[k]_ [)] – **G** [(] _[k]_ [)] ˆ 1 **B** [(] [1,] _[k]_ [)], where partially correlated factor matrices are **B** [(] [1,] _[k]_ [)] =

[ **B** _C_ [(] [1] [)] [,] **[ B]** [(] _I_ [1,] _[k]_ [)] ] P **R** _[I]_ [1] [ˆ] _[R]_ _[k]_, ( _k_ = 1, 2, . . ., _K_ ) . The objective is to find the common

components, **B** _C_ [(] [1] [)] P **R** _[I]_ [1] [ˆ] _[C]_, and individual components, **B** [(] _I_ [1,] _[k]_ [)] P **R** _[I]_ [1] [ˆ] [(] _[R]_ _[k]_ [´] _[C]_ [)],
where _C_ ď mint _R_ 1, . . ., _R_ _K_ u is the number of common components in
mode-1.


of the absolute minimum of the cost function, [ř] _k_ _[K]_ = 1 [}] **[X]** _[k]_ [´] **[ AG]** _[k]_ **[B]** [T] [}] [2] _F_ [, and]
for data corrupted by Gaussian noise, better performance can be achieved
using the HOOI-2 given in Algorithm 4, for _N_ = 3. An improved PVD
algorithm referred to as Tucker-2 algorithm is given in Algorithm 8 [170].
**Linked MWCA.** Consider the analysis of multi-modal high-dimensional


102


data collected under the same or very similar conditions, for example, a set
of EEG and MEG or EEG and fMRI signals recorded for different subjects
over many trials and under the same experimental configurations and
mental tasks. Such data share some common latent (hidden) components
but can also have their own independent features. As a result, it is
advantageous and natural to analyze such data in a linked way instead
of treating them independently. In such a scenario, the PVD model can be
generalized to multi-block matrix/tensor datasets [38,237,239].
The linked multiway component analysis (LMWCA) for multi-block
tensor data can therefore be formulated as a set of approximate
simultaneous (joint) Tucker- ( 1, _N_ ) decompositions of a set of data tensors,

**X** [(] _[k]_ [)] P **R** _[I]_ 1 [(] _[k]_ [)] [ˆ] _[I]_ 2 [(] _[k]_ [)] [ˆ¨¨¨ˆ] _[I]_ _N_ [(] _[k]_, with [)] _I_ 1 [(] _[k]_ [)] = _I_ 1 for _k_ = 1, 2, . . ., _K_, in the form (see
Figure 3.13)


**X** [(] _[k]_ [)] = **G** [(] _[k]_ [)] ˆ 1 **B** [(] [1,] _[k]_ [)], ( _k_ = 1, 2, . . . _K_ ) (3.47)


where each factor (component) matrix, **B** [(] [1,] _[k]_ [)] = [ **B** _C_ [(] [1] [)] [,] **[ B]** [(] _I_ [1,] _[k]_ [)] ] P **R** _[I]_ [1] [ˆ] _[R]_ _[k]_,

comprises two sets of components: (1) Components **B** _C_ [(] [1] [)] P **R** _[I]_ [1] [ˆ] _[C]_ (with
0 ď _C_ ď _R_ _k_ ), @ _k_, which are common for all the available blocks
and correspond to identical or maximally correlated components, and (2)
**B** [(] [1,] _[k]_ [)] P **R** _[I]_ [1] [ˆ] [(] _[R]_ _[k]_ [´] _[C]_ [)]
components _I_, which are different independent processes
for each block, _k_, these can be, for example, latent variables independent
of excitations or stimuli/tasks. The objective is therefore to estimate
**B** [(] [1] [)]
the common (strongly correlated) components, _C_ [, and statistically]

independent (individual) components, **B** [(] _I_ [1,] _[k]_ [)] [38].

If **B** [(] _[n]_ [,] _[k]_ [)] = **B** _C_ [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ for a specific mode _n_ (in our case _n_ = 1),
and under the additional assumption that the block tensors are of the
same order and size, the problem simplifies into generalized Common
Component Analysis or tensor Population Value Decomposition (PVD)
and can be solved by concatenating all data tensors along one mode,
followed by constrained Tucker or CP decompositions [173].
In a more general scenario, when _C_ _n_ ă _R_ _n_, we can unfold each data
tensor **X** [(] _[k]_ [)] in the common mode, and perform a set of simultaneous matrix
factorizations, e.g., **X** [(] ( 1 _[k]_ [)] ) [–] **[ B]** _C_ [(] [1] [)] **[A]** _C_ [(] [1,] _[k]_ [)] + **B** [(] _I_ [1,] _[k]_ [)] **A** [(] _I_ [1,] _[k]_ [)], through solving the


103


(a)


(b)









=


|G ( N|Col2|
|---|---|
|||
||_I_|


|I<br>N<br>I<br>1|X<br>...<br>...<br>I<br>n|
|---|---|
|_I_2<br>_I_3<br>|_I_2<br>_I_3<br>|







_J_ 2 _J_ _J_ _n_ _J_ 1 _=I_





_J_ 1 _=I_ 1 _J_ 2 _=I_ 2 _J_ 3 _J_ _N_ -1 _J_ N




|J<br>N<br>J<br>1|Y<br>...<br>...<br>J<br>n|
|---|---|
|_J_2<br><br>_J_3<br>|_J_2<br><br>_J_3<br>|








|I<br>N<br>I<br>1|X<br>...<br>...<br>I<br>n|
|---|---|
|_I_2<br>_I_3<br>|_I_2<br>_I_3<br>|








|J<br>N<br>J =I<br>1 1|Y<br>...<br>...<br>J<br>n|
|---|---|
|_J_2<br><br>_J_3|_J_2<br><br>_J_3|



Figure 3.14: Conceptual models of generalized Linked Multiway
Component Analysis (LMWCA) applied to the cores of high-order TNs.
The objective is to find a suitable tensor decomposition which yields the
maximum number of cores that are as much correlated as possible. (a)
Linked Tensor Train (TT) networks. (b) Linked Hierarchical Tucker (HT)
networks with the correlated cores indicated by ellipses in broken lines.


104


constrained optimization problems



min



_K_
ÿ } **X** [(] ( 1 _[k]_ [)] ) [´] **[ B]** _C_ [(] [1] [)] **[A]** _C_ [(] [1,] _[k]_ [)] ´ **B** [(] _I_ [1,] _[k]_ [)] **A** [(] _I_ [1,] _[k]_ [)] } _F_

_k_ = 1

+ _P_ ( **B** _C_ [(] [1] [)] [)] [,] _[ s]_ [.] _[t]_ [.] **[ B]** _C_ [(] [1] [)] [ T] **B** [(] _I_ [1,] _[k]_ [)] = **0** @ _k_,



(3.48)



where the symbol _P_ denotes the penalty terms which impose additional
**B** [(] [1] [)]
constraints on the common components, _C_ [, in order to extract as many]
common components as possible. In the special case of orthogonality
constraints, the problem can be transformed into a generalized eigenvalue
**B** [(] [1] [)]
problem. The key point is to assume that common factor submatrices, _C_ [,]
are present in all data blocks and hence reflect structurally complex latent
(hidden) and intrinsic links between the data blocks. _In practice, the number_
_of common components, C, is unknown and should be estimated_ [237].
The linked multiway component analysis (LMWCA) model
complements currently available techniques for group component analysis
and feature extraction from multi-block datasets, and is a natural extension
of group ICA, PVD, and CCA/PLS methods (see [38, 231, 237, 239] and
references therein). Moreover, the concept of LMWCA can be generalized
to tensor networks, as illustrated in Figure 3.14.


**3.10** **Nonlinear Tensor Decompositions – Infinite**
**Tucker**


The Infinite Tucker model and its modification, the Distributed Infinite
Tucker (DinTucker), generalize the standard Tucker decomposition
to infinitely dimensional feature spaces using kernel and Bayesian
approaches [201,225,233].
Consider the classic Tucker- _N_ model of an _N_ th-order tensor **X** P
**R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, given by


**X** = **G** ˆ 1 **B** [(] [1] [)] ˆ 2 **B** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **B** [(] _[N]_ [)]

= � **G** ; **B** [(] [1] [)], **B** [(] [2] [)], . . ., **B** [(] _[N]_ [)] � (3.49)


in its vectorized version


vec ( **X** ) = ( **B** [(] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **B** [(] _[N]_ [)] ) vec ( **G** ) .


Furthermore, assume that the noisy data tensor is modeled as


**Y** = **X** + **E**, (3.50)


105


where **E** represents the tensor of additive Gaussian noise. Using the
Bayesian framework and tensor-variate Gaussian processes (TGP) for
Tucker decomposition, a standard normal prior can be assigned over each
entry, _g_ _r_ 1, _r_ 2,..., _r_ _N_, of an _N_ th-order core tensor, **G** P **R** _[R]_ [1] [ˆ¨¨¨ˆ] _[R]_ _[N]_, in order to
marginalize out **G** and express the probability density function of tensor
**X** [36,225,233] in the form


_p_ **X** | **B** [(] [1] [)], . . ., **B** [(] _[N]_ [)] [�] = _N_ vec ( **X** ) ; **0**, **C** [(] [1] [)] b _L_ ¨ ¨ ¨ b _L_ **C** [(] _[N]_ [)] [�]
� �




[1]

2 [}][�] **[X]** [;] [ (] **[C]** [(] [1] [)] [)] [´][1/2] [, . . .,] [ (] **[C]** [(] _[N]_ [)] [)] [´][1/2] [�][}] [2] _F_
�



´ [1]
exp 2
= �



(3.51)
( 2 _π_ ) _[I]_ [/2] ~~[ ś]~~ _n_ _[N]_ = 1 [|] **[C]** [(] _[n]_ [)] [|] [´] _[I]_ [/] [(] [2] _[I]_ _[n]_ [)]



where _I_ = [ś] _n_ _[I]_ _[n]_ [ and] **[ C]** [(] _[n]_ [)] [ =] **[ B]** [(] _[n]_ [)] **[ B]** [(] _[n]_ [)] [ T] [ P] **[ R]** _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [ for] _[ n]_ [ =] [ 1, 2, . . .,] _[ N]_ [.]
In order to model unknown, complex, and potentially nonlinear
interactions between the latent factors, each row, **b** [¯] _i_ [(] _n_ _[n]_ [)] P **R** [1][ˆ] _[R]_ _[n]_, within **B** [(] _[n]_ [)],

is replaced by a nonlinear feature transformation Φ ( **b** [¯] _i_ [(] _n_ _[n]_ [)] [)] [ using the kernel]
trick [232], whereby the nonlinear covariance matrix **C** [(] _[n]_ [)] = _k_ ( **B** [(] _[n]_ [)], **B** [(] _[n]_ [)] )
replaces the standard covariance matrix, **B** [(] _[n]_ [)] **B** [(] _[n]_ [)] [ T] . Using such a nonlinear
feature mapping, the original Tucker factorization is performed in an
infinite feature space, while Eq. (3.51) defines a Gaussian process (GP) on
a tensor, called the Tensor-variate GP (TGP), where the inputs come from a
set of factor matrices t **B** [(] [1] [)], . . ., **B** [(] _[N]_ [)] u = t **B** [(] _[n]_ [)] u.
For a noisy data tensor **Y**, the joint probability density function is given
by


_p_ ( **Y**, **X**, t **B** [(] _[n]_ [)] u ) = _p_ ( t **B** [(] _[n]_ [)] u ) _p_ ( **X** | t **B** [(] _[n]_ [)] u ) _p_ ( **Y** | **X** ) . (3.52)


To improve scalability, the observed noisy tensor **Y** can be split into _K_
subtensors t **Y** 1, . . ., **Y** _K_ u, whereby each subtensor **Y** _k_ is sampled from its
own GP based model with factor matrices, t **B** [˜] [(] _k_ _[n]_ [)] [u] [ =] [ t][ ˜] **[B]** [(] _k_ [1] [)] [, . . ., ˜] **[B]** [(] _k_ _[N]_ [)] u. The
factor matrices can then be merged via a prior distribution



_p_ ( t **B** [˜] [(] _k_ _[n]_ [)] [u|t] **[B]** [(] _[n]_ [)] [u] [)] =


=



_N_
ź _p_ ( **B** [˜] [(] _k_ _[n]_ [)] [|] **[B]** [(] _[n]_ [)] [)]

_n_ = 1


_N_
ź _N_ ( vec ( **B** [˜] [(] _k_ _[n]_ [)] [)] [|][vec] [(] **[B]** [(] _[n]_ [)] [))] [,] _[ λ]_ **[I]** [)] [,] (3.53)

_n_ = 1



where _λ_ ą 0 is a variance parameter which controls the similarity between
the corresponding factor matrices. The above model is referred to as
DinTucker [233].


106


The full covariance matrix, **C** [(] [1] [)] b ¨ ¨ ¨ b **C** [(] _[N]_ [)] P **R** ś _n_ _[I]_ _[n]_ [ˆ][ś] _n_ _[I]_ _[n]_, may have
a prohibitively large size and can be extremely sparse. For such cases,
an alternative nonlinear tensor decomposition model has been recently
developed, which does not, either explicitly or implicitly, exploit the
Kronecker structure of covariance matrices [41]. Within this model, for
each tensor entry, _x_ _i_ 1,..., _i_ _N_ = _x_ **i**, with **i** = ( _i_ 1, _i_ 2, . . ., _i_ _N_ ), an input vector
**b** **i** is constructed by concatenating the corresponding row vectors of factor
(latent) matrices, **B** [(] _[n]_ [)], for all _N_ modes, as

**b** **i** = [ **b** [¯] _i_ [(] 1 [1] [)] [, . . ., ¯] **[b]** _i_ [(] _N_ _[N]_ [)] []] [ P] **[ R]** [1][ˆ][ř] _n_ _[N]_ = 1 _[R]_ _[n]_ . (3.54)


We can formalize an (unknown) nonlinear transformation as

_x_ **i** = _f_ ( **b** **i** ) = _f_ ([ **b** [¯] _i_ [(] 1 [1] [)] [, . . ., ¯] **[b]** _i_ [(] _N_ _[N]_ [)] [])] (3.55)


for which a zero-mean multivariate Gaussian distribution is determined
by **B** _S_ = t **b** **i** 1, . . ., **b** **i** _M_ u and **f** _S_ = t _f_ ( **b** **i** 1 ), . . ., _f_ ( **b** **i** _M_ ) u. This allows us to
construct the following probability function


_p_ **f** _S_ |t **B** [(] _[n]_ [)] u = _N_ ( **f** _S_ | **0**, _k_ ( **B** _S_, **B** _S_ )), (3.56)
� �


where _k_ ( ¨, ¨ ) is a nonlinear covariance function which can be expressed as
_k_ ( **b** **i**, **b** **j** ) = _k_ (([ **b** [¯] _i_ [(] 1 [1] [)] [, . . ., ¯] **[b]** _i_ [(] _N_ _[N]_ [)] [])] [,] [ ([] [ ¯] **[b]** [(] _j_ 1 [1] [)] [, . . ., ¯] **[b]** [(] _j_ _N_ _[N]_ [)] []))] [ and] _[ S]_ [ = [] **[i]** [1] [, . . .,] **[ i]** _[M]_ []] [.]
In order to assign a standard normal prior over the factor matrices,
t **B** [(] _[n]_ [)] u, we assume that for selected entries, **x** = [ _x_ **i** 1, . . ., _x_ **i** _M_ ], of a tensor **X**,
the noisy entries, **y** = [ _y_ **i** 1, . . ., _y_ **i** _M_ ], of the observed tensor **Y**, are sampled
from the following joint probability model


_p_ ( **y**, **x**, t **B** [(] _[n]_ [)] u ) (3.57)



=



_N_
ź _N_ ( vec ( **B** [(] _[n]_ [)] ) | **0**, **I** ) _N_ ( **x** | **0**, _k_ ( **B** _S_, **B** _S_ )) _N_ ( **y** | **x**, _β_ [´][1] **I** ),

_n_ = 1



where _β_ represents noise variance.
These nonlinear and probabilistic models can be potentially applied
for data tensors or function-related tensors comprising large number of
entries, typically with millions of non-zero entries and billions of zero
entries. Even if only nonzero entries are used, exact inference of the
above nonlinear tensor decomposition models may still be intractable. To
alleviate this problem, a distributed variational inference algorithm has
been developed, which is based on sparse GP, together with an efficient
MapReduce framework which uses a small set of inducing points to break
up the dependencies between random function values [204,233].


107


**Chapter 4**


**Tensor Train Decompositions:**
**Graphical Interpretations and**
**Algorithms**


Efficient implementation of the various operations in tensor train (TT)
formats requires compact and easy-to-understand mathematical and
graphical representations [37, 39]. To this end, we next present
mathematical formulations of the TT decompositions and demonstrate
their advantages in both theoretical and practical scenarios.


**4.1** **Tensor Train Decomposition – Matrix Product**
**State**


The tensor train (TT/MPS) representation of an _N_ th-order data tensor, **X** P
**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, can be described in several equivalent forms (see Figures 4.1,
4.2 and Table 4.1) listed below:


1. The entry-wise scalar form, given by



_x_ _i_ 1, _i_ 2,..., _i_ _N_ –



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ _g_ 1, [(] [1] _i_ [)] 1, _r_ 1 _[g]_ _r_ [(] 1 [2], [)] _i_ 2, _r_ 2 [¨ ¨ ¨] _[ g]_ _r_ [(] _N_ _[N]_ ´ [)] 1, _i_ _N_,1 [.]

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



(4.1)


2. The slice representation (see Figure 2.19) in the form

_x_ _i_ 1, _i_ 2,..., _i_ _N_ – **G** _i_ [(] 1 [1] [)] **G** _i_ [(] 2 [2] [)] [¨ ¨ ¨] **[ G]** _i_ [(] _N_ _[N]_ [)] [,] (4.2)


108


(a)


(b)













_R_
1













(4)


3
















|(1)|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|(1)||||||||
|(1)|_R_|||_R_2<br>**G**<br>(2)<br>**G**<br>(3)<br>_R_3|||_I_4<br>**G**<br><br>**g**<br>)<br>1<br>|
||1<br>(1<br>_r_<br>**g**<br>1<br>)<br>_R_|)|_R_|_I_2<br>_I_3<br>_I_2<br>_R_2<br>_R_3<br>_R_3<br>_R_2<br>1<br>1<br>2<br>(2)<br>,<br>_r r_<br>**g**<br>2<br>3<br>(3)<br>,<br>_r r_<br>**g**<br>3<br>_I_|_I_4<br><br>(_R_<br>3|||
|||||||||
|||||||_I_<br>|_I_<br>|
|||||1<br>2<br>2<br>(<br>)<br>_R_<br>_I_<br>_R_<br>2<br>3<br>3<br>(<br>)<br>_R_<br>_I_<br>_R_<br>|1<br>2<br>2<br>(<br>)<br>_R_<br>_I_<br>_R_<br>2<br>3<br>3<br>(<br>)<br>_R_<br>_I_<br>_R_<br>|1<br>2<br>2<br>(<br>)<br>_R_<br>_I_<br>_R_<br>2<br>3<br>3<br>(<br>)<br>_R_<br>_I_<br>_R_<br>|1<br>2<br>2<br>(<br>)<br>_R_<br>_I_<br>_R_<br>2<br>3<br>3<br>(<br>)<br>_R_<br>_I_<br>_R_<br>|
|||||||||



(1)
**G**


_I_
1

_R_
1



(2)
**G**


_I_ 2


_R_ 1 _R_ 2

|Col1|Col2|I|
|---|---|---|
||||
||||
||||



_R_
2



(3)
**G**

|Col1|Col2|Col3|I|
|---|---|---|---|
|||||
|||||
|||||
|||||



_R_
3



(4)
**G**



_I_ 3



_I_ 4


|Col1|Col2|
|---|---|
|||
|||
|||
|||



( _I_ 1 _R_ 1 ) ( _R I_ 1 2 _R_ 2 ) ( _R I_ 2 3 _R_ 3 ) ( _R I_ 3 4 1)


Figure 4.1: TT decomposition of a 4th-order tensor, **X**, for which the TT rank
is _R_ 1 = 3, _R_ 2 = 4, _R_ 3 = 5. (a) (Upper panel) Representation of the TT
via a multilinear product of the cores, **X** – **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] **G** [(] [3] [)] ˆ [1] **G** [(] [4] [)] =
xx **G** [(] [1] [)], **G** [(] [2] [)], **G** [(] [3] [)], **G** [(] [4] [)] yy, and (lower panel) an equivalent representation via the
outer product of mode-2 fibers (sum of rank-1 tensors) in the form, **X** –
ř _rR_ 11 = 1 ř _rR_ 22 = 1 ř _rR_ 33 = 1 ř _rR_ 44 = 1 [(] **[g]** [(] _r_ [1] 1 [)] ˝ **g** [(] _r_ [2] 1 [)], _r_ 2 [˝] **[ g]** [(] _r_ [3] 2 [)], _r_ 3 [˝] **[ g]** [(] _r_ [4] 3 [)] [)] [. (b) TT decomposition]
in a vectorized form represented via strong Kronecker products of block matrices,
**x** – **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| **G** [r] [(] [3] [)] |b| **G** [r] [(] [4] [)] P **R** _[I]_ [1] _[I]_ [2] _[I]_ [3] _[I]_ [4], where the block matrices are defined
as **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, with block vectors **g** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ] [1] [,] _[ n]_ [ =] [ 1, . . ., 4 and]
_R_ 0 = _R_ 4 = 1.


109


Table 4.1: Equivalent representations of the Tensor Train decomposition
(MPS with open boundary conditions) approximating an _N_ th-order tensor
**X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ . It is assumed that the TT rank is **r** _TT_ = t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u,
with _R_ 0 = _R_ _N_ = 1.


Tensor representation: Multilinear products of TT-cores


**X** = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [)] P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_


with the 3rd-order cores **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, ( _n_ = 1, 2, . . ., _N_ )


Tensor representation: Outer products



**X** =



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ **g** [(] 1, [1] [)] _r_ 1 [˝] **[ g]** [(] _r_ [2] 1 [)], _r_ 2 [˝ ¨ ¨ ¨ ˝] **[ g]** [(] _r_ _[N]_ _N_ [´] ´2 [1], [)] _r_ _N_ ´1 [˝] **[ g]** [(] _r_ _[N]_ _N_ [)] ´1, 1

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



where **g** [(] _r_ _[n]_ _n_ [)] ´1, _r_ _n_ [=] **[ G]** [(] _[n]_ [)] [(] _[r]_ _n_ ´1 [, :,] _[ r]_ _n_ [)] [ P] **[ R]** _[I]_ _[n]_ [ are fiber vectors.]


Vector representation: Strong Kronecker products


**x** = **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **G** [r] [(] _[N]_ [)] P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_, where


r
**G** [(] _[n]_ [)] P **R** _[R]_ _n_ ´1 _[I]_ _n_ [ˆ] _[R]_ _n_ are block matrices with blocks **g** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_


Scalar representation



_x_ _i_ 1, _i_ 2,..., _i_ _N_ =



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ _g_ [(] 1, [1] [)] _i_ 1, _r_ 1 _[g]_ [(] _r_ [2] 1 [)], _i_ 2, _r_ 2 [¨ ¨ ¨] _[ g]_ [(] _r_ _[N]_ _N_ [´] ´2 [1], [)] _i_ _N_ ´1, _r_ _N_ ´1 _[g]_ [(] _r_ _[N]_ _N_ [)] ´1, _i_ _N_,1

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



where _g_ [(] _r_ _[n]_ _n_ [)] ´1, _i_ _n_, _r_ _n_ [are entries of a 3rd-order core] **[ G]** [(] _[n]_ [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_


Slice (MPS) representation


_x_ _i_ 1, _i_ 2,..., _i_ _N_ = **G** _i_ [(] 1 [1] [)] **G** _i_ [(] 2 [2] [)] ¨ ¨ ¨ **G** _i_ [(] _N_ _[N]_ [)] [,] where


**G** _i_ [(] _n_ _[n]_ [)] = **G** [(] _[n]_ [)] ( :, _i_ _n_, : ) P **R** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ are lateral slices of **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_


110


Table 4.2: Equivalent representations of the Tensor Chain (TC)
decomposition (MPS with periodic boundary conditions) approximating
an _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ . It is assumed that the TC rank is
**r** _TC_ = t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1, _R_ _N_ u.


Tensor representation: Trace of multilinear products of cores


**X** = Tr ( **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [)] ) P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_


with the 3rd-order cores **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, _R_ 0 = _R_ _N_, _n_ = 1, 2, . . ., _N_


Tensor/Vector representation: Outer/Kronecker products



**X** =


**x** =



_R_ 1, _R_ 2,..., _R_ _N_
ÿ **g** [(] _r_ [1] _N_ [)], _r_ 1 [˝] **[ g]** [(] _r_ [2] 1 [)], _r_ 2 [˝ ¨ ¨ ¨ ˝] **[ g]** [(] _r_ _[N]_ _N_ [)] ´1, _r_ _N_ [P] **[ R]** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_

_r_ 1, _r_ 2,..., _r_ _N_ = 1


_R_ 1, _R_ 2,..., _R_ _N_
ÿ **g** [(] _r_ [1] _N_ [)], _r_ 1 [b] _L_ **[g]** [(] _r_ [2] 1 [)], _r_ 2 [b] _L_ [¨ ¨ ¨ b] _L_ **[g]** [(] _r_ _[N]_ _N_ [)] ´1, _r_ _N_ [P] **[ R]** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_

_r_ 1, _r_ 2,..., _r_ _N_ = 1



where **g** [(] _r_ _[n]_ _n_ [)] ´1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ are fiber vectors within] **[ G]** [(] _[n]_ [)] [(] _[r]_ _n_ ´1 [, :,] _[ r]_ _n_ [)] [ P] **[ R]** _[I]_ _[n]_


Vector representation: Strong Kronecker products



**x** =



_R_ _N_
ÿ ( **G** [r] [(] _r_ [1] _N_ [)] [|b|][ r] **[G]** [(] [2] [)] [ |b| ¨ ¨ ¨ |b|][ r] **[G]** [(] _[N]_ [´][1] [)] [ |b|][ r] **[G]** [(] _r_ _[N]_ _N_ [)] [)] [ P] **[ R]** _[I]_ [1] _[ I]_ [2] [¨¨¨] _[I]_ _[N]_ where

_r_ _N_ = 1



r
**G** [(] _[n]_ [)] P **R** _[R]_ _n_ ´1 _[I]_ _n_ [ˆ] _[R]_ _n_ are block matrices with blocks **g** [(] _r_ _[n]_ _n_ [)] ´1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [,]


r
**G** [(] _r_ [1] _N_ [)] [P] **[ R]** _[I]_ [1] [ˆ] _[R]_ [1] [ is a matrix with blocks (columns)] **[ g]** [(] _r_ [1] _N_ [)], _r_ 1 [P] **[ R]** _[I]_ [1] [,]


r
**G** [(] _r_ _[N]_ _N_ [)] [P] **[ R]** _[R]_ _[N]_ [´][1] _[ I]_ _[N]_ [ˆ][1] [ is a block vector with blocks] **[ g]** [(] _r_ _[N]_ _N_ [)] ´1, _r_ _N_ [P] **[ R]** _[I]_ _[N]_


Scalar representations



_x_ _i_ 1, _i_ 2,..., _i_ _N_ = tr ( **G** _i_ [(] 1 [1] [)] **[G]** _i_ [(] 2 [2] [)] [¨ ¨ ¨] **[ G]** _i_ [(] _N_ _[N]_ [)] [) =]



_R_ _N_
ÿ



ÿ ( **g** _r_ [(] [1] _N_ [)], [ T] _i_ 1, : **[G]** _i_ [(] 2 [2] [)] [¨ ¨ ¨] **[ G]** _i_ [(] _N_ _[N]_ ´ [´] 1 [1] [)] **g** [(] :, _[N]_ _i_ _N_ [)], _r_ _N_ [)]

_r_ _N_ = 1



where **g** _r_ [(] [1] _N_ [)], _i_ 1, : [=] **[ G]** [(] [1] [)] [(] _[r]_ _[N]_ [,] _[ i]_ [1] [, :] [)] [ P] **[ R]** _[R]_ [1] [,] **[ g]** [(] :, _[N]_ _i_ _N_ [)], _r_ _N_ [=] **[ G]** [(] _[N]_ [)] [(] [:,] _[ i]_ _[N]_ [,] _[ r]_ _[N]_ [)] [ P] **[ R]** _[R]_ _[N]_ [´][1]


111


(a)


(b)


(c)



2



**x**


_I = I I_ 1 2 _I_ _N_



**X**



3



(2)



(1) (2) ( ) _n_

**G** **G** **G**



_R_



(1) (2) ( ) _n_

**G** **G** **G**



**G**



_R_



( ) _N_



**G**



_R_ 1 _R_ 2 _R_ _n_ -1 _R_ _n_ _R_ _N_ -1



_I_ 1 _I_ 2 _I_ _n_ _I_ _N_



_R_



_R_



_I_
_N_

_R_
_N_ -1



_R_



1 _R_ 1 1 1 _R_ _n_ -1 1 1



_I_
1



_I_
2



(1 _I_ 1 _R_ 1 ) ( _R_ 1 _I_ 2 _R_ 2 )



_I_
_n_


( _R_ _n_ ~~1~~ _I_ _n_ _R_ _n_ ) ( _R_ _N_ ~~1~~ _I_ _N_ 1)



(2)
**G**



( _I_ 2 1)



( _N_ )
**G**





( _[I]_ _n_ 1)



( _I_ _N_ 1)







_I_
1



(1)
**G**


_R_
1



_R_ ... ...
1 ... ...



...



( _I_ 1 1)




|Col1|Col2|...|...|
|---|---|---|---|
|||~~...~~<br>|~~...~~<br>|
||......<br>|...|...|
|||...|...|



_R_
2


( _I_ 1 _R_ 1 ) ( _R I_ 1 2 _R_ 2 )



|( ) n G ...|Col2|
|---|---|
|...<br>|...|
|~~...~~|~~...~~|
|...<br>... ...<br>|...<br>... ...<br>|
|...|...|


_R_ _n_


( _R_ _n_ ~~1~~ _I_ _n_ _R_ _n_ ) ( _R_ _N_ ~~1~~ _I_ _N_ 1)



Figure 4.2: TT/MPS decomposition of an _N_ th-order data tensor, **X**, for
which the TT rank is t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u. (a) Tensorization of a hugescale vector, **x** P **R** _[I]_, into an _N_ th-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ . (b)
The data tensor can be represented exactly or approximately via a tensor
train (TT/MPS), consisting of 3rd-order cores in the form **X** – **G** [(] [1] [)] ˆ [1]

**G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [)] = xx **G** [(] [1] [)], **G** [(] [2] [)], . . ., **G** [(] _[N]_ [)] yy, where **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_
for _n_ = 1, 2, . . ., _N_ with _R_ 0 = _R_ _N_ = 1. (c) Equivalently, using the strong
Kronecker products, the TT tensor can be expressed in a vectorized form,
**x** – **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **G** [r] [(] _[N]_ [)] P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_, where the block matrices are
defined as **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, with blocks **g** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ][1] [.]


112


where the slice matrices are defined as


**G** _i_ [(] _n_ _[n]_ [)] = **G** [(] _[n]_ [)] ( :, _i_ _n_, : ) P **R** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_, _i_ _n_ = 1, 2, . . ., _I_ _n_


with **G** _i_ [(] _n_ _[n]_ [)] being the _i_ _n_ th lateral slice of the core **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_,
_n_ = 1, 2, . . ., _N_ and _R_ 0 = _R_ _N_ = 1.


3. The (global) tensor form, based on multilinear products (contraction)
of cores (see Figure 4.1(a)) given by


**X** – **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [´][1] [)] ˆ [1] **G** [(] _[N]_ [)]

= xx **G** [(] [1] [)], **G** [(] [2] [)], . . ., **G** [(] _[N]_ [´][1] [)], **G** [(] _[N]_ [)] yy, (4.3)


where the 3rd-order cores [1] **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, _n_ = 1, 2, . . ., _N_ and
_R_ 0 = _R_ _N_ = 1 (see also Figure 4.2(b)).


4. The tensor form, expressed as a sum of rank-1 tensors (see Figure
4.1(a))



**X** –



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ **g** [(] 1, [1] _r_ [)] 1 [˝] **[ g]** [(] _r_ [2] 1 [)], _r_ 2 [˝ ¨ ¨ ¨ ˝] **[ g]** [(] _r_ _[N]_ _N_ ´ [´] 2 [1], _r_ [)] _N_ ´1 [˝] **[ g]** [(] _r_ _[N]_ _N_ ´ [)] 1, 1 [,] (4.4)

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



where **g** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [=] **[ G]** [(] _[n]_ [)] [(] _[r]_ _n_ ´1 [, :,] _[ r]_ _n_ [)] [ P] **[ R]** _[I]_ _[n]_ [ are mode-2 fibers,] _[ n]_ [ =]
1, 2, . . ., _N_ and _R_ 0 = _R_ _N_ = 1.


5. A vector form, expressed by Kronecker products of the fibers



**x** –



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ **g** [(] 1, [1] _r_ [)] 1 [b] _[L]_ **[ g]** [(] _r_ [2] 1 [)], _r_ 2 [b] _L_

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1

¨ ¨ ¨ b _L_ **g** [(] _r_ _[N]_ _N_ ´ [´] 2 [1], _r_ [)] _N_ ´1 [b] _L_ **[g]** [(] _r_ _[N]_ _N_ ´ [)] 1, 1 [,] (4.5)



where **x** = vec ( **X** ) P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_ .


6. An alternative vector form, produced by strong Kronecker products
of block matrices (see Figure 4.1(b)) and Figure 4.2(c)), given by


**x** – **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **G** [r] [(] _[N]_ [)], (4.6)


1 Note that the cores **G** ( 1 ) and **G** ( _N_ ) are now two-dimensional arrays (matrices), but for
a uniform representation, we assume that these matrices are treated as 3rd-order cores of
sizes 1 ˆ _I_ 1 ˆ _R_ 1 and _R_ _N_ ´1 ˆ _I_ _N_ ˆ 1, respectively.


113


where the block matrices **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, for _n_ = 1, 2, . . ., _N_,
consist of blocks **g** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ][1] [,] _[ n]_ [ =] [ 1, 2, . . .,] _[ N]_ [, with] _[ R]_ 0 [=] _[ R]_ _N_ [=] [ 1,]
and the symbol |b| denotes the strong Kronecker product.


Analogous relationships can be established for Tensor Chain (i.e., MPS
with PBC (see Figure 2.19(b)) and summarized in Table 4.2.


**–**
**4.2** **Matrix** **TT** **Decomposition** **Matrix** **Product**
**Operator**


The matrix tensor train, also called the Matrix Product Operator (MPO)
with open boundary conditions (TT/MPO), is an important TN model
which first represents huge-scale structured matrices, **X** P **R** _[I]_ [ˆ] _[J]_, as 2 _N_ thorder tensors, **X** P **R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ] _[I]_ [2] [ˆ] _[J]_ [2] [ˆ¨¨¨] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_, where _I_ = _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _N_ and _J_ =
_J_ 1 _J_ 2 ¨ ¨ ¨ _J_ _N_ (see Figures 4.3, 4.4 and Table 4.3). Then, the matrix TT/MPO
converts such a 2 _N_ th-order tensor into a chain (train) of 4th-order cores [2] .
It should be noted that the matrix TT decomposition is equivalent to the
vector TT, created by merging all index pairs ( _i_ _n_, _j_ _n_ ) into a single index
ranging from 1 to _I_ _n_ _J_ _n_, in a reverse lexicographic order.
Similarly to the vector TT decomposition, a large scale 2 _N_ th-order
tensor, **X** P **R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ] _[I]_ [2] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_, can be represented in a TT/MPO format
via the following mathematical representations:


1. The scalar (entry-wise) form



_R_ _N_ ´1
ÿ _g_ [(] 1, [1] _i_ [)] 1, _j_ 1, _r_ 1 _[g]_ [(] _r_ [2] 1 [)], _i_ 2, _j_ 2, _r_ 2

_r_ _N_ ´1 = 1



_x_ _i_ 1, _j_ 1,..., _i_ _N_, _j_ _N_ –


2. The slice representation



_R_ 1
ÿ

_r_ 1 = 1



_R_ 2


¨ ¨ ¨

ÿ

_r_ 2 = 1



¨ ¨ ¨ _g_ [(] _r_ _[N]_ _N_ ´ [´] 2 [1], _i_ [)] _N_ ´1, _j_ _N_ ´1, _r_ _N_ ´1 _[g]_ [(] _r_ _[N]_ _N_ ´ [)] 1, _i_ _N_, _j_ _N_, 1 [.] (4.7)



_x_ _i_ 1, _j_ 1,..., _i_ _N_, _j_ _N_ – **G** _i_ [(] 1 [1], [)] _j_ 1 **[G]** _i_ [(] 2 [2], [)] _j_ 2 [¨ ¨ ¨] **[ G]** _i_ [(] _N_ _[N]_, _j_ [)] _N_ [,] (4.8)


where **G** [(] _[n]_ [)]
_i_ _n_, _j_ _n_ [=] **[ G]** [(] _[n]_ [)] [(] [:,] _[ i]_ _[n]_ [,] _[ j]_ _[n]_ [, :] [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ [ are slices of the cores]

**G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_, _n_ = 1, 2, . . ., _N_ and _R_ 0 = _R_ _N_ = 1.


2 The cores **G** ( 1 ) and **G** ( _N_ ) are in fact three-dimensional arrays, however for uniform
representation, we treat them as 4th-order cores of sizes 1 ˆ _I_ 1 ˆ _J_ 1 ˆ _R_ 1 and _R_ _N_ ´1 ˆ _I_ _N_ ˆ
_J_ _N_ ˆ 1.


114


3. The compact tensor form based on multilinear products (Figure
4.4(b))


**X** – **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [)]

= xx **G** [(] [1] [)], **G** [(] [2] [)], . . ., **G** [(] _[N]_ [)] yy, (4.9)


where the TT-cores are defined as **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_, _n_ =
1, 2, . . ., _N_ and _R_ 0 = _R_ _N_ = 1.


4. A matrix form, based on strong Kronecker products of block matrices
(Figures 4.3(b) and 4.4(c))


**X** – **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **G** [r] [(] _[N]_ [)] P **R** _[I]_ [1] [¨¨¨] _[I]_ _[N]_ [ ˆ] _[ J]_ [1] [¨¨¨] _[J]_ _[N]_, (4.10)


where **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_ are block matrices with blocks **G** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P]
**R** _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ and the number of blocks is _R_ _n_ ´1 ˆ _R_ _n_ . In a special case, when
the TT ranks _R_ _n_ = 1, @ _n_, the strong Kronecker products simplify into
standard (left) Kronecker products.


The strong Kronecker product representation of a TT is probably the
most comprehensive and useful form for displaying tensor trains in their
vector/matrix form, since it allows us to perform many operations using
relatively small block matrices.
**Example.** For two matrices (in the TT format) expressed via the strong
Kronecker products, **A** = **A** [˜] [(] [1] [)] |b| **A** [˜] [(] [2] [)] |b| ¨ ¨ ¨ |b| **A** [˜] [(] _[N]_ [)] and **B** = **B** [˜] [(] [1] [)] |b
| **B** [˜] [(] [2] [)] |b| ¨ ¨ ¨ |b| **B** [˜] [(] _[N]_ [)], their Kronecker product can be efficiently computed
as **A** b _L_ **B** = **A** [˜] [(] [1] [)] |b| ¨ ¨ ¨ |b| **A** [˜] [(] _[N]_ [)] |b| **B** [˜] [(] [1] [)] |b| ¨ ¨ ¨ |b| **B** [˜] [(] _[N]_ [)] . Furthermore, if the
matrices **A** and **B** have the same mode sizes [3], then their linear combination,
**C** = _α_ **A** + _β_ **B** can be compactly expressed as [112,113,158]



**A** ˜ ( 2 ) **0**
**C** = [ **A** [˜] [(] [1] [)] [ ˜] **B** [(] [1] [)] ] |b| ˜
� **0** **B** [(] [2] [)]



**A** ˜ ( _N_ ´1 ) 0

˜
|b| ¨ ¨ ¨ |b|
� � 0 **B** [(] _[N]_ [´][1] [)]



_α_ ˜ **A** ( _N_ )
|b|
� � _β_ **B** [˜] [(] _[N]_ [)]



.
�



Consider its reshaped tensor **C** = xx **C** [(] [1] [)], **C** [(] [2] [)], . . ., **C** [(] _[N]_ [)] yy in the TT format;
then its cores **C** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_, _n_ = 1, 2, . . ., _N_ can be expressed
through their unfolding matrices, **C** ă [(] _[n]_ _n_ [)] ą [P] **[ R]** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_ [, or equivalently by]


3 Note that, wile original matrices **A** P **R** _I_ 1 ¨¨¨ _I_ _N_ ˆ _J_ 1 ¨¨¨ _J_ _N_ and **B** P **R** _I_ 1 ¨¨¨ _I_ _N_ ˆ _J_ 1 ¨¨¨ _J_ _N_ must have

the same mode sizes, the corresponding core tenors, **A** [(] _[n]_ [)] = P **R** _[R]_ _n_ _[A]_ ´1 [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _n_ _[A]_ and **B** [(] _[n]_ [)] = P
**R** _[R]_ _n_ _[B]_ ´1 [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _n_ _[B]_, may have arbitrary mode sizes.


115


(a)


(b)










|G(2) R 2 J 3|(3) J<br>G R 4<br>3|
|---|---|
|_I_3|_I_4|



_R_
3



_J_
3


_I_
3



_R_
2


_R_
2



_J_
2


_J_
2



_J_ 3
3

_J_
4 _[I]_ [4]



_R_
3



_I_
2



_I_
3



_R_


_I_
1



_J_ 1



1



_R_
_I_ 2 1


_J_ 2 _R_ 2


_I_
2



_R_
3



1 1



_J_
3


_I_
3


_J_
3


_I_
3



_R_
3



_R_
2



_R_
3



(1 × _I_ 1 × _J_ 1 × _R_ 1 ) ( _R_ 1 × _I_ 2 × _J_ 2 × _R_ 2 ) ( _R_ 2 × _I_ 3 × _J_ 3 × _R_ 3 ) ( _R_ 3 × _I_ 4 × _J_ 4 × 1)



(4)
**G**





(1)
**G**



(2)
**G**



( _I_ 4 × _J_ 4 )

( _I_ 1 × _J_ 1 )
( _I_ 2 × _J_ 2 ) ( _I_ 3 × _J_ 3 )

|Col1|Col2|Col3|
|---|---|---|
||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||


|Col1|Col2|(3 G|3)|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||


|)|Col2|
|---|---|
|)||
|)||
|)||
|)||



( _I_ 1 × _R J_ 1 1 ) ( _R_ 1 2 _I_ × _R J_ 2 2 ) ( _R_ 2 3 _I_ × _R J_ 3 3 ) ( _R_ 3 4 _I_ × _J_ 4 )


Figure 4.3: TT/MPO decomposition of a matrix, **X** P **R** _[I]_ [ˆ] _[J]_, reshaped as an
8th-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ [4] [ˆ] _[J]_ [4], where _I_ = _I_ 1 _I_ 2 _I_ 3 _I_ 4 and _J_ = _J_ 1 _J_ 2 _J_ 3 _J_ 4 .
(a) Basic TT representation via multilinear products (tensor contractions)
of cores **X** = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] **G** [(] [3] [)] ˆ [1] **G** [(] [4] [)], with **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ for
_R_ 1 = 3, _R_ 2 = 4, _R_ 3 = 5, _R_ 0 = _R_ 4 = 1. (b) Representation of a matrix or a
matricized tensor via strong Kronecker products of block matrices, in the
form **X** = **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| **G** [r] [(] [3] [)] |b| **G** [r] [(] [4] [)] P **R** _[I]_ [1] _[I]_ [2] _[I]_ [3] _[I]_ [4] [ ˆ] _[ J]_ [1] _[J]_ [2] _[J]_ [3] _[J]_ [4] .


116


(a)


(b)









(1) (2) ( ) _n_

**G** **G** **G**





(1)



**G**



(2)







**G**



( ) _N_

**G**





_J_ _N_ _I_ _N_



_J_ _n_

_I_ _n_



_R_ _n_



_I_ 1



_J_ 2


_I_ 2



_R_ 2



_I_ 2





1 1 1 1 1



_J_ _n_

_I_ _n_



_R_ _n_ -1 1 1 _R_ _N_ -1



_R_
1


|R1 .|J1 .|Col3|
|---|---|---|
|...|||
||||
||||



...



...



(1 _I_ 1 _J_ 1 _R_ 1 ) ( _R_ 1 _I_ 2 _J_ 2 _R_ 2 ) ( _R_ _n_ ~~1~~ _I_ _n_ _J_ _n_ _R_ _n_ ) ( _R_ _N_ ~~1~~ _I_ _N_ _J_ _N_ 1)


(c)



( _I_ _N_ _J_ _N_ )


...





(1)
**G**


_R_ 1





...





_R_ _N_ -1



( _I_ 1 _J_ 1 ) ... ( _I_ 2 _J_ 2 ) ... ( _I_ _n_ _J_ _n_ )




|)|( ) N G|
|---|---|
|||
|||
||...|
|||




|Col1|Col2|(2) G ...|Col4|Col5|
|---|---|---|---|---|
|||~~...~~<br>|||
|||~~...~~<br>|||
|...|...|...|...|...|
|||...|||


|Col1|Col2|( ) n G|Col4|Col5|
|---|---|---|---|---|
|||...<br>...|||
||||||
|...|...|...|...|...|
|||...|||



( _I_ 1 _R J_ 1 1 ) ( _R I_ 1 2 _R J_ 2 2 )



( _R_ _n_ ~~1~~ _I_ _n_ _R J_ _n n_ ) ( _R_ _N_ ~~1~~ _I_ _N_ _J_ _N_ )



Figure 4.4: Representations of huge matrices by “linked” block matrices.
(a) Tensorization of a huge-scale matrix, **X** P **R** _[I]_ [ˆ] _[J]_, into a 2 _N_ th-order
tensor **X** P **R** _[I]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_ . (b) The TT/MPO decomposition of a huge
matrix, **X**, expressed by 4th-order cores, **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_ . (c)
Alternative graphical representation of a matrix, **X** P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_ [ ˆ] _[ J]_ [1] _[J]_ [2] [¨¨¨] _[J]_ _[N]_,
via strong Kronecker products of block matrices **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[ I]_ _[n]_ [ ˆ] _[ R]_ _[n]_ _[ J]_ _[n]_ for
_n_ = 1, 2, . . ., _N_ with _R_ 0 = _R_ _N_ = 1.


117


Table 4.3: Equivalent forms of the matrix Tensor Train decomposition
(MPO with open boundary conditions) for a 2 _N_ th-order tensor **X** P
**R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ] _[I]_ [2] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_ . It is assumed that the TT rank is t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u,
with _R_ 0 = _R_ _N_ = 1.


Tensor representation: Multilinear products (tensor contractions)


**X** = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **G** [(] _[N]_ [´][1] [)] ˆ [1] **G** [(] _[N]_ [)]


with 4th-order cores **G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_, ( _n_ = 1, 2, . . ., _N_ )


Tensor representation: Outer products



**X** =



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ **G** [(] 1, [1] [)] _r_ 1 [˝] **[ G]** [(] _r_ [2] 1 [)], _r_ 2 [˝ ¨ ¨ ¨ ˝] **[ G]** [(] _r_ _[N]_ _N_ [´] ´2 [1], [)] _r_ _N_ ´1 [˝] **[ G]** [(] _r_ _[N]_ _N_ [)] ´1, 1

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



where **G** [(] _r_ _[n]_ _n_ [)] ´1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ are blocks of][ r] **[G]** [(] _[n]_ [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_


Matrix representation: Strong Kronecker products


**X** = **G** [r] [(] [1] [)] |b| **G** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **G** [r] [(] _[N]_ [)] P **R** _[I]_ [1] [¨¨¨] _[I]_ _[N]_ [ ˆ] _[ J]_ [1] [¨¨¨] _[J]_ _[N]_


where **G** [r] [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_ are block matrices with blocks


**G** [(] _[n]_ [)] ( _r_ _n_ ´1, :, :, _r_ _n_ )


Scalar representation



_x_ _i_ 1, _j_ 1, _i_ 2, _j_ 2,..., _i_ _N_, _j_ _N_ =



_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ _g_ [(] 1, [1] [)] _i_ 1, _j_ 1, _r_ 1 _[g]_ [(] _r_ [2] 1 [)], _i_ 2, _j_ 2, _r_ 2 [¨ ¨ ¨] _[ g]_ [(] _r_ _[N]_ _N_ [)] ´1, _i_ _N_, _j_ _N_,1

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1



where _g_ [(] _r_ _[n]_ _n_ [)] ´1, _i_ _n_, _j_ _n_, _r_ _n_ [are entries of a 4th-order core]

**G** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_


Slice (MPS) representation


_x_ _i_ 1, _j_ 1, _i_ 2, _j_ 2,..., _i_ _N_, _j_ _N_ = **G** _i_ [(] 1 [1], [)] _j_ 1 **[G]** _i_ [(] 2 [2], [)] _j_ 2 [¨ ¨ ¨] **[ G]** _i_ [(] _N_ _[N]_, _j_ [)] _N_ where


**G** [(] _[n]_ [)]
_i_ _n_, _j_ _n_ [=] **[ G]** [(] _[n]_ [)] [(] [:,] _[ i]_ _[n]_ [,] _[ j]_ _[n]_ [, :] [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ [ are slices of] **[ G]** [(] _[n]_ [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_


118


the lateral slices, **C** _i_ [(] _n_ _[n]_, [)] _j_ _n_ [P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[R]_ _[n]_ [, as follows]



**C** [(] _[n]_ [)] =
_i_ _n_, _j_ _n_



**A** [(] _[n]_ [)] **0**
_i_ _n_, _j_ _n_

**0** **B** [(] _[n]_ [)]

� _i_ _n_, _j_ _n_



�



, _n_ = 2, 3, . . ., _N_ ´ 1, (4.11)



while for the border cores


**C** [(] [1] [)] **A** [(] [1] [)]
_i_ 1, _j_ 1 [=] � _i_ 1, _j_ 1 **[B]** _i_ [(] 1 [1], [)] _j_ 1



�, **C** _i_ [(] _N_ _[N]_, _j_ [)] _N_ =



_α_ **A** [(] _[N]_ [)]
_i_ _N_, _j_ _N_
� _β_ **B** _i_ [(] _N_ _[N]_, _j_ [)] _N_



�



(4.12)



for _i_ _n_ = 1, 2, . . ., _I_ _n_, _j_ _n_ = 1, 2, . . ., _J_ _N_, _n_ = 1, 2, . . ., _N_ .
Note that the various mathematical and graphical representations of
TT/MPS and TT/MPO can be used interchangeably for different purposes
or applications. With these representations, all basic mathematical
operations in TT format can be performed on the constitutive block
matrices, even without the need to explicitly construct core tensors [67,158].


**Remark.** In the TT/MPO paradigm, compression of large matrices is
not performed by global (standard) low-rank matrix approximations, but
by low-rank approximations of block-matrices (submatrices) arranged in
a hierarchical (linked) fashion. However, to achieve a low-rank TT and
consequently a good compression ratio, ranks of all the corresponding
unfolding matrices of a specific structured data tensor must be low, i.e.,
their singular values must rapidly decrease to zero. While this is true for
many structured matrices, unfortunately in general, this assumption does
not hold.


**4.3** **Links Between CP, BTD Formats and TT/TC**

**Formats**


It is important to note that any specific TN format can be converted into the
TT format. This very useful property is next illustrated for two simple but
important cases which establish links between the CP and TT and the BTD
and TT formats.


1. A tensor in the CP format, given by



**X** =



_R_
ÿ **a** _r_ [(] [1] [)] ˝ **a** _r_ [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **a** _r_ [(] _[N]_ [)], (4.13)

_r_ = 1


119


(a)


(b)












|R R G(2) G(1)|Col2|
|---|---|
|||
|_I_1|_I_2|























1


|R|(2)<br>A|Col3|
|---|---|---|
|_R_<br><br>**A**<br>(2)|**A**<br>(2)|**A**<br>(2)|
|_R_<br><br>**A**<br>(2)|**A**<br>(2)||


|R|N-1)|Col3|
|---|---|---|
|**A**(_N_-1<br>_R_|_N_-1|_N_-1|
|**A**(_N_-1<br>_R_|_N_-1||



_I_ 1 _I_ 2 _I_ _N_ -1 _I_ _N_
(1× _I_ × 1 _R_ ) ( _R_ × _I_ 2 × _R_ ) ( _R_ × _I_ _N_ × -1 _R_ ) ( _R_ × _I_ _N_ ×1)

















_R_


_I_ 1


|R G (1)|G (2)|R G  G ( ) N R R (N-1)|
|---|---|---|
|_R_<br>_J_1<br><br>~~_I_~~2||_R_<br><br><br>_I_<br>_J_2<br>_IN_<br>_R_<br>_J_<br>_R_<br>_N-_1<br>_N-_1|
|_R_<br>_J_1<br><br>~~_I_~~2|||



1 _R_ 1 1

_J_ 2





_J_ _N-_ 1







_I_ _N-_ 1



_J_ _N-_ 1



_J_ _N_ _I_ _N_



_I_ 2



_R_



1



_R_



_R_


...



_R_


|..|J1 ..|
|---|---|
|.||
|||
|||
|||



_I_ 2



...



_I_ _N-_ 1



(1× _I_ 1 × _J_ 1 × _R_ ) ( _R_ × _I_ 2 × _J_ 2 × _R_ ) ( _R_ × _I_ _N-_ 1 × _J_ _N-_ 1 × _R_ ) ( _R_ × _I_ _N_ × _J_ _N_ ×1)



|Col1|Col2|R|Col4|
|---|---|---|---|
|||...||
|||||


( _I_ × 1 _J_ 1 )







( _I_ _N_ × _J_ _N_ )



...


( _I_ × 2 _J_ 2 ) ... ( _I_ _N-_ × 1 _J_ _N-_ 1 )



_R_


|Col1|Col2|
|---|---|
|...|...|
|||


|Col1|Col2|...|Col4|Col5|
|---|---|---|---|---|
|||~~...~~<br>|||
|...|...|...|...|...|
|||...|||


|Col1|Col2|...<br>...|Col4|
|---|---|---|---|
|||||
|...|...|...|...|
|||...||



( _I_ × 1 _RJ_ 1 ) ( _RI_ × 2 _RJ_ 2 ) ( _RI_ _N-_ × 1 _RJ_ _N-_ 1 ) ( _RI_ _N_ × _J_ _N_ )


Figure 4.5: Links between the TT format and other tensor network formats.
(a) Representation of the CP decomposition for an _N_ th-order tensor, **X** =
**I** ˆ 1 **A** [(] [1] [)] ˆ 2 **A** [(] [2] [)] ¨ ¨ ¨ ˆ _N_ **A** [(] _[N]_ [)], in the TT format. (b) Representation of the
BTD model given by Eqs. (4.15) and (4.16) in the TT/MPO format. Observe
that the TT-cores are very sparse and the TT ranks are t _R_, _R_, . . ., _R_ u. Similar
relationships can be established straightforwardly for the TC format.


120


can be straightforwardly converted into the TT/MPS format as
follows. Since each of the _R_ rank-1 tensors can be represented in the
TT format of TT rank ( 1, 1, . . ., 1 ), using formulas (4.11) and (4.12), we
have



**X** =



_R_
ÿ

_r_ = 1



xx **a** _r_ [(] [1] [)] [T], **a** _r_ [(] [2] [)] [T], . . ., **a** _r_ [(] _[N]_ [)] [T] yy (4.14)



= xx **G** [(] [1] [)], **G** [(] [2] [)], . . ., **G** [(] _[N]_ [´][1] [)], **G** [(] _[N]_ [)] yy,


where the TT-cores **G** [(] _[n]_ [)] P **R** _[R]_ [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ have diagonal lateral slices **G** [(] _[n]_ [)] ( :
, _i_ _n_, : ) = **G** _i_ [(] _n_ _[n]_ [)] = diag ( _a_ _i_ _n_,1, _a_ _i_ _n_,2, . . ., _a_ _i_ _n_, _R_ ) P **R** _[R]_ [ˆ] _[R]_ for _n_ = 2, 3, . . ., _N_ ´
1 and **G** [(] [1] [)] = **A** [(] [1] [)] P **R** _[I]_ [1] [ˆ] _[R]_ and **G** [(] _[N]_ [)] = **A** [(] _[N]_ [)] [ T] P **R** _[R]_ [ˆ] _[I]_ _[N]_ (see Figure
4.5(a)).


2. A more general Block Term Decomposition (BTD) for a 2 _N_ th-order
data tensor



**X** =



_R_
ÿ ( **A** _r_ [(] [1] [)] ˝ **A** _r_ [(] [2] [)] ˝ ¨ ¨ ¨ ˝ **A** _r_ [(] _[N]_ [)] ) P **R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_ (4.15)

_r_ = 1



with full rank matrices, **A** _r_ [(] _[n]_ [)] P **R** _[I]_ _[n]_ [ˆ] _[J]_ _[n]_, @ _r_, can be converted into a
matrix TT/MPO format, as illustrated in Figure 4.5(b).


Note that (4.15) can be expressed in a matricized (unfolding) form via
strong Kronecker products of block diagonal matrices (see formulas
(4.11)), given by



**X** =



_R_
ÿ ( **A** _r_ [(] [1] [)] b _L_ **A** _r_ [(] [2] [)] b _L_ ¨ ¨ ¨ b _L_ **A** _r_ [(] _[N]_ [)] ) (4.16)

_r_ = 1



= **G** r [(] [1] [)] |b| r **G** [(] [2] [)] |b| ¨ ¨ ¨ |b| r **G** [(] _[N]_ [)] P **R** _[I]_ 1 [¨¨¨] _[I]_ _N_ [ˆ] _[ J]_ 1 [¨¨¨ˆ] _[J]_ _N_,


with the TT rank, _R_ _n_ = _R_ for _n_ = 1, 2, . . . _N_ ´ 1, and the block
diagonal matrices, **G** [r] [(] _[n]_ [)] = diag ( **A** 1 [(] _[n]_ [)] [,] **[ A]** 2 [(] _[n]_ [)] [, . . .,] **[ A]** [(] _R_ _[n]_ [)] [)] [ P] **[ R]** _[RI]_ _[n]_ [ˆ] _[RJ]_ _[n]_ [, for]

_n_ = 2, 3, . . ., _N_ ´ 1, while **G** [r] [(] [1] [)] = [ **A** 1 [(] [1] [)] [,] **[ A]** 2 [(] [1] [)] [, . . .,] **[ A]** [(] _R_ [1] [)] []] [ P] **[ R]** _[I]_ [1] [ˆ] _[RJ]_ [1] [ is a]





 P **R** _RI_ _N_ ˆ _J_ _N_ a column block



row block matrix, and **G** [r] [(] _[N]_ [)] =


matrix (see Figure 4.5(b)).









**A** [(] _[N]_ [)]
1

...
**A** [(] _[N]_ [)]
_R_



121


6
_I_ = 2





TT

(1) (2) (3) (4) (5) (6)
**G** **G** **G** **G** **G** **G**


|Col1|2|Col3|
|---|---|---|
||||
|2|||
|2|||



(2 2 2 2 2 2)


(64 1)

|. . .|I|
|---|---|
|. . .|(|



Figure 4.6: Concept of tensorization/quantization of a large-scale vector
into a higher-order quantized tensor. In order to achieve a good
compression ratio, we need to apply a suitable tensor decomposition such
as the quantized TT (QTT) using 3rd-order cores, **X** = **G** [(] [1] [)] ˆ [1] **G** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1]

**G** [(] [6] [)] .


Several algorithms exist for decompositions in the form (4.15) and (4.16)

[14, 15, 181]. In this way, TT/MPO decompositions for huge-scale
structured matrices can be constructed indirectly.


**4.4** **Quantized Tensor Train (QTT) – Blessing of**
**Dimensionality**


The procedure of creating a higher-order tensor from lower-order original
data is referred to as tensorization, while in a special case where each mode
has a very small size 2, 3 or 4, it is referred to as quantization. In addition to
vectors and matrices, lower-order tensors can also be reshaped into higherorder tensors. By virtue of quantization, low-rank TN approximations with
high compression ratios can be obtained, which is not possible to achieve
with original raw data formats. [114,157].
Therefore, _the quantization can be considered as a special form of tensorization_
_where size of each mode is very small, typically 2 or 3_ . The concept of quantized
tensor networks (QTN) was first proposed in [157] and [114], whereby lowsize 3rd-order cores are sparsely interconnected via tensor contractions.
The so obtained model often provides an efficient, highly compressed, and
low-rank representation of a data tensor and helps to mitigate the curse of


122


dimensionality, as illustrated below.


**Example.** The quantization of a huge vector, **x** P **R** _[I]_, _I_ = 2 _[K]_, can be
achieved through reshaping to give a ( 2 ˆ 2 ˆ ¨ ¨ ¨ ˆ 2 ) tensor **X** of order _K_,
as illustrated in Figure 4.6. For structured data such a quantized tensor,
**X**, often admits low-rank TN approximation, so that a good compression
of a huge vector **x** can be achieved by enforcing the maximum possible
low-rank structure on the tensor network. Even more generally, an _N_ thorder tensor, **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with _I_ _n_ = _q_ _[K]_ _[n]_, can be quantized in all modes
simultaneously to yield a ( _q_ ˆ _q_ ˆ ¨ ¨ ¨ ˆ _q_ ) quantized tensor of higher-order
and with small value of _q_ .


**Example.** Since large-scale tensors (even of low-order) cannot be loaded
directly into the computer memory, our approach to this problem is
to represent the huge-scale data by tensor networks in a distributed
and compressed TT format, so as to avoid the explicit requirement for
unfeasible large computer memory.
In the example shown in Figure 4.7, the tensor train of a huge
3rd-order tensor is expressed by the strong Kronecker products of
block tensors with relatively small 3rd-order tensor blocks. The
QTT is mathematically represented in a distributed form via strong
Kronecker products of block 5th-order tensors. Recall that the strong



Kronecker product of two block core tensors, **G** [r]



( _n_ )
P **R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_ [ˆ] _[K]_ _[n]_



r ( _n_ + 1 )
and **G** P **R** _[R]_ _[n]_ _[I]_ _[n]_ [+] [1] [ˆ] _[R]_ _[n]_ [+] [1] _[J]_ _[n]_ [+] [1] [ˆ] _[K]_ _[n]_ [+] [1], is defined as the block tensor,



**C** = **G** [r] ( _n_ ) |b| **G** r ( _n_ + 1 ) P **R** _R_ _n_ ´1 _I_ _n_ _I_ _n_ + 1 ˆ _R_ _n_ + 1 _J_ _n_ _J_ _n_ + 1 ˆ _K_ _n_ _K_ _n_ + 1, with 3rd-order tensor

blocks, **C** _r_ _n_ ´1, _r_ _n_ + 1 = [ř] _r_ _[R]_ _n_ _[n]_ = 1 **[G]** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [b] _L_ **[G]** _r_ [(] _n_ _[n]_, [+] _r_ _n_ [1] + [)] 1 [P]



( _n_ ) r
|b| **G**



**C** = **G** [r]



**R** _[I]_ _[n]_ _[I]_ _[n]_ [+] [1] [ˆ] _[J]_ _[n]_ _[J]_ _[n]_ [+] [1] [ˆ] _[K]_ _[n]_ _[K]_ _[n]_ [+] [1], where **G** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ P **R** _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[K]_ _[n]_ and **G** _r_ [(] _n_ _[n]_, [+] _r_ _n_ [1] + [)] 1 P



( _n_ + 1 )
, respectively.



**R** _[I]_ _[n]_ [+] [1] [ˆ] _[J]_ _[n]_ [+] [1] [ˆ] _[K]_ _[n]_ [+] [1] are the block tensors of **G** [r]



( _n_ ) r
and **G**



In practice, a fine ( _q_ = 2, 3, 4 ) quantization is desirable to create
as many virtual (additional) modes as possible, thus allowing for the
implementation of efficient low-rank tensor approximations. For example,
the binary encoding ( _q_ = 2) reshapes an _N_ th-order tensor with ( 2 _[K]_ [1] ˆ 2 _[K]_ [2] ˆ
¨ ¨ ¨ ˆ 2 _[K]_ _[N]_ ) elements into a tensor of order ( _K_ 1 + _K_ 2 + ¨ ¨ ¨ + _K_ _N_ ), with the
same number of elements. In other words, the idea is to quantize each of
the _n_ “physical” modes (dimensions) by replacing them with _K_ _n_ “virtual”
modes, provided that the corresponding mode sizes, _I_ _n_, are factorized as
_I_ _n_ = _I_ _n_,1 _I_ _n_,2 ¨ ¨ ¨ _I_ _n_, _K_ _n_ . This, in turn, corresponds to reshaping the _n_ th mode


123


Table 4.4: Storage complexities of tensor decomposition models for
an _N_ th-order tensor, **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, for which the original storage
complexity is _O_ ( _I_ _[N]_ ), where _I_ = maxt _I_ 1, _I_ 2, . . ., _I_ _N_ u, while _R_ is the upper
bound on the ranks of tensor decompositions considered, that is, _R_ =
maxt _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u or _R_ = maxt _R_ 1, _R_ 2, . . ., _R_ _N_ u.


1. Full (raw) tensor format _O_ ( _I_ _[N]_ )


2. CP _O_ ( _NIR_ )


3. Tucker _O_ ( _NIR_ + _R_ _[N]_ )


4. TT/MPS _O_ ( _NIR_ [2] )


5. TT/MPO _O_ ( _NI_ [2] _R_ [2] )


6. Quantized TT/MPS (QTT) _O_ ( _NR_ [2] log _q_ ( _I_ ))


7. QTT+Tucker _O_ ( _NR_ [2] log _q_ ( _I_ ) + _NR_ [3] )


8. Hierarchical Tucker (HT) _O_ ( _NIR_ + _NR_ [3] )


of size _I_ _n_ into _K_ _n_ modes of sizes _I_ _n_,1, _I_ _n_,2, . . ., _I_ _n_, _K_ _n_ .
The TT decomposition applied to quantized tensors is referred to as
the QTT, Quantics-TT or Quantized-TT, and was first introduced as a
compression scheme for large-scale matrices [157], and also independently
for more general settings.
The attractive properties of QTT are:


1. Not only QTT ranks are typically small (usually, below 20) but
they are also almost independent [4] of the data size (even for _I_ =
2 [50] ), thus providing a logarithmic (sub-linear) reduction of storage
requirements from _O_ ( _I_ _[N]_ ) to _O_ ( _NR_ [2] log _q_ ( _I_ )) which is referred to as
super-compression [68, 70, 111, 112, 114]. Comparisons of the storage
complexity of various tensor formats are given in Table 4.4.


2. Compared to the TT decomposition (without quantization), _the QTT_
_format often represents deep structures in the data by introducing “virtual”_
_dimensions or modes_ . For data which exhibit high degrees of structure,


4 At least uniformly bounded.


124


(a)


(b)



























































Figure 4.7: Tensorization/quantization of a huge-scale 3rd-order tensor
into a higher order tensor and its TT representation. (a) Example of
tensorization/quantization of a 3rd-order tensor, **X** P **R** _[I]_ [ˆ] _[J]_ [ˆ] _[K]_, into a 3 _N_ thorder tensor, assuming that the mode sizes can be factorized as, _I_ =
_I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _N_, _J_ = _J_ 1 _J_ 2 ¨ ¨ ¨ _J_ _N_ and _K_ = _K_ 1 _K_ 2 ¨ ¨ ¨ _K_ _N_ . (b) Decomposition of the
high-order tensor via a generalized Tensor Train and its representation



r ( 1 )
by the strong Kronecker product of block tensors as **X** – **G** |b



| **G** r ( 2 ) |b| ¨ ¨ ¨ |b| **G** r ( _N_ ) P **R** _I_ 1 ¨¨¨ _I_ _N_ ˆ _J_ 1 ¨¨¨ _J_ _N_ ˆ _K_ 1 ¨¨¨ _K_ _N_, where each block **G** r ( _n_ ) P

**R** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[J]_ _[n]_ [ˆ] _[K]_ _[n]_ is also a 3rd-order tensor of size ( _I_ _n_ ˆ _J_ _n_ ˆ _K_ _n_ ), for _n_ =
1, 2, . . ., _N_ with _R_ 0 = _R_ _N_ = 1. In the special case when _J_ = _K_ = 1, the
model simplifies into the standard TT/MPS model.



( _N_ ) P **R** _I_ 1 ¨¨¨ _I_ _N_ ˆ _J_ 1 ¨¨¨ _J_ _N_ ˆ _K_ 1 ¨¨¨ _K_ _N_, where each block **G** r



r
| **G**



( 2 ) r
|b| ¨ ¨ ¨ |b| **G**



125


(a)


(b)







_R_
_N_





























Figure 4.8: The QTT-Tucker or alternatively QTC-Tucker (Quantized
Tensor-Chain-Tucker) format. (a) Distributed representation of a matrix
**A** _n_ P **R** _[I]_ _[n]_ [ˆ][ ˆ] _[R]_ _[n]_ with a very large value of _I_ _n_ via QTT, by tensorization
to a high-order quantized tensor, followed by QTT decomposition. (b)
Distributed representation of a large-scale Tucker- _N_ model, **X** – **G** ˆ 1
**A** 1 ˆ **A** 2 ¨ ¨ ¨ ˆ _N_ **A** _N_, via a quantized TC model in which the core tensor
**G** P **R** _[R]_ [ˆ] [1] [ˆ][ ˆ] _[R]_ [2] [ˆ¨¨¨ˆ][ ˆ] _[R]_ _[N]_ and optionally all large-scale factor matrices **A** _n_ ( _n_ =
1, 2, . . ., _N_ ) are represented by MPS models (for more detail see [68]).


126


the high compressibility of the QTT approximation is a consequence
of the better separability properties of the quantized tensor.


3. The fact that the QTT ranks are often moderate or even low [5] offers
unique advantages in the context of big data analytics (see [112,
114, 115] and references therein), together with high efficiency of
multilinear algebra within the TT/QTT algorithms which rests upon
the well-posedness of the low-rank TT approximations.


The ranks of the QTT format often grow dramatically with data size, but
with a linear increase in the approximation accuracy. To overcome this
problem, Dolgov and Khoromskij proposed the QTT-Tucker format [68]
(see Figure 4.8), which exploits the TT approximation not only for the
Tucker core tensor, but also for the factor matrices. This model naturally
admits distributed computation, and often yields bounded ranks, thus
avoiding the curse of dimensionality.
The TT/QTT tensor networks have already found application in very
large-scale problems in scientific computing, such as in eigenanalysis,
super-fast Fourier transforms, and in solving huge systems of large linear
equations (see [68,70,102,120,123,218] and references therein).


**4.5** **Basic Operations in TT Formats**


For big tensors in their TT formats, basic mathematical operations, such
as the addition, inner product, computation of tensor norms, Hadamard
and Kronecker product, and matrix-by-vector and matrix-by-matrix
multiplications can be very efficiently performed using block (slice)
matrices of individual (relatively small size) core tensors.


Consider two _N_ th-order tensors in the TT format


**X** = xx **X** [(] [1] [)], **X** [(] [2] [)], . . ., **X** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_

**Y** = xx **Y** [(] [1] [)], **Y** [(] [2] [)], . . ., **Y** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_,


for which the TT ranks are **r** _X_ = t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u and **r** _Y_ =
t _R_ [˜] 1, _R_ [˜] 2, . . ., _R_ [˜] _N_ ´1 u. The following operations can then be performed
directly in the TT formats.


5 The TT/QTT ranks are constant or growing linearly with respect to the tensor order _N_
and are constant or growing logarithmically with respect to the dimension of tensor modes
_I_ .


127


**Tensor addition.** The sum of two tensors


**Z** = **X** + **Y** = xx **Z** [(] [1] [)], **Z** [(] [2] [)], . . ., **Z** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ (4.17)


has the TT rank **r** _Z_ = **r** _X_ + **r** _Y_ and can be expressed via lateral slices of the
cores **Z** P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ as



**Z** [(] _[n]_ [)] =
_i_ _n_



**X** [(] _[n]_ [)] **0**
_i_ _n_

**0** **Y** [(] _[n]_ [)]

� _i_ _n_



�



, _n_ = 2, 3, . . ., _N_ ´ 1. (4.18)



For the border cores, we have


**Z** [(] [1] [)] = **X** [(] [1] [)] **Y** [(] [1] [)]
_i_ 1 � _i_ 1 _i_ 1



�, **Z** _i_ [(] _N_ _[N]_ [)] =



**X** [(] _[N]_ [)]
_i_ _N_
**Y** [(] _[N]_ [)]
� _i_ _N_



�



(4.19)



for _i_ _n_ = 1, 2, . . ., _I_ _n_, _n_ = 1, 2, . . ., _N_ .


**Hadamard product.** The computation of the Hadamard (element-wise)
product, **Z** = **X** f **Y**, of two tensors, **X** and **Y**, of the same order and the
same size can be performed very efficiently in the TT format by expressing
the slices of the cores, **Z** P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_, as


**Z** _i_ [(] _n_ _[n]_ [)] = **X** _i_ [(] _n_ _[n]_ [)] [b] **[ Y]** _i_ [(] _n_ _[n]_ [)] [,] _n_ = 1, . . ., _N_, _i_ _n_ = 1, . . ., _I_ _n_ . (4.20)


This increases the TT ranks for the tensor **Z** to at most _R_ _n_ _R_ [˜] _n_, _n_ = 1, 2, . . ., _N_,
but the associated computational complexity can be reduced from being
exponential in _N_, _O_ ( _I_ _[N]_ ), to being linear in both _I_ and _N_, _O_ ( _IN_ ( _RR_ [˜] ) [2] )) .


**Super fast Fourier transform** of a tensor in the TT format (MATLAB
functions: fftn ( **X** ) and fft ( **X** [(] _[n]_ [)], [], 2 ) ) can be computed as


_F_ ( **X** ) = xx _F_ ( **X** [(] [1] [)] ), _F_ ( **X** [(] [2] [)] ), . . ., _F_ ( **X** [(] _[N]_ [)] ) yy

= _F_ ( **X** [(] [1] [)] ) ˆ [1] _F_ ( **X** [(] [2] [)] ) ˆ [1] ¨ ¨ ¨ ˆ [1] _F_ ( **X** [(] _[N]_ [)] ) . (4.21)


It should be emphasized that performing computation of the FFT on
relatively small core tensors **X** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _reduces dramatically_
_computational complexity under condition that a data tensor admits low-rank_
_TT approximation_ . This approach is referred to as the super fast Fourier
transform (SFFT) in TT format. Wavelets, DCT, and other linear integral


128


transformations admit a similar form to the SFFT in (4.21), for example, for
the wavelet transform in the TT format, we have


_W_ ( **X** ) = xx _W_ ( **X** [(] [1] [)] ), _W_ ( **X** [(] [2] [)] ), . . ., _W_ ( **X** [(] _[N]_ [)] ) yy

= _W_ ( **X** [(] [1] [)] ) ˆ [1] _W_ ( **X** [(] [2] [)] ) ˆ [1] ¨ ¨ ¨ ˆ [1] _W_ ( **X** [(] _[N]_ [)] ) . (4.22)


**The N-D discrete convolution in a TT format** of tensors **X** P **R** _[I]_ [1] [ˆ¨¨¨ˆ] _[I]_ _[N]_
with TT rank t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u and **Y** P **R** _[J]_ [1] [ˆ¨¨¨ˆ] _[J]_ _[N]_ with TT rank
t _Q_ 1, _Q_ 2, . . ., _Q_ _N_ ´1 u can be computed as


**Z** = **X** ˚ **Y** (4.23)
= xx **Z** [(] [1] [)], **Z** [(] [2] [)], . . ., **Z** [(] _[N]_ [)] yy P **R** [(] _[I]_ [1] [+] _[J]_ [1] [´][1] [)] [ˆ] [(] _[I]_ [2] [+] _[J]_ [2] [´][1] [)] [ˆ¨¨¨ˆ] [(] _[I]_ _[N]_ [+] _[J]_ _[N]_ [´][1] [)],


with the TT-cores given by


**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] d 2 **Y** [(] _[n]_ [)] P **R** [(] _[R]_ _[n]_ [´][1] _[Q]_ _[n]_ [´][1] [)] [ˆ] [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)] [ˆ] [(] _[R]_ _[n]_ _[Q]_ _[n]_ [)], (4.24)


or, equivalently, using the standard convolution **Z** [(] _[n]_ [)] ( _s_ _n_ ´1, :, _s_ _n_ ) =
**X** [(] _[n]_ [)] ( _r_ _n_ ´1, :, _r_ _n_ ) ˚ **Y** [(] _[n]_ [)] ( _q_ _n_ ´1, :, _q_ _n_ ) P **R** [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)] for _s_ _n_ = 1, 2, . . ., _R_ _n_ _Q_ _n_ and
_n_ = 1, 2, . . ., _N_, _R_ 0 = _R_ _N_ = 1.


**Inner product.** The computation of the inner (scalar, dot) product of
two _N_ th-order tensors, **X** = xx **X** [(] [1] [)], **X** [(] [2] [)], . . ., **X** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and
**Y** = xx **Y** [(] [1] [)], **Y** [(] [2] [)], . . ., **Y** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, is given by


x **X**, **Y** y = xvec ( **X** ), vec ( **Y** ) y (4.25)



_I_ _N_
ÿ _x_ _i_ 1 ... _i_ _n_ _y_ _i_ 1 ¨¨¨ _i_ _N_

_i_ _N_ = 1



=



_I_ 1


¨ ¨ ¨

ÿ

_i_ 1 = 1



and has the complexity of _O_ ( _I_ _[N]_ ) in the raw tensor format. In TT formats,
the inner product can be computed with the reduced complexity of only
_O_ ( _NI_ ( _R_ [2] [ ˜] _R_ + _RR_ [˜] [2] )) when the inner product is calculated by moving
TT-cores from left to right and performing calculations on relatively small
matrices, **S** _n_ = **X** [(] _[n]_ [)] ˆ [1,2] 1,2 [(] **[Y]** [(] _[n]_ [)] [ ˆ] [1] **[ S]** _[n]_ [´][1] [)] [ P] **[ R]** _[R]_ _[n]_ [ˆ] _[R]_ [r] _[n]_ [ for] _[ n]_ [ =] [ 1, 2, . . .,] _[ N]_ [.]

The results are then sequentially multiplied by the next core **Y** [(] _[n]_ [+] [1] [)] (see
Algorithm 9).


**Computation of the Frobenius norm.** In a similar way, we can efficiently
compute the Frobenius norm of a tensor, } **X** } _F_ = ~~a~~ x **X**, **X** y, in the TT format.


129


**Algorithm 9** : **Inner product of two large-scale tensors in the TT**
**Format [67,** **158]**

**Input:** _N_ th-order tensors, **X** = xx **X** [(] [1] [)], **X** [(] [2] [)], . . ., **X** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_

and **Y** = xx **Y** [(] [1] [)], **Y** [(] [2] [)], . . ., **Y** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ in TT formats, with
TT-cores **X** P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ and **Y** P **R** _[R]_ [r] _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ [r] _[n]_
and _R_ 0 = _R_ [r] 0 = _R_ _N_ = _R_ [r] _N_ = 1
**Output:** Inner product x **X**, **Y** y = vec ( **X** ) [T] vec ( **Y** )

1: Initialization **S** 0 = 1
2: **for** _n_ = 1 to _N_ **do**
3: **Z** [(] ( 1 _[n]_ ) [)] [=] **[ S]** _[n]_ [´] [1] **[Y]** [(] ( 1 _[n]_ ) [)] [P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ r] _[R]_ _[n]_

4: **S** _n_ = **X** ă [(] _[n]_ 2 [)] ą [ T] **[Z]** ă [(] _[n]_ 2 [)] ą [P] **[ R]** _[R]_ _[n]_ [ˆ] _[R]_ [r] _[n]_

5: **end for**
6: **return** Scalar x **X**, **Y** y = **S** _N_ P **R** _[R]_ _[N]_ [ˆ] _[R]_ [r] _[N]_ = **R**, with _R_ _N_ = _R_ [r] _N_ = 1


For the so-called _n_ -orthogonal [6] TT format, it is easy to show that


} **X** } _F_ = } **X** [(] _[n]_ [)] } _F_ . (4.26)


**Matrix-by-vector multiplication.** Consider a huge-scale matrix equation
(see Figure 4.9 and Figure 4.10)


**Ax** = **y**, (4.27)


where **A** P **R** _[I]_ [ˆ] _[J]_, **x** P **R** _[J]_ and **y** P **R** _[I]_ are represented approximately in
the TT format, with _I_ = _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _N_ and _J_ = _J_ 1 _J_ 2 ¨ ¨ ¨ _J_ _N_ . As shown in Figure
4.9(a), the cores are defined as **A** [(] _[n]_ [)] P **R** _[P]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[P]_ _[n]_, **X** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_
and **Y** [(] _[n]_ [)] P **R** _[Q]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[Q]_ _[n]_ .
Upon representing the entries of the matrix **A** and vectors **x** and **y** in


6 An _N_ th-order tensor **X** = xx **X** ( 1 ), **X** ( 2 ) . . ., **X** ( _N_ ) yy in the TT format is called _n_ -orthogonal
if all the cores to the left of the core **X** [(] _[n]_ [)] are left-orthogonalized and all the cores to the right
of the core **X** [(] _[n]_ [)] are right-orthogonalized (see Part 2 for more detail).


130


their tensorized forms, given by



**A** =


**X** =


**Y** =



_P_ 1, _P_ 2,..., _P_ _N_ ´1

**A** [(] [1] [)]

ÿ 1, _p_ 1 [˝] **[ A]** [(] _p_ [2] 1 [)], _p_ 2 [˝ ¨ ¨ ¨ ˝] **[ A]** [(] _p_ _[N]_ _N_ [)] ´1,1

_p_ 1, _p_ 2,..., _p_ _N_ ´1 = 1


_R_ 1, _R_ 2,..., _R_ _N_ ´1
ÿ **x** [(] _r_ [1] 1 [)] [˝] **[ x]** [(] _r_ [2] 1 [)], _r_ 2 [˝ ¨ ¨ ¨ ˝] **[ x]** [(] _r_ _[N]_ _N_ [)] ´1 (4.28)

_r_ 1, _r_ 2,..., _r_ _N_ ´1 = 1


_Q_ 1, _Q_ 2,..., _Q_ _N_ ´1
ÿ **y** [(] _q_ [1] 1 [)] [˝] **[ y]** [(] _q_ [2] 1 [)], _q_ 2 [˝ ¨ ¨ ¨ ˝] **[ y]** [(] _q_ _[N]_ _N_ [)] ´1 [,]

_q_ 1, _q_ 2,..., _q_ _N_ ´1 = 1



we arrive at a simple formula for the tubes of the tensor **Y**, in the form


**y** [(] _q_ _[n]_ _n_ ´ [)] 1, _q_ _n_ [=] **[ y]** ~~_r_~~ [(] _n_ _[n]_ ´ [)] 1 ~~_p_~~ _n_ ´1 ~~,~~ ~~_r_~~ _n_ ~~_p_~~ _n_ [=] **[ A]** [(] _p_ _[n]_ _n_ ´ [)] 1, _p_ _n_ **[x]** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [,]


with _Q_ _n_ = _P_ _n_ _R_ _n_ for _n_ = 1, 2, . . ., _N_ .
Furthermore, by representing the matrix **A** and vectors **x**, **y** via the
strong Kronecker products


˜
**A** = **A** [(] [1] [)] |b| ˜ **A** [(] [2] [)] |b| ¨ ¨ ¨ |b| ˜ **A** [(] _[N]_ [)]


˜
**x** = **X** [(] [1] [)] |b| ˜ **X** [(] [2] [)] |b| ¨ ¨ ¨ |b| ˜ **X** [(] _[N]_ [)] (4.29)


˜
**y** = **Y** [(] [1] [)] |b| ˜ **Y** [(] [2] [)] |b| ¨ ¨ ¨ |b| ˜ **Y** [(] _[N]_ [)],


˜
with **A** [˜] [(] _[n]_ [)] P **R** _[P]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ _[P]_ _[n]_, **X** [(] _[n]_ [)] P **R** _[R]_ _n_ ´1 _[J]_ _n_ [ˆ] _[R]_ _n_ and ˜ **Y** [(] _[n]_ [)] P **R** _[Q]_ _n_ ´1 _[I]_ _n_ [ˆ] _[Q]_ _n_, we can
establish a simple relationship


˜
**Y** [(] _[n]_ [)] = ˜ **A** [(] _[n]_ [)] |‚| ˜ **X** [(] _[n]_ [)] P **R** _[R]_ _n_ ´1 _[P]_ _n_ ´1 _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ _[P]_ _[n]_, _n_ = 1, . . ., _N_, (4.30)


where the operator |‚| represents the C (Core) product of two block
matrices.
The C product of a block matrix **A** [(] _[n]_ [)] P **R** _[P]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[P]_ _[n]_ _[J]_ _[n]_ with blocks
**A** [(] _p_ _[n]_ _n_ ´ [)] 1, _p_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [, and a block matrix] **[ B]** [(] _[n]_ [)] [ P] **[ R]** _[R]_ _[n]_ [´][1] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_ _[K]_ _[n]_ [, with blocks]

**B** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[J]_ _[n]_ [ˆ] _[K]_ _[n]_ [, is defined as] **[ C]** [(] _[n]_ [)] [ =] **[ A]** [(] _[n]_ [)] [ |‚|] **[ B]** [(] _[n]_ [)] [ P] **[ R]** _[Q]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[Q]_ _[n]_ _[K]_ _[n]_ [, the]

blocks of which are given by **C** [(] _q_ _[n]_ _n_ ´ [)] 1, _q_ _n_ [=] **[ A]** [(] _p_ _[n]_ _n_ ´ [)] 1, _p_ _n_ **[B]** _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [P] **[ R]** _[I]_ _[n]_ [ˆ] _[K]_ _[n]_ [, where]
_q_ _n_ = ~~_p_~~ _n_ ~~_r_~~ _n_ ~~,~~ as illustrated in Figure 4.11.
Note that, equivalently to Eq. (4.30), for **Ax** = **y**, we can use a slice
representation, given by



**Y** [(] _[n]_ [)] =
_i_ _n_



_J_ _n_
ÿ ( **A** _i_ [(] _n_ _[n]_, [)] _j_ _n_ [b] _[L]_ **[ X]** [(] _j_ _n_ _[n]_ [)] [)] [,] (4.31)

_j_ _n_ = 1


131


(a)


(b)



**X**


**Y**






















|A(|1)|
|---|---|
|~~**A**(~~||


|Col1|(2)|
|---|---|
|||


|(|n)|
|---|---|
|||


|(|N)|
|---|---|
|||
|||








































|Col1|(n)|
|---|---|
|~~**X**~~||


|Col1|(N)|
|---|---|
|||


|Col1|K<br>1<br>(1)|Col3|Col4|
|---|---|---|---|
|~~**X**~~<br>|~~(1)~~||(2)|
|~~**X**~~<br>||||
|||||
|~~**A**~~<br><br>|~~(1)~~|~~(1)~~|~~(1)~~|
|~~**A**~~<br><br>||||


|A|(2)|
|---|---|
|||


|Col1|(n)|
|---|---|
|~~**A**~~<br>||


|A|(N)|
|---|---|
|||





_K_ _N_



_K_ _n_





~~**X**~~ ~~(1)~~ ~~**X**~~ (2) ~~**X**~~ _n_ ~~**X**~~ ( _N_ )





_R_ 1 _R_ 2 _R_ _n_



_J_ _n_ _J_ _N_











~~**A**~~ ~~(1)~~ _P_ 1 ~~**A**~~ (2) _P_ 2 ~~**A**~~ ( _n_ ) _P_ _n_ ~~**A**~~ ( _N_ )



_I_ _n_ _I_ _N_









_I_ 1 _I_ 2


|Col1|K<br>1|K ・・・<br>2|K ・・・<br>n|Col5|
|---|---|---|---|---|
||||||
||_J_1|_J_2<br>・・・|_Jn_<br>・・・|_Jn_<br>・・・|
||||||
||_I_1|_I_2<br>・・・|_In_<br>・・・|_In_<br>・・・|





**X**


**A**


**Y**



_K_ 1 _K_ 2 _K_ _n_ _K_ _N_













_Q_



1 _Q_ 2 _Q_





~~**Y**~~ (2) ~~**Y**~~ _n_ ~~**Y**~~ ( _N_ )



_n_


|Y|(1)|
|---|---|
|~~**Y**~~<br>||


|Y|(2)|
|---|---|
|||


|Y|(n)|
|---|---|
|||


|Y|(N)|
|---|---|
|||



_I_ 1 _I_ 2 _I_ _n_ _I_ _N_












|Col1|K<br>1|K ・・・<br>2|K ・・・<br>n|Col5|
|---|---|---|---|---|
||||||
||||||
||_I_1|_I_2<br>・・・|_In_<br>・・・|_In_<br>・・・|



Figure 4.9: Linear systems represented by arbitrary tensor networks ( _left_ )
and TT networks ( _right_ ) for (a) **Ax** – **y** and (b) **AX** – **Y** .


132


Table 4.5: Basic operations on tensors in TT formats, where **X** = **X** [(] [1] [)] ˆ [1]

**X** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **X** [(] _[N]_ [)] P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, **Y** = **Y** [(] [1] [)] ˆ [1] **Y** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **Y** [(] _[N]_ [)] P
**R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[N]_, and **Z** = **Z** [(] [1] [)] ˆ [1] **Z** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **Z** [(] _[N]_ [)] P **R** _[K]_ [1] [ˆ] _[K]_ [2] [ˆ¨¨¨ˆ] _[K]_ _[N]_ .


Operation TT-cores


**Z** = **X** + **Y** = **X** [(] [1] [)] ‘ 2 **Y** [(] [1] [)] [�] ˆ [1] [ �] **X** [(] [2] [)] ‘ 2 **Y** [(] [2] [)] [�] ˆ [1] ¨ ¨ ¨ ˆ [1] [ �] **X** [(] _[N]_ [)] ‘ 2 **Y** [(] _[N]_ [)] [�]
�


**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] ‘ 2 **Y** [(] _[n]_ [)], with TT core slices **Z** _i_ [(] _n_ _[n]_ [)] = **X** _i_ [(] _n_ _[n]_ [)] [‘] **[ Y]** _i_ [(] _n_ _[n]_ [)] [,] [ (] _[I]_ _[n]_ [ =] _[ J]_ _[n]_ [ =] _[ K]_ _[n]_ [,][ @] _[n]_ [)]


**Z** = **X** ‘ **Y** = **X** [(] [1] [)] ‘ **Y** [(] [1] [)] [�] ˆ [1] [�] **X** [(] [2] [)] ‘ **Y** [(] [2] [)] [�] ˆ [1] ¨ ¨ ¨ ˆ [1] [�] **X** [(] _[N]_ [)] ‘ **Y** [(] _[N]_ [)] [�]
�


**Z** = **X** f **Y** = **X** [(] [1] [)] d 2 **Y** [(] [1] [)] [�] ˆ [1] [ �] **X** [(] [2] [)] d 2 **Y** [(] [2] [)] [�] ˆ [1] ¨ ¨ ¨ ˆ [1] [ �] **X** [(] _[N]_ [)] d 2 **Y** [(] _[N]_ [)] [�]
�


**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] d 2 **Y** [(] _[n]_ [)], with TT core slices **Z** _i_ [(] _n_ _[n]_ [)] = **X** _i_ [(] _n_ _[n]_ [)] [b] **[ Y]** _i_ [(] _n_ _[n]_ [)] [,] [ (] _[I]_ _[n]_ [ =] _[ J]_ _[n]_ [ =] _[ K]_ _[n]_ [,][ @] _[n]_ [)]


**Z** = **X** b **Y** = **X** [(] [1] [)] b **Y** [(] [1] [)] [�] ˆ [1] [ �] **X** [(] [2] [)] b **Y** [(] [2] [)] [�] ˆ [1] ¨ ¨ ¨ ˆ [1] [ �] **X** [(] _[N]_ [)] b **Y** [(] _[N]_ [)] [�]
�

**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] b **Y** [(] _[n]_ [)], with TT core slices **Z** [(] _k_ _[n]_ _n_ [)] [=] **[ X]** _i_ [(] _n_ _[n]_ [)] [b] **[ Y]** [(] _j_ _n_ _[n]_ [)] ( _k_ _n_ = _i_ _n_ _j_ _n_ )


**Z** = **X** ˚ **Y** = ( **X** [(] [1] [)] d 2 **Y** [(] [1] [)] ) ˆ [1] ¨ ¨ ¨ ˆ [1] ( **X** [(] _[N]_ [)] d 2 **Y** [(] _[N]_ [)] )


**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] d 2 **Y** [(] _[n]_ [)] P **R** [(] _[R]_ _[n]_ [´][1] _[Q]_ _[n]_ [´][1] [)] [ˆ] [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)] [ˆ] [(] _[R]_ _[n]_ _[Q]_ _[n]_ [)], with vectors


**Z** [(] _[n]_ [)] ( _s_ _n_ ´1, :, _s_ _n_ ) = **X** [(] _[n]_ [)] ( _r_ _n_ ´1, :, _r_ _n_ ) ˚ **Y** [(] _[n]_ [)] ( _q_ _n_ ´1, :, _q_ _n_ ) P **R** [(] _[I]_ _[n]_ [+] _[J]_ _[n]_ [´][1] [)]


for _s_ _n_ = 1, 2, . . ., _R_ _n_ _Q_ _n_ and _n_ = 1, 2, . . ., _N_, _R_ 0 = _R_ _N_ = 1.


**Z** = **X** ˆ _n_ **A** = **X** [(] [1] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **X** [(] _[n]_ [´][1] [)] ˆ [1] [�] **X** [(] _[n]_ [)] ˆ 2 **A** ˆ [1] **X** [(] _[n]_ [+] [1] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **X** [(] _[N]_ [)]
�


_z_ = x **X**, **Y** y = **Z** [(] [1] [)] ˆ [1] **Z** [(] [2] [)] ˆ [1] ¨ ¨ ¨ ˆ [1] **Z** [(] _[N]_ [)] = **Z** [(] [1] [)] **Z** [(] [2] [)] ¨ ¨ ¨ **Z** [(] _[N]_ [)]



**Z** [(] _[n]_ [)] = **X** [(] _[n]_ [)] d 2 **Y** [(] _[n]_ [)] [�]
�



~~ˆ~~ 2 **1** _I_ _n_ = [ř] _i_ _n_ **[X]** _i_ [(] _n_ _[n]_ [)] [b] **[ Y]** _i_ [(] _n_ _[n]_ [)] ( _I_ _n_ = _J_ _n_, @ _n_ )


133


Table 4.6: Basic operations in the TT format expressed via the strong
Kronecker and C products of block matrices, where **A** = **A** [r] [(] [1] [)] |b| **A** [r] [(] [2] [)] |b
| ¨ ¨ ¨ |b| **A** [r] [(] _[N]_ [)], **B** = **B** [r] [(] [1] [)] |b| **B** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **B** [r] [(] _[N]_ [)], **x** = **X** [r] [(] [1] [)] |b| **X** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **X** [r] [(] _[N]_ [)],
**y** = **Y** [r] [(] [1] [)] |b| **Y** [r] [(] [2] [)] |b| ¨ ¨ ¨ |b| **Y** [r] [(] _[N]_ [)] and the block matrices **A** [r] [(] _[n]_ [)] P **R** _[R]_ _n_ _[A]_ ´1 _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ _[R]_ _n_ _[A]_,

r
**B** [(] _[n]_ [)] P **R** _[R]_ _n_ _[B]_ ´1 _[J]_ _[n]_ [ˆ] _[K]_ _[n]_ _[R]_ _n_ _[B]_, r **X** [(] _[n]_ [)] P **R** _[R]_ _n_ _[x]_ ´1 _[I]_ _[n]_ [ˆ] _[R]_ _n_ _[x]_, r **Y** [(] _[n]_ [)] P **R** _[R]_ _n_ _[y]_ ´1 _[I]_ _[n]_ [ˆ] _[R]_ _n_ _[y]_ .


Operation Block matrices of TT-cores

**Z** = **A** + **B**



�



�



�



r r
= **A** [(] [1] [)] **B** [(] [1] [)] |b|
� �



**A** r [(] [2] [)] **0**

r
**0** **B** [(] [2] [)]
�



|b| ¨ ¨ ¨ |b|



**A** r [(] _[N]_ [´] [1] [)] **0**

r
**0** **B** [(] _[N]_ [´] [1] [)]
�



|b|



**A** r [(] _[N]_ [)]

r
**B** [(] _[N]_ [)]
�



**Z** = **A** b **B** = **A** [r] [(] [1] [)] |b| ¨ ¨ ¨ |b| **A** [r] [(] _[N]_ [)] |b| **B** [r] [(] [1] [)] |b| ¨ ¨ ¨ |b| **B** [r] [(] _[N]_ [)]


r r
_z_ = **x** [T] **y** = x **x**, **y** y = **X** [(] [1] [)] |‚| r **Y** [(] [1] [)] [�] |b| ¨ ¨ ¨ |b| **X** [(] _[N]_ [)] |‚| r **Y** [(] _[N]_ [)] [�]
� �


**Z** r [(] _[n]_ [)] = r **X** [(] _[n]_ [)] |‚| r **Y** [(] _[n]_ [)] P **R** _[R]_ _n_ _[x]_ ´1 _[R]_ _[y]_ _n_ ´1 [ˆ] _[R]_ _[xn]_ _[R]_ _n_ _[y]_, with core slices **Z** ( _n_ ) = [ř] _i_ _n_ **[X]** _i_ [(] _n_ _[n]_ [)] [b] **[ Y]** _i_ [(] _n_ _[n]_ [)]


r r
**z** = **Ax** = **A** [(] [1] [)] |‚| r **X** [(] [1] [)] [�] |b| ¨ ¨ ¨ |b| **A** [(] _[N]_ [)] |‚| r **X** [(] _[N]_ [)] [�]
� �


r r
**Z** [(] _[n]_ [)] = r **A** [(] _[n]_ [)] ˆ [1] **X** [(] _[n]_ [)], with blocks (vectors)

**z** [(] _s_ _n_ _[n]_ ´ [)] 1, _s_ _n_ [=] **[ A]** _r_ [(] _n_ _[n]_ _[A]_ ´ [)] 1 [,] _[r]_ _[An]_ **[ x]** _r_ [(] _n_ _[x]_ _[n]_ ´ [)] 1 [,] _[r]_ _[xn]_ ( _s_ _n_ = _r_ _n_ _[A]_ _r_ _n_ _[x]_ )


r r
**Z** = **AB** = **A** [(] [1] [)] |‚| r **B** [(] [1] [)] [�] |b| ¨ ¨ ¨ |b| **A** [(] _[N]_ [)] |‚| r **B** [(] _[N]_ [)] [�]
� �


r
**Z** [(] _[n]_ [)] = r **A** [(] _[n]_ [)] |‚| r **B** [(] _[n]_ [)], with blocks

**Z** [(] _s_ _n_ _[n]_ ´ [)] 1, _s_ _n_ [=] **[ A]** _r_ [(] _n_ _[n]_ _[A]_ ´ [)] 1 [,] _[r]_ _[An]_ **[ B]** _r_ [(] _n_ _[B]_ _[n]_ ´ [)] 1 [,] _[r]_ _[Bn]_ ( _s_ _n_ = _r_ _n_ _[A]_ _r_ _n_ _[B]_ )


_z_ = **x** [T] **Ax** = x **x**, **Ax** y

r r
= **X** [(] [1] [)] |‚| r **A** [(] [1] [)] |‚| r **X** [(] [1] [)] [�] |b| ¨ ¨ ¨ |b| **X** [(] _[N]_ [)] |‚| r **A** [(] _[N]_ [)] |‚| r **X** [(] _[N]_ [)] [�]
� �


r
**Z** [(] _[n]_ [)] = r **X** [(] _[n]_ [)] |‚| r **A** [(] _[n]_ [)] |‚| r **X** [(] _[n]_ [)] P **R** _[R]_ _n_ _[x]_ ´1 _[R]_ _n_ _[A]_ ´1 _[R]_ _n_ _[x]_ ´1 [ˆ] _[R]_ _[xn]_ _[R]_ _[An]_ _[ R]_ _[xn]_, with blocks (entries)



_z_ [(] _s_ _n_ _[n]_ ´ [)] 1, _s_ _n_ [=] B **x** _r_ [(] _n_ _[x]_ _[n]_ ´ [)] 1 [,] _[r]_ _[xn]_ [,] **[ A]** _r_ [(] _n_ _[n]_ _[A]_ ´ [)] 1 [,] _[r]_ _[An]_ **[ x]** _r_ [(] _n_ _[y]_ _[n]_ ´ [)] 1 [,] _[r]_ _n_ _[y]_



( _s_ _n_ = _r_ _n_ _[x]_ _r_ _n_ _[A]_ _r_ _n_ ~~_[y]_~~ [)]
F


134


(a)


(b)









{



T








|A(|1)|
|---|---|
|**A**~~(~~||


|(|2)|
|---|---|
|||
|||


|Col1|(N)|
|---|---|
|||



_J_ 1 _J_ 2 _J_ _n_ _J_ _N_

**x** **[X]** [(1)] _R_ 1 ~~**X**~~ [(2)] _R_ 2 _R_ _n-_ 1 ~~**X**~~ [( )] _n_ _R_ _n_ ~~**X**~~ [(] _N_ )

{




|Col1|Col2|( n)|
|---|---|---|
|_Pn-_1<br>_Jn_<br>_Rn-_1<br><br>|||


|A|(1)|
|---|---|
|||


|A|(2)|
|---|---|
|||


|A|( n)|
|---|---|
|||


|A(|N)|
|---|---|
|||


|A|(1)|
|---|---|
|**A**||
|||


|Col1|(2)|
|---|---|
|||


|Col1|( n)|
|---|---|
|~~**A**~~||


|(|N)|
|---|---|
|||
|||











**A** ~~[(1)]~~ ~~**A**~~ [(2)] ~~**A**~~ [( )] _n_ ~~**A**~~ ~~[(]~~ _N_ )

_P_ 1 _P_ 2 _P_ _n-_ 1 _P_ _n_





**A**

{

















































Figure 4.10: Representation of typical cost functions by arbitrary TNs and
by TT networks: (a) _J_ 1 ( **x** ) = **y** [T] **Ax** and (b) _J_ 2 ( **x** ) = **x** [T] **A** [T] **Ax** . Note that
tensors **A**, **X** and **Y** can be, in general, approximated by any TNs that
provide good low-rank representations.


which can be implemented by fast matrix-by matrix multiplication
algorithms (see Algorithm 10). In practice, for very large scale data, we
usually perform TT core contractions (MPO-MPS product) approximately,
with reduced TT ranks, e.g., via the “zip-up” method proposed by [198].
In a similar way, the matrix equation


**Y** – **AX**, (4.32)


where **A** P **R** _[I]_ [ˆ] _[J]_, **X** P **R** _[J]_ [ˆ] _[K]_, **Y** P **R** _[I]_ [ˆ] _[K]_, with _I_ = _I_ 1 _I_ 2 ¨ ¨ ¨ _I_ _N_, _J_ = _J_ 1 _J_ 2 ¨ ¨ ¨ _J_ _N_
and _K_ = _K_ 1 _K_ 2 ¨ ¨ ¨ _K_ _N_, can be represented in TT formats. This is illustrated


135


|A<br>11|A<br>12|
|---|---|
|~~**A**~~21|~~**A**~~22|






















|A B<br>11 11|A B<br>11 12|A B<br>12 11|A B<br>12 12|
|---|---|---|---|
|**A**11<br>21<br>**B**|**A**11<br>22<br>**B**|**A**12<br>21<br>**B**|**A**12<br>22<br>**B**|
|**A**11<br>31<br>**B**|**A**11<br>32<br>**B**|**A**12<br>31<br>**B**|**A**12<br>32<br>**B**|
|~~**A**~~21<br>11<br>~~**B**~~|~~**A**~~21<br>12<br>~~**B**~~|**A**22<br>11<br>**B**<br><br>|**A**22<br>12<br>**B**|
|~~**A**~~21<br>21<br>~~**B**~~<br><br>|~~**A**~~21<br>22<br>~~**B**~~<br><br>|~~**A**~~22<br>21<br>~~**B**~~|~~**A**~~22<br>22<br>~~**B**~~|
|~~**A**~~21<br>31<br>~~**B**~~|~~**A**~~21<br>32<br>~~**B**~~|**A**22<br>31<br>**B**|**A**22<br>32<br>**B**|





Figure 4.11: Graphical illustration of the C product of two block matrices.


in Figure 4.9(b) for the corresponding TT-cores defined as


**A** [(] _[n]_ [)] P **R** _[P]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[J]_ _[n]_ [ˆ] _[P]_ _[n]_


**X** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[J]_ _[n]_ [ˆ] _[K]_ _[n]_ [ˆ] _[R]_ _[n]_


**Y** [(] _[n]_ [)] P **R** _[Q]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[K]_ _[n]_ [ˆ] _[Q]_ _[n]_ .


It is straightforward to show that when the matrices, **A** P **R** _[I]_ [ˆ] _[J]_ and
**X** P **R** _[J]_ [ˆ] _[K]_, are represented in their TT formats, they can be expressed via
a strong Kronecker product of block matrices as **A** = **A** [˜] [(] [1] [)] |b| **A** [˜] [(] [2] [)] |b| ¨ ¨ ¨ |b
|˜ **A** [˜] [(] _[N]_ [)] and **X** = **X** [˜] [(] [1] [)] |b| **X** [˜] [(] [2] [)] |b| ¨ ¨ ¨ |b| **X** [˜] [(] _[N]_ [)], where the factor matrices are
**A** [(] _[n]_ [)] P **R** _[P]_ _n_ ´1 _[I]_ _n_ [ˆ] _[J]_ _n_ _[P]_ _n_ and ˜ **X** [(] _[n]_ [)] P **R** _[R]_ _n_ ´1 _[J]_ _n_ [ˆ] _[K]_ _n_ _[R]_ _n_ . Then, the matrix **Y** = **AX**
can also be expressed via the strong Kronecker products, **Y** = **Y** [˜] [(] [1] [)] |b| ¨ ¨ ¨ |b
| **Y** [˜] [(] _[N]_ [)], where **Y** [˜] [(] _[n]_ [)] = **A** [˜] [(] _[n]_ [)] |‚| **X** [˜] [(] _[n]_ [)] P **R** _[Q]_ _[n]_ [´][1] _[ I]_ _[n]_ [ˆ] _[K]_ _[n]_ _[ Q]_ _[n]_, ( _n_ = 1, 2, . . ., _N_ ), with
blocks **Y** [˜] [(] _q_ _[n]_ _n_ ´ [)] 1, _q_ _n_ [=] [ ˜] **[A]** [(] _p_ _[n]_ _n_ ´ [)] 1, _p_ _n_ **[X]** [˜] _r_ [(] _n_ _[n]_ ´ [)] 1, _r_ _n_ [, where] _[ Q]_ _n_ [=] _[ R]_ _n_ _[P]_ _n_ [,] _[ q]_ _n_ [=] ~~_[ p]_~~ _n_ ~~_[r]_~~ _n_ ~~[,]~~ [ @] _[n]_ [.]
Similarly, a quadratic form, _z_ = **x** [T] **Ax**, for a huge symmetric matrix
**A**, can be computed by first computing (in TT formats), a vector **y** = **Ax**,
followed by the inner product **x** [T] **y** .
Basic operations in the TT format are summarized in Table 4.5, while
Table 4.6 presents these operations expressed via strong Kronecker and
C products of block matrices of TT-cores. For more advanced and
sophisticated operations in TT/QTT formats, see [112,113,128].


**4.6** **Algorithms for TT Decompositions**


We have shown that a major advantage of the TT decomposition is the
existence of efficient algorithms for an exact representation of higher

136


**Algorithm 10** : **Computation of a Matrix-by-Vector Product in the TT**
**Format**

**Input:** Matrix **A** P **R** _[I]_ [ˆ] _[J]_ and vector **x** P **R** _[J]_ in their respective TT format
**A** = xx **A** [(] [1] [)], **A** [(] [2] [)], . . ., **A** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[J]_ [1] [ˆ] _[I]_ [2] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ [ˆ] _[J]_ _[N]_,
and **X** = xx **X** [(] [1] [)], **X** [(] [2] [)], . . ., **X** [(] _[N]_ [)] yy P **R** _[J]_ [1] [ˆ] _[J]_ [2] [ˆ¨¨¨ˆ] _[J]_ _[N]_,
with TT-cores **X** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[n]_ and **A** [(] _[n]_ [)] P **R** _[R]_ _n_ _[A]_ ´1 [ˆ] _[I]_ _[n]_ [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _n_ _[A]_
**Output:** Matrix by vector product **y** = **Ax** in the TT format
**Y** = xx **Y** [(] [1] [)], **Y** [(] [2] [)], . . ., **Y** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_, with cores
**Y** [(] _[n]_ [)] P **R** _[R]_ _[Y]_ _n_ ´1 [ˆ] _[J]_ _[n]_ [ˆ] _[R]_ _[Y]_ _n_


1: **for** _n_ = 1 to _N_ **do**


2: **for** _i_ _n_ = 1 to _I_ _n_ **do**



3: **Y** _i_ [(] _n_ _[n]_ [)] = [ř] _j_ _[J]_ _n_ _[n]_ = 1 � **A** _i_ [(] _n_ _[n]_, [)] _j_ _n_ [b] _[L]_ **[ X]** [(] _j_ _n_ _[n]_ [)]

4: **end for**


5: **end for**



�



6: **return y** P **R** _[I]_ [1] _[I]_ [2] [¨¨¨] _[I]_ _[N]_ in the TT format **Y** = xx **Y** [(] [1] [)], **Y** [(] [2] [)], . . ., **Y** [(] _[N]_ [)] yy


order tensors and/or their low-rank approximate representations with a
prescribed accuracy. Similarly to the quasi-best approximation property

( 1 ) p ( 2 ) p ( _N_ )

of the HOSVD, the TT approximation **X** [p] = xx **X** [p], **X**, . . ., **X** yy P

**R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ (with core tensors denoted by **X** [(] _[n]_ [)] = **G** [(] _[n]_ [)] ), obtained by the
TT-SVD algorithm, satisfies the following inequality



( 2 ) p
, . . ., **X**



of the HOSVD, the TT approximation **X** [p] = xx **X** [p]



( 1 ) p
, **X**



_I_ _n_
ÿ _σ_ _j_ [2] [(] **[X]** [ă] _[n]_ [ą] [)] [,] (4.33)

_j_ = _R_ _n_ + 1



} **X** ´ **X** [p] } [2] 2 [ď]



_N_ ´1
ÿ

_n_ = 1



where the _ℓ_ 2 -norm of a tensor is defined via its vectorization and _σ_ _j_ ( **X** ă _n_ ą )
denotes the _j_ th largest singular value of the unfolding matrix **X** ă _n_ ą [158].
The two basic approaches to perform efficiently TT decompositions are
based on: (1) low-rank matrix factorizations (LRMF), and (2) constrained
Tucker-2 decompositions.


**4.7** **Sequential SVD/LRMF Algorithms**


The most important algorithm for the TT decomposition is the TT-SVD
algorithm (see Algorithm 11) [161, 216], which applies the truncated SVD
sequentially to the unfolding matrices, as illustrated in Figure 4.12. Instead
of SVD, alternative and efficient LRMF algorithms can be used [50], see


137


**X**



**M =X** 1 (1)





Reshape _I_ 1 _I_ 2 _I_ 3 _I_ 4 _I_ 5







tSVD _I_ 1 **U** 1 _R_ 1 _R_ 1 **S** 1 **V** 1 T _I_ 2 _I_ 3 _I_ 4 _I_ 5



Reshape **X** (1) _R_ 1 _R_ 1 _I_ 2 **M** 2 _I_ 3 _I_ 4 _I_ 5



**S** **V** [T]
_R_ 2 [2 2] _I_ 3 _I_ 4 _I_ 5



tSVD


tSVD


Reshape



(1)
**X** _R_ 1


(1)
**X** _R_ 1



(1)
**X**



_R_ 1 _I_ 2 **U** 2 _R_ 2





_R_ 3 _I_ 4 **U** 4 _R_ 4 _R_ 4 **S V** 4 4 [T] _I_ 5



(2)

_R_ 1 **X** _R_ 2


(2)
**X**



(3)

_R_ 2 **X** _R_ 3


(3)
**X**





(4)
**X**





(5)
**X**



Figure 4.12: The TT-SVD algorithm for a 5th-order data tensor using
truncated SVD. Instead of the SVD, any alternative LRMF algorithm can
be employed, such as randomized SVD, RPCA, CUR/CA, NMF, SCA, ICA.
Top panel: A 6th-order tensor **X** of size _I_ 1 ˆ _I_ 2 ˆ ¨ ¨ ¨ˆ _I_ 5 is first reshaped into
a long matrix **M** 1 of size _I_ 1 ˆ _I_ 2 ¨ ¨ ¨ _I_ 5 . Second panel: The tSVD is performed
to produce low-rank matrix factorization, with _I_ 1 ˆ _R_ 1 factor matrix **U** 1 and
the _R_ 1 ˆ _I_ 2 ¨ ¨ ¨ _I_ 5 matrix **S** 1 **V** [T] 1 [, so that] **[ M]** [1] [ –] **[ U]** [1] **[S]** [1] **[V]** [T] 1 [. Third panel: the]
matrix **U** 1 becomes the first core core **X** [(] [1] [)] P **R** [1][ˆ] _[I]_ [1] [ˆ] _[R]_ [1], while the matrix
**S** 1 **V** [T] 1 [is reshaped into the] _[ R]_ [1] _[I]_ [2] [ ˆ] _[ I]_ [3] _[I]_ [4] _[I]_ [5] [ matrix] **[ M]** [2] [.] Remaining panels:
Perform tSVD to yield **M** 2 – **U** 2 **S** 2 **V** [T] 2 [, reshape] **[ U]** [2] [ into an] _[ R]_ [1] [ ˆ] _[ I]_ [2] [ ˆ] _[ R]_ [2] [ core]
**X** [(] [2] [)] and repeat the procedure until all the five cores are extracted (bottom
panel). The same procedure applies to higher order tensors of any order.


also Algorithm 12). For example, in [162] a new approximate formula
for TT decomposition is proposed, where an _N_ th-order data tensor **X**
is interpolated using a special form of cross-approximation. In fact,
the TT-Cross-Approximation is analogous to the TT-SVD algorithm, but
uses adaptive cross-approximation instead of the computationally more


138


**Algorithm 11** : **TT-SVD Decomposition using truncated SVD**
**(tSVD) or randomized SVD (rSVD) [158,** **216]**



**Input:** _N_ th-order tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and approximation accuracy _ε_
**Output:** Approximative representation of a tensor in the TT format
**X** p = xx **X** p ( 1 ), **X** p ( 2 ), . . ., **X** p ( _N_ ) yy, such that } **X** ´ **X** p} _F_ ď _ε_



**X** p = xx **X** p ( 1 ), **X** p ( 2 ), . . ., **X** p ( _N_ ) yy, such that } **X** ´ **X** p} _F_ ď _ε_

1: Unfolding of tensor **X** in mode-1 **M** 1 = **X** ( 1 )
2: Initialization _R_ 0 = 1
3: **for** _n_ = 1 to _N_ ´ 1 **do**
4: Perform tSVD [ **U** _n_, **S** _n_, **V** _n_ ] = tSVD ( **M** _n_, _ε_ /? _N_



( 1 ) p
, **X**



( 2 ) p
, . . ., **X**



4: Perform tSVD [ **U** _n_, **S** _n_, **V** _n_ ] = tSVD ( **M** _n_, _ε_ /? _N_ ´ 1 )

5: Estimate _n_ th TT rank _R_ _n_ = size ( **U** _n_, 2 )
6: Reshape orthogonal matrix **U** _n_ into a 3rd-order core
p ( _n_ )
**X** = **U** _R_ ´ _I_ _R_



p ( _n_ )
**X** = reshape ( **U** _n_, [ _R_ _n_ ´ 1, _I_ _n_, _R_ _n_ ])

7: Reshape the matrix **V** _n_ into a matrix



**M** _n_ + 1 = reshape � **S** _n_ **V** [T] _n_ [,] [ [] _[R]_ _[n]_ _[I]_ _n_ + 1 [,][ ś] _[N]_ _p_ = _n_ + 2 _[I]_ _[p]_ []] �

8: **end for**



9: Construct the last core as **X** [p]



( _N_ )
= reshape ( **M** _N_, [ _R_ _N_ ´ 1, _I_ _N_, 1 ])



( 2 ) p
, . . ., **X**



( _N_ )
yy.



10: **return** xx **X** [p]



( 1 ) p
, **X**



expensive SVD. The complexity of the cross-approximation algorithms
scales linearly with the order _N_ of a data tensor.


**4.8** **Tucker-2/PVD** **Algorithms** **for** **Large-scale** **TT**
**Decompositions**


The key idea in this approach is to reshape any _N_ th-order data tensor,
**X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ with _N_ ą 3, into a suitable 3rd-order tensor, e.g., **X** [r] P
**R** _[I]_ [1] [ ˆ] _[I]_ _[N]_ [ ˆ] _[ I]_ [2] [¨¨¨] _[I]_ _[N]_ [´][1], in order to apply the Tucker-2 decomposition as follows
(see Algorithm 8 and Figure 4.13(a))


r
**X** = **G** [(] [2,] _[N]_ [´][1] [)] ˆ 1 **X** [(] [1] [)] ˆ 2 **X** [(] _[N]_ [)] = **X** [(] [1] [)] ˆ [1] **G** [(] [2,] _[N]_ [´][1] [)] ˆ [1] **X** [(] _[N]_ [)], (4.34)


which, by using frontal slices of the involved tensors, can also be expressed
in the matrix form


**X** _k_ 1 = **X** [(] [1] [)] **G** _k_ 1 **X** [(] _[N]_ [)], _k_ 1 = 1, 2, . . ., _I_ 2 ¨ ¨ ¨ _I_ _N_ ´1 . (4.35)


Such representations allow us to compute the tensor, **G** [(] [2,] _[N]_ [´][1] [)], the first
TT-core, **X** [(] [1] [)], and the last TT-core, **X** [(] _[N]_ [)] . The procedure can be repeated


139


(a)


(b)


(c)



=





_I_ 1



_I_ 1


|I N-1 { X- ~|Col2|{ X- ~|
|---|---|---|
|...<br>_I_<br>_I_<br>_=_<br><br>...<br>...<br>2<br>_N_<br>`{`<br>|...<br>_I_<br>_I_<br>_=_<br><br>...<br>...<br>2<br>_N_<br>`{`<br>||
|...<br>_I_<br>_I_<br>_=_<br><br>...<br>...<br>2<br>_N_<br>`{`<br>|||
||||



_I_ _N_








|(2,N-1)<br>-G =-G { N-1<br>2 I...2...<br>I<br>=<br>K 1<br>...<br>X(1) R 1 R N-1 X(N)<br>R 1 R N-1 G I N k<br>1|Col2|Col3|(2,N-1)<br>-G =-G { N-1<br>2 I...2...<br>I<br>=<br>K 1<br>...<br>R 1 R N-1 X(N)|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|**X**(1)<br>**X**(_N_)<br>_RN_-1<br>2<br>_IN_-1<br>_K   =_<br>1 <br>_IN_<br>**G**<br>_k_<br>_R_1<br>...<br>...<br>1<br>_I_<br>...<br>_R_1<br>_RN_-1<br>`{`<br>**G-**<br>(2,_N_-1)<br>=**G-**2||(1)<br>**X**(_N_)<br>_RN_-1<br>2<br><br>_K   =_<br>1 <br>_IN_<br>**G**<br>_k_<br>1<br>...<br>...<br>1<br>_I_<br>...<br>_R_1<br>_RN_-1<br>`{`<br><br>|2<br><br>_K   =_<br>1 <br>...<br>_I_<br>...<br>_R_1<br><br>`{`|2<br><br>_K   =_<br>1 <br>...<br>_I_<br>...<br>_R_1<br><br>`{`|||
|**X**(1)<br>**X**(_N_)<br>_RN_-1<br>2<br>_IN_-1<br>_K   =_<br>1 <br>_IN_<br>**G**<br>_k_<br>_R_1<br>...<br>...<br>1<br>_I_<br>...<br>_R_1<br>_RN_-1<br>`{`<br>**G-**<br>(2,_N_-1)<br>=**G-**2||(1)<br>**X**(_N_)<br>_RN_-1<br>2<br><br>_K   =_<br>1 <br>_IN_<br>**G**<br>_k_<br>1<br>...<br>...<br>1<br>_I_<br>...<br>_R_1<br>_RN_-1<br>`{`<br><br>|2<br><br>_K   =_<br>1 <br>...<br>_I_<br>...<br>_R_1<br><br>`{`||||
||**X**<br>|(1)<br>|(1)<br>|(1)<br>|(1)<br>|(1)<br>|
||||||||



**-** **G~** _n_ **-** **G** _n_ +1



=



**X** <2> [(] _[n]_ [)] _R_ _n_



_R_ _p_ -1 **X** <1> [(] _[p]_ [)]


_I R_ _p_ _p_


|... { ...1+n... I = n|Col2|Col3|
|---|---|---|
|...<br>_I_<br><br>_  =_<br>_n_<br>...<br>...<br><br>_n_+1<br><br>`{`<br><br><br><br>|||
||||


|{ . I N-n|{ G - n+ -n|
|---|---|
|...<br><br>...<br>_I_<br><br>_n_+1<br><br><br>||
|||
|||



_I_ _p_ _R_ _p_



_R_ _n_ _R_ _p_ -1 **G**
_k_



_n_



_R_ _p_ -1



_n_



**X**







Reshape _I_ 1 **M** 1 _I_ 5









PVD or Tucker2 _I_ 1 **A** 1 _R_ 1 _R_ 1 **G** 2 _R_ 4



**B** T5



**G** 2 _R_ _R_ **B** 5 _I_



_R_ 4 5 _I_ 5





**~**





**B** T5



Reshape _I_ 1 **A** 1 _R_ 1 _R_ 1



4 _[R]_ 4 _R_ 4 _I_ 5




|R1 G 2 I2 I 4R4 ~|Col2|
|---|---|
|1<br>2<br><br>4|1<br>2<br><br>4|
||_I_3|



PVD or Tucker2 _I_ 1 **A** 1 _R_ 1 _R_ 1 _I_ 2 **A** 2 _R_ 2 _R_ 2 **G** 3 _R_



**B** T4 _I_ _R_ _R_ **B** T5





**G** 3 _R_ _R_ **B** 4 _I_ _R_ _R_ **B** 5 _I_



3 3 4 4 4 5







(5)
_R_ 4 **X**



(2)
**X**



(3)
**X**





(4)
**X**



Reshape



(1)
**X** _R_ 1



_R_











Figure 4.13: TT decomposition based on the Tucker-2/PVD model. (a) Extraction
of the first and the last core. (b) The procedure can be repeated sequentially for
reshaped 3rd-order tensors **G** _n_ (for _n_ = 2, 3, . . . and _p_ = _N_ ´ 1, _N_ ´ 2, . . .). (c)
Illustration of a TT decomposition for a 5th-order data tensor, using an algorithm
based on sequential Tucker-2/PVD decompositions.


140


**Algorithm 12** : **TT Decomposition using any efficient LRMF**



**Input:** Tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and the approximation accuracy _ε_
**Output:** Approximate tensor representation in the TT format
p p ( 1 ) p ( 2 ) p ( _N_ )
**X** – xx **X**, **X**, . . ., **X** yy



( 1 ) p
, **X**



( 2 ) p
, . . ., **X**



( _N_ )
yy



1: Initialization _R_ 0 = 1

2: Unfolding of tensor **X** in mode-1 as **M** 1 = **X** ( 1 )
3: **for** _n_ = 1 to _N_ ´ 1 **do**

4: Perform LRMF, e.g., CUR, RPCA, ...

[ **A** _n_, **B** _n_ ] = LRMF ( **M** _n_, _ε_ ), i.e., **M** _n_ – **A** _n_ **B** [T] _n_
5: Estimate _n_ th TT rank, _R_ _n_ = size ( **A** _n_, 2 )

6: Reshape matrix **A** _n_ into a 3rd-order core, as
p ( _n_ )
**X** = reshape ( **A** _n_, [ _R_ _n_ ´1, _I_ _n_, _R_ _n_ ])

7: Reshape the matrix **B** _n_ into the ( _n_ + 1 ) th unfolding matrix

**M** _n_ + 1 = reshape � **B** [T] _n_ [,] [ [] _[R]_ _[n]_ _[I]_ _[n]_ [+] [1] [,][ ś] _[N]_ _p_ = _n_ + 2 _[I]_ _[p]_ []] �

8: **end for**



9: Construct the last core as **X** [p]



( _N_ )
= reshape ( **M** _N_, [ _R_ _N_ ´1, _I_ _N_, 1 ])



( _N_ )
yy.



10: **return** TT-cores: xx **X** [p]



( 1 ) p ( 2 ) p
, **X**, . . ., **X**



sequentially for reshaped tensors **G** [r] _n_ = **G** [(] _[n]_ [+] [1,] _[N]_ [´] _[n]_ [)] for _n_ = 1, 2, . . ., in order
to extract subsequent TT-cores in their matricized forms, as illustrated
in Figure 4.13(b). See also the detailed step-by-step procedure shown in
Figure 4.13(c).
Such a simple recursive procedure for TT decomposition can be used in
conjunction with any efficient algorithm for Tucker-2/PVD decompositions
or the nonnegative Tucker-2 decomposition (NTD-2) (see also Section 3).


**4.9** **Tensor Train Rounding – TT Recompression**


_Mathematical operations in TT format produce core tensors with ranks which_
_are not guaranteed to be optimal with respect to the desired approximation_
_accuracy_ . For example, matrix-by-vector or matrix-by-matrix products
considerably increase the TT ranks, which quickly become computationally
prohibitive, so that a truncation or low-rank TT approximations are
necessary for mathematical tractability. To this end, the TT–rounding
(also called truncation or recompression) may be used as a post-processing
procedure to reduce the TT ranks. The TT rounding algorithms are


141


**Algorithm 13** : **TT Rounding (Recompression) [158]**

**Input:** _N_ th-order tensor **X** = xx **X** [(] [1] [)], **X** [(] [2] [)], . . ., **X** [(] _[N]_ [)] yy P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_,
in a TT format with an overestimated TT rank,
**r** _TT_ = t _R_ 1, _R_ 2, . . ., _R_ _N_ ´1 u, and TT-cores **X** P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_,
absolute tolerance _ε_, and maximum rank _R_ max
**Output:** _N_ th-order tensor **X** [p] with a reduced TT rank; the cores are
rounded (reduced) according to the input tolerance _ε_ and/or ranks
bounded by _R_ max, such that } **X** ´ **X** [p] } _F_ ď _ε_ } **X** } _F_
1: Initialization **X** [p] = **X** and _δ_ = _ε_ /? _N_ ´ 1


2: **for** _n_ = 1 to _N_ ´ 1 **do**

3: QR decomposition **X** ă [(] _[n]_ 2 [)] ą [=] **[ Q]** _[n]_ **[R]** [, with] **[ X]** ă [(] _[n]_ 2 [)] ą [P] **[ R]** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_

4: Replace cores **X** ă [(] _[n]_ 2 [)] ą [=] **[ Q]** _[n]_ [ and] **[ X]** ă [(] _[n]_ 1 [+] ą [1] [)] [Ð] **[ RX]** ă [(] _[n]_ 1 [+] ą [1] [)] [, with]

**X** [(] _[n]_ [+] [1] [)]
ă1ą [P] **[ R]** _[R]_ _[n]_ [ˆ] _[I]_ _[n]_ [+] [1] _[R]_ _[n]_ [+] [1]

5: **end for**


6: **for** _n_ = _N_ to 2 **do**

7: Perform _δ_ -truncated SVD **X** [(] _[n]_ [)]
ă1ą [=] **[ U]** [ diag][t] _**[σ]**_ [u] **[V]** [T]

8: Determine minimum rank _R_ [p] _n_ ´1 such that [ř] _r_ ą _R_ _n_ ´1 _[σ]_ _r_ [2] [ď] _[ δ]_ [2] [}] _**[σ]**_ [}] [2]

9: Replace cores **X** [p] ă [(] _[n]_ 2 [´] ą [1] [)] [Ð][ p] **[X]** ă [(] _[n]_ 2 [´] ą [1] [)] **[U]** [p] [ diag][t] _**[σ]**_ [p][u][ and][ p] **[X]** ă [(] _[n]_ 1 [)] ą [=] [ p] **[V]** [T]

10: **end for**



11: **return** _N_ th-order tensor
p p ( 1 ) p ( 2 ) p ( _N_ )
**X** = xx **X**, **X**, . . ., **X**



( 1 ) p
, **X**



( 2 ) p
, . . ., **X**



( _N_ ) yy P **R** _I_ 1 ˆ _I_ 2 ˆ¨¨¨ˆ _I_ _N_,



with reduced cores **X** [p] ( _n_ ) P **R** _R_ p _n_ ´1 ˆ _I_ _n_ ˆ _R_ p _n_



typically implemented via QR/SVD with the aim to approximate, with a
desired prescribed accuracy, the TT core tensors, **G** [(] _[n]_ [)] = **X** [(] _[n]_ [)], by other core
tensors with minimum possible TT-ranks (see Algorithm 13). Note that TT
rounding is mathematically the same as the TT-SVD, but is more efficient
owing to the to use of TT format.
The complexity of TT-rounding procedures is only _O_ ( _NIR_ [3] ), since
all operations are performed in TT format which requires the SVD to
be computed only for a relatively small matricized core tensor at each
iteration. A similar approach has been developed for the HT format

[74,86,87,122].


142


**4.10** **Orthogonalization of Tensor Train Network**



The orthogonalization of core tensors is an essential procedure in many
algorithms for the TT formats [67,70,97,120,158,196,197].
For convenience, we divide a TT network, which represents a tensor
**X** p = xx **X** p ( 1 ), **X** p ( 2 ), . . ., **X** p ( _N_ ) yy P **R** _I_ 1 ˆ _I_ 2 ˆ¨¨¨ˆ _I_ _N_, into sub-trains. In this way, a



**X** p = xx **X** p ( 1 ), **X** p ( 2 ), . . ., **X** p ( _N_ ) yy P **R** _I_ 1 ˆ _I_ 2 ˆ¨¨¨ˆ _I_ _N_, into sub-trains. In this way, a

large-scale task is replaced by easier-to-handle sub-tasks, whereby the aim
is to extract a specific TT core or its slices from the whole TT network. For
this purpose, the TT sub-trains can be defined as follows



( 1 ) p
, **X**



( 2 ) p
, . . ., **X**



( 2 ) p
, . . ., **X**



( _n_ ´1 ) yy P **R** _I_ 1 ˆ _I_ 2 ˆ¨¨¨ˆ _I_ _n_ ´1 ˆ _R_ _n_ ´1 (4.36)



p ă _n_
**X** = xx **X** [p]


p ą _n_
**X** = xx **X** [p]



( 1 ) p
, **X**



( _n_ + 1 ) p
, **X**



( _n_ + 2 ) p
, . . ., **X**



( _N_ ) yy P **R** _R_ _n_ ˆ _I_ _n_ + 1 ˆ¨¨¨ˆ _I_ _N_ (4.37)



while the corresponding unfolding matrices, also called interface matrices,
are defined by


p p
**X** [ď] _[n]_ P **R** _[I]_ 1 _[I]_ 2 [¨¨¨] _[I]_ _n_ [ˆ] _[R]_ _n_, **X** [ą] _[n]_ P **R** _[R]_ _n_ [ˆ] _[I]_ _n_ + 1 [¨¨¨] _[I]_ _N_ . (4.38)


The left and right unfolding of the cores are defined as


p
**X** [(] _[n]_ [)] = **X** [p] [(] _[n]_ [)] = **X** [(] _[n]_ [)]
_L_ ă2ą [P] **[ R]** _[R]_ _[n]_ [´][1] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ [ and][ p] **[X]** [(] _R_ _[n]_ [)] ă1ą [P] **[ R]** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ _[R]_ _[n]_ [ .]


**The** _n_ **-orthogonality of tensors.** An _N_ th-order tensor in a TT format **X** [p] =



xx **X** [p]



( 1 ) p ( _N_ )
, . . ., **X** yy, is called _n_ -orthogonal with 1 ď _n_ ď _N_, if

( **X** [p] [(] _L_ _[m]_ [)] ) [T] **X** [p] [(] _L_ _[m]_ [)] = **I** _R_ _m_, _m_ = 1, . . ., _n_ ´ 1 (4.39)


p
**X** [(] _R_ _[m]_ [)] [(] **[X]** [p] [(] _R_ _[m]_ [)] [)] [T] [ =] **[ I]** _[R]_ _m_ ´1 [,] _[ m]_ [ =] _[ n]_ [ +] [ 1, . . .,] _[ N]_ [.] (4.40)



The tensor is called left-orthogonal if _n_ = _N_ and right-orthogonal if _n_ = 1.
When considering the _n_ th TT core, it is usually assumed that all cores
to the left are left-orthogonalized, and all cores to the right are rightorthogonalized. Notice that if a TT tensor [7], **X** [p], is _n_ -orthogonal then the
“left” and “right” interface matrices have orthonormal columns and rows,
that is


p p
( **X** [p] [ă] _[n]_ ) [T] [ p] **X** [ă] _[n]_ = **I** _R_ _n_ ´1, **X** [ą] _[n]_ ( **X** [ą] _[n]_ ) [T] = **I** _R_ _n_ . (4.41)


A tensor in a TT format can be orthogonalized efficiently using recursive
QR and LQ decompositions (see Algorithm 14). From the above definition,
for _n_ = _N_ the algorithms perform left-orthogonalization and for _n_ = 1
right-orthogonalization of the whole TT network.


7 By a TT-tensor we refer to as a tensor represented in the TT format.


143


**Algorithm 14** : **Left-orthogonalization, right-orthogonalization and**
_n_ **-orthogonalization of a tensor in the TT format**



( 2 ) p
, . . ., **X**



( _N_ ) yy P **R** _I_ 1 ˆ _I_ 2 ˆ¨¨¨ˆ _I_ _N_,



**Input:** _N_ th-order tensor **X** [p] = xx **X** [p]



( 1 ) p
, **X**



with TT cores **X** [p]



( _n_ ) P **R** _R_ _n_ ´1 ˆ _I_ _n_ ˆ _R_ _n_ and _R_ 0 = _R_ _N_ = 1



( _n_ ´1 )
become left-orthogonal, while the



**Output:** Cores **X** [p]



( 1 ) p
, . . ., **X**



remaining cores are right-orthogonal, except for the core **X** [p] ( _n_ )

1: **for** _m_ = 1 to _n_ ´ 1 **do**

2: Perform the QR decomposition [ **Q**, **R** ] Ð _qr_ ( **X** [p] [(] _L_ _[m]_ [)] ) for the

**X** [p] [(] _[m]_ [)] P **R** _[R]_ _[m]_ [´][1] _[I]_ _[m]_ [ˆ] _[R]_ _[m]_
unfolding cores _L_



( _m_ + 1 ) p

3: Replace the cores **X** [p] [(] _L_ _[m]_ [)] Ð **Q** and **X** [p] Ð **X**

4: **end for**


5: **for** _m_ = _N_ to _n_ + 1 **do**



( _m_ + 1 )
ˆ 1 **R**



6: Perform QR decomposition [ **Q**, **R** ] Ð _qr_ (( **X** [p] [(] _R_ _[m]_ [)] [)] [T] [)] [ for the]

unfolding cores ( **X** [p] [(] _R_ _[m]_ [)] [)] [ P] **[ R]** _[R]_ _[m]_ [´][1] [ˆ] _[I]_ _[m]_ _[R]_ _[m]_ [,]



( _m_ ´1 ) ˆ 3 **R** T



7: Replace the cores: **G** [(] _R_ _[m]_ [)] Ð **Q** [T] and **X** [p]

8: **end for**



( _m_ ´1 ) p
Ð **X**



9: **return** Left-orthogonal TT cores with ( **X** [p] [(] _L_ _[m]_ [)] ) [T] **X** [p] [(] _L_ _[m]_ [)] = **I** _R_ _m_ for

_m_ = 1, 2, . . ., _n_ ´ 1 and right-orthogonal cores **X** [p] [(] _R_ _[m]_ [)] [(] **[X]** [p] [(] _R_ _[m]_ [)] [)] [T] [ =] **[ I]** _[R]_ _m_ ´1
for _m_ = _N_, _N_ ´ 1, . . ., _n_ + 1.


**–**
**4.11** **Improved** **TT** **Decomposition** **Algorithm**
**Alternating Single Core Update (ASCU)**



Finally, we next present an efficient algorithm for TT decomposition,
referred to as the Alternating Single Core Update (ASCU), which
sequentially optimizes a single TT-core tensor while keeping the other TTcores fixed in a manner similar to the modified ALS [170].



( 1 ) p
, **X**



( _N_ )
yy is left- and right


Assume that the TT-tensor **X** [p] = xx **X** [p]



( 2 ) p
, . . ., **X**



( _k_ )
~~ă~~ 2ą [for] _[ k]_ [ =]



orthogonalized up to **X** [p]



( _n_ ) p
, i.e., the unfolding matrices **X**



1, . . ., _n_ ´ 1 have orthonormal columns, and **X** [p]



( _m_ )
( 1 ) [for] _[ m]_ [ =] _[ n]_ [ +] [ 1, . . .,] _[ N]_



have orthonormal rows. Then, the Frobenius norm of the TT-tensor **X** [p] is

equivalent to the Frobenius norm of **X** [p] ( _n_ ), that is, } **X** p} 2 _F_ [=] [}] **[X]** [p] ( _n_ ) } 2 _F_ [, so that]

the Frobenius norm of the approximation error between a data tensor **X**



( _n_ ), that is, } **X** p} 2 _F_ [=] [}] **[X]** [p]



equivalent to the Frobenius norm of **X** [p]



144


and its approximate representation in the TT format **X** [p] can be written as


_J_ ( **X** [(] _[n]_ [)] ) = } **X** ´ **X** [p] } [2] _F_ (4.42)

= } **X** } [2] _F_ [+] [}] **[X]** [p][}] [2] _F_ [´][ 2][x] **[X]** [,] **[X]** [p][y]



= } **X** } [2] _F_ [+] [}] **[X]** [p]



( _n_ ) } 2 _F_ [´][ 2][x] **[C]** [(] _[n]_ [)] [,] **[X]** [p] ( _n_ ) y



= } **X** } [2] _F_ [´ ][}] **[C]** [(] _[n]_ [)] [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [ ´] **[X]** [p] ( _n_ ) } 2 _F_ [,] _n_ = 1, . . ., _N_,



where **C** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ represents a tensor contraction of **X** and **X** [p] along
all modes but the mode- _n_, as illustrated in Figure 4.14. The **C** [(] _[n]_ [)] can be
efficiently computed through left contractions along the first ( _n_ ´ 1 ) -modes
and right contractions along the last ( _N_ ´ _m_ ) -modes, expressed as



ą _n_
. (4.43)



**L** [ă] _[n]_ = **X** [p]



ă _n_
˙ _n_ ´1 **X**, **C** [(] _[n]_ [)] = **L** [ă] _[n]_ ¸ _N_ ´ _n_ **X** [p]



The symbols ˙ _n_ and ¸ _m_ stand for the tensor contractions between two
_N_ th-order tensors along their first _n_ modes and last _m_ = _N_ ´ _n_ modes,
respectively.
The optimization problem in (4.42) is usually performed subject to the
following constraint


} **X** ´ **X** [p] } [2] _F_ [ď] _[ ε]_ [2] (4.44)


such that the TT-rank of **X** [p] is minimum.
Observe that the constraint in (4.44) for left- and right-orthogonalized
TT-cores is equivalent to the set of sub-constraints



} **C** [(] _[n]_ [)] ´ **X** [p]



( _n_ ) 2
} _F_ [ď] _[ ε]_ [2] _n_ _n_ = 1, . . ., _N_, (4.45)



whereby the _n_ th core **X** [(] _[n]_ [)] P **R** _[R]_ _[n]_ [´][1] [ˆ] _[I]_ _[n]_ [ˆ] _[R]_ _[n]_ should have minimum ranks _R_ _n_ ´1
and _R_ _n_ . Furthermore, _ε_ [2] _n_ [=] _[ ε]_ [2] [ ´ ][}] **[X]** [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [}] [2] _F_ [is assumed to be non-]
negative. Finally, we can formulate the following sequential optimization
problem


min ( _R_ _n_ ´1 ¨ _R_ _n_ ),



s.t. } **C** [(] _[n]_ [)] ´ **X** [p]



( _n_ ) 2
} _F_ [ď] _[ ε]_ [2] _n_ [,] _n_ = 1, 2, . . ., _N_ . (4.46)



By expressing the TT-core tensor **X** [p] ( _n_ ) as a TT-tensor of three factors, i.e.,

in a Tucker-2 format given by



p
**X**



( _n_ ) = **A** _n_ ˆ 1 ˜ **X** ( _n_ ) ˆ 1 **B** _n_,


145


**C** [( )] _n_










|Col1|Col2|
|---|---|
||_In_+1<br>･･･|









**X** _[<n]_ **L** [<] _[n]_ **X** [>] _[n]_


Figure 4.14: Illustration of the contraction of tensors in the Alternating
Single Core Update (ASCU) algorithm (see Algorithm 15). All the cores
to the left of **X** [(] _[n]_ [)] are left-orthogonal and all cores to its right are rightorthogonal.


the above optimization problem with the constraint (4.45) reduces to
performing a Tucker-2 decomposition (see Algorithm 8). The aim is to

compute **A** _n_, **B** _n_ (orthogonal factor matrices) and a core tensor **X** [˜] [(] _[n]_ [)] which
approximates tensor **C** [(] _[n]_ [)] with a minimum TT-rank- ( _R_ [˜] _n_ ´1, _R_ [˜] _n_ ), such that


} **C** [(] _[n]_ [)] ´ **A** _n_ ˆ [1] [ ˜] **X** [(] _[n]_ [)] ˆ [1] **B** _n_ } [2] _F_ [ď] _[ ε]_ [2] _n_ [,]


where **A** _n_ P **R** _[R]_ _[n]_ [´][1] [ˆ][ ˜] _[R]_ _[n]_ [´][1] and **B** _n_ P **R** _[R]_ [˜] _[n]_ [ˆ] _[R]_ _[n]_, with _R_ [˜] _n_ ´1 Ð _R_ _n_ ´1 and _R_ [˜] _n_ Ð _R_ _n_ .
Note that the new estimate of **X** is still of _N_ th-order because the factor



( _n_ + 1 )
as follows



matrices **A** _n_ and **B** _n_ can be embedded into **X** [p]



( _n_ ´1 ) p
and **X**



( _n_ ´1 ) ˆ 1 **A** _n_ ) ˆ 1 ˜ **X** ( _n_ ) ˆ 1 ( **B** _n_ ˆ 1 p **X**



( _n_ + 1 )
)



p p
**X** = **X**



( 1 ) ˆ 1 ¨ ¨ ¨ ˆ 1 ( **X** p



ˆ [1] ¨ ¨ ¨ ˆ [1] [ p] **X**



( _N_ )
.



In this way, the three TT-cores **X** [p] ( _n_ ´1 ), **X** p ( _n_ ) and **X** p ( _n_ + 1 ) are updated. Since

**A** _n_ and **B** [T] _n_ [have respectively orthonormal columns and rows, the newly]



( _n_ ) p
and **X**



In this way, the three TT-cores **X** [p]



( _n_ ´1 ) p
, **X**



adjusted cores ( **X** [p] ( _n_ ´1 ) ˆ 1 **A** _n_ ) and ( **B** _n_ ˆ 1 p **X** ( _n_ + 1 ) ) obey the left- and right
orthogonality conditions. Algorithm 15 outlines such a single-core update
algorithm based on the Tucker-2 decomposition. In the pseudo-code, the
left contracted tensor **L** [ă] _[n]_ is computed efficiently through a progressive



( _n_ ´1 ) ˆ 1 **A** _n_ ) and ( **B** _n_ ˆ 1 p **X**



adjusted cores ( **X** [p]



146


**Algorithm 15** : **The Alternating Single-Core Update Algorithm (two-**
**sides rank adjustment) [170]**

**Input:** Data tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and approximation accuracy _ε_



TT-tensor **X** [p] = **X** [p] ( 1 ) ˆ 1 p **X** ( 2 ) ˆ 1 ¨ ¨ ¨ ˆ 1 p **X** ( _N_ ) of minimum

TT-rank such that } **X** ´ **X** [p] } [2] _F_ [ď] _[ ε]_ [2]



( 1 ) ˆ 1 p **X**



**Output:** TT-tensor **X** [p] = **X** [p]



( 2 ) ˆ 1 ¨ ¨ ¨ ˆ 1 p **X**



( 1 ) p ( 2 ) p ( _N_ )

1: Initialize **X** [p] = xx **X** [p], **X**, . . ., **X** yy

2: **repeat**
3: **for** _n_ = 1, 2, . . ., _N_ ´ 1 **do**



( 1 ) p
, **X**



1: Initialize **X** [p] = xx **X** [p]



( 2 ) p
, . . ., **X**



4: Compute contracted tensor **C** [(] _[n]_ [)] = **L** [ă] _[n]_ ¸ _N_ ´ _n_ **X** [p]

5: Solve a Tucker-2 decomposition



ą _n_



} **C** [(] _[n]_ [)] ´ **A** _n_ ˆ [1] [ p] **X** ( _n_ ) ˆ 1 **B** _n_ } 2 _F_ [ď] _[ ε]_ [2] [ ´ ][}] **[X]** [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [}] [2] _F_

6: Adjust adjacent cores
**X** p ( _n_ ´ 1 ) Ð **X** p ( _n_ ´ 1 ) ˆ 1 **A** **X** p ( _n_ + 1 ) Ð **B** 1 p **X** ( _n_



} **C** [(] _[n]_ [)] ´ **A** _n_ ˆ [1] [ p] **X**



( _n_ ´ 1 ) p
Ð **X**



( _n_ ´ 1 ) ˆ 1 **A** _n_, **X** p



( _n_ + 1 ) Ð **B** _n_ ˆ 1 p **X**



( _n_ + 1 )



7: Perform left-orthogonalization of **X** [p]

8: Update left-side contracted tensors



( _n_ )



**L** [ă] _[n]_ Ð **A** [T] _n_ [ˆ] [1] **[ L]** [ă] _[n]_ [,] **L** [ă] [(] _[n]_ [+] [1] [)] Ð **X** [p] ( _n_ ) ˙ 2 **L** ă _n_

9: **end for**

10: **for** _n_ = _N_, _N_ ´ 1, . . ., 2 **do**

11: Compute contracted tensor **C** [(] _[n]_ [)] = **L** [ă] _[n]_ ¸ _N_ ´ _n_ **X** [p]

12: Solve a constrained Tucker-2 decomposition



ą _n_



} **C** [(] _[n]_ [)] ´ **A** _n_ ˆ [1] [ p] **X**



( _n_ ) ˆ 1 **B** _n_ } 2 _F_ [ď] _[ ε]_ [2] [ ´ ][}] **[X]** [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [}] [2] _F_



( _n_ + 1 ) Ð **B** _n_ ˆ 1 p **X**



( _n_ + 1 )



p
13: **X**



( _n_ ´ 1 ) p
Ð **X**



( _n_ ´ 1 ) ˆ 1 **A** _n_, **X** p



14: Perform right-orthogonalization of **X** [p] ( _n_ )

15: **end for**
16: **until** a stopping criterion is met



( 2 ) p
, . . ., **X**



( _N_ )
yy.



17: **return** xx **X** [p]



( 1 ) p
, **X**



contraction in the form [101,182]


**L** [ă] _[n]_ = **X** p ( _n_ ´1 ) ˙ 2 **L** ă ( _n_ ´1 ), (4.47)



where **L** [ă][1] = **X** .
Alternatively, instead of adjusting the two TT ranks, _R_ _n_ ´1 and _R_ _n_, of
p ( _n_ )
**X**, we can update only one rank, either _R_ _n_ ´1 or _R_ _n_, corresponding to the

right-to-left or left-to-right update order procedure. Assuming that the core



tensors are updated in the left-to-right order, we need to find **X** [p]


147



( _n_ )
which


**Algorithm 16** : **The Alternating Single-Core Update Algorithm (one-**
**side rank adjustment) [170]**

**Input:** Data tensor **X** P **R** _[I]_ [1] [ˆ] _[I]_ [2] [ˆ¨¨¨ˆ] _[I]_ _[N]_ and approximation accuracy _ε_



TT-tensor **X** [p] = **X** [p] ( 1 ) ˆ 1 p **X** ( 2 ) ˆ 1 ¨ ¨ ¨ ˆ 1 p **X** ( _N_ ) of minimum

TT-rank such that } **X** ´ **X** [p] } [2] _F_ [ď] _[ ε]_ [2]



( 2 ) ˆ 1 ¨ ¨ ¨ ˆ 1 p **X**



**Output:** TT-tensor **X** [p] = **X** [p]



( 1 ) ˆ 1 p **X**



1: Initialize TT-cores **X** [p] ( _n_ ), @ _n_

2: **repeat**
3: **for** _n_ = 1, 2, . . ., _N_ ´ 1 **do**

4: Compute the contracted tensor **C** [(] _[n]_ [)] = **L** [ă] _[n]_ ¸ _N_ ´ _n_ **X** [p]

5: Truncated SVD:
} [ **C** [(] _[n]_ [)] ] ă 2 ą ´ **U Σ V** [T] } [2] _F_ [ď] _[ ε]_ [2] [ ´ ][}] **[X]** [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [}] [2] _F_



ą _n_



6: Update **X** [p]



( _n_ )
= reshape ( **U**, _R_ _n_ ´ 1 ˆ _I_ _n_ ˆ _R_ _n_ )



( _n_ + 1 )



7: Adjust adjacent core **X** [p]



( _n_ + 1 ) Ð ( **Σ V** T ) ˆ 1 p **X**



8: Update left-side contracted tensors



**L** [ă] [(] _[n]_ [+] [1] [)] Ð **X** [p]



( _n_ ) ˙ 2 **L** ă _n_



9: **end for**

10: **for** _n_ = _N_, _N_ ´ 1, . . ., 2 **do**

11: Compute contracted tensor **C** [(] _[n]_ [)] = **L** [ă] _[n]_ ¸ _N_ ´ _n_ **X** [p]

12: Truncated SVD:
} [ **C** [(] _[n]_ [)] ] ( 1 ) ´ **U Σ V** [T] } [2] _F_ [ď] _[ ε]_ [2] [ ´ ][}] **[X]** [}] [2] _F_ [+] [}] **[C]** [(] _[n]_ [)] [}] [2] _F_ [;]



ą _n_



13: **X** p ( _n_ ) = reshape ( **V** T, _R_ _n_ ´ 1 ˆ _I_ _n_ ˆ _R_ _n_ )



14: **X** p ( _n_ ´ 1 ) Ð **X** p ( _n_ ´ 1 ) ˆ 1 ( **U Σ** )

15: **end for**
16: **until** a stopping criterion is met



( _n_ ´ 1 ) p
Ð **X**



p
14: **X**



( 2 ) p
, . . ., **X**



( _N_ )
yy.



17: **return** xx **X** [p]



( 1 ) p
, **X**



has a minimum rank- _R_ _n_ and satisfies the constraints



} **C** [(] _[n]_ [)] ´ **X** [p] ( _n_ ) ˆ 1 **B** _n_ } 2 _F_ [ď] _[ ε]_ [2] _n_ [,] _n_ = 1, . . ., _N_ .



This problem reduces to the truncated SVD of the mode-t1, 2u matricization
of **C** [(] _[n]_ [)] with an accuracy _ε_ [2] _n_ [, that is]


[ **C** [(] _[n]_ [)] ] ă2ą « **U** _n_ **Σ V** [T] _n_ [,]


where **Σ** = diag ( _σ_ _n_,1, . . ., _σ_ _n_, _R_ ‹ _n_ ) . Here, for the new optimized rank _R_ [‹] _n_ [, the]
following holds



_R_ [‹] _n_
ÿ _σ_ _n_ [2], _r_ [ě ][}] **[X]** [}] [2] _F_ [´] _[ ε]_ [2] [ ą]

_r_ = 1


148



_R_ [‹] _n_ [´][1]
ÿ _σ_ _n_ [2], _r_ [.] (4.48)

_r_ = 1


The core tensor **X** [p]



( _n_ )
is then updated by reshaping **U** _n_ to an order-3 tensor of



size _R_ _n_ ´1 ˆ _I_ _n_ ˆ _R_ [‹] _n_ [, while the core] **[X]** [p] ( _n_ + 1 ) needs to be adjusted accordingly


as



p
**X**



( _n_ + 1 ) ‹ = **Σ V** T _n_ [ˆ] [1] [ p] **[X]** ( _n_ + 1 ) . (4.49)



When the algorithm updates the core tensors in the right-to-left order, we



update **X** [p]



( _n_ ) by using the _R_ ‹ _n_ ´1 [leading right singular vectors of the mode-1]



matricization of **C** [(] _[n]_ [)], and adjust the core **X** [p] ( _n_ ´1 ) accordingly, that is,


[ **C** [(] _[n]_ [)] ] ( 1 ) – **U** _n_ **Σ V** [T] _n_



p
**X**



( _n_ ) ‹ = reshape ( **V** T _n_ [,] [ [] _[R]_ [‹] _n_ ´1 [,] _[ I]_ _[n]_ [,] _[ R]_ _[n]_ [])]



( _n_ ´1 ) ˆ 1 ( **U** _n_ **Σ** ) . (4.50)



p
**X**



( _n_ ´1 ) ‹ p
= **X**



To summarise, the ASCU method performs a sequential update of one core
and adjusts (or rotates) another core. Hence, it updates two cores at a time
(for detail see Algorithm 16).
The ASCU algorithm can be implemented in an even more efficient way,
if the data tensor **X** is already given in a TT format (with a non-optimal
TT ranks for the prescribed accuracy). Detailed MATLAB implementations
and other variants of the TT decomposition algorithm are provided in [170].


149


**Chapter 5**


**Discussion and Conclusions**


In Part 1 of this monograph, we have provided a systematic and
example-rich guide to the basic properties and applications of tensor
network methodologies, and have demonstrated their promise as a tool
for the analysis of extreme-scale multidimensional data. Our main aim
has been to illustrate that, owing to the intrinsic compression ability
that stems from the distributed way in which they represent data and
process information, TNs can be naturally employed for linear/multilinear
dimensionality reduction. Indeed, current applications of TNs include
generalized multivariate regression, compressed sensing, multi-way blind
source separation, sparse representation and coding, feature extraction,
classification, clustering and data fusion.
With multilinear algebra as their mathematical backbone, TNs have
been shown to have intrinsic advantages over the flat two-dimensional
view provided by matrices, including the ability to model both strong and
weak couplings among multiple variables, and to cater for multimodal,
incomplete and noisy data.
In Part 2 of this monograph we introduce a scalable framework
for distributed implementation of optimization algorithms, in order
to transform huge-scale optimization problems into linked small-scale
optimization sub-problems of the same type. In that sense, TNs can be seen
as a natural bridge between small-scale and very large-scale optimization
paradigms, which allows for any efficient standard numerical algorithm to
be applied to such local optimization sub-problems.
Although research on tensor networks for dimensionality reduction
and optimization problems is only emerging, given that in many modern
applications, multiway arrays (tensors) arise, either explicitly or indirectly,


150


through the tensorization of vectors and matrices, we foresee this material
serving as a useful foundation for further studies on a variety of machine
learning problems for data of otherwise prohibitively large volume, variety,
or veracity. We also hope that the readers will find the approaches
presented in this monograph helpful in advancing seamlessly from
numerical linear algebra to numerical multilinear algebra.


151


**Bibliography**


[1] E. Acar and B. Yener. Unsupervised multiway data analysis:
A literature survey. _IEEE Transactions on Knowledge and Data_
_Engineering_, 21:6–20, 2009.


[2] I. Affleck, T. Kennedy, E.H. Lieb, and H. Tasaki. Rigorous results
on valence-bond ground states in antiferromagnets. _Physical Review_
_Letters_, 59(7):799, 1987.


[3] A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade, and M. Telgarsky.
Tensor decompositions for learning latent variable models. _Journal_
_of Machine Learning Research_, 15:2773–2832, 2014.


[4] D. Anderson, S. Du, M. Mahoney, C. Melgaard, K. Wu, and M. Gu.
Spectral gap error bounds for improving CUR matrix decomposition
and the Nystr¨om method. In _Proceedings of the 18th International_
_Conference on Artificial Intelligence and Statistics_, pages 19–27, 2015.


[5] W. Austin, G. Ballard, and T.G. Kolda. Parallel tensor compression
for large-scale scientific data. _[arXiv preprint arXiv:1510.06689](http://arxiv.org/abs/1510.06689)_, 2015.


[6] F.R. Bach and M.I. Jordan. Kernel independent component analysis.
_The Journal of Machine Learning Research_, 3:1–48, 2003.


[7] M. Bachmayr, R. Schneider, and A. Uschmajew. Tensor networks
and hierarchical tensors for the solution of high-dimensional partial
differential equations. _Foundations of Computational Mathematics_,
16(6):1423–1472, 2016.


[8] B.W. Bader and T.G. Kolda. MATLAB tensor toolbox version 2.6,
February 2015.


152


[9] J. Ballani and L. Grasedyck. Tree adaptive approximation in the
hierarchical tensor format. _SIAM Journal on Scientific Computing_,
36(4):A1415–A1431, 2014.


[10] J. Ballani, L. Grasedyck, and M. Kluge. A review on adaptive lowrank approximation techniques in the hierarchical tensor format. In
_Extraction of Quantifiable Information from Complex Systems_, pages 195–
210. Springer, 2014.


[11] G. Ballard, A.R. Benson, A. Druinsky, B. Lipshitz, and O. Schwartz.
Improving the numerical stability of fast matrix multiplication
algorithms. _[arXiv preprint arXiv:1507.00687](http://arxiv.org/abs/1507.00687)_, 2015.


[12] G. Ballard, A. Druinsky, N. Knight, and O. Schwartz. Brief
announcement: Hypergraph partitioning for parallel sparse matrixmatrix multiplication. In _Proceedings of the 27th ACM on Symposium on_
_Parallelism in Algorithms and Architectures_, pages 86–88. ACM, 2015.


[13] G. Barcza, ¨O. Legeza, K.H. Marti, and M. Reiher. Quantuminformation analysis of electronic states of different molecular
structures. _Physical Review A_, 83(1):012508, 2011.


[14] K. Batselier, H. Liu, and N. Wong. A constructive algorithm for
decomposing a tensor into a finite sum of orthonormal rank-1 terms.
_SIAM Journal on Matrix Analysis and Applications_, 36(3):1315–1337,
2015.


[15] K. Batselier and N. Wong. A constructive arbitrary-degree Kronecker
product decomposition of tensors. _[arXiv preprint arXiv:1507.08805](http://arxiv.org/abs/1507.08805)_,
2015.


[16] M. Bebendorf. Adaptive cross-approximation of multivariate
functions. _Constructive Approximation_, 34(2):149–179, 2011.


[17] M. Bebendorf, C. Kuske, and R. Venn. Wideband nested cross
approximation for Helmholtz problems. _Numerische Mathematik_,
130(1):1–34, 2015.


[18] R.E. Bellman. _Adaptive Control Processes_ . Princeton University Press,
Princeton, NJ, 1961.


[19] P. Benner, V. Khoromskaia, and B.N. Khoromskij. A reduced basis
approach for calculation of the Bethe–Salpeter excitation energies


153


by using low-rank tensor factorisations. _Molecular Physics_, 114(78):1148–1161, 2016.


[20] A.R. Benson, J.D. Lee, B. Rajwa, and D.F. Gleich. Scalable methods for
nonnegative matrix factorizations of near-separable tall-and-skinny
matrices. In _Proceedings of Neural Information Processing Systems_
_(NIPS)_, pages 945–953, 2014.


[21] D. Bini. Tensor and border rank of certain classes of matrices and
the fast evaluation of determinant inverse matrix and eigenvalues.
_Calcolo_, 22(1):209–228, 1985.


[22] M. Bolten, K. Kahl, and S. Sokolovi´c. Multigrid Methods for Tensor
Structured Markov Chains with Low Rank Approximation. _SIAM_
_Journal on Scientific Computing_, 38(2):A649–A667, 2016.


[23] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed
optimization and statistical learning via the alternating direction
method of multipliers. _Foundations and Trends in Machine Learning_,
3(1):1–122, 2011.


[24] A. Bruckstein, D. Donoho, and M. Elad. From sparse solutions of
systems of equations to sparse modeling of signals and images. _SIAM_
_Review_, 51(1):34–81, 2009.


[25] H.-J. Bungartz and M. Griebel. Sparse grids. _Acta Numerica_, 13:147–
269, 2004.


[26] C. Caiafa and A. Cichocki. Generalizing the column-row matrix
decomposition to multi-way arrays. _Linear Algebra and its_
_Applications_, 433(3):557–573, 2010.


[27] C. Caiafa and A Cichocki. Computing sparse representations of
multidimensional signals using Kronecker bases. _Neural Computaion_,
25(1):186–220, 2013.


[28] C. Caiafa and A. Cichocki. Stable, robust, and super–fast
reconstruction of tensors using multi-way projections. _IEEE_
_Transactions on Signal Processing_, 63(3):780–793, 2015.


[29] J.D. Carroll and J.-J. Chang. Analysis of individual differences in
multidimensional scaling via an N-way generalization of ”EckartYoung” decomposition. _Psychometrika_, 35(3):283–319, 1970.


154


[30] V. Cevher, S. Becker, and M. Schmidt. Convex optimization for big
data: Scalable, randomized, and parallel algorithms for big data
analytics. _IEEE Signal Processing Magazine_, 31(5):32–43, 2014.


[31] G. Chabriel, M. Kleinsteuber, E. Moreau, H. Shen, P. Tichavsky, and
A. Yeredor. Joint matrix decompositions and blind source separation:
A survey of methods, identification, and applications. _IEEE Signal_
_Processing Magazine_, 31(3):34–43, 2014.


[32] V. Chandola, A. Banerjee, and V. Kumar. Anomaly detection: A
survey. _ACM Computing Surveys (CSUR)_, 41(3):15, 2009.


[33] T.-L. Chen, D. D. Chang, S.-Y. Huang, H. Chen, C. Lin, and
W. Wang. Integrating multiple random sketches for singular value
decomposition. _arXiv e-prints_, 2016.


[34] H. Cho, D. Venturi, and G.E. Karniadakis. Numerical methods for
high-dimensional probability density function equations. _Journal of_
_Computational Physics_, 305:817–837, 2016.


[35] J.H. Choi and S. Vishwanathan. DFacTo: Distributed factorization of
tensors. In _Advances in Neural Information Processing Systems_, pages
1296–1304, 2014.


[36] W. Chu and Z. Ghahramani. Probabilistic models for incomplete
multi-dimensional arrays. In _JMLR Workshop and Conference_
_Proceedings Volume 5:_ _AISTATS 2009_, volume 5, pages 89–
96. Microtome Publishing (paper) Journal of Machine Learning
Research, 2009.


[37] A. Cichocki. Era of big data processing: A new approach via tensor
networks and tensor decompositions, (invited). In _Proceedings of the_
_International Workshop on Smart Info-Media Systems in Asia (SISA2013)_,
September 2013.


[38] A. Cichocki. Tensor decompositions: A new concept in brain data
analysis? _[arXiv preprint arXiv:1305.0395](http://arxiv.org/abs/1305.0395)_, 2013.


[39] A. Cichocki. Tensor networks for big data analytics and large-scale
optimization problems. _[arXiv preprint arXiv:1407.3124](http://arxiv.org/abs/1407.3124)_, 2014.


[40] A. Cichocki and S. Amari. _Adaptive Blind Signal and Image Processing:_
_Learning Algorithms and Applications_ . John Wiley & Sons, Ltd, 2003.


155


[41] A. Cichocki, S. Cruces, and S. Amari. Log-determinant divergences
revisited: Alpha-beta and gamma log-det divergences. _Entropy_,
17(5):2988–3034, 2015.


[42] A. Cichocki, D. Mandic, C. Caiafa, A.H. Phan, G. Zhou, Q. Zhao,
and L. De Lathauwer. Tensor decompositions for signal processing
applications: From two-way to multiway component analysis. _IEEE_
_Signal Processing Magazine_, 32(2):145–163, 2015.


[43] A. Cichocki, R Zdunek, A.-H. Phan, and S. Amari. _Nonnegative Matrix_
_and Tensor Factorizations: Applications to Exploratory Multi-way Data_
_Analysis and Blind Source Separation_ . Wiley, Chichester, 2009.


[44] N. Cohen, O. Sharir, and A. Shashua. On the expressive power
of deep learning: A tensor analysis. In _29th Annual Conference on_
_Learning Theory_, pages 698–728, 2016.


[45] N. Cohen and A. Shashua. Convolutional rectifier networks as
generalized tensor decompositions. In _Proceedings of The 33rd_
_International Conference on Machine Learning_, pages 955–963, 2016.


[46] P. Comon. Tensors: a brief introduction. _IEEE Signal Processing_
_Magazine_, 31(3):44–53, 2014.


[47] P. Comon and C. Jutten. _Handbook of Blind Source Separation:_
_Independent Component Analysis and Applications_ . Academic Press,
2010.


[48] P.G. Constantine and D.F. Gleich. Tall and skinny QR factorizations
in MapReduce architectures. In _Proceedings of the Second International_
_Workshop on MapReduce and its Applications_, pages 43–50. ACM, 2011.


[49] P.G. Constantine, D.F Gleich, Y. Hou, and J. Templeton. Model
reduction with MapReduce-enabled tall and skinny singular value
decomposition. _SIAM Journal on Scientific Computing_, 36(5):S166–
S191, 2014.


[50] E. Corona, A. Rahimian, and D. Zorin. A Tensor-Train accelerated
solver for integral equations in complex geometries. _arXiv preprint_
_[arXiv:1511.06029](http://arxiv.org/abs/1511.06029)_, November 2015.


[51] C. Crainiceanu, B. Caffo, S. Luo, V. Zipunnikov, and N. Punjabi.
Population value decomposition, a framework for the analysis of


156


image populations. _Journal of the American Statistical Association_,
106(495):775–790, 2011.


[52] A. Critch and J. Morton. Algebraic geometry of matrix product
states. _Symmetry, Integrability and Geometry: Methods and Applications_
_(SIGMA)_, 10:095, 2014.


[53] A.J. Critch. _Algebraic Geometry of Hidden Markov and Related Models_ .
PhD thesis, University of California, Berkeley, 2013.


[54] A.L.F. de Almeida, G. Favier, J.C.M. Mota, and J.P.C.L. da Costa.
Overview of tensor decompositions with applications to
communications. In R.F. Coelho, V.H. Nascimento, R.L. de Queiroz,
J.M.T. Romano, and C.C. Cavalcante, editors, _Signals and Images:_
_Advances and Results in Speech, Estimation, Compression, Recognition,_
_Filtering, and Processing_, chapter 12, pages 325–355. CRC Press, 2015.


[55] F. De la Torre. A least-squares framework for component analysis.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_,
34(6):1041–1055, 2012.


[56] L. De Lathauwer. A link between the canonical decomposition in
multilinear algebra and simultaneous matrix diagonalization. _SIAM_
_Journal on Matrix Analysis and Applications_, 28:642–666, 2006.


[57] L. De Lathauwer. Decompositions of a higher-order tensor in
block terms — Part I and II. _SIAM Journal on Matrix Analysis_
_and Applications_, 30(3):1022–1066, 2008. Special Issue on Tensor
Decompositions and Applications.


[58] L. De Lathauwer. Blind separation of exponential polynomials and
the decomposition of a tensor in rank- ( _L_ _r_, _L_ _r_, 1 ) terms. _SIAM Journal_
_on Matrix Analysis and Applications_, 32(4):1451–1474, 2011.


[59] L. De Lathauwer, B. De Moor, and J. Vandewalle. A Multilinear
Singular Value Decomposition. _SIAM Journal on Matrix Analysis_
_Applications_, 21:1253–1278, 2000.


[60] L. De Lathauwer, B. De Moor, and J. Vandewalle. On the best rank1 and rank- ( _R_ 1, _R_ 2, ..., _R_ _N_ ) approximation of higher-order tensors.
_SIAM Journal of Matrix Analysis and Applications_, 21(4):1324–1342,
2000.


157


[61] L. De Lathauwer and D. Nion. Decompositions of a higher-order
tensor in block terms – Part III: Alternating least squares algorithms.
_SIAM Journal on Matrix Analysis and Applications_, 30(3):1067–1083,
2008.


[62] W. de Launey and J. Seberry. The strong Kronecker product. _Journal_
_of Combinatorial Theory, Series A_, 66(2):192–213, 1994.


[63] V. de Silva and L.-H. Lim. Tensor rank and the ill-posedness of
the best low-rank approximation problem. _SIAM Journal on Matrix_
_Analysis and Applications_, 30:1084–1127, 2008.


[64] A. Desai, M. Ghashami, and J.M. Phillips. Improved practical matrix
sketching with guarantees. _IEEE Transactions on Knowledge and Data_
_Engineering_, 28(7):1678–1690, 2016.


[65] I.S. Dhillon. Fast Newton-type methods for nonnegative matrix and
tensor approximation. The NSF Workshop, Future Directions in
Tensor-Based Computation and Modeling, 2009.


[66] E. Di Napoli, D. Fabregat-Traver, G. Quintana-Ort´ı, and P. Bientinesi.
Towards an efficient use of the BLAS library for multilinear tensor
contractions. _Applied Mathematics and Computation_, 235:454–468, 2014.


[67] S.V. Dolgov. _Tensor Product Methods in Numerical Simulation of High-_
_dimensional Dynamical Problems_ . PhD thesis, Faculty of Mathematics
and Informatics, University Leipzig, Germany, Leipzig, Germany,
2014.


[68] S.V. Dolgov and B.N. Khoromskij. Two-level QTT-Tucker format
for optimized tensor calculus. _SIAM Journal on Matrix Analysis and_
_Applications_, 34(2):593–623, 2013.


[69] S.V. Dolgov and B.N. Khoromskij. Simultaneous state-time
approximation of the chemical master equation using tensor product
formats. _Numerical Linear Algebra with Applications_, 22(2):197–219,
2015.


[70] S.V. Dolgov, B.N. Khoromskij, I.V. Oseledets, and D.V. Savostyanov.
Computation of extreme eigenvalues in higher dimensions using
block tensor train format. _Computer Physics Communications_,
185(4):1207–1216, 2014.


158


[71] S.V. Dolgov and D.V. Savostyanov. Alternating minimal energy
methods for linear systems in higher dimensions. _SIAM Journal on_
_Scientific Computing_, 36(5):A2248–A2271, 2014.


[72] P. Drineas and M.W. Mahoney. A randomized algorithm for a tensorbased generalization of the singular value decomposition. _Linear_
_Algebra and its Applications_, 420(2):553–571, 2007.


[73] G. Ehlers, J. S´olyom, O. Legeza, and R.M. Noack. [¨] Entanglement
structure of the hubbard model in momentum space. _Physical Review_
_B_, 92(23):235116, 2015.


[74] M Espig, M Schuster, A Killaitis, N Waldren, P W¨ahnert,
S Handschuh, and H Auer. TensorCalculus library, 2012.


[75] F. Esposito, T. Scarabino, A. Hyv¨arinen, J. Himberg, E. Formisano,
S. Comani, G. Tedeschi, R. Goebel, E. Seifritz, and F. Di Salle.
Independent component analysis of fMRI group studies by selforganizing clustering. _NeuroImage_, 25(1):193–205, 2005.


[76] G. Evenbly and G. Vidal. Algorithms for entanglement
renormalization. _Physical Review B_, 79(14):144108, 2009.


[77] G. Evenbly and S. R. White. Entanglement Renormalization and
Wavelets. _Physical Review Letters_, 116(14):140403, 2016.


[78] H. Fanaee-T and J. Gama. Tensor-based anomaly detection: An
interdisciplinary survey. _Knowledge-Based Systems_, 2016.


[79] G. Favier and A. de Almeida. Overview of constrained PARAFAC
models. _EURASIP Journal on Advances in Signal Processing_, 2014(1):1–
25, 2014.


[80] J. Garcke, M. Griebel, and M. Thess. Data mining with sparse grids.
_Computing_, 67(3):225–253, 2001.


[81] S. Garreis and M. Ulbrich. Constrained optimization with low-rank
tensors and applications to parametric problems with PDEs. _SIAM_
_Journal on Scientific Computing_, (accepted), 2016.


[82] M. Ghashami, E. Liberty, and J.M. Phillips. Efficient frequent
directions algorithm for sparse matrices. _arXiv_ _preprint_
_[arXiv:1602.00412](http://arxiv.org/abs/1602.00412)_, 2016.


159


[83] V. Giovannetti, S. Montangero, and R. Fazio. Quantum multiscale
entanglement renormalization ansatz channels. _Physical Review_
_Letters_, 101(18):180503, 2008.


[84] S.A. Goreinov, E.E. Tyrtyshnikov, and N.L. Zamarashkin. A theory of
pseudo-skeleton approximations. _Linear Algebra and its Applications_,
261:1–21, 1997.


[85] S.A. Goreinov, N.L. Zamarashkin, and E.E. Tyrtyshnikov. Pseudoskeleton approximations by matrices of maximum volume.
_Mathematical Notes_, 62(4):515–519, 1997.


[86] L. Grasedyck. Hierarchical singular value decomposition of tensors.
_SIAM Journal on Matrix Analysis and Applications_, 31(4):2029–2054,
2010.


[87] L. Grasedyck, D. Kessner, and C. Tobler. A literature survey of lowrank tensor approximation techniques. _GAMM-Mitteilungen_, 36:53–
78, 2013.


[88] A.R. Groves, C.F. Beckmann, S.M. Smith, and M.W. Woolrich.
Linked independent component analysis for multimodal data fusion.
_NeuroImage_, 54(1):2198 – 21217, 2011.


[89] Z.-C. Gu, M. Levin, B. Swingle, and X.-G. Wen. Tensor-product
representations for string-net condensed states. _Physical Review B_,
79(8):085118, 2009.


[90] M. Haardt, F. Roemer, and G. Del Galdo. Higher-order SVD based
subspace estimation to improve the parameter estimation accuracy
in multi-dimensional harmonic retrieval problems. _IEEE Transactions_
_on Signal Processing_, 56:3198–3213, July 2008.


[91] W. Hackbusch. _Tensor Spaces and Numerical Tensor Calculus_, volume 42
of _Springer Series in Computational Mathematics_ . Springer, Heidelberg,
2012.


[92] W. Hackbusch and S. K¨uhn. A new scheme for the tensor
representation. _Journal of Fourier Analysis and Applications_, 15(5):706–
722, 2009.


[93] N. Halko, P. Martinsson, and J. Tropp. Finding structure with
randomness: Probabilistic algorithms for constructing approximate
matrix decompositions. _SIAM Review_, 53(2):217–288, 2011.


160


[94] S. Handschuh. _Numerical Methods in Tensor Networks_ . PhD
thesis, Facualty of Mathematics and Informatics, University Leipzig,
Germany, Leipzig, Germany, 2015.


[95] R.A. Harshman. Foundations of the PARAFAC procedure: Models
and conditions for an explanatory multimodal factor analysis. _UCLA_
_Working Papers in Phonetics_, 16:1–84, 1970.


[96] F.L. Hitchcock. Multiple invariants and generalized rank of a _p_ -way
matrix or tensor. _Journal of Mathematics and Physics_, 7:39–79, 1927.


[97] S. Holtz, T. Rohwedder, and R. Schneider. The alternating linear
scheme for tensor optimization in the tensor train format. _SIAM_
_Journal on Scientific Computing_, 34(2), 2012.


[98] M. Hong, M. Razaviyayn, Z.Q. Luo, and J.S. Pang. A unified
algorithmic framework for block-structured optimization involving
big data with applications in machine learning and signal processing.
_IEEE Signal Processing Magazine_, 33(1):57–77, 2016.


[99] H. Huang, C. Ding, D. Luo, and T. Li. Simultaneous tensor
subspace selection and clustering: The equivalence of high order
SVD and K-means clustering. In _Proceedings of the 14th ACM SIGKDD_
_International Conference on Knowledge Discovery and Data mining_, pages
327–335. ACM, 2008.


[100] R. H¨ubener, V. Nebendahl, and W. D¨ur. Concatenated tensor network
states. _New Journal of Physics_, 12(2):025004, 2010.


[101] C. Hubig, I.P. McCulloch, U. Schollw¨ock, and F.A. Wolf. Strictly
single-site DMRG algorithm with subspace expansion. _Physical_
_Review B_, 91(15):155115, 2015.


[102] T. Huckle, K. Waldherr, and T. Schulte-Herbriggen. Computations
in quantum tensor networks. _Linear Algebra and its Applications_,
438(2):750 – 781, 2013.


[103] A. Hyv¨arinen. Independent component analysis: Recent advances.
_Philosophical Transactions of the Royal Society A_, 371(1984):20110534,
2013.


[104] I. Jeon, E.E. Papalexakis, C. Faloutsos, L. Sael, and U. Kang. Mining
billion-scale tensors: Algorithms and discoveries. _The VLDB Journal_,
pages 1–26, 2016.


161


[105] B. Jiang, F. Yang, and S. Zhang. Tensor and its Tucker core: The
invariance relationships. _[arXiv e-prints arXiv:1601.01469](http://arxiv.org/abs/1601.01469)_, January
2016.


[106] U. Kang, E.E. Papalexakis, A. Harpale, and C. Faloutsos. GigaTensor:
Scaling tensor analysis up by 100 times - algorithms and discoveries.
In _Proceedings of the 18th ACM SIGKDD International Conference on_
_Knowledge Discovery and Data Mining (KDD ’12)_, pages 316–324,
August 2012.


[107] Y.-J. Kao, Y.-D. Hsieh, and P. Chen. Uni10: An open-source library
for tensor network algorithms. In _Journal of Physics: Conference Series_,
volume 640, page 012040. IOP Publishing, 2015.


[108] L. Karlsson, D. Kressner, and A. Uschmajew. Parallel algorithms for
tensor completion in the CP format. _Parallel Computing_, 57:222–234,
2016.


[109] J.-P. Kauppi, J. Hahne, K.R. M¨uller, and A. Hyv¨arinen. Three-way
analysis of spectrospatial electromyography data: Classification and
interpretation. _PloS One_, 10(6):e0127231, 2015.


[110] V.A. Kazeev, M. Khammash, M. Nip, and C. Schwab. Direct solution
of the chemical master equation using quantized tensor trains. _PLoS_
_Computational Biology_, 10(3):e1003359, 2014.


[111] V.A. Kazeev and B.N. Khoromskij. Low-rank explicit QTT
representation of the Laplace operator and its inverse. _SIAM Journal_
_on Matrix Analysis and Applications_, 33(3):742–758, 2012.


[112] V.A. Kazeev, B.N. Khoromskij, and E.E. Tyrtyshnikov. Multilevel
Toeplitz matrices generated by tensor-structured vectors and
convolution with logarithmic complexity. _SIAM Journal on Scientific_
_Computing_, 35(3):A1511–A1536, 2013.


[113] V.A. Kazeev, O. Reichmann, and C. Schwab. Low-rank tensor
structure of linear diffusion operators in the TT and QTT formats.
_Linear Algebra and its Applications_, 438(11):4204–4221, 2013.


[114] B.N. Khoromskij. _O_ ( _d_ log _N_ ) -quantics approximation of _N_ - _d_
tensors in high-dimensional numerical modeling. _Constructive_
_Approximation_, 34(2):257–280, 2011.


162


[115] B.N. Khoromskij. Tensors-structured numerical methods in scientific
computing: Survey on recent advances. _Chemometrics and Intelligent_
_Laboratory Systems_, 110(1):1–19, 2011.


[116] B.N. Khoromskij and A. Veit. Efficient computation of highly
oscillatory integrals by using QTT tensor approximation.
_Computational Methods in Applied Mathematics_, 16(1):145–159, 2016.


[117] H.-J. Kim, E. Ollila, V. Koivunen, and H.V. Poor. Robust iteratively
reweighted Lasso for sparse tensor factorizations. In _IEEE Workshop_
_on Statistical Signal Processing (SSP)_, pages 420–423, 2014.


[118] S. Klus and C. Sch¨utte. Towards tensor-based methods for the
numerical approximation of the Perron-Frobenius and Koopman
operator. _[arXiv e-prints arXiv:1512.06527](http://arxiv.org/abs/1512.06527)_, December 2015.


[119] T.G. Kolda and B.W. Bader. Tensor decompositions and applications.
_SIAM Review_, 51(3):455–500, 2009.


[120] D. Kressner, M. Steinlechner, and A. Uschmajew. Low-rank
tensor methods with subspace correction for symmetric eigenvalue
problems. _SIAM Journal on Scientific Computing_, 36(5):A2346–A2368,
2014.


[121] D. Kressner, M. Steinlechner, and B. Vandereycken. Low-rank tensor
completion by Riemannian optimization. _BIT Numerical Mathematics_,
54(2):447–468, 2014.


[122] D. Kressner and C. Tobler. Algorithm 941: HTucker–A MATLAB
toolbox for tensors in hierarchical Tucker format. _ACM Transactions_
_on Mathematical Software_, 40(3):22, 2014.


[123] D. Kressner and A. Uschmajew. On low-rank approximability of
solutions to high-dimensional operator equations and eigenvalue
problems. _Linear Algebra and its Applications_, 493:556–572, 2016.


[124] P.M. Kroonenberg. _Applied Multiway Data Analysis_ . John Wiley &
Sons Ltd, New York, 2008.


[125] J.B. Kruskal. Three-way arrays: Rank and uniqueness of trilinear
decompositions, with application to arithmetic complexity and
statistics. _Linear Algebra and its Applications_, 18(2):95–138, 1977.


163


[126] V. Kuleshov, A.T. Chaganty, and P. Liang. Tensor factorization via
matrix factorization. In _Proceedings of the Eighteenth International_
_Conference on Artificial Intelligence and Statistics_, pages 507–516, 2015.


[127] N. Lee and A. Cichocki. Estimating a few extreme singular values
and vectors for large-scale matrices in Tensor Train format. _SIAM_
_Journal on Matrix Analysis and Applications_, 36(3):994–1014, 2015.


[128] N. Lee and A. Cichocki. Fundamental tensor operations for largescale data analysis using tensor network formats. _Multidimensional_
_Systems and Signal Processing_, (accepted), 2016.


[129] N. Lee and A. Cichocki. Regularized computation of approximate
pseudoinverse of large matrices using low-rank tensor train
decompositions. _SIAM Journal on Matrix Analysis and Applications_,
37(2):598–623, 2016.


[130] N. Lee and A. Cichocki. Tensor train decompositions for higher
order regression with LASSO penalties. In _Workshop on Tensor_
_Decompositions and Applications (TDA2016)_, 2016.


[131] J. Li, C. Battaglino, I. Perros, J. Sun, and R. Vuduc. An
input-adaptive and in-place approach to dense tensor-times-matrix
multiply. In _Proceedings of the International Conference for High_
_Performance Computing, Networking, Storage and Analysis_, page 76.
ACM, 2015.


[132] M. Li and V. Monga. Robust video hashing via multilinear subspace
projections. _IEEE Transactions on Image Processing_, 21(10):4397–4409,
2012.


[133] S. Liao, T. Vejchodsk´y, and R. Erban. Tensor methods for parameter
estimation and bifurcation analysis of stochastic reaction networks.
_Journal of the Royal Society Interface_, 12(108):20150233, 2015.


[134] A.P. Liavas and N.D. Sidiropoulos. Parallel algorithms for
constrained tensor factorization via alternating direction method of
multipliers. _IEEE Transactions on Signal Processing_, 63(20):5450–5463,
2015.


[135] L.H. Lim and P. Comon. Multiarray signal processing: Tensor
decomposition meets compressed sensing. _Comptes_ _Rendus_
_Mecanique_, 338(6):311–320, 2010.


164


[136] M.S. Litsarev and I.V. Oseledets. A low-rank approach to the
computation of path integrals. _Journal of Computational Physics_,
305:557–574, 2016.


[137] H. Lu, K.N. Plataniotis, and A.N. Venetsanopoulos. A survey of
multilinear subspace learning for tensor data. _Pattern Recognition_,
44(7):1540–1551, 2011.


[138] M. Lubasch, J.I. Cirac, and M.-C. Banuls. Unifying projected
entangled pair state contractions. _New Journal of Physics_, 16(3):033014,
2014.


[139] C. Lubich, T. Rohwedder, R. Schneider, and B. Vandereycken.
Dynamical approximation of hierarchical Tucker and tensor-train
tensors. _SIAM Journal on Matrix Analysis and Applications_, 34(2):470–
494, 2013.


[140] M.W. Mahoney. Randomized algorithms for matrices and data.
_Foundations and Trends in Machine Learning_, 3(2):123–224, 2011.


[141] M.W. Mahoney and P. Drineas. CUR matrix decompositions for
improved data analysis. _Proceedings of the National Academy of_
_Sciences_, 106:697–702, 2009.


[142] M.W. Mahoney, M. Maggioni, and P. Drineas. Tensor-CUR
decompositions for tensor-based data. _SIAM Journal on Matrix_
_Analysis and Applications_, 30(3):957–987, 2008.


[143] H. Matsueda. Analytic optimization of a MERA network and its
relevance to quantum integrability and wavelet. _arXiv preprint_
_[arXiv:1608.02205](http://arxiv.org/abs/1608.02205)_, 2016.


[144] A.Y. Mikhalev and I.V. Oseledets. Iterative representing set selection
for nested cross–approximation. _Numerical Linear Algebra with_
_Applications_, 2015.


[145] L. Mirsky. Symmetric gauge functions and unitarily invariant norms.
_The Quarterly Journal of Mathematics_, 11:50–59, 1960.


[146] J. Morton. Tensor networks in algebraic geometry and statistics.
_Lecture at Networking Tensor Networks, Centro de Ciencias de Benasque_
_Pedro Pascual, Benasque, Spain_, 2012.


165


[147] M. Mørup. Applications of tensor (multiway array) factorizations
and decompositions in data mining. _Wiley Interdisciplinary Review:_
_Data Mining and Knowledge Discovery_, 1(1):24–40, 2011.


[148] V. Murg, F. Verstraete, R. Schneider, P.R. Nagy, and O. Legeza.
Tree tensor network state with variable tensor order: An efficient
multireference method for strongly correlated systems. _Journal of_
_Chemical Theory and Computation_, 11(3):1027–1036, 2015.


[149] N Nakatani and G.K.L. Chan. Efficient tree tensor network states
(TTNS) for quantum chemistry: Generalizations of the density
matrix renormalization group algorithm. _The Journal of Chemical_
_Physics_, 2013.


[150] Y. Nesterov. Efficiency of coordinate descent methods on huge-scale
optimization problems. _SIAM Journal on Optimization_, 22(2):341–362,
2012.


[151] Y. Nesterov. Subgradient methods for huge-scale optimization
problems. _Mathematical Programming_, 146(1-2):275–297, 2014.


[152] N. H. Nguyen, P. Drineas, and T. D. Tran. Tensor sparsification via
a bound on the spectral norm of random tensors. _Information and_
_Inference_, page iav004, 2015.


[153] M. Nickel, K. Murphy, V. Tresp, and E. Gabrilovich. A review of
relational machine learning for knowledge graphs. _Proceedings of the_
_IEEE_, 104(1):11–33, 2016.


[154] A. Novikov and R.A. Rodomanov. Putting MRFs on a tensor train. In
_Proceedings of the International Conference on Machine Learning (ICML_
_’14)_, 2014.


[155] A.C. Olivieri. Analytical advantages of multivariate data processing.
One, two, three, infinity? _Analytical Chemistry_, 80(15):5713–5720,
2008.


[156] R. Or´us. A practical introduction to tensor networks: Matrix product
states and projected entangled pair states. _Annals of Physics_, 349:117–
158, 2014.


[157] I.V. Oseledets. Approximation of 2 _[d]_ ˆ 2 _[d]_ matrices using tensor
decomposition. _SIAM Journal on Matrix Analysis and Applications_,
31(4):2130–2145, 2010.


166


[158] I.V. Oseledets. Tensor-train decomposition. _SIAM Journal on Scientific_
_Computing_, 33(5):2295–2317, 2011.


[159] I.V. Oseledets and S.V. Dolgov. Solution of linear systems and matrix
inversion in the TT-format. _SIAM Journal on Scientific Computing_,
34(5):A2718–A2739, 2012.


[160] I.V. Oseledets, S.V. Dolgov, V.A. Kazeev, D. Savostyanov,
O. Lebedeva, P. Zhlobich, T. Mach, and L. Song. TT-Toolbox,
2012.


[161] I.V. Oseledets and E.E. Tyrtyshnikov. Breaking the curse of
dimensionality, or how to use SVD in many dimensions. _SIAM_
_Journal on Scientific Computing_, 31(5):3744–3759, 2009.


[162] I.V. Oseledets and E.E. Tyrtyshnikov. TT cross–approximation
for multidimensional arrays. _Linear Algebra and its Applications_,
432(1):70–88, 2010.


[163] E.E. Papalexakis, C. Faloutsos, and N.D. Sidiropoulos. Tensors for
data mining and data fusion: Models, applications, and scalable
algorithms. _ACM Transactions on Intelligent Systems and Technology_
_(TIST)_, 8(2):16, 2016.


[164] E.E. Papalexakis, N. Sidiropoulos, and R. Bro. From K-means to
higher-way co-clustering: Multilinear decomposition with sparse
latent factors. _IEEE Transactions on Signal Processing_, 61(2):493–506,
2013.


[165] N. Parikh and S.P. Boyd. Proximal algorithms. _Foundations and Trends_
_in Optimization_, 1(3):127–239, 2014.


[166] D. Perez-Garcia, F. Verstraete, M.M. Wolf, and J.I. Cirac. Matrix
product state representations. _Quantum Information & Computation_,
7(5):401–430, July 2007.


[167] R. Pfeifer, G. Evenbly, S. Singh, and G. Vidal. NCON: A tensor
network contractor for MATLAB. _[arXiv preprint arXiv:1402.0939](http://arxiv.org/abs/1402.0939)_,
2014.


[168] N. Pham and R. Pagh. Fast and scalable polynomial kernels via
explicit feature maps. In _Proceedings of the 19th ACM SIGKDD_
_international conference on Knowledge discovery and data mining_, pages
239–247. ACM, 2013.


167


[169] A-H. Phan and A. Cichocki. Extended HALS algorithm for
nonnegative Tucker decomposition and its applications for multiway
analysis and classification. _Neurocomputing_, 74(11):1956–1969, 2011.


[170] A.-H. Phan, A. Cichocki, A. Uschmajew, P. Tichavsky, G. Luta, and
D. Mandic. Tensor networks for latent variable analysis. Part I:
Algorithms for tensor train decomposition. _ArXiv e-prints_, 2016.


[171] A.-H. Phan, P. Tichavsk`y, and A. Cichocki. Fast alternating ls
algorithms for high order candecomp/parafac tensor factorizations.
_IEEE Transactions on Signal Processing_, 61(19):4834–4846, 2013.


[172] A.-H. Phan, P. Tichavsk`y, and A. Cichocki. Tensor deflation for
candecomp/parafacpart i: Alternating subspace update algorithm.
_IEEE Transactions on Signal Processing_, 63(22):5924–5938, 2015.


[173] A.H. Phan and A. Cichocki. Tensor decompositions for feature
extraction and classification of high dimensional datasets. _Nonlinear_
_Theory and its Applications, IEICE_, 1(1):37–68, 2010.


[174] A.H. Phan, A. Cichocki, P. Tichavsky, D. Mandic, and K. Matsuoka.
On revealing replicating structures in multiway data: A novel tensor
decomposition approach. In _Proceedings of the 10th International_
_Conference LVA/ICA, Tel Aviv, March 12-15_, pages 297–305. Springer,
2012.


[175] A.H. Phan, A. Cichocki, P. Tichavsk´y, R. Zdunek, and S.R. Lehky.
From basis components to complex structural patterns. In _Proceedings_
_of the IEEE International Conference on Acoustics, Speech and Signal_
_Processing, ICASSP 2013, Vancouver, BC, Canada, May 26-31, 2013_,
pages 3228–3232, 2013.


[176] A.H. Phan, P. Tichavsk´y, and A. Cichocki. Low complexity damped
Gauss-Newton algorithms for CANDECOMP/PARAFAC. _SIAM_
_Journal on Matrix Analysis and Applications (SIMAX)_, 34(1):126–147,
2013.


[177] A.H. Phan, P. Tichavsk´y, and A. Cichocki. Low rank tensor
deconvolution. In _Proceedings of the IEEE International Conference_
_on Acoustics Speech and Signal Processing, ICASSP_, pages 2169–2173,
April 2015.


168


[178] S. Ragnarsson. _Structured Tensor Computations: Blocking Symmetries_
_and Kronecker Factorization_ . PhD dissertation, Cornell University,
Department of Applied Mathematics, 2012.


[179] M.V. Rakhuba and I.V. Oseledets. Fast multidimensional convolution
in low-rank tensor formats via cross–approximation. _SIAM Journal on_
_Scientific Computing_, 37(2):A565–A582, 2015.


[180] P. Richt´arik and M. Tak´aˇc. Parallel coordinate descent methods for
big data optimization. _Mathematical Programming_, 156:433–484, 2016.


[181] J. Salmi, A. Richter, and V. Koivunen. Sequential unfolding SVD
for tensors with applications in array signal processing. _IEEE_
_Transactions on Signal Processing_, 57:4719–4733, 2009.


[182] U. Schollw¨ock. The density-matrix renormalization group in the age
of matrix product states. _Annals of Physics_, 326(1):96–192, 2011.


[183] U. Schollw¨ock. Matrix product state algorithms: DMRG, TEBD and
relatives. In _Strongly Correlated Systems_, pages 67–98. Springer, 2013.


[184] N. Schuch, I. Cirac, and D. P´erez-Garc´ıa. PEPS as ground states:
Degeneracy and topology. _Annals of Physics_, 325(10):2153–2192, 2010.


[185] N. Sidiropoulos, R. Bro, and G. Giannakis. Parallel factor analysis
in sensor array processing. _IEEE Transactions on Signal Processing_,
48(8):2377–2388, 2000.


[186] N.D. Sidiropoulos. Generalizing Caratheodory’s uniqueness of
harmonic parameterization to N dimensions. _IEEE Transactions on_
_Information Theory_, 47(4):1687–1690, 2001.


[187] N.D. Sidiropoulos. Low-rank decomposition of multi-way arrays: A
signal processing perspective. In _Proceedings of the IEEE Sensor Array_
_and Multichannel Signal Processing Workshop (SAM 2004)_, July 2004.


[188] N.D. Sidiropoulos and R. Bro. On the uniqueness of multilinear
decomposition of N-way arrays. _Journal of Chemometrics_, 14(3):229–
239, 2000.


[189] N.D. Sidiropoulos, L. De Lathauwer, X. Fu, K. Huang, E.E.
Papalexakis, and C. Faloutsos. Tensor decomposition for signal
processing and machine learning. _[arXiv e-prints arXiv:1607.01668](http://arxiv.org/abs/1607.01668)_,
2016.


169


[190] A. Smilde, R. Bro, and P. Geladi. _Multi-way Analysis: Applications in_
_the Chemical Sciences_ . John Wiley & Sons Ltd, New York, 2004.


[191] S.M. Smith, A. Hyv¨arinen, G. Varoquaux, K.L. Miller, and C.F.
Beckmann. Group-PCA for very large fMRI datasets. _NeuroImage_,
101:738–749, 2014.


[192] L. Sorber, I. Domanov, M. Van Barel, and L. De Lathauwer. Exact line
and plane search for tensor optimization. _Computational Optimization_
_and Applications_, 63(1):121–142, 2016.


[193] L. Sorber, M. Van Barel, and L. De Lathauwer. Optimizationbased algorithms for tensor decompositions: Canonical Polyadic
Decomposition, decomposition in rank- ( _L_ _r_, _L_ _r_, 1 ) terms and a new
generalization. _SIAM Journal on Optimization_, 23(2), 2013.


[194] M. Sørensen and L. De Lathauwer. Blind signal separation via tensor
decomposition with Vandermonde factor. Part I: Canonical polyadic
decomposition. _IEEE Transactions on Signal Processing_, 61(22):5507–
5519, 2013.


[195] M. Sørensen, L. De Lathauwer, P. Comon, S. Icart, and L. Deneire.
Canonical Polyadic Decomposition with orthogonality constraints.
_SIAM Journal on Matrix Analysis and Applications_, 33(4):1190–1213,
2012.


[196] M. Steinlechner. Riemannian optimization for high-dimensional
tensor completion. Technical report, Technical report MATHICSE
5.2015, EPF Lausanne, Switzerland, 2015.


[197] M.M. Steinlechner. _Riemannian Optimization for Solving High-_
_Dimensional Problems with Low-Rank Tensor Structure_ . PhD thesis,
´Ecole Polytechnnque F´ed´erale de Lausanne, 2016.


[198] E.M. Stoudenmire and Steven R. White. Minimally entangled typical
thermal state algorithms. _New Journal of Physics_, 12(5):055026, 2010.


[199] J. Sun, D. Tao, and C. Faloutsos. Beyond streams and graphs:
Dynamic tensor analysis. In _Proceedings of the 12th ACM SIGKDD_
_international conference on Knowledge Discovery and Data Mining_, pages
374–383. ACM, 2006.


170


[200] S.K. Suter, M. Makhynia, and R. Pajarola. TAMRESH - tensor
approximation multiresolution hierarchy for interactive volume
visualization. _Computer Graphics Forum_, 32(3):151–160, 2013.


[201] Y. Tang, R. Salakhutdinov, and G. Hinton. Tensor analyzers. In
_Proceedings of the 30th International Conference on Machine Learning,_
_(ICML 2013), Atlanta, USA_, 2013.


[202] D. Tao, X. Li, X. Wu, and S. Maybank. General tensor discriminant
analysis and Gabor features for gait recognition. _IEEE Transactions on_
_Pattern Analysis and Machine Intelligence_, 29(10):1700–1715, 2007.


[203] P. Tichavsky and A. Yeredor. Fast approximate joint diagonalization
incorporating weight matrices. _IEEE Transactions on Signal Processing_,
47(3):878–891, 2009.


[204] M.K. Titsias. Variational learning of inducing variables in sparse
Gaussian processes. In _Proceedings of the 12th International Conference_
_on Artificial Intelligence and Statistics_, pages 567–574, 2009.


[205] C. Tobler. _Low-rank tensor methods for linear systems and eigenvalue_
_problems_ . PhD thesis, ETH Z¨urich, 2012.


[206] L.N. Trefethen. Cubature, approximation, and isotropy in the
hypercube. _SIAM Review (to appear)_, 2017.


[207] V. Tresp, Y. Esteban, C.and Yang, S. Baier, and D. Krompaß. Learning
with memory embeddings. _[arXiv preprint arXiv:1511.07972](http://arxiv.org/abs/1511.07972)_, 2015.


[208] J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher. Randomized
single-view algorithms for low-rank matrix approximation. _arXiv e-_
_prints_, 2016.


[209] L. Tucker. Some mathematical notes on three-mode factor analysis.
_Psychometrika_, 31(3):279–311, 1966.


[210] L.R. Tucker. The extension of factor analysis to three-dimensional
matrices. In H. Gulliksen and N. Frederiksen, editors, _Contributions to_
_Mathematical Psychology_, pages 110–127. Holt, Rinehart and Winston,
New York, 1964.


[211] A. Uschmajew and B. Vandereycken. The geometry of algorithms
using hierarchical tensors. _Linear Algebra and its Applications_,
439:133—166, 2013.


171


[212] N. Vannieuwenhoven, R. Vandebril, and K. Meerbergen. A
new truncation strategy for the higher-order singular value
decomposition. _SIAM Journal on Scientific Computing_, 34(2):A1027–
A1052, 2012.


[213] M.A.O. Vasilescu and D. Terzopoulos. Multilinear analysis of image
ensembles: Tensorfaces. In _Proceedings of the European Conference on_
_Computer Vision (ECCV)_, volume 2350, pages 447–460, Copenhagen,
Denmark, May 2002.


[214] F. Verstraete, V. Murg, and I. Cirac. Matrix product states,
projected entangled pair states, and variational renormalization
group methods for quantum spin systems. _Advances in Physics_,
57(2):143–224, 2008.


[215] N. Vervliet, O. Debals, L. Sorber, and L. De Lathauwer. Breaking the
curse of dimensionality using decompositions of incomplete tensors:
Tensor-based scientific computing in big data analysis. _IEEE Signal_
_Processing Magazine_, 31(5):71–79, 2014.


[216] G. Vidal. Efficient classical simulation of slightly entangled quantum
computations. _Physical Review Letters_, 91(14):147902, 2003.


[217] S.A. Vorobyov, Y. Rong, N.D. Sidiropoulos, and A.B. Gershman.
Robust iterative fitting of multilinear models. _IEEE Transactions on_
_Signal Processing_, 53(8):2678–2689, 2005.


[218] S. Wahls, V. Koivunen, H.V. Poor, and M. Verhaegen. Learning
multidimensional Fourier series with tensor trains. In _IEEE Global_
_Conference on Signal and Information Processing (GlobalSIP)_, pages 394–
398. IEEE, 2014.


[219] D. Wang, H. Shen, and Y. Truong. Efficient dimension reduction
for high-dimensional matrix-valued data. _Neurocomputing_, 190:25–
34, 2016.


[220] H. Wang and M. Thoss. Multilayer formulation of the
multiconfiguration time-dependent Hartree theory. _Journal of_
_Chemical Physics_, 119(3):1289–1299, 2003.


[221] H. Wang, Q. Wu, L. Shi, Y. Yu, and N. Ahuja. Out-of-core tensor
approximation of multi-dimensional matrices of visual data. _ACM_
_Transactions on Graphics_, 24(3):527–535, 2005.


172


[222] S. Wang and Z. Zhang. Improving CUR matrix decomposition and
the Nystr¨om approximation via adaptive sampling. _The Journal of_
_Machine Learning Research_, 14(1):2729–2769, 2013.


[223] Y. Wang, H.-Y. Tung, A. Smola, and A. Anandkumar. Fast and
guaranteed tensor decomposition via sketching. In _Advances in_
_Neural Information Processing Systems_, pages 991–999, 2015.


[224] S.R. White. Density-matrix algorithms for quantum renormalization
groups. _Physical Review B_, 48(14):10345, 1993.


[225] Z. Xu, F. Yan, and Y. Qi. Infinite Tucker decomposition:
Nonparametric Bayesian models for multiway data analysis. In
_Proceedings of the 29th International Conference on Machine Learning_
_(ICML)_, ICML ’12, pages 1023–1030. Omnipress, July 2012.


[226] Y. Yang and T. Hospedales. Deep multi-task representation learning:
A tensor factorisation approach. _[arXiv preprint arXiv:1605.06391](http://arxiv.org/abs/1605.06391)_,
2016.


[227] T. Yokota, N. Lee, and A. Cichocki. Robust multilinear tensor rank
estimation using Higher Order Singular Value Decomposition and
Information Criteria. _IEEE Transactions on Signal Processing_, accepted,
2017.


[228] T. Yokota, Q. Zhao, and A. Cichocki. Smooth PARAFAC
decomposition for tensor completion. _IEEE Transactions on Signal_
_Processing_, 64(20):5423–5436, 2016.


[229] Z. Zhang, X. Yang, I.V. Oseledets, G.E. Karniadakis, and L. Daniel.
Enabling high-dimensional hierarchical uncertainty quantification
by ANOVA and tensor-train decomposition. _IEEE Transactions on_
_Computer-Aided Design of Integrated Circuits and Systems_, 34(1):63–76,
2015.


[230] H.H. Zhao, Z.Y. Xie, Q.N. Chen, Z.C. Wei, J.W. Cai, and T. Xiang.
Renormalization of tensor-network states. _Physical Review B_,
81(17):174411, 2010.


[231] Q. Zhao, C. Caiafa, D.P. Mandic, Z.C. Chao, Y. Nagasaka, N. Fujii,
L. Zhang, and A. Cichocki. Higher order partial least squares
(HOPLS): A generalized multilinear regression method. _IEEE_
_Transactions on Pattern Analysis and Machine Intelligence_, 35(7):1660–
1673, 2013.


173


[232] Q. Zhao, G. Zhou, T. Adali, L. Zhang, and A. Cichocki. Kernelization
of tensor-based models for multiway data analysis: Processing of
multidimensional structured data. _IEEE Signal Processing Magazine_,
30(4):137–148, 2013.


[233] S. Zhe, Y. Qi, Y. Park, Z. Xu, I. Molloy, and S. Chari. DinTucker:
Scaling up Gaussian process models on large multidimensional
arrays. In _Proceedings of the Thirtieth AAAI Conference on Artificial_
_Intelligence_, 2016.


[234] G. Zhou and A. Cichocki. Canonical Polyadic Decomposition based
on a single mode blind source separation. _IEEE Signal Processing_
_Letters_, 19(8):523–526, 2012.


[235] G. Zhou and A. Cichocki. Fast and unique Tucker decompositions
via multiway blind source separation. _Bulletin of Polish Academy of_
_Science_, 60(3):389–407, 2012.


[236] G. Zhou, A. Cichocki, and S. Xie. Fast nonnegative matrix/tensor
factorization based on low-rank approximation. _IEEE Transactions on_
_Signal Processing_, 60(6):2928–2940, June 2012.


[237] G. Zhou, A. Cichocki, Y. Zhang, and D.P. Mandic. Group component
analysis for multiblock data: Common and individual feature
extraction. _IEEE Transactions on Neural Networks and Learning Systems_,
(in print), 2016.


[238] G. Zhou, A. Cichocki, Q. Zhao, and S. Xie. Efficient nonnegative
Tucker decompositions: Algorithms and uniqueness. _IEEE_
_Transactions on Image Processing_, 24(12):4990–5003, 2015.


[239] G. Zhou, Q. Zhao, Y. Zhang, T. Adali, S. Xie, and A. Cichocki.
Linked component analysis from matrices to high-order tensors:
Applications to biomedical data. _Proceedings of the IEEE_, 104(2):310–
331, 2016.


174


