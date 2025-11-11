rsta.royalsocietypublishing.org

### Review


Article submitted to journal


**Subject Areas:**


Statistical Mechanics, Machine


Learning


**Keywords:**


measure concentration; extreme


points; ensemble equivalence;


Fisher’s discriminant; linear


separability


**Author for correspondence:**


I.Y. Tyukin


[e-mail: I.Tyukin@le.ac.uk](mailto:I.Tyukin@le.ac.uk)


## Blessing of dimensionality: mathematical foundations of the statistical physics of data

A.N. Gorban [1], I.Y. Tyukin [2]


1 Department of Mathematics University of Leicester,


Leicester LE1 7RH, UK
2 Department of Mathematics University of Leicester,


Leicester LE1 7RH, UK and Department of Automation


and Control Processes, Saint-Petersburg State


Electrotechnical University, Saint-Petersburg, 197376,


Russia


The concentration of measure phenomena were
discovered as the mathematical background of
statistical mechanics at the end of the XIX  
beginning of the XX century and were then explored
in mathematics of the XX-XXI centuries. At the

beginning of the XXI century, it became clear that
the proper utilisation of these phenomena in machine
learning might transform the _curse of dimensionality_
into the _blessing of dimensionality_ .
This paper summarises recently discovered phenomena of measure concentration which drastically
simplify some machine learning problems in high
dimension, and allow us to correct legacy artificial
intelligence systems. The classical concentration of
measure theorems state that i.i.d. random points are
concentrated in a thin layer near a surface (a sphere or
equators of a sphere, an average or median level set of
energy or another Lipschitz function, etc.).
The new _stochastic separation theorems_ describe the
thin structure of these thin layers: the random
points are not only concentrated in a thin layer but
are all linearly separable from the rest of the set,
even for exponentially large random sets. The linear
functionals for separation of points can be selected in
the form of the linear Fisher’s discriminant.

All artificial intelligence systems make errors.
Non-destructive correction requires separation of the
situations (samples) with errors from the samples
corresponding to correct behaviour by a simple and
robust classifier. The stochastic separation theorems
provide us by such classifiers and a non-iterative
(one-shot) procedure for learning.


_⃝_ c The Authors. Published by the Royal Society under the terms of the


Creative Commons Attribution License http://creativecommons.org/licenses/


by/4.0/, which permits unrestricted use, provided the original author and


source are credited.


#### 1. Introduction: Five “Foundations”, from geometry to probability, quantum mechanics, statistical physics and machine learning

It’s not given us to foretell
How our words will echo through the

ages,...


F.I. Tyutchev, English Translation by
F.Jude


The Sixth Hilbert Problem was inspired by the “investigations on the foundations of geometry”

[1], i.e. by Hilbert’s work “The Foundations of Geometry” [2], which firmly implanted the
axiomatic method not only in the field of geometry, but also in other branches of mathematics.
The Sixth Problem proclaimed expansion of the axiomatic method beyond existent mathematical
disciplines, into physics and further on.
The Sixth Problem sounds very unusual and not purely mathematical. This may be a reason
why some great works which have been inspired by this problem have no reference to it. The
most famous example is the von Neumann book [3] “Mathematical foundations of quantum
mechanics”. John von Neumann was the assistant of Hilbert and they worked together on the
mathematical foundation of quantum mechanics. This work was obviously in the framework of
the Sixth Problem, but this framework was not mentioned in the book.
In 1933, Kolmogorov answered the Hilbert challenge of axiomatization of the theory of
probability [4]. He did not cite the sixth problem but explicitly referred to Hilbert’s “Foundations
of Geometry” as the prototype for “the purely mathematical development” of the theory. But
Hilbert in his 6th Problem asked for more, for “a rigorous and satisfactory development of the
method of the mean values in mathematical physics”. He had in mind statistical physics and “in
particular the kinetic theory of gases”. The 6th chapter of Kolmogorov’s book contains a survey
of some results of the author and Khinchin about independence and the law of large numbers,
and the Appendix includes a description of the 0-1 laws in probability. These are the first steps to
a rigorous basis of “the method of mean values”. Ten years later, in 1943, Khinchin published a
book “Mathematical foundations of statistical mechanics” [5]. This has brought an answer to the
Sixth Problem one step closer, but again without explicit reference to Hilbert’s talk. The analogy
between the titles of von Neumann and Khinchin books is obvious.

The main idea of statistical mechanics, in its essence, can be called the _blessing of dimensionality_ :
if a system can be presented as a union of many weakly interacting subsystems then, in the
thermodynamic limit (when the number of such subsystems tends to infinity), the whole system
can be described by relatively simple deterministic relations in the low-dimensional space of
macroscopic variables. _More means less_ – in very high-dimensional spaces many differences
between sets and functions become negligible (vanish) and the laws become simpler. This point of
view on statistical mechanics was developed mainly by Gibbs (1902) (ensemble equivalence) [6]
but Khinchin made the following remark about this work: “although the arguments are clear
from the logical standpoint, they do not pretend to any analytical rigor”, exactly in the spirit of
Hilbert’s request for “a rigorous and satisfactory development”. The devil is in the detail: how
should we define the thermodynamic limit and in which sense the ensembles are equivalent? For
some rigorously formulated conditions, the physical statements become exact theorems.
Khinchin considered two types of background theorems: ergodic theorems and limit theorems
for high-dimensional distributions. He claimed that the foundations of statistical mechanics
should be a complete abstraction from the nature of the forces. Limit theorems utilize very
general properties of distributions in high dimension, indeed, but the expectations that ergodicity
is a typical and universal property of smooth high-dimensional multiparticle Hamiltonian
systems were not met [7]. To stress that the ergodicity problem is nontrivial, we have to refer
to the Oxtoby–Ulam theorem about metric transitivity of a generic _continuous_ transformation,



**2**


which preserves volume [8]. (We see that typical properties of continuous transformations differ
significantly from typical properties of smooth transformations).
Various programmes proposed for the mathematical foundation of statistical mechanics were
discussed, for example, by Dobrushin [9] and Batterman [10]. Despite the impressive proof of
ergodicity of some systems (hyperbolic flows or some billiard systems, for example), the Jaynes
point of view [11] on the role of ergodicity in the foundations of statistical mechanics now became
dominant; the Ergodic Hypothesis is neither necessary nor sufficient condition for the foundation
of statistical mechanics (Dobrushin [9] attributed this opinion to Lebowitz, while Jaynes [11]
referred to Gibbs [6], who, perhaps, “did not consider ergodicity as relevant to the foundation
of the subject”).
Through the efforts of many mathematicians, the limit theorems from probability theory and
results about ensemble equivalence from the foundation of statistical physics were developed
far enough to become the general theory of measure concentration phenomena. Three works
were especially important for our work [12–14]. The book [15] gives an introduction into
the mathematical theory of measure concentration. A simple geometric introduction into this
phenomena was given by Ball [16].
Perhaps, the simplest manifestation of measure concentration is the concentration of the
volume of the high-dimensional ball near the sphere. Let _V_ _n_ ( _r_ ) be a volume of the n-dimensional
ball of radius _r_ . It is useful to stress that the ‘ball’ here is not necessarily Euclidean and means
the ball of _any_ norm. Lévy [17] recognised this phenomenon as a very important property of
geometry of high-dimensional spaces. He also proved that equidistributions in the balls are
asymptotically equivalent in high dimensions to the Gaussian distributions with the same mean
value of squared radius. Gibbs de-facto used these properties for sublevel sets of energy to
demonstrate equivalence of ensembles (microcanonical distribution on the surface of constant
energy and canonical distribution in the phase space with the same mean energy).
Maxwell used the concentration of measure phenomenon in the following settings. Consider
a rotationally symmetric probability distribution on the _n_ -dimensional unit sphere. Then its
orthogonal projection on a line will be a Gaussian distribution with small variance 1 _/n_ (for large
_n_ with high accuracy). This is exactly the Maxwellian distribution for one degree of freedom in a
gas (and the distribution on the unit sphere is the microcanonical distribution of kinetic energy of
gas, when the potential energy is negligibly small). Geometrically it means that if we look at the
one-dimensional projections of the unit sphere then the “observable diameter” will be small, of
the order of 1 _/_ ~~_[√]_~~ ~~_n_~~ ~~.~~
Lévy noticed that instead of orthogonal projections on a straight line we can use any _η_  Lipschitz function _f_ (with _∥f_ ( _x_ ) _−_ _f_ ( _y_ ) _∥≤_ _η∥x −_ _y∥_ ). Let points _x_ be distributed on a unit
_n_ -dimensional sphere with rotationally symmetric probability distribution. Then the values of
_f_ will be distributed ‘not more widely’ than a normal distribution around the mean value **E** _f_ ; for
all _ε >_ 0



**3**



**P** ( _|f −_ **E** _f_ _| ≥_ _ε_ ) _≤_ 2 exp � _−_ 2 _[nε]_ _cη_ [2][2]



_,_
�



where _c_ is a constant, _c ≤_ 9 _π_ [3] . Interestingly, if we use in this inequality the _median value_ of _f_,
_M_ _f_, instead of the mean, then the estimate of the constant _c_ can be decreased: _c ≤_ 1. From the
statistical mechanics point of view, this Lévy Lemma describes the upper limit of fluctuations
in gas for an arbitrary observable quantity _f_ . The only condition is the sufficient regularity of _f_
(Lipschitz property).
Hilbert’s 6th Problem influenced this stream of research either directly (Kolmogorov and,
perhaps, Khinchin among others) or indirectly, through the directly affected works. And it keeps
to transcend this influence to other areas, including high-dimensional data analysis, data mining,
and machine learning.
On the turn of the millennium, Donoho gave a lecture about main problems of highdimensional data analysis [18] with the impressive subtitle: “The curses and blessings of
dimensionality”. He used the term _curse of dimensionality_ “to refer to the apparent intractability


of systematically searching through a high-dimensional space, the apparent intractability of
accurately approximating a general high-dimensional function, the apparent intractability of
integrating a high-dimensional function.” To describe the blessing of dimensionality he referred
to the concentration of measure phenomenon, “which suggest that statements about very highdimensional settings may be made where moderate dimensions would be too complicated.”
Anderson et al characterised some manifestations of this phenomenon as “The More, the Merrier”

[19].
In 1997, Kainen described the phenomenon of blessing of dimensionality, illustrated them with
a number of different examples in which high dimension actually facilitated computation, and
suggested connections with geometric phenomena in high-dimensional spaces [20].
The claim of Donoho’s talk was similar to Hilbert’s talk and he cited this talk explicitly. (“My
personal research experiences, cited above, convince me of Hilbert’s position, as a long run
proposition, operating on the scale of centuries rather than decades.”) The role of Hilbert’s 6th
Problem in the analysis of the curse and blessing of dimensionality was not mentioned again.
The blessing of dimensionality and the curse of dimensionality are two sides of the same
coin. For example, the typical property of a random finite set in a high-dimensional space is: the
squared distance of these points to a selected point are, with high probability close to the average
(or median) squared distance. This property drastically simplifies the expected geometry of data
(blessing) [21,22] but, at the same time, makes the similarity search in high dimensions difficult
and even useless (curse) [23].
Extension of the 6th Hilbert Problem to data mining and machine learning is a challenging
task. There exist no unified general definition of machine learning. Most classical texts consider
machine learning through formalisation and analysis of a set of standardised tasks [24–26].
Traditionally, these tasks are:


_•_ Classification – learning to predict a categorical attribute using values of given attributes
on the basis of given examples (supervised learning);

_•_ Regression – learning to predict numerical attributes using values of given attributes on
the basis of given examples (supervised learning);

_•_ Clustering – joining of similar objects in several clusters (unsupervised learning) [27];

_•_ Various data approximation and reduction problems: linear and nonlinear principal
components [28], principal graphs [29], independent components [30], etc. (clustering
can be also considered as a data approximation problem [31]);

_•_ Probability distribution estimation.


For example, Cucker and Smale [24] considered the least square regression problem. This is
the problem of the best approximation of an unknown function _f_ : _X →_ _Y_ from a random sample
of pairs ( _x, y_ ) _∈_ _X × Y_ . Selection of “the best” regression function means minimization of the
mean square error deviation of the observed _y_ from the value _f_ ( _x_ ). They use the concentration
inequalities to evaluate the probability that the approximation has a given accuracy.
It is important to mention that the Cucker–Smale approach was inspired in particular by J.
von Neumann: “We try to write in the spirit of H. Weyl and J. von Neumann’s contributions to
the foundations of quantum mechanics” [24]. The J. von Neumann book [3] was a step in the
realisation of Hilbert’s 6th problem programme, as we perfectly know. Therefore, the Cucker–
Smale “Mathematical foundation of learning” is a grandchild of the 6th problem. This is the fourth
“Foundation” (after Kolmogorov, von Neumann, and Khinchin). Indeed, it was an attempt to give
a rigorous development of what they “have found to be the central ideas of learning theory”. This
problem statement follows Hilbert’s request for “rigorous and satisfactory development of the
method of mean values”, but this time the development was done for machine learning instead
of mathematical physics.
Cucker and Smale followed Gauss and proved that the least squares solution enjoys
remarkable statistical properties. i.e. it provides the _minimum variance estimate_ [24]. Nevertheless,
non-quadratic functionals are employed for solution of many problems: to enhance robustness, to



**4**


avoid oversensitivity to outliers, to find sparse regression with exclusion of non-necessary input
variables, etc. [25,26]. Even non-convex quasinorms and their tropical approximations are used
efficiently to provide sparse and robust learning results [32]. Vapnik [26] defined a formalised
fragment of machine learning using minimisation of a _risk functional_ that is the mathematical
expectation of a general loss function.
M. Gromov [33] proposed a radically different concept of ergosystems which function
by building their “internal structure” out of the “raw structures” in the incoming flows of
signals. The essential mechanism of egrgosystem learning is goal free and independent of any
reinforcement. In a broad sense, loosely speaking, in this concept “structure” = “interesting
structure” and learning of structure is goal-free and should be considered as a structurally
interesting process.
There are many other approaches and algorithms in machine learning, which use some
specific ideas from statistical mechanics: annealing, spin glasses, etc. (see, for example, [34])
and randomization. It was demonstrated recently that the assignment of random parameters
should be data-dependent to provide the efficient and universal approximation property of
the randomized learner model [35]. Various methods for evaluation of the output weights
of the hidden nodes after random generation of new nodes were also tested [35]. Swarm
optimization methods for learning with random re-generation of the swarm (“virtual particles”)
after several epochs of learning were developed in 1990 [36]. Sequential Monte Carlo methods
for learning neural networks were elaborated and tested [37]. A comprehensive overview of the
classical algorithms and modern achievements in stochastic approaches to neural networks was
performed by Scardapane and Wang [38].
In our paper, we do not discuss these ideas, instead we focus on a deep and general similarity
between high-dimensional problems in learning and statistical physics. We summarise some
phenomena of measure concentration which drastically affect machine learning problems in high
dimension.

#### 2. Waist concentration and random bases in machine learning


After classical works of Fisher [39] and Rosenblatt [40], linear classifiers have been considered as
inception of Data Analytics and Machine Learning (see e.g. [26,41,42], and references therein).
The mathematical machinery powering these developments is based on the concept of linear
separability.


**Definition 2.1.** _Let X and Y be subsets of_ R _[n]_ _. Recall that a linear functional l on_ R _[n]_ _separates X and Y_
_if there exists a t ∈_ R _such that_
_l_ ( _**x**_ ) _> t > l_ ( _**y**_ ) _∀_ _**x**_ _∈X_ _,_ _**y**_ _∈Y._


_A set S ⊂_ R _[n]_ _is_ linearly separable _if for each_ _**x**_ _∈S there exists a linear functional l such that l_ ( _**x**_ ) _> l_ ( _**y**_ )
_for all y ∈S,_ _**y**_ _̸_ = _**x**_ _._


If _X ⊂_ R _[n]_ is a set of measurements or data samples that are labelled as “Class 1”, and _Y_ is a set
of data labelled as “Class 2” then a functional _l_ separating _X_ and _Y_ is the corresponding linear
classifier. The fundamental question, however, is whether such functionals exist for the given _X_
and _Y_, and if the answer is “Yes” then how to find them?
It is well-known that if (i) _X_ and _Y_ are disjoint, (ii) the cardinality, _|X ∪Y|_, of _X ∪Y_ does
not exceed _n_ + 1, and (iii) elements of _X ∪Y_ are in general position, then they are vertices of a
simplex. Hence, in this setting, there always is a linear functional _l_ separating _X_ and _Y_ .
Rosenblatt’s _α_ -perceptron [40] used a population of linear threshold elements with random
synaptic weights ( _A_ -elements) as layer before an _R_ -element, that is a linear threshold element
which learns iteratively (authors of some papers and books called the _R_ -elements “perceptrons”
and lose the complex structure of _α_ -perceptron with a layer of random _A_ -elements). The
randomly initiated elements of the first layer can undergo selection of the most relevant elements.



**5**


According to Rosenblatt [40], any set of data vectors becomes linear separable after
transformation by the layer of _A_ -elements, if the number of these randomly chosen elements
is sufficiently large. Therefore the perceptron can solve any classification problem, where classes
are defined by pointing out examples (ostensive definition). But this “sufficiently large” number
of random elements depends on the problem and may be large, indeed. It can grow for a
classification task proportionally to the number of the examples. The perceptron with sufficiently
large number of _A_ -elements can approximate binary-valued functions on finite domains with
arbitrary accuracy. Recently, the bounds on errors of these approximations are derived [43]. It
is proven that unless the number of network units grows faster than any polynomial of the
logarithm of the size of the domain, a good approximation cannot be achieved for almost any
uniformly randomly chosen function. The results are obtained by application of concentration
inequalities.
The method of random projections became popular in machine learning after the JohnsonLindenstrauss Lemma [44], which states that relatively large sets of _m_ vectors in a highdimensional Euclidean space R _[d]_ can be linearly mapped into a space of much lower dimension
_n_ with approximate preservation of distances. This mapping can be constructed (with high
probability) as a projection on _n_ random basis vectors with rescaling of the projection with a factor
~~_√_~~ _d_ [45]. Repeating the projection _O_ ( _m_ ) times and selecting the best of them, one can achieve the

appropriate accuracy of the distance preservation. The number of points _m_ can be exponentially
large with _n_ ( _m ≤_ exp( _cn_ )).
Two unit random vectors in high dimension are almost orthogonal with high probability. This
is a simple manifestation of the so-called _waist concentration_ [13]. A high-dimensional sphere is
concentrated near its equator. This is obvious: just project a sphere onto a hyperplane and use the
concentration argument for a ball on the hyperplane (with a simple trigonometric factor). This
seems highly non-trivial, if we ask: near which equator? The answer is: near each equator. This
answer is obvious because of rotational symmetry but it seems to be counter-intuitive.
We call vectors _**x**_, _**y**_ from Euclidean space R _[n]_ _ε-orthogonal_ if _|_ ( _**x**_ _,_ _**y**_ ) _| < ε_ ( _ε >_ 0). Let _**x**_ and
_**y**_ be i.i.d. random vectors distributed uniformly (rotationally invariant) on the unit sphere in
Euclidean space R _[n]_ . Then the distribution of their inner product satisfies the inequality (see, for
example [16] or [46] and compare to Maxwellian and Lévy’s lemma):



**6**



**P** ( _|_ ( _**x**_ _,_ _**y**_ ) _| < ε_ ) _≥_ 1 _−_ 2 exp _−_ [1] _._
� 2 _[nε]_ [2] �



**Proposition 2.1.** _Let_ _**x**_ 1 _, . . .,_ _**x**_ _N_ _be be i.i.d. random vectors distributed uniformly (rotationally_
_invariant) on the unit sphere in Euclidean space_ R _[n]_ _. For_



_ε_ [2] _n_ 1
_N < e_ 4 ln

� � 1 _−_ _ϑ_



�� [1] 2



(2.1)



_all vectors_ _**x**_ 1 _, . . .,_ _**x**_ _N_ _are pairwise ε-orthogonal with probability P >_ 1 _−_ _ϑ. [46]_


There are two consequences of this statement: (i) in high dimension there exist exponentially
many pairwise almost orthogonal vectors in R _[n]_, and (ii) _N_ random vectors are _ε_ -orthogonal
with high probability _P >_ 1 _−_ _ϑ_ even for exponentially large _N_ (2.1). Existence of exponentially
large _ε_ -orthogonal systems in high-dimensional spaces was discovered in 1993 by Kainen and
K˚urková [47]. They introduced the notion of _quasiorthogonal dimension_, which was immediately
utilised in the problem of random indexing of high-dimensional data [21]. The fact that an
exponentially large random set consists of pairwise _ε_ -orthogonal vectors with high probability
was demonstrated in the work [46] and used for analysis of data approximation problem in


random bases. We show that not only such _ε_ -orthogonal sets exist, but also that they are typical

in some sense.

_N_ randomly generated vectors _**x**_ _i_ will be almost orthogonal to a given data vector _**y**_ (the angle
between _**x**_ and _**y**_ will be close to _π/_ 2 with probability close to one). Therefore, the coefficients
in the approximation of _**y**_ by a linear combination of _**x**_ _i_ could be arbitrarily large and the
approximation problem will be ill-conditioned, with high probability. The following alternative
is proven for approximation by random bases:


_•_ Approximation of a high-dimensional data vector by linear combinations of randomly
and independently chosen vectors requires (with high probability) generation of
exponentially large “bases”, if we would like to use bounded coefficients in linear
combinations.

_•_ If arbitrarily large coefficients are allowed, then the number of randomly generated
elements that are sufficient for approximation is even less than dimension. We have to pay
for such a reduction of the number of elements by ill-conditioning of the approximation
problem.


We have to choose between a well-conditioned approximation problem in exponentially large
random bases and an ill-conditional problem in relatively small (moderate) random bases.
This dichotomy is fundamental, and it is a direct consequence of the waist concentration
phenomenon. In what follows, we will formally present another concentration phenomenon,
stochastic separation theorems [48,49], and outline their immediate applications in AI and

neuroscience.

#### 3. Stochastic separation theorems and their applications in Artificial Intelligence systems


(a) Stochastic separation theorems


Existence of a linear functional that separates two finite sets _X_ _, Y ⊂_ R _[n]_ is no longer obvious
when _|X ∪Y| ≫_ _n_ . A possible way to answer both questions could be to cast the problem as a
constrained optimization problem within the framework of e.g. support vector machines [26]. The
issue with this approach is that theoretical worst-case estimates of computational complexity for
determining such functions are of the order _O_ ( _|X ∪Y|_ [3] ) (for quadratic loss functions); a posteriori
analysis of experiments on practical use cases, however, suggest that the complexity could be
much smaller and than _O_ ( _|X ∪Y|_ [3] ) and reduce to linear or even sublinear in _|X ∪Y|_ [50].
This apparent discrepancy between the worst-case estimates and a-posteriori evaluation of
computational complexities can be resolved if concentration effects are taken into account. If
the dimension _n_ of the underlying topological vector space is large then random finite but
exponentially large in _n_ samples are linearly separable, with high probability, for a range of
practically relevant classes of distributions. Moreover, we show that the corresponding separating
functionals can be derived using Fisher linear discriminants [39]. Computational complexity of
the latter is linear in _|X ∪Y|_ . It can be made sub-linear too in if proper sampling is used to
estimate corresponding covariance matrices. As we have shown in [49], the results hold for i.i.d.
random points from equidistributions in a ball, a cube, and from distributions that are products
of measures with bounded support. The conclusions are based on stochastic separation theorems
for which the statements for relevant classes of distributions are provided below.



**7**



**Theorem 3.1** (Equidistribution in B _n_ (1) [48,49]) **.** _Let {_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _} be a set of M i.i.d. random points_
_from the equidustribution in the unit ball_ B _n_ (1) _. Let_ 0 _< r <_ 1 _, and ρ_ = ~~_√_~~ 1 _−_ _r_ [2] _. Then_



**P** _∥_ _**x**_ _M_ _∥_ _> r and_ _**x**_ _i_ _,_ _**x**_ _M_
� � _∥_ _**x**_ _M_ _∥_



_< r for all i ̸_ = _M_ _≥_ 1 _−_ _r_ _[n]_ _−_ 0 _._ 5( _M −_ 1) _ρ_ _[n]_ ; (3.1)
� �


**8**





ρ = |A`O`|


_r_ = |O O`|


|O A| < |O O`|





**Figure 1.** Illustration to Theorem 3.1.



**P** _∥_ _**x**_ _j_ _∥_ _> r and_ _**x**_ _i_ _,_ _**x**_ _j_ _< r for all i, j, i ̸_ = _j_ _≥_ 1 _−_ _Mr_ _[n]_ _−_ 0 _._ 5 _M_ ( _M −_ 1) _ρ_ _[n]_ ; (3.2)
� � _∥_ _**x**_ _j_ _∥_ � �



_**x**_ _i_ _**x**_ _j_
**P** _∥_ _**x**_ _j_ _∥_ _> r and_
� � _∥_ _**x**_ _i_ _∥_ _[,]_ _∥_ _**x**_ _j_ _∥_



_< r for all i, j, i ̸_ = _j_ _≥_ 1 _−_ _Mr_ _[n]_ _−_ _M_ ( _M −_ 1) _ρ_ _[n]_ _._ (3.3)
� �



The proof of the theorem can be illustrated with Fig. 1. The probability that a single element,
_**x**_ _M_, belongs to the difference B _n_ (1) _\_ B _n_ ( _r_ ) of two _n_ -balls centred at _O_ is not smaller than 1 _−_ _r_ _[n]_ .
Consider the hyperplane



_**x**_ _M_
_l_ ( _**x**_ ) = _r,_ where _l_ ( _**x**_ ) = _**x**_ _,_
� _∥_ _**x**_ _M_ _∥_



_._
�



This hyperplane partitions the unit ball B _n_ (1) centred at _O_ into two disjoint subsets: the spherical
cap (shown as grey shaded area in Fig. 1) and the rest of the ball. The element _**x**_ _M_ is in the
shaded area and is on the line containing the vector _OO_ _[′]_ . The volume of this spherical cap does
not exceed the volume of the half-ball of radius _ρ_ centred at _O_ _[′]_ (the ball B _n_ ( _ρ_ ) is shown as a blue
dashed circle in the figure). Recall that



**P** ( _A_ 1 & _A_ 2 & _. . ._ & _A_ _m_ ) _≥_ 1 _−_ �(1 _−_ **P** ( _A_ _i_ )) for any events _A_ 1 _, . . ., A_ _m_ _._ (3.4)


_i_



This assures that (3.1) holds. Applying the same argument to all elements of the set _S_ results in
(3.2). Finally, to show that (3.3) holds, observe that the length of the segment _OA_ on the tangent
line to the sphere S _n−_ 1 ( _ρ_ ) centred at _O_ _[′]_ is always smaller than _r_ = _|OO_ _[′]_ _|_ . Hence the cosine of
the angle between an element from (B _n_ (1) _\_ B _n_ ( _r_ )) _\_ B _n_ ( _ρ_ ) and the vector _OO_ _[′]_ is bounded from
above by cos(∠( _OA_ _[′]_ _, OO_ _[′]_ )) = _r_ . The estimate now follows from (3.4).
According to Theorem 3.1, the probability that a single element _**x**_ _M_ from the sample _S_ =
_{_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _}_ is linearly separated from the set _S \ {_ _**x**_ _M_ _}_ by the hyperplane _l_ ( _x_ ) = _r_ is at least



2

1 _−_ _r_ _[n]_ _−_ 0 _._ 5( _M −_ 1) 1 _−_ _r_ [2] [�] _[n]_ _._
�



This probability estimate depends on both _M_ = _|S|_ and dimensionality _n_ . An interesting
consequence of the theorem is that if one picks a probability value, say 1 _−_ _ϑ_, then the maximal
possible values of _M_ for which the set _S_ remains linearly separable with probability that is no
less than 1 _−_ _ϑ_ grows at least exponentially with _n_ . In particular, the following holds



**Corollary 3.1.** _Let {_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _} be a set of M i.i.d. random points from the equidustribution in the unit_
_ball_ B _n_ (1) _. Let_ 0 _< r, ϑ <_ 1 _, and ρ_ = ~~_√_~~ 1 _−_ _r_ [2] _. If_


_M <_ 2( _ϑ −_ _r_ _[n]_ ) _/ρ_ _[n]_ _,_ (3.5)


_then_ **P** (( _**x**_ _i_ _,_ _**x**_ _M_ ) _< r∥_ _**x**_ _M_ _∥_ _for all i_ = 1 _, . . ., M −_ 1) _>_ 1 _−_ _ϑ. If_



_M <_ ( _r/ρ_ ) _[n]_ � _−_ 1 + ~~�~~ 1 + 2 _ϑρ_ _[n]_ _/r_ [2] _[n]_ � _,_ (3.6)


_then_ **P** (( _**x**_ _i_ _,_ _**x**_ _j_ ) _< r∥_ _**x**_ _i_ _∥_ _for all i, j_ = 1 _, . . ., M, i ̸_ = _j_ ) _≥_ 1 _−_ _ϑ._
_In particular, if inequality (_ 3.6 _) holds then the set {_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _} is linearly separable with probability_
_p >_ 1 _−_ _ϑ._


The linear separability property of finite but exponentially large samples of random i.i.d.
elements is not restricted to equidistributions in B _n_ (1). As has been noted in [22], it holds for
equidistributions in ellipsoids as well as for the Gaussian distributions. Moreover, it can be
generalized to product distributions in a unit cube. Consider, e.g. the case when coordinates
of the vectors _**x**_ = ( _X_ 1 _, . . ., X_ _n_ ) in the set _S_ are independent random variables _X_ _i_, _i_ = 1 _, . . ., n_
with expectations _X_ _i_ and variances _σ_ _i_ [2] _[> σ]_ 0 [2] _[>]_ [ 0][. Let][ 0] _[ ≤]_ _[X]_ _i_ _[≤]_ [1][ for all] _[ i]_ [ = 1] _[, . . ., n]_ [. The following]
analogue of Theorem 3.1 can now be stated.


**Theorem 3.2** (Product distribution in a cube [49]) **.** _Let {_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _} be i.i.d. random points from the_
_product distribution in a unit cube. Let_


_̸_



**9**


_̸_



_R_ 0 [2] [=] � _σ_ _i_ [2] _[≥]_ _[nσ]_ 0 [2] _[,]_


_i_


_̸_



_and_ 0 _< δ <_ 2 _/_ 3 _. Then_


_̸_



(3.7)


(3.8)


_̸_



�


_̸_



_−_ ~~_**x**_~~ _**x**_ _M_ _−_ ~~_**x**_~~

_R_ 0 _,_ _∥_ _**x**_ _M_ _−_ ~~_**x**_~~ ~~_∥_~~


_̸_



�


_̸_



_<_ ~~_√_~~ 1 _−_ _δ for all i, j, i ̸_ = _M_
�


_̸_



**P**


_̸_



1 _−_ _δ ≤_ _[∥]_ _**[x]**_ _[j]_ _[ −]_ ~~_**[x]**_~~ ~~_[∥]_~~ [2]


_̸_




_[ −]_ ~~_**[x]**_~~ ~~_[∥]_~~ _**x**_ _i_ _−_ ~~_**x**_~~

_≤_ 1 + _δ and_
_R_ 0 [2] � _R_ 0


_̸_



_≥_ 1 _−_ 2 _M_ exp _−_ 2 _δ_ [2] _R_ 0 [4] _[/n]_ _−_ ( _M −_ 1) exp _−_ 2 _R_ 0 [4] [(2] _[ −]_ [3] _[δ]_ [)] [2] _[/n]_ ;
� � � �


_̸_



_i_ _−_ ~~_**x**_~~ _**x**_ _j_ _−_ ~~_**x**_~~

_R_ 0 _,_ _∥_ _**x**_ _j_ _−_ ~~_**x**_~~ ~~_∥_~~


_̸_



�


_̸_



_<_ ~~_√_~~ 1 _−_ _δ for all i, j, i ̸_ = _j_
�


_̸_



**P**


_̸_



�


_̸_



1 _−_ _δ ≤_ _[∥]_ _**[x]**_ _[j]_ _[ −]_ ~~_**[x]**_~~ ~~_[∥]_~~ [2]


_̸_




_[ −]_ ~~_**[x]**_~~ ~~_[∥]_~~ _**x**_ _i_ _−_ ~~_**x**_~~

_≤_ 1 + _δ and_
_R_ 0 [2] � _R_ 0


_̸_



_≥_ 1 _−_ 2 _M_ exp _−_ 2 _δ_ [2] _R_ 0 [4] _[/n]_ _−_ _M_ ( _M −_ 1) exp _−_ 2 _R_ 0 [4] [(2] _[ −]_ [3] _[δ]_ [)] [2] _[/n]_ _._
� � � �


The proof is based on concentration inequalities in product spaces [14,51]. Numerous
generalisations of Theorems 3.1, 3.2 are possible for different classes of distributions, for example,
for weakly dependent variables, etc.
Linear separability, as an inherent property of data sets in high dimension, is not necessarily
confined to cases whereby a linear functional separates a single element of a set from the
rest. Theorems 3.1, 3.2 be generalized to account for _m_ -tuples, _m >_ 1 too. An example of such
generalization is provided in the next theorem.


**Theorem 3.3** (Separation of _m_ -tuples [52]) **.** _Let X_ = _{_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _} and Y_ = _{_ _**x**_ _M_ +1 _, . . .,_ _**x**_ _M_ + _k_ _}_
_be i.i.d. samples from the equidistribution in_ B _n_ (1) _. Let Y_ _c_ = _{_ _**x**_ _M_ + _r_ 1 _, . . .,_ _**x**_ _M_ + _r_ _m_ _} be a subset of m_
_elements from Y such that_


_̸_



_β_ 2 ( _m −_ 1) _≤_ �


_r_ _j_ _, r_ _j_ = _̸_ _r_ _i_



� _**x**_ _M_ + _r_ _i_ _,_ _**x**_ _M_ + _r_ _j_ � _≤_ _β_ 1 ( _m −_ 1) _for all i_ = 1 _, . . ., m._ (3.9)

_̸_



_̸_


_Then_



_̸_


_n_

2

**P** ( _∃_ _a linear functional separating X and Y_ _c_ ) _≥_ max 1 _−_ _[∆]_ [(] _[ε][,][ m]_ [)]
_ε∈_ (0 _,_ 1) [(1] _[ −]_ [(1] _[ −]_ _[ε]_ [)] _[n]_ [)] _[m]_ � 2



_̸_


_M_

_,_
�



_̸_


(3.10)

_where_



_̸_


2

_,_
�



_̸_


_∆_ ( _ε, m_ ) = 1 _−_ [1]

_m_



_̸_


(1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1)
� ~~�~~ 1 + ( _m −_ 1) _β_ 1



_̸_


_subject to:_
(1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1) _>_ 0 _,_ 1 + ( _m −_ 1) _β_ 1 _>_ 0 _._


The separating linear functional is again the inner product, and the separating hyperplane can
be taken in the form [52]:



**10**



�



(1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1)
� ~~�~~ 1 + ( _m −_ 1) _β_ 1



_**y**_ ¯
_l_ ( _**x**_ ) = _r_ ; where _l_ ( _**x**_ ) = _**x**_ _,_ ¯
� _∥_ _**y**_ _∥_



1
_, r_ =
� ~~_√m_~~



_,_ (3.11)



and _ε_ is the maximizer of the nonlinear program in the right-hand side of (3.10), and ¯ _**y**_ =
_m_ 1 � _mi_ =1 _**[x]**_ _[M]_ [+] _[r]_ _i_ [. To see this, observe that] _[ ∥]_ _**[x]**_ _[M]_ [+] _[r]_ _i_ _[∥≥]_ [1] _[ −]_ _[ε]_ [,] _[ ε][ ∈]_ [(0] _[,]_ [ 1)][, for all] _[ i]_ [ = 1] _[, . . ., m]_ [, with]
probability (1 _−_ (1 _−_ _ε_ ) _[n]_ ) _[m]_ . With this probability the following estimate holds:

_**y**_ ¯ 1

¯ _≥_ ¯ (1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1) _._

� _∥_ _**y**_ _∥_ _[,]_ _**[ x]**_ _[M]_ [+] _[r]_ _[i]_ � _m∥_ _**y**_ _∥_ � �



_∥_ _**y**_ ¯ _∥_ _[,]_ _**[ x]**_ _[M]_ [+] _[r]_ _[i]_



1
_≥_ ¯
� _m∥_ _**y**_ _∥_



(1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1) _._
� �



Hence
1
_m_ [(1 + (] _[m][ −]_ [1)] _[β]_ [1] [)] _[ ≥]_ [(¯] _**[y]**_ _[,]_ [ ¯] _**[y]**_ [)] _[ ≥]_ _m_ [1]



(1 _−_ _ε_ ) [2] + _β_ 2 ( _m −_ 1) _,_
� �



and _l_ ( _**x**_ ) in (3.11) is the required functional (see also Fig. 1).
If the elements of _Y_ _c_ are uncorrelated, i.e. the values of _β_ 1 ( _m −_ 1) _, β_ 2 ( _m −_ 1) are small, then
the distance from the spherical cap induced by linear functional (3.11) to the center of the ball
decreases as _O_ (1 _/_ ~~_[√]_~~ ~~_m_~~ ~~)~~ . This means that the lower-bound probability estimate in (3.10) is expected
to decrease too. On the other hand, if the elements of _Y_ _c_ are all positively correlated, i.e. 1 _≥_ _β_ 1 _>_
_β_ 2 _>_ 0, then one can derive a lower-bound probability estimate which does not depend on _m_ .
Peculiar properties of data in high dimension, expressed in terms of linear separability,
have several consequences and applications in the realm of Artificial Intelligence and Machine
Learning of which the examples are provided in the next sections.


(b) Correction of legacy AI systems


Legacy AI systems, i.e. AI systems that have been deployed and are currently in operation, are
becoming more and more wide-spread. Well-known commercial examples are provided by global
multi-nationals, including Google, IMB, Amazon, Microsoft, and Apple. Numerous open-source
legacy AIs have been created to date, together with dedicated software for their creation (e.g.
Caffe [53], MXNet [54], Deeplearning4j [55], and Tensorflow [56] packages). These AI systems
require significant computational and human resources to build. Regardless of resources spent,
virtually any AI and/or machine learning-based systems are likely to make a mistake. Real-time
correction of these mistakes by re-training is not always viable due to the resources involved. AI
re-training is not necessarily desirable either, since AI’s performance after re-training may not
always be guaranteed to exceed that of the old one. We can, therefore, formulate the technical
requirements for the correction procedures. Corrector should: (i) be simple; (ii) not change the
skills of the legacy system; (iii) allow fast non-iterative learning; and (iv) allow correction of new
mistakes without destroying of previous corrections.
A possible remedy to this issue is the AI correction method [22] based on stochastic separation
theorems. Suppose that at a time instance _t_ values of signals from inputs, outputs, and internal
state of a legacy AI system could be combined together to form a single measurement object,
_**x**_ = ( _x_ 1 _, . . ., x_ _n_ ). All _n_ entries in this object are numerical values, and each measurement _**x**_
corresponds to a relevant decision of the AI system at time _t_ . Over the course of the system’s
existence a set _S_ of such measurements is collected. For each element in the set _S_ a label “correct”

or “incorrect” is assigned, depending on external evaluation of the system’s performance.
Elements corresponding to “incorrect” labels are then filtered out and dealt with separated by
an additional subsystem, a corrector. A diagram illustrating the process is shown in Fig. 2. In
this diagram, the original legacy AI system (shown as Legacy AI System 1) is supplied with
a corrector altering its responses. The combined new AI system can in turn be augmented by
another corrector, leading to a cascade of AI correctors (see Fig. 2).
If distributions modelling elements of the set _S_ are e.g. an equidistribution in a ball or an
ellipsoid, product of measures distribution, a Gaussian etc., then


**Legacy AI System 2**

Inputs Outputs



**11**







Corrector 2



**Figure 2.** Cascade of AI correctors



700


600


500


400


300


200


100


1 40 80 120 160


False positives removed (out of 189)


**Figure 3.** True positives removed as a function of false positives removed by a single-functional corrector [22].


_•_ Theorems 3.1–3.3 guarantee that construction of such AI correctors can be achieved using
mere linear functionals.

_•_ These linear functions admit a closed-form formulae (Fisher linear discriminant) and can
be determined in a non-iterative way.

_•_ Availability of explicit closed-form formulae in the form of Fisher discriminant offers
major computational benefits as it eliminates the need to employ iterative and more
computationally expensive alternatives such as e.g. SVMs.

_•_ If a cascade of correctors is employed, performance of the corrected system drastically
improves [22].


The results, perhaps, can be generalized to other classes of distributions that are regular enough
to enjoy the stochastic separability property.
The corrector principle has been demonstrated in [22] for a legacy AI system in the form of
a convolutional neural network trained to detect pedestrians in images. AI errors were set to be
false positives, and the corrector system had to remove labeled false positives by a single linear
functional. Detailed description of the experiment is provided in [22], and a performance snapshot
is shown in Fig. 3. Dimensionality _n_ of the vectors _**x**_ was 2000. As we can see from Fig. 3, single
linear functionals are capable of removing several errors of a legacy AI without compromising the
system’s performance. Note that AI errors, i.e. false positives, were chosen at random and have
not been grouped or clustered to take advantage of positive correlation. (The definition of clusters
could vary [27].) As the number of errors to be removed grows, performance starts to deteriorate.
This is in agreement with our theoretical predictions (Theorem 3.3).


**Student AI** **Teacher AI**



**12**





of Student AI’s state by Teacher AI


**Figure 4.** AI Knowledge Transfer


(c) Knowledge transfer between AI systems


Legacy AI correctors can be generalized to a computational framework for automated AI
knowledge transfer whereby labelling of the set _S_ is provided by an external AI system.
AI knowledge transfer has been in the focus of growing attention during last decade [57].
Application of stochastic separation theorems to AI knowledge transfer was proposd in [52], and
the corresponding functional diagram of this automated setup is shown in Fig. 4. In this setup a
student AI, denoted as AI _s_, is monitored by a teacher AI, denoted as AI _t_ . Over a period of activity
system AI _s_ generates a set _S_ of objects _**x**_, _**x**_ _∈_ R _[n]_ . Exact composition of the set _S_ depends on a
task at hand. If AI _s_ outputs differ to that of AI _t_ for the same input then an error is registered in
the system. Objects _**x**_ _∈S_ associated with errors are combined into the set _Y_ . The process gives
rise to two disjoint sets:


_X_ = _{_ _**x**_ 1 _, . . .,_ _**x**_ _M_ _}, X_ = _S \ Y,_ and _Y_ = _{_ _**x**_ _M_ +1 _, . . .,_ _**x**_ _M_ + _k_ _}._


Having created these two sets, knowledge transfer from AI _t_ to AI _s_ can now be organized in
accordance with Algorithm 1. Note that data regularization and whitening are included in the
pre-processing step of Algorithm 1. The algorithm can be used for AI correctors too. Similar to AI
correction, AI knowledge transfer can be cascaded as well. Specific examples and illustrations of
AI knowledge transfer based on stochastic separation theorems are discussed in [52].


(d) Grandmother cells, memory, and high-dimensional brain


Stochastic separation theorems are a generic phenomenon, and their applications are not
limited to AI and machine learning systems. An interesting consequence of these theorems for
neuroscience has been discovered and presented in [58]. Recently, it has been shown that in
humans new memories can be learnt very rapidly by supposedly individual neurons from a
limited number of experiences [59]. Moreover, neurons can exhibit remarkable selectivity to
complex stimuli, the evidence that has led to debates around the existence of the so-called
“grandmother” and “concept” cells [60–62], and their role as elements of a declarative memory.
These findings suggest that not only the brain can learn rapidly but also it can respond selectively
to “rare” individual stimuli. Moreover, experimental evidence indicates that such a cognitive
functionality can be delivered by single neurons [59–61]. The fundamental questions, hence, are:
How is this possible? and What could be the underlying functional mechanisms?
It has been shown in [58] that stochastic separation theorems offer a simple answer to these
fundamental questions. In particular, extreme neuronal selectivity and rapid learning can already
be explained by these theorems. Model-wise, explanation of extreme selectivity is based on
conventional and widely accepted phenomenological generic description of neural response to
stimulation. Rapid acquisition of selective response to multiple stimuli by single neurons is
ensured by classical Hebbian synaptic plasticity [63].


**Algorithm 1** AI Knowledge Transfer/Correction [52]


(i) **Pre-processing**


(a) _Centering_ . For the given set _S_, determine the set average, ¯ _**x**_ ( _S_ ), and generate sets _S_ _c_


_S_ _c_ = _{_ _**x**_ _∈_ R _[n]_ _|_ _**x**_ = _**ξ**_ _−_ _**x**_ ¯( _S_ ) _,_ _**ξ**_ _∈S},_
_Y_ _c_ = _{_ _**x**_ _∈_ R _[n]_ _|_ _**x**_ = _**ξ**_ _−_ _**x**_ ¯( _S_ ) _,_ _**ξ**_ _∈Y}._



**13**



(b) _Regularization_ . Determine covariance matrices Cov( _S_ _c_ ), Cov( _S_ _c_ _\ Y_ _c_ ) of the sets _S_ _c_
and _S_ _c_ _\ Y_ _c_ . Let _λ_ _i_ (Cov( _S_ _c_ )), _λ_ _i_ (Cov( _S_ _c_ _\ Y_ _c_ )) be their corresponding eigenvalues,
and _h_ 1 _, . . ., h_ _n_ be the eigenvectors of Cov( _S_ _c_ ). If some of _λ_ _i_ (Cov( _S_ _c_ )), _λ_ _i_ (Cov( _S_ _c_ _\_
_Y_ _c_ )) are zero or if the ratio [max] min _i_ _[i]_ _{_ _[{]_ _λ_ _[λ]_ _i_ _[i]_ ( [(] _Σ_ _[Σ]_ ( [(] _S_ _[S]_ _c_ _[c]_ )) [))] _}_ _[}]_ [is too large, project] _[ S]_ _[c]_ [ and] _[ Y]_ _[c]_ [ onto]

appropriately chosen set of _m < n_ eigenvectors, _h_ _n−m_ +1 _, . . ., h_ _n_ :


_S_ _r_ = _{_ _**x**_ _∈_ R _[n]_ _|_ _**x**_ = _H_ _[T]_ _**ξ**_ _,_ _**ξ**_ _∈S_ _c_ _},_
_Y_ _r_ = _{_ _**x**_ _∈_ R _[n]_ _|_ _**x**_ = _H_ _[T]_ _**ξ**_ _,_ _**ξ**_ _∈Y_ _c_ _},_


where _H_ = ( _h_ _n−m_ +1 _· · · h_ _n_ ) is the matrix comprising of _m_ significant principal
components of _S_ _c_ .
(c) _Whitening_ . For the centred and regularized dataset _S_ _r_, derive its covariance matrix,
Cov( _S_ _r_ ), and generate whitened sets



_S_ _w_ = _{_ _**x**_ _∈_ R _[m]_ _|_ _**x**_ = Cov( _S_ _r_ ) _[−]_ 2 [1]



_S_ _w_ = _{_ _**x**_ _∈_ R _[m]_ _|_ _**x**_ = Cov( _S_ _r_ ) _[−]_ 2 _**ξ**_ _,_ _**ξ**_ _∈S_ _r_ _},_

_Y_ _w_ = _{_ _**x**_ _∈_ R _[m]_ _|_ _**x**_ = Cov( _S_ _r_ ) _[−]_ [1] 2 _**ξ**_ _,_ _**ξ**_ _∈Y_ _r_ _}._



2 _**ξ**_ _,_ _**ξ**_ _∈Y_ _r_ _}._



(ii) **Knowledge transfer**


(a) _Clustering_ . Pick _p ≥_ 1, _p ≤_ _k_, _p ∈_ N, and partition the set _Y_ _w_ into _p_ clusters
_Y_ _w,_ 1 _, . . . Y_ _w,p_ so that elements of these clusters are, on average, pairwise positively
correlated. That is there are _β_ 1 _≥_ _β_ 2 _>_ 0 such that:



_β_ 2 ( _|Y_ _w,i_ _| −_ 1) _≤_ � ( _**ξ**_ _,_ _**x**_ ) _≤_ _β_ 1 ( _|Y_ _w,i_ _| −_ 1) for any _**x**_ _∈Y_ _w,i_ _._

_ξ∈Y_ _w,i_ _\{_ _**x**_ _}_



(b) _Construction of Auxiliary Knowledge Units_ . For each cluster _Y_ _w,i_, _i_ = 1 _, . . ., p_,
construct separating linear functionals _ℓ_ _i_ and thresholds _c_ _i_ :



_**w**_ _ℓ_ _i_ ( _i_ _**x**_ ) == ��Cov( _∥_ _**ww**_ _ii_ _∥_ _S_ _[,]_ _**[ x]**_ _w_ _\ Y_ � _,_ _w,i_ ) + Cov( _Y_ _w,i_ )� _−_ 1 � _**x**_ ¯( _Y_ _w,i_ ) _−_ _**x**_ ¯( _S_ _w_ _\ Y_ _w,i_ )� _,_



_c_ _i_ = min _**ξ**_ _∈Y_ _w,i_ � _∥_ _**ww**_ _ii_ _∥_ _[,]_ _**[ ξ]**_ � _,_


where ¯ _**x**_ ( _Y_ _w,i_ ), ¯ _**x**_ ( _S_ _w_ _\ Y_ _w,i_ ) are the averages of _Y_ _w,i_ and _S_ _w_ _\ Y_ _w,i_, respectively.
The separating hyperplane is _ℓ_ _i_ ( _**x**_ ) = _c_ _i_ .
(c) _Integration_ . Integrate Auxiliary Knowledge Units into decision-making pathways
of AI _s_ . If, for an _**x**_ generated by an input to AI _s_, any of _ℓ_ _i_ ( _**x**_ ) _≥_ _c_ _i_ then report _**x**_
accordingly (swap labels, report as an error etc.)

#### 4. Conclusion


Twenty-three Hilbert’s problems created important “focus points” for the concentration of efforts
of mathematicians for a century. The Sixth Problem differs significantly from the other twentytwo problems. It is very far from being a purely mathematical problem. It seems to be impossible
to imagine it’s “final solution”. The Sixth Problem is a “programmatic call” [64], and it works:


_•_ We definitely know that the Sixth Problem had great influence on the formulation of the
mathematical foundation of quantum mechanics [3] and on the development of axiomatic
quantum field theory [65].

_•_ We have no doubt (but the authors have no direct evidence) that Sixth Problem has
significantly affected research in the foundation of probability theory [4] and statistical
mechanics [5].

_•_ The modern theory of measure concentration phenomena has direct relations to the
mathematical foundations of probability and statistical mechanics, uses results of
Kolmogorov and Khinchin (among others), and definitely helps to create “a rigorous and
satisfactory development of the method of the mean values...”.

_•_ Some of the recent attempts of rigorous approach to machine learning [24] used parts of
the Sixth Problem programme [3] as a prototype for their conceptual approach.

_•_ The modern idea of _blessing of dimensionality_ in high-dimensional data analysis [18,19,48]
is, in its essence, an extension and further development of ideas from the mathematical
foundations of statistical mechanics.


The classical measure concentration theorems state that random points in a highly-dimensional
data distribution are concentrated in a thin layer near an average or median level set of a
Lipschitz function. The stochastic separation theorems describe the fine structure of these thin
layers: the random points are all linearly separable from the rest of the set even for exponentially
large random sets. Of course, for all these concentration and separation theorems the probability
distribution should be “genuinely” high-dimensional. Equidistributions in balls or ellipsoids or
the products of distributions with compact support and non-vanishing variance are the simple
examples of such distributions. Various generalizations are possible.
For which dimensions does the blessing of dimensionality work? This is a crucial question. The
naïve point of view that dimension of data is just a number of coordinates is wrong. This is the
dimension of the dataspace, where data are originally situated. The notion of _intrinsic_ dimension
of data is needed [66,67]. The situation when the number of data points _N_ is less (or even much
less) than the dimension _d_ of the data space is not exotic. Moreover, Donoho [18] considered the
property _d > N_ as a generic case in the “post-classical world” of data analysis. In such a situation
we really explore data on a _d −_ 1 dimensional plane and should modestly reduce our highdimensional claim. Projection of data on that plane can be performed by various methods. We
can use as new coordinates projections of points on the known datapoints or Pearson’s correlation
coefficients, when it is suitable, for example, when the datapoints are fragments of time series or
large spectral images, etc. In these new coordinates the datatable becomes a square matrix and
further dimensionality reduction could be performed using good old PCA (principal component
analysis), or its nonlinear versions like principal manifolds [28] or neural autoencoders [68].
A standard example can be found in [69]: the initial dataspace consisted of fluorescence
diagrams and had dimension 5 _._ 2 _·_ 10 [5] . There were 62 datapoints, and a combination of correlation
coordinates with PCA showed intrinsic dimension 4 or 5. For selection of relevant principal
components the Kaiser rule, the broken stick models or other heuristical or statistical methods
can be used [70].
Similar preprocessing ritual is helpful even in more “classical” cases when _d < N_ . The
correlation (or projection) transformation is not essential here, but formation of relevant features
with dimension reduction is important. If after model reduction and _whitening_ (transformation
of coordinates to get the unit covariance matrix, step i.c in Algorithm 1) the new dimension
_D_ ≳ 100 then for ≲ 10 [6] datapoints we can expect that the stochastic separation theorems work
with probability _>_ 99%. Thus separation of errors with Fisher’s linear discriminant is possible,
and many other “blessing of dimensionality benefits” are achievable. Of course, some additional
hypotheses about the distribution functions are needed for a rigorous proof, but there is
practically no chance to check them _a priori_ and the validation of the whole system _a posteriori_
is necessary. In smaller dimensions (for example, less than 10), nonlinear data approximation
methods can work well capturing the intrinsic complexity of data, like principal graphs do [29,71].



**14**


We have an alternative: either essentially high-dimensional data with thin shell concentrations,
stochastic separation theorems, and efficient linear methods, or essentially low-dimensional data
with efficient complex nonlinear methods. There is a problem of the ‘no man’s land’ in-between.
To explore this land, we can extract the most interesting low-dimensional structure and then
consider the residual as an essentially high-dimensional random set, which obeys stochastic
separation theorems. We do not know now a theoretically justified efficient approach to this
area, but here we should say following Hilbert: “Wir müssen wissen, wir werden wissen” (“We
must know, we shall know”).


Competing Interests. The authors declare that they have no competing interests.


Authors’ Contributions. Both authors made substantial contributions to conception, proof of the theorems,
analysis of applications, drafting the article, revising it critically, and final approval of the version to be
published.


Funding. This work was supported by Innovate UK grants KTP009890 and KTP010522. IT was supported
by the Russian Ministry of Education and Science, projects 8.2080.2017/4.6 (assessment and computational
support for knowledge transfer algorithms between AI systems) and 2.6553.2017/BCH Basic Part.

#### References


1. Hilbert D. 1902 Mathematical problems. _Bull._ _Amer._ _Math._ _Soc._ **8** (10), 437–479.
[(doi:10.1090/S0002-9904-1902-00923-3)](https://doi.org/10.1090/S0002-9904-1902-00923-3)
2. Hilbert D. 1902 _The Foundations of Geometry_ . La Salle IL: Open court publishing Company. See

[http://www.gutenberg.org/ebooks/17384.](http://www.gutenberg.org/ebooks/17384)
3. Von Neumann J. 1955 _Mathematical Foundations of Quantum Mechanics_ . Princeton: Princeton
University Press. (English translation from German Edition, Springer, Berlin, 1932.)
4. Kolmogorov AN. 1956 _Foundations of the Theory of Probability._ New York: Chelsea Publ.
(English translation from German edition, Springer, Berlin, 1933.)
5. Khinchin AY. 1949 _Mathematical Foundations of Statistical Mechanics._ New York: Courier
Corporation. (English translation from the Russian edition, Moscow – Leningrad, 1943.)
6. Gibbs GW. 1960 [1902] _Elementary Principles in Statistical Mechanics, Developed With Especial_
_Reference to the Rational Foundation of Thermodynamics_ . Dover Publications, New York.
7. Markus L, Meyer KR. 1974 _Generic Hamiltonian Dynamical Systems are Neither Integrable Nor_
_Ergodic. Memoirs of Amer. Math. Soc._ [144. (http://dx.doi.org/10.1090/memo/0144)](http://dx.doi.org/10.1090/memo/0144)
8. Oxtoby JC, Ulam SM. 1941 Measure-preserving homeomorphisms and metrical transitivity.
_Ann. Math._ **42** [. 874–920. (doi:10.2307/1968772)](https://doi.org/10.2307/1968772)
9. Dobrushin RL. 1997 A mathematical approach to foundations of statistical
mechanics. _Atti dei Convegni Lincei – Accademia Nazionale dei Lincei_ **131**, 227–244. See

_∼_
[http://www.mat.univie.ac.at/](http://www.mat.univie.ac.at/~esiprpr/esi179.pdf) esiprpr/esi179.pdf
10. Batterman RW. 1998 Why equilibrium statistical mechanics works: universality and the
renormalization group. _Philos. Sci._ **65** [, 183–208. (doi:10.1086/392634)](https://doi.org/10.1086/392634)
11. Jaynes ET. 1967 Foundations of probability theory and statistical mechanics. In _Delaware_
_Seminar in the Foundations of Physics_ [. Springer, Berlin – Heidelberg, 77–101. (doi:10.1007/978-](https://doi.org/10.1007/978-3-642-86102-4_6)
[3-642-86102-4_6](https://doi.org/10.1007/978-3-642-86102-4_6)
12. Giannopoulos AA, Milman VD. 2000 Concentration property on probability spaces. _Adv._
_Math._ **156** [. 77–106. (doi:10.1006/aima.2000.1949)](https://doi.org/10.1006/aima.2000.1949)
13. Gromov M. 2003 Isoperimetry of waists and concentration of maps. _Geom. Funct. Anal._ **13**,
[178–215. (doi:10.1007/s00039-009-0703-1)](https://doi.org/10.1007/s00039-009-0703-1)
14. Talagrand M. 1995 Concentration of measure and isoperimetric inequalities in product spaces.
_Publications Mathematiques de l’IHES_ **81** [, 73–205. (doi:10.1007/BF02699376)](https://doi.org/10.1007/BF02699376)
15. Ledoux M. 2001 _The Concentration of Measure Phenomenon._ (Mathematical Surveys &
[Monographs No. 89). Providence: AMS. (doi:10.1090/surv/089)](https://dx.doi.org/10.1090/surv/089)
16. Ball K. 1997 _An_ _Elementary_ _Introduction_ _to_ _Modern_ _Convex_ _Geometry._
Flavors of Geometry, Vol. 31. Cambridge, UK: MSRI Publications. See
[http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.4601.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.4601)
17. Lévy P. 1951 _Problèmes concrets d’analyse fonctionnelle._ Paris: Gauthier-Villars.
18. Donoho DL. 2000 High-dimensional data analysis: The curses and
blessings of dimensionality. _AMS_ _Math_ _Challenges_ _Lecture_, 1, 32 pp. See
http://statweb.stanford.edu/ _∼_ [donoho/Lectures/AMS2000/Curses.pdf.](http://statweb.stanford.edu/~donoho/Lectures/AMS2000/Curses.pdf)



**15**


19. Anderson J, Belkin M, Goyal N, Rademacher L, Voss J. 2014 The More, the
Merrier: the Blessing of dimensionality for learning large Gaussian mixtures, _Journal_
_of_ _Machine_ _Learning_ _Research:_ _Workshop_ _and_ _Conference_ _Proceedings_ **35**, 1–30. See
[http://proceedings.mlr.press/v35/anderson14.pdf.](http://proceedings.mlr.press/v35/anderson14.pdf)
20. Kainen PC.1997 Utilizing geometric anomalies of high dimension: when complexity makes
computation easier. In _Computer-Intensive Methods in Control and Signal Processing: The Curse of_
_Dimensionality_ [. New York, Springer, 283–294. (10.1007/978-1-4612-1996-5_18)](https://doi.org/10.1007/978-1-4612-1996-5_18)
21. Hecht-Nielsen R. 1994 Context vectors: General-purpose approximate meaning
representations self-organized from raw data. In _Zurada J, Marks R, Robinson C, eds._
_Computational Intelligence: Imitating Life_ . New York: IEEE Press, 43–56.
22. Gorban AN, Romanenko I, Burton R, Tyukin I. 2016 _One-trial correction of legacy AI systems and_
_stochastic separation theorems._ [See arXiv:1610.00494 [stat.ML]](https://arxiv.org/abs/1610.00494)
23. Pestov V. 2013 Is the _k_ -NN classifier in high dimensions affected by the curse of
dimensionality? _Comput. Math. Appl._ **65** [, 1427–1437. (doi:10.1016/j.camwa.2012.09.011)](https://doi.org/10.1016/j.camwa.2012.09.011)
24. Cucker F, Smale S. 2002 On the mathematical foundations of learning. _Bull. Amer. Math. Soc._,
**39** [, 1–49. (doi:10.1090/S0273-0979-01-00923-5)](https://doi.org/10.1090/S0273-0979-01-00923-5)
25. Friedman J, Hastie T, Tibshirani R. 2009 _The Elements of Statistical Learning_ . New York: Springer.
[(doi:10.1007/978-0-387-84858-7)](https://doi.org/10.1007/978-0-387-84858-7)
26. Vapnik V. 2000 _The Nature of Statistical Learning Theory_ [. New York: Springer. (doi:10.1007/978-](https://doi.org/10.1007/978-1-4757-3264-1)
[1-4757-3264-1)](https://doi.org/10.1007/978-1-4757-3264-1)
27. Xu R, Wunsch D. 2008. _Clustering_ [. Hoboken: John Wiley & Sons. (doi:10.1002/9780470382776)](http://doi.org/10.1002/9780470382776)
28. Gorban AN, Kégl B, Wunsch D, Zinovyev A. (Eds.) 2008 _Principal Manifolds for Data_
_Visualisation and Dimension Reduction_ . Lect. Notes Comput. Sci. Eng., Vol. 58. Berlin –
[Heidelberg: Springer. (doi:10.1007/978-3-540-73750-6)](https://doi.org/10.1007/978-3-540-73750-6)
29. Gorban AN, Zinovyev A. 2010 Principal manifolds and graphs in practice:
from molecular biology to dynamical systems. _Int._ _J._ _Neural_ _Syst._ **20**, 219–232.
[(doi:10.1142/S0129065710002383)](https://doi.org/10.1142/S0129065710002383)
30. Hyvärinen A, Oja E. 2000 Independent component analysis: algorithms and applications.
_Neural Netw._ **13** [, 41–430. (doi:10.1016/S0893-6080(00)00026-5)](https://doi.org/10.1016/S0893-6080(00)00026-5)
31. Mirkin B. 2012 _Clustering:_ _A_ _Data_ _Recovery_ _Approach._ Boca Raton: CRC Press.
[(doi:10.1201/b13101)](https://doi.org/10.1201/b13101)
32. Gorban AN, Mirkes EM, Zinovyev A. 2016. Piece-wise quadratic approximations of arbitrary
error functions for fast and robust machine learning. _Neural Netw._ **84** (2016), 28–38.
[(doi:10.1016/j.neunet.2016.08.007)](https://doi.org/10.1016/j.neunet.2016.08.007)
33. Gromov M. 2011 _Structures, Learning and Ergosystems: Chapters 1-4, 6._ IHES, Bures-sur-Ivette,
Île-de-France. See http://www.ihes.fr/ _∼_ [gromov/PDF/ergobrain.pdf.](http://www.ihes.fr/~gromov/PDF/ergobrain.pdf)
34. Engel A, Van den Broeck C. 2001 _Statistical Mechanics of Learning._ Cambridge, UK: Cambridge
University Press.
35. Wang D, Li M. 2017 Stochastic configuration networks: Fundamentals and algorithms. _IEEE_
_Trans. On Cybernetics_ **47** [, 3466–3479. (doi:10.1109/TCYB.2017.2734043)](https://doi.org/10.1109/TCYB.2017.2734043)
36. Gorban AN. 1990 _Training_ _Neural_ _Networks_, Moscow: USSR-USA JV “ParaGraph”.
[(doi:10.13140/RG.2.1.1784.4724)](https://doi.org/10.13140/RG.2.1.1784.4724)
37. De Freitas N, Andrieu C, Højen-Sørensen P, Niranjan M, Gee A. 2001 Sequential Monte
Carlo methods for neural networks. In _Sequential Monte Carlo Methods in Practice_ . New York:
[Springer, 359–379. (doi:10.1007/978-1-4757-3437-9_17)](https://doi.org/10.1007/978-1-4757-3437-9_17)
38. Scardapane S, Wang D. 2017 Randomness in neural networks: an overview. _WIREs Data_
_Mining Knowl. Discov._ **7** [, e1200. (doi:doi.org/10.1002/widm.1200)](https://doi.org/10.1002/widm.1200)
39. Fisher RA. 1936 The use of multiple measurements in taxonomic problems. _Ann. Hum. Genet._
**7** [, 179–188. (doi:10.1111/j.1469-1809.1936.tb02137.x)](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)
40. Rosenblatt F. 1962 _Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms._
[Washington DC: Spartan Books. See http://www.dtic.mil/docs/citations/AD0256582.](http://www.dtic.mil/docs/citations/AD0256582)
41. Duda RD, Hart PE, and Stork DG. 2012 _Pattern classification_ . New York: John Wiley and Sons.
42. Aggarwal CC. 2015 _Data Mining: The Textbook_ . Cham – Heidelberg – New York – Dordrecht –
[London: Springer. (doi:10.1007/978-3-319-14142-8)](https://doi.org/10.1007/978-3-319-14142-8)
43. K˚urková V, Sanguineti M. 2017 Probabilistic lower bounds for approximation by shallow
perceptron networks. _Neural Netw._ **91** [, 34–41. (doi:10.1016/j.neunet.2017.04.003)](https://doi.org/10.1016/j.neunet.2017.04.003)
44. Johnson WB, Lindenstrauss J. 1984 Extensions of Lipschitz mappings into a Hilbert space.
_Contemp. Math._ **26** [, 189–206. (doi:10.1090/conm/026/737400)](https://doi.org/10.1090/conm/026/737400)



**16**


45. Dasgupta S, Gupta A. 2003 An elementary proof of a theorem of Johnson and Lindenstrauss.
_Random Structures & Algorithms_ **22** [, 60–65. (doi:10.1002/rsa.10073)](https://doi.org/10.1002/rsa.10073)
46. Gorban AN, Tyukin I, Prokhorov D, Sofeikov K. 2016 Approximation with random bases: Pro
et contra. _Inf. Sci._ **364–365** [, 129–145. (doi:10.1016/j.ins.2015.09.021)](https://doi.org/10.1016/j.ins.2015.09.021)
47. Kainen P, K˚urková V. 1993 Quasiorthogonal dimension of Euclidian spaces. _Appl. Math. Lett._
**6** [, 7–10. (doi:10.1016/0893-9659(93)90023-G)](https://doi.org/10.1016/0893-9659(93)90023-G)
48. Gorban AN, Tyukin IY, Romanenko I. 2016. The blessing of dimensionality:
Separation theorems in the thermodynamic limit. _IFAC-PapersOnLine_ **49**, 64–69. See
[doi:10.1016/j.ifacol.2016.10.755).](https://doi.org/10.1016/j.ifacol.2016.10.755)
49. Gorban AN, Tyukin IY. 2017 Stochastic separation theorems. _Neural Netw._ **94**, 255–259.
[(doi:10.1016/j.neunet.2017.07.014)](https://doi.org/10.1016/j.neunet.2017.07.014)
50. Chapelle O. 2007 Training a Support Vector Machine in the Primal. _Neural Comput._ **19**, 1155–
[1178. doi:10.1162/neco.2007.19.5.1155)](https://doi.org/10.1162/neco.2007.19.5.1155)
51. Hoeffding W. 1963. Probability inequalities for sums of bounded random variables. _J. Amer._
_Statist. Assoc._ **301** [, 13–30. (doi:10.1080/01621459.1963.10500830)](https://doi.org/10.1080/01621459.1963.10500830)
52. Tyukin IY, Gorban AN, Sofeikov K, Romanenko I. 2017 _Knowledge transfer between artificial_
_intelligence systems._ [See arXiv:1709.01547 [cs.AI].](https://arxiv.org/abs/1709.01547)
53. Jia Y. 2013 _Caffe: An open source convolutional architecture for fast feature embedding_ . See

[http://caffe.berkeleyvision.org/.](http://caffe.berkeleyvision.org/)
54. Chen T, Li M, Li Y, Lin M, Wang N, Xiao T, Xu B, Zhang C, Zhang Z. 2015 _MXNet:_
_A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems_ . See
[https://github.com/dmlc/mxnet.](https://github.com/dmlc/mxnet)
55. Team DD. 2016 _Deeplearning4j: Open-source distributed deep learning for the JVM_, Apache
[Software Foundation License 2.0. See http://deeplearning4j.org.](http://deeplearning4j.org)
56. Abadi M, Agarwal A, Barham P et al. 2015 _TensorFlow: Large-scale machine learning on_
_heterogeneous systems_ . Software available from An open-source software library for Machine
[Intelligence. See https://www.tensorflow.org/.](https://www.tensorflow.org/)
57. Buchtala O, Sick B. 2007 Basic technologies for knowledge transfer in intelligent systems. In
_Artificial Life_, ALIFE’07. New York: IEEE Press. 251–258. (10.1109/ALIFE.2007.367804)
58. Tyukin IY, Gorban AN, Calvo C, Makarova J, Makarov VA. 2017 _High-dimensional brain. A tool_
_for encoding and rapid learning of memories by single neurons._ [See arXiv:1710.11227 [q-bio.NC].](https://arxiv.org/abs/1710.11227)
59. Ison MJ, Quian Quiroga R, Fried I. 2015 Rapid encoding of new memories by individual
neurons in the human brain. _Neuron_ **87** [:220–230. (doi:10.1016/j.neuron.2015.06.016)](https://doi.org/10.1016/j.neuron.2015.06.016)
60. Quian Quiroga R, Reddy, L, Kreiman, G, Koch, C, Fried, I. 2005 Invariant visual representation
by single neurons in the human brain. _Nature_ **435** [, 1102–1107. (doi:10.1038/nature03687)](https://doi.org/10.1038/nature03687)
61. Viskontas IV, Quian Quiroga R, Fried, I. 2009. Human medial temporal lobe neurons
respond preferentially to personally relevant images. _Proc. Nat. Acad. Sci._ **106**, 21329–21334.
[(doi:10.1073/pnas.0902319106)](https://doi.org/10.1073/pnas.0902319106)
62. Quian Quiroga, R. 2012 Concept cells: the building blocks of declarative memory functions.
_Nat. Rev. Neurosci._ **13** [, 587–597. (doi:10.1038/nrn3251)](https://doi.org/10.1038/nrn3251)
63. Oja E. 1982 A simplified neuron model as a principal component analyzer. _J. Math. Biol._ **15**,
[267–273. (doi:10.1007/BF00275687)](https://doi.org/10.1007/BF00275687)
64. Corry L. 1997 David Hilbert and the axiomatization of physics (1894–1905). _Arch. Hist. Exact_
_Sci._ **51** [, 83–198. (doi:10.1007/BF00375141)](https://doi.org/10.1007/BF00375141)
65. Wightman AS. 1976 Hilbert’s sixth problem: Mathematical treatment of the axioms of physics.
In _Browder FE (ed.). Mathematical Developments Arising from Hilbert Problems. Proceedings of_
_Symposia in Pure Mathematics. XXVIII._ [AMS, 147–240. (doi:10.1090/pspum/028.1/0436800)](https://doi.org/10.1090/pspum/028.1/0436800)
66. Kégl B. 2003 Intrinsic dimension estimation using packing numbers. In _Advances in neural_
_information processing systems_ 15 (NIPS 2002), Cambridge, US: MIT Press, 697–704.
67. Levina E, Bickel PJ. 2005 Maximum likelihood estimation of intrinsic dimension. In _Advances_
_in neural information processing systems_ 17 (NIPS 2004). Cambridge, US: MIT Press, 777–784.
68. Bengio Y. 2009 Learning deep architectures for AI. _Found. Trends Mach. Learn._ **2**, 1–127.
[(doi:10.1561/2200000006)](https://doi.org/10.1561/2200000006)
69. Moczko E, Mirkes EM, Ceceres C, Gorban AN, Piletsky S. 2016 Fluorescence-based assay as a
new screening tool for toxic chemicals. _Sci. Rep._ **6** [, 33922. (doi:10.1038/srep33922)](https://doi.org/10.1038/srep33922)
70. Cangelosi R, Goriely A. 2007 Component retention in principal component analysis with
application to cDNA microarray data. _Biol. Direct_ **2** [, 2. (doi:10.1186/1745-6150-2-2)](https://doi.org/10.1186/1745-6150-2-2)
71. Zinovyev A, Mirkes E. 2013. Data complexity measured by principal graphs. _Comput. Math._
_Appl._ **65** [, 1471–1482. (doi:10.1016/j.camwa.2012.12.009)](https://doi.org/10.1016/j.camwa.2012.12.009)



**17**


