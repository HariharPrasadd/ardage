## Streaming Weak Submodularity: Interpreting Neural Networks on the Fly

Ethan R. Elenberg [1], Alexandros G. Dimakis [1], Moran Feldman [2], and Amin Karbasi [3]


1 Department of Electrical and Computer Engineering
The University of Texas at Austin
`elenberg@utexas.edu`, `dimakis@austin.utexas.edu`
2 Department of Mathematics and Computer Science
Open University of Israel
```
               moranfe@openu.ac.il
```

3 Department of Electrical Engineering, Department of Computer Science
Yale University
```
              amin.karbasi@yale.edu

```

November 27, 2017


**Abstract**


In many machine learning applications, it is important to explain the predictions of a black-box
classifier. For example, why does a deep neural network assign an image to a particular class? We cast
interpretability of black-box classifiers as a combinatorial maximization problem and propose an efficient
streaming algorithm to solve it subject to cardinality constraints. By extending ideas from Badanidiyuru
et al. [2014], we provide a constant factor approximation guarantee for our algorithm in the case of
random stream order and a _weakly submodular_ objective function. This is the first such theoretical
guarantee for this general class of functions, and we also show that no such algorithm exists for a worst
case stream order. Our algorithm obtains similar explanations of Inception V3 predictions 10 times faster
than the state-of-the-art LIME framework of Ribeiro et al. [2016].

### **1 Introduction**


Consider the following combinatorial optimization problem. Given a ground set _N_ of _N_ elements and a set
function _f_ : 2 _[N]_ _�→_ R _≥_ 0, find the set _S_ of size _k_ which maximizes _f_ ( _S_ ). This formulation is at the heart
of many machine learning applications such as sparse regression, data summarization, facility location, and
graphical model inference. Although the problem is intractable in general, if _f_ is assumed to be _submodular_
then many approximation algorithms have been shown to perform provably within a constant factor from
the best solution.
Some disadvantages of the standard greedy algorithm of Nemhauser et al. [1978] for this problem are
that it requires repeated access to each data element and a large total number of function evaluations.
This is undesirable in many large-scale machine learning tasks where the entire dataset cannot fit in main
memory, or when a single function evaluation is time consuming. In our main application, each function
evaluation corresponds to inference on a large neural network and can take a few seconds. In contrast,
streaming algorithms make a small number of passes (often only one) over the data and have sublinear space
complexity, and thus, are ideal for tasks of the above kind.
Recent ideas, algorithms, and techniques from submodular set function theory have been used to derive
similar results in much more general settings. For example, Elenberg et al. [2016a] used the concept of _weak_
_submodularity_ to derive approximation and parameter recovery guarantees for nonlinear sparse regression.


1


Thus, a natural question is whether recent results on streaming algorithms for maximizing submodular
functions [Badanidiyuru et al., 2014; Buchbinder et al., 2015; Chekuri et al., 2015] extend to the weakly
submodular setting.
This paper answers the above question by providing the first analysis of a streaming algorithm for
any class of approximately submodular functions. We use key algorithmic components of Sieve-Streaming

[Badanidiyuru et al., 2014], namely greedy thresholding and binary search, combined with a novel analysis to
prove a constant factor approximation for _γ_ -weakly submodular functions (defined in Section 3). Specifically,
our contributions are as follows.


_•_ An impossibility result showing that, even for 0 _._ 5-weakly submodular objectives, no randomized streaming algorithm which uses _o_ ( _N_ ) memory can have a constant approximation ratio when the ground set
elements arrive in a worst case order.




_•_ Streak: a greedy, deterministic streaming algorithm for maximizing _γ_ -weakly submodular functions
which uses _O_ ( _ε_ _[−]_ [1] _k_ log _k_ ) memory and has an approximation ratio of (1 _−ε_ ) _[γ]_ _[·]_ [(3] _[−][e]_ _[−][γ/]_ [2] _[−]_ [2] _√_ 2 _−_ _e_ _[−][γ/]_ [2] )




_[γ]_ 2 _[·]_ [(3] _[−][e]_ _[−][γ/]_ [2] _[−]_ [2] _√_



which uses _O_ ( _ε_ _[−]_ [1] _k_ log _k_ ) memory and has an approximation ratio of (1 _−ε_ ) _[γ]_ 2 _[·]_ [(3] _[−][e]_ _[−][γ/]_ [2] _[−]_ [2] _√_ 2 _−_ _e_ _[−][γ/]_ [2] )

when the ground set elements arrive in a random order.




_•_ An experimental evaluation of our algorithm in two applications: nonlinear sparse regression using
pairwise products of features and interpretability of black-box neural network classifiers.


The above theoretical impossibility result is quite surprising since it stands in sharp contrast to known
streaming algorithms for submodular objectives achieving a constant approximation ratio even for worst
case stream order.

One advantage of our approach is that, while our approximation guarantees are in terms of _γ_, our
algorithm Streak runs without requiring prior knowledge about the value of _γ_ . This is important since
the weak submodularity parameter _γ_ is hard to compute, especially in streaming applications, as a single
element can alter _γ_ drastically.
We use our streaming algorithm for neural network interpretability on Inception V3 [Szegedy et al.,
2016]. For that purpose, we define a new set function maximization problem similar to LIME [Ribeiro et al.,
2016] and apply our framework to approximately maximize this function. Experimentally, we find that our
interpretability method produces explanations of similar quality as LIME, but runs approximately 10 times
faster.

### **2 Related Work**


Monotone submodular set function maximization has been well studied, starting with the classical analysis
of greedy forward selection subject to a matroid constraint [Nemhauser et al., 1978; Fisher et al., 1978].
For the special case of a uniform matroid constraint, the greedy algorithm achieves an approximation ratio
of 1 _−_ [1] _/_ _e_ [Fisher et al., 1978], and a more involved algorithm obtains this ratio also for general matroid
constraints [Călinescu et al., 2011]. In general, no polynomial-time algorithm can have a better approximation
ratio even for a uniform matroid constraint [Nemhauser and Wolsey, 1978; Feige, 1998]. However, it is possible
to improve upon this bound when the data obeys some additional guarantees [Conforti and Cornuéjols, 1984;
Vondrák, 2010; Sviridenko et al., 2015]. For maximizing nonnegative, not necessarily monotone, submodular
functions subject to a general matroid constraint, the state-of-the-art randomized algorithm achieves an
approximation ratio of 0 _._ 385 [Buchbinder and Feldman, 2016b]. Moreover, for uniform matroids there is also
a deterministic algorithm achieving a slightly worse approximation ratio of [1] _/_ _e_ [Buchbinder and Feldman,
2016a]. The reader is referred to Bach [2013] and Krause and Golovin [2014] for surveys on submodular
function theory.
A recent line of work aims to develop new algorithms for optimizing submodular functions suitable for
large-scale machine learning applications. Algorithmic advances of this kind include Stochastic-Greedy

[Mirzasoleiman et al., 2015], Sieve-Streaming [Badanidiyuru et al., 2014], and several distributed approaches [Mirzasoleiman et al., 2013; Barbosa et al., 2015, 2016; Pan et al., 2014; Khanna et al., 2017b]. Our
algorithm extends ideas found in Sieve-Streaming and uses a different analysis to handle more general
functions. Additionally, submodular set functions have been used to prove guarantees for online and active


2


learning problems [Hoi et al., 2006; Wei et al., 2015; Buchbinder et al., 2015]. Specifically, in the online setting corresponding to our setting ( _i.e._, maximizing a monotone function subject to a cardinality constraint),
Chan et al. [2017] achieve a competitive ratio of about 0 _._ 3178 when the function is submodular.
The concept of weak submodularity was introduced in Krause and Cevher [2010]; Das and Kempe [2011],
where it was applied to the specific problem of feature selection in linear regression. Their main results state
that if the data covariance matrix is not too correlated (using either incoherence or restricted eigenvalue
assumptions), then maximizing the goodness of fit _f_ ( _S_ ) = _R_ _S_ [2] [as a function of the feature set] _[ S]_ [ is weakly]
submodular. This leads to constant factor approximation guarantees for several greedy algorithms. Weak
submodularity was connected with Restricted Strong Convexity in Elenberg et al. [2016a,b]. This showed that
the same assumptions which imply the success of regularization also lead to guarantees on greedy algorithms.
This framework was later used for additional algorithms and applications [Khanna et al., 2017a,b]. Other
approximate versions of submodularity were used for greedy selection problems in Horel and Singer [2016];
Hassidim and Singer [2017]; Altschuler et al. [2016]; Bian et al. [2017]. To the best of our knowledge, this is
the first analysis of streaming algorithms for approximately submodular set functions.
Increased interest in interpretable machine learning models has led to extensive study of sparse feature
selection methods. For example, Bahmani et al. [2013] consider greedy algorithms for logistic regression,
and Yang et al. [2016] solve a more general problem using _ℓ_ 1 regularization. Recently, Ribeiro et al. [2016]
developed a framework called LIME for interpreting black-box neural networks, and Sundararajan et al.

[2017] proposed a method that requires access to the network’s gradients with respect to its inputs. We
compare our algorithm to variations of LIME in Section 6.2.

### **3 Preliminaries**


First we establish some definitions and notation. Sets are denoted with capital letters, and all big O notation
is assumed to be scaling with respect to _N_ (the number of elements in the input stream). Given a set function
_f_, we often use the discrete derivative _f_ ( _B | A_ ) ≜ _f_ ( _A ∪_ _B_ ) _−_ _f_ ( _A_ ). _f_ is monotone if _f_ ( _B | A_ ) _≥_ 0 _, ∀A, B_
and nonnegative if _f_ ( _A_ ) _≥_ 0 _, ∀A_ . Using this notation one can define weakly submodular functions based on
the following ratio.


**Definition 3.1** (Weak Submodularity, adapted from Das and Kempe [2011]) **.** _A monotone nonnegative set_
_function f_ : 2 _[N]_ _�→_ R _≥_ 0 _is called γ-weakly submodular for an integer r if_



�



_γ ≤_ _γ_ _r_ ≜ min
_L,S⊆N_ :
_|L|,|S\L|≤r_



_j∈S\L_ _[f]_ [(] _[j][ |][ L]_ [)]



_f_ ( _S | L_ ) _,_



_where the ratio is considered to be equal to_ 1 _when its numerator and denominator are both_ 0 _._


This generalizes submodular functions by relaxing the _diminishing returns_ property of discrete derivatives.
It is easy to show that _f_ is submodular if and only if _γ_ _|N |_ = 1.


**Definition 3.2** (Approximation Ratio) **.** _A streaming maximization algorithm_ ALG _which returns a set S_
_has approximation ratio R ∈_ [0 _,_ 1] _if_ E[ _f_ ( _S_ )] _≥_ _R · f_ ( _OPT_ ) _, where OPT is the optimal solution and the_
_expectation is over the random decisions of the algorithm and the randomness of the input stream order_
_(when it is random)._


Formally our problem is as follows. Assume that elements from a ground set _N_ arrive in a stream at
either random or worst case order. The goal is then to design a one pass streaming algorithm that given
oracle access to a nonnegative set function _f_ : 2 _[N]_ _�→_ R _≥_ 0 maintains at most _o_ ( _N_ ) elements in memory and
returns a set _S_ of size at most _k_ approximating


max
_|T |≤k_ _[f]_ [(] _[T]_ [)] _[,]_


up to an approximation ratio _R_ ( _γ_ _k_ ). Ideally, this approximation ratio should be as large as possible, and we
also want it to be a function of _γ_ _k_ and nothing else. In particular, we want it to be independent of _k_ and _N_ .
To simplify notation, we use _γ_ in place of _γ_ _k_ in the rest of the paper. Additionally, **proofs for all our**
**theoretical results are deferred to the Appendix.**


3


### **4 Impossibility Result**

To prove our negative result showing that no streaming algorithm for our problem has a constant approximation ratio against a worst case stream order, we first need to construct a weakly submodular set function
_f_ _k_ . Later we use it to construct a bad instance for any given streaming algorithm.
Fix some _k ≥_ 1, and consider the ground set _N_ _k_ = _{u_ _i_ _, v_ _i_ _}_ _[k]_ _i_ =1 [. For ease of notation, let us define for]
every subset _S ⊆N_ _k_
_u_ ( _S_ ) = _|S ∩{u_ _i_ _}_ _[k]_ _i_ =1 _[|][,]_ _v_ ( _S_ ) = _|S ∩{v_ _i_ _}_ _[k]_ _i_ =1 _[|][ .]_


Now we define the following set function:


_f_ _k_ ( _S_ ) = min _{_ 2 _· u_ ( _S_ ) + 1 _,_ 2 _· v_ ( _S_ ) _}_ _∀_ _S ⊆N_ _k_ _._


**Lemma 4.1.** _f_ _k_ _is nonnegative, monotone and_ 0 _._ 5 _-weakly submodular for the integer |N_ _k_ _|._


Since _|N_ _k_ _|_ = 2 _k_, the maximum value of _f_ _k_ is _f_ _k_ ( _N_ _k_ ) = 2 _· v_ ( _N_ _k_ ) = 2 _k_ . We now extend the ground set
of _f_ _k_ by adding to it an arbitrary large number _d_ of dummy elements which do not affect _f_ _k_ at all. Clearly,
this does not affect the properties of _f_ _k_ proved in Lemma 4.1. However, the introduction of dummy elements
allows us to assume that _k_ is an arbitrary small value compared to _N_, which is necessary for the proof of
the next theorem. In a nutshell, this proof is based on the observation that the elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [are]
indistinguishable from the dummy elements as long as no element of _{v_ _i_ _}_ _[k]_ _i_ =1 [has arrived yet.]


**Theorem 4.2.** _For every constant c ∈_ (0 _,_ 1] _there is a large enough k such that no randomized streaming_
_algorithm that uses o_ ( _N_ ) _memory to solve_ max _|S|≤_ 2 _k_ _f_ _k_ ( _S_ ) _has an approximation ratio of c for a worst case_
_stream order._


We note that _f_ _k_ has strong properties. In particular, Lemma 4.1 implies that it is 0 _._ 5-weakly submodular
for every 0 _≤_ _r ≤|N|_ . In contrast, the algorithm we show later assumes weak submodularity only for the
cardinality constraint _k_ . Thus, the above theorem implies that worst case stream order precludes a constant
approximation ratio even for functions with much stronger properties compared to what is necessary for
getting a constant approximation ratio when the order is random.
The proof of Theorem 4.2 relies critically on the fact that each element is seen exactly once. In other
words, once the algorithm decides to discard an element from its memory, this element is gone forever, which
is a standard assumption for streaming algorithms. Thus, the theorem does not apply to algorithms that
use multiple passes over _N_, or non-streaming algorithms that use _o_ ( _N_ ) writable memory, and their analysis
remains an interesting open problem.

### **5 Streaming Algorithms**


In this section we give a deterministic streaming algorithm for our problem which works in a model in
which the stream contains the elements of _N_ in a random order. We first describe in Section 5.1 such a
streaming algorithm assuming access to a value _τ_ which approximates _aγ · f_ ( _OPT_ ), where _a_ is a shorthand
for _a_ = ( _√_ 2 _−_ _e_ _[−][γ/]_ [2] _−_ 1) _/_ 2. Then, in Section 5.2 we explain how this assumption can be removed to obtain

Streak and bound its approximation ratio, space complexity, and running time.


**5.1** **Algorithm with access to** _τ_


Consider Algorithm 1. In addition to the input instance, this algorithm gets a parameter _τ ∈_ [0 _, aγ_ _·f_ ( _OPT_ )].
One should think of _τ_ as close to _aγ · f_ ( _OPT_ ), although the following analysis of the algorithm does not
rely on it. We provide an outline of the proof, but defer the technical details to the Appendix.


**Theorem 5.1.** _The expected value of the set produced by Algorithm 1 is at least_



_τ_ _√_
_a_ _[·]_ [ 3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ 2 [2]



2 _−_ _e_ _[−][γ/]_ [2]



2 _−_ _e_ _[−][γ/]_ [2] _−_ 1) _._



_−_ _e_

= _τ ·_ ( ~~�~~
2



4


**Algorithm 1** Threshold Greedy( _f, k, τ_ )

Let _S ←_ ∅.

**while** there are more elements **do**


Let _u_ be the next element.
**if** _|S| < k_ and _f_ ( _u | S_ ) _≥_ _τ/k_ **then**

Update _S ←_ _S ∪{u}_ .
**end if**

**end while**

**return:** _S_


_Proof (Sketch)._ Let _E_ be the event that _f_ ( _S_ ) _< τ_, where _S_ is the output produced by Algorithm 1. Clearly
_f_ ( _S_ ) _≥_ _τ_ whenever _E_ does not occur, and thus, it is possible to lower bound the expected value of _f_ ( _S_ )
using _E_ as follows.


**Observation 5.2.** _Let S denote the output of Algorithm 1, then_ E[ _f_ ( _S_ )] _≥_ (1 _−_ Pr[ _E_ ]) _· τ_ _._


The lower bound given by Observation 5.2 is decreasing in Pr[ _E_ ]. Proposition 5.4 provides another lower
bound for E[ _f_ ( _S_ )] which increases with Pr[ _E_ ]. An important ingredient of the proof of this proposition is
the next observation, which implies that the solution produced by Algorithm 1 is always of size smaller than
_k_ when _E_ happens.


**Observation 5.3.** _If at some point Algorithm 1 has a set S of size k, then f_ ( _S_ ) _≥_ _τ_ _._


The proof of Proposition 5.4 is based on the above observation and on the observation that the random
arrival order implies that every time that an element of _OPT_ arrives in the stream we may assume it is a
random element out of all the _OPT_ elements that did not arrive yet.


**Proposition 5.4.** _For the set S produced by Algorithm 1,_


E[ _f_ ( _S_ )] _≥_ [1] _γ ·_ [Pr[ _E_ ] _−_ _e_ _[−][γ/]_ [2] ] _· f_ ( _OPT_ ) _−_ 2 _τ_ _._

2 _[·]_ � �



The theorem now follows by showing that for every possible value of Pr[ _E_ ] the guarantee of the theorem
is implied by either Observation 5.2 or Proposition 5.4. Specifically, the former happens when Pr[ _E_ ] _≤_
2 _−_ _√_ 2 _−_ _e_ _[−][γ/]_ [2] and the later when Pr[ _E_ ] _≥_ 2 _−_ _√_ 2 _−_ _e_ _[−][γ/]_ [2] .



2 _−_ _e_ _[−][γ/]_ [2] and the later when Pr[ _E_ ] _≥_ 2 _−_ _√_



2 _−_ _e_ _[−][γ/]_ [2] .



**5.2** **Algorithm without access to** _τ_


In this section we explain how to get an algorithm which does not depend on _τ_ . Instead, Streak (Algorithm 2) receives an accuracy parameter _ε ∈_ (0 _,_ 1). Then, it uses _ε_ to run several instances of Algorithm 1
stored in a collection denoted by _I_ . The algorithm maintains two variables throughout its execution: _m_ is
the maximum value of a singleton set corresponding to an element that the algorithm already observed, and
_u_ _m_ references an arbitrary element satisfying _f_ ( _u_ _m_ ) = _m_ .
The collection _I_ is updated as follows after each element arrival. If previously _I_ contained an instance
of Algorithm 1 with a given value for _τ_, and it no longer should contain such an instance, then the instance
is simply removed. In contrast, if _I_ did not contain an instance of Algorithm 1 with a given value for _τ_,
and it should now contain such an instance, then a new instance with this value for _τ_ is created. Finally, if
_I_ contained an instance of Algorithm 1 with a given value for _τ_, and it should continue to contain such an
instance, then this instance remains in _I_ as is.


**Theorem 5.5.** _The approximation ratio of_ Streak _is at least_



_√_
(1 _−_ _ε_ ) _γ ·_ [3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ [2]



2 _−_ _e_ _[−][γ/]_ [2]



2 _._



5


**Algorithm 2** Streak( _f, k, ε_ )
Let _m ←_ 0, and let _I_ be an (originally empty) collection of instances of Algorithm 1.
**while** there are more elements **do**


Let _u_ be the next element.
**if** _f_ ( _u_ ) _≥_ _m_ **then**

Update _m ←_ _f_ ( _u_ ) and _u_ _m_ _←_ _u_ .
**end if**
Update _I_ so that it contains an instance of Algorithm 1 with _τ_ = _x_ for every _x ∈{_ (1 _−_ _ε_ ) _[i]_ _| i ∈_
Z and (1 _−_ _ε_ ) _m/_ (9 _k_ [2] ) _≤_ (1 _−_ _ε_ ) _[i]_ _≤_ _mk}_, as explained in Section 5.2.
Pass _u_ to all instances of Algorithm 1 in _I_ .
**end while**

**return:** the best set among all the outputs of the instances of Algorithm 1 in _I_ and the singleton set
_{u_ _m_ _}_ .


The proof of Theorem 5.5 shows that in the final collection _I_ there is an instance of Algorithm 1 whose _τ_
provides a good approximation for _aγ · f_ ( _OPT_ ), and thus, this instance of Algorithm 1 should (up to some
technical details) produce a good output set in accordance with Theorem 5.1.
It remains to analyze the space complexity and running time of Streak. We concentrate on bounding
the number of elements Streak keeps in its memory at any given time, as this amount dominates the space
complexity as long as we assume that the space necessary to keep an element is at least as large as the space
necessary to keep each one of the numbers used by the algorithm.


**Theorem 5.6.** _The space complexity of_ Streak _is O_ ( _ε_ _[−]_ [1] _k_ log _k_ ) _elements._


The running time of Algorithm 1 is _O_ ( _Nf_ ) where, abusing notation, _f_ is the running time of a single
oracle evaluation of _f_ . Therefore, the running time of Streak is _O_ ( _Nfε_ _[−]_ [1] log _k_ ) since it uses at every given
time only _O_ ( _ε_ _[−]_ [1] log _k_ ) instances of the former algorithm. Given multiple threads, this can be improved to
_O_ ( _Nf_ + _ε_ _[−]_ [1] log _k_ ) by running the _O_ ( _ε_ _[−]_ [1] log _k_ ) instances of Algorithm 1 in parallel.

### **6 Experiments**


We evaluate the performance of our streaming algorithm on two sparse feature selection applications. [1]

Features are passed to all algorithms in a random order to match the setting of Section 5.



600


400


200


0
**Random** **Streak(0.75)** **Streak(0.1)** **Local Search**


1.00



15000


10000


5000


0 **Random** **Streak(0.75)** **Streak(0.1)** **Local Search**


400000


300000


200000


100000


0 **Random** **Streak(0.75)** **Streak(0.1)** **Local Search**



(b) Cost





Figure 1: Logistic Regression, Phishing dataset with pairwise feature products. Our algorithm is comparable
to LocalSearch in both log likelihood and generalization accuracy, with much lower running time and
number of model fits in most cases. Results averaged over 40 iterations, error bars show 1 standard deviation.


6


|Accu<br>0.95<br>0.90<br>Generalization<br>0.85<br>0.80<br>0.75<br>0.70 Random Streak(0.75) Streak(0.1) Local Search<br>k=20 k=40 k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model fits in most cases. Results averaged<br>1Code for these experiments is available at https://github|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Random**<br>**Streak(0.75)**<br>**Streak(0.1)**<br>**Local Search**<br>0.70<br>0.75<br>0.80<br>0.85<br>0.90<br>0.95<br><br>**Generalization Accu**<br>k=20<br>k=40<br>k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model ﬁts in most cases. Results averaged<br>1Code for these experiments is available at` https://github`|||||||||||||||||||
|**Random**<br>**Streak(0.75)**<br>**Streak(0.1)**<br>**Local Search**<br>0.70<br>0.75<br>0.80<br>0.85<br>0.90<br>0.95<br><br>**Generalization Accu**<br>k=20<br>k=40<br>k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model ﬁts in most cases. Results averaged<br>1Code for these experiments is available at` https://github`|||||||||||||||||||
|**Random**<br>**Streak(0.75)**<br>**Streak(0.1)**<br>**Local Search**<br>0.70<br>0.75<br>0.80<br>0.85<br>0.90<br>0.95<br><br>**Generalization Accu**<br>k=20<br>k=40<br>k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model ﬁts in most cases. Results averaged<br>1Code for these experiments is available at` https://github`||**Ra**<br>gr<br> b<br>in|**ndo**<br>es<br>ot<br> m|**m**|**S**<br>|**S**<br>|**tre**<br>|**ak(0**<br>|**.75**|**)**<br><br>|**Stre**<br>|**ak(**|**0.1)**|**L**<br>|**oca**<br>|**l Se**<br> w<br>ge<br>ra<br>`/g`|**arc**<br>it<br>ne<br>ge<br>`it`|**h**<br>h<br>ra<br>d<br>`hub`|
|**Random**<br>**Streak(0.75)**<br>**Streak(0.1)**<br>**Local Search**<br>0.70<br>0.75<br>0.80<br>0.85<br>0.90<br>0.95<br><br>**Generalization Accu**<br>k=20<br>k=40<br>k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model ﬁts in most cases. Results averaged<br>1Code for these experiments is available at` https://github`||**Ra**<br>gr<br> b<br>in|**ndo**<br>es<br>ot<br> m||||=20|||k|=40|||k=|80|80|80|80|
|**Random**<br>**Streak(0.75)**<br>**Streak(0.1)**<br>**Local Search**<br>0.70<br>0.75<br>0.80<br>0.85<br>0.90<br>0.95<br><br>**Generalization Accu**<br>k=20<br>k=40<br>k=80<br>(a) Performance<br>Figure 1: Logistic Regression, Phishing dataset with<br>to LocalSearch in both log likelihood and genera<br>number of model ﬁts in most cases. Results averaged<br>1Code for these experiments is available at` https://github`||**Ra**<br>gr<br> b<br>in|**ndo**<br>es<br>ot<br> m|si<br>h<br>os|(a<br>on<br> lo<br>t|(a<br>on<br> lo<br>t|)<br>, P<br>g<br>ca|Per<br>h<br> li<br>se|fo<br>is<br>ke<br>s.|rm<br>hi<br>lih<br>R|an<br>ng<br>o<br>es|ce<br> d<br>od<br>ul|at<br> a<br>ts|as<br>n<br> a<br>`ps`|et<br>d<br>ve<br>`:/`|et<br>d<br>ve<br>`:/`|et<br>d<br>ve<br>`:/`|et<br>d<br>ve<br>`:/`|
|1Code for these exper|1Code for these exper|im|en|ts|is|is|av|ail|ab|le|at|` h`|`tt`|`tt`|`tt`|`tt`|`tt`|`tt`|


700


650


600


550





50010 [0] 10 [1] 10 [2] 10 [3] 10 [4] 10 [5]

**Running Time (s)**


(a) Sparse Regression



2500


2000


1500


1000


500


0 **LIME + Max Wts LIME + FS** **LIME + Lasso** **Streak**


(b) Interpretability



Figure 2: 2(a): Logistic Regression, Phishing dataset with pairwise feature products, _k_ = 80 features.
By varying the parameter _ε_, our algorithm captures a time-accuracy tradeoff between RandomSubset and
LocalSearch. Results averaged over 40 iterations, standard deviation shown with error bars. 2(b): Running
times of interpretability algorithms on the Inception V3 network, _N_ = 30, _k_ = 5. Streaming maximization
runs 10 times faster than the LIME framework. Results averaged over 40 total iterations using 8 example
explanations, error bars show 1 standard deviation.

**6.1** **Sparse Regression with Pairwise Features**


In this experiment, a sparse logistic regression is fit on 2000 training and 2000 test observations from the
Phishing dataset [Lichman, 2013]. This setup is known to be weakly submodular under mild data assumptions

[Elenberg et al., 2016a]. First, the categorical features are one-hot encoded, increasing the feature dimension
to 68. Then, all pairwise products are added for a total of _N_ = 4692 features. To reduce computational
cost, feature products are generated and added to the stream on-the-fly as needed. We compare with 2 other
algorithms. RandomSubset selects the first _k_ features from the random stream. LocalSearch first fills a
buffer with the first _k_ features, and then swaps each incoming feature with the feature from the buffer which
yields the largest nonnegative improvement.
Figure 1(a) shows both the final log likelihood and the generalization accuracy for RandomSubset,
LocalSearch, and our Streak algorithm for _ε_ = _{_ 0 _._ 75 _,_ 0 _._ 1 _}_ and _k_ = _{_ 20 _,_ 40 _,_ 80 _}_ . As expected, the
RandomSubset algorithm has much larger variation since its performance depends highly on the random
stream order. It also performs significantly worse than LocalSearch for both metrics, whereas Streak
is comparable for most parameter choices. Figure 1(b) shows two measures of computational cost: running
time and the number of oracle evaluations (regression fits). We note Streak scales better as _k_ increases;
for example, Streak with _k_ = 80 and _ε_ = 0 _._ 1 ( _ε_ = 0 _._ 75) runs in about 70% (5%) of the time it takes to run
LocalSearch with _k_ = 40. Interestingly, our speedups are more substantial with respect to running time.
In some cases Streak actually fits more regressions than LocalSearch, but still manages to be faster. We
attribute this to the fact that nearly all of LocalSearch’s regressions involve _k_ features, which are slower
than many of the small regressions called by Streak.
Figure 2(a) shows the final log likelihood versus running time for _k_ = 80 and _ε ∈_ [0 _._ 05 _,_ 0 _._ 75]. By varying
the precision _ε_, we achieve a gradual tradeoff between speed and performance. This shows that Streak can
reduce the running time by over an order of magnitude with minimal impact on the final log likelihood.


**6.2** **Black-Box Interpretability**


Our next application is interpreting the predictions of black-box machine learning models. Specifically, we
begin with the Inception V3 deep neural network [Szegedy et al., 2016] trained on ImageNet. We use this
network for the task of classifying 5 types of flowers via transfer learning. This is done by adding a final
softmax layer and retraining the network.
We compare our approach to the LIME framework [Ribeiro et al., 2016] for developing sparse, interpretable explanations. The final step of LIME is to fit a _k_ -sparse linear regression in the space of interpretable
features. Here, the features are superpixels determined by the SLIC image segmentation algorithm [Achanta


7


et al., 2012] (regions from any other segmentation would also suffice). The number of superpixels is bounded
by _N_ = 30. After a feature selection step, a final regression is performed on only the selected features. The
following feature selection methods are supplied by LIME: _1. Highest Weights:_ fits a full regression and keep
the _k_ features with largest coefficients. _2. Forward Selection:_ standard greedy forward selection. _3. Lasso:_
_ℓ_ 1 regularization.
We introduce a novel method for black-box interpretability that is similar to but simpler than LIME. As
before, we segment an image into _N_ superpixels. Then, for a subset _S_ of those regions we can create a new
image that contains only these regions and feed this into the black-box classifier. For a given model _M_, an
input image _I_, and a label **L** 1 we ask for an explanation: why did model _M_ label image _I_ with label **L** 1 .
We propose the following solution to this problem. Consider the set function _f_ ( _S_ ) giving the likelihood that
image _I_ ( _S_ ) has label **L** 1 . We approximately solve


_|_ max _S|≤k_ _[f]_ [(] _[S]_ [)] _[,]_


using Streak. Intuitively, we are limiting the number of superpixels to _k_ so that the output will include only
the most important superpixels, and thus, will represent an interpretable explanation. In our experiments
we set _k_ = 5.
Note that the set function _f_ ( _S_ ) depends on the black-box classifier and is neither monotone nor submodular in general. Still, we find that the greedy maximization algorithm produces very good explanations
for the flower classifier as shown in Figure 3 and the additional experiments in the Appendix. Figure 2(b)
shows that our algorithm is much faster than the LIME approach. This is primarily because LIME relies on
generating and classifying a large set of randomly perturbed example images.

### **7 Conclusions**


We propose Streak, the first streaming algorithm for maximizing weakly submodular functions, and prove
that it achieves a constant factor approximation assuming a random stream order. This is useful when the
set function is not submodular and, additionally, takes a long time to evaluate or has a very large ground set.
Conversely, we show that under a worst case stream order no algorithm with memory sublinear in the ground
set size has a constant factor approximation. We formulate interpretability of black-box neural networks as
set function maximization, and show that Streak provides interpretable explanations faster than previous
approaches. We also show experimentally that Streak trades off accuracy and running time in nonlinear
sparse regression.
One interesting direction for future work is to tighten the bounds of Theorems 5.1 and 5.5, which are
nontrivial but somewhat loose. For example, there is a gap between the theoretical guarantee of the stateof-the-art algorithm for submodular functions and our bound for _γ_ = 1. However, as our algorithm performs
the same computation as that state-of-the-art algorithm when the function is submodular, this gap is solely
an analysis issue. Hence, the real theoretical performance of our algorithm is better than what we have been
able to prove in Section 5.

### **8 Acknowledgments**


This research has been supported by NSF Grants CCF 1344364, 1407278, 1422549, 1618689, ARO YIP
W911NF-14-1-0258, ISF Grant 1357/16, Google Faculty Research Award, and DARPA Young Faculty Award
(D16AP00046).


8


(a) (b)


(c) (d)


Figure 3: Comparison of interpretability algorithms for the Inception V3 deep neural network. We have
used transfer learning to extract features from Inception and train a flower classifier. In these four input
images the flower types were correctly classified (from (a) to (d): rose, sunflower, daisy, and daisy). We ask
the question of interpretability: _why_ did this model classify this image as rose. We are using our framework
(and the recent prior work LIME [Ribeiro et al., 2016]) to see which parts of the image the neural network
is looking at for these classification tasks. As can be seen Streak correctly identifies the flower parts of
the images while some LIME variations do not. More importantly, Streak is creating subsampled images
on-the-fly, and hence, runs approximately 10 times faster. Since interpretability tasks perform multiple calls
to the black-box model, the running times can be quite significant.


9


### **References**

Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. SLIC
Superpixels Compared to State-of-the-art Superpixel Methods. _IEEE Transactions on Pattern Analysis_
_and Machine Intelligence_, 34(11):2274–2282, 2012.


Jason Altschuler, Aditya Bhaskara, Gang (Thomas) Fu, Vahab Mirrokni, Afshin Rostamizadeh, and Morteza
Zadimoghaddam. Greedy Column Subset Selection: New Bounds and Distributed Algorithms. In _ICML_,
pages 2539–2548, 2016.


Francis R. Bach. Learning with Submodular Functions: A Convex Optimization Perspective. _Foundations_
_and Trends in Machine Learning_, 6, 2013.


Ashwinkumar Badanidiyuru, Baharan Mirzasoleiman, Amin Karbasi, and Andreas Krause. Streaming Submodular Maximization: Massive Data Summarization on the Fly. In _KDD_, pages 671–680, 2014.


Sohail Bahmani, Bhiksha Raj, and Petros T. Boufounos. Greedy Sparsity-Constrained Optimization. _Journal_
_of Machine Learning Research_, 14:807–841, 2013.


Rafael da Ponte Barbosa, Alina Ene, Huy L. Nguyen, and Justin Ward. The Power of Randomization:
Distributed Submodular Maximization on Massive Datasets. In _ICML_, pages 1236–1244, 2015.


Rafael da Ponte Barbosa, Alina Ene, Huy L. Nguyen, and Justin Ward. A New Framework for Distributed
Submodular Maximization. In _FOCS_, pages 645–654, 2016.


Andrew An Bian, Baharan Mirzasoleiman, Joachim M. Buhmann, and Andreas Krause. Guaranteed Nonconvex Optimization: Submodular Maximization over Continuous Domains. In _AISTATS_, pages 111–120,
2017.


Niv Buchbinder and Moran Feldman. Deterministic Algorithms for Submodular Maximization Problems. In
_SODA_, pages 392–403, 2016a.


Niv Buchbinder and Moran Feldman. Constrained Submodular Maximization via a Non-symmetric Technique. _CoRR_, abs/1611.03253, 2016b. URL `[http://arxiv.org/abs/1611.03253](http://arxiv.org/abs/1611.03253)` .


Niv Buchbinder, Moran Feldman, and Roy Schwartz. Online Submodular Maximization with Preemption.
In _SODA_, pages 1202–1216, 2015.


Gruia Călinescu, Chandra Chekuri, Martin Pál, and Jan Vondrák. Maximizing a Monotone Submodular
Function Subject to a Matroid Constraint. _SIAM J. Comput._, 40(6):1740–1766, 2011.


T-H. Hubert Chan, Zhiyi Huang, Shaofeng H.-C. Jiang, Ning Kang, and Zhihao Gavin Tang. Online
Submodular Maximization with Free Disposal: Randomization Beats [1] _/_ 4 for Partition Matroids. In _SODA_,
pages 1204–1223, 2017.


Chandra Chekuri, Shalmoli Gupta, and Kent Quanrud. Streaming Algorithms for Submodular Function
Maximization. In _ICALP_, pages 318–330, 2015.


Michele Conforti and Gérard Cornuéjols. Submodular set functions, matroids and the greedy algorithm:
Tight worst-case bounds and some generalizations of the Rado-Edmonds theorem. _Discrete Applied Math-_
_ematics_, 7(3):251–274, March 1984.


Abhimanyu Das and David Kempe. Submodular meets Spectral: Greedy Algorithms for Subset Selection,
Sparse Approximation and Dictionary Selection. In _ICML_, pages 1057–1064, 2011.


Ethan R. Elenberg, Rajiv Khanna, Alexandros G. Dimakis, and Sahand Negahban. Restricted Strong
Convexity Implies Weak Submodularity. _CoRR_, abs/1612.00804, 2016a. URL `[http://arxiv.org/abs/](http://arxiv.org/abs/1612.00804)`

`[1612.00804](http://arxiv.org/abs/1612.00804)` .


10


Ethan R. Elenberg, Rajiv Khanna, Alexandros G. Dimakis, and Sahand Negahban. Restricted Strong Convexity Implies Weak Submodularity. In _NIPS Workshop on Learning in High Dimensions with Structure_,
2016b.


Uriel Feige. A Threshold of ln n for Approximating Set Cover. _Journal of the ACM (JACM)_, 45(4):634–652,
1998.


Marshall L. Fisher, George L. Nemhauser, and Laurence A. Wolsey. An analysis of approximations for
maximizing submodular set functions–II. In M. L. Balinski and A. J. Hoffman, editors, _Polyhedral Com-_
_binatorics: Dedicated to the memory of D.R. Fulkerson_, pages 73–87. Springer Berlin Heidelberg, Berlin,
Heidelberg, 1978.


Avinatan Hassidim and Yaron Singer. Submodular Optimization Under Noise. In _COLT_, pages 1069–1122,
2017.


Steven C. H. Hoi, Rong Jin, Jianke Zhu, and Michael R. Lyu. Batch Mode Active Learning and its Application
to Medical Image Classification. In _ICML_, pages 417–424, 2006.


Thibaut Horel and Yaron Singer. Maximization of Approximately Submodular Functions. In _NIPS_, 2016.


Rajiv Khanna, Ethan R. Elenberg, Alexandros G. Dimakis, Joydeep Ghosh, and Sahand Negahban. On
Approximation Guarantees for Greedy Low Rank Optimization. In _ICML_, pages 1837–1846, 2017a.


Rajiv Khanna, Ethan R. Elenberg, Alexandros G. Dimakis, Sahand Negahban, and Joydeep Ghosh. Scalable
Greedy Support Selection via Weak Submodularity. In _AISTATS_, pages 1560–1568, 2017b.


Andreas Krause and Volkan Cevher. Submodular Dictionary Selection for Sparse Representation. In _ICML_,
pages 567–574, 2010.


Andreas Krause and Daniel Golovin. Submodular Function Maximization. _Tractability: Practical Approaches_
_to Hard Problems_, 3:71–104, 2014.


Moshe Lichman. UCI machine learning repository, 2013. URL `[http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)` .


Baharan Mirzasoleiman, Amin Karbasi, Rik Sarkar, and Andreas Krause. Distributed Submodular Maximization: Identifying Representative Elements in Massive Data. _NIPS_, pages 2049–2057, 2013.


Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrák, and Andreas Krause.
Lazier Than Lazy Greedy. In _AAAI_, pages 1812–1818, 2015.


George L. Nemhauser and Laurence A. Wolsey. Best Algorithms for Approximating the Maximum of a
Submodular Set Function. _Math. Oper. Res._, 3(3):177–188, August 1978.


George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for
maximizing submodular set functions–I. _Mathematical Programming_, 14(1):265–294, 1978.


Xinghao Pan, Stefanie Jegelka, Joseph E. Gonzalez, Joseph K. Bradley, and Michael I. Jordan. Parallel
Double Greedy Submodular Maximization. In _NIPS_, pages 118–126, 2014.


Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why Should I Trust You?” Explaining the
Predictions of Any Classifier. In _KDD_, pages 1135–1144, 2016.


Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic Attribution for Deep Networks. In _ICML_,
pages 3319–3328, 2017.


Maxim Sviridenko, Jan Vondrák, and Justin Ward. Optimal approximation for submodular and supermodular optimization with bounded curvature. In _SODA_, pages 1134–1148, 2015.


Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the
Inception Architecture for Computer Vision. In _CVPR_, pages 2818–2826, 2016.


11


Jan Vondrák. Submodularity and curvature: the optimal algorithm. _RIMS Kôkyûroku Bessatsu B23_, pages
253–266, 2010.


Kai Wei, Iyer Rishabh, and Jeff Bilmes. Submodularity in Data Subset Selection and Active Learning.
_ICML_, pages 1954–1963, 2015.


Zhuoran Yang, Zhaoran Wang, Han Liu, Yonina C. Eldar, and Tong Zhang. Sparse Nonlinear Regression:
Parameter Estimation and Asymptotic Inference. _ICML_, pages 2472–2481, 2016.


12


### **A Appendix**

**A.1** **Proof of Lemma 4.1**


The nonnegativity and monotonicity of _f_ _k_ follow immediately from the fact that _u_ ( _S_ ) and _v_ ( _S_ ) have these
properties. Thus, it remains to prove that _f_ _k_ is 0 _._ 5-weakly submodular for _|N_ _k_ _|_, _i.e._, that for every pair of
arbitrary sets _S, L ⊆N_ _k_ it holds that

� _f_ _k_ ( _w | L_ ) _≥_ 0 _._ 5 _· f_ _k_ ( _S | L_ ) _._

_w∈S\L_



There are two cases to consider. The first case is that _f_ _k_ ( _L_ ) = 2 _· u_ ( _L_ ) + 1. In this case _S \ L_ must contain
at least _⌈f_ _k_ ( _S | L_ ) _/_ 2 _⌉_ elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [. Additionally, the marginal contribution to] _[ L]_ [ of every element of]
_{u_ _i_ _}_ _[k]_ _i_ =1 [which does not belong to] _[ L]_ [ is at least][ 1][. Thus, we get]

� _f_ _k_ ( _w | L_ ) _≥_ � _f_ _k_ ( _w | L_ ) _≥|_ ( _S \ L_ ) _∩{u_ _i_ _}_ _[k]_ _i_ =1 _[|]_



� _f_ _k_ ( _w | L_ ) _≥_ �

_w∈S\L_ _w∈_ ( _S\L_ )



_f_ _k_ ( _w | L_ ) _≥|_ ( _S \ L_ ) _∩{u_ _i_ _}_ _[k]_ _i_ =1 _[|]_



_w∈_ ( _S\L_ ) _∩{u_ _i_ _}_ _[k]_ _i_ =1



_≥⌈f_ _k_ ( _S | L_ ) _/_ 2 _⌉≥_ 0 _._ 5 _· f_ _k_ ( _S | L_ ) _._



The second case is that _f_ _k_ ( _L_ ) = 2 _· v_ ( _L_ ). In this case _S \ L_ must contain at least _⌈f_ _k_ ( _S | L_ ) _/_ 2 _⌉_ elements of
_{v_ _i_ _}_ _[k]_ _i_ =1 [, and in addition, the marginal contribution to] _[ L]_ [ of every element of] _[ {][v]_ _[i]_ _[}]_ _[k]_ _i_ =1 [which does not belong]
to _L_ is at least 1. Thus, we get in this case again

� _f_ _k_ ( _w | L_ ) _≥_ � _f_ _k_ ( _w | L_ ) _≥|_ ( _S \ L_ ) _∩{v_ _i_ _}_ _[k]_ _i_ =1 _[|]_



� _f_ _k_ ( _w | L_ ) _≥_ �

_w∈S\L_ _w∈_ ( _S\L_ )



_f_ _k_ ( _w | L_ ) _≥|_ ( _S \ L_ ) _∩{v_ _i_ _}_ _[k]_ _i_ =1 _[|]_



_w∈_ ( _S\L_ ) _∩{v_ _i_ _}_ _[k]_ _i_ =1



_≥⌈f_ _k_ ( _S | L_ ) _/_ 2 _⌉≥_ 0 _._ 5 _· f_ _k_ ( _S | L_ ) _._


**A.2** **Proof of Theorem 4.2**


Consider an arbitrary (randomized) streaming algorithm ALG aiming to maximize _f_ _k_ ( _S_ ) subject to the
cardinality constraint _|S| ≤_ 2 _k_ . Since _ALG_ uses _o_ ( _N_ ) memory, we can guarantee, by choosing a large
enough _d_, that ALG uses no more than ( _c/_ 4) _· N_ memory. In order to show that ALG performs poorly,
consider the case that it gets first the elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [and the dummy elements (in some order to be]
determined later), and only then it gets the elements of _{v_ _i_ _}_ _[k]_ _i_ =1 [. The next lemma shows that some order of]
the elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [and the dummy elements is bad for][ ALG][.]


**Lemma A.1.** _There is an order for the elements of {u_ _i_ _}_ _[k]_ _i_ =1 _[and the dummy elements which guarantees that]_
_in expectation_ ALG _returns at most_ ( _c/_ 2) _· k elements of {u_ _i_ _}_ _[k]_ _i_ =1 _[.]_


_Proof._ Let _W_ be the set of the elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [and the dummy elements. Observe that the value of] _[ f]_ _[k]_
for every subset of _W_ is 0. Thus, ALG has no way to differentiate between the elements of _W_ until it views
the first element of _{v_ _i_ _}_ _[k]_ _i_ =1 [, which implies that the probability of every element] _[ w][ ∈]_ _[W]_ [ to remain in][ ALG][’s]
memory until the moment that the first element of _{v_ _i_ _}_ _[k]_ _i_ =1 [arrives is determined only by] _[ w]_ [’s arrival position.]
Hence, by choosing an appropriate arrival order one can guarantee that the sum of the probabilities of the
elements of _{u_ _i_ _}_ _[k]_ _i_ =1 [to be at the memory of][ ALG][ at this point is at most]



_kM_
_|W_ _|_ _[≤]_ _[k]_ [(] _[c]_ _k_ _[/]_ + [4][)] _d_ _[ ·][ N]_




_[/]_ [4][)] _[ ·][ N]_ [(][2] _[k]_ [ +] _[ d]_ [)]

= _[k]_ [(] _[c][/]_ [4][)] _[ ·]_
_k_ + _d_ _k_ + _d_




[)] _[ ·]_ [(][2] _[k]_ [ +] _[ d]_ [)]

_≤_ _[kc]_
_k_ + _d_ 2



2 _,_



where _M_ is the amount of memory ALG uses.


The expected value of the solution produced by ALG for the stream order provided by Lemma A.1 is at
most _ck_ + 1. Hence, its approximation ratio for _k >_ [1] _/_ _c_ is at most



_ck_ + 1




_[c]_ [1]

2 [+] 2



= _[c]_
2 _k_



2 _k_ _[< c .]_



13


**A.3** **Proof of Observation 5.3**


Algorithm 1 adds an element _u_ to the set _S_ only when the marginal contribution of _u_ with respect to _S_ is
at least _τ/k_ . Thus, it is always true that

_f_ ( _S_ ) _≥_ _[τ][ · ][|][S][|]_ _._

_k_


**A.4** **Proof of Proposition 5.4**


We begin by proving several intermediate lemmas. Recall that _γ_ ≜ _γ_ _k_, and notice that by the monotonicity
of _f_ we may assume that _OPT_ is of size _k_ . For every 0 _≤_ _i ≤|OPT_ _|_ = _k_, let _OPT_ _i_ be the random set
consisting of the last _i_ elements of _OPT_ according to the input order. Note that _OPT_ _i_ is simply a uniformly
random subset of _OPT_ of size _i_ . Thus, we can lower bound its expected value as follows.


**Lemma A.2.** _For every_ 0 _≤_ _i ≤_ _k,_ E[ _f_ ( _OPT_ _i_ )] _≥_ [1 _−_ (1 _−_ _γ/k_ ) _[i]_ ] _· f_ ( _OPT_ ) _._


_Proof._ We prove the lemma by induction on _i_ . For _i_ = 0 the lemma follows from the nonnegativity of _f_
since
_f_ ( _OPT_ 0 ) _≥_ 0 = [1 _−_ (1 _−_ _γ/k_ ) [0] ] _· f_ ( _OPT_ ) _._


Assume now that the lemma holds for some 0 _≤_ _i −_ 1 _< k_, and let us prove it holds also for _i_ . Since
_OPT_ _i−_ 1 is a uniformly random subset of _OPT_ of size _i −_ 1, and _OPT_ _i_ is a uniformly random subset of _OPT_
of size _i_, we can think of _OPT_ _i_ as obtained from _OPT_ _i−_ 1 by adding to this set a uniformly random element
of _OPT \ OPT_ _i−_ 1 . Taking this point of view, we get, for every set _T ⊆_ _OPT_ of size _i −_ 1,



E[ _f_ ( _OPT_ _i_ ) _| OPT_ _i−_ 1 = _T_ ] = _f_ ( _T_ ) +



� _u∈OP T_ _\T_ _[f]_ [(] _[u][ |][ T]_ [)]

_|OPT \ T_ _|_



_≥_ _f_ ( _T_ ) + [1]



� _f_ ( _u | T_ )

_u∈OP T \T_



_k_ _[·]_ �



_≥_ _f_ ( _T_ ) + _[γ]_

_k_ _[·][ f]_ [(] _[OPT][ \][ T][ |][ T]_ [)]



= �1 _−_ _[γ]_ _k_




_· f_ ( _T_ ) + _[γ]_
� _k_ _[·][ f]_ [(] _[OPT]_ [)] _[,]_



where the last inequality holds by the _γ_ -weak submodularity of _f_ . Taking expectation over the set _OPT_ _i−_ 1,
the last inequality becomes



E[ _f_ ( _OPT_ _i_ )] _≥_ 1 _−_ _[γ]_
� _k_


_≥_ 1 _−_ _[γ]_
� _k_



E[ _f_ ( _OPT_ _i−_ 1 )] + _[γ]_
� _k_ _[·][ f]_ [(] _[OPT]_ [)]



� _·_ �1 _−_ �1 _−_ _[γ]_ _k_



_i−_ 1 [�]

_· f_ ( _OPT_ ) + _[γ]_
� _k_ _[·][ f]_ [(] _[OPT]_ [)]



= �1 _−_ �1 _−_ _[γ]_ _k_



_i_ [�]

_· f_ ( _OPT_ ) _,_
�



where the second inequality follows from the induction hypothesis.


Let us now denote by _o_ 1 _, o_ 2 _, . . ., o_ _k_ the _k_ elements of _OPT_ in the order in which they arrive, and, for every
1 _≤_ _i ≤_ _k_, let _S_ _i_ be the set _S_ of Algorithm 1 immediately before the algorithm receives _o_ _i_ . Additionally,
let _A_ _i_ be an event fixing the arrival time of _o_ _i_, the set of elements arriving before _o_ _i_ and the order in which
they arrive. Note that conditioned on _A_ _i_, the sets _S_ _i_ and _OPT_ _k−i_ +1 are both deterministic.


**Lemma A.3.** _For every_ 1 _≤_ _i ≤_ _k and event A_ _i_ _,_ E[ _f_ ( _o_ _i_ _| S_ _i_ ) _| A_ _i_ ] _≥_ ( _γ/k_ ) _·_ [ _f_ ( _OPT_ _k−i_ +1 ) _−_ _f_ ( _S_ _i_ )] _, where_
_OPT_ _k−i_ +1 _and S_ _i_ _represent the deterministic values these sets take given A_ _i_ _._


_Proof._ By the monotonicity and _γ_ -weak submodularity of _f_, we get

� _f_ ( _u | S_ _i_ ) _≥_ _γ · f_ ( _OPT_ _k−i_ +1 _| S_ _i_ )

_u∈OP T_ _k−i_ +1


= _γ ·_ [ _f_ ( _OPT_ _k−i_ +1 _∪_ _S_ _i_ ) _−_ _f_ ( _S_ _i_ )]


_≥_ _γ ·_ [ _f_ ( _OPT_ _k−i_ +1 ) _−_ _f_ ( _S_ _i_ )] _._


14


Since _o_ _i_ is a uniformly random element of _OPT_ _k−i_ +1, even conditioned on _A_ _i_, the last inequality implies



E[ _f_ ( _o_ _i_ _| S_ _i_ ) _| A_ _i_ ] =


_≥_



� _u∈OP T_ _k−i_ +1 _[f]_ [(] _[u][ |][ S]_ _[i]_ [)]

_k −_ _i_ + 1

� _u∈OP T_ _k−i_ +1 _[f]_ [(] _[u][ |][ S]_ _[i]_ [)]


_k_




[[] _[f]_ [(] _[OPT]_ _[k][−][i]_ [+1] [)] _[ −]_ _[f]_ [(] _[S]_ _[i]_ [)]]
_≥_ _[γ][ ·]_ _._

_k_



Let ∆ _i_ be the increase in the value of _S_ in the iteration of Algorithm 1 in which it gets _o_ _i_ .


**Lemma A.4.** _Fix_ 1 _≤_ _i ≤_ _k and event A_ _i_ _, and let OPT_ _k−i_ +1 _and S_ _i_ _represent the deterministic values_
_these sets take given A_ _i_ _. If f_ ( _S_ _i_ ) _< τ_ _, then_ E[∆ _i_ _| A_ _i_ ] _≥_ [ _γ · f_ ( _OPT_ _k−i_ +1 ) _−_ 2 _τ_ ] _/k._


_Proof._ Notice that by Observation 5.3 the fact that _f_ ( _S_ _i_ ) _< τ_ implies that _S_ _i_ contains less than _k_ elements.
Thus, conditioned on _A_ _i_, Algorithm 1 adds _o_ _i_ to _S_ whenever _f_ ( _o_ _i_ _| S_ _i_ ) _≥_ _τ/k_, which means that



∆ _i_ =


One implication of the last equality is



_f_ ( _o_ _i_ _| S_ _i_ ) if _f_ ( _o_ _i_ _| S_ _i_ ) _≥_ _τ/k,_

0 otherwise _._
�



E[∆ _i_ _| A_ _i_ ] _≥_ E[ _f_ ( _o_ _i_ _| S_ _i_ ) _| A_ _i_ ] _−_ _τ/k,_


which intuitively means that the contribution to E[ _f_ ( _o_ _i_ _| S_ _i_ ) _| A_ _i_ ] of values of _f_ ( _o_ _i_ _| S_ _i_ ) which are too small
to make the algorithm add _o_ _i_ to _S_ is at most _τ/k_ . The lemma now follows by observing that Lemma A.3
and the fact that _f_ ( _S_ _i_ ) _< τ_ guarantee


E[ _f_ ( _o_ _i_ _| S_ _i_ ) _| A_ _i_ ] _≥_ ( _γ/k_ ) _·_ [ _f_ ( _OPT_ _k−i_ +1 ) _−_ _f_ ( _S_ _i_ )]


_>_ ( _γ/k_ ) _·_ [ _f_ ( _OPT_ _k−i_ +1 ) _−_ _τ_ ]


_≥_ [ _γ · f_ ( _OPT_ _k−i_ +1 ) _−_ _τ_ ] _/k ._


We are now ready to put everything together and get a lower bound on E[∆ _i_ ].


**Lemma A.5.** _For every_ 1 _≤_ _i ≤_ _k,_


[[][Pr][[] _[E]_ []] _[ −]_ [(][1] _[ −]_ _[γ/][k]_ [)] _[k][−][i]_ [+1] []] _[ ·]_ _[f]_ [(] _[OPT]_ [)] _[ −]_ [2] _[τ]_
E[∆ _i_ ] _≥_ _[γ][ ·]_ _._

_k_


_Proof._ Let _E_ _i_ be the event that _f_ ( _S_ _i_ ) _< τ_ . Clearly _E_ _i_ is the disjoint union of the events _A_ _i_ which imply
_f_ ( _S_ _i_ ) _< τ_, and thus, by Lemma A.4,


E[∆ _i_ _| E_ _i_ ] _≥_ [ _γ ·_ E[ _f_ ( _OPT_ _k−i_ +1 ) _| E_ _i_ ] _−_ 2 _τ_ ] _/k ._


Note that ∆ _i_ is always nonnegative due to the monotonicity of _f_ . Thus,


E[∆ _i_ ] = Pr[ _E_ _i_ ] _·_ E[∆ _i_ _| E_ _i_ ] + Pr[ _E_ [¯] _i_ ] _·_ E[∆ _i_ _|_ _E_ [¯] _i_ ] _≥_ Pr[ _E_ _i_ ] _·_ E[∆ _i_ _| E_ _i_ ]


_≥_ [ _γ ·_ Pr[ _E_ _i_ ] _·_ E[ _f_ ( _OPT_ _k−i_ +1 ) _| E_ _i_ ] _−_ 2 _τ_ ] _/k ._


It now remains to lower bound the expression Pr[ _E_ _i_ ] _·_ E[ _f_ ( _OPT_ _k−i_ +1 ) _| E_ _i_ ] on the rightmost hand side of
the last inequality.


Pr[ _E_ _i_ ] _·_ E[ _f_ ( _OPT_ _k−i_ +1 ) _| E_ _i_ ] = E[ _f_ ( _OPT_ _k−i_ +1 )] _−_ Pr[ _E_ [¯] _i_ ] _·_ E[ _f_ ( _OPT_ _k−i_ +1 ) _|_ _E_ [¯] _i_ ]

_≥_ [1 _−_ (1 _−_ _γ/k_ ) _[k][−][i]_ [+1] _−_ (1 _−_ Pr[ _E_ _i_ ])] _· f_ ( _OPT_ )

_≥_ [Pr[ _E_ ] _−_ (1 _−_ _γ/k_ ) _[k][−][i]_ [+1] ] _· f_ ( _OPT_ )


where the first inequality follows from Lemma A.2 and the monotonicity of _f_, and the second inequality
holds since _E_ implies _E_ _i_ which means that Pr[ _E_ _i_ ] _≥_ Pr[ _E_ ] for every 1 _≤_ _i ≤_ _k_ .


15


Proposition 5.4 follows quite easily from the last lemma.


_Proof of Proposition 5.4._ Lemma A.5 implies, for every 1 _≤_ _i ≤⌈k/_ 2 _⌉_,




_[γ]_

_k_ _[f]_ [(] _[OPT]_ [)[Pr[] _[E]_ []] _[ −]_ [(1] _[ −]_ _[γ/k]_ [)] _[k][−⌈][k/]_ [2] _[⌉]_ [+1] []] _[ −]_ [2] _k_ _[τ]_



E[∆ _i_ ] _≥_ _[γ]_



_k_




_[γ]_

_k_ _[f]_ [(] _[OPT]_ [)[Pr[] _[E]_ []] _[ −]_ [(1] _[ −]_ _[γ/k]_ [)] _[k/]_ [2] []] _[ −]_ [2] _k_ _[τ]_



_≥_ _[γ]_



_k_



_≥_ _γ ·_ [Pr[ _E_ ] _−_ _e_ _[−][γ/]_ [2] ] _· f_ ( _OPT_ ) _−_ 2 _τ_ _/k ._
� �


The definition of ∆ _i_ and the monotonicity of _f_ imply together



E[ _f_ ( _S_ )] _≥_



_b_
� E[∆ _i_ ]


_i_ =1



for every integer 1 _≤_ _b ≤_ _k_ . In particular, for _b_ = _⌈k/_ 2 _⌉_, we get


E[ _f_ ( _S_ )] _≥_ _[b]_ _γ ·_ [Pr[ _E_ ] _−_ _e_ _[−][γ/]_ [2] ] _· f_ ( _OPT_ ) _−_ 2 _τ_

_k_ _[·]_ � �

_≥_ [1] _γ ·_ [Pr[ _E_ ] _−_ _e_ _[−][γ/]_ [2] ] _· f_ ( _OPT_ ) _−_ 2 _τ_ _._

2 _[·]_ � �


**A.5** **Proof of Theorem 5.1**


In this section we combine the previous results to prove Theorem 5.1. Recall that Observation 5.2 and
Proposition 5.4 give two lower bounds on E[ _f_ ( _S_ )] that depend on Pr[ _E_ ]. The following lemmata use these
lower bounds to derive another lower bound on this quantity which is independent of Pr[ _E_ ]. For ease of the
reading, we use in this section the shorthand _γ_ _[′]_ = _e_ _[−][γ/]_ [2] .




_[τ]_

_a_ _[·]_ [ 3] _[−][e]_ _[−][γ/]_ [2] _[−]_ [2] 2 _[√]_



**Lemma A.6.** E[ _f_ ( _S_ )] _≥_ 2 _τa_ [(3] _[ −]_ _[γ]_ _[′]_ _[ −]_ [2] _[√]_ [2] _[ −]_ _[γ]_ ~~_[′]_~~ [) =] _[τ]_ _a_


_Proof._ By the lower bound given by Proposition 5.4,



_−e_

2 _whenever_ Pr[ _E_ ] _≥_ 2 _−_ _[√]_ 2 _−_ _γ_ ~~_[′]_~~ _._



2 _−e_ _[−][γ/]_ [2]



E[ _f_ ( _S_ )] _≥_ [1]

2 _[· {][γ][ ·]_ [ [Pr[] _[E]_ []] _[ −]_ _[γ]_ _[′]_ []] _[ ·][ f]_ [(] _[OPT]_ [)] _[ −]_ [2] _[τ]_ _[}]_



_≥_ [1]

2

= [1]



2 _−_ _γ_ _[′]_ _−_ _γ_ _[′]_ [�] _· f_ ( _OPT_ ) _−_ (�



2 _[·]_ � _γ ·_ �2 _−_ ~~�~~

2 [1] _[·]_ � _γ ·_ �2 _−_ ~~�~~



2 _−_ _γ_ _[′]_ _−_ _γ_ _[′]_ [�] _· f_ ( _OPT_ ) _−_ 2 _τ_
�



2 _−_ _γ_ _[′]_ _−_ 1) _·_ _[τ]_

_a_



�



2 _[τ]_ _a_ _[·]_ �2 _−_ ~~�~~



2 _−_ _γ_ _[′]_ + 1
�



_≥_ _[τ]_



2 _−_ _γ_ _[′]_ _−_ _γ_ _[′]_ _−_ �




_[τ]_

_a_ _[·]_ [ 3] _[ −]_ _[γ]_ _[′]_ _[ −]_ 2 [2] _[√]_ [2] _[ −]_ _[γ]_ ~~_[′]_~~



= _[τ]_



2 _,_



where the first equality holds since _a_ = ( _[√]_ 2 _−_ _γ_ ~~_[′]_~~ _−_ 1) _/_ 2, and the last inequality holds since _aγ · f_ ( _OPT_ ) _≥_

_τ_ .




_[τ]_

_a_ _[·]_ [ 3] _[−][e]_ _[−][γ/]_ [2] _[−]_ [2] 2 _[√]_



**Lemma A.7.** E[ _f_ ( _S_ )] _≥_ 2 _τa_ [(3] _[ −]_ _[γ]_ _[′]_ _[ −]_ [2] _[√]_ [2] _[ −]_ _[γ]_ ~~_[′]_~~ [) =] _[τ]_ _a_



_−e_

2 _whenever_ Pr[ _E_ ] _≤_ 2 _−_ _[√]_ 2 _−_ _γ_ ~~_[′]_~~ _._



2 _−e_ _[−][γ/]_ [2]



_Proof._ By the lower bound given by Observation 5.2,


E[ _f_ ( _S_ )] _≥_ (1 _−_ Pr[ _E_ ]) _· τ ≥_ �1 _−_ 2 + ~~�~~ 2 _−_ _γ_ _[′]_ � _· τ_




_[τ]_ _a_ [= 3] _[ −]_ _[γ]_ _[′]_ _[ −]_ 2 [2] _[√]_ [2] _[ −]_ _[γ]_ ~~_[′]_~~



_a_ _[.]_



=
��



2 _−_ _γ_ _[′]_ _−_ 1 _·_ _√_ 2 _−_ _γ_ ~~_′_~~ _−_ 1
� 2



_γ_ _−_ _·_ _[τ]_

2




_[ −]_ _[γ]_

_·_ _[τ]_
2 _a_



Combining Lemmata A.6 and A.7 we get the theorem.


16


**A.6** **Proof of Theorem 5.5**


There are two cases to consider. If _γ <_ [4] _/_ 3 _· k_ _[−]_ [1], then we use the following simple observation.


**Observation A.8.** _The final value of the variable m is f_ [max] ≜ max _{f_ ( _u_ ) _| u ∈N} ≥_ _[γ]_ _k_ _[·][ f]_ [(] _[OPT]_ [)] _[.]_


_Proof._ The way _m_ is updated by Algorithm 2 guarantees that its final value is _f_ [max] . To see why the other
part of the observation is also true, note that the _γ_ -weak submodularity of _f_ implies


_f_ [max] _≥_ max _{f_ ( _u_ ) _| u ∈_ _OPT_ _}_ = _f_ (∅) + max _{f_ ( _u |_ ∅) _| u ∈_ _OPT_ _}_



_≥_ _f_ (∅) + [1]

_k_



� _f_ ( _u |_ ∅) _≥_ _f_ (∅) + _[γ]_ _k_

_u∈OP T_



�




_[γ]_

_k_ _[f]_ [(] _[OPT][ |]_ [ ∅][)] _[ ≥]_ _[γ]_ _k_



_k_ _[·][ f]_ [(] _[OPT]_ [)] _[ .]_



By Observation A.8, the value of the solution produced by Streak is at least




_[γ]_

_k_ _[·][ f]_ [(] _[OPT]_ [)] _[ ≥]_ [3] _[γ]_ 4 [2]



_f_ ( _u_ _m_ ) = _m ≥_ _[γ]_




_· f_ ( _OPT_ )
4



_≥_ (1 _−_ _ε_ ) _γ ·_ [3][(] _[γ/]_ [2][)] _· f_ ( _OPT_ )

2



_≥_ (1 _−_ _ε_ ) _γ ·_ [3] _[ −]_ [3] _[e]_ _[−][γ/]_ [2] _· f_ ( _OPT_ )

2



_√_
_≥_ (1 _−_ _ε_ ) _γ ·_ [3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ [2]



2 _−_ _e_ _[−][γ/]_ [2]




_· f_ ( _OPT_ ) _,_
2



where the second to last inequality holds since 1 _−_ _[γ]_ _/_ 2 _≤_ _e_ _[−]_ _[γ]_ _[/]_ [2], and the last inequality holds since _e_ _[−][γ]_ + _e_ _[−][γ/]_ [2] _≤_
2.
It remains to consider the case _γ ≥_ [4] _/_ 3 _· k_ _[−]_ [1], which has a somewhat more involved proof. Observe that
the approximation ratio of Streak is 1 whenever _f_ ( _OPT_ ) = 0 because the value of any set, including the
output set of the algorithm, is nonnegative. Thus, we can safely assume in the rest of the analysis of the
approximation ratio of Algorithm 2 that _f_ ( _OPT_ ) _>_ 0.
Let _τ_ _[∗]_ be the maximal value in the set _{_ (1 _−ε_ ) _[i]_ _| i ∈_ Z _}_ which is not larger than _aγ_ _·f_ ( _OPT_ ). Note that _τ_ _[∗]_

exists by our assumption that _f_ ( _OPT_ ) _>_ 0. Moreover, we also have (1 _−ε_ ) _·aγ_ _·f_ ( _OPT_ ) _< τ_ _[∗]_ _≤_ _aγ_ _·f_ ( _OPT_ ).
The following lemma gives an interesting property of _τ_ _[∗]_ . To understand the lemma, it is important to note
that the set of values for _τ_ in the instances of Algorithm 1 appearing in the final collection _I_ is deterministic
because the final value of _m_ is always _f_ [max] .


**Lemma A.9.** _If there is an instance of Algorithm 1 with τ_ = _τ_ _[∗]_ _in I when_ Streak _terminates, then in_
_expectation_ Streak _has an approximation ratio of at least_



_√_
(1 _−_ _ε_ ) _γ ·_ [3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ [2]



2 _−_ _e_ _[−][γ/]_ [2]



2 _._



_Proof._ Consider a value of _τ_ for which there is an instance of Algorithm 1 in _I_ when Algorithm 2 terminates,
and consider the moment that Algorithm 2 created this instance. Since the instance was not created earlier,
we get that _m_ was smaller than _τ/k_ before this point. In other words, the marginal contribution of every
element that appeared before this point to the empty set was less than _τ/k_ . Thus, even if the instance had
been created earlier it would not have taken any previous elements.
An important corollary of the above observation is that the output of every instance of Algorithm 1 that
appears in _I_ when Streak terminates is equal to the output it would have had if it had been executed on
the entire input stream from its beginning (rather than just from the point in which it was created). Since we
assume that there is an instance of Algorithm 1 with _τ_ = _τ_ _[∗]_ in the final collection _I_, we get by Theorem 5.1
that the expected value of the output of this instance is at least



_τ_ _[∗]_ [3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ [2] _√_

_a_ _[·]_ 2



2 _−_ _e_ _[−][γ/]_ [2]



_τ_ _[∗]_




[2] _√_ 2 _−_ _e_ _[−][γ/]_ [2] _√_

_>_ (1 _−_ _ε_ ) _γ · f_ ( _OPT_ ) _·_ [3] _[ −]_ _[e]_ _[−][γ/]_ [2] _[ −]_ [2]
2 2



2 _−_ _e_ _[−][γ/]_ [2]



2 _._



The lemma now follows since the output of Streak is always at least as good as the output of each one of
the instances of Algorithm 1 in its collection _I_ .


17


We complement the last lemma with the next one.


**Lemma A.10.** _If γ ≥_ [4] _/_ 3 _· k_ _[−]_ [1] _, then there is an instance of Algorithm 1 with τ_ = _τ_ _[∗]_ _in I when_ Streak
_terminates._


_Proof._ We begin by bounding the final value of _m_ . By Observation A.8 this final value is _f_ [max] _≥_ _[γ]_ _k_ _[·][f]_ [(] _[OPT]_ [)][.]

On the other hand, _f_ ( _u_ ) _≤_ _f_ ( _OPT_ ) for every element _u ∈N_ since _{u}_ is a possible candidate to be _OPT_,
which implies _f_ [max] _≤_ _f_ ( _OPT_ ). Thus, the final collection _I_ contains an instance of Algorithm 1 for every
value of _τ_ within the set

�(1 _−_ _ε_ ) _[i]_ _| i ∈_ Z and (1 _−_ _ε_ ) _· f_ [max] _/_ (9 _k_ [2] ) _≤_ (1 _−_ _ε_ ) _[i]_ _≤_ _f_ [max] _· k_ �

_⊇_ �(1 _−_ _ε_ ) _[i]_ _| i ∈_ Z and (1 _−_ _ε_ ) _· f_ ( _OPT_ ) _/_ (9 _k_ [2] ) _≤_ (1 _−_ _ε_ ) _[i]_ _≤_ _γ · f_ ( _OPT_ )� _._


To see that _τ_ _[∗]_ belongs to the last set, we need to verify that it obeys the two inequalities defining this set.
On the one hand, _a_ = ( _√_ 2 _−_ _e_ _[−][γ/]_ [2] _−_ 1) _/_ 2 _<_ 1 implies


_τ_ _[∗]_ _≤_ _aγ · f_ ( _OPT_ ) _≤_ _γ · f_ ( _OPT_ ) _._


On the other hand, _γ ≥_ [4] _/_ 3 _· k_ _[−]_ [1] and 1 _−_ _e_ _[−][γ/]_ [2] _≥_ _γ/_ 2 _−_ _γ_ [2] _/_ 8 imply


_τ_ _[∗]_ _>_ (1 _−_ _ε_ ) _· aγ · f_ ( _OPT_ ) = (1 _−_ _ε_ ) _·_ ( ~~�~~ 2 _−_ _e_ _[−][γ/]_ [2] _−_ 1) _· γ · f_ ( _OPT_ ) _/_ 2



_≥_ (1 _−_ _ε_ ) _·_ (�1 + _γ/_ 2 _−_ _γ_ [2] _/_ 8 _−_ 1) _· γ · f_ ( _OPT_ ) _/_ 2

_≥_ (1 _−_ _ε_ ) _·_ (�1 + _γ/_ 4 + _γ_ [2] _/_ 64 _−_ 1) _· γ · f_ ( _OPT_ ) _/_ 2

= (1 _−_ _ε_ ) _·_ (�(1 + _γ/_ 8) [2] _−_ 1) _· γ · f_ ( _OPT_ ) _/_ 2 _≥_ (1 _−_ _ε_ ) _· γ_ [2] _· f_ ( _OPT_ ) _/_ 16



_≥_ (1 _−_ _ε_ ) _· f_ ( _OPT_ ) _/_ (9 _k_ [2] ) _._


Combining Lemmata A.9 and A.10 we get the desired guarantee on the approximation ratio of Streak.


**A.7** **Proof of Theorem 5.6**


Observe that Streak keeps only one element ( _u_ _m_ ) in addition to the elements maintained by the instances
of Algorithm 1 in _I_ . Moreover, Algorithm 1 keeps at any given time at most _O_ ( _k_ ) elements since the set _S_ it
maintains can never contain more than _k_ elements. Thus, it is enough to show that the collection _I_ contains
at every given time at most _O_ ( _ε_ _[−]_ [1] log _k_ ) instances of Algorithm 1. If _m_ = 0 then this is trivial since _I_ = ∅.
Thus, it is enough to consider the case _m >_ 0. Note that in this case


_mk_
_|I| ≤_ 1 _−_ log 1 _−ε_
(1 _−_ _ε_ ) _m/_ (9 _k_ [2] ) [= 2] _[ −]_ ln(1 [ln][(][9] _−_ _[k]_ [3] _ε_ [)] )




[ln 9 + 3 ln] _[ k]_ = 2 _−_ _[O]_ [(][ln] _[ k]_ [)]

ln(1 _−_ _ε_ ) ln(1 _−_ _ε_



= 2 _−_ [ln 9 + 3 ln] _[ k]_



ln(1 _−_ _ε_ ) _[.]_



We now need to upper bound ln(1 _−_ _ε_ ). Recall that 1 _−_ _ε ≤_ _e_ _[−][ε]_ . Thus, ln(1 _−_ _ε_ ) _≤−ε_ . Plugging this into
the previous inequality gives


_|I| ≤_ 2 _−_ _[O]_ [(] _−_ [ln] _ε_ _[ k]_ [)] = 2 + _O_ ( _ε_ _[−]_ [1] ln _k_ ) = _O_ ( _ε_ _[−]_ [1] ln _k_ ) _._


18


**A.8** **Additional Experiments**


(a) (b)


(c) (d)


Figure 4: In addition to the experiment in Section 6.2, we also replaced LIME’s default feature selection
algorithms with Streak and then fit the same sparse regression on the selected superpixels. This method
is captioned “LIME + Streak.” Since LIME fits a series of nested regression models, the corresponding
set function is guaranteed to be monotone, but is not necessarily submodular. We see that results look
qualitatively similar and are in some instances better than the default methods. However, the running time
of this approach is similar to the other LIME algorithms.


19


(a) (b)


Figure 5: Here we used the same setup described in Figure 4, but compared explanations for predicting 2
different classes for the same base image: 5(a) the highest likelihood label (sunflower) and 5(b) the secondhighest likelihood label (rose). All algorithms perform similarly for the sunflower label, but our algorithms
identify the most rose-like parts of the image.


20


