## **Counterfactual Fairness**



**Chris Russell** _[∗]_
The Alan Turing Institute and
University of Surrey
```
crussell@turing.ac.uk

```


**Matt Kusner** _[∗]_
The Alan Turing Institute and
University of Warwick
```
 mkusner@turing.ac.uk

```


**Joshua Loftus** _[∗]_
New York University
```
 loftus@nyu.edu

```


**Ricardo Silva**
The Alan Turing Institute and
University College London
```
              ricardo@stats.ucl.ac.uk

```

**Abstract**


Machine learning can impact people with legal or ethical consequences when
it is used to automate decisions in areas such as insurance, lending, hiring, and
predictive policing. In many of these scenarios, previous decisions have been made
that are unfairly biased against certain subpopulations, for example those of a
particular race, gender, or sexual orientation. Since this past data may be biased,
machine learning predictors must account for this to avoid perpetuating or creating
discriminatory practices. In this paper, we develop a framework for modeling
fairness using tools from causal inference. Our definition of _counterfactual fairness_
captures the intuition that a decision is fair towards an individual if it is the same in
(a) the actual world and (b) a counterfactual world where the individual belonged
to a different demographic group. We demonstrate our framework on a real-world
problem of fair prediction of success in law school.


**1** **Contribution**


Machine learning has spread to fields as diverse as credit scoring [ 20 ], crime prediction [ 5 ], and loan
assessment [ 25 ]. Decisions in these areas may have ethical or legal implications, so it is necessary for
the modeler to think beyond the objective of maximizing prediction accuracy and consider the societal
impact of their work. For many of these applications, it is crucial to ask if the predictions of a model
are _fair_ . Training data can contain unfairness for reasons having to do with historical prejudices or
other factors outside an individual’s control. In 2016, the Obama administration released a report [2]
which urged data scientists to analyze “how technologies can deliberately or inadvertently perpetuate,
exacerbate, or mask discrimination."


There has been much recent interest in designing algorithms that make fair predictions [ 4, 6, 10,
12, 14, 16 – 19, 22, 24, 36 – 39 ]. In large part, the literature has focused on formalizing fairness
into quantitative definitions and using them to solve a discrimination problem in a certain dataset.
Unfortunately, for a practitioner, law-maker, judge, or anyone else who is interested in implementing
algorithms that control for discrimination, it can be difficult to decide _which_ definition of fairness to
choose for the task at hand. Indeed, we demonstrate that depending on the relationship between a
protected attribute and the data, certain definitions of fairness can actually _increase discrimination_ .


_∗_ Equal contribution. This work was done while JL was a Research Fellow at the Alan Turing Institute.
2 https://obamawhitehouse.archives.gov/blog/2016/05/04/big-risks-big-opportunities-intersection-big-dataand-civil-rights


31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.


In this paper, we introduce the first explicitly causal approach to address fairness. Specifically, we
leverage the causal framework of Pearl [30] to model the relationship between protected attributes
and data. We describe how techniques from causal inference can be effective tools for designing fair
algorithms and argue, as in DeDeo [9], that it is essential to properly address causality in fairness. In
perhaps the most closely related prior work, Johnson et al. [15] make similar arguments but from a
non-causal perspective. An alternative use of causal modeling in the context of fairness is introduced
independently by [21].


In Section 2, we provide a summary of basic concepts in fairness and causal modeling. In Section 3,
we provide the formal definition of _counterfactual fairness_, which enforces that a distribution over
possible predictions for an individual should remain unchanged in a world where an individual’s
protected attributes had been different in a causal sense. In Section 4, we describe an algorithm to
implement this definition, while distinguishing it from existing approaches. In Section 5, we illustrate
the algorithm with a case of fair assessment of law school success.


**2** **Background**


This section provides a basic account of two separate areas of research in machine learning, which
are formally unified in this paper. We suggest Berk et al. [1] and Pearl et al. [29] as references.
Throughout this paper, we will use the following notation. Let _A_ denote the set of _protected attributes_
of an individual, variables that must not be discriminated against in a formal sense defined differently
by each notion of fairness discussed. The decision of whether an attribute is protected or not is taken
as a primitive in any given problem, regardless of the definition of fairness adopted. Moreover, let
_X_ denote the other observable attributes of any particular individual, _U_ the set of relevant latent
attributes which are not observed, and let _Y_ denote the outcome to be predicted, which itself might
be contaminated with historical biases. Finally, _Y_ [ˆ] is the _predictor_, a random variable that depends on
_A, X_ and _U_, and which is produced by a machine learning algorithm as a prediction of _Y_ .


**2.1** **Fairness**


There has been much recent work on fair algorithms. These include fairness through unawareness

[ 12 ], individual fairness [ 10, 16, 24, 38 ], demographic parity/disparate impact [ 36 ], and equality of
opportunity [ 14, 37 ]. For simplicity we often assume _A_ is encoded as a binary attribute, but this can
be generalized.


**Definition 1** (Fairness Through Unawareness (FTU)) **.** _An algorithm is fair so long as any protected_
_attributes A are not explicitly used in the decision-making process._


Any mapping _Y_ [ˆ] : _X →_ _Y_ that excludes _A_ satisfies this. Initially proposed as a baseline, the approach
has found favor recently with more general approaches such as Grgic-Hlaca et al. [12] . Despite its
compelling simplicity, FTU has a clear shortcoming as elements of _X_ can contain discriminatory
information analogous to _A_ that may not be obvious at first. The need for expert knowledge in
assessing the relationship between _A_ and _X_ was highlighted in the work on individual fairness:


**Definition 2** (Individual Fairness (IF)) **.** _An algorithm is fair if it gives similar predictions to similar_
_individuals. Formally, given a metric_ _d_ ( _·, ·_ ) _, if individuals_ _i_ _and_ _j_ _are similar under this metric (i.e.,_
_d_ ( _i, j_ ) _is small) then their predictions should be similar:_ _Y_ [ˆ] ( _X_ [(] _[i]_ [)] _, A_ [(] _[i]_ [)] ) _≈_ _Y_ [ˆ] ( _X_ [(] _[j]_ [)] _, A_ [(] _[j]_ [)] ) _._


As described in [ 10 ], the metric _d_ ( _·, ·_ ) must be carefully chosen, requiring an understanding of the
domain at hand beyond black-box statistical modeling. This can also be contrasted against population
level criteria such as

**Definition 3** (Demographic Parity (DP)) **.** _A predictor_ _Y_ [ˆ] _satisfies demographic parity if_ _P_ ( _Y_ [ˆ] _|A_ =
0) = _P_ ( _Y_ [ˆ] _|A_ = 1) _._

**Definition 4** (Equality of Opportunity (EO)) **.** _A predictor_ _Y_ [ˆ] _satisfies equality of opportunity if_
_P_ ( _Y_ [ˆ] = 1 _|A_ = 0 _, Y_ = 1) = _P_ ( _Y_ [ˆ] = 1 _|A_ = 1 _, Y_ = 1) _._


These criteria can be incompatible in general, as discussed in [ 1, 7, 22 ]. Following the motivation of
IF and [ 15 ], we propose that knowledge about relationships between all attributes should be taken
into consideration, even if strong assumptions are necessary. Moreover, it is not immediately clear


2


for any of these approaches in which ways historical biases can be tackled. We approach such issues
from an explicit causal modeling perspective.


**2.2** **Causal Models and Counterfactuals**


We follow Pearl [28], and define a causal model as a triple ( _U, V, F_ ) of sets such that


_• U_ is a set of latent **background** variables,which are factors not caused by any variable in
the set _V_ of **observable** variables;

_• F_ is a set of functions _{f_ 1 _, . . ., f_ _n_ _}_, one for each _V_ _i_ _∈_ _V_, such that _V_ _i_ = _f_ _i_ ( _pa_ _i_ _, U_ _pa_ _i_ ),
_pa_ _i_ _⊆_ _V \{V_ _i_ _}_ and _U_ _pa_ _i_ _⊆_ _U_ . Such equations are also known as **structural equations** [ 2 ].


The notation “ _pa_ _i_ ” refers to the “parents” of _V_ _i_ and is motivated by the assumption that the model
factorizes as a directed graph, here assumed to be a directed acyclic graph (DAG). The model is causal
in that, given a distribution _P_ ( _U_ ) over the background variables _U_, we can derive the distribution of a
subset _Z ⊆_ _V_ following an **intervention** on _V \ Z_ . An intervention on variable _V_ _i_ is the substitution
of equation _V_ _i_ = _f_ _i_ ( _pa_ _i_ _, U_ _pa_ _i_ ) with the equation _V_ _i_ = _v_ for some _v_ . This captures the idea of an
agent, external to the system, modifying it by forcefully assigning value _v_ to _V_ _i_, for example as in a
randomized experiment.


The specification of _F_ is a strong assumption but allows for the calculation of **counterfactual**
quantities. In brief, consider the following counterfactual statement, “the value of _Y_ if _Z_ had taken
value _z_ ”, for two observable variables _Z_ and _Y_ . By assumption, the state of any observable variable is
fully determined by the background variables and structural equations. The counterfactual is modeled
as the solution for _Y_ for a given _U_ = _u_ where the equations for _Z_ are replaced with _Z_ = _z_ . We
denote it by _Y_ _Z←z_ ( _u_ ) [28], and sometimes as _Y_ _z_ if the context of the notation is clear.


Counterfactual inference, as specified by a causal model ( _U, V, F_ ) given evidence _W_, is the computation of probabilities _P_ ( _Y_ _Z←z_ ( _U_ ) _| W_ = _w_ ), where _W_, _Z_ and _Y_ are subsets of _V_ . Inference proceeds
in three steps, as explained in more detail in Chapter 4 of Pearl et al. [29] : 1. **Abduction** : for a given
prior on _U_, compute the posterior distribution of _U_ given the evidence _W_ = _w_ ; 2. **Action** : substitute
the equations for _Z_ with the interventional values _z_, resulting in the modified set of equations _F_ _z_ ;
3. **Prediction** : compute the implied distribution on the remaining elements of _V_ using _F_ _z_ and the
posterior _P_ ( _U |W_ = _w_ ).


**3** **Counterfactual Fairness**


Given a predictive problem with fairness considerations, where _A_, _X_ and _Y_ represent the protected
attributes, remaining attributes, and output of interest respectively, let us assume that we are given a
causal model ( _U, V, F_ ), where _V ≡_ _A ∪_ _X_ . We postulate the following criterion for predictors of _Y_ .

**Definition 5** (Counterfactual fairness) **.** _Predictor_ _Y_ [ˆ] _is_ **counterfactually fair** _if under any context_
_X_ = _x and A_ = _a,_


_P_ ( _Y_ [ˆ] _A←a_ ( _U_ ) = _y | X_ = _x, A_ = _a_ ) = _P_ ( _Y_ [ˆ] _A←a_ _′_ ( _U_ ) = _y | X_ = _x, A_ = _a_ ) _,_ (1)


_for all y and for any value a_ _[′]_ _attainable by A._


This notion is closely related to **actual causes** [ 13 ], or token causality in the sense that, to be fair,
_A_ should not be a cause of _Y_ [ˆ] in any individual instance. In other words, changing _A_ while holding
things which are not causally dependent on _A_ constant will not change the distribution of _Y_ [ˆ] . We also
emphasize that counterfactual fairness is an individual-level definition. This is substantially different
from comparing different individuals that happen to share the same “treatment” _A_ = _a_ and coincide
on the values of _X_, as discussed in Section 4.3.1 of [ 29 ] and the Supplementary Material. Differences
between _X_ _a_ and _X_ _a_ _′_ must be caused by variations on _A_ only. Notice also that this definition is
agnostic with respect to how good a predictor _Y_ [ˆ] is, which we discuss in Section 4.


**Relation to individual fairness** . IF is agnostic with respect to its notion of similarity metric, which
is both a strength (generality) and a weakness (no unified way of defining similarity). Counterfactuals
and similarities are related, as in the classical notion of distances between “worlds” corresponding to
different counterfactuals [ 23 ]. If _Y_ [ˆ] is a deterministic function of _W ⊂_ _A ∪_ _X ∪_ _U_, as in several of


3


### _U A_







( _c_ )









( _d_ ) ( _e_ )



( _a_ ) ( _b_ )



Figure 1: (a), (b) Two causal models for different real-world fair prediction scenarios. See Section 3.1
for discussion. (c) The graph corresponding to a causal model with _A_ being the protected attribute and
_Y_ some outcome of interest, with background variables assumed to be independent. (d) Expanding
the model to include an intermediate variable indicating whether the individual is employed with
two (latent) background variables **Prejudiced** (if the person offering the job is prejudiced) and
**Qualifications** (a measure of the individual’s qualifications). (e) A twin network representation of
this system [ 28 ] under two different counterfactual levels for _A_ . This is created by copying nodes
descending from _A_, which inherit unaffected parents from the factual world.


our examples to follow, then IF can be defined by treating equally two individuals with the same _W_
in a way that is also counterfactually fair.


**Relation to Pearl et al. [29]** . In Example 4.4.4 of [ 29 ], the authors condition instead on _X_, _A_, and
the observed realization of _Y_ [ˆ], and calculate the probability of the counterfactual realization _Y_ [ˆ] _A←a_ _[′]_
differing from the factual. This example conflates the predictor _Y_ [ˆ] with the outcome _Y_, of which
we remain agnostic in our definition but which is used in the construction of _Y_ [ˆ] as in Section 4. Our
framing makes the connection to machine learning more explicit.


**3.1** **Examples**


To provide an intuition for counterfactual fairness, we will consider two real-world fair prediction scenarios: **insurance pricing** and **crime prediction** . Each of these correspond to one of the two causal
graphs in Figure 1(a),(b). The Supplementary Material provides a more mathematical discussion of
these examples with more detailed insights.


**Scenario 1: The Red Car.** A car insurance company wishes to price insurance for car owners
by predicting their accident rate _Y_ . They assume there is an unobserved factor corresponding to
aggressive driving _U_, that (a) causes drivers to be more likely have an accident, and (b) causes
individuals to prefer red cars (the observed variable _X_ ). Moreover, individuals belonging to a
certain race _A_ are more likely to drive red cars. However, these individuals are no more likely to be
aggressive or to get in accidents than any one else. We show this in Figure 1(a). Thus, using the
red car feature _X_ to predict accident rate _Y_ would seem to be an unfair prediction because it may
charge individuals of a certain race more than others, even though no race is more likely to have an
accident. Counterfactual fairness agrees with this notion: changing _A_ while holding _U_ fixed will also
change _X_ and, consequently, _Y_ [ˆ] . Interestingly, we can show (Supplementary Material) that in a linear
model, regressing _Y_ on _A_ and _X_ is equivalent to regressing on _U_, so off-the-shelf regression here is
counterfactually fair. Regressing _Y_ on _X_ alone obeys the FTU criterion but is not counterfactually
fair, so _omitting A (FTU) may introduce unfairness into an otherwise fair world._


**Scenario 2: High Crime Regions.** A city government wants to estimate crime rates by neighborhood to allocate policing resources. Its analyst constructed training data by merging (1) a registry of
residents containing their neighborhood _X_ and race _A_, with (2) police records of arrests, giving each
resident a binary label with _Y_ = 1 indicating a criminal arrest record. Due to historically segregated
housing, the location _X_ depends on _A_ . Locations _X_ with more police resources have larger numbers
of arrests _Y_ . And finally, _U_ represents the totality of socioeconomic factors and policing practices
that both influence where an individual may live and how likely they are to be arrested and charged.
This can all be seen in Figure 1(b).


In this example, higher observed arrest rates in some neighborhoods are due to greater policing there,
not because people of different races are any more or less likely to break the law. The label _Y_ = 0


4


does not mean someone has never committed a crime, but rather that they have not been caught. _If_
_individuals in the training data have not already had equal opportunity, algorithms enforcing EO will_
_not remedy such unfairness_ . In contrast, a counterfactually fair approach would model differential
enforcement rates using _U_ and base predictions on this information rather than on _X_ directly.


In general, we need a multistage procedure in which we first derive latent variables _U_, and then based
on them we minimize some loss with respect to _Y_ . This is the core of the algorithm discussed next.


**3.2** **Implications**


One simple but important implication of the definition of counterfactual fairness is the following:

**Lemma 1.** _Let_ _G_ _be the causal graph of the given model_ ( _U, V, F_ ) _. Then_ _Y_ [ˆ] _will be counterfactually_
_fair if it is a function of the non-descendants of A._


_Proof._ Let _W_ be any non-descendant of _A_ in _G_ . Then _W_ _A←a_ ( _U_ ) and _W_ _A←a_ _′_ ( _U_ ) have the same
distribution by the three inferential steps in Section 2.2. Hence, the distribution of any function _Y_ [ˆ] of
the non-descendants of _A_ is invariant with respect to the counterfactual values of _A_ .


This does not exclude using a descendant _W_ of _A_ as a possible input to _Y_ [ˆ] . However, this will only
be possible in the case where the overall dependence of _Y_ [ˆ] on _A_ disappears, which will not happen in
general. Hence, Lemma 1 provides the most straightforward way to achieve counterfactual fairness.
In some scenarios, it is desirable to define path-specific variations of counterfactual fairness that allow
for the inclusion of some descendants of _A_, as discussed by [ 21, 27 ] and the Supplementary Material.


**Ancestral closure of protected attributes.** Suppose that a parent of a member of _A_ is not in _A_ .
Counterfactual fairness allows for the use of it in the definition of _Y_ [ˆ] . If this seems counterintuitive,
then we argue that the fault should be at the postulated set of protected attributes rather than with the
definition of counterfactual fairness, and that typically we should expect set _A_ to be closed under
ancestral relationships given by the causal graph. For instance, if _Race_ is a protected attribute, and
_Mother’s race_ is a parent of _Race_, then it should also be in _A_ .

_Y_ **Dealing with historical biases and an existing fairness paradox.** ˆ and _Y_ allows us to tackle historical biases. For instance, let _Y_ be an indicator of whether a client The explicit difference between
defaults on a loan, while _Y_ [ˆ] is the actual decision of giving the loan. Consider the DAG _A →_ _Y_,
shown in Figure 1(c) with the explicit inclusion of set _U_ of independent background variables. _Y_ is
the objectively ideal measure for decision making, the binary indicator of the event that the individual
defaults on a loan. If _A_ is postulated to be a protected attribute, then the predictor _Y_ [ˆ] = _Y_ = _f_ _Y_ ( _A, U_ )
is not counterfactually fair, with the arrow _A →_ _Y_ being (for instance) the result of a world that
punishes individuals in a way that is out of their control. Figure 1(d) shows a finer-grained model,
where the path is mediated by a measure of whether the person is employed, which is itself caused
by two background factors: one representing whether the person hiring is prejudiced, and the other
the employee’s qualifications. In this world, _A_ is a cause of defaulting, even if mediated by other
variables [3] . The counterfactual fairness principle however forbids us from using _Y_ : using the twin
network [4] of Pearl [28], we see in Figure 1(e) that _Y_ _a_ and _Y_ _a_ _′_ need not be identically distributed
given the background variables.


In contrast, any function of variables not descendants of _A_ can be used a basis for fair decision
making. This means that any variable _Y_ [ˆ] defined by _Y_ [ˆ] = _g_ ( _U_ ) will be counterfactually fair for any
function _g_ ( _·_ ) . Hence, given a causal model, the functional defined by the function _g_ ( _·_ ) minimizing
some predictive error for _Y_ will satisfy the criterion, as proposed in Section 4.1. We are essentially
learning a projection of _Y_ into the space of fair decisions, removing historical biases as a by-product.


Counterfactual fairness also provides an answer to some problems on the incompatibility of fairness
criteria. In particular, consider the following problem raised independently by different authors (e.g.,


3
For example, if the function determining employment _f_ _E_ ( _A, P, Q_ ) _≡_ _I_ ( _Q>_ 0 _,P_ =0 or _A_ = _̸_ _a_ ) then an individual
with sufficient qualifications and prejudiced potential employer may have a different counterfactual employment
value for _A_ = _a_ compared to _A_ = _a_ _[′]_, and a different chance of default.
4 In a nutshell, this is a graph that simultaneously depicts “multiple worlds” parallel to the factual realizations.
In this graph, all multiple worlds share the same background variables, but with different consequences in the
remaining variables depending on which counterfactual assignments are provided.


5


[ 7, 22 ]), illustrated below for the binary case: ideally, we would like our predictors to obey both
Equality of Opportunity and the _predictive parity_ criterion defined by satisfying


_P_ ( _Y_ = 1 _|_ _Y_ [ˆ] = 1 _, A_ = 1) = _P_ ( _Y_ = 1 _|_ _Y_ [ˆ] = 1 _, A_ = 0) _,_


as well as the corresponding equation for _Y_ [ˆ] = 0 . It has been shown that if _Y_ and _A_ are marginally
associated (e.g., recidivism and race are associated) and _Y_ is not a deterministic function of _Y_ [ˆ],
then the two criteria cannot be reconciled. Counterfactual fairness throws a light in this scenario,
suggesting that both EO and predictive parity may be insufficient if _Y_ and _A_ are associated: assuming
that _A_ and _Y_ are unconfounded (as expected for demographic attributes), this is the result of _A_ being
a cause of _Y_ . By counterfactual fairness, we should _not_ want to use _Y_ as a basis for our decisions,
instead aiming at some function _Y_ ˆ is defined in such a way that is an estimate of the “closest” _Y_ _⊥_ _A_ of variables which are not caused by _Y_ _⊥_ _A_ to _Y_ according to some preferred _A_ but are predictive of _Y_ .
risk function. _This makes the incompatibility between EO and predictive parity irrelevant_, as _A_ and
_Y_ _⊥_ _A_ will be independent by construction given the model assumptions.


**4** **Implementing Counterfactual Fairness**


As discussed in the previous Section, we need to relate _Y_ [ˆ] to _Y_ if the predictor is to be useful, and we
restrict _Y_ [ˆ] to be a (parameterized) function of the non-descendants of _A_ in the causal graph following
Lemma 1. We next introduce an algorithm, then discuss assumptions that can be used to express
counterfactuals.


**4.1** **Algorithm**


Let _Y_ [ˆ] _≡_ _g_ _θ_ ( _U, X_ ⊁ _A_ ) be a predictor parameterized by _θ_, such as a logistic regression or a neural
network, and where _X_ ⊁ _A_ _⊆_ _X_ are non-descendants of _A_ . Given a loss function _l_ ( _·, ·_ ) such as
squared loss or log-likelihood, and training data _D ≡{_ ( _A_ [(] _[i]_ [)] _, X_ [(] _[i]_ [)] _, Y_ [(] _[i]_ [)] ) _}_ for _i_ = 1 _,_ 2 _, . . ., n_, we
define _L_ ( _θ_ ) _≡_ [�] _[n]_ _i_ =1 [E][[] _[l]_ [(] _[y]_ [(] _[i]_ [)] _[, g]_ _[θ]_ [(] _[U]_ [ (] _[i]_ [)] _[, x]_ [(] ⊁ _[i]_ [)] _A_ [))] _[ |][ x]_ [(] _[i]_ [)] _[, a]_ [(] _[i]_ [)] []] _[/n]_ [ as the empirical loss to be minimized]
with respect to _θ_ . Each expectation is with respect to random variable _U_ [(] _[i]_ [)] _∼_ _P_ _M_ ( _U | x_ [(] _[i]_ [)] _, a_ [(] _[i]_ [)] )
where _P_ _M_ ( _U | x, a_ ) is the conditional distribution of the background variables as given by a causal
model _M_ that is available by assumption. If this expectation cannot be calculated analytically,
Markov chain Monte Carlo (MCMC) can be used to approximate it as in the following algorithm.

1: **procedure** F AIR L EARNING ( _D, M_ ) _▷_ Learned parameters _θ_ [ˆ]

2: For each data point _i ∈D_, sample _m_ MCMC samples _U_ 1 [(] _[i]_ [)] _[, . . ., U]_ _m_ [ (] _[i]_ [)] _[∼]_ _[P]_ _M_ [(] _[U][ |][ x]_ [(] _[i]_ [)] _[, a]_ [(] _[i]_ [)] [)][.]
3: Let _D_ _[′]_ be the augmented dataset where each point ( _a_ [(] _[i]_ [)] _, x_ [(] _[i]_ [)] _, y_ [(] _[i]_ [)] ) in _D_ is replaced with the
corresponding _m_ points _{_ ( _a_ [(] _[i]_ [)] _, x_ [(] _[i]_ [)] _, y_ [(] _[i]_ [)] _, u_ [(] _j_ _[i]_ [)] [)] _[}]_ [.]


ˆ
4: _θ ←_ argmin _θ_ � _i_ _[′]_ _∈D_ _[′]_ _[ l]_ [(] _[y]_ [(] _[i]_ _[′]_ [)] _[, g]_ _[θ]_ [(] _[U]_ [ (] _[i]_ _[′]_ [)] _[, x]_ [(] ⊁ _[i]_ _[′]_ _A_ [)] [))][.]

5: **end procedure**


At prediction time, we report _Y_ [˜] _≡_ E[ _Y_ [ˆ] ( _U_ _[⋆]_ _, x_ _[⋆]_ ⊁ _A_ [)] _[ |][ x]_ _[⋆]_ _[, a]_ _[⋆]_ []][ for a new data point][ (] _[a]_ _[⋆]_ _[, x]_ _[⋆]_ [)][.]


**Deconvolution perspective.** The algorithm can be understood as a deconvolution approach that,
given observables _A ∪_ _X_, extracts its latent sources and pipelines them into a predictive model. We
advocate that _counterfactual assumptions must underlie all approaches that claim to extract the_
_sources of variation of the data as “fair” latent components_ . As an example, Louizos et al. [24] start
from the DAG _A →_ _X ←_ _U_ to extract _P_ ( _U | X, A_ ) . As _U_ and _A_ are not independent given _X_ in this
representation, a type of penalization is enforced to create a posterior _P_ _fair_ ( _U |A, X_ ) that is close
to the model posterior _P_ ( _U | A, X_ ) while satisfying _P_ _fair_ ( _U |A_ = _a, X_ ) _≈_ _P_ _fair_ ( _U |A_ = _a_ _[′]_ _, X_ ) .
But _this is neither necessary nor sufficient for counterfactual fairness_ . The model for _X_ given _A_
and _U_ must be justified by a causal mechanism, and that being the case, _P_ ( _U | A, X_ ) requires no
postprocessing. As a matter of fact, model _M_ can be learned by penalizing empirical dependence
measures between _U_ and _pa_ _i_ for a given _V_ _i_ (e.g. Mooij et al. [26] ), but this concerns _M_ and not _Y_ [ˆ],
and is motivated by explicit assumptions about structural equations, as described next.


6


**4.2** **Designing the Input Causal Model**


Model _M_ must be provided to algorithm F AIR L EARNING . Although this is well understood, it is
worthwhile remembering that causal models always require strong assumptions, even more so when
making counterfactual claims [ 8 ]. Counterfactuals assumptions such as structural equations are in
general unfalsifiable even if interventional data for all variables is available. This is because there
are infinitely many structural equations compatible with the same observable distribution [ 28 ], be it
observational or interventional. Having passed testable implications, the remaining components of a
counterfactual model should be understood as conjectures formulated according to the best of our
knowledge. Such models should be deemed provisional and prone to modifications if, for example,
new data containing measurement of variables previously hidden contradict the current model.


We point out that we do not need to specify a fully deterministic model, and structural equations can
be relaxed as conditional distributions. In particular, the concept of counterfactual fairness holds
under three levels of assumptions of increasing strength:


**Level 1.** Build _Y_ [ˆ] using only the observable non-descendants of _A_ . This only requires partial
causal ordering and no further causal assumptions, but in many problems there will be few, if any,
observables which are not descendants of protected demographic factors.


**Level 2.** Postulate background latent variables that act as non-deterministic causes of observable
variables, based on explicit domain knowledge and learning algorithms [5] . Information about _X_ is
passed to _Y_ [ˆ] via _P_ ( _U | x, a_ ).


**Level 3.** Postulate a fully deterministic model with latent variables. For instance, the distribution
_P_ ( _V_ _i_ _| pa_ _i_ ) can be treated as an additive error model, _V_ _i_ = _f_ _i_ ( _pa_ _i_ )+ _e_ _i_ [ 31 ]. The error term _e_ _i_ then
becomes an input to _Y_ [ˆ] as calculated from the observed variables. This maximizes the information
extracted by the fair predictor _Y_ [ˆ] .


**4.3** **Further Considerations on Designing the Input Causal Model**


One might ask what we can lose by defining causal fairness measures involving only noncounterfactual causal quantities, such as enforcing _P_ ( _Y_ [ˆ] = 1 _| do_ ( _A_ = _a_ )) = _P_ ( _Y_ [ˆ] = 1 _| do_ ( _A_ = _a_ _[′]_ ))
instead of our counterfactual criterion. The reason is that the above equation is only a constraint
on an average effect. Obeying this criterion provides no guarantees against, for example, having
half of the individuals being strongly “negatively” discriminated and half of the individuals strongly
“positively” discriminated. We advocate that, for fairness, society should not be satisfied in pursuing
only counterfactually-free guarantees. While one may be willing to claim posthoc that the equation
above masks no balancing effect so that individuals receive approximately the same distribution of
outcomes, _that itself is just a counterfactual claim in disguise._ Our approach is to make counterfactual
assumptions explicit. When unfairness is judged to follow only some “pathways” in the causal graph
(in a sense that can be made formal, see [ 21, 27 ]), nonparametric assumptions about the independence
of counterfactuals may suffice, as discussed by [ 27 ]. In general, nonparametric assumptions may not
provide identifiable adjustments even in this case, as also discussed in our Supplementary Material.
If competing models with different untestable assumptions are available, there are ways of simultaneously enforcing a notion of approximate counterfactual fairness in all of them, as introduced by us in

[32]. Other alternatives include exploiting bounds on the contribution of hidden variables [29, 33].


Another issue is the interpretation of causal claims involving demographic variables such as race
and sex. Our view is that such constructs are the result of translating complex events into random
variables and, despite some controversy, we consider counterproductive to claim that e.g. race and sex
cannot be causes. An idealized intervention on some _A_ at a particular time can be seen as a notational
shortcut to express a conjunction of more specific interventions, which may be individually doable
but jointly impossible in practice. It is the plausibility of complex, even if impossible to practically
manipulate, causal chains from _A_ to _Y_ that allows us to claim that unfairness is real [ 11 ]. Experiments
for constructs exist, such as randomizing names in job applications to make them race-blind. They do
not contradict the notion of race as a cause, and can be interpreted as an intervention on a particular
aspect of the construct “race,” such as “race perception” (e.g. Section 4.4.4 of [29]).


5 In some domains, it is actually common to build a model entirely around latent constructs with few or no
observable parents nor connections among observed variables [2].


7


**5** **Illustration: Law School Success**


We illustrate our approach on a practical problem that requires fairness, the _prediction of success in_
_law school_ . A second problem, _understanding the contribution of race to police stops_, is described in
the Supplementary Material. Following closely the usual framework for assessing causal models in
the machine learning literature, the goal of this experiment is to quantify how our algorithm behaves
with finite sample sizes while assuming ground truth compatible with a synthetic model.


**Problem definition: Law school success**


The Law School Admission Council conducted a survey across 163 law schools in the United States

[ 35 ]. It contains information on 21,790 law students such as their entrance exam scores (LSAT), their
grade-point average (GPA) collected prior to law school, and their first year average grade (FYA).


Given this data, a school may wish to predict if an applicant will have a high FYA. The school would
also like to make sure these predictions are not biased by an individual’s race and sex. However, the
LSAT, GPA, and FYA scores, may be biased due to social factors. We compare our framework with
two unfair baselines: 1. **Full** : the standard technique of using all features, including sensitive features
such as race and sex to make predictions; 2. **Unaware** : fairness through unawareness, where we
do not use race and sex as features. For comparison, we generate predictors _Y_ [ˆ] for all models using
logistic regression.


**Fair prediction.** As described in Section 4.2, there are three ways in which we can model a
counterfactually fair predictor of FYA. Level 1 uses any features which are not descendants of race
and sex for prediction. Level 2 models latent ‘fair’ variables which are parents of observed variables.
These variables are independent of both race and sex. Level 3 models the data using an additive error
model, and uses the independent error terms to make predictions. These models make increasingly
strong assumptions corresponding to increased predictive power. We split the dataset 80/20 into a
train/test set, preserving label balance, to evaluate the models.


As we believe LSAT, GPA, and FYA are all biased by race and sex, we cannot use any observed
features to construct a counterfactually fair predictor as described in Level 1.


In Level 2, we postulate that a latent variable: a student’s **knowledge** (K), affects GPA, LSAT, and
FYA scores. The causal graph corresponding to this model is shown in Figure 2, ( **Level 2** ). This is a
short-hand for the distributions:


GPA _∼N_ ( _b_ _G_ + _w_ _G_ _[K]_ _[K]_ [ +] _[ w]_ _G_ _[R]_ _[R]_ [ +] _[ w]_ _G_ _[S]_ _[S, σ]_ _[G]_ [)] _[,]_ FYA _∼N_ ( _w_ _F_ _[K]_ _[K]_ [ +] _[ w]_ _F_ _[R]_ _[R]_ [ +] _[ w]_ _F_ _[S]_ _[S,]_ [ 1)] _[,]_
LSAT _∼_ Poisson(exp( _b_ _L_ + _w_ _L_ _[K]_ _[K]_ [ +] _[ w]_ _L_ _[R]_ _[R]_ [ +] _[ w]_ _L_ _[S]_ _[S]_ [))] _[,]_ K _∼N_ (0 _,_ 1)


We perform inference on this model using an observed training set to estimate the posterior distribution
of _K_ . We use the probabilistic programming language Stan [ 34 ] to learn _K_ . We call the predictor
constructed using _K_, **Fair** _K_ .


black _$_ white asian _$_ white mexican _$_ white female _$_ male















































**Level 2** **Level 3**









V FYA pred_zfya V FYA





V FYA V pred_zfya FYA





V FYA V FYA pred_zfya V FYA pred_zfya V FYA V FYA V pred_zfya FYA


|3<br>2 dende snisi tty y<br>1<br>0|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|3<br>ype2 original e sns iit tyy|Col11|Col12|Col13|Col14|Col15|Col16|3<br>type2 original e sns iit tyy|Col18|Col19|Col20|Col21|Col22|Col23|3<br>type2 original e sns iit tyy|Col25|Col26|Col27|Col28|Col29|typeoriginal|Col31|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0<br>1<br>2<br>3<br><br>density<br>**density**<br>|||||||||||||||||||||||||||||||
|0<br>1<br>2<br>3<br><br>density<br>**density**<br>||||||||t<br>|t<br>||||||||||||||||||||||
|0<br>1<br>2<br>3<br><br>density<br>**density**<br>|||||||||swapped<br>0<br>1<br>−1.0<br>d<br>**den**|||||||swapped<br>0<br>1<br>−1.0<br>d<br>**den**|||||||swapped<br>0<br>1<br>−1.0<br>d<br>**den**||||||0.5<br>swapped<br>**original**<br>**data**<br>**counter-**<br>**factual**|0.5<br>swapped<br>**original**<br>**data**<br>**counter-**<br>**factual**|
|0<br>1<br>2<br>3<br><br>density<br>**density**<br>|||||||||||||||||||||||||||||||
|0<br>1<br>2<br>3<br><br>density<br>**density**<br>||−1.0|−|0.5<br>pred_|0.<br>zfya|0|0.5|||−|0.5<br>pred|_zfya|0.0||0.5|0.5|−0|.5<br>pred|0<br>_zfya|.0||0.5|0.5|−|0.5<br>pre|d_zfya|0.0|||**ounter-**<br>**factual**|
|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|||||||t<br>|ype<br><br>1.5<br>2.0<br>sity<br>**ity**|||||||type<br><br>1.5<br>2.0<br>sity<br>**ity**|||||||type<br><br>1.5<br>2.0<br>sity<br>**ity**||||||||
|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>||||||||original<br>sw~~apped~~<br>0.0<br>0.5<br>1.0<br>den<br>**den**|||||||original<br>swapped<br>0.0<br>0.5<br>1.0<br>den<br>**den**|||||||original<br>sw~~apped~~<br>0.0<br>0.5<br>1.0<br>den<br>**den**||||||||
|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>||||||||||||||||||||||||||||||
|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>||||||||||||||||||||||||||||||
|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|0.0<br>0.5<br>1.0<br>1.5<br>2.0<br>−0.5<br>0.0<br>0.5<br>pred_zfya<br>density<br>t<br><br>V<br>**density**<br>|||||||||−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|−0.4<br>0.0<br>0.4<br>0.8<br>pred_zfya<br><br>V<br><br>V|



Figure 2: **Left:** A causal model for the problem of predicting law school success fairly. **Right:**
Density plots of predicted FYA _a_ and FYA _a_ _′_ .


In Level 3, we model GPA, LSAT, and FYA as continuous variables with additive error terms
independent of race and sex (that may in turn be correlated with one-another). This model is shown


8


Table 1: Prediction results using logistic regression. Note that we must sacrifice a small amount of
accuracy to ensuring counterfactually fair prediction (Fair _K_, Fair Add), versus the models that use
unfair features: GPA, LSAT, race, sex (Full, Unaware).

**Full** **Unaware** **Fair** _K_ **Fair Add**

RMSE 0.873 0.894 0.929 0.918


in Figure 2, ( **Level 3** ), and is expressed by:


GPA = _b_ _G_ + _w_ _G_ _[R]_ _[R]_ [ +] _[ w]_ _G_ _[S]_ _[S]_ [ +] _[ ϵ]_ _[G]_ _[, ϵ]_ _[G]_ _[∼]_ _[p]_ [(] _[ϵ]_ _[G]_ [)]

LSAT = _b_ _L_ + _w_ _L_ _[R]_ _[R]_ [ +] _[ w]_ _L_ _[S]_ _[S]_ [ +] _[ ϵ]_ _[L]_ _[, ϵ]_ _[L]_ _[∼]_ _[p]_ [(] _[ϵ]_ _[L]_ [)]

FYA = _b_ _F_ + _w_ _F_ _[R]_ _[R]_ [ +] _[ w]_ _F_ _[S]_ _[S]_ [ +] _[ ϵ]_ _[F]_ _[, ϵ]_ _[F]_ _[∼]_ _[p]_ [(] _[ϵ]_ _[F]_ [)]


We estimate the error terms _ϵ_ _G_ _, ϵ_ _L_ by first fitting two models that each use race and sex to individually
predict GPA and LSAT. We then compute the residuals of each model (e.g., _ϵ_ _G_ = GPA _−Y_ [ˆ] GPA ( _R, S_ ) ).
We use these residual estimates of _ϵ_ _G_ _, ϵ_ _L_ to predict FYA. We call this _Fair Add_ .


**Accuracy.** We compare the RMSE achieved by logistic regression for each of the models on the test
set in Table 1. The **Full** model achieves the lowest RMSE as it uses race and sex to more accurately
reconstruct FYA. Note that in this case, this model is not fair even if the data was generated by one of
the models shown in Figure 2 as it corresponds to Scenario 3. The (also unfair) **Unaware** model still
uses the unfair variables GPA and LSAT, but because it does not use race and sex it cannot match the
RMSE of the **Full** model. As our models satisfy counterfactual fairness, they trade off some accuracy.
Our first model **Fair** _K_ uses weaker assumptions and thus the RMSE is highest. Using the Level 3
assumptions, as in **Fair Add** we produce a counterfactually fair model that trades slightly stronger
assumptions for lower RMSE.


**Counterfactual fairness.** We would like to empirically test whether the baseline methods are
counterfactually fair. To do so we will assume the true model of the world is given by Figure 2,
( **Level 2** ). We can fit the parameters of this model using the observed data and evaluate counterfactual
fairness by sampling from it. Specifically, we will generate samples from the model given either
the observed race and sex, or _counterfactual_ race and sex variables. We will fit models to both the
original and counterfactual sampled data and plot how the distribution of predicted FYA changes for
both baseline models. Figure 2 shows this, where each row corresponds to a baseline predictor and
each column corresponds to the counterfactual change. In each plot, the blue distribution is density of
predicted FYA for the original data and the red distribution is this density for the counterfactual data. If
a model is counterfactually fair we would expect these distributions to lie exactly on top of each other.
Instead, we note that the **Full** model exhibits counterfactual unfairness for all counterfactuals except
sex. We see a similar trend for the **Unaware** model, although it is closer to being counterfactually
fair. To see why these models seem to be fair w.r.t. to sex we can look at weights of the DAG which
generates the counterfactual data. Specifically the DAG weights from (male,female) to GPA are
( 0 _._ 93, 1 _._ 06 ) and from (male,female) to LSAT are ( 1 _._ 1, 1 _._ 1 ). Thus, these models are fair w.r.t. to sex
simply because of a very weak causal link between sex and GPA/LSAT.


**6** **Conclusion**


We have presented a new model of fairness we refer to as _counterfactual fairness_ . It allows us
to propose algorithms that, rather than simply ignoring protected attributes, are able to take into
account the different social biases that may arise towards individuals based on ethically sensitive
attributes and compensate for these biases effectively. We experimentally contrasted our approach
with previous fairness approaches and show that our explicit causal models capture these social biases
and make clear the implicit trade-off between prediction accuracy and fairness in an unfair world. We
propose that fairness should be regulated by explicitly modeling the causal structure of the world.
Criteria based purely on probabilistic independence cannot satisfy this and are unable to address _how_
unfairness is occurring in the task at hand. By providing such causal tools for addressing fairness
questions we hope we can provide practitioners with customized techniques for solving a wide array
of fairness modeling problems.


9


**Acknowledgments**


This work was supported by the Alan Turing Institute under the EPSRC grant EP/N510129/1. CR
acknowledges additional support under the EPSRC Platform Grant EP/P022529/1. We thank Adrian
Weller for insightful feedback, and the anonymous reviewers for helpful comments.


**References**


[1] Berk, R., Heidari, H., Jabbari, S., Kearns, M., and Roth, A. Fairness in criminal justice risk
assessments: The state of the art. _arXiv:1703.09207v1_, 2017.


[2] Bollen, K. _Structural Equations with Latent Variables_ . John Wiley & Sons, 1989.


[3] Bollen, K. and (eds.), J. Long. _Testing Structural Equation Models_ . SAGE Publications, 1993.


[4] Bolukbasi, Tolga, Chang, Kai-Wei, Zou, James Y, Saligrama, Venkatesh, and Kalai, Adam T.
Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In
_Advances in Neural Information Processing Systems_, pp. 4349–4357, 2016.


[5] Brennan, Tim, Dieterich, William, and Ehret, Beate. Evaluating the predictive validity of the
compas risk and needs assessment system. _Criminal Justice and Behavior_, 36(1):21–40, 2009.


[6] Calders, Toon and Verwer, Sicco. Three naive bayes approaches for discrimination-free classification. _Data Mining and Knowledge Discovery_, 21(2):277–292, 2010.


[7] Chouldechova, A. Fair prediction with disparate impact: a study of bias in recidivism prediction
instruments. _Big Data_, 2:153–163, 2017.


[8] Dawid, A. P. Causal inference without counterfactuals. _Journal of the American Statistical_
_Association_, pp. 407–448, 2000.


[9] DeDeo, Simon. Wrong side of the tracks: Big data and protected categories. _arXiv preprint_
_arXiv:1412.4643_, 2014.


[10] Dwork, Cynthia, Hardt, Moritz, Pitassi, Toniann, Reingold, Omer, and Zemel, Richard. Fairness
through awareness. In _Proceedings of the 3rd Innovations in Theoretical Computer Science_
_Conference_, pp. 214–226. ACM, 2012.


[11] Glymour, C. and Glymour, M. R. Commentary: Race and sex are causes. _Epidemiology_, 25(4):
488–490, 2014.


[12] Grgic-Hlaca, Nina, Zafar, Muhammad Bilal, Gummadi, Krishna P, and Weller, Adrian. The case
for process fairness in learning: Feature selection for fair decision making. _NIPS Symposium on_
_Machine Learning and the Law_, 2016.


[13] Halpern, J. _Actual Causality_ . MIT Press, 2016.


[14] Hardt, Moritz, Price, Eric, Srebro, Nati, et al. Equality of opportunity in supervised learning. In
_Advances in Neural Information Processing Systems_, pp. 3315–3323, 2016.


[15] Johnson, Kory D, Foster, Dean P, and Stine, Robert A. Impartial predictive modeling: Ensuring
fairness in arbitrary models. _arXiv preprint arXiv:1608.00528_, 2016.


[16] Joseph, Matthew, Kearns, Michael, Morgenstern, Jamie, Neel, Seth, and Roth, Aaron. Rawlsian
fairness for machine learning. _arXiv preprint arXiv:1610.09559_, 2016.


[17] Kamiran, Faisal and Calders, Toon. Classifying without discriminating. In _Computer, Control_
_and Communication, 2009. IC4 2009. 2nd International Conference on_, pp. 1–6. IEEE, 2009.


[18] Kamiran, Faisal and Calders, Toon. Data preprocessing techniques for classification without
discrimination. _Knowledge and Information Systems_, 33(1):1–33, 2012.


[19] Kamishima, Toshihiro, Akaho, Shotaro, and Sakuma, Jun. Fairness-aware learning through
regularization approach. In _Data Mining Workshops (ICDMW), 2011 IEEE 11th International_
_Conference on_, pp. 643–650. IEEE, 2011.


10


[20] Khandani, Amir E, Kim, Adlar J, and Lo, Andrew W. Consumer credit-risk models via
machine-learning algorithms. _Journal of Banking & Finance_, 34(11):2767–2787, 2010.


[21] Kilbertus, N., Carulla, M. R., Parascandolo, G., Hardt, M., Janzing, D., and Schölkopf, B.
Avoiding discrimination through causal reasoning. _Advances in Neural Information Processing_
_Systems 30_, 2017.


[22] Kleinberg, J., Mullainathan, S., and Raghavan, M. Inherent trade-offs in the fair determination
of risk scores. _Proceedings of The 8th Innovations in Theoretical Computer Science Conference_
_(ITCS 2017)_, 2017.


[23] Lewis, D. _Counterfactuals_ . Harvard University Press, 1973.


[24] Louizos, Christos, Swersky, Kevin, Li, Yujia, Welling, Max, and Zemel, Richard. The variational
fair autoencoder. _arXiv preprint arXiv:1511.00830_, 2015.


[25] Mahoney, John F and Mohen, James M. Method and system for loan origination and underwriting, October 23 2007. US Patent 7,287,008.


[26] Mooij, J., Janzing, D., Peters, J., and Scholkopf, B. Regression by dependence minimization
and its application to causal inference in additive noise models. In _Proceedings of the 26th_
_Annual International Conference on Machine Learning_, pp. 745–752, 2009.


[27] Nabi, R. and Shpitser, I. Fair inference on outcomes. _arXiv:1705.10378v1_, 2017.


[28] Pearl, J. _Causality: Models, Reasoning and Inference_ . Cambridge University Press, 2000.


[29] Pearl, J., Glymour, M., and Jewell, N. _Causal Inference in Statistics: a Primer_ . Wiley, 2016.


[30] Pearl, Judea. Causal inference in statistics: An overview. _Statistics Surveys_, 3:96–146, 2009.


[31] Peters, J., Mooij, J. M., Janzing, D., and Schölkopf, B. Causal discovery with continuous
additive noise models. _Journal of Machine Learning Research_, 15:2009–2053, 2014. URL
`[http://jmlr.org/papers/v15/peters14a.html](http://jmlr.org/papers/v15/peters14a.html)` .


[32] Russell, C., Kusner, M., Loftus, J., and Silva, R. When worlds collide: integrating different
counterfactual assumptions in fairness. _Advances in Neural Information Processing Systems_,
31, 2017.


[33] Silva, R. and Evans, R. Causal inference through a witness protection program. _Journal of_
_Machine Learning Research_, 17(56):1–53, 2016.


[34] Stan Development Team. Rstan: the r interface to stan, 2016. R package version 2.14.1.


[35] Wightman, Linda F. Lsac national longitudinal bar passage study. lsac research report series.
1998.


[36] Zafar, Muhammad Bilal, Valera, Isabel, Rodriguez, Manuel Gomez, and Gummadi, Krishna P.
Learning fair classifiers. _arXiv preprint arXiv:1507.05259_, 2015.


[37] Zafar, Muhammad Bilal, Valera, Isabel, Rodriguez, Manuel Gomez, and Gummadi, Krishna P.
Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. _arXiv preprint arXiv:1610.08452_, 2016.


[38] Zemel, Richard S, Wu, Yu, Swersky, Kevin, Pitassi, Toniann, and Dwork, Cynthia. Learning
fair representations. _ICML (3)_, 28:325–333, 2013.


[39] Zliobaite, Indre. A survey on measuring indirect discrimination in machine learning. _arXiv_
_preprint arXiv:1511.00148_, 2015.


11


**S1 Population Level vs Individual Level Causal Effects**


As discussed in Section 3, counterfactual fairness is an individual-level definition. This is fundamentally different from comparing different units that happen to share the same “treatment” _A_ = _a_
and coincide on the values of _X_ . To see in detail what this means, consider the following thought
experiment.


Let us assess the causal effect of _A_ on _Y_ [ˆ] by controlling _A_ at two levels, _a_ and _a_ _[′]_ . In Pearl’s notation,
where “ _do_ ( _A_ = _a_ )” expresses an intervention on _A_ at level _a_, we have that


E[ _Y_ [ˆ] _| do_ ( _A_ = _a_ ) _, X_ = _x_ ] _−_ E[ _Y_ [ˆ] _| do_ ( _A_ = _a_ _[′]_ ) _, X_ = _x_ ] _,_ (2)


is a measure of causal effect, sometimes called the average causal effect (ACE). It expresses the
change that is expected when we intervene on _A_ while observing the attribute set _X_ = _x_, under two
levels of treatment. If this effect is non-zero, _A_ is considered to be a cause of _Y_ [ˆ] .

This raises a subtlety that needs to be addressed: in general, this effect will be non-zero _even if_ _Y_ [ˆ] _is_
_counterfactually fair_ . This may sound counter-intuitive: protected attributes such as race and gender
are causes of our counterfactually fair decisions.


In fact, this is not a contradiction, as the ACE in Equation (2) is different from counterfactual effects.
The ACE contrasts two independent exchangeable units of the population, and it is a perfectly
valid way of performing decision analysis. However, the value of _X_ = _x_ is affected by different
background variables corresponding to different individuals. That is, the causal effect (2) contrasts
two units that receive different treatments but which happen to coincide on _X_ = _x_ . To give a synthetic
example, imagine the simple structural equation


_X_ = _A_ + _U._


The ACE quantifies what happens among people with _U_ = _x −_ _a_ against people with _U_ _[′]_ = _x −_ _a_ _[′]_ .
If, for instance, _Y_ [ˆ] = _λU_ for _λ ̸_ = 0, then the effect (2) is _λ_ ( _a −_ _a_ _[′]_ ) _̸_ = 0.


Contrary to that, the counterfactual difference is zero. That is,


E[ _Y_ [ˆ] _A←a_ ( _U_ ) _| A_ = _a, X_ = _x_ ] _−_ E[ _Y_ [ˆ] _A←a_ _′_ ( _U_ ) _| A_ = _a, X_ = _x_ ] = _λU −_ _λU_ = 0 _._


In another perspective, we can interpret the above just as if we had _measured_ _U_ from the beginning
rather than performing abduction. We then generate _Y_ [ˆ] from some _g_ ( _U_ ), so _U_ is the within-unit cause
of _Y_ [ˆ] and not _A_ .


If _U_ cannot be deterministically derived from _{A_ = _a, X_ = _x}_, the reasoning is similar. By
abduction, the distribution of _U_ will typically depend on _A_, and hence so will _Y_ [ˆ] when marginalizing
over _U_ . Again, this seems to disagree with the intuition that our predictor should be not be caused by
_A_ . However, this once again is a comparison _across individuals_, not within an individual.


It is this balance among ( _A, X, U_ ) that explains, in the examples of Section 3.1, why some predictors
are counterfactually fair even though they are functions of the same variables _{A, X}_ used by unfair
predictors: such functions must correspond to particular ways of balancing the observables that, by
way of the causal assumptions, cancel out the effect of _A_ .


**More on conditioning and alternative definitions.** As discussed in Example 4.4.4 of Pearl et al.

[29], a different proposal for assessing fairness can be defined via the following concept:
**Definition 6** (Probability of sufficiency) **.** _We define the probability of event_ _{A_ = _a}_ _being a_
sufficient cause _for our decision_ _Y_ [ˆ] _, contrasted against {A_ = _a_ _[′]_ _}, as_


_P_ ( _Y_ [ˆ] _A←a_ _′_ ( _U_ ) _̸_ = _y | X_ = _x, A_ = _a,_ _Y_ [ˆ] = _y_ ) _._ (3)


We can then, for instance, claim that _Y_ [ˆ] is a fair predictor if this probability is below some pre-specified
bound for all ( _x, a, a_ _[′]_ ) . The shortcomings of this definition come from its original motivation: to
_explain_ the behavior of an _existing_ decision protocol, where _Y_ [ˆ] is the current practice and which in
a unclear way is conflated with _Y_ . The implication is that if _Y_ [ˆ] is to be designed instead of being a
natural measure of existing behaviour, then we are using _Y_ [ˆ] itself as evidence for the background


12


variables _U_ . This does not make sense if _Y_ [ˆ] is yet to be designed by us. If _Y_ [ˆ] is to be interpreted as _Y_,
then this does not provide a clear recipe on how to build _Y_ [ˆ] : while we can use _Y_ to learn a causal
model, we cannot use it to collect training data evidence for _U_ _as the outcome_ _Y_ _will not be available_
_to us at prediction time_ . For this reason, we claim that while probability of sufficiency is useful as a
way of assessing an existing decision making process, it is not as natural as counterfactual fairness in
the context of machine learning.


**Approximate fairness and model validation.** The notion of probability of sufficiency raises the
question on how to define approximate, or high probability, counterfactual fairness. This is an
important question that we address in [ 32 ]. Before defining an approximation, it is important to first
expose in detail what the exact definition is, which is the goal of this paper.


We also do not address the validation of the causal assumptions used by the input causal model of the
F AIR L EARNING algorithm in Section 4.1. The reason is straightforward: this validation is an entirely
self-contained step of the implementation of counterfactual fairness. An extensive literature already
exists in this topic which the practitioner can refer to (a classic account for instance is [ 3 ]), and which
can be used as-is in our context.


The experiments performed in Section 5 can be criticized by the fact that they rely on a model
that obeys our assumptions, and “obviously” our approach should work better than alternatives.
This criticism is not warranted: in machine learning, causal inference is typically assessed through
simulations which assume that the true model lies in the family covered by the algorithm. Algorithms,
including F AIR L EARNING, are justified in the population sense. How different competitors behave
with finite sample sizes is the primary question to be studied in an empirical study of a new concept,
where we control for the correctness of the assumptions. Although sensitivity analysis is important,
there are many degrees of freedom on how this can be done. Robustness issues are better addressed
by extensions focusing on approximate versions of counterfactual fairness. This will be covered in
later work.


**Stricter version.** For completeness of exposition, notice that the definition of counterfactual fairness
could be strengthened to


_P_ ( _Y_ [ˆ] _A←a_ ( _U_ ) = _Y_ [ˆ] _A←a_ _′_ ( _U_ ) _| X_ = _x, A_ = _a_ ) = 1 _._ (4)


This is different from the original definition in the case where _Y_ [ˆ] ( _U_ ) is a random variable with a
different source of randomness for different counterfactuals (for instance, if _Y_ [ˆ] is given by some
black-box function of _U_ with added noise that is independent across each countefactual value of
_A_ ). In such a situation, the event _{Y_ [ˆ] _A←a_ ( _U_ ) = _Y_ [ˆ] _A←a_ _′_ ( _U_ ) _}_ will itself have probability zero even
if _P_ ( _Y_ [ˆ] _A←a_ ( _U_ ) = _y | X_ = _x, A_ = _a_ ) = _P_ ( _Y_ [ˆ] _A←a_ _′_ ( _U_ ) = _y | X_ = _x, A_ = _a_ ) for all _y_ . We do not
consider version (4) as in our view it does not feel as elegant as the original, and it is also unclear
whether adding an independent source of randomness fed to _Y_ [ˆ] would itself be considered unfair.
Moreover, if _Y_ [ˆ] ( _U_ ) is assumed to be a deterministic function of _U_ and _X_, as in F AIR L EARNING,
then the two definitions are the same [6] . Informally, this stricter definition corresponds to a notion
of “almost surely equality” as opposed to “equality in distribution.” Without assuming that _Y_ [ˆ] is a
deterministic function of _U_ and _X_, even the stricter version does not protect us against measure zero
events where the counterfactuals are different. The definition of counterfactual fairness concisely
emphasizes that _U_ can be a random variable, and clarifies which conditional distribution it follows.
Hence, it is our preferred way of introducing the concept even though it does not explicit suggests
whether _Y_ [ˆ] ( _U_ ) has random inputs besides _U_ .


**S2 Relation to Demographic Parity**


Consider the graph _A →_ _X →_ _Y_ . In general, if _Y_ [ˆ] is a function of _X_ only, then _Y_ [ˆ] need not obey
demographic parity, i.e.


_P_ ( _Y_ [ˆ] _| A_ = _a_ ) _̸_ = _P_ ( _Y_ [ˆ] _| A_ = _a_ _[′]_ ) _,_


6 Notice that ˆ _Y_ ( _U_ ) is itself a random variable if _U_ is, but the source of randomness, _U_, is the same across all
counterfactuals.


13


where, since _Y_ [ˆ] is a function of _X_, the probabilities are obtained by marginalizing over _P_ ( _X | A_ = _a_ )
and _P_ ( _X | A_ = _a_ _[′]_ ), respectively.


If we postulate a structural equation _X_ = _αA_ + _e_ _X_, then given _A_ and _X_ we can deduce _e_ _X_ . If _Y_ [ˆ] is
a function of _e_ _X_ only and, by assumption, _e_ _X_ is marginally independent of _A_, then _Y_ [ˆ] is marginally
independent of _A_ : this follows the interpretation given in the previous section, where we interpret _e_ _X_
as “known” despite being mathematically deduced from the observation ( _A_ = _a, X_ = _x_ ) . Therefore,
the assumptions imply that _Y_ [ˆ] will satisfy demographic parity, and that can be falsified. By way
of contrast, if _e_ _X_ is not uniquely identifiable from the structural equation and ( _A, X_ ), then the
distribution of _Y_ [ˆ] depends on the value of _A_ as we marginalize _e_ _X_, and demographic parity will not
follow. This leads to the following:


**Lemma 2.** _If all background variables_ _U_ _[′]_ _⊆_ _U_ _in the definition of_ _Y_ [ˆ] _are determined from_ _A_ _and_ _X_ _,_
_and all observable variables in the definition of_ _Y_ [ˆ] _are independent of_ _A_ _given_ _U_ _[′]_ _, then_ _Y_ [ˆ] _satisfies_
_demographic parity._


Thus, counterfactual fairness can be thought of as a counterfactual analog of demographic parity, as
present in the Red Car example further discussed in the next section.


**S3 Examples Revisited**


In Section 3.1, we discussed two examples. We reintroduce them here briefly, add a third example, and
explain some consequences of their causal structure to the design of counterfactually fair predictors.


**Scenario 1: The Red Car Revisited.** In that scenario, the structure _A →_ _X ←_ _U →_ _Y_ implies
that _Y_ [ˆ] should not use either _X_ or _A_ . On the other hand, it is acceptable to use _U_ . It is interesting to
realize, however, that since _U_ is related to _A_ and _X_, there will be some association between _Y_ and
_{A, X}_ as discussed in Section S1. In particular, if the structural equation for _X_ is linear, then _U_ is
a linear function of _A_ and _X_, and as such _Y_ [ˆ] will also be a function of both _A_ and _X_ . This is not
a problem, as it is still the case that the model implies that this is merely a functional dependence
that disappears by conditioning on a postulated latent attribute _U_ . Surprisingly, we must make _Y_ [ˆ] a
indirect function of _A_ if we want a counterfactually fair predictor, as shown in the following Lemma.


**Lemma 3.** _Consider a linear model with the structure in Figure 1(a). Fitting a linear predictor to_ _X_
only _is not counterfactually fair, while the same algorithm will produce a fair predictor using_ both _A_
_and X._


_Proof._ As in the definition, we will consider the population case, where the joint distribution is
known. Consider the case where the equations described by the model in Figure 1(a) are deterministic
and linear:


_X_ = _αA_ + _βU,_ _Y_ = _γU._


Denote the variance of _U_ as _v_ _U_, the variance of _A_ as _v_ _A_, and assume all coefficients are non-zero.
The predictor _Y_ [ˆ] ( _X_ ) defined by least-squares regression of _Y_ on _only_ _X_ is given by _Y_ [ˆ] ( _X_ ) _≡_ _λX_,
where _λ_ = _Cov_ ( _X, Y_ ) _/V ar_ ( _X_ )= _βγv_ _U_ _/_ ( _α_ [2] _v_ _A_ + _β_ [2] _v_ _U_ ) _̸_ = 0 . This predictor follows the concept
of fairness through unawareness.


We can test whether a predictor _Y_ [ˆ] is counterfactually fair by using the procedure described in
Section 2.2:


_(i)_ Compute _U_ given observations of _X, Y, A_ ; _(ii)_ Substitute the equations involving _A_ with an
interventional value _a_ _[′]_ ; _(iii)_ Compute the variables _X, Y_ with the interventional value _a_ _[′]_ . It is clear
here that _Y_ [ˆ] _a_ ( _U_ )= _λ_ ( _αa_ + _βU_ ) _̸_ = _Y_ [ˆ] _a_ _′_ ( _U_ ) . This predictor is not counterfactually fair. Thus, in this
case fairness through unawareness actually perpetuates unfairness.


Consider instead doing least-squares regression of _Y_ on _X_ _and_ _A_ . Note that _Y_ [ˆ] ( _X, A_ ) _≡_ _λ_ _X_ _X_ + _λ_ _A_ _A_
where _λ_ _X_ _, λ_ _A_ can be derived as follows:


14


_λ_ _X_
� _λ_ _A_


_̸_



� = � _CovV ar_ ( _X, A_ ( _X_ ) ) _CovV ar_ ( _A, X_ ( _A_ ) )


_̸_



_−_ 1
_Cov_ ( _X, Y_ )
� � _Cov_ ( _A, Y_ )


_̸_



�


_̸_



_βγv_ _U_
0
��


_̸_



�


_̸_



1

=
_β_ [2] _v_ _U_ _v_ _A_


_̸_



_v_ _A_ _−αv_ _A_
� _−αv_ _A_ _α_ [2] _v_ _A_ + _β_ [2] _v_ _U_


_̸_



_γ_
= _−βαγ_
� _β_


_̸_



(5)
�


_̸_



Now imagine we have observed _Y_ ˆ ( _X, a_ ) = _β_ _[γ]_ [(] _[αa]_ [ +] _[ βU]_ [) +] _[ −]_ _β_ _[α][γ]_ _[a]_ _A_ [ =] _[ γU]_ = _a_ [. Thus, if we substitute] . This implies that _X_ _[ a]_ = [ with a counterfactual] _αa_ + _βU_ and our predictor is _[ a]_ _[′]_ [ (the action]

step described in Section 2.2) the predictor _Y_ [ˆ] ( _X, A_ ) is unchanged. This is because our predictor is
constructed in such a way that any change in _X_ caused by a change in _A_ is cancelled out by the _λ_ _A_ .
Thus this predictor is counterfactually fair.


Note that if Figure 1(a) is the true model for the real world then _Y_ [ˆ] ( _X, A_ ) will also satisfy demographic
parity and equality of opportunity as _Y_ [ˆ] will be unaffected by _A_ .


The above lemma holds in a more general case for the structure given in Figure 1(a): any non-constant
estimator that depends only on _X_ is not counterfactually fair as changing _A_ always alters _X_ .


_̸_



_β_ _[γ]_ [(] _[αa]_ [ +] _[ βU]_ [) +] _[ −]_ _β_ _[α][γ]_


_̸_




_[α][γ]_

_β_ _[a]_ [ =] _[ γU]_ [. Thus, if we substitute] _[ a]_ [ with a counterfactual] _[ a]_ _[′]_ [ (the action]


_̸_



**Scenario 2: High Crime Regions Revisited.** The causal structure differs from the previous example by the extra edge _X →_ _Y_ . For illustration purposes, assume again that the model is linear. Unlike
the previous case, a predictor _Y_ [ˆ] trained using _X_ and _A_ is not counterfactually fair. The only change
from Scenario 1 is that now _Y_ depends on _X_ as follows: _Y_ = _γU_ + _θX_ . Now if we solve for _λ_ _X_ _, λ_ _A_
it can be shown that _Y_ [ˆ] ( _X, a_ )=( _γ −_ _[α]_ _βv_ [2] _[θv]_ _U_ _[A]_ [)] _[U]_ [ +] _[ αθa]_ [. As this predictor depends on the values of] _[ A]_

that are not explained by _U_, then _Y_ [ˆ] ( _X, a_ ) = _̸_ _Y_ [ˆ] ( _X, a_ _[′]_ ) and thus _Y_ [ˆ] ( _X, A_ ) is not counterfactually fair.


The following extra example complements the previous two examples.


**Scenario 3: University Success.** A university wants to know if students will be successful postgraduation _Y_ . They have information such as: grade point average (GPA), advanced placement
(AP) exams results, and other academic features _X_ . The university believes however, that an
individual’s gender _A_ may influence these features and their post-graduation success _Y_ due to social
discrimination. They also believe that independently, an individual’s latent talent _U_ causes _X_ and _Y_ .
The structure is similar to Figure 1(a), with the extra edge _Y_ ˆ ( _X, A_ ) counterfactually fair? In this case, the different between this and Scenario 1 is that _A →_ _Y_ . We can again ask, is the predictor _Y_ is
a function of _Y_ ˆ ( _X, a_ )=( _γ U −_ and _[α]_ _βv_ _[η][v]_ _U_ _[A]_ _A_ [)] _[U]_ as follows: [ +] _[ ηa]_ [. Again] _Y_ = [ ˆ] _[Y]_ [ (] _γU_ _[X, A]_ + _ηA_ [)] [ is a function of] . We can again solve for _[ A]_ [ not explained by] _λ_ _X_ _, λ_ _A_ _[ U]_ and show that [, so it cannot]

be counterfactually fair.


**S4 Analysis of Individual Pathways**


By way of an example, consider the following adaptation of the scenario concerning claims of
gender bias in UC Berkeley’s admission process in the 1970s, commonly used a textbook example
of Simpson’s Paradox. For each candidate student’s application, we have _A_ as a binary indicator
of whether the applicant is female, _X_ as the choice of course to apply for, and _Y_ a binary indicator
of whether the application was successful or not. Let us postulate the causal graph that includes
the edges _A →_ _X_ and _X →_ _Y_ only. We observe that _A_ and _Y_ are negatively associated, which
in first instance might suggest discrimination, as gender is commonly accepted here as a protected
attribute for college admission. However, in the postulated model it turns out that _A_ and _Y_ are
causally independent given _X_ . More specifically, women tend to choose more competitive courses
(those with higher rejection rate) than men when applying. Our judgment is that the higher rejection
among female than male applicants is acceptable, if the mechanism _A →_ _X_ is interpreted as a choice
which is under the control of the applicant. That is, free-will overrides whatever possible cultural
background conditions that led to this discrepancy. In the framework of counterfactual fairness, we


15


could claim that _A_ is not a protected attribute to begin with once we understand how the world
works, and that including _A_ in the predictor of success is irrelevant anyway once we include _X_ in the
classifier.


However, consider the situation where there is an edge _A →_ _Y_, interpreted purely as the effect of
discrimination after causally controlling for _X_ . While it is now reasonable to postulate _A_ to be a
protected attribute, we can still judge that _X_ is not an unfair outcome: there is no need to “deconvolve”
_A_ out of _X_ to obtain an estimate of the other causes _U_ _X_ in the _A →_ _X_ mechanism. This suggests
a simple modification of the definition of counterfactual fairness. First, given the causal graph _G_
assumed to encode the causal relationships in our system, define _P_ _G_ _A_ as the set of all directed paths
from _A_ to _Y_ in _G_ which are postulated to correspond to all unfair chains of events where _A_ causes _Y_ .
Let _X_ _P_ _G_ _[c]_ _A_ _[⊆]_ _[X]_ [ be the subset of covariates not present in any path in] _[ P]_ _[G]_ _[A]_ [. Also, for any vector] _[ x]_ [, let]
_x_ _s_ represent the corresponding subvector indexed by _S_ . The corresponding uppercase version _X_ _S_ is
used for random vectors.
**Definition 7** ((Path-dependent) counterfactual fairness) **.** _Predictor_ _Y_ [ˆ] _is_ **(path-dependent) counter-**
**factually fair** _with respect to path set P_ _G_ _A_ _if under any context X_ = _x and A_ = _a,_

_P_ ( _Y_ [ˆ] _A←a,X_ _PGc_ _A_ _←_ _x_ _PGc_ _A_ [(] _[U]_ [) =] _[ y][ |][ X]_ [ =] _[ x, A]_ [ =] _[ a]_ [) =]

_P_ ( _Y_ [ˆ] _A←a_ _′_ _,X_ _̸PGc_ _A_ _←_ _x_ _PGc_ _A_ [(] _[U]_ [) =] _[ y][ |][ X]_ [ =] _[ x, A]_ [ =] _[ a]_ [)] _[,]_ (6)


_for all y and for any value a_ _[′]_ _attainable by A._


This notion is related to _controlled direct effects_ [ 29 ], where we intervene on some paths from _A_ to
_Y_, but not others. Paths in _P_ _G_ _A_ are considered here to be the “direct” paths, and we condition on _X_
and _A_ similarly to the definition of probability of sufficiency (3). This definition is the same as the
original counterfactual fairness definition for the case where _P_ _G_ _[c]_ _A_ [=] _[ ∅]_ [. Its interpretation is analogous]
to the original, indicating that for any _X_ 0 _∈_ _X_ _P_ _G_ _[c]_ _A_ [we are allowed to propagate information from the]
factual assigment _A_ = _a_, along with what we learned about the background causes _U_ _X_ 0, in order to
reconstruct _X_ 0 . The contribution of _A_ is considered acceptable in this case and does not need to be
“deconvolved.” The implication is that any member of _X_ _̸P_ _G_ _[c]_ _A_ [can be included in the definition of] [ ˆ] _[Y]_ [ .]
In the example of college applications, we are allowed to use the choice of course _X_ even though _A_
is a confounder for _X_ and _Y_ . We are still not allowed to use _A_ directly, bypassing the background
variables.


As discussed by [ 27 ], there are some counterfactual manipulations usable in a causal definition of
fairness that can be performed by exploiting only independence constraints among the counterfactuals:
that is, without requiring the explicit description of structural equations or other models for latent
variables. A contrast between the two approaches is left for future work, although we stress that they
are in some sense complementary: we are motivated mostly by problems such as the one in Figure
1(d), where many of the mediators themselves are considered to be unfairly affected by the protected
attribute, and independence constraints among counterfactuals alone are less likely to be useful in
identifying constraints for the fitting of a fair predictor.


**S5 The Multifaceted Dynamics of Fairness**


One particularly interesting question was raised by one of the reviewers: what is the effect of
continuing discrimination after fair decisions are made? For instance, consider the case where banks
enforce a fair allocation of loans for business owners regardless of, say, gender. This does not mean
such businesses will thrive at a balanced rate if customers continue to avoid female owned business at
a disproportionate rate for unfair reasons. Is there anything useful that can be said about this issue
from a causal perspective?


The work here proposed regards only what we can influence by changing how machine learningaided decision making takes place at specific problems. It cannot change directly how society as a
whole carry on with their biases. Ironically, it may sound unfair to banks to enforce the allocation
of resources to businesses at a rate that does not correspond to the probability of their respective
success, even if the owners of the corresponding businesses are not to be blamed by that. One way of
conciliating the different perspectives is by modeling how a fair allocation of loans, even if it does
not come without a cost, can nevertheless increase the proportion of successful female businesses


16


Figure 3: A causal model for the stop and frisk dataset.


compared to the current baseline. This change can by itself have an indirect effect on the culture and
behavior of a society, leading to diminishing continuing discrimination by a feedback mechanism, as
in affirmative action. We believe that in the long run isolated acts of fairness are beneficial even if
we do not have direct control on all sources of unfairness in any specific problem. Causal modeling
can help on creating arguments about the long run impact of individual contributions as e.g. a type
of macroeconomic assessment. There are many challenges, and we should not pretend that precise
answers can be obtained, but in theory we should aim at educated quantitative assessments validating
how a systemic improvement in society can emerge from localized ways of addressing fairness.


**S6 Case Study: NYC Stop-and-Frisk Data**


Since 2002, the New York Police Department (NYPD) has recorded information about every time
a police officer has stopped someone. The officer records information such as if the person was
searched or frisked, if a weapon was found, their appearance, whether an arrest was made or a
summons issued, if force was used, etc. We consider the data collected on males stopped during
2014 which constitutes 38,609 records. We limit our analysis to looking at just males stopped as this
accounts for more than 90% of the data. We fit a model which postulates that police interactions is
caused by race and a single latent factor labeled _Criminality_ that is meant to index other aspects of
the individual that have been used by the police and which are independent of race. We do not claim
that this model has a solid theoretical basis, we use it below as an illustration on how to carry on an
analysis of counterfactually fair decisions. We also describe a spatial analysis of the estimated latent
factors.


**Model.** We model this stop-and-frisk data using the graph in Figure 3. Specifically, we posit main
causes for the observations: _Arrest_ (if an individual was arrested), _Force_ (some sort of force was
used during the stop), _Frisked_, and _Searched_ . The first cause of these observations is some measure
of an individual’s latent _Criminality_, which we do not observe. We believe that _Criminality_ also
directly affects _Weapon_ (an individual was found to be carrying a weapon). For all of the features
previously mentioned we believe there is an additional cause, an individual’s _Race_ which we do
observe. This factor is introduced as we believe that these observations may be biased based on an
officer’s perception of whether an individual is likely a criminal or not, affected by an individual’s
_Race_ . Thus note that, in this model, _Criminality_ is counterfactually fair for the prediction of any
characteristic of the individual for problems where _Race_ is a protected attribute.


**Visualization on a map of New York City.** Each of the stops can be mapped to longitude and
latitude points for where the stop occurred [7] . This allows us to visualize the distribution of two distinct
populations: the stops of White and Black Hispanic individuals, shown in Figure 4. We note that
there are more White individuals stopped ( 4492 ) than Black Hispanic individuals ( 2414 ). However,
if we look at the arrest distribution (visualized geographically in the second plot) the rate of arrest
for White individuals is lower ( 12 _._ 1% ) than for Black Hispanic individuals ( 19 _._ 8%, the highest rate
for any race in the dataset). Given our model we can ask: “If every individual had been White,


7 https://github.com/stablemarkets/StopAndFrisk


17


Figure 4: How race affects arrest. The above maps show how altering one’s race affects whether or
not they will be arrested, according to the model. The left-most plot shows the distribution of White
and Black Hispanic populations in the stop-and-frisk dataset. The second plot shows the true arrests
for all of the stops. Given our model we can compute whether or not every individual in the dataset
would be arrest _had they been white_ . We show this counterfactual in the third plot. Similarly, we can
compute this counterfactual if everyone had been Black Hispanic, as shown in the fourth plot.


would they have been arrested?”. The answer to this is in the third plot. We see that the overall
number of arrests decreases (from 5659 to 3722 ). What if every individual had been Black Hispanic?
The fourth plot shows an increase in the number of arrests had individuals been Black Hispanic,
according to the model (from 5659 to 6439 ). The yellow and purple circles show two regions where
the difference in counterfactual arrest rates is particularly striking. Thus, the model indicates that,
even when everything else in the model is held constant, race has a differential affect on arrest rate
under the (strong) assumptions of the model.


18


