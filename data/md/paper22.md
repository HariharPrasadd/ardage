## **Towards Poisoning of Deep Learning Algorithms with** **Back-gradient Optimization**



Luis Muñoz-González Battista Biggio
Imperial College London, UK DIEE, University of Cagliari, Italy
Pluribus One



Ambra Demontis

DIEE, University of Cagliari, Italy



Andrea Paudice Vasin Wongrassamee Emil C. Lupu
Imperial College London, UK Imperial College London, UK Imperial College London, UK


Fabio Roli

DIEE, University of Cagliari, Italy
Pluribus One



**ABSTRACT**


A number of online services nowadays rely upon machine learning
to extract valuable information from data collected in the wild.

This exposes learning algorithms to the threat of data poisoning,
i.e., a coordinate attack in which a fraction of the training data is
controlled by the attacker and manipulated to subvert the learning
process. To date, these attacks have been devised only against a
limited class of binary learning algorithms, due to the inherent
complexity of the gradient-based procedure used to optimize the
poisoning points (a.k.a. adversarial training examples). In this work,
we first extend the definition of poisoning attacks to multiclass
problems. We then propose a novel poisoning algorithm based on
the idea of back-gradient optimization, i.e., to compute the gradient
of interest through automatic differentiation, while also reversing
the learning procedure to drastically reduce the attack complexity.
Compared to current poisoning strategies, our approach is able to
target a wider class of learning algorithms, trained with gradientbased procedures, including neural networks and deep learning
architectures. We empirically evaluate its effectiveness on several
application examples, including spam filtering, malware detection,
and handwritten digit recognition. We finally show that, similarly
to adversarial test examples, adversarial training examples can also
be transferred across different learning algorithms.


**CCS CONCEPTS**


**Computing Methodologies** →Machine Learning;


**KEYWORDS**


Adversarial Machine Learning, Training Data Poisoning, Adversarial Examples, Deep Learning.


**1** **INTRODUCTION**


In recent years technology has become pervasive, enabling a rapid
a disruptive change in the way society is organized. Our data is
provided to third-party services which are supposed to facilitate
and protect our daily work and activities. Most of these services
leverage machine learning to extract valuable information from
the overwhelming amount of input data received. Although this
provides advantages to the users themselves, e.g., in terms of usability and functionality of such services, it is also clear that these



services may be abused, providing great opportunities for cybercriminals to conduct novel, illicit, and highly-profitable activities.
Being one of the main components behind such services makes
machine learning an appealing target for attackers, who may gain
a significant advantage by gaming the learning algorithm. Notably,
machine learning itself can be the _weakest link_ in the security chain,
as its vulnerabilities can be exploited by the attacker to compromise the whole system infrastructure. To this end, she may inject
malicious data to poison the learning process, or manipulate data
at test time to evade detection. [1] These kinds of attack have been

reported against anti-virus engines, anti-spam filters, and systems
aimed to detect fake profiles or news in social networks – all problems involving a well-crafted deployment of machine learning algorithms [ 8, 16, 17, 19, 23, 25, 32, 34, 37 – 39 ]. Such attacks have fuelled
a growing interest in the research area of _adversarial machine learn-_
_ing_, at the intersection of cybersecurity and machine learning. This
recent research field aims at understanding the security properties
of current learning algorithms, as well as at developing more secure
ones [7, 16, 17].
Among the different attack scenarios envisaged against machine
learning, _poisoning attacks_ are considered one of the most relevant
and emerging security threats for data-driven technologies, i.e.,
technologies relying upon the collection of large amounts of data
in the wild [ 17 ]. In a poisoning attack, the attacker is assumed to
control a fraction of the training data used by the learning algorithm,
with the goal of subverting the entire learning process, or facilitate
subsequent system evasion [ 8, 19, 23, 25, 32, 39 ]. More practically,
data poisoning is already a relevant threat in different application
domains. For instance, some online services directly exploit users’
feedback on their decisions to update the trained model. PDFRate [2]

is an online malware detection tool that analyzes the submitted
PDF files to reveal the presence of embedded malware [ 34 ]. After
classification, it allows the user to provide feedback on its decision,
i.e., to confirm or not the classification result. A malicious user
may thus provide wrong feedback to gradually poison the system
and compromise its performance over time. Notably, there is a
more general underlying problem related to the collection of large
data volumes with reliable labels. This is a well-known problem


1 We refer to the attacker here as feminine due to the common interpretation as “Eve”
or “Carol” in cryptography and security.
2 [http://pdfrate.com](http://pdfrate.com)



1


in malware detection, where malware samples are collected by
means of compromised machines with known vulnerabilities (i.e.,
honeypots), or via other online services, like VirusTotal, [3] in which
labelling errors are often reported.
Previous work has developed poisoning attacks against popular
learning algorithms like Support Vector Machines (SVMs), LASSO,
logistic and ridge regression, in different applications, like spam
and malware detection [ 8, 19, 20, 23, 25, 32, 39 ]. The main technical
difficulty in devising a poisoning attack is the computation of the
poisoning samples, also recently referred to as _adversarial training_
_examples_ [ 20 ]. This requires solving a bilevel optimization problem
in which the outer optimization amounts to maximizing the classification error on an untainted validation set, while the inner optimization corresponds to training the learning algorithm on the poisoned
data [ 23 ]. Since solving this problem with black-box optimization
is too computationally demanding, previous work has exploited
gradient-based optimization, along with the idea of _implicit differ-_
_entiation_ . The latter consists of replacing the inner optimization
problem with its stationarity (Karush-Kuhn-Tucker, KKT) conditions to derive an implicit equation for the gradient [ 8, 20, 23, 39 ].
This approach however can only be used against a limited class of
learning algorithms, excluding neural networks and deep learning
architectures, due to the inherent complexity of the procedure used
to compute the required gradient. Another limitation is that, to
date, previous work has only considered poisoning of two-class
learning algorithms.
In this work, we overcome these limitations by first extending
the threat model proposed in [ 1, 2, 7, 16 ] to account for multiclass
poisoning attacks (Sect. 2). We then exploit a recent technique
called _back-gradient optimization_, originally proposed for hyperparameter optimization [ 3, 14, 22, 31 ], to implement a much more
computationally-efficient poisoning attack. The underlying idea is
to compute the gradient of interest through reverse-mode (automatic) differentiation (i.e., _back-propagation_ ), while reversing the
underlying learning procedure to trace back the entire sequence
of parameter updates performed during learning, without storing
it. In fact, storing this sequence in memory would be infeasible for
learning algorithms that optimize a large set of parameters across
several iterations. Our poisoning algorithm only requires the learning algorithm to update its parameters during training in a _smooth_
manner (e.g., through gradient descent), to correctly trace these
changes backwards. Accordingly, compared to previously proposed
poisoning strategies, our approach is the first capable of targeting
a wider class of learning algorithms, trainable with gradient-based
procedures, like neural networks and deep learning architectures
(Sect. 3).
Another important contribution of this work is to show how
the performance of learning algorithms may be drastically compromised even by the presence of a small fraction of poisoning points
in the training data, in the context of real-world applications like
spam filtering, malware detection, and handwritten digit recognition (Sect. 4). We also investigate the _transferability_ property of
poisoning attacks, i.e., the extent to which attacks devised against
a specific learning algorithm are effective against different ones.
To our knowledge, this property has been investigated for evasion


3 [https://virustotal.com](https://virustotal.com)



attacks (a.k.a. adversarial test examples), i.e., attacks aimed to evade
a trained classifier at test time [ 6, 24, 27, 37 ], but never for poisoning
attacks. We conclude our work by discussing related work (Sect. 5),
the main limitations of our approach, and future research directions
(Sect. 6).


**2** **THREAT MODEL**


In this section, we summarize the framework originally proposed
in [ 1, 2, 16 ] and subsequently extended in [ 7 ], which enables one
to envision different attack scenarios against learning algorithms
(including deep learning ones), and to craft the corresponding attack
samples. Remarkably, these include attacks at training and at test
time, usually referred to as poisoning and evasion attacks [ 6 – 8, 16,
23, 39 ] or, more recently, as adversarial (training and test) examples
(when crafted against deep learning algorithms) [27, 28, 36].
The framework characterizes the attacker according to her goal,
knowledge of the targeted system, and capability of manipulating
the input data. Based on these assumptions, it allows one to define
an optimal attack strategy as an optimization problem whose solution amounts to the construction of the attack samples, i.e., of the
_adversarial examples_ .
In this work, we extend this framework, originally developed for
binary classification problems, to multiclass classification. While
this generalization holds for evasion attacks too, we only detail
here the main poisoning attack scenarios.


**Notation.** In a classification task, given the instance space X and
the label space Y, the learner aims to estimate the underlying (possibly noisy) latent function _f_ that maps X �→Y . Given a training
set D tr = { _x_ _i_, _y_ _i_ } _i_ _[n]_ =1 [with] _[ n]_ [ i.i.d. samples drawn from the under-]
lying probability distribution _p_ (X, Y), [4] we can estimate _f_ with a
parametric or non-parametric model M trained by minimizing an
objective function L(D, _w_ ) (normally, a tractable estimate of the
generalization error), w.r.t. its parameters and/or hyperparameters
_w_ . [5]


Thus, while L denotes the learner’s objective function (possibly
including regularization), we use _L_ (D, _w_ ) to denote only the _loss_
incurred when evaluating the learner parameterized by _w_ on the
samples in D.


**2.1** **Attacker’s Goal**


The goal of the attack is determined in terms of the desired **secu-**
**rity violation** and **attack specificity** . In multiclass classification,
misclassifying a sample does not have a unique meaning, as there
is more than one class different from the correct one. Accordingly,
we extend the current framework by introducing the concept of
**error specificity** . These three characteristics are detailed below.


**Security Violation.** This characteristic defines the high-level security violation caused by the attack, as normally done in security
engineering. It can be: an _integrity_ violation, if malicious activities
evade detection without compromising normal system operation;


4 While normally the set notation { _x_ _i_, _y_ _i_ } _in_ =1 [does not admit duplicate entries, we]
admit our data sets to contain potentially duplicated points.
5 For instance, for kernelized SVMs, _w_ may include the dual variables _α_, the bias _b_, and
even the regularization parameter _C_ . In this work, as in [ 8, 23, 39 ], we however consider
only the optimization of the model parameters, and not of its hyperparameters.


an _availability_ violation, if normal system functionality is compromised, e.g., by increasing the classification error; or a _privacy_
violation, if the attacker obtains private information about the system, its users or data by reverse-engineering the learning algorithm.

**Attack Specificity.** This characteristic ranges from _targeted_ to
_indiscriminate_, respectively, if the attack aims to cause misclassification of a specific set of samples (to target a given system user or
protected service), or of any sample (to target any system user or
protected service).

**Error Specificity.** We introduce here this characteristic to disambiguate the notion of misclassification in multiclass problems. The
error specificity can thus be: _specific_, if the attacker aims to have a
sample misclassified as a specific class; or _generic_, if the attacker
aims to have a sample misclassified as any of the classes different
from the true class. [6]


**2.2** **Attacker’s Knowledge**


The attacker can have different levels of knowledge of the targeted
system, including: ( _k_ . _i_ ) the training data D tr ; ( _k_ . _ii_ ) the feature set
X ; ( _k_ . _iii_ ) the learning algorithm M, along with the objective function L minimized during training; and, possibly, ( _k_ . _iv_ ) its (trained)
parameters _w_ . The attacker’s knowledge can thus be characterized
in terms of a space Θ that encodes the aforementioned assumptions
( _k_ . _i_ )-( _k_ . _iv_ ) as _θ_ = (D, X, M, _w_ ) . Depending on the assumptions
made on each of these components, one can envisage different attack scenarios. Typically, two main settings are considered, referred
to as attacks with _perfect_ and _limited_ knowledge.

**Perfect-Knowledge (PK) Attacks.** In this case, the attacker is
assumed to know everything about the targeted system. Although
this setting may be not always representative of practical cases, it
enables us to perform a worst-case evaluation of the security of
learning algorithms under attack, highlighting the upper bounds on
the performance degradation that may be incurred by the system
under attack. In this case, we have _θ_ PK = (D, X, M, _w_ ).

**Limited-Knowledge (LK) Attacks.** Although LK attacks admit a
wide range of possibilities, the attacker is typically assumed to know
the feature representation X and the learning algorithm M, but not
the training data (for which surrogate data from similar sources can
be collected). We refer to this case here as LK attacks with Surrogate
Data (LK-SD), and denote it with _θ_ LK−SD = (D [ˆ], X, M, ˆ _w_ ) (using
the _hat_ symbol to denote limited knowledge of a given component).
Notably, in this case, as the attacker is only given a surrogate data
set D [ˆ], also the learner’s parameters have to be estimated by the
attacker, e.g., by optimizing L on D [ˆ] .
Similarly, we refer to the case in which the attacker knows the
training data (e.g., if the learning algorithm is trained on publiclyavailable data), but not the learning algorithm (for which a surrogate learner can be trained on the available data) as LK attacks
with Surrogate Learners (LK-SL). This scenario can be denoted
with _θ_ LK−SL = (D, X, M [ˆ], ˆ _w_ ), even though the parameter vector
_w_ ˆ may belong to a different vector space than that of the targeted
learner. Note that LK-SL attacks also include the case in which the


6 In [ 28 ], the authors defined _targeted_ and _indiscriminate_ attacks (at test time) depending
on whether the attacker aims to cause _specific_ or _generic_ errors. Here we do not follow
their naming convention, as it can cause confusion with the interpretation of _targeted_
and _indiscriminate_ attacks introduced in previous work [1, 2, 4, 7, 9, 10, 16, 39].



attacker knows the learning algorithm, but she is not able to derive
an optimal attack strategy against it (e.g., if the corresponding optimization problem is not tractable or difficult to solve), and thus
uses a surrogate learning model to this end. Experiments on the
_transferability_ of attacks among learning algorithms, firstly demonstrated in [ 6 ] and then in subsequent work on deep learners [ 27 ],
fall under this category of attacks.


**2.3** **Attacker’s Capability**


This characteristic is defined based on the **influence** that the attacker has on the input data, and on the presence of **data manip-**
**ulation constraints** .

**Attack Influence.** In supervised learning, the attack influence can
be causative, if the attacker can influence both training and test
data, or exploratory, if the attacker can only manipulate test data.
These settings are more commonly referred to as _poisoning_ and
_evasion_ attacks [2, 6–8, 16, 23, 39].

**Data Manipulation Constraints.** Another aspect related to the
attacker’s capability is the presence of constraints on the manipulation of input data, which is however strongly dependent on the
given practical scenario. For example, if the attacker aims to evade a
malware classification system, she should manipulate the exploitation code embedded in the malware sample without compromising
its intrusive functionality. In the case of poisoning, the labels assigned to the training samples are not typically under the control
of the attacker. She should thus consider additional constraints

while manipulating the poisoning samples to have them labelled as
desired. Typically, these constraints can be nevertheless accounted
for in the definition of the optimal attack strategy. In particular, we
characterize them by assuming that an initial set of attack samples
D _c_ is given, and that it is modified according to a space of possible
modifications Φ(D _c_ ).


**2.4** **Attack Strategy**


Given the attacker’s knowledge _θ_ ∈ Θ and a set of manipulated attack samples D _c_ [′] ∈ Φ(D _c_ ), the attacker’s goal can be characterized
in terms of an objective function A(D _c_ [′], _θ_ ) ∈ R which evaluates
how effective the attacks D _c_ [′] are. The optimal attack strategy can
be thus given as:


D _c_ [⋆] [∈] [arg max] A(D _c_ [′] [,] _[θ]_ [)] (1)
D _c_ [′] ∈Φ(D _c_ )


While this high-level formulation encompasses both evasion and
poisoning attacks, in both binary and multiclass problems, in the
remainder of this work we only focus on the definition of some
poisoning attack scenarios.


**2.5** **Poisoning Attack Scenarios**


We focus here on two poisoning attack scenarios of interest for multiclass problems, noting that other attack scenarios can be derived
in a similar manner.

**Error-Generic Poisoning Attacks.** The most common scenario
considered in previous work [ 8, 23, 39 ] considers poisoning twoclass learning algorithms to cause a _denial of service_ . This is an
availability attack, and it could be targeted or indiscriminate, depending on whether it affects a specific system user or service, or


any of them. In the multiclass case, it is thus natural to extend this
scenario assuming that the attacker is not aiming to cause specific
errors, but only _generic_ misclassifications. As in [ 8, 23, 39 ], this
poisoning attack (as any other poisoning attack) requires solving
a bilevel optimization, where the inner problem is the learning
problem. This can be made explicit by rewriting Eq. (1) as:


D _c_ [⋆] [∈] [arg max] A(D _c_ [′] [,] _[θ]_ [)][ =] _[ L]_ [(][ ˆ][D] val [,][ ˆ] _[w]_ [)][,] (2)
D _c_ [′] ∈Φ(D _c_ )

s.t. _w_ ˆ ∈ arg min L(D [ˆ] tr ∪D _c_ [′] [,] _[w]_ [′] [)][,] (3)
_w_ [′] ∈W


where the surrogate data D [ˆ] available to the attacker is divided into
two disjoint sets D [ˆ] tr and D [ˆ] val . The former, along with the poisoning
points D _c_ [′] is used to learn the surrogate model, while the latter is
used to evaluate the impact of the poisoning samples on untainted
data, through the function A(D _c_ [′], _θ_ ) . In this case, the function
A(D _c_ [′], _θ_ ) is simply defined in terms of a loss function _L_ (D [ˆ] val, ˆ _w_ )
that evaluates the performance of the (poisoned) surrogate model
on D [ˆ] val . The dependency of A on D _c_ [′] is thus indirectly encoded
through the parameters ˆ _w_ of the (poisoned) surrogate model. [7] Note
that, since the learning algorithm (even if convex) may not exhibit
a unique solution in the feasible set W, the outer problem has
to be evaluated using the exact solution ˆ _w_ found by the inner
optimization. Worth remarking, this formulation encompasses all
previously-proposed poisoning attacks against binary learners [ 8,
23, 39 ], provided that the loss function _L_ is selected accordingly
(e.g., using the hinge loss against SVMs [ 8 ]). In the multiclass case,
one can use a multiclass loss function, like the log-loss with softmax
activation, as done in our experiments.

**Error-Specific Poisoning Attacks.** Here, we assume that the attacker’s goal is to cause specific misclassifications – a plausible
scenario only for multiclass problems. This attack can cause an
integrity or an availability violation, and it can also be targeted or
indiscriminate, depending on the desired misclassifications. The

                              poisoning problem remains that given by Eqs. (2) (3), though the
objective is defined as:

A(D _c_ [′] [,] _[θ]_ [)][ =][ −] _[L]_ [(][ ˆ][D] val [′] [,][ ˆ] _[w]_ [)][,] (4)

where D [ˆ] val [′] [is a set that contains the same data as] [ ˆ][D] [val] [, though with]
different labels, chosen by the attacker. These labels correspond to
the desired misclassifications, and this is why there is a minus sign
in front of _L_, i.e., the attacker effectively aims at _minimizing_ the loss
on her desired set of labels. Note that, to implement an integrity
violation or a targeted attack, some of these labels may actually be
the same as the true labels (such that normal system operation is
not compromised, or only specific system users are affected).


**3** **POISONING ATTACKS WITH**

**BACK-GRADIENT OPTIMIZATION**


In this section, we first discuss how the bilevel optimization given

      by Eqs. (2) (3) has been solved in previous work to develop gradientbased poisoning attacks [ 8, 20, 23, 39 ]. As we will see, these attacks
can only be used against a limited class of learning algorithms, excluding neural networks and deep learning architectures, due to the


7 Note that A can also be directly dependent on D _c_ ′ [, as in the case of nonparametric]
models; e.g., in kernelized SVMs, when the poisoning points are support vectors [8].



inherent complexity of the procedure used to compute the required
gradient. To overcome this limitation, we exploit a recent technique
called _back-gradient optimization_ [ 14, 22 ], which allows computing the gradient of interest in a more computationally-efficient
and stabler manner. Notably, this enables us to devise the first poisoning attack able to target neural networks and deep learning
architectures (without using any surrogate model).
Before delving into the technical details, we make the same
assumptions made in previous work [ 8, 23, 39 ] to reduce the complexity of Problem (2) - (3) : ( _i_ ) we consider the optimization of one
poisoning point at a time, denoted hereafter with _x_ _c_ ; and ( _ii_ ) we
assume that its label _y_ _c_ is initially chosen by the attacker, and kept
fixed during the optimization. The poisoning problem can be thus
simplified as:

_x_ _c_ [⋆] [∈] arg max A({ _x_ _c_ [′] [,] _[y]_ _[c]_ [}][,] _[θ]_ [)][ =] _[ L]_ [(][ ˆ][D] val [,][ ˆ] _[w]_ [)][,] (5)
_x_ [′] _c_ ∈Φ({ _x_ _c_, _y_ _c_ })

s.t. _w_ ˆ ∈ arg min L( _x_ _c_ [′] [,] _[w]_ [′] [)][ .] (6)
_w_ [′] ∈W


The function Φ imposes constraints on the manipulation of _x_ _c_, e.g.,
upper and lower bounds on its manipulated values. These may also
depend on _y_ _c_, e.g., to ensure that the poisoning sample is labelled
as desired when updating the targeted classifier. Note also that, for
notational simplicity, we only report _x_ _c_ [′] as the first argument of L
instead of D [ˆ] tr ∪{ _x_ _c_ [′], _y_ _c_ }.

**Gradient-based Poisoning Attacks.** We discuss here how Problem (5) - (6) has been solved in previous work [ 8, 20, 23, 39 ]. For
some classes of loss functions _L_ and learning objective functions
L, this problem can be indeed solved through _gradient ascent_ . In
particular, provided that the loss function _L_ is differentiable w.r.t. _w_
and _x_ _c_, we can compute the gradient ∇ _x_ _c_ A using the chain rule:



w.r.t. [∂] _[w]_ [ˆ]

∂ _x_ _c_ [, and substitute its expression in Eq. (7), yielding:]

∇ _x_ _c_ A = ∇ _x_ _c_ _L_ −(∇ _x_ _c_ ∇ _w_ L)(∇ _w_ [2] [L)] [−][1] [∇] _[w]_ _[L]_ [ .] (8)


This gradient is then iteratively used to update the poisoning point
through gradient ascent, as shown in Algorithm 1. [8] Recall that the


8 Note that Algorithm 1 can be exploited to optimize multiple poisoning points too. As
in [ 39 ], the idea is to perform several passes over the set of poisoning samples, using
Algorithm 1 to optimize each poisoning point at a time, while keeping the other points
fixed. Line searches can also be exploited to reduce complexity.



∇ _x_ _c_ A = ∇ _x_ _c_ _L_ + ∂ [∂] _x_ _[w]_ [ˆ] _c_



⊤
∇ _w_ _L_, (7)



where _L_ (D [ˆ] val, ˆ _w_ ) is evaluated on the parameters ˆ _w_ learned after
training (including the poisoning point). The main difficulty here is
computing ∂ [∂] _x_ _[w]_ [ˆ] _c_ [, i.e., understanding how the solution of the learning]

algorithm varies w.r.t. the poisoning point. Under some regularity
conditions, this can be done by replacing the inner learning problem
with its stationarity (KKT) conditions. For example, this holds if
the learning problem L is convex, which implies that all stationary
points are global minima [ 31 ]. In fact, poisoning attacks have been
developed so far only against learning algorithms with convex
objectives [ 8, 20, 23, 39 ]. The trick here is to replace the inner
optimization with the implicit function ∇ _w_ L(D tr ∪{ _x_ _c_, _y_ _c_ }, ˆ _w_ ) = 0,
corresponding to its KKT conditions. Then, assuming that it is
differentiable w.r.t. _x_ _c_, one yields the linear system ∇ _x_ _c_ ∇ _w_ L +
∂∂ _xw_ ˆ _c_ ⊤ ∇ _w_ 2 L = 0 . If ∇ _w_ 2 L is not singular, we can solve this system


**Algorithm 1** Poisoning Attack Algorithm


**Input:** D [ˆ] tr, D [ˆ] val, L, _L_, the initial poisoning point _x_ _c_ [(][0][)] [, its label] _[ y]_ _c_ [,]
the learning rate _η_, a small positive constant _ε_ .


1: _i_ ← 0 (iteration counter)


2: **repeat**

3: _w_ ˆ ∈ arg min _w_ ′ L( _x_ _c_ [(] _[i]_ [)] [,] _[w]_ [′] [)][ (train learning algorithm)]

4: _x_ _c_ [(] _[i]_ [+][1][)] ← Π Φ _x_ _c_ [(] _[i]_ [)] + _η_ ∇ _x_ _c_ A({ _x_ _c_ [(] _[i]_ [)] [,] _[y]_ _c_ [})]
� �

5: _i_ ← _i_ + 1

6: **until** A({ _x_ _c_ [(] _[i]_ [)] [,] _[y]_ _c_ [}) −A({] _[x]_ _c_ [(] _[i]_ [−][1][)], _y_ _c_ }) < _ε_

**Output:** the final poisoning point _x_ _c_ ← _x_ _c_ [(] _[i]_ [)]


projection operator Π Φ is used to map the current poisoning point
onto the feasible set Φ (cf. Eqs. 5-6).
This is the state-of-the-art approach used to implement current
poisoning attacks [ 8, 20, 23, 39 ]. The problem here is that computing
and inverting ∇ _w_ [2] L scales in time as O( _p_ [3] ) and in memory as O( _p_ [2] ),
being _p_ the cardinality of _w_ . Moreover, Eq. (8) requires solving one
linear system per parameter. These aspects make it prohibitive to
assess the effectiveness of poisoning attacks in a variety of practical
settings.
To mitigate these issues, as suggested in [ 13, 14, 20, 22 ], one can
apply conjugate gradient descent to solve a simpler linear system,
obtained by a trivial re-organization of the terms in the second
part of Eq. (8) . In particular, one can set (∇ _w_ [2] L) v = ∇ _w_ _L_, and
compute ∇ x c A = ∇ _x_ _c_ _L_ −∇ x c ∇ _w_ L v . The computation of the
matrices ∇ _x_ _c_ ∇ _w_ L and ∇ _w_ [2] L can also be avoided using Hessianvector products [30]:


1
(∇ _x_ _c_ ∇ _w_ L) z = lim �∇ _x_ _c_ L � _x_ ′ _c_ [,][ ˆ] _[w]_ [ +] _[ h]_ [z][�] [−∇] _[x]_ _c_ [L][ �] _[x]_ [ ′] _c_ [,][ ˆ] _[w]_ [��] [,]
_h_ →0 _h_


1
(∇ _w_ ∇ _w_ L) z = lim �∇ _w_ L � _x_ ′ _c_ [,][ ˆ] _[w]_ [ +] _[ h]_ [z][�] [−∇] _[w]_ [L][ �] _[x]_ [ ′] _c_ [,][ ˆ] _[w]_ [��] [.]
_h_ →0 _h_


Although this approach allows poisoning learning algorithms more
efficiently w.r.t. previous work [ 8, 23, 39 ], it still requires the inner
learning problem to be solved exactly. From a practical perspective,
this means that the KKT conditions have to be met with satisfying
numerical accuracy. However, as these problems are always solved
to a finite accuracy, it may happen that the gradient ∇ _x_ _c_ A is not
sufficiently precise, especially if convergence thresholds are too
loose [14, 22].
It is thus clear that such an approach can not be used, in practice,
to poison learning algorithms like neural networks and deep learning architectures, as it may not only be difficult to derive proper
stationarity conditions involving all parameters, but also as it may
be too computationally demanding to train such learning algorithms with sufficient precision to correctly compute the gradient
∇ _x_ _c_ A.


**Poisoning with Back-gradient Optimization.** In this work, we
overcome this limitation by exploiting _back-gradient optimization_ [ 14,
22 ]. This technique has been first exploited in the context of energybased models and hyperparameter optimization, to solve bilevel
optimization problems similar to the poisoning problem discussed
before. The underlying idea of this approach is to replace the inner
optimization with a set of iterations performed by the learning



**Algorithm 2** Gradient Descent


**Input:** initial parameters _w_ 0, learning rate _η_, D [ˆ] tr, L.


1: **for** _t_ = 0, . . ., _T_ − 1 **do**

2: g _t_ = ∇ _w_ L(D [ˆ] tr, _w_ _t_ )


3: _w_ _t_ +1 ← _w_ _t_ − _η_ g _t_
4: **end for**

**Output:** trained parameters _w_ _T_


**Algorithm 3** Back-gradient Descent


**Input:** trained parameters _w_ _T_, learning rate _η_, D [ˆ] tr, D [ˆ] val,
poisoning point _x_ _c_ [′], _y_ _c_, loss function _L_, learner’s objective L.
initialize _dx_ _c_ ← 0, _dw_ ←∇ _w_ _L_ (D [ˆ] val, _w_ _T_ )


1: **for** _t_ = _T_, . . ., 1 **do**

2: _dx_ _c_ ← _dx_ _c_ [′] − _η dw_ ∇ _x_ _c_ ∇ _w_ L( _x_ _c_ [′], _w_ _t_ )

3: _dw_ ← _dw_ − _η dw_ ∇ _w_ ∇ _w_ L( _x_ _c_ [′], _w_ _t_ )

4: g _t_ −1 = ∇ _w_ _t_ L( _x_ _c_ [′], _w_ _t_ )


5: _w_ _t_ −1 = _w_ _t_ + _α_ g _t_ −1
6: **end for**

**Output:** ∇ _x_ _c_ A = ∇ _x_ _c_ _L_ + _dx_ _c_


algorithm to update the parameters _w_, provided that such updates
are _smooth_, as in the case of gradient-based learning algorithms.
According to [ 14 ], this technique allows to compute the desired gradients in the outer problem using the parameters _w_ _T_ obtained from
an incomplete optimization of the inner problem (after _T_ iterations).
This represent a significant computational improvement compared
to traditional gradient-based approaches, since it only requires a
reduced number of training iterations for the learning algorithm.
This is especially important in large neural networks and deep
learning algorithms, where the computational cost per iteration
can be high. Then, assuming that the inner optimization runs for
_T_ iterations, the idea is to exploit reverse-mode differentiation, or
_back-propagation_, to compute the gradient of the outer objective.
However, using back-propagation in a naïve manner would not
work for this class of problems, as it requires storing the whole
set of parameter updates _w_ 1, . . ., _w_ _T_ performed during training,
along with the forward derivatives. These are indeed the elements
required to compute the gradient of the outer objective with a
_backward_ pass (we refer the reader to [ 22 ] for more details). This
process can be extremely memory-demanding if the learning algorithm runs for a large number of iterations _T_, and especially if the
number of parameters _w_ is large (as in deep networks). Therefore,
to avoid storing the whole training trajectory _w_ 1, . . ., _w_ _T_ and the
required forward derivatives, Domke [14] and Maclaurin et al . [22]
proposed to compute them directly during the backward pass, by
_reversing_ the steps followed by the learning algorithm to update
them. Computing _w_ _T_, . . ., _w_ 1 in reverse order w.r.t. the forward
step is clearly feasible only if the learning procedure can be exactly traced backwards. Nevertheless, this happens to be feasible
for a large variety of gradient-based procedures, including gradient
descent with fixed step size, and stochastic gradient descent with

momentum.


**Figure 1: Error-generic (top row) and error-specific (bottom row) poisoning attacks on a three-class synthetic dataset, against**
**a multiclass logistic classifier. In the error-specific case, the attacker aims to have red points misclassified as blue, while pre-**
**serving the labels of the other points. We report the decision regions on the clean (first column) and on the poisoned (second**
**column) data, in which we only add a poisoning point labelled as blue (highlighted with a blue circle). The validation loss**
_L_ (D [ˆ] val, ˆ _w_ ) **and** _L_ (D [ˆ] val [′] [,][ ˆ] _[w]_ [)] **[, respectively maximized in error-generic and minimized in error-specific attacks, is shown in colors,]**
**as a function of the attack point** _x_ _c_ **(third column), along with the corresponding back-gradients (shown as arrows), and the**
**path followed while optimizing** _x_ _c_ **. To show that the logistic loss used to estimate** _L_ **provides a good approximation of the true**
**error, we also report the validation error measured with the zero-one loss on the same data (fourth column).**



In this work, we leverage back-gradient descent to compute
∇ _x_ _c_ A (Algorithm 3) by reversing a standard gradient-descent procedure with fixed step size that runs for a truncated training of the
learning algorithm to _T_ iterations (Algorithm 2). Notably, lines 2-3
in Algorithm 3 can be efficiently computed with Hessian-vector
products, as discussed before. We exploit this algorithm to compute
the gradient ∇ _x_ _c_ A in line 4 of our poisoning attack algorithm
(Algorithm 1). In this case, line 3 of Algorithm 1 is replaced with
the incomplete optimization of the learning algorithm, truncated
to _T_ iterations. Note finally that, as in [ 14, 22 ], the time complexity
of our back-gradient descent is O( _T_ ) . This drastically reduces the
complexity of the computation of the outer gradient, making it feasible to evaluate the effectiveness of poisoning attacks also against
large neural networks and deep learning algorithms. Moreover, this
outer gradient can be accurately estimated from a truncated optimization of the inner problem with a reduced number of iterations.
This allows for a tractable computation of the poisoning points in
Algorithm 1, since training the learning algorithm at each iteration
can be prohibitive, especially for deep networks.
We conclude this section by noting that, in the case of errorspecific poisoning attacks (Sect. 2.5), the outer objective in Problem (5) - (6) is − _L_ (D [ˆ] _val_ [′] [,][ ˆ] _[w]_ [)] [. This can be regarded as a minimization]
problem, and it thus suffices to modify line 4 in Algorithm 1 to
update the poisoning point along the opposite direction. We clarify
this in Fig. 1, where we also discuss the different effect of errorgeneric and error-specific poisoning attacks in a multiclass setting.



**4** **EXPERIMENTAL ANALYSIS**


In this section, we first evaluate the effectiveness of the backgradient poisoning attacks described in Sect. 3 on spam and malware
detection tasks. In these cases, we also assess whether poisoning
samples can be _transferred_ across different learning algorithms.
We then investigate the impact of error-generic and error-specific
poisoning attacks in the well-known multiclass problem of handwritten digit recognition. In this case, we also report the first proofof-concept adversarial training examples computed by poisoning a
convolutional neural network in an _end-to-end_ manner (i.e., not just
using a surrogate model trained on the deep features, as in [20]).


**4.1** **Spam and Malware Detection**


We consider here two distinct datasets, respectively representing
a spam email classification problem ( Spambase ) and a malware
detection task ( Ransomware ). The Spambase data [ 11 ] consists of a
collection of 4, 601 emails, including 1, 813 spam emails. Each email
is encoded as a feature vector consisting of 54 binary features, each
denoting the presence or absence of a given word in the email. The
Ransomware data [ 33 ] consists of 530 ransomware samples and 549
benign applications. Ransomware is a very recent kind of malware
which encrypts the data on the infected machine, and requires the
victim to pay a ransom to obtain the decryption key. This dataset
has 400 binary features accounting for different sets of actions, API
invocations, and modifications in the file system and registry keys
during the execution of the software.


We consider the following leaning algorithms: ( _i_ ) Multi-Layer
Perceptrons (MLPs) with one hidden layer consisting of 10 neurons;
( _ii_ ) Logistic Regression (LR); and ( _iii_ ) Adaline (ADA). For MLPs, we
have used hyperbolic tangent activation functions for the neurons
in the hidden layer, and softmax activations in the output layer.
Moreover, for MLPs and LR, we use the cross-entropy (or log-loss)
as the loss function, while we use the mean squared error for ADA.
We assume here that the attacker aims to cause a denial of ser
vice, and thus runs a poisoning _availability_ attack whose goal is
simply to maximize the classification error. Accordingly, we run
Algorithm 1 injecting up to 20 poisoning points in the training data.
We initialize the poisoning points by cloning training points and
flipping their label. We set the number of iterations _T_ for obtaining
stable back-gradients to 200, 100, and 80, respectively for MLPs, LR
and ADA. We further consider two distinct settings: PK attacks,
in which the attacker is assumed to have full knowledge of the
attacked system (for a worst-case performance assessment); and
LK-SL attacks, in which she knows everything except for the learning algorithm, and thus uses a surrogate learner M [ˆ] . This scenario,
as discussed in Sect. 2.2, is useful to assess the _transferability_ property of the attack samples. To the best of our knowledge, this has
been demonstrated in [ 6, 27 ] for evasion attacks (i.e., adversarial _test_
examples) but never for poisoning attacks (i.e., adversarial _training_
examples). To this end, we optimize the poisoning samples using
alternatively MLPs, LR or ADA as the surrogate learner, and then
evaluate the impact of the corresponding attacks against the other
two algorithms.
The experimental results, shown in Figs. 2-3, are averaged on 10
independent random data splits. In each split, we use 100 samples
for training and 400 for validation, i.e., to respectively construct
D tr and D val . Recall indeed that in both PK and LK-SL settings,
the attacker has perfect knowledge of the training set used to learn
the true (attacked) model, i.e., D [ˆ] tr = D tr . The remaining samples
are used for testing, i.e., to assess the classification error under
poisoning. [9]

We can observe from Fig. 2 that PK poisoning attacks can significantly compromise the performance of all the considered classifiers.
In particular, on Spambase, they cause the classification error of
ADA and LR to increase up to 30% even if the attacker only controls
15% of the training data. Although the MLP is more resilient to
poisoning than these linear classifiers, its classification error also
increases significantly, up to 25%, which is not tolerable in several
practical settings. The results for PK attacks on Ransomware are
similar, although the MLP seems as vulnerable as ADA and LR in
this case.


**Transferability of Poisoning Samples.** Regarding LK-SL poisoning attacks, we can observe from Fig. 3 that the attack points generated using a linear classifier (either ADA or LR) as the surrogate
model have a very similar impact on the other linear classifier. In
contrast, the poisoning points crafted with these linear algorithms
have a lower impact against the MLP, although its performance is
still noticeably affected. When the MLP is used as the surrogate
model, instead, the performance degradation of the other algorithms is similar. However, the impact of these attacks is much


9 Note indeed that the validation error only provides a biased estimate of the true
classification error, as it is used by the attacker to optimize the poisoning points [8].



0.05

0 0.05 0.1 0.15

Fraction of Attack Points in Training Data


**Figure 2: Results for PK poisoning attacks.**


lower. To summarize, our results show that the attack points can
be effectively transferred across linear algorithms and also have
a noticeable impact on (nonlinear) neural networks. In contrast,
transferring poisoning samples from nonlinear to linear models
seems to be less effective.


**4.2** **Handwritten Digit Recognition**


We consider here the problem of handwritten digit recognition,
which involves 10 classes (each corresponding to a digit, from 0 to 9),
using the MNIST data [ 21 ]. Each digit image consists of 28 × 28 = 784
pixels, ranging from 0 to 255 (images are in grayscale). We divide
each pixel value by 255 and use it as a feature. We evaluate the effect
of error-generic and error-specific poisoning strategies against a
multiclass LR classifier using softmax activation and the log-loss as
the loss function.


**Error-generic attack.** In this case, the attacker aims to maximize
the classification error regardless of the resulting kinds of error,
as described in Sect. 2.5. This is thus an _availability_ attack, aimed
to cause a denial of service. We generate 10 independent random
splits using 1000 samples for training, 1000 for validation, and
8000 for testing. To compute the back-gradients ∇ _x_ _c_ A required
by our poisoning attack, we use _T_ = 60 iterations. We initialize
the poisoning points by cloning randomly-chosen training points
and changing their label at random In addition, we compare our
poisoning attack strategy here against a label-flip attack in which
the attack points are drawn from the validation set and their labels
are flipped at random. In both cases, we inject up to 60 attack points
into the training set.
The results are shown in Fig. 4 (top row). Note first that our
error-generic poisoning attack almost doubles the classification
error in the absence of poisoning, with less than 6% of poisoning
points. It is also much more effective than random label flips and,
as expected, it causes a similar increase of the classification error
over all classes (although some classes are easier to poison, like
digit 5). This is even more evident from the difference between the



**SPAMBASE**



0.35


0.3


0.25


0.2


0.15


0.1





0 0.05 0.1 0.15

Fraction of Attack Points in Training Data



**RANSOMWARE**
0.3


0.25


0.2


0.15


0.1






**MLP vs All**
0.35



**LR vs All**
0.35


0.3



0.3


0.25


0.2


0.15


0.1





0.25


0.2


0.15


0.1









**Adaline vs All**
0.35


0.3


0.25


0.2


0.15


0.1





0 0.05 0.1 0.15

Fraction of Attack Points in Training Data


**Adaline vs All**



0 0.05 0.1 0.15

Fraction of Attack Points in Training Data


**MLP vs All**



0 0.05 0.1 0.15

Fraction of Attack Points in Training Data


**LR vs All**



0.25


0.2


0.15


0.1


0.05









0.25


0.2


0.15


0.1


0.05









0.25


0.2


0.15


0.1


0.05





0 0.05 0.1 0.15

Fraction of Attack Points in Training Data



0 0.05 0.1 0.15

Fraction of Attack Points in Training Data



0 0.05 0.1 0.15

Fraction of Attack Points in Training Data



**Figure 3: Results for LK-SL poisoning attacks (transferability of poisoning samples) on Spambase (top row) and Ransomware**
**(bottom row).**



confusion matrix obtained under 6% poisoning and that obtained
in the absence of attack.


**Error-specific attack.** Here, we assume that the attacker aims to
misclassify 8s as 3s, while not having any preference regarding
the classification of the other digits. This can be thus regarded
as an _availability_ attack, targeted to cause the misclassification
of a specific set of samples. We generate 10 independent random
splits with 1000 training samples, 4000 samples for validation, and
5000 samples for testing. Recall that the goal of the attacker in this
scenario is described by Eq. (4) . In particular, she aims at minimizing
_L_ (D [ˆ] _val_ [′] [,][ ˆ] _[w]_ [)] [, where the samples in the validation set] [ ˆ][D] _val_ [′] [are re-]
labelled according to the attacker’s goal. Here, the validation set
thus only consists of digits of class 8 labelled as 3. We set _T_ = 60
to compute the back-gradients used in our poisoning attack, and
inject up to 40 poisoning points into the training set. We initialize
the poisoning points by cloning randomly-chosen samples from
the classes 3 and 8 in the training set, and flipping their label from
3 to 8, or vice-versa. We consider only these two classes here as
they are the only two actively involved in the attack.
The results are shown in Fig. 4 (bottom row). We can observe that
only the classification error rate for digit 8 is significantly affected,
as expected. In particular, it is clear from the difference of the
confusion matrix obtained under poisoning and the one obtained
in the absence of attack that most of the 8s are misclassified as 3s.
After adding less than 4% of poisoning points, in fact, the error rate
for digit 8 increases approximately from 20% to 50%. Note that, as a
side effect, the error rate of digit 3 also slightly increases, though
not to a significant extent.


**Poisoning Deep Neural Networks.** We finally report a proofof-concept experiment to show the applicability of our attack algorithm to poison a deep network in an _end-to-end_ manner, i.e.,
accounting for all weight updates in each layer (instead of using



a surrogate model trained on a frozen deep feature representation [ 20 ]). To this end, we consider the convolutional neural network (CNN) proposed in [21] for classification of the MNIST digit
data, which requires optimizing more than 450, 000 parameters. [10]

In this proof-of-concept attack, we inject 10 poisoning points into
the training data, and repeat the experiment on 5 independent data
splits, considering 1, 000 samples for training, and 2, 000 for validation and testing. For simplicity, we only consider the classes of
digits 1, 5, and 6 in this case. We use Algorithm 1 to craft each single
poisoning point, but, similarly to [ 39 ], we optimize them iteratively,
making 2 passes over the whole set of poisoning samples. We also
use the line search exploited in [ 39 ], instead of a fixed gradient step
size, to reduce the attack complexity (i.e., the number of training
updates to the deep network). Under this setting, however, we find
that our attack points only slightly increase the classification error,
though not significantly, while random label flips do not have any
substantial effect. For comparison, we also attack a multiclass LR
classifier under the same setting, yielding an increase of the error
rate from 2% to 4 . 3% with poisoning attacks, and to only 2 . 1% with
random label flips. This shows that, at least in this simple case, deep
networks seem to be more resilient against (a very small fraction of)
poisoning attacks (i.e., less than 1%). Some of the poisoning samples
crafted against the CNN and the LR are shown in Figs. 5 and 6. We
report the initial digit (and its true label _y_ ), its poisoned version
(and its label _y_ _c_ ), and the difference between the two images, in
absolute value (rescaled to visually appreciate the modified pixels).
Notably, similarly to adversarial test examples, also poisoning samples against deep networks are visually indistinguishable from the
initial image (as in [20]), while this is not the case when targeting
the LR classifier. This might be due to the specific shape of the
decision function learned by the deep network in the input space,
as explained in the case of adversarial test examples [ 15, 36 ]. We


10 [We use the implementation available at https://github.com/tflearn/tflearn/blob/](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py)
[master/examples/images/convnet_mnist.py.](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py)


0.22


0.2


0.18







1

0.1

2

4

0

5

7

-0.1

8



0.15


0.1


0.05





0



0.16


0.14


0.12



-0.05


-0.1



0 0.01 0.02 0.03 0.04 0.05 0.06

Fraction of Attack Points in Training Data



0.3


0.2


0.1


0

0 0.01 0.02 0.03 0.04 0.05 0.06

Fraction of Attack Points in Training Data



0 1 2 3 4 5 6 7 8 9

Predicted Class



1


0.9


0.8


0.7


0.6



0.5





0

2

4

0

5

7

9
-0.3



0.3


0.2


0.1



0.4


0.3


0.2



0





-0.1


-0.2



0 0.01 0.02 0.03 0.04

Fraction of Attack Points in Training Data



0.1


0

0 0.01 0.02 0.03 0.04

Fraction of Attack Points in Training Data



0 1 2 3 4 5 6 7 8 9

Predicted Class



**Figure 4: Error-generic (top row) and error-specific (bottom row) poisoning against multiclass LR on the MNIST data. In the first**
**column, we report the test error (which, for error-specific poisoning attacks, is computed using the attacker’s labels instead**
**of the true labels, and so it decreases while approaching the attacker’s goal). In the second column, we report the error per**
**class, i.e., the probability of misclassifying a digit given that it belongs to the class reported in the legend. In the third column,**
**we report the difference between the confusion matrix obtained under poisoning (after injecting the maximum number of**
**poisoning samples) and that obtained in the absence of attack, to highlight how the errors affect each class.**



however leave a more detailed investigation of this aspect to future
work, along with a more systematic security evaluation of deep
networks against poisoning attacks. We conclude this section with
a simple _transferability_ experiment, in which we use the poisoning
samples crafted against the LR classifier to attack the CNN, and
vice-versa. In the former case, the attack is totally ineffective, while
in the latter case it has a similar effect to that of random label
flips (as the minimal modifications to the CNN-poisoning digits are
clearly irrelevant for the LR classifier).


**5** **RELATED WORK**


Seminal work on the analysis of supervised learning in the presence of _omniscient_ attackers that can compromise the training data
has been presented in [ 12, 18 ]. While their results show the infeasibility of learning in such settings, their analysis reports an
overly-pessimistic perspective on the problem. The first practical
poisoning attacks against two-class classification algorithms have
been proposed in [ 19, 26 ], in the context of spam filtering and anomaly detection. However, such attacks do not easily generalize to
different learning algorithms. More systematic attacks, based on
the exploitation of KKT conditions to solve the bilevel problem
corresponding to poisoning attacks have been subsequently proposed in [ 8, 20, 23, 39 ]. In particular, Biggio et al . [8] have been the
first to demonstrate the vulnerability of SVMs to poisoning attacks.
Following the same approach, Xiao et al . [39] have shown how to
poison LASSO, ridge regression, and the elastic net. Finally, Mei and
Zhu [23] has systematized such attacks under a unified framework
to poison convex learning algorithms with Tikhonov regularizers,


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|0<br>20|


|Col1|Poisoning sample (y=1)|Col3|
|---|---|---|
|0<br>|||
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|0<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|0<br>20|



**Figure 5: Poisoning samples targeting the CNN.**


based on the concept of machine teaching [ 29, 40 ]. The fact that
these techniques require full re-training of the learning algorithm
at each iteration (to fulfil the KKT conditions up to a sufficient finite
precision), along with the intrinsic complexity required to compute


















|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|Col1|Initial sample (y=5)|Col3|
|---|---|---|
|0<br>|||
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|Col1|Poisoning sample (y=6)|Col3|
|---|---|---|
|0<br>|||
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|Col1|Attack changes|Col3|
|---|---|---|
|0<br>|||
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|Col1|Initial sample (y=6)|Col3|
|---|---|---|
|0<br>|||
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|


|0|Col2|Col3|
|---|---|---|
|5<br>0<br>|5<br>0<br>|5<br>0<br>|
|5<br>0<br>5|5<br>0<br>5|5<br>0<br>5|
|5<br>0<br>5|5<br>0<br>5|10<br>20|



**Figure 6: Poisoning samples targeting the LR.**


the corresponding gradients, makes them too computationally demanding for several practical settings. Furthermore, this limits their
applicability to a wider class of learning algorithms, including those
based on gradient descent and subsequent variants, like deep neural
networks, as their optimization is often truncated prior to meeting
the stationarity conditions with the precision required to compute
the poisoning gradients effectively. Note also that, despite recent
work [ 20 ] has provided a first proof of concept of the existence
of _adversarial training examples_ against deep networks, this has
been shown on a binary classification task using a surrogate model
(attacked with standard KKT-based poisoning). In particular, the
authors have generated the poisoning samples by attacking a logistic classifier trained on the features extracted from the penultimate
layer of the network (which have been kept fixed). Accordingly, to
our knowledge, our work is thus the first to show how to poison a
deep neural network in an _end-to-end_ manner, considering all its
parameters and layers, and without using any surrogate model. Notably, our work is also the first to show (in a more systematic way)
that poisoning samples can be _transferred_ across different learning
algorithms, using _substitute_ (a.k.a. _surrogate_ ) models, as similarly
demonstrated for evasion attacks (i.e., adversarial test examples)
in [ 6, 37 ] against SVMs and NNs, and subsequently in [ 27 ] against
deep networks.


**6** **CONCLUSIONS, LIMITATIONS AND**

**FUTURE WORK**


Advances in machine learning have led to a massive use of datadriven technologies with emerging applications in many different fields, including cybersecurity, self-driving cars, data analytics,
biometrics and industrial control systems. At the same time, the
variability and sophistication of cyberattacks have tremendously
increased, making machine learning systems an appealing target
for cybercriminals [2, 16].



In this work, we have considered the threat of training data
poisoning, i.e., an attack in which the training data is purposely
manipulated to maximally degrade the classification performance
of learning algorithms. While previous work has shown the effectiveness of such attacks against binary learners [ 8, 20, 23, 39 ], in
this work we have been the first to consider poisoning attacks in
multiclass classification settings. To this end, we have extended the
commonly-used threat model proposed in [ 1, 2, 16 ] by introducing
the concept of _error specificity_, to denote whether the attacker aims
to cause specific misclassification errors (i.e., misclassifying samples as a specific class), or generic ones (i.e., misclassifying samples
as any class different than the correct one).
Another important contribution of this work has been to overcome the limitations of state-of-the-art poisoning attacks, which
require exploiting the stationarity (KKT) conditions of the attacked
learning algorithms to optimize the poisoning samples [ 8, 20, 23, 39 ].
As discussed throughout this work, this requirement, as well as
the intrinsic complexity of such attacks, limits their application
only to a reduced class of learning algorithms. In this work, we
have overcome these limitations by proposing a novel poisoning
algorithm based on back-gradient optimization [ 14, 22, 31 ]. Our
approach can be applied to a wider class of learning algorithms,
as it only requires the learning algorithm to update smoothly its
parameters during training, without even necessarily fulfilling the
optimality conditions with very high precision. Moreover, the gradients can be accurately estimated with the parameters obtained
from an incomplete optimization of the learning algorithm truncated to a reduced number of iterations. This enables the efficient
application of our attack strategy to large neural networks and
deep learning architectures, as well as any other learning algorithm
trained through gradient-based procedures. Our empirical evaluation on spam filtering, malware detection, and handwritten digit
recognition has shown that neural networks can be significantly
compromised even if the attacker only controls a small fraction of
training points. We have also empirically shown that poisoning
samples designed against one learning algorithm can be rather
effective also in poisoning another algorithm, highlighting an interesting _transferability_ property, as that shown for evasion attacks
(a.k.a. adversarial test examples) [6, 27, 37].
The main limitation of this work is that we have not run an

extensive evaluation of poisoning attacks against deep networks,
to thoroughly assess their security to poisoning. Although our
preliminary experiments seem to show that they can be more resilient against this threat than other learning algorithms, a more
complete and systematic analysis remains to be performed. Therefore, we plan to more systematically investigate the effectiveness
of our back-gradient poisoning attack against deep networks in
the very near future. Besides the extension and evaluation of this
poisoning attack strategy to different deep learning architectures
and nonparametric models, further research avenues include: the
investigation of the existence of _universal perturbations_ (not dependent on the initial attack point) for poisoning samples against deep
networks, similarly to the case of universal adversarial test examples [ 15, 24 ]; and the evaluation of defense mechanisms against
poisoning attacks, through the exploitation of data sanitization and
robust learning algorithms [5, 32, 35].


**REFERENCES**


[1] Marco Barreno, Blaine Nelson, Anthony Joseph, and J. Tygar. 2010. The security
of machine learning. _Machine Learning_ 81 (2010), 121–148. Issue 2.

[2] Marco Barreno, Blaine Nelson, Russell Sears, Anthony D. Joseph, and J. D. Tygar. 2006. Can machine learning be secure?. In _Proc. ACM Symp. Information,_
_Computer and Comm. Sec. (ASIACCS ’06)_ . ACM, New York, NY, USA, 16–25.

[3] Y. Bengio. 2000. Gradient-based optimization of hyperparameters. _Neural Com-_
_putation_ 12, 8 (2000), 1889–1900.

[4] Battista Biggio, Samuel Rota Bulò, Ignazio Pillai, Michele Mura, Eyasu Zemene
Mequanint, Marcello Pelillo, and Fabio Roli. 2014. Poisoning complete-linkage
hierarchical clustering. In _Joint IAPR Int’l Workshop on Structural, Syntactic,_
_and Statistical Pattern Recognition (Lecture Notes in Computer Science)_, P. Franti,
G. Brown, M. Loog, F. Escolano, and M. Pelillo (Eds.), Vol. 8621. Springer Berlin
Heidelberg, Joensuu, Finland, 42–52.

[5] Battista Biggio, Igino Corona, Giorgio Fumera, Giorgio Giacinto, and Fabio
Roli. 2011. Bagging Classifiers for Fighting Poisoning Attacks in Adversarial
Classification Tasks. In _10th International Workshop on Multiple Classifier Systems_
_(MCS) (Lecture Notes in Computer Science)_, Carlo Sansone, Josef Kittler, and Fabio
Roli (Eds.), Vol. 6713. Springer-Verlag, 350–359.

[6] B. Biggio, I. Corona, D. Maiorca, B. Nelson, N. Šrndić, P. Laskov, G. Giacinto, and
F. Roli. 2013. Evasion attacks against machine learning at test time. In _Machine_
_Learning and Knowledge Discovery in Databases (ECML PKDD), Part III (LNCS)_,
Hendrik Blockeel, Kristian Kersting, Siegfried Nijssen, and Filip Železný (Eds.),
Vol. 8190. Springer Berlin Heidelberg, 387–402.

[7] Battista Biggio, Giorgio Fumera, and Fabio Roli. 2014. Security Evaluation of
Pattern Classifiers Under Attack. _IEEE Transactions on Knowledge and Data_
_Engineering_ 26, 4 (April 2014), 984–996.

[8] Battista Biggio, Blaine Nelson, and Pavel Laskov. 2012. Poisoning attacks against
support vector machines, In 29th Int’l Conf. on Machine Learning, John Langford
and Joelle Pineau (Eds.). _Int’l Conf. on Machine Learning (ICML)_, 1807–1814.

[9] Battista Biggio, Ignazio Pillai, Samuel Rota Bulò, Davide Ariu, Marcello Pelillo,
and Fabio Roli. 2013. Is Data Clustering in Adversarial Settings Secure?. In
_Proceedings of the 2013 ACM Workshop on Artificial Intelligence and Security_
_(AISec ’13)_ . ACM, New York, NY, USA, 87–98.

[10] Battista Biggio, Konrad Rieck, Davide Ariu, Christian Wressnegger, Igino Corona,
Giorgio Giacinto, and Fabio Roli. 2014. Poisoning Behavioral Malware Clustering.
In _2014 Workshop on Artificial Intelligent and Security (AISec ’14)_ . ACM, New
York, NY, USA, 27–36.

[11] C. Blake and C.J. Merz. 1998. UCI Repository of machine learning databases.
_http://www. ics. uci. edu/˜ mlearn/MLRepository. html_ (1998).

[12] NaderH. Bshouty, Nadav Eiron, and Eyal Kushilevitz. 1999. PAC Learning with
Nasty Noise. In _Algorithmic Learning Theory_, Osamu Watanabe and Takashi
Yokomori (Eds.). Lecture Notes in Computer Science, Vol. 1720. Springer Berlin
[Heidelberg, 206–218. https://doi.org/10.1007/3-540-46769-6_17](https://doi.org/10.1007/3-540-46769-6_17)

[13] C. Do, C.S. Foo, and A.Y. Ng. 2008. Efficient multiple hyperparameter learning
for log-linear models. In _Advances in Neural Information Processing Systems_ .
377–384.

[14] Justin Domke. 2012. Generic Methods for Optimization-Based Modeling. In _15th_
_Int’l Conf. Artificial Intelligence and Statistics (Proceedings of Machine Learning_
_Research)_, Neil D. Lawrence and Mark Girolami (Eds.), Vol. 22. PMLR, La Palma,
Canary Islands, 318–326.

[15] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining
and Harnessing Adversarial Examples. In _International Conference on Learning_
_Representations_ .

[16] L. Huang, A. D. Joseph, B. Nelson, B. Rubinstein, and J. D. Tygar. 2011. Adversarial
Machine Learning. In _4th ACM Workshop on Artificial Intelligence and Security_
_(AISec 2011)_ . Chicago, IL, USA, 43–57.

[17] Anthony D. Joseph, Pavel Laskov, Fabio Roli, J. Doug Tygar, and Blaine Nelson.
2013. Machine Learning Methods for Computer Security (Dagstuhl Perspectives
Workshop 12371). _Dagstuhl Manifestos_ 3, 1 (2013), 1–30.

[18] Michael Kearns and Ming Li. 1993. Learning in the presence of malicious errors.
_SIAM J. Comput._ [22, 4 (1993), 807–837. https://doi.org/10.1137/0222052](https://doi.org/10.1137/0222052)

[19] Marius Kloft and Pavel Laskov. 2012. Security Analysis of Online Centroid
Anomaly Detection. _Journal of Machine Learning Research_ 13 (2012), 3647–3690.

[20] P. W. Koh and P. Liang. 2017. Understanding Black-box Predictions via Influence
Functions. In _International Conference on Machine Learning (ICML)_ .

[21] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. GradientBased Learning Applied to Document Recognition. In _Proceedings of the IEEE_,
Vol. 86. 2278–2324.

[22] Dougal Maclaurin, David Duvenaud, and Ryan P. Adams. 2015. Gradient-based
Hyperparameter Optimization Through Reversible Learning. In _Proceedings of the_
_32Nd International Conference on International Conference on Machine Learning -_
_Volume 37 (ICML’15)_ . JMLR.org, 2113–2122.

[23] Shike Mei and Xiaojin Zhu. 2015. Using Machine Teaching to Identify Optimal Training-Set Attacks on Machine Learners. In _29th AAAI Conf. Artificial_
_Intelligence (AAAI ’15)_ .




[24] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal
Frossard. 2017. Universal adversarial perturbations. In _CVPR_ .

[25] B. Nelson, M. Barreno, F.J. Chi, A.D. Joseph, B.I.P. Rubinstein, U. Saini, C.A.
Sutton, J.D. Tygar, and K. Xia. 2008. Exploiting Machine Learning to Subvert
your Spam Filter. _LEET_ 8 (2008), 1–9.

[26] Blaine Nelson, Marco Barreno, Fuching Jack Chi, Anthony D. Joseph, Benjamin
I. P. Rubinstein, Udam Saini, Charles Sutton, J. D. Tygar, and Kai Xia. 2008.
Exploiting machine learning to subvert your spam filter. In _LEET’08: Proceedings_
_of the 1st Usenix Workshop on Large-Scale Exploits and Emergent Threats_ . USENIX
Association, Berkeley, CA, USA, 1–9.

[27] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik,
and Ananthram Swami. 2017. Practical Black-Box Attacks Against Machine
Learning. In _Proceedings of the 2017 ACM on Asia Conference on Computer and_
_Communications Security (ASIA CCS ’17)_ . ACM, New York, NY, USA, 506–519.

[28] Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay
Celik, and Ananthram Swami. 2016. The Limitations of Deep Learning in Adversarial Settings. In _Proc. 1st IEEE European Symposium on Security and Privacy_ .
IEEE, 372–387.

[29] K.R. Patil, X. Zhu, L. Kopeć, and B.C. Love. 2014. Optimal teaching for limitedcapacity human learners. In _Advances in Neural Information Processing Systems_ .
2465–2473.

[30] B.A. Pearlmutter. 1994. Fast Exact Multiplication by the Hessian. _Neural Compu-_
_tation_ 6, 1 (1994), 147–160.

[31] F. Pedregosa. 2016. Hyperparameter optimization with approximate gradient.
In _33rd International Conference on Machine Learning (Proceedings of Machine_
_Learning Research)_, Maria Florina Balcan and Kilian Q. Weinberger (Eds.), Vol. 48.
PMLR, New York, New York, USA, 737–746.

[32] Benjamin I.P. Rubinstein, Blaine Nelson, Ling Huang, Anthony D. Joseph, Shinghon Lau, Satish Rao, Nina Taft, and J. D. Tygar. 2009. ANTIDOTE: understanding
and defending against poisoning of anomaly detectors. In _Proceedings of the 9th_
_ACM SIGCOMM Internet Measurement Conference (IMC ’09)_ . ACM, New York,
NY, USA, 1–14.

[33] D. Sgandurra, L. Muñoz-González, R. Mohsen, and E.C. Lupu. 2016. Automated
Dynamic Analysis of Ransomware: Benefits, Limitations and use for Detection.
_arXiv preprint arXiv:1609.03020_ (2016).

[34] Charles Smutz and Angelos Stavrou. 2012. Malicious PDF Detection Using
Metadata and Structural Features. In _Proceedings of the 28th Annual Computer_
_Security Applications Conference (ACSAC ’12)_ . ACM, New York, NY, USA, 239–
248.

[35] J. Steinhardt, P. W. Koh, and P. Liang. 2017. Certified Defenses for Data Poisoning
Attacks. _arXiv preprint arXiv:1706.03691_ (2017).

[36] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru
Erhan, Ian Goodfellow, and Rob Fergus. 2014. Intriguing properties of neural
networks. In _International Conference on Learning Representations_ [. http://arxiv.](http://arxiv.org/abs/1312.6199)
[org/abs/1312.6199](http://arxiv.org/abs/1312.6199)

[37] Nedim Šrndic and Pavel Laskov. 2014. Practical Evasion of a Learning-Based
Classifier: A Case Study. In _Proc. 2014 IEEE Symp. Security and Privacy (SP ’14)_ .
IEEE CS, Washington, DC, USA, 197–211.

[38] Gang Wang, Tianyi Wang, Haitao Zheng, and Ben Y. Zhao. 2014. Man vs. Machine:
Practical Adversarial Detection of Malicious Crowdsourcing Workers. In _23rd_
_USENIX Security Symposium (USENIX Security 14)_ . USENIX Association, San
Diego, CA.

[39] Huang Xiao, Battista Biggio, Gavin Brown, Giorgio Fumera, Claudia Eckert, and
Fabio Roli. 2015. Is Feature Selection Secure against Training Data Poisoning?.
In _JMLR W&CP - Proc. 32nd Int’l Conf. Mach. Learning (ICML)_, Francis Bach and
David Blei (Eds.), Vol. 37. 1689–1698.

[40] X. Zhu. 2013. Machine Teaching for Bayesian Learners in the Exponential Family.
In _Advances in Neural Information Processing Systems_ . 1905–1913.


