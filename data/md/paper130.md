## **Shapley Flow:** **A Graph-based Approach to Interpreting Model Predictions**

**Jiaxuan Wang** **Jenna Wiens** **Scott Lundberg**
University of Michigan University of Michigan Microsoft Research
```
 jiaxuan@umich.edu wiensj@umich.edu scott.lundberg@microsoft.com

```


**Abstract**


Many existing approaches for estimating feature importance are problematic because
they ignore or hide dependencies among features. A causal graph, which encodes the relationships among input variables, can aid in
assigning feature importance. However, current approaches that assign credit to nodes
in the causal graph fail to explain the entire
graph. In light of these limitations, we propose Shapley Flow, a novel approach to interpreting machine learning models. It considers the entire causal graph, and assigns
credit to _edges_ instead of treating nodes as
the fundamental unit of credit assignment.
Shapley Flow is the unique solution to a generalization of the Shapley value axioms for
directed acyclic graphs. We demonstrate the
benefit of using Shapley Flow to reason about
the impact of a model’s input on its output.
In addition to maintaining insights from existing approaches, Shapley Flow extends the
flat, set-based, view prevalent in game theory based explanation methods to a deeper,
_graph-based_, view. This graph-based view enables users to understand the flow of importance through a system, and reason about
potential interventions.


**1** **Introduction**


Explaining a model’s predictions by assigning importance to its inputs ( _i.e._, feature attribution) is critical to many applications in which a user interacts
with a model to either make decisions or gain a bet

Proceedings of the 24 [th] International Conference on Artificial Intelligence and Statistics (AISTATS) 2021, San Diego,
California, USA. PMLR: Volume 130. Copyright 2021 by
the author(s).



Figure 1: Causal graph for the sprinkler example from
Chapter 1.2 of Pearl (2009). The model, _𝑓_, can be expanded into its own graph. To simplify the exposition,
although _𝑓_ takes 4 variables as input, we arbitrarily assumed that it only depends on _𝑋_ 3 and _𝑋_ 4 directly ( _i.e._,
_𝑓_ ( _𝑋_ 1 _, 𝑋_ 2 _, 𝑋_ 3 _, 𝑋_ 4 ) = _𝑔_ ( _𝑋_ 3 _, 𝑋_ 4 ) for some _𝑔_ ).


ter understanding of a system (Simonyan et al., 2013;
Lundberg and Lee, 2017; Zhou et al., 2016; Shrikumar
et al., 2017; Baehrens et al., 2010; Binder et al., 2016;
Springenberg et al., 2014; Sundararajan et al., 2017;
Fisher et al., 2018; Breiman, 2001). However, correlation among input features presents a challenge when
estimating feature importance.


Consider a motivating example adapted from Pearl
(2009), in which we are given a model _𝑓_ that takes
as input four features: the season of the year ( _𝑋_ 1 ),
whether or not it’s raining ( _𝑋_ 2 ), whether the sprinkler
is on ( _𝑋_ 3 ), and whether the pavement is wet ( _𝑋_ 4 ) and
outputs a prediction _𝑓_ ( **x** ), representing the probability that the pavement is slippery (capital _𝑋_ denotes
a random variable; lower case **x** denotes a particular
sample). Assume, the inputs are related through the
causal graph in **Figure 1** . When assigning feature importance, existing approaches that ignore this causal
structure (Janzing et al., 2020; Sundararajan and Najmi, 2019; Datta et al., 2016) assign zero importance to
the season, since it only indirectly affects the outcome
through the other input variables. However, such a
conclusion may lead a user astray - since changing _𝑋_ 1
would most definitely affect the outcome.


Recognizing this limitation, researchers have recently


proposed approaches that leverage the causal structure among the input variables when assigning credit
(Frye et al., 2019; Heskes et al., 2020). However, such
approaches provide an incomplete picture of a system
as they only assign credit to nodes in a graph. For example, the ASV method of Frye et al. (2019) solves the
earlier problem of ignoring indirect or upstream effects,
but it does so by ignoring direct or downstream effects.
In our example, season would get all the credit despite
the importance of the other variables. This again may
lead a user astray - since intervening on _𝑋_ 3 or _𝑋_ 4 would
affect the outcome, yet they are given no credit. The
Causal Shapley values of Heskes et al. (2020) do assign credit to _𝑋_ 3 and _𝑋_ 4, but force this credit to be
divided with _𝑋_ 1 . This leads to the problem of features being given less importance simply because their
downstream variables are also included in the graph.


Given that current approaches end up ignoring or dividing either downstream ( _i.e._, direct) or upstream
( _i.e._, indirect) effects, we develop Shapley Flow, a comprehensive approach to interpreting a model (or system) that incorporates the causal relationship among
input variables, while accounting for both direct and
indirect effects. In contrast to prior work, we accomplish this by reformulating the problem as one related
to assigning credit to _edges_ in a causal graph, instead
of _nodes_ ( **Figure 2c** ). Our key contributions are as
follows.


 We propose the first (to the best of our knowledge) generalization of Shapley value feature attribution to graphs, providing a complete systemlevel view of a model.


 - Our approach unifies three previous game theoretic approaches to estimating feature impor
tance.


 - Through examples on real data, we demonstrate
how our approach facilitates understanding feature importance.


In this work, we take an axiomatic approach motivated by cooperative game theory, extending Shapley
values to graphs. The resulting algorithm, Shapley
Flow, generalizes past work on estimating feature importance (Lundberg and Lee, 2017; Frye et al., 2019;
L´opez and Saboya, 2009). The estimates produced by
Shapley Flow represent the unique allocation of credit
that conforms to several natural axioms. Applied to
real-world systems, Shapley Flow can help a user understand both the direct and indirect impact of changing a variable, generating insights beyond current feature attribution methods.



(a) Independent (b) ASV (c) Shapley Flow


Figure 2: Top: Output of attribution methods for
the example in **Figure 1** . Bottom: Causal structure
(black edges) and explanation boundaries used by each
method. As a reference, we copied the true causal
links (red) from **Figure 1** . An explanation boundary B := ( _𝐷, 𝐹_ ) is a cut in the graph that defines a
“model” _𝐹_ (nodes in the shaded area in each figure)
to be explained. Refer to **Section 2.2** for a detailed
discussion.


**2** **Problem Setup & Background**


Given a model, or more generally a system, that takes
a set of inputs and produces an output, we focus on the
problem of quantifying the effect of each input on the
output. Here, building off previous work, we formalize
the problem setting.


**2.1** **Problem Setup**


Quantifying the effect of each input on a model’s output can be formulated as a credit assignment problem.
Formally, given a target sample input _𝒙_, a background
sample input _𝒙_ **[′]**, and a model _𝑓_ : R _[𝑑]_ → R, we aim
to explain the difference in output _i.e._, _𝑓_ ( _𝒙_ ) − _𝑓_ ( _𝒙_ **[′]** ).
We assume _𝒙_ and _𝒙_ [′] are of the same dimension _𝑑_, and
each entry can be either discrete or continuous.


We also assume access to a causal graph, as formally
defined in Chapter 6 of Peters et al. (2017), over the
_𝑑_ input variables. Given this graph, we seek an assignment function _𝜙_ that assigns credit _𝜙_ ( _𝑒_ ) ∈ R to
each edge _𝑒_ in the causal graph such that they collectively explain the difference _𝑓_ ( _𝒙_ ) − _𝑓_ ( _𝒙_ **[′]** ). In contrast with the classical setting (Lundberg and Lee,
2017; Sundararajan et al., 2017; Frye et al., 2020;
Aas et al., 2019) in which credit is placed on features
( _ı.e._, seeking a node assignment function _𝜓_ ( _𝑖_ ) ∈ R for
_𝑖_ ∈[1 · · · _𝑑_ ]), our edge-based approach is more flexible
because we can recover node _𝑖_ ’s importance by defining
_𝜓_ ( _𝑖_ ) = [�] _𝑒_ ∈ i’s outgoing edges _[𝜙]_ [(] _[𝑒]_ [)][. This exactly matches]
the classic Shapley axioms (Shapley, 1953) when the
causal graph is degenerate with a single source node
connected directly to all the input features.


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**



Here, the effect of the input on the output is measured
with respect to a background sample. For example, in
a healthcare setting, we may set the features in the
background sample to values that are deemed typical
for a disease. We assume a single background value for
notational convenience, but the formalism easily extends to the common scenario of multiple background
values or a distribution of background values, _𝑃_, by
defining the explanation target to be _𝑓_ ( _𝒙_ )−E _**𝒙**_ **[′]** ∼ _𝑃_ _𝑓_ ( _𝒙_ **[′]** ).


**2.2** **Feature Attribution with a Causal Graph**


In our problem setup, we assume access to a causal
graph, which can help in reasoning about the relationship among input variable. However, even with a
causal graph, feature attribution remains challenging
because it is unclear how to rightfully allocate credit
for a prediction among the nodes and/or edges of the
graph. Marrying interpretation with causality is an
active field (see Moraffah et al. (2020) for a survey).
A causal graph in and of itself does not solve feature
attribution. While a causal graph can be used to answer a specific question with a specific counterfactual,
summarizing many counterfactuals to give a comprehensive picture of the model is nontrivial. Furthermore, each node in a causal graph could be a blackbox
model that needs to be explained. To address this challenge, we generalize game theoretic fairness principles
to graphs.


Given a graph, G, that consists of a causal graph over
the the model of interest _𝑓_ and its inputs, we define
the **boundary of explanation** as a cut B := ( _𝐷, 𝐹_ )
that partitions the input variables and the output of
the model ( _i.e._, the nodes of the graph) into _𝐷_ and _𝐹_
where source nodes (nodes with no incoming edges) are
in _𝐷_ and sink nodes (nodes with no outgoing edges)
are in _𝐹_ . Note that G has a single sink, _𝑓_ ( _𝒙_ ) ∈ R. A
cut set is the set of edges with one endpoint in _𝐷_ and
another endpoint in _𝐹_, denoted as _𝑐𝑢𝑡_ (B). It is helpful
to think of _𝐹_ as an alternative model definition, where
a boundary of explanation ( _i.e._, a model boundary)
defines what part of the graph we consider to be the
“model”. If we collapse _𝐹_ into a single node that subsumes _𝑓_, then _𝑐𝑢𝑡_ (B) represents the direct inputs to
this new model.


Depending on the causal graph, multiple boundaries of
explanation may exist. Recognizing this multiplicity of
choices helps shed light on an ongoing debate in the
community regarding feature attribution and whether
one should perturb features while staying on the data
manifold or perturb them independently (Chen et al.,
2020; Janzing et al., 2020; Sundararajan and Najmi,
2019). On one side, many argue that perturbing features independently reveals the functional dependence
of the model, and is thus _true to the model_ (Janz


ing et al., 2020; Sundararajan and Najmi, 2019; Datta
et al., 2016). However, independent perturbation of
the data can create unrealistic or invalid sets of model

input values. Thus, on the other side, researchers argue that one should perturb features while staying on
the data manifold, and so be _true to the data_ (Aas
et al., 2019; Frye et al., 2019). However, this can result
in situations in which features not used by the model
are given non-zero attribution. Explanation boundaries help us unify these two viewpoints. As illustrated in **Figure 2a**, when we independently perturb
features, we assume the causal graph is flat and the explanation boundary lies between _𝒙_ and _𝑓_ ( _i.e._, _𝐷_ contains all of the input variables). In this example, since
features are assumed independent all credit is assigned
to the features that directly impact the model output,
and indirect effects are ignored (no credit is assigned to
_𝑋_ 1 and _𝑋_ 2 ). In contrast, when we perform on-manifold
perturbations with a causal structure, as is the case in
Asymmetric Shapley Values (ASV) (Frye et al., 2019),
all the credit is assigned to the source node because the
source node determines the value of all nodes in the
graph ( **Figure 2b** ). This results in a different boundary of explanation, one between the source nodes and
the remainder of the graph. Although giving _𝑋_ 1 credit
does not reflect the true functional dependence of _𝑓_, it
does for the model defined by _𝐹_ 2 ( **Figure 2c** ). Perturbations that were previously faithful to the data are
faithful to a “model”, just one that corresponds to a
different boundary. See **Section 6** in the Appendix
for how on-manifold perturbation (without a causal
graph) can be unified using explanation boundaries.


Beyond the boundary directly adjacent to the model
of interest, _𝑓_, and the boundary directly adjacent to
the source nodes, there are other potential boundaries ( **Figure 2c** ) a user may want to consider. However, simply generating explanations for each possible
boundary can quickly overwhelm the user ( **Figures**
**2a, 2b** in the main text, and **8a** in the Appendix).
Our approach sidesteps the issue of selecting a single
explanation boundary by considering all explanation
boundaries simultaneously. This is made possible by
assigning credit to the edges in a causal graph ( **Figure**
**2c** ). Edge attribution is strictly more powerful than
feature attribution because we can simultaneously capture the direct and indirect impact of edges. We note
that concurrent work by Heskes et al. (2020) also recognized that existing methods have difficulty capturing
the direct and indirect effects simultaneously. Their
solution however is node based, so it is forced to split
credit between parents and children in the graph.


While other approaches to assign credit on a graph exist, ( _e.g._, Conductance from Dhamdhere et al. (2018)
and DeepLift from Shrikumar et al. (2016)), they


(a) _𝑒_ 2 updates after _𝑒_ 1


(b) _𝑒_ 2 updates before _𝑒_ 1


Figure 3: Edge importance is measured by the change
in output when an edge is added. When a model is
non-linear, say _𝑓_ = _𝑂𝑅_, we need to average over all
scenarios in which _𝑒_ 2 can be added to gauge its importance. **Section 3.1** has a detailed discussion.


were proposed in the context of understanding internal nodes of a neural network, and depend on implicit
linearity and continuity assumptions about the model.
We aim to understand the causal structure among the
input nodes in a fully model agnostic manner, where
discrete variables are allowed, and no differentiability assumption is made. To do this we generalize the
widely used Shapley value (Adadi and Berrada, 2018;
Mittelstadt et al., 2019; Lundberg et al., 2018; Sundararajan and Najmi, 2019; Frye et al., 2019; Janzing
et al., 2020; Chen et al., 2020) to graphs.


**3** **Proposed Approach: Shapley Flow**


Our proposed approach, Shapley Flow, attributes
credit to edges of the causal graph. In this section, we
present the intuition behind our approach and then
formally show that it uniquely satisfies a generalization of the classic Shapley value axioms, while unifying
previously proposed approaches.


**3.1** **Assigning Credit to Edges: Intuition**


Given a causal graph defining the relationship among
input variables, we re-frame the problem of feature attribution to focus on the edges of a graph rather than
nodes. Our approach results in edge credit assignments as shown in **Figure 2c** . As mentioned above,
this eliminates the need for multiple explanations ( _i.e._,
bar charts) pertaining to each explanation boundary.
Moreover, it allows a user to better understand the nu


ances of a system by providing information regarding
what would happen if a single causal link breaks.


**Shapley Flow is the unique assignment of credit**
**to edges such that a relaxation of the classic**
**Shapley value axioms are satisfied for all possi-**
**ble boundaries of explanation.** Specifically, we extend the efficiency, dummy, and linearity axioms from
Shapley (1953) and add a new axiom related to boundary consistency. Efficiency states that the attribution
of edges on any boundary must add up to _𝑓_ ( **x** ) − _𝑓_ ( **x** [′] ).
Linearity states that explaining a linear combination
of models is the same as explaining each model, and
linearly combining the resulting attributions. Dummy
states that if adding an edge does not change the output in any scenarios, the edge should be assigned 0
credit. Boundary consistency states that edges shared
by different boundaries need to have the same attribution when explained using either boundary. These
concepts are illustrated in **Figure 4** and formalized in
**Section 3.3** .


An edge is important if removing it causes a large
change in the model’s prediction. However, what does
it mean to remove an edge? If we imagine every edge in
the graph as a channel that sends its source node’s current value to its target node, then removing an edge _𝑒_
simply means messages sent through _𝑒_ fail. In the context of feature attribution, in which we aim to measure
the difference between _𝑓_ ( **x** ) − _𝑓_ ( **x** [′] ), this means that
_𝑒_ ’s target node still relies on the source’s background
value in **x** [′] to update its current value, as opposed to
the source node’s foreground value in **x**, as illustrated
in **Figure 3a** . Note that treating edge removal as replacing the parent node with the background value is
equivalent to the approach advocated by Janzing et al.
(2020), and matches the default behavior of SHAP and
related methods. However, we cannot simply toggle
edges one at a time. Consider a simple OR function
_𝑔_ ( _𝑋_ 1 _, 𝑋_ 2 ) = _𝑋_ 1 ∨ _𝑋_ 2, with _𝑥_ 1 = 1, _𝑥_ 2 = 1, _𝑥_ 1 [′] [=][ 0,] _[ 𝑥]_ 2 [′] [=][ 0.]
Removing either of the edges alone, would not affect
the output and both _𝑥_ 1 and _𝑥_ 2 would be (erroneously)
assigned 0 credit.


To account for this, we consider all scenarios (or partial histories) in which the edge we care about can be
added (see **Figure 3b** ). Here, _𝜈_ is a function that
takes a list of edges and evaluates the network with
edges updated in the order specified by the list. For
example, _𝜈_ ([ _𝑒_ 1 ]) corresponds to the evaluation of _𝑓_
when only _𝑒_ 1 is updated. Similarly _𝜈_ ([ _𝑒_ 1 _, 𝑒_ 2 ]) is the
evaluation of _𝑓_ when _𝑒_ 1 is updated followed by _𝑒_ 2 . The
list [ _𝑒_ 1 _, 𝑒_ 2 ] is also referred to as a (complete) _history_
as it specifies how **x** [′] changes to **x** .


For the same edge, attributions derived from different
explanation boundaries should agree, otherwise simply


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**



including more details of a model in the causal graph
would change upstream credit allocation, even though
the model implementation was unchanged. We refer
to this property as _boundary consistency_ . The Shapley Flow value for an edge is the difference in model
output when removing the edge averaged over all histories that are boundary consistent (as defined below).


**3.2** **Model explanation as value assignments**
**in games**


The concept of Shapley value stems from game theory, and has been extensively applied in model interpretability (Strumbelj and Kononenko, 2014; Datta [ˇ]
et al., 2016; Lundberg and Lee, 2017; Frye et al., 2019;
Janzing et al., 2020). Before we formally extend it to
the context of graphs, we define the credit assignment
problem from a game theoretic perspective.


Given the message passing system in **Section 3.1**, we
formulate the credit assignment problem as a game
specific to an explanation boundary B := ( _𝐷, 𝐹_ ). The
game consists of a set of players P B, and a payoff function _𝜈_ B . We model each edge external to _𝐹_ as a player.
A _history_ is a list of edges detailing the event from _𝑡_ = 0
(values being **x** [′] ) to _𝑡_ = _𝑇_ (values being **x** ). For example, the history [ _𝑖, 𝑗, 𝑖_ ] means that the edge _𝑖_ finishes
transmitting a message containing its source node’s
most recent value to its target node, followed by the
edge _𝑗_, and followed by the edge _𝑖_ again. A _coalition_
is a partial history from _𝑡_ = 0 to any _𝑡_ ∈[0 · · · _𝑇_ ]. The
_payoff function_, _𝜈_, associates each coalition with a real
number, and is defined in our case as the evaluation of
_𝐹_ following the coalition.


This setup is a generalization of a typical cooperative
game in which the ordering of players does not matter
(only the set of players matters). However, given our
message passing system, history is important. In the
following sections, we denote ‘+’ as list concatenation,
possible histories.‘[]’ as an empty coalition, andWe denote H H˜ BB ⊆H as the set of all B as the set
of boundary consistent histories. The corresponding
coalitions for H B and H [˜] B are denoted as C B and C [˜] B
respectively. A sample game setup is illustrated in
**Figure 3** .


**3.3** **Axioms**


We formally extend the classic Shapley value axioms
(efficiency, linearity, and dummy) and include one additional axiom, the boundary consistency axiom, that
connects all boundaries together.


 - Boundary consistency: for any two boundaries
B 1 = ( _𝐷_ 1 _, 𝐹_ 1 ) and B 2 = ( _𝐷_ 2 _, 𝐹_ 2 ), _𝜙_ _𝜈_ B 1 ( _𝑖_ ) = _𝜙_ _𝜈_ B 2 ( _𝑖_ )
for _𝑖_ ∈ _𝑐𝑢𝑡_ (B 1 ) ∩ _𝑐𝑢𝑡_ (B 2 )



(a) Effi. + Bound. Consist. (b) Dummy player


(c) Linearity


Figure 4: Illustration for axioms for Shapley Flow. Except for boundary consistency, all axioms stem from
Shapley value’s axioms (Shapley, 1953). Detailed explanations are included in **Section 3.3** .


For edges that are shared between boundaries,
their attributions must agree. In **Figure 4a**, the
edge wrapped by a teal band is shared by both
the blue and green boundaries, forcing them to
give the same attribution to the edge.


In the general setting, not all credit assignments are
boundary consistent; different boundaries could result
in different attributions for the same edge [1] . This occurs when histories associated with different boundaries are inconsistent ( **Figure 5** ). Moving the boundary from B to B [∗] (where B [∗] is the boundary with _𝐷_
containing _𝑓_ ’s inputs), results in a more detailed set of
histories. This expansion has 2 constraints. First, any
history in the expanded set follows the message passing system in **Section 3.1** . Second, when a message
passes through the boundary, it immediately reaches
the end of computation as _𝐹_ is assumed to be a blackbox.


Denoting the history expansion function into B [∗] as
_𝐻𝐸_ ( _i.e._, _𝐻𝐸_ takes a history _ℎ_ as input and expand it
into a set of histories in B [∗] as output) and denoting
the set of all boundaries as M, a history _ℎ_ is _boundary_
_consistent_ if ∃ _ℎ_ B ∈H B for all B ∈M such that


( _𝐻𝐸_ ( _ℎ_ B )) ∩ _𝐻𝐸_ ( _ℎ_ ) ≠ ∅
�

B∈M


tory in which all boundaries can agree on.That is _ℎ_ needs to have at least one fully detailed his-H˜ is all


1 We include an example in the Appendix **Section 11**
to demonstrate why considering all histories H can violate
boundary consistency, thus motivating the need to only
focus on boundary consistent histories.


Figure 5: Boundary Consistency. For the blue boundary (upper), we show one potential history _ℎ_ . When we
expand _ℎ_ to the red boundary (lower), _ℎ_ corresponds
to multiple histories as long as each history contains
states that match (i) (ii) and (iii). (i’) matches (i), no
messages are received in both states. (ii’) matches (ii),
the full impact of message transmitted through the
left edge is received at the end of computation. (iii’)
matches (iii), all messages are received. In contrast,
the history containing (iv’) has no state matching (ii),
and thus is inconsistent with _ℎ_ .


histories in H that are boundary consistent. We rely
on this notion of boundary consistency in generalizing
the Shapley axioms to any explanation boundary, B:


 - Efficiency: [�] _𝑖_ ∈ _𝑐𝑢𝑡_ (B) _[𝜙]_ _𝜈_ B [(] _[𝑖]_ [)][ =] _[ 𝑓]_ [(] **[x]** [) −] _[𝑓]_ [(] **[x]** [′] [)][.]


In the general case where _𝜈_ B can depend on the or_𝜈_ B ( _ℎ_ )
dering of _ℎ_, the sum is [�] _ℎ_ ∈H [˜] B | H [˜] B | [−] _[𝜈]_ [B] [([])][.] But

when the game is defined by a model function _𝑓_,
� _ℎ_ ∈H [˜] B _[𝜈]_ [B] [(] _[ℎ]_ [)/|][ ˜] H B | = _𝑓_ ( _𝒙_ ) and _𝜈_ B ([]) = _𝑓_ ( _𝒙_ **[′]** ). An
illustration with 3 boundaries is shown in **Figure 4a** .


  - Linearity: _𝜙_ _𝛼𝑢_ + _𝛽𝑣_ = _𝛼𝜙_ _𝑢_ + _𝛽𝜙_ _𝑣_ for any payoff functions
_𝑢_ and _𝑣_ and scalars _𝛼_ and _𝛽_ .


Linearity enables us to compute a linear ensemble of
models by independently explaining each model and
then linearly weighting the attributions. Similarly, we
can explain _𝑓_ ( **x** )−E( _𝑓_ ( _𝑋_ [′] )) by independently computing attributions for each background sample **x** [(] **[i]** [)] [′] and
then taking the average of the attributions, without
recomputing from scratch whenever the background
sample’s distribution changes. An illustration with 2
background samples is shown in **Figure 4c** .


 - Dummy player: _𝜙_ _𝜈_ B ( _𝑖_ ) = 0 if _𝜈_ B ( _𝑆_ + [ _𝑖_ ]) = _𝜈_ B ( _𝑆_ ) for
all _𝑆, 𝑆_ + [ _𝑖_ ] ∈ C [˜] B for _𝑖_ ∈ _𝑐𝑢𝑡_ (B).


Dummy player states that if an edge does not change
the model’s output when added to in all possible coalitions, it should be given 0 attribution. In **Figure 4b**,
_𝑒_ 2 is a dummy edge because starting from any coalition, adding _𝑒_ 2 wouldn’t change the output.


These last three axioms are extensions of Shapley’s axioms. Note that Shapley value also requires the symmetry axiom because the game is defined on a set of
players. For Shapley Flow values this symmetry assumption is encoded through our choice of an ordered
history formulation. (Appendix **Section 8** ).



**3.4** **Shapley Flow is the unique solution**


Shapley Flow uniquely satisfies all axioms from the
previous section. Here, we describe the algorithm,
show its formulae, and state its properties. Please refer to **Appendix 7** and **8** for the pseudo code [2] and
proof.


**Description** : Define a configuration of a graph as an
arbitrary ordering of outgoing edges of a node when it
is traversed by depth first search. For each configuration, we run depth first search starting from the source
node, processing edges in the order of the configuration. When processing an edge, we update the value
of the edge’s target node by making the edge’s source
node value visible to its function. If the edge’s target
node is the sink node, the difference in the sink node’s
output is credited to every edge along the search path
from source to sink. The final result averages over
attributions for all configurations.


**Formulae** : Denote the attribution of Shapley Flow to
a path as _𝜙_ [˜] _𝜈_, and the set of all possible orderings of
source nodes to a sink path generated by depth first
search (DFS) as Π dfs . For each ordering _𝜋_ ∈ Π dfs, the
inequality of _𝜋_ ( _𝑗_ ) _< 𝜋_ ( _𝑖_ ) denotes that path _𝑗_ precedes
path _𝑖_ under _𝜋_ . Since _𝜈_ ’s input is a list of edges, we
define ˜ _𝜈_ to work on a list of paths. The evaluation of
_𝜈_ ˜ on a list of paths is the value of _𝑣_ evaluated on the
corresponding edge traversal ordering. Then



˜
_𝜙_ _𝜈_ ( _𝑖_ ) =
∑︁

_𝜋_ ∈Π dfs



_𝜈_ ˜([ _𝑗_ : _𝜋_ ( _𝑗_ ) ≤ _𝜋_ ( _𝑖_ )]) − _𝜈_ ˜([ _𝑗_ : _𝜋_ ( _𝑗_ ) _< 𝜋_ ( _𝑖_ )])

|Π dfs |


(1)



To obtain an edge _𝑒_ ’s attribution _𝜙_ _𝑣_ ( _𝑒_ ), we sum the
path attributions for all paths that contains _𝑒_ .


_𝜙_ _𝜈_ ( _𝑒_ ) = ∑︁ 1 _𝑝_ contains ( _𝑒_ ) _𝜙_ [˜] _𝜈_ ( _𝑝_ ) (2)

_𝑝_ ∈ paths in G


**Additional properties** : Shapley Flow has the following beneficial properties beyond the axioms.


Generalization of SHAP: if the graph is flat, the edge
attribution is equal to feature attribution from SHAP
because each input node is paired with a single edge
leading to the model.


Generalization of ASV: the attribution to the source

nodes is the same as in ASV if all the dependencies
among features are modeled by the causal graph.


Generalization of Owen value: if the graph is a tree,
the edge attribution for incoming edges to the leaf


2 code can be found in `[https://github.com/](https://github.com/nathanwang000/Shapley-Flow)`
```
nathanwang000/Shapley-Flow

```

**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


l



nodes is the Owen value (L´opez and Saboya, 2009)
with a coalition structure defined by the tree.


Implementation invariance: implementation invariance means that no matter how the function is implemented, so long as the input and output remain unchanged, so does the attribution (Sundararajan et al.,
2017), which directly follows boundary consistency
( _i.e._, knowing _𝑓_ ’s computational graph or not wouldn’t
change the upstream attribution).


Conservation of fowl : efficiency and boundary consistency imply that the sum of attributions on a node’s
incoming edges equals the sum of its outgoing edges.


Model agnostic: Shapley Flow can explain arbitrary
(non-differentiable) machine learning pipelines.


**4** **Practical Application**


Shapley Flow highlights both the direct and indirect
impact of features. In this section, we consider several
applications of Shapley Flow. First, in the context of
a linear model, we verify that the attributions match
our intuition. Second, we show how current feature
attribution approaches lead to an incomplete understanding of a system compared to Shapley Flow.


**4.1** **Experimental Setup**


We illustrate the application of Shapley Flow to a synthetic and a real dataset. In addition, we include results for a third dataset in the Appendix. Note that
our algorithm assumes a causal graph is provided as input. In recent years there has been significant progress
in causal graph estimation (Glymour et al., 2019; Peters et al., 2017). However, since our focus is not on
causal inference, we make simplifying assumptions in
estimating the causal graphs (see **Section 9.2** of the
Appendix for details).


**Datasets.** _Synthetic_ : As a sanity check, we first experiment with synthetic data. We create a random
graph dataset with 10 nodes. A node _𝑖_ is randomly
connected to node _𝑗_ (with _𝑗_ pointing to _𝑖_ ) with 0 _._ 5
probability if _𝑖> 𝑗_, otherwise 0. The function at each
node is linear with weights generated from a standard
normal distribution. Sources follow a _𝑁_ (0 _,_ 1) distribution. This results in a graph with a single sink node
associated with function _𝑓_ ( _i.e._, the ‘model’ of interest). The remainder of the graph corresponds to the
causal structure among the input variables.


_National Health and Nutrition Examination Survey_ :
This dataset consists of 9 _,_ 932 individuals with 18 demographic and laboratory measurements (Cox, 1998).
We used the same preprocessing as described by Lundberg et al. (2020). Given these inputs, the model, _𝑓_,



aims to predict survival.


**Model training.** We train _𝑓_ using an 80/20 random
train/test split. For experiments with linear models,
_𝑓_ is trained with linear regression. For experiments
with non-linear models, _𝑓_ is fitted by 100 XGBoost
trees with a max depth of 3 for up to 1000 epochs,
using the Cox loss.


**Causal Graph.** For the nutrition dataset, we constructed a causal graph ( **Figure 9a** ) based on our lim
l ited understanding of the causal relationship among

input variables. This graph represents an oversimplification of the true underlying causal relationships
and is for illustration purposes only. We assigned attributes predetermined at birth (age, race, and sex) as
source nodes because they temporally precede all other
features. Poverty index depends on age, race, and sex
(among other variables captured by the poverty index
noise variable) and impacts one’s health. Other features pertaining to health depend on age, race, sex,
and poverty index. Note that the relationship among
some features is deterministic. For example, pulse
pressure is the difference between systolic and diastolic blood pressure. We include causal edges to account for such facts. We also account for when features

have natural groupings. For example, transferrin saturation (TS), total iron binding capacity (TIBC), and
serum iron are all related to blood iron. Serum albu
min and serum protein are both blood protein measures. Systolic and diastolic blood pressure can be
grouped into blood pressure. Sedimentation rate and
white blood cell counts both measure inflammation.
We add these higher level grouping concepts as new
latent variables in the graph. To account for noise
in modeling the outcome ( _i.e._, the effect of exogenous
variables that are not used as input to the model),
we add an independent noise node to each node (detailed in **Section 9.2** in the Appendix). **The result-**
**ing causal structure is an oversimplification of**
**the true causal structure; the relationship be-**
**tween source nodes (e.g., race) and biomarkers**
**is far more complex (Robinson et al., 2020).**
**Nonetheless, it can help in understanding the**
**in/direct effects of input variables on the out-**

**come** .


**4.2** **Baselines**


We compare Shapley Flow with other game theoretic feature attribution methods: independent SHAP
(Lundberg and Lee, 2017), on-manifold SHAP (Aas
et al., 2019), and ASV (Frye et al., 2019), covering
both independent and on-manifold feature attribution.


Since Shapley value based methods are expensive to
compute exactly, we use a Monte Carlo approximation


of **Equation 1** . In particular, we sample orderings
from Π dfs and average across those orderings.We randomly selected a background sample from each dataset
and share it across methods so that each uses the

same background. A single background sample allows
us to ignore differences in methods due to variations
in background sampling and is easier to explain the
behavior of baselines (Merrick and Taly, 2020). To
show that our result is not dependent on the particular choice of background sample, we include an example averaged over 100 background samples in **Sec-**
**tion 10.4** in the Appendix (the qualitative results
shown with a single background still holds). We sample 10 _,_ 000 orderings from each approach to generate
the results. Since there’s no publicly available implementation for ASV, we show the attribution for source
nodes (the noise node associated with each feature)
obtained from Shapley Flow (summing attributions of
outgoing edges), as they are equivalent given the same
causal graph. Since noise node’s credit is used, intermediate nodes can report non zero credit in ASV.


For convenience of visual inspection, we show top 10
links used by Shapley Flow (credit measured in absolute value) on the nutrition dataset.


**4.3** **Sanity checks with linear models**


To build intuition, we first examine linear models ( _i.e._,
_𝑓_ ( **x** ) = **w** [⊤] **x** + _𝑏_ where **w** ∈ R _[𝑑]_ and _𝑏_ ∈ R; the causal
dependence inside the graph is also linear). When using a linear model the ground truth direct impact of
changing feature _𝑋_ _𝑖_ is _𝑤_ _𝑖_ ( _𝑥_ _𝑖_ − _𝑥_ _𝑖_ [′] [)][ (that is the change]
in output due to _𝑋_ _𝑖_ directly), and the ground truth indirect impact is defined as the change in output when
an intervention changes _𝑥_ _𝑖_ [′] [to] _[ 𝑥]_ _[𝑖]_ [. Note that when the]
model is linear, only 1 Monte Carlo sample is sufficient to recover the exact attribution because feature
ordering doesn’t matter (the output function is linear
in any boundary edges, thus only the background and
foreground value of a feature matters). This allows us
to bypass sampling errors and focus on analyzing the
algorithms.


Results for explaining the datasets are included in **Ta-**
**ble 1** . We report the mean absolute error (and its variance) associated with the estimated attribution (compared against the ground truth attribution), averaged
across 1 _,_ 000 randomly selected test examples and all
graph nodes for both datasets. Note that only Shapley flow results in no error for both direct and indirect
effects.


**4.4** **Examples with non-linear models**


We demonstrate the benefits of Shapley Flow with
non-linear models containing both discrete and con


Methods Nutrition ( **D** ) Synthetic ( **D** ) Nutrition ( **I** ) Synthetic ( **I** )


Independent **0.0** (± 0.0) **0.0** (± 0.0) 0.8 (± 2.7) 1.1 (± 1.4)
On-manifold 1.3 (± 2.5) 0.8 (± 0.7) 0.9 (± 1.6) 1.5 (± 1.5)
ASV 1.5 (± 3.3) 1.2 (± 1.4) 0.6 (± 1.9) 1.1 (± 1.5)
Shapley Flow **0.0** (± 0.0) **0.0** (± 0.0) **0.0** (± 0.0) **0.0** (± 0.0)


Table 1: Mean absolute error (std) for all methods
on direct ( **D** ) and indirect ( **I** ) effect for linear models.
Shapley Flow makes no mistake across the board.


tinuous variables. As a reminder, the baseline methods are not competing with Shapley Flow as the latter
can recover all the baselines given the corresponding
causal structure ( **Figure 2** ). Instead, we highlight
why a holistic understanding of the system is better.


**Independent SHAP ignores the indirect impact**
**of features** . Take an example from the nutrition
dataset ( **Figure 6** ). Independent SHAP gives lower
attribution to age compared to ASV. This happens because age, in addition to its direct impact, indirectly
affects the output through blood pressure, as shown by
Shapley Flow ( **Figure 6a** ). Independent SHAP fails
to account for the indirect impact of age, leaving the
user with a potentially misleading impression that age
is less important than it actually is.


**On-manifold SHAP provides a misleading in-**
**terpretation** . With the same example ( **Figure 6** ),
we observe that on-manifold SHAP strongly disagrees
with independent SHAP, ASV, and Shapley Flow on
the importance of age. Not only does it assign more
credit to age, it also flips the sign, suggesting that age
is protective. However, **Figure 7a** shows that age and
earlier mortality are positively correlated; then how
could age be protective? **Figure 7b** provides an explanation. Since SHAP considers all partial histories
regardless of the causal structure, when we focus on
serum magnesium and age, there are two cases: serum
magnesium updates before or after age. We focus on
the first case because it is where on-manifold SHAP
differs from other baselines (all baselines already consider the second case as it satisfies the causal ordering).
When serum magnesium updates before age, the expected age given serum magnesium is higher than the
foreground age (yellow line above the black marker).
Therefore when age updates to its foreground value,
we observe a decrease in age, leading to a decrease
in the output (so age appears to be protective). From
both an in/direct impact perspective, on-manifold perturbation can be misleading since it is based not on
causal but on observational relationships.


**ASV ignores the direct impact of features** . As
shown in **Figure 6**, serum protein appears to be more
important in independent SHAP compared to ASV.
From Shapley Flow ( **Figure 6a** ), we know serum protein is not given attribution in ASV because its up

**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**



stream node, blood protein, gets all the credit. However, looking at ASV alone, one fails to understand
that intervening on serum protein could have a larger
impact on the output.


**Shapley Flow shows both direct and indirect**
**impacts of features** . Focusing on the attribution
given by Shapley Flow ( **Figure 6a** ). We not only
observe similar direct impacts in variables compared
to independent SHAP, but also can trace those impacts to their source nodes, similar to ASV. Furthermore, Shapley Flow provides more detail compared to
other approaches. For example, using Shapley Flow
we gain a better understanding of the ways in which
age impacts survival. The same goes for all other features. This is useful because causal links can change
(or break) over time. Our method provides a way to
reason through the impact of such a change.


More case studies with an additional dataset are in
cluded in the Appendix.


**5** **Discussion and Conclusion**


We extend the classic Shapley value axioms to causal
graphs, resulting in a unique edge attribution method:
Shapley Flow. It unifies three previous Shapley value
based feature attribution methods, and enables the
joint understanding of both the direct and indirect
impact of features. This more comprehensive understanding is useful when interpreting any machine
learning model, both ‘black box’ methods, and ‘interpretable’ methods (such as linear models).


The key message of the paper is that model interpretation methods should include the whole machine learn
ing pipeline in order to understand when a model can
be applied. While our approach relies on access to
a complete causal graph, Shapley Flow is still valuable because a) there are well-established causal relationships in domains such as healthcare, and ignoring such relationships can produce confusing explanations; b) recent advancements in causal estimation are
complementary to our work and make defining these
graphs easier; c) finally and most importantly, existing
methods already implicitly make causal assumptions,
Shapley Flow just makes these assumptions explicit
( **Figure 2** ). However, this does open up new research
opportunities. Can Shapley Flow work with partially
defined causal graphs? How to explore Shapley Flow
attribution when the causal graph is complex? Can
Shapley Flow be useful for feature selection? We leave
those questions for future work.



Top features Age Serum Magnesium Serum Protein


Background sample 35 1.37 7.6
Foreground sample 40 1.19 6.5


Attributions Independent On-manifold ASV


Iron 0.0 0.0 -0.0

Inflamation 0.0 0.0 0.0


(a) Shapley Flow


Figure 6: Comparison among baselines on a sample
(top table) from the nutrition dataset, showing top 10
features/edges.


(a) Age vs. output (b) Age vs. magnesium


Figure 7: Age appears to be protective in on-manifold
SHAP because it steals credit from other variables.


**References**


Aas, K., Jullum, M., and Løland, A. (2019). Explaining individual predictions when features are dependent: More accurate approximations to shapley values. _arXiv preprint arXiv:1903.10464_ .


Adadi, A. and Berrada, M. (2018). Peeking inside the
black-box: A survey on explainable artificial intelligence (xai). _IEEE Access_, 6:52138–52160.


Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe,
M., Hansen, K., and M¨uller, K.-R. (2010). How to
explain individual classification decisions. _The Jour-_
_nal of Machine Learning Research_, 11:1803–1831.


Binder, A., Montavon, G., Lapuschkin, S., M¨uller, K.R., and Samek, W. (2016). Layer-wise relevance
propagation for neural networks with local renormalization layers. In _International Conference on_
_Artificial Neural Networks_, pages 63–71. Springer.


Breiman, L. (2001). Random forests. _Machine learn-_
_ing_, 45(1):5–32.


Chen, H., Janizek, J. D., Lundberg, S., and Lee, S.I. (2020). True to the model or true to the data?
_arXiv preprint arXiv:2006.16234_ .


Cox, C. S. (1998). _Plan and operation of the NHANES_
_I Epidemiologic Followup Study, 1992_ . Number 35.
National Ctr for Health Statistics.


Datta, A., Sen, S., and Zick, Y. (2016). Algorithmic
transparency via quantitative input influence: Theory and experiments with learning systems. In _2016_
_IEEE symposium on security and privacy (SP)_,
pages 598–617. IEEE.


Dhamdhere, K., Sundararajan, M., and Yan, Q.
(2018). How important is a neuron? _arXiv preprint_
_arXiv:1805.12233_ .


Fisher, A., Rudin, C., and Dominici, F. (2018). All
models are wrong but many are useful: Variable importance for black-box, proprietary, or misspecified
prediction models, using model class reliance. _arXiv_
_preprint arXiv:1801.01489_, pages 237–246.


Frye, C., de Mijolla, D., Cowton, L., Stanley, M., and
Feige, I. (2020). Shapley-based explainability on the
data manifold. _arXiv preprint arXiv:2006.01272_ .


Frye, C., Feige, I., and Rowat, C. (2019). Asymmetric shapley values: incorporating causal knowledge
into model-agnostic explainability. _arXiv preprint_
_arXiv:1910.06358_ .


Glymour, C., Zhang, K., and Spirtes, P. (2019). Review of causal discovery methods based on graphical
models. _Frontiers in genetics_, 10:524.


Heskes, T., Sijben, E., Bucur, I. G., and Claassen, T.
(2020). Causal shapley values: Exploiting causal



knowledge to explain individual predictions of complex models. _Advances in neural information pro-_
_cessing systems_ .


Janzing, D., Minorics, L., and Bl¨obaum, P. (2020).
Feature relevance quantification in explainable ai: A
causal problem. In _International Conference on Ar-_
_tificial Intelligence and Statistics_, pages 2907–2916.
PMLR.


L´opez, S. and Saboya, M. (2009). On the relationship
between shapley and owen values. _Central European_
_Journal of Operations Research_, 17(4):415.


Lundberg, S. M., Erion, G., Chen, H., DeGrave, A.,
Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J.,
Bansal, N., and Lee, S.-I. (2020). From local explanations to global understanding with explainable ai
for trees. _Nature machine intelligence_, 2(1):2522–
5839.


Lundberg, S. M., Erion, G. G., and Lee, S.-I. (2018).
Consistent individualized feature attribution for tree

ensembles. _arXiv preprint arXiv:1802.03888_ .


Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In _Ad-_
_vances in neural information processing systems_,
pages 4765–4774.


Merrick, L. and Taly, A. (2020). The explanation
game: Explaining machine learning models using
shapley values. In Holzinger, A., Kieseberg, P.,
Tjoa, A. M., and Weippl, E., editors, _Machine_
_Learning and Knowledge Extraction_, pages 17–38,
Cham. Springer International Publishing.


Mittelstadt, B., Russell, C., and Wachter, S. (2019).
Explaining explanations in ai. In _Proceedings of the_
_conference on fairness, accountability, and trans-_
_parency_, pages 279–288.


Moraffah, R., Karami, M., Guo, R., Raglin, A., and
Liu, H. (2020). Causal interpretability for machine
learning-problems, methods and evaluation. _ACM_
_SIGKDD Explorations Newsletter_, 22(1):18–33.


Pearl, J. (2009). _Causality_ . Cambridge university

press.


Peters, J., Janzing, D., and Sch¨olkopf, B. (2017). _El-_
_ements of causal inference_ . The MIT Press.


Robinson, W. R., Renson, A., and Naimi, A. I.
(2020). Teaching yourself about structural racism
will improve your machine learning. _Biostatistics_,
21(2):339–344.


Shapley, L. S. (1953). A value for n-person games.
_Contributions to the Theory of Games_, 2(28):307–
317.


Shrikumar, A., Greenside, P., and Kundaje, A.
(2017). Learning important features through prop

**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


agating activation differences. _arXiv preprint_
_arXiv:1704.02685_ .


Shrikumar, A., Greenside, P., Shcherbina, A., and
Kundaje, A. (2016). Not just a black box: Learning
important features through propagating activation
differences. _arXiv preprint arXiv:1605.01713_ .


Simonyan, K., Vedaldi, A., and Zisserman, A. (2013).
Deep inside convolutional networks: Visualising image classification models and saliency maps. _arXiv_
_preprint arXiv:1312.6034_ .


Springenberg, J. T., Dosovitskiy, A., Brox, T.,
and Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. _arXiv preprint_
_arXiv:1412.6806_ .
ˇStrumbelj, E. and Kononenko, I. (2014). Explaining
prediction models and individual predictions with
feature contributions. _Knowledge and information_
_systems_, 41(3):647–665.


Sundararajan, M. and Najmi, A. (2019). The many
shapley values for model explanation. _arXiv preprint_
_arXiv:1908.08474_ .


Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic attribution for deep networks. _International_
_Conference on Machine Learning_ .


Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and
Torralba, A. (2016). Learning deep features for discriminative localization. In _Proceedings of the IEEE_
_conference on computer vision and pattern recogni-_
_tion_, pages 2921–2929.


(a) On-manifold attribution (b) On-manifold boundary


Figure 8: On manifold perturbation methods can be computed using Shapley Flow with a specific explanation
boundary.


**6** **Explanation boundary for on-manifold methods without a causal graph**


On-manifold perturbation using conditional expectations can be unified with Shapley Flow using explanation
boundaries ( **Figure 8a** ). Here we introduce _𝑋_ [˜] _𝑖_ as an auxiliary variable that represent the imputed version of
_𝑋_ _𝑖_ . Perturbing any feature _𝑋_ _𝑖_ affects all input to the model ( _𝑋_ [˜] 1, _𝑋_ [˜] 2, _𝑋_ [˜] 3, _𝑋_ [˜] 4 ) so that they respect the correlation
in the data after the perturbation. When _𝑋_ _𝑖_ has not been perturbed, _𝑋_ [˜] _𝑗_ treats it as missing for _𝑖, 𝑗_ ∈[1 _,_ 2 _,_ 3 _,_ 4]
and would sample _𝑋_ [˜] _𝑗_ from the conditional distribution of _𝑋_ _𝑗_ given non-missing predecessors. The red edges
contain causal links from **Figure 1**, whereas the black edges are the causal structure used by the on-manifold
perturbation method. The credit is equally split among the features because they are all correlated. Again,
although giving _𝑋_ 1 and _𝑋_ 2 credit is not true to _𝑓_, it is true to the model defined by _𝐹_ .


**7** **The Shapley Flow algorithm**


A pseudo code implementation highlighting the main ideas for Shapley Flow is included in **Algorithm 1** . For
approximations, instead of trying all edge orderings in line 15 of **Algorithm 1**, one can try random orderings
and average over the number of orderings tried.


**8** **Shapley Flow’s uniqueness proof**


Without loss of generality, we can assume G has a single source node _𝑠_ . We can do this because every node in
a causal graph is associated with an independent noise node (Peters et al., 2017, Chapter 6). For deterministic
relationships, the function for a node doesn’t depend on its noise. Treating those noise nodes as a single node, _𝑠_,
wouldn’t have changed any boundaries that already exist in the original graph. Therefore we can assume there
is a single source node _𝑠_ .


**8.1** **At most one solution satisfies the axioms**


Assuming that a solution exists, we show that it must be unique.


_Proof._ We adapt the argument from the Shapley value uniqueness proof [3], by defining basis payoff functions as
carrier games. Choose any boundary B, we show here that any game defined on the boundary has a unique
attribution. We also drop the subscript B in the proof as there is no ambiguity. Note that since every edge will
appear in some boundary, if all boundary edges are uniquely attributed to, all edges have unique attributions. A
carrier game associated with coalition (ordered list) _𝑂_ is a game with payoff function _𝑣_ _[𝑂]_ such that _𝑣_ _[𝑂]_ ( _𝑆_ ) = 1(0)
if coalition _𝑆_ starts with _𝑂_ (otherwise 0). By dummy player, we know that only the last edge _𝑒_ in _𝑂_ gets credit
and all other edges in the cut set are dummy because a coalition is constructed in order (only adding _𝑒_ changes
the payoff from 0 to 1). Note that in contrast with the traditional symmetry axiom (Shapley, 1953) defined


3 `[https://ocw.mit.edu/courses/economics/14-126-game-theory-spring-2016/lecture-notes/MIT14_126S16_](https://ocw.mit.edu/courses/economics/14-126-game-theory-spring-2016/lecture-notes/MIT14_126S16_cooperative.pdf)`
```
cooperative.pdf

```

**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


**Algorithm 1** Shapley Flow pseudo code
**Input:** A computational graph G (each node _𝑖_ has a function _𝑓_ _𝑖_ ), foreground sample **x**, background sample **x** [′]

**Output:** Edge attribution _𝜙_ : _𝐸_ → R
**Initialization:**

G: add an new source node pointing to original source nodes.

1: **function** ShapleyFlow(G, **x** [′], **x** )
2: Initialize(G, **x** [′], **x** ) _⊲_ Set up game _𝜈_ for any boundary in G
3: _𝑠_ ← source(G) _⊲_ Obtain the source node
4: **return** DFS( _𝑠_, {}, [])
5: **end function**


6: **function** DFS( _𝑠_, _𝐷_, _𝑆_ )
7: _⊲𝑠_ is a node, _𝐷_ is the data side of the current boundary, _𝑆_ is coalition
8: _⊲_ Using Python list slice notation
9: Initialize _𝜙_ to output 0 for all edges
10: **if** IsSinkNode(s) **then**
11: _⊲_ Here we overload _𝐷_ to refer to its boundary
12: _𝜙_ ( _𝑆_ [−1]) ← _𝜈_ _𝐷_ ( _𝑆_ ) − _𝜈_ _𝐷_ ( _𝑆_ [: −1]) _⊲_ Difference in output is attributed to the edge
13: **return** _𝜙_
14: **end if**


15: **for** _𝑝_ ← AllOrderings(Children( _𝑠_ )) **do** _⊲_ Try all orderings/permutations of the node’s children
16: **for** _𝑐_ ← _𝑝_ **do** _⊲_ Follow the permutation to get the node one by one
17: edgeCredit ← DFS( _𝑐_, _𝐷_ ∪{ _𝑠_ }, _𝑆_ + [( _𝑠, 𝑐_ )]) _⊲_ Recurse downward


18: _𝜙_ ← _𝜙_ + NumChildrenedgeCredit ( _𝑠_ ) ! _⊲_ Average attribution over number of runs

19: _𝜙_ ( _𝑆_ [−1]) ← _𝜙_ ( _𝑆_ [−1]) + NumChildrenedgeCredit ( _𝑠,_ ( _𝑐𝑠_ )) ! _⊲_ Propagate upward
20: **end for**

21: **end for**

22: **return** _𝜙_
23: **end function**


on a set of players, the symmetry axiom is not explicit in our case (it is made implicitly) because not all edges
in the carrier game are symmetric with each other (observe that _𝑒_ is different from all other edges, which are
dummy), thus we do not need an explicit symmetry axiom to argue for unique attribution in the carrier game.
Furthermore, _𝑒_ must be an edge in the boundary to form a valid game because boundary edges are the only
edges that are connected to the model defined by the boundary. Therefore we give 0 credit to edges in the cut
_𝜈_ B ( _ℎ_ )
set other than _𝑒_ (because they are _dummy players_ ). By the _efficiency axiom_, we give [�] _ℎ_ ∈H [˜] | H| [˜] [−] _[𝜈]_ [B] [([])][ credit]

to _𝑒_ where H [˜] is the set of all possible boundary consistent histories as defined in **Section 3.3** . This uniquely
attributed the boundary edges for this game.


We show that the set of carrier games associated with every coalition that ends in a boundary edge (denoted
as C [ˆ] ) form basis functions for all payoff functions associated with the system. Recall from **Section 3.2** that C [˜]
is the set of _boundary consistent coalitions_ . We show here that payoff value on coalitions from C [˜] is redundant
given C [ˆ] . Note that C [˜] _\_ C [ˆ] represents all the coalitions that do not end in a boundary edge. For _𝑐_ ∈ C [˜] _\_ C [ˆ],
_𝑣_ _[𝑂]_ ( _𝑐_ ) = _𝑣_ _[𝑂]_ ( _𝑐_ [: −1]) (using Python’s slice notation on list) because only boundary edges are connected to the
model defined by the boundary. Therefore it suffices to show that _𝑣_ _[𝑂]_ is linearly independent for _𝑂_ ∈ C [ˆ] . For
a contradiction, assume for all _𝑐_ ∈ C [ˆ], [�] _𝑂_ ⊆ C [ˆ] _[𝛼]_ _[𝑂]_ _[𝑣]_ _[𝑂]_ [(] _[𝑐]_ [)][ =][ 0, with some non zero] _[ 𝛼]_ _[𝑂]_ [∈] [R][ (definition of linear]
dependence). Let _𝑆_ be a coalition with minimal length such that _𝛼_ _[𝑆]_ ≠ 0. We have [�] _𝑂_ ⊆ C [ˆ] _[𝛼]_ _[𝑂]_ _[𝑣]_ _[𝑂]_ [(] _[𝑆]_ [)][ =] _[ 𝛼]_ _[𝑆]_ [, a]
contradiction.


Therefore for any _𝜈_ we have unique _𝛼_ ’s such that _𝜈_ = [�] _𝑂_ ⊆ C [ˆ] _[𝛼]_ _[𝑂]_ _[𝑣]_ _[𝑂]_ [. Using the] _[ linearity axiom]_ [, we have]


_𝜙_ _𝜈_ = _𝜙_ [�] _𝑂_ ⊆C [ˆ] _[𝛼]_ _[𝑂]_ _[𝑣]_ _[𝑂]_ [=] ∑︁ _𝛼_ _[𝑂]_ _𝜙_ _𝑣_ _𝑂_

_𝑂_ ⊆ C [ˆ]


The uniqueness of _𝛼_ and _𝜙_ _𝑣_ _𝑂_ makes the attribution unique if a solution exists. Axioms used in the proof are
italicized.


                                      

**8.2** **Shapley Flow satisfies the axioms**


_Proof._ We first demonstrate how to generate all boundaries. Then we show that Shapley Flow gives boundary
consistent attributions. Following that, we look at the set of histories that can be generated by DFS in boundary
B, denoted as Π [dfs] B [. We show that][ Π] [dfs] B [=][ ˜] H B . Using this fact, we check the axioms one by one.


 - Every boundary can be “grown” one node at a time from _𝐷_ = { _𝑠_ } where _𝑠_ is the source node: Since the
computational graph G is a directed acyclic graph (DAG), we can obtain a topological ordering of the nodes
in G. Starting by including the first node in the ordering (the source node _𝑠_ ), which defines a boundary
as ( _𝐷_ = { _𝑠_ } _, 𝐹_ = Nodes(G)\ _𝐷_ ), we grow the boundary by adding nodes to _𝐷_ (removing nodes from _𝐹_ ) one
by one following the topological ordering. This ordering ensures the corresponding explanation boundary is
valid because the cut set only flows from _𝐷_ to _𝐹_ (if that’s not true, then one of the dependency nodes is
not in _𝐷_, which violates topological ordering).


Now we show every boundary can be “grown” in this fashion. In other words, starting from an arbitrary
boundary B 1 = ( _𝐷_ 1 _, 𝐹_ 1 ), we can “shrink” one node at a time to _𝐷_ = { _𝑠_ } by reversing the growing procedure.
First note that, _𝐷_ 1 must have a node with outgoing edges only pointing to nodes in _𝐹_ 1 (if that’s not the
case, we have a cycle in this graph because we can always choose to go through edges internal to _𝐷_ 1 and
loop indefinitely). Therefore we can just remove that node to arrive at a new boundary (now its incoming
edges are in the cut set). By the same argument, we can keep removing nodes until _𝐷_ = { _𝑠_ }, completing the
proof.


 - Shapley Flow gives boundary consistent attributions: We show that every boundary grown has edge attribution consistent with the previous boundary. Therefore all boundaries have consistent edge attribution
because the boundary formed by any two boundary’s common set of nodes can be grown into those two
boundaries using the property above. Let’s focus on the newly added node _𝑐_ from one boundary to the
next. Note that a property of depth first search is that every time _𝑐_ ’s value is updated, its outgoing edges
are activated in an atomic way (no other activation of edges occur between the activation of _𝑐_ ’s outgoing


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


edges). Therefore, the change in output due to the activation of new edges occur together in the view of
edges upstream of _𝑐_, thus not changing their attributions. Also, since _𝑐_ ’s outgoing edges must point to the
model defined by the current boundary (otherwise it cannot be a valid topological ordering), they don’t
have down stream edges, concluding the proof.


 - Π [dfs] B = H [˜] B : Since attribution is boundary consistent, we can treat the model as a blackbox and only look
at the DFS ordering on the data side. Observe that the edge traversal ordering in DFS is a valid history
because a) every edge traversal can be understood as a message received through edge, b) when every
message is received, the node’s value is updated, and c) the new node’s value is sent out through every
outgoing edge by the recursive call in DFS. Therefore the two side of the equation are at least holding the
same type of object.

We first show that Π [dfs] B ⊆ H [˜] B . Take _ℎ_ ∈ Π [dfs] B [, we need to find a history] _[ ℎ]_ [∗] [in][ B] [∗] [such that a)] _[ ℎ]_ [can be]
expanded into _ℎ_ [∗] and b) for any boundary, there is a history in that boundary that can be expanded into
_ℎ_ [∗] . Let _ℎ_ [∗] be any history expanded using DFS that is aligned with _ℎ_ . To show that every boundary can
expand into _ℎ_ [∗], we just need to show that the boundaries generated through the growing process introduced
in the first bullet point can be expanded into _ℎ_ [∗] . The base case is _𝐷_ = { _𝑠_ }. There must have an ordering
to expand into _ℎ_ [∗] because _ℎ_ [∗] is generated by DFS, and that DFS ensures that every edge’s impact on the
boundary is propagated to the end of computation before another edge in _𝐷_ is traversed. Similarly, for the
inductive step, when a new node _𝑐_ is added, we just follow the expansion of its previous boundary to reach
_ℎ_ [∗] .

Next we show that H˜ B ⊆ Π [dfs] B [.] First observe that for history _ℎ_ 1 in B 1 = ( _𝐷_ 1 _, 𝐹_ 1 ) and history _ℎ_ 2 in
B 2 = ( _𝐷_ 2 _, 𝐹_ 2 ) with _𝐹_ 2 ⊆ _𝐹_ 1, if _ℎ_ 1 cannot be expanded into _ℎ_ 2, then _𝐻𝐸_ ( _ℎ_ 1 ) ∩ _𝐻𝐸_ ( _ℎ_ 2 ) = ∅ because they
already have mismatches for histories that doesn’t involve passing through B 1 . Assume we do have _ℎ_ ∈ H [˜] B
but _ℎ_ ∉Π [dfs] B [. To derive a contradiction, we shrink the boundary one node at a time from][ B][, again using]
the procedure described in the first bullet point. We denote the resulting boundary formed by removing _𝑛_
nodes as B − _𝑛_ . Since _ℎ_ is assumed to be boundary consistent, there exist _ℎ_ B − 1 ∈H B − 1 such that _ℎ_ B − 1 must
be able to expand into _ℎ_ . Say the two boundaries differ in node _𝑐_ . Note that any update to _𝑐_ crosses B − 1,
therefore its impact must be reached by _𝐹_ before another event occurs in _𝐷_ − 1 . Since all of _𝑐_ ’s outgoing edges
crosses B, any ordering of messages sent through those edges is a DFS ordering from _𝑐_ . This means that if
_ℎ_ B − 1 can be reached by DFS, so can _ℎ_ B, violating the assumption. Therefore, _ℎ_ B − 1 ∉Π [dfs] B − 1 [and] _[ ℎ]_ [B] [−] [1] [∈] H [˜] B − 1
(the latter because _ℎ_ B − 1 can expand into a history that is consistent with all boundaries by first expanding
into _ℎ_ ). We run the same argument until _𝐷_ = { _𝑠_ }. This gives a contradiction because in this boundary, all
histories can be produced by DFS.


 - Efficiency: Since we are attributing credit by the change in the target node’s value following a history _ℎ_
given by DFS, the target for this particular DFS run is thus _𝜈_ B ( _ℎ_ ) − _𝜈_ B ([]). Average over all DFS runs
and noting that H [˜] B = Π [dfs] B gives the target [�] _ℎ_ ∈H [˜] B _[𝜈]_ [B] [(] _[ℎ]_ [)/|][ ˜] H B | − _𝜈_ B ([]). Noting that each update in the
target node’s value must flow through one of the boundary edges. Therefore the sum of boundary edges’
attribution equals to the target.


 - Linearity: For two games of the same boundary _𝑣_ and _𝑢_, following any history, the sum of output differences
between the two games is the output difference of the sum of the two games, therefore _𝜙_ _𝑣_ + _𝑢_ would not differ
from _𝜙_ _𝑣_ + _𝜙_ _𝑢_ . It’s easy to see that extending addition to any linear combination wouldn’t matter.


 Dummy player: Since Shapley Flow is boundary consistent, we can just run DFS up to the boundary (treat
_𝐹_ as a blackbox). Since every step in DFS remains in the coalition C [˜] B because Π [dfs] B ⊆ H [˜] B, if an edge is
dummy, every time it is traversed through by DFS, the output won’t change by definition, thus giving it 0
credit.


                                      

Therefore Shapley Flow uniquely satisfies the axioms. We note that efficiency requirement simplifies to _𝑓_ ( _𝒙_ ) −
_𝑓_ ( _𝒙_ **[′]** ) when applying it to an actual model because all histories from DFS would lead the target node to its
target value. We can prove a stronger claim that actually all nodes would reach its target value when DFS
finishes. To see that, we do an induction on a topological ordering of the nodes. The source nodes reaches its
final value by definition. Assume this holds for the _𝑘_ [th] node. For the _𝑘_ + 1 [th] node, its parents achieves target


value by induction. Therefore DFS would make the parents’ final values visible to this node, thus updating it to
the target value.


**9** **Causal graphs**


While the nutrition dataset is introduced in the main text, we describe an additional dataset to further demonstrate the usefulness of Shapley Flow. Moreover, we describe in detail how the causal relationship is estimated.
The resulting causal graphs for the nutrition dataset and the income dataset are visualized in **Figure 9** .


**9.1** **The Census Income dataset**


The Census Income dataset consists of 32 _,_ 561 samples with 12 features. The task is to predict whether one’s
annual income exceeds 50 _𝑘_ . We assume a causal graph, similar to that used by Frye et al. (2019) ( **Figure 9b** ).
Attributes determined at birth e.g., sex, native country, and race act as source nodes. The remaining features
(marital status, education, relationship, occupation, capital gain, work hours per week, capital loss, work class)
have fully connected edges pointing from their causal ancestors. All features have a directed edge pointing to
the model.


**9.2** **Causal Effect Estimation**


Given the causal structure described above, we estimate the relationship among variables using XGBoost. More
specifically, using an 80/20 train test split, we use XGBoost to learn the function for each node. If the node value
is categorical, we train to minimize cross entropy loss. Otherwise, we minimize mean squared error. Models are
fitted by 100 XGBoost trees with a max depth of 3 for up to 1000 epochs. Since features are rarely perfectly
determined by their dependency node, we add independent noise nodes to account for this effect. That is, each
non-sink node is pointed to by a unique noise node that account for the residue effect of the prediction.


Depending on whether the variable is discrete or continuous, we handle the noise differently. For continuous
variables, the noise node’s value is the residue between the prediction and the actual value. For discrete variables,
we assume the actual value is sampled from the categorical distribution specified by the prediction. Therefore the
noise node’s value is any possible random number that could result in the actual value. As a concrete example
for handling discrete variable, consider a binary variable _𝑦_, and assume the trained categorical function _𝑓_ gives
_𝑓_ ( _𝒙_ ) = [0 _._ 3 _,_ 0 _._ 7] where _𝒙_ is the foreground value of the input to predict _𝑦_ . We view the data generation as the
following. The noise term associated with _𝑦_ is treated as a uniform random variable between 0 and 1. If it lands
within 0 to 0 _._ 3, _𝑦_ is sampled to be 0, otherwise 1 (matching the categorical function of 70% chance of sampling
_𝑦_ to be 1). Now if we observe the foreground value of _𝑦_ to be 0, it means the foreground value of noise must be
uniform between 0 to 0 _._ 3. Although we cannot infer the exact value of the noise, we can sample the noise from
0 to 0 _._ 3 multiple times and average the resulting attribution.


**10** **Additional Results**


In this section, we first present additional sanity checks with synthetic data. Then we show additional examples
from both the nutrition and income datasets to demonstrate how a complete view of boundaries should be
preferable over single boundary approaches.


**10.1** **Additional Sanity Checks**


We include further sanity check experiments in this section. The first sanity check consists of a chain with 4
variables. Each node along the chain is an identical copy of its predecessor and the function to explain only
depends on _𝑋_ 4 ( **Figure 10** ). The dataset is created by sampling _𝑋_ 1 ∼N (0 _,_ 1), that is a standard normal
distribution, with 1000 samples. We use the first sample as background, and explain the second sample (one
can choose arbitrary samples to obtain the same insights). As shown in **Figure 10**, independent SHAP fails to
show the indirect impact of _𝑋_ 1, _𝑋_ 2, and _𝑋_ 3, ASV fails to show the direct impact of _𝑋_ 4, on manifold SHAP fails
to fully capture both the direct and indirect importance of any edge.


The second sanity check consists of linear models as described in **Section 4.3** . We include the full result with


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


(a) Causal graph for the nutrition dataset


(b) Causal graph for the Census Income dataset


Figure 9: The causal graphs we used for the two real datasets. Note that each node in the causal graph for (a) is
given a noise node to account for random effects. The noise nodes are omitted for better readability for (b). The
resulting causal structures are over-simplifications of the true causal structure; the relationship between source
nodes (e.g., race and sex) and other features is far more complex. They are used as a proof of concept to show
both the direct and indirect effect of features on the prediction output.


(a) chain dataset


Independent On-manifold ASV


(b) Shapley Flow


Figure 10: **(a)** The chain dataset contains exact copies of nodes. The dashed edges denotes dummy dependencies.
**(b)** While Shapley Flow shows the entire path of influence, other baselines fails to capture either direct and
indirect effects.


Methods Income Nutrition Synthetic


Independent **0.0** (± 0.0) **0.0** (± 0.0) **0.0** (± 0.0)
On-manifold 0.4 (± 0.3) 1.3 (± 2.5) 0.8 (± 0.7)
ASV 0.4 (± 0.6) 1.5 (± 3.3) 1.2 (± 1.4)
Shapley Flow **0.0** (± 0.0) **0.0** (± 0.0) **0.0** (± 0.0)


Table 2: Shapley Flow and independent SHAP have lower mean absolute error (std) for direct effect of features
on linear models.


Methods Income Nutrition Synthetic


Independent 0.1 (± 0.2) 0.8 (± 2.7) 1.1 (± 1.4)
On-manifold 0.4 (± 0.3) 0.9 (± 1.6) 1.5 (± 1.5)
ASV 0.1 (± 0.1) 0.6 (± 1.9) 1.1 (± 1.5)
Flow **0.0** (± 0.0) **0.0** (± 0.0) **0.0** (± 0.0)


Table 3: Shapley Flow and ASV have lower mean absolute error (std) for indirect effect on linear models.


the income dataset added in **Table 2** and **Table 3** for direct and indirect effects respectively. The trend for
the income dataset algins with the nutrition and synthetic dataset: only Shapley Flow makes no mistake for
estimating both direct and indirect impact. Independent Shap only does well for direct effect. ASV only does
well for indirect effects (it only reaches zero error when evaluated on source nodes).


**10.2** **Additional examples**


In this section, we analyze another example from the nutrition dataset ( **Figure 11** ) and 3 additional example
from the adult censor dataset.


**Independent SHAP ignores the indirect impact of features** . Take an example from the nutrition dataset
( **Figure 11** ). The race feature is given low attribution with independent SHAP, but high importance in ASV.
This happens because race, in addition to its direct impact, indirectly affects the output through blood pressure,
serum magnesium, and blood protein, as shown by Shapley Flow ( **Figure 11a** ). In particular, race partially
accounts for the impact of serum magnesium because changing race from Black to White on average increases
serum magnesium by 0 _._ 07 meg/L in the dataset (thus partially explaining the increase in serum magnesium
changing from the background sample to the foreground). Independent SHAP fails to account for the indirect
impact of race, leaving the user with a potentially misleading impression that race is irrelevant for the prediction.


**On-manifold SHAP provides a misleading interpretation** . With the same example ( **Figure 11** ), we
observe that on-manifold SHAP strongly disagrees with independent SHAP, ASV, and Shapley Flow on the
importance of age. Not only does it assign more credit to age, it also flips the sign, suggesting that age is
protective. However, **Figure 12a** shows that age and earlier mortality are positively correlated; then how could
age be protective? **Figure 12b** provides an explanation. Since SHAP considers all partial histories regardless
of the causal structure, when we focus on serum magnesium and age, there are two cases: serum magnesium
updates before or after age. We focus on the first case because it is where on-manifold SHAP differs from
other baselines (all baselines already consider the second case as it satisfies the causal ordering). When serum
magnesium updates before age, the expected age given serum magnesium is higher than the foreground age
(yellow line above the black marker). Therefore when age updates to its foreground value, we observe a decrease
in age, leading to a decrease in the output (so age appears to be protective). Serum magnesium is just one
variable from which age steals credit. Similar logic applies to TIBC, red blood cells, serum iron, serum protein,
serum cholesterol, and diastolic BP. From both an in/direct impact perspective, on-manifold perturbation can
be misleading since it is based not on causal but on observational relationships.


**ASV ignores the direct impact of features** . As shown in **Figure 11**, serum magnesium appears to be
more important in independent SHAP compared to ASV. From Shapley Flow ( **Figure 11a** ), this difference is
explained by race as its edge to serum magnesium has a negative impact. However, looking at ASV alone, one
fails to understand that intervening on serum magnesium could have a larger impact on the output.


**Shapley Flow shows both direct and indirect impacts of features** . Focusing on the attribution given
by Shapley Flow ( **Figure 11a** ). We not only observe similar direct impacts in variables compared to inde

**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


pendent SHAP, but also can trace those impacts to their source nodes, similar to ASV. Furthermore, Shapley
Flow provides more detail compared to other approaches. For example, using Shapley Flow we gain a better
understanding of the ways in which race impacts survival. The same goes for all other features. This is useful
because causal links can change (or break) over time. Our method provides a way to reason through the impact
of such a change.


**Figure 13** gives an example of applying Shapley Flow and baselines on the income dataset. Note that the
attribution to capital gain drops from independent SHAP to on-manifold SHAP and ASV. From Shapley Flow,
we know the decreased attribution is due to age and race. More examples are shown in **Figure 14** and **15** .


**10.3** **A global understanding with Shapley Flow**


In addition to explaining a particular example, one can explain an entire dataset with Shapley Flow. Specifically,
for multi-class classification problems, we take the average of attributions for the probability predicted for the
actual class, in accordance with (Frye et al., 2019). A demonstration on the income dataset using 1000 randomly
selected examples is included in **Figure 16** . As before, we use a single shared background sample for explanation.
Here, we observe that although the relative importance across independent SHAP, on-manifold SHAP, and ASV
are similar, age and sex have opposite direct versus indirect impact as shown by Shapley Flow.


**10.4** **Example with multiple background samples**


An example with 100 background samples is shown in **Figure 17** . Shapley Flow shows a holistic picture of
feature importance, while other baselines only show part of the picture.


**Independent SHAP ignores the indirect impact of features** . Take an example from the nutrition dataset
( **Figure 17** ). Independent SHAP only considers the direct impact of systolic blood pressure, and ignores its
potential impact on pulse pressure (as shown by Shapley Flow in **Figure 17a** ). If the causal graph is correct,
independent SHAP would underestimate the effect of intervening on Systolic BP.


**On-manifold SHAP provides a misleading interpretation** . With the same example ( **Figure 17** ), we
observe that on-manifold SHAP strongly disagrees with independent SHAP, ASV, and Shapley Flow on the
importance of age. In particular, it flips the sign on the importance of age. Since the background age (50) is
very close to the foreground age (51), we would not expected age to significantly affect the prediction. **Figure**
**18b** provides an explanation. Since SHAP considers all partial histories regardless of the causal structure, when
we focus on systolic blood pressure and age, there are two cases: systolic blood pressure updates before or after
age. We focus on the first case because it is where on-manifold SHAP differs from other baselines (all baselines
already consider the second case as it satisfies the causal ordering). When systolic blood pressure updates before
age, the expected age given systolic blood pressure is lower than the foreground age (yellow line below the black
marker). Therefore when age updates to its foreground value, we observe a large increase in age, leading to a
increase in the output (so age appears to be riskier). from both an in/direct impact perspective, on-manifold
perturbation can be misleading since it is based not on causal but on observational relationships.


**ASV ignores the direct impact of features** . As shown in **Figure 17**, ASV gives no credit systolic blood
pressure because it is an intermediate node. However, it is clear from Shapley Flow that intervening on systolic
blood pressure has a large impact on the outcome.


**Shapley Flow shows both direct and indirect impacts of features** . Focusing on the attribution given by
Shapley Flow ( **Figure 17a** ). We not only observe similar direct impacts in variables compared to independent
SHAP, but also can trace those impacts to their source nodes, similar to ASV.


**11** **Considering all histories could lead to boundary inconsistency**


In this section, we give an example of how considering all history H in the axioms (as opposed to H˜ )
could lead to inconsistent attributions across boundaries. Consider two cuts for the same causal graph
shown in **Figure 19** . Note that both the green and the red cut share the edge “a”. We have 8
possible message transmission histories (‘c’, ‘b’ can be transmitted only after ‘d’ has been transmitted):
{[ _𝑎, 𝑑, 𝑐, 𝑏_ ] _,_ [ _𝑎, 𝑑, 𝑏, 𝑐_ ] _,_ [ _𝑑, 𝑎, 𝑐, 𝑏_ ] _,_ [ _𝑑, 𝑎, 𝑏, 𝑐_ ] _,_ [ _𝑑, 𝑐, 𝑎, 𝑏_ ] _,_ [ _𝑑, 𝑐, 𝑏, 𝑎_ ] _,_ [ _𝑑, 𝑏, 𝑎, 𝑐_ ] _,_ [ _𝑑, 𝑏, 𝑐, 𝑎_ ]}. We use the same notation for carrier games (defined in **Section 8** ) and construct a game as the following:


Top features Age Serum Magnesium Race


Background sample 35.0 1.37 Black
Foreground sample 42.0 1.63 white


Attributions Independent On-manifold ASV

Iron 0.0 0.0 0.0


(a) Shapley Flow


Figure 11: Comparison among baselines on a sample (top table) from the nutrition dataset, showing top 10
features/edges. As noted in the main text this graph is an oversimplification and is not necessarily representative
of the true underlying causal relationship.


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


(a) Age vs. output (b) Age vs. magnesium


Figure 12: Age appears to be protective in on-manifold SHAP because it steals credit from other variables.


_𝑣_ _𝑟𝑒𝑑_ = _𝑣_ _𝑟𝑒𝑑_ _[𝑑𝑐𝑎]_ [−] _[𝑣]_ _𝑟𝑒𝑑_ _[𝑑𝑐𝑎𝑏]_ + _𝑣_ _𝑟𝑒𝑑_ _[𝑑𝑏𝑎]_ [−] _[𝑣]_ _𝑟𝑒𝑑_ _[𝑑𝑏𝑎𝑐]_

Because of the linearity axiom, we have _𝜙_ _𝑣_ _𝑟𝑒𝑑_ ( _𝑎_ ) _>_ 0 _, 𝜙_ _𝑣_ _𝑟𝑒𝑑_ ( _𝑏_ ) _<_ 0 _, 𝜙_ _𝑣_ _𝑟𝑒𝑑_ ( _𝑐_ ) _<_ 0 _, 𝜙_ _𝑣_ _𝑟𝑒𝑑_ ( _𝑑_ ) = 0.


However, when we consider the green boundary, the ordering _𝑑𝑐𝑎𝑏_ and _𝑑𝑏𝑎𝑐_ does not exist because in the green
boundary _𝐴_ and _𝑌_ are assumed to be a black-box. Therefore, _𝑣_ _𝑔𝑟𝑒𝑒𝑛_ = **0**, which means _𝑎_ is now a dummy edge:
_𝜙_ _𝑣_ _𝑔𝑟𝑒𝑒𝑛_ ( _𝑎_ ) = 0 ≠ _𝜙_ _𝑣_ _𝑟𝑒𝑑_ ( _𝑎_ ). This demonstrate that we cannot consider all histories in H and being boundary
consistent.


Background sample Foreground sample


Age 39 35
Workclass State-gov Federal-gov
Education-Num 13 5
Marital Status Never-married Married-civ-spouse
Occupation Adm-clerical Farming-fishing
Relationship Not-in-family Husband
Race White Black

Sex Male Male
Capital Gain 2174 0
Capital Loss 0 0
Hours per week 40 40
Country United-States United-States


Independent On-manifold ASV


(a) Shapley Flow


Figure 13: Comparison between independent SHAP, on-manifold SHAP, ASV, and Shapley Flow on a sample
from the income dataset. Shapley flow shows the top 10 links. The direct impact of capital gain is not represented by on-manifold SHAP. As noted in the text this graph is based on previous work and is not necessarily
representative of the true underlying causal relationship.


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


Background sample foreground sample


Age 39 30
Workclass State-gov State-gov
Education-Num 13 13
Marital Status Never-married Married-civ-spouse
Occupation Adm-clerical Prof-specialty
Relationship Not-in-family Husband
Race White Asian-Pac-Islander

Sex Male Male
Capital Gain 2174 0
Capital Loss 0 0
Hours per week 40 40
Country United-States India


Independent On-manifold ASV


(a) Shapley Flow


Figure 14: Comparison between independent SHAP, on-manifold SHAP, ASV, and Shapley Flow on a sample
from the income dataset. Shapley flow shows the top 10 links. The indirect impact of age is only highlighted
by Shapley Flow and ASV. As noted in the text this graph is based on previous work and is not necessarily
representative of the true underlying causal relationship.


Background sample Foreground sample


Age 39 30
Workclass State-gov Federal-gov
Education-Num 13 10
Marital Status Never-married Married-civ-spouse
Occupation Adm-clerical Adm-clerical
Relationship Not-in-family Own-child
Race White White

Sex Male Male
Capital Gain 2174 0
Capital Loss 0 0
Hours per week 40 40
Country United-States United-States


Attributions Independent On-manifold ASV


(a) Shapley Flow


Figure 15: Comparison between independent SHAP, on-manifold SHAP, ASV, and Shapley Flow on a sample
from the income dataset. Shapley flow shows the top 10 links. Note that although age appears to be not
important for all baselines, its impact through different causal edges are opposite as shown by Shapley Flow.


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**


Independent On-manifold ASV


(a) Shapley Flow


Figure 16: Comparison of global understanding between independent SHAP, on-manifold SHAP, ASV, and
Shapley Flow on the income dataset. Showing only the top 10 attributions for Shapley Flow for visual clarity.


Top features Sex Age Systolic BP


Background mean NaN 50 135
Foreground sample Female 51 118


Attributions Independent On-manifold ASV


(a) Shapley Flow


Figure 17: Comparison among methods on 100 background samples from the nutrition dataset, showing top 10
features/edges.


(a) Age vs. output (b) Age vs. systolic blood pressure


Figure 18: Age appears to be highly risky in on-manifold SHAP because it steals credit from other variables.


**Jiaxuan Wang, Jenna Wiens, Scott Lundberg**





(a) Red cut





(b) Green cut



Figure 19: Two cuts that represent two boundaries for the same causal graph.


