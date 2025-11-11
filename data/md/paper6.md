## **GNNExplainer: Generating Explanations** **for Graph Neural Networks**

Rex Ying _[†]_ Dylan Bourgeois _[†]_ [,] _[‡]_ Jiaxuan You _[†]_ Marinka Zitnik _[†]_ Jure Leskovec _[†]_


_†_ Department of Computer Science, Stanford University

_‡_ Robust.AI
_{_ `rexying, dtsbourg, jiaxuan, marinka, jure` _}_ `@cs.stanford.edu`


**Abstract**


Graph Neural Networks (GNNs) are a powerful tool for machine learning on
graphs. GNNs combine node feature information with the graph structure by
recursively passing neural messages along edges of the input graph. However, incorporating both graph structure and feature information leads to complex models
and explaining predictions made by GNNs remains unsolved. Here we propose
G NN E XPLAINER, the first general, model-agnostic approach for providing interpretable explanations for predictions of any GNN-based model on any graph-based
machine learning task. Given an instance, G NN E XPLAINER identifies a compact
subgraph structure and a small subset of node features that have a crucial role in
GNN’s prediction. Further, G NN E XPLAINER can generate consistent and concise
explanations for an entire class of instances. We formulate G NN E XPLAINER as an
optimization task that maximizes the mutual information between a GNN’s prediction and distribution of possible subgraph structures. Experiments on synthetic and
real-world graphs show that our approach can identify important graph structures
as well as node features, and outperforms alternative baseline approaches by up to
43.0% in explanation accuracy. G NN E XPLAINER provides a variety of benefits,
from the ability to visualize semantically relevant structures to interpretability, to
giving insights into errors of faulty GNNs.


**1** **Introduction**


In many real-world applications, including social, information, chemical, and biological domains,
data can be naturally modeled as graphs [ 9, 41, 49 ]. Graphs are powerful data representations but
are challenging to work with because they require modeling of rich relational information as well
as node feature information [ 45, 46 ]. To address this challenge, Graph Neural Networks (GNNs)
have emerged as state-of-the-art for machine learning on graphs, due to their ability to recursively
incorporate information from neighboring nodes in the graph, naturally capturing both graph structure
and node features [16, 21, 40, 44].


Despite their strengths, GNNs lack transparency as they do not easily allow for a human-intelligible
explanation of their predictions. Yet, the ability to understand GNN’s predictions is important and
useful for several reasons: (i) it can increase trust in the GNN model, (ii) it improves model’s
transparency in a growing number of decision-critical applications pertaining to fairness, privacy and
other safety challenges [ 11 ], and (iii) it allows practitioners to get an understanding of the network
characteristics, identify and correct systematic patterns of mistakes made by models before deploying
them in the real world.


While currently there are no methods for explaining GNNs, recent approaches for explaining other
types of neural networks have taken one of two main routes. One line of work locally approximates


33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.


GNN model training and predictions Explaning GNN’s predictions


GNNExplainer


Figure 1: G NN E XPLAINER provides interpretable explanations for predictions made by any GNN model on any
graph-based machine learning task. Shown is a hypothetical node classification task where a GNN model Φ is
trained on a social interaction graph to predict future sport activities. Given a trained GNN Φ and a prediction ˆ _y_ _i_
= “Basketball” for person _v_ _i_, G NN E XPLAINER generates an explanation by identifying a small subgraph of the
input graph together with a small subset of node features (shown on the right) that are most influential for ˆ _y_ _i_ .
Examining explanation for ˆ _y_ _i_, we see that many friends in one part of _v_ _i_ ’s social circle enjoy ball games, and so
the GNN predicts that _v_ _i_ will like basketball. Similarly, examining explanation for ˆ _y_ _j_, we see that _v_ _j_ ’s friends
and friends of his friends enjoy water and beach sports, and so the GNN predicts ˆ _y_ _j_ = “Sailing.”


models with simpler surrogate models, which are then probed for explanations [ 25, 29, 30 ]. Other
methods carefully examine models for relevant features and find good qualitative interpretations of
high level features [ 6, 13, 27, 32 ] or identify influential input instances [ 23, 38 ]. However, these
approaches fall short in their ability to incorporate relational information, the essence of graphs.
Since this aspect is crucial for the success of machine learning on graphs, any explanation of GNN’s
predictions should leverage rich relational information provided by the graph as well as node features.


Here we propose G NN E XPLAINER, an approach for explaining predictions made by GNNs. G NN E X PLAINER takes a trained GNN and its prediction(s), and it returns an explanation in the form of a small
subgraph of the input graph together with a small subset of node features that are most influential for
the prediction(s) (Figure 1). The approach is model-agnostic and can explain predictions of any GNN
on any machine learning task for graphs, including node classification, link prediction, and graph
classification. It handles single- as well as multi-instance explanations. In the case of single-instance
explanations, G NN E XPLAINER explains a GNN’s prediction for one particular instance ( _i.e._, a node
label, a new link, a graph-level label). In the case of multi-instance explanations, G NN E XPLAINER
provides an explanation that consistently explains a set of instances ( _e.g._, nodes of a given class).


G NN E XPLAINER specifies an explanation as a rich subgraph of the entire graph the GNN was
trained on, such that the subgraph maximizes the mutual information with GNN’s prediction(s).
This is achieved by formulating a mean field variational approximation and learning a real-valued
_graph mask_ which selects the important subgraph of the GNN’s computation graph. Simultaneously,
G NN E XPLAINER also learns a _feature mask_ that masks out unimportant node features (Figure 1).


We evaluate G NN E XPLAINER on synthetic as well as real-world graphs. Experiments show that
G NN E XPLAINER provides consistent and concise explanations of GNN’s predictions. On synthetic
graphs with planted network motifs, which play a role in determining node labels, we show that
G NN E XPLAINER accurately identifies the subgraphs/motifs as well as node features that determine
node labels outperforming alternative baseline approaches by up to 43.0% in explanation accuracy.
Further, using two real-world datasets we show how G NN E XPLAINER can provide important domain
insights by robustly identifying important graph structures and node features that influence a GNN’s
predictions. Specifically, using molecular graphs and social interaction networks, we show that
G NN E XPLAINER can identify important domain-specific graph structures, such as _NO_ 2 chemical
groups or ring structures in molecules, and star structures in Reddit threads. Overall, experiments
demonstrate that G NN E XPLAINER provides consistent and concise explanations for GNN-based
models for different machine learning tasks on graphs.


**2** **Related work**


Although the problem of explaining GNNs is not well-studied, the related problems of interpretability
and neural debugging received substantial attention in machine learning. At a high level, we can
group those interpretability methods for non-graph neural networks into two main families.


2


A B


**,** **,** **,** **...**


Figure 2: **A.** GNN computation graph _G_ _c_ (green and orange) for making prediction ˆ _y_ at node _v_ . Some edges
in _G_ _c_ form important neural message-passing pathways (green), which allow useful node information to be
propagated across _G_ _c_ and aggregated at _v_ for prediction, while other edges do not (orange). However, GNN
needs to aggregate important as well as unimportant messages to form a prediction at node _v_, which can dilute the
signal accumulated from _v_ ’s neighborhood. The goal of G NN E XPLAINER is to identify a small set of important
features and pathways (green) that are crucial for prediction. **B.** In addition to _G_ _S_ (green), G NN E XPLAINER
identifies what feature dimensions of _G_ _S_ ’s nodes are important for prediction by learning a node feature mask.


Methods in the first family formulate simple proxy models of full neural networks. This can be done
in a model-agnostic way, usually by learning a locally faithful approximation around the prediction,
for example through linear models [ 29 ] or sets of rules, representing sufficient conditions on the
prediction [ 3, 25, 47 ]. Methods in the second family identify important aspects of the computation, for
example, through feature gradients [ 13, 43 ], backpropagation of neurons’ contributions to the input
features [ 6, 31, 32 ], and counterfactual reasoning [ 19 ]. However, the saliency maps [ 43 ] produced
by these methods have been shown to be misleading in some instances [ 2 ] and prone to issues like
gradient saturation [ 31, 32 ]. These issues are exacerbated on discrete inputs such as graph adjacency
matrices since the gradient values can be very large but only on very small intervals. Because of that,
such approaches are not suitable for explaining predictions made by neural networks on graphs.


Instead of creating new, inherently interpretable models, post-hoc interpretability methods [ 1, 14, 15,
17, 23, 38 ] consider models as black boxes and then probe them for relevant information. However, no
work has been done to leverage relational structures like graphs. The lack of methods for explaining
predictions on graph-structured data is problematic, as in many cases, predictions on graphs are
induced by a complex combination of nodes and paths of edges between them. For example, in some
tasks, an edge is important only when another alternative path exists in the graph to form a cycle, and
those two features, only when considered together, can accurately predict node labels [ 10, 12 ]. Their
joint contribution thus cannot be modeled as a simple linear combinations of individual contributions.


Finally, recent GNN models augment interpretability via attention mechanisms [ 28, 33, 34 ]. However,
although the learned edge attention values can indicate important graph structure, the values are the
same for predictions across all nodes. Thus, this contradicts with many applications where an edge is
essential for predicting the label of one node but not the label of another node. Furthermore, these
approaches are either limited to specific GNN architectures or cannot explain predictions by jointly
considering both graph structure and node feature information.


**3** **Formulating explanations for graph neural networks**


Let _G_ denote a graph on edges _E_ and nodes _V_ that are associated with _d_ -dimensional node features
_X_ = _{x_ 1 _, . . ., x_ _n_ _}_, _x_ _i_ _∈_ R _[d]_ . Without loss of generality, we consider the problem of explaining a
node classification task (see Section 4.4 for other tasks). Let _f_ denote a label function on nodes
_f_ : _V �→{_ 1 _, . . ., C}_ that maps every node in _V_ to one of _C_ classes. The GNN model Φ is optimized
on all nodes in the training set and is then used for prediction, _i.e._, to approximate _f_ on new nodes.


**3.1** **Background on graph neural networks**


At layer _l_, the update of GNN model Φ involves three key computations [ 4, 45, 46 ]. (1) First, the
model computes neural messages between every pair of nodes. The message for node pair ( _v_ _i_ _, v_ _j_ ) is a
function M SG of _v_ _i_ ’s and _v_ _j_ ’s representations **h** _[l]_ _i_ _[−]_ [1] and **h** _[l]_ _j_ _[−]_ [1] in the previous layer and of the relation
_r_ _ij_ between the nodes: _m_ _[l]_ _ij_ [=][ M] [SG] [(] **[h]** _i_ _[l][−]_ [1] _,_ **h** _[l]_ _j_ _[−]_ [1] _, r_ _ij_ ) _._ (2) Second, for each node _v_ _i_, GNN aggregates


3


messages from _v_ _i_ ’s neighborhood _N_ _v_ _i_ and calculates an aggregated message _M_ _i_ via an aggregation
method A GG [ 16, 35 ]: _M_ _i_ _[l]_ [=][ A] [GG] [(] _[{][m]_ _[l]_ _ij_ _[|][v]_ _[j]_ _[ ∈N]_ _[v]_ _i_ _[}]_ [)] _[,]_ [ where] _[ N]_ _[v]_ _i_ [is neighborhood of node] _[ v]_ _[i]_ [ whose]
definition depends on a particular GNN variant. (3) Finally, GNN takes the aggregated message _M_ _i_ _[l]_
along with _v_ _i_ ’s representation **h** _[l]_ _i_ _[−]_ [1] from the previous layer, and it non-linearly transforms them to
obtain _v_ _i_ ’s representation **h** _[l]_ _i_ [at layer] _[ l]_ [:] **[ h]** _[l]_ _i_ [=][ U] [PDATE] [(] _[M]_ _[ l]_ _i_ _[,]_ **[ h]** _[l]_ _i_ _[−]_ [1] ) _._ The final embedding for node _v_ _i_
after _L_ layers of computation is **z** _i_ = **h** _[L]_ _i_ [. Our G] [NN] [E] [XPLAINER] [ provides explanations for any GNN]
that can be formulated in terms of M SG, A GG, and U PDATE computations.


**3.2** **G** **NN** **E** **XPLAINER** **: Problem formulation**


Our key insight is the observation that the computation graph of node _v_, which is defined by the
GNN’s neighborhood-based aggregation (Figure 2), fully determines all the information the GNN
uses to generate prediction ˆ _y_ at node _v_ . In particular, _v_ ’s computation graph tells the GNN how to
generate _v_ ’s embedding **z** . Let us denote that computation graph by _G_ _c_ ( _v_ ), the associated binary
adjacency matrix by _A_ _c_ ( _v_ ) _∈{_ 0 _,_ 1 _}_ _[n][×][n]_, and the associated feature set by _X_ _c_ ( _v_ ) = _{x_ _j_ _|v_ _j_ _∈_ _G_ _c_ ( _v_ ) _}_ .
The GNN model Φ learns a conditional distribution _P_ Φ ( _Y |G_ _c_ _, X_ _c_ ), where _Y_ is a random variable
representing labels _{_ 1 _, . . ., C}_, indicating the probability of nodes belonging to each of _C_ classes.


A GNN’s prediction is given by ˆ _y_ = Φ( _G_ _c_ ( _v_ ) _, X_ _c_ ( _v_ )), meaning that it is fully determined by the
model Φ, graph structural information _G_ _c_ ( _v_ ), and node feature information _X_ _c_ ( _v_ ) . In effect, this
observation implies that we only need to consider graph structure _G_ _c_ ( _v_ ) and node features _X_ _c_ ( _v_ )
to explain ˆ _y_ (Figure 2A). Formally, G NN E XPLAINER generates explanation for prediction ˆ _y_ as
( _G_ _S_ _, X_ _S_ _[F]_ [)] [, where] _[ G]_ _[S]_ [ is a small subgraph of the computation graph.] _[ X]_ _[S]_ [ is the associated feature of]
_G_ _S_, and _X_ _S_ _[F]_ [is a small subset of node features (masked out by the mask] _[ F]_ [,] _[ i.e.]_ [,] _[ X]_ _S_ _[F]_ [=] _[ {][x]_ _[F]_ _j_ _[|][v]_ _[j]_ _[ ∈]_
_G_ _S_ _}_ ) that are most important for explaining ˆ _y_ (Figure 2B).


**4** **G** **NN** **E** **XPLAINER**


Next we describe our approach G NN E XPLAINER . Given a trained GNN model Φ and a prediction
( _i.e._, single-instance explanation, Sections 4.1 and 4.2) or a set of predictions ( _i.e._, multi-instance
explanations, Section 4.3), the G NN E XPLAINER will generate an explanation by identifying a
subgraph of the computation graph and a subset of node features that are most influential for the
model Φ ’s prediction. In the case of explaining a set of predictions, G NN E XPLAINER will aggregate
individual explanations in the set and automatically summarize it with a prototype. We conclude this
section with a discussion on how G NN E XPLAINER can be used for any machine learning task on
graphs, including link prediction and graph classification (Section 4.4).


**4.1** **Single-instance explanations**


Given a node _v_, our goal is to identify a subgraph _G_ _S_ _⊆_ _G_ _c_ and the associated features _X_ _S_ =
_{x_ _j_ _|v_ _j_ _∈_ _G_ _S_ _}_ that are important for the GNN’s prediction ˆ _y_ . For now, we assume that _X_ _S_ is a
small subset of _d_ -dimensional node features; we will later discuss how to automatically determine
which dimensions of node features need to be included in explanations (Section 4.2). We formalize
the notion of importance using mutual information _MI_ and formulate the G NN E XPLAINER as the
following optimization framework:


max (1)
_G_ _S_ _[MI]_ [ (] _[Y,]_ [ (] _[G]_ _[S]_ _[, X]_ _[S]_ [)) =] _[ H]_ [(] _[Y]_ [ )] _[ −]_ _[H]_ [(] _[Y][ |][G]_ [ =] _[ G]_ _[S]_ _[, X]_ [ =] _[ X]_ _[S]_ [)] _[.]_


For node _v_, _MI_ quantifies the change in the probability of prediction ˆ _y_ = Φ( _G_ _c_ _, X_ _c_ ) when _v_ ’s
computation graph is limited to explanation subgraph _G_ _S_ and its node features are limited to _X_ _S_ .


For example, consider the situation where _v_ _j_ _∈_ _G_ _c_ ( _v_ _i_ ) _, v_ _j_ _̸_ = _v_ _i_ . Then, if removing _v_ _j_ from _G_ _c_ ( _v_ _i_ )
strongly decreases the probability of prediction ˆ _y_ _i_, the node _v_ _j_ is a good counterfactual explanation
for the prediction at _v_ _i_ . Similarly, consider the situation where ( _v_ _j_ _, v_ _k_ ) _∈_ _G_ _c_ ( _v_ _i_ ) _, v_ _j_ _, v_ _k_ _̸_ = _v_ _i_ . Then,
if removing an edge between _v_ _j_ and _v_ _k_ strongly decreases the probability of prediction ˆ _y_ _i_ then the
absence of that edge is a good counterfactual explanation for the prediction at _v_ _i_ .


Examining Eq. (1), we see that the entropy term _H_ ( _Y_ ) is constant because Φ is fixed for a trained
GNN. As a result, maximizing mutual information between the predicted label distribution _Y_ and


4


explanation ( _G_ _S_ _, X_ _S_ ) is equivalent to minimizing conditional entropy _H_ ( _Y |G_ = _G_ _S_ _, X_ = _X_ _S_ ),
which can be expressed as follows:


_H_ ( _Y |G_ = _G_ _S_ _, X_ = _X_ _S_ ) = _−_ E _Y |G_ _S_ _,X_ _S_ [log _P_ Φ ( _Y |G_ = _G_ _S_ _, X_ = _X_ _S_ )] _._ (2)


Explanation for prediction ˆ _y_ is thus a subgraph _G_ _S_ that minimizes uncertainty of Φ when the GNN
computation is limited to _G_ _S_ . In effect, _G_ _S_ maximizes probability of ˆ _y_ (Figure 2). To obtain a
compact explanation, we impose a constraint on _G_ _S_ ’s size as: _|G_ _S_ _| ≤_ _K_ _M_ _,_ so that _G_ _S_ has at most
_K_ _M_ nodes. In effect, this implies that G NN E XPLAINER aims to denoise _G_ _c_ by taking _K_ _M_ edges
that give the highest mutual information with the prediction.
**G** **NN** **E** **XPLAINER** **’s optimization framework.** Direct optimization of G NN E XPLAINER ’s objective
is not tractable as _G_ _c_ has exponentially many subgraphs _G_ _S_ that are candidate explanations for ˆ _y_ . We
thus consider a fractional adjacency matrix [1] for subgraphs _G_ _S_, _i.e._, _A_ _S_ _∈_ [0 _,_ 1] _[n][×][n]_, and enforce the
subgraph constraint as: _A_ _S_ [ _j, k_ ] _≤_ _A_ _c_ [ _j, k_ ] for all _j, k_ . This continuous relaxation can be interpreted
as a variational approximation of distribution of subgraphs of _G_ _c_ . In particular, if we treat _G_ _S_ _∼G_
as a random graph variable, the objective in Eq. (2) becomes:


min _G_ [E] _[G]_ _[S]_ _[∼G]_ _[H]_ [(] _[Y][ |][G]_ [ =] _[ G]_ _[S]_ _[, X]_ [ =] _[ X]_ _[S]_ [)] _[,]_ (3)


With convexity assumption, Jensen’s inequality gives the following upper bound:


min _G_ _[H]_ [(] _[Y][ |][G]_ [ =][ E] _[G]_ [[] _[G]_ _[S]_ []] _[, X]_ [ =] _[ X]_ _[S]_ [)] _[.]_ (4)


In practice, due to the complexity of neural networks, the convexity assumption does not hold.
However, experimentally, we found that minimizing this objective with regularization often leads to a
local minimum corresponding to high-quality explanations.


To tractably estimate E _G_, we use mean-field variational approximation and decompose _G_ into a
multivariate Bernoulli distribution as: _P_ _G_ ( _G_ _S_ ) = [�] ( _j,k_ ) _∈G_ _c_ _[A]_ _[S]_ [[] _[j, k]_ []] [. This allows us to estimate the]
expectation with respect to the mean-field approximation, thereby obtaining _A_ _S_ in which ( _j, k_ ) -th
entry represents the expectation on whether edge ( _v_ _j_ _, v_ _k_ ) exists. We observed empirically that this
approximation together with a regularizer for promoting discreteness [ 40 ] converges to good local
minima despite the non-convexity of GNNs. The conditional entropy in Equation 4 can be optimized
by replacing the E _G_ [ _G_ _S_ ] to be optimized by a masking of the computation graph of adjacency matrix,
_A_ _c_ _⊙_ _σ_ ( _M_ ), where _M ∈_ R _[n][×][n]_ denotes the mask that we need to learn, _⊙_ denotes element-wise
multiplication, and _σ_ denotes the sigmoid that maps the mask to [0 _,_ 1] _[n][×][n]_ .


In some applications, instead of finding an explanation in terms of model’s confidence, the users care
more about “why does the trained model predict a certain class label”, or “how to make the trained
model predict a desired class label”. We can modify the conditional entropy objective in Equation 4
with a cross entropy objective between the label class and the model prediction [2] . To answer these
queries, a computationally efficient version of G NN E XPLAINER ’s objective, which we optimize using
gradient descent, is as follows:



min
_M_ _[−]_



_C_
� 1 [ _y_ = _c_ ] log _P_ Φ ( _Y_ = _y|G_ = _A_ _c_ _⊙_ _σ_ ( _M_ ) _, X_ = _X_ _c_ ) _,_ (5)


_c_ =1



The masking approach is also found in Neural Relational Inference [ 22 ], albeit with different
motivation and objective. Lastly, we compute the element-wise multiplication of _σ_ ( _M_ ) and _A_ _c_ and
remove low values in _M_ through thresholding to arrive at the explanation _G_ _S_ for the GNN model’s
prediction ˆ _y_ at node _v_ .


**4.2** **Joint learning of graph structural and node feature information**


To identify what node features are most important for prediction ˆ _y_, G NN E XPLAINER learns a feature
selector _F_ for nodes in explanation _G_ _S_ . Instead of defining _X_ _S_ to consists of all node features, _i.e._,


1 For typed edges, we define _G_ _S_ _∈_ [0 _,_ 1] _C_ _e_ _×n×n_ where _C_ _e_ is the number of edge types.
2 The label class is the predicted label class by the GNN model to be explained, when answering “why does
the trained model predict a certain class label”. “how to make the trained model predict a desired class label”
can be answered by using the ground-truth label class.


5


_X_ _S_ = _{x_ _j_ _|v_ _j_ _∈_ _G_ _S_ _}_, G NN E XPLAINER considers _X_ _S_ _[F]_ [as a subset of features of nodes in] _[ G]_ _[S]_ [, which]
are defined through a binary feature selector _F ∈{_ 0 _,_ 1 _}_ _[d]_ (Figure 2B):

_X_ _S_ _[F]_ [=] _[ {][x]_ _[F]_ _j_ _[|][v]_ _[j]_ _[∈]_ _[G]_ _[S]_ _[}][,]_ _x_ _[F]_ _j_ [= [] _[x]_ _[j,t]_ 1 _[, . . ., x]_ _[j,t]_ _k_ []][ for] _[ F]_ _[t]_ _i_ [= 1] _[,]_ (6)

where _x_ _[F]_ _j_ [has node features that are not masked out by] _[ F]_ [. Explanation] [ (] _[G]_ _[S]_ _[, X]_ _[S]_ [)] [ is then jointly]
optimized for maximizing the mutual information objective:


max _S_ [)] _[,]_ (7)
_G_ _S_ _,F_ _[MI]_ [ (] _[Y,]_ [ (] _[G]_ _[S]_ _[, F]_ [)) =] _[ H]_ [(] _[Y]_ [ )] _[ −]_ _[H]_ [(] _[Y][ |][G]_ [ =] _[ G]_ _[S]_ _[, X]_ [ =] _[ X]_ _[F]_


which represents a modified objective function from Eq. (1) that considers structural and node feature
information to generate an explanation for prediction ˆ _y_ .
**Learning binary feature selector** _F_ **.** We specify _X_ _S_ _[F]_ [as] _[ X]_ _[S]_ _[ ⊙]_ _[F]_ [, where] _[ F]_ [ acts as a feature mask]
that we need to learn. Intuitively, if a particular feature is not important, the corresponding weights in
GNN’s weight matrix take values close to zero. In effect, this implies that masking the feature out
does not decrease predicted probability for ˆ _y._ Conversely, if the feature is important then masking it
out would decrease predicted probability. However, in some cases this approach ignores features that
are important for prediction but take values close to zero. To address this issue we marginalize over
all feature subsets and use a Monte Carlo estimate to sample from empirical marginal distribution for
nodes in _X_ _S_ during training [ 48 ]. Further, we use a reparametrization trick [ 20 ] to backpropagate
gradients in Eq. (7) to the feature mask _F_ . In particular, to backpropagate through a _d_ -dimensional
random variable _X_ we reparametrize _X_ as: _X_ = _Z_ + ( _X_ _S_ _−_ _Z_ ) _⊙_ _F_ s.t. [�] _j_ _[F]_ _[j]_ _[ ≤]_ _[K]_ _[F]_ [, where] _[ Z]_

is a _d_ -dimensional random variable sampled from the empirical distribution and _K_ _F_ is a parameter
representing the maximum number of features to be kept in the explanation.
**Integrating additional constraints into explanations.** To impose further properties on the explanation we can extend G NN E XPLAINER ’s objective function in Eq. (7) with regularization terms. For
example, we use element-wise entropy to encourage structural and node feature masks to be discrete.
Further, G NN E XPLAINER can encode domain-specific constraints through techniques like Lagrange
multiplier of constraints or additional regularization terms. We include a number of regularization
terms to produce explanations with desired properties. We penalize large size of the explanation by
adding the sum of all elements of the mask paramters as the regularization term.


Finally, it is important to note that each explanation must be a valid computation graph. In particular,
explanation ( _G_ _S_ _, X_ _S_ ) needs to allow GNN’s neural messages to flow towards node _v_ such that
GNN can make prediction ˆ _y_ . Importantly, G NN E XPLAINER automatically provides explanations that
represent valid computation graphs because it optimizes structural masks across entire computation
graphs. Even if a disconnected edge is important for neural message-passing, it will not be selected
for explanation as it cannot influence GNN’s prediction. In effect, this implies that the explanation
_G_ _S_ tends to be a small connected subgraph.


**4.3** **Multi-instance explanations through graph prototypes**


The output of a single-instance explanation (Sections 4.1 and 4.2) is a small subgraph of the input
graph and a small subset of associated node features that are most influential for a single prediction.
To answer questions like “How did a GNN predict that a given set of nodes all have label _c_ ?”, we
need to obtain a global explanation of class _c_ . Our goal here is to provide insight into how the
identified subgraph for a particular node relates to a graph structure that explains an entire class.
G NN E XPLAINER can provide multi-instance explanations based on graph alignments and prototypes.
Our approach has two stages:


First, for a given class _c_ (or, any set of predictions that we want to explain), we first choose a
reference node _v_ _c_, for example, by computing the mean embedding of all nodes assigned to _c_ . We
then take explanation _G_ _S_ ( _v_ _c_ ) for reference _v_ _c_ and align it to explanations of other nodes assigned to
class _c_ . Finding optimal matching of large graphs is challenging in practice. However, the singleinstance G NN E XPLAINER generates small graphs (Section 4.2) and thus near-optimal pairwise graph
matchings can be efficiently computed.


Second, we aggregate aligned adjacency matrices into a graph prototype _A_ proto using, for example, a
robust median-based approach. Prototype _A_ proto gives insights into graph patterns shared between
nodes that belong to the same class. One can then study prediction for a particular node by comparing
explanation for that node’s prediction ( _i.e._, returned by single-instance explanation approach) to the
prototype (see Appendix for more information).


6


**4.4** **G** **NN** **E** **XPLAINER** **model extensions**


**Any machine learning task on graphs.** In addition to explaining node classification, G NN E X PLAINER provides explanations for link prediction and graph classification with no change to its
optimization algorithm. When predicting a link ( _v_ _j_ _, v_ _k_ ), G NN E XPLAINER learns two masks _X_ _S_ ( _v_ _j_ )
and _X_ _S_ ( _v_ _k_ ) for both endpoints of the link. When classifying a graph, the adjacency matrix in Eq. (5)
is the union of adjacency matrices for all nodes in the graph whose label we want to explain. However,
note that in graph classification, unlike node classification, due to the aggregation of node embeddings, it is no longer true that the explanation _G_ _S_ is necessarily a connected subgraph. Depending on
application, in some scenarios such as chemistry where explanation is a functional group and should
be connected, one can extract the largest connected component as the explanation.
**Any GNN model.** Modern GNNs are based on message passing architectures on the input graph. The
message passing computation graphs can be composed in many different ways and G NN E XPLAINER
can account for all of them. Thus, G NN E XPLAINER can be applied to: Graph Convolutional
Networks [ 21 ], Gated Graph Sequence Neural Networks [ 26 ], Jumping Knowledge Networks [ 36 ],
Attention Networks [ 33 ], Graph Networks [ 4 ], GNNs with various node aggregation schemes [ 7, 5, 18,
16, 40, 39, 35], Line-Graph NNs [8], position-aware GNN [42], and many other GNN architectures.
**Computational complexity.** The number of parameters in G NN E XPLAINER ’s optimization depends
on the size of computation graph _G_ _c_ for node _v_ whose prediction we aim to explain. In particular,
_G_ _c_ ( _v_ ) ’s adjacency matrix _A_ _c_ ( _v_ ) is equal to the size of the mask _M_, which needs to be learned
by G NN E XPLAINER . However, since computation graphs are typically relatively small, compared
to the size of exhaustive _L_ -hop neighborhoods ( _e.g._, 2-3 hop neighborhoods [ 21 ], sampling-based
neighborhoods [ 39 ], neighborhoods with attention [ 33 ]), G NN E XPLAINER can effectively generate
explanations even when input graphs are large.


**5** **Experiments**


We begin by describing the graphs, alternative baseline approaches, and experimental setup. We then
present experiments on explaining GNNs for node classification and graph classification tasks. Our
qualitative and quantitative analysis demonstrates that G NN E XPLAINER is accurate and effective in
identifying explanations, both in terms of graph structure and node features.
**Synthetic datasets.** We construct four kinds of node classification datasets (Table 1). (1) In BAS HAPES, we start with a base Barabasi-Albert (BA) graph on 300 nodes and a set of 80 five-node ´
“house”-structured network motifs, which are attached to randomly selected nodes of the base graph.
The resulting graph is further perturbed by adding 0 _._ 1 _N_ random edges. Nodes are assigned to 4
classes based on their structural roles. In a house-structured motif, there are 3 types of roles: the top,
middle and bottom node of the house. Therefore there are 4 different classes, corresponding to nodes
at the top, middle, bottom of houses, and nodes that do not belong to a house. (2) BA-C OMMUNITY
dataset is a union of two BA-S HAPES graphs. Nodes have normally distributed feature vectors and
are assigned to one of 8 classes based on their structural roles and community memberships. (3)
In T REE -C YCLES, we start with a base 8-level balanced binary tree and 80 six-node cycle motifs,
which are attached to random nodes of the base graph. (4) T REE -G RID is the same as T REE -C YCLES
except that 3-by-3 grid motifs are attached to the base tree graph in place of cycle motifs.

**Real-world datasets.** We consider two graph classification datasets: (1) M UTAG is a dataset of
4 _,_ 337 molecule graphs labeled according to their mutagenic effect on the Gram-negative bacterium _S._
_typhimurium_ [ 10 ]. (2) R EDDIT -B INARY is a dataset of 2 _,_ 000 graphs, each representing an online
discussion thread on Reddit. In each graph, nodes are users participating in a thread, and edges
indicate that one user replied to another user’s comment. Graphs are labeled according to the type of
user interactions in the thread: _r/IAmA_ and _r/AskReddit_ contain Question-Answer interactions, while
_r/TrollXChromosomes_ and _r/atheism_ contain Online-Discussion interactions [37].

**Alternative baseline approaches.** Many explainability methods cannot be directly applied to graphs
(Section 2). Nevertheless, we here consider the following alternative approaches that can provide
insights into predictions made by GNNs: (1) G RAD is a gradient-based method. We compute gradient
of the GNN’s loss function with respect to the adjacency matrix and the associated node features,
similar to a saliency map approach. (2) A TT is a graph attention GNN (GAT) [ 33 ] that learns attention
weights for edges in the computation graph, which we use as a proxy measure of edge importance.
While A TT does consider graph structure, it does not explain using node features and can only explain
GAT models. Furthermore, in A TT it is not obvious which attention weights need to be used for edge


7


**BA-Shapes**



**BA-Community** **Tree-Cycles**


**Community 0** **Community 1**



**Tree-Grid**


None



**Base**



**Motif**


**Node Features** None where  = community ID None



**Explanation** Graph structure
**content** Graph structure Node feature information

**Explanation accuracy**



Graph structure Graph structure



0.612


0.667

**0.875**



Att


Grad


G NN Explainer



0.815 0.739



0.882

**0.925**



0.750

**0.836**



0.824


0.905

**0.948**



Table 1: Illustration of synthetic datasets (refer to “Synthetic datasets” for details) together with performance
evaluation of G NN E XPLAINER and alternative baseline explainability approaches.


**A** **B**

**Computation graph** **GNNExplainer** **Grad** **Att** **Ground Truth** **Computation graph** **GNNExplainer** **Grad** **Att** **Ground Truth**


Figure 3: Evaluation of single-instance explanations. **A-B.** Shown are exemplar explanation subgraphs for node
classification task on four synthetic datasets. Each method provides explanation for the red node’s prediction.


importance, since a 1-hop neighbor of a node can also be a 2-hop neighbor of the same node due to
cycles. Each edge’s importance is thus computed as the average attention weight across all layers.
**Setup and implementation details.** For each dataset, we first train a single GNN for each dataset,
and use G RAD and G NN E XPLAINER to explain the predictions made by the GNN. Note that
the A TT baseline requires using a graph attention architecture like GAT [ 33 ]. We thus train a
separate GAT model on the same dataset and use the learned edge attention weights for explanation.
Hyperparameters _K_ _M_ _, K_ _F_ control the size of subgraph and feature explanations respectively, which
is informed by prior knowledge about the dataset. For synthetic datasets, we set _K_ _M_ to be the
size of ground truth. On real-world datasets, we set _K_ _M_ = 10 . We set _K_ _F_ = 5 for all datasets.
We further fix our weight regularization hyperparameters across all node and graph classification
experiments. We refer readers to the Appendix for more training details (Code and datasets are
available at https://github.com/RexYing/gnn-model-explainer).

**Results.** We investigate questions: Does G NN E XPLAINER provide sensible explanations? How
do explanations compare to the ground-truth knowledge? How does G NN E XPLAINER perform on
various graph-based prediction tasks? Can it explain predictions made by different GNNs?
**1) Quantitative analyses.** Results on node classification datasets are shown in Table 1. We have
ground-truth explanations for synthetic datasets and we use them to calculate explanation accuracy for
all explanation methods. Specifically, we formalize the explanation problem as a binary classification
task, where edges in the ground-truth explanation are treated as labels and importance weights given
by explainability method are viewed as prediction scores. A better explainability method predicts


**A** **B**
**Computation graph** **GNNExplainer** **Grad** **Att** **Ground Truth** **Computation graph** **GNNExplainer** **Grad** **Att** **Ground Truth**


Ring
structure


NO group 2



Topic
reactions



Experts answering
multiple questions



Figure 4: Evaluation of single-instance explanations. **A-B.** Shown are exemplar explanation subgraphs for graph
classification task on two datasets, M UTAG and R EDDIT -B INARY .


8


A Graph classification B Node classification


Input to nodefeatures node featureswith node
GNN


GNN’s
Prediction Molecule’s mutagenicity Node’s structural role



Figure 5: Visualization of features that are important
for a GNN’s prediction. **A.** Shown is a representative
molecular graph from M UTAG dataset (top). Importance of the associated graph features is visualized
with a heatmap (bottom). In contrast with baselines,
G NN E XPLAINER correctly identifies features that are
important for predicting the molecule’s mutagenicity,
_i.e._ C, O, H, and N atoms. **B.** Shown is a computation
graph of a red node from BA-C OMMUNITY dataset
(top). Again, G NN E XPLAINER successfully identifies
the node feature that is important for predicting the
structural role of the node but baseline methods fail.



Att



Not applicable Not applicable



high scores for edges that are in the ground-truth explanation, and thus achieves higher explanation
accuracy. Results show that G NN E XPLAINER outperforms alternative approaches by 17.1% on
average. Further, G NN E XPLAINER achieves up to 43.0% higher accuracy on the hardest T REE -G RID
dataset.

**2) Qualitative analyses.** Results are shown in Figures 3–5. In a topology-based prediction task with
no node features, _e.g._ BA-S HAPES and T REE -C YCLES, G NN E XPLAINER correctly identifies network
motifs that explain node labels, _i.e._ structural labels (Figure 3). As illustrated in the figures, house,
cycle and tree motifs are identified by G NN E XPLAINER but not by baseline methods. In Figure 4,
we investigate explanations for graph classification task. In M UTAG example, colors indicate node
features, which represent atoms (hydrogen H, carbon C, _etc_ ). G NN E XPLAINER correctly identifies
carbon ring as well as chemical groups _NH_ 2 and _NO_ 2, which are known to be mutagenic [10].


Further, in R EDDIT -B INARY example, we see that Question-Answer graphs (2nd row in Figure 4B)
have 2-3 high degree nodes that simultaneously connect to many low degree nodes, which makes
sense because in QA threads on Reddit we typically have 2-3 experts who all answer many different
questions [ 24 ]. Conversely, we observe that discussion patterns commonly exhibit tree-like patterns
(2nd row in Figure 4A), since a thread on Reddit is usually a reaction to a single topic [ 24 ]. On the
other hand, G RAD and A TT methods give incorrect or incomplete explanations. For example, both
baseline methods miss cycle motifs in M UTAG dataset and more complex grid motifs in T REE -G RID
dataset. Furthermore, although edge attention weights in A TT can be interpreted as importance scores
for message passing, the weights are shared across all nodes in input the graph, and as such A TT fails
to provide high quality single-instance explanations.


An essential criterion for explanations is that they must be interpretable, _i.e._, provide a qualitative
understanding of the relationship between the input nodes and the prediction. Such a requirement
implies that explanations should be easy to understand while remaining exhaustive. This means
that a GNN explainer should take into account both the structure of the underlying graph as well as
the associated features when they are available. Figure 5 shows results of an experiment in which
G NN E XPLAINER jointly considers structural information as well as information from a small number
of feature dimensions [3] . While G NN E XPLAINER indeed highlights a compact feature representation
in Figure 5, gradient-based approaches struggle to cope with the added noise, giving high importance
scores to irrelevant feature dimensions.


Further experiments on multi-instance explanations using graph prototypes are in Appendix.


**6** **Conclusion**


We present G NN E XPLAINER, a novel method for explaining predictions of any GNN on any graphbased machine learning task without requiring modification of the underlying GNN architecture or
re-training. We show how G NN E XPLAINER can leverage recursive neighborhood-aggregation scheme
of graph neural networks to identify important graph pathways as well as highlight relevant node
feature information that is passed along edges of the pathways. While the problem of explainability of
machine-learning predictions has received substantial attention in recent literature, our work is unique
in the sense that it presents an approach that operates on relational structures—graphs with rich


3 Feature explanations are shown for the two datasets with node features, _i.e._, M UTAG and BA-C OMMUNITY .


9


node features—and provides a straightforward interface for making sense out of GNN predictions,
debugging GNN models, and identifying systematic patterns of mistakes.


**Acknowledgments**


Jure Leskovec is a Chan Zuckerberg Biohub investigator. We gratefully acknowledge the support
of DARPA under FA865018C7880 (ASED) and MSC; NIH under No. U54EB020405 (Mobilize);
ARO under No. 38796-Z8424103 (MURI); IARPA under No. 2017-17071900005 (HFC), NSF
under No. OAC-1835598 (CINES) and HDR; Stanford Data Science Initiative, Chan Zuckerberg
Biohub, JD.com, Amazon, Boeing, Docomo, Huawei, Hitachi, Observe, Siemens, UST Global.
The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes
notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the
views, policies, or endorsements, either expressed or implied, of DARPA, NIH, ONR, or the U.S.
Government.


**References**


[1] A. Adadi and M. Berrada. Peeking Inside the Black-Box: A Survey on Explainable Artificial
Intelligence (XAI). _IEEE Access_, 6:52138–52160, 2018.


[2] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim. Sanity checks for
saliency maps. In _NeurIPS_, 2018.


[3] M. Gethsiyal Augasta and T. Kathirvalavakumar. Reverse Engineering the Neural Networks for
Rule Extraction in Classification Problems. _Neural Processing Letters_, 35(2):131–150, April
2012.


[4] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner,
et al. Relational inductive biases, deep learning, and graph networks. _arXiv:1806.01261_, 2018.


[5] J. Chen, J. Zhu, and L. Song. Stochastic training of graph convolutional networks with variance
reduction. In _ICML_, 2018.


[6] Jianbo Chen, Le Song, Martin J Wainwright, and Michael I Jordan. Learning to explain: An
information-theoretic perspective on model interpretation. _arXiv preprint arXiv:1802.07814_,
2018.


[7] Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: fast learning with graph convolutional networks
via importance sampling. In _ICLR_, 2018.


[8] Z. Chen, L. Li, and J. Bruna. Supervised community detection with line graph neural networks.
In _ICLR_, 2019.


[9] E. Cho, S. Myers, and J. Leskovec. Friendship and mobility: user movement in location-based
social networks. In _KDD_, 2011.


[10] A. Debnath et al. Structure-activity relationship of mutagenic aromatic and heteroaromatic
nitro compounds. correlation with molecular orbital energies and hydrophobicity. _Journal of_
_Medicinal Chemistry_, 34(2):786–797, 1991.


[11] F. Doshi-Velez and B. Kim. Towards A Rigorous Science of Interpretable Machine Learning.
2017. arXiv: 1702.08608.


[12] D. Duvenaud et al. Convolutional networks on graphs for learning molecular fingerprints. In
_NIPS_, 2015.


[13] D. Erhan, Y. Bengio, A. Courville, and P. Vincent. Visualizing higher-layer features of a deep
network. _University of Montreal_, 1341(3):1, 2009.


[14] A. Fisher, C. Rudin, and F. Dominici. All Models are Wrong but many are Useful: Variable
Importance for Black-Box, Proprietary, or Misspecified Prediction Models, using Model Class
Reliance. January 2018. arXiv: 1801.01489.


[15] R. Guidotti et al. A Survey of Methods for Explaining Black Box Models. _ACM Comput. Surv._,
51(5):93:1–93:42, 2018.


10


[16] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In
_NIPS_, 2017.


[17] G. Hooker. Discovering additive structure in black box functions. In _KDD_, 2004.


[18] W.B. Huang, T. Zhang, Y. Rong, and J. Huang. Adaptive sampling towards fast graph representation learning. In _NeurIPS_, 2018.


[19] Bo Kang, Jefrey Lijffijt, and Tijl De Bie. Explaine: An approach for explaining network
embedding-based link predictions. _arXiv:1904.12694_, 2019.


[20] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In _NeurIPS_, 2013.


[21] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks.
In _ICLR_, 2016.


[22] Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural
relational inference for interacting systems. In _ICML_, 2018.


[23] P. W. Koh and P. Liang. Understanding black-box predictions via influence functions. In _ICML_,
2017.


[24] Srijan Kumar, William L Hamilton, Jure Leskovec, and Dan Jurafsky. Community interaction
and conflict on the web. In _WWW_, pages 933–943, 2018.


[25] H. Lakkaraju, E. Kamar, R. Caruana, and J. Leskovec. Interpretable & Explorable Approximations of Black Box Models, 2017.


[26] Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel. Gated graph sequence neural networks.
_arXiv:1511.05493_, 2015.


[27] S. Lundberg and Su-In Lee. A Unified Approach to Interpreting Model Predictions. In _NIPS_,
2017.


[28] D. Neil et al. Interpretable Graph Convolutional Neural Networks for Inference on Noisy
Knowledge Graphs. In _ML4H Workshop at NeurIPS_, 2018.


[29] M. Ribeiro, S. Singh, and C. Guestrin. Why should i trust you?: Explaining the predictions of
any classifier. In _KDD_, 2016.


[30] G. J. Schmitz, C. Aldrich, and F. S. Gouws. ANN-DT: an algorithm for extraction of decision
trees from artificial neural networks. _IEEE Transactions on Neural Networks_, 1999.


[31] A. Shrikumar, P. Greenside, and A. Kundaje. Learning Important Features Through Propagating
Activation Differences. In _ICML_, 2017.


[32] M. Sundararajan, A. Taly, and Q. Yan. Axiomatic Attribution for Deep Networks. In _ICML_,
2017.


[33] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio. Graph attention `
networks. In _ICLR_, 2018.


[34] T. Xie and J. Grossman. Crystal graph convolutional neural networks for an accurate and
interpretable prediction of material properties. In _Phys. Rev. Lett._, 2018.


[35] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In _ICRL_,
2019.


[36] K. Xu, C. Li, Y. Tian, T. Sonobe, K. Kawarabayashi, and S. Jegelka. Representation learning on
graphs with jumping knowledge networks. In _ICML_, 2018.


[37] Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In _KDD_, pages 1365–1374. ACM,
2015.


[38] C. Yeh, J. Kim, I. Yen, and P. Ravikumar. Representer point selection for explaining deep neural
networks. In _NeurIPS_, 2018.


[39] R. Ying, R. He, K. Chen, P. Eksombatchai, W. Hamilton, and J. Leskovec. Graph convolutional
neural networks for web-scale recommender systems. In _KDD_, 2018.


[40] Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec. Hierarchical graph
representation learning with differentiable pooling. In _NeurIPS_, 2018.


[41] J. You, B. Liu, R. Ying, V. Pande, and J. Leskovec. Graph convolutional policy network for
goal-directed molecular graph generation. 2018.


11


[42] J. You, Rex Ying, and J. Leskovec. Position-aware graph neural networks. In _ICML_, 2019.


[43] M. Zeiler and R. Fergus. Visualizing and Understanding Convolutional Networks. In _ECCV_ .
2014.


[44] M. Zhang and Y. Chen. Link prediction based on graph neural networks. In _NIPS_, 2018.


[45] Z. Zhang, Peng C., and W. Zhu. Deep Learning on Graphs: A Survey. _arXiv:1812.04202_, 2018.


[46] J. Zhou, G. Cui, Z. Zhang, C. Yang, Z. Liu, and M. Sun. Graph Neural Networks: A Review of
Methods and Applications. _arXiv:1812.08434_, 2018.


[47] J. Zilke, E. Loza Mencia, and F. Janssen. DeepRED - Rule Extraction from Deep Neural
Networks. In _Discovery Science_ . Springer International Publishing, 2016.


[48] L. Zintgraf, T. Cohen, T. Adel, and M. Welling. Visualizing deep neural network decisions:
Prediction difference analysis. In _ICLR_, 2017.


[49] M. Zitnik, M. Agrawal, and J. Leskovec. Modeling polypharmacy side effects with graph
convolutional networks. _Bioinformatics_, 34, 2018.


**A** **Multi-instance explanations**


The problem of multi-instance explanations for graph neural networks is challenging and an important
area to study.


Here we propose a solution based on G NN E XPLAINER to find common components of explanations
for a set of 10 explanations for 10 different instances in the same label class. More research in this
area is necessary to design efficient Multi-instance explanation methods. The main challenges in
practice is mainly due to the difficulty to perform graph alignment under noise and variances of node
neighborhood structures for nodes in the same class. The problem is closely related to finding the
maximum common subgraphs of explanation graphs, which is an NP-hard problem. In the following
we introduces a neural approach to this problem. However, note that existing graph libraries (based
on heuristics or integer programming relaxation) to find the maximal common subgraph of graphs can
be employed to replace the neural components of the following procedure, when trying to identify
and align with a prototype.


The output of a single-instance G NN E XPLAINER indicates what graph structural and node feature
information is important for a given prediction. To obtain an understanding of “why is a given set of
nodes classified with label _y_ ”, we want to also obtain a global explanation of the class, which can
shed light on how the identified structure for a given node is related to a prototypical structure unique
for its label. To this end, we propose an alignment-based multi-instance G NN E XPLAINER .


For any given class, we first choose a reference node. Intuitively, this node should be a prototypical
node for the class. Such node can be found by computing the mean of the embeddings of all nodes in
the class, and choose the node whose embedding is the closest to the mean. Alternatively, if one has
prior knowledge about the important computation subgraph, one can choose one which matches most
to the prior knowledge.


Given the reference node for class _c_, _v_ _c_, and its associated important computation subgraph _G_ _S_ ( _v_ _c_ ),
we align each of the identified computation subgraphs for all nodes in class _c_ to the reference _G_ _S_ ( _v_ _c_ ) .
Utilizing the idea in the context of differentiable pooling [ 40 ], we use the a relaxed alignment matrix
to find correspondence between nodes in an computation subgraph _G_ _S_ ( _v_ ) and nodes in the reference
computation subgraph _G_ _S_ ( _v_ _c_ ) . Let _A_ _v_ and _X_ _v_ be the adjacency matrix and the associated feature
matrix of the to-be-aligned computation subgraph. Similarly let _A_ _[∗]_ be the adjacency matrix and
associated feature matrix of the reference computation subgraph. Then we optimize the relaxed
alignment matrix _P ∈_ R _[n]_ _[v]_ _[×][n]_ _[∗]_, where _n_ _v_ is the number of nodes in _G_ _S_ ( _v_ ), and _n_ _[∗]_ is the number of
nodes in _G_ _S_ ( _v_ _c_ ) as follows:


min (8)
_P_ _[|][P]_ _[ T]_ _[ A]_ _[v]_ _[P][ −]_ _[A]_ _[∗]_ _[|]_ [ +] _[ |][P]_ _[ T]_ _[ X]_ _[v]_ _[ −]_ _[X]_ _[∗]_ _[|][.]_


The first term in Eq. (8) specifies that after alignment, the aligned adjacency for _G_ _S_ ( _v_ ) should be as
close to _A_ _[∗]_ as possible. The second term in the equation specifies that the features should for the
aligned nodes should also be close.


12


Figure 6: G NN E XPLAINER is able to provide a prototype for a given node class, which can help identify
functional subgraphs, e.g. a mutagenic compound from the M UTAG dataset.


In practice, it is often non-trivial for the relaxed graph matching to find a good optimum for matching
2 large graphs. However, thanks to the single-instance explainer, which produces concise subgraphs
for important message-passing, a matching that is close to the best alignment can be efficiently
computed.
**Prototype by alignment.** We align the adjacency matrices of all nodes in class _c_, such that they are
aligned with respect to the ordering defined by the reference adjacency matrix. We then use median
to generate a prototype that is resistent to outliers, _A_ proto = median( _A_ _i_ ), where _A_ _i_ is the aligned
adjacency matrix representing explanation for _i_ -th node in class _c_ . Prototype _A_ proto allows users to
gain insights into structural graph patterns shared between nodes that belong to the same class. Users
can then investigate a particular node by comparing its explanation to the class prototype.


**B** **Experiments on multi-instance explanations and prototypes**


In the context of multi-instance explanations, an explainer must not only highlight information locally
relevant to a particular prediction, but also help emphasize higher-level correlations across instances.
These instances can be related in arbitrary ways, but the most evident is class-membership. The
assumption is that members of a class share common characteristics, and the model should help
highlight them. For example, mutagenic compounds are often found to have certain characteristic
functional groups that such _NO_ 2, a pair of Oxygen atoms together with a Nitrogen atom. A trained
eye might notice that Figure 6 already hints at their presence. The evidence grows stronger when a
prototype is generated by G NN E XPLAINER, shown in Figure 6. The model is able to pick-up on this
functional structure, and promote it as archetypal of mutagenic compounds.


**C** **Further implementation details**


**Training details.** We use the Adam optimizer to train both the GNN and explaination methods. All
GNN models are trained for 1000 epochs with learning rate 0.001, reaching accuracy of at least 85%
for graph classification datasets, and 95% for node classification datasets. The train/validation/test
split is 80 _/_ 10 _/_ 10% for all datasets. In G NN E XPLAINER, we use the same optimizer and learning
rate, and train for 100 - 300 epochs. This is efficient since G NN E XPLAINER only needs to be trained
on a local computation graph with _<_ 100 nodes.
**Regularization.** In addition to graph size constraint and graph laplacian constraint, we further impose
the feature size constraint, which constrains that the number of unmasked features do not exceed a
threshold. The regularization hyperparameters for subgraph size is 0 _._ 005 ; for laplacian is 0 _._ 5 ; for
feature explanation is 0 _._ 1. The same values of hyperparameters are used across all experiments.
**Subgraph extraction.** To extract the explanation subgraph _G_ _S_, we first compute the importance
weights on edges (gradients for G RAD baseline, attention weights for A TT baseline, and masked
adjacency for G NN E XPLAINER ). A threshold is used to remove low-weight edges, and identify the
explanation subgraph _G_ _S_ . The ground truth explanations of all datasets are connected subgraphs.
Therefore, we identify the explanation as the connected component containing the explained node in
_G_ _S_ . For graph classification, we identify the explanation by the maximum connected component of
_G_ _S_ . For all methods, we perform a search to find the maximum threshold such that the explanation is
at least of size _K_ _M_ . When multiple edges have tied importance weights, all of them are included in
the explanation.


13


