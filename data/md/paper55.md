## **Handling Missing Data with** **Graph Representation Learning**

**Jiaxuan You** **[1]** _[∗]_ **Xiaobai Ma** **[2]** _[∗]_ **Daisy Yi Ding** **[3]** _[∗]_ **Mykel Kochenderfer** **[2]** **Jure Leskovec** **[1]**


1 Department of Computer Science, 2 Department of Aeronautics and Astronautics,
and [3] Department of Biomedical Data Science, Stanford University
{jiaxuan, jure}@cs.stanford.edu
{maxiaoba, dingd, mykel}@stanford.edu


**Abstract**


Machine learning with missing data has been approached in two different ways,
including _feature imputation_ where missing feature values are estimated based on
observed values and _label prediction_ where downstream labels are learned directly
from incomplete data. However, existing imputation models tend to have strong
prior assumptions and cannot learn from downstream tasks, while models targeting
label prediction often involve heuristics and can encounter scalability issues. Here
we propose G RAPE, a graph-based framework for feature imputation as well as label
prediction. G RAPE tackles the missing data problem using a _graph representation_,
where the observations and features are viewed as two types of nodes in a bipartite
graph, and the observed feature values as edges. Under the G RAPE framework,
the _feature imputation_ is formulated as an _edge-level prediction_ task and the _label_
_prediction_ as a _node-level prediction_ task. These tasks are then solved with Graph
Neural Networks. Experimental results on nine benchmark datasets show that
G RAPE yields 20% lower mean absolute error for imputation tasks and 10% lower
for label prediction tasks, compared with existing state-of-the-art methods.


**1** **Introduction**


Issues with learning from incomplete data arise in many domains including computational biology,
clinical studies, survey research, finance, and economics [ 6, 32, 46, 47, 53 ]. The missing data problem
has previously been approached in two different ways: _feature imputation_ and _label prediction_ .
Feature imputation involves estimating missing feature values based on observed values [ 8, 9, 11,
14, 15, 17, 22, 34, 44, 45, 47 – 50, 56 ], and label prediction aims to directly accomplish a downstream
task, such as classification or regression, with the missing values present in the input data [ 2, 5, 10,
15, 16, 23, 37, 40, 42, 52, 54].


Statistical methods for feature imputation often provide useful theoretical properties but exhibit
notable shortcomings: (1) they tend to make strong assumptions about the data distribution; (2)
they lack the flexibility for handling mixed data types that include both continuous and categorical
variables; (3) matrix completion based approaches cannot generalize to unseen samples and require
retraining when the model encounters new data samples [ 8, 9, 22, 34, 44, 47 ]. When it comes to
models for label prediction, existing approaches such as tree-based methods rely on heuristics [ 5 ]
and tend to have scalability issues. For instance, one of the most popular procedures called surrogate
splitting does not scale well, because each time an original splitting variable is missing for some
observation it needs to rank all other variables as surrogate candidates and select the best alternative.


Recent advances in deep learning have enabled new approaches to handle missing data. Existing
imputation approaches often use deep generative models, such as Generative Adversarial Networks


_∗_ Equal contribution


34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.


Data Matrix
with Missing Values



Labels



Feature Imputation as

Edge-level Prediction


Node Embeddings 0.3





Edge Embeddings


Message Passing


Missing Feature Values


Downstream Labels
















|Col1|0.3|0.5|NA|0.1|
|---|---|---|---|---|
||NA|NA|0.6|0.2|
||0.3|NA|NA|0.5|



Label Prediction as
Bipartite Graph Node-level Prediction

































Observations



Features



Figure 1: In the G RAPE framework, we construct a bipartite graph from the data matrix with missing
feature values, where the entries of the matrix in red indicate the missing values ( **Top Left** ). To
construct the graph, the observations _O_ and features _F_ are considered as two types of nodes and the
observed values in the data matrix are viewed as weighted/attributed edges between the observation
and feature nodes ( **Bottom Left** ). With the constructed graph, we formulate the feature imputation
problem and the label prediction problem as edge-level ( **Top right** ) and node-level ( **Bottom right** )
prediction tasks, respectively. The tasks can then be solved with our G RAPE GNN model that learns
node and edge embeddings through rounds of message passing.


(GANs) [ 56 ] or autoencoders [ 17, 50 ], to reconstruct missing values. While these models are flexible,
they have several limitations: (1) when imputing missing feature values for a given observation, these
models fail to make full use of feature values from other observations; (2) they tend to make biased
assumptions about the missing values by initializing them with special default values.


Here, we propose G RAPE [1], a general framework for feature imputation and label prediction in the
presence of missing data. Our key innovation is to formulate the problem using a _graph representation_,
where we construct a bipartite graph with observations and features as two types of nodes, and the
observed feature values as attributed edges between the observation and feature nodes (Figure 1).
Under this graph representation, the _feature imputation_ can then be naturally formulated as an
_edge-level prediction_ task, and the _label prediction_ as a _node-level prediction_ task.


G RAPE solves both tasks via Graph Neural Networks (GNNs). Specifically, G RAPE adopts a GNN
architecture inspired by the GraphSAGE model [ 20 ], while having three innovations in its design:
(1) since the edges in the graph are constructed based on the data matrix and have rich attribute
information, we introduce _edge embeddings_ during message passing and incorporate both discrete
and continuous edge features in the message computation; (2) we design _augmented node features_ to
initialize observation and feature nodes, which provides greater representation power and maintains
inductive learning capabilities; (3) to overcome the common issue of overfitting in the missing data
problem, we employ an _edge dropout_ technique that greatly boosts the performance of G RAPE .


We compare G RAPE with the state-of-the-art feature imputation and label prediction algorithms on 9
benchmark datasets from the UCI Machine Learning Repository [ 1 ]. In particular, G RAPE yields 20%
lower mean absolute error (MAE) for the imputation tasks and 10% lower MAE for the prediction
tasks at the 30% data missing rate. Finally, we demonstrate G RAPE ’s strong generalization ability by
showing its superior performance on unseen observations without the need for retraining.


1 [Project website with data and code: http://snap.stanford.edu/grape](http://snap.stanford.edu/grape)


2


Overall, our approach has several important benefits: (1) by creating a bipartite graph structure
we create connections between different features (via observations) and similarly between the
observations (via features); (2) GNN elegantly harnesses this structure by learning to propagate and
borrow information from other features/observations in a graph localized way; (3) GNN allows us to
model both feature imputation as well as label prediction in an end-to-end fashion, which as we show
in experiments leads to strong performance improvements.


**2** **Related Work**


**Feature imputation** . Successful statistical approaches for imputation include joint modeling with
Expectation-Maximization [ 11, 14, 15, 25 ], multivariate imputation by chained equations (MICE)

[ 7, 38, 45, 48, 49 ], _k_ -nearest neighbors (KNN) [ 27, 47 ], and matrix completion [ 8, 9, 22, 34, 44, 47 ].
However, joint modeling tends to make assumptions about the data distribution through a parametric
density function; joint modeling and matrix completion lack the flexibility to handle data of mixed
modalities; MICE and KNN cannot accomplish imputation while adapting to downstream tasks.


Recently, deep learning models have also been used to tackle the feature imputation problem [ 17, 43,
50, 56 ]. However, these models have important limitations. Denoising autoencoder (DAE) models

[ 17, 50 ] and GAIN [ 56 ] only use a single observation as input to impute the missing features. In
contrast, G RAPE explicitly captures the complex interactions between multiple observations and
features. GNN-based approaches have also been proposed in the context of matrix completion

[ 3, 21, 35, 62, 63 ]. However, they often make the assumption of finite, known-range values in their
model design, which limits their applicability to imputation problems with continuous values. In
contrast, G RAPE can handle both continuous and discrete feature values.


**Label prediction with the presence of missing data** . Various models have been adapted for label
prediction with the presence of missing data, including tree-based approaches [ 5, 54 ], probabilistic
modeling [ 15 ], logistic regression [ 52 ], support vector machines [ 10, 37 ], deep learning-based models

[ 2, 18, 42 ], and many others [ 16, 23, 30, 40 ]. Specifically, decision tree is a classical statistical
approach that can handle missing values for the label prediction task [ 5 ]. With the surrogate splitting
procedure, decision tree uses a single surrogate variable to replace the original splitting variable with
missing values, which is effective but inefficient, and has been shown to be inferior to the “impute
and then predict” procedure [ 13 ]. Random forests further suffer from the scalability issues as they
consist of multiple decision trees [ 31, 54 ]. In contrast, G RAPE handles the missing feature entries
naturally with the _graph representation_ without any additional heuristics. The computation of G RAPE
is efficient and easily parallelizable with modern deep learning frameworks.


**Overall discussion** . In G RAPE implementation, we adopt several successful GNN design principles.
Concretely, our core architecture is inspired by GraphSAGE [ 20 ]; we apply GraphSAGE to bipartite
graphs following G2SAT [ 59 ]; we use edge dropout in [ 39 ]; we use one-hot auxiliary node features
which has been used in [ 36, 60 ]; we follow the GNN design guidelines in [ 61 ] to select hyperparameters. Moreover, matrix completion tasks have been formulated as bipartite graphs and solved via
GNNs in [ 3, 62 ]; however, they only consider the feature imputation task with discrete feature values.
We emphasize that our main contribution is _not the particular GNN model but the graph-based_
_framework for the general missing data problem_ . G RAPE is the first graph-based solution to both
feature imputation and label prediction aspects of the missing data problem.


**3** **The G** **RAPE** **Framework**


**3.1** **Problem Definition**


Let **D** _∈_ R _[n][×][m]_ be a feature matrix consisting of _n_ data points and _m_ features. The _j_ -th feature of
the _i_ -th data point is denoted as **D** _ij_ . In the missing data problem, certain feature values are missing,
denoted as a mask matrix **M** _∈{_ 0 _,_ 1 _}_ _[n][×][m]_ where the value of **D** _ij_ can be observed only if **M** _ij_ = 1 .
Usually, datasets come with labels of a downstream task. Let **Y** _∈_ R _[n]_ be the label for a downstream
task and **V** _∈{_ 0 _,_ 1 _}_ _[n]_ the train/test partition, where **Y** _i_ can be observed at training test only if **V** _i_ = 1 .
We consider two tasks: (1) feature imputation, where the goal is to predict the missing feature values
**D** _ij_ at **M** _ij_ = 0; (2) label prediction, where the goal is to predict test labels **Y** _i_ at **V** _i_ = 0.


3


**3.2** **Missing Data Problem as a Graph Prediction Task**


The key insight of this paper is to represent the feature matrix with missing values as a _bipartite graph_ .
Then the feature imputation problem and the label prediction problem can naturally be formulated as
node prediction and edge prediction tasks (Figure 1).


**Feature matrix as a bipartite graph** . The feature matrix **D** and the mask **M** can be represented as
an undirected bipartite graph _G_ = ( _V, E_ ), where _V_ is the node set that consists of two types of nodes
_V_ = _V_ _D_ _∪V_ _F_, _V_ _D_ = _{u_ 1 _, ..., u_ _n_ _}_ and _V_ _F_ = _{v_ 1 _, . . ., v_ _m_ _}_, _E_ is the edge set where edges only exist
between nodes in different partitions: _E_ = _{_ ( _u_ _i_ _, v_ _j_ _,_ **e** _u_ _i_ _v_ _j_ ) _| u_ _i_ _∈V_ _D_ _, v_ _j_ _∈V_ _F_ _,_ **M** _ij_ = 1 _}_, where
the edge feature, **e** _u_ _i_ _v_ _j_, takes the value of the corresponding feature **e** _u_ _i_ _v_ _j_ = **D** _ij_ . If **D** _ij_ is a discrete
variable then it is transformed to a one-hot vector then assigned to **e** _u_ _i_ _v_ _j_ . To simplify the notation
**e** _u_ _i_ _v_ _j_, we use **e** _ij_ in the context of feature matrix **D**, and **e** _uv_ in the context of graph _G_ .


**Feature imputation as edge-level prediction** . Using the definitions above, imputing missing features can be represented as learning the edge value prediction mapping: **D** [ˆ] _ij_ = ˆ **e** _ij_ = _f_ _ij_ ( _G_ ) by
minimizing the difference between **D** [ˆ] _ij_ and **D** _ij_ _, ∀_ **M** _ij_ = 0 . When imputing discrete attributes, we
use cross entropy loss. When imputing continuous values, we use MSE loss.


**Label prediction as node-level prediction** . Predicting downstream node labels can be represented
as learning the mapping: **Y** [ˆ] _i_ = _g_ _i_ ( _G_ ) by minimizing the difference between **Y** [ˆ] _i_ and **Y** _i_ _, ∀_ **V** _i_ = 0.


**3.3** **Learning with G** **RAPE**


G RAPE adopts a GNN architecture inspired by GraphSAGE [ 20 ], which is a variant of GNNs that
has been shown to have strong inductive learning capabilities across different graphs. We extend
GraphSAGE to a bipartite graph setting by adding multiple important components that ensure its
successful application to the missing data problem.


**G** **RAPE** **GNN architecture** . Given that our bipartite graph _G_ has important information on its edges,
we modify GraphSAGE architecture by introducing _edge embeddings_ . At each GNN layer _l_, the
message passing function takes the concatenation of the embedding of the source node **h** [(] _v_ _[l][−]_ [1)] and
the edge embedding **e** [(] _uv_ _[l][−]_ [1)] as the input:


**n** [(] _v_ _[l]_ [)] = A GG _l_ _σ_ ( **P** [(] _[l]_ [)] _·_ C ONCAT ( **h** [(] _v_ _[l][−]_ [1)] _,_ **e** [(] _uv_ _[l][−]_ [1)] ) _| ∀u ∈N_ ( _v, E_ _drop_ )) (1)
� �


where A GG _l_ is the aggregation function, _σ_ is the non-linearity, **P** [(] _[l]_ [)] is the trainable weight, _N_ is the
node neighborhood function. Node embedding **h** [(] _v_ _[l]_ [)] [is then updated using:]


**h** [(] _v_ _[l]_ [)] = _σ_ ( **Q** [(] _[l]_ [)] _·_ C ONCAT ( **h** [(] _v_ _[l][−]_ [1)] _,_ **n** [(] _v_ _[l]_ [)] [))] (2)


where **Q** [(] _[l]_ [)] is the trainable weight, we additionally update the edge embedding **e** [(] _uv_ _[l]_ [)] [by:]


**e** [(] _uv_ _[l]_ [)] [=] _[ σ]_ [(] **[W]** [(] _[l]_ [)] _[ ·]_ [ C] [ONCAT] [(] **[e]** [(] _uv_ _[l][−]_ [1)] _,_ **h** [(] _u_ _[l]_ [)] _[,]_ **[ h]** [(] _v_ _[l]_ [)] [))] (3)


where **W** [(] _[l]_ [)] is the trainable weight. To make edge level predictions at the _L_ -th layer:


**D** ˆ _uv_ = **O** _edge_ (C ONCAT ( **h** [(] _u_ _[L]_ [)] _[,]_ **[ h]** [(] _v_ _[L]_ [)] )) (4)


The node-level prediction is made using the imputed dataset **D** [ˆ] :


**Y** ˆ _u_ = **O** _node_ ( ˆ **D** _u·_ ) (5)


where **O** _edge_ and **O** _node_ are feedforward neural networks.


**Augmented node features for bipartite message passing** . Based on our definition, nodes in _V_ _D_
and _V_ _F_ do not naturally come with features. The straightforward approach would be to augment
nodes with constant features. However, such formulation would make G RAPE hard to differentiate
messages from different feature nodes in _V_ _F_ . In real-world applications, different features can
represent drastically different semantics or modalities. For example in the _Boston Housing_ dataset
from UCI [ 1 ], some features are categorical such as if the house is by the Charles River, while others
are continuous such as the size of the house.


4


**Algorithm 1** G RAPE forward computation

**Input:** Graph _G_ = ( _V_ ; _E_ ) ; Number of layers _L_ ; Edge dropout rate _r_ _drop_ ; Weight matrices **P** [(] _[l]_ [)] for
_message passing_, **Q** [(] _[l]_ [)] for _node updating_, and **W** [(] _[l]_ [)] for _edge updating_ ; non-linearity _σ_ ; aggregation
functions A GG _l_ ; neighborhood function _N_ : _v × E →_ 2 _[V]_
**Output:** Node embeddings **h** _v_ corresponding to each _v ∈V_

1: **h** [(0)] _v_ _←_ I NIT ( _v_ ) _, ∀v ∈V_

2: **e** [(0)] _uv_ _[←]_ **[e]** _uv_ _[,][ ∀]_ **[e]** _uv_ _[∈E]_
3: _E_ _drop_ _←_ D ROP E DGE ( _E, r_ _drop_ )
4: **for** _l ∈{_ 1 _, . . ., L}_
5: **for** _v ∈V_
6: **n** [(] _v_ _[l]_ [)] = A GG _l_ _σ_ ( **P** [(] _[l]_ [)] _·_ C ONCAT ( **h** [(] _v_ _[l][−]_ [1)] _,_ **e** [(] _uv_ _[l][−]_ [1)] ) _| ∀u ∈N_ ( _v, E_ _drop_ ))
� �

7: **h** [(] _v_ _[l]_ [)] = _σ_ ( **Q** [(] _[l]_ [)] _·_ C ONCAT ( **h** [(] _v_ _[l][−]_ [1)] _,_ **n** [(] _v_ _[l]_ [)] [))]
8: **for** ( _u, v_ ) _∈E_ _drop_
9: **e** [(] _uv_ _[l]_ [)] [=] _[ σ]_ [(] **[W]** [(] _[l]_ [)] _[ ·]_ [ C] [ONCAT] [(] **[e]** [(] _uv_ _[l][−]_ [1)] _,_ **h** [(] _u_ _[l]_ [)] _[,]_ **[ h]** [(] _v_ _[l]_ [)] [))]
10: _z_ _v_ _←_ _h_ _[L]_ _v_


Instead, we propose to use _m_ -dimensional one-hot node features for each node in _V_ _F_ ( _m_ = _|V_ _F_ _|_ ),
while using _m_ -dimensional [1] constant vectors as node feature for data nodes in _V_ _F_ :


**1** _v ∈V_ _D_
I NIT ( _v_ ) = (6)
�O NE H OT _v ∈V_ _F_


Such a formulation leads to a better representational power to differentiate feature nodes with different
underlying semantics or modalities. Additionally, the formulation has the capability of generalizing
the trained G RAPE to completely unseen data points in the given dataset. Furthermore, it allows us to
transfer knowledge from an external dataset with the same set of features to the dataset of interest,
which is particularly useful when the external dataset provides rich information on the interaction
between observations and features (as captured by G RAPE ). For example, as a real-world application
in biomedicine, gene expression data can be used to predict disease types and frequently contain
missing values. If we aim to impute missing values in a gene expression dataset of a small cohort
of lung cancer patients, public datasets, e.g., the Cancer Genome Atlas Program (TCGA) [ 51 ] can
be first leveraged to train G RAPE, where rich interactions between patients and features are learned.
Then, the trained G RAPE can be applied to our smaller dataset of interest to accomplish imputation.


**Improved model generalization with edge dropout** . When doing feature imputation, a naive way
of training G RAPE is to directly feed _G_ = ( _V_ ; _E_ ) as the input. However, since all the observed edge
values are used as the input, an identity mapping **D** [ˆ] _ij_ = **e** [(0)] _ij_ [is enough to minimize the training loss;]
therefore, G RAPE trained under this setting easily overfits the training set. To force the model to
generalize to unseen edge values, we randomly mask out edges _E_ with dropout rate _r_ _drop_ :


D ROP E DGE ( _E, r_ _drop_ ) = _{_ ( _u_ _i_ _, v_ _j_ _,_ _ij_ ) _|_ ( _u_ _i_ _, v_ _j_ _,_ **e** _ij_ ) _∈E,_ **M** _drop,ij_ _> r_ _drop_ _}_ (7)


where **M** _drop_ _∈_ R _[n][×][m]_ is a random matrix sampled uniformly in (0 _,_ 1) . This approach is similar to
DropEdge [ 39 ], but with a more direct motivation for feature imputation. At test time, we feed the
full graph _G_ to G RAPE . Overall, the complete computation of G RAPE is summarized in Algorithm 1.


**4** **Experiments**


**4.1** **Experimental Setup**


**Datasets** . We conduct experiments on 9 datasets from the UCI Machine Learning Repository [ 1 ]. The
datasets come from different domains including civil engineering ( CONCRETE, ENERGY ), biology
( PROTEIN ), thermal dynamics ( NAVAL ), etc. The smallest dataset ( YACHT ) has 314 observations and
6 features, while the largest dataset ( PROTEIN ) has over 45,000 observations and 9 features. The
datasets are fully observed; therefore, we introduce missing values by randomly removing values in
the data matrix. The attribute values are scaled to [0 _,_ 1] with a MinMax scaler [29].


1 We make data nodes and feature nodes to have the same feature dimension for the ease of implementation.


5


|Col1|Col2|Col3|
|---|---|---|
|~~rete energy housing kin8nm~~<br>~~nav~~<br>Data|~~rete energy housing kin8nm~~<br>~~nav~~<br>Data|~~al~~<br>~~power~~<br>~~protein~~<br>~~wine~~<br>~~yac~~<br>set|








|rete energy housing kin8nm naval power protein wine yac Dataset|Col2|
|---|---|
|||
|~~crete energy housing kin8nm~~<br>~~nav~~<br>Data|~~al~~<br>~~power~~<br>~~protein~~<br>~~wine~~<br>~~yac~~<br>set|



Figure 2: Averaged MAE of _feature imputation_ (upper) and _label prediction_ (lower) on UCI datasets
over 5 trials at data missing level of 0.3. The result is normalized by the average performance of
Mean imputation. G RAPE yields 20% lower MAE for imputation and 10% lower MAE for prediction
compared with the best baselines (KNN for imputation and MICE for prediction).


**Baseline models** . We compare our model against five commonly used imputation methods. We also
compare with a state-of-the-art deep learning based imputation model as well as a decision tree based
label prediction model. More details on the baseline models are provided in the Appendix.


 - Mean imputation (Mean): The method imputes the missing **D** _ij_ with the mean of all the samples
with observed values in dimension _j_ .

 - K-nearest neighbors (KNN): The method imputes the missing value **D** _ij_ using the KNNs that
have observed values in dimension _j_ with weights based on the Euclidean distance to sample _i_ .

 - Multivariate imputation by chained equations (MICE): The method runs multiple regression
where each missing value is modeled conditioned on the observed non-missing values.

 - Iterative SVD (SVD) [ 47 ]: The method imputes missing values based on matrix completion with
iterative low-rank SVD decomposition.

 - Spectral regularization algorithm (Spectral) [ 34 ]: This matrix completion model uses the nuclear
norm as a regularizer and imputes missing values with iterative soft-thresholded SVD.

 - GAIN [56], state-of-the-art deep imputation model with generative adversarial training [19].

 - Decision tree (Tree) [ 5 ], a commonly used statistical method that can handle missing values for
label prediction. We consider this baseline only for the label prediction task. [1]


**G** **RAPE** **configurations** . For all experiments, we train G RAPE for 20,000 epochs using the Adam
optimizer [ 28 ] with a learning rate at 0.001. For all _feature imputation_ tasks, we use a 3-layer GNN
with 64 hidden units and R E LU activation. The AGG _l_ is implemented as a mean pooling function
M EAN ( _·_ ) and **O** _edge_ as a multi-layer perceptron (MLP) with 64 hidden units. For _label prediction_
tasks, we use two GNN layers with 16 hidden units. **O** _edge_ and **O** _node_ are implemented as linear
layers. The edge dropout rate is set to _r_ _drop_ = 0 _._ 3 . For all experiments, we run 5 trials with different
random seeds and report the mean and standard deviation of the results.


**4.2** **Feature Imputation**


**Setup** . We first compare the feature imputation performance of G RAPE and all other imputation
baselines. Given a full data matrix **D** _∈_ R _[n][×][m]_, we generate a random mask matrix **M** _∈{_ 0 _,_ 1 _}_ _[n][×][m]_


1 Random forest is not included due to the lack of a public implementation that can handle missing data
without imputation.


6


|protein|protein|Col3|Col4|
|---|---|---|---|
|0.10<br>0.30<br>0.50<br>0.70<br>Missing data ratio<br>0.02<br>0.04<br>0.06<br>0.08<br>0.10<br>0.12<br>0.14<br><br>Mea<br>KNN<br>MIC<br>SVD<br>Spec<br>GAIN<br>GRA|2<br>4<br>6<br>8<br>0<br>2<br>4<br>|||
|0.10<br>0.30<br>0.50<br>0.70<br>Missing data ratio<br>0.02<br>0.04<br>0.06<br>0.08<br>0.10<br>0.12<br>0.14<br><br>Mea<br>KNN<br>MIC<br>SVD<br>Spec<br>GAIN<br>GRA|2<br>4<br>6<br>8<br>0<br>2<br>4<br>||Mea<br>KNN<br>MIC<br>SVD<br>Spec<br>GAI|
|0.10<br>0.30<br>0.50<br>0.70<br>Missing data ratio<br>0.02<br>0.04<br>0.06<br>0.08<br>0.10<br>0.12<br>0.14<br><br>Mea<br>KNN<br>MIC<br>SVD<br>Spec<br>GAIN<br>GRA|2<br>4<br>6<br>8<br>0<br>2<br>4<br>|||






















|0.10 0.30 0.50 0.70 Missing data ratio 02 protein|0 0.30 0.50 0.70 Missing data ratio protein|Col3|Col4|
|---|---|---|---|
|0.10<br>0.30<br>0.50<br>0.70<br>Missing data ratio<br>.00<br>.20<br>.40<br>.60<br>.80<br>.00<br>.20<br>.40<br><br>Me<br>KN<br>MI<br>SV<br>Sp<br>GA<br>Tre<br>GR||M|M|
|0.10<br>0.30<br>0.50<br>0.70<br>Missing data ratio<br>.00<br>.20<br>.40<br>.60<br>.80<br>.00<br>.20<br>.40<br><br>Me<br>KN<br>MI<br>SV<br>Sp<br>GA<br>Tre<br>GR|||KN<br>MI<br>SV<br>Sp<br>|



Figure 3: Averaged MAE of _feature imputation_ (upper) and _label prediction_ (lower) with _different_
_missing ratios_ over 5 trials. G RAPE yields 12% lower MAE on imputation and 2% lower MAE on
prediction tasks across different missing data ratios.


with _P_ ( **M** _ij_ = 0) = _r_ _miss_ at a data missing level _r_ _miss_ = 0 _._ 3 . A bipartite graph _G_ = ( _V, E_ ) is then
constructed based on **D** and **M** as described in Section 3.2. _G_ is used as the input to G RAPE at both
the training and test time. The training loss is defined as the mean squared error (MSE) between **D** _ij_
and **D** ˆ _ij_, **D** _∀_ [ˆ] _ij_ **M**, _∀_ _ij_ **M** = 0 _ij_ = 1. . The test metric is defined as the mean absolute error (MAE) between **D** _ij_ and


**Results** . As shown in Figure 2, G RAPE has the lowest MAE on all datasets and its average error is
20% lower compared with the best baseline (KNN). Since there are significant differences between
the characteristics of different datasets, statistical methods often need to adjust its hyper-parameters
accordingly, such as the cluster number in KNN, the rank in SVD, and the sparsity in Spectral. On the
contrary, G RAPE is able to adjust its trainable parameters adaptively through loss backpropagation
and learn different observation-feature relations for different datasets. Compared with GAIN, which
uses an MLP as the generative model, the GNN used in G RAPE is able to explicitly model the
information propagation process for predicting missing feature values.


**4.3** **Label Prediction**


**Setup** . For label prediction experiments, with the same input graph _G_, we have an additional label
vector **Y** _∈_ R _[n]_ . We randomly split the labels **Y** into 70/30% training and test sets, **Y** _train_ and **Y** _test_
respectively. The training loss is defined as the MSE between the true **Y** ˆ _train_ . The test metric is calculated based on the MAE between **Y** _test_ **Y** and _train_ ˆ **Y** _test_ and the predicted. For baselines
except decision tree, since no end-to-end approach is available, we first impute the data and then do
linear regression on the imputed data matrix for predicting **Y** [ˆ] .


**Results** . As is shown in Figure 2, on all datasets except NAVAL and WINE, G RAPE has the best
performance. On WINE dataset, all methods have comparable performance. The fact that the
performance of all methods are close to the Mean method indicates that the relation between the
labels and observations in WINE is relatively simple. For the dataset NAVAL, the imputation errors
of all models are very small (both relative to Mean and on absolute value). In this case, a linear
regression on the imputed data is enough for label prediction. Across all datasets, G RAPE yields 10%
lower MAE compared with best baselines. The improvement of G RAPE could be explained by two
reasons: first, the better handling of missing data with G RAPE where the known information and the
missing values are naturally embedded in the graph; and second, the end-to-end training.


7


|Col1|Col2|Col3|
|---|---|---|
||||
|~~rete energy housing kin8nm~~<br>~~naval~~<br>~~power~~<br>~~protein~~<br>~~wine~~<br>~~yac~~<br>Dataset|~~rete energy housing kin8nm~~<br>~~naval~~<br>~~power~~<br>~~protein~~<br>~~wine~~<br>~~yac~~<br>Dataset|~~rete energy housing kin8nm~~<br>~~naval~~<br>~~power~~<br>~~protein~~<br>~~wine~~<br>~~yac~~<br>Dataset|



Figure 4: Averaged MAE of _feature imputation on unseen data_ in UCI datasets over 5 trials. The
result is normalized by the average performance of Mean imputation. G RAPE yields 21% lower MAE
compared with best baselines (MICE).


**4.4** **Robustness against Different Data Missing Levels**


**Setup** . To examine the robustness of G RAPE with respect to the missing level of the data matrix. We
conduct the same experiments as in Sections 4.2 and 4.3 with different missing levels of _r_ _miss_ _∈_
_{_ 0 _._ 1 _,_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_ .


**Results** . The curves in Figure 3 demonstrate the performance change of all methods as the missing
ratio increases. G RAPE yields -8%, 20%, 20%, and 17% lower MAE on imputation tasks, and -15%,
10%, 10%, and 4% lower MAE on prediction tasks across all datasets over missing ratios of 0.1, 0.3,
0.5, and 0.7, respectively. In missing ratio of 0.1, the only baseline that behaves better than G RAPE is
KNN. As in this case, the known information is adequate for the nearest-neighbor method to make
good predictions. As the missing ratio increases, the prediction becomes harder and the G RAPE ’s
ability to coherently combine all known information becomes more important.


**4.5** **Generalization on New Observations**


**Setup** . We further investigate the _generalization_ ability of G RAPE . Concretely, we examine whether
a trained G RAPE can be successfully applied to new observations that are not in the training dataset.
A good generalization ability reduces the effort of re-training when there are new observations being
recorded after the model is trained. We randomly divide the _n_ observations in **D** _∈_ R _[n][×][m]_ into two
sets, represented as **D** _train_ _∈_ R _[n]_ _[train]_ _[×][m]_ and **D** _test_ _∈_ R _[n]_ _[test]_ _[×][m]_, where **D** _train_ and **D** _test_ contain
70% and 30% of the observations, respectively. The missing rate _r_ _miss_ is at 0.3. We construct two
graphs _G_ _train_ and _G_ _test_ based on **D** _train_ and **D** _test_, respectively. We then train G RAPE with **D** _train_
and _G_ _train_ using the same procedure as described in Section 4.2. At test time, we directly feed _G_ _test_
to the trained G RAPE and evaluate its performance on predicting the missing values in **D** _test_ . We
repeat the same procedure for GAIN where training is also required. For all other baselines, since
they do not need to be trained, we directly apply them to impute on **D** _test_ .


**Results** . As shown in Figure 4, G RAPE yields 21% lower MAE compared with best baselines (MICE)
without being retrained, indicating that our model generalizes seamlessly to unseen observations.
Statistical methods have difficulties transferring the knowledge in the training data to new data. While
GAIN is able to encode such information in the generator network, it lacks the ability to adapt to
observations coming from a different distribution. However, by using a GNN, G RAPE is able to make
predictions conditioning on the entire new datasets, and thus capture the distributional changes.


**4.6** **Ablation Study**


**Edge dropout** . We test the influence of the edge dropout on the performance of G RAPE . We repeat
the experiments in Section 4.2 for G RAPE with no edge dropout and the comparison results are
shown in Section 4.6. The edge dropout reduces the test MAE by 33% on average, which verifies our
assumption that using edge dropout could help the model learn to predict unseen edge values.


**Aggregation function** . We further investigate how the aggregation function ( S UM ( _·_ ), M AX ( _·_ ),
M EAN ( _·_ ) ) of GNN affects G RAPE ’s performance. While S UM ( _·_ ) is theoretically most expressive,
in our setting the degree of a specific node is determined by the number of missing values which is


8


Table 1: **Ablation study for G** **RAPE** . Averaged MAE of G RAPE on UCI datasets over 5 trials. Edge
dropout (upper) reduces the average MAE by 33% on feature imputation tasks. M EAN ( _·_ ) is adopted
in our implementation. End-to-End training (lower) reduces the average MAE by 19% on prediction
tasks (excluding two outliers).


concrete energy housing kin8nm naval power protein wine yacht


Without edge dropout 0.171 0.148 0.104 0.262 0.021 0.192 0.047 0.094 0.204
**With edge dropout** **0.090** **0.136** **0.075** **0.249** **0.008 0.102** **0.027** **0.063 0.151**


S UM ( _·_ ) 0.094 0.143 0.078 0.277 0.024 0.134 0.040 0.069 0.154
M AX ( _·_ ) **0.088** 0.142 **0.074** 0.252 **0.006 0.102** **0.024** **0.063** 0.153
**M** **EAN** ( _·_ ) 0.090 **0.136** 0.075 **0.249** 0.008 **0.102** 0.027 **0.063 0.151**


Impute then predict 9.36 2.59 3.80 0.181 **0.004** 4.80 4.48 **0.524** 9.02
**End-to-End** **7.88** **1.65** **3.39** **0.163** 0.007 **4.61** **4.23** 0.535 **4.72**


random and unrelated to the missing data task; in contrast, the M EAN ( _·_ ) and M AX ( _·_ ) aggregators are
not affected by this inherent randomness of node degree, therefore they perform better.


**End-to-end downstream regression** . To show the benefits of using end-to-end training in label
prediction, we repeat the experiments in Section 4.3 by first using G RAPE to impute the missing
data and then perform linear regression on the imputed dataset for node labels (which is the same
prediction model as the linear layer used by G RAPE ). The results are shown in Section 4.6. The
end-to-end training gets 19% less averaged MAE over all datasets except NAVAL and WINE . The
reason for the two exceptions is similar as described in Section 4.3.


**4.7** **Further Discussions**


**Scalability** . In our paper, we use UCI datasets as they are widely-used datasets for benchmarking
imputation methods, with _both discrete and continuous features_ . G RAPE can easily scale to datasets
with thousands of features. We provide additional results on larger-scale benchmarks, including
Flixster (2956 features), Douban (3000 features), and Yahoo (1363 features) in the Appendix. G RAPE
can be modified to scale to even larger datasets. We can use scalable GNN implementations which
have been successfully applied to graphs with billions of edges [ 55, 58 ]; when the number of features
is prohibitively large, we can use a trainable embedding matrix to replace one-hot node features.


**Applicability of G** **RAPE** . In the paper, we adopt the most common evaluation regime used in missing
data papers, i.e., features are missing completely at random. G RAPE can be easily applied to other
missing data regimes where feature are not missing at random, since G RAPE is fully data-driven.


**More intuitions on why G** **RAPE** **works** . When a feature matrix does not have missing values, to
make downstream label predictions, a reasonable solution will be directly feeding the feature matrix
into an MLP. As is discussed in [ 57 ], an MLP can in fact be viewed as a GNN over a complete graph,
where the message function is matrix multiplication. Under this interpretation, G RAPE extends a
simple MLP by allowing it to operate on sparse graphs ( _i.e._, feature matrix with missing values),
enabling it for missing feature imputation tasks, and adopting a more complex message computation
as we have outlined in Algorithm 1.


**5** **Conclusion**


In this work, we propose G RAPE, a framework to coherently understand and solve missing data
problems using _graphs_ . By formulating the _feature imputation_ and _label prediction_ tasks as edge-level
and node-level predictions on the graph, we are able to train a Graph Neural Network to solve the
tasks end-to-end. We further propose to adapt existing GNN structures to handle continuous edge
values. Our model shows significant improvement in both tasks compared against state-of-the-art
imputation approaches on nine standard UCI datasets. It also generalizes robustly to unseen data
points and different data missing ratios. We hope our work will open up new directions on handling
missing data problems with graphs.


9


**Broader Impact**


The problem of missing data arises in almost all practical statistical analyses. The quality of the
imputed data influences the reliability of the dataset itself as well as the success of the downstream
tasks. Our research provides a new point of view for analysing and handling missing data problems
with _graph representations_ . There are many benefits to using this framework. First, different from
many existing imputation methods which rely on good heuristics to ensure the performance [ 43 ],
G RAPE formulates the problem in a natural way without the need of handcrafted features and
heuristics. This makes our method ready to use for datasets coming from different domains. Second,
similar to convolutional neural networks [ 24, 41 ], G RAPE is suitable to serve as a pre-processing
module to be connected with downstream task-specific modules. G RAPE could either be pre-trained
and fixed or concurrently learned with downstream modules. Third, G RAPE is general and flexible.
There is little limitation on the architecture of the graph neural network as well as the imputation
( **O** _edge_ ) and prediction ( **O** _node_ ) module. Therefore, researchers can easily plug in domain-specific
neural architectures, e.g., BERT [ 12 ], to the design of G RAPE . Overall, we see exciting opportunities
for G RAPE to help researchers handle missing data and thus boost their research.


**Acknowledgments**


We gratefully acknowledge the support of DARPA under Nos. FA865018C7880 (ASED),
N660011924033 (MCS); ARO under Nos. W911NF-16-1-0342 (MURI), W911NF-16-1-0171
(DURIP); NSF under Nos. OAC-1835598 (CINES), OAC-1934578 (HDR), CCF-1918940 (Expeditions), IIS-2030477 (RAPID); Stanford Data Science Initiative, Wu Tsai Neurosciences Institute,
Chan Zuckerberg Biohub, Amazon, Boeing, JPMorgan Chase, Docomo, Hitachi, JD.com, KDDI,
NVIDIA, Dell. J. L. is a Chan Zuckerberg Biohub investigator.


**References**


[1] A. Asuncion and D. Newman. UCI Machine Learning Repository, 2007.


[2] Y. Bengio and F. Gingras. Recurrent neural networks for missing or asynchronous data. In
_Advances in Neural Information Processing Systems (NeurIPS)_, 1996.


[3] R. v. d. Berg, T. N. Kipf, and M. Welling. Graph convolutional matrix completion.
_arXiv:1706.02263_, 2017.


[4] R. v. d. Berg, T. N. Kipf, and M. Welling. Graph convolutional matrix completion. _arXiv_
_preprint arXiv:1706.02263_, 2017.


[5] L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen. _Classification and Regression Trees_ .
CRC Press, 1984.


[6] J. M. Brick and G. Kalton. Handling missing data in survey research. _Statistical Methods in_
_Medical Research_, 5(3):215–238, 1996.


[7] L. F. Burgette and J. P. Reiter. Multiple imputation for missing data via sequential regression
trees. _American Journal of Epidemiology_, 172(9):1070–1076, 2010.


[8] J.-F. Cai, E. J. Candès, and Z. Shen. A singular value thresholding algorithm for matrix
completion. _SIAM Journal on Optimization_, 20(4):1956–1982, 2010.


[9] E. J. Candès and B. Recht. Exact matrix completion via convex optimization. _Foundations of_
_Computational Mathematics_, 9(6):717–772, 2009.


[10] G. Chechik, G. Heitz, G. Elidan, P. Abbeel, and D. Koller. Max-margin classification of data
with absent features. _Journal of Machine Learning Research_, 9(Jan):1–21, 2008.


[11] A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data
via the EM algorithm. _Journal of the Royal Statistical Society: Series B (Methodological)_,
39(1):1–22, 1977.


10


[12] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding. _Annual Conference of the North American Chapter of_
_the Association for Computational Linguistics (NAACL)_, 2019.


[13] A. Feelders. Handling missing data in trees: surrogate splits or statistical imputation? In
_European Conference on Principles of Data Mining and Knowledge Discovery_, 1999.


[14] P. J. García-Laencina, J.-L. Sancho-Gómez, and A. R. Figueiras-Vidal. Pattern classification
with missing data: a review. _Neural Computing and Applications_, 19(2):263–282, 2010.


[15] Z. Ghahramani and M. I. Jordan. Supervised learning from incomplete data via an em approach.
In _Advances in Neural Information Processing Systems (NeurIPS)_, 1994.


[16] A. Goldberg, B. Recht, J. Xu, R. Nowak, and J. Zhu. Transduction with matrix completion:
Three birds with one stone. In _Advances in Neural Information Processing Systems (NeurIPS)_,
2010.


[17] L. Gondara and K. Wang. Multiple imputation using deep denoising autoencoders. _Pacific-Asia_
_Conference on Knowledge Discovery and Data Mining_, 2018.


[18] I. Goodfellow, M. Mirza, A. Courville, and Y. Bengio. Multi-prediction deep boltzmann
machines. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2013.


[19] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and
Y. Bengio. Generative adversarial nets. In _Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2014.


[20] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In
_Advances in Neural Information Processing Systems (NeurIPS)_, 2017.


[21] J. Hartford, D. R. Graham, K. Leyton-Brown, and S. Ravanbakhsh. Deep models of interactions
across sets. _International Conference on Machine Learning (ICML)_, 2018.


[22] T. Hastie, R. Mazumder, J. D. Lee, and R. Zadeh. Matrix completion and low-rank svd via fast
alternating least squares. _Journal of Machine Learning Research_, 16:3367–3402, 2015.


[23] E. Hazan, R. Livni, and Y. Mansour. Classification with low rank and missing data. In
_International Conference on Machine Learning (ICML)_, pages 257–266, 2015.


[24] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In _IEEE_
_Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)_, pages
770–778, 2016.


[25] J. Honaker, G. King, and M. Blackwell. Amelia II: A program for missing data. _Journal of_
_Statistical Software_, 45(7):1–47, 2011.


[26] J. Josse, F. Husson, et al. missmda: a package for handling missing values in multivariate data
analysis. _Journal of Statistical Software_, 70(1):1–31, 2016.


[27] K.-Y. Kim, B.-J. Kim, and G.-S. Yi. Reuse of imputed data in microarray analysis increases
imputation efficiency. _BMC Bioinformatics_, 5(1):160, 2004.


[28] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. _arXiv:1412.6980_, 2014.


[29] J. Leskovec, A. Rajaraman, and J. Ullman. _Mining of Massive Datasets_ . Cambridge University
Press, 3 edition, 2020.


[30] X. Liao, H. Li, and L. Carin. Quadratically gated mixture of experts for incomplete data
classification. In _International Conference on Machine Learning (ICML)_, 2007.


[31] A. Liaw and M. Wiener. Classification and regression by randomforest. _R News_, 2(3):18–22,
2002.


[32] R. J. A. Little and D. B. Rubin. _Statistical Analysis with Missing Data_ . Wiley, 2019.


11


[33] P.-A. Mattei and J. Frellsen. MIWAE: Deep generative modelling and imputation of incomplete
data sets. In _International Conference on Machine Learning (ICML)_, 2019.


[34] R. Mazumder, T. Hastie, and R. Tibshirani. Spectral regularization algorithms for learning large
incomplete matrices. _Journal of Machine Learning Research_, 11:2287–2322, 2010.


[35] F. Monti, M. Bronstein, and X. Bresson. Geometric matrix completion with recurrent multigraph neural networks. In _Advances in Neural Information Processing Systems (NeurIPS)_,
2017.


[36] R. L. Murphy, B. Srinivasan, V. Rao, and B. Ribeiro. Relational pooling for graph representations. _International Conference on Machine Learning (ICML)_, 2019.


[37] K. Pelckmans, J. De Brabanter, J. A. Suykens, and B. De Moor. Handling missing values in
support vector machine classifiers. _Neural Networks_, 18(5-6):684–692, 2005.


[38] T. E. Raghunathan, J. M. Lepkowski, J. Van Hoewyk, and P. Solenberger. A multivariate
technique for multiply imputing missing values using a sequence of regression models. _Survey_
_Methodology_, 27(1):85–96, 2001.


[39] Y. Rong, W. Huang, T. Xu, and J. Huang. Dropedge: Towards deep graph convolutional
networks on node classification. In _International Conference on Learning Representations_
_(ICLR)_, 2019.


[40] P. K. Shivaswamy, C. Bhattacharyya, and A. J. Smola. Second order cone programming
approaches for handling missing and uncertain data. _Journal of Machine Learning Research_,
7(Jul):1283–1314, 2006.


[41] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image
recognition. _International Conference on Learning Representations (ICLR)_, 2015.


[42] M. Smieja, Ł. Struski, J. Tabor, B. Zieli´nski, and P. Spurek. Processing of missing data by [´]
neural networks. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2018.


[43] I. Spinelli, S. Scardapane, and A. Uncini. Missing data imputation with adversarially-trained
graph convolutional networks. _Neural Networks_, 2020.


[44] N. Srebro, J. Rennie, and T. S. Jaakkola. Maximum-margin matrix factorization. In _Advances_
_in Neural Information Processing Systems (NeurIPS)_, 2005.


[45] D. J. Stekhoven and P. Bühlmann. Missforest—non-parametric missing value imputation for
mixed-type data. _Bioinformatics_, 28(1):112–118, 2012.


[46] J. A. C. Sterne, I. R. White, J. B. Carlin, M. Spratt, P. Royston, M. G. Kenward, A. M. Wood,
and J. R. Carpenter. Multiple imputation for missing data in epidemiological and clinical
research: potential and pitfalls. _BMJ_, 338:b2393, 2009.


[47] O. Troyanskaya, M. Cantor, G. Sherlock, P. Brown, T. Hastie, R. Tibshirani, D. Botstein,
and R. B. Altman. Missing value estimation methods for dna microarrays. _Bioinformatics_,
17(6):520–525, 2001.


[48] S. van Buuren. Multiple imputation of discrete and continuous data by fully conditional
specification. _Statistical Methods in Medical Research_, 16(3):219–242, 2007.


[49] S. van Buuren and K. Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations
in R. _Journal of Statistical Software_, pages 1–68, 2010.


[50] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust
features with denoising autoencoders. In _International Conference on Machine Learning_
_(ICML)_, pages 1096–1103, 2008.


[51] J. N. Weinstein, E. A. Collisson, G. B. Mills, K. R. M. Shaw, B. A. Ozenberger, K. Ellrott,
I. Shmulevich, C. Sander, J. M. Stuart, C. G. A. R. Network, et al. The cancer genome atlas
pan-cancer analysis project. _Nature genetics_, 45(10):1113, 2013.


12


[52] D. Williams, X. Liao, Y. Xue, and L. Carin. Incomplete-data classification using logistic
regression. In _International Conference on Machine Learning (ICML)_, pages 972–979, 2005.


[53] J. M. Wooldridge. Inverse probability weighted estimation for general missing data problems.
_Journal of Econometrics_, 141(2):1281–1301, 2007.


[54] J. Xia, S. Zhang, G. Cai, L. Li, Q. Pan, J. Yan, and G. Ning. Adjusted weight voting algorithm
for random forests in handling missing values. _Pattern Recognition_, 69:52–60, 2017.


[55] R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec. Graph convolutional neural networks for web-scale recommender systems. _ACM SIGKDD International_
_Conference on Knowledge Discovery and Data Mining (KDD)_, 2018.


[56] J. Yoon, J. Jordon, and M. Van Der Schaar. GAIN: Missing data imputation using generative
adversarial nets. _International Conference on Machine Learning (ICML)_, 2018.


[57] J. You, J. Leskovec, K. He, and S. Xie. Graph structure of neural networks. _International_
_Conference on Machine Learning (ICML)_, 2020.


[58] J. You, Y. Wang, A. Pal, P. Eksombatchai, C. Rosenburg, and J. Leskovec. Hierarchical temporal
convolutional networks for dynamic recommender systems. In _The Web Conference (WWW)_,
2019.


[59] J. You, H. Wu, C. Barrett, R. Ramanujan, and J. Leskovec. G2SAT: Learning to generate sat
formulas. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2019.


[60] J. You, R. Ying, and J. Leskovec. Position-aware graph neural networks. _International_
_Conference on Machine Learning (ICML)_, 2019.


[61] J. You, R. Ying, and J. Leskovec. Design space for graph neural networks. In _Advances in_
_Neural Information Processing Systems (NeurIPS)_, 2020.


[62] M. Zhang and Y. Chen. Inductive matrix completion based on graph neural networks. _Interna-_
_tional Conference on Learning Representations (ICLR)_, 2020.


[63] L. Zheng, C.-T. Lu, F. Jiang, J. Zhang, and P. S. Yu. Spectral collaborative filtering. In _ACM_
_Conference on Recommender Systems_, pages 311–319, 2018.


13


**A** **Additional Details on Baseline Implementation**


For imputation baselines including Mean, KNN, MICE, SVD, and Spectral, we use the implementation provided in the _fancyimpute_ package [1] . For KNN, we use 50 nearest neighbors. For SVD, we
set the _rank_ equal to _m −_ 1, where _m_ is the number of features. For MICE, we set the _maximum_
_iteration number_ to 3. For Spectral, we found the default heuristic for _shrinkage value_ works the best.
For a detailed explanation of the meaning of the parameters, we refer readers to the documentation of
_fancyimpute_ package. The hyper-parameter values are chosen by comparing the average imputation
performance over all datasets. For GAIN, we use the source code released by the authors. All
the hyper-parameters are the same as in the source code [2] . We use the _rpart_ R package for the
implementation of the decision tree method.


**B** **Running Time Comparison**


Here we report the running clock time for feature imputation of different methods at test time. For
Mean, KNN, MICE, SAC, and Spectral, this means the running time of one function call for imputing
the entire dataset. For GAIN and G RAPE, this means one forward pass of the network. Appendix B
shows the averaged running time over 5 different trials with the same setting as described in Section
4.2.


Table 2: Running clock time (second) for feature imputation of different methods at test time.


concrete energy housing kin8nm naval power protein wine yacht


Mean 0 _._ 000806 0 _._ 000922 0 _._ 000942 0 _._ 00242 0 _._ 00596 0 _._ 00147 0 _._ 0127 0 _._ 00121 0 _._ 00064

KNN 0 _._ 225 0 _._ 134 0 _._ 0913 9 _._ 95 30 _._ 1 11 _._ 4 656 0 _._ 504 0 _._ 0268

MICE 0 _._ 0294 0 _._ 0311 0 _._ 0499 0 _._ 0749 0 _._ 256 0 _._ 0249 0 _._ 271 0 _._ 0531 0 _._ 027

SVD 0 _._ 0659 0 _._ 0192 0 _._ 0359 0 _._ 162 0 _._ 0612 0 _._ 142 0 _._ 593 0 _._ 0564 0 _._ 0412
Spectral 0 _._ 0718 0 _._ 0565 0 _._ 0541 0 _._ 268 0 _._ 405 0 _._ 199 1 _._ 63 0 _._ 0978 0 _._ 0311
GAIN 0 _._ 0119 0 _._ 0125 0 _._ 0131 0 _._ 017 0 _._ 0298 0 _._ 0146 0 _._ 0457 0 _._ 0131 0 _._ 0116

G RAPE 0 _._ 0263 0 _._ 011 0 _._ 0115 0 _._ 0874 0 _._ 259 0 _._ 0488 0 _._ 568 0 _._ 0199 0 _._ 00438


**C** **Comparisons with Additional Baselines**


We additionally provide the comparison results of our method with two other state-of-the-art baselines:
missMDA [ 26 ], a statistical multiple imputation approach, and MIWAE [ 33 ], a deep generative model.
We adapt the same setting as in Section 4.1 and the results are shown in Appendix C. G RAPE yields
the smallest imputation error on all datasets compared with the two other baselines.


Table 3: Averaged MAE of _feature imputation_ on UCI datasets at data missing level of 0.3.


concrete energy housing kin8nm naval power protein wine yacht


missMDA 0.190 0.225 0.142 0.285 0.038 0.215 0.068 0.090 0.226

MIWAE 0.156 0.153 0.098 0.262 0.020 0.117 0.042 0.087 0.224

G RAPE **0.090** **0.136** **0.075** **0.249** **0.008** **0.102** **0.027** **0.063** **0.151**


**D** **Experiments on Larger Datasets**


To test the scalability of G RAPE, we perform additional _feature imputation_ tests on the Flixter,
Douban, and YahooMusic detests with preprocessed subsets and splits provided by [ 35 ]. The Flixster
dataset has 2341 observations and 2956 features. The Douban dataset has 3000 observations and

3000 features. The YahooMusic dataset has 1357 observations and 1363 features. These datasets


1 [https://github.com/iskandr/fancyimpute](https://github.com/iskandr/fancyimpute)
2 [https://github.com/jsyoon0823/GAIN](https://github.com/jsyoon0823/GAIN)


14


only have discrete values. We compare G RAPE with two GNN-based approaches, GC-MC [ 4 ] and
IGMC [62]. The results are shown in Table 4, where the results of GC-MC and IGMC are provided
by [ 62 ]. On all datasets, G RAPE shows a reasonable performance which is better than GC-MC and
close to IGMC. Notice that the two baselines are specially designed for discrete matrix completion,
where G RAPE is applicable to both continuous and discrete feature values and is general for both
feature imputation and label prediction tasks.


Table 4: RMSE test results on Flixster, Douban, and YahooMusic.


Flixster Douban Yahoo


GC-MC 0.917 0.734 20.5

IGMC **0.872** **0.721** **19.1**

Ours 0.899 0.733 19.4


15


