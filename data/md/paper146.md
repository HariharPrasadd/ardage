## **SCOUT: Self-aware Discriminant Counterfactual Explanations**

Pei Wang Nuno Vasconcelos
Department of Electrical and Computer Engineering
University of California, San Diego


_{_ pew062,nuno _}_ @ucsd.edu



**Abstract**


_The problem of counterfactual visual explanations is_
_considered. A new family of discriminant explanations is_
_introduced._ _These produce heatmaps that attribute high_
_scores to image regions informative of a classifier predic-_
_tion but not of a counter class. They connect attributive_
_explanations, which are based on a single heat map, to_
_counterfactual explanations, which account for both pre-_
_dicted class and counter class._ _The latter are shown to_

_be computable by combination of two discriminant expla-_
_nations, with reversed class pairs. It is argued that self-_
_awareness, namely the ability to produce classification con-_
_fidence scores, is important for the computation of discrimi-_
_nant explanations, which seek to identify regions where it is_
_easy to discriminate between prediction and counter class._
_This suggests the computation of discriminant explanations_
_by the combination of three attribution maps. The resulting_
_counterfactual explanations are optimization free and thus_
_much faster than previous methods. To address the diffi-_
_culty of their evaluation, a proxy task and set of quantita-_
_tive metrics are also proposed. Experiments under this pro-_
_tocol show that the proposed counterfactual explanations_
_outperform the state of the art while achieving much higher_
_speeds, for popular networks. In a human-learning machine_
_teaching experiment, they are also shown to improve mean_
_student accuracy from chance level to_ 95% _._


**1. Introduction**


Deep learning (DL) systems are difficult to deploy in
specialized domains, such as medical diagnosis or biology,
requiring very fine-grained distinctions between visual features unnoticeable to the untrained eye. Two main difficulties arise. The first is the black-box nature of DL. When
high-stakes decisions are involved, e.g. a tumor diagnosis, the system users, e.g. physicians, require a justification for its predictions. The second is the large data labeling requirements of DL. Since supervised training is usually needed for optimal classification, modern networks are



Figure 1: Given a query image (Cardinal) and a counterfactual class (Summer Tanager), discriminant explanations bridge the gap between attributions and counterfactual explanations. This enables a fast optimization-free
computation of the latter.


trained with large datasets, manually annotated on Amazon
MTurk. However, because MTurk annotators lack domain
expertise, the approach does not scale to specialized domains.

Both problems can be addressed by explainable AI
(XAI) techniques, which complement network predictions
with human-understandable explanations. These can both
circumvent the black-box nature of DL and enable the de
sign of machine teaching systems that provide feedback to
annotators when they make mistakes [43]. In computer vision, the dominant XAI paradigm is _attribution,_ which consists of computing a heatmap of how strongly each image
pixel [31, 3, 29, 1] or region [41, 28] contributes to a network prediction. For example, when asked “why is this
a truck?” an attributive system would answer or visualize something like “because it has wheels, a hood, seats, a
steering wheel, a flatbed, head and tail lights, and rearview
mirrors.”

While useful to a naive user, this explanation is less useful to an expert in the domain. The latter is likely to be interested in more _precise_ feedback, asking instead the question




























“Why is it not a car?” The answer “because it has a flatbed.
If it did not have a flatbed it would be a car,” is known as a
_counterfactual_ or _contrastive explanation_ [36, 8, 24]. Such
explanations are more desirable in expert domains. When
faced with a prediction of lesion _A_, a doctor would naturally ask “why _A_ but not _B_ ?” The same question would
be posed by a student that incorrectly assigned an image to
class _B_ upon receiving feedback that it belongs to class _A_ .
By supporting a specific query with respect to a _counter-_
_factual_ class ( _B_ ), these explanations allow expert users to
zero-in on a specific ambiguity between two classes, which
they already _know_ to be plausible prediction outcomes. Unlike attributions, counterfactual explanations scale naturally
with user expertise. As the latter increases, the class and
counterfactual class simply become more _fine-grained._

In computer vision, counterfactual explanations have
only recently received attention. They are usually implemented as “correct class is _A_ . Class _B_ would require changing the image as follows,” where “as follows” is some visual
transformations. Possible transformations include image
perturbations [8], synthesis [36] or the exhaustive search of
a large feature pool, to find replacement features that map
the image from class _A_ to _B_ [12]. However, image perturbations and synthesis frequently leave the space of natural
images only working on simple non-expert domains, and
feature search is too complex for interactive applications.

In this work, a new procedure is proposed to generate _Self-aware disCriminant cOUnterfactual explanaTions_
(SCOUT). We show that counterfactual explanations can
be much more efficiently generated by a combination of
attributive explanations and self-awareness mechanisms,
which quantify the confidence of the predictions of a DL
system. For this, we start by introducing _discriminant ex-_
_planations_ that, as shown in Figure 1, connect attributive to
counterfactual explanations. Like attributive explanations,
they consist of a single heatmap. This, however, is an attribution map for the _discrimination_ of classes _A_ and _B_,
attributing high scores to image regions that are informative
of _A_ but not of _B_ . In this sense, discriminant explanations
are similar to counterfactual explanations and more precise
than attributive explanations (see Figure 1). A _counterfac-_
_tual explanation_ can then be produced by the computation
of two discriminant explanations, with the roles of _A_ and _B_
reversed.

We next consider how to compute discriminant explanations and argue for the importance of self-awareness. A
system is self-aware if it can _quantify the confidence_ with
which it classifies an image. This is generally true for DL
systems, which complement a class prediction with an estimate of the posterior class distribution, from which a confidence score can be derived [10, 39]. The attribution map
of this score is an indicator of the image regions where the
classification is easy. This fits nicely in the discriminant



explanation framework, where the goal is to find the spatial regions predictive of class A but unpredictive of class
B. It leads to the _definition of discriminant explanations_ as
image regions that simultaneously: 1) have high attribution
for class _A_, 2) have low attribution for class _B_, and 3) are
classified with high confidence. It follows that, as shown
in Figure 1, discriminant explanations can be computed by
combination of three attribution maps. This, in turn, shows
that counterfactual explanations can be seen as a _general-_
_ization_ of attributive explanations and computed by a _com-_
_bination_ of attribution [31, 3, 29, 34, 1] and confidence prediction methods [10, 39, 37] that is much more efficient to
compute than previous methods.
Beyond explanations, a significant challenge to XAI is
the lack of explanation ground truth for performance evaluation. Besides user-based evaluations [12], whose results
are difficult to replicate, we propose a quantitative metric
based on a proxy localization task. To the best of our knowledge, this is the first proposal for semantically quantitative
evaluation of counterfactual visual explanations independently of human experiments. Compared to the latter, the
proposed proxy evaluation is substantially easier to replicate. This evaluation shows that SCOUT _both_ outperforms
the state of the art [12] and is 50 _×_ to 1000 _×_ faster for popular networks. This is quite important for applications such
as machine teaching, where explanation algorithms should
operate in real-time, and ideally in low-complexity platforms such as mobile devices.

Overall, the paper makes five contributions. First, a new
family of discriminant explanations, which are substantially
more precise than attributive explanations. Second, the use
of self-awareness to improve the accuracy of attributive explanations. Third, the derivation of counterfactual explanations by combination of discriminant explanations, making
them more efficient to compute. Fourth, a new experimental
protocol for quantitative evaluation of counterfactual explanations. Fifth, experimental results using both this protocol
and machine teaching experiments, showing that the proposed SCOUT outperforms previous methods and is substantially faster.


**2. Related work**


In this section we review the literature on explanations,
self-awareness, and machine teaching.
**Explanations:** Two main approaches to explainable AI
(XAI) have emerged in computer vision. Natural language
(NL) systems attempt to produce a textual explanation understandable to humans [13, 2, 26]. Since image to text
translation is still a difficult problem, full blown NL explanations tend to target specific applications, like self driving [6]. More robust systems tend to use a limited vocabulary, e.g. a set of image attributes [2, 13]. For example, [2] proposed counterfactual NL image descriptions


and [13] produces counterfactual explanations by extracting noun phrases from the counter-class, which are filtered
with an evidence checker. Since phrases are defined by attributes, this boils down to detecting presence/absence of attributes in the query image. These methods require a priori
definition of a vocabulary (e.g. attributes), training data for
each vocabulary term, and training of the classifier to produce this side information. Due to these difficulties, most
explanation methods rely instead on visualizations. While
the ideas proposed in this work could be extended to NL
systems, we consider only visual explanations.

**Attributive explanations:** The most popular approach to
visual explanations is to rely on attributions [3, 29, 34].
These methods produce a heatmap that encodes how much
the classifier prediction can be attributed to each pixel or
image region. Many attribution functions have been proposed [31, 3, 29, 34, 1]. The most popular framework is to
compute some variant of the gradient of the classifier prediction with respect a chosen layer of the network and then
backproject to the input [28, 41]. These techniques tend
to work well when the object of the predicted class is immersed in a large background (as in object detection), but
are less useful when the image contains the object alone
(as in recognition). In this setting, the most suitable for the
close inspection required in expert domains, the heat map
frequently covers the whole object. This is illustrated in
Figure 1. Counterfactual explanations, which involve differences with respect to a counterfactual class, tend not to
suffer from this problem.

**Counterfactual explanations:** Given an image of class _A_
and a counterfactual class _B_, counterfactual explanations
(also known as contrastive [8]) produce an image transformation that elicits the classification as _B_ [35, 36, 21, 44].
The simplest example are adversarial attacks [8, 35, 43],
which optimize perturbations to map an image of class _A_
into class _B_ . However, adversarial perturbations usually
push the perturbed image outside the boundaries of the
space of natural images. Generative methods have been proposed to address this problem, computing large perturbations that generate realistic images [21, 23]. This is guaranteed by the introduction of regularization constraints, autoencoders, or GANs [11]. However, because realistic images
are difficult to synthesize, these approaches have only been
applied to simple MNIST or CelebA [22] style datasets, not
expert domains. A more plausible alternative is to exhaustively search the space of features extracted from a large collection of images, to find replacement features that map the
image from class _A_ to _B_ [12]. While this has been shown
to perform well on fine-grained datasets, exhaustive search
is too complex for interactive applications.

**Evaluation:** The performance of explanation algorithms
is frequently only illustrated by the display of visualizations. In some cases, explanations are evaluated quantita


tively with recourse to human experiments. This involves
the design of a system to elicit user feedback on how trustful a deep learning system is [28, 12, 8, 38] or evaluate if
explanations improve user performance on some tasks [12].
While we present results of this type, they have several limitations: it can be difficult to replicate system design, conclusions can be affected by the users that participate in the
experiments, and the experiments can be cumbersome to
both set up and perform. In result, the experimental results
are rarely replicable or even comparable. This hampers the
scalable evaluation of algorithms. In this work, we introduce a quantitative protocol for the evaluation of counterfactual explanations, which overcomes these problems.
**Self-awareness:** Self-aware systems are systems with some
abilities to measure their limitations or predict failures. This
includes topics such as out-of-distribution detection [14, 20,
7, 18, 19] or open set recognition [27, 5], where classifiers are trained to reject non-sensical images, adversarial
attacks, or images from classes on which they were not
trained. All these problems require the classifier to produce
a confidence score for image rejection. The most popular
solution is to guarantee that the posterior class distribution
is uniform, or has high entropy, outside the space covered
by training images [18, 15]. This, however, is not sufficient for counterfactual explanations, which require more
precise confidence scores explicitly addressing class _A_ or
_B_ . In this sense, the latter are more closely related to realistic classification [37], where a classifier must identify and
reject examples that it deems too difficult to classify.
**Machine teaching:** Machine teaching systems [43] are
usually designed to teach some tasks to human learners,
e.g. image labeling. These systems usually leverage
a model of student learning to optimize teaching performance [33, 4, 17, 25]. Counterfactual explanations are naturally suited for machine teaching, because they provide
feedback on why a mistake (the choice of the counterfactual
class _B_ ) was made. While the goal of this work is not to design a full blown machine teaching system, we investigate
if counterfactual explanations can improve human labeling
performance. This follows the protocol introduced by [12],
which highlights matching bounding boxes on paired images (what part of _A_ should be replaced by what part of _B_ )
to provide feedback to students. Besides improved labeling
performance, the proposed explanations are orders of magnitude faster than the exhaustive search of [12].


**3. Discriminant Counterfactual Explanations**


In this section, we briefly review the main ideas behind previous explanation approaches and introduce the
proposed explanation technique.
**Counterfactual explanations:** Consider a recognition
problem, mapping images **x** _∈X_ into classes _y ∈Y_ =
_{_ 1 _, . . ., C}_ . Images are classified by an object recognition


system _H_ : _X →Y_ of the form


_y_ _[∗]_ = arg max _h_ _y_ ( **x** ) _,_ (1)
_y_


where **h** ( **x** ) : _X →_ [0 _,_ 1] _[C]_ is a C-dimensional probability distribution with [�] _[C]_ _y_ =1 _[h]_ _[y]_ [(] **[x]** [) = 1][, usually com-]
puted by a convolutional neural network (CNN). The classifier is learned on a training set _D_ of _N_ i.i.d. samples
_D_ = _{_ ( **x** _i_ _, y_ _i_ ) _}_ _[N]_ _i_ =1 [, where] _[ y]_ _[i]_ _[ ∈Y]_ [ is the label of im-]
age **x** _i_ _∈X_, and its performance evaluated on a test set
_T_ = _{_ ( **x** _j_ _, y_ _j_ ) _}_ _[M]_ _j_ =1 [. Given an image] **[ x]** [, for which the clas-]
sifier predicts class _y_ _[∗]_, counterfactual explanations answer
the question of why the image does not belong to a counterfactual class (also denoted counter class) _y_ _[c]_ = _̸_ _y_ _[∗]_, chosen
by the user who receives the explanation.
**Visual explanations:** Counterfactual explanations for vision systems are usually based on visualizations. Two possibilities exist. The first is to explicitly transform the image **x** into an image **x** _[c]_ of class _y_ _[c]_, by replacing some of
its pixel values. The transformation can consist of applying an image perturbation akin to those used in adversarial
attacks [8], or replacing regions of **x** by regions of some
images in the counter class _y_ _[c]_ [12]. Due to the difficulties
of realistic image synthesis, these methods are only feasible
when **x** is relatively simple, e.g. an MNIST digit.
A more plausible alternative is to use an already available image **x** _[c]_ from class _y_ _[c]_ and highlight the differences
between **x** and **x** _[c]_ . [12] proposed to do this by displaying
matched bounding boxes on the two images, and showed
that explanation performance is nearly independent of the
choice of **x** _[c]_, i.e. it suffices to use a random image **x** _[c]_ from
class _y_ _[c]_ . We adopt a similar strategy in this work. For these
approaches, the explanation consists of


_C_ ( **x** _, y_ _[∗]_ _, y_ _[c]_ _,_ **x** _[c]_ ) = ( **c** _[∗]_ ( **x** ) _,_ **c** _[c]_ ( **x** _[c]_ )) _,_ (2)


where **c** _[∗]_ ( **x** ) and **c** _[c]_ ( **x** _[c]_ ) are _counterfactual heatmaps_ for
images **x** and **x** _[c]_, respectively, from which region segments
**r** _[∗]_ ( **x** ) and **r** _[c]_ ( **x** _[c]_ ) can be obtained, usually by thresholding.
The question is how to compute these heatmaps. [12] proposed to search by exhaustively matching all combinations
of features in **x** and **x** _[c]_, which is expensive. In this work,
we propose a much simpler and more effective procedure
that leverages a large literature on _attributive_ explanations.
**Attributive explanations:** Attributive explanations are a
family of explanations based on the attribution of the prediction _y_ _[∗]_ to regions of **x** [31, 3, 29, 34, 1]. They are usually
produced by applying an attribution function to a tensor of
activations **F** _∈_ R _[W][ ×][H][×][D]_ of spatial dimensions _W × H_
and _D_ channels, extracted at any layer of a deep network.
While many attribution functions have been proposed, they
are usually some variant of the gradient of _h_ _y_ _∗_ ( **x** ) with respect to **F** . This results in an _attribution map a_ _i,j_ ( **x** ) whose
amplitude encodes the attribution of the prediction to each



entry _i, j_ along the spatial dimensions of **F** . Attributive explanations produce heat maps of the form


_A_ ( **x** _, y_ _[∗]_ ) = **a** ( _h_ _y_ _∗_ ( **x** )) (3)


for some attribution function **a** ( _._ ). Two examples of attributive heatmaps of an image of a ”Cardinal,” with respect to
predictions ”Cardinal” and ”Summer Tanager,” are shown
in the top row of Figure 1.
**Discriminant explanations:** In this work, we propose a
new class of explanations, which is denoted as _discriminant_
and defined as


_D_ ( **x** _, y_ _[∗]_ _, y_ _[c]_ ) = **d** ( _h_ _y_ _∗_ ( **x** ) _, h_ _y_ _c_ ( **x** )) _,_ (4)

_̸_

which have commonalities with both attributive and coun
terfactual explanations. Like counterfactual explanations,
they consider both the prediction _y_ _[∗]_ and a counterfactual class _y_ _[c]_ . Like attributive explanations, they compute a _single attribution map_ through **d** ( _., ._ ). The difference is that this map _attributes the discrimination between_
_the prediction y_ _[∗]_ _and counter y_ _[c]_ _class_ to regions of **x** .
While **a** ( _h_ _y_ _∗_ ( **x** )) assigns large attribution to pixels that are
strongly informative of class _y_ _[∗]_, **d** ( _h_ _y_ _∗_ ( **x** ) _, h_ _y_ _c_ ( **x** )) does
the same to pixels that are _strongly informative_ of class _y_ _[∗]_

but _uninformative_ of class _y_ _[c]_ .
Discriminant explanations can be used to compute counterfactual explanations by implementing (2) with


_C_ ( **x** _, y_ _[∗]_ _, y_ _[c]_ _,_ **x** _[c]_ ) = ( _D_ ( **x** _, y_ _[∗]_ _, y_ _[c]_ ) _, D_ ( **x** _[c]_ _, y_ _[c]_ _, y_ _[∗]_ )) _._ (5)


The first map identifies the regions of **x** that are informative
of the predicted class but not the counter class while the second identifies the regions of **x** _[c]_ informative of the counter
class but not of the predicted class. Altogether, the explanation shows that the regions highlighted in the two images
are matched: the region of the first image depicts features
that _only_ appear in the predicted class while that of the second depicts features that _only_ appear in the counterfactual
class. Figure 1 illustrates the construction of a counterfactual explanation with two discriminant explanations.
**Self-awareness:** Discriminant maps could be computed by
combining attributive explanations with respect to the predicted and counter class. Assuming that binary ground
truth segmentation maps _s_ _[∗]_ _i,j_ [and] _[ s]_ _[c]_ _i,j_ [are available for the]
attributions of the predicted and counter classes, respectively, this could be done with the segmentation map _s_ _i,j_ =
_s_ _[∗]_
_i,j_ _[.]_ [(1] _[ −]_ _[s]_ _[c]_ _i,j_ [)] _[.]_ [ This map would identify image regions at-]
tributable to the predicted class _y_ _[∗]_ but not the counter class
_y_ _[c]_ . In practice, segmentation maps are not available and
can only be estimated from attribution maps _a_ _[∗]_ _i,j_ [and] _[ a]_ _[c]_ _i,j_ [.]
While this could work well when the two classes are very
different, it is not likely to work when they are similar. This
is because, as shown in Figure 1, attribution maps usually
cover substantial parts of the object. When the two classes


Figure 2: Discriminant explanation architecture ( **x** : Cardinal, **x** _[c]_ : Summer Tanager.). Feature activations **F** _h_ and **F** _s_ are computed for some layers of the
classifier (upper branch) and confidence predictor (lower branch), respectively. Attributions for prediction _h_ _y_ _∗_, counter class _h_ _y_ _[c]_, and confidence score _s_
are computed by attribution functions _q_ ( _., ._ ) according to (10) and then combined with (6) to obtain the discriminant map. Counterfactual explanations are
obtained by reversing the roles of **x** and **x** _[c]_ and thresholding the discriminant heat maps.



differ only in small parts or details, they lack the precision
to allow the identification of the associated regions. This
is critical for expert domains, where users are likely to ask
questions involving very similar classes.
Addressing this problem requires some ways to sharpen
attribution maps. In this work, we advocate for the use of
self-awareness. We assume that the classifier produces a
_confidence score s_ ( **x** ) _∈_ [0 _,_ 1], which encodes the strength
of its belief that the image belongs to the predicted class.
Regions that clearly belong to the predicted class _y_ _[∗]_ render
a score close to 1 while regions that clearly do not render a
score close to 0. This score is _self-referential_ if generated
by the classifier itself and _not self-referential_ if generated by
a separate network. The discriminant maps of (4) are then
implemented as


**d** ( _h_ _y_ _∗_ ( **x** ) _, h_ _y_ _c_ ( **x** )) = **a** ( _h_ _y_ _∗_ ( **x** )) _·_ ~~**a**~~ ~~(~~ _h_ _y_ _c_ ( **x** )) _·_ **a** ( _s_ ( **x** )) (6)


where ~~**a**~~ ~~(~~ _._ ) is the complement of **a** ( _._ ), i.e.


~~**a**~~ _i,j_ = max _i,j_ **[a]** _[i,j]_ _[ −]_ **[a]** _[i,j]_ _[.]_ (7)


The discriminant map **d** is large only at locations ( _i, j_ )
that contribute strongly to the prediction of class _y_ _[∗]_ but little
to that of class _y_ _[c]_, _and where the discrimination between the_
_two classes is easy, i.e. the classifier is confident._ This, in
turn, implies that location ( _i, j_ ) is strongly specific to class
_y_ _[∗]_ but non specific to class _y_ _[c]_, which is the essence of the
counterfactual explanation. Figure 1 shows how the selfawareness attribution map is usually much sharper than the
other two maps.
**Segmentations:** For discriminant explanations, the discriminant map of **x** is thresholded to obtain the segmentation mask


**r** _{y_ _[∗]_ _, y_ _[c]_ _}_ ( **x** ) = 1 **d** ( _h_ _y∗_ ( **x** ) _,h_ _yc_ ( **x** )) _>T_ _,_ (8)


where 1 _S_ is the indicator function of set _S_ and _T_ a threshold. For counterfactual explanations, segmentation masks



Its computation requires an additional layer of log nonlinearities and average pooling. These two scores are selfreferential. We also consider the non-self-referential _easi-_

_ness score_ of [37],


_s_ _[e]_ ( **x** ) = 1 _−_ _s_ _[hp]_ ( **x** ) (13)


where _s_ _[hp]_ ( **x** ) is computed by an external hardness predictor _S_, which is jointly trained with the classifier. _S_ is implemented with a network _s_ _[hp]_ ( **x** ) : _X →_ [0 _,_ 1] whose output
is a sigmoid unit.



are also generated for **x** _[c]_, using


**r** _{y_ _[c]_ _, y_ _[∗]_ _}_ ( **x** _[c]_ ) = 1 **d** ( _h_ _yc_ ( **x** _c_ ) _,h_ _y∗_ ( **x** _c_ )) _>T_ _._ (9)


**Attribution maps:** The attribution maps of (6) can be
computed with any attribution function **a** ( _._ ) in the literature [34, 29, 3]. In our implementation, we use the gradientbased function of [30]. This calculates the dot-product of
the partial derivatives of the prediction _p_ with respect to the
activations **F** ( **x** ) of a CNN layer and the activations, i.e.



_a_ _i,j_ ( _h_ _p_ ) = _q_ � **f** _i,j_ _,_ _[∂h]_ _∂_ _[p]_ **f** _i,j_ [(] **[F]** [)]



� = � _∂h∂_ _p_ **f** _i,j_ ( **F** ) _,_ **f** _i,j_



_,_ (10)
�



where we omit the dependency on **x** for simplicity.
**Confidence scores:** Like attribution maps, many existing
confidence or hardness scores can be leveraged. We considered three scores of different characteristics. The _softmax_
_score_ [10] is the largest class posterior probability


_s_ _[s]_ ( **x** ) = max _h_ _y_ ( **x** ) _._ (11)
_y_


It is computed by adding a max pooling layer to the network output. The _certainty score_ is the complement of the
normalized entropy of the softmax distribution [39],



1
_s_ _[c]_ ( **x** ) = 1 +
log _C_



� _h_ _y_ ( **x** ) log _h_ _y_ ( **x** ) _._ (12)


_y_


**Network implementation:** Figure 2 shows a network implementation of (6). Given a query image **x** of class _y_ _[∗]_, a
user-chosen counter class _y_ _[c]_ = _̸_ _y_ _[∗]_, a predictor _h_ _y_ ( **x** ), and
a confidence predictor _s_ ( **x** ) are used to produce the explanation. Note that _s_ ( **x** ) can share weights with _h_ _y_ ( **x** ) (selfreferential) or be separate (non-self-referential). **x** is forwarded through the network, generating activation tensors
**F** _h_ ( **x** ), **F** _s_ ( **x** ) in pre-chosen network layers and predictions
_h_ _y_ _∗_ ( **x** ), _h_ _y_ _c_ ( **x** ), _s_ ( **x** ). The attributions of _y_ _[∗]_, _y_ _[c]_ and _s_ ( **x** )
to **x**, i.e. _A_ ( **x** _, y_ _[∗]_ ), _A_ ( **x** _, y_ _[c]_ ), _A_ ( **x** _, s_ ( **x** )) are then computed
with (10), which reduce to a backpropagation step with respect to the desired layer activations and a few additional
operations. Finally, the three attributions are combined with
(6). Thresholding the resulting heatmap with (8) produces
the discriminant explanation **r** _{y_ _[∗]_ _, y_ _[c]_ _}_ ( **x** ). To further obtain a counterfactual explanation, the network is simply applied to **x** _[c]_ and **r** _{y_ _[c]_ _, y_ _[∗]_ _}_ ( **x** _[c]_ ) computed.


**4. Evaluation**


**Challenges:** Explanations are difficult to evaluate because
ground truth is unavailable. Previous works mainly presented qualitative results [13, 12]. [12] also performed a human evaluation on MTurk, using a machine teaching task.
However, this evaluation had a few flaws, which are discussed in Section 5.4. In any case, human evaluation is
cumbersome and difficult to replicate. To avoid this, we
introduce an alternative evaluation strategy based on the
proxy task of localization. Because this leverages datasets
with annotations for part locations and attributes [1], we sometimes refer to image regions (segments or keypoints) as

parts.
**Ground-truth:** The goal of counterfactual explanations is
to localize a region predictive of class _A_ but unpredictive of
class _B_ . Hence, parts with attributes specific to _A_ and that
do not appear in _B_ can be seen as ground truth counterfactual regions. This enables the evaluation of counterfactual
explanations as a part localization problem. To synthesize
ground truth, the _k_ _[th]_ part of an object of class _c_ is represented by a semantic descriptor _φ_ _[k]_ _c_ [containing the attributes]
present in this class. For example, an “eye” part can have
color attributes “red”, “blue”, “grey”, etc. The descriptor is
a probability distribution over these attributes, characterizing the attribute variability of the part under each class.
The dissimilarity between classes _a_ and _b_, according to
part _k_, is defined as _α_ _a,b_ _[k]_ [=] _[ γ]_ [(] _[φ]_ _a_ _[k]_ _[, φ]_ _[k]_ _b_ [)][, where] _[ γ]_ [(] _[., .]_ [)][ is a]
dataset dependent function. Large dissimilarities indicate
that part _k_ is a discriminant for classes _a_ and _b_ . The values
of _α_ _a,b_ _[k]_ [are computed for all class pairs][ (] _[a, b]_ [)][ and parts] **[ p]** _[k]_ [.]
The _M_ triplets _G_ = _{_ ( **p** _i_ _, a_ _i_ _, b_ _i_ ) _}_ _[M]_ _i_ =1 [of largest dissimilarity]
are selected as counterfactual ground-truth.


1 note that part and attribute annotations are only required for performance evaluation, not to compute the visualizations.



**Evaluation metrics:** The metrics of explanation performance depend on the nature of part annotations. On datasets

_̸_ where part locations are labelled with a single point, i.e.

**p** _i_ is a point (usually the geometric center of the part),
the quality of region **r** _{a, b}_ ( **x** ) is calculated by precision (P) and recall (R), where _P_ = _|{k|_ **p** _J_ _k_ _∈_ **r** _}|_ [,] _[ R]_ [ =]

_|{i|_ ( **p** _i_ _,a_ _i_ _,b_ _i_ ) _∈GJ_ _,a_ _i_ = _a,b_ _i_ = _b}|_ [, and] _[ J]_ = _|{i|_ **p** _i_ _∈_ **r** _, a_ _i_ =
_a, b_ _i_ = _b}|_ is the number of included ground truth parts
of generated regions. Precision-recall curves are produced
by varying the threshold _T_ used in (8). For datasets where
parts are annotated with segmentation masks, the quality
of **r** _{a, b}_ ( **x** ) is evaluated using the intersection over union
_|_ **r** _∩_ **p** _|_
(IoU) metric IoU = _|_ **r** _∪_ **p** _|_ [, where] **[ p]** [ =] _[ {]_ **[p]** _[i]_ _[|]_ [(] **[p]** _[i]_ _[, a]_ _[i]_ _[, b]_ _[i]_ [)] _[ ∈]_
_G, a_ _i_ = _a, b_ _i_ = _b}_ .
For counterfactual explanations, we define a measure of
the semantic consistency of two segments, **r** _{a, b}_ ( **x** ) and
**r** _{b, a}_ ( **x** _[c]_ ), by calculating the consistency of the parts included in them. This is denoted as the part IoU (PIoU),


PIoU = _|{|{kk||_ (( **pp** _kk,a,b,a,b_ )) _∈∈_ **rr** _{{a,ba,b}}_ (( **xx** )) _}}∪{∩{kk||_ (( **pp** _kk,b,a,b,a_ )) _∈∈_ **rr** _{{b,ab,a}}_ (( **xx** _[c]_ ~~_[c]_~~ )) _}|}|_ _[.]_ (14)


These metrics allow the quantitative comparison of different counterfactual explanation methods. On datasets
with point-based ground truth, this is based on precision and
recall of the generated counterfactual regions. On datasets
with mask-based ground truth, the IoU is used. After conducting the whole process on both **x** and **x** _[c]_, PIoU can
be computed to further measure the semantic matching between the highlighted regions in the two images. As long
as the compared counterfactual regions of different methods have the same size, the comparison is fair. For SCOUT,
region size can be controlled by manipulating _T_ in (8) and
(9) .
User expertise has an impact on counterfactual explanations. Beginner users tend to choose random counterfactual
classes, while experts tend to pick counterfactual classes
similar to the true class. Hence, explanation performance
should be measured over the two user types. In this paper,
users are simulated by choosing a random counterfactual
class _b_ for beginners and the class predicted by a small CNN
for advanced users. Class _a_ is the prediction of the classifier
used to generate the explanation, which is a larger CNN.


**5. Experiments**


All experiments are performed on two datasets.
CUB200 [40] consists of 200 fine-grained bird classes, annotated with 15 part locations (points) including back, beak,
belly, breast, crown, forehead, left/right eye, left/right leg,
left/right wing, nape, tail and throat. Each part is associated
with attribute information [40] and dissimilarities _α_ _a,b_ _[k]_ [are]

computed with _γ_ ( _φ_ _[k]_ _a_ _[, φ]_ _[k]_ _b_ [) =] _[ e]_ _[{]_ [KL][(] _**[φ]**_ _a_ _[k]_ _[||]_ _**[φ]**_ _[k]_ _b_ [)+][KL][(] _**[φ]**_ _[k]_ _b_ _[||]_ _**[φ]**_ _[k]_ _a_ [)] _[}]_ [9],
where _φ_ _[k]_ _c_ [is a probability distribution over all attributes of]


|Col1|Col2|Beginner User Ad|Col4|
|---|---|---|---|
|Arch.|Metric Go|yal [12]|SCOUT<br>Goyal [|
|VGG16|R<br>0.0<br>P<br>0.7<br>PIoU 0.1<br>IPS<br>0.0|2 (0.01)<br>6 (0.01)<br>3 (0.00)<br>2 (0.00)|**0.05** (0.01)<br>**0.05** (0<br>**0.84** (0.01)<br>0.56 (0<br>**0.15** (0.00)<br>0.09 (0<br>** 26.51** (0.71)|
|ResNet-50|R<br>0.0<br>P<br>0.7<br>PIoU** 0.1**<br>IPS<br>1.1|3 (0.01)<br>7 (0.01)<br>**8** (0.01)<br>3 (0.07)|**0.09** (0.02)<br>0.12 (0<br>**0.81** (0.01)<br>0.57 (0<br>0.16 (0.01)<br>**0.15** (0<br>**78.54** (11.87)|



Figure 3: Comparison to attributive explanations. Left: beginners, right:
advanced users.


the _k_ _[th]_ part under class _c_ and KL is the Kullback-Leibler
divergence. _M_ is chosen to leave 80% largest triplets
( **p** _i_ _, a_ _i_ _, b_ _i_ ) as ground truth. The majority of ( **p** _i_ _, a_ _i_ _, b_ _i_ ) are
selected because dissimilar parts dominate in _α_ _a,b_ _[k]_ [space.]
The second dataset is ADE20K [42] with more than
1000 fine-grained scene categories. Segmentation masks
are given for 150 objects. In this case, objects are seen as
scene parts and each object has a single attribute, i.e. _φ_ _[k]_ _c_ [is]
scalar (where _k ∈{_ 1 _, ...,_ 150 _}_ ), which is the probability of
occurrence of the object in a scene of class _c_ . This is estimated by the relative frequency with which the part appears
in scenes of class _c_ . Ground truth consists of the triplets
( **p** _i_ _, a_ _i_ _, b_ _i_ ) with _φ_ _[k]_ _a_ _[>]_ [ 0][ and] _[ φ]_ _[k]_ _b_ [= 0][, i.e. where object] _[ k]_
appears in class _a_ but not in class _b_ .
In the discussion below, results are obtained on CUB200,
except as otherwise stated. ADE20K results are presented
in the supplementary materials. Unless otherwise noted, visualizations are based on the last convolutional layer output
of VGG16 [32], a widely used network in visualization papers. All counterfactual explanation results are presented
for two types of virtual users. Randomly chosen labels
mimic beginners while AlexNet predictions [16] mimic advanced users.


**5.1. Comparison to attributive explanations**


Figure 3 compares the discriminant explanations of (6),
to attributive explanations **a** ( _h_ _y_ _∗_ ( **x** )), for the two user
types. Several conclusions are possible: 1) discriminant
maps significantly outperform attributions for both user
types, independently of the confidence score used; 2) best
performance is achieved with the easiness score of (13); 3)
the gains are larger for expert users than beginners. This
is because the counter and predicted classes tend to be
more similar for the former and the corresponding attribution maps overlap. In this case, pure attributive explanations are very uninformative. The result also shows that
self-awareness is most useful in expert domains.


**5.2. Comparison to state of the art**


Table 1 presents a comparison between SCOUT and the
method of [12] which obtained the best results by exhaustive search, for the two user types. For fair comparison,
these experiments use the softmax score of (11), so that


|0.9<br>0.8<br>0.7<br>0.6<br>PIoU<br>0.5<br>0.4<br>0.3<br>0.2<br>0.0 0.1|Col2|Col3|
|---|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|||
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|||
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|||
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|||
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|users<br>on be<br>|ginner users<br>|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|~~on ad~~|~~vanced users~~|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>PIoU|0.2<br>0.3<br>threshold|0.4<br>0.|



Figure 4: PIoU of SCOUT as a function of the segmentation threshold on
CUB200. Left: VGG16, right: ResNet-50.


model sizes are equal for both approaches. The size of the
counterfactual region is the receptive field size of one unit
( 14 _∗_ 114 _[≈]_ [0] _[.]_ [005][ of image size on VGG16 and] 7 _∗_ 17 _[≈]_ [0] _[.]_ [02]
on ResNet-50). This was constrained by the speed of the algorithm of [12], where the counterfactual region is detected
by exhaustive feature matching.
Several conclusions could be drawn from the table. First,
SCOUT outperforms [12] in almost all cases. Second,
SCOUT is much faster, improving the speed of [12] by
1000+ times on VGG and 50+ times on ResNet. This
is because it does not require exhaustive feature matching.
These gains increase with the size of the counterfactual region, since computation time is constant for the proposed
approach but exponential on region size for [12]. Third,
due to the small size used in these experiments, PIoU is relatively low for both methods. It is, however, larger for the
proposed explanations with large gains in some cases (VGG
& advanced). Figure 4 shows that the PIoU can raise up to
0 _._ 5 for regions of 10% image size (VGG) or 20% (ResNet).
This suggests that, for regions of this size, the region pairs
have matching semantics.


**5.3. Visualizations**


Figure 5 shows two examples of counterfactual visualizations derived from the ResNet50 on CUB200. The re
gions selected in the query and counter class image are
shown in red. The true _y_ _[∗]_ and counter _y_ _[c]_ class are shown
below the images and followed by the ground truth discriminative attributes for the image pair. Note how the
proposed explanations identify semantically matched and
class-specific bird parts on both images. For example, the
throat and bill that distinguish Laysan from Sooty Albatrosses. This feedback enables a user to learn that Laysans



Table 1: Comparison to the state of the art. (IPS: images per second, implemented on NVIDIA TITAN Xp. Results are shown as mean(stddev))














True: Laysan Albatross (white throat, yellow bill)
Counter: Sooty Albatross (black throat, black bill)


True: Green Violetear (needle bill shape, black bill color)
Counter: Indigo Bunting (cone bill shape, grey bill color)

**Ours** **Goyal** _**et al**_

Figure 5: Comparison of counterfactual explanations (true and counter
classes shown below each example, and ground truth class-specific part
attributes in parenthesis).



True: Promenade deck (floor, ceiling, sea)
Counter: Bridge (tree, river, bridge)



True: Playroom (desk, ceiling)
Counter: Playground (tree, sky)



True: Bus interior (pole) True: Parking garage indoor (ceiling)
Counter: Storage room (box, closet) Counter: Street (sky, streetlight)

Figure 6: Counterfactual explanations on ADE20K.



have white throats and yellow bills, while Sootys have
black throats and bills. This is unlike the regions produced
by [12], which sometimes highlight irrelevant cues, such
as the background. Figure 6 presents similar figures for
ADE20K, where the proposed explanations tend to identify
scene-discriminative objects. For example, that a promenade deck contains objects ‘floor’, ‘ceiling’, ‘sea,’ while a
bridge scene includes ‘tree’, ‘river’ and ‘bridge’.


**5.4. Application to machine teaching**


[12] used counterfactual explanations to design an experiment to teach humans distinguish two bird classes. During
a training stage, learners are asked to classify birds. When
they make a mistake, they are shown counterfactual feedback of the type of Figure 5, using the true class as _y_ _[∗]_ and
the class they chose as _y_ _[c]_ . This helps them understand why
they chose the wrong label, and learn how to better distinguish the classes. In a test stage, learners are then asked
to classify a bird without visual aids. Experiments reported
in [12] show that this is much more effective than simply
telling them whether their answer is correct/incorrect, or
other simple training strategies. We made two modifica


Kentucky Warbler Setophaga Citrina
Figure 7: Visualization of machine teaching experiment.


tions to this set-up. The first was to replace bounding boxes
with highlighting of the counterfactual reasons, as shown in
Figure 7. We also instructed learners not to be distracted by
the darkened regions. Unlike the set-up of [12], this guarantees that they do not exploit cues outside the counterfactual regions to learn bird differences. Second, to check this,
we added two contrast experiments where 1) highlighted regions are generated randomly (without telling the learners);
2) the entire images are lighted. If these produce the same
results, one can conclude that the explanations do not promote learning.
We also chose two more difficult birds, the Setophaga
Citrina and the Kentucky Warbler (see Figure 7), than those
used in [12]. This is because these classes have large intraclass diversity. The two classes also cannot be distinguished
by color alone, unlike those used in [12]. The experiment
has three steps. The first is a pre-learning test, where students are asked to classify 20 examples of the two classes,
or choose a ‘Don’t know’ option. The second is a learning stage, where counterfactual explanations are provided
for 10 bird pairs. The third is a post-learning test, where
students are asked to answer 20 binary classification questions. In this experiment, all students chose ‘Don’t know’ in
the pre-learning test. However, after the learning step, they
achieved 95% mean accuracy, compared to 60% (random
highlighted regions) and 77% (entire images lighted) in the
contrast settings. These results suggest that SCOUT can
help teach non-expert humans distinguish categories from
an expert domain.


**6. Conclusion**


In this work, we proposed a new family of discriminant
explanations, which leverage self-awareness and bridge the
gap between attributions and counterfactuals. A quantitative evaluation protocol was also proposed. Experiments
under both this protocol and machine teaching experiments
show that both the proposed discriminant and counterfactual explanations achieve much better performance than existing attributive and counterfactual methods.


**Acknowledgements** This work was partially funded by
NSF awards IIS-1637941, IIS-1924937, and NVIDIA GPU

donations.


**References**


[1] Marco Ancona, Enea Ceolini, Cengiz Oztireli, and Markus [¨]
Gross. A unified view of gradient-based attribution methods
for deep neural networks. In _NIPS 2017-Workshop on In-_
_terpreting, Explaining and Visualizing Deep Learning_ . ETH
Zurich, 2017.

[2] Lisa Anne Hendricks, Ronghang Hu, Trevor Darrell, and
Zeynep Akata. Grounding visual explanations. In _The Eu-_
_ropean Conference on Computer Vision (ECCV)_, September
2018.

[3] Sebastian Bach, Alexander Binder, Gr´egoire Montavon,
Frederick Klauschen, Klaus-Robert M¨uller, and Wojciech
Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. _PloS one_,
10(7):e0130140, 2015.

[4] Sumit Basu and Janara Christensen. Teaching classification
boundaries to humans. In _Twenty-Seventh AAAI Conference_
_on Artificial Intelligence_, 2013.

[5] Abhijit Bendale and Terrance E Boult. Towards open set
deep networks. In _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pages 1563–1572,
2016.

[6] Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic,
Luc Van Gool, and Marie-Francine Moens. Talk2car:
Taking control of your self-driving car. _arXiv preprint_
_arXiv:1909.10838_, 2019.

[7] Terrance DeVries and Graham W Taylor. Learning confidence for out-of-distribution detection in neural networks.

_arXiv preprint arXiv:1802.04865_, 2018.

[8] Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen
Tu, Paishun Ting, Karthikeyan Shanmugam, and Payel Das.
Explanations based on the missing: Towards contrastive explanations with pertinent negatives. In _Advances in Neural_
_Information Processing Systems_, pages 592–603, 2018.

[9] Dominik Maria Endres and Johannes E Schindelin. A new
metric for probability distributions. _IEEE Transactions on_
_Information theory_, 2003.

[10] Yonatan Geifman and Ran El-Yaniv. Selective classification
for deep neural networks. In _Advances in neural information_
_processing systems_, pages 4878–4887, 2017.

[11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial nets. In _Advances_
_in neural information processing systems_, pages 2672–2680,
2014.

[12] Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh,
and Stefan Lee. Counterfactual visual explanations. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, _Pro-_
_ceedings of the 36th International Conference on Machine_
_Learning_, volume 97 of _Proceedings of Machine Learning_
_Research_, pages 2376–2384, 2019.

[13] Lisa Anne Hendricks, Ronghang Hu, Trevor Darrell, and
Zeynep Akata. Generating counterfactual explanations with
natural language. _arXiv preprint arXiv:1806.09809_, 2018.

[14] Dan Hendrycks and Kevin Gimpel. A baseline for detecting
misclassified and out-of-distribution examples in neural net


works. _Proceedings of International Conference on Learning_
_Representations_, 2017.

[15] Dan Hendrycks, Mantas Mazeika, and Thomas G Dietterich.
Deep anomaly detection with outlier exposure. _Proceedings_
_of International Conference on Learning Representations_,
2019.

[16] Alex Krizhevsky and Geoffrey Hinton. Learning multiple
layers of features from tiny images. Technical report, Citeseer, 2009.

[17] Ronan Le Hy, Anthony Arrigoni, Pierre Bessi`ere, and Olivier
Lebeltel. Teaching bayesian behaviours to video game characters. _Robotics and Autonomous Systems_, 47(2-3):177–185,
2004.

[18] Kimin Lee, Honglak Lee, Kibok Lee, and Jinwoo Shin.
Training confidence-calibrated classifiers for detecting outof-distribution samples. _Proceedings of International Con-_
_ference on Learning Representations_, 2018.

[19] Yi Li and Nuno Vasconcelos. Background data resampling
for outlier-aware classification. In _Proceedings of the IEEE_
_Conference on Computer Vision and Pattern Recognition_,
2020.

[20] Shiyu Liang, Yixuan Li, and R Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. _Proceedings of International Conference on Learning_
_Representations_, 2018.

[21] Shusen Liu, Bhavya Kailkhura, Donald Loveland, and Yong
Han. Generative counterfactual introspection for explainable
deep learning. _arXiv preprint arXiv:1907.03077_, 2019.

[22] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
Deep learning face attributes in the wild. In _Proceedings of_
_International Conference on Computer Vision (ICCV)_, December 2015.

[23] Ronny Luss, Pin-Yu Chen, Amit Dhurandhar, Prasanna Sattigeri, Karthikeyan Shanmugam, and Chun-Chen Tu. Generating contrastive explanations with monotonic attribute functions. _arXiv preprint arXiv:1905.12698_, 2019.

[24] Tim Miller. Contrastive explanation: A structural-model approach. _arXiv preprint arXiv:1811.03163_, 2018.

[25] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas J Guibas, and Jascha SohlDickstein. Deep knowledge tracing. In _Advances in neural_
_information processing systems_, pages 505–513, 2015.

[26] Shubham Rathi. Generating counterfactual and contrastive
explanations using shap. _arXiv preprint arXiv:1906.09293_,
2019.

[27] Walter J Scheirer, Anderson de Rezende Rocha, Archana
Sapkota, and Terrance E Boult. Toward open set recognition. _IEEE transactions on pattern analysis and machine_
_intelligence_, 35(7):1757–1772, 2012.

[28] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das,
Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.
Grad-cam: Visual explanations from deep networks via
gradient-based localization. In _Proceedings of the IEEE In-_
_ternational Conference on Computer Vision_, pages 618–626,
2017.

[29] Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje.
Learning important features through propagating activation


differences. In _Proceedings of the 34th International Con-_
_ference on Machine Learning-Volume 70_, pages 3145–3153.
JMLR. org, 2017.

[30] Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, and
Anshul Kundaje. Not just a black box: Learning important
features through propagating activation differences. _arXiv_
_preprint arXiv:1605.01713_, 2016.

[31] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman.
Deep inside convolutional networks: Visualising image classification models and saliency maps. _Workshop at Interna-_
_tional Conference on Learning Representations_, 2014.

[32] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv_
_preprint arXiv:1409.1556_, 2014.

[33] Adish Singla, Ilija Bogunovic, G´abor Bart´ok, Amin Karbasi,
and Andreas Krause. Near-optimally teaching the crowd to
classify. In _ICML_, page 3, 2014.

[34] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic
attribution for deep networks. In _Proceedings of the 34th_
_International Conference on Machine Learning-Volume 70_,
pages 3319–3328. JMLR. org, 2017.

[35] Arnaud Van Looveren and Janis Klaise. Interpretable counterfactual explanations guided by prototypes. _arXiv preprint_
_arXiv:1907.02584_, 2019.

[36] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the gpdr. _Harv. JL & Tech._, 31:841,
2017.

[37] Pei Wang and Nuno Vasconcelos. Towards realistic predictors. In _The European Conference on Computer Vision_, 2018.

[38] Pei Wang and Nuno Vasconcelos. Deliberative explanations:
visualizing network insecurities. In _Advances in Neural In-_
_formation Processing Systems 32_, pages 1374–1385, 2019.

[39] Xin Wang, Yujia Luo, Daniel Crankshaw, Alexey Tumanov,
Fisher Yu, and Joseph E Gonzalez. Idk cascades: Fast
deep learning by learning not to overthink. _arXiv preprint_
_arXiv:1706.00885_, 2017.

[40] P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Belongie, and P. Perona. Caltech-UCSD Birds 200. Technical
Report CNS-TR-2010-001, California Institute of Technology, 2010.

[41] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva,
and Antonio Torralba. Learning deep features for discriminative localization. In _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pages 2921–2929,
2016.

[42] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela
Barriuso, and Antonio Torralba. Scene parsing through
ade20k dataset. In _Proceedings of the IEEE Conference on_
_Computer Vision and Pattern Recognition_, 2017.

[43] Xiaojin Zhu, Adish Singla, Sandra Zilles, and Anna N. Rafferty. An overview of machine teaching. _arXiv preprint_
_arXiv:1801.05927_, 2018.

[44] Luisa M Zintgraf, Taco S Cohen, Tameem Adel, and Max
Welling. Visualizing deep neural network decisions: Prediction difference analysis. _ICLR_, 2017.


