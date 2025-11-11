Published as a conference paper at ICLR 2022

## L EVERAGING U NLABELED D ATA TO P REDICT - O UT OF -D ISTRIBUTION P ERFORMANCE



**Saurabh Garg** [˚]
Carnegie Mellon University
[sgarg2@andrew.cmu.edu](mailto:sgarg2@andrew.cmu.edu)



**Sivaraman Balakrishnan**
Carnegie Mellon University

[sbalakri@andrew.cmu.edu](mailto:sbalakri@andrew.cmu.edu)



**Zachary C. Lipton**
Carnegie Mellon University
[zlipton@andrew.cmu.edu](mailto:zlipton@andrew.cmu.edu)



**Behnam Neyshabur**
Google Research, Blueshift team
[neyshabur@google.com](mailto:neyshabur@google.com)



**Hanie Sedghi**
Google Research, Brain team
[hsedghi@google.com](mailto:hsedghi@google.com)



A BSTRACT


Real-world machine learning deployments are characterized by mismatches between the source (training) and target (test) distributions that may cause performance drops. In this work, we investigate methods for predicting the target domain
accuracy using only labeled source data and unlabeled target data. We propose Average Thresholded Confidence (ATC), a practical method that learns a _threshold_ on
the model’s confidence, predicting accuracy as the fraction of unlabeled examples
for which model confidence exceeds that threshold. ATC outperforms previous
methods across several model architectures, types of distribution shifts (e.g., due to
synthetic corruptions, dataset reproduction, or novel subpopulations), and datasets
(W ILDS, ImageNet, B REEDS, CIFAR, and MNIST). In our experiments, ATC
estimates target performance 2–4ˆ more accurately than prior methods. We also
explore the theoretical foundations of the problem, proving that, in general, identifying the accuracy is just as hard as identifying the optimal predictor and thus, the
efficacy of any method rests upon (perhaps unstated) assumptions on the nature
of the shift. Finally, analyzing our method on some toy distributions, we provide
insights concerning when it works [1] .


1 I NTRODUCTION


Machine learning models deployed in the real world typically encounter examples from previously
unseen distributions. While the IID assumption enables us to evaluate models using held-out data
from the _source_ distribution (from which training data is sampled), this estimate is no longer valid
in presence of a distribution shift. Moreover, under such shifts, model accuracy tends to degrade
(Szegedy et al., 2014; Recht et al., 2019; Koh et al., 2021). Commonly, the only data available to
the practitioner are a labeled training set (source) and unlabeled deployment-time data which makes
the problem more difficult. In this setting, detecting shifts in the distribution of covariates is known
to be possible (but difficult) in theory (Ramdas et al., 2015), and in practice (Rabanser et al., 2018).
However, producing an optimal predictor using only labeled source and unlabeled target data is
well-known to be impossible absent further assumptions (Ben-David et al., 2010; Lipton et al., 2018).


Two vital questions that remain are: (i) the precise conditions under which we can estimate a classifier’s target-domain accuracy; and (ii) which methods are most practically useful. To begin, the
straightforward way to assess the performance of a model under distribution shift would be to collect
labeled (target domain) examples and then to evaluate the model on that data. However, collecting fresh labeled data from the target distribution is prohibitively expensive and time-consuming,
especially if the target distribution is non-stationary. Hence, instead of using labeled data, we aim
to use unlabeled data from the target distribution, that is comparatively abundant, to predict model
performance. Note that in this work, our focus is _not_ to improve performance on the target but, rather,
to estimate the accuracy on the target for a given classifier.


˚ Work done in part while Saurabh Garg was interning at Google
1
[Code is available at https://github.com/saurabhgarg1996/ATC_code.](https://github.com/saurabhgarg1996/ATC_code)


1


Published as a conference paper at ICLR 2022


Figure 1: _Illustration of our proposed method ATC._ **Left** : using source domain validation data, we
identify a _threshold_ on a score (e.g. negative entropy) computed on model confidence such that
fraction of examples above the threshold matches the validation set accuracy. ATC estimates accuracy
on unlabeled target data as the fraction of examples with the score above the threshold. Interestingly,
this threshold yields accurate estimates on a wide set of target distributions resulting from natural
and synthetic shifts. **Right** : Efficacy of ATC over previously proposed approaches on our testbed
with a post-hoc calibrated model. To obtain errors on the same scale, we rescale all errors with
Average Confidence (AC) error. Lower estimation error is better. See Table 1 for exact numbers and
comparison on various types of distribution shift. See Sec. 5 for details on our testbed.


Recently, numerous methods have been proposed for this purpose (Deng & Zheng, 2021; Chen et al.,
2021b; Jiang et al., 2021; Deng et al., 2021; Guillory et al., 2021). These methods either require
calibration on the target domain to yield consistent estimates (Jiang et al., 2021; Guillory et al.,
2021) or additional labeled data from several target domains to learn a linear regression function on a
distributional distance that then predicts model performance (Deng et al., 2021; Deng & Zheng, 2021;
Guillory et al., 2021). However, methods that require calibration on the target domain typically yield
poor estimates since deep models trained and calibrated on source data are not, in general, calibrated
on a (previously unseen) target domain (Ovadia et al., 2019). Besides, methods that leverage labeled
data from target domains rely on the fact that unseen target domains exhibit strong linear correlation
with seen target domains on the underlying distance measure and, hence, can be rendered ineffective
when such target domains with labeled data are unavailable (in Sec. 5.1 we demonstrate such a failure
on a real-world distribution shift problem). Therefore, throughout the paper, we assume access to
labeled source data and only unlabeled data from target domain(s).


In this work, we first show that absent assumptions on the source classifier or the nature of the shift,
no method of estimating accuracy will work generally (even in non-contrived settings). To estimate
accuracy on target domain _perfectly_, we highlight that even given perfect knowledge of the labeled
source distribution (i.e., _p_ _s_ p _x, y_ q ) and unlabeled target distribution (i.e., _p_ _t_ p _x_ q ), we need restrictions
on the nature of the shift such that we can uniquely identify the target conditional _p_ _t_ p _y_ | _x_ q . Thus, in
general, identifying the accuracy of the classifier is as hard as identifying the optimal predictor.


Second, motivated by the superiority of methods that use maximum softmax probability (or logit) of
a model for Out-Of-Distribution (OOD) detection (Hendrycks & Gimpel, 2016; Hendrycks et al.,
2019), we propose a simple method that leverages softmax probability to predict model performance.
Our method, Average Thresholded Confidence (ATC), learns a threshold on a score (e.g., maximum
confidence or negative entropy) of model confidence on validation source data and predicts target
domain accuracy as the fraction of unlabeled target points that receive a score above that threshold.
ATC selects a threshold on validation source data such that the fraction of source examples that
receive the score above the threshold match the accuracy of those examples. Our primary contribution
in ATC is the proposal of obtaining the threshold and observing its efficacy on (practical) accuracy
estimation. Importantly, our work takes a step forward in positively answering the question raised in
Deng & Zheng (2021); Deng et al. (2021) about a practical strategy to select a threshold that enables
accuracy prediction with thresholded model confidence.


2


Published as a conference paper at ICLR 2022


ATC is simple to implement with existing frameworks, compatible with arbitrary model classes, and
dominates other contemporary methods. Across several model architectures on a range of benchmark
vision and language datasets, we verify that ATC outperforms prior methods by at least 2 – 4ˆ in
predicting target accuracy on a variety of distribution shifts. In particular, we consider shifts due
to common corruptions (e.g., ImageNet-C), natural distribution shifts due to dataset reproduction
(e.g., ImageNet-v2, ImageNet-R), shifts due to novel subpopulations (e.g., B REEDS ), and distribution
shifts faced in the wild (e.g., W ILDS ).


As a starting point for theory development, we investigate ATC on a simple toy model that models
distribution shift with varying proportions of the population with spurious features, as in Nagarajan
et al. (2020). Finally, we note that although ATC achieves superior performance in our empirical
evaluation, like all methods, it must fail (returns inconsistent estimates) on certain types of distribution
shifts, per our impossibility result.


2 P RIOR W ORK


**Out-of-distribution detection.** The main goal of OOD detection is to identify previously unseen
examples, i.e., samples out of the support of training distribution. To accomplish this, modern methods
utilize confidence or features learned by a deep network trained on some source data. Hendrycks &
Gimpel (2016); Geifman & El-Yaniv (2017) used the confidence score of an (already) trained deep
model to identify OOD points. Lakshminarayanan et al. (2016) use entropy of an ensemble model to
evaluate prediction uncertainty on OOD points. To improve OOD detection with model confidence,
Liang et al. (2017) propose to use temperature scaling and input perturbations. Jiang et al. (2018)
propose to use scores based on the relative distance of the predicted class to the second class. Recently,
residual flow-based methods were used to obtain a density model for OOD detection (Zhang et al.,
2020). Ji et al. (2021) proposed a method based on subfunction error bounds to compute unreliability
per sample. Refer to Ovadia et al. (2019); Ji et al. (2021) for an overview and comparison of methods
for prediction uncertainty on OOD data.


**Predicting model generalization.** Understanding generalization capabilities of overparameterized
models on in-distribution data using conventional machine learning tools has been a focus of a long
line of work; representative research includes Neyshabur et al. (2015; 2017); Neyshabur (2017);
Neyshabur et al. (2018); Dziugaite & Roy (2017); Bartlett et al. (2017); Zhou et al. (2018); Long
& Sedghi (2019); Nagarajan & Kolter (2019a). At a high level, this line of research bounds the
generalization gap directly with complexity measures calculated on the trained model. However, these
bounds typically remain numerically loose relative to the true generalization error (Zhang et al., 2016;
Nagarajan & Kolter, 2019b). On the other hand, another line of research departs from complexitybased approaches to use unseen unlabeled data to predict in-distribution generalization (Platanios
et al., 2016; 2017; Garg et al., 2021; Jiang et al., 2021).


Relevant to our work are methods for predicting the error of a classifier on OOD data based on
unlabeled data from the target (OOD) domain. These methods can be characterized into two broad
categories: (i) Methods which explicitly predict correctness of the model on individual unlabeled
points (Deng & Zheng, 2021; Jiang et al., 2021; Deng et al., 2021; Chen et al., 2021a); and (ii)
Methods which directly obtain an estimate of error with unlabeled OOD data without making a
point-wise prediction (Chen et al., 2021b; Guillory et al., 2021; Chuang et al., 2020).


To achieve a consistent estimate of the target accuracy, Jiang et al. (2021); Guillory et al. (2021)
require calibration on target domain. However, these methods typically yield poor estimates as
deep models trained and calibrated on some source data are seldom calibrated on previously unseen
domains (Ovadia et al., 2019). Additionally, Deng & Zheng (2021); Guillory et al. (2021) derive
model-based distribution statistics on unlabeled target set that correlate with the target accuracy and
propose to use a subset of _labeled_ target domains to learn a (linear) regression function that predicts
model performance. However, there are two drawbacks with this approach: (i) the correlation of
these distribution statistics can vary substantially as we consider different nature of shifts (refer to
Sec. 5.1, where we empirically demonstrate this failure); (ii) even if there exists a (hypothetical)
statistic with strong correlations, obtaining labeled target domains (even simulated ones) with strong
correlations would require significant _a priori_ knowledge about the nature of shift that, in general,
might not be available before models are deployed in the wild. Nonetheless, in our work, we only
assume access to labeled data from the source domain presuming no access to labeled target domains
or information about how to simulate them.


3


Published as a conference paper at ICLR 2022


Moreover, unlike the parallel work of Deng et al. (2021), we do not focus on methods that alter the
training on source data to aid accuracy prediction on the target data. Chen et al. (2021b) propose
an importance re-weighting based approach that leverages (additional) information about the axis
along which distribution is shifting in form of “slicing functions”. In our work, we make comparisons
with importance re-weighting baseline from Chen et al. (2021b) as we do not have any additional
information about the axis along which the distribution is shifting.


3 P ROBLEM S ETUP


**Notation.** By ||¨||, and x¨ _,_ ¨y we denote the Euclidean norm and inner product, respectively. For a
vector _v_ P R _[d]_, we use _v_ _j_ to denote its _j_ [th] entry, and for an event _E_ we let I r _E_ s denote the binary
indicator of the event.


Suppose we have a multi-class classification problem with the input domain _X_ Ď R _[d]_ and label
space _Y_ “ t1 _,_ 2 _, . . ., k_ u . For binary classification, we use _Y_ “ t0 _,_ 1u . By _D_ [S] and _D_ [T], we denote
source and target distribution over _X_ ˆ _Y_ . For distributions _D_ [S] and _D_ [T], we define _p_ S or _p_ T as
the corresponding probability density (or mass) functions. A dataset _S_ :“ tp _x_ _i_ _, y_ _i_ qu _[n]_ _i_ “1 [„ p] _[D]_ [S] [q] _[n]_

contains _n_ points sampled i.i.d. from _D_ [S] . Let _F_ be a class of hypotheses mapping _X_ to ∆ _[k]_ [´][1] where
∆ _[k]_ [´][1] is a simplex in _k_ dimensions. Given a classifier _f_ P _F_ and datum p _x, y_ q, we denote the 0-1
error (i.e., classification error) on that point by _E_ p _f_ p _x_ q _, y_ q :“ I “ _y_ R arg max _j_ P _Y_ _f_ _j_ p _x_ q‰ . Given a
model _f_ P _F_, our goal in this work is to understand the performance of _f_ on _D_ [T] without access to
labeled data from _D_ [T] . Note that our goal is not to adapt the model to the target data. Concretely,
we aim to predict accuracy of _f_ on _D_ [T] . Throughout this paper, we assume we have access to the
following: (i) model _f_ ; (ii) previously-unseen (validation) data from _D_ [S] ; and (iii) unlabeled data
from target distribution _D_ [T] .


3.1 A CCURACY E STIMATION : P OSSIBILITY AND I MPOSSIBILITY R ESULTS


First, we investigate the question of when it is possible to estimate the target accuracy of an arbitrary
classifier, even given knowledge of the full source distribution _p_ _s_ p _x, y_ q and target marginal _p_ _t_ p _x_ q .
Absent assumptions on the nature of shift, estimating target accuracy is impossible. Even given
access to _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q, the problem is fundamentally unidentifiable because _p_ _t_ p _y_ | _x_ q can shift
arbitrarily. In the following proposition, we show that absent assumptions on the classifier _f_ (i.e.,
when _f_ can be any classifier in the space of all classifiers on _X_ ), we can estimate accuracy on the
target data iff assumptions on the nature of the shift, together with _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q, uniquely
identify the (unknown) target conditional _p_ _t_ p _y_ | _x_ q. We relegate proofs from this section to App. A.
**Proposition 1.** _Absent further assumptions, accuracy on the target is identifiable iff_ _p_ _t_ p _y_ | _x_ q _is_
_uniquely identified given p_ _s_ p _x, y_ q _and p_ _t_ p _x_ q _._


Proposition 1 states that we need enough constraints on nature of shift such that _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q
identifies unique _p_ _t_ p _y_ | _x_ q . It also states that under some assumptions on the nature of the shift, we
can hope to estimate the model’s accuracy on target data. We will illustrate this on two common
assumptions made in domain adaptation literature: (i) covariate shift (Heckman, 1977; Shimodaira,
2000) and (ii) label shift (Saerens et al., 2002; Zhang et al., 2013; Lipton et al., 2018). Under
covariate shift assumption, that the target marginal support **supp** p _p_ _t_ p _x_ qq is a subset of the source
marginal support **supp** p _p_ _s_ p _x_ qq and that the conditional distribution of labels given inputs does not
change within support, i.e., _p_ _s_ p _y_ | _x_ q “ _p_ _t_ p _y_ | _x_ q, which, trivially, identifies a unique target conditional
_p_ _t_ p _y_ | _x_ q . Under label shift, the reverse holds, i.e., the class-conditional distribution does not change
( _p_ _s_ p _x_ | _y_ q “ _p_ _t_ p _x_ | _y_ q ) and, again, information about _p_ _t_ p _x_ q uniquely determines the target conditional
_p_ _t_ p _y_ | _x_ q (Lipton et al., 2018; Garg et al., 2020). In these settings, one can estimate an arbitrary
classifier’s accuracy on the target domain either by using importance re-weighting with the ratio
_p_ _t_ p _x_ q{ _p_ _s_ p _x_ q in case of covariate shift or by using importance re-weighting with the ratio _p_ _t_ p _y_ q{ _p_ _s_ p _y_ q
in case of label shift. While importance ratios in the former case can be obtained directly when _p_ _t_ p _x_ q
and _p_ _s_ p _x_ q are known, the importance ratios in the latter case can be obtained by using techniques from
Saerens et al. (2002); Lipton et al. (2018); Azizzadenesheli et al. (2019); Alexandari et al. (2019).
In App. B,we explore accuracy estimation in the setting of these shifts and present extensions to
generalized notions of label shift (Tachet des Combes et al., 2020) and covariate shift (Rojas-Carulla
et al., 2018).


As a corollary of Proposition 1, we now present a simple impossibility result, demonstrating that no
single method can work for all families of distribution shift.


4


Published as a conference paper at ICLR 2022


**Corollary 1.** _Absent assumptions on the classifier_ _f_ _, no method of estimating accuracy will work in_
_all scenarios, i.e., for different nature of distribution shifts._


Intuitively, this result states that every method of estimating accuracy on target data is tied up with
some assumption on the nature of the shift and might not be useful for estimating accuracy under
a different assumption on the nature of the shift. For illustration, consider a setting where we have
access to distribution _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q . Additionally, assume that the distribution can shift only
due to covariate shift or label shift without any knowledge about which one. Then Corollary 1 says
that it is impossible to have a single method that will simultaneously for both label shift and covariate
shift as in the following example (we spell out the details in App. A):


**Example 1.** Assume binary classification with _p_ _s_ p _x_ q “ _α_ ¨ _φ_ p _µ_ 1 q ` p1 ´ _α_ q ¨ _φ_ p _µ_ 2 q,
_p_ _s_ p _x_ | _y_ “ 0q “ _φ_ p _µ_ 1 q, _p_ _s_ p _x_ | _y_ “ 1q “ _φ_ p _µ_ 2 q, and _p_ _t_ p _x_ q “ _β_ ¨ _φ_ p _µ_ 1 q ` p1 ´ _β_ q ¨ _φ_ p _µ_ 2 q
where _φ_ p _µ_ q “ _N_ p _µ,_ 1q, _α, β_ P p0 _,_ 1q, and _α_ ‰ _β_ . Error of a classifier _f_ on target
_p_ _t_ p _x_ q
data is given by _E_ 1 “ E p _x,y_ q„ _p_ _s_ p _x,y_ q ” _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] ı under covariate shift and by _E_ 2 “

E p _x,y_ q„ _p_ _s_ p _x,y_ q ”´ _αβ_ [I][ r] _[y]_ [ “][ 0][s `] 1 [1] ´ [´] _α_ _[β]_ [I][ r] _[y]_ [ “][ 1][s] ¯ I r _f_ p _x_ q ‰ _y_ sı under label shift. In App. A, we show

that _E_ 1 ‰ _E_ 2 for all _f_ . Thus, given access to _p_ _s_ p _x, y_ q, and _p_ _t_ p _x_ q, any method that consistently
estimates error of a classifer under covariate shift will give an incorrect estimate of error under label
shift and vice-versa. The reason is that the same _p_ _t_ p _x_ q and _p_ _s_ p _x, y_ q can correspond to error _E_ 1 (under
covariate shift) or error _E_ 2 (under label shift) and determining which scenario one faces requires
further assumptions on the nature of shift.


4 P REDICTING ACCURACY WITH A VERAGE T HRESHOLDED C ONFIDENCE


In this section, we present our method ATC that leverages a black box classifier _f_ and (labeled)
validation source data to predict accuracy on target domain given access to unlabeled target data.
Throughout the discussion, we assume that the classifier _f_ is fixed.


Before presenting our method, we introduce some terminology. Define a score function _s_ : ∆ _[k]_ [´][1] Ñ
R that takes in the softmax prediction of the function _f_ and outputs a scalar. We want a score function
such that if the score function takes a high value at a datum p _x, y_ q then _f_ is likely to be correct. In
this work, we explore two such score functions: (i) Maximum confidence, i.e., _s_ p _f_ p _x_ qq “ max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ [q] [;]

and (ii) Negative Entropy, i.e., _s_ p _f_ p _x_ qq “ [ř] _j_ _[f]_ _[j]_ [p] _[x]_ [q][ log][p] _[f]_ _[j]_ [p] _[x]_ [qq] [. Our method identifies a threshold] _[ t]_

on source data _D_ [S] such that the expected number of points that obtain a score less than _t_ match the
error of _f_ on _D_ [S], i.e.,


E _x_ „ _D_ S rI r _s_ p _f_ p _x_ qq ă _t_ ss “ E p _x,y_ q„ _D_ S I arg max _f_ _j_ p _x_ q ‰ _y_ _,_ (1)
„ „ _j_ P _Y_ 


and then our error estimate ATC _D_ T p _s_ q on the target domain _D_ [T] is given by the expected number of
target points that obtain a score less than _t_, i.e.,


ATC _D_ T p _s_ q “ E _x_ „ _D_ T rI r _s_ p _f_ p _x_ qq ă _t_ ss _._ (2)


In short, in (1), ATC selects a threshold on the score function such that the error in the source domain
matches the expected number of points that receive a score below _t_ and in (2), ATC predicts error
on the target domain as the fraction of unlabeled points that obtain a score below that threshold _t_ .
Note that, in principle, there exists a different threshold _t_ [1] on the target distribution _D_ [T] such that (1)
is satisfied on _D_ [T] . However, in our experiments, the same threshold performs remarkably well. The
main empirical contribution of our work is to show that the threshold obtained with (1) might be used
effectively in condunction with modern deep networks in a wide range of settings to estimate error on
the target data. In practice, to obtain the threshold with ATC, we minimize the difference between the
expression on two sides of (1) using finite samples. In the next section, we show that ATC precisely
predicts accuracy on the OOD data on the desired line _y_ “ _x_ . In App. C, we discuss an alternate
interpretation of the method and make connections with OOD detection methods.


5 E XPERIMENTS


We now empirical evaluate ATC and compare it with existing methods. In each of our main
experiment, keeping the underlying model fixed, we vary target datasets and make a prediction


5


Published as a conference paper at ICLR 2022



90

80

70

60

50

40

30



CIFAR10

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||DOC<br>GDE<br>||
|||~~ATC (O~~|~~urs)~~|



40 60 80
OOD Accuracy



80


60


40


20



ImageNet200

|Col1|Col2|
|---|---|
||DOC<br>~~GDE~~<br>ATC (Ours)|



20 40 60 80
OOD Accuracy



80


60


40


20


0



Living17-Breeds

|Col1|Col2|
|---|---|
||DOC<br>~~GDE~~<br>ATC (Ours)|



0 20 40 60 80
OOD Accuracy



Figure 2: _Scatter plot of predicted accuracy versus (true) OOD accuracy._ Each point denotes a
different OOD dataset, all evaluated with the same DenseNet121 model. We only plot the best three
methods. With ATC (ours), we refer to ATC-NE. We observe that ATC significantly outperforms
other methods and with ATC, we recover the desired line _y_ “ _x_ with a robust linear fit. Aggregated
estimation error in Table 1 and plots for other datasets and architectures in App. H.


of the target accuracy with various methods given access to only unlabeled data from the target.
Unless noted otherwise, all models are trained only on samples from the source distribution with the
main exception of pre-training on a different distribution. We use labeled examples from the target
distribution to only obtain true error estimates.


**Datasets.** First, we consider synthetic shifts induced due to different visual corruptions (e.g., shot
noise, motion blur etc.) under ImageNet-C (Hendrycks & Dietterich, 2019). Next, we consider
natural shifts due to differences in the data collection process of ImageNet (Russakovsky et al., 2015),
e.g, ImageNetv2 (Recht et al., 2019). We also consider images with artistic renditions of object
classes, i.e., ImageNet-R (Hendrycks et al., 2021) and ImageNet-Sketch (Wang et al., 2019). Note
that renditions dataset only contains a subset 200 classes from ImageNet. To include renditions
dataset in our testbed, we include results on ImageNet restricted to these 200 classes (which we call
ImageNet-200) along with full ImageNet.


Second, we consider B REEDS (Santurkar et al., 2020) to assess robustness to subpopulation shifts, in
particular, to understand how accuracy estimation methods behave when novel subpopulations not
observed during training are introduced. B REEDS leverages class hierarchy in ImageNet to create 4
datasets E NTITY -13, E NTITY -30, L IVING -17, N ON - LIVING -26. We focus on natural and synthetic
shifts as in ImageNet on same and different subpopulations in BREEDs. Third, from W ILDS (Koh
et al., 2021) benchmark, we consider FMoW-W ILDS (Christie et al., 2018), RxRx1-W ILDS (Taylor
et al., 2019), Amazon-W ILDS (Ni et al., 2019), CivilComments-W ILDS (Borkan et al., 2019) to
consider distribution shifts faced in the wild.


Finally, similar to ImageNet, we consider (i) synthetic shifts (CIFAR-10-C) due to common corruptions; and (ii) natural shift (i.e., CIFARv2 (Recht et al., 2018)) on CIFAR-10 (Krizhevsky & Hinton,
2009). On CIFAR-100, we just have synthetic shifts due to common corruptions. For completeness,
we also consider natural shifts on MNIST (LeCun et al., 1998) as in the prior work (Deng & Zheng,
2021). We use three real shifted datasets, i.e., USPS (Hull, 1994), SVHN (Netzer et al., 2011) and
QMNIST (Yadav & Bottou, 2019). We give a detailed overview of our setup in App. F.


**Architectures and Evaluation.** For ImageNet, B REEDS, CIFAR, FMoW-W ILDS, RxRx1-W ILDS
datasets, we use DenseNet121 (Huang et al., 2017) and ResNet50 (He et al., 2016) architectures. For
Amazon-W ILDS and CivilComments-W ILDS, we fine-tune a DistilBERT-base-uncased (Sanh et al.,
2019) model. For MNIST, we train a fully connected multilayer perceptron. We use standard training
with benchmarked hyperparameters. To compare methods, we report average absolute difference
between the true accuracy on the target data and the estimated accuracy on the same unlabeled
examples. We refer to this metric as Mean Absolute estimation Error (MAE). Along with MAE,
we also show scatter plots to visualize performance at individual target sets. Refer to App. G for
additional details on the setup.


**Methods** With ATC-NE, we denote ATC with negative entropy score function and with ATC-MC,
we denote ATC with maximum confidence score function. For all methods, we implement _post-hoc_
calibration on validation source data with Temperature Scaling (TS; Guo et al. (2017)). Below we
briefly discuss baselines methods compared in our work and relegate details to App. E.


6


Published as a conference paper at ICLR 2022


IM AC DOC GDE ATC-MC (Ours) ATC-NE (Ours)
Dataset Shift
Pre T Post T Pre T Post T Pre T Post T Post T Pre T Post T Pre T Post T


Natural 6 _._ 60 5 _._ 74 9 _._ 88 6 _._ 89 7 _._ 25 6 _._ 07 4 _._ 77 3 _._ 21 3 _._ 02 2 _._ 99 **2** _._ **85**
CIFAR10
Synthetic 12 _._ 33 10 _._ 20 16 _._ 50 11 _._ 91 13 _._ 87 11 _._ 08 6 _._ 55 4 _._ 65 4 _._ 25 4 _._ 21 **3** _._ **87**


CIFAR100 Synthetic 13 _._ 69 11 _._ 51 23 _._ 61 13 _._ 10 14 _._ 60 10 _._ 14 9 _._ 85 5 _._ 50 **4** _._ **75** **4** _._ **72** 4 _._ 94


Natural 12 _._ 37 8 _._ 19 22 _._ 07 8 _._ 61 15 _._ 17 7 _._ 81 5 _._ 13 4 _._ 37 2 _._ 04 3 _._ 79 **1** _._ **45**
ImageNet200
Synthetic 19 _._ 86 12 _._ 94 32 _._ 44 13 _._ 35 25 _._ 02 12 _._ 38 5 _._ 41 5 _._ 93 3 _._ 09 5 _._ 00 **2** _._ **68**


Natural 7 _._ 77 6 _._ 50 18 _._ 13 6 _._ 02 8 _._ 13 5 _._ 76 6 _._ 23 3 _._ 88 2 _._ 17 2 _._ 06 **0** _._ **80**
ImageNet
Synthetic 13 _._ 39 10 _._ 12 24 _._ 62 8 _._ 51 13 _._ 55 7 _._ 90 6 _._ 32 3 _._ 34 **2** _._ **53** **2** _._ **61** 4 _._ 89


FMoW- WILDS Natural 5 _._ 53 4 _._ 31 33 _._ 53 12 _._ 84 5 _._ 94 4 _._ 45 5 _._ 74 3 _._ 06 **2** _._ **70** 3 _._ 02 **2** _._ **72**


RxRx1- WILDS Natural 5 _._ 80 5 _._ 72 7 _._ 90 4 _._ 84 5 _._ 98 5 _._ 98 6 _._ 03 4 _._ 66 **4** _._ **56** **4** _._ **41** **4** _._ **47**


Amazon- WILDS Natural 2 _._ 40 2 _._ 29 8 _._ 01 2 _._ 38 2 _._ 40 2 _._ 28 17 _._ 87 1 _._ 65 **1** _._ **62** **1** _._ **60** **1** _._ **59**


CivilCom.- WILDS Natural 12 _._ 64 10 _._ 80 16 _._ 76 11 _._ 03 13 _._ 31 10 _._ 99 16 _._ 65 **7** _._ **14**


MNIST Natural 18 _._ 48 15 _._ 99 21 _._ 17 14 _._ 81 20 _._ 19 14 _._ 56 24 _._ 42 5 _._ 02 **2** _._ **40** 3 _._ 14 3 _._ 50


Same 16 _._ 23 11 _._ 14 24 _._ 97 10 _._ 88 19 _._ 08 10 _._ 47 10 _._ 71 5 _._ 39 **3** _._ **88** 4 _._ 58 4 _._ 19
E NTITY -13
Novel 28 _._ 53 22 _._ 02 38 _._ 33 21 _._ 64 32 _._ 43 21 _._ 22 20 _._ 61 13 _._ 58 10 _._ 28 12 _._ 25 **6** _._ **63**


Same 18 _._ 59 14 _._ 46 28 _._ 82 14 _._ 30 21 _._ 63 13 _._ 46 12 _._ 92 9 _._ 12 **7** _._ **75** 8 _._ 15 **7** _._ **64**
E NTITY -30
Novel 32 _._ 34 26 _._ 85 44 _._ 02 26 _._ 27 36 _._ 82 25 _._ 42 23 _._ 16 17 _._ 75 14 _._ 30 15 _._ 60 **10** _._ **57**


Same 18 _._ 66 17 _._ 17 26 _._ 39 16 _._ 14 19 _._ 86 15 _._ 58 16 _._ 63 10 _._ 87 **10** _._ **24** 10 _._ 07 **10** _._ **26**
N ONLIVING -26
Novel 33 _._ 43 31 _._ 53 41 _._ 66 29 _._ 87 35 _._ 13 29 _._ 31 29 _._ 56 21 _._ 70 20 _._ 12 19 _._ 08 **18** _._ **26**


Same 12 _._ 63 11 _._ 05 18 _._ 32 10 _._ 46 14 _._ 43 10 _._ 14 9 _._ 87 4 _._ 57 **3** _._ **95** **3** _._ **81** 4 _._ 21
L IVING -17
Novel 29 _._ 03 26 _._ 96 35 _._ 67 26 _._ 11 31 _._ 73 25 _._ 73 23 _._ 53 16 _._ 15 14 _._ 49 12 _._ 97 **11** _._ **39**


Table 1: _Mean Absolute estimation Error (MAE) results for different datasets in our setup grouped by_
_the nature of shift._ ‘Same’ refers to same subpopulation shifts and ‘Novel’ refers novel subpopulation
shifts. We include details about the target sets considered in each shift in Table 2. Post T denotes use
of TS calibration on source. Across all datasets, we observe that ATC achieves superior performance
(lower MAE is better). For language datasets, we use DistilBERT-base-uncased, for vision dataset we
report results with DenseNet model with the exception of MNIST where we use FCN. We include
results on other architectures in App. H. For GDE post T and pre T estimates match since TS doesn’t
alter the argmax prediction. Results reported by aggregating MAE numbers over 4 different seeds.
We include results with standard deviation values in Table 3.


_Average Confidence (AC)._ Error is estimated as the expected value of the maximum softmax
confidence on the target data, i.e, AC _D_ T “ E _x_ „ _D_ T rmax _j_ P _Y_ _f_ _j_ p _x_ qs.


_Difference Of Confidence (DOC)._ We estimate error on target by subtracting difference of confidences
on source and target (as a surrogate to distributional distance Guillory et al. (2021)) from the error on
source distribution, i.e, DOC _D_ T “ E _x_ „ _D_ S “I “arg max _j_ P _Y_ _f_ _j_ p _x_ q ‰ _y_ ‰‰ ` E _x_ „ _D_ T rmax _j_ P _Y_ _f_ _j_ p _x_ qs ´
E _x_ „ _D_ S rmax _j_ P _Y_ _f_ _j_ p _x_ qs. This is referred to as DOC-Feat in (Guillory et al., 2021).


_Importance re-weighting (IM)._ We estimate the error of the classifier with importance re-weighting
of 0-1 error in the pushforward space of the classifier. This corresponds to M ANDOLIN using one
slice based on the underlying classifier confidence Chen et al. (2021b).


_Generalized Disagreement Equality (GDE)._ Error is estimated as the expected disagreement of two
models (trained on the same training set but with different randomization) on target data (Jiang et al.,
2021), i.e., GDE _D_ T “ E _x_ „ _D_ T rI r _f_ p _x_ q ‰ _f_ [1] p _x_ qss where _f_ and _f_ [1] are the two models. Note that GDE
requires two models trained independently, doubling the computational overhead while training.


5.1 R ESULTS


In Table 1, we report MAE results aggregated by the nature of the shift in our testbed. In Fig. 2
and Fig. 1(right), we show scatter plots for predicted accuracy versus OOD accuracy on several
datasets. We include scatter plots for all datasets and parallel results with other architectures in
App. H. In App. H.1, we also perform ablations on CIFAR using a pre-trained model and observe
that pre-training doesn’t change the efficacy of ATC.


7


Published as a conference paper at ICLR 2022



90

80

70

60

50

40 f

30

20








|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||~~Diﬀ-subpopulati~~<br>|~~on~~<br>|~~on~~<br>|
||~~Same-subpopula~~|~~tion~~||
|20<br>40<br>60<br>80<br>OOD Accuracy|20<br>40<br>60<br>80<br>OOD Accuracy|20<br>40<br>60<br>80<br>OOD Accuracy|20<br>40<br>60<br>80<br>OOD Accuracy|


|90<br>80<br>70<br>60<br>50<br>40 DOC (w/o fit)<br>DOC (w/ fit)<br>30<br>ATC (w/o fit)<br>20<br>0 20 40 60 80<br>OOD Accuracy|Col2|Col3|Col4|
|---|---|---|---|
|0<br>20<br>40<br>60<br>80<br>OOD Accuracy<br>20<br>30<br>40<br>50<br>60<br>70<br>80<br>90<br>DOC (w/o ﬁt)<br>DOC (w/ ﬁt)<br>ATC (w/o ﬁt)||||
|0<br>20<br>40<br>60<br>80<br>OOD Accuracy<br>20<br>30<br>40<br>50<br>60<br>70<br>80<br>90<br>DOC (w/o ﬁt)<br>DOC (w/ ﬁt)<br>ATC (w/o ﬁt)||||
|0<br>20<br>40<br>60<br>80<br>OOD Accuracy<br>20<br>30<br>40<br>50<br>60<br>70<br>80<br>90<br>DOC (w/o ﬁt)<br>DOC (w/ ﬁt)<br>ATC (w/o ﬁt)||DOC (w/<br>DOC (w/|o ﬁt)<br> ﬁt)|
|0<br>20<br>40<br>60<br>80<br>OOD Accuracy<br>20<br>30<br>40<br>50<br>60<br>70<br>80<br>90<br>DOC (w/o ﬁt)<br>DOC (w/ ﬁt)<br>ATC (w/o ﬁt)||ATC (w/o|ﬁt)|



Figure 3: **Left:** Predicted accuracy with DOC on Living17 B REEDS dataset. We observe a substantial
gap in the linear fit of same and different subpopulations highlighting poor correlation. **Middle:**
After fitting a robust linear model for DOC on same subpopulation, we show predicted accuracy on
different subpopulations with fine-tuned DOC (i.e., DOC (w/ fit)) and compare with ATC without any
regression model, i.e., ATC (w/o fit). While observe substantial improvements in MAE from 24 _._ 41
with DOC (w/o fit) to 13 _._ 26 with DOC (w/ fit), ATC (w/o fit) continues to outperform even DOC
(w/ fit) with MAE 10 _._ 22 . We show parallel results with other B REEDS datasets in App. H.2. **Right :**
Empirical validation of our toy model. We show that ATC perfectly estimates target performance as
we vary the degree of spurious correlation in target. ‘ˆ’ represents accuracy on source.


We predict accuracy on the target data before and after calibration with TS. First, we observe that
both ATC-NE and ATC-MC (even without TS) obtain significantly lower MAE when compared with
other methods (even with TS). Note that with TS we observe substantial improvements in MAE
for all methods. Overall, ATC-NE (with TS) typically achieves the smallest MAE improving by
more than 2ˆ on CIFAR and by 3 – 4ˆ on ImageNet over GDE (the next best alternative to ATC).
Alongside, we also observe that a linear fit with robust regression (Siegel, 1982) on the scatter plot
recovers a line close to _x_ “ _y_ for ATC-NE with TS while the line is far away from _x_ “ _y_ for other
methods (Fig. 2 and Fig. 1(right)). Remarkably, MAE is in the range of 0 _._ 4 – 5 _._ 8 with ATC for CIFAR,
ImageNet, MNIST, and Wilds. However, MAE is much higher on B REEDS benchmark with novel
subpopulations. While we observe a small MAE (i.e., comparable to our observations on other
datasets) on B REEDS with natural and synthetic shifts from the same sub-population, MAE on shifts
with novel population is significantly higher with all methods. Note that even on novel populations,
ATC continues to dominate all other methods across all datasets in B REEDS .


Additionally, for different subpopulations in B REEDS setup, we observe a poor linear correlation
of the estimated performance with the actual performance as shown in Fig. 3 (left)(we notice a
similar gap in the linear fit for all other methods). Hence in such a setting, we would expect methods
that fine-tune a regression model on labeled target examples from shifts with one subpopulation
will perform poorly on shifts with different subpopulations. Corroborating this intuition, next, we
show that even after fitting a regression model for DOC on natural and synthetic shifts with source
subpopulations, ATC without regression model continues to outperform DOC with regression model
on shifts with novel subpopulation.


**Fitting a regression model on B** **REEDS** **with DOC.** Using label target data from natural and
synthetic shifts for the same subpopulation (same as source), we fit a robust linear regression
model (Siegel, 1982) to fine-tune DOC as in Guillory et al. (2021). We then evaluate the fine-tuned
DOC (i.e., DOC with linear model) on natural and synthetic shifts from novel subpopulations on
B REEDS benchmark. Although we observe significant improvements in the performance of finetuned DOC when compared with DOC (without any fine-tuning), ATC without any regression model
continues to perform better (or similar) to that of fine-tuned DOC on novel subpopulations (Fig. 3
(middle)). Refer to App. H.2 for details and Table 5 for MAE on B REEDS with regression model.


6 I NVESTIGATING ATC ON T OY M ODEL


In this section, we propose and analyze a simple theoretical model that distills empirical phenomena
from the previous section and highlights efficacy of ATC. Here, our aim is not to obtain a general
model that captures complicated real distributions on high dimensional input space as the images in
ImageNet. Instead to further our understanding, we focus on an _easy-to-learn_ binary classification
task from Nagarajan et al. (2020) with linear classifiers, that is rich enough to exhibit some of the
same phenomena as with deep networks on real data distributions.


8


Published as a conference paper at ICLR 2022


Consider a easy-to-learn binary classification problem with two features _x_ “ r _x_ inv _, x_ sp s P R [2] where
_x_ inv is fully predictive invariant feature with a margin _γ_ ą 0 and _x_ sp P t´1 _,_ 1u is a spurious feature
(i.e., a feature that is correlated but not predictive of the true label). Conditional on _y_, the distribution
over _x_ inv is given as follows: _x_ inv |p _y_ “ 1q „ _U_ r _γ, c_ s and _x_ inv |p _y_ “ 0q „ _U_ r´ _c,_ ´ _γ_ s, where _c_ is a
fixed constant greater than _γ_ . For simplicity, we assume that label distribution on source is uniform
on t´1 _,_ 1u . _x_ sp is distributed such that _P_ _s_ r _x_ sp ¨ p2 _y_ ´ 1q ą 0s “ _p_ sp, where _p_ sp P p0 _._ 5 _,_ 1 _._ 0q controls
the degree of spurious correlation. To model distribution shift, we simulate target data with different
degree of spurious correlation, i.e., in target distribution _P_ _t_ r _x_ sp ¨ p2 _y_ ´ 1q ą 0s “ _p_ [1] sp [P r][0] _[,]_ [ 1][s] [. Note]
that here we do not consider shifts in the label distribution but our result extends to arbitrary shifts in
the label distribution as well.


1 _e_ _[wT x]_
In this setup, we examine linear sigmoid classifiers of the form _f_ p _x_ q “ ” 1` _e_ _[wT x]_ _[,]_ 1` _e_ _[wT x]_ ı where

_w_ “ r _w_ inv _, w_ sp s P R [2] . While there exists a linear classifier with _w_ “ r1 _,_ 0s that correctly classifies all
the points with a margin _γ_, Nagarajan et al. (2020) demonstrated that a linear classifier will typically
have a dependency on the spurious feature, i.e., _w_ sp ‰ 0 . They show that due to geometric skews,
despite having positive dependencies on the invariant feature, a max-margin classifier trained on
finite samples relies on the spurious feature. Refer to App. D for more details on these skews. In
our work, we show that given a linear classifier that relies on the spurious feature and achieves a
non-trivial performance on the source (i.e., _w_ inv ą 0 ), ATC with maximum confidence score function
_consistently_ estimates the accuracy on the target distribution.

**Theorem 1** (Informal) **.** _Given any classifier with_ _w_ _inv_ ą 0 _in the above setting, the threshold obtained_
_in_ (1) _together with ATC as in_ (2) _with maximum confidence score function obtains a consistent_
_estimate of the target accuracy._


Consider a classifier that depends positively on the spurious feature (i.e., _w_ sp ą 0 ). Then as the
spurious correlation decreases in the target data, the classifier accuracy on the target will drop and
vice-versa if the spurious correlation increases on the target data. Theorem 1 shows that the threshold
identified with ATC as in (1) remains invariant as the distribution shifts and hence ATC as in (2)
will correctly estimate the accuracy with shifting distributions. Next, we illustrate Theorem 1 by
simulating the setup empirically. First we pick a arbitrary classifier (which can also be obtained by
training on source samples), tune the threshold on hold-out source examples and predict accuracy
with different methods as we shift the distribution by varying the degree of spurious correlation.


**Empirical validation and comparison with other methods.** Fig. 3(right) shows that as the degree
of spurious correlation varies, our method accurately estimates the target performance where all other
methods fail to accurately estimate the target performance. Understandably, due to poor calibration of
the sigmoid linear classifier AC, DOC and GDE fail. While in principle IM can perfectly estimate the
accuracy on target in this case, we observe that it is highly sensitive to the number bins and choice of
histogram binning (i.e., uniform mass or equal width binning). We elaborate more on this in App. D.


**Biased estimation with ATC.** Now we discuss changes in the above setup where ATC yields
inconsistent estimates. We assumed that both in source and target _x_ inv | _y_ “ 1 is uniform between r _γ, c_ s
and _x_ | _y_ “ ´1 is uniform between r´ _c,_ ´ _γ_ s . Shifting the support of target class conditional _p_ _t_ p _x_ inv | _y_ q
may introduce a bias in ATC estimates, e.g., shrinking the support to _c_ 1 ( ă _c_ ) (while maintaining
uniform distribution) in the target will lead to an over-estimation of the target performance with
ATC. In App. D.1, we elaborate on this failure and present a general (but less interpretable) classifier
dependent distribution shift condition where ATC is guaranteed to yield consistent estimates.


7 C ONCLUSION AND FUTURE WORK


In this work, we proposed ATC, a simple method for estimating target domain accuracy based on
unlabeled target (and labeled source data). ATC achieves remarkably low estimation error on several
synthetic and natural shift benchmarks in our experiments. Notably, our work draws inspiration
from recent state-of-the-art methods that use softmax confidences below a certain threshold for OOD
detection (Hendrycks & Gimpel, 2016; Hendrycks et al., 2019) and takes a step forward in answering
questions raised in Deng & Zheng (2021) about the practicality of threshold based methods.


Our distribution shift toy model justifies ATC on an easy-to-learn binary classification task. In our
experiments, we also observe that calibration significantly improves estimation with ATC. Since in
binary classification, post hoc calibration with TS does not change the effective threshold, in future
work, we hope to extend our theoretical model to multi-class classification to understand the efficacy


9


Published as a conference paper at ICLR 2022


of calibration. Our theory establishes that a classifier’s accuracy is not, in general identified, from
labeled source and unlabeled target data alone, absent considerable additional constraints on the
target conditional _p_ _t_ p _y_ | _x_ q . In light of this finding, we also hope to extend our understanding beyond
the simple theoretical toy model to characterize broader sets of conditions under which ATC might
be guaranteed to obtain consistent estimates. Finally, we should note that while ATC outperforms
previous approaches, it still suffers from large estimation error on datasets with novel populations,
e.g., B REEDS . We hope that our findings can lay the groundwork for future work for improving
accuracy estimation on such datasets.


**Reproducibility Statement** Our code to reproduce all the results is available at [https://](https://github.com/saurabhgarg1996/ATC_code)
[github.com/saurabhgarg1996/ATC_code](https://github.com/saurabhgarg1996/ATC_code) . We have been careful to ensure that our results are reproducible. We have stored all models and logged all hyperparameters and seeds to
facilitate reproducibility. Note that throughout our work, we do not perform any hyperparameter
tuning, instead, using benchmarked hyperparameters and training procedures to make our results easy
to reproduce. While, we have not released code yet, the appendix provides all the necessary details to
replicate our experiments and results.


A CKNOWLEDGEMENT


Authors would like to thank Ariel Kleiner and Sammy Jerome as the problem formulation and
motivation of this paper was highly influenced by initial discussions with them.


R EFERENCES


Amr Alexandari, Anshul Kundaje, and Avanti Shrikumar. Adapting to label shift with bias-corrected
calibration. In _arXiv preprint arXiv:1901.06852_, 2019.


Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and Animashree Anandkumar. Regularized learning
for domain adaptation under label shifts. In _International Conference on Learning Representations_
_(ICLR)_, 2019.


Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for
neural networks. In _Advances in neural information processing systems_, pp. 6240–6249, 2017.


Shai Ben-David, Tyler Lu, Teresa Luu, and David P ´ al. Impossibility Theorems for Domain Adaptation. ´
In _International Conference on Artificial Intelligence and Statistics (AISTATS)_, 2010.


Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Nuanced metrics
for measuring unintended bias with real data for text classification. In _Companion Proceedings of_
_The 2019 World Wide Web Conference_, 2019.


Jiefeng Chen, Frederick Liu, Besim Avci, Xi Wu, Yingyu Liang, and Somesh Jha. Detecting errors
and estimating accuracy on unlabeled data with self-training ensembles. _Advances in Neural_
_Information Processing Systems_, 34:14980–14992, 2021a.


Mayee Chen, Karan Goel, Nimit S Sohoni, Fait Poms, Kayvon Fatahalian, and Christopher Re. ´
Mandoline: Model evaluation under distribution shift. In _International Conference on Machine_
_Learning_, pp. 1617–1629. PMLR, 2021b.


Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In
_Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2018.


Ching-Yao Chuang, Antonio Torralba, and Stefanie Jegelka. Estimating generalization under distribution shifts via domain-invariant representations. _arXiv preprint arXiv:2007.03511_, 2020.


Weijian Deng and Liang Zheng. Are labels always necessary for classifier accuracy evaluation?
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp.
15069–15078, 2021.


Weijian Deng, Stephen Gould, and Liang Zheng. What does rotation prediction tell us about classifier
accuracy under varying testing environments? _arXiv preprint arXiv:2106.05961_, 2021.


10


Published as a conference paper at ICLR 2022


Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for
deep (stochastic) neural networks with many more parameters than training data. _arXiv preprint_
_arXiv:1703.11008_, 2017.


Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, and Zachary C Lipton. A unified view of label
shift estimation. _arXiv preprint arXiv:2003.07554_, 2020.


Saurabh Garg, Sivaraman Balakrishnan, J Zico Kolter, and Zachary C Lipton. Ratt: Leveraging
unlabeled data to guarantee generalization. _arXiv preprint arXiv:2105.00303_, 2021.


Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. _arXiv preprint_
_arXiv:1705.08500_, 2017.


Devin Guillory, Vaishaal Shankar, Sayna Ebrahimi, Trevor Darrell, and Ludwig Schmidt. Predicting
with confidence on unseen distributions. _arXiv preprint arXiv:2107.03315_, 2021.


Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. In _International Conference on Machine Learning (ICML)_, 2017.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image
Recognition. In _Computer Vision and Pattern Recognition (CVPR)_, 2016.


James J Heckman. Sample Selection Bias as a Specification Error (With an Application to the
Estimation of Labor Supply Functions), 1977.


Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common
corruptions and perturbations. _arXiv preprint arXiv:1903.12261_, 2019.


Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution
examples in neural networks. _arXiv preprint arXiv:1610.02136_, 2016.


Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt,
and Dawn Song. Scaling out-of-distribution detection for real-world settings. _arXiv preprint_
_arXiv:1911.11132_, 2019.


Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul
Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer.
The many faces of robustness: A critical analysis of out-of-distribution generalization. _ICCV_,
2021.


Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected
convolutional networks. In _Proceedings of the IEEE conference on computer vision and pattern_
_recognition_, pp. 4700–4708, 2017.


Jonathan J. Hull. A database for handwritten text recognition research. _IEEE Transactions on pattern_
_analysis and machine intelligence_, 16(5):550–554, 1994.


Xu Ji, Razvan Pascanu, Devon Hjelm, Andrea Vedaldi, Balaji Lakshminarayanan, and Yoshua Bengio.
Predicting unreliable predictions by shattering a neural network. _arXiv preprint arXiv:2106.08365_,
2021.


Heinrich Jiang, Been Kim, Melody Y Guan, and Maya R Gupta. To trust or not to trust a classifier.
In _NeurIPS_, pp. 5546–5557, 2018.


Yiding Jiang, Vaishnavh Nagarajan, Christina Baek, and J Zico Kolter. Assessing generalization of
sgd via disagreement. _arXiv preprint arXiv:2106.13799_, 2021.


Diederik P Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. _arXiv Preprint_
_arXiv:1412.6980_, 2014.


Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne
David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec,
Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A
benchmark of in-the-wild distribution shifts. In _International Conference on Machine Learning_
_(ICML)_, 2021.


11


Published as a conference paper at ICLR 2022


Alex Krizhevsky and Geoffrey Hinton. Learning Multiple Layers of Features from Tiny Images.
Technical report, Citeseer, 2009.


Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. _arXiv preprint arXiv:1612.01474_, 2016.


Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-Based Learning Applied ´
to Document Recognition. _Proceedings of the IEEE_, 86, 1998.


Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution
image detection in neural networks. _arXiv preprint arXiv:1706.02690_, 2017.


Zachary C Lipton, Yu-Xiang Wang, and Alex Smola. Detecting and Correcting for Label Shift with
Black Box Predictors. In _International Conference on Machine Learning (ICML)_, 2018.


Philip M Long and Hanie Sedghi. Generalization bounds for deep convolutional neural networks.
_arXiv preprint arXiv:1905.12600_, 2019.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint_
_arXiv:1711.05101_, 2017.


Vaishnavh Nagarajan and J Zico Kolter. Deterministic pac-bayesian generalization bounds for deep
networks via generalizing noise-resilience. _arXiv preprint arXiv:1905.13344_, 2019a.


Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization
in deep learning. In _Advances in Neural Information Processing Systems_, pp. 11615–11626, 2019b.


Vaishnavh Nagarajan, Anders Andreassen, and Behnam Neyshabur. Understanding the failure modes
of out-of-distribution generalization. _arXiv preprint arXiv:2010.15775_, 2020.


Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading
digits in natural images with unsupervised feature learning. In _Advances in Neural Information_
_Processing Systems (NIPS)_, 2011.


Behnam Neyshabur. Implicit regularization in deep learning. _arXiv preprint arXiv:1709.01953_, 2017.


Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural
networks. In _Conference on Learning Theory_, pp. 1376–1401, 2015.


Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nathan Srebro. Exploring generalization in deep learning. _arXiv preprint arXiv:1706.08947_, 2017.


Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, and Nathan Srebro. The role
of over-parametrization in generalization of neural networks. In _International Conference on_
_Learning Representations_, 2018.


Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled
reviews and fine-grained aspects. In _Proceedings of the 2019 Conference on Empirical Methods in_
_Natural Language Processing and the 9th International Joint Conference on Natural Language_
_Processing (EMNLP-IJCNLP)_, 2019.


Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua V
Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty?
evaluating predictive uncertainty under dataset shift. _arXiv preprint arXiv:1906.02530_, 2019.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward
Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep
learning library. In _Advances in Neural Information Processing Systems 32_, 2019.


Emmanouil A Platanios, Hoifung Poon, Tom M Mitchell, and Eric Horvitz. Estimating accuracy
from unlabeled data: A probabilistic logic approach. _arXiv preprint arXiv:1705.07086_, 2017.


12


Published as a conference paper at ICLR 2022


Emmanouil Antonios Platanios, Avinava Dubey, and Tom Mitchell. Estimating accuracy from
unlabeled data: A bayesian approach. In _International Conference on Machine Learning_, pp.
1416–1425. PMLR, 2016.


Stephan Rabanser, Stephan Gunnemann, and Zachary C Lipton. Failing loudly: An empirical study ¨
of methods for detecting dataset shift. _arXiv preprint arXiv:1810.11953_, 2018.


Aaditya Ramdas, Sashank Jakkam Reddi, Barnabas P ´ oczos, Aarti Singh, and Larry A Wasserman. ´
On the Decreasing Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High
Dimensions. In _Association for the Advancement of Artificial Intelligence (AAAI)_, 2015.


Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do cifar-10 classifiers
generalize to cifar-10? _arXiv preprint arXiv:1806.00451_, 2018.


Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers
generalize to imagenet? In _International Conference on Machine Learning_, pp. 5389–5400. PMLR,
2019.


Mateo Rojas-Carulla, Bernhard Scholkopf, Richard Turner, and Jonas Peters. Invariant models for ¨
causal transfer learning. _The Journal of Machine Learning Research_, 19(1):1309–1342, 2018.


Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition
challenge. _International journal of computer vision_, 115(3):211–252, 2015.


Marco Saerens, Patrice Latinne, and Christine Decaestecker. Adjusting the Outputs of a Classifier to
New a Priori Probabilities: A Simple Procedure. _Neural Computation_, 2002.


Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of
bert: smaller, faster, cheaper and lighter. _ArXiv_, abs/1910.01108, 2019.


Shibani Santurkar, Dimitris Tsipras, and Aleksander Madry. Breeds: Benchmarks for subpopulation
shift. _arXiv preprint arXiv:2008.04859_, 2020.


Hidetoshi Shimodaira. Improving Predictive Inference Under Covariate Shift by Weighting the
Log-Likelihood Function. _Journal of Statistical Planning and Inference_, 2000.


Andrew F Siegel. Robust regression using repeated medians. _Biometrika_, 69(1):242–244, 1982.


Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow,
and Rob Fergus. Intriguing Properties of Neural Networks. In _International Conference on_
_Learning Representations (ICLR)_, 2014.


Remi Tachet des Combes, Han Zhao, Yu-Xiang Wang, and Geoffrey J Gordon. Domain adaptation
with conditional distribution matching and generalized label shift. _Advances in Neural Information_
_Processing Systems_, 33, 2020.


J. Taylor, B. Earnshaw, B. Mabey, M. Victors, and J. Yosinski. Rxrx1: An image set for cellular
morphological variation across many experimental batches. In _International Conference on_
_Learning Representations (ICLR)_, 2019.


Antonio Torralba, Rob Fergus, and William T. Freeman. 80 million tiny images: A large data set for
nonparametric object and scene recognition. _IEEE Transactions on Pattern Analysis and Machine_
_Intelligence_, 30(11):1958–1970, 2008.


Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations
by penalizing local predictive power. In _Advances in Neural Information Processing Systems_, pp.
10506–10518, 2019.


Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von ´
Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama
Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language
processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language_
_Processing: System Demonstrations_, pp. 38–45. Association for Computational Linguistics, 2020.


13


Published as a conference paper at ICLR 2022


Chhavi Yadav and Leon Bottou. Cold case: The lost mnist digits. In ´ _Advances in Neural Information_
_Processing Systems 32_, 2019.


Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. _arXiv preprint arXiv:1611.03530_, 2016.


Hongjie Zhang, Ang Li, Jie Guo, and Yanwen Guo. Hybrid models for open set recognition. In
_European Conference on Computer Vision_, pp. 102–117. Springer, 2020.


Kun Zhang, Bernhard Scholkopf, Krikamol Muandet, and Zhikun Wang. Domain Adaptation Under ¨
Target and Conditional Shift. In _International Conference on Machine Learning (ICML)_, 2013.


Wenda Zhou, Victor Veitch, Morgane Austern, Ryan P Adams, and Peter Orbanz. Non-vacuous
generalization bounds at the imagenet scale: a pac-bayesian compression approach. _arXiv preprint_
_arXiv:1804.05862_, 2018.


14


Published as a conference paper at ICLR 2022


A PPENDIX


A P ROOFS FROM S EC . 3


Before proving results from Sec. 3, we introduce some notations. Define _E_ p _f_ p _x_ q _, y_ q :“
I “ _y_ R arg max _j_ P _Y_ _f_ _j_ p _x_ q‰ . We express the _population error_ on distribution _D_ as _E_ _D_ p _f_ q :“
E p _x,y_ q„ _D_ r _E_ p _f_ p _x_ q _, y_ qs.


_Proof of Proposition 1._ Consider a binary classification problem. Assume _P_ be the set of possible
target conditional distribution of labels given _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q.


The forward direction is simple. If _P_ “ t _p_ _t_ p _y_ | _x_ qu is singleton given _p_ _s_ p _x, y_ q and _p_ _t_ p _x_ q, then the
error of any classifier _f_ on the target domain is identified and is given by



_E_ _D_ _T_ p _f_ q “ E _x_ „ _p_ _t_ p _x_ q _,y_ „ _p_ _t_ p _y_ | _x_ q



I arg max _f_ _j_ p _x_ q ‰ _y_ _._ (3)
„ „ _j_ P _Y_ 



For the reverse direction assume that given _p_ _t_ p _x_ q and _p_ _s_ p _x, y_ q, we have two possible distributions _D_ _[T]_

and _D_ _[T]_ [ 1] with _p_ _t_ p _y_ | _x_ q _, p_ [1] _t_ [p] _[y]_ [|] _[x]_ [q P] _[ P]_ [ such that on some] _[ x]_ [ with] _[ p]_ _[t]_ [p] _[x]_ [q ą][ 0] [, we have] _[ p]_ _[t]_ [p] _[y]_ [|] _[x]_ [q ‰] _[ p]_ [1] _t_ [p] _[y]_ [|] _[x]_ [q] [.]
Consider _X_ _M_ “ t _x_ P _X_ | _p_ _t_ p _x_ q ą 0 and _p_ _t_ p _y_ “ 1| _x_ q ‰ _p_ [1] _t_ [p] _[y]_ [ “][ 1][|] _[x]_ [qu] [ be the set of all input covariates]
where the two distributions differ. We will now choose a classifier _f_ such that the error on the two
distributions differ. On a subset _X_ _M_ [1] [“ t] _[x]_ [ P] _[ X]_ [|] _[p]_ _[t]_ [p] _[x]_ [q ą][ 0][ and] _[ p]_ _[t]_ [p] _[y]_ [ “][ 1][|] _[x]_ [q ą] _[ p]_ _t_ [1] [p] _[y]_ [ “][ 1][|] _[x]_ [qu] [, assume]
_f_ p _x_ q “ 0 and on a subset _X_ _M_ [2] [“ t] _[x]_ [ P] _[ X]_ [|] _[p]_ _[t]_ [p] _[x]_ [q ą][ 0][ and] _[ p]_ _[t]_ [p] _[y]_ [ “][ 1][|] _[x]_ [q ă] _[ p]_ _t_ [1] [p] _[y]_ [ “][ 1][|] _[x]_ [qu] [, assume]
_f_ p _x_ q “ 1 . We will show that the error of _f_ on distribution with _p_ _t_ p _y_ | _x_ q is strictly greater than the
error of _f_ on distribution with _p_ [1] _t_ [p] _[y]_ [|] _[x]_ [q][. Formally,]
_E_ _D_ _T_ p _f_ q ´ _E_ _D_ _T_ [1] p _f_ q



I arg max _f_ _j_ p _x_ q ‰ _y_
„ „ _j_ P _Y_ 



“ E _x_ „ _p_ _t_ p _x_ q _,y_ „ _p_ _t_ p _y_ | _x_ q



I arg max _f_ _j_ p _x_ q ‰ _y_ ´ E _x_ „ _p_ _t_ p _x_ q _,y_ „ _p_ [1] _t_ [p] _[y]_ [|] _[x]_ [q]
„ „ _j_ P _Y_ 



“ I r _f_ p _x_ q ‰ 0s ` _p_ _t_ p _y_ “ 0| _x_ q ´ _p_ [1] _t_ [p] _[y]_ [ “][ 0][|] _[x]_ [q] ˘ _p_ _t_ p _x_ q _dx_
ż _x_ P _X_ _M_


` I r _f_ p _x_ q ‰ 1s ` _p_ _t_ p _y_ “ 1| _x_ q ´ _p_ [1] _t_ [p] _[y]_ [ “][ 1][|] _[x]_ [q] ˘ _p_ _t_ p _x_ q _dx_
ż _x_ P _X_ _M_



“
ż _x_ P _X_ _M_ [2]



` _p_ _t_ p _y_ “ 0| _x_ q ´ _p_ [1] _t_ [p] _[y]_ [ “][ 0][|] _[x]_ [q] ˘ _p_ _t_ p _x_ q _dx_ `
ż _x_ P _X_ _M_ [1]



` _p_ _t_ p _y_ “ 1| _x_ q ´ _p_ [1] _t_ [p] _[y]_ [ “][ 1][|] _[x]_ [q] ˘ _p_ _t_ p _x_ q _dx_



ą 0 _,_ (4)
where the last step follows by construction of the set _X_ _M_ [1] [and] _[ X]_ [ 2] _M_ [. Since] _[ E]_ _[D]_ _[T]_ [ p] _[f]_ [q ‰] _[ E]_ _D_ _[T]_ [ 1] [p] _[f]_ [q] [, given]
the information of _p_ _t_ p _x_ q and _p_ _s_ p _x, y_ q it is impossible to distinguish the two values of the error with
classifier _f_ . Thus, we obtain a contradiction on the assumption that _p_ _t_ p _y_ | _x_ q ‰ _p_ [1] _t_ [p] _[y]_ [|] _[x]_ [q] [. Hence, we]
must pose restrictions on the nature of shift such that _P_ is singleton to to identify accuracy on the
target.


_Proof of Corollary 1._ The corollary follows directly from Proposition 1. Since two different target
conditional distribution can lead to different error estimates without assumptions on the classifier, no
method can estimate two different quantities from the same given information. We illustrate this in
Example 1 next.


B E STIMATING ACCURACY IN COVARIATE SHIFT OR LABEL SHIFT


**Accuracy estimation under covariate shift assumption** Under the assumption that _p_ _t_ p _y_ | _x_ q “
_p_ _s_ p _y_ | _x_ q, accuracy on the target domain can be estimated as follows:

_p_ _t_ p _x,_ _y_ q
_E_ _D_ T p _f_ q “ E p _x,y_ q„ _D_ S (5)
„ _p_ _s_ p _x, y_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 

“ E p _x,y_ q„ _D_ S _p_ _t_ p _x_ q _._ (6)
„ _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 


15


Published as a conference paper at ICLR 2022


Given access to _p_ _t_ p _x_ q and _p_ _s_ p _x_ q, one can directly estimate the expression in (6).


**Accuracy estimation under label shift assumption** Under the assumption that _p_ _t_ p _x_ | _y_ q “ _p_ _s_ p _x_ | _y_ q,
accuracy on the target domain can be estimated as follows:

_p_ _t_ p _x,_ _y_ q
_E_ _D_ T p _f_ q “ E p _x,y_ q„ _D_ S (7)
„ _p_ _s_ p _x, y_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 

“ E p _x,y_ q„ _D_ S _p_ _t_ p _y_ q _._ (8)
„ _p_ _s_ p _y_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 


Estimating importance ratios _p_ _t_ p _x_ q{ _p_ _s_ p _x_ q is straightforward under covariate shift assumption when
the distributions _p_ _t_ p _x_ q and _p_ _s_ p _x_ q are known. For label shift, one can leverage moment matching
approach called BBSE (Lipton et al., 2018) or likelihood minimization approach MLLS (Garg et al.,
2020). Below we discuss the objective of MLLS:
_w_ “ arg max E _x_ „ _p_ _t_ p _x_ q “log _p_ _s_ p _y_ | _x_ q _[T]_ _w_ ‰ _,_ (9)
_w_ P _W_

where _W_ “ t _w_ | @ _y, w_ _y_ ě 0 and [ř] _[k]_ _y_ “1 _[w]_ _[y]_ _[p]_ _[s]_ [p] _[y]_ [q “][ 1][u] [. MLLS objective is guaranteed to obtain]
consistent estimates for the importance ratios _w_ [˚] p _y_ q “ _p_ _t_ p _y_ q{ _p_ _s_ p _y_ q under the following condition.
**Theorem 2** (Theorem 1 (Garg et al., 2020)) **.** _If the distributions_ t _p_ p _x_ q| _y_ q : _y_ “ 1 _, . . ., k_ u _are strictly_
_linearly independent, then w_ [˚] _is the unique maximizer of the MLLS objective_ (9) _._


We refer interested reader to Garg et al. (2020) for details.


Above results of accuracy estimation under label shift and covariate shift can be extended to a
generalized label shift and covariate shift settings. Assume a function _h_ : _X_ Ñ _Z_ such that _y_ is
independent of _x_ given _h_ p _x_ q . In other words _h_ p _x_ q contains all the information needed to predict
label _y_ . With help of _h_, we can extend estimation to following settings: (i) _Generalized covariate_
_shift_, i.e., _p_ _s_ p _y_ | _h_ p _x_ qq “ _p_ _t_ p _y_ | _h_ p _x_ qq and _p_ _s_ p _h_ p _x_ qq ą 0 for all _x_ P _X_ _t_ ; (ii) _Generalized label shift_, i.e.,
_p_ _s_ p _h_ p _x_ q| _y_ q “ _p_ _t_ p _h_ p _x_ q| _y_ q and _p_ _s_ p _y_ q ą 0 for all _y_ P _Y_ _t_ . By simply replacing _x_ with _h_ p _x_ q in (6) and
(9), we will obtain consistent error estimates under these generalized conditions.


_Proof of Example 1._ Under covariate shift using (6), we get



_p_ _t_ p _x_ q

_E_ 1 “ E p _x,y_ q„ _p_ _s_ p _x,y_ q

„ _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 



_p_ _t_ p _x_ q

` E _x_ „ _p_ _s_ p _x,y_ “1q

„ _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰][ 0][s] 



_p_ _t_ p _x_ q
„ _p_ _s_ p _x_ q



_p_ _t_ p _x_ q
„ _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰][ 1][s] 



“ E _x_ „ _p_ _s_ p _x,y_ “0q



“ I r _f_ p _x_ q ‰ 0s _p_ _t_ p _x_ q _p_ _s_ p _y_ “ 0| _x_ q _dx_ ` I r _f_ p _x_ q ‰ 1s _p_ _t_ p _x_ q _p_ _s_ p _y_ “ 1| _x_ q _dx_
ż ż

Under label shift using (8), we get

_p_ _t_ p _y_ q
_E_ 2 “ E p _x,y_ q„ _D_ S
„ _p_ _s_ p _y_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 



1 ´ _β_
„ 1 ´ _α_ [I][ r] _[f]_ [p] _[x]_ [q ‰][ 1][s] 



“ E _x_ „ _p_ _s_ p _x,y_ “0q



_β_

` E _x_ „ _p_ _s_ p _x,y_ “1q

„ _α_ [I][ r] _[f]_ [p] _[x]_ [q ‰][ 0][s] 




_[β]_ I r _f_ p _x_ q ‰ 1s [p][1][ ´] _[β]_ [q]

_α_ _[p]_ _[s]_ [p] _[y]_ [ “][ 0][|] _[x]_ [q] _[p]_ _[s]_ [p] _[x]_ [q] _[dx]_ [ `] ż p1 ´ _α_ q



“ I r _f_ p _x_ q ‰ 0s _[β]_
ż _α_



p1 ´ _α_ q _[p]_ _[s]_ [p] _[y]_ [ “][ 1][|] _[x]_ [q] _[p]_ _[s]_ [p] _[x]_ [q] _[dx]_



Then _E_ 1 ´ _E_ 2 is given by

_E_ 1 ´ _E_ 2 “ I r _f_ p _x_ q ‰ 0s _p_ _s_ p _y_ “ 0| _x_ q _p_ _t_ p _x_ q ´ _[β]_ _dx_
ż „ _α_ _[p]_ _[s]_ [p] _[x]_ [q] 


_[β]_ [q]
` I r _f_ p _x_ q ‰ 1s _p_ _s_ p _y_ “ 1| _x_ q _p_ _t_ p _x_ q ´ [p][1][ ´] _dx_
ż „ p1 ´ _α_ q _[p]_ _[s]_ [p] _[x]_ [q] 


_[β]_ [q]
“ I r _f_ p _x_ q ‰ 0s _p_ _s_ p _y_ “ 0| _x_ q [p] _[α]_ [ ´] _φ_ p _µ_ 2 q _dx_
ż _α_


_[β]_ [q]
` I r _f_ p _x_ q ‰ 1s _p_ _s_ p _y_ “ 1| _x_ q [p] _[α]_ [ ´] (10)
ż 1 ´ _α_ _[φ]_ [p] _[µ]_ [1] [q] _[dx .]_


16


Published as a conference paper at ICLR 2022


If _α_ ą _β_, then _E_ 1 ą _E_ 2 and if _α_ ă _β_, then _E_ 1 ă _E_ 2 . Since _E_ 1 ‰ _E_ 2 for arbitrary _f_, given access to
_p_ _s_ p _x, y_ q, and _p_ _t_ p _x_ q, any method that consistently estimates error under covariate shift will give an
incorrect estimate under label shift and vice-versa. The reason being that the same _p_ _t_ p _x_ q and _p_ _s_ p _x, y_ q
can correspond to error _E_ 1 (under covariate shift) or error _E_ 2 (under label shift) either of which is not
discernable absent further assumptions on the nature of shift.


C A LTERNATE INTERPRETATION OF ATC


Consider the following framework: Given a datum p _x, y_ q, define a binary classification problem
of whether the model prediction arg max _f_ p _x_ q was correct or incorrect. In particular, if the model
prediction matches the true label, then we assign a label 1 (positive) and conversely, if the model
prediction doesn’t match the true label then we assign a label 0 (negative).


Our method can be interpreted as identifying examples for correct and incorrect prediction based
on the value of the score function _s_ p _f_ p _x_ qq, i.e., if the score _s_ p _f_ p _x_ qq is greater than or equal to
the threshold _t_ then our method predicts that the classifier correctly predicted datum p _x, y_ q and
vice-versa if the score is less than _t_ . A method that can solve this task will perfectly estimate the
target performance. However, such an expectation is unrealistic. Instead, ATC expects that _most_ of
the examples with score above threshold are correct and most of the examples below the threshold
are incorrect. More importantly, ATC selects a threshold such that the number of falsely identified
correct predictions match falsely identified incorrect predictions on source distribution, thereby
balancing incorrect predictions. We expect useful estimates of accuracy with ATC if the threshold
transfers to target, i.e. if the number of falsely identified correct predictions match falsely identified
incorrect predictions on target. This interpretation relates our method to the OOD detection literature
where Hendrycks & Gimpel (2016); Hendrycks et al. (2019) highlight that classifiers tend to assign
higher confidence to in-distribution examples and leverage maximum softmax confidence (or logit)
to perform OOD detection.


D D ETAILS ON THE T OY M ODEL


**Skews observed in this toy model** In Fig. 4, we illustrate the toy model used in our empirical
experiment. In the same setup, we empirically observe that the margin on population with less density
is large, i.e., margin is much greater than _γ_ when the number of observed samples is small (in Fig. 4
(d)). Building on this observation, Nagarajan et al. (2020) showed in cases when margin decreases
with number of samples, a max margin classifier trained on finite samples is bound to depend on the
spurious features in such cases. They referred to this skew as _geometric skew_ .


Moreover, even when the number of samples are large so that we do not observe geometric skews,
Nagarajan et al. (2020) showed that training for finite number of epochs, a linear classifier will have a
non zero dependency on the spurious feature. They referred to this skew as _statistical skew_ . Due both
of these skews, we observe that a linear classifier obtained with training for finite steps on training
data with finite samples, will have a non-zero dependency on the spurious feature. We refer interested
reader to Nagarajan et al. (2020) for more details.


**Proof of Theorem 1** Recall, we consider a easy-to-learn binary classification problem with two
features _x_ “ r _x_ inv _, x_ sp s P R [2] where _x_ inv is fully predictive invariant feature with a margin _γ_ ą 0 and
_x_ sp P t´1 _,_ 1u is a spurious feature (i.e., a feature that is correlated but not predictive of the true label).
Conditional on _y_, the distribution over _x_ inv is given as follows:


_U_ r _γ, c_ s _y_ “ 1
_x_ inv | _y_ „ (11)
" _U_ r´ _c,_ ´ _γ_ s _y_ “ ´1 _[,]_


where _c_ is a fixed constant greater than _γ_ . For simplicity, we assume that label distribution on source
is uniform on t´1 _,_ 1u . _x_ sp is distributed such that _P_ _s_ r _x_ sp ¨p2 _y_ ´ 1q ą 0s “ _p_ sp, where _p_ sp P p0 _._ 5 _,_ 1 _._ 0q
controls the degree of spurious correlation. To model distribution shift, we simulate target data with
different degree of spurious correlation, i.e., in target distribution _P_ _t_ r _x_ sp ¨p2 _y_ ´1q ą 0s “ _p_ [1] sp [P r][0] _[,]_ [ 1][s] [.]
Note that here we do not consider shifts in the label distribution but our result extends to arbitrary
shifts in the label distribution as well.


17


Published as a conference paper at ICLR 2022


(a) (b)


(c) (d)


Figure 4: Illustration of toy model. (a) Source data at _n_ “ 100 . (b) Target data with _p_ [1] _s_ [“][ 0] _[.]_ [5] [. (b)]
Target data with _p_ [1] _s_ [“][ 0] _[.]_ [9] [. (c) Margin of] _[ x]_ [inv] [in the minority group in source data. As sample size]
increases the margin saturates to true margin _γ_ “ 0 _._ 1.


1 _e_ _[wT x]_
In this setup, we examine linear sigmoid classifiers of the form _f_ p _x_ q “ ” 1` _e_ _[wT x]_ _[,]_ 1` _e_ _[wT x]_ ı where

_w_ “ r _w_ inv _, w_ sp s P R [2] . We show that given a linear classifier that relies on the spurious feature and
achieves a non-trivial performance on the source (i.e., _w_ inv ą 0 ), ATC with maximum confidence
score function _consistently_ estimates the accuracy on the target distribution. Define _X_ _M_ “ t _x_ | _x_ sp ¨
p2 _y_ ´ 1q ă 0u and _X_ _C_ “ t _x_ | _x_ sp ¨ p2 _y_ ´ 1q ą 0u . Notice that in target distributions, we are changing
the fraction of examples in _X_ _M_ and _X_ _C_ but we are not changing the distribution of examples within
individual set.


**Theorem 3.** _Given any classifier_ _f_ _with_ _w_ _inv_ ą 0 _in the above setting, assume that the threshold_ _t_ _is_
_obtained with finite sample approximation of_ (1) _, i.e., t is selected such that_ [2]



_n_
ÿ

_i_ “1



„I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ _[i]_ [q ă] _[ t]_  “



_n_
ÿ

_i_ “1



I arg max _f_ _j_ p _x_ _i_ q ‰ _y_ _i_ _,_ (12)
„ „ _j_ P _Y_ 



_where_ tp _x_ _i_ _, y_ _i_ qu _[n]_ _i_ “1 [„ p] _[D]_ _[S]_ [q] _[n]_ _[ are]_ _[ n]_ _[ samples from source distribution. Fix a]_ _[ δ]_ [ ą][ 0] _[. Assuming]_
_n_ ě 2 logp4{ _δ_ q{p1 ´ _p_ _sp_ q [2] _, then the estimate of accuracy by ATC as in_ (2) _satisfies the following_
_with probability at least_ 1 ´ _δ,_



E _x_ „ _D_ _T_ rI r _s_ p _f_ p _x_ qq ă _t_ ss ´ E p _x,y_ q„ _D_ _T_ I arg max _f_ _j_ p _x_ q ‰ _y_ ď
���� „ „ _j_ P _Y_ ����



~~d~~



logp8{ _δ_ q

_,_ (13)
_n_ ¨ _c_ _sp_



_where_ _D_ _[T]_ _is any target distribution considered in our setting and_ _c_ _sp_ “ p1 ´ _p_ _sp_ q _if_ _w_ _sp_ ą 0 _and_
_c_ _sp_ “ _p_ _sp_ _otherwise._


2 Note that this is possible because a linear classifier with sigmoid activation assigns a unique score to each
point in source distribution.


18


Published as a conference paper at ICLR 2022


_Proof._ First we consider the case of _w_ sp ą 0 . The proof follows in two simple steps. First we notice
that the classifier will make an error only on some points in _X_ _M_ and the threshold _t_ will be selected
such that the fraction of points in _X_ _M_ with maximum confidence less than the threshold _t_ will match
the error of the classifier on _X_ _M_ . Classifier with _w_ sp ą 0 and _w_ inv ą 0 will classify all the points in
_X_ _C_ correctly. Second, since the distribution of points is not changing within _X_ _M_ and _X_ _C_, the same
threshold continues to work for arbitrary shift in the fraction of examples in _X_ _M_, i.e., _p_ [1] sp [.]


Note that when _w_ sp ą 0, the classifier makes no error on points in _X_ _C_ and makes an error on a
subset _X_ err “ t _x_ | _x_ sp ¨ p2 _y_ ´ 1q ă 0 & p _w_ inv _x_ inv ` _w_ sp _x_ sp q ¨ p2 _y_ ´ 1q ď 0u of _X_ _M_, i.e., _X_ err Ď _X_ _M_ .
Consider _X_ thres “ t _x_ | arg max _y_ P _Y_ _f_ _y_ p _x_ q ď _t_ u as the set of points that obtain a score less than or
equal to _t_ . Now we will show that ATC chooses a threshold _t_ such that all points in _X_ _C_ gets a score
above _t_, i.e., _X_ thres Ď _X_ _M_ . First note that the score of points close to the true separator in _X_ _C_, i.e., at
_x_ 1 “ p _γ,_ 1q and _x_ 2 “ p´ _γ,_ ´1q match. In other words, score at _x_ 1 matches with the score of _x_ 2 by
symmetricity, i.e.,


_e_ _[w]_ [inv] _[γ]_ [`] _[w]_ [sp]
arg max _y_ P _Y_ _f_ _y_ p _x_ 1 q “ arg max _y_ P _Y_ _f_ _y_ p _x_ 2 q “ p1 ` _e_ _[w]_ [inv] _[γ]_ [`] _[w]_ [sp] q _[.]_ (14)


Hence, if _t_ ě arg max _y_ P _Y_ _f_ _y_ p _x_ 1 q then we will have _|X_ err _|_ ă _|X_ thres _|_ which is contradiction violating
definition of _t_ as in (12). Thus _X_ thres Ď _X_ _M_ .



Now we will relate LHS and RHS of (12) with their expectations using Hoeffdings and DKW
inequality to conclude (13). Using Hoeffdings’ bound, we have with probability at least 1 ´ _δ_ {4
����� _i_ P ÿ _X_ _M_ “I “arg max _j_ _|X_ P _YM_ _f|_ _j_ p _x_ _i_ q ‰ _y_ _i_ ‰‰ ´ E p _x,y_ q„ _D_ T „I „arg max _j_ P _Y_ _f_ _j_ p _x_ q ‰ _y_  [�] ���� ď ~~d~~ lo2 _|_ g _X_ p8 _M_ { _δ|_ q _[.]_



~~d~~



ÿ



“I “arg max _j_ P _Y_ _f_ _j_ p _x_ _i_ q ‰ _y_ _i_ ‰‰



logp8{ _δ_ q



_i_ P _X_ _M_



_j_ P _Y_ _f_ _j_ p _x_ _i_ q ‰ _y_ _i_ ‰‰ ´ E p _x,y_ q„ _D_ T I arg max _f_ _j_ p _x_ q ‰ _y_ ď

_|X_ _M_ _|_ „ „ _j_ P _Y_  [�] ����



2 _|X_ _M_ _|_ _[.]_



(15)



With DKW inequality, we have with probability at least 1 ´ _δ_ {4
����� _i_ P ÿ _X_ _M_ rI rmax _j_ P _|_ _Y_ _Xf_ _Mj_ p _|x_ _i_ q ă _t_ [1] ss ´ E p _x,y_ q„ _D_ T „I „max _j_ P _Y_ _[f]_ _[j]_ [p]



~~d~~



logp8{ _δ_ q

(16)
2 _|X_ _M_ _|_ _[,]_



ÿ



rI rmax _j_ P _Y_ _f_ _j_ p _x_ _i_ q ă _t_ [1] ss



logp8{ _δ_ q



_i_ P _X_ _M_



_|_ _Y_ _Xf_ _Mj_ p _|x_ _i_ q ă _t_ [1] ss ´ E p _x,y_ q„ _D_ T „I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ [q ă] _[ t]_ [1]  [�] ď
����



for all _t_ [1] ą 0 . Combining (15) and (16) at _t_ [1] “ _t_ with definition (12), we have with probability at
least 1 ´ _δ_ {2


logp8{ _δ_ q

E _x_ „ _D_ T rI r _s_ p _f_ p _x_ qq ă _t_ ss ´ E p _x,y_ q„ _D_ T I arg max _f_ _j_ p _x_ q ‰ _y_ ď (17)
���� „ „ _j_ P _Y_ ���� ~~d~~ 2 _|X_ _M_ _|_ _[.]_



~~d~~



logp8{ _δ_ q



(17)
2 _|X_ _M_ _|_ _[.]_



Now for the case of _w_ sp ă 0, we can use the same arguments on _X_ _C_ . That is, since now all the error
will be on points in _X_ _C_ and classifier will make no error _X_ _M_, we can show that threshold _t_ will be
selected such that the fraction of points in _X_ _C_ with maximum confidence less than the threshold _t_ will
match the error of the classifier on _X_ _C_ . Again, since the distribution of points is not changing within
_X_ _M_ and _X_ _C_, the same threshold continues to work for arbitrary shift in the fraction of examples in
_X_ _M_, i.e., _p_ [1] sp [. Thus with similar arguments, we have]


logp8{ _δ_ q

E _x_ „ _D_ T rI r _s_ p _f_ p _x_ qq ă _t_ ss ´ E p _x,y_ q„ _D_ T I arg max _f_ _j_ p _x_ q ‰ _y_ ď (18)
���� „ „ _j_ P _Y_ ���� ~~d~~ 2 _|X_ _C_ _|_ _[.]_



~~d~~



logp8{ _δ_ q



(18)
2 _|X_ _C_ _|_ _[.]_



Using Hoeffdings’ bound, with probability at least 1 ´ _δ_ {2, we have



_|X_ _M_ ´ _n_ ¨ p1 ´ _p_ sp q _|_ ď


With probability at least 1 ´ _δ_ {2, we have



~~c~~



_n_ ¨ _log_ p4{ _δ_ q

_._ (19)
2



_|X_ _C_ ´ _n_ ¨ _p_ sp _|_ ď



~~c~~



_n_ ¨ _log_ p4{ _δ_ q

_._ (20)
2



Combining (19) and (17), we get the desired result for _w_ sp ą 0 . For _w_ sp ă 0, we combine (20) and
(18) to get the desired result.


19


Published as a conference paper at ICLR 2022

**i**



(a)


Figure 5: Failure of ATC in our toy model. Shifting the support of target class conditional _p_ _t_ p _x_ inv | _y_ q
may introduce a bias in ATC estimates, e.g., shrinking the support to _c_ 1 ( ă _c_ ) (while maintaining
uniform distribution) in the target leads to overestimation bias.


**Issues with IM in toy setting** As described in App. E, we observe that IM is sensitive to binning
strategy. In the main paper, we include IM result with uniform mass binning with 100 bins. Empirically, we observe that we recover the true performance with IM if we use equal width binning with
number of bins greater than 5.


**Biased estimation with ATC in our toy model** We assumed that both in source and target _x_ inv | _y_ “ 1
is uniform between r _γ, c_ s and _x_ | _y_ “ ´1 is uniform between r´ _c,_ ´ _γ_ s . Shifting the support of target
class conditional _p_ _t_ p _x_ inv | _y_ q may introduce a bias in ATC estimates, e.g., shrinking the support to
_c_ 1 ( ă _c_ ) (while maintaining uniform distribution) in the target will lead to an over-estimation of the
target performance with ATC. We show this failure in Fig. 5. The reason being that with the same
threshold that we see more examples falsely identified as correct as compared to examples falsely
identified as incorrect.


D.1 A M ORE G ENERAL R ESULT


Recall, for a given threshold _t_, we categorize an example p _x, y_ q as a falsely identified correct
prediction (ficp) if the predicted label p _y_ “ arg max _f_ p _x_ q is not the same as _y_ but the predicted score
_f_ _y_ p p _x_ q is greater than _t_ . Similarly, an example is falsely identified incorrect prediction (f **i** p) if the
predicted label p _y_ is the same as _y_ but the predicted score _f_ _y_ p p _x_ q is less than _t_ .


In general, we believe that our method will obtain consistent estimates in scenarios where the relative
distribution of covariates doesn’t change among examples that are falsely identified as incorrect
and examples that are falsely identified as correct. In other words, ATC is expected to work if the
distribution shift is such that falsely identified incorrect predictions match falsely identified correct
prediction.


D.2 ATC PRODUCES CONSISTENT ESTIMATE ON SOURCE DISTRIBUTION


**Proposition 2.** _Given labeled validation data_ tp _x_ _i_ _, y_ _i_ qu _[n]_ _i_ “1 _[from a distribution]_ _[ D]_ _[S]_ _[ and a model]_ _[ f]_ _[,]_
_choose a threshold t as in_ (1) _. Then for δ_ ą 0 _, with probability at least_ 1 ´ _δ, we have_



**i**


E p _x,y_ q„ _D_



**i**


I max ´ I arg max _f_ _j_ p _x_ q ‰ _y_ ď 2
„ „ _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ [q ă] _[ t]_  „ _j_ P _Y_  ~~c~~



**i**


logp4{ _δ_ q

(21)
2 _n_



**i**


logp4{ _δ_ q



**i**


_Proof._ The proof uses (i) Hoeffdings’ inequality to relate the accuracy with expected accuracy; and
(ii) DKW inequality to show the concentration of the estimated accuracy with our proposed method.
Finally, we combine (i) and (ii) using the fact that at selected threshold _t_ the number of false positives
is equal to the number of false negatives.



**i**


Using Hoeffdings’ bound, we have with probability at least 1 ´ _δ_ {2


_n_
ÿ I arg max _f_ _j_ p _x_ _i_ q ‰ _y_ _i_ ´ E p _x,y_ q„ _D_ I arg max _f_ _j_ p _x_

����� _i_ “1 „ „ _j_ P _Y_  „ „ _j_ P _Y_



**i**


_n_
ÿ

_i_ “1



**i**


I arg max _f_ _j_ p _x_ _i_ q ‰ _y_ _i_
„ „ _j_ P _Y_



**i**


´ E p _x,y_ q„ _D_




**i**


I arg max _f_ _j_ p _x_ q ‰ _y_ ď
„ „ _j_ P _Y_  [�] ����



**i**


~~c~~



**i**


logp4{ _δ_ q

_._ (22)
2 _n_



**i**


20


Published as a conference paper at ICLR 2022



With DKW inequality, we have with probability at least 1 ´ _δ_ {2


_n_

����� _i_ ÿ “1 „I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ _[i]_ [q ă] _[ t]_ [1]  ´ E p _x,y_ q„ _D_ „I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_



_n_
ÿ

_i_ “1



„I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ _[i]_ [q ă] _[ t]_ [1]  ´ E p _x,y_ q„ _D_



„I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ [q ă] _[ t]_ [1]  [�] ď
����



~~c~~



logp4{ _δ_ q

2 _n_ _,_ (23)



for all _t_ [1] ą 0. Finally by definition, we have



_n_
ÿ

_i_ “1



„I „max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ _[i]_ [q ă] _[ t]_ [1]  “



_n_
ÿ

_i_ “1



I arg max _f_ _j_ p _x_ _i_ q ‰ _y_ _i_
„ „ _j_ P _Y_



(24)




Combining (22), (23) at _t_ [1] “ _t_, and (24), we have the desired result.


E B ASLINE M ETHODS


**Importance-re-weighting (IM)** If we can estimate the importance-ratios _p_ _[p]_ _s_ _[t]_ [p] p _[x]_ _x_ [q] q [with just the unla-]

beled data from the target and validation labeled data from source, then we can estimate the accuracy
as on target as follows:


_p_ _t_ p _x_ q
_E_ _D_ T p _f_ q “ E p _x,y_ q„ _D_ S _._ (25)
„ _p_ _s_ p _x_ q [I][ r] _[f]_ [p] _[x]_ [q ‰] _[ y]_ [s] 


As previously discussed, this is particularly useful in the setting of covariate shift (within support)
where importance ratios estimation has been explored in the literature in the past. Mandolin (Chen
et al., 2021b) extends this approach. They estimate importance-weights with use of extra supervision
about the axis along which the distribution is shifting.


In our work, we experiment with uniform mass binning and equal width binning with the number of
bins in r5 _,_ 10 _,_ 50s . Overall, we observed that equal width binning works the best with 10 bins. Hence
throughout this paper we perform equal width binning with 10 bins to include results with IM.


**Average Confidence (AC)** If we expect the classifier to be argmax calibrated on the target then
average confidence is equal to accuracy of the classifier. Formally, by definition of argmax calibration
of _f_ on any distribution _D_, we have



„max _j_ P _Y_ _[f]_ _[j]_ [p] _[x]_ [q]  _._ (26)



_E_ _D_ p _f_ q “ E p _x,y_ q„ _D_



I _y_ R arg max _f_ _j_ p _x_ q “ E p _x,y_ q„ _D_
„ „ _j_ P _Y_ 



**Difference Of Confidence** We estimate the error on target by subtracting difference of confidences
on source and target (as a distributional distance (Guillory et al., 2021)) from expected error on
source distribution, i.e, DOC _D_ T “ E _x_ „ _D_ S “I “arg max _j_ P _Y_ _f_ _j_ p _x_ q ‰ _y_ ‰‰ ` E _x_ „ _D_ T rmax _j_ P _Y_ _f_ _j_ p _x_ qs ´
E _x_ „ _D_ S rmax _j_ P _Y_ _f_ _j_ p _x_ qs. This is referred to as DOC-Feat in (Guillory et al., 2021).


**Generalized Disagreement Equality (GDE)** Jiang et al. (2021) proposed average disagreement of
two models (trained on the same training set but with different initialization and/or different data
ordering) as a approximate measure of accuracy on the underlying data, i.e.,


_E_ _D_ p _f_ q “ E p _x,y_ q„ _D_ “I “ _f_ p _x_ q ‰ _f_ [1] p _x_ q‰‰ _._ (27)


They show that marginal calibration of the model is sufficient to have expected test error equal to the
expected of average disagreement of two models where the latter expectation is also taken over the
models used to calculate disagreement.


F D ETAILS ON THE D ATASET S ETUP


In our empirical evaluation, we consider both natural and synthetic distribution shifts. We consider shifts on ImageNet (Russakovsky et al., 2015), CIFAR Krizhevsky & Hinton (2009), FMoWW ILDS (Christie et al., 2018), RxRx1-W ILDS (Taylor et al., 2019), Amazon-W ILDS (Ni et al., 2019),
CivilComments-W ILDS (Borkan et al., 2019), and MNIST LeCun et al. (1998) datasets.


21


Published as a conference paper at ICLR 2022


Train (Source) Valid (Source) Evaluation (Target)


MNIST (train) MNIST (valid) USPS, SVHN and Q-MNIST

CIFAR10 (train) CIFAR10 (valid) CIFAR10v2, 95 CIFAR10-C datasets (Fog and Motion blur, etc. )
CIFAR100 (train) CIFAR100 (valid) 95 CIFAR100-C datasets (Fog and Motion blur, etc. )


FMoW _{_ (2013-15, 2016-17) ˆ
FMoW (2002-12) (train) FMoW (2002-12) (valid)
(All, Africa, Americas, Oceania, Asia, and Europe) _}_


RxRx1 (train) RxRx1(id-val) RxRx1 (id-test, OOD-val, OOD-test)

Amazon (train) Amazon (id-val) Amazon (OOD-val, OOD-test)


CiviComments (8 demographic identities male, female, LGBTQ,
CivilComments (train) CivilComments (id-val)
Christian, Muslim, other religions, Black, and White)


3 ImageNetv2 datasets, ImageNet-Sketch,
ImageNet (train) ImageNet (valid)
95 ImageNet-C datasets


3 ImageNet-200v2 datasets, ImageNet-R,
ImageNet-200 (train) ImageNet-200 (valid)
ImageNet200-Sketch, 95 ImageNet200-C datasets


Same subpopulations as train but unseen images from natural
B REEDS (train) B REEDS (valid) and synthetic shifts in ImageNet, Novel subpopulations on
natural and synthetic shifts


Table 2: Details of the test datasets considered in our evaluation.


_ImageNet setup._ First, we consider synthetic shifts induced to simulate 19 different visual corruptions
(e.g., shot noise, motion blur, pixelation etc.) each with 5 different intensities giving us a total of 95
datasets under ImageNet-C (Hendrycks & Dietterich, 2019). Next, we consider natural distribution
shifts due to differences in the data collection process. In particular, we consider 3 ImageNetv2 (Recht
et al., 2019) datasets each using a different strategy to collect test sets. We also evaluate performance
on images with artistic renditions of object classes, i.e., ImageNet-R (Hendrycks et al., 2021) and
ImageNet-Sketch (Wang et al., 2019) with hand drawn sketch images. Note that renditions dataset
only contains 200 classes from ImageNet. Hence, in the main paper we include results on ImageNet
restricted to these 200 classes, which we call as ImageNet-200, and relegate results on ImageNet with
1k classes to appendix.


We also consider B REEDS benchmark (Santurkar et al., 2020) in our evaluation to assess robustness
to subpopulation shifts, in particular, to understand how accuracy estimation methods behave when
novel subpopulations not observed during training are introduced. B REEDS leverages class hierarchy
in ImageNet to repurpose original classes to be the subpopulations and defines a classification task
on superclasses. Subpopulation shift is induced by directly making the subpopulations present
in the training and test distributions disjoint. Overall, B REEDS benchmark contains 4 datasets
E NTITY -13, E NTITY -30, L IVING -17, N ON - LIVING -26, each focusing on different subtrees in the
hierarchy. To generate B REEDS dataset on top of ImageNet, we use the open source library: [https:](https://github.com/MadryLab/BREEDS-Benchmarks)
[//github.com/MadryLab/BREEDS-Benchmarks](https://github.com/MadryLab/BREEDS-Benchmarks) . We focus on natural and synthetic shifts
as in ImageNet on same and different subpopulations in BREEDs. Thus for both the subpopulation
(same or novel), we obtain a total of 99 target datasets.


_CIFAR setup._ Similar to the ImageNet setup, we consider (i) synthetic shifts (CIFAR-10-C) due to
common corruptions; and (ii) natural distribution shift (i.e., CIFARv2 (Recht et al., 2018; Torralba
et al., 2008)) due to differences in data collection strategy on on CIFAR-10 (Krizhevsky & Hinton,
2009). On CIFAR-100, we just have synthetic shifts due to common corruptions.


_FMoW-_ W ILDS _setup._ In order to consider distribution shifts faced in the wild, we consider FMoWWILDS (Koh et al., 2021; Christie et al., 2018) from W ILDS benchmark, which contains satellite
images taken in different geographical regions and at different times. We obtain 12 different OOD
target sets by considering images between years 2013 – 2016 and 2016 – 2018 and by considering five
geographical regions as subpopulations (Africa, Americas, Oceania, Asia, and Europe) separately
and together.


_RxRx1–_ W ILDS _setup._ Similar to FMoW, we consider RxRx1-W ILDS (Taylor et al., 2019) from
W ILDS benchmark, which contains image of cells obtained by fluorescent microscopy and the task


22


Published as a conference paper at ICLR 2022


is to genetic treatments the cells received. We obtain 3 target datasets with shift induced by batch
effects which make it difficult to draw conclusions from data across experimental batches.


_Amazon-_ W ILDS _setup._ For natural language task, we consider Amazon-W ILDS (Ni et al., 2019)
dataset from W ILDS benchmark, which contains review text and the task is get a corresponding star
rating from 1 to 5 . We obtain 2 target datasets by considered shifts induced due to different set of
reviewers than the training set.


_CivilComments-_ W ILDS _setup._ We also consider CivilComments-W ILDS (Borkan et al., 2019) from
W ILDS benchmark, which contains text comments and the task is to classify them for toxicity. We
obtain 18 target datasets depending on whether a comment mentions each of the 8 demographic
identities male, female, LGBTQ, Christian, Muslim, other religions, Black, and White.


_MNIST setup._ For completeness, we also consider distribution shifts on MNIST (LeCun et al., 1998)
digit classification as in the prior work (Deng & Zheng, 2021). We use three real shifted datasets, i.e.,
USPS (Hull, 1994), SVHN (Netzer et al., 2011) and QMNIST (Yadav & Bottou, 2019).


G D ETAILS ON THE E XPERIMENTAL S ETUP


All experiments were run on NVIDIA Tesla V100 GPUs. We used PyTorch (Paszke et al., 2019) for
experiments.


**Deep nets** We consider a 4-layered MLP. The PyTorch code for 4-layer MLP is as follows:


nn.Sequential(nn.Flatten(),
nn.Linear(input ~~d~~ im, 5000, bias=True),
nn.ReLU(),
nn.Linear(5000, 5000, bias=True),
nn.ReLU(),
nn.Linear(5000, 50, bias=True),
nn.ReLU(),
nn.Linear(50, num ~~l~~ abel, bias=True)
)


We mainly experiment convolutional nets. In particular, we use ResNet18 (He et al., 2016), ResNet50,
and DenseNet121 (Huang et al., 2017) architectures with their default implementation in PyTorch.
Whenever we initial our models with pre-trained models, we again use default models in PyTorch.


**Hyperparameters and Training details** As mentioned in the main text we do not alter the standard
training procedures and hyperparameters for each task. We present results at final model, however,
we observed that the same results extend to an early stopped model as well. For completeness, we
include these details below:


_CIFAR10 and CIFAR100_ We train DenseNet121 and ResNet18 architectures from scratch. We use
SGD training with momentum of 0 _._ 9 for 300 epochs. We start with learning rate 0 _._ 1 and decay it by
multiplying it with 0 _._ 1 every 100 epochs. We use a weight decay of 5 [´] 4 . We use batch size of 200 .
For CIFAR10, we also experiment with the same models pre-trained on ImageNet.


_ImageNet_ For training, we use Adam with a batch size of 64 and learning rate 0 _._ 0001 . Due to
huge size of ImageNet, we could only train two models needed for GDE for 10 epochs. Hence, for
relatively small scale experiments, we also perform experiments on ImageNet subset with 200 classes,
which we call as ImageNet-200 with the same training procedure. These 200 classes are the same
classes as in ImageNet-R dataset. This not only allows us to train ImageNet for 50 epochs but also
allows us to use ImageNet-R in our testbed. On the both the datasets, we observe a similar superioriy
with ATC. Note that all the models trained here were initialized with a pre-trained ImageNet model
with the last layer replaced with random weights.


_FMoW-_ WILDS For all experiments, we follow Koh et al. (2021) and use two architectures
DenseNet121 and ResNet50, both pre-trained on ImageNet. We use the Adam optimizer (Kingma &
Ba, 2014) with an initial learning rate of 10 [´][4] that decays by 0 _._ 96 per epoch, and train for 50 epochs
and with a batch size of 64.


23


Published as a conference paper at ICLR 2022


_RxRx1-_ WILDS For all experiments, we follow Koh et al. (2021) and use two architectures
DenseNet121 and ResNet50, both pre-trained on ImageNet. We use Adam optimizer with a learning
rate of 1 _e_ ´ 4 and L2-regularization strength of 1 _e_ ´ 5 with a batch size of 75 for 90 epochs. We
linearly increase the learning rate for 10 epochs, then decreasing it following a cosine learning rate
schedule. Finally, we pick the model that obtains highest in-distribution validation accuracy.


_Amazon-_ WILDS For all experiments, we follow Koh et al. (2021) and finetuned DistilBERTbase-uncased models (Sanh et al., 2019), using the implementation from Wolf et al. (2020), and
with the following hyperparameter settings: batch size 8 ; learning rate 1 _e_ ´ 5 with the AdamW
optimizer (Loshchilov & Hutter, 2017); L2-regularization strength 0 _._ 01 ; 3 epochs with early stopping;
and a maximum number of tokens of 512.


_CivilComments-_ WILDS For all experiments, we follow Koh et al. (2021) and fine-tuned DistilBERTbase-uncased models (Sanh et al., 2019), using the implementation from Wolf et al. (2020) and
with the following hyperparameter settings: batch size 16 ; learning rate 1 _e_ ´ 5 with the AdamW
optimizer (Loshchilov & Hutter, 2017) for 5 epochs; L2-regularization strength 0 _._ 01 ; and a maximum
number of tokens of 300.


_Living17 and Nonliving26 from_ B REEDS For training, we use SGD with a batch size of 128, weight
decay of 10 [´][4], and learning rate 0 _._ 1 . Models were trained until convergence. Models were trained
for a total of 450 epochs, with 10-fold learning rate drops every 150 epochs. Note that since we want
to evaluate models for novel subpopulations no pre-training was used. We train two architectures
DenseNet121 and ResNet50.


_Entity13 and Entity30 from_ B REEDS For training, we use SGD with a batch size of 128, weight
decay of 10 [´][4], and learning rate 0 _._ 1 . Models were trained until convergence. Models were trained
for a total of 300 epochs, with 10-fold learning rate drops every 100 epochs. Note that since we want
to evaluate models for novel subpopulations no pre-training was used. We train two architectures
DenseNet121 and ResNet50.


_MNIST_ For MNIST, we train a MLP described above with SGD with momentum 0 _._ 9 and learning
rate 0 _._ 01 for 50 epochs. We use weight decay of 10 [´][5] and batch size as 200.


We have a single number for CivilComments because it is a binary classification task. For multiclass
problems, ATC-NE and ATC-MC can lead to different ordering of examples when ranked with
the corresponding scoring function. Temperature scaling on top can further alter the ordering of
examples. The changed ordering of examples yields different thresholds and different accuracy
estimates. However for binary classification, the two scoring functions are the same as entropy
(i.e. _p_ logp _p_ q ` p1 ´ _p_ q logp _p_ q ) has a one-to-one mapping to the max conf for _p_ P r0 _,_ 1s . Moreover,
temperature scaling also doesn’t change the order of points for binary classification problems. Hence
for the binary classification problems, both the scoring functions with and without temperature scaling
yield the same estimates. We have made this clear in the updated draft.


**Implementation for Temperature Scaling** We use temperature scaling implementation from
[https://github.com/kundajelab/abstention](https://github.com/kundajelab/abstention) . We use validation set (the same we use
to obtain ATC threshold or DOC source error estimate) to tune a single temperature parameter.


G.1 D ETAILS ON F IG . 1 ( RIGHT ) SETUP


For vision datasets, we train a DenseNet model with the exception of FCN model for MNIST dataset.
For language datasets, we fine-tune a DistilBERT-base-uncased model. For each of these models,
we use the exact same setup as described Sec. G. Importantly, to obtain errors on the same scale, we
rescale all the errors by subtracting the error of Average Confidence method for each model. Results
are reported as mean of the re-scaled errors over 4 seeds.


24


Published as a conference paper at ICLR 2022


H S UPPLEMENTARY R ESULTS


H.1 CIFAR PRETRAINING A BLATION


CIFAR10-Pretraining



90

80

70

60

50

40

30

20




|Col1|Col2|
|---|---|
|||
|DOC||
|GDE<br>||
|~~ATC~~<br>|~~-NE (Ours)~~|
|~~IM~~<br>AC||
|||



20 40 60 80
OOD Accuracy


(a)


Figure 6: Results with a pretrained DenseNet121 model on CIFAR10. We observe similar behaviour
as that with a model trained from scratch.


H.2 B REEDS RESULTS WITH REGRESSION MODEL


|Col1|Col2|Col3|
|---|---|---|
||DOC (w/<br>DOC (w/|o ﬁt)<br> ﬁt)|
||ATC (w/o|ﬁt)|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||DOC (w/<br>DOC (w/|o ﬁt)<br> ﬁt)|
|||ATC (w/o|ﬁt)|





Entity30



90 i

80 i

70 i

60

50

40

30

20



Entity13


0 20 40 60 80
OOD Accuracy



90 i

80 i

70 i

60

50

40

30

20



Nonliving26


0 20 40 60 80
OOD Accuracy





i 90


i 80


i 70


60

50

40

30

20



Figure 7: Scatter plots for DOC with linear fit. Results parallel to Fig. 3(Middle) on other
B REEDS dataset.


Dataset DOC (w/o fit) DOC (w fit) ATC-MC (Ours) (w/o fit)


L IVING -17 24 _._ 32 13 _._ 65 **10** _._ **07**


N ONLIVING -26 29 _._ 91 **18** _._ **13** 19 _._ 37


E NTITY -13 22 _._ 18 8 _._ 63 8 _._ 01


E NTITY -30 24 _._ 71 12 _._ 28 **10** _._ **21**


Table 5: _Mean Absolute estimation Error (MAE) results for BREEDs datasets with novel populations_
_in our setup._ After fitting a robust linear model for DOC on same subpopulation, we show predicted
accuracy on different subpopulations with fine-tuned DOC (i.e., DOC (w/ fit)) and compare with
ATC without any regression model, i.e., ATC (w/o fit). While observe substantial improvements in
MAE from DOC (w/o fit) to DOC (w/ fit), ATC (w/o fit) continues to outperform even DOC (w/ fit).


25


Published as a conference paper at ICLR 2022


CIFAR10



CIFAR100



ImageNet200





90

80

70

60

50

40

30


70

60

50

40

30

20

10

0











40 60 80
OOD Accuracy


ImageNet



20 40 60 80
OOD Accuracy


FMoW-Wilds



20 40 60 80
OOD Accuracy


RxRx1-Wilds









0 20 40 60
OOD Accuracy


Amazon-Wilds



40 50 60 70
OOD Accuracy


CivilComments-Wilds



20 25 30 35
OOD Accuracy


MNIST



90.0

87.5

85.0

82.5

80.0

77.5

75.0

72.5


80


60


40


20


0







80


60


40


20


37.5

35.0

32.5

30.0

27.5

25.0

22.5

20.0

17.5


100


80


60


40


20


80


60


40


20





20 40 60 80 100
OOD Accuracy


entity30



75 80 85 90
OOD Accuracy


living17



80

70

60

50

40

30

20

10


75

70

65

60

55

50

45

40

35


90


80


70


60


80


60


40


20



60 70 80 90
OOD Accuracy


entity13





0 20 40 60 80
OOD Accuracy



20 40 60 80
OOD Accuracy



20 40 60 80
OOD Accuracy



Figure 8: Scatter plot of predicted accuracy versus (true) OOD accuracy. For vision datasets except
MNIST we use a DenseNet121 model. For MNIST, we use a FCN. For language datasets, we use
DistillBert-base-uncased. Results reported by aggregating accuracy numbers over 4 different seeds.


26


Published as a conference paper at ICLR 2022


CIFAR10



CIFAR100



ImageNet200





90

80

70

60

50

40

30

20


70

60

50

40

30

20

10

0


80

70

60

50

40

30

20

10

0







80


60


40


20


0





20 40 60 80
OOD Accuracy


ImageNet



20 40 60 80
OOD Accuracy


FMoW-Wilds



0 20 40 60 80
OOD Accuracy


RxRx1-Wilds









0 20 40 60
OOD Accuracy


Living17



40 50 60
OOD Accuracy


Nonliving26



20 25 30 35
OOD Accuracy


Entity13







37.5

35.0

32.5

30.0

27.5

25.0

22.5

20.0

17.5


90

80

70

60

50

40

30

20

10





0 20 40 60 80
OOD Accuracy



80

70

60

50

40

30

20

10


65

60

55

50

45

40

35


80

70

60

50

40

30

20

10


80

70

60

50

40

30

20

10



20 40 60 80
OOD Accuracy


Entity30



20 40 60 80
OOD Accuracy



20 40 60 80
OOD Accuracy


Figure 9: Scatter plot of predicted accuracy versus (true) OOD accuracy for vision datasets except
MNIST with a ResNet50 model. Results reported by aggregating MAE numbers over 4 different
seeds.


27


Published as a conference paper at ICLR 2022


IM AC DOC GDE ATC-MC (Ours) ATC-NE (Ours)
Dataset Shift
Pre T Post T Pre T Post T Pre T Post T Post T Pre T Post T Pre T Post T


6 _._ 60 5 _._ 74 9 _._ 88 6 _._ 89 7 _._ 25 6 _._ 07 4 _._ 77 3 _._ 21 3 _._ 02 2 _._ 99 **2** _._ **85**
Natural
p0 _._ 35q p0 _._ 30q p0 _._ 16q p0 _._ 13q p0 _._ 15q p0 _._ 16q p0 _._ 13q p0 _._ 49q p0 _._ 40q p0 _._ 37q p0 _._ 29q
CIFAR10

12 _._ 33 10 _._ 20 16 _._ 50 11 _._ 91 13 _._ 87 11 _._ 08 6 _._ 55 4 _._ 65 4 _._ 25 4 _._ 21 **3** _._ **87**
Synthetic
p0 _._ 51q p0 _._ 48q p0 _._ 26q p0 _._ 17q p0 _._ 18q p0 _._ 17q p0 _._ 35q p0 _._ 55q p0 _._ 55q p0 _._ 55q p0 _._ 75q


13 _._ 69 11 _._ 51 23 _._ 61 13 _._ 10 14 _._ 60 10 _._ 14 9 _._ 85 5 _._ 50 **4** _._ **75** **4** _._ **72** 4 _._ 94
CIFAR100 Synthetic
p0 _._ 55q p0 _._ 41q p1 _._ 16q p0 _._ 80q p0 _._ 77q p0 _._ 64q p0 _._ 57q p0 _._ 70q p0 _._ 73q p0 _._ 74q p0 _._ 74q


12 _._ 37 8 _._ 19 22 _._ 07 8 _._ 61 15 _._ 17 7 _._ 81 5 _._ 13 4 _._ 37 2 _._ 04 3 _._ 79 **1** _._ **45**
Natural
p0 _._ 25q p0 _._ 33q p0 _._ 08q p0 _._ 25q p0 _._ 11q p0 _._ 29q p0 _._ 08q p0 _._ 39q p0 _._ 24q p0 _._ 30q p0 _._ 27q
ImageNet200

19 _._ 86 12 _._ 94 32 _._ 44 13 _._ 35 25 _._ 02 12 _._ 38 5 _._ 41 5 _._ 93 3 _._ 09 5 _._ 00 **2** _._ **68**
Synthetic
p1 _._ 38q p1 _._ 81q p1 _._ 00q p1 _._ 30q p1 _._ 10q p1 _._ 38q p0 _._ 89q p1 _._ 38q p0 _._ 87q p1 _._ 28q p0 _._ 45q


7 _._ 77 6 _._ 50 18 _._ 13 6 _._ 02 8 _._ 13 5 _._ 76 6 _._ 23 3 _._ 88 2 _._ 17 2 _._ 06 **0** _._ **80**
Natural
p0 _._ 27q p0 _._ 33q p0 _._ 23q p0 _._ 34q p0 _._ 27q p0 _._ 37q p0 _._ 41q p0 _._ 53q p0 _._ 62q p0 _._ 54q p0 _._ 44q
ImageNet

13 _._ 39 10 _._ 12 24 _._ 62 8 _._ 51 13 _._ 55 7 _._ 90 6 _._ 32 3 _._ 34 **2** _._ **53** **2** _._ **61** 4 _._ 89
Synthetic
p0 _._ 53q p0 _._ 63q p0 _._ 64q p0 _._ 71q p0 _._ 61q p0 _._ 72q p0 _._ 33q p0 _._ 53q p0 _._ 36q p0 _._ 33q p0 _._ 83q


5 _._ 53 4 _._ 31 33 _._ 53 12 _._ 84 5 _._ 94 4 _._ 45 5 _._ 74 3 _._ 06 **2** _._ **70** 3 _._ 02 **2** _._ **72**
FMoW- WILDS Natural
p0 _._ 33q p0 _._ 63q p0 _._ 13q p12 _._ 06q p0 _._ 36q p0 _._ 77q p0 _._ 55q p0 _._ 36q p0 _._ 54q p0 _._ 35q p0 _._ 44q


5 _._ 80 5 _._ 72 7 _._ 90 4 _._ 84 5 _._ 98 5 _._ 98 6 _._ 03 4 _._ 66 **4** _._ **56** **4** _._ **41** **4** _._ **47**
RxRx1- WILDS Natural
p0 _._ 17q p0 _._ 15q p0 _._ 24q p0 _._ 09q p0 _._ 15q p0 _._ 13q p0 _._ 08q p0 _._ 38q p0 _._ 38q p0 _._ 31q p0 _._ 26q


2 _._ 40 2 _._ 29 8 _._ 01 2 _._ 38 2 _._ 40 2 _._ 28 17 _._ 87 1 _._ 65 **1** _._ **62** **1** _._ **60** **1** _._ **59**
Amazon- WILDS Natural
p0 _._ 08q p0 _._ 09q p0 _._ 53q p0 _._ 17q p0 _._ 09q p0 _._ 09q p0 _._ 18q p0 _._ 06q p0 _._ 05q p0 _._ 14q p0 _._ 15q


12 _._ 64 10 _._ 80 16 _._ 76 11 _._ 03 13 _._ 31 10 _._ 99 16 _._ 65 **7** _._ **14**
CivilCom.- WILDS Natural
p0 _._ 52q p0 _._ 48q p0 _._ 53q p0 _._ 49q p0 _._ 52q p0 _._ 49q p0 _._ 25q p0 _._ 41q


18 _._ 48 15 _._ 99 21 _._ 17 14 _._ 81 20 _._ 19 14 _._ 56 24 _._ 42 5 _._ 02 **2** _._ **40** 3 _._ 14 3 _._ 50
MNIST Natural
p0 _._ 45q p1 _._ 53q p0 _._ 24q p3 _._ 89q p0 _._ 23q p3 _._ 47q p0 _._ 41q p0 _._ 44q p1 _._ 83q p0 _._ 49q p0 _._ 17q


16 _._ 23 11 _._ 14 24 _._ 97 10 _._ 88 19 _._ 08 10 _._ 47 10 _._ 71 5 _._ 39 **3** _._ **88** 4 _._ 58 4 _._ 19
Same
p0 _._ 77q p0 _._ 65q p0 _._ 70q p0 _._ 77q p0 _._ 65q p0 _._ 72q p0 _._ 74q p0 _._ 92q p0 _._ 61q p0 _._ 85q p0 _._ 16q
E NTITY -13

28 _._ 53 22 _._ 02 38 _._ 33 21 _._ 64 32 _._ 43 21 _._ 22 20 _._ 61 13 _._ 58 10 _._ 28 12 _._ 25 **6** _._ **63**
Novel
p0 _._ 82q p0 _._ 68q p0 _._ 75q p0 _._ 86q p0 _._ 69q p0 _._ 80q p0 _._ 60q p1 _._ 15q p1 _._ 34q p1 _._ 21q p0 _._ 93q


18 _._ 59 14 _._ 46 28 _._ 82 14 _._ 30 21 _._ 63 13 _._ 46 12 _._ 92 9 _._ 12 **7** _._ **75** 8 _._ 15 **7** _._ **64**
Same
p0 _._ 51q p0 _._ 52q p0 _._ 43q p0 _._ 71q p0 _._ 37q p0 _._ 59q p0 _._ 14q p0 _._ 62q p0 _._ 72q p0 _._ 68q p0 _._ 88q
E NTITY -30

32 _._ 34 26 _._ 85 44 _._ 02 26 _._ 27 36 _._ 82 25 _._ 42 23 _._ 16 17 _._ 75 14 _._ 30 15 _._ 60 **10** _._ **57**
Novel
p0 _._ 60q p0 _._ 58q p0 _._ 56q p0 _._ 79q p0 _._ 47q p0 _._ 68q p0 _._ 12q p0 _._ 76q p0 _._ 85q p0 _._ 86q p0 _._ 86q


18 _._ 66 17 _._ 17 26 _._ 39 16 _._ 14 19 _._ 86 15 _._ 58 16 _._ 63 10 _._ 87 **10** _._ **24** 10 _._ 07 **10** _._ **26**
Same
N ONLIVING -26 p0 _._ 76q p0 _._ 74q p0 _._ 82q p0 _._ 81q p0 _._ 67q p0 _._ 76q p0 _._ 45q p0 _._ 98q p0 _._ 83q p0 _._ 92q p1 _._ 18q

33 _._ 43 31 _._ 53 41 _._ 66 29 _._ 87 35 _._ 13 29 _._ 31 29 _._ 56 21 _._ 70 20 _._ 12 19 _._ 08 **18** _._ **26**
Novel
p0 _._ 67q p0 _._ 65q p0 _._ 67q p0 _._ 71q p0 _._ 54q p0 _._ 64q p0 _._ 21q p0 _._ 86q p0 _._ 75q p0 _._ 82q p1 _._ 12q


12 _._ 63 11 _._ 05 18 _._ 32 10 _._ 46 14 _._ 43 10 _._ 14 9 _._ 87 4 _._ 57 **3** _._ **95** **3** _._ **81** 4 _._ 21
Same
p1 _._ 25q p1 _._ 20q p1 _._ 01q p1 _._ 12q p1 _._ 11q p1 _._ 16q p0 _._ 61q p0 _._ 71q p0 _._ 48q p0 _._ 22q p0 _._ 53q
L IVING -17

29 _._ 03 26 _._ 96 35 _._ 67 26 _._ 11 31 _._ 73 25 _._ 73 23 _._ 53 16 _._ 15 14 _._ 49 12 _._ 97 **11** _._ **39**
Novel
p1 _._ 44q p1 _._ 38q p1 _._ 09q p1 _._ 27q p1 _._ 19q p1 _._ 35q p0 _._ 52q p1 _._ 36q p1 _._ 46q p1 _._ 52q p1 _._ 72q


Table 3: _Mean Absolute estimation Error (MAE) results for different datasets in our setup grouped by_
_the nature of shift._ ‘Same’ refers to same subpopulation shifts and ‘Novel’ refers novel subpopulation
shifts. We include details about the target sets considered in each shift in Table 2. Post T denotes
use of TS calibration on source. For language datasets, we use DistilBERT-base-uncased, for vision
dataset we report results with DenseNet model with the exception of MNIST where we use FCN.
Across all datasets, we observe that ATC achieves superior performance (lower MAE is better). For
GDE post T and pre T estimates match since TS doesn’t alter the argmax prediction. Results reported
by aggregating MAE numbers over 4 different seeds. Values in parenthesis (i.e., p¨q ) denote standard
deviation values.


28


Published as a conference paper at ICLR 2022


IM AC DOC GDE ATC-MC (Ours) ATC-NE (Ours)
Dataset Shift
Pre T Post T Pre T Post T Pre T Post T Post T Pre T Post T Pre T Post T


7 _._ 14 6 _._ 20 10 _._ 25 7 _._ 06 7 _._ 68 6 _._ 35 5 _._ 74 4 _._ 02 3 _._ 85 3 _._ 76 **3** _._ **38**
Natural
p0 _._ 14q p0 _._ 11q p0 _._ 31q p0 _._ 33q p0 _._ 28q p0 _._ 27q p0 _._ 25q p0 _._ 38q p0 _._ 30q p0 _._ 33q p0 _._ 32q
CIFAR10

12 _._ 62 10 _._ 75 16 _._ 50 11 _._ 91 13 _._ 93 11 _._ 20 7 _._ 97 5 _._ 66 5 _._ 03 4 _._ 87 **3** _._ **63**
Synthetic
p0 _._ 76q p0 _._ 71q p0 _._ 28q p0 _._ 24q p0 _._ 29q p0 _._ 28q p0 _._ 13q p0 _._ 64q p0 _._ 71q p0 _._ 71q p0 _._ 62q


12 _._ 77 12 _._ 34 16 _._ 89 12 _._ 73 11 _._ 18 9 _._ 63 12 _._ 00 5 _._ 61 **5** _._ **55** 5 _._ 65 5 _._ 76
CIFAR100 Synthetic
p0 _._ 43q p0 _._ 68q p0 _._ 20q p2 _._ 59q p0 _._ 35q p1 _._ 25q p0 _._ 48q p0 _._ 51q p0 _._ 55q p0 _._ 35q p0 _._ 27q


12 _._ 63 7 _._ 99 23 _._ 08 7 _._ 22 15 _._ 40 6 _._ 33 5 _._ 00 4 _._ 60 1 _._ 80 4 _._ 06 **1** _._ **38**
Natural
p0 _._ 59q p0 _._ 47q p0 _._ 31q p0 _._ 22q p0 _._ 42q p0 _._ 24q p0 _._ 36q p0 _._ 63q p0 _._ 17q p0 _._ 69q p0 _._ 29q
ImageNet200

20 _._ 17 11 _._ 74 33 _._ 69 9 _._ 51 25 _._ 49 8 _._ 61 4 _._ 19 5 _._ 37 2 _._ 78 4 _._ 53 3 _._ 58
Synthetic
p0 _._ 74q p0 _._ 80q p0 _._ 73q p0 _._ 51q p0 _._ 66q p0 _._ 50q p0 _._ 14q p0 _._ 88q p0 _._ 23q p0 _._ 79q p0 _._ 33q


8 _._ 09 6 _._ 42 21 _._ 66 5 _._ 91 8 _._ 53 5 _._ 21 5 _._ 90 3 _._ 93 1 _._ 89 2 _._ 45 **0** _._ **73**
Natural
p0 _._ 25q p0 _._ 28q p0 _._ 38q p0 _._ 22q p0 _._ 26q p0 _._ 25q p0 _._ 44q p0 _._ 26q p0 _._ 21q p0 _._ 16q p0 _._ 10q
ImageNet

13 _._ 93 9 _._ 90 28 _._ 05 7 _._ 56 13 _._ 82 6 _._ 19 6 _._ 70 3 _._ 33 2 _._ 55 2 _._ 12 5 _._ 06
Synthetic
p0 _._ 14q p0 _._ 23q p0 _._ 39q p0 _._ 13q p0 _._ 31q p0 _._ 07q p0 _._ 52q p0 _._ 25q p0 _._ 25q p0 _._ 31q p0 _._ 27q


5 _._ 15 3 _._ 55 34 _._ 64 5 _._ 03 5 _._ 58 3 _._ 46 5 _._ 08 2 _._ 59 2 _._ 33 2 _._ 52 **2** _._ **22**
FMoW- WILDS Natural
p0 _._ 19q p0 _._ 41q p0 _._ 22q p0 _._ 29q p0 _._ 17q p0 _._ 37q p0 _._ 46q p0 _._ 32q p0 _._ 28q p0 _._ 25q p0 _._ 30q


6 _._ 17 6 _._ 11 21 _._ 05 **5** _._ **21** 6 _._ 54 6 _._ 27 6 _._ 82 5 _._ 30 **5** _._ **20** **5** _._ **19** 5 _._ 63
RxRx1- WILDS Natural
p0 _._ 20q p0 _._ 24q p0 _._ 31q p0 _._ 18q p0 _._ 21q p0 _._ 20q p0 _._ 31q p0 _._ 30q p0 _._ 44q p0 _._ 43q p0 _._ 55q


18 _._ 32 14 _._ 38 27 _._ 79 13 _._ 56 20 _._ 50 13 _._ 22 16 _._ 09 9 _._ 35 7 _._ 50 7 _._ 80 **6** _._ **94**
Same
p0 _._ 29q p0 _._ 53q p1 _._ 18q p0 _._ 58q p0 _._ 47q p0 _._ 58q p0 _._ 84q p0 _._ 79q p0 _._ 65q p0 _._ 62q p0 _._ 71q
E NTITY -13

28 _._ 82 24 _._ 03 38 _._ 97 22 _._ 96 31 _._ 66 22 _._ 61 25 _._ 26 17 _._ 11 13 _._ 96 14 _._ 75 **9** _._ **94**
Novel
p0 _._ 30q p0 _._ 55q p1 _._ 32q p0 _._ 59q p0 _._ 54q p0 _._ 58q p1 _._ 08q p0 _._ 84q p0 _._ 93q p0 _._ 64q p0 _._ 78q


16 _._ 91 14 _._ 61 26 _._ 84 14 _._ 37 18 _._ 60 13 _._ 11 13 _._ 74 8 _._ 54 7 _._ 94 **7** _._ **77** 8 _._ 04
Same
p1 _._ 33q p1 _._ 11q p2 _._ 15q p1 _._ 34q p1 _._ 69q p1 _._ 30q p1 _._ 07q p1 _._ 47q p1 _._ 38q p1 _._ 44q p1 _._ 51q
E NTITY -30

28 _._ 66 25 _._ 83 39 _._ 21 25 _._ 03 30 _._ 95 23 _._ 73 23 _._ 15 15 _._ 57 13 _._ 24 12 _._ 44 **11** _._ **05**
Novel
p1 _._ 16q p0 _._ 88q p2 _._ 03q p1 _._ 11q p1 _._ 64q p1 _._ 11q p0 _._ 51q p1 _._ 44q p1 _._ 15q p1 _._ 26q p1 _._ 13q


17 _._ 43 15 _._ 95 27 _._ 70 15 _._ 40 18 _._ 06 14 _._ 58 16 _._ 99 10 _._ 79 **10** _._ **13** **10** _._ **05** 10 _._ 29
Same
p0 _._ 90q p0 _._ 86q p0 _._ 90q p0 _._ 69q p1 _._ 00q p0 _._ 78q p1 _._ 25q p0 _._ 62q p0 _._ 32q p0 _._ 46q p0 _._ 79q
N ONLIVING -26

29 _._ 51 27 _._ 75 40 _._ 02 26 _._ 77 30 _._ 36 25 _._ 93 27 _._ 70 19 _._ 64 17 _._ 75 16 _._ 90 **15** _._ **69**
Novel
p0 _._ 86q p0 _._ 82q p0 _._ 76q p0 _._ 82q p0 _._ 95q p0 _._ 80q p1 _._ 42q p0 _._ 68q p0 _._ 53q p0 _._ 60q p0 _._ 83q


14 _._ 28 12 _._ 21 23 _._ 46 11 _._ 16 15 _._ 22 10 _._ 78 10 _._ 49 4 _._ 92 **4** _._ **23** **4** _._ **19** 4 _._ 73
Same
p0 _._ 96q p0 _._ 93q p1 _._ 16q p0 _._ 90q p0 _._ 96q p0 _._ 99q p0 _._ 97q p0 _._ 57q p0 _._ 42q p0 _._ 35q p0 _._ 24q
L IVING -17

28 _._ 91 26 _._ 35 38 _._ 62 24 _._ 91 30 _._ 32 24 _._ 52 22 _._ 49 15 _._ 42 13 _._ 02 12 _._ 29 **10** _._ **34**
Novel
p0 _._ 66q p0 _._ 73q p1 _._ 01q p0 _._ 61q p0 _._ 59q p0 _._ 74q p0 _._ 85q p0 _._ 59q p0 _._ 53q p0 _._ 73q p0 _._ 62q


Table 4: _Mean Absolute estimation Error (MAE) results for different datasets in our setup grouped_
_by the nature of shift for ResNet model._ ‘Same’ refers to same subpopulation shifts and ‘Novel’ refers
novel subpopulation shifts. We include details about the target sets considered in each shift in Table 2.
Post T denotes use of TS calibration on source. Across all datasets, we observe that ATC achieves
superior performance (lower MAE is better). For GDE post T and pre T estimates match since TS
doesn’t alter the argmax prediction. Results reported by aggregating MAE numbers over 4 different
seeds. Values in parenthesis (i.e., p¨q) denote standard deviation values.


29


