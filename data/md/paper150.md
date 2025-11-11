# Cyberattack Detection in Mobile Cloud Computing: A Deep Learning Approach

Khoi Khac Nguyen [1], Dinh Thai Hoang [2], Dusit Niyato [2], Ping Wang [2], Diep Nguyen [3], and Eryk Dutkiewicz [3]

1 School of Information and Communication Technology, Hanoi University of Science and Technology, Vietnam
2 School of Computer Science and Engineering, Nanyang Technological University, Singapore
3 School of Computing and Communications, University of Technology Sydney, Australia



_**Abstract**_ **—With the rapid growth of mobile applications and**
**cloud computing, mobile cloud computing has attracted great**
**interest from both academia and industry. However, mobile cloud**
**applications are facing security issues such as data integrity,**
**users’ confidentiality, and service availability. A preventive ap-**
**proach to such problems is to detect and isolate cyber threats**
**before they can cause serious impacts to the mobile cloud**
**computing system. In this paper, we propose a novel framework**
**that leverages a deep learning approach to detect cyberattacks**
**in mobile cloud environment. Through experimental results, we**
**show that our proposed framework not only recognizes diverse**
**cyberattacks, but also achieves a high accuracy (up to 97.11%) in**
**detecting the attacks. Furthermore, we present the comparisons**
**with current machine learning-based approaches to demonstrate**
**the effectiveness of our proposed solution.**


_Keywords-_ Cybersecurity, cyberattack, intrusion detection,
mobile cloud, and deep learning.


I. I NTRODUCTION


Mobile cloud computing (MCC) is an emerging architecture
which has been developed based on the power of cloud
computing to serve mobile devices [1]. MCC allows mobile
applications to be stored and remotely executed on cloud
servers, thereby reducing computing and energy costs for
mobile devices. Furthermore, MCC brings a huge profit to
cloud service providers by optimizing cloud resource usage
through advanced virtualization technologies. Forbes magazine
predicts that worldwide spending on public cloud services
will grow at a 19.4% compound annual growth rate (CAGR)
from nearly $70 billion in 2015 to more than $141 billion
in 2019 [2]. However, the MCC is challenged by cybercrime.
According to UK Government, 74% of small firms in the UK
experienced a cybersecurity breach, and 90% of large firms
were also targeted in 2014 [3].
To counter cyberattacks in MCC, it is crucial to early detect
cyber threats, thereby implementing prompt countermeasures
to prevent the risks. Currently, there are some approaches
proposed to detect and prevent cyberattacks in cloud environment. For example, the authors of [4], [5], and [6]
introduced solutions to detect DoS attacks. Alternatively, one
can also rely on attacks’ patterns and risk assessment [9],
game theory [10], and supervised learning [11] to detect and
counter cyber threats. The common limitation of these methods
is the relatively low accuracy in detecting cyberattacks, and
they are unable to work effectively in real-time cloud systems
with different types of attacks. In this paper, we propose a
framework with an advanced detection mechanism developed



from deep learning technique which allows to detect various
attacks with high accuracy.
Deep learning is a sub-field of machine learning concerned
with algorithms inspired by the structure and functions of neural networks [12]. Over the past few years, deep learning has
been implemented successfully in many areas. For example,
deep learning can be used in automatic translation machines to
improve the reliability, in recommendation systems to suggest
what customers are interested in or want to buy next, and in
image recognition systems to detect objects [12]. In this paper,
we introduce a framework to detect and prevent cyberattacks
through using deep learning technology. The core idea of
deep learning method is using a training dataset to train the
pre-established neural network in offline mode with the aim
to adjust weights of the neural network. Then, the neural
network will be used to detect cyberattacks in the cloud
system in online mode. Through experimental results, we
show that our proposed framework can detect diverse attacks
with very high accuracy. In addition, we have also compared
the performance of our proposed framework with those of
conventional intrusion detection methods to demonstrate the

efficiency of our solution.


II. R ELATED W ORK


There have been a rich literature dealing with DoS attacks in the cloud computing environment. In particular, the
authors in [4] introduced a method taking advantages of
virtual machine status including CPU and network usage to
identify Denial-of-Service (DoS) attacks in cloud data centers.
The authors found that malicious virtual machines exhibit

similar status patterns when a DoS attack is launched, and
thus information entropy can be applied in monitoring the
status of virtual machines to identify attack behaviors. The
authors in [5] adopted a convariance matrix method relying
on investigating the correlation of several features in the IP
header. The authors in [6] developed a classification method to
determine the behavior of packets based on Kappa coefficient.
Besides DoS attacks, other attacks have been also reported
and studied. For example, in cloud computing, since multiple
Virtual Machines (VM) share the same physical machine, this
creates a great opportunity to carry out Cache-based Side
Channel Attack (CSCA). A detection technique using Bloom
Filter (BF) was developed in [7] to address this problem. The
core idea of the BF is to reduce the performance overhead
and a mean calculator to predict the cache behavior most


|Suspicious request|Col2|
|---|---|
|||
|||









Fig. 1. System model for cyberattack detection in mobile cloud computing system.



probably caused by CSCA. Alternatively, in [8], the SQL
injection attack detection method was introduced to prevent
unauthorized accesses. The method first obtains SQL keywords
through the analysis of lexical regulation for SQL statement,
then analyzes the syntax regulation of SQL statement to create
the rule tree and traverses ternary tree to detect the attacks.
The common limitation of existing intrusion detection approaches in cloud environment is that they are unable to simultaneously detect a wide range of attacks with high accuracy.
For example, all aforementioned solutions are able to detect
one type of attacks only. In this work, we develop an intrusion
detection framework which is able to detect diverse attacks in

MCC system with high accuracy.


III. S YSTEM M ODEL

In this section, we first describe the proposed system model
for cyberattack detection along with main functions, and then
explain how the system works. As shown in Fig. 1, when
a request, i.e., a packet, from a mobile user is sent to the
system, it will be passed to the _attack detection module_ . This
module has three main functions, i.e., data collection and preprocessing, attack recognition, and request processing.

_•_ _Data collection and pre-processing function:_ is responsible for collecting data and pre-processing the request
to fit the deep learning model. This function is essential
to enhance the performance of our model, and helps the
gradient descent algorithm used in the training process to
converge much faster.

_•_ _Attack detection (online component) function:_ is used to
classify the incoming requests based on the trained deep
learning model. After the deep learning model has been
trained in an offline mode, it will be used for this function
to detect malicious requests.

_•_ _Request processing:_ Given an incoming request, the attack detection function will mark this request as a normal
or suspicious request. If the request is normal, it will
be served by available cloud resources. Otherwise, the
request will be reported to the _security control module_ .
When a suspicious request is sent to the security control
module, the _request verifying function_ will be activated. In



particular, the request will be verified carefully by comparing
with the current database and/or sending to security service
providers for double-checking. If the request is identified to
be harmless, it will be served as normal. On the other hand,
the request will be treated as a malicious request, and the
_attack defend function_ will be activated to implement prompt
security policies to prevent the spread as well as impacts of
this attack. For example, if a request is identified to be a DoS
attack, the security manager can immediately implement filters
at the gateway to block packets from the same IP address.


IV. D EEP L EARNING M ODEL


In this section, we present the deep learning model for
cyberattack detection and explain how the learning model
detects cyberattacks in the cloud system. As shown in Fig. 2,
there are two phases in the learning model, i.e., feature analysis
and learning process.


_A. Features Analysis and Dimension Reduction_


_1) Features Analysis:_ Feature analysis is the first step in the
deep learning model. The aim of this step is to extract features
and learn from the features. Different types of malicious
packets may have special features which are different from
the normal ones, and thus by extracting and analyzing the
abnormal attributes of packets, we can determine whether a
packet is malicious or not. For example, packet features such as
_source bytes_, _percentage of packets with errors_, and _IP packet_
_entropy_ are important features to detect DoS attacks [13].
_2) Dimension Reduction:_ Data packets contain many attributes with different features. For example, each record in
the KDDcup 1999 dataset [16] and NSL-KDD dataset [17]
consists of 41 features. However, not all the 41 features are

useful for intrusion detection. Some features are irrelevant and

redundant resulting in a long detection process and degrading
the performance. Therefore, selecting features which preserve
the most important information of a dataset is essential to
reduce the computation complexity and increase the accuracy
for the learning process.


DoS U2R



|Col1|Col2|Col3|Col4|
|---|---|---|---|
|Hidden<br>layers<br>Output<br>layer<br>Input<br> layer<br>Pre-learning<br>GRBM<br> Deep learning<br>RBMs<br>Softmax<br>regression|Hidden<br>layers<br>Output<br>layer<br>Input<br> layer<br>Pre-learning<br>GRBM<br> Deep learning<br>RBMs<br>Softmax<br>regression|Hidden<br>layers<br>Output<br>layer<br>Input<br> layer<br>Pre-learning<br>GRBM<br> Deep learning<br>RBMs<br>Softmax<br>regression|Hidden<br>layers<br>Output<br>layer<br>Input<br> layer<br>Pre-learning<br>GRBM<br> Deep learning<br>RBMs<br>Softmax<br>regression|
|Featu<br>analysis<br>dimens<br>reducti<br>Requests|Featu<br>analysis<br>dimens<br>reducti<br>Requests|Featu<br>analysis<br>dimens<br>reducti<br>Requests|Featu<br>analysis<br>dimens<br>reducti<br>Requests|


Fig. 2. Deep learning model.


Principal Component Analysis (PCA) is an effective technique which is often used in machine learning to emphasize
variation and determine strong patterns in a dataset. The core
idea of PCA is to reduce the dimensionality of a dataset
consisting of a large number of interrelated variables, while
retaining as much as possible the variation presented in the
dataset [14]. Thus, in this paper, we adopt the PCA to reduce
dimensions for considered datasets.

Mathematically, the PCA maps a dataset from an _n_  dimensional space to an _r_ -dimensional space where _r ≤_ _n_ to
minimize the residual sum of squares (RSS) of the projection.
This is equivalent to maximize the covariance matrix of the
projected dataset [14]. The dataset in the new domain has two
important properties, i.e., the different dimensions of the data
have no correlation anymore and the dimensions are ordered
according to the importance of their information. We define
**X** as a ( _m × n_ ) matrix with _m_ observations of _n_ different
variables. Then, the covariance matrix **C** is given by:


1
**C** = (1)
_n −_ 1 **[X]** _[⊤]_ **[X]** _[.]_


Since **C** is a symmetric matrix, it can be diagonalized as
follows [15]:


**C** = **VLV** _[⊤]_ _,_ (2)



where **V** is a matrix of eigenvectors and **L** = _diag_ ( _λ_ 1 _, ..., λ_ _p_ )
is a diagonal matrix of eigenvalues in a decreasing order. If
we use singular value decomposition (SVD) to perform PCA,
we will obtain the decomposition as follow [15]:


**X** = **U** Σ **V** _[⊤]_ _,_ (3)


where **U** and **V** are orthonormal matrices meaning that
**U** _[T]_ **U** = **UU** _[T]_ = **I** and **V** _[T]_ **V** = **VV** _[T]_ = **I**, and
Σ = _diag_ ( _s_ 1 _, ..., s_ _n_ ) is a diagonal matrix of singular values
_s_ _i_ . Then, we derive the following results:



1 1
**C** =
_n −_ 1 **[X]** _[⊤]_ **[X]** [ =] _n −_ 1 [(] **[V]** [Σ] **[U]** _[⊤]_ [)(] **[U]** [Σ] **[V]** _[⊤]_ [)] (4)

1

=
_n −_ 1 **[V]** [Σ] [2] **[V]** _[⊤]_ _[.]_



(4) implies that singular vector **V** has principal directions
and singular value _s_ _i_ is related to the eigenvalue _λ_ _i_ of
covariance matrix **C** via _λ_ _i_ = _s_ [2] _i_ _[/]_ [(] _[n][ −]_ [1)][. Thus, we are able]
to define the principal components (PCs) as follows [15]:


**P** = **XV** = **U** Σ **V** _[T]_ **V** = **U** Σ _,_ (5)


where the columns of matrix **P** are the PCs and the matrix

**V** is called the loading matrix which contains the linear
combination coefficients of the variables for each PC. We want
to project the dataset from _n_ -dimensional to _r_ -dimensional
while retaining the important dimensions of the data. In other
words, we have to find the smallest value _r_ such that the
following condition holds:

_r_
~~�~~ � ~~_n_~~ _ji_ ==11 _[λ][λ]_ _[i][j]_ _≥_ _α,_ (6)


where _α_ is the percentage of information which needs to
be reserved after reducing the dimension of input data to _r_ dimensional. We can observe that the PCA will choose the

PCs, i.e., important features, that maximize the variance _α_ .


_B. Learning Process_


The learning process includes three layers, i.e., input layer,
output layer, and some hidden layers, as depicted in Fig. 2.
The refined features will be used as the input data of the input
layer. After the learning process, we can determine whether the
packet is normal or malicious. The learning process includes
three main steps, i.e., pre-learning, deep learning, and softmax
regression steps as shown in Fig. 2.
_1) Pre-learning Process:_ This step uses a Gaussian Binary
Restricted Boltzmann Machine (GRBM) to transform real
values, i.e., input data of the input layer, into binary codes
which will be used in the hidden layers. The GRBM has
_I_ visible units and _J_ hidden units. The number of visible
units (i.e., the number of neurons) is defined as the number of
features after reducing dimension, and the number of hidden
units is pre-defined in advance. The energy function of the
GRBM is defined by [22]:



_J_
� _w_ _ij_ _h_ _j_ _σv_ _ii_ _−_

_j_ =1



_J_
� _b_ _j_ _h_ _j_ _,_ (7)

_j_ =1



_I_
�


_i_ =1



_E_ ( **v** _,_ **h** ) =



_I_
�


_i_ =1



( _v_ _i_ _−_ 2 _σ_ _i_ [2] _a_ _i_ ) [2] _−_


where **v** is visible vector and **h** is hidden vector. _a_ _i_ and _b_ _j_ are
biases corresponding to visible and hidden units, respectively.
_w_ _ij_ is the connecting weight between the visible and hidden
units, and _σ_ _i_ is the standard deviation associated with Gaussian
visible unit _v_ _i_ . Then, the network assigns a probability to every
possible pair of a visible and a hidden vector via the energy
function. The probability is defined as follows:


_e_ _[−][E]_ [(] **[v]** _[,]_ **[h]** [)]
_p_ ( **v** _,_ **h** ) = ~~�~~ **v** _,_ **h** _[e]_ _[−][E]_ [(] **[v]** _[,]_ **[h]** [)] _[.]_ (8)


From (8), we can derive the probability that the network is
assigned to a visible vector **v** as follows:



where parameters _w_ _ij_, _a_ _i_, and _b_ _j_ are defined similarly as in (7).
The conditional probability of a single variable being one
(e.g., _p_ ( _h_ _j_ = 1 _|_ **v** )) can be interpreted as the firing rate of a
neuron with the sigmoid activation function as follows [22]:



_D_
�



� _w_ _ij_ _v_ _i_ + _b_ _j_ ) _,_ (14)


_i_ =1



_p_ ( _h_ _j_ = 1 _|_ **v** ) = _sigm_ (



_F_
�



_p_ ( _v_ _i_ = 1 _|_ **h** ) = _sigm_ (



� _w_ _ij_ _h_ _j_ + _a_ _i_ ) _._ (15)

_j_ =1



� **h** _[e]_ _[−][E]_ [(] **[v]** _[,]_ **[h]** [)]

~~�~~ **v** **h** _[e]_ _[−][E]_ [(] **[v]** _[,]_ **[h]**



�
_p_ ( **v** ) =




_[.]_ (9)
**v** _,_ **h** _[e]_ _[−][E]_ [(] **[v]** _[,]_ **[h]** [)]



From the probability _p_ ( **v** ), we can derive the learning update
rule for performing stochastic steepest descent in the log
probability of the training data as follows:



g _p_ ( **v** ) = 1

_∂w_ _ij_ � _σ_



�



1
_−_ _v_ _i_ _h_ _j_
data � _σ_ _i_



1

_−_
data � _σ_ _i_



�



_,_

model



_∂_ log _p_ ( **v** )



_v_ _i_ _h_ _j_
_σ_ _i_



Similar to the pre-learning step, we can derive the learning
update rule for the weights of the RBM as follows:


∆ _w_ _ij_ = _ϵ_ ( _⟨v_ _i_ _h_ _j_ _⟩_ _data_ _−⟨v_ _i_ _h_ _j_ _⟩_ _model_ ) _,_ (16)


where _ϵ_ is the learning rate.
_3) Softmax Regression Step:_ The output of the last hidden
layer, i.e., **x**, will be used as the input of the softmax regression
(at the output layer) to classify the packet. A packet can be
classified into _M_ = ( _K_ + 1) classes, where _K_ denotes all
types of attacks. Mathematically, the probability that an output
prediction _Y_ is class _i_, is determined by:



1


_σ_

��



� model



�



(10)

_,_



∆ _w_ _ij_ = _ϵ_



_v_ _i_ _h_ _j_
_σ_ _i_



�



1
_−_ _v_ _i_ _h_ _j_
data � _σ_ _i_



1

_−_
data � _σ_ _i_



where _ϵ_ is the learning rate and _⟨·⟩_ is used to denote the
expectation under a distribution specified by the subscript that
follows [22].
Getting an unbiased sample of _⟨v_ _i_ _h_ _j_ _⟩_ _model_ is difficult because there is no connection between hidden units and between

visible units in a GRBM. Therefore, sampling methods can
be applied to address this problem. In particular, we can
start at any random state of the visible units and perform
Gibbs sampling alternately. Each iteration of alternating Gibbs
sampling involves updating all the hidden units parallelly
using (11) followed by updating all visible units parallelly
using (12).



_e_ _[W]_ _[i]_ **[x]** [+] _[b]_ _[i]_
_p_ ( _Y_ = _i|_ **x** _,_ **W** _,_ **b** ) = _softmax_ _i_ ( **Wx** + **b** ) =
~~�~~ _[e]_ _[W]_ _[j]_ **[x]**




_[,]_
_j_ _[e]_ _[W]_ _[j]_ **[x]** [+] _[b]_ _[j]_



_p_ ( _h_ _j_ = 1 _|_ **v** ) = _sigm_ � _b_ _j_ + � _w_ _ij_ _σv_ _ii_

_i_



_,_ (11)
�



(17)
where **W** is a weight matrix between the last hidden layer
and the output layer, and **b** is a bias vector. Then, the model’s
prediction _y_ pd is the class whose probability is maximal,
specifically:


_y_ pd = arg max _i_ � _p_ ( _Y_ = _i|_ **x** _,_ **W** _,_ **b** )� _, ∀i ∈{_ 1 _,_ 2 _, . . ., M_ _}._
(18)


_C. Offline Deep Training and Online Cyberattack Detection_


The deep training consists of two phases, i.e., pre-training
phase and fine-tuning phase.


_•_ _Pre-training_ : This phase requires only unlabeled data
which is cheap and easy to collect from the Internet for
training. In [18], the authors introduced an efficient way
to learn a complicated model by using a set of simple
sub-models which are learned sequentially. The greedy
layer-wise learning algorithm allows each sub-model in
the sequence to have a different representation of the data.
The sub-model performs a non-linear transformation on
its input vectors to produce output vectors that will be
used as inputs of the next sub-model in the sequence.
The principle of greedy layer-wise unsupervised training
for each layer can be applied with RBMs as the building
blocks for each layer [18], [19], [20]. Our training process
is executed through Gibbs sampling using CD as the
approximation to the gradient [21].

_•_ _Fine-tuning_ : We use the available set of labeled data for
fine-tuning. After pre-training phase, we have a sensible
set of weights for one layer at a time. Thus, bottom-up
back-propagation can be used to fine-tune the model for
better discrimination.



_p_ ( _v_ _i_ _|_ **h** ) = _N_ ( _v_ _i_ _|a_ _i_ + � _h_ _j_ _w_ _ij_ _, σ_ _i_ [2] [)] _[,]_ (12)

_j_


where _sigm_ ( _x_ ) = 1 _/_ (1 + _exp_ ( _−x_ )) is the sigmoid function
and _N_ ( _·|µ, σ_ [2] ) denotes a Gaussian probability density function
with mean _µ_ and standard deviation _σ_ .
_2) Deep Learning Step:_ This step includes a series of
learning processes which are performed in sequence to adjust weights of the neural network. Each learning process is
performed between two successive layers in the hidden layers
through a Restricted Boltzmann Machine (RBM). The RBM
is a particular type of Markov random field. It has a twolayer architecture in which the visible binary stochastic units
**v** _∈{_ 0 _,_ 1 _}_ _[D]_ are connected to the hidden binary stochastic units
**h** _∈{_ 0 _,_ 1 _}_ _[F]_ . Here, _D_ and _F_ are the numbers of visible and
hidden units, respectively. Then, the energy of state _{_ **v** _,_ **h** _}_ can
be calculated by [22]:



_F_
� _b_ _j_ _h_ _j_ (13)

_j_ =1



_D_
� _a_ _i_ _v_ _i_ _−_


_i_ =1



_E_ ( **v** _,_ **h** ) = _−_



_D_
�


_i_ =1



_F_
� _w_ _ij_ _v_ _i_ _h_ _j_ _−_

_j_ =1



_F_
�


( _a_ ) ( _b_ ) ( _c_ )


Fig. 3. Visualizations of three datasets: (a) NSL-KDD, (b) UNSW-NB15, and (c) KDDcup 1999, by using PCA with 3 most important features. Grey circles
represent normal packets, while circles with other colors than grey express the different types of attacks.



After the offline deep training is completed, we will obtain a
deep learning model with trained weights. This learning model
will be then implemented on the attack detection module to
detect malicious packets in an online fashion.


V. D ATASET C OLLECTION AND E VALUATION M ETHODS


In this section, we give a brief overview of common cyberattacks in MCC, three real datasets are used in our experiments.
We then present the methods to evaluate the experimental
results.


_A. Dataset Collection_


To verify the accuracy of the deep-learning cyber attack
detection, we use three empirical public datasets.

_1) KDDcup 1999 Dataset:_ The KDDcup 1999 dataset [16]
is widely used as a benchmark for the intrusion detection
network model. Each record in the dataset contains 41 features

and is labeled as either normal or a specific type of attack.
The training dataset contains 22 types of attacks, while testing
dataset contains additional 17 types.

_2) NSL-KDD Dataset:_ The NSL-KDD Dataset was presented in [17] to solve some inherent problems of the KDDCup
1999 dataset such as the huge number of redundant records
both in the training and testing dataset. Each traffic sample
has 41 features. Attacks in the dataset are categorized into four
categories: DoS, R2L, U2R, and Probe attacks. The training
dataset includes 24 attack types, while the testing dataset
contains 38 attack types.

_3) UNSW-NB15 Dataset:_ This dataset has nine families of
attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits,
Generic, Reconnaissance, Shellcode and Worms. _Argus_ and
_Bro-IDS_ network monitoring tools were used and 12 algorithms were developed to generate totally 49 features. The
number of records in the training dataset is 175,340 records
and that in testing dataset is 82,331 records from different
types of attacks.



_B. Evaluation Methods_


In this study, we use accuracy, precision, and recall which
are typical parameters used in machine learning (deeplearning.net) as performance metrics to evaluate the deep-learning
cyberattack detection model.


_•_ _Accuracy (ACC)_ indicates the ratio of correct detection
over total traffic trace: _T P_ + _T NT P_ ++ _T NF P_ + _F N_ [where] _[ TP]_ [,] _[ TN]_ [,]
_FP_, and _FN_ stand for “true positive”, “true negative”,
“false positive”, and “false negative”, respectively. Thus,
the average prediction accuracy of _M_ supported classes
is defined by:




_•_ _Precision (PPV)_ shows how many attacks predicted are
actual attacks. _PPV_ is defined as the ratio of the number
of _TP_ records over the number of _TP_ and _FP_ records.


_TP_
_PPV_ = (20)
_TP_ + _FP_ _[.]_


_•_ _Recall (TPR)_ shows the percentage of attacks that are
correctly predicted versus all attacks happened. _TPR_ is
defined as the ratio of number of _TP_ records divided by
the number of _TP_ and _FN_ records.


_TP_
_TPR_ = (21)
_TP_ + _FN_ _[.]_


VI. E XPERIMENTAL R ESULTS


_A. Visualizations of Datasets by PCA_


Fig. 3 illustrates the visualization of three datasets using
PCA with 3 most important features. After using PCA, the
normal and malicious packets can be detected effectively with
high accuracy. In particular, in Fig. 3, normal packets are
grouped together and separated from malicious packets. Thus,
reducing dimensions in a high dimension dataset not only
reduces the computational complexity, but also diminishes
significant amount of noise in the dataset, thereby increasing
the accuracy in predicting malicious packets.



_ACC_ = [1]

_M_



_M_
�


_i_



_TP_ _i_ + _TN_ _i_
_._ (19)
_TP_ _i_ + _TN_ _i_ + _FP_ _i_ + _FN_ _i_


TABLE I

T HE COMPARISON BETWEEN OUR PROPOSE MODEL WITH OTHER MACHINE LEARNING ALGORITHMS ON THREE DATASETS .


|Col1|NSL-KDD|Col3|Col4|UNSW-NB15|Col6|Col7|KDDcup 1999|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||ACC|PPV|TPR|ACC|PPV|TPR|ACC|PPV|TPR|
|Decision Tree|87.91|63.62|68.50|93.78|76.42|68.92|97.01|94.14|92.52|
|K-means|82.78|84.96|56.95|87.05|74.01|35.23|86.19|89.16|65.47|
|K Neighbours Classiﬁer|88.56|77.19|71.39|94.31|77.42|71.52|96.85|94.12|92.13|
|Logistic Regression|89.52|62.04|73.79|92.52|71.05|62.61|96.2|86.29|90.69|
|Multilayer Perceptron (MLP)|87.91|63.62|68.5|90.16|76.72|75.39|96.77|90.87|91.91|
|Gaussian Naive Bayes|88.33|73.98|41.67|88.34|73.98|41.67|89.29|83.91|73.22|
|Multinomial Naive Bayes|83.96|65.52|59.90|90.97|55.40|54.86|89.66|83.65|74.16|
|Bernoulli Naive Bayes|74.60|87.47|36.49|91.31|55.07|56.52|90.94|89.54|77.35|
|Random Forest Classiﬁer|88.39|71.21|70.99|94.44|80.29|72.21|97.02|94.42|92.56|
|Support Vector Machine (SVM)|88.32|64.70|70.80|93.38|76.91|66.90|96.74|91.59|91.86|
|**Our Proposed Deep Learning Approach**|**90.99**|**81.95**|**77.48**|**95.84**|**83.40**|**79.19**|**97.11**|**94.43**|**92.77**|



_B. Performance Evaluation_


Table I compares the performance of the deep learning
approach with those of other machine learning algorithms,
some of which include K-means, K-neighbors classifier, and
the random forest classifier. We observe that the proposed
deep learning approach always achieves the best performance
in the terms of accuracy, precision, and recall (as defined in
Section V-B), for the same datasets. In particular, the accuracy
of the deep learning approach is 90 _._ 99%, 95 _._ 84%, and 97 _._ 11%,
for NSL-KDD, UNSW-NB15, and KDDcup 1999 datasets,
respectively. Furthermore, both precision and recall parameters
achieved from the proposed approach are also much higher
than those of other machine learning algorithms.
In summary, there are two important observations from our
proposed deep learning model.


_•_ _Stability:_ Deep learning processes allow to achieve high
accuracy (up to 95.84%) under different settings of layers
and the number of neurons per layer.

_•_ _Robustness and flexibility:_ Our deep learning model can
be used effectively to detect a variety of attacks on
different datasets with very high accuracy.


VII. S UMMARY


In this paper, we have introduced a deep learning approach
to detect cyber threats in the mobile cloud environment.
Through experimental results, we have demonstrated that
our proposed learning model can achieve high accuracy in
detecting cyber attacks, and outperform other existing machine
learning methods. In addition, we have shown the stability, efficiency, flexibility, and robustness of our deep learning model
which can be applied to many mobile cloud applications.
For the future work, we will implement our proposed deep
learning model on the real devices and evaluate the accuracy
of the model on the real time basis. Furthermore, the energy
consumption and detection time of the deep learning model
will be evaluated and compared with other methods.


R EFERENCES


[1] D. T. Hoang, C. Lee, D. Niyato, and P. Wang, “A survey of mobile
cloud computing: Architecture, applications, and approaches,” _Wireless_
_Communications and Mobile Computing_, vol. 13, no. 18, pp.1587-1611,
Dec. 2013.




[2] Louis Columbus, _Roundup of cloud computing forecasts and market_
_estimates 2016_, Forbes magazine.

[3] 2015 information security breaches survey, Technical Report, PWC.

[4] J. Cao, B. Yu, F. Dong, X. Zhu, and S. Xu, “Entropy-based denial-ofservice attack detection in cloud data center,” _Concurrency and Compu-_
_tation: Practice and Experience_, vol. 27, no. 18, pp. 5623-5639, Dec.
2015.

[5] M. N. Ismail, A. Aborujilah, S. Musa, and A. Shahzad, “Detecting
flooding based DoS attack in cloud computing environment using covariance matrix approach,” in _IEEE International Conference on Ubiquitous_
_Information Management and Communication_, Kota Kinabalu, Malaysia,
Jan. 2013.

[6] A. Sahi, D. Lai, Y. Li, and M. Diykh, “An Efficient DDoS TCP Flood
Attack Detection and Prevention System in a Cloud Environment,” _IEEE_
_Access_, vol. 5, pp. 6036-6048, Apr. 2017.

[7] M. Chouhan and H. Hasbullah, “Adaptive detection technique for cachebased side channel attack using Bloom Filter for secure cloud,” in _IEEE_
_International Conference on Computer and Information Sciences_, pp. 293297, Aug. 2016.

[8] K. Wang and Y. Hou, “Detection method of SQL injection attack in cloud
computing environment,” in _IEEE Advanced Information Management,_
_Communicates, Electronic and Automation Control Conference_, pp. 487493, Oct 2016.

[9] B. C. Youssef, M. Nada, B. Elmehdi, and R. Boubker, “Intrusion detection
in cloud computing based attacks patterns and risk assessment,” in
_International Conference on Systems of Collaboration_, pp. 1-4, Nov. 2016.

[10] A. Nezarat, “A game theoretic method for VM-to-hypervisor attacks
detection in cloud environment,” in _Proceedings of the 17th IEEE/ACM_
_International Symposium on Cluster, Cloud and Grid Computing_, pp.
1127-1132, May 2017.

[11] G. Nenvani and H. Gupta, “A survey on attack detection on cloud using
supervised learning techniques,” in _IEEE Symposium on Colossal Data_
_Analysis and Networking_, pp. 1-5, Mar. 2016.

[12] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” _Nature_, vol. 521,
no. 7553, pp. 436-444, May 2015.

[13] K. Kurihara and K. Katagishi, “A Simple Detection Method for DoS
Attacks based on IP Packets Entropy values,” _IEEE Asia Joint Conference_
_on Information Security_, Wuhan, China, Sept. 2014.

[14] I. T. Jolliffe, _Principal Component Analysis and Factor Analysis_, Principal component analysis. Springer New York, 1986.

[15] J. Shlens, “A tutorial on principal component analysis,” _arXiv preprint_
_[arXiv:1404.1100](http://arxiv.org/abs/1404.1100)_, 2014.

[[16] http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

[17] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 data set,” _IEEE Symposium on Computational_
_Intelligence for Security and Defense Applications_, pp. 1-6, 2009.

[18] G. E. Hinton, S. Osindero, and Y-W. Teh, “A fast learning algorithm
for deep belief nets,” _Neural computation_, vol. 18, no.7, pp. 1527-1554,
2006.

[19] G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality
of data with neural networks,” _Science_, vol. 313, no. 5786, pp. 504-507,
2006.

[20] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle, “Greedy layerwise training of deep networks,” _Advances in neural information process-_
_ing systems_, pp. 153-160, 2007.

[21] G. E. Hinton, “Training products of experts by minimizing contrastive
divergence,” _Neural computation_, vol.14, no. 8, pp. 1771-1800, 2002.


[22] G. E. Hinton, “A practical guide to training restricted Boltzmann
machines,” _Momentum 9_, no. 1, pp. 926, 2010.

[23] N. Mowla, I. Doh, and K. Chae, “Evolving neural network intrusion
detection system for MCPS,” in _IEEE International Conference on_
_Advanced Communication Technology_, pp. 183-187, Feb. 2017.


