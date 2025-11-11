## **A Survey of Privacy Attacks in Machine Learning**

MARIA RIGAKI, Czech Technical University in Prague
SEBASTIAN GARCIA, Czech Technical University in Prague


As machine learning becomes more widely used, the need to study its implications in security and privacy
becomes more urgent. Although the body of work in privacy has been steadily growing over the past few years,
research on the privacy aspects of machine learning has received less focus than the security aspects. Our
contribution in this research is an analysis of more than 40 papers related to privacy attacks against machine
learning that have been published during the past seven years. We propose an attack taxonomy, together with
a threat model that allows the categorization of different attacks based on the adversarial knowledge, and the
assets under attack. An initial exploration of the causes of privacy leaks is presented, as well as a detailed
analysis of the different attacks. Finally, we present an overview of the most commonly proposed defenses
and a discussion of the open problems and future directions identified during our analysis.


CCS Concepts: • **Computing methodologies** → **Machine learning** ; • **Security and privacy** ;


Additional Key Words and Phrases: privacy, machine learning, membership inference, property inference,
model extraction, reconstruction, model inversion


**1** **INTRODUCTION**


Fueled by large amounts of available data and hardware advances, machine learning has experienced
tremendous growth in academic research and real world applications. At the same time, the impact
on the security, privacy, and fairness of machine learning is receiving increasing attention. In terms
of privacy, our personal data are being harvested by almost every online service and are used to
train models that power machine learning applications. However, it is not well known if and how
these models reveal information about the data used for their training. If a model is trained using
sensitive data such as location, health records, or identity information, then an attack that allows
an adversary to extract this information from the model is highly undesirable. At the same time, if
private data has been used without its owners’ consent, the same type of attack could be used to
determine the unauthorized use of data and thus work in favor of the user’s privacy.
Apart from the increasing interest on the attacks themselves, there is a growing interest in
uncovering what causes privacy leaks and under which conditions a model is susceptible to
different types of privacy-related attacks. There are multiple reasons why models leak information.
Some of them are structural and have to do with the way models are constructed, while others are
due to factors such as poor generalization or memorization of sensitive data samples. Training for
adversarial robustness can also be a factor that affects the degree of information leakage.
The focus of this survey is the privacy and confidentiality attacks on machine learning algorithms.
That is, attacks that try to extract information about the training data or to extract the model itself.
Some existing surveys [ 8, 89 ] provide partial coverage of privacy attacks and there are a few other
peer-reviewed works on the topic [ 2, 47 ]. However, these papers are either too high level or too
specialized in a narrow subset of attacks.
The security of machine learning and the impact of adversarial attacks on the performance of
the models have been widely studied in the community, with several surveys highlighting the
major advances in the area [ 8, 65, 71, 90, 117 ]. Based on the taxonomy proposed in [ 8 ], there are
three types of attacks on machine learning systems: i) attacks against integrity, e.g., evasion and
poisoning backdoor attacks that cause misclassification of specific samples, ii) attacks against a


Authors’ addresses: Maria Rigaki, maria.rigaki@fel.cvut.cz, Czech Technical University in Prague, Karlovo náměstí 13,
Prague, Czech Republic, 120 00; Sebastian Garcia, sebastian.garcia@agents.fel.cvut.cz, Czech Technical University in Prague,
Karlovo náměstí 13, Prague, Czech Republic, 120 00.


2 Rigaki and Garcia


system’s availability, such as poisoning attacks that try to maximize the misclassification error
and iii) attacks against privacy and confidentiality, i.e., attacks that try to infer information about
user data and models. While all attacks on machine learning are adversarial in nature, the term
"adversarial attacks" is commonly used to refer to security-related attacks and more specifically to
adversarial samples. In this survey, we only focus on privacy and confidentiality attacks.
An attack that extracts information about the model’s structure and parameters is, strictly
speaking, an attack against model confidentiality. The decision to include model extraction attacks
was made because in the existing literature, attacks on model confidentiality are usually grouped
together with privacy attacks [ 8, 90 ]. Another important reason is that stealing model functionality
may be considered a privacy breach as well. Veale et al. [ 112 ] made the argument that privacy
attacks such as membership inference (Section 4.1) increase the risk of machine learning models
being classified as personal data under European Union’s General Data Protection Regulation
(GDPR) law because they can render a person identifiable. Although models are currently not
covered by the GDPR, it may happen that they will be considered as personal data, and then attacks
against them may fall under the same scope as attacks against personal data. This may be further
complicated by the fact that model extraction attacks can be used as a stepping stone for other
attacks.

This paper is, as far as we know, the first _comprehensive_ survey of privacy-related attacks
against machine learning. It reviews and systematically analyzes over 40 research papers. The
papers have been published in top tier conferences and journals in the areas of security, privacy,
and machine learning during 2014-2020. An initial set of papers was selected in Google Scholar
using keyword searches related to "privacy", "machine learning", and the names of the attacks
themselves ("membership inference", "model inversion", "property inference", model stealing",
"model extraction", etc.). After the initial set of papers was selected, more papers were added by
backward search based on their references as well as by forward search based on the papers that
cited them.

The main contributions of this paper are:


  - The first comprehensive study of attacks on privacy and confidentiality of machine learning
systems.

   - A unifying taxonomy of attacks against machine learning privacy.

   - A discussion on the probable causes of privacy leaks in machine learning systems.

   - An in-depth presentation of the implementation of the attacks.

  - An overview of the different defensive measures tested to protect against the different
attacks.


**1.1** **Organization of the Paper**

The rest of the paper is organized as follows: Section 2 introduces some basic concepts related to
machine learning that are relevant to the implementation of the attacks which are presented in
Section 6. The threat model is presented in Section 3 and the taxonomy of the attacks and their
definition are the focus of Section 4. In Section 5 we present the causes of machine learning leaks
that are known or have been investigated so far. An overview of the proposed defences per attack
type is the focus of Section 7. Finally, Section 8 contains a discussion on the current and future
research directions and Section 9 offers concluding remarks.


**2** **MACHINE LEARNING**

Machine learning (ML) is a field that studies the problem of learning from data without being
explicitly programmed. The purpose of this section is to provide a non-exhaustive overview of


A Survey of Privacy Attacks in Machine Learning 3


machine learning as it pertains to this survey and to facilitate the discussion in the subsequent
chapters. We briefly introduce a high level view of different machine learning paradigms and
categorizations as well as machine learning architectures. Finally, we present a brief discussion
on model training and inference. For the interested reader, there are several textbooks such as

[9, 29, 78, 97] that provide a thorough coverage of the topic.


**2.1** **Types of Learning**

At a very high level, ML is traditionally split into three major areas: _supervised_, _unsupervised_ and
_reinforcement_ learning. Each of these areas has its own subdivisions. Over the years, new categories
have emerged to capture types of learning that are not easily fit under these three areas such as
_semi-supervised_ and _self-supervised_ learning, or other ways to categorize models such as _generative_
and _discriminative_ ones.


_2.1.1_ _Supervised Learning._ In a supervised learning setting, a model _𝑓_ with parameters _𝜃_ is a
mapping function between inputs x and outputs y = _𝑓_ ( x; _𝜃_ ), where x is a vector of attributes or
features with dimensionality _𝑛_ . The output or label y can assume different dimensions depending on
the learning task. A training set D used for training the model is a set of data points D = {( x _𝑖_ _,_ y _𝑖_ )} _[𝑚]_ _𝑖_ =1 [,]
where _𝑚_ is the number of input-output pairs. The most common supervised learning tasks are
_classification_ and _regression_ . Examples of supervised learning algorithms include linear regression,
logistic regression, decision trees, support vector machines, and many more. The vast majority of
the attack papers thus far are focused in supervised learning using deep neural networks.


_2.1.2_ _Unsupervised Learning._ In unsupervised learning, there are no labels y. The training set D
consists only of the inputs x _𝑖_ . Unsupervised algorithms aim to find structure or patterns in the
data without having access to labels. Usual tasks in unsupervised learning are _clustering feature_
_learning_, _anomaly detection_ and _dimensionality reduction_ . In the context of this survey, attacks on
unsupervised learning appear mostly as attacks on language models.


_2.1.3_ _Reinforcement Learning._ Reinforcement learning concerns itself with agents that make
observations of the environment and use these to take actions with the goal of maximizing a reward
signal. In the most general formulation, the set of actions is not predefined and the rewards are
not necessarily immediate but can occur after a sequence of actions [ 108 ]. To our knowledge, no
privacy-related attacks against reinforcement learning have been reported, but it has been used to
launch other privacy-related attacks [87].


_2.1.4_ _Semi-supervised Learning._ In many real-world settings, the amount of labeled data can
be significantly smaller than that of unlabeled ones, and it might be too costly to obtain highquality labels. Semi-supervised learning algorithms aim to use unlabeled data to learn higher level
representations and then use the labeled examples to guide the downstream learning task. An
example of semi-supervised learning would be to use an unsupervised learning technique such
as clustering on unlabeled data and then use a classifier to separate representative training data
from each cluster. Other notable examples are generative models such as Generative Adversarial
Networks (GANs) [30].


_2.1.5_ _Generative and Discriminative Learning._ Another categorization of learning algorithms is that
of _discriminative_ vs _generative_ algorithms. Discriminative classifiers try to model the conditional
probability _𝑝_ ( _𝑦_ | x ), i.e., they try to learn the decision boundaries that separate the different classes
directly based on the input data x. Examples of such algorithms are logistic regression and neural
networks. Generative classifiers try to capture the joint distribution _𝑝_ ( x _,𝑦_ ) . An example of such a
classifier is Naive Bayes. Generative models that do not require labels, but they try to model _𝑝_ ( x ),


4 Rigaki and Garcia


explicitly or implicitly. Notable examples are language models that predict the next word(s) given
some input text or GANs and Variational Autoencoders (VAEs) [ 57 ] that are able to generate data
samples that match the properties of the training data.


**2.2** **Learning Architectures**


From a system architecture point of view, we view the learning process as either a centralized or a
distributed one. The main criterion behind this categorization is whether the data and the model
are collocated or not.


_2.2.1_ _Centralized Learning._ In a centralized learning setting, the data and the model are collocated.
There can be one or multiple data producers or owners, but all data are gathered in one central
place and used for the training of the model. The location of the data can be in a single or even
multiple machines in the same data center. While using parallelism in the form of multiple GPUs
and CPUs could be considered a distributed learning mode, it is not for us since we use the model
and data collocation as the main criterion for the distinction between centralized and distributed

learning. The centralized learning architecture includes the Machine Learning as a Service (MLaaS)
setup, where the data owner uploads their data to a cloud-based service that is tasked with creating
the best possible model.


_2.2.2_ _Distributed Learning._ The requirements that drive the need for distributed learning architectures are the handling and processing of large amounts of data, the need for computing and
memory capacity, and even privacy concerns. From the existing variants of distributed learning,
we present those that are relevant from a privacy perspective, namely _collaborative_ or _federated_
_learning_ (FL), _fully decentralized_ or _peer-to-peer_ (P2P) learning and _split learning_ .
Collaborative or federated learning is a form of decentralized training where the goal is to learn
one global model from data stored in multiple remote devices or locations [ 61 ]. The main idea is
that the data do not leave the remote devices. Data are processed locally and then used to update
the local models. Intermediate model updates are sent to the central server that aggregates them
and creates a global model. The central server then sends the global model back to all participant
devices.

In fully decentralized learning or Peer-to-Peer (P2P) learning, there is no central orchestration
server. Instead, the devices communicate in a P2P fashion and exchange their updates directly with
other devices. This setup may be interesting from a privacy perspective, since it alleviates the need
to trust a central server. However, attacks on P2P systems are relevant in such settings and need to
be taken into account. Up to now, there were no privacy-based attacks reported on such systems;
although they may become relevant in the future. Moreover, depending on the type of information
shared between the peers, several of the attacks on collaborative learning may be applicable.
In split learning, the trained model is split into two or more parts. The edge devices keep the
initial layers of the deep learning model and the centralized server keeps the final layers [ 34, 54 ].
The reason for the split is mainly to lower communication costs by sending intermediate model
outputs instead of the input data. This setup is also relevant in situations where remote or edge
devices have limited resources and are connected to a central cloud server. This scenario is common

for Internet of Things (IoT) devices.


**2.3** **Training and Inference**

Training of supervised ML models usually follows the Empirical Risk Minimization (ERM) approach [ 111 ], where the objective is to find the parameters _𝜃_ [∗] that minimize the _risk_ or _objective_


A Survey of Privacy Attacks in Machine Learning 5


_function_, which is calculated as an average over the training dataset:



J (D; _𝜃_ ) = [1]

_𝑚_



_𝑚_
∑︁ _𝑙_ ( _𝑓_ ( _𝑥_ _𝑖_ ; _𝜃_ ) _,𝑦_ _𝑖_ ) (1)

_𝑖_ =1



where _𝑙_ (·) is a loss function, e.g. cross entropy loss, and _𝑚_ is the number of data points in the
dataset D.

The idea behind ERM is that the training dataset is a subset drawn from the unknown true data
distribution for the learning task. Since we have no knowledge of the true data distribution, we
cannot minimize the true objective function, but instead we can minimize the estimated objective
over the data samples that we have. In some cases, a regularization term is added to the objective
function to reduce overfitting and stabilize the training process.


_2.3.1_ _Training in Centralized Settings._ The training process usually involves an iterative optimization algorithm such as gradient descent [ 12 ], which aims to minimize the objective function by
following the path induced by its gradients. When the dataset is large, as is often the case with deep
neural networks, taking one gradient step becomes too costly. In that case, variants of gradient
descent which involve steps taken over smaller batches of data are preferred. One such optimization
method is called Stochastic Gradient Descent (SGD) [93] defined by:


_𝜃_ _𝑡_ +1 = _𝜃_ _𝑡_ − _𝜂_ **g** (2)



**g** = _𝑚_ [1] [′] [∇] _[𝜃]_



_𝑚_ [′]
∑︁ _𝑙_ ( _𝑓_ (x _𝑖_ ; _𝜃_ ) _,_ y _𝑖_ ) (3)

_𝑖_ =1



where _𝜂_ is the learning rate and g is the gradient of the loss function with respect to parameters
_𝜃_ . In the original formulation of SGD the gradient g is calculated over a single data point from
D, chosen randomly, hence the name stochastic. In practice, it is common to use mini-batches of
size _𝑚_ [′] where _𝑚_ [′] _< 𝑚_, instead of a single data point to calculate the loss gradient at each step
(Equation 3). Mini-batches lower the variance of the stochastic gradient estimate, but the size _𝑚_ [′]

is a tunable parameter that can affect the performance of the algorithm. While SGD is still quite
popular, several improvements have been proposed to try to speed up convergence by adding
momentum [ 91 ], by using adaptive learning rates as, for example, in the RMSprop algorithm [ 40 ],
or by combining both improvements as in the Adam algorithm [56].


_2.3.2_ _Training in Distributed Settings._ The most popular learning algorithm for federated learning
is federated averaging [ 73 ], where each remote device calculates one step of gradient descent from
the locally stored data and then shares the updated model weights with the parameter server. The
parameter server averages the weights of all remote participants and updates the global model
which is subsequently shared again with the remote devices. It can be defined by:



_𝜃_ _𝑡_ +1 = _𝐾_ [1]



_𝐾_
∑︁ _𝜃_ _𝑡_ [(] _[𝑘]_ [)] (4)

_𝑘_ =1



where K is the number of remote participants and the parameters _𝜃_ _𝑡_ [(] _[𝑘]_ [)] of participant _𝑘_ have been
calculated locally based on Equations 2 and 3.
Another approach that comes from the area of distributed computing is downpour (or synchronized) SGD [ 19 ], which proposes to share the loss gradients of the distributed devices with the
parameter server that aggregates them and then performs one step of gradient descent. It can be
defined by:


6 Rigaki and Garcia



_𝜃_ _𝑡_ +1 = _𝜃_ _𝑡_ − _𝜂_



_𝐾_
∑︁

_𝑘_ =1



_𝑚_ [(] _[𝑘]_ [)]

_𝑀_ [g] _𝑡_ [(] _[𝑘]_ [)] (5)



where g _𝑡_ [(] _[𝑘]_ [)] is the gradient computed by participant _𝑘_ based on Equation 3 using their local data,
_𝑚_ [(] _[𝑘]_ [)] is the number of data points in the remote participant and _𝑀_ is the total number of data points
in the training data. After the calculation of Equation 5, the parameter server sends the updated
model parameters _𝜃_ _𝑡_ +1 to the remote participants.


_2.3.3_ _Inference._ Once the models are trained, they can be used to make inferences or predictions
over previously unseen data. At this stage, the assumption is that the model parameters are fixed,
although the models are usually monitored, evaluated, and retrained if necessary. The majority of
the attacks in this survey are attacks during the inference phase of the model lifecycle except for
the attacks on collaborative learning which are usually performed during training.


**3** **THREAT MODEL**

To understand and defend against attacks in machine learning from a privacy perspective, it is
useful to have a general model of the environment, the different actors, and the assets to protect.
From a threat model perspective, the assets that are sensitive and are potentially under attack are
the training dataset D, the model itself, its parameters _𝜃_, its hyper-parameters, and its architecture.
The actors identified in this threat model are:


(1) The **data owners**, whose data may be sensitive.
(2) The **model owners**, which may or may not own the data and may or may not want to
share information about their models.

(3) The **model consumers**, that use the services that the model owner exposes, usually via
some sort of programming or user interface.
(4) The **adversaries**, that may also have access to the model’s interfaces as a normal consumer
does. If the model owner allows, they may have access to the model itself.

Figure 1 depicts the assets and the identified actors under the threat model, as well as the
information flow and possible actions. This threat model is a logical model and it does not preclude
the possibility that some of these assets may be collocated or spread in multiple locations.
Distributed modes of learning, such as federated or collaborative learning, introduce different
spatial models of adversaries. In a federated learning setting, the adversary can be collocated with
the global model, but it can also be a local attacker. Figure 2 shows the threat model in a collaborative
learning setting. The presence of multiple actors allows also the possibility of _colluding_ adversaries
that join forces.
The different attack surfaces against machine learning models can be modelled in terms of
**adversarial knowledge** . The range of knowledge varies from limited, e.g., having access to a
machine learning API, to having knowledge of the full model parameters and training settings.
In between these two extremes, there is a range of possibilities such as partial knowledge of the
model architecture, its hyper-parameters, or training setup. The knowledge of the adversary can
also be considered from a dataset point of view. In the majority of the papers reviewed, the authors
assume that the adversaries have no knowledge of the training data samples, but they may have
some knowledge of the underlying data distribution.
From a taxonomy point of view, attacks where the adversary has no knowledge of the model
parameters, architecture, or training data are called **black-box** attacks. An example of a black-box
system is Machine Learning as a Service (MLaaS) where the users usually provide some input and
receive either a prediction vector or a class label from a pre-trained model hosted in the cloud.


A Survey of Privacy Attacks in Machine Learning 7


Fig. 1. Threat Model of privacy and confidentiality attacks against machine learning systems. The human
figure represents actors and the symbols represent the assets. Dashed lines represent data and information
flow, while full lines represent possible actions. In red are the actions of the adversaries, available under the
threat model.


Most black-box papers assume the existence of a prediction vector. In a similar fashion, **white-box**
attacks are those where the adversary has either complete access to the target model parameters
or their loss gradients during training. This is the case, for example, in most distributed modes of
training. In between the two extremes, there are also attacks that make stronger assumptions than
the black-box ones, but do not assume full access to the model parameters. We refer to these attacks
as **partial white-box** attacks. It is important to add here that the majority of the works assume
full knowledge of the expected input, although some form of preprocessing might be required.
The time of the attack is another parameter to consider from a taxonomy point of view. The
majority of the research in the area is dealing with attacks during **inference**, however most
collaborative learning attacks assume access to the model parameters or gradients during **training** .
Attacks during the training phase of the model open up the possibility for different types of
adversarial behavior. A **passive** or _honest-but-curious_ attacker does not interfere with the training
process and they are only trying to infer knowledge during or after the training. If the adversary
interferes with the training in any way, they are considered an **active** attacker.
Finally, since the interest of this survey is in privacy attacks based on unintentional information
leakage regarding the data or the machine learning model, there is no coverage of _security-based_
attacks, such as model poisoning or evasion attacks, or attacks against the infrastructure that hosts
the data, models or provided services.


**4** **ATTACK TYPES**


In privacy-related attacks, the goal of an adversary is to gain knowledge that was not intended to
be shared. Such knowledge can be about the training data D or information about the model, or
even extracting information about properties of the data such as unintentionally encoded biases. In


8 Rigaki and Garcia


Fig. 2. Threat model in a collaborative learning setting. Dashed lines represent data and information flows,
while full lines represent possible actions. In red are the actions of the adversaries, available under the threat
model. In this setting the adversary can be placed either at the parameter server or locally. Model consumers
are not depicted for reasons of simplicity. In a federated learning setting, local model owners are also model

consumers.


our taxonomy, the privacy attacks studied are categorized into four types: **membership inference**,
**reconstruction**, **property inference**, and **model extraction** .


**4.1** **Membership Inference Attacks**


Membership inference tries to determine whether an input sample x was used as part of the training
set D . This is the most popular category of attacks and was first introduced by Shokri et al. [ 101 ].
The attack only assumes knowledge of the model’s output prediction vector (black-box) and was
carried out against supervised machine learning models. White-box attacks in this category are
also a threat, especially in a collaborative setting, where an adversary can mount both passive and
active attacks. If there is access to the model parameters and gradients, then this allows for more
effective white-box membership inference attacks in terms of accuracy [80].
Apart from supervised models, generative models such as GANs and VAEs are also susceptible
to membership inference attacks [ 15, 35, 39 ]. The goal of the attack, in this case, is to retrieve
information about the training data using varying degrees of knowledge of the data generating
components.


A Survey of Privacy Attacks in Machine Learning 9


Finally, these types of attacks can be viewed from a different perspective, that of the data owner.
In such a scenario, the owner of the data may have the ability to audit black-box models to see if
the data have been used without authorization [41, 103].


**4.2** **Reconstruction Attacks**


Reconstruction attacks try to recreate one or more training samples and/or their respective training
labels. The reconstruction can be partial or full. Previous works have also used the terms **attribute**
**inference** or **model inversion** to describe attacks that, given output labels and partial knowledge
of some features, try to recover sensitive features or the full data sample. For the purpose of this
survey, all these attacks are considered as part of the larger set of reconstruction attacks. The term
**attribute inference** has been used in other parts of the privacy related literature to describe attacks
that infer sensitive "attributes" of a targeted user by leveraging publicly accessible data [ 28, 48 ].
These attacks are not part of this review as they are mounted against the individual’s data directly
and not against ML models.
A major distinction between the works of this category is between those that create an actual
reconstruction of the data [ 36, 119, 123, 129, 130 ] and the ones that create class representatives or
probable values of sensitive features that do not necessarily belong to the training dataset [ 25, 38,
42, 123 ]. In classification models, the latter case is limited to scenarios where classes are made up
of one type of object, e.g., faces of the same person. While this limits the applicability of the attack,
it can still be an interesting scenario in some cases.


**4.3** **Property Inference Attacks**


The ability to extract dataset properties which were not explicitly encoded as features or were not
correlated to the learning task, is called **property inference** . An example of property inference is
the extraction of information about the ratio of women and men in a patient dataset when this
information was not an encoded attribute or a label of the dataset. Or having a neural network that
performs gender classification and can be used to infer if people in the training dataset wear glasses
or not. In some settings, this type of leak can have privacy implications. These types of properties
can also be used to get more insight about the training data, which can lead to adversaries using
this information to create similar models [ 3 ] or even have security implications when the learned
property can be used to detect vulnerabilities of a system [26].
Property inference aims to extract information that was learned from the model unintentionally
and that is not related to the training task. Even well generalized models may learn properties
that are relevant to the whole input data distribution and sometimes this is unavoidable or even
necessary for the learning process. What is more interesting from an adversarial perspective, are
properties that may be inferred from a specific subset of training data, or eventually about a specific
individual.

Property inference attacks so far target either dataset-wide properties [ 3, 26, 102 ] or the emergence of properties within a batch of data [ 75 ]. The latter attack was performed on the collaborative
training of a model.


**4.4** **Model Extraction Attacks**


**Model extraction** is a class of black-box attacks where the adversary tries to extract information
and potentially fully reconstruct a model by creating a substitute model _𝑓_ [ˆ] that behaves very
similarly to the model under attack _𝑓_ . There are two main focus for substitute models. First, to
create models that match the accuracy of the target model _𝑓_ in a test set that is drawn from the input
data distribution and related to the learning task [ 58, 77, 87, 109 ]. Second, to create a substitute


10 Rigaki and Garcia


model _𝑓_ [ˆ] that matches _𝑓_ at a set of input points that are not necessarily related to the learning
task [ 18, 45, 51, 109 ]. Jagielski et al. [ 45 ] referred to the former attack as **task accuracy** extraction
and the latter as **fidelity** extraction. In task accuracy extraction, the adversary is interested in
creating a substitute that learns the same task as the target model equally well or better. In the
latter case, the adversary aims to create a substitute that replicates the decision boundary of _𝑓_ as
faithfully as possible. This type of attack can be later used as a stepping stone before launching
other types of attacks such as adversarial attacks [ 51, 89 ] or membership inference attacks [ 80 ].
In both cases, it is assumed that the adversary wants to be as efficient as possible, i.e., to use as
few queries as possible. Knowledge of the target model architecture is assumed in some works, but
it is not strictly necessary if the adversary selects a substitute model that has the same or higher
complexity than the model under attack [51, 58, 87].
Apart from creating substitute models, there are also approaches that focus on recovering
information from the target model, such as hyper-parameters in the objective function [ 116 ]
or information about various neural network architectural properties such as activation types,
optimisation algorithm, number of layers, etc [86].


**5** **CAUSES OF PRIVACY LEAKS**

The conditions under which machine learning models leak is a research topic that has started to
emerge in the past few years. Some models leak information due to the way they are constructed.
An example of such a case is Support Vector Machines (SVMs), where the support vectors are
data points from the training dataset. Other models, such as linear classifiers are relatively easy
to "reverse engineer" and to retrieve their parameters just by having enough input / output data
pairs [ 109 ]. Larger models such as deep neural networks usually have a large number of parameters
and simple attacks are not feasible. However, under certain assumptions and conditions, it is
possible to retrieve information about either the training data or the models themselves.


**5.1** **Causes of Membership Inference Attacks**


One of the conditions that has been shown to improve the accuracy of membership inference is the
poor generalization of the model. The connection between overfitting and black-box membership
inference was initially investigated by Shokri et al. [ 101 ]. This paper was the first to examine
membership inference attacks on neural networks. The authors measured the effect of overfitting
on the attack accuracy by training models in different MLaaS platforms using the same dataset.
The authors showed experimentally that overfitting can lead to privacy leakage but also noted that
it is not the only condition, since some models that had lower generalization error where more
prone to membership leaks. The effect of overfitting was later corroborated formally by Yeom et
al. [ 125 ]. The authors defined membership advantage as a measure of how well an attacker can
distinguish whether a data sample belongs to the training set or not, given access to the model.
They proved that the membership advantage is proportional to the generalization error of the
model and that overfitting is a sufficient condition for performing membership inference attacks
but not a necessary one. Additionally, Long et al. [ 67 ] showed that even in well-generalized models,
it is possible to perform membership inference for a subset of the training data which they named
_vulnerable records_ .

Other factors, such as the model architecture, model type, and dataset structure, affect the attack
accuracy. Similarly to [ 101 ] but in the white-box setting, Nasr et al. [ 80 ] showed that two models
with the same generalization error showed different degrees of leakage. More specifically, the most
complex model in terms of number of parameters exhibited higher attack accuracy, showing that
model complexity is also an important factor.


A Survey of Privacy Attacks in Machine Learning 11


Truex et al. [ 110 ] ran different types of experiments to measure the significance of the model
type as well as the the number of classes present in the dataset. They found that certain model
types such as Naive Bayes are less susceptible to membership inference attacks than decision trees
or neural networks. They also showed that as the number of classes in the dataset increases, so
does the potential of membership leaks. This finding agrees with the results in [101].
Securing machine learning models against adversarial attacks can also have an adverse effect
on the model’s privacy as shown by Song et al. [ 105 ]. Current state of the art proposals for robust
model training, such as projective gradient descent (PGD) adversarial training [ 69 ], increase the
model’s susceptibility to membership inference attacks. This is not unexpected since robust training
methods (both empirical and provable defenses) tend to increase the generalization error. As
previously discussed, the generalization error is related to the success of the attack. Furthermore,
the authors of [ 105 ] argue that robust training may lead to increased model sensitivity to the
training data, which can also affect membership inference.
The generalization error is easily measurable in supervised learning under the assumption
that the test data can capture the nuances of the real data distribution. In generative models and
specifically in GANs this is not the case, hence the notion of overfitting is not directly applicable.
All three papers that deal with membership inference attacks against GANs mention overfitting as
an important factor behind successful attacks [ 15, 35, 39 ]. In this case, overfitting means that the
generator has memorized and replays part of the training data. This is further corroborated in the
study in [ 15 ], where their attacks are shown to be less successful as the training data size increases.


**5.2** **Causes of Reconstruction Attacks**

Regarding reconstruction attacks, Yeom et al. [ 125 ] showed that a higher generalization error
can lead to a higher probability to infer data attributes, but also that the influence of the target
feature on the model is an important factor. However, the authors assumed that the adversary has
knowledge of the prior distribution of the target features and labels. Using weaker assumptions
about the adversary’s knowledge, Zhang et al. [ 129 ] showed theoretically and experimentally
that a model that has high predictive power is more susceptible to reconstruction attacks. Finally,
similarly to vulnerable records in membership inference, memorization and retrieval of data which
are _out-of-distribution_ was shown to be the case even for models that do not overfit [11].


**5.3** **Causes of Property Inference Attacks**

Property inference is possible even with well-generalized models [ 26, 75 ] so overfitting does not
seem to be a cause of property inference attacks. Unfortunately, regarding property inference
attacks, we have less information about what makes them possible and under which circumstances
they appear to be effective. This is an interesting avenue for future research, both from a theoretical
and an empirical point of view.


**5.4** **Causes of Model Extraction**

While overfitting increases the success of black-box membership inference attacks, the exact
opposite holds for model extraction attacks. It is possible to steal model parameters when the
models under attack have 98% or higher accuracy in the test set [ 86 ]. Also models with a higher
generalization error are harder to steal, probably due to the fact that they may have memorized
samples that are not part of the attacker’s dataset [ 66 ]. Another factor that may affect model
extraction success is the dataset used for training. Higher number of classes may lead to worse
attack performance [66].


12 Rigaki and Garcia


**6** **IMPLEMENTATION OF THE ATTACKS**

More than 40 papers were analyzed in relation to privacy attacks against machine learning. This
section describes in some detail the most commonly used techniques as well as the essential
differences between them. The papers are discussed in two sections: attacks on centralized learning
and attacks on distributed learning.


**6.1** **Attacks Against Centralized Learning**

In the centralized learning setting, the main assumption is that models and data are collocated
during the training phase. The next subsection introduces a common design approach that is used
by multiple papers, namely, the use of _shadow models_ or _shadow training_ . The rest of the subsections
are dedicated to the different attack types and introduce the assumptions, common elements as
well as differences of the reviewed papers.


_6.1.1_ _Shadow training._ A common design pattern for a lot of supervised learning attacks is the use
of **shadow models** and **meta-models** or **attack-models** [ 3, 26, 41, 46, 86, 92, 94, 95, 101, 110 ].
The general shadow training architecture is depicted in Figure 3. The main intuition behind this
design is that models behave differently when they see data that do not belong to the training
dataset. This difference is captured in the model outputs as well as in their internal representations.
In most designs there is a target model and a target dataset. The adversary is trying to infer either
membership or properties of the training data. They train a number of shadow models using
shadow datasets D _𝑠ℎ𝑎𝑑𝑜𝑤_ = { **x** _𝑠ℎ𝑎𝑑𝑜𝑤,𝑖_ _,_ **y** _𝑠ℎ𝑎𝑑𝑜𝑤,𝑖_ } _[𝑛]_ _𝑖_ =1 [that usually are assumed to come from the]
same distribution as the target dataset. After the shadow models’ training, the adversary constructs
an attack dataset D _𝑎𝑡𝑡𝑎𝑐𝑘_ = { _𝑓_ _𝑖_ ( **x** _𝑠ℎ𝑎𝑑𝑜𝑤,𝑖_ ) _,_ **y** _𝑠ℎ𝑎𝑑𝑜𝑤,𝑖_ } _[𝑛]_ _𝑖_ =1 [, where] _[ 𝑓]_ _[𝑖]_ [is the respective shadow model.]
The attack dataset is used to train the meta-model, which essentially performs inference based on
the outputs of the shadow models. Once the meta-model is trained, it is used for testing using the
outputs of the target model.


_6.1.2_ _Membership inference attacks._ In _membership inference_ black-box attacks, the most common
attack pattern is the use of shadow models. The output of the shadow models is usually a prediction
vector [ 46, 92, 95, 101, 110 ]. The labels used for the attack dataset come from the test and training
splits of the shadow data, where the data points that belong to the test set are labeled as nonmembers of the training set. The meta-model is trained to recognize patterns in the prediction
vector output of the target model. These patterns allow the meta-model to infer whether a data
point belongs to the training dataset or not. The number of shadow models affects the attack
accuracy, but it also incurs cost to the attackers. Salem et al. [ 95 ] showed that membership inference
attacks are possible with as little as one shadow model.
Shadow training can be further reduced to a threshold-based attack, where instead of training a
meta-model, one can calculate a suitable threshold function that indicates whether a sample is a
member of the training set. The threshold can be learned from multiple shadow models [ 94 ] or
even without using any shadow models [ 125 ]. Sablayrolles et al. [ 94 ] showed that a Bayes optimal
membership inference attack depends only on the loss and their attack outperforms previous
attacks such as [ 101, 125 ]. In terms of attack accuracy, they reported up to 90.8% on large neural
network models such as VGG16 [ 64 ] that were performing classification on the Imagenet [ 20 ]
dataset.

In addition to relaxations on the number of shadow models, attacks have been shown to be data
driven, i.e., an attack can be successful even if the target model is different than the shadow and
meta-models [ 110 ]. The authors tested several types of models such as k-NN, logistic regression,
decision trees and naive Bayes classifiers in different combinations on the role of the target model,


A Survey of Privacy Attacks in Machine Learning 13


Fig. 3. Shadow training architecture. At first, a number of shadow models are trained with their respective
shadow datasets in order to emulate the behavior of the target model. At the second stage, a meta-model
is being trained from the outputs of the shadow models and the known labels of the shadow datasets. The
meta-model is used to infer membership or properties of data or the model given the output of the target
model.


shadow and meta model. The results showed that i) using different types of models did not affect the
attack accuracy and ii) in most cases, models such as decision trees outperformed neural networks
in terms of attack accuracy and precision.
Shadow model training requires a shadow dataset. One of the main assumptions of membership
inference attacks on supervised learning models is that the adversary has no or limited knowledge
of the training samples used. However, the adversary knows something about the underlying
data distribution of the training data. If the adversary does not have access to a suitable dataset,
they can try to generate one [ 101, 110 ]. Access to statistics about the probability distribution
of several features allows an attacker to create the shadow dataset using sampling techniques.
If a statistics-based generation is not possible, a query-based approach using the target models’
prediction vectors is another possibility. Generating auxiliary data using GANs was also proposed
by Hayes et al. [ 35 ]. If the adversary manages to find input data that generate predictions with high
confidence, then no prior knowledge of the data distribution is required for a successful attack [ 101 ].
Salem et al. [ 95 ] went so far as to show that it is not even necessary to train the shadow models
using data from the same distribution as the target, making the attack more realistic since it does
not assume any knowledge of the training data.
The previous discussion is mostly relevant to supervised classification or regression tasks. The
efficacy of membership inference attacks against sequence-to-sequence models training for machine
translation, was studied by [ 41 ]. The authors used shadow models that try to mimic the target
model’s behavior and then used a meta-model to infer membership. They found that sequence
generation models are much harder to attack compared to other types of models such as image


14 Rigaki and Garcia


classification. However, membership of _out-of-domain_ and out-of-vocabulary data was easier to
infer.

Membership inference attacks are also applicable to deep generative models such as GANs and
VAEs [ 15, 35, 39 ]. Since these models have more than one component (generator/discriminator,
encoder/decoder), adversarial knowledge needs to take that into account. For these types of models,
the taxonomy proposed by Chen et al. [ 15 ] is partially followed. We consider black-box access to
the generator as the ability to access generated samples and partial black-box access, the ability to
provide inputs _𝑧_ and generate samples. Having access to the generator model and its parameters is
considered a white-box attack. The ability to query the discriminator is also a white-box attack.
The full white-box attacks with access to the GAN discriminator are based on the assumption that
if the GAN has "overfitted", then the data points used for its training will receive higher confidence
values as output by the discriminator [ 35 ]. In addition to the previous attack, Hayes et al. [ 35 ]
proposed a set of attacks in the partial black-box setting. These attacks are applicable to both GANs
and VAEs or any generative model. If the adversary has no auxiliary data, they can attempt to train
an auxiliary GAN whose discriminator distinguishes between the data generated by the target
generator and the data generated by the auxiliary GAN. Once the auxiliary GAN is trained, its
discriminator can be used for the white-box attack. The authors considered also scenarios where

the adversary may have auxiliary information such as knowledge of training and test data. Using
the auxiliary data, they can train another GAN whose discriminator would be able to distinguish
between members of the original training set and non-members.
A distance-based attack over the nearest neighbors of a data point was proposed by Chen et
al. [ 15 ] for the full black-box model. In this case, a data point x is a member of the training set if
within its k-nearest neighbors there is at least one point that has a distance lower than a threshold
_𝜖_ . The authors proposed more complex attacks as the level of knowledge of the adversary increases,
based on the idea that the reconstruction error between the real data point _𝑥_ and a sample generated
by the generator given some input _𝑧_ should be smaller if the data point is coming from the training

set.


_6.1.3_ _Reconstruction attacks._ The initial reconstruction attacks were based on the assumption that
the adversary has access to the model _𝑓_, the priors of the sensitive and nonsensitive features, and the
output of the model for a specific input _𝑥_ . The attack was based on estimating the values of sensitive
features, given the values of nonsensitive features and the output label [ 25 ]. This method used a
maximum a posteriori (MAP) estimate of the attribute that maximizes the probability of observing
the known parameters. Hidano et al. [ 38 ] used a similar attack but they made no assumption about
the knowledge of the nonsensitive attributes. In order for their attack to work, they assumed that
the adversary can perform a _model poisoning_ attack during training.
Both previous attacks worked against linear regression models, but as the number of features
and their range increases, the attack feasibility decreases. To overcome the limitations of the MAP
attack, Fredrikson et al. [ 24 ] proposed another inversion attack which recovers features using target
labels and optional auxiliary information. The attack was formulated as an optimization problem
where the objective function is based on the observed model output and uses gradient descent in the
input space to recover the input data point. The method was tested on image reconstruction. The
result was a class representative image which in some cases was quite blurry even after denoising.
A formalization of the model inversion attacks in [24, 25] was later proposed by Wu et al. [120].
Since the optimization problem in [ 24 ] is quite hard to solve, Zhang et al. [ 129 ] proposed to
use a GAN to learn some auxiliary information of the training data and produce better results.
The auxiliary information in this case is the presence of blurring or masks in the input images.
The attack first uses the GAN to learn to generate realistic looking images from masked or blurry


A Survey of Privacy Attacks in Machine Learning 15


images using public data. The second step is a GAN inversion that calculates the latent vector ˆ _𝑧_
which generates the most likely image:


ˆ
_𝑧_ = arg min _𝑧_ _𝐿_ _𝑝𝑟𝑖𝑜𝑟_ ( _𝑧_ ) + _𝜆𝐿_ _𝑖𝑑_ ( _𝑧_ ) (6)


where the prior loss _𝐿_ _𝑝𝑟𝑖𝑜𝑟_ is ensuring the generation of realistic images and _𝐿_ _𝑖𝑑_ ensures that the
images have a high likelihood in the target network. The attack is quite successful, especially on
masked images.
The only black-box reconstruction attack until now was proposed by Yang et al. [ 123 ]. This
attack employs an additional classifier that performs an inversion from the output of the target
model _𝑓_ ( _𝑥_ ) to a candidate output ˆ _𝑥_ . The setup is similar to that of an autoencoder, only in this
case the target network that plays the role of the encoder is a black box and it is not trainable. The
attack was tested on different types of target model outputs: the full prediction vector, a truncated
vector, and the target label only. When the full prediction vector is available, the attack performs a
good reconstruction, but with less available information, the produced data point looks more like a
class representative.


_6.1.4_ _Property inference attacks._ In _property inference_ the shadow datasets are labeled based on
the properties that the adversary wants to infer, so the adversary needs access to data that have
the property and data that do not have it. The meta-model is then trained to infer differences in
the output vectors of the data that have the property versus the ones that they do not have it. In
white-box attacks, the meta-model input can be other feature representations such as the support
vectors of an SVM [ 3 ] or transformations of neural network layer outputs [ 26 ]. When attacking
language model embeddings, the embedding vectors themselves can be used to train a classifier to
distinguish between properties such as text authorship [102].


_6.1.5_ _Model extraction attacks._ When the adversary has access to the inputs and prediction outputs
of a model, it is possible to view these pairs of inputs and outputs as a system of equations, where the
unknowns are the model parameters [ 109 ] or hyper-parameters of the objective function [ 116 ]. In
the case of a linear binary classifier, the system of equations is linear and only _𝑑_ + 1 queries are necessary to retrieve the model parameters, where _𝑑_ is the dimension of the parameter vector _𝜃_ . In more
complex cases, such as multi-class linear regression or multi-layer perceptrons, the systems of equations are no longer linear. Optimization techniques such as Broyden–Fletcher–Goldfarb–Shanno
(BFGS) [ 85 ] or stochastic gradient descent are then used to approximate the model parameters [ 109 ].
Lack of prediction vectors or a high number of model parameters renders equation solving
attacks inefficient. A strategy is required to select the inputs that will provide the most useful
information for model extraction. From this perspective, model extraction is quite similar to _active_
_learning_ [ 14 ]. Active learning makes use of an external oracle that provides labels to input queries.
The oracle can be a human expert or a system. The labels are then used to train or update the
model. In the case of model extraction, the target model plays the role of the oracle.
Following the active learning approach, several papers propose an adaptive training strategy.
They start with some initial data points or _seeds_ which they use to query the target model and
retrieve labels or prediction vectors which they use to train the substitute model _𝑓_ [ˆ] . For a number
of subsequent rounds, they extend their dataset with new synthetic data points based on some
adaptive strategy that allows them to find points close to the decision boundary of the target
model [ 14, 51, 89, 109 ]. Chandrasekaran et al. [ 14 ] provided a more query efficient method of
extracting nonlinear models such as kernel SVMs, with slightly lower accuracy than the method
proposed by Tramer et al. [109], while the opposite was true for Decision Tree models.


16 Rigaki and Garcia


Several other strategies for selecting the most suitable data for querying the target model use: (i)
data that are not synthetic but belong to different domains such as images from different datasets [ 6,
18, 87 ], (ii) semi-supervised learning techniques such as rotation loss [ 127 ] or MixMatch [ 7 ] to
augment the dataset [ 45 ] or (iii) randomly generated input data [ 51, 58, 109 ]. In terms of efficiency,
semi-supervised methods such as MixMatch require much fewer queries than fully supervised
extraction methods to perform similarly or better in terms of task accuracy and fidelity, against
models trained for classification using CIFAR-10 and SVHN datasets [ 45 ]. For larger models,
trained for Imagenet classification, even querying a 10% of the Imagenet data, gives a comparable
performance to the target model [ 45 ]. Against a deployed MLaaS service that provides facial
characteristics, Orekondy et al. [ 87 ] managed to create a substitute model that performs at 80% of
the target in task accuracy, spending as little as $30.
Some, mostly theoretical, work has demonstrated the ability to perform direct model extraction
beyond linear models [ 45, 77 ]. Full model extraction was shown to be theoretically possible against
two-layer fully connected neural networks with rectified linear unit (ReLU) activations by Milli
et al. [ 77 ]. However, their assumption was that the attacker has access to the loss gradients with
respect to the inputs. Jagielski et al. [ 45 ] managed to do a full extraction of a similar network
without the need of gradients. Both approaches take into account that ReLUs transforms the neural
network into a piecewise linear function of the inputs. By probing the model with different inputs,
it is possible to identify where the linearity breaks and use this knowledge to calculate the network
parameters. In a hybrid approach that uses both a learning strategy and direct extraction, Jagielski
et al. [ 45 ], showed that they can extract a model trained on MNIST with almost 100% fidelity by
using an average of 2 [19] _[.]_ [2] to 2 [22] _[.]_ [2] queries against models that contain up to 400,000 parameters.
However, this attack assumes access to the loss gradients similarly to [77].
Finally, apart from learning substitute models directly, there is also the possibility of extracting
model information such as architecture, optimization methods and hyper-parameters using shadow
models [ 86 ]. The majority of attacks were performed against neural networks trained on MNIST.
Using the shadow models’ prediction vectors as input, the meta-models managed to learn to
distinguish whether a model has certain architectural properties. An additional attack by the
same authors, proposed to generate adversarial samples which were created by models that have
the property in question. The generated samples were created in a way that makes a classifier
output a certain prediction if they have the attribute in question. The target model’s prediction on
this adversarial sample is then used to establish if the target model has a specific property. The
combination of the two attacks proved to be the most effective approach. Some properties such as
activation function, presence of dropout, and max-pooling were the most successfully predicted.


**6.2** **Attacks Against Distributed Learning**


In the federated learning setting, multiple devices acquire access to the global model that is trained
from data that belong to different end users. Furthermore, the parameter server has access to the
model updates of each participant either in the form of model parameters or that of loss gradients.
In split learning settings, the central server also gains access to the outputs of each participant’s
intermediate neural network layers. This type of information can be used to mount different types
of attacks by actors that are either residing in a central position or even by individual participants.
The following subsection presents the types of attacks in distributed settings, as well as their
common elements, differences, and assumptions.


_6.2.1_ _Membership inference attacks._ Nasr et al. [ 79 ] showed that a membership inference attack is
more effective than the black-box one, under the assumption that the adversary has some auxiliary
knowledge about the training data, i.e., has access to some data from the training dataset, either


A Survey of Privacy Attacks in Machine Learning 17


explicitly or because they are part of a larger set of data that the adversary possesses. The adversary
can use the model parameters and the loss gradients as inputs to another model which is trained
to distinguish between members and non-members. The white-box attack accuracy with various
neural network architectures was up to 75.1%, however, all target models had a high generalization

error.

In the active attack scenario, the attacker, which is also a local participant, alters the gradient
updates to perform a gradient ascent instead of descent for the data whose membership is under
question. If some other participant uses the data for training, then their local SGD will significantly
reduce the gradient of the loss and the change will be reflected in the updated model, allowing the
adversary to extract membership information. Attacks from a local active participant reached an
attack accuracy of 76.3% and in general, the active attack accuracy was higher than the passive
accuracy in all tested scenarios. However, as the number of participants increases, it has adverse
effects on the attack accuracy, which drops significantly after five or more participants. A global
active attacker which is in a more favourable position, can isolate the model parameter updates
they receive from each participant. Such an active attacker reached an attack accuracy of 92.1%.


_6.2.2_ _Property inference attacks._ Passive property inference requires access to some data that
possess the property and some that do not. The attack applies to both federated average and
synchronized SGD settings, where each remote participant receives parameter updates from the
parameter server after each training round [ 75 ]. The initial dataset is of the form D [′] = {( x _,_ y _,_ y [′] )},
where x and y are the data used for training the distributed model and y [′] are the property labels.
Every time the local model is updated, the adversary calculates the loss gradients for two batches of
data. One batch that has the property in question and one that does not. This allows the construction
of a new dataset that consists of gradients and property labels (∇ _𝐿,_ y [′] ) . Once enough labeled data
have been gathered, a second model, _𝑓_ [′], is trained to distinguish between loss gradients of data that
have the property versus those that do not. This model is then used to infer whether subsequent
model updates were made using data that have the property. The model updates are assumed to
be done in batches of data. The attack reaches an attack area under the curve (AUC) score of 98%
and becomes increasingly more successful as the number of epochs increases. Attack accuracy
also increases as the fraction of data with the property in question also increases. However, as
the number of participants in the distributed model increases, the attack performance decreases
significantly.


_6.2.3_ _Reconstruction attacks._ Some data reconstruction attacks in a federated learning setting use
generative models and specifically GANs [ 42, 119 ]. When the adversary is one of the participants,
they can force the victims to release more information about the class they are interested in
reconstructing [ 42 ]. This attack works as follows: The potential victim has data for a class "A"
that the adversary wants to reconstruct. The adversary trains an additional GAN model. After
each training round, the adversary uses the target model parameters for the GAN discriminator,
whose purpose is to decide whether the input data come from the class "A" or are generated by the
generator. The aim of the GAN is to create a generator that is able to generate faithful class "A"
samples. In the next training step of the target model, the adversary generates some data using the
GAN and labels them as class "B". This forces the target model to learn to discriminate between
classes "A" and "B" which in turn improves the GAN training and its ability to generate class "A"
representatives.
If the adversary has access to the central parameter server, they have direct access to the model
updates of each remote participant. This makes it possible to perform more successful reconstruction
attacks [ 119 ]. In this case, the GAN discriminator is again using the shared model parameters and
learns to distinguish between real and generated data, as well as the identity of the participant.


18 Rigaki and Garcia


Once the generator is trained, the reconstructed samples are created using an optimization method
that minimizes the distance between the real model updates and the updates due to the generated
data. Both GAN based methods assume access to some auxiliary data that belong to the victims.
However, the former method generates only class representatives.
In a synchronized SGD setting, an adversary with access to the parameter server has access
to the loss gradients of each participant during training. Using the loss gradients is enough to
produce a high quality reconstruction of the training data samples, especially when the batch size is
small [ 130 ]. The attack uses a second "dummy" model. Starting with random dummy inputs _𝑥_ [′] and
labels _𝑦_ [′], the adversary tries to match the dummy model’s loss gradients ∇ _𝜃_ J [′] to the participant’s
loss gradients ∇ _𝜃_ J . This gradient matching is formulated as an optimization task that seeks to
find the optimal _𝑥_ [′] and _𝑦_ [′] that minimize the gradients’ distance:


_𝑥_ [∗] _,𝑦_ [∗] = arg min (7)
_𝑥_ [′] _,𝑦_ [′] [ ∥∇] _[𝜃]_ [J] [ ′] [(D] [′] [;] _[𝜃]_ [) −∇] _[𝜃]_ [J (D][;] _[𝜃]_ [)∥] [2]


The minimization problem in Equation 7 is solved using limited memory BFGS (L-BFGS) [ 62 ]. The
size of the training batch is an important factor in the speed of convergence in this attack.
Data reconstruction attacks are also possible during the inference phase in the split learning
scenario [ 36 ]. When the local nodes process new data, they perform inference on these initial layers
and then send their outputs to the centralized server. In this attack, the adversary is placed in the
centralized server and their goal is to try to reconstruct the data used for inference. He et al. [ 36 ]
cover a range of scenarios: (i) white-box, where the adversary has access to the initial layers and
uses them to reconstruct the images, (ii) black-box where the adversary has no knowledge of the
initial layers but can query them and thus recreate the missing layers and (iii) query-free where the
adversary cannot query the remote participant and tries to create a substitute model that allows
data reconstruction. The latter attack produces the worst results, as expected, since the adversary
is the weakest. The split of the layers between the edge device and the centralized server is also
affecting the quality of reconstruction. Fewer layers in the edge neural network allow for better
reconstruction in the centralized server.


**6.3** **Summary of Attacks**

To summarize the attacks proposed against machine learning privacy, Table 1 presents the 42
papers analyzed in terms of adversarial knowledge, model under attack, attack type, and timing of
the attack.

In terms of model types, 83.3% of the papers dealt with attacks against neural networks, with
decision trees being the second most popular model to attack at 11.9% (some papers covered attacks
against multiple model types). The concept of neural networks groups together both shallow and
deep models, as well as multiple architectures, such as convolutional neural networks, recurrent
neural networks, while under SVMs we group together both linear and nonlinear versions.
The most popular attack types are membership inference and reconstruction attacks (35.7% of
the papers, respectively), with model extraction the next most popular (31%). The majority of the
proposed attacks are performed during the inference phase (88%). Attacks during training are
mainly on distributed forms of learning. Black-box and white-box attacks were studied in 66.7% and
54.8% of the papers, respectively (some papers covered both settings). In the white-box category,
we also include partial white-box attacks.
The focus on neural networks in the existing literature as well as the focus on supervised learning
is also apparent in Figure 4. The figure depicts types of machine learning algorithms versus the
types of attacks that have been studied so far based on the existing literature. The list of algorithms
is indicative and not exhaustive, but it contains the most popular ones in terms of research and


A Survey of Privacy Attacks in Machine Learning 19


Table 1. Summary of papers on privacy attacks on machine learning systems, including information of
their assumptions about adversarial knowledge (black / white-box), the type of model(s) under attack, the
attack type, and the timing of the attack (during training or during inference). The transparent circle in the
Knowledge column indicates partial white-box attacks.



Reference Year Knowledge ML Algorithms Attack Type Timing


Fredrikson et al. [25] 2014 - - - Fredrikson et al. [24] 2015 - - - - - Ateniese et al. [3] 2015 - - - - Tramer et al. [109] 2016 - - - - - - - Wu et al. [120] 2016 - - - - - Hidano et al. [38] 2017 - - - Hitaj et al. [42] 2017 - - - Papernot et al. [89] 2017 - - - Shokri et al. [101] 2017 - - - Correia-Silva et al. [18] 2018 - - - Ganju et al. [26] 2018 - - - Oh et al. [86] 2018 - - - Long et al. [67] 2018 - - - Rahman et al. [92] 2018 - - - Wang & Gong [116] 2018 - - - - - - Yeom et al. [125] 2018 - ◦ - - - - Carlini et al. [11] 2019 - - - Hayes et al. [35] 2019 - - - - He et al. [36] 2019 - - - - Hilprecht et al. [39] 2019 - - - Jayaraman & Evans [46] 2019 - - - - - Juuti et al. [51] 2019 - - - Milli et al. [77] 2019 - - - Nasr et al. [80] 2019 - - - Melis et al. [75] 2019 - - - - Orekondy et al. [87] 2019 - - - Sablayrolles et al. [94] 2019 ◦ - - Salem et al. [95] 2019 - - - Song L. et al. [105] 2019 - - - Truex, et al. [110] 2019 - - - - - Wang et al. [119] 2019 - - - Yang et al. [123] 2019 - - - Zhu et al. [130] 2019 - - - Barbalau et al. [6] 2020 - - - Chandrasekaran et al. [14] 2020 - - - - - Chen et al. [15] 2020 - - - - Hishamoto et al. [41] 2020 - - - Jagielski et al. [45] 2020 - - - Krishna et al. [58] 2020 - - - Pan et al. [88] 2020 - - - Song & Raghunathan [102] 2020 - - - - - - Zhang et al. [129] 2020 - - - 


deployment in real-world systems. Algorithms such as random forests [ 10 ] or gradient boosting
trees [ 16, 55 ] have received little to no focus and the same holds for whole areas of machine learning
such as reinforcement learning.


20 Rigaki and Garcia


Fig. 4. Map of attack types per algorithm. The list of algorithm presented is not exhaustive but indicative.
Underneath each algorithm or area of machine learning there is an indication of the attacks that have been
studied so far. A red box indicates no attack.


Fig. 5. Number of papers used against each learning task and attack type. Classification includes both binary
and multi-class classification. Darker gray means higher number of papers.


Another dimension that is interesting to analyze is the types of learning tasks that have been
the target of attacks so far. Figure 5 presents information about the number of papers in relation to
the learning task and the attack type. By learning task, we refer to the task in which the target
model is initially trained. As the figure clearly shows, the majority of the attacks are on models
that were trained for classification tasks, both binary and multiclass. This is the case across all four
attack types.
While there is a diverse set of reviewed papers, it is possible to discern some high-level patterns
in the proposed attacking techniques. Figure 6 shows the number of papers in relation to the


A Survey of Privacy Attacks in Machine Learning 21


Fig. 6. Number of papers that used an attacking technique for each attack type. Darker gray means higher
number of papers.


attacking technique and attack type. Most notably, nine papers used shadow training mainly for
membership and property inference attacks. Active learning was quite popular in model extraction
attacks and was proposed by four papers. Generative models (mostly GANs) were used in five
papers across all attack types and another three papers used gradient matching techniques. It should
be noted here that the "Learning" technique includes a number of different approaches, spanning
from using model parameters and gradients as inputs to classifiers [ 75, 79 ] to using input-output
queries for substitute model creation [ 18, 45, 87 ] and learning classifiers from language models
for reconstruction [ 88 ] and property inference [ 102 ]. In "Threshold" based attacks, we categorized
the attacks proposed in [ 125 ] and [ 94 ] and subsequent papers that used them for membership and
property inference.
Some attacks may be applicable to multiple learning tasks and datasets, however, this is not the
case universally. Dataset size, number of classes, and features might also be factors for the success
of certain attacks, especially since most of them are empirical. Table 2 is a summary of the datasets
used in all attack papers along with the data types of their features, the learning task they were
used for, and the dataset size. The datasets were used during the training of the target models and
in some cases as auxiliary information during the attacks. The table contains 51 unique datasets
used across 42 papers, an indication of the variation of different approaches.
This high variation is both a blessing and a curse. On the one hand, it is highly desirable to use
multiple types of datasets to test different hypotheses and the majority of the reviewed research
follows that approach. On the other hand, these many options make it harder to compare methods.
As it is evident from Table 2, some of the datasets are quite popular. MNIST, CIFAR-10, CIFAR-100,
and UCI Adult have been used by more than six papers, while 26 datasets have been used by only

one paper.
The number of model parameters varies based on the model, task and datasets used in the
experiments. As it can be seen in Table 2, most datasets are not extremely large, hence the models
under attack are not extremely large. Given that most papers deal with neural networks, this might
indicate that most attacks focused on smaller datasets and models which might not be representative
of realistic scenarios. However, privacy attacks do not necessarily have to target large models with


22 Rigaki and Garcia


Table 2. Summary of datasets used in the papers about privacy attacks on machine learning systems. The
size of each dataset is measured by the number of samples unless otherwise indicated. A range in the size
column indicates that different papers used different subsets of the dataset.


**Name** **Data Type** **Learning Task** **Reference(s)** **Size (Samples)**
538 Steak Survey [37] mixed features multi-class classification [14, 24, 38, 109] 332
AT&T Faces [4] images multi-class classification [24, 42, 119] 400
Bank Marketing [21] mixed features multi-class classification [116] 45,210
Bitcoin prices time series regression [109] 1,076
Book Corpus [131] text word-level language model [102] 14,000 sent.
Breast Cancer [21] numerical feat. binary classification [14, 67, 109] 699
Caltech 256 [31] images multi-class classification [87] 30,607
Caltech birds [115] images multi-class classification [87] 6,033
CelebA [63] images binary classification [6, 15, 26, 123, 129] 20-202,599
CIFAR-10 [59] images image generation, multi-class [ 6, 35, 36, 39, 45, 77, 92, 60,000
classification 94, 95, 101, 105, 110, 123,
125]

CIFAR-100 [59] images multi-class classification [ 6, 46, 80, 95, 101, 125, 60,000
130]

CLiPS stylometry [113] text binary classification [75] 1,412 reviews
Chest X-ray [118] images multi-class classification [129] 10,000
Diabetes [21] time series binary class., regression [14, 109, 116] 768
Diabetic ret. [53] images image generation [35, 87] 88,702
Enron emails text char-level language model [11]  Eyedata [96] numerical feat. regression [125] 120
FaceScrub [83] images binary classification [75, 123] 18,809-48,579
Fashion-MNIST [121] images multi-class classification [6, 39, 45, 105] 60,000
Foursquare [122] mixed features binary classification [75, 95, 101] 528,878
Geog. Orig. Music [21] numerical feat. regression [116] 1,059
German Credit [21] mixed features binary classification [109] 1,000
GSS marital survey [32] mixed features multi-class classification [14, 24, 109] 16127
GTSRB [107] images multi-class classification [51, 89] 51839
HW Perf. Counters (private) numerical feat. binary classification [26] 36,000
Imagenet [20] images multi-class classification [6, 45, 86, 94] 14,000,000
Instagram [5] location data vector generation [15]  Iris [23] numerical feat. multi-class classification [14, 109] 150
IWPC [17] mixed features regression [25, 125] 3497
IWSLT Eng-Vietnamese [68] text neural machine translation [11]  LFW [43] images image generation [35, 75, 130] 13233
Madelon [21] mixed features multi-class classification [116] 4,400
MIMIC-III [50] binary features record generation [15] 41,307
Movielens 1M [33] numerical feat. regression [38] 1,000,000
MNIST [60] images multi-class classification [ 14, 26, 36, 39, 42, 45, 51, 70,000
67, 77, 86, 89, 92, 95, 101,

109, 110, 119, 123, 125,
129, 130]

Mushrooms [21] categorical feat. binary classification [14, 109] 8,124
Netflix [81] binary features binary classification [125] 2,416
Netflows (private) network data binary classification [3]  PTB [72] text char-level language model [11] 5 MB
PiPA [128] images binary classification [75] 18,000
Purchase-100 [52] binary features multi-class classification [46, 80, 101, 110] 197,324
SVHN [82] images multi-class classification [45, 130] 60,000
TED talks [44] text machine translation [11] 100,000 pairs
Texas-100 [13] mixed features multi-class classification [80, 101] 67,330
UJIndoor [21] mixed features regression [116] 19,937
UCI / Adult [21] various binary classification [ 14, 26, 67, 95, 101, 109, 48,842
110]

Voxforge [114] audio speech recognition [3] 11,137 rec.
Wikipedia [70] text language model [102] 150,000 articles
Wikitext-103 [76] text word-level language model [11, 58] 500 MB
Yale-Face [27] images multi-class classification [105] 2,414
Yelp reviews [124] text binary classification [75] 16-40,000


extreme amounts of data; and neural networks, however popular, are not necessarily the most used
models in the "real world".


A Survey of Privacy Attacks in Machine Learning 23


**7** **DEFENDING MACHINE LEARNING PRIVACY**

Leaking personal information such as medical records or credit card numbers is usually an undesirable situation. The purpose of studying attacks against machine learning models is to be able
to explore the limitations and assumptions of machine learning and to anticipate the adversaries’
actions. Most of the analyzed papers propose and test mitigations to counter their attacks. In the
next subsections, we present the various defences proposed in several papers organized by the type
of attack they attempt to defend against.


**7.1** **Defenses Against Membership Inference Attacks**

The most prominent defense against membership inference attacks is Differential Privacy (DP),
which provides a guarantee on the impact that single data records have on the output of an algorithm
or a model. However, other defenses have been tested empirically and are also presented in the
following subsections.


_7.1.1_ _Differential Privacy._ Differential privacy started as a privacy definition for data analysis and
it is based on the idea of "learning nothing about an individual while learning useful information
about a population" [ 22 ]. Its definition is based on the notion that if two databases differ only by
one record and are used by the same algorithm (or mechanism), the output of that algorithm should
be similar. More formally,


_Definition 7.1 ((_ _𝜖,𝛿_ _)-Differential Privacy)._ A randomized mechanism M with domain R and
output S is ( _𝜖,𝛿_ )-differentially private if for any adjacent inputs _𝐷, 𝐷_ [′] ∈R and for any subsets of
outputs S it holds that:


_𝑃𝑟_ [M( _𝐷_ ) ∈S] ≤ _𝑒_ _[𝜖]_ _𝑃𝑟_ [M( _𝐷_ [′] ) ∈S] + _𝛿_ (8)


where _𝜖_ is the privacy budget and _𝛿_ is the failure probability.


The original definition of DP did not include _𝛿_ which was introduced as a relaxation that allows
some outputs not to be bounded by _𝑒_ _[𝜖]_ .
The usual application of DP is to add Laplacian or Gaussian noise to the output of a query or
function over the database. The amount of noise is relevant to the _sensitivity_ which gives an upper
bound on how much we must perturb the output of the mechanism to preserve privacy [22]:


_Definition 7.2. 𝑙_ 1 (or _𝑙_ 2 )-Sensitivity of a function _𝑓_ is defined as


Δ _𝑓_ = _𝐷,𝐷_ [′] _,_ max ∥ _𝐷_ − _𝐷_ [′] ∥=1 [∥] _[𝑓]_ [(] _[𝐷]_ [) −] _[𝑓]_ [(] _[𝐷]_ [′] [)∥] (9)


where ∥ _._ ∥ is the _𝑙_ 1 or the _𝑙_ 2 -norm and the max is calculated over all possible inputs _𝐷, 𝐷_ [′] .
From a machine learning perspective, _𝐷_ and _𝐷_ [′] are two datasets that differ by one training
sample and the randomized mechanism M is the machine learning training algorithm. In deep
learning, the noise is added at the gradient calculation step. Because it is necessary to bound the
gradient norm, gradient clipping is also applied [1].
Differential privacy offers a trade-off between privacy protection and utility or model accuracy.
Evaluation of differentially private machine learning models against membership inference attacks
concluded that the models could offer privacy protection only when they considerably sacrifice
their utility [ 46, 92 ]. Jayaraman et al. [ 46 ] evaluated several relaxations of DP in both logistic
regression and neural network models against membership inference attacks. They showed that
these relaxations have an impact on the utility-privacy trade-off. While they reduce the required
added noise, they also increase the privacy leakage.


24 Rigaki and Garcia


Distributed learning scenarios require additional considerations regarding differential privacy.
In a centralized model, the focus is on sample level DP, i.e., on protecting privacy at the individual
data point level. In a federated learning setting where there are multiple participants, we not only
care about the individual training data points they use, but also about ensuring privacy at the
participant level. A proposal which applies DP at the participant level was introduced by McMahan
et al. [ 74 ] however, it requires a large number of participants. When it was tested with a number as
low as 30, the method was deemed unsuccessful [75].


_7.1.2_ _Regularization._ Regularization techniques in machine learning aim to reduce overfitting
and increase model generalization performance. Dropout [ 106 ] is a form of regularization that
randomly drops a predefined percentage of neural network units during training. Given that
black-box membership inference attacks are connected to overfitting, it is a sensible approach
to this type of attack and multiple papers have proposed it as a defense with varying levels of
success [ 35, 75, 95, 101, 105 ]. Another form of regularization uses techniques that combine multiple
models that are trained separately. One of those methods, model stacking, was tested in [ 95 ] and
produced positive results against membership inference. An advantage of model stacking or similar
techniques is that they are model agnostic and do not require that the target model is a neural
network.


_7.1.3_ _Prediction vector tampering._ As many models assume access to the prediction vector during
inference, one of the countermeasures proposed was the restriction of the output to the top k classes
or predictions of a model [ 101 ]. However, this restriction, even in the strictest form (outputting
only the class label) did not seem to fully mitigate membership inference attacks, since information
leaks can still happen due to model misclassifications. Another option is to lower the precision of
the prediction vector, which leads to less information leakage [ 101 ]. Adding noise to the output
vector also affected membership inference attacks [49].


**7.2** **Defenses Against Reconstruction Attacks**

Reconstruction attacks often require access to the loss gradients during training. Most of the
defences against reconstruction attacks propose techniques that affect the information retrieved
from these gradients. Setting all loss gradients which are below a certain threshold to zero, was
proposed as a defence against reconstruction attacks in deep learning. This technique proved quite
effective with as little as 20% of the gradients set to zero and with negligible effects on model
performance [ 130 ]. On the other hand, performing quantization or using half-precision floating
points for neural network weights did not seem to deter the attacks in [ 11 ] and [ 130 ], respectively.


**7.3** **Defenses Against Property Inference Attacks**

Differential privacy is designed to provide privacy guarantees in membership inference attack
scenarios and it does not seem to offer protection against property inference attacks [ 3 ]. In addition
to DP, Melis et al. [75] explored other defenses against property inference attacks. Regularization
(dropout) had an adverse effect and actually made the attacks stronger. Since the attacks in [ 75 ]
were performed in a collaborative setting, the authors tested the proposal in [ 99 ], which is to share
fewer gradients between training participants. Although sharing less information made the attacks
less effective, it did not alleviate them completely.


**7.4** **Defenses Against Model Extraction Attacks**

Model extraction attacks usually require that the attacker performs a number of queries on the
target model. The goal of the proposed defenses so far has been the detection of these queries. This
contrasts with the previously presented defences that mainly try to prevent attacks.


A Survey of Privacy Attacks in Machine Learning 25


_7.4.1_ _Protecting against DNN Model Stealing Attacks (PRADA)._ Detecting model stealing attacks
based on the model queries that are used by the adversary was proposed by Juuti et al. [ 51 ]. The
detection is based on the assumption that model queries that try to explore decision boundaries
will have a different distribution than the normal ones. While the detection was successful, the
authors noted that it is possible to be evaded if the adversary adapts their strategy.


_7.4.2_ _Membership inference._ The idea of using membership inference to defend against model
extraction was studied by Krishna et al. [ 58 ]. It is based on the premise that using membership
inference, the model owner can distinguish between legitimate user queries and nonsensical ones
whose only purpose is to extract the model. The authors note that this type of defence has limitations
such as potentially flagging legitimate but out-of-distribution queries made by legitimate users, but
more importantly that they can be evaded by adversaries that make adaptive queries.


**8** **DISCUSSION**


Attacks on machine learning privacy have been increasingly brought to light. However, we are still
at an exploratory stage. Many of the attacks are applicable only under specific sets of assumptions
or do not scale to larger training data sets, number of classes, number of participants, etc. The
attacks will keep improving and to successfully defend against them, the community needs to
answer fundamental questions about why they are possible in the first place. While progress has
been made in the theoretical aspects of some of the attacks, there is still a long way to go to achieve
a better theoretical understanding of privacy leaks in machine learning.
As much as we need answers about why leaks happen at a theoretical level, we also need to
know how well privacy attacks work on real deployed systems. Adversarial attacks on realistic
systems bring to light the issue of additional constraints that need to be in place for the attacks to
work. When creating glasses that can fool a face recognition system, Sharif et al. [ 98 ], they had to
pose constraints that had to do with physical realizations, e.g., that the color of the glasses should
be printable. In privacy-related attacks, the most realistic cases come from the model extraction
area, where attacks against MLaaS systems have been demonstrated in multiple papers. For the
majority of other attacks, it is certainly an open question of how well they would perform on
deployed models and what kind of additional requirements need to be in place for them to succeed.
At the same time, the main research focus up to now has been supervised learning. Even within
supervised learning, there are areas and learning tasks that have been largely unexplored, and there
are few attacks reported on popular algorithms such as random forests or gradient boosting trees
despite their wide application. In unsupervised and semi-supervised learning, the focus is mainly
on generative models and only just recently, papers started exploring areas such as representation
learning and language models. Some attacks on image classifiers do not transfer that well to natural
language processing tasks [ 41 ] while others do, but may require different sets of assumptions and
design considerations [88].
Beyond expanding the focus on different learning tasks, there is the question of datasets. The
impact of datasets on the attack success has been demonstrated by several papers. Yet, currently,
we lack a common approach as to which datasets are best suited to evaluate privacy attacks, or
constitute the minimum requirement for a successful attack. Several questions are worth considering:
do we need standardized datasets and if yes, how do we go about and create them? Are all data
worth protecting and if some are more interesting than others, should we not be testing attacks
beyond popular image datasets?
Finally, as we strive to understand the privacy implications of machine learning, we also realize
that several research areas are connected and affect each other. We know, for instance, that adversarial training affects membership inference [ 100 ] and that model censoring can still leak private


26 Rigaki and Garcia


attributes [ 104 ]. Property inference attacks can deduce properties of the training dataset that were
not specifically encoded or were not necessarily correlated to the learning task. This can be understood as a form of bias detection, which means that relevant literature in the area of model fairness
should be reviewed as potentially complementary. Furthermore, while deep learning models are
considered black-boxes in terms of explainability, work that sheds light on what kind of data make
neurons activate [ 84, 126 ] can be relevant to discovering information about the training dataset
and can therefore lead to privacy leaks. All these are examples of potential inter-dependencies
between different areas of machine learning research, therefore, a better understanding of privacy
attacks calls for an interdisciplinary approach.


**9** **CONCLUSION**


As machine learning becomes ubiquitous, the scientific community becomes increasingly interested
in its impact and side-effects in terms of security, privacy, fairness, and explainability. This survey
conducted a comprehensive study of the state-of-the-art privacy-related attacks and proposed a
threat model and a unifying taxonomy of the different types of attacks based on their characteristics.
An in-depth examination of the current state of the art research allowed us to perform a detailed
analysis which revealed common design patterns and differences between them.
Several open problems that merit further research were identified. First, our analysis revealed
a somewhat narrow focus of the research conducted so far, which is dominated by attacks on
deep learning models. We believe that there are several popular algorithms and models in terms
of real-world deployment and applicability that merit a closer examination. Second, a thorough
theoretical understanding of the reasons behind privacy leaks is still underdeveloped and this
affects both the proposed defensive measures and our understanding of the limitations of privacy
attacks. Experimental studies on factors that affect privacy leaks have provided useful insights so
far. However, in total, there are very few works that test attacks in realistic conditions in terms of
dataset size and deployment. Finally, examining the impact of other adjacent study areas such as
security, explainability, and fairness is also a topic that calls for further exploration. Even though
it may not be possible to construct and deploy models that are fully private against all types of
adversaries, understanding the inter-dependencies that affect privacy will help make more informed
decisions.

While the community is still in an exploratory mode regarding privacy leaks of machine learning
systems, we hope that this survey will provide the necessary background to both the interested
readers as well as the researchers that wish to work on this topic.


**ACKNOWLEDGMENTS**

This work was partially supported by Avast Software and the OP RDE funded project Research
Center for Informatics No.: CZ.02.1.01/0.0./0.0./16_019/0000765.


**REFERENCES**


[1] Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016.
Deep Learning with Differential Privacy. In _Proceedings of the 2016 ACM SIGSAC Conference on Computer and_
_Communications Security (CCS ’16)_ [. Association for Computing Machinery, New York, NY, USA, 308–318. https:](https://doi.org/10.1145/2976749.2978318)
[//doi.org/10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318)

[2] Mohammad Al-Rubaie and Morris J. Chang. 2019. Privacy-Preserving Machine Learning: Threats and Solutions. _IEEE_
_Security & Privacy_ [17, 2 (2019), 49–58. https://doi.org/10.1109/MSEC.2018.2888775](https://doi.org/10.1109/MSEC.2018.2888775)

[3] Giuseppe Ateniese, Luigi V. Mancini, Angelo Spognardi, Antonio Villani, Domenico Vitali, and Giovanni Felici. 2015.
Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers.
_International Journal of Security and Networks_ [10, 3 (Sept. 2015), 137–150. https://doi.org/10.1504/IJSN.2015.071829](https://doi.org/10.1504/IJSN.2015.071829)

[[4] AT&T 1994. Database of Faces. Retrieved April 17, 2020 from http://cam-orl.co.uk/facedatabase.html](http://cam-orl.co.uk/facedatabase.html)


A Survey of Privacy Attacks in Machine Learning 27


[5] Michael Backes, Mathias Humbert, Jun Pang, and Yang Zhang. 2017. Walk2friends: Inferring Social Links from
Mobility Profiles. In _Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS_
_’17)_ [. Association for Computing Machinery, New York, NY, USA, 1943–1957. https://doi.org/10.1145/3133956.3133972](https://doi.org/10.1145/3133956.3133972)

[6] Antonio Barbalau, Adrian Cosma, Radu Tudor Ionescu, and Marius Popescu. 2020. Black-Box Ripper: Copying
black-box models using generative evolutionary algorithms. In _Advances in Neural Information Processing Systems_,
H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 20120–20129.
[https://proceedings.neurips.cc/paper/2020/file/e8d66338fab3727e34a9179ed8804f64-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/e8d66338fab3727e34a9179ed8804f64-Paper.pdf)

[7] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. 2019. Mixmatch:
A holistic approach to semi-supervised learning. In _Advances in Neural Information Processing Systems_ . NeurIPS,
Vancouver, Canada, 5050–5060.

[8] Battista Biggio and Fabio Roli. 2018. Wild patterns: Ten years after the rise of adversarial machine learning. _Pattern_
_Recognition_ 84 (2018), 317–331.

[9] Christopher M. Bishop. 2006. _Pattern Recognition and Machine Learning (Information Science and Statistics)_ . SpringerVerlag, Berlin, Heidelberg.

[10] Leo Breiman. 2001. Random forests. _Machine learning_ 45, 1 (2001), 5–32.

[11] Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. 2019. The Secret Sharer: Evaluating and
Testing Unintended Memorization in Neural Networks. In _28th USENIX Security Symposium (USENIX Security 19)_ .
USENIX Association, Santa Clara, CA, 267–284.

[12] Augustin Cauchy et al . 1847. Méthode générale pour la résolution des systemes d’équations simultanées. _Comp._
_Rend. Sci. Paris_ 25, 1847 (1847), 536–538.

[13] Texas Health Care Information Collection Center. 2006-2009. Texas Inpatient Public Use Data File (PUDF). Retrieved
[April 17, 2020 from https://www.dshs.texas.gov/thcic/hospitals/Inpatientpudf.shtm](https://www.dshs.texas.gov/thcic/hospitals/Inpatientpudf.shtm)

[14] Varun Chandrasekaran, Kamalika Chaudhuri, Irene Giacomelli, Somesh Jha, and Songbai Yan. 2020. Exploring
Connections Between Active Learning and Model Extraction. In _29th USENIX Security Symposium (USENIX Secu-_
_rity 20)_ . USENIX Association, Boston, MA. [https://www.usenix.org/conference/usenixsecurity20/presentation/](https://www.usenix.org/conference/usenixsecurity20/presentation/chandrasekaran)
[chandrasekaran](https://www.usenix.org/conference/usenixsecurity20/presentation/chandrasekaran)

[15] Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. 2020. GAN-Leaks: A Taxonomy of Membership Inference Attacks
against Generative Models. In _Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications_
_Security (CCS ’20)_ [. Association for Computing Machinery, New York, NY, USA, 343–362. https://doi.org/10.1145/](https://doi.org/10.1145/3372297.3417238)

[3372297.3417238](https://doi.org/10.1145/3372297.3417238)

[16] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In _Proceedings of the 22nd ACM_
_SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’16)_ . Association for Computing
[Machinery, New York, NY, USA, 785–794. https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

[17] International Warfarin Pharmacogenetics Consortium. 2009. Estimation of the warfarin dose with clinical and
pharmacogenetic data. _New England Journal of Medicine_ 360, 8 (2009), 753–764.

[18] Jackson Rodrigues Correia-Silva, Rodrigo F. Berriel, Claudine Badue, Alferto F. de Souza, and Thiago Oliveira-Santos.
2018. Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data. In _2018_
_International Joint Conference on Neural Networks (IJCNN)_ [. IEEE, Rio de Janeiro, Brazil, 1–8. https://doi.org/10.1109/](https://doi.org/10.1109/IJCNN.2018.8489592)
[IJCNN.2018.8489592](https://doi.org/10.1109/IJCNN.2018.8489592)

[19] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, Andrew
Senior, Paul Tucker, Ke Yang, et al . 2012. Large scale distributed deep networks. In _Advances in Neural Information_
_Processing Systems_ . NIPS, USA, 1223–1231.

[20] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical
Image Database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ . IEEE, Miami, FL, USA, 248–255.

[21] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. [Retrieved April 17, 2020 from http:](http://archive.ics.uci.edu/ml)
[//archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)

[22] Cynthia Dwork and Aaron Roth. 2013. The algorithmic foundations of differential privacy. _Foundations and Trends in_
_Theoretical Computer Science_ [9, 3-4 (2013), 211–487. https://doi.org/10.1561/0400000042](https://doi.org/10.1561/0400000042)

[23] Ronald A Fisher. 1936. The use of multiple measurements in taxonomic problems. _Annals of eugenics_ 7, 2 (1936),

179–188.

[24] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks That Exploit Confidence
Information and Basic Countermeasures. In _Proceedings of the 22nd ACM SIGSAC Conference on Computer and_
_Communications Security (CCS ’15)_ [. Association for Computing Machinery, New York, NY, USA, 1322–1333. https:](https://doi.org/10.1145/2810103.2813677)
[//doi.org/10.1145/2810103.2813677](https://doi.org/10.1145/2810103.2813677)

[25] Matthew Fredrikson, Eric Lantz, Somesh Jha, Simon Lin, David Page, and Thomas Ristenpart. 2014. Privacy in
Pharmacogenetics: An End-to-End Case Study of Personalized Warfarin Dosing. In _23rd USENIX Security Symposium_
_(USENIX Security 14)_ . USENIX Association, San Diego, CA, 17–32.


28 Rigaki and Garcia


[26] Karan Ganju, Qi Wang, Wei Yang, Carl A. Gunter, and Nikita Borisov. 2018. Property Inference Attacks on Fully
Connected Neural Networks Using Permutation Invariant Representations. In _Proceedings of the 2018 ACM SIGSAC_
_Conference on Computer and Communications Security (CCS ’18)_ . Association for Computing Machinery, New York,
[NY, USA, 619–633. https://doi.org/10.1145/3243734.3243834](https://doi.org/10.1145/3243734.3243834)

[27] Athinodoros S. Georghiades, Peter N. Belhumeur, and David J. Kriegman. 2001. From few to many: Illumination
cone models for face recognition under variable lighting and pose. _IEEE transactions on pattern analysis and machine_
_intelligence_ 23, 6 (2001), 643–660.

[28] Neil Zhenqiang Gong and Bin Liu. 2016. You are who you know and how you behave: Attribute inference attacks via
users’ social friends and behaviors. In _25th_ { _USENIX_ } _Security Symposium (_ { _USENIX_ } _Security 16)_ . Usenix, Austin,
TX, USA, 979–995.

[29] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. _Deep Learning_ . The MIT Press, Cambridge, MA, USA.

[30] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
and Yoshua Bengio. 2014. Generative Adversarial Nets. In _Advances in Neural Information Processing Systems 27_,
Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (Eds.). Curran Associates, Inc., Montreal,
[Canada, 2672–2680. http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[31] Gregory Griffin, Alex Holub, and Pietro Perona. 2007. _The Caltech 256_ . Technical Report. Pasadena, CA, USA.

[32] GSS. 2006. _Social science research on pornography_ . [Retrieved April 17, 2020 from https://byuresearch.org/ssrp/](https://byuresearch.org/ssrp/downloads/GSS.xls)
[downloads/GSS.xls](https://byuresearch.org/ssrp/downloads/GSS.xls)

[33] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. _ACM Trans. Interact._
_Intell. Syst._ [5, 4, Article 19 (Dec. 2015), 19 pages. https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

[34] Johann Hauswald, Thomas Manville, Qi Zheng, Ronald Dreslinski, Chaitali Chakrabarti, and Trevor Mudge. 2014. A
hybrid approach to offloading mobile image classification. In _2014 IEEE International Conference on Acoustics, Speech_
_and Signal Processing (ICASSP)_ . IEEE, Florence, Italy, 8375–8379.

[35] Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. 2019. LOGAN: Membership inference attacks
against generative models. _Proceedings on Privacy Enhancing Technologies_ 2019, 1 (2019), 133–152.

[36] Zecheng He, Tianwei Zhang, and Ruby B. Lee. 2019. Model Inversion Attacks against Collaborative Inference. In
_Proceedings of the 35th Annual Computer Security Applications Conference (ACSAC ’19)_ . Association for Computing
[Machinery, New York, NY, USA, 148–162. https://doi.org/10.1145/3359789.3359824](https://doi.org/10.1145/3359789.3359824)

[37] [Walt Hickey. 2014. DataLab: How americans like their steak. Retrieved April 17, 2020 from http://fivethirtyeight.](http://fivethirtyeight.com/datalab/how-americans-like-their-steak)
[com/datalab/how-americans-like-their-steak](http://fivethirtyeight.com/datalab/how-americans-like-their-steak)

[38] Seira Hidano, Takao Murakami, Shuichi Katsumata, Shinsaku Kiyomoto, and Goichiro Hanaoka. 2017. Model
Inversion Attacks for Prediction Systems: Without Knowledge of Non-Sensitive Attributes. In _2017 15th Annual_
_Conference on Privacy, Security and Trust (PST)_ . IEEE, Calgary, AB, Canada, 115–124.

[39] Benjamin Hilprecht, Martin Härterich, and Daniel Bernau. 2019. Monte Carlo and Reconstruction Membership
Inference Attacks against Generative Models. _Proceedings on Privacy Enhancing Technologies_ 2019, 4 (2019), 232–249.

[40] Geoffrey Hinton, Nitsh Srivastava, and Kevin Swersky. 2012. Neural networks for machine learning. _Coursera, video_
_lectures_ 264, 1 (2012).

[41] Sorami Hisamoto, Matt Post, and Kevin Duh. 2020. Membership Inference Attacks on Sequence-to-Sequence Models:
Is My Data In Your Machine Translation System? _Transactions of the Association for Computational Linguistics_ 8
[(2020), 49–63. https://doi.org/10.1162/tacl_a_00299](https://doi.org/10.1162/tacl_a_00299)

[42] Briland Hitaj, Giuseppe Ateniese, and Fernando Perez-Cruz. 2017. Deep Models Under the GAN: Information
Leakage from Collaborative Deep Learning. In _Proceedings of the 2017 ACM SIGSAC Conference on Computer and_
_Communications Security (CCS ’17)_ [. Association for Computing Machinery, New York, NY, USA, 603–618. https:](https://doi.org/10.1145/3133956.3134012)
[//doi.org/10.1145/3133956.3134012](https://doi.org/10.1145/3133956.3134012)

[43] Gary B. Huang, Marwan Mattar, Tamara Berg, and Eric Learned-Miller. 2008. Labeled Faces in the Wild: A Database
forStudying Face Recognition in Unconstrained Environments. In _Workshop on Faces in ’Real-Life’ Images: Detection,_
_Alignment, and Recognition_ . Erik Learned-Miller and Andras Ferencz and Frédéric Jurie, HAL, Marseille, France.
[https://hal.inria.fr/inria-00321923](https://hal.inria.fr/inria-00321923)

[44] International Conference on Spoken Language Translation 2015. IWSLT Evaluation 2015. Retrieved April 17, 2020
[from https://sites.google.com/site/iwsltevaluation2015](https://sites.google.com/site/iwsltevaluation2015)

[45] Matthew Jagielski, Nicholas Carlini, David Berthelot, Alex Kurakin, and Nicolas Papernot. 2020. High Accuracy and
High Fidelity Extraction of Neural Networks. In _29th USENIX Security Symposium (USENIX Security 20)_ . USENIX
[Association, Boston, MA. https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski](https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski)

[46] Bargav Jayaraman and David Evans. 2019. Evaluating Differentially Private Machine Learning in Practice. In _28th_
_USENIX Security Symposium (USENIX Security 19)_ . USENIX Association, Santa Clara, CA, 1895–1912.

[47] Malhar S. Jere, Tyler Farnan, and Farinaz Koushanfar. 2020. A Taxonomy of Attacks on Federated Learning. _IEEE_
_Security & Privacy_ [(2020), 0–0. https://doi.org/10.1109/MSEC.2020.3039941](https://doi.org/10.1109/MSEC.2020.3039941)


A Survey of Privacy Attacks in Machine Learning 29


[48] Jinyuan Jia and Neil Zhenqiang Gong. 2018. AttriGuard: A Practical Defense Against Attribute Inference Attacks
via Adversarial Machine Learning. In _27th USENIX Security Symposium (USENIX Security 18)_ . USENIX Association,
[Baltimore, MD, 513–529. https://www.usenix.org/conference/usenixsecurity18/presentation/jia-jinyuan](https://www.usenix.org/conference/usenixsecurity18/presentation/jia-jinyuan)

[49] Jinyuan Jia, Ahmed Salem, Michael Backes, Yang Zhang, and Neil Zhenqiang Gong. 2019. MemGuard: Defending
against Black-Box Membership Inference Attacks via Adversarial Examples. In _Proceedings of the 2019 ACM SIGSAC_
_Conference on Computer and Communications Security (CCS ’19)_ . Association for Computing Machinery, New York,
[NY, USA, 259–274. https://doi.org/10.1145/3319535.3363201](https://doi.org/10.1145/3319535.3363201)

[50] Alistair EW Johnson, Tom J Pollard, Lu Shen, H Lehman Li-wei, Mengling Feng, Mohammad Ghassemi, Benjamin
Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. 2016. MIMIC-III, a freely accessible critical care
database. _Scientific data_ 3 (2016), 160035.

[51] Mika Juuti, Sebastian Szyller, Samuel Marchal, and N Asokan. 2019. PRADA: protecting against DNN model stealing
attacks. In _2019 IEEE European Symposium on Security and Privacy (EuroS&P)_ . IEEE, Stockholm, Sweden, 512–527.

[52] Kaggle. 2014. _Acquire Valued Shoppers Challenge_ [. Retrieved April 17, 2020 from https://www.kaggle.com/c/acquire-](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)
[valued-shoppers-challenge/data](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)

[53] [Kaggle. 2015. Diabetic Retinopathy Detection. Retrieved April 17, 2020 from https://www.kaggle.com/c/diabetic-](https://www.kaggle.com/c/diabetic-retinopathy-detection)
[retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

[54] Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, and Lingjia Tang. 2017.
Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. In _Proceedings of the Twenty-Second_
_International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS ’17)_ .
[Association for Computing Machinery, New York, NY, USA, 615–629. https://doi.org/10.1145/3037697.3037698](https://doi.org/10.1145/3037697.3037698)

[55] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017.
LightGBM: A Highly Efficient Gradient Boosting Decision Tree. _Advances in Neural Information Processing Systems_
30 (2017), 3146–3154.

[56] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_
(2014).

[57] Diederik P Kingma and Max Welling. 2014. Auto-encoding variational bayes. In _2nd International Conference on_
_Learning Representations, ICLR 2014_, Vol. 1. ICLR, Banff, Canada.

[58] Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, and Mohit Iyyer. 2020. Thieves on Sesame
Street! Model Extraction of BERT-based APIs. In _International Conference on Learning Representations_ . ICLR, Virtual
[Conference, formerly Addis Ababa, Ethiopia. https://openreview.net/forum?id=Byl5NREFDr](https://openreview.net/forum?id=Byl5NREFDr)

[59] Alex Krizhevsky, Geoffrey Hinton, et al. 2009. Learning multiple layers of features from tiny images. (2009).

[60] Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. 1998. The MNIST database of handwritten digits.
[Retrieved April 17, 2020 from http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

[61] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. 2019. Federated learning: Challenges, methods, and
[future directions. (2019). arXiv:arXiv:1908.07873](https://arxiv.org/abs/arXiv:1908.07873)

[62] Dong C Liu and Jorge Nocedal. 1989. On the limited memory BFGS method for large scale optimization. _Mathematical_
_programming_ 45, 1-3 (1989), 503–528.

[63] Jian Liu, Mika Juuti, Yao Lu, and N. Asokan. 2017. Oblivious Neural Network Predictions via MiniONN Transformations. In _Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS ’17)_ .
[Association for Computing Machinery, New York, NY, USA, 619–631. https://doi.org/10.1145/3133956.3134056](https://doi.org/10.1145/3133956.3134056)

[64] Shuying Liu and Weihong Deng. 2015. Very deep convolutional neural network based image classification using
small training sample size. In _2015 3rd IAPR Asian conference on pattern recognition (ACPR)_ . IEEE, 730–734.

[65] Ximeng Liu, Lehui Xie, Yaopeng Wang, Jian Zou, Jinbo Xiong, Zuobin Ying, and Athanasios V. Vasilakos. 2021.
Privacy and Security Issues in Deep Learning: A Survey. _IEEE Access_ [9 (2021), 4566–4593. https://doi.org/10.1109/](https://doi.org/10.1109/ACCESS.2020.3045078)

[ACCESS.2020.3045078](https://doi.org/10.1109/ACCESS.2020.3045078)

[66] Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang, Michael Backes, Emiliano De Cristofaro, Mario Fritz,
and Yang Zhang. 2021. ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models.
_arXiv preprint arXiv:2102.02551_ (2021).

[67] Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A Gunter, and Kai Chen.
2018. Understanding membership inferences on well-generalized learning models. _arXiv preprint arXiv:1802.04889_
(2018).

[68] Minh-Thang Luong and Christopher D. Manning. 2015. Stanford Neural Machine Translation Systems for Spoken
Language Domain. In _International Workshop on Spoken Language Translation_ . Da Nang, Vietnam.

[69] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. Towards Deep
Learning Models Resistant to Adversarial Attacks. In _International Conference on Learning Representations_ . ICLR,
Vancouver, Canada.


30 Rigaki and Garcia


[70] [Matt Mahoney. [n.d.]. Large Text Compression Benchmark. Retrieved March 8, 2021 from http://mattmahoney.net/](http://mattmahoney.net/dc/text.html)
[dc/text.html](http://mattmahoney.net/dc/text.html)

[71] Davide Maiorca, Battista Biggio, and Giorgio Giacinto. 2019. Towards adversarial malware detection: Lessons learned
from PDF-based attacks. _ACM Computing Surveys (CSUR)_ 52, 4 (2019), 1–36.

[72] Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. 1993. Building a Large Annotated Corpus of
English: The Penn Treebank. _Comput. Linguist._ 19, 2 (June 1993), 313–330.

[73] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. CommunicationEfficient Learning of Deep Networks from Decentralized Data. In _Proceedings of the 20th International Conference_
_on Artificial Intelligence and Statistics (Proceedings of Machine Learning Research)_, Aarti Singh and Jerry Zhu (Eds.),
[Vol. 54. PMLR, Fort Lauderdale, FL, USA, 1273–1282. http://proceedings.mlr.press/v54/mcmahan17a.html](http://proceedings.mlr.press/v54/mcmahan17a.html)

[74] Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang. 2018. Learning Differentially Private Recurrent
Language Models. In _International Conference on Learning Representations_ . ICLR, Vancouver, Canada. [https://](https://openreview.net/forum?id=BJ0hF1Z0b)
[openreview.net/forum?id=BJ0hF1Z0b](https://openreview.net/forum?id=BJ0hF1Z0b)

[75] Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. 2019. Exploiting unintended feature
leakage in collaborative learning. In _2019 IEEE Symposium on Security and Privacy (SP)_ . IEEE, San Francisco, CA, USA,

691–706.

[76] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. (2016).

[arXiv:arXiv:1609.07843](https://arxiv.org/abs/arXiv:1609.07843)

[77] Smitha Milli, Ludwig Schmidt, Anca D. Dragan, and Moritz Hardt. 2019. Model Reconstruction from Model Explanations. In _Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* ’19)_ . Association for
[Computing Machinery, New York, NY, USA, 1–9. https://doi.org/10.1145/3287560.3287562](https://doi.org/10.1145/3287560.3287562)

[78] Kevin P Murphy. 2012. _Machine learning: a probabilistic perspective_ . MIT press, Cambridge, MA, USA.

[79] Milad Nasr, Reza Shokri, and Amir Houmansadr. 2018. Machine learning with membership privacy using adversarial
regularization. In _Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security_ . 634–646.

[80] Milad Nasr, Reza Shokri, and Amir Houmansadr. 2019. Comprehensive privacy analysis of deep learning: Passive and
active white-box inference attacks against centralized and federated learning. In _2019 IEEE Symposium on Security_
_and Privacy (SP)_ . IEEE, San Francisco, CA, USA, 739–753.

[[81] Netflix 2009. Netflix prize. Retrieved April 17, 2020 from https://www.netflixprize.com](https://www.netflixprize.com)

[82] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. 2011. Reading Digits in
Natural Images with Unsupervised Feature Learning. In _NIPS Workshop on Deep Learning and Unsupervised Feature_
_Learning 2011_ [. NIPS, Granada, Spain. http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)

[83] Hong-Wei Ng and Stefan Winkler. 2014. A data-driven approach to cleaning large face datasets. In _2014 IEEE_
_international conference on image processing (ICIP)_ . IEEE, Paris, France, 343–347.

[84] Anh Nguyen, Alexey Dosovitskiy, Jason Yosinski, Thomas Brox, and Jeff Clune. 2016. Synthesizing the Preferred Inputs
for Neurons in Neural Networks via Deep Generator Networks. In _Proceedings of the 30th International Conference on_
_Neural Information Processing Systems (NIPS’16)_ . Curran Associates Inc., Red Hook, NY, USA, 3395–3403.

[85] Jorge Nocedal and Stephen J Wright. 2006. _Numerical Optimization_ . Springer, Berlin, Heidelberg.

[86] Seong Joon Oh, Max Augustin, Mario Fritz, and Bernt Schiele. 2018. Towards Reverse-Engineering Black-Box
Neural Networks. In _Sixth International Conference on Learning Representations_ [. ICLR, Vancouver, Canada. https:](https://openreview.net/forum?id=BydjJte0-)
[//openreview.net/forum?id=BydjJte0-](https://openreview.net/forum?id=BydjJte0-)

[87] Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. 2019. Knockoff nets: Stealing functionality of black-box
models. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ . IEEE, Long Beach, CA,
USA, 4954–4963.

[88] Xudong Pan, Mi Zhang, Shouling Ji, and Min Yang. 2020. Privacy Risks of General-Purpose Language Models.. In
_2020 IEEE Symposium on Security and Privacy (SP)_ . IEEE, San Francisco, CA, USA, 1314–1331.

[89] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. 2017.
Practical Black-Box Attacks against Machine Learning. In _Proceedings of the 2017 ACM on Asia Conference on Computer_
_and Communications Security (ASIA CCS ’17)_ . Association for Computing Machinery, New York, NY, USA, 506–519.
[https://doi.org/10.1145/3052973.3053009](https://doi.org/10.1145/3052973.3053009)

[90] Nicholas Papernot, Patrick McDaniel, Arunesh Sinha, and Michael P. Wellman. 2018. SoK: Security and Privacy in
Machine Learning. In _2018 IEEE European Symposium on Security and Privacy (EuroS P)_ . IEEE, London, UK, 399–414.

[91] Boris T Polyak. 1964. Some methods of speeding up the convergence of iteration methods. _Ussr computational_
_mathematics and mathematical physics_ 4, 5 (1964), 1–17.

[92] Md Atiqur Rahman, Tanzila Rahman, Robert Laganière, Noman Mohammed, and Yang Wang. 2018. Membership
Inference Attack against Differentially Private Deep Learning Model. _Transactions on Data Privacy_ 11, 1 (2018),

61–79.


A Survey of Privacy Attacks in Machine Learning 31


[93] Herbert Robbins and Sutton Monro. 1951. A stochastic approximation method. _The annals of mathematical statistics_
(1951), 400–407.

[94] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Herve Jegou. 2019. White-box vs
Black-box: Bayes Optimal Strategies for Membership Inference. In _Proceedings of the 36th International Conference on_
_Machine Learning (Proceedings of Machine Learning Research)_, Kamalika Chaudhuri and Ruslan Salakhutdinov (Eds.),
[Vol. 97. PMLR, Long Beach, California, USA, 5558–5567. http://proceedings.mlr.press/v97/sablayrolles19a.html](http://proceedings.mlr.press/v97/sablayrolles19a.html)

[95] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. 2019. ML-Leaks:
Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models. In _26th_
_Annual Network and Distributed System Security Symposium, NDSS_ . NDSS, San Diego, California, USA.

[96] Todd E Scheetz, Kwang-Youn A Kim, Ruth E Swiderski, Alisdair R Philp, Terry A Braun, Kevin L Knudtson, Anne M
Dorrance, Gerald F DiBona, Jian Huang, Thomas L Casavant, et al . 2006. Regulation of gene expression in the
mammalian eye and its relevance to eye disease. _Proceedings of the National Academy of Sciences_ 103, 39 (2006),

14429–14434.

[97] Shai Shalev-Shwartz and Shai Ben-David. 2014. _Understanding Machine Learning: From Theory to Algorithms_ .
Cambridge University Press, USA.

[98] Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K. Reiter. 2016. Accessorize to a Crime: Real and
Stealthy Attacks on State-of-the-Art Face Recognition. In _Proceedings of the 2016 ACM SIGSAC Conference on Computer_
_and Communications Security (CCS ’16)_ . Association for Computing Machinery, New York, NY, USA, 1528–1540.
[https://doi.org/10.1145/2976749.2978392](https://doi.org/10.1145/2976749.2978392)

[99] Reza Shokri and Vitaly Shmatikov. 2015. Privacy-Preserving Deep Learning. In _Proceedings of the 22nd ACM SIGSAC_
_Conference on Computer and Communications Security (CCS ’15)_ . Association for Computing Machinery, New York,
[NY, USA, 1310–1321. https://doi.org/10.1145/2810103.2813687](https://doi.org/10.1145/2810103.2813687)

[100] Reza Shokri, Martin Strobel, and Yair Zick. 2019. Privacy risks of explaining machine learning models. (2019).

[arXiv:arXiv:1907.00164](https://arxiv.org/abs/arXiv:1907.00164)

[101] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Membership inference attacks against
machine learning models. In _2017 IEEE Symposium on Security and Privacy (SP)_ . IEEE, San Francisco, CA, USA, 3–18.

[102] Congzheng Song and Ananth Raghunathan. 2020. Information Leakage in Embedding Models. In _Proceedings of_
_the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS ’20)_ . Association for Computing
[Machinery, New York, NY, USA, 377–390. https://doi.org/10.1145/3372297.3417270](https://doi.org/10.1145/3372297.3417270)

[103] Congzheng Song and Vitaly Shmatikov. 2019. Auditing Data Provenance in Text-Generation Models. In _Proceedings_
_of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19)_ . Association for
[Computing Machinery, New York, NY, USA, 196–206. https://doi.org/10.1145/3292500.3330885](https://doi.org/10.1145/3292500.3330885)

[104] Congzheng Song and Vitaly Shmatikov. 2020. Overlearning Reveals Sensitive Attributes. In _International Conference_
_on Learning Representations_ . ICLR, virtual conference.

[105] Liwei Song, Reza Shokri, and Prateek Mittal. 2019. Privacy Risks of Securing Machine Learning Models against
Adversarial Examples. In _Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security_
_(CCS ’19)_ [. Association for Computing Machinery, New York, NY, USA, 241–257. https://doi.org/10.1145/3319535.](https://doi.org/10.1145/3319535.3354211)

[3354211](https://doi.org/10.1145/3319535.3354211)

[106] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout:
a simple way to prevent neural networks from overfitting. _The journal of machine learning research_ 15, 1 (2014),

1929–1958.

[107] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. 2011. The German traffic sign recognition
benchmark: a multi-class classification competition. In _The 2011 international joint conference on neural networks_ .
IEEE, San Jose, CA, USA, 1453–1460.

[108] Richard S Sutton and Andrew G Barto. 2018. _Reinforcement learning: An introduction_ . MIT press, Cambridge, MA.

[109] Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, and Thomas Ristenpart. 2016. Stealing Machine Learning
Models via Prediction APIs. In _25th USENIX Security Symposium (USENIX Security 16)_ . USENIX Association, Austin,
TX, 601–618.

[110] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. 2019. Demystifying Membership Inference
Attacks in Machine Learning as a Service. _IEEE Transactions on Services Computing_ [-, - (2019), 1–1. https://doi.org/10.](https://doi.org/10.1109/TSC.2019.2897554)
[1109/TSC.2019.2897554](https://doi.org/10.1109/TSC.2019.2897554)

[111] Vladimir Vapnik. 1992. Principles of risk minimization for learning theory. In _Advances in neural information_
_processing systems_ . 831–838.

[112] Michael Veale, Reuben Binns, and Lilian Edwards. 2018. Algorithms that remember: model inversion attacks and
data protection law. _Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences_
376, 2133 (2018), 20180083.


32 Rigaki and Garcia


[113] Ben Verhoeven and Walter Daelemans. 2014. CLiPS Stylometry Investigation (CSI) corpus: a Dutch corpus for
the detection of age, gender, personality, sentiment and deception in text. In _Proceedings of the Ninth International_
_Conference on Language Resources and Evaluation (LREC-2014)_ . European Language Resources Association (ELRA),
Reykjavik, Iceland, 3081–3085.

[[114] VoxForge 2009. VoxForge Speech Corpus. Retrieved April 17, 2020 from http://www.voxforge.org/](http://www.voxforge.org/)

[115] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. 2011. _The caltech UCSD birds-200-_
_2011 dataset_ . Technical Report. Pasadena, CA, USA.

[116] Binghui Wang and Neil Zhenqiang Gong. 2018. Stealing hyperparameters in machine learning. In _2018 IEEE Symposium_
_on Security and Privacy (SP)_ . IEEE, San Francisco, CA, USA, 36–52.

[117] Xianmin Wang, Jing Li, Xiaohui Kuang, Yu-an Tan, and Jin Li. 2019. The security of machine learning in an adversarial
setting: A survey. _J. Parallel and Distrib. Comput._ 130 (2019), 12–23.

[118] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. 2017. ChestX-ray8:
Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
Common Thorax Diseases. In _The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ . IEEE, Honolulu,
HI, USA.

[119] Zhibo Wang, Mengkai Song, Zhifei Zhang, Yang Song, Qian Wang, and Hairong Qi. 2019. Beyond Inferring Class
Representatives: User-Level Privacy Leakage From Federated Learning. In _IEEE INFOCOM 2019 - IEEE Conference on_
_Computer Communications_ . IEEE, Paris, France, 2512–2520.

[120] Xi Wu, Matthew Fredrikson, Somesh Jha, and Jeffrey F Naughton. 2016. A methodology for formalizing modelinversion attacks. In _2016 IEEE 29th Computer Security Foundations Symposium (CSF)_ . IEEE, Lisbon, Portugal, 355–370.

[121] Han Xiao, Kashif Rasul, and Roland Vollgraf. 2017. Fashion-mnist: a novel image dataset for benchmarking machine
[learning algorithms. (2017). arXiv:arXiv:1708.07747](https://arxiv.org/abs/arXiv:1708.07747)

[122] Dingqi Yang, Daqing Zhang, Longbiao Chen, and Bingqing Qu. 2015. Nationtelescope: Monitoring and visualizing
large-scale collective behavior in lbsns. _Journal of Network and Computer Applications_ 55 (2015), 170–180.

[123] Ziqi Yang, Jiyi Zhang, Ee-Chien Chang, and Zhenkai Liang. 2019. Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment. In _Proceedings of the 2019 ACM SIGSAC Conference on Computer_
_and Communications Security (CCS ’19)_ . Association for Computing Machinery, New York, NY, USA, 225–240.
[https://doi.org/10.1145/3319535.3354261](https://doi.org/10.1145/3319535.3354261)

[[124] Yelp. [n.d.]. Yelp Open Dataset. Retrieved April 17, 2020 from https://www.yelp.com/dataset](https://www.yelp.com/dataset)

[125] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018. Privacy risk in machine learning: Analyzing
the connection to overfitting. In _2018 IEEE 31st Computer Security Foundations Symposium (CSF)_ . IEEE, Oxford, UK,

268–282.

[126] Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. 2015. Understanding neural networks
through deep visualization. _arXiv preprint arXiv:1506.06579_ (2015).

[127] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. 2019. S4l: Self-supervised semi-supervised
learning. In _Proceedings of the IEEE international conference on computer vision_ . IEEE, Seoul, Korea (South), 1476–1485.

[128] Ning Zhang, Manohar Paluri, Yaniv Taigman, Rob Fergus, and Lubomir Bourdev. 2015. Beyond frontal faces:
Improving person recognition using multiple cues. In _Proceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition_ . IEEE, Boston, MA, USA, 4804–4813.

[129] Yuheng Zhang, Ruoxi Jia, Hengzhi Pei, Wenxiao Wang, Bo Li, and Dawn Song. 2020. The secret revealer: generative
model-inversion attacks against deep neural networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_ . IEEE, 253–261.

[130] Ligeng Zhu, Zhijian Liu, and Song Han. 2019. Deep Leakage from Gradients. In _Advances in Neural Information_
_Processing Systems 32_, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.). Curran
Associates, Inc., Vancouver, Canada, 14747–14756.

[131] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler.
2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In
_Proceedings of the IEEE international conference on computer vision_ . 19–27.


