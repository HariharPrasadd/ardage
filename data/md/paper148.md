# Detecting Deceptive Reviews using Generative Adversarial Networks

Hojjat Aghakhani [1], Aravind Machiry [1], Shirin Nilizadeh [2], Christopher Kruegel [1], and Giovanni Vigna [1]


1 University of California, Santa Barbara
1 {hojjat, machiry, chris, vigna}@cs.ucsb.edu
2 Carnegie Mellon University Silicon Valley
2 shirin.nilizadeh@sv.cmu.edu



_**Abstract**_ **—In the past few years, consumer review sites have**
**become the main target of** _**deceptive opinion spam**_ **, where fictitious**
**opinions or reviews are deliberately written to sound authentic.**
**Most of the existing work to detect the deceptive reviews focus**
**on building supervised classifiers based on syntactic and lexical**
**patterns of an opinion. With the successful use of Neural**
**Networks on various classification applications, in this paper, we**
**propose FakeGAN a system that for the first time augments**
**and adopts Generative Adversarial Networks (GANs) for a text**
**classification task, in particular, detecting deceptive reviews.**
**Unlike standard GAN models which have a single Generator**
**and Discriminator model, FakeGAN uses two discriminator**
**models and one generative model. The generator is modeled**
**as a stochastic policy agent in reinforcement learning (RL),**
**and the discriminators use Monte Carlo search algorithm to**
**estimate and pass the intermediate action-value as the RL**
**reward to the generator. Providing the generator model with two**
**discriminator models avoids the mod collapse issue by learning**
**from both distributions of truthful and deceptive reviews. Indeed,**
**our experiments show that using two discriminators provides**
**FakeGAN high stability, which is a known issue for GAN**
**architectures. While FakeGAN is built upon a** _**semi-supervised**_
_**classifier**_ **, known for less accuracy, our evaluation results on a**
**dataset of TripAdvisor hotel reviews show the same performance**
**in terms of accuracy as of the state-of-the-art approaches that**
**apply supervised machine learning. These results indicate that**
**GANs can be effective for text classification tasks. Specifically,**
**FakeGAN is effective at detecting deceptive reviews.**


I. I NTRODUCTION


In the current world, we habitually turn to the wisdom of
our peers, and often complete strangers, for advice, instead of
merely taking the word of an advertiser or business owner. A
2015 study by marketing research company Mintel [ 1 ] found
nearly 70 percent of Americans seek out others’ opinions online
before making a purchase. Many platforms such as Yelp.com
and TripAdvisor.com have sprung up to facilitate this sharing of
ideas amongst users. The heavy reliance on review information
by the users has dramatic effects on business owners. It has been
shown that an extra half-star rating on Yelp helps restaurants
to sell out 19 percentage points more frequently [2].
This phenomenon has also lead to a market for various kinds
of fraud. In simple cases, this could be a business rewarding
its customers with a discount, or outright paying them, to
write a favorable review. In more complex cases, this could
involve astroturfing, opinion spamming [ 3 ] or _deceptive opinion_



_spamming_ [ 4 ], where fictitious reviews are deliberately written
to sound authentic. Figure 1 shows an example of a truthful
and deceptive review written for the same hotel. It is estimated
that up to 25% of Yelp reviews are fraudulent [5], [6].
Detecting deceptive reviews is a text classification problem.
In recent years, deep learning techniques based on natural
language processing have been shown to be successful for
text classification tasks. Recursive Neural Network (RecursiveNN) [ 7 ], [ 8 ], [ 9 ] has shown good performance classifying
texts, while Recurrent Neural Network (RecurrentNN) [ 10 ]
better captures the contextual information and is ideal for
realizing semantics of long texts. However, RecurrentNN is a
biased model, where later words in a text have more influence
than earlier words [ 11 ]. This is not suitable for tasks such
as detection of deceptive reviews that depend on an unbiased
semantics of the entire document (review). Recently, techniques
based on Convolutional Neural Network (CNN) [ 12 ], [ 13 ]
were shown to be effective for text classification. However, the
effectiveness of these techniques depends on careful selection
of the window size [11], which controls the parameter space.
Moreover, in general, the main problem with applying
classification methods for detecting deceptive reviews is the
lack of substantial ground truth datasets required for most of the
supervised machine learning techniques. This problem worsens
for neural networks based methods, whose complexity requires
much bigger dataset to reach a reasonable performance.
To address the limitations of the existing techniques, we
propose FakeGAN, which is a technique based on Generative
Adversarial Network (GAN) [ 14 ]. GANs are a class of artificial
intelligence algorithms used in unsupervised machine learning,
implemented by a system of two neural networks contesting
with each other in a zero-sum game framework. GANs have
been used mostly for image-based applications [ 14 ], [ 15 ], [ 16 ],

[ 17 ]. In this paper, for the first time, we propose the use of
GANs for a text classification task, i.e., detecting deceptive
reviews. Moreover, the use of a semi-supervised learning
method like GAN can eliminate the problem of ground truth
scarcity that in general hinders the detection success [ 4 ], [ 18 ],

[19].
We augment GAN models for our application in such a way
that unlike standard GAN models which have a single Generator
and Discriminator model, FakeGAN uses two discriminator


models _D_, _D_ _[′]_ and one generative model _G_ . The discriminator
model _D_ tries to distinguish between truthful and deceptive
reviews whereas _D_ _[′]_ tries to distinguish between reviews
generated by the generative model _G_ and samples from
_deceptive_ reviews distribution. The discriminator model _D_ _[′]_

helps _G_ to generate reviews close to the deceptive reviews
distribution, while _D_ helps _G_ to generate reviews which are
classified by _D_ as truthful.
Our intuition behind using two discriminators is to create a
stronger generator model. If in the adversarial learning phase,
the generator gets rewards only from _D_, the GAN may face
the mod collapse issue [ 20 ], as it tries to learn two different
distributions (truthful and deceptive reviews). The combination
of _D_ and _D_ _[′]_ trains _G_ to generate better deceptive reviews
which in turn train _D_ to be a better discriminator.
Indeed, our evaluation using the TripAdvisor [1] hotel reviews
dataset shows that the discriminator _D_ generated by FakeGAN
performs on par with the state-of-the-art methods that apply
supervised machine learning, with an accuracy of 89.1%. These
results indicate that GANs can be effective for text classification
tasks, specifically, FakeGAN is effective at detecting deceptive
reviews. To the best of our knowledge, FakeGAN is the first
work that use GAN to generate better discriminator model (i.e.,
_D_ ) in contrast to the common GAN applications which aim
to improve the generator model.
In summary, following are our contributions:

1) We propose FakeGAN, a deceptive review detection
system based on a double discriminator GAN.
2) We believe that FakeGAN demonstrates a good first step
towards using GANs for text classification tasks.
3) To the best of our knowledge, FakeGAN is the first system using semi-supervised neural network-based learning
methods for detecting deceptive fraudulent reviews.
4) Our evaluation results demonstrate that FakeGAN is
as effective as the state-of-the-art methods that apply
supervised machine learning for detecting deceptive
reviews.


II. A PPROACH


Generative Adversarial Network (GAN) [ 14 ] is a promising
framework for generating high-quality samples with the same
distribution as the target dataset. FakeGAN leverages GAN to
learn the distributions of truthful and deceptive reviews and
to build a semi-supervised classifier using the corresponding
distributions.

A GAN consists of two models: a generative model _G_
which tries to capture the data distribution, and a discriminative
model _D_ that distinguishes between samples coming from the
training data or the generator _G_ . These two models are trained
simultaneously, where _G_ is trying to fool the discriminator _D_,
while _D_ is maximizing its probability estimation that whether
a sample comes from the training data or is produced by
the generator. In a nutshell, this framework corresponds to a
minimax two-player game.


1 Tripadvisor.com



The feedback or the gradient update from discriminator
model plays a vital role in the effectiveness of a GAN. In
the case of text generation, it is difficult to pass the gradient
update because the generative model produces discrete tokens
(words), but the discriminative model makes a decision for
complete sequence or sentence. Inspired by SeqGAN [ 21 ] that
uses GAN model for Chinese poem generation, in this work,
we model the generator as a stochastic policy in reinforcement
learning (RL), where the gradient update or RL reward signal
is provided by the discriminator using Monte Carlo search.
Monte Carlo is a heuristic search algorithm for identifying the
most promising moves in a game. In summary, in each state
of the game, it plays out the game to the very end for a fixed
number of times according to a given policy. To find the most
promising move, it must be provided by reward signals for a
complete sequence of moves.
All the existing applications use GAN to create a strong
generator, where the main issue is the convergence of generator
model [ 22 ], [ 23 ], [ 20 ]. _Mode collapse_ in particular is a known
problem in GANs, where complexity and multimodality of the
input distribution cause the generator to produce samples from a
single mode. The generator may switch between modes during
the learning phase, and this cat-and-mouse game may never
end [ 24 ], [ 20 ]. Although no formal proof exists for convergence,
in Section III we show that the FakeGAN’s discriminator

converges in practice.
Unlike the typical applications of GANs, where the ultimate
goal is to have a strong generator, FakeGAN leverages GAN to
create a well-trained discriminator, so that it can successfully
distinguish truthful and deceptive reviews. However, to avoid
the stability issues inherent to GANs we augment our network
to have two discriminator models though we use only one
of them as our intended classifier. Note that leveraging
samples generated by the generator makes our classifier a
_semi-supervised_ classifier.


_Definitions_


We start with defining certain symbols which will be used
throughout this section to define various steps of our approach.
The training dataset, _X_ = _X_ _D_ _∪_ _X_ _T_, consists of two parts,
deceptive reviews _X_ _D_ and truthful reviews _X_ _T_ . We use _χ_ to
denote the vocabulary of all tokens (i.e., words) which are
available in _X_ .

Our generator model _G_ _α_ parametrized by _α_ produces each
review _S_ 1: _L_ as a sequence of tokens of length _L_ where _S_ 1: _L_ _∈_
_χ_ _[L]_ . We use _Z_ _G_ to indicate all the reviews generated by our
generator model _G_ _α_ .
We use two discriminator models _D_ and _D_ _[′]_ . The discrimi
nator _D_ distinguishes between truthful and deceptive reviews,
as such _D_ ( _S_ 1: _L_ ) is the probability that the sequence of tokens
comes from _X_ _T_ or _X_ _D_ _∪_ _Z_ _G_ . Similarly, _D_ _[′]_ distinguishes
between deceptive samples in the dataset and samples generated
by _G_ _α_ consequently _D_ _[′]_ ( _S_ 1: _L_ ) is a probability indicating how
likely the sequence of tokens comes from _X_ _D_ or _Z_ _G_ .
The discriminator _D_ _[′]_ guides the generator _G_ _α_ to produce
samples similar to _X_ _D_ whereas _D_ guides _G_ _α_ to generate



2


(a) A truthful review provided by a high profile user on TripAdvisor (b) A deceptive review written by an Amazon Mechanical worker


Fig. 1: A truthful review versus a deceptive review, both written for the same hotel.


corresponding action-value function is:


_A_ _G_ _α_ _,D,D_ _′_ ( _a_ = _S_ _L_ _, s_ = _S_ 1: _L−_ 1 ) = _D_ ( _S_ 1: _L_ ) + _D_ _[′]_ ( _S_ 1: _L_ ) (1)


As mentioned before, _G_ _α_ produces a review token by token.
However, the discriminators provide the reward for a complete
sequence. Moreover, _G_ _α_ should care about the long-term
reward, similar to playing Chess where players sometimes
prefer to give up immediate good moves for a long-term goal
of victory [ 25 ]. Therefore, to estimate the action-value function
in every timestep _t_, we apply the Monte Carlo search _N_ times
with a roll-out policy _G_ _[′]_ _γ_ [to sample the undetermined last] _[ L]_ _[−]_ _[t]_
tokens. We define an _N_ -time Monte Carlo search as


_{S_ 1: [1] _L_ _[, S]_ 1: [2] _L_ _[, ..., S]_ 1: _[N]_ _L_ _[}]_ [ =] _[ MC]_ _[G]_ _[′]_ _γ_ [(] _[S]_ [1:] _[t]_ _[, N]_ [)] (2)


where for 1 _≤_ _i ≤_ _N_


_S_ 1: _[i]_ _t_ [= (] _[S]_ [1] _[, ..., S]_ _[t]_ [)] (3)



Fig. 2: The overview of FakeGAN. The symbols + and _−_
indicates positive and negative samples respectively. Note that,
these are different from truthful and deceptive reviews.


samples which seems truthful to _D_ . So in each round of training,
by using the feedback from _D_ and _D_ _[′]_, the generator _G_ _α_ tries
to fool _D_ _[′]_ and _D_ by generating reviews that seems deceptive
(not generated by _G_ _α_ ) to _D_ _[′]_, and truthful (not generated by
_G_ _α_ or comes from _X_ _D_ ) to _D_ .

Figure 2 shows an overview of FakeGAN. During pretraining, we use the Maximum Likelihood Estimation (MLE)
to train the generator _G_ _α_ on deceptive reviews _X_ _D_ from the
training dataset. We also use minimizing the cross-entropy
technique to pre-train the discriminators.


The generator _G_ _α_ is defined as a policy model in reinforcement learning. In timestep _t_, the state _s_ is the sequence of
produced tokens, and the action _a_ is the next token. The policy
model _G_ _α_ ( _S_ _t_ _|S_ 1: _t−_ 1 ) is stochastic. Furthermore, the generator
_G_ _α_ is trained by using a policy gradient and Monte Carlo (MC)
search on the expected end reward from the discriminative
models _D_ and _D_ _[′]_ . Similar to [ 21 ], we consider the estimated
probability _D_ ( _S_ 1: _L_ ) + _D_ _[′]_ ( _S_ 1: _L_ ) as the reward. Formally, the



and _S_ _t_ _[i]_ +1: _L_ [is sampled via roll-out policy] _[ G]_ _[′]_ _γ_ [based on the]
current state _S_ 1: _[i]_ _t−_ 1 [. The complexity of action-value estimation]
function mainly depends on the roll-out policy. While one
might use a simple version (e.g., random sampling or sampling
based on n-gram features) as the policy to train the GAN
fast, to be more efficient, we use the same generative model
( _G_ _[′]_ _γ_ [=] _[ G]_ _[α]_ [at time] _[ t]_ [). Note that, a higher value of] _[ N]_ [ results]
in less variance and more accurate evaluation of the action
value function. We can now define the action-value estimation
function at _t_ as


_A_ _G_ _α_ _,D,D_ _′_ ( _a_ = _S_ _t_ _, s_ = _S_ 1: _t−_ 1 ) =
_N_ 1 � _Ni_ =1 [(] _[D]_ [(] _[S]_ 1: _[i]_ _L_ [) +] _[ D]_ _[′]_ [(] _[S]_ 1: _[i]_ _L_ [))] if _t ≤_ _L_ (4)
� _D_ ( _S_ 1: _L_ ) + _D_ _[′]_ ( _S_ 1: _L_ ) if _t_ = _L_


where _S_ 1: _[i]_ _L_ [s are created according to the Equation 2. As there]
is no intermediate reward for the generator, we define the the
objective function for the generator _G_ _α_ (based on [ 26 ]) to
produce a sequence from the start state _S_ 0 to maximize its
final reward:


_J_ ( _α_ ) = � _G_ _α_ ( _S_ 1 _|S_ 0 ) _. A_ _G_ _α_ _,D,D_ _′_ ( _a_ = _S_ 1 _, s_ = _S_ 0 ) (5)

_S_ 1 _∈χ_


Conseqently, the gradient of the objective function _J_ ( _α_ ) is:



� _∇_ _α_ _G_ _α_ ( _S_ _t_ _|S_ 1: _t−_ 1 ) _. A_ _G_ _α_ _,D,D_ _′_ ( _a_ = _S_ _t_ _, s_ = _S_ 1: _t−_ 1 )] (6)

_S_ _t_ _∈χ_



_∇_ _α_ _J_ ( _α_ ) =



_T_
�



� E _S_ 1: _t−_ 1 _∼G_ _α_ [ �

_t_ =1 _S_ _∈_



3


We update the generator’s parameters ( _α_ ) as:


_α ←_ _α_ + _λ∇_ _α_ _J_ ( _α_ ) (7)


where _λ_ is the learning rate.
By dynamically updating the discriminative models, we can
further improve the generator. So, after generating _g_ samples,
we will re-train the discriminative models _D_ and _D_ _[′]_ for _d_

steps using the following objective functions respectively:


_min_ ( _−_ E _S∼X_ _T_ [log _D_ ( _S_ )] _−_ E _S∼X_ _D_ _∨G_ _α_ [1 _−_ log _D_ ( _S_ )]) (8)


_min_ ( _−_ E _S∼X_ _D_ [log _D_ _[′]_ ( _S_ )] _−_ E _S∼G_ _α_ [1 _−_ log _D_ _[′]_ ( _S_ )]) (9)


In each of the _d_ steps, we use _G_ _α_ to generate the same number
of samples as number of truthful reviews i.e., _|X_ _G_ _|_ = _|X_ _T_ _|_ .
The updated discriminators will be used to update the generator,
and this cycle continues until FakeGAN converges. Algorithm 1
formally defines all the above steps.


**Algorithm 1** FakeGAN


**Require:** discriminators _D_ and _D_ _[′]_, generator _G_ _α_, roll-out
policy _G_ _γ_, dataset _X_
Initialize _α_ with random weight.
Load word2vec vector embeddings into _G_ _α_, _D_ and _D_ _[′]_

models

Pre-train _G_ _α_ using MLE on _X_ _D_
Pre-train _D_ by minimizing the cross entropy
Generate negative examples by _G_ _α_ for training _D_ _[′]_

Pre-train _D_ _[′]_ by minimizing the cross entropy

_γ ←_ _α_
**repeat**

**for** g-steps **do**

Generate a sequence of tokens _S_ 1: _L_ = ( _S_ 1 _, ..., S_ _L_ ) _∼_
_G_ _α_
**for** _t_ in 1 : _L_ **do**

Compute _A_ _G_ _α_ _,D_ _β_ _,D_ _θ_ _[′]_ [(] _[a]_ [ =] _[ S]_ _[t]_ _[, s]_ [ =] _[ S]_ [1:] _[t][−]_ [1] [)][ by Eq. 4]
**end for**

Update _α_ via policy gradient Eq. 7
**end for**

**for** d-steps **do**

Use _G_ _α_ to generate _X_ _G_ .
Train discriminator _D_ by Eq. 8
Train discriminator _D_ _[′]_ by Eq. 9
**end for**

_γ ←_ _α_
**until** D reaches a stable accuracy.


_The Generative Model_


We use RecurrentNNs (RNNs) to construct the generator.
An RNN maps the input embedding representations _s_ 1 _, ..., s_ _L_
of the input sequence of tokens _S_ 1 _, ..., S_ _L_ into hidden states
_h_ 1 _, ..., h_ _L_ by using the following recursive function.


_h_ _t_ = _g_ ( _h_ _t−_ 1 _, s_ _t_ ) (10)



Finally, a softmax output layer _z_ with bias vector _c_ and weight
matrix _V_ maps the hidden layer neurons into the output token
distribution as


_p_ ( _s|s_ 1 _, ..., s_ _t_ ) = _z_ ( _h_ _t_ ) = softmax( _c_ + _V.h_ _t_ ) (11)


To deal with the common vanishing and exploding gradient
problem [ 27 ] of the backpropagation through time, we exploit
the Long Short-Term Memory (LSTM) cells [28].


_The Discriminator Model_

For the discriminators, we select the CNN because of
their effectiveness for text classification tasks [ 29 ]. First, we
construct the matrix of the sequence by concatenating the input
embedding representations of the sequence of tokens _s_ 1 _, ..., s_ _L_

as:

_ζ_ 1: _L_ = _s_ 1 _⊕_ _... ⊕_ _s_ _L_ (12)


Then a kernel _w_ computes a convolutional operation to a
window size of _l_ by using a non-linear function _π_, which
results in a feature map:


_f_ _i_ = _π_ ( _w ⊗_ _ζ_ _i_ : _i_ + _l−_ 1 + _b_ ) (13)


Where _⊗_ is the inner product of two vectors, and _b_ is a bias
term. Usually, various numbers of kernels with different window
sizes are used in CNN. We hyper-tune size of kernels by trying
kernels which have been successfully used in text classification
tasks by community [ 13 ], [ 30 ], [ 11 ]. Then we apply a maxover-time pooling operation over the feature maps to allow
us to combine the outputs of different kernels. Based on [ 31 ]
we add the highway architecture to improve the performance.
In the end, a fully connected layer with sigmoid activation
functions is used to output the class probability of the input

sequence.


III. E VALUATION

We implemented FakeGAN using the TensorFlow [ 32 ]
framework. We chose the dataset from [ 4 ] which has 800
reviews of 20 Chicago hotels with positive sentiment. The
dataset consists of 400 truthful reviews provided by high profile
users on TripAdvisor and 400 deceptive reviews written by
Amazon Mechanical Workers. To the best of our knowledge,
this is the biggest available dataset of labeled reviews and has
been used by many related works [ 4 ], [ 18 ], [ 33 ]. Similar to
SeqGAN [ 21 ], the generator in FakeGAN only creates fixed
length sentences. Since the majority of reviews in this dataset
has a length less than 200 words, we set the sequence length
of FakeGAN ( _L_ ) to 200. For sentences whose length is less
than 200, we pad them with a fixed token <END> to reach the
size of 200 resulting in 332 truthful and 353 deceptive reviews.
Note that, having a larger dataset results in a less training time.
Although larger dataset makes each adversarial step slower, it
provides _G_ a richer distribution of samples, thus reduces the
number of adversarial steps resulting in less training time.
We used the k-fold cross-validation with k=5 to evaluate
FakeGAN. We leveraged GloVe vectors [2] for word representation [ 34 ]. Similar to SeqGAN [ 21 ], the convergence


2 Check “glove.6B.200d.txt” from https://nlp.stanford.edu/projects/glove/



4


of FakeGAN varies with the training parameters _g_ and _d_
of generator and discriminative models respectively. After
experimenting with different values, we observed that following
values _g_ = 1 and _d_ = 6 are optimal. For pre-training phase, we
trained the generator and the discriminators until convergence,
which took 120 and 50 steps respectively. The adversarial
learning starts after the pre-training phase. All our experiments
were run on a 40-core machine, where the pre-training took
_∼_ one hour, and the adversarial training took _∼_ 11 hours with
a total of _∼_ 12 hours.


_A. Accuracy of Discriminator D_


As mentioned before, the goal of FakeGAN is to generate a
highly accurate discriminator model, _D_, that can distinguish
deceptive and truthful reviews. Figure 3a shows the accuracy
trend for this model; for simplicity, the trend is shown only
for the first iteration of k-fold cross-validation. During the
pre-training phase, the accuracy of _D_ stabilized at 50 _[th]_ step.
We set the adversarial learning to begin at step 51. After
a little decrease in accuracy at the beginning, the accuracy
increases and converges to 89 _._ 2%, which is on-par with the
accuracy of state-of-the-art approach [ 4 ] that applied supervised
machine learning on the same dataset ( _∼_ 89 _._ 8% ). The accuracy,
precision and recall for k-fold cross-validation are 89.1%, 98%
and 81% all with a standard deviation of 0.5. This supports our
hypothesis that adversarial training can be used for detecting
deceptive reviews. Interestingly even though FakeGAN relies
on semi-supervised learning, it yields similar performance as
of a fully-supervised classification algorithm.


_B. Accuracy of Discriminator D_ _[′]_


Figure 3b shows the accuracy trend for the discriminator _D_ _[′]_ .
Similar to _D_, _D_ _[′]_ converges after 450 steps with an accuracy of
_∼_ 99% accuracy. It means that at this point, the generator _G_
will not be able to make any progress trying to fool _D_ _[′]_, and the
output distribution of _G_ will stay almost same. Thus, continuing
adversarial learning does not result in any improvement of the
accuracy of our main discriminator, _D_ .


_C. Comparing FakeGAN with the original GAN approach_


To justify the use of two discriminators in FakeGAN, we
tried using just one discriminator (only _D_ ) in two different
settings. In the first case, the generator _G_ is pre-trained to
learn only _truthful reviews_ distribution. Here the discriminator
_D_ reached 83% accuracy in pre-training, and the accuracy of
adversarial learning, i.e., the classifier, reduces to about 65% .
In the second case, the generator _G_ is pre-trained to learn only
_deceptive reviews_ distribution. Unlike the first case, adversarial
learning improved the performance of _D_ by converging at 84%,
however, still, the performance is lower than that of FakeGAN.
These results demonstrate that using two discriminators is
necessary to improve the accuracy of FakeGAN.


_D. Scalability Discussion_


We argue that the time complexity of our proposed augmented GAN with two discriminators is the same as of original



GANs because their bottleneck is the MC search, where using
the rollout policy (which is _G_ until the time) generates 16
complete sequences, to help the generator _G_ for just outputting
the most promising token as its current action. This happens for
every token of a sequence which is generated by _G_ . However,
compared to MC search, discriminators _D_ and _D_ _[′]_ are efficient
and not time-consuming.


_E. Stability Discussion_


As we discussed in Section II, the _stability_ of GANs is a
known issue. We observed that the parameters _g_ and _d_ have a
large effect on the convergence and performance of FakeGAN
as illustrated in the Figure 4, when _d_ and _g_ are both equal
to one. We believe that the stability of GAN makes hypertuning of FakeGAN a challenging task thus prevents it from
outperforming the state-of-the-art methods based on supervised
machine learning. However, with the following values _d_ = 6
and _g_ = 1, FakeGAN converges and performs on par with the
state-of-the-art approach.


IV. R ELATED WORK


Text classification has been used extensively in email
spam [ 35 ] detection and link spam detection in web pages [ 36 ],

[ 37 ], [ 38 ]. Over the last decade, researchers have been working
on _deceptive opinion spam_ .
Jindal et al. [ 3 ] first introduced _deceptive opinion spam_
problem as a widespread phenomenon and showed that it is
different from other traditional spam activities. They built their
ground truth dataset by considering the duplicate reviews as
spam reviews and the rest as nonspam reviews. They extracted
features related to review, product and reviewer, and trained a
Logistic Regression model on these features to find fraudulent
reviews on Amazon. Wu et al. [ 39 ] claimed that deleting
dishonest reviews will distort the popularity significantly. They
leveraged this idea to detect deceptive opinion spam in the
absence of ground truth data. Both of these heuristic evaluation
approaches are not necessarily true and thorough.
Yoo et al. [ 19 ] instructed a group of tourism marketing
students to write a hotel review from the perspective of a
hotel manager. They gathered 40 truthful and 42 deceptive
hotel reviews and found that truthful and deceptive reviews
have different lexical complexity. Ott et al. [ 4 ] created a
much larger dataset of 800 opinions by crowdsourcing [3] the
job of writing fraudulent reviews for existing businesses.
They combined work from psychology and computational
linguistics to develop and compare three [4] approaches for
detecting deceptive opinion spam. On a similar dataset, Feng
et al. [ 33 ] trained Support Vector Machine model based on
syntactic stylometry features for deception detection. Li et
al. [ 18 ] also combined ground truth dataset created by Ott et
al. [ 4 ] with their employee (domain-expert) generated deceptive
reviews to build a feature-based additive model for exploring
the general rule for deceptive opinion spam detection. Rahman


3 They used Amazon Mechanical Turk
4 Genre identification, psycholinguistic deception detection, and text categorization.



5


(a) Accuracy of FakeGAN (Discriminator _D_ ) at each step by
feeding the testing dataset to _D_ . While minimizing cross entropy
method for pre-training _D_ converges and reaches accuracy at
_∼_ 82%, adversarial training phase boosts the accuracy to _∼_ 89% .



(b) Accuracy of _D_ _[′]_ at each step by feeding the testing dataset
and generated samples by _G_ to _D_ _[′]_ . Similar to figure 3a, this
plot shows that _D_ _[′]_ converged after 450 steps resulting in the
convergence of FakeGAN.



Fig. 3: The accuracy of _D_ and _D_ _[′]_ on the test dataset over epochs. The vertical dashed line shows the beginning of adversarial
training.



(a) The accuracy of _D_ fluctuates around 77% in constrast to the
stabilization at 89 _._ 1% in Figure 3a (with values g=1 and d=6)



(b) Accuracy of _D_ _[′]_ . Unlike in Figure 3b, this plot shows that
_D_ _[′]_ is not stable.



Fig. 4: The accuracy of _D_ and _D_ _[′]_ on the test dataset over epochs while both _g_ and _d_ are one.



et al. [ 40 ] developed a system to detect venues that are targets
of deceptive opinions. Although, this easies the identification
of deceptive reviews considerable effort is still involved in
identifying the actual deceptive reviews. In almost all these
works, the size of the dataset limits the proposed model to
reach its real capacity.
To alleviate these issues with the ground truth, we use a
Generative adversarial network, which is more an unsupervised
learning method rather than supervised. We start with an
existing dataset and use the generator model to create necessary
reviews to strengthen the classifier (discriminator).


V. F UTURE WORK


Contrary to the popular belief that supervised learning techniques are superior to unsupervised techniques, the accuracy of
FakeGAN, a semi-supervised learning technique is comparable
to the state-of-the-art supervised techniques on the same dataset.
We believe that this is a preliminary step which we plan
to extend by trying different architectures like Conditional
GAN [41] and better hyper-tuning.


VI. C ONCLUSION


In this paper, we propose FakeGAN, a technique to detect
deceptive reviews using Generative Adversarial Networks
(GAN). To the best of our knowledge, this is the first work



to leverage GANs and semi-supervised learning methods to
identify deceptive reviews. Our evaluation using a dataset of
800 reviews from 20 Chicago hotels of TripAdvisor shows that
FakeGAN with an accuracy of 89.1% performed on par with the
state-of-the-art models. We believe that FakeGAN demonstrates

a good first step towards using GAN for text classification tasks,
specifically those requiring very large ground truth datasets.


A CKNOWLEDGEMENTS


We would like to thank the anonymous reviewers for
their valuable comments. This material is based on research

sponsored by the Office of Naval Research under grant numbers
N00014-15-1-2948, N00014-17-1-2011 and by DARPA under
agreement number FA8750-15-2-0084. The U.S. Government
is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation
thereon. This work is also sponsored by a gift from Google’s
Anti-Abuse group. The views and conclusions contained herein
are those of the authors and should not be interpreted as
necessarily representing the official policies or endorsements,
either expressed or implied, of DARPA or the U.S. Government.



6


R EFERENCES


[1] M. marketing research company, “Seven in 10 americans seek out opinion before making purchases,”
[http://www.mintel.com/press-centre/social-and-lifestyle/](http://www.mintel.com/press-centre/social-and-lifestyle/seven-in-10-americans-seek-out-opinions-before-making-purchases)
[seven-in-10-americans-seek-out-opinions-before-making-purchases,](http://www.mintel.com/press-centre/social-and-lifestyle/seven-in-10-americans-seek-out-opinions-before-making-purchases)
2015.

[2] M. Anderson and J. Magruder, “Learning from the crowd: Regression
discontinuity estimates of the effects of an online review database,” _The_
_Economic Journal_, vol. 122, no. 563, pp. 957–989, 2012.

[3] N. Jindal and B. Liu, “Opinion spam and analysis,” in _Proceedings of_
_the 2008 International Conference on Web Search and Data Mining_, ser.
WSDM ’08. New York, NY, USA: ACM, 2008, pp. 219–230. [Online].
[Available: http://doi.acm.org/10.1145/1341531.1341560](http://doi.acm.org/10.1145/1341531.1341560)

[4] M. Ott, Y. Choi, C. Cardie, and J. T. Hancock, “Finding deceptive
opinion spam by any stretch of the imagination,” in _Proceedings of the_
_49th Annual Meeting of the Association for Computational Linguistics:_
_Human Language Technologies-Volume 1_ . Association for Computational
Linguistics, 2011, pp. 309–319.

[5] B. Technology, “Yelp admits a quarter of submitted reviews could be
[fake,” September 2013, http://www.bbc.com/news/technology-24299742.](http://www.bbc.com/news/technology-24299742)

[6] M. Luca and G. Zervas, “Fake it till you make it: Reputation, competition,
and yelp review fraud,” _Management Science_, 2016.

[7] R. Socher, J. Pennington, E. H. Huang, A. Y. Ng, and C. D. Manning,
“Semi-supervised recursive autoencoders for predicting sentiment distributions,” in _Proceedings of the conference on empirical methods in natural_
_language processing_ . Association for Computational Linguistics, 2011,
pp. 151–161.

[8] R. Socher, E. H. Huang, J. Pennin, C. D. Manning, and A. Y. Ng,
“Dynamic pooling and unfolding recursive autoencoders for paraphrase
detection,” in _Advances in neural information processing systems_, 2011,
pp. 801–809.

[9] R. Socher, A. Perelygin, J. Y. Wu, J. Chuang, C. D. Manning, A. Y. Ng,
C. Potts _et al._, “Recursive deep models for semantic compositionality
over a sentiment treebank,” in _Proceedings of the conference on empirical_
_methods in natural language processing (EMNLP)_, vol. 1631, 2013, p.
1642.

[10] J. L. Elman, “Finding structure in time,” _Cognitive science_, vol. 14, no. 2,
pp. 179–211, 1990.

[11] S. Lai, L. Xu, K. Liu, and J. Zhao, “Recurrent convolutional neural
networks for text classification.” in _AAAI_, vol. 333, 2015, pp. 2267–
2273.

[12] Y. Kim, “Convolutional neural networks for sentence classification,” _arXiv_
_preprint arXiv:1408.5882_, 2014.

[13] X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks
for text classification,” in _Advances in neural information processing_
_systems_, 2015, pp. 649–657.

[14] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in
_Advances in neural information processing systems_, 2014, pp. 2672–2680.

[15] A. Radford, L. Metz, and S. Chintala, “Unsupervised representation
learning with deep convolutional generative adversarial networks,” _arXiv_
_preprint arXiv:1511.06434_, 2015.

[16] K. Ehsani, R. Mottaghi, and A. Farhadi, “Segan: Segmenting and
generating the invisible,” _arXiv preprint arXiv:1703.10239_, 2017.

[17] E. L. Denton, S. Chintala, R. Fergus _et al._, “Deep generative image
models using a laplacian pyramid of adversarial networks,” in _Advances_
_in neural information processing systems_, 2015, pp. 1486–1494.

[18] J. Li, M. Ott, C. Cardie, and E. H. Hovy, “Towards a general rule for
identifying deceptive opinion spam.” in _ACL (1)_ . Citeseer, 2014, pp.
1566–1576.

[19] K.-H. Yoo and U. Gretzel, “Comparison of deceptive and truthful travel
reviews,” _Information and communication technologies in tourism 2009_,
pp. 37–47, 2009.

[20] L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein, “Unrolled generative
adversarial networks,” _arXiv preprint arXiv:1611.02163_, 2016.

[21] L. Yu, W. Zhang, J. Wang, and Y. Yu, “Seqgan: sequence generative
adversarial nets with policy gradient,” in _Thirty-First AAAI Conference_
_on Artificial Intelligence_, 2017.

[22] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein gan,” _arXiv_
_preprint arXiv:1701.07875_, 2017.

[23] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville,
“Improved training of wasserstein gans,” _arXiv preprint arXiv:1704.00028_,
2017.




[24] I. Goodfellow, “Nips 2016 tutorial: Generative adversarial networks,”
_arXiv preprint arXiv:1701.00160_, 2016.

[25] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van
Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam,
M. Lanctot _et al._, “Mastering the game of go with deep neural networks
and tree search,” _Nature_, vol. 529, no. 7587, pp. 484–489, 2016.

[26] R. S. Sutton, D. A. McAllester, S. P. Singh, and Y. Mansour, “Policy
gradient methods for reinforcement learning with function approximation,”
in _Advances in neural information processing systems_, 2000, pp. 1057–
1063.

[27] I. Goodfellow, Y. Bengio, and A. Courville, _Deep learning_ . MIT Press,
2016.

[28] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” _Neural_
_computation_, vol. 9, no. 8, pp. 1735–1780, 1997.

[29] X. Zhang and Y. LeCun, “Text understanding from scratch,” _arXiv_
_preprint arXiv:1502.01710_, 2015.

[30] W. Y. Wang, “" liar, liar pants on fire": A new benchmark dataset for
fake news detection,” _arXiv preprint arXiv:1705.00648_, 2017.

[31] R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway networks,”
_arXiv preprint arXiv:1505.00387_, 2015.

[32] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S.
Corrado, A. Davis, J. Dean, M. Devin _et al._, “Tensorflow: Large-scale
machine learning on heterogeneous distributed systems,” _arXiv preprint_
_arXiv:1603.04467_, 2016.

[33] S. Feng, R. Banerjee, and Y. Choi, “Syntactic stylometry for deception
detection,” in _Proceedings of the 50th Annual Meeting of the Association_
_for Computational Linguistics: Short Papers-Volume 2_ . Association for
Computational Linguistics, 2012, pp. 171–175.

[34] J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors
for word representation,” in _Empirical Methods in Natural Language_
_Processing (EMNLP)_, 2014, pp. 1532–1543. [Online]. Available:
[http://www.aclweb.org/anthology/D14-1162](http://www.aclweb.org/anthology/D14-1162)

[35] H. Drucker, D. Wu, and V. N. Vapnik, “Support vector machines for
spam categorization,” _IEEE Transactions on Neural networks_, vol. 10,
no. 5, pp. 1048–1054, 1999.

[36] Z. Gyöngyi, H. Garcia-Molina, and J. Pedersen, “Combating web spam
with trustrank,” in _Proceedings of the Thirtieth international conference_
_on Very large data bases-Volume 30_ . VLDB Endowment, 2004, pp.
576–587.

[37] A. Ntoulas, M. Najork, M. Manasse, and D. Fetterly, “Detecting spam web
pages through content analysis,” in _Proceedings of the 15th international_
_conference on World Wide Web_ . ACM, 2006, pp. 83–92.

[38] Z. Gyongyi and H. Garcia-Molina, “Web spam taxonomy,” in _First_
_international workshop on adversarial information retrieval on the web_
_(AIRWeb 2005)_, 2005.

[39] G. Wu, D. Greene, B. Smyth, and P. Cunningham, “Distortion as
a validation criterion in the identification of suspicious reviews,” in
_Proceedings of the First Workshop on Social Media Analytics_ . ACM,
2010, pp. 10–13.

[40] M. Rahman, B. Carbunar, J. Ballesteros, G. Burri, D. Horng _et al._,
“Turning the tide: Curbing deceptive yelp behaviors.” in _SDM_ . SIAM,
2014, pp. 244–252.

[41] M. Mirza and S. Osindero, “Conditional generative adversarial
nets,” _CoRR_ [, vol. abs/1411.1784, 2014. [Online]. Available: http:](http://arxiv.org/abs/1411.1784)
[//arxiv.org/abs/1411.1784](http://arxiv.org/abs/1411.1784)



7


