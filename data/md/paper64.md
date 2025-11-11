# Deep Reinforcement Learning for Imbalanced Classification

Enlu Lin [1] Qiong Chen [2,*] Xiaoming Qi [3]

_School of Computer Science and Engineering_
_South China University of Technology_
Guangzhou, China
linenus@outlook.com csqchen@scut.edu.cn qxmscut@126.com



_**Abstract**_ **—Data in real-world application often exhibit skewed**
**class distribution which poses an intense challenge for machine**
**learning. Conventional classification algorithms are not effective**
**in the case of imbalanced data distribution, and may fail when the**
**data distribution is highly imbalanced. To address this issue, we**
**propose a general imbalanced classification model based on deep**
**reinforcement learning. We formulate the classification problem**
**as a sequential decision-making process and solve it by deep Q-**
**learning network. The agent performs a classification action on**
**one sample at each time step, and the environment evaluates**
**the classification action and returns a reward to the agent. The**
**reward from minority class sample is larger so the agent is more**
**sensitive to the minority class. The agent finally finds an optimal**
**classification policy in imbalanced data under the guidance of**
**specific reward function and beneficial learning environment.**
**Experiments show that our proposed model outperforms the**
**other imbalanced classification algorithms, and it can identify**
**more minority samples and has great classification performance.**


_**Index Terms**_ **—imbalanced classification, deep reinforcement**
**learning, reward function, classification policy**


I. I NTRODUCTION


Imbalanced data classification has been widely researched
in the field of machine learning [1]–[3]. In some real-world
classification problems, such as abnormal detection, disease
diagnosis, risk behavior recognition, etc., the distribution of
data across different classes is highly skewed. The instances
in one class (e.g., cancer) can be 1000 times less than that
in another class (e.g., healthy patient). Most machine learning
algorithms are suitable for balanced training data set. When
facing imbalanced scenarios, these models often provide a
good recognition rates to the majority instances, whereas the
minority instances are distorted. The instances in minority
class are difficult to detect because of their infrequency and
casualness; however, misclassifying minority class instances
can result in heavy costs.
A range of imbalanced data classification algorithms have
been developed during the past two decades. The methods
to tackle these issues are mainly divided into two groups

[4]: the data level and the algorithmic level. The former
group modifies the collection of instances to balance the class
distribution by re-sampling the training data, which often
represents as different types of data manipulation techniques.
The latter group modifies the existing learners to alleviate
their bias towards majority class, which often assigns higher



misclassification cost to the minority class. However, with the
rapid developments of big data, a large amount of complex
data with high imbalanced ratio is generated which brings an
enormous challenge in imbalanced data classification. Conventional methods are inadequate to cope with more and
more complex data so that novel deep learning approaches
are increasingly popular.
In recent years, deep reinforcement learning has been
successfully applied to computer games, robots controlling,
recommendation systems [5]–[7] and so on. For classification
problems, deep reinforcement learning has served in eliminating noisy data and learning better features, which made a great
improvement in classification performance. However, there
has been little research work on applying deep reinforcement
learning to imbalanced data learning. In fact, deep reinforcement learning is ideally suitable for imbalanced data learning
as its learning mechanism and specific reward function are
easy to pay more attention to minority class by giving higher
reward or penalty.
A deep _Q_ -learning network (DQN) based model for imbalanced data classification is proposed in this paper. In our
model, the imbalanced classification problem is regarded as
a guessing game which can be decomposed into a sequential
decision-making process. At each time step, the agent receives
an environment state which is represented by a training sample
and then performs a classification action under the guidance
of a policy. If the agent performs a correct classification action
it will be given a positive reward, otherwise, it will be given
a negative reward. The reward from minority class is higher
than that of majority class. The goal of the agent is to obtain
as more cumulative rewards as possible during the process of
sequential decision-making, that is, to correctly recognize the
samples as much as possible.
The contributions of this paper are summarized as follows: 1) Formulate the classification problem as a sequential
decision-making process and propose a deep reinforcement
learning framework for imbalanced data classification. 2) Design and implement the DQN based imbalanced classification
model DQNimb, which mainly includes building the simulation environment, defining the interaction rules between agent
and environment, and designing the specific reward function.
3) Study the performance of our model through experiments
and compare with the other methods of imbalanced data


learning.
The rest of this paper is organized as follows: The second
section introduces the research methodology of imbalanced
data classification and the applications of deep reinforcement
learning for classification problems. The third section elaborates the proposed model and analyzes it theoretically. The
fourth section shows the experimental results and evaluates the
performance of our method compared with the other methods.
The last section summarizes the work of this paper and looks
forward to the future work.


II. R ELATED W ORK


_A. Imbalanced data classification_


The previous research work in imbalanced data classification concentrate mainly on two levels: the data level [8]–[11]
and the algorithmic level [12]–[21]. Data level methods aim
to balance the class distribution by manipulating the training samples, including over-sampling minority class, undersampling majority class and the combinations of the two above
methods [11]. SMOTE is a well-known over-sampling method,
which generates new samples by linear interpolation between
adjacent minority samples [9]. NearMiss is a typical undersample method based on the nearest neighbor algorithm [10].
However, over-sampling can potentially lead to overfitting
while under-sampling may lose valuable information on the
majority class. The algorithmic level methods aim to lift
the importance of minority class by improving the existing
algorithms, including cost-sensitive learning, ensemble learning, and decision threshold adjustment. The cost-sensitive
learning methods assign various misclassification costs to
different classes by modifying the loss function, in which
the misclassification cost of minority class is higher than
that of majority class. The ensemble learning based methods
train multiple individual sub-classifiers, and then use voting
or combining to get better results. The threshold-adjustment
methods train the classifier in original imbalanced data and
change the decision threshold in test time. A number of
deep learning based methods have recently been proposed
for imbalanced data classification [22]–[26]. Wang _et al_ . [22]
proposed a new loss function in deep neural network which
can capture classification errors from both majority class and
minority class equally. Huang _et al_ . [23] studied a method
that learns more discriminative feature of imbalanced data

by maintaining both inter-cluster and inter-class margins. Yan
_et al_ . [24] used a bootstrapping sampling algorithm which
ensures the training data in each mini-batch for convolutional
neural network is balanced. A method to optimize the network
parameters and the class-sensitive costs jointly was presented
in [25]. In [26] Dong _et al_ . mined hard samples in minority
classes and improved the algorithm by batch-wise optimization
with Class Rectification Loss function.


_B. Reinforcement learning for classification problem_


Deep reinforcement learning has recently achieved excellent
results in classification tasks as it can assist classifiers to learn
advantageous features or select high-quality instances from



noisy data. In [27], the classification task was constructed
into a sequential decision-making process, which uses multiple agents to interact with the environment to learn the
optimal classification policy. However, the intricate simulation
between agents and environment caused extremely high time
complexity. Feng _et al_ . [28] proposed a deep reinforcement
learning based model to learn the relationship classification in
noisy text data. The model is divided into instance selector
and relational classifier. The instance selector selects highquality sentence from noisy data under the guidance of agent
while the relational classifier learns better performance from
selected clean data and feeds back a delayed reward to the
instance selector. The model finally obtains a better classifier
and high-quality data set. The work in [29]–[32] utilized
deep reinforcement learning to learn advantageous features
of training data in their respective applications. In general,
the advantageous features improve the classifier while the
better classifier feeds back a higher reward which encourages
the agent to select more advantageous features. Martinez _et_
_al_ . [33] proposed a deep reinforcement learning framework
for time series data classification in which the definition of
specific reward function and the Markov process are clearly
formulated. Research in imbalanced data classification with
reinforcement learning was quite limited. In [34] an ensemble
pruning method was presented that selected the best subclassifiers by using reinforcement learning. However, this
method was merely suitable for traditional small dataset because it was inefficient to select classifiers when there were

                                      plenty of sub-classifiers. In this paper, we propose deep _Q_
network based model for imbalanced classification which is
efficient in complex high-dimensional data such as image
or text and has a good performance compared to the other
imbalanced classification methods.


III. M ETHODOLOGY


_A. Imbalanced Classification Markov Decision Process_


Reinforcement learning algorithms that incorporate deep
learning have defeated world champions at the game of Go as
well as human experts playing numerous Atari video games.
Now we regard classification problem as a guessing game,
the agent receives a sample at each time step and guesses
(classifies) which category the sample belongs to, and then
the environment returns it an immediate reward and the next

sample, as shown in Fig.1. A positive reward is given to the
agent by the environment when the agent correctly guesses
the category of sample, otherwise a negative reward is given
to the agent. When the agent learns an optimal behavior
from its interaction with environment to get the maximum
accumulative rewards, it can correctly classify samples as
much as possible.
Now we formalize the Imbalanced Classification Markov
Decision Process (ICMDP) framework which decomposes
imbalanced data classification task into a sequential decisionmaking problem. Assume that the imbalanced training data
set is _D_ = _{_ ( _x_ 1 _, l_ 1 ) _,_ ( _x_ 2 _, l_ 2 ) _, ...,_ ( _x_ _n_ _, l_ _n_ ) _}_ where _x_ _i_ is the ith


**Agent**



**Storing interactive experience**















when the agent misclassifies the sample from minority
class.


_•_ **Policy** _π_ _θ_ : The policy _π_ _θ_ is a mapping function _π_ : _S →_
_A_ where _π_ _θ_ ( _s_ _t_ ) denotes the action _a_ _t_ performed by agent
in state _s_ _t_ . The policy _π_ _θ_ in ICMDP can be considered
as a classifier with the parameter _θ_ .


With the definitions and notations above, the imbalanced
classification problem is formally defined as to find an optimal
classification policy _π_ _[∗]_ : _S →A_, which maximized the
cumulative rewards in ICMDP.


_B. Reward function for imbalanced data classification_


The minority class samples are difficult to be identified
correctly in imbalance data set. In order to better recognize the
minority class samples, the algorithm should be more sensitive
to the minority class. A large reward or punishment is returned
to agent when it meets a minority sample. The reward function
is defined as follows:









_**s**_ _**t+1**_


_**s**_ _**t+2**_



Fig. 1. Overall process of ICMDP.


sample and _l_ _i_ is the label of the ith sample. We propose to
train a classifier as an agent evolving in ICMDP where:


_•_ **State** _S_ : The state of environment is determined by the
training sample. At the beginning of training, the agent
receives the first sample _x_ 1 as its initial state _s_ 1 . The state
_s_ _t_ of environment at each time step corresponds to the
sample _x_ _t_ . When the new episode begins, environment
shuffles the order of samples in training data set.

_•_ **Action** _A_ : The action of agent is associated with the label
of training data set. The action _a_ _t_ taken by agent is to
predict a class label. For binary classification problem,
_A_ = _{_ 0 _,_ 1 _}_ where 0 represents the minority class and 1
represents the majority class.

_•_ **Reward** _R_ : A reward _r_ _t_ is the feedback from environment by which we measure the success or failure
of an agent’s actions. In order to guide the agent to
learn the optimal classification policy in imbalanced data,
the absolute reward value of sample in minority class
is higher than that in majority class. That is, when the
agent correctly or incorrectly recognizes minority class
sample, the environment feedback agent a larger reward
or punishment.

_•_ **Transition** **probability** _P_ : Transition probability
_p_ ( _s_ _t_ +1 _|s_ _t_ _, a_ _t_ ) in ICMDP is deterministic. The agent
moves from the current state _s_ _t_ to the next state _s_ _t_ +1
according to the order of samples in the training data

set.

_•_ **Discount factor** _γ_ : _γ ∈_ [0 _,_ 1] is to balance the immediate
and future reward.


_•_ **Episode** : Episode in reinforcement learning is a transition trajectory from the initial state to the terminal
state _{s_ 1 _, a_ 1 _, r_ 1 _, s_ 2 _, a_ 2 _, r_ 2 _, ..., s_ _t_ _, a_ _t_ _, r_ _t_ _}_ . An episode ends
when all samples in training data set are classified or



_R_ ( _s_ _t_ _, a_ _t_ _, l_ _t_ ) =



 _λ,−_ +11 _,,_ _aaa_ _ttt_ = _̸_ == _l l l_ _ttt_ and and and _s s s_ _ttt_ _∈ ∈ ∈_ _DDD_ _NPP_

 _−λ,_ _a_ _t_ _̸_ = _l_ _t_ and _s_ _t_ _∈_ _D_ _N_



(1)



where _λ ∈_ [0 _,_ 1], _D_ _P_ is minority class sample set, _D_ _N_ is
majority class sample set, _l_ _t_ is the class label of the sample
in state _s_ _t_ . Let the reward value be 1 _/ −_ 1 when the agent

_−_
correctly/incorrectly classifies a minority class sample, be _λ/_
_λ_ when the agent correctly/incorrectly classifies a majority
class sample.
The value of reward function is the prediction cost of agent.
For imbalanced data set ( _λ <_ 1), the prediction cost values of
minority class are higher than that of majority class. If the
class distribution of training data set is balanced, then _λ_ = 1,
the prediction cost values are the same for all classes. In fact,
_λ_ is a trade-off parameter to adjust the importance of majority
class. Our model achieves the best performance in experiment
when _λ_ is equal to the imbalanced ratio _ρ_ = _|_ _[|]_ _D_ _[D]_ _N_ _[P]_ _[ |]_ _|_ [. We will]

discuss it in Section IV-F.


_C. DQN based imbalanced classification algorithm_


_1) Deep Q-learning for ICMDP:_ In ICMDP, the classification policy _π_ is a function which receives a sample and return
the probabilities of all labels.


_π_ ( _a|s_ ) = _P_ ( _a_ _t_ = _a|s_ _t_ = _s_ ) (2)


The classifier agent’s goal is to correctly recognize the
sample of training data as much as possible. As the classifier
agent can get a positive reward when it correctly recognizes
a sample, thus it can achieve its goal by maximizing the
cumulative rewards _g_ _t_ :



_g_ _t_ =



_∞_
� _γ_ _[k]_ _r_ _t_ + _k_ (3)


_k_ =0


In reinforcement learning, there is a function that calculates
the quality of a state-action combination, called the _Q_ function:


_Q_ _[π]_ ( _s, a_ ) = _E_ _π_ [ _g_ _t_ _|s_ _t_ = _s, a_ _t_ = _a_ ] (4)


According to the Bellman equation [35], the _Q_ function can
be expressed as:


_Q_ _[π]_ ( _s, a_ ) = _E_ _π_ [ _r_ _t_ + _γQ_ _[π]_ ( _s_ _t_ +1 _, a_ _t_ +1 ) _|s_ _t_ = _s, a_ _t_ = _a_ ] (5)


The classifier agent can maximize the cumulative rewards
by solving the optimal _Q_ _[∗]_ function, and the greedy policy
under the optimal _Q_ _[∗]_ function is the optimal classification
policy _π_ _[∗]_ for ICMDP.



_2) Influence of reward function:_ In imbalanced data, the
trained _Q_ network will be biased toward the majority class.
However, due to the aforementioned reward function (1), it
assigns different rewards for different classes and ultimately
makes the samples from different classes have the same impact
on _Q_ network.
Suppose the positive and negative samples are denoted as
_s_ [+] and _s_ _[−]_, their target _Q_ values are represented as _y_ [+] and
_y_ _[−]_ . According to (1) and (9), the target _Q_ value of positive
and negative samples is expressed as:



_y_ [+] =



( _−_ 1) [1] _[−][I]_ [(] _[a]_ [=] _[l]_ [)] _,_ _terminal_ =True
�( _−_ 1) [1] _[−][I]_ [(] _[a]_ [=] _[l]_ [)] + _γ_ max _a_ _′_ _Q_ ( _s_ _[′]_ _, a_ _[′]_ ) _,_ _terminal_ =False
(11)



_y_ _[−]_ = ( _−_ 1) [1] _[−][I]_ [(] _[a]_ [=] _[l]_ [)] _λ,_ _terminal_ =True

�( _−_ 1) [1] _[−][I]_ [(] _[a]_ [=] _[l]_ [)] _λ_ + _γ_ max _a_ _[′]_ _Q_ ( _s_ _[′]_ _, a_ _[′]_ ) _,_ _terminal_ =False
(12)
where _I_ ( _x_ ) is an indicator function.
Rewrite the loss function _L_ ( _θ_ _k_ ) of _Q_ network to the form
of the sum of positive class loss function _L_ + ( _θ_ _k_ ) and negative
class loss function _L_ _−_ ( _θ_ _k_ ). The derivative of _L_ + ( _θ_ _k_ ) and
_L_ _−_ ( _θ_ _k_ ) is shown as follows:



_π_ _[∗]_ ( _a|s_ ) =



1 _,_ if _a_ = arg max _a_ _Q_ _[∗]_ ( _s, a_ )
(6)
�0 _,_ else



Substituting (6) into (5), the optimal _Q_ _[∗]_ function can be
shown as:


_Q_ _[∗]_ ( _s, a_ ) = _E_ _π_ [ _r_ _t_ + _γ_ max _Q_ _[∗]_ ( _s_ _t_ +1 _, a_ _t_ +1 ) _|s_ _t_ = _s, a_ _t_ = _a_ ]
_a_
(7)
In the low-dimensional finite state space, _Q_ functions are
recorded by a table. However, in the high-dimensional continuous state space, _Q_ functions cannot be resolved until deep
_Q_ -learning algorithm was proposed, which fits the _Q_ function
with a deep neural network. In deep _Q_ -learning algorithm, the
interaction data ( _s, a, r, s_ _[′]_ ) obtained from (7) are stored in the
experience replay memory _M_ . The agent randomly samples a
mini-batch of transitions _B_ from _M_ and performs a gradient
descent step on the Deep _Q_ network according to the loss
function as follow:


_L_ ( _θ_ _k_ ) = � ( _y −_ _Q_ ( _s, a_ ; _θ_ _k_ )) [2] (8)

( _s,a,r,s_ _[′]_ ) _∈B_


where _y_ is the target estimate of the _Q_ function, the expression
of _y_ is:



_P_
_−_ 2 �



_P_

_i_ =1 [(] _[−]_ [1)] [1] _[−][I]_ [(] _[a]_ _[i]_ [=] _[l]_ _[i]_ [)] _[ ∇][Q]_ [(] _[s]_ _∇_ _[i]_ _[,]_ _θ_ _[ a]_ _[i]_ [;] _[ θ]_ _[k]_ [)]



_∇L_ + ( _θ_ _k_ )



_P_ _∇Q_ ( _s_ [+] _i_ _[,][ a]_ _[i]_ [;] _[ θ]_ _[k]_ [)]

_i_ =1 � _y_ _i_ [+] _[−]_ _[Q]_ [(] _[s]_ [+] _i_ _[, a]_ _[i]_ [;] _[ θ]_ _[k]_ [)] � _∇θ_ _k_
(13)



_L∇_ + _θ_ ( _k_ _θ_ _k_ ) = _−_ 2 � _Pi_



_∇L∇_ _−_ _θ_ ( _k_ _θ_ _k_ ) = _−_ 2 � _Nj_ =1 � _y_ _j_ _[−]_ _[−]_ _[Q]_ [(] _[s]_ _[−]_ _j_ _[, a]_ _[j]_ [;] _[ θ]_ _[k]_ [)] � _∇Q_ ( _s∇_ _[−]_ _j_ _θ_ _[, a]_ _k_ _[j]_ [;] _[ θ]_ _[k]_ [)]

(14)
where _P_ is the total number of the positive samples set, _N_ is
the total number of the negative samples set.
Substituting (11) into (13), (12) into (14) and adding the
derivative of _L_ + ( _θ_ _k_ ) and _L_ _−_ ( _θ_ _k_ ), then we get the following:



_∇L_ _−_ ( _θ_ _k_ )



_L∇_ _−_ _θ_ ( _k_ _θ_ _k_ ) = _−_ 2 � _Nj_



_∇L_ ( _θ_ _k_ )



_Q_ ( _s_ _[′]_ _m_ _[, a]_ _m_ _[′]_ [;] _[ θ]_ _[k][−]_ [1] [)]
_m_ =1 [((1] _[ −]_ _[t]_ _[m]_ [)] _[γ]_ [ max] _a_ _[′]_ _m_



_∇L_ ( _θθ_ _kk_ ) = _−_ 2 � _Pm_ +=1 _N_



_−_ _Q_ ( _s_ _m_ _, a_ _m_ ; _θ_ _k_ )) _[∇][Q]_ [(] _[s]_ _[m]_ _[,][ a]_ _[m]_ [;] _[ θ]_ _[k]_ [)]

_∇θ_ _k_



_∇θ_ _k_



_∇θ_ _k_
(15)



_y_ =



_r,_ _terminal_ =True
(9)
� _r_ + _γ_ max _a_ _′_ _Q_ ( _s_ _[′]_ _, a_ _[′]_ ; _θ_ _k−_ 1 ) _,_ _terminal_ =False



_N_
_−_ 2 _λ_ �



_N_ _[∇][Q]_ [(] _[s]_ _[j]_ _[,][ a]_ _[j]_ [;] _[ θ]_ _[k]_ [)]

_j_ =1 [(] _[−]_ [1)] [1] _[−][I]_ [(] _[a]_ _[j]_ [=] _[l]_ _[j]_ [)] _∇θ_



where _s_ _[′]_ is the next state of _s_, _a_ _[′]_ is the action performed by
agent in state _s_ _[′]_ .
The derivative of loss function (8) with respect to _θ_ is:



_∇L_ ( _θ_ _k_ )



� ( _y −_ _Q_ ( _s, a_ ; _θ_ _k_ )) _[∇][Q]_ [(] _∇_ _[s][,]_ _θ_ _[ a]_ _k_ [;] _[ θ]_ _[k]_ [)]

( _s,a,r,s_ _[′]_ ) _∈B_



_k_

_∇θ_ _k_ = _−_ 2 �



_∇θ_ _k_ _∇θ_ _k_

( _s,a,r,s_ _[′]_ ) _∈B_

(10)
Now we can obtain the optimal _Q_ _[∗]_ function by minimizing
the loss function (8), the greedy policy (6) under the optimal
_Q_ _[∗]_ function will get the maximum cumulative rewards. So
the optimal classification policy _π_ _[∗]_ : _S →A_ for ICMDP is
achieved.



where _t_ _m_ =1 if _terminal_ =True, otherwise _t_ _m_ =0.
In (15), the second item relates to the minority class and the
third item relates to the majority class. For imbalanced data set
( _N > P_ ), if _λ_ = 1, the immediate rewards of the two classes
are identical, the value of the third item is larger than that of
the second item because the number of samples in majority
class are much more than that in minority class. So the model
is biased to the majority class. If _λ <_ 1, _λ_ can reduce the
immediate rewards of negative samples and weakens their
impact on the loss function of _Q_ network. What’s more, the
second item has the same value as the third item when _λ_ is

equal to the imbalanced ratio _ρ_ .


**Algorithm 1:** Training

**Input:** Training data _D_ = _{_ ( _x_ 1 _, l_ 1 ) _,_ ( _x_ 2 _, l_ 2 ) _, ...,_ ( _x_ _T_ _, l_ _T_ ) _}_ .
Episode number K.
Initialize experience replay memory _M_
Randomly initialize parameters _θ_
Initialize simulation environments _ε_

**for** _episode k_ = 1 _to K_ **do**

Shuffle the training data _D_
Initialize state _s_ 1 = _x_ 1

**for** _t_ = 1 _to T_ **do**

Choose an action based _ϵ_ -greedy policy:
_a_ _t_ = _π_ _θ_ ( _s_ _t_ )
_r_ _t_ _, terminal_ _t_ = _STEP_ ( _a_ _t_ _, l_ _t_ )
Set _s_ _t_ +1 = _x_ _t_ +1
Store ( _s_ _t_ _, a_ _t_ _, r_ _t_ _, s_ _t_ +1 _, terminal_ _t_ ) to _M_
Randomly sample ( _s_ _j_ _, a_ _j_ _, r_ _j_ _, s_ _j_ +1 _, terminal_ _j_ )
from _M_

Set _y_ _j_ =
_r_ _j_ _,_ _terminal_ _j_ =True
� _r_ _j_ + _γ_ max _a_ _′_ _Q_ ( _s_ _j_ +1 _, a_ _[′]_ ; _θ_ ) _,_ _terminal_ _j_ =False
Perform a gradient descent step on _L_ ( _θ_ ) w.r.t. _θ_ :
_L_ ( _θ_ ) = ( _y_ _j_ _−_ _Q_ ( _s_ _j_ _, a_ _j_ ; _θ_ )) [2]

**if** _terminal_ _t_ _=True_ **then**

break


**Algorithm 2:** Environment simulation


_D_ _P_ represents the minority class sample set.
**Function** _STEP(a_ _t_ _∈A, l_ _t_ _∈_ _L)_

Initialize _terminal_ _t_ =False
**if** _s_ _t_ _∈_ _D_ _P_ **then**

**if** _a_ _t_ = _l_ _t_ **then**

Set _r_ _t_ =1

**else**

Set _r_ _t_ =-1
_terminal_ _t_ =True

**else**


**if** _a_ _t_ = _l_ _t_ **then**

Set _r_ _t_ = _λ_

**else**

Set _r_ _t_ = _−λ_


return _r_ _t_ _, terminal_ _t_


_3) Training details:_ We construct the simulation environment according to the definition of ICMDP. The architecture
of the _Q_ network depends on the complexity and amount of
training data set. The input of the _Q_ network is consistent
with the structure of training sample, and the number of
outputs is equal to the number of sample categories. In fact,
the _Q_ network is a neural network classifier without the final
softmax layer. The training process of _Q_ network is described
in Algorithm 1. In an episode, the agent uses the _ϵ_ -greedy
policy to pick the action, and then obtains the reward from



the environment through the _STEP_ function in Algorithm 2.
The deep _Q_ -learning algorithm will be running about 120000
iterations (updates of network parameters _θ_ ). We save the
parameters of the converged _Q_ network which plus a softmax
layer can be regarded as a neural network classifier trained by
imbalanced data.


IV. E XPERIMENT


_A. Comparison Methods and Evaluation Metrics_


We compare our method DQNimb with five imbalanced data
learning methods from the data level and the algorithmic level,
including sampling techniques, and cost-sensitive learning
methods and decision threshold adjustment method. A deep
neural network trained with cross entropy loss function will be
used as baseline in our experiments. The comparison methods
are shown as follows:


_•_ **DNN** : A method which trains the deep neural network using cross entropy loss function without any improvement
strategy in imbalanced data set.

_•_ **ROS** : A re-sampling method to build a more balanced
data set through over-sampling minority classes by random replication [8].

_•_ **RUS** : A re-sampling method to build a more balanced
data set through under-sampling majority classes by random sample removal [8].

_•_ **MFE** : A method to improve the classification performance of deep neural network in imbalanced data sets
by using mean false error loss function [22]

_•_ **CSM** : A cost sensitive method which assigns greater
misclassification cost to minority class and smaller cost
to majority class in loss function [17]

_•_ **DTA** : A method to train the deep neural network in imbalanced data and to adjust the model decision threshold
in test time by incorporating the class prior probability

[19]

In our experiment, to evaluate the classification performance
in imbalanced data sets more reasonably, G-mean and Fmeasure metrics [36] which are popularly used in imbalanced
data sets are adopted. G-mean is the geometric mean of
sensitivity and precision: G-mean=� _T PT P_ + _F N_ _[×]_ _T NT N_ + _F P_ [. F-]



sensitivity and precision: G-mean=� _T PT P_ + _F N_ _[×]_ _T NT N_ + _F P_ [. F-]

measure represents a harmonic mean between recall and precision: F-measure= _T P_ _T P_
� _T P_ + _F N_ _[×]_ _T P_ + _F P_ [. The higher the G-]



cision: F-measure= _T P_ _T P_
� _T P_ + _F N_ _[×]_ _T P_ + _F P_ [. The higher the G-]

mean score and F-measure score are, the better the algorithm
performs.



_B. Dataset_


In this paper, we mainly study the binary imbalanced
classification with deep reinforcement learning. We perform
experiments on IMDB, Cifar-10, Mnist and Fashion-Minist.
Our approach is evaluated on the deliberately imbalanced
splits. The simulated datasets used for the experiments are
shown in Table I.

**IMDB** is a text dataset, which contains 50000 movies
reviews labeled by sentiment (positive/negative). Reviews have
been preprocessed, and each review is encoded as a sequence


TABLE I

D ATASET OF E XPERIMENTS











TABLE II

N ETWORK ARCHITECTURE USED FOR TEXT DATASET


Layer Input Output


Embedding 500 (500,64)
Flatten (500,64) (32000)
FullyConnected (32000) 250
ReLU       -       FullyConnected 250 2
Softmax 2 2


TABLE III

N ETWORK ARCHITECTURE USED FOR IMAGE DATASET


Layer Width Height Depth Kernel size Stride


Input 28(32) 28(32) 1(3)  -  Convolution 28(32) 28(32) 32 5 1
ReLU 28(32) 28(32) 32 - MaxPooling 14(16) 14(16) 32 2 2
Convolution 14(16) 14(16) 32 5 1
ReLU 14(16) 14(16) 32 - MaxPooling 7(8) 7(8) 32 2 2
Flatten 1 1 1568(2048)  -  FullyConnected 1 1 256  -  ReLU 1 1 256 - FullyConnected 1 1 2  -  Softmax 1 1 2  -  

The training dataset with different imbalance levels are obtained by reducing the number of positive class to _ρ×N_ where
_N_ is the total number of negative class and _ρ_ is imbalanced
ratio of dataset. The detail description of experiment dataset
is shown in Table I.


_C. Network Architecture_


We use deep neural network to learn the feature representation from the imbalanced and high dimensional datasets.
For the compared algorithms, the network architecture used
for text (IMDB) dataset has a embedding layer and two fully
connected layers and a softmax output layer. The detailed parameters are given in Table II. The network architecture that is
used for image (Mnist, Fashion-Mnist,Cifar-10) classification
has two convolution layers and two fully connected layers and
a softmax output layer. Its detailed parameters are given in
Table III. For our model, the _Q_ network architecture is similar
to the network structure of compared algorithms, but the final
softmax output layer is removed because it does not need to
scale the _Q_ value of different actions between 0 and 1.


TABLE IV

E XPERIMENT RESULTS ON BALANCED DATASETS


Dataset G-mean F-measure
(balanced) DNN DQNimb DNN DQNimb


IMDB **0.864** **0.864** 0.863 **0.865**
Cifar-10(1) 0.962 **0.967** 0.941 **0.950**
Cifar-10(2) 0.959 **0.963** 0.946 **0.952**
Fashion-Mnist(1) 0.978 **0.984** 0.978 **0.984**
Fashion-Mnist(2) 0.990 **0.991** 0.990 **0.991**
Mnist 0.995 **0.997** 0.985 **0.992**



|Dataset|Dimension<br>of sample|Imbalance<br>ratio ρ|Training data|Col5|Test data|Col7|
|---|---|---|---|---|---|---|
|Dataset|Dimension<br>of sample|Imbalance<br>ratio_ ρ_|Pt.nmb~~a~~|Ng.nmb~~b~~|Pt.nmb|Ng.nmb|
|IMDB|1*500|10%|1250|12000|12500|12500|
|IMDB|1*500|5%|625|625|625|625|
|IMDB|1*500|2%|250|250|250|250|
|Cifar<br>-10(1)|32*32*3|4%|400|10000|1000|2000|
|Cifar<br>-10(1)|32*32*3|2%|200|200|200|200|
|Cifar<br>-10(1)|32*32*3|1%|100|100|100|100|
|Cifar<br>-10(1)|32*32*3|0.5%|50|50|50|50|
|Cifar<br>-10(2)|Cifar<br>-10(2)|4%|800|20000|1000|4000|
|Cifar<br>-10(2)|Cifar<br>-10(2)|2%|400|400|400|400|
|Cifar<br>-10(2)|Cifar<br>-10(2)|1%|200|200|200|200|
|Cifar<br>-10(2)|Cifar<br>-10(2)|0.5%|100|100|100|100|
|Fashion-<br>Mnist(1)|28*28*1<br><br>|4%|480|12000|2000|2000|
|Fashion-<br>Mnist(1)|28*28*1<br><br>|2%|240|240|240|240|
|Fashion-<br>Mnist(1)|28*28*1<br><br>|1%|120|120|120|120|
|Fashion-<br>Mnist(1)|28*28*1<br><br>|0.5%|60|60|60|60|
|Fashion-<br>Mnist(2)|Fashion-<br>Mnist(2)|4%|720|18000|3000|3000|
|Fashion-<br>Mnist(2)|Fashion-<br>Mnist(2)|2%|360|360|360|360|
|Fashion-<br>Mnist(2)|Fashion-<br>Mnist(2)|1%|180|180|180|180|
|Fashion-<br>Mnist(2)|Fashion-<br>Mnist(2)|0.5%|90|90|90|90|
|Mnist|28*28*1|1%|540|54042|1032|8968|
|Mnist|28*28*1|0.2%|108|108|108|108|
|Mnist|28*28*1|0.1%|54|54|54|54|
|Mnist|28*28*1|0.05%|27|27|27|27|


~~a~~ Number of Positive class samples. ~~b~~ Number of Negative class samples.


of word indexes. The standard train/test split for each class is
12500/12500. The positive reviews are regarded as the positive
class in our experiment.
**Mnist** is a simple image dataset. It consists of 28 _×_ 28
grayscale images. There are 10 classes corresponding to digits
from 0 to 9. The number of train/test samples per class is
almost 6000/1000. We let the images with label 2 as the
positive class and the rest images as the negative class in our
experiment.
**Fashion-Mnist** is a new dataset comprising of 28 _×_ 28
grayscale images of 70000 fashion products with 10 categories. It is designed to serve as a direct drop-in replacement
for the original Mnist dataset. The training dataset has 6000
images per class while the test dataset has 1000 images per
class. To evaluate our algorithm on various scales of datasets,
two simulated data sets of different sizes are extracted from

this dataset. The first one chooses the images labeled by 0,2
(T-Shirt, Pullover) as the positive class and the images labeled
by 1,3 (Trouser, Dress) as the negative class. The second one
chooses the images labeled by 4,5,6 (Coat, Sandal, Shirt) as
the positive class and the images labeled by 7,8,9 (Sneaker,
Bag, Ankle boot) as the negative class.
**Cifar-10** is a more complex image dataset than FashionMnist. It contains 32x32 color images with 10 classes of
natural objects. The standard train/test split for each class
is 5000/1000. There are two simulated data sets of different

sizes are extracted from this dataset. The first one chooses the
images labeled by 1 (automobile) as the positive class and the
images labeled by 3,4,5,6 (cat, deer, dog, frog) as the negative
class.The other one takes the images labeled by 7 (horse) as
the positive class and the images labeled by 8,9 (ship, truck)
as the negative class.


TABLE V

G- MEAN SCORE OF EXPERIMENT RESULTS


Imbalance DQNimb Baseline MFE loss Over-sampling Under-sampling Cost-sensitive Threshold-Adjustment
Dataset
ratio _ρ_ (Ours) (DNN) (MFE) (ROS) (RUS) (CSM) (DTA)


10% **0.820** 0.548 0.687 0.681 0.740 **0.743** 0.678

IMDB 5% **0.781** 0.299 0.589 0.632 0.622 **0.696** 0.599

2% **0.682** 0.034 0.351 0.343 0.510 **0.559** 0.355


4% **0.956** 0.869 0.939 **0.947** 0.945 0.944 0.946

2% **0.941** 0.824 0.908 0.925 **0.929** 0.922 0.928
Cifar-10(1)
1% **0.917** 0.730 0.859 0.897 0.896 0.884 **0.912**


0.5% **0.890** 0.579 0.759 0.838 0.866 0.853 **0.901**


4% **0.925** 0.815 0.882 0.904 0.906 0.911 **0.915**

2% **0.917** 0.758 0.852 0.894 0.887 0.886 **0.908**
Cifar-10(2)
1% **0.883** 0.677 0.769 0.854 0.859 0.850 **0.873**


0.5% **0.829** 0.513 0.693 0.792 **0.822** 0.816 0.821


4% **0.971** 0.921 0.960 0.962 0.957 **0.964** **0.964**

2% **0.966** 0.885 0.947 0.957 0.953 0.956 **0.962**
Fashion-Mnist(1)
1% **0.959** 0.853 0.934 0.948 0.943 0.946 **0.952**


0.5% **0.950** 0.757 0.901 0.927 0.934 0.924 **0.944**


4% **0.985** 0.951 0.968 0.972 0.967 0.973 **0.977**

2% **0.982** 0.926 0.960 0.963 0.956 0.966 **0.970**
Fashion-Mnist(2)
1% **0.979** 0.872 0.940 0.949 0.946 0.958 **0.962**


0.5% **0.972** 0.821 0.912 0.935 0.937 0.950 **0.953**


1% **0.991** 0.967 **0.982** 0.981 0.978 **0.982** 0.978

0.2% **0.983** 0.923 0.949 0.944 0.953 0.951 **0.961**
Mnist
0.1% **0.968** 0.856 0.921 0.911 0.929 **0.942** 0.937


0.05% **0.941** 0.694 0.842 0.858 0.907 **0.921** 0.916


The 1 _[st]_ _/_ 2 _[nd]_ best results are indicated in red/blue.



_D. Parameter Setting_


We use _ϵ_ -greedy policy for DQN based imbalanced classification model in which the probability of exploration _ϵ_ is
linearly attenuated from 1.0 to 0.01. The size of experience replay memory is 50 000 and the interactions between agent and
environment are approximately 120 000 steps. The discount
factor of immediate reward _γ_ is 0.1. Adam algorithm is used
to optimize the parameters of _Q_ -network and its learning rate
is 0.00025. For other algorithms, the optimizer is Adam and
its learning rate is 0.0005, the batch size is 64. We randomly
select 10% samples of training data as the verification data
and use early stopping technique [37] which monitors the
validation loss to train the deep neural network for 100 epochs.


_E. Experiment Result_


Before the research of imbalanced data learning, we compare our DQNimb model to the DNN that is a supervised
deep learning model in balanced data sets. The experiments
were conducted on the six data sets (the imbalance ratio
_ρ_ is 1) in Table I. The number of positive samples and
negative samples are equal, so the reward function of the
DQNimb model assigns the same reward or punishment to
the positive and negative samples. For fairness and convincing
comparisons, the network architecture of the DNN model is
the same as the Q network architecture of the DQNimb model.
The G-mean scores and F-measure scores of the experimental
results are shown in Table IV. Despite of the different learning
mechanisms, that the DQNimb model obtains the optimal classification strategy by maximizing the cumulative rewards in



the Markov process, while the DNN gets the optimal network
parameters by minimize the cross-entropy loss function, both
models demonstrate good performance in experimental results.
The G-mean scores and F-measure scores of the DQNimb
model are slightly better than those of the DNN model.
Given the number of the negative samples of the imbalanced
data set is _N_, we randomly select _ρ × N_ positive samples according to the imbalance ratio _ρ_, and conducted 6 experiments.
We report the G-mean scores of our method and the other
methods on the different imbalanced data sets in Table V. Each

training was repeated 5 times on the same data set. The results
of data sampling methods, cost-sensitive learning methods
and threshold adjustment method are much better than DNN
model in imbalanced classification problems, however, our
model DQNimb achieves an outstanding performance with an
overwhelming superiority. In the IMDB text dataset, G-mean
score of our method DQNimb are normally 7.7% higher than
the second-ranked method CSM, and are even 12.3% higher
when the imbalance ratio is 2%.

We report the F-measure scores of different algorithms in
Fig.2. With the increase of data imbalance level, the F-measure
scores of each algorithm show a significant decline. The DNN
model suffers the most serious declination, that is, DNN
can hardly identify any minority class sample when the data
distribution is extremely imbalanced. Meanwhile, our model
DQNimb enjoys the smallest decrease because our algorithm
possesses both the advantages of the data level models and the
algorithmic level models. In the data level, our model DQNimb
has an experience replay memory of storing interactive data


(a) Mnist (b) Fashion-Mnist(1) (c) Fashion-Mnist(2)


(d) Cifar-10(1) (e) Cifar-10(2) (f) IMDB


Fig. 2. Comparison of methods with respect to F-measure score on different datasets.


during the learning process. When the model misclassifies
a positive sample, the current episode will be terminated,
this can alleviate the skewed distribution of the samples in
the experience replay memory. In the algorithmic level, the
DQNimb model gives a higher reward or penalty for positive
samples, which raises the attention to the samples in minority
class and increases the probabilities that positive samples are
correctly identified.



_F. Exploration On Reward Function_

Reward function is used to evaluate the value of actions

performed by agent and inspires it to work toward to the
goal. In DQNimb model, the reward of minority class is 1
and the reward of majority class is _λ_ . In above experiments,
we let _λ_ = _ρ_ . To study the effect of different values of
_λ_ on the classification performance, we test values of _λ ∈_
_{_ 0 _._ 05 _ρ,_ 0 _._ 1 _ρ,_ 0 _._ 5 _ρ, ρ,_ 5 _ρ,_ 10 _ρ,_ 20 _ρ}_ . The experimental results
are shown in Fig.3.
In the same dataset of distinct imbalanced degree, the model
performs best when the reward of majority class _λ_ is equal to
the imbalanced ratio _ρ_ . In different datasets with the same
imbalanced ratio, the closer the reward of majority class _λ_
is to _ρ_, the better the classification performance of model
is, that is, the different values of _λ_ can adjust the impact of
majority samples on classification performance. Increasing or
decreasing the value of _λ_ = _ρ_ will break the balance of the
second item and the third item in (15) and lead to a poor
classification performance.


V. C ONCLUSION

This paper introduces a novel model for imbalanced classification using a deep reinforcement learning. The model



(a) Exploration on the same dataset
of different imbalanced ratio


(b) Exploration on the different datasets
with the same imbalanced ratio


Fig. 3. Different rewards for majority class to find the optimal reward
function.


formulates the classification problem as a sequential decisionmaking process (ICMDP), in which the environment returns
a high reward for minority class sample but a low reward
for majority class sample, and the episode will be terminated
when the agent misclassifies the minority class sample. We
use deep Q learning algorithm to find the optimal classification
policy for ICMDP, and theoretically analyze the impact of the
specific reward function on the loss function of Q network
when training. The effect of the two types of samples on the
loss function can be balanced by reducing the reward value the
agent receives from the majority samples. Experiments showed
that our model’s classification performance in imbalanced data
sets is better than other imbalanced classification methods,
especially in text data sets and extremely imbalanced data sets.
In the future work, we will apply improved deep reinforcement
learning algorithms to our model, and explore the design of
reward function and the establishment of learning environment
for classification in imbalanced multi-class data sets.


R EFERENCES


[1] N. Japkowicz and S. Stephen, “The class imbalance problem: A systematic study,” _Intelligent data analysis_, vol. 6, no. 5, pp. 429–449, 2002.

[2] G. M. Weiss, “Mining with rarity: a unifying framework,” _ACM Sigkdd_
_Explorations Newsletter_, vol. 6, no. 1, pp. 7–19, 2004.

[3] H. He and E. A. Garcia, “Learning from imbalanced data,” _IEEE_
_Transactions on Knowledge & Data Engineering_, no. 9, pp. 1263–1284,
2008.

[4] G. Haixiang, L. Yijing, J. Shang, G. Mingyun, H. Yuanyue, and
G. Bing, “Learning from class-imbalanced data: Review of methods and
applications,” _Expert Systems with Applications_, vol. 73, pp. 220–239,
2017.

[5] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” _arXiv preprint arXiv:1312.5602_, 2013.

[6] S. Gu, E. Holly, T. Lillicrap, and S. Levine, “Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates,” in
_Robotics and Automation (ICRA), 2017 IEEE International Conference_
_on_ . IEEE, 2017, pp. 3389–3396.

[7] X. Zhao, L. Zhang, Z. Ding, D. Yin, Y. Zhao, and J. Tang, “Deep
reinforcement learning for list-wise recommendations,” _arXiv preprint_
_arXiv:1801.00209_, 2017.

[8] C. Drummond, R. C. Holte _et al._, “C4. 5, class imbalance, and cost
sensitivity: why under-sampling beats over-sampling,” in _Workshop on_
_learning from imbalanced datasets II_, vol. 11. Citeseer, 2003, pp. 1–8.

[9] H. Han, W.-Y. Wang, and B.-H. Mao, “Borderline-smote: a new oversampling method in imbalanced data sets learning,” in _International_
_Conference on Intelligent Computing_ . Springer, 2005, pp. 878–887.

[10] I. Mani and I. Zhang, “knn approach to unbalanced data distributions:
a case study involving information extraction,” in _Proceedings of work-_
_shop on learning from imbalanced datasets_, vol. 126, 2003.

[11] G. E. Batista, R. C. Prati, and M. C. Monard, “A study of the behavior
of several methods for balancing machine learning training data,” _ACM_
_SIGKDD explorations newsletter_, vol. 6, no. 1, pp. 20–29, 2004.

[12] K. Veropoulos, C. Campbell, N. Cristianini _et al._, “Controlling the sensitivity of support vector machines,” in _Proceedings of the international_
_joint conference on AI_, vol. 55, 1999, p. 60.

[13] G. Wu and E. Y. Chang, “Kba: Kernel boundary alignment considering
imbalanced data distribution,” _IEEE Transactions on knowledge and data_
_engineering_, vol. 17, no. 6, pp. 786–795, 2005.

[14] Y. Tang, Y.-Q. Zhang, N. V. Chawla, and S. Krasser, “Svms modeling
for highly imbalanced classification,” _IEEE Transactions on Systems,_
_Man, and Cybernetics, Part B (Cybernetics)_, vol. 39, no. 1, pp. 281–
288, 2009.

[15] B. Zadrozny and C. Elkan, “Learning and making decisions when costs
and probabilities are both unknown,” in _Proceedings of the seventh ACM_
_SIGKDD international conference on Knowledge discovery and data_
_mining_ . ACM, 2001, pp. 204–213.




[16] B. Zadrozny, J. Langford, and N. Abe, “Cost-sensitive learning by costproportionate example weighting,” in _Data Mining, 2003. ICDM 2003._
_Third IEEE International Conference on_ . IEEE, 2003, pp. 435–442.

[17] Z.-H. Zhou and X.-Y. Liu, “Training cost-sensitive neural networks with
methods addressing the class imbalance problem,” _IEEE Transactions on_
_Knowledge and Data Engineering_, vol. 18, no. 1, pp. 63–77, 2006.

[18] B. Krawczyk and M. Wo´zniak, “Cost-sensitive neural network with rocbased moving threshold for imbalanced classification,” in _International_
_Conference on Intelligent Data Engineering and Automated Learning_ .
Springer, 2015, pp. 45–52.

[19] J. Chen, C.-A. Tsai, H. Moon, H. Ahn, J. Young, and C.-H. Chen,
“Decision threshold adjustment in class prediction,” _SAR and QSAR in_
_Environmental Research_, vol. 17, no. 3, pp. 337–352, 2006.

[20] H. Yu, C. Sun, X. Yang, W. Yang, J. Shen, and Y. Qi, “Odoc-elm:
Optimal decision outputs compensation-based extreme learning machine
for classifying imbalanced data,” _Knowledge-Based Systems_, vol. 92, pp.
55–70, 2016.

[21] K. M. Ting, “A comparative study of cost-sensitive boosting algorithms,”
in _In Proceedings of the 17th International Conference on Machine_
_Learning_ . Citeseer, 2000.

[22] S. Wang, W. Liu, J. Wu, L. Cao, Q. Meng, and P. J. Kennedy, “Training
deep neural networks on imbalanced data sets,” in _Neural Networks_
_(IJCNN), 2016 International Joint Conference on_ . IEEE, 2016, pp.
4368–4374.

[23] C. Huang, Y. Li, C. Change Loy, and X. Tang, “Learning deep representation for imbalanced classification,” in _Proceedings of the IEEE_
_Conference on Computer Vision and Pattern Recognition_, 2016, pp.
5375–5384.

[24] Y. Yan, M. Chen, M.-L. Shyu, and S.-C. Chen, “Deep learning for
imbalanced multimedia data classification,” in _Multimedia (ISM), 2015_
_IEEE International Symposium on_ . IEEE, 2015, pp. 483–488.

[25] S. H. Khan, M. Hayat, M. Bennamoun, F. A. Sohel, and R. Togneri,
“Cost-sensitive learning of deep feature representations from imbalanced
data,” _IEEE transactions on neural networks and learning systems_,
vol. 29, no. 8, pp. 3573–3587, 2018.

[26] Q. Dong, S. Gong, and X. Zhu, “Imbalanced deep learning by minority
class incremental rectification,” _IEEE Transactions on Pattern Analysis_
_and Machine Intelligence_, 2018.

[27] M. A. Wiering, H. van Hasselt, A.-D. Pietersma, and L. Schomaker, “Reinforcement learning algorithms for solving classification problems,” in
_Adaptive Dynamic Programming And Reinforcement Learning (ADPRL),_
_2011 IEEE Symposium on_ . IEEE, 2011, pp. 91–96.

[28] J. Feng, M. Huang, L. Zhao, Y. Yang, and X. Zhu, “Reinforcement
learning for relation classification from noisy data,” in _Proceedings of_
_AAAI_, 2018.

[29] T. Zhang, M. Huang, and L. Zhao, “Learning structured representation
for text classification via reinforcement learning.” AAAI, 2018.

[30] D. Liu and T. Jiang, “Deep reinforcement learning for surgical gesture segmentation and classification,” _arXiv preprint arXiv:1806.08089_,
2018.

[31] D. Zhao, Y. Chen, and L. Lv, “Deep reinforcement learning with visual
attention for vehicle classification,” _IEEE Transactions on Cognitive and_
_Developmental Systems_, vol. 9, no. 4, pp. 356–367, 2017.

[32] J. Janisch, T. Pevn`y, and V. Lis`y, “Classification with costly features
using deep reinforcement learning,” _arXiv preprint arXiv:1711.07364_,
2017.

[33] C. Martinez, G. Perrin, E. Ramasso, and M. Rombaut, “A deep reinforcement learning approach for early classification of time series,” in
_EUSIPCO 2018_, 2018.

[34] L. Abdi and S. Hashemi, “An ensemble pruning approach based on
reinforcement learning in presence of multi-class imbalanced data,” in
_Proceedings of the Third International Conference on Soft Computing_
_for Problem Solving_ . Springer, 2014, pp. 589–600.

[35] A. K. Dixit, J. J. Sherrerd _et al._, _Optimization in economic theory_ .
Oxford University Press on Demand, 1990.

[36] Q. Gu, L. Zhu, and Z. Cai, “Evaluation measures of the classification
performance of imbalanced data sets,” in _International Symposium on_
_Intelligence Computation and Applications_ . Springer, 2009, pp. 461–
471.

[37] Y. Bengio, “Practical recommendations for gradient-based training of
deep architectures,” in _Neural networks: Tricks of the trade_ . Springer,
2012, pp. 437–478.


